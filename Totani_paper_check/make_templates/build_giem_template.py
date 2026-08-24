#!/usr/bin/env python3
r"""
Build the Fermi-LAT Galactic Interstellar Emission Model (GIEM, gll_iem_v07)
diffuse template on the ROI grid, as the gas+ICS alternative to the GALPROP
templates (the LAT-standard-background cross-check Totani 2025 also performs).

Produces:
    templates/giem_dnde.fits        (intensity dN/dE, ph cm^-2 s^-1 sr^-1 MeV^-1)
    templates/mu_giem_counts.fits   (expected counts = dN/dE * expo * dOmega * dE)

Then the systematics variant `diffuse_giem` (write_global_systematics_configs.py)
uses label `giem` in place of `gas`+`ics`.

NOTE on units: gll_iem_v07.fits stores *differential intensity* dN/dE
[MeV^-1 cm^-2 s^-1 sr^-1] with the energy grid in an `ENERGIES` BINTABLE extension
(unlike GALPROP-54 skymaps, which store E^2 dN/dE on a log-linear CRVAL3/CDELT3
grid). This script handles both; default assumes the gll_iem (dN/dE) convention.

For the pixelwise PSF fit, additionally PSF-convolve mu_giem_counts.fits ->
mu_giem_psf_counts.fits using mcmc/psf_utils.convolve_template_file.
"""

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (  # noqa: E402
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    reproject_cube_to_target,
    resample_exposure_logE,
    write_cube,
)

from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def read_giem_cube(path):
    """Read gll_iem_v07-style diffuse model: (nE, ny, nx) cube, WCS, energies [MeV]."""
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header.copy()
        E_mev = None
        for ext in h:
            if ext.name.upper() == "ENERGIES":
                col = ext.data.names[0]
                E_mev = np.asarray(ext.data[col], dtype=float)
                break
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D GIEM cube in {path}; got {data.shape}")
    if E_mev is None:
        loge0, dloge = hdr.get("CRVAL3"), hdr.get("CDELT3")
        if loge0 is None or dloge is None:
            raise RuntimeError("No ENERGIES extension and no CRVAL3/CDELT3 in GIEM file.")
        E_mev = 10.0 ** (float(loge0) + float(dloge) * np.arange(data.shape[0]))
    return data, WCS(hdr).celestial, E_mev


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed",
                    "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expcube", default=os.path.join(DATA_DIR, "processed",
                    "expcube_1000to1000000.fits"))
    ap.add_argument("--giem", default=os.path.join(DATA_DIR, "templates", "gll_iem_v07.fits"),
                    help="Fermi GIEM mapcube (gll_iem_v07.fits).")
    ap.add_argument("--units", choices=["dnde", "e2dnde"], default="dnde",
                    help="gll_iem stores dN/dE (default); GALPROP-54 stores E^2 dN/dE.")
    ap.add_argument("--outdir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--binsz", type=float, default=0.125)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    _, hdr, _, _, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)
    w_tgt = WCS(hdr).celestial
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)

    expo, E_exp = read_exposure(args.expcube)
    if expo.shape[0] != nE and E_exp is not None:
        expo = resample_exposure_logE(expo, E_exp, Ectr_mev)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")

    cube, w_src, E_src = read_giem_cube(args.giem)
    if args.units == "e2dnde":
        cube = cube / (E_src[:, None, None] ** 2)
    # interpolate GIEM energy planes to the counts energy grid, then reproject
    cube_E = resample_exposure_logE(cube, E_src, Ectr_mev) if cube.shape[0] != nE else cube
    giem_dnde = reproject_cube_to_target(src_cube=cube_E, w_src=w_src, w_tgt=w_tgt,
                                         ny_tgt=ny, nx_tgt=nx)

    mu_giem = giem_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    out_dnde = os.path.join(args.outdir, "giem_dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_giem_counts.fits")
    write_cube(out_dnde, giem_dnde, hdr, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_mu, mu_giem, hdr, bunit="counts")
    print("✓ Wrote", out_dnde)
    print("✓ Wrote", out_mu, " total counts:", float(np.nansum(mu_giem)))
    print("Next: PSF-convolve for the pixelwise fit:")
    print("  python -c \"import sys; sys.path.insert(0,'../mcmc'); from psf_utils import "
          "convolve_template_file as c; "
          f"c('{out_mu}', '{args.counts}')\"   # -> mu_giem_psf_counts.fits")


if __name__ == "__main__":
    main()
