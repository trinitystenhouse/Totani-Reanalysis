#!/usr/bin/env python3

import os
import sys

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_mapcube_primary,
    read_exposure,
    reproject_cube_to_target,
    resample_exposure_logE,
    write_cube,
)

from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def read_galprop_skymap(path):
    from astropy.io import fits

    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header.copy()

    if data.ndim == 4:
        if data.shape[0] != 1:
            raise RuntimeError(f"Unexpected leading axis in {path}: shape={data.shape}")
        cube = data[0]
    elif data.ndim == 3:
        cube = data
    else:
        raise RuntimeError(f"Expected 3D or 4D skymap in {path}; got shape {data.shape}")

    crval1 = float(hdr.get("CRVAL1", 0.25))
    cdelt1 = float(hdr.get("CDELT1", 0.5))
    crpix1 = float(hdr.get("CRPIX1", 1.0))
    crval2 = float(hdr.get("CRVAL2", -89.75))
    cdelt2 = float(hdr.get("CDELT2", 0.5))
    crpix2 = float(hdr.get("CRPIX2", 1.0))
    w = WCS(naxis=2)
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.crval = [crval1, crval2]
    w.wcs.cdelt = [cdelt1, cdelt2]
    w.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]

    loge0 = hdr.get("CRVAL3")
    dloge = hdr.get("CDELT3")
    if loge0 is None or dloge is None:
        raise RuntimeError(f"Could not infer energy grid from header of {path}")
    E_mev = 10.0 ** (float(loge0) + float(dloge) * np.arange(cube.shape[0], dtype=float))

    grid_info = {
        "lon0": crval1,
        "lat0": crval2,
        "dlon": cdelt1,
        "dlat": cdelt2,
        "x0": crpix1 - 1.0,
        "y0": crpix2 - 1.0,
    }
    return cube, w.celestial, E_mev, grid_info

def interp_energy_planes(cube, E_src, E_tgt):
    if E_src is None:
        raise RuntimeError("No energy axis available to interpolate.")

    return resample_exposure_logE(cube, E_src, E_tgt)


def _reproject_regular_lonlat_cube_to_target(*, src_cube, lon2d_tgt, lat2d_tgt, grid_info):
    src_cube = np.asarray(src_cube, dtype=float)
    lon_mod = np.mod(np.asarray(lon2d_tgt, dtype=float), 360.0)
    lat = np.asarray(lat2d_tgt, dtype=float)

    x = (lon_mod - float(grid_info["lon0"])) / float(grid_info["dlon"]) + float(grid_info["x0"])
    y = (lat - float(grid_info["lat0"])) / float(grid_info["dlat"]) + float(grid_info["y0"])

    coords = np.vstack([y.ravel(), x.ravel()])
    out = np.empty((src_cube.shape[0],) + lon2d_tgt.shape, dtype=float)

    for k in range(src_cube.shape[0]):
        out[k] = map_coordinates(
            src_cube[k],
            coords,
            order=1,
            mode="wrap",
            cval=np.nan,
            prefilter=False,
        ).reshape(lon2d_tgt.shape)

    y_valid = (y >= -0.5) & (y <= (src_cube.shape[1] - 0.5))
    out[:, ~y_valid] = np.nan
    return out

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--expcube",
        default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--ics",
        default=os.path.join(DATA_DIR, "templates", "ics_isotropic_skymap_54_0e7e0006.fits.gz"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    _, hdr_cnt, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)

    w_tgt = WCS(hdr_cnt).celestial
    ny, nx = hdr_cnt["NAXIS2"], hdr_cnt["NAXIS1"]
    lon2d_tgt, lat2d_tgt = lonlat_grids(w_tgt, ny, nx)

    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)

    expo, E_exp = read_exposure(args.expcube)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")
    if expo.shape[0] != nE:
        if E_exp is None:
            raise RuntimeError("Exposure has different #planes and no energy axis.")
        expo = interp_energy_planes(expo, E_exp, Ectr_mev)

    if str(args.ics).endswith(".fits.gz"):
        ics_cube, w_src, E_ics, grid_ics = read_galprop_skymap(args.ics)
    else:
        ics_cube, w_src, E_ics, _hdr_ics = read_mapcube_primary(args.ics)
        grid_ics = None
    if E_ics is None:
        if ics_cube.shape[0] != nE:
            raise RuntimeError("ICS cube has no energies and does not match counts energy axis.")
        ics_E = ics_cube
    else:
        if ics_cube.shape[0] != len(E_ics):
            raise RuntimeError("ICS cube energy axis length mismatch.")
        ics_E = interp_energy_planes(ics_cube, E_ics, Ectr_mev) if ics_cube.shape[0] != nE else ics_cube

    if grid_ics is not None:
        ics_on_grid = _reproject_regular_lonlat_cube_to_target(
            src_cube=ics_E, lon2d_tgt=lon2d_tgt, lat2d_tgt=lat2d_tgt, grid_info=grid_ics
        )
    else:
        ics_on_grid = reproject_cube_to_target(src_cube=ics_E, w_src=w_src, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx)

    # GALPROP 54 skymaps store E²·dN/dE [MeV·ph/cm²/s/sr], not dN/dE.
    # Divide by E² to recover differential intensity dN/dE [ph/cm²/s/sr/MeV].
    ics_dnde = ics_on_grid / (Ectr_mev[:, None, None] ** 2)

    # Expected counts per bin:
    # mu = (dN/dE) * exposure * Ω_pix * ΔE
    mu_ics = ics_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    out_dnde = os.path.join(args.outdir, "ics_dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_ics_counts.fits")

    write_cube(out_dnde, ics_dnde, hdr_cnt, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_mu, mu_ics, hdr_cnt, bunit="counts")

    print("✓ Wrote", out_dnde)
    print("✓ Wrote", out_mu)
    print("  mu_ics total:", float(np.nansum(mu_ics)))


if __name__ == "__main__":
    main()
