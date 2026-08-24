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
    """
    Read the newer GALPROP skymap products, which are stored as a 4D primary
    array with shape (1, nE, ny, nx) and encode the energy grid in the header.
    """
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


def _mapcube_to_counts_energy_grid(cube, E_cube, E_tgt):
    if E_cube is None:
        if cube.shape[0] != len(E_tgt):
            raise RuntimeError("Mapcube has no energies and does not match counts energy axis.")
        return cube

    if cube.shape[0] != len(E_cube):
        raise RuntimeError("Mapcube energy axis length mismatch.")

    return interp_energy_planes(cube, E_cube, E_tgt) if cube.shape[0] != len(E_tgt) else cube


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
        "--pion",
        default=os.path.join(DATA_DIR, "templates", "pion_decay_skymap_54_0e7e0006.fits.gz"),
    )
    ap.add_argument(
        "--bremss",
        default=os.path.join(DATA_DIR, "templates", "bremss_skymap_54_0e7e0006.fits.gz"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    _, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)

    w_tgt = WCS(hdr).celestial
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    lon2d_tgt, lat2d_tgt = lonlat_grids(w_tgt, ny, nx)

    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)

    expo, E_exp = read_exposure(args.expcube)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")
    if expo.shape[0] != nE:
        if E_exp is None:
            raise RuntimeError("Exposure has different #planes and no energy axis.")
        expo = interp_energy_planes(expo, E_exp, Ectr_mev)

    if str(args.pion).endswith(".fits.gz"):
        pion_cube, w_pion, E_pion, grid_pion = read_galprop_skymap(args.pion)
    else:
        pion_cube, w_pion, E_pion, _hdr_pion = read_mapcube_primary(args.pion)
        grid_pion = None
    if str(args.bremss).endswith(".fits.gz"):
        brem_cube, w_brem, E_brem, grid_brem = read_galprop_skymap(args.bremss)
    else:
        brem_cube, w_brem, E_brem, _hdr_brem = read_mapcube_primary(args.bremss)
        grid_brem = None

    pion_E = _mapcube_to_counts_energy_grid(pion_cube, E_pion, Ectr_mev)
    brem_E = _mapcube_to_counts_energy_grid(brem_cube, E_brem, Ectr_mev)

    if grid_pion is not None:
        pion_on_grid = _reproject_regular_lonlat_cube_to_target(
            src_cube=pion_E, lon2d_tgt=lon2d_tgt, lat2d_tgt=lat2d_tgt, grid_info=grid_pion
        )
    else:
        pion_on_grid = reproject_cube_to_target(src_cube=pion_E, w_src=w_pion, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx)
    if grid_brem is not None:
        brem_on_grid = _reproject_regular_lonlat_cube_to_target(
            src_cube=brem_E, lon2d_tgt=lon2d_tgt, lat2d_tgt=lat2d_tgt, grid_info=grid_brem
        )
    else:
        brem_on_grid = reproject_cube_to_target(src_cube=brem_E, w_src=w_brem, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx)

    # GALPROP 54 skymaps store E²·dN/dE [MeV·ph/cm²/s/sr], not dN/dE.
    # Divide by E² to recover differential intensity dN/dE [ph/cm²/s/sr/MeV].
    gas_dnde = (pion_on_grid + brem_on_grid) / (Ectr_mev[:, None, None] ** 2)

    mu_gas = gas_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    out_dnde = os.path.join(args.outdir, "gas_dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_gas_counts.fits")

    write_cube(out_dnde, gas_dnde, hdr, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_mu, mu_gas, hdr, bunit="counts")

    print("✓ Wrote", out_dnde)
    print("✓ Wrote", out_mu)
    print("  mu_gas total:", float(np.nansum(mu_gas)))


if __name__ == "__main__":
    main()
