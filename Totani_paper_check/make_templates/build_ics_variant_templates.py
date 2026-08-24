#!/usr/bin/env python3
r"""
Build ICS (inverse-Compton) spatial-morphology variant templates to bracket the
ISRF / CR-electron uncertainty that competes with a smooth high-latitude halo.

Why morphology, not normalisation
---------------------------------
The global fit is performed independently in each energy bin, so a pure spectral
re-tilt (scaling each energy plane by a constant) of the ICS template is exactly
degenerate with the free per-bin ICS coefficient and would have NO effect on the
recovered halo. The ICS systematic that the halo is actually sensitive to is the
LATITUDE morphology: a different ISRF vertical scale height changes how steeply
the ICS falls toward high |b|, which a single per-bin coefficient cannot absorb.

This script reweights the nominal ICS template by a latitude-dependent factor

    w(b) = ( max(|b|, b_floor) / b_piv ) ** alpha ,

then renormalises each energy plane to preserve the total ICS counts inside the
analysis ROI (|l|<=roi_lon, disk_cut<=|b|<=roi_lat). The result is a pure
morphology change at fixed per-bin flux:

    alpha > 0  ("flatb")  -> flatter latitude gradient, MORE ICS at high |b|
    alpha < 0  ("steepb") -> steeper latitude gradient, LESS ICS at high |b|

Both the unconvolved (mu_ics_*_counts.fits) and PSF-folded
(mu_ics_*_psf_counts.fits) templates are produced so the cellwise and pixelwise
fits can both use them.

Usage:
    python build_ics_variant_templates.py            # alpha=0.5, b_piv=30
    python build_ics_variant_templates.py --alpha 0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
TEMPLATES = REPO / "fermi_data" / "totani" / "processed" / "templates"


def _lat_grid(header) -> np.ndarray:
    """Galactic latitude [deg] for each pixel row, CAR linear mapping from WCS."""
    ny = int(header["NAXIS2"]); nx = int(header["NAXIS1"])
    crpix2 = float(header["CRPIX2"]); crval2 = float(header["CRVAL2"]); cdelt2 = float(header["CDELT2"])
    j = np.arange(ny)
    b = crval2 + (j + 1 - crpix2) * cdelt2          # 1-based FITS pixel convention
    return np.broadcast_to(b[:, None], (ny, nx))


def _lon_grid(header) -> np.ndarray:
    ny = int(header["NAXIS2"]); nx = int(header["NAXIS1"])
    crpix1 = float(header["CRPIX1"]); crval1 = float(header["CRVAL1"]); cdelt1 = float(header["CDELT1"])
    i = np.arange(nx)
    l = crval1 + (i + 1 - crpix1) * cdelt1
    l = (l + 180.0) % 360.0 - 180.0
    return np.broadcast_to(l[None, :], (ny, nx))


def _reweight(cube, w2d, roi):
    """Apply latitude weight w2d, renormalise each plane to preserve ROI total."""
    out = np.array(cube, dtype=float, copy=True)
    for k in range(cube.shape[0]):
        plane = cube[k]
        base_tot = float(np.nansum(plane[roi]))
        wp = plane * w2d
        new_tot = float(np.nansum(wp[roi]))
        scale = (base_tot / new_tot) if new_tot > 0 else 1.0
        out[k] = wp * scale
    return out


def build(alpha, b_piv, b_floor, roi_lon, roi_lat, disk_cut):
    made = []
    for src_name, suffix in [("mu_ics_counts.fits", ""), ("mu_ics_psf_counts.fits", "_psf")]:
        src = TEMPLATES / src_name
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        with fits.open(src) as hdul:
            cube = np.asarray(hdul[0].data, float)
            header = hdul[0].header.copy()
        lat = _lat_grid(header)
        lon = _lon_grid(header)
        roi = (np.abs(lon) <= roi_lon) & (np.abs(lat) >= disk_cut) & (np.abs(lat) <= roi_lat)
        babs = np.clip(np.abs(lat), b_floor, None)
        for tag, a in [("flatb", +abs(alpha)), ("steepb", -abs(alpha))]:
            w = (babs / b_piv) ** a
            out_cube = _reweight(cube, w, roi)
            h = header.copy()
            h["ICSVAR"] = tag
            h["ICSALPHA"] = float(a)
            h["ICSBPIV"] = float(b_piv)
            h["HISTORY"] = f"ICS latitude-morphology variant '{tag}' (alpha={a}, b_piv={b_piv})"
            out = TEMPLATES / f"mu_ics_{tag}{suffix}_counts.fits"
            fits.HDUList([fits.PrimaryHDU(out_cube.astype(np.float32), header=h)]).writeto(out, overwrite=True)
            made.append(out.name)
            print(f"  wrote {out.name}  (ROI-total preserved per plane)")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.5, help="latitude tilt exponent (|alpha|).")
    ap.add_argument("--b-piv", type=float, default=30.0, help="pivot latitude [deg].")
    ap.add_argument("--b-floor", type=float, default=5.0, help="floor on |b| to avoid div-by-zero [deg].")
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)
    args = ap.parse_args()
    made = build(args.alpha, args.b_piv, args.b_floor, args.roi_lon, args.roi_lat, args.disk_cut)
    print(f"\nWrote {len(made)} ICS variant templates to {TEMPLATES}")


if __name__ == "__main__":
    main()
