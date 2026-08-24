#!/usr/bin/env python3
"""
mask_point_sources.py
=====================
Build a BRIGHT POINT-SOURCE keep-mask for the GLOBAL (pixel-level) Fermi-LAT fit.

Why this exists
---------------
`mask_extended_sources.py` in this repo masks only *extended* sources (the
`ExtendedSources` HDU of gll_psc_v35: LMC/SMC, SNRs/PWNe, lobes, ...). Bright
*point* sources (Vela, Geminga, bright blazars) live in the main catalogue HDU and
are NOT masked — they are modelled with a single, globally-rescaled `mu_ps` template.

For Totani's 10 deg cellwise fit that is fine: each cell averages over the source and
its PSF wings. For the 0.125 deg global fit it is not: a bright point source is a
sharp PSF-convolved spike, subtracted with an approximate PSF and a single shared
normalisation, so its residual is localised and can leak into the free-sign halo.
Standard pixel-level analyses therefore additionally *mask* the brightest sources.

This script reads the main 4FGL point-source table, selects sources above a flux (or
significance) threshold, and masks a disc around each. It can AND the result with the
existing extended-source mask to produce a single combined keep-mask. The output has
the same (1=keep, 0=masked) convention as `mask_extended_sources.fits`, so it drops
straight into the global fit's `--ext-mask` argument. Nothing the Totani pipeline
uses is modified.

Mask radius
-----------
By default each masked source gets a disc of `--radius-deg` (default 1.0 deg). With
`--scale-radius-with-flux`, brighter sources get a larger disc up to `--max-radius-deg`
(default 2.0 deg), reflecting that bright wings extend further. Use `--energy-dependent`
to instead size the radius per energy plane from the PSF R68(E) (produces a 3D mask;
`load_mask_any_shape` in totani_io already accepts that).

CLI
---
    python mask_point_sources.py \
        --counts /path/processed/counts_ccube_1000to1000000.fits \
        --psc    /path/templates/gll_psc_v35.fit \
        --combine-with /path/templates/mask_extended_sources.fits \
        --out    /path/templates/mask_sources_combined.fits \
        --flux-threshold 3e-9 --scale-radius-with-flux

Importable API
--------------
    build_point_source_mask(counts_fits, psc_fits, ...)         -> (mask2d|mask3d, info)
    build_combined_mask(counts_fits, psc_fits, extended_mask, out_fits, ...) -> out_fits
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _as_str(x) -> str:
    return x.decode(errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)


def _read_counts_grid(counts_fits: str):
    """Return (hdr, wcs, ny, nx, binsz_deg, Ectr_mev)."""
    with fits.open(counts_fits) as h:
        hdr = h[0].header
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        Ectr_mev = None
        if "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin_mev = np.asarray(eb["E_MIN"], float) / 1000.0  # keV -> MeV (repo convention)
            Emax_mev = np.asarray(eb["E_MAX"], float) / 1000.0
            Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    wcs = WCS(hdr).celestial
    binsz = abs(float(hdr.get("CDELT2", hdr.get("CDELT1", 0.125)))) or 0.125
    return hdr, wcs, ny, nx, binsz, Ectr_mev


def _lat_r68_deg(E_mev, a=0.8, b=0.8, c=0.1):
    E_gev = np.asarray(E_mev, float) / 1000.0
    return np.sqrt((a * E_gev ** (-b)) ** 2 + c ** 2)


# ---------------------------------------------------------------------------
# Core mask builder
# ---------------------------------------------------------------------------

def build_point_source_mask(
    counts_fits: str,
    psc_fits: str,
    *,
    flux_threshold: float = 3e-9,        # Flux1000 [ph cm^-2 s^-1]; sources >= this are masked
    signif_threshold: Optional[float] = None,  # if set, use Signif_Avg >= this instead of flux
    radius_deg: float = 1.0,
    scale_radius_with_flux: bool = False,
    max_radius_deg: float = 2.0,
    gc_radius_deg: float = 90.0,         # only consider sources within this of the GC
    energy_dependent: bool = False,
    psf_radius_factor: float = 2.0,      # radius = factor * R68(E) when energy_dependent
    hdu_index: int = 1,
    verbose: bool = True,
):
    """
    Build a bright-point-source keep-mask (True=keep, False=masked).

    Returns
    -------
    mask : (ny, nx) bool  (or (nE, ny, nx) bool if energy_dependent)
    info : dict  with 'n_selected', 'masked_fraction', and the selection used.
    """
    hdr, wcs, ny, nx, binsz, Ectr_mev = _read_counts_grid(counts_fits)

    with fits.open(psc_fits) as f:
        cat = f[hdu_index].data
    colnames = {c.lower() for c in cat.columns.names}

    glon = np.asarray(cat["GLON"], float)
    glat = np.asarray(cat["GLAT"], float)

    # --- selection ---
    if signif_threshold is not None and "signif_avg" in colnames:
        metric = np.asarray(cat["Signif_Avg"], float)
        sel = np.isfinite(metric) & (metric >= float(signif_threshold))
        sel_desc = f"Signif_Avg >= {signif_threshold}"
        bright_metric = metric
    else:
        if "flux1000" not in colnames:
            raise KeyError("Catalogue has no Flux1000 column; pass --signif-threshold instead.")
        metric = np.asarray(cat["Flux1000"], float)
        sel = np.isfinite(metric) & (metric >= float(flux_threshold))
        sel_desc = f"Flux1000 >= {flux_threshold:g}"
        bright_metric = metric

    # restrict to a cone around the GC (matches the PS template's <=90 deg)
    gc = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")
    src_all = SkyCoord(glon * u.deg, glat * u.deg, frame="galactic")
    sel &= src_all.separation(gc).deg <= float(gc_radius_deg)

    idx = np.flatnonzero(sel)
    if verbose:
        print(f"[PS-mask] {idx.size} sources selected ({sel_desc}, within {gc_radius_deg:g} deg of GC)")

    # --- per-source radius ---
    def _radius_for(metric_value: float) -> float:
        if not scale_radius_with_flux:
            return float(radius_deg)
        ref = float(signif_threshold) if signif_threshold is not None else float(flux_threshold)
        if ref <= 0 or not np.isfinite(metric_value) or metric_value <= 0:
            return float(radius_deg)
        # gentle log scaling: +0.5 deg per decade above threshold, capped
        r = radius_deg + 0.5 * np.log10(max(metric_value / ref, 1.0))
        return float(np.clip(r, radius_deg, max_radius_deg))

    # --- precompute lon/lat grid once ---
    yy, xx = np.mgrid[0:ny, 0:nx]
    lon_grid, lat_grid = wcs.pixel_to_world_values(xx, yy)
    grid = SkyCoord(lon_grid * u.deg, lat_grid * u.deg, frame="galactic")

    def _mask_one(keep2d, src_coord, r_deg):
        """Mask a disc using a local bounding box for speed."""
        x_c, y_c = wcs.world_to_pixel(src_coord)
        if not (np.isfinite(x_c) and np.isfinite(y_c)):
            return
        ix, iy = int(round(float(x_c))), int(round(float(y_c)))
        # bounding box half-size in pixels (cos(lat) widens it in x; pad generously)
        cb = max(np.cos(np.deg2rad(src_coord.b.deg)), 0.05)
        half_y = int(np.ceil((r_deg + binsz) / binsz))
        half_x = int(np.ceil((r_deg + binsz) / (binsz * cb)))
        y0, y1 = max(0, iy - half_y), min(ny, iy + half_y + 1)
        x0, x1 = max(0, ix - half_x), min(nx, ix + half_x + 1)
        if y0 >= y1 or x0 >= x1:
            return
        sep = grid[y0:y1, x0:x1].separation(src_coord).deg
        keep2d[y0:y1, x0:x1][sep < r_deg] = False

    if not energy_dependent:
        keep = np.ones((ny, nx), dtype=bool)
        for j in idx:
            _mask_one(keep, src_all[j], _radius_for(float(bright_metric[j])))
        masked_frac = 1.0 - float(keep.mean())
        if verbose:
            print(f"[PS-mask] 2D mask: masked fraction = {masked_frac:.4f}")
        info = {"n_selected": int(idx.size), "masked_fraction": masked_frac,
                "selection": sel_desc, "energy_dependent": False, "binsz_deg": binsz}
        return keep, info

    # energy-dependent 3D mask
    if Ectr_mev is None:
        raise RuntimeError("energy_dependent requested but counts file has no EBOUNDS.")
    nE = Ectr_mev.size
    keep3d = np.ones((nE, ny, nx), dtype=bool)
    r68 = _lat_r68_deg(Ectr_mev)
    for k in range(nE):
        r_k = max(float(psf_radius_factor * r68[k]), 0.3)
        plane = keep3d[k]
        for j in idx:
            _mask_one(plane, src_all[j], r_k)
    masked_frac = 1.0 - float(keep3d.mean())
    if verbose:
        print(f"[PS-mask] 3D (energy-dependent) mask: mean masked fraction = {masked_frac:.4f}")
    info = {"n_selected": int(idx.size), "masked_fraction": masked_frac,
            "selection": sel_desc, "energy_dependent": True, "binsz_deg": binsz}
    return keep3d, info


def build_combined_mask(
    counts_fits: str,
    psc_fits: str,
    extended_mask_fits: Optional[str],
    out_fits: str,
    *,
    overwrite: bool = True,
    **kwargs,
) -> str:
    """
    Build the bright-PS mask, AND it with the extended-source mask (if given),
    and write a combined keep-mask FITS (1=keep, 0=masked). Returns out_fits.
    """
    hdr, wcs, ny, nx, binsz, _ = _read_counts_grid(counts_fits)
    ps_mask, info = build_point_source_mask(counts_fits, psc_fits, **kwargs)

    combined = ps_mask
    if extended_mask_fits and os.path.exists(extended_mask_fits):
        ext = fits.getdata(extended_mask_fits).astype(bool)
        if ext.ndim == 2 and combined.ndim == 3:
            ext = np.broadcast_to(ext[None], combined.shape)
        elif ext.ndim == 3 and combined.ndim == 2:
            combined = np.broadcast_to(combined[None], ext.shape)
        if ext.shape != combined.shape:
            raise RuntimeError(f"Extended mask shape {ext.shape} != PS mask shape {combined.shape}")
        combined = combined & ext
        print(f"[PS-mask] combined with extended-source mask: {os.path.basename(extended_mask_fits)}")

    out_hdr = hdr.copy()
    out_hdr["COMMENT"] = "Combined source keep-mask (1=keep, 0=masked): bright PS + extended"
    out_hdr["PSMASK"] = (True, "Bright point sources masked")
    out_hdr["PSNSEL"] = (info["n_selected"], "Number of bright point sources masked")
    out_hdr["PSSEL"] = (info["selection"][:68], "Selection criterion")
    fits.PrimaryHDU(combined.astype(np.uint8), header=out_hdr).writeto(out_fits, overwrite=overwrite)
    print(f"[PS-mask] wrote {out_fits}  (final masked fraction = {1.0 - float(combined.mean()):.4f})")
    return out_fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a bright point-source keep-mask for the pixel-level global fit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--counts", required=True, help="Counts CCUBE (defines grid + WCS).")
    ap.add_argument("--psc", required=True, help="4FGL catalogue FITS (gll_psc_v35.fit).")
    ap.add_argument("--out", required=True, help="Output mask FITS (1=keep, 0=masked).")
    ap.add_argument("--combine-with", default=None,
                    help="Extended-source mask to AND with (e.g. mask_extended_sources.fits).")
    ap.add_argument("--flux-threshold", type=float, default=3e-9,
                    help="Mask sources with Flux1000 >= this [ph cm^-2 s^-1].")
    ap.add_argument("--signif-threshold", type=float, default=None,
                    help="If set, use Signif_Avg >= this instead of the flux cut.")
    ap.add_argument("--radius-deg", type=float, default=1.0, help="Base mask radius per source.")
    ap.add_argument("--scale-radius-with-flux", action="store_true", default=False,
                    help="Scale radius up with brightness (capped at --max-radius-deg).")
    ap.add_argument("--max-radius-deg", type=float, default=2.0)
    ap.add_argument("--gc-radius-deg", type=float, default=90.0)
    ap.add_argument("--energy-dependent", action="store_true", default=False,
                    help="Produce a 3D mask sized from the PSF R68(E) per energy plane.")
    ap.add_argument("--psf-radius-factor", type=float, default=2.0)
    ap.add_argument("--hdu-index", type=int, default=1, help="Catalogue HDU with the point-source table.")
    args = ap.parse_args()

    build_combined_mask(
        args.counts, args.psc, args.combine_with, args.out,
        flux_threshold=args.flux_threshold,
        signif_threshold=args.signif_threshold,
        radius_deg=args.radius_deg,
        scale_radius_with_flux=args.scale_radius_with_flux,
        max_radius_deg=args.max_radius_deg,
        gc_radius_deg=args.gc_radius_deg,
        energy_dependent=args.energy_dependent,
        psf_radius_factor=args.psf_radius_factor,
        hdu_index=args.hdu_index,
    )


if __name__ == "__main__":
    main()