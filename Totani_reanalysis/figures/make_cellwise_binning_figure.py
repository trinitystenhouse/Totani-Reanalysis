#!/usr/bin/env python3
r"""
Three-column illustration of the cellwise binning used by the Totani fits.

The cellwise fit does not fit pixels. It tiles the ROI into square cells
(default 10 deg, matching Totani 2025), sums counts within each cell, and fits
one Poisson likelihood per cell per energy bin. That aggregation step is the
single biggest structural difference from the pixelwise reanalysis, and it is
easier to show than to describe. The three panels are that story, left to right:

  (a) counts        the sky as measured, at one energy bin
  (b) binning       the same map with the cell grid drawn on it, masked pixels
                    knocked out -- i.e. exactly which pixels feed which cell
  (c) cell counts   the aggregated per-cell counts the likelihood actually sees

Panel (c) is deliberately the same colour scale family as (a) so the eye reads
it as "the same sky, coarser", which is the point being made.

Usage
-----
  python make_cellwise_binning_figure.py                       # paper style
  python make_cellwise_binning_figure.py --style conference    # dark slide
  python make_cellwise_binning_figure.py --energy-gev 21 --cell-deg 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
_REPO = _REANALYSIS.parent

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "shared"))
sys.path.insert(0, str(_REPO / "Totani_paper_check" / "mcmc"))

from totani_helpers.totani_io import (  # noqa: E402
    lonlat_grids, load_mask_any_shape, read_counts_and_ebounds,
)
from helpers.plot_style import (  # noqa: E402
    add_style_argument, apply_plot_style, current_savefig_kwargs,
    normalise_plot_style, scaled_figsize, style_sizing, theme_ink,
)

_DATA = _REPO / "fermi_data" / "totani"
_TEMPLATES = _DATA / "processed" / "templates"
_COUNTS = _DATA / "processed" / "counts_ccube_1000to1000000.fits"
_MASK = _TEMPLATES / "mask_extended_sources.fits"

FULL_W = 7.1          # revtex two-column width, inches
HEIGHT = 2.7


def _cell_edges(roi_deg: float, cell_deg: float) -> np.ndarray:
    """Cell boundaries in degrees, matching build_cell_id_map's tiling."""
    n = int(round((2.0 * roi_deg) / cell_deg))
    return -roi_deg + cell_deg * np.arange(n + 1)


def _aggregate_to_cells(plane, cell_id, n_cells):
    """Total counts per cell -> painted back onto the pixel grid.

    Painting the per-cell total back onto its pixels (rather than plotting a
    12x12 image) keeps panel (c) on the same sky axes as (a) and (b), so the
    three panels are directly comparable and the mask footprint stays visible.
    """
    flat_id = cell_id.reshape(-1)
    flat_val = np.nan_to_num(plane, nan=0.0).reshape(-1)
    good = flat_id >= 0
    totals = np.bincount(flat_id[good], weights=flat_val[good], minlength=n_cells)

    out = np.full(cell_id.shape, np.nan)
    painted = totals[flat_id[good]]
    out.reshape(-1)[np.flatnonzero(good)] = painted
    return out, totals


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path, default=_COUNTS)
    ap.add_argument("--ext-mask", type=Path, default=_MASK)
    ap.add_argument("--energy-gev", type=float, default=21.0,
                    help="Energy bin to display; the nearest bin centre is used.")
    ap.add_argument("--cell-deg", type=float, default=10.0)
    ap.add_argument("--roi-deg", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path,
                    default=_HERE / "cellwise_binning")
    add_style_argument(ap)
    args = ap.parse_args(argv)

    style = normalise_plot_style(args.style, default="paper")
    sizing = style_sizing(style)
    apply_plot_style(style, base_fontsize=sizing["fontsize"],
                     linewidth=sizing["linewidth"], n_colors=8,
                     cmap_name="plasma")

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from astropy.wcs import WCS

    from run_mcmc import build_cell_id_map

    counts, hdr, _emin, _emax, ectr_mev, _de = read_counts_and_ebounds(str(args.counts))
    k = int(np.argmin(np.abs(ectr_mev / 1000.0 - args.energy_gev)))
    e_gev = float(ectr_mev[k]) / 1000.0

    nE, ny, nx = counts.shape
    lon, lat = lonlat_grids(WCS(hdr).celestial, ny, nx)
    mask = load_mask_any_shape(str(args.ext_mask), counts.shape)[k]

    cell_id, n_cells = build_cell_id_map(
        lon=lon, lat=lat, roi_lon=args.roi_deg, roi_lat=args.roi_deg,
        cell_deg=args.cell_deg)

    plane = counts[k]
    # NB the mask is KEEP-true, not masked-true: build_fit_mask3d ANDs it into
    # the fit mask, so a True pixel is one the likelihood uses. Applied before
    # the cell sums, or panel (c) would show counts the fit never saw.
    keep = mask
    masked_plane = np.where(keep, plane, np.nan)
    cell_id_fit = np.where(keep, cell_id, -1)
    cell_painted, totals = _aggregate_to_cells(masked_plane, cell_id_fit, n_cells)

    in_roi = cell_id >= 0
    extent = [args.roi_deg, -args.roi_deg, -args.roi_deg, args.roi_deg]  # l decreases rightward

    def _clip(a):
        return np.where(in_roi, a, np.nan)

    pos = plane[in_roi & keep]
    pos = pos[np.isfinite(pos) & (pos > 0)]
    vmin, vmax = np.percentile(pos, [5, 99.5])
    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    cnorm = LogNorm(vmin=max(np.nanpercentile(totals[totals > 0], 2), 1e-3),
                    vmax=np.nanmax(totals))

    fig, axes = plt.subplots(
        1, 3, figsize=scaled_figsize(FULL_W, HEIGHT), constrained_layout=True)

    ink = theme_ink("black")
    panels = [
        (_clip(plane), norm, "(a) counts", False),
        (_clip(masked_plane), norm, f"(b) {args.cell_deg:g}$^\\circ$ cell grid", True),
        (_clip(cell_painted), cnorm, "(c) counts per cell", True),
    ]

    for ax, (img, nrm, title, grid) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", extent=extent, norm=nrm,
                       cmap="plasma", interpolation="nearest")
        if grid:
            edges = _cell_edges(args.roi_deg, args.cell_deg)
            for e in edges:
                ax.axvline(e, color=ink, lw=0.45, alpha=0.55)
                ax.axhline(e, color=ink, lw=0.45, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel(r"$\ell$ [deg]")
        ax.set_xlim(args.roi_deg, -args.roi_deg)
        ax.set_ylim(-args.roi_deg, args.roi_deg)
        ax.set_xticks([60, 30, 0, -30, -60])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.tick_params(direction="out")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    axes[0].set_ylabel(r"$b$ [deg]")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    n_used = int(np.sum(totals > 0))
    fig.suptitle(
        f"Cellwise binning at $E$ = {e_gev:.1f} GeV: "
        f"{nx}$\\times${ny} pixels $\\rightarrow$ {n_used} cells",
        y=1.04)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_kw = current_savefig_kwargs(bbox_inches="tight")
    stem = args.out_dir / "cellwise_binning"
    for ext in ("pdf", "png"):
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=220, **save_kw)
        print(f"  saved {p}")
    plt.close(fig)

    print(f"  E = {e_gev:.2f} GeV (bin k={k}), cell = {args.cell_deg:g} deg, "
          f"{n_used}/{n_cells} cells populated, "
          f"masked pixel fraction = {1.0 - np.mean(keep[in_roi]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
