#!/usr/bin/env python3
r"""
Totani figures 4, 5, 6 and 7 as one multi-column panel: the halo comparison.

Figs 4/5/6/7 are the same cellwise component fit repeated under four halo
assumptions -- no halo, then NFW rho^2.5, rho^2 and rho^1. Shown one per slide
they are hard to compare, because the interesting difference is in one component
(nfw) while the other seven are near-identical by construction. Side by side on
a shared y-axis, the comparison is immediate. The no-halo panel leads, so the
reader sees the baseline the halo is being added to.

The summed 'total' curve is not drawn: it is just the sum of the components,
sits a decade above them, and stretches the shared log axis they live in.

The curves are READ BACK from the `mcmc_figN_f_ml.txt` tables that
make_figures_from_config.py writes next to each standalone figure, rather than
recomputed here. That is deliberate: this panel can then never disagree with the
single-panel versions, and no fit logic is duplicated. Run the cellwise family
first (any style -- the tables are style-independent):

    bash plot_all_figures.sh --mode cellwise

Usage
-----
  python make_fig567_three_column.py                      # paper style
  python make_fig567_three_column.py --style conference   # dark slide
  python make_fig567_three_column.py --stat f_p50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
_REPO = _REANALYSIS.parent

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "shared"))

from totani_helpers.plotting import totani_component_style  # noqa: E402
from helpers.plot_style import (  # noqa: E402
    add_style_argument, apply_plot_style, current_savefig_kwargs,
    get_cmap_colors, normalise_plot_style, scaled_figsize, style_sizing,
    theme_ink,
)

# fig -> the halo assumption that fig was fit under. Totani's numbering, kept in
# the panel labels so a reader can map back to the single-panel figures. fig4 is
# the no-halo baseline and leads, so the eye reads left-to-right as "no halo,
# then progressively shallower halo profiles".
PANELS = [
    ("fig4", "(a) no halo"),
    ("fig5", r"(b) NFW $\rho^{2.5}$"),
    ("fig6", r"(c) NFW $\rho^{2}$"),
    ("fig7", r"(d) NFW $\rho^{1}$"),
]

# Width scales with the panel count so each panel keeps the aspect it had at
# three columns; the style's own canvas scale is applied on top by _figsize.
PANEL_W = 2.4
HEIGHT = 3.0

# Draw + legend order. 'total' is deliberately absent: it is the sum of
# everything else, sits a decade above the components, and on a shared log axis
# it just compresses the range the components live in.
PREFERRED = ["gas", "ics", "ps", "iso", "loopI", "fb_pos", "fb_neg", "nfw"]
EXCLUDE = {"total"}

# Conference palette: sample the active style's plasma ramp rather than reusing
# the print scheme's named colours (green/brown/royalblue/black). The dark
# styles lift these samples off a brightness floor, so every component stays
# legible on a slide. fb_pos and fb_neg intentionally share a colour -- they are
# one component split by sign, and the up/down markers carry the distinction,
# exactly as in the print scheme.
def _palette():
    hues = get_cmap_colors(cmap_name="plasma", n=6, start=0.05, end=0.95)
    fb = hues[1]
    return {
        "gas": hues[0], "ics": hues[2], "ps": hues[3], "iso": hues[4],
        "loopI": hues[5], "fb_pos": fb, "fb_neg": fb,
        # The halo is the subject of the figure and the only curve that changes
        # across panels, so it takes the theme ink -- the one value guaranteed
        # to contrast with a plasma ramp in both themes -- not a ramp sample.
        "nfw": theme_ink("black"),
    }


def _read_curves(path: Path):
    """(E_GeV, {label: (y, yerr[2,N])}) from a saved mcmc_figN_<stat>.txt."""
    with path.open() as f:
        header = f.readline().lstrip("#").split()
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    cols = {name: arr[:, i] for i, name in enumerate(header)}

    E = cols["E_GeV"]
    out = {}
    for name in header:
        if name == "E_GeV" or name.endswith("_err_lo") or name.endswith("_err_hi"):
            continue
        lo, hi = cols.get(f"{name}_err_lo"), cols.get(f"{name}_err_hi")
        yerr = np.vstack([lo, hi]) if lo is not None and hi is not None else None
        out[name] = (cols[name], yerr)
    return E, out


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots-root", type=Path,
                    default=_REPO / "Totani_paper_check" / "figures",
                    help="Directory holding plots_fig4/5/6/7 (the cellwise outputs).")
    ap.add_argument("--stat", default="f_ml", choices=["f_ml", "f_p50", "f_p16", "f_p84"])
    ap.add_argument("--out-dir", type=Path, default=_HERE / "fig4567_columns")
    add_style_argument(ap)
    args = ap.parse_args(argv)

    style = normalise_plot_style(args.style, default="paper")
    sizing = style_sizing(style)
    apply_plot_style(style, base_fontsize=sizing["fontsize"],
                     linewidth=sizing["linewidth"], n_colors=12, cmap_name="plasma")

    import matplotlib.pyplot as plt

    # After apply_plot_style, so the samples pick up the style's brightness floor.
    palette = _palette()

    tables = []
    for fig_name, _label in PANELS:
        p = args.plots_root / f"plots_{fig_name}" / f"mcmc_{fig_name}_{args.stat}.txt"
        if not p.exists():
            raise SystemExit(
                f"missing input table: {p}\n"
                "Run the cellwise family first:\n"
                "  bash Totani_reanalysis/figures/plot_all_figures.sh --mode cellwise"
            )
        tables.append(_read_curves(p))

    fig, axes = plt.subplots(1, len(PANELS),
                             figsize=scaled_figsize(PANEL_W * len(PANELS), HEIGHT),
                             sharex=True, sharey=True)

    # One shared y-range across all panels -- the whole point is that the
    # panels are comparable, which a per-panel autoscale would quietly destroy.
    finite = np.concatenate([
        np.abs(y)[np.isfinite(y) & (np.abs(y) > 0)]
        for _E, curves in tables
        for k, (y, _e) in curves.items() if k not in EXCLUDE
    ])
    ymin, ymax = np.percentile(finite, 0.5), finite.max()

    handles, labels = [], []
    for ax, (fig_name, panel_label), (E, curves) in zip(axes, PANELS, tables):
        order = [k for k in PREFERRED if k in curves] + \
                [k for k in curves if k not in PREFERRED and k not in EXCLUDE]
        for name in order:
            y, yerr = curves[name]
            st = totani_component_style(name)
            m = np.isfinite(E) & np.isfinite(y) & (E > 0) & (y > 0)
            if not np.any(m):
                continue
            kw = dict(color=palette.get(name, st.get("color")), linestyle=st.get("linestyle", ":"),
                      marker=st.get("marker", "o"), linewidth=st.get("linewidth", 1.6),
                      markersize=st.get("markersize", 6) * 0.8)
            if yerr is not None:
                h = ax.errorbar(E[m], y[m], yerr=yerr[:, m], capsize=2, **kw)
            else:
                h, = ax.plot(E[m], y[m], **kw)
            if name not in labels:
                handles.append(h)
                labels.append(name)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel_label)
        ax.set_xlabel("Energy (GeV)")
        ax.grid(alpha=0.14, which="both")

    axes[0].set_ylim(ymin, ymax * 1.6)
    axes[0].set_ylabel(
        r"$E^2\,\langle \mathrm{d}N/\mathrm{d}E \rangle$"
        "\n"
        r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    # Single figure-level legend: nine components repeated three times would eat
    # the panels, and every panel carries the same component set by construction.
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                     frameon=False, handlelength=1.6, columnspacing=1.0,
                     handletextpad=0.4, borderaxespad=0.0,
                     fontsize=max(7, sizing["fontsize"] - 3))
    # Lay the axes out first, then hang the legend directly under whatever the
    # axes actually occupy (x-labels included). Reserving a fixed fraction, or
    # even a legend-height fraction, leaves a dead band whose size depends on
    # the style: tight_layout only pads the axes down to the rect, it does not
    # pull the legend up to meet them. Measuring the drawn result is the only
    # way to get the same snug spacing in print and on a slide.
    fig.tight_layout()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_h = fig.get_figheight() * fig.dpi
    axes_bottom = min(ax.get_tightbbox(rend).y0 for ax in axes) / fig_h
    leg_h = leg.get_window_extent(rend).height / fig_h
    gap = 0.025
    leg.set_bbox_to_anchor((0.5, axes_bottom - leg_h - gap),
                           transform=fig.transFigure)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_kw = current_savefig_kwargs(bbox_inches="tight")
    stem = args.out_dir / "fig567_halo_profiles"
    for ext in ("pdf", "png"):
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=220, **save_kw)
        print(f"  saved {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
