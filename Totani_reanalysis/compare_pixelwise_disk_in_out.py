#!/usr/bin/env python3
r"""
DIAGNOSTIC FIGURE (not for the paper).

Pixelwise MCMC halo pole-flux spectrum, disk-included vs disk-excluded, for all
three NFW morphologies (rho^2.5, rho^2, rho^1) -- one panel per profile.

Reuses the loaders/helpers/style of make_paper_results_figures.py so the curves
are identical to the paper's Fig. 1, just restricted to the pixelwise fit and
contrasting the two ROI conventions.

Usage (from Totani_reanalysis/):
  python compare_pixelwise_disk_in_out.py
  python compare_pixelwise_disk_in_out.py --copy-to <somewhere>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: F401  (kept for parity / quick edits)

import make_paper_results_figures as M


def main(argv=None):
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # NOTE: diagnostics dir, deliberately NOT paper_figures
    ap.add_argument("--out-dir", type=Path, default=M.PC / "figures" / "diagnostics")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    M.apply_plot_style("paper", base_fontsize=10, linewidth=1.4, n_colors=8, cmap_name="plasma")
    colors = M.get_cmap_colors(cmap_name="plasma", n=5, start=0.15, end=0.88)

    fig, axes = plt.subplots(3, 1, figsize=(5.55, 8.4), sharex=True)
    for ax, panel in zip(axes, M.FIG8_PANELS):
        fn = int(panel["fig"])
        he = M.load_halo_spectrum(M.pixelwise_dir(fn, False),
                                  nfw_label=panel["label"], central_stat="f_p50")
        hi = M.load_halo_spectrum(M.pixelwise_dir(fn, True),
                                  nfw_label=panel["label"], central_stat="f_p50")
        ax.axhline(0.0, color="0.5", lw=0.7, ls=":")
        ax.axvline(20.0, color="0.8", lw=0.8, ls=":")
        M._plot_halo_points(ax, he, color=colors[1], marker="s",
                            label="Pixelwise, disk excluded", mfc="white",
                            x_shift=0.97, zorder=6)
        M._plot_halo_points(ax, hi, color=colors[3], marker="^",
                            label="Pixelwise, disk included", mfc=colors[3],
                            x_shift=1.03, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(1.0, 1000.0)
        ax.set_ylim(*panel["ylim"])
        ax.set_yticks(panel["yticks"])
        ax.text(0.04, 0.84, panel["title"], transform=ax.transAxes, fontsize=10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(alpha=0.13, which="both")

    axes[0].legend(fontsize=8.5, frameon=True, framealpha=0.88, loc="upper right")
    axes[-1].set_xlabel("photon energy [GeV]")
    fig.text(0.035, 0.5,
             r"flux $E^2 dN/dE$ at Galactic poles" "\n"
             r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
             rotation=90, va="center", fontsize=10.5)
    fig.tight_layout(rect=(0.09, 0.04, 1.0, 1.0), pad=0.5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = args.out_dir / f"f_pixelwise_disk_in_out.{ext}"
        fig.savefig(p, dpi=220, **M.current_savefig_kwargs(bbox_inches="tight"))
        paths.append(p)
    plt.close(fig)
    print("saved", paths[0])
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[0], args.copy_to / paths[0].name)
        print("copied ->", args.copy_to / paths[0].name)


if __name__ == "__main__":
    main()
