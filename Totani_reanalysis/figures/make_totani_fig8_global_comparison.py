#!/usr/bin/env python3
"""
Paper-ready stacked Totani Fig. 8 comparison.

This overlays the halo spectrum at the Galactic poles for all three Totani
Fig. 8 profiles:

  - cellwise fit, disk excluded
  - global fit, disk excluded
  - global fit, disk included, where available

The flux convention is the same as make_totani_fig8.py:
E^2 dN/dE at the Galactic pole = f_NFW * iso_target_e2.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06
_MCMC = _PROJECT / "mcmc"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))
sys.path.insert(0, str(_REPO / "Totani_reanalysis" / "core"))

from helpers.plot_style import current_savefig_kwargs, get_cmap_colors, set_paper_style  # noqa: E402
from core.totani_data_loader import load_halo_spectrum  # noqa: E402


PANELS = [
    {
        "fig": 5,
        "title": r"NFW-$\rho^{2.5}$ (GCE)",
        "label": "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        "ylim": (-1.0e-5, 2.0e-5),
        "yticks": np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * 1e-5,
        "cellwise_dirs": ("mcmc_results_fig5_10deg", "mcmc_results_fig5"),
        "global_dir": "global_fit_results_fig5",
        "global_disk_dir": "global_fit_results_fig5_w_disk",
    },
    {
        "fig": 6,
        "title": r"NFW-$\rho^2$",
        "label": "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
        "ylim": (-2.0e-5, 4.25e-5),
        "yticks": np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1e-5,
        "cellwise_dirs": ("mcmc_results_fig6_10deg", "mcmc_results_fig6"),
        "global_dir": "global_fit_results_fig6",
        "global_disk_dir": "global_fit_results_fig6_w_disk",
    },
    {
        "fig": 7,
        "title": r"NFW-$\rho^1$",
        "label": "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno",
        "ylim": (-1.0e-4, 2.0e-4),
        "yticks": np.array([-1.0, 0.0, 1.0, 2.0]) * 1e-4,
        "cellwise_dirs": ("mcmc_results_fig7_10deg", "mcmc_results_fig7"),
        "global_dir": "global_fit_results_fig7",
        "global_disk_dir": "global_fit_results_fig7_w_disk",
    },
]


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return Path(paths[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay stacked Totani Fig. 8 cellwise/global disk variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir", type=Path, default=_HERE / "plots_fig8_global_comparison")
    parser.add_argument("--central-stat", default="f_p50", choices=["f_p50", "f_ml"])
    parser.add_argument(
        "--global-tag",
        default="global",
        choices=["global", "pixelwise", "pixelwise_mcmc"],
        help="Use L-BFGS-B global results or pixelwise MCMC global-validation results.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--auto-ylim", action="store_true", help="Use data-driven y limits instead of Totani Fig. 8 limits.")
    return parser.parse_args()


def _load_series(path: Path, *, nfw_label: str, central_stat: str):
    return load_halo_spectrum(
        path,
        nfw_label=nfw_label,
        central_stat=central_stat,
    )


def _plot_series(
    ax,
    halo,
    *,
    color,
    marker,
    label: str,
    markerfacecolor,
    x_shift: float = 1.0,
) -> None:
    x = np.asarray(halo.E_bins_GeV, float) * float(x_shift)
    y = np.asarray(halo.phi, float)
    yerr = np.vstack([halo.phi_err_lo, halo.phi_err_hi])
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(finite):
        return
    ax.errorbar(
        x[finite],
        y[finite],
        yerr=yerr[:, finite],
        linestyle="none",
        color=color,
        ecolor=color,
        marker=marker,
        mfc=markerfacecolor,
        mec=color,
        mew=0.75,
        markersize=5.2,
        elinewidth=0.75,
        capsize=2.0,
        label=label,
        zorder=5,
    )


def _auto_ylim(halos) -> tuple[float, float]:
    vals = []
    for halo in halos:
        y = np.asarray(halo.phi, float)
        err_lo = np.asarray(halo.phi_err_lo, float)
        err_hi = np.asarray(halo.phi_err_hi, float)
        finite = np.isfinite(y)
        if np.any(finite):
            vals.append((y - err_lo)[finite])
            vals.append((y + err_hi)[finite])
    if not vals:
        return -2.0e-5, 4.0e-5
    all_vals = np.concatenate(vals)
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    pad = 0.12 * max(ymax - ymin, 1e-12)
    return min(ymin - pad, -0.25e-5), max(ymax + pad, 3.8e-5)


def _panel_paths(panel: dict, *, global_tag: str) -> tuple[Path, Path, Path | None]:
    cellwise = _first_existing(*[_MCMC / name for name in panel["cellwise_dirs"]])
    if str(global_tag).strip().lower() in {"pixelwise", "pixelwise_mcmc"}:
        global_no_disk = _MCMC / f"pixelwise_mcmc_results_fig{panel['fig']}"
        global_with_disk = _MCMC / f"pixelwise_mcmc_results_fig{panel['fig']}_w_disk"
    else:
        global_no_disk = _MCMC / str(panel["global_dir"])
        global_disk_name = panel.get("global_disk_dir")
        global_with_disk = _MCMC / str(global_disk_name) if global_disk_name else None
    if global_with_disk is not None and not global_with_disk.exists():
        global_with_disk = None
    return cellwise, global_no_disk, global_with_disk


def main() -> None:
    args = parse_args()
    set_paper_style(style="paper", base_fontsize=10, linewidth=1.5, n_colors=10, cmap_name="plasma")
    args.outdir.mkdir(parents=True, exist_ok=True)

    colors = get_cmap_colors(cmap_name="plasma", n=5, start=0.15, end=0.88)
    fg = plt.rcParams.get("text.color", "black")

    fig, axes = plt.subplots(3, 1, figsize=(5.55, 8.4), sharex=True)
    input_lines = []
    input_lines.append(f"global_tag: {args.global_tag}")
    for ax, panel in zip(axes, PANELS):
        cellwise_path, global_path, global_disk_path = _panel_paths(panel, global_tag=args.global_tag)
        input_lines.append(f"Fig. {panel['fig']} cellwise disk excluded : {cellwise_path}")
        input_lines.append(f"Fig. {panel['fig']} global disk excluded   : {global_path}")
        if global_disk_path is not None:
            input_lines.append(f"Fig. {panel['fig']} global disk included   : {global_disk_path}")

        cellwise = _load_series(cellwise_path, nfw_label=panel["label"], central_stat=args.central_stat)
        global_no_disk = _load_series(global_path, nfw_label=panel["label"], central_stat=args.central_stat)
        global_with_disk = (
            _load_series(global_disk_path, nfw_label=panel["label"], central_stat=args.central_stat)
            if global_disk_path is not None
            else None
        )

        ax.axhline(0.0, color="0.35", lw=0.75, ls=":", zorder=1)
        _plot_series(
            ax,
            cellwise,
            color=fg,
            marker="o",
            label="Cellwise fit, disk excluded",
            markerfacecolor="white",
            x_shift=0.940,
        )
        _plot_series(
            ax,
            global_no_disk,
            color=colors[1],
            marker="s",
            label="Global fit, disk excluded",
            markerfacecolor=colors[1],
            x_shift=1.000,
        )
        if global_with_disk is not None:
            _plot_series(
                ax,
                global_with_disk,
                color=colors[3],
                marker="^",
                label="Global fit, disk included",
                markerfacecolor=colors[3],
                x_shift=1.064,
            )

        ax.set_xscale("log")
        ax.set_xlim(1.0, 1000.0)
        if args.auto_ylim:
            halos = [cellwise, global_no_disk] + ([global_with_disk] if global_with_disk is not None else [])
            ax.set_ylim(*_auto_ylim(halos))
        else:
            ax.set_ylim(*panel["ylim"])
            ax.set_yticks(panel["yticks"])
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.text(0.04, 0.84, panel["title"], transform=ax.transAxes, fontsize=10.0)

    axes[0].legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.88)
    axes[-1].set_xlabel("photon energy [GeV]")
    fig.text(
        0.035,
        0.5,
        r"flux $E^2 dN/dE$ at Galactic poles"
        "\n"
        r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        rotation=90,
        va="center",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.09, 0.04, 1.0, 1.0), pad=0.5)

    stem = "totani_fig8_stacked_global_cellwise_disk_comparison"
    for ext in args.formats:
        out = args.outdir / f"{stem}.{ext}"
        fig.savefig(out, dpi=args.dpi, **current_savefig_kwargs(bbox_inches="tight"))
        print(f"  saved {out}")
    plt.close(fig)

    print("Inputs:")
    for line in input_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
