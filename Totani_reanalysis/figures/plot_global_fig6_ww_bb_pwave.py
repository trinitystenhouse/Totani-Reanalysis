#!/usr/bin/env python3
r"""
Make the global Fig. 6 PPPC comparison figures for the paper plotting style.

Outputs, by default:

  1. halo_spectrum_bestfit_global_fig6_WW_bb_residuals.png/pdf
     Global halo spectrum, best-fit PPPC WW and bb spectra, and residuals,
     with Hoof s-wave and Boddy p-wave dSph tension annotations.

  2. velocity_limits_overlay_global_fig6_XX.png/pdf
     Two-panel halo best-fit <sigma v>(m) with Hoof s-wave and Boddy p-wave
     dSph limits; WW on the left and bb on the right.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06

sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))
sys.path.insert(0, str(_REPO / "Totani_reanalysis" / "core"))

from helpers.plot_style import (  # noqa: E402
    apply_plot_style,
    current_savefig_kwargs,
    get_cmap_colors,
)
from plot_pppc_mass_scan import (  # noqa: E402
    DEFAULT_PROFILE_MAX_ABS_DLOGLIKE,
    DEFAULT_PROFILE_MAX_ABS_LOGLIKE,
    FIT_RESULT_SUBDIR,
    HALO_REFERENCE_CHOICES,
    RESULTS_BASE,
    apply_reference_spectral_profile,
    load_channel,
    _load_reference_halo_spectrum,
)
from core.dsph_limits_multi import dsph_upper_limit  # noqa: E402


CHANNELS = ("WW", "bb")
CHANNEL_LABELS = {"WW": r"$W^+W^-$", "bb": r"$b\bar b$"}
THERMAL_RELIC = 2.2e-26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit", default="global_fig6", choices=sorted(FIT_RESULT_SUBDIR))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=_HERE / "plots_pppc_global_fig6")
    parser.add_argument("--halo-reference", default="auto", choices=HALO_REFERENCE_CHOICES)
    parser.add_argument("--style", default="paper")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is not None:
        return args.results_dir
    result_subdir = FIT_RESULT_SUBDIR[str(args.fit)]
    return RESULTS_BASE if result_subdir is None else RESULTS_BASE / str(result_subdir)


def _load_channels(args: argparse.Namespace) -> dict[str, dict]:
    results_dir = _results_dir(args)
    data: dict[str, dict] = {}
    for channel in CHANNELS:
        loaded = load_channel(
            channel,
            results_dir,
            fit=args.fit,
            dsph_source="hoof",
            max_abs_loglike=DEFAULT_PROFILE_MAX_ABS_LOGLIKE,
            max_abs_dlnl=DEFAULT_PROFILE_MAX_ABS_DLOGLIKE,
        )
        if loaded is not None:
            data[channel] = loaded
    if not data:
        raise FileNotFoundError(f"No global Fig. 6 PPPC scan results found in {results_dir}")

    missing_spectral = [ch for ch, d in data.items() if "spectral_model_phi" not in d]
    if missing_spectral:
        data = apply_reference_spectral_profile(
            data,
            fit=args.fit,
            halo_reference=args.halo_reference,
            dsph_source="hoof",
        )
    return data


def _best_index(d: dict) -> int:
    if "spectral_chi2" in d and np.any(np.isfinite(d["spectral_chi2"])):
        return int(np.nanargmin(np.asarray(d["spectral_chi2"], float)))
    total_lnL = np.asarray(d["total_lnL"], float)
    if not np.any(np.isfinite(total_lnL)):
        raise ValueError("No finite likelihood/profile values in channel data.")
    return int(np.nanargmax(total_lnL))


def _tension_value(mass: float, sigmav: float, channel: str, source: str) -> float:
    limit = dsph_upper_limit(float(mass), channel, source)
    if not (np.isfinite(limit) and limit > 0 and np.isfinite(sigmav)):
        return np.nan
    return float(sigmav / limit)


def _format_r(value: float) -> str:
    return f"{value:.2g}" if np.isfinite(value) else "--"


def _save(fig, outdir: Path, stem: str, formats: list[str], dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, **current_savefig_kwargs(bbox_inches="tight"))
        print(f"  saved {path}")


def plot_halo_spectrum_residuals(
    channel_data: dict[str, dict],
    *,
    fit: str,
    halo_reference: str,
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    halo, halo_label, resolved = _load_reference_halo_spectrum(halo_reference, fit=fit)
    print(f"  spectrum reference halo: {resolved} ({halo.mcmc_dir})")

    colors = {
        "WW": get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)[1],
        "bb": get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)[3],
    }
    linestyles = {"WW": "-", "bb": "--"}

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15], "hspace": 0.05},
    )

    ax.errorbar(
        halo.E_bins_GeV,
        halo.phi,
        yerr=[halo.phi_err_lo, halo.phi_err_hi],
        fmt="o",
        color="k",
        ecolor="k",
        mfc="white",
        mec="k",
        mew=0.8,
        ms=4.8,
        lw=0.9,
        elinewidth=0.8,
        capsize=2.0,
        label="_nolegend_",
        zorder=8,
    )

    tension_rows: list[str] = []
    for channel in CHANNELS:
        if channel not in channel_data:
            continue
        d = channel_data[channel]
        idx = _best_index(d)
        masses = np.asarray(d["masses_GeV"], float)
        sigmav = np.asarray(d["sigmav_best"], float)
        model = np.asarray(d["spectral_model_phi"], float)[idx, :]
        finite = np.isfinite(halo.E_bins_GeV) & np.isfinite(model)
        sigma = halo.phi_err_sym
        resid = np.divide(
            halo.phi - model,
            sigma,
            out=np.full_like(model, np.nan, dtype=float),
            where=np.isfinite(halo.phi) & np.isfinite(model) & np.isfinite(sigma) & (sigma > 0),
        )
        ax_res.axhline(0.0, color="0.25", lw=0.8, ls=":")

        hoof_r = _tension_value(float(masses[idx]), float(sigmav[idx]), channel, "hoof")
        boddy_r = _tension_value(float(masses[idx]), float(sigmav[idx]), channel, "boddy_pwave")
        tension_rows.append(
            rf"{CHANNEL_LABELS[channel]}: "
            rf"$\mathcal{{R}}_\mathrm{{Hoof}}={_format_r(hoof_r)}$, "
            rf"$\mathcal{{R}}_{{p\mathrm{{-wave}}}}={_format_r(boddy_r)}$"
        )

        ax.plot(
            halo.E_bins_GeV[finite],
            model[finite],
            color=colors[channel],
            ls=linestyles[channel],
            lw=1.95,
            label=rf"{CHANNEL_LABELS[channel]}: $m_\chi={masses[idx]:.0f}$ GeV",
            zorder=6,
        )
        ax_res.plot(
            halo.E_bins_GeV[finite],
            resid[finite],
            color=colors[channel],
            ls=linestyles[channel],
            marker="o" if channel == "WW" else "s",
            ms=3.8,
            lw=1.2,
            label=CHANNEL_LABELS[channel],
        )

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylabel(r"$E^2\,dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8,
        frameon=False,
        ncol=3,
    )
    if tension_rows:
        ax.annotate(
            "\n".join(tension_rows),
            xy=(0.66, 0.965),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", alpha=0.88),
        )
    ax.grid(True, which="both", alpha=0.20)

    ax_res.set_xscale("log")
    ax_res.set_xlabel(r"Photon energy [GeV]")
    ax_res.set_ylabel(r"$(\mathrm{data}-\mathrm{model})/\sigma$")
    ax_res.set_ylim(-4.8, 4.8)
    ax_res.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax_res.grid(True, which="both", alpha=0.20)

    fig.subplots_adjust(top=0.82)
    _save(fig, outdir, "halo_spectrum_bestfit_global_fig6_WW_bb_residuals", formats, dpi)
    plt.close(fig)


def _plot_limit_panel(ax, channel: str, d: dict, color_signal, color_hoof, color_pwave) -> None:
    import matplotlib.ticker as mticker

    masses = np.asarray(d["masses_GeV"], float)
    sigmav = np.asarray(d["sigmav_best"], float)
    best = _best_index(d)
    mass_grid = np.geomspace(max(50.0, np.nanmin(masses) * 0.9), np.nanmax(masses) * 1.05, 320)

    finite_signal = np.isfinite(masses) & np.isfinite(sigmav) & (sigmav > 0)
    ax.plot(
        masses[finite_signal],
        sigmav[finite_signal],
        color=color_signal,
        lw=2.0,
        marker="o",
        ms=4.2,
        label=rf"Halo best fit, {CHANNEL_LABELS[channel]}",
        zorder=7,
    )
    ax.plot(
        [masses[best]],
        [sigmav[best]],
        marker="*",
        ms=11,
        color=color_signal,
        mec="k",
        mew=0.6,
        linestyle="none",
        zorder=8,
    )

    for source, label, linestyle, color in [
        ("hoof", "Hoof s-wave dSph", "-", color_hoof),
        ("boddy_pwave", "Boddy p-wave dSph", "--", color_pwave),
    ]:
        limits = np.array([dsph_upper_limit(float(m), channel, source) for m in mass_grid])
        ok = np.isfinite(limits) & (limits > 0)
        ax.plot(mass_grid[ok], limits[ok], color=color, ls=linestyle, lw=1.7, label=label)

    ax.axhline(
        THERMAL_RELIC,
        color="0.45",
        lw=0.9,
        ls=(0, (1.2, 1.2)),
        label=r"Thermal relic",
        zorder=1,
    )
    ax.text(
        0.04,
        0.06,
        rf"$m_\chi^\mathrm{{best}}={masses[best]:.0f}$ GeV",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    hoof_r = _tension_value(float(masses[best]), float(sigmav[best]), channel, "hoof")
    boddy_r = _tension_value(float(masses[best]), float(sigmav[best]), channel, "boddy_pwave")
    ax.annotate(
        "\n".join([
            rf"$\mathcal{{R}}_\mathrm{{Hoof}}={_format_r(hoof_r)}$",
            rf"$\mathcal{{R}}_{{p\mathrm{{-wave}}}}={_format_r(boddy_r)}$",
        ]),
        xy=(0.04, 0.17),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", alpha=0.86),
    )
    if channel == "WW":
        ax.text(0.62, 0.77, "Boddy p-wave", transform=ax.transAxes,
                color=color_pwave, fontsize=8.5, ha="left", va="center")
        ax.text(0.44, 0.47, "Halo best fit", transform=ax.transAxes,
                color=color_signal, fontsize=8.5, ha="left", va="center")
        ax.text(0.55, 0.30, "Hoof s-wave", transform=ax.transAxes,
                color=color_hoof, fontsize=8.5, ha="left", va="center")
        ax.text(0.63, 0.06, "Thermal relic", transform=ax.transAxes,
                color="0.45", fontsize=8.0, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mass_grid.min(), mass_grid.max())
    ax.set_xticks([200, 300, 500, 1000, 2000])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.text(
        0.95,
        0.93,
        CHANNEL_LABELS[channel],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.grid(True, which="both", alpha=0.20)


def plot_velocity_overlay_two_panel(
    channel_data: dict[str, dict],
    *,
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    palette = get_cmap_colors(cmap_name="plasma", n=8, start=0.12, end=0.92)
    signal = {"WW": palette[1], "bb": palette[4]}
    color_hoof = "0.15"
    color_pwave = palette[6]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.05), sharey=True)
    for ax, channel in zip(axes, CHANNELS):
        if channel not in channel_data:
            ax.set_axis_off()
            continue
        _plot_limit_panel(ax, channel, channel_data[channel], signal[channel], color_hoof, color_pwave)

    axes[0].set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]")
    fig.subplots_adjust(bottom=0.16, top=0.97, wspace=0.07)
    _save(fig, outdir, "velocity_limits_overlay_global_fig6_XX", formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_plot_style(args.style, base_fontsize=10, linewidth=1.5, n_colors=10, cmap_name="plasma")
    channel_data = _load_channels(args)
    plot_halo_spectrum_residuals(
        channel_data,
        fit=args.fit,
        halo_reference=args.halo_reference,
        outdir=args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
    )
    plot_velocity_overlay_two_panel(
        channel_data,
        outdir=args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
