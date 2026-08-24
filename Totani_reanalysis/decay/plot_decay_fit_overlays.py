#!/usr/bin/env python3
"""Plot rho^1 halo spectra with best-fit decaying-DM PPPC templates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.ticker as mticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(HERE))

os.environ.setdefault("REPO_PATH", str(REPO_DIR))

from helpers.plot_style import (  # noqa: E402
    apply_plot_style,
    current_savefig_kwargs,
    get_cmap_colors,
)
from decay_fit import D_pole_GeV_cm2, TAU_IGRB_LOWER, fit_channel  # noqa: E402
from core.spectral_reshaping import pppc_energy_flux_template  # noqa: E402
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402


_PAPER_CHANNEL_COLORS = get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)
CHANNEL_STYLES = {
    "WW": {"color": _PAPER_CHANNEL_COLORS[1], "ls": "-", "marker": "o", "label": r"$W^+W^-$"},
    "bb": {"color": _PAPER_CHANNEL_COLORS[3], "ls": "--", "marker": "s", "label": r"$b\bar b$"},
    "tautau": {
        "color": _PAPER_CHANNEL_COLORS[4],
        "ls": "-.",
        "marker": "^",
        "label": r"$\tau^+\tau^-$",
    },
}


def _profile_label(profile: str) -> str:
    labels = {
        "global_rho1": r"Global NFW-$\rho^1$",
        "rho1": r"Totani/cellwise NFW-$\rho^1$",
    }
    return labels.get(profile, profile.replace("_", r"\_"))


def _best_fits(profile: str, channels: list[str], masses: np.ndarray, table_path: str | None):
    mcmc_dir = _MCMC_DIRS.get(profile, Path(profile))
    halo = load_halo_spectrum(mcmc_dir)
    results = {}
    for channel in channels:
        best = fit_channel(channel, halo, masses, table_path=table_path)
        if not np.isfinite(best.get("chi2", np.inf)):
            continue
        shape = pppc_energy_flux_template(
            halo.E_bins_GeV,
            best["m_chi"] / 2.0,
            channel=channel,
            primary="gamma",
            table_path=table_path,
            normalise=False,
        )
        model = best["norm"] * shape
        tau = 1000.0 * D_pole_GeV_cm2() / (4.0 * np.pi * best["m_chi"] * best["norm"])
        results[channel] = {**best, "model": model, "tau": tau}
    return halo, results


def _plot_profile(
    *,
    profile: str,
    channels: list[str],
    masses: np.ndarray,
    table_path: str | None,
    outdir: Path,
    formats: list[str],
) -> list[Path]:
    halo, results = _best_fits(profile, channels, masses, table_path)
    E = np.asarray(halo.E_bins_GeV, dtype=float)
    phi = np.asarray(halo.phi, dtype=float)
    err_lo = np.asarray(halo.phi_err_lo, dtype=float)
    err_hi = np.asarray(halo.phi_err_hi, dtype=float)
    err_sym = 0.5 * (err_lo + err_hi)
    mask = halo.positive_mask & halo.finite_mask & (err_sym > 0)

    fig, (ax, rx) = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15], "hspace": 0.05},
    )

    ax.errorbar(
        E,
        phi,
        yerr=[err_lo, err_hi],
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

    lifetime_rows = []
    for channel in channels:
        fit = results.get(channel)
        if fit is None:
            continue
        style = CHANNEL_STYLES.get(channel, {})
        color = style.get("color")
        ls = style.get("ls", "-")
        marker = style.get("marker", "o")
        channel_label = style.get("label", channel)
        label = (
            rf"{channel_label}: $m_\chi={fit['m_chi']:.0f}$ GeV"
        )
        ax.plot(E, fit["model"], color=color, ls=ls, lw=1.95, label=label, zorder=6)
        residual = (phi - fit["model"]) / err_sym
        rx.plot(
            E[mask],
            residual[mask],
            color=color,
            ls=ls,
            marker=marker,
            ms=3.8,
            lw=1.2,
            label=channel_label,
        )
        lifetime_rows.append(
            rf"{channel_label}: "
            rf"$\tau/\tau_\mathrm{{IGRB}}={fit['tau'] / TAU_IGRB_LOWER:.2g}$"
        )

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylabel(r"$E^2 dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8,
        frameon=False,
        ncol=min(3, max(1, len(results))),
    )
    if lifetime_rows:
        ax.annotate(
            "\n".join(lifetime_rows),
            xy=(0.66, 0.965),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", alpha=0.88),
        )
    ax.text(
        0.02,
        0.94,
        _profile_label(profile),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    rx.axhline(0.0, color="0.25", lw=0.8, ls=":")
    rx.set_xscale("log")
    rx.set_xlabel(r"Photon energy [GeV]")
    rx.set_ylabel(r"$(\mathrm{data}-\mathrm{model})/\sigma$")
    rx.legend(loc="upper right", fontsize=8, frameon=False, ncol=min(3, max(1, len(results))))

    for axis in (ax, rx):
        axis.grid(True, which="both", alpha=0.2, lw=0.5)

    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    fig.subplots_adjust(top=0.82)
    for fmt in formats:
        path = outdir / f"decay_fit_overlay_{profile}.{fmt}"
        fig.savefig(path, dpi=220, **current_savefig_kwargs(bbox_inches="tight"))
        saved.append(path)
    plt.close(fig)

    print(f"\n{profile}:")
    print(f"{'chan':6s} {'m_chi[GeV]':>10s} {'tau[s]':>12s} {'chi2/ndof':>10s}")
    for channel in channels:
        fit = results.get(channel)
        if fit is None:
            print(f"{channel:6s}  no finite fit")
            continue
        print(
            f"{channel:6s} {fit['m_chi']:10.0f} {fit['tau']:12.3e} "
            f"{fit['chi2'] / fit['ndof']:10.2f}"
        )
    for path in saved:
        print(f"Saved: {path}")
    return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["global_rho1", "rho1"])
    parser.add_argument("--channels", nargs="+", default=["WW", "bb", "tautau"])
    parser.add_argument("--m-min", type=float, default=100.0)
    parser.add_argument("--m-max", type=float, default=6000.0)
    parser.add_argument("--n-mass", type=int, default=120)
    parser.add_argument("--pppc-table", default=None)
    parser.add_argument("--outdir", type=Path, default=HERE / "results" / "decay_fit_overlays")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--style", default=os.environ.get("PLOT_STYLE", "paper"))
    args = parser.parse_args(argv)

    apply_plot_style(args.style, n_colors=max(3, len(args.channels)))
    masses = np.geomspace(args.m_min, args.m_max, args.n_mass)
    for profile in args.profiles:
        _plot_profile(
            profile=profile,
            channels=list(args.channels),
            masses=masses,
            table_path=args.pppc_table,
            outdir=args.outdir,
            formats=list(args.formats),
        )


if __name__ == "__main__":
    main()
