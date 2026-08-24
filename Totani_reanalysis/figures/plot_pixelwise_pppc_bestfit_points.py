#!/usr/bin/env python3
r"""
Paper-style PPPC spectral best-fit overlays for extracted halo spectra.

Two figures are produced:

  * annihilation_pppc_bestfit_points_<tag>_rho2
  * decay_pppc_bestfit_points_<tag>_rho1

The models are fitted spectrally: for each channel and scanned mass, one
normalisation is fitted to the extracted halo spectrum. No per-mass spatial
refit is run here.
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
    _fit_pppc_reference_spectrum,
)
from core.spectral_reshaping import pppc_energy_flux_template  # noqa: E402
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402
from decay_fit import D_pole_GeV_cm2, TAU_IGRB_LOWER, fit_channel  # noqa: E402


CHANNEL_LABELS = {
    "WW": r"$W^+W^-$",
    "bb": r"$b\bar b$",
    "tautau": r"$\tau^+\tau^-$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ann-channels", nargs="+", default=["WW", "bb"])
    parser.add_argument("--decay-channels", nargs="+", default=["WW", "bb", "tautau"])
    parser.add_argument("--ann-m-min", type=float, default=150.0)
    parser.add_argument("--ann-m-max", type=float, default=2500.0)
    parser.add_argument("--ann-n-mass", type=int, default=180)
    parser.add_argument("--decay-m-min", type=float, default=100.0)
    parser.add_argument("--decay-m-max", type=float, default=6000.0)
    parser.add_argument("--decay-n-mass", type=int, default=180)
    parser.add_argument("--pppc-table", default=None)
    parser.add_argument("--outdir", type=Path, default=_HERE / "plots_pppc_pixelwise_bestfit")
    parser.add_argument(
        "--ann-profile",
        default="pixelwise_global_rho2",
        choices=sorted(_MCMC_DIRS),
        help="Halo spectrum/profile key used for the annihilation overlay.",
    )
    parser.add_argument(
        "--decay-profile",
        default="pixelwise_global_rho1",
        choices=sorted(_MCMC_DIRS),
        help="Halo spectrum/profile key used for the decay overlay.",
    )
    parser.add_argument(
        "--tag",
        default="pixelwise",
        help="Short filename/summary tag for the selected profile family.",
    )
    parser.add_argument(
        "--ann-title",
        default=None,
        help="Plot title label for the annihilation halo spectrum.",
    )
    parser.add_argument(
        "--decay-title",
        default=None,
        help="Plot title label for the decay halo spectrum.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--style", default=os.environ.get("PLOT_STYLE", "paper"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _fit_annihilation(halo, channel: str, masses: np.ndarray) -> dict:
    rows = []
    for mass in masses:
        fit = _fit_pppc_reference_spectrum(halo, float(mass), channel)
        rows.append((float(mass), fit))
    chi2 = np.asarray([row[1]["chi2"] for row in rows], float)
    if not np.any(np.isfinite(chi2)):
        return {"ok": False, "channel": channel}
    best_idx = int(np.nanargmin(chi2))
    best_mass, best = rows[best_idx]
    return {
        "ok": True,
        "channel": channel,
        "mass": best_mass,
        "model": np.asarray(best["model_phi"], float),
        "chi2": float(best["chi2"]),
        "ndof": int(best["ndof"]),
        "sigmav": float(best["sigmav"]),
    }


def _fit_decay(halo, channel: str, masses: np.ndarray, table_path: str | None) -> dict:
    best = fit_channel(channel, halo, masses, table_path=table_path)
    if not np.isfinite(best.get("chi2", np.inf)):
        return {"ok": False, "channel": channel}
    model_shape = pppc_energy_flux_template(
        halo.E_bins_GeV,
        best["m_chi"] / 2.0,
        channel=channel,
        primary="gamma",
        table_path=table_path,
        normalise=False,
    )
    model = best["norm"] * model_shape
    tau = 1000.0 * D_pole_GeV_cm2() / (4.0 * np.pi * best["m_chi"] * best["norm"])
    return {
        "ok": True,
        "channel": channel,
        "mass": float(best["m_chi"]),
        "model": np.asarray(model, float),
        "chi2": float(best["chi2"]),
        "ndof": int(best["ndof"]),
        "tau": float(tau),
    }


def _save(fig, outdir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, **current_savefig_kwargs(bbox_inches="tight"))
        saved.append(path)
        print(f"  saved {path}")
    return saved


def _draw_overlay(
    *,
    halo,
    fits: list[dict],
    title_label: str,
    quantity: str,
    stem: str,
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    E = np.asarray(halo.E_bins_GeV, float)
    phi = np.asarray(halo.phi, float)
    err_lo = np.asarray(halo.phi_err_lo, float)
    err_hi = np.asarray(halo.phi_err_hi, float)
    err = np.asarray(halo.phi_err_sym, float)
    mask = halo.finite_mask & halo.positive_mask & np.isfinite(err) & (err > 0.0)

    colors = get_cmap_colors(cmap_name="plasma", n=max(5, len(fits) + 2), start=0.18, end=0.86)
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]

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
        label="Pixelwise halo spectrum",
        zorder=8,
    )

    summary_rows = []
    for i, fit in enumerate(f for f in fits if f.get("ok")):
        channel = str(fit["channel"])
        label = CHANNEL_LABELS.get(channel, channel)
        color = colors[i + 1]
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        model = np.asarray(fit["model"], float)
        finite = np.isfinite(E) & np.isfinite(model)
        fit_label = rf"{label}: $m_\chi={fit['mass']:.0f}$ GeV"

        ax.plot(E[finite], model[finite], color=color, ls=ls, lw=1.7, alpha=0.72)
        ax.plot(
            E[finite],
            model[finite],
            linestyle="none",
            marker=marker,
            color=color,
            mfc="white",
            mec=color,
            mew=0.9,
            ms=4.8,
            label=fit_label,
            zorder=7,
        )

        residual = np.divide(
            phi - model,
            err,
            out=np.full_like(phi, np.nan, dtype=float),
            where=np.isfinite(phi) & np.isfinite(model) & np.isfinite(err) & (err > 0.0),
        )
        rx.plot(
            E[mask],
            residual[mask],
            color=color,
            ls=ls,
            marker=marker,
            ms=3.6,
            lw=1.15,
            label=label,
        )

        if quantity == "annihilation":
            summary_rows.append(
                rf"{label}: $\langle\sigma v\rangle={fit['sigmav']:.2g}$ cm$^3$ s$^{-1}$, "
                rf"$\chi^2/\nu={fit['chi2'] / max(fit['ndof'], 1):.2f}$"
            )
        else:
            summary_rows.append(
                rf"{label}: $\tau/\tau_\mathrm{{IGRB}}={fit['tau'] / TAU_IGRB_LOWER:.2g}$, "
                rf"$\chi^2/\nu={fit['chi2'] / max(fit['ndof'], 1):.2f}$"
            )

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylabel(r"$E^2\,dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.text(0.02, 0.94, title_label, transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8,
        frameon=False,
        ncol=min(3, max(1, len(fits) + 1)),
    )
    if summary_rows:
        ax.annotate(
            "\n".join(summary_rows),
            xy=(0.56, 0.965),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", alpha=0.88),
        )
    ax.grid(True, which="both", alpha=0.20)

    rx.axhline(0.0, color="0.25", lw=0.8, ls=":")
    rx.set_xscale("log")
    rx.set_xlabel(r"Photon energy [GeV]")
    rx.set_ylabel(r"$(\mathrm{data}-\mathrm{model})/\sigma$")
    rx.set_ylim(-4.8, 4.8)
    rx.legend(loc="upper right", fontsize=8, frameon=False, ncol=min(3, max(1, len(fits))))
    rx.grid(True, which="both", alpha=0.20)

    fig.subplots_adjust(top=0.82)
    _save(fig, outdir, stem, formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_plot_style(args.style, base_fontsize=10, linewidth=1.5, n_colors=10, cmap_name="plasma")
    tag = str(args.tag).strip().replace(" ", "_")
    ann_title = args.ann_title or rf"{tag.replace('_', ' ').title()} NFW-$\rho^2$"
    decay_title = args.decay_title or rf"{tag.replace('_', ' ').title()} NFW-$\rho^1$"

    ann_halo = load_halo_spectrum(_MCMC_DIRS[str(args.ann_profile)])
    ann_masses = np.geomspace(args.ann_m_min, args.ann_m_max, args.ann_n_mass)
    ann_fits = [_fit_annihilation(ann_halo, channel, ann_masses) for channel in args.ann_channels]
    _draw_overlay(
        halo=ann_halo,
        fits=ann_fits,
        title_label=ann_title,
        quantity="annihilation",
        stem=f"annihilation_pppc_bestfit_points_{tag}_rho2",
        outdir=args.outdir,
        formats=list(args.formats),
        dpi=args.dpi,
    )

    decay_halo = load_halo_spectrum(_MCMC_DIRS[str(args.decay_profile)])
    decay_masses = np.geomspace(args.decay_m_min, args.decay_m_max, args.decay_n_mass)
    decay_fits = [
        _fit_decay(decay_halo, channel, decay_masses, args.pppc_table)
        for channel in args.decay_channels
    ]
    _draw_overlay(
        halo=decay_halo,
        fits=decay_fits,
        title_label=decay_title,
        quantity="decay",
        stem=f"decay_pppc_bestfit_points_{tag}_rho1",
        outdir=args.outdir,
        formats=list(args.formats),
        dpi=args.dpi,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = [
        f"{tag} PPPC spectral best-fit point overlays",
        f"ann_profile={args.ann_profile}",
        f"decay_profile={args.decay_profile}",
        "",
    ]
    for name, fits in [("annihilation rho2", ann_fits), ("decay rho1", decay_fits)]:
        summary.append(name)
        for fit in fits:
            if not fit.get("ok"):
                summary.append(f"  {fit['channel']}: no finite fit")
                continue
            extra = (
                f"sigmav={fit['sigmav']:.6e} cm3/s"
                if "sigmav" in fit
                else f"tau={fit['tau']:.6e} s"
            )
            summary.append(
                f"  {fit['channel']}: m_chi={fit['mass']:.6g} GeV, "
                f"{extra}, chi2={fit['chi2']:.6g}, ndof={fit['ndof']}"
            )
        summary.append("")
    path = args.outdir / f"{tag}_pppc_bestfit_points_summary.txt"
    path.write_text("\n".join(summary))
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
