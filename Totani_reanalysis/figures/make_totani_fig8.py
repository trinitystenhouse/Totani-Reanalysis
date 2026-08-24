#!/usr/bin/env python3
"""
Reproduce Totani Fig. 8: halo-component E²dN/dE at the Galactic poles.

Three panels (linear y-axis, log x-axis):
  Top:    NFW-ρ^2.5  (halo fit, disk-excluded, mcmc_results_fig5)
  Middle: NFW-ρ^2    (halo fit, disk-excluded, mcmc_results_fig6)
  Bottom: NFW-ρ^1    (halo fit, disk-excluded, mcmc_results_fig7)

Units: E²dN/dE at the Galactic poles [MeV cm⁻² s⁻¹ sr⁻¹]

Central value: posterior median (f_p50), matching Totani's fig 8.
Error bars: 1σ band [p16, p84].

Why f_p50 and not f_ml
-----------------------
f_ml is the single highest-likelihood sample found in the chain. For components
that are near zero and constrained ≥ 0 (e.g. the halo at low energies), the
posterior is skewed: most of the probability mass sits above zero but the ML
point can land below the 16th percentile, producing negative yerr. The median
(p50) is the correct central value because it sits inside the posterior mass by
definition, guaranteeing p16 ≤ p50 ≤ p84. This matches how Totani quotes his
MCMC error bars.

Why this works without data at b=90°
-------------------------------------
The NFW template is built with --norm pole, meaning the LOS integral J(l,b)
is divided by J_pole = ∫ρᵖ ds|_{b=90°} before computing counts:

    mu_nfw[k] = (iso_target_E2 / Ectr_mev[k]²) × (J / J_pole) × expo[k] × ω × dE[k]

So at the Galactic pole, J/J_pole = 1, and at f=1:

    E²dN/dE|_pole = f × iso_target_e2

This is an analytic consequence of the template normalisation — no pixel data
outside the ROI (|b| ≤ 60°) is needed. iso_target_e2 is stored in each npz.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from totani_helpers.totani_io import read_counts_and_ebounds
from helpers.fermi_plotting import (
    CONF_ANNOT_FS,
    CONF_LABEL_FS,
    CONF_TICK_FS,
)
from helpers.plot_style import current_style_is_transparent, normalise_plot_style, set_paper_style

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
MCMC_DIR = os.path.join(REPO_DIR, "Totani_paper_check", "mcmc")

ISO_TARGET_E2_DEFAULT = 1e-4   # MeV cm⁻² s⁻¹ sr⁻¹ — fallback if not stored in npz

FONT_PRESETS = {
    "paper": {
        "panel": 10.0,
        "label": 10.0,
        "ticks": 9.0,
        "ylabel": 11.0,
    },
    "conference": {
        "panel": float(CONF_ANNOT_FS),
        "label": float(CONF_LABEL_FS),
        "ticks": float(CONF_TICK_FS),
        "ylabel": float(CONF_LABEL_FS),
    },
}


def resolve_plot_config(cmap_name):
    cmap_key = str(cmap_name).strip()
    try:
        style_key = normalise_plot_style(cmap_key if cmap_key not in {"", "default"} else None, default="paper")
    except ValueError:
        style_key = "paper"
    if style_key != "paper":
        set_paper_style(style=style_key, linewidth=1.6, n_colors=12, cmap_name="plasma")
        fg = plt.rcParams.get("text.color", "white")
        return {
            "mode": "conference",
            "fonts": FONT_PRESETS["conference"],
            "fg": fg,
            "markerface": "none",
            "figsize": (8.0, 12.0),
            # Wide layout: three short panels in a row. Not the stacked size
            # transposed -- a 12x8 row would leave each panel taller than it is
            # wide, which defeats the point of going wide for a 16:9 slide.
            "figsize_wide": (14.0, 4.6),
        }

    set_paper_style(style="paper", linewidth=1.6, n_colors=12, cmap_name="plasma")
    return {
        "mode": "paper",
        "fonts": FONT_PRESETS["paper"],
        "fg": "k",
        "markerface": "white",
        "figsize": (5.0, 8.0),
        "figsize_wide": (9.5, 3.2),
    }


def _load_pole_flux(
    mcmc_dir: str,
    nfw_label: str,
    nE: int,
    central_stat: str = "f_p50",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NFW pole flux and 1σ error band directly from MCMC npz files.

    Central value is f_p50 (posterior median) so that p16 ≤ central ≤ p84
    is guaranteed by construction, keeping all yerr values non-negative.

    Returns
    -------
    y_ctr, y_p16, y_p84 : (nE,) arrays in MeV cm⁻² s⁻¹ sr⁻¹
        E²dN/dE at the Galactic poles = f_stat × iso_target_e2
    """
    y_ctr = np.full(nE, np.nan)
    y_p16 = np.full(nE, np.nan)
    y_p84 = np.full(nE, np.nan)

    for k in range(nE):
        path = os.path.join(mcmc_dir, f"mcmc_results_k{k:02d}.npz")
        if not os.path.exists(path):
            continue

        npz = np.load(path, allow_pickle=True)

        # iso_target_e2 sets the physical scale: f=1 → E²dN/dE|_pole = iso_target_e2
        iso_target_e2 = float(
            npz["iso_target_e2"]
            if "iso_target_e2" in npz
            else ISO_TARGET_E2_DEFAULT
        )

        labels = [str(x) for x in npz["labels"].tolist()]
        # Match the NFW label — exact first, then single-nfw fallback
        idx = None
        if nfw_label in labels:
            idx = labels.index(nfw_label)
        else:
            nfw_candidates = [i for i, l in enumerate(labels) if "nfw" in l.lower()]
            if len(nfw_candidates) == 1:
                idx = nfw_candidates[0]
            elif len(nfw_candidates) > 1:
                raise RuntimeError(
                    f"Multiple NFW labels in bin k={k}: "
                    f"{[labels[i] for i in nfw_candidates]}. "
                    "Pass the exact label via --nfw-label-*."
                )

        if idx is None:
            raise KeyError(
                f"NFW label '{nfw_label}' not found in bin k={k}. "
                f"Available labels: {labels}"
            )

        if central_stat not in npz:
            raise KeyError(
                f"Stat '{central_stat}' not found in {path}. "
                f"Available keys: {list(npz.keys())}"
            )

        y_ctr[k] = float(npz[central_stat][idx]) * iso_target_e2
        y_p16[k] = float(npz["f_p16"      ][idx]) * iso_target_e2
        y_p84[k] = float(npz["f_p84"      ][idx]) * iso_target_e2

    return y_ctr, y_p16, y_p84


def main():
    ap = argparse.ArgumentParser(description="Reproduce Totani Fig. 8 (halo pole flux).")
    ap.add_argument("--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig8"))
    ap.add_argument("--mcmc-dir-gce",
        default=os.path.join(MCMC_DIR, "mcmc_results_fig2_3"),
        help="Full-disk GCE fit (fig2_3) — kept for reference but NOT used in fig8 panels.")
    ap.add_argument("--mcmc-dir-rho25",
        default=os.path.join(MCMC_DIR, "mcmc_results_fig5"),
        help="NFW-rho^2.5 halo fit, disk-excluded (fig5) — used for the top panel of fig8.")
    ap.add_argument("--mcmc-dir-rho2",
        default=os.path.join(MCMC_DIR, "mcmc_results_fig6"))
    ap.add_argument("--mcmc-dir-rho1",
        default=os.path.join(MCMC_DIR, "mcmc_results_fig7"))
    ap.add_argument("--central-stat", default="f_p50",
        choices=["f_p50", "f_ml", "f_p05", "f_p16", "f_p84", "f_p95"],
        help="MCMC statistic used as the central (plotted) value. "
             "Default f_p50 (posterior median) guarantees non-negative error bars. "
             "Use f_ml only if you understand the skewness issue described in the docstring.")
    ap.add_argument(
        "--cmap",
        default="default",
        help="Display mode. Use 'dark_transparent' for transparent dark styling with conference-size fonts.",
    )
    ap.add_argument(
        "--layout", default="stacked", choices=["stacked", "wide"],
        help="Panel arrangement. 'stacked' (default) is Totani's own 3x1 column "
             "and writes totani_fig8.png; 'wide' is the same three panels in a "
             "row for a 16:9 slide and writes totani_fig8_wide.png.",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_cfg = resolve_plot_config(args.cmap)

    # Energy bin centres — only used for the x-axis
    _, _, _, _, Ectr_mev, _ = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)
    Ectr_gev = Ectr_mev / 1000.0

    # ---------------------------------------------------------------------------
    # Axis limits and ticks matching Totani Fig. 8 exactly.
    #
    # Totani uses three independent linear y-axes scaled to each model's range:
    #
    #   Top    (NFW-ρ^2.5 / GCE):  y ∈ [-1e-5,  1e-5],  ticks at -1,-0.5,0,0.5,1 × 10^-5
    #   Middle (NFW-ρ^2):           y ∈ [-2e-5,  4e-5],  ticks at -2,-1,0,1,2,3,4 × 10^-5
    #   Bottom (NFW-ρ^1):           y ∈ [-1e-4,  2e-4],  ticks at -1,0,1,2         × 10^-4
    #
    # sharex=True (x-axis shared); y-axes are NOT shared.
    # ---------------------------------------------------------------------------
    panels = [
        {
            # Totani Fig. 8 top panel: NFW-ρ^2.5 from the halo-region fit
            # (disk excluded, fb_pos/fb_neg templates) = mcmc_results_fig5.
            # Do NOT use fig2_3 here — that fit includes the disk and uses
            # fb_flat, producing a completely different lnL scale and spectrum.
            "title": r"NFW-$\rho^{2.5}$ (GCE)",
            "mcmc_dir": args.mcmc_dir_rho25,   # fig5: disk-excluded, fb_pos/fb_neg templates
            "nfw_label": "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
            "marker": "^",
            # Totani top panel: y ∈ [-1e-5, 2e-5], ticks at -1,-0.5,0,0.5,1 × 10^-5
            "ylim":   (-1.0e-5, 2.0e-5),
            "yticks": np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * 1e-5,
        },
        {
            "title": r"NFW-$\rho^2$",
            "mcmc_dir": args.mcmc_dir_rho2,
            "nfw_label": "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
            "marker": "o",
            # Totani middle panel: y ∈ [-2e-5, 4e-5], ticks -2,-1,0,1,2,3,4 × 10^-5
            # (data peak ~3.4e-5, needs clear headroom above)
            "ylim":   (-2.0e-5, 4.0e-5),
            "yticks": np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1e-5,
        },
        {
            "title": r"NFW-$\rho^1$",
            "mcmc_dir": args.mcmc_dir_rho1,
            "nfw_label": "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno",
            "marker": "s",
            # Totani bottom panel: y ∈ [-1e-4, 2e-4], ticks -1,0,1,2 × 10^-4
            "ylim":   (-1.0e-4, 2.0e-4),
            "yticks": np.array([-1.0, 0.0, 1.0, 2.0]) * 1e-4,
        },
    ]

    # y-axes are NOT shared — each panel has its own scale as in Totani.
    # --layout wide puts the same three panels in a row instead of a column.
    # The stacked form is Totani's own and stays the default; wide exists
    # because a 3-high column is a tall sliver on a 16:9 slide.
    wide = (args.layout == "wide")
    if wide:
        fig, axes = plt.subplots(
            1, 3, figsize=plot_cfg.get("figsize_wide", plot_cfg["figsize"]))
    else:
        fig, axes = plt.subplots(3, 1, figsize=plot_cfg["figsize"], sharex=True)

    for ax, panel in zip(axes, panels):
        y_ctr, y_p16, y_p84 = _load_pole_flux(
            mcmc_dir=panel["mcmc_dir"],
            nfw_label=panel["nfw_label"],
            nE=nE,
            central_stat=args.central_stat,
        )

        yerr = np.vstack([y_ctr - y_p16, y_p84 - y_ctr])

        ax.set_xscale("log")
        ax.axhline(0.0, color=plot_cfg["fg"], lw=0.7, ls=":")
        ax.errorbar(
            Ectr_gev,
            y_ctr,
            yerr=yerr,
            linestyle="none",
            color=plot_cfg["fg"],
            ecolor=plot_cfg["fg"],
            marker=panel["marker"],
            mfc=plot_cfg["markerface"],
            mec=plot_cfg["fg"],
            mew=0.7,
            markersize=5.5,
            elinewidth=0.7,
            capsize=2,
        )
        if wide:
            # In-panel at (0.62, 0.84) lands on the rho^1 spectrum once the
            # panels are short and wide; above the axes is clear in all three.
            ax.set_title(panel["title"], fontsize=plot_cfg["fonts"]["panel"])
        else:
            ax.text(0.62, 0.84, panel["title"], transform=ax.transAxes,
                    fontsize=plot_cfg["fonts"]["panel"])
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.tick_params(labelsize=plot_cfg["fonts"]["ticks"])
        ax.set_xlim(1.0, 1000.0)
        # A narrow panel makes the log locator drop the intermediate decades,
        # leaving only 10^0 and 10^3 on the wide layout. Pin all four.
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=4))
        ax.set_ylim(*panel["ylim"])
        ax.set_yticks(panel["yticks"])
        # ScalarFormatter renders the ×10^N offset label in the top-left corner,
        # exactly as in Totani's figures.
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if wide:
        # Every panel needs its own x-label once they no longer share a column,
        # and the shared y-label is set on the leftmost axes rather than as a
        # fig.text at a hand-picked x -- that x was tuned for the tall figure.
        for ax in axes:
            ax.set_xlabel("photon energy [GeV]", fontsize=plot_cfg["fonts"]["label"])
        # Rotated text needs its full length VERTICALLY. The stacked figure is
        # 12 in tall so the long one-line label fits; the wide one is ~4.6 in,
        # less than the label's own length at the conference size, so it ran off
        # both ends of the canvas. Wrap it rather than abbreviate -- the wording
        # is the same as the stacked version, only the line breaks differ, so
        # the two layouts still read as the same figure.
        # labelpad: three rotated lines are three lines WIDE, so the innermost
        # one lands on the y tick labels at the default pad.
        axes[0].set_ylabel(
            "flux $E^2 dN/dE$ at\nGalactic poles\n[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
            fontsize=plot_cfg["fonts"]["ticks"],
            labelpad=plot_cfg["fonts"]["ticks"] * 0.9,
        )
        fig.tight_layout()
    else:
        axes[-1].set_xlabel("photon energy [GeV]", fontsize=plot_cfg["fonts"]["label"])
        fig.text(
            0.03, 0.5,
            r"flux $E^2 dN/dE$ at Galactic poles  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
            rotation=90, va="center", fontsize=plot_cfg["fonts"]["ylabel"],
        )
        fig.tight_layout(rect=(0.08, 0.04, 1.0, 1.0))

    out = os.path.join(args.outdir,
                       "totani_fig8_wide.png" if wide else "totani_fig8.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig)
    print("✓ wrote", out)


if __name__ == "__main__":
    main()
