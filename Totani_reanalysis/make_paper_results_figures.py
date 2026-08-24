#!/usr/bin/env python3
r"""
Paper result figures for the 20 GeV halo dark-matter interpretation, all in the
shared "paper" plotting style with errors propagated from the pixel-level MCMC
posteriors (p16/p50/p84).

Figures
-------
  f8_pole_flux_profiles      (--only 1) Totani Fig. 8: pole E^2 dN/dE for NFW
                              rho^2.5 / rho^2 / rho^1, pixel-level fits with the
                              Galactic disk EXCLUDED and INCLUDED overlaid.
  f9_likelihood_profiles     (--only 2) Totani Fig. 9: per-bin halo Delta lnL for
                              the three profiles, pixel-level disk-excluded.
  f_annih_bb_WW              (--only 3) bb and WW annihilation fits on the rho^2
                              pixel-level halo, with residuals.
  f_dsph_tension_WW_bb       (--only 4) full-width two-panel <sigma v>(m) vs the
                              s-wave (Hoof) and p-wave (Boddy) dSph limits.
  f_systematics_impact       (--only 5) systematics suite: per-variant halo
                              spectra vs baseline band + tension forest.
  f_decay_rho1              (--only 6) tautau / bb / WW decay fits on the rho^1
                              pixel-level halo, with residuals.
  f_posterior_template_covariance
                              (--only 19) posterior-covariance robustness
                              check: (a) 9x9 posterior
                              correlation matrix of the fitted template
                              normalisations at the 20.76 GeV peak bin; (b)
                              halo correlation with every other template vs
                              energy, 10-40 GeV signal band shaded.
  f_hardic_injected_population_contest
                              (--only 20) hard-IC injected-population
                              robustness check: per hard-IC
                              injected-population morphology, (a) the halo
                              amplitude is unchanged and (b) the injected
                              population cannot replace the halo, with the
                              unphysical CMB-only ISRF bracket set apart.

Usage:
  python make_paper_results_figures.py                 # all figures
  python make_paper_results_figures.py --only 3 4      # selected
  python make_paper_results_figures.py --copy-to <overleaf>/paper_plots
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # Totani_reanalysis
_REPO = _HERE.parent                             # repository root
sys.path.insert(0, str(_REPO))                   # shared, helpers, totani_helpers
from shared.paths import (  # noqa: E402
    PAPER_CHECK,
    REANALYSIS_FIGURES,
    REANALYSIS_MCMC,
    REANALYSIS_RESULTS,
    REPO_ROOT,
)
_REPO = REPO_ROOT
os.environ.setdefault("REPO_PATH", str(_REPO))
sys.path.insert(0, str(_HERE))                   # core

PC = PAPER_CHECK
MCMC = REANALYSIS_MCMC
PPPC = _HERE / "results" / "pppc_mass_scan"
SYST = REANALYSIS_FIGURES / "systematics" / "plots_global_systematics"
SYST_LEGACY = REANALYSIS_FIGURES / "plots_global_systematics"
SYST_BAND = REANALYSIS_RESULTS / "global-systematics" / "systematics_science_band.txt"
PPPC_TABLE = Path(os.environ.get(
    "PPPC4DMID_GAMMAS",
    _REPO / "Totani_reanalysis" / "data" / "AtProduction_gammas.dat"))

from helpers.plot_style import (  # noqa: E402
    add_style_argument, apply_plot_style, current_savefig_kwargs, get_cmap_colors,
    normalise_plot_style, plasma_color, scaled_figsize as _figsize, style_sizing,
    theme_accent as _accent, theme_ink as _ink, theme_legend_kw,
)
from core.totani_data_loader import load_halo_spectrum  # noqa: E402
from core.spectral_reshaping import pppc_energy_flux_template  # noqa: E402
from core.dsph_limits_multi import dsph_upper_limit  # noqa: E402

CH_LABEL = {"bb": r"$b\bar b$", "WW": r"$W^+W^-$", "tautau": r"$\tau^+\tau^-$"}
YLAB_MAIN = r"$E^2\,\mathrm{d}N/\mathrm{d}E$ at galactic poles"
YLAB_UNITS = r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
YLAB = YLAB_MAIN + "\n" + YLAB_UNITS
XLAB = r"$E$ [GeV]"
THERMAL_RELIC = 2.2e-26

# Print geometry: figure widths equal their LaTeX print width so figures are
# included at scale 1 (no LaTeX rescaling) and fonts are consistent across the
# paper. Single-column figures -> COL_W (\columnwidth); full-width figure* -> FULL_W
# (\textwidth) in PRD revtex4 twocolumn.
COL_W = 3.4
FULL_W = 7.05
# Structural text (axis labels, ticks, legends, titles) inherits the global
# rcParams scheme from apply_plot_style. ANNOT_FS is the ONE size used for dense
# in-panel data annotations (R-values, variant names) that must be smaller to fit.
ANNOT_FS = 8

# ---------------------------------------------------------------------------
# Total s-wave dSph tension band, from systematics/total_systematic_band.py.
# Foreground leg = FOREGROUND_GROUPS in systematics/propagate_systematics.py
# {baseline, diffuse, diffuse_ics, diffuse_grid_bubbles}, envelope x[0.375, 1.320]
# (low: diffuse_giem_only; high: iem_so_z6_bubbles_reextracted), combined in
# quadrature with the halo-J (+/-48%) and dwarf-J (+/-54%) legs, anchored on
# R0 = 5.121 (global_fig6 bb, 200-point scan over 20-2000 GeV).
#   History: [1.6, 9.3] pre-IEM-grid, [1.7, 10.0] on the 14/80-point grids,
#   [1.6, 9.8] once every scan was put on the same 200-point grid. The lower
#   edge returning to 1.6 is the re-anchoring, not a reversion to the old band.
# Every figure that draws this band must read these, not literals.
R_BAND_LO, R_BAND_HI = 1.6, 9.8
R_BAND_CENTRAL = 5.1
# Per-channel bands. bb is the module default above and drives every bb-only
# figure. WW was propagated separately: propagate_systematics.py --channel WW
# anchored on the global_fig6 scan (557 GeV, 1.001e-24, 4.073) with foreground
# leg x[0.341, 1.314], then total_systematic_band.py --R0 4.073 --fg-lo 0.341
# --fg-hi 1.314, giving 4.1 [1.2, 7.7]. Any two-channel figure must read this
# dict, not the scalars, or it draws the bb band over a WW panel.
R_BAND = {
    "bb": (R_BAND_LO, R_BAND_HI, R_BAND_CENTRAL),
    "WW": (1.2, 7.7, 4.1),
}
# ---------------------------------------------------------------------------
# Fig 2 (f9) packs a two-column factored legend into the top-right corner, so
# it runs a size below the rcParams legend default to fit.
F9_LEGEND_FS = 7.5
LEGEND_KW = dict(frameon=True, framealpha=0.6, facecolor="white", edgecolor="0.7")

# axes.labelsize at the print style (base_fontsize 10 -> 11). stacked_ylabel
# quotes its dx gap against this, so the gap can scale with the active style.
_PRINT_LABEL_SIZE = 11.0

# ANNOT_FS and F9_LEGEND_FS are absolute point sizes, so unlike the structural
# text (which rides on rcParams) they would stay at print size while the canvas
# grew on a slide style. Keep the print baseline and rescale in _set_theme.
_PRINT_BASE_FS = 10.0
_PRINT_ANNOT_FS = ANNOT_FS
_PRINT_F9_LEGEND_FS = F9_LEGEND_FS

# Theme-aware ink (_ink), canvas scale (_figsize) and per-style font/line
# sizing all live in the shared package now, so this script and the
# EFT_Operator_Scan figure set move from print to projection identically.
# See helpers.plot_style: theme_ink / scaled_figsize / STYLE_SIZING.
# The only thing kept local is fig1's below-axes legend geometry, which is
# specific to that figure's stacked-panel layout.
_THEME = {"legend_rect": 0.10, "legend_y": 0.04}
#: True on every non-paper (slide) style; see _slide_wide.
_SLIDE_STYLE = False


def _set_theme(style_norm):
    """Set the per-figure layout knobs that depend on the active style."""
    global ANNOT_FS, F9_LEGEND_FS, _SLIDE_STYLE
    _SLIDE_STYLE = style_norm != "paper"

    # Rescale the absolute point sizes from the print baseline, so in-panel
    # annotations and f9's packed legend grow with the canvas instead of
    # shrinking away on a slide.
    _fs_scale = style_sizing(style_norm)["fontsize"] / _PRINT_BASE_FS
    ANNOT_FS = round(_PRINT_ANNOT_FS * _fs_scale, 1)
    F9_LEGEND_FS = round(_PRINT_F9_LEGEND_FS * _fs_scale, 1)

    # The non-paper styles set a larger base font, so fig1's bottom legend
    # grows and needs a deeper reserved strip -- at print values it rides up
    # over the "E [GeV]" x-axis label.
    _THEME.update(
        legend_rect=0.10 if style_norm == "paper" else 0.17,
        legend_y=0.04 if style_norm == "paper" else 0.012,
    )
    # LEGEND_KW is splatted into every ax.legend()/fig.legend() call in this
    # file, so retheme it in place rather than at each call site.
    LEGEND_KW.clear()
    LEGEND_KW.update(theme_legend_kw())


def stacked_ylabel(target, main, units, *, units_frac=0.78, dx=0.065,
                   x=0.008, y=0.5):
    """Rotated y-axis label with `units` in a smaller font beside `main`.

    A single Text can't mix font sizes, so we draw two rotated text artists,
    each individually centred at `y` so they share the same vertical midline
    (main outer at `x`, units inner at `x + dx`). `target` may be an Axes
    (coords in transAxes) or a Figure (coords in transFigure).
    """
    import matplotlib.pyplot as plt
    import matplotlib.figure as mfig

    base = plt.rcParams["axes.labelsize"]
    if isinstance(base, str):
        base = plt.rcParams["font.size"]
    # dx is quoted at the PRINT label size. These two texts are rotated 90deg,
    # so their on-screen WIDTH is the font height: the gap between them has to
    # grow with the font or the units run straight through the main label on
    # the larger slide styles. Scale the gap, and move the outer (main) label
    # out by the same amount so the inner edge stays clear of the tick labels.
    _dx_scaled = dx * base / _PRINT_LABEL_SIZE
    x -= _dx_scaled - dx
    dx = _dx_scaled
    if isinstance(target, mfig.Figure):
        add, tr = target.text, target.transFigure
    else:  # Axes
        add, tr = target.text, target.transAxes
    common = dict(rotation=90, rotation_mode="anchor", ha="center",
                  va="center", transform=tr, clip_on=False)
    t_main = add(x, y, main, size=base, **common)
    t_units = add(x + dx, y, units, size=base * units_frac, **common)
    return t_main, t_units


def _style_legend(leg, *, zorder=None):
    if leg is None:
        return None
    frame = leg.get_frame()
    frame.set_facecolor(LEGEND_KW["facecolor"])
    frame.set_edgecolor(LEGEND_KW["edgecolor"])
    frame.set_alpha(LEGEND_KW["framealpha"])
    if zorder is not None:
        leg.set_zorder(zorder)
    return leg


def _fit_bottom_legend(fig, leg, *, rect_left=0.0, gap=0.018, **tight_kw):
    """Reserve exactly as much bottom strip as `leg` actually needs.

    A hand-picked `rect` bottom has to be tuned per style: the legend's rows are
    set in points, so the strip it needs grows faster than the canvas does, and
    one constant either clips the legend on a slide or strands it a long way
    below the axes in print. Measure the legend instead — draw once to get a
    renderer, convert its height to a figure fraction, and hand tight_layout a
    rect bottom of exactly that plus `gap`. The legend is then re-anchored so
    it sits `gap` above the canvas floor, i.e. snug under the x-axis label in
    every style.
    """
    fig.canvas.draw()
    h = (leg.get_window_extent(fig.canvas.get_renderer()).height
         / (fig.get_figheight() * fig.dpi))
    fig.tight_layout(rect=(rect_left, min(h + 2 * gap, 0.45), 1.0, 1.0), **tight_kw)
    leg.set_bbox_to_anchor((0.5, gap), transform=fig.transFigure)
    return leg


def _mathtext_sci_axis(ax, axis="y"):
    """Use mathtext scientific offsets, e.g. x10^{-4}, instead of plain 1e-4."""
    import matplotlib.ticker as mticker

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    getattr(ax, f"{axis}axis").set_major_formatter(formatter)


def _use_mathtext_scalar_offsets(fig):
    """Make all plain scalar axis offsets in this figure use math text."""
    import matplotlib.ticker as mticker

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, mticker.ScalarFormatter):
                formatter.set_useMathText(True)


#: Energy bands highlighted on the disk-comparison figure. Both are the repo's
#: existing conventions, not new choices: 10-40 GeV is the band that defines the
#: "20 GeV halo" (check_null_and_hemisphere.py / compare_quadrants.py PEAK_LO,
#: PEAK_HI), and 1-4 GeV is the few-GeV GCE / MSP-cutoff region shaded in
#: plot_halo_vs_gce.py.
GCE_BAND = (1.0, 4.0)
HALO_BAND = (10.0, 40.0)
#: Band tints. Cool hues, deliberately: the data in this figure are warm (pink
#: and yellow), so a cool wash separates "context" from "measurement" instead of
#: competing with it. Saturated colours at low alpha, NOT theme_ink greys --
#: an ink grey inverts with the theme and lands under the background on a dark
#: slide, which is exactly how the science-band ended up invisible earlier.
GCE_BAND_COLOR = "#D2803C"    # muted orange
HALO_BAND_COLOR = "#7B5EA7"   # muted purple

#: House line weights, keyed to f9 (fig2 / _likelihood_profiles_panel) -- the
#: reference one-column figure. Every panel drew its own weight before, so a
#: "normal series" ranged 1.2-1.9 across the set and the same kind of curve
#: looked heavier in one figure than another.
SERIES_LW = 1.65   # an ordinary data series
EMPH_LW = 2.2      # the one curve a panel is about (baseline / measured halo)
GUIDE_LW = 0.75    # zero lines and other non-data furniture
SERIES_MS = 4.6    # marker size on a data series


def _signal_band(ax, band=HALO_BAND, *, zorder=0):
    """Shade a signal band the way f9 does. f9 IS the house reference.

    Neutral grey, alpha 0.35, no edge lines and no caption inside the axes.
    Several panels had drifted to a coloured span with dashed edges and a
    "signal band" label written into the plot area, so the same 10-40 GeV
    interval read as a different object in every figure. Centralised here so
    they cannot drift apart again.

    The dashed edges those panels carried were there for greyscale print,
    where a 20%-alpha tint washes out. f9's answer is a stronger fill (0.35,
    not 0.20-0.22) in neutral grey, which survives the conversion on its own
    without spending two more line styles on furniture.

    _ink grey, not a fixed tint: verified visible in BOTH styles -- the band
    reads as a clear block on the dark slide as well as on the page.
    """
    ax.axvspan(band[0], band[1], color=_ink("0.85"), alpha=0.35, lw=0,
               zorder=zorder)

FIG8_PANELS = [
    dict(
        fig=5,
        title=r"NFW-$\rho^{2.5}$ (GCE)",
        label="nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        ylim=(-1.0e-5, 2.0e-5),
        yticks=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * 1e-5,
    ),
    dict(
        fig=6,
        title=r"NFW-$\rho^{2}$",
        label="nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
        ylim=(-2.0e-5, 4.25e-5),
        yticks=np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1e-5,
    ),
    dict(
        fig=7,
        title=r"NFW-$\rho^{1}$",
        label="nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno",
        ylim=(-1.0e-4, 2.4e-4),
        yticks=np.array([-1.0, 0.0, 1.0, 2.0]) * 1e-4,
    ),
]

# Via Lactea II NFW (decay D-factor; matches decay_fit.py)
_RS, _R0, _RHOS, _GAMMA = 21.0, 8.0, 8.1e6, 1.0
_MSUN_KPC3_TO_GEV_CM3 = 3.817e-8
_KPC_TO_CM = 3.0857e21


# --------------------------------------------------------------------------- #
#  data helpers
# --------------------------------------------------------------------------- #
def pixelwise_dir(fig_n: int, with_disk: bool) -> str:
    # pixelwise full-MCMC outputs (post-reorg: mcmc/fit_results/pixelwise_mcmc/)
    suf = "_w_disk" if with_disk else ""
    return str(MCMC / "fit_results" / "pixelwise_mcmc" / f"pixelwise_mcmc_results_fig{fig_n}{suf}")


def cellwise_dir(fig_n: int) -> str:
    # cellwise (Totani replication) MCMC now lives in Totani_paper_check/mcmc/
    tagged = PC / "mcmc" / f"mcmc_results_fig{fig_n}_10deg"
    default = PC / "mcmc" / f"mcmc_results_fig{fig_n}"
    return str(tagged if tagged.exists() else default)


def halo(fig_n: int, with_disk: bool = False):
    return load_halo_spectrum(pixelwise_dir(fig_n, with_disk))


def _clean_yerr(lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    lo = np.where(np.isfinite(lo) & (lo >= 0.0), lo, 0.0)
    hi = np.where(np.isfinite(hi) & (hi >= 0.0), hi, 0.0)
    return np.vstack([lo, hi])


def _plot_halo_points(ax, h, *, color, marker, label, mfc, x_shift=1.0, zorder=5):
    E = np.asarray(h.E_bins_GeV, float) * float(x_shift)
    y = np.asarray(h.phi, float)
    yerr = _clean_yerr(h.phi_err_lo, h.phi_err_hi)
    finite = np.isfinite(E) & np.isfinite(y) & (E > 0)
    if not np.any(finite):
        return
    ax.errorbar(
        E[finite],
        y[finite],
        yerr=yerr[:, finite],
        fmt=marker,
        color=color,
        ecolor=color,
        ms=4.5,
        mfc=mfc,
        mec=color,
        mew=0.9,
        elinewidth=0.85,
        capsize=2.0,
        linestyle="none",
        label=label,
        zorder=zorder,
    )


def _trapz(y, x):
    return getattr(np, "trapezoid", np.trapz)(y, x)


def _D_pole_GeV_cm2():
    s = np.linspace(0.0, 50.0 * 402.0, 400000)
    r = np.sqrt(_R0 ** 2 + s ** 2)
    rho = _RHOS * (r / _RS) ** (-_GAMMA) * (1.0 + r / _RS) ** (_GAMMA - 3.0)
    return _trapz(rho, s) * _MSUN_KPC3_TO_GEV_CM3 * _KPC_TO_CM


def fit_template(h, m_ann, channel):
    """Weighted-LS normalisation of a PPPC E^2 dN/dE template to the halo phi.
    Returns (A, model_on_all_bins, chi2, ndof, mask)."""
    E = np.asarray(h.E_bins_GeV, float)
    phi = np.asarray(h.phi, float)
    err = 0.5 * (np.asarray(h.phi_err_hi, float) + np.asarray(h.phi_err_lo, float))
    mask = h.positive_mask & h.finite_mask & (err > 0)
    shape = pppc_energy_flux_template(E, m_ann, channel=channel, primary="gamma",
                                      table_path=str(PPPC_TABLE), normalise=False)
    shape = np.asarray(shape, float)
    if not np.any(shape[mask] > 0):
        return None
    w = 1.0 / err[mask] ** 2
    A = float(np.sum(w * shape[mask] * phi[mask]) / np.sum(w * shape[mask] ** 2))
    if not np.isfinite(A) or A <= 0:
        return None
    chi2 = float(np.sum(((A * shape[mask] - phi[mask]) / err[mask]) ** 2))
    return A, A * shape, chi2, int(mask.sum()) - 1, mask


def best_fit(h, channel, masses, decay=False):
    best = None
    for m in masses:
        m_ann = m / 2.0 if decay else m
        r = fit_template(h, m_ann, channel)
        if r is None:
            continue
        if best is None or r[2] < best[0]:
            best = (r[2], m, r)
    if best is None:
        return None
    chi2, m, (A, model, _c, ndof, mask) = best
    return dict(m_chi=m, norm=A, model=model, chi2=chi2, ndof=ndof, mask=mask)


# --------------------------------------------------------------------------- #
#  Fig 1 : Totani Fig. 8, pixelwise, disk in/out, three profiles
# --------------------------------------------------------------------------- #
def _pole_flux_panel(ax, panel, colors, fg, *, title_y=0.84, title_above=False):
    """One profile's pole spectrum: cellwise vs pixelwise, disk in/out.

    Shared by the stacked fig1 and the side-by-side fig14 so the two layouts
    cannot drift apart -- a change to the series, shifts or guides shows up in
    both.

    title_above puts the profile label over the axes instead of inside it. The
    wide panels are short and the rho^1 spectrum peaks in the top-left corner,
    exactly where the in-panel label sits, so there is no interior anchor that
    stays clear in all three panels; above the axes always is.
    """
    import matplotlib.ticker as mticker

    fn = int(panel["fig"])
    cell = load_halo_spectrum(cellwise_dir(fn), nfw_label=panel["label"], central_stat="f_p50")
    he = load_halo_spectrum(pixelwise_dir(fn, False), nfw_label=panel["label"], central_stat="f_p50")
    hi = load_halo_spectrum(pixelwise_dir(fn, True), nfw_label=panel["label"], central_stat="f_p50")
    ax.axhline(0.0, color=_ink("0.5"), lw=0.7, ls=":")
    ax.axvline(20.0, color=_ink("0.8"), lw=0.8, ls=":")
    _plot_halo_points(
        ax, cell, color=fg, marker="o", label="Cellwise, disk excl.",
        mfc=_ink("white"), x_shift=0.94, zorder=7,
    )
    _plot_halo_points(
        ax, he, color=colors[1], marker="s", label="Pixelwise, disk excl.",
        mfc=colors[1], x_shift=1.00, zorder=6,
    )
    _plot_halo_points(
        ax, hi, color=colors[3], marker="^", label="Pixelwise, disk incl.",
        mfc=colors[3], x_shift=1.064, zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(*panel["ylim"])
    ax.set_yticks(panel["yticks"])
    if title_above:
        # Pinned y so all three titles share one line: left to itself,
        # matplotlib lifts a title clear of the y-axis offset text ("x10^-5")
        # whenever the two would collide, which only the longest label --
        # "NFW-rho^2.5 (GCE)" -- triggers, leaving that panel's title floating
        # above the other two.
        #
        # loc="right" is what makes the shared line safe. The offset sits at the
        # top LEFT, so a centred title grows toward it and a long one lands on
        # top of it; anchored right, every title grows away from the offset and
        # none can reach it however long the label gets.
        ax.set_title(panel["title"], loc="right", y=1.0, pad=ANNOT_FS * 0.4)
    else:
        ax.text(0.04, title_y, panel["title"], transform=ax.transAxes)
    # A narrow panel makes matplotlib's log locator drop the intermediate
    # decades, leaving only 10^0 and 10^3 on the wide layout. Pin all four.
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
    _mathtext_sci_axis(ax, "y")
    if title_above:
        # The "x10^-5" offset defaults to the tick size, which is more weight
        # than a unit multiplier needs and is what crowds the titles in the
        # wide layout. Drop it to the annotation size.
        ax.yaxis.get_offset_text().set_fontsize(ANNOT_FS)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(alpha=0.13, which="both")


def fig1(outdir):
    import matplotlib.pyplot as plt

    colors = get_cmap_colors(cmap_name="plasma", n=5, start=0.15, end=0.65)
    fg = plt.rcParams.get("text.color", "black")
    fig, axes = plt.subplots(3, 1, figsize=_figsize(COL_W, 7.1), sharex=True)
    for ax, panel in zip(axes, FIG8_PANELS):
        _pole_flux_panel(ax, panel, colors, fg)
    axes[-1].set_xlabel(r"$E$ [GeV]")
    fig.text(
        0.008,
        0.58,
        r"$E^2\,\mathrm{d}N/\mathrm{d}E$ at galactic poles [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        rotation=90,
        va="center",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    # Two legend rows never needed the flat 0.10/0.17 strip the other bottom-
    # legend figures reserve, so the legend sat marooned below the x-label —
    # worst on the slide styles, where the strip is deepest but the legend is
    # only ~11% taller. Size the strip to the legend itself instead.
    leg = _style_legend(fig.legend(handles, labels, loc="lower center",
                                   bbox_to_anchor=(0.5, _THEME["legend_y"]),
                                   columnspacing=1.2, ncols=2,
                                   handletextpad=0.4, **LEGEND_KW))
    _fit_bottom_legend(fig, leg, rect_left=0.055, pad=0.6, w_pad=1.4)
    return _save(fig, outdir, "f8_pole_flux_profiles")


# --------------------------------------------------------------------------- #
#  Fig 2 : Totani Fig. 9, per-bin halo Delta lnL, pixelwise disk-excluded
# --------------------------------------------------------------------------- #
def _parse_loglike_txt(path: Path) -> float:
    if not Path(path).exists():
        return np.nan
    with Path(path).open("r") as f:
        for line in f:
            m = re.search(r"Log-likelihood \(ML\):\s*([+-]?\S+)", line)
            if m:
                return float(m.group(1))
    return np.nan


def _loglike_by_bin(mcmc_dir):
    mcmc_dir = Path(mcmc_dir)
    E, lnL = [], []
    for k in range(13):
        npz = mcmc_dir / f"mcmc_results_k{k:02d}.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        val = _parse_loglike_txt(mcmc_dir / f"mcmc_results_k{k:02d}.txt")
        if not np.isfinite(val):
            continue
        E.append(float(z["Ectr_mev"]) / 1000.0)
        lnL.append(val)
    return np.array(E), np.array(lnL)


def _nfw_stats_by_bin(mcmc_dir, nfw_label):
    mcmc_dir = Path(mcmc_dir)
    out = {key: [] for key in ("E", "ML", "p16", "p84", "p05", "p95")}
    for k in range(13):
        txt = mcmc_dir / f"mcmc_results_k{k:02d}.txt"
        npz = mcmc_dir / f"mcmc_results_k{k:02d}.npz"
        if not txt.exists() or not npz.exists():
            continue
        with np.load(npz, allow_pickle=True) as z:
            E = float(z["Ectr_mev"]) / 1000.0
        row = None
        in_table = False
        header_cols = []
        for line in txt.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("Component") and "ML" in stripped:
                header_cols = stripped.split()[1:]
                in_table = True
                continue
            if not in_table:
                continue
            if stripped.startswith("-"):
                continue
            if not stripped:
                in_table = False
                continue
            parts = stripped.split()
            n_num = len(header_cols)
            if len(parts) < n_num + 1:
                continue
            label = " ".join(parts[: len(parts) - n_num])
            if nfw_label not in label and "nfw" not in label.lower():
                continue
            try:
                row = {col: float(val) for col, val in zip(header_cols, parts[len(parts) - n_num :])}
            except ValueError:
                row = None
            break
        if row is None:
            continue
        out["E"].append(E)
        for key in ("ML", "p16", "p84", "p05", "p95"):
            out[key].append(float(row.get(key, np.nan)))
    return {key: np.asarray(vals, float) for key, vals in out.items()}


def _fig9_boundary(dlnl_best, nfw_stats):
    ml = np.asarray(nfw_stats["ML"], float)
    p16 = np.asarray(nfw_stats["p16"], float)
    p84 = np.asarray(nfw_stats["p84"], float)
    sigma = 0.5 * (p84 - p16)
    lo = np.asarray(nfw_stats["p05"], float)
    hi = np.asarray(nfw_stats["p95"], float)
    lo = np.where(np.isfinite(lo), lo, ml - 2.0 * sigma)
    hi = np.where(np.isfinite(hi), hi, ml + 2.0 * sigma)
    boundary = np.full_like(np.asarray(dlnl_best, float), np.nan, dtype=float)
    for i, dL in enumerate(dlnl_best):
        if not (np.isfinite(dL) and np.isfinite(ml[i]) and np.isfinite(sigma[i]) and sigma[i] > 0):
            boundary[i] = dL
            continue
        vals = []
        for edge in (lo[i], hi[i]):
            if np.isfinite(edge):
                vals.append(float(dL - (edge - ml[i]) ** 2 / (2.0 * sigma[i] ** 2)))
        boundary[i] = min(vals) if vals else dL
    return boundary


def _delta_lnl_overlay_series(fig_n, *, pixelwise: bool, with_disk: bool = False):
    """Per-bin delta-lnL of the halo fit against the no-halo (fig4) fit.

    with_disk selects the disk-included fit variant, and it MUST move the
    no-halo baseline with it: delta-lnL is only meaningful between two fits
    over the same pixels, so pairing a disk-included halo fit against a
    disk-excluded baseline would silently compare different data sets.
    Cellwise has no disk-included variant, so the flag applies to pixelwise.
    """
    if with_disk and not pixelwise:
        raise ValueError("with_disk is only available for the pixelwise fits")
    halo_dir = pixelwise_dir(fig_n, with_disk) if pixelwise else cellwise_dir(fig_n)
    base_dir = pixelwise_dir(4, with_disk) if pixelwise else cellwise_dir(4)
    E_halo, lnL_halo = _loglike_by_bin(halo_dir)
    E_base, lnL_base = _loglike_by_bin(base_dir)
    if E_halo.size == 0 or E_base.size == 0:
        missing = [d for d, E in ((halo_dir, E_halo), (base_dir, E_base)) if E.size == 0]
        raise FileNotFoundError(
            "no usable per-bin log-likelihoods in: "
            + ", ".join(str(d) for d in missing)
            + "\n_loglike_by_bin needs BOTH mcmc_results_k*.npz AND the matching "
            "mcmc_results_k*.txt. The .txt carries the full-precision "
            "'Log-likelihood (ML)'; the npz copy (loglike_ml) is float32, whose "
            "ulp at ~1.8e7 is 2.0 -- far too coarse for a delta-lnL of order tens. "
            "Re-download the folder from Drive: see restore_mcmc_from_drive.sh."
        )

    base = {round(float(E), 8): float(v) for E, v in zip(E_base, lnL_base)}
    nfw_label = next(p["label"] for p in FIG8_PANELS if int(p["fig"]) == int(fig_n))
    stats = _nfw_stats_by_bin(halo_dir, nfw_label)
    stat_by_e = {
        round(float(E), 8): {key: np.asarray(stats[key], float)[i] for key in ("ML", "p16", "p84", "p05", "p95")}
        for i, E in enumerate(stats["E"])
    }
    E, dlnl, rows = [], [], []
    for E_i, lnL_i in zip(E_halo, lnL_halo):
        key = round(float(E_i), 8)
        if key not in base:
            continue
        val = float(lnL_i - base[key])
        if np.isfinite(val) and abs(val) < 1.0e6:
            E.append(float(E_i))
            dlnl.append(val)
            rows.append(stat_by_e.get(key, {k: np.nan for k in ("ML", "p16", "p84", "p05", "p95")}))
    dlnl = np.array(dlnl)
    stats_aligned = {key: np.array([row[key] for row in rows], float) for key in ("ML", "p16", "p84", "p05", "p95")}
    boundary = _fig9_boundary(dlnl, stats_aligned) if len(rows) else np.array([])
    return np.array(E), dlnl, boundary


def _likelihood_profiles_panel(ax, *, title=None):
    """Per-bin halo delta-lnL, cellwise vs pixelwise, for all three profiles.

    Split out of fig2 so fig13 can draw the same panel beside the angular
    profile without the two renderings drifting apart.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    cols = get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)
    styles = [
        (5, r"$\rho^{2.5}$", "^", cols[0]),
        (6, r"$\rho^{2}$", "o", cols[2]),
        (7, r"$\rho^{1}$", "s", cols[4]),
    ]
    finite_vals = []
    cellwise_handles = []
    pixelwise_handles = []
    profile_labels = []
    for fn, lab, mk, c in styles:
        E_t, dl_t, bd_t = _delta_lnl_overlay_series(fn, pixelwise=False)
        E_p, dl_p, bd_p = _delta_lnl_overlay_series(fn, pixelwise=True)
        ax.plot(
            E_t,
            dl_t,
            color=c,
            lw=1,
            ls="-",
            marker=mk,
            ms=4.6,
            mfc=_ink("white"),
            mec=c,
            mew=0.9,
            label="_nolegend_",
        )
        ok_t = np.isfinite(E_t) & np.isfinite(dl_t) & np.isfinite(bd_t)
        if np.any(ok_t):
            ax.fill_between(
                E_t[ok_t],
                dl_t[ok_t],
                bd_t[ok_t],
                color=c,
                alpha=0.13,
                linewidth=0.0,
                zorder=1,
                label="_nolegend_",
            )
        ax.plot(
            E_p,
            dl_p,
            color=c,
            lw=1,
            ls="--",
            marker=mk,
            ms=4.6,
            mfc=c,
            mec=c,
            mew=0.7,
            label="_nolegend_",
        )
        ok_p = np.isfinite(E_p) & np.isfinite(dl_p) & np.isfinite(bd_p)
        if np.any(ok_p):
            ax.fill_between(
                E_p[ok_p],
                dl_p[ok_p],
                bd_p[ok_p],
                color=c,
                alpha=0.13,
                linewidth=0.0,
                zorder=1,
                label="_nolegend_",
            )
        for values in (dl_t, dl_p, bd_t, bd_p):
            values = np.asarray(values, float)
            values = values[np.isfinite(values)]
            if values.size:
                finite_vals.append(values)
        cellwise_handles.append(
            Line2D([0], [0], color=c, lw=1.65, ls="-", marker=mk, ms=4.6,
                   mfc=_ink("white"), mec=c, mew=0.9)
        )
        pixelwise_handles.append(
            Line2D([0], [0], color=c, lw=1.65, ls="--", marker=mk, ms=4.6,
                   mfc=c, mec=c, mew=0.7)
        )
        profile_labels.append(lab)
    _signal_band(ax)
    ax.axhline(0.0, color=_ink("0.45"), lw=GUIDE_LW, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(r"$\ln(L)-\ln(L_{\rm no\text{-}halo})$")
    if finite_vals:
        y = np.concatenate(finite_vals)
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.08 * (ymax - ymin)
            ax.set_ylim(min(ymin - pad, -0.5), ymax + pad)
    # Factored legend in the top-right corner (the high-energy region is empty,
    # so it never covers the 20 GeV peak). Two side-by-side sub-legends:
    #   left  = NFW profile -> colour
    #   right = fit method  -> linestyle + marker fill
    colour_handles = [Line2D([0], [0], color=c, lw=2.0, marker=mk, ms=4.6,
                             mfc=c, mec=c)
                      for (_, _, mk, c) in styles]
    colour_labels = [lab for (_, lab, _, _) in styles]
    method_handles = [
        Line2D([0], [0], color=_ink("0.35"), lw=1.65, ls="-", marker="o", ms=4.6,
               mfc=_ink("white"), mec=_ink("0.35"), mew=0.9),
        Line2D([0], [0], color=_ink("0.35"), lw=1.65, ls="--", marker="o", ms=4.6,
               mfc=_ink("0.35"), mec=_ink("0.35"), mew=0.7),
    ]
    method_labels = ["Cellwise", "Pixelwise"]

    handles = colour_handles + method_handles
    labels = colour_labels + method_labels
    _style_legend(ax.legend(handles, labels, loc="upper right",
                            bbox_to_anchor=(0.995, 0.985), fontsize=F9_LEGEND_FS,
                            handlelength=2.0, borderpad=0.4, labelspacing=0.3,
                            **LEGEND_KW))

    ax.grid(alpha=0.14, which="both")
    if title:
        ax.set_title(title)


def fig2(outdir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=_figsize(COL_W, 2.8))
    _likelihood_profiles_panel(ax)
    fig.tight_layout()
    return _save(fig, outdir, "f9_likelihood_profiles")


# --------------------------------------------------------------------------- #
#  Fig 3 / 6 : spectral overlay + residual
# --------------------------------------------------------------------------- #
def _overlay_with_residual(
    outdir,
    stem,
    h,
    channels,
    decay,
    masses,
    title,
    *,
    legend_loc="lower center",
    legend_anchor=(0.5, 0.02),
    legend_ncol=None,
):
    import matplotlib.pyplot as plt
    cols = get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.76)
    chan_col = {"bb": cols[0], "WW": cols[2], "tautau": cols[4]}
    chan_marker = {"bb": "s", "WW": "o", "tautau": "^"}
    E = h.E_bins_GeV
    err = _clean_yerr(h.phi_err_lo, h.phi_err_hi)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=_figsize(COL_W, 3.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    ax.axhline(0.0, color=_ink("0.5"), lw=0.7, ls=":")
    data_mask = h.finite_mask & np.isfinite(E) & (E > 0)
    ax.errorbar(
        E[data_mask],
        h.phi[data_mask],
        yerr=err[:, data_mask],
        fmt="o",
        color=_ink("k"),
        ecolor=_ink("k"),
        ms=4.0,
        mfc=_ink("white"),
        mec=_ink("k"),
        mew=1.0,
        elinewidth=0.85,
        capsize=2.0,
        linestyle="none",
        zorder=7,
        label="Pixelwise halo",
    )
    fits = {}
    shifts = np.geomspace(0.94, 1.06, max(len(channels), 1))
    if len(channels) == 1:
        shifts = np.array([1.0])
    residual_values = []
    for ch, x_shift in zip(channels, shifts):
        bf = best_fit(h, ch, masses, decay=decay)
        if bf is None:
            continue
        fits[ch] = bf
        lab = f"{CH_LABEL[ch]}, $m_\\chi$={bf['m_chi']:.0f} GeV"
        m = bf["mask"]
        ax.errorbar(
            E[m] * x_shift,
            bf["model"][m],
            yerr=err[:, m],
            fmt=chan_marker.get(ch, "o"),
            color=chan_col[ch],
            ecolor=chan_col[ch],
            ms=3.8,
            mfc=chan_col[ch],
            mec=chan_col[ch],
            mew=0.6,
            elinewidth=0.8,
            capsize=1.8,
            linestyle="none",
            alpha=0.95,
            zorder=6,
            label=lab,
        )
        res = np.divide(
            h.phi[m] - bf["model"][m],
            h.phi[m],
            out=np.full(np.count_nonzero(m), np.nan, dtype=float),
            where=np.isfinite(h.phi[m]) & (h.phi[m] != 0.0),
        )
        res_err = np.divide(
            err[:, m],
            np.abs(h.phi[m])[None, :],
            out=np.full((2, np.count_nonzero(m)), np.nan, dtype=float),
            where=np.isfinite(h.phi[m][None, :]) & (h.phi[m][None, :] != 0.0),
        )
        residual_values.append(res[np.isfinite(res)])
        axr.errorbar(
            E[m] * x_shift,
            res,
            yerr=res_err,
            marker=chan_marker.get(ch, "o"),
            ls="none",
            ms=3.4,
            color=chan_col[ch],
            ecolor=chan_col[ch],
            mfc=chan_col[ch],
            mec=chan_col[ch],
            mew=0.6,
            elinewidth=0.75,
            capsize=1.6,
            alpha=0.95,
            label=CH_LABEL[ch],
        )
    axr.axhline(0.0, color=_ink("0.5"), lw=0.8)
    for lev in (-1, 1):
        axr.axhline(lev, color=_ink("0.8"), lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_xlim(1.0, 1000.0)
    _mathtext_sci_axis(ax, "y")
    ax.set_ylabel(" ")  # reserve the gutter; real label drawn stacked below
    main_ylabel_x = -0.17
    main_ylabel_dx = 0.065
    stacked_ylabel(ax, YLAB_MAIN, YLAB_UNITS, x=main_ylabel_x, dx=main_ylabel_dx, y=0.5)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.text(0.04, 0.9, title, transform=ax.transAxes, va="top")
    ax.grid(alpha=0.13, which="both")
    axr.set_xlabel(XLAB)
    axr.set_ylabel(r"$\frac{d-p}{d}$")
    axr.yaxis.set_label_coords(main_ylabel_x + 0.8 * main_ylabel_dx, 0.5)
    if residual_values:
        rv = np.concatenate([v for v in residual_values if v.size])
        if rv.size:
            ymin = float(np.nanmin(rv))
            ymax = float(np.nanmax(rv))
            pad = 0.15 * max(ymax - ymin, 0.2)
            axr.set_ylim(max(ymin - pad, -3.0), min(ymax + pad, 3.0))
    axr.grid(alpha=0.13, which="both")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.92, bottom=0.12)
    # Legend below the whole figure: never overlaps the data or the y-labels
    # (savefig bbox_inches="tight" captures it).
    _style_legend(fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=legend_ncol or 2, columnspacing=1.3, handletextpad=0.4,
        borderpad=0.4, handlelength=1.2, **LEGEND_KW,
    ))
    return _save(fig, outdir, stem), fits


def fig3(outdir):
    h = halo(6, with_disk=False)
    masses = _saved_scan_mass_grid(
        "pixelwise_global_fig6", ["bb", "WW"],
        fallback=np.geomspace(100.0, 4000.0, 160),
    )
    out, fits = _overlay_with_residual(
        outdir, "f_annih_bb_WW", h, ["bb", "WW"], decay=False, masses=masses,
        title=r"NFW-$\rho^{2}$",
        legend_loc="lower center",
        legend_anchor=(0.5, 0.02),
        legend_ncol=2,
    )
    for ch, bf in fits.items():
        print(f"    annih {ch}: m_chi={bf['m_chi']:.0f} GeV  chi2/ndof={bf['chi2']/bf['ndof']:.2f}")
    return out


def fig6(outdir):
    h = halo(7, with_disk=False)
    masses = np.geomspace(100.0, 6000.0, 160)
    Dpole = _D_pole_GeV_cm2()
    out, fits = _overlay_with_residual(
        outdir, "f_decay_rho1", h, ["tautau", "bb", "WW"], decay=True, masses=masses,
        title=r"NFW-$\rho^{1}$",
        legend_loc="lower right",
        legend_anchor=(0.98, 0.02),
        legend_ncol=2,
    )
    for ch, bf in fits.items():
        tau = 1000.0 * Dpole / (4.0 * np.pi * bf["m_chi"] * bf["norm"])
        print(f"    decay {ch}: m_chi={bf['m_chi']:.0f} GeV  tau={tau:.2e} s  "
              f"chi2/ndof={bf['chi2']/bf['ndof']:.2f}")
    return out


# --------------------------------------------------------------------------- #
#  Fig 4 : dSph tension, WW | bb, full width
# --------------------------------------------------------------------------- #
def _saved_scan_mass_grid(fit, channels, *, fallback):
    grids = []
    for channel in channels:
        path = PPPC / fit / channel / "mass_scan_results.npz"
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        if "masses_GeV" in z:
            grids.append(np.asarray(z["masses_GeV"], float))
    if not grids:
        return np.asarray(fallback, float)
    first = grids[0]
    if all(g.shape == first.shape and np.allclose(g, first, rtol=0.0, atol=1.0e-6) for g in grids[1:]):
        return first
    return np.asarray(fallback, float)


def _load_signal(fit, channel):
    path = PPPC / fit / channel / "mass_scan_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing PPPC mass scan for {fit}/{channel}: {path}")
    z = np.load(path, allow_pickle=True)
    d = {key: z[key] for key in z.files}
    m = np.asarray(d["masses_GeV"], float)
    lnL = np.asarray(d["total_lnL"], float)
    if np.any(np.isfinite(lnL)):
        best = int(np.nanargmax(lnL))
    else:
        best = int(np.nanargmin(np.asarray(d["spectral_chi2"], float)))
    d["masses_GeV"] = m
    d["sigmav_best"] = np.asarray(d["sigmav_best"], float)
    d["best_index"] = best
    return d


def _pixelwise_bb_bestfit_anchor():
    """Best-fit point from the validated pixelwise rho^2 bb mass scan."""
    try:
        d = _load_signal("pixelwise_global_fig6", "bb")
    except FileNotFoundError:
        return None
    best = int(d["best_index"])
    m = float(d["masses_GeV"][best])
    sv = float(d["sigmav_best"][best])
    lim = dsph_upper_limit(m, "bb", "hoof")
    tension = float(sv / lim) if np.isfinite(lim) and lim > 0 else np.nan
    return dict(m=m, sv=sv, tension=tension)


def _limit_styles():
    colors = get_cmap_colors(cmap_name="plasma", n=8, start=0.12, end=0.92)
    return {
        # Near-black on paper; _ink flips it to near-white so the operative
        # limit stays the highest-contrast curve on a dark slide too.
        "hoof": (r"$s$-wave dSph (operative)", "-", _ink("0.15")),
        "boddy_pwave": (r"$p$-wave dSph", "--", colors[5]),
        "boddy_dwave": (r"$d$-wave dSph", ":", colors[3]),
        "boddy_somm": (r"Sommerfeld (unsaturated)", "-.", colors[6]),
    }


def _format_r(value):
    if not np.isfinite(value):
        return "--"
    if 0.1 <= abs(value) < 1000:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _dsph_tension_panel(ax, ch, *, fit, sig_col, limit_styles):
    """Draw one annihilation channel of the dwarf-tension comparison.

    Factored out of fig4 so the same panel can be drawn either as one half of
    the two-channel paper figure or as a standalone single-channel figure
    (fig9 / fig10) without the two drifting apart.
    """
    import matplotlib.ticker as mticker
    d = _load_signal(fit, ch)
    m = np.asarray(d["masses_GeV"], float)
    sv = np.asarray(d["sigmav_best"], float)
    best = int(d["best_index"])
    mb, svb = m[best], sv[best]
    mg = np.geomspace(max(150.0, m.min() * 0.9), m.max() * 1.05, 300)
    finite = np.isfinite(m) & np.isfinite(sv) & (sv > 0)
    ax.plot(m[finite], sv[finite], "-", color=sig_col[ch], lw=2.2, ms=4.0, zorder=6,
            label="Halo best fit")
    ax.plot([mb], [svb], "*", ms=13, color=sig_col[ch], mec=_ink("k"), mew=0.6, zorder=7)
    # Systematic normalisation range: the baseline s-wave tension R_s widens to
    # ~[R_BAND_LO, R_BAND_HI] under the foreground/J-factor budget (Sec.~systematics). Overlay
    # that range on the halo so its lower edge is seen approaching the operative
    # (s-wave) limit -- individual favourable variants dip below it (Fig. syst).
    band_lo, band_hi, _band_central = R_BAND.get(
        ch, (R_BAND_LO, R_BAND_HI, R_BAND_CENTRAL))
    _ls_mb = dsph_upper_limit(float(mb), ch, "hoof")
    if np.isfinite(_ls_mb) and _ls_mb > 0:
        Rs_base = svb / _ls_mb
        sv_lo = sv * (band_lo / Rs_base)
        sv_hi = sv * (band_hi / Rs_base)
        ax.fill_between(m[finite], sv_lo[finite], sv_hi[finite], color=sig_col[ch],
                        alpha=0.16, lw=0.0, zorder=4, label="halo (syst. range)")
    txt = []
    limlines = {}
    for src, (lab, ls, col) in limit_styles.items():
        lim = np.array([dsph_upper_limit(float(x), ch, src) for x in mg])
        limlines[src] = lim
        ok = np.isfinite(lim) & (lim > 0)
        # Demote the unsaturated Coulomb Sommerfeld curve to a faint reference:
        # it is the limiting case, not the physical (saturated) model.
        faint = (src == "boddy_somm")
        ax.plot(mg[ok], lim[ok], ls=ls, color=col,
                lw=1.1 if faint else 1.9, alpha=0.5 if faint else 1.0,
                zorder=4, label=lab)
        lb = dsph_upper_limit(float(mb), ch, src)
        if np.isfinite(lb) and lb > 0:
            tag = {
                "hoof": "s",
                "boddy_pwave": "p",
                "boddy_dwave": "d",
                "boddy_somm": "S",
            }[src]
            txt.append(rf"$\mathcal{{R}}_{{{tag}}}={_format_r(svb/lb)}$")

    # (a) Resonant/saturated Sommerfeld. The 1/v Coulomb enhancement plateaus
    # below a critical velocity, so the cold-dwarf boost is bounded. The
    # physical dwarf limit then lies between the unsaturated curve (fully
    # boosted dwarfs) and the s-wave curve (full saturation, where the common
    # enhancement cancels in the halo/dwarf ratio). Shade that range.
    l_s, l_S = limlines.get("hoof"), limlines.get("boddy_somm")
    if l_s is not None and l_S is not None:
        okb = np.isfinite(l_s) & np.isfinite(l_S) & (l_s > 0) & (l_S > 0)
        ax.fill_between(mg[okb], l_S[okb], l_s[okb],
                        color=limit_styles["boddy_somm"][2], alpha=0.16, lw=0,
                        zorder=1, label=r"Sommerfeld, saturated (range)")

    # (b) Thermal-relic present-day halo rate for each velocity law. Fixing the
    # coupling at freeze-out (<(v/c)^2>_fo = 6/x_f = 0.24) and evaluating it in
    # the halo (<(v/c)^2>_halo = 1e-6): s-wave stays at the thermal value, but
    # p- and d-wave fall many orders below the ~1e-24 needed to source the halo,
    # so a thermal p/d-wave particle cannot produce the signal.
    V2_HALO, V2_FO = 1.0e-6, 0.24
    relic_p = THERMAL_RELIC * (V2_HALO / V2_FO)
    relic_d = THERMAL_RELIC * (V2_HALO / V2_FO) ** 2
    ylo = 1.0e-27
    # s-wave thermal-relic reference stays on-axis; the p/d-wave thermal
    # predictions lie far below the plotted range, so mark them with a single
    # downward arrow from the halo and label the order-of-magnitude gaps.
    ax.axhline(THERMAL_RELIC, color=_ink("0.55"), lw=1.1, ls=(0, (1, 1)), zorder=2,
               label=r"$s$-wave relic (thermal)")
    dex_p = np.log10(svb / relic_p)
    dex_d = np.log10(svb / relic_d)
    ax.annotate("", xy=(mb, band_lo * ylo), xytext=(mb, svb * 0.08),
                arrowprops=dict(arrowstyle="-|>", color=_ink("0.35"), lw=1.3), zorder=8)
    ax.text(mb * 1.10, np.sqrt(svb * 0.08 * band_lo * ylo),
            "thermal relic:\n"
            rf"$p$-wave $\downarrow${dex_p:.0f} dex" "\n"
            rf"$d$-wave $\downarrow${dex_d:.0f} dex",
            fontsize=ANNOT_FS - 1, color=_ink("0.3"), va="center", ha="left", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc=_ink("white"), ec=_ink("0.7"), alpha=0.75))
    # call out the operative boundary on the s-wave curve
    _ys0 = dsph_upper_limit(float(mg[3]), ch, "hoof")
    if np.isfinite(_ys0) and _ys0 > 0:
        ax.annotate("operative limit\n($s$-wave)",
                    xy=(mg[3], _ys0), xytext=(mg[3] * 1.15, _ys0 * 5.0),
                    fontsize=ANNOT_FS - 1, color=_ink("0.1"), ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=_ink("0.1"), lw=0.7), zorder=10)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(200, 2000)
    xticks = [200, 300, 500, 700, 1000]
    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{x:g}" for x in xticks]))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    ax.set_ylim(ylo, 6e-20)
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.text(0.04, 0.95, CH_LABEL[ch], transform=ax.transAxes, va="top",
            zorder=20, bbox=dict(boxstyle="round,pad=0.15", fc=_ink("white"), ec="none", alpha=0.7))
    ax.text(0.03, 0.72,
            rf"$m_\chi^{{\rm best}}={mb:.0f}$ GeV" + "\n"
            + "  ".join(txt[:2]) + "\n" + "  ".join(txt[2:]),
            transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_FS - 1,
            zorder=20,
            bbox=dict(boxstyle="round", fc=_ink("white"), ec=_ink("0.7"), alpha=0.85))
    ax.grid(True, which="both", alpha=0.14)


def _dsph_tension_halo_colour():
    """The single colour the halo best-fit line/band/star share in both channels."""
    return get_cmap_colors(cmap_name="plasma", n=8, start=0.12, end=0.92)[4]


def fig4(outdir, fit="pixelwise_global_fig6"):
    import matplotlib.pyplot as plt
    halo_col = _dsph_tension_halo_colour()
    sig_col = {"WW": halo_col, "bb": halo_col}
    limit_styles = _limit_styles()
    fig, axes = plt.subplots(1, 2, figsize=_figsize(FULL_W, 4.0), sharey=True)
    for ax, ch in zip(axes, ["WW", "bb"]):
        _dsph_tension_panel(ax, ch, fit=fit, sig_col=sig_col, limit_styles=limit_styles)
    axes[0].set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]")
    handles, labels = axes[1].get_legend_handles_labels()
    _style_legend(fig.legend(handles, labels, loc="lower center", ncol=4,
                             bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
                             handletextpad=0.4, **LEGEND_KW), zorder=20)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return _save(fig, outdir, "f_dsph_tension_WW_bb")


def _dsph_tension_single(outdir, ch, fit="pixelwise_global_fig6"):
    """One channel of the dwarf-tension comparison, on its own.

    Same panel as its half of Fig 4 -- identical curves, limits, annotations and
    axis window -- so the standalone and the paired version cannot disagree.
    """
    import matplotlib.pyplot as plt
    halo_col = _dsph_tension_halo_colour()
    # COL_W in print: this is a one-column figure and FULL_W*0.55 (3.88 in) is
    # neither column width nor text width, so it emitted ~3.7 in against a
    # 3.4 in \columnwidth and LaTeX shrank it ~8% -- its type ending up smaller
    # than the rest of the one-column set for no reason anyone chose. Slide
    # geometry is left alone; it was tuned against a 16:9 frame, not a column.
    fig, ax = plt.subplots(figsize=_figsize(FULL_W * 0.55 if _SLIDE_STYLE else COL_W, 4.0))
    _dsph_tension_panel(ax, ch, fit=fit, sig_col={ch: halo_col},
                        limit_styles=_limit_styles())
    ax.set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]")
    handles, labels = ax.get_legend_handles_labels()
    # Eight entries over roughly half the paper width: two columns rather than
    # Fig 4's four, or the legend strip ends up taller than the panel above it.
    # In print the type also comes down a step: at COL_W two columns of eight
    # entries at full size are WIDER than the axes, and a legend wider than the
    # axes is billed back as figure width by bbox_inches="tight" -- which is
    # the column overrun this figure was just resized to fix.
    leg = _style_legend(fig.legend(handles, labels, loc="lower center", ncol=2,
                                   bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
                                   handletextpad=0.4,
                                   **({} if _SLIDE_STYLE else
                                      dict(fontsize=max(ANNOT_FS - 1.0, 6.0))),
                                   **LEGEND_KW), zorder=20)
    # Measured, not the hand-picked rect bottom of 0.21 this had. That constant
    # was chosen when the figure was FULL_W*0.55 with full-size legend type; at
    # COL_W with the smaller type the legend needs far less, and the fixed
    # reservation stranded it a long way below the x-label.
    _fit_bottom_legend(fig, leg)
    return _save(fig, outdir, f"f_dsph_tension_{ch}")


def fig9(outdir):
    """Dwarf tension, $W^+W^-$ channel only (Fig 4 left panel, standalone)."""
    return _dsph_tension_single(outdir, "WW")


def fig10(outdir):
    """Dwarf tension, $b\\bar b$ channel only (Fig 4 right panel, standalone)."""
    return _dsph_tension_single(outdir, "bb")


# --------------------------------------------------------------------------- #
#  Fig 5 : systematics suite impact (spectra + tension forest)
# --------------------------------------------------------------------------- #
def _systematics_table(name):
    for root in dict.fromkeys((SYST, SYST_LEGACY)):
        t = root / name / "mcmc_global_f_ml.txt"
        if t.exists():
            return t
    return None


def _fig8_rho2_baseline_halo():
    panel = FIG8_PANELS[1]
    h = load_halo_spectrum(
        pixelwise_dir(int(panel["fig"]), False),
        nfw_label=panel["label"],
        central_stat="f_p50",
    )
    return np.asarray(h.E_bins_GeV, float), np.asarray(h.phi, float)


def _raw_systematics_table_halo(name):
    t = _systematics_table(name)
    if t is None:
        return None
    a = np.genfromtxt(t, names=True, comments="#")
    return np.asarray(a["E_GeV"], float), np.asarray(a["nfw"], float)


_SYST_POLE_RESCALE = None


def _systematics_pole_rescale():
    global _SYST_POLE_RESCALE
    if _SYST_POLE_RESCALE is not None:
        return _SYST_POLE_RESCALE
    raw = _raw_systematics_table_halo("baseline_rho2")
    if raw is None:
        _SYST_POLE_RESCALE = 1.0
        return _SYST_POLE_RESCALE
    E_raw, y_raw = raw
    E_f8, y_f8 = _fig8_rho2_baseline_halo()
    y_ref = np.interp(E_raw, E_f8, y_f8, left=np.nan, right=np.nan)
    good = np.isfinite(y_ref) & np.isfinite(y_raw) & (np.abs(y_raw) > 0.0)
    if not np.any(good):
        _SYST_POLE_RESCALE = 1.0
    else:
        _SYST_POLE_RESCALE = float(np.nanmedian(y_ref[good] / y_raw[good]))
    return _SYST_POLE_RESCALE


def _variant_halo(name):
    if name == "baseline_rho2":
        return _fig8_rho2_baseline_halo()
    raw = _raw_systematics_table_halo(name)
    if raw is None:
        return None
    E, y = raw
    return E, y * _systematics_pole_rescale()


def _read_science_band():
    """Parse systematics_science_band.txt -> (rows, anchors). Regenerate if absent."""
    f = SYST_BAND
    if not f.exists():
        return None
    rows, anchor = [], {}
    for ln in f.read_text().splitlines():
        if ln.startswith("# dSph tension:"):
            # "# dSph tension: 5.2  (+0.171 / -3.4) ..."
            anchor["tension"] = float(ln.split(":")[1].split()[0])
        if ln.startswith("#") or ln.startswith("variant") or not ln.strip():
            continue
        p = ln.split()
        if len(p) >= 6:
            rows.append(dict(name=p[0], group=p[1], m=float(p[2]),
                             sv=float(p[3]), tension=float(p[4])))
    return rows, anchor


def _systematics_manifest_rows():
    f = MCMC / "configs" / "systematics" / "manifest.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    rows = []
    for item in data.get("variants", []):
        name = item.get("name")
        if name:
            rows.append(dict(name=name, group=item.get("group", "other")))
    return rows


def _all_systematics_spectrum_rows(science_rows):
    """Rows with available halo spectra, ordered by science table then manifest."""
    out, seen = [], set()

    def add(name, group):
        if name in seen or _variant_halo(name) is None:
            return
        seen.add(name)
        out.append(dict(name=name, group=group or "other"))

    for r in science_rows:
        add(r["name"], r.get("group", "other"))
    for r in _systematics_manifest_rows():
        add(r["name"], r.get("group", "other"))
    for root in dict.fromkeys((SYST, SYST_LEGACY)):
        for t in sorted(root.glob("*/mcmc_global_f_ml.txt")):
            add(t.parent.name, "other")
    return out


def _systematics_context():
    """Load and normalise everything both systematics panels need.

    Factored out of fig5 so the two panels can be drawn either side by side
    (fig5) or as standalone figures (fig11 / fig12) off one identical set of
    rows -- the anchoring rescale below must not be applied twice, and running
    it once here is what guarantees that.
    """
    parsed = _read_science_band()
    if parsed is None:
        raise SystemExit("run propagate_systematics.py first (systematics_science_band.txt missing)")
    rows, anchor = parsed
    base = next((r for r in rows if r["name"] == "baseline_rho2"), rows[0])
    # The systematics tables sit on a different flux normalisation than the
    # validated halo (see propagate_systematics.py): convert each variant's raw
    # tension to the validated, anchored s-wave tension R = R_anchor * (t / t_base),
    # so the baseline matches the run_pppc_mass_scan value (R_anchor ~ 5.2) and the
    # scale is consistent with Fig. 4.
    R_anchor = anchor.get("tension", 5.2)
    t_base = base["tension"]
    for r in rows:
        r["tension"] = R_anchor * r["tension"] / t_base

    # The baseline star should be the validated pixelwise rho^2 bb best-fit
    # point, i.e. the same mass-scan source used by the dSph-tension panel.
    bb_anchor = _pixelwise_bb_bestfit_anchor()
    if bb_anchor is not None:
        for r in rows:
            if r["name"] == "baseline_rho2":
                r.update(bb_anchor)
                break
    base = next((r for r in rows if r["name"] == "baseline_rho2"), rows[0])

    grp_col = {
        "baseline": _ink("k"),
        "diffuse": plasma_color(0.8),
        "diffuse_ics": plasma_color(0.6),
        # IEM grid (Ackermann et al. 2012, ApJ 750, 3): 14 published diffuse
        # models, refit with the Fermi-bubble templates re-extracted per
        # model. This is the band-DEFINING foreground group (see
        # FOREGROUND_GROUPS in propagate_systematics.py), so it needs its own
        # clearly-distinguished series.
        #
        # SWAPPED with point_source_mask: the grid held 1.0, the ramp's far
        # end, which is a pale yellow -- the weakest ink in the figure against
        # a white page, and the most numerous group (14 of 27 points) was
        # wearing it. 0.4 is a magenta that carries far better, and PS mask,
        # with four points and no in-plot label, can afford the pale end.
        "diffuse_grid_bubbles": plasma_color(0.4),
        "hemisphere": plasma_color(0.3),
        "point_source_mask": plasma_color(1.0),
        "psf": plasma_color(0.2),
        "mock": plasma_color(0.95),
        "other": _ink("0.65"),
    }
    grp_lab = {
        "baseline": "baseline",
        "diffuse": "diffuse model",
        "diffuse_ics": "ICS morphology",
        "diffuse_grid_bubbles": "IEM grid",
        "hemisphere": "N/S split",
        "point_source_mask": "PS mask",
        "psf": "PSF",
        "mock": "mock/null",
        "other": "other",
    }
    variant_style = {
        "diffuse_giem_only": (0, (1.2, 1.2)),
        "diffuse_flat_bubbles": "--",
        "diffuse_no_loopI": "-.",
        "diffuse_no_bubbles": (0, (3.0, 1.0, 1.0, 1.0)),
        "diffuse_giem": (0, (5.0, 1.0)),
        "ics_morphology_flat": "--",
        "ics_morphology_steep": ":",
        "mask_extended_only": "-",
        "mask_loose_bright": "--",
        "mask_aggressive_bright": ":",
        "mask_energy_dependent": "-.",
        "psf_none": "-",
        "psf_halo_only": "--",
        "psf_wide_kernel": ":",
        "roi_north": "-",
        "roi_south": "--",
        "mock_null_nohalo": ":",
        "mock_recovery_rho2": "--",
        "mock_diffuse_ics_null": (0, (1.0, 1.0)),
        # High edge of the IEM grid: the one grid variant drawn and labelled
        # individually (the other 13 are a light unlabelled bundle -- see
        # _systematics_variants_panel). Dashed, same weight as the other
        # named foreground variants above.
        "iem_so_z6_bubbles_reextracted": "--",
    }
    variant_col = {
        "diffuse_no_bubbles": plasma_color(0.2),
        "diffuse_giem": plasma_color(0.2),
    }
    excluded_shape_tests = {"diffuse_no_bubbles", "diffuse_giem"}
    # diffuse_grid_bubbles (grid gas/ICS, Fermi-bubble templates re-extracted
    # per model) is the band-defining IEM-grid leg and belongs in the budget
    # alongside diffuse/diffuse_ics. The plain diffuse_grid group (same 14
    # models but with the baseline Lorimer bubbles) is deliberately NOT
    # listed here -- including both would double-count the same 14 models,
    # and diffuse_grid exists only to quantify the bubble-consistency effect
    # (reported separately, not as a band member). cr_injection, quadrant,
    # hemisphere, point_source_mask, psf and mock are instrument/ROI checks
    # or an additional-template test, not alternative foreground models, and
    # point_source_mask/psf are budget-tracked (fit systematics) but are not
    # part of the foreground leg itself.
    budget_groups = {
        "baseline", "diffuse", "diffuse_ics", "diffuse_grid_bubbles",
        "point_source_mask", "psf",
    }
    budget_rows = [
        r for r in rows
        if r["group"] in budget_groups and r["name"] not in excluded_shape_tests
    ]
    return dict(rows=rows, base=base, grp_col=grp_col, grp_lab=grp_lab,
                variant_style=variant_style, variant_col=variant_col,
                excluded_shape_tests=excluded_shape_tests,
                budget_rows=budget_rows)


def _systematics_variants_panel(axL, ctx, *, title=True):
    """(a) Halo spectra: science-budget envelope plus the variants that move it."""
    import matplotlib.pyplot as plt
    rows = ctx["rows"]
    grp_col = ctx["grp_col"]
    variant_style, variant_col = ctx["variant_style"], ctx["variant_col"]
    excluded_shape_tests = ctx["excluded_shape_tests"]
    budget_rows = ctx["budget_rows"]

    # (a) spectra: science-budget envelope + the variants that visibly move it.
    spec_rows = _all_systematics_spectrum_rows(rows)
    spec_by_name = {}
    for r in spec_rows:
        s = _variant_halo(r["name"])
        if s is not None:
            spec_by_name[r["name"]] = (r, s)
    budget_names = {r["name"] for r in budget_rows}
    stress_names = {"diffuse_no_bubbles", "diffuse_giem"}
    # The 14-model IEM grid is thinned to the one variant that sets the high
    # edge of the science band; the other 13 are drawn as a light, unlabelled
    # bundle below (see the "IEM grid bundle" block) rather than as thirteen
    # more individually-labelled curves. The low edge, diffuse_giem_only, is
    # already an individually-labelled "diffuse"-group curve below.
    grid_highlight_names = {"iem_so_z6_bubbles_reextracted"}
    visible_spectra = [
        (name, r["group"])
        for name, (r, _s) in spec_by_name.items()
        if (
            r["group"] in {"diffuse", "diffuse_ics"}
            and ((name in budget_names and name != "baseline_rho2") or name in stress_names)
        )
        or (r["group"] == "diffuse_grid_bubbles" and name in grid_highlight_names)
    ]
    # Highlighted grid variant first: it carries the longest label in the
    # legend ("IEM grid, high edge"), and the legend fills column-major, so
    # leading with it keeps that width in column 1 rather than column 3.
    visible_spectra.sort(key=lambda nv: nv[0] not in grid_highlight_names)
    band_specs = [
        spec_by_name[r["name"]][1]
        for r in budget_rows
        if r["name"] in spec_by_name
    ]
    if band_specs:
        E0 = band_specs[0][0]
        M = np.vstack([s[1] for s in band_specs])
        ylo = np.nanmin(M, axis=0)
        yhi = np.nanmax(M, axis=0)
        # 0.60 inverts to grey 0.40 on the dark styles -- darker than the
        # slide background, so at alpha 0.18 the band was effectively
        # invisible there. 0.45 gives a mid grey in print and a LIGHT grey on
        # a dark slide, and the edge keeps the band readable where it pinches.
        axL.fill_between(E0, ylo, yhi, color=_ink("0.45"), alpha=0.30,
                         lw=0.8, edgecolor=_ink("0.35"), label="science-band")
        for name, group in visible_spectra:
            _, (E, y) = spec_by_name[name]
            axL.plot(
                E,
                y,
                color=variant_col.get(name, grp_col[group]),
                lw=1.35,
                ls=variant_style.get(name, "-"),
                alpha=0.78 if name in excluded_shape_tests else 0.92,
                zorder=3 if name in excluded_shape_tests else 4,
            )
        # --- IEM grid bundle (Ackermann+12, bubbles re-extracted) --------
        # The 13 grid variants that are not individually highlighted, drawn
        # once as a light, unlabelled bundle in the group colour -- the same
        # "many thin traces, one legend entry" treatment as the hard-IC
        # family just below. This is what stands in for the grid without
        # drawing 13 more named curves: the reader sees the spread, and the
        # two variants that actually set the band (GIEM only, labelled above
        # under "diffuse"; and the grid high edge, labelled below) are
        # findable individually.
        _grid_bundle_drawn = False
        for r in budget_rows:
            if r["group"] != "diffuse_grid_bubbles" or r["name"] in grid_highlight_names:
                continue
            s = spec_by_name.get(r["name"])
            if s is None:
                continue
            _, (_gE, _gy) = s
            axL.plot(_gE, _gy, color=grp_col["diffuse_grid_bubbles"],
                     lw=0.85, alpha=0.30, zorder=2, label="_nolegend_")
            _grid_bundle_drawn = True

        # The hard-IC injected-population family is deliberately NOT drawn here.
        # It has its own dedicated figures (fig20, and the angular-profile
        # overlay fig21); on this panel it added six traces that sit on the
        # baseline and only crowded the systematics budget the panel is about.
        bE, by = _variant_halo("baseline_rho2")
        axL.plot(bE, by, "o-", color=_ink("k"), lw=1.55, ms=3.2, label="baseline", zorder=10)
        peak_idx = int(np.nanargmax(by)) if np.any(np.isfinite(by)) else None
        if peak_idx is not None:
            axL.annotate(
                # One decimal: the bin centre is 20.76 GeV and ":.0f" rounded it
                # to a bare "21", which reads as a different (round) number
                # rather than as the actual bin this analysis peaks in.
                f"baseline peak: {bE[peak_idx]:.1f} GeV",
                xy=(bE[peak_idx], by[peak_idx]),
                xytext=(0.53, 0.82),
                xycoords="data",
                textcoords="axes fraction",
                ha="left",
                va="top",
                fontsize=ANNOT_FS,
                color=_ink("k"),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=_ink("0.25")),
            )
        # axL.text(
        #     0.94,
        #     0.51,
        #     "PS mask, PSF and ICS variants\noverlie the baseline",
        #     transform=axL.transAxes,
        #     ha="right",
        #     va="bottom",
        #     fontsize=ANNOT_FS,
        #     color="0.25",
        #     zorder=20,
        #     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.76),
        # )
    axL.axhline(0.0, color=_ink("0.5"), lw=0.7, ls=":")
    axL.axvline(20.0, color=_ink("0.8"), lw=0.8, ls=":")
    axL.set_xscale("log"); axL.set_xlim(1.0, 1000.0)
    axL.set_xlabel(XLAB)
    axL.set_ylabel(" ")  # reserve the gutter; real label drawn stacked below
    stacked_ylabel(axL, r"$E^2\,\mathrm{d}N/\mathrm{d}E$ at galactic poles",
                   YLAB_UNITS, x=-0.17, y=0.5)
    # "(a)" only means something next to a "(b)". The standalone renders of this
    # panel (fig11/fig12) pass title=False so they carry no orphaned panel letter.
    if title:
        axL.set_title("(a) Halo variants")
    _mathtext_sci_axis(axL, "y")
    handles = [
        plt.Line2D([0], [0], color=_ink("k"), lw=1.55, marker="o", ms=3.2),
        # facecolor, NOT color: passing `color` alongside `edgecolor` makes
        # matplotlib override the edge ("Setting the 'color' property will
        # override the edgecolor or facecolor properties"), so the swatch lost
        # the edge that the actual science band has and no longer matched it.
        plt.Rectangle((0, 0), 1, 1, facecolor=_ink("0.45"), alpha=0.30,
                      lw=0.8, edgecolor=_ink("0.35")),
    ]
    labels = [
        "baseline",
        "science-band",
    ]
    spectrum_label = {
        "diffuse_giem_only": "GIEM only",
        "diffuse_flat_bubbles": "flat bubbles",
        "diffuse_no_loopI": "no Loop I",
        "diffuse_no_bubbles": r"no bubbles$^{*}$",
        "diffuse_giem": r"GIEM$^{*}$",
        "ics_morphology_flat": "ICS flat",
        "ics_morphology_steep": "ICS steep",
        "iem_so_z6_bubbles_reextracted": "IEM grid, high edge",
    }
    for name, group in visible_spectra:
        handles.append(
            plt.Line2D(
                [0], [0], color=variant_col.get(name, grp_col[group]),
                lw=1.35, ls=variant_style.get(name, "-"),
            )
        )
        labels.append(spectrum_label.get(name, name.replace("_", " ")))
    # Bundle entry LAST -- shortest label, so it lands in the final column and
    # leaves that column narrow (see the note above visible_spectra.sort).
    if _grid_bundle_drawn:
        handles.append(plt.Line2D([0], [0], color=grp_col["diffuse_grid_bubbles"],
                                  lw=1.2, alpha=0.35))
        labels.append("IEM grid")
    leg = axL.legend(
        handles,
        labels,
        fontsize=ANNOT_FS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.9,
        handletextpad=0.45,
        borderpad=0.32,
        **LEGEND_KW,
    )
    _style_legend(leg, zorder=20)
    # axL.text(
    #     0.75,
    #     -0.10,
    #     r"$^{*}$ stress test; excluded"+ "\n" + r"from systematics budget",
    #     transform=axL.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=ANNOT_FS,
    #     color="0.25",
    #     zorder=20,
    #     clip_on=False,
    # )
    axL.grid(alpha=0.13, which="both")


def _systematics_tension_panel(axR, ctx, *, title=True):
    """(b) dSph tension vs best-fit mass, one point per systematic variant."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    base = ctx["base"]
    grp_col, grp_lab = ctx["grp_col"], ctx["grp_lab"]
    budget_rows = ctx["budget_rows"]

    # (b) direct diagnostic: each point is one systematic variant.
    def short_name(name):
        mapping = {
            "baseline_rho2": "baseline",
            "mask_energy_dependent": "energy mask",
            "mask_extended_only": "extended only",
            "mask_loose_bright": "loose mask",
            "mask_aggressive_bright": "aggr. mask",
            "psf_none": "no PSF",
            "psf_halo_only": "halo PSF",
            "psf_wide_kernel": "wide PSF",
            "diffuse_giem_only": "GIEM only",
            "diffuse_flat_bubbles": "flat bubbles",
            "diffuse_no_loopI": "no Loop I",
            "diffuse_no_bubbles": "no bubbles",
            "diffuse_giem": "GIEM",
            "ics_morphology_flat": "ICS flat",
            "ics_morphology_steep": "ICS steep",
            "roi_north": "north",
            "roi_south": "south",
            "mock_diffuse_ics_null": "diffuse mock",
            "iem_so_z6_bubbles_reextracted": "IEM grid, high edge",
        }
        return mapping.get(name, name.replace("_", " "))

    plot_rows = budget_rows
    tens = np.array([r["tension"] for r in plot_rows], float)
    masses = np.array([r["m"] for r in plot_rows], float)
    xlo = 0.65
    # Post-IEM-grid total systematic band (systematics/total_systematic_band.py,
    # quadrature case): R = 5.121 [1.6, 9.8] for bb, with the foreground leg at
    # x[0.375, 1.320] once the 14-model grid (bubbles re-extracted) is folded in
    # as FOREGROUND_GROUPS. This panel is bb-only, so it reads the module
    # scalars rather than R_BAND. Superseded values that may still appear in old
    # notes: [1.6, 9.3] pre-grid, [1.7, 10.0] on the 14/80-point grids, and
    # foreground legs x[0.359, 1.032] and x[0.384, 1.339].
    outer_lo, outer_hi = R_BAND_LO, R_BAND_HI
    xhi = max(outer_hi, 6.1, tens.max() + 0.35)
    # Two nested bands, and the nesting carries the argument:
    #   outer  1.7 -> axis edge : the FULL band, i.e. the fit systematics above
    #          PLUS the halo J-factor and dSph J-factor uncertainties. 1.7 is
    #          the floor -- even with every uncertainty pushed the favourable
    #          way, the dSph tension does not reach 1.
    #   inner  tens.min -> tens.max : the fit-systematics spread alone, which
    #          (with the IEM grid now in budget_rows) already coincides with
    #          the reported foreground-leg range x[0.384, 1.339] * R0 = 5.2,
    #          i.e. [~2.0, ~7.0] -- diffuse_giem_only at the low edge and the
    #          grid's iem_so_z6_bubbles_reextracted at the high edge.
    #
    # This outer span was already drawn but at _ink("0.88") and unlabelled --
    # 0.88 inverts to grey 0.12 on the dark styles, darker than the slide
    # background, so it was invisible exactly where it matters. The two greys
    # are chosen so the INNER band reads as the stronger one in both themes
    # (theme_ink inverts, so "stronger" is darker in print, lighter on a slide).
    axR.axvspan(outer_lo, xhi, color=_ink("0.70"), alpha=0.22, zorder=0)
    axR.axvline(outer_lo, color=_ink("0.45"), lw=1.0, ls=(0, (5, 2)), zorder=1)
    axR.axvspan(xlo, 1.0, color="#F3D35B", alpha=0.32, zorder=0)
    axR.axvspan(tens.min(), tens.max(), color=_ink("0.45"), alpha=0.30, zorder=0)
    axR.axvline(1.0, color=_ink("0.25"), lw=1.1, ls=":", label=r"no-tension boundary")
    axR.axvline(base["tension"], color=_ink("k"), lw=1.25, ls="--", label=f"baseline tension")
    axR.axhline(base["m"], color=_ink("k"), lw=1.0, ls=":", alpha=0.65, label=f"baseline mass")

    # Offsets are quoted in POINTS at the print annotation size. They must
    # scale with the type: on a slide ANNOT_FS is 1.5x, so an unscaled offset
    # displaces a label by two-thirds as many text-heights as it does in print
    # and every callout drifts in toward its point -- which is why the
    # conference layout did not match the paper one. Scaling keeps the label's
    # position relative to its marker identical in both styles.
    _off = ANNOT_FS / _PRINT_ANNOT_FS
    label_offsets = {
        name: (dx * _off, dy * _off) for name, (dx, dy) in {
            "baseline_rho2": (-27, 6),
            "diffuse_giem_only": (-9, 25),
            "diffuse_flat_bubbles": (-38, 15),
            "ics_morphology_flat": (-12, 6),
            "ics_morphology_steep": (-29, -20),
            "iem_so_z6_bubbles_reextracted": (-50, -22),
        }.items()
    }
    label_ha = {"ics_morphology_flat": "right"}
    # No text-colour override needed any more. This existed because the grid
    # marker used to be plasma(1.0), a near-white yellow that worked as a
    # black-edged dot but not as label TEXT on a white page (~1.6:1 contrast).
    # Now that the grid wears plasma(0.4) magenta, its label reads perfectly
    # well in the marker colour, and the label can simply match the point.
    label_text_col = {}
    for r in plot_rows:
        color = grp_col.get(r["group"], _ink("0.5"))
        text_color = label_text_col.get(r["group"], color)
        marker = "*" if r["name"] == "baseline_rho2" else "o"
        size = 190 if r["name"] == "baseline_rho2" else 48
        axR.scatter(
            r["tension"],
            r["m"],
            s=size,
            marker=marker,
            color=color,
            edgecolor=_ink("k"),
            linewidth=1.05 if r["name"] == "baseline_rho2" else 0.55,
            # The 14-model IEM grid goes UNDER the other variants. It is the
            # most numerous group by far (14 of 27 points) and clusters right
            # where the ICS/PS-mask/PSF points sit, so drawn at the common
            # zorder it covered the individually-labelled variants that the
            # panel is actually about. 3 keeps it above the shaded bands
            # (zorder 0-1) but below every other point.
            zorder=(8 if r["name"] == "baseline_rho2"
                    else 3 if r["group"] == "diffuse_grid_bubbles"
                    else 5),
        )
        if r["name"] in label_offsets:
            arrowprops = dict(arrowstyle="-", lw=0.55, color=_ink("0.25"))
            axR.annotate(
                short_name(r["name"]),
                xy=(r["tension"], r["m"]),
                xytext=label_offsets[r["name"]],
                textcoords="offset points",
                ha=label_ha.get(r["name"], "left"),
                fontsize=ANNOT_FS,
                color=text_color,
                arrowprops=arrowprops,
                rotation=0,
                rotation_mode="anchor",
                # Text and annotate default to zorder 3, but the scatter points
                # are drawn at 5 (8 for the baseline star), so every callout and
                # its leader line was being covered by the data it points at.
                # 20 puts the whole label layer on top, as intended.
                zorder=20,
            )

    axR.text(
        0.85,
        0.35,
        r"no dSph tension ($\mathcal{R}_{bb}<1$)",
        transform=axR.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=ANNOT_FS,
        rotation=90,
        # Same reason as label_text_col above -- this olive is unreadable on
        # the dark styles without theme_accent lifting it.
        color=_accent("#6B4F00"),
        zorder=20,
    )
    axR.text(
        1.18,
        0.90,
        r"$\mathcal{R}_{bb}=1$",
        transform=axR.get_xaxis_transform(),
        ha="center",
        va="center",
        rotation=90,
        fontsize=ANNOT_FS,
        color=_ink("0.2"),
        zorder=20,
    )
    # Names the outer band in-plot, matching how the R_bb=1 boundary and the
    # no-tension region are labelled. A legend entry would have been the other
    # option, but this panel's legend is built from explicit group handles and
    # is already three columns wide.
    axR.text(
        outer_lo + 0.20,
        0.55,
        r"full band",
        transform=axR.get_xaxis_transform(),
        ha="center",
        va="center",
        rotation=90,
        fontsize=ANNOT_FS,
        color=_ink("0.3"),
        zorder=20,
    )
    axR.set_xlabel(r"s-wave dSph tension ($\mathcal{R}_{bb}$)")
    axR.set_ylabel(r"best-fit $m_\chi$ [GeV]")
    if title:
        axR.set_title(r"(b) dSph tension ($b\bar b$ channel)")
    # 6.2 was tuned to the pre-grid points (max ~5.7); the grid's high edge
    # (iem_so_z6_bubbles_reextracted, tension ~7.0) needs headroom too, so the
    # right edge now tracks the data instead of staying pinned at the old
    # constant.
    axR.set_xlim(xlo, max(6.2, tens.max() + 0.6))
    axR.set_ylim(max(0.0, masses.min() - 70.0), masses.max() + 70.0)
    axR.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
    axR.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    axR.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    axR.xaxis.set_minor_formatter(mticker.NullFormatter())
    group_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=grp_col[g],
                   markeredgecolor=_ink("k"), markeredgewidth=0.5, ms=5.5, label=grp_lab[g])
        for g in ["baseline", "diffuse", "diffuse_ics", "diffuse_grid_bubbles",
                  "point_source_mask", "psf"]
        if any(r["group"] == g for r in plot_rows)
    ]
    _style_legend(axR.legend(
        handles=group_handles,
        fontsize=ANNOT_FS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.45,
        borderpad=0.32,
        **LEGEND_KW,
    ))
    axR.grid(alpha=0.13, which="both")


def fig5(outdir):
    import matplotlib.pyplot as plt
    ctx = _systematics_context()
    # Height only; the rect bottom stays 0.20. Both legends are CHILDREN of
    # their axes, so tight_layout counts them inside the axes' tight bbox --
    # which means raising the rect bottom squeezes the axes rather than
    # freeing the legend, and lowering the canvas height does the same. Adding
    # height with the rect unchanged is what actually gives the legend room:
    #   5.0 -> axes 2.072 in, legend top 0.425 vs x-label bottom 0.424 (touching)
    #   5.6 -> axes 2.469 in, legend top 0.403 vs x-label bottom 0.415 (clear)
    # so the axes end up LARGER than before, not smaller.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=_figsize(FULL_W, 5.6),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    _systematics_variants_panel(axL, ctx)
    _systematics_tension_panel(axR, ctx)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    return _save(fig, outdir, "f_systematics_impact")


def fig11(outdir):
    """Systematics halo variants, on its own (Fig 5 panel (a), standalone)."""
    import matplotlib.pyplot as plt
    ctx = _systematics_context()
    # 6.0, not 5.0: same reason as fig5. At 5.0 this panel's legend top sat
    # 0.002 figure-fractions under the x-label -- clear only by accident, and
    # one legend row from breaking. The extra height also grows the axes
    # (2.15 -> 2.81 in) rather than shrinking them.
    # COL_W in print: this is a one-column figure and FULL_W*0.55 (3.88 in) is
    # neither column width nor text width, so it emitted ~3.7 in against a
    # 3.4 in \columnwidth and LaTeX shrank it ~8% -- its type ending up smaller
    # than the rest of the one-column set for no reason anyone chose. Slide
    # geometry is left alone; it was tuned against a 16:9 frame, not a column.
    fig, ax = plt.subplots(figsize=_figsize(FULL_W * 0.55 if _SLIDE_STYLE else COL_W, 6.0))
    _systematics_variants_panel(ax, ctx, title=False)
    # The panel's own legend already hangs below the axes at bbox y=-0.18; on a
    # single panel it needs the same bottom reservation Fig 5 gives the pair.
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    return _save(fig, outdir, "f_systematics_halo_variants")


def fig12(outdir):
    """Systematics dSph tension, on its own (Fig 5 panel (b), standalone)."""
    import matplotlib.pyplot as plt
    ctx = _systematics_context()
    # COL_W in print: this is a one-column figure and FULL_W*0.55 (3.88 in) is
    # neither column width nor text width, so it emitted ~3.7 in against a
    # 3.4 in \columnwidth and LaTeX shrank it ~8% -- its type ending up smaller
    # than the rest of the one-column set for no reason anyone chose. Slide
    # geometry is left alone; it was tuned against a 16:9 frame, not a column.
    fig, ax = plt.subplots(figsize=_figsize(FULL_W * 0.55 if _SLIDE_STYLE else COL_W, 5.0))
    _systematics_tension_panel(ax, ctx, title=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    return _save(fig, outdir, "f_systematics_dsph_tension")


# --------------------------------------------------------------------------- #
def _save(fig, outdir, stem):
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    _use_mathtext_scalar_offsets(fig)

    # Save the vector PDF (what the paper uses) and a raster PNG alongside it
    # (for slides / quick viewing). Both come off the SAME figure object before
    # it is closed, so the two can never drift out of sync.
    kw = current_savefig_kwargs(bbox_inches="tight")
    p = outdir / f"{stem}.pdf"
    fig.savefig(p, dpi=220, **kw)
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=220, **kw)
    plt.close(fig)
    print(f"  saved {p.name} + {png.name}")
    return p


# --------------------------------------------------------------------------- #
#  Fig 7 : excess morphology -- model-independent angular profile
# --------------------------------------------------------------------------- #
def _angular_profile_panel(ax, *, emin=10.0, emax=40.0, title=None):
    """Model-independent angular profile of the no-halo residual vs NFW^p /
    isotropic morphologies. Reuses the computation in
    morphology/plot_angular_profile_vs_models.py so the physics has one source
    of truth; only the rendering is done here, in the paper style.

    Split out of fig7 so fig13 can pair it with the likelihood profiles.
    """
    from astropy.io import fits
    sys.path.insert(0, str(_HERE / "morphology"))
    import plot_angular_profile_vs_models as morph

    fitdir = MCMC / "fit_results" / "pixelwise_mcmc" / "pixelwise_mcmc_results_fig4_resid"
    if not any(fitdir.glob("global_fit_k*.npz")):
        import make_pixelwise_fig4_resid as make_resid
        make_resid.main([
            "--mcmc-dir",
            str(MCMC / "fit_results" / "pixelwise_mcmc" / "pixelwise_mcmc_results_fig4"),
            "--out-dir",
            str(fitdir),
        ])

    hdr = fits.getheader(morph.COUNTS)
    L, B = morph._lonlat(hdr)
    exc, mask = morph.excess_map(fitdir, emin, emax)
    edges = np.linspace(10.0, 70.0, 13)
    ctr, val, err = morph.ring_profile(exc, mask, L, B, edges)

    models = {r"NFW $\rho^2$": 2.0, r"NFW $\rho^{2.5}$": 2.5, r"NFW $\rho^1$": 1.0,
               "isotropic": None}
    fits_out = {}
    for name, p in models.items():
        m = np.ones_like(ctr) if p is None else np.array([morph.nfw_los(x, p) for x in ctr])
        A, C, chi2n = morph.fit_amp_offset(m, val, err)
        fits_out[name] = (A * m + C, chi2n)
    print("  morphology chi2/dof:",
          ", ".join(f"{n.split()[-1]}={fits_out[n][1]:.1f}" for n in models))

    colors = get_cmap_colors(cmap_name="plasma", n=6, start=0.15, end=0.85)
    cols = [colors[2], colors[0], colors[3], colors[5]]  # NFW2, NFW1, NFW2.5, isotropic
    ax.errorbar(ctr, val, yerr=err, fmt="o", color=_ink("k"), ms=4, capsize=2, lw=1.2,
                zorder=5, label=r"excess (data $-$ no-halo)")
    style = {r"NFW $\rho^2$": "-", r"NFW $\rho^{2.5}$": ":", 
             r"NFW $\rho^1$": "--","isotropic": "-."}
    for (name, p), c in zip(models.items(), cols):
        pred, chi2n = fits_out[name]
        ax.plot(ctr, pred, style[name], color=c, lw=1.8,
                label=fr"{name} ($\chi^2/\nu$={chi2n:.1f})")
    ax.axhline(0.0, color=_ink("0.6"), lw=0.7, ls=":")
    _mathtext_sci_axis(ax, "y")
    ax.set_xlabel(r"angle from GC, $\psi$ [deg]")
    ax.set_ylabel(" ")  # reserve the gutter; real label drawn stacked below
    stacked_ylabel(ax, r"excess (data$-$model)", r"[counts, arb.]", x=-0.15, dx=0.055, y=0.5)
    _style_legend(ax.legend(loc="upper right", borderpad=0.4,
                            labelspacing=0.3, **LEGEND_KW))
    ax.grid(alpha=0.13, which="both")
    if title:
        ax.set_title(title)


def fig7(outdir, emin=10.0, emax=40.0):
    import matplotlib.pyplot as plt
    # COL_W x 2.8 in every style -- the original geometry. A 16:9 stretch was
    # tried here and made the panel long and thin; the angular profile is a
    # compact falling curve and does not want the extra width.
    fig, ax = plt.subplots(figsize=_figsize(COL_W, 2.8))
    _angular_profile_panel(ax, emin=emin, emax=emax)
    fig.tight_layout()
    return _save(fig, outdir, "f_angular_profile_morphology")


# --------------------------------------------------------------------------- #
#  Fig 13 : angular profile + likelihood profiles, side by side
# --------------------------------------------------------------------------- #
#: Where run_global_systematics.sh leaves the per-variant halo spectra. The
#: reorg moved these out of Totani_paper_check; check_null_and_hemisphere.py
#: still pointed at the old path and silently reported every variant as
#: "not run yet", so resolve both and prefer the current one.
_SYST_DIRS = [
    _HERE / "figures" / "systematics" / "plots_global_systematics",
    PC / "figures" / "plots_global_systematics",
]


def _load_syst_variant(name):
    """Halo (nfw) spectrum for one systematics variant, or None if absent.

    Some variants (the deterministic mock injections) carry zero error columns
    rather than a posterior width; 'has_err' says whether error bars are
    meaningful so the caller can draw them as a line instead of 0-length bars.
    """
    for root in _SYST_DIRS:
        t = root / name / "mcmc_global_f_ml.txt"
        if not t.exists():
            continue
        a = np.genfromtxt(t, names=True, comments="#")
        cols = a.dtype.names
        y = np.asarray(a["nfw"], float)
        lo = np.asarray(a["nfw_err_lo"], float) if "nfw_err_lo" in cols else np.zeros_like(y)
        hi = np.asarray(a["nfw_err_hi"], float) if "nfw_err_hi" in cols else np.zeros_like(y)
        return dict(E=np.asarray(a["E_GeV"], float), y=y, lo=lo, hi=hi,
                    has_err=bool(np.any(lo > 0) or np.any(hi > 0)))
    return None


def _syst_series(ax, d, *, color, marker, label, ls="none", lw=1.4, ms=4.0, z=5):
    """One variant's halo spectrum. Deterministic mocks carry no posterior
    width, so those are drawn as a line rather than zero-length error bars."""
    if d is None:
        return
    if d["has_err"]:
        ax.errorbar(d["E"], d["y"], yerr=[d["lo"], d["hi"]], fmt=marker, ms=ms,
                    color=color, ecolor=color, capsize=2, lw=lw, ls=ls,
                    label=label, zorder=z)
    else:
        ax.plot(d["E"], d["y"], ls="-" if ls == "none" else ls, lw=lw,
                color=color, label=label, zorder=z)


def _finish_syst_panel(ax, *, headroom):
    """Shared axis dressing + a bottom-centre legend that clears the data.

    headroom is the fraction of the FINAL axes height reserved below the data
    for the legend. Extending the lower ylim (rather than letting the legend
    sit at the natural bottom) is what keeps the box off the y=0 guide line,
    which otherwise runs straight through it.
    """
    ax.axhline(0.0, color=_ink("0.5"), lw=0.7, ls=":")
    ax.axvline(20.0, color=_ink("0.8"), lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_xlabel(XLAB)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    _mathtext_sci_axis(ax, "y")
    ax.set_ylabel("halo amplitude")

    # Drop the floor so the reserved band is `headroom` of the final height:
    # span_new = span_data / (1 - headroom).
    lo, hi = ax.get_ylim()
    ax.set_ylim(hi - (hi - lo) / (1.0 - headroom), hi)

    leg = _style_legend(ax.legend(loc="lower center", borderpad=0.35,
                                  labelspacing=0.25, fontsize=ANNOT_FS,
                                  handletextpad=0.4, **LEGEND_KW),
                        zorder=20)
    # zorder alone is not enough: the house framealpha is 0.55, so anything
    # passing under the box still reads through it. _style_legend re-applies
    # LEGEND_KW["framealpha"], so this must come after it.
    leg.get_frame().set_alpha(0.93)


def _syst_variants(*names):
    """Load the named variants, erroring with the command that produces them."""
    out = {k: _load_syst_variant(k) for k in names}
    if out.get("baseline_rho2") is None:
        raise SystemExit(
            "global systematics variants missing -- run:\n"
            "  bash Totani_reanalysis/run_global_systematics.sh")
    return out


def fig18(outdir):
    r"""Disk included vs excluded, for both cuspy profiles, pixelwise throughout.

    (a) pole spectrum, (b) per-bin delta-lnL against the no-halo fit. Each
    panel carries the same four series: NFW rho^2 and rho^2.5, each fit with
    the disk excluded and included.

    Encoding follows this file's convention -- COLOUR is the profile, and the
    disk variant is carried by linestyle + marker fill (open/solid = excluded/
    included). That keeps the legend factored into 2 + 2 entries instead of 4,
    and makes the comparison the figure is about (within a colour) the easy
    one to read.

    The point of showing both profiles together is that the disk does NOT act
    on them alike. Summed over 10-40 GeV the TS = 2*delta-lnL goes 927 -> 917
    for rho^2 (a 1% change) but 795 -> 367 for rho^2.5, which more than halves.
    rho^2.5 is the cuspier, more disk-concentrated template, so it is the one
    that competes with disk emission.

    Both panels are pixelwise: cellwise has no disk-included variant, so this
    comparison is only defined there. Each delta-lnL curve is differenced
    against the no-halo fit run over the SAME pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cols = get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)
    # (fig number, label, colour) -- rho^2 first, it is the headline profile.
    # Pink and yellow -- the two ends of this ramp -- rather than adjacent
    # samples, because here the two profiles are overlaid on the same axes and
    # differ only by a dashed/solid partner curve, so they need maximum
    # separation. cols[2] (orange) sits too close to cols[0] for that.
    profiles = [(6, r"$\rho^{2}$", cols[4]),
                (5, r"$\rho^{2.5}$", cols[0])]
    # (with_disk, linestyle, marker-face maker, short label)
    variants = [(False, "-", lambda c: _ink("white"), "disk excl."),
                (True, "--", lambda c: c, "disk incl.")]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=_figsize(FULL_W, 3.0))

    for fig_n, plab, col in profiles:
        nfw_label = next(q["label"] for q in FIG8_PANELS if int(q["fig"]) == fig_n)
        for with_disk, ls, mfc_of, vlab in variants:
            # --- (a) spectrum ------------------------------------------
            h = load_halo_spectrum(pixelwise_dir(fig_n, with_disk),
                                   nfw_label=nfw_label, central_stat="f_p50")
            _plot_halo_points(axA, h, color=col,
                              marker="o" if not with_disk else "^",
                              label=None, mfc=mfc_of(col),
                              x_shift=1.0 if not with_disk else 1.06,
                              zorder=6 if not with_disk else 5)
            # --- (b) delta-lnL -----------------------------------------
            E, dl, _bd = _delta_lnl_overlay_series(fig_n, pixelwise=True,
                                                   with_disk=with_disk)
            axB.plot(E, dl, ls=ls, marker="o" if not with_disk else "^",
                     ms=4.0, lw=1.5, color=col, mfc=mfc_of(col), mec=col,
                     zorder=6 if not with_disk else 5)

    # rho^2 has the wider range of the two, so its limits hold both.
    rho2 = next(q for q in FIG8_PANELS if int(q["fig"]) == 6)
    axA.set_ylim(*rho2["ylim"])
    axA.set_yticks(rho2["yticks"])
    axA.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ at poles"
                   "\n" r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    # Plain centred set_title, matching fig13's "(a) Angular profile" /
    # "(b) Likelihood profiles". Lettered (c)/(d) so the four panels read as
    # one sequence with fig13 across consecutive slides.
    axA.set_title("(c) Pole spectrum")
    _mathtext_sci_axis(axA, "y")

    axB.set_ylabel(r"$\ln(L)-\ln(L_{\rm no\text{-}halo})$")
    axB.set_title("(d) Halo likelihood")

    for ax in (axA, axB):
        ax.axhline(0.0, color=_ink("0.5"), lw=0.7, ls=":")
        # The two bands replace the old single 20 GeV guide line. Same ink at
        # two strengths: the halo band is the subject, the GCE band is context
        # for where the low-energy disk-included excursion actually sits.
        ax.axvspan(*GCE_BAND, color=GCE_BAND_COLOR, alpha=0.16, lw=0, zorder=0)
        ax.axvspan(*HALO_BAND, color=HALO_BAND_COLOR, alpha=0.22, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(1.0, 1000.0)
        ax.set_xlabel(XLAB)
        ax.grid(alpha=0.13, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        # Labelled in both panels so neither borrows context from the other.
        # (c) puts them at the BOTTOM: its spectrum peaks along the top of the
        # axes, so a top label sits on the points. (d) has its clear space at
        # the top instead, both curves being near zero across the bands there.
        at_bottom = ax is axA
        y_lab, va = (0.03, "bottom") if at_bottom else (0.965, "top")
        for (lo, hi), lab, col in ((GCE_BAND, "GCE", GCE_BAND_COLOR),
                                   (HALO_BAND, "halo", HALO_BAND_COLOR)):
            ax.text(np.sqrt(lo * hi), y_lab, lab,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va=va, fontsize=ANNOT_FS,
                    color=col, zorder=2)

    # Factored legend: colour = profile, linestyle/fill = disk variant.
    handles = [Line2D([0], [0], color=c, lw=2.0, marker="o", ms=4.6, mfc=c, mec=c)
               for _, _, c in profiles]
    labels = [f"NFW {lab}" for _, lab, _ in profiles]
    handles += [Line2D([0], [0], color=_ink("0.35"), lw=1.6, ls=ls,
                       marker="o" if not wd else "^", ms=4.6,
                       mfc=_ink("white") if not wd else _ink("0.35"),
                       mec=_ink("0.35"), mew=0.9)
                for wd, ls, _, _ in variants]
    labels += [vlab for _, _, _, vlab in variants]
    # Upper right: both disk-included curves spike below ~7 GeV and the
    # disk-excluded pair peaks through 10-40, so the left half is full. Above
    # ~100 GeV every series has decayed to zero, leaving that corner clear.
    leg = _style_legend(axB.legend(handles, labels, loc="upper right",
                                   borderpad=0.3, labelspacing=0.25,
                                   fontsize=ANNOT_FS, handletextpad=0.4,
                                   handlelength=2.0, **LEGEND_KW), zorder=20)
    leg.get_frame().set_alpha(0.93)

    fig.tight_layout()
    return _save(fig, outdir, "f_disk_spectrum_and_lnl")


def fig16(outdir):
    r"""N/S hemisphere split of the halo spectrum.

    Loop I, the Fermi bubbles and the ICS gradient are all strongly asymmetric
    about the Galactic plane, so a foreground artefact should not reproduce
    itself above AND below it. The excess does: the two hemisphere peaks agree
    to ~1.4 sigma.

    Standalone companion to fig17; the two were one two-panel figure until
    2026-08-12. No panel-letter title, per the convention that a standalone
    render carries no orphaned "(a)".
    """
    import matplotlib.pyplot as plt

    d = _syst_variants("baseline_rho2", "roi_north", "roi_south")
    cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.12, end=0.88)
    base = d["baseline_rho2"]

    fig, ax = plt.subplots(figsize=_figsize(COL_W, 2.8))
    ax.plot(base["E"], base["y"], "-", color=_ink("black"), lw=1.6,
            label="full ROI", zorder=4)
    _syst_series(ax, d["roi_north"], color=cols[1], marker="o",
                 label=r"north, $b>0$", z=6)
    _syst_series(ax, d["roi_south"], color=cols[4], marker="s",
                 label=r"south, $b<0$", z=6)
    _finish_syst_panel(ax, headroom=0.26)
    fig.tight_layout()
    return _save(fig, outdir, "f_hemisphere_split")


def fig17(outdir):
    r"""Injection tests: what foreground mismodelling can and cannot produce.

    On one axis, and this is the point of the figure:
      * an injected halo comes back at the right amplitude, so the pipeline is
        unbiased and would have found a foreground-shaped signal;
      * a deliberately mismodelled ICS morphology, and a no-halo sky, produce
        essentially nothing at 20 GeV.

    NB the mismodelled-ICS null is marginal rather than clean -- its worst
    peak-band bin sits ~2 sigma from zero (check_null_and_hemisphere.py grades
    it MARGINAL). The honest claim is "at most a 2 sigma, ~10%-amplitude
    blip", not "nothing at all".
    """
    import matplotlib.pyplot as plt

    d = _syst_variants("baseline_rho2", "mock_recovery_rho2",
                       "mock_diffuse_ics_null", "mock_null_nohalo")
    cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.12, end=0.88)
    base = d["baseline_rho2"]

    fig, ax = plt.subplots(figsize=_figsize(COL_W, 2.8))
    ax.plot(base["E"], base["y"], "-", color=_ink("black"), lw=1.6,
            label="measured halo", zorder=4)
    _syst_series(ax, d["mock_recovery_rho2"], color=cols[1], marker="^",
                 label="injected halo, recovered", ls="--", z=6)
    _syst_series(ax, d["mock_diffuse_ics_null"], color=cols[4], marker="o",
                 label="mismodelled ICS", z=7)
    _syst_series(ax, d["mock_null_nohalo"], color=cols[3], marker="v",
                 label="no-halo sky", ls=":", z=6)
    # Four entries rather than three, so a deeper reserved band.
    _finish_syst_panel(ax, headroom=0.33)
    fig.tight_layout()
    return _save(fig, outdir, "f_injection_nulls")


def fig15(outdir):
    """The NFW-rho^2 pole spectrum alone: disk excluded vs disk included.

    Fig 1 / Fig 14 show this comparison three times, once per halo profile.
    rho^2 is the one the halo result rests on, so this is that panel by itself
    for a slide that is only making the disk-systematic point:

        open circles  cellwise, disk excluded   (the Totani-replication fit)
        squares       pixelwise, disk excluded
        triangles     pixelwise, disk included

    Same panel function as the multi-panel versions, so the series, marker
    shifts, guides and y-scale are identical -- this is a crop, not a redraw.
    Compare with fig8 (f_cellwise_nfw_rho2), which is the same profile with the
    cellwise series ONLY and so makes no statement about the disk.
    """
    import matplotlib.pyplot as plt

    panel = next(p for p in FIG8_PANELS if int(p["fig"]) == 6)   # NFW-rho^2
    colors = get_cmap_colors(cmap_name="plasma", n=5, start=0.15, end=0.65)
    fg = plt.rcParams.get("text.color", "black")

    # COL_W, exactly as every other single-panel figure here (fig7, fig8,
    # fig2 are all COL_W x 2.8). That is not cosmetic: the font and tick sizes
    # are absolute points set by the style, so widening the canvas shrinks the
    # type RELATIVE to the plot and this figure stops matching its siblings.
    # Geometry copied from fig8 (f_cellwise_nfw_rho2), which is this same
    # profile with the cellwise series only: COL_W x 2.8, legend INSIDE the
    # axes, tight_layout. Matching it exactly is the point -- these two are the
    # same measurement at different levels of detail and belong on consecutive
    # slides, so any difference in canvas or type scale reads as a mistake.
    # An inside legend also costs the panel no height at all.
    fig, ax = plt.subplots(figsize=_figsize(COL_W, 2.8))
    _pole_flux_panel(ax, panel, colors, fg)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(" ")  # reserve the gutter; real label drawn stacked below
    stacked_ylabel(ax, YLAB_MAIN, YLAB_UNITS, x=-0.17, dx=0.055, y=0.5)
    # Lower right, as in fig8: the three series converge toward zero above
    # ~100 GeV, so the bottom-right corner is the one quadrant they all leave
    # clear. (Bottom-LEFT is where the two negative low-energy points sit.)
    _style_legend(ax.legend(loc="lower right", borderpad=0.3,
                            labelspacing=0.25, handletextpad=0.4,
                            fontsize=ANNOT_FS, **LEGEND_KW))
    fig.tight_layout()
    return _save(fig, outdir, "f_pole_flux_rho2_disk_comparison")


def fig14(outdir):
    """Fig 1 (f8) laid out side by side instead of stacked.

    Same three profiles, same panel function -- only the arrangement differs.
    The stacked version suits a paper column; on a 16:9 slide it is a tall
    sliver, so this is the wide equivalent. Each panel keeps its own y-scale
    (the three profiles differ by an order of magnitude, so a shared one would
    flatten two of them), hence per-panel tick labels and one shared y-label.
    """
    import matplotlib.pyplot as plt

    colors = get_cmap_colors(cmap_name="plasma", n=5, start=0.15, end=0.65)
    fg = plt.rcParams.get("text.color", "black")
    fig, axes = plt.subplots(1, 3, figsize=_figsize(FULL_W, 2.9))
    for ax, panel in zip(axes, FIG8_PANELS):
        _pole_flux_panel(ax, panel, colors, fg, title_above=True)
        ax.set_xlabel(r"$E$ [GeV]")
    axes[0].set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ at galactic poles"
                       "\n" r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")

    handles, labels = axes[0].get_legend_handles_labels()
    leg = _style_legend(fig.legend(handles, labels, loc="lower center",
                                   bbox_to_anchor=(0.5, _THEME["legend_y"]),
                                   columnspacing=1.2, ncols=3,
                                   handletextpad=0.4, **LEGEND_KW))
    _fit_bottom_legend(fig, leg, pad=0.6, w_pad=1.2)
    return _save(fig, outdir, "f8_pole_flux_profiles_wide")


def fig13(outdir, emin=10.0, emax=40.0):
    """The two morphology/significance results on one slide.

    Left is the model-independent angular profile (Fig 7 standalone), right the
    per-bin delta-lnL (Fig 2 / Totani Fig. 9). Both are drawn by the SAME panel
    functions the standalone figures use, so the pair can never drift from its
    parts. Panel letters are carried here and only here -- the standalone
    renders stay untitled.
    """
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=_figsize(FULL_W, 3.0))
    _angular_profile_panel(axL, emin=emin, emax=emax,
                           title="(a) Angular profile")
    _likelihood_profiles_panel(axR, title="(b) Likelihood profiles")
    fig.tight_layout(w_pad=2.0)
    return _save(fig, outdir, "f_morphology_and_likelihood")


# --------------------------------------------------------------------------- #
#  Fig 8 : cellwise NFW-rho^2 pole spectrum, on its own
# --------------------------------------------------------------------------- #
def fig8(outdir):
    """The cellwise (Totani-replication) NFW-rho^2 halo spectrum, single panel.

    Fig 1 shows this series as one of three overlaid on each of three density
    profiles. This is the same measurement isolated: the rho^2 profile, cellwise
    fit only, disk excluded -- for use where the three-way comparison is not the
    point and the reader just needs the spectrum itself.

    Deliberately shares Fig 1's conventions (marker, open face, zero line,
    20 GeV guide, log-x window, y-limits and ticks) so the two read as the same
    measurement at different levels of detail, and the panel can be lifted into
    a talk beside Fig 1 without a visual jump.
    """
    import matplotlib.pyplot as plt

    panel = next(p for p in FIG8_PANELS if int(p["fig"]) == 6)   # NFW-rho^2
    fg = plt.rcParams.get("text.color", "black")

    h = load_halo_spectrum(cellwise_dir(int(panel["fig"])),
                           nfw_label=panel["label"], central_stat="f_p50")

    fig, ax = plt.subplots(figsize=_figsize(COL_W, 2.8))
    ax.axhline(0.0, color=_ink("0.5"), lw=0.7, ls=":")
    ax.axvline(20.0, color=_ink("0.8"), lw=0.8, ls=":")
    _plot_halo_points(ax, h, color=fg, marker="o",
                      label="Cellwise, disk excl.", mfc=_ink("white"), zorder=7)

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(*panel["ylim"])
    ax.set_yticks(panel["yticks"])
    ax.text(0.04, 0.88, panel["title"], transform=ax.transAxes)
    _mathtext_sci_axis(ax, "y")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(" ")  # reserve the gutter; real label drawn stacked below
    stacked_ylabel(ax, YLAB_MAIN, YLAB_UNITS, x=-0.17, dx=0.055, y=0.5)
    # Lower right: the spectrum peaks near 20 GeV and the upper-left corner
    # carries the profile label, so that is the only quadrant the data leaves
    # clear across the whole x-range.
    _style_legend(ax.legend(loc="lower right", borderpad=0.4,
                            labelspacing=0.3, **LEGEND_KW))
    ax.grid(alpha=0.13, which="both")
    fig.tight_layout()
    return _save(fig, outdir, "f_cellwise_nfw_rho2")


# --------------------------------------------------------------------------- #
#  Fig 19 : posterior template covariance
# --------------------------------------------------------------------------- #
POSTCOV_NPZ = REANALYSIS_RESULTS / "posterior-covariance" / "posterior_covariance.npz"

# The three templates that visibly move with the halo (systematics/
# posterior_covariance.py LARGE_SCALE, minus the members that stay near zero);
# everything else is grouped into one "other templates" trace in panel (b) so
# the legend and the plot both stay legible.
_POSTCOV_BIG3 = ("ICS", r"bubbles $+$", "gas")


def _load_postcov():
    if not POSTCOV_NPZ.exists():
        raise SystemExit(
            f"missing {POSTCOV_NPZ} -- run:\n"
            "  python systematics/posterior_covariance.py"
        )
    d = np.load(POSTCOV_NPZ, allow_pickle=True)
    return dict(
        corr=np.asarray(d["corr"], float),
        energies_gev=np.asarray(d["energies_mev"], float) / 1000.0,
        disp=[str(x) for x in d["display_labels"]],
        halo_idx=int(d["halo_index"]),
    )


def _slide_wide(w, h):
    """On the slide styles, widen a figure to a 16:9 canvas at the same height.

    Talk figures land on a 16:9 frame, and a 4:3-ish paper figure dropped onto
    one either letterboxes or gets scaled up until its type is oversized.
    Widening rather than shortening keeps the panel height -- and so the data's
    vertical resolution and the type-to-panel ratio -- exactly as the paper
    version has it; the extra width goes to the margins and legends. Print is
    untouched.
    """
    return (max(w, h * 16.0 / 9.0), h) if _SLIDE_STYLE else (w, h)


def _cell_luminance(im, value):
    """Relative luminance of the colour `im`'s colormap assigns to `value`.

    Used to pick readable text over a heatmap cell. Independent of the active
    plot style: the mappable carries its own norm and colormap, so this gives
    the same answer in print and on a dark slide -- which is the point.
    """
    r, g, b = im.cmap(im.norm(float(value)))[:3]
    def _lin(u):
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _postcov_heatmap_panel(ax, ctx, *, peak_k, title=True):
    """(a) 9x9 posterior correlation matrix at the peak bin on the house plasma
    ramp, scaled over the full r in [-1, +1] so the colour axis is symmetric
    about zero even though the ramp itself is sequential. Because plasma is
    monotonic in lightness, r still reads off the page as brightness alone:
    dark blue-purple at -1, mid magenta at 0, yellow at +1.

    Normalize, not TwoSlopeNorm: with vmin/vmax symmetric about vcenter=0 the
    two are numerically identical, and the plain one does not imply a diverging
    scale that this colormap does not have."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    disp = ctx["disp"]
    n = len(disp)
    m = ctx["corr"][peak_k]
    norm = Normalize(vmin=-1.0, vmax=1.0)
    # Diverging, built from the plasma ramp's own colours: plasma(0.0) at
    # r = -1, white at r = 0, plasma(0.5) -- pink -- at r = +1. A correlation
    # matrix has a meaningful zero, and on a straight sequential ramp r = 0
    # landed mid-scale where nothing marked it; white puts the uncorrelated
    # bulk visibly at rest and lets sign read as hue, magnitude as saturation.
    #
    # NOT floor-lifted. The style brightness floor exists so thin ink -- lines,
    # markers -- clears a dark background; a heatmap fills whole cells and
    # covers the background completely, so it gains nothing and pays a third of
    # the ramp for it. Floored, the negative half collapsed into orange-magenta
    # and -0.93 was barely separable from -0.48 on a slide. Filled maps
    # elsewhere in the repo pass a raw ramp for the same reason
    # (make_cellwise_binning_figure.py).
    from matplotlib.colors import LinearSegmentedColormap
    _plasma = plt.get_cmap("plasma")
    cmap = LinearSegmentedColormap.from_list(
        "plasma_corr", [_plasma(0.0), "#ffffff", _plasma(0.5)])
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(disp, rotation=45, ha="right", fontsize=ANNOT_FS)
    ax.set_yticklabels(disp, fontsize=ANNOT_FS)
    # On a slide, annotate only the halo row and column. All 81 numbers is the
    # opposite of "minimal annotation": at slide type size they abut each other
    # and none of them is readable from the back of a room anyway. The halo
    # column carries what the slide is actually about (halo-ICS -0.48,
    # halo-bubbles+ -0.37, halo-iso/PS ~0), and the colours still carry the
    # rest of the matrix. The COLUMN only, not the row as well: the matrix is
    # symmetric so the row repeats the same nine numbers, and across a row they
    # abut at slide type size where down a column they have room. Print keeps
    # the full set.
    halo_idx = ctx["halo_idx"]
    for i in range(n):
        for j in range(n):
            v = m[i, j]
            if not np.isfinite(v):
                continue
            if _SLIDE_STYLE and j != halo_idx:
                continue
            # "-0.00" is a signed-zero artefact of %.2f on a small negative
            # (PS is -0.0047 at the peak bin) and reads as a defect rather
            # than as "consistent with zero". Print an unsigned 0.00 instead.
            txt = f"{v:.2f}"
            if txt == "-0.00":
                txt = "0.00"
            # Contrast against the CELL, not against the theme. _ink() inverts
            # with the theme, so on a dark slide _ink("k") became WHITE and
            # every light cell had white text on it. Reading the luminance back
            # out of the mappable is theme-independent and, more to the point,
            # colormap-independent: it re-derives the threshold from whatever
            # ramp is in force, so it stayed correct across the RdBu -> plasma
            # change without being retuned.
            # Pick whichever of black/white actually wins on WCAG contrast
            # rather than comparing luminance to a hand-tuned threshold. The
            # threshold was 0.55, tuned for a colormap whose midtones were
            # dark; this one passes through WHITE at r = 0, so mid-|r| cells
            # are pale and 0.55 put white text on them -- r = +0.5 scored
            # 2.05:1, below any legibility floor. Deriving the choice makes it
            # correct for whatever ramp is in force instead of for one of them.
            _lum = _cell_luminance(im, v)
            _on_white = (_lum + 0.05) / 0.05          # black text on this cell
            _on_black = 1.05 / (_lum + 0.05)          # white text on this cell
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=max(ANNOT_FS - 1.0, 6.0),
                    color="white" if _on_black > _on_white else "black")
    ax.set_xticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.grid(which="minor", color=_ink("white"), lw=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", bottom=False, left=False, top=False, right=False)
    if title:
        ax.set_title(f"(a) Posterior correlation, {ctx['energies_gev'][peak_k]:.2f} GeV peak (k{peak_k:02d})")
    # Colourbar as an INSET of the heatmap axes, not `ax=ax`. The latter sizes
    # itself from the axes position at creation time, so once fig19 anchors the
    # heatmap north (its aspect="equal" box having been shrunk and centred) the
    # colourbar stayed behind at the old centred position and no longer lined
    # up with the map. An inset is expressed in the parent's own axes
    # fractions, so it follows the box wherever the anchor puts it.
    cax = ax.inset_axes([1.045, 0.0, 0.045, 1.0])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("correlation coefficient, $r$")
    cb.ax.tick_params(labelsize=ANNOT_FS)
    return im


def _postcov_energy_panel(ax, ctx, *, title=True):
    """(b) the halo's correlation with every other template vs energy, with
    the 10-40 GeV signal band shaded. The three templates that move with the
    halo (ICS, bubbles+, gas) are drawn in colour; the rest (Loop I A/B, iso,
    PS, bubbles-) stay near zero throughout and are grouped into one grey
    trace so the panel does not need an 8-entry legend."""
    from matplotlib.lines import Line2D

    disp = ctx["disp"]
    halo_idx = ctx["halo_idx"]
    E = ctx["energies_gev"]
    corr = ctx["corr"]
    cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.10, end=0.82)
    big3_style = {
        "ICS": dict(color=cols[0], marker="o"),
        r"bubbles $+$": dict(color=cols[3], marker="s"),
        "gas": dict(color=cols[5], marker="^"),
    }
    _signal_band(ax)
    ax.axhline(0.0, color=_ink("0.5"), lw=GUIDE_LW, ls=":")
    other_drawn = False
    for j, lab in enumerate(disp):
        if j == halo_idx:
            continue
        y = corr[:, halo_idx, j]
        if not np.any(np.isfinite(y)):
            continue
        if lab in big3_style:
            st = big3_style[lab]
            ax.plot(E, y, color=st["color"], marker=st["marker"], ls="-",
                    lw=SERIES_LW, ms=SERIES_MS, mfc=st["color"], mec=st["color"], mew=0.7,
                    zorder=6, label="_nolegend_")
        else:
            # Thinner and paler than the three named traces, so the bundle
            # reads as background context. It was lw 0.9 / alpha 0.85, close
            # enough to the named traces' weight to be mistaken for a fourth
            # and fifth series.
            ax.plot(E, y, color=_ink("0.72"), lw=0.7, ls="-",
                    alpha=0.60, zorder=2, label="_nolegend_")
            other_drawn = True
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(-0.68, 0.18)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(r"corr. with halo normalisation, $r$")
    if title:
        ax.set_title("(b) Halo correlation vs energy")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(alpha=0.13, which="both")

    handles = [Line2D([0], [0], color=st["color"], lw=1.9, marker=st["marker"],
                      ms=4.6, mfc=st["color"], mec=st["color"])
               for st in big3_style.values()]
    labels = list(big3_style.keys())
    if other_drawn:
        handles.append(Line2D([0], [0], color=_ink("0.72"), lw=0.9, alpha=0.60))
        labels.append("other templates")
    # Upper right: every trace stays below r~0.08 across the WHOLE energy
    # range (the only positive excursion is iso's +0.08 at the low-energy
    # edge), so this corner is clear of data everywhere, unlike either lower
    # corner where ICS/bubbles+/gas all sit.
    _style_legend(ax.legend(handles, labels, loc="upper right", borderpad=0.35,
                            labelspacing=0.3, fontsize=ANNOT_FS,
                            handletextpad=0.5, **LEGEND_KW), zorder=20)


def _hardic_family_colour():
    """One house-plasma tone for the whole hard-IC injected-population family.

    Taken from the same get_cmap_colors ramp every other figure here uses, and
    resolved at DRAW time so it picks up the active style's colormap floor.
    This was a hand-picked steel hex before, which sat outside the palette.

    Index 0 -- the ramp's far end -- deliberately: plasma is a single warm
    sweep, so an index ADJACENT to the NFW references (cols[2..5] in fig21)
    renders the two families the same colour and they stop being separable.
    Putting the injected family at one end and the references at the other is
    what buys family separation without leaving the scheme.

    The deliberately unphysical CMB-only bracket takes this SAME colour and is
    separated by dash pattern and alpha instead of by hue -- so the family
    reads as one family, and "ruled out" is carried by line style rather than
    by inventing a second colour outside the scheme.
    """
    return get_cmap_colors(cmap_name="plasma", n=6, start=0.15, end=0.85)[0]


def _hardic_angular_ctx(emin=10.0, emax=40.0):
    r"""Ring profile plus every morphology's fitted curve and chi2/nu.

    Reuses morphology/angular_profile_hardic.py wholesale -- its excess_map,
    ring_profile, fit_amp_offset and nfw_los -- so the curves drawn here and
    the chi2/nu it tabulates can never diverge.

    NOTE the residual: that script works from the GLOBAL no-halo fit
    (quick_global/global_fit_results_fig4), whereas fig7 works from the
    PIXELWISE one (pixelwise_mcmc_results_fig4_resid). The hard-IC templates
    were only ever run against the global residual, so the whole contest has
    to sit on that one. The consequence is that the NFW/isotropic chi2/nu here
    (7.09 / 5.06 / 15.39 / 28.80) differ slightly from fig7's (7.0 / 4.9 /
    15.2 / 29.1): same method, different residual. They are NOT interchangeable
    and this figure recomputes all of them on one footing rather than importing
    fig7's numbers.
    """
    from astropy.io import fits
    sys.path.insert(0, str(_HERE / "morphology"))
    import angular_profile_hardic as ah

    hdr = fits.getheader(ah.COUNTS)
    L, B = ah._lonlat(hdr)
    exc, mask, bins = ah.excess_map(ah.FITDIR, emin, emax)
    edges = np.linspace(10.0, 70.0, 13)
    ctr, val, err = ah.ring_profile(exc, mask, L, B, edges)

    entries = []
    for name, p in ((r"NFW $\rho^{2}$", 2.0), (r"NFW $\rho^{2.5}$", 2.5),
                    (r"NFW $\rho^{1}$", 1.0), ("isotropic", None)):
        m = np.ones_like(ctr) if p is None else np.array([ah.nfw_los(x, p) for x in ctr])
        # fit_amp_offset returns (amplitude, offset, chi2/nu) -- the fitted
        # CURVE is A*m + C, not the first element.
        A, C, chi2n = ah.fit_amp_offset(m, val, err)
        entries.append(dict(key=name, label=name, pred=A * m + C, chi2n=chi2n,
                            family="nfw", physical=True))

    missing = []
    for tag in ah.HARDIC_TAGS:
        band = ah.hardic_band_map(tag, bins, psf=True)
        if band is None:
            missing.append(tag)
            continue
        _, mv, _ = ah.ring_profile(band, mask, L, B, edges)
        A, C, chi2n = ah.fit_amp_offset(mv, val, err)
        entries.append(dict(key=tag, label=_HARDIC_LABELS[tag], pred=A * mv + C, chi2n=chi2n,
                            family="hardic", physical=tag != "cmbonly_burst_m"))
    if missing:
        raise SystemExit("missing hard-IC band templates for: " + ", ".join(missing))
    return dict(ctr=ctr, val=val, err=err, entries=entries, emin=emin, emax=emax)


def fig21(outdir, emin=10.0, emax=40.0):
    r"""Angular profile with the hard-IC injected morphologies overlaid.

    Honest-disclosure companion to fig7, NOT a replacement: two injected
    morphologies fit the no-halo residual's run with angle better than
    NFW rho^2 does. One of them (the CMB-only ISRF bracket, 4.78) is the
    deliberately unphysical case that fig20 rules out on energetics -- it is
    drawn but flagged. The other (burst, lambda=6 kpc, 5.29) is physical and
    genuinely beats NFW rho^2 (7.09) on this statistic alone.

    Colour separates the two families: plasma for the NFW/isotropic
    references, one steel tone for every injected population so the family
    reads at a glance, individuals separated by marker.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ctx = _hardic_angular_ctx(emin=emin, emax=emax)
    ctr, val, err = ctx["ctr"], ctx["val"], ctx["err"]

    # Ten curves share this panel, so hue is spent where it does the most work:
    # the SIX injected morphologies each get their own plasma tone, and the four
    # NFW/isotropic references drop to the house neutral ink, separated from
    # each other by line style. Previously the whole injected family shared one
    # colour and was told apart by marker alone, which is what made the panel
    # hard to read. Family membership now reads off saturation (coloured =
    # injected, neutral = reference) instead of the two families competing for
    # slots on a single warm ramp.
    nfw_style = {
        r"NFW $\rho^{2}$":   dict(color=_ink("0.10"), ls="-"),
        r"NFW $\rho^{2.5}$": dict(color=_ink("0.38"), ls=":"),
        r"NFW $\rho^{1}$":   dict(color=_ink("0.55"), ls="--"),
        "isotropic":         dict(color=_ink("0.72"), ls="-."),
    }
    # Ramp end is style-dependent. In print, plasma's top is a pale yellow with
    # almost no contrast against a white page, so the ramp stops at orange. On
    # a dark slide the opposite holds -- yellow is among the most legible tones
    # there, and the style's colormap floor has already compressed the low end,
    # so stopping early would leave all six injected curves bunched in the
    # oranges. Running to 0.95 restores their separation.
    hardic_cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.05,
                                  end=0.95 if _SLIDE_STYLE else 0.80)
    # The deliberately unphysical bracket takes the DARKEST, most muted slot.
    # Assigned in tag order it landed on the brightest tone, so the one curve
    # the figure rules out was the loudest thing on the panel.
    hardic_style = {
        "cmbonly_burst_m": dict(color=hardic_cols[0], marker="P"),
        "burst_s":         dict(color=hardic_cols[1], marker="o"),
        "burst_m":         dict(color=hardic_cols[2], marker="s"),
        "burst_l":         dict(color=hardic_cols[3], marker="^"),
        "cont":            dict(color=hardic_cols[4], marker="v"),
        "adv":             dict(color=hardic_cols[5], marker="D"),
    }

    fig, ax = plt.subplots(figsize=_figsize(FULL_W, 3.4))
    ax.errorbar(ctr, val, yerr=err, fmt="o", color=_ink("k"), ms=4, capsize=2,
                lw=1.2, zorder=9, label=r"excess (data $-$ no-halo)")

    for e in sorted(ctx["entries"], key=lambda d: d["chi2n"]):
        lab = rf"{e['label']} ($\chi^2/\nu$={e['chi2n']:.2f})"
        if e["family"] == "nfw":
            st = nfw_style[e["key"]]
            ax.plot(ctr, e["pred"], st["ls"], color=st["color"], lw=EMPH_LW,
                    zorder=7, label=lab)
        else:
            phys = e["physical"]
            st = hardic_style[e["key"]]
            c = st["color"]
            # The unphysical bracket keeps its own colour but is dashed with an
            # open marker, so it is identifiable as a member of the family and
            # still flagged as the ruled-out case.
            ax.plot(ctr, e["pred"], ls="-" if phys else (0, (5, 2)),
                    color=c, lw=SERIES_LW,
                    marker=st["marker"], ms=SERIES_MS, markevery=2,
                    mfc=c if phys else _ink("white"), mec=c, mew=0.9,
                    zorder=5, label=lab)

    ax.axhline(0.0, color=_ink("0.6"), lw=0.7, ls=":")
    _mathtext_sci_axis(ax, "y")
    ax.set_xlabel(r"angle from GC, $\psi$ [deg]")
    # Plain two-line ylabel rather than stacked_ylabel: the stacked form places
    # two ROTATED texts a fixed axes-fraction apart, and on this figure's wide
    # slide canvas (the legend sits outside the axes, so the axes itself is
    # comparatively narrow) that gap collapsed and the two lines overprinted.
    ax.set_ylabel("excess (data$-$model)\n[counts, arb.]")
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    # Outside the axes, right: eleven entries ordered best-to-worst chi2/nu is
    # the readable form of the ranking this figure exists to disclose, and it
    # buries the data if placed inside the panel.
    leg = _style_legend(ax.legend(loc="center left", bbox_to_anchor=(1.015, 0.5),
                                  borderpad=0.4, labelspacing=0.34,
                                  fontsize=ANNOT_FS - 0.5, handletextpad=0.6,
                                  handlelength=2.4, **LEGEND_KW), zorder=20)
    leg.get_frame().set_alpha(0.95)
    # Footnote below the axes, not inside: at y=0.03 it lay across the
    # isotropic line and the outer data points. "Fig.~20" also rendered its
    # LaTeX tilde literally here (this is matplotlib text, not LaTeX).
    fig.text(0.012, 0.015,
             r"$^{*}$deliberately unphysical (ruled out by energetics, Fig. 20)",
             ha="left", va="bottom", fontsize=ANNOT_FS - 1, color=_ink("0.35"))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    print("  hard-IC angular chi2/dof: " + ", ".join(
        f"{e['key']}={e['chi2n']:.2f}" for e in sorted(ctx["entries"],
                                                       key=lambda d: d["chi2n"])))
    return _save(fig, outdir, "f_angular_profile_morphology_hardic")


HARDIC_NOHALO_DIR = _HERE / "figures" / "plots_hardic_only"


def _hardic_nohalo_ctx():
    """Halo-removed ("C" run) hard-IC spectra plus the halo baseline.

    Both come from systematics/hardic_nohalo_spectra.py, which drives the same
    make_plots_from_mcmc path as every other global table -- so the injected
    spectra and the halo they are being compared against are on one footing.
    """
    def _read(sub, col):
        t = HARDIC_NOHALO_DIR / sub / "mcmc_global_f_ml.txt"
        if not t.exists():
            raise SystemExit(
                f"missing {t} -- run:\n"
                "  python Totani_reanalysis/systematics/hardic_nohalo_spectra.py")
        a = np.genfromtxt(t, names=True, comments="#")
        if col not in (a.dtype.names or ()):
            raise SystemExit(f"{t} has no '{col}' column")
        names = a.dtype.names or ()
        # The per-bin posterior spread on the template normalisation, which
        # make_plots_from_mcmc already writes alongside every component. Fall
        # back to zeros rather than failing: a table without them still plots,
        # it just plots without bars, which is honest about what it has.
        def _err(suffix):
            k = f"{col}_{suffix}"
            return (np.asarray(a[k], float) if k in names
                    else np.zeros(len(a["E_GeV"])))
        return (np.asarray(a["E_GeV"], float), np.asarray(a[col], float),
                _err("err_lo"), _err("err_hi"))

    E, halo, halo_lo, halo_hi = _read("_baseline_halo", "nfw")
    series = []
    for tag in _HARDIC_TAGS:
        Et, y, lo, hi = _read(tag, f"hardic_{tag}")
        series.append(dict(tag=tag, E=Et, y=y, lo=lo, hi=hi,
                           label=_HARDIC_LABELS[tag],
                           physical=tag != "cmbonly_burst_m"))
    return dict(E=E, halo=halo, halo_lo=halo_lo, halo_hi=halo_hi, series=series)


def fig22(outdir):
    r"""What the injected hard-IC population recovers once the halo is removed.

    The "C" runs: the NFW halo is taken OUT of the template stack and the
    hard-IC component is left free to absorb the 20 GeV excess by itself. If an
    injected cosmic-ray population were really behind the excess, this is where
    it would show up at full strength.

    It does not. Across the 10-40 GeV band the five physical morphologies reach
    only 6-22% of the measured halo's peak flux; only the deliberately
    unphysical CMB-only bracket gets to 56%, and fig20 rules that out on
    energetics. Companion to fig20 (which makes the same point in likelihood)
    and fig21 (which makes it in angular morphology) -- this is the same
    statement in flux.
    """
    import matplotlib.pyplot as plt

    ctx = _hardic_nohalo_ctx()
    cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.05,
                           end=0.95 if _SLIDE_STYLE else 0.80)
    # Same assignment as fig21: darkest/most muted slot to the unphysical
    # bracket, so the one case the analysis rules out is not the loudest curve.
    order = ["cmbonly_burst_m", "burst_s", "burst_m", "burst_l", "cont", "adv"]
    colour = {t: cols[i] for i, t in enumerate(order)}
    marker = {"cmbonly_burst_m": "P", "burst_s": "o", "burst_m": "s",
              "burst_l": "^", "cont": "v", "adv": "D"}

    # PRINT: COL_W, because this is a one-column PRD figure and the file's
    # convention (see COL_W/FULL_W) is that a figure's width IS its print
    # width, so it is included at scale 1. At the FULL_W*0.60 it used to be it
    # emitted 4.13 in against a 3.4 in \columnwidth -- LaTeX would have shrunk
    # it ~18%, putting its 8 pt annotations at an effective 6.5 pt while every
    # other one-column figure in the paper sat at 8. Matching COL_W makes its
    # type match theirs.
    #
    # SLIDE: unchanged. The wide geometry here was tuned against a 16:9 frame,
    # not a paper column, and COL_W*1.35 would have narrowed the slide version
    # for a reason that only applies in print.
    #
    # Height 4.3/3.9: the legend strip below is a fixed number of POINTS tall,
    # so every extra inch of canvas goes to the panel rather than the key --
    # the plot gets taller without the legend growing with it.
    _w, _h = (FULL_W * 0.60, 4.4) if _SLIDE_STYLE else (COL_W, 3.66)
    fig, ax = plt.subplots(figsize=_figsize(_w, _h))
    _signal_band(ax)
    ax.axhline(0.0, color=_ink("0.5"), lw=GUIDE_LW, ls=":", zorder=1)

    # Marker and error-bar furniture lifted wholesale from f_annih_bb_WW
    # (_overlay_with_residual), which is the house convention for a measured
    # spectrum with per-bin uncertainties. SERIES_MS (4.6) is a LINE-series
    # marker, sized to stay visible while a curve runs through it; a dense
    # error-bar panel wants the smaller 4.0 that figure uses. Slides keep 4.6:
    # marker size is in points and does not scale with the canvas, so the
    # print size would read as undersized from the back of a room.
    _ms = SERIES_MS if _SLIDE_STYLE else 4.0
    _bar = dict(elinewidth=0.85, capsize=2.0, mew=1.0, ls="none")

    # The halo is drawn the same way as everything else -- points with bars --
    # so the comparison is like for like: it is one more per-bin fit, and
    # showing it as a smooth curve implied a continuity the fit never had.
    # It keeps the house "measured data" dress from f_annih_bb_WW (open marker,
    # neutral ink) so it still reads as the reference rather than a seventh
    # competing morphology.
    # ", halo fit" dropped from the label: at COL_W a two-column legend has
    # ~1.6 in per column and the longer wording pushed the legend wider than
    # the axes, which tight bbox then bills back as figure width.
    # Open in print, FILLED on a slide. _ink("white") resolves to the current
    # SURFACE, so on a dark style the open marker's fill is the slide
    # background and the halo survives only as a thin ring -- which vanished
    # everywhere the light signal band was not behind it. The reference series
    # has to be the most legible thing in the panel, so on dark it takes a
    # solid ink fill instead. Print keeps the open marker, which is the house
    # "measured data" dress from f_annih_bb_WW.
    ax.errorbar(ctx["E"], ctx["halo"],
                yerr=np.vstack([ctx["halo_lo"], ctx["halo_hi"]]),
                marker="o", ms=_ms, color=_ink("k"), ecolor=_ink("k"),
                mfc=_ink("k") if _SLIDE_STYLE else _ink("white"),
                mec=_ink("k"), zorder=8, **_bar,
                label="measured halo (NFW $\\rho^{2}$)")

    # Dodged in energy, or all seven land on the same x and their bars
    # overprint into a single smear. geomspace, not linspace, and for the same
    # reason f_annih_bb_WW uses it: on a log axis a constant FACTOR is a
    # constant on-screen shift at every decade. The halo sits at 1.0, in the
    # gap the six leave at the centre. Adjacent bins are ~1.8x apart, so even
    # the full 0.88-1.12 spread is under a third of the inter-bin gap and none
    # of it can be misread as a different energy.
    _ordered = sorted(ctx["series"], key=lambda d: -np.nanmax(d["y"]))
    _shifts = np.geomspace(0.88, 1.12, len(_ordered))
    for _i, s in enumerate(_ordered):
        t = s["tag"]
        phys = s["physical"]
        ax.errorbar(s["E"] * _shifts[_i], s["y"],
                    yerr=np.vstack([s["lo"], s["hi"]]),
                    marker=marker[t], ms=_ms,
                    color=colour[t], ecolor=colour[t],
                    mfc=colour[t] if phys else _ink("white"), mec=colour[t],
                    zorder=5, **_bar, label=s["label"])

    import matplotlib.ticker as mticker
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    # The legend sits outside the axes, so the panel is narrow enough that the
    # log locator drops 10^1 and 10^2 and leaves only the two endpoints.
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
    ax.set_xlabel(XLAB)
    # SLIDE: a plain three-line label. A rotated label's LENGTH runs along the
    # y axis, and at slide type size the two-line version was longer than the
    # panel is tall and overflowed the canvas. Wrapping trades vertical length
    # for width, which is the dimension that has room there. Plain rather than
    # stacked_ylabel for the reason fig21 records: the stacked form spaces two
    # rotated texts a fixed AXES-fraction apart, which is fragile on the wide
    # slide canvas.
    #
    # PRINT: stacked_ylabel, exactly as fig8/fig13 use it for this same label.
    # The plain two-line form spent 0.98 in of a 3.4 in column on the left
    # margin against fig7's 0.60 in, which is most of why this panel's axes was
    # half an inch narrower than every other one-column figure's. Stacking the
    # units in smaller type beside the main label reclaims that width.
    if _SLIDE_STYLE:
        ax.set_ylabel("$E^2\\,\\mathrm{d}N/\\mathrm{d}E$ at\n"
                      "galactic poles\n"
                      "[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    else:
        ax.set_ylabel(" ")  # reserve the gutter; real label drawn stacked below
        stacked_ylabel(ax, YLAB_MAIN, YLAB_UNITS, x=-0.17, dx=0.055, y=0.5)
    _mathtext_sci_axis(ax, "y")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(alpha=0.13, which="both")

    # Legend BELOW the axes, two columns. Seven entries in two columns is four
    # rows, which is shallower than the panel is tall, so the plot keeps the
    # full canvas width instead of surrendering a third of it to a side legend.
    handles_, labels_ = ax.get_legend_handles_labels()
    leg = _style_legend(fig.legend(handles_, labels_, loc="lower center",
                                   bbox_to_anchor=(0.5, 0.0), ncol=2,
                                   borderpad=0.4, labelspacing=0.30,
                                   columnspacing=1.2,
                                   fontsize=max(ANNOT_FS - 1.0, 6.0),
                                   handletextpad=0.5, handlelength=2.0,
                                   **LEGEND_KW), zorder=20)
    leg.get_frame().set_alpha(0.95)
    # No in-figure footnote for the asterisk on the CMB-only entry. That marker
    # is explained in the LaTeX caption, which is how the same convention is
    # already handled on fig5 -- its legend carries "no bubbles$^{*}$" /
    # "GIEM$^{*}$" and its own footnote text sits commented out for exactly
    # this reason. Removed in EVERY style, print and slide alike, so the two do
    # not drift apart on a detail the caption already covers.
    #
    # Lay the axes out first, then hang the legend off the MEASURED bottom of
    # the axes (x-label included). Computing a rect from the legend's height
    # alone leaves a gap whose size depends on how far the x-label happens to
    # reach -- tight_layout pads the axes down to the rect but never pulls the
    # legend up to meet them.
    # Two different gaps, not one. `gap` is the legend's clearance from the
    # canvas floor; `lead` is the space between the bottom of the x-axis label
    # and the top of the legend, which wants to be visibly larger so the key
    # reads as a separate block rather than as another line of the axis
    # furniture. Sharing one constant made the legend sit tight under the
    # x-label.
    gap = 0.018
    lead = 0.045
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_h = fig.get_figheight() * fig.dpi
    leg_h = leg.get_window_extent(rend).height / fig_h
    fig.tight_layout(rect=(0, min(leg_h + lead + gap, 0.45), 1, 1))
    fig.canvas.draw()
    ax_bottom = ax.get_tightbbox(rend).y0 / fig_h
    leg_y = max(ax_bottom - leg_h - lead, gap)
    leg.set_bbox_to_anchor((0.5, leg_y), transform=fig.transFigure)

    band = (ctx["E"] >= HALO_BAND[0]) & (ctx["E"] <= HALO_BAND[1])
    hpk = np.nanmax(ctx["halo"][band])
    print("  halo-removed band peak / halo peak: " + ", ".join(
        f"{s['tag']}={np.nanmax(s['y'][band]) / hpk:.2f}" for s in ctx["series"]))
    return _save(fig, outdir, "f_hardic_nohalo_spectra")


def fig19(outdir):
    r"""Posterior template covariance.

    (a) the 9x9 posterior correlation matrix of the fitted template
    normalisations at the 20.76 GeV peak bin (k05); (b) the halo's
    correlation with every other template as a function of energy, with the
    10-40 GeV signal band shaded. Halo <-> ICS is the dominant degeneracy
    throughout (r ~ -0.5 to -0.6); halo <-> iso and halo <-> PS are
    essentially zero everywhere, which is quantitative support for the
    existing claim that the excess is not an isotropic/IGRB
    mis-normalisation. See systematics/posterior_covariance.py (this figure
    reworks that script's two draft plots into one publication panel pair).
    """
    import matplotlib.pyplot as plt
    ctx = _load_postcov()
    peak_k = int(np.argmin(np.abs(ctx["energies_gev"] - 20.76)))
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=_figsize(*_slide_wide(FULL_W, 4.5)),
        gridspec_kw=dict(width_ratios=[1.05, 1.35]),
    )
    _postcov_heatmap_panel(axL, ctx, peak_k=peak_k)
    _postcov_energy_panel(axR, ctx)
    fig.tight_layout(w_pad=2.2)
    # AFTER tight_layout, which resets it: aspect="equal" shrinks the heatmap's
    # axes box to the data aspect and, with the default anchor "C", centres the
    # shrunk box vertically -- leaving panel (a)'s title floating well below
    # panel (b)'s. Anchoring north puts the two boxes' tops, and so the two
    # titles, on one line.
    axL.set_anchor("N")
    # Match panel (b)'s height to panel (a)'s. The heatmap is aspect="equal",
    # so its box is shrunk to a square at DRAW time and ends up shorter than
    # the energy panel beside it, leaving the two bottoms badly out of line.
    # The heatmap wins: it cannot be stretched without making the correlation
    # cells non-square, whereas (b) has no aspect constraint and can simply be
    # told to occupy the same vertical span.
    #
    # get_position(original=False) is the ACTIVE box -- the post-apply_aspect
    # one -- and it only exists after a draw, hence the explicit draw here.
    fig.canvas.draw()
    pL = axL.get_position(original=False)
    pR = axR.get_position(original=False)
    axR.set_position([pR.x0, pL.y0, pR.width, pL.height])
    return _save(fig, outdir, "f_posterior_template_covariance")


# --------------------------------------------------------------------------- #
#  Fig 20 : hard-IC injected-population contest
# --------------------------------------------------------------------------- #
_HARDIC_TAGS = ("burst_s", "burst_m", "burst_l", "cont", "adv", "cmbonly_burst_m")
_HARDIC_LABELS = {
    "burst_s": r"burst, $\lambda=1.5$ kpc",
    "burst_m": r"burst, $\lambda=3$ kpc",
    "burst_l": r"burst, $\lambda=6$ kpc",
    "cont": "continuous injection",
    "adv": "advected",
    "cmbonly_burst_m": r"CMB-only ISRF bracket$^{*}$",
}
# Milky Way rest-mass energy, M ~ 1e12 Msun: the scale against which the
# CMB-only bracket's runaway E_inj is quoted ("~230x the Milky Way's
# rest-mass energy" at its worst bin).
_MW_MC2_ERG = 1.79e66


def _hardic_module():
    """Import systematics/hardic_report.py so this figure reuses its own
    loaders (_load_run / _find_index / _read_einj_ref) rather than
    duplicating them -- the numbers can then never drift from the
    write-up that quotes them."""
    sys.path.insert(0, str(_HERE / "systematics"))
    import hardic_report as hr
    return hr


def _hardic_metrics(tags=_HARDIC_TAGS):
    """Band-averaged (12.29/20.76/35.06 GeV) headline metrics per tag:
    theta_halo(B)/theta_halo(A), the halo-survival statistic
    2*sum[lnL(B)-lnL(C)], the morphology-contest margin -2*sum[lnL(C)-lnL(A)]
    (positive: halo alone beats the injected population alone by this much),
    and the largest implied injection energy at any bin (in erg)."""
    hr = _hardic_module()
    band_bins = set(hr.DEFAULT_BAND_BINS)
    A = hr._load_run(hr.DEFAULT_BASELINE_DIR, hr.DEFAULT_NBINS)
    if not A:
        raise SystemExit(f"missing hard-IC baseline (A) fit results under {hr.DEFAULT_BASELINE_DIR}")
    rows = []
    for tag in tags:
        B = hr._load_run(os.path.join(hr.DEFAULT_SYSTEMATICS_DIR, f"hardic_{tag}"), hr.DEFAULT_NBINS)
        C = hr._load_run(os.path.join(hr.DEFAULT_HARDIC_ONLY_DIR, tag), hr.DEFAULT_NBINS)
        if not B or not C:
            raise SystemExit(f"missing hard-IC B/C fit results for tag={tag}")
        common_bc = sorted(set(B) & set(C))
        band_bc = [k for k in common_bc if k in band_bins]
        halo_survival = 2.0 * sum(B[k]["loglike_ml"] - C[k]["loglike_ml"] for k in band_bc)
        common_ca = sorted(set(C) & set(A))
        band_ca = [k for k in common_ca if k in band_bins]
        morph_contest = 2.0 * sum(C[k]["loglike_ml"] - A[k]["loglike_ml"] for k in band_ca)

        ratios = []
        for k in band_bins:
            if k not in B or k not in A:
                continue
            i_hB = hr._find_index(B[k]["labels"], ["nfw"])
            i_hA = hr._find_index(A[k]["labels"], ["nfw"])
            if i_hB is None or i_hA is None:
                continue
            th_B = B[k]["theta_ml"][i_hB]
            th_A = A[k]["theta_ml"][i_hA]
            if np.isfinite(th_A) and th_A != 0:
                ratios.append(th_B / th_A)
        ratio = float(np.mean(ratios)) if ratios else np.nan

        einj_ref = hr._read_einj_ref(hr.DEFAULT_TEMPLATES_DIR, tag)
        max_einj = 0.0
        for k in sorted(B):
            i_hc = hr._find_index(B[k]["labels"], ["hardic"])
            if i_hc is None:
                continue
            theta_hc = B[k]["theta_ml"][i_hc]
            if np.isfinite(theta_hc):
                max_einj = max(max_einj, float(theta_hc) * einj_ref)

        rows.append(dict(
            tag=tag, ratio=ratio,
            halo_survival=halo_survival,
            morph_margin=-morph_contest,   # positive: halo alone beats hard-IC alone
            morph_contest=morph_contest,   # as tabulated in the write-up (negative)
            max_einj=max_einj,
        ))
    return rows


def _sci_tex(x, sig=1):
    """Format a positive float as LaTeX scientific notation, e.g. 4.2e68 ->
    '4.2\\times10^{68}'."""
    if not np.isfinite(x) or x <= 0:
        return "--"
    exp = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** exp
    return rf"{mant:.{sig}f}\times10^{{{exp}}}"


def fig20(outdir):
    r"""Hard-IC injected-population contest.

    Per injected-population morphology: (a) the halo amplitude is unchanged
    (theta(B)/theta(A) ~ 1.000, band-averaged over 12-35 GeV) for every
    physical case; (b) forced to stand in for the halo, the injected
    population loses badly -- the halo alone beats it alone by
    -2*sum[lnL(C)-lnL(A)] ~ 610-688, and the halo cannot be replaced even
    when both are free simultaneously (2*sum[lnL(B)-lnL(C)] ~ 611-688). The
    sixth case, cmbonly_burst_m, is a deliberately unphysical ISRF bracket
    (inverse-Compton on the CMB alone, no starlight or FIR) -- it is the only
    one that dents the halo (ratio 0.812) and it still loses to the halo,
    just by a much smaller margin, at the cost of an implied injection
    energy up to ~230x the Milky Way's rest-mass energy. See
    systematics/hardic_report.py (this figure reuses that script's
    loaders).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = _hardic_metrics()
    by_tag = {r["tag"]: r for r in rows}
    physical = ["burst_s", "burst_m", "burst_l", "cont", "adv"]
    y_of = {tag: 5 - i for i, tag in enumerate(physical)}   # 5,4,3,2,1
    y_cmb = -0.6
    y_of["cmbonly_burst_m"] = y_cmb
    order = physical + ["cmbonly_burst_m"]

    cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.12, end=0.85)
    col_survival, col_margin = cols[1], cols[4]
    col_cmb_survival, col_cmb_margin = _ink("0.55"), _ink("0.78")
    halo_col = _dsph_tension_halo_colour()

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=_figsize(*_slide_wide(FULL_W, 4.6)),
        gridspec_kw=dict(width_ratios=[0.85, 1.55]),
        sharey=True,
    )

    # Shared "unphysical bracket" backdrop on both panels: a light band behind
    # the cmbonly row plus a divider, so the deliberately-unphysical case
    # reads as visually separated from the five physical morphologies before
    # a reader parses any label.
    for ax in (axA, axB):
        # 0.75, not 0.90: theme_ink INVERTS greyscale, so 0.90 becomes grey
        # 0.10 on the dark styles -- darker than the #111217 slide background,
        # i.e. the "unphysical" backdrop and its legend swatch both vanished
        # exactly where this figure is the headline slide. 0.75 gives a light
        # grey in print and a subtly lighter-than-background band on a slide.
        ax.axhspan(y_cmb - 0.55, y_cmb + 0.55, color=_ink("0.75"), alpha=0.6, zorder=0)
        ax.axhline(0.25, color=_ink("0.6"), lw=0.8, ls=(0, (4, 2)), zorder=1)

    # --- (a) halo amplitude ratio, drawn as a deviation from theta(B)/theta(A)=1.
    # A bar alone is imperceptible for the five physical cases (deviation
    # <0.001), so a marker at the tip anchors every row with a visible mark
    # regardless of how small the shift is -- the bar length then carries the
    # magnitude for the one case (cmbonly) where it is large enough to show.
    for tag in order:
        r = by_tag[tag]
        y = y_of[tag]
        is_cmb = tag == "cmbonly_burst_m"
        c = _ink("0.55") if is_cmb else halo_col
        axA.barh(y, r["ratio"] - 1.0, left=1.0, height=0.62,
                color=c, edgecolor=_ink("k"), linewidth=0.6,
                hatch="///" if is_cmb else None, zorder=5)
        tip = r["ratio"]
        # White halo under the marker. For the five physical cases the ratio is
        # 1.000 to three decimals, so the marker lands exactly on the
        # theta(B)/theta(A)=1 reference line and, drawn flat, read as a tick on
        # that line -- i.e. as an ABSENT measurement rather than as the
        # headline result that the halo is untouched. The halo separates the
        # datum from the line it happens to coincide with.
        axA.plot(tip, y, "o", ms=8.6, color=_ink("white"), mec=_ink("white"),
                 mew=0.0, zorder=6)
        axA.plot(tip, y, "o", ms=6.4, color=c, mec=_ink("k"), mew=0.9, zorder=7)
        ha, dx = ("left", 0.009) if tip >= 1.0 else ("right", -0.009)
        axA.text(tip + dx, y, f"{tip:.3f}", ha=ha, va="center",
                 fontsize=max(ANNOT_FS - 0.5, 6.0), color=_ink("k"), zorder=6)
    axA.axvline(1.0, color=_ink("0.45"), lw=0.9, ls=":", zorder=4)
    axA.set_xlim(0.75, 1.09)
    axA.set_ylim(y_cmb - 0.75, 5.65)
    axA.set_xlabel(r"halo amplitude ratio, $\theta_{\rm halo}({\rm B})/\theta_{\rm halo}({\rm A})$")
    axA.set_title("(a) Halo amplitude")
    axA.set_yticks([y_of[t] for t in order])
    axA.set_yticklabels([_HARDIC_LABELS[t] for t in order], fontsize=ANNOT_FS)
    # left=False: the y axis is categorical (morphology names), so its tick
    # marks carry no information -- and pointing them inward put a short dash
    # immediately left of the cmbonly row's "0.812" label, which read as a
    # minus sign on a positive ratio. Dropping the marks removes the ambiguity
    # rather than trying to dodge it with offsets.
    axA.tick_params(direction="in", which="both", top=True, right=True,
                    labelleft=True, left=False)
    axA.grid(alpha=0.13, axis="x")

    # --- (b) likelihood forest: halo survival vs halo-alone margin
    dy = 0.19
    for tag in order:
        r = by_tag[tag]
        y = y_of[tag]
        is_cmb = tag == "cmbonly_burst_m"
        cs = col_cmb_survival if is_cmb else col_survival
        cm = col_cmb_margin if is_cmb else col_margin
        hatch = "///" if is_cmb else None
        axB.barh(y + dy, r["halo_survival"], height=2 * dy * 0.85, color=cs,
                 edgecolor=_ink("k"), linewidth=0.5, hatch=hatch, zorder=5)
        axB.barh(y - dy, r["morph_margin"], height=2 * dy * 0.85, color=cm,
                 edgecolor=_ink("k"), linewidth=0.5, hatch=hatch, zorder=5)
    axB.axvline(0.0, color=_ink("k"), lw=0.8)
    axB.set_xlim(0.0, 780.0)
    axB.set_xlabel(r"signal-band $2\,\Delta\ln L$ (12$-$35 GeV)")
    axB.set_title("(b) Injected population vs halo")
    axB.tick_params(direction="in", which="both", top=True, right=True,
                    labelleft=False, left=False)
    axB.grid(alpha=0.13, axis="x")

    # Energetics callout on the cmbonly row only -- the one number the task
    # asks to surface, kept off the five physical rows so it cannot clutter.
    r_cmb = by_tag["cmbonly_burst_m"]
    axB.annotate(
        rf"$E_{{\rm inj}}$ up to ${_sci_tex(r_cmb['max_einj'])}$ erg"
        "\n" rf"($\sim$230$\times\,M_{{\rm MW}}c^2$) $-$ ruled out",
        xy=(max(r_cmb["halo_survival"], r_cmb["morph_margin"]), y_cmb),
        # Right-anchored at 0.975: left-anchoring at 0.60 put the box's
        # right edge wherever the E_inj string happened to end, which is
        # 20 characters of LaTeX and sat hard against the axes edge. Anchored
        # this way the box grows leftward, into the empty middle of the
        # unphysical row, and cannot overflow however the number formats.
        xytext=(0.975, 0.085), textcoords="axes fraction",
        fontsize=max(ANNOT_FS - 1, 6.0), ha="right", va="center", color=_ink("0.25"),
        arrowprops=dict(arrowstyle="-", lw=0.6, color=_ink("0.35")),
        bbox=dict(boxstyle="round,pad=0.25", fc=_ink("white"), ec=_ink("0.7"), alpha=0.9),
        zorder=20,
    )

    handles = [
        Patch(facecolor=col_survival, edgecolor=_ink("k"), linewidth=0.5,
              label=r"halo survival, $2\Delta\ln L({\rm B}{-}{\rm C})$"),
        Patch(facecolor=col_margin, edgecolor=_ink("k"), linewidth=0.5,
              label=r"halo alone vs. hard-IC alone, $-2\Delta\ln L({\rm C}{-}{\rm A})$"),
        Patch(facecolor=_ink("0.75"), alpha=0.6, edgecolor="none",
              label="deliberately unphysical (ruled out by energetics)"),
    ]
    leg = _style_legend(fig.legend(
        handles, [h.get_label() for h in handles], loc="lower center",
        bbox_to_anchor=(0.5, 0.0), ncol=1, columnspacing=1.2,
        handletextpad=0.5, borderpad=0.4, fontsize=ANNOT_FS, **LEGEND_KW,
    ))
    _fit_bottom_legend(fig, leg, rect_left=0.0, pad=0.6, w_pad=1.6)
    return _save(fig, outdir, "f_hardic_injected_population_contest")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8,
        # Standalone single-panel versions of the two-panel figures above.
        # The paired originals are unchanged; these draw the SAME panel
        # functions, so a change to a panel shows up in both.
        9: fig9,    # Fig 4 left  -> f_dsph_tension_WW
        10: fig10,  # Fig 4 right -> f_dsph_tension_bb
        11: fig11,  # Fig 5 (a)   -> f_systematics_halo_variants
        12: fig12,  # Fig 5 (b)   -> f_systematics_dsph_tension
        # Pairing in the other direction: two standalone panels combined.
        13: fig13,  # Fig 7 + Fig 2 -> f_morphology_and_likelihood
        # Same content as Fig 1, arranged wide for a slide.
        14: fig14,  # Fig 1 side by side -> f8_pole_flux_profiles_wide
        # One panel of Fig 1, for the disk-systematic slide.
        15: fig15,  # Fig 1 rho^2 only  -> f_pole_flux_rho2_disk_comparison
        # Robustness slide: the excess does not behave like a foreground.
        16: fig16,  # hemisphere split    -> f_hemisphere_split
        17: fig17,  # injection nulls     -> f_injection_nulls
        18: fig18,  # disk in/out spectrum + lnL -> f_disk_spectrum_and_lnl
        # Robustness-study figures (covariance / hard-IC injected population).
        19: fig19,  # posterior template covariance -> f_posterior_template_covariance
        20: fig20,  # hard-IC injected-population contest -> f_hardic_injected_population_contest
        # Honest-disclosure variant (b) of fig7; fig7 itself is variant (a).
        21: fig21,  # angular profile + hard-IC -> f_angular_profile_morphology_hardic
        # Same argument as fig20/fig21, in flux: what the injected
        # population recovers once the halo is taken out of the fit.
        22: fig22}  # halo-removed hard-IC spectra -> f_hardic_nohalo_spectra


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(FIGS), default=sorted(FIGS))
    ap.add_argument("--out-dir", type=Path, default=PC / "figures" / "paper_figures")
    ap.add_argument("--copy-to", type=Path, default=None)
    add_style_argument(ap)
    args = ap.parse_args(argv)

    # Canvas scale, font size and line width all come from the shared
    # STYLE_SIZING table, so the slide styles get a bigger figure AND heavier
    # ink AND larger type together. apply_plot_style installs the active scale
    # that _figsize() (= scaled_figsize) reads, so it must run before any
    # figure is built.
    style = normalise_plot_style(args.style, default="paper")
    # This figure set was tuned at lw 1.4 in print, slightly under the package
    # default of 1.5; keep that so the paper renders exactly as published.
    sizing = style_sizing(style, linewidth=1.4 if style == "paper" else None)
    apply_plot_style(style, base_fontsize=sizing["fontsize"],
                     linewidth=sizing["linewidth"], n_colors=8, cmap_name="plasma")
    # AFTER apply_plot_style: theme_ink/theme_legend_kw read the style it just
    # installed, so _set_theme must not run before it.
    _set_theme(style)
    import matplotlib.pyplot as plt
    plt.rcParams["axes.formatter.use_mathtext"] = True
    made = []
    for n in args.only:
        print(f"[fig {n}]")
        made.append(FIGS[n](args.out_dir))
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, args.copy_to / p.name)
            print("  copied ->", (args.copy_to / p.name).name)


if __name__ == "__main__":
    main()
