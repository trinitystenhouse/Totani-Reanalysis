#!/usr/bin/env python3
"""
Reproduce Totani Fig. 9: ln(L_halo) - ln(L_no-halo) vs photon energy.

Methodology (Totani §3.2)
--------------------------
Totani runs two *separate* maximum-likelihood fits per energy bin:

  1. No-halo fit  (fig4): gas, iso, ps, loopA, loopB, ics, fb_pos, fb_neg
  2. Halo fits (fig5/6/7): same templates + one NFW halo model each

Best-fit (thick) line:
  ΔlnL_best(k) = lnL_ML(halo fit, k) − lnL_ML(no-halo fit, k)
  where lnL_ML is the maximum log-likelihood saved in each txt file.

Top-95% boundary (thin) line (Totani §2.2):
  "the boundary that includes the top 95% in the MCMC chain"

  This is the minimum ΔlnL within the highest-posterior-density 95%
  region of the MCMC chain.  Totani evaluates this directly from saved
  chains.  Since fig5/6/7 do not save full chains, we approximate it
  using the parabolic profile-likelihood:

    ΔlnL(a) ≈ ΔlnL_ML − (a − a_ML)² / (2σ²)

  where σ = (p84 − p16) / 2  (half the 1σ interval, always positive).
  The boundary is evaluated at the 95th percentile of the posterior:
  a_95 = a_ML + 2σ  (Gaussian approximation of the 95% upper edge,
  or a_05 = a_ML − 2σ for the lower edge), then we take the lower of
  the two edge ΔlnL values.

  This approach is robust because σ = (p84−p16)/2 is always positive
  by construction, unlike curvature estimates that break down when
  the posterior is skewed (p84 < ML or p16 > ML).
"""

import argparse
import os
import re
import sys
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from totani_helpers.totani_io import read_counts_and_ebounds
from helpers.fermi_plotting import (
    CONF_ANNOT_FS,
    CONF_LABEL_FS,
    CONF_LEGEND_FS,
    CONF_TICK_FS,
)
from helpers.plot_style import (
    current_style_is_transparent, get_cmap_colors, normalise_plot_style,
    set_paper_style,
)

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
MCMC_DIR = os.path.join(REPO_DIR, "Totani_paper_check", "mcmc")
DEFAULT_MAX_ABS_LOGLIKE = 1e10
DEFAULT_MAX_ABS_DLOGLIKE = 1e6

FONT_PRESETS = {
    "paper": {
        "annot": 14.0,
        "label": 18.0,
        "legend": 14.0,
        "ticks": 14.0,
    },
    "conference": {
        "annot": float(CONF_ANNOT_FS),
        "label": float(CONF_LABEL_FS),
        "legend": float(CONF_LEGEND_FS),
        "ticks": float(CONF_TICK_FS),
    },
}

# Conference canvas. This started at 13 x 7.6 in, which made it by far the
# largest figure in the set and filled a slide on its own. It is a single panel
# with an inside legend, so it does not need the room.
CONF_FIG_W, CONF_FIG_H = 5.6, 4.8   # height buys the legend strip, not a bigger plot

# The CONF_* sizes are absolute points chosen for the original 13 in canvas, so
# they have to come down with it or the type renders ~2x larger relative to the
# axes. Deliberately NOT the full 5.6/13 = 0.43, though: scaling type strictly
# with the canvas just yields the same image at fewer pixels, which is no use
# when the point is to occupy less of the slide while staying readable. 0.60
# keeps the type a little larger than proportional, so the figure can sit at
# roughly half-slide width and still be legible from the back of a room.
_CONF_FONT_SCALE = 0.60
FONT_PRESETS["conference"] = {
    k: v * _CONF_FONT_SCALE for k, v in FONT_PRESETS["conference"].items()
}


def _rho_colors():
    """Per-profile line colours, in variant order: rho^2.5, rho^2, rho^1.

    Kept byte-for-byte identical to the companion figure's palette in
    Totani_reanalysis/make_paper_results_figures.py (_likelihood_profiles_panel),
    which samples cols[0], cols[2], cols[4] of this same 5-colour ramp. Call it
    the same way rather than hard-coding hex values, so the two cannot drift if
    the ramp or the style's colour floor is ever retuned. Must be called AFTER
    the style is installed -- the floor is style-dependent.
    """
    cols = get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)
    return [cols[0], cols[2], cols[4]]


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
            "figsize": (CONF_FIG_W, CONF_FIG_H),
            # Slide styles put the key under the axes. Paper keeps the side
            # panel, which is the published arrangement for this figure.
            "legend_below": True,
            # Match the companion likelihood figure in Totani_reanalysis
            # (make_paper_results_figures.py, f9_likelihood_profiles): same
            # call, same samples, so the two read as one figure on a slide.
            # get_cmap_colors -- not a raw colormap lookup -- because it also
            # applies the dark style's brightness floor, which is what lifts
            # these off the background there.
            "line_colors": _rho_colors(),
        }

    set_paper_style(style="paper", linewidth=1.6, n_colors=12, cmap_name="plasma")
    
    if cmap_key in {"", "default", "paper"}:
        cmap = plt.get_cmap("plasma")
        line_colors = [cmap(x) for x in (0.15, 0.55, 0.9)]
    else:
        cmap = plt.get_cmap(cmap_key)
        line_colors = [cmap(x) for x in (0.15, 0.55, 0.9)]

    return {
        "mode": "paper",
        "fonts": FONT_PRESETS["paper"],
        "fg": "k",
        "markerface": "white",
        "figsize": (9.2, 5.4),
        "line_colors": line_colors,
    }


def draw_legend_table_below(fig, best_handles, bound_handles, legend_labels, plot_cfg):
    """The same 2-style x 3-profile key, transposed into a strip under the axes.

    The side-panel version has to reserve 40% of the figure width, and its
    columns are laid out in fractions of a fixed inset while its type is sized
    in points -- so it needs refitting every time the canvas changes, and it
    still ends up near the data. Beneath the plot the grid transposes to
    profiles-across / styles-down, which is short and wide: it costs height
    instead of width, never overlaps a curve, and the axes get the full width.

        profile ->   NFW-rho^2.5   NFW-rho^2   NFW-rho^1
        best-fit        ---o---      ---o---     ---o---
        top 95%         - -o- -      - -o- -     - -o- -
    """
    text_color = plot_cfg["fg"]
    fs = plot_cfg["fonts"]["legend"]

    # Measure where the main axes actually END (tick labels and x-label
    # included) and hang the strip below that. Hand-picking a height for the
    # strip does not work: how far the x-label reaches depends on the font
    # scale and the canvas, so a fixed fraction either overlaps the label or
    # leaves a gap. Requires the axes to be laid out first -- see the call site.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_h_px = fig.get_figheight() * fig.dpi
    main_ax = fig.axes[0]
    ax_bottom = main_ax.get_tightbbox(rend).y0 / fig_h_px

    strip_h = min(max(ax_bottom - 0.03, 0.10), 0.30)
    legend_ax = fig.add_axes([0.0, 0.0, 1.0, strip_h])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    n = len(legend_labels)
    x_rowlab = 0.20                     # right edge of the "best-fit"/"top 95%" column
    x_first, x_last = 0.34, 0.90        # centres of the first and last profile columns
    xs = [x_first + (x_last - x_first) * i / max(n - 1, 1) for i in range(n)]
    half = 0.055                        # half-length of a handle sample
    y_head, y_best, y_bound = 0.86, 0.52, 0.16

    for x, tag in zip(xs, legend_labels):
        legend_ax.text(x, y_head, tag, color=text_color, fontsize=fs,
                       ha="center", va="center", transform=legend_ax.transAxes)

    for y, label in ((y_best, "best-fit"), (y_bound, "top 95%")):
        legend_ax.text(x_rowlab, y, label, color=text_color, fontsize=fs,
                       ha="right", va="center", transform=legend_ax.transAxes)

    for x, lbest, lbound in zip(xs, best_handles, bound_handles):
        legend_ax.plot([x - half, x + half], [y_best, y_best],
                       color=lbest.get_color(), ls=lbest.get_linestyle(), lw=2.0,
                       marker=lbest.get_marker(), markersize=7,
                       markerfacecolor=lbest.get_markerfacecolor(),
                       markeredgecolor=lbest.get_markeredgecolor(),
                       transform=legend_ax.transAxes, clip_on=False)
        legend_ax.plot([x - half, x + half], [y_bound, y_bound],
                       color=lbound.get_color(), ls=lbound.get_linestyle(), lw=0.9,
                       marker=lbound.get_marker(), markersize=5,
                       markerfacecolor=lbound.get_markerfacecolor(),
                       markeredgecolor=lbound.get_markeredgecolor(),
                       alpha=0.7, transform=legend_ax.transAxes, clip_on=False)


def draw_legend_table(fig, best_handles, bound_handles, legend_labels, plot_cfg):
    """Draw a stable 3-column legend table in a dedicated figure panel."""
    if plot_cfg.get("legend_below"):
        return draw_legend_table_below(
            fig, best_handles, bound_handles, legend_labels, plot_cfg)

    text_color = plot_cfg["fg"]
    fs = plot_cfg["fonts"]["legend"]

    # The table's contents are sized in POINTS but its columns are placed in
    # fractions of this inset, so a fixed fraction only works at one figure
    # width -- at any narrower canvas the headers overprint each other.
    # Size the inset in inches from the font (2.32 in held the table at 16 pt,
    # and it scales linearly), then convert to a fraction.
    #
    # Placement keys off where the ROW LABELS end, not the inset's own right
    # edge: the labels start at x_label (0.73) and are drawn with clip_on=False
    # precisely because they are wider than the column left for them, so they
    # always overhang the inset. Pinning the inset's right edge instead let
    # them run off the canvas as soon as the figure got narrower.
    fig_w, fig_h = fig.get_size_inches()
    w_in = min(2.32 * fs / 16.0, 0.46 * fig_w)
    h_in = min(1.65 * fs / 16.0, 0.42 * fig_h)
    label_w_in = 0.075 * fs                    # ~"NFW-rho^2.5" at this size
    right_in = 0.97 * fig_w                    # where the labels should end
    x0_in = right_in - label_w_in - 0.73 * w_in
    legend_ax = fig.add_axes([max(x0_in, 0.0) / fig_w,
                              0.88 - h_in / fig_h,
                              w_in / fig_w, h_in / fig_h])
    legend_ax.set_axis_off()

    x_best = 0.06
    x_bound = 0.40
    x_label = 0.73
    y_header = 0.92
    row_step = 0.24
    y_rows = [y_header - row_step * (i + 1) for i in range(len(legend_labels))]

    legend_ax.text(x_best, y_header, "best-fit", color=text_color, fontsize=fs,
                   ha="left", va="center", transform=legend_ax.transAxes)
    legend_ax.text(x_bound, y_header, "top 95%", color=text_color, fontsize=fs,
                   ha="left", va="center", transform=legend_ax.transAxes)

    for y, lbest, lbound, tag in zip(y_rows, best_handles, bound_handles, legend_labels):
        c_best = lbest.get_color()
        c_bound = lbound.get_color()
        ls_best = lbest.get_linestyle()
        ls_bound = lbound.get_linestyle()
        mk_best = lbest.get_marker()
        mk_bound = lbound.get_marker()
        mfc_best = lbest.get_markerfacecolor()
        mfc_bound = lbound.get_markerfacecolor()
        mec_best = lbest.get_markeredgecolor()
        mec_bound = lbound.get_markeredgecolor()

        legend_ax.plot([x_best, x_best + 0.16], [y, y], color=c_best, ls=ls_best,
                       lw=2.0, marker=mk_best, markersize=7,
                       markerfacecolor=mfc_best, markeredgecolor=mec_best,
                       transform=legend_ax.transAxes, clip_on=False)
        legend_ax.plot([x_bound, x_bound + 0.16], [y, y], color=c_bound, ls=ls_bound,
                       lw=0.9, marker=mk_bound, markersize=5,
                       markerfacecolor=mfc_bound, markeredgecolor=mec_bound,
                       alpha=0.7, transform=legend_ax.transAxes, clip_on=False)
        legend_ax.text(x_label, y, tag, color=text_color, fontsize=fs,
                       ha="left", va="center", transform=legend_ax.transAxes)

    return legend_ax


def _first_existing_dir(*paths: str) -> str:
    """Return the first existing directory, or the first candidate if none exist."""
    for path in paths:
        if os.path.isdir(path):
            return path
    return paths[0]


def _fit_dirs(fit_mode: str, *, with_disk: bool = False, base: Optional[str] = None) -> dict:
    if with_disk and str(fit_mode) not in {"global", "pixelwise"}:
        raise SystemExit("--with-disk is only defined for --fit-mode global or pixelwise.")
    base_dir = base or MCMC_DIR
    if str(fit_mode) == "global":
        prefix = "global_fit_results"
    elif str(fit_mode) == "pixelwise":
        prefix = "pixelwise_mcmc_results"
    else:
        prefix = "mcmc_results"
    suffix = "_w_disk" if with_disk else ""
    return {
        "nohalo": os.path.join(base_dir, f"{prefix}_fig4{suffix}"),
        "rho25": os.path.join(base_dir, f"{prefix}_fig5{suffix}"),
        "rho2": os.path.join(base_dir, f"{prefix}_fig6{suffix}"),
        "rho1": os.path.join(base_dir, f"{prefix}_fig7{suffix}"),
    }


def _validate_same_disk_variant(paths: dict[str, str]) -> None:
    """Catch accidental comparisons between disk-included and disk-excluded fits."""
    disk_flags = {key: "_w_disk" in os.path.basename(str(path)) for key, path in paths.items()}
    if len(set(disk_flags.values())) <= 1:
        return
    disk_included = ", ".join(key for key, is_disk in disk_flags.items() if is_disk)
    disk_excluded = ", ".join(key for key, is_disk in disk_flags.items() if not is_disk)
    raise SystemExit(
        "Fig. 9 compares likelihoods only between fits with the same disk mask. "
        f"Disk-included dirs: {disk_included}; disk-excluded dirs: {disk_excluded}."
    )


def _require_existing_dirs(paths: dict[str, str], *, context: str) -> None:
    missing = [f"{key}: {path}" for key, path in paths.items() if not os.path.isdir(str(path))]
    if not missing:
        return
    missing_text = "\n  ".join(missing)
    raise SystemExit(
        f"{context} requires matching result directories, but these are missing:\n"
        f"  {missing_text}"
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _parse_txt(path: str) -> dict:
    """
    Parse a saved mcmc_results_k*.txt file.
    Returns dict with:
      lnL_ml    : float   — best-fit log-likelihood (ML sample)
      E_ctr_mev : float
      coeffs    : dict[label -> dict{ML, p05, p16, p50, p84, p95}]
                  (p05/p95 present only in fig2_3; others have p16/p50/p84)
    """
    result = {"lnL_ml": np.nan, "E_ctr_mev": np.nan, "coeffs": {}}
    if not os.path.exists(path):
        return result

    with open(path) as fh:
        lines = fh.readlines()

    for line in lines:
        m = re.search(r"E_ctr=(\S+)\s*MeV", line)
        if m:
            result["E_ctr_mev"] = float(m.group(1))
        m = re.search(r"Log-likelihood \(ML\):\s*([+-]?\S+)", line)
        if m:
            result["lnL_ml"] = float(m.group(1))

    # Parse component table — handle both 4-col (ML/p16/p50/p84) and
    # 6-col (ML/p05/p16/p50/p84/p95) layouts.
    in_table = False
    header_cols: list = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Component") and "ML" in stripped:
            header_cols = stripped.split()[1:]
            in_table = True
            continue
        if in_table:
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
            nums  = parts[len(parts) - n_num :]
            try:
                vals = {col: float(v) for col, v in zip(header_cols, nums)}
            except ValueError:
                continue
            result["coeffs"][label] = vals

    return result


def _load_lnL_ml(
    mcmc_dir: str,
    nE: int,
    *,
    max_abs_loglike: Optional[float] = DEFAULT_MAX_ABS_LOGLIKE,
    label: str = "",
) -> np.ndarray:
    """ML log-likelihood per energy bin, from txt files."""
    lnL = np.full(nE, np.nan)
    masked = []
    for k in range(nE):
        path = os.path.join(mcmc_dir, f"mcmc_results_k{k:02d}.txt")
        lnL[k] = _parse_txt(path)["lnL_ml"]
        if (
            max_abs_loglike is not None
            and max_abs_loglike > 0
            and np.isfinite(lnL[k])
            and abs(float(lnL[k])) > float(max_abs_loglike)
        ):
            masked.append((k, float(lnL[k])))
            lnL[k] = np.nan
    if masked:
        tag = f" for {label}" if label else ""
        bins = ", ".join(f"k{k:02d}={val:.3e}" for k, val in masked)
        print(f"WARNING: masked pathological log-likelihoods{tag}: {bins}")
    return lnL


def _mask_pathological_dlnL(
    dlnL: np.ndarray,
    *,
    max_abs_dlnl: Optional[float],
    label: str = "",
) -> np.ndarray:
    if max_abs_dlnl is None or max_abs_dlnl <= 0:
        return dlnL
    out = np.asarray(dlnL, float).copy()
    mask = np.isfinite(out) & (np.abs(out) > float(max_abs_dlnl))
    if np.any(mask):
        tag = f" for {label}" if label else ""
        bins = ", ".join(f"k{k:02d}={out[k]:.3e}" for k in np.flatnonzero(mask))
        print(f"WARNING: masked pathological ΔlnL values{tag}: {bins}")
        out[mask] = np.nan
    return out


def _load_nfw_stats(mcmc_dir: str, nE: int, nfw_label: str) -> dict:
    """
    Return dict of arrays (nE,) for each saved percentile of the NFW amplitude:
      ml, p16, p50, p84, and (if available) p05, p95.
    Missing values → nan.
    """
    keys = ["ML", "p05", "p16", "p50", "p84", "p95"]
    out = {k: np.full(nE, np.nan) for k in keys}

    for k_bin in range(nE):
        path = os.path.join(mcmc_dir, f"mcmc_results_k{k_bin:02d}.txt")
        d = _parse_txt(path)
        # Find NFW row — exact match first, then substring
        row = None
        if nfw_label in d["coeffs"]:
            row = d["coeffs"][nfw_label]
        else:
            for lab, vals in d["coeffs"].items():
                if "nfw" in lab.lower():
                    row = vals
                    break
        if row is None:
            continue
        for col_key in keys:
            if col_key in row:
                out[col_key][k_bin] = row[col_key]

    return out


def _load_chain_top95(mcmc_dir: str, nE: int, nfw_label: str):
    """
    Where full chain + logprob are saved (fig2_3 npz files), build the
    halo amplitude interval from the highest-logprob 95% of samples.
    Returns (a_lo, a_hi) arrays or None if chains are unavailable.
    """
    a_lo = np.full(nE, np.nan)
    a_hi = np.full(nE, np.nan)
    found = False

    for k in range(nE):
        path = os.path.join(mcmc_dir, f"mcmc_results_k{k:02d}.npz")
        if not os.path.exists(path):
            continue
        try:
            with np.load(path, allow_pickle=True) as npz:
                if "chain" not in npz or "logprob" not in npz:
                    continue
                labels = [str(x) for x in npz["labels"].tolist()]
                if nfw_label not in labels:
                    continue
                chain   = np.asarray(npz["chain"],   float)
                logprob = np.asarray(npz["logprob"], float)
                if chain.ndim != 3 or logprob.ndim != 2:
                    continue
                burn = int(npz["burn"])       if "burn"      in npz else 0
                thin = int(npz["thin_save"]) if "thin_save" in npz else 1
                burn_idx = max(0, int(np.ceil(burn / max(thin, 1))))
                chain_p   = chain[burn_idx:, :, :]
                logprob_p = logprob[burn_idx:, :]
                if chain_p.size == 0:
                    continue
                halo_idx     = labels.index(nfw_label)
                halo_samples = chain_p[:, :, halo_idx].reshape(-1)
                lp_samples   = logprob_p.reshape(-1)
                good = np.isfinite(halo_samples) & np.isfinite(lp_samples)
                if good.sum() < 10:
                    continue
                hs = halo_samples[good]
                lp = lp_samples[good]
                lp_cut = np.quantile(lp, 0.05)
                keep = lp >= lp_cut
                if keep.sum() < 2:
                    continue
                a_lo[k] = float(np.min(hs[keep]))
                a_hi[k] = float(np.max(hs[keep]))
                found = True
        except Exception:
            continue

    return (a_lo, a_hi) if found else None


# ---------------------------------------------------------------------------
# Boundary ΔlnL
#
# Estimate the profile log-likelihood curvature from two anchor points
# (p16 and p84, where we know the ΔlnL = ΔlnL_best - 0.5 under a Gaussian
# approximation).  Then evaluate at the 95% boundary edges (p05/p95 when
# available, otherwise p16/p84).
#
# For a true Gaussian posterior:
#   ΔlnL(a) = ΔlnL_ML + curv * (a - a_ML)² / 2
#   ΔlnL(a_p16) = ΔlnL_ML - 0.5        (because p16 is 1σ below ML)
#   → curv = -1 / (a_ML - a_p16)²
#
# This is strictly correct for a symmetric Gaussian posterior.  For skewed
# posteriors we average the curvature estimates from p16 and p84 sides,
# which is more robust than using σ = (p84-p16)/2.
# ---------------------------------------------------------------------------

def _boundary_dlnL(
    dlnL_best: np.ndarray,
    nfw_stats: dict,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
) -> np.ndarray:
    """
    Compute the top-95% boundary ΔlnL curve using the parabolic
    profile-likelihood approximation:

        ΔlnL(a) ≈ ΔlnL_ML − (a − a_ML)² / (2σ²)

    σ = (p84 − p16) / 2  is always positive, making this robust to
    skewed posteriors where p84 < ML or p16 > ML (which break curvature
    estimates based on individual arm lengths).

    The boundary is evaluated at a_lo and a_hi (the 95% interval edges)
    and the minimum of the two values is returned, allowing the curve to
    go negative at bins where the posterior straddles zero.
    """
    nE       = len(dlnL_best)
    boundary = np.full(nE, np.nan)

    a_ml  = nfw_stats["ML"]
    a_p16 = nfw_stats["p16"]
    a_p84 = nfw_stats["p84"]

    for k in range(nE):
        dL  = dlnL_best[k]
        am  = a_ml[k]
        lo  = a_lo[k]
        hi  = a_hi[k]
        p16 = a_p16[k]
        p84 = a_p84[k]

        if not np.isfinite(dL):
            boundary[k] = np.nan
            continue

        # σ from the symmetric 1σ interval: always positive
        if np.isfinite(p16) and np.isfinite(p84) and p84 > p16:
            sigma = (p84 - p16) / 2.0
        else:
            # No width info: boundary equals best-fit
            boundary[k] = dL
            continue

        def dlnL_at(a):
            """Parabolic profile-likelihood anchored at (a_ML, ΔlnL_ML)."""
            if not np.isfinite(am):
                # If ML amplitude unknown, anchor at zero
                return dL - a**2 / (2.0 * sigma**2)
            return dL - (a - am)**2 / (2.0 * sigma**2)

        vals = []
        for a in (lo, hi):
            if np.isfinite(a):
                vals.append(dlnL_at(a))

        boundary[k] = float(min(vals)) if vals else dL

    return boundary


# ---------------------------------------------------------------------------
# Variant plotting
# ---------------------------------------------------------------------------

def _build_variants(dirs: dict) -> list:
    """Return the (tag, mcmc_dir, nfw_label, style) tuples for one dir-set."""
    return [
        (
            r"NFW-$\rho^{2.5}$",
            str(dirs["rho25"]),
            "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
            dict(ls="-.", marker="^"),
        ),
        (
            r"NFW-$\rho^{2}$",
            str(dirs["rho2"]),
            "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
            dict(ls="-", marker="o"),
        ),
        (
            r"NFW-$\rho^{1}$",
            str(dirs["rho1"]),
            "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno",
            dict(ls="--", marker="s"),
        ),
    ]


def _load_nohalo_baseline(mcmc_dir_nohalo: str, nE: int, max_abs_loglike) -> np.ndarray:
    """No-halo ML log-likelihoods (the common denominator), with masking."""
    lnL_nohalo = _load_lnL_ml(
        mcmc_dir_nohalo, nE, max_abs_loglike=max_abs_loglike, label="no-halo"
    )
    if not np.any(np.isfinite(lnL_nohalo)):
        raise RuntimeError(
            "No finite no-halo log-likelihoods remain after masking. "
            f"Checked: {mcmc_dir_nohalo}"
        )
    return lnL_nohalo


def _plot_variant_curves(
    ax,
    plot_cfg,
    variants,
    lnL_nohalo,
    nE,
    Ectr_gev,
    *,
    max_abs_loglike,
    max_abs_dlnl,
    ls_override=None,
    label_suffix="",
):
    """Plot best-fit + top-95% ΔlnL curves for each NFW variant.

    ``ls_override`` forces a single linestyle for every variant (used by the
    overlay mode to encode disk state); when None each variant keeps its own
    style.  Returns (best_handles, bound_handles, legend_labels).
    """
    best_handles, bound_handles, legend_labels = [], [], []
    for color, (tag, mcmc_dir, nfw_label, sty) in zip(plot_cfg["line_colors"], variants):

        # ---- best-fit ΔlnL -------------------------------------------
        lnL_halo = _load_lnL_ml(
            mcmc_dir, nE, max_abs_loglike=max_abs_loglike, label=tag
        )
        dlnL_best = _mask_pathological_dlnL(
            lnL_halo - lnL_nohalo, max_abs_dlnl=max_abs_dlnl, label=tag
        )
        if not np.any(np.isfinite(dlnL_best)):
            print(f"WARNING: no finite ΔlnL values for {tag}{label_suffix}; skipping {mcmc_dir}")
            continue

        # ---- NFW amplitude statistics --------------------------------
        nfw_stats = _load_nfw_stats(mcmc_dir, nE, nfw_label)

        sigma_arr = (nfw_stats["p84"] - nfw_stats["p16"]) / 2.0
        a_ml_arr = nfw_stats["ML"]
        a_lo = np.where(np.isfinite(nfw_stats["p05"]),
                        nfw_stats["p05"],
                        a_ml_arr - 2.0 * sigma_arr)
        a_hi = np.where(np.isfinite(nfw_stats["p95"]),
                        nfw_stats["p95"],
                        a_ml_arr + 2.0 * sigma_arr)

        chain95 = _load_chain_top95(mcmc_dir, nE, nfw_label)
        if chain95 is not None:
            c_lo, c_hi = chain95
            mask = np.isfinite(c_lo) & np.isfinite(c_hi)
            a_lo[mask] = c_lo[mask]
            a_hi[mask] = c_hi[mask]

        dlnL_boundary = _boundary_dlnL(dlnL_best, nfw_stats, a_lo, a_hi)

        ls = ls_override if ls_override is not None else sty["ls"]
        lbest, = ax.plot(Ectr_gev, dlnL_best, lw=2.0,
                         color=color, ls=ls,
                         marker=sty["marker"], mfc=plot_cfg["markerface"], mec=color,
                         mew=1.2, markersize=7)
        lbound, = ax.plot(Ectr_gev, dlnL_boundary, lw=0.9,
                          color=color, ls=ls,
                          marker=sty["marker"], mfc=plot_cfg["markerface"], mec=color,
                          mew=0.8, markersize=5, alpha=0.7)
        best_handles.append(lbest)
        bound_handles.append(lbound)
        legend_labels.append(f"{tag}{label_suffix}")

        peak_k = int(np.nanargmax(dlnL_best))
        print(f"{tag}{label_suffix}: peak ΔlnL = {dlnL_best[peak_k]:.1f} "
              f"at {Ectr_gev[peak_k]:.1f} GeV")

    return best_handles, bound_handles, legend_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Totani Fig. 9: ΔlnL(ML) vs energy."
    )
    ap.add_argument("--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig9"))
    ap.add_argument(
        "--fit-mode",
        default="mcmc",
        choices=["mcmc", "global", "pixelwise"],
        help="Choose the Fig. 4-7 result family to plot.",
    )
    ap.add_argument(
        "--with-disk",
        action="store_true",
        help=(
            "Diagnostic global-fit variant only: use global_fit_results_fig[4-7]_w_disk. "
            "Totani Fig. 9 itself uses the default disk-excluded Fig. 4-7 fits. "
            "Appends '_w_disk' to the output directory."
        ),
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help=(
            "Plot the disk-excluded and disk-included Fig. 4-7 fits on the same axes "
            "(colour = NFW variant, solid = no disk, dashed = disk). Derives both dir "
            "sets from --fit-mode and ignores the --mcmc-dir-* overrides. Appends "
            "'_overlay' to the output directory."
        ),
    )
    ap.add_argument(
        "--mcmc-base",
        default=None,
        help=(
            "Base directory holding the fig4-7 result folders. Overrides the built-in "
            "default (Totani_paper_check/mcmc). Required to reach the global fits under "
            "Totani_reanalysis/mcmc/fit_results/quick_global when using --with-disk or "
            "--overlay (which derive dir names from --fit-mode and ignore --mcmc-dir-*)."
        ),
    )
    ap.add_argument("--mcmc-dir-nohalo",
        default=None)
    ap.add_argument("--mcmc-dir-rho25",
        default=None,
        help="MCMC results for NFW-rho2.5 halo fit (Totani fig 5). "
             "Must use fb_pos/fb_neg templates and disk excluded, "
             "matching the no-halo baseline (fig4). "
             "Do NOT use mcmc_results_fig2_3 here — that fit uses "
             "the flat bubble template and includes the disk, so its "
             "lnL is not comparable to the fig4 baseline.")
    ap.add_argument("--mcmc-dir-rho2",
        default=None)
    ap.add_argument("--mcmc-dir-rho1",
        default=None)
    ap.add_argument(
        "--max-abs-loglike",
        type=float,
        default=DEFAULT_MAX_ABS_LOGLIKE,
        help=(
            "Mask saved ML log-likelihood values with abs(lnL) above this threshold. "
            "Use <=0 to disable. Default masks catastrophic failed-fit values."
        ),
    )
    ap.add_argument(
        "--max-abs-dlnl",
        type=float,
        default=DEFAULT_MAX_ABS_DLOGLIKE,
        help=(
            "Mask ΔlnL values with absolute value above this threshold. "
            "Use <=0 to disable. This catches failed global fits whose absolute "
            "lnL values are not large enough to trip --max-abs-loglike."
        ),
    )
    ap.add_argument(
        "--cmap",
        default="default",
        help="Display mode. Use 'dark_transparent' for transparent dark styling with conference-size fonts.",
    )
    args = ap.parse_args()

    # Distinguish output directories per variant so they don't overwrite.
    if args.overlay:
        args.outdir = args.outdir + "_overlay"
    elif args.with_disk:
        args.outdir = args.outdir + "_w_disk"
    os.makedirs(args.outdir, exist_ok=True)
    plot_cfg = resolve_plot_config(args.cmap)

    max_abs_loglike = None if float(args.max_abs_loglike) <= 0 else float(args.max_abs_loglike)
    max_abs_dlnl = None if float(args.max_abs_dlnl) <= 0 else float(args.max_abs_dlnl)

    _, _, _, _, Ectr_mev, _ = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)
    Ectr_gev = Ectr_mev / 1000.0

    # ------------------------------------------------------------------
    # Figure (shared setup)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=plot_cfg["figsize"])
    ax.set_xscale("log")
    ax.set_xlabel("photon energy [GeV]", fontsize=plot_cfg["fonts"]["label"])
    ax.set_ylabel(r"$\ln(L) - \ln(L_{\rm no\text{-}halo})$", fontsize=plot_cfg["fonts"]["label"])
    ax.axhline(0, color=plot_cfg["fg"], lw=0.5, ls=":")
    ax.set_xlim(1.0, 1000.0)
    #ax.set_ylim(-20, 120)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.tick_params(labelsize=plot_cfg["fonts"]["ticks"])

    if args.overlay:
        # ------------------------------------------------------------------
        # Overlay: disk-excluded and disk-included fits on the same axes.
        # Colour encodes the NFW variant; linestyle encodes the disk state.
        # Both dir sets come from --fit-mode; --mcmc-dir-* overrides are ignored.
        # ------------------------------------------------------------------
        print(f"Fig. 9 fit mode: {args.fit_mode} (overlay: disk vs no-disk)")
        overlay_specs = [
            ("no disk", _fit_dirs(args.fit_mode, with_disk=False, base=args.mcmc_base), "-",  "  [no disk]"),
            ("disk",    _fit_dirs(args.fit_mode, with_disk=True,  base=args.mcmc_base), "--", "  [disk]"),
        ]
        n_curves = 0
        variant_meta = None
        for state, dirs, ls_override, suffix in overlay_specs:
            _require_existing_dirs(dirs, context=f"Overlay Fig. 9 ({state})")
            print(f"  [{state}] no-halo: {dirs['nohalo']}")
            lnL_nohalo = _load_nohalo_baseline(dirs["nohalo"], nE, max_abs_loglike)
            variants = _build_variants(dirs)
            if variant_meta is None:
                variant_meta = list(zip(plot_cfg["line_colors"], variants))
            best_handles, _, _ = _plot_variant_curves(
                ax, plot_cfg, variants, lnL_nohalo, nE, Ectr_gev,
                max_abs_loglike=max_abs_loglike, max_abs_dlnl=max_abs_dlnl,
                ls_override=ls_override, label_suffix=suffix,
            )
            n_curves += len(best_handles)

        if n_curves == 0:
            raise RuntimeError(
                "No Fig. 9 likelihood curves could be built after masking pathological bins."
            )

        # Two-part legend: colour -> NFW variant, linestyle -> disk state.
        from matplotlib.lines import Line2D
        variant_proxies = [
            Line2D([0], [0], color=c, lw=2.0, ls="-", marker=v[3]["marker"],
                   mfc=plot_cfg["markerface"], mec=c, markersize=7)
            for c, v in variant_meta
        ]
        variant_names = [v[0] for _, v in variant_meta]
        state_proxies = [
            Line2D([0], [0], color=plot_cfg["fg"], lw=2.0, ls="-"),
            Line2D([0], [0], color=plot_cfg["fg"], lw=2.0, ls="--"),
        ]
        leg = ax.legend(
            variant_proxies + state_proxies,
            variant_names + ["no disk", "disk"],
            fontsize=plot_cfg["fonts"]["legend"],
            loc="upper left", frameon=False,
        )
        ax.add_artist(leg)
        fig.tight_layout()
        out_name = "totani_fig9_lnl_diff_overlay.png"
    else:
        # ------------------------------------------------------------------
        # Single disk variant (Totani default, or --with-disk diagnostic).
        # ------------------------------------------------------------------
        mode_dirs = _fit_dirs(args.fit_mode, with_disk=bool(args.with_disk), base=args.mcmc_base)
        mcmc_dir_nohalo = args.mcmc_dir_nohalo or mode_dirs["nohalo"]
        mcmc_dir_rho25 = args.mcmc_dir_rho25 or mode_dirs["rho25"]
        mcmc_dir_rho2 = args.mcmc_dir_rho2 or mode_dirs["rho2"]
        mcmc_dir_rho1 = args.mcmc_dir_rho1 or mode_dirs["rho1"]
        selected_dirs = {
            "nohalo": mcmc_dir_nohalo,
            "rho25": mcmc_dir_rho25,
            "rho2": mcmc_dir_rho2,
            "rho1": mcmc_dir_rho1,
        }
        _validate_same_disk_variant(selected_dirs)
        if args.with_disk:
            _require_existing_dirs(
                selected_dirs,
                context=(
                    "Disk-included Fig. 9 is a diagnostic, not Totani's default Fig. 9, "
                    "so it"
                ),
            )
        print(f"Fig. 9 fit mode: {args.fit_mode}")
        print("Fig. 9 disk variant: " + ("disk included" if args.with_disk else "disk excluded (Totani default)"))
        print(f"  no-halo: {mcmc_dir_nohalo}")
        print(f"  rho2.5 : {mcmc_dir_rho25}")
        print(f"  rho2   : {mcmc_dir_rho2}")
        print(f"  rho1   : {mcmc_dir_rho1}")

        lnL_nohalo = _load_nohalo_baseline(mcmc_dir_nohalo, nE, max_abs_loglike)
        variants = _build_variants(selected_dirs)
        best_handles, bound_handles, legend_labels = _plot_variant_curves(
            ax, plot_cfg, variants, lnL_nohalo, nE, Ectr_gev,
            max_abs_loglike=max_abs_loglike, max_abs_dlnl=max_abs_dlnl,
        )
        if not best_handles:
            raise RuntimeError(
                "No Fig. 9 likelihood curves could be built after masking pathological bins."
            )
        if plot_cfg.get("legend_below"):
            # right=0.60 existed only to reserve the side panel; with the key
            # underneath, the axes get the full width back and the bottom
            # margin makes room for the strip. Adjust BEFORE drawing the
            # legend -- draw_legend_table_below measures the laid-out axes
            # (x-label included) to decide where the strip starts.
            # The plot style sets figure.autolayout=True, so matplotlib
            # re-runs tight_layout on EVERY draw and silently discards
            # subplots_adjust. Turning the layout engine off is what makes the
            # reserved bottom margin real. (The same clobbering is why the old
            # side-panel `right=0.60` never actually held the axes back, and the
            # legend inset ended up sitting over the curves.)
            fig.set_layout_engine("none")
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.34, top=0.95)
            draw_legend_table(fig, best_handles, bound_handles, legend_labels, plot_cfg)
        else:
            draw_legend_table(fig, best_handles, bound_handles, legend_labels, plot_cfg)
            fig.subplots_adjust(left=0.11, right=0.60, bottom=0.12, top=0.96)
        out_name = "totani_fig9_lnl_diff.png"

    out_png = os.path.join(args.outdir, out_name)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig)
    print(f"✓ wrote {out_png}")


if __name__ == "__main__":
    main()
