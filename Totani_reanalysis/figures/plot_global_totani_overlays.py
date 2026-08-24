#!/usr/bin/env python3
"""
Overlay the global-fit spectra and Totani-style spectra for paper figures 2-9.

This script deliberately works from the already generated ``*_f_ml.txt`` tables.
It does not rerun any fit and does not touch the MCMC/global-fit directories.

For figures 2-7 it makes a component overlay with:
  - Totani-style result: solid line, filled marker, optional light error band
  - Global-fit result: dashed line, open marker

For figure 8 it uses the NFW component from figures 5, 6, and 7, matching the
three halo-profile panels in Totani Fig. 8.

For figure 9 it overlays the best-fit per-bin likelihood improvement for the
Totani-style and global fits, each referenced to its own Fig. 4 no-halo fit.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_DIR = Path(
    os.environ.get(
        "REPO_PATH",
        Path(__file__).resolve().parents[2],
    )
).resolve()
CHECK_DIR = REPO_DIR / "Totani_paper_check"
REANALYSIS_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent
# Totani reference figure tables (plots_figN[_tag]/mcmc_figN_f_ml.txt) live in
# Totani_paper_check/figures, alongside the reference chains below.
TOTANI_FIG_DIR = CHECK_DIR / "figures"
# Totani reference chains (mcmc_results_figN) live in Totani_paper_check/mcmc.
MCMC_DIR = CHECK_DIR / "mcmc"
# Our own global/pixelwise fits (fit_results/{quick_global,pixelwise_mcmc}) live
# in Totani_reanalysis/mcmc -- a distinct base from the Totani reference above.
FIT_MCMC_DIR = REANALYSIS_DIR / "mcmc"
DATA_DIR = REPO_DIR / "fermi_data" / "totani"

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(CHECK_DIR))
from helpers.plot_style import apply_plot_style, current_style_is_transparent, plasma_color  # noqa: E402
from totani_helpers.totani_io import read_counts_and_ebounds  # noqa: E402


DEFAULT_COMPONENTS = [
    "nfw",
    "gas",
    "ics",
    "ps",
    "iso",
    "fb_pos",
    "fb_neg",
    "loopI",
]

RATIO_COMPONENTS = ["nfw"]

COMPONENT_LABELS = {
    "total": "Total",
    "gas": "Gas",
    "iso": "Isotropic",
    "ps": "Point sources",
    "ics": "ICS",
    "nfw": "NFW",
    "fb_pos": "Fermi bubbles +",
    "fb_neg": "Fermi bubbles -",
    "loopI": "Loop I",
}

COMPONENT_COLORS = {
    "total": "#111111",
    "nfw": plasma_color(0.15),
    "gas": plasma_color(0.28),
    "ics": plasma_color(0.41),
    "ps": plasma_color(0.54),
    "iso": "#6E6E6E",
    "fb_pos": plasma_color(0.68),
    "fb_neg": plasma_color(0.81),
    "loopI": plasma_color(0.94),
}

COMPONENT_MARKERS = {
    "total": "o",
    "nfw": "D",
    "gas": "s",
    "ics": "^",
    "ps": "v",
    "iso": "P",
    "fb_pos": ">",
    "fb_neg": "<",
    "loopI": "X",
}

FIG_TITLES = {
    2: r"Fig. 2: fit including Galactic disk region ($|b| < 10^\circ$)",
    3: r"Fig. 3: fit including Galactic disk region, with this region removed",
    4: "Fig. 4: Baseline halo fit",
    5: r"Fig. 5: NFW-$\rho^{2.5}$",
    6: r"Fig. 6: NFW-$\rho^2$",
    7: r"Fig. 7: NFW-$\rho^1$",
}

FIG8_PANELS = [
    (
        5,
        r"NFW-$\rho^{2.5}$ (GCE)",
        "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        "^",
        (-1.0e-5, 2.0e-5),
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * 1e-5,
    ),
    (
        6,
        r"NFW-$\rho^2$",
        "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
        "o",
        (-2.0e-5, 4.0e-5),
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1e-5,
    ),
    (
        7,
        r"NFW-$\rho^1$",
        "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno",
        "s",
        (-1.0e-4, 2.0e-4),
        np.array([-1.0, 0.0, 1.0, 2.0]) * 1e-4,
    ),
]

FIG9_VARIANTS = [
    (
        5,
        r"NFW-$\rho^{2.5}$",
        "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        "^",
        plasma_color(0.18),
    ),
    (
        6,
        r"NFW-$\rho^2$",
        "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
        "o",
        plasma_color(0.52),
    ),
    (
        7,
        r"NFW-$\rho^1$",
        "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno",
        "s",
        plasma_color(0.86),
    ),
]

DEFAULT_FIG9_MAX_ABS_LOGLIKE = 1e10
DEFAULT_FIG9_MAX_ABS_DLOGLIKE = 1e6


class CurveTable:
    def __init__(self, path: Path, columns: Mapping[str, np.ndarray]):
        self.path = Path(path)
        self.columns = dict(columns)

    @property
    def energy(self) -> np.ndarray:
        return self.columns["E_GeV"]

    def has(self, component: str) -> bool:
        return component in self.columns

    def components(self) -> List[str]:
        comps = []
        for name in self.columns:
            if name == "E_GeV" or name.endswith("_err_lo") or name.endswith("_err_hi"):
                continue
            comps.append(name)
        return comps

    def y(self, component: str) -> np.ndarray:
        return self.columns[component]

    def yerr(self, component: str) -> np.ndarray | None:
        lo_key = f"{component}_err_lo"
        hi_key = f"{component}_err_hi"
        if lo_key not in self.columns or hi_key not in self.columns:
            return None
        lo = np.asarray(self.columns[lo_key], float)
        hi = np.asarray(self.columns[hi_key], float)
        err = np.vstack([np.clip(lo, 0.0, None), np.clip(hi, 0.0, None)])
        err[~np.isfinite(err)] = 0.0
        return err


def _parse_header(path: Path) -> List[str]:
    with path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip().split()
    raise ValueError(f"No '# ...' header found in {path}")


def load_curve_table(path: Path) -> CurveTable:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    names = _parse_header(path)
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(names):
        raise ValueError(f"{path} has {arr.shape[1]} columns but header has {len(names)} names")

    columns = {name: arr[:, i] for i, name in enumerate(names)}
    if "E_GeV" not in columns:
        raise ValueError(f"{path} does not contain E_GeV")
    return CurveTable(path, columns)


def _first_existing(paths: Iterable[Path]) -> Path:
    candidates = [Path(p) for p in paths]
    for path in candidates:
        if path.exists():
            return path
    joined = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No input table found. Tried:\n  {joined}")


def _fit_comparison_colors() -> dict[str, object]:
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = cycle.by_key().get("color", []) if cycle is not None else []
    return {
        "totani": plt.rcParams.get("text.color", "black"),
        "global": colors[3] if len(colors) > 3 else plasma_color(0.58),
        "global_disk": colors[1] if len(colors) > 1 else plasma_color(0.30),
    }


def totani_table_path(fig: int, tag: str) -> Path:
    suffix = "" if str(tag).lower() in {"", "none", "default"} else f"_{tag}"
    return _first_existing(
        [
            TOTANI_FIG_DIR / f"plots_fig{fig}{suffix}" / f"mcmc_fig{fig}_f_ml.txt",
            TOTANI_FIG_DIR / f"plots_fig{fig}" / f"mcmc_fig{fig}_f_ml.txt",
            # Legacy fallback: tables that used to live next to this script.
            FIGURES_DIR / f"plots_fig{fig}{suffix}" / f"mcmc_fig{fig}_f_ml.txt",
            FIGURES_DIR / f"plots_fig{fig}" / f"mcmc_fig{fig}_f_ml.txt",
        ]
    )


def global_table_path(fig: int, tag: str = "global") -> Path:
    tag_l = str(tag).strip().lower()
    if tag_l in {"pixelwise", "pixelwise_mcmc"}:
        return _first_existing(
            [
                FIGURES_DIR / "pixelwise_fit" / f"plots_pixelwise_global_fig{fig}" / "mcmc_global_f_ml.txt",
                FIGURES_DIR / f"plots_pixelwise_global_fig{fig}" / "mcmc_global_f_ml.txt",
            ]
        )
    return _first_existing(
        [
            FIGURES_DIR / "global_fit" / f"plots_global_fig{fig}" / "mcmc_global_f_ml.txt",
            FIGURES_DIR / f"plots_global_fig{fig}" / "mcmc_global_f_ml.txt",
            FIGURES_DIR / "plots_global" / "mcmc_global_f_ml.txt",
        ]
    )


def totani_mcmc_dir(fig: int, tag: str) -> Path:
    suffix = "" if str(tag).lower() in {"", "none", "default"} else f"_{tag}"
    return _first_existing(
        [
            MCMC_DIR / f"mcmc_results_fig{fig}{suffix}",
            MCMC_DIR / f"mcmc_results_fig{fig}",
        ]
    )


def global_mcmc_dir(fig: int, tag: str = "global") -> Path:
    tag_l = str(tag).strip().lower()
    if tag_l in {"pixelwise", "pixelwise_mcmc"}:
        fig_key = "2_3" if int(fig) in (2, 3) else str(int(fig))
        return _first_existing(
            [
                FIT_MCMC_DIR / "fit_results" / "pixelwise_mcmc" / f"pixelwise_mcmc_results_fig{fig_key}",
                FIT_MCMC_DIR / f"pixelwise_mcmc_results_fig{fig_key}",
            ]
        )
    if int(fig) in (2, 3):
        return _first_existing(
            [
                FIT_MCMC_DIR / "fit_results" / "quick_global" / "global_fit_results_fig2_3",
                FIT_MCMC_DIR / "global_fit_results_fig2_3",
            ]
        )
    return _first_existing(
        [
            FIT_MCMC_DIR / "fit_results" / "quick_global" / f"global_fit_results_fig{fig}",
            FIT_MCMC_DIR / f"global_fit_results_fig{fig}",
        ]
    )


def global_with_disk_mcmc_dir(fig: int, tag: str = "global") -> Path:
    """Disk-included counterpart of :func:`global_mcmc_dir` (the ``*_w_disk`` fit)."""
    tag_l = str(tag).strip().lower()
    if tag_l in {"pixelwise", "pixelwise_mcmc"}:
        return _first_existing(
            [
                FIT_MCMC_DIR / "fit_results" / "pixelwise_mcmc" / f"pixelwise_mcmc_results_fig{fig}_w_disk",
                FIT_MCMC_DIR / f"pixelwise_mcmc_results_fig{fig}_w_disk",
            ]
        )
    return _first_existing(
        [
            FIT_MCMC_DIR / "fit_results" / "quick_global" / f"global_fit_results_fig{fig}_w_disk",
            FIT_MCMC_DIR / f"global_fit_results_fig{fig}_w_disk",
        ]
    )


def _load_energy_centres_gev(counts_path: Path) -> np.ndarray:
    _, _, _, _, ectr_mev, _ = read_counts_and_ebounds(str(counts_path))
    return np.asarray(ectr_mev, float) / 1000.0


def _load_pole_flux_from_npz(
    *,
    mcmc_dir: Path,
    nfw_label: str,
    nE: int,
    central_stat: str,
    iso_target_e2_default: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load Fig. 8 NFW pole flux from all per-energy-bin MCMC npz files.

    This intentionally does no likelihood-quality filtering. If a bin has a
    finite stored MCMC result, it is plotted.
    """
    y_ctr = np.full(int(nE), np.nan)
    y_p16 = np.full(int(nE), np.nan)
    y_p84 = np.full(int(nE), np.nan)

    for k in range(int(nE)):
        path = Path(mcmc_dir) / f"mcmc_results_k{k:02d}.npz"
        if not path.exists():
            continue

        npz = np.load(path, allow_pickle=True)
        iso_target_e2 = float(npz["iso_target_e2"] if "iso_target_e2" in npz else iso_target_e2_default)
        labels = [str(x) for x in npz["labels"].tolist()]

        if nfw_label in labels:
            idx = labels.index(nfw_label)
        else:
            nfw_candidates = [i for i, label in enumerate(labels) if "nfw" in label.lower()]
            if len(nfw_candidates) != 1:
                raise RuntimeError(
                    f"Could not identify one NFW component in {path}. "
                    f"Requested '{nfw_label}', available NFW-like labels: "
                    f"{[labels[i] for i in nfw_candidates]}"
                )
            idx = nfw_candidates[0]

        if central_stat not in npz:
            raise KeyError(f"Stat '{central_stat}' not found in {path}. Keys: {list(npz.keys())}")

        y_ctr[k] = float(npz[central_stat][idx]) * iso_target_e2
        if "f_p16" in npz and "f_p84" in npz:
            y_p16[k] = float(npz["f_p16"][idx]) * iso_target_e2
            y_p84[k] = float(npz["f_p84"][idx]) * iso_target_e2

    if np.all(~np.isfinite(y_p16)) or np.all(~np.isfinite(y_p84)):
        return y_ctr, None

    err_lo = np.clip(y_ctr - y_p16, 0.0, None)
    err_hi = np.clip(y_p84 - y_ctr, 0.0, None)
    err = np.vstack([err_lo, err_hi])
    err[~np.isfinite(err)] = 0.0
    return y_ctr, err


def _parse_loglike_txt(path: Path) -> float:
    if not Path(path).exists():
        return np.nan
    with Path(path).open("r") as f:
        for line in f:
            m = re.search(r"Log-likelihood \(ML\):\s*([+-]?\S+)", line)
            if m:
                return float(m.group(1))
    return np.nan


def _load_loglike_series(
    *,
    mcmc_dir: Path,
    nE: int,
    max_abs_loglike: float | None,
    label: str,
) -> np.ndarray:
    lnL = np.full(int(nE), np.nan)
    masked = []
    for k in range(int(nE)):
        lnL[k] = _parse_loglike_txt(Path(mcmc_dir) / f"mcmc_results_k{k:02d}.txt")
        if (
            max_abs_loglike is not None
            and max_abs_loglike > 0
            and np.isfinite(lnL[k])
            and abs(float(lnL[k])) > float(max_abs_loglike)
        ):
            masked.append((k, float(lnL[k])))
            lnL[k] = np.nan
    if masked:
        bins = ", ".join(f"k{k:02d}={val:.3e}" for k, val in masked)
        print(f"WARNING: masked pathological log-likelihoods for {label}: {bins}")
    return lnL


def _mask_fig9_dlnl(
    dlnl: np.ndarray,
    *,
    max_abs_dlnl: float | None,
    label: str,
) -> np.ndarray:
    if max_abs_dlnl is None or max_abs_dlnl <= 0:
        return np.asarray(dlnl, float)
    out = np.asarray(dlnl, float).copy()
    mask = np.isfinite(out) & (np.abs(out) > float(max_abs_dlnl))
    if np.any(mask):
        bins = ", ".join(f"k{k:02d}={out[k]:.3e}" for k in np.flatnonzero(mask))
        print(f"WARNING: masked pathological Fig. 9 ΔlnL values for {label}: {bins}")
        out[mask] = np.nan
    return out


def _available_components(
    totani: CurveTable,
    global_fit: CurveTable,
    requested: Sequence[str],
) -> List[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        ordered = DEFAULT_COMPONENTS + [
            c
            for c in totani.components() + global_fit.components()
            if c not in DEFAULT_COMPONENTS and c != "total"
        ]
    else:
        ordered = [str(c) for c in requested]

    seen = set()
    out = []
    for comp in ordered:
        if comp in seen:
            continue
        seen.add(comp)
        if totani.has(comp) or global_fit.has(comp):
            out.append(comp)
    return out


def _finite_y_values(*tables: CurveTable, components: Sequence[str]) -> np.ndarray:
    ys = []
    for table in tables:
        for comp in components:
            if table.has(comp):
                y = np.asarray(table.y(comp), float)
                ys.append(y[np.isfinite(y)])
    if not ys:
        return np.array([], dtype=float)
    return np.concatenate(ys)


def _symlog_linthresh(*tables: CurveTable, components: Sequence[str]) -> float:
    y = _finite_y_values(*tables, components=components)
    y = np.abs(y[np.isfinite(y) & (y != 0.0)])
    if y.size == 0:
        return 1e-8
    return float(max(np.nanpercentile(y, 10.0) / 5.0, np.nanmax(y) * 1e-4, 1e-12))


def _positive_ylim_if_possible(*tables: CurveTable, components: Sequence[str]) -> tuple[float, float] | None:
    y = _finite_y_values(*tables, components=components)
    y = y[np.isfinite(y)]
    if y.size == 0 or np.nanmin(y) <= 0:
        return None
    ymin = 10e-7
    ymax = float(np.nanmax(y))
    if ymin <= 0 or ymax <= 0 or not np.isfinite(ymin) or not np.isfinite(ymax):
        return None
    return ymin / 1.8, ymax * 1.8


def _mask_xy(table: CurveTable, component: str) -> np.ndarray:
    x = np.asarray(table.energy, float)
    y = np.asarray(table.y(component), float)
    return np.isfinite(x) & np.isfinite(y) & (x > 0)


def _plot_component(
    ax,
    table: CurveTable,
    component: str,
    *,
    kind: str,
    show_error_bands: bool,
    show_errorbars: bool,
    label: str | None,
    y_scale: float = 1.0,
) -> None:
    if not table.has(component):
        return

    x = np.asarray(table.energy, float)
    y = np.asarray(table.y(component), float) / float(y_scale)
    m = _mask_xy(table, component)
    if not np.any(m):
        return

    color = COMPONENT_COLORS.get(component, None)
    marker = COMPONENT_MARKERS.get(component, "o")
    is_global = kind == "global"
    lw = 2.4 if component == "total" else 1.45
    ms = 4.4 if component == "total" else 3.6
    alpha = 0.96 if component in {"total", "nfw"} else 0.78

    ax.plot(
        x[m],
        y[m],
        color=color,
        lw=lw,
        ls="--" if is_global else "-",
        marker=marker,
        ms=ms,
        markerfacecolor="white" if is_global else color,
        markeredgecolor=color,
        markeredgewidth=1.0,
        alpha=alpha,
        label=label,
        zorder=4 if component in {"total", "nfw"} else 3,
    )

    yerr = table.yerr(component)
    if yerr is None:
        return
    err = np.asarray(yerr, float)
    err = err / float(y_scale)

    if show_error_bands:
        lo = y - err[0]
        hi = y + err[1]
        ax.fill_between(
            x[m],
            lo[m],
            hi[m],
            color=color,
            alpha=0.09 if is_global else 0.12,
            linewidth=0,
            zorder=1,
        )

    if show_errorbars:
        ax.errorbar(
            x[m],
            y[m],
            yerr=err[:, m],
            color=color,
            fmt="none",
            elinewidth=0.8,
            capsize=1.8,
            alpha=0.35 if is_global else 0.45,
            zorder=2,
        )


def _fractional_difference(
    global_fit: CurveTable,
    totani: CurveTable,
    component: str,
    *,
    floor_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    if not (global_fit.has(component) and totani.has(component)):
        return np.array([]), np.array([])
    xg = np.asarray(global_fit.energy, float)
    xt = np.asarray(totani.energy, float)
    if xg.shape != xt.shape or not np.allclose(xg, xt, rtol=1e-5, atol=1e-8):
        return np.array([]), np.array([])

    yg = np.asarray(global_fit.y(component), float)
    yt = np.asarray(totani.y(component), float)
    finite = np.isfinite(xg) & np.isfinite(yg) & np.isfinite(yt) & (xg > 0)
    scale = np.nanmax(np.abs(yt[finite])) if np.any(finite) else np.nan
    floor = float(floor_fraction) * scale if np.isfinite(scale) and scale > 0 else 1e-30
    denom = np.maximum(np.abs(yt), floor)
    frac = (yg - yt) / denom
    frac[~np.isfinite(frac)] = np.nan
    return xg[finite], frac[finite]


def _plot_fig8_series(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    *,
    color: str,
    marker: str,
    label: str,
    markerfacecolor: str,
    show_errorbars: bool,
    x_shift: float = 1.0,
) -> None:
    x = np.asarray(x, float) * float(x_shift)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(m):
        return

    if show_errorbars and yerr is not None:
        err = np.asarray(yerr, float)
        ax.errorbar(
            x[m],
            y[m],
            yerr=err[:, m],
            linestyle="none",
            color=color,
            ecolor=color,
            marker=marker,
            mfc=markerfacecolor,
            mec=color,
            mew=0.7,
            markersize=5.5,
            elinewidth=0.7,
            capsize=2,
            label=label,
        )
        return

    ax.plot(
        x[m],
        y[m],
        linestyle="none",
        color=color,
        marker=marker,
        mfc=markerfacecolor,
        mec=color,
        mew=0.7,
        markersize=5.5,
        label=label,
    )


FIG8_X_SHIFT_TOTANI = 0.965
FIG8_X_SHIFT_GLOBAL = 1.037
FIG8_X_SHIFT_GLOBAL_DISK = 1.075


def _fig8_zoom_ylim(*series: tuple[np.ndarray, np.ndarray | None]) -> tuple[float, float]:
    vals = []
    for y, yerr in series:
        y = np.asarray(y, float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        if yerr is not None:
            err = np.asarray(yerr, float)
            vals.append((y - err[0])[finite])
            vals.append((y + err[1])[finite])
        else:
            vals.append(y[finite])

    if not vals:
        return FIG8_PANELS[0][4]

    all_vals = np.concatenate(vals)
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        return FIG8_PANELS[0][4]

    pad = 0.12 * (ymax - ymin)
    ymin = min(ymin - pad, 0.0 - 0.08 * (ymax - ymin))
    ymax = max(ymax + pad, 0.0 + 0.08 * (ymax - ymin))
    return ymin, ymax


def plot_fig8_top_panel_zoom_overlay(
    *,
    outdir: Path,
    counts_path: Path,
    totani_tag: str,
    global_tag: str,
    central_stat: str,
    manifest_lines: List[str],
    show_errorbars: bool,
) -> Path:
    energy_gev = _load_energy_centres_gev(counts_path)
    nE = int(energy_gev.size)
    source_fig, title, nfw_label, marker, _, _ = FIG8_PANELS[0]
    totani_dir = totani_mcmc_dir(source_fig, totani_tag)
    global_dir = global_mcmc_dir(source_fig, global_tag)
    manifest_lines.append(f"fig8 top zoom source fig{source_fig} totani_mcmc: {totani_dir}")
    manifest_lines.append(f"fig8 top zoom source fig{source_fig} global_mcmc : {global_dir}")

    y_totani, yerr_totani = _load_pole_flux_from_npz(
        mcmc_dir=totani_dir,
        nfw_label=nfw_label,
        nE=nE,
        central_stat=central_stat,
    )
    y_global, yerr_global = _load_pole_flux_from_npz(
        mcmc_dir=global_dir,
        nfw_label=nfw_label,
        nE=nE,
        central_stat=central_stat,
    )

    fit_colors = _fit_comparison_colors()
    fg = fit_colors["totani"]
    global_color = fit_colors["global"]
    fig_obj, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))

    _plot_fig8_series(
        ax,
        energy_gev,
        y_totani,
        yerr_totani,
        color=fg,
        marker=marker,
        label="Totani",
        markerfacecolor="white",
        show_errorbars=show_errorbars,
        x_shift=FIG8_X_SHIFT_TOTANI,
    )
    _plot_fig8_series(
        ax,
        energy_gev,
        y_global,
        yerr_global,
        color=global_color,
        marker=marker,
        label="Global fit",
        markerfacecolor=global_color,
        show_errorbars=show_errorbars,
        x_shift=FIG8_X_SHIFT_GLOBAL,
    )

    ax.set_xscale("log")
    ax.axhline(0.0, color=fg, lw=0.7, ls=":")
    ax.text(0.04, 0.89, title, transform=ax.transAxes, fontsize=10.5)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=9.5)
    ax.grid(True, which="major", alpha=0.16, linewidth=0.6)
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(-0.7e-5, 2.2e-5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("photon energy [GeV]", fontsize=10.5)
    ax.set_ylabel(
        r"flux $E^2 dN/dE$ at Galactic poles  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        fontsize=10.5,
    )
    ax.legend(loc="upper right", fontsize=9.2, frameon=False)
    fig_obj.tight_layout()

    out = outdir / "fig8_top_panel_zoom_global_vs_totani_overlay.png"
    fig_obj.savefig(out, dpi=300, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig_obj)
    return out


def plot_fig8_middle_panel_zoom_overlay(
    *,
    outdir: Path,
    counts_path: Path,
    totani_tag: str,
    global_tag: str,
    central_stat: str,
    manifest_lines: List[str],
    show_errorbars: bool,
) -> Path:
    energy_gev = _load_energy_centres_gev(counts_path)
    nE = int(energy_gev.size)
    source_fig, title, nfw_label, marker, _, _ = FIG8_PANELS[1]
    totani_dir = totani_mcmc_dir(source_fig, totani_tag)
    global_dir = global_mcmc_dir(source_fig, global_tag)
    if str(global_tag).strip().lower() in {"pixelwise", "pixelwise_mcmc"}:
        global_with_disk_dir = _first_existing(
            [
                FIT_MCMC_DIR / "fit_results" / "pixelwise_mcmc" / "pixelwise_mcmc_results_fig6_w_disk",
                FIT_MCMC_DIR / "pixelwise_mcmc_results_fig6_w_disk",
            ]
        )
    else:
        global_with_disk_dir = _first_existing(
            [
                FIT_MCMC_DIR / "fit_results" / "quick_global" / "global_fit_results_fig6_w_disk",
                FIT_MCMC_DIR / "global_fit_results_fig6_w_disk",
            ]
        )
    manifest_lines.append(f"fig8 middle zoom source fig{source_fig} totani_mcmc: {totani_dir}")
    manifest_lines.append(f"fig8 middle zoom source fig{source_fig} global_mcmc_no_disk : {global_dir}")
    manifest_lines.append(
        f"fig8 middle zoom source fig{source_fig} global_mcmc_with_disk : {global_with_disk_dir}"
    )

    y_totani, yerr_totani = _load_pole_flux_from_npz(
        mcmc_dir=totani_dir,
        nfw_label=nfw_label,
        nE=nE,
        central_stat=central_stat,
    )
    y_global, yerr_global = _load_pole_flux_from_npz(
        mcmc_dir=global_dir,
        nfw_label=nfw_label,
        nE=nE,
        central_stat=central_stat,
    )
    y_global_with_disk, yerr_global_with_disk = _load_pole_flux_from_npz(
        mcmc_dir=global_with_disk_dir,
        nfw_label=nfw_label,
        nE=nE,
        central_stat=central_stat,
    )

    fit_colors = _fit_comparison_colors()
    fg = fit_colors["totani"]
    global_color = fit_colors["global"]
    global_with_disk_color = fit_colors["global_disk"]
    fig_obj, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))

    _plot_fig8_series(
        ax,
        energy_gev,
        y_totani,
        yerr_totani,
        color=fg,
        marker=marker,
        label="Totani (fit excluding disk)",
        markerfacecolor="white",
        show_errorbars=show_errorbars,
        x_shift=FIG8_X_SHIFT_TOTANI,
    )
    _plot_fig8_series(
        ax,
        energy_gev,
        y_global,
        yerr_global,
        color=global_color,
        marker=marker,
        label="Global fit excluding disk",
        markerfacecolor=global_color,
        show_errorbars=show_errorbars,
        x_shift=FIG8_X_SHIFT_GLOBAL,
    )
    _plot_fig8_series(
        ax,
        energy_gev,
        y_global_with_disk,
        yerr_global_with_disk,
        color=global_with_disk_color,
        marker=marker,
        label="Global fit including disk",
        markerfacecolor=global_with_disk_color,
        show_errorbars=show_errorbars,
        x_shift=FIG8_X_SHIFT_GLOBAL_DISK,
    )

    ax.set_xscale("log")
    ax.axhline(0.0, color=fg, lw=0.7, ls=":")
    ax.text(0.04, 0.89, title, transform=ax.transAxes, fontsize=10.5)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=9.5)
    ax.grid(True, which="major", alpha=0.16, linewidth=0.6)
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(-0.7e-5, 4.2e-5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("photon energy [GeV]", fontsize=10.5)
    ax.set_ylabel(
        r"flux $E^2 dN/dE$ at Galactic poles  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        fontsize=10.5,
    )
    ax.legend(loc="upper right", fontsize=9.2, frameon=False)
    fig_obj.tight_layout()

    out = outdir / "fig8_middle_panel_zoom_global_vs_totani_overlay.png"
    fig_obj.savefig(out, dpi=300, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig_obj)
    return out


def _style_spectrum_axes(ax, ratio_ax, *, title: str, use_log_y: bool, linthresh: float) -> None:
    del title
    ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
        ax.axhline(0.0, color="0.45", lw=0.75, alpha=0.7)
    ax.set_ylabel(r"$E^2\,dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)

    ratio_ax.set_xscale("log")
    ratio_ax.axhline(0.0, color="0.35", lw=0.8)
    ratio_ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    ratio_ax.set_xlabel("Energy [GeV]")
    ratio_ax.set_ylabel(r"$\Delta$/Totani")


def plot_fig_overlay(
    *,
    fig: int,
    totani: CurveTable,
    global_fit: CurveTable,
    outdir: Path,
    components: Sequence[str],
    ratio_components: Sequence[str],
    show_error_bands: bool,
    show_errorbars: bool,
    force_symlog: bool,
    ymin: float | None,
) -> Path:
    comps = _available_components(totani, global_fit, components)
    if not comps:
        raise RuntimeError(f"No requested components are present for fig{fig}")

    use_log_y = False
    requested_negative_ymin = ymin is not None and np.isfinite(ymin) and ymin <= 0.0
    positive_ylim = None if (force_symlog or requested_negative_ymin) else _positive_ylim_if_possible(totani, global_fit, components=comps)
    if positive_ylim is not None:
        use_log_y = True
    linthresh = _symlog_linthresh(totani, global_fit, components=comps)

    fig_obj, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.0], "hspace": 0.06},
    )

    for comp in comps:
        comp_label = COMPONENT_LABELS.get(comp, comp)
        _plot_component(
            ax,
            totani,
            comp,
            kind="totani",
            show_error_bands=show_error_bands,
            show_errorbars=show_errorbars,
            label=comp_label,
        )
        _plot_component(
            ax,
            global_fit,
            comp,
            kind="global",
            show_error_bands=False,
            show_errorbars=show_errorbars and comp in {"total", "nfw"},
            label=None,
        )

    ratio_seen = False
    for comp in ratio_components:
        if comp not in comps:
            continue
        x, frac = _fractional_difference(global_fit, totani, comp)
        if x.size == 0:
            continue
        rax.plot(
            x,
            frac,
            color=COMPONENT_COLORS.get(comp),
            marker=COMPONENT_MARKERS.get(comp, "o"),
            lw=1.4,
            ms=3.5,
            label=COMPONENT_LABELS.get(comp, comp),
        )
        ratio_seen = True

    if not ratio_seen:
        rax.text(0.03, 0.5, "No matching ratio components", transform=rax.transAxes, va="center")
    rax.set_ylim(-2.2, 2.2)

    _style_spectrum_axes(
        ax,
        rax,
        title=FIG_TITLES.get(fig, f"Fig. {fig}"),
        use_log_y=use_log_y,
        linthresh=linthresh,
    )
    if positive_ylim is not None:
        ax.set_ylim(*positive_ylim)
    elif ymin is not None and np.isfinite(ymin):
        _, ymax = ax.get_ylim()
        finite_y = _finite_y_values(totani, global_fit, components=comps)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size:
            ymax = max(float(ymax), float(np.nanmax(finite_y)) * 1.15)
        ax.set_ylim(float(ymin), float(ymax))

    comp_legend = ax.legend(
        loc="upper right",
        ncols=2,
        fontsize=8.2,
        frameon=False,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=2.0,
    )
    ax.add_artist(comp_legend)

    fit_colors = _fit_comparison_colors()
    style_handles = [
        Line2D(
            [0],
            [0],
            color=fit_colors["totani"],
            lw=2.0,
            ls="-",
            marker="o",
            markerfacecolor=fit_colors["totani"],
            label="Totani",
        ),
        Line2D(
            [0],
            [0],
            color=fit_colors["totani"],
            lw=2.0,
            ls="--",
            marker="o",
            markerfacecolor="white",
            label="Global fit",
        ),
    ]
    ax.legend(
        handles=style_handles,
        loc="lower left",
        fontsize=8.4,
        frameon=False,
        borderpad=0.45,
        handlelength=2.4,
    )
    if ratio_seen:
        rax.legend(loc="best", fontsize=8.0, frameon=False, ncols=2)

    out = outdir / f"fig{fig}_global_vs_totani_overlay.png"
    fig_obj.savefig(out, dpi=300, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig_obj)
    return out


def plot_fig8_overlay(
    *,
    outdir: Path,
    counts_path: Path,
    totani_tag: str,
    global_tag: str,
    central_stat: str,
    manifest_lines: List[str],
    show_error_bands: bool,
    show_errorbars: bool,
) -> Path:
    del show_error_bands

    energy_gev = _load_energy_centres_gev(counts_path)
    nE = int(energy_gev.size)
    fig_obj, axes = plt.subplots(3, 1, figsize=(5.0, 8.0), sharex=True)
    fit_colors = _fit_comparison_colors()
    fg = fit_colors["totani"]
    global_color = fit_colors["global"]
    global_with_disk_color = fit_colors["global_disk"]
    fit_word = "Pixelwise" if str(global_tag).strip().lower() in {"pixelwise", "pixelwise_mcmc"} else "Global"

    for ax, (source_fig, title, nfw_label, marker, ylim, yticks) in zip(axes, FIG8_PANELS):
        totani_dir = totani_mcmc_dir(source_fig, totani_tag)
        global_dir = global_mcmc_dir(source_fig, global_tag)
        manifest_lines.append(f"fig8 source fig{source_fig} totani_mcmc: {totani_dir}")
        manifest_lines.append(f"fig8 source fig{source_fig} global_mcmc : {global_dir}")

        y_totani, yerr_totani = _load_pole_flux_from_npz(
            mcmc_dir=totani_dir,
            nfw_label=nfw_label,
            nE=nE,
            central_stat=central_stat,
        )
        y_global, yerr_global = _load_pole_flux_from_npz(
            mcmc_dir=global_dir,
            nfw_label=nfw_label,
            nE=nE,
            central_stat=central_stat,
        )

        # Disk-included counterpart. Skip gracefully if the *_w_disk fit is absent.
        try:
            global_with_disk_dir = global_with_disk_mcmc_dir(source_fig, global_tag)
        except FileNotFoundError:
            global_with_disk_dir = None
            manifest_lines.append(
                f"fig8 source fig{source_fig} global_mcmc_with_disk : (missing, skipped)"
            )
        else:
            manifest_lines.append(
                f"fig8 source fig{source_fig} global_mcmc_with_disk : {global_with_disk_dir}"
            )
            y_global_with_disk, yerr_global_with_disk = _load_pole_flux_from_npz(
                mcmc_dir=global_with_disk_dir,
                nfw_label=nfw_label,
                nE=nE,
                central_stat=central_stat,
            )

        _plot_fig8_series(
            ax,
            energy_gev,
            y_totani,
            yerr_totani,
            color=fg,
            marker=marker,
            label="Totani",
            markerfacecolor="white",
            show_errorbars=show_errorbars,
            x_shift=FIG8_X_SHIFT_TOTANI,
        )
        _plot_fig8_series(
            ax,
            energy_gev,
            y_global,
            yerr_global,
            color=global_color,
            marker=marker,
            label=f"{fit_word} fit (excl. disk)",
            markerfacecolor=global_color,
            show_errorbars=show_errorbars,
            x_shift=FIG8_X_SHIFT_GLOBAL,
        )
        if global_with_disk_dir is not None:
            _plot_fig8_series(
                ax,
                energy_gev,
                y_global_with_disk,
                yerr_global_with_disk,
                color=global_with_disk_color,
                marker=marker,
                label=f"{fit_word} fit (incl. disk)",
                markerfacecolor=global_with_disk_color,
                show_errorbars=show_errorbars,
                x_shift=FIG8_X_SHIFT_GLOBAL_DISK,
            )

        ax.set_xscale("log")
        ax.axhline(0.0, color=fg, lw=0.7, ls=":")
        ax.text(0.62, 0.84, title, transform=ax.transAxes, fontsize=10.0)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.tick_params(labelsize=9.0)
        ax.set_xlim(1.0, 1000.0)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    axes[-1].set_xlabel("photon energy [GeV]", fontsize=10.0)
    fig_obj.text(
        0.03,
        0.5,
        r"flux $E^2 dN/dE$ at Galactic poles  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        rotation=90,
        va="center",
        fontsize=11.0,
    )
    fig_obj.tight_layout(rect=(0.08, 0.04, 1.0, 1.0))

    out = outdir / "fig8_global_vs_totani_overlay.png"
    fig_obj.savefig(out, dpi=300, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig_obj)
    return out


def _plot_fig9_series(
    ax,
    energy_gev: np.ndarray,
    dlnl: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
    markerfacecolor: str,
    label: str,
) -> None:
    x = np.asarray(energy_gev, float)
    y = np.asarray(dlnl, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(m):
        return
    ax.plot(
        x[m],
        y[m],
        color=color,
        lw=1.9,
        ls=linestyle,
        marker=marker,
        ms=5.2,
        mfc=markerfacecolor,
        mec=color,
        mew=1.0,
        label=label,
    )


def plot_fig9_overlay(
    *,
    outdir: Path,
    counts_path: Path,
    totani_tag: str,
    global_tag: str,
    manifest_lines: List[str],
    max_abs_loglike: float | None,
    max_abs_dlnl: float | None,
) -> Path:
    energy_gev = _load_energy_centres_gev(counts_path)
    nE = int(energy_gev.size)

    totani_nohalo_dir = totani_mcmc_dir(4, totani_tag)
    global_nohalo_dir = global_mcmc_dir(4, global_tag)
    manifest_lines.append("fig9 disk variant: disk excluded, matching Totani Fig. 4-7 likelihood fits")
    manifest_lines.append(f"fig9 totani no-halo baseline, disk excluded: {totani_nohalo_dir}")
    manifest_lines.append(f"fig9 global no-halo baseline, disk excluded : {global_nohalo_dir}")

    lnL_totani_nohalo = _load_loglike_series(
        mcmc_dir=totani_nohalo_dir,
        nE=nE,
        max_abs_loglike=max_abs_loglike,
        label="Totani fig4 no halo",
    )
    lnL_global_nohalo = _load_loglike_series(
        mcmc_dir=global_nohalo_dir,
        nE=nE,
        max_abs_loglike=max_abs_loglike,
        label="Global fig4 no halo",
    )

    fig_obj, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
    any_series = False
    finite_values = []

    for source_fig, title, _nfw_label, marker, color in FIG9_VARIANTS:
        totani_dir = totani_mcmc_dir(source_fig, totani_tag)
        global_dir = global_mcmc_dir(source_fig, global_tag)
        manifest_lines.append(f"fig9 source fig{source_fig} totani_mcmc, disk excluded: {totani_dir}")
        manifest_lines.append(f"fig9 source fig{source_fig} global_mcmc, disk excluded : {global_dir}")

        lnL_totani_halo = _load_loglike_series(
            mcmc_dir=totani_dir,
            nE=nE,
            max_abs_loglike=max_abs_loglike,
            label=f"Totani fig{source_fig}",
        )
        lnL_global_halo = _load_loglike_series(
            mcmc_dir=global_dir,
            nE=nE,
            max_abs_loglike=max_abs_loglike,
            label=f"Global fig{source_fig}",
        )

        dlnl_totani = _mask_fig9_dlnl(
            lnL_totani_halo - lnL_totani_nohalo,
            max_abs_dlnl=max_abs_dlnl,
            label=f"Totani {title}",
        )
        dlnl_global = _mask_fig9_dlnl(
            lnL_global_halo - lnL_global_nohalo,
            max_abs_dlnl=max_abs_dlnl,
            label=f"Global {title}",
        )

        _plot_fig9_series(
            ax,
            energy_gev,
            dlnl_totani,
            color=color,
            marker=marker,
            linestyle="-",
            markerfacecolor="white",
            label=f"Totani {title}",
        )
        _plot_fig9_series(
            ax,
            energy_gev,
            dlnl_global,
            color=color,
            marker=marker,
            linestyle="--",
            markerfacecolor=color,
            label=f"Global {title}",
        )

        for values in (dlnl_totani, dlnl_global):
            values = np.asarray(values, float)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)
                any_series = True

    if not any_series:
        raise RuntimeError("No finite Fig. 9 likelihood overlay series could be built.")

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.axhline(0.0, color="0.45", lw=0.75, ls=":")
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=9.2)
    ax.set_xlabel("photon energy [GeV]", fontsize=10.5)
    ax.set_ylabel(r"$\ln(L)-\ln(L_{\rm no\text{-}halo})$", fontsize=10.5)

    if finite_values:
        y = np.concatenate(finite_values)
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.08 * (ymax - ymin)
            ax.set_ylim(min(ymin - pad, -0.5), ymax + pad)

    ax.legend(
        loc="upper right",
        fontsize=8.4,
        frameon=False,
        borderpad=0.45,
        handlelength=2.4,
    )
    fig_obj.tight_layout()

    out = outdir / "fig9_global_vs_totani_overlay.png"
    fig_obj.savefig(out, dpi=300, bbox_inches="tight", transparent=current_style_is_transparent())
    plt.close(fig_obj)
    return out


def _parse_list(raw: str | Sequence[str]) -> List[str]:
    if isinstance(raw, str):
        parts = []
        for chunk in raw.split(","):
            stripped = chunk.strip()
            if stripped:
                parts.append(stripped)
        return parts
    return [str(x) for x in raw]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Overlay global-fit and Totani-style spectrum results for figures 2-9."
    )
    ap.add_argument("--figs", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument(
        "--totani-tag",
        default="10deg",
        help="Totani plot directory suffix, e.g. '10deg' uses plots_figN_10deg. Use 'none' for plots_figN.",
    )
    ap.add_argument(
        "--global-tag",
        default="global",
        choices=["global", "pixelwise", "pixelwise_mcmc"],
        help=(
            "Global comparison family. 'global' uses L-BFGS-B global_fit_results_* "
            "and plots_global_fig*. 'pixelwise' uses pixelwise_mcmc_results_* and "
            "plots_pixelwise_global_fig*."
        ),
    )
    ap.add_argument(
        "--outdir",
        default=str(FIGURES_DIR / "plots_global_totani_overlay"),
        help="Directory where overlay figures are written.",
    )
    ap.add_argument(
        "--counts",
        default=str(DATA_DIR / "processed" / "counts_ccube_1000to1000000.fits"),
        help="Counts cube used only to recover the Fig. 8 energy bin centres.",
    )
    ap.add_argument(
        "--components",
        default="all",
        help="Comma-separated component list for figs 2-7, or 'all'.",
    )
    ap.add_argument(
        "--ratio-components",
        default="nfw",
        help="Comma-separated components to show in the fractional-difference panel.",
    )
    ap.add_argument(
        "--style",
        default="paper",
        choices=[
            "paper",
            "conference",
            "conference_dark",
            "conference_light",
            "dark_transparent",
            "light_transparent",
            "light",
            "dark",
        ],
    )
    ap.add_argument(
        "--no-error-bands",
        action="store_true",
        help="Disable shaded uncertainty bands on the main panel.",
    )
    ap.add_argument(
        "--errorbars",
        action="store_true",
        help="Draw error bars as well as lines. Useful for Fig. 8 or diagnostic plots.",
    )
    ap.add_argument(
        "--force-symlog",
        action="store_true",
        help="Use a symmetric-log y-axis even when all plotted values are positive.",
    )
    ap.add_argument(
        "--fig27-ymin",
        type=float,
        default=1e-7,
        help="Lower y-limit for figures 2-7. Default: +1e-7.",
    )
    ap.add_argument(
        "--fig8-central-stat",
        default="f_p50",
        choices=["f_p50", "f_ml", "f_p05", "f_p16", "f_p84", "f_p95"],
        help="Central MCMC statistic for Fig. 8. Default f_p50, matching make_totani_fig8.py.",
    )
    ap.add_argument(
        "--fig9-max-abs-loglike",
        type=float,
        default=DEFAULT_FIG9_MAX_ABS_LOGLIKE,
        help=(
            "Mask Fig. 9 saved ML log-likelihood values with abs(lnL) above this threshold. "
            "Use <=0 to disable. Default: 1e10."
        ),
    )
    ap.add_argument(
        "--fig9-max-abs-dlnl",
        type=float,
        default=DEFAULT_FIG9_MAX_ABS_DLOGLIKE,
        help=(
            "Mask Fig. 9 likelihood-improvement values with abs(Delta lnL) above this threshold. "
            "Use <=0 to disable. Default: 1e6."
        ),
    )
    args = ap.parse_args(argv)

    apply_plot_style(args.style, base_fontsize=10, linewidth=1.4, n_colors=12, cmap_name="plasma")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    components = _parse_list(args.components)
    ratio_components = _parse_list(args.ratio_components)

    made: List[Path] = []
    manifest_lines = [
        "Global-fit vs Totani overlay inputs",
        f"totani_tag: {args.totani_tag}",
        f"global_tag: {args.global_tag}",
        f"components: {','.join(components)}",
        f"ratio_components: {','.join(ratio_components)}",
        f"fig8_counts: {Path(args.counts).resolve()}",
        f"fig8_central_stat: {args.fig8_central_stat}",
        f"fig9_max_abs_loglike: {args.fig9_max_abs_loglike}",
        f"fig9_max_abs_dlnl: {args.fig9_max_abs_dlnl}",
        "",
    ]
    totani_cache: MutableMapping[int, CurveTable] = {}
    global_cache: MutableMapping[int, CurveTable] = {}
    manifest_seen = set()
    skipped: List[tuple[int, str]] = []

    def load_pair(fig: int) -> tuple[CurveTable, CurveTable]:
        if fig not in totani_cache:
            totani_cache[fig] = load_curve_table(totani_table_path(fig, args.totani_tag))
        if fig not in global_cache:
            global_cache[fig] = load_curve_table(global_table_path(fig, args.global_tag))
        if fig not in manifest_seen:
            manifest_lines.append(f"fig{fig} totani: {totani_cache[fig].path}")
            manifest_lines.append(f"fig{fig} global : {global_cache[fig].path}")
            manifest_seen.add(fig)
        return totani_cache[fig], global_cache[fig]

    for fig in args.figs:
        try:
            if fig == 8:
                made.append(
                    plot_fig8_overlay(
                        outdir=outdir,
                        counts_path=Path(args.counts).resolve(),
                        totani_tag=args.totani_tag,
                        global_tag=args.global_tag,
                        central_stat=args.fig8_central_stat,
                        manifest_lines=manifest_lines,
                        show_error_bands=not args.no_error_bands,
                        show_errorbars=True,
                    )
                )
                made.append(
                    plot_fig8_top_panel_zoom_overlay(
                        outdir=outdir,
                        counts_path=Path(args.counts).resolve(),
                        totani_tag=args.totani_tag,
                        global_tag=args.global_tag,
                        central_stat=args.fig8_central_stat,
                        manifest_lines=manifest_lines,
                        show_errorbars=True,
                    )
                )
                made.append(
                    plot_fig8_middle_panel_zoom_overlay(
                        outdir=outdir,
                        counts_path=Path(args.counts).resolve(),
                        totani_tag=args.totani_tag,
                        global_tag=args.global_tag,
                        central_stat=args.fig8_central_stat,
                        manifest_lines=manifest_lines,
                        show_errorbars=True,
                    )
                )
                continue

            if fig == 9:
                max_abs_loglike = (
                    None if float(args.fig9_max_abs_loglike) <= 0 else float(args.fig9_max_abs_loglike)
                )
                max_abs_dlnl = None if float(args.fig9_max_abs_dlnl) <= 0 else float(args.fig9_max_abs_dlnl)
                made.append(
                    plot_fig9_overlay(
                        outdir=outdir,
                        counts_path=Path(args.counts).resolve(),
                        totani_tag=args.totani_tag,
                        global_tag=args.global_tag,
                        manifest_lines=manifest_lines,
                        max_abs_loglike=max_abs_loglike,
                        max_abs_dlnl=max_abs_dlnl,
                    )
                )
                continue

            totani, global_fit = load_pair(fig)
            made.append(
                plot_fig_overlay(
                    fig=fig,
                    totani=totani,
                    global_fit=global_fit,
                    outdir=outdir,
                    components=components,
                    ratio_components=ratio_components,
                    show_error_bands=not args.no_error_bands,
                    show_errorbars=bool(args.errorbars),
                    force_symlog=bool(args.force_symlog),
                    ymin=float(args.fig27_ymin),
                )
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append((fig, reason))
            manifest_lines.append(f"fig{fig} skipped: {reason}")
            print(f"Skipping fig{fig}: {reason}", file=sys.stderr)
            continue

    if made:
        print("Wrote overlay figures:")
        for path in made:
            print(f"  {path}")
    else:
        print("No overlay figures were written.", file=sys.stderr)
    if skipped:
        print("Skipped figures:")
        for fig, reason in skipped:
            print(f"  fig{fig}: {reason}")
    manifest_path = outdir / "overlay_inputs.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"Input manifest:\n  {manifest_path}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
