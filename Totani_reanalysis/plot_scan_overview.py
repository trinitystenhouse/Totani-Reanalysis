#!/usr/bin/env python3
"""Visualise scattering-scan best fits across halo profiles.

The script reads one or more ``compiled_best_fits*/scattering_*.csv`` tables and
produces:

  - a combined CSV of all scan rows
  - per-profile heatmaps for fit quality and best-fit parameters
  - a compact best-operator summary plot across profiles/channels
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
sys.path.insert(0, str(REPO_DIR))
from helpers.plot_style import apply_plot_style, current_savefig_kwargs  # noqa: E402


CHANNELS = ["WW", "bb", "tautau"]
OPERATORS = [
    "dipole_magnetic",
    "dipole_electric",
    "charge_radius",
    "anapole",
    "rayleigh_even",
    "rayleigh_odd",
    "rayleigh_full",
    "scalar_rayleigh",
]
PROFILE_ORDER = ["global_rho2", "global_rho2.5", "global_rho1", "rho2", "rho2.5", "rho1"]

METRICS = [
    ("delta_chi2_improvement", r"$\Delta\chi^2$ improvement", "viridis", "linear"),
    ("fit_chi2", r"best-fit $\chi^2$", "magma_r", "linear"),
    ("best_m_ann_GeV", r"$m_\mathrm{ann}$ [GeV]", "plasma", "log10"),
    ("best_m_scat_GeV", r"$m_\mathrm{scat}$ [GeV]", "cividis", "log10"),
    ("best_Lambda_GeV", r"$\Lambda$ [GeV]", "viridis", "log10"),
    ("tau_max", r"$\tau_\mathrm{max}$", "cividis", "log10"),
    ("best_tension_x_dsph", r"dSph tension", "magma", "log10"),
]


def _float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out


def _discover_compiled_dirs(results_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in results_dir.glob("compiled_best_fits*")
        if p.is_dir() and list(p.glob("scattering_annihilation_best_fits*.csv"))
    )


def _read_rows(compiled_dirs: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for compiled_dir in compiled_dirs:
        for csv_path in sorted(compiled_dir.glob("scattering_annihilation_best_fits*.csv")):
            with csv_path.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    row = dict(row)
                    row["compiled_dir"] = str(compiled_dir)
                    row["compiled_csv"] = str(csv_path)
                    rows.append(row)
    return rows


def _write_combined_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dedupe_best_chi2(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row.get("profile")), str(row.get("channel")), str(row.get("operator")))
        old = best.get(key)
        if old is None or _float(row.get("fit_chi2")) < _float(old.get("fit_chi2")):
            best[key] = dict(row)
    return list(best.values())


def _profiles(rows: Sequence[Dict[str, object]]) -> List[str]:
    present = sorted({str(r.get("profile", "")) for r in rows if str(r.get("profile", ""))})
    return [p for p in PROFILE_ORDER if p in present] + [p for p in present if p not in PROFILE_ORDER]


def _transform(value: float, mode: str) -> float:
    if not math.isfinite(value):
        return math.nan
    if mode == "log10":
        return math.log10(value) if value > 0 else math.nan
    return value


def _format_value(value: float, mode: str) -> str:
    if not math.isfinite(value):
        return ""
    if mode == "log10":
        raw = 10.0**value
        if raw < 1e-2 or raw >= 1e4:
            return f"{raw:.1e}"
        return f"{raw:.2g}"
    if abs(value) < 1e-2 or abs(value) >= 1e4:
        return f"{value:.1e}"
    return f"{value:.2f}"


def _matrix_for_profile(rows: Sequence[Dict[str, object]], profile: str, metric: str, mode: str) -> np.ndarray:
    matrix = np.full((len(OPERATORS), len(CHANNELS)), np.nan)
    for row in rows:
        if str(row.get("profile")) != profile:
            continue
        op = str(row.get("operator"))
        ch = str(row.get("channel"))
        if op not in OPERATORS or ch not in CHANNELS:
            continue
        matrix[OPERATORS.index(op), CHANNELS.index(ch)] = _transform(_float(row.get(metric)), mode)
    return matrix


def _plot_heatmap(
    *,
    rows: Sequence[Dict[str, object]],
    profile: str,
    metric: str,
    title: str,
    cmap: str,
    mode: str,
    outdir: Path,
) -> Path:
    mat = _matrix_for_profile(rows, profile, metric, mode)
    fig, ax = plt.subplots(figsize=(5.8, 6.0))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(CHANNELS)), CHANNELS)
    ax.set_yticks(np.arange(len(OPERATORS)), OPERATORS)
    ax.tick_params(axis="x", rotation=0)
    ax.set_title(f"{profile}: {title}")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if math.isfinite(float(val)):
                ax.text(j, i, _format_value(float(val), mode), ha="center", va="center", fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"log10({title})" if mode == "log10" else title)
    out = outdir / "heatmaps" / f"{profile}_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    return out


def _best_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    best = []
    for profile in _profiles(rows):
        for channel in CHANNELS:
            subset = [
                r for r in rows
                if str(r.get("profile")) == profile
                and str(r.get("channel")) == channel
                and math.isfinite(_float(r.get("fit_chi2")))
            ]
            if not subset:
                continue
            best.append(min(subset, key=lambda r: _float(r.get("fit_chi2"))))
    return best


def _plot_best_summary(rows: Sequence[Dict[str, object]], outdir: Path) -> Path:
    best = _best_rows(rows)
    if not best:
        raise RuntimeError("No finite best-fit rows to plot")

    labels = [f"{r['profile']}\n{r['channel']}" for r in best]
    chi2 = np.array([_float(r.get("fit_chi2")) for r in best], dtype=float)
    delta = np.array([_float(r.get("delta_chi2_improvement")) for r in best], dtype=float)
    tension = np.array([_float(r.get("best_tension_x_dsph")) for r in best], dtype=float)
    ops = [str(r.get("operator")) for r in best]

    fig, axes = plt.subplots(3, 1, figsize=(max(7.0, 0.62 * len(best)), 7.0), sharex=True)
    x = np.arange(len(best))

    axes[0].bar(x, chi2, color="#4C78A8")
    axes[0].set_ylabel(r"best $\chi^2$")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, delta, color="#59A14F")
    axes[1].axhline(0, color="0.25", lw=0.8)
    axes[1].set_ylabel(r"$\Delta\chi^2$")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, tension, color="#E15759")
    axes[2].axhline(1, color="0.25", lw=0.8, ls="--")
    axes[2].set_ylabel("dSph tension")
    axes[2].set_xticks(x, labels, rotation=45, ha="right")
    axes[2].grid(axis="y", alpha=0.25)

    for ax in axes:
        for i, op in enumerate(ops):
            ax.text(i, ax.get_ylim()[1], op.replace("_", "\n"), ha="center", va="top", fontsize=6.8)

    fig.suptitle("Best operator per halo profile and channel")
    out = outdir / "best_operator_by_profile_channel.png"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)
    return out


def _write_best_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    best = _best_rows(rows)
    fields = [
        "profile",
        "channel",
        "operator",
        "fit_chi2",
        "delta_chi2_improvement",
        "best_m_ann_GeV",
        "best_m_scat_GeV",
        "best_Lambda_GeV",
        "tau_max",
        "best_tension_x_dsph",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in best:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--compiled-dirs",
        nargs="*",
        default=None,
        help="Compiled result dirs. Default discovers Totani_reanalysis/results/compiled_best_fits*.",
    )
    p.add_argument(
        "--outdir",
        default=str(HERE / "results" / "scan_overview_global_profiles"),
    )
    p.add_argument(
        "--dedupe",
        default="best_chi2",
        choices=["best_chi2", "none"],
        help="How to handle duplicate profile/channel/operator rows across input dirs.",
    )
    p.add_argument("--style", default="paper")
    args = p.parse_args()

    apply_plot_style(args.style, base_fontsize=10, linewidth=1.4, n_colors=12, cmap_name="viridis")

    if args.compiled_dirs:
        compiled_dirs = [Path(p).resolve() for p in args.compiled_dirs]
    else:
        compiled_dirs = _discover_compiled_dirs(HERE / "results")
    if not compiled_dirs:
        raise SystemExit("No compiled result dirs found.")

    rows = _read_rows(compiled_dirs)
    if not rows:
        raise SystemExit("No scattering rows found in compiled dirs.")
    all_rows = rows
    if args.dedupe == "best_chi2":
        rows = _dedupe_best_chi2(rows)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _write_combined_csv(outdir / "combined_scattering_best_fits_all_inputs.csv", all_rows)
    _write_combined_csv(outdir / "combined_scattering_best_fits.csv", rows)
    _write_best_csv(outdir / "best_operator_by_profile_channel.csv", rows)

    made = []
    for profile in _profiles(rows):
        for metric, title, cmap, mode in METRICS:
            made.append(
                _plot_heatmap(
                    rows=rows,
                    profile=profile,
                    metric=metric,
                    title=title,
                    cmap=cmap,
                    mode=mode,
                    outdir=outdir,
                )
            )
    made.append(_plot_best_summary(rows, outdir))

    print(f"Read {len(rows)} rows from {len(compiled_dirs)} compiled dirs")
    print(f"Wrote overview outputs to {outdir}")
    for path in made:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
