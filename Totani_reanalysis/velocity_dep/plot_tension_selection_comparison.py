#!/usr/bin/env python3
"""Compare spectral and constrained dSph-tension selections on one plot.

This reads saved ``annihilation_fit_comparison.npz`` files from the tension
scan directories. No scans are rerun.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
sys.path.insert(0, str(REPO_DIR))

from helpers.plot_style import apply_plot_style, current_savefig_kwargs, get_cmap_colors  # noqa: E402


CHANNELS = ["WW", "bb", "tautau"]
CHANNEL_LABELS = {
    "WW": r"$W^+W^-$",
    "bb": r"$b\bar{b}$",
    "tautau": r"$\tau^+\tau^-$",
}
CHANNEL_MARKERS = {"WW": "s", "bb": "o", "tautau": "^"}

PROFILE_LABELS = {
    "rho2.5": r"$\rho^{2.5}$",
    "rho2": r"$\rho^2$",
    "rho1": r"$\rho^1$",
    "global_rho2.5": r"global $\rho^{2.5}$",
    "global_rho2": r"global $\rho^2$",
    "global_rho1": r"global $\rho^1$",
}

BASE_PROFILES = ["rho2.5", "rho2", "rho1"]
BASE_PROFILE_LABELS = {
    "rho2.5": r"$\rho^{2.5}$",
    "rho2": r"$\rho^2$",
    "rho1": r"$\rho^1$",
}
SERIES = {
    "totani_spectral": {
        "label": "Totani spectral",
        "sample": "Totani",
        "target": "spectral",
        "profile_prefix": "",
        "offset": -0.27,
        "color": "#4C78A8",
    },
    "totani_constrained": {
        "label": "Totani constrained",
        "sample": "Totani",
        "target": "constrained",
        "profile_prefix": "",
        "offset": -0.09,
        "color": "#59A14F",
    },
    "global_spectral": {
        "label": "Global spectral",
        "sample": "Global fit",
        "target": "spectral",
        "profile_prefix": "global_",
        "offset": 0.09,
        "color": "#F28E2B",
    },
    "global_constrained": {
        "label": "Global constrained",
        "sample": "Global fit",
        "target": "constrained",
        "profile_prefix": "global_",
        "offset": 0.27,
        "color": "#E15759",
    },
}


def _scalar(value) -> str:
    arr = np.asarray(value)
    return str(arr.item()) if arr.shape == () else str(value)


def _load_results(root: Path) -> List[dict]:
    rows = []
    for path in sorted(Path(root).glob("annihilation_fit_*_*/annihilation_fit_comparison.npz")):
        with np.load(path, allow_pickle=False) as npz:
            row = {key: npz[key] for key in npz.files}
        row["_path"] = str(path)
        rows.append(row)
    return rows


def _selection_values(result: dict, selection: str) -> dict:
    selection = str(selection).strip().lower()
    if selection == "spectral":
        return {
            "valid": True,
            "selection": "spectral",
            "mass": float(result["spectral_best_ann_mass_GeV"]),
            "chi2": float(result["spectral_best_chi2"]),
            "delta_chi2": float(result["spectral_best_delta_chi2"]),
            "chi2_per_dof": float(result["spectral_best_chi2_per_dof"]),
            "sigmav": float(result["spectral_best_sigmav_cm3_s"]),
            "dsph": float(result["spectral_best_sigmav_dsph_cm3_s"]),
            "tension": float(result["spectral_best_tension"]),
            "fallback": False,
        }

    if "constrained_best_i" in result and int(result["constrained_best_i"]) >= 0:
        return {
            "valid": True,
            "selection": "min_tension_in_cl",
            "mass": float(result["constrained_best_ann_mass_GeV"]),
            "chi2": float(result["constrained_best_chi2"]),
            "delta_chi2": float(result["constrained_best_delta_chi2"]),
            "chi2_per_dof": float(result["constrained_best_chi2_per_dof"]),
            "sigmav": float(result["constrained_best_sigmav_cm3_s"]),
            "dsph": float(result["constrained_best_sigmav_dsph_cm3_s"]),
            "tension": float(result["constrained_best_tension"]),
            "fallback": False,
        }

    idx = int(result["min_cl_tension_i"])
    if idx >= 0:
        chi2 = np.asarray(result["chi2"], dtype=float)
        sigmav = np.asarray(result["sigmav_cm3_s"], dtype=float)
        dsph = np.asarray(result["sigmav_dsph_cm3_s"], dtype=float)
        return {
            "valid": True,
            "selection": "min_tension_in_cl",
            "mass": float(result["min_cl_tension_ann_mass_GeV"]),
            "chi2": float(chi2[idx]) if idx < chi2.size else np.nan,
            "delta_chi2": float(chi2[idx] - float(result["chi2_min"])) if idx < chi2.size else np.nan,
            "chi2_per_dof": float(chi2[idx] / max(1, int(result["ndof"]))) if idx < chi2.size else np.nan,
            "sigmav": float(sigmav[idx]) if idx < sigmav.size else np.nan,
            "dsph": float(dsph[idx]) if idx < dsph.size else np.nan,
            "tension": float(result["min_cl_tension"]),
            "fallback": False,
        }

    return {
        "valid": False,
        "selection": "none",
        "mass": np.nan,
        "chi2": np.nan,
        "delta_chi2": np.nan,
        "chi2_per_dof": np.nan,
        "sigmav": np.nan,
        "dsph": np.nan,
        "tension": np.nan,
        "fallback": True,
    }


def _row_valid_with_dsph(vals: dict, result: dict) -> bool:
    tension = float(vals["tension"])
    delta = float(vals["delta_chi2"])
    delta_allowed = float(result.get("delta_chi2_allowed", np.nan))
    return (
        bool(vals["valid"])
        and np.isfinite(tension)
        and tension <= 1.0
        and np.isfinite(delta)
        and (not np.isfinite(delta_allowed) or delta <= delta_allowed + 1e-6)
    )


def _append_rows_for_dir(out: List[dict], *, root: Path, sample: str, target: str) -> None:
    selection = "spectral" if target == "spectral" else "dsph"
    for result in _load_results(root):
        profile = _scalar(result["halo_profile"])
        channel = _scalar(result["ann_channel"])
        vals = _selection_values(result, selection)
        out.append(
            {
                "sample": sample,
                "profile": profile,
                "channel": channel,
                "target": target,
                "valid": vals["valid"],
                "valid_within_cl_and_dsph": _row_valid_with_dsph(vals, result),
                "actual_selection": vals["selection"],
                "fallback": vals["fallback"],
                "m_ann_GeV": vals["mass"],
                "chi2": vals["chi2"],
                "delta_chi2": vals["delta_chi2"],
                "delta_chi2_allowed": float(result.get("delta_chi2_allowed", np.nan)),
                "chi2_per_dof": vals["chi2_per_dof"],
                "sigmav_cm3_s": vals["sigmav"],
                "dsph_limit_cm3_s": vals["dsph"],
                "tension_x_dsph": vals["tension"],
                "path": result["_path"],
            }
        )


def _build_rows(
    *,
    totani_spectral_dir: Path,
    totani_constrained_dir: Path,
    global_spectral_dir: Path,
    global_constrained_dir: Path,
) -> List[dict]:
    out = []
    _append_rows_for_dir(out, root=totani_spectral_dir, sample="Totani", target="spectral")
    _append_rows_for_dir(out, root=totani_constrained_dir, sample="Totani", target="constrained")
    _append_rows_for_dir(out, root=global_spectral_dir, sample="Global fit", target="spectral")
    _append_rows_for_dir(out, root=global_constrained_dir, sample="Global fit", target="constrained")
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "sample",
        "profile",
        "channel",
        "target",
        "valid",
        "valid_within_cl_and_dsph",
        "actual_selection",
        "fallback",
        "m_ann_GeV",
        "chi2",
        "delta_chi2",
        "delta_chi2_allowed",
        "chi2_per_dof",
        "sigmav_cm3_s",
        "dsph_limit_cm3_s",
        "tension_x_dsph",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _row_by_key(rows: Sequence[dict]) -> Dict[tuple[str, str, str, str], dict]:
    return {
        (str(r["sample"]), str(r["profile"]), str(r["channel"]), str(r["target"])): r
        for r in rows
    }


def _finite_positive(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]


def plot_comparison(rows: Sequence[dict], out_path: Path) -> None:
    apply_plot_style("paper", base_fontsize=10, linewidth=1.5, n_colors=8, cmap_name="plasma")
    by_key = _row_by_key(rows)

    all_y = _finite_positive([r["tension_x_dsph"] for r in rows])
    ymax = max(130.0, max(all_y) * 1.7) if all_y else 130.0
    ymin = min(0.08, min(all_y) / 1.6) if all_y else 0.08

    fig, ax = plt.subplots(1, 1, figsize=(10.6, 6.6))
    x_base = {profile: i + 1.0 for i, profile in enumerate(BASE_PROFILES)}
    channel_offsets = {"WW": -0.035, "bb": 0.0, "tautau": 0.035}
    ax.axhspan(ymin, 1.0, color="#7CE36B", alpha=0.12, zorder=0)
    ax.axhline(1.0, color="#2CBA4B", lw=2.3, label="dSph limit", zorder=1)
    ax.axhline(2.0, color="#17BECF", lw=1.7, ls="--", label="2x guide", zorder=1)

    for series_key, series in SERIES.items():
        color = series["color"]
        sample = series["sample"]
        target = series["target"]
        prefix = series["profile_prefix"]
        for channel in CHANNELS:
            xs = []
            ys = []
            fill_ok = []
            for base_profile in BASE_PROFILES:
                profile = f"{prefix}{base_profile}"
                row = by_key.get((sample, profile, channel, target))
                if row is None:
                    continue
                y = float(row["tension_x_dsph"])
                if not np.isfinite(y) or y <= 0:
                    continue
                x = x_base[base_profile] + float(series["offset"]) + channel_offsets[channel]
                xs.append(x)
                ys.append(y)
                fill_ok.append(bool(row.get("valid_within_cl_and_dsph", False)))

            if len(xs) >= 2:
                ax.plot(xs, ys, color=color, lw=1.7, alpha=0.78, zorder=2)
            for x, y, ok in zip(xs, ys, fill_ok):
                ax.plot(
                    x,
                    y,
                    marker=CHANNEL_MARKERS[channel],
                    ms=8.8,
                    color=color,
                    mec="k",
                    mfc=color if ok else "white",
                    mew=0.95 if ok else 1.3,
                    lw=0,
                    zorder=4,
                )

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.55, len(BASE_PROFILES) + 0.45)
    ax.set_xticks([x_base[p] for p in BASE_PROFILES])
    ax.set_xticklabels([BASE_PROFILE_LABELS[p] for p in BASE_PROFILES])
    ax.grid(axis="y", which="major", alpha=0.28)
    ax.grid(axis="y", which="minor", alpha=0.12)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Halo profile")
    ax.set_ylabel(
        r"Tension  $\langle\sigma v\rangle_\mathrm{fit}/\langle\sigma v\rangle_\mathrm{dSph}$"
    )

    channel_handles = [
        Line2D(
            [0],
            [0],
            marker=CHANNEL_MARKERS[ch],
            ms=8.5,
            color="0.25",
            mfc="0.25",
            mec="k",
            lw=0,
            label=CHANNEL_LABELS[ch],
        )
        for ch in CHANNELS
    ]
    fill_handles = [
        Line2D([0], [0], marker="o", ms=8.5, color="0.25", mfc="0.25", mec="k", lw=0,
               label="filled: within 95% CL and dSph-safe"),
        Line2D([0], [0], marker="o", ms=8.5, color="0.25", mfc="white", mec="k", lw=0,
               label="open: not dSph-safe within 95% CL"),
    ]
    series_handles = [
        Line2D([0], [0], color=series["color"], lw=2.0, label=series["label"])
        for series in SERIES.values()
    ]
    ref_handles = [
        Line2D([0], [0], color="#2CBA4B", lw=2.3, label="dSph limit (=1)"),
        Line2D([0], [0], color="#17BECF", lw=1.7, ls="--", label="2x guide"),
    ]

    ax.legend(
        handles=channel_handles + fill_handles + series_handles + ref_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=8.5,
    )

    fig.suptitle("Spectral and constrained tension by halo profile", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 0.76, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", **current_savefig_kwargs())
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--totani-dir",
        default=str(HERE / "results" / "tension_scan"),
        help="Backward-compatible default for both Totani spectral and constrained inputs.",
    )
    p.add_argument(
        "--global-dir",
        default=str(HERE / "results" / "tension_scan_global_spectral"),
        help="Backward-compatible default for both global spectral and constrained inputs.",
    )
    p.add_argument(
        "--totani-spectral-dir",
        default=None,
        help="Tension scan directory for Totani spectral target.",
    )
    p.add_argument(
        "--totani-constrained-dir",
        default=None,
        help="Tension scan directory for Totani constrained dSph-tension target.",
    )
    p.add_argument(
        "--global-spectral-dir",
        default=None,
        help="Tension scan directory for global spectral target.",
    )
    p.add_argument(
        "--global-constrained-dir",
        default=None,
        help="Tension scan directory for global constrained dSph-tension target.",
    )
    p.add_argument(
        "--outdir",
        default=str(HERE / "results" / "tension_selection_comparison"),
    )
    args = p.parse_args(argv)

    totani_spectral_dir = Path(args.totani_spectral_dir or args.totani_dir)
    totani_constrained_dir = Path(args.totani_constrained_dir or args.totani_dir)
    global_spectral_dir = Path(args.global_spectral_dir or args.global_dir)
    global_constrained_dir = Path(args.global_constrained_dir or args.global_dir)

    rows = _build_rows(
        totani_spectral_dir=totani_spectral_dir,
        totani_constrained_dir=totani_constrained_dir,
        global_spectral_dir=global_spectral_dir,
        global_constrained_dir=global_constrained_dir,
    )
    if not rows:
        raise SystemExit("No tension scan NPZ rows found.")

    outdir = Path(args.outdir)
    _write_csv(outdir / "tension_selection_comparison.csv", rows)
    plot_comparison(rows, outdir / "tension_selection_comparison.png")

    print(f"Wrote {outdir / 'tension_selection_comparison.csv'}")
    print(f"Wrote {outdir / 'tension_selection_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
