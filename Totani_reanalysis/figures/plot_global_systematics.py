#!/usr/bin/env python3
"""Plot deterministic global-fit systematics for the rho2 halo spectrum."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RHO2 = "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"


def _resolve_halo_column(names: tuple[str, ...], label: str, path: Path) -> str:
    if label in names:
        return label
    if "nfw" in names:
        return "nfw"
    candidates = [
        n for n in names
        if "nfw" in n.lower() and not n.lower().endswith(("_err_lo", "_err_hi"))
    ]
    if len(candidates) != 1:
        raise KeyError(f"Could not identify NFW column in {path}; candidates={candidates}")
    return candidates[0]


def _load_table(path: Path, label: str = RHO2) -> dict:
    arr = np.genfromtxt(path, names=True, comments="#", dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([arr])
    names = arr.dtype.names or ()
    label = _resolve_halo_column(names, label, path)
    lo = f"{label}_err_lo"
    hi = f"{label}_err_hi"
    return {
        "E": np.asarray(arr["E_GeV"], float),
        "phi": np.asarray(arr[label], float),
        "err_lo": np.asarray(arr[lo], float) if lo in names else np.full(arr.size, np.nan),
        "err_hi": np.asarray(arr[hi], float) if hi in names else np.full(arr.size, np.nan),
        "label": label,
    }


def _load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return list(data.get("variants", []))


def _peak(E: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(E) & np.isfinite(y)
    if not np.any(good):
        return np.nan, np.nan
    idx_local = int(np.nanargmax(y[good]))
    return float(E[good][idx_local]), float(y[good][idx_local])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "mcmc" / "configs" / "systematics" / "manifest.json",
    )
    ap.add_argument("--global-root", type=Path, default=PROJECT / "mcmc" / "global_systematics")
    ap.add_argument("--plot-root", type=Path, default=HERE / "plots_global_systematics")
    ap.add_argument("--outdir", type=Path, default=HERE / "plots_global_systematics_summary")
    ap.add_argument("--baseline", default="baseline_rho2")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    variants = _load_manifest(args.manifest)
    loaded: dict[str, dict] = {}
    for item in variants:
        name = item["name"]
        table = args.plot_root / name / "mcmc_global_f_ml.txt"
        if not table.exists():
            table = args.global_root / name / "mcmc_global_f_ml.txt"
        if not table.exists():
            print(f"[skip] Missing coefficient table for {name}: {table}")
            continue
        try:
            loaded[name] = {**item, **_load_table(table)}
        except Exception as exc:
            print(f"[skip] Could not load {name}: {exc}")

    if args.baseline not in loaded:
        raise SystemExit(f"Baseline {args.baseline!r} missing; run it first.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = loaded[args.baseline]
    E0 = base["E"]
    phi0 = base["phi"]

    groups = sorted({d["group"] for d in loaded.values()})
    summary_lines = [
        "Global deterministic rho2 systematics summary",
        f"Baseline: {args.baseline}",
        "",
        f"{'variant':28s} {'group':18s} {'E_peak':>10s} {'phi_peak':>12s} {'max|frac shift|':>16s}",
        "-" * 90,
    ]
    for name, d in loaded.items():
        Epk, ypk = _peak(d["E"], d["phi"])
        interp_base = np.interp(d["E"], E0, phi0, left=np.nan, right=np.nan)
        frac = (d["phi"] - interp_base) / np.where(np.abs(interp_base) > 0, np.abs(interp_base), np.nan)
        max_shift = np.nanmax(np.abs(frac)) if np.any(np.isfinite(frac)) else np.nan
        summary_lines.append(
            f"{name:28s} {d['group']:18s} {Epk:10.3g} {ypk:12.4e} {max_shift:16.3g}"
        )
    (args.outdir / "systematics_summary.txt").write_text("\n".join(summary_lines) + "\n")

    for group in groups:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.errorbar(
            E0,
            phi0,
            yerr=[base["err_lo"], base["err_hi"]],
            fmt="o-",
            lw=2.0,
            ms=4.0,
            color="black",
            label="baseline",
            zorder=5,
        )
        for name, d in loaded.items():
            if name == args.baseline or d["group"] != group:
                continue
            ax.plot(d["E"], d["phi"], marker="o", ms=3.0, lw=1.4, label=name.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel(r"$E^2 dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.set_title(f"Global rho2 systematics: {group.replace('_', ' ')}")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for fmt in args.formats:
            fig.savefig(args.outdir / f"systematics_{group}.{fmt}", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axhline(0.0, color="0.6", lw=0.8)
    for name, d in loaded.items():
        if name == args.baseline:
            continue
        interp_base = np.interp(d["E"], E0, phi0, left=np.nan, right=np.nan)
        frac = (d["phi"] - interp_base) / np.where(np.abs(interp_base) > 0, np.abs(interp_base), np.nan)
        ax.plot(d["E"], frac, marker="o", ms=3.0, lw=1.0, label=name.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"fractional shift from baseline")
    ax.set_title("Global rho2 systematics fractional shifts")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for fmt in args.formats:
        fig.savefig(args.outdir / f"systematics_fractional_shifts.{fmt}", dpi=220)
    plt.close(fig)

    print(f"Wrote systematics summary plots to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
