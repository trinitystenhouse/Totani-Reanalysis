#!/usr/bin/env python3
"""Paper-summary plots for deterministic global-fit systematics.

This script consumes existing outputs only.  It expects the per-variant
``mcmc_global_f_ml.txt`` coefficient tables made by ``run_global_systematics.sh``
and, when available, the underlying ``global_fit_k??.npz`` files.

Outputs:
  * systematics_peak_energy_vs_amplitude.[png,pdf]
  * systematics_integrated_flux_ratio.[png,pdf]
  * systematics_delta_loglike_stability.[png,pdf]  (if baseline/no-halo exist)
  * residual_template_correlation_heatmap.[png,pdf] (if residual diagnostics exist)
  * systematics_dashboard_summary.txt
"""

from __future__ import annotations

import argparse
import json
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


def _load_manifest(path: Path) -> list[dict]:
    return list(json.loads(path.read_text()).get("variants", []))


def _load_table(path: Path, label: str = RHO2) -> dict:
    arr = np.genfromtxt(path, names=True, comments="#", dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([arr])
    names = arr.dtype.names or ()
    label = _resolve_halo_column(names, label, path)
    return {
        "E": np.asarray(arr["E_GeV"], float),
        "phi": np.asarray(arr[label], float),
        "label": label,
        "path": str(path),
    }


def _load_systematics(manifest: Path, plot_root: Path, global_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in _load_manifest(manifest):
        name = str(item["name"])
        table = plot_root / name / "mcmc_global_f_ml.txt"
        if not table.exists():
            table = global_root / name / "mcmc_global_f_ml.txt"
        if not table.exists():
            print(f"[skip] Missing systematics coefficient table for {name}: {table}")
            continue
        try:
            out[name] = {**item, **_load_table(table)}
        except Exception as exc:
            print(f"[skip] Could not load {name}: {exc}")
    return out


def _peak(E: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(E) & np.isfinite(y)
    if not np.any(good):
        return np.nan, np.nan
    idx = int(np.nanargmax(y[good]))
    return float(E[good][idx]), float(y[good][idx])


def _integrated_flux(E: np.ndarray, phi: np.ndarray, emin: float, emax: float) -> float:
    E = np.asarray(E, float)
    phi = np.asarray(phi, float)
    mask = np.isfinite(E) & np.isfinite(phi) & (E >= emin) & (E <= emax)
    if np.count_nonzero(mask) < 1:
        return np.nan
    order = np.argsort(E[mask])
    x = E[mask][order]
    y = phi[mask][order]
    if x.size == 1:
        return float(y[0])
    return float(np.trapz(y, x=np.log(x)))


def _load_loglikes(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(results_dir.glob("global_fit_k*.npz"))
    if not files:
        return np.array([]), np.array([])
    E = []
    lnL = []
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as npz:
                E.append(float(npz["Ectr_mev"]) / 1000.0)
                lnL.append(float(npz["loglike_ml"]))
        except Exception as exc:
            print(f"[warn] Could not load loglike from {path}: {exc}")
    return np.asarray(E, float), np.asarray(lnL, float)


def _plot_peak_scatter(data: dict[str, dict], outdir: Path, formats: list[str], baseline: str) -> list[str]:
    import matplotlib.pyplot as plt

    groups = sorted({d["group"] for d in data.values()})
    cmap = {g: f"C{i}" for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    rows = []
    for name, d in data.items():
        Epk, ypk = _peak(d["E"], d["phi"])
        rows.append((name, d["group"], Epk, ypk))
        if name == baseline:
            ax.scatter(Epk, ypk, marker="*", s=220, color="black", label="baseline", zorder=5)
        else:
            ax.scatter(Epk, ypk, s=55, color=cmap[d["group"]], label=d["group"])
            ax.annotate(name.replace("_", " "), (Epk, ypk), fontsize=6, xytext=(3, 3), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"peak energy [GeV]")
    ax.set_ylabel(r"peak $E^2 dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title("Systematics: halo-spectrum peak stability")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(outdir / f"systematics_peak_energy_vs_amplitude.{fmt}", dpi=230)
    plt.close(fig)
    return [f"{n:28s} {g:18s} E_peak={e:.6g} phi_peak={p:.6e}" for n, g, e, p in rows]


def _plot_flux_ratios(
    data: dict[str, dict],
    outdir: Path,
    formats: list[str],
    baseline: str,
    emin: float,
    emax: float,
) -> list[str]:
    import matplotlib.pyplot as plt

    if baseline not in data:
        return []
    base_flux = _integrated_flux(data[baseline]["E"], data[baseline]["phi"], emin, emax)
    rows = []
    for name, d in data.items():
        flux = _integrated_flux(d["E"], d["phi"], emin, emax)
        ratio = flux / base_flux if np.isfinite(base_flux) and base_flux != 0 else np.nan
        rows.append((name, d["group"], flux, ratio))
    rows_sorted = sorted(rows, key=lambda r: (r[1], r[0]))
    names = [r[0].replace("_", "\n") for r in rows_sorted]
    ratios = np.asarray([r[3] for r in rows_sorted], float)
    colors = [f"C{sorted({d['group'] for d in data.values()}).index(r[1])}" for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(rows_sorted)), 5.0))
    ax.axhline(1.0, color="black", lw=1.0)
    ax.bar(np.arange(len(rows_sorted)), ratios, color=colors, alpha=0.8)
    ax.set_xticks(np.arange(len(rows_sorted)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("integrated flux ratio to baseline")
    ax.set_title(rf"Systematics: integrated halo flux ratio ({emin:g}--{emax:g} GeV)")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(outdir / f"systematics_integrated_flux_ratio.{fmt}", dpi=230)
    plt.close(fig)
    return [f"{n:28s} {g:18s} flux={f:.6e} ratio={r:.6g}" for n, g, f, r in rows_sorted]


def _plot_delta_loglike(
    data: dict[str, dict],
    global_root: Path,
    nohalo_dir: Path,
    outdir: Path,
    formats: list[str],
    baseline: str,
) -> list[str]:
    import matplotlib.pyplot as plt

    E0, lnL0 = _load_loglikes(nohalo_dir)
    if E0.size == 0:
        print(f"[skip] No no-halo global_fit_k*.npz found in {nohalo_dir}; skipping Delta lnL plot")
        return []
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    rows = []
    for name, d in data.items():
        res_dir = global_root / name
        E, lnL = _load_loglikes(res_dir)
        if E.size == 0:
            continue
        baseline_interp = np.interp(E, E0, lnL0, left=np.nan, right=np.nan)
        dlnL = lnL - baseline_interp
        rows.append(f"{name:28s} max_dlnL={np.nanmax(dlnL):.6g} sum_dlnL={np.nansum(dlnL):.6g}")
        lw = 2.2 if name == baseline else 1.1
        color = "black" if name == baseline else None
        ax.plot(E, dlnL, marker="o", ms=3.0, lw=lw, color=color, label=name.replace("_", " "))
    if not rows:
        plt.close(fig)
        return []
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("E [GeV]")
    ax.set_ylabel(r"$\Delta\ln\mathcal{L}$ vs no-halo")
    ax.set_title("Systematics: halo likelihood-improvement stability")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(outdir / f"systematics_delta_loglike_stability.{fmt}", dpi=230)
    plt.close(fig)
    return rows


def _plot_residual_correlation(residual_table: Path, outdir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    if not residual_table.exists():
        print(f"[skip] Missing residual diagnostics table: {residual_table}")
        return []
    arr = np.genfromtxt(residual_table, names=True, comments="#", dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([arr])
    cols = ["corr_nfw", "corr_gas", "corr_ics", "corr_ps", "corr_fb_pos", "corr_fb_neg"]
    missing = [c for c in cols if c not in (arr.dtype.names or ())]
    if missing:
        print(f"[skip] Residual table lacks correlation columns: {missing}")
        return []
    mat = np.vstack([np.asarray(arr[c], float) for c in cols]).T
    E = np.asarray(arr["E_GeV"], float)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("corr_", "") for c in cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(E)))
    ax.set_yticklabels([f"{e:.2g}" for e in E])
    ax.set_ylabel("E [GeV]")
    ax.set_title("Residual-template correlation by energy")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Pearson correlation")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(outdir / f"residual_template_correlation_heatmap.{fmt}", dpi=230)
    plt.close(fig)
    return [
        f"{float(arr['E_GeV'][i]):.6g} "
        + " ".join(f"{cols[j]}={mat[i, j]:.6g}" for j in range(len(cols)))
        for i in range(mat.shape[0])
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=PROJECT / "mcmc/configs/systematics/manifest.json")
    ap.add_argument("--plot-root", type=Path, default=HERE / "plots_global_systematics")
    ap.add_argument("--global-root", type=Path, default=PROJECT / "mcmc/global_systematics")
    ap.add_argument("--nohalo-dir", type=Path, default=PROJECT / "mcmc/global_fit_results_fig4")
    ap.add_argument("--residual-table", type=Path, default=PROJECT / "results/mcmc-residual-diagnostics/global_fit_results_fig6/residual_metrics_by_energy.txt")
    ap.add_argument("--outdir", type=Path, default=HERE / "plots_global_systematics_dashboard")
    ap.add_argument("--baseline", default="baseline_rho2")
    ap.add_argument("--emin", type=float, default=5.0)
    ap.add_argument("--emax", type=float, default=100.0)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data = _load_systematics(args.manifest, args.plot_root, args.global_root)
    if args.baseline not in data:
        raise SystemExit(f"Baseline {args.baseline!r} not loaded; run global systematics first.")

    sections = ["Global Systematics Dashboard Summary", ""]
    sections += ["[peak_energy_vs_amplitude]"]
    sections += _plot_peak_scatter(data, args.outdir, args.formats, args.baseline)
    sections += ["", "[integrated_flux_ratio]"]
    sections += _plot_flux_ratios(data, args.outdir, args.formats, args.baseline, args.emin, args.emax)
    sections += ["", "[delta_loglike_stability]"]
    sections += _plot_delta_loglike(data, args.global_root, args.nohalo_dir, args.outdir, args.formats, args.baseline)
    sections += ["", "[residual_template_correlation]"]
    sections += _plot_residual_correlation(args.residual_table, args.outdir, args.formats)
    (args.outdir / "systematics_dashboard_summary.txt").write_text("\n".join(sections) + "\n")
    print(f"Wrote dashboard plots to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
