#!/usr/bin/env python3
"""
plot_scattering_summary.py
==========================
For one annihilation channel and mass, compare available unscattered and
scattered PPPC MCMC runs.

Outputs:
  1. A single-bin comparison plot like the run_scattering_constraints summary.
  2. A halo-spectrum overlay against the Totani rho^2 halo spectrum.
  3. A text summary saying whether each scattered run improves or worsens the
     dSph tension relative to the unscattered run.

Example
-------
    python figures/plot_scattering_summary.py --channel WW --mass 700 --energy-bin 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))
sys.path.insert(0, str(_REPO / "Totani_reanalysis" / "core"))

SIGMA_V_REF = 1e-25

from dsph_limits import DSph_LIMIT_SOURCES, dsph_upper_limit  # noqa: E402

# Set from --dsph-source in main(); read by dsph_limit().
_DSPH_SOURCE = "hoof"


@dataclass
class RunSummary:
    label: str
    short_label: str
    path: Path
    pppc_label: str
    is_unscattered: bool
    E_GeV: np.ndarray
    f16: np.ndarray
    f50: np.ndarray
    f84: np.ndarray
    phi16: np.ndarray
    phi50: np.ndarray
    phi84: np.ndarray
    sv: float
    sv16: float
    sv84: float
    tension: float
    tension16: float
    tension84: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scattering comparison and halo spectrum for one PPPC mass/channel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--channel", default="WW", choices=["WW", "bb", "tautau"])
    parser.add_argument("--mass", type=float, required=True, help="DM annihilation mass [GeV].")
    parser.add_argument("--energy-bin", type=int, default=5, help="Energy bin used for the tension comparison.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--include", nargs="*", default=None, help="Optional substrings to filter scattered runs.")
    parser.add_argument("--dsph-source", default="hoof", choices=list(DSph_LIMIT_SOURCES),
                        help="dSph limit set: s-wave (hoof/mcdaniel/boddy_swave) or "
                             "velocity-dependent (boddy_pwave/boddy_dwave/boddy_somm).")
    return parser.parse_args()


def dsph_limit(mass: float, channel: str, source: str | None = None) -> float:
    return dsph_upper_limit(float(mass), channel, source=source or _DSPH_SOURCE)


def discover_run_dirs(channel: str, mass: float, include: list[str] | None = None) -> list[Path]:
    tag = f"{channel}_m{int(round(mass)):04d}"
    candidates: list[Path] = []
    for root in (_PROJECT, _PROJECT / "mcmc"):
        candidates.extend(sorted(root.glob(f"mcmc_results_pppc_pppc_{tag}*")))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if include and ("_scat_" in path.name):
            lower = path.name.lower()
            if not any(piece.lower() in lower for piece in include):
                continue
        if list(path.glob("mcmc_results_k*.npz")):
            out.append(path)

    out.sort(key=lambda p: (("_scat_" in p.name), p.name))
    return out


def pppc_label_from_npz(path: Path) -> str | None:
    with np.load(path, allow_pickle=True) as npz:
        labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
    cands = [label for label in labels if "pppc" in label.lower()]
    return cands[0] if len(cands) == 1 else None


def load_coefficients(run_dir: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(run_dir.glob("mcmc_results_k*.npz"))
    if not files:
        raise FileNotFoundError(f"No mcmc_results_k*.npz in {run_dir}")

    max_k = max(int(path.stem.split("_k")[-1]) for path in files)
    nE = max_k + 1
    E = np.full(nE, np.nan)
    f16 = np.full(nE, np.nan)
    f50 = np.full(nE, np.nan)
    f84 = np.full(nE, np.nan)

    pppc_label = pppc_label_from_npz(files[0])
    if pppc_label is None:
        raise KeyError(f"Could not find unique PPPC label in {files[0]}")

    for path in files:
        k = int(path.stem.split("_k")[-1])
        with np.load(path, allow_pickle=True) as npz:
            labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
            if pppc_label not in labels:
                cands = [i for i, label in enumerate(labels) if "pppc" in label.lower()]
                if len(cands) != 1:
                    continue
                idx = cands[0]
            else:
                idx = labels.index(pppc_label)
            if "Ectr_mev" in npz:
                E[k] = float(npz["Ectr_mev"]) / 1000.0
            f16[k] = float(np.atleast_1d(npz["f_p16"])[idx])
            f50[k] = float(np.atleast_1d(npz["f_p50"])[idx])
            f84[k] = float(np.atleast_1d(npz["f_p84"])[idx])

    return pppc_label, E, f16, f50, f84


def template_pole_flux(
    pppc_label: str,
    *,
    channel: str,
    mass: float,
    E_GeV: np.ndarray,
) -> np.ndarray:
    """
    Return the pole-normalized reference spectrum for this template.

    For unscattered PPPC, use the analytic physical normalization. For scattered
    templates, infer only the spectral transfer ratio from the E2dnde FITS cubes:
    scattered_cube / unscattered_cube. The common spatial J/J_pole map cancels.
    """
    from plot_pppc_mass_scan import _pppc_reference_pole_flux

    source = np.full(len(E_GeV), np.nan)
    finite_E = np.isfinite(E_GeV) & (E_GeV > 0.0)
    source[finite_E] = _pppc_reference_pole_flux(E_GeV[finite_E], mass, channel)

    base_label = f"pppc_{channel}_m{int(round(mass)):04d}"
    if pppc_label == base_label:
        return source

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("Reading E2dnde FITS templates requires astropy.") from exc

    template_dir = _REPO / "fermi_data" / "totani" / "processed" / "templates"
    base_path = template_dir / f"{base_label}_E2dnde.fits"
    scat_path = template_dir / f"{pppc_label}_E2dnde.fits"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing unscattered E2dnde template: {base_path}")
    if not scat_path.exists():
        raise FileNotFoundError(f"Missing scattered E2dnde template: {scat_path}")

    base_cube = np.asarray(fits.getdata(base_path), dtype=float)
    scat_cube = np.asarray(fits.getdata(scat_path), dtype=float)
    nE = min(len(source), base_cube.shape[0], scat_cube.shape[0])
    ratio = np.full(len(source), np.nan)
    for k in range(nE):
        base_slice = base_cube[k]
        scat_slice = scat_cube[k]
        mask = (
            np.isfinite(base_slice)
            & np.isfinite(scat_slice)
            & (base_slice > 0.0)
        )
        if np.any(mask):
            base_sum = float(np.nansum(base_slice[mask]))
            scat_sum = float(np.nansum(scat_slice[mask]))
            if base_sum > 0.0 and np.isfinite(base_sum) and np.isfinite(scat_sum):
                ratio[k] = scat_sum / base_sum

    return source * ratio


def human_label(path: Path, is_unscattered: bool) -> str:
    if is_unscattered:
        return "Unscattered"
    name = path.name.lower()
    if "dipole_magnetic" in name or "dipole" in name:
        return "Dipole Dirac"
    if "anapole" in name:
        return "Anapole Maj."
    if "higgs" in name:
        return "Higgs portal"
    return path.name.split("_scat_", 1)[-1].replace("_", " ")


def load_run_summary(run_dir: Path, mass: float, channel: str, energy_bin: int) -> RunSummary:
    pppc_label, E, f16, f50, f84 = load_coefficients(run_dir)
    source = template_pole_flux(pppc_label, channel=channel, mass=mass, E_GeV=E)
    phi16 = f16 * source
    phi50 = f50 * source
    phi84 = f84 * source

    if energy_bin >= len(f50) or not np.isfinite(f50[energy_bin]):
        raise KeyError(f"{run_dir.name} has no finite k{energy_bin:02d} PPPC posterior")

    is_unscattered = "_scat_" not in run_dir.name
    sv = float(f50[energy_bin] * SIGMA_V_REF)
    sv16 = float(f16[energy_bin] * SIGMA_V_REF)
    sv84 = float(f84[energy_bin] * SIGMA_V_REF)
    limit = dsph_limit(mass, channel)

    return RunSummary(
        label=run_dir.name,
        short_label=human_label(run_dir, is_unscattered),
        path=run_dir,
        pppc_label=pppc_label,
        is_unscattered=is_unscattered,
        E_GeV=E,
        f16=f16,
        f50=f50,
        f84=f84,
        phi16=phi16,
        phi50=phi50,
        phi84=phi84,
        sv=sv,
        sv16=sv16,
        sv84=sv84,
        tension=sv / limit,
        tension16=sv16 / limit,
        tension84=sv84 / limit,
    )


def load_halo():
    from totani_data_loader import load_halo_spectrum_rho2
    return load_halo_spectrum_rho2()


def plot_comparison(runs: list[RunSummary], mass: float, channel: str, energy_bin: int, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    ref = next((run for run in runs if run.is_unscattered), None)
    x = np.arange(len(runs))
    sv = np.array([run.sv for run in runs]) / 1e-25
    sv16 = np.array([run.sv16 for run in runs]) / 1e-25
    sv84 = np.array([run.sv84 for run in runs]) / 1e-25
    yerr = np.vstack([sv - sv16, sv84 - sv])
    labels = [run.short_label.replace(" ", "\n") for run in runs]
    colors = ["0.25" if run.is_unscattered else f"C{i}" for i, run in enumerate(runs)]

    delta = np.zeros(len(runs))
    if ref is not None:
        delta = np.array([(run.sv - ref.sv) / ref.sv * 100.0 for run in runs])

    fig, (ax, ax_delta) = plt.subplots(
        2, 1, figsize=(6.8, 5.4), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    ax.errorbar(
        x, sv, yerr=yerr, fmt="o", ms=7, color="k", ecolor="k",
        capsize=4, mfc="white", mec="k", mew=1.2, zorder=3,
    )
    for xi, yi, color in zip(x, sv, colors):
        ax.scatter([xi], [yi], s=72, color=color, edgecolor="k", linewidth=0.8, zorder=4)

    if ref is not None:
        ax.axhline(ref.sv / 1e-25, color="k", ls="--", lw=1.1, label="Unscattered median")
        ax.fill_between(
            [-0.6, len(runs) - 0.4],
            ref.sv16 / 1e-25,
            ref.sv84 / 1e-25,
            color="0.2",
            alpha=0.10,
            label="Unscattered 68% interval",
        )

    e_label = runs[0].E_GeV[energy_bin] if energy_bin < len(runs[0].E_GeV) else np.nan
    ax.set_title(f"{channel} scattering comparison at m={mass:.0f} GeV (k={energy_bin}, E={e_label:.2g} GeV)")
    ax.set_ylabel(r"$\langle\sigma v\rangle$ [$10^{-25}$ cm$^3$ s$^{-1}$]")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    ax_delta.axhline(0, color="k", lw=0.9)
    ax_delta.bar(x, delta, color=colors, alpha=0.82, width=0.55)
    ax_delta.set_ylabel(r"$\Delta f$ [%]")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(labels)
    ax_delta.grid(True, axis="y", alpha=0.25)
    if np.any(np.isfinite(delta)):
        ax_delta.set_ylim(min(-0.5, float(np.nanmin(delta)) - 0.35), max(0.8, float(np.nanmax(delta)) + 0.35))
    for xi, val in zip(x, delta):
        if np.isfinite(val) and abs(val) > 0:
            ax_delta.text(xi, val + (0.1 if val >= 0 else -0.1), f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    out = outdir / f"scattering_comparison_{channel}_m{int(round(mass)):04d}_k{energy_bin:02d}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spectrum(runs: list[RunSummary], mass: float, channel: str, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    halo = load_halo()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    scale = 1e5
    ax.errorbar(
        halo.E_bins_GeV,
        halo.phi * scale,
        yerr=[halo.phi_err_lo * scale, halo.phi_err_hi * scale],
        fmt="ko",
        ms=4.8,
        capsize=3,
        label=r"Totani halo (NFW-$\rho^2$)",
        zorder=5,
    )

    for i, run in enumerate(runs):
        color = "0.25" if run.is_unscattered else f"C{i}"
        finite = np.isfinite(run.E_GeV) & np.isfinite(run.phi50)
        if not np.any(finite):
            continue
        y = run.phi50[finite] * scale
        yerr = np.vstack([
            (run.phi50[finite] - run.phi16[finite]) * scale,
            (run.phi84[finite] - run.phi50[finite]) * scale,
        ])
        ax.errorbar(
            run.E_GeV[finite],
            y,
            yerr=yerr,
            color=color,
            marker="o",
            ms=3.8,
            lw=1.2,
            capsize=2,
            label=run.short_label,
            alpha=0.9,
        )

    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Photon energy [GeV]")
    ax.set_ylabel(r"$E^2 dN/dE$ [$10^{-5}$ MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title(f"Halo spectrum overlay: {channel}, m={mass:.0f} GeV")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out = outdir / f"halo_spectrum_{channel}_m{int(round(mass)):04d}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary(runs: list[RunSummary], mass: float, channel: str, energy_bin: int, outdir: Path) -> Path:
    ref = next((run for run in runs if run.is_unscattered), None)
    limit = dsph_limit(mass, channel)
    lines = [
        f"Scattering summary: channel={channel}, mass={mass:.0f} GeV, k={energy_bin}",
        f"dSph limit: {limit:.6e} cm^3/s",
        "",
        f"{'Run':<24} {'sigmav':>13} {'tension':>10} {'vs unscat':>12}  Verdict",
        "-" * 78,
    ]

    for run in runs:
        if ref is None or run.is_unscattered:
            delta = 0.0
            verdict = "baseline"
        else:
            delta = (run.tension - ref.tension) / ref.tension * 100.0
            if run.tension < ref.tension:
                verdict = "better (lower dSph tension)"
            elif run.tension > ref.tension:
                verdict = "worse (higher dSph tension)"
            else:
                verdict = "unchanged"
            if abs(delta) < 1.0:
                verdict += "; negligible shift"
        lines.append(
            f"{run.short_label:<24} {run.sv:13.4e} {run.tension:10.4f} {delta:11.3f}%  {verdict}"
        )

    out = outdir / f"summary_{channel}_m{int(round(mass)):04d}_k{energy_bin:02d}.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main() -> None:
    args = parse_args()
    global _DSPH_SOURCE
    _DSPH_SOURCE = args.dsph_source
    outdir = Path(args.output_dir) if args.output_dir else _HERE / "plots_scattering_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    dirs = discover_run_dirs(args.channel, args.mass, include=args.include)
    if not dirs:
        raise SystemExit(f"No MCMC result dirs found for {args.channel} m={args.mass:g}")

    runs: list[RunSummary] = []
    skipped: list[str] = []
    for run_dir in dirs:
        try:
            runs.append(load_run_summary(run_dir, args.mass, args.channel, args.energy_bin))
        except Exception as exc:
            skipped.append(f"{run_dir.name}: {exc}")

    if not any(run.is_unscattered for run in runs):
        raise SystemExit("Need an unscattered baseline run with the requested energy bin.")
    if len(runs) < 2:
        raise SystemExit("Need at least one scattered run with the requested energy bin.")

    comparison = plot_comparison(runs, args.mass, args.channel, args.energy_bin, outdir)
    spectrum = plot_spectrum(runs, args.mass, args.channel, outdir)
    summary = write_summary(runs, args.mass, args.channel, args.energy_bin, outdir)

    print(f"\nSaved comparison plot: {comparison}")
    print(f"Saved spectrum plot:    {spectrum}")
    print(f"Saved summary:          {summary}")
    if skipped:
        print("\nSkipped runs:")
        for item in skipped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
