#!/usr/bin/env python3
r"""
Read the diffuse-mismodel null and the N/S hemisphere variant fits and report,
at a glance, the two things that matter:

  1. mock_diffuse_ics_null : is the recovered halo consistent with ZERO?
     (a significant 20 GeV halo here would mean the feature can be manufactured
     by ICS morphology mismodelling -> a failed null test).
  2. roi_north vs roi_south : is the halo spectrum consistent between hemispheres?
     (a large peak difference would point to residual Loop I / bubble contamination
     rather than a spherical halo).

Reads figures/plots_global_systematics/<variant>/mcmc_global_f_ml.txt (the same
tables propagate_systematics / Fig 5 use). Prints a summary and writes a 2-panel
diagnostic plot.

Usage (from the repo root):
    python check_null_and_hemisphere.py
    python check_null_and_hemisphere.py --copy-to <overleaf>/paper_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06
sys.path.insert(0, str(_REPO))

def _find_syst_dir():
    """Locate the global-systematics variant tables.

    The reorg moved these under Totani_reanalysis/figures/systematics/ but this
    script kept pointing at the old Totani_paper_check location, so every
    variant silently read back as "[not run yet]" -- the loader returns None on
    a missing file, so a wrong root looks exactly like un-run fits. Try the
    current location first, then the legacy one.
    """
    for cand in (
        _REPO / "Totani_reanalysis" / "figures" / "systematics" / "plots_global_systematics",
        _REPO / "Totani_paper_check" / "figures" / "plots_global_systematics",
    ):
        if cand.is_dir():
            return cand
    return _REPO / "Totani_reanalysis" / "figures" / "systematics" / "plots_global_systematics"


SYST = _find_syst_dir()
PEAK_LO, PEAK_HI = 10.0, 40.0   # GeV band that defines the "20 GeV halo"


def load(variant):
    t = SYST / variant / "mcmc_global_f_ml.txt"
    if not t.exists():
        return None
    a = np.genfromtxt(t, names=True, comments="#")
    E = np.asarray(a["E_GeV"], float)
    y = np.asarray(a["nfw"], float)
    lo = np.asarray(a["nfw_err_lo"], float) if "nfw_err_lo" in a.dtype.names else np.zeros_like(y)
    hi = np.asarray(a["nfw_err_hi"], float) if "nfw_err_hi" in a.dtype.names else np.zeros_like(y)
    sig = 0.5 * (lo + hi)
    return dict(E=E, y=y, lo=lo, hi=hi, sig=sig)


def _peak(d):
    m = (d["E"] >= PEAK_LO) & (d["E"] <= PEAK_HI)
    k = np.nanargmax(np.where(m, d["y"], -np.inf))
    return d["E"][k], d["y"][k], d["sig"][k], k


def report_null(d):
    print("\n=== mock_diffuse_ics_null : spurious-halo test ===")
    if d is None:
        print("  [not run yet]  -> figures/plots_global_systematics/mock_diffuse_ics_null/")
        return
    sig = np.where(d["sig"] > 0, d["sig"], np.nan)
    z = d["y"] / sig
    m = (d["E"] >= PEAK_LO) & (d["E"] <= PEAK_HI)
    zmax = np.nanmax(np.abs(z[m])) if np.any(m) else np.nan
    Ep, yp, sp, _ = _peak(d)
    print(f"  peak-band ({PEAK_LO:.0f}-{PEAK_HI:.0f} GeV) max |halo/sigma| = {zmax:.2f}")
    print(f"  halo at peak bin {Ep:.1f} GeV: {yp:.3e} +/- {sp:.1e}  ({yp/sp:+.2f} sigma)")
    verdict = "PASS (consistent with zero)" if zmax < 2.0 else \
              "MARGINAL" if zmax < 3.0 else "FAIL (significant spurious halo)"
    print(f"  -> {verdict}")


def report_hemisphere(n, s, base):
    print("\n=== roi_north vs roi_south : hemisphere consistency ===")
    if n is None or s is None:
        print(f"  [not run yet]  north={'ok' if n else 'missing'}  south={'ok' if s else 'missing'}")
        return
    En, yn, sn, _ = _peak(n)
    Es, ys, ss, _ = _peak(s)
    print(f"  north peak: {yn:.3e} +/- {sn:.1e} at {En:.1f} GeV")
    print(f"  south peak: {ys:.3e} +/- {ss:.1e} at {Es:.1f} GeV")
    diff = (yn - ys) / np.hypot(sn, ss) if (sn > 0 and ss > 0) else np.nan
    ratio = yn / ys if ys != 0 else np.nan
    print(f"  N/S peak ratio = {ratio:.2f}   difference = {diff:+.2f} sigma")
    # per-bin worst discrepancy on the common grid
    if np.allclose(n["E"], s["E"]):
        sden = np.hypot(n["sig"], s["sig"])
        zb = np.where(sden > 0, (n["y"] - s["y"]) / sden, np.nan)
        mb = (n["E"] >= PEAK_LO) & (n["E"] <= PEAK_HI)
        print(f"  worst per-bin N-S discrepancy in peak band = {np.nanmax(np.abs(zb[mb])):.2f} sigma")
    verdict = "consistent" if (np.isfinite(diff) and abs(diff) < 2.0) else "DISCREPANT (check Loop I/bubbles)"
    print(f"  -> {verdict}")


def make_plot(n, s, base, null, out_dir):
    try:
        from helpers.plot_style import apply_plot_style, get_cmap_colors
        apply_plot_style("paper", base_fontsize=10, linewidth=1.4, n_colors=8, cmap_name="plasma")
        cols = get_cmap_colors(cmap_name="plasma", n=6, start=0.15, end=0.82)
    except Exception:
        cols = ["C0", "C1", "C2", "C3", "C4", "C5"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axN, axM) = plt.subplots(1, 2, figsize=(7.1, 3.2))
    # (a) hemisphere
    if base is not None:
        axN.plot(base["E"], base["y"], "-", color="k", lw=1.5, label="baseline")
    if n is not None:
        axN.errorbar(n["E"] * 0.97, n["y"], yerr=[n["lo"], n["hi"]], fmt="o", ms=3.8,
                     color=cols[1], capsize=2, lw=1.3, label="north (b>0)")
    if s is not None:
        axN.errorbar(s["E"] * 1.03, s["y"], yerr=[s["lo"], s["hi"]], fmt="s", ms=3.8,
                     color=cols[4], capsize=2, lw=1.3, label="south (b<0)")
    axN.axhline(0.0, color="0.5", lw=0.7, ls=":"); axN.axvline(20.0, color="0.8", lw=0.8, ls=":")
    axN.set_xscale("log"); axN.set_xlim(1, 1000)
    axN.set_xlabel(r"$E$ [GeV]"); axN.set_ylabel("halo amplitude (nfw)")
    axN.set_title("(a) N/S hemisphere split", fontsize=9)
    axN.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axN.legend(fontsize=8, frameon=False); axN.grid(alpha=0.13, which="both")
    # (b) null
    if null is not None:
        axM.errorbar(null["E"], null["y"], yerr=[null["lo"], null["hi"]], fmt="o", ms=4,
                     color=cols[3], capsize=2, lw=1.4, label="recovered halo")
    axM.axhline(0.0, color="k", lw=1.0)
    axM.axvspan(PEAK_LO, PEAK_HI, color="0.85", alpha=0.5, zorder=0, label="20 GeV band")
    axM.set_xscale("log"); axM.set_xlim(1, 1000)
    axM.set_xlabel(r"$E$ [GeV]"); axM.set_ylabel("halo amplitude (nfw)")
    axM.set_title("(b) diffuse-mismodel null", fontsize=9)
    axM.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axM.legend(fontsize=8, frameon=False); axM.grid(alpha=0.13, which="both")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"f_null_hemisphere_checks.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight"); paths.append(p)
    plt.close(fig)
    print(f"\nsaved {paths[0]}")
    return paths[0]


def main(argv=None):
    import shutil
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "Totani_paper_check" / "figures" / "paper_figures")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    base = load("baseline_rho2")
    north = load("roi_north")
    south = load("roi_south")
    null = load("mock_diffuse_ics_null")

    report_null(null)
    report_hemisphere(north, south, base)
    if all(v is None for v in (north, south, null)):
        print("\nNothing to plot yet — run the variant fits first.")
        return
    p = make_plot(north, south, base, null, args.out_dir)
    if args.copy_to and p is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, args.copy_to / p.name)
        print("copied ->", args.copy_to / p.name)


if __name__ == "__main__":
    main()
