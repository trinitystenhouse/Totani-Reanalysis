#!/usr/bin/env python3
r"""
Morphology likelihood / TS comparison for the 20 GeV excess.

Question: is the high-latitude excess a *structured, centrally-concentrated*
component (DM-like, NFW rho^p), or could the free isotropic component absorb it
(i.e. it is just excess isotropic / IGRB-like emission)?

Method: read the deterministic global fits (per energy bin) and compare
  - no-halo baseline (fig4): gas, ICS, iso, ps, Loop I, bubbles -- iso is FREE,
    so this is the best the conventional + isotropic model can do;
  - halo fits (fig5/6/7): the same model + an NFW rho^2.5 / rho^2 / rho^1 halo.
For each profile we report
  TS = 2 [ lnL(halo) - lnL(no-halo) ]   summed over the 13 energy bins,
which adds one halo amplitude per bin (13 d.o.f.). A large TS over the iso-free
baseline is the core evidence that the excess is NOT isotropic -- the free iso
component already had the chance to absorb it and could not. The profile with the
highest lnL indicates the preferred concentration.

This is necessary but NOT sufficient for a DM interpretation: NFW rho^p are all
centrally concentrated, so this shows "structured, and which slope", not
"NFW vs a non-DM structured template". Pair it with the model-independent angular
profile (plot_angular_profile_vs_models.py), which compares the excess shape
directly against NFW^p AND isotropic.

Caveat: the templates are correlated and the diffuse model is not exhaustive, so
the chi^2 p-value is a guide, not a calibrated discovery significance (same Wilks
caveat as the per-bin Delta lnL in the paper).

Usage (from Totani_reanalysis/morphology/):
  python morphology_ts_table.py
  python morphology_ts_table.py --fit-root ../mcmc/fit_results/quick_global
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent

PROFILES = {"fig5": r"NFW rho^2.5", "fig6": r"NFW rho^2", "fig7": r"NFW rho^1"}
PEAK_LO, PEAK_HI = 10.0, 40.0       # GeV band that defines the 20 GeV excess
NBINS = 13


def load_lnL(fit_dir: Path):
    """Per-bin (loglike_ml, Ectr_GeV) from global_fit_k{kk}.npz."""
    lnL = np.full(NBINS, np.nan)
    E = np.full(NBINS, np.nan)
    for k in range(NBINS):
        p = fit_dir / f"global_fit_k{k:02d}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        lnL[k] = float(z["loglike_ml"])
        E[k] = float(z["Ectr_mev"]) / 1000.0
    return lnL, E


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-root", type=Path, default=_REANALYSIS / "mcmc" / "fit_results" / "quick_global")
    ap.add_argument("--nohalo", default="global_fit_results_fig4")
    ap.add_argument("--max-abs-dlnl", type=float, default=1e6,
                    help="drop bins with |dlnL| above this (non-converged fits).")
    args = ap.parse_args(argv)

    lnL0, E = load_lnL(args.fit_root / args.nohalo)
    if not np.any(np.isfinite(lnL0)):
        raise SystemExit(f"no no-halo fit found under {args.fit_root/args.nohalo}")
    band = (E >= PEAK_LO) & (E <= PEAK_HI)

    print(f"no-halo baseline: {args.nohalo}  (iso free; {int(np.isfinite(lnL0).sum())} bins)\n")

    MAXABS = args.max_abs_dlnl
    # load all profiles, then restrict to a COMMON set of good bins so the
    # profile-to-profile lnL ranking is fair (same bins for every profile).
    lnLh = {tag: load_lnL(args.fit_root / f"global_fit_results_{tag}")[0] for tag in PROFILES}
    good = np.isfinite(lnL0)
    for tag in PROFILES:
        d = lnLh[tag] - lnL0
        good &= np.isfinite(lnLh[tag]) & (np.abs(np.nan_to_num(d, nan=np.inf)) <= MAXABS)
    dropped = [k for k in range(NBINS) if np.isfinite(lnL0[k]) and not good[k]]
    if dropped:
        print(f"[warn] dropping bins {dropped} (a fit is non-converged there; rerun those); "
              f"comparison uses the {int(good.sum())} common good bins.\n")
    dof = int(good.sum())

    rows = []
    for tag, label in PROFILES.items():
        d = lnLh[tag] - lnL0
        ts = float(2.0 * np.nansum(d[good]))
        ts_band = float(2.0 * np.nansum(d[good & band]))
        rows.append((tag, label, ts, ts_band, float(np.nansum(lnLh[tag][good]))))
    best = max(rows, key=lambda r: r[4])
    top_lnL = best[4]

    hdr2 = f"{'profile':12s} {'TS':>9s} {'TS(10-40GeV)':>13s} {'dlnL vs best':>13s}"
    print(hdr2); print("-" * len(hdr2))
    for tag, label, ts, ts_band, lnL in rows:
        print(f"{label:12s} {ts:9.1f} {ts_band:13.1f} {lnL-top_lnL:+13.1f}"
              f"{'   <--' if tag==best[0] else ''}")

    print(f"\n(common good bins: {dof}; one halo amplitude per bin)")
    print("\nInterpretation:")
    print("  * TS > 0 over the iso-free no-halo baseline => the excess is spatially")
    print("    STRUCTURED: the free isotropic component cannot absorb it.")
    print(f"  * Preferred concentration (highest lnL): {best[1]}.")
    print("  * CAVEAT: the absolute TS is NOT a calibrated significance -- the same")
    print("    unmodelled, spatially-coherent diffuse residuals that give chi2/ndof~8")
    print("    inflate it. Trust the RELATIVE ordering (which profile, vs isotropic),")
    print("    not the absolute TS/sigma.")
    print("  * NFW^p are all centrally concentrated; this does not distinguish NFW")
    print("    from another structured template -- see plot_angular_profile_vs_models.py.")


if __name__ == "__main__":
    main()
