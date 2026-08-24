#!/usr/bin/env python3
r"""
Analyse the dual-NFW (rho^2.5 + rho^2) disk-included global fast-fit diagnostic.

Question: when an NFW-rho^2.5 and an NFW-rho^2 halo template are fitted
*independently and simultaneously* (disk included), does the second halo do
anything -- i.e. does the excess split into two distinct centrally-concentrated
components, or does one collapse to ~0 because the data only want one?

It compares the summed maximum-likelihood log-likelihood of the dual fit against
the single-halo and no-halo disk-included fits, and prints the two halo
amplitudes per energy bin so you can see whether both are non-zero (and with
which sign).

Default inputs (override with flags):
  no-halo   : mcmc/fit_results/quick_global/global_fit_results_fig4_w_disk
  rho2.5    : mcmc/fit_results/quick_global/global_fit_results_fig5_w_disk
  rho2      : mcmc/fit_results/quick_global/global_fit_results_fig6_w_disk
  dual      : mcmc/global_fit_results_nfw25_nfw2_pos_w_disk   (the _pos run)

Usage (from Totani_reanalysis/mcmc/):
  python analyse_dual_nfw_fit.py
  python analyse_dual_nfw_fit.py --dual ../global_fit_results_nfw25_nfw2_w_disk
  python analyse_dual_nfw_fit.py --band 10 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent                 # Totani_reanalysis/mcmc
_QG = _HERE / "fit_results" / "quick_global"
NBINS = 13

RHO25 = "rho2.5"
# match the bare rho^2 label without also matching rho2.5
RHO2_SUFFIX = "rho2_rs21_R08_rvir402_ns2048_normpole_pheno"


def _bin_files(d: Path):
    return [d / f"global_fit_k{k:02d}.npz" for k in range(NBINS)
            if (d / f"global_fit_k{k:02d}.npz").exists()]


def load_lnL(d: Path) -> dict:
    """Return {bin_index: (loglike_ml, Ectr_GeV)} for a fit directory."""
    out = {}
    for k in range(NBINS):
        p = d / f"global_fit_k{k:02d}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            out[k] = (float(z["loglike_ml"]), float(z["Ectr_mev"]) / 1000.0)
    return out


def _theta_vector(z) -> np.ndarray:
    for key in ("theta_ml", "theta", "theta_p50", "f_ml", "f_p50"):
        if key in z.files:
            return np.asarray(z[key], float)
    raise KeyError(f"no amplitude vector found; keys = {list(z.files)}")


def load_dual_amplitudes(d: Path):
    """Return (E, rho25_amp, rho2_amp, lnL) arrays over bins for the dual fit."""
    E, a25, a2, lnL = [], [], [], []
    for k in range(NBINS):
        p = d / f"global_fit_k{k:02d}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            labels = [str(x) for x in np.atleast_1d(z["labels"]).tolist()]
            theta = _theta_vector(z)
            i25 = next((i for i, l in enumerate(labels) if RHO25 in l), None)
            i2 = next((i for i, l in enumerate(labels) if l.endswith(RHO2_SUFFIX)), None)
            if i25 is None or i2 is None:
                raise SystemExit(
                    f"could not find both NFW labels in {p.name}; labels = {labels}")
            E.append(float(z["Ectr_mev"]) / 1000.0)
            a25.append(float(theta[i25]))
            a2.append(float(theta[i2]))
            lnL.append(float(z["loglike_ml"]))
    return (np.array(E), np.array(a25), np.array(a2), np.array(lnL))


def _sum_common(La: dict, Lb: dict, band=None) -> float:
    """2 * sum_k [La - Lb] over common, finite, converged bins (optionally banded)."""
    tot = 0.0
    for k in sorted(set(La) & set(Lb)):
        da = La[k][0] - Lb[k][0]
        if not np.isfinite(da) or abs(da) > 1e6:
            continue
        if band is not None and not (band[0] <= La[k][1] <= band[1]):
            continue
        tot += da
    return 2.0 * tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nohalo", type=Path, default=_QG / "global_fit_results_fig4_w_disk")
    ap.add_argument("--rho25", type=Path, default=_QG / "global_fit_results_fig5_w_disk")
    ap.add_argument("--rho2", type=Path, default=_QG / "global_fit_results_fig6_w_disk")
    ap.add_argument("--dual", type=Path,
                    default=_HERE / "global_fit_results_nfw25_nfw2_pos_w_disk")
    ap.add_argument("--band", type=float, nargs=2, default=(10.0, 40.0),
                    metavar=("ELO", "EHI"))
    args = ap.parse_args(argv)

    fits = {"no-halo": args.nohalo, "rho2.5": args.rho25,
            "rho2": args.rho2, "dual": args.dual}
    L = {}
    for name, d in fits.items():
        if not _bin_files(d):
            print(f"[warn] no global_fit_k*.npz in {d}  ({name}) -- skipping")
            continue
        L[name] = load_lnL(d)

    if "dual" not in L:
        raise SystemExit(f"dual fit not found at {args.dual}; run the config first.")

    band = tuple(args.band)
    print(f"Dual NFW (rho^2.5 + rho^2) disk-included diagnostic")
    print(f"  dual dir : {args.dual}")
    print(f"  band     : {band[0]:g}-{band[1]:g} GeV\n")

    # --- likelihood comparison (2 dlnL of dual relative to each reference) ---
    print(f"{'reference':12s} {'2dlnL(all)':>12s} {'2dlnL(band)':>12s}   "
          f"interpretation of dual gain")
    print("-" * 78)
    order = [n for n in ("no-halo", "rho2.5", "rho2") if n in L]
    for ref in order:
        d_all = _sum_common(L["dual"], L[ref], band=None)
        d_band = _sum_common(L["dual"], L[ref], band=band)
        note = {
            "no-halo": "dual vs no halo at all",
            "rho2.5": "gain of adding rho^2 on top of rho^2.5",
            "rho2": "gain of adding rho^2.5 on top of rho^2",
        }[ref]
        print(f"{ref:12s} {d_all:12.1f} {d_band:12.1f}   {note}")

    if {"rho2.5", "rho2"} <= set(L):
        best_single = max(("rho2.5", "rho2"), key=lambda n: sum(v[0] for v in L[n].values()))
        d_all = _sum_common(L["dual"], L[best_single], band=None)
        d_band = _sum_common(L["dual"], L[best_single], band=band)
        print("-" * 78)
        print(f"best single halo = {best_single};  dual improves on it by "
              f"2dlnL = {d_all:.1f} (all), {d_band:.1f} (band)")
        print("  small (<~ a few)  => second halo adds nothing; one morphology suffices.")
        print("  large + both amps > 0 in band => excess splits into two components.")

    # --- per-bin amplitudes of the two halos in the dual fit ---
    E, a25, a2, _ = load_dual_amplitudes(args.dual)
    print(f"\nPer-bin halo amplitudes in the dual fit (theta_ml):")
    print(f"{'bin':>3s} {'E[GeV]':>9s} {'rho2.5':>13s} {'rho2':>13s}   sign")
    for k, (e, x25, x2) in enumerate(zip(E, a25, a2)):
        inb = "  *band" if band[0] <= e <= band[1] else ""
        sign = ("++" if x25 > 0 and x2 > 0 else
                "+-" if x25 > 0 > x2 else
                "-+" if x25 < 0 < x2 else "--")
        print(f"{k:3d} {e:9.2f} {x25:13.4e} {x2:13.4e}   {sign}{inb}")

    # band-summed amplitude signs (quick read of "do both contribute")
    inband = (E >= band[0]) & (E <= band[1])
    if np.any(inband):
        s25 = float(np.nansum(a25[inband]))
        s2 = float(np.nansum(a2[inband]))
        print(f"\nBand-summed amplitudes: rho2.5 = {s25:.4e}, rho2 = {s2:.4e}")
        if s25 > 0 and s2 > 0:
            print("  -> both positive in the signal band: the excess supports BOTH morphologies.")
        elif s25 <= 0 or s2 <= 0:
            print("  -> one halo is <=0 in the band: the data do not want two components there.")


if __name__ == "__main__":
    main()
