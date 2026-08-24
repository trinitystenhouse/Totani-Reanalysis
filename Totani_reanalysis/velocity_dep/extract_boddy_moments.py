#!/usr/bin/env python3
r"""
Parse Boddy et al. 2019 (arXiv:1909.13197) effective-J posteriors into the
stacked J-weighted velocity moments consumed by ``extract_boddy_limits.py
--mode boddy``.

INPUT
-----
The per-dwarf ``.npy`` files in
    J-Factor-Scaling/repo_1909_13197/Jfactors/<dwarf>.npy
(git-lfs).  Each is a structured array of weighted posterior samples with columns
    weights, dist, rs, rhos,
    j_s_0d1 j_s_0d2 j_s_0d5 j_s_10d0   (s-wave effective J at 0.1/0.2/0.5/10 deg)
    j_p_*  j_d_*  j_som_*              (p-/d-wave, Sommerfeld; same apertures)
in units of Msol^2/kpc^5 (linear, NOT log10).

MOMENTS
-------
For each velocity law n, Boddy's effective J already carries the velocity
weighting:  j_n = ∫ rho^2 <(v/c)^{2n}> .  Hence the dimensionless J-weighted
dwarf moment is the ratio
    u_dwarf_n = j_n / j_s .
We need a single moment to pair with the *stacked* s-wave dwarf limit (Hoof /
McDaniel).  The stack's effective weighting is the s-wave-J-weighted mean of the
ratio, which equals
    u_dwarf_n^stack = ( sum_dwarfs j_n ) / ( sum_dwarfs j_s ) ,
each dwarf entering at its posterior (weighted) median.  This is the principled,
transparent choice; the true Fermi stack weights also fold in per-dwarf
exposure/background, so treat it as a close approximation (the high-J dwarfs that
dominate sum(j_s) also dominate the real stack).

OUTPUT
------
A one-row CSV with columns
    u_dwarf_swave,u_dwarf_pwave,u_dwarf_dwave,u_dwarf_somm
ready for:
    python extract_boddy_limits.py --mode boddy --boddy-csv boddy_moments_0d5.csv
Also prints the per-dwarf ratios and the stack so the spread is visible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DEFAULT_JDIR = _HERE.parent / "J-Factor-Scaling" / "repo_1909_13197" / "Jfactors"
_WAVES = {"swave": "j_s", "pwave": "j_p", "dwave": "j_d", "somm": "j_som"}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return float(v[np.searchsorted(cw, 0.5)])


def dwarf_medians(path: Path, aperture: str) -> dict[str, float] | None:
    a = np.load(path, allow_pickle=True)
    names = a.dtype.names or ()
    w = np.asarray(a["weights"], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    out: dict[str, float] = {}
    for wave, stem in _WAVES.items():
        col = f"{stem}_{aperture}"
        if col not in names:
            return None
        out[wave] = weighted_median(np.asarray(a[col], dtype=float), w)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jdir", type=Path, default=_DEFAULT_JDIR,
                   help="Path to repo_1909_13197/Jfactors")
    p.add_argument("--aperture", default="0d5", choices=["0d1", "0d2", "0d5", "10d0"],
                   help="Integration aperture (deg); 0d5 = 0.5 deg is the Fermi-dSph standard.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Dwarf name substrings to skip (e.g. _k15 variants).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default boddy_moments_<aperture>.csv next to this script).")
    args = p.parse_args(argv)

    files = sorted(args.jdir.glob("*.npy"))
    if not files:
        p.error(f"No .npy files in {args.jdir} (is git-lfs pulled?).")

    rows = []
    sums = {w: 0.0 for w in _WAVES}
    print(f"{'dwarf':22s} {'j_s':>10s} {'u_pwave':>11s} {'u_dwave':>11s} {'u_somm':>11s}")
    for f in files:
        name = f.stem
        if any(s in name for s in args.exclude):
            continue
        med = dwarf_medians(f, args.aperture)
        if med is None or med["swave"] <= 0:
            print(f"{name:22s}  (skipped: no {args.aperture} columns or J_s<=0)")
            continue
        u_p = med["pwave"] / med["swave"]
        u_d = med["dwave"] / med["swave"]
        u_s = med["somm"] / med["swave"]
        rows.append((name, med["swave"], u_p, u_d, u_s))
        for w in _WAVES:
            sums[w] += med[w]
        print(f"{name:22s} {med['swave']:10.3e} {u_p:11.3e} {u_d:11.3e} {u_s:11.3e}")

    if not rows:
        p.error("No usable dwarfs parsed.")

    u_p_stack = sums["pwave"] / sums["swave"]
    u_d_stack = sums["dwave"] / sums["swave"]
    u_s_stack = sums["somm"] / sums["swave"]

    print("\n" + "=" * 70)
    print(f"STACK (sum j_n / sum j_s) over {len(rows)} dwarfs, aperture {args.aperture}:")
    print(f"  u_dwarf_pwave = {u_p_stack:.4e}   (= J-weighted <(v/c)^2>_dSph; "
          f"v_rms = {np.sqrt(u_p_stack)*299792.458:.1f} km/s)")
    print(f"  u_dwarf_dwave = {u_d_stack:.4e}   (= <(v/c)^4>_dSph)")
    print(f"  u_dwarf_somm  = {u_s_stack:.4e}   (= <(c/v)>_dSph; "
          f"v = {299792.458/u_s_stack:.1f} km/s)")

    out = args.out or (_HERE / f"boddy_moments_{args.aperture}.csv")
    with open(out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["u_dwarf_swave", "u_dwarf_pwave", "u_dwarf_dwave", "u_dwarf_somm"])
        wcsv.writerow([1.0, u_p_stack, u_d_stack, u_s_stack])
    print(f"\nWrote {out}")
    print("Next:")
    print(f"  python extract_boddy_limits.py --mode boddy --boddy-csv {out.name} "
          f"--v-halo-kms <ROI_value>")


if __name__ == "__main__":
    main()
