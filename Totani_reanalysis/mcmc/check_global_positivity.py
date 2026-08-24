#!/usr/bin/env python3
"""
check_global_positivity.py
==========================
Global-fit-only validation: with a free-sign halo (nfw/fb_neg), the total model
Cexp can in principle go negative in some pixels. The fitter floors Cexp before the
log, so if that floor ever *binds* at the best fit, the Poisson likelihood and the
Fisher curvature are wrong in those pixels. This script verifies it never binds.

It loads the per-bin `global_fit_k??.npz` files (which store `model_map` and
`mask_2d`) and reports any energy bin where the best-fit model is <= 0 inside the
fit mask.

Usage
-----
    python check_global_positivity.py --results-dir mcmc/global_fit_results_fig6
    python check_global_positivity.py --results-dir <dir> --floor 1e-30
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def check_results_dir(results_dir: str, *, floor: float = 1e-30) -> int:
    files = sorted(glob.glob(os.path.join(results_dir, "global_fit_k*.npz")))
    if not files:
        print(f"No global_fit_k*.npz in {results_dir}")
        return 1

    n_bad_bins = 0
    n_checked = 0
    print(f"{'bin':>4}  {'E[GeV]':>9}  {'min model':>12}  {'<=0 px':>8}  {'near floor':>11}  status")
    print("-" * 64)
    for path in files:
        d = np.load(path, allow_pickle=True)
        k = int(d["k"]) if "k" in d else -1
        E = float(d["Ectr_mev"]) / 1e3 if "Ectr_mev" in d else float("nan")

        if "positivity_min_model" in d:
            vmin = float(d["positivity_min_model"])
            n_le0 = int(d["positivity_n_le0"]) if "positivity_n_le0" in d else -1
            n_floor = int(d["positivity_n_floor"]) if "positivity_n_floor" in d else -1
        elif "model_map" in d and "mask_2d" in d:
            model = np.asarray(d["model_map"], float)
            mask = np.asarray(d["mask_2d"], bool)
            vals = model[mask & np.isfinite(model)]
            if vals.size == 0:
                continue
            vmin = float(vals.min())
            n_le0 = int(np.sum(vals <= 0.0))
            n_floor = int(np.sum(vals <= 10.0 * floor))
        else:
            print(f"{k:>4}  {E:>9.3f}  {'nan':>12}  {'?':>8}  {'?':>11}  no model/diagnostics")
            continue

        n_checked += 1
        ok = (n_le0 == 0) and (n_floor == 0)
        if not ok:
            n_bad_bins += 1
        print(f"{k:>4}  {E:>9.3f}  {vmin:>12.3e}  {n_le0:>8d}  {n_floor:>11d}  "
              f"{'OK' if ok else 'FLOOR BINDS — bin unreliable'}")

    print("-" * 64)
    if n_checked == 0:
        print("No checkable bins found: save model maps or rerun with the new compact positivity diagnostics.")
        return 1
    if n_bad_bins == 0:
        print("All bins positive within the fit mask: the floor never binds. Inference is valid.")
        return 0
    print(f"WARNING: {n_bad_bins} bin(s) hit the floor. Re-fit those with negative_mode='strict' "
          f"or inspect whether the free-sign halo is being driven unphysically negative.")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Check global-fit model positivity per energy bin.")
    ap.add_argument("--results-dir", required=True, help="Dir with global_fit_k??.npz files.")
    ap.add_argument("--floor", type=float, default=1e-30, help="poisson_cexp_floor used in the fit.")
    args = ap.parse_args()
    return check_results_dir(args.results_dir, floor=float(args.floor))


if __name__ == "__main__":
    raise SystemExit(main())
