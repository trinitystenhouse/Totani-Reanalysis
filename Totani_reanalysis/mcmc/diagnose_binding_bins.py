#!/usr/bin/env python3
"""
diagnose_binding_bins.py
========================
For each energy bin where the global-fit model goes <= 0 inside the fit mask,
report *why*: the free-sign component coefficients, how many pixels are negative,
and whether the most-negative pixel sits at the GC cusp (NFW-driven) or off-plane
(bubble / fb_neg-driven). This distinguishes a benign low-count cusp dip from a
runaway free-sign degeneracy, and tells you whether re-extracting with the
Gaussian-flux likelihood is sufficient (it is for the former).

Usage
-----
    python diagnose_binding_bins.py --results-dir global_fit_results_fig6
    python diagnose_binding_bins.py --results-dir <dir> --floor 1e-30 --center-deg 5
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

FREE_SIGN_SUBSTRINGS = ("nfw", "halo", "fb_neg")


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose negative-model (floor-binding) bins.")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--floor", type=float, default=1e-30)
    ap.add_argument("--center-deg", type=float, default=5.0,
                    help="Pixels within this of the grid centre count as 'GC cusp'.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "global_fit_k*.npz")))
    if not files:
        print(f"No global_fit_k*.npz in {args.results_dir}")
        return 1

    n_binding = 0
    for path in files:
        d = np.load(path, allow_pickle=True)
        if "model_map" not in d or "mask_2d" not in d:
            print("  (maps not saved — re-run the fit without --no-save-maps to diagnose)")
            return 1

        labels = [str(x) for x in np.atleast_1d(d["labels"]).tolist()]
        theta = np.asarray(d["theta_ml"], float)
        model = np.asarray(d["model_map"], float)
        mask = np.asarray(d["mask_2d"], bool)
        k = int(d["k"]) if "k" in d else -1
        E = float(d["Ectr_mev"]) / 1e3 if "Ectr_mev" in d else float("nan")
        roi_lat = float(d["roi_lat"]) if "roi_lat" in d else None

        in_mask = mask & np.isfinite(model)
        vals = model[in_mask]
        if vals.size == 0:
            continue
        n_le0 = int(np.sum(vals <= 10.0 * args.floor))
        if n_le0 == 0:
            continue
        n_binding += 1

        # locate the most-negative pixel
        m = np.where(in_mask, model, np.inf)
        iy, ix = np.unravel_index(int(np.argmin(m)), m.shape)
        ny, nx = model.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        # approximate pixel scale from ROI extent if available
        binsz = (2.0 * roi_lat / ny) if (roi_lat and ny) else 0.125
        dist_deg = float(np.hypot(iy - cy, ix - cx) * binsz)
        where = "GC cusp (central)" if dist_deg <= args.center_deg else f"off-centre (~{dist_deg:.1f} deg)"

        # free-sign coefficients
        fs = [(lab, float(theta[j])) for j, lab in enumerate(labels)
              if any(s in lab.lower() for s in FREE_SIGN_SUBSTRINGS)]
        driver = min(fs, key=lambda t: t[1]) if fs else ("(none)", 0.0)

        print(f"\n[bin {k:02d}]  E = {E:.2f} GeV   negative pixels: {n_le0}   min model = {float(vals.min()):.3e}")
        print(f"   most-negative pixel: {where}")
        for lab, th in fs:
            tag = "  <-- most negative" if (lab, th) == driver else ""
            print(f"   free-sign  {lab:<48s} theta = {th:+.4e}{tag}")
        verdict = ("NFW cusp driven negative — Gaussian-flux re-extraction fixes this"
                   if ("nfw" in driver[0].lower() and "cusp" in where)
                   else "fb_neg/bubble driven negative — check for degeneracy, then Gaussian-flux"
                   if "fb_neg" in driver[0].lower()
                   else "mixed/other — inspect")
        print(f"   verdict: {verdict}")

    print("\n" + "=" * 60)
    if n_binding == 0:
        print("No binding bins found in this directory.")
    else:
        print(f"{n_binding} binding bin(s). If these are low-count and/or cusp-driven with "
              f"modest |theta|, re-extract the spectrum with the Gaussian-flux likelihood "
              f"(stat='gaussian_flux') — do NOT use strict (it biases the deficit bins upward).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())