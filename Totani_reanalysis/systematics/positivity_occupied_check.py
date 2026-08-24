#!/usr/bin/env python3
"""
Does the total-positivity floor ever bind at an OCCUPIED pixel?

The pipeline fits a sign-free halo and a sign-free negative-bubble template,
so it needs to be verified that the interaction between the negative-bubble
template and the total-positivity floor does not shape the fit at pixels
that actually carry photons.

With a free-sign halo and negative-bubble template, the total model can go
non-positive in some pixels; the fitter floors it before taking the log. If that
floor bound at a pixel with counts > 0, the Poisson likelihood there would be
wrong. This script shows it never does.

Counts are reconstructed exactly as resid_map + model_map (verified integer to
machine precision), so no FITS access is required.

Usage:
    python systematics/positivity_occupied_check.py
    python systematics/positivity_occupied_check.py --results-dir <dir>
"""

import argparse
import glob
import os

import numpy as np


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=os.path.join(repo, "mcmc", "fit_results",
                                         "quick_global", "global_fit_results_fig6"))
    ap.add_argument("--outfile",
                    default=os.path.join(repo, "results", "posterior-covariance",
                                         "positivity_occupied_check.txt"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "global_fit_k*.npz")))
    if not files:
        raise SystemExit(f"no global_fit_k*.npz in {args.results_dir}")
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    rows, total_bad, integer_ok = [], 0, True
    for f in files:
        d = np.load(f, allow_pickle=True)
        model, resid, mask = d["model_map"], d["resid_map"], d["mask_2d"]
        counts = resid + model

        finite = np.isfinite(model) & mask
        # counts must be integer Poisson data; verify the reconstruction
        frac = np.abs(counts[finite] - np.round(counts[finite]))
        if frac.size and np.nanmax(frac) > 1e-6:
            integer_ok = False

        floored = finite & (model <= 0)
        occupied = finite & (counts > 0)
        bad = floored & occupied
        total_bad += int(bad.sum())

        rows.append((
            int(d["k"]), float(d["Ectr_mev"]) / 1e3, int(finite.sum()),
            int(floored.sum()), 100.0 * floored.sum() / max(finite.sum(), 1),
            int(occupied.sum()), 100.0 * occupied.sum() / max(finite.sum(), 1),
            int(bad.sum()),
        ))

    lines = []
    lines.append("# Positivity floor vs pixel occupancy")
    lines.append(f"# source: {args.results_dir}")
    lines.append(f"# counts reconstructed as resid_map + model_map; integer check "
                 f"{'PASSED' if integer_ok else 'FAILED'}")
    lines.append("")
    lines.append(f"{'bin':>4} {'E[GeV]':>9} {'fit px':>9} {'model<=0':>9} {'%':>7} "
                 f"{'occupied':>10} {'occ %':>7} {'floored & occupied':>19}")
    lines.append("-" * 82)
    for k, e, npix, nfl, pfl, nocc, pocc, nbad in rows:
        lines.append(f"{k:>4} {e:>9.2f} {npix:>9d} {nfl:>9d} {pfl:>7.3f} "
                     f"{nocc:>10d} {pocc:>7.2f} {nbad:>19d}")
    lines.append("")
    lines.append(f"TOTAL pixels where the floor binds AND counts > 0 : {total_bad}")
    lines.append("")
    if total_bad == 0:
        lines.append("CONCLUSION: the positivity floor never binds at a pixel carrying")
        lines.append("photons. Every floored pixel is empty, so the floor cannot shape")
        lines.append("the fit -- an empty pixel's Poisson term contributes only -mu, and")
        lines.append("the floor acts solely to keep log(mu) finite. The negative-bubble")
        lines.append("template therefore does not distort the likelihood at occupied pixels.")
    else:
        lines.append(f"WARNING: {total_bad} occupied pixels are floored. The Poisson")
        lines.append("likelihood and Fisher curvature are wrong there. Investigate before")
        lines.append("quoting any result from this fit.")

    out = "\n".join(lines)
    print(out)
    with open(args.outfile, "w") as fh:
        fh.write(out + "\n")
    print(f"\nwrote -> {args.outfile}")


if __name__ == "__main__":
    main()
