#!/usr/bin/env python3
r"""
hardic_report.py
=================
Collect and tabulate the hard inverse-Compton injected-population metrics
from the three deterministic-fit run families:

    A = baseline_rho2       (halo, no hardic)   -- mcmc/global_systematics/baseline_rho2/
    B = hardic_<tag>        (halo + hardic)     -- mcmc/global_systematics/hardic_<tag>/
    C = hardic_only_<tag>   (hardic, no halo)   -- mcmc/hardic_only/<tag>/
    A_nohalo (optional)     (no halo, no hardic; the existing fig4-style
                             baseline) -- mcmc/global_fit_results_fig4/ by default

All three of A/B/C share the same base config (mask/ROI/pixel set), so
per-bin ``loglike_ml`` differences are directly comparable.  Reads
``global_fit_k{k:02d}.npz`` written by ``run_global_fit.py`` -- see that
script's ``ext_dict`` for the exact schema this reads (``labels``,
``theta_ml``, ``theta_cov``, ``loglike_ml``).

Metrics reported per tag (items 1-5; item 6, tension propagation, is
automatic once the variant is in the systematics manifest --
``systematics/propagate_systematics.py`` -- and is not duplicated here):

    1. Halo survival statistic:      2 * sum_k [lnL(B_k) - lnL(C_k)]
    2. Morphology contest:           2 * sum_k [lnL(C_k) - lnL(A_nohalo,k)]
                                      2 * sum_k [lnL(C_k) - lnL(A_k)]
    3. Halo amplitude change:        theta_halo(B) / theta_halo(A), peak
                                      bin (20.76 GeV) and band-average
    4. Correlation:                  r(halo, hardic), r(hardic, ics),
                                      r(hardic, fb_pos) per bin, from B's
                                      theta_cov (Fisher/curvature estimate)
    5. Physicality:                  theta_hardic(E) and the implied
                                      E_inj = theta_hardic * E_inj_ref
                                      (E_inj_ref read from the template
                                      FITS header HCEINJ keyword, default
                                      1e52 erg at theta=1)

The default signal band is bins k04-k06 (10-40 GeV; verified bin-centre
energies 12.29 / 20.76 / 35.06 GeV).

Usage
-----
    python hardic_report.py --tags burst_s burst_m burst_l cont adv cmbonly_burst_m

    # against custom result directories:
    python hardic_report.py --tags burst_m \
        --systematics-dir mcmc/global_systematics \
        --hardic-only-dir mcmc/hardic_only \
        --nohalo-dir mcmc/global_fit_results_fig4

Output: a plain-text table to stdout, and (with --outfile) the same table
written to disk.  This script only reads existing global_fit_k*.npz
files; it does not run any fits itself (see build_hardic_template.py and
run_global_from_config.py for that).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REANALYSIS = os.path.dirname(HERE)

DEFAULT_SYSTEMATICS_DIR = os.path.join(REANALYSIS, "mcmc", "global_systematics")
DEFAULT_HARDIC_ONLY_DIR = os.path.join(REANALYSIS, "mcmc", "hardic_only")
# The deterministic production runs live under mcmc/fit_results/quick_global/, NOT
# directly under mcmc/. Pointing these at mcmc/global_fit_results_fig4 (as an earlier
# revision did) silently finds nothing, and pointing the baseline at
# global_systematics/baseline_rho2/ picks up a 1-bin smoke test rather than the full
# 13-bin production run -- both give a report that looks empty rather than erroring.
DEFAULT_QUICK_GLOBAL = os.path.join(REANALYSIS, "mcmc", "fit_results", "quick_global")
DEFAULT_NOHALO_DIR = os.path.join(DEFAULT_QUICK_GLOBAL, "global_fit_results_fig4")
# A = halo, no hard-IC: the rho^2 baseline production run (all 13 bins).
DEFAULT_BASELINE_DIR = os.path.join(DEFAULT_QUICK_GLOBAL, "global_fit_results_fig6")
DEFAULT_TEMPLATES_DIR = os.path.join(
    os.path.dirname(REANALYSIS), "fermi_data", "totani", "processed", "templates",
)

RHO2_LABEL = "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"
DEFAULT_BAND_BINS = (4, 5, 6)  # 12.29 / 20.76 / 35.06 GeV -- the "10-40 GeV" signal band
DEFAULT_NBINS = 13


def _load_bin(run_dir: str, k: int):
    path = os.path.join(run_dir, f"global_fit_k{k:02d}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as d:
        return {
            "labels": [str(x) for x in d["labels"]],
            "theta_ml": np.asarray(d["theta_ml"], dtype=float),
            "theta_cov": np.asarray(d["theta_cov"], dtype=float) if "theta_cov" in d else None,
            "loglike_ml": float(d["loglike_ml"]),
            "Ectr_mev": float(d["Ectr_mev"]) if "Ectr_mev" in d else None,
        }


def _load_run(run_dir: str, nbins: int = DEFAULT_NBINS):
    bins = {}
    for k in range(nbins):
        b = _load_bin(run_dir, k)
        if b is not None:
            bins[k] = b
    return bins


def _find_index(labels, target_substrings, exclude_substrings=()):
    """First label containing ALL of target_substrings and NONE of
    exclude_substrings (case-insensitive); None if not found."""
    for i, lab in enumerate(labels):
        lab_l = lab.lower()
        if all(t.lower() in lab_l for t in target_substrings) and not any(
            e.lower() in lab_l for e in exclude_substrings
        ):
            return i
    return None


def _corr(cov, i, j):
    if cov is None or i is None or j is None:
        return np.nan
    vii, vjj, vij = cov[i, i], cov[j, j], cov[i, j]
    if vii <= 0 or vjj <= 0:
        return np.nan
    return float(vij / np.sqrt(vii * vjj))


def _read_einj_ref(templates_dir: str, tag: str) -> float:
    """Read HCEINJ (erg at theta=1) from the template FITS header; falls
    back to the module default (1e52 erg) if astropy/the file are unavailable."""
    default = 1.0e52
    path = os.path.join(templates_dir, f"hardic_{tag}_dnde.fits")
    if not os.path.exists(path):
        return default
    try:
        from astropy.io import fits

        with fits.open(path) as h:
            return float(h[0].header.get("HCEINJ", default))
    except Exception:
        return default


def report_tag(tag: str, *, systematics_dir: str, hardic_only_dir: str, nohalo_dir: str,
                templates_dir: str, band_bins, nbins: int, baseline_name: str = "baseline_rho2",
                baseline_dir: str = None):
    label_hardic = f"hardic_{tag}"
    # Prefer an explicit baseline directory; fall back to the old
    # <systematics_dir>/<baseline_name> layout only if none was given.
    A_dir = baseline_dir or os.path.join(systematics_dir, baseline_name)
    A = _load_run(A_dir, nbins)
    B = _load_run(os.path.join(systematics_dir, label_hardic), nbins)
    C = _load_run(os.path.join(hardic_only_dir, tag), nbins)
    A_nohalo = _load_run(nohalo_dir, nbins)

    lines = []
    lines.append(f"=== {label_hardic} ===")
    if not B:
        lines.append(f"  NO RESULTS FOUND under {os.path.join(systematics_dir, label_hardic)} "
                      f"-- run the fit first (see module docstring / build_hardic_template.py).")
        return "\n".join(lines)

    einj_ref = _read_einj_ref(templates_dir, tag)

    # ---- 1. halo survival statistic ----
    common_bk = sorted(set(B) & set(C))
    band_bk = [k for k in common_bk if k in band_bins]
    two_dlnl_all = 2.0 * sum(B[k]["loglike_ml"] - C[k]["loglike_ml"] for k in common_bk)
    two_dlnl_band = 2.0 * sum(B[k]["loglike_ml"] - C[k]["loglike_ml"] for k in band_bk)
    lines.append(f"  [1] Halo survival 2*sum[lnL(B)-lnL(C)]: "
                 f"all-bins={two_dlnl_all:.2f} (n={len(common_bk)}), "
                 f"band={two_dlnl_band:.2f} (n={len(band_bk)}) "
                 f"[paper baseline (disk-excl., deterministic): ~870]")

    # ---- 2. morphology contest ----
    if A_nohalo:
        common_ca = sorted(set(C) & set(A_nohalo))
        band_ca = [k for k in common_ca if k in band_bins]
        two_dlnl_c_anohalo_all = 2.0 * sum(C[k]["loglike_ml"] - A_nohalo[k]["loglike_ml"] for k in common_ca)
        two_dlnl_c_anohalo_band = 2.0 * sum(C[k]["loglike_ml"] - A_nohalo[k]["loglike_ml"] for k in band_ca)
        lines.append(f"  [2a] Morphology 2*sum[lnL(C)-lnL(A_nohalo)]: "
                     f"all-bins={two_dlnl_c_anohalo_all:.2f}, band={two_dlnl_c_anohalo_band:.2f}")
    else:
        lines.append(f"  [2a] A_nohalo results not found under {nohalo_dir} -- skipped.")

    if A:
        common_ca2 = sorted(set(C) & set(A))
        band_ca2 = [k for k in common_ca2 if k in band_bins]
        two_dlnl_c_a_all = 2.0 * sum(C[k]["loglike_ml"] - A[k]["loglike_ml"] for k in common_ca2)
        two_dlnl_c_a_band = 2.0 * sum(C[k]["loglike_ml"] - A[k]["loglike_ml"] for k in band_ca2)
        lines.append(f"  [2b] Morphology 2*sum[lnL(C)-lnL(A)]:        "
                     f"all-bins={two_dlnl_c_a_all:.2f}, band={two_dlnl_c_a_band:.2f}")
    else:
        lines.append(f"  [2b] A (baseline_rho2) results not found under "
                     f"{A_dir} -- skipped.")

    # ---- 3, 4, 5: per-bin halo amplitude / correlation / hardic physicality ----
    lines.append(f"  {'k':>3} {'E[GeV]':>9} {'theta_halo(A)':>14} {'theta_halo(B)':>14} "
                 f"{'ratio':>7} {'r(halo,hc)':>11} {'r(hc,ics)':>10} {'r(hc,fbpos)':>12} "
                 f"{'theta_hc':>10} {'E_inj[erg]':>11}")
    ratios = []
    for k in sorted(B):
        b = B[k]
        labels = b["labels"]
        i_halo = _find_index(labels, ["nfw"])
        i_hc = _find_index(labels, ["hardic"])
        i_ics = _find_index(labels, ["ics"], exclude_substrings=["hardic"])
        i_fbpos = _find_index(labels, ["fb_pos"])

        theta_halo_B = b["theta_ml"][i_halo] if i_halo is not None else np.nan
        theta_halo_A = (A[k]["theta_ml"][_find_index(A[k]["labels"], ["nfw"])]
                         if (A and k in A and _find_index(A[k]["labels"], ["nfw"]) is not None)
                         else np.nan)
        ratio = theta_halo_B / theta_halo_A if (np.isfinite(theta_halo_A) and theta_halo_A != 0) else np.nan
        if k in band_bins and np.isfinite(ratio):
            ratios.append(ratio)

        r_halo_hc = _corr(b["theta_cov"], i_halo, i_hc)
        r_hc_ics = _corr(b["theta_cov"], i_hc, i_ics)
        r_hc_fbpos = _corr(b["theta_cov"], i_hc, i_fbpos)
        theta_hc = b["theta_ml"][i_hc] if i_hc is not None else np.nan
        e_inj = theta_hc * einj_ref if np.isfinite(theta_hc) else np.nan
        E_gev = (b["Ectr_mev"] / 1.0e3) if b["Ectr_mev"] else np.nan

        band_flag = "*" if k in band_bins else " "
        lines.append(f"  {k:>3}{band_flag} {E_gev:>8.2f} {theta_halo_A:>14.4e} {theta_halo_B:>14.4e} "
                     f"{ratio:>7.3f} {r_halo_hc:>11.3f} {r_hc_ics:>10.3f} {r_hc_fbpos:>12.3f} "
                     f"{theta_hc:>10.4e} {e_inj:>11.3e}")

    if ratios:
        lines.append(f"  Band-average halo amplitude ratio theta(B)/theta(A): "
                     f"{np.mean(ratios):.3f} (n={len(ratios)} bins)")
    lines.append(f"  (E_inj reference: theta=1 <-> {einj_ref:.3e} erg, from template FITS header "
                 f"HCEINJ; '*' marks the {band_bins} signal-band bins.)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+",
                    default=["burst_s", "burst_m", "burst_l", "cont", "adv", "cmbonly_burst_m"])
    ap.add_argument("--systematics-dir", default=DEFAULT_SYSTEMATICS_DIR)
    ap.add_argument("--hardic-only-dir", default=DEFAULT_HARDIC_ONLY_DIR)
    ap.add_argument("--nohalo-dir", default=DEFAULT_NOHALO_DIR)
    ap.add_argument("--templates-dir", default=DEFAULT_TEMPLATES_DIR)
    ap.add_argument("--baseline-name", default="baseline_rho2")
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR,
                    help="explicit directory for the A family (halo, no hard-IC); "
                         "defaults to the 13-bin rho^2 production run")
    ap.add_argument("--band-bins", nargs="+", type=int, default=list(DEFAULT_BAND_BINS))
    ap.add_argument("--nbins", type=int, default=DEFAULT_NBINS)
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    band_bins = set(args.band_bins)
    blocks = [report_tag(
        tag, systematics_dir=args.systematics_dir, hardic_only_dir=args.hardic_only_dir,
        nohalo_dir=args.nohalo_dir, templates_dir=args.templates_dir,
        band_bins=band_bins, nbins=args.nbins, baseline_name=args.baseline_name,
        baseline_dir=args.baseline_dir,
    ) for tag in args.tags]

    text = "\n\n".join(blocks) + "\n"
    print(text)
    if args.outfile:
        os.makedirs(os.path.dirname(os.path.abspath(args.outfile)) or ".", exist_ok=True)
        with open(args.outfile, "w") as f:
            f.write(text)
        print(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main()
