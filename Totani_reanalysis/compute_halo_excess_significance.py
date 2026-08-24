#!/usr/bin/env python3
"""
Summarise the significance of the halo-like excess from saved MCMC text outputs.

This compares a "halo" fit directory against a baseline "no-halo" directory,
computes per-bin and total Delta lnL, and reports:

  - TS = 2 * sum(Delta lnL)
  - sqrt(TS), the usual single-parameter Wilks-equivalent sigma
  - a chi^2-based p-value/sigma using ndof = number of included bins

The latter is more conservative when the fit introduces one halo amplitude per
energy bin.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List

import numpy as np


DEFAULT_BASE = os.path.join(
    os.environ.get(
        "REPO_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    ),
    "Totani_paper_check",
    "mcmc",
)

PROFILE_TO_FIG = {
    "rho25": "fig5",
    "rho2.5": "fig5",
    "fig5": "fig5",
    "rho2": "fig6",
    "fig6": "fig6",
    "rho1": "fig7",
    "fig7": "fig7",
}


def parse_result(path: str) -> dict | None:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    e_match = re.search(r"E_ctr=(\S+)\s*MeV", text)
    l_match = re.search(r"Log-likelihood \(ML\):\s*([+-]?\S+)", text)
    if e_match is None or l_match is None:
        return None

    return {
        "energy_gev": float(e_match.group(1)) / 1000.0,
        "lnl": float(l_match.group(1)),
    }


def parse_npz_result(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as npz:
        if "loglike_ml" not in npz:
            return None
        energy_gev = math.nan
        if "Ectr_mev" in npz:
            energy_gev = float(np.asarray(npz["Ectr_mev"]).reshape(-1)[0]) / 1000.0
        return {
            "energy_gev": energy_gev,
            "lnl": float(np.asarray(npz["loglike_ml"]).reshape(-1)[0]),
        }


def load_directory(path: str, *, use_npz: bool = True) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if not os.path.isdir(path):
        return rows

    txt_names = sorted(name for name in os.listdir(path) if re.fullmatch(r"mcmc_results_k\d+\.txt", name))
    for name in txt_names:
        if not re.fullmatch(r"mcmc_results_k\d+\.txt", name):
            continue
        parsed = parse_result(os.path.join(path, name))
        if parsed is not None:
            rows[name] = parsed

    if rows or not use_npz:
        return rows

    for name in sorted(os.listdir(path)):
        if not re.fullmatch(r"mcmc_results_k\d+\.npz", name):
            continue
        parsed = parse_npz_result(os.path.join(path, name))
        if parsed is not None:
            rows[name.replace(".npz", ".txt")] = parsed
    return rows


def require_existing_dirs(paths: dict[str, str], *, context: str) -> None:
    missing = [f"{key}: {path}" for key, path in paths.items() if not os.path.isdir(path)]
    if not missing:
        return
    missing_text = "\n  ".join(missing)
    raise SystemExit(
        f"{context} requires matching result directories, but these are missing:\n"
        f"  {missing_text}"
    )


def resolve_result_dirs(
    *,
    base_dir: str,
    family: str,
    halo_profile: str,
    tag: str | None,
    with_disk: bool,
) -> tuple[str, str]:
    profile_key = str(halo_profile).strip().lower()
    if profile_key not in PROFILE_TO_FIG:
        choices = ", ".join(sorted(PROFILE_TO_FIG))
        raise SystemExit(f"Unknown --halo-profile {halo_profile!r}. Choices/aliases: {choices}")
    fig = PROFILE_TO_FIG[profile_key]

    if family == "global":
        suffix = "_w_disk" if with_disk else ""
        baseline = os.path.join(base_dir, f"global_fit_results_fig4{suffix}")
        halo = os.path.join(base_dir, f"global_fit_results_{fig}{suffix}")
        return baseline, halo

    if with_disk:
        raise SystemExit("--with-disk is only defined for --family global.")

    suffix = ""
    if tag:
        tag_clean = str(tag).strip().strip("_")
        suffix = f"_{tag_clean}" if tag_clean else ""
    baseline = os.path.join(base_dir, f"mcmc_results_fig4{suffix}")
    halo = os.path.join(base_dir, f"mcmc_results_{fig}{suffix}")
    return baseline, halo


def gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), stdlib-only."""
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 1.0

    eps = 3.0e-14
    fpmin = 1.0e-300
    gln = math.lgamma(a)

    if x < a + 1.0:
        ap = a
        delta = 1.0 / a
        total = delta
        while True:
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    i = 1
    while True:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
        i += 1
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def chi2_sf(x: float, ndof: int) -> float:
    if ndof <= 0:
        raise ValueError("ndof must be positive")
    if x <= 0.0:
        return 1.0
    return gammq(0.5 * ndof, 0.5 * x)


def p_to_sigma_one_sided(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    if p <= 0.0:
        return math.inf
    if p >= 1.0:
        return 0.0

    # Invert p = 0.5 * erfc(z / sqrt(2)) with bisection so we remain stable
    # even when 1 - p rounds to 1.0 in floating-point arithmetic.
    lo = 0.0
    hi = 1.0
    while 0.5 * math.erfc(hi / math.sqrt(2.0)) > p and hi < 64.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p_mid = 0.5 * math.erfc(mid / math.sqrt(2.0))
        if p_mid > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def iter_deltas(
    base_rows: Dict[str, dict],
    halo_rows: Dict[str, dict],
    emin_gev: float,
    emax_gev: float,
) -> List[dict]:
    out: List[dict] = []
    for key in sorted(base_rows):
        if key not in halo_rows:
            continue
        energy = halo_rows[key]["energy_gev"]
        if not (emin_gev <= energy <= emax_gev):
            continue
        dlnl = halo_rows[key]["lnl"] - base_rows[key]["lnl"]
        out.append(
            {
                "bin": key,
                "energy_gev": energy,
                "delta_lnl": dlnl,
                "ts_bin": 2.0 * dlnl,
                "sigma_bin": math.sqrt(max(0.0, 2.0 * dlnl)),
            }
        )
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute halo excess significance from saved Totani MCMC outputs."
    )
    ap.add_argument(
        "--baseline-dir",
        default=None,
        help="No-halo results directory. Overrides --family/--tag if supplied.",
    )
    ap.add_argument(
        "--halo-dir",
        default=None,
        help="Halo-fit results directory to compare against baseline. Overrides --family/--halo-profile if supplied.",
    )
    ap.add_argument(
        "--base-dir",
        default=DEFAULT_BASE,
        help="Base directory containing mcmc/global result folders.",
    )
    ap.add_argument(
        "--family",
        choices=["totani", "global"],
        default="totani",
        help="Use Totani/cellwise MCMC folders or global-fit folders.",
    )
    ap.add_argument(
        "--halo-profile",
        default="rho2",
        help="Halo profile to compare: rho25/fig5, rho2/fig6, or rho1/fig7.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Optional Totani/cellwise suffix such as '10deg'. Ignored for --family global.",
    )
    ap.add_argument(
        "--with-disk",
        action="store_true",
        help=(
            "For --family global, use matching global_fit_results_fig4_w_disk and "
            "global_fit_results_fig[5-7]_w_disk directories. Totani Fig. 9 uses the "
            "default disk-excluded fits."
        ),
    )
    ap.add_argument(
        "--no-npz",
        action="store_true",
        help="Disable NPZ fallback when mcmc_results_k*.txt sidecars are absent.",
    )
    ap.add_argument(
        "--emin-gev",
        type=float,
        default=1.0,
        help="Minimum photon energy to include [GeV].",
    )
    ap.add_argument(
        "--emax-gev",
        type=float,
        default=1000.0,
        help="Maximum photon energy to include [GeV].",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many highest-Delta lnL bins to print.",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()

    resolved_baseline, resolved_halo = resolve_result_dirs(
        base_dir=args.base_dir,
        family=args.family,
        halo_profile=args.halo_profile,
        tag=args.tag,
        with_disk=args.with_disk,
    )
    baseline_dir = args.baseline_dir or resolved_baseline
    halo_dir = args.halo_dir or resolved_halo
    if args.with_disk:
        require_existing_dirs(
            {"baseline": baseline_dir, "halo": halo_dir},
            context=(
                "Disk-included halo significance is a diagnostic, not Totani's default Fig. 9, "
                "so it"
            ),
        )

    use_npz = not bool(args.no_npz)
    base_rows = load_directory(baseline_dir, use_npz=use_npz)
    halo_rows = load_directory(halo_dir, use_npz=use_npz)
    deltas = iter_deltas(base_rows, halo_rows, args.emin_gev, args.emax_gev)

    if not deltas:
        raise SystemExit(
            "No overlapping result bins found in the requested energy range.\n"
            f"  baseline-dir={baseline_dir}\n"
            f"  halo-dir={halo_dir}\n"
            f"  txt/npz fallback enabled={use_npz}"
        )

    total_dlnl = sum(row["delta_lnl"] for row in deltas)
    ts_total = 2.0 * total_dlnl
    nbins = len(deltas)
    p_multi = chi2_sf(ts_total, nbins)
    sigma_multi = p_to_sigma_one_sided(p_multi)
    sigma_single = math.sqrt(max(0.0, ts_total))

    halo_label = os.path.basename(os.path.normpath(halo_dir))
    base_label = os.path.basename(os.path.normpath(baseline_dir))

    print("=" * 88)
    print("Halo Excess Significance Summary")
    print("=" * 88)
    print(f"Family       : {args.family}")
    print(f"Halo profile : {args.halo_profile}")
    if args.tag:
        print(f"Tag          : {args.tag}")
    if args.with_disk:
        print("With disk    : true")
    print(f"Baseline dir : {baseline_dir}")
    print(f"Halo dir     : {halo_dir}")
    print(f"Comparison   : {halo_label} vs {base_label}")
    print(f"Energy range : {args.emin_gev:g} to {args.emax_gev:g} GeV")
    print(f"Bins used    : {nbins}")
    print(f"Sum Delta lnL: {total_dlnl:.6f}")
    print(f"TS total     : {ts_total:.6f}")
    print(f"sqrt(TS)     : {sigma_single:.3f} sigma")
    print(f"chi2 p-value : {p_multi:.3e}  (ndof={nbins})")
    if math.isfinite(sigma_multi):
        print(f"chi2 sigma   : {sigma_multi:.3f} sigma")
    else:
        print("chi2 sigma   : > 8 sigma (p-value underflow beyond normal precision)")

    print("\nTop contributing bins")
    print("-" * 88)
    print(f"{'bin':<20} {'E [GeV]':>10} {'Delta lnL':>14} {'TS':>12} {'sqrt(2DlnL)':>14}")
    for row in sorted(deltas, key=lambda r: r["delta_lnl"], reverse=True)[: max(0, args.top)]:
        print(
            f"{row['bin']:<20} {row['energy_gev']:10.3f} {row['delta_lnl']:14.3f} "
            f"{row['ts_bin']:12.3f} {row['sigma_bin']:14.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
