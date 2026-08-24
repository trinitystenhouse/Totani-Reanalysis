#!/usr/bin/env python3
r"""
Build a per-energy-bin SYSTEMATIC error budget for the halo flux from the global
systematics suite, for use in the spectral DM fit.

Method
------
Each systematics variant produces a halo spectrum phi_v(E) (the `nfw` column of its
`mcmc_global_f_ml.txt`).  For each systematic GROUP g (diffuse, point_source_mask,
psf, ...), the 1-sigma systematic is the maximum absolute deviation from baseline
across that group's variants:

    sigma_g(E) = max_{v in g} | phi_v(E) - phi_base(E) | .

Independent groups combine in quadrature:

    sigma_sys(E) = sqrt( sum_g sigma_g(E)^2 ) .

This per-bin sigma_sys(E) is added in quadrature to the statistical halo error in
the spectral fit (scan_tension_resolution.py --sys-budget ...), which is the fix
for the chi2/ndof ~ 8 (statistics-only) problem.

Notes
-----
* `mock` variants are validation (closure / null), NOT alternative models, so they
  are excluded from the additive budget by default and reported separately:
  the `mock_null_nohalo` halo flux is the spurious-halo FLOOR (a bias diagnostic).
* Extreme stress-test variants that remove a real component entirely (default:
  diffuse_no_bubbles) are excluded from the realistic budget by default but reported;
  pass --include-stress to fold them in for a conservative bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

DEFAULT_GROUPS = ("diffuse", "point_source_mask", "psf")
DEFAULT_EXCLUDE = ("diffuse_no_bubbles",   # stress test: removes a real component
                   "diffuse_giem")          # double-counts Loop I/bubbles; use diffuse_giem_only
MOCK_GROUP = "mock"


def _resolve_halo_column(names, label="nfw"):
    if label in names:
        return label
    for n in names:
        if "nfw" in n.lower() and not n.lower().endswith(("_err_lo", "_err_hi")):
            return n
    raise KeyError(f"no halo column in {names}")


def load_halo(path: Path):
    arr = np.genfromtxt(path, names=True, comments="#", dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([arr])
    names = arr.dtype.names or ()
    lab = _resolve_halo_column(names)
    return np.asarray(arr["E_GeV"], float), np.asarray(arr[lab], float)


def variant_table(name, plot_root, global_root):
    for root in (plot_root, global_root):
        p = root / name / "mcmc_global_f_ml.txt"
        if p.exists():
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT / "mcmc" / "configs" / "systematics" / "manifest.json")
    ap.add_argument("--plot-root", type=Path,
                    default=PROJECT / "figures" / "plots_global_systematics")
    ap.add_argument("--global-root", type=Path,
                    default=PROJECT / "mcmc" / "global_systematics")
    ap.add_argument("--baseline", default="baseline_rho2")
    ap.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    ap.add_argument("--exclude", nargs="+", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--include-stress", action="store_true",
                    help="Fold the stress-test variants (default-excluded) into the budget.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "results" / "global-systematics" / "systematic_budget.txt")
    args = ap.parse_args(argv)

    variants = json.loads(args.manifest.read_text()).get("variants", [])
    exclude = set() if args.include_stress else set(args.exclude)

    # baseline
    bpath = variant_table(args.baseline, args.plot_root, args.global_root)
    if bpath is None:
        raise SystemExit(f"baseline {args.baseline} table not found")
    E0, phi0 = load_halo(bpath)

    # per-group deviations
    group_sigma = {g: np.zeros_like(phi0) for g in args.groups}
    used = {g: [] for g in args.groups}
    mock_null = np.full_like(phi0, np.nan)
    for item in variants:
        name, grp = item["name"], item.get("group")
        if name == args.baseline:
            continue
        p = variant_table(name, args.plot_root, args.global_root)
        if p is None:
            print(f"[skip] no table for {name}")
            continue
        Ev, phiv = load_halo(p)
        phiv_i = np.interp(E0, Ev, phiv, left=np.nan, right=np.nan)
        if grp == MOCK_GROUP:
            if name == "mock_null_nohalo":
                mock_null = phiv_i
            continue
        if grp in group_sigma and name not in exclude:
            dev = np.abs(phiv_i - phi0)
            group_sigma[grp] = np.fmax(group_sigma[grp], np.nan_to_num(dev, nan=0.0))
            used[grp].append(name)

    sigma_sys = np.sqrt(sum(group_sigma[g] ** 2 for g in args.groups))
    frac = sigma_sys / np.where(np.abs(phi0) > 0, np.abs(phi0), np.nan)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["E_GeV", "phi_baseline"] + [f"sig_{g}" for g in args.groups] + \
           ["sigma_sys", "sigma_sys_frac", "mock_null_floor"]
    with open(args.out, "w") as fh:
        fh.write("# Per-bin halo systematic budget (max-deviation per group, quadrature combine).\n")
        fh.write(f"# baseline={args.baseline}  groups={args.groups}  excluded={sorted(exclude)}\n")
        for g in args.groups:
            fh.write(f"# group {g}: variants used = {used[g]}\n")
        fh.write("# units: phi, sig_* in MeV cm^-2 s^-1 sr^-1 ; frac dimensionless\n")
        fh.write("# " + " ".join(cols) + "\n")
        for i in range(len(E0)):
            row = [E0[i], phi0[i]] + [group_sigma[g][i] for g in args.groups] + \
                  [sigma_sys[i], frac[i], mock_null[i]]
            fh.write(" ".join(f"{v:.6e}" for v in row) + "\n")

    print(f"Wrote {args.out}")
    print(f"{'E[GeV]':>8} {'phi_base':>11} {'sigma_sys':>11} {'frac':>7} {'mock_null':>11}")
    for i in range(len(E0)):
        print(f"{E0[i]:8.1f} {phi0[i]:11.3e} {sigma_sys[i]:11.3e} {frac[i]:7.2f} {mock_null[i]:11.3e}")
    pk = int(np.nanargmax(phi0))
    print(f"\nAt the peak ({E0[pk]:.1f} GeV): stat-only relerr ~ 0.05, systematic frac = {frac[pk]:.2f}")


if __name__ == "__main__":
    main()
