#!/usr/bin/env python3
r"""
Propagate the systematics suite through the DM fit the STATISTICALLY CORRECT way:
re-fit the annihilation spectrum to EACH variant's halo spectrum and report the
spread of the science results (best-fit m_chi, <sigma v>, dSph tension).

Why not just inflate per-bin errors?
------------------------------------
Template systematics (alternative bubble/diffuse model, PSF, mask) shift the whole
halo spectrum COHERENTLY across energy bins. Adding their per-bin envelope in
quadrature to the diagonal errors treats them as independent random scatter, which
(i) is the wrong likelihood and (ii) de-localizes the mass (the fit absorbs the
coherent shift). The correct systematic on a derived quantity X (= m_chi, <sv>,
tension) is the spread of X across the alternative analyses:

    X_central +(max_v X_v - X_central) / -(X_central - min_v X_v) .

That is what this script computes. (build_systematic_budget.py's diagonal budget is
retained only as a conservative goodness-of-fit diagnostic, not for the science band.)

Each variant halo spectrum is the `nfw` column of its mcmc_global_f_ml.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))

from shared.paths import GLOBAL_SYSTEMATICS, REANALYSIS, REANALYSIS_RESULTS  # noqa: E402

from core.spectral_reshaping import pppc_energy_flux_template, smooth_nfw_sigma_v_from_norm  # noqa: E402
from core.dsph_limits_multi import dsph_upper_limit  # noqa: E402

_SYSTEMATICS_PLOT_ROOT = REANALYSIS / "figures" / "systematics" / "plots_global_systematics"
_LEGACY_PLOT_ROOT = REANALYSIS / "figures" / "plots_global_systematics"


def _halo_col(names):
    if "nfw" in names:
        return "nfw"
    for n in names:
        if "nfw" in n.lower() and not n.lower().endswith(("_err_lo", "_err_hi")):
            return n
    raise KeyError("no nfw column")


def load_variant(path: Path):
    a = np.genfromtxt(path, names=True, comments="#", dtype=float)
    names = a.dtype.names
    lab = _halo_col(names)
    E = np.asarray(a["E_GeV"], float)
    phi = np.asarray(a[lab], float)
    lo, hi = f"{lab}_err_lo", f"{lab}_err_hi"
    err = 0.5 * (np.asarray(a[lo], float) + np.asarray(a[hi], float)) if lo in names else np.full_like(phi, np.nan)
    return E, phi, err


def fit_one(E, phi, err, channel, masses, source, table_path):
    mask = np.isfinite(phi) & np.isfinite(err) & (phi > 0) & (err > 0)
    best = dict(chi2=np.inf)
    for m in masses:
        try:
            shape = pppc_energy_flux_template(E, m, channel=channel, primary="gamma",
                                              table_path=table_path, normalise=False)
        except Exception:
            continue
        if not np.any(shape[mask] > 0):
            continue
        w = 1.0 / err[mask] ** 2
        A = np.sum(w * shape[mask] * phi[mask]) / np.sum(w * shape[mask] ** 2)
        if not np.isfinite(A) or A <= 0:
            continue
        chi2 = float(np.sum(((A * shape[mask] - phi[mask]) / err[mask]) ** 2))
        if chi2 < best["chi2"]:
            sv = smooth_nfw_sigma_v_from_norm(A, m)
            lim = dsph_upper_limit(m, channel, source)
            best = dict(chi2=chi2, ndof=int(mask.sum()) - 1, m=float(m),
                        sv=float(sv), tension=float(sv / lim) if np.isfinite(lim) and lim > 0 else np.nan)
    return best


# Groups that constitute the FOREGROUND leg of the systematic band. The baseline
# itself is always included as the reference point. See the note at the nb= line.
#
# NOTE ON BUBBLES -- diffuse_grid_bubbles, NOT diffuse_grid.
# The Fermi-bubble templates are residual-derived, so they depend on the diffuse
# model they were extracted against. Every other member of this leg already
# carries its own per-variant bubbles (diffuse_no_loopI, ics_morphology_flat,
# ics_morphology_steep) or does not use them by construction
# (diffuse_flat_bubbles uses fb_flat; diffuse_giem_only has none, since the GIEM
# already contains the bubbles). The IEM-grid variants are fitted BOTH ways:
#   diffuse_grid          -- grid gas/ICS with the BASELINE (Lorimer) bubbles
#   diffuse_grid_bubbles  -- grid gas/ICS with bubbles re-extracted per model
# Only the latter is internally consistent, and it is the same argument used to
# exclude diffuse_giem (double-counting large-scale residuals). Including both
# would also double-count the same fourteen models. The plain diffuse_grid fits
# are retained and reported separately as the size of the bubble-consistency
# effect (it moves the upper edge from 1.34 to 1.65).
FOREGROUND_GROUPS = {"baseline", "diffuse", "diffuse_ics", "diffuse_grid_bubbles"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path,
                   default=REANALYSIS / "mcmc" / "configs" / "systematics" / "manifest.json")
    p.add_argument("--plot-root", type=Path, default=_SYSTEMATICS_PLOT_ROOT)
    p.add_argument("--global-root", type=Path, default=GLOBAL_SYSTEMATICS)
    p.add_argument("--baseline", default="baseline_rho2")
    p.add_argument("--channel", default="bb")
    p.add_argument("--dsph-source", default="hoof")
    p.add_argument("--m-min", type=float, default=20.0)
    p.add_argument("--m-max", type=float, default=2000.0)
    p.add_argument("--n-mass", type=int, default=200)
    p.add_argument("--pppc-table", default=None)
    p.add_argument("--anchor-mass", type=float, default=736.0,
                   help="Validated baseline m_chi from run_pppc_mass_scan, global_fig6 "
                        "preset, 200-point grid over 20-2000 GeV [GeV].")
    p.add_argument("--anchor-sv", type=float, default=9.803e-25,
                   help="Validated baseline <sigma v> [cm^3/s].")
    p.add_argument("--anchor-tension", type=float, default=5.121,
                   help="Validated baseline dSph tension.")
    p.add_argument("--exclude", nargs="+",
                   default=["diffuse_no_bubbles", "diffuse_giem"],
                   help="Excluded from the band: diffuse_no_bubbles (stress test) and "
                        "diffuse_giem (double-counts Loop I/bubbles -> use diffuse_giem_only).")
    p.add_argument("--include-mock", action="store_true",
                   help="Also fit the mock variants (closure/null) -- diagnostic only.")
    p.add_argument("--out", type=Path,
                   default=REANALYSIS_RESULTS / "global-systematics" / "systematics_science_band.txt")
    args = p.parse_args(argv)

    variants = json.loads(args.manifest.read_text()).get("variants", [])
    masses = np.geomspace(args.m_min, args.m_max, args.n_mass)

    def table(name):
        for r in dict.fromkeys((args.plot_root, _LEGACY_PLOT_ROOT, args.global_root)):
            q = r / name / "mcmc_global_f_ml.txt"
            if q.exists():
                return q
        return None

    rows = []
    for it in variants:
        name, grp = it["name"], it.get("group")
        if name in args.exclude:
            continue
        if grp == "mock" and not args.include_mock:
            continue
        t = table(name)
        if t is None:
            print(f"[skip] {name}: no table"); continue
        E, phi, err = load_variant(t)
        b = fit_one(E, phi, err, args.channel, masses, args.dsph_source, args.pppc_table)
        if not np.isfinite(b["chi2"]):
            print(f"[skip] {name}: no fit"); continue
        rows.append((name, grp, b))

    base = next((b for n, g, b in rows if n == args.baseline), None)
    if base is None:
        raise SystemExit("baseline not fit")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdr = f"{'variant':24s} {'group':16s} {'m_chi':>7s} {'<sv>':>11s} {'tension':>8s} {'chi2/ndof':>9s}"
    lines = [f"# Systematics science band, channel={args.channel}, dsph={args.dsph_source}", hdr]
    print(hdr)
    for n, g, b in rows:
        line = f"{n:24s} {g or '':16s} {b['m']:7.0f} {b['sv']:11.3e} {b['tension']:8.2f} {b['chi2']/b['ndof']:9.2f}"
        print(line); lines.append(line)

    # NOTE: the systematics-suite tables sit on a different flux normalization than
    # the validated run_pppc_mass_scan halo spectrum, so the systematic is reported
    # as a FRACTIONAL band (normalization-independent) and anchored to the validated
    # baseline values (--anchor-*). The best-fit MASS is normalization-independent
    # and reported absolute.
    # FOREGROUND LEG. The band's foreground contribution is the extremal envelope
    # over variations of the INTERSTELLAR EMISSION MODEL only:
    #
    #   diffuse       -- GALPROP<->GIEM, Loop I, bubble parametrisation
    #   diffuse_ics   -- ICS/ISRF morphology
    #   diffuse_grid  -- the published Ackermann et al. (2012) IEM grid
    #
    # Everything else is a SEPARATE consistency check and must not enter this
    # envelope: quadrant/hemisphere are ROI tests, not diffuse-model variations;
    # cr_injection tests an additional template rather than an alternative
    # foreground; point_source_mask and psf are instrument-response checks
    # (measured at <=few %, reported separately); mock is a recovery test.
    #
    # Previously this line took every non-mock variant, which silently pulled the
    # quadrant tests into the foreground spread and disagreed with the hard-coded
    # edges in total_systematic_band.py. Selecting by group makes the choice
    # explicit and keeps the two scripts consistent.
    nb = [(n, g, b) for n, g, b in rows if g in FOREGROUND_GROUPS]
    if not nb:
        raise SystemExit(
            "No variants in the foreground groups %s -- nothing to build a band from. "
            "Check the group labels in the systematics manifest." % (sorted(FOREGROUND_GROUPS),)
        )
    other = sorted({g for _, g, _ in rows if g not in FOREGROUND_GROUPS and g != "baseline"})
    lines.append("")
    lines.append("# Foreground leg groups: %s" % ", ".join(sorted(FOREGROUND_GROUPS)))
    lines.append("# Reported separately (NOT in the foreground envelope): %s"
                 % (", ".join(other) if other else "none"))
    anchors = {"m": args.anchor_mass, "sv": args.anchor_sv, "tension": args.anchor_tension}
    lines.append("")
    lines.append(f"# Validated baseline anchors (from run_pppc_mass_scan): "
                 f"m={args.anchor_mass} GeV, <sv>={args.anchor_sv:.3g}, tension={args.anchor_tension:.3g}")
    for key, lab in [("m", "m_chi [GeV]"), ("sv", "<sigma v>"), ("tension", "dSph tension")]:
        vals = np.array([b[key] for _, _, b in nb], float)
        c = base[key]
        frac_lo, frac_hi = np.nanmin(vals) / c, np.nanmax(vals) / c
        a = anchors[key]
        if key == "m":  # mass is normalization-independent: report absolute spread directly
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        else:
            lo, hi = a * frac_lo, a * frac_hi
        worst = max(nb, key=lambda r: abs(r[2][key] / c - 1.0))[0]
        lines.append(f"# {lab}: {a:.4g}  (+{hi-a:.3g} / -{a-lo:.3g})  "
                     f"[frac {frac_lo:.3f}-{frac_hi:.3f}; dominant variant: {worst}]")
    print("\n".join(l for l in lines if l.startswith("#")))
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
