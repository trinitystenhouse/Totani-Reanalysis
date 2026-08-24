#!/usr/bin/env python3
r"""
run_hardic_bubble_reextraction.py
==================================
Driver for the Fermi-bubble re-extraction self-consistency check: re-extracts
the Fermi-bubble residual templates (fb_pos/fb_neg) under a Stage-2
background that includes the hard inverse-Compton injected-CR-population
template, for the two prioritised morphologies -- ``burst_m`` (primary
morphology) and ``cont`` (second) -- then regenerates the systematics configs
so a comparator fit picks up the re-extracted bubbles in place of the
baseline ones.

WHY (see rebuild_bubbles_per_variant.py:4-17 for the underlying rationale)
---------------------------------------------------------------------------
fb_pos/fb_neg are RESIDUAL templates: residual = data - (gas + ICS + ...)
background fit, then split by sign (Totani's Stage-2 procedure). Adding
hardic_<tag> to the Stage-2 background changes that background exactly as the
ICS-morphology variants already in VARIANT_COMPONENTS do (rebuild_bubbles_
per_variant.py:120-126, both re-extracted). The builder accepts arbitrary
--components entries as long as mu_{name}_counts.fits exists (verified at
build_bubbles_templates.py:989-1001 and, live, by this repo's own hardic
template builds -- mu_hardic_burst_m_counts.fits and mu_hardic_cont_counts.fits
already exist under fermi_data/totani/processed/templates/, shape (13,960,960),
all-finite, built for the hardic_report.py fits already committed today).

WHAT THIS SCRIPT DOES (three steps)
------------------------------------
1. RE-EXTRACT (expensive -- NOT run unless you pass --run-reextraction):
   calls ``rebuild_bubbles_per_variant.py --only hardic_burst_m hardic_cont``,
   which for each tag runs build_bubbles_templates.py's Stage-2 cellwise
   Poisson MCMC background fit (--use-mcmc, --disk-cut-deg 10,
   --bubbles-split-mode totani) over all 13 energy bins, then copies the
   resulting mu_fb_{pos,neg}_counts.fits into the shared templates dir as
   mu_fb_{pos,neg}_hardic_<tag>_counts.fits (rebuild_bubbles_per_variant.py's
   own copy step, with a built-in all-zero guard that refuses to copy a dead
   fit -- see its run_variant()).

2. REGENERATE CONFIGS (cheap -- JSON only, always safe to re-run): calls
   mcmc/write_global_systematics_configs.py, which (idempotently) rewrites
   every systematics config, including the two
   ``hardic_<tag>_bubbles_reextracted`` variants defined in
   write_global_systematics_configs.py::cr_injection_bubbles_reextracted_
   variants(). Those configs are identical to the existing hardic_<tag>.json
   except fb_pos/fb_neg -> fb_pos_hardic_<tag>/fb_neg_hardic_<tag>, and they
   write to a SEPARATE global_dir (.../global_systematics/hardic_<tag>_
   bubbles_reextracted) so the already-completed hardic_<tag> baseline-bubble
   fits are never overwritten -- the Stage-1-vs-Stage-2 delta can only be
   read off if both are kept.

3. REFIT (NOT run by this script -- printed as the exact commands to run):
   the deterministic global fit itself is cheap (seconds per bin, verified
   from logs/logs_global_systematics/*.log "Total wall time" lines), so it
   is left for you to launch by hand once step 1's templates exist on disk.

EXPECTED IMPACT ON THE HARD-IC RESULT -- read before running
-------------------------------------------------------------------
For all five physical hard-IC morphologies (including burst_m and cont), the
hard-IC coefficient is pinned at exactly zero in every signal-band bin
(12.29 / 20.76 / 35.06 GeV). It is at its zero bound because the fit would
prefer a *negative* injected population there and the template is
sign-constrained. Re-extracting the bubbles changes the *background* the
residual is taken against; it has no mechanism to un-pin a coefficient
sitting on its bound because the data disfavour more of that component in
that band, not because of a bubble-shape mismatch. The honest expectation is
little to no change in the reported halo-survival numbers for burst_m/cont.
This closes out a self-consistency point that follows from fb_pos/fb_neg
being residual-derived (see rebuild_bubbles_per_variant.py's own docstring);
it is not expected to be load-bearing for the hard-IC conclusion, and the
run should not be mistaken for a critical-path result.

RUNTIME -- read before running --run-reextraction without --quick
----------------------------------------------------------------------
rebuild_bubbles_per_variant.py's own comment (lines ~109-114) describes the
canonical settings (300000 steps, 64 walkers, cellwise Poisson MCMC, 13
bins) as "EXPENSIVE ... order days per variant". That figure assumes
early-stop never fires. In practice it
does: the committed baseline Fermi-bubble Stage-2 result at
Totani_paper_check/mcmc/mcmc_results_fig2_3/ was built with these EXACT
canonical flags and an IDENTICAL 8-component background (gas, ics, iso, ps,
loopA, loopB, the rho^2.5 halo, fb_flat -- see Totani_reanalysis/logs/
build_templates_retry.log) and its chain arrays show early-stop terminated
each bin at ~60000 raw steps (20% of the 300000 cap; mcmc_results_k00.npz
chain shape (6000,64,8) at thin_save=10). File mtimes on the 13
mcmc_results_k*.npz files span 02:03:33-02:20:52 -- roughly 85-100 seconds per
bin, ~19 minutes total for that 13-bin, 8-component run.

hardic_<tag> adds a 9th free component (one more parameter for the cellwise
fit), and the ROI here is disk-EXCLUDED (--disk-cut-deg 10, fewer 10-degree
cells than the disk-included reference run above), so the two effects should
roughly offset. A live smoke test in this session (nwalkers=64, only
100 raw steps, early-stop and the autocorr requirement both disabled to force
a fixed short run) exercised the FULL background list with hardic_burst_m
included -- all 13 bins, Stage-2 residual split, output write -- and completed
in 22.8s wall (isolated scratch --out-dir, not the real templates dir).

Best honest estimate for ONE tag at canonical settings: same order as the
verified ~19-minute reference run, i.e. tens of minutes, NOT days -- but this
is an extrapolation from a different (disk-included, 8-component) run, not a
completed hardic timing, so treat it as a estimate, not a guarantee.
--quick (nsteps=5000, burn=1000, no early-stop headroom to trigger) should be
correspondingly a few minutes per tag and is the recommended first run.

USAGE
-----
    # default (no flags): prints the two re-extraction commands and the
    # expected-impact note above; touches NOTHING on disk.
    python run_hardic_bubble_reextraction.py

    # cheap pipeline-validation pass (--quick: nsteps=5000, burn=1000) --
    # still not the canonical result, just confirms the pipeline runs
    # end-to-end for both tags and writes real (if noisy) tagged templates.
    python run_hardic_bubble_reextraction.py --run-reextraction --quick

    # the real thing (canonical settings; see RUNTIME above)
    python run_hardic_bubble_reextraction.py --run-reextraction

    # re-extraction already done by hand -- just (re)wire the configs
    python run_hardic_bubble_reextraction.py --configs-only

Either way this script never calls mcmc/run_global_from_config.py itself --
it only ever prints the refit commands for you to run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # Totani_reanalysis/systematics
_REANALYSIS = _HERE.parent                        # Totani_reanalysis

REBUILD_BUBBLES = _HERE / "rebuild_bubbles_per_variant.py"
WRITE_CONFIGS = _REANALYSIS / "mcmc" / "write_global_systematics_configs.py"

# Must match the keys added to VARIANT_COMPONENTS in rebuild_bubbles_per_variant.py
# and the tags consumed by cr_injection_bubbles_reextracted_variants() in
# write_global_systematics_configs.py.
TAGS = ["hardic_burst_m", "hardic_cont"]

EXPECTED_IMPACT_NOTE = (
    "Expected impact: LITTLE TO NONE. The hard-IC coefficient is pinned at "
    "exactly zero in every signal-band bin for burst_m/cont because the fit "
    "would prefer a *negative* injected population there and the template is "
    "sign-constrained. Re-extracting the bubbles changes the background the "
    "residual is taken against; it cannot un-pin a coefficient sitting on its "
    "zero bound for that reason. This run closes out a self-consistency point "
    "that follows from fb_pos/fb_neg being residual-derived, so re-extraction "
    "is required 'in principle' when the background changes; it is not "
    "expected to change the hard-IC conclusion."
)


def _run(cmd: list[str], dry_run: bool) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    sys.stdout.flush()  # keep our own prints ahead of the child's inherited-fd output
    if dry_run:
        return
    subprocess.run([str(c) for c in cmd], check=True)


def step1_reextract(args) -> None:
    print(f"\n[1/3] Re-extract bubbles for: {', '.join(t.replace('hardic_', '') for t in TAGS)}")
    if not args.run_reextraction:
        print("      (not executed -- pass --run-reextraction to actually run this; "
              "printing the command rebuild_bubbles_per_variant.py would run instead.)")
    cmd = [sys.executable, str(REBUILD_BUBBLES), "--only", *TAGS]
    if args.quick:
        cmd.append("--quick")
    if args.templates_dir:
        cmd += ["--templates-dir", args.templates_dir]
    for flag, val in [
        ("--mcmc-nwalkers", args.mcmc_nwalkers),
        ("--mcmc-nsteps", args.mcmc_nsteps),
        ("--mcmc-burn", args.mcmc_burn),
        ("--mcmc-thin", args.mcmc_thin),
    ]:
        if val is not None:
            cmd += [flag, str(val)]
    # Always show what rebuild_bubbles_per_variant.py itself would run (its own
    # --dry-run prints the full build_bubbles_templates.py invocation per tag).
    _run(cmd + (["--dry-run"] if not args.run_reextraction else []), dry_run=False)


def step2_regenerate_configs(args) -> None:
    print("\n[2/3] Regenerate systematics configs (cheap, JSON only, idempotent)")
    if args.skip_configs:
        print("      (skipped: --skip-configs)")
        return
    _run([sys.executable, str(WRITE_CONFIGS)], dry_run=args.dry_run_configs)


def step3_print_refit_commands(args) -> None:
    print("\n[3/3] Refit commands (NOT run by this script -- run these yourself once "
          "step 1's templates exist on disk):")
    print("      cd \"" + str(_REANALYSIS) + "\"")
    for tag in TAGS:
        short = tag.replace("hardic_", "")
        print(f"      python mcmc/run_global_from_config.py "
              f"--config mcmc/configs/systematics/hardic_{short}_bubbles_reextracted.json")
    print("      # compare against the already-committed baseline-bubble fits:")
    for tag in TAGS:
        short = tag.replace("hardic_", "")
        print(f"      #   mcmc/global_systematics/hardic_{short}/            (Stage 1, baseline bubbles)")
        print(f"      #   mcmc/global_systematics/hardic_{short}_bubbles_reextracted/  (Stage 2, re-extracted)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-reextraction", action="store_true",
                     help="Actually execute rebuild_bubbles_per_variant.py for burst_m/cont "
                          "(the expensive step -- see RUNTIME in this script's docstring). "
                          "Without this flag, step 1 only prints the command it would run.")
    ap.add_argument("--quick", action="store_true",
                     help="Forwarded to rebuild_bubbles_per_variant.py --quick "
                          "(nsteps=5000, burn=1000): cheap pipeline-validation pass, "
                          "not the canonical result. Recommended before a full run.")
    ap.add_argument("--mcmc-nwalkers", type=int, default=None, help="Forwarded override.")
    ap.add_argument("--mcmc-nsteps", type=int, default=None, help="Forwarded override.")
    ap.add_argument("--mcmc-burn", type=int, default=None, help="Forwarded override.")
    ap.add_argument("--mcmc-thin", type=int, default=None, help="Forwarded override.")
    ap.add_argument("--templates-dir", default=None, help="Forwarded override.")
    ap.add_argument("--configs-only", action="store_true",
                     help="Skip step 1 entirely (assume re-extraction was already run) "
                          "and just regenerate the systematics configs.")
    ap.add_argument("--skip-configs", action="store_true",
                     help="Skip step 2 (config regeneration).")
    args = ap.parse_args(argv)

    # --dry-run-configs is internal bookkeeping, not exposed on the CLI: config
    # regeneration is cheap and safe, so it always actually runs unless
    # --skip-configs is given (there is nothing to "dry run" about writing JSON).
    args.dry_run_configs = False

    print(EXPECTED_IMPACT_NOTE)

    if not args.configs_only:
        step1_reextract(args)
    else:
        print("\n[1/3] Skipped (--configs-only): assuming re-extraction was already run.")

    step2_regenerate_configs(args)
    step3_print_refit_commands(args)

    if not args.run_reextraction and not args.configs_only:
        print("\nNothing was executed (dry preview only). Re-run with --run-reextraction "
              "[--quick] to actually perform the re-extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
