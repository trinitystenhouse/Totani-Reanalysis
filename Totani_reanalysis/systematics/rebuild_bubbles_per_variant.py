#!/usr/bin/env python3
r"""
Per-diffuse-variant re-extraction of the Fermi-bubble residual templates.

WHY (the systematic this closes)
--------------------------------
fb_pos / fb_neg are RESIDUAL templates: residual = data - (gas + ICS + ... )
background fit, then the positive/negative bubble morphology is iterated out of
that residual (Totani's Stage-2 procedure). The morphology therefore depends on
the gas+ICS diffuse model used to extract it. Carrying the baseline (GALPROP)
bubble templates unchanged into a fit that uses a *different* diffuse model
(e.g. the ICS/ISRF morphology variants) is inconsistent -- the bubble residual
and the gas/ICS templates no longer correspond to the same background, and the
halo can absorb the mismatch. This driver re-derives fb_pos/fb_neg
self-consistently under each diffuse variant's background, so the systematics
suite tests the diffuse model AND its associated bubble residual together.

WHAT IT DOES
------------
For each GALPROP-family diffuse variant it runs the Stage-2 bubble extractor
(make_templates/build_bubbles_templates.py) with that variant's --components
(the gas/ICS set + fb_flat), and copies the resulting residual bubble counts
templates into the shared templates dir under variant-tagged names:

    mu_fb_pos_<variant>_counts.fits
    mu_fb_neg_<variant>_counts.fits

The matching systematics fit then references those tagged bubble labels instead
of the shared fb_pos/fb_neg (see CONFIG WIRING below).

Notes
-----
* fb_flat (Stage 1) is the geometric bubble region (built from the vertices
  polygons), independent of the diffuse model -> built once, reused. Only the
  Stage-2 pos/neg residual is re-extracted here.
* The bubble templates are NOT PSF-forward-folded in the pixelwise fit
  (large-scale, PSF-insensitive), so no mu_fb_*_<variant>_psf_counts.fits is
  needed.
* GIEM variants are SKIPPED: gll_iem_v07 already contains the Fermi bubbles and
  the Loop-I residual patches, so no separate bubble template is extracted
  (this is what diffuse_giem_only encodes). Re-extracting bubbles on top of
  GIEM would double-count.
* GALPROP IEM grid variants (Ackermann et al. 2012 grid; see the
  grid-implementation plan for the interstellar-emission-model systematic,
  Stage 2) are included below under their iem_<tag> keys, one per published
  grid model.
  Each model supplies its own gas_<tag>/ics_<tag> pair, substituted for the
  baseline gas/ics exactly like the ics_morphology_* and hardic_* entries
  substitute their own diffuse piece -- see the design-decision comments
  above VARIANT_COMPONENTS for which models are included and why.

CONFIG WIRING (after running this)
----------------------------------
In write_global_systematics_configs.py, for each re-extracted variant replace
the bubble labels in its `labels` list:
    "fb_pos", "fb_neg"  ->  "fb_pos_<variant>", "fb_neg_<variant>"
e.g. ics_morphology_flat:
    [..., "ics_flatb", RHO2, "fb_pos_ics_morphology_flat", "fb_neg_ics_morphology_flat"]
then regenerate the configs. (Left as a manual/edit step on purpose -- this
script only produces the templates; it does not touch the configs.)

Usage (from Totani_reanalysis/systematics/)
-------------------------------------------
    python rebuild_bubbles_per_variant.py --dry-run        # print commands only
    python rebuild_bubbles_per_variant.py                  # run all variants
    python rebuild_bubbles_per_variant.py --only ics_morphology_flat
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # Totani_reanalysis/systematics
_REANALYSIS = _HERE.parent                        # Totani_reanalysis
_REPO = _HERE.parents[1]                           # repository root

# Locations (post-reorg defaults; override on the CLI if the layout changes).
BUILDER = _REPO / "Totani_paper_check" / "make_templates" / "build_bubbles_templates.py"
TEMPLATES_DIR = _REPO / "fermi_data" / "totani" / "processed" / "templates"
WORK_DIR = _HERE / "bubble_reextraction"          # per-variant scratch + diagnostics

# The reorg split the builder's dependencies across folders: `totani_helpers`
# now lives under shared/, and `mcmc_helper` under Totani_reanalysis/mcmc/. The
# builder imports both by bare name, so we put their parent dirs on PYTHONPATH
# for the subprocess (no edit to the builder needed; remove once a proper shared
# package + import fix lands).
PYTHONPATH_DIRS = [
    _REPO / "shared",                 # for `from totani_helpers...`
    _REANALYSIS / "mcmc",             # for `from mcmc_helper import ...`
    _REPO / "Totani_paper_check",     # builder's own package root (fallback)
]


def _subprocess_env() -> dict:
    env = dict(os.environ)
    extra = os.pathsep.join(str(p) for p in PYTHONPATH_DIRS)
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env

# The canonical Totani bubble extraction uses the rho^2.5 (GCE) halo in the
# background, NOT rho^2 -- see make_templates/build_templates.sh. The bubble
# residual is a separate procedure with its own background; we keep rho^2.5 to
# reproduce the committed baseline and vary only the diffuse (ICS / Loop I) part.
RHO25 = "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno"

# Stage-2 flags that reproduce the canonical extraction (build_templates.sh).
# --bubbles-split-mode totani fixes the pos/neg morphology at 4.3 GeV (k=2) and
# scales amplitude per bin, so the bubbles cannot absorb the 20 GeV halo.
# --use-mcmc runs the 10deg cell Poisson MCMC background (NNLS sets iso->0).
# These are EXPENSIVE (300k-step MCMC per energy bin per variant); use --quick
# for a cheap validation pass first.
STAGE2_BASE_FLAGS = [
    "--use-mcmc",
    "--force-bubbles-close-residual",
    "--bubbles-split-mode", "totani",
    "--disk-cut-deg", "10",
]

# fb_flat (Stage 1, geometric within the vertices polygons) is reused from the
# committed template -- only the diffuse-dependent pos/neg residual is re-extracted.
# Mask/PSF variants are omitted (they don't change the diffuse model). GIEM
# variants are omitted by design (gll_iem_v07 already contains the bubbles).
VARIANT_COMPONENTS = {
    # reference re-extraction: should reproduce the committed baseline (validation)
    "baseline_rho2":        ["gas", "ics",        "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "ics_morphology_flat":  ["gas", "ics_flatb",  "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "ics_morphology_steep": ["gas", "ics_steepb", "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "diffuse_no_loopI":     ["gas", "ics",        "iso", "ps",                   RHO25, "fb_flat"],
    # Hard-IC injected-CR-population templates: adding a
    # new central diffuse component (hardic_<tag>) changes the Stage-2 background
    # exactly as the ICS-morphology variants above do, so the bubble residual is
    # re-extracted with it included. Only the two morphologies prioritised for
    # re-extraction (burst_m = primary, cont = second) are listed here; add more
    # HARDIC_TAGS entries from write_global_systematics_configs.py only if a
    # later morphology also needs it. Halo stays rho^2.5 in the background (as
    # for every other variant here) -- the science fit uses rho^2 (RHO2) with
    # this residual, exactly matching baseline_rho2's own halo/fit split.
    "hardic_burst_m": ["gas", "ics", "iso", "ps", "loopA", "loopB", RHO25, "hardic_burst_m", "fb_flat"],
    "hardic_cont":     ["gas", "ics", "iso", "ps", "loopA", "loopB", RHO25, "hardic_cont",     "fb_flat"],

    # GALPROP IEM grid (Ackermann et al. 2012, grid-implementation plan for
    # the interstellar-emission-model systematic, Stage 2): 14 published models spanning
    # propagation halo height z_h, cosmic-ray source distribution (SL = Lorimer
    # pulsars, SS = SNR, SY = Yusifov pulsars, SO = OB stars), spin temperature
    # T_S, halo radius R_h and E(B-V) cut. The converter (Stage 1, separate
    # script) preserves the existing gas/ICS split per model: pi0+bremss ->
    # gas_<tag>, combined ICS -> ics_<tag>. Those two are substituted for the
    # baseline gas/ics below, exactly as ics_morphology_* substitutes ics_flatb/
    # ics_steepb and hardic_* adds its own extra component -- iso/ps/loopA/loopB
    # are shared baseline components, not part of the grid axis, so they are
    # left as-is. Halo stays rho^2.5 (RHO25) in the background, as for every
    # other entry in this dict -- the science fit uses rho^2 (RHO2) with this
    # bubble residual, matching baseline_rho2's own halo/fit split. Only fb_flat
    # (geometric, diffuse-independent) is reused unchanged; fb_pos/fb_neg are
    # what get re-extracted per model.
    #
    # DESIGN DECISION 1 -- which models: all 14 published-grid tags, not a
    # prioritised subset. Re-extraction is tens of minutes per model (measured:
    # ~19 min/model for the canonical early-stopped ~60k-of-300k-step run across
    # 13 bins), so all 14 is a few hours end-to-end -- acceptable, and the
    # implementation plan (Stage 2) budgets ~0.5 day for exactly this. These 14
    # tags already span every axis of interest (z_h, CR source
    # population, T_S, R_h, E(B-V) cut); the plan's own risk list (#4) flags
    # going further to the full 128-model grid as scope creep that adds
    # processing time without adding argument. A hand-picked subset would also
    # reopen the "why did you cherry-pick" objection the paper is trying to
    # close for the excluded diffuse_giem variant -- running the whole
    # published grid forecloses that question instead.
    #
    # DESIGN DECISION 2 -- anchor flagging: iem_sl_z6 corresponds to our
    # existing baseline GALDEF (galdef_54_Lorimer_z6kpc_R30kpc_Ts150K_
    # EBV2mag_TT25) run through the same grid converter as every other tag, so
    # its re-extracted fb_pos/fb_neg should come out very close to the
    # committed baseline bubbles -- the same validation role baseline_rho2
    # plays above, but for the grid path specifically. It is NOT given a
    # separate dict key or special-cased in the code here -- it is one of the
    # 14 physical variants and the fit that uses gas_iem_sl_z6/ics_iem_sl_z6 as
    # its diffuse model needs its own re-extracted bubbles like every other
    # variant. Operationally, though, run it FIRST
    # (`--only iem_sl_z6`) and diff its mu_fb_{pos,neg}_iem_sl_z6_counts.fits
    # against the committed mu_fb_{pos,neg}_counts.fits before trusting the
    # other 13: if the anchor does not reproduce the baseline, the bug is in
    # this re-extraction path or the upstream grid conversion, and it applies
    # identically to every other tag. Whether iem_sl_z6 should additionally
    # count as an independent systematic excursion in the propagated band, or
    # only as a validation check, is a write_global_systematics_configs.py /
    # propagate_systematics.py decision -- out of scope for this file, flagged
    # here for whoever wires that up.
    "iem_sl_z6":      ["gas_iem_sl_z6",      "ics_iem_sl_z6",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sl_z4":      ["gas_iem_sl_z4",      "ics_iem_sl_z4",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sl_z8":      ["gas_iem_sl_z8",      "ics_iem_sl_z8",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sl_z10":     ["gas_iem_sl_z10",     "ics_iem_sl_z10",     "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_ss_z6":      ["gas_iem_ss_z6",      "ics_iem_ss_z6",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sy_z6":      ["gas_iem_sy_z6",      "ics_iem_sy_z6",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_so_z6":      ["gas_iem_so_z6",      "ics_iem_so_z6",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_ss_z4":      ["gas_iem_ss_z4",      "ics_iem_ss_z4",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_ss_z10":     ["gas_iem_ss_z10",     "ics_iem_ss_z10",     "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sy_z4":      ["gas_iem_sy_z4",      "ics_iem_sy_z4",      "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sy_z10":     ["gas_iem_sy_z10",     "ics_iem_sy_z10",     "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sl_z6_tinf": ["gas_iem_sl_z6_tinf", "ics_iem_sl_z6_tinf", "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sl_z6_c5":   ["gas_iem_sl_z6_c5",   "ics_iem_sl_z6_c5",   "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
    "iem_sl_z6_r20":  ["gas_iem_sl_z6_r20",  "ics_iem_sl_z6_r20",  "iso", "ps", "loopA", "loopB", RHO25, "fb_flat"],
}


def build_cmd(variant: str, components: list[str], templates_dir: Path, out_dir: Path,
              nsteps: int, nwalkers: int, burn: int, thin: int) -> list[str]:
    return [
        sys.executable, str(BUILDER),
        "--templates-dir", str(templates_dir),
        "--components", *components,
        *STAGE2_BASE_FLAGS,
        "--mcmc-nwalkers", str(nwalkers),
        "--mcmc-nsteps", str(nsteps),
        "--mcmc-burn", str(burn),
        "--mcmc-thin", str(thin),
        "--out-dir", str(out_dir),
        # CRITICAL: redirect the Stage-2 background MCMC summaries to a per-variant
        # scratch dir. The builder DEFAULTS this to Totani_paper_check/mcmc/
        # mcmc_results_fig2_3 -- the committed disk-included Fig.2/3 replication
        # result -- and would otherwise overwrite it with each variant's background fit.
        "--mcmc-results-dir", str(out_dir / "mcmc_fig2_3"),
        "--out-plots", str(out_dir / "plots"),
        # Stage 2 (pos/neg) is the default mode; do NOT pass --build-flat-only.
        # pos/neg counts are written whenever build_flat_only is False, so no
        # --save-counts-templates is needed (it is a no-op in the builder).
        # fb_flat is geometric (uniform-in-vertices) and diffuse-INDEPENDENT, so
        # the committed fb_flat in --templates-dir is reused unchanged: the ONLY
        # difference from the canonical Totani build is the diffuse --components.
    ]


def _missing_component_templates(components: list[str], templates_dir: Path) -> list[str]:
    """Component labels whose mu_<label>_counts.fits is not (yet) present.

    Guard used before spawning the (expensive) Stage-2 subprocess: the builder
    itself raises FileNotFoundError for a missing component
    (totani_helpers.fit_utils.load_mu_templates_from_fits), but only after the
    subprocess has already started, argparsed and begun reading FITS -- and if
    this runs unattended over several variants, a missing template several
    models into the batch would otherwise surface as an obscure traceback deep
    in the child process rather than a clear, checkable skip here. Several of
    the gas_iem_*/ics_iem_* pairs are still landing (module docstring; grid
    converter is a separate, concurrently-developed script), so this matters
    now, not just hypothetically.
    """
    return [lab for lab in components if not (templates_dir / f"mu_{lab}_counts.fits").exists()]


def run_variant(variant: str, components: list[str], templates_dir: Path,
                dry_run: bool, nsteps: int, nwalkers: int, burn: int, thin: int) -> None:
    out_dir = WORK_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(variant, components, templates_dir, out_dir, nsteps, nwalkers, burn, thin)
    print(f"\n[{variant}] components = {components}")
    print("  $ " + " ".join(cmd))
    if dry_run:
        return
    missing = _missing_component_templates(components, templates_dir)
    if missing:
        missing_files = [f"mu_{lab}_counts.fits" for lab in missing]
        print(f"[SKIP] {variant}: missing component template(s) {missing_files} "
              f"in {templates_dir} -- not re-extracting bubbles for this variant yet.")
        return
    subprocess.run(cmd, check=True, env=_subprocess_env())
    from astropy.io import fits
    import numpy as np
    # copy the Stage-2 outputs into the shared templates dir under tagged names
    for sign in ("pos", "neg"):
        src = out_dir / f"mu_fb_{sign}_counts.fits"
        if not src.exists():
            raise FileNotFoundError(f"expected {src} from the builder; check Stage-2 ran")
        tot = float(np.nansum(np.asarray(fits.getdata(src), float)))
        if tot == 0.0:
            raise RuntimeError(
                f"{src} is all-zero -- Stage-2 produced no bubble residual. "
                f"Check the --use-mcmc / --bubbles-split-mode totani flags and the "
                f"component templates exist. NOT copying garbage into the templates dir.")
        dst = templates_dir / f"mu_fb_{sign}_{variant}_counts.fits"
        shutil.copy2(src, dst)
        print(f"  -> {dst.name}  (sum={tot:.3e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(VARIANT_COMPONENTS),
                    help="re-extract only these variants (default: all).")
    ap.add_argument("--templates-dir", type=Path, default=TEMPLATES_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print commands without running.")
    # canonical MCMC settings (build_templates.sh); --quick for a cheap validation pass.
    ap.add_argument("--mcmc-nsteps", type=int, default=300000)
    ap.add_argument("--mcmc-nwalkers", type=int, default=64)
    ap.add_argument("--mcmc-burn", type=int, default=50000)
    ap.add_argument("--mcmc-thin", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="cheap settings (nsteps=5000,burn=1000) to validate the pipeline "
                         "before the full canonical run.")
    args = ap.parse_args(argv)

    nsteps, burn = (5000, 1000) if args.quick else (args.mcmc_nsteps, args.mcmc_burn)

    if not BUILDER.exists():
        print(f"[warn] builder not found at {BUILDER} -- fix --templates-dir / paths "
              f"(repo is mid-reorg).", file=sys.stderr)

    variants = args.only or sorted(VARIANT_COMPONENTS)
    for v in variants:
        run_variant(v, VARIANT_COMPONENTS[v], args.templates_dir, args.dry_run,
                    nsteps, args.mcmc_nwalkers, burn, args.mcmc_thin)

    if not args.dry_run:
        print("\nDone. Next: in write_global_systematics_configs.py, point each variant's "
              "bubble labels at fb_pos_<variant>/fb_neg_<variant>, regenerate configs, "
              "re-run those variant fits, then propagate_systematics.py.")


if __name__ == "__main__":
    main()
