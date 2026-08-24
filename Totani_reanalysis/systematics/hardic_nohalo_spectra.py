#!/usr/bin/env python3
r"""
Flux tables for the hard-IC "C" runs: injected population fitted with the halo REMOVED.

mcmc/hardic_only/<tag>/ holds the fits in which the NFW halo is
taken out of the template stack and the hard-IC component is left free to absorb
the 20 GeV excess on its own. Those runs deliberately carry no flux table: the
manifest-driven tabulators (figures/plot_global_systematics.py,
systematics/propagate_systematics.py) require an 'nfw' column and raise without
one, so the runs are documented as "read theta directly from the npz"
(mcmc/configs/hardic/hardic_only_*.json).

That is fine for the likelihood numbers hardic_report.py quotes, but it leaves no
way to SEE the recovered spectrum. This script fills that gap by driving the same
make_plots_from_mcmc path that produced every other global table, with the plot
settings taken from figures/configs/systematics/hardic_*.json so the output is on
the same footing as the halo-retained ("B") tables:

    exclude_disk_in_plot=True, roi 60x60, disk_cut 10 deg, binsz 0.125,
    ext_mask = mask_extended_sources.fits, template_labels defaulting to labels
    (i.e. the un-convolved cubes, as the published systematics tables use).

Validated by re-deriving the halo-retained hardic_burst_m table through this same
path and comparing against the published one: agrees to 1.9e-3 fractional on the
hardic column, 6.7e-4 on nfw. (Not bit-identical -- the published tables predate
the current fit outputs.)

It also writes the halo baseline (A run, global_fit_results_fig6) through the
identical path, so the reference curve a figure draws is like-for-like rather
than borrowed from the pixelwise pipeline on a different normalisation.

Usage:
    python systematics/hardic_nohalo_spectra.py
    python systematics/hardic_nohalo_spectra.py --only burst_m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
_REPO = _REANALYSIS.parent
sys.path.insert(0, str(_REPO / "shared"))
sys.path.insert(0, str(_REPO))

TAGS = ["burst_s", "burst_m", "burst_l", "cont", "adv", "cmbonly_burst_m"]
DATA = _REPO / "fermi_data" / "totani" / "processed"
OUT = _REANALYSIS / "figures" / "plots_hardic_only"
BASELINE_DIR = _REANALYSIS / "mcmc" / "fit_results" / "quick_global" / "global_fit_results_fig6"
BASELINE_LABELS = [
    "gas", "iso", "ps", "loopA", "loopB", "ics",
    "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno", "fb_pos", "fb_neg",
]


def _common():
    return dict(
        fig="global",
        counts_path=str(DATA / "counts_ccube_1000to1000000.fits"),
        expo_path=str(DATA / "expcube_1000to1000000.fits"),
        templates_dir=str(DATA / "templates"),
        mcmc_stat="f_ml",
        plot_style="totani",
        ext_mask_path=str(DATA / "templates" / "mask_extended_sources.fits"),
        roi_lon=60.0, roi_lat=60.0, disk_cut=10.0, binsz=0.125,
        exclude_disk_in_plot=True,
    )


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    from totani_helpers.mcmc_plotter import make_plots_from_mcmc

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=TAGS, choices=TAGS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args(argv)

    for tag in args.only:
        cfg_path = _REANALYSIS / "mcmc" / "configs" / "hardic" / f"hardic_only_{tag}.json"
        if not cfg_path.exists():
            raise SystemExit(f"missing config: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        run = _REANALYSIS / "mcmc" / "hardic_only" / tag
        if not any(run.glob("global_fit_k*.npz")):
            raise SystemExit(f"no fit results in {run}")
        make_plots_from_mcmc(mcmc_dir=str(run), outdir=str(args.out / tag),
                             labels=cfg["labels"], **_common())
        print(f"  wrote {args.out / tag}")

    if not args.skip_baseline:
        make_plots_from_mcmc(mcmc_dir=str(BASELINE_DIR),
                             outdir=str(args.out / "_baseline_halo"),
                             labels=BASELINE_LABELS, **_common())
        print(f"  wrote {args.out / '_baseline_halo'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
