#!/usr/bin/env python3
r"""
Write the configs for the low-energy shoulder diagnostic.

The question
-----------
Across the fourteen-model interstellar-emission grid the halo spectral peak stays
in the 20.76 GeV bin, but the flux below about 5 GeV grows monotonically with how
centrally concentrated the cosmic-ray source distribution is. Integrating 1.5 to
4.4 GeV over the 12 to 36 GeV signal band gives 0.05 for baseline and Lorimer, and
0.57 at 32 sigma for OB stars. Refitting with the low-energy bins excluded
collapses the grid mass spread from x2.10 to x1.22, so that shoulder is what
drives the widened upper edge of the dSph tension band.

What remains unknown is WHERE the shoulder lives. If it is confined to the inner
edge of the ROI it is inner-Galaxy contamination consistent with a GCE tail, and
the affected variants can be treated accordingly. If it is spread evenly to 60
degrees it cannot be the GCE, which does not reach that far, and it is instead a
genuine mid-latitude diffuse deficit the halo template is absorbing.

The tests
---------
Both are config-only, since run_global_from_config.py already passes --roi-lat
and --disk-cut straight from the config.

  Test A, latitude bands   disk_cut/roi_lat = 10/20, 20/40, 40/60
  Test B, raised inner cut disk_cut/roi_lat = 15/60, 20/60

Each is run for three parents chosen to bracket the effect:

  baseline_rho2                    shoulder 0.05, the reference
  iem_sl_z6_bubbles_reextracted    shoulder 0.05, a grid model with NO shoulder
  iem_so_z6_bubbles_reextracted    shoulder 0.57, the worst case

The Lorimer control matters. If the shoulder's latitude profile differs between
sl_z6 and so_z6, the effect is specific to the source distribution. If both behave
the same, it is a property of the ROI rather than of the diffuse model.

Groups are named lat_band and inner_cut, neither of which is in FOREGROUND_GROUPS
in propagate_systematics.py, so these diagnostics cannot leak into the published
systematic band.

Usage
-----
    python3 mcmc/write_shoulder_test_configs.py            # write configs
    python3 mcmc/write_shoulder_test_configs.py --commands # print run commands only
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "configs" / "systematics"
FIGCFG = HERE.parent / "figures" / "configs" / "systematics"

PARENTS_DEFAULT = ["baseline_rho2",
                   "iem_sl_z6_bubbles_reextracted",
                   "iem_so_z6_bubbles_reextracted"]

# The full re-extracted grid, for --parents all. Fourteen models plus the
# baseline, five regions each, so 75 deterministic fits.
PARENTS_ALL = ["baseline_rho2"] + [
    f"iem_{t}_bubbles_reextracted" for t in
    ("sl_z4", "sl_z6", "sl_z8", "sl_z10", "sl_z6_c5", "sl_z6_r20", "sl_z6_tinf",
     "ss_z4", "ss_z6", "ss_z10", "sy_z4", "sy_z6", "sy_z10", "so_z6")]

# (suffix, disk_cut, roi_lat, group, description)
CASES = [
    ("lat10_20", 10.0, 20.0, "lat_band",
     "restricted to |b| = 10-20 deg, the inner edge of the ROI"),
    ("lat20_40", 20.0, 40.0, "lat_band",
     "restricted to |b| = 20-40 deg"),
    ("lat40_60", 40.0, 60.0, "lat_band",
     "restricted to |b| = 40-60 deg, the outer ROI where no GCE tail should reach"),
    ("cut15",    15.0, 60.0, "inner_cut",
     "inner latitude cut raised from 10 to 15 deg, full outer ROI"),
    ("cut20",    20.0, 60.0, "inner_cut",
     "inner latitude cut raised from 10 to 20 deg, full outer ROI"),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commands", action="store_true",
                   help="print the run commands and exit, writing nothing")
    p.add_argument("--out-manifest", type=Path,
                   default=CFG / "manifest_shoulder_test.json")
    p.add_argument("--parents", nargs="+", default=None,
                   help="parent variants to split. Use 'all' for the full "
                        "re-extracted grid, 75 fits. Default is the three-parent "
                        "bracket: baseline, Lorimer z6, OB-star z6.")
    a = p.parse_args(argv)

    parents = PARENTS_DEFAULT
    if a.parents:
        parents = PARENTS_ALL if a.parents == ["all"] else a.parents

    made = []
    for parent in parents:
        src = CFG / f"{parent}.json"
        psrc = FIGCFG / f"{parent}.json"
        if not src.exists():
            print(f"[skip] missing parent config: {src}")
            continue
        base = json.loads(src.read_text())
        pbase = json.loads(psrc.read_text()) if psrc.exists() else None

        for suf, dcut, rlat, grp, why in CASES:
            name = f"{parent}_{suf}"
            c = copy.deepcopy(base)
            c["name"] = f"systematics_{name}"
            c["systematics_group"] = grp
            c["description"] = (f"Shoulder diagnostic. Parent {parent}, {why}. "
                                "Not part of the foreground systematic band.")
            c["global_dir"] = f"../../global_systematics/{name}"
            c["mcmc_dir"] = f"../../global_systematics/{name}"
            c["plot_configs"] = [f"../../../figures/configs/systematics/{name}.json"]
            c["disk_cut"] = dcut
            c["roi_lat"] = rlat
            c["include_disk"] = False

            if not a.commands:
                (CFG / f"{name}.json").write_text(json.dumps(c, indent=2) + "\n")
                if pbase is not None:
                    pc = copy.deepcopy(pbase)
                    pc["mcmc_dir"] = f"../../../mcmc/global_systematics/{name}"
                    pc["outdir"] = f"../../plots_global_systematics/{name}"
                    pc["disk_cut"] = dcut
                    pc["roi_lat"] = rlat
                    (FIGCFG / f"{name}.json").write_text(json.dumps(pc, indent=2) + "\n")
            made.append(dict(name=name, group=grp, parent=parent,
                             disk_cut=dcut, roi_lat=rlat,
                             config=f"mcmc/configs/systematics/{name}.json",
                             plot_config=f"figures/configs/systematics/{name}.json",
                             global_dir=f"../../global_systematics/{name}"))

    if not a.commands:
        a.out_manifest.write_text(json.dumps({"variants": made}, indent=2) + "\n")
        print(f"wrote {len(made)} fit configs to {CFG}")
        print(f"wrote {len(made)} plot configs to {FIGCFG}")
        print(f"wrote manifest {a.out_manifest}")

    print("\n# ---- run the fits, from the repo root ----")
    for m in made:
        print(f"python3 mcmc/run_global_from_config.py --config {m['config']}")
    print("\n# NOTE: run_global_from_config.py is the DETERMINISTIC global path. It writes")
    print("# global_fit_k??.npz and, via the config's plot_configs, the coefficient table")
    print("# figures/plots_global_systematics/<name>/mcmc_global_f_ml.txt that everything")
    print("# downstream reads. run_mcmc_from_config.py is the pixelwise MCMC path and")
    print("# writes mcmc_results_k??.npz instead, which the tabulation step cannot use.")
    print("# The pipeline README names the wrong script here.")
    print("\n# ---- read off the shoulder per band ----")
    print("python3 systematics/shoulder_by_region.py")


if __name__ == "__main__":
    main()
