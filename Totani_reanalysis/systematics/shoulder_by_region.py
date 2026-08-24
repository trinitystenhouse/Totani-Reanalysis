#!/usr/bin/env python3
r"""
Read off the low-energy shoulder per latitude band and per inner cut.

Run after the configs from mcmc/write_shoulder_test_configs.py have been fitted
and tabulated. Reports, for each parent model and each region:

  shoulder      1.5-4.4 GeV halo flux divided by the 12-36 GeV signal band flux
  significance  the same low-energy flux over its own combined error
  band flux     the 12-36 GeV flux itself, so that a falling shoulder is not
                confused with a falling halo

The reading
-----------
  Shoulder confined to |b| = 10-20 deg and small at 20-60 deg
      inner-Galaxy contamination, consistent with a GCE tail. The affected
      variants can be excluded from the foreground leg on the same
      double-counting grounds already used for diffuse_giem, or the mass can be
      quoted from the signal band alone.

  Shoulder roughly flat across all three bands
      not the GCE, which does not reach 40-60 deg. A genuine mid-latitude
      diffuse deficit absorbed by the halo template. Report it and keep the band.

  Signal-band flux falling with the shoulder
      the whole component is diffuse-model driven. Nothing so far suggests this,
      since the band-summed amplitude is stable to five per cent across the grid,
      but this script would show it.

The Lorimer control is the discriminator. If sl_z6 and so_z6 have the same
latitude profile, the effect belongs to the ROI. If only so_z6 shows it, the
effect belongs to the source distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))

from shared.paths import REANALYSIS, REANALYSIS_RESULTS  # noqa: E402

ROOTS = [REANALYSIS / "figures" / "systematics" / "plots_global_systematics",
         REANALYSIS / "figures" / "plots_global_systematics"]

PARENTS = ["baseline_rho2",
           "iem_sl_z6_bubbles_reextracted",
           "iem_so_z6_bubbles_reextracted"]

REGIONS = [("full ROI 10-60", ""),
           ("band 10-20", "_lat10_20"),
           ("band 20-40", "_lat20_40"),
           ("band 40-60", "_lat40_60"),
           ("cut 15-60", "_cut15"),
           ("cut 20-60", "_cut20")]


def read(name):
    for r in ROOTS:
        p = Path(r) / name / "mcmc_global_f_ml.txt"
        if p.exists():
            hdr = open(p).readline().lstrip("#").split()
            i = next(j for j, h in enumerate(hdr)
                     if "nfw" in h.lower() and not h.endswith(("_err_lo", "_err_hi")))
            a = np.loadtxt(p)
            return a[:, 0], a[:, i], 0.5 * (a[:, i + 1] + a[:, i + 2])
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=REANALYSIS_RESULTS / "global-systematics" / "shoulder_by_region.txt")
    a = p.parse_args(argv)

    lines = []
    for parent in PARENTS:
        lines.append(f"\n=== {parent} ===")
        lines.append(f"{'region':16s} {'shoulder':>9s} {'low-E sig':>10s} {'band flux':>11s}")
        for label, suf in REGIONS:
            got = read(parent + suf)
            if got is None:
                lines.append(f"{label:16s} {'not fitted yet':>32s}")
                continue
            E, v, e = got
            ml = (E >= 1.5) & (E <= 4.4)
            mb = (E >= 12.0) & (E <= 36.0)
            lo, bd = v[ml].sum(), v[mb].sum()
            elo = np.sqrt((e[ml] ** 2).sum())
            sh = lo / bd if bd else np.nan
            sig = lo / elo if elo else np.nan
            lines.append(f"{label:16s} {sh:9.2f} {sig:10.1f} {bd:11.3e}")

    out = "\n".join(lines)
    print(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("# Low-energy shoulder by region\n" + out + "\n")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
