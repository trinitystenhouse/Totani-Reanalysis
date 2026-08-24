#!/usr/bin/env python3
"""Audit dwarf-spheroidal limit tables and citation metadata used by plots."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
sys.path.insert(0, str(REPO / "Totani_reanalysis"))
sys.path.insert(0, str(REPO / "Totani_reanalysis" / "core"))

from dsph_limits import (  # noqa: E402
    DSph_LIMIT_SOURCES,
    available_dsph_channels,
    dsph_limit_source,
    load_dsph_limit_table,
)


CITATIONS = {
    "hoof": (
        "Hoof, Geringer-Sameth & Trotta, JCAP 02 (2020) 012, "
        "arXiv:1812.06986; limit files from Zenodo record 2612268."
    ),
    "mcdaniel": (
        "McDaniel et al., Phys. Rev. D 109 (2024) 063024, "
        "arXiv:2311.04982; stacked dSph limit tables in dSph_upper_limits/."
    ),
    "boddy_swave": (
        "Boddy et al., Phys. Rev. D 102 (2020) 023029, "
        "arXiv:1909.13197; effective-J dSph limits, n=0."
    ),
    "boddy_pwave": (
        "Boddy et al., Phys. Rev. D 102 (2020) 023029, "
        "arXiv:1909.13197; effective-J dSph limits, p-wave n=2."
    ),
    "boddy_dwave": (
        "Boddy et al., Phys. Rev. D 102 (2020) 023029, "
        "arXiv:1909.13197; effective-J dSph limits, d-wave n=4."
    ),
    "boddy_somm": (
        "Boddy et al., Phys. Rev. D 102 (2020) 023029, "
        "arXiv:1909.13197; effective-J dSph limits, Coulomb/Sommerfeld n=-1."
    ),
}


def main() -> int:
    outdir = PROJECT / "results" / "dsph-limit-audit"
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Dwarf-spheroidal limit table audit",
        f"REPO_PATH={os.environ.get('REPO_PATH', str(REPO))}",
        "",
        f"{'source':15s} {'channel':8s} {'rows':>6s} {'m_min':>10s} {'m_max':>10s} {'sv_min':>12s} {'sv_max':>12s}",
        "-" * 92,
    ]
    for source in DSph_LIMIT_SOURCES:
        channels = available_dsph_channels(source)
        if not channels:
            lines.append(f"{source:15s} {'-':8s} {'MISSING':>6s}")
            continue
        for channel in channels:
            try:
                mass, sigmav = load_dsph_limit_table(channel, source)
                lines.append(
                    f"{source:15s} {channel:8s} {mass.size:6d} "
                    f"{np.nanmin(mass):10.4g} {np.nanmax(mass):10.4g} "
                    f"{np.nanmin(sigmav):12.4e} {np.nanmax(sigmav):12.4e}"
                )
                lines.append(f"  label: {dsph_limit_source(channel, source=source)}")
                lines.append(f"  citation: {CITATIONS.get(source, 'VERIFY CITATION')}")
            except Exception as exc:
                lines.append(f"{source:15s} {channel:8s} ERROR: {exc}")
    path = outdir / "dsph_limit_audit.txt"
    path.write_text("\n".join(lines) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
