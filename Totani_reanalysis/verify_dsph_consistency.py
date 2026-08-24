#!/usr/bin/env python3
r"""
Consistency audit: prove that each velocity law's dSph limit is built from THAT
law's dwarf data, and is compared against THAT law's halo signal.

Data flow per wave n (sigma v = a_n v^{2n}):
  * absolute Fermi sensitivity (velocity-INDEPENDENT photon-flux limit) comes from
    the s-wave dwarf analysis: sigma_v0_UL  [Hoof, the --s-source anchor].
  * velocity weighting comes from THAT WAVE's data:
        dwarf side: u_dwarf_n = j_n / j_s   (Boddy 2019 p/d/Sommerfeld effective J)
        halo  side: u_halo_n  = <m_n>_halo  (Jeans, compute_u_halo.py)
  * halo-comparable limit:   L_n(m) = sigma_v0_UL(m) * u_halo_n / u_dwarf_n
  * scan tension:            t_n = <sv>_halo,eff / L_n
                                 = (a_n u_halo_n) / (sigma_v0_UL u_halo_n / u_dwarf_n)
                                 = a_n u_dwarf_n / sigma_v0_UL
                                 = (a_n j_n) / (sigma_v0_UL j_s)   <-- predicted dwarf
                                   p-wave signal over the p-wave dwarf limit. QED.

This script asserts the table relations numerically so a mismatch can't slip in:
  boddy_swave == hoof                          (s-wave limit on s-wave data)
  boddy_pwave == hoof * u_halo_p / u_dwarf_p   (p-wave limit on p-wave data)
  boddy_dwave == hoof * u_halo_d / u_dwarf_d
  boddy_somm  == hoof * u_halo_s / u_dwarf_s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from core.dsph_limits_multi import dsph_upper_limit  # noqa: E402

BODDY_CSV = _HERE / "boddy_moments_0d5.csv"
HALO_CSV = _HERE / "halo_moments_bary.csv"
CHANNELS = ("bb", "WW", "tautau")
MASSES = (150.0, 300.0, 579.0, 768.0, 1500.0)


def _read_csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return {n: float(d[n]) for n in d.dtype.names}


def main():
    bo = _read_csv(BODDY_CSV)
    ha = _read_csv(HALO_CSV)
    expected = {
        "boddy_swave": 1.0,
        "boddy_pwave": ha["u_halo_pwave"] / bo["u_dwarf_pwave"],
        "boddy_dwave": ha["u_halo_dwave"] / bo["u_dwarf_dwave"],
        "boddy_somm": ha["u_halo_somm"] / bo["u_dwarf_somm"],
    }
    print("Expected halo-comparable factors (relative to the s-wave anchor):")
    for k, v in expected.items():
        print(f"  {k:12s}: {v:.4e}")
    print(f"\nProvenance:")
    print(f"  s-wave anchor sigma_v0_UL : Hoof 2020 (velocity-independent photon-flux limit)")
    print(f"  p-wave weighting           : Boddy j_p/j_s = {bo['u_dwarf_pwave']:.3e}  &  "
          f"halo <v^2> = {ha['u_halo_pwave']:.3e}")
    print(f"  d-wave weighting           : Boddy j_d/j_s = {bo['u_dwarf_dwave']:.3e}  &  "
          f"halo <v^4> = {ha['u_halo_dwave']:.3e}")
    print(f"  Sommerfeld weighting       : Boddy j_som/j_s = {bo['u_dwarf_somm']:.3e}  &  "
          f"halo <c/v> = {ha['u_halo_somm']:.3e}")

    print("\nNumerical check (table / hoof should equal the expected factor):")
    ok = True
    for ch in CHANNELS:
        for m in MASSES:
            s = dsph_upper_limit(m, ch, "hoof")
            if not np.isfinite(s):
                continue
            for src, fac in expected.items():
                got = dsph_upper_limit(m, ch, src) / s
                if not np.isclose(got, fac, rtol=2e-3):
                    print(f"  MISMATCH {src} {ch} m={m}: got {got:.4e} expected {fac:.4e}")
                    ok = False
    print("  all source/channel/mass relations hold." if ok else "  *** MISMATCH FOUND ***")

    # The cross-check that matters most: s-wave source must NOT equal p-wave source.
    # Use RATIO-based comparisons -- absolute tolerances are meaningless at ~1e-25.
    s_sw = dsph_upper_limit(768, "bb", "boddy_swave")
    s_pw = dsph_upper_limit(768, "bb", "boddy_pwave")
    s_hoof = dsph_upper_limit(768, "bb", "hoof")
    assert s_pw / s_sw > 10.0, "s-wave and p-wave tables are too similar -- possible mislabel!"
    assert abs(s_sw / s_hoof - 1.0) < 2e-3, "boddy_swave must equal the s-wave anchor (hoof)."
    print("\n  boddy_swave == hoof (s-wave on s-wave): PASS")
    print(f"  boddy_pwave != boddy_swave (distinct p-wave limit): PASS "
          f"(ratio {s_pw / s_sw:.1f}x)")

    if not ok:
        sys.exit(1)
    print("\nAUDIT PASSED: s-wave limits use s-wave data; each velocity law uses its own data.")


if __name__ == "__main__":
    main()
