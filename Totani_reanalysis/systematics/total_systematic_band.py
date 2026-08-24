#!/usr/bin/env python3
r"""
Combine the systematic legs into a TOTAL band on the s-wave dSph tension R.

Legs (each as a multiplicative factor on R = <sv>_halo / limit):
  1. Foreground/diffuse  : from propagate_systematics.py refit-per-variant spread
                           (GALPROP vs GIEM, Loop I/bubble, PS-mask, PSF).
                           Asymmetric, model-selection -> use the measured frac range.
  2. Halo profile / J     : R ~ 1/J_halo; J_halo ~ rho_s^2 ~ rho_sun^2, plus inner-slope.
                           Estimated fractional uncertainty (default 0.55 ~ +/-0.44 dex/2),
                           the dominant ASTROPHYSICAL term (cf. Totani 2025).
  3. Dwarf J-factor       : limit ~ 1/J_dSph -> R ~ J_dSph; measured from the Boddy 2019
                           posteriors (stacked sum(j_s) fractional uncertainty).

Symmetric legs (2,3) combine in quadrature in ln-space; the asymmetric foreground leg
is applied as its measured multiplicative range, then combined with the symmetric
quadrature. Both a quadrature band and an extreme (linear) worst case are reported.

NOTE: the disk toggle is NOT a leg here -- the disk-included fit is a different ROI
(Totani's high-latitude test), not a systematic on the high-latitude <sv>.
"""

from __future__ import annotations

import argparse
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--R0", type=float, default=5.121, help="central s-wave tension (bb).")
    # NOTE ON THE FOREGROUND EDGES
    # ----------------------------
    # These are the extremal envelope of the FOREGROUND leg, i.e. the min/max of the
    # dSph-tension fraction over the interstellar-emission-model groups only
    # (diffuse, diffuse_ics, diffuse_grid) -- see FOREGROUND_GROUPS in
    # propagate_systematics.py, which prints exactly this range as
    #   "# dSph tension: ... [frac LO-HI; dominant variant: ...]".
    #
    # They are NOT the spread over every variant: quadrant/hemisphere (ROI tests),
    # cr_injection (an extra template, not an alternative foreground), mask and PSF
    # (instrument-response checks) are reported separately and must not widen this leg.
    #
    # The defaults below are the POST-IEM-GRID, RE-ANCHORED values, from
    # propagate_systematics.py over FOREGROUND_GROUPS =
    # {baseline, diffuse, diffuse_ics, diffuse_grid_bubbles} on the 200-point
    # mass grid over 20-2000 GeV:
    #   # dSph tension: 5.121 (+1.64 / -3.2) [frac 0.375-1.320; dominant: diffuse_giem_only]
    # Low edge: diffuse_giem_only.  High edge: iem_so_z6_bubbles_reextracted (OB-star
    # source distribution), which pulls the best-fit mass down to ~369 GeV where the
    # dwarf limits are tighter.
    #
    # History of this leg, so a reader can tell a re-anchoring from a regression:
    #   0.359-1.032  pre-IEM-grid            -> band [1.6, 9.3], R0 = 5.2
    #   0.384-1.339  grid on 14/80-pt scans  -> band [1.7, 10.0], R0 = 5.2
    #   0.375-1.320  everything on 200-pt    -> band [1.6, 9.8],  R0 = 5.121
    # The lower edge returning to 1.6 is the re-anchoring, not a reversion.
    # Re-derive and update these whenever the variant suite changes.
    p.add_argument("--fg-lo", type=float, default=0.375,
                   help="foreground frac low edge; take from propagate_systematics.py "
                        "'dSph tension ... [frac LO-HI]' over the foreground groups.")
    p.add_argument("--fg-hi", type=float, default=1.320,
                   help="foreground frac high edge; see --fg-lo.")
    p.add_argument("--halo-J-frac", type=float, default=0.48,
                   help="halo J_pole fractional uncertainty on <sv> (computed: rho_sun-dominated, "
                        "rho_sun=0.42+/-0.10 GeV/cm^3; r_s and gamma negligible at the pole).")
    p.add_argument("--dwarf-J-frac", type=float, default=0.537,
                   help="dwarf J-factor fractional uncertainty (Boddy stacked sum j_s).")
    a = p.parse_args(argv)

    # symmetric legs in ln-space
    s_haloJ = np.log1p(a.halo_J_frac)
    s_dwarfJ = np.log1p(a.dwarf_J_frac)
    sig_sym = np.hypot(s_haloJ, s_dwarfJ)          # quadrature of the two J legs
    # foreground asymmetric in ln-space
    fg_lo_ln, fg_hi_ln = np.log(a.fg_lo), np.log(a.fg_hi)

    # quadrature band: combine foreground edge (in ln) with the symmetric sigma
    lo_q = a.R0 * np.exp(-np.hypot(-fg_lo_ln, sig_sym))
    hi_q = a.R0 * np.exp(+np.hypot(fg_hi_ln, sig_sym))
    # extreme (linear) worst/best case: all push the same way
    lo_x = a.R0 * a.fg_lo / (1 + a.halo_J_frac) / (1 + a.dwarf_J_frac)
    hi_x = a.R0 * a.fg_hi * (1 + a.halo_J_frac) * (1 + a.dwarf_J_frac)

    print(f"central R = {a.R0}")
    print(f"legs (fractional):")
    print(f"  foreground/diffuse : x[{a.fg_lo:.2f}, {a.fg_hi:.2f}]  (GALPROP<->GIEM dominated)")
    print(f"  halo profile / J   : +/-{a.halo_J_frac*100:.0f}%  (rho_sun/inner-slope; dominant astro)")
    print(f"  dwarf J-factor     : +/-{a.dwarf_J_frac*100:.0f}%  (Boddy 2019 stacked posteriors)")
    print(f"\nTOTAL s-wave tension band:")
    print(f"  quadrature : R = {a.R0:.1f}  [{lo_q:.1f}, {hi_q:.1f}]")
    print(f"  extreme    : R = {a.R0:.1f}  [{lo_x:.1f}, {hi_x:.1f}]")
    print(f"\nInterpretation: lower edge {lo_q:.1f} (quadrature) / {lo_x:.1f} (extreme) -> the s-wave")
    print(f"tension is ~{lo_q:.0f}-{hi_q:.0f}x at face value but is consistent with ~1 within the full")
    print(f"systematic budget (diffuse model + halo profile + dwarf J-factor), matching Totani 2025.")
    print(f"p-wave inherits the foreground+halo-J factors on <sv> and remains << 1 throughout.")


if __name__ == "__main__":
    main()
