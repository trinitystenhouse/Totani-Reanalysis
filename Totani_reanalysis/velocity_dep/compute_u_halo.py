#!/usr/bin/env python3
r"""
Compute the ROI- and rho^2-weighted halo velocity moments u_halo needed to turn
the Boddy dwarf moments into a halo-comparable velocity-dependent dSph limit.

WHAT IT PRODUCES
----------------
The rho^2-weighted line-of-sight averages over Totani's ROI (|l|<=60 deg,
10<=|b|<=60 deg) of the relative-velocity moments:

    u_halo_pwave = << (v_rel/c)^2 >>      (for sigma v ~ v^2)
    u_halo_dwave = << (v_rel/c)^4 >>      (for sigma v ~ v^4)
    u_halo_somm  = << (c/v_rel)  >>       (for sigma v ~ 1/v)

where <<X>> = integral_ROI dOmega integral_los ds rho^2 X / integral rho^2, the
SAME rho^2 weight that defines the annihilation J-factor.  These pair with the
Boddy dwarf moments (j_n/j_s) so that, per wave,

    factor_n = u_halo_n / u_dwarf_n ,   limit_halo = sigma_v0_UL * factor_n .

PHYSICS
-------
* sigma_r(r): isotropic spherical Jeans on the SAME (g)NFW density used for the
  annihilation morphology, in the chosen gravitational potential:
      sigma_r^2(r) = (1/rho(r)) * integral_r^inf rho(r') G M(<r') / r'^2 dr' .
* Relative-velocity second moment is exact: <v_rel^2> = 2<v^2> = 6 sigma_r^2,
  independent of the velocity-distribution shape.  => u_halo_pwave is rigorous
  given sigma_r(r).
* d-wave and Sommerfeld use a Maxwellian closure for the relative-velocity
  distribution (1D dispersion s_rel = sqrt(2) sigma_r):
      <v_rel^4> = 60 sigma_r^4   (Maxwell kurtosis 5/3),
      <1/v_rel> = 1/(sqrt(pi) sigma_r).
  Replace with an Eddington f(v) for a shape-exact d/Sommerfeld number; p-wave
  is unaffected.

MODEL CHOICE (see --baryons)
----------------------------
Reuse the SAME NFW you fit (do NOT switch density models -- consistency between
the rho^2 weight and the velocity field requires it).  The only modelling choice
that matters here is the *potential* entering Jeans: NFW self-gravity alone
under-counts sigma_r in the inner few kpc, where the rho^2 weight is largest, so
a baryonic (disk+bulge) contribution raises u_halo.  Run with --baryons none and
--baryons hernquist to bracket it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

G = 4.30091e-6          # kpc (km/s)^2 / Msun
C_KMS = 299_792.458
GEVCM3_TO_MSUNKPC3 = 2.6358e7   # 1 GeV/cm^3 = 2.6358e7 Msun/kpc^3


def gnfw_rho(r, rho_s, rs, gamma):
    x = r / rs
    return rho_s * x ** (-gamma) * (1.0 + x) ** (gamma - 3.0)


def build_sigma_r(rho_s, rs, gamma, *, baryons, m_bary, a_bary, r_max, n=4000):
    """Return (r_grid, sigma_r[km/s]) from isotropic Jeans."""
    r = np.geomspace(1e-3, r_max, n)
    rho = gnfw_rho(r, rho_s, rs, gamma)

    # Enclosed DM mass by cumulative integration of 4 pi rho r^2.
    integrand_m = 4.0 * np.pi * rho * r ** 2
    M_dm = np.concatenate([[0.0], np.cumsum(0.5 * (integrand_m[1:] + integrand_m[:-1]) * np.diff(r))])

    M = M_dm.copy()
    if baryons == "hernquist":
        # Spherical Hernquist baryon bulge+disk proxy: M_b(<r) = M_bary r^2/(r+a)^2.
        M = M + m_bary * r ** 2 / (r + a_bary) ** 2

    # Jeans: sigma_r^2 = (1/rho) * int_r^inf rho G M / r'^2 dr'  (integrate inward).
    h = rho * G * M / r ** 2
    # cumulative integral from outermost point inward
    seg = 0.5 * (h[1:] + h[:-1]) * np.diff(r)
    I_out = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])  # I_out[i]=int_{r_i}^{r_max}
    sigma2 = I_out / rho
    sigma2 = np.clip(sigma2, 0.0, None)
    return r, np.sqrt(sigma2)


def roi_weighted_moments(r_grid, sigma_r, rho_s, rs, gamma, *,
                         r_sun, l_max, b_min, b_max, s_max, nl=120, nb=100, ns=600):
    """rho^2-weighted ROI averages of the three relative-velocity moments."""
    l = np.linspace(0.0, np.deg2rad(l_max), nl)      # symmetric in l -> [0,l_max]
    b = np.linspace(np.deg2rad(b_min), np.deg2rad(b_max), nb)
    s = np.linspace(1e-3, s_max, ns)
    L, B, S = np.meshgrid(l, b, s, indexing="ij")

    cos_psi = np.cos(B) * np.cos(L)
    r = np.sqrt(r_sun ** 2 + S ** 2 - 2.0 * r_sun * S * cos_psi)

    rho = gnfw_rho(r, rho_s, rs, gamma)
    sig = np.interp(r, r_grid, sigma_r)
    w = rho ** 2 * np.cos(B)        # dOmega = cos b db dl ; ds along los

    # relative-velocity moments (km/s based), then convert to c-units.
    vrel2 = 6.0 * sig ** 2
    vrel4 = 60.0 * sig ** 4
    inv_v = 1.0 / (np.sqrt(np.pi) * np.clip(sig, 1e-6, None))   # <1/v_rel>, (km/s)^-1

    W = np.sum(w)
    up = np.sum(w * vrel2) / W / C_KMS ** 2
    ud = np.sum(w * vrel4) / W / C_KMS ** 4
    us = np.sum(w * inv_v) / W * C_KMS            # <c/v_rel>, dimensionless
    # rho^2-weighted effective galactocentric radius (diagnostic)
    r_eff = np.sum(w * r) / W
    return up, ud, us, r_eff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # NFW (defaults = Totani fig6 label: g1, rs21, R08, rvir402)
    p.add_argument("--gamma", type=float, default=1.0, help="gNFW inner slope.")
    p.add_argument("--rs", type=float, default=21.0, help="NFW scale radius [kpc].")
    p.add_argument("--rvir", type=float, default=402.0, help="virial radius [kpc].")
    p.add_argument("--r-sun", type=float, default=8.0, help="solar radius [kpc].")
    p.add_argument("--rho-sun", type=float, default=0.42,
                   help="local DM density at r_sun [GeV/cm^3] (Totani/Via Lactea II: 0.42, "
                        "rho_s=8.1e6 Msun/kpc^3).")
    # ROI
    p.add_argument("--l-max", type=float, default=60.0)
    p.add_argument("--b-min", type=float, default=10.0)
    p.add_argument("--b-max", type=float, default=60.0)
    p.add_argument("--s-max", type=float, default=200.0, help="LOS depth [kpc].")
    # baryons
    p.add_argument("--baryons", choices=["none", "hernquist"], default="none")
    p.add_argument("--m-bary", type=float, default=8.0e10,
                   help="Hernquist baryon mass [Msun] (disk+bulge proxy).")
    p.add_argument("--a-bary", type=float, default=2.5, help="Hernquist scale [kpc].")
    p.add_argument("--out", type=Path, default=_HERE / "halo_moments.csv")
    args = p.parse_args(argv)

    # Normalise rho_s from local density.
    rho_sun = args.rho_sun * GEVCM3_TO_MSUNKPC3
    x0 = args.r_sun / args.rs
    rho_s = rho_sun / (x0 ** (-args.gamma) * (1.0 + x0) ** (args.gamma - 3.0))

    r_grid, sigma_r = build_sigma_r(
        rho_s, args.rs, args.gamma,
        baryons=args.baryons, m_bary=args.m_bary, a_bary=args.a_bary,
        r_max=5.0 * args.rvir,
    )
    up, ud, us, r_eff = roi_weighted_moments(
        r_grid, sigma_r, rho_s, args.rs, args.gamma,
        r_sun=args.r_sun, l_max=args.l_max, b_min=args.b_min, b_max=args.b_max,
        s_max=args.s_max,
    )

    sig_at_sun = float(np.interp(args.r_sun, r_grid, sigma_r))
    print(f"NFW: gamma={args.gamma} rs={args.rs} rvir={args.rvir} rho_sun={args.rho_sun} GeV/cm^3"
          f"  baryons={args.baryons}")
    print(f"  sigma_r(R_sun)        = {sig_at_sun:.1f} km/s")
    print(f"  rho^2-weighted r_eff  = {r_eff:.2f} kpc  (where the ROI signal mostly comes from)")
    print(f"  u_halo_pwave = {up:.4e}   (<(v_rel/c)^2>; v_rel,rms = {np.sqrt(up)*C_KMS:.1f} km/s)  [RIGOROUS]")
    print(f"  u_halo_dwave = {ud:.4e}   (<(v_rel/c)^4>)                                  [Maxwellian closure]")
    print(f"  u_halo_somm  = {us:.4e}   (<c/v_rel>)                                      [Maxwellian closure]")

    with open(args.out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["u_halo_pwave", "u_halo_dwave", "u_halo_somm"])
        wcsv.writerow([up, ud, us])
    print(f"\nWrote {args.out}")
    print("Next:")
    print(f"  python extract_boddy_limits.py --mode boddy --boddy-csv boddy_moments_0d5.csv "
          f"--halo-csv {args.out.name}")


if __name__ == "__main__":
    main()
