#!/usr/bin/env python3
r"""
Decaying-dark-matter fit to the 20 GeV halo (eps_gamma ~ rho^1, the NFW-rho1 /
global_rho1 morphology), the decay counterpart of the annihilation fit.

Physics
-------
A two-body decay chi -> b bbar (or W+W-, tau+tau-) of mass m_chi produces the
SAME photon spectrum as annihilation of mass m_ann = m_chi/2 (equal CM energy),
so  dN/dE_decay(m_chi) = dN/dE_ann(m_chi/2).  The decay flux scales with the
first power of density:

    dPhi/dE = (1 / (4 pi m_chi tau)) * D * dN/dE_decay ,   D = integral rho dl .

Using the SAME pole-flux convention as the annihilation pipeline
(smooth_nfw_sigma_v_from_norm: fitted_norm = 1000 <sv> J_pole / (8 pi m^2)),
the decay analog is

    fitted_norm = 1000 * D_pole / (4 pi m_chi tau)      =>
    tau = 1000 * D_pole_GeV_cm2 / (4 pi * m_chi * fitted_norm) ,

with fitted_norm the amplitude that best matches the rho^1 halo E^2 dN/dE.

Outputs: best-fit m_chi, lifetime tau (per channel), a J_pole units cross-check,
and the comparison to published decay lower limits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("REPO_PATH", str(_HERE.parent))  # so the loader finds the fits

from core.spectral_reshaping import pppc_energy_flux_template  # noqa: E402
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402

# Via Lactea II NFW (Totani Sec. 3.3) -- same as the templates.
RS, RVIR, R0, RHOS, GAMMA = 21.0, 402.0, 8.0, 8.1e6, 1.0   # kpc, Msun/kpc^3
MSUN_KPC3_TO_GEV_CM3 = 3.817e-8
KPC_TO_CM = 3.0857e21

# Published decay lower limits on tau (s), b bbar, ~TeV (order-of-magnitude, for context):
#   IGRB (Blanco & Hooper 2019; Cohen et al. 2017): tau > few x 10^28 s
#   CMB / others comparable or weaker at these masses.
TAU_IGRB_LOWER = 3e28  # s, representative IGRB bound for ~TeV bbar


def _rho(r):
    x = r / RS
    return RHOS * x ** (-GAMMA) * (1.0 + x) ** (GAMMA - 3.0)


def pole_los_integral(power, smax=50.0 * RVIR, ns=400000):
    """integral rho^power dl at the Galactic pole (b=90 -> r=sqrt(R0^2+s^2)), Msun/kpc^(...)."""
    s = np.linspace(0.0, smax, ns)
    r = np.sqrt(R0 ** 2 + s ** 2)
    return np.trapezoid(_rho(r) ** power, s)  # Msun^power / kpc^(3power-1)


def D_pole_GeV_cm2():
    D_msun_kpc2 = pole_los_integral(1.0)              # Msun/kpc^2
    return D_msun_kpc2 * MSUN_KPC3_TO_GEV_CM3 * KPC_TO_CM


def J_pole_GeV2_cm5():
    J_msun2_kpc5 = pole_los_integral(2.0)             # Msun^2/kpc^5
    return J_msun2_kpc5 * MSUN_KPC3_TO_GEV_CM3 ** 2 * KPC_TO_CM


def fit_channel(channel, halo, masses, table_path=None):
    E = halo.E_bins_GeV
    phi = np.asarray(halo.phi, float)
    err = 0.5 * (np.asarray(halo.phi_err_hi, float) + np.asarray(halo.phi_err_lo, float))
    mask = halo.positive_mask & halo.finite_mask & (err > 0)
    best = dict(chi2=np.inf)
    for m_chi in masses:
        try:
            shape = pppc_energy_flux_template(
                E, m_chi / 2.0, channel=channel, primary="gamma",
                table_path=table_path, normalise=False,
            )  # E^2 dN/dE per decay (= ann at m/2), GeV units
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
            best = dict(chi2=chi2, m_chi=float(m_chi), norm=float(A),
                        ndof=int(mask.sum()) - 1)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", nargs="+", default=["bb", "WW", "tautau"])
    p.add_argument("--profile", default="global_rho1",
                   help="rho^1 halo morphology (decay): global_rho1 (fig7) or rho1 (cellwise).")
    p.add_argument("--m-min", type=float, default=100.0)
    p.add_argument("--m-max", type=float, default=6000.0)
    p.add_argument("--n-mass", type=int, default=120)
    p.add_argument("--pppc-table", default=None)
    args = p.parse_args(argv)

    Dpole = D_pole_GeV_cm2()
    Jpole = J_pole_GeV2_cm5()
    print(f"Units cross-check: J_pole = {Jpole:.3e} GeV^2/cm^5 (expect ~4.0e21)")
    print(f"                   D_pole = {Dpole:.3e} GeV/cm^2")

    mcmc_dir = _MCMC_DIRS.get(args.profile, args.profile)
    halo = load_halo_spectrum(mcmc_dir)
    masses = np.geomspace(args.m_min, args.m_max, args.n_mass)
    print(f"\nDecay fit to the {args.profile} (rho^1) halo:")
    print(f"{'chan':6s} {'m_chi[GeV]':>10s} {'tau[s]':>12s} {'chi2/ndof':>10s} "
          f"{'tau/tau_IGRB':>12s}")
    results = {}
    for ch in args.channels:
        b = fit_channel(ch, halo, masses, table_path=args.pppc_table)
        if not np.isfinite(b["chi2"]):
            print(f"{ch:6s}  no finite fit"); continue
        tau = 1000.0 * Dpole / (4.0 * np.pi * b["m_chi"] * b["norm"])
        results[ch] = dict(m_chi=b["m_chi"], tau=tau, chi2_ndof=b["chi2"] / b["ndof"])
        print(f"{ch:6s} {b['m_chi']:10.0f} {tau:12.3e} {b['chi2']/b['ndof']:10.2f} "
              f"{tau / TAU_IGRB_LOWER:12.2e}")
    print(f"\n(IGRB decay lower limit used for context: tau > {TAU_IGRB_LOWER:.0e} s, ~TeV bbar.)")
    return results


if __name__ == "__main__":
    main()
