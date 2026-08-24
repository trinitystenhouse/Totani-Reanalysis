#!/usr/bin/env python3
r"""
hardic_physics.py
==================
Physics engine for the "hard inverse-Compton" (hardic) injected cosmic-ray
population template: a spectral-mimic alternative-source test against the
NFW-squared halo interpretation of Stenhouse, Ghag & Deppisch
(arXiv:2607.08552). Quantifies whether a centrally-injected leptonic
population, radiating via inverse-Compton off the interstellar radiation
field, can reproduce the observed 20 GeV halo morphology and spectrum well
enough to compete with (or replace) the dark-matter-annihilation halo
template in the pixel-level fit.

What this computes
-------------------
A burst (or continuous, or crudely advected) population of e+ + e- injected
at the Galactic Centre, propagated with the paper's own galdef transport
(spherical diffusion + radiative losses, Syrovatskii Green's function, free
z-escape boundary via image charges), then converted to inverse-Compton
gamma-ray emissivity using the **full Klein-Nishina** kernel (Blumenthal &
Gould 1970) against a three-component interstellar radiation field (CMB,
FIR/dust, optical/UV starlight), and finally projected along the line of
sight onto the analysis grid.

Why full Klein-Nishina: at E_gamma ~ 20 GeV the starlight channel has
4*gamma*eps/(m_e c^2) ~ 0.9 (order unity) -- the Thomson approximation is
wrong for that channel.  CMB stays Thomson-safe, FIR picks up ~10%
corrections.

Everything internal to this module is computed in eV / cm / s / erg-free
units (electron and photon energies in eV) to avoid unit-conversion bugs
in the Klein-Nishina kernel; the public "build a template" entry points
(`emissivity_grid`, `project_los`) accept/return MeV consistent with the
rest of the pipeline (`totani_helpers.totani_io`), since that is what
`build_hardic_template.py` needs to write FITS templates.

Physical model
--------------
Injection (burst, e+ + e- combined, at the Galactic Centre, t=0):

    Q(E) = Q0 * (E / 1 GeV)^(-Gamma) * exp(-E / Ec)

normalised so that ``E_inj = integral_{1 GeV}^inf E Q(E) dE`` equals a
chosen total injected energy (default convention: theta=1 <-> E_inj = 1e52
erg, see `q0_from_einj`).

Transport: spherically symmetric diffusion with homogeneous radiative
losses, dE/dt = -b0 E^2 (Syrovatskii 1959; Atoyan, Aharonian & Voelk 1995):

    n_e(E, r, t) = Q(E0) * (1 - b0 t E)^(Gamma-2)
                   * exp(-r^2 / lambda^2) / (pi^1.5 lambda^3)

    E0 = E / (1 - b0 t E),   zero if b0 t E >= 1

    lambda^2(E, t) = [4 D(E) / ((1-delta) b0 E)]
                     * [1 - (1 - b0 t E)^(1-delta)]
                     --> 4 D(E) t   as t << t_cool

D(E) = D0 * (R(E)/R_br)^delta with R(E) the rigidity (~E for ultra-
relativistic leptons); D0, delta, R_br and the halo half-height z_h are
read from the paper's own galdef
(``fermi_data/totani/templates/galdef_54_Lorimer_z6kpc_R30kpc_...``):
D0 = 7.97593e28 cm^2/s @ 4 GV, delta = 0.33, z_h = 6 kpc, convection off.
The z_h free-escape boundary is imposed by the standard method-of-images
sum over the z-coordinate only (the model has no boundary in R).

b0 = (4/3) sigma_T c U_eff / (m_e c^2)^2, with U_eff a Klein-Nishina-
weighted effective interstellar radiation field energy density (see
`effective_u_eff`): the starlight contribution to the LOSS rate is reduced
by the same physics that suppresses the emission kernel, evaluated at a
single reference electron energy (0.7 TeV, matching the paper's own
lambda(0.7 TeV) grid convention).  This keeps b0 a single constant, which
is what makes the closed-form Green's function above valid; a fully
energy-dependent b0(E) would require solving the loss trajectory
numerically and was judged not worth the added complexity for a
template-level physicality check (flagged here as an explicit assumption:
**this is the one place this module goes beyond a literal "KN-weighted"
prescription**, since that requirement does not by itself specify how the
weighting should be implemented).

Continuous-injection variant:

    n_cont(E, r) = (1/T) * integral_0^T n_burst(E, r, t') dt'

evaluated by (Ee-adaptive) numerical quadrature over ~30-60 age nodes
(the integrand is defined to be exactly zero for t' >= 1/(b0 E), i.e. the
"zero for b0 t E >= 1" branch above, so the *effective* upper limit is
min(T, 1/(b0 E)), while the normalising divisor stays the fixed T).

Advected variant (optional, crude bracket, NOT a solved wind solution):
the burst Green's function evaluated with z -> |z| - v_wind * t.

Klein-Nishina IC kernel
------------------------
Blumenthal & Gould (1970) isotropic single-electron / isotropic-photon-
field kernel:

    dN_gamma/(dt dE1) = (2 pi r_e^2 c / gamma^2)
                         * integral deps [n(eps)/eps] * F(q, Gamma_e)

    F(q, Gamma_e) = 2 q ln q + (1+2q)(1-q)
                    + 0.5 (Gamma_e q)^2 (1-q) / (1 + Gamma_e q)

    Gamma_e = 4 eps gamma / (m_e c^2),  q = E1 / [Gamma_e (gamma m_e c^2 - E1)]

valid for 1/(4 gamma^2) <= q <= 1 (zero outside).  Reduces to the Thomson
kernel as Gamma_e -> 0 (verified in `_selftest_thomson_limit`).

ISRF: each of the three components (CMB, FIR/dust, optical/UV) is
represented as a single diluted blackbody, with the FIR and optical
components additionally following a double-exponential Galactic disc,
U_i(R, z) = U_i,sun * exp(-(R-R_sun)/R_i) * exp(-|z|/z_i).  Parameter
values (local energy densities, disc scale lengths/heights) are
*representative*, chosen to reproduce standard literature values for the
mean photon energies and local energy densities of each ISRF component to
within a quoted precision -- the actual GALPROP/Porter et al. ISRF FITS
maps are not available on this machine.  This is flagged as the
least-defensible ingredient in the model; the CMB-only bracket variant
(``--isrf cmb_only``) exists specifically to test sensitivity to this
choice, since the CMB component is exact and uniform.

Units summary (internal)
-------------------------
Electron energy E_e, photon energies eps/E_gamma: eV.
Radial coordinates R, z, r, lambda: kpc (converted to cm internally where
needed for the diffusion/Gaussian maths).
Time t, T: seconds internally; helper constructors accept Myr.
Number density n_e(E_e, r): eV^-1 cm^-3.
Emissivity q(E_gamma, r): photons cm^-3 s^-1 eV^-1 (*total*, not per
steradian -- `project_los` divides by 4 pi, following the standard
I = (1/4 pi) integral q ds line-of-sight convention).
Final public dN/dE output (from `emissivity_grid` + `project_los` via
`build_hardic_template.py`): ph cm^-2 s^-1 sr^-1 MeV^-1, matching
`totani_helpers.totani_io` / `build_giem_template.py` convention.

Self-tests
----------
Run ``python hardic_physics.py --selftest`` for the unit-level checks:
(1) diffusion-loss length closed form vs. direct numerical
integration of D(E(t')) along the loss trajectory; (2) small-t limit
lambda^2 -> 4 D t; (3) free-space burst Green's function integrates to
unity (number conservation) when losses and the z_h boundary are switched
off; (4) energy budget: total radiated + surviving spectral energy does
not exceed the injected energy; (5) full Klein-Nishina IC kernel reduces
to the Thomson kernel/total power when 4 gamma eps / m_e c^2 << 1.

References
----------
Syrovatskii, S. I. 1959, Soviet Astronomy, 3, 22.
Atoyan, A. M., Aharonian, F. A., & Voelk, H. J. 1995, Phys. Rev. D, 52, 3265.
Blumenthal, G. R., & Gould, R. J. 1970, Rev. Mod. Phys., 42, 237.
Jones, F. C. 1968, Phys. Rev., 167, 1159.
Porter, T. A., Moskalenko, I. V., & Strong, A. W. 2006/2008 (ISRF, class
    of model this module's parametric ISRF is a stand-in for; the actual
    tabulated maps are not on disk).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import constants as _const
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Physical constants (CODATA-ish, via scipy.constants for traceability)
# ---------------------------------------------------------------------------

M_E_EV = _const.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
# electron rest-mass energy [eV]  (~510998.95 eV)

SIGMA_T_CM2 = _const.physical_constants["Thomson cross section"][0] * 1e4
# Thomson cross-section [cm^2] (scipy gives m^2)

C_CM_S = _const.c * 1e2  # speed of light [cm/s]

R_E_CM2 = 3.0 * SIGMA_T_CM2 / (8.0 * np.pi)  # classical electron radius^2 [cm^2]
# (sigma_T = (8 pi/3) r_e^2)

H_EV_S = _const.h / _const.eV  # Planck constant [eV s]
HC_EV_CM = H_EV_S * C_CM_S  # h*c [eV cm]
K_B_EV_K = _const.k / _const.eV  # Boltzmann constant [eV/K]

KPC_CM = _const.parsec * 1e3 * 1e2  # 1 kpc in cm
MYR_S = 1.0e6 * 365.25 * 86400.0  # 1 Myr in seconds (Julian year)
EV_PER_ERG = 1.0 / (_const.eV * 1e7)  # eV per erg (erg -> eV multiplier)
EV_PER_GEV = 1.0e9
EV_PER_MEV = 1.0e6

# Stefan-Boltzmann / radiation constant, converted to eV cm^-3 K^-4
_A_RAD_SI = 4.0 * _const.Stefan_Boltzmann / _const.c  # J m^-3 K^-4
A_RAD_EV_CM3_K4 = _A_RAD_SI * EV_PER_ERG * 1.0e-7 * 1.0e-6
# J -> eV: * (1/eV); here _A_RAD_SI is already J m^-3 K^-4, convert J->eV
# directly (not via erg) and m^-3 -> cm^-3 (*1e-6).
A_RAD_EV_CM3_K4 = (_A_RAD_SI / _const.eV) * 1.0e-6


def u_blackbody_ev_cm3(T_K: float) -> float:
    """Total (undiluted) blackbody energy density a*T^4 [eV cm^-3]."""
    return float(A_RAD_EV_CM3_K4 * T_K ** 4)


def blackbody_number_density(eps_ev, T_K: float):
    """Undiluted blackbody photon number density dn/deps [cm^-3 eV^-1].

    n(eps) = 8 pi eps^2 / [ (hc)^3 (exp(eps/kT) - 1) ]
    """
    eps_ev = np.asarray(eps_ev, dtype=float)
    kT = K_B_EV_K * T_K
    x = eps_ev / kT
    # avoid overflow for large x and division issues at x=0
    with np.errstate(over="ignore", invalid="ignore"):
        expm1 = np.expm1(np.clip(x, -700, 700))
    out = np.where(
        x > 1.0e-8,
        8.0 * np.pi * eps_ev ** 2 / (HC_EV_CM ** 3 * np.where(expm1 > 0, expm1, np.inf)),
        0.0,
    )
    return out


# ---------------------------------------------------------------------------
# ISRF model: three diluted-blackbody components on a double-exponential disc
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ISRFComponent:
    name: str
    T_K: float  # representative blackbody temperature [K]
    U0_ev_cm3: float  # local (R=R_sun, z=0) energy density [eV cm^-3]
    R_scale_kpc: Optional[float]  # radial e-folding scale; None => uniform
    z_scale_kpc: Optional[float]  # vertical e-folding scale; None => uniform

    def U_at(self, R_kpc, z_kpc, R_sun_kpc: float):
        R_kpc = np.asarray(R_kpc, dtype=float)
        z_kpc = np.asarray(z_kpc, dtype=float)
        if self.R_scale_kpc is None or self.z_scale_kpc is None:
            return np.full(np.broadcast(R_kpc, z_kpc).shape, self.U0_ev_cm3, dtype=float)
        return (
            self.U0_ev_cm3
            * np.exp(-(R_kpc - R_sun_kpc) / self.R_scale_kpc)
            * np.exp(-np.abs(z_kpc) / self.z_scale_kpc)
        )


R_SUN_KPC = 8.2  # matches the paper's own LOS geometry convention

# Representative ISRF components. CMB is exact; FIR/optical are
# parametric stand-ins for the (unavailable) GALPROP/Porter et al. maps --
# see module docstring.  T chosen so the number-weighted mean photon energy
# <eps> = 2.701 kT matches standard literature values to the quoted
# precision; U0/R_scale/z_scale are round numbers within the ranges
# commonly quoted for the solar-neighbourhood ISRF (e.g. Porter,
# Moskalenko & Strong-class models), NOT a fit to a specific published map
# (unavailable on this machine).
T_CMB_K = 2.72548  # Fixsen 2009 COBE/FIRAS
CMB_FIELD = ISRFComponent(
    name="cmb", T_K=T_CMB_K, U0_ev_cm3=u_blackbody_ev_cm3(T_CMB_K),
    R_scale_kpc=None, z_scale_kpc=None,
)
FIR_FIELD = ISRFComponent(
    name="fir", T_K=30.0, U0_ev_cm3=0.40, R_scale_kpc=5.0, z_scale_kpc=0.15,
)
OPTICAL_FIELD = ISRFComponent(
    name="optical", T_K=3900.0, U0_ev_cm3=0.30, R_scale_kpc=3.5, z_scale_kpc=0.35,
)
DEFAULT_FIELDS: Sequence[ISRFComponent] = (CMB_FIELD, FIR_FIELD, OPTICAL_FIELD)
CMB_ONLY_FIELDS: Sequence[ISRFComponent] = (CMB_FIELD,)


# ---------------------------------------------------------------------------
# Transport parameters, matched to the paper's own galdef
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportParams:
    """Diffusion + escape-boundary parameters, matched to
    ``fermi_data/totani/templates/galdef_54_Lorimer_z6kpc_R30kpc_...``
    (verified values: D0_xx=7.97593e28 cm^2/s, D_rigid_br=4.0e3 MV,
    D_g_1=D_g_2=0.33 [no break], convection=0)."""

    D0_cm2_s: float = 7.97593e28
    R_br_gv: float = 4.0
    delta: float = 0.33
    z_h_kpc: float = 6.0  # halo half-height (free-escape boundary)
    n_images: int = 12  # image-charge terms each side (converges fast)

    def D_of_E(self, E_gev):
        """Diffusion coefficient D(E) [cm^2/s]; rigidity R[GV] ~= E[GeV]
        for ultra-relativistic e+/e- (exact relativistic p*c used)."""
        E_gev = np.asarray(E_gev, dtype=float)
        m_e_gev = M_E_EV / EV_PER_GEV
        pc_gev = np.sqrt(np.maximum(E_gev ** 2 - m_e_gev ** 2, 0.0))
        return self.D0_cm2_s * (pc_gev / self.R_br_gv) ** self.delta


DEFAULT_TRANSPORT = TransportParams()


def b0_from_ueff(U_eff_ev_cm3: float) -> float:
    """Energy-loss coefficient b0 [GeV^-1 s^-1] for dE/dt = -b0 E^2 (E in
    GeV), b0 = (4/3) sigma_T c U_eff / (m_e c^2)^2.

    Verified against the standard analytic estimate
    t_cool ~= 0.31 Myr (E/TeV)^-1 (U_eff/eV cm^-3)^-1
    in `_selftest_b0_calibration`.
    """
    U_eff_ev_cm3 = float(U_eff_ev_cm3)
    b0_per_ev_s = (4.0 / 3.0) * SIGMA_T_CM2 * C_CM_S * U_eff_ev_cm3 / (M_E_EV ** 2)
    # b0_per_ev_s has units eV^-1 s^-1 (dE/dt[eV/s] = -b0_per_ev_s * E[eV]^2)
    return float(EV_PER_GEV * b0_per_ev_s)  # -> GeV^-1 s^-1


# ---------------------------------------------------------------------------
# Klein-Nishina IC kernel (Blumenthal & Gould 1970)
# ---------------------------------------------------------------------------


def _bg_F(q, Gamma_e):
    q = np.asarray(q, dtype=float)
    Gamma_e = np.asarray(Gamma_e, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnq = np.log(np.where(q > 0, q, 1.0))
    term1 = 2.0 * q * lnq
    term2 = (1.0 + 2.0 * q) * (1.0 - q)
    term3 = 0.5 * (Gamma_e * q) ** 2 * (1.0 - q) / (1.0 + Gamma_e * q)
    return term1 + term2 + term3


def ic_kernel_per_target_photon(E_e_ev, eps_ev, E_gamma_ev):
    """Blumenthal & Gould (1970) isotropic single-electron / single-target-
    photon-energy IC kernel.

    Returns K [cm^3 s^-1 eV^-2] such that, for a target photon field with
    number density n(eps) [cm^-3 eV^-1],
        dN_gamma/(dt dE_gamma) = integral deps  K(E_e, eps, E_gamma) * n(eps).

    Broadcasts over its arguments (use e.g. eps_ev[:, None] * ones and
    E_gamma_ev as scalar, or vice versa, as convenient).
    """
    E_e_ev = np.asarray(E_e_ev, dtype=float)
    eps_ev = np.asarray(eps_ev, dtype=float)
    E_gamma_ev = np.asarray(E_gamma_ev, dtype=float)

    gamma = E_e_ev / M_E_EV
    Gamma_e = 4.0 * eps_ev * gamma / M_E_EV
    Ee_total = gamma * M_E_EV  # == E_e_ev; kept for clarity/units
    denom = Gamma_e * (Ee_total - E_gamma_ev)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(denom > 0, E_gamma_ev / np.where(denom != 0, denom, np.inf), -1.0)
    q_min = 1.0 / (4.0 * gamma ** 2)
    valid = (q >= q_min) & (q <= 1.0) & (denom > 0) & (E_gamma_ev < Ee_total) & np.isfinite(q)

    q_safe = np.where(valid, np.clip(q, 1.0e-300, 1.0), 0.5)
    F = _bg_F(q_safe, Gamma_e)
    pref = 2.0 * np.pi * R_E_CM2 * C_CM_S / np.maximum(gamma, 1.0e-30) ** 2
    K = np.where(valid & (eps_ev > 0), pref * F / np.maximum(eps_ev, 1.0e-300), 0.0)
    return K


def kernel_table_per_unit_dilution(E_e_grid_ev, E_gamma_grid_ev, field: ISRFComponent,
                                    n_eps: int = 96, eps_span_kT: float = 40.0):
    r"""kappa_ref[k, j] = integral deps K(E_e_grid[j], eps, E_gamma_grid[k]) *
    n_BB(eps, field.T_K)  [s^-1 eV^-1 per electron, PER UNIT DILUTION, i.e.
    for the *undiluted* blackbody at field.T_K -- multiply by
    U(R,z)/U_blackbody(T) to get the actual per-electron photon production
    rate at that location; see module docstring].

    Returns array shape (n_Egamma, n_Ee).
    """
    E_e_grid_ev = np.asarray(E_e_grid_ev, dtype=float)
    E_gamma_grid_ev = np.asarray(E_gamma_grid_ev, dtype=float)
    kT = K_B_EV_K * field.T_K
    eps_grid = np.geomspace(kT * 1.0e-3, kT * eps_span_kT, n_eps)
    n_bb = blackbody_number_density(eps_grid, field.T_K)  # (n_eps,)

    n_Eg = E_gamma_grid_ev.size
    n_Ee = E_e_grid_ev.size
    out = np.empty((n_Eg, n_Ee), dtype=float)

    # K(E_e, eps, E_gamma): broadcast eps over axis 0, E_e over axis 1
    Ee_b = E_e_grid_ev[None, :]  # (1, n_Ee)
    eps_b = eps_grid[:, None]  # (n_eps, 1)
    for k, Eg in enumerate(E_gamma_grid_ev):
        K = ic_kernel_per_target_photon(Ee_b, eps_b, Eg)  # (n_eps, n_Ee)
        integrand = K * n_bb[:, None]
        out[k, :] = np.trapz(integrand, eps_grid, axis=0)
    return out


def effective_u_eff(E_ref_gev: float = 700.0, R_kpc: float = R_SUN_KPC, z_kpc: float = 0.0,
                     fields: Sequence[ISRFComponent] = DEFAULT_FIELDS) -> float:
    """Klein-Nishina-weighted effective interstellar radiation-field energy
    density [eV cm^-3] for use in the single constant-b0 transport formula.

    For each field, the KN suppression factor eta_i(E_ref) = P_i(E_ref) /
    P_i,Thomson(E_ref) is computed from the exact BG kernel (independent of
    position, since Thomson-regime universality means P_Thomson depends
    only on U_i, not T_i), then U_eff = sum_i eta_i * U_i(R,z).  This is
    the simplest defensible implementation of the requirement that losses
    use the same U_eff, KN-weighted for the optical term -- flagged in the
    module docstring as the one deliberate departure from a literal
    (fully energy-dependent) treatment.
    """
    gamma_ref = (E_ref_gev * EV_PER_GEV) / M_E_EV
    u_eff = 0.0
    for fld in fields:
        U_here = float(fld.U_at(R_kpc, z_kpc, R_SUN_KPC))
        if U_here <= 0:
            continue
        eta = _kn_suppression_factor(gamma_ref, fld)
        u_eff += eta * U_here
    return float(u_eff)


def _kn_suppression_factor(gamma: float, field: ISRFComponent, n_eps: int = 200,
                            eps_span_kT: float = 60.0) -> float:
    """eta(gamma) = P_exact(gamma) / P_Thomson(gamma) for a single electron
    of Lorentz factor `gamma` scattering an undiluted blackbody at
    field.T_K (dimensionless; independent of the field's normalisation)."""
    kT = K_B_EV_K * field.T_K
    eps_grid = np.geomspace(kT * 1.0e-3, kT * eps_span_kT, n_eps)
    n_bb = blackbody_number_density(eps_grid, field.T_K)

    E_e_ev = gamma * M_E_EV
    Ee_total = E_e_ev
    # Egamma grid: sample densely enough to capture the full BG spectrum,
    # up to just below the kinematic ceiling Gamma_e*Ee/(1+Gamma_e).
    Gamma_e_max = 4.0 * eps_grid[-1] * gamma / M_E_EV
    Eg_ceiling = Gamma_e_max * Ee_total / (1.0 + Gamma_e_max)
    Eg_floor = 1.0e-6 * Eg_ceiling if Eg_ceiling > 0 else 1.0
    Eg_grid = np.geomspace(max(Eg_floor, 1.0e-8), Eg_ceiling * (1.0 - 1.0e-9), 400)

    eps_b = eps_grid[:, None]
    P_exact = 0.0
    dNdE = np.empty_like(Eg_grid)
    for k, Eg in enumerate(Eg_grid):
        K = ic_kernel_per_target_photon(E_e_ev, eps_b, Eg)  # (n_eps,1) shape via broadcast
        integrand = K.ravel() * n_bb
        dNdE[k] = np.trapz(integrand, eps_grid)
    P_exact = np.trapz(Eg_grid * dNdE, Eg_grid)  # eV/s per unit blackbody U (see below)

    U_bb = u_blackbody_ev_cm3(field.T_K)
    P_thomson = (4.0 / 3.0) * SIGMA_T_CM2 * C_CM_S * gamma ** 2 * U_bb
    if P_thomson <= 0:
        return 1.0
    return float(np.clip(P_exact / P_thomson, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Injection spectrum
# ---------------------------------------------------------------------------


def injection_shape(E_gev, Gamma_inj: float, Ec_gev: float):
    """Unnormalised Q(E)/Q0 = (E/1 GeV)^-Gamma * exp(-E/Ec), E in GeV."""
    E_gev = np.asarray(E_gev, dtype=float)
    return E_gev ** (-Gamma_inj) * np.exp(-E_gev / Ec_gev)


def q0_from_einj(E_inj_erg: float, Gamma_inj: float, Ec_gev: float, E_min_gev: float = 1.0):
    """Solve for Q0 [GeV^-1] such that integral_{E_min}^inf E Q(E) dE =
    E_inj (erg), via numerical quadrature (robust for any Gamma_inj,
    including Gamma_inj close to/above 2 where an incomplete-gamma
    closed form would need analytic continuation)."""
    from scipy.integrate import quad

    integrand = lambda E: E * injection_shape(E, Gamma_inj, Ec_gev)
    val, _err = quad(integrand, E_min_gev, max(200.0 * Ec_gev, E_min_gev * 10), limit=400)
    E_inj_gev = E_inj_erg * EV_PER_ERG / EV_PER_GEV
    if val <= 0 or not np.isfinite(val):
        raise RuntimeError(f"Injection-spectrum normalisation integral invalid: {val}")
    return float(E_inj_gev / val)


# ---------------------------------------------------------------------------
# Syrovatskii Green's function (burst), with z_h image-charge boundary
# ---------------------------------------------------------------------------


def lambda_squared_cm2(E_gev, t_s, transport: TransportParams, b0_gev1_s1):
    """Closed-form diffusion-loss length squared [cm^2] (see module
    docstring).  Vectorised over E_gev and/or t_s (broadcast).  Zero
    (with a NaN->0 guard) for b0*t*E >= 1 (fully cooled).

    Special-cases b0*t*E -> 0 (either b0=0, the pure-diffusion/no-losses
    limit, or genuinely small b0*t*E) via the analytic small-x limit
    lambda^2 -> 4 D(E) t, since the closed form [1-(1-x)^(1-delta)]/x is a
    removable-singularity 0/0 there and naive floating point evaluation at
    exactly b0=0 returns NaN (verified failure mode; see
    `_selftest_number_conservation`, which exercises exactly this limit).
    """
    E_gev = np.asarray(E_gev, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    delta = transport.delta
    D = transport.D_of_E(E_gev)
    b0tE = np.asarray(b0_gev1_s1) * t_s * E_gev
    valid = b0tE < 1.0
    small = np.abs(b0tE) < 1.0e-6  # Taylor regime; relative error < 1e-6

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        b0tE_safe = np.clip(np.where(valid, b0tE, 0.5), -1.0e30, 1.0 - 1e-15)
        bracket_over_x = np.where(
            small,
            1.0,  # [1-(1-x)^(1-delta)]/x -> (1-delta) as x->0, folded below
            (1.0 - (1.0 - b0tE_safe) ** (1.0 - delta)) / np.where(b0tE_safe != 0, b0tE_safe, 1.0),
        )
        # lam2 = 4 D t * [bracket/x] * [(1-delta)]^{-1} * (1-delta) ... simplify:
        # lam2 = 4D/((1-delta) b0 E) * bracket = 4 D t * (bracket/x) / (1-delta)
        # (using x = b0 t E, so 4D/((1-delta) b0 E) = 4 D t / ((1-delta) x))
        lam2_general = 4.0 * D * t_s * bracket_over_x / (1.0 - delta)
        lam2_small = 4.0 * D * t_s  # exact small-x / b0=0 limit
        lam2 = np.where(valid, np.where(small, lam2_small, lam2_general), 0.0)
    return np.where(np.isfinite(lam2) & (lam2 >= 0), lam2, 0.0)


def lambda_squared_numeric_cm2(E_gev: float, t_s: float, transport: TransportParams,
                                b0_gev1_s1: float, n_quad: int = 4000):
    """Independent cross-check of `lambda_squared_cm2`: lambda^2 = 4
    integral_0^t D(E(t')) dt', with E(t') the loss trajectory that reaches
    E at t' = t (solved backwards from E0 = E/(1-b0 t E)).  Scalar-only
    (used in self-tests)."""
    b0tE = b0_gev1_s1 * t_s * E_gev
    if b0tE >= 1.0:
        return 0.0
    E0 = E_gev / (1.0 - b0tE)
    # E(t') = E0 / (1 + b0 E0 t'); integrate D(E(t')) dt' from 0 to t via
    # substitution x = b0 E0 t', dt' = dx/(b0 E0)
    X = b0_gev1_s1 * E0 * t_s
    x = np.linspace(0.0, X, n_quad)
    Et = E0 / (1.0 + x)
    D_of_t = transport.D_of_E(Et)
    integral_over_x = np.trapz(D_of_t, x)  # integral D dx
    integral_over_tprime = integral_over_x / (b0_gev1_s1 * E0)  # dt' = dx/(b0 E0)
    return float(4.0 * integral_over_tprime)


def _image_sum_z(z_kpc, z0_eff_kpc, lam_kpc, z_h_kpc, n_images):
    """sum_{n=-N}^{N} (-1)^n exp(-(z - z0_eff - 2 n z_h)^2 / lam^2), the
    1D method-of-images factor enforcing n=0 at z = +/- z_h for a source
    displaced to z0_eff (z0_eff=0 for the un-advected burst)."""
    z_kpc = np.asarray(z_kpc, dtype=float)
    lam_kpc = np.asarray(lam_kpc, dtype=float)
    total = np.zeros(np.broadcast(z_kpc, lam_kpc).shape, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(-n_images, n_images + 1):
            dz = z_kpc - z0_eff_kpc - 2.0 * n * z_h_kpc
            arg = (dz / np.where(lam_kpc > 0, lam_kpc, np.inf)) ** 2
            term = np.where(lam_kpc > 0, np.exp(-np.clip(arg, 0, 700)), 0.0)
            total += ((-1.0) ** n) * term
    return total


def burst_density(E_gev, R_kpc, z_kpc, t_s, *, Gamma_inj: float, Ec_gev: float, Q0_gev1: float,
                   transport: TransportParams = DEFAULT_TRANSPORT, b0_gev1_s1: float,
                   z0_eff_kpc=0.0):
    """n_e(E, R, z, t) [eV^-1 cm^-3] for the burst Green's function
    (§2.2), with the z_h image-charge boundary.  `z0_eff_kpc` is the
    (possibly advected, |z|-shifted) effective source-plane offset; pass 0
    for the plain burst.  Returns 0 where b0 t E >= 1 (fully cooled).

    Broadcasts R_kpc, z_kpc against each other; E_gev, t_s are scalars
    (this function is called once per (E_e, t) pair with R,z arrays --
    see `emissivity_grid` / `continuous_injection_density`).
    """
    b0tE = b0_gev1_s1 * t_s * E_gev
    if b0tE >= 1.0:
        R_kpc = np.asarray(R_kpc, dtype=float)
        z_kpc = np.asarray(z_kpc, dtype=float)
        return np.zeros(np.broadcast(R_kpc, z_kpc).shape, dtype=float)

    E0_gev = E_gev / (1.0 - b0tE)
    Qshape = injection_shape(E0_gev, Gamma_inj, Ec_gev)
    Q_val = Q0_gev1 * Qshape  # GeV^-1, injected spectrum evaluated at E0
    prefactor = Q_val * (1.0 - b0tE) ** (Gamma_inj - 2.0)  # GeV^-1
    # Convert prefactor (per GeV of E) to per-eV: dN/dE[eV^-1] =
    # dN/dE[GeV^-1] / EV_PER_GEV
    prefactor_ev1 = prefactor / EV_PER_GEV

    lam2_cm2 = lambda_squared_cm2(np.asarray(E_gev), np.asarray(t_s), transport, b0_gev1_s1)
    lam2_cm2 = float(np.asarray(lam2_cm2).reshape(-1)[0])
    if lam2_cm2 <= 0:
        R_kpc = np.asarray(R_kpc, dtype=float)
        z_kpc = np.asarray(z_kpc, dtype=float)
        return np.zeros(np.broadcast(R_kpc, z_kpc).shape, dtype=float)
    lam_kpc = np.sqrt(lam2_cm2) / KPC_CM

    R_kpc = np.asarray(R_kpc, dtype=float)
    z_kpc = np.asarray(z_kpc, dtype=float)
    radial = np.exp(-np.clip((R_kpc / lam_kpc) ** 2, 0, 700))
    z_images = _image_sum_z(z_kpc, z0_eff_kpc, lam_kpc, transport.z_h_kpc, transport.n_images)

    # normalisation: pi^{3/2} lambda^3 in cm^3 (isotropic 3D Gaussian norm),
    # split as [pi lambda^2 (radial, "R" 2D part folded w/ 1/(sqrt(pi) lam)
    # for the z part] -- implement directly as the standard 3D form:
    lam_cm = lam_kpc * KPC_CM
    norm_cm3 = (np.pi ** 1.5) * lam_cm ** 3
    dens = prefactor_ev1 * radial * z_images / norm_cm3
    return np.maximum(dens, 0.0)


def continuous_injection_density(E_gev, R_kpc, z_kpc, T_myr: float, *, Gamma_inj: float,
                                  Ec_gev: float, Q0_gev1: float,
                                  transport: TransportParams = DEFAULT_TRANSPORT,
                                  b0_gev1_s1: float, n_age_nodes: int = 40,
                                  z0_eff_kpc=0.0):
    """n_cont(E, R, z) = (1/T) integral_0^T n_burst(E, R, z, t') dt' [eV^-1
    cm^-3], T in Myr.  The age grid is adapted per-E to the physically
    valid window [~0, min(T, 1/(b0 E))] (log-spaced), since the integrand
    is identically zero beyond t_cool(E) -- see module docstring."""
    T_s = T_myr * MYR_S
    t_cool_s = 1.0 / max(b0_gev1_s1 * E_gev, 1e-300)
    t_upper = min(T_s, t_cool_s * (1.0 - 1.0e-9))
    if t_upper <= 0:
        R_kpc = np.asarray(R_kpc, dtype=float)
        z_kpc = np.asarray(z_kpc, dtype=float)
        return np.zeros(np.broadcast(R_kpc, z_kpc).shape, dtype=float)

    t_min = t_upper * 1.0e-4
    t_nodes = np.geomspace(max(t_min, 1.0), t_upper, n_age_nodes)

    R_kpc = np.asarray(R_kpc, dtype=float)
    z_kpc = np.asarray(z_kpc, dtype=float)
    shp = np.broadcast(R_kpc, z_kpc).shape
    stack = np.empty((n_age_nodes,) + shp, dtype=float)
    for i, tp in enumerate(t_nodes):
        stack[i] = burst_density(
            E_gev, R_kpc, z_kpc, tp, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev, Q0_gev1=Q0_gev1,
            transport=transport, b0_gev1_s1=b0_gev1_s1, z0_eff_kpc=z0_eff_kpc,
        )
    integral = np.trapz(stack, t_nodes, axis=0)
    return integral / T_s


# ---------------------------------------------------------------------------
# Age-solving helper (target lambda(0.7 TeV) -> burst age, with automatic
# D0-boost fallback when the target exceeds the loss-limited asymptote)
# ---------------------------------------------------------------------------


@dataclass
class MorphologyCalibration:
    t_s: float
    d0_scale: float
    lambda_target_kpc: float
    lambda_achieved_kpc: float
    lambda_max_kpc: float  # loss-limited asymptote at the calibration D0_scale


def calibrate_burst_age(lambda_target_kpc: float, *, E_ref_gev: float = 700.0,
                         transport: TransportParams = DEFAULT_TRANSPORT,
                         b0_gev1_s1: float, margin: float = 1.02) -> MorphologyCalibration:
    """Solve for the burst age t such that lambda(E_ref, t) = target, at
    the baseline D0.  If the target exceeds the loss-limited asymptote
    lambda_max = lambda(E_ref, t->t_cool^-), automatically scales D0 up
    (D0_scale = (target*margin / lambda_max)^2, since lambda_max^2 propto
    D0) so the target is reachable at a finite age close to t_cool, and
    reports the applied scale factor (this is the "requires enhanced D"
    branch anticipated for the wider-lambda morphologies (burst_m,
    burst_l); it is reported, not hidden -- see build_hardic_template.py
    CLI output and the DBOOST FITS header keyword)."""
    t_cool = 1.0 / (b0_gev1_s1 * E_ref_gev)

    def lam_kpc(t_s, d0_scale):
        D0 = transport.D0_cm2_s * d0_scale
        tp = TransportParams(D0_cm2_s=D0, R_br_gv=transport.R_br_gv, delta=transport.delta,
                              z_h_kpc=transport.z_h_kpc, n_images=transport.n_images)
        lam2 = lambda_squared_cm2(E_ref_gev, t_s, tp, b0_gev1_s1)
        return float(np.sqrt(max(lam2, 0.0)) / KPC_CM)

    lam_max_baseline = lam_kpc(t_cool * (1.0 - 1.0e-9), 1.0)

    if lambda_target_kpc < lam_max_baseline * 0.98:
        t_sol = brentq(lambda t: lam_kpc(t, 1.0) - lambda_target_kpc,
                        1.0, t_cool * (1.0 - 1.0e-9))
        return MorphologyCalibration(t_s=t_sol, d0_scale=1.0,
                                      lambda_target_kpc=lambda_target_kpc,
                                      lambda_achieved_kpc=lam_kpc(t_sol, 1.0),
                                      lambda_max_kpc=lam_max_baseline)

    # Target beyond the baseline loss-limited asymptote: boost D0 so the
    # (D0-dependent) asymptote sits `margin` above the target, then solve
    # for a finite age near (but below) the boosted t_cool.  b0/t_cool do
    # not depend on D0, so t_cool is unchanged.
    d0_scale = (lambda_target_kpc * margin / lam_max_baseline) ** 2
    t_sol = brentq(lambda t: lam_kpc(t, d0_scale) - lambda_target_kpc,
                    1.0, t_cool * (1.0 - 1.0e-9))
    return MorphologyCalibration(
        t_s=t_sol, d0_scale=d0_scale, lambda_target_kpc=lambda_target_kpc,
        lambda_achieved_kpc=lam_kpc(t_sol, d0_scale),
        lambda_max_kpc=lam_kpc(t_cool * (1.0 - 1.0e-9), d0_scale),
    )


# ---------------------------------------------------------------------------
# Emissivity assembly and LOS projection
# ---------------------------------------------------------------------------


def emissivity_grid(E_gamma_mev: np.ndarray, ne_of_Ee, E_e_grid_gev: np.ndarray,
                     R_grid_kpc: np.ndarray, z_grid_kpc: np.ndarray,
                     fields: Sequence[ISRFComponent] = DEFAULT_FIELDS,
                     verbose: bool = True):
    """Compute q(E_gamma; R, z) [photons cm^-3 s^-1 MeV^-1] on the (R,z)
    grid for each of the requested E_gamma bin centres.

    Parameters
    ----------
    E_gamma_mev : (nE,) bin-centre energies [MeV] (e.g. from
        `totani_helpers.totani_io.read_counts_and_ebounds`).
    ne_of_Ee : callable(E_e_gev: float) -> (nR, nZ) array, the electron
        number density [eV^-1 cm^-3] at each (R,z) grid point for that
        single electron energy -- i.e. one call per E_e_grid_gev entry.
        (Kept as a callable rather than a precomputed cube so the caller
        can choose burst / continuous / advected without this function
        needing to know which.)
    E_e_grid_gev : (nEe,) electron energy grid [GeV] the injection/
        transport machinery is evaluated on.
    R_grid_kpc, z_grid_kpc : 1D grids defining the (R,z) plane.
    fields : ISRF components to include (use `CMB_ONLY_FIELDS` for the
        CMB-only bracket).

    Returns
    -------
    q : (nE, nR, nZ) array [photons cm^-3 s^-1 MeV^-1]
    """
    E_gamma_ev = np.asarray(E_gamma_mev, dtype=float) * EV_PER_MEV
    E_e_grid_ev = np.asarray(E_e_grid_gev, dtype=float) * EV_PER_GEV
    nE = E_gamma_ev.size
    nEe = E_e_grid_ev.size
    Rg, Zg = np.meshgrid(R_grid_kpc, z_grid_kpc, indexing="ij")
    nR, nZ = Rg.shape

    # (nEe, nR, nZ) electron density cube (one call per E_e; cheap relative
    # to the kernel contraction below).
    ne_cube = np.empty((nEe, nR, nZ), dtype=float)
    for j, Ee_gev in enumerate(E_e_grid_gev):
        ne_cube[j] = ne_of_Ee(float(Ee_gev))

    q = np.zeros((nE, nR, nZ), dtype=float)
    U_bb_ref = {fld.name: u_blackbody_ev_cm3(fld.T_K) for fld in fields}
    for fld in fields:
        if verbose:
            print(f"  [hardic_physics] Klein-Nishina kernel table: {fld.name} "
                  f"(T={fld.T_K:.0f} K)")
        kappa_ref = kernel_table_per_unit_dilution(E_e_grid_ev, E_gamma_ev, fld)  # (nE, nEe)
        w_map = fld.U_at(Rg, Zg, R_SUN_KPC) / U_bb_ref[fld.name]  # (nR, nZ), dimensionless
        for k in range(nE):
            # integral dEe ne(Ee,R,z) * kappa_ref[k, :] over Ee (eV)
            integrand = ne_cube * kappa_ref[k, :, None, None]
            per_electron = np.trapz(integrand, E_e_grid_ev, axis=0)  # (nR,nZ) [s^-1 eV^-1]
            q[k] += w_map * per_electron
    # convert photons s^-1 eV^-1 cm^-3 -> photons s^-1 MeV^-1 cm^-3
    q *= EV_PER_MEV
    return q


def project_los(q_grid, R_grid_kpc, z_grid_kpc, lon_deg, lat_deg, *, R_sun_kpc=R_SUN_KPC,
                 s_max_kpc=30.0, n_steps=300, s_min_kpc=0.05, row_chunk=32):
    """I(E_gamma, l, b) = (1/4 pi) integral_0^{s_max} q(E_gamma; R(s),
    z(s)) ds  [photons cm^-2 s^-1 sr^-1 MeV^-1], via bilinear interpolation
    of `q_grid` (already evaluated for one energy bin, shape (nR, nZ)) and
    trapezoidal quadrature in s (converted to cm).

    lon_deg, lat_deg : arrays of identical shape (the sky grid pixels).

    Processes the sky grid in row-chunks of `row_chunk` (rather than
    allocating one (ny, nx, n_steps) array for the whole grid at once):
    for the full 960x960 grid at n_steps~300 a single unchunked array
    would be ~2 GB, which OOMs on a memory-constrained machine (observed
    in the sandbox this module was developed/tested in, 3.8 GB total
    RAM); chunking keeps the peak well under 100 MB with no accuracy cost.
    """
    from scipy.interpolate import RegularGridInterpolator

    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    out_shape = lon.shape
    if lon.ndim != 2:
        # flatten to a single "row" so the chunking loop below still works
        lon = lon.reshape(1, -1)
        lat = lat.reshape(1, -1)

    s = np.linspace(s_min_kpc, s_max_kpc, n_steps)  # (n_steps,)
    ds_cm = (s[1] - s[0]) * KPC_CM

    interp = RegularGridInterpolator(
        (R_grid_kpc, z_grid_kpc), q_grid, bounds_error=False, fill_value=0.0,
    )

    n_rows = lon.shape[0]
    out = np.empty(lon.shape, dtype=float)
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        cosb = np.cos(lat[r0:r1])[..., None]
        sinb = np.sin(lat[r0:r1])[..., None]
        cosl = np.cos(lon[r0:r1])[..., None]
        sinl = np.sin(lon[r0:r1])[..., None]

        x = R_sun_kpc - s * cosb * cosl
        y = -s * cosb * sinl
        z = s * sinb
        R = np.sqrt(x ** 2 + y ** 2)

        pts = np.stack([R.ravel(), z.ravel()], axis=-1)
        q_along = interp(pts).reshape(R.shape)  # (chunk_rows, nx, n_steps)

        out[r0:r1] = np.trapz(q_along, dx=ds_cm, axis=-1)

    return (out / (4.0 * np.pi)).reshape(out_shape)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _selftest_lambda_closed_form():
    transport = DEFAULT_TRANSPORT
    b0 = b0_from_ueff(0.85)
    E_gev, t_myr = 700.0, 0.3
    t_s = t_myr * MYR_S
    lam2_closed = lambda_squared_cm2(E_gev, t_s, transport, b0)
    lam2_closed = float(np.asarray(lam2_closed))
    lam2_numeric = lambda_squared_numeric_cm2(E_gev, t_s, transport, b0)
    rel = abs(lam2_closed - lam2_numeric) / lam2_numeric
    ok = rel < 1.0e-4
    print(f"  [lambda closed-form vs numeric] closed={lam2_closed:.6e} "
          f"numeric={lam2_numeric:.6e} rel.diff={rel:.2e}  {'PASS' if ok else 'FAIL'}")
    assert ok, "lambda^2 closed form does not match numerical loss-trajectory integral"

    # small-t limit: lambda^2 -> 4 D(E) t
    t_small_s = 1.0e-4 * (1.0 / (b0 * E_gev))  # << t_cool
    lam2_small = float(np.asarray(lambda_squared_cm2(E_gev, t_small_s, transport, b0)))
    lam2_expect = 4.0 * transport.D_of_E(E_gev) * t_small_s
    rel2 = abs(lam2_small - lam2_expect) / lam2_expect
    ok2 = rel2 < 1.0e-3
    print(f"  [lambda small-t limit -> 4Dt] rel.diff={rel2:.2e}  {'PASS' if ok2 else 'FAIL'}")
    assert ok2, "lambda^2 does not reduce to 4Dt in the small-t limit"


def _selftest_b0_calibration():
    # standard estimate: t_cool ~= 0.31 Myr (E/TeV)^-1 (U_eff/eV cm^-3)^-1
    b0 = b0_from_ueff(1.0)
    t_cool_s = 1.0 / (b0 * 1000.0)  # E=1 TeV=1000 GeV, U_eff=1 eV/cm3
    t_cool_myr = t_cool_s / MYR_S
    ok = abs(t_cool_myr - 0.31) / 0.31 < 0.02
    print(f"  [b0 calibration] t_cool(1 TeV, 1 eV/cm3) = {t_cool_myr:.4f} Myr "
          f"(standard estimate: ~0.31 Myr)  {'PASS' if ok else 'FAIL'}")
    assert ok, "b0 normalisation does not reproduce the standard t_cool coefficient"


def _selftest_number_conservation():
    # Free space (no boundary: use a huge z_h so images are irrelevant),
    # no losses (b0=0): the volume integral of the burst Green's function
    # at a fixed E must recover Q(E) exactly (not 1 -- Q0=1 GeV^-1 is a
    # choice, not a normalisation to unity; with b0=0, E0=E so Q(E0)=Q(E)
    # trivially, and this is the "conserves particle number" check: the
    # SAME Q(E) comes out regardless of the propagation time t, since with
    # no losses and no boundary, diffusion only spreads particles in
    # space, never destroys them).
    Gamma_inj, Ec_gev, Q0_gev1 = 2.2, 1.0e4, 1.0
    Q_expected = Q0_gev1 * injection_shape(10.0, Gamma_inj, Ec_gev)
    transport = TransportParams(z_h_kpc=1.0e6, n_images=0)
    E_gev, t_s = 10.0, 1.0e13
    lam2 = lambda_squared_cm2(E_gev, t_s, transport, 0.0)
    lam_kpc = np.sqrt(float(np.asarray(lam2))) / KPC_CM
    n = 400
    R = np.linspace(1e-6, 12 * lam_kpc, n)
    z = np.linspace(-12 * lam_kpc, 12 * lam_kpc, 2 * n)
    Rg, Zg = np.meshgrid(R, z, indexing="ij")
    dens = burst_density(E_gev, Rg, Zg, t_s, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev, Q0_gev1=Q0_gev1,
                          transport=transport, b0_gev1_s1=0.0)
    # dens [eV^-1 cm^-3]; integrate 2 pi R dens dR dz over the (R,z)
    # half-plane (R>=0) with the standard cylindrical volume element.
    # dens is a density per cm^3; R, z above are in kpc, so the raw
    # trapz over (R,z) in kpc units needs one factor of KPC_CM^3 to become
    # a proper volume integral in cm^3 (dR dz R ~ length^3).
    integrand = 2.0 * np.pi * Rg * dens
    total = np.trapz(np.trapz(integrand, z, axis=1), R, axis=0) * (KPC_CM ** 3)
    # dens is per eV; convert the volume integral back to "per GeV" to
    # compare directly with Q_expected [GeV^-1].
    total_per_gev = total * EV_PER_GEV
    rel = abs(total_per_gev - Q_expected) / Q_expected
    ok = rel < 2.0e-2
    print(f"  [number conservation, free space] integral = {total_per_gev:.6e} "
          f"(expect Q(E)={Q_expected:.6e}) rel.diff={rel:.3f}  {'PASS' if ok else 'FAIL'}")
    assert ok, "free-space burst Green's function does not conserve particle number"

    # Repeat at a different propagation time t2 != t: with no losses and
    # no boundary, the SAME Q(E) must come out (diffusion only spreads
    # particles in space; it never creates or destroys them).
    t2_s = 5.0 * t_s
    dens2 = burst_density(E_gev, Rg, Zg, t2_s, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev, Q0_gev1=Q0_gev1,
                           transport=transport, b0_gev1_s1=0.0)
    integrand2 = 2.0 * np.pi * Rg * dens2
    total2_per_gev = (np.trapz(np.trapz(integrand2, z, axis=1), R, axis=0)
                       * (KPC_CM ** 3) * EV_PER_GEV)
    rel2 = abs(total2_per_gev - Q_expected) / Q_expected
    ok2 = rel2 < 2.0e-2
    print(f"  [number conservation, free space, t x5] integral = {total2_per_gev:.6e} "
          f"rel.diff={rel2:.3f}  {'PASS' if ok2 else 'FAIL'}")
    assert ok2, "particle number is not conserved as the propagation time changes"


def _selftest_energy_budget():
    # Radiated + surviving spectral energy must not exceed injected energy.
    transport = DEFAULT_TRANSPORT
    b0 = b0_from_ueff(0.85)
    Gamma_inj, Ec_gev = 2.2, 1.0e4
    E_inj_erg = 1.0e52
    Q0 = q0_from_einj(E_inj_erg, Gamma_inj, Ec_gev)
    t_s = 0.3 * MYR_S

    # Surviving spectrum at age t: integral over all space of n_e(E,r,t) dr
    # at each E equals Q(E0)*(1-b0tE)^(Gamma-2) * [volume integral of the
    # normalised spatial Gaussian, which is <=1 once the z_h boundary
    # removes escaped particles]. Use free space (upper bound) for a
    # clean, boundary-independent energy check.
    free_transport = TransportParams(z_h_kpc=1.0e6, n_images=0)
    E_grid = np.geomspace(1.0, 0.999 / (b0 * 1e-3) if False else 5.0e4, 400)
    surviving_dNdE = np.empty_like(E_grid)
    for i, E in enumerate(E_grid):
        b0tE = b0 * t_s * E
        if b0tE >= 1.0:
            surviving_dNdE[i] = 0.0
            continue
        E0 = E / (1.0 - b0tE)
        surviving_dNdE[i] = Q0 * injection_shape(E0, Gamma_inj, Ec_gev) * (1.0 - b0tE) ** (Gamma_inj - 2.0)

    E_surviving_gev = np.trapz(E_grid * surviving_dNdE, E_grid)
    E_inj_gev = E_inj_erg * EV_PER_ERG / EV_PER_GEV
    frac = E_surviving_gev / E_inj_gev
    ok = 0.0 <= frac <= 1.0 + 1.0e-6
    print(f"  [energy budget] surviving/injected = {frac:.4f} "
          f"(must be in [0,1])  {'PASS' if ok else 'FAIL'}")
    assert ok, "surviving spectral energy exceeds injected energy (radiated fraction > 1)"


def _selftest_thomson_limit():
    # Low-energy electron + very cold/dilute photon so Gamma_e << 1;
    # compare total IC power to the exact Thomson formula (4/3) sigma_T c
    # gamma^2 U for a single target photon of energy eps and number
    # density n0.
    gamma = 50.0  # E_e ~ 25 MeV -- keeps Gamma_e small even for eV photons
    eps_ev = 1.0e-3  # 1 meV target photon (way sub-CMB, ensures Gamma_e<<1)
    Gamma_e = 4.0 * eps_ev * gamma / M_E_EV
    assert Gamma_e < 1.0e-3, f"Gamma_e={Gamma_e} not small enough for a Thomson-limit test"

    n0 = 1.0  # cm^-3, monochromatic target (delta-function n(eps)=n0 delta(eps-eps0))
    # Evaluate dN/dtdE1 on a fine E1 grid and integrate E1*dN/dtdE1 dE1;
    # since our kernel already carries the 1/eps and n(eps) factors, for a
    # monochromatic field we just evaluate K(E_e,eps0,E1)*n0 directly
    # (no eps integral -- treat n0 as a number density "at" eps0, i.e.
    # equivalent to n(eps)=n0*eps0*delta(eps-eps0) folded into K/eps... to
    # keep this unambiguous, integrate a narrow top-hat n(eps) instead).
    eps_grid = np.linspace(eps_ev * 0.999, eps_ev * 1.001, 5)
    n_of_eps = np.full_like(eps_grid, n0 / (eps_grid[-1] - eps_grid[0]))

    Ee_ev = gamma * M_E_EV
    Eg_ceiling = Gamma_e * Ee_ev / (1.0 + Gamma_e)
    Eg_grid = np.geomspace(Eg_ceiling * 1.0e-4, Eg_ceiling * (1.0 - 1.0e-8), 500)

    dNdE = np.empty_like(Eg_grid)
    for k, Eg in enumerate(Eg_grid):
        K = ic_kernel_per_target_photon(Ee_ev, eps_grid, Eg)
        dNdE[k] = np.trapz(K * n_of_eps, eps_grid)
    P_exact = np.trapz(Eg_grid * dNdE, Eg_grid)  # eV/s

    P_thomson = (4.0 / 3.0) * SIGMA_T_CM2 * C_CM_S * gamma ** 2 * eps_ev * n0
    rel = abs(P_exact - P_thomson) / P_thomson
    ok = rel < 0.03
    print(f"  [Klein-Nishina -> Thomson limit] P_exact={P_exact:.6e} eV/s "
          f"P_thomson={P_thomson:.6e} eV/s  rel.diff={rel:.3f}  {'PASS' if ok else 'FAIL'}")
    assert ok, "Klein-Nishina kernel does not reduce to the Thomson power formula"


def _selftest_kn_suppression_starlight():
    # At E_e ~ 65 GeV (Thomson estimate for 20.8 GeV photons off
    # starlight), 4 gamma eps/mec^2 ~ 0.9 -- order-unity KN
    # suppression is expected for the optical field at the energies that
    # matter for this template.
    gamma = (700.0 * EV_PER_GEV) / M_E_EV  # 0.7 TeV reference
    eta = _kn_suppression_factor(gamma, OPTICAL_FIELD)
    eta_cmb = _kn_suppression_factor(gamma, CMB_FIELD)
    ok = eta < 0.9 and eta_cmb > 0.95
    print(f"  [KN suppression @ 0.7 TeV] eta_optical={eta:.3f} (expect <0.9), "
          f"eta_CMB={eta_cmb:.3f} (expect >0.95)  {'PASS' if ok else 'FAIL'}")
    assert ok, "KN suppression factors are not in the physically expected regime"


def run_selftests(verbose: bool = True) -> bool:
    tests = [
        ("lambda closed form", _selftest_lambda_closed_form),
        ("b0 calibration", _selftest_b0_calibration),
        ("number conservation (free space)", _selftest_number_conservation),
        ("energy budget", _selftest_energy_budget),
        ("Klein-Nishina Thomson limit", _selftest_thomson_limit),
        ("KN suppression regime (starlight vs CMB)", _selftest_kn_suppression_starlight),
    ]
    n_ok = 0
    for name, fn in tests:
        print(f"[selftest] {name}")
        try:
            fn()
            n_ok += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
    print(f"\n{n_ok}/{len(tests)} self-tests passed.")
    return n_ok == len(tests)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="Run unit-level self-tests and exit.")
    args = ap.parse_args()
    if args.selftest:
        ok = run_selftests()
        sys.exit(0 if ok else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
