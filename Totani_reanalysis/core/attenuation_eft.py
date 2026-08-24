import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.eft_validity import (
    eft_kinematic_lambda_curve as _shared_eft_kinematic_lambda_curve,
    unitarity_lambda_curve as _shared_unitarity_lambda_curve,
)
# Canonical lab-frame kinematics live in cross_sections.py; import rather than
# duplicate (these were byte-identical copies — see 2026-06 audit dedup).
from core.cross_sections import (
    get_s_lab_DMrest,
    get_t_lab_DMrest,
    lab_recoil_ratio,
)
from helpers.plot_style import current_savefig_kwargs, set_plot_style

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# =============================================================================
# CONSTANTS & NFW PARAMETERS (Totani / Via Lactea II)
# =============================================================================

# NFW parameters from Totani (Via Lactea II)
rho_s_tot   = 8.1e6          # M_sun / kpc^3  — Totani scale density
r_s_tot     = 21.0           # kpc             — Totani scale radius
r_sun       = 8.0            # kpc             — Sun–GC distance (Totani uses 8 kpc)
r_vir       = 402.0          # kpc             — virial radius

# Conversion factors
MSUN_KPC3_TO_GEV_CM3 = 3.817e-8   # 1 M_sun/kpc^3 = 3.817e-8 GeV/cm^3
KPC_TO_CM             = 3.0857e21  # 1 kpc in cm

rho_s_GeV = rho_s_tot * MSUN_KPC3_TO_GEV_CM3   # GeV/cm^3
r_s_cm    = r_s_tot   * KPC_TO_CM               # cm
r_sun_cm  = r_sun     * KPC_TO_CM               # cm
r_vir_cm  = r_vir     * KPC_TO_CM               # cm

# ---------------------------------------------------------------------------
# Totani energy bins and halo spectrum — loaded from MCMC posteriors
# (Totani_paper_check/mcmc/mcmc_results_fig6, NFW-rho^2, disk excluded)
#
# These replace the previous hand-digitised read-offs from Totani Fig. 8,
# which had incorrect bin centres (29.5 GeV instead of 20.757 GeV for the
# peak bin) and coarse flux values. The MCMC posteriors are the ground truth.
#
# E_BINS_GEV  : 13 bin centres from Ectr_mev stored in mcmc_results_k*.npz
# PHI_TOTANI  : f_nfw_p50 * iso_target_e2  [MeV cm^-2 s^-1 sr^-1]
# SIGMA_TOTANI: 0.5*(f_p84 - f_p16) * iso_target_e2  (symmetric 1-sigma)
#
# Peak bin: k=5, E=20.757 GeV, phi=3.360e-5, SNR=14.14
# ---------------------------------------------------------------------------

def _load_totani_mcmc_arrays(mcmc_dir=None):
    """Load E_bins, phi, sigma from MCMC posteriors. Falls back to legacy
    hard-coded values if the mcmc directory cannot be found.

    Parameters
    ----------
    mcmc_dir : Path or str, optional
        Directory containing mcmc_results_k*.npz files.  If None, the default
        rho2 MCMC path is used (with legacy fallback).
    """
    import sys
    if mcmc_dir is not None:
        _pc_dir = Path(mcmc_dir)
    else:
        _ts_dir = Path(__file__).resolve().parent.parent
        _mcmc_base = _ts_dir.parent / "Totani_paper_check" / "mcmc"
        _candidates = [
            _mcmc_base / "mcmc_results_fig6",
            _mcmc_base / "mcmc_results_fig6_10deg",
            _mcmc_base / "global_fit_results_fig6",
            _mcmc_base / "global_fit_results_fig5",
        ]
        _pc_dir = next(
            (p for p in _candidates if p.exists() and any(p.glob("mcmc_results_k*.npz"))),
            _candidates[0],
        )
    if not _pc_dir.exists() or not any(_pc_dir.glob("mcmc_results_k*.npz")):
        # Fail loudly rather than silently returning legacy hard-coded arrays
        # with KNOWN-WRONG bin centres (29.5 vs 20.757 GeV for the peak bin).
        # A misconfigured run must not silently produce wrong physics.
        raise FileNotFoundError(
            f"Totani MCMC results not found at {_pc_dir} "
            "(expected mcmc_results_k*.npz). Run the Totani_paper_check pipeline "
            "to generate them, or pass an explicit mcmc_dir. The legacy hard-coded "
            "fallback was removed because its bin centres were incorrect."
        )

    iso_e2 = 1e-4   # MeV cm^-2 s^-1 sr^-1 (default; overwritten per bin)
    nE = 13
    E_mev  = np.full(nE, np.nan)
    f_p50  = np.full(nE, np.nan)
    f_p16  = np.full(nE, np.nan)
    f_p84  = np.full(nE, np.nan)
    iso_arr = np.full(nE, iso_e2)

    for k in range(nE):
        npz_path = _pc_dir / f"mcmc_results_k{k:02d}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        if "Ectr_mev" in npz.files:
            E_mev[k] = float(npz["Ectr_mev"])
        if "iso_target_e2" in npz.files:
            v = float(np.atleast_1d(npz["iso_target_e2"]).flat[0])
            if np.isfinite(v) and v > 0:
                iso_arr[k] = v
        labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
        nfw_idx = next((i for i, l in enumerate(labels) if "nfw" in l.lower()), None)
        if nfw_idx is None:
            continue
        if "f_p50" in npz.files:
            f_p50[k] = float(np.atleast_1d(npz["f_p50"])[nfw_idx])
        if "f_p16" in npz.files:
            f_p16[k] = float(np.atleast_1d(npz["f_p16"])[nfw_idx])
        if "f_p84" in npz.files:
            f_p84[k] = float(np.atleast_1d(npz["f_p84"])[nfw_idx])

    phi   = f_p50 * iso_arr
    sigma = 0.5 * np.abs(f_p84 - f_p16) * iso_arr
    E_GeV = E_mev / 1000.0
    return E_GeV, phi, sigma


E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI = _load_totani_mcmc_arrays()


def configure_totani_arrays(mcmc_dir) -> None:
    """Reconfigure the module-level E_BINS_GEV / PHI_TOTANI / SIGMA_TOTANI
    to use data from *mcmc_dir* (must contain mcmc_results_k*.npz files).

    Call this once at the start of any script that selects a non-default
    halo profile, e.g. global_rho2.5 or global_rho1, so that all
    downstream defaults derived from these arrays are consistent.
    """
    global E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI
    E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI = _load_totani_mcmc_arrays(Path(mcmc_dir))


# Forward-scattering angular cutoff. PHYSICAL, not merely a regulator.
#
# 1 - cos(theta) < 1e-4 corresponds to a deflection of 0.81 deg, comparable to
# the Fermi-LAT PSF (68% containment ~0.8 deg at 1 GeV; Atwood+ 2009). A photon
# scattered inside this cone is not resolved as displaced from its original
# direction, and by the Compton relation it also retains essentially all of its
# energy, so it has not been removed from the measured spectrum in any
# observable sense. sigma_tot below is therefore the cross section for
# OBSERVABLE REMOVAL from the line of sight, not the full elastic cross
# section. See Sec. IV A of the paper.
#
# It originated as a regulator for the gravitational case (t -> 0 Rutherford
# divergence) and was long assumed redundant for the EFT operators, on the
# grounds that dsigma/dOmega ~ (-t)^n with n >= 1 is regular at theta = 0.
# That assumption is FALSE for omega >> m_chi: the Compton kinematics compress
# most of the momentum-transfer range into 1 - cos(theta) <~ m_chi/omega, so
# once m_chi/omega < 1e-4 the cut removes the bulk of the t-integral. Fraction
# of the full elastic sigma_tot retained:
#
#   omega/m_chi     1e2     1e3     1e4     1e5     1e7
#   scalar         1.000   0.999   0.875   0.249   0.003
#   Rayleigh-odd   1.000   1.000   0.938   0.317   0.004
#   dipole         0.990   0.909   0.500   0.091   0.001
#
# Do not raise this to 1.0 without also changing what the paper claims
# sigma_tot means.
COS_THETA_MAX = 1.0 - 1e-4

# Unit conversions
HC2_GEV2_TO_M2 = 3.89379e-32   # 1 GeV^-2 = 3.89379e-32 m^2
GEV2_TO_FB     = 3.89379e11    # 1 GeV^-2 = 3.89379e11 fb
FB_TO_CM2      = 1e-39         # 1 fb = 1e-39 cm^2
ALPHA_EM       = 1.0 / 137.035999084


_ROI_PREFAC_CACHE = {}


OPERATOR_METADATA = {
    "dipole_magnetic": {
        "paper_label": r"$O_{\psi\psi F}$",
        "lambda_power": 2,
        "coefficient_power": 0.5,
    },
    "dipole_electric": {
        "paper_label": r"$O_{\psi5\psi F}$",
        "lambda_power": 2,
        "coefficient_power": 0.5,
    },
    "charge_radius": {
        "paper_label": r"$O_{\psi\psi \partial F}$",
        "lambda_power": 4,
        "coefficient_power": 0.5,
    },
    "anapole": {
        "paper_label": r"$O_{\chi5\chi \partial F}$",
        "lambda_power": 4,
        "coefficient_power": 0.5,
    },
    "rayleigh_even": {
        "paper_label": r"$O_{\chi\chi FF}$ / $O_{\psi\psi FF}$",
        "lambda_power": 6,
        "coefficient_power": 1.0 / 3.0,
    },
    "rayleigh_odd": {
        "paper_label": r"$O_{\chi5\chi FF}$ / $O_{\psi5\psi FF}$",
        "lambda_power": 6,
        "coefficient_power": 1.0 / 3.0,
    },
    "scalar_rayleigh": {
        "paper_label": r"$O_{RRFF}$",
        "lambda_power": 4,
        "coefficient_power": 0.5,
    },
}


# =============================================================================
# NFW PROFILE AND GEOMETRY
# =============================================================================

def nfw_density_GeV_cm3(r_cm):
    """NFW density in GeV/cm^3 using Totani / Via Lactea II parameters."""
    x = r_cm / r_s_cm
    return rho_s_GeV / (x * (1.0 + x)**2)


def galactocentric_radius_cm(ell_cm, l_deg, b_deg):
    """Galactocentric radius for a point at distance ell_cm along line of sight (l, b)."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    r2 = (ell_cm**2
          + r_sun_cm**2
          - 2.0 * ell_cm * r_sun_cm * np.cos(b) * np.cos(l))
    return np.sqrt(np.maximum(r2, 0.0))


def los_length_cm(l_deg, b_deg, r_vir_cm=r_vir_cm):
    """Maximum integration length along line of sight (l, b) out to virial radius."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    A = 1.0
    B = -2.0 * r_sun_cm * np.cos(b) * np.cos(l)
    C = r_sun_cm**2 - r_vir_cm**2
    disc = np.maximum(B**2 - 4.0 * A * C, 0.0)
    L = (-B + np.sqrt(disc)) / 2.0
    return L


def compute_J_los(l_deg, b_deg, power=2, n_points=500):
    """J-factor integral  J = int rho^power dl  along line of sight (l, b)."""
    L = los_length_cm(l_deg, b_deg)
    ell = np.linspace(0.0, L, n_points)
    r   = galactocentric_radius_cm(ell, l_deg, b_deg)
    rho = nfw_density_GeV_cm3(r)
    integrand = rho**power
    return np.trapezoid(integrand, ell)


# =============================================================================
# EFFECTIVE OPERATOR CROSS SECTIONS
# =============================================================================

def get_s_max_lab_DMrest(mchi, omega_max):
    return float(mchi)**2 + 2.0 * float(mchi) * float(omega_max)


def get_t_abs_max_lab_DMrest(mchi, omega_max):
    denom = 1.0 + (2.0 * float(omega_max) / float(mchi))
    return 4.0 * float(omega_max)**2 / denom


def eft_validity_lambda_curve(m_chi_arr, *, omega_max, eft_kinematic_factor=1.0):
    return _shared_eft_kinematic_lambda_curve(
        m_chi_arr,
        omega_max=omega_max,
        eft_kinematic_factor=eft_kinematic_factor,
    )


def unitarity_lambda_curve(operator, m_chi_arr):
    return _shared_unitarity_lambda_curve(operator, m_chi_arr)


def lab_dsigma_prefactor(mchi, omega, theta):
    """
    Lab-frame two-body phase-space prefactor for gamma + chi(rest) -> gamma + chi.

    For target-at-rest Compton kinematics, dσ/dΩ_lab carries an extra
    (omega'/omega)^2 recoil Jacobian and m_chi^2 in the denominator.  Using
    the CM expression 1/(64π²s) directly with the lab scattering angle gives
    the wrong angular/energy weighting once omega/m_chi is not tiny.
    """
    ratio = lab_recoil_ratio(mchi, omega, theta)
    return ratio**2 / (64.0 * np.pi**2 * mchi**2)


# ---------- FERMIONIC DM OPERATORS ----------

def dsigma_dOmega_fermionic(mchi, theta, E_gamma, c_s, c_p, Lambda, operator="rayleigh_full", majorana=False):
    """
    Differential cross section for fermionic DM effective operators.
    
    Parameters
    ----------
    mchi : float
        DM mass [GeV]
    theta : array
        Scattering angle [radians]
    E_gamma : float
        Photon energy [GeV]
    c_s, c_p : float
        Wilson coefficients (scalar/pseudoscalar)
    Lambda : float
        EFT cutoff scale [GeV]
    operator : str
        Operator type: 'rayleigh_full', 'rayleigh_even', 'rayleigh_odd',
        'dipole_magnetic', 'dipole_electric', 'charge_radius', 'anapole'
    majorana : bool
        If True, enforces that dipole operators are zero (they vanish
        for Majorana fermions by charge-conjugation symmetry).
        Only anapole, rayleigh_even, rayleigh_odd are non-zero for Majorana DM.

    Returns
    -------
    dsigma/dOmega [fb/sr]

    def dsigma_dOmega_fermionic(mchi, theta, E_gamma, c_s, c_p, Lambda, operator="rayleigh_full"):
    
    Amplitude-squared conventions
    -----------------------------
    All |M|^2 expressions are summed (not averaged) over initial DM spins
    and summed over final spins, then divided by the flux factor 2s in the
    lab frame. The phase-space denominator is therefore 64 pi^2 s (not
    128 pi^2 s).

    Cross-check references (corrected 2026-06 against the project knowledge base;
    the earlier "Bernal, Hessler, Ibarra (2016) / 1607.06064", "Banks, Fornal,
    McKeen (2010) / 1007.1005" and "Fortin & Tait (2012) / 1103.3289" citations
    were placeholders NOT present in the literature library — 1607.06064 is in
    fact an ACTPol CMB-detector paper and 1007.1005 a CoGeNT/DAMA paper):
      - Operator |M|^2 framework: Arina et al. (2020), "Light and Darkness:
        consistently coupling dark matter to photons via effective operators",
        arXiv:2005.12789 (EPJC 81 (2021) 333), citekey DMPhotonCoupling2021 —
        dim 5-7 operator basis and cross sections used here.  [Verified 2026-06:
        the digitised limit data in constraints_data/ is attributed to this same
        paper, confirming it as the operator source.]
      - Direct-detection cross-check: Ibarra, Reichard & Tomar (2024), "Probing
        Dark Matter Electromagnetic Properties in Direct Detection Experiments",
        arXiv:2408.15760 (JCAP 2025) — all five EM operators incl. anapole and
        charge radius, from LZ + XENON1T + Migdal.
      - Dipole (MDM/EDM): Sigurdson et al. (2004), "Dark-Matter Electric and
        Magnetic Dipole Moments" (citekey Sigurdson2004); cf. Banks et al.
        (2010), citekey Banks2010.
      - Rayleigh (scalar & fermionic): Weiner & Yavin (2012 MiDM, 2013 UV
        completions); "Scalar Rayleigh Dark Matter" review.
    SPIN CONVENTION — NOT YET VERIFIED: the summed-vs-averaged choice per operator
    must match the |M|^2 normalisation in the operator framework paper
    (DMPhotonCoupling2021). If that reference *averages* over initial spins,
    insert 1/(2 s_chi + 1) = 1/2 per initial DM particle (spin-1/2 DM), times a
    further 1/2 for the photon initial-spin average.  [The previous note quoted
    1/(2 s_chi + 1)^2 = 1/4, the wrong spin count for a single initial DM particle.]
    """

    t = get_t_lab_DMrest(mchi, E_gamma, theta)
    phase = lab_dsigma_prefactor(mchi, E_gamma, theta)
    
    MAJORANA_FORBIDDEN = {"dipole_magnetic", "dipole_electric", "charge_radius"}
    if majorana and operator in MAJORANA_FORBIDDEN:
        raise ValueError(
            f"Operator '{operator}' vanishes for Majorana DM. "
            f"Use anapole, rayleigh_even, or rayleigh_odd instead."
        )

    if operator == "rayleigh_even":
        amp2 = c_s**2 * (4 * mchi**2 - t) * t**2 / (4.0 * Lambda**6)
        val = amp2 * phase
    elif operator == "rayleigh_odd":
        amp2 = c_p**2 * (-t)**3 / (4.0 * Lambda**6)
        val = amp2 * phase
    elif operator == "rayleigh_full":
        amp2 = (
            c_s**2 * (4 * mchi**2 - t) * t**2 / (4.0 * Lambda**6)
            + c_p**2 * (-t)**3 / (4.0 * Lambda**6)
        )
        val = amp2 * phase
    elif operator in ("dipole_magnetic", "dipole_electric"):
        c = c_s if operator == "dipole_magnetic" else c_p
        amp2 = 4.0 * ALPHA_EM * (c**2 / Lambda**2) * (-t)
        val = amp2 * phase
    elif operator in ("charge_radius", "anapole"):
        c = c_s if operator == "charge_radius" else c_p
        amp2 = 4.0 * ALPHA_EM * (c**2 / Lambda**4) * (t**2)
        val = amp2 * phase
    else:
        raise ValueError(f"Unknown operator: {operator}")
    
    return val * GEV2_TO_FB


# ---------- SCALAR DM OPERATORS ----------

def dsigma_dOmega_scalar(mchi, theta, E_gamma, c_phi, Lambda):
    """
    Differential cross section for scalar DM effective operator.
    
    Parameters
    ----------
    mchi : float
        DM mass [GeV]
    theta : array
        Scattering angle [radians]
    E_gamma : float
        Photon energy [GeV]
    c_phi : float
        Wilson coefficient
    Lambda : float
        EFT cutoff scale [GeV]
    
    Returns
    -------
    dsigma/dOmega [fb/sr]
    """
    t = get_t_lab_DMrest(mchi, E_gamma, theta)
    phase = lab_dsigma_prefactor(mchi, E_gamma, theta)
    
    amp2 = c_phi**2 * t**2 / (4.0 * Lambda**4)
    val = amp2 * phase
    return val * GEV2_TO_FB


# ---------- TOTAL CROSS SECTIONS ----------

def sigma_tot_fermionic(E_gamma, mchi, c_s, c_p, Lambda, operator="rayleigh_full", majorana=False, n_theta=200):
    """Total cross section for fermionic DM [cm^2]."""
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)
    
    dsig = dsigma_dOmega_fermionic(mchi, theta_vals, E_gamma, c_s, c_p, Lambda, operator, majorana=majorana)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    
    sig = 2.0 * np.pi * np.trapezoid(dsig, cos_vals) * FB_TO_CM2
    return sig


def sigma_tot_fermionic_array(E_gamma_arr, mchi, c_s, c_p, Lambda, operator="rayleigh_full", majorana=False, n_theta=200):
    """Vectorized total cross section for fermionic DM [cm^2] over an energy array."""
    E_gamma_arr = np.asarray(E_gamma_arr, dtype=float)
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)

    # Broadcast to (nE, nTheta)
    E2 = E_gamma_arr[:, None]
    th2 = theta_vals[None, :]
    dsig = dsigma_dOmega_fermionic(mchi, th2, E2, c_s, c_p, Lambda, operator, majorana=majorana)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    dsig = np.where(dsig > 0.0, dsig, 0.0)

    sig_fb = 2.0 * np.pi * np.trapezoid(dsig, cos_vals, axis=1)
    return sig_fb * FB_TO_CM2


def sigma_tot_scalar(E_gamma, mchi, c_phi, Lambda, n_theta=200):
    """Total cross section for scalar DM [cm^2]."""
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)
    
    dsig = dsigma_dOmega_scalar(mchi, theta_vals, E_gamma, c_phi, Lambda)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    
    sig = 2.0 * np.pi * np.trapezoid(dsig, cos_vals) * FB_TO_CM2
    return sig


def sigma_tot_scalar_array(E_gamma_arr, mchi, c_phi, Lambda, n_theta=200):
    """Vectorized total cross section for scalar DM [cm^2] over an energy array."""
    E_gamma_arr = np.asarray(E_gamma_arr, dtype=float)
    cos_vals = np.linspace(-1.0, COS_THETA_MAX, n_theta)
    theta_vals = np.arccos(cos_vals)

    E2 = E_gamma_arr[:, None]
    th2 = theta_vals[None, :]
    dsig = dsigma_dOmega_scalar(mchi, th2, E2, c_phi, Lambda)
    dsig = np.nan_to_num(dsig, nan=0.0, posinf=0.0, neginf=0.0)
    dsig = np.where(dsig > 0.0, dsig, 0.0)

    sig_fb = 2.0 * np.pi * np.trapezoid(dsig, cos_vals, axis=1)
    return sig_fb * FB_TO_CM2


def _roi_prefactor_key(l_grid, b_grid):
    l_grid = np.asarray(l_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    return (
        tuple(np.round(l_grid, 8).tolist()),
        tuple(np.round(b_grid, 8).tolist()),
    )


def roi_tau_prefactor(l_grid, b_grid, n_points_J=500):
    """
    Return K = sum_{ROI} (J1 * J2) / sum_{ROI} J2  [GeV/cm^2]

    where J1 = int rho dl  (column density, linear in rho — enters tau)
          J2 = int rho^2 dl (emissivity — enters the NFW-rho^2 halo flux)

    The rho^2 weighting is intentional: Totani's halo excess photons are
    produced by DM annihilation (rho^2 emissivity), so high-emissivity
    lines of sight contribute more to the observed flux and should receive
    proportionally more weight in the mean optical depth.

    tau_bar(E) = K * sigma(E) / m_chi
    """
    key = _roi_prefactor_key(l_grid, b_grid)
    cached = _ROI_PREFAC_CACHE.get(key)
    if cached is not None:
        return float(cached)

    l_grid = np.asarray(l_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    sum_J2 = 0.0
    sum_J1J2 = 0.0

    for b in b_grid:
        for l in l_grid:
            J2 = float(compute_J_los(float(l), float(b), power=2, n_points=int(n_points_J)))
            if not np.isfinite(J2) or J2 <= 0.0:
                continue

            L = los_length_cm(float(l), float(b))
            ell = np.linspace(0.0, L, int(n_points_J))
            r = galactocentric_radius_cm(ell, float(l), float(b))
            rho = nfw_density_GeV_cm3(r)
            J1 = float(np.trapezoid(rho, ell))

            if not np.isfinite(J1) or J1 <= 0.0:
                continue

            sum_J2 += J2
            sum_J1J2 += (J1 * J2)

    if sum_J2 <= 0.0:
        raise ValueError("ROI prefactor failed: sum_J2 <= 0.")

    K = sum_J1J2 / sum_J2
    _ROI_PREFAC_CACHE[key] = float(K)
    return float(K)


# =============================================================================
# OPTICAL DEPTH CALCULATIONS
# =============================================================================

def compute_tau_los_eft(l_deg, b_deg, E_GeV, m_chi_GeV, Lambda_GeV, 
                        dm_type="fermionic", operator="rayleigh_full",
                        c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False, n_los=300):
    """
    Optical depth along a single line of sight for EFT operators.
    
    tau = (1/m_chi) * int rho(r(ell)) * sigma dl
    """
    if dm_type == "fermionic":
        sig = sigma_tot_fermionic(E_GeV, m_chi_GeV, c_s, c_p, Lambda_GeV, operator, majorana=majorana)
    else:  # scalar
        sig = sigma_tot_scalar(E_GeV, m_chi_GeV, c_phi, Lambda_GeV)
    
    L   = los_length_cm(l_deg, b_deg)
    ell = np.linspace(0.0, L, n_los)
    r   = galactocentric_radius_cm(ell, l_deg, b_deg)
    rho = nfw_density_GeV_cm3(r)
    
    J_los = np.trapezoid(rho, ell)  # GeV/cm^2
    tau = J_los * sig / m_chi_GeV
    
    return tau


def compute_tau_bar_eft(E_GeV, m_chi_GeV, Lambda_GeV, l_grid, b_grid,
                        dm_type="fermionic", operator="rayleigh_full",
                        c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """
    Emissivity-weighted average optical depth over the ROI.
    
    tau_bar(E) = sum_{l,b} [tau(l,b) * J2(l,b)] / sum_{l,b} J2(l,b)
    """
    if dm_type == "fermionic":
        sig = sigma_tot_fermionic(E_GeV, m_chi_GeV, c_s, c_p, Lambda_GeV, operator, majorana=majorana)
    else:
        sig = sigma_tot_scalar(E_GeV, m_chi_GeV, c_phi, Lambda_GeV)

    K = roi_tau_prefactor(l_grid, b_grid)
    return float(K) * float(sig) / float(m_chi_GeV)


def compute_tau_bar_spectrum_eft(E_bins, m_chi_GeV, Lambda_GeV, l_grid, b_grid,
                                  dm_type="fermionic", operator="rayleigh_full",
                                  c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False,
                                  K_override=None):
    """Compute tau_bar at each energy bin.

    Parameters
    ----------
    K_override : float or None, optional
        Scalar column-density prefactor K [GeV/cm^2] to use in place of the
        emissivity-weighted ROI average computed from ``l_grid``/``b_grid``.
        When set, the ROI integration is bypassed entirely. Use this when the
        line-of-sight column is a well-defined single scalar (dwarf spheroidals
        with a catalog J-factor; the cosmological IGRB baseline). The default
        ``None`` preserves the original halo-ROI behaviour via
        :func:`roi_tau_prefactor`.
    """
    E_bins = np.asarray(E_bins, dtype=float)
    if dm_type == "fermionic":
        sig_arr = sigma_tot_fermionic_array(E_bins, m_chi_GeV, c_s, c_p, Lambda_GeV, operator, majorana=majorana)
    else:
        sig_arr = sigma_tot_scalar_array(E_bins, m_chi_GeV, c_phi, Lambda_GeV)

    if K_override is not None:
        K = float(K_override)
    else:
        K = roi_tau_prefactor(l_grid, b_grid)
    tau_bar = (float(K) / float(m_chi_GeV)) * np.asarray(sig_arr, dtype=float)
    tau_bar = np.nan_to_num(tau_bar, nan=0.0, posinf=0.0, neginf=0.0)
    tau_bar = np.where(tau_bar > 0.0, tau_bar, 0.0)
    return tau_bar


# =============================================================================
# CHI-SQUARED AND CONSTRAINTS
# =============================================================================

def compute_attenuated_spectrum(tau_bar, phi_totani=PHI_TOTANI):
    """Apply scattering attenuation: Phi_att(E) = Phi_Totani(E) * exp(-tau_bar(E))."""
    return phi_totani * np.exp(-tau_bar)


def compute_chi2(tau_bar, phi_totani=PHI_TOTANI, sigma_totani=SIGMA_TOTANI):
    """Chi-squared between attenuated and observed halo flux."""
    phi_att = compute_attenuated_spectrum(tau_bar, phi_totani)
    mask = phi_totani > 0
    assert np.all(phi_att[mask] >= 0), "Attenuated flux went negative in valid bins"
    residuals = (phi_att[mask] - phi_totani[mask]) / sigma_totani[mask]
    return np.sum(residuals**2)


def chi2_grid_scan_eft(m_chi_arr, Lambda_arr, E_bins, l_grid, b_grid,
                       dm_type="fermionic", operator="rayleigh_full",
                       c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """
    Scan (m_chi, Lambda) parameter space and compute chi^2 at each point.
    
    Returns chi2_grid of shape (len(m_chi_arr), len(Lambda_arr)).
    """
    chi2_grid = np.zeros((len(m_chi_arr), len(Lambda_arr)))
    
    for i, m_chi in enumerate(m_chi_arr):
        for j, Lambda in enumerate(Lambda_arr):
            tau_bar = compute_tau_bar_spectrum_eft(
                E_bins, m_chi, Lambda, l_grid, b_grid,
                dm_type=dm_type, operator=operator,
                c_s=c_s, c_p=c_p, c_phi=c_phi, majorana=majorana
            )
            chi2_grid[i, j] = compute_chi2(tau_bar)
            
            if (i * len(Lambda_arr) + j) % 10 == 0:
                pct = 100 * (i * len(Lambda_arr) + j) / (len(m_chi_arr) * len(Lambda_arr))
                print(f"Progress: {pct:.1f}%")
    
    return chi2_grid


def _operator_metadata(dm_type, operator):
    if dm_type == "scalar" and operator == "rayleigh":
        key = "scalar_rayleigh"
    else:
        key = operator
    return OPERATOR_METADATA.get(key, {
        "paper_label": operator,
        "lambda_power": None,
        "coefficient_power": None,
    })


def _effective_coefficient(dm_type, operator, c_s=1.0, c_p=1.0, c_phi=1.0):
    if dm_type == "scalar":
        return float(c_phi)
    if operator in ("dipole_magnetic", "charge_radius", "rayleigh_even"):
        return float(c_s)
    if operator in ("dipole_electric", "anapole", "rayleigh_odd"):
        return float(c_p)
    return 1.0


def _paper_y_axis_values(Lambda_arr, dm_type, operator, c_s=1.0, c_p=1.0, c_phi=1.0):
    meta = _operator_metadata(dm_type, operator)
    coeff_power = meta.get("coefficient_power", None)
    if coeff_power is None:
        return np.asarray(Lambda_arr, dtype=float)
    coeff = _effective_coefficient(dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi)
    if coeff <= 0:
        raise ValueError("Operator coefficient must be positive for paper-style rescaling.")
    return np.asarray(Lambda_arr, dtype=float) / (coeff ** coeff_power)


def extract_90cl_boundary(m_chi_arr, Lambda_arr, chi2_grid):
    """
    Extract the 90% CL lower-limit boundary from the actual
    matplotlib contour used in the chi2 plot.

    Points with Delta-chi^2 >= 4.61 are taken to be excluded. Since the
    EFT cross sections grow as Lambda decreases, the excluded region is
    the area below the 90% CL contour in the (m_chi, Lambda) plane.
    We therefore extract the upper envelope of the Delta-chi^2 = 4.61
    contour directly, so the saved boundary matches the plotted contour.
    """
    threshold = 4.61
    chi2_min = np.min(chi2_grid)
    dchi2 = chi2_grid - chi2_min

    M, L = np.meshgrid(m_chi_arr, Lambda_arr, indexing='ij')
    fig, ax = plt.subplots()
    cs = ax.contour(np.log10(M), np.log10(L), dchi2, levels=[threshold])
    segments = cs.allsegs[0] if cs.allsegs else []
    plt.close(fig)

    if not segments:
        return np.empty((0, 2))

    x_eval = np.linspace(np.log10(np.min(m_chi_arr)), np.log10(np.max(m_chi_arr)), 800)
    y_best = np.full_like(x_eval, np.nan, dtype=float)

    for seg in segments:
        seg = np.asarray(seg, dtype=float)
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue

        x = seg[:, 0]
        y = seg[:, 1]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        if x_unique.size < 2:
            continue

        mask = (x_eval >= x_unique.min()) & (x_eval <= x_unique.max())
        if not np.any(mask):
            continue

        y_interp = np.interp(x_eval[mask], x_unique, y_unique)
        current = y_best[mask]
        y_best[mask] = np.where(np.isnan(current), y_interp, np.maximum(current, y_interp))

    good = np.isfinite(y_best)
    if not np.any(good):
        return np.empty((0, 2))

    return np.column_stack((10 ** x_eval[good], 10 ** y_best[good]))


def save_boundary_npz(boundary, dm_type, operator, c_s=1.0, c_p=1.0, c_phi=1.0,
                      majorana=False, *, omega_max_for_validity=None, eft_kinematic_factor=1.0):
    """
    Save a 90% CL boundary in both raw Lambda and paper-style Lambda/C^(1/n) form.
    Uses compressed format and float32 to minimize file size.
    """
    outdir = Path(__file__).resolve().parent.parent / "constraint_boundaries"
    outdir.mkdir(exist_ok=True)
    meta = _operator_metadata(dm_type, operator)
    if boundary.size == 0:
        return None

    lambda_raw = boundary[:, 1]
    lambda_plot = _paper_y_axis_values(
        lambda_raw, dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
    )
    suffix = "_majorana" if majorana else ""
    outpath = outdir / f"totani_{dm_type}_{operator}{suffix}_90cl.npz"
    np.savez_compressed(
        outpath,
        mchi_GeV=boundary[:, 0].astype(np.float32),
        lambda_GeV=lambda_raw.astype(np.float32),
        lambda_plot_GeV=lambda_plot.astype(np.float32),
        paper_label=meta["paper_label"],
        dm_type=dm_type,
        operator=operator,
        coefficient_power=meta["coefficient_power"],
        c_s=np.float32(c_s),
        c_p=np.float32(c_p),
        c_phi=np.float32(c_phi),
        majorana=majorana,
        omega_max_for_validity=np.float32(omega_max_for_validity) if omega_max_for_validity is not None else np.float32(np.nan),
        eft_kinematic_factor=np.float32(eft_kinematic_factor),
        validity_guides="kinematic_eft_and_unitarity",
        boundary_extraction="unfiltered_scan_contour",
    )
    print(f"Saved boundary: {outpath}")
    return outpath


def load_fermi_spectrum_energies(default_path=None):
    if default_path is None:
        default_path = Path(__file__).resolve().parent.parent.parent / "fermi_data" / "york" / "processed" / "spectrum_data.txt"
    arr = np.loadtxt(str(default_path))
    return np.asarray(arr[:, 0], dtype=float)


def extract_tau_needed_boundary(m_chi_arr, Lambda_arr, E_bins, l_grid, b_grid, *,
                                tau_needed,
                                dm_type="fermionic", operator="rayleigh_full",
                                c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """Extract a naive sensitivity boundary defined by max_E tau_bar(E) >= tau_needed.

    For fixed m_chi, tau decreases monotonically with increasing Lambda, so we
    extract the crossing tau_max(Lambda) = tau_needed in log-log space.
    """
    tau_needed = float(tau_needed)
    m_chi_arr = np.asarray(m_chi_arr, dtype=float)
    Lambda_arr = np.asarray(Lambda_arr, dtype=float)
    E_bins = np.asarray(E_bins, dtype=float)

    boundary = []
    for m_chi in m_chi_arr:
        tau_max_vs_lambda = np.zeros(len(Lambda_arr), dtype=float)
        for j, Lambda in enumerate(Lambda_arr):
            tau_bar = compute_tau_bar_spectrum_eft(
                E_bins, float(m_chi), float(Lambda), l_grid, b_grid,
                dm_type=dm_type, operator=operator,
                c_s=c_s, c_p=c_p, c_phi=c_phi, majorana=majorana,
            )
            tau_max_vs_lambda[j] = float(np.max(np.asarray(tau_bar, dtype=float))) if len(tau_bar) else 0.0

        good = np.isfinite(tau_max_vs_lambda) & (tau_max_vs_lambda > 0.0)
        if not np.any(good):
            continue

        tau_vals = np.asarray(tau_max_vs_lambda, dtype=float)
        meets = tau_vals >= tau_needed
        if not np.any(meets):
            continue

        idx_last = int(np.max(np.where(meets)[0]))
        if idx_last >= (len(Lambda_arr) - 1):
            boundary.append((float(m_chi), float(Lambda_arr[idx_last])))
            continue

        Lam0 = float(Lambda_arr[idx_last])
        Lam1 = float(Lambda_arr[idx_last + 1])
        t0 = float(tau_vals[idx_last])
        t1 = float(tau_vals[idx_last + 1])

        if t0 <= 0.0 or t1 <= 0.0 or not np.isfinite(t0) or not np.isfinite(t1) or (t0 == t1):
            boundary.append((float(m_chi), float(Lam0)))
            continue

        x0 = np.log10(Lam0)
        x1 = np.log10(Lam1)
        y0 = np.log10(t0)
        y1 = np.log10(t1)
        yT = np.log10(max(tau_needed, 1e-300))
        xT = x0 + (x1 - x0) * (yT - y0) / (y1 - y0)
        boundary.append((float(m_chi), float(10 ** xT)))

    return np.asarray(boundary, dtype=float)


def save_naive_boundary_npz(boundary, dm_type, operator, *, dip_depth,
                            c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False,
                            omega_max_for_validity=None, eft_kinematic_factor=1.0):
    outdir = Path(__file__).resolve().parent.parent / "constraint_boundaries"
    outdir.mkdir(exist_ok=True)
    meta = _operator_metadata(dm_type, operator)
    if boundary.size == 0:
        return None

    lambda_raw = boundary[:, 1]
    lambda_plot = _paper_y_axis_values(
        lambda_raw, dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
    )
    suffix = "_majorana" if majorana else ""
    outpath = outdir / f"fermi_naive_{dm_type}_{operator}{suffix}_{float(dip_depth):g}.npz"
    np.savez_compressed(
        outpath,
        mchi_GeV=boundary[:, 0].astype(np.float32),
        lambda_GeV=lambda_raw.astype(np.float32),
        lambda_plot_GeV=lambda_plot.astype(np.float32),
        paper_label=meta["paper_label"],
        dm_type=dm_type,
        operator=operator,
        coefficient_power=meta["coefficient_power"],
        c_s=np.float32(c_s),
        c_p=np.float32(c_p),
        c_phi=np.float32(c_phi),
        dip_depth=np.float32(dip_depth),
        tau_needed=np.float32(-np.log(1.0 - float(dip_depth))) if float(dip_depth) > 0 else np.float32(0.0),
        majorana=majorana,
        omega_max_for_validity=np.float32(omega_max_for_validity) if omega_max_for_validity is not None else np.float32(np.nan),
        eft_kinematic_factor=np.float32(eft_kinematic_factor),
        validity_guides="kinematic_eft_and_unitarity",
        boundary_extraction="naive_tau_threshold",
    )
    print(f"Saved boundary: {outpath}")
    return outpath


def run_naive_fermi_constraints(m_chi_arr, Lambda_arr, *,
                               dip_depth=0.01,
                               E_bins=None,
                               dm_type="fermionic", operator="rayleigh_full",
                               c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False,
                               eft_kinematic_factor=1.0,
                               l_grid=None, b_grid=None):
    if E_bins is None:
        E_bins = load_fermi_spectrum_energies()
    if l_grid is None:
        l_grid = np.linspace(-60, 60, 15)
    if b_grid is None:
        b_grid = np.concatenate([
            np.linspace(-60, -10, 8),
            np.linspace(10, 60, 8),
        ])

    dip_depth = float(dip_depth)
    dip_depth = min(max(dip_depth, 0.0), 0.999999)
    tau_needed = -np.log(1.0 - dip_depth) if dip_depth > 0 else 0.0

    boundary = extract_tau_needed_boundary(
        m_chi_arr,
        Lambda_arr,
        E_bins,
        l_grid,
        b_grid,
        tau_needed=tau_needed,
        dm_type=dm_type,
        operator=operator,
        c_s=c_s,
        c_p=c_p,
        c_phi=c_phi,
        majorana=majorana,
    )
    return save_naive_boundary_npz(
        boundary,
        dm_type,
        operator,
        dip_depth=dip_depth,
        c_s=c_s,
        c_p=c_p,
        c_phi=c_phi,
        majorana=majorana,
        omega_max_for_validity=float(np.max(E_bins)),
        eft_kinematic_factor=float(eft_kinematic_factor),
    )


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_chi2_contours_eft(m_chi_arr, Lambda_arr, chi2_grid, 
                           dm_type="fermionic", operator="rayleigh_full",
                           c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False):
    """Plot Delta-chi^2 contours in (m_chi, Lambda) space."""
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=12, linewidth=1.8, n_colors=10)
    chi2_min  = np.min(chi2_grid)
    dchi2     = chi2_grid - chi2_min
    
    fig, ax = plt.subplots(figsize=(9, 6))
    M, L = np.meshgrid(m_chi_arr, Lambda_arr, indexing='ij')
    
    # Filled contours
    levels_fill = np.linspace(0, min(50, np.max(dchi2)), 50)
    cf = ax.contourf(np.log10(M), np.log10(L), dchi2,
                     levels=levels_fill, cmap='plasma', alpha=0.8)
    
    # Confidence level contours
    contour_levels = [4.61, 9.21]
    cs = ax.contour(np.log10(M), np.log10(L), dchi2,
                    levels=contour_levels, colors='red', linewidths=2.5)
    
    ax.clabel(cs, contour_levels, inline=True, fontsize=11,
              fmt={4.61: '90% CL', 9.21: '99% CL'})
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, label=r'$\Delta \chi^2$')
    cbar.ax.tick_params(labelsize=11)
    
    # Labels
    ax.set_xlabel(r'$\log_{10}(m_\chi / \mathrm{GeV})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\Lambda / \mathrm{GeV})$', fontsize=14)
    
    suffix = ' (Majorana)' if majorana else ''
    title = f'{dm_type.capitalize()} DM: {operator} operator{suffix}'
    ax.set_title(f'Constraint from Fermi halo excess\n{title}', fontsize=13, pad=10)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    suffix_file = '_majorana' if majorana else ''
    filename = f'chi2_contours_{dm_type}_{operator}{suffix_file}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', pil_kwargs={'optimize': True}, **current_savefig_kwargs())
    plt.close()
    print(f"Saved: {filename}")


def plot_exclusion_curve_eft(m_chi_arr, Lambda_arr, chi2_grid,
                              dm_type="fermionic", operator="rayleigh_full",
                              c_s=1.0, c_p=1.0, c_phi=1.0, majorana=False,
                              omega_max_for_validity=None, eft_kinematic_factor=1.0):
    """Extract and plot 90% CL exclusion boundary."""
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=12, linewidth=1.8, n_colors=10)
    Lambda_upper_90 = extract_90cl_boundary(m_chi_arr, Lambda_arr, chi2_grid)
    if omega_max_for_validity is None:
        omega_max_for_validity = float(np.max(E_BINS_GEV))
    save_boundary_npz(
        Lambda_upper_90,
        dm_type,
        operator,
        c_s=c_s,
        c_p=c_p,
        c_phi=c_phi,
        majorana=majorana,
        omega_max_for_validity=float(omega_max_for_validity),
        eft_kinematic_factor=float(eft_kinematic_factor),
    )
    lambda_plot_arr = _paper_y_axis_values(
        Lambda_arr, dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
    )
    if len(Lambda_upper_90) > 0:
        lambda_plot_boundary = _paper_y_axis_values(
            Lambda_upper_90[:, 1], dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
        )
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Exclusion region (below the curve is excluded for EFT)
        ax.fill_between(np.log10(Lambda_upper_90[:, 0]),
                        np.log10(lambda_plot_arr[0]),
                        np.log10(lambda_plot_boundary),
                        alpha=0.3, color='red', label='Excluded (90% CL)')
        
        # Boundary line
        ax.plot(np.log10(Lambda_upper_90[:, 0]),
                np.log10(lambda_plot_boundary),
                'r-', lw=2.5, label='90% CL lower limit')

        lam_kin = eft_validity_lambda_curve(
            m_chi_arr,
            omega_max=float(omega_max_for_validity),
            eft_kinematic_factor=float(eft_kinematic_factor),
        )
        kin_good = np.isfinite(lam_kin) & (lam_kin > 0.0)
        if np.any(kin_good):
            kin_plot = _paper_y_axis_values(
                lam_kin[kin_good], dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
            )
            ax.plot(
                np.log10(np.asarray(m_chi_arr, dtype=float)[kin_good]),
                np.log10(kin_plot),
                color='magenta', lw=1.8, ls='--',
                label='Kinematic EFT validity',
            )

        lam_unit = unitarity_lambda_curve(operator, m_chi_arr)
        unit_good = np.isfinite(lam_unit) & (lam_unit > 0.0)
        if np.any(unit_good):
            unit_plot = _paper_y_axis_values(
                lam_unit[unit_good], dm_type, operator, c_s=c_s, c_p=c_p, c_phi=c_phi
            )
            ax.plot(
                np.log10(np.asarray(m_chi_arr, dtype=float)[unit_good]),
                np.log10(unit_plot),
                color='0.6', lw=1.4, ls=':',
                label='Unitarity guide',
            )
        
        # Formatting
        ax.set_xlabel(r'$\log_{10}(m_\chi / \mathrm{GeV})$', fontsize=14)
        meta = _operator_metadata(dm_type, operator)
        cpow = meta.get("coefficient_power", None)
        if cpow == 0.5:
            ylabel = r'$\log_{10}(\Lambda / C^{1/2})\ [\mathrm{GeV}]$'
        elif cpow == 1.0 / 3.0:
            ylabel = r'$\log_{10}(\Lambda / C^{1/3})\ [\mathrm{GeV}]$'
        else:
            ylabel = r'$\log_{10}(\Lambda / \mathrm{GeV})$'
        ax.set_ylabel(ylabel, fontsize=14)
        
        suffix = ' (Majorana)' if majorana else ''
        title = f'{dm_type.capitalize()} DM: {operator} operator{suffix}'
        ax.set_title(f'Exclusion from Fermi halo excess\n{title}',
                     fontsize=13, pad=10)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        
        ax.set_xlim(np.log10(m_chi_arr[0]), np.log10(m_chi_arr[-1]))
        ax.set_ylim(np.log10(lambda_plot_arr[0]), np.log10(lambda_plot_arr[-1]))
        
        plt.tight_layout()
        suffix_file = '_majorana' if majorana else ''
        filename = f'exclusion_curve_{dm_type}_{operator}{suffix_file}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', pil_kwargs={'optimize': True}, **current_savefig_kwargs())
        plt.close()
        print(f"Saved: {filename}")
    else:
        print("Warning: No 90% CL exclusion boundary found")


# =============================================================================
# DRIVER: RUN SCANS
# =============================================================================

if __name__ == "__main__":
    
    # Grid over Totani's ROI
    l_grid = np.linspace(-60, 60, 15)
    b_grid = np.concatenate([
        np.linspace(-60, -10, 8),
        np.linspace( 10,  60, 8)
    ])
    
    print("=" * 70)
    print("EFT OPERATOR CONSTRAINTS FROM FERMI HALO EXCESS")
    print("=" * 70)
    
    # ========== SCALAR DM ==========
    print("\n" + "=" * 70)
    print("SCALAR DM: Rayleigh-like operator (c_phi = 1)")
    print("=" * 70)
    
    m_chi_arr_scalar = np.logspace(-6, 20, 50)
    Lambda_arr_scalar = np.logspace(-3, 6, 40)   # 100 GeV to 1 PeV
    
    chi2_scalar = chi2_grid_scan_eft(
        m_chi_arr_scalar, Lambda_arr_scalar, E_BINS_GEV, l_grid, b_grid,
        dm_type="scalar", operator="rayleigh", c_phi=1.0
    )
    
    plot_chi2_contours_eft(m_chi_arr_scalar, Lambda_arr_scalar, chi2_scalar,
                           dm_type="scalar", operator="rayleigh", c_phi=1.0)
    plot_exclusion_curve_eft(m_chi_arr_scalar, Lambda_arr_scalar, chi2_scalar,
                             dm_type="scalar", operator="rayleigh", c_phi=1.0)
    run_naive_fermi_constraints(
        m_chi_arr_scalar,
        Lambda_arr_scalar,
        dip_depth=0.01,
        E_bins=load_fermi_spectrum_energies(),
        dm_type="scalar",
        operator="rayleigh",
        c_phi=1.0,
        majorana=False,
    )
    
    # ========== FERMIONIC DM: RAYLEIGH OPERATOR ==========
    print("\n" + "=" * 70)
    print("FERMIONIC DM: Rayleigh operator (c_s = c_p = 1)")
    print("=" * 70)
    
    m_chi_arr_ferm   = np.logspace(-6, 20, 50)
    Lambda_arr_ferm  = np.logspace(-3,  7, 40)
    
    chi2_ferm_rayleigh = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="rayleigh_full", c_s=1.0, c_p=1.0, majorana=False
    )
    
    plot_chi2_contours_eft(m_chi_arr_ferm, Lambda_arr_ferm, chi2_ferm_rayleigh,
                           dm_type="fermionic", operator="rayleigh_full",
                           c_s=1.0, c_p=1.0, majorana=False)
    plot_exclusion_curve_eft(m_chi_arr_ferm, Lambda_arr_ferm, chi2_ferm_rayleigh,
                             dm_type="fermionic", operator="rayleigh_full", majorana=False)
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_ferm,
        dip_depth=0.01,
        E_bins=load_fermi_spectrum_energies(),
        dm_type="fermionic",
        operator="rayleigh_full",
        c_s=1.0,
        c_p=1.0,
        majorana=False,
    )

    print("\n" + "=" * 70)
    print("FERMIONIC DM: Rayleigh-even")
    print("=" * 70)
    chi2_ferm_rayleigh_even = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="rayleigh_even", c_s=1.0, c_p=0.0, majorana=False
    )
    plot_chi2_contours_eft(m_chi_arr_ferm, Lambda_arr_ferm, chi2_ferm_rayleigh_even,
                        dm_type="fermionic", operator="rayleigh_even",
                        c_s=1.0, c_p=0.0, majorana=False)
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_ferm_rayleigh_even,
        dm_type="fermionic", operator="rayleigh_even", c_s=1.0, c_p=0.0, majorana=False
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_ferm,
        dip_depth=0.01,
        E_bins=load_fermi_spectrum_energies(),
        dm_type="fermionic",
        operator="rayleigh_even",
        c_s=1.0,
        c_p=0.0,
        majorana=False,
    )

    print("\n" + "=" * 70)
    print("FERMIONIC DM: Rayleigh-odd")
    print("=" * 70)
    chi2_ferm_rayleigh_odd = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="rayleigh_odd", c_s=0.0, c_p=1.0, majorana=False
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_ferm_rayleigh_odd,
        dm_type="fermionic", operator="rayleigh_odd", c_s=0.0, c_p=1.0, majorana=False
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_ferm_rayleigh_odd,
        dm_type="fermionic", operator="rayleigh_odd", c_s=0.0, c_p=1.0, majorana=False
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_ferm,
        dip_depth=0.01,
        E_bins=load_fermi_spectrum_energies(),
        dm_type="fermionic",
        operator="rayleigh_odd",
        c_s=0.0,
        c_p=1.0,
        majorana=False,
    )
    
    # ========== FERMIONIC DM: MAGNETIC DIPOLE ==========
    print("\n" + "=" * 70)
    print("FERMIONIC DM: Magnetic dipole (c_s = 1)")
    print("=" * 70)
    
    Lambda_arr_dipole = np.logspace(-3,  7, 40)  # 1 GeV to 10 TeV
    
    chi2_ferm_dipole = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="dipole_magnetic", c_s=1.0, c_p=0.0, majorana=False
    )
    
    plot_chi2_contours_eft(m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_dipole,
                           dm_type="fermionic", operator="dipole_magnetic",
                           c_s=1.0, c_p=0.0, majorana=False)
    plot_exclusion_curve_eft(m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_dipole,
                             dm_type="fermionic", operator="dipole_magnetic",
                             c_s=1.0, c_p=0.0, majorana=False)

    print("\n" + "=" * 70)
    print("FERMIONIC DM: Electric dipole (c_p = 1)")
    print("=" * 70)
    chi2_ferm_edm = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="dipole_electric", c_s=0.0, c_p=1.0, majorana=False
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_edm,
        dm_type="fermionic", operator="dipole_electric",
        c_s=0.0, c_p=1.0, majorana=False
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_edm,
        dm_type="fermionic", operator="dipole_electric",
        c_s=0.0, c_p=1.0, majorana=False
    )

    print("\n" + "=" * 70)
    print("FERMIONIC DM: Charge radius (c_s = 1)")
    print("=" * 70)
    chi2_ferm_cr = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="charge_radius", c_s=1.0, c_p=0.0, majorana=False
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_cr,
        dm_type="fermionic", operator="charge_radius",
        c_s=1.0, c_p=0.0, majorana=False
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_cr,
        dm_type="fermionic", operator="charge_radius",
        c_s=1.0, c_p=0.0, majorana=False
    )

    print("\n" + "=" * 70)
    print("FERMIONIC DM: Anapole (c_p = 1)")
    print("=" * 70)
    chi2_ferm_anapole = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="anapole", c_s=0.0, c_p=1.0, majorana=False
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_anapole,
        dm_type="fermionic", operator="anapole",
        c_s=0.0, c_p=1.0, majorana=False
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_ferm_anapole,
        dm_type="fermionic", operator="anapole",
        c_s=0.0, c_p=1.0, majorana=False
    )

    E_bins_fermi = load_fermi_spectrum_energies()
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_dipole,
        dip_depth=0.01,
        E_bins=E_bins_fermi,
        dm_type="fermionic",
        operator="dipole_magnetic",
        c_s=1.0,
        c_p=0.0,
        majorana=False,
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_dipole,
        dip_depth=0.01,
        E_bins=E_bins_fermi,
        dm_type="fermionic",
        operator="dipole_electric",
        c_s=0.0,
        c_p=1.0,
        majorana=False,
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_dipole,
        dip_depth=0.01,
        E_bins=E_bins_fermi,
        dm_type="fermionic",
        operator="charge_radius",
        c_s=1.0,
        c_p=0.0,
        majorana=False,
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_dipole,
        dip_depth=0.01,
        E_bins=E_bins_fermi,
        dm_type="fermionic",
        operator="anapole",
        c_s=0.0,
        c_p=1.0,
        majorana=False,
    )

    # ========== MAJORANA FERMIONIC DM ==========
    print("\n" + "=" * 70)
    print("MAJORANA FERMIONIC DM: Rayleigh-even")
    print("=" * 70)
    chi2_majorana_rayleigh_even = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="rayleigh_even", c_s=1.0, c_p=0.0, majorana=True
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_majorana_rayleigh_even,
        dm_type="fermionic", operator="rayleigh_even", c_s=1.0, c_p=0.0, majorana=True
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_majorana_rayleigh_even,
        dm_type="fermionic", operator="rayleigh_even", c_s=1.0, c_p=0.0, majorana=True
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_ferm,
        dip_depth=0.01,
        E_bins=load_fermi_spectrum_energies(),
        dm_type="fermionic",
        operator="rayleigh_even",
        c_s=1.0,
        c_p=0.0,
        majorana=True,
    )

    print("\n" + "=" * 70)
    print("MAJORANA FERMIONIC DM: Rayleigh-odd")
    print("=" * 70)
    chi2_majorana_rayleigh_odd = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="rayleigh_odd", c_s=0.0, c_p=1.0, majorana=True
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_majorana_rayleigh_odd,
        dm_type="fermionic", operator="rayleigh_odd", c_s=0.0, c_p=1.0, majorana=True
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_ferm, chi2_majorana_rayleigh_odd,
        dm_type="fermionic", operator="rayleigh_odd", c_s=0.0, c_p=1.0, majorana=True
    )
    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_ferm,
        dip_depth=0.01,
        E_bins=load_fermi_spectrum_energies(),
        dm_type="fermionic",
        operator="rayleigh_odd",
        c_s=0.0,
        c_p=1.0,
        majorana=True,
    )

    print("\n" + "=" * 70)
    print("MAJORANA FERMIONIC DM: Anapole")
    print("=" * 70)
    chi2_majorana_anapole = chi2_grid_scan_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, E_BINS_GEV, l_grid, b_grid,
        dm_type="fermionic", operator="anapole", c_s=0.0, c_p=1.0, majorana=True
    )
    plot_chi2_contours_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_majorana_anapole,
        dm_type="fermionic", operator="anapole", c_s=0.0, c_p=1.0, majorana=True
    )
    plot_exclusion_curve_eft(
        m_chi_arr_ferm, Lambda_arr_dipole, chi2_majorana_anapole,
        dm_type="fermionic", operator="anapole", c_s=0.0, c_p=1.0, majorana=True
    )

    run_naive_fermi_constraints(
        m_chi_arr_ferm,
        Lambda_arr_dipole,
        dip_depth=0.01,
        E_bins=E_bins_fermi,
        dm_type="fermionic",
        operator="anapole",
        c_s=0.0,
        c_p=1.0,
        majorana=True,
    )

    m_chi_arr_scalar = np.logspace(-6, 20, 20)
    Lambda_arr_scalar = np.logspace(-3, 6, 15)
    run_naive_fermi_constraints(
        m_chi_arr_scalar,
        Lambda_arr_scalar,
        dip_depth=0.01,
        E_bins=E_bins_fermi,
        dm_type="scalar",
        operator="rayleigh",
        c_phi=1.0,
    )
    
    print("\n" + "=" * 70)
    print("ALL SCANS COMPLETE (INCLUDING MAJORANA DM)")
    print("=" * 70)
