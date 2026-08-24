"""
kinematics.py
=============
Photon-DM Compton-like scattering kinematics in the DM rest frame.

All energies in GeV throughout. The heavy-DM regime (E_gamma << m_chi)
is the physically relevant one for the Totani halo excess, where
E_gamma ~ 1-800 GeV and m_chi ~ 0.5-0.8 TeV, giving E/m_chi ~ O(0.1).

Public API
----------
scattered_energy(E, theta, m_chi)      : E'(E, theta, m_chi)
recoil_fraction(E, theta, m_chi)       : (E - E') / E
jacobian_dOmega_dE(E, theta, m_chi)    : |dOmega/dE'| for change-of-variables
max_energy_loss_fraction(E, m_chi)     : Delta E / E at theta = pi (backscatter)
omega_out_grid(E, cos_theta, m_chi)    : vectorised E' over a (nE, nTheta) grid
build_redistribution_matrix(E_bins, cos_theta_vals, dsigma_dOmega, m_chi)
    : assemble the (nE_out, nE_in) kernel K[i,j] = probability that a photon
      born in energy bin j is observed in bin i after one scatter.

Notes
-----
Kinematics are exact (not heavy-DM expanded). The expansion parameter is
    eps = E / m_chi
which ranges from ~0.002 at 1 GeV / 500 GeV to ~1.6 at 800 GeV / 500 GeV,
so the full expressions are required across the Totani energy range.

Sign conventions: theta in [0, pi], cos_theta in [-1, 1].
Forward scattering (theta = 0) preserves photon energy exactly.
Backscattering (theta = pi) gives maximum energy loss.
"""

import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Core kinematics
# ---------------------------------------------------------------------------

def scattered_energy(
    E: ArrayLike,
    theta: ArrayLike,
    m_chi: float,
) -> np.ndarray:
    """
    Scattered photon energy E'(E, theta, m_chi) from Compton-like kinematics
    with electron mass replaced by m_chi.

        E' = E / (1 + (E / m_chi) * (1 - cos(theta)))

    Parameters
    ----------
    E : float or array
        Incoming photon energy [GeV].
    theta : float or array
        Scattering angle [rad], broadcast-compatible with E.
    m_chi : float
        DM mass [GeV].

    Returns
    -------
    E_out : ndarray
        Scattered photon energy [GeV], same shape as broadcast(E, theta).
    """
    E = np.asarray(E, dtype=float)
    theta = np.asarray(theta, dtype=float)
    denom = 1.0 + (E / m_chi) * (1.0 - np.cos(theta))
    return E / denom


def recoil_fraction(
    E: ArrayLike,
    theta: ArrayLike,
    m_chi: float,
) -> np.ndarray:
    """
    Fractional energy loss (E - E') / E for a single scatter.

    Ranges from 0 at forward scatter to ~2E/m_chi at backscatter (heavy-DM limit).

    Parameters
    ----------
    E, theta, m_chi : as in scattered_energy.

    Returns
    -------
    delta : ndarray  in [0, 1)
    """
    E_out = scattered_energy(E, theta, m_chi)
    return (np.asarray(E, dtype=float) - E_out) / np.asarray(E, dtype=float)


def max_energy_loss_fraction(E: ArrayLike, m_chi: float) -> np.ndarray:
    """
    Maximum fractional energy loss (backscatter, theta = pi).

        Delta_max / E = 2 * (E/m_chi) / (1 + 2 * E/m_chi)

    In the heavy-DM limit (E << m_chi) this approaches 2E/m_chi.

    Parameters
    ----------
    E : float or array   [GeV]
    m_chi : float        [GeV]

    Returns
    -------
    frac : ndarray   dimensionless, in [0, 1)
    """
    E = np.asarray(E, dtype=float)
    eps = E / m_chi
    return 2.0 * eps / (1.0 + 2.0 * eps)


def scattered_energy_grid(
    E_bins: np.ndarray,
    cos_theta_vals: np.ndarray,
    m_chi: float,
) -> np.ndarray:
    """
    Build a 2D grid of scattered energies.

    Parameters
    ----------
    E_bins : (nE,)           Photon energies [GeV]
    cos_theta_vals : (nTh,)  cos(theta) values in [-1, 1]
    m_chi : float            DM mass [GeV]

    Returns
    -------
    E_out : (nE, nTh)   E'[i, k] = scattered energy for E_bins[i], cos_theta[k]
    """
    E_bins = np.asarray(E_bins, dtype=float)[:, None]      # (nE, 1)
    cos_th = np.asarray(cos_theta_vals, dtype=float)[None, :]  # (1, nTh)
    denom = 1.0 + (E_bins / m_chi) * (1.0 - cos_th)
    return E_bins / denom                                   # (nE, nTh)


def jacobian_dE_dcostheta(
    E: ArrayLike,
    cos_theta: ArrayLike,
    m_chi: float,
) -> np.ndarray:
    """
    Jacobian |dE' / d(cos theta)| for the change-of-variables
    d(cos theta) -> dE'.

    From E' = E / [1 + (E/m)(1 - cos theta)]:

        dE'/d(cos theta) = E^2 / (m_chi * denom^2)   > 0

    Used to convert dσ/dΩ into dσ/dE' via:

        dσ/dE' = (2π / dE'/d(cosθ)) * dσ/dΩ

    Parameters
    ----------
    E : float or array   [GeV]
    cos_theta : float or array   dimensionless
    m_chi : float   [GeV]

    Returns
    -------
    jac : ndarray   [GeV^-1 sr]  (positive definite)
    """
    E = np.asarray(E, dtype=float)
    cos_theta = np.asarray(cos_theta, dtype=float)
    denom = 1.0 + (E / m_chi) * (1.0 - cos_theta)
    return E**2 / (m_chi * denom**2)


# ---------------------------------------------------------------------------
# Redistribution matrix construction
# ---------------------------------------------------------------------------

def build_redistribution_matrix(
    E_bins: np.ndarray,
    cos_theta_vals: np.ndarray,
    dsigma_dOmega: np.ndarray,
    m_chi: float,
    *,
    sigma_tot: np.ndarray | None = None,
    in_roi_weight: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assemble the energy redistribution kernel K[i, j].

    K[i, j] is the fraction of photons born in energy bin j that are
    observed in energy bin i after one scatter, integrated over all
    scattering angles whose kinematic output E' falls in bin i.

    Concretely:

        K[i, j] = (2π / σ_tot(E_j)) *
                  ∫_{cos θ : E'(E_j,θ) ∈ [E_i_lo, E_i_hi]}
                    (dσ/dΩ)(E_j, θ) d(cos θ)

    so that sum_i K[i,j] ≤ 1  (equality if no flux leaks below E_min).

    The survival diagonal (photons that are NOT scattered) is handled
    separately via the exp(-τ) factor and is NOT included in K.

    Parameters
    ----------
    E_bins : (nE,)
        Photon energy bin centres [GeV]. Assumed log-spaced but not required.
    cos_theta_vals : (nTh,)
        Integration nodes for cos(theta) in (-1, 1]. Should be fine enough
        to resolve the angular structure of dσ/dΩ (typically ~500–2000 points).
    dsigma_dOmega : (nE, nTh)
        Differential cross section at each (E, cos_theta) node [cm^2 / sr].
        Must be non-negative; NaN/Inf are treated as zero.
    m_chi : float
        DM mass [GeV].
    sigma_tot : (nE,) optional
        Pre-computed total cross sections [cm^2]. If None, computed internally
        by trapezoid integration of dsigma_dOmega. Providing it avoids
        redundant computation when called inside a parameter scan.
    in_roi_weight : (nTh,) optional
        Per-angle weight in [0, 1] representing the probability that a photon
        scattered at that angle remains within the LAT ROI. If None, unity
        weights are used (all scattered photons recovered). Pass a precomputed
        array from `roi_recovery_fraction` for a more accurate treatment.

    Returns
    -------
    K : (nE, nE)   float64, upper-triangular (scattering moves E down only)
        Normalised so that column sums ≤ 1.
        K[i, j] gives the fraction of flux from bin j redistributed to bin i.

    Notes
    -----
    Upper-triangular structure: E' ≤ E always (photons lose energy), so
    K[i, j] = 0 for i > j. For i == j, K includes the fraction of photons
    that scatter but remain in the same energy bin (small forward-scatter
    events).

    Bin boundaries are taken as the geometric midpoints between adjacent
    bin centres, with the lower boundary of the first bin and the upper
    boundary of the last bin extrapolated at the same log-spacing.
    """
    E_bins = np.asarray(E_bins, dtype=float)
    cos_theta_vals = np.asarray(cos_theta_vals, dtype=float)
    dsigma_dOmega = np.asarray(dsigma_dOmega, dtype=float)
    dsigma_dOmega = np.where(np.isfinite(dsigma_dOmega) & (dsigma_dOmega >= 0.0),
                              dsigma_dOmega, 0.0)

    nE = len(E_bins)
    nTh = len(cos_theta_vals)

    if dsigma_dOmega.shape != (nE, nTh):
        raise ValueError(
            f"dsigma_dOmega must have shape (nE={nE}, nTh={nTh}), "
            f"got {dsigma_dOmega.shape}"
        )

    if in_roi_weight is None:
        w_roi = np.ones(nTh, dtype=float)
    else:
        w_roi = np.asarray(in_roi_weight, dtype=float)
        if w_roi.shape != (nTh,):
            raise ValueError(f"in_roi_weight must have shape (nTh={nTh},)")

    # --- Energy bin boundaries (geometric midpoints) ---
    log_E = np.log(E_bins)
    log_edges = np.empty(nE + 1)
    log_edges[1:-1] = 0.5 * (log_E[:-1] + log_E[1:])
    log_edges[0] = log_E[0] - 0.5 * (log_E[1] - log_E[0])
    log_edges[-1] = log_E[-1] + 0.5 * (log_E[-1] - log_E[-2])
    E_lo = np.exp(log_edges[:-1])   # (nE,)
    E_hi = np.exp(log_edges[1:])    # (nE,)

    # --- Total cross sections (for normalisation) ---
    if sigma_tot is None:
        # σ_tot(E) = 2π ∫ (dσ/dΩ)(E, cosθ) d(cosθ)
        sigma_tot = 2.0 * np.pi * np.trapezoid(
            dsigma_dOmega * w_roi[None, :],
            cos_theta_vals,
            axis=1,
        )   # (nE,)
    else:
        sigma_tot = np.asarray(sigma_tot, dtype=float)

    # --- Scattered energy grid E'[j, k] for all (input-bin j, angle k) ---
    # Shape: (nE, nTh)
    E_out = scattered_energy_grid(E_bins, cos_theta_vals, m_chi)   # (nE, nTh)

    # --- Assemble K[i, j] ---
    K = np.zeros((nE, nE), dtype=float)

    # Integration weights: trapezoid rule over cos_theta
    d_cos = np.gradient(cos_theta_vals)     # (nTh,) variable spacing safe

    for j in range(nE):
        if sigma_tot[j] <= 0.0:
            continue

        E_j = E_bins[j]
        E_out_j = E_out[j]                # (nTh,) scattered energies for this input bin
        dsig_j = dsigma_dOmega[j]         # (nTh,)

        norm_j = sigma_tot[j] / (2.0 * np.pi)   # denominator in normalised K

        for i in range(j + 1):            # upper-triangular: i <= j only
            # Find angles whose E' falls in bin i
            in_bin_i = (E_out_j >= E_lo[i]) & (E_out_j < E_hi[i])

            if not np.any(in_bin_i):
                continue

            # Weighted integration: (dσ/dΩ) * w_roi * d(cosθ) over bin i angles
            integrand = dsig_j * w_roi    # (nTh,)
            K[i, j] = np.sum(integrand[in_bin_i] * d_cos[in_bin_i]) / norm_j

    return K


# ---------------------------------------------------------------------------
# ROI recovery fraction
# ---------------------------------------------------------------------------

def roi_recovery_fraction(
    cos_theta_vals: np.ndarray,
    roi_half_angle_deg: float = 60.0,
) -> np.ndarray:
    """
    Approximate fraction of scattered photons that remain within the LAT ROI
    for each scattering angle, assuming a circular ROI of half-angle roi_half_angle_deg
    and isotropic azimuthal distribution of the scatter.

    For a photon originally on the ROI axis, a scatter at polar angle theta
    moves it to theta from the original direction. The fraction of azimuthal
    angles phi for which the scattered direction stays within a cone of
    half-opening angle alpha is:

        f(theta, alpha) =
            0                        if theta >= 2*alpha  (always outside)
            1                        if theta == 0        (no scatter)
            arccos(...) / pi         otherwise (geometric chord fraction)

    This is approximate (assumes the photon originates at the ROI centre),
    but captures the correct qualitative behaviour: forward scatters stay in
    the ROI, large-angle scatters do not.

    For the Totani halo template, which is extended and fills the full ROI,
    the right treatment averages over source positions. A conservative approach
    is to use unity weights (all photons recovered), which overestimates
    in-scatter and gives a more conservative constraint. This function provides
    the geometric estimate as an optional refinement.

    Parameters
    ----------
    cos_theta_vals : (nTh,)   cos(theta) of scattering angle
    roi_half_angle_deg : float   half-opening angle of ROI [degrees]

    Returns
    -------
    f_roi : (nTh,)   recovery fraction in [0, 1]
    """
    cos_theta_vals = np.asarray(cos_theta_vals, dtype=float)
    alpha = np.deg2rad(roi_half_angle_deg)
    theta = np.arccos(np.clip(cos_theta_vals, -1.0, 1.0))

    f_roi = np.ones_like(theta)

    # Angles larger than 2*alpha always exit the ROI (geometric limit)
    always_out = theta >= 2.0 * alpha
    f_roi[always_out] = 0.0

    # Intermediate regime: analytic chord fraction
    mid = ~always_out & (theta > 0.0)
    if np.any(mid):
        th_m = theta[mid]
        # From spherical geometry: fraction of azimuthal circle inside cone
        cos_arg = np.clip(
            (np.cos(alpha) - np.cos(th_m)) / (np.sin(alpha) * np.sin(th_m) + 1e-300),
            -1.0, 1.0,
        )
        f_roi[mid] = np.arccos(cos_arg) / np.pi

    return f_roi


# ---------------------------------------------------------------------------
# Convenience: flux-normalised in-scatter term
# ---------------------------------------------------------------------------

def inscatter_flux(
    Phi_0: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    E_bins: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the in-scatter contribution to the observed spectrum.

    For each output energy bin i, photons that were born at higher energies j
    and scattered into bin i contribute:

        Phi_in[i] = sum_{j > i} K[i,j] * tau[j] * Phi_0[j] * exp(-tau[j])

    The factor tau[j] * exp(-tau[j]) is the single-scatter probability in the
    optically thin regime. For tau << 1, this approximates to tau[j] * Phi_0[j].

    Parameters
    ----------
    Phi_0 : (nE,)   unattenuated source spectrum (arbitrary units)
    K : (nE, nE)    photon-number redistribution matrix from build_redistribution_matrix
    tau : (nE,)     optical depth per energy bin
    E_bins : (nE,) optional
        If provided, Phi_0 is treated as an energy-flux-like spectrum
        (approximately E^2 dN/dE) and K is converted with an E_i/E_j factor
        before being applied. If omitted, Phi_0 is treated as photon number.

    Returns
    -------
    Phi_in : (nE,)  in-scatter contribution, same units as Phi_0
    """
    Phi_0 = np.asarray(Phi_0, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if E_bins is not None:
        E_bins = np.asarray(E_bins, dtype=float)
        K = K * (E_bins[:, None] / E_bins[None, :])

    # Weight source by single-scatter probability
    scatter_weight = tau * np.exp(-tau)          # (nE,)  peaks at tau=1
    return K @ (scatter_weight * Phi_0)          # (nE,)


def reshaped_spectrum(
    Phi_0: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    E_bins: np.ndarray | None = None,
) -> np.ndarray:
    """
    Full reshaping model for the observed spectrum.

    Phi_obs[i] = Phi_0[i] * exp(-tau[i])              [survival term]
               + sum_{j>=i} K[i,j] * tau[j] * Phi_0[j] * exp(-tau[j])   [in-scatter]

    This is the single-scatter approximation. It is valid when tau << 1
    everywhere (the Totani halo case: tau ~ 10^{-3} to 10^{-6}).

    For completeness, the diagonal term K[i,i] * tau[i] * Phi_0[i] * exp(-tau[i])
    represents photons that scatter but remain in the same energy bin (near-forward
    scatters). This is typically very small because the bin width >> kinematic
    energy shift for sub-TeV DM, but is included for consistency.

    Note
    ----
    This is NOT a full radiative-transfer solution. It neglects:
    - Multiple scatters (order tau^2 corrections)
    - Scattering of already-in-scattered photons
    Both are completely negligible at tau << 1.

    Parameters
    ----------
    Phi_0 : (nE,)   unattenuated source spectrum
    K : (nE, nE)    photon-number redistribution matrix
    tau : (nE,)     optical depth per energy bin
    E_bins : (nE,) optional
        If provided, apply the photon-number-to-energy-flux E_i/E_j correction.

    Returns
    -------
    Phi_obs : (nE,)  reshaped observed spectrum, same units as Phi_0
    """
    Phi_0 = np.asarray(Phi_0, dtype=float)
    tau = np.asarray(tau, dtype=float)

    survival = Phi_0 * np.exp(-tau)
    inscatter = inscatter_flux(Phi_0, K, tau, E_bins=E_bins)
    return survival + inscatter
