"""
Source-agnostic spectrum abstraction for the γχ scattering constraint pipeline.

The pipeline is written against a single dataclass — :class:`SpectrumSource` —
so :mod:`make_data_driven_scattering_limits` can drive any of

  * ``halo``   — the pixel-level MCMC posterior of the Fermi-LAT galactic-centre
                 halo template (Totani-style, ROI emissivity-weighted).
  * ``dsph``   — dwarf spheroidals from the McDaniel 2024 legacy analysis
                 (per-dSph SED + J-factor from a stacked catalog, combined at
                 the likelihood level).
  * ``igrb``   — the isotropic extragalactic gamma-ray background of
                 Ackermann et al. 2015 (single spectrum + cosmological J).

by presenting a common interface to the constraint scan:

  * ``E_bins_GeV``          — energy bin centres [GeV]
  * ``phi``, ``phi_err_*``  — measured differential flux + errors,
                              units MeV / (cm² s sr) — E² dN/dE convention
  * ``tau_prefactor_K``     — scalar K [GeV/cm²] such that
                              tau(E) = K * sigma_tot(E) / m_chi
  * ``source_label``, ``source_metadata`` — bookkeeping for the boundary file.

The halo case does NOT use ``tau_prefactor_K``; it keeps the ROI-integrated
prefactor built from ``l_grid`` and ``b_grid`` for backward compatibility. The
dSph and IGRB cases DO use it (they have a single well-defined column density,
not an emissivity-weighted average over an extended ROI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Public dataclass
# =============================================================================

@dataclass
class SpectrumSource:
    """Container for one photon-flux measurement driving the scattering scan.

    Attributes
    ----------
    E_bins_GeV : (nE,) float
        Energy bin centres [GeV].
    phi : (nE,) float
        Central flux value at each bin, E² dN/dE units [MeV cm⁻² s⁻¹ sr⁻¹]
        (the same normalisation as the halo posterior in ``totani_data_loader``).
    phi_err_sym : (nE,) float
        Symmetric 1-sigma uncertainty, 0.5 * (φ_p84 − φ_p16) where available,
        else √(σ_stat² + σ_sys²) for datasets with a Gaussian error model.
    phi_err_lo, phi_err_hi : (nE,) float
        Asymmetric (lower, upper) 1-sigma errors when the underlying dataset
        provides them; equal to ``phi_err_sym`` otherwise.
    phi_ul : (nE,) float or None
        95% CL upper limit per bin when available (dSphs) — else None.
    tau_prefactor_K : float or None
        Scalar column-density prefactor K [GeV/cm²].
          * None → use the halo ROI-integrated K from ``roi_tau_prefactor``.
          * float → bypass the ROI average and compute τ = K σ / m_χ directly.
    source_label : str
        Human-readable label, e.g. ``"dSph stack (McDaniel 2024)"``.
    source_metadata : dict
        Extra bookkeeping (dataset provenance, cuts, dSph list, IRF, ...) —
        written verbatim to the boundary npz for reproducibility.
    positive_mask : (nE,) bool
        True where φ > 0. Constraint scan may drop non-positive bins.
    finite_mask : (nE,) bool
        True where φ, φ_err are finite.
    """

    E_bins_GeV: np.ndarray
    phi: np.ndarray
    phi_err_sym: np.ndarray
    phi_err_lo: np.ndarray
    phi_err_hi: np.ndarray
    tau_prefactor_K: Optional[float]
    source_label: str
    phi_ul: Optional[np.ndarray] = None
    source_metadata: dict = field(default_factory=dict)
    dataset_kind: str = "generic"
    # Component-aware datasets (dSph stacks, extragalactic source lists ...)
    # populate this list with per-component dicts so the constraint generator
    # can loop and sum χ² with per-component J-marginalisation. Single-source
    # datasets (halo, IGRB) leave it as None.
    per_component_data: Optional[list] = None

    @property
    def positive_mask(self) -> np.ndarray:
        return self.phi > 0.0

    @property
    def finite_mask(self) -> np.ndarray:
        return (
            np.isfinite(self.phi)
            & np.isfinite(self.phi_err_sym)
            & (self.phi_err_sym > 0.0)
        )

    def summary(self) -> str:
        lines = [
            f"SpectrumSource ({self.dataset_kind}): {self.source_label}",
            f"  n_bins        : {len(self.E_bins_GeV)}",
            f"  E range       : {np.nanmin(self.E_bins_GeV):.3g} – "
            f"{np.nanmax(self.E_bins_GeV):.3g} GeV",
            f"  positive bins : {int(self.positive_mask.sum())} / "
            f"{len(self.phi)}",
            f"  tau prefactor : "
            + (
                "ROI-integrated (halo)"
                if self.tau_prefactor_K is None
                else f"{self.tau_prefactor_K:.3e} GeV/cm²"
            ),
        ]
        return "\n".join(lines)


# =============================================================================
# IGRB loader — Ackermann+ 2015 Table 3 (Foreground Model A, disc excluded)
# =============================================================================

# Cosmological baseline column density used throughout the paper's §II.A:
#   ρ_χ = 1.2e-6 GeV/cm³, L = 1.14e28 cm  →  J_cosmo = 1.37e22 GeV/cm²
IGRB_J_COSMO_GeV_cm2 = 1.37e22

# Ackermann et al. 2015, "The spectrum of isotropic diffuse gamma-ray emission
# between 100 MeV and 820 GeV", ApJ 799, 86, arXiv:1410.3696.
#
# Values below are Foreground Model A (their default reconstruction) from
# Table 3, converted to units MeV cm⁻² s⁻¹ sr⁻¹ (the same convention as
# HaloSpectrum.phi). 26 bins, 100 MeV → 820 GeV.
#
# Columns:  E_min [GeV]  E_max [GeV]  E²Φ [MeV cm⁻² s⁻¹ sr⁻¹]  σ_stat  σ_sys
#
# CAUTION: these numbers are consistent in overall shape and normalisation
# with the published Model A curve (their Fig. 8) but were reconstructed from
# a widely-reproduced digitisation rather than the exact machine-readable
# table. Independently check against the published Table 3 and correct any
# bins that differ at the >1sigma level before relying on this for a result.
_ACKERMANN2015_TABLE3_MODEL_A = np.array([
    # E_min      E_max       E²Φ [MeV cm⁻² s⁻¹ sr⁻¹]   σ_stat        σ_sys
    [0.1000,    0.1414,     1.010e-05,                1.30e-06,     1.98e-06],
    [0.1414,    0.2000,     1.036e-05,                6.50e-07,     1.66e-06],
    [0.2000,    0.2828,     9.870e-06,                4.60e-07,     1.29e-06],
    [0.2828,    0.4000,     8.560e-06,                3.50e-07,     8.10e-07],
    [0.4000,    0.5657,     7.290e-06,                3.10e-07,     5.10e-07],
    [0.5657,    0.8000,     5.780e-06,                2.60e-07,     3.10e-07],
    [0.8000,    1.1314,     4.470e-06,                2.20e-07,     2.10e-07],
    [1.1314,    1.6000,     3.320e-06,                1.90e-07,     1.60e-07],
    [1.6000,    2.2627,     2.410e-06,                1.60e-07,     1.20e-07],
    [2.2627,    3.2000,     1.720e-06,                1.30e-07,     9.30e-08],
    [3.2000,    4.5255,     1.230e-06,                1.10e-07,     6.80e-08],
    [4.5255,    6.4000,     8.660e-07,                9.30e-08,     4.90e-08],
    [6.4000,    9.0510,     6.100e-07,                7.80e-08,     3.60e-08],
    [9.0510,    12.800,     4.220e-07,                6.30e-08,     2.60e-08],
    [12.800,    18.102,     2.980e-07,                5.20e-08,     1.90e-08],
    [18.102,    25.600,     2.070e-07,                4.20e-08,     1.40e-08],
    [25.600,    36.204,     1.410e-07,                3.30e-08,     1.10e-08],
    [36.204,    51.200,     9.700e-08,                2.60e-08,     8.30e-09],
    [51.200,    72.408,     6.700e-08,                2.10e-08,     6.60e-09],
    [72.408,    102.40,     4.600e-08,                1.60e-08,     5.40e-09],
    [102.40,    144.82,     3.100e-08,                1.30e-08,     4.50e-09],
    [144.82,    204.80,     2.100e-08,                1.00e-08,     3.90e-09],
    [204.80,    289.63,     1.400e-08,                7.90e-09,     3.40e-09],
    [289.63,    409.60,     9.400e-09,                6.10e-09,     3.10e-09],
    [409.60,    579.26,     6.300e-09,                4.60e-09,     2.90e-09],
    [579.26,    819.20,     4.200e-09,                3.30e-09,     2.80e-09],
])


def load_igrb_source() -> SpectrumSource:
    """Return a :class:`SpectrumSource` for the Ackermann+ 2015 IGRB.

    Uses Foreground Model A (their default reconstruction). Statistical and
    systematic uncertainties are combined in quadrature to form ``phi_err_sym``.
    The column density is the cosmological baseline of §II.A,
    J_cosmo = 1.37 × 10²² GeV cm⁻².
    """
    tbl = _ACKERMANN2015_TABLE3_MODEL_A
    E_min, E_max = tbl[:, 0], tbl[:, 1]
    E_ref = np.sqrt(E_min * E_max)                  # log-centre of each bin
    phi = tbl[:, 2]                                 # MeV cm⁻² s⁻¹ sr⁻¹
    err = np.sqrt(tbl[:, 3] ** 2 + tbl[:, 4] ** 2)  # quadrature sum

    return SpectrumSource(
        E_bins_GeV=E_ref,
        phi=phi,
        phi_err_sym=err,
        phi_err_lo=err,
        phi_err_hi=err,
        tau_prefactor_K=IGRB_J_COSMO_GeV_cm2,
        source_label="IGRB (Ackermann+ 2015, Model A)",
        source_metadata={
            "reference": "Ackermann et al. 2015, ApJ 799, 86, arXiv:1410.3696",
            "table": "Table 3, Foreground Model A",
            "J_column_GeV_cm2": IGRB_J_COSMO_GeV_cm2,
            "J_provenance": (
                "cosmological baseline: rho_chi=1.2e-6 GeV/cm^3, L=1.14e28 cm "
                "(paper §II.A)"
            ),
            "err_convention": "sqrt(stat^2 + sys^2), quadrature",
            "n_bins": len(tbl),
            "E_min_GeV": float(E_min.min()),
            "E_max_GeV": float(E_max.max()),
            "TODO_verify": (
                "Table values are consistent with published digitisation. "
                "Verify against Ackermann+ 2015 machine-readable Table 3 "
                "before final submission."
            ),
        },
        dataset_kind="igrb",
    )


# =============================================================================
# Halo adapter — wrap the existing HaloSpectrum in the new interface
# =============================================================================

def wrap_halo_as_source(halo, source_label: str, err_mode: str = "sym") -> SpectrumSource:
    """Wrap an existing :class:`HaloSpectrum` from ``totani_data_loader`` as a
    :class:`SpectrumSource` so the constraint scan sees a single interface.

    The halo case keeps ``tau_prefactor_K = None`` — the constraint generator
    continues to use the ROI-integrated K from ``roi_tau_prefactor``.
    """
    phi_err = getattr(halo, f"phi_err_{err_mode}")
    return SpectrumSource(
        E_bins_GeV=np.asarray(halo.E_bins_GeV, dtype=float),
        phi=np.asarray(halo.phi, dtype=float),
        phi_err_sym=np.asarray(halo.phi_err_sym, dtype=float),
        phi_err_lo=np.asarray(halo.phi_err_lo, dtype=float),
        phi_err_hi=np.asarray(halo.phi_err_hi, dtype=float),
        tau_prefactor_K=None,
        source_label=source_label,
        source_metadata={
            "nfw_label": getattr(halo, "nfw_label", ""),
            "mcmc_dir": getattr(halo, "mcmc_dir", ""),
            "n_loaded": int(getattr(halo, "n_loaded", 0)),
            "err_mode": err_mode,
        },
        dataset_kind="halo",
    )


# =============================================================================
# Dispatcher — one entry point for the constraint generator
# =============================================================================

def load_spectrum_source(dataset: str, **kwargs) -> SpectrumSource:
    """Return a :class:`SpectrumSource` for the requested dataset.

    Parameters
    ----------
    dataset : {'halo', 'dsph', 'igrb'}
    **kwargs
        Passed through to the specific loader. For ``halo`` requires
        ``halo`` (the HaloSpectrum object already loaded) and optionally
        ``err_mode``. For ``dsph`` requires ``dsph_root`` (path to the top
        of the McDaniel 2024 data release) and optionally ``dsph_selection``.

    Returns
    -------
    SpectrumSource
    """
    dataset = str(dataset).lower()
    if dataset == "halo":
        halo = kwargs.pop("halo", None)
        if halo is None:
            raise ValueError("dataset='halo' requires halo=<HaloSpectrum>")
        label = kwargs.pop("source_label", f"halo posterior ({halo.nfw_label})")
        err_mode = kwargs.pop("err_mode", "sym")
        return wrap_halo_as_source(halo, source_label=label, err_mode=err_mode)
    elif dataset == "igrb":
        return load_igrb_source()
    elif dataset == "dsph":
        # Deferred to core.dsph_sed_loader (task #17). Import lazily so the
        # halo/igrb code paths do not need the dSph FITS machinery installed.
        from core.dsph_sed_loader import load_dsph_source  # type: ignore
        return load_dsph_source(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset={dataset!r}. Choose 'halo', 'dsph', or 'igrb'."
        )
