"""
totani_data_loader.py
=====================
Load the Totani halo-component spectrum directly from the MCMC posterior
files produced by Totani_paper_check.

This replaces the hard-coded PHI_TOTANI / SIGMA_TOTANI arrays in
attenuation_eft.py, which are coarse read-offs from Totani's Figure 8.
The MCMC files are the ground truth: each npz stores the full posterior
over template coefficients f for one energy bin, fitted to the actual
Fermi-LAT counts via a binned Poisson likelihood.

Physical convention (from make_totani_fig8.py)
----------------------------------------------
The NFW halo template is pole-normalised:

    mu_nfw[k, pixel] = (iso_target_e2 / E_ctr[k]^2)
                       * (J(l,b) / J_pole)
                       * expo[k, pixel] * omega[pixel] * dE[k]

so at the galactic pole J/J_pole = 1, and at template coefficient f:

    E^2 dN/dE |_pole = f * iso_target_e2          [MeV cm^-2 s^-1 sr^-1]

This is an exact algebraic consequence of the normalisation choice and
requires no pixel data outside the ROI.

Central value: posterior median f_p50 (guarantees p16 <= p50 <= p84,
avoiding negative errorbars near the positivity boundary — matches
Totani's Figure 8 convention).

Public API
----------
load_halo_spectrum(mcmc_dir, nfw_label=None)
    -> HaloSpectrum dataclass with E_bins_MeV, E_bins_GeV, phi, phi_p16,
       phi_p84, phi_err_sym, iso_target_e2 per bin.

load_component_spectra(mcmc_dir, labels=None)
    -> ComponentSpectra dataclass with posterior spectra for every requested
       MCMC template label. These are useful inputs for a multi-component
       scattering transfer, but their physical geometry still has to be chosen
       component by component.

load_halo_spectrum_rho25(...)  rho^2.5, disk excluded
load_halo_spectrum_rho2(...)   rho^2,   disk excluded  (Totani preferred)
load_halo_spectrum_rho1(...)   rho^1,   disk excluded

available_mcmc_dirs()
    -> dict mapping profile name -> absolute path, for whichever dirs exist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Default paths — mirror the REPO_DIR logic used throughout Totani_paper_check
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_TS_DIR = _HERE.parent
_REPO_DIR = Path(
    os.environ.get(
        "REPO_PATH",
        str(_TS_DIR.parent),
    )
)
_PAPER_CHECK = _REPO_DIR / "Totani_paper_check" / "mcmc"

# Cellwise (Totani-style) results live under Totani_paper_check/mcmc/ above.
#   * global (single-likelihood) fits live under Totani_reanalysis/mcmc/fit_results/quick_global/
#   * pixel-level posteriors live at the repo root under pixelwise_mcmc/
#     (thinned summaries in this repo; point $HALO_POSTERIOR_ROOT at a
#     directory with the full chains if you have one — see docs/DATA_FETCH.md)
# Keep _PAPER_CHECK as a fallback so checkouts predating the reorg still resolve.
_REANALYSIS_QGLOBAL = _REPO_DIR / "Totani_reanalysis" / "mcmc" / "fit_results" / "quick_global"
_REANALYSIS_PIXEL   = Path(os.environ.get("HALO_POSTERIOR_ROOT", str(_REPO_DIR / "pixelwise_mcmc")))


def _first_existing_dir(*paths: Path) -> Path:
    """Return the first existing path, or the first candidate if none exist yet."""
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return Path(paths[0])


_MCMC_DIRS = {
    "rho2.5": _first_existing_dir(_PAPER_CHECK / "mcmc_results_fig5", _PAPER_CHECK / "mcmc_results_fig5_10deg"),
    "rho2":   _first_existing_dir(_PAPER_CHECK / "mcmc_results_fig6", _PAPER_CHECK / "mcmc_results_fig6_10deg"),
    "rho1":   _first_existing_dir(_PAPER_CHECK / "mcmc_results_fig7", _PAPER_CHECK / "mcmc_results_fig7_10deg"),
    "fig2_3": _first_existing_dir(_PAPER_CHECK / "mcmc_results_fig2_3", _PAPER_CHECK / "mcmc_results_fig2_3_10deg"),
    "fig4":   _first_existing_dir(_PAPER_CHECK / "mcmc_results_fig4", _PAPER_CHECK / "mcmc_results_fig4_10deg"),
    # Global (single-likelihood) fits — disk excluded
    "global_rho2.5": _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig5",
        _PAPER_CHECK       / "global_fit_results_fig5",
    ),
    "global_rho2":   _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig6",
        _PAPER_CHECK       / "global_fit_results_fig6",
    ),
    "global_rho1":   _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig7",
        _PAPER_CHECK       / "global_fit_results_fig7",
    ),
    "global_fig2_3": _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig2_3",
        _PAPER_CHECK       / "global_fit_results_fig2_3",
    ),
    "global_fig4":   _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig4",
        _PAPER_CHECK       / "global_fit_results_fig4",
    ),
    # Global fits — disk INCLUDED (companion-paper systematics variant)
    "global_rho2.5_w_disk": _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig5_w_disk",
        _PAPER_CHECK       / "global_fit_results_fig5_w_disk",
    ),
    "global_rho2_w_disk":   _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig6_w_disk",
        _PAPER_CHECK       / "global_fit_results_fig6_w_disk",
    ),
    "global_rho1_w_disk":   _first_existing_dir(
        _REANALYSIS_QGLOBAL / "global_fit_results_fig7_w_disk",
        _PAPER_CHECK       / "global_fit_results_fig7_w_disk",
    ),
    # Pixel-level fits — disk excluded (the PRIMARY profile for the EFT scattering paper)
    "pixelwise_global_rho2.5": _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig5",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig5",
    ),
    "pixelwise_global_rho2":   _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig6",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig6",
    ),
    "pixelwise_global_rho1":   _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig7",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig7",
    ),
    "pixelwise_global_fig2_3": _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig2_3",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig2_3",
    ),
    "pixelwise_global_fig4":   _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig4",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig4",
    ),
    # Pixel-level fits — disk INCLUDED (systematics variant)
    "pixelwise_global_rho2.5_w_disk": _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig5_w_disk",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig5_w_disk",
    ),
    "pixelwise_global_rho2_w_disk":   _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig6_w_disk",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig6_w_disk",
    ),
    "pixelwise_global_rho1_w_disk":   _first_existing_dir(
        _REANALYSIS_PIXEL / "pixelwise_mcmc_results_fig7_w_disk",
        _PAPER_CHECK      / "pixelwise_mcmc_results_fig7_w_disk",
    ),
}

_ISO_TARGET_E2_DEFAULT = 1e-4  # MeV cm^-2 s^-1 sr^-1 — fallback if not in npz


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HaloSpectrum:
    """
    Totani halo-component spectrum, loaded from MCMC posterior files.

    All flux quantities are E^2 dN/dE at the galactic pole
    [MeV cm^-2 s^-1 sr^-1].

    Attributes
    ----------
    E_bins_MeV : (nE,)   energy bin centres [MeV]
    E_bins_GeV : (nE,)   energy bin centres [GeV]
    phi        : (nE,)   central value (posterior median f_p50 * iso_target_e2)
    phi_p16    : (nE,)   16th percentile (lower 1-sigma)
    phi_p84    : (nE,)   84th percentile (upper 1-sigma)
    phi_err_lo : (nE,)   phi - phi_p16  (always >= 0)
    phi_err_hi : (nE,)   phi_p84 - phi  (always >= 0)
    phi_err_sym: (nE,)   symmetric 1-sigma = 0.5*(phi_p84 - phi_p16)
    iso_target_e2 : (nE,) iso_target_e2 value used in each energy bin [MeV cm^-2 s^-1 sr^-1]
    f_nfw_p50  : (nE,)   raw posterior median NFW coefficient (dimensionless)
    f_nfw_p16  : (nE,)   raw 16th percentile
    f_nfw_p84  : (nE,)   raw 84th percentile
    nfw_label  : str     NFW template label matched in the npz files
    mcmc_dir   : str     directory the data was loaded from
    n_loaded   : int     number of energy bins successfully loaded
    """
    E_bins_MeV:    np.ndarray
    E_bins_GeV:    np.ndarray
    phi:           np.ndarray
    phi_p16:       np.ndarray
    phi_p84:       np.ndarray
    phi_err_lo:    np.ndarray
    phi_err_hi:    np.ndarray
    phi_err_sym:   np.ndarray
    iso_target_e2: np.ndarray
    f_nfw_p50:     np.ndarray
    f_nfw_p16:     np.ndarray
    f_nfw_p84:     np.ndarray
    nfw_label:     str
    mcmc_dir:      str
    n_loaded:      int

    @property
    def positive_mask(self) -> np.ndarray:
        """True where the central value is positive (use for chi2 fits)."""
        return self.phi > 0.0

    @property
    def finite_mask(self) -> np.ndarray:
        """True where all of phi, phi_p16, phi_p84 are finite."""
        return (
            np.isfinite(self.phi)
            & np.isfinite(self.phi_p16)
            & np.isfinite(self.phi_p84)
        )

    def to_fit_arrays(
        self,
        *,
        positive_only: bool = True,
        err_mode: str = "sym",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (E_GeV, phi, sigma) arrays ready for use in chi2 fitting,
        replacing the old PHI_TOTANI / SIGMA_TOTANI / E_BINS_GEV.

        Parameters
        ----------
        positive_only : bool
            If True (default), return only bins where phi > 0. This matches
            the convention in attenuation_eft.py and fit_totani_dm_scattering.py.
        err_mode : str
            'sym'  — symmetric 1-sigma = 0.5*(p84-p16)
            'lo'   — lower asymmetric errorbar (phi - p16)
            'hi'   — upper asymmetric errorbar (p84 - phi)
            'max'  — conservative: max(lo, hi)

        Returns
        -------
        E_GeV, phi, sigma : (n,) arrays
        """
        mask = self.finite_mask
        if positive_only:
            mask = mask & self.positive_mask

        if err_mode == "sym":
            sigma = self.phi_err_sym
        elif err_mode == "lo":
            sigma = self.phi_err_lo
        elif err_mode == "hi":
            sigma = self.phi_err_hi
        elif err_mode == "max":
            sigma = np.maximum(self.phi_err_lo, self.phi_err_hi)
        else:
            raise ValueError(f"err_mode must be 'sym', 'lo', 'hi', or 'max'; got {err_mode!r}")

        # Guard: never return zero or negative sigma
        sigma = np.where(
            (sigma > 0.0) & np.isfinite(sigma),
            sigma,
            np.nan,
        )
        mask = mask & np.isfinite(sigma)

        return (
            self.E_bins_GeV[mask],
            self.phi[mask],
            sigma[mask],
        )

    def summary(self) -> str:
        lines = [
            f"HaloSpectrum from: {self.mcmc_dir}",
            f"  NFW label : {self.nfw_label}",
            f"  nE loaded : {self.n_loaded} / {len(self.E_bins_MeV)}",
            f"  E range   : {self.E_bins_GeV[0]:.2f} – {self.E_bins_GeV[-1]:.2f} GeV",
            f"  phi range : {np.nanmin(self.phi):.3e} – {np.nanmax(self.phi):.3e} MeV cm^-2 s^-1 sr^-1",
            f"  positive bins: {int(self.positive_mask.sum())} / {len(self.phi)}",
        ]
        return "\n".join(lines)


@dataclass
class ComponentSpectra:
    """
    Posterior spectra for one or more MCMC template components.

    The flux conversion mirrors HaloSpectrum: coefficient f times
    iso_target_e2 for each energy bin. For the NFW template this has the clean
    pole-normalised meaning described above. For other templates it should be
    treated as the fitted component-amplitude spectrum in the same plotting
    convention unless its template normalisation has been checked separately.
    """

    E_bins_MeV: np.ndarray
    E_bins_GeV: np.ndarray
    labels: list[str]
    phi: dict[str, np.ndarray]
    phi_p16: dict[str, np.ndarray]
    phi_p84: dict[str, np.ndarray]
    phi_err_sym: dict[str, np.ndarray]
    iso_target_e2: np.ndarray
    mcmc_dir: str

    @property
    def finite_mask(self) -> np.ndarray:
        masks = [np.isfinite(self.E_bins_GeV)]
        for label in self.labels:
            masks.append(np.isfinite(self.phi[label]))
            masks.append(np.isfinite(self.phi_err_sym[label]))
        return np.logical_and.reduce(masks)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_halo_spectrum(
    mcmc_dir: str | Path,
    *,
    nfw_label: Optional[str] = None,
    counts_path: Optional[str | Path] = None,
    n_energy_bins: Optional[int] = None,
    central_stat: str = "f_p50",
) -> HaloSpectrum:
    """
    Load the Totani NFW halo-component spectrum from MCMC npz files.

    Parameters
    ----------
    mcmc_dir : str or Path
        Directory containing mcmc_results_k00.npz ... mcmc_results_kNN.npz.
    nfw_label : str, optional
        Exact NFW template label to look for in each npz. If None (default),
        the loader uses the first template whose name contains 'nfw'
        (case-insensitive). Raises if zero or more than one are found.
    counts_path : str or Path, optional
        Path to the counts CCUBE FITS file, used to read the energy axis
        (Ectr_mev per bin). If None, the energy axis is inferred from the
        npz Ectr_mev fields stored by run_mcmc.py.
    n_energy_bins : int, optional
        Expected number of energy bins. If provided and the loader finds fewer
        or more, it warns but continues. Inferred automatically if None.
    central_stat : str
        Which posterior statistic to use as the central value.
        'f_p50' (default, recommended) — posterior median.
        'f_ml' — maximum-likelihood point (can produce negative errorbars
                 near zero; only use for diagnostics).

    Returns
    -------
    HaloSpectrum

    Raises
    ------
    FileNotFoundError
        If mcmc_dir does not exist or contains no mcmc_results_k*.npz files.
    KeyError
        If the NFW label cannot be matched in any energy bin.
    """
    mcmc_dir = Path(mcmc_dir)
    if not mcmc_dir.exists():
        raise FileNotFoundError(
            f"MCMC results directory not found: {mcmc_dir}\n"
            "Run the Totani_paper_check pipeline first, or set REPO_PATH correctly."
        )

    # Discover all result files
    npz_files = sorted(mcmc_dir.glob("mcmc_results_k*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No mcmc_results_k*.npz files found in {mcmc_dir}"
        )

    nE = len(npz_files)
    if n_energy_bins is not None and nE != n_energy_bins:
        import warnings
        warnings.warn(
            f"Expected {n_energy_bins} energy bins but found {nE} npz files in {mcmc_dir}.",
            stacklevel=2,
        )

    # Optionally load the energy axis from the counts FITS
    E_bins_MeV = np.full(nE, np.nan)
    if counts_path is not None:
        try:
            from totani_helpers.totani_io import read_counts_and_ebounds
            import sys
            sys.path.insert(0, str(_REPO_DIR / "Totani_paper_check"))
            _, _, _, _, ectr, _ = read_counts_and_ebounds(str(counts_path))
            E_bins_MeV[:min(nE, len(ectr))] = ectr[:nE]
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Could not load energy axis from {counts_path}: {exc}. "
                "Falling back to Ectr_mev stored in npz files.",
                stacklevel=2,
            )

    # Allocate output arrays
    f_p50   = np.full(nE, np.nan)
    f_p16   = np.full(nE, np.nan)
    f_p84   = np.full(nE, np.nan)
    iso_e2  = np.full(nE, _ISO_TARGET_E2_DEFAULT)
    matched_label: Optional[str] = None
    n_loaded = 0

    for path in npz_files:
        # Extract bin index from filename
        stem = path.stem  # e.g. 'mcmc_results_k06'
        try:
            k = int(stem.split("_k")[-1])
        except ValueError:
            continue

        if k >= nE:
            continue

        npz = np.load(path, allow_pickle=True)

        # Energy axis from npz if not loaded from FITS
        if not np.isfinite(E_bins_MeV[k]) and "Ectr_mev" in npz:
            E_bins_MeV[k] = float(npz["Ectr_mev"])

        # iso_target_e2 for this bin
        if "iso_target_e2" in npz:
            val = npz["iso_target_e2"]
            v = float(val) if val.ndim == 0 else float(val.flat[0])
            if np.isfinite(v) and v > 0.0:
                iso_e2[k] = v

        # Match NFW label
        labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
        idx = _find_nfw_index(labels, nfw_label, k)
        if idx is None:
            continue

        # Track which label was matched (for reporting)
        if matched_label is None:
            matched_label = labels[idx]

        # Read posterior statistics
        stat_map = {
            "f_p50": "f_p50",
            "f_ml":  "f_ml",
        }
        central_key = stat_map.get(central_stat, "f_p50")
        if central_key not in npz or "f_p16" not in npz or "f_p84" not in npz:
            continue

        f_p50[k] = float(np.atleast_1d(npz[central_key])[idx])
        f_p16[k] = float(np.atleast_1d(npz["f_p16"])[idx])
        f_p84[k] = float(np.atleast_1d(npz["f_p84"])[idx])
        n_loaded += 1

    if n_loaded == 0:
        raise KeyError(
            f"No NFW halo component could be matched in any energy bin in {mcmc_dir}. "
            f"Tried label: {nfw_label!r}. "
            "Check that the MCMC runs completed and that the NFW template label is correct."
        )

    if matched_label is None:
        matched_label = nfw_label or "nfw (inferred)"

    # Convert f -> E^2 dN/dE at the galactic pole
    phi     = f_p50 * iso_e2
    phi_p16 = f_p16 * iso_e2
    phi_p84 = f_p84 * iso_e2

    phi_err_lo  = np.maximum(phi - phi_p16, 0.0)
    phi_err_hi  = np.maximum(phi_p84 - phi, 0.0)
    phi_err_sym = 0.5 * (phi_err_lo + phi_err_hi)

    return HaloSpectrum(
        E_bins_MeV    = E_bins_MeV,
        E_bins_GeV    = E_bins_MeV / 1000.0,
        phi           = phi,
        phi_p16       = phi_p16,
        phi_p84       = phi_p84,
        phi_err_lo    = phi_err_lo,
        phi_err_hi    = phi_err_hi,
        phi_err_sym   = phi_err_sym,
        iso_target_e2 = iso_e2,
        f_nfw_p50     = f_p50,
        f_nfw_p16     = f_p16,
        f_nfw_p84     = f_p84,
        nfw_label     = matched_label,
        mcmc_dir      = str(mcmc_dir),
        n_loaded      = n_loaded,
    )


def load_component_spectra(
    mcmc_dir: str | Path,
    *,
    labels: Optional[list[str]] = None,
    counts_path: Optional[str | Path] = None,
    central_stat: str = "f_p50",
) -> ComponentSpectra:
    """
    Load fitted spectra for all requested MCMC template components.

    This is a setup helper for multi-component scattering calculations. It
    exposes the spectra needed to build PhotonTransferComponent objects, but it
    deliberately does not assign scattering geometries. A disk foreground, an
    isotropic component, and an NFW halo component should not silently inherit
    the same optical-depth average.
    """
    mcmc_dir = Path(mcmc_dir)
    if not mcmc_dir.exists():
        raise FileNotFoundError(f"MCMC results directory not found: {mcmc_dir}")

    npz_files = sorted(mcmc_dir.glob("mcmc_results_k*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No mcmc_results_k*.npz files found in {mcmc_dir}")

    nE = len(npz_files)
    E_bins_MeV = np.full(nE, np.nan)
    if counts_path is not None:
        try:
            from totani_helpers.totani_io import read_counts_and_ebounds
            import sys
            sys.path.insert(0, str(_REPO_DIR / "Totani_paper_check"))
            _, _, _, _, ectr, _ = read_counts_and_ebounds(str(counts_path))
            E_bins_MeV[:min(nE, len(ectr))] = ectr[:nE]
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Could not load energy axis from {counts_path}: {exc}. "
                "Falling back to Ectr_mev stored in npz files.",
                stacklevel=2,
            )

    first = np.load(npz_files[0], allow_pickle=True)
    available = [str(x) for x in np.atleast_1d(first["labels"]).tolist()]
    selected = available if labels is None else [str(x) for x in labels]
    missing = [lab for lab in selected if lab not in available]
    if missing:
        raise KeyError(
            f"Requested component labels not found in {npz_files[0]}: {missing}. "
            f"Available labels: {available}"
        )

    central_key = {"f_p50": "f_p50", "f_ml": "f_ml"}.get(central_stat, "f_p50")
    f_mid = {lab: np.full(nE, np.nan) for lab in selected}
    f_p16 = {lab: np.full(nE, np.nan) for lab in selected}
    f_p84 = {lab: np.full(nE, np.nan) for lab in selected}
    iso_e2 = np.full(nE, _ISO_TARGET_E2_DEFAULT)

    for path in npz_files:
        try:
            k = int(path.stem.split("_k")[-1])
        except ValueError:
            continue
        if k >= nE:
            continue

        npz = np.load(path, allow_pickle=True)
        if not np.isfinite(E_bins_MeV[k]) and "Ectr_mev" in npz:
            E_bins_MeV[k] = float(npz["Ectr_mev"])
        if "iso_target_e2" in npz:
            val = npz["iso_target_e2"]
            v = float(val) if val.ndim == 0 else float(val.flat[0])
            if np.isfinite(v) and v > 0.0:
                iso_e2[k] = v

        path_labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
        if central_key not in npz or "f_p16" not in npz or "f_p84" not in npz:
            continue

        for lab in selected:
            if lab not in path_labels:
                continue
            idx = path_labels.index(lab)
            f_mid[lab][k] = float(np.atleast_1d(npz[central_key])[idx])
            f_p16[lab][k] = float(np.atleast_1d(npz["f_p16"])[idx])
            f_p84[lab][k] = float(np.atleast_1d(npz["f_p84"])[idx])

    phi = {lab: f_mid[lab] * iso_e2 for lab in selected}
    phi16 = {lab: f_p16[lab] * iso_e2 for lab in selected}
    phi84 = {lab: f_p84[lab] * iso_e2 for lab in selected}
    err = {
        lab: 0.5 * (np.maximum(phi[lab] - phi16[lab], 0.0)
                    + np.maximum(phi84[lab] - phi[lab], 0.0))
        for lab in selected
    }

    return ComponentSpectra(
        E_bins_MeV=E_bins_MeV,
        E_bins_GeV=E_bins_MeV / 1000.0,
        labels=selected,
        phi=phi,
        phi_p16=phi16,
        phi_p84=phi84,
        phi_err_sym=err,
        iso_target_e2=iso_e2,
        mcmc_dir=str(mcmc_dir),
    )


def _find_nfw_index(
    labels: list[str],
    requested: Optional[str],
    k: int,
) -> Optional[int]:
    """
    Find the index of the NFW component in a list of labels.

    If `requested` is given, match it exactly first; fall back to
    case-insensitive substring match on 'nfw'. If exactly one candidate
    is found, return its index. If zero or multiple, return None.
    """
    if requested is not None and requested in labels:
        return labels.index(requested)

    # Substring search
    candidates = [i for i, lab in enumerate(labels) if "nfw" in lab.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        import warnings
        warnings.warn(
            f"Multiple NFW labels in bin k={k}: "
            f"{[labels[i] for i in candidates]}. "
            "Pass nfw_label= explicitly to resolve ambiguity. Skipping bin.",
            stacklevel=3,
        )
    return None


# ---------------------------------------------------------------------------
# Named convenience loaders
# ---------------------------------------------------------------------------

def load_halo_spectrum_rho25(
    nfw_label: Optional[str] = None,
    counts_path: Optional[str | Path] = None,
) -> HaloSpectrum:
    """NFW rho^2.5, disk excluded (Totani Fig. 5 / Fig. 8 top panel)."""
    return load_halo_spectrum(
        _MCMC_DIRS["rho2.5"],
        nfw_label=nfw_label,
        counts_path=counts_path,
    )


def load_halo_spectrum_rho2(
    nfw_label: Optional[str] = None,
    counts_path: Optional[str | Path] = None,
) -> HaloSpectrum:
    """
    NFW rho^2, disk excluded.

    This is Totani's primary result and the one to use for the
    annihilation + scattering parameter extraction.
    """
    return load_halo_spectrum(
        _MCMC_DIRS["rho2"],
        nfw_label=nfw_label,
        counts_path=counts_path,
    )


def load_halo_spectrum_rho1(
    nfw_label: Optional[str] = None,
    counts_path: Optional[str | Path] = None,
) -> HaloSpectrum:
    """NFW rho^1, disk excluded (Totani Fig. 7 / Fig. 8 bottom panel)."""
    return load_halo_spectrum(
        _MCMC_DIRS["rho1"],
        nfw_label=nfw_label,
        counts_path=counts_path,
    )


# ---------------------------------------------------------------------------
# Utility: list available MCMC directories
# ---------------------------------------------------------------------------

def available_mcmc_dirs() -> dict[str, str]:
    """Return a dict of profile_name -> absolute_path for existing MCMC dirs."""
    return {
        name: str(path)
        for name, path in _MCMC_DIRS.items()
        if path.exists()
    }


# ---------------------------------------------------------------------------
# Drop-in replacement arrays for attenuation_eft.py / fit_totani_dm_scattering.py
# ---------------------------------------------------------------------------

def get_phi_totani_from_mcmc(
    profile: str = "rho2",
    nfw_label: Optional[str] = None,
    counts_path: Optional[str | Path] = None,
    err_mode: str = "sym",
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (E_bins_GeV, phi, sigma) as drop-in replacements for the hardcoded
    E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI arrays in attenuation_eft.py.

    Parameters
    ----------
    profile : str
        One of 'rho2.5', 'rho2' (default), 'rho1'.
    nfw_label : str, optional
        Explicit NFW template label. If None, inferred automatically.
    counts_path : str or Path, optional
        Path to the counts CCUBE FITS for the energy axis.
    err_mode : str
        Error convention: 'sym' (default), 'lo', 'hi', 'max'.
    positive_only : bool
        If True, return only bins where phi > 0.

    Returns
    -------
    E_bins_GeV : (nE,)  [GeV]
    phi        : (nE,)  [MeV cm^-2 s^-1 sr^-1]
    sigma      : (nE,)  [MeV cm^-2 s^-1 sr^-1]
    """
    mcmc_dir = _MCMC_DIRS.get(profile)
    if mcmc_dir is None:
        raise ValueError(
            f"Unknown profile {profile!r}. "
            f"Available: {list(_MCMC_DIRS.keys())}"
        )
    hs = load_halo_spectrum(mcmc_dir, nfw_label=nfw_label, counts_path=counts_path)
    return hs.to_fit_arrays(positive_only=positive_only, err_mode=err_mode)


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Available MCMC directories:")
    for name, path in available_mcmc_dirs().items():
        print(f"  {name}: {path}")

    print("\nLoading Fig. 6 (rho^2, primary Totani result)...")
    try:
        hs = load_halo_spectrum_rho2()
        print(hs.summary())
        print("\nFirst 5 bins:")
        for k in range(min(5, len(hs.E_bins_GeV))):
            print(
                f"  k={k:02d}  E={hs.E_bins_GeV[k]:.2f} GeV  "
                f"phi={hs.phi[k]:.3e}  +{hs.phi_err_hi[k]:.2e}/-{hs.phi_err_lo[k]:.2e}"
                f"  [MeV cm^-2 s^-1 sr^-1]"
            )
        E, phi, sigma = hs.to_fit_arrays(positive_only=True)
        print(f"\nPositive-bin fit arrays: {len(E)} bins, "
              f"E in [{E[0]:.2f}, {E[-1]:.2f}] GeV")
    except FileNotFoundError as e:
        print(f"  Could not load: {e}")
        print("  Run the Totani_paper_check MCMC pipeline first.")
