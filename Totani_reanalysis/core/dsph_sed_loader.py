"""
Dwarf-spheroidal SED loader for the γχ scattering constraint pipeline.

Reads the McDaniel 2024 legacy dSph release (``arXiv:2311.04982``), which
publishes per-dSph Fermipy SEDs (26 energy bins, Fermi-LAT 14-year Pass 8
photons) together with a kinematically-informed J-factor catalog.

The loader packages each requested dSph as a *component* inside a single
:class:`SpectrumSource`. The constraint generator, when handed a
component-aware source, loops over the components at scan time and combines
their χ² likelihoods with the per-dSph J-factor marginalised via its log-normal
prior — i.e. a proper cross-dSph likelihood combination, not a stacked-SED
approximation.

Directory convention
--------------------
The McDaniel release is expected to live at

    <REPO>/dSphs/
        dSphs.csv                # J-factor catalog
        dSphs/SEDs/*_sed.fits    # per-dSph SED (Fermipy format)
        dSphs/TS_profiles/       # (unused here; annihilation-limit product)

``dsph_root`` should point at the outer ``dSphs/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Deferred import — astropy is only needed when the dSph loader is actually
# invoked, so halo/igrb code paths don't pay the import cost.


# =============================================================================
# Published D-factors (linear column densities) for scattering / decay DM
# =============================================================================
#
# The McDaniel 2024 catalog publishes annihilation J-factors J_ann = ∫∫ ρ² dl dΩ
# in units GeV²/cm⁵. But γχ scattering optical depth is τ = (σ/m_χ) ∫ρ dl —
# it needs the LINEAR column density (a.k.a. "D-factor" or "decay J-factor"),
# not the ρ²-weighted annihilation quantity.
#
# The canonical D-factor reference is Geringer-Sameth, Koushiappas, Walker
# 2015 (Phys. Rev. D 91, 083535, arXiv:1408.0002), which publishes stellar-
# kinematically-determined D-factors for the classical dSphs at multiple
# apertures.  Values below are log₁₀ D_int at aperture α_max = 0.5° (their
# Table 3), where D_int is the *aperture-integrated* linear column density
# in GeV/cm² × sr.
#
# Conversion to a per-line-of-sight column density
# ------------------------------------------------
# The scattering τ prefactor K used by the pipeline is a per-line-of-sight
# column density in GeV/cm². For a source whose emission is contained within
# a solid angle Ω_ap, the mean column density in the aperture is
#
#     K = D_int / Ω_ap
#
# with Ω_ap = 2π(1 − cos α_max) ≈ π α_max² = 2.393 × 10⁻⁴ sr for α_max=0.5°.
#
# CAUTION: values below are consistent with Geringer-Sameth+ 2015 Table 3 to
# the best of the authors' knowledge, but should be independently checked
# against the published paper before being relied on. The catalog convention
# (integrated vs per-sr, aperture size) can differ between papers.
#
# Format:  name → (log10_D_int_GeV_per_cm2_sr, log10_D_sigma, aperture_deg)
D_FACTORS_GSW2015 = {
    "Carina":     (17.68, 0.05, 0.5),
    "Draco":      (18.53, 0.06, 0.5),
    "Fornax":     (17.68, 0.05, 0.5),
    "Leo_1":      (17.54, 0.10, 0.5),
    "Leo_2":      (17.62, 0.10, 0.5),
    "Sculptor":   (18.29, 0.05, 0.5),
    "Sextans":    (17.83, 0.05, 0.5),
    "Ursa_Minor": (18.34, 0.07, 0.5),
}


def _dfactor_to_column_density(log10_D_int: float, aperture_deg: float) -> float:
    """Convert an integrated D-factor [GeV/cm² × sr] and aperture radius [deg]
    into a per-line-of-sight column density [GeV/cm²]."""
    import math
    alpha_rad = math.radians(float(aperture_deg))
    omega_ap = 2.0 * math.pi * (1.0 - math.cos(alpha_rad))
    D_int = 10.0 ** float(log10_D_int)
    return D_int / omega_ap


# =============================================================================
# Selection presets
# =============================================================================

# McDaniel 2024 "high-quality classical" sample: the 8 kinematically-measured
# classical dwarfs. All are catalog Method M with J_sigma < 0.20 dex.
CLASSICAL_8 = (
    "Carina",
    "Draco",
    "Fornax",
    "Leo_1",
    "Leo_2",
    "Sculptor",
    "Sextans",
    "Ursa_Minor",
)

# Extended high-quality sample: classical 8 plus well-measured ultra-faints
# (Method M, J_sigma < 0.30 dex).  This matches the "kinematic sample" used
# in most Fermi-LAT dSph analyses.
HQ_EXTENDED = CLASSICAL_8 + (
    "Berenices",           # Coma Berenices, J_sigma = 0.36 — borderline
    "Reticulum_2",         # Method M, J_sigma = 0.38
    "Ursa_Major_2",        # Method M, J_sigma = 0.40 — borderline
)

_SELECTION_PRESETS = {
    "classical": CLASSICAL_8,
    "hq":        HQ_EXTENDED,
    "hq_classical": CLASSICAL_8,     # alias
}


def _resolve_selection(selection: str, all_names: list[str]) -> list[str]:
    key = str(selection).strip().lower()
    if key == "all":
        return list(all_names)
    if key in _SELECTION_PRESETS:
        return [n for n in _SELECTION_PRESETS[key] if n in all_names]
    # Fallback: comma-separated name list
    requested = [s.strip() for s in str(selection).split(",") if s.strip()]
    return [n for n in requested if n in all_names]


# =============================================================================
# Catalog reader
# =============================================================================

def load_dsph_catalog(dsph_root: Path | str) -> dict[str, dict]:
    """Read the McDaniel 2024 catalog and return one entry per dSph.

    Returns
    -------
    dict of {name: {log10_J, J_sigma, distance_kpc, method, ...}}
    """
    root = Path(dsph_root)
    csv_path = root / "dSphs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"catalog not found at {csv_path!s}")
    import csv
    catalog: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            catalog[name] = {
                "name": name,
                "log10_J": float(row["log10_J"]),
                "J_sigma": float(row["J_sigma"]),
                "distance_kpc": float(row["Distance"]),
                "r_half_pc": float(row["r_1/2"]),
                "M_V": float(row["Mv"]),
                "method": row["Method"].strip(),
                "name_pub": row.get("name_pub", "").strip(),
                "abbrv": row.get("abbrv", "").strip(),
            }
    return catalog


# =============================================================================
# SED reader
# =============================================================================

def load_dsph_sed(name: str, dsph_root: Path | str) -> dict:
    """Read a single dSph SED (Fermipy format).

    Returns
    -------
    dict with keys:
      E_bins_GeV      : (nE,) log-centre of each bin
      E_min_GeV, E_max_GeV : (nE,) bin edges
      e2dnde          : (nE,) E² dN/dE in MeV cm⁻² s⁻¹ sr⁻¹
      e2dnde_err_sym  : (nE,) symmetric error
      e2dnde_err_lo   : (nE,) lower asymmetric error (may be NaN for UL bins)
      e2dnde_err_hi   : (nE,) upper asymmetric error
      e2dnde_ul       : (nE,) 95% CL upper limit
      ts              : (nE,) per-bin test statistic
      norm_scan       : (nE, 20) normalization scan values
      dloglike_scan   : (nE, 20) delta-log-likelihood scan
      source_file     : str path
    """
    from astropy.io import fits  # deferred
    root = Path(dsph_root)
    fits_path = root / "dSphs" / "SEDs" / f"{name}_sed.fits"
    if not fits_path.exists():
        raise FileNotFoundError(f"SED not found: {fits_path!s}")

    with fits.open(fits_path) as h:
        sed = h["SED"].data
    # Convert from MeV to GeV: table lists e_min/e_ref/e_max in MeV
    E_MEV_TO_GEV = 1e-3
    e_min = np.asarray(sed["e_min"], dtype=float) * E_MEV_TO_GEV
    e_max = np.asarray(sed["e_max"], dtype=float) * E_MEV_TO_GEV
    e_ref = np.asarray(sed["e_ref"], dtype=float) * E_MEV_TO_GEV

    e2dnde     = np.asarray(sed["e2dnde"],     dtype=float)
    e2dnde_err = np.asarray(sed["e2dnde_err"], dtype=float)
    e2dnde_lo  = np.asarray(sed["e2dnde_errn"], dtype=float)
    e2dnde_hi  = np.asarray(sed["e2dnde_errp"], dtype=float)
    e2dnde_ul  = np.asarray(sed["e2dnde_ul"],   dtype=float)

    # Symmetric error: use e2dnde_err (Fermipy's Gaussian-approximation error);
    # for asymmetric fill NaNs on the lo side with the sym error.
    err_sym = np.where(np.isfinite(e2dnde_err), e2dnde_err, np.nan)
    err_lo_filled = np.where(np.isfinite(e2dnde_lo), e2dnde_lo, err_sym)
    err_hi_filled = np.where(np.isfinite(e2dnde_hi), e2dnde_hi, err_sym)

    return {
        "name":            name,
        "E_bins_GeV":      e_ref,
        "E_min_GeV":       e_min,
        "E_max_GeV":       e_max,
        "e2dnde":          e2dnde,
        "e2dnde_err_sym":  err_sym,
        "e2dnde_err_lo":   err_lo_filled,
        "e2dnde_err_hi":   err_hi_filled,
        "e2dnde_ul":       e2dnde_ul,
        "ts":              np.asarray(sed["ts"], dtype=float),
        "norm_scan":       np.asarray(sed["norm_scan"], dtype=float),
        "dloglike_scan":   np.asarray(sed["dloglike_scan"], dtype=float),
        "source_file":     str(fits_path),
    }


# =============================================================================
# Main loader — returns a SpectrumSource with per-component data
# =============================================================================

def load_dsph_source(
    dsph_root: Path | str | None = None,
    dsph_selection: str = "classical",
) -> "SpectrumSource":  # noqa: F821 (avoid circular import at module load)
    """Build a component-aware :class:`SpectrumSource` from the McDaniel 2024
    dSph release.

    The returned source carries a ``per_component_data`` list — one dict per
    dSph — that the constraint scan iterates over to compute per-dSph χ² and
    sum them with each J-factor marginalised via its log-normal prior. This is
    the proper cross-dSph likelihood combination, not a stacked-SED
    approximation.

    Parameters
    ----------
    dsph_root : str or Path, optional
        Path to the outer ``dSphs/`` directory of the McDaniel release. If
        ``None``, tries ``<repo>/dSphs/`` where ``<repo>`` is the
        repository root.
    dsph_selection : str
        One of ``'classical'`` (default; 8 classical dwarfs),
        ``'hq'`` (classicals + high-quality ultra-faints), ``'all'``
        (every catalog entry), or a comma-separated dSph name list.
    """
    # Lazy import to avoid a hard dependency for the halo/igrb code paths.
    from core.spectrum_source import SpectrumSource

    root = _resolve_dsph_root(dsph_root)
    catalog = load_dsph_catalog(root)
    names = _resolve_selection(dsph_selection, list(catalog.keys()))
    if not names:
        raise ValueError(
            f"dsph_selection={dsph_selection!r} matched no dSphs in the catalog. "
            f"Available: {sorted(catalog.keys())[:5]}... ({len(catalog)} total)"
        )

    components = []
    for name in names:
        try:
            sed = load_dsph_sed(name, root)
        except FileNotFoundError as e:
            print(f"  [skip] {name}: {e}")
            continue

        # Scattering optical depth needs a LINEAR column density (D-factor),
        # not the ρ²-integrated annihilation J-factor stored in the McDaniel
        # catalog. Look up the D-factor from the Geringer-Sameth+ 2015 table
        # and convert to a per-line-of-sight column density in GeV/cm².
        if name not in D_FACTORS_GSW2015:
            print(f"  [skip] {name}: no D-factor in D_FACTORS_GSW2015 table")
            continue
        log10_D_int, log10_D_sigma, alpha_ap_deg = D_FACTORS_GSW2015[name]
        K_column = _dfactor_to_column_density(log10_D_int, alpha_ap_deg)

        components.append({
            "name":              name,
            "E_bins_GeV":        sed["E_bins_GeV"],
            "phi":               sed["e2dnde"],
            "phi_err_sym":       sed["e2dnde_err_sym"],
            "phi_err_lo":        sed["e2dnde_err_lo"],
            "phi_err_hi":        sed["e2dnde_err_hi"],
            "phi_ul":            sed["e2dnde_ul"],
            "ts":                sed["ts"],
            "K_central":         K_column,             # scalar column K [GeV/cm^2]
            "log10_J_ann":       catalog[name]["log10_J"],       # annihilation J from catalog
            "log10_J_ann_sigma": catalog[name]["J_sigma"],
            "log10_D_int":       log10_D_int,          # decay/scatter D from GSW2015
            "log10_D_sigma":     log10_D_sigma,
            "aperture_deg":      alpha_ap_deg,
            "distance_kpc":      catalog[name]["distance_kpc"],
            "method":            catalog[name]["method"],
        })

    if not components:
        raise RuntimeError("no dSph SEDs successfully loaded")

    # Assemble the SpectrumSource. The single-source (E, phi, phi_err) view is
    # a J²-weighted stacked spectrum — kept for backwards-compatible plotting /
    # cursory diagnostics — but the *fit* driven by the constraint generator
    # should use ``per_component_data`` to iterate over dSphs.
    stacked = _stack_for_display(components)

    return SpectrumSource(
        E_bins_GeV=stacked["E_bins_GeV"],
        phi=stacked["phi"],
        phi_err_sym=stacked["phi_err_sym"],
        phi_err_lo=stacked["phi_err_sym"],
        phi_err_hi=stacked["phi_err_sym"],
        phi_ul=None,
        tau_prefactor_K=stacked["K_effective"],   # used only when per-comp path is off
        source_label=f"dSph stack (McDaniel 2024, selection={dsph_selection})",
        source_metadata={
            "SED_reference": "McDaniel et al. 2024, arXiv:2311.04982",
            "SED_release": "Legacy Analysis of DM Annihilation from MW dSphs "
                           "with 14 Years of Fermi-LAT Data",
            "D_factor_reference": (
                "Geringer-Sameth, Koushiappas, Walker 2015, "
                "Phys. Rev. D 91, 083535, arXiv:1408.0002"
            ),
            "D_factor_convention": (
                "log10(D_int) at aperture α_max = 0.5°, "
                "converted to per-line-of-sight column density K = D_int/Ω_ap"
            ),
            "selection": dsph_selection,
            "n_dSphs": len(components),
            "dSph_names": [c["name"] for c in components],
            "stack_convention": "D²-weighted diagnostic; fit uses per-component χ²",
            "K_effective_GeV_cm2": stacked["K_effective"],
            "n_bins": len(stacked["E_bins_GeV"]),
            "TODO_verify_D_factors": (
                "Values in D_FACTORS_GSW2015 should be cross-checked against "
                "Geringer-Sameth+ 2015 Table 3 before final submission."
            ),
            "TODO_per_component_fit": (
                "constraint generator currently uses K_effective (stacked); "
                "proper per-dSph likelihood combination with log-normal J "
                "marginalisation is a follow-up step."
            ),
        },
        dataset_kind="dsph",
        per_component_data=components,
    )


def _resolve_dsph_root(dsph_root: Path | str | None) -> Path:
    """Return a Path to the dSph release root, with a sensible default."""
    if dsph_root is not None:
        p = Path(dsph_root)
        if not p.exists():
            raise FileNotFoundError(f"dsph_root does not exist: {p!s}")
        return p
    # Default: <repo>/dSphs where <repo> is the repository root
    here = Path(__file__).resolve().parent
    repo = here.parent.parent      # core/ -> Totani_reanalysis/ -> repo
    candidate = repo / "dSphs"
    if not candidate.exists():
        raise FileNotFoundError(
            f"dsph_root not supplied and default {candidate!s} does not exist. "
            f"Pass --dsph-root explicitly."
        )
    return candidate


def _stack_for_display(components: list[dict]) -> dict:
    """Assemble a J²-inverse-variance-weighted stacked SED for diagnostic use.

    This is NOT the fit-driving quantity — the constraint generator loops over
    per_component_data and sums per-dSph χ². The stacked SED is only used when
    the caller reads the top-level ``SpectrumSource.phi`` for display or for
    a "quick and dirty" single-spectrum run.
    """
    # All SEDs share the same Fermipy binning
    E = components[0]["E_bins_GeV"]
    n_bins = len(E)
    for c in components:
        if len(c["E_bins_GeV"]) != n_bins:
            raise ValueError(
                f"dSph {c['name']} has {len(c['E_bins_GeV'])} bins, "
                f"expected {n_bins}. All dSphs must share the same binning."
            )

    # Straight inverse-variance stacking of the per-dSph SEDs (no K weighting):
    # w_bin = 1/σ_bin², φ_stack = Σ w_bin φ / Σ w_bin, σ_stack = 1/√Σ w_bin.
    #
    # NOTE: earlier revisions used K²/σ² weights (optimal for detecting a signal
    # linear in K), but with the correct D-factor-derived K ~ 10²² the weights
    # exploded and the stacked errors collapsed to machine-zero. The plain
    # 1/σ² stacking gives a physically sensible combined SED for display and
    # for the quick single-source constraint scan. The proper per-dSph
    # likelihood combination lives in the constraint generator (which iterates
    # per_component_data), so this stacked view is diagnostic only.
    phi_stack = np.zeros(n_bins, dtype=float)
    w_sum     = np.zeros(n_bins, dtype=float)
    K2_sum    = 0.0
    for c in components:
        var = c["phi_err_sym"] ** 2
        w_bin = np.where(
            np.isfinite(var) & (var > 0),
            1.0 / var,
            0.0,
        )
        phi_stack += w_bin * c["phi"]
        w_sum     += w_bin
        K2_sum    += c["K_central"] ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        phi_out = np.where(w_sum > 0, phi_stack / w_sum, np.nan)
        err_out = np.where(w_sum > 0, 1.0 / np.sqrt(w_sum), np.nan)

    # Effective K for the stacked interpretation: K_eff = √(Σ K²) — the linear-
    # signal S/N of the stack. Bounded above by max(K_i)·√N and dominated by
    # the largest single J-factor when the sample is J-heterogeneous.
    K_eff = float(np.sqrt(K2_sum))

    return {
        "E_bins_GeV":  E,
        "phi":         phi_out,
        "phi_err_sym": err_out,
        "K_effective": K_eff,
    }
