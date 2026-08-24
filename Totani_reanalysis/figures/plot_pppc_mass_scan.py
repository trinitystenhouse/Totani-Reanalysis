#!/usr/bin/env python3
"""
plot_pppc_mass_scan.py
======================
Publication-quality plots from the PPPC mass scan results.

Produces four figures:

Figure 1 — profile_likelihood.pdf
    Spectral Delta ln L (m_chi) for each channel on the same axes.
    95% CL interval from Delta lnL = -1.92 shown.

Figure 2 — sigmav_vs_mass.pdf
    <sigma v>_best(m_chi) for each channel vs. configured dSph upper limits.
    Totani's pure-annihilation reported range overlaid.
    Thermal relic line at 3e-26 cm^3/s.

Figure 3 — tension_vs_mass.pdf
    Tension factor <sigma v>_best / <sigma v>_dSph as a function of mass.
    Unity line (no tension) and Totani baseline (factor ~3-5) marked.

Figure 4 — halo_spectrum_bestfit.pdf
    Best-fit single-normalization PPPC model spectrum vs the chosen halo
    reference data for each channel's preferred mass.

Usage
-----
    # After running run_pppc_mass_scan.py for all three channels:
    python figures/plot_pppc_mass_scan.py

    # Custom results directories or single channel:
    python figures/plot_pppc_mass_scan.py --channels WW --results-dir results/pppc_mass_scan

    # Totani Fig. 8-style PPPC spectrum panel at a chosen mass:
    python figures/plot_pppc_mass_scan.py --channels WW --style totani --spectrum-mass 650
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))
sys.path.insert(0, str(_REPO / "Totani_reanalysis" / "core"))

RESULTS_BASE = _PROJECT / "results" / "pppc_mass_scan"
PLOT_DIR     = _HERE / "plots_pppc"

FIT_RESULT_SUBDIR = {
    "fig6": None,
    "global_fig6": "global_fig6",
    "global_fig6_w_disk": "global_fig6_w_disk",
    "pixelwise_global_fig6": "pixelwise_global_fig6",
}

FIT_MCMC_PREFIX = {
    "fig6": "",
    "global_fig6": "global_fig6_",
    "global_fig6_w_disk": "global_fig6_w_disk_",
    "pixelwise_global_fig6": "pixelwise_global_fig6_",
}

DEFAULT_PROFILE_MAX_ABS_LOGLIKE = 1e10
DEFAULT_PROFILE_MAX_ABS_DLOGLIKE = 1e6

HALO_REFERENCE_CHOICES = [
    "auto",
    "totani_rho2",
    "global_rho2",
    "global_rho2_w_disk",
    "pixelwise_global_rho2",
]

from dsph_limits import (  # noqa: E402
    DSph_LIMIT_SOURCES,
    available_dsph_channels,
    dsph_limit_source,
    dsph_upper_limit,
    has_dsph_limit_table,
    load_dsph_limit_table,
)

CHANNEL_STYLE = {
    "WW":     dict(color="C0", ls="-",  marker="o", label=r"$W^+W^-$"),
    "bb":     dict(color="C1", ls="--", marker="s", label=r"$b\bar{b}$"),
    "tautau": dict(color="C2", ls=":",  marker="^", label=r"$\tau^+\tau^-$"),
}

THERMAL_RELIC = 3e-26   # cm^3/s
SIGMA_V_REF = 1e-25     # must match run_pppc_mass_scan.py / build_pppc_template.py

_PPPC_TABLE_CACHE: dict[Path, tuple[list[str], np.ndarray]] = {}
_PPPC_CHANNELS = {
    "WW": "W",
    "bb": "b",
    "tautau": r"\[Tau]",
}


def _normalise_cli_channel(channel: str) -> str:
    return "tautau" if str(channel).strip().lower() in ("tau", "tautau") else str(channel)


def _validate_dsph_source_channels(channels: list[str], dsph_source: str) -> None:
    missing = [ch for ch in channels if not has_dsph_limit_table(ch, dsph_source)]
    if missing:
        available = ", ".join(available_dsph_channels(dsph_source))
        raise SystemExit(
            f"ERROR: dSph source {dsph_source!r} has no limit table for: {', '.join(missing)}. "
            f"Available channels for this source: {available}. "
            "Use --dsph-source hoof for WW, or plot only the McDaniel channels."
        )


def profile_likelihood_from_bin_loglikes(
    loglike_by_mass: np.ndarray,
    *,
    max_abs_loglike: float | None,
    max_abs_dlnl: float | None,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lnL = np.asarray(loglike_by_mass, float).copy()
    if lnL.ndim != 2:
        raise ValueError(f"loglike_by_mass must be 2D, got shape {lnL.shape}")

    tag = f" for {label}" if label else ""
    if max_abs_loglike is not None and max_abs_loglike > 0:
        bad = np.isfinite(lnL) & (np.abs(lnL) > float(max_abs_loglike))
        if np.any(bad):
            print(f"  [profile] Masked {int(np.count_nonzero(bad))} pathological lnL values{tag}")
            lnL[bad] = np.nan

    per_bin_best = np.full(lnL.shape[1], np.nan)
    for k in range(lnL.shape[1]):
        finite = np.isfinite(lnL[:, k])
        if np.any(finite):
            per_bin_best[k] = np.nanmax(lnL[finite, k])

    dlnL_by_bin = lnL - per_bin_best[None, :]
    if max_abs_dlnl is not None and max_abs_dlnl > 0:
        bad = np.isfinite(dlnL_by_bin) & (np.abs(dlnL_by_bin) > float(max_abs_dlnl))
        if np.any(bad):
            print(f"  [profile] Masked {int(np.count_nonzero(bad))} pathological per-bin ΔlnL values{tag}")
            dlnL_by_bin[bad] = np.nan

    bins_used = np.all(np.isfinite(dlnL_by_bin), axis=0)
    profile_lnL = np.full(lnL.shape[0], np.nan)
    if np.any(bins_used):
        profile_lnL = np.sum(dlnL_by_bin[:, bins_used], axis=1)
    else:
        print(f"  [profile] WARNING: no common finite energy bins remain{tag}")

    delta_lnL = profile_lnL.copy()
    if np.any(np.isfinite(delta_lnL)):
        delta_lnL = delta_lnL - np.nanmax(delta_lnL)

    print(f"  [profile] Using {int(np.count_nonzero(bins_used))}/{lnL.shape[1]} common energy bins{tag}")
    return profile_lnL, delta_lnL, bins_used


def _reprofile_channel_data(
    data: dict,
    channel: str,
    *,
    fit: str,
    max_abs_loglike: float | None,
    max_abs_dlnl: float | None,
) -> dict:
    if "loglike_ml_by_bin" in data:
        loglike_by_bin = np.asarray(data["loglike_ml_by_bin"], float)
    elif "mcmc_dirs" in data and "pppc_labels" in data:
        rows = []
        for mcmc_dir_raw, label_raw in zip(np.asarray(data["mcmc_dirs"], object), np.asarray(data["pppc_labels"], object)):
            post = _load_pppc_posteriors(Path(str(mcmc_dir_raw)), str(label_raw))
            if post is None:
                rows.append(np.full(13, np.nan))
            else:
                rows.append(np.asarray(post["loglike_ml"], float))
        loglike_by_bin = np.vstack(rows) if rows else np.empty((0, 13))
    else:
        return data

    if loglike_by_bin.size == 0:
        return data

    profile_lnL, delta_lnL, bins_used = profile_likelihood_from_bin_loglikes(
        loglike_by_bin,
        max_abs_loglike=max_abs_loglike,
        max_abs_dlnl=max_abs_dlnl,
        label=f"{fit}/{channel}",
    )
    out = dict(data)
    if "total_lnL_raw" not in out and "total_lnL" in out:
        out["total_lnL_raw"] = np.asarray(out["total_lnL"], float)
    out["total_lnL"] = profile_lnL.astype(np.float32)
    out["delta_lnL"] = delta_lnL.astype(np.float32)
    out["loglike_ml_by_bin"] = loglike_by_bin.astype(np.float32)
    out["profile_bins_used"] = bins_used.astype(bool)
    out["profile_max_abs_loglike"] = np.float32(max_abs_loglike if max_abs_loglike is not None else 0.0)
    out["profile_max_abs_dlnl"] = np.float32(max_abs_dlnl if max_abs_dlnl is not None else 0.0)
    return out


def load_channel(
    channel: str,
    results_dir: Path,
    *,
    fit: str,
    dsph_source: str,
    max_abs_loglike: float | None,
    max_abs_dlnl: float | None,
) -> dict | None:
    npz_path = results_dir / channel / "mass_scan_results.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        loaded = {k: data[k] for k in data.files}
        reprofiled = _reprofile_channel_data(
            loaded,
            channel,
            fit=fit,
            max_abs_loglike=max_abs_loglike,
            max_abs_dlnl=max_abs_dlnl,
        )
        if "loglike_ml_by_bin" not in reprofiled:
            print(f"  [profile] {npz_path} lacks per-bin/profile metadata; scanning MCMC dirs instead")
            fallback = load_channel_from_mcmc_dirs(
                channel,
                fit=fit,
                dsph_source=dsph_source,
                max_abs_loglike=max_abs_loglike,
                max_abs_dlnl=max_abs_dlnl,
            )
            return fallback if fallback is not None else reprofiled
        return reprofiled

    print(f"  [partial] {npz_path} not found; scanning completed MCMC dirs for {channel}")
    return load_channel_from_mcmc_dirs(
        channel,
        fit=fit,
        dsph_source=dsph_source,
        max_abs_loglike=max_abs_loglike,
        max_abs_dlnl=max_abs_dlnl,
    )


def _dsph_limit(m_GeV: float, channel: str, *, source: str = "hoof") -> float:
    return dsph_upper_limit(m_GeV, channel, source=source)


def _load_pppc_posteriors(mcmc_dir: Path, pppc_label: str, nE: int = 13) -> dict | None:
    f_p50 = np.full(nE, np.nan)
    lnL = np.full(nE, np.nan)
    n_ok = 0

    for k in range(nE):
        path = mcmc_dir / f"mcmc_results_k{k:02d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as npz:
            labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
            idx = labels.index(pppc_label) if pppc_label in labels else None
            if idx is None:
                cands = [i for i, label in enumerate(labels) if "pppc" in label.lower()]
                if len(cands) == 1:
                    idx = cands[0]
            if idx is None:
                continue
            f_p50[k] = float(np.atleast_1d(npz["f_p50"])[idx])
            lnL[k] = float(npz["loglike_ml"])
            n_ok += 1

    if n_ok == 0:
        return None
    return {"f_p50": f_p50, "loglike_ml": lnL, "n_loaded": n_ok}


def _pppc_reference_pole_flux(E_GeV: np.ndarray, mass_GeV: float, channel: str) -> np.ndarray:
    """
    PPPC pole flux for SIGMA_V_REF, in Totani Fig. 8 units.

    The MCMC coefficient multiplies a physical PPPC counts template, not the
    phenomenological NFW iso_target_e2 shortcut. This mirrors the convention in
    spectral_reshaping.smooth_nfw_sigma_v_from_norm.
    """
    j_pole_msun2_kpc5 = 8.93e14
    msun_kpc3_to_gev_cm3 = 3.817e-8
    kpc_to_cm = 3.0857e21
    j_pole_gev2_cm5 = j_pole_msun2_kpc5 * msun_kpc3_to_gev_cm3**2 * kpc_to_cm
    norm_ref = 1000.0 * SIGMA_V_REF * j_pole_gev2_cm5 / (8.0 * np.pi * mass_GeV**2)
    dnde = _pppc_dnde_from_table(E_GeV, mass_GeV, channel)
    return norm_ref * E_GeV**2 * dnde


def _default_pppc_table_path() -> Path:
    if os.environ.get("PPPC4DMID_GAMMAS"):
        return Path(os.environ["PPPC4DMID_GAMMAS"])
    return _REPO / "Totani_reanalysis" / "data" / "AtProduction_gammas.dat"


def _read_pppc_table(path: Path) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    cached = _PPPC_TABLE_CACHE.get(path)
    if cached is not None:
        return cached
    with path.open("r") as handle:
        columns = handle.readline().split()
    values = np.loadtxt(path, skiprows=1)
    _PPPC_TABLE_CACHE[path] = (columns, values)
    return columns, values


def _pppc_dnde_from_table(E_GeV: np.ndarray, mass_GeV: float, channel: str) -> np.ndarray:
    """Return PPPC dN/dE [GeV^-1] without requiring astropy."""
    table_path = _default_pppc_table_path()
    if not table_path.exists():
        raise FileNotFoundError(
            f"PPPC gamma table not found: {table_path}. "
            "Set PPPC4DMID_GAMMAS or place AtProduction_gammas.dat under "
            "Totani_reanalysis/data/."
        )

    columns, values = _read_pppc_table(table_path)
    channel_col = _PPPC_CHANNELS.get(channel, channel)
    if channel_col not in columns:
        raise ValueError(f"PPPC channel {channel!r} maps to missing table column {channel_col!r}")

    masses = values[:, columns.index("mDM")]
    log10x = values[:, columns.index("Log[10,x]")]
    dndlogx = values[:, columns.index(channel_col)]

    E_GeV = np.asarray(E_GeV, float)
    logm_target = np.log10(float(mass_GeV))
    logx_target = np.log10(E_GeV / float(mass_GeV))
    unique_masses = np.unique(masses)
    unique_logm = np.log10(unique_masses)

    spectra = []
    for mass in unique_masses:
        mask = masses == mass
        order = np.argsort(log10x[mask])
        spectra.append(
            np.interp(
                logx_target,
                log10x[mask][order],
                dndlogx[mask][order],
                left=0.0,
                right=0.0,
            )
        )
    spectra = np.asarray(spectra, float)
    dndlogx_target = np.array([
        np.interp(logm_target, unique_logm, spectra[:, i])
        for i in range(len(E_GeV))
    ])
    return dndlogx_target / (E_GeV * np.log(10.0))


def _load_pppc_spectrum(
    mcmc_dir: Path,
    pppc_label: str,
    *,
    channel: str,
    mass_GeV: float,
    nE: int | None = None,
) -> dict | None:
    """
    Load a PPPC component spectrum in the same pole-flux convention used by
    Totani Fig. 8: E^2 dN/dE = f_pppc * PPPC reference pole flux.
    """
    files = sorted(mcmc_dir.glob("mcmc_results_k*.npz"))
    if not files:
        return None

    if nE is None:
        nE = max(int(path.stem.split("_k")[-1]) for path in files) + 1

    E_GeV = np.full(nE, np.nan)
    f_p16 = np.full(nE, np.nan)
    f_p50 = np.full(nE, np.nan)
    f_p84 = np.full(nE, np.nan)
    n_loaded = 0

    for path in files:
        try:
            k = int(path.stem.split("_k")[-1])
        except ValueError:
            continue
        if k >= nE:
            continue

        with np.load(path, allow_pickle=True) as npz:
            labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
            idx = labels.index(pppc_label) if pppc_label in labels else None
            if idx is None:
                cands = [i for i, label in enumerate(labels) if "pppc" in label.lower()]
                if len(cands) == 1:
                    idx = cands[0]
            if idx is None:
                continue

            if "Ectr_mev" in npz:
                E_GeV[k] = float(npz["Ectr_mev"]) / 1000.0
            f_p16[k] = float(np.atleast_1d(npz["f_p16"])[idx])
            f_p50[k] = float(np.atleast_1d(npz["f_p50"])[idx])
            f_p84[k] = float(np.atleast_1d(npz["f_p84"])[idx])
            n_loaded += 1

    if n_loaded == 0:
        return None

    source_ref = np.full(nE, np.nan)
    finite_E = np.isfinite(E_GeV) & (E_GeV > 0.0)
    source_ref[finite_E] = _pppc_reference_pole_flux(E_GeV[finite_E], mass_GeV, channel)

    phi = f_p50 * source_ref
    phi_p16 = f_p16 * source_ref
    phi_p84 = f_p84 * source_ref
    return {
        "E_GeV": E_GeV,
        "phi": phi,
        "phi_err_lo": np.maximum(phi - phi_p16, 0.0),
        "phi_err_hi": np.maximum(phi_p84 - phi, 0.0),
        "n_loaded": n_loaded,
    }


def _load_reference_halo_spectrum(reference: str, *, fit: str):
    from totani_data_loader import _MCMC_DIRS, load_halo_spectrum, load_halo_spectrum_rho2

    if reference == "auto":
        if fit == "global_fig6_w_disk":
            reference = "global_rho2_w_disk"
        elif fit == "pixelwise_global_fig6":
            reference = "pixelwise_global_rho2"
        elif str(fit).startswith("global_"):
            reference = "global_rho2"
        else:
            reference = "totani_rho2"

    if reference == "totani_rho2":
        return load_halo_spectrum_rho2(), "Totani halo (MCMC, NFW-$\\rho^2$)", reference
    if reference == "global_rho2":
        return (
            load_halo_spectrum(_MCMC_DIRS["global_rho2"]),
            "Global halo (NFW-$\\rho^2$, disk excluded)",
            reference,
        )
    if reference == "global_rho2_w_disk":
        return (
            load_halo_spectrum(_PROJECT / "mcmc" / "global_fit_results_fig6_w_disk"),
            "Global halo (NFW-$\\rho^2$, disk included)",
            reference,
        )
    if reference == "pixelwise_global_rho2":
        return (
            load_halo_spectrum(_MCMC_DIRS["pixelwise_global_rho2"]),
            "Pixelwise global halo (NFW-$\\rho^2$, disk excluded)",
            reference,
        )

    raise ValueError(f"Unknown halo reference {reference!r}. Choices: {HALO_REFERENCE_CHOICES}")


def _best_fit_normalization(model_shape: np.ndarray, data: np.ndarray, err: np.ndarray) -> float:
    """Analytic weighted least-squares normalization for data ~= A * model."""
    model_shape = np.asarray(model_shape, float)
    data = np.asarray(data, float)
    err = np.asarray(err, float)
    mask = (
        np.isfinite(model_shape)
        & np.isfinite(data)
        & np.isfinite(err)
        & (err > 0.0)
    )
    if not np.any(mask):
        return np.nan
    denom = np.sum((model_shape[mask] / err[mask]) ** 2)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    numer = np.sum(data[mask] * model_shape[mask] / err[mask] ** 2)
    return float(numer / denom)


def _fit_pppc_reference_spectrum(halo, mass_GeV: float, channel: str) -> dict:
    """
    Fit one PPPC normalization to the extracted halo spectrum.

    This mirrors Totani_reanalysis/core/spectral_reshaping.py: fit a single
    weighted least-squares amplitude and rank masses by chi2. The per-bin MCMC
    PPPC coefficients are not used here because they allow the spectrum to
    change independently in every energy bin.
    """
    E_fit, phi_fit, sigma_fit = halo.to_fit_arrays(positive_only=True, err_mode="sym")
    source_fit = _pppc_reference_pole_flux(E_fit, mass_GeV, channel)
    norm = _best_fit_normalization(source_fit, phi_fit, sigma_fit)
    if not np.isfinite(norm) or norm < 0.0:
        return {
            "chi2": np.nan,
            "norm": np.nan,
            "sigmav": np.nan,
            "ndof": 0,
            "model_phi": np.full_like(halo.E_bins_GeV, np.nan, dtype=float),
        }

    model_fit = norm * source_fit
    mask = (
        np.isfinite(model_fit)
        & np.isfinite(phi_fit)
        & np.isfinite(sigma_fit)
        & (sigma_fit > 0.0)
    )
    chi2 = float(np.sum(((model_fit[mask] - phi_fit[mask]) / sigma_fit[mask]) ** 2))
    model_full = norm * _pppc_reference_pole_flux(halo.E_bins_GeV, mass_GeV, channel)
    return {
        "chi2": chi2,
        "norm": norm,
        "sigmav": norm * SIGMA_V_REF,
        "ndof": max(int(np.count_nonzero(mask)) - 1, 0),
        "model_phi": model_full,
    }


def apply_reference_spectral_profile(
    channel_data: dict[str, dict],
    *,
    fit: str,
    halo_reference: str,
    dsph_source: str,
) -> dict[str, dict]:
    try:
        halo, _, resolved_halo_reference = _load_reference_halo_spectrum(
            halo_reference,
            fit=fit,
        )
    except Exception as e:
        print(f"  Could not load halo reference for spectral mass profile: {e}")
        return channel_data

    print(
        "  Spectral mass profile: fitting one PPPC normalization per mass "
        f"against {resolved_halo_reference} ({halo.mcmc_dir})"
    )
    out: dict[str, dict] = {}
    for ch, d in channel_data.items():
        masses = np.asarray(d["masses_GeV"], float)
        chi2 = np.full(masses.size, np.nan)
        norm = np.full(masses.size, np.nan)
        sigmav = np.full(masses.size, np.nan)
        ndof = np.full(masses.size, 0, dtype=int)
        model_phi = np.full((masses.size, len(halo.E_bins_GeV)), np.nan)

        for i, mass in enumerate(masses):
            if not np.isfinite(mass):
                continue
            fit_result = _fit_pppc_reference_spectrum(halo, float(mass), ch)
            chi2[i] = fit_result["chi2"]
            norm[i] = fit_result["norm"]
            sigmav[i] = fit_result["sigmav"]
            ndof[i] = fit_result["ndof"]
            model_phi[i, :] = fit_result["model_phi"]

        spectral_lnL = -0.5 * chi2
        spectral_delta_lnL = spectral_lnL.copy()
        if np.any(np.isfinite(spectral_delta_lnL)):
            spectral_delta_lnL -= np.nanmax(spectral_delta_lnL)
        sigmav_dsph = np.asarray([_dsph_limit(float(m), ch, source=dsph_source) for m in masses], float)
        tensions = np.divide(
            sigmav,
            sigmav_dsph,
            out=np.full_like(sigmav, np.nan),
            where=np.isfinite(sigmav) & np.isfinite(sigmav_dsph) & (sigmav_dsph > 0.0),
        )

        updated = dict(d)
        updated["mcmc_profile_total_lnL"] = np.asarray(d.get("total_lnL", []), float)
        updated["mcmc_profile_delta_lnL"] = np.asarray(d.get("delta_lnL", []), float)
        updated["mcmc_sigmav_best"] = np.asarray(d.get("sigmav_best", []), float)
        updated["total_lnL"] = spectral_lnL.astype(np.float32)
        updated["delta_lnL"] = spectral_delta_lnL.astype(np.float32)
        updated["sigmav_best"] = sigmav.astype(np.float32)
        updated["sigmav_dsph"] = sigmav_dsph.astype(np.float32)
        updated["tensions"] = tensions.astype(np.float32)
        updated["spectral_chi2"] = chi2.astype(np.float32)
        updated["spectral_norm"] = norm.astype(np.float32)
        updated["spectral_ndof"] = ndof.astype(np.int16)
        updated["spectral_model_phi"] = model_phi.astype(np.float32)
        updated["spectral_reference"] = np.array(resolved_halo_reference)
        out[ch] = updated

        if np.any(np.isfinite(chi2)):
            best_idx = int(np.nanargmin(chi2))
            print(
                f"    {ch}: best spectral mass = {masses[best_idx]:.0f} GeV, "
                f"chi2/ndof = {chi2[best_idx]:.2f}/{ndof[best_idx]}, "
                f"<sigma v> = {sigmav[best_idx]:.3e}"
            )

    return out


def load_channel_from_mcmc_dirs(
    channel: str,
    *,
    fit: str,
    dsph_source: str,
    max_abs_loglike: float | None,
    max_abs_dlnl: float | None,
) -> dict | None:
    """
    Reconstruct a partial mass-scan result from completed per-mass MCMC dirs.

    This lets plotting work while run_pppc_mass_scan.py is still in progress or
    when the compact master NPZ has not yet been written.
    """
    fit_prefix = FIT_MCMC_PREFIX[str(fit)]
    pattern = re.compile(rf"^mcmc_results_pppc_{re.escape(fit_prefix)}pppc_{re.escape(channel)}_m(?P<mass>\d+)")
    rows = []

    for mcmc_dir in sorted(_PROJECT.glob(f"mcmc_results_pppc_{fit_prefix}pppc_{channel}_m*")):
        match = pattern.match(mcmc_dir.name)
        if match is None:
            continue
        mass = float(int(match.group("mass")))
        pppc_label = f"pppc_{channel}_m{int(round(mass)):04d}"
        post = _load_pppc_posteriors(mcmc_dir, pppc_label)
        if post is None:
            print(f"    [skip] {mcmc_dir.name}: no PPPC posteriors found")
            continue

        f_med = post["f_p50"]
        finite_pos = np.isfinite(f_med) & (f_med > 0.0)
        f_mean = float(np.nanmean(f_med[finite_pos])) if np.any(finite_pos) else np.nan
        sigmav_best = f_mean * SIGMA_V_REF if np.isfinite(f_mean) else np.nan
        sigmav_dsph = _dsph_limit(mass, channel, source=dsph_source)
        tension = (
            sigmav_best / sigmav_dsph
            if np.isfinite(sigmav_best) and np.isfinite(sigmav_dsph) and sigmav_dsph > 0.0
            else np.nan
        )
        rows.append(
            dict(
                mass=mass,
                sigmav_best=sigmav_best,
                sigmav_dsph=sigmav_dsph,
                tension=tension,
                total_lnL=float(np.nansum(post["loglike_ml"])),
                n_loaded=int(post["n_loaded"]),
                mcmc_dir=str(mcmc_dir),
                pppc_label=pppc_label,
            )
        )

    if not rows:
        print(f"  [skip] No completed MCMC dirs found for {channel}")
        return None

    rows.sort(key=lambda r: r["mass"])
    loglike_by_bin = []
    for row in rows:
        post = _load_pppc_posteriors(Path(str(row["mcmc_dir"])), str(row["pppc_label"]))
        loglike_by_bin.append(
            np.asarray(post["loglike_ml"], float) if post is not None else np.full(13, np.nan)
        )
    loglike_by_bin = np.vstack(loglike_by_bin)
    total_lnL_raw = np.array([r["total_lnL"] for r in rows], dtype=float)
    total_lnL, delta_lnL, bins_used = profile_likelihood_from_bin_loglikes(
        loglike_by_bin,
        max_abs_loglike=max_abs_loglike,
        max_abs_dlnl=max_abs_dlnl,
        label=f"{fit}/{channel}",
    )
    print(f"  [partial] Loaded {len(rows)} completed {channel} mass point(s) from MCMC dirs")
    return {
        "masses_GeV": np.array([r["mass"] for r in rows], dtype=np.float32),
        "sigmav_best": np.array([r["sigmav_best"] for r in rows], dtype=np.float32),
        "sigmav_dsph": np.array([r["sigmav_dsph"] for r in rows], dtype=np.float32),
        "tensions": np.array([r["tension"] for r in rows], dtype=np.float32),
        "total_lnL": total_lnL.astype(np.float32),
        "total_lnL_raw": total_lnL_raw.astype(np.float32),
        "delta_lnL": delta_lnL.astype(np.float32),
        "loglike_ml_by_bin": loglike_by_bin.astype(np.float32),
        "profile_bins_used": bins_used.astype(bool),
        "profile_max_abs_loglike": np.float32(max_abs_loglike if max_abs_loglike is not None else 0.0),
        "profile_max_abs_dlnl": np.float32(max_abs_dlnl if max_abs_dlnl is not None else 0.0),
        "n_loaded": np.array([r["n_loaded"] for r in rows], dtype=np.int8),
        "channel": np.array(channel),
        "mcmc_dirs": np.array([r["mcmc_dir"] for r in rows], dtype=object),
        "pppc_labels": np.array([r["pppc_label"] for r in rows], dtype=object),
        "partial_from_mcmc_dirs": np.array(True),
    }


def plot_profile_likelihood(channel_data: dict[str, dict], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for ch, d in channel_data.items():
        st = CHANNEL_STYLE[ch]
        masses = np.asarray(d["masses_GeV"], float)
        dlnL   = np.asarray(d["delta_lnL"], float)
        fin    = np.isfinite(dlnL)
        ax.plot(masses[fin], dlnL[fin],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=6, lw=1.8, label=st["label"])

    ax.axhline(-1.92, color="k", lw=1.0, ls="--",
               label=r"$\Delta\ln\mathcal{L} = -1.92$ (95% CL)")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axvspan(500, 800, alpha=0.08, color="gray",
               label="Totani preferred range")
    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=12)
    ax.set_ylabel(r"$\Delta\ln\mathcal{L}(m_\chi)$", fontsize=12)
    ax.set_title("Spectral profile likelihood: one PPPC normalization per mass",
                 fontsize=10)
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(150, 2200)

    fig.tight_layout()
    out = outdir / "profile_likelihood.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sigmav_vs_mass(channel_data: dict[str, dict], outdir: Path, *, dsph_source: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # dSph upper limits
    for ch, d in channel_data.items():
        st = CHANNEL_STYLE[ch]
        try:
            dsph_mass, dsph_limit = load_dsph_limit_table(ch, dsph_source)
        except Exception as exc:
            print(f"  Could not load {dsph_source} dSph limit table for {ch}: {exc}")
            continue
        ax.plot(dsph_mass, dsph_limit,
                color=st["color"], ls=":", lw=1.2, alpha=0.7)

    # Best-fit <sigma v>
    for ch, d in channel_data.items():
        st = CHANNEL_STYLE[ch]
        masses = np.asarray(d["masses_GeV"], float)
        sv     = np.asarray(d["sigmav_best"], float)
        fin    = np.isfinite(sv) & (sv > 0)
        if "sigmav_p16" in d and "sigmav_p84" in d:
            sv16 = np.asarray(d["sigmav_p16"], float)
            sv84 = np.asarray(d["sigmav_p84"], float)
            band = fin & np.isfinite(sv16) & np.isfinite(sv84) & (sv16 > 0) & (sv84 > 0)
            if np.any(band):
                ax.fill_between(
                    masses[band],
                    sv16[band],
                    sv84[band],
                    color=st["color"],
                    alpha=0.16,
                    linewidth=0,
                )
        ax.plot(masses[fin], sv[fin],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=6, lw=1.8, label=st["label"])

    # Reference lines
    ax.axhline(THERMAL_RELIC, color="gray", lw=0.9, ls="--", alpha=0.7,
               label=r"Thermal relic ($3\times10^{-26}$ cm$^3$/s)")
    ax.axhspan(5e-25, 8e-25, alpha=0.12, color="C3",
               label=r"Totani: $5$–$8\times10^{-25}$ cm$^3$/s")
    ax.axvspan(500, 800, alpha=0.07, color="C3")

    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=12)
    ax.set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]", fontsize=12)
    ax.set_title(
        r"Required $\langle\sigma v\rangle$ vs configured dSph limits"
        "\n(dashed lines = dSph limits per channel)",
        fontsize=10,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150, 2200)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.22, which="both")

    fig.tight_layout()
    out = outdir / "sigmav_vs_mass.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_tension_vs_mass(channel_data: dict[str, dict], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for ch, d in channel_data.items():
        st = CHANNEL_STYLE[ch]
        masses  = np.asarray(d["masses_GeV"], float)
        tension = np.asarray(d["tensions"], float)
        fin     = np.isfinite(tension)
        ax.plot(masses[fin], tension[fin],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=6, lw=1.8, label=st["label"])

    ax.axhline(1.0, color="green", lw=1.5, ls="-",
               label="No tension with dSph (= 1)")
    ax.axhline(2.0, color="orange", lw=1.0, ls="--", alpha=0.7)
    ax.axhspan(3, 5, alpha=0.10, color="red",
               label=r"Totani tension range (3–5$\times$)")
    ax.axhspan(0, 1, alpha=0.08, color="green",
               label="Resolved region")
    ax.axvspan(500, 800, alpha=0.07, color="gray")

    ax.set_xlabel(r"$m_\chi$ [GeV]", fontsize=12)
    ax.set_ylabel(
        r"Tension: $\langle\sigma v\rangle_\mathrm{best}\ /\ \langle\sigma v\rangle_\mathrm{dSph}$",
        fontsize=11,
    )
    ax.set_title("dSph tension as a function of annihilation mass", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150, 2200)
    ax.set_ylim(0.05, 20)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.22, which="both")

    fig.tight_layout()
    out = outdir / "tension_vs_mass.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_best_fit_spectrum(
    channel_data: dict[str, dict],
    results_base: Path,
    outdir: Path,
    *,
    style: str = "standard",
    spectrum_mass: float | None = None,
    fit: str = "fig6",
    halo_reference: str = "auto",
) -> None:
    """
    For each channel's best-fit mass, overlay the single-normalization PPPC
    spectrum used for the spectral chi2 profile. If that profile is not
    available, fall back to the per-bin MCMC posterior reconstruction.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        return
    try:
        halo, halo_label, resolved_halo_reference = _load_reference_halo_spectrum(
            halo_reference,
            fit=fit,
        )
    except Exception as e:
        print(f"  Could not load halo reference data for spectrum plot: {e}")
        return
    print(f"  Spectrum reference halo: {resolved_halo_reference} ({halo.mcmc_dir})")

    totani_style = style == "totani"
    fig, ax = plt.subplots(figsize=(5.0, 3.4) if totani_style else (9, 6))
    scale = 1.0 if totani_style else 1e5
    selected_masses: list[float] = []

    ax.errorbar(
        halo.E_bins_GeV, halo.phi * scale,
        yerr=[halo.phi_err_lo * scale, halo.phi_err_hi * scale],
        fmt="o",
        color="k",
        ecolor="k",
        mfc="white" if totani_style else "k",
        mec="k",
        mew=0.7 if totani_style else 1.0,
        ms=5.5 if totani_style else 5,
        lw=0.8 if totani_style else 1.2,
        capsize=2 if totani_style else 3,
        label=halo_label,
        zorder=5,
    )

    for ch, d in channel_data.items():
        st = CHANNEL_STYLE[ch]
        masses  = np.asarray(d["masses_GeV"], float)
        total_lnL = np.asarray(d["total_lnL"], float)
        mcmc_dirs = np.asarray(d["mcmc_dirs"], object) if "mcmc_dirs" in d else None
        pppc_labels = np.asarray(d["pppc_labels"], object) if "pppc_labels" in d else None
        spectral_model = np.asarray(d["spectral_model_phi"], float) if "spectral_model_phi" in d else None

        if not np.any(np.isfinite(total_lnL)):
            continue

        if spectrum_mass is None:
            best_idx  = int(np.nanargmax(total_lnL))
            best_mass = float(masses[best_idx])
            selected_idx = best_idx
            mass_label = fr"$m_\chi={best_mass:.0f}$ GeV"
        else:
            finite_mass = np.isfinite(masses)
            if not np.any(finite_mass):
                continue
            finite_indices = np.flatnonzero(finite_mass)
            nearest_idx = int(np.nanargmin(np.abs(masses[finite_indices] - spectrum_mass)))
            selected_idx = int(finite_indices[nearest_idx])
            best_mass = float(masses[selected_idx])
            mass_label = fr"$m_\chi={best_mass:.0f}$ GeV"
            if abs(best_mass - spectrum_mass) > 1e-6:
                print(f"  {ch}: requested {spectrum_mass:g} GeV; using nearest completed mass {best_mass:g} GeV")
        selected_masses.append(best_mass)

        if spectral_model is not None and selected_idx < spectral_model.shape[0]:
            E_model = np.asarray(halo.E_bins_GeV, float)
            phi_model = spectral_model[selected_idx, :]
            spec = None
        else:
            mass_tag  = f"{ch}_m{int(round(best_mass)):04d}"
            if pppc_labels is not None and selected_idx < pppc_labels.size:
                pppc_label = str(pppc_labels[selected_idx])
            else:
                pppc_label = f"pppc_{mass_tag}"
            if mcmc_dirs is not None and selected_idx < mcmc_dirs.size:
                mcmc_dir = Path(str(mcmc_dirs[selected_idx]))
            else:
                mcmc_dir = _REPO / "Totani_paper_check" / f"mcmc_results_pppc_{pppc_label}"

            if not mcmc_dir.exists():
                print(f"  MCMC dir not found for best-fit spectrum: {mcmc_dir.name}")
                continue

            spec = _load_pppc_spectrum(
                mcmc_dir,
                pppc_label,
                channel=ch,
                mass_GeV=best_mass,
                nE=len(halo.E_bins_GeV),
            )
            if spec is None:
                print(f"  PPPC spectrum not found in: {mcmc_dir.name}")
                continue

            E_model = spec["E_GeV"]
            phi_model = spec["phi"]

        fin = np.isfinite(E_model) & np.isfinite(phi_model) & (E_model > 0)
        sigmav = np.asarray(d.get("sigmav_best", []), float)
        if selected_idx < sigmav.size and np.isfinite(sigmav[selected_idx]):
            label = fr"{st['label']}: {mass_label}, $\langle\sigma v\rangle={sigmav[selected_idx]:.1e}$"
        else:
            label = fr"{st['label']}: {mass_label}"
        if totani_style and spec is not None:
            ax.errorbar(
                E_model[fin],
                phi_model[fin] * scale,
                yerr=[
                    spec["phi_err_lo"][fin] * scale,
                    spec["phi_err_hi"][fin] * scale,
                ],
                color=st["color"],
                ecolor=st["color"],
                linestyle="none",
                marker=st["marker"],
                mfc="white",
                mec=st["color"],
                mew=0.8,
                markersize=5.2,
                linewidth=1.4,
                elinewidth=0.7,
                capsize=2,
                label=label,
            )
        else:
            ax.plot(
                E_model[fin], phi_model[fin] * scale,
                color=st["color"], ls=st["ls"], lw=1.8,
                label=label,
            )

    ax.axhline(0, color="k" if totani_style else "0.6", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1.0 if totani_style else 1.2, 1000.0 if totani_style else None)
    ax.set_xlabel("photon energy [GeV]" if totani_style else "Photon energy [GeV]", fontsize=11)
    if totani_style:
        ax.set_ylabel(
            r"flux $E^2 dN/dE$ at Galactic poles"
            "\n"
            r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
            fontsize=10,
        )
        ax.set_ylim(-2.0e-5, 4.0e-5)
        ax.set_yticks(np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1e-5)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.text(0.62, 0.84, r"NFW-$\rho^2$", transform=ax.transAxes, fontsize=10)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    else:
        ax.set_ylabel(
            r"$E^2\,dN/dE\ [\times10^{-5}\ \mathrm{MeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$",
            fontsize=10,
        )
        ax.set_title(
            "Best-fit annihilation model vs reference halo spectrum\n"
            "(single-normalization PPPC spectrum at galactic pole)",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.22)

    fig.tight_layout()
    if spectrum_mass is None:
        stem = "halo_spectrum_bestfit"
    elif len(selected_masses) == 1:
        stem = f"halo_spectrum_m{int(round(selected_masses[0])):04d}"
    else:
        stem = f"halo_spectrum_requested_m{int(round(spectrum_mass)):04d}"
    if totani_style:
        stem += "_totani_style"

    out = outdir / f"{stem}.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if totani_style:
        png = out.with_suffix(".png")
        fig.savefig(png, dpi=220, bbox_inches="tight")
        print(f"  Saved: {png}")
    plt.close(fig)
    print(f"  Saved: {out}")


def _mass_point_source(
    d: dict,
    channel: str,
    idx: int,
    *,
    fit: str,
) -> tuple[float, str, Path]:
    masses = np.asarray(d["masses_GeV"], float)
    mass = float(masses[idx])
    mass_tag = f"{channel}_m{int(round(mass)):04d}"

    pppc_labels = np.asarray(d["pppc_labels"], object) if "pppc_labels" in d else None
    if pppc_labels is not None and idx < pppc_labels.size:
        pppc_label = str(pppc_labels[idx])
    else:
        pppc_label = f"pppc_{mass_tag}"

    mcmc_dirs = np.asarray(d["mcmc_dirs"], object) if "mcmc_dirs" in d else None
    if mcmc_dirs is not None and idx < mcmc_dirs.size:
        mcmc_dir = Path(str(mcmc_dirs[idx]))
    else:
        fit_prefix = FIT_MCMC_PREFIX[str(fit)]
        mcmc_dir = _PROJECT / f"mcmc_results_pppc_{fit_prefix}{pppc_label}"

    return mass, pppc_label, mcmc_dir


def plot_all_mass_spectra(
    channel_data: dict[str, dict],
    outdir: Path,
    *,
    style: str = "standard",
    fit: str = "fig6",
    halo_reference: str = "auto",
) -> None:
    """
    For each channel, overlay every completed PPPC mass shape after fitting one
    overall normalization against the reference halo spectrum.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.ticker as mticker
    except ImportError:
        return
    try:
        halo, halo_label, resolved_halo_reference = _load_reference_halo_spectrum(
            halo_reference,
            fit=fit,
        )
    except Exception as e:
        print(f"  Could not load halo reference data for all-mass spectrum plot: {e}")
        return

    print(f"  All-mass spectrum reference halo: {resolved_halo_reference} ({halo.mcmc_dir})")
    totani_style = style == "totani"
    scale = 1.0 if totani_style else 1e5

    for ch, d in channel_data.items():
        masses = np.asarray(d["masses_GeV"], float)
        finite_indices = np.flatnonzero(np.isfinite(masses))
        if finite_indices.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(5.4, 3.8) if totani_style else (9, 6))

        ax.errorbar(
            halo.E_bins_GeV,
            halo.phi * scale,
            yerr=[halo.phi_err_lo * scale, halo.phi_err_hi * scale],
            fmt="o",
            color="k",
            ecolor="k",
            mfc="white" if totani_style else "k",
            mec="k",
            mew=0.7 if totani_style else 1.0,
            ms=5.2 if totani_style else 5,
            lw=0.8 if totani_style else 1.2,
            capsize=2 if totani_style else 3,
            label=halo_label,
            zorder=8,
        )

        norm = mpl.colors.Normalize(
            vmin=float(np.nanmin(masses[finite_indices])),
            vmax=float(np.nanmax(masses[finite_indices])),
        )
        cmap = plt.get_cmap("viridis")
        plotted = 0
        for idx in finite_indices:
            mass = float(masses[idx])
            if "spectral_model_phi" in d:
                E_model = np.asarray(halo.E_bins_GeV, float)
                phi_model = np.asarray(d["spectral_model_phi"], float)[idx, :]
            else:
                mass, pppc_label, mcmc_dir = _mass_point_source(d, ch, int(idx), fit=fit)
                if not mcmc_dir.exists():
                    print(f"  {ch} {mass:g} GeV: MCMC dir not found: {mcmc_dir.name}")
                    continue

                spec = _load_pppc_spectrum(
                    mcmc_dir,
                    pppc_label,
                    channel=ch,
                    mass_GeV=mass,
                    nE=len(halo.E_bins_GeV),
                )
                if spec is None:
                    print(f"  {ch} {mass:g} GeV: PPPC spectrum not found in {mcmc_dir.name}")
                    continue

                E_model = np.asarray(spec["E_GeV"], float)
                phi_model = np.asarray(spec["phi"], float)
            fin = np.isfinite(E_model) & np.isfinite(phi_model) & (E_model > 0)
            if not np.any(fin):
                continue

            color = cmap(norm(mass))
            label = f"{mass:.0f} GeV"
            if totani_style:
                ax.plot(
                    E_model[fin],
                    phi_model[fin] * scale,
                    color=color,
                    lw=1.15,
                    marker="o",
                    ms=2.7,
                    alpha=0.85,
                    label=label,
                )
            else:
                ax.plot(
                    E_model[fin],
                    phi_model[fin] * scale,
                    color=color,
                    lw=1.4,
                    alpha=0.88,
                    label=label,
                )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$m_\chi$ [GeV]", fontsize=9 if totani_style else 10)

        ax.axhline(0, color="k" if totani_style else "0.6", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1.0 if totani_style else 1.2, 1000.0 if totani_style else None)
        ax.set_xlabel("photon energy [GeV]" if totani_style else "Photon energy [GeV]", fontsize=11)
        if totani_style:
            ax.set_ylabel(
                r"flux $E^2 dN/dE$ at Galactic poles"
                "\n"
                r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
                fontsize=10,
            )
            ax.set_ylim(-2.0e-5, 4.0e-5)
            ax.set_yticks(np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 1e-5)
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.text(0.62, 0.84, CHANNEL_STYLE[ch]["label"], transform=ax.transAxes, fontsize=10)
            ax.legend(fontsize=6.5, loc="upper left", frameon=False, ncols=2)
        else:
            ax.set_ylabel(
                r"$E^2\,dN/dE\ [\times10^{-5}\ \mathrm{MeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$",
                fontsize=10,
            )
            ax.set_title(
                f"PPPC mass grid vs reference halo spectrum: {ch}",
                fontsize=10,
            )
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7, loc="upper right", ncols=2)

        fig.tight_layout()
        stem = f"halo_spectra_all_masses_{ch}"
        if totani_style:
            stem += "_totani_style"
        out = outdir / f"{stem}.pdf"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        if totani_style:
            png = out.with_suffix(".png")
            fig.savefig(png, dpi=220, bbox_inches="tight")
            print(f"  Saved: {png}")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Produce publication plots from PPPC mass scan results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--channels", nargs="+", default=["WW", "bb", "tautau"],
                   choices=["WW", "bb", "tautau", "tau"],
                   help="Channels to include in plots.")
    p.add_argument("--fit", default="fig6", choices=sorted(FIT_RESULT_SUBDIR),
                   help="Fit preset whose PPPC scan results should be plotted.")
    p.add_argument("--results-dir", default=None,
                   help="Root results directory. Defaults to "
                        "Totani_paper_check/results/pppc_mass_scan/.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for plots. Defaults to figures/plots_pppc/.")
    p.add_argument("--style", default="standard", choices=["standard", "totani"],
                   help="Plot style. 'totani' makes the spectrum panel mimic "
                        "Totani Fig. 8 and writes PDF plus PNG.")
    p.add_argument("--spectrum-mass", type=float, default=None,
                   help="Mass [GeV] to use for the spectrum overlay. Defaults "
                        "to each channel's best-fit mass; if the exact mass is "
                        "not present, the nearest completed mass is used.")
    p.add_argument("--halo-reference", default="auto", choices=HALO_REFERENCE_CHOICES,
                   help="Halo spectrum to overlay in the best-fit spectrum panel. "
                        "'auto' uses Totani rho2 for fig6 and global rho2 for global fits.")
    p.add_argument("--dsph-source", default="hoof", choices=list(DSph_LIMIT_SOURCES),
                   help="dSph upper-limit table source for tension calculations. "
                        "s-wave: hoof / mcdaniel / boddy_swave; velocity-dependent: "
                        "boddy_pwave / boddy_dwave / boddy_somm. 'mcdaniel' falls back "
                        "to Hoof for missing channels such as WW.")
    p.add_argument("--mcdaniel", action="store_true", default=False,
                   help="Shortcut for --dsph-source mcdaniel.")
    p.add_argument("--no-all-mass-spectra", action="store_true",
                   help="Do not make the per-channel overlay of every mass spectrum.")
    p.add_argument("--profile-max-abs-loglike", type=float,
                   default=DEFAULT_PROFILE_MAX_ABS_LOGLIKE,
                   help="Mask per-bin ML log-likelihoods with abs(lnL) above this "
                        "threshold before plotting the mass profile. Use <=0 to disable.")
    p.add_argument("--profile-max-abs-dlnl", type=float,
                   default=DEFAULT_PROFILE_MAX_ABS_DLOGLIKE,
                   help="Mask per-bin mass-profile Delta lnL values above this "
                        "threshold after subtracting each bin's best mass. Use <=0 to disable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mcdaniel:
        args.dsph_source = "mcdaniel"
    args.channels = [_normalise_cli_channel(ch) for ch in args.channels]
    _validate_dsph_source_channels(list(args.channels), str(args.dsph_source))

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        result_subdir = FIT_RESULT_SUBDIR[str(args.fit)]
        results_dir = RESULTS_BASE if result_subdir is None else RESULTS_BASE / str(result_subdir)

    if args.output_dir:
        outdir = Path(args.output_dir)
    elif args.fit == "fig6":
        outdir = PLOT_DIR
    else:
        outdir = _HERE / f"plots_pppc_{args.fit}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    max_abs_loglike = None if float(args.profile_max_abs_loglike) <= 0 else float(args.profile_max_abs_loglike)
    max_abs_dlnl = None if float(args.profile_max_abs_dlnl) <= 0 else float(args.profile_max_abs_dlnl)
    channel_data = {}
    for ch in args.channels:
        d = load_channel(
            ch,
            results_dir,
            fit=args.fit,
            dsph_source=args.dsph_source,
            max_abs_loglike=max_abs_loglike,
            max_abs_dlnl=max_abs_dlnl,
        )
        if d is not None:
            channel_data[ch] = d

    if not channel_data:
        print("No results found. Run run_pppc_mass_scan.py first.")
        return

    channel_data = apply_reference_spectral_profile(
        channel_data,
        fit=args.fit,
        halo_reference=args.halo_reference,
        dsph_source=args.dsph_source,
    )

    print(f"\nProducing plots -> {outdir}")
    plot_profile_likelihood(channel_data, outdir)
    plot_sigmav_vs_mass(channel_data, outdir, dsph_source=args.dsph_source)
    plot_tension_vs_mass(channel_data, outdir)
    plot_best_fit_spectrum(
        channel_data,
        results_dir,
        outdir,
        style=args.style,
        spectrum_mass=args.spectrum_mass,
        fit=args.fit,
        halo_reference=args.halo_reference,
    )
    if not args.no_all_mass_spectra:
        plot_all_mass_spectra(
            channel_data,
            outdir,
            style=args.style,
            fit=args.fit,
            halo_reference=args.halo_reference,
        )

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
