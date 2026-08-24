#!/usr/bin/env python3
"""
run_pppc_mass_scan.py
=====================
Run a PPPC annihilation mass scan.

The default is the cheap spectral scan: for each mass, build the PPPC pole
spectrum from the table, fit one normalization to the chosen halo spectrum,
and convert that coefficient to <sigma v>. This is the physically relevant
mass ranking and does not run any per-energy-bin MCMC/global fits.

The old spatial/template fit can still be requested with --run-spatial-fit.

After all masses:
    5. Compute the spectral profile Delta_lnL(m_chi) by fitting one PPPC
       normalization per mass to the selected halo spectrum.
    6. Compare <sigma v>_best(m_chi) against the configured dSph upper limits.
    7. Save a master results NPZ and produce summary plots.

Usage
-----
    # WW only, standard mass grid, spectral-only scan
    python run_pppc_mass_scan.py --channel WW

    # WW only, global Fig. 6 halo spectrum as the spectral reference
    python run_pppc_mass_scan.py --fit global_fig6 --channel WW

    # Re-run the old full per-mass spatial fits, then make the spectral summary
    python run_pppc_mass_scan.py --fit global_fig6 --channel WW --run-spatial-fit

    # All three channels, custom mass range
    python run_pppc_mass_scan.py --all-channels --mass 400 500 650 800 1000

    # Generated linear mass grid
    python run_pppc_mass_scan.py --channels bb WW --mass-min 200 --mass-max 2000 --n-masses 20

    # Single mass for testing
    python run_pppc_mass_scan.py --channel WW --mass 650 --dry-run

    # With scattering transfer applied to all templates
    python run_pppc_mass_scan.py --channel WW \\
        --scatter-operator dipole_magnetic --scatter-Lambda 1000

    # Skip template build if doing the explicit spatial fit and templates exist
    python run_pppc_mass_scan.py --channel WW --run-spatial-fit --skip-build

Output structure
----------------
Totani_paper_check/
  mcmc_results_pppc_{channel}_{mass_tag}/
    mcmc_results_k00.npz ... mcmc_results_k12.npz   (only with --run-spatial-fit)
    mcmc_results_k00.txt ... mcmc_results_k12.txt

Totani_paper_check/results/pppc_mass_scan/
  {channel}/
    mass_scan_results.npz     master results
    mass_scan_summary.txt     human-readable table
    lnL_vs_mass.pdf           profile likelihood plot
    sigmav_vs_mass.pdf        <sigma v>(m_chi) vs dSph limits
"""

from __future__ import annotations

import argparse
import io
import json
import os
import runpy
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = Path(os.environ.get(
    "REPO_PATH",
    str(_HERE.parent),
))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "mcmc"))
sys.path.insert(0, str(_HERE / "figures"))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))
sys.path.insert(0, str(_REPO / "Totani_reanalysis" / "core"))

DATA_DIR     = _REPO / "fermi_data" / "totani" / "processed"
TEMPLATE_DIR = DATA_DIR / "templates"
MCMC_DIR_BASE = _HERE / "mcmc"
RESULTS_DIR  = _HERE / "results" / "pppc_mass_scan"
DEFAULT_PROFILE_MAX_ABS_LOGLIKE = 1e10
DEFAULT_PROFILE_MAX_ABS_DLOGLIKE = 1e6

FIT_PRESETS = {
    "fig6": {
        "description": "standard Fig. 6 cellwise MCMC fit, disk excluded",
        "runner": "mcmc",
        "base_config": _HERE / "mcmc" / "configs" / "base.json",
        "include_disk": False,
        "dir_prefix": "",
        "result_subdir": None,
    },
    "global_fig6": {
        "description": "global pixel-level Fig. 6 fit, disk excluded",
        "runner": "global",
        "base_config": _HERE / "mcmc" / "configs" / "gce_global_base.json",
        "include_disk": False,
        "dir_prefix": "global_fig6_",
        "result_subdir": "global_fig6",
    },
    "global_fig6_w_disk": {
        "description": "global pixel-level Fig. 6 fit, disk included",
        "runner": "global",
        "base_config": _HERE / "mcmc" / "configs" / "gce_global_base.json",
        "include_disk": True,
        "dir_prefix": "global_fig6_w_disk_",
        "result_subdir": "global_fig6_w_disk",
    },
    "pixelwise_global_fig6": {
        "description": "pixelwise MCMC Fig. 6 fit, disk excluded",
        "runner": "mcmc",
        "base_config": _HERE / "mcmc" / "configs" / "base_pixelwise.json",
        "include_disk": False,
        "dir_prefix": "pixelwise_global_fig6_",
        "result_subdir": "pixelwise_global_fig6",
    },
}


def savez_compressed_max(path: Path, **arrays) -> None:
    """
    Write a maximally compressed NPZ.

    np.savez_compressed uses ZIP_DEFLATED, but does not expose the compression
    level. This helper keeps the same .npz format while requesting zlib level 9.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            tmp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as zf:
            for key, value in arrays.items():
                if not key:
                    raise ValueError("NPZ keys must be non-empty.")
                if "/" in key or "\\" in key:
                    raise ValueError(f"Invalid NPZ key {key!r}: path separators are not allowed.")
                buffer = io.BytesIO()
                np.save(buffer, np.asanyarray(value), allow_pickle=False)
                zf.writestr(f"{key}.npy", buffer.getvalue())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

from dsph_limits import (  # noqa: E402
    DSph_LIMIT_SOURCES,
    available_dsph_channels,
    dsph_limit_source,
    dsph_upper_limit as dsph_limit,
    has_dsph_limit_table,
)


# ---------------------------------------------------------------------------
# Reference cross section (must match build_pppc_template.py)
# ---------------------------------------------------------------------------
SIGMA_V_REF     = 1e-25   # cm^3 s^-1
J_POLE_MSUN2_KPC5 = 8.93e14

MSUN_TO_GEV = 1.115e57
KPC_TO_CM   = 3.086e21
J_POLE_GEV2_CM5 = J_POLE_MSUN2_KPC5 * (MSUN_TO_GEV ** 2) / (KPC_TO_CM ** 5)


# ---------------------------------------------------------------------------
# Convert f_pppc -> <sigma v>
# ---------------------------------------------------------------------------

def f_to_sigmav(f_pppc: float) -> float:
    """f_pppc * sigma_v_ref = <sigma v> [cm^3 s^-1]."""
    return float(f_pppc) * SIGMA_V_REF


def _shrink_mcmc_outputs(mcmc_outdir: Path) -> None:
    """Recompress per-bin MCMC NPZ files and drop bulky chain/logprob arrays."""
    try:
        from shrink_mcmc_outfiles import shrink_npz
    except Exception as exc:
        print(f"  [compact] WARNING: could not import shrinker: {exc}")
        return

    npz_files = sorted(mcmc_outdir.glob("mcmc_results_k*.npz"))
    if not npz_files:
        return

    old_total = 0.0
    new_total = 0.0
    for path in npz_files:
        try:
            old_mb, new_mb = shrink_npz(
                str(path),
                drop_chain=True,
                drop_logprob=True,
                backup_suffix=None,
            )
            old_total += old_mb
            new_total += new_mb
        except Exception as exc:
            print(f"  [compact] WARNING: could not shrink {path.name}: {exc}")
    if old_total > 0.0:
        saved = old_total - new_total
        print(f"  [compact] MCMC NPZ: {old_total:.2f} MB -> {new_total:.2f} MB "
              f"(saved {saved:.2f} MB)")


def _delete_matching(paths: list[Path], *, label: str) -> None:
    removed = 0
    freed = 0
    for path in paths:
        try:
            if not path.exists():
                continue
            freed += path.stat().st_size
            path.unlink()
            removed += 1
        except Exception as exc:
            print(f"  [compact] WARNING: could not remove {path}: {exc}")
    if removed:
        print(f"  [compact] Removed {removed} {label} file(s), freed {freed/1e6:.2f} MB")


# ---------------------------------------------------------------------------
# Extract f_pppc posteriors from a set of per-energy-bin npz files
# ---------------------------------------------------------------------------

def _load_pppc_posteriors(mcmc_dir: Path, pppc_label: str, nE: int = 13) -> dict:
    """
    Load f_pppc {ml, p16, p50, p84} and loglike_ml from all energy bins.

    Returns dict with keys:
        f_p50, f_p16, f_p84, f_ml    : (nE,) arrays
        loglike_ml                    : (nE,) array
        Ectr_mev                      : (nE,) array
        iso_target_e2                 : (nE,) array
        n_loaded                      : int
    """
    f_p50 = np.full(nE, np.nan)
    f_p16 = np.full(nE, np.nan)
    f_p84 = np.full(nE, np.nan)
    f_ml  = np.full(nE, np.nan)
    lnL   = np.full(nE, np.nan)
    Ectr  = np.full(nE, np.nan)
    iso_e2= np.full(nE, np.nan)
    n_ok  = 0

    for k in range(nE):
        path = mcmc_dir / f"mcmc_results_k{k:02d}.npz"
        if not path.exists():
            continue
        npz = np.load(path, allow_pickle=True)
        labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]

        # Match PPPC label: exact first, then any label containing 'pppc'
        idx = None
        if pppc_label in labels:
            idx = labels.index(pppc_label)
        else:
            cands = [i for i, l in enumerate(labels) if "pppc" in l.lower()]
            if len(cands) == 1:
                idx = cands[0]

        if idx is None:
            continue

        f_p50[k] = float(np.atleast_1d(npz["f_p50"])[idx])
        f_p16[k] = float(np.atleast_1d(npz["f_p16"])[idx])
        f_p84[k] = float(np.atleast_1d(npz["f_p84"])[idx])
        f_ml[k]  = float(np.atleast_1d(npz["f_ml"])[idx])
        lnL[k]   = float(npz["loglike_ml"])
        Ectr[k]  = float(npz["Ectr_mev"])
        if "iso_target_e2" in npz:
            iso_e2[k] = float(npz["iso_target_e2"])
        n_ok += 1

    return {
        "f_p50": f_p50, "f_p16": f_p16, "f_p84": f_p84, "f_ml": f_ml,
        "loglike_ml": lnL, "Ectr_mev": Ectr, "iso_target_e2": iso_e2,
        "n_loaded": n_ok,
    }


def _profile_likelihood_from_bin_loglikes(
    loglike_by_mass: np.ndarray,
    *,
    max_abs_loglike: float | None,
    max_abs_dlnl: float | None,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a robust mass profile from per-energy-bin log-likelihoods.

    Each energy bin is profiled over the scanned masses first. The final
    profile sums the per-bin deltas only over bins that remain finite for every
    mass, so a single global-fit failure bin cannot dominate the mass curve.
    """
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
    finite_profile = np.isfinite(delta_lnL)
    if np.any(finite_profile):
        delta_lnL = delta_lnL - np.nanmax(delta_lnL)

    print(f"  [profile] Using {int(np.count_nonzero(bins_used))}/{lnL.shape[1]} common energy bins{tag}")
    return profile_lnL, delta_lnL, bins_used


def _reference_spectral_profile(
    *,
    masses: np.ndarray,
    channel: str,
    fit: str,
    halo_reference: str,
    dsph_source: str = "hoof",
) -> dict | None:
    """
    Rank masses like the Totani spectral fits: one normalization per mass.

    The per-bin MCMC PPPC coefficients are useful diagnostics, but they do not
    impose a physical spectrum because each energy bin can choose an independent
    amplitude. This profile compares the physical PPPC shape at each mass to
    the extracted halo spectrum with a single best-fit normalization.
    """
    try:
        from plot_pppc_mass_scan import (
            _fit_pppc_reference_spectrum,
            _load_reference_halo_spectrum,
        )
    except Exception as exc:
        print(f"  [spectral] WARNING: could not import spectral profile helpers: {exc}")
        return None

    try:
        halo, _, resolved_halo_reference = _load_reference_halo_spectrum(
            halo_reference,
            fit=fit,
        )
    except Exception as exc:
        print(f"  [spectral] WARNING: could not load halo reference: {exc}")
        return None

    chi2 = np.full(masses.size, np.nan)
    norm = np.full(masses.size, np.nan)
    sigmav = np.full(masses.size, np.nan)
    ndof = np.full(masses.size, 0, dtype=int)
    model_phi = np.full((masses.size, len(halo.E_bins_GeV)), np.nan)

    for i, mass in enumerate(masses):
        if not np.isfinite(mass):
            continue
        fit_result = _fit_pppc_reference_spectrum(halo, float(mass), channel)
        chi2[i] = fit_result["chi2"]
        norm[i] = fit_result["norm"]
        sigmav[i] = fit_result["sigmav"]
        ndof[i] = fit_result["ndof"]
        model_phi[i, :] = fit_result["model_phi"]

    if not np.any(np.isfinite(chi2)):
        print("  [spectral] WARNING: no finite spectral chi2 values")
        return None

    total_lnL = -0.5 * chi2
    delta_lnL = total_lnL - np.nanmax(total_lnL)
    sigmav_dsph = np.asarray([dsph_limit(float(m), channel, source=dsph_source) for m in masses], float)
    tensions = np.divide(
        sigmav,
        sigmav_dsph,
        out=np.full_like(sigmav, np.nan),
        where=np.isfinite(sigmav) & np.isfinite(sigmav_dsph) & (sigmav_dsph > 0.0),
    )
    best_idx = int(np.nanargmin(chi2))
    print(
        f"  [spectral] Reference: {resolved_halo_reference} ({halo.mcmc_dir})\n"
        f"  [spectral] Best mass: {masses[best_idx]:.0f} GeV, "
        f"chi2/ndof={chi2[best_idx]:.2f}/{ndof[best_idx]}, "
        f"<sigma v>={sigmav[best_idx]:.3e}"
    )
    return {
        "total_lnL": total_lnL,
        "delta_lnL": delta_lnL,
        "chi2": chi2,
        "norm": norm,
        "sigmav_best": sigmav,
        "sigmav_dsph": sigmav_dsph,
        "tensions": tensions,
        "ndof": ndof,
        "model_phi": model_phi,
        "reference": resolved_halo_reference,
    }


def _split_normal_draws(
    rng: np.random.Generator,
    centre: np.ndarray,
    err_lo: np.ndarray,
    err_hi: np.ndarray,
    n_draws: int,
) -> np.ndarray:
    """Draw independent asymmetric marginal posterior samples per energy bin."""
    centre = np.asarray(centre, float)
    err_lo = np.asarray(err_lo, float)
    err_hi = np.asarray(err_hi, float)
    side = rng.random((int(n_draws), centre.size)) < 0.5
    width = np.where(side, err_lo[None, :], err_hi[None, :])
    width = np.where(np.isfinite(width) & (width > 0.0), width, np.nan)
    delta = np.abs(rng.standard_normal((int(n_draws), centre.size))) * width
    return centre[None, :] + np.where(side, -delta, delta)


def _reference_uncertainty_propagation(
    *,
    masses: np.ndarray,
    channel: str,
    fit: str,
    halo_reference: str,
    dsph_source: str,
    n_draws: int,
    seed: int,
) -> dict | None:
    """Propagate MCMC halo-spectrum marginal errors into PPPC fit intervals."""
    if int(n_draws) <= 0:
        return None
    try:
        from plot_pppc_mass_scan import (
            _load_reference_halo_spectrum,
            _pppc_reference_pole_flux,
        )
    except Exception as exc:
        print(f"  [uncertainty] WARNING: could not import PPPC helpers: {exc}")
        return None

    try:
        halo, _, resolved_halo_reference = _load_reference_halo_spectrum(
            halo_reference,
            fit=fit,
        )
    except Exception as exc:
        print(f"  [uncertainty] WARNING: could not load halo reference: {exc}")
        return None

    E = np.asarray(halo.E_bins_GeV, float)
    phi = np.asarray(halo.phi, float)
    err_lo = np.asarray(halo.phi_err_lo, float)
    err_hi = np.asarray(halo.phi_err_hi, float)
    sigma = 0.5 * (err_lo + err_hi)
    fit_mask = (
        np.isfinite(E)
        & np.isfinite(phi)
        & np.isfinite(sigma)
        & (sigma > 0.0)
        & (phi > 0.0)
    )
    if np.count_nonzero(fit_mask) < 2:
        print("  [uncertainty] WARNING: too few positive finite halo bins for propagation")
        return None

    E_fit = E[fit_mask]
    sigma_fit = sigma[fit_mask]
    rng = np.random.default_rng(int(seed))
    draws = _split_normal_draws(rng, phi[fit_mask], err_lo[fit_mask], err_hi[fit_mask], int(n_draws))

    nM = len(masses)
    nD = int(n_draws)
    sigmav_draw = np.full((nD, nM), np.nan, float)
    chi2_draw = np.full((nD, nM), np.nan, float)
    for im, mass in enumerate(np.asarray(masses, float)):
        source = _pppc_reference_pole_flux(E_fit, float(mass), channel)
        good = np.isfinite(source) & np.isfinite(sigma_fit) & (sigma_fit > 0.0)
        if not np.any(good):
            continue
        denom = np.sum((source[good] / sigma_fit[good]) ** 2)
        if not np.isfinite(denom) or denom <= 0:
            continue
        numer = np.sum(draws[:, good] * source[good][None, :] / sigma_fit[good][None, :] ** 2, axis=1)
        norm = numer / denom
        norm = np.where(norm > 0.0, norm, np.nan)
        model = norm[:, None] * source[good][None, :]
        chi2 = np.sum(((model - draws[:, good]) / sigma_fit[good][None, :]) ** 2, axis=1)
        sigmav_draw[:, im] = norm * SIGMA_V_REF
        chi2_draw[:, im] = chi2

    finite_any = np.any(np.isfinite(chi2_draw), axis=1)
    if not np.any(finite_any):
        print("  [uncertainty] WARNING: no finite propagated PPPC fits")
        return None

    best_idx = np.full(nD, -1, int)
    for i in np.where(finite_any)[0]:
        best_idx[i] = int(np.nanargmin(chi2_draw[i]))
    best_mass = np.where(best_idx >= 0, np.asarray(masses, float)[best_idx.clip(min=0)], np.nan)
    best_sigmav = np.where(best_idx >= 0, sigmav_draw[np.arange(nD), best_idx.clip(min=0)], np.nan)
    best_limit = np.asarray(
        [
            dsph_limit(float(m), channel, source=dsph_source) if np.isfinite(m) else np.nan
            for m in best_mass
        ],
        float,
    )
    best_tension = np.divide(
        best_sigmav,
        best_limit,
        out=np.full_like(best_sigmav, np.nan),
        where=np.isfinite(best_sigmav) & np.isfinite(best_limit) & (best_limit > 0),
    )

    q = [16, 50, 84]
    return {
        "reference": resolved_halo_reference,
        "n_draws": np.int32(nD),
        "seed": np.int32(seed),
        "fit_mask": fit_mask,
        "sigmav_p16": np.nanpercentile(sigmav_draw, q[0], axis=0),
        "sigmav_p50": np.nanpercentile(sigmav_draw, q[1], axis=0),
        "sigmav_p84": np.nanpercentile(sigmav_draw, q[2], axis=0),
        "chi2_p16": np.nanpercentile(chi2_draw, q[0], axis=0),
        "chi2_p50": np.nanpercentile(chi2_draw, q[1], axis=0),
        "chi2_p84": np.nanpercentile(chi2_draw, q[2], axis=0),
        "best_mass_p16": np.nanpercentile(best_mass, q[0]),
        "best_mass_p50": np.nanpercentile(best_mass, q[1]),
        "best_mass_p84": np.nanpercentile(best_mass, q[2]),
        "best_sigmav_p16": np.nanpercentile(best_sigmav, q[0]),
        "best_sigmav_p50": np.nanpercentile(best_sigmav, q[1]),
        "best_sigmav_p84": np.nanpercentile(best_sigmav, q[2]),
        "best_tension_p16": np.nanpercentile(best_tension, q[0]),
        "best_tension_p50": np.nanpercentile(best_tension, q[1]),
        "best_tension_p84": np.nanpercentile(best_tension, q[2]),
    }


# ---------------------------------------------------------------------------
# Write per-mass MCMC config
# ---------------------------------------------------------------------------

def _write_mcmc_config(
    *,
    pppc_label: str,
    mcmc_outdir: Path,
    config_path: Path,
    channel: str,
    mass: float,
    fit: str,
) -> None:
    """
    Write a JSON config that runs the Totani fig6 setup with the PPPC template
    replacing (or supplementing) the NFW shape template.

    The PPPC template is added as a free-sign component alongside the existing
    diffuse templates. The NFW shape template is removed — we are fitting a
    physical model, not a morphology proxy.

    Labels used (matching fig6_structured.json minus nfw, plus pppc):
        gas, iso, ps, loopA, loopB, ics, fb_pos, fb_neg, pppc_{...}
    """
    preset = FIT_PRESETS[str(fit)]
    labels = [
        "gas",
        "iso",
        "ps",
        "loopA",
        "loopB",
        "ics",
        pppc_label,
        "fb_pos",
        "fb_neg",
    ]
    cfg = {
        "base_config": str(preset["base_config"]),
        "name": f"pppc_{channel}_m{int(round(mass)):04d}",
        "include_disk": bool(preset["include_disk"]),
        "energy_bins": "all",
        "labels": labels,
        # pppc template is a halo-like component: allowed negative, starts at 0
        "negative_keys": ["pppc", "fb_neg"],
        "iso_target_e2": 1e-4,
        "verbosity": 1,
    }
    if preset["runner"] == "global":
        cfg.update({
            "global_dir": str(mcmc_outdir),
            "no_save_maps": True,
            # Global pixel-level fits need PSF-forward-folded morphology.
            # The base global config already covers gas/ics/loopA/loopB/nfw;
            # PPPC replaces the NFW halo template, so include it explicitly.
            "psf_labels": ["nfw", "gas", "ics", "loopA", "loopB", "pppc"],
        })
    else:
        cfg.update({
            "mcmc_dir": str(mcmc_outdir),
            "no_plots": True,
            "extra_argv": [
                # Defensive: if a base config ever enables chain/logprob saving,
                # keep saved arrays extremely sparse. With the current runner these
                # flags do nothing unless --save-chain/--save-logprob are present.
                "--thin-save", "1000000",
            ],
        })
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Run one mass point end-to-end
# ---------------------------------------------------------------------------

def run_one_mass(
    *,
    channel:         str,
    mass:            float,
    args:            argparse.Namespace,
    table_path:      Path,
    dry_run:         bool = False,
) -> dict | None:
    """
    Build template -> write config -> run 13-bin MCMC -> extract posteriors.
    Returns a summary dict for this mass point, or None if skipped/failed.
    """
    mass_tag    = f"{channel}_m{int(round(mass)):04d}"
    pppc_label  = f"pppc_{mass_tag}"
    if args.scatter_operator:
        c_str = f"{args.scatter_coupling:.2e}".replace("+","").replace(".","p")
        pppc_label += f"_scat_{args.scatter_operator}_{c_str}"

    preset = FIT_PRESETS[str(args.fit)]
    fit_prefix = str(preset["dir_prefix"])
    mcmc_outdir = _HERE / f"mcmc_results_pppc_{fit_prefix}{pppc_label}"
    config_path = _HERE / "mcmc" / "configs" / f"pppc_{fit_prefix}{pppc_label}.json"

    print(f"\n{'='*65}")
    print(f"  Mass point: {channel}  m_ann = {mass:.0f} GeV")
    print(f"  Fit:        {args.fit} ({preset['description']})")
    print(f"  PPPC label: {pppc_label}")
    print(f"  MCMC dir:   {mcmc_outdir.name}")
    print(f"{'='*65}")

    # --- Step 1: Build template ---
    mu_path = TEMPLATE_DIR / f"mu_{pppc_label}_counts.fits"
    if mu_path.exists() and not args.force_rebuild:
        print(f"  Template already exists: {mu_path.name}  (use --force-rebuild to overwrite)")
    elif not args.skip_build:
        if dry_run:
            print(f"  [DRY-RUN] Would build template: {mu_path.name}")
        else:
            from make_templates.build_pppc_template import build_pppc_template
            build_pppc_template(
                channel=channel,
                m_ann_GeV=mass,
                outdir=TEMPLATE_DIR,
                counts_path=DATA_DIR / "counts_ccube_1000to1000000.fits",
                expo_path=DATA_DIR / "expcube_1000to1000000.fits",
                table_path=table_path,
                n_s=args.n_s,
                binsz=0.125,
                scatter_operator=args.scatter_operator,
                scatter_coupling=args.scatter_coupling,
                scatter_dm_type=args.scatter_dm_type,
                scatter_majorana=args.scatter_majorana,
                scatter_c_s=args.scatter_c_s,
                scatter_c_p=args.scatter_c_p,
                scatter_c_phi=args.scatter_c_phi,
                scatter_n_theta=args.scatter_n_theta,
            )
    else:
        if not mu_path.exists():
            print(f"  WARNING: --skip-build set but template not found: {mu_path.name}. Skipping mass.")
            return None

    # --- Step 2: Write MCMC config ---
    _write_mcmc_config(
        pppc_label=pppc_label,
        mcmc_outdir=mcmc_outdir,
        config_path=config_path,
        channel=channel,
        mass=mass,
        fit=args.fit,
    )
    if dry_run:
        print(f"  [DRY-RUN] Would write config: {config_path.name}")
        print(f"  [DRY-RUN] Would run 13-bin {preset['runner']} fit -> {mcmc_outdir.name}/")
        return None

    # --- Step 3: Run fit ---
    t0 = time.time()
    mcmc_outdir.mkdir(parents=True, exist_ok=True)
    try:
        argv = ["--config", str(config_path)]
        if args.energy_bin is not None:
            argv += ["--energy-bin", str(args.energy_bin)]
        _run_fit_from_config(argv, runner=str(preset["runner"]))
    except Exception as e:
        print(f"  ERROR running {preset['runner']} fit for {pppc_label}: {e}")
        return None
    dt = time.time() - t0
    print(f"  Fit completed in {dt/60:.1f} min")

    # --- Step 4: Extract posteriors ---
    post = _load_pppc_posteriors(mcmc_outdir, pppc_label)
    if post["n_loaded"] == 0:
        print(f"  WARNING: no posteriors found for {pppc_label}")
        return None

    # Profile log-likelihood: sum over energy bins (marginalised over all other components)
    total_lnL = float(np.nansum(post["loglike_ml"]))

    # Best-fit <sigma v> from median f_pppc summed over positive bins
    f_med = post["f_p50"]
    finite_pos = np.isfinite(f_med) & (f_med > 0)
    # For a single spatial template, f_pppc is the same in all energy bins —
    # take the weighted mean across bins where the signal is detected (f_med > 0).
    f_mean = float(np.nanmean(f_med[finite_pos])) if np.any(finite_pos) else np.nan
    sigmav_best = f_to_sigmav(f_mean)
    sigmav_dsph = dsph_limit(mass, channel, source=str(args.dsph_source))
    tension     = sigmav_best / sigmav_dsph if (np.isfinite(sigmav_dsph) and np.isfinite(sigmav_best)) else np.nan

    result = {
        "channel":      channel,
        "fit":          args.fit,
        "mass_GeV":     mass,
        "pppc_label":   pppc_label,
        "mcmc_dir":     str(mcmc_outdir),
        "total_lnL":    total_lnL,
        "f_p50":        f_med,
        "f_p16":        post["f_p16"],
        "f_p84":        post["f_p84"],
        "f_ml":         post["f_ml"],
        "loglike_ml":   post["loglike_ml"],
        "Ectr_mev":     post["Ectr_mev"],
        "n_loaded":     post["n_loaded"],
        "sigmav_best":  sigmav_best,
        "sigmav_dsph":  sigmav_dsph,
        "tension":      tension,
        "wall_time_s":  dt,
    }

    print(f"  <sigma v>_best = {sigmav_best:.3e} cm^3/s")
    print(f"  <sigma v>_dSph = {sigmav_dsph:.3e} cm^3/s   tension = {tension:.2f}x")
    print(f"  total lnL      = {total_lnL:.2f}")

    # --- Step 5: Compact intermediate outputs immediately ---
    if args.compact_mcmc:
        _shrink_mcmc_outputs(mcmc_outdir)

    if not args.keep_mcmc_txt:
        _delete_matching(sorted(mcmc_outdir.glob("mcmc_results_k*.txt")), label="MCMC text summary")

    if not args.keep_templates:
        _delete_matching(
            [
                TEMPLATE_DIR / f"mu_{pppc_label}_counts.fits",
                TEMPLATE_DIR / f"mu_{pppc_label}_psf_counts.fits",
                TEMPLATE_DIR / f"{pppc_label}_E2dnde.fits",
            ],
            label="intermediate template",
        )

    return result


def _run_fit_from_config(argv: list[str], *, runner: str) -> None:
    if runner == "global":
        script = str(MCMC_DIR_BASE / "run_global_from_config.py")
    elif runner == "mcmc":
        script = str(MCMC_DIR_BASE / "run_mcmc_from_config.py")
    else:
        raise ValueError(f"Unknown fit runner: {runner}")
    old = sys.argv
    try:
        sys.argv = [script] + argv
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# Save master results and produce plots
# ---------------------------------------------------------------------------

def _save_and_plot(results: list[dict], channel: str, outdir: Path, args: argparse.Namespace) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    fit         = str(results[0].get("fit", "fig6")) if results else "fig6"
    masses      = np.array([r["mass_GeV"]    for r in results])
    sigmav_best = np.array([r["sigmav_best"] for r in results])
    sigmav_dsph = np.array([r["sigmav_dsph"] for r in results])
    tensions    = np.array([r["tension"]     for r in results])
    total_lnL_raw = np.array([r["total_lnL"] for r in results])
    loglike_by_bin = np.vstack([np.asarray(r["loglike_ml"], float) for r in results])
    n_loaded    = np.array([r["n_loaded"]     for r in results])

    max_abs_loglike = None if float(args.profile_max_abs_loglike) <= 0 else float(args.profile_max_abs_loglike)
    max_abs_dlnl = None if float(args.profile_max_abs_dlnl) <= 0 else float(args.profile_max_abs_dlnl)
    total_lnL, delta_lnL, profile_bins_used = _profile_likelihood_from_bin_loglikes(
        loglike_by_bin,
        max_abs_loglike=max_abs_loglike,
        max_abs_dlnl=max_abs_dlnl,
        label=f"{fit}/{channel}",
    )
    mcmc_profile_total_lnL = total_lnL.copy()
    mcmc_profile_delta_lnL = delta_lnL.copy()
    mcmc_sigmav_best = sigmav_best.copy()
    mcmc_tensions = tensions.copy()

    spectral = _reference_spectral_profile(
        masses=masses,
        channel=channel,
        fit=fit,
        halo_reference=str(args.halo_reference),
        dsph_source=str(args.dsph_source),
    )
    spectral_profile_used = spectral is not None
    spectral_chi2 = np.full_like(masses, np.nan, dtype=float)
    spectral_norm = np.full_like(masses, np.nan, dtype=float)
    spectral_ndof = np.zeros_like(masses, dtype=int)
    spectral_model_phi = np.empty((masses.size, 0), dtype=float)
    spectral_reference = "unavailable"
    if spectral_profile_used:
        total_lnL = np.asarray(spectral["total_lnL"], float)
        delta_lnL = np.asarray(spectral["delta_lnL"], float)
        sigmav_best = np.asarray(spectral["sigmav_best"], float)
        sigmav_dsph = np.asarray(spectral["sigmav_dsph"], float)
        tensions = np.asarray(spectral["tensions"], float)
        spectral_chi2 = np.asarray(spectral["chi2"], float)
        spectral_norm = np.asarray(spectral["norm"], float)
        spectral_ndof = np.asarray(spectral["ndof"], int)
        spectral_model_phi = np.asarray(spectral["model_phi"], float)
        spectral_reference = str(spectral["reference"])
    uncertainty = _reference_uncertainty_propagation(
        masses=masses,
        channel=channel,
        fit=fit,
        halo_reference=str(args.halo_reference),
        dsph_source=str(args.dsph_source),
        n_draws=int(args.uncertainty_draws) if args.propagate_uncertainty else 0,
        seed=int(args.uncertainty_seed),
    )
    if uncertainty is None:
        uncertainty = {
            "reference": "unavailable",
            "n_draws": np.int32(0),
            "seed": np.int32(args.uncertainty_seed),
            "fit_mask": np.zeros(0, dtype=bool),
            "sigmav_p16": np.full_like(masses, np.nan, dtype=float),
            "sigmav_p50": np.full_like(masses, np.nan, dtype=float),
            "sigmav_p84": np.full_like(masses, np.nan, dtype=float),
            "chi2_p16": np.full_like(masses, np.nan, dtype=float),
            "chi2_p50": np.full_like(masses, np.nan, dtype=float),
            "chi2_p84": np.full_like(masses, np.nan, dtype=float),
            "best_mass_p16": np.nan,
            "best_mass_p50": np.nan,
            "best_mass_p84": np.nan,
            "best_sigmav_p16": np.nan,
            "best_sigmav_p50": np.nan,
            "best_sigmav_p84": np.nan,
            "best_tension_p16": np.nan,
            "best_tension_p50": np.nan,
            "best_tension_p84": np.nan,
        }

    savez_compressed_max(
        outdir / "mass_scan_results.npz",
        masses_GeV    = masses.astype(np.float32),
        sigmav_best   = sigmav_best.astype(np.float32),
        sigmav_dsph   = sigmav_dsph.astype(np.float32),
        tensions      = tensions.astype(np.float32),
        total_lnL     = total_lnL.astype(np.float32),
        total_lnL_raw = total_lnL_raw.astype(np.float32),
        delta_lnL     = delta_lnL.astype(np.float32),
        loglike_ml_by_bin = loglike_by_bin.astype(np.float32),
        mcmc_profile_total_lnL = mcmc_profile_total_lnL.astype(np.float32),
        mcmc_profile_delta_lnL = mcmc_profile_delta_lnL.astype(np.float32),
        mcmc_sigmav_best = mcmc_sigmav_best.astype(np.float32),
        mcmc_tensions = mcmc_tensions.astype(np.float32),
        spectral_profile_used = np.array(spectral_profile_used),
        spectral_chi2 = spectral_chi2.astype(np.float32),
        spectral_norm = spectral_norm.astype(np.float32),
        spectral_ndof = spectral_ndof.astype(np.int16),
        spectral_model_phi = spectral_model_phi.astype(np.float32),
        spectral_reference = np.array(spectral_reference),
        profile_bins_used = profile_bins_used.astype(bool),
        profile_max_abs_loglike = np.float32(args.profile_max_abs_loglike),
        profile_max_abs_dlnl = np.float32(args.profile_max_abs_dlnl),
        n_loaded      = n_loaded.astype(np.int8),
        channel       = np.array(channel),
        fit           = np.array(fit),
        mcmc_dirs      = np.array([str(r["mcmc_dir"]) for r in results], dtype=str),
        pppc_labels    = np.array([str(r["pppc_label"]) for r in results], dtype=str),
        dsph_limit_source = np.array(dsph_limit_source(channel, source=str(args.dsph_source))),
        sigma_v_ref   = np.float32(SIGMA_V_REF),
        uncertainty_propagated = np.array(bool(int(uncertainty["n_draws"]) > 0)),
        uncertainty_reference = np.array(str(uncertainty["reference"])),
        uncertainty_n_draws = np.asarray(uncertainty["n_draws"], dtype=np.int32),
        uncertainty_seed = np.asarray(uncertainty["seed"], dtype=np.int32),
        uncertainty_fit_mask = np.asarray(uncertainty["fit_mask"], dtype=bool),
        sigmav_p16 = np.asarray(uncertainty["sigmav_p16"], dtype=np.float32),
        sigmav_p50 = np.asarray(uncertainty["sigmav_p50"], dtype=np.float32),
        sigmav_p84 = np.asarray(uncertainty["sigmav_p84"], dtype=np.float32),
        chi2_p16 = np.asarray(uncertainty["chi2_p16"], dtype=np.float32),
        chi2_p50 = np.asarray(uncertainty["chi2_p50"], dtype=np.float32),
        chi2_p84 = np.asarray(uncertainty["chi2_p84"], dtype=np.float32),
        best_mass_p16 = np.float32(uncertainty["best_mass_p16"]),
        best_mass_p50 = np.float32(uncertainty["best_mass_p50"]),
        best_mass_p84 = np.float32(uncertainty["best_mass_p84"]),
        best_sigmav_p16 = np.float32(uncertainty["best_sigmav_p16"]),
        best_sigmav_p50 = np.float32(uncertainty["best_sigmav_p50"]),
        best_sigmav_p84 = np.float32(uncertainty["best_sigmav_p84"]),
        best_tension_p16 = np.float32(uncertainty["best_tension_p16"]),
        best_tension_p50 = np.float32(uncertainty["best_tension_p50"]),
        best_tension_p84 = np.float32(uncertainty["best_tension_p84"]),
    )
    print(f"\nCompressed results: {outdir / 'mass_scan_results.npz'}")

    # Text summary table
    lines = [
        f"PPPC mass scan results: {channel}",
        f"Fit preset: {fit}",
        "=" * 70,
        f"dSph limits: {dsph_limit_source(channel, source=str(args.dsph_source))}",
        (
            "Primary mass ranking: spectral chi2 with one PPPC normalization per mass"
            if spectral_profile_used
            else "Primary mass ranking: MCMC per-bin likelihood profile (spectral profile unavailable)"
        ),
        f"Spectral halo reference: {spectral_reference}",
        (
            "Uncertainty propagation: "
            f"{int(uncertainty['n_draws'])} asymmetric-MCMC draws from {uncertainty['reference']}"
        ),
        (
            f"MCMC diagnostic profile: per-bin gated, using "
            f"{int(np.count_nonzero(profile_bins_used))}/{profile_bins_used.size} common energy bins"
        ),
        f"  max abs lnL mask: {args.profile_max_abs_loglike:g}  (<=0 disables)",
        f"  max abs per-bin ΔlnL mask: {args.profile_max_abs_dlnl:g}  (<=0 disables)",
        "",
        f"{'m_ann [GeV]':>12}  {'<sv>_best':>12}  {'<sv>_dSph':>12}  "
        f"{'tension':>9}  {'DeltalnL':>10}  {'chi2/ndof':>12}  {'n_bins':>7}",
        "-" * 70,
    ]
    for i, r in enumerate(results):
        chi2_ndof = (
            f"{spectral_chi2[i]:.2f}/{spectral_ndof[i]:d}"
            if spectral_profile_used and np.isfinite(spectral_chi2[i])
            else "n/a"
        )
        lines.append(
            f"{r['mass_GeV']:12.0f}  {sigmav_best[i]:12.3e}  "
            f"{sigmav_dsph[i]:12.3e}  {tensions[i]:9.3f}  "
            f"{delta_lnL[i]:10.2f}  {chi2_ndof:>12}  {r['n_loaded']:7d}"
        )
    lines += [
        "-" * 70,
        f"Best-fit mass: {masses[np.nanargmax(total_lnL)]:.0f} GeV",
        f"  <sigma v>_best = {sigmav_best[np.nanargmax(total_lnL)]:.3e} cm^3/s",
        f"  tension = {tensions[np.nanargmax(total_lnL)]:.2f}x dSph limit",
        (
            "Propagated best-fit intervals: "
            f"m={uncertainty['best_mass_p50']:.0f}"
            f" (+{uncertainty['best_mass_p84']-uncertainty['best_mass_p50']:.0f}"
            f"/-{uncertainty['best_mass_p50']-uncertainty['best_mass_p16']:.0f}) GeV, "
            f"<sigma v>={uncertainty['best_sigmav_p50']:.3e}"
            f" (+{uncertainty['best_sigmav_p84']-uncertainty['best_sigmav_p50']:.2e}"
            f"/-{uncertainty['best_sigmav_p50']-uncertainty['best_sigmav_p16']:.2e}) cm^3/s"
            if int(uncertainty["n_draws"]) > 0
            else "Propagated best-fit intervals: not requested"
        ),
        "",
        "Totani's pure-annihilation result (for comparison):",
        "  m_chi ~ 500-800 GeV,  <sigma v> ~ 5-8e-25 cm^3/s,  tension ~ 3-5x",
    ]
    summary_path = outdir / "mass_scan_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"\nSummary: {summary_path}")

    # Plots
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: profile likelihood
        ax1.plot(masses, delta_lnL, "C0o-", ms=6, lw=1.8)
        ax1.axhline(-1.92, color="k", lw=0.9, ls="--", label=r"$\Delta\ln\mathcal{L} = -1.92$ (95% CL)")
        ax1.axhline(0, color="gray", lw=0.6, ls=":")
        ax1.set_xlabel(r"$m_\chi$ [GeV]", fontsize=11)
        ax1.set_ylabel(r"$\Delta\ln\mathcal{L}(m_\chi)$", fontsize=11)
        ax1.set_title(f"Spectral profile log-likelihood: {channel} channel", fontsize=10)
        ax1.set_xscale("log")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)

        # Panel 2: <sigma v> vs dSph
        fin = np.isfinite(sigmav_best)
        ax2.fill_between(masses[fin], sigmav_dsph[fin], 1e-22,
                         alpha=0.12, color="red", label="dSph excluded")
        ax2.plot(masses[fin], sigmav_best[fin], "C0o-", ms=6, lw=1.8,
                 label=r"$\langle\sigma v\rangle_\mathrm{best}$ (this work)")
        ax2.plot(masses[fin], sigmav_dsph[fin], "r--", lw=1.5,
                 label="dSph 95% CL")

        # Totani's reported range
        ax2.axhspan(5e-25, 8e-25, alpha=0.15, color="C1",
                    label=r"Totani range ($5$–$8\times10^{-25}$ cm$^3$/s)")
        ax2.axvspan(500, 800, alpha=0.08, color="C1")

        ax2.set_xlabel(r"$m_\chi$ [GeV]", fontsize=11)
        ax2.set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]", fontsize=11)
        ax2.set_title(r"Required $\langle\sigma v\rangle$ vs dSph upper limit", fontsize=10)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.25, which="both")

        fig.tight_layout()
        plot_path = outdir / "mass_scan_overview.pdf"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plots.")

    return {
        "delta_lnL": delta_lnL,
        "sigmav_best": sigmav_best,
        "tensions": tensions,
    }


def _save_spectral_only_scan(
    *,
    channel: str,
    masses: np.ndarray,
    fit: str,
    outdir: Path,
    args: argparse.Namespace,
    table_path: Path,
) -> dict | None:
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ["PPPC4DMID_GAMMAS"] = str(table_path)

    spectral = _reference_spectral_profile(
        masses=masses,
        channel=channel,
        fit=fit,
        halo_reference=str(args.halo_reference),
        dsph_source=str(args.dsph_source),
    )
    if spectral is None:
        return None

    total_lnL = np.asarray(spectral["total_lnL"], float)
    delta_lnL = np.asarray(spectral["delta_lnL"], float)
    sigmav_best = np.asarray(spectral["sigmav_best"], float)
    sigmav_dsph = np.asarray(spectral["sigmav_dsph"], float)
    tensions = np.asarray(spectral["tensions"], float)
    spectral_chi2 = np.asarray(spectral["chi2"], float)
    spectral_norm = np.asarray(spectral["norm"], float)
    spectral_ndof = np.asarray(spectral["ndof"], int)
    spectral_model_phi = np.asarray(spectral["model_phi"], float)
    spectral_reference = str(spectral["reference"])
    n_loaded = np.full(masses.size, spectral_model_phi.shape[1], dtype=int)
    uncertainty = _reference_uncertainty_propagation(
        masses=masses,
        channel=channel,
        fit=fit,
        halo_reference=str(args.halo_reference),
        dsph_source=str(args.dsph_source),
        n_draws=int(args.uncertainty_draws) if args.propagate_uncertainty else 0,
        seed=int(args.uncertainty_seed),
    )
    if uncertainty is None:
        uncertainty = {
            "reference": "unavailable",
            "n_draws": np.int32(0),
            "seed": np.int32(args.uncertainty_seed),
            "fit_mask": np.zeros(0, dtype=bool),
            "sigmav_p16": np.full_like(masses, np.nan, dtype=float),
            "sigmav_p50": np.full_like(masses, np.nan, dtype=float),
            "sigmav_p84": np.full_like(masses, np.nan, dtype=float),
            "chi2_p16": np.full_like(masses, np.nan, dtype=float),
            "chi2_p50": np.full_like(masses, np.nan, dtype=float),
            "chi2_p84": np.full_like(masses, np.nan, dtype=float),
            "best_mass_p16": np.nan,
            "best_mass_p50": np.nan,
            "best_mass_p84": np.nan,
            "best_sigmav_p16": np.nan,
            "best_sigmav_p50": np.nan,
            "best_sigmav_p84": np.nan,
            "best_tension_p16": np.nan,
            "best_tension_p50": np.nan,
            "best_tension_p84": np.nan,
        }

    savez_compressed_max(
        outdir / "mass_scan_results.npz",
        masses_GeV=masses.astype(np.float32),
        sigmav_best=sigmav_best.astype(np.float32),
        sigmav_dsph=sigmav_dsph.astype(np.float32),
        tensions=tensions.astype(np.float32),
        total_lnL=total_lnL.astype(np.float32),
        total_lnL_raw=np.full(masses.size, np.nan, dtype=np.float32),
        delta_lnL=delta_lnL.astype(np.float32),
        loglike_ml_by_bin=np.empty((masses.size, 0), dtype=np.float32),
        mcmc_profile_total_lnL=np.full(masses.size, np.nan, dtype=np.float32),
        mcmc_profile_delta_lnL=np.full(masses.size, np.nan, dtype=np.float32),
        mcmc_sigmav_best=np.full(masses.size, np.nan, dtype=np.float32),
        mcmc_tensions=np.full(masses.size, np.nan, dtype=np.float32),
        spectral_profile_used=np.array(True),
        spectral_chi2=spectral_chi2.astype(np.float32),
        spectral_norm=spectral_norm.astype(np.float32),
        spectral_ndof=spectral_ndof.astype(np.int16),
        spectral_model_phi=spectral_model_phi.astype(np.float32),
        spectral_reference=np.array(spectral_reference),
        profile_bins_used=np.zeros(0, dtype=bool),
        profile_max_abs_loglike=np.float32(args.profile_max_abs_loglike),
        profile_max_abs_dlnl=np.float32(args.profile_max_abs_dlnl),
        n_loaded=n_loaded.astype(np.int8),
        channel=np.array(channel),
        fit=np.array(fit),
        mcmc_dirs=np.array([""] * masses.size, dtype=str),
        pppc_labels=np.array(
            [f"pppc_{channel}_m{int(round(float(m))):04d}" for m in masses],
            dtype=str,
        ),
        dsph_limit_source=np.array(dsph_limit_source(channel, source=str(args.dsph_source))),
        sigma_v_ref=np.float32(SIGMA_V_REF),
        scan_mode=np.array("spectral_only"),
        uncertainty_propagated=np.array(bool(int(uncertainty["n_draws"]) > 0)),
        uncertainty_reference=np.array(str(uncertainty["reference"])),
        uncertainty_n_draws=np.asarray(uncertainty["n_draws"], dtype=np.int32),
        uncertainty_seed=np.asarray(uncertainty["seed"], dtype=np.int32),
        uncertainty_fit_mask=np.asarray(uncertainty["fit_mask"], dtype=bool),
        sigmav_p16=np.asarray(uncertainty["sigmav_p16"], dtype=np.float32),
        sigmav_p50=np.asarray(uncertainty["sigmav_p50"], dtype=np.float32),
        sigmav_p84=np.asarray(uncertainty["sigmav_p84"], dtype=np.float32),
        chi2_p16=np.asarray(uncertainty["chi2_p16"], dtype=np.float32),
        chi2_p50=np.asarray(uncertainty["chi2_p50"], dtype=np.float32),
        chi2_p84=np.asarray(uncertainty["chi2_p84"], dtype=np.float32),
        best_mass_p16=np.float32(uncertainty["best_mass_p16"]),
        best_mass_p50=np.float32(uncertainty["best_mass_p50"]),
        best_mass_p84=np.float32(uncertainty["best_mass_p84"]),
        best_sigmav_p16=np.float32(uncertainty["best_sigmav_p16"]),
        best_sigmav_p50=np.float32(uncertainty["best_sigmav_p50"]),
        best_sigmav_p84=np.float32(uncertainty["best_sigmav_p84"]),
        best_tension_p16=np.float32(uncertainty["best_tension_p16"]),
        best_tension_p50=np.float32(uncertainty["best_tension_p50"]),
        best_tension_p84=np.float32(uncertainty["best_tension_p84"]),
    )
    print(f"\nCompressed spectral results: {outdir / 'mass_scan_results.npz'}")

    lines = [
        f"PPPC spectral mass scan results: {channel}",
        f"Fit preset / halo selection: {fit}",
        "=" * 78,
        "Scan mode: spectral_only (no templates, no per-bin MCMC/global fits)",
        f"dSph limits: {dsph_limit_source(channel, source=str(args.dsph_source))}",
        "Primary mass ranking: spectral chi2 with one PPPC normalization per mass",
        f"Spectral halo reference: {spectral_reference}",
        (
            "Uncertainty propagation: "
            f"{int(uncertainty['n_draws'])} asymmetric-MCMC draws from {uncertainty['reference']}"
        ),
        "",
        f"{'m_ann [GeV]':>12}  {'<sv>_best':>12}  {'<sv>_dSph':>12}  "
        f"{'tension':>9}  {'DeltalnL':>10}  {'chi2/ndof':>12}",
        "-" * 78,
    ]
    for i, mass in enumerate(masses):
        lines.append(
            f"{mass:12.0f}  {sigmav_best[i]:12.3e}  "
            f"{sigmav_dsph[i]:12.3e}  {tensions[i]:9.3f}  "
            f"{delta_lnL[i]:10.2f}  {spectral_chi2[i]:.2f}/{spectral_ndof[i]:d}"
        )
    best_idx = int(np.nanargmax(total_lnL))
    lines += [
        "-" * 78,
        f"Best-fit mass: {masses[best_idx]:.0f} GeV",
        f"  <sigma v>_best = {sigmav_best[best_idx]:.3e} cm^3/s",
        f"  tension = {tensions[best_idx]:.2f}x dSph limit",
        (
            "Propagated best-fit intervals: "
            f"m={uncertainty['best_mass_p50']:.0f}"
            f" (+{uncertainty['best_mass_p84']-uncertainty['best_mass_p50']:.0f}"
            f"/-{uncertainty['best_mass_p50']-uncertainty['best_mass_p16']:.0f}) GeV, "
            f"<sigma v>={uncertainty['best_sigmav_p50']:.3e}"
            f" (+{uncertainty['best_sigmav_p84']-uncertainty['best_sigmav_p50']:.2e}"
            f"/-{uncertainty['best_sigmav_p50']-uncertainty['best_sigmav_p16']:.2e}) cm^3/s"
            if int(uncertainty["n_draws"]) > 0
            else "Propagated best-fit intervals: not requested"
        ),
    ]
    summary_path = outdir / "mass_scan_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Summary: {summary_path}")

    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(masses, delta_lnL, "C0o-", ms=6, lw=1.8)
        ax1.axhline(-1.92, color="k", lw=0.9, ls="--", label=r"$\Delta\ln\mathcal{L} = -1.92$")
        ax1.axhline(0, color="gray", lw=0.6, ls=":")
        ax1.set_xlabel(r"$m_\chi$ [GeV]", fontsize=11)
        ax1.set_ylabel(r"$\Delta\ln\mathcal{L}(m_\chi)$", fontsize=11)
        ax1.set_title(f"Spectral profile log-likelihood: {channel}", fontsize=10)
        ax1.set_xscale("log")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)

        fin = np.isfinite(sigmav_best)
        ax2.fill_between(masses[fin], sigmav_dsph[fin], 1e-22, alpha=0.12, color="red", label="dSph excluded")
        ax2.plot(masses[fin], sigmav_best[fin], "C0o-", ms=6, lw=1.8, label=r"$\langle\sigma v\rangle_\mathrm{best}$")
        ax2.plot(masses[fin], sigmav_dsph[fin], "r--", lw=1.5, label="dSph 95% CL")
        ax2.axhspan(5e-25, 8e-25, alpha=0.15, color="C1", label=r"Totani range")
        ax2.axvspan(500, 800, alpha=0.08, color="C1")
        ax2.set_xlabel(r"$m_\chi$ [GeV]", fontsize=11)
        ax2.set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]", fontsize=11)
        ax2.set_title(r"Required $\langle\sigma v\rangle$ vs dSph upper limit", fontsize=10)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.25, which="both")

        fig.tight_layout()
        plot_path = outdir / "mass_scan_overview.pdf"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {plot_path}")
    except ImportError:
        print("matplotlib not available - skipping plots.")

    return {
        "delta_lnL": delta_lnL,
        "sigmav_best": sigmav_best,
        "tensions": tensions,
        "masses": masses,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MASS_SCAN_GEV = [200, 300, 400, 500, 600, 650, 700, 750, 800, 900, 1000, 1200, 1500, 2000]


def _resolve_mass_grid(args: argparse.Namespace) -> list[float]:
    if args.mass is not None:
        return [float(m) for m in args.mass]

    generated_parts = [
        args.mass_min is not None,
        args.mass_max is not None,
        args.n_masses is not None,
    ]
    if any(generated_parts):
        if not all(generated_parts):
            raise ValueError("--mass-min, --mass-max, and --n-masses must be supplied together.")
        if args.mass_min <= 0.0 or args.mass_max <= 0.0:
            raise ValueError("--mass-min and --mass-max must be positive.")
        if args.mass_max < args.mass_min:
            raise ValueError("--mass-max must be greater than or equal to --mass-min.")
        if args.n_masses < 1:
            raise ValueError("--n-masses must be at least 1.")
        if args.n_masses == 1:
            return [float(args.mass_min)]
        return np.linspace(float(args.mass_min), float(args.mass_max), int(args.n_masses)).tolist()

    return [float(m) for m in MASS_SCAN_GEV]


def _normalise_cli_channel(channel: str) -> str:
    return "tautau" if str(channel).strip().lower() in ("tau", "tautau") else str(channel)


def _validate_dsph_source_channels(channels: list[str], dsph_source: str) -> None:
    missing = [ch for ch in channels if not has_dsph_limit_table(ch, dsph_source)]
    if missing:
        available = ", ".join(available_dsph_channels(dsph_source))
        raise SystemExit(
            f"ERROR: dSph source {dsph_source!r} has no limit table for: {', '.join(missing)}. "
            f"Available channels for this source: {available}. "
            "Use --dsph-source hoof for WW, or run only the McDaniel channels."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full PPPC annihilation mass scan against Totani halo excess.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Channel / mass
    p.add_argument("--channel", default="WW",
                   choices=["WW", "bb", "tautau", "tau"],
                   help="Annihilation channel.")
    p.add_argument("--channels", nargs="+", default=None,
                   choices=["WW", "bb", "tautau", "tau"],
                   help="Explicit channel list. Overrides --channel/--all-channels.")
    p.add_argument("--all-channels", action="store_true", default=False,
                   help="Run WW, bb, and tautau sequentially.")
    p.add_argument("--mass", type=float, nargs="+", default=None,
                   help="Mass grid [GeV]. Defaults to standard 14-point grid.")
    p.add_argument("--mass-min", type=float, default=None,
                   help="Minimum mass [GeV] for a generated linear mass grid. "
                        "Ignored if --mass is supplied.")
    p.add_argument("--mass-max", type=float, default=None,
                   help="Maximum mass [GeV] for a generated linear mass grid. "
                        "Ignored if --mass is supplied.")
    p.add_argument("--n-masses", type=int, default=None,
                   help="Number of masses in the generated linear mass grid. "
                        "Ignored if --mass is supplied.")
    p.add_argument("--energy-bin", type=int, default=None,
                   help="Run only this single energy bin (for testing).")
    p.add_argument("--fit", default="fig6", choices=sorted(FIT_PRESETS),
                   help="Fit preset used to choose the default halo spectrum. "
                        "Only controls the expensive backend when --run-spatial-fit is set.")

    # Template building
    p.add_argument("--skip-build", action="store_true", default=False,
                   help="Skip template build (assumes files already exist).")
    p.add_argument("--force-rebuild", action="store_true", default=False,
                   help="Rebuild templates even if they already exist.")
    p.add_argument("--pppc-table", default=None,
                   help="Path to AtProduction_gammas.dat.")
    p.add_argument("--n-s", type=int, default=2048,
                   help="LOS integration steps for J_2 map in template build.")

    # Scattering transfer (optional — applied to templates before MCMC)
    p.add_argument("--scatter-operator", default=None,
                   choices=[
                       "dipole_magnetic", "dipole_electric", "charge_radius",
                       "anapole", "rayleigh_even", "rayleigh_odd", "rayleigh_full",
                       "higgs_portal",
                   ],
                   help="Apply scattering transfer with this operator before MCMC. "
                        "If not set, pure annihilation fit.")
    p.add_argument("--scatter-Lambda",  type=float, default=1e3,
                   help="EFT Lambda [GeV] for scattering transfer.")
    p.add_argument("--scatter-yeff",    type=float, default=1.0,
                   help="y_eff for higgs_portal scattering transfer.")
    p.add_argument("--scatter-dm-type", default="fermionic",
                   choices=["fermionic", "scalar"])
    p.add_argument("--scatter-majorana",action="store_true", default=False)
    p.add_argument("--scatter-c-s",     type=float, default=1.0)
    p.add_argument("--scatter-c-p",     type=float, default=0.0)
    p.add_argument("--scatter-c-phi",   type=float, default=1.0)
    p.add_argument("--scatter-n-theta", type=int,   default=600)

    # Run control
    p.add_argument("--run-spatial-fit", action="store_true", default=False,
                   help="Run the old expensive template + per-bin MCMC/global fit for "
                        "each mass. By default the scan is spectral-only.")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print what would be done without executing.")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory for results NPZ and plots.")
    p.add_argument("--compact-mcmc", action=argparse.BooleanOptionalAction, default=True,
                   help="After each mass point, recompress per-bin MCMC NPZ files "
                        "and drop chain/logprob arrays if present.")
    p.add_argument("--keep-mcmc-txt", action="store_true", default=False,
                   help="Keep per-energy-bin text summaries. By default they are "
                        "deleted after the compact NPZ summary is read.")
    p.add_argument("--keep-templates", action="store_true", default=False,
                   help="Keep intermediate PPPC FITS templates. By default the "
                        "mass-specific counts and E2dnde templates are deleted "
                        "after the MCMC outputs have been extracted.")
    p.add_argument("--profile-max-abs-loglike", type=float,
                   default=DEFAULT_PROFILE_MAX_ABS_LOGLIKE,
                   help="Mask per-bin ML log-likelihoods with abs(lnL) above this "
                        "threshold before building the mass profile. Use <=0 to disable.")
    p.add_argument("--profile-max-abs-dlnl", type=float,
                   default=DEFAULT_PROFILE_MAX_ABS_DLOGLIKE,
                   help="Mask per-bin mass-profile Delta lnL values above this "
                        "threshold after subtracting each bin's best mass. Use <=0 to disable.")
    p.add_argument("--halo-reference", default="auto",
                   choices=[
                       "auto",
                       "totani_rho2",
                       "global_rho2",
                       "global_rho2_w_disk",
                       "pixelwise_global_rho2",
                   ],
                   help="Halo spectrum used for the spectral mass ranking. "
                        "'auto' uses Totani rho2 for fig6 and global rho2 for global fits.")
    p.add_argument("--dsph-source", default="hoof", choices=list(DSph_LIMIT_SOURCES),
                   help="dSph upper-limit table source for tension calculations. "
                        "s-wave: hoof / mcdaniel / boddy_swave; velocity-dependent: "
                        "boddy_pwave / boddy_dwave / boddy_somm. 'mcdaniel' falls back "
                        "to Hoof for missing channels such as WW.")
    p.add_argument("--mcdaniel", action="store_true", default=False,
                   help="Shortcut for --dsph-source mcdaniel.")
    p.add_argument("--propagate-uncertainty", action="store_true", default=False,
                   help="Propagate the extracted halo-spectrum MCMC marginal "
                        "uncertainties into the PPPC mass/cross-section fit by "
                        "Monte Carlo sampling asymmetric per-bin errors.")
    p.add_argument("--uncertainty-draws", type=int, default=5000,
                   help="Number of halo-spectrum draws used with --propagate-uncertainty.")
    p.add_argument("--uncertainty-seed", type=int, default=12345,
                   help="Random seed for PPPC uncertainty propagation.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mcdaniel:
        args.dsph_source = "mcdaniel"

    # Resolve scatter coupling
    if args.scatter_operator == "higgs_portal":
        args.scatter_coupling = float(args.scatter_yeff)
    else:
        args.scatter_coupling = float(args.scatter_Lambda)

    # Resolve PPPC table
    if args.pppc_table is not None:
        table_path = Path(args.pppc_table)
    else:
        table_path = _REPO / "Totani_reanalysis" / "data" / "AtProduction_gammas.dat"
    if not table_path.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"PPPC table not found: {table_path}\n"
            "Pass --pppc-table or place AtProduction_gammas.dat at "
            "Totani_reanalysis/data/AtProduction_gammas.dat."
        )
    os.environ["PPPC4DMID_GAMMAS"] = str(table_path)

    if args.scatter_operator and not args.run_spatial_fit:
        raise ValueError(
            "The default spectral-only scan currently supports pure PPPC spectra. "
            "Use --run-spatial-fit for scattering-transfer templates."
        )

    if args.channels is not None:
        channels = [_normalise_cli_channel(ch) for ch in args.channels]
    elif args.all_channels:
        channels = ["WW", "bb", "tautau"]
    else:
        channels = [_normalise_cli_channel(args.channel)]
    _validate_dsph_source_channels(channels, str(args.dsph_source))
    masses = _resolve_mass_grid(args)

    for channel in channels:
        preset = FIT_PRESETS[str(args.fit)]
        print(f"\n{'#'*65}")
        print(f"  CHANNEL: {channel}")
        print(f"  FIT:     {args.fit} ({preset['description']})")
        print(f"  MODE:    {'spatial fit + spectral summary' if args.run_spatial_fit else 'spectral only'}")
        print(f"  Masses:  {masses}")
        print(f"  Scattering operator: {args.scatter_operator or 'none (pure annihilation)'}")
        print(f"{'#'*65}")

        if args.output_dir:
            outdir = Path(args.output_dir) / channel
        elif preset["result_subdir"] is None:
            outdir = RESULTS_DIR / channel
        else:
            outdir = RESULTS_DIR / str(preset["result_subdir"]) / channel
        if args.scatter_operator:
            c_str = f"{args.scatter_coupling:.2e}".replace("+","").replace(".","p")
            outdir = outdir.parent / f"{outdir.name}_scat_{args.scatter_operator}_{c_str}"

        results = []
        t_total = time.time()

        if not args.run_spatial_fit:
            masses_arr = np.asarray(sorted(float(m) for m in masses), dtype=float)
            if args.dry_run:
                print("  [DRY-RUN] Would fit one PPPC spectral normalization for each mass.")
                print("  [DRY-RUN] Would not build templates, write MCMC configs, or run per-bin fits.")
                saved_profile = None
            else:
                saved_profile = _save_spectral_only_scan(
                    channel=channel,
                    masses=masses_arr,
                    fit=str(args.fit),
                    outdir=outdir,
                    args=args,
                    table_path=table_path,
                )

            print(f"\n{'='*65}")
            print(f"  Channel {channel} complete.  "
                  f"Wall time: {(time.time() - t_total)/60:.1f} min")
            if saved_profile is not None and np.any(np.isfinite(saved_profile["delta_lnL"])):
                best_idx = int(np.nanargmax(saved_profile["delta_lnL"]))
                print(f"  Preferred mass: {saved_profile['masses'][best_idx]:.0f} GeV")
                print(f"  <sigma v>_best: {saved_profile['sigmav_best'][best_idx]:.3e} cm^3/s")
                print(f"  dSph tension:   {saved_profile['tensions'][best_idx]:.2f}x")
            continue

        for mass in sorted(masses):
            r = run_one_mass(
                channel=channel,
                mass=float(mass),
                args=args,
                table_path=table_path,
                dry_run=args.dry_run,
            )
            if r is not None:
                results.append(r)

        saved_profile = None
        if results and not args.dry_run:
            saved_profile = _save_and_plot(results, channel, outdir, args)

        print(f"\n{'='*65}")
        print(f"  Channel {channel} complete.  "
              f"Wall time: {(time.time() - t_total)/60:.1f} min")
        if results:
            if saved_profile is not None and np.any(np.isfinite(saved_profile["delta_lnL"])):
                best_idx = int(np.nanargmax(saved_profile["delta_lnL"]))
                sigmav_print = float(saved_profile["sigmav_best"][best_idx])
                tension_print = float(saved_profile["tensions"][best_idx])
            else:
                best_idx = int(np.nanargmax([r["total_lnL"] for r in results]))
                sigmav_print = float(results[best_idx]["sigmav_best"])
                tension_print = float(results[best_idx]["tension"])
            best = results[best_idx]
            print(f"  Preferred mass: {best['mass_GeV']:.0f} GeV")
            print(f"  <sigma v>_best: {sigmav_print:.3e} cm^3/s")
            print(f"  dSph tension:   {tension_print:.2f}x")


if __name__ == "__main__":
    main()
