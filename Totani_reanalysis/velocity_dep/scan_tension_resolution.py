#!/usr/bin/env python3
"""
Compare Totani spectral fits against the dSph-safe fit definition.

This script fits the Totani MCMC halo spectrum with unmodified PPPC
annihilation templates.  For each annihilation channel it reports two points:

  1. Spectral best fit:
     the annihilation mass with minimum chi2.

  2. New dSph-safe best fit:
     the minimum-chi2 point satisfying

         chi2 <= chi2_min + Delta chi2
         <sigma v> <= <sigma v>_dSph,95

The output is a tension-summary panel, per-channel text summaries, and NPZ
tables.

Example
-------
python Totani_reanalysis/velocity_dep/scan_tension_resolution.py \
  --ann-mass-min 100 \
  --ann-mass-max 5000 \
  --n-ann-mass 80 \
  --cl-threshold 0.95
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.spectral_reshaping import (  # noqa: E402
    best_fit_normalization,
    pppc_energy_flux_template,
    smooth_nfw_sigma_v_from_norm,
)
from core.dsph_limits_multi import (  # noqa: E402
    DSph_LIMIT_SOURCES,
    dsph_limit_source,
    dsph_upper_limit,
)
from core.totani_data_loader import _MCMC_DIRS, load_halo_spectrum  # noqa: E402


_ALL_ANN_CHANNELS = ("WW", "bb", "tautau")
_DEFAULT_PROFILE_ORDER = ("rho2", "rho2.5", "rho1")
_GLOBAL_PROFILE_ORDER = ("global_rho2", "global_rho2.5", "global_rho1")
_DEFAULT_CL_THRESHOLD = 0.95
_DEFAULT_DELTA_CHI2_2DOF = float(-2.0 * np.log(1.0 - _DEFAULT_CL_THRESHOLD))

_PROFILE_LABELS = {
    "rho2": r"$\rho^2$",
    "rho2.5": r"$\rho^{2.5}$",
    "rho1": r"$\rho^1$",
    "fig2_3": "Fig. 2/3",
    "fig4": "Fig. 4",
    "global_rho2": r"global $\rho^2$",
    "global_rho2.5": r"global $\rho^{2.5}$",
    "global_rho1": r"global $\rho^1$",
    "global_fig2_3": "global Fig. 2/3",
    "global_fig4": "global Fig. 4",
}
_POINT_LABEL_OFFSETS = {
    "rho2": [(-18, 27), (0, -25), (18, 23)],
    "rho2.5": [(-30, -23), (0, -23), (-34, -15)],
    "rho1": [(0, 22), (0, -25), (-32, -15)],
    "global_rho2": [(-18, 27), (0, -25), (18, 23)],
    "global_rho2.5": [(-30, -23), (0, -23), (-34, -15)],
    "global_rho1": [(0, 22), (0, -25), (-32, -15)],
}
_CHANNEL_MARKERS = {"WW": "o", "bb": "s", "tautau": "^"}
_CHANNEL_MS = {"WW": 14, "bb": 13, "tautau": 14}
_BOUNDARY_LIME = "#7CFC00"
_BOUNDARY_CYAN = "#00E5FF"


@dataclass
class FitPoint:
    ann_mass: float
    chi2: float
    ndof: int
    norm: float
    sigmav: float
    sigmav_dsph: float
    tension: float
    source: np.ndarray
    model: np.ndarray


def _normalise_ann_channel(channel: str) -> str:
    key = str(channel).strip().lower().replace("+", "").replace("-", "").replace("_", "")
    if key in ("ww", "w"):
        return "WW"
    if key in ("bb", "bbar", "bbbar", "b"):
        return "bb"
    if key in ("tautau", "tau", "tauplustau", "tauplustauminus"):
        return "tautau"
    return str(channel).strip()


def _ann_channels_from_args(args: argparse.Namespace) -> list[str]:
    raw = str(args.ann_channel).strip()
    if raw.lower() in ("all", "all3", "all-three", "all_three"):
        return list(_ALL_ANN_CHANNELS)
    if "," in raw:
        return [_normalise_ann_channel(ch) for ch in raw.split(",") if ch.strip()]
    return [_normalise_ann_channel(raw)]


def _halo_profiles_from_args(args: argparse.Namespace) -> list[str]:
    raw = str(args.halo_profile).strip()
    if raw.lower() == "all":
        return [p for p in _DEFAULT_PROFILE_ORDER if p in _MCMC_DIRS]
    if raw.lower() in ("all_global", "global_all", "all-global"):
        return [p for p in _GLOBAL_PROFILE_ORDER if p in _MCMC_DIRS]
    return [raw]


def _channel_plot_label(channel: str) -> str:
    ch = _normalise_ann_channel(channel)
    if ch == "bb":
        return r"$b\bar{b}$"
    if ch == "tautau":
        return r"$\tau^+\tau^-$"
    return ch


def _cl_from_delta_chi2_2d(delta_chi2: float) -> float:
    return float(1.0 - np.exp(-0.5 * float(delta_chi2)))


def _delta_chi2_from_cl_2d(cl: float) -> float:
    cl = float(cl)
    if not (0.0 < cl < 1.0):
        raise ValueError("--cl-threshold must be between 0 and 1, e.g. 0.95.")
    return float(-2.0 * np.log(1.0 - cl))


def _load_sys_budget(path: str, E_bins_GeV: np.ndarray) -> np.ndarray:
    """Interpolate the per-bin halo systematic sigma_sys(E) onto the fit energy bins.

    Reads the table written by Totani_paper_check/mcmc/build_systematic_budget.py;
    the absolute systematic is the 'sigma_sys' column (3rd from last). Interpolated
    in log-E and returned in the same flux units as the halo spectrum.
    """
    data = np.loadtxt(path, comments="#")
    E = np.asarray(data[:, 0], float)
    sigma_sys = np.asarray(data[:, -3], float)
    lE = np.log(np.asarray(E_bins_GeV, float))
    return np.interp(lE, np.log(E), sigma_sys, left=sigma_sys[0], right=sigma_sys[-1])


def evaluate_fit_point(
    *,
    ann_mass: float,
    E_bins_GeV: np.ndarray,
    phi_data: np.ndarray,
    phi_err: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
    ann_channel: str,
) -> FitPoint:
    nE = len(E_bins_GeV)
    ndof = int(mask.sum()) - 1
    nan_vec = np.full(nE, np.nan)

    try:
        source = pppc_energy_flux_template(
            E_bins_GeV,
            ann_mass,
            channel=ann_channel,
            primary="gamma",
            table_path=args.pppc_gamma_table,
            normalise=False,
        )
    except (ValueError, ImportError, FileNotFoundError):
        return FitPoint(
            ann_mass, np.nan, ndof, np.nan, np.nan, np.nan, np.nan,
            nan_vec.copy(), nan_vec.copy(),
        )

    if not np.any(source > 0.0):
        return FitPoint(
            ann_mass, np.nan, ndof, np.nan, np.nan, np.nan, np.nan,
            source.copy(), nan_vec.copy(),
        )

    norm = best_fit_normalization(source, phi_data, phi_err, mask)
    if not np.isfinite(norm) or norm < 0.0:
        return FitPoint(
            ann_mass, np.nan, ndof, norm, np.nan, np.nan, np.nan,
            source.copy(), nan_vec.copy(),
        )

    model = norm * source
    chi2 = float(np.sum(((model[mask] - phi_data[mask]) / phi_err[mask]) ** 2))
    sigmav = smooth_nfw_sigma_v_from_norm(norm, ann_mass)
    sigmav_dsph = dsph_upper_limit(
        ann_mass, ann_channel, source=getattr(args, "dsph_source", "hoof")
    )
    tension = sigmav / sigmav_dsph if np.isfinite(sigmav_dsph) and sigmav_dsph > 0.0 else np.nan

    return FitPoint(
        ann_mass, chi2, ndof, norm, sigmav, sigmav_dsph, tension,
        source.copy(), model.copy(),
    )


def run_channel_scan(
    *,
    halo,
    phi_err: np.ndarray,
    mask: np.ndarray,
    ann_masses: np.ndarray,
    args: argparse.Namespace,
    ann_channel: str,
    outdir: Path,
) -> dict:
    E = halo.E_bins_GeV
    phi = halo.phi
    err = np.asarray(phi_err, dtype=float)
    ndof = int(mask.sum()) - 1
    cl = float(getattr(args, "cl_threshold_effective", _DEFAULT_CL_THRESHOLD))
    delta_chi2_allowed = float(args.delta_chi2_allowed)

    points = [
        evaluate_fit_point(
            ann_mass=float(m),
            E_bins_GeV=E,
            phi_data=phi,
            phi_err=err,
            mask=mask,
            args=args,
            ann_channel=ann_channel,
        )
        for m in ann_masses
    ]

    chi2_arr = np.asarray([p.chi2 for p in points], dtype=float)
    norm_arr = np.asarray([p.norm for p in points], dtype=float)
    sigmav_arr = np.asarray([p.sigmav for p in points], dtype=float)
    dsph_arr = np.asarray([p.sigmav_dsph for p in points], dtype=float)
    tension_arr = np.asarray([p.tension for p in points], dtype=float)

    finite = np.isfinite(chi2_arr)
    if not np.any(finite):
        print(f"\n[{ann_channel}] no finite PPPC fits.")
        return {}

    spectral_i = int(np.nanargmin(chi2_arr))
    spectral = points[spectral_i]
    chi2_min = float(spectral.chi2)
    cl_allowed = finite & (chi2_arr <= chi2_min + delta_chi2_allowed)
    dsph_safe = np.isfinite(tension_arr) & (tension_arr <= 1.0)
    new_best_mask = cl_allowed & dsph_safe

    if np.any(new_best_mask):
        new_i = int(np.nanargmin(np.where(new_best_mask, chi2_arr, np.nan)))
        new_best = points[new_i]
        new_valid = True
    else:
        new_i = -1
        new_best = None
        new_valid = False

    constrained_mask = cl_allowed & np.isfinite(tension_arr)
    if np.any(constrained_mask):
        min_tension_i = int(np.nanargmin(np.where(constrained_mask, tension_arr, np.nan)))
        constrained_best = points[min_tension_i]
        constrained_valid = True
    else:
        min_tension_i = -1
        constrained_best = None
        constrained_valid = False

    print(f"\n[{ann_channel}] spectral vs constrained dSph-tension fit")
    print(
        f"  Spectral best: m_ann={spectral.ann_mass:.4g} GeV, "
        f"chi2={spectral.chi2:.3f} ({spectral.chi2 / ndof:.2f}/dof), "
        f"<sv>={spectral.sigmav:.3e}, tension={spectral.tension:.3f}x"
    )
    if constrained_best is not None:
        print(
            f"  Constrained best: m_ann={constrained_best.ann_mass:.4g} GeV, "
            f"delta_chi2={constrained_best.chi2 - chi2_min:.3f}, "
            f"<sv>={constrained_best.sigmav:.3e}, "
            f"tension={constrained_best.tension:.3f}x"
        )
    else:
        print(
            f"  Constrained best: none with finite dSph tension inside "
            f"{100.0 * cl:.1f}% CL (Delta chi2 <= {delta_chi2_allowed:.3g})"
        )
    if new_best is not None:
        print(
            f"  dSph-safe best: m_ann={new_best.ann_mass:.4g} GeV, "
            f"delta_chi2={new_best.chi2 - chi2_min:.3f}, "
            f"<sv>={new_best.sigmav:.3e}, tension={new_best.tension:.3f}x"
        )
    else:
        print(
            f"  dSph-safe best: none inside {100.0 * cl:.1f}% CL "
            f"(Delta chi2 <= {delta_chi2_allowed:.3g})"
        )

    result = dict(
        ann_channel=np.array(ann_channel),
        halo_profile=np.array(args.halo_profile),
        ann_masses_GeV=ann_masses.astype(np.float32),
        chi2=chi2_arr.astype(np.float32),
        chi2_per_dof=(chi2_arr / ndof).astype(np.float32),
        norm=norm_arr.astype(np.float32),
        sigmav_cm3_s=sigmav_arr.astype(np.float32),
        sigmav_dsph_cm3_s=dsph_arr.astype(np.float32),
        tension=tension_arr.astype(np.float32),
        cl_allowed_mask=cl_allowed.astype(np.uint8),
        dsph_safe_mask=dsph_safe.astype(np.uint8),
        new_best_mask=new_best_mask.astype(np.uint8),
        ndof=np.int32(ndof),
        chi2_min=np.float32(chi2_min),
        delta_chi2_allowed=np.float32(delta_chi2_allowed),
        cl_threshold=np.float32(cl),
        err_mode=np.array(args.err_mode),
        data_source=np.array("Totani MCMC halo posterior, central=f_p50"),
        dsph_limit_source=np.array(
            dsph_limit_source(ann_channel, source=getattr(args, "dsph_source", "hoof"))
        ),
        n_finite=np.int32(np.sum(finite)),
        n_cl_allowed=np.int32(np.sum(cl_allowed)),
        n_dsph_safe=np.int32(np.sum(dsph_safe)),
        n_new_best_candidates=np.int32(np.sum(new_best_mask)),
        n_constrained_candidates=np.int32(np.sum(constrained_mask)),
        spectral_best_i=np.int32(spectral_i),
        spectral_best_ann_mass_GeV=np.float32(spectral.ann_mass),
        spectral_best_chi2=np.float32(spectral.chi2),
        spectral_best_delta_chi2=np.float32(0.0),
        spectral_best_chi2_per_dof=np.float32(spectral.chi2 / ndof),
        spectral_best_sigmav_cm3_s=np.float32(spectral.sigmav),
        spectral_best_sigmav_dsph_cm3_s=np.float32(spectral.sigmav_dsph),
        spectral_best_tension=np.float32(spectral.tension),
        spectral_best_model=spectral.model.astype(np.float32),
        spectral_best_source=spectral.source.astype(np.float32),
        new_best_valid=np.bool_(new_valid),
        new_best_i=np.int32(new_i),
        new_best_ann_mass_GeV=np.float32(new_best.ann_mass) if new_best is not None else np.float32(np.nan),
        new_best_chi2=np.float32(new_best.chi2) if new_best is not None else np.float32(np.nan),
        new_best_delta_chi2=np.float32(new_best.chi2 - chi2_min) if new_best is not None else np.float32(np.nan),
        new_best_chi2_per_dof=np.float32(new_best.chi2 / ndof) if new_best is not None else np.float32(np.nan),
        new_best_sigmav_cm3_s=np.float32(new_best.sigmav) if new_best is not None else np.float32(np.nan),
        new_best_sigmav_dsph_cm3_s=np.float32(new_best.sigmav_dsph) if new_best is not None else np.float32(np.nan),
        new_best_tension=np.float32(new_best.tension) if new_best is not None else np.float32(np.nan),
        new_best_model=new_best.model.astype(np.float32) if new_best is not None else np.full_like(E, np.nan, dtype=np.float32),
        new_best_source=new_best.source.astype(np.float32) if new_best is not None else np.full_like(E, np.nan, dtype=np.float32),
        constrained_best_valid=np.bool_(constrained_valid),
        constrained_best_i=np.int32(min_tension_i),
        constrained_best_ann_mass_GeV=np.float32(constrained_best.ann_mass) if constrained_best is not None else np.float32(np.nan),
        constrained_best_chi2=np.float32(constrained_best.chi2) if constrained_best is not None else np.float32(np.nan),
        constrained_best_delta_chi2=np.float32(constrained_best.chi2 - chi2_min) if constrained_best is not None else np.float32(np.nan),
        constrained_best_chi2_per_dof=np.float32(constrained_best.chi2 / ndof) if constrained_best is not None else np.float32(np.nan),
        constrained_best_sigmav_cm3_s=np.float32(constrained_best.sigmav) if constrained_best is not None else np.float32(np.nan),
        constrained_best_sigmav_dsph_cm3_s=np.float32(constrained_best.sigmav_dsph) if constrained_best is not None else np.float32(np.nan),
        constrained_best_tension=np.float32(constrained_best.tension) if constrained_best is not None else np.float32(np.nan),
        constrained_best_model=constrained_best.model.astype(np.float32) if constrained_best is not None else np.full_like(E, np.nan, dtype=np.float32),
        constrained_best_source=constrained_best.source.astype(np.float32) if constrained_best is not None else np.full_like(E, np.nan, dtype=np.float32),
        min_cl_tension_i=np.int32(min_tension_i),
        min_cl_tension=np.float32(tension_arr[min_tension_i]) if min_tension_i >= 0 else np.float32(np.nan),
        min_cl_tension_ann_mass_GeV=np.float32(ann_masses[min_tension_i]) if min_tension_i >= 0 else np.float32(np.nan),
        E_bins_GeV=E.astype(np.float32),
        phi_data=phi.astype(np.float32),
        phi_err=err.astype(np.float32),
        mask=mask.astype(np.uint8),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "annihilation_fit_comparison.npz"
    np.savez_compressed(str(npz_path), **result)
    write_summary(result, outdir / "summary.txt")
    print(f"  Saved: {npz_path}")
    return result


def write_summary(result: dict, path: Path) -> None:
    cl_pct = 100.0 * float(result["cl_threshold"])
    constrained_valid = bool(result.get("constrained_best_valid", int(result["min_cl_tension_i"]) >= 0))
    constrained_i = int(result.get("constrained_best_i", result["min_cl_tension_i"]))
    chi2 = np.asarray(result["chi2"], dtype=float)
    constrained_chi2 = (
        float(result["constrained_best_chi2"])
        if "constrained_best_chi2" in result
        else (float(chi2[constrained_i]) if 0 <= constrained_i < chi2.size else np.nan)
    )
    constrained_delta = constrained_chi2 - float(result["chi2_min"]) if np.isfinite(constrained_chi2) else np.nan
    constrained_mass = float(
        result.get("constrained_best_ann_mass_GeV", result["min_cl_tension_ann_mass_GeV"])
    )
    constrained_tension = float(result.get("constrained_best_tension", result["min_cl_tension"]))
    constrained_sigmav = float(result.get("constrained_best_sigmav_cm3_s", np.nan))
    constrained_dsph = float(result.get("constrained_best_sigmav_dsph_cm3_s", np.nan))
    constrained_chi2_per_dof = constrained_chi2 / max(1, int(result["ndof"])) if np.isfinite(constrained_chi2) else np.nan
    lines = [
        "Spectral best fit vs constrained dSph-tension fit",
        "=================================================",
        f"channel       : {str(result['ann_channel'])}",
        f"halo profile  : {str(result['halo_profile'])}",
        f"data source   : {str(result['data_source'])}",
        f"dSph limits   : {str(result.get('dsph_limit_source', dsph_limit_source(str(result['ann_channel']))))}",
        f"error mode    : posterior interval, {str(result['err_mode'])}",
        f"ndof          : {int(result['ndof'])}",
        f"allowed region: chi2 <= chi2_min + {float(result['delta_chi2_allowed']):.4g} "
        f"({cl_pct:.1f}% CL, 2 dof)",
        f"n_finite      : {int(result['n_finite'])}",
        f"n_cl_allowed  : {int(result['n_cl_allowed'])}",
        f"n_dsph_safe   : {int(result['n_dsph_safe'])}",
        f"n_new_best    : {int(result['n_new_best_candidates'])} "
        "(CL-allowed and dSph-safe candidates)",
        f"n_constrained : {int(result.get('n_constrained_candidates', 0))} "
        "(CL-allowed candidates with finite dSph tension)",
        "",
        "Spectral best fit (Totani-style):",
        "  definition             : minimum chi2 over annihilation mass; dSph limits are not used",
        f"  m_ann                  : {float(result['spectral_best_ann_mass_GeV']):.4g} GeV",
        f"  chi2                   : {float(result['spectral_best_chi2']):.4f}",
        f"  delta chi2             : {float(result['spectral_best_delta_chi2']):.4f}",
        f"  chi2 / ndof            : {float(result['spectral_best_chi2_per_dof']):.4f}",
        f"  <sigma v>              : {float(result['spectral_best_sigmav_cm3_s']):.4e} cm^3/s",
        f"  <sigma v>_dSph (95%CL) : {float(result['spectral_best_sigmav_dsph_cm3_s']):.4e} cm^3/s",
        f"  tension factor         : {float(result['spectral_best_tension']):.3f}x",
        "",
        "Constrained best fit (dSph-tension target):",
        "  definition             : minimum dSph tension among points inside the data CL window",
        f"  valid                  : {constrained_valid}",
        f"  m_ann                  : {constrained_mass:.4g} GeV",
        f"  chi2                   : {constrained_chi2:.4f}",
        f"  delta chi2             : {constrained_delta:.4f}",
        f"  chi2 / ndof            : {constrained_chi2_per_dof:.4f}",
        f"  <sigma v>              : {constrained_sigmav:.4e} cm^3/s",
        f"  <sigma v>_dSph (95%CL) : {constrained_dsph:.4e} cm^3/s",
        f"  tension factor         : {constrained_tension:.3f}x",
        "",
        "95% CL + dSph-safe best fit:",
        "  definition             : minimum chi2 among points inside the data CL window "
        "and with <sigma v> <= <sigma v>_dSph,95",
        f"  valid                  : {bool(result['new_best_valid'])}",
        f"  m_ann                  : {float(result['new_best_ann_mass_GeV']):.4g} GeV",
        f"  chi2                   : {float(result['new_best_chi2']):.4f}",
        f"  delta chi2             : {float(result['new_best_delta_chi2']):.4f}",
        f"  chi2 / ndof            : {float(result['new_best_chi2_per_dof']):.4f}",
        f"  <sigma v>              : {float(result['new_best_sigmav_cm3_s']):.4e} cm^3/s",
        f"  <sigma v>_dSph (95%CL) : {float(result['new_best_sigmav_dsph_cm3_s']):.4e} cm^3/s",
        f"  tension factor         : {float(result['new_best_tension']):.3f}x",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Summary: {path}")


def _scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def load_existing_scan_results(root_out: Path) -> list[dict]:
    """Load previously saved per-channel/profile scan NPZ files for summary plots."""
    results: list[dict] = []
    root_out = Path(root_out)
    for path in sorted(root_out.glob("annihilation_fit_*_*/annihilation_fit_comparison.npz")):
        try:
            with np.load(path, allow_pickle=False) as npz:
                result = {key: npz[key] for key in npz.files}
            results.append(result)
        except Exception as exc:
            print(f"  [summary] WARNING: could not load {path}: {exc}")
    return results


def merge_scan_results(existing: list[dict], current: list[dict]) -> list[dict]:
    """Merge result dictionaries by (halo_profile, ann_channel), current wins."""
    merged: dict[tuple[str, str], dict] = {}
    for result in existing + current:
        key = (_scalar_str(result["halo_profile"]), _scalar_str(result["ann_channel"]))
        merged[key] = result
    return list(merged.values())


def _style_summary_ax(ax) -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("#444")


def _result_plot_selection(result: dict, *, selection: str = "constrained") -> tuple[float, float, float, int]:
    mode = str(selection).strip().lower()
    if mode in ("spectral", "totani", "direct"):
        return (
            float(result["spectral_best_tension"]),
            float(result["spectral_best_ann_mass_GeV"]),
            float(result["spectral_best_chi2"]),
            int(result["spectral_best_i"]),
        )

    if "constrained_best_i" in result:
        idx = int(result["constrained_best_i"])
        if idx >= 0:
            return (
                float(result["constrained_best_tension"]),
                float(result["constrained_best_ann_mass_GeV"]),
                float(result["constrained_best_chi2"]),
                idx,
            )

    idx = int(result["min_cl_tension_i"])
    if idx >= 0:
        chi2 = np.asarray(result["chi2"], dtype=float)
        return (
            float(result["min_cl_tension"]),
            float(result["min_cl_tension_ann_mass_GeV"]),
            float(chi2[idx]) if idx < chi2.size else np.nan,
            idx,
        )

    if bool(result["new_best_valid"]):
        idx = int(result["new_best_i"])
        return (
            float(result["new_best_tension"]),
            float(result["new_best_ann_mass_GeV"]),
            float(result["new_best_chi2"]),
            idx,
        )

    return np.nan, np.nan, np.nan, -1


def _result_plot_values(result: dict, *, selection: str = "constrained") -> tuple[float, float, float]:
    y, mass, chi2, _ = _result_plot_selection(result, selection=selection)
    return y, mass, chi2


def make_tension_summary_plot(
    results: list[dict],
    out_path: Path,
    *,
    selection: str = "constrained",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.ticker import LogLocator, NullFormatter
        from helpers.fermi_plotting import (
            CONF_ANNOT_FS,
            CONF_HEADER_FS,
            CONF_LABEL_FS,
            CONF_LEGEND_FS,
            CONF_TICK_FS,
            CONF_TITLE_FS,
        )
        from helpers.plot_style import get_cmap_colors, set_paper_style
    except ImportError:
        return

    if not results:
        return

    set_paper_style(linewidth=1.5, n_colors=10, cmap_name="plasma")

    preferred_order = list(_DEFAULT_PROFILE_ORDER) + list(_GLOBAL_PROFILE_ORDER)
    profiles = [p for p in preferred_order if any(str(r["halo_profile"]) == p for r in results)]
    profiles += sorted({str(r["halo_profile"]) for r in results} - set(profiles))
    channels = [ch for ch in _ALL_ANN_CHANNELS if any(str(r["ann_channel"]) == ch for r in results)]
    by_key = {(str(r["halo_profile"]), str(r["ann_channel"])): r for r in results}
    plasma_colors = get_cmap_colors(cmap_name="plasma", n=7, start=0.15, end=0.95)
    profile_colors = [plasma_colors[1], plasma_colors[3], plasma_colors[5]]
    channel_colors = [plasma_colors[3], plasma_colors[5], plasma_colors[6]]
    x_pos = list(range(1, len(profiles) + 1))
    offsets = [-0.20, 0.0, 0.20]
    if len(channels) != 3:
        offsets = list(np.linspace(-0.20, 0.20, len(channels))) if len(channels) > 1 else [0.0]

    fig, ax = plt.subplots(1, 1, figsize=(16.0, 10.5))
    ax.axhline(1.0, color=_BOUNDARY_LIME, lw=3.0, ls="-", alpha=1.0, label="dSph 95% CL limit", zorder=2)
    ax.axhline(2.0, color=_BOUNDARY_CYAN, lw=2.4, ls="--", alpha=1.0, label="2x guide", zorder=2)
    ax.axhspan(0.0, 1.0, alpha=0.12, color=_BOUNDARY_LIME, zorder=1)

    for ci, channel in enumerate(channels):
        xs = []
        ys = []
        for pi, profile in enumerate(profiles):
            result = by_key.get((profile, channel))
            if result is None:
                continue
            y, _, _ = _result_plot_values(result, selection=selection)
            if not np.isfinite(y) or y <= 0.0:
                continue
            xs.append(x_pos[pi] + offsets[ci])
            ys.append(y)
        if xs:
            ax.plot(xs, ys, color=channel_colors[ci % len(channel_colors)],
                    lw=2.4, ls="-", alpha=1.0, zorder=3)

    for ci, channel in enumerate(channels):
        marker = _CHANNEL_MARKERS.get(channel, "D")
        ms = _CHANNEL_MS.get(channel, 14)
        for pi, profile in enumerate(profiles):
            result = by_key.get((profile, channel))
            if result is None:
                continue

            xval = x_pos[pi] + offsets[ci]
            plot_tension, plot_mass, plot_chi2, _ = _result_plot_selection(result, selection=selection)
            if not np.isfinite(plot_tension) or plot_tension <= 0.0:
                continue
            ndof = max(1, int(result["ndof"]))
            poor = np.isfinite(plot_chi2) and plot_chi2 / ndof > 3.0
            color = profile_colors[pi % len(profile_colors)]
            ax.plot(
                xval,
                plot_tension,
                marker=marker,
                ms=ms,
                color=color,
                mec="k",
                mfc=color if not poor else "none",
                mew=1.2 if not poor else 1.8,
                lw=0,
                zorder=5,
                alpha=1.0,
            )

    for ci, channel in enumerate(channels):
        for pi, profile in enumerate(profiles):
            result = by_key.get((profile, channel))
            if result is None:
                continue
            plot_tension, plot_mass, _ = _result_plot_values(result, selection=selection)
            if not np.isfinite(plot_tension) or plot_tension <= 0.0:
                continue
            xval = x_pos[pi] + offsets[ci]
            default_offset = (0, 24 if plot_tension < 1.0 else -26)
            dx, dy = _POINT_LABEL_OFFSETS.get(profile, [default_offset] * len(channels))[ci]
            label = rf"{plot_tension:.2f}$\times$" + "\n" + rf"$m_{{\rm ann}}={plot_mass:.0f}\,\mathrm{{GeV}}$"
            ax.annotate(
                label,
                xy=(xval, plot_tension),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=max(11.0, CONF_ANNOT_FS - 11),
                color="white",
                alpha=1.0,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=(0.05, 0.05, 0.08, 0.55), edgecolor="none"),
            )

    tau_channel_i = channels.index("tautau") if "tautau" in channels else None
    if tau_channel_i is not None:
        tau_points = []
        for pi, profile in enumerate(profiles):
            result = by_key.get((profile, "tautau"))
            if result is None:
                continue
            y, _, chi2 = _result_plot_values(result, selection=selection)
            ndof = max(1, int(result["ndof"]))
            if np.isfinite(y) and np.isfinite(chi2) and chi2 / ndof > 3.0:
                tau_points.append((pi, y, chi2 / ndof))
        if tau_points:
            pi, y, _ = max(tau_points, key=lambda item: item[1])
            ax.annotate(
                r"$\tau^+\tau^-$: poor spectral fit" "\n"
                r"(open symbols)",
                xy=(x_pos[pi] + offsets[tau_channel_i], y),
                xytext=(1.90, min(62.0, max(4.0, y * 1.4))),
                color="k",
                fontsize=max(14.0, CONF_ANNOT_FS - 7),
                alpha=1.0,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.6),
            )

    profile_handles = [
        mpatches.Patch(color=profile_colors[i % len(profile_colors)], label=_PROFILE_LABELS.get(profile, profile))
        for i, profile in enumerate(profiles)
    ]
    channel_handles = [
        plt.Line2D([0], [0], marker=_CHANNEL_MARKERS.get(ch, "D"), ms=10,
                   color=channel_colors[i % len(channel_colors)],
                   mfc="none", mec="k", mew=1.2, lw=0,
                   label=_channel_plot_label(ch))
        for i, ch in enumerate(channels)
    ]
    ref_handles = [
        plt.Line2D([0], [0], color=_BOUNDARY_LIME, lw=3.0, label="dSph limit (=1)"),
        plt.Line2D([0], [0], color=_BOUNDARY_CYAN, lw=2.4, ls="--", label="2x guide"),
    ]

    leg1 = ax.legend(handles=profile_handles, title="NFW profile",
                     title_fontsize=max(12.0, CONF_LEGEND_FS - 3),
                     loc="upper left", fontsize=max(12.0, CONF_LEGEND_FS - 4),
                     bbox_to_anchor=(1.02, 1.0),
                     borderaxespad=0.0, framealpha=0.18, edgecolor="#555")
    for text in leg1.get_texts():
        text.set_color("k")
    leg1.get_title().set_color("k")

    leg2 = ax.legend(handles=channel_handles + ref_handles,
                     title="Channel / reference",
                     title_fontsize=max(12.0, CONF_LEGEND_FS - 3),
                     loc="upper left", fontsize=max(12.0, CONF_LEGEND_FS - 4),
                     bbox_to_anchor=(1.02, 0.58),
                     borderaxespad=0.0, framealpha=0.18, edgecolor="#555")
    for text in leg2.get_texts():
        text.set_color("k")
    leg2.get_title().set_color("k")
    ax.add_artist(leg1)

    ax.set_yscale("log")
    ax.set_xlim(0.55, max(3.45, len(profiles) + 0.45))
    # Data-driven y-limits: velocity-dependent tensions can sit far below 1
    # (p-/d-wave) or far above it (Sommerfeld); a fixed window clips them off.
    _yvals = []
    for _r in results:
        _y, _, _ = _result_plot_values(_r, selection=selection)
        if np.isfinite(_y) and _y > 0.0:
            _yvals.append(float(_y))
    if _yvals:
        ymin = min(min(_yvals), 1.0) / 3.0
        ymax = max(max(_yvals), 2.0) * 3.0
    else:
        ymin, ymax = 0.12, 130.0
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_PROFILE_LABELS.get(p, p) for p in profiles],
                       fontsize=CONF_TICK_FS, color="k")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="both", colors="k", labelsize=CONF_TICK_FS)
    ax.tick_params(axis="x", colors="k", length=0)
    ax.grid(axis="y", which="major", color="#333", lw=0.6, alpha=0.7)
    ax.grid(axis="y", which="minor", color="#1e1e1e", lw=0.4, alpha=0.5)
    ax.set_ylabel(
        r"Tension  $\langle\sigma v\rangle_\mathrm{fit}\ /\ \langle\sigma v\rangle_\mathrm{dSph}$",
        fontsize=CONF_LABEL_FS, labelpad=8)
    if str(selection).strip().lower() in ("spectral", "totani", "direct"):
        title_tail = r"(direct spectral best fit, Totani-style)"
    else:
        title_tail = r"(minimum dSph tension inside the $\chi^2$ CL window)"
    ax.set_title(
        "Profile & channel dependence\n" + title_tail,
        fontsize=CONF_TITLE_FS, pad=8)
    _style_summary_ax(ax)

    # fig.suptitle(
    #     "Totani halo excess: profile/channel tension summary",
    #     color="white",
    #     fontsize=CONF_HEADER_FS,
    #     y=0.995,
    # )
    fig.tight_layout(rect=(0.0, 0.0, 0.72, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tension summary plot: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare the Totani spectral best fit to the dSph-safe best fit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--halo-profile", default="all",
                   choices=["all", "all_global"] + list(_MCMC_DIRS.keys()),
                   help="Halo spectrum to fit. Use all_global for the global-fit rho^2/rho^2.5/rho^1 summary.")
    p.add_argument("--nfw-label", default=None,
                   help="Exact NFW template label in MCMC npz files.")
    p.add_argument("--counts-path", default=None,
                   help="Path to counts CCUBE FITS for energy axis.")
    p.add_argument("--err-mode", default="sym",
                   choices=["sym", "lo", "hi", "max"],
                   help="Which halo posterior error bar to use in chi2.")
    p.add_argument("--include-nonpositive-bins", action="store_true", default=False,
                   help="Include finite nonpositive central-value bins in chi2.")

    p.add_argument("--ann-channel", default="all",
                   help="PPPC channel: WW, bb/bbar, tautau, comma-separated list, or all.")
    p.add_argument("--pppc-gamma-table", default=None,
                   help="Path to PPPC4DMID AtProduction_gammas.dat.")
    p.add_argument("--sys-budget", default=None,
                   help="Path to per-bin systematic budget (build_systematic_budget.py); "
                        "added in quadrature to the statistical halo error before fitting.")
    p.add_argument("--dsph-source", default="hoof", choices=list(DSph_LIMIT_SOURCES),
                   help="dSph limit set: s-wave (hoof/mcdaniel/boddy_swave) or "
                        "velocity-dependent (boddy_pwave/boddy_dwave/boddy_somm).")
    p.add_argument("--ann-mass", type=float, default=None,
                   help="Fix annihilation mass [GeV]. Overrides the mass grid.")
    p.add_argument("--ann-mass-min", type=float, default=100.0)
    p.add_argument("--ann-mass-max", type=float, default=5000.0)
    p.add_argument("--n-ann-mass", type=int, default=80)

    p.add_argument("--cl-threshold", type=float, default=None,
                   help="Confidence level for the allowed data window.")
    p.add_argument("--delta-chi2-allowed", type=float, default=_DEFAULT_DELTA_CHI2_2DOF,
                   help="Allowed-region threshold relative to the spectral best fit.")
    p.add_argument("--quick", action="store_true", default=False,
                   help="Use a coarse 20-point annihilation mass grid.")
    p.add_argument("--output-dir", default=None,
                   help="Root output directory.")
    p.add_argument("--summary-only", action="store_true", default=False,
                   help="Do not run scans; rebuild tension_summary_a.png from saved result NPZ files in --output-dir.")
    p.add_argument(
        "--summary-selection",
        default="constrained",
        choices=["constrained", "spectral"],
        help=(
            "Which fit to show in the summary plot/table. 'spectral' is the direct "
            "minimum-chi2 mass, matching Totani-style best-fit comparisons. "
            "'constrained' chooses the minimum dSph-tension point inside the "
            "--cl-threshold / --delta-chi2-allowed spectral window."
        ),
    )
    p.add_argument(
        "--style",
        default=None,
        help="Plot style: paper, conference/dark_transparent, or conference_light/light_transparent.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.style:
        os.environ["PLOT_STYLE"] = args.style
    if args.cl_threshold is not None:
        args.delta_chi2_allowed = _delta_chi2_from_cl_2d(args.cl_threshold)
        args.cl_threshold_effective = float(args.cl_threshold)
    else:
        args.cl_threshold_effective = _cl_from_delta_chi2_2d(args.delta_chi2_allowed)

    if args.quick:
        args.n_ann_mass = 20
        print("[quick mode] using 20 annihilation mass points")

    if args.ann_mass is not None:
        ann_masses = np.asarray([float(args.ann_mass)], dtype=float)
    else:
        ann_masses = np.logspace(
            np.log10(float(args.ann_mass_min)),
            np.log10(float(args.ann_mass_max)),
            int(args.n_ann_mass),
        )

    root_out = (
        Path(args.output_dir)
        if args.output_dir
        else _HERE / "results" / "tension_scan"
    )

    t0 = time.time()
    current_results = []
    if not args.summary_only:
        for halo_profile in _halo_profiles_from_args(args):
            print(f"\nLoading halo spectrum: {halo_profile}")
            halo = load_halo_spectrum(
                _MCMC_DIRS[halo_profile],
                nfw_label=args.nfw_label,
                counts_path=args.counts_path,
            )
            print(halo.summary())

            phi_err = getattr(halo, f"phi_err_{args.err_mode}", halo.phi_err_sym)
            phi_err = np.asarray(phi_err, dtype=float).copy()
            if args.sys_budget:
                sig_sys = _load_sys_budget(args.sys_budget, halo.E_bins_GeV)
                phi_err = np.sqrt(phi_err ** 2 + sig_sys ** 2)
                print(f"  Added systematic budget from {Path(args.sys_budget).name}: "
                      f"median sigma_sys/stat = "
                      f"{np.nanmedian(sig_sys / np.maximum(getattr(halo, f'phi_err_{args.err_mode}', halo.phi_err_sym), 1e-300)):.1f}x")
            mask = (
                halo.finite_mask if args.include_nonpositive_bins
                else halo.positive_mask & halo.finite_mask & (phi_err > 0)
            )
            print(f"  Chi2 mask: {int(mask.sum())} / {len(halo.E_bins_GeV)} bins")

            args.halo_profile = halo_profile
            for ann_channel in _ann_channels_from_args(args):
                outdir = root_out / f"annihilation_fit_{ann_channel}_{halo_profile}"
                result = run_channel_scan(
                    halo=halo,
                    phi_err=phi_err,
                    mask=mask,
                    ann_masses=ann_masses,
                    args=args,
                    ann_channel=ann_channel,
                    outdir=outdir,
                )
                if result:
                    current_results.append(result)

    existing_results = load_existing_scan_results(root_out)
    results = merge_scan_results(existing_results, current_results)
    if existing_results:
        print(f"\n[summary] loaded {len(existing_results)} saved scan result(s) from {root_out}")
    print(f"[summary] plotting {len(results)} merged profile/channel result(s)")
    summary_stem = (
        "tension_summary_spectral.png"
        if str(args.summary_selection).strip().lower() == "spectral"
        else "tension_summary_a.png"
    )
    make_tension_summary_plot(
        results,
        root_out / summary_stem,
        selection=str(args.summary_selection),
    )

    print(f"\n{'=' * 78}")
    print("ANNIHILATION FIT COMPARISON COMPLETE")
    print(f"{'=' * 78}")
    print(
        f"{'Profile':<14} {'Channel':<8} {'m_spec':>10} {'t_spec':>9} "
        f"{'m_plot':>10} {'dchi2_plot':>11} {'t_plot':>9} {'selection':>11}"
    )
    print("-" * 88)
    for r in results:
        plot_tension, plot_mass, plot_chi2, _ = _result_plot_selection(
            r,
            selection=str(args.summary_selection),
        )
        chi2_min = float(r["chi2_min"])
        selection_mode = str(args.summary_selection).strip().lower()
        if selection_mode == "spectral":
            mode = "spectral"
        elif np.isfinite(plot_tension):
            mode = "min-tens"
        else:
            mode = "none"
        plot_delta = plot_chi2 - chi2_min if np.isfinite(plot_chi2) else np.nan
        print(
            f"{str(r['halo_profile']):<14} "
            f"{str(r['ann_channel']):<8} "
            f"{float(r['spectral_best_ann_mass_GeV']):>10.3g} "
            f"{float(r['spectral_best_tension']):>9.3f} "
            f"{plot_mass:>10.3g} "
            f"{plot_delta:>11.3f} "
            f"{plot_tension:>9.3f} "
            f"{mode:>10}"
        )
    print(f"{'=' * 78}")
    print(f"\nTotal wall time: {time.time() - t0:.1f} s")
    print(f"All results in: {root_out}")


if __name__ == "__main__":
    main()
