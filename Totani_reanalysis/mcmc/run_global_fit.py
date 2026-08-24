#!/usr/bin/env python3
"""
Global pixel-level likelihood maximization for Fermi-LAT GCE analysis.

Architecture mirrors traditional GCE papers (Ackermann+2017, Calore+2015):
  - Single global theta_m per template per energy bin (no cell-by-cell normalization)
  - Poisson log-likelihood summed over all fine pixels in a 15x15 deg ROI
  - MAP estimate via scipy L-BFGS-B (analytic gradient for speed)
  - Residual maps output: (d_ij - mu_ij) for GCE inspection

Run directly:
    python run_global_fit.py --energy-bin 20
    python run_global_fit.py  # fits all energy bins

Run via config wrapper:
    python run_global_from_config.py --config configs/gce_global.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from astropy.wcs import WCS
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    load_mask_any_shape,
)
from totani_helpers.fit_utils import (
    build_fit_mask3d,
    load_mu_templates_from_fits,
)
from mcmc_helper import totani_bounds

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None

try:
    from helpers.heartbar import heart_tqdm as _tqdm  # type: ignore  # noqa: F811
except Exception:
    pass

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

_DEFAULT_LABELS = [
    "gas",
    "iso",
    "ps",
    "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
    "loopA",
    "loopB",
    "ics",
    "fb_flat",
]

# ---------------------------------------------------------------------------
# Spatial helpers (self-contained; do not import from run_mcmc.py)
# ---------------------------------------------------------------------------

def _build_roi_box_mask(lon, lat, roi_lon, roi_lat):
    return (np.abs(lon) <= float(roi_lon)) & (np.abs(lat) <= float(roi_lat))


# ---------------------------------------------------------------------------
# Isotropic initialisation (replicated from run_mcmc.py to avoid import cycle)
# ---------------------------------------------------------------------------

def _infer_iso_E2dNdE_at_f1(mu_iso_vec, denom_vec, Ectr_mev):
    mu_iso_vec = np.asarray(mu_iso_vec, float).ravel()
    denom_vec = np.asarray(denom_vec, float).ravel()
    good = np.isfinite(mu_iso_vec) & np.isfinite(denom_vec) & (denom_vec > 0) & (mu_iso_vec > 0)
    if not np.any(good):
        raise RuntimeError("Cannot infer iso normalisation: no finite denom>0 & mu_iso>0 pixels.")
    I_med = float(np.median(mu_iso_vec[good] / denom_vec[good]))
    E2I = float(Ectr_mev ** 2) * I_med
    return E2I, I_med


def _build_global_init(
    *,
    labels,
    mu_pix,
    denom_pix,
    Ectr_mev,
    iso_target_E2=1e-4,
    iso_keys=("iso", "isotropic"),
    start_at_one_keys=("ps", "point_sources", "gas", "ics"),
    verbosity=1,
):
    """
    Totani-style initialisation adapted for the pixel-level global fit.
    Returns theta_init (ncomp,) — dimensionless template scalings.
    """
    labels_l = [str(s).lower() for s in labels]
    ncomp = len(labels)
    theta = np.zeros(ncomp, dtype=float)

    start1 = {k.lower() for k in start_at_one_keys}
    for j, lab in enumerate(labels_l):
        if lab in start1:
            theta[j] = 1.0

    if iso_target_E2 is not None:
        jiso = next((labels_l.index(k) for k in iso_keys if k in labels_l), None)
        if jiso is not None:
            E2_at_f1, _ = _infer_iso_E2dNdE_at_f1(mu_pix[jiso], denom_pix, Ectr_mev)
            if np.isfinite(E2_at_f1) and E2_at_f1 > 0:
                theta[jiso] = float(iso_target_E2) / E2_at_f1
                if verbosity >= 2:
                    print(
                        f"  iso init: E²dN/dE(f=1)={E2_at_f1:.3e}, "
                        f"target={iso_target_E2:.3e}, theta_iso={theta[jiso]:.4e}"
                    )

    return theta


# ---------------------------------------------------------------------------
# Core likelihood (pixel-level, analytic gradient)
# ---------------------------------------------------------------------------

def _nll_and_grad(theta, mu_pix, d_pix, cexp_floor):
    """
    Negative Poisson log-likelihood and its gradient w.r.t. theta.

    lnL = sum_j [ d_j * ln(mu_j) - mu_j ]   where  mu_j = sum_m theta_m * P_mj

    Gradient: d(lnL)/d(theta_m) = sum_j [ (d_j / mu_j - 1) * P_mj ]
    """
    Cexp = theta @ mu_pix                           # (npix,)
    Cexp = np.maximum(Cexp, cexp_floor)

    ratio = d_pix / Cexp                            # (npix,)
    ll = float(np.sum(d_pix * np.log(Cexp) - Cexp))
    grad_ll = mu_pix @ (ratio - 1.0)               # (ncomp,)

    return -ll, -grad_ll


def _poisson_covariance_from_curvature(theta, mu_pix, d_pix, cexp_floor, scipy_bounds):
    """
    Local covariance from the exact Poisson curvature at the optimum.

    For C_j = sum_m theta_m P_mj, the observed Hessian of the negative
    log-likelihood is

        H_mn = sum_j d_j P_mj P_nj / C_j^2 .

    This is much better behaved than the limited-memory inverse-Hessian object
    returned by L-BFGS-B, which is an optimizer preconditioner approximation,
    not a reliable statistical covariance estimate.
    """
    theta = np.asarray(theta, dtype=float)
    mu_pix = np.asarray(mu_pix, dtype=float)
    d_pix = np.asarray(d_pix, dtype=float)
    ncomp = int(theta.size)

    Cexp = theta @ mu_pix
    Cexp = np.maximum(Cexp, float(cexp_floor))

    free = np.ones(ncomp, dtype=bool)
    tol = 1e-8 * (1.0 + np.abs(theta))
    for m, (lo, hi) in enumerate(scipy_bounds):
        if lo is not None and hi is not None and np.isclose(float(lo), float(hi)):
            free[m] = False
            continue
        if lo is not None and theta[m] <= float(lo) + tol[m]:
            free[m] = False
            continue
        if hi is not None and theta[m] >= float(hi) - tol[m]:
            free[m] = False

    cov = np.zeros((ncomp, ncomp), dtype=float)
    if not np.any(free):
        return cov, free

    P = mu_pix[free]
    weights = d_pix / (Cexp * Cexp)
    H = (P * weights[np.newaxis, :]) @ P.T

    if np.any(~np.isfinite(H)):
        return cov, free

    try:
        cov_free = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_free = np.linalg.pinv(H, rcond=1e-10)

    cov_free = 0.5 * (cov_free + cov_free.T)
    idx = np.flatnonzero(free)
    cov[np.ix_(idx, idx)] = cov_free
    return cov, free


# ---------------------------------------------------------------------------
# Per-bin optimizer
# ---------------------------------------------------------------------------

def fit_global_one_bin(
    *,
    d_pix,
    mu_pix,
    theta_init,
    scipy_bounds,
    cexp_floor=1e-30,
    max_iter=5000,
    error_method="fisher",
    verbosity=1,
):
    """
    Maximize pixel-level Poisson likelihood via L-BFGS-B.

    Parameters
    ----------
    d_pix        : (npix,) observed counts per pixel
    mu_pix       : (ncomp, npix) template expected counts per pixel
    theta_init   : (ncomp,) initial coefficient vector
    scipy_bounds : list of (lo|None, hi|None) per component
    cexp_floor   : floor applied to Cexp before log (avoids log(0))
    max_iter     : maximum L-BFGS-B iterations
    verbosity    : 0 = silent, 1 = warnings only, 2 = verbose

    Returns
    -------
    dict with theta_ml, loglike_ml, success, message, nfev
    """
    res = minimize(
        _nll_and_grad,
        x0=theta_init.copy(),
        args=(mu_pix, d_pix, cexp_floor),
        jac=True,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-9},
    )

    theta_ml = res.x
    loglike_ml = -float(res.fun)

    # Extract 1-sigma standard errors.  Prefer the exact Poisson curvature at
    # the optimum; keep the L-BFGS-B inverse-Hessian estimate available only as
    # a diagnostic/backward-compatible option.
    ncomp = len(theta_ml)
    std = np.zeros(ncomp)
    cov = np.zeros((ncomp, ncomp), dtype=float)
    free_for_error = np.ones(ncomp, dtype=bool)

    method = str(error_method).strip().lower()
    if method in ("fisher", "curvature", "observed_fisher"):
        cov, free_for_error = _poisson_covariance_from_curvature(
            theta_ml, mu_pix, d_pix, cexp_floor, scipy_bounds
        )
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    elif method in ("lbfgsb", "lbfgs", "hess_inv"):
        try:
            if hasattr(res, "hess_inv") and res.hess_inv is not None:
                eye = np.eye(ncomp)
                cov_diag = np.array(
                    [float(res.hess_inv.dot(eye[m])[m]) for m in range(ncomp)]
                )
                cov[np.diag_indices(ncomp)] = cov_diag
                std = np.sqrt(np.maximum(cov_diag, 0.0))
        except Exception:
            pass
    elif method in ("none", "off"):
        pass
    else:
        raise ValueError(f"Unknown error_method={error_method!r}")

    if verbosity >= 2:
        print(f"  L-BFGS-B: {res.message} (nfev={res.nfev}, success={res.success})")
        print(f"  lnL(ML) = {loglike_ml:.4f}")
    elif verbosity >= 1 and not res.success:
        print(f"  WARNING: optimizer did not converge: {res.message}")

    return {
        "theta_ml": theta_ml,
        "std_phi":  std,
        "cov_phi": cov,
        "free_for_error": free_for_error,
        "error_method": method,
        "loglike_ml": loglike_ml,
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
    }


# ---------------------------------------------------------------------------
# Residual map builder
# ---------------------------------------------------------------------------

def make_residual_map(*, theta_ml, mu_k, counts_2d, mask_2d):
    """
    Build 2D residual map (d_ij - mu_ij) after global fit.

    Pixels outside the fit mask are set to NaN.

    Parameters
    ----------
    theta_ml   : (ncomp,) best-fit coefficients
    mu_k       : (ncomp, ny, nx) template maps for this energy bin
    counts_2d  : (ny, nx) observed counts
    mask_2d    : (ny, nx) boolean fit mask

    Returns
    -------
    resid_map  : (ny, nx)  d_ij - mu_ij,  NaN outside mask
    model_map  : (ny, nx)  mu_ij,          NaN outside mask
    frac_resid : (ny, nx)  (d-mu)/mu,      NaN where mu<=0 or outside mask
    """
    ny, nx = counts_2d.shape
    mu_total = np.zeros((ny, nx), dtype=float)
    for m, th in enumerate(theta_ml):
        mu_total += th * mu_k[m]

    nan_outside = np.full((ny, nx), np.nan)
    resid_map = nan_outside.copy()
    model_map = nan_outside.copy()
    frac_resid = nan_outside.copy()

    model_map[mask_2d] = mu_total[mask_2d]
    resid_map[mask_2d] = counts_2d[mask_2d].astype(float) - mu_total[mask_2d]

    nonzero = mask_2d & (mu_total > 0)
    frac_resid[nonzero] = resid_map[nonzero] / mu_total[nonzero]

    return resid_map, model_map, frac_resid


def check_model_positivity(*, model_map, mask_2d, floor: float) -> dict:
    """Return positivity diagnostics for the best-fit model inside the fit mask."""
    model = np.asarray(model_map, float)
    mask = np.asarray(mask_2d, bool) & np.isfinite(model)
    vals = model[mask]
    if vals.size == 0:
        return {
            "positivity_ok": False,
            "positivity_min_model": np.nan,
            "positivity_n_le0": -1,
            "positivity_n_floor": -1,
        }
    n_le0 = int(np.sum(vals <= 0.0))
    n_floor = int(np.sum(vals <= 10.0 * float(floor)))
    return {
        "positivity_ok": (n_le0 == 0 and n_floor == 0),
        "positivity_min_model": float(np.min(vals)),
        "positivity_n_le0": n_le0,
        "positivity_n_floor": n_floor,
    }


# ---------------------------------------------------------------------------
# scipy bounds conversion
# ---------------------------------------------------------------------------

def _to_scipy_bounds(bnds):
    """Convert list of (lo, hi) with ±inf to scipy-compatible (None → unbounded)."""
    return [
        (None if not np.isfinite(lo) else float(lo),
         None if not np.isfinite(hi) else float(hi))
        for lo, hi in bnds
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Global pixel-level likelihood fit for Fermi-LAT GCE analysis. "
            "Single theta per template, Poisson likelihood over all ROI pixels. "
            "Mimics Ackermann+2017 / Calore+2015 architecture."
        )
    )
    ap.add_argument(
        "--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--expo",
        default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--templates-dir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument(
        "--ext-mask",
        default=os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits"),
        help="Extended-source keep-mask FITS (True=keep). Pass '' to skip.",
    )
    ap.add_argument(
        "--roi-lon",
        type=float,
        default=7.5,
        help="ROI half-width in Galactic longitude (deg). Default: 7.5 -> 15x15 deg ROI.",
    )
    ap.add_argument(
        "--roi-lat",
        type=float,
        default=7.5,
        help="ROI half-width in Galactic latitude (deg). Default: 7.5 -> 15x15 deg ROI.",
    )
    ap.add_argument(
        "--exclude-disk",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Exclude |b| < disk_cut from the ROI.",
    )
    ap.add_argument("--disk-cut", type=float, default=2.0, help="Disk cut half-width (deg).")
    ap.add_argument(
        "--hemisphere",
        choices=["both", "north", "south"],
        default="both",
        help="Restrict the ROI to one Galactic hemisphere (b>0 north, b<0 south). "
             "A genuinely spherical halo must give a consistent spectrum in both; "
             "Loop I / bubble residuals are north-south asymmetric, so this is a "
             "systematics cross-check rather than a nominal fit.",
    )
    ap.add_argument(
        "--lon-sign",
        choices=["both", "east", "west"],
        default="both",
        help="Restrict the ROI in Galactic longitude (east=l>0, west=l<0). "
             "Combine with --hemisphere to isolate a quadrant: e.g. north+east = NE. "
             "A spherical halo must be consistent across all four quadrants.",
    )
    ap.add_argument("--binsz", type=float, default=0.125, help="Pixel size in degrees.")
    ap.add_argument(
        "--energy-bin",
        type=int,
        default=None,
        help="Single energy bin index (0-based). If omitted, all bins are fitted.",
    )
    ap.add_argument(
        "--energy-bins",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of energy bin indices. Overrides --energy-bin.",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Component labels. Each maps to templates_dir/mu_{label}_counts.fits.",
    )
    ap.add_argument(
        "--template-labels",
        nargs="+",
        default=None,
        help=(
            "Optional FITS template labels to load instead of --labels. "
            "Use this when the fitted coefficient label should remain stable "
            "but the input file is e.g. mu_<label>_psf_counts.fits."
        ),
    )
    ap.add_argument(
        "--negative-keys",
        nargs="+",
        default=["nfw", "fb_neg"],
        help="Substrings identifying components allowed to go negative (default: nfw, fb_neg).",
    )
    ap.add_argument(
        "--iso-target-e2",
        type=float,
        default=1e-4,
        help="Isotropic init target E²dN/dE [MeV cm⁻² s⁻¹ sr⁻¹].",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_fit_results"),
    )
    ap.add_argument(
        "--poisson-cexp-floor",
        type=float,
        default=1e-30,
        help="Floor applied to expected counts before log (avoids log(0)).",
    )
    ap.add_argument(
        "--error-method",
        choices=["fisher", "lbfgsb", "none"],
        default="fisher",
        help=(
            "How to estimate coefficient errors. 'fisher' uses the exact local "
            "Poisson curvature at the optimum; 'lbfgsb' uses scipy's optimizer "
            "inverse-Hessian approximation; 'none' writes zero-width intervals."
        ),
    )
    ap.add_argument("--max-iter", type=int, default=5000, help="Max L-BFGS-B iterations.")
    ap.add_argument(
        "--fix-labels",
        nargs="+",
        default=None,
        help=(
            "Labels to hold fixed at their init value during optimization "
            "(e.g. --fix-labels iso). Useful when a component is degenerate "
            "in the small ROI (iso is nearly flat in 15x15 deg)."
        ),
    )
    ap.add_argument(
        "--no-save-maps",
        action="store_true",
        help="Do not save residual/model maps (saves disk space for large ROI runs).",
    )
    ap.add_argument(
        "--check-positivity",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Check that the best-fit model is positive in the fit mask.",
    )
    ap.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0=silent, 1=summary (default), 2=verbose diagnostics.",
    )

    args = ap.parse_args()
    verbosity = int(args.verbosity)

    def vprint(level, *p, **k):
        if verbosity >= level:
            print(*p, **k)

    t_start = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    labels = list(args.labels) if args.labels is not None else list(_DEFAULT_LABELS)
    template_labels = list(args.template_labels) if args.template_labels is not None else list(labels)
    if len(template_labels) != len(labels):
        raise SystemExit(
            f"--template-labels length {len(template_labels)} does not match "
            f"--labels length {len(labels)}"
        )
    ncomp = len(labels)
    negative_keys = list(args.negative_keys) if args.negative_keys else ["nfw", "fb_neg"]

    vprint(1, "=" * 60)
    vprint(1, "Global Pixel-Level Fit  (GCE / Ackermann+2017 architecture)")
    vprint(1, "=" * 60)
    vprint(1, f"  ROI:        |l| <= {args.roi_lon}°,  |b| <= {args.roi_lat}°  ({2*args.roi_lon}x{2*args.roi_lat} deg)")
    vprint(1, f"  Components: {labels}")
    if template_labels != labels:
        vprint(1, f"  Template files: {template_labels}")
    vprint(1, f"  Outdir:     {args.outdir}")

    # ------------------------------------------------------------------
    # Load FITS data
    # ------------------------------------------------------------------
    vprint(1, "\nLoading FITS data...")

    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial
    vprint(2, f"  Counts cube: {counts.shape}")

    expo_raw, E_expo_mev = read_exposure(args.expo)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    vprint(2, f"  Exposure resampled: {expo.shape}")

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi2d = _build_roi_box_mask(lon, lat, args.roi_lon, args.roi_lat)
    if args.exclude_disk:
        roi2d = roi2d & (np.abs(lat) >= float(args.disk_cut))
    if args.hemisphere == "north":
        roi2d = roi2d & (lat > 0.0)
    elif args.hemisphere == "south":
        roi2d = roi2d & (lat < 0.0)
    # Longitude split (orthogonal to hemisphere); combine the two for quadrants.
    # Convention: "east" = l>0, "west" = l<0 in the wrapped [-180,180] grid.
    if args.lon_sign == "east":
        roi2d = roi2d & (lon > 0.0)
    elif args.lon_sign == "west":
        roi2d = roi2d & (lon < 0.0)
    vprint(1, f"  ROI pixels:  {roi2d.sum()} / {roi2d.size}  "
              f"(hemisphere={args.hemisphere}, lon_sign={args.lon_sign})")

    # Extended-source mask
    srcmask = np.ones((nE, ny, nx), bool)
    ext_mask_path = str(args.ext_mask).strip() if args.ext_mask else ""
    if ext_mask_path and os.path.exists(ext_mask_path):
        ext_keep3d = load_mask_any_shape(ext_mask_path, counts.shape)
        srcmask &= ext_keep3d
        vprint(2, f"  Extended-source mask applied: {ext_mask_path}")

    # ------------------------------------------------------------------
    # Load templates  (mu_{label}_counts.fits, shape (nE, ny, nx))
    # ------------------------------------------------------------------
    vprint(1, "\nLoading templates...")
    mu_list, _ = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=template_labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
        progress=verbosity > 0,
        progress_desc="Templates",
    )
    # Stack into (ncomp, nE, ny, nx) — kept in memory; 8 comps * 60 bins * 512^2 ~ 1 GB float32
    mu_cube = np.stack([np.asarray(m, dtype=np.float32) for m in mu_list], axis=0)
    vprint(2, f"  Template cube: {mu_cube.shape}")

    # ------------------------------------------------------------------
    # Energy bins
    # ------------------------------------------------------------------
    if args.energy_bins is not None:
        bins_to_run = sorted(set(int(b) for b in args.energy_bins))
    elif args.energy_bin is not None:
        bins_to_run = [int(args.energy_bin)]
    else:
        bins_to_run = list(range(nE))

    vprint(1, f"\nFitting {len(bins_to_run)} energy bin(s).")

    # ------------------------------------------------------------------
    # Bounds (same Totani convention as run_mcmc.py)
    # ------------------------------------------------------------------
    bnds_list = totani_bounds(labels, negative_keys=negative_keys)
    scipy_bnds = _to_scipy_bounds(bnds_list)

    # ------------------------------------------------------------------
    # Per-bin fit
    # ------------------------------------------------------------------
    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=srcmask,
        counts=counts,
        expo=expo,
    )

    summary_rows = []

    for k in bins_to_run:
        if k < 0 or k >= nE:
            vprint(1, f"  [Bin {k:02d}] SKIP: out of range [0, {nE-1}]")
            continue

        Ectr_k = float(Ectr_mev[k])
        dE_k = float(dE_mev[k])
        vprint(1, f"\n[Bin {k:02d}]  E_ctr = {Ectr_k:.1f} MeV ({Ectr_k/1e3:.3f} GeV)")

        # 2D mask for this bin; also exclude pixels where any template is non-finite
        mask_k = fit_mask3d[k].copy()
        for m in range(ncomp):
            mask_k &= np.isfinite(mu_cube[m, k])
        npix_k = int(mask_k.sum())
        vprint(2, f"  Fit pixels: {npix_k}")

        if npix_k < max(ncomp + 1, 10):
            vprint(1, f"  SKIP: too few valid pixels ({npix_k})")
            continue

        # ---- Extract pixel vectors (no cell aggregation) ----
        d_pix = counts[k][mask_k].astype(float)                     # (npix,)
        mu_pix = np.stack(
            [mu_cube[m, k][mask_k].astype(float) for m in range(ncomp)],
            axis=0,
        )                                                             # (ncomp, npix)
        denom_pix = (expo[k] * omega * dE_k)[mask_k].astype(float)  # (npix,)

        # ---- Initialise theta ----
        theta_init = _build_global_init(
            labels=labels,
            mu_pix=mu_pix,
            denom_pix=denom_pix,
            Ectr_mev=Ectr_k,
            iso_target_E2=args.iso_target_e2,
            verbosity=verbosity,
        )

        # ---- Fix selected components at their init value ----
        # Components in fix_labels are held at theta_init by tightening their
        # bounds to a point: [init, init].  The optimizer sees zero gradient
        # contribution from them (bounds clamp them), and the residual from their
        # fixed contribution is absorbed by the free components.
        fix_labels_set = set(str(x).lower() for x in (args.fix_labels or []))
        scipy_bnds_cur = list(scipy_bnds)
        if fix_labels_set:
            for m, lab in enumerate(labels):
                if lab.lower() in fix_labels_set:
                    v = float(theta_init[m])
                    scipy_bnds_cur[m] = (v, v)
                    vprint(2, f"  Fixing {lab} at theta={v:.4e}")

        # ---- Template rescaling for well-conditioned optimization ----
        # Templates can vary by many orders of magnitude in pixel sum across the
        # 15x15 ROI (e.g. loopA/loopB have sums ~1e13 vs gas ~2e5).  L-BFGS-B
        # computes step sizes from gradient magnitudes, so this disparity causes
        # it to converge at the initial point after a single tiny step.
        # Fix: work in a normalised coordinate  phi_m = theta_m * S_m
        # where S_m = sum_j(mu_pix[m,j]) / sum_j(d_pix), so every component's
        # template sums to data_sum in phi-space and gradients are O(data_sum).
        # Physical theta is recovered as  theta_m = phi_m / S_m.
        data_sum = float(d_pix.sum())
        tmpl_sums = mu_pix.sum(axis=1)                            # (ncomp,)
        S = np.where(tmpl_sums > 0, tmpl_sums / max(data_sum, 1.0), 1.0)

        mu_pix_norm = mu_pix / S[:, np.newaxis]                   # (ncomp, npix)
        phi_init = theta_init * S                                  # (ncomp,)

        # Bounds in phi space: multiply finite sides by S (sign-preserving)
        scipy_bnds_phi = [
            (None if lo is None else lo * s,
             None if hi is None else hi * s)
            for (lo, hi), s in zip(scipy_bnds_cur, S)
        ]

        vprint(2, f"  Template sums (at theta=1): " +
               ", ".join(f"{lab[:6]}={ts:.2e}" for lab, ts in zip(labels, tmpl_sums)))
        vprint(2, f"  Scale factors S: " +
               ", ".join(f"{lab[:6]}={sv:.2e}" for lab, sv in zip(labels, S)))

        # ---- Optimize (in phi space) ----
        fit_res = fit_global_one_bin(
            d_pix=d_pix,
            mu_pix=mu_pix_norm,
            theta_init=phi_init,
            scipy_bounds=scipy_bnds_phi,
            cexp_floor=float(args.poisson_cexp_floor),
            max_iter=int(args.max_iter),
            error_method=str(args.error_method),
            verbosity=verbosity,
        )

        # ---- Recover physical theta and std ----
        phi_ml  = fit_res["theta_ml"]
        std_phi = fit_res["std_phi"]       # 1-sigma in phi space from selected error method
        theta_ml = phi_ml / S
        # std converts back the same way: sigma_theta = sigma_phi / S
        std_theta = np.where(S > 0, std_phi / S, 0.0)
        cov_phi = np.asarray(fit_res.get("cov_phi", np.zeros((ncomp, ncomp))), dtype=float)
        cov_theta = cov_phi / np.outer(S, S)
        fit_res["theta_ml"] = theta_ml

        if verbosity >= 1:
            print("  Best-fit theta:")
            for lab, th in zip(labels, theta_ml):
                print(f"    {lab:14s}  theta = {th:+.6f}")

        # ---- Residual map ----
        resid_map, model_map, frac_resid = make_residual_map(
            theta_ml=theta_ml,
            mu_k=mu_cube[:, k, :, :].astype(float),
            counts_2d=counts[k].astype(float),
            mask_2d=mask_k,
        )
        positivity = check_model_positivity(
            model_map=model_map,
            mask_2d=mask_k,
            floor=float(args.poisson_cexp_floor),
        )
        if bool(args.check_positivity) and not positivity["positivity_ok"]:
            vprint(
                1,
                "  WARNING: model positivity check failed: "
                f"min={positivity['positivity_min_model']:.3e}, "
                f"<=0 px={positivity['positivity_n_le0']}, "
                f"near floor={positivity['positivity_n_floor']}",
            )

        # ---- Save per-bin results ----
        # Write three files:
        #   mcmc_results_k{k:02d}.npz  — MCMC-compatible .npz so make_figures_from_config.py,
        #                                 make_fig11.py and make_totani_fig8.py work unchanged.
        #   mcmc_results_k{k:02d}.txt  — MCMC-compatible .txt so make_totani_fig9_likelihood.py
        #                                 can read lnL_ml and coefficient tables.
        #   global_fit_k{k:02d}.npz    — extended format with residual maps and scale factors.

        f_ml  = theta_ml.astype(np.float64)
        f_std = std_theta.astype(np.float64)

        lower_theta = np.array(
            [-np.inf if lo is None else float(lo) for lo, _hi in scipy_bnds_cur],
            dtype=float,
        )
        upper_theta = np.array(
            [np.inf if hi is None else float(hi) for _lo, hi in scipy_bnds_cur],
            dtype=float,
        )
        f_p16 = np.maximum(f_ml - f_std, lower_theta)
        f_p84 = np.minimum(f_ml + f_std, upper_theta)

        mcmc_compat = {
            "labels": np.array(labels, dtype=object),
            "template_labels": np.array(template_labels, dtype=object),
            "f_ml":  f_ml,
            "f_p16": f_p16,
            "f_p50": f_ml,
            "f_p84": f_p84,
            "loglike_ml": np.float32(fit_res["loglike_ml"]),
            "energy_bin": np.int32(k),
            "Ectr_mev": np.float32(Ectr_k),
            "used_emcee": np.bool_(False),
            "acceptance_fraction": np.ones(1, dtype=np.float32),
        }
        mcmc_path = os.path.join(args.outdir, f"mcmc_results_k{k:02d}.npz")
        np.savez(mcmc_path, **mcmc_compat)
        vprint(2, f"  Saved (MCMC-compat npz): {mcmc_path}")

        # .txt sidecar — required by make_totani_fig9_likelihood.py.
        # p16/p84 are local 1-sigma estimates from the selected error method.
        txt_path = os.path.join(args.outdir, f"mcmc_results_k{k:02d}.txt")
        with open(txt_path, "w") as _f:
            _f.write(f"Energy bin: k={k}, E_ctr={Ectr_k:.1f} MeV ({Ectr_k/1e3:.3f} GeV)\n")
            _f.write(f"Components: {', '.join(labels)}\n")
            _f.write(f"Global pixel-level MAP fit ({fit_res['error_method']} 1-sigma)\n")
            _f.write(f"Log-likelihood (ML): {fit_res['loglike_ml']:.3f}\n\n")
            _f.write(f"{'Component':<50s}  {'ML':>12s}  {'p16':>12s}  {'p50':>12s}  {'p84':>12s}\n")
            _f.write("-" * 100 + "\n")
            for lab, ml, p16, p84 in zip(labels, f_ml, f_p16, f_p84):
                _f.write(f"{lab:<50s}  {ml:+12.6e}  {p16:+12.6e}  {ml:+12.6e}  {p84:+12.6e}\n")
        vprint(2, f"  Saved (MCMC-compat txt): {txt_path}")

        # Extended file with residual maps and diagnostics
        ext_dict = {
            "k": np.int32(k),
            "Ectr_mev": np.float64(Ectr_k),
            "dE_mev": np.float64(dE_k),
            "labels": np.array(labels, dtype=object),
            "template_labels": np.array(template_labels, dtype=object),
            "theta_ml": f_ml,
            "theta_p16": f_p16,
            "theta_p84": f_p84,
            "theta_std": f_std,
            "theta_cov": cov_theta.astype(np.float64),
            "error_method": np.array(str(fit_res["error_method"]), dtype=object),
            "free_for_error": np.asarray(fit_res["free_for_error"], dtype=bool),
            "loglike_ml": np.float64(fit_res["loglike_ml"]),
            "success": np.bool_(fit_res["success"]),
            "nfev": np.int32(fit_res["nfev"]),
            "roi_lon": np.float64(args.roi_lon),
            "roi_lat": np.float64(args.roi_lat),
            "npix": np.int32(npix_k),
            "template_scale_S": S.astype(np.float64),
            "positivity_ok": np.bool_(positivity["positivity_ok"]),
            "positivity_min_model": np.float64(positivity["positivity_min_model"]),
            "positivity_n_le0": np.int32(positivity["positivity_n_le0"]),
            "positivity_n_floor": np.int32(positivity["positivity_n_floor"]),
        }
        if not args.no_save_maps:
            ext_dict["resid_map"] = resid_map.astype(np.float32)
            ext_dict["model_map"] = model_map.astype(np.float32)
            ext_dict["frac_resid"] = frac_resid.astype(np.float32)
            ext_dict["mask_2d"] = mask_k

        ext_path = os.path.join(args.outdir, f"global_fit_k{k:02d}.npz")
        np.savez(ext_path, **ext_dict)
        vprint(1, f"  Saved: {ext_path}")

        summary_rows.append(
            (k, Ectr_k, theta_ml.copy(), fit_res["loglike_ml"], fit_res["success"])
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if verbosity >= 1 and summary_rows:
        print("\n" + "=" * 60)
        print("Summary: best-fit theta per energy bin")
        print("=" * 60)
        header = f"{'k':>4}  {'Ectr[MeV]':>12}  {'lnL_ML':>14}  {'OK':>4}"
        for j, lab in enumerate(labels):
            header += f"  {lab[:10]:>12}"
        print(header)
        for k, Ectr_k, theta_ml, lnL, ok in summary_rows:
            row = f"{k:>4}  {Ectr_k:>12.1f}  {lnL:>14.2f}  {str(ok):>4}"
            for th in theta_ml:
                row += f"  {th:>+12.4f}"
            print(row)

    vprint(1, f"\nTotal wall time: {time.time() - t_start:.1f} s")
    vprint(1, "Done.")


if __name__ == "__main__":
    main()
