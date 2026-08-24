#!/usr/bin/env python3
"""
Angular-profile morphology test including the hard-IC injected-population templates.

The existing angular-profile test
(`plot_angular_profile_vs_models.py`) asks a model-independent question: taking the
NO-HALO global fit residual, which morphology describes its run with angle from the
Galactic Centre? It compares NFW rho^2 / rho^1 / rho^2.5 against isotropic.

This script adds the hard-IC morphologies to that same contest, so the injected
cosmic-ray hypothesis is tested on the residual directly, without fitting the full
template stack. Method, binning, weighting and the amplitude+offset fit are taken
unchanged from `plot_angular_profile_vs_models.py` so the numbers are directly
comparable; the NFW/isotropic values are recomputed here as a validation that this
script reproduces the published ones.

Usage:
    python morphology/angular_profile_hardic.py
    python morphology/angular_profile_hardic.py --emin 10 --emax 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
_REPO = _REANALYSIS.parent

NBINS = 13
COUNTS = _REPO / "fermi_data" / "totani" / "processed" / "counts_ccube_1000to1000000.fits"
TEMPLATES = _REPO / "fermi_data" / "totani" / "processed" / "templates"
FITDIR = _REANALYSIS / "mcmc" / "fit_results" / "quick_global" / "global_fit_results_fig4"

RS, R0, RVIR = 21.0, 8.0, 402.0
HARDIC_TAGS = ["burst_s", "burst_m", "burst_l", "cont", "adv", "cmbonly_burst_m"]


def _lonlat(header):
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    j = np.arange(ny); i = np.arange(nx)
    b = header["CRVAL2"] + (j + 1 - header["CRPIX2"]) * header["CDELT2"]
    l = header["CRVAL1"] + (i + 1 - header["CRPIX1"]) * header["CDELT1"]
    l = (l + 180.0) % 360.0 - 180.0
    return np.meshgrid(l, b)


def excess_map(fitdir, emin, emax):
    """Sum the no-halo residual over the signal band; return 2D excess, mask, bin list."""
    tot = mask = None
    used = []
    for k in range(NBINS):
        p = fitdir / f"global_fit_k{k:02d}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        E = float(z["Ectr_mev"]) / 1000.0
        if not (emin <= E <= emax):
            continue
        used.append(k)
        r = np.asarray(z["resid_map"], float)
        m = np.asarray(z.get("mask_2d", np.isfinite(r)), bool)
        tot = r.copy() if tot is None else tot + np.where(np.isfinite(r), r, 0.0)
        mask = m if mask is None else (mask & m)
    if tot is None:
        raise SystemExit(f"no residual maps in {emin}-{emax} GeV under {fitdir}")
    return tot, mask, used


def nfw_los(psi_deg, power, smax=RVIR, ns=1500):
    s = np.linspace(1e-3, smax, ns)
    psi = np.deg2rad(psi_deg)
    r = np.sqrt(R0 ** 2 + s ** 2 - 2.0 * R0 * s * np.cos(psi))
    x = r / RS
    rho = x ** (-1.0) * (1.0 + x) ** (-2.0)
    integrate = getattr(np, "trapezoid", np.trapz)
    return integrate(rho ** power, s)


def ring_profile(excess, mask, L, B, edges, roi_lon=60.0, disk_cut=10.0, roi_lat=60.0):
    roi = mask & (np.abs(L) <= roi_lon) & (np.abs(B) >= disk_cut) & (np.abs(B) <= roi_lat)
    psi = np.rad2deg(np.arccos(np.clip(np.cos(np.deg2rad(B)) * np.cos(np.deg2rad(L)), -1, 1)))
    w = np.cos(np.deg2rad(B))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    val = np.full(ctr.size, np.nan); err = np.full(ctr.size, np.nan)
    for i in range(ctr.size):
        sel = roi & (psi >= edges[i]) & (psi < edges[i + 1])
        n = int(sel.sum())
        if n < 5:
            continue
        ww = w[sel]
        val[i] = np.sum(ww * excess[sel]) / np.sum(ww)
        err[i] = np.std(excess[sel]) / np.sqrt(n)
    return ctr, val, err


def fit_amp_offset(model, data, err):
    """Least-squares data ~ A*model + C (C absorbs the free-iso zero point)."""
    ok = np.isfinite(data) & np.isfinite(err) & (err > 0) & np.isfinite(model)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    M = np.vstack([model[ok], np.ones(ok.sum())]).T
    W = 1.0 / err[ok] ** 2
    beta, *_ = np.linalg.lstsq(np.sqrt(W)[:, None] * M, np.sqrt(W) * data[ok], rcond=None)
    chi2 = float(np.sum(((data[ok] - M @ beta) / err[ok]) ** 2))
    return beta[0], beta[1], chi2 / (ok.sum() - 2)


def hardic_band_map(tag, bins, psf=True):
    """Band-summed hard-IC template map; None if the file is absent."""
    name = f"mu_hardic_{tag}_{'psf_' if psf else ''}counts.fits"
    path = TEMPLATES / name
    if not path.exists():
        return None
    with fits.open(path) as h:
        cube = np.asarray(h[0].data, float)
    if cube.ndim != 3:
        raise SystemExit(f"{name}: expected a 3D cube, got shape {cube.shape}")
    return np.nansum(cube[bins, :, :], axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emin", type=float, default=10.0)
    ap.add_argument("--emax", type=float, default=40.0)
    ap.add_argument("--fitdir", type=Path, default=FITDIR)
    ap.add_argument("--no-psf", action="store_true",
                    help="use the un-convolved template cube instead of the PSF-folded one")
    ap.add_argument("--outfile", type=Path,
                    default=_REANALYSIS / "results" / "posterior-covariance"
                    / "angular_profile_hardic.txt")
    args = ap.parse_args(argv)

    hdr = fits.getheader(COUNTS)
    L, B = _lonlat(hdr)
    exc, mask, bins = excess_map(args.fitdir, args.emin, args.emax)

    edges = np.linspace(10.0, 70.0, 13)
    ctr, val, err = ring_profile(exc, mask, L, B, edges)
    n_rings = int(np.isfinite(val).sum())

    results = {}

    # --- reference morphologies (validation against the published values) ---
    for name, p in {"NFW rho^2": 2.0, "NFW rho^1": 1.0,
                    "NFW rho^2.5": 2.5, "isotropic": None}.items():
        m = np.ones_like(ctr) if p is None else np.array([nfw_los(x, p) for x in ctr])
        results[name] = fit_amp_offset(m, val, err)[2]

    # --- hard-IC morphologies ---
    missing = []
    for tag in HARDIC_TAGS:
        band = hardic_band_map(tag, bins, psf=not args.no_psf)
        if band is None:
            missing.append(tag)
            continue
        _, mv, _ = ring_profile(band, mask, L, B, edges)
        results[f"hard-IC {tag}"] = fit_amp_offset(mv, val, err)[2]

    lines = []
    lines.append(f"# Angular-profile morphology contest, {args.emin:.0f}-{args.emax:.0f} GeV")
    lines.append(f"# no-halo residual from {args.fitdir}")
    lines.append(f"# band bins used: {bins};  rings with data: {n_rings}/{ctr.size}")
    lines.append(f"# hard-IC templates: {'PSF-folded' if not args.no_psf else 'un-convolved'}")
    lines.append("# amplitude and offset free for every model; lower chi2/dof = preferred")
    lines.append("")
    lines.append(f"{'morphology':<22s} {'chi2/dof':>10s}")
    lines.append("-" * 34)
    for name, c in sorted(results.items(), key=lambda kv: (np.inf if not np.isfinite(kv[1]) else kv[1])):
        lines.append(f"{name:<22s} {c:>10.2f}")
    if missing:
        lines.append("")
        lines.append(f"# templates not found (build them first): {', '.join(missing)}")

    out = "\n".join(lines)
    print(out)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text(out + "\n")
    print(f"\nwrote -> {args.outfile}")


if __name__ == "__main__":
    main()
