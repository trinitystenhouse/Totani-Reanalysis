#!/usr/bin/env python3
r"""
Model-independent angular profile of the 20 GeV excess vs morphology models.

The single most direct test of "is the excess a centrally-concentrated halo or
an isotropic/IGRB-like residual?". We take the NO-HALO global fit (fig4: gas,
ICS, iso free, ps, Loop I, bubbles) and use its residual map
    excess(n) = data - conventional_model(no halo)
summed over the signal band (default 10-40 GeV). Because the isotropic component
is free in that fit, any residual is the part the conventional+isotropic model
CANNOT absorb -- i.e. the genuinely structured excess. We average it in rings of
angle psi from the Galactic centre over the high-latitude ROI (|l|<=60,
10<=|b|<=60), and compare the ring profile to the line-of-sight predictions of
  NFW rho^2 (annihilation), rho^1 (decay-like), rho^2.5 (cuspier), and isotropic,
each fit to the rings with a single amplitude (+ free offset absorbed by iso).
The model with the lowest chi^2 is the preferred morphology.

If NFW rho^2 fits and isotropic does not, the excess is a centrally-concentrated
halo distinct from IGRB. If isotropic fits as well, the DM-halo morphology is
not supported over a smooth residual.

Usage (from Totani_reanalysis/morphology/):
  python plot_angular_profile_vs_models.py
  python plot_angular_profile_vs_models.py --emin 10 --emax 40 --copy-to <overleaf>/paper_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.parsers.expat import model

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.io import fits

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
_REPO = _REANALYSIS.parent
sys.path.insert(0, str(_REPO))            # helpers.plot_style

from helpers.plot_style import apply_plot_style, current_savefig_kwargs, get_cmap_colors  # noqa: E402

NBINS = 13
COUNTS = _REPO / "fermi_data" / "totani" / "processed" / "counts_ccube_1000to1000000.fits"
# The no-halo residual the PAPER uses. make_paper_results_figures.py builds this
# directory from the production pixelwise chains via make_pixelwise_fig4_resid.py
# and passes it explicitly, so the published Fig. 3 has always been the pixelwise
# residual. This default previously pointed at the DETERMINISTIC quick_global fit,
# which gives a visibly different answer on the same rings (chi2/nu 7.09 / 5.06 /
# 15.39 / 28.80 against the pixelwise 6.96 / 4.93 / 15.22 / 29.10). Every
# morphology script imports this constant, so a standalone run used to disagree
# with the figure it was supposed to reproduce. Keep these in step.
FITDIR = (_REANALYSIS / "mcmc" / "fit_results" / "pixelwise_mcmc"
          / "pixelwise_mcmc_results_fig4_resid")
# Via Lactea II NFW (shape only; normalisation cancels)
RS, R0, RVIR = 21.0, 8.0, 402.0


def _lonlat(header):
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    j = np.arange(ny); i = np.arange(nx)
    b = header["CRVAL2"] + (j + 1 - header["CRPIX2"]) * header["CDELT2"]
    l = header["CRVAL1"] + (i + 1 - header["CRPIX1"]) * header["CDELT1"]
    l = (l + 180.0) % 360.0 - 180.0
    L, B = np.meshgrid(l, b)
    return L, B


def excess_map(fitdir, emin, emax):
    """Sum the no-halo residual over the signal band; return 2D excess + mask."""
    tot = None; mask = None
    for k in range(NBINS):
        p = fitdir / f"global_fit_k{k:02d}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        E = float(z["Ectr_mev"]) / 1000.0
        if not (emin <= E <= emax):
            continue
        r = np.asarray(z["resid_map"], float)
        m = np.asarray(z.get("mask_2d", np.isfinite(r)), bool)
        tot = r.copy() if tot is None else tot + np.where(np.isfinite(r), r, 0.0)
        mask = m if mask is None else (mask & m)
    if tot is None:
        raise SystemExit(f"no fig4 residual maps in band {emin}-{emax} GeV under {fitdir}")
    return tot, mask


def nfw_los(psi_deg, power, smax=RVIR, ns=1500):
    s = np.linspace(1e-3, smax, ns)
    psi = np.deg2rad(psi_deg)
    r = np.sqrt(R0 ** 2 + s ** 2 - 2.0 * R0 * s * np.cos(psi))
    x = r / RS
    rho = x ** (-1.0) * (1.0 + x) ** (-2.0)        # NFW gamma=1 density
    integrate = getattr(np, "trapezoid", np.trapz)
    return integrate(rho ** power, s)


def ring_profile(excess, mask, L, B, edges, roi_lon=60.0, disk_cut=10.0, roi_lat=60.0):
    roi = mask & (np.abs(L) <= roi_lon) & (np.abs(B) >= disk_cut) & (np.abs(B) <= roi_lat)
    psi = np.rad2deg(np.arccos(np.clip(np.cos(np.deg2rad(B)) * np.cos(np.deg2rad(L)), -1, 1)))
    w = np.cos(np.deg2rad(B))                       # solid-angle weight
    ctr = 0.5 * (edges[1:] + edges[:-1])
    val = np.full(ctr.size, np.nan); err = np.full(ctr.size, np.nan)
    for i in range(ctr.size):
        sel = roi & (psi >= edges[i]) & (psi < edges[i + 1])
        n = int(sel.sum())
        if n < 5:
            continue
        ww = w[sel]
        mean = np.sum(ww * excess[sel]) / np.sum(ww)
        val[i] = mean
        err[i] = np.std(excess[sel]) / np.sqrt(n)   # ring scatter / sqrt(N)
    return ctr, val, err


def fit_amp_offset(model, data, err):
    """Least-squares data ~ A*model + C (C absorbs the free-iso zero point)."""
    ok = np.isfinite(data) & np.isfinite(err) & (err > 0) & np.isfinite(model)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    M = np.vstack([model[ok], np.ones(ok.sum())]).T
    W = 1.0 / err[ok] ** 2
    beta, *_ = np.linalg.lstsq(np.sqrt(W)[:, None] * M, np.sqrt(W) * data[ok], rcond=None)
    pred = M @ beta
    chi2 = float(np.sum(((data[ok] - pred) / err[ok]) ** 2))
    return beta[0], beta[1], chi2 / (ok.sum() - 2)

def main(argv=None):
    import shutil
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emin", type=float, default=10.0)
    ap.add_argument("--emax", type=float, default=40.0)
    ap.add_argument("--fitdir", type=Path, default=FITDIR)
    ap.add_argument("--out-dir", type=Path, default=_REANALYSIS / "figures" / "paper_figures")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    hdr = fits.getheader(COUNTS)
    L, B = _lonlat(hdr)
    exc, mask = excess_map(args.fitdir, args.emin, args.emax)

    edges = np.linspace(10.0, 70.0, 13)
    ctr, val, err = ring_profile(exc, mask, L, B, edges)

    models = {r"NFW $\rho^2$": 2.0, r"NFW $\rho^1$": 1.0,
              r"NFW $\rho^{2.5}$": 2.5, "isotropic": None}
    fits_out = {}
    for name, p in models.items():
        m = np.ones_like(ctr) if p is None else np.array([nfw_los(x, p) for x in ctr])
        A, C, chi2n = fit_amp_offset(m, val, err)
        fits_out[name] = (A * m + C, chi2n)

    print(f"excess band {args.emin}-{args.emax} GeV; rings used: {int(np.isfinite(val).sum())}")
    print("morphology fit to the ring profile (lower chi2/dof = preferred):")
    for name, (_, chi2n) in sorted(fits_out.items(), key=lambda kv: kv[1][1]):
        print(f"  {name:16s} chi2/dof = {chi2n:.2f}")

    # ---- plot ----
    import matplotlib.pyplot as plt
    apply_plot_style("paper", base_fontsize=10, linewidth=1.4, n_colors=8, cmap_name="plasma")
    cols = get_cmap_colors(cmap_name="plasma", n=4, start=0.12, end=0.85)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.errorbar(ctr, val, yerr=err, fmt="o", color="k", ms=4, capsize=2, lw=1.2,
                zorder=5, label="excess (data $-$ no-halo model)")
    style = {r"NFW $\rho^2$": "-", r"NFW $\rho^1$": "--",
             r"NFW $\rho^{2.5}$": ":", "isotropic": "-."}
    for (name, p), c in zip(models.items(), cols):
        pred, chi2n = fits_out[name]
        ax.plot(ctr, pred, style[name], color=c, lw=1.8,
                label=fr"{name} ($\chi^2/\nu$={chi2n:.1f})")
    ax.axhline(0.0, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel(r"angle from GC, $\psi$ [deg]")
    ax.set_ylabel(r"excess (data$-$model) [counts, arb.]")
    ax.set_title(fr"Excess morphology, {args.emin:.0f}-{args.emax:.0f} GeV")
    ax.legend(frameon=False)
    ax.grid(alpha=0.13)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        pth = args.out_dir / f"f_angular_profile_morphology.{ext}"
        fig.savefig(pth, dpi=220, **current_savefig_kwargs(bbox_inches="tight"))
        paths.append(pth)
    plt.close(fig)
    print("saved", paths[0])
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[0], args.copy_to / paths[0].name)
        print("copied ->", args.copy_to / paths[0].name)


if __name__ == "__main__":
    main()
