#!/usr/bin/env python3
"""Quantitative residual diagnostics from existing per-bin MCMC summaries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from totani_helpers.totani_io import load_mask_any_shape, read_counts_and_ebounds  # noqa: E402


def _lonlat_from_header(header, shape2d: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape2d
    w = WCS(header).celestial
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = w.all_pix2world(xx, yy, 0)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def _load_template(templates_dir: Path, label: str, k: int) -> np.ndarray:
    path = templates_dir / f"mu_{label}_counts.fits"
    if not path.exists() and not label.endswith("_psf"):
        psf = templates_dir / f"mu_{label}_psf_counts.fits"
        if psf.exists():
            path = psf
    if not path.exists():
        raise FileNotFoundError(path)
    data = fits.getdata(path)
    return np.asarray(data[k] if data.ndim == 3 else data, float)


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a, float)[mask]
    y = np.asarray(b, float)[mask]
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 3:
        return np.nan
    x = x[good] - np.nanmean(x[good])
    y = y[good] - np.nanmean(y[good])
    den = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    return float(np.nansum(x * y) / den) if den > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcmc-dir", type=Path, required=True)
    ap.add_argument("--counts", type=Path, default=REPO / "fermi_data/totani/processed/counts_ccube_1000to1000000.fits")
    ap.add_argument("--templates-dir", type=Path, default=REPO / "fermi_data/totani/processed/templates")
    ap.add_argument("--ext-mask", type=Path, default=REPO / "fermi_data/totani/processed/templates/mask_sources_combined_edep.fits")
    ap.add_argument("--outdir", type=Path, default=PROJECT / "figures/plots_mcmc_residual_diagnostics")
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)
    ap.add_argument("--stat", default="f_p50", choices=["f_p50", "f_ml"])
    args = ap.parse_args()

    counts, header, _, _, ectr, _ = read_counts_and_ebounds(str(args.counts))
    nE, ny, nx = counts.shape
    lon, lat = _lonlat_from_header(header, (ny, nx))
    roi = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat) & (np.abs(lat) >= args.disk_cut)
    if args.ext_mask.exists():
        keep = load_mask_any_shape(str(args.ext_mask), counts.shape).astype(bool)
    else:
        keep = np.ones_like(counts, dtype=bool)

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        "k E_GeV npix resid_mean resid_rms frac_mean frac_rms corr_nfw corr_gas corr_ics corr_ps corr_fb_pos corr_fb_neg"
    ]
    band_rows = ["k E_GeV band npix resid_mean resid_rms frac_rms"]
    rms = np.full(nE, np.nan)
    frac_rms = np.full(nE, np.nan)

    for k in range(nE):
        path = args.mcmc_dir / f"mcmc_results_k{k:02d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as npz:
            labels = [str(x) for x in np.atleast_1d(npz["labels"]).tolist()]
            raw_template_labels = npz["template_labels"] if "template_labels" in npz.files else npz["labels"]
            template_labels = [str(x) for x in np.atleast_1d(raw_template_labels).tolist()]
            coeff = np.asarray(npz[args.stat], float)
        model = np.zeros((ny, nx), float)
        template_maps = {}
        for lab, tlab, c in zip(labels, template_labels, coeff):
            try:
                t = _load_template(args.templates_dir, tlab, k)
            except FileNotFoundError:
                print(f"[warn] k={k} missing template {tlab}; skipping component {lab}")
                continue
            model += float(c) * t
            template_maps[lab] = t
        mask = roi & keep[k] & np.isfinite(model) & np.isfinite(counts[k]) & (model > 0)
        resid = counts[k] - model
        frac = resid / np.where(model > 0, model, np.nan)
        if np.any(mask):
            rms[k] = float(np.sqrt(np.nanmean(resid[mask] ** 2)))
            frac_rms[k] = float(np.sqrt(np.nanmean(frac[mask] ** 2)))
        def tcorr(key: str) -> float:
            cands = [v for lab, v in template_maps.items() if key in lab.lower()]
            return _corr(resid, cands[0], mask) if cands else np.nan
        rows.append(
            f"{k:d} {float(ectr[k])/1000.0:.8g} {int(np.sum(mask))} "
            f"{float(np.nanmean(resid[mask])):.8e} {rms[k]:.8e} "
            f"{float(np.nanmean(frac[mask])):.8e} {frac_rms[k]:.8e} "
            f"{tcorr('nfw'):.8e} {tcorr('gas'):.8e} {tcorr('ics'):.8e} "
            f"{tcorr('ps'):.8e} {tcorr('fb_pos'):.8e} {tcorr('fb_neg'):.8e}"
        )
        for lo, hi in [(10, 20), (20, 40), (40, 60)]:
            bmask = mask & (np.abs(lat) >= lo) & (np.abs(lat) < hi)
            if np.any(bmask):
                band_rows.append(
                    f"{k:d} {float(ectr[k])/1000.0:.8g} b{lo}_{hi} {int(np.sum(bmask))} "
                    f"{float(np.nanmean(resid[bmask])):.8e} "
                    f"{float(np.sqrt(np.nanmean(resid[bmask]**2))):.8e} "
                    f"{float(np.sqrt(np.nanmean(frac[bmask]**2))):.8e}"
                )

    (args.outdir / "residual_metrics_by_energy.txt").write_text("\n".join(rows) + "\n")
    (args.outdir / "residual_metrics_by_latitude_band.txt").write_text("\n".join(band_rows) + "\n")

    try:
        import matplotlib.pyplot as plt
        E = np.asarray(ectr, float) / 1000.0
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(E, rms, "o-", label="counts residual RMS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("E [GeV]")
        ax.set_ylabel("RMS residual [counts/pixel]")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "residual_rms_vs_energy.png", dpi=220)
        fig.savefig(args.outdir / "residual_rms_vs_energy.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(E, frac_rms, "o-", color="C1", label="fractional residual RMS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("E [GeV]")
        ax.set_ylabel("RMS fractional residual")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "fractional_residual_rms_vs_energy.png", dpi=220)
        fig.savefig(args.outdir / "fractional_residual_rms_vs_energy.pdf")
        plt.close(fig)
    except ImportError:
        pass

    print(f"Wrote residual diagnostics to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
