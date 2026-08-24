#!/usr/bin/env python3
"""
psf_utils.py
============
Energy-dependent PSF convolution for the GLOBAL (pixel-level) Fermi-LAT fit.

Why this exists
---------------
Totani's published analysis bins the sky into 10 deg cells, which washes out the
LAT point-spread function, so he does NOT convolve the NFW/halo template with the
PSF. That is correct for a 10 deg cellwise fit. The local *global* fit in this repo
works at 0.125 deg pixels, where the PSF is NOT negligible: a cuspy rho^2 / rho^2.5
template (and the diffuse templates) are sharper than the PSF-smeared data they are
fit to. Leaving them un-convolved biases the inner-region morphology and lets a
free-sign halo absorb the mismatch.

This module forward-folds any counts template cube (nE, ny, nx) through an
energy-dependent PSF, conserving counts per energy plane. It is intended to be run
ONCE as a preprocessing step that produces new `mu_<label>_psf_counts.fits` files;
the originals are left untouched so the Totani pipeline keeps using them.

PSF model
---------
Default is a single King (Moffat) profile with an energy-dependent 68% containment
radius R68(E) approximating Fermi-LAT P8R3_SOURCE (front+back combined):

    R68(E) = sqrt( (a * (E/GeV)^(-b))^2 + c^2 )      [deg]

with a=0.8, b=0.8, c=0.1 by default (~0.8 deg at 1 GeV, ~0.16 deg at 10 GeV,
floor ~0.1 deg at high E). This is a faithful *approximation*. For a fully rigorous
result, pass `--psf-fits` pointing at a `gtpsf` output (not required for the
robustness improvement this provides over the current Gaussian PS smoothing).

The King shape captures the broad wings (heavier than a Gaussian), which is exactly
what a Gaussian PS smoothing misses and what matters for source-to-diffuse leakage.

CLI
---
    # Convolve one NFW template against the P8R3-like King PSF
    python psf_utils.py \
        --in  /path/templates/mu_nfw_NFW_g1_rho2_..._counts.fits \
        --counts /path/processed/counts_ccube_1000to1000000.fits \
        --out /path/templates/mu_nfw_NFW_g1_rho2_..._psf_counts.fits

    # Gaussian instead of King (to reproduce the old PS-smoothing behaviour)
    python psf_utils.py --in ... --counts ... --kind gauss

Importable API
--------------
    convolve_cube_with_psf(cube, Ectr_mev, binsz_deg, kind="king", gamma=2.5)
    convolve_template_file(in_fits, counts_fits, out_fits=None, ...)
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Optional

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Energy-dependent 68% containment radius (P8R3_SOURCE-like approximation)
# ---------------------------------------------------------------------------

def lat_r68_deg(E_mev: np.ndarray, a: float = 0.8, b: float = 0.8, c: float = 0.1) -> np.ndarray:
    """
    Approximate Fermi-LAT P8R3 (front+back) 68% containment radius in degrees.

        R68(E) = sqrt( (a * (E/GeV)^(-b))^2 + c^2 )

    Defaults give ~0.8 deg @ 1 GeV, ~0.16 deg @ 10 GeV, ~0.1 deg floor at high E.
    """
    E_gev = np.asarray(E_mev, float) / 1000.0
    return np.sqrt((a * E_gev ** (-b)) ** 2 + c ** 2)


# ---------------------------------------------------------------------------
# Kernel builders (normalised to sum = 1)
# ---------------------------------------------------------------------------

def _king_sigma_from_r68(r68_deg: float, gamma: float) -> float:
    """
    Closed-form sigma for a single King profile given R68 and gamma.

    King enclosed fraction:  F(<R) = 1 - [1 + (1/(2 gamma)) (R/sigma)^2]^(1-gamma)
    Setting F(R68) = 0.68 and solving:
        sigma = R68 / sqrt( 2 gamma (0.32^(1/(1-gamma)) - 1) )
    """
    denom = 2.0 * gamma * (0.32 ** (1.0 / (1.0 - gamma)) - 1.0)
    return float(r68_deg) / float(np.sqrt(denom))


def king_kernel_2d(r68_deg: float, binsz_deg: float, gamma: float = 2.5,
                   rmax_factor: float = 6.0) -> np.ndarray:
    """
    Normalised 2D King kernel on the pixel grid.

    Profile (up to normalisation): [1 + (r/sigma)^2 / (2 gamma)]^(-gamma)
    """
    sigma = _king_sigma_from_r68(r68_deg, gamma)
    half = max(1, int(np.ceil(rmax_factor * r68_deg / binsz_deg)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    r_deg = np.hypot(xx, yy) * binsz_deg
    k = (1.0 + (r_deg / sigma) ** 2 / (2.0 * gamma)) ** (-gamma)
    s = k.sum()
    return k / s if s > 0 else k


def gaussian_kernel_2d(r68_deg: float, binsz_deg: float, rmax_factor: float = 5.0) -> np.ndarray:
    """
    Normalised 2D Gaussian kernel. For a 2D Gaussian, R68 = 1.5096 * sigma.
    (Provided for comparison / to reproduce the old PS Gaussian smoothing.)
    """
    sigma = float(r68_deg) / 1.5095921854516636
    half = max(1, int(np.ceil(rmax_factor * max(r68_deg, sigma) / binsz_deg)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    r2 = (np.hypot(xx, yy) * binsz_deg) ** 2
    k = np.exp(-0.5 * r2 / sigma ** 2)
    s = k.sum()
    return k / s if s > 0 else k


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

def _convolve_plane(plane: np.ndarray, kernel: np.ndarray, conserve: bool = True) -> np.ndarray:
    """FFT-convolve a single 2D plane; optionally rescale to conserve total counts."""
    src = np.nan_to_num(np.asarray(plane, float), nan=0.0, posinf=0.0, neginf=0.0)
    pre = float(src.sum())
    out = fftconvolve(src, kernel, mode="same")
    out[out < 0.0] = 0.0  # clip tiny negative FFT ringing
    if conserve and pre > 0.0:
        post = float(out.sum())
        if post > 0.0:
            out *= pre / post
    return out.astype(np.float32)


def convolve_cube_with_psf(
    cube: np.ndarray,
    Ectr_mev: np.ndarray,
    binsz_deg: float,
    *,
    kind: str = "king",
    gamma: float = 2.5,
    rmax_factor: float = 6.0,
    r68_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    conserve: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convolve a counts cube (nE, ny, nx) with an energy-dependent PSF.

    Parameters
    ----------
    cube       : (nE, ny, nx) counts template.
    Ectr_mev   : (nE,) geometric bin-centre energies in MeV.
    binsz_deg  : pixel size in degrees.
    kind       : "king" (default) or "gauss".
    gamma      : King index (ignored for gauss). 2.5 is a reasonable LAT-like value.
    r68_func   : optional callable E_mev -> R68_deg. Defaults to lat_r68_deg.
    conserve   : rescale each plane to conserve total counts (default True).
    """
    cube = np.asarray(cube, float)
    if cube.ndim != 3:
        raise ValueError(f"cube must be (nE, ny, nx); got {cube.shape}")
    Ectr_mev = np.asarray(Ectr_mev, float).reshape(-1)
    if Ectr_mev.size != cube.shape[0]:
        raise ValueError(f"Ectr_mev has {Ectr_mev.size} entries, cube has {cube.shape[0]} planes.")

    r68_func = r68_func or lat_r68_deg
    out = np.zeros_like(cube, dtype=np.float32)
    for k in range(cube.shape[0]):
        r68 = float(r68_func(np.array([Ectr_mev[k]]))[0])
        if kind.lower() == "king":
            kern = king_kernel_2d(r68, binsz_deg, gamma=gamma, rmax_factor=rmax_factor)
        elif kind.lower() in ("gauss", "gaussian"):
            kern = gaussian_kernel_2d(r68, binsz_deg, rmax_factor=max(rmax_factor, 5.0))
        else:
            raise ValueError("kind must be 'king' or 'gauss'")
        out[k] = _convolve_plane(cube[k], kern, conserve=conserve)
        if verbose:
            print(f"  bin {k:02d}: E={Ectr_mev[k]/1e3:8.3f} GeV  R68={r68:.3f} deg  "
                  f"kernel={kern.shape[0]}x{kern.shape[1]}")
    return out


# ---------------------------------------------------------------------------
# FITS helpers (self-contained; compatible with this repo's counts/templates)
# ---------------------------------------------------------------------------

def _read_ectr_mev_and_binsz(counts_fits: str) -> tuple[np.ndarray, float]:
    """Read geometric bin-centre energies (MeV) and pixel size (deg) from a counts CCUBE."""
    with fits.open(counts_fits) as h:
        hdr = h[0].header
        eb = h["EBOUNDS"].data
        # This repo's EBOUNDS E_MIN/E_MAX are in keV; /1000 -> MeV (matches totani_io).
        Emin_mev = np.asarray(eb["E_MIN"], float) / 1000.0
        Emax_mev = np.asarray(eb["E_MAX"], float) / 1000.0
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    binsz = abs(float(hdr.get("CDELT2", hdr.get("CDELT1", 0.125))))
    if binsz <= 0:
        binsz = 0.125
    return Ectr_mev, binsz


def convolve_template_file(
    in_fits: str,
    counts_fits: str,
    out_fits: Optional[str] = None,
    *,
    kind: str = "king",
    gamma: float = 2.5,
    rmax_factor: float = 6.0,
    binsz_deg: Optional[float] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> str:
    """
    Read a `mu_<label>_counts.fits` template, PSF-convolve it, write a new FITS.

    If out_fits is None, the output path is the input with `_counts.fits` replaced
    by `_psf_counts.fits`.  Returns the output path.
    """
    Ectr_mev, binsz_from_hdr = _read_ectr_mev_and_binsz(counts_fits)
    binsz = float(binsz_deg) if binsz_deg else binsz_from_hdr

    with fits.open(in_fits) as h:
        cube = np.asarray(h[0].data, float)
        hdr = h[0].header.copy()

    if cube.shape[0] != Ectr_mev.size:
        raise ValueError(
            f"Template {os.path.basename(in_fits)} has {cube.shape[0]} energy planes but "
            f"counts file has {Ectr_mev.size}. They must share the energy grid."
        )

    if verbose:
        print(f"Convolving {os.path.basename(in_fits)}  (kind={kind}, gamma={gamma}, binsz={binsz} deg)")
    conv = convolve_cube_with_psf(
        cube, Ectr_mev, binsz, kind=kind, gamma=gamma, rmax_factor=rmax_factor, verbose=verbose
    )

    if out_fits is None:
        if in_fits.endswith("_counts.fits"):
            out_fits = in_fits[: -len("_counts.fits")] + "_psf_counts.fits"
        else:
            base, ext = os.path.splitext(in_fits)
            out_fits = f"{base}_psf{ext}"

    hdr["PSFCONV"] = (True, "PSF-convolved for pixel-level global fit")
    hdr["PSFKIND"] = (str(kind), "PSF kernel type")
    hdr["PSFGAMMA"] = (float(gamma), "King index (if kind=king)")
    hdr["PSFSRC"] = (os.path.basename(in_fits), "Un-convolved source template")
    fits.PrimaryHDU(conv.astype("f4"), header=hdr).writeto(out_fits, overwrite=overwrite)
    if verbose:
        print(f"  -> wrote {out_fits}")
    return out_fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="PSF-convolve a Fermi-LAT counts template cube for the pixel-level global fit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--in", dest="in_fits", required=True, help="Input mu_<label>_counts.fits")
    ap.add_argument("--counts", required=True, help="Counts CCUBE (for EBOUNDS energies + pixel size)")
    ap.add_argument("--out", default=None, help="Output FITS (default: *_psf_counts.fits)")
    ap.add_argument("--kind", choices=["king", "gauss"], default="king")
    ap.add_argument("--gamma", type=float, default=2.5, help="King index (LAT-like ~2.5)")
    ap.add_argument("--rmax-factor", type=float, default=6.0, help="Kernel half-width in units of R68")
    ap.add_argument("--binsz", type=float, default=None, help="Override pixel size (deg)")
    args = ap.parse_args()

    convolve_template_file(
        args.in_fits, args.counts, args.out,
        kind=args.kind, gamma=args.gamma, rmax_factor=args.rmax_factor, binsz_deg=args.binsz,
    )


if __name__ == "__main__":
    main()