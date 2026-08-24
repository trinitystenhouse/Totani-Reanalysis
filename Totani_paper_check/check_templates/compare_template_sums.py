#!/usr/bin/env python3
"""
Compare sum(mu_nfw_rho2[k=5]) and sum(mu_fb_pos[k=5]) over the ROI.
"""

import argparse
import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from totani_helpers.totani_io import lonlat_grids

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def main():
    ap = argparse.ArgumentParser(
        description="Compare template sums for NFW-rho2 and FB_pos at k=5"
    )
    ap.add_argument(
        "--templates-dir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
        help="Directory containing template FITS files",
    )
    ap.add_argument(
        "--nfw-template",
        default="mu_nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno_counts.fits",
        help="NFW rho2 template filename",
    )
    ap.add_argument(
        "--fb-template",
        default="mu_fb_pos_counts.fits",
        help="FB pos template filename",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="Energy bin index to analyze",
    )
    ap.add_argument(
        "--roi-lon",
        type=float,
        default=60.0,
        help="ROI half-width in longitude [deg]",
    )
    ap.add_argument(
        "--roi-lat",
        type=float,
        default=60.0,
        help="ROI half-width in latitude [deg]",
    )
    args = ap.parse_args()

    # Load NFW template
    nfw_path = os.path.join(args.templates_dir, args.nfw_template)
    if not os.path.exists(nfw_path):
        raise FileNotFoundError(f"NFW template not found: {nfw_path}")
    
    with fits.open(nfw_path) as hdul:
        mu_nfw_rho2 = hdul[0].data
        hdr = hdul[0].header
    
    # Load FB_pos template
    fb_path = os.path.join(args.templates_dir, args.fb_template)
    if not os.path.exists(fb_path):
        raise FileNotFoundError(f"FB_pos template not found: {fb_path}")
    
    with fits.open(fb_path) as hdul:
        mu_fb_pos = hdul[0].data
    
    # Get WCS and create lon/lat grids
    wcs = WCS(hdr).celestial
    nE, ny, nx = mu_nfw_rho2.shape
    lon, lat = lonlat_grids(wcs, ny, nx)
    
    # Define ROI mask
    roi2d = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    
    # Extract k-th energy bin
    k = args.k
    if k >= nE:
        raise ValueError(f"Energy bin k={k} out of range (nE={nE})")
    
    nfw_k = mu_nfw_rho2[k]
    fb_k = mu_fb_pos[k]
    
    # Apply ROI mask and compute sums
    nfw_roi = nfw_k[roi2d]
    fb_roi = fb_k[roi2d]
    
    sum_nfw = np.nansum(nfw_roi)
    sum_fb = np.nansum(fb_roi)
    
    # Compute ratio
    if sum_fb != 0:
        ratio = sum_nfw / sum_fb
    else:
        ratio = np.inf
    
    # Print results
    print("=" * 70)
    print(f"Template Sum Comparison at Energy Bin k={k}")
    print("=" * 70)
    print(f"NFW template:    {args.nfw_template}")
    print(f"FB_pos template: {args.fb_template}")
    print(f"ROI: |lon| <= {args.roi_lon}°, |lat| <= {args.roi_lat}°")
    print(f"Number of pixels in ROI: {np.sum(roi2d)}")
    print("-" * 70)
    print(f"sum(mu_nfw_rho2[k={k}]) over ROI:  {sum_nfw:.6e}")
    print(f"sum(mu_fb_pos[k={k}]) over ROI:    {sum_fb:.6e}")
    print("-" * 70)
    print(f"Ratio (NFW/FB_pos):                {ratio:.6f}")
    print("=" * 70)
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"NFW - mean:   {np.nanmean(nfw_roi):.6e}")
    print(f"NFW - median: {np.nanmedian(nfw_roi):.6e}")
    print(f"NFW - std:    {np.nanstd(nfw_roi):.6e}")
    print(f"NFW - min:    {np.nanmin(nfw_roi):.6e}")
    print(f"NFW - max:    {np.nanmax(nfw_roi):.6e}")
    print()
    print(f"FB_pos - mean:   {np.nanmean(fb_roi):.6e}")
    print(f"FB_pos - median: {np.nanmedian(fb_roi):.6e}")
    print(f"FB_pos - std:    {np.nanstd(fb_roi):.6e}")
    print(f"FB_pos - min:    {np.nanmin(fb_roi):.6e}")
    print(f"FB_pos - max:    {np.nanmax(fb_roi):.6e}")


if __name__ == "__main__":
    main()
