#!/bin/bash
# Abort on the first failing stage. Without this the bubble stages in
# particular fail silently and later stages consume missing/stale inputs:
# Stage 2 reads Stage 1's mu_fb_flat_counts.fits, and the NFW/GIEM steps read
# the masks built at the top. A partial template set is worse than none,
# because every downstream figure then plots without complaint.
set -euo pipefail

# None of the build_*.py steps create their own output directory, and
# fermi_data/ is gitignored, so on a fresh checkout every stage dies on the
# final writeto() after doing all of its work.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${REPO_ROOT}/fermi_data/totani/processed/templates"

TOTANI_DEBUG_ENERGY=1 python mask_extended_sources.py
TOTANI_DEBUG_ENERGY=1 python build_ps_template.py
TOTANI_DEBUG_ENERGY=1 python build_gas_template.py
TOTANI_DEBUG_ENERGY=1 python build_ics_template.py
TOTANI_DEBUG_ENERGY=1 python build_iso_template.py
TOTANI_DEBUG_ENERGY=1 python build_loopI_template.py
TOTANI_DEBUG_ENERGY=1 python build_nfw_template.py

# ============================================================================
# FERMI BUBBLES: Two-stage construction (Totani Sec 3.1)
# ============================================================================
# Stage 1: Build flat bubble template from vertex-defined boundaries.
#   Uses NFW-rho^2.5 (GCE) to account for GC excess. Fit includes disk.
#   Creates mu_fb_flat_counts.fits.
# Stage 2: Build structured pos/neg templates.
#   Fit all components INCLUDING fb_flat at 4.3 GeV (k=2).
#   Bubble image = fb_flat_contribution + fit_residuals = data - non-bubble components.
#   Convert to flux, smooth with sigma=1deg, split by sign across full ROI.
#   This reproduces Totani's "residual map (including flat_fb) smoothed at 1-degree scale".
# ============================================================================

# Stage 1: flat bubble template (NFW-rho^2.5 required for GC excess, fit includes disk)
# NOTE: this line was missing its `python` (and the .py is not executable), so
# Stage 1 died with "command not found" -- and without set -e the run carried on
# into Stage 2, which needs Stage 1's mu_fb_flat_counts.fits.
TOTANI_DEBUG_ENERGY=1 python build_bubbles_templates.py \
  --build-flat-only \
  --restrict-to-vertices \
  --disk-cut-deg 0 \
  --components gas ics iso ps loopA loopB nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno

# Stage 2: structured pos/neg templates (Totani method)
# Uses cell-level Poisson MCMC (same as Stage 1) to fit background components,
# which correctly constrains the isotropic component (pixel NNLS sets iso→0).
# --use-mcmc: run 10°×10° cell Poisson MCMC per energy bin
# --bubbles-split-mode totani: spatial morphology (pos/neg shape) is FIXED from k2 (4.3 GeV)
#   and scaled in amplitude to each energy bin's total bubble counts. This is the correct
#   Totani method — the spatial boundary between pos/neg is determined at 4.3 GeV only,
#   preventing the bubble templates from absorbing spectral structure (e.g. the 20 GeV halo)
#   at other energy bins. Confirmed by direct correspondence with Totani.
python build_bubbles_templates.py \
  --use-mcmc \
  --force-bubbles-close-residual \
  --bubbles-split-mode totani \
  --mcmc-nwalkers 64 \
  --mcmc-nsteps 300000 \
  --mcmc-burn 50000 \
  --mcmc-thin 10 \
  --disk-cut-deg 10 \
  --components gas ics iso ps loopA loopB \
      nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno fb_flat
      
TOTANI_DEBUG_ENERGY=1 python build_nfw_template.py --rho-power 2
TOTANI_DEBUG_ENERGY=1 python build_nfw_template.py --rho-power 1
