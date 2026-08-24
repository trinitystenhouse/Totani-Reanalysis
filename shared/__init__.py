"""Shared infrastructure for the Totani 20 GeV halo reanalysis.

This package centralises code and data-path configuration used by the two
Totani-related analyses in this repo:
  - Totani_paper_check  (cellwise replication of Totani's published result,
                          arXiv:2507.07209)
  - Totani_reanalysis   (the pixel-level reanalysis behind arXiv:2607.08552)

Contents:
  - paths            : single source of truth for repo/data locations
  - totani_helpers/  : IO + MCMC helpers shared by both analyses
  - (core/            : dsph_limits_multi, spectral_reshaping,
                        totani_data_loader — local to Totani_reanalysis/core/)
  - plot_style       : the shared paper plot style (helpers/)

Install once for editable, import-from-anywhere use, from the repo root:
    pip install -e .
"""
from . import paths  # noqa: F401
