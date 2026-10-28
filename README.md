# Totani-Reanalysis

Code that reproduces the results of **"The 20 GeV Galactic Halo Excess:
Pixel-Level Confirmation and Consistency with Sub-TeV WIMP Annihilation"**
(T. R. Stenhouse, C. Ghag, F. F. Deppisch,
[arXiv:2607.08552](https://arxiv.org/abs/2607.08552)), together with a set of
further robustness tests of the excess.

| | |
|---|---|
| Paper | [arXiv:2607.08552](https://arxiv.org/abs/2607.08552) |
| Code archive | [doi:10.5281/zenodo.21280724](https://doi.org/10.5281/zenodo.21280724) |
| Reproducing a figure or number | [REPRODUCE.md](REPRODUCE.md) |
| Data | *Fermi*-LAT Pass 8, fetched as described in [docs/DATA_FETCH.md](docs/DATA_FETCH.md) |
| Licence | MIT |

## Background

T. Totani reported a halo-like excess peaking near 20 GeV in fifteen years of
*Fermi*-LAT diffuse emission, with a morphology resembling dark matter
annihilation in the Galactic halo
([arXiv:2507.07209](https://arxiv.org/abs/2507.07209)). This repository tests
that claim independently. It first replicates Totani's own cell-based fit, then
repeats the measurement with a pixel-level Poisson likelihood, interprets the
result as dark matter and confronts it with dwarf-spheroidal limits, and finally
asks, test by test, whether a conventional explanation or an analysis choice
could produce the same feature.

## What is in this repository

### Reproducing the paper

| Step | What it does | Where |
|---|---|---|
| Replication | Totani's cell-based fit, with the same event selection and template set | `Totani_paper_check/` |
| Pixel-level fit | Poisson likelihood over 750,384 pixels at 0.125 deg, 13 energy bins from 1.16 GeV to 1 TeV, with PSF forward-folding and a bright-source mask | `Totani_reanalysis/mcmc/` |
| Dark matter fit | Annihilation mass and cross section for b-bbar, W+W- and tau+tau- on one fixed mass grid, and a decaying dark matter alternative | `Totani_reanalysis/run_pppc_mass_scan.py`, `Totani_reanalysis/decay/` |
| Dwarf comparison | The halo cross section against the Hoof et al. stacked limits ([arXiv:1812.06986](https://arxiv.org/abs/1812.06986)) and the McDaniel et al. per-dwarf likelihoods ([arXiv:2311.04982](https://arxiv.org/abs/2311.04982)) | `Totani_reanalysis/core/dsph_limits*.py`, `dSphs/` |
| Velocity dependence | How p-wave, d-wave and enhanced annihilation change the halo-to-dwarf ratio | `Totani_reanalysis/velocity_dep/` |
| Figures | Every paper figure from one script | `Totani_reanalysis/make_paper_results_figures.py` |

### Robustness tests

Each test asks one question of the excess.

| Test | Question | Where |
|---|---|---|
| Diffuse-model variants | Does the halo depend on the choice of foreground model (no Loop I, flat bubbles, GIEM only, steep or flat inverse-Compton morphology)? | `mcmc/write_global_systematics_configs.py`, `systematics/propagate_systematics.py` |
| Fourteen-model GALPROP grid | Does it survive across the Ackermann et al. (2012) interstellar-emission grid, with the Fermi bubbles re-extracted against each model? | `systematics/rebuild_bubbles_per_variant.py`, `systematics/total_systematic_band.py` |
| Injected electron population | Can a burst of cosmic-ray electrons at the Galactic centre, radiating by inverse Compton, mimic the halo in spectrum and morphology? | `systematics/run_hardic_bubble_reextraction.py`, `systematics/hardic_report.py`, `morphology/angular_profile_hardic.py` |
| Mismodelled-foreground null | If the true sky has a different inverse-Compton morphology and no halo, does the fit invent one? | `mcmc/build_diffuse_mismodel_mock.py`, `check_null_and_hemisphere.py` |
| Sky splits | Is the halo spectrum consistent between hemispheres and quadrants? | `check_null_and_hemisphere.py`, `compare_quadrants.py` |
| Disk inclusion | Does adding the Galactic plane change the result? | `compare_pixelwise_disk_in_out.py`, `compare_dlnl_disk_in_out.py` |
| Template covariance | How strongly is the halo correlated with each other component in the posterior? | `systematics/posterior_covariance.py` |
| Low-energy shoulder | Where on the sky does the 1.5-4.4 GeV emission sit, and does excluding it move the dark matter mass? | `systematics/shoulder_by_region.py`, `systematics/refit_mass_energy_range.py` |
| Morphology by energy | Does the angular profile of the excess change across the signal band? | `morphology/angular_profile_by_energy.py` |
| Positivity | Is the best-fit model positive in every occupied pixel? | `systematics/positivity_occupied_check.py`, `mcmc/check_global_positivity.py` |

Paths in this table are relative to `Totani_reanalysis/`.

## Main results

The pixel-level fit confirms an NFW²-like halo excess independently. Adding the
halo improves the fit by $2\sum_k\Delta\ln\mathcal{L}_k \simeq 1.7\times10^{3}$
across the 13 energy bins, about 930 of it in the 10-40 GeV signal band. This is
a template-level diagnostic of fit improvement, not a significance in sigma,
because the conditions for a chi-squared-distributed likelihood ratio are not
met here.

Read as dark matter annihilation, the excess needs a cross section a few times
above the stacked dwarf limit. The ratio of the two is R_bb = 5.1 (1.6 to 9.8)
and R_WW = 4.1 (1.2 to 7.7) across the full systematic budget. The repository
computes these numbers and does not resolve the tension. The result is framed
as consistency with dark matter annihilation, not as a detection.

## Install

Python 3.10 or later:

```bash
pip install -r requirements.txt
pip install -e .
```

Installing in editable mode exposes `core`, `shared`, `helpers` and
`totani_helpers` as top-level importable packages regardless of your working
directory (see `pyproject.toml`).

```bash
python -m pytest tests/
```

The smoke tests need no external data — they check that the repo's own
layout resolves and that the analysis modules import.

## Repository layout

```
Totani_reanalysis/        pixel-level reanalysis (arXiv:2607.08552)
  core/                      physics library: cross sections, transport,
                             halo/IGRB spectrum loaders, dSph limits
  mcmc/                      global (single-likelihood) pixel-level fit code
  systematics/               Fermi-bubble re-extraction, diffuse-mismodel checks,
                             the fourteen-model GALPROP IEM grid, low-energy
                             shoulder localisation and mass-refit diagnostics
  velocity_dep/               velocity-dependent cross section / dSph tension study
  decay/                      decaying-DM alternative fit
  morphology/                 angular-profile diagnostics, including the
                             per-energy-bin inner-power scan
  figures/                    per-figure plotting scripts
  data/                       PPPC4DMID table, Boddy/Hoof reference limits
  *.py                        top-level analysis/audit scripts

Totani_paper_check/        cellwise replication of Totani (2025)
  data_download/              Fermi-LAT query + gtselect/gtbin/gtexpcube2 pipeline
  make_templates/             counts-space template construction (gas, ICS, NFW, ...)
  check_templates/            per-template validation plots
  mcmc/                       cellwise MCMC fit code
  sanity_checks/               counts/exposure sanity plots

dSphs/                      McDaniel et al. 2024 per-dwarf SED likelihoods / TS profiles
shared/                     path resolution (paths.py) + totani_helpers/ (shared
                            FITS I/O, likelihood fitting, MCMC coefficient loading)
helpers/                    shared plotting style (plot_style.py) and utilities
docs/                       DATA_FETCH.md — exact Fermi-LAT selection and processing
tests/                      smoke tests
```

## Regenerating the posterior and figures

See **[REPRODUCE.md](REPRODUCE.md)** for the exact commands, and
**[docs/DATA_FETCH.md](docs/DATA_FETCH.md)** for the *Fermi*-LAT Pass 8 event
selection, time range and IRF version needed to fetch and process the raw
data this repository does not ship.

### What's committed vs. what you regenerate

This repository ships the analysis code, the digitised reference limit
tables, and the per-dwarf SED/TS-profile data. It does **not** ship the raw
*Fermi*-LAT event files, the intermediate counts/exposure/template FITS
cubes, or any fit output — the MCMC posteriors, the derived posterior
summaries and covariances, and the rendered figures are all regenerated from
the code and the data fetch described above. `REPRODUCE.md` gives the
commands and the expected runtimes.

## Citation

> T. R. Stenhouse, C. Ghag and F. F. Deppisch, "The 20 GeV Galactic Halo
> Excess: Pixel-Level Confirmation and Consistency with Sub-TeV WIMP
> Annihilation," [arXiv:2607.08552](https://arxiv.org/abs/2607.08552) (2026).

To cite the code itself, use the Zenodo archive,
[doi:10.5281/zenodo.21280724](https://doi.org/10.5281/zenodo.21280724).
`CITATION.cff` carries both, and GitHub's "Cite this repository" button
reads it.

If you use the cellwise replication independently, please also cite the
original result:

> T. Totani, "20 GeV halo-like excess of the Galactic diffuse emission and
> implications for dark matter annihilation,"
> [arXiv:2507.07209](https://arxiv.org/abs/2507.07209) (2025).

## License

MIT — see `LICENSE`.
