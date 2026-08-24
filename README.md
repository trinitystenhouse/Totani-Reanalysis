# Totani_Reanalysis_public

Analysis code for **"The 20 GeV Galactic Halo Excess: Pixel-Level
Confirmation and Consistency with Sub-TeV WIMP Annihilation"**
(T. R. Stenhouse, C. Ghag, F. F. Deppisch),
[arXiv:2607.08552](https://arxiv.org/abs/2607.08552).

It reanalyses the result reported in **T. Totani**, "20 GeV halo-like excess
of the Galactic diffuse emission and implications for dark matter
annihilation", [arXiv:2507.07209](https://arxiv.org/abs/2507.07209).

## What this repository does

Totani reported a ~20 GeV excess in *Fermi*-LAT data toward the
Galactic centre with a spatial profile resembling an NFW-squared dark-matter
annihilation signal. This repository:

1. **Replicates Totani's published cellwise (per-sky-cell) fit**
   (`Totani_paper_check/`) as an independent cross-check of the original
   claim, using the same *Fermi*-LAT Pass 8 event selection and template set.
2. **Extends the analysis to pixel level** (`Totani_reanalysis/`) — a
   per-pixel Poisson likelihood fit over the same 13-energy-bin, 960×960
   pixel grid, combined across energy bins into a single joint significance,
   together with systematics (Fermi-bubble re-extraction, diffuse-model
   variants, disk-inclusion checks), a velocity-dependence / dwarf-spheroidal
   tension study, and a decaying-dark-matter alternative fit.
3. **Compares against dwarf-spheroidal annihilation limits**
   (`dSphs/`, `Totani_reanalysis/core/dsph_limits*.py`) using both the
   standard Hoof et al. (2020, [arXiv:1812.06986](https://arxiv.org/abs/1812.06986))
   stacked s-wave limits and the McDaniel et al. (2024,
   [arXiv:2311.04982](https://arxiv.org/abs/2311.04982)) per-dwarf SED
   likelihoods.

**What the code establishes:** the pixel-level fit independently confirms an
NFW²-like halo excess, with a log-likelihood improvement over the no-halo
baseline of $2\sum_k\Delta\ln\mathcal{L}_k \simeq 1.7\times10^{3}$ across the
13 retained energy bins ($\simeq930$ in the 10-40 GeV signal band alone).
This is a template-level fit-improvement diagnostic, not a calibrated
significance in sigma — the paper is explicit that the regularity conditions
for a chi-squared-distributed likelihood ratio are not met here, so no such
conversion is offered. The annihilation cross section required to produce
the excess sits a factor of a few above the dwarf-spheroidal stacked limit
(R = 5.1, spanning 1.6-9.8 across the full systematic budget). This
repository computes those numbers; it does not resolve that tension. A
two-component dark-sector / spectral-reshaping interpretation is explored in
the paper as a mechanism that *lessens* — not resolves — the discrepancy.

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

If you use the cellwise replication independently, please also cite the
original result:

> T. Totani, "20 GeV halo-like excess of the Galactic diffuse emission and
> implications for dark matter annihilation,"
> [arXiv:2507.07209](https://arxiv.org/abs/2507.07209) (2025).

## License

MIT — see `LICENSE`.
