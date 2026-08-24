# Totani Paper Check

Independent scripts for reproducing and checking the Totani-style Fermi-LAT
template pipeline (Totani, arXiv:2507.07209). This is the cellwise
replication half of the repository; `Totani_reanalysis/` alongside it is the
pixel-level extension behind the companion paper (arXiv:2607.08552). Run from
the repository root (this directory's parent).

## Layout

- `data_download/`: download and process Fermi-LAT query products into counts,
  livetime and exposure cubes.
- `make_templates/`: build point-source, gas, ICS, isotropic, Loop I, NFW and
  Fermi-bubble count templates.
- `mcmc/`: pixelwise and global likelihood runners plus JSON configs.
- `figures/`: plotting scripts and per-figure configs.
- `check_templates/` and `sanity_checks/`: validation plots and FITS checks.

Large generated data products live under `fermi_data/totani/` at the repository
root and should not be committed. Generated MCMC chains, `.npz` products, plots,
logs, `__pycache__/`, `.DS_Store` and `.egg-info/` directories should also stay
out of git.

The large FSSC Galactic diffuse model `gll_iem_v07.fits` is also not committed.
Download it from:

```text
https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/4fgl/gll_iem_v07.fits
```

and place it at:

```text
fermi_data/totani/templates/gll_iem_v07.fits
```

## Environment

Use a Fermitools-capable conda environment. On a fresh machine, install the
Python package in editable mode from the repository root:

```bash
cd /path/to/Totani_Reanalysis_public
python -m pip install -r requirements.txt
python -m pip install -e .
export REPO_PATH="$PWD"
```

For the Fermitools steps, activate the environment that provides `gtselect`,
`gtmktime`, `gtbin`, `gtltcube` and `gtexpcube2` (see
[docs/DATA_FETCH.md](../docs/DATA_FETCH.md) at the repo root for the exact
event selection and processing commands).

## Data Download And Processing

Download all currently used Fermi query chunks with the manifest-driven script:

```bash
cd /path/to/Totani_Reanalysis_public
bash Totani_paper_check/data_download/download_fermi_totani_query.sh all
```

This writes:

- `fermi_data/totani/315_0/`
- `fermi_data/totani/45_0/`
- `fermi_data/totani/0_45/`
- `fermi_data/totani/0_-45/`
- `fermi_data/totani/0_0/`

Process the downloaded files into the products consumed by the templates and
fits:

```bash
bash Totani_paper_check/data_download/process_totani_full.sh
```

The main outputs are:

- `fermi_data/totani/processed/counts_ccube_1000to1000000.fits`
- `fermi_data/totani/processed/expcube_1000to1000000.fits`
- `fermi_data/totani/processed/templates/`

You can override locations without editing scripts:

```bash
REPO_DIR=/path/to/Totani_Reanalysis_public \
OUT_DIR=/scratch/totani/processed \
bash Totani_paper_check/data_download/process_totani_full.sh /scratch/totani/315_0 /scratch/totani/45_0
```

## Templates, Fits And Figures

Build templates:

```bash
cd /path/to/Totani_Reanalysis_public
bash Totani_paper_check/make_templates/build_templates.sh
```

Run configured fits, for example:

```bash
python Totani_paper_check/mcmc/run_mcmc_from_config.py \
  --config Totani_paper_check/mcmc/configs/base.json
```

Generate the paper-check figures:

```bash
bash Totani_reanalysis/figures/plot_all_cellwise_figures.sh
```

The legacy `Totani_paper_check/figures/plot_all_figures.sh` entrypoint still
works, but it now delegates to the canonical runner in `Totani_reanalysis`.

## Repeated Files To Consolidate

The legacy `download_fermi_totani_*.sh` scripts are repeated wrappers around the
same curl workflow. Prefer `data_download/download_fermi_totani_query.sh` as the
master script and keep the old files only as compatibility entry points, or
delete them after downstream notes have been updated.

`Totani_reanalysis/figures/plot_all_figures.sh` is now the master figure runner
for `cellwise`, `global` and `pixelwise` modes. The duplicated paper-check
figure scripts are compatibility wrappers, so make future figure logic changes
in `Totani_reanalysis/figures`.
