# Fetching the Fermi-LAT data

Raw *Fermi*-LAT event files, spacecraft files, and the processed counts/
exposure cubes built from them are **not** part of this repository (they run
to hundreds of MB–GB and are exactly reproducible from the public archive).
This page is the exact procedure, transcribed from
`Totani_paper_check/data_download/process_totani_full.sh` and the download
scripts alongside it — nothing here is approximate.

## 1. Query the Fermi Science Support Center (FSSC)

Photon and spacecraft files for the ROI are pulled with the LAT Data server
queries in `Totani_paper_check/data_download/download_fermi_totani_*.sh`
(one script per sky-pointing quadrant) and
`download_fermi_totani_query.sh` (the query manifest). Submit each query at
<https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi> or via the
FSSC's astroquery/wget interface, using:

| Parameter | Value |
|---|---|
| Time range | 2008 August 4 – 2026 March 11 (the mission span used in arXiv:2607.08552; extend the query end date to "now" for a fresh pull, but note that additional exposure beyond 11 March 2026 was not used in the published analysis and will not exactly reproduce its numbers) |
| Event class | Pass 8 **UltraCleanVeto** (`EVCLASS=1024`) |
| Event type | FRONT+BACK (`EVTYPE=3`) |
| Energy range (preselection) | 1000 – 1,057,896.18 MeV |
| Search radius | 90° cone about RA=266.4051°, Dec=−28.936175° (Galactic centre), large enough not to clip the analysis ROI below |
| Zenith cut | `ZMAX=100` deg |

Four sky-pointing quadrants are queried separately to keep individual
downloads tractable; `download_fermi_totani_query.sh` records the FSSC query
IDs used for the original pull.

**FSSC asynchronous query results expire.** The pre-recorded query tags in
`download_fermi_totani_query.sh` point at a specific completed FSSC job and
will eventually 404 once that job's results are cleared from the server
(the FSSC does not guarantee long-term retention of query output). If a
tagged download fails, resubmit a fresh query with the exact parameters in
the table above via the LAT Data Server link or its astroquery interface,
and point `BASE_URL`/the individual download scripts at the new query's
output directory.

## 2. Filter, bin and compute exposure

Run the Fermi Science Tools (`fermitools`, install via `conda install -c
conda-forge fermitools` or the FSSC's own instructions) pipeline exactly as
`Totani_paper_check/data_download/process_totani_full.sh` drives it:

```bash
cd Totani_paper_check/data_download
bash download_fermi_totani_0_0.sh    # and the other quadrant scripts
bash process_totani_full.sh
```

`process_totani_full.sh` chains `gtselect` → `gtmktime` → `gtbin` (CCUBE) →
`gtltcube` → `gtexpcube2` with:

| Step | Setting | Value |
|---|---|---|
| `gtselect` | event class / type | UltraCleanVeto, FRONT+BACK (`evclass=1024`, `evtype=3`) |
| | zenith cut | `zmax=100` |
| | energy preselection | 1000 – 1,057,896.180080 MeV |
| | prefilter cone | RA=266.4051°, Dec=−28.936175°, radius 90° |
| `gtmktime` | GTI filter | `(DATA_QUAL>0)&&(LAT_CONFIG==1)` |
| `gtbin` (CCUBE) | projection | CAR, Galactic coordinates, `AXISROT=0` |
| | map size | 960 × 960 pixels, 0.125°/pixel, `|l|,|b| ≤ 60°` |
| | energy binning | 13 bins, log-spaced, edges 1161.872047 – 1,057,896.180080 MeV (matches `data_download/totani_13bins_gev_edges.txt`, bin centres 1.51 GeV – 814 GeV) |
| `gtltcube` | livetime cube | `dcostheta=0.025`, `binsz=1`, `zmax=100` |
| `gtexpcube2` | IRFs | `P8R3_ULTRACLEANVETO_V3` |

This exactly matches Totani's §2.1 selection (arXiv:2507.07209) so the
two analyses in this repo (`Totani_paper_check/`, the cellwise replication;
`Totani_reanalysis/`, the pixel-level extension behind arXiv:2607.08552) sit
on the same photon sample.

**Outputs**, written under `fermi_data/totani/processed/` (or wherever
`$FERMI_DATA_ROOT` points — see `shared/paths.py`):

- `counts_ccube_1000to1000000.fits` — `(13, 960, 960)` counts cube
- `expcube_1000to1000000.fits` — matching exposure cube

Both are then read by `Totani_paper_check/make_templates/*.py` to build the
counts-space emission templates (`mu_{label}_counts.fits`, same `(13, 960,
960)` shape) that the MCMC fits in `Totani_paper_check/mcmc/` and
`Totani_reanalysis/mcmc/` scale against the data.

## 3. What's shipped instead

This repository ships:

- All the code above, so the pipeline can be re-run byte-for-byte once the
  raw data is fetched.
- The **thinned posterior summaries** the figures and downstream analyses
  actually consume — `mcmc_results_k*.npz`/`.txt` under `pixelwise_mcmc/`,
  `Totani_paper_check/mcmc/`, and
  `Totani_reanalysis/mcmc/fit_results/quick_global/`. Each `.npz` carries the
  posterior median and 16th/84th-percentile bounds per template component
  (`f_p16`, `f_p50`, `f_p84`, …) and the maximum-likelihood point, but not the
  raw per-walker chain or log-probability arrays (each of those was 1.5–12 MB
  per energy bin; stripping them keeps every number the code needs while
  cutting >99% of the size). If you need the full chains — e.g. to check
  walker convergence or redraw a corner plot — regenerate them locally with
  `Totani_paper_check/mcmc/run_mcmc.py` /
  `Totani_reanalysis/mcmc/run_global_fit.py`, or get in touch with the
  authors.
- Small reference tables in `data/` (Boddy et al. 2019 velocity-dependent
  J-factors, the Hoof et al. dSph limit set, PPPC4DMID) with their own
  provenance notes.

Not shipped: raw event/spacecraft files, the intermediate counts/exposure/
template FITS cubes (each ~45–50 MB and fully reproducible from the steps
above), the per-variant Fermi-bubble systematics cubes under
`Totani_reanalysis/systematics/` output, and full MCMC run logs. All are
regenerable from this repo's code plus the data pulled above.

## 4. The 4FGL-DR4 point-source catalogue

The point-source template (`Totani_paper_check/make_templates/build_ps_template.py`)
and the pixel-level bright-source mask are built from the LAT 14-year Source
Catalog (4FGL-DR4), `gll_psc_v35.fit` (paper Sec. II A). Download it from:

```text
https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/gll_psc_v35.fit
```

and place it at `fermi_data/totani/templates/gll_psc_v35.fit` (or wherever
`$FERMI_DATA_ROOT` points, see `shared/paths.py`). The extended-source list
used for the extended-source mask ships in the same catalogue release.

## 5. The GALPROP interstellar-emission model grid

The baseline gas/IC diffuse templates and the fourteen-model systematic
bracket both come from the Ackermann et al. (2012, ApJ 750, 3) GALPROP
supplementary grid (128 published models; 14 used here, spanning cosmic-ray
source distribution, propagation halo height, radial boundary, spin
temperature and E(B-V) cut — see `REPRODUCE.md` for the full model list).
Fetch it with:

```bash
bash Totani_reanalysis/systematics/get_galprop.sh
```

which pulls the GALDEF files and pre-calculated GALPROP output from
`https://galprop.stanford.edu/PaperIISuppMaterial` into
`fermi_data/totani/galprop_iem_grid/`. See the script's header for the
naming convention and resume/verification behaviour.

## 6. Dwarf-spheroidal annihilation-limit tables

Sec. IV B compares the halo cross section against two independent limit
sets, both built on the public *Fermi*-LAT dwarf programme:

- **Hoof, Geringer-Sameth & Trotta (2020)**, JCAP 02 (2020) 012,
  arXiv:1812.06986 — the 11-year, 27-dwarf global s-wave analysis. The
  digitised one-dimensional frequentist limit curves used here
  (`Totani_reanalysis/data/oneD_frequentist_limits_*_channel.txt`) are small
  text tables read off the published figures and are committed directly; the
  full likelihood/data release behind that paper is archived on Zenodo at
  <https://doi.org/10.5281/zenodo.2612268> if you need the underlying
  per-dwarf likelihoods rather than the published limit curve.
- **McDaniel et al. (2024)**, Phys. Rev. D 109, 063024, arXiv:2311.04982 —
  the 14-year, legacy stacked analysis, used for the cross-check comparison
  in Sec. IV B. This repository currently ships the full per-dwarf SED and
  TS-profile release under `dSphs/` rather than fetching it (see the release
  notes below for why that is flagged for follow-up).
- **Boddy et al. (2019)**, Phys. Rev. D 102, 023029, arXiv:1909.13197 —
  velocity-dependent (p-wave, d-wave, resonant-Sommerfeld) effective
  J-factors for the same dwarf sample, used in Sec. IV C. The digitised
  tables (`Totani_reanalysis/data/boddy2019_*.txt`) are likewise small and
  committed directly.

**Note on `dSphs/`:** unlike the tables above, `dSphs/dSphs/{SEDs,TS_profiles}/`
is a full local copy of McDaniel et al.'s public data release (279 files,
~6.5 MB) rather than a fetch script. It is small enough not to trip the
repository's size checks, but it is third-party data with its own
citation and, if you are auditing this repository for data-shipping
hygiene, it is the one item that should arguably be replaced with a fetch
step against the original release rather than bundled. It is not
required to reproduce any number in Table/Fig. references that come from
the Hoof-limit (`--dsph-source hoof`, the default throughout) pipeline —
only the McDaniel cross-check uses it.
