# Reproducing the results

Exact commands behind the cellwise replication, the pixel-level reanalysis,
and the dwarf-spheroidal comparison in "The 20 GeV Galactic Halo Excess:
Pixel-Level Confirmation and Consistency with Sub-TeV WIMP Annihilation"
(arXiv:2607.08552).

There are three stages, in increasing order of what they require:

| Stage | Produces | External data needed |
|-------|----------|----------------------|
| 0 | Installed package, passing smoke tests | none |
| 1 | Posterior-derived spectra, dSph limit comparisons, figures | the posterior from stage 2 |
| 2 | The posterior itself (cellwise + pixel-level MCMC fits) | raw *Fermi*-LAT data (docs/DATA_FETCH.md) |

Run everything from the repository root.

### What you need to fetch yourself, up front

Nothing below is shipped in this repository. Full detail, exact URLs and
commands are in **[docs/DATA_FETCH.md](docs/DATA_FETCH.md)**; this is the
one-paragraph summary, cross-checked against Sec. II A of the paper:

1. **Fermi-LAT Pass 8 photon and spacecraft files**, UltraCleanVeto event
   class, FRONT+BACK, 4 August 2008 to 11 March 2026, from the LAT Data
   Server. ROI |l| <= 60 deg, 10 <= |b| <= 60 deg (disk excluded), 13
   log-spaced energy bins from 1.16 to 1058 GeV, binned on a 0.125 deg CAR
   grid (960x960 pixels).
2. **The 4FGL-DR4 point-source catalogue** (`gll_psc_v35.fit`), from
   `fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/`.
3. **The Ackermann et al. (2012, ApJ 750, 3) GALPROP interstellar-emission
   model grid** — 14 of the 128 published models, fetched by
   `Totani_reanalysis/systematics/get_galprop.sh`.
4. **Dwarf-spheroidal limit tables**: Hoof, Geringer-Sameth & Trotta (2020,
   arXiv:1812.06986; full data release at
   `doi.org/10.5281/zenodo.2612268`), Boddy et al. (2019, arXiv:1909.13197)
   and McDaniel et al. (2024, arXiv:2311.04982). The Hoof and Boddy limit
   curves this repo actually fits against are small digitised tables already
   committed under `Totani_reanalysis/data/`; the McDaniel per-dwarf
   SED/TS-profile release is committed in full under `dSphs/` rather than
   fetched (flagged in the audit notes below).

---

## Stage 0 — Install

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests/
```

Expected: 5 passing. These check that the repo's own layout resolves
(`shared.paths`), that the physics library and shared helpers import, and
that the `HALO_POSTERIOR_ROOT` override is honoured so a regenerated
posterior can be pointed at from anywhere. They require no fit output — the
posterior itself is produced in Stage 2.

---

## Stage 1 — Working from the posterior

No fit output is shipped with this repository. Run Stage 2 first to produce
the posterior, then point `HALO_POSTERIOR_ROOT` at it; everything in this
stage reads from there.

### Load the halo spectrum

```python
from core.totani_data_loader import load_halo_spectrum_rho2

spec = load_halo_spectrum_rho2()   # NFW rho^2, disk excluded — the published profile
print(spec.E_bins_GeV, spec.phi, spec.phi_p16, spec.phi_p84)
```

`core/totani_data_loader.py` documents every available profile key
(`available_mcmc_dirs()`): `rho2`, `rho2.5`, `rho1` (cellwise, Totani-style),
and `global_rho2` / `pixelwise_global_rho2` etc. (single-likelihood and
pixel-level global fits, disk included/excluded variants).

### Dwarf-spheroidal comparison

```bash
python Totani_reanalysis/audit_dsph_limits.py
python Totani_reanalysis/verify_dsph_consistency.py
```

Reads `Totani_reanalysis/data/oneD_frequentist_limits_*.txt` (Hoof et al.
2020 stacked s-wave limits) and `dSphs/dSphs/{SEDs,TS_profiles}/` (McDaniel
et al. 2024 per-dwarf likelihoods).

### Velocity-dependent tension study

```bash
python Totani_reanalysis/velocity_dep/scan_tension_resolution.py --help
```

Reads `Totani_reanalysis/data/boddy2019_*.txt` (s/p/d-wave and
resonant-Sommerfeld effective J-factors) — the correct comparison set for any
non-s-wave hypothesis, since the Hoof/McDaniel limits above assume s-wave.

### Posterior template covariance

```bash
python Totani_reanalysis/make_paper_results_figures.py --only 19 \
    --out-dir Totani_reanalysis/paper_plots
```

Works from a fresh clone: this figure reads only the committed
`Totani_reanalysis/results/posterior-covariance/posterior_covariance.npz`
(verified — regenerates `f_posterior_template_covariance.{pdf,png}`
end-to-end with no external data). It reports the posterior correlation
between the fitted template normalisations (diffuse, bubble, Loop I, halo),
quantifying the degeneracy that a quadrature-sum systematic budget ignores.
It also writes plain-text tables (`correlation_matrices.txt`,
`halo_correlation_summary.txt`) alongside the `.npz`.

To regenerate `posterior_covariance.npz` itself (rather than just redrawing
the figure from it) you need the *full* per-walker pixelwise MCMC chains for
the rho² profile, retained from the Stage 2 run (see "What's committed vs.
what you regenerate" in README.md):

```bash
python Totani_reanalysis/systematics/posterior_covariance.py
python Totani_reanalysis/systematics/positivity_occupied_check.py
```

`positivity_occupied_check.py` additionally needs the per-pixel
`global_fit_k*.npz` residual/model maps (also not shipped, ~8–12 MB each)
from Stage 2's pixel-level global fit.

### Hard-IC injected-population contest

Tests whether a centrally-injected hard inverse-Compton leptonic population
(propagated with the paper's own transport, emitted via full Klein-Nishina
inverse-Compton off a three-component ISRF) can mimic or displace the
NFW-squared halo template. Code and self-tests run from a fresh clone with
no external data:

```bash
python Totani_paper_check/make_templates/hardic_physics.py --selftest   # 6/6 physics checks
python Totani_paper_check/make_templates/verify_hardic_naming.py        # label-safety checks
```

Building the templates and fitting them needs the raw *Fermi*-LAT-derived
counts/exposure cubes and templates (Stage 2, below):

```bash
python Totani_paper_check/make_templates/build_hardic_template.py --variant burst_m
# ... burst_s, burst_l, cont, adv, cmbonly_burst_m
python Totani_reanalysis/mcmc/write_global_systematics_configs.py   # writes mcmc/configs/systematics/hardic_*.json
python Totani_reanalysis/mcmc/run_global_from_config.py --config Totani_reanalysis/mcmc/configs/systematics/hardic_burst_m.json
# ... and the mcmc/configs/hardic/hardic_only_*.json halo-removed comparators
python Totani_reanalysis/systematics/hardic_report.py --tags burst_s burst_m burst_l cont adv cmbonly_burst_m
python Totani_reanalysis/morphology/angular_profile_hardic.py
python Totani_reanalysis/make_paper_results_figures.py --only 20 --out-dir Totani_reanalysis/paper_plots
# --only 19 posterior template covariance, --only 21 angular profile + hard-IC
# morphology, --only 22 halo-removed hard-IC spectra.
# f_angular_profile_by_energy is made by morphology/angular_profile_by_energy.py,
# not by this script.
```

**Verified:** `hardic_report.py` and `--only 20` both need the per-pixel
`global_fit_k*.npz` maps this repo does not ship — run against a fresh
clone with no local fits, `hardic_report.py` reports "NO RESULTS FOUND" per
tag (not a crash) and `--only 20` raises a clear "missing hard-IC baseline"
`SystemExit`; both point at the directory to populate. The report
(`Totani_reanalysis/results/posterior-covariance/angular_profile_hardic.txt`)
and the per-template posterior summaries under
`Totani_reanalysis/mcmc/global_systematics/hardic_*/` and
`Totani_reanalysis/mcmc/hardic_only/*/` (`mcmc_results_k*.{txt,npz}`, same
schema as the rest of the repo) are written by that run.

The Fermi-bubble re-extraction self-consistency check
(`systematics/run_hardic_bubble_reextraction.py`) does not run the
expensive re-extraction fit unless you pass `--run-reextraction` — but by
default it *does* rewrite `mcmc/configs/systematics/*.json` (cheap,
idempotent JSON regeneration; verified — running it with no flags reprints
the two re-extraction commands as a dry preview and regenerates the
systematics configs to byte-identical content):

```bash
python Totani_reanalysis/systematics/run_hardic_bubble_reextraction.py            # dry preview + config regen
python Totani_reanalysis/systematics/run_hardic_bubble_reextraction.py --skip-configs   # preview only, no disk writes
```

### Figures

```bash
python Totani_reanalysis/make_paper_results_figures.py --help
```

Per-figure scripts live in `Totani_reanalysis/figures/` and
`Totani_paper_check/figures/` is superseded by them — see
`Totani_paper_check/README.md`. Output is not committed (`.gitignore`
excludes `paper_plots/`, `plots/`, `conference_figs/`); regenerate locally.

---

## Stage 2 — Regenerating the posterior itself

### Get the raw data

See **[docs/DATA_FETCH.md](docs/DATA_FETCH.md)** for the exact *Fermi*-LAT
Pass 8 UltraCleanVeto event selection, time range (2008-08-04 to
2026-03-11, matching Sec. II A of the paper), IRF version
(`P8R3_ULTRACLEANVETO_V3`), and the
`gtselect`/`gtmktime`/`gtbin`/`gtltcube`/`gtexpcube2` chain that turns it
into the counts/exposure cubes both analyses below read. The same page also
covers the 4FGL-DR4 point-source catalogue and the Ackermann et al. (2012)
interstellar-emission grid.

```bash
export FERMI_DATA_ROOT=/path/to/fermi_data   # default: ./fermi_data
cd Totani_paper_check/data_download
bash download_fermi_totani_query.sh all
bash process_totani_full.sh
```

### Build the templates (Totani_paper_check)

```bash
bash Totani_paper_check/make_templates/build_templates.sh
```

Builds the counts-space templates (`mu_{label}_counts.fits`, shape
`(13, 960, 960)`) for gas, isotropic, ICS, Loop I, point sources, the NFW
halo profile(s), and the Fermi bubbles.

### Cellwise fit (Totani replication)

```bash
python Totani_paper_check/mcmc/run_mcmc_from_config.py \
    --config Totani_paper_check/mcmc/configs/fig6_structured_og.json
```

or run all five published configurations in sequence — the same command
`runallTotani.sh` driver:

```bash
for cfg in fig6 fig2_3 fig4 fig5 fig7; do
  python Totani_paper_check/mcmc/run_mcmc_from_config.py \
      --config "Totani_paper_check/mcmc/configs/${cfg}_structured_og.json"
done
```

Writes `mcmc_results_k00.npz` … `mcmc_results_k12.npz` (one per energy bin)
into `Totani_paper_check/mcmc/mcmc_results_<fig>/`, each carrying the full
per-walker chain — this is what gets thinned to the summary form used
throughout the paper (see below).

### Pixel-level global fit (the reanalysis)

```bash
python Totani_reanalysis/mcmc/run_global_from_config.py --help
```

Configs in `Totani_reanalysis/mcmc/configs/`. Writes
`mcmc_results_k*.npz` (posterior summary) and `global_fit_k*.npz` (per-pixel
residual/model maps, ~8 MB each — not committed) into
`Totani_reanalysis/mcmc/fit_results/quick_global/<profile>/`.

### The dark-matter mass fit: pipeline order and the fixed grid

**This is the step most likely to silently produce wrong numbers if run out
of order or on an ad-hoc grid, so read this before running anything below.**
The correct sequence is:

```
mass scans (run_pppc_mass_scan.py)
    -> propagate the variant suite (systematics/propagate_systematics.py)
        -> total band (systematics/total_systematic_band.py)
            -> figures (make_paper_results_figures.py)
```

Each stage reads the previous stage's output. Regenerating a mass scan
without re-running `propagate_systematics.py` and `total_systematic_band.py`
afterwards leaves a new best-fit mass sitting inside an old, stale
tension band; regenerating figures without redoing the band above them
plots an inconsistent combination. Always re-run the whole chain downstream
of whatever you changed, not just the one script you touched.

**Every mass scan must be run on the same grid:
`--mass-min 20 --mass-max 2000 --n-masses 200`.** This is not a stylistic
preference. A best fit that lands on a grid boundary is a bound on the
scan range, not a measurement of the mass, and this repository's history
has three documented cases of exactly that going wrong: a $\tau^+\tau^-$
fit that railed at a 200 GeV grid floor and was reported at
$\chi^2/\nu=502/11$ when the true minimum (on a wider grid) is 70 GeV at
14.6; a per-bin angular-profile power that railed at the top of a
0.8-3.2 scan; and four different "baseline" $b\bar b$ masses (716, 725,
736, 750 GeV) in circulation from four different ad-hoc grids run at
different times. If you change the grid, expect the masses, the tension
band and the angular-profile power scan to all move, and re-run
everything downstream.

```bash
# Run every channel on every fit preset, all on the same 200-point grid.
for fit in fig6 global_fig6 pixelwise_global_fig6; do
  for channel in bb WW tautau; do
    python Totani_reanalysis/run_pppc_mass_scan.py \
        --channel "$channel" --fit "$fit" \
        --mass-min 20 --mass-max 2000 --n-masses 200 \
        --skip-build
  done
done
```

#### Which preset produces which number

The three presets are three different fits to the same data, not three
attempts at one fit, so **they legitimately return different masses and it is
not a discrepancy to be reconciled.** Quoting a number without naming its
preset is the commonest way to misread this repository's output.

| `--fit` preset | What it is | bb best-fit mass | Where it appears |
|---|---|---|---|
| `global_fig6` | Deterministic (L-BFGS-B) maximum-likelihood global pixel-level fit. Fast; seeds the MCMC and drives the whole systematics suite. | **736 GeV** (9.803e-25 cm^3/s, R = 5.121) | **The band anchor.** Everything in `systematics/` is anchored here. |
| `pixelwise_global_fig6` | The production pixel-level Poisson MCMC on the native 0.125 deg maps. | **716 GeV** (9.76e-25 cm^3/s) | Sec. IV A and Figs. 7 and 8. WW on this preset: 557 GeV, 1.03e-24 cm^3/s. |
| `fig6` | The cellwise 10 deg-cell replication of the original analysis (`totani_rho2` reference), for comparison with the published cell-aggregated result. | **776 GeV** (chi^2/nu = 2.68) | Sec. III A, the cellwise-vs-pixelwise comparison. Note the different ndof: 10, not 11. |

The deterministic and MCMC fits agree to about 5% across the well-populated
bins (about 2% in the 10-40 GeV signal band), degrading to 15-40% in the top
three bins where pixel occupancy collapses. The 736-vs-716 GeV gap is that
same difference expressed as a mass. The cellwise 776 GeV differs from both
for a separate reason: aggregating to 10 deg cells discards the sub-cell
spatial information the pixel-level likelihood uses.

A fourth number, **739 GeV**, appears in `propagate_systematics.py`'s own
output. That is the baseline of the *variant* grid it refits internally (see
"Systematics" below), not a fifth fit of the data.

Drop `--skip-build` only if the underlying MCMC/global fit for that preset
has not been run yet (Stage 2 above); with it set, the scan re-uses the
committed/regenerated posterior spectrum and only refits the PPPC
normalisation per mass point, which is what the numbers above assume.
`tautau` is expected to fail to reproduce Totani's solution: our fit gives
$m_\chi\simeq70$ GeV at $\chi^2/\nu\simeq14.6$, roughly twice the $b\bar b$/WW
baseline and worse than any foreground variant in either of those channels
(Sec. IV A) — that is the expected, published result, not a bug.

### Systematics

```bash
python Totani_reanalysis/systematics/rebuild_bubbles_per_variant.py --help
python Totani_reanalysis/systematics/propagate_systematics.py --help
python Totani_reanalysis/systematics/total_systematic_band.py --help
```

Rebuilds the Fermi-bubble re-extraction variants and propagates the
resulting systematic band. The per-variant bubble-template FITS cubes
(~48 MB × 12 variants) are not committed.

`propagate_systematics.py` re-fits every variant's own halo spectrum on its
own internal mass grid (`--m-min`/`--m-max`/`--n-mass`), now defaulting to
the same **20-2000 GeV, 200 points** as every other scan in the repository.
It is used to compute the min/max tension *fraction* across variants.

Read its output with one distinction in mind: **the table columns
(`m`, `<sigma v>`, `tension`, `chi2/ndof`) and the fraction range are
computed from each variant's own fit and are anchor-independent. Only the
three `# m_chi` / `# <sigma v>` / `# dSph tension` summary lines in the
footer are `anchor x fraction`.** A run left on the wrong channel's anchors
therefore has a wrong footer and perfectly correct columns.

The `--anchor-*` defaults are now the published bb anchor, so a bare run is
already correct for bb:

```bash
# bb (defaults now carry the published anchor: 736 GeV, 9.803e-25, R=5.121)
python Totani_reanalysis/systematics/propagate_systematics.py
```

#### Both channel bands

The **W+W- band is propagated separately** from bb, on its own anchor. It is
not a rescaling of the bb band, and the two-panel dwarf figure (Fig. 8) draws
a different band on each panel. Run both.

```bash
# --- b-bbar: the module defaults are the published anchor -------------------
python Totani_reanalysis/systematics/propagate_systematics.py
python Totani_reanalysis/systematics/total_systematic_band.py
#   -> R_bb = 5.1, spanning [1.6, 9.8]

# --- W+W-: anchored on the global_fig6 WW scan ------------------------------
python3 systematics/propagate_systematics.py --channel WW \
    --anchor-mass 557 --anchor-sv 1.001e-24 --anchor-tension 4.073 \
    --out results/global-systematics/systematics_science_band_WW.txt
python3 systematics/total_systematic_band.py \
    --R0 4.073 --fg-lo 0.341 --fg-hi 1.314
#   -> R_WW = 4.1, spanning [1.2, 7.7]
```

(The two WW commands are written relative to `Totani_reanalysis/`, which is
where the `results/` and `systematics/` paths in them resolve. Either `cd
Totani_reanalysis` first, or prefix both paths as the bb commands above do.)

The WW foreground leg is `x[0.341, 1.314]`, against bb's `x[0.375, 1.320]`.
That is the same behaviour in both channels, which is the point of
propagating it twice: it shows the band is not an artefact of the bb
spectrum.

**Both `total_systematic_band.py` runs above combine their foreground leg in
quadrature with the halo-J (+/-48%) and dwarf-J (+/-54%) legs**, from its
`--halo-J-frac 0.48` / `--dwarf-J-frac 0.537` defaults. The bb call with no
arguments reproduces the published band from its baked-in defaults
(`--R0 5.121`, `--fg-lo 0.375`, `--fg-hi 1.320`), giving R = 5.1 spanning
[1.6, 9.8] in quadrature and [0.8, 15.4] as an extreme linear worst case.

Only re-derive `--fg-lo`/`--fg-hi` from a fresh `propagate_systematics.py`
run if you have changed the variant suite (new IEM grid models, a different
foreground group membership, etc.) — otherwise the committed defaults are
the published numbers.

Both bands are also hard-coded, with their provenance, near the top of
`Totani_reanalysis/make_paper_results_figures.py`, as `R_BAND_LO`/`R_BAND_HI`/
`R_BAND_CENTRAL` (bb) and the per-channel `R_BAND` dict. **If you re-derive a
band, update that dict too** — every figure that shades a band reads it from
there, and a two-channel figure that reads the scalars instead of the dict
silently draws the bb band over the WW panel.

### The fourteen-model GALPROP interstellar-emission grid

The foreground leg of the systematic band is bracketed by fourteen models from
the Ackermann et al. (2012) grid (ApJ 750, 3), spanning four cosmic-ray source
distributions (Lorimer, supernova-remnant, Yusifov, OB-star), diffusion halo
half-heights z_h = 4, 6, 8, 10 kpc, radial boundaries R = 20 and 30 kpc, spin
temperatures T_S = 150 K and 1e5 K, and E(B-V) magnitude cuts of 2 and 5.

```bash
# 1. fetch the public GALDEF files and pre-calculated GALPROP output
bash Totani_reanalysis/systematics/get_galprop.sh

# 2. build gas_<tag> and ics_<tag> counts templates for all fourteen models
python Totani_paper_check/make_templates/build_iem_grid_all.py

# 3. fit each model twice, once on the baseline bubbles and once on bubbles
#    re-extracted self-consistently against that model
for c in Totani_reanalysis/mcmc/configs/systematics/iem_*.json; do
  python Totani_reanalysis/mcmc/run_global_from_config.py --config "$c"
done

# 4. propagate. Only the *_bubbles_reextracted fits enter the band.
python Totani_reanalysis/systematics/propagate_systematics.py
python Totani_reanalysis/systematics/total_systematic_band.py
```

`propagate_systematics.py` restricts the foreground leg to `FOREGROUND_GROUPS`
= {baseline, diffuse, diffuse_ics, diffuse_grid_bubbles}. Quadrant, hemisphere,
mask and PSF variants are reported separately and must not widen that leg. The
current edges are `--fg-lo 0.375 --fg-hi 1.320` for bb and
`--fg-lo 0.341 --fg-hi 1.314` for WW; in both channels the low edge is
`diffuse_giem_only` and the high edge `iem_so_z6_bubbles_reextracted`.

Two GIEM configurations are deliberately **excluded** from the band by
`--exclude`, and they push in opposite directions, so this is not selective
pruning: `diffuse_giem` (GIEM swapped in but the Loop I and bubble templates
retained, which double-counts large-scale flux, since the data-driven GIEM
already contains it) and `diffuse_no_bubbles` (bubbles dropped with no
replacement, an upward stress test). The third configuration,
`diffuse_giem_only` — GIEM replacing gas and ICS with Loop I and the bubbles
dropped — **is** in the band, and sets its lower edge.

### The low-energy shoulder diagnostics

The models with the most centrally concentrated source distributions return
extra halo flux below 5 GeV. These tools localise it and quantify what it does
to the inferred mass.

```bash
# write the latitude-band (10-20, 20-40, 40-60 deg) and raised-inner-cut
# (15, 20 deg) configs, then fit them
python Totani_reanalysis/mcmc/write_shoulder_test_configs.py --parents all
python Totani_reanalysis/mcmc/write_shoulder_test_configs.py --parents all --commands

# read off the 1.5-4.4 GeV / 12-36 GeV ratio per region
python Totani_reanalysis/systematics/shoulder_by_region.py

# refit the annihilation mass with the low-energy bins excluded
python Totani_reanalysis/systematics/refit_mass_energy_range.py
```

Use `run_global_from_config.py`, not `run_mcmc_from_config.py`. The former is
the deterministic global path that writes `global_fit_k??.npz` and the
coefficient tables everything downstream reads.

### Per-energy-bin morphology

```bash
python Totani_reanalysis/morphology/angular_profile_by_energy.py
```

Fits the excess ring profile independently in each energy bin and scans the
preferred NFW emissivity power p in rho^p. **The default scan is now the wide
`--powers 0.4 5.0 231` range**, which is the one the published p = 2.8 +/- 0.2
comes from; an earlier, narrower default railed the ~35 GeV bin against its
own upper boundary. `--powers 1.0 3.2 12` gives a fast two-minute version for
smoke-testing, but do not quote a number from it.

The output table carries `p_lo`/`p_hi` (the delta-chi2 = 1 interval), a
constant-p fit across the signal band, the scanned grid in its header, and a
`RAIL` flag on any bin whose best fit or interval edge sits on the grid
boundary. **A RAIL-flagged value is a bound set by the scan range, not a
measurement.** Four bins still rail even on the wide grid (2.55, 4.31, 482
and 814 GeV), all of them outside the 10-40 GeV signal band. Writes
`results/global-systematics/angular_profile_by_energy.txt` and
`figures/paper_figures/f_angular_profile_by_energy.{pdf,png}`.

Note that `plot_angular_profile_vs_models.py`, which every morphology script
imports its `FITDIR` constant from, points at the **pixelwise** no-halo
residual (`pixelwise_mcmc_results_fig4_resid`), because that is the residual
the published Fig. 3 is drawn from. The deterministic `quick_global` residual
gives visibly different numbers on the same rings.

### Thinning a chain to the summary format

The summary form used throughout the paper drops the `chain` and `logprob`
arrays from the full-chain output above and keeps everything
else (`f_ml`, `f_p16`, `f_p50`, `f_p84`, `labels`, `loglike_ml`,
`acceptance_fraction`, `energy_bin`, `Ectr_mev`, `iso_target_e2`, …). If
you've regenerated a chain locally and want the same thinned form:

```python
import numpy as np
d = np.load("mcmc_results_k00.npz", allow_pickle=True)
keep = {k: d[k] for k in d.files if k not in {"chain", "logprob"}}
np.savez_compressed("mcmc_results_k00_summary.npz", **keep)
```

After regenerating, rerun Stage 1 to redraw the figures against the new
posterior.

---

## Every paper figure, and the command that makes it

All nine figures are drawn by `Totani_reanalysis/make_paper_results_figures.py
--only <N> --out-dir Totani_reanalysis/paper_plots` (run with no `--only` to
make all of them). This table is read off that script's own docstring and
`FIGS` dispatch table, cross-checked against `main.tex`'s figure labels —
not guessed.

**The `--only` numbers are not the paper's figure numbers.** They are the
script's internal figure ids, and the two orders do not coincide: paper
Fig. 3 is `--only 7`, and paper Fig. 6 is `--only 22`. Ids 8 to 21 that do
not appear in the table below are standalone single panels and slide variants
of the same content, not additional paper figures. To regenerate the complete
published set in one call:

```bash
python Totani_reanalysis/make_paper_results_figures.py \
    --only 1 2 3 4 5 6 7 19 22 \
    --out-dir Totani_reanalysis/paper_plots
```

Run the mass scans, `propagate_systematics.py` (both channels) and
`total_systematic_band.py` **before** this, in that order. See the pipeline
note above. Figures drawn over a stale band are the most common way to get a
self-inconsistent set of plots out of this repository.

| main.tex figure | Output file | `--only` | What it needs |
|---|---|---|---|
| Fig. 1 (`fig:global_overlays`) | `f8_pole_flux_profiles` | 1 | Stage 2 pixel-level global fit, disk in/out, all three profiles |
| Fig. 2 (`fig:likelihoods`) | `f9_likelihood_profiles` | 2 | Same, plus the no-halo baseline fit |
| Fig. 3 (`fig:angprof`) | `f_angular_profile_morphology` | 7 | `morphology/angular_profile_by_energy.py` output (or its band-summed sibling `plot_angular_profile_vs_models.py`) |
| Fig. 4 (`fig:systematics`) | `f_systematics_impact` | 5 | The full systematics suite, incl. the 14-model IEM grid |
| Fig. 5 (`fig:covariance`) | `f_posterior_template_covariance` | 19 | `systematics/posterior_covariance.py` output (committed; runs from a fresh clone) |
| Fig. 6 (`fig:hardic`) | `f_hardic_nohalo_spectra` | 22 | `systematics/hardic_nohalo_spectra.py`, the six hard-IC halo-removed comparator fits |
| Fig. 7 (`fig:annihilation`) | `f_annih_bb_WW` | 3 | The bb/WW mass scans on `pixelwise_global_fig6` (200-point grid) |
| Fig. 8 (`fig:veldep`) | `f_dsph_tension_WW_bb` | 4 | The bb/WW mass scans plus **both** bands from `total_systematic_band.py` — this is the two-panel figure that needs the per-channel `R_BAND` dict |
| Fig. 9 (`fig:decay`) | `f_decay_rho1` | 6 | The tautau/bb/WW mass scans on the rho^1 (decay-like) profile |

### Headline numbers and where they come from

| Number in the paper | Value | Command |
|---|---|---|
| dSph tension band, bb | R = 5.1, spanning [1.6, 9.8] | `systematics/total_systematic_band.py` (no args) |
| dSph tension band, WW | R = 4.1, spanning [1.2, 7.7] | `systematics/total_systematic_band.py --R0 4.073 --fg-lo 0.341 --fg-hi 1.314`, after the WW `propagate_systematics.py` run above |
| Foreground leg on R, bb | x[0.375, 1.320] | `systematics/propagate_systematics.py` over `FOREGROUND_GROUPS`, printed as `[frac LO-HI]` |
| Foreground leg on R, WW | x[0.341, 1.314] | same, `--channel WW` with the WW anchors |
| Band anchor (bb, `global_fig6`) | 736 GeV, 9.803e-25 cm^3/s, R=5.121 | `run_pppc_mass_scan.py --channel bb --fit global_fig6 --mass-min 20 --mass-max 2000 --n-masses 200` |
| Band anchor (WW, `global_fig6`) | 557 GeV, 1.001e-24 cm^3/s, R=4.073 | same, `--channel WW` |
| bb best fit (`pixelwise_global_fig6`) | 716 GeV, 9.76e-25 cm^3/s | same, `--fit pixelwise_global_fig6` |
| WW best fit (`pixelwise_global_fig6`) | 557 GeV, 1.03e-24 cm^3/s | same, `--channel WW --fit pixelwise_global_fig6` |
| bb best fit (cellwise `fig6`) | 776 GeV, chi2/nu = 2.68 (ndof 10) | same, `--fit fig6` |
| tautau | 70 GeV, chi2/nu = 14.6 | same, `--channel tautau` |
| Angular-profile chi2/nu (band-summed, Fig. 3) | 7.1 (rho^2), 5.1 (rho^2.5), 15.4 (rho^1), 28.8 (isotropic) | `morphology/plot_angular_profile_vs_models.py` |
| Per-bin emissivity power p | 2.8+/-0.3, 2.7+/-0.2, 3.5+/-0.5 in the three signal-band bins; weighted mean 2.8+/-0.2, chi2 = 2.3 for 2 dof against a constant | `morphology/angular_profile_by_energy.py` (no args; its default is now the wide `--powers 0.4 5.0 231` grid these numbers come from). Check the `RAIL` column before quoting any bin |
| Halo-ICS posterior correlation | -0.481 | `systematics/posterior_covariance.py`, peak (20.76 GeV) bin |
| Halo-isotropic correlation | -0.007 | same |
| Isotropic-Loop I A correlation | -0.93 | same |
| Likelihood preference, 2*sum(Delta lnL) | ~1.7e3 (13 bins), ~930 (signal band) | `compute_halo_excess_significance.py`, disk-excluded pixelwise fit. The paper treats this as a template-level fit-improvement diagnostic only. The script also prints a Wilks sqrt(TS) "sigma" for convenience — the paper explicitly does NOT adopt that conversion (Sec. III B: the regularity conditions for a chi^2-distributed likelihood ratio are not met), so quote only the 2*sum(Delta lnL) number, never the sigma the script prints alongside it |

If a command above gives you a different number, work back up the pipeline
order in the previous section before assuming the code is wrong: the most
common cause is a stale systematics band sitting under a freshly re-run mass
scan, or a mass scan run on a grid other than the fixed one.

---

## Directory reference

| Path | Role |
|------|------|
| `Totani_reanalysis/core/` | Cross sections, spectral transport, halo/IGRB loaders, dSph limits |
| `Totani_reanalysis/mcmc/` | Pixel-level global-fit code (writes its posteriors here) |
| `Totani_reanalysis/systematics/` | Fermi-bubble re-extraction, systematic-band propagation |
| `Totani_reanalysis/velocity_dep/` | Velocity-dependent cross section / dSph tension study |
| `Totani_reanalysis/decay/` | Decaying-DM alternative fit |
| `Totani_paper_check/data_download/` | Fermi-LAT query + Fermitools processing pipeline |
| `Totani_paper_check/make_templates/` | Counts-space template construction |
| `Totani_paper_check/mcmc/` | Cellwise fit code (writes its posteriors here) |
| `dSphs/` | Per-dwarf SED likelihoods and TS profiles (McDaniel et al. 2024) — bundled in full rather than fetched, see docs/DATA_FETCH.md sec. 6 |
| `shared/`, `helpers/` | Path resolution, FITS I/O, plotting style |
| `docs/DATA_FETCH.md` | Exact *Fermi*-LAT selection and processing commands |
