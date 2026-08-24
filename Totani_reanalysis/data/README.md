# Reference tables

Small, publicly-sourced reference tables used by the halo-spectrum and
dwarf-spheroidal-limit loaders. All are committed directly (largest is
3.8 MB); nothing here needs a separate download step.

## PPPC4DMID gamma-ray yield table

`AtProduction_gammas.dat` — PPPC4DMID Release 6.0 gamma-ray yield table,
Cirelli et al., JCAP 1103 (2011) 051 [arXiv:1012.4515], erratum JCAP 1210
(2012) E01. Source: <https://www.marcocirelli.net/PPPC4DMID.html>. Used by
`core/spectral_reshaping.py` and the PPPC-template scripts to translate a
dark-matter mass and annihilation/decay channel into a photon spectrum.
Overridable via `PPPC4DMID_GAMMAS=/path/to/AtProduction_gammas.dat`.

## Boddy et al. 2019 velocity-dependent J-factors

`boddy2019_{swave,pwave,dwave,sommerfeld}_{bb,ww,tautau}.txt` — effective
J-factor tables for s/p/d-wave and resonant-Sommerfeld velocity dependence,
digitised from Boddy, Kumar, Strigari & Wu, Phys. Rev. D 100, 063019 (2019)
[arXiv:1905.03797]. Used by `Totani_reanalysis/velocity_dep/` for the
velocity-dependent dwarf-spheroidal comparison (the correct comparison set
for any non-s-wave hypothesis — the standard Hoof/McDaniel limits below
assume s-wave).

## Hoof et al. 2020 dSph limits

`oneD_frequentist_limits_{bb,ww,tautau,tt}_channel.txt` — one-dimensional
frequentist 95% CL dwarf-spheroidal annihilation limits, Hoof, Geringer-Sameth
& Trotta, JCAP 02 (2020) 012 [arXiv:1812.06986]. Source:
<https://zenodo.org/records/2612268>. Used as the default s-wave dSph
comparison in `core/dsph_limits.py` / `core/dsph_limits_multi.py`.

See also `dSphs/` at the repo root for the McDaniel et al. 2024 legacy
per-dwarf SED likelihoods and TS profiles [arXiv:2311.04982] used for the
individual-dwarf (rather than stacked one-dimensional) comparison.
