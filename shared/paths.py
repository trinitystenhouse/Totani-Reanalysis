"""Single source of truth for repo and data locations.

Import this instead of hard-coding relative paths or
``Path(__file__).resolve().parent.parent`` walks. It locates this repo's root
by walking up to the directory that contains the marker directories
``helpers`` and ``Totani_reanalysis``, with a ``TOTANI_REPO_ROOT``
environment-variable override for unusual layouts.

Usage
-----
    from shared.paths import FERMI_DATA, PIXELWISE_MCMC, REANALYSIS
    counts = FERMI_DATA / "totani" / "processed" / "counts.fits"

Every path is a ``pathlib.Path``. Nothing here touches the filesystem at
import time except the read-only root search.

Data not shipped in this repository (raw Fermi-LAT event files, the MCMC
posteriors and everything derived from them) is resolved through
environment-variable overrides documented in docs/DATA_FETCH.md and
REPRODUCE.md, defaulting to a repo-local path that will simply not exist
until you have regenerated it.
"""
from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("helpers", "Totani_reanalysis")


def _find_repo_root(start: Path) -> Path:
    for var in ("TOTANI_REPO_ROOT", "REPO_PATH", "DMPS_REPO_ROOT"):
        env = os.environ.get(var)
        if env:
            p = Path(env).expanduser().resolve()
            if p.is_dir():
                return p
    p = start.resolve()
    for cand in (p, *p.parents):
        if all((cand / m).is_dir() for m in _MARKERS):
            return cand
    raise RuntimeError(
        "Could not locate the Totani_Reanalysis_public repo root from "
        f"{start!r}. Expected an ancestor containing {_MARKERS}. "
        "Set TOTANI_REPO_ROOT to override."
    )


# --- repo root -------------------------------------------------------------
REPO_ROOT: Path = _find_repo_root(Path(__file__))

# --- shared infrastructure -------------------------------------------------
SHARED: Path = REPO_ROOT / "shared"
HELPERS: Path = REPO_ROOT / "helpers"

# --- raw + processed Fermi-LAT data (NOT shipped — see docs/DATA_FETCH.md) -
FERMI_DATA: Path = Path(
    os.environ.get("FERMI_DATA_ROOT", str(REPO_ROOT / "fermi_data"))
)

# --- Totani project roots ---------------------------------------------------
PAPER_CHECK: Path = REPO_ROOT / "Totani_paper_check"
REANALYSIS: Path = REPO_ROOT / "Totani_reanalysis"

# --- shared analysis data (PPPC, Boddy velocity J-factors, Hoof dSph) -------
REANALYSIS_DATA: Path = REANALYSIS / "data"
PPPC_TABLE: Path = REANALYSIS_DATA / "AtProduction_gammas.dat"

# --- reanalysis sub-trees ----------------------------------------------------
REANALYSIS_MCMC: Path = REANALYSIS / "mcmc"
REANALYSIS_FIT_RESULTS: Path = REANALYSIS_MCMC / "fit_results"
# The per-variant deterministic global fits that the systematic band is built
# from. Imported by systematics/propagate_systematics.py and
# systematics/refit_mass_energy_range.py, which are the two scripts that
# produce the published dSph tension band, so it must stay exported.
GLOBAL_SYSTEMATICS: Path = REANALYSIS_MCMC / "global_systematics"
REANALYSIS_RESULTS: Path = REANALYSIS / "results"
REANALYSIS_FIGURES: Path = REANALYSIS / "figures"

# --- paper_check sub-trees ---------------------------------------------------
PAPER_CHECK_MCMC: Path = PAPER_CHECK / "mcmc"
PAPER_CHECK_FIGURES: Path = PAPER_CHECK / "figures"
PAPER_CHECK_RESULTS: Path = PAPER_CHECK / "results"

# --- pixel-level MCMC halo posterior ----------------------------------------
# Not shipped. Regenerate it with Totani_reanalysis/mcmc/run_global_from_config.py
# (see REPRODUCE.md, Stage 2) and point HALO_POSTERIOR_ROOT at the result;
# it defaults to a repo-local pixelwise_mcmc/ that will not exist until then.
PIXELWISE_MCMC: Path = Path(
    os.environ.get("HALO_POSTERIOR_ROOT", str(REPO_ROOT / "pixelwise_mcmc"))
)

__all__ = [
    "REPO_ROOT", "SHARED", "HELPERS", "FERMI_DATA",
    "PAPER_CHECK", "REANALYSIS",
    "REANALYSIS_DATA", "PPPC_TABLE",
    "REANALYSIS_MCMC", "REANALYSIS_FIT_RESULTS", "GLOBAL_SYSTEMATICS",
    "REANALYSIS_RESULTS", "REANALYSIS_FIGURES",
    "PAPER_CHECK_MCMC", "PAPER_CHECK_FIGURES", "PAPER_CHECK_RESULTS",
    "PIXELWISE_MCMC",
]
