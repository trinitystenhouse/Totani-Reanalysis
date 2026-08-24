"""
Source-aware dwarf-spheroidal upper limits on the annihilation cross section,
with support for *velocity-dependent* (p-wave / d-wave / Sommerfeld) limits.

This is a drop-in superset of ``core.dsph_limits``.  Differences:

  * Multiple limit *sources* selectable by string:
        hoof          - Hoof et al. 2020 s-wave (default, = the legacy table)
        mcdaniel      - McDaniel et al. 2024 s-wave (bb, tautau; Hoof fallback)
        boddy_swave   - Boddy et al. 2019 s-wave   (effective J, n=0)
        boddy_pwave   - Boddy et al. 2019 p-wave    (sigma v propto v^2)
        boddy_dwave   - Boddy et al. 2019 d-wave    (sigma v propto v^4)
        boddy_somm    - Boddy et al. 2019 Sommerfeld (Coulomb, sigma v propto 1/v)

  * Out-of-range queries return ``np.nan`` (NOT ``np.inf``).  The legacy
    ``np.inf`` return silently produced *zero* tension for masses outside the
    table (tension = sigmav / inf = 0 -> spuriously "resolved"), which is the
    artefact suspected behind the old "tension dissolves above ~650 GeV"
    narrative.  ``np.nan`` propagates honestly and is masked by the scan.

  * The velocity-dependent tables store a *halo-comparable* limit: the upper
    limit on the **halo-frame** <sigma v> implied by the dwarf non-detection
    under the given velocity law, so that the existing tension definition
        tension = <sigma v>_halo / <sigma v>_dSph_limit
    remains a like-for-like ratio.  See ``extract_boddy_limits.py`` and the
    derivation in README_dsph_sources.md for the exact rescaling.

Tables live in ``Totani_reanalysis/data`` as two-column ASCII
(``# comment`` allowed): mass [GeV] | <sigma v>_UL [cm^3/s].  Interpolation is
log-log, matching the legacy convention.

Author note: keep this importable without the velocity-dependent tables present
(they are produced by extract_boddy_limits.py); a missing table raises a clear
FileNotFoundError only when that source is actually requested.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data"
_REPO_DIR = _HERE.parent.parent
_MCDANIEL_DIR = _REPO_DIR / "dSph_upper_limits"

# --- s-wave legacy tables (Hoof) -------------------------------------------
_HOOF_FILES = {
    "WW": _DATA_DIR / "oneD_frequentist_limits_ww_channel.txt",
    "bb": _DATA_DIR / "oneD_frequentist_limits_bb_channel.txt",
    "tautau": _DATA_DIR / "oneD_frequentist_limits_tautau_channel.txt",
}

# --- McDaniel s-wave tables -------------------------------------------------
_MCDANIEL_FILES = {
    "bb": _MCDANIEL_DIR / "dsph_upper_limits_bb_Jprior.csv",
    "tautau": _MCDANIEL_DIR / "dsph_upper_limits_tau_Jprior.csv",
}

# --- Boddy 2019 velocity-dependent tables (produced by extract_boddy_limits) -
# Filename convention: boddy2019_<wave>_<channel>.txt
_BODDY_WAVES = {
    "boddy_swave": "swave",
    "boddy_pwave": "pwave",
    "boddy_dwave": "dwave",
    "boddy_somm": "sommerfeld",
}
_BODDY_CHANNELS = ("WW", "bb", "tautau")


def _boddy_file(wave_key: str, channel: str) -> Path:
    wave = _BODDY_WAVES[wave_key]
    ch = normalise_ann_channel(channel).lower()
    return _DATA_DIR / f"boddy2019_{wave}_{ch}.txt"


_SOURCE_LABELS = {
    "hoof": (
        "Hoof et al. 2020 Fig. 10 oneD frequentist 95% s-wave dSph limits "
        "(arXiv:1812.06986; zenodo 2612268)"
    ),
    "mcdaniel": (
        "McDaniel et al. 2024 stacked dSph s-wave limits "
        "(arXiv:2311.04982; figshare 24058650; Hoof fallback where absent)"
    ),
    "boddy_swave": "Boddy et al. 2019 s-wave effective-J dSph limits (arXiv:1909.13197)",
    "boddy_pwave": "Boddy et al. 2019 p-wave (sigma v ~ v^2) dSph limits (arXiv:1909.13197)",
    "boddy_dwave": "Boddy et al. 2019 d-wave (sigma v ~ v^4) dSph limits (arXiv:1909.13197)",
    "boddy_somm": "Boddy et al. 2019 Sommerfeld (Coulomb, sigma v ~ 1/v) dSph limits (arXiv:1909.13197)",
}

DSph_LIMIT_SOURCES = (
    "hoof",
    "mcdaniel",
    "boddy_swave",
    "boddy_pwave",
    "boddy_dwave",
    "boddy_somm",
)

# Sources whose stored sigma v is a velocity-dependent (halo-comparable) limit.
VELOCITY_DEPENDENT_SOURCES = ("boddy_pwave", "boddy_dwave", "boddy_somm")


def normalise_ann_channel(channel: str) -> str:
    key = str(channel).strip().lower().replace("+", "").replace("-", "").replace("_", "")
    if key in ("ww", "w"):
        return "WW"
    if key in ("bb", "bbar", "bbbar", "b"):
        return "bb"
    if key in ("tautau", "tau", "tauplustau", "tauplustauminus"):
        return "tautau"
    return str(channel).strip()


def normalise_limit_source(source: str = "hoof") -> str:
    key = str(source).strip().lower().replace("-", "_")
    aliases = {
        "default": "hoof",
        "hoof2020": "hoof",
        "hoof_et_al": "hoof",
        "mcdaniel2024": "mcdaniel",
        "mcdaniel_dwarfs": "mcdaniel",
        "boddy": "boddy_swave",
        "boddy2019": "boddy_swave",
        "boddy_s": "boddy_swave",
        "boddy_p": "boddy_pwave",
        "boddy_d": "boddy_dwave",
        "boddy_sommerfeld": "boddy_somm",
        "sommerfeld": "boddy_somm",
    }
    key = aliases.get(key, key)
    if key in DSph_LIMIT_SOURCES:
        return key
    raise ValueError(
        f"Unknown dSph limit source {source!r}. Choices: {DSph_LIMIT_SOURCES}"
    )


def _limit_file_for(channel: str, source: str) -> Path | None:
    ch = normalise_ann_channel(channel)
    src = normalise_limit_source(source)
    if src == "hoof":
        return _HOOF_FILES.get(ch)
    if src == "mcdaniel":
        # McDaniel only ships bb / tautau; fall back to Hoof for the rest (e.g. WW).
        return _MCDANIEL_FILES.get(ch) or _HOOF_FILES.get(ch)
    if src in _BODDY_WAVES:
        if ch not in _BODDY_CHANNELS:
            return None
        return _boddy_file(src, ch)
    return None


def resolved_limit_source(channel: str, source: str = "hoof") -> str:
    """Concrete source after McDaniel->Hoof fallback (Boddy has no fallback)."""
    ch = normalise_ann_channel(channel)
    src = normalise_limit_source(source)
    if src == "mcdaniel" and ch not in _MCDANIEL_FILES:
        return "hoof"
    return src


def is_velocity_dependent(source: str) -> bool:
    return normalise_limit_source(source) in VELOCITY_DEPENDENT_SOURCES


def available_dsph_channels(source: str = "hoof") -> tuple[str, ...]:
    src = normalise_limit_source(source)
    if src == "hoof":
        return tuple(_HOOF_FILES.keys())
    if src == "mcdaniel":
        return tuple(_MCDANIEL_FILES.keys())
    if src in _BODDY_WAVES:
        return tuple(ch for ch in _BODDY_CHANNELS if _boddy_file(src, ch).exists())
    return ()


def has_dsph_limit_table(channel: str, source: str = "hoof") -> bool:
    path = _limit_file_for(channel, source)
    return path is not None and path.exists()


def dsph_limit_source(channel: str | None = None, *, source: str = "hoof") -> str:
    src = normalise_limit_source(source)
    label = _SOURCE_LABELS[src]
    if channel is None:
        return label
    ch = normalise_ann_channel(channel)
    resolved = resolved_limit_source(ch, src)
    path = _limit_file_for(ch, src)
    if path is None:
        return f"{label}; no {ch} table configured"
    if src == "mcdaniel" and resolved == "hoof":
        return f"{label}; {ch} table: {path.name} (Hoof fallback)"
    return f"{label}; {ch} table: {path.name}"


@lru_cache(maxsize=None)
def load_dsph_limit_table(channel: str, source: str = "hoof") -> tuple[np.ndarray, np.ndarray]:
    ch = normalise_ann_channel(channel)
    src = normalise_limit_source(source)
    resolved = resolved_limit_source(ch, src)
    path = _limit_file_for(ch, src)
    if path is None:
        raise ValueError(f"No {src} dSph limit table configured for channel {channel!r}.")
    if not path.exists():
        raise FileNotFoundError(
            f"dSph limit table not found: {path}\n"
            f"  source={src!r} channel={ch!r}. "
            "Boddy velocity-dependent tables are produced by extract_boddy_limits.py."
        )

    if resolved == "mcdaniel":
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        mass = np.asarray(data["mass_GeV"], dtype=float)
        sigmav = np.asarray(data["sigmav_ul_cm3_s"], dtype=float)
    else:
        data = np.loadtxt(path, comments="#", dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"dSph limit table must have >=2 columns: {path}")
        mass = np.asarray(data[:, 0], dtype=float)
        sigmav = np.asarray(data[:, 1], dtype=float)

    finite = np.isfinite(mass) & np.isfinite(sigmav) & (mass > 0.0) & (sigmav > 0.0)
    if not np.any(finite):
        raise ValueError(f"dSph limit table has no finite positive rows: {path}")

    order = np.argsort(mass[finite])
    return mass[finite][order], sigmav[finite][order]


def dsph_upper_limit(
    m_chi_GeV: float,
    channel: str = "WW",
    source: str = "hoof",
    *,
    extrapolate: bool = False,
) -> float:
    """Interpolated dSph 95% CL upper limit on <sigma v> [cm^3/s].

    For velocity-dependent sources the returned value is the *halo-comparable*
    limit (see module docstring).  Out-of-range masses return ``np.nan`` unless
    ``extrapolate=True`` (log-log linear extrapolation, use with care).
    """
    ch = normalise_ann_channel(channel)
    mass, sigmav = load_dsph_limit_table(ch, normalise_limit_source(source))
    m = float(m_chi_GeV)
    lm = np.log(m)
    lmass = np.log(mass)
    if not extrapolate and (m < float(mass[0]) or m > float(mass[-1])):
        return np.nan
    return float(np.exp(np.interp(lm, lmass, np.log(sigmav))))


__all__ = [
    "DSph_LIMIT_SOURCES",
    "VELOCITY_DEPENDENT_SOURCES",
    "normalise_ann_channel",
    "normalise_limit_source",
    "resolved_limit_source",
    "is_velocity_dependent",
    "available_dsph_channels",
    "has_dsph_limit_table",
    "dsph_limit_source",
    "load_dsph_limit_table",
    "dsph_upper_limit",
]
