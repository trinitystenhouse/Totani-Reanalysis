"""
Backward-compatible shim.

The dwarf-spheroidal limit logic now lives in ``core.dsph_limits_multi`` (which
adds velocity-dependent Boddy 2019 sources and returns ``np.nan`` -- not
``np.inf`` -- for out-of-range masses).  This module re-exports that API under
the original names so existing imports keep working:

    from core.dsph_limits import dsph_upper_limit, dsph_limit_source

Both now accept an optional ``source=`` keyword (default "hoof"), so callers can
opt into "boddy_pwave"/"boddy_dwave"/"boddy_somm"/"mcdaniel" without changing
their import.  The np.inf -> np.nan change is behaviour-preserving for callers
that guard with ``np.isfinite(...)`` (np.isfinite is False for both inf and nan).
"""

from __future__ import annotations

from core.dsph_limits_multi import (  # noqa: F401
    DSph_LIMIT_SOURCES,
    VELOCITY_DEPENDENT_SOURCES,
    available_dsph_channels,
    dsph_limit_source,
    dsph_upper_limit,
    has_dsph_limit_table,
    is_velocity_dependent,
    load_dsph_limit_table,
    normalise_ann_channel,
    normalise_limit_source,
    resolved_limit_source,
)

# Legacy aliases that older modules referenced.
DSph_LIMIT_SOURCES = DSph_LIMIT_SOURCES

__all__ = [
    "DSph_LIMIT_SOURCES",
    "VELOCITY_DEPENDENT_SOURCES",
    "available_dsph_channels",
    "dsph_limit_source",
    "dsph_upper_limit",
    "has_dsph_limit_table",
    "is_velocity_dependent",
    "load_dsph_limit_table",
    "normalise_ann_channel",
    "normalise_limit_source",
    "resolved_limit_source",
]
