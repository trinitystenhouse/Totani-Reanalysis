#!/usr/bin/env python3
r"""
Produce velocity-dependent (p-/d-wave, Sommerfeld) dwarf-spheroidal limit tables
for the Totani 20 GeV halo tension scan, in the two-column format consumed by
``core.dsph_limits_multi``.

WHY THIS EXISTS
---------------
The pipeline's tension is

    tension = <sigma v>_halo / <sigma v>_dSph_UL ,

where both sides are *s-wave* (velocity-independent).  For a velocity-dependent
annihilation hypothesis the halo and the dwarfs sample different relative-velocity
distributions, so the s-wave dwarf limit is the wrong comparator.  This script
converts an s-wave dwarf limit into the **halo-comparable** velocity-dependent
limit, i.e. the upper limit on the halo-frame <sigma v> implied by the dwarf
non-detection under the chosen velocity law.  The scan can then use it unchanged.

DERIVATION (units of c throughout; n labels the partial wave)
------------------------------------------------------------
Write the velocity law as  sigma v = a_n * (v_rel/c)^{2n}  with
    n = 0 (s-wave), 1 (p-wave), 2 (d-wave);  Sommerfeld Coulomb is the n=-1/2 case
    sigma v = a_S * (v_rel/c)^{-1}.

The dwarf gamma-ray flux for a given (mass, channel) -- hence the Fermi-LAT 95%
flux upper limit -- is the same spectral template scaled by

    Phi  ∝  a_n * Jtilde_n ,   Jtilde_n ≡ ∫ rho^2 <(v_rel/c)^{2n}> dV dΩ .

Matching the *same* measured flux limit Phi_UL between the s-wave analysis (which
reports sigma_v0_UL with Jtilde_0 = J) and the n-wave hypothesis gives

    a_n_UL = sigma_v0_UL * J / Jtilde_n = sigma_v0_UL / <(v/c)^{2n}>_dSph ,

where <(v/c)^{2n}>_dSph ≡ Jtilde_n / J is the J-weighted dwarf velocity moment.
The halo-frame cross section under the same law is  <sigma v>_halo = a_n * u_h^n
with u_h ≡ <(v_rel/c)^2>_halo, so the halo-comparable dwarf limit is

    <sigma v>_halo_UL  =  sigma_v0_UL * <(v/c)^{2n}>_halo / <(v/c)^{2n}>_dSph .

Closed forms used here:
    s-wave :  factor = 1
    p-wave :  factor = u_h   / u_d                       (EXACT: Jtilde_1/J ≡ u)
    d-wave :  factor = (u_h  / u_d)^2 * (K_h / K_d)       (K = <v^4>/<v^2>^2 kurtosis)
    Somm   :  factor = sqrt(u_d / u_h)                    (Coulomb 1/v; STRENGTHENS)

with u_h = (v_halo/c)^2, u_d = (v_dwarf/c)^2.  For p-wave the factor is exact in
terms of the J-weighted moments; only the *numerical values* of u_h, u_d carry
modelling uncertainty.  For d-wave the gaussian kurtosis ratio defaults to 1
(K_h=K_d); supply Boddy moments for a rigorous number.

DATA / RIGOUR LADDER
--------------------
  --mode analytic   (default)  rescale by analytic velocity moments from
                               --v-halo-kms / --v-dwarf-kms.  Runs today; the
                               p-wave result is exact up to the moment values.
  --mode boddy      read Boddy et al. 2019 J-weighted moments from a CSV
                    (--boddy-csv) and use the real per-sample <(v/c)^{2n}>_dSph.
                    Get the data from:
                       https://github.com/apace7/J-Factor-Scaling
                       repo_1909_13197/Jfactors  (arXiv:1909.13197)
                    Expected CSV columns (one row per dwarf or one summary row):
                       name, u_dwarf_swave, u_dwarf_pwave, u_dwarf_dwave, u_dwarf_somm
                    where u_dwarf_* = Jtilde_n / J in units of c^2 (i.e. the
                    J-weighted <(v/c)^{2n}>).  If only log10 J columns are present
                    use --jfactor-csv mode (see _moments_from_jfactor_csv).

Outputs: data/boddy2019_<wave>_<channel>.txt  (two columns: mass[GeV], <sv>_UL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
if str(_REANALYSIS) not in sys.path:
    sys.path.insert(0, str(_REANALYSIS))

# Reuse the s-wave loader so we inherit Hoof/McDaniel tables and log-log interp.
from core.dsph_limits_multi import (  # noqa: E402
    load_dsph_limit_table,
    normalise_ann_channel,
    normalise_limit_source,
)

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _REANALYSIS / "data"

C_KMS = 299_792.458  # speed of light, km/s

# Defaults reproduce the verified briefing estimate u_d/u_h = (7/175)^2 = 1.6e-3
# (p-wave dwarf limit ~625x weaker -> tension /625).  REPLACE for publication:
#   v_halo  -> J/density-weighted <v_rel^2>^(1/2) over the ROI (Jeans/Eddington
#              from the same NFW you fit; varies with |b|),
#   v_dwarf -> Boddy J-weighted dwarf moment (use --mode boddy).
DEFAULT_V_HALO_KMS = 175.0
DEFAULT_V_DWARF_KMS = 7.0

_WAVES = ("swave", "pwave", "dwave", "sommerfeld")


def velocity_factor(
    wave: str,
    u_halo: float,
    u_dwarf: float,
    *,
    kurt_halo: float = 1.0,
    kurt_dwarf: float = 1.0,
) -> float:
    """Halo-comparable rescaling factor (see module docstring)."""
    r = u_halo / u_dwarf
    if wave == "swave":
        return 1.0
    if wave == "pwave":
        return r
    if wave == "dwave":
        return r * r * (kurt_halo / kurt_dwarf)
    if wave == "sommerfeld":
        return np.sqrt(u_dwarf / u_halo)
    raise ValueError(f"Unknown wave {wave!r}; choices {_WAVES}")


def _moments_from_jfactor_csv(path: Path) -> dict[str, float]:
    """Derive J-weighted <(v/c)^{2n}>_dSph from a Boddy effective-J CSV.

    Accepts either:
      (a) columns u_dwarf_swave,u_dwarf_pwave,u_dwarf_dwave,u_dwarf_somm  (already
          the moments Jtilde_n/J in c^2 units), or
      (b) log10 effective-J columns logJ_swave,logJ_pwave,logJ_dwave,logJ_somm,
          in which case the moment ratio is 10^(logJ_n - logJ_swave) (the velocity
          weighting that distinguishes the effective J from the s-wave J).
    When multiple dwarfs are present the *strongest* (smallest moment, i.e. most
    conservative limit) row is used, matching a single-best-dwarf bound; replace
    with the published stacked limit if you have it.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    cols = set(data.dtype.names or ())
    out: dict[str, float] = {}
    if {"u_dwarf_pwave"} <= cols:
        getter = lambda w: np.atleast_1d(data[f"u_dwarf_{ 'somm' if w=='sommerfeld' else w}"]).astype(float)  # noqa: E731
        for w in _WAVES:
            vals = getter(w)
            out[w] = float(np.nanmin(vals)) if w != "swave" else 1.0
    elif {"logJ_pwave", "logJ_swave"} <= cols:
        s = np.atleast_1d(data["logJ_swave"]).astype(float)
        for w in _WAVES:
            key = "logJ_" + ("somm" if w == "sommerfeld" else w)
            if w == "swave":
                out[w] = 1.0
            elif key in cols:
                ratio = 10.0 ** (np.atleast_1d(data[key]).astype(float) - s)
                out[w] = float(np.nanmin(ratio))
            else:
                out[w] = np.nan
    else:
        raise ValueError(
            f"{path}: need either u_dwarf_* columns or logJ_* columns; got {sorted(cols)}"
        )
    return out


def build_tables(
    channels: list[str],
    waves: list[str],
    s_source: str,
    v_halo_kms: float,
    v_dwarf_kms: float,
    boddy_moments: dict[str, float] | None,
    out_dir: Path,
    extrapolate: bool = False,
    halo_moments: dict[str, float] | None = None,
) -> list[Path]:
    u_halo = (v_halo_kms / C_KMS) ** 2
    u_dwarf_default = (v_dwarf_kms / C_KMS) ** 2
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for ch in channels:
        ch_n = normalise_ann_channel(ch)
        mass, sv_swave = load_dsph_limit_table(ch_n, normalise_limit_source(s_source))
        for wave in waves:
            if boddy_moments is not None and wave != "swave":
                u_d_eff = boddy_moments.get(wave, np.nan)
                if halo_moments is not None:
                    # Symmetric, shape-consistent: factor = <m_n>_halo / <m_n>_dSph,
                    # both sides the same relative-velocity moment (compute_u_halo.py).
                    hk = {"pwave": "pwave", "dwave": "dwave", "sommerfeld": "somm"}[wave]
                    factor = halo_moments[hk] / u_d_eff
                # else: analytic-halo closures (v_halo_kms), dwarf moments from Boddy.
                elif wave == "pwave":
                    factor = u_halo / u_d_eff
                elif wave == "dwave":
                    # u_d_eff here is <v^4>_dSph in c^4; numerator <v^4>_halo ~ u_halo^2
                    factor = (u_halo ** 2) / u_d_eff
                elif wave == "sommerfeld":
                    # u_d_eff here is <1/v>_dSph in 1/c; numerator <1/v>_halo ~ 1/sqrt(u_halo)
                    factor = (1.0 / np.sqrt(u_halo)) / u_d_eff
                else:
                    factor = 1.0
            else:
                factor = velocity_factor(wave, u_halo, u_dwarf_default)

            sv_out = sv_swave * factor
            path = out_dir / f"boddy2019_{wave}_{ch_n.lower()}.txt"
            _write_table(path, mass, sv_out, ch_n, wave, s_source,
                         v_halo_kms, v_dwarf_kms, factor,
                         used_boddy=boddy_moments is not None and wave != "swave")
            written.append(path)
    return written


def _write_table(path, mass, sv, channel, wave, s_source,
                 v_halo, v_dwarf, factor, used_boddy):
    hdr = [
        f"# Boddy et al. 2019 (arXiv:1909.13197) {wave} dSph limit, channel {channel}.",
        f"# HALO-COMPARABLE limit on <sigma v>_halo [cm^3/s] (see extract_boddy_limits.py).",
        f"# Built by rescaling s-wave source '{s_source}' by velocity factor = {factor:.4e}.",
    ]
    if used_boddy:
        hdr.append("# Velocity moments: Boddy J-weighted (from --boddy-csv).")
    else:
        hdr.append(f"# Velocity moments: analytic, v_halo={v_halo} km/s, v_dwarf={v_dwarf} km/s.")
        hdr.append("# NOTE: provisional analytic moments -- swap to --mode boddy for publication.")
    hdr.append("# Columns: m_chi [GeV] | <sigma v>_halo upper limit [cm^3/s]")
    with open(path, "w") as fh:
        fh.write("\n".join(hdr) + "\n")
        for m, s in zip(mass, sv):
            fh.write(f"{m:.6e} {s:.6e}\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", nargs="+", default=["bb", "tautau", "WW"])
    p.add_argument("--waves", nargs="+", default=list(_WAVES), choices=_WAVES)
    p.add_argument("--s-source", default="hoof",
                   help="s-wave base table to rescale (hoof|mcdaniel|boddy_swave).")
    p.add_argument("--mode", choices=["analytic", "boddy"], default="analytic")
    p.add_argument("--boddy-csv", type=Path, default=None,
                   help="CSV of Boddy J-weighted moments (required for --mode boddy).")
    p.add_argument("--halo-csv", type=Path, default=None,
                   help="CSV of halo moments (u_halo_pwave,u_halo_dwave,u_halo_somm) from "
                        "compute_u_halo.py. If given, factor = <m_n>_halo/<m_n>_dSph (preferred "
                        "over the analytic --v-halo-kms closures).")
    p.add_argument("--v-halo-kms", type=float, default=DEFAULT_V_HALO_KMS)
    p.add_argument("--v-dwarf-kms", type=float, default=DEFAULT_V_DWARF_KMS)
    p.add_argument("--out-dir", type=Path, default=_DATA_DIR)
    args = p.parse_args(argv)

    boddy_moments = None
    if args.mode == "boddy":
        if args.boddy_csv is None or not args.boddy_csv.exists():
            p.error("--mode boddy requires an existing --boddy-csv "
                    "(download from github.com/apace7/J-Factor-Scaling, repo_1909_13197).")
        boddy_moments = _moments_from_jfactor_csv(args.boddy_csv)
        print(f"[boddy] J-weighted moments: {boddy_moments}")

    halo_moments = None
    if args.halo_csv is not None:
        if not args.halo_csv.exists():
            p.error(f"--halo-csv not found: {args.halo_csv} (run compute_u_halo.py first).")
        hd = np.genfromtxt(args.halo_csv, delimiter=",", names=True, dtype=float)
        halo_moments = {
            "pwave": float(hd["u_halo_pwave"]),
            "dwave": float(hd["u_halo_dwave"]),
            "somm": float(hd["u_halo_somm"]),
        }
        print(f"[halo] moments: {halo_moments}")

    written = build_tables(
        channels=[normalise_ann_channel(c) for c in args.channels],
        waves=args.waves,
        s_source=args.s_source,
        v_halo_kms=args.v_halo_kms,
        v_dwarf_kms=args.v_dwarf_kms,
        boddy_moments=boddy_moments,
        out_dir=args.out_dir,
        halo_moments=halo_moments,
    )
    u_h = (args.v_halo_kms / C_KMS) ** 2
    u_d = (args.v_dwarf_kms / C_KMS) ** 2
    if boddy_moments is not None:
        if halo_moments is not None:
            print("[extract_boddy_limits] mode=boddy  halo moments from --halo-csv (Jeans)")
        else:
            print(f"[extract_boddy_limits] mode=boddy  u_halo={u_h:.3e} (analytic, v_halo={args.v_halo_kms} km/s)")
        print("  velocity factors (halo-comparable, Boddy J-weighted dwarf moments):")
        for w in args.waves:
            if halo_moments is not None and w != "swave":
                hk = {"pwave": "pwave", "dwave": "dwave", "sommerfeld": "somm"}[w]
                fac = halo_moments[hk] / boddy_moments[w]
            elif w == "pwave":
                fac = u_h / boddy_moments["pwave"]
            elif w == "dwave":
                fac = (u_h ** 2) / boddy_moments["dwave"]
            elif w == "sommerfeld":
                fac = (1.0 / np.sqrt(u_h)) / boddy_moments["sommerfeld"]
            else:
                fac = 1.0
            print(f"    {w:11s}: {fac:.3e}")
    else:
        print(f"[extract_boddy_limits] mode=analytic  u_halo={u_h:.3e}  u_dwarf={u_d:.3e}  u_h/u_d={u_h/u_d:.1f}")
        print("  velocity factors (halo-comparable):")
        for w in args.waves:
            print(f"    {w:11s}: {velocity_factor(w, u_h, u_d):.3e}")
    print(f"[extract_boddy_limits] wrote {len(written)} tables to {args.out_dir}")
    for w in written:
        print("   ", w)


if __name__ == "__main__":
    main()
