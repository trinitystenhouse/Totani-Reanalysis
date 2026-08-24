#!/usr/bin/env python3
r"""
Refit the annihilation mass to each variant's halo spectrum over a RESTRICTED
energy range, to separate the 20 GeV halo from any low-energy shoulder.

Why this exists
---------------
Across the fourteen-model interstellar-emission grid the halo spectral PEAK does
not move: it sits in the 20.76 GeV bin in every variant, interpolating to 17-21
GeV throughout. What does move is the flux below about 5 GeV. Integrating 1.5 to
4.4 GeV and dividing by the 12 to 36 GeV signal band gives 0.05 for the baseline
and Lorimer models, rising monotonically with how centrally concentrated the
cosmic-ray source distribution is, to 0.42-0.48 for the supernova-remnant family
and 0.57 for OB stars, the last at 32 sigma.

That shoulder is almost perfectly anti-correlated with the fitted mass
(r = -0.944), and completely uncorrelated with the fit quality (r = -0.021),
because a lower-mass b-bbar spectrum naturally carries more low-energy flux and
therefore absorbs a shoulder without degrading chi2. So chi2 cannot be used to
detect the contamination, and the fitted mass is not a robust quantity when it
is present.

This script refits with the low-energy bins excluded. If the mass becomes stable
across variants once the shoulder is removed, then the mass spread, and hence the
widened upper edge of the dSph tension band, is driven by the shoulder rather
than by the halo. If the mass still spreads, the spectral shape genuinely
differs between diffuse models and the wider band stands.

It does not modify propagate_systematics.py, and writes to its own output file.

Usage
-----
    python3 systematics/refit_mass_energy_range.py
    python3 systematics/refit_mass_energy_range.py --e-min 7 10 12 --channel bb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))

from shared.paths import GLOBAL_SYSTEMATICS, REANALYSIS, REANALYSIS_RESULTS  # noqa: E402

from propagate_systematics import (  # noqa: E402
    load_variant, _SYSTEMATICS_PLOT_ROOT, _LEGACY_PLOT_ROOT,
)
from core.spectral_reshaping import pppc_energy_flux_template, smooth_nfw_sigma_v_from_norm  # noqa: E402
from core.dsph_limits_multi import dsph_upper_limit  # noqa: E402


_SHAPE_CACHE: dict = {}
_LIMIT_CACHE: dict = {}


def shapes_for(E, masses, channel, table_path):
    """PPPC shapes depend only on (E grid, mass, channel), never on the variant.

    Every variant shares the same thirteen-bin energy grid, so computing these
    once and reusing them turns 80 masses x 20 variants x 3 energy cuts into 80
    table lookups. Without this the script does not finish in reasonable time.
    """
    key = (tuple(np.round(E, 6)), channel, str(table_path))
    if key not in _SHAPE_CACHE:
        out = {}
        for m in masses:
            try:
                out[float(m)] = pppc_energy_flux_template(
                    E, m, channel=channel, primary="gamma",
                    table_path=table_path, normalise=False)
            except Exception:
                continue
        _SHAPE_CACHE[key] = out
    return _SHAPE_CACHE[key]


def fit_range(E, phi, err, channel, masses, source, table_path, e_min, e_max):
    """As propagate_systematics.fit_one, but restricted to e_min <= E <= e_max."""
    mask = (np.isfinite(phi) & np.isfinite(err) & (phi > 0) & (err > 0)
            & (E >= e_min) & (E <= e_max))
    best = dict(chi2=np.inf, nbins=int(mask.sum()))
    if mask.sum() < 3:
        return best
    cache = shapes_for(E, masses, channel, table_path)
    for m, shape in cache.items():
        if not np.any(shape[mask] > 0):
            continue
        w = 1.0 / err[mask] ** 2
        A = np.sum(w * shape[mask] * phi[mask]) / np.sum(w * shape[mask] ** 2)
        if not np.isfinite(A) or A <= 0:
            continue
        chi2 = float(np.sum(((A * shape[mask] - phi[mask]) / err[mask]) ** 2))
        if chi2 < best["chi2"]:
            sv = smooth_nfw_sigma_v_from_norm(A, m)
            lk = (float(m), channel, source)
            if lk not in _LIMIT_CACHE:
                _LIMIT_CACHE[lk] = dsph_upper_limit(m, channel, source)
            lim = _LIMIT_CACHE[lk]
            best = dict(chi2=chi2, ndof=int(mask.sum()) - 1, nbins=int(mask.sum()),
                        m=float(m), sv=float(sv),
                        tension=float(sv / lim) if np.isfinite(lim) and lim > 0 else np.nan)
    return best


def shoulder_fraction(E, phi):
    lo = phi[(E >= 1.5) & (E <= 4.4)].sum()
    bd = phi[(E >= 12.0) & (E <= 36.0)].sum()
    return lo / bd if bd != 0 else np.nan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path,
                   default=REANALYSIS / "mcmc" / "configs" / "systematics" / "manifest.json")
    p.add_argument("--global-root", type=Path, default=GLOBAL_SYSTEMATICS)
    p.add_argument("--baseline", default="baseline_rho2")
    p.add_argument("--channel", default="bb")
    p.add_argument("--dsph-source", default="hoof")
    p.add_argument("--m-min", type=float, default=20.0)
    p.add_argument("--m-max", type=float, default=2000.0)
    p.add_argument("--n-mass", type=int, default=200)
    p.add_argument("--pppc-table", default=None)
    p.add_argument("--e-min", type=float, nargs="+", default=[0.0, 7.0, 10.0],
                   help="lower energy cuts to scan, in GeV. 0 reproduces the full-range fit.")
    p.add_argument("--e-max", type=float, default=1e6)
    p.add_argument("--groups", nargs="+",
                   default=["baseline", "diffuse", "diffuse_ics", "diffuse_grid_bubbles"],
                   help="systematics groups to include, default is the foreground leg.")
    p.add_argument("--out", type=Path,
                   default=REANALYSIS_RESULTS / "global-systematics" / "mass_vs_energy_range.txt")
    a = p.parse_args(argv)

    variants = json.loads(a.manifest.read_text()).get("variants", [])
    masses = np.geomspace(a.m_min, a.m_max, a.n_mass)

    def table(name):
        for r in dict.fromkeys((_SYSTEMATICS_PLOT_ROOT, _LEGACY_PLOT_ROOT, a.global_root)):
            q = Path(r) / name / "mcmc_global_f_ml.txt"
            if q.exists():
                return q
        return None

    keep = [it for it in variants if it.get("group") in a.groups]
    results, lines = {}, []
    for emin in a.e_min:
        rows = []
        for it in keep:
            t = table(it["name"])
            if t is None:
                continue
            E, phi, err = load_variant(t)
            b = fit_range(E, phi, err, a.channel, masses, a.dsph_source,
                          a.pppc_table, emin, a.e_max)
            if not np.isfinite(b["chi2"]):
                continue
            b["shoulder"] = shoulder_fraction(E, phi)
            rows.append((it["name"], it.get("group"), b))
        results[emin] = rows

    hdr = (f"{'variant':40s} {'nbins':>5s} {'m_chi':>7s} {'<sv>':>11s} "
           f"{'tension':>8s} {'chi2/ndof':>9s} {'shoulder':>8s}")
    for emin in a.e_min:
        rows = results[emin]
        if not rows:
            continue
        ms = np.array([b["m"] for _, _, b in rows])
        base = next((b["m"] for n, _, b in rows if n == a.baseline), np.nan)
        title = (f"\n=== E >= {emin:g} GeV "
                 f"({'full range' if emin <= 0 else 'shoulder excluded'}) ===")
        block = [title, hdr]
        for n, g, b in sorted(rows, key=lambda r: r[2]["m"]):
            block.append(f"{n:40s} {b['nbins']:5d} {b['m']:7.0f} {b['sv']:11.3e} "
                         f"{b['tension']:8.2f} {b['chi2']/max(b['ndof'],1):9.2f} "
                         f"{b['shoulder']:8.2f}")
        spread = ms.max() / ms.min()
        block.append(f"  mass range {ms.min():.0f} to {ms.max():.0f} GeV, "
                     f"spread x{spread:.2f}, baseline {base:.0f} GeV")
        te = np.array([b["tension"] for _, _, b in rows])
        tb = next((b["tension"] for n, _, b in rows if n == a.baseline), np.nan)
        block.append(f"  tension envelope x[{te.min()/tb:.3f}, {te.max()/tb:.3f}] "
                     f"relative to baseline")
        lines += block
        print("\n".join(block))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(
        [f"# Mass refit versus lower energy cut, channel={a.channel}, dsph={a.dsph_source}"]
        + lines) + "\n")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
