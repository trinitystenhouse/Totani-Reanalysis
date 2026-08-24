#!/usr/bin/env python3
r"""
Compare the rho^2 halo spectrum across the four ROI quadrants (NE/NW/SE/SW), for
both disk-excluded and disk-included fits.

A genuinely spherical, isotropic 20 GeV halo must give a consistent spectrum in
all four quadrants. Asymmetries point instead to localised foreground structure
(Loop I is northern; the Fermi bubbles straddle the centre; the disk dominates
the low-|b| quadrants when included). The disk-excluded panel is the science
configuration; the disk-included panel shows how much each quadrant is pulled by
the few-GeV inner-Galaxy component.

Reads figures/plots_global_systematics/quad_<Q>_<excl|incl>/mcmc_global_f_ml.txt
(produced by run_global_from_config.py on the quad_* configs).

Usage (from the repo root):
    python compare_quadrants.py
    python compare_quadrants.py --copy-to <overleaf>/paper_plots
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06
sys.path.insert(0, str(_REPO))
SYST = _REPO / "Totani_paper_check" / "figures" / "plots_global_systematics"
QUADS = ["NE", "NW", "SE", "SW"]
PEAK_LO, PEAK_HI = 10.0, 40.0


def load(variant):
    t = SYST / variant / "mcmc_global_f_ml.txt"
    if not t.exists():
        return None
    a = np.genfromtxt(t, names=True, comments="#")
    E = np.asarray(a["E_GeV"], float)
    y = np.asarray(a["nfw"], float)
    lo = np.asarray(a["nfw_err_lo"], float) if "nfw_err_lo" in a.dtype.names else np.zeros_like(y)
    hi = np.asarray(a["nfw_err_hi"], float) if "nfw_err_hi" in a.dtype.names else np.zeros_like(y)
    return dict(E=E, y=y, lo=lo, hi=hi, sig=0.5 * (lo + hi))


def peak(d):
    m = (d["E"] >= PEAK_LO) & (d["E"] <= PEAK_HI)
    if not np.any(m):
        return np.nan, np.nan, np.nan
    k = np.nanargmax(np.where(m, d["y"], -np.inf))
    return d["E"][k], d["y"][k], d["sig"][k]


def report(state, label):
    print(f"\n=== {label} ===")
    ds = {q: load(f"quad_{q}_{state}") for q in QUADS}
    have = {q: d for q, d in ds.items() if d is not None}
    if not have:
        print("  [not run yet]")
        return ds
    peaks = {}
    for q in QUADS:
        d = ds[q]
        if d is None:
            print(f"  {q}: [missing]"); continue
        Ep, yp, sp = peak(d)
        peaks[q] = (yp, sp)
        print(f"  {q}: peak {yp:.3e} +/- {sp:.1e} at {Ep:.1f} GeV")
    if len(peaks) == 4:
        ys = np.array([peaks[q][0] for q in QUADS])
        ss = np.array([peaks[q][1] for q in QUADS])
        spread = (np.nanmax(ys) - np.nanmin(ys)) / np.hypot(np.nanmax(ss), np.nanmin(ss))
        ratio = np.nanmax(ys) / np.nanmin(ys)
        hi_q = QUADS[int(np.nanargmax(ys))]; lo_q = QUADS[int(np.nanargmin(ys))]
        print(f"  -> peak max/min ratio = {ratio:.2f} ({hi_q}/{lo_q}); "
              f"spread = {spread:.2f} sigma  "
              f"{'(consistent)' if spread < 2 else '(ASYMMETRIC)'}")
    return ds


def make_plot(ds_excl, ds_incl, out_dir):
    try:
        from helpers.plot_style import apply_plot_style, get_cmap_colors
        apply_plot_style("paper", base_fontsize=10, linewidth=1.4, n_colors=8, cmap_name="plasma")
        cols = get_cmap_colors(cmap_name="plasma", n=4, start=0.12, end=0.85)
    except Exception:
        cols = ["C0", "C1", "C2", "C3"]
    qcol = dict(zip(QUADS, cols))
    qmark = dict(NE="o", NW="s", SE="^", SW="v")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # explicit y-limits from all finite data (avoids NaN autoscale with shared y)
    allvals = []
    for ds in (ds_excl, ds_incl):
        for d in ds.values():
            if d is None:
                continue
            yv = np.asarray(d["y"], float)
            ev = np.asarray(d["hi"], float)
            allvals.append(yv[np.isfinite(yv)])
            allvals.append((yv + np.where(np.isfinite(ev), ev, 0.0))[np.isfinite(yv)])
    allvals = np.concatenate(allvals) if allvals else np.array([0.0, 1e-4])
    ymax = float(np.nanmax(allvals)) if allvals.size else 1e-4
    ymin = float(np.nanmin(allvals)) if allvals.size else 0.0
    pad = 0.1 * (ymax - ymin if ymax > ymin else abs(ymax) or 1e-4)
    ylim = (ymin - pad, ymax + pad)

    fig, (axE, axI) = plt.subplots(1, 2, figsize=(7.1, 3.2), sharey=True)
    for ax, ds, title in [(axE, ds_excl, "(a) disk excluded"), (axI, ds_incl, "(b) disk included")]:
        for i, q in enumerate(QUADS):
            d = ds.get(q)
            if d is None:
                continue
            E = np.asarray(d["E"], float)
            y = np.asarray(d["y"], float)
            lo = np.where(np.isfinite(d["lo"]) & (d["lo"] >= 0), d["lo"], 0.0)
            hi = np.where(np.isfinite(d["hi"]) & (d["hi"] >= 0), d["hi"], 0.0)
            ok = np.isfinite(E) & np.isfinite(y) & (E > 0)
            if not np.any(ok):
                continue
            ax.errorbar(E[ok] * (0.94 + 0.04 * i), y[ok], yerr=[lo[ok], hi[ok]],
                        fmt=qmark[q], color=qcol[q], ms=3.6, capsize=1.8, lw=1.2,
                        elinewidth=0.8, label=q)
        ax.axhline(0.0, color="0.5", lw=0.7, ls=":")
        ax.axvline(20.0, color="0.8", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlim(1, 1000)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_title(title, fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.13, which="both")
    axE.set_ylabel(r"NFW halo flux $E^2 dN/dE$" + "\n" + r"[MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=8.5)
    axE.legend(fontsize=8, frameon=False, ncol=2, title="quadrant", title_fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"f_quadrant_halo.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight"); paths.append(p)
    plt.close(fig)
    print(f"\nsaved {paths[0]}")
    return paths[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "Totani_paper_check" / "figures" / "paper_figures")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    ds_excl = report("excl", "disk EXCLUDED (science configuration)")
    ds_incl = report("incl", "disk INCLUDED")
    if all(d is None for d in {**ds_excl, **ds_incl}.values()):
        print("\nNothing to plot yet — run the quad_* configs first.")
        return
    p = make_plot(ds_excl, ds_incl, args.out_dir)
    if args.copy_to and p is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, args.copy_to / p.name)
        print("copied ->", args.copy_to / p.name)


if __name__ == "__main__":
    main()
