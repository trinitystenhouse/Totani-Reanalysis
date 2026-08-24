#!/usr/bin/env python3
r"""
Generate the three result figures referenced by the paper (main.tex), in the
shared "paper" plot style, as PDFs:

  f1_cellwise_spectrum.pdf          -- cellwise 10deg NFW-rho^2 halo (this work),
                                       reproducing the Totani halo.
  f2_global_vs_totani_profiles.pdf  -- global (pixel-level, this work) vs cellwise
                                       for gamma = 2.5, 2, 1.
  f3_disk_toggle.pdf                -- global gamma=2 halo with the disk EXCLUDED
                                       (20 GeV halo) vs INCLUDED (few-GeV peak).

Data provenance (validated tables already on disk):
  cellwise (Totani-proxy) : plots_fig{5,6,7}_10deg/mcmc_fig{N}_f_ml.txt
  global  (this work)      : plots_global_fig{5,6,7}/mcmc_global_f_ml.txt
  global, disk included    : plots_global_fig6_w_disk/mcmc_global_f_ml.txt
The "nfw" column is the halo E^2 dN/dE at the poles [MeV cm^-2 s^-1 sr^-1].
The cellwise fit is this work's reproduction of Totani's method and is used as the
Totani reference throughout the pipeline (no separate digitised published curve exists).

Run: python make_paper_figures.py   [--copy-to /path/to/overleaf/paper_plots]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent           # .../Totani_paper_check/figures
_DMP = _HERE.parent.parent                          # repository root
sys.path.insert(0, str(_DMP))

# profile -> (global dir, cellwise dir, cellwise table tag)
PROFILES = {
    2.5: ("plots_global_fig5", "plots_fig5_10deg", 5),
    2.0: ("plots_global_fig6", "plots_fig6_10deg", 6),
    1.0: ("plots_global_fig7", "plots_fig7_10deg", 7),
}
DISK_INCL_DIR = "plots_global_fig6_w_disk"   # gamma=2, disk included (few-GeV peak)

GLOBAL_C = "#1b1bbf"     # this work, global pixel-level
CELL_C = "#d95f02"       # cellwise (Totani-proxy)
DISK_C = "#6a3d9a"       # disk-included few-GeV component
YLABEL = r"$E^2\,dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
XLABEL = r"$E$ [GeV]"


def _load(table: Path):
    a = np.genfromtxt(table, names=True, comments="#")
    E = np.asarray(a["E_GeV"], float)
    y = np.asarray(a["nfw"], float)
    lo = np.asarray(a["nfw_err_lo"], float) if "nfw_err_lo" in a.dtype.names else np.zeros_like(y)
    hi = np.asarray(a["nfw_err_hi"], float) if "nfw_err_hi" in a.dtype.names else np.zeros_like(y)
    return E, y, np.vstack([lo, hi])


def cellwise(fig_tag):
    return _load(_HERE / f"plots_fig{fig_tag}_10deg" / f"mcmc_fig{fig_tag}_f_ml.txt")


def global_fit(dirname):
    return _load(_HERE / dirname / "mcmc_global_f_ml.txt")


def _style():
    try:
        from helpers.plot_style import apply_plot_style
        apply_plot_style("paper", base_fontsize=10, linewidth=1.5, n_colors=8, cmap_name="plasma")
    except Exception as exc:  # pragma: no cover
        print("  (style fallback:", exc, ")")


def _decorate(ax, *, title=None):
    ax.set_xscale("log")
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.axvline(20.0, color="0.75", lw=0.8, ls=":", zorder=0)
    ax.grid(True, which="both", alpha=0.15)
    if title:
        ax.set_title(title, fontsize=9)


def fig1(outdir):
    import matplotlib.pyplot as plt
    E, y, err = cellwise(6)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.errorbar(E, y, yerr=err, fmt="o-", color=CELL_C, lw=1.8, ms=4.5, capsize=2,
                label=r"cellwise $10^\circ$ (this work)")
    _decorate(ax)
    ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "f1_cellwise_spectrum")


def fig2(outdir):
    import matplotlib.pyplot as plt
    order = [(2.5, r"NFW-$\rho^{2.5}$"), (2.0, r"NFW-$\rho^{2}$"), (1.0, r"NFW-$\rho^{1}$")]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 7.2), sharex=True)
    for ax, (g, lab) in zip(axes, order):
        gdir, _, ctag = PROFILES[g]
        Ec, yc, ec = cellwise(ctag)
        Eg, yg, eg = global_fit(gdir)
        ax.errorbar(Ec, yc, yerr=ec, fmt="s--", color=CELL_C, lw=1.5, ms=4, capsize=2,
                    label="cellwise (Totani-style)")
        ax.errorbar(Eg, yg, yerr=eg, fmt="o-", color=GLOBAL_C, lw=1.7, ms=4, capsize=2,
                    label="global (this work)")
        _decorate(ax, title=lab)
        ax.set_ylabel(YLABEL, fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    axes[-1].set_xlabel(XLABEL)
    return _save(fig, outdir, "f2_global_vs_totani_profiles")


def fig3(outdir):
    import matplotlib.pyplot as plt
    Ex, yx, ex = global_fit(PROFILES[2.0][0])    # disk excluded -> 20 GeV halo
    Ei, yi, ei = global_fit(DISK_INCL_DIR)        # disk included -> few-GeV peak
    Ec, yc, ec = cellwise(6)                        # cellwise reference
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.errorbar(Ec, yc, yerr=ec, fmt="s:", color=CELL_C, lw=1.3, ms=3.5, capsize=2,
                alpha=0.8, label="cellwise (Totani-style)")
    ax.errorbar(Ex, yx, yerr=ex, fmt="o-", color=GLOBAL_C, lw=1.8, ms=4.5, capsize=2,
                label=r"global, disk excl. (20 GeV halo)")
    ax.errorbar(Ei, yi, yerr=ei, fmt="^-", color=DISK_C, lw=1.8, ms=4.5, capsize=2,
                label=r"global, disk incl. (few-GeV)")
    _decorate(ax)
    ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
    ax.legend(fontsize=7.5, frameon=False)
    return _save(fig, outdir, "f3_disk_toggle")


def _save(fig, outdir, stem):
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("saved", paths[0])
    return paths[0]


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=_HERE / "paper_figures")
    ap.add_argument("--copy-to", type=Path, default=None,
                    help="also copy the three PDFs into this directory (e.g. overleaf paper_plots)")
    args = ap.parse_args(argv)

    _style()
    pdfs = [fig1(args.out_dir), fig2(args.out_dir), fig3(args.out_dir)]

    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for p in pdfs:
            shutil.copy2(p, args.copy_to / p.name)
            print("copied ->", args.copy_to / p.name)


if __name__ == "__main__":
    main()
