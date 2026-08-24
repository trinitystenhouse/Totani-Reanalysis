#!/usr/bin/env python3
r"""
Spectral distinctness of the 20 GeV halo from the GCE / MSP region.

Single-panel figure: the disk-excluded 20 GeV halo spectrum vs the disk-included
few-GeV, GCE-like component from the global fit, with a representative pulsar/MSP
spectrum (power law x exponential cutoff, E_c=2 GeV) and the E_c~1-4 GeV cutoff
band shaded. The 20 GeV peak sits an order of magnitude above the few-GeV GCE
peak and well above the pulsar/MSP cutoff, so the two cannot be the same
population and an MSP-like origin is disfavoured.

NOTE: the *spatial* distinctness argument is made by the model-independent
angular profile (morphology/plot_angular_profile_vs_models.py), which compares
the measured excess profile to NFW vs isotropic. This figure is the spectral
half only -- the earlier two-panel version's cartoon angular panel (assumed
template shapes, not data) has been retired in favour of the data-driven one.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

_HERE = Path(__file__).resolve().parent          # Totani_reanalysis
sys.path.insert(0, str(_HERE.parent))            # helpers.plot_style

# post-reorg data locations
HALO_TAB = _HERE / "figures/systematics/plots_global_systematics/baseline_rho2/mcmc_global_f_ml.txt"
GCE_TAB = _HERE / "figures/global_fit/plots_global_fig6_w_disk/mcmc_global_f_ml.txt"


def table(p: Path):
    a = np.genfromtxt(p, names=True, comments="#")
    return np.asarray(a["E_GeV"], float), np.asarray(a["nfw"], float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halo-tab", type=Path, default=HALO_TAB)
    ap.add_argument("--gce-tab", type=Path, default=GCE_TAB)
    ap.add_argument("--out-dir", type=Path, default=_HERE / "figures" / "paper_figures")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    import matplotlib.pyplot as plt
    try:
        from helpers.plot_style import apply_plot_style
        apply_plot_style("paper", base_fontsize=10, linewidth=1.5, n_colors=8, cmap_name="plasma")
    except Exception as exc:
        print("(style fallback:", exc, ")")
    HALO_C, GCE_C, MSP_C = "#1b9e77", "#d95f02", "0.45"

    Eh, yh = table(args.halo_tab)
    Eg, yg = table(args.gce_tab)
    mh, mg = yh > 0, yg > 0

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(Eh[mh], yh[mh], "o-", color=HALO_C, lw=2.0, ms=4.5,
            label="20 GeV halo (disk-excl.)")
    ax.plot(Eg[mg], yg[mg], "s--", color=GCE_C, lw=2.0, ms=4.5,
            label="GCE / few-GeV (disk-incl.)")
    E = np.geomspace(1.0, 300.0, 200)
    Gam, Ec = 1.3, 2.0
    msp = E ** (2 - Gam) * np.exp(-E / Ec)
    msp = msp / msp.max() * np.nanmax(yg[mg])
    ax.plot(E, msp, ":", color=MSP_C, lw=2.2, label=r"pulsar/MSP ($E_c=2$ GeV)")
    ax.axvspan(1.0, 4.0, color=MSP_C, alpha=0.10, label=r"MSP cutoff $E_c\sim1$--$4$ GeV")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"$E^2\,dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_ylim(np.nanmax(yh) * 3e-3, np.nanmax(yh) * 3)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = args.out_dir / f"f_halo_vs_gce_spectrum.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight"); paths.append(p)
    plt.close(fig)
    print("saved", paths[0])
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[0], args.copy_to / paths[0].name)
        print("copied ->", args.copy_to / paths[0].name)


if __name__ == "__main__":
    main()
