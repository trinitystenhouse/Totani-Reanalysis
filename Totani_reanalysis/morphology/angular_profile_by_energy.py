#!/usr/bin/env python3
r"""
Fit the excess ring profile SEPARATELY IN EACH ENERGY BIN and scan the preferred
NFW inner power.

Why
---
Splitting the ROI into latitude annuli showed that the recovered halo is not the
same spectral object everywhere. At |b| = 10-20 deg the spectrum rises to a peak
in the 20.76 GeV bin at 12.8 sigma. At |b| = 40-60 deg it is roughly flat from
1.5 to 12 GeV and the 20.76 GeV bin is only 3.7 sigma, lower than the 7.28 GeV
bin. The peak-bin normalisation, which is quoted at the poles and should
therefore be annulus-independent for a correct template, differs between the two
by 4.6 sigma.

That is a morphology statement inferred from two annuli. This script tests it
directly and with far more leverage, by fitting the ring profile in each energy
bin and asking which NFW power the data prefer as a function of energy.

A genuine annihilation halo has an energy-INDEPENDENT spatial profile: the
morphology is set by the density field, not by the spectrum. Di Mauro (2021)
verified exactly this for the GCE up to 30 GeV. So:

  preferred power roughly constant with energy
      consistent with a single spatial component. The latitude-band result is
      then a low-significance artefact of narrow annuli and should not go in
      the paper.

  preferred power varies systematically with energy
      more than one spatial component is present, and the halo template is
      absorbing a second, differently distributed one. This bears directly on
      the spherical-halo interpretation and must be reported.

Reuses excess_map, ring_profile, fit_amp_offset and nfw_los from
plot_angular_profile_vs_models.py unchanged, so the method, binning, weighting
and amplitude-plus-offset fit are identical to the published Sec. III B figure.

Usage
-----
    python3 morphology/angular_profile_by_energy.py
    python3 morphology/angular_profile_by_energy.py --powers 0.8 3.2 25 --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.io import fits

_HERE = Path(__file__).resolve().parent
_REANALYSIS = _HERE.parent
_REPO = _REANALYSIS.parent
for _p in (str(_REPO), str(_REANALYSIS), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_angular_profile_vs_models import (  # noqa: E402
    COUNTS, FITDIR, NBINS, _lonlat, excess_map, ring_profile, fit_amp_offset, nfw_los,
)
from helpers.plot_style import apply_plot_style, get_cmap_colors  # noqa: E402


def bin_centres(fitdir):
    out = []
    for k in range(NBINS):
        p = Path(fitdir) / f"global_fit_k{k:02d}.npz"
        if p.exists():
            out.append((k, float(np.load(p, allow_pickle=True)["Ectr_mev"]) / 1000.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fitdir", type=Path, default=FITDIR)
    ap.add_argument("--powers", type=float, nargs=3, default=[0.4, 5.0, 231],
                    metavar=("LO", "HI", "N"),
                    help="scan of NFW inner power p in rho^p: lo, hi, number of steps")
    ap.add_argument("--out-dir", type=Path,
                    default=_REANALYSIS / "figures" / "paper_figures")
    ap.add_argument("--out-txt", type=Path,
                    default=_REANALYSIS / "results" / "global-systematics"
                    / "angular_profile_by_energy.txt")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    hdr = fits.getheader(COUNTS)
    L, B = _lonlat(hdr)
    edges = np.linspace(10.0, 70.0, 13)
    powers = np.linspace(a.powers[0], a.powers[1], int(a.powers[2]))

    cent = bin_centres(a.fitdir)
    if not cent:
        raise SystemExit(f"no global_fit_k*.npz under {a.fitdir}")

    lines = ["# Ring-profile morphology fitted per energy bin",
             "# p_best is the preferred NFW inner power in rho^p.",
             "# p_lo, p_hi are the delta-chi2 = 1 interval on p; RAIL marks a",
             "# best fit or interval edge sitting on the scanned grid boundary,",
             "# in which case p is bounded by the grid and not by the data.",
             f"# scanned grid: p = {powers[0]:.2f} to {powers[-1]:.2f} "
             f"in {len(powers)} steps",
             f"{'E [GeV]':>9s} {'rings':>6s} {'p_best':>7s} {'p_lo':>6s} "
             f"{'p_hi':>6s} {'chi2/nu':>8s} "
             f"{'rho2':>7s} {'rho2.5':>7s} {'iso':>7s} {'flag':>6s}"]
    Es, pbest, chis, perr = [], [], [], []
    for k, E in cent:
        lo, hi = E / 1.30, E * 1.30          # bins are log-spaced, ratio ~1.69
        try:
            exc, mask = excess_map(a.fitdir, lo, hi)
        except SystemExit:
            continue
        ctr, val, err = ring_profile(exc, mask, L, B, edges)
        nring = int(np.isfinite(val).sum())
        if nring < 5:
            continue
        scan = []
        for p in powers:
            m = np.array([nfw_los(x, p) for x in ctr])
            _, _, c = fit_amp_offset(m, val, err)
            scan.append(c)
        scan = np.asarray(scan)
        if not np.any(np.isfinite(scan)):
            continue
        j = int(np.nanargmin(scan))
        ref = {}
        for nm, p in ((r"rho2", 2.0), (r"rho2.5", 2.5), ("iso", None)):
            m = np.ones_like(ctr) if p is None else np.array([nfw_los(x, p) for x in ctr])
            ref[nm] = fit_amp_offset(m, val, err)[2]
        # delta-chi2 = 1 interval on p.  scan holds chi2/nu, so rescale by nu.
        nu = max(nring - 2, 1)
        ok = np.isfinite(scan) & ((scan - scan[j]) * nu <= 1.0)
        p_lo, p_hi = float(powers[ok].min()), float(powers[ok].max())
        railed = (j == 0 or j == len(powers) - 1
                  or ok[0] or ok[-1])
        flag = "RAIL" if railed else ""
        Es.append(E); pbest.append(powers[j]); chis.append(scan[j])
        perr.append(0.5 * (p_hi - p_lo))
        lines.append(f"{E:9.2f} {nring:6d} {powers[j]:7.2f} {p_lo:6.2f} "
                     f"{p_hi:6.2f} {scan[j]:8.2f} "
                     f"{ref['rho2']:7.2f} {ref['rho2.5']:7.2f} {ref['iso']:7.2f} "
                     f"{flag:>6s}")

    Es = np.array(Es); pbest = np.array(pbest); perr = np.array(perr)
    band = (Es >= 10) & (Es <= 40)
    lines.append("")
    lines.append(f"# signal band 10-40 GeV : p_best = "
                 f"{pbest[band].min():.2f} to {pbest[band].max():.2f}")
    lines.append(f"# all bins              : p_best = "
                 f"{pbest.min():.2f} to {pbest.max():.2f}")
    if band.sum() and (~band).sum():
        lines.append(f"# mean p_best in band {pbest[band].mean():.2f} "
                     f"versus outside band {pbest[~band].mean():.2f}")
    if band.sum() > 1:
        # is a single power an adequate description of the band?
        w = 1.0 / np.clip(perr[band], 1e-6, None) ** 2
        pm = float((w * pbest[band]).sum() / w.sum())
        pm_err = float(1.0 / np.sqrt(w.sum()))
        c2 = float((w * (pbest[band] - pm) ** 2).sum())
        ndof = int(band.sum() - 1)
        lines.append(f"# constant-p fit across band: p = {pm:.2f} +/- "
                     f"{pm_err:.2f}, chi2 = {c2:.2f} for {ndof} dof")
    out = "\n".join(lines)
    print(out)
    a.out_txt.parent.mkdir(parents=True, exist_ok=True)
    a.out_txt.write_text(out + "\n")
    print(f"\nwritten: {a.out_txt}")

    if a.no_plot or not len(Es):
        return
    import matplotlib.pyplot as plt
    apply_plot_style("paper", base_fontsize=10, linewidth=1.4, n_colors=8, cmap_name="plasma")
    cols = get_cmap_colors(cmap_name="plasma", n=3, start=0.15, end=0.8)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.plot(Es, pbest, "o-", color=cols[0], ms=4, lw=1.5,
            label="preferred inner power")
    ax.axhline(2.0, ls="--", color=cols[1], lw=1.2, label=r"NFW $\rho^2$")
    ax.axvspan(10, 40, color="0.85", zorder=0, label="signal band")
    ax.set_xscale("log")
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"preferred inner power $p$ in $\rho^{\,p}$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.13)
    fig.tight_layout()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = a.out_dir / f"f_angular_profile_by_energy.{ext}"
        fig.savefig(f, dpi=200)
        print(f"written: {f}")


if __name__ == "__main__":
    main()
