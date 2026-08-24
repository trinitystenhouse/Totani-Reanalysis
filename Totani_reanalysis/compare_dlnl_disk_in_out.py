#!/usr/bin/env python3
r"""
DIAGNOSTIC FIGURE (not for the paper).

Per-energy-bin halo preference Delta lnL = lnL(halo) - lnL(no-halo baseline),
as in Fig. 9, but contrasting the disk-INCLUDED and disk-EXCLUDED ROI
conventions for all three NFW morphologies (rho^2.5, rho^2, rho^1).

Each (profile, convention) series is computed self-consistently: halo and the
no-halo baseline (fig4) are read from the SAME method. The full pixelwise MCMC
is used whenever BOTH the halo and the no-halo baseline exist for that
convention; otherwise the deterministic global fit is used. The disk-included
no-halo pixelwise baseline (pixelwise fig4_w_disk) does not exist yet, so the
disk-included series currently falls back to the deterministic fit -- re-running
this script after that fit finishes upgrades it automatically. The source used
for each series is printed and annotated on the figure.

Usage (from Totani_reanalysis/):
  python compare_dlnl_disk_in_out.py
  python compare_dlnl_disk_in_out.py --copy-to <somewhere>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

import make_paper_results_figures as M


def _quickglobal_dir(fig_n: int, with_disk: bool) -> Path:
    suf = "_w_disk" if with_disk else ""
    return M.MCMC / "fit_results" / "quick_global" / f"global_fit_results_fig{fig_n}{suf}"


def _loglike_ml_by_bin(d) -> dict:
    """Per-bin ML log-likelihood keyed by rounded E[GeV], from either the
    deterministic (global_fit_k*.npz) or MCMC (mcmc_results_k*.npz) outputs."""
    d = Path(d)
    out = {}
    for k in range(13):
        for stem in (f"global_fit_k{k:02d}.npz", f"mcmc_results_k{k:02d}.npz"):
            p = d / stem
            if not p.exists():
                continue
            with np.load(p, allow_pickle=True) as z:
                if "loglike_ml" not in z.files:
                    continue
                val = float(z["loglike_ml"])
                E = float(z["Ectr_mev"]) / 1000.0
            if np.isfinite(val):
                out[round(E, 8)] = val
            break
    return out


def _dlnl_series(fig_n: int, with_disk: bool):
    """Return (E, dlnl, source). Halo and baseline use the SAME source."""
    Lh = _loglike_ml_by_bin(M.pixelwise_dir(fig_n, with_disk))
    Lb = _loglike_ml_by_bin(M.pixelwise_dir(4, with_disk))
    source = "pixelwise"
    if not (Lh and Lb):
        Lh = _loglike_ml_by_bin(_quickglobal_dir(fig_n, with_disk))
        Lb = _loglike_ml_by_bin(_quickglobal_dir(4, with_disk))
        source = "deterministic"
    E, dl = [], []
    for e in sorted(set(Lh) & set(Lb)):
        v = Lh[e] - Lb[e]
        if np.isfinite(v) and abs(v) < 1.0e6:
            E.append(e)
            dl.append(v)
    return np.array(E), np.array(dl), source


def main(argv=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=M.PC / "figures" / "diagnostics")
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    M.apply_plot_style("paper", base_fontsize=10, linewidth=1.4, n_colors=8, cmap_name="plasma")
    cols = M.get_cmap_colors(cmap_name="plasma", n=5, start=0.18, end=0.86)
    styles = [
        (5, r"NFW-$\rho^{2.5}$", "^", cols[0]),
        (6, r"NFW-$\rho^{2}$", "o", cols[2]),
        (7, r"NFW-$\rho^{1}$", "s", cols[4]),
    ]

    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    finite_vals = []
    excl_handles, incl_handles, labels = [], [], []
    sources = set()
    for fn, lab, mk, c in styles:
        E_e, dl_e, src_e = _dlnl_series(fn, with_disk=False)
        E_i, dl_i, src_i = _dlnl_series(fn, with_disk=True)
        sources.update({f"excl:{src_e}", f"incl:{src_i}"})
        print(f"[{lab}] disk-excluded source={src_e}, disk-included source={src_i}")
        # disk-excluded: solid + open marker
        ax.plot(E_e, dl_e, color=c, lw=1.6, ls="-", marker=mk, ms=4.6,
                mfc="white", mec=c, mew=0.9, label="_nolegend_")
        # disk-included: dashed + filled marker
        ax.plot(E_i, dl_i, color=c, lw=1.6, ls="--", marker=mk, ms=4.6,
                mfc=c, mec=c, mew=0.7, label="_nolegend_")
        for v in (dl_e, dl_i):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            if v.size:
                finite_vals.append(v)
        excl_handles.append(Line2D([0], [0], color=c, lw=1.6, ls="-", marker=mk,
                                   ms=4.6, mfc="white", mec=c, mew=0.9))
        incl_handles.append(Line2D([0], [0], color=c, lw=1.6, ls="--", marker=mk,
                                   ms=4.6, mfc=c, mec=c, mew=0.7))
        labels.append(lab)

    ax.axvspan(10.0, 40.0, color="0.85", alpha=0.35, zorder=0)
    ax.axhline(0.0, color="0.45", lw=0.75, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1000.0)
    ax.set_xlabel("photon energy [GeV]")
    ax.set_ylabel(r"$\ln(L)-\ln(L_{\rm no\text{-}halo})$")
    if finite_vals:
        y = np.concatenate(finite_vals)
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.08 * (ymax - ymin)
            ax.set_ylim(min(ymin - pad, -0.5), ymax + pad)

    leg = ax.legend(excl_handles + incl_handles, labels + labels,
                    title="disk excluded      disk included",
                    fontsize=7.0, title_fontsize=7.2, frameon=True,
                    framealpha=0.88, ncol=2, loc="upper right",
                    handlelength=2.0, columnspacing=1.2, borderpad=0.45)
    leg.get_frame().set_edgecolor("0.65")
    # source note (deterministic vs pixelwise), since they can differ per series
    note = " / ".join(sorted(sources))
    ax.text(0.015, 0.02, note, transform=ax.transAxes, fontsize=6.0,
            color="0.4", va="bottom", ha="left")
    ax.grid(alpha=0.14, which="both")
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = args.out_dir / f"f_dlnl_disk_in_out.{ext}"
        fig.savefig(p, dpi=220, **M.current_savefig_kwargs(bbox_inches="tight"))
        paths.append(p)
    plt.close(fig)
    print("saved", paths[0])
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[0], args.copy_to / paths[0].name)
        print("copied ->", args.copy_to / paths[0].name)


if __name__ == "__main__":
    main()
