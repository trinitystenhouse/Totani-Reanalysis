#!/usr/bin/env python3
r"""
Paper figure: halo best-fit <sigma v>(m) against the s-wave, p-wave, d-wave and
Sommerfeld dwarf-spheroidal limits on a single axis.

The halo signal curve (the cross section that reproduces the 20 GeV halo at each
mass) is read from the global-fit PPPC mass scan; the dwarf-limit curves are the
halo-comparable limits built by extract_boddy_limits.py (core.dsph_limits_multi).
Where the signal lies above a limit curve, that velocity law is in tension; below,
it is allowed. This single frame shows the s-wave tension and its resolution.

Usage (from the repo root):
    python Totani_reanalysis/velocity_dep/plot_velocity_limits_overlay.py --fit global_fig6 --channels bb WW
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as _REPO  # path-fix 2026-06
_REANALYSIS = _HERE.parent
_PPPC = _REPO / "Totani_paper_check" / "results" / "pppc_mass_scan"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REANALYSIS))

from helpers.plot_style import apply_plot_style, current_savefig_kwargs, get_cmap_colors  # noqa: E402
from core.dsph_limits_multi import dsph_upper_limit  # noqa: E402

THERMAL_RELIC = 2.2e-26  # cm^3/s

CH_LABEL = {"bb": r"$b\bar b$", "WW": r"$W^+W^-$", "tautau": r"$\tau^+\tau^-$"}


def _limit_styles() -> dict[str, tuple[str, str, object]]:
    colors = get_cmap_colors(cmap_name="plasma", n=8, start=0.12, end=0.92)
    return {
        "hoof":        (r"$s$-wave dSph limit",        "-",   "0.15"),
        "boddy_pwave": (r"$p$-wave dSph limit",        "--",  colors[5]),
        "boddy_dwave": (r"$d$-wave dSph limit",        ":",   colors[3]),
        "boddy_somm":  (r"Sommerfeld dSph limit",      "-.",  colors[6]),
    }


def load_signal(fit: str, channel: str):
    npz = _PPPC / fit / channel / "mass_scan_results.npz"
    if not npz.exists():
        raise FileNotFoundError(f"missing {npz}; run run_pppc_mass_scan.py --fit {fit}")
    d = np.load(npz, allow_pickle=True)
    m = np.asarray(d["masses_GeV"], float)
    sv = np.asarray(d["sigmav_best"], float)
    lnL = np.asarray(d["total_lnL"], float)
    best = int(np.nanargmax(lnL)) if np.any(np.isfinite(lnL)) else int(np.nanargmin(d["spectral_chi2"]))
    return m, sv, best


def make_figure(fit: str, channel: str, sources: list[str], out_dir: Path):
    import matplotlib.pyplot as plt

    m, sv, best = load_signal(fit, channel)
    m_best, sv_best = m[best], sv[best]
    signal_color = get_cmap_colors(cmap_name="plasma", n=6, start=0.18, end=0.86)[1]
    limit_styles = _limit_styles()

    # smooth mass grid for the limit curves
    mg = np.geomspace(max(50.0, m.min() * 0.9), m.max() * 1.05, 300)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    # halo signal
    ax.plot(m, sv, color=signal_color, lw=2.2, marker="o", ms=4.2, zorder=6,
            label=rf"Halo best fit ({CH_LABEL.get(channel, channel)})")
    ax.plot([m_best], [sv_best], "*", ms=12, color=signal_color, mec="k", mew=0.6, zorder=7)
    ax.annotate(rf"$m_\chi={m_best:.0f}$ GeV", xy=(m_best, sv_best),
                xytext=(8, 10), textcoords="offset points", fontsize=11)

    # thermal relic
    ax.axhline(THERMAL_RELIC, color="0.5", lw=1.2, ls=(0, (1, 1)), zorder=2)
    ax.text(mg[-1], THERMAL_RELIC * 1.15, "thermal relic", ha="right", va="bottom",
            fontsize=9, color="0.4")

    # limit curves + tension annotation at the best-fit mass
    txt = []
    for src in sources:
        label, ls, col = limit_styles[src]
        lim = np.array([dsph_upper_limit(float(x), channel, src) for x in mg])
        ok = np.isfinite(lim) & (lim > 0)
        ax.plot(mg[ok], lim[ok], ls=ls, color=col, lw=2.0, zorder=4, label=label)
        lim_best = dsph_upper_limit(float(m_best), channel, src)
        if np.isfinite(lim_best) and lim_best > 0:
            R = sv_best / lim_best
            txt.append(rf"{label.split(' dSph')[0]}: $\mathcal{{R}}={R:.2g}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]")
    ax.set_title(rf"{CH_LABEL.get(channel, channel)} velocity scaling comparison", fontsize=11)
    # show all curves: span signal and limits
    ax.set_xlim(mg.min(), mg.max())
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.98, 0.03, "  ".join(txt) + r"  (at $m_\chi^{\rm best}$)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.grid(True, which="both", alpha=0.15)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"velocity_limits_overlay_{fit}_{channel}.{ext}"
        fig.savefig(p, dpi=300, **current_savefig_kwargs(bbox_inches="tight"))
        print(f"  saved {p}")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="global_fig6")
    p.add_argument("--channels", nargs="+", default=["bb", "WW"])
    p.add_argument("--sources", nargs="+",
                   default=["hoof", "boddy_pwave", "boddy_dwave", "boddy_somm"])
    p.add_argument("--out-dir", type=Path, default=_HERE / "results" / "velocity_overlay")
    p.add_argument("--style", default=os.environ.get("PLOT_STYLE", "paper"))
    args = p.parse_args(argv)
    apply_plot_style(args.style, base_fontsize=10, linewidth=1.5, n_colors=10, cmap_name="plasma")
    for ch in args.channels:
        make_figure(args.fit, ch, args.sources, args.out_dir)


if __name__ == "__main__":
    main()
