#!/usr/bin/env python3
r"""
Reproduce Totani (2025) Fig. 10 -- the halo radial *angular* profile at 21 GeV --
analytically from the shared Via Lactea II NFW parameters, and overlay digitized
Totani data points for comparison.

Totani's NFW (his Sec. 3.3): rho(r) = rho_s x^-1 (1+x)^-2, x=r/rs,
  rs = 21 kpc, rho_s = 8.1e6 Msun/kpc^3, R0 = 8 kpc, rvir = 402 kpc, rho_sun = 0.42 GeV/cm^3.
This is identical to the rs21_R08_rvir402 template, so the *model* curves below ARE
Totani's curves; the only thing to digitize is his DATA POINTS (and only if you want
them overlaid).  The three emissivity laws are eps ~ rho^n with n = 2 (NFW-rho2,
preferred), 2.5 (NFW-rho2.5, GC-excess-like) and 1 (NFW-rho1, decay/subhalo-like).

The curve is the ROI-ring-averaged line-of-sight integral
    P_n(psi) = < integral rho(r)^n ds >_ring ,  r = sqrt(R0^2 + s^2 - 2 R0 s cos psi),
averaged over the ROI (|l|<=60, 10<=|b|<=60) at angular distance psi from the GC.

Normalization:
  --abs none   (default) shape-normalized: each curve / its value at the first ring.
  --pole-flux-21 F   put the rho^2 curve in absolute units F * P_2(psi)/P_2(pole),
                     F = your global-fit halo pole flux at 21 GeV [MeV cm^-2 s^-1 sr^-1]
                     (read it off your Fig. 8 / halo spectrum); rho2.5 and rho1 are
                     then shown shape-matched to rho2 at the first ring.

Digitizing Totani Fig. 10 (only if you want his points): open
  https://arxiv.org/html/2507.07209v3/x10.png  in WebPlotDigitizer
  (https://automeris.io/WebPlotDigitizer), calibrate the log axes, export one CSV per
  series as  psi_deg,flux  and pass with --totani-csv label=path ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from helpers.plot_style import (  # noqa: E402
    add_style_argument, apply_plot_style, current_savefig_kwargs,
    normalise_plot_style, style_sizing, theme_ink,
)

# Totani / Via Lactea II NFW
RS_KPC = 21.0
RVIR_KPC = 402.0
R0_KPC = 8.0
RHOS = 8.1e6          # Msun/kpc^3  (only the shape matters here)
GAMMA = 1.0

MODELS = {  # label -> (n, color, linestyle)
    "NFW-rho2":   (2.0, "#1b9e77", "-"),
    "NFW-rho2.5": (2.5, "#d95f02", "--"),
    "NFW-rho1":   (1.0, "#7570b3", ":"),
}


def rho(r):
    x = r / RS_KPC
    return RHOS * x ** (-GAMMA) * (1.0 + x) ** (GAMMA - 3.0)


def los_integral(psi_rad, n, smax=RVIR_KPC, ns=2000):
    s = np.linspace(1e-3, smax, ns)
    cps = np.cos(psi_rad)
    r = np.sqrt(R0_KPC ** 2 + s ** 2 - 2.0 * R0_KPC * s * cps)
    return getattr(np, "trapezoid", np.trapz)(rho(r) ** n, s)


def ring_profile(n, psi_edges, nl=240, nb=200):
    """ROI-ring-averaged LOS integral P_n(psi) over |l|<=60, 10<=|b|<=60."""
    l = np.deg2rad(np.linspace(0.0, 60.0, nl))      # symmetric in l
    b = np.deg2rad(np.linspace(10.0, 60.0, nb))
    L, B = np.meshgrid(l, b, indexing="ij")
    psi = np.arccos(np.clip(np.cos(B) * np.cos(L), -1, 1))
    w = np.cos(B)                                    # solid-angle weight
    # LOS integral at each grid psi (vectorom over unique-ish psi via interpolation grid)
    psi_grid = np.linspace(psi.min(), psi.max(), 200)
    I_grid = np.array([los_integral(p, n) for p in psi_grid])
    I = np.interp(psi, psi_grid, I_grid)
    centers = 0.5 * (psi_edges[1:] + psi_edges[:-1])
    P = np.full(centers.size, np.nan)
    pd = np.rad2deg(psi)
    for i in range(centers.size):
        m = (pd >= psi_edges[i]) & (pd < psi_edges[i + 1])
        if np.any(m):
            P[i] = np.sum(I[m] * w[m]) / np.sum(w[m])
    return centers, P


def make_plot(args):
    import matplotlib.pyplot as plt

    psi_edges = np.linspace(10.0, 70.0, args.n_rings + 1)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    # pole LOS integral (b=90 -> psi=90) for absolute normalization
    psi_pole = np.deg2rad(90.0)

    first_rho2 = None
    for label, (n, col, ls) in MODELS.items():
        psi_c, P = ring_profile(n, psi_edges)
        if args.pole_flux_21 and label == "NFW-rho2":
            Ppole = los_integral(psi_pole, n)
            y = args.pole_flux_21 * P / Ppole
            first_rho2 = (psi_c, y)
        elif args.pole_flux_21 and first_rho2 is not None:
            # shape-match other models to rho2 at the first finite ring
            ref = np.nanmax(P[np.isfinite(P)])
            y = (first_rho2[1][0] / np.nanmax(ring_profile(2.0, psi_edges)[1])) * P
        else:
            y = P / P[np.isfinite(P)][0]            # shape-normalized to first ring
        ax.plot(psi_c, y, color=col, ls=ls, lw=2.2, label=label)

    for spec in (args.totani_csv or []):
        if "=" not in spec:
            continue
        lab, path = spec.split("=", 1)
        d = np.loadtxt(path, delimiter=",")
        # theme_ink so the digitised points stay visible on a dark slide, where
        # a literal "k" edge would draw black-on-black.
        ax.plot(d[:, 0], d[:, 1], "o", ms=6, mfc="none", mec=theme_ink("k"), mew=1.3,
                label=f"Totani {lab} (digitized)")

    ax.set_xlabel(r"angular distance from GC  $\psi$ [deg]")
    if args.pole_flux_21:
        ax.set_ylabel(r"$E^2\,dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.set_yscale("log")
    else:
        ax.set_ylabel(r"angular profile (normalized to first ring)")
        ax.set_yscale("log")
    # Title and legend were pinned at the print point sizes; scale them with the
    # active style so they don't shrink away on the larger slide canvas.
    _fs = style_sizing(normalise_plot_style(getattr(args, "style", None)))["fontsize"]
    ax.set_title("Halo angular profile at 21 GeV vs Totani Fig. 10 (Via Lactea II NFW)",
                 fontsize=_fs)
    ax.legend(fontsize=_fs - 1)
    ax.grid(True, which="both", alpha=0.15)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # current_savefig_kwargs carries transparent=True on the slide styles.
    save_kw = current_savefig_kwargs(bbox_inches="tight")
    for ext in ("pdf", "png"):
        p = args.out_dir / f"angular_profile_21GeV.{ext}"
        fig.savefig(p, dpi=220, **save_kw)
        print(f"  saved {p}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-rings", type=int, default=12)
    p.add_argument("--pole-flux-21", type=float, default=None,
                   help="Absolute rho^2 normalization: global-fit halo pole flux at 21 GeV.")
    p.add_argument("--totani-csv", nargs="+", default=None,
                   help="Digitized Totani Fig.10 series as label=path.csv (cols: psi_deg,flux).")
    p.add_argument("--out-dir", type=Path, default=_HERE / "results" / "angular_profile")
    add_style_argument(p)
    main_args = p.parse_args(argv)
    # Must run before the figure is built: it installs the rcParams, canvas
    # scale and ink that make_plot() then reads.
    style = normalise_plot_style(main_args.style, default="paper")
    sizing = style_sizing(style)
    apply_plot_style(style, base_fontsize=sizing["fontsize"],
                     linewidth=sizing["linewidth"], n_colors=8, cmap_name="plasma")
    make_plot(main_args)


if __name__ == "__main__":
    main()
