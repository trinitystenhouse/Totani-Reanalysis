"""
plot_reshaping.py
================
Plotting utilities for the spectral reshaping analysis.

All functions follow the existing style in attenuation_eft.py / make_paper_style_operator_overlays.py:
- Figures saved to the `plots/` subdirectory of Totani_reanalysis.
- Log-log axes, publication-quality labels.
- Each function is self-contained: takes data arrays and saves a figure.

Public API
----------
plot_spectrum_comparison(...)     : Phi_obs vs Phi_0 vs simple attenuation
plot_kernel_heatmap(...)          : K[i,j] redistribution matrix visualisation
plot_inscatter_decomposition(...) : breakdown of in-scatter by source bin
plot_reshaping_vs_attenuation_chi2(...) : delta_chi2 map from scan results
plot_energy_loss_summary(...)     : max delta_E/E vs E for the kinematic regime
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.attenuation_eft import E_BINS_GEV, PHI_TOTANI, SIGMA_TOTANI
from helpers.plot_style import current_savefig_kwargs, set_plot_style


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def _plots_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    d = here / "plots"
    d.mkdir(exist_ok=True)
    return d


def _output_path(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute() or path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return _plots_dir() / path


def _apply_style() -> None:
    set_plot_style(style="paper", cmap_name="plasma", base_fontsize=10, linewidth=1.6, n_colors=10)


# ---------------------------------------------------------------------------
# Spectrum comparison
# ---------------------------------------------------------------------------

def plot_spectrum_comparison(
    E_bins: np.ndarray,
    phi_0: np.ndarray,
    phi_err: np.ndarray,
    phi_obs_reshaping: np.ndarray,
    phi_obs_attenuation: np.ndarray,
    *,
    tau: Optional[np.ndarray] = None,
    label_reshaping: str = "Reshaping",
    label_attenuation: str = "Attenuation only",
    title: str = "",
    filename: str = "spectrum_reshaping_comparison.pdf",
    show: bool = False,
) -> Path:
    """
    Compare the reshaped spectrum, simple-attenuation spectrum, and data.

    Includes a lower panel showing the residuals (model - data) / sigma
    for each model, making it easy to judge which bins drive chi2.

    Parameters
    ----------
    E_bins : (nE,)       energy bin centres [GeV]
    phi_0 : (nE,)        data (Totani halo component) [MeV cm^-2 s^-1 sr^-1]
    phi_err : (nE,)      1-sigma errors
    phi_obs_reshaping : (nE,)   model with full redistribution
    phi_obs_attenuation : (nE,) model with simple attenuation
    tau : (nE,) optional  optical depths, shown as annotation
    label_reshaping, label_attenuation : legend labels
    title : str   figure suptitle
    filename : str   output filename in plots/
    show : bool   call plt.show() after saving

    Returns
    -------
    out_path : Path
    """
    _apply_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    # --- Top panel: spectra ---
    ax_top.errorbar(
        E_bins, phi_0 * 1e5, yerr=phi_err * 1e5,
        fmt="ko", ms=5, lw=1.2, zorder=5,
        label="Totani NFW-ρ² (Fig. 8)",
    )
    ax_top.plot(
        E_bins, phi_obs_reshaping * 1e5,
        "C0-", lw=2.0, label=label_reshaping,
    )
    ax_top.plot(
        E_bins, phi_obs_attenuation * 1e5,
        "C1--", lw=1.8, label=label_attenuation,
    )
    ax_top.axhline(0, color="0.6", lw=0.8, ls=":")

    ax_top.set_xscale("log")
    ax_top.set_ylabel(r"$E^2\,dN/dE\ [\times 10^{-5}\ \mathrm{MeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$",
                      fontsize=11)
    ax_top.legend(fontsize=9, framealpha=0.9)
    if title:
        ax_top.set_title(title, fontsize=10, pad=6)

    # Annotate tau values if provided
    if tau is not None:
        for k, (E, t) in enumerate(zip(E_bins, tau)):
            if t > 1e-8:
                ax_top.annotate(
                    fr"$\tau$={t:.1e}",
                    xy=(E, phi_0[k] * 1e5),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=6, ha="center", color="0.5",
                )

    # --- Bottom panel: residuals ---
    resid_r = (phi_obs_reshaping - phi_0) / phi_err
    resid_a = (phi_obs_attenuation - phi_0) / phi_err

    ax_bot.axhline(0, color="k", lw=0.8)
    ax_bot.axhline(1, color="0.7", lw=0.6, ls="--")
    ax_bot.axhline(-1, color="0.7", lw=0.6, ls="--")
    ax_bot.plot(E_bins, resid_r, "C0o-", ms=4, lw=1.4, label=label_reshaping)
    ax_bot.plot(E_bins, resid_a, "C1^--", ms=4, lw=1.2, label=label_attenuation)
    ax_bot.set_xlabel("Photon energy [GeV]", fontsize=11)
    ax_bot.set_ylabel(r"$(\Phi_\mathrm{mod} - \Phi_\mathrm{data})\,/\,\sigma$", fontsize=9)
    ax_bot.set_ylim(-4, 4)
    ax_bot.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = _output_path(filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Redistribution kernel heatmap
# ---------------------------------------------------------------------------

def plot_kernel_heatmap(
    K: np.ndarray,
    E_bins: np.ndarray,
    *,
    title: str = "",
    filename: str = "redistribution_kernel.pdf",
    log_scale: bool = True,
    show: bool = False,
) -> Path:
    """
    Visualise the redistribution matrix K[i,j] as a heatmap.

    The upper-triangular structure (i ≤ j) is immediately visible.
    The diagonal represents near-forward scatters that stay in the same bin.
    Off-diagonal entries show how flux leaks from bin j to bin i < j.

    Parameters
    ----------
    K : (nE, nE)     redistribution matrix
    E_bins : (nE,)   energy centres [GeV]
    title : str
    filename : str   in plots/
    log_scale : bool  use log10 colour scale (recommended: values span ~10 decades)
    show : bool

    Returns
    -------
    out_path : Path
    """
    K = np.asarray(K, dtype=float)
    nE = K.shape[0]

    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    if log_scale:
        # Mask exact zeros for log scale
        K_plot = np.where(K > 0.0, K, np.nan)
        vmax = np.nanmax(K_plot)
        vmin = np.nanmin(K_plot)
        
        # Handle edge case: all NaN or invalid values
        if not np.isfinite(vmax) or not np.isfinite(vmin):
            # Fall back to linear scale with dummy values
            K_plot = np.zeros_like(K)
            norm = None
            cmap = "plasma"
        else:
            vmin = max(vmin, vmax * 1e-8)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = "plasma"
    else:
        K_plot = K
        norm = None
        cmap = "plasma"

    im = ax.imshow(
        K_plot,
        origin="lower",
        aspect="auto",
        norm=norm,
        cmap=cmap,
        interpolation="nearest",
    )

    # Axis labels: use bin indices with E-value annotations
    tick_idx = np.arange(0, nE, max(1, nE // 6))
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels([f"{E_bins[k]:.0f}" for k in tick_idx], fontsize=8)
    ax.set_yticklabels([f"{E_bins[k]:.0f}" for k in tick_idx], fontsize=8)

    ax.set_xlabel("Input energy $E_j$ [GeV]  (source bin)", fontsize=10)
    ax.set_ylabel("Output energy $E_i$ [GeV]  (observed bin)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar_label = r"$K_{ij}$  (log scale)" if log_scale else r"$K_{ij}$"
    cbar.set_label(cbar_label, fontsize=9)

    if title:
        ax.set_title(title, fontsize=10, pad=6)

    # Draw upper-triangular boundary
    ax.plot([0, nE - 1], [0, nE - 1], color="white", lw=0.8, ls="--", alpha=0.5,
            label="diagonal")

    plt.tight_layout()
    out_path = _output_path(filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# In-scatter decomposition
# ---------------------------------------------------------------------------

def plot_inscatter_decomposition(
    E_bins: np.ndarray,
    phi_0: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    *,
    n_source_bins: int = 5,
    title: str = "",
    filename: str = "inscatter_decomposition.pdf",
    show: bool = False,
) -> Path:
    """
    Show which source energy bins contribute most to the in-scatter flux
    at each observed energy.

    For each source bin j (the top `n_source_bins` by contribution), plot
    K[i,j] * tau[j] * phi_0[j] * exp(-tau[j]) as a function of E_i.
    Stack the contributions to show the total in-scatter.

    Parameters
    ----------
    E_bins : (nE,)
    phi_0 : (nE,)   source spectrum
    K : (nE, nE)    redistribution matrix
    tau : (nE,)     optical depths
    n_source_bins : int   number of largest contributors to highlight
    title, filename, show : as in plot_spectrum_comparison

    Returns
    -------
    out_path : Path
    """
    phi_0 = np.asarray(phi_0, dtype=float)
    tau = np.asarray(tau, dtype=float)
    K = np.asarray(K, dtype=float)

    scatter_weight = tau * np.exp(-tau)            # (nE,)
    # contributions[i, j] = K[i,j] * scatter_weight[j] * phi_0[j]
    contributions = K * (scatter_weight * phi_0)[None, :]   # (nE, nE)

    total_inscatter = contributions.sum(axis=1)    # (nE,)

    # Identify top contributors by total contribution to all output bins
    col_totals = contributions.sum(axis=0)         # (nE,) — total from each source bin
    top_j = np.argsort(col_totals)[::-1][:n_source_bins]

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_source_bins))

    # Stacked area: sum of top contributors + remainder
    bottom = np.zeros_like(total_inscatter)
    for idx, j in enumerate(top_j):
        contrib_j = contributions[:, j]
        ax.fill_between(
            E_bins,
            bottom * 1e5,
            (bottom + contrib_j) * 1e5,
            color=colors[idx],
            alpha=0.75,
            label=fr"Source bin $E_j={E_bins[j]:.0f}$ GeV",
        )
        bottom += contrib_j

    # Remainder
    remaining = total_inscatter - bottom
    if np.any(remaining > 0):
        ax.fill_between(
            E_bins,
            bottom * 1e5,
            (bottom + np.maximum(remaining, 0.0)) * 1e5,
            color="0.7",
            alpha=0.5,
            label="Other source bins",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Observed energy $E_i$ [GeV]", fontsize=11)
    ax.set_ylabel(
        r"In-scatter flux $[\times 10^{-5}\ \mathrm{MeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    if title:
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    out_path = _output_path(filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Reshaping vs attenuation delta-chi2
# ---------------------------------------------------------------------------

def plot_reshaping_vs_attenuation_chi2(
    scan_result: dict,
    *,
    filename: str = "delta_chi2_reshaping_vs_attenuation.pdf",
    title: str = "",
    show: bool = False,
) -> Path:
    """
    Plot the difference delta_chi2 = chi2_reshaping - chi2_attenuation over
    the (m_chi, Lambda) parameter space.

    - Positive delta_chi2 (red): reshaping gives *worse* fit than attenuation.
      Physical interpretation: in-scatter flux fills in the dip, reducing
      the apparent attenuation and worsening the fit if data favours a dip.
    - Negative delta_chi2 (blue): reshaping gives *better* fit.
      Physical interpretation: in-scatter redistributes flux towards lower
      energies where data has residuals.

    Parameters
    ----------
    scan_result : dict   output of scan_reshaping_chi2
    filename : str   in plots/
    title : str
    show : bool

    Returns
    -------
    out_path : Path
    """
    m_chi = scan_result["m_chi_arr"]
    Lambda = scan_result["Lambda_arr"]
    delta = scan_result["delta_chi2"]

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    M, L = np.meshgrid(m_chi, Lambda, indexing="ij")
    vabs = np.nanpercentile(np.abs(delta[np.isfinite(delta)]), 95)
    vabs = max(vabs, 1e-3)

    cf = ax.contourf(
        np.log10(M), np.log10(L), delta,
        levels=np.linspace(-vabs, vabs, 51),
        cmap="RdBu_r",
        extend="both",
    )
    ax.contour(
        np.log10(M), np.log10(L), delta,
        levels=[0.0], colors="k", linewidths=1.0, linestyles="--",
    )

    cbar = fig.colorbar(cf, ax=ax, label=r"$\Delta\chi^2_\mathrm{reshape} - \Delta\chi^2_\mathrm{atten}$")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel(r"$\log_{10}(m_\chi\,/\,\mathrm{GeV})$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\Lambda\,/\,\mathrm{GeV})$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25, ls="--")

    _op = scan_result.get("operator", "")
    _dm = scan_result.get("dm_type", "")
    ax.set_title(
        title or rf"$\Delta\chi^2$ reshaping$-$attenuation: {_dm} {_op}",
        fontsize=10, pad=6,
    )

    plt.tight_layout()
    out_path = _output_path(filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Kinematic regime summary
# ---------------------------------------------------------------------------

def plot_energy_loss_summary(
    E_bins: np.ndarray,
    m_chi_values: list[float],
    *,
    filename: str = "energy_loss_kinematics.pdf",
    show: bool = False,
) -> Path:
    """
    Plot max fractional energy loss Delta_max/E vs photon energy for several
    DM masses. Overlays the Totani energy bin range and a reference line at
    the typical LAT energy resolution (~10%).

    Provides an at-a-glance check of whether the reshaping effect is resolvable
    at current LAT energy resolution for the DM masses of interest.

    Parameters
    ----------
    E_bins : (nE,)    photon energies [GeV]
    m_chi_values : list of float   DM masses to compare [GeV]
    filename, show : as before

    Returns
    -------
    out_path : Path
    """
    from core.kinematics import max_energy_loss_fraction

    E_bins = np.asarray(E_bins, dtype=float)
    E_fine = np.geomspace(E_bins[0] * 0.5, E_bins[-1] * 2.0, 300)

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(m_chi_values)))
    for m_chi, col in zip(m_chi_values, colors):
        delta = max_energy_loss_fraction(E_fine, m_chi)
        ax.plot(E_fine, delta * 100.0, color=col,
                label=fr"$m_\chi = {m_chi:.0e}$ GeV")

    # LAT energy resolution reference (~10% at these energies)
    ax.axhline(10.0, color="k", lw=1.0, ls=":", label="LAT resolution ≈ 10%")

    # Shade Totani bin range
    ax.axvspan(E_bins[0], E_bins[-1], alpha=0.08, color="steelblue",
               label="Totani energy range")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Photon energy [GeV]", fontsize=11)
    ax.set_ylabel(r"Max energy loss $\Delta E / E$ at backscatter [%]", fontsize=10)
    ax.set_title("Kinematic regime: maximum fractional energy loss per scatter", fontsize=10)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25, which="both", ls="--")

    plt.tight_layout()
    out_path = _output_path(filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", **current_savefig_kwargs())
    if show:
        plt.show()
    plt.close(fig)
    return out_path
