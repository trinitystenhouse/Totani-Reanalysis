#!/usr/bin/env python3
"""
Reproduce Totani (2025) Figures 11, 12, and 13.

Figure 11  (p.14):
    2x2 map panels at the ~21 GeV bin for the NFW-rho^2 fit.
    Fixed colourscale +/-2e-12 cm^-2 s^-1 sr^-1 MeV^-1.
    TL: no-halo fit residual       (data - sum(non-halo components, no-halo fit))
    TR: halo model + residual      (data - sum(non-halo components, halo fit))
    BL: halo model map             (f_halo[k] * mu_halo[k])
    BR: halo fit residual          (data - full model, halo fit)

Figure 12  (p.15):
    2x2 panels of "halo model + residual" (= TR panel of Fig 11) for 4 bins
    below 21 GeV: ~1.5, ~2.5, ~4.3, ~12 GeV.
    Each panel has its own fixed colourscale read from the paper axis labels.
    Uses the with-halo (NFW-rho^2) fit only.

Figure 13  (p.16):
    Same as Fig 12 but for 4 bins above 21 GeV: ~35, ~59, ~100, ~170 GeV.

All maps:
  - Gaussian-smoothed with sigma = 1 degree
  - Disk |b| < 10 deg shown as solid grey band
  - Extended-source masked pixels shown grey (NaN -> grey via cmap.set_bad)
  - Colourmap: cyan -> blue -> black -> orangered -> yellow (matching paper)
  - Colourbar placed above each panel (horizontal), matching paper layout
  - Panel energy labels come from Ectr_gev[k] (the actual bin centre),
    NOT from the hard-coded target values, so they are correct after
    your binning fix.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    load_mask_any_shape,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.mcmc_io import load_mcmc_coeffs_by_label
from helpers.fermi_plotting import (
    CONF_ANNOT_FS,
    CONF_LABEL_FS,
    CONF_TICK_FS,
)
from helpers.plot_style import current_style_is_transparent, normalise_plot_style, set_paper_style

# ---------------------------------------------------------------------------
# Colourmap: cyan -> blue -> black -> orangered -> yellow  (matches paper)
# ---------------------------------------------------------------------------
def make_totani_cmap(*, bad_color="0.55"):
    cmap = LinearSegmentedColormap.from_list(
        "totani",
        ["cyan", "blue", "black", "orangered", "yellow"],
        N=512,
    )
    cmap.set_bad(color=bad_color)   # NaN -> mid-grey
    return cmap


def make_named_cmap(name, *, bad_color="0.55"):
    cmap = plt.get_cmap(str(name)).copy()
    cmap.set_bad(color=bad_color)
    return cmap


def make_plasma_diverging_cmap(*, bad_color="0.55"):
    plasma = plt.get_cmap("plasma")
    cmap = LinearSegmentedColormap.from_list(
        "plasma_diverging",
        [
            plasma(0.06),   # deep blue-purple: most negative
            plasma(0.20),   # purple
            plasma(0.34),   # brighter violet
            (0.0, 0.0, 0.0, 1.0),  # zero -> black
            plasma(0.58),   # magenta-pink
            plasma(0.78),   # orange-yellow
            plasma(0.96),   # bright yellow: most positive
        ],
        N=512,
    )
    cmap.set_bad(color=bad_color)
    return cmap


FONT_PRESETS = {
    "paper": {
        "panel": 8.0,
        "cbar_label": 6.5,
        "cbar_ticks": 6.0,
        "axis_label": 7.5,
        "ticks": 6.5,
    },
    "conference": {
        "panel": float(CONF_ANNOT_FS),
        "cbar_label": float(CONF_LABEL_FS),
        "cbar_ticks": float(CONF_TICK_FS),
        "axis_label": float(CONF_LABEL_FS),
        "ticks": float(CONF_TICK_FS),
    },
}

RHO2_LABEL = r"NFW-$\rho^2$"


def resolve_plot_config(cmap_name):
    cmap_key = str(cmap_name).strip()
    try:
        style_key = normalise_plot_style(cmap_key if cmap_key not in {"", "default", "totani"} else None, default="paper")
    except ValueError:
        style_key = "paper"
    if style_key != "paper":
        set_paper_style(style=style_key, linewidth=1.5, n_colors=12, cmap_name="plasma")
        text_color = plt.rcParams.get("text.color", "white")
        return {
            "mode": "conference",
            "fonts": FONT_PRESETS["conference"],
            "transparent": current_style_is_transparent(),
            "text_color": text_color,
            "disk_color": 0.35,
            "cmap": make_plasma_diverging_cmap(bad_color="0.20"),
        }

    set_paper_style(style="paper", linewidth=1.5, n_colors=12, cmap_name="plasma")
    
    if cmap_key in {"", "totani", "default", "paper"}:
        cmap = make_plasma_diverging_cmap(bad_color="0.55")
    else:
        cmap = make_named_cmap(cmap_key, bad_color="0.55")

    return {
        "mode": "paper",
        "fonts": FONT_PRESETS["paper"],
        "transparent": current_style_is_transparent(),
        "text_color": "black",
        "disk_color": 0.55,
        "cmap": cmap,
    }

# ---------------------------------------------------------------------------
# Fixed colourscale limits read directly from the paper axis labels
# ---------------------------------------------------------------------------
FIG11_VLIM = 2e-12   # all four Fig 11 panels share +/-2e-12

# (approximate target energy in GeV,  symmetric vlim)
FIG12_BINS = [
    (1.5,   4e-10),   # ~1.5 GeV,  +/-4e-10
    (2.5,   1e-10),   # ~2.5 GeV,  +/-1e-10
    (4.3,   2e-11),   # ~4.3 GeV,  +/-2e-11
    (12.0,  5e-12),   # ~12  GeV,  +/-5e-12
]
FIG13_BINS = [
    (35.0,  5e-13),   # ~35  GeV,  +/-5e-13
    (59.0,  2e-13),   # ~59  GeV,  +/-2e-13
    (100.0, 5e-14),   # ~100 GeV,  +/-5e-14
    (170.0, 2e-14),   # ~170 GeV,  +/-2e-14
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _read_cube(path, expected_shape):
    with fits.open(path) as h:
        d = np.array(h[0].data, dtype=float)
    if d.shape != expected_shape:
        raise RuntimeError(f"{path}: shape {d.shape}, expected {expected_shape}")
    return d


def smooth_nan(arr2d, sigma_pix):
    """Weight-normalised Gaussian smooth that does not bleed across NaN boundaries."""
    nan = ~np.isfinite(arr2d)
    src = np.where(nan, 0.0, arr2d)
    wt  = np.where(nan, 0.0, 1.0)
    num = gaussian_filter(src, sigma=sigma_pix, mode="constant", cval=0.0)
    den = gaussian_filter(wt,  sigma=sigma_pix, mode="constant", cval=0.0)
    out = np.full_like(arr2d, np.nan)
    ok  = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def counts_to_flux(counts2d, expo2d, omega2d, dE_mev):
    """Counts -> physical flux [cm^-2 s^-1 sr^-1 MeV^-1]."""
    denom = expo2d * omega2d * float(dE_mev)
    out = np.full_like(counts2d, np.nan, dtype=float)
    ok = (denom > 0) & np.isfinite(denom)
    out[ok] = np.asarray(counts2d, float)[ok] / denom[ok]
    return out


def build_model_map_k(k, coeffs_by_label, mu_by_label):
    """
    Sum f_l[k] * mu_l[k] for every label present in both dicts.
    Returns a 2-D counts map (ny, nx).
    """
    ny, nx = next(iter(mu_by_label.values())).shape[1:]
    model = np.zeros((ny, nx), dtype=float)
    for lab, mu in mu_by_label.items():
        if lab not in coeffs_by_label:
            continue
        ak = float(np.asarray(coeffs_by_label[lab], float).ravel()[k])
        if np.isfinite(ak) and ak != 0.0:
            model += ak * np.asarray(mu[k], float)
    return model


def prep_flux(counts2d, expo2d, omega2d, dE_mev, display_mask2d, sigma_pix):
    """counts -> flux -> NaN outside display region -> smooth."""
    flux = counts_to_flux(counts2d, expo2d, omega2d, dE_mev)
    flux[~display_mask2d] = np.nan
    return smooth_nan(flux, sigma_pix)


def nearest_bin(Ectr_gev, target_gev):
    return int(np.argmin(np.abs(Ectr_gev - target_gev)))


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, flux2d_smooth, vlim, label_text, disk2d, plot_cfg):
    """
    Draw one map panel with Totani colourmap, fixed colourscale, grey disk
    band overlay, and panel label in the top-left corner.
    """
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    fonts = plot_cfg["fonts"]
    im = ax.imshow(flux2d_smooth, origin="lower",
                   cmap=plot_cfg["cmap"], norm=norm, interpolation="nearest")

    # Solid grey band for the Galactic disk
    if disk2d is not None:
        grey = np.where(disk2d, plot_cfg["disk_color"], np.nan).astype(float)
        ax.imshow(grey, origin="lower", cmap="gray",
                  vmin=0.0, vmax=1.0, alpha=1.0, interpolation="nearest")

    ax.text(0.02, 0.97, label_text, transform=ax.transAxes,
            color=plot_cfg["text_color"], fontsize=fonts["panel"], ha="left", va="top")

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        location="top", pad=0.02, fraction=0.05, aspect=30)
    cbar.set_label(r"flux [cm$^{-2}$s$^{-1}$sr$^{-1}$MeV$^{-1}$]", fontsize=fonts["cbar_label"])
    cbar.ax.tick_params(labelsize=fonts["cbar_ticks"])

    ax.set_xlabel("longitude $l$ [deg]", fontsize=fonts["axis_label"])
    ax.set_ylabel("latitude $b$ [deg]", fontsize=fonts["axis_label"])
    ax.tick_params(labelsize=fonts["ticks"])


# ---------------------------------------------------------------------------
# Figure 11
# ---------------------------------------------------------------------------

def make_fig11(k, Ectr_gev, counts, expo, omega, dE_mev,
               coeffs_nohalo, coeffs_halo,
               mu_non_halo, mu_halo, halo_label,
               display3d, disk2d, sigma_pix, outdir, plot_cfg):
    """
    2x2 panels at energy bin k.

    TL: data - sum(non-halo, no-halo fit)        no-halo residual
    TR: data - sum(non-halo, halo fit)            halo model + residual
    BL: f_halo[k] * mu_halo[k]                   halo model map
    BR: data - sum(all, halo fit)                 halo fit residual
    """
    disp2d  = display3d[k]
    expo_k  = expo[k]
    dE_k    = float(dE_mev[k])
    C_k     = counts[k].astype(float)
    E_label = f"{Ectr_gev[k]:.1f} GeV"

    # Non-halo model sums from each fit
    model_nh_nohalo = build_model_map_k(k, coeffs_nohalo, mu_non_halo)
    model_nh_halo   = build_model_map_k(k, coeffs_halo,   mu_non_halo)

    # Halo model at this bin
    a_halo_k = float(np.asarray(coeffs_halo[halo_label], float).ravel()[k])
    halo_map = a_halo_k * np.asarray(mu_halo[k], float)

    # Four count-space images before flux conversion
    tl_counts = C_k - model_nh_nohalo              # no-halo residual
    tr_counts = C_k - model_nh_halo                # halo model + residual
    bl_counts = halo_map                            # halo model only
    br_counts = C_k - model_nh_halo - halo_map     # halo fit residual

    vlim = FIG11_VLIM
    tl = prep_flux(tl_counts, expo_k, omega, dE_k, disp2d, sigma_pix)
    tr = prep_flux(tr_counts, expo_k, omega, dE_k, disp2d, sigma_pix)
    bl = prep_flux(bl_counts, expo_k, omega, dE_k, disp2d, sigma_pix)
    br = prep_flux(br_counts, expo_k, omega, dE_k, disp2d, sigma_pix)

    figsize = (16, 13) if plot_cfg["mode"] == "conference" else (11, 9.5)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    draw_panel(axes[0, 0], tl, vlim, f"{E_label}, no-halo fit residual",  disk2d, plot_cfg)
    draw_panel(axes[0, 1], tr, vlim, f"{E_label}, {RHO2_LABEL} model+residual", disk2d, plot_cfg)
    draw_panel(axes[1, 0], bl, vlim, f"{E_label}, {RHO2_LABEL} model",          disk2d, plot_cfg)
    draw_panel(axes[1, 1], br, vlim, f"{E_label}, {RHO2_LABEL} fit residual",   disk2d, plot_cfg)

    out = os.path.join(outdir, "totani_fig11.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


# ---------------------------------------------------------------------------
# Figures 12 and 13
# ---------------------------------------------------------------------------

def make_fig12_or_13(bins_spec, fig_num,
                     Ectr_gev, counts, expo, omega, dE_mev,
                     coeffs_halo, mu_non_halo, mu_halo, halo_label,
                     display3d, disk2d, sigma_pix, outdir, plot_cfg):
    """
    2x2 panels of "halo model + residual" at four energy bins.

    bins_spec : list of (target_gev, vlim) — exactly 4 entries.
    fig_num   : 12 or 13 (used for filename).

    Panel label text matches the paper: the first panel explicitly says
    "NFW-rho^2 model+residual"; subsequent panels just carry the energy.
    This matches Fig 12 p.15 where only the top-left panel has that label.
    """
    assert len(bins_spec) == 4

    figsize = (16, 13) if plot_cfg["mode"] == "conference" else (11, 9.5)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    for idx, (ax, (target_gev, vlim)) in enumerate(zip(axes.ravel(), bins_spec)):
        k       = nearest_bin(Ectr_gev, target_gev)
        disp2d  = display3d[k]
        dE_k    = float(dE_mev[k])
        C_k     = counts[k].astype(float)
        # Label uses the actual bin centre from the FITS file
        E_label = f"{Ectr_gev[k]:.1f} GeV"

        # halo model + residual = counts - non-halo model (halo fit)
        model_nh  = build_model_map_k(k, coeffs_halo, mu_non_halo)
        tr_counts = C_k - model_nh
        flux_smooth = prep_flux(tr_counts, expo[k], omega, dE_k, disp2d, sigma_pix)

        # Paper labels the first panel with the full descriptor, rest with energy only
        if idx == 0:
            label = f"{E_label}, {RHO2_LABEL} model+residual"
        else:
            label = E_label

        draw_panel(ax, flux_smooth, vlim, label, disk2d, plot_cfg)
        print(f"  Fig {fig_num} panel {idx+1}: k={k}, {E_label}, vlim=+/-{vlim:.1e}")

    out = os.path.join(outdir, f"totani_fig{fig_num}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Totani (2025) Figures 11, 12, 13."
    )

    repo_dir = os.environ.get(
        "REPO_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    data_dir  = os.path.join(repo_dir, "fermi_data", "totani")
    mcmc_base = os.path.join(repo_dir, "Totani_paper_check", "mcmc")

    ap.add_argument("--counts",
        default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo",
        default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir",
        default=os.path.join(data_dir, "processed", "templates"))
    ap.add_argument("--ext-mask",
        default=os.path.join(data_dir, "processed", "templates",
                             "mask_extended_sources.fits"))
    ap.add_argument("--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_figs11_12_13"))

    # ------------------------------------------------------------------
    # MCMC directories
    # Fig 11 TL uses the no-halo fit; everything else uses the NFW-rho^2 fit.
    # These correspond to mcmc_results_fig4 and mcmc_results_fig6 in the paper.
    # ------------------------------------------------------------------
    ap.add_argument("--mcmc-dir-nohalo",
        default=os.path.join(mcmc_base, "mcmc_results_fig4"),
        help="No-halo MCMC dir (Fig 11 top-left panel only).")
    ap.add_argument("--mcmc-dir-halo",
        default=os.path.join(mcmc_base, "mcmc_results_fig6"),
        help="NFW-rho^2 with-halo MCMC dir (Fig 11 TR/BL/BR and Figs 12/13).")
    ap.add_argument("--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"], default="f_ml")

    ap.add_argument("--halo-label",
        default="nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
        help="Template key for the NFW-rho^2 halo component in MCMC outputs.")
    ap.add_argument(
        "--halo-template-label",
        default=None,
        help="Template FITS key for the halo component. Defaults to --halo-label.",
    )
    ap.add_argument(
        "--non-halo-labels",
        nargs="+",
        default=None,
        help="Coefficient labels for non-halo components.",
    )
    ap.add_argument(
        "--template-labels",
        nargs="+",
        default=None,
        help="Template FITS labels corresponding to --non-halo-labels.",
    )

    ap.add_argument("--binsz",    type=float, default=0.125)
    ap.add_argument("--roi-lon",  type=float, default=60.0)
    ap.add_argument("--roi-lat",  type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0,
        help="Latitude cut for disk band [deg]. Disk region is shown grey.")

    ap.add_argument("--target-gev", type=float, default=21.0,
        help="Target energy for Fig 11. Nearest bin to this value is used.")
    ap.add_argument(
        "--cmap",
        default="totani",
        help="Colormap / display mode. Use 'dark_transparent' for transparent dark styling with conference-size fonts.",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_cfg = resolve_plot_config(args.cmap)

    # ------------------------------------------------------------------ data
    counts, hdr, _Emin, _Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial
    Ectr_gev = Ectr_mev / 1000.0

    expo_raw, E_expo = read_exposure(args.expo)
    expo = resample_exposure_logE(expo_raw, E_expo, Ectr_mev)

    omega    = pixel_solid_angle_map(wcs, ny, nx, float(args.binsz))
    lon, lat = lonlat_grids(wcs, ny, nx)
    sigma_pix = 1.0 / float(args.binsz)   # 1 deg in pixels

    print(f"Data: nE={nE}, ny={ny}, nx={nx}")
    print("Bin centres (GeV):", np.round(Ectr_gev, 2).tolist())

    # ----------------------------------------------------------------- masks
    roi2d   = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    hilat2d = np.abs(lat) >= args.disk_cut   # True = outside disk band
    disk2d  = roi2d & ~hilat2d               # True = inside disk (rendered grey)

    ext_keep3d = np.ones((nE, ny, nx), dtype=bool)
    if args.ext_mask and os.path.exists(args.ext_mask):
        ext_keep3d = load_mask_any_shape(args.ext_mask, counts.shape)

    # Pixels shown in colour: inside ROI, outside disk, not extended-source masked.
    # Everything else -> NaN -> grey via cmap.set_bad.
    display3d = roi2d[None, :, :] & hilat2d[None, :, :] & ext_keep3d

    # -------------------------------------------------------- templates
    # Both fits share the same non-halo template set.
    # loopA and loopB are kept as separate templates (each with its own mu
    # file); build_model_map_k sums them individually using each one's
    # fitted coefficient, which is equivalent to treating them as one
    # combined loop-I component.
    halo_label = str(args.halo_label)
    halo_template_label = str(args.halo_template_label or args.halo_label)
    non_halo_labels = args.non_halo_labels or [
        "gas", "ics", "iso", "ps", "loopA", "loopB", "fb_pos", "fb_neg"
    ]
    template_labels = args.template_labels or list(non_halo_labels)
    if len(template_labels) != len(non_halo_labels):
        raise SystemExit("--template-labels must have the same length as --non-halo-labels")

    def _load(label):
        path = os.path.join(args.templates_dir, f"mu_{label}_counts.fits")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found: {path}")
        return _read_cube(path, expected_shape=(nE, ny, nx))

    print("Loading non-halo templates …")
    mu_non_halo = {
        str(coeff_label): _load(str(template_label))
        for coeff_label, template_label in zip(non_halo_labels, template_labels)
    }

    print(f"Loading halo template ({halo_template_label}) …")
    mu_halo = _load(halo_template_label)

    # ------------------------------------------------ MCMC coefficients
    def _load_coeffs(mcmc_dir):
        tab = load_mcmc_coeffs_by_label(
            mcmc_dir=str(mcmc_dir), stat=str(args.mcmc_stat), nE=nE
        )
        return dict(tab.coeffs_by_label)

    print("Loading no-halo MCMC coefficients …")
    coeffs_nohalo = _load_coeffs(args.mcmc_dir_nohalo)

    print("Loading with-halo MCMC coefficients …")
    coeffs_halo_dict = _load_coeffs(args.mcmc_dir_halo)

    # Auto-detect halo label if the default key is absent
    if halo_label not in coeffs_halo_dict:
        nfw_keys = sorted(k for k in coeffs_halo_dict
                          if str(k).lower().startswith("nfw_"))
        if len(nfw_keys) == 1:
            halo_label = nfw_keys[0]
            print(f"[info] auto-selected halo label: {halo_label}")
        else:
            raise SystemExit(
                f"Halo label '{halo_label}' not in with-halo MCMC.\n"
                f"nfw_* keys found: {nfw_keys}"
            )

    # ================================================================= Fig 11
    print("\n--- Figure 11 ---")
    k21 = nearest_bin(Ectr_gev, float(args.target_gev))
    print(f"Using bin k={k21} ({Ectr_gev[k21]:.2f} GeV) for Fig 11")
    make_fig11(
        k=k21,
        Ectr_gev=Ectr_gev,
        counts=counts, expo=expo, omega=omega, dE_mev=dE_mev,
        coeffs_nohalo=coeffs_nohalo,
        coeffs_halo=coeffs_halo_dict,
        mu_non_halo=mu_non_halo, mu_halo=mu_halo, halo_label=halo_label,
        display3d=display3d, disk2d=disk2d,
        sigma_pix=sigma_pix, outdir=args.outdir, plot_cfg=plot_cfg,
    )

    # ================================================================= Fig 12
    print("\n--- Figure 12 ---")
    make_fig12_or_13(
        bins_spec=FIG12_BINS, fig_num=12,
        Ectr_gev=Ectr_gev,
        counts=counts, expo=expo, omega=omega, dE_mev=dE_mev,
        coeffs_halo=coeffs_halo_dict,
        mu_non_halo=mu_non_halo, mu_halo=mu_halo, halo_label=halo_label,
        display3d=display3d, disk2d=disk2d,
        sigma_pix=sigma_pix, outdir=args.outdir, plot_cfg=plot_cfg,
    )

    # ================================================================= Fig 13
    print("\n--- Figure 13 ---")
    make_fig12_or_13(
        bins_spec=FIG13_BINS, fig_num=13,
        Ectr_gev=Ectr_gev,
        counts=counts, expo=expo, omega=omega, dE_mev=dE_mev,
        coeffs_halo=coeffs_halo_dict,
        mu_non_halo=mu_non_halo, mu_halo=mu_halo, halo_label=halo_label,
        display3d=display3d, disk2d=disk2d,
        sigma_pix=sigma_pix, outdir=args.outdir, plot_cfg=plot_cfg,
    )

    print("\nDone. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
