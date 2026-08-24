#!/usr/bin/env python3

import argparse
import os
import json
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from totani_helpers.totani_io import (
    load_mask_any_shape,
    lonlat_grids,
    read_counts_and_ebounds,
)
from helpers.fermi_plotting import (
    CONF_LABEL_FS,
    CONF_TICK_FS,
    CONF_TITLE_FS,
)
from helpers.plot_style import current_style_is_transparent, normalise_plot_style, set_paper_style


FONT_PRESETS = {
    "paper": {
        "title": 12.0,
        "label": 10.0,
        "ticks": 9.0,
    },
    "conference": {
        "title": float(CONF_TITLE_FS),
        "label": float(CONF_LABEL_FS),
        "ticks": float(CONF_TICK_FS),
    },
}


def resolve_plot_config(cmap_name: str):
    cmap_key = str(cmap_name).strip()
    try:
        style_key = normalise_plot_style(cmap_key if cmap_key not in {"", "default"} else None, default="paper")
    except ValueError:
        style_key = "paper"
    if style_key != "paper":
        set_paper_style(style=style_key, linewidth=1.5, n_colors=12, cmap_name="plasma")
        return {
            "mode": "conference",
            "fonts": FONT_PRESETS["conference"],
            "grid_cmap": "plasma",
            "corr_cmap": "plasma",
            "transparent": current_style_is_transparent(),
        }

    set_paper_style(style="paper", linewidth=1.5, n_colors=12, cmap_name="plasma")
    return {
        "mode": "paper",
        "fonts": FONT_PRESETS["paper"],
        "grid_cmap": cmap_key if cmap_key else "plasma",
        "corr_cmap": cmap_key if cmap_key else "plasma",
        "transparent": current_style_is_transparent(),
    }


def _safe_lognorm(img: np.ndarray):
    z = np.asarray(img, float)
    pos = z[np.isfinite(z) & (z > 0)]
    if pos.size == 0:
        return None
    vmin = float(np.nanpercentile(pos, 1))
    vmax = float(np.nanpercentile(pos, 99.9))
    if (not np.isfinite(vmin)) or vmin <= 0:
        vmin = float(np.nanmin(pos))
    if (not np.isfinite(vmax)) or vmax <= vmin:
        vmax = float(np.nanmax(pos))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= 0:
        return None
    return LogNorm(vmin=max(vmin, 1e-300), vmax=max(vmax, vmin * 1.0001))


def _read_cube(path: str, expected_shape: Tuple[int, int, int]):
    from astropy.io import fits

    with fits.open(path) as h:
        cube = np.asarray(h[0].data, float)
    if cube.shape != expected_shape:
        raise RuntimeError(f"{path} shape {cube.shape} != expected {expected_shape}")
    return cube


def _corr_matrix(templates_2d: List[np.ndarray], mask2d: np.ndarray) -> np.ndarray:
    vecs = []
    for t in templates_2d:
        v = np.asarray(t, float)[mask2d]
        v = v[np.isfinite(v)]
        if v.size == 0:
            vecs.append(np.zeros(int(np.sum(mask2d)), float))
            continue
        vv = np.asarray(t, float)[mask2d]
        mu = float(np.nanmean(vv))
        sig = float(np.nanstd(vv))
        if (not np.isfinite(sig)) or sig == 0:
            sig = 1.0
        vecs.append((vv - mu) / sig)
    X = np.vstack(vecs)
    return np.corrcoef(X)


def _maybe_git_hash(repo_dir: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        s = (r.stdout or "").strip()
        return s if s else None
    except Exception:
        return None


def _autodetect_ics_subcomponents(templates_dir: str) -> List[Tuple[str, str]]:
    """Return a list of (name, path) for any recognized ICS subcomponent templates.

    This is intentionally conservative: it only adds items if the file exists.
    """
    out: List[Tuple[str, str]] = []
    candidates = {
        "ics_opt": [
            "mu_ics_opt_counts.fits",
            "mu_ics_optical_counts.fits",
        ],
        "ics_ir": [
            "mu_ics_ir_counts.fits",
            "mu_ics_infrared_counts.fits",
        ],
        "ics_cmb": [
            "mu_ics_cmb_counts.fits",
        ],
    }

    for key, fnames in candidates.items():
        for fn in fnames:
            p = os.path.join(str(templates_dir), fn)
            if os.path.exists(p):
                out.append((key, p))
                break
    return out


def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    templates_dir_default = os.path.join(data_dir, "processed", "templates")

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=templates_dir_default)
    ap.add_argument("--ext-mask", default=os.path.join(templates_dir_default, "mask_extended_sources.fits"))

    ap.add_argument("--Egev", type=float, default=21.0)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)

    ap.add_argument(
        "--halo-label",
        default="nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
    )

    ap.add_argument("--ics-opt", default=None)
    ap.add_argument("--ics-ir", default=None)
    ap.add_argument("--ics-cmb", default=None)

    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_template_grid_21gev"))
    ap.add_argument(
        "--cmap",
        default="magma",
        help="Image colormap / display mode. Use 'dark_transparent' for transparent dark styling with conference-size fonts.",
    )
    args = ap.parse_args()
    plot_cfg = resolve_plot_config(args.cmap)

    counts, hdr, _Emin, _Emax, Ectr_mev, _dE = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0
    k = int(np.argmin(np.abs(Ectr_gev - float(args.Egev))))

    lon2d, lat2d = lonlat_grids(wcs, ny, nx)
    roi2d = (np.abs(lon2d) <= float(args.roi_lon)) & (np.abs(lat2d) <= float(args.roi_lat))
    lon_extent = (-float(args.roi_lon), float(args.roi_lon))
    lat_extent = (-float(args.roi_lat), float(args.roi_lat))

    ext_keep3d = np.ones((nE, ny, nx), dtype=bool)
    if args.ext_mask and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), (nE, ny, nx)).astype(bool)

    display_mask2d = roi2d
    point_source_keep2d = ext_keep3d[k]
    corr_mask2d = roi2d & point_source_keep2d

    items: List[Tuple[str, str]] = [
        ("gas", os.path.join(args.templates_dir, "mu_gas_counts.fits")),
        ("ps", os.path.join(args.templates_dir, "mu_ps_counts.fits")),
        ("ics", os.path.join(args.templates_dir, "mu_ics_counts.fits")),
        ("iso", os.path.join(args.templates_dir, "mu_iso_counts.fits")),
        ("loopI", os.path.join(args.templates_dir, "mu_loopI_counts.fits")),
        ("fb_pos", os.path.join(args.templates_dir, "mu_fb_pos_counts.fits")),
        ("fb_neg", os.path.join(args.templates_dir, "mu_fb_neg_counts.fits")),
        ("halo", os.path.join(args.templates_dir, f"mu_{str(args.halo_label)}_counts.fits")),
    ]

    if args.ics_opt:
        items.append(("ics_opt", str(args.ics_opt)))
    if args.ics_ir:
        items.append(("ics_ir", str(args.ics_ir)))
    if args.ics_cmb:
        items.append(("ics_cmb", str(args.ics_cmb)))

    # If not explicitly provided, attempt to autodetect common filenames.
    if (not args.ics_opt) and (not args.ics_ir) and (not args.ics_cmb):
        items.extend(_autodetect_ics_subcomponents(str(args.templates_dir)))

    names: List[str] = []
    maps: List[np.ndarray] = []

    for name, path in items:
        if not os.path.exists(path):
            print(f"[skip] missing: {name} -> {path}")
            continue
        cube = _read_cube(path, expected_shape=(nE, ny, nx))
        img = np.asarray(cube[k], float)
        img = np.where(display_mask2d & point_source_keep2d, img, np.nan)
        names.append(str(name))
        maps.append(img)

    if not maps:
        raise SystemExit("No templates found to plot.")

    os.makedirs(args.outdir, exist_ok=True)

    # Reproducibility outputs: dump config + cache intermediates.
    cfg = {
        "counts": str(args.counts),
        "templates_dir": str(args.templates_dir),
        "ext_mask": str(args.ext_mask),
        "Egev_requested": float(args.Egev),
        "Egev_selected": float(Ectr_gev[k]),
        "k_selected": int(k),
        "roi_lon": float(args.roi_lon),
        "roi_lat": float(args.roi_lat),
        "disk_cut": float(args.disk_cut),
        "halo_label": str(args.halo_label),
        "display_npix": int(np.sum(display_mask2d)),
        "corr_mask2d_npix": int(np.sum(corr_mask2d)),
        "inputs": [{"name": n, "path": p} for (n, p) in items],
        "git_hash": _maybe_git_hash(str(repo_dir)),
    }
    out_cfg = os.path.join(args.outdir, f"template_grid_E{Ectr_gev[k]:.2f}GeV_config.json")
    with open(out_cfg, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    n = len(maps)
    ncol = 4
    nrow = int(np.ceil(n / ncol))

    cell_w = 6.0 if plot_cfg["mode"] == "conference" else 4.6
    cell_h = 4.8 if plot_cfg["mode"] == "conference" else 3.7
    fig, axes = plt.subplots(nrow, ncol, figsize=(cell_w * ncol, cell_h * nrow), squeeze=False)
    for i, (name, img) in enumerate(zip(names, maps)):
        ax = axes[i // ncol][i % ncol]
        norm = _safe_lognorm(img)
        im = ax.imshow(
            img,
            origin="lower",
            cmap=plot_cfg["grid_cmap"],
            norm=norm,
            extent=(lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]),
            aspect="auto",
        )
        ax.set_title(f"{name}  k={k}  E={Ectr_gev[k]:.2f} GeV", fontsize=plot_cfg["fonts"]["title"])
        ax.set_xlabel(r"longitude $l$ [deg]", fontsize=plot_cfg["fonts"]["label"])
        ax.set_ylabel(r"latitude $b$ [deg]", fontsize=plot_cfg["fonts"]["label"])
        ax.set_xlim(*lon_extent)
        ax.set_ylim(*lat_extent)
        ax.tick_params(labelsize=plot_cfg["fonts"]["ticks"])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=plot_cfg["fonts"]["ticks"])

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.tight_layout()
    out_png = os.path.join(args.outdir, f"template_grid_E{Ectr_gev[k]:.2f}GeV.png")
    fig.savefig(out_png, dpi=220, transparent=plot_cfg["transparent"])
    plt.close(fig)
    print("✓ wrote", out_png)

    C = _corr_matrix(maps, mask2d=corr_mask2d)
    side = 0.75 if plot_cfg["mode"] == "conference" else 0.55
    fig = plt.figure(figsize=(side * len(names) + 6.0, side * len(names) + 5.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(C, origin="lower", cmap=plot_cfg["corr_cmap"], vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=plot_cfg["fonts"]["ticks"])
    ax.set_yticklabels(names, fontsize=plot_cfg["fonts"]["ticks"])
    ax.set_title(f"ROI correlation matrix @ {Ectr_gev[k]:.2f} GeV", fontsize=plot_cfg["fonts"]["title"])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr")
    cbar.ax.tick_params(labelsize=plot_cfg["fonts"]["ticks"])
    cbar.set_label("corr", fontsize=plot_cfg["fonts"]["label"])
    fig.tight_layout()
    out_png = os.path.join(args.outdir, f"template_corr_E{Ectr_gev[k]:.2f}GeV.png")
    fig.savefig(out_png, dpi=220, transparent=plot_cfg["transparent"])
    plt.close(fig)
    print("✓ wrote", out_png)

    out_cache = os.path.join(args.outdir, f"template_grid_E{Ectr_gev[k]:.2f}GeV_cache.npz")
    np.savez_compressed(
        out_cache,
        names=np.asarray(names, dtype=object),
        k=np.int32(k),
        Egev=np.float32(Ectr_gev[k]),
        display_mask2d=display_mask2d.astype(np.uint8),
        corr_mask2d=corr_mask2d.astype(np.uint8),
        corr=C.astype(np.float32),
    )
    print("✓ wrote", out_cache)
    print("✓ wrote", out_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
