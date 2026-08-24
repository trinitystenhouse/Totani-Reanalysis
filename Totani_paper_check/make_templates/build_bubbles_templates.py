#!/usr/bin/env python3
"""build_bubbles_templates.py

Construct Totani-style Fermi Bubbles *flat* positive/negative morphology from data residuals.

Outputs (in --out-dir):
- fb_flat_pos_mask.fits : 2D mask (1 inside positive bubbles, 0 outside)
- fb_flat_neg_mask.fits : 2D mask (1 inside negative bubbles, 0 outside)

Optional (if --save-counts-templates):
- fb_flat_pos_counts.fits : 3D counts template cube (nE,ny,nx)
- fb_flat_neg_counts.fits : 3D counts template cube (nE,ny,nx)

Also saves diagnostics:
- bubbles_smoothed_iterXX.png
- bubbles_mask_evolution.png

Notes:
- The Stage 2 background fit is done in the full ROI including the Galactic disk; the later halo/bubble residual
  partition uses the disk-cut selection when requested.
- Morphology is derived from residual flux combined over chosen energy bins.
- All energies are MeV internally.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.bubbles_templates import (
    build_flat_counts_template,
    build_model_counts,
    iterate_bubbles_masks,
    nnls_fit_per_energy,
)
from totani_helpers.fit_utils import E2_from_pred_counts_maps, load_mu_templates_from_fits
from totani_helpers.mcmc_io import save_coeff_table_txt
from totani_helpers.plotting import plot_E2_dnde_multi_totani
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    load_mask_any_shape,
)


def _read_vertices_lonlat(path: str) -> np.ndarray:
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        raise RuntimeError(f"Need >=3 vertices in {path}")
    return np.asarray(pts, dtype=float)


def _polygon_mask_from_lonlat_vertices(wcs, ny: int, nx: int, verts_lonlat_deg: np.ndarray) -> np.ndarray:
    lon = np.mod(np.asarray(verts_lonlat_deg[:, 0], float), 360.0)
    lat = np.asarray(verts_lonlat_deg[:, 1], float)
    x, y = wcs.world_to_pixel_values(lon, lat)
    poly = Path(np.vstack([x, y]).T)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    return poly.contains_points(pts).reshape(ny, nx)


def _parse_energy_bins(arg: str, Ectr_mev: np.ndarray) -> List[int]:
    if str(arg).strip().lower() == "auto":
        targets_gev = [1.5, 4.3]
        Egev = np.asarray(Ectr_mev, float) / 1000.0
        ks = [int(np.argmin(np.abs(Egev - t))) for t in targets_gev]
        ks = sorted(set(ks))
        return ks

    parts = [p.strip() for p in str(arg).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--energy-bins-for-bubbles must be 'auto' or 'Emin,Emax' in GeV")

    Emin_gev = float(parts[0])
    Emax_gev = float(parts[1])
    if Emin_gev <= 0 or Emax_gev <= 0 or Emax_gev < Emin_gev:
        raise ValueError("Invalid energy range")

    Egev = np.asarray(Ectr_mev, float) / 1000.0
    ks = [int(k) for k in np.where((Egev >= Emin_gev) & (Egev <= Emax_gev))[0]]
    if len(ks) == 0:
        # fallback: nearest to mid
        mid = 0.5 * (Emin_gev + Emax_gev)
        ks = [int(np.argmin(np.abs(Egev - mid)))]
    return ks


def _save_mask_fits(out_path: str, mask2d: np.ndarray, hdr_counts) -> None:
    h = hdr_counts.copy()
    h["BUNIT"] = "dimensionless"
    fits.PrimaryHDU(np.asarray(mask2d, np.uint8), header=h).writeto(out_path, overwrite=True)


def _plot_smoothed(out_png: str, smoothed: np.ndarray, pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    img = np.asarray(smoothed, float)
    pos = np.asarray(pos, bool)
    neg = np.asarray(neg, bool)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)

    finite = np.isfinite(img)
    lim = float(np.nanpercentile(np.abs(img[finite]), 99.0)) if finite.any() else 1.0
    lim = max(lim, 1e-30)

    im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.contour(pos.astype(float), levels=[0.5], colors=["gold"], linewidths=1.5)
    ax.contour(neg.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _save_curves_txt(*, out_txt: str, Ectr_mev: np.ndarray, curves):
    Egev = np.asarray(Ectr_mev, float) / 1000.0
    labels = [str(x[0]) for x in curves]
    ys = [np.asarray(x[1], float).reshape(-1) for x in curves]
    nE = int(Egev.shape[0])
    for y in ys:
        if y.shape[0] != nE:
            raise ValueError(f"Curve length mismatch for {out_txt}: got {y.shape[0]} expected {nE}")

    arr = np.column_stack([Egev] + ys)
    header = "E_GeV " + " ".join(labels)
    np.savetxt(out_txt, arr, header=header)


def _posterior_quantiles(*, chain: np.ndarray, logprob: np.ndarray, burn: int, thin: int) -> Dict[str, np.ndarray]:
    nsteps_eff = int(chain.shape[0])
    if burn >= nsteps_eff:
        raise ValueError(f"burn must be < nsteps_done. burn={burn}, nsteps_done={nsteps_eff}")

    post = np.asarray(chain[burn::thin, :, :], float).reshape(-1, chain.shape[-1])
    post_lp = np.asarray(logprob[burn::thin, :], float).reshape(-1)
    if post.shape[0] == 0:
        raise RuntimeError(f"No posterior samples after burn/thin. nsteps_done={nsteps_eff}, burn={burn}, thin={thin}.")

    imax = int(np.nanargmax(post_lp))
    q05, q16, q50, q84, q95 = np.quantile(post, [0.05, 0.16, 0.50, 0.84, 0.95], axis=0)
    return {
        "f_ml": post[imax].copy(),
        "f_p05": q05,
        "f_p16": q16,
        "f_p50": q50,
        "f_p84": q84,
        "f_p95": q95,
        "loglike_ml": np.array(float(post_lp[imax]), dtype=float),
    }


def _save_mcmc_bin_result(
    *,
    outdir: str,
    k: int,
    Ectr_mev: float,
    labels: Sequence[str],
    summary: Dict[str, np.ndarray],
    acceptance_fraction: np.ndarray,
    used_emcee: bool,
    iso_target_e2: float,
    nwalkers: int,
    nsteps: int,
    burn: int,
    thin: int,
    chain: Optional[np.ndarray] = None,
    logprob: Optional[np.ndarray] = None,
    thin_save: int = 10,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, f"mcmc_results_k{k:02d}.npz")
    outfile_tmp = outfile + ".tmp.npz"
    save_dict = {
        "labels": np.asarray(list(labels)),
        "f_ml": np.asarray(summary["f_ml"], np.float32),
        "f_p05": np.asarray(summary["f_p05"], np.float32),
        "f_p16": np.asarray(summary["f_p16"], np.float32),
        "f_p50": np.asarray(summary["f_p50"], np.float32),
        "f_p84": np.asarray(summary["f_p84"], np.float32),
        "f_p95": np.asarray(summary["f_p95"], np.float32),
        "loglike_ml": np.float32(float(summary["loglike_ml"])),
        "acceptance_fraction": np.asarray(acceptance_fraction, np.float32),
        "used_emcee": bool(used_emcee),
        "energy_bin": int(k),
        "Ectr_mev": np.float32(float(Ectr_mev)),
        "iso_target_e2": np.float32(float(iso_target_e2)),
        "burn": int(burn),
        "thin": int(thin),
        "thin_save": int(thin_save),
    }
    thin_save = max(1, int(thin_save))
    if chain is not None:
        save_dict["chain"] = np.asarray(chain[::thin_save, :, :], np.float32)
    if logprob is not None:
        save_dict["logprob"] = np.asarray(logprob[::thin_save, :], np.float32)

    try:
        if os.path.exists(outfile_tmp):
            os.remove(outfile_tmp)
        np.savez_compressed(outfile_tmp, **save_dict)
        os.replace(outfile_tmp, outfile)
    finally:
        if os.path.exists(outfile_tmp):
            try:
                os.remove(outfile_tmp)
            except Exception:
                pass

    txtfile = os.path.join(outdir, f"mcmc_results_k{k:02d}.txt")
    txtfile_tmp = txtfile + ".tmp"
    try:
        if os.path.exists(txtfile_tmp):
            os.remove(txtfile_tmp)
        with open(txtfile_tmp, "w") as f:
            f.write("MCMC Fit Results\n")
            f.write(f"Energy bin: k={k}, E_ctr={float(Ectr_mev):.1f} MeV ({float(Ectr_mev)/1000.0:.3f} GeV)\n")
            f.write(f"Components: {', '.join(str(x) for x in labels)}\n")
            f.write(f"MCMC: {int(nwalkers)} walkers, {int(nsteps)} steps, burn={int(burn)}, thin={int(thin)}\n")
            f.write(f"Used emcee: {bool(used_emcee)}\n")
            f.write(f"iso init target E^2 dN/dE: {float(iso_target_e2):.6e}\n")
            f.write(f"Acceptance fraction: mean={float(np.mean(acceptance_fraction)):.3f}\n")
            f.write(f"Log-likelihood (ML): {float(summary['loglike_ml']):.3f}\n\n")
            f.write(f"Saved chain: {chain is not None} (thin_save={thin_save})\n")
            f.write(f"Saved logprob: {logprob is not None} (thin_save={thin_save})\n\n")
            f.write(
                f"{'Component':<50s}  {'ML':>12s}  {'p05':>12s}  {'p16':>12s}  {'p50':>12s}  {'p84':>12s}  {'p95':>12s}\n"
            )
            f.write("-" * 128 + "\n")
            for i, lab in enumerate(labels):
                f.write(
                    f"{str(lab):<50s}  {float(summary['f_ml'][i]):+12.6e}  {float(summary['f_p05'][i]):+12.6e}  "
                    f"{float(summary['f_p16'][i]):+12.6e}  {float(summary['f_p50'][i]):+12.6e}  "
                    f"{float(summary['f_p84'][i]):+12.6e}  {float(summary['f_p95'][i]):+12.6e}\n"
                )
        os.replace(txtfile_tmp, txtfile)
    finally:
        if os.path.exists(txtfile_tmp):
            try:
                os.remove(txtfile_tmp)
            except Exception:
                pass


def _save_mcmc_coeff_tables(
    *,
    outdir: str,
    labels: Sequence[str],
    Ectr_mev: np.ndarray,
    summaries_by_stat: Dict[str, np.ndarray],
) -> None:
    coeffs_by_stat = {
        stat: {str(lab): np.asarray(summaries_by_stat[stat][:, j], float) for j, lab in enumerate(labels)}
        for stat in summaries_by_stat
    }
    x_k = np.arange(len(Ectr_mev), dtype=float)
    x_gev = np.asarray(Ectr_mev, float) / 1000.0

    for stat, coeffs_by_label in coeffs_by_stat.items():
        save_coeff_table_txt(
            out_txt=os.path.join(outdir, f"mcmc_coefficients_{stat}_by_k.txt"),
            x=x_k,
            coeffs_by_label=coeffs_by_label,
            keys=list(labels),
            x_label="k",
        )
        save_coeff_table_txt(
            out_txt=os.path.join(outdir, f"mcmc_coefficients_{stat}_by_EGeV.txt"),
            x=x_gev,
            coeffs_by_label=coeffs_by_label,
            keys=list(labels),
            x_label="E_GeV",
        )


def _plot_fig2_fig3_like_from_nnls(
    *,
    outdir: str,
    fig_label: str,
    Ectr_mev: np.ndarray,
    dE_mev: np.ndarray,
    expo: np.ndarray,
    omega: np.ndarray,
    lat2d: np.ndarray,
    base_mask2d: np.ndarray,
    disk_cut_deg_for_plot: float,
    coeffs_all: np.ndarray,
    templates_counts: Dict[str, np.ndarray],
    labels: List[str],
    exclude_disk_in_plot: bool,
    fit_method: str = "MCMC",
) -> None:
    os.makedirs(outdir, exist_ok=True)

    mask2d = np.asarray(base_mask2d, bool)
    if bool(exclude_disk_in_plot):
        mask2d = mask2d & (np.abs(np.asarray(lat2d, float)) >= float(disk_cut_deg_for_plot))

    curves = []
    model_total = np.zeros_like(np.asarray(templates_counts[labels[0]], float))

    def _pred_cube_for(lab: str) -> np.ndarray:
        j = labels.index(lab)
        return np.asarray(coeffs_all[:, j], float)[:, None, None] * np.asarray(templates_counts[lab], float)

    comp_to_plot: List[str] = []
    if ("loopA" in labels) and ("loopB" in labels):
        comp_to_plot = [x for x in labels if x not in ("loopA", "loopB")]
        comp_to_plot.append("loopI")
    else:
        comp_to_plot = list(labels)

    for name in comp_to_plot:
        if name == "loopI":
            pred = _pred_cube_for("loopA") + _pred_cube_for("loopB")
        else:
            pred = _pred_cube_for(name)

        model_total += pred
        E2 = E2_from_pred_counts_maps(
            pred_counts_map=pred,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=mask2d,
        )
        curves.append((name, E2))

    E2_tot = E2_from_pred_counts_maps(
        pred_counts_map=model_total,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask2d=mask2d,
    )
    curves.append(("total", E2_tot))

    out_png = os.path.join(outdir, f"{fig_label}.png")
    out_txt = os.path.join(outdir, f"{fig_label}.txt")
    title = f"{fit_method} background components"
    if bool(exclude_disk_in_plot):
        title = f"{title}, |b|>={float(disk_cut_deg_for_plot):g} deg"

    plot_E2_dnde_multi_totani(Ectr_mev, curves, out_png=out_png, title=title)
    _save_curves_txt(out_txt=out_txt, Ectr_mev=Ectr_mev, curves=curves)


def _mcmc_fit_per_energy(
    *,
    counts: np.ndarray,
    templates_counts: Dict[str, np.ndarray],
    labels: List[str],
    expo: np.ndarray,
    omega: np.ndarray,
    dE_mev: np.ndarray,
    Ectr_mev: np.ndarray,
    fit2d: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    roi_lon: float,
    roi_lat: float,
    cell_deg: float,
    nwalkers: int,
    nsteps: int,
    burn: int,
    thin: int,
    iso_target_e2: float,
    negative_keys: List[str],
    require_autocorr: bool,
    early_stop: bool,
    autocorr_target: float,
    autocorr_check_every: int,
    autocorr_min_steps: int,
    mcmc_results_dir: Optional[str] = None,
    save_chain: bool = True,
    save_logprob: bool = True,
    thin_save: int = 10,
    progress: bool = True,
    verbosity: int = 1,
) -> Dict[str, np.ndarray]:
    """Run cell-level Poisson MCMC for each energy bin; optionally save results in run_mcmc format."""
    # The reorg split these two apart: `run_mcmc` stayed in
    # Totani_paper_check/mcmc, but `mcmc_helper` moved to
    # Totani_reanalysis/mcmc. Both are imported by bare name below, so both
    # parents have to be on sys.path -- with only the first, a direct
    # `python build_bubbles_templates.py` (which is how build_templates.sh
    # invokes it) dies here with ModuleNotFoundError: mcmc_helper.
    # systematics/rebuild_bubbles_per_variant.py works around the same split by
    # setting PYTHONPATH for its subprocess; fixing it here makes that
    # workaround redundant rather than load-bearing.
    _here = os.path.dirname(__file__)
    mcmc_dirs = [
        os.path.abspath(os.path.join(_here, "..", "mcmc")),                    # run_mcmc
        os.path.abspath(os.path.join(_here, "..", "..", "Totani_reanalysis", "mcmc")),  # mcmc_helper
    ]
    for mcmc_dir in mcmc_dirs:
        if mcmc_dir not in sys.path:
            sys.path.insert(0, mcmc_dir)

    from mcmc_helper import totani_bounds, totani_mcmc_fit
    from run_mcmc import (
        build_cell_id_map,
        aggregate_to_cells,
        build_totani_init_for_mu_counts,
    )

    nE, ny, nx = counts.shape
    nComp = len(labels)

    cell_id_map, ncells = build_cell_id_map(
        lon=lon2d, lat=lat2d, roi_lon=roi_lon, roi_lat=roi_lat, cell_deg=cell_deg
    )
    fit2d_b = np.asarray(fit2d, bool)
    bounds = totani_bounds(labels, negative_keys=negative_keys)

    coeffs_all = np.zeros((nE, nComp), dtype=float)
    f_p05_all = np.zeros((nE, nComp), dtype=float)
    f_p16_all = np.zeros((nE, nComp), dtype=float)
    f_p50_all = np.zeros((nE, nComp), dtype=float)
    f_p84_all = np.zeros((nE, nComp), dtype=float)
    f_p95_all = np.zeros((nE, nComp), dtype=float)
    loglike_all = np.full(nE, np.nan, dtype=float)

    for k in range(nE):
        Cobs_vec, _ = aggregate_to_cells(
            x2d=counts[k], mask2d=fit2d_b, cell_id_map=cell_id_map, ncells=ncells
        )
        denom_k = np.asarray(expo[k], float) * np.asarray(omega, float) * float(dE_mev[k])
        denom_cells, _ = aggregate_to_cells(
            x2d=denom_k, mask2d=fit2d_b, cell_id_map=cell_id_map, ncells=ncells
        )
        mu_list = []
        for lab in labels:
            t_k = np.asarray(templates_counts[lab], float)[k]
            mu_cells, _ = aggregate_to_cells(
                x2d=t_k, mask2d=fit2d_b, cell_id_map=cell_id_map, ncells=ncells
            )
            mu_list.append(mu_cells)
        mu = np.array(mu_list)  # (nComp, ncells)

        f0, _ = build_totani_init_for_mu_counts(
            labels=labels,
            mu=mu,
            denom_vec=denom_cells,
            Ectr_mev=float(Ectr_mev[k]),
            iso_target_E2=iso_target_e2,
            verbosity=0,
        )

        res = totani_mcmc_fit(
            Cobs=Cobs_vec,
            mu=mu,
            labels=labels,
            f_init=f0,
            bounds=bounds,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burn=burn,
            thin=thin,
            require_autocorr=require_autocorr,
            early_stop=early_stop,
            autocorr_target=autocorr_target,
            autocorr_check_every=autocorr_check_every,
            autocorr_min_steps=autocorr_min_steps,
            progress=progress,
            verbosity=verbosity,
            poisson_negative_mode="clip",
            poisson_cexp_floor=1e-3,
        )
        summary = _posterior_quantiles(chain=res.chain, logprob=res.logprob, burn=burn, thin=thin)
        coeffs_all[k] = np.asarray(summary["f_ml"], float)
        f_p05_all[k] = np.asarray(summary["f_p05"], float)
        f_p16_all[k] = np.asarray(summary["f_p16"], float)
        f_p50_all[k] = np.asarray(summary["f_p50"], float)
        f_p84_all[k] = np.asarray(summary["f_p84"], float)
        f_p95_all[k] = np.asarray(summary["f_p95"], float)
        loglike_all[k] = float(summary["loglike_ml"])

        if mcmc_results_dir:
            _save_mcmc_bin_result(
                outdir=mcmc_results_dir,
                k=k,
                Ectr_mev=float(Ectr_mev[k]),
                labels=labels,
                summary=summary,
                acceptance_fraction=np.asarray(res.acceptance_fraction, float),
                used_emcee=bool(res.used_emcee),
                iso_target_e2=float(iso_target_e2),
                nwalkers=int(nwalkers),
                nsteps=int(nsteps),
                burn=int(burn),
                thin=int(thin),
                chain=np.asarray(res.chain, float) if save_chain else None,
                logprob=np.asarray(res.logprob, float) if save_logprob else None,
                thin_save=int(thin_save),
            )
        if verbosity >= 1:
            fml_str = "  ".join(f"{lab}={summary['f_ml'][j]:.3e}" for j, lab in enumerate(labels))
            print(f"[mcmc k={k:02d} E={float(Ectr_mev[k])/1000:.2f}GeV]  {fml_str}")

    summaries = {
        "f_ml": coeffs_all,
        "f_p05": f_p05_all,
        "f_p16": f_p16_all,
        "f_p50": f_p50_all,
        "f_p84": f_p84_all,
        "f_p95": f_p95_all,
        "loglike_ml": loglike_all,
    }
    if mcmc_results_dir:
        _save_mcmc_coeff_tables(
            outdir=mcmc_results_dir,
            labels=labels,
            Ectr_mev=Ectr_mev,
            summaries_by_stat={k: v for k, v in summaries.items() if k != "loglike_ml"},
        )

    return summaries


def _smooth_nan(arr2d: np.ndarray, sigma_pix: float) -> np.ndarray:
    nan = ~np.isfinite(arr2d)
    src = np.where(nan, 0.0, arr2d)
    wt = np.where(nan, 0.0, 1.0)
    num = gaussian_filter(src, sigma=sigma_pix, mode="constant", cval=0.0)
    den = gaussian_filter(wt, sigma=sigma_pix, mode="constant", cval=0.0)
    out = np.full_like(arr2d, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def _read_extended_source_coords(psc_path: str):
    with fits.open(psc_path) as f:
        psc = f[1].data
        cols = {c.upper() for c in psc.columns.names}

        if "EXTENDED_SOURCE_NAME" in cols:
            names = np.array(
                [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in psc["EXTENDED_SOURCE_NAME"]]
            )
            is_ext = np.array([len(s.strip()) > 0 and s.strip().upper() not in ("", "NONE") for s in names])
        elif "EXTENDED" in cols:
            is_ext = np.asarray(psc["EXTENDED"]).astype(bool)
        else:
            return [], [], []

        psc_ext = psc[is_ext]
        lon = np.asarray(psc_ext["GLON"], float)
        lat = np.asarray(psc_ext["GLAT"], float)
        if "SEMIMAJOR_DEG" in cols:
            r = 2.0 * np.asarray(psc_ext["SEMIMAJOR_DEG"], float)
        elif "R95_SEMIMAJOR" in cols:
            r = 2.0 * np.asarray(psc_ext["R95_SemiMajor"], float)
        else:
            r = np.ones(is_ext.sum()) * 1.0
    return lon, lat, r


def _plot_totani_fig1_like(
    *,
    out_png: str,
    counts: np.ndarray,
    wcs: WCS,
    expo: np.ndarray,
    omega: np.ndarray,
    dE_mev: np.ndarray,
    Ectr_mev: np.ndarray,
    roi2d: np.ndarray,
    lat2d: np.ndarray,
    flat_mask2d: np.ndarray,
    coeffs_all: np.ndarray,
    templates_by_label: Dict[str, np.ndarray],
    labels: List[str],
    binsz_deg: float,
    disk_cut_deg: float,
    target_mev: List[float],
    vmax_mev: List[float],
    psc_path: str | None,
) -> None:
    if "fb_flat" not in labels:
        raise RuntimeError("Cannot build Totani Fig.1-style bubble image without 'fb_flat' in --components")

    ext_lon, ext_lat, ext_rad = [], [], []
    if psc_path and os.path.exists(str(psc_path)):
        try:
            ext_lon, ext_lat, ext_rad = _read_extended_source_coords(str(psc_path))
        except Exception:
            ext_lon, ext_lat, ext_rad = [], [], []

    non_fb_labels = [l for l in labels if l != "fb_flat"]

    sigma_pix = 1.0 / float(binsz_deg)

    fig = plt.figure(figsize=(12.0, 5.2))
    vmaxes = list(vmax_mev)
    if len(vmaxes) < len(target_mev):
        vmaxes = vmaxes + [vmaxes[-1]] * (len(target_mev) - len(vmaxes))

    for panel_i, (t_mev, vmax) in enumerate(zip(target_mev, vmaxes)):
        k = int(np.argmin(np.abs(np.asarray(Ectr_mev, float) - float(t_mev))))

        model_excl_fb = np.zeros_like(flat_mask2d, dtype=float)
        for lab in non_fb_labels:
            j = labels.index(lab)
            model_excl_fb += float(coeffs_all[k, j]) * np.asarray(templates_by_label[lab], float)[k]
        bubble_counts_k = np.asarray(counts[k], float) - model_excl_fb

        denom = np.asarray(expo[k], float) * np.asarray(omega, float) * float(dE_mev[k])
        flux2d = np.full_like(bubble_counts_k, np.nan, dtype=float)
        ok = (denom > 0) & np.isfinite(denom) & np.asarray(roi2d, bool)
        flux2d[ok] = bubble_counts_k[ok] / denom[ok]
        flux2d[np.abs(np.asarray(lat2d, float)) < float(disk_cut_deg)] = np.nan

        flux_smooth = _smooth_nan(flux2d, sigma_pix)

        ax = fig.add_subplot(1, len(target_mev), panel_i + 1, projection=wcs)
        im = ax.imshow(
            flux_smooth,
            origin="lower",
            cmap="RdBu_r",
            vmin=-float(vmax),
            vmax=float(vmax),
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, fraction=0.04)
        cbar.set_label(r"flux [cm$^{-2}$s$^{-1}$sr$^{-1}$MeV$^{-1}$]", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(f"{float(Ectr_mev[k]) / 1000:.2g} GeV", fontsize=10)
        ax.set_xlabel("longitude $l$ [deg]", fontsize=9)
        ax.set_ylabel("latitude $b$ [deg]", fontsize=9)

        try:
            ax.contour(
                np.asarray(flat_mask2d, bool).astype(float),
                levels=[0.5],
                colors="grey",
                linewidths=0.8,
                alpha=0.9,
                transform=ax.get_transform("pixel"),
            )
        except Exception:
            pass

        ax_transform = ax.get_transform("galactic")
        for glon, glat, rad in zip(ext_lon, ext_lat, ext_rad):
            if abs(glon) <= 65 and abs(glat) <= 65:
                c = Circle(
                    (float(glon), float(glat)),
                    radius=float(rad),
                    transform=ax_transform,
                    color="grey",
                    alpha=0.6,
                    zorder=5,
                )
                ax.add_patch(c)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_mask_only(out_png: str, pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    pos = np.asarray(pos, bool)
    neg = np.asarray(neg, bool)
    img = np.zeros_like(pos, dtype=float)
    img[pos] = 1.0
    img[neg] = -1.0

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("mask: +1 pos, -1 neg")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_lonlat_overlay(
    out_png: str,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    smoothed: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    roi2d: np.ndarray,
    title: str,
) -> None:
    lon2d = np.asarray(lon2d, float)
    lat2d = np.asarray(lat2d, float)
    img = np.asarray(smoothed, float).copy()
    pos = np.asarray(pos, bool)
    neg = np.asarray(neg, bool)
    roi2d = np.asarray(roi2d, bool)

    img[~roi2d] = np.nan
    finite = np.isfinite(img)
    lim = float(np.nanpercentile(np.abs(img[finite]), 99.0)) if finite.any() else 1.0
    lim = max(lim, 1e-30)

    fig = plt.figure(figsize=(7.8, 5.2))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(lon2d, lat2d, img, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax.contour(lon2d, lat2d, pos.astype(float), levels=[0.5], colors=["gold"], linewidths=1.5)
    ax.contour(lon2d, lat2d, neg.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.5)
    ax.set_xlabel("l [deg]")
    ax.set_ylabel("b [deg]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    repo_dir = os.environ.get(
        "REPO_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

    ap.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(data_dir, "processed", "templates"))

    ap.add_argument("--srcmask", default=os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits"))

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut-deg", type=float, default=0.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument(
        "--iso-target-e2",
        type=float,
        default=1e-4,
        help="Reference E^2 dN/dE [MeV cm^-2 s^-1 sr^-1] used to set f=1 normalization.",
    )
    ap.add_argument(
        "--rescale-to-data-sum",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="After building mu, rescale each energy plane so sum(mu)=sum(data) within ROI/srcmask/data coverage.",
    )
    ap.add_argument(
        "--report-bin",
        type=int,
        default=None,
        help="If set, print per-bin diagnostics for this energy bin index.",
    )

    here = os.path.dirname(__file__)
    ap.add_argument("--verts-north", default=os.path.join(repo_dir, "fermi_data", "totani", "templates", "bubble_vertices_north.txt"))
    ap.add_argument("--verts-south", default=os.path.join(repo_dir, "fermi_data", "totani", "templates", "bubble_vertices_south.txt"))
    ap.add_argument(
        "--restrict-to-vertices",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Restrict morphology iteration to the union of the north/south vertices-defined polygons.",
    )

    ap.add_argument(
        "--energy-bins-for-bubbles",
        default="auto",
        help="Either 'auto' or 'Emin,Emax' in GeV. Used to combine residual flux into morphology image.",
    )

    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--smooth-sigma-deg", type=float, default=2.0)
    ap.add_argument("--pos-thresh", type=float, default=0.0)
    ap.add_argument("--neg-thresh", type=float, default=0.0)
    ap.add_argument("--morph-open-deg", type=float, default=1.0)
    ap.add_argument("--morph-close-deg", type=float, default=2.0)

    ap.add_argument(
        "--keep-by-hemisphere",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If enabled, keep the largest connected component separately in the north and south hemispheres.",
    )

    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--stop-frac", type=float, default=0.01)

    ap.add_argument("--out-dir", default=os.path.join(data_dir, "processed", "templates"))
    ap.add_argument("--save-counts-templates", action="store_true")
    ap.add_argument("--out-plots", default=os.path.join(here, "plots"))

    ap.add_argument(
        "--plot-fig1-maps",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, plot a Totani Fig.1-style two-panel bubble image using the NNLS coefficients computed here.",
    )
    ap.add_argument(
        "--fig1-disk-cut-deg",
        type=float,
        default=10.0,
    )
    ap.add_argument(
        "--fig1-target-mev",
        type=float,
        nargs="+",
        default=[1500.0, 4300.0],
    )
    ap.add_argument(
        "--fig1-vmax",
        type=float,
        nargs="+",
        default=[5e-10, 5e-11],
    )
    ap.add_argument(
        "--psc",
        default=os.path.join(repo_dir, "fermi_data", "totani", "templates", "gll_psc_v35.fit"),
    )

    ap.add_argument(
        "--plot-fig2-3",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, plot Fig.2/Fig.3-like E^2 dN/dE component curves using NNLS coefficients.",
    )
    ap.add_argument(
        "--fig23-disk-cut-deg",
        type=float,
        default=0.0,
        help="Disk cut used for the Fig.3-style plot mask (|b|>=cut).",
    )

    ap.add_argument(
        "--build-flat-only",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Build only the flat bubble template from vertices (Stage 1). "
            "Skips pos/neg morphology extraction. Use this first, then run again "
            "with fb_flat in --components to extract pos/neg (Stage 2)."
        ),
    )

    ap.add_argument(
        "--force-bubbles-close-residual",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Force the positive and negative bubble templates to partition the post-background residual exactly "
            "in the construction bin, rather than treating them as independent free templates in a later fit."
        ),
    )
    ap.add_argument(
        "--k-construct",
        type=int,
        default=2,
        help="Energy-bin index to enforce exact residual closure (default: 2).",
    )

    ap.add_argument(
        "--resid-smooth-deg",
        type=float,
        default=1.0,
        help="Smoothing scale (Gaussian sigma) in degrees applied to the residual map before pos/neg splitting in --force-bubbles-close-residual mode.",
    )

    ap.add_argument(
        "--bubbles-split-mode",
        choices=["closure", "totani"],
        default="closure",
        help=(
            "How to split residual into fb_pos/fb_neg in --force-bubbles-close-residual mode. "
            "'closure' assigns UNSMOOTHED residual amplitudes into regions defined by the smoothed residual sign "
            "(exact closure at k-construct). "
            "'totani' saves Totani-style sign-definite residual maps by splitting the 1-degree SMOOTHED residual "
            "by sign per energy bin (no exact closure)."
        ),
    )

    ap.add_argument(
        "--use-mcmc",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Use cell-level Poisson MCMC (via totani_mcmc_fit) instead of pixel NNLS "
            "to fit background components. Produces correct iso normalization."
        ),
    )
    ap.add_argument("--mcmc-nwalkers", type=int, default=64, help="Number of emcee walkers for background MCMC.")
    ap.add_argument("--mcmc-nsteps", type=int, default=10000, help="Number of MCMC steps for background fit.")
    ap.add_argument("--mcmc-burn", type=int, default=2000, help="Burn-in steps for background MCMC.")
    ap.add_argument("--mcmc-thin", type=int, default=5, help="Thinning factor for background MCMC.")
    ap.add_argument("--mcmc-cell-deg", type=float, default=10.0, help="Cell size in degrees for MCMC aggregation.")
    ap.add_argument(
        "--mcmc-results-dir",
        default=os.path.join(repo_dir, "Totani_paper_check", "mcmc", "mcmc_results_fig2_3"),
        help="Directory where the Stage 2 Fig.2/3-style MCMC results are saved.",
    )
    ap.add_argument(
        "--save-chain",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save a thinned MCMC chain in each mcmc_results_kXX.npz so later figures can reuse this run.",
    )
    ap.add_argument(
        "--save-logprob",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save a thinned per-sample log-probability array in each mcmc_results_kXX.npz.",
    )
    ap.add_argument(
        "--thin-save",
        type=int,
        default=10,
        help="Thinning factor used when saving chain/logprob to the Fig.2/3 MCMC result files.",
    )
    ap.add_argument(
        "--require-autocorr",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Require the background MCMC autocorr-time convergence check, matching the Fig.2/3 config.",
    )
    ap.add_argument(
        "--early-stop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run background MCMC in chunks and stop once the autocorr convergence target is met, matching the Fig.2/3 config.",
    )
    ap.add_argument(
        "--autocorr-target",
        type=float,
        default=50.0,
        help="Background MCMC target for N/tau when checking convergence.",
    )
    ap.add_argument(
        "--autocorr-check-every",
        type=int,
        default=20000,
        help="When --early-stop is enabled, check autocorr convergence every N steps (Fig.2/3 default).",
    )
    ap.add_argument(
        "--autocorr-min-steps",
        type=int,
        default=30000,
        help="When --early-stop is enabled, wait this many steps before attempting autocorr checks (Fig.2/3 default).",
    )
    ap.add_argument(
        "--mcmc-negative-keys",
        nargs="+",
        default=["nfw", "fb_neg"],
        help="Template labels allowed to have negative coefficients in MCMC.",
    )

    ap.add_argument(
        "--components",
        nargs="+",
        default=[
            "gas",
            "ics",
            "iso",
            "ps",
            "loopA",
            "loopB",
            "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        ],
        help="Base background components to fit (must exist as mu_{name}_counts.fits).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_plots, exist_ok=True)

    counts, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape

    expo_raw, E_expo_mev = read_exposure(args.expo)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != counts.shape:
        raise SystemExit(f"Exposure shape {expo.shape} != counts shape {counts.shape}")

    wcs = WCS(hdr).celestial
    omega = pixel_solid_angle_map(wcs, ny, nx, float(args.binsz))
    lon2d, lat2d = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon2d) <= float(args.roi_lon)) & (np.abs(lat2d) <= float(args.roi_lat))

    src_keep_3d = load_mask_any_shape(args.srcmask, counts.shape) if args.srcmask else np.ones_like(counts, bool)
    src_keep_2d = np.all(np.asarray(src_keep_3d, bool), axis=0)

    fit2d = roi2d & src_keep_2d

    # Load base background templates in counts-space
    labels = list(args.components)
    mu_list, _hdrs = load_mu_templates_from_fits(
        template_dir=str(args.templates_dir),
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )

    templates_counts: Dict[str, np.ndarray] = {lab: np.asarray(mu, float) for lab, mu in zip(labels, mu_list)}

    # parse energy bins for bubbles morphology
    k_bins = _parse_energy_bins(args.energy_bins_for_bubbles, Ectr_mev)
    print(f"[bubbles] using k bins {k_bins} (Ectr_GeV={np.asarray(Ectr_mev)[k_bins]/1000.0})")

    # Check for two-stage construction mode
    build_flat_only = bool(args.build_flat_only)
    if build_flat_only:
        print("[bubbles] STAGE 1: Building flat bubble template only (no pos/neg extraction)")
    else:
        print("[bubbles] STAGE 2: Building pos/neg bubble templates")
        if "fb_flat" in labels:
            print("[bubbles] fb_flat included in background - extracting residual pos/neg morphology")
        else:
            print("[bubbles] WARNING: fb_flat NOT in background components!")
            print("[bubbles] According to Totani methodology, fb_flat should be included.")

    if bool(args.restrict_to_vertices):
        if (not os.path.exists(str(args.verts_north))) or (not os.path.exists(str(args.verts_south))):
            raise SystemExit("Vertices file not found. Pass --no-restrict-to-vertices or set --verts-north/--verts-south.")
        verts_n = _read_vertices_lonlat(str(args.verts_north))
        verts_s = _read_vertices_lonlat(str(args.verts_south))
        boundary2d = _polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_n) | _polygon_mask_from_lonlat_vertices(
            wcs, ny, nx, verts_s
        )
    else:
        boundary2d = None

    # Default thresholds if not specified: use symmetric percentiles of initial residual flux
    # We do this by taking a quick 1-iter run with pos/neg=0, then setting thresholds to percentiles.
    pos_thr = float(args.pos_thresh)
    neg_thr = float(args.neg_thresh)

    if (pos_thr == 0.0) and (neg_thr == 0.0):
        # Use small non-zero defaults; adaptive thresholding will adjust if empty.
        # neg must be strictly negative, otherwise (smoothed < neg_thr) never selects negatives.
        pos_thr = 0.0
        neg_thr = -1e-30
    elif neg_thr == 0.0:
        neg_thr = -1e-30
    elif neg_thr > 0.0:
        neg_thr = -abs(neg_thr)

    # Stage 1: Build flat-only - skip morphology iteration
    if build_flat_only:
        # Use vertex-defined boundaries directly for flat mask
        if not bool(args.restrict_to_vertices):
            raise SystemExit("--build-flat-only requires --restrict-to-vertices")
        if (not os.path.exists(str(args.verts_north))) or (not os.path.exists(str(args.verts_south))):
            raise SystemExit("Vertices files not found for --build-flat-only mode")
        
        verts_n = _read_vertices_lonlat(str(args.verts_north))
        verts_s = _read_vertices_lonlat(str(args.verts_south))
        wcs_obj = WCS(hdr).celestial
        flat_mask = _polygon_mask_from_lonlat_vertices(wcs_obj, ny, nx, verts_n) | \
                    _polygon_mask_from_lonlat_vertices(wcs_obj, ny, nx, verts_s)
        
        pos_mask = np.zeros((ny, nx), dtype=bool)
        neg_mask = np.zeros((ny, nx), dtype=bool)
        hist = []  # No iteration history in flat-only mode
        
        print(f"[bubbles] Flat mask created from vertices: {int(np.sum(flat_mask))} pixels")
    else:
        # Stage 2: Extract pos/neg morphology (original behavior)
        pos_mask, neg_mask, hist = iterate_bubbles_masks(
            counts=counts,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            iso_target_E2=float(args.iso_target_e2),
            templates_counts=templates_counts,
            base_component_order=labels,
            roi2d=roi2d,
            srcmask2d=src_keep_2d,
            boundary2d=boundary2d,
            lat_deg_2d=lat2d,
            disk_cut_deg=float(args.disk_cut_deg),
            k_bins_for_bubbles=k_bins,
            n_iter=int(args.n_iter),
            smooth_sigma_deg=float(args.smooth_sigma_deg),
            binsz_deg=float(args.binsz),
            pos_thresh=float(pos_thr),
            neg_thresh=float(neg_thr),
            morph_open_deg=float(args.morph_open_deg),
            morph_close_deg=float(args.morph_close_deg),
            alpha=float(args.alpha),
            stop_frac=float(args.stop_frac),
            ridge=0.0,
            include_bubbles_in_fit=(not bool(args.force_bubbles_close_residual)),
            keep_by_hemisphere=bool(args.keep_by_hemisphere),
        )
        
    # Save masks
     # In flat-only mode, flat_mask is already set from vertices
    # In pos/neg mode, construct from union or boundary
    if not build_flat_only:
        pos_out = os.path.join(args.out_dir, "fb_pos_mask.fits")
        neg_out = os.path.join(args.out_dir, "fb_neg_mask.fits")
        _save_mask_fits(pos_out, pos_mask.astype(np.uint8), hdr)
        _save_mask_fits(neg_out, neg_mask.astype(np.uint8), hdr)
        print("[out]", pos_out)
        print("[out]", neg_out)

    if boundary2d is None:
        flat_mask = (np.asarray(pos_mask, bool) | np.asarray(neg_mask, bool))
    else:
        flat_mask = np.asarray(boundary2d, bool)
    flat_out = os.path.join(args.out_dir, "fb_flat_mask.fits")
    _save_mask_fits(flat_out, flat_mask.astype(np.uint8), hdr)

    print("[out]", flat_out)


    iso_target_E2 = float(args.iso_target_e2)

    mu_flat = build_flat_counts_template(
        mask2d=flat_mask,
        roi2d=roi2d,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        iso_target_E2=iso_target_E2,
    )

    # Skip forced closure in flat-only mode
    if bool(args.force_bubbles_close_residual) and not build_flat_only:
        k2 = int(args.k_construct)
        if k2 < 0 or k2 >= nE:
            raise ValueError(f"k-construct {k2} out of range [0,{nE-1}]")

        if "fb_flat" not in labels:
            print("[WARNING] fb_flat not in --components. Totani's method requires fb_flat in the background")
            print("[WARNING] so its contribution can be added back to the residuals (= full bubble image).")

        # ── Get background coefficients for all energy bins ─────────────────────
        # Totani used 10°×10° cell-level Poisson MCMC (not pixel NNLS).
        # MCMC properly constrains iso; NNLS at pixel level sets iso→0 (degenerate).
        base_templates_all = [np.asarray(templates_counts[lab], float) for lab in labels]

        if bool(args.use_mcmc):
            print(f"\n[bubbles] Running cell-level Poisson MCMC for all {nE} energy bins...")
            print(f"          nwalkers={args.mcmc_nwalkers}, nsteps={args.mcmc_nsteps}, burn={args.mcmc_burn}, thin={args.mcmc_thin}")
            print("          fit region: full ROI including the Galactic disk (Fig.2/3-style fit)")
            print(f"          saving Fig.2/3 MCMC summaries to {args.mcmc_results_dir}")
            mcmc_results = _mcmc_fit_per_energy(
                counts=counts,
                templates_counts=templates_counts,
                labels=labels,
                expo=expo,
                omega=omega,
                dE_mev=dE_mev,
                Ectr_mev=Ectr_mev,
                fit2d=fit2d,
                lon2d=lon2d,
                lat2d=lat2d,
                roi_lon=float(args.roi_lon),
                roi_lat=float(args.roi_lat),
                cell_deg=float(args.mcmc_cell_deg),
                nwalkers=int(args.mcmc_nwalkers),
                nsteps=int(args.mcmc_nsteps),
                burn=int(args.mcmc_burn),
                thin=int(args.mcmc_thin),
                iso_target_e2=float(iso_target_E2),
                negative_keys=list(args.mcmc_negative_keys),
                require_autocorr=bool(args.require_autocorr),
                early_stop=bool(args.early_stop),
                autocorr_target=float(args.autocorr_target),
                autocorr_check_every=int(args.autocorr_check_every),
                autocorr_min_steps=int(args.autocorr_min_steps),
                mcmc_results_dir=str(args.mcmc_results_dir),
                save_chain=bool(args.save_chain),
                save_logprob=bool(args.save_logprob),
                thin_save=int(args.thin_save),
                progress=True,
                verbosity=1,
            )
            coeffs_all = np.asarray(mcmc_results["f_ml"], float)
            fit_method = "MCMC"
        else:
            print("\n[bubbles] Fitting background with pixel-level NNLS (full ROI)...")
            nnls_mask2d = np.asarray(fit2d, bool).copy()
            mask3d_full = np.broadcast_to(nnls_mask2d[None, :, :], (nE, ny, nx)).copy()
            coeffs_all = nnls_fit_per_energy(
                counts=counts,
                templates=base_templates_all,
                mask3d=mask3d_full,
                ridge=0.0,
            )
            fit_method = "NNLS"

        model_all = build_model_counts(coeffs=coeffs_all, templates=base_templates_all)

        # ── mfit_totani: template pixel region (full ROI minus disk, with src mask) ──
        disk_cut = np.abs(lat2d) >= float(args.disk_cut_deg)
        totani_region = roi2d & disk_cut & src_keep_2d
        mfit_totani = np.asarray(totani_region, bool)

        from scipy.ndimage import gaussian_filter as _gaussian_filter

        sigma_pix = float(args.resid_smooth_deg) / float(args.binsz)

        # ── k2 diagnostic: residual split check ─────────────────────────────────
        coeffs_k2 = coeffs_all[k2]
        base_templates_k2 = [t[k2] for t in base_templates_all]
        model_k2 = model_all[k2]
        resid2d_totani_k2 = np.where(mfit_totani, counts[k2] - model_k2, 0.0)

        fb_flat_idx = labels.index("fb_flat") if "fb_flat" in labels else None
        if fb_flat_idx is not None:
            fb_flat_contrib_k2 = coeffs_k2[fb_flat_idx] * base_templates_k2[fb_flat_idx]
            bubble2d_k2 = np.where(mfit_totani, resid2d_totani_k2 + fb_flat_contrib_k2, 0.0)
        else:
            bubble2d_k2 = resid2d_totani_k2

        denom_k2 = expo[k2] * omega * dE_mev[k2]
        denom_k2_safe = np.where(denom_k2 > 0, denom_k2, 1.0)
        flux2d_k2 = np.where(mfit_totani, bubble2d_k2 / denom_k2_safe, 0.0)
        flux2d_smoothed_k2 = _gaussian_filter(flux2d_k2, sigma=sigma_pix, mode="constant", cval=0.0)
        resid2d_smoothed_k2 = flux2d_smoothed_k2 * denom_k2

        pos_sel = mfit_totani & np.isfinite(resid2d_smoothed_k2) & (resid2d_smoothed_k2 > 0)
        neg_sel = mfit_totani & np.isfinite(resid2d_smoothed_k2) & (resid2d_smoothed_k2 < 0)
        unassigned = mfit_totani & ~(pos_sel | neg_sel)
        pos_sel |= unassigned & np.isfinite(resid2d_totani_k2) & (resid2d_totani_k2 >= 0)
        neg_sel |= unassigned & np.isfinite(resid2d_totani_k2) & (resid2d_totani_k2 < 0)

        split_mode = str(args.bubbles_split_mode).lower().strip()
        if split_mode == "totani":
            pos_shape = np.where(pos_sel, np.clip(resid2d_smoothed_k2, 0.0, None), 0.0)
            neg_shape = np.where(neg_sel, np.clip(resid2d_smoothed_k2, None, 0.0), 0.0)
        elif split_mode == "closure":
            # Use the full bubble image (residual + fb_flat contribution) so that
            # sum(pos_shape + neg_shape) = bubble2d_k2, guaranteeing exact closure.
            pos_shape = np.where(pos_sel, np.clip(bubble2d_k2, 0.0, None), 0.0)
            neg_shape = np.where(neg_sel, np.clip(bubble2d_k2, None, 0.0), 0.0)
        else:
            raise ValueError(f"Unknown --bubbles-split-mode={args.bubbles_split_mode}")

        pos_total = float(pos_shape[mfit_totani].sum())
        neg_total = float(neg_shape[mfit_totani].sum())

        final_model_k2 = model_k2 + pos_shape + neg_shape
        closure_resid = counts[k2] - final_model_k2

        print(f"\n=== Residual split check at k={k2} (E={Ectr_mev[k2]/1000:.2f} GeV), mode='{split_mode}', fit={fit_method} ===")
        print(f"  {fit_method} coefficients: {dict(zip(labels, coeffs_k2))}")
        print(f"  data sum:             {float(counts[k2][mfit_totani].sum()):.6e}")
        print(f"  bg model sum:         {float(model_k2[mfit_totani].sum()):.6e}")
        print(f"  pos_shape sum:        {pos_total:.6e}")
        print(f"  neg_shape sum:        {neg_total:.6e}")
        if split_mode == "closure":
            print(f"  closure residual sum: {float(closure_resid[mfit_totani].sum()):.6e}  <-- must be ~0")
            print(f"  max |closure resid|:  {float(np.abs(closure_resid[mfit_totani]).max()):.6e}  <-- must be ~0")
        else:
            print(f"  closure residual sum: {float(closure_resid[mfit_totani].sum()):.6e}")
            print(f"  max |closure resid|:  {float(np.abs(closure_resid[mfit_totani]).max()):.6e}")

        # Overwrite saved masks so they correspond to the actual pos/neg selection.
        pos_out = os.path.join(args.out_dir, "fb_pos_mask.fits")
        neg_out = os.path.join(args.out_dir, "fb_neg_mask.fits")
        _save_mask_fits(pos_out, pos_sel.astype(np.uint8), hdr)
        _save_mask_fits(neg_out, neg_sel.astype(np.uint8), hdr)
        print("[out]", pos_out)
        print("[out]", neg_out)

        nnls_mask2d = np.asarray(fit2d, bool).copy()

        # ── Propagate to all energy bins ──────────────────────────────────────────
        print("\n[bubbles] Building fb_pos/fb_neg templates for all energy bins...")

        mu_pos = np.zeros((nE, ny, nx), dtype=float)
        mu_neg = np.zeros((nE, ny, nx), dtype=float)

        fb_flat_idx_all = labels.index("fb_flat") if "fb_flat" in labels else None

        if split_mode == "totani":
            # Totani method: spatial morphology (pos/neg shape) is FIXED from k2 (4.3 GeV).
            # The smoothed flux map at k2 defines which pixels are pos/neg and their relative
            # weights. At each other energy bin k, the same shape is used but its amplitude
            # is scaled to match the total bubble counts (residual + fb_flat) at that bin.
            # This prevents the bubble templates from absorbing the spectral structure of
            # any other component (e.g. the 20 GeV halo excess) at other energies.
            #
            # flux2d_smoothed_k2 is already computed above (units: flux, i.e. ph/cm2/s/sr/MeV)
            # pos_shape / neg_shape are the k2 smoothed bubble counts maps (already computed).
            # For each bin k: mu_pos[k] = pos_shape_normalised * total_pos_counts_k
            #                 mu_neg[k] = neg_shape_normalised * total_neg_counts_k
            # where total_pos/neg_counts_k = sum of bubble image counts in pos/neg region at k.

            # Precompute total pos/neg counts at k2 for normalisation
            pos_total_k2 = float(pos_shape[mfit_totani].sum())
            neg_total_k2 = float(neg_shape[mfit_totani].sum())  # negative number

            for k in range(nE):
                resid_k = np.where(mfit_totani, counts[k] - model_all[k], 0.0)
                if fb_flat_idx_all is not None:
                    fb_flat_k = coeffs_all[k, fb_flat_idx_all] * base_templates_all[fb_flat_idx_all][k]
                    bubble_k = np.where(mfit_totani, resid_k + fb_flat_k, 0.0)
                else:
                    bubble_k = resid_k

                # Total bubble counts in each morphological region at this energy bin
                pos_counts_k = float(bubble_k[pos_shape > 0].sum())
                neg_counts_k = float(bubble_k[neg_shape < 0].sum())

                # Scale the fixed k2 shape to match the amplitude at bin k
                if pos_total_k2 > 0:
                    mu_pos[k] = pos_shape * (pos_counts_k / pos_total_k2)
                if neg_total_k2 < 0:
                    mu_neg[k] = neg_shape * (neg_counts_k / neg_total_k2)

            # At k2, restore exact values (no rounding from ratio)
            mu_pos[k2] = pos_shape
            mu_neg[k2] = neg_shape
        else:
            # closure mode: shapes fixed from k2 bubble; amplitudes at other bins scale
            # with the bubble counts (residual + fb_flat contribution) in that bin.
            for k in range(nE):
                resid_k = np.where(mfit_totani, counts[k] - model_all[k], 0.0)
                if fb_flat_idx_all is not None:
                    fb_flat_k = coeffs_all[k, fb_flat_idx_all] * base_templates_all[fb_flat_idx_all][k]
                    bubble_k_cl = np.where(mfit_totani, resid_k + fb_flat_k, 0.0)
                else:
                    bubble_k_cl = resid_k
                pos_counts_k = float(bubble_k_cl[pos_shape > 0].sum())
                neg_counts_k = float(bubble_k_cl[neg_shape < 0].sum())
                if pos_total > 0:
                    mu_pos[k] = pos_shape * (pos_counts_k / pos_total)
                if neg_total < 0:
                    mu_neg[k] = neg_shape * (neg_counts_k / neg_total)
            # At k2, the closure mode gives exact closure by construction (pos_shape + neg_shape = bubble2d_k2).
            mu_pos[k2] = pos_shape
            mu_neg[k2] = neg_shape

        if split_mode == "closure":
            print(f"  mu_pos[k2] sum: {float(mu_pos[k2][mfit_totani].sum()):.6e}  (should equal {pos_total:.6e})")
            print(f"  mu_neg[k2] sum: {float(mu_neg[k2][mfit_totani].sum()):.6e}  (should equal {neg_total:.6e})")
        else:
            print(f"  mu_pos[k2] sum: {float(mu_pos[k2][mfit_totani].sum()):.6e}")
            print(f"  mu_neg[k2] sum: {float(mu_neg[k2][mfit_totani].sum()):.6e}")
        print(f"  mu_pos nonzero bins: {[k for k in range(nE) if mu_pos[k].sum() != 0]}")
        print(f"  mu_neg nonzero bins: {[k for k in range(nE) if mu_neg[k].sum() != 0]}")
    else:
        # No forced residual-closure step requested: keep the Stage 2 masks, but do not build
        # propagated fb_pos/fb_neg count templates from the closure residuals in this branch.
        mu_pos = np.zeros((nE, ny, nx), dtype=float)
        mu_neg = np.zeros((nE, ny, nx), dtype=float)
        coeffs_all = None
        nnls_mask2d = None
        if not build_flat_only:
            print("[WARNING] force-bubbles-close-residual is off — the closure-based fb_pos/fb_neg count templates are not built in this branch")

    # ── rescale-to-data-sum (optional) ──────────────────────────────────────────
    scales_pos = np.ones(nE, dtype=float)
    scales_neg = np.ones(nE, dtype=float)
    scales_flat = np.ones(nE, dtype=float)
    if bool(args.rescale_to_data_sum):
        for k in range(nE):
            y = float(np.nansum(counts[k][fit2d]))

            sp = float(np.nansum(mu_pos[k][fit2d]))
            if np.isfinite(y) and (y > 0) and np.isfinite(sp) and (sp > 0):
                scales_pos[k] = y / sp
                mu_pos[k] *= scales_pos[k]

            sn = float(np.nansum(mu_neg[k][fit2d]))
            if np.isfinite(y) and (y > 0) and np.isfinite(sn) and (sn < 0):
                scales_neg[k] = y / sn
                mu_neg[k] *= scales_neg[k]

            sf = float(np.nansum(mu_flat[k][fit2d]))
            if np.isfinite(y) and (y > 0) and np.isfinite(sf) and (sf > 0):
                scales_flat[k] = y / sf
                mu_flat[k] *= scales_flat[k]

    if args.report_bin is not None:
        k = int(args.report_bin)
        if k < 0 or k >= nE:
            raise ValueError(f"report-bin {k} out of range [0,{nE-1}]")
        _template_scale_report(k=k, data2d=counts[k], mu2d=mu_pos[k], mask2d=fit2d, label="fb_pos")
        _template_scale_report(k=k, data2d=counts[k], mu2d=mu_neg[k], mask2d=fit2d, label="fb_neg")
        _template_scale_report(k=k, data2d=counts[k], mu2d=mu_flat[k], mask2d=fit2d, label="fb_flat")

    hdr_out = hdr.copy()
    hdr_out["ISOE2"] = (float(iso_target_E2), "Reference E^2 dN/dE for f=1 [MeV cm-2 s-1 sr-1]")
    hdr_out["RESCLSUM"] = (bool(args.rescale_to_data_sum), "Rescaled each bin to match data sum in fit mask")

    if not build_flat_only:
        fits.PrimaryHDU(mu_pos.astype(np.float32), header=hdr_out).writeto(
            os.path.join(args.out_dir, "mu_fb_pos_counts.fits"), overwrite=True
        )
        fits.PrimaryHDU(mu_neg.astype(np.float32), header=hdr_out).writeto(
            os.path.join(args.out_dir, "mu_fb_neg_counts.fits"), overwrite=True
        )
    fits.PrimaryHDU(mu_flat.astype(np.float32), header=hdr_out).writeto(
        os.path.join(args.out_dir, "mu_fb_flat_counts.fits"), overwrite=True
    )

    if bool(args.plot_fig1_maps):
        if not bool(args.force_bubbles_close_residual) or (coeffs_all is None):
            print("[WARNING] --plot-fig1-maps requires --force-bubbles-close-residual. Skipping.")
        else:
            templates_by_label = {lab: np.asarray(templates_counts[lab], float) for lab in labels}
            out_png = os.path.join(args.out_plots, "totani_fig1_like.png")
            _plot_totani_fig1_like(
                out_png=out_png,
                counts=counts,
                wcs=wcs,
                expo=expo,
                omega=omega,
                dE_mev=dE_mev,
                Ectr_mev=Ectr_mev,
                roi2d=roi2d,
                lat2d=lat2d,
                flat_mask2d=flat_mask,
                coeffs_all=np.asarray(coeffs_all, float),
                templates_by_label=templates_by_label,
                labels=labels,
                binsz_deg=float(args.binsz),
                disk_cut_deg=float(args.fig1_disk_cut_deg),
                target_mev=list(args.fig1_target_mev),
                vmax_mev=list(args.fig1_vmax),
                psc_path=str(args.psc) if args.psc else None,
            )
            print("[out]", out_png)

    if bool(args.plot_fig2_3):
        if not bool(args.force_bubbles_close_residual) or (coeffs_all is None):
            print("[WARNING] --plot-fig2-3 requires --force-bubbles-close-residual. Skipping.")
        else:
            base_mask = np.asarray(nnls_mask2d if nnls_mask2d is not None else fit2d, bool)
            outdir = os.path.join(args.out_plots, "plots_fig2_3")
            _fit_method = "MCMC" if bool(args.use_mcmc) else "NNLS"
            _plot_fig2_fig3_like_from_nnls(
                outdir=outdir,
                fig_label="fig2",
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                expo=expo,
                omega=omega,
                lat2d=lat2d,
                base_mask2d=base_mask,
                disk_cut_deg_for_plot=float(args.fig23_disk_cut_deg),
                coeffs_all=np.asarray(coeffs_all, float),
                templates_counts={lab: np.asarray(templates_counts[lab], float) for lab in labels},
                labels=labels,
                exclude_disk_in_plot=False,
                fit_method=_fit_method,
            )
            _plot_fig2_fig3_like_from_nnls(
                outdir=outdir,
                fig_label="fig3",
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                expo=expo,
                omega=omega,
                lat2d=lat2d,
                base_mask2d=base_mask,
                disk_cut_deg_for_plot=float(args.fig23_disk_cut_deg),
                coeffs_all=np.asarray(coeffs_all, float),
                templates_counts={lab: np.asarray(templates_counts[lab], float) for lab in labels},
                labels=labels,
                exclude_disk_in_plot=True,
                fit_method=_fit_method,
            )
            print("[out]", outdir)

    np.savez(
        os.path.join(args.out_dir, "fb_flat_rescale_factors.npz"),
        scales_pos=scales_pos,
        scales_neg=scales_neg,
        scales_flat=scales_flat,
    )


    # Diagnostics (skip in flat-only mode since no iteration)
    if not build_flat_only and len(hist) > 0:
        frac_pos = [r.frac_change_pos for r in hist]
        frac_neg = [r.frac_change_neg for r in hist]

        fig = plt.figure(figsize=(6.8, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(hist)), frac_pos, marker="o", label="pos")
        ax.plot(np.arange(len(hist)), frac_neg, marker="o", label="neg")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("fractional pixel change")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_plots, "bubbles_mask_evolution.png"), dpi=200)
        plt.close(fig)

        for i, r in enumerate(hist):
            _plot_smoothed(
                os.path.join(args.out_plots, f"bubbles_smoothed_iter{i:02d}.png"),
                r.smoothed,
                r.pos_mask,
                r.neg_mask,
                title=(
                    f"Bubbles smoothed iter={i}  pos_thr={r.pos_thresh_used:.3e}  neg_thr={r.neg_thresh_used:.3e}"
                ),
            )

    # Final easy-to-interpret plots: masks only + lon/lat overlay
    if not build_flat_only:
        _plot_mask_only(
            os.path.join(args.out_plots, "bubbles_masks_final.png"),
            pos_mask,
            neg_mask,
            title="Fermi Bubbles masks (final): pos=gold / neg=cyan",
        )
    _plot_mask_only(
        os.path.join(args.out_plots, "bubbles_flat_mask_final.png"),
        flat_mask,
        np.zeros_like(flat_mask, dtype=bool),
        title="Fermi Bubbles flat mask (final)",
    )
    if not build_flat_only and len(hist) > 0:
        rfin = hist[-1]
        _plot_lonlat_overlay(
            os.path.join(args.out_plots, "bubbles_lonlat_overlay_final.png"),
            lon2d,
            lat2d,
            rfin.smoothed,
            pos_mask,
            neg_mask,
            roi2d=roi2d,
            title="Bubbles (final) overlay on smoothed residual image",
        )
        _plot_lonlat_overlay(
            os.path.join(args.out_plots, "bubbles_lonlat_overlay_flat_final.png"),
            lon2d,
            lat2d,
            rfin.smoothed,
            flat_mask,
            np.zeros_like(flat_mask, dtype=bool),
            roi2d=roi2d,
            title="Bubbles (final) flat mask overlay on smoothed residual image",
        )

    # Print summary
    ov = int(np.sum(pos_mask & neg_mask))
    if ov != 0:
        raise SystemExit(f"pos/neg overlap detected after resolve: {ov} pixels")

    print("[summary] pos pixels:", int(np.sum(pos_mask)))
    print("[summary] neg pixels:", int(np.sum(neg_mask)))
    print("[summary] overlap pixels:", ov)


if __name__ == "__main__":
    raise SystemExit(main())
