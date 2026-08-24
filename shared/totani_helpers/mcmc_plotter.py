"""
MCMC-specific utilities for Figures 2-3 spectral analysis.

This module provides specialized functions for processing MCMC results and
generating energy spectrum plots that reproduce Figures 2 and 3 from Totani (2025).

Key Functions
-------------
_build_counts_cubes_from_coeffs : Reconstruct component counts from MCMC coefficients
compute_E2_dnde_from_mcmc : Convert MCMC coefficients to E² dN/dE spectra
make_plots_from_mcmc : Generate publication-quality spectral plots

The workflow:
1. Load MCMC coefficients from output files
2. Reconstruct counts cubes for each component
3. Convert to physical flux units (E² dN/dE)
4. Plot with Totani (2025) styling

Notes
-----
This module assumes MCMC output files contain coefficient tables with
standard component names (gas, ics, ps, iso, nfw, loopI, fb_pos, fb_neg).
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fit_utils import E2_from_pred_counts_maps, build_fit_mask3d, load_mu_templates_from_fits
from .mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, pick_coeff, save_coeff_table_txt
from .plotting import plot_E2_dnde_multi_totani
from .totani_io import (
    load_mask_any_shape,
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)


def _plot_component_key(label: str) -> str:
    """Map implementation-specific template labels to stable plot component keys."""
    key = str(label)
    if key.startswith("nfw_"):
        return "nfw"
    if key.endswith("_psf"):
        return key[: -len("_psf")]
    return key


def _build_counts_cubes_from_coeffs(*, templates_counts: Dict[str, np.ndarray], coeffs_by_label: Dict[str, np.ndarray]):
    names = list(templates_counts.keys())
    first = templates_counts[names[0]]
    nE, ny, nx = first.shape

    comp_counts: Dict[str, np.ndarray] = {}
    model_total = np.zeros((nE, ny, nx), dtype=float)

    for name in names:
        T = np.asarray(templates_counts[name], float)
        a = pick_coeff(coeffs_by_label=coeffs_by_label, template_key=name)
        a = np.asarray(a, float).reshape(-1)
        if a.shape[0] != nE:
            raise RuntimeError(f"Coeff for '{name}' has length {a.shape[0]} but nE={nE}")

        cube = a[:, None, None] * T
        comp_counts[name] = cube
        model_total += cube

    return comp_counts, model_total


def _save_curves_txt(*, out_txt: str, Ectr_mev: np.ndarray, curves: Sequence[Tuple[str, np.ndarray]]):
    Egev = np.asarray(Ectr_mev, float) / 1000.0
    nE = int(Egev.shape[0])

    cols = [Egev]
    header_cols = ["E_GeV"]
    for lab, *vals in curves:
        if len(vals) < 1:
            raise ValueError(f"Curve for '{lab}' must include y values")
        y = np.asarray(vals[0], float).reshape(-1)
        if y.shape[0] != nE:
            raise ValueError(f"Curve length mismatch for {out_txt}: got {y.shape[0]} expected {nE}")

        lab = str(lab)
        cols.append(y)
        header_cols.append(lab)

        if len(vals) >= 2 and vals[1] is not None:
            yerr = np.asarray(vals[1], float)
            if yerr.shape == y.shape:
                err_lo = yerr.reshape(-1)
                err_hi = yerr.reshape(-1)
            elif (yerr.ndim == 2) and (yerr.shape[0] == 2) and (yerr.shape[1] == nE):
                err_lo = yerr[0]
                err_hi = yerr[1]
            elif (yerr.ndim == 2) and (yerr.shape[1] == 2) and (yerr.shape[0] == nE):
                err_lo = yerr[:, 0]
                err_hi = yerr[:, 1]
            else:
                raise ValueError(f"Error-bar shape mismatch for '{lab}' in {out_txt}: got {yerr.shape}")

            cols.extend([err_lo, err_hi])
            header_cols.extend([f"{lab}_err_lo", f"{lab}_err_hi"])

    arr = np.column_stack(cols)
    header = " ".join(header_cols)
    np.savetxt(out_txt, arr, header=header)


def plot_E2_dnde_multi_diagnostic(Ectr_mev, curves, *, out_png=None, title=None):
    import matplotlib.pyplot as plt

    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0

    plt.figure(figsize=(8, 6))
    for item in curves:
        lab, y_in = item[0], item[1]
        yerr = item[2] if len(item) >= 3 else None
        y = np.asarray(y_in, float)
        m = np.isfinite(Ectr_gev) & np.isfinite(y) & (Ectr_gev > 0)
        if not np.any(m):
            continue
        if yerr is None:
            plt.plot(Ectr_gev[m], y[m], marker="o", label=str(lab))
        else:
            yerr = np.asarray(yerr, float)
            if yerr.shape == y.shape:
                yerr = yerr[m]
            elif (yerr.ndim == 2) and (yerr.shape[0] == 2) and (yerr.shape[1] == y.shape[0]):
                yerr = yerr[:, m]
            elif (yerr.ndim == 2) and (yerr.shape[1] == 2) and (yerr.shape[0] == y.shape[0]):
                yerr = yerr[m, :].T
            plt.errorbar(Ectr_gev[m], y[m], yerr=yerr, marker="o", capsize=2, label=str(lab))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"$E^2 \,\langle \mathrm{d}N/\mathrm{d}E \rangle$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    if title is not None:
        plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
        plt.close()
    else:
        plt.show()


def _plot_multi(Ectr_mev, curves, *, out_png=None, title=None, plot_style: str = "diagnostic"):
    if plot_style == "totani":
        plot_E2_dnde_multi_totani(Ectr_mev, curves, out_png=out_png, title=title)
    else:
        plot_E2_dnde_multi_diagnostic(Ectr_mev, curves, out_png=out_png, title=title)


def _asymmetric_yerr_from_bounds(y: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    y1 = np.asarray(y1, float)
    y2 = np.asarray(y2, float)

    lo_ref = np.minimum(y1, y2)
    hi_ref = np.maximum(y1, y2)
    err_lo = np.clip(y - lo_ref, 0.0, None)
    err_hi = np.clip(hi_ref - y, 0.0, None)

    err = np.vstack([err_lo, err_hi])
    err[~np.isfinite(err)] = 0.0
    return err


def make_plots_from_mcmc(
    *,
    fig: str,
    counts_path: str,
    expo_path: str,
    templates_dir: str,
    mcmc_dir: str,
    outdir: str,
    mcmc_stat: str = "f_ml",
    plot_style: str = "diagnostic",
    ext_mask_path: Optional[str] = None,
    roi_lon: float = 60.0,
    roi_lat: float = 60.0,
    disk_cut: float = 10.0,
    binsz: float = 0.125,
    labels: Optional[Sequence[str]] = None,
    template_labels: Optional[Sequence[str]] = None,
    exclude_disk_in_plot: bool = False,
):
    os.makedirs(outdir, exist_ok=True)

    counts, hdr, _Emin, _Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(counts_path)
    nE, ny, nx = counts.shape

    expo_raw, E_expo_mev = read_exposure(expo_path)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != counts.shape:
        raise ValueError(f"Exposure shape {expo.shape} != counts shape {counts.shape}")

    from astropy.wcs import WCS

    wcs = WCS(hdr).celestial
    omega = pixel_solid_angle_map(wcs, ny, nx, binsz)
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon) <= float(roi_lon)) & (np.abs(lat) <= float(roi_lat))

    if ext_mask_path is None:
        ext_mask3d = np.ones((nE, ny, nx), dtype=bool)
    else:
        ext_mask3d = load_mask_any_shape(ext_mask_path, counts.shape)

    fit_mask3d = build_fit_mask3d(roi2d=roi2d, srcmask3d=ext_mask3d, counts=counts, expo=expo)

    if labels is None:
        labels = [
            "gas",
            "iso",
            "ps",
            "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
            "loopA",
            "loopB",
            "ics",
            "fb_flat",
        ]
    labels = [str(x) for x in labels]
    if template_labels is None:
        template_labels = list(labels)
    else:
        template_labels = [str(x) for x in template_labels]
    if len(template_labels) != len(labels):
        raise ValueError(
            f"template_labels length {len(template_labels)} != labels length {len(labels)}"
        )

    mu_list, _headers = load_mu_templates_from_fits(
        template_dir=templates_dir,
        labels=list(template_labels),
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )

    templates_counts: Dict[str, np.ndarray] = {}
    for lab, mu in zip(labels, mu_list):
        key = _plot_component_key(str(lab))
        templates_counts[key] = np.asarray(mu, float)

    tab = load_mcmc_coeffs_by_label(mcmc_dir=mcmc_dir, stat=mcmc_stat, nE=nE)
    coeffs_by_label = tab.coeffs_by_label

    coeffs_p16 = None
    coeffs_p84 = None
    try:
        coeffs_p16 = load_mcmc_coeffs_by_label(mcmc_dir=mcmc_dir, stat="f_p16", nE=nE).coeffs_by_label
        coeffs_p84 = load_mcmc_coeffs_by_label(mcmc_dir=mcmc_dir, stat="f_p84", nE=nE).coeffs_by_label
    except Exception:
        coeffs_p16 = None
        coeffs_p84 = None

    # Save coefficients used for plotting, mapped to template keys (and including loopI if available).
    coeffs_plot = dict(coeffs_by_label)
    if ("loopA" in coeffs_plot) and ("loopB" in coeffs_plot):
        coeffs_plot = combine_loopI(coeffs_by_label=coeffs_plot, out_key="loopI", drop_inputs=False)

    keys_for_coeff_dump: List[str] = []
    for k in templates_counts.keys():
        if k in ("loopA", "loopB") and ("loopI" in coeffs_plot):
            continue
        keys_for_coeff_dump.append(str(k))
    if "loopI" in coeffs_plot and "loopI" not in keys_for_coeff_dump:
        keys_for_coeff_dump.append("loopI")

    # x-axis is energy-bin index, since Ectr values live in counts FITS.
    xk = np.arange(nE, dtype=int)
    save_coeff_table_txt(
        out_txt=os.path.join(outdir, f"fit_coefficients_mcmc_{fig}_{mcmc_stat}.txt"),
        x=xk,
        coeffs_by_label=coeffs_plot,
        keys=keys_for_coeff_dump,
        x_label="k",
    )
    if nE > 0:
        k0_vals = {k: float(np.asarray(coeffs_plot[k]).reshape(-1)[0]) for k in keys_for_coeff_dump if k in coeffs_plot}
        print(f"[mcmc coeffs] k=0: {k0_vals}")

    comp_counts_dict, model_counts_total = _build_counts_cubes_from_coeffs(
        templates_counts=templates_counts,
        coeffs_by_label=coeffs_by_label,
    )

    comp_counts_lo = None
    model_counts_total_lo = None
    comp_counts_hi = None
    model_counts_total_hi = None
    if (coeffs_p16 is not None) and (coeffs_p84 is not None):
        comp_counts_lo, model_counts_total_lo = _build_counts_cubes_from_coeffs(
            templates_counts=templates_counts,
            coeffs_by_label=coeffs_p16,
        )
        comp_counts_hi, model_counts_total_hi = _build_counts_cubes_from_coeffs(
            templates_counts=templates_counts,
            coeffs_by_label=coeffs_p84,
        )

    # Combine Loop I for plotting
    if ("loopA" in comp_counts_dict) and ("loopB" in comp_counts_dict):
        comp_counts_dict = dict(comp_counts_dict)
        comp_counts_dict["loopI"] = np.asarray(comp_counts_dict["loopA"], float) + np.asarray(comp_counts_dict["loopB"], float)
        del comp_counts_dict["loopA"]
        del comp_counts_dict["loopB"]
        bkg_names = [k for k in templates_counts.keys() if k not in ("loopA", "loopB")] + ["loopI"]
    else:
        bkg_names = list(templates_counts.keys())

    if (comp_counts_lo is not None) and ("loopA" in comp_counts_lo) and ("loopB" in comp_counts_lo):
        comp_counts_lo = dict(comp_counts_lo)
        comp_counts_lo["loopI"] = np.asarray(comp_counts_lo["loopA"], float) + np.asarray(comp_counts_lo["loopB"], float)
        del comp_counts_lo["loopA"]
        del comp_counts_lo["loopB"]
    if (comp_counts_hi is not None) and ("loopA" in comp_counts_hi) and ("loopB" in comp_counts_hi):
        comp_counts_hi = dict(comp_counts_hi)
        comp_counts_hi["loopI"] = np.asarray(comp_counts_hi["loopA"], float) + np.asarray(comp_counts_hi["loopB"], float)
        del comp_counts_hi["loopA"]
        del comp_counts_hi["loopB"]

    # Fig3 historically excluded the disk in the *plot* (not necessarily in the fit).
    # Allow overriding that behavior via exclude_disk_in_plot.
    if bool(exclude_disk_in_plot) or str(fig) == "fig3":
        plot_mask = fit_mask3d & (np.abs(lat) >= float(disk_cut))[None, :, :]
    else:
        plot_mask = fit_mask3d

    curves = []
    for name in bkg_names:
        pred_counts = np.asarray(comp_counts_dict[name], float)
        # fb_neg template contains negative counts; reverse sign for plotting
        # (matches Totani Fig 4 caption: "sign of negative template is inverted so flux becomes positive")
        if str(name).lower() in ("fb_neg", "fb_negative"):
            pred_counts = -pred_counts
        E2 = E2_from_pred_counts_maps(
            pred_counts_map=pred_counts,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=plot_mask,
        )
        if (comp_counts_lo is not None) and (comp_counts_hi is not None):
            pred_counts_lo = np.asarray(comp_counts_lo[name], float)
            pred_counts_hi = np.asarray(comp_counts_hi[name], float)
            if str(name).lower() in ("fb_neg", "fb_negative"):
                pred_counts_lo = -pred_counts_lo
                pred_counts_hi = -pred_counts_hi
            E2_lo = E2_from_pred_counts_maps(
                pred_counts_map=pred_counts_lo,
                expo=expo,
                omega=omega,
                dE_mev=dE_mev,
                Ectr_mev=Ectr_mev,
                mask2d=plot_mask,
            )
            E2_hi = E2_from_pred_counts_maps(
                pred_counts_map=pred_counts_hi,
                expo=expo,
                omega=omega,
                dE_mev=dE_mev,
                Ectr_mev=Ectr_mev,
                mask2d=plot_mask,
            )
            curves.append((name, E2, _asymmetric_yerr_from_bounds(E2, E2_lo, E2_hi)))
        else:
            curves.append((name, E2))
    E2_tot = E2_from_pred_counts_maps(
        pred_counts_map=np.asarray(model_counts_total, float),
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask2d=plot_mask,
    )
    if (model_counts_total_lo is not None) and (model_counts_total_hi is not None):
        E2_tot_lo = E2_from_pred_counts_maps(
            pred_counts_map=np.asarray(model_counts_total_lo, float),
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=plot_mask,
        )
        E2_tot_hi = E2_from_pred_counts_maps(
            pred_counts_map=np.asarray(model_counts_total_hi, float),
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=plot_mask,
        )
        curves.append(("total", E2_tot, _asymmetric_yerr_from_bounds(E2_tot, E2_tot_lo, E2_tot_hi)))
    else:
        curves.append(("total", E2_tot))

    _save_curves_txt(
        out_txt=os.path.join(outdir, f"mcmc_{fig}_{mcmc_stat}.txt"),
        Ectr_mev=Ectr_mev,
        curves=curves,
    )

    if bool(exclude_disk_in_plot) or str(fig) == "fig3":
        title = f"MCMC background components ({mcmc_stat}), |b|>={float(disk_cut):g} deg"
    else:
        title = f"MCMC background components ({mcmc_stat})"

    _plot_multi(
        Ectr_mev,
        curves,
        out_png=os.path.join(outdir, f"mcmc_{fig}_{mcmc_stat}.png"),
        title=title,
        plot_style=plot_style,
    )

    return {
        "Ectr_mev": Ectr_mev,
        "dE_mev": dE_mev,
        "bkg_names": bkg_names,
        "curves": curves,
        "coeffs_by_label": coeffs_plot,
    }
