#!/usr/bin/env python3
"""
Config-file wrapper for run_global_fit.py.

Usage:
    python run_global_from_config.py --config configs/gce_global.json
    python run_global_from_config.py --config configs/gce_global.json --energy-bin 20
    python run_global_from_config.py --config configs/gce_global.json --dry-run

Config structure mirrors the cellwise run_mcmc_from_config.py convention
(base_config merging supported).  Only the fields relevant to the global
pixel-level fit are forwarded; MCMC-specific keys (nwalkers, nsteps, etc.)
are silently ignored.

Supported config keys
---------------------
base_config          path to a base JSON file to merge from
name                 human label (unused at runtime)
outdir / global_dir  output directory for global_fit_k??.npz files
counts               path to counts_ccube FITS
expo                 path to exposure FITS
templates_dir        directory containing mu_{label}_counts.fits
ext_mask             extended-source mask FITS ('' to skip)
roi_lon              ROI half-width in longitude (default 7.5 deg)
roi_lat              ROI half-width in latitude  (default 7.5 deg)
include_disk         bool; if false, --exclude-disk is added
disk_cut             disk exclusion half-width (deg, default 2.0)
binsz                pixel size in degrees (default 0.125)
energy_bins          "all" or list of int
    labels               list of component label strings
template_labels         optional FITS template labels; defaults to labels
negative_keys        list of negative-allowed substrings
iso_target_e2        isotropic E²dN/dE target (MeV cm⁻² s⁻¹ sr⁻¹)
poisson_cexp_floor   floor for Cexp in log (default 1e-30)
error_method         fisher|lbfgsb|none (default fisher)
max_iter             L-BFGS-B max iterations (default 5000)
no_save_maps         bool; if true, skip residual map saving
verbosity            0|1|2
mask_bright_ps       bool; if true, build/use templates_dir/mask_sources_combined.fits
ps_catalog           4FGL point-source catalogue FITS
ps_*                 bright point-source mask options (flux threshold, radius, etc.)
psf_labels           list of substrings/labels to PSF-convolve before fitting
psf_kind/gamma       PSF convolution options
check_global_positivity  bool; run post-fit positivity validation
"""

import argparse
import json
import os
import runpy
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import read_counts_and_ebounds


HERE = os.path.dirname(os.path.abspath(__file__))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(a)
    for k, vb in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(vb, dict):
            out[k] = _deep_merge(out[k], vb)
        else:
            out[k] = vb
    return out


def _abspath_from(base_dir: str, p: Optional[Union[str, os.PathLike]]) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    if not p.strip():
        return ""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _energy_bins_from_cfg(cfg: Dict[str, Any], counts_path: str) -> List[int]:
    bins = cfg.get("energy_bins", "all")
    if isinstance(bins, str):
        if bins.strip().lower() != "all":
            raise SystemExit("energy_bins must be 'all' or a list of integers")
        counts, *_ = read_counts_and_ebounds(counts_path)
        return list(range(int(counts.shape[0])))
    if isinstance(bins, (list, tuple)):
        return [int(x) for x in bins]
    raise SystemExit("energy_bins must be 'all' or a list of integers")


def _add_arg(argv: List[str], flag: str, value) -> None:
    if value is None:
        return
    argv.append(str(flag))
    argv.append(str(value))


def _add_boolopt(argv: List[str], flag: str, value: Optional[bool]) -> None:
    if value is None:
        return
    argv.append(str(flag) if bool(value) else "--no-" + str(flag).lstrip("-"))


def _needs_rebuild(out_path: str, *src_paths: str) -> bool:
    if not os.path.exists(out_path):
        return True
    out_mtime = os.path.getmtime(out_path)
    return any(os.path.exists(src) and os.path.getmtime(src) > out_mtime for src in src_paths)


def _as_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    if key not in cfg:
        return bool(default)
    return bool(cfg.get(key))


def _as_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _find_point_source_catalog(
    *,
    cfg: Dict[str, Any],
    cfg_dir: str,
    templates_dir: str,
) -> str:
    configured = cfg.get("ps_catalog") or cfg.get("psc")
    if configured:
        path = str(_abspath_from(cfg_dir, configured))
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Point-source catalogue not found: {path}")

    candidates = [
        os.path.join(templates_dir, "gll_psc_v35.fit"),
        os.path.abspath(os.path.join(templates_dir, "..", "..", "templates", "gll_psc_v35.fit")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Point-source catalogue not found. Set config key 'ps_catalog'. "
        f"Tried: {candidates}"
    )


def _selected_for_psf(label: str, selectors: Iterable[str]) -> bool:
    lab_l = str(label).lower()
    for sel in selectors:
        sel_l = str(sel).lower()
        if lab_l == sel_l or sel_l in lab_l:
            return True
    return False


def _build_bright_ps_mask(cfg: Dict[str, Any], *, cfg_dir: str) -> None:
    if not _as_bool(cfg, "mask_bright_ps", False):
        return

    import mask_point_sources as mps

    counts = str(_abspath_from(cfg_dir, cfg.get("counts")))
    templates_dir = str(_abspath_from(cfg_dir, cfg.get("templates_dir")))
    ext_mask = str(_abspath_from(cfg_dir, cfg.get("ext_mask") or "")) if cfg.get("ext_mask") else ""
    psc = _find_point_source_catalog(cfg=cfg, cfg_dir=cfg_dir, templates_dir=templates_dir)

    out_name = str(cfg.get("bright_ps_mask") or "mask_sources_combined.fits")
    out_path = out_name if os.path.isabs(out_name) else os.path.join(templates_dir, out_name)
    combine_with = None if os.path.abspath(ext_mask) == os.path.abspath(out_path) else (ext_mask or None)

    srcs = [src for src in (counts, psc, combine_with) if src and os.path.exists(src)]
    if _needs_rebuild(out_path, *srcs) or _as_bool(cfg, "force_rebuild_global_artifacts", False):
        mps.build_combined_mask(
            counts,
            psc,
            combine_with,
            out_path,
            flux_threshold=float(cfg.get("ps_flux_threshold", 3e-9)),
            signif_threshold=_as_float_or_none(cfg.get("ps_signif_threshold")),
            radius_deg=float(cfg.get("ps_radius_deg", 1.0)),
            scale_radius_with_flux=bool(cfg.get("ps_scale_radius_with_flux", False)),
            max_radius_deg=float(cfg.get("ps_max_radius_deg", 2.0)),
            gc_radius_deg=float(cfg.get("ps_gc_radius_deg", 90.0)),
            energy_dependent=bool(cfg.get("ps_energy_dependent", False)),
            psf_radius_factor=float(cfg.get("psf_radius_factor", 2.0)),
            hdu_index=int(cfg.get("ps_hdu_index", 1)),
        )
    else:
        print(f"  [global prep] bright-PS mask up to date: {out_path}")

    cfg["ext_mask"] = out_path
    cfg["bright_ps_mask"] = out_path
    cfg["ps_catalog"] = psc


def _build_psf_templates(cfg: Dict[str, Any], *, cfg_dir: str) -> None:
    selectors = cfg.get("psf_labels", [])
    if selectors is None:
        selectors = []
    if isinstance(selectors, str):
        selectors = [selectors]
    selectors = [str(x) for x in selectors if str(x).strip()]
    if not selectors:
        return

    import psf_utils as psf

    counts = str(_abspath_from(cfg_dir, cfg.get("counts")))
    templates_dir = str(_abspath_from(cfg_dir, cfg.get("templates_dir")))
    labels = [str(x) for x in cfg.get("labels", [])]
    template_labels: List[str] = []
    built: List[str] = []

    for label in labels:
        if label.endswith("_psf") or not _selected_for_psf(label, selectors):
            template_labels.append(label)
            continue

        in_fits = os.path.join(templates_dir, f"mu_{label}_counts.fits")
        out_fits = os.path.join(templates_dir, f"mu_{label}_psf_counts.fits")
        if not os.path.exists(in_fits):
            if os.path.exists(out_fits):
                print(
                    "  [global prep] PSF source template missing; using cached "
                    f"PSF template: {out_fits}"
                )
                template_labels.append(f"{label}_psf")
            else:
                print(f"  [global prep] WARNING: PSF source template missing, keeping original: {in_fits}")
                template_labels.append(label)
            continue

        if _needs_rebuild(out_fits, in_fits, counts) or _as_bool(cfg, "force_rebuild_global_artifacts", False):
            psf.convolve_template_file(
                in_fits,
                counts,
                out_fits,
                kind=str(cfg.get("psf_kind", "king")),
                gamma=float(cfg.get("psf_gamma", 2.5)),
                rmax_factor=float(cfg.get("psf_rmax_factor", 6.0)),
                verbose=bool(cfg.get("psf_verbose", True)),
            )
        else:
            print(f"  [global prep] PSF template up to date: {out_fits}")
        template_labels.append(f"{label}_psf")
        built.append(label)

    if built:
        print(f"  [global prep] PSF-convolved labels: {built}")
    cfg["template_labels"] = template_labels


def _prepare_global_artifacts(cfg: Dict[str, Any], *, cfg_dir: str) -> Dict[str, Any]:
    cfg = dict(cfg)
    _build_bright_ps_mask(cfg, cfg_dir=cfg_dir)
    _build_psf_templates(cfg, cfg_dir=cfg_dir)
    return cfg


def _build_argv(cfg: Dict[str, Any], *, cfg_dir: str) -> List[str]:
    """Translate structured config dict into run_global_fit.py argv."""
    argv: List[str] = []

    # Prefer 'global_dir' key; fall back to 'outdir' or 'mcmc_dir'
    raw_outdir = cfg.get("global_dir") or cfg.get("outdir") or cfg.get("mcmc_dir")
    outdir = _abspath_from(cfg_dir, raw_outdir)
    _add_arg(argv, "--outdir", outdir)

    counts = _abspath_from(cfg_dir, cfg.get("counts"))
    expo = _abspath_from(cfg_dir, cfg.get("expo"))
    templates_dir = _abspath_from(cfg_dir, cfg.get("templates_dir"))
    ext_mask = _abspath_from(cfg_dir, cfg.get("ext_mask"))

    _add_arg(argv, "--counts", counts)
    _add_arg(argv, "--expo", expo)
    _add_arg(argv, "--templates-dir", templates_dir)
    if ext_mask is not None:
        _add_arg(argv, "--ext-mask", ext_mask)

    _add_arg(argv, "--roi-lon", cfg.get("roi_lon"))
    _add_arg(argv, "--roi-lat", cfg.get("roi_lat"))
    _add_arg(argv, "--disk-cut", cfg.get("disk_cut"))
    _add_arg(argv, "--hemisphere", cfg.get("hemisphere"))
    _add_arg(argv, "--lon-sign", cfg.get("lon_sign"))
    _add_arg(argv, "--binsz", cfg.get("binsz"))

    include_disk = cfg.get("include_disk")
    if include_disk is not None and not bool(include_disk):
        argv.append("--exclude-disk")

    labels = cfg.get("labels")
    if labels is not None:
        argv.append("--labels")
        argv.extend([str(x) for x in labels])

    template_labels = cfg.get("template_labels")
    if template_labels is not None:
        argv.append("--template-labels")
        argv.extend([str(x) for x in template_labels])

    negative_keys = cfg.get("negative_keys")
    if negative_keys is not None:
        argv.append("--negative-keys")
        argv.extend([str(x) for x in negative_keys])

    _add_arg(argv, "--iso-target-e2", cfg.get("iso_target_e2"))
    _add_arg(argv, "--poisson-cexp-floor", cfg.get("poisson_cexp_floor"))
    _add_arg(argv, "--error-method", cfg.get("error_method"))

    fix_labels = cfg.get("fix_labels")
    if fix_labels is not None:
        argv.append("--fix-labels")
        argv.extend([str(x) for x in fix_labels])
    _add_arg(argv, "--max-iter", cfg.get("max_iter"))
    _add_arg(argv, "--verbosity", cfg.get("verbosity"))

    if cfg.get("no_save_maps", False):
        argv.append("--no-save-maps")
    check_fit_positivity = cfg.get("check_positivity", cfg.get("check_global_positivity"))
    _add_boolopt(argv, "--check-positivity", check_fit_positivity)

    extra_argv = cfg.get("extra_argv")
    if isinstance(extra_argv, list):
        argv.extend([str(x) for x in extra_argv])

    return argv


def _run_one(argv_tail: List[str]) -> None:
    script_path = os.path.abspath(os.path.join(HERE, "run_global_fit.py"))
    old_argv = sys.argv
    try:
        sys.argv = [script_path] + list(argv_tail)
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run global pixel-level fits from a JSON config file."
    )
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument(
        "--base-config",
        default=None,
        help="Optional base JSON config with shared parameters (overridden by --config).",
    )
    ap.add_argument(
        "--energy-bin",
        type=int,
        default=None,
        help="Override: run only this single energy bin.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved argv but do not execute.",
    )
    args = ap.parse_args(argv)

    cfg_path = os.path.abspath(str(args.config))
    cfg_dir = os.path.dirname(cfg_path)
    run_cfg = _load_json(cfg_path)

    base_cfg: Dict[str, Any] = {}
    base_path = args.base_config or run_cfg.get("base_config")
    if base_path is not None:
        base_abs = _abspath_from(cfg_dir, base_path)
        base_cfg = _load_json(str(base_abs))

    cfg = _deep_merge(base_cfg, run_cfg)
    cfg = _prepare_global_artifacts(cfg, cfg_dir=cfg_dir)

    counts_path = cfg.get("counts")
    if counts_path is None:
        raise SystemExit("Config must set 'counts'.")
    counts_path = str(_abspath_from(cfg_dir, counts_path))

    # Determine which energy bins to process
    if args.energy_bin is not None:
        bins_to_run = [int(args.energy_bin)]
    else:
        bins_to_run = _energy_bins_from_cfg(cfg, counts_path)

    # Build base argv (everything except --energy-bin / --energy-bins)
    base_argv = _build_argv(cfg, cfg_dir=cfg_dir)

    print(f"\nGlobal fit config: {cfg_path}")
    print(f"Processing {len(bins_to_run)} energy bin(s): "
          f"{bins_to_run[:6]}{'...' if len(bins_to_run) > 6 else ''}")

    for k in bins_to_run:
        argv_k = base_argv + ["--energy-bin", str(k)]
        if args.dry_run:
            print("run_global_fit.py", " ".join(argv_k))
        else:
            _run_one(argv_k)

    global_dir = cfg.get("global_dir") or cfg.get("outdir") or cfg.get("mcmc_dir")
    global_dir_abs = str(_abspath_from(cfg_dir, global_dir)) if global_dir else None
    if (
        cfg.get("check_global_positivity", False)
        and not args.dry_run
        and global_dir_abs
    ):
        from check_global_positivity import check_results_dir

        floor = float(cfg.get("positivity_floor", cfg.get("poisson_cexp_floor", 1e-30)))
        print(f"\nChecking global positivity: {global_dir_abs}")
        status = check_results_dir(global_dir_abs, floor=floor)
        if status != 0 and cfg.get("positivity_fail_on_error", False):
            raise SystemExit(status)

    # ---- Run plots (same pattern as run_mcmc_from_config.py) ----
    plot_cfgs = cfg.get("plot_configs")
    if plot_cfgs and not args.dry_run and args.energy_bin is None:
        if not isinstance(plot_cfgs, list):
            raise SystemExit("Config key 'plot_configs' must be a list of paths")

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from figures.make_figures_from_config import main as _plot_main

        # Resolve overrideable data paths from merged config
        def _resolve(key):
            v = cfg.get(key)
            return str(_abspath_from(cfg_dir, v)) if v else None

        for pcfg in plot_cfgs:
            pcfg_abs = str(_abspath_from(cfg_dir, pcfg))
            argv_plot: List[str] = ["--config", pcfg_abs]
            if global_dir_abs:
                argv_plot += ["--mcmc-dir", global_dir_abs]
            if cfg.get("template_labels"):
                argv_plot += ["--template-labels"]
                argv_plot.extend([str(x) for x in cfg["template_labels"]])
            for flag, key in [("--counts", "counts"), ("--expo", "expo"),
                               ("--templates-dir", "templates_dir"),
                               ("--ext-mask", "ext_mask")]:
                v = _resolve(key)
                if v:
                    argv_plot += [flag, v]
            print(f"\nRunning plots: {pcfg_abs}")
            _plot_main(argv_plot)

    elif plot_cfgs and args.dry_run:
        print("\n[dry-run] Would run plot configs:", plot_cfgs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
