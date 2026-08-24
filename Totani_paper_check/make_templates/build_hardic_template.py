#!/usr/bin/env python3
r"""
build_hardic_template.py
=========================
CLI builder for the "hard inverse-Compton" (hardic) injected cosmic-ray
population templates, a spectral-mimic alternative to the NFW-squared halo
tested against Stenhouse, Ghag & Deppisch (arXiv:2607.08552).  See
``hardic_physics.py`` for the physics (Syrovatskii burst/continuous/advected
transport + full Klein-Nishina IC emission).

Follows the same file-writing steps as the GIEM builder (this repo's
established pattern for a non-GALPROP template, `build_giem_template.py`):

    1. Read the counts-cube header + EBOUNDS for the target grid and the
       13 bin-centre energies (`read_counts_and_ebounds`).
    2. Read + resample the exposure onto those bin centres.
    3. Compute the injected-population IC intensity dN/dE on the counts
       WCS grid (this script's job -- there is no external source cube to
       reproject, unlike GIEM/GALPROP; the intensity is computed directly
       on the sky grid via `hardic_physics.project_los`).
    4. Convert to expected counts: mu = dnde * expo * omega * dE
       (identical to build_giem_template.py:106 / build_ics_template.py).
    5. Write ``hardic_<tag>_dnde.fits`` and ``mu_hardic_<tag>_counts.fits``.
    6. PSF-convolve directly via `psf_utils.convolve_template_file`
       (as build_giem_template.py's docstring instructs for GIEM), writing
       ``mu_hardic_<tag>_psf_counts.fits``.  This is a *convenience* pre-
       build; `run_global_from_config.py`'s `_build_psf_templates` will
       also auto-build/refresh it from the `psf_labels` selector in a
       variant config if it is missing or stale, so step 6 is not a hard
       requirement for the fit to run, only for quick inspection.

Morphology grid (lambda quoted at the 0.7 TeV reference, matching the
paper's own convention)
--------------------------------------------------------------------------
    --variant burst_s   burst, lambda(0.7 TeV) ~= 1.5 kpc  (baseline transport)
    --variant burst_m   burst, lambda(0.7 TeV) ~= 3   kpc  (the critical case)
    --variant burst_l   burst, lambda(0.7 TeV) ~= 6   kpc  (tests the extreme)
    --variant cont       continuous injection, 50 Myr, same transport as burst_m
    --variant adv        burst_m + bipolar v_wind=300 km/s displacement (optional)
    --variant cmbonly_<tag>   any of the above, CMB-only ISRF bracket
                              (e.g. cmbonly_burst_m)

burst_s reaches its target at the *baseline* D0 (galdef value); burst_m
and burst_l exceed the loss-limited asymptote at baseline D0/U_eff and so
require an automatic D0 boost -- `hardic_physics.calibrate_burst_age`
computes and reports the required factor (also written to the FITS header
as DBOOST).  This is an explicit, quantifiable and reportable physicality
axis, not a bug: it is expected, and it is exactly what the naive
Thomson-scale sizes for these morphologies anticipate -- reproducing the
observed halo morphology with an injected leptonic population already
requires generous transport.

Usage
-----
    python build_hardic_template.py --variant burst_s
    python build_hardic_template.py --variant burst_m
    python build_hardic_template.py --variant burst_l
    python build_hardic_template.py --variant cont
    python build_hardic_template.py --variant adv
    python build_hardic_template.py --variant cmbonly_burst_m

    # fast/coarse grid for smoke-testing before a full-resolution build
    # (full resolution -- the default -- is a ~200x200 (R,z) grid and
    # matches "minutes of compute per template"; --fast trades accuracy
    # for a <1 minute smoke test):
    python build_hardic_template.py --variant burst_s --fast

    # inspect the calibration (age, D0 boost, U_eff) without writing FITS:
    python build_hardic_template.py --variant burst_l --dry-run

Output (into ``fermi_data/totani/processed/templates/``):
    hardic_<tag>_dnde.fits         dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
    mu_hardic_<tag>_counts.fits    expected counts [counts]
    mu_hardic_<tag>_psf_counts.fits   PSF-forward-folded expected counts

Registering a new template in the fit: add its label (``hardic_<tag>``,
never containing "nfw"/"fb_neg"/"gas"/"ics"/"loopA"/"loopB" as a
substring -- see ``verify_hardic_naming.py`` alongside this script, which
tests exactly this) to a variant config's ``labels`` list and add
``"hardic"`` to that config's ``psf_labels``.  See
``mcmc/write_global_systematics_configs.py``'s
``cr_injection`` variant group and ``mcmc/configs/hardic/*.json`` for the
configs this task added.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (  # noqa: E402
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    write_cube,
)

from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardic_physics as hp  # noqa: E402

DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

BASE_MORPHOLOGIES = {
    "burst_s": dict(kind="burst", lambda_target_kpc=1.5),
    "burst_m": dict(kind="burst", lambda_target_kpc=3.0),
    "burst_l": dict(kind="burst", lambda_target_kpc=6.0),
    "cont": dict(kind="cont", lambda_target_kpc=3.0),   # transport calibrated as burst_m
    "adv": dict(kind="adv", lambda_target_kpc=3.0),      # transport calibrated as burst_m
}


def _parse_variant(variant: str):
    """Returns (tag, base_key, isrf_kind). tag is what goes into the FITS
    label hardic_<tag>; base_key indexes BASE_MORPHOLOGIES; isrf_kind is
    'full' or 'cmb_only'."""
    if variant.startswith("cmbonly_"):
        base_key = variant[len("cmbonly_"):]
        if base_key not in BASE_MORPHOLOGIES:
            raise SystemExit(f"Unknown base morphology '{base_key}' in --variant {variant!r}")
        return variant, base_key, "cmb_only"
    if variant not in BASE_MORPHOLOGIES:
        raise SystemExit(
            f"Unknown --variant {variant!r}. Choose one of "
            f"{list(BASE_MORPHOLOGIES)} or 'cmbonly_<one of those>'."
        )
    return variant, variant, "full"


def _make_ne_of_Ee(kind, Rg, Zg, *, Gamma_inj, Ec_gev, Q0_gev1, transport, b0_gev1_s1,
                    t_s=None, T_myr=None, n_age_nodes=40, v_wind_kpc_per_s=None):
    if kind == "burst":
        def ne_of_Ee(Ee_gev):
            return hp.burst_density(Ee_gev, Rg, Zg, t_s, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev,
                                     Q0_gev1=Q0_gev1, transport=transport, b0_gev1_s1=b0_gev1_s1)
        return ne_of_Ee
    if kind == "cont":
        def ne_of_Ee(Ee_gev):
            return hp.continuous_injection_density(
                Ee_gev, Rg, Zg, T_myr, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev, Q0_gev1=Q0_gev1,
                transport=transport, b0_gev1_s1=b0_gev1_s1, n_age_nodes=n_age_nodes,
            )
        return ne_of_Ee
    if kind == "adv":
        # Crude bipolar-outflow bracket (NOT a solved wind solution): the
        # burst Green's function is evaluated with z -> |z| - v_wind*t
        # (see hardic_physics module docstring).
        Zeff = np.abs(Zg) - v_wind_kpc_per_s * t_s

        def ne_of_Ee(Ee_gev):
            return hp.burst_density(Ee_gev, Rg, Zeff, t_s, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev,
                                     Q0_gev1=Q0_gev1, transport=transport, b0_gev1_s1=b0_gev1_s1)
        return ne_of_Ee
    raise ValueError(f"Unknown morphology kind {kind!r}")


def build(args) -> str:
    t_start = time.time()
    tag, base_key, isrf_kind = _parse_variant(args.variant)
    morph = BASE_MORPHOLOGIES[base_key]
    fields = hp.CMB_ONLY_FIELDS if isrf_kind == "cmb_only" else hp.DEFAULT_FIELDS

    print(f"=== hardic template: variant={args.variant!r}  (tag=hardic_{tag}, "
          f"kind={morph['kind']}, isrf={isrf_kind}) ===")

    # ------------------------------------------------------------------
    # Grid + energies from the counts file (exact pipeline convention)
    # ------------------------------------------------------------------
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)
    from astropy.wcs import WCS
    w_tgt = WCS(hdr).celestial
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)
    lon_deg, lat_deg = lonlat_grids(w_tgt, ny, nx)

    expo, E_exp = read_exposure(args.expo)
    if expo.shape[0] != nE and E_exp is not None:
        expo = resample_exposure_logE(expo, E_exp, Ectr_mev)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")

    print(f"  Sky grid: {nx}x{ny} @ {args.binsz} deg;  {nE} energy bins "
          f"({Ectr_mev[0]/1e3:.2f}-{Ectr_mev[-1]/1e3:.1f} GeV centres)")

    # ------------------------------------------------------------------
    # Physics calibration
    # ------------------------------------------------------------------
    Ec_gev = args.ec_tev * 1.0e3
    Gamma_inj = args.gamma_inj
    Q0_gev1 = hp.q0_from_einj(args.einj_erg, Gamma_inj, Ec_gev)

    transport0 = hp.TransportParams(n_images=args.n_images)
    if args.u_eff_ev is not None:
        U_eff = float(args.u_eff_ev)
    else:
        U_eff = hp.effective_u_eff(E_ref_gev=700.0, fields=hp.DEFAULT_FIELDS)
        # NOTE: U_eff (and hence transport/loss physics) always uses the
        # FULL ISRF, even for the cmbonly_* emission bracket -- see the
        # module docstring's discussion of why the CMB-only bracket must
        # be ISRF-independent in *both* loss rate and emission kernel to
        # be a meaningful sensitivity check. Pass --u-eff-ev to override.
        if isrf_kind == "cmb_only" and args.cmb_only_transport:
            U_eff = hp.effective_u_eff(E_ref_gev=700.0, fields=hp.CMB_ONLY_FIELDS)
    b0 = hp.b0_from_ueff(U_eff)
    print(f"  U_eff = {U_eff:.4f} eV/cm^3  ->  b0 = {b0:.4e} GeV^-1 s^-1  "
          f"(t_cool(0.7 TeV) = {1.0/(b0*700.0)/hp.MYR_S:.4f} Myr)")

    lambda_target = morph["lambda_target_kpc"]
    calib = hp.calibrate_burst_age(lambda_target, transport=transport0, b0_gev1_s1=b0)
    print(f"  Calibration @ 0.7 TeV for lambda_target={lambda_target} kpc: "
          f"t = {calib.t_s/hp.MYR_S:.4f} Myr, D0_scale = {calib.d0_scale:.3f}, "
          f"lambda_achieved = {calib.lambda_achieved_kpc:.3f} kpc "
          f"(lambda_max@this D0 = {calib.lambda_max_kpc:.3f} kpc)")
    if calib.d0_scale > 1.01:
        print(f"  NOTE: target exceeds the baseline loss-limited asymptote "
              f"({calib.lambda_max_kpc/calib.d0_scale**0.5:.3f} kpc @ D0_scale=1); "
              f"D0 boosted by a factor {calib.d0_scale:.2f} to reach it "
              f"(physicality flag -- an enhanced diffusion coefficient is "
              f"required for this morphology).")

    transport = hp.TransportParams(
        D0_cm2_s=transport0.D0_cm2_s * calib.d0_scale,
        R_br_gv=transport0.R_br_gv, delta=transport0.delta,
        z_h_kpc=transport0.z_h_kpc, n_images=transport0.n_images,
    )

    # ------------------------------------------------------------------
    # (R, z) grid and electron-energy grid
    # ------------------------------------------------------------------
    R_grid = np.linspace(1.0e-3, args.r_max_kpc, args.nr)
    z_grid = np.linspace(-args.z_max_kpc, args.z_max_kpc, args.nz)
    Rg, Zg = np.meshgrid(R_grid, z_grid, indexing="ij")

    Ee_max_gev = max(args.ee_max_gev, 30.0 * Ec_gev)
    E_e_grid_gev = np.geomspace(args.ee_min_gev, Ee_max_gev, args.n_ee)

    v_wind_kpc_per_s = (args.v_wind_km_s * 1.0e5) / hp.KPC_CM  # km/s -> kpc/s

    ne_of_Ee = _make_ne_of_Ee(
        morph["kind"], Rg, Zg, Gamma_inj=Gamma_inj, Ec_gev=Ec_gev, Q0_gev1=Q0_gev1,
        transport=transport, b0_gev1_s1=b0, t_s=calib.t_s, T_myr=args.t_cont_myr,
        n_age_nodes=args.n_age_nodes, v_wind_kpc_per_s=v_wind_kpc_per_s,
    )

    print(f"  (R,z) grid: {args.nr}x{args.nz} out to ({args.r_max_kpc},"
          f"{args.z_max_kpc}) kpc;  {args.n_ee} electron-energy nodes "
          f"({args.ee_min_gev:.1f}-{Ee_max_gev:.3g} GeV)")

    q_grid = hp.emissivity_grid(
        Ectr_mev, ne_of_Ee, E_e_grid_gev, R_grid, z_grid, fields=fields, verbose=args.verbose,
    )
    print(f"  Emissivity grid computed: shape {q_grid.shape}, "
          f"finite={np.all(np.isfinite(q_grid))}, min={q_grid.min():.3e}, "
          f"max={q_grid.max():.3e}  ({time.time()-t_start:.1f} s elapsed)")

    if np.any(~np.isfinite(q_grid)) or np.any(q_grid < 0):
        raise RuntimeError("Non-finite or negative emissivity encountered; aborting.")

    if args.dry_run:
        print("  --dry-run: stopping before LOS projection / FITS write.")
        return ""

    # ------------------------------------------------------------------
    # Line-of-sight projection -> dN/dE(l, b) per energy bin
    # ------------------------------------------------------------------
    dnde = np.empty((nE, ny, nx), dtype=float)
    for k in range(nE):
        dnde[k] = hp.project_los(
            q_grid[k], R_grid, z_grid, lon_deg, lat_deg,
            s_max_kpc=args.s_max_kpc, n_steps=args.n_los_steps,
        )
        if args.verbose:
            print(f"    LOS projection bin {k:02d} (E={Ectr_mev[k]/1e3:.2f} GeV): "
                  f"max={dnde[k].max():.3e} ph/cm2/s/sr/MeV")

    if np.any(~np.isfinite(dnde)) or np.any(dnde < 0):
        raise RuntimeError("Non-finite or negative dN/dE after LOS projection; aborting.")

    print(f"  dN/dE cube: finite={np.all(np.isfinite(dnde))}, "
          f"min={dnde.min():.3e}, max={dnde.max():.3e} ph cm^-2 s^-1 sr^-1 MeV^-1")

    # ------------------------------------------------------------------
    # mu = dnde * expo * omega * dE  (build_giem_template.py:106)
    # ------------------------------------------------------------------
    mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    os.makedirs(args.outdir, exist_ok=True)
    out_dnde = os.path.join(args.outdir, f"hardic_{tag}_dnde.fits")
    out_mu = os.path.join(args.outdir, f"mu_hardic_{tag}_counts.fits")

    hdr_out = hdr.copy()
    hdr_out["HCVARIAN"] = (str(args.variant), "hardic template variant")
    hdr_out["HCKIND"] = (str(morph["kind"]), "burst | cont | adv")
    hdr_out["HCISRF"] = (str(isrf_kind), "full | cmb_only")
    hdr_out["HCLAMTGT"] = (float(lambda_target), "target lambda(0.7 TeV) [kpc]")
    hdr_out["HCLAMACH"] = (float(calib.lambda_achieved_kpc), "achieved lambda(0.7 TeV) [kpc]")
    hdr_out["HCTAGE"] = (float(calib.t_s / hp.MYR_S), "burst age [Myr]")
    hdr_out["DBOOST"] = (float(calib.d0_scale), "D0 boost factor applied (1.0 = none)")
    hdr_out["HCUEFF"] = (float(U_eff), "effective ISRF energy density used for losses [eV/cm3]")
    hdr_out["HCGAMMA"] = (float(Gamma_inj), "injection spectral index")
    hdr_out["HCEC"] = (float(Ec_gev), "injection cutoff energy [GeV]")
    hdr_out["HCEINJ"] = (float(args.einj_erg), "total injected energy at theta=1 [erg]")

    write_cube(out_dnde, dnde, hdr_out, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_mu, mu, hdr_out, bunit="counts")
    print(f"  Wrote {out_dnde}")
    print(f"  Wrote {out_mu}  total counts: {float(np.nansum(mu)):.3f}")

    if not args.skip_psf:
        mcmc_dir = os.path.join(REPO_DIR, "Totani_reanalysis", "mcmc")
        if mcmc_dir not in sys.path:
            sys.path.insert(0, mcmc_dir)
        import psf_utils as psf  # noqa: E402  (Totani_reanalysis/mcmc/psf_utils.py)

        out_psf = os.path.join(args.outdir, f"mu_hardic_{tag}_psf_counts.fits")
        psf.convolve_template_file(out_mu, args.counts, out_psf, verbose=args.verbose)
        print(f"  Wrote {out_psf}")

    print(f"Done in {time.time()-t_start:.1f} s.")
    return out_mu


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    help="burst_s | burst_m | burst_l | cont | adv | cmbonly_<one of those>")
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed",
                    "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(DATA_DIR, "processed",
                    "expcube_1000to1000000.fits"))
    ap.add_argument("--outdir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--gamma-inj", type=float, default=2.2, help="Injection spectral index.")
    ap.add_argument("--ec-tev", type=float, default=10.0, help="Injection cutoff energy [TeV].")
    ap.add_argument("--einj-erg", type=float, default=1.0e52,
                    help="Total injected e+/e- energy at theta=1 [erg].")
    ap.add_argument("--u-eff-ev", type=float, default=None,
                    help="Override the KN-weighted effective ISRF energy density [eV/cm^3] "
                         "used for the transport/loss rate (default: computed).")
    ap.add_argument("--cmb-only-transport", default=True, action=argparse.BooleanOptionalAction,
                    help="For --variant cmbonly_*, also use the CMB-only U_eff for the "
                         "loss rate (default: yes -- see module docstring on why this "
                         "matters for the bracket to be genuinely ISRF-independent).")
    ap.add_argument("--v-wind-km-s", type=float, default=300.0,
                    help="Bipolar wind speed for --variant adv [km/s].")
    ap.add_argument("--t-cont-myr", type=float, default=50.0,
                    help="Injection duration for --variant cont [Myr].")
    ap.add_argument("--n-age-nodes", type=int, default=40,
                    help="Age-integration nodes for --variant cont.")
    ap.add_argument("--n-images", type=int, default=12,
                    help="Method-of-images terms each side of the z_h boundary.")

    ap.add_argument("--nr", type=int, default=200, help="(R,z) grid points in R.")
    ap.add_argument("--nz", type=int, default=200, help="(R,z) grid points in z.")
    ap.add_argument("--r-max-kpc", type=float, default=40.0)
    ap.add_argument("--z-max-kpc", type=float, default=40.0)
    ap.add_argument("--n-ee", type=int, default=90, help="Electron-energy grid points.")
    ap.add_argument("--ee-min-gev", type=float, default=1.0)
    ap.add_argument("--ee-max-gev", type=float, default=0.0,
                    help="0 (default) -> 30x the cutoff energy.")
    ap.add_argument("--s-max-kpc", type=float, default=30.0, help="LOS integration length.")
    ap.add_argument("--n-los-steps", type=int, default=300)

    ap.add_argument("--fast", action="store_true",
                    help="Coarse grid for a <1 minute smoke test (overrides nr/nz/n-los-steps/n-ee).")
    ap.add_argument("--skip-psf", action="store_true", help="Do not pre-build the PSF-convolved template.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute the calibration + emissivity grid, print diagnostics, stop "
                         "before LOS projection / FITS write.")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    if args.fast:
        args.nr, args.nz = 40, 40
        args.n_ee = 40
        args.n_los_steps = 60

    build(args)


if __name__ == "__main__":
    main()
