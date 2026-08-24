#!/usr/bin/env python3
"""
build_pppc_template.py
======================
Build a PPPC4DMID annihilation counts template on the Fermi-LAT CCUBE grid.

Spatial morphology
------------------
The template inherits the same NFW-rho^2 line-of-sight integral used by
Totani's halo template (J_2 map, already built by build_nfw_template.py):

    J_2(l, b) = integral rho^2 dl   [M_sun^2 kpc^-5]

Energy dependence
-----------------
Unlike the phenomenological NFW shape template (which has identical spatial
morphology in every energy bin), the PPPC template is *energy-dependent*.
Each bin k gets a different amplitude from the PPPC spectrum:

    mu_pppc(E_k, pixel) = (sigma_v_ref / 8 pi m_chi^2)
                          * (dN/dE)_PPPC(E_k, m_chi)          [GeV^-1]
                          * J_2(pixel)                          [M_sun^2 kpc^-5]
                          * unit_conversion
                          * expo(E_k, pixel)                    [cm^2 s]
                          * omega(pixel)                         [sr]
                          * dE_k                                [MeV]

The fitted coefficient f_pppc from the MCMC then gives:

    <sigma v>  =  f_pppc * sigma_v_ref   [cm^3 s^-1]

Physical normalisation
----------------------
We use sigma_v_ref = 1e-25 cm^3 s^-1 as the reference cross section.
This keeps f_pppc ~ O(1-10) in the interesting region, avoiding numerical
issues in the MCMC sampler.

Pole normalisation
------------------
J_2 is pole-normalised (value = 1 at b = 90 deg), exactly as in
build_nfw_template.py. This means the PPPC template at the galactic pole is:

    E^2 dN/dE|_pole = f_pppc * sigma_v_ref / (8 pi m_chi^2)
                      * E_k^2 * (dN/dE)_PPPC(E_k) * J_pole

where J_pole = 8.93e14 M_sun^2 kpc^-5 in physical units (used in
smooth_nfw_sigma_v_from_norm in spectral_reshaping.py).

Output files
------------
For each (channel, mass) pair:

    mu_pppc_{channel}_{mass_tag}_counts.fits   counts template (nE, ny, nx)
    pppc_{channel}_{mass_tag}_E2dnde.fits      E^2 dN/dE map (nE, ny, nx)

where mass_tag = e.g. "WW_m0500" for a 500 GeV WW annihilation.

Scattering transfer (optional)
-------------------------------
Pass --scatter-operator and --scatter-Lambda (or --scatter-yeff for
higgs_portal) to apply the single-scatter redistribution kernel to
the PPPC spectrum before building the counts template. The transferred
template label gets a "_scat_{operator}_{coupling}" suffix and can be
used as a drop-in replacement in the MCMC config.

Usage
-----
    # Build WW template at 500 GeV
    python build_pppc_template.py --channel WW --mass 500

    # Build bb and tautau at several masses
    python build_pppc_template.py --channel bb --mass 400 500 650 800

    # Build all three channels at Totani's preferred mass range
    python build_pppc_template.py --all-channels --mass-scan

    # With scattering transfer (dipole_magnetic, Lambda=1 TeV)
    python build_pppc_template.py --channel WW --mass 650 \\
        --scatter-operator dipole_magnetic --scatter-Lambda 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = Path(os.environ.get(
    "REPO_PATH",
    str(_HERE.parent.parent),
))
_PAPER_CHECK = _HERE.parent
sys.path.insert(0, str(_PAPER_CHECK))
sys.path.insert(0, str(_REPO / "Totani_reanalysis"))

DATA_DIR = _REPO / "fermi_data" / "totani" / "processed"
TEMPLATE_DIR = DATA_DIR / "templates"

from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    write_cube,
)
from make_templates.build_nfw_template import (
    los_integral_rhopow_map,
    pole_normalization_value,
    smax_to_rvir_kpc,
)

# ---------------------------------------------------------------------------
# Physical constants and NFW parameters (Totani / Via Lactea II)
# ---------------------------------------------------------------------------

SIGMA_V_REF     = 1e-25       # cm^3 s^-1  — reference cross section for f_pppc
J_POLE_MSUN2_KPC5 = 8.93e14  # M_sun^2 kpc^-5 — Totani Eq. 4.2

# Unit conversion: M_sun^2 kpc^-5 cm^3 s^-1 GeV^-1 -> counts-compatible units
# We need J in cm^-5 so we can multiply by sigma_v [cm^3/s] / (8 pi m^2 [GeV^2]) * dN/dE [1/GeV].
# 1 M_sun = 1.989e30 kg = 1.989e30 / 1.783e-27 GeV/c^2 = 1.115e57 GeV
# 1 kpc   = 3.086e21 cm
MSUN_TO_GEV     = 1.115e57    # GeV per M_sun
KPC_TO_CM       = 3.086e21    # cm per kpc

# J_2 in GeV^2 cm^-5
J_POLE_GEV2_CM5 = J_POLE_MSUN2_KPC5 * (MSUN_TO_GEV ** 2) / (KPC_TO_CM ** 5)

# Standard NFW parameters (Totani / Via Lactea II)
NFW_RS_KPC  = 21.0
NFW_RHOS    = 8.1e6    # M_sun kpc^-3
NFW_R0_KPC  = 8.0
NFW_RVIR_KPC = 402.0

# PPPC channels available in the gamma table
PPPC_CHANNELS = {
    "WW":     "W",
    "bb":     "b",
    "tautau": r"\[Tau]",
}


# ---------------------------------------------------------------------------
# PPPC spectrum loader  (identical logic to spectral_reshaping.pppc_dnde_from_table)
# ---------------------------------------------------------------------------

def _pppc_dnde(E_bins_GeV: np.ndarray, m_ann_GeV: float, channel: str,
               table_path: Path) -> np.ndarray:
    """Return dN/dE [GeV^-1] at each E_bins_GeV for the given mass and channel."""
    from astropy.table import Table

    E_bins_GeV = np.asarray(E_bins_GeV, float)
    col = PPPC_CHANNELS.get(channel, channel)
    table = Table.read(str(table_path), format="ascii.fast_basic", guess=False, delimiter=" ")
    if col not in table.colnames:
        raise ValueError(
            f"Channel '{channel}' -> column '{col}' not in PPPC table. "
            f"Available: {table.colnames}"
        )

    masses   = np.asarray(table["mDM"], float)
    log10x   = np.asarray(table["Log[10,x]"], float)
    dndlogx  = np.asarray(table[col], float)

    logm_target = np.log10(float(m_ann_GeV))
    logx_target = np.log10(E_bins_GeV / float(m_ann_GeV))
    unique_masses = np.unique(masses)
    unique_logm   = np.log10(unique_masses)

    if logm_target < unique_logm.min() or logm_target > unique_logm.max():
        raise ValueError(
            f"m_ann={m_ann_GeV:g} GeV outside PPPC table range "
            f"[{unique_masses.min():g}, {unique_masses.max():g}] GeV."
        )

    spectra = []
    for m in unique_masses:
        msk = masses == m
        order = np.argsort(log10x[msk])
        lx = log10x[msk][order]
        vals = dndlogx[msk][order]
        spectra.append(np.interp(logx_target, lx, vals, left=0.0, right=0.0))
    spectra = np.array(spectra)   # (n_masses, nE)

    dnde_logx = np.array([np.interp(logm_target, unique_logm, spectra[:, i])
                           for i in range(len(E_bins_GeV))])
    # Convert dN/dlog10x -> dN/dE:  dN/dE = dN/dlog10x / (E * ln10)
    return dnde_logx / (E_bins_GeV * np.log(10.0))


# ---------------------------------------------------------------------------
# Optional scattering transfer
# ---------------------------------------------------------------------------

def _apply_scattering_transfer(
    dnde: np.ndarray,
    E_bins_GeV: np.ndarray,
    m_chi: float,
    operator: str,
    coupling: float,
    dm_type: str,
    majorana: bool,
    c_s: float, c_p: float, c_phi: float,
    n_theta: int = 600,
) -> np.ndarray:
    """
    Apply single-scatter spectral transfer to a dN/dE spectrum.
    Returns modified dN/dE [GeV^-1] on the same energy axis.
    """
    from spectral_reshaping import (
        ReshapingConfig,
        build_dsigma_grid,
        build_kernel,
        compute_tau_spectrum,
        apply_single_scatter_transfer,
    )

    # phi_0 proportional to E^2 * dN/dE (the quantity the reshaping operates on)
    phi0 = E_bins_GeV**2 * dnde

    cfg = ReshapingConfig(
        m_chi=m_chi,
        Lambda=coupling if operator != "higgs_portal" else 1e3,
        dm_type=dm_type,
        operator=operator,
        c_s=c_s, c_p=c_p, c_phi=c_phi,
        majorana=majorana,
        y_eff=coupling if operator == "higgs_portal" else 1.0,
        n_theta=n_theta,
        E_bins=E_bins_GeV,
        phi_0=phi0,
        phi_data=phi0,   # not used in transfer, just needs same shape
        phi_err=np.ones_like(phi0),
    )

    cos_theta, dsig, sigma_tot = build_dsigma_grid(cfg)
    tau = compute_tau_spectrum(cfg)
    K   = build_kernel(cfg, cos_theta, dsig, sigma_tot)
    phi_transferred, _, _ = apply_single_scatter_transfer(phi0, tau, K, E_bins_GeV)

    # Convert back to dN/dE
    dnde_out = phi_transferred / E_bins_GeV**2
    dnde_out = np.where(np.isfinite(dnde_out) & (dnde_out >= 0.0), dnde_out, 0.0)
    print(f"  Scattering transfer: tau_max={tau.max():.3e}  "
          f"max fractional change in dN/dE = "
          f"{np.max(np.abs(dnde_out - dnde) / (dnde + 1e-300)):.3e}")
    return dnde_out


# ---------------------------------------------------------------------------
# NFW J_2 map (pole-normalised)
# ---------------------------------------------------------------------------

def _build_j2_pole_normalised(lon2d, lat2d, *, n_s: int = 2048) -> np.ndarray:
    """Build J_2 / J_pole: the pole-normalised NFW-rho^2 LOS integral."""
    J = los_integral_rhopow_map(
        lon2d, lat2d,
        rho_power=2.0,
        gamma=1.0,
        rs_kpc=NFW_RS_KPC,
        rhos=NFW_RHOS,
        R0_kpc=NFW_R0_KPC,
        rvir_kpc=NFW_RVIR_KPC,
        n_s=n_s,
    )
    J_pole = pole_normalization_value(
        rho_power=2.0,
        gamma=1.0,
        rs_kpc=NFW_RS_KPC,
        rhos=NFW_RHOS,
        R0_kpc=NFW_R0_KPC,
        rvir_kpc=NFW_RVIR_KPC,
        n_s=max(4096, n_s),
    )
    return J / J_pole   # dimensionless, = 1 at pole


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_pppc_template(
    *,
    channel:      str,
    m_ann_GeV:    float,
    outdir:       Path,
    counts_path:  Path,
    expo_path:    Path,
    table_path:   Path,
    n_s:          int   = 2048,
    binsz:        float = 0.125,
    # Scattering transfer (optional)
    scatter_operator: str | None = None,
    scatter_coupling: float      = 1e3,
    scatter_dm_type:  str        = "fermionic",
    scatter_majorana: bool       = False,
    scatter_c_s:      float      = 1.0,
    scatter_c_p:      float      = 0.0,
    scatter_c_phi:    float      = 1.0,
    scatter_n_theta:  int        = 600,
) -> Path:
    """
    Build and save a PPPC counts template.  Returns the path of the saved
    mu_pppc_*_counts.fits file.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    mass_tag = f"{channel}_m{int(round(m_ann_GeV)):04d}"
    scat_tag = ""
    if scatter_operator is not None:
        c_str = f"{scatter_coupling:.2e}".replace("+", "").replace(".", "p")
        scat_tag = f"_scat_{scatter_operator}_{c_str}"
    label = f"pppc_{mass_tag}{scat_tag}"

    out_mu    = outdir / f"mu_{label}_counts.fits"
    out_e2    = outdir / f"{label}_E2dnde.fits"

    print(f"\nBuilding PPPC template: {label}")
    print(f"  m_ann = {m_ann_GeV:.1f} GeV  channel = {channel}")

    # --- Load CCUBE grid ---
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(str(counts_path))
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial
    E_bins_GeV = Ectr_mev / 1000.0   # [GeV]

    # --- Load + resample exposure ---
    expo_raw, E_expo_mev = read_exposure(str(expo_path))
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)  # (nE, ny, nx)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling failed: nE mismatch.")

    # --- Solid angle + lon/lat ---
    omega  = pixel_solid_angle_map(wcs, ny, nx, binsz)        # (ny, nx) [sr]
    lon2d, lat2d = lonlat_grids(wcs, ny, nx)

    # --- PPPC dN/dE [GeV^-1] at each energy bin ---
    print(f"  Loading PPPC spectrum from {table_path.name}...")
    dnde = _pppc_dnde(E_bins_GeV, m_ann_GeV, channel, table_path)  # (nE,)
    print(f"  dN/dE range: {dnde.min():.3e} – {dnde.max():.3e} GeV^-1")

    # --- Optional scattering transfer ---
    if scatter_operator is not None:
        print(f"  Applying scattering transfer: operator={scatter_operator}  "
              f"coupling={scatter_coupling:.2e}")
        dnde = _apply_scattering_transfer(
            dnde, E_bins_GeV, m_ann_GeV,
            scatter_operator, scatter_coupling,
            scatter_dm_type, scatter_majorana,
            scatter_c_s, scatter_c_p, scatter_c_phi,
            n_theta=scatter_n_theta,
        )

    # --- NFW J_2 / J_pole spatial map ---
    print(f"  Computing NFW-rho^2 J map (n_s={n_s})...")
    j2_norm = _build_j2_pole_normalised(lon2d, lat2d, n_s=n_s)  # (ny, nx)

    # --- Prefactor: sigma_v_ref / (8 pi m_chi^2) * J_pole_GeV2_CM5 ---
    # Units check:
    #   sigma_v_ref [cm^3 s^-1] / (8 pi m^2 [GeV^2]) * J_pole [GeV^2 cm^-5]
    #   * dN/dE [GeV^-1] * dE_k [MeV] * expo [cm^2 s] * omega [sr]
    # = [cm^3/s / (GeV^2) * GeV^2/cm^5] * [1/GeV] * [MeV] * [cm^2 s] * [sr]
    # = [cm^{-2}] * [MeV/GeV] * [cm^2 s] * [sr] / s
    # = [MeV/GeV sr] = [1e-3 sr] -> divide by 1e3 to get sr-less counts
    # Actually expo * omega * dE in standard Fermi-LAT units -> counts directly.

    prefactor = SIGMA_V_REF / (8.0 * np.pi * (m_ann_GeV**2)) * J_POLE_GEV2_CM5
    # Units: cm^3/s / GeV^2 * GeV^2/cm^5 = cm^-2/s
    # Then * dnde [1/GeV] * dE [MeV -> GeV x 1e-3] * expo [cm^2 s] * omega [sr]
    # * j2_norm [dimensionless]
    # = cm^-2/s * 1/GeV * GeV * 1e-3 * cm^2 s * sr = 1e-3 sr ... almost
    # The standard LAT counts formula is: mu = intensity * expo * omega * dE
    # with intensity in [ph cm^-2 s^-1 sr^-1 MeV^-1] and dE in MeV.
    # Our intensity = sigma_v/(8pi m^2) * J_pole * j2_norm * dN/dE [GeV^-1]
    #               = sigma_v/(8pi m^2) * J_pole * j2_norm * dN/dE * 1e-3 [MeV^-1]
    # So mu_k = prefactor * 1e-3 * dnde_k * j2_norm * expo_k * omega * dE_k
    # where dE_k is in MeV and dnde_k in GeV^-1.

    prefactor_mev = prefactor * 1e-3   # absorb the GeV -> MeV conversion for dnde

    # --- Build E^2 dN/dE and counts cubes ---
    E2dnde_cube = np.zeros((nE, ny, nx), dtype=np.float32)
    mu_cube     = np.zeros((nE, ny, nx), dtype=np.float32)

    for k in range(nE):
        if dnde[k] <= 0.0 or not np.isfinite(dnde[k]):
            continue
        # E^2 dN/dE at pole [MeV cm^-2 s^-1 sr^-1], at j2=1
        # = prefactor * dnde_k [GeV^-1] * J_pole * E_k^2 [MeV^2]
        # but for plotting convenience we store E2 in MeV cm^-2 s^-1 sr^-1:
        # intensity [ph cm^-2 s^-1 sr^-1 MeV^-1] * E^2 [MeV^2]
        intensity_k = prefactor_mev * dnde[k]  # [ph cm^-2 s^-1 sr^-1 MeV^-1] at j2=1
        e2_k = intensity_k * (Ectr_mev[k] ** 2)  # E^2 dN/dE [MeV cm^-2 s^-1 sr^-1] at j2=1

        # Spatial map
        E2dnde_cube[k] = (e2_k * j2_norm).astype(np.float32)

        # Counts: intensity * j2_norm * expo[k] * omega * dE_k
        mu_k = intensity_k * j2_norm * expo[k] * omega * dE_mev[k]
        mu_cube[k] = mu_k.astype(np.float32)

    # Zero pixels outside ROI and non-finite
    roi = (np.abs(lon2d) <= 60.0) & (np.abs(lat2d) <= 60.0)
    for k in range(nE):
        mu_cube[k][~roi | ~np.isfinite(mu_cube[k])] = 0.0
        E2dnde_cube[k][~roi | ~np.isfinite(E2dnde_cube[k])] = 0.0

    # --- Save ---
    hdr_out = hdr.copy()
    hdr_out["PPPC_CH"]  = (channel,         "PPPC annihilation channel")
    hdr_out["PPPC_M"]   = (m_ann_GeV,       "Annihilation mass [GeV]")
    hdr_out["SIGV_REF"] = (SIGMA_V_REF,     "Reference sigma_v [cm^3/s]; f_pppc * SIGV_REF = <sigma v>")
    hdr_out["J_POLE"]   = (J_POLE_MSUN2_KPC5, "Galactic pole J factor [Msun^2 kpc^-5]")
    hdr_out["SCAT_OP"]  = (scatter_operator or "none", "Scattering operator applied")
    hdr_out["SCAT_C"]   = (scatter_coupling, "Scattering coupling (Lambda [GeV] or y_eff)")

    write_cube(str(out_mu), mu_cube, hdr_out, bunit="counts")
    write_cube(str(out_e2), E2dnde_cube, hdr_out, bunit="MeV cm-2 s-1 sr-1")

    mu_sum = float(np.nansum(mu_cube))
    print(f"  Saved: {out_mu.name}  (total counts in cube = {mu_sum:.3e})")
    print(f"  Saved: {out_e2.name}")
    print(f"  Label for MCMC config: {label}")
    return out_mu


# ---------------------------------------------------------------------------
# Mass scan grid
# ---------------------------------------------------------------------------

MASS_SCAN_GEV = [
    200, 300, 400, 500, 600, 650, 700, 750, 800, 900, 1000,
    1200, 1500, 2000,
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PPPC4DMID annihilation counts template on the Fermi-LAT CCUBE grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Channel / mass
    p.add_argument("--channel", default="WW",
                   choices=list(PPPC_CHANNELS.keys()),
                   help="Annihilation channel.")
    p.add_argument("--all-channels", action="store_true", default=False,
                   help="Build templates for WW, bb, and tautau.")
    p.add_argument("--mass", type=float, nargs="+", default=[650.0],
                   help="Annihilation mass(es) [GeV].")
    p.add_argument("--mass-scan", action="store_true", default=False,
                   help=f"Build templates for the standard mass grid: {MASS_SCAN_GEV}")

    # Paths
    p.add_argument("--counts",    default=str(DATA_DIR / "counts_ccube_1000to1000000.fits"))
    p.add_argument("--expo",      default=str(DATA_DIR / "expcube_1000to1000000.fits"))
    p.add_argument("--outdir",    default=str(TEMPLATE_DIR))
    p.add_argument("--pppc-table",default=None,
                   help="Path to AtProduction_gammas.dat. "
                        "Defaults to Totani_reanalysis/data/AtProduction_gammas.dat.")

    # NFW integration
    p.add_argument("--n-s", type=int, default=2048,
                   help="LOS integration steps for J_2 map.")
    p.add_argument("--binsz", type=float, default=0.125)

    # Scattering transfer (optional)
    p.add_argument("--scatter-operator", default=None,
                   choices=[
                       "dipole_magnetic", "dipole_electric", "charge_radius",
                       "anapole", "rayleigh_even", "rayleigh_odd", "rayleigh_full",
                       "higgs_portal",
                   ],
                   help="Apply scattering transfer with this operator. "
                        "If not set, no transfer is applied.")
    p.add_argument("--scatter-Lambda", type=float, default=1e3,
                   help="EFT cutoff scale Lambda [GeV] for scattering transfer "
                        "(ignored for higgs_portal).")
    p.add_argument("--scatter-yeff",   type=float, default=1.0,
                   help="Effective coupling y_eff for higgs_portal scattering transfer.")
    p.add_argument("--scatter-dm-type", default="fermionic",
                   choices=["fermionic", "scalar"])
    p.add_argument("--scatter-majorana", action="store_true", default=False)
    p.add_argument("--scatter-c-s",   type=float, default=1.0)
    p.add_argument("--scatter-c-p",   type=float, default=0.0)
    p.add_argument("--scatter-c-phi", type=float, default=1.0)
    p.add_argument("--scatter-n-theta", type=int, default=600)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve PPPC table path
    if args.pppc_table is not None:
        table_path = Path(args.pppc_table)
    else:
        table_path = _REPO / "Totani_reanalysis" / "data" / "AtProduction_gammas.dat"
    if not table_path.exists():
        raise FileNotFoundError(
            f"PPPC table not found: {table_path}\n"
            "Download AtProduction_gammas.dat from "
            "https://www.marcocirelli.net/PPPC4DMID.html and place it at "
            "Totani_reanalysis/data/AtProduction_gammas.dat, or pass --pppc-table."
        )

    channels = list(PPPC_CHANNELS.keys()) if args.all_channels else [args.channel]
    masses   = MASS_SCAN_GEV if args.mass_scan else [float(m) for m in args.mass]

    # Scattering coupling: Lambda or y_eff depending on operator
    if args.scatter_operator == "higgs_portal":
        scatter_coupling = float(args.scatter_yeff)
    else:
        scatter_coupling = float(args.scatter_Lambda)

    outdir = Path(args.outdir)
    built  = []

    for channel in channels:
        for mass in masses:
            out = build_pppc_template(
                channel=channel,
                m_ann_GeV=mass,
                outdir=outdir,
                counts_path=Path(args.counts),
                expo_path=Path(args.expo),
                table_path=table_path,
                n_s=args.n_s,
                binsz=args.binsz,
                scatter_operator=args.scatter_operator,
                scatter_coupling=scatter_coupling,
                scatter_dm_type=args.scatter_dm_type,
                scatter_majorana=args.scatter_majorana,
                scatter_c_s=args.scatter_c_s,
                scatter_c_p=args.scatter_c_p,
                scatter_c_phi=args.scatter_c_phi,
                scatter_n_theta=args.scatter_n_theta,
            )
            built.append(out)

    print(f"\n{'='*60}")
    print(f"Built {len(built)} template(s):")
    for p in built:
        print(f"  {p.name}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
