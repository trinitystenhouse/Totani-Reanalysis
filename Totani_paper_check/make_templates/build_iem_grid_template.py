#!/usr/bin/env python3
r"""
Build gas (pi0 decay + bremsstrahlung) and ICS diffuse templates from the
Ackermann et al. (2012) GALPROP v54 model grid
(galprop.stanford.edu/PaperIISuppMaterial/), for use as a genuine physical
diffuse-model bracket alongside the existing baseline templates
(gas_dnde.fits / ics_dnde.fits, built from the anchor model SL_Z6_R30_T150_C2
via build_gas_template.py / build_ics_template.py).

Why this script exists: the systematic budget needs a real diffuse-model
bracket, not an asserted one. The grid's SL_Z6_R30_T150_C2 model has the same
GALDEF filename stem as the existing baseline
(galdef_54_Lorimer_z6kpc_R30kpc_Ts150K_EBV2mag_TT25), so it is the anchor;
SL_Z4/Z8/Z10 vary the CR halo height z_h and form that bracket's halo-height
axis (R30/T150/C2 held fixed).

Unlike the GIEM cross-check (build_giem_template.py), which replaces gas+ICS
with a single combined map, this grid gives pi0 decay, bremsstrahlung and ICS
as separate HEALPix skymaps, so the existing gas/ICS split is preserved
exactly:
    gas_<tag> = pi0_decay_healpix + bremss_healpix   (fixed-ratio combination,
                                                        as in the baseline)
    ics_<tag> = ics_isotropic_healpix                (combined ICS)

Sky format (verified against the anchor SL_Z6_R30_T150_C2/OUTPUT,
2026-08-21): HEALPix NSIDE=128, ORDERING=RING, per the SKYMAP extension
header (PIXTYPE/ORDERING/NSIDE keywords). Confirmed empirically, not just
trusted from the header: pixel values at several (l, b) test points agree
with the existing CAR-format skymap for the same anchor model to the
~10-40% level expected from resolution-mismatched nearest-pixel comparison
(HEALPix nside=128 ~0.46 deg vs the CAR file's 0.5 deg pixels) -- not the
order-of-magnitude scramble a NEST/RING mislabel would produce. So no
`galpropskymapco` old-convention fix is needed for this grid's output.

Units: the HEALPix "Spectra" column (SKYMAP extension, TUNIT1='Intensity')
is differential intensity dN/dE [ph cm^-2 s^-1 sr^-1 MeV^-1] directly --
NOT E^2 dN/dE, unlike the older CAR-format *_skymap_54_*.fits.gz files read
by build_gas_template.py / build_ics_template.py. Verified by direct pixel
comparison against pion_decay_skymap_54_0e7e0006.fits.gz (the existing
CAR-format anchor file, same underlying model): hpx / (car / E^2) ~ 1
(0.88-1.36 across five widely spaced test points -- consistent with
resolution-limited pixel matching), whereas hpx / car ~ 1e-8, which rules
out the E^2 dN/dE convention. So, unlike the older scripts, this script does
NOT divide by E^2 when forming gas_dnde / ics_dnde.

Energy grid: 38 log10-spaced bins, 50 MeV to 814.008 GeV (0.1138286 dex per
bin) -- identical to the CAR-format anchor's CRVAL3/CDELT3 grid. The
analysis grid's 13 bin CENTRES (the interpolation target; the converter
interpolates onto centres, matching build_gas_template.py /
build_ics_template.py) span 1.510-814.000 GeV. The top centre sits 0.001%
inside the native maximum -- interpolation, not extrapolation, but with
essentially no margin at the top bin. Flag this if the native grid ever
changes.

Combined vs ring-decomposed maps: this script reads the pre-combined
`pi0_decay_healpix`, `bremss_healpix`, `ics_isotropic_healpix` maps rather
than summing the 51 ring x gas-phase files per component itself. Spot-check:
summing bremss_HII_ring_{1..17}_healpix and comparing to the grid's own
bremss_ionized_healpix (its pre-combined analogue) agrees in total flux to
0.03%. Using the combined maps avoids re-implementing that sum in this
script and is what the grid provides them for.

Naming: labels use the `iem_sl_z<N>` tag family (e.g. `gas_iem_sl_z4`,
`ics_iem_sl_z4`), verified against every substring matcher in the pipeline
with verify_hardic_naming.py --label. Deliberately does NOT contain "nfw" or
"fb_neg" (would become sign-unconstrained via totani_bounds' negative_keys)
and deliberately DOES contain "gas"/"ics" as a prefix (so it correctly picks
up the baseline PSF-convolution selector, exactly like the existing gas/ics
templates -- this is intended, not a naming collision).

Pipeline per template: read the two/one HEALPix component map(s) -> combine
(gas only) -> interpolate onto the 13 analysis energy-bin centres in log(E)
-> reproject HEALPix -> the 0.125 deg CAR ROI grid via bilinear HEALPix
interpolation (healpy.get_interp_val) -> multiply by exposure and solid
angle and dE -> write mu_<name>_counts.fits. PSF convolution is then handled
automatically downstream by run_global_from_config.py:_build_psf_templates.

Optional --compare-gas / --compare-ics let this same converter double as the
Stage 0 validation tool: given the anchor model, compare the freshly built
gas_dnde / ics_dnde against the existing baseline dnde maps and report
fractional agreement per energy bin over the ROI (|l| <= 60 deg,
10 <= |b| <= 60 deg).
"""

import argparse
import os
import sys

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Insert the repo root itself (for `shared.paths`) and repo_root/shared (so
# `totani_helpers` resolves as a plain subdirectory import) explicitly and
# ahead of anything already on sys.path. This sidesteps environments where a
# *different* editable-installed repo also happens to expose top-level
# `shared`/`totani_helpers` package names (e.g. a sibling public-mirror
# checkout) and would otherwise shadow the real one.
sys.path.insert(0, os.path.join(_REPO_ROOT, "shared"))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (  # noqa: E402
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    resample_exposure_logE_interp,
    write_cube,
)

from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
GRID_DIR = os.path.join(DATA_DIR, "galprop_iem_grid")

# Anchor model: exact GALDEF-filename match to the existing baseline.
ANCHOR_TAG = "iem_sl_z6"
ANCHOR_DIR = os.path.join(GRID_DIR, "OUTPUT")

# ROI used throughout the Totani reanalysis: |l| <= 60 deg, 10 <= |b| <= 60 deg.
ROI_LMAX = 60.0
ROI_BMIN = 10.0
ROI_BMAX = 60.0


def read_galprop_healpix_skymap(path):
    """
    Read one Ackermann-grid HEALPix skymap (.gz, gzipped FITS with a SKYMAP
    BINTABLE extension: one row per pixel, one 'Spectra' column of NBRBINS
    energy values, plus an ENERGIES extension giving those energies in MeV).

    Returns
    -------
    spectra : (npix, nE) float64 array, dN/dE [ph cm^-2 s^-1 sr^-1 MeV^-1]
    E_mev   : (nE,) float64 array, native energy grid [MeV]
    nside   : int
    ordering: str, 'RING' or 'NESTED' (as declared in the SKYMAP header;
              see module docstring for the empirical cross-check that this
              declaration is trustworthy for this grid)
    """
    with fits.open(path) as h:
        sky_hdr = h["SKYMAP"].header
        spectra = np.array(h["SKYMAP"].data["Spectra"], dtype=np.float64)
        en = h["ENERGIES"].data
        E_mev = np.array(en[en.columns.names[0]], dtype=np.float64)

    nside = int(sky_hdr["NSIDE"])
    ordering = str(sky_hdr["ORDERING"]).strip().upper()
    if spectra.shape[1] != E_mev.size:
        raise RuntimeError(
            f"{path}: SKYMAP has {spectra.shape[1]} energy columns but "
            f"ENERGIES has {E_mev.size} entries."
        )
    if spectra.shape[0] != hp.nside2npix(nside):
        raise RuntimeError(
            f"{path}: SKYMAP has {spectra.shape[0]} rows, expected "
            f"{hp.nside2npix(nside)} for NSIDE={nside}."
        )
    return spectra, E_mev, nside, ordering


class EnergyGridCoverageError(RuntimeError):
    """Raised when a model's native energy grid does not fully cover the
    analysis bin centres, so the caller must not proceed to interpolate."""


def check_energy_coverage(E_src, E_tgt, label):
    """
    Fail loudly if any analysis bin centre in E_tgt falls outside the native
    grid E_src, rather than letting resample_exposure_logE_interp's endpoint
    clamp silently substitute the boundary value for an out-of-range bin.

    The anchor model's top analysis bin (814.000 GeV) sits only 0.001%
    inside its native grid maximum (814.008 GeV) -- a margin thin enough
    that a differently-truncated grid in another model of the 128-model set
    would otherwise be extrapolated (via silent clamping) without any
    indication in the output.
    """
    src_min = float(np.min(E_src))
    src_max = float(np.max(E_src))
    tgt_min = float(np.min(E_tgt))
    tgt_max = float(np.max(E_tgt))
    if tgt_min < src_min or tgt_max > src_max:
        raise EnergyGridCoverageError(
            f"{label}: analysis bin centres span [{tgt_min:.6f}, {tgt_max:.6f}] MeV but "
            f"the native GALPROP grid only spans [{src_min:.6f}, {src_max:.6f}] MeV -- "
            f"refusing to interpolate, since resample_exposure_logE_interp would "
            f"otherwise clamp the out-of-range bin(s) to the boundary value "
            f"(silent extrapolation) rather than raising."
        )


def interp_energy_planes_healpix(spectra, E_src, E_tgt, label="<unknown component>"):
    """
    Interpolate a (npix, nE_src) HEALPix map in log(E) onto E_tgt (nE_tgt,).
    Guarded by check_energy_coverage() first (see there); once that passes,
    reuses totani_io.resample_exposure_logE_interp, which operates on
    (nE, ny, nx); here ny=npix, nx=1.

    Returns (npix, nE_tgt).
    """
    check_energy_coverage(E_src, E_tgt, label)
    cube_like = spectra.T[:, :, None]  # (nE_src, npix, 1)
    out = resample_exposure_logE_interp(cube_like, E_src, E_tgt)  # (nE_tgt, npix, 1)
    return out[:, :, 0].T  # (npix, nE_tgt)


def reproject_healpix_to_car(spectra_tgt, nside, ordering, lon2d_tgt, lat2d_tgt):
    """
    Bilinear HEALPix interpolation of a (npix, nE) map onto a target
    (ny, nx) lon/lat grid, for every energy plane in one vectorised call.

    Returns (nE, ny, nx).
    """
    ny, nx = lon2d_tgt.shape
    lon_flat = np.asarray(lon2d_tgt, dtype=np.float64).ravel()
    lat_flat = np.asarray(lat2d_tgt, dtype=np.float64).ravel()
    m = np.ascontiguousarray(spectra_tgt.T)  # (nE, npix)
    out = hp.get_interp_val(m, lon_flat, lat_flat, lonlat=True, nest=(ordering == "NESTED"))
    nE = spectra_tgt.shape[1]
    return out.reshape(nE, ny, nx)


def roi_mask(lon2d, lat2d):
    return (np.abs(lon2d) <= ROI_LMAX) & (np.abs(lat2d) >= ROI_BMIN) & (np.abs(lat2d) <= ROI_BMAX)


def report_fractional_agreement(name, new_dnde, old_path, roi, Ectr_mev):
    """Print per-energy-bin fractional agreement between new_dnde and the
    dnde cube at old_path, over the ROI mask."""
    with fits.open(old_path) as h:
        old = np.asarray(h[0].data, dtype=float)
    if old.shape != new_dnde.shape:
        print(f"  [{name}] SHAPE MISMATCH: new {new_dnde.shape} vs baseline {old.shape} "
              f"({old_path}) -- cannot compare directly.")
        return
    print(f"  Comparison: {name} vs {os.path.basename(old_path)}  (ROI: |l|<={ROI_LMAX:.0f}, "
          f"{ROI_BMIN:.0f}<=|b|<={ROI_BMAX:.0f} deg)")
    for k in range(new_dnde.shape[0]):
        a = new_dnde[k][roi]
        b = old[k][roi]
        good = np.isfinite(a) & np.isfinite(b) & (b != 0)
        if not np.any(good):
            print(f"    bin {k:2d} (E={Ectr_mev[k]/1000:8.3f} GeV): no valid ROI pixels")
            continue
        frac = (a[good] - b[good]) / b[good]
        print(f"    bin {k:2d} (E={Ectr_mev[k]/1000:8.3f} GeV): "
              f"median frac diff = {np.median(frac)*100:+7.3f}%   "
              f"mean = {np.mean(frac)*100:+7.3f}%   "
              f"|frac| 5-95pct = [{np.percentile(np.abs(frac),5)*100:6.3f}, "
              f"{np.percentile(np.abs(frac),95)*100:6.3f}]%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=ANCHOR_DIR,
                    help="Directory containing pi0_decay_healpix_54_*, bremss_healpix_54_* "
                         "and ics_isotropic_healpix_54_* for one grid model "
                         f"(default: anchor, {ANCHOR_DIR}).")
    ap.add_argument("--tag", default=ANCHOR_TAG,
                    help=f"Short label-family tag, e.g. 'iem_sl_z4' (default: '{ANCHOR_TAG}', "
                         "the anchor). Output templates are gas_<tag> / ics_<tag>.")
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed",
                    "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expcube", default=os.path.join(DATA_DIR, "processed",
                    "expcube_1000to1000000.fits"))
    ap.add_argument("--outdir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--compare-gas", default=None,
                    help="Path to an existing gas_dnde.fits to compare against (ROI, per bin). "
                         "Use with --model-dir pointing at the anchor to run the Stage 0 check.")
    ap.add_argument("--compare-ics", default=None,
                    help="Path to an existing ics_dnde.fits to compare against (ROI, per bin).")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    _, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)
    w_tgt = WCS(hdr).celestial
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    lon2d_tgt, lat2d_tgt = lonlat_grids(w_tgt, ny, nx)
    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)

    expo, E_exp = read_exposure(args.expcube)
    if expo.shape[0] != nE and E_exp is not None:
        expo = resample_exposure_logE(expo, E_exp, Ectr_mev)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")

    def find_component(prefix):
        cands = [f for f in os.listdir(args.model_dir)
                 if f.startswith(prefix) and f.endswith(".gz")]
        if len(cands) != 1:
            raise RuntimeError(
                f"Expected exactly one file starting with '{prefix}' in {args.model_dir}; "
                f"found {cands}"
            )
        return os.path.join(args.model_dir, cands[0])

    def load_component_on_grid(prefix):
        path = find_component(prefix)
        spectra, E_src, nside, ordering = read_galprop_healpix_skymap(path)
        spectra_tgt = interp_energy_planes_healpix(spectra, E_src, Ectr_mev, label=os.path.basename(path))
        cube = reproject_healpix_to_car(spectra_tgt, nside, ordering, lon2d_tgt, lat2d_tgt)
        return cube, path

    pi0_cube, pi0_path = load_component_on_grid("pi0_decay_healpix_54_")
    brem_cube, brem_path = load_component_on_grid("bremss_healpix_54_")
    ics_cube, ics_path = load_component_on_grid("ics_isotropic_healpix_54_")

    # Native units are already dN/dE [ph cm^-2 s^-1 sr^-1 MeV^-1] -- no E^2 division
    # (unlike the CAR-format anchor scripts; see module docstring).
    gas_dnde = pi0_cube + brem_cube
    ics_dnde = ics_cube

    mu_gas = gas_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
    mu_ics = ics_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    tag = args.tag
    out_gas_dnde = os.path.join(args.outdir, f"gas_{tag}_dnde.fits")
    out_gas_mu = os.path.join(args.outdir, f"mu_gas_{tag}_counts.fits")
    out_ics_dnde = os.path.join(args.outdir, f"ics_{tag}_dnde.fits")
    out_ics_mu = os.path.join(args.outdir, f"mu_ics_{tag}_counts.fits")

    write_cube(out_gas_dnde, gas_dnde, hdr, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_gas_mu, mu_gas, hdr, bunit="counts")
    write_cube(out_ics_dnde, ics_dnde, hdr, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_ics_mu, mu_ics, hdr, bunit="counts")

    print(f"Model dir: {args.model_dir}")
    print(f"  pi0:    {os.path.basename(pi0_path)}")
    print(f"  bremss: {os.path.basename(brem_path)}")
    print(f"  ics:    {os.path.basename(ics_path)}")
    print("Wrote", out_gas_dnde)
    print("Wrote", out_gas_mu, " total counts:", float(np.nansum(mu_gas)))
    print("Wrote", out_ics_dnde)
    print("Wrote", out_ics_mu, " total counts:", float(np.nansum(mu_ics)))

    n_nonfinite = int(np.sum(~np.isfinite(gas_dnde)) + np.sum(~np.isfinite(ics_dnde)))
    n_negative = int(np.sum(gas_dnde < 0) + np.sum(ics_dnde < 0))
    print(f"Finiteness/positivity check: {n_nonfinite} non-finite values, "
          f"{n_negative} negative values (gas+ics combined, {gas_dnde.size + ics_dnde.size} total).")

    roi = roi_mask(lon2d_tgt, lat2d_tgt)
    if args.compare_gas:
        report_fractional_agreement("gas", gas_dnde, args.compare_gas, roi, Ectr_mev)
    if args.compare_ics:
        report_fractional_agreement("ics", ics_dnde, args.compare_ics, roi, Ectr_mev)


if __name__ == "__main__":
    main()
