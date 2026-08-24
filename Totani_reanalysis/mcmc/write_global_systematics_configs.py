#!/usr/bin/env python3
"""Write deterministic global-fit configs for paper systematics checks.

The generated matrix targets the robustness checks for the
rho^2 halo spectrum: diffuse/template composition, point-source masking, PSF
treatment, and deterministic null/recovery mock counts.  The fits are ordinary
L-BFGS-B global fits, not MCMC.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parent
CFG_DIR = HERE / "configs" / "systematics"
FIG_CFG_DIR = PROJECT / "figures" / "configs" / "systematics"

RHO2 = "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"
BASE_LABELS = ["gas", "iso", "ps", "loopA", "loopB", "ics", RHO2, "fb_pos", "fb_neg"]
COMMON_PATHS = {
    "counts": "../../../../fermi_data/totani/processed/counts_ccube_1000to1000000.fits",
    "expo": "../../../../fermi_data/totani/processed/expcube_1000to1000000.fits",
    "templates_dir": "../../../../fermi_data/totani/processed/templates",
    "ext_mask": "../../../../fermi_data/totani/processed/templates/mask_extended_sources.fits",
    "ps_catalog": "../../../../fermi_data/totani/templates/gll_psc_v35.fit",
}


def _variant(name: str, group: str, description: str, **overrides) -> dict:
    cfg = {
        "base_config": "../gce_global_base.json",
        "name": f"systematics_{name}",
        "systematics_group": group,
        "description": description,
        "global_dir": f"../../global_systematics/{name}",
        "mcmc_dir": f"../../global_systematics/{name}",
        "plot_configs": [f"../../../figures/configs/systematics/{name}.json"],
        "include_disk": False,
        "energy_bins": "all",
        "labels": list(BASE_LABELS),
        "negative_keys": ["nfw", "fb_neg"],
        "no_save_maps": False,
        "verbosity": 1,
        "plot_style": "totani",
        **COMMON_PATHS,
    }
    cfg.update(overrides)
    return cfg


def quadrant_variants() -> list[dict]:
    """NE/NW/SE/SW quadrant fits, each with the disk excluded and included.
    Quadrant = hemisphere (N: b>0, S: b<0) x longitude (E: l>0, W: l<0).
    Uses the rho2 (gamma=2) halo -- the annihilation morphology the science is
    built on -- to test whether the 20 GeV halo is consistent across the sky."""
    out = []
    for tag, hemi, lon in [("NE", "north", "east"), ("NW", "north", "west"),
                           ("SE", "south", "east"), ("SW", "south", "west")]:
        for disk, dlab in [(False, "excl"), (True, "incl")]:
            out.append(_variant(
                f"quad_{tag}_{dlab}", "quadrant",
                f"{tag} quadrant (hemisphere={hemi}, lon={lon}), "
                f"disk {'included' if disk else 'excluded'}.",
                hemisphere=hemi, lon_sign=lon, include_disk=disk,
            ))
    return out


def variants() -> list[dict]:
    return [
        _variant("baseline_rho2", "baseline", "Nominal disk-excluded rho2 global fit."),
        _variant(
            "diffuse_no_loopI",
            "diffuse",
            "Remove both Loop I shell templates. Bubbles re-extracted self-consistently "
            "without Loop I in the background (rebuild_bubbles_per_variant.py).",
            labels=["gas", "iso", "ps", "ics", RHO2,
                    "fb_pos_diffuse_no_loopI", "fb_neg_diffuse_no_loopI"],
        ),
        _variant(
            "diffuse_flat_bubbles",
            "diffuse",
            "Replace positive/negative bubble residual templates with a single flat bubble template.",
            labels=["gas", "iso", "ps", "loopA", "loopB", "ics", RHO2, "fb_flat"],
            negative_keys=["nfw"],
        ),
        _variant(
            "diffuse_no_bubbles",
            "diffuse",
            "Remove the bubble residual templates.",
            labels=["gas", "iso", "ps", "loopA", "loopB", "ics", RHO2],
            negative_keys=["nfw"],
        ),
        _variant(
            "diffuse_giem",
            "diffuse",
            "Replace the GALPROP gas+ICS diffuse with the Fermi-LAT Galactic "
            "Interstellar Emission Model (GIEM, gll_iem_v07). Requires the prepared "
            "template mu_giem_counts.fits (and mu_giem_psf_counts.fits for the "
            "pixelwise PSF fit) in templates_dir: reproject gll_iem_v07.fits onto the "
            "ROI CAR grid and multiply by exposure. This is the cross-check against "
            "the LAT standard background that Totani (2025) also performs.",
            labels=["giem", "iso", "ps", "loopA", "loopB", RHO2, "fb_pos", "fb_neg"],
        ),
        _variant(
            "diffuse_giem_only",
            "diffuse",
            "GIEM (gll_iem_v07) as the SOLE large-scale diffuse model: replaces gas+ICS "
            "AND Loop I + Fermi bubbles, since gll_iem_v07 already contains large-scale "
            "residual structures that overlap them. Diagnostic for whether the "
            "diffuse_giem halo suppression is double-counting (Loop I/bubbles fit twice) "
            "or a genuine diffuse-model effect. Compare the halo to diffuse_giem.",
            labels=["giem", "iso", "ps", RHO2],
            negative_keys=["nfw"],
        ),
        _variant(
            "mask_extended_only",
            "point_source_mask",
            "Use only the extended-source mask; do not mask bright point sources.",
            mask_bright_ps=False,
        ),
        _variant(
            "mask_loose_bright",
            "point_source_mask",
            "Less aggressive bright point-source mask.",
            bright_ps_mask="mask_sources_combined_loose.fits",
            ps_flux_threshold=1e-8,
            ps_radius_deg=0.75,
            ps_energy_dependent=False,
        ),
        _variant(
            "mask_aggressive_bright",
            "point_source_mask",
            "More aggressive bright point-source mask.",
            bright_ps_mask="mask_sources_combined_aggressive.fits",
            ps_flux_threshold=1e-9,
            ps_radius_deg=1.5,
            ps_max_radius_deg=3.0,
            ps_energy_dependent=False,
        ),
        _variant(
            "mask_energy_dependent",
            "point_source_mask",
            "Energy-dependent bright point-source mask using the LAT PSF radius scaling.",
            bright_ps_mask="mask_sources_combined_edep.fits",
            ps_flux_threshold=3e-9,
            ps_energy_dependent=True,
            psf_radius_factor=2.0,
        ),
        _variant(
            "psf_none",
            "psf",
            "Disable PSF forward-folding.",
            psf_labels=[],
        ),
        _variant(
            "psf_halo_only",
            "psf",
            "PSF-forward-fold only the NFW halo template.",
            psf_labels=["nfw"],
        ),
        _variant(
            "psf_wide_kernel",
            "psf",
            "Use the nominal PSF labels with a wider convolution truncation radius.",
            psf_rmax_factor=8.0,
        ),
        _variant(
            "mock_null_nohalo",
            "mock",
            "Fit deterministic no-halo mock counts built from the no-halo global model.",
            counts="../../mock_counts/global_mock_null_nohalo_counts.fits",
        ),
        _variant(
            "mock_recovery_rho2",
            "mock",
            "Fit deterministic no-halo mock counts with the baseline rho2 halo injected.",
            counts="../../mock_counts/global_mock_recovery_rho2_counts.fits",
        ),
        # --- ICS / ISRF spatial-morphology variation -----------------------------
        # Requires build_ics_variant_templates.py to have written
        # mu_ics_flatb{,_psf}_counts.fits and mu_ics_steepb{,_psf}_counts.fits.
        # The per-bin fit absorbs ICS *normalisation*, so only the latitude
        # MORPHOLOGY (flatter/steeper high-|b| gradient) is a real systematic.
        _variant(
            "ics_morphology_flat",
            "diffuse_ics",
            "Replace the nominal ICS template with a flatter-latitude ISRF variant "
            "(more inverse-Compton at high |b|), per-plane flux preserved. Tests the "
            "ISRF vertical scale-height uncertainty that competes with a smooth halo.",
            labels=["gas", "iso", "ps", "loopA", "loopB", "ics_flatb", RHO2,
                    "fb_pos_ics_morphology_flat", "fb_neg_ics_morphology_flat"],
        ),
        _variant(
            "ics_morphology_steep",
            "diffuse_ics",
            "Replace the nominal ICS template with a steeper-latitude ISRF variant "
            "(less inverse-Compton at high |b|), per-plane flux preserved.",
            labels=["gas", "iso", "ps", "loopA", "loopB", "ics_steepb", RHO2,
                    "fb_pos_ics_morphology_steep", "fb_neg_ics_morphology_steep"],
        ),
        # --- North / South hemisphere split --------------------------------------
        # A genuinely spherical halo must give a consistent spectrum in both
        # hemispheres; Loop I / bubble residuals are N-S asymmetric.
        _variant(
            "roi_north",
            "hemisphere",
            "Restrict the ROI to the northern Galactic hemisphere (b>0).",
            hemisphere="north",
        ),
        _variant(
            "roi_south",
            "hemisphere",
            "Restrict the ROI to the southern Galactic hemisphere (b<0).",
            hemisphere="south",
        ),
        # --- Diffuse-mismodelling injection null ---------------------------------
        # Requires build_diffuse_mismodel_mock.py. Truth = no-halo model + ICS
        # morphology distortion, NO halo: a recovered halo ~ 0 is the passing
        # result; a 20 GeV halo would show the feature can be a diffuse artefact.
        _variant(
            "mock_diffuse_ics_null",
            "mock",
            "Fit the diffuse-mismodelling mock: no-halo truth with a distorted ICS "
            "morphology and no injected halo. Spurious-halo null test.",
            counts="../../mock_counts/global_mock_diffuse_ics_null_counts.fits",
        ),
    ] + quadrant_variants() + cr_injection_variants() + cr_injection_bubbles_reextracted_variants() \
      + iem_grid_variants() + iem_grid_bubbles_reextracted_variants()


# --- Hard inverse-Compton injected-CR-population templates --
# See make_templates/hardic_physics.py, make_templates/build_hardic_template.py.
# Halo (RHO2) is
# RETAINED in every variant here (the point is to test whether it survives
# alongside a centrally-injected leptonic component, not to replace it -- the
# halo-removed comparator configs live separately, outside this manifest, in
# mcmc/configs/hardic/hardic_only_<tag>.json; see that directory's docstring
# for why they are deliberately NOT wired into write_global_systematics_configs
# / the manifest.json machinery).
HARDIC_TAGS = ["burst_s", "burst_m", "burst_l", "cont", "adv", "cmbonly_burst_m"]


def cr_injection_variants() -> list[dict]:
    out = []
    for tag in HARDIC_TAGS:
        label = f"hardic_{tag}"
        out.append(_variant(
            f"hardic_{tag}",
            "cr_injection",
            f"Centrally-injected hard inverse-Compton leptonic population, variant "
            f"'{tag}' (see hardic_physics.py module docstring for the transport/"
            f"emission model). Halo (RHO2) retained: tests whether the 20 GeV halo "
            f"survives alongside this component, not whether it replaces it.",
            labels=BASE_LABELS[:-2] + [label, "fb_pos", "fb_neg"],
            psf_labels=["nfw", "gas", "ics", "loopA", "loopB", "hardic"],
        ))
    return out


# --- Bubbles re-extracted under the hard-IC background --
# fb_pos/fb_neg are residual templates (see rebuild_bubbles_per_variant.py header),
# so strictly the hardic_<tag> variants above should use bubbles re-extracted with
# hardic_<tag> in the Stage-2 background, not the baseline fb_pos/fb_neg. Staged
# deliberately: re-extraction is prioritised for burst_m (primary morphology)
# and cont (second), not all six HARDIC_TAGS.
#
# These are DELIBERATELY separate variants from hardic_<tag> above (different
# name, different global_dir), not an in-place label swap like diffuse_no_loopI
# or ics_morphology_flat/steep use -- the hardic_<tag> Stage-1 (baseline-bubble)
# fits already exist and are the reported result; this keeps that result
# intact and lets the Stage-1 vs Stage-2 delta be read off directly.
#
# Requires mu_fb_pos_hardic_<tag>_counts.fits / mu_fb_neg_hardic_<tag>_counts.fits
# in templates_dir -- i.e. rebuild_bubbles_per_variant.py must have been run for
# that tag first (see systematics/run_hardic_bubble_reextraction.py). Until then
# these configs exist but will fail at fit time with a missing-file error, which
# is the correct behaviour (no silent fallback to the baseline bubbles).
BUBBLE_REEXTRACTED_HARDIC_TAGS = ["burst_m", "cont"]


def cr_injection_bubbles_reextracted_variants() -> list[dict]:
    out = []
    for tag in BUBBLE_REEXTRACTED_HARDIC_TAGS:
        label = f"hardic_{tag}"
        out.append(_variant(
            f"hardic_{tag}_bubbles_reextracted",
            "cr_injection_bubbles",
            f"Same as hardic_{tag}, but fb_pos/fb_neg are re-extracted with "
            f"hardic_{tag} included in the Stage-2 background (rebuild_bubbles_"
            f"per_variant.py), rather than reusing the baseline bubble residual. "
            f"Tests the sensitivity of the hard-IC result to this bubble "
            f"self-consistency point.",
            labels=BASE_LABELS[:-2] + [label, f"fb_pos_{label}", f"fb_neg_{label}"],
            psf_labels=["nfw", "gas", "ics", "loopA", "loopB", "hardic"],
        ))
    return out


# --- Ackermann et al. (2012, ApJ 750, 3) GALPROP IEM model grid -------------
# Genuine physically-varied interstellar-emission bracket: propagation halo
# height (z_h), cosmic-ray source distribution, HI spin temperature (T_S),
# halo radius (R), and E(B-V) reddening cut. Each model supplies its own
# gas_<tag> (pi0 + bremsstrahlung) and ics_<tag> template, which here replace
# the baseline "gas"/"ics" labels; every other template (iso, ps, loopA,
# loopB, the rho^2 halo, fb_pos, fb_neg) is kept as in the baseline. See the
# grid-implementation plan for the interstellar-emission-model systematic
# (Stage 3).
#
# iem_sl_z6 (SL_Z6_R30_T150_C2) is the ANCHOR: it is identified with the
# existing analysis baseline GALDEF (galdef_54_Lorimer_z6kpc_R30kpc_
# Ts150K_EBV2mag_TT25). It is included here as a validation cross-check --
# its fit is expected to closely match baseline_rho2 -- and is NOT an
# independent excursion of the diffuse model. Whoever later computes the
# systematic spread over the diffuse_grid group should exclude iem_sl_z6
# from that spread (see its description below).
#
# Tags are the fixed contract from the implementation plan; do not invent
# new ones here. The SS/SY/SO cosmic-ray source-distribution labels are
# carried as published (Ackermann et al. 2012); only SL (Lorimer) has been
# independently confirmed against our own GALDEF, per the implementation
# plan's Stage 0 note -- descriptions below are phrased accordingly.
IEM_GRID_MODELS = [
    # (GALPROP model id, tag, description of what varies vs. the anchor)
    ("SL_Z6_R30_T150_C2", "iem_sl_z6",
     "ANCHOR / VALIDATION model: identified with the existing analysis "
     "baseline GALDEF (Lorimer pulsar cosmic-ray sources, z_h=6 kpc, "
     "R=30 kpc halo radius, T_S=150 K HI spin temperature, E(B-V) cut "
     "2 mag). Included as a consistency check -- its fit is expected to "
     "closely match baseline_rho2 -- NOT an independent excursion; exclude "
     "it when computing the systematic spread over this group."),
    ("SL_Z4_R30_T150_C2", "iem_sl_z4",
     "Lorimer cosmic-ray source distribution, propagation halo height "
     "z_h=4 kpc (vs. 6 kpc anchor), otherwise as the anchor."),
    ("SL_Z8_R30_T150_C2", "iem_sl_z8",
     "Lorimer cosmic-ray source distribution, propagation halo height "
     "z_h=8 kpc (vs. 6 kpc anchor), otherwise as the anchor."),
    ("SL_Z10_R30_T150_C2", "iem_sl_z10",
     "Lorimer cosmic-ray source distribution, propagation halo height "
     "z_h=10 kpc (vs. 6 kpc anchor), otherwise as the anchor."),
    ("SS_Z6_R30_T150_C2", "iem_ss_z6",
     "SS cosmic-ray source distribution (Ackermann et al. 2012 model grid) "
     "at z_h=6 kpc, otherwise as the anchor."),
    ("SY_Z6_R30_T150_C2", "iem_sy_z6",
     "SY cosmic-ray source distribution at z_h=6 kpc, otherwise as the "
     "anchor."),
    ("SO_Z6_R30_T150_C2", "iem_so_z6",
     "SO cosmic-ray source distribution at z_h=6 kpc, otherwise as the "
     "anchor."),
    ("SS_Z4_R30_T150_C2", "iem_ss_z4",
     "SS cosmic-ray source distribution, propagation halo height z_h=4 kpc."),
    ("SS_Z10_R30_T150_C2", "iem_ss_z10",
     "SS cosmic-ray source distribution, propagation halo height z_h=10 kpc."),
    ("SY_Z4_R30_T150_C2", "iem_sy_z4",
     "SY cosmic-ray source distribution, propagation halo height z_h=4 kpc."),
    ("SY_Z10_R30_T150_C2", "iem_sy_z10",
     "SY cosmic-ray source distribution, propagation halo height z_h=10 kpc."),
    ("SL_Z6_R30_T100000_C2", "iem_sl_z6_tinf",
     "Lorimer cosmic-ray source distribution, HI spin temperature "
     "T_S=100000 K (effectively the optically-thin HI limit, vs. 150 K "
     "anchor), otherwise as the anchor."),
    ("SL_Z6_R30_T150_C5", "iem_sl_z6_c5",
     "Lorimer cosmic-ray source distribution, E(B-V) reddening cut of "
     "5 mag (vs. 2 mag anchor, admits more dust-obscured sky), otherwise "
     "as the anchor."),
    ("SL_Z6_R20_T150_C2", "iem_sl_z6_r20",
     "Lorimer cosmic-ray source distribution, propagation halo radius "
     "R=20 kpc (vs. 30 kpc anchor), otherwise as the anchor."),
]


def iem_grid_variants() -> list[dict]:
    """Ackermann et al. (2012) GALPROP IEM model-grid bracket: one config per
    model, replacing the baseline gas/ics labels with that model's own
    gas_<tag>/ics_<tag> templates. Every other template (iso, ps, loopA,
    loopB, the rho^2 halo, fb_pos, fb_neg) is kept as in the baseline -- the
    associated bubble re-extraction is a separate variant family, see
    iem_grid_bubbles_reextracted_variants below."""
    out = []
    for model_id, tag, desc in IEM_GRID_MODELS:
        labels = [
            f"gas_{tag}" if lab == "gas" else
            f"ics_{tag}" if lab == "ics" else
            lab
            for lab in BASE_LABELS
        ]
        out.append(_variant(
            tag, "diffuse_grid",
            f"GALPROP IEM model grid (Ackermann et al. 2012, ApJ 750, 3), "
            f"model '{model_id}'. Replaces the baseline gas+ICS diffuse pair "
            f"with this model's own gas_{tag}/ics_{tag} templates; every "
            f"other template unchanged. {desc}",
            labels=labels,
        ))
    return out


# --- Bubbles re-extracted under each IEM grid model's own gas/ics background --
# fb_pos/fb_neg are residual templates (see rebuild_bubbles_per_variant.py
# header), so as with the hardic_<tag> Stage-2 re-extraction, the grid
# variants above should strictly use bubbles re-extracted with gas_<tag>/
# ics_<tag> in the Stage-2 background, not the baseline fb_pos/fb_neg.
# Requires mu_fb_pos_<tag>_counts.fits / mu_fb_neg_<tag>_counts.fits in
# templates_dir -- i.e. rebuild_bubbles_per_variant.py must have been run
# for that tag first. Until then these configs exist but will fail at fit
# time with a missing-file error, which is the correct behaviour (no silent
# fallback to the baseline bubbles).
def iem_grid_bubbles_reextracted_variants() -> list[dict]:
    """Same 14 IEM grid models as iem_grid_variants, but with fb_pos/fb_neg
    swapped for fb_pos_<tag>/fb_neg_<tag>, re-extracted under each model's
    own gas_<tag>/ics_<tag> background (rebuild_bubbles_per_variant.py),
    following exactly how cr_injection_bubbles_reextracted_variants() does
    it for the hardic_<tag> family."""
    out = []
    for model_id, tag, desc in IEM_GRID_MODELS:
        labels = [
            f"gas_{tag}" if lab == "gas" else
            f"ics_{tag}" if lab == "ics" else
            f"fb_pos_{tag}" if lab == "fb_pos" else
            f"fb_neg_{tag}" if lab == "fb_neg" else
            lab
            for lab in BASE_LABELS
        ]
        out.append(_variant(
            f"{tag}_bubbles_reextracted", "diffuse_grid_bubbles",
            f"Same as {tag}, but fb_pos/fb_neg are re-extracted with "
            f"gas_{tag}/ics_{tag} in the Stage-2 background (rebuild_bubbles_"
            f"per_variant.py), rather than reusing the baseline bubble "
            f"residual. Model '{model_id}'. {desc}",
            labels=labels,
        ))
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_CFG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cfg in variants():
        name = cfg["name"].replace("systematics_", "")
        cfg_path = CFG_DIR / f"{name}.json"
        fig_cfg_path = FIG_CFG_DIR / f"{name}.json"
        _write_json(cfg_path, cfg)
        fig_cfg = {
            "fig": "global",
            "plot_style": "totani",
            "mcmc_stat": "f_ml",
            "mcmc_dir": f"../../../mcmc/global_systematics/{name}",
            "outdir": f"../../plots_global_systematics/{name}",
            "roi_lon": 60,
            "roi_lat": 60,
            "disk_cut": 10.0,
            "binsz": 0.125,
            "exclude_disk_in_plot": True,
            "labels": cfg["labels"],
        }
        _write_json(fig_cfg_path, fig_cfg)
        manifest.append({
            "name": name,
            "group": cfg["systematics_group"],
            "description": cfg["description"],
            "config": str(cfg_path.relative_to(PROJECT)),
            "plot_config": str(fig_cfg_path.relative_to(PROJECT)),
            "global_dir": cfg["global_dir"],
        })

    _write_json(CFG_DIR / "manifest.json", {"variants": manifest})
    print(f"Wrote {len(manifest)} systematics configs under {CFG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
