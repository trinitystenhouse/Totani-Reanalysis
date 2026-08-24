#!/usr/bin/env python3
r"""
Build a "diffuse-mismodelling" injection-null mock counts cube.

This is the key null test for the 20 GeV halo: if the true sky has an ICS
morphology that differs from the template the fit assumes, but contains NO halo,
does the baseline fit invent a spurious halo to absorb the residual?

Construction
------------
Truth = (no-halo best-fit model) + ICS morphology distortion, with NO halo:

    counts_true[k] = model_nohalo[k] + theta_ics[k] * ( ics_variant[k] - ics_base[k] )

inside the fit mask (outside, the observed counts are kept). model_nohalo and
theta_ics come from the no-halo global fit (fig4); ics_base is the nominal ICS
template and ics_variant is a latitude-morphology variant from
build_ics_variant_templates.py (default 'flatb'). The distortion preserves the
per-plane ICS flux (build_ics_variant_templates renormalises the ROI total), so
it is a pure shape mismodelling, not an overall-normalisation change.

Fit this cube with the BASELINE templates (config mock_diffuse_ics_null): a
recovered halo amplitude consistent with zero is the desired (passing) result; a
significant 20 GeV halo would show the feature can be a diffuse artefact.

Usage:
    python build_diffuse_mismodel_mock.py                 # uses ics flatb variant
    python build_diffuse_mismodel_mock.py --ics-variant steepb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
TEMPLATES = REPO / "fermi_data" / "totani" / "processed" / "templates"


def _load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _ics_index(labels) -> int:
    labs = [str(x) for x in labels.tolist()]
    if "ics" in labs:
        return labs.index("ics")
    raise KeyError(f"no 'ics' label in {labs}")


def _plane(path: Path, k: int) -> np.ndarray:
    d = fits.getdata(path)
    return np.asarray(d[k] if np.ndim(d) == 3 else d, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path,
                    default=REPO / "fermi_data/totani/processed/counts_ccube_1000to1000000.fits")
    ap.add_argument("--nohalo-dir", type=Path, default=PROJECT / "mcmc/global_fit_results_fig4")
    ap.add_argument("--ics-variant", default="flatb", choices=["flatb", "steepb"])
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "mcmc/mock_counts/global_mock_diffuse_ics_null_counts.fits")
    args = ap.parse_args()

    base_tpl = TEMPLATES / "mu_ics_counts.fits"
    var_tpl = TEMPLATES / f"mu_ics_{args.ics_variant}_counts.fits"
    if not var_tpl.exists():
        raise SystemExit(f"missing {var_tpl}; run build_ics_variant_templates.py first")

    with fits.open(args.counts) as hdul:
        counts0 = np.asarray(hdul[0].data, float)
        header = hdul[0].header.copy()
        extra = [h.copy() for h in hdul[1:]]

    nE = counts0.shape[0]
    cube = np.array(counts0, float, copy=True)
    rows = ["k E_GeV theta_ics injected_distortion_counts"]
    for k in range(nE):
        r = _load_npz(args.nohalo_dir / f"global_fit_k{k:02d}.npz")
        model = np.asarray(r["model_map"], float)
        mask = np.asarray(r.get("mask_2d", np.isfinite(model)), bool)
        idx = _ics_index(np.asarray(r["labels"], object))
        theta_ics = float(np.asarray(r["theta_ml"], float)[idx])
        distortion = theta_ics * (_plane(var_tpl, k) - _plane(base_tpl, k))
        plane = np.where(mask, model + distortion, counts0[k])
        cube[k] = np.clip(plane, 0.0, None)
        rows.append(f"{k:d} {float(r['Ectr_mev'])/1000.0:.6g} {theta_ics:.6e} "
                    f"{float(np.nansum(distortion[mask])):.6e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    h = header.copy()
    h["MOCKTYPE"] = "diffuse-mismodel ICS null (no halo)"
    h["ICSVAR"] = args.ics_variant
    fits.HDUList([fits.PrimaryHDU(cube.astype(np.float32), header=h), *extra]).writeto(args.out, overwrite=True)
    (args.out.parent / "global_mock_diffuse_ics_null_summary.txt").write_text("\n".join(rows) + "\n")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.parent / 'global_mock_diffuse_ics_null_summary.txt'}")


if __name__ == "__main__":
    main()
