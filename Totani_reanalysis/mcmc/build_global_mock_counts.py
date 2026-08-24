#!/usr/bin/env python3
"""Build deterministic mock counts cubes for global null/recovery tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
RHO2 = "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"


def _load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def _label_index(labels: np.ndarray, needle: str) -> int:
    labs = [str(x) for x in labels.tolist()]
    if needle in labs:
        return labs.index(needle)
    cands = [i for i, lab in enumerate(labs) if "nfw" in lab.lower()]
    if len(cands) != 1:
        raise KeyError(f"Could not identify halo label in {labs}")
    return cands[0]


def _read_template(templates_dir: Path, label: str, k: int) -> np.ndarray:
    path = templates_dir / f"mu_{label}_counts.fits"
    if not path.exists():
        raise FileNotFoundError(path)
    data = fits.getdata(path)
    if data.ndim == 3:
        return np.asarray(data[k], float)
    return np.asarray(data, float)


def build_mock(
    *,
    counts_path: Path,
    nohalo_dir: Path,
    halo_dir: Path,
    templates_dir: Path,
    out_null: Path,
    out_recovery: Path,
    halo_multiplier: float,
    poisson_seed: int | None,
) -> None:
    with fits.open(counts_path) as hdul:
        counts0 = np.asarray(hdul[0].data, float)
        header = hdul[0].header.copy()
        extra_hdus = [hdu.copy() for hdu in hdul[1:]]

    nE = counts0.shape[0]
    null_cube = np.array(counts0, dtype=float, copy=True)
    rec_cube = np.array(counts0, dtype=float, copy=True)
    rng = np.random.default_rng(poisson_seed) if poisson_seed is not None else None

    rows = [
        "k E_GeV theta_halo injected_counts_null_model injected_counts_halo",
    ]
    for k in range(nE):
        nohalo = _load_result(nohalo_dir / f"global_fit_k{k:02d}.npz")
        halo = _load_result(halo_dir / f"global_fit_k{k:02d}.npz")
        model = np.asarray(nohalo["model_map"], float)
        mask = np.asarray(nohalo.get("mask_2d", np.isfinite(model)), bool)

        idx = _label_index(np.asarray(halo["labels"], object), RHO2)
        template_labels = np.asarray(halo.get("template_labels", halo["labels"]), object)
        template_label = str(template_labels[idx])
        theta = float(np.asarray(halo["theta_ml"], float)[idx])
        halo_template = _read_template(templates_dir, template_label, k)
        injected = float(halo_multiplier) * theta * halo_template

        null_plane = np.where(mask, model, counts0[k])
        rec_plane = np.where(mask, model + injected, counts0[k])
        null_plane = np.clip(null_plane, 0.0, None)
        rec_plane = np.clip(rec_plane, 0.0, None)
        if rng is not None:
            null_plane = rng.poisson(null_plane)
            rec_plane = rng.poisson(rec_plane)
        null_cube[k] = null_plane
        rec_cube[k] = rec_plane
        rows.append(
            f"{k:d} {float(halo['Ectr_mev'])/1000.0:.8g} {theta:.8e} "
            f"{float(np.nansum(model[mask])):.8e} {float(np.nansum(injected[mask])):.8e}"
        )

    for out, cube, kind in [(out_null, null_cube, "no-halo null"), (out_recovery, rec_cube, "rho2 recovery")]:
        out.parent.mkdir(parents=True, exist_ok=True)
        h = header.copy()
        h["MOCKTYPE"] = kind
        h["HALOMULT"] = float(halo_multiplier)
        h["POISSON"] = poisson_seed is not None
        fits.HDUList([fits.PrimaryHDU(cube.astype(np.float32), header=h), *extra_hdus]).writeto(
            out,
            overwrite=True,
        )
        print(f"Wrote {kind} mock counts: {out}")
    summary = out_null.parent / "global_mock_counts_summary.txt"
    summary.write_text("\n".join(rows) + "\n")
    print(f"Wrote mock summary: {summary}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=Path, default=REPO / "fermi_data/totani/processed/counts_ccube_1000to1000000.fits")
    ap.add_argument("--nohalo-dir", type=Path, default=PROJECT / "mcmc/global_fit_results_fig4")
    ap.add_argument("--halo-dir", type=Path, default=PROJECT / "mcmc/global_fit_results_fig6")
    ap.add_argument("--templates-dir", type=Path, default=REPO / "fermi_data/totani/processed/templates")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "mcmc/mock_counts")
    ap.add_argument("--halo-multiplier", type=float, default=1.0)
    ap.add_argument("--poisson-seed", type=int, default=None)
    args = ap.parse_args()
    build_mock(
        counts_path=args.counts,
        nohalo_dir=args.nohalo_dir,
        halo_dir=args.halo_dir,
        templates_dir=args.templates_dir,
        out_null=args.out_dir / "global_mock_null_nohalo_counts.fits",
        out_recovery=args.out_dir / "global_mock_recovery_rho2_counts.fits",
        halo_multiplier=args.halo_multiplier,
        poisson_seed=args.poisson_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
