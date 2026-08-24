#!/usr/bin/env python3
r"""
Reconstruct the PIXELWISE fig4 (no-halo) residual maps from the MCMC coefficient
summaries, and write them as global_fit_k{NN}.npz files with the same keys the
morphology code expects (resid_map, mask_2d, Ectr_mev).

The pixelwise MCMC fit stores only per-template coefficients (f_ml / f_p50), not
a residual map -- unlike the deterministic quick_global fit (run_global_fit.py),
which saves resid_map directly. This script rebuilds the residual
    resid_k = counts_k - sum_m f[m,k] * template_m(k)
so the angular-profile figure can use the pixelwise baseline instead of the
deterministic one.

Usage (from Totani_reanalysis/):
    python morphology/make_pixelwise_fig4_resid.py            # f_ml, default dirs
    python morphology/make_pixelwise_fig4_resid.py --stat f_p50
then point the figure at the output:
    python morphology/plot_angular_profile_vs_models.py \
        --fitdir mcmc/fit_results/pixelwise_mcmc/pixelwise_mcmc_results_fig4_resid
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
from astropy.io import fits

_HERE = Path(__file__).resolve().parent
_REAN = _HERE.parent
_TD = _REAN.parent / "fermi_data" / "totani" / "processed" / "templates"
_COUNTS = _TD.parent / "counts_ccube_1000to1000000.fits"
_EDEP = _TD / "mask_sources_combined_edep.fits"


def _tmpl(label: str, k: int) -> np.ndarray:
    """Unit-amplitude counts template for `label` in energy bin k."""
    cands = sorted(glob.glob(str(_TD / f"mu_{label}_counts.fits"))
                   + glob.glob(str(_TD / f"mu_{label}*counts.fits")))
    if not cands:
        raise FileNotFoundError(f"no counts template for {label}")
    d = fits.getdata(cands[0])
    return np.asarray(d[k] if np.ndim(d) == 3 else d, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcmc-dir", type=Path,
                    default=_REAN / "mcmc/fit_results/pixelwise_mcmc/pixelwise_mcmc_results_fig4")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <mcmc-dir>_resid")
    ap.add_argument("--stat", default="f_ml", choices=["f_ml", "f_p50"])
    ap.add_argument("--nbins", type=int, default=13)
    a = ap.parse_args(argv)
    out = a.out_dir or a.mcmc_dir.with_name(a.mcmc_dir.name + "_resid")
    out.mkdir(parents=True, exist_ok=True)

    counts = fits.getdata(_COUNTS).astype(float)          # (nbins, ny, nx)
    edep = fits.getdata(_EDEP)
    for k in range(a.nbins):
        f = a.mcmc_dir / f"mcmc_results_k{k:02d}.npz"
        if not f.exists():
            print(f"[skip] {f.name} missing"); continue
        z = np.load(f, allow_pickle=True)
        labels = [str(x) for x in z["labels"]]
        coeff = np.asarray(z[a.stat], float)
        model = np.zeros_like(counts[k])
        for lab, c in zip(labels, coeff):
            model += c * _tmpl(lab, k)
        resid = counts[k] - model
        mask = np.asarray(edep[k] if np.ndim(edep) == 3 else edep, bool) & np.isfinite(resid)
        resid_out = np.where(mask, resid, np.nan)
        np.savez(out / f"global_fit_k{k:02d}.npz",
                 resid_map=resid_out, mask_2d=mask,
                 Ectr_mev=float(z["Ectr_mev"]))
        print(f"  k{k:02d}  E={float(z['Ectr_mev'])/1000:6.1f} GeV  "
              f"npix={int(mask.sum())}  wrote {out.name}/global_fit_k{k:02d}.npz")
    print(f"\nDone ({a.stat}). Now run:\n  python morphology/plot_angular_profile_vs_models.py "
          f"--fitdir {out.relative_to(_REAN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
