import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    write_cube,
)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_dir = os.environ.get(
        "REPO_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    tmpl_dir = os.path.join(data_dir, "processed", "templates")
    mu_file = os.path.join(tmpl_dir, args.mu)

    with fits.open(mu_file) as hdul:
        data = np.array(hdul[0].data, float)
        hdr = hdul[0].header

    if data.ndim != 3:
        raise RuntimeError(f"Expected a 3D cube (nE, ny, nx), got shape {data.shape}")

    w = WCS(hdr).celestial
    mu_map = np.nansum(data, axis=0)
    mu_per_E = np.nansum(data, axis=(1, 2))

    fig = plt.figure(figsize=(11, 4))
    ax0 = fig.add_subplot(1, 2, 1, projection=w)
    im = ax0.imshow(mu_map, origin="lower")
    ax0.set_title("mu_iso summed over energy")
    ax0.set_xlabel("RA")
    ax0.set_ylabel("Dec")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("counts")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(np.arange(mu_per_E.size), mu_per_E, lw=1.5)
    ax1.set_title("Total expected counts per energy bin")
    ax1.set_xlabel("energy bin index")
    ax1.set_ylabel("counts")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.mu))[0]
        out = os.path.join(os.path.dirname(args.mu), f"{base}_check.png")
    fig.savefig(out, dpi=200)
    print("✓ Wrote", out)


if __name__ == "__main__":
    main()
