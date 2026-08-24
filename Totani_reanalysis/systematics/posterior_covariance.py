#!/usr/bin/env python3
"""
Posterior template covariance from the pixelwise MCMC chains.

The diffuse, bubble, Loop I and halo templates are competing descriptions of
the same smooth Galactocentric residual, so combining one-at-a-time
systematic variations in quadrature assumes an independence the model does
not have. This script extracts the *actual* posterior correlation between
template normalisations, per energy bin, directly from the retained
pixelwise chains -- no re-running required.

Inputs : mcmc/fit_results/pixelwise_mcmc/pixelwise_mcmc_results_fig6/mcmc_results_k*.npz
         each holding chain (steps, walkers, n_templates) and logprob (steps, walkers).

Outputs (default results/posterior-covariance/):
  posterior_covariance.npz            machine-readable: corr, cov, labels, energies
  correlation_matrices.txt            labelled 9x9 correlation matrix per energy bin
  halo_correlation_summary.txt        halo-vs-each-template correlation vs energy
  correlation_matrix_peak.{pdf,png}   heatmap at the 20.8 GeV peak bin
  halo_correlations_vs_energy.{pdf,png}

NOTE ON BURN-IN: the saved chains are NOT burn-in-trimmed. Early steps sit at
logprob ~ -1e13 versus a stationary ~ -1.6e5. Correlations computed on the raw
chain are meaningless. Burn-in is detected per bin from the logprob trace and
discarded with a safety margin; the discarded fraction is reported.

Usage:
    python systematics/posterior_covariance.py
    python systematics/posterior_covariance.py --results-dir <dir> --outdir <dir>
"""

import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ----------------------------------------------------------------------------
# presentation
# ----------------------------------------------------------------------------

# Short, paper-facing names for the fitted template normalisations.
SHORT = {
    "gas_psf": "gas",
    "gas": "gas",
    "iso": "iso",
    "ps": "PS",
    "loopA_psf": "Loop I A",
    "loopA": "Loop I A",
    "loopB_psf": "Loop I B",
    "loopB": "Loop I B",
    "ics_psf": "ICS",
    "ics": "ICS",
    "fb_pos": r"bubbles $+$",
    "fb_neg": r"bubbles $-$",
    "giem": "GIEM",
}

# Templates that are competing descriptions of the same smooth, large-scale
# Galactocentric residual. The halo is added at runtime.
LARGE_SCALE = ["gas", "ics", "loopA", "loopB", "fb_pos", "fb_neg", "iso"]


def short_label(lab):
    """Map a raw chain label to a short display name."""
    lab = str(lab)
    if lab.startswith("nfw"):
        return r"halo ($\rho^2$)"
    return SHORT.get(lab, lab)


def is_halo(lab):
    return str(lab).startswith("nfw")


def base_key(lab):
    """Strip the _psf suffix so LARGE_SCALE membership can be tested."""
    return str(lab)[:-4] if str(lab).endswith("_psf") else str(lab)


# ----------------------------------------------------------------------------
# burn-in
# ----------------------------------------------------------------------------


def detect_burnin(logprob, tol_sigma=10.0, margin=2.0, min_frac=0.05):
    """Locate the end of burn-in from the logprob trace.

    The stationary level and its scatter are estimated from the last quarter of
    the chain. Burn-in ends at the first step from which *every* walker stays
    within tol_sigma of that level. A multiplicative margin and a floor of
    min_frac of the chain are then applied, so the answer is conservative.

    Returns (n_discard, diagnostics dict).
    """
    n_steps = logprob.shape[0]
    tail = logprob[-max(n_steps // 4, 10):]
    level = np.median(tail)
    scatter = np.std(tail)
    # Guard against a pathologically tight tail making the test unsatisfiable.
    tol = max(tol_sigma * scatter, 1e-6 * abs(level), 1.0)

    within = np.all(np.abs(logprob - level) < tol, axis=1)
    if within.any():
        # first index from which the condition holds for the remainder
        ok_from = n_steps
        for i in range(n_steps):
            if within[i:].all():
                ok_from = i
                break
    else:
        ok_from = n_steps // 2

    n_discard = int(np.ceil(ok_from * margin))
    n_discard = max(n_discard, int(np.ceil(min_frac * n_steps)))
    n_discard = min(n_discard, n_steps - 50)  # always keep >=50 steps
    n_discard = max(n_discard, 0)

    return n_discard, {
        "stationary_logprob": float(level),
        "stationary_scatter": float(scatter),
        "first_stationary_step": int(ok_from),
        "n_discard": int(n_discard),
        "n_steps": int(n_steps),
    }


# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------


def load_bin(path):
    d = np.load(path, allow_pickle=True)
    if "chain" not in d:
        raise KeyError(
            f"{os.path.basename(path)} has no 'chain' array -- it was stripped by "
            "shrink_mcmc_outfiles.py. Use the pixelwise_mcmc results, not quick_global."
        )
    return (
        np.asarray(d["chain"], dtype=float),
        np.asarray(d["logprob"], dtype=float),
        [str(x) for x in d["labels"]],
        float(d["Ectr_mev"]),
    )


def corr_from_chain(chain, n_discard):
    """Flatten a post-burn-in chain and return (corr, cov, means, stds, n_samples)."""
    flat = chain[n_discard:].reshape(-1, chain.shape[-1])
    cov = np.cov(flat, rowvar=False)
    std = np.sqrt(np.diag(cov))
    # Templates pinned at a bound have ~zero variance; their correlations are
    # undefined rather than zero, so mark them NaN instead of dividing by ~0.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)
    corr[~np.isfinite(corr)] = np.nan
    return corr, cov, flat.mean(axis=0), std, flat.shape[0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results-dir",
        default=os.path.join(
            repo, "mcmc", "fit_results", "pixelwise_mcmc",
            "pixelwise_mcmc_results_fig6"),
        help="directory of mcmc_results_k*.npz with retained chains",
    )
    ap.add_argument("--outdir",
                    default=os.path.join(repo, "results", "posterior-covariance"))
    ap.add_argument("--peak-mev", type=float, default=20756.9,
                    help="energy of the peak bin to highlight (MeV)")
    ap.add_argument("--band-mev", type=float, nargs=2, default=(10000.0, 40000.0),
                    help="signal band for the averaged summary (MeV)")
    ap.add_argument("--tol-sigma", type=float, default=10.0,
                    help="burn-in tolerance in units of stationary logprob scatter")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.results_dir, "mcmc_results_k*.npz")))
    if not files:
        raise SystemExit(f"no mcmc_results_k*.npz found in {args.results_dir}")

    energies, corrs, covs, labels_ref = [], [], [], None
    means, stds, nsamp, burn = [], [], [], []

    for f in files:
        chain, logprob, labels, ectr = load_bin(f)
        if labels_ref is None:
            labels_ref = labels
        elif labels != labels_ref:
            raise SystemExit(f"label mismatch in {f}: {labels} != {labels_ref}")

        n_discard, diag = detect_burnin(logprob, tol_sigma=args.tol_sigma)
        corr, cov, mu, sd, n = corr_from_chain(chain, n_discard)

        energies.append(ectr)
        corrs.append(corr)
        covs.append(cov)
        means.append(mu)
        stds.append(sd)
        nsamp.append(n)
        burn.append(diag)

    energies = np.array(energies)
    corrs = np.array(corrs)
    covs = np.array(covs)
    means = np.array(means)
    stds = np.array(stds)

    disp = [short_label(l) for l in labels_ref]
    halo_idx = next(i for i, l in enumerate(labels_ref) if is_halo(l))
    peak_k = int(np.argmin(np.abs(energies - args.peak_mev)))
    in_band = (energies >= args.band_mev[0]) & (energies <= args.band_mev[1])

    # ---------------- machine-readable ----------------
    np.savez_compressed(
        os.path.join(args.outdir, "posterior_covariance.npz"),
        corr=corrs, cov=covs, mean=means, std=stds,
        energies_mev=energies, labels=np.array(labels_ref),
        display_labels=np.array(disp), halo_index=halo_idx,
        n_samples=np.array(nsamp),
        n_discard=np.array([b["n_discard"] for b in burn]),
    )

    # ---------------- correlation matrices ----------------
    with open(os.path.join(args.outdir, "correlation_matrices.txt"), "w") as fh:
        fh.write("# Posterior correlation between fitted template normalisations\n")
        fh.write(f"# source: {args.results_dir}\n")
        fh.write("# burn-in detected per bin from the logprob trace and discarded\n\n")
        for k, e in enumerate(energies):
            b = burn[k]
            fh.write(f"=== bin k{k:02d}  E = {e/1000.0:.2f} GeV ===\n")
            fh.write(f"    burn-in discarded: {b['n_discard']}/{b['n_steps']} steps "
                     f"(stationary from step {b['first_stationary_step']}); "
                     f"{nsamp[k]} samples used\n")
            head = "".join(f"{d:>13s}" for d in disp)
            fh.write(f"{'':>13s}{head}\n")
            for i, d in enumerate(disp):
                row = "".join(
                    f"{corrs[k, i, j]:>13.3f}" if np.isfinite(corrs[k, i, j])
                    else f"{'--':>13s}" for j in range(len(disp)))
                fh.write(f"{d:>13s}{row}\n")
            fh.write("\n")

    # ---------------- halo summary ----------------
    ls_idx = [i for i, l in enumerate(labels_ref)
              if base_key(l) in LARGE_SCALE and not is_halo(l)]
    with open(os.path.join(args.outdir, "halo_correlation_summary.txt"), "w") as fh:
        fh.write("# Posterior correlation of the halo normalisation with every other template\n")
        fh.write("# These are the degeneracies that a quadrature-sum systematic budget ignores.\n\n")
        hdr = "".join(f"{d:>13s}" for j, d in enumerate(disp) if j != halo_idx)
        fh.write(f"{'E [GeV]':>10s}{hdr}{'  |  halo sigma/mean':>22s}\n")
        for k, e in enumerate(energies):
            row = "".join(f"{corrs[k, halo_idx, j]:>13.3f}"
                          if np.isfinite(corrs[k, halo_idx, j]) else f"{'--':>13s}"
                          for j in range(len(disp)) if j != halo_idx)
            frac = stds[k, halo_idx] / means[k, halo_idx] if means[k, halo_idx] else np.nan
            fh.write(f"{e/1000.0:>10.2f}{row}{frac:>22.4f}\n")

        fh.write("\n# Signal-band mean (%.0f-%.0f GeV)\n"
                 % (args.band_mev[0] / 1000, args.band_mev[1] / 1000))
        band = "".join(f"{np.nanmean(corrs[in_band, halo_idx, j]):>13.3f}"
                       for j in range(len(disp)) if j != halo_idx)
        fh.write(f"{'band':>10s}{band}\n")

        fh.write("\n# Strongest halo degeneracies at the peak bin "
                 f"(E = {energies[peak_k]/1000:.2f} GeV)\n")
        order = np.argsort([-abs(corrs[peak_k, halo_idx, j]) if j != halo_idx else 0
                            for j in range(len(disp))])
        for j in order:
            if j == halo_idx or not np.isfinite(corrs[peak_k, halo_idx, j]):
                continue
            fh.write(f"    halo <-> {disp[j]:<14s} r = {corrs[peak_k, halo_idx, j]:+.3f}\n")

        # How much of the halo's own uncertainty is shared with the large-scale set?
        fh.write("\n# Large-scale degeneracy summary at the peak bin\n")
        sub = [halo_idx] + ls_idx
        c = corrs[peak_k][np.ix_(sub, sub)]
        r2 = np.nansum(corrs[peak_k, halo_idx, ls_idx] ** 2)
        fh.write(f"    sum of squared halo correlations with large-scale templates: {r2:.3f}\n")
        fh.write("    (a value approaching 1 means the halo normalisation is almost\n")
        fh.write("     entirely determined by the other large-scale components)\n")

    # ---------------- figures ----------------
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": 0.8, "savefig.bbox": "tight",
    })

    # (a) correlation heatmap at the peak bin
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    m = np.ma.masked_invalid(corrs[peak_k])
    im = ax.imshow(m, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax.set_xticks(range(len(disp)))
    ax.set_yticks(range(len(disp)))
    ax.set_xticklabels(disp, rotation=45, ha="right")
    ax.set_yticklabels(disp)
    for i in range(len(disp)):
        for j in range(len(disp)):
            v = corrs[peak_k, i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_title(f"Posterior template correlation, E = {energies[peak_k]/1000:.1f} GeV")
    fig.colorbar(im, ax=ax, fraction=0.046, label="correlation coefficient")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"correlation_matrix_peak.{ext}"), dpi=200)
    plt.close(fig)

    # (b) halo correlations vs energy
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for j in range(len(disp)):
        if j == halo_idx:
            continue
        y = corrs[:, halo_idx, j]
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(energies / 1000.0, y, marker="o", ms=3, lw=1.2, label=disp[j])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvspan(args.band_mev[0] / 1000, args.band_mev[1] / 1000,
               color="0.85", zorder=0, label="signal band")
    ax.set_xscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("correlation with halo normalisation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"halo_correlations_vs_energy.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- console ----------------
    print(f"wrote -> {args.outdir}")
    print(f"bins: {len(energies)}   peak bin k{peak_k:02d} at {energies[peak_k]/1000:.2f} GeV")
    print(f"burn-in discarded (steps): {[b['n_discard'] for b in burn]}")
    print(f"\nhalo correlations at {energies[peak_k]/1000:.2f} GeV:")
    for j in range(len(disp)):
        if j == halo_idx:
            continue
        v = corrs[peak_k, halo_idx, j]
        print(f"   halo <-> {disp[j]:<16s} r = {v:+.3f}" if np.isfinite(v)
              else f"   halo <-> {disp[j]:<16s} r = undefined (zero variance)")


if __name__ == "__main__":
    main()
