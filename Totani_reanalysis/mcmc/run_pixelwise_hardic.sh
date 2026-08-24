#!/usr/bin/env bash
# Launch the pixelwise MCMC hardening runs for the hard inverse-Compton
# (injected cosmic-ray population) variants, burst_m and cont.
#
# Why: the deterministic per-bin fits (mcmc/configs/systematics/hardic_burst_m.json,
# hardic_cont.json) already show the hard-IC coefficient pinned at exactly
# zero in the signal band (k04/k05/k06) and the halo amplitude ratio at 1.000,
# and agree with the MCMC to ~5% across well-populated bins. This script
# exists to harden the *correlation* estimate (r(halo, hard-IC) etc.) with
# real posterior chains, via systematics/posterior_covariance.py, for these
# same two variants. It is not expected to change the qualitative conclusion.
#
# Cost, read from the code (not guessed): base_pixelwise.json (used by every
# pixelwise_* config, this one included) sets nwalkers=64, nsteps<=60000,
# early_stop=true with autocorr_target=50. The one measured comparator
# (pixelwise_fig6, same walker/step settings) cost ~3-7 h per energy bin
# (individual bins logged at 25586 s and 10883 s wall; see
# logs/logs_pixelwise_global_paper/mcmc_pixelwise_fig6.log), i.e. roughly
# 2-3 days serial for the 13 energy bins. Each of the two configs below is
# that same cost again. DO NOT launch both unattended without accounting for
# ~4-6 days of combined wall time.
#
# Usage (run from the repo root, fermi conda env active):
#   bash Totani_reanalysis/mcmc/run_pixelwise_hardic.sh                 # both variants, full run
#   bash Totani_reanalysis/mcmc/run_pixelwise_hardic.sh burst_m         # one variant only
#   bash Totani_reanalysis/mcmc/run_pixelwise_hardic.sh cont
#
# Cheap verification before launching anything (recommended; seconds, not hours):
#   bash Totani_reanalysis/mcmc/run_pixelwise_hardic.sh --dry-run
#   bash Totani_reanalysis/mcmc/run_pixelwise_hardic.sh burst_m --dry-run
#
# Single-bin smoke test (still slow -- one full-settings bin is ~3-7 h; only
# use --energy-bin with a config that has had nsteps/nwalkers reduced first):
#   python Totani_paper_check/mcmc/run_mcmc_from_config.py \
#     --config Totani_reanalysis/mcmc/configs/pixelwise_hardic_burst_m.json \
#     --energy-bin 5
#
# After a run completes, sync outputs to the layout systematics/posterior_covariance.py
# reads by default (mirrors RUN_PAPER_RESULTS.md sec 4):
#   mkdir -p Totani_reanalysis/mcmc/fit_results/pixelwise_mcmc
#   for d in Totani_reanalysis/mcmc/pixelwise_mcmc_results_hardic_*; do
#     dest="Totani_reanalysis/mcmc/fit_results/pixelwise_mcmc/$(basename "$d")"
#     rm -rf "$dest"; cp -R "$d" "$dest"
#   done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUNNER="Totani_paper_check/mcmc/run_mcmc_from_config.py"

# ---- parse args ---------------------------------------------------------------
# NOTE: macOS ships bash 3.2, where expanding an EMPTY array under `set -u` is an
# unbound-variable error, and ${#arr[@]:-0} is not valid syntax. Plain
# space-separated strings sidestep the whole problem; the only values here are
# "burst_m", "cont" and "--dry-run", none of which contain whitespace.
DRY_RUN=""
VARIANTS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
        burst_m|cont)
            VARIANTS="$VARIANTS $arg"
            ;;
        *)
            echo "Unrecognised argument: $arg" >&2
            echo "Usage: $0 [burst_m|cont] [--dry-run]" >&2
            exit 1
            ;;
    esac
done
if [[ -z "${VARIANTS// /}" ]]; then
    VARIANTS="burst_m cont"
fi

# ---- helper ---------------------------------------------------------------------
run_variant() {
    local tag="$1"
    local cfg="Totani_reanalysis/mcmc/configs/pixelwise_hardic_${tag}.json"
    echo ""
    echo "========================================================"
    echo "  Pixelwise MCMC: hardic_${tag}  (config: ${cfg})"
    echo "========================================================"
    python "$RUNNER" --config "$cfg" $DRY_RUN
}

# ---- run --------------------------------------------------------------------
FAILED=""
for tag in $VARIANTS; do
    if run_variant "$tag"; then
        echo "  [OK] hardic_${tag}"
    else
        echo "  [FAILED] hardic_${tag}"
        FAILED="$FAILED $tag"
    fi
done

# ---- summary ----------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Pixelwise hard-IC run complete"
echo "========================================================"
if [[ -z "${FAILED// /}" ]]; then
    echo "  All variants succeeded."
else
    echo "  FAILED variants:"
    for f in $FAILED; do
        echo "    - $f"
    done
    exit 1
fi
