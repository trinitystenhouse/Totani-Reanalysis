#!/usr/bin/env bash
set -euo pipefail

# Build residual diagnostics from existing MCMC coefficient summaries.
# This does not launch any new MCMC fit.
#
# Usage:
#   MCMC_DIR=Totani_reanalysis/mcmc/fit_results/pixelwise_mcmc/pixelwise_mcmc_results_fig6 \
#     bash Totani_reanalysis/run_mcmc_residual_diagnostics.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_PATH="$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.matplotlib}"

cd "$REPO_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate fermi
fi

MCMC_DIR="${MCMC_DIR:-Totani_paper_check/mcmc/global_fit_results_fig6}"
OUTDIR="${OUTDIR:-Totani_paper_check/results/mcmc-residual-diagnostics/$(basename "$MCMC_DIR")}"
STAT="${STAT:-f_p50}"

mkdir -p "$OUTDIR" "$MPLCONFIGDIR"

python Totani_paper_check/figures/mcmc_residual_diagnostics.py \
  --mcmc-dir "$MCMC_DIR" \
  --outdir "$OUTDIR" \
  --stat "$STAT"

echo "Residual diagnostics written to: $OUTDIR"
