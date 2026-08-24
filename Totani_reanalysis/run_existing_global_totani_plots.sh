#!/usr/bin/env bash
set -euo pipefail

# Regenerate paper plots from the existing L-BFGS-B global fits and Totani-style
# cellwise fits. This does not run any global/MCMC sky fit. Set:
#
#   SKIP_FIGURES=1   to skip regenerating Fig. 2-7 global spectrum products
#   SKIP_PPPC=1      to skip the spectral PPPC scan/PPPC plotting products
#   SKIP_COLLECT=1   to skip collecting outputs into results/totani-global
#   COLLECT_MODE=move to move outputs instead of copying them
#
# Default collection mode is copy, so the source figure directories remain where
# the plotting scripts expect them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_PATH="$REPO_ROOT"
export PLOT_STYLE="${PLOT_STYLE:-paper}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.matplotlib}"

cd "$REPO_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate fermi
fi

LOG_DIR="$REPO_ROOT/Totani_paper_check/logs_existing_global_totani"
RESULTS_DIR="$REPO_ROOT/Totani_paper_check/results/totani-global"
mkdir -p "$LOG_DIR"
mkdir -p "$MPLCONFIGDIR"

run_logged() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo
  echo "================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name"
  echo "Log: $log"
  echo "Command: $*"
  echo "================================================================"
  "$@" 2>&1 | tee "$log"
}

require_dir() {
  local path="$1"
  if [ ! -d "$path" ]; then
    echo "Required directory is missing: $path" >&2
    exit 1
  fi
}

collect_path() {
  local src="$1"
  local dst_parent="$2"
  local mode="${COLLECT_MODE:-copy}"
  if [ ! -e "$src" ]; then
    echo "Skipping missing output: $src"
    return 0
  fi
  mkdir -p "$dst_parent"
  if [ "$mode" = "move" ]; then
    mv "$src" "$dst_parent/"
  else
    cp -R "$src" "$dst_parent/"
  fi
}

require_dir "Totani_paper_check/mcmc/global_fit_results_fig2_3"
require_dir "Totani_paper_check/mcmc/global_fit_results_fig4"
require_dir "Totani_paper_check/mcmc/global_fit_results_fig5"
require_dir "Totani_paper_check/mcmc/global_fit_results_fig6"
require_dir "Totani_paper_check/mcmc/global_fit_results_fig7"
require_dir "Totani_paper_check/mcmc/mcmc_results_fig2_3"
require_dir "Totani_paper_check/mcmc/mcmc_results_fig4"
require_dir "Totani_paper_check/mcmc/mcmc_results_fig5"
require_dir "Totani_paper_check/mcmc/mcmc_results_fig6"
require_dir "Totani_paper_check/mcmc/mcmc_results_fig7"

if [ "${SKIP_FIGURES:-0}" != "1" ]; then
  run_logged figures_global_fig2 \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig2_global.json

  run_logged figures_global_fig3 \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig3_global.json

  run_logged figures_global_fig4 \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig4_global.json

  run_logged figures_global_fig5 \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig5_global.json

  run_logged figures_global_fig6 \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig6_global.json

  run_logged figures_global_fig7 \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig7_global.json

  run_logged figures_global_fig5_w_disk \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig5_global_w_disk.json

  run_logged figures_global_fig6_w_disk \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig6_global_w_disk.json

  run_logged figures_global_fig7_w_disk \
    python Totani_paper_check/figures/make_figures_from_config.py \
      --config Totani_paper_check/figures/configs/fig7_global_w_disk.json
fi

run_logged plots_global_totani_overlay \
  python Totani_paper_check/figures/plot_global_totani_overlays.py \
    --global-tag global \
    --outdir Totani_paper_check/figures/plots_global_totani_overlay \
    --style "$PLOT_STYLE"

run_logged plots_fig8_global_comparison \
  python Totani_paper_check/figures/make_totani_fig8_global_comparison.py \
    --global-tag global \
    --outdir Totani_paper_check/figures/plots_fig8_global_comparison

if [ "${SKIP_PPPC:-0}" != "1" ]; then
  run_logged pppc_mass_scan_global_fig6 \
    python Totani_paper_check/run_pppc_mass_scan.py \
      --fit global_fig6 \
      --channels WW bb \
      --halo-reference global_rho2

  run_logged plot_global_fig6_ww_bb_pwave \
    python Totani_paper_check/figures/plot_global_fig6_ww_bb_pwave.py \
      --fit global_fig6 \
      --halo-reference global_rho2 \
      --output-dir Totani_paper_check/figures/plots_pppc_global_fig6 \
      --style "$PLOT_STYLE"

  run_logged velocity_overlay_global \
    python Totani_reanalysis/velocity_dep/plot_velocity_limits_overlay.py \
      --fit global_fig6 \
      --channels bb WW \
      --out-dir Totani_reanalysis/results/velocity_overlay \
      --style "$PLOT_STYLE"

  run_logged global_pppc_bestfit_points \
    python Totani_paper_check/figures/plot_pixelwise_pppc_bestfit_points.py \
      --ann-profile global_rho2 \
      --decay-profile global_rho1 \
      --tag global \
      --ann-title 'Global NFW-$\rho^2$' \
      --decay-title 'Global NFW-$\rho^1$' \
      --outdir Totani_paper_check/figures/plots_pppc_global_bestfit \
      --style "$PLOT_STYLE"
fi

if [ "${SKIP_COLLECT:-0}" != "1" ]; then
  mkdir -p "$RESULTS_DIR"

  collect_path "Totani_paper_check/figures/plots_global_fig2" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig3" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig4" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig5" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig6" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig7" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig5_w_disk" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig6_w_disk" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_fig7_w_disk" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_global_totani_overlay" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_fig8_global_comparison" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_pppc_global_fig6" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/figures/plots_pppc_global_bestfit" "$RESULTS_DIR/figures"
  collect_path "Totani_paper_check/results/pppc_mass_scan/global_fig6" "$RESULTS_DIR/data"
  collect_path "Totani_reanalysis/results/velocity_overlay" "$RESULTS_DIR/velocity_overlay"

  echo
  echo "Collected global/Totani outputs in: $RESULTS_DIR"
fi

echo
echo "Done. Logs are in: $LOG_DIR"
