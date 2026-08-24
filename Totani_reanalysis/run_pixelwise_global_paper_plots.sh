#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere. Set SKIP_MCMC=1 to regenerate only the figures/PPPC
# products from already completed pixelwise MCMC outputs.

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

LOG_DIR="$REPO_ROOT/Totani_paper_check/logs_pixelwise_global_paper"
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

# run_logged build_energy_dependent_point_source_mask \
#   python Totani_paper_check/mcmc/mask_point_sources.py \
#     --counts fermi_data/totani/processed/counts_ccube_1000to1000000.fits \
#     --psc fermi_data/totani/templates/gll_psc_v35.fit \
#     --combine-with fermi_data/totani/processed/templates/mask_extended_sources.fits \
#     --out fermi_data/totani/processed/templates/mask_sources_combined_edep.fits \
#     --flux-threshold 3e-9 \
#     --energy-dependent \
#     --psf-radius-factor 2.0

if [ "${SKIP_MCMC:-0}" != "1" ]; then
  # run_logged mcmc_pixelwise_fig4 \
  #   python Totani_paper_check/mcmc/run_mcmc_from_config.py \
  #     --config Totani_paper_check/mcmc/configs/pixelwise_fig4.json

  # run_logged mcmc_pixelwise_fig5 \
  #   python Totani_paper_check/mcmc/run_mcmc_from_config.py \
  #     --config Totani_paper_check/mcmc/configs/pixelwise_fig5.json

  # run_logged mcmc_pixelwise_fig5_w_disk \
  #   python Totani_paper_check/mcmc/run_mcmc_from_config.py \
  #     --config Totani_paper_check/mcmc/configs/pixelwise_fig5_w_disk.json

  # run_logged mcmc_pixelwise_fig6 \
  #   python Totani_paper_check/mcmc/run_mcmc_from_config.py \
  #     --config Totani_paper_check/mcmc/configs/pixelwise_fig6.json

  # run_logged mcmc_pixelwise_fig6_w_disk \
  #   python Totani_paper_check/mcmc/run_mcmc_from_config.py \
  #     --config Totani_paper_check/mcmc/configs/pixelwise_fig6_w_disk.json

  run_logged mcmc_pixelwise_fig7 \
    python Totani_paper_check/mcmc/run_mcmc_from_config.py \
      --config Totani_paper_check/mcmc/configs/pixelwise_fig7.json

  run_logged mcmc_pixelwise_fig7_w_disk \
    python Totani_paper_check/mcmc/run_mcmc_from_config.py \
      --config Totani_paper_check/mcmc/configs/pixelwise_fig7_w_disk.json
  
  run_logged mcmc_pixelwise_fig2_3 \
    python Totani_paper_check/mcmc/run_mcmc_from_config.py \
      --config Totani_paper_check/mcmc/configs/pixelwise_fig2_3.json

fi

# Rebuild the per-figure spectra/tables from the pixelwise MCMC outputs.
# run_logged figures_pixelwise_fig2 \
#   python Totani_paper_check/figures/make_figures_from_config.py \
#     --config Totani_paper_check/figures/configs/fig2_pixelwise_global.json

# run_logged figures_pixelwise_fig3 \
#   python Totani_paper_check/figures/make_figures_from_config.py \
#     --config Totani_paper_check/figures/configs/fig3_pixelwise_global.json

run_logged figures_pixelwise_fig4 \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig4_pixelwise_global.json

run_logged figures_pixelwise_fig5 \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig5_pixelwise_global.json

run_logged figures_pixelwise_fig5_w_disk \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig5_pixelwise_global_w_disk.json

run_logged figures_pixelwise_fig6 \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig6_pixelwise_global.json

run_logged figures_pixelwise_fig6_w_disk \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig6_pixelwise_global_w_disk.json

run_logged figures_pixelwise_fig7 \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig7_pixelwise_global.json

run_logged figures_pixelwise_fig7_w_disk \
  python Totani_paper_check/figures/make_figures_from_config.py \
    --config Totani_paper_check/figures/configs/fig7_pixelwise_global_w_disk.json

# Regenerate the Totani/global overlay directory using the pixelwise fit.
run_logged plots_global_totani_overlay_pixelwise \
  python Totani_paper_check/figures/plot_global_totani_overlays.py \
    --global-tag pixelwise \
    --outdir Totani_paper_check/figures/plots_global_totani_overlay \
    --style "$PLOT_STYLE"

# Cheap spectral-only PPPC scan against the pixelwise rho^2 halo.
run_logged pppc_mass_scan_pixelwise_fig6 \
  python Totani_paper_check/run_pppc_mass_scan.py \
    --fit pixelwise_global_fig6 \
    --channels WW bb \
    --halo-reference pixelwise_global_rho2

# Regenerate the Fig. 6 WW/bb + p-wave comparison in the paper plotting style.
run_logged plot_global_fig6_ww_bb_pwave_pixelwise \
  python Totani_paper_check/figures/plot_global_fig6_ww_bb_pwave.py \
    --fit pixelwise_global_fig6 \
    --halo-reference pixelwise_global_rho2 \
    --output-dir Totani_paper_check/figures/plots_pppc_pixelwise_global_fig6 \
    --style "$PLOT_STYLE"

# Regenerate the Fig. 8 comparison directory with pixelwise global curves.
run_logged plots_fig8_global_comparison_pixelwise \
  python Totani_paper_check/figures/make_totani_fig8_global_comparison.py \
    --global-tag pixelwise \
    --outdir Totani_paper_check/figures/plots_fig8_global_comparison

# Regenerate the standalone velocity_overlay plots from the same pixelwise scan.
run_logged velocity_overlay_pixelwise \
  python Totani_reanalysis/velocity_dep/plot_velocity_limits_overlay.py \
    --fit pixelwise_global_fig6 \
    --channels bb WW \
    --out-dir Totani_reanalysis/results/velocity_overlay \
    --style "$PLOT_STYLE"

# New point-model spectral overlays with residual panels:
#   annihilation WW/bb on pixelwise rho^2;
#   decay WW/bb/tautau on pixelwise rho^1.
run_logged pixelwise_pppc_bestfit_points \
  python Totani_paper_check/figures/plot_pixelwise_pppc_bestfit_points.py \
    --style "$PLOT_STYLE" \
    --outdir Totani_paper_check/figures/plots_pppc_pixelwise_bestfit

echo
echo "Done. Logs are in: $LOG_DIR"
