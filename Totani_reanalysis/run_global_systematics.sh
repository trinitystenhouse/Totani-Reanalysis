#!/usr/bin/env bash
set -euo pipefail

# Run deterministic global-fit systematics for the rho2 halo spectrum.
#
# Controls:
#   ONLY_VARIANT=name1,name2   run only selected variant config basenames
#   SKIP_MOCKS=1               do not rebuild null/recovery mock counts
#   SKIP_FITS=1                do not run the global fits
#   SKIP_PLOTS=1               do not regenerate summary plots
#   SKIP_COLLECT=1             do not copy outputs into results/global-systematics

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

LOG_DIR="$REPO_ROOT/Totani_paper_check/logs_global_systematics"
RESULTS_DIR="$REPO_ROOT/Totani_paper_check/results/global-systematics"
mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

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

selected_variant() {
  local v="$1"
  if [ -z "${ONLY_VARIANT:-}" ]; then
    return 0
  fi
  case ",$ONLY_VARIANT," in
    *",$v,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_logged write_global_systematics_configs \
  python Totani_paper_check/mcmc/write_global_systematics_configs.py

if [ "${SKIP_MOCKS:-0}" != "1" ]; then
  run_logged build_global_mock_counts \
    python Totani_paper_check/mcmc/build_global_mock_counts.py
fi

if [ "${SKIP_FITS:-0}" != "1" ]; then
  shopt -s nullglob
  for cfg in Totani_paper_check/mcmc/configs/systematics/*.json; do
    name="$(basename "$cfg" .json)"
    if [ "$name" = "manifest" ]; then
      continue
    fi
    if ! selected_variant "$name"; then
      echo "Skipping unselected variant: $name"
      continue
    fi
    run_logged "global_systematics_${name}" \
      python Totani_paper_check/mcmc/run_global_from_config.py \
        --config "$cfg"
  done
fi

if [ "${SKIP_PLOTS:-0}" != "1" ]; then
  run_logged plot_global_systematics \
    python Totani_paper_check/figures/plot_global_systematics.py
  run_logged plot_global_systematics_dashboard \
    python Totani_paper_check/figures/plot_global_systematics_dashboard.py
fi

if [ "${SKIP_COLLECT:-0}" != "1" ]; then
  mkdir -p "$RESULTS_DIR"
  copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
      rm -rf "$dst"
      cp -R "$src" "$dst"
    else
      echo "Warning: missing optional output, not collecting: $src"
    fi
  }
  copy_if_exists Totani_paper_check/mcmc/configs/systematics "$RESULTS_DIR/configs"
  copy_if_exists Totani_paper_check/figures/plots_global_systematics_summary "$RESULTS_DIR/summary_plots"
  copy_if_exists Totani_paper_check/figures/plots_global_systematics_dashboard "$RESULTS_DIR/dashboard_plots"
  copy_if_exists Totani_paper_check/mcmc/mock_counts "$RESULTS_DIR/mock_counts"
  echo
  echo "Collected systematics outputs in: $RESULTS_DIR"
fi

echo
echo "Done. Logs are in: $LOG_DIR"
