#!/usr/bin/env bash
# Cellwise-mode entrypoint. Thin wrapper around plot_all_figures.sh --mode
# cellwise; all options (--style, --only, --no-overlays, ...) are forwarded.
#
#   bash plot_all_cellwise_figures.sh
#   bash plot_all_cellwise_figures.sh --only fig8
#   bash plot_all_cellwise_figures.sh --style conference
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/plot_all_figures.sh" --mode cellwise "$@"
