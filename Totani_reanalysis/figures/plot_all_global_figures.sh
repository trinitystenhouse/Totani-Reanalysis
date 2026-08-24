#!/usr/bin/env bash
# Global-mode entrypoint. Thin wrapper around plot_all_figures.sh --mode
# global; all options (--style, --only, --no-overlays, ...) are forwarded.
#
#   bash plot_all_global_figures.sh
#   bash plot_all_global_figures.sh --only fig5_global
#   bash plot_all_global_figures.sh --no-overlays --style conference
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/plot_all_figures.sh" --mode global "$@"
