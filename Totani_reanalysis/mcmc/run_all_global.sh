#!/usr/bin/env bash
# Run all global pixel-level fits and produce the corresponding plots.
# Parallel to run_mcmc_from_config.py calls for the cellwise pipeline.
#
# Each config mirrors one of the Totani paper figures:
#   gce_global_fig2_3  ->  fig2_global + fig3_global  (full ROI, with/without disk)
#   gce_global_fig4    ->  fig4_global  (no NFW, fb_pos + fb_neg)
#   gce_global_fig5    ->  fig5_global  (NFW rho2.5, fb_pos + fb_neg, disk cut)
#   gce_global_fig6    ->  fig6_global  (NFW rho2,   fb_pos + fb_neg, disk cut)
#   gce_global_fig7    ->  fig7_global  (NFW rho1,   fb_pos + fb_neg, disk cut)
#
# Usage (with fermi env already active):
#   bash run_all_global.sh
#
# To run a single config only:
#   bash run_all_global.sh fig2_3
#   bash run_all_global.sh fig4
#   ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- helper ------------------------------------------------------------------
run_config() {
    local cfg="$1"
    echo ""
    echo "========================================================"
    echo "  Running: configs/${cfg}.json"
    echo "========================================================"
    python run_global_from_config.py --config "configs/${cfg}.json"
}

# ---- select which configs to run --------------------------------------------
if [[ $# -gt 0 ]]; then
    # Explicit subset passed on command line, e.g. "fig2_3 fig4"
    CONFIGS=("$@")
    # Prefix each bare name with gce_global_ if not already present
    RESOLVED=()
    for c in "${CONFIGS[@]}"; do
        if [[ "$c" != gce_global_* ]]; then
            RESOLVED+=("gce_global_${c}")
        else
            RESOLVED+=("$c")
        fi
    done
else
    # Default: run everything
    RESOLVED=(
        gce_global_fig2_3
        gce_global_fig4
        gce_global_fig5
        gce_global_fig6
        gce_global_fig7
    )
fi

# ---- run --------------------------------------------------------------------
FAILED=()
for cfg in "${RESOLVED[@]}"; do
    if run_config "$cfg"; then
        echo "  [OK] $cfg"
    else
        echo "  [FAILED] $cfg"
        FAILED+=("$cfg")
    fi
done

# ---- summary ----------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Global fit run complete"
echo "========================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  All configs succeeded."
else
    echo "  FAILED configs:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
    exit 1
fi
