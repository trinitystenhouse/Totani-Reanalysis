#!/usr/bin/env bash
# Generate all Totani-reproduction figure products for one fit family
# (cellwise, global Poisson or pixelwise) and, by default for the reanalysis
# modes, the Totani overlays.
#
# Usage:
#   bash plot_all_figures.sh --mode cellwise
#   bash plot_all_figures.sh --mode global
#   bash plot_all_figures.sh --mode pixelwise
#   bash plot_all_figures.sh --mode global --no-overlays
#   bash plot_all_figures.sh --mode pixelwise --style conference
#   bash plot_all_figures.sh --mode cellwise --only fig4
#   bash plot_all_figures.sh --mode global --only fig5_global
#   bash plot_all_figures.sh --mode pixelwise --only fig8
#   bash plot_all_figures.sh --mode cellwise --style conference --out-dir /tmp/x
#
# Any style other than 'paper' writes to a conference_figs/ subdirectory of the
# mode's normal output root, so a slide run can never overwrite the paper
# products. --out-dir overrides that choice.
#
# --mode selects the fit family (required):
#   cellwise  -> Totani_paper_check/mcmc/mcmc_results_*, paper-check configs,
#                outputs under Totani_paper_check/figures/plots_*
#   global     -> mcmc/fit_results/quick_global,   configs/figN_global.json,
#                 outputs under figures/global_fit/plots_global_*
#   pixelwise  -> mcmc/fit_results/pixelwise_mcmc,  configs/figN_pixelwise_global.json,
#                 outputs under figures/pixelwise_fit/plots_pixelwise_global_*
#
# --only <name> runs a single target. Valid names are the config basenames for
# the chosen mode (e.g. fig5_global or fig5_pixelwise_global), plus the special
# targets: fig8 fig9 fig10 fig11 overlay.

set -euo pipefail

PLOT_STYLE="${PLOT_STYLE:-paper}"
WITH_OVERLAYS=1
ONLY=""
MODE=""
OUT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?--mode requires a value}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --style)
            PLOT_STYLE="${2:?--style requires a value}"
            shift 2
            ;;
        --style=*)
            PLOT_STYLE="${1#*=}"
            shift
            ;;
        --only)
            ONLY="${2:?--only requires a value}"
            shift 2
            ;;
        --only=*)
            ONLY="${1#*=}"
            shift
            ;;
        --out-dir)
            OUT_DIR_OVERRIDE="${2:?--out-dir requires a value}"
            shift 2
            ;;
        --out-dir=*)
            OUT_DIR_OVERRIDE="${1#*=}"
            shift
            ;;
        --with-overlays)
            WITH_OVERLAYS=1
            shift
            ;;
        --no-overlays|--skip-overlays)
            WITH_OVERLAYS=0
            shift
            ;;
        -h|--help)
            sed -n '2,26p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "--mode {cellwise|global|pixelwise} is required. See --help." >&2
    exit 2
fi

# should_run <target-name> -> succeeds if that target should be plotted.
should_run() {
    [[ -z "$ONLY" || "$ONLY" == "$1" ]]
}

first_existing_dir() {
    for path in "$@"; do
        if [[ -d "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    printf '%s\n' "$1"
}

export PLOT_STYLE="$PLOT_STYLE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REANALYSIS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${REANALYSIS_DIR}/.." && pwd)"
TEMPLATE_DIR="${REPO_DIR}/fermi_data/totani/processed/templates"

# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------
case "$MODE" in
    cellwise)
        MODE_LABEL="cellwise"
        MCMC_DIR="${REPO_DIR}/Totani_paper_check/mcmc"
        RESULT_PREFIX="mcmc_results"
        OUT_DIR="${REPO_DIR}/Totani_paper_check/figures"
        OUT_PREFIX="plots"
        OVERLAY_TAG=""
        FIT_MODE="mcmc"
        HALO_LABEL="nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"
        FIG11_EXTRA=()
        configs=(
            "${REPO_DIR}/Totani_paper_check/figures/configs/fig2.json"
            "${REPO_DIR}/Totani_paper_check/figures/configs/fig3.json"
            "${REPO_DIR}/Totani_paper_check/figures/configs/fig4.json"
            "${REPO_DIR}/Totani_paper_check/figures/configs/fig5.json"
            "${REPO_DIR}/Totani_paper_check/figures/configs/fig6.json"
            "${REPO_DIR}/Totani_paper_check/figures/configs/fig7.json"
        )
        labels=(
            "Figure 2 cellwise (spectral components, all latitudes)"
            "Figure 3 cellwise (spectral components, |b|>=10 deg)"
            "Figure 4 cellwise (spectral components with fb_pos/fb_neg)"
            "Figure 5 cellwise (NFW rho2.5)"
            "Figure 6 cellwise (NFW rho2)"
            "Figure 7 cellwise (NFW rho1)"
        )
        ;;
    global)
        MODE_LABEL="global"
        MCMC_DIR="${REANALYSIS_DIR}/mcmc/fit_results/quick_global"
        RESULT_PREFIX="global_fit_results"
        OUT_DIR="${SCRIPT_DIR}/global_fit"
        OUT_PREFIX="plots_global"
        OVERLAY_TAG="global"
        FIT_MODE="global"
        HALO_LABEL="nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"
        FIG11_EXTRA=()
        configs=(
            configs/fig2_global.json
            configs/fig3_global.json
            configs/fig4_global.json
            configs/fig4_global_w_disk.json
            configs/fig5_global.json
            configs/fig5_global_w_disk.json
            configs/fig6_global.json
            configs/fig6_global_w_disk.json
            configs/fig7_global.json
            configs/fig7_global_w_disk.json
        )
        labels=(
            "Figure 2 global (spectral components, all latitudes)"
            "Figure 3 global (spectral components, |b|>=10 deg)"
            "Figure 4 global (spectral components with fb_pos/fb_neg)"
            "Figure 4 global with disk (spectral components with fb_pos/fb_neg)"
            "Figure 5 global (NFW rho2.5)"
            "Figure 5 global with disk (NFW rho2.5)"
            "Figure 6 global (NFW rho2)"
            "Figure 6 global with disk (NFW rho2)"
            "Figure 7 global (NFW rho1)"
            "Figure 7 global with disk (NFW rho1)"
        )
        ;;
    pixelwise)
        MODE_LABEL="pixelwise"
        MCMC_DIR="${REANALYSIS_DIR}/mcmc/fit_results/pixelwise_mcmc"
        RESULT_PREFIX="pixelwise_mcmc_results"
        OUT_DIR="${SCRIPT_DIR}/pixelwise_fit"
        OUT_PREFIX="plots_pixelwise_global"
        OVERLAY_TAG="pixelwise"
        FIT_MODE="pixelwise"
        HALO_LABEL="nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno_psf"
        FIG11_EXTRA=(
            --ext-mask "${TEMPLATE_DIR}/mask_sources_combined_edep.fits"
            --halo-template-label "${HALO_LABEL}"
            --non-halo-labels gas_psf ics_psf iso ps loopA_psf loopB_psf fb_pos fb_neg
            --template-labels gas_psf ics_psf iso ps loopA_psf loopB_psf fb_pos fb_neg
        )
        configs=(
            configs/fig2_pixelwise_global.json
            configs/fig3_pixelwise_global.json
            configs/fig4_pixelwise_global.json
            configs/fig4_pixelwise_global_w_disk.json
            configs/fig5_pixelwise_global.json
            configs/fig5_pixelwise_global_w_disk.json
            configs/fig6_pixelwise_global.json
            configs/fig6_pixelwise_global_w_disk.json
            configs/fig7_pixelwise_global.json
            configs/fig7_pixelwise_global_w_disk.json
        )
        labels=(
            "Figure 2 pixelwise (spectral components, all latitudes)"
            "Figure 3 pixelwise (spectral components, |b|>=10 deg)"
            "Figure 4 pixelwise (spectral components with fb_pos/fb_neg)"
            "Figure 4 pixelwise with disk (spectral components with fb_pos/fb_neg)"
            "Figure 5 pixelwise (NFW rho2.5)"
            "Figure 5 pixelwise with disk (NFW rho2.5)"
            "Figure 6 pixelwise (NFW rho2)"
            "Figure 6 pixelwise with disk (NFW rho2)"
            "Figure 7 pixelwise (NFW rho1)"
            "Figure 7 pixelwise with disk (NFW rho1)"
        )
        ;;
    *)
        echo "Unknown --mode: $MODE (expected 'cellwise', 'global' or 'pixelwise')." >&2
        exit 2
        ;;
esac

FIG23_DIR="$(first_existing_dir \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig2_3" \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig2_3_10deg")"
FIG4_DIR="$(first_existing_dir \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig4" \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig4_10deg")"
FIG5_DIR="$(first_existing_dir \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig5" \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig5_10deg")"
FIG6_DIR="$(first_existing_dir \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig6" \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig6_10deg")"
FIG7_DIR="$(first_existing_dir \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig7" \
    "${MCMC_DIR}/${RESULT_PREFIX}_fig7_10deg")"

# ---------------------------------------------------------------------------
# Style-aware output root
# ---------------------------------------------------------------------------
# OUT_DIR above is the PAPER location for this mode. The slide styles render
# the same target names, so without this a conference run silently replaces the
# paper products with transparent dark-slide versions -- the figures are only
# distinguishable by their alpha channel, so the damage is easy to miss and
# expensive to undo. Send every non-paper style to a sibling conference_figs/
# instead, and let --out-dir override both.
if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
    OUT_DIR="$OUT_DIR_OVERRIDE"
elif [[ "$PLOT_STYLE" != "paper" ]]; then
    OUT_DIR="${OUT_DIR}/conference_figs"
fi
mkdir -p "$OUT_DIR"

cd "$SCRIPT_DIR"

echo "=============================================="
echo "Generating all ${MODE_LABEL} fit figures"
echo "Plot style: ${PLOT_STYLE}"
echo "Output dir: ${OUT_DIR}"
echo "Totani overlays: $([[ "$WITH_OVERLAYS" -eq 1 ]] && echo yes || echo no)"
echo "=============================================="
echo ""

n_configs="${#configs[@]}"

# Map each config to its --only target name (basename without directory/extension).
only_matched=0
for i in "${!configs[@]}"; do
    key="$(basename "${configs[$i]}" .json)"
    should_run "$key" || continue
    only_matched=1
    echo "[$((i + 1))/${n_configs}] ${labels[$i]}..."
    # --outdir is essential, not cosmetic: without it the driver falls back to
    # the config's own 'outdir' key, which points at the PAPER directory, so a
    # conference run wrote slide figures over the paper products regardless of
    # OUT_DIR. Every other target in this script already takes an explicit
    # output path; these were the exception.
    # --outdir-root keeps each config's own leaf name (plots_fig2 /
    # plots_global_fig2 / plots_pixelwise_global_fig2 -- it differs per mode)
    # and only re-bases it under OUT_DIR.
    python make_figures_from_config.py --config "${configs[$i]}" \
        --outdir-root "${OUT_DIR}"
done

if should_run fig8; then
    only_matched=1
    echo "[fig8] Figure 8 ${MODE_LABEL} (halo E^2dN/dE at Galactic poles)..."
    python "${SCRIPT_DIR}/make_totani_fig8.py" \
        --mcmc-dir-gce   "${FIG23_DIR}" \
        --mcmc-dir-rho25 "${FIG5_DIR}" \
        --mcmc-dir-rho2  "${FIG6_DIR}" \
        --mcmc-dir-rho1  "${FIG7_DIR}" \
        --outdir         "${OUT_DIR}/${OUT_PREFIX}_fig8"
fi

if should_run fig9; then
    only_matched=1
    echo "[fig9] Figure 9 ${MODE_LABEL} (delta lnL vs energy)..."
    python "${SCRIPT_DIR}/make_totani_fig9_likelihood.py" \
        --fit-mode        "${FIT_MODE}" \
        --mcmc-dir-nohalo "${FIG4_DIR}" \
        --mcmc-dir-rho25  "${FIG5_DIR}" \
        --mcmc-dir-rho2   "${FIG6_DIR}" \
        --mcmc-dir-rho1   "${FIG7_DIR}" \
        --outdir          "${OUT_DIR}/${OUT_PREFIX}_fig9"
fi

if should_run fig10; then
    only_matched=1
    echo "[fig10] Figure 10 ${MODE_LABEL} (halo angular profile at 21 GeV)..."
    python "${REANALYSIS_DIR}/plot_angular_profile.py" \
        --out-dir "${OUT_DIR}/${OUT_PREFIX}_fig10"
fi

if should_run fig11; then
    only_matched=1
    echo "[fig11] Figures 11-13 ${MODE_LABEL} (residual maps, NFW-rho2 fit)..."
    python "${SCRIPT_DIR}/make_fig11.py" \
        --mcmc-dir-nohalo "${FIG4_DIR}" \
        --mcmc-dir-halo   "${FIG6_DIR}" \
        --halo-label      "${HALO_LABEL}" \
        ${FIG11_EXTRA[@]+"${FIG11_EXTRA[@]}"} \
        --roi-lon  60.0 \
        --roi-lat  60.0 \
        --disk-cut 10.0 \
        --outdir   "${OUT_DIR}/${OUT_PREFIX}_figs11_12_13"
fi

if should_run overlay; then
    only_matched=1
    if [[ "$MODE" == "cellwise" ]]; then
        echo "[overlay] Skipping: Totani overlay comparison is defined for global/pixelwise modes."
    elif [[ "$WITH_OVERLAYS" -eq 1 ]]; then
        echo "[overlay] ${MODE_LABEL} fit vs Totani overlays..."
        python plot_global_totani_overlays.py \
            --global-tag "${OVERLAY_TAG}" \
            --style "${PLOT_STYLE}" \
            --outdir "${OUT_DIR}/${OUT_PREFIX}_totani_overlay"
    fi
fi

if [[ -n "$ONLY" && "$only_matched" -eq 0 ]]; then
    echo "Unknown --only target: $ONLY" >&2
    echo "Run with --help to see valid target names." >&2
    exit 2
fi

echo ""
echo "=============================================="
echo "All ${MODE_LABEL} figures generated successfully."
echo "Outputs: ${OUT_DIR}"
echo "=============================================="
