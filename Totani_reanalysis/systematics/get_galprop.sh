#!/usr/bin/env bash
# Download the Ackermann et al. (2012), ApJ 750, 3 interstellar-emission model grid.
# Supplementary material: GALDEF files + pre-calculated GALPROP output, one .tar per model.
#
# Naming: S<source>_Z<z_h>_R<R_h>_T<T_S>_C<E(B-V) cut>
#   source: SS = SNR, SL = Lorimer pulsars, SY = Yusifov pulsars, SO = OB stars
#   T100000 = optically thin spin temperature
#
# SL_Z6_R30_T150_C2 is our analysis baseline
# (galdef_54_Lorimer_z6kpc_R30kpc_Ts150K_EBV2mag) and is the validation anchor.
#
# Safe to re-run: files whose local size already matches the server's
# Content-Length are skipped; partial files resume from where they stopped.
#
# Env overrides:
#   DATA_DIR=...    destination (default ./galprop_iem_grid)
#   GALDEF_URL=...  exact URL of the GALDEF bundle, if the candidates below miss
#   VERIFY=0        skip the tar-integrity pass (it re-reads every archive)

set -uo pipefail
# Resolve repo root (two levels above this script: Totani_paper_check/data_download -> repo root)
if [[ -z "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

DATA_DIR="${REPO_DIR}/fermi_data/totani/galprop_iem_grid"
BASE_URL="https://galprop.stanford.edu/PaperIISuppMaterial"
GALDEF_URL="${GALDEF_URL:-}"
VERIFY="${VERIFY:-1}"

MODELS=(
    # anchor -- our existing baseline, download first and validate against it
    SL_Z6_R30_T150_C2
    # propagation halo height, baseline source distribution
    SL_Z4_R30_T150_C2
    SL_Z8_R30_T150_C2
    SL_Z10_R30_T150_C2
    # cosmic-ray source distribution, baseline halo height
    SS_Z6_R30_T150_C2
    SY_Z6_R30_T150_C2
    SO_Z6_R30_T150_C2
    # z_h x source-distribution corners (gives the interaction term)
    SS_Z4_R30_T150_C2
    SS_Z10_R30_T150_C2
    SY_Z4_R30_T150_C2
    SY_Z10_R30_T150_C2
    # gas-side axes at baseline (expected marginal at |b| >= 10 deg)
    SL_Z6_R30_T100000_C2
    SL_Z6_R30_T150_C5
    SL_Z6_R20_T150_C2
)

# The page offers "GALDEF files for the runs <download all>" but the target
# filename is not known from the listing. Try the plausible ones; if all miss,
# copy the real link address off the page and pass it as GALDEF_URL=...
GALDEF_CANDIDATES=(
    "GALDEF.tar"
    "galdef.tar"
    "GALDEFs.tar"
    "galdefs.tar"
    "GALDEF_files.tar"
    "galdef_files.tar"
)

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}" || exit 1

echo "===================================="
echo "Downloading GALPROP IEM model grid"
echo "Models: ${#MODELS[@]}"
echo "Destination: ${DATA_DIR}"
echo "===================================="
echo ""

FAILED=""
SKIPPED=0
FETCHED=0

local_size() {
    local file=$1
    [ -f "${file}" ] || { echo 0; return; }
    # stat is BSD on macOS, GNU on Linux
    stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}" 2>/dev/null || echo 0
}

# Is this archive already complete? Tested by asking tar to list it, which needs
# no network and is format-aware: a truncated download fails, and so does an
# HTML error page saved under a .tar name. Preferred over comparing sizes
# against Content-Length, which this server may not report.
is_complete() {
    local file=$1
    [ -s "${file}" ] || return 1
    tar -tf "${file}" >/dev/null 2>&1
}

# Helper function
download() {
    local file=$1
    local url="${2:-${BASE_URL}/${file}}"
    local have

    if is_complete "${file}"; then
        printf "Skipping   %-28s (already complete, %s)\n" "${file}" "$(du -h "${file}" | cut -f1)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    have=$(local_size "${file}")
    if [ "${have}" -gt 0 ] 2>/dev/null; then
        echo "Resuming   ${file} (have $(du -h "${file}" | cut -f1)) ..."
    else
        echo "Downloading ${file} ..."
    fi

    if curl -L -C - --fail --retry 3 --retry-delay 5 -o "${file}" "${url}"; then
        :
    elif [ "${have}" -gt 0 ] 2>/dev/null; then
        # Some servers refuse ranged requests (curl exits 33/36); start over.
        echo "  resume refused or failed, restarting ${file} from scratch ..."
        rm -f "${file}"
        curl -L --fail --retry 3 --retry-delay 5 -o "${file}" "${url}" || true
    fi

    if is_complete "${file}"; then
        FETCHED=$((FETCHED + 1))
        echo ""
        return 0
    fi
    echo "  FAILED: ${file}"
    FAILED="${FAILED} ${file}"
    echo ""
    return 1
}

for m in "${MODELS[@]}"; do
    download "${m}.tar"
done

# ---- GALDEF files -------------------------------------------------------------
# Small, and needed to confirm the SS/SY/SO source-distribution decoding rather
# than trusting it.
echo "------------------------------------"
echo "GALDEF files"
echo "------------------------------------"
if [ -n "${GALDEF_URL}" ]; then
    download "$(basename "${GALDEF_URL}")" "${GALDEF_URL}"
else
    got_galdef=0
    for cand in "${GALDEF_CANDIDATES[@]}"; do
        if is_complete "${cand}"; then
            printf "Skipping   %-28s (already complete)\n" "${cand}"
            SKIPPED=$((SKIPPED + 1))
            got_galdef=1
            break
        fi
        # Probe with a 1-byte ranged GET rather than HEAD -- more widely supported.
        if curl -sL --fail --max-time 30 -r 0-0 -o /dev/null "${BASE_URL}/${cand}" 2>/dev/null; then
            if download "${cand}"; then
                got_galdef=1
                break
            fi
        fi
    done
    if [ "${got_galdef}" -eq 0 ]; then
        echo "  Could not guess the GALDEF bundle filename."
        echo "  Copy the link address from \"GALDEF files for the runs <download all>\""
        echo "  on ${BASE_URL}/ and re-run as:"
        echo "      GALDEF_URL='<paste>' bash $0"
        echo "  (The galdef may also ship inside each model tar -- check with"
        echo "   tar -tf ${MODELS[0]}.tar | grep -i galdef)"
        FAILED="${FAILED} GALDEF"
    fi
fi
echo ""

# ---- verification -------------------------------------------------------------
if [ "${VERIFY}" = "1" ]; then
    echo "===================================="
    echo "Verifying archives"
    echo "===================================="
    for f in *.tar; do
        [ -e "$f" ] || continue
        if tar -tf "$f" >/dev/null 2>&1; then
            printf "  OK    %-28s %s\n" "$f" "$(du -h "$f" | cut -f1)"
        else
            printf "  BAD   %-28s (truncated, or an HTML error page)\n" "$f"
            FAILED="${FAILED} ${f}"
        fi
    done
    echo ""
else
    echo "(verification skipped: VERIFY=0)"
    echo ""
fi

echo "===================================="
echo "Fetched: ${FETCHED}   Already complete: ${SKIPPED}"
if [ -z "${FAILED// /}" ]; then
    echo "All downloads complete and verified."
else
    echo "PROBLEMS WITH:${FAILED}"
    echo "Re-run to resume -- completed files are skipped automatically."
fi
echo "Total: $(du -sh "${DATA_DIR}" | cut -f1)"
echo "===================================="