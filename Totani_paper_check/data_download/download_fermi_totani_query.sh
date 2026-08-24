#!/usr/bin/env bash
set -euo pipefail

# Manifest-driven downloader for the public Fermi-LAT query products used by
# process_totani_full.sh. Run from anywhere:
#
#   bash Totani_paper_check/data_download/download_fermi_totani_query.sh all
#   bash Totani_paper_check/data_download/download_fermi_totani_query.sh 0_0 0_45
#
# Data are written under ${REPO_DIR:-<repo root>}/fermi_data/totani/<query>/.

if [[ -z "${REPO_DIR:-}" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

BASE_URL="${BASE_URL:-https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/fermi_data/totani}"

usage() {
  cat <<'EOF'
Usage:
  download_fermi_totani_query.sh all
  download_fermi_totani_query.sh QUERY [QUERY ...]

Known queries:
  315_0   L2603110708053F2F87D150   PH00
  45_0    L2603110706603F2F87D174   PH00
  0_45    L2603110711113F2F87D111   PH00 PH01 PH02
  0_-45   L2603110707403F2F87D153   PH00
  0_0     L2603110706243F2F87D111   PH00 PH01 PH02 PH03 PH04 PH05 PH06

Environment overrides:
  REPO_DIR=/path/to/this/repo
  DATA_ROOT=/path/to/output/totani
  BASE_URL=https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries
EOF
}

query_tag() {
  case "$1" in
    315_0) echo "L2603110708053F2F87D150" ;;
    45_0) echo "L2603110706603F2F87D174" ;;
    0_45) echo "L2603110711113F2F87D111" ;;
    0_-45) echo "L2603110707403F2F87D153" ;;
    0_0) echo "L2603110706243F2F87D111" ;;
    *) return 1 ;;
  esac
}

query_photon_parts() {
  case "$1" in
    315_0|45_0|0_-45) echo "PH00" ;;
    0_45) echo "PH00 PH01 PH02" ;;
    0_0) echo "PH00 PH01 PH02 PH03 PH04 PH05 PH06" ;;
    *) return 1 ;;
  esac
}

download_file() {
  local tag="$1"
  local suffix="$2"
  local file="${tag}_${suffix}.fits"
  echo "Downloading ${file}"
  curl -L -C - -o "${file}" "${BASE_URL}/${file}"
}

if [[ "$#" -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$1" == "all" ]]; then
  queries=(315_0 45_0 0_45 0_-45 0_0)
else
  queries=("$@")
fi

for query in "${queries[@]}"; do
  tag="$(query_tag "$query")" || {
    echo "ERROR: unknown query '${query}'" >&2
    usage >&2
    exit 1
  }
  parts="$(query_photon_parts "$query")"
  out_dir="${DATA_ROOT}/${query}"

  mkdir -p "${out_dir}"
  cd "${out_dir}"

  echo "===================================="
  echo "Fermi LAT Totani query: ${query}"
  echo "Tag: ${tag}"
  echo "Destination: ${out_dir}"
  echo "===================================="

  for part in ${parts}; do
    download_file "${tag}" "${part}"
  done

  ls -lh ./*.fits
done

echo
echo "Download complete. Next:"
echo "  bash Totani_paper_check/data_download/process_totani_full.sh"
