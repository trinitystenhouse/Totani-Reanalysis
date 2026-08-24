#!/usr/bin/env python3
"""Compile annihilation-only and scattering best-fit summaries into CSV tables.

This is intentionally a post-processing script: it parses the ``summary.txt``
files already written by ``scan_tension_resolution.py`` and
``scan_deconvolved_scattering_fit.py``. It does not rerun any physics scan.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List


HERE = Path(__file__).resolve().parent
from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06
PROFILE_ORDER = {
    "global_rho2.5": 0,
    "global_rho2": 1,
    "global_rho1": 2,
    "rho2.5": 0,
    "rho2": 1,
    "rho1": 2,
}
CHANNEL_ORDER = {"WW": 0, "bb": 1, "tautau": 2}


def _float(text: str | None) -> float:
    if text is None:
        return math.nan
    raw = str(text).strip()
    if raw.lower() in {"", "nan", "none"}:
        return math.nan
    return float(raw)


def _match(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _section(text: str, title: str) -> str:
    idx = text.find(title)
    if idx < 0:
        return ""
    rest = text[idx + len(title) :]
    next_idx = rest.find("\n\n")
    return rest if next_idx < 0 else rest[:next_idx]


def _find_summaries(root: Path, prefix: str | None = None) -> Iterable[Path]:
    if not root.exists():
        return []
    paths = sorted(root.glob("**/summary.txt"))
    if prefix is None:
        return paths
    return [p for p in paths if p.parent.name.startswith(prefix)]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def _sort_key(row: Dict[str, object]) -> tuple:
    profile = str(row.get("profile"))
    channel = str(row.get("channel"))
    return (
        PROFILE_ORDER.get(profile, 99),
        CHANNEL_ORDER.get(channel, 99),
        str(row.get("operator", "")),
    )


def parse_annihilation_summary(path: Path) -> Dict[str, object]:
    text = path.read_text()
    spec = _section(text, "Spectral best fit:")
    return {
        "profile": _match(text, r"^halo profile\s*:\s*(.+)$"),
        "channel": _match(text, r"^channel\s*:\s*(.+)$"),
        "m_ann_GeV": _float(_match(spec, r"m_ann\s*:\s*([^\s]+)\s*GeV")),
        "chi2": _float(_match(spec, r"chi2\s*:\s*([^\s]+)")),
        "chi2_per_ndof": _float(_match(spec, r"chi2 / ndof\s*:\s*([^\s]+)")),
        "sigmav_cm3_s": _float(_match(spec, r"<sigma v>\s*:\s*([^\s]+)\s*cm\^3/s")),
        "dsph_limit_cm3_s": _float(_match(spec, r"<sigma v>_dSph.*:\s*([^\s]+)\s*cm\^3/s")),
        "tension_x_dsph": _float(_match(spec, r"tension factor\s*:\s*([^\sx]+)x")),
        "path": _display_path(path),
    }


def parse_scattering_summary(path: Path) -> Dict[str, object]:
    text = path.read_text()
    pure = _section(text, "--- Pure annihilation")
    scat = _section(text, "--- Best scattering fit")
    operator_raw = _match(text, r"^operator\s*:\s*(.+)$") or ""
    op_match = re.match(r"([^\s(]+)(?:\s*\(([^)]+)\))?", operator_raw)
    operator = op_match.group(1) if op_match else operator_raw
    dm_type = op_match.group(2) if op_match and op_match.group(2) else ""

    pure_chi2 = _float(_match(pure, r"chi2\s*:\s*([^\s]+)"))
    fit_chi2 = _float(_match(scat, r"fit chi2\s*:\s*([^\s]+)"))
    return {
        "profile": _match(text, r"^halo profile\s*:\s*(.+)$"),
        "channel": _match(text, r"^channel\s*:\s*(.+)$"),
        "operator": operator,
        "dm_type": dm_type,
        "best_fit_type": _match(text, r"^best-fit type\s*:\s*(.+)$"),
        "best_m_ann_GeV": _float(_match(scat, r"best m_ann\s*:\s*([^\s]+)\s*GeV")),
        "best_m_scat_GeV": _float(_match(scat, r"best m_scat\s*:\s*([^\s]+)\s*GeV")),
        "best_Lambda_GeV": _float(_match(scat, r"best Lambda\s*:\s*([^\s]+)\s*GeV")),
        "fit_chi2": fit_chi2,
        "pure_chi2": pure_chi2,
        "delta_chi2_improvement": pure_chi2 - fit_chi2 if math.isfinite(pure_chi2) and math.isfinite(fit_chi2) else math.nan,
        "diagnostic_deconv_chi2": _float(_match(scat, r"diagnostic deconv chi2\s*:\s*([^\s]+)")),
        "tau_max": _float(_match(scat, r"tau_max\s*:\s*([^\s]+)")),
        "best_sigmav_cm3_s": _float(_match(scat, r"<sigma v>\s*:\s*([^\s]+)\s*cm\^3/s")),
        "best_dsph_limit_cm3_s": _float(_match(scat, r"dSph limit\s*:\s*([^\s]+)\s*cm\^3/s")),
        "best_tension_x_dsph": _float(_match(scat, r"tension\s*:\s*([^\sx]+)x\s*dSph")),
        "scan_points_kept": int(_float(_match(text, r"^scan points kept\s*:\s*([^\s]+)$"))),
        "path": _display_path(path),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 5) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}g}"
    return str(value)


def _write_markdown(path: Path, ann_rows: List[Dict[str, object]], scat_rows: List[Dict[str, object]]) -> None:
    lines = ["# Best-fit Results Summary", ""]

    if ann_rows:
        lines += [
            "## Annihilation-only global fits",
            "",
            "| NFW profile | channel | best m_ann [GeV] | chi2/ndof | <sigma v> [cm^3/s] | dSph tension |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in ann_rows:
            lines.append(
                f"| {r['profile']} | {r['channel']} | {_fmt(r['m_ann_GeV'])} | "
                f"{_fmt(r['chi2_per_ndof'], 4)} | {_fmt(r['sigmav_cm3_s'], 4)} | "
                f"{_fmt(r['tension_x_dsph'], 4)}x |"
            )
        lines.append("")

    if scat_rows:
        lines += [
            "## Scattering + annihilation fits",
            "",
            "| profile | channel | operator | m_ann [GeV] | m_scat [GeV] | Lambda [GeV] | fit chi2 | Delta chi2 improvement | tau_max | dSph tension |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in scat_rows:
            lines.append(
                f"| {r['profile']} | {r['channel']} | {r['operator']} | "
                f"{_fmt(r['best_m_ann_GeV'])} | {_fmt(r['best_m_scat_GeV'])} | "
                f"{_fmt(r['best_Lambda_GeV'])} | {_fmt(r['fit_chi2'])} | "
                f"{_fmt(r['delta_chi2_improvement'])} | {_fmt(r['tau_max'])} | "
                f"{_fmt(r['best_tension_x_dsph'], 4)}x |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--annihilation-dir",
        default=str(HERE / "results" / "tension_scan_global"),
        help="Directory containing annihilation_fit_* summary folders.",
    )
    p.add_argument(
        "--scattering-dir",
        default=str(HERE / "results" / "deconvolved_scattering_fit"),
        help="Directory containing rho2_* scattering summary folders.",
    )
    p.add_argument(
        "--scattering-prefix",
        default="rho2_",
        help="Only compile scattering summaries whose parent directory starts with this prefix.",
    )
    p.add_argument(
        "--output-dir",
        default=str(HERE / "results" / "compiled_best_fits"),
    )
    args = p.parse_args()

    ann_rows = [
        parse_annihilation_summary(path)
        for path in _find_summaries(Path(args.annihilation_dir), "annihilation_fit_")
    ]
    scat_rows = [
        parse_scattering_summary(path)
        for path in _find_summaries(Path(args.scattering_dir), args.scattering_prefix)
    ]

    ann_rows.sort(key=_sort_key)
    scat_rows.sort(key=_sort_key)

    outdir = Path(args.output_dir)
    _write_csv(
        outdir / "annihilation_best_fit_masses_global.csv",
        ann_rows,
        [
            "profile",
            "channel",
            "m_ann_GeV",
            "chi2",
            "chi2_per_ndof",
            "sigmav_cm3_s",
            "dsph_limit_cm3_s",
            "tension_x_dsph",
            "path",
        ],
    )
    _write_csv(
        outdir / "scattering_annihilation_best_fits_rho2.csv",
        scat_rows,
        [
            "profile",
            "channel",
            "operator",
            "dm_type",
            "best_fit_type",
            "best_m_ann_GeV",
            "best_m_scat_GeV",
            "best_Lambda_GeV",
            "fit_chi2",
            "pure_chi2",
            "delta_chi2_improvement",
            "diagnostic_deconv_chi2",
            "tau_max",
            "best_sigmav_cm3_s",
            "best_dsph_limit_cm3_s",
            "best_tension_x_dsph",
            "scan_points_kept",
            "path",
        ],
    )
    _write_markdown(outdir / "best_fit_results_summary.md", ann_rows, scat_rows)

    print(f"Wrote {len(ann_rows)} annihilation rows and {len(scat_rows)} scattering rows to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
