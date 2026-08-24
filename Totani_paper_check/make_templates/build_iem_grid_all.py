#!/usr/bin/env python3
r"""
Driver that turns downloaded Ackermann et al. (2012) GALPROP grid archives
into gas/ICS diffuse templates, by repeatedly invoking the validated
converter, build_iem_grid_template.py (Stage 1 of the grid-implementation
plan for the interstellar-emission-model systematic).

Why this script exists: the converter is correct but its invocation is
manual and fiddly. Each of the 14 grid models must be extracted into its
own directory with the right layout, then the converter called with the
right --model-dir and --tag. Doing that by hand 14 times is error-prone,
and a wrong --tag silently mislabels a template -- the pipeline's
negative_keys / PSF-convolution selectors match by case-insensitive
substring (mcmc/mcmc_helper.py, ~lines 262-284), so a mistagged model does
not error, it just fits wrong. This driver removes the manual steps:

  1. locate each model's data -- an already-extracted directory, or a
     .tar to pull the three needed members from (handling the anchor's
     non-standard layout: its files sit directly in
     galprop_iem_grid/OUTPUT/, not in a per-model subdirectory);
  2. extract only the three combined HEALPix maps a model needs
     (pi0_decay_healpix_54_*, bremss_healpix_54_*,
     ics_isotropic_healpix_54_*) -- never the full 119-member archive,
     since each full tar is ~1.3 GB against ~150 MB for the three maps;
  3. invoke the converter with --model-dir/--tag fixed by the
     model-to-tag contract below (not invented here);
  4. skip a model whose gas_<tag>/ics_<tag> templates already exist and
     pass a validity check, so re-running the driver is cheap and safe;
  5. run the anchor's Stage 0 comparison (--compare-gas/--compare-ics
     against the existing baseline gas_dnde.fits/ics_dnde.fits) and fail
     loudly if the agreement regresses beyond a few percent -- a
     regression there means the converter or the anchor identification is
     wrong for every model built on it, not just the anchor;
  6. report a summary table: model, tag, status, and a physics sanity
     number per model (ICS total counts, and a central-concentration
     measure -- the ratio of ICS counts at 10 <= |b| < 25 deg to
     25 <= |b| <= 60 deg, both restricted to |l| <= 60 deg).

Machine-sharing notes: a pixelwise MCMC may be running across most cores
and downloads may be in progress elsewhere on the same disk, so this
driver is single-threaded, processes models strictly sequentially, and
never deletes or extracts from a .tar that might still be downloading.
Completeness is checked three ways before any extraction: the archive's
byte size against galprop_download's manifest.tsv (if available), a
short stability probe (size read twice, one second apart), and a
structural tar -tf read -- the same completeness test used elsewhere in
this project. A tar that fails any of these is left untouched and the
model is reported as not-ready, never as an error.

This script does not modify build_iem_grid_template.py and does not run
any fit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# Same sys.path setup as build_iem_grid_template.py, so totani_helpers and
# shared.paths resolve identically regardless of invocation cwd.
sys.path.insert(0, os.path.join(_REPO_ROOT, "shared"))
sys.path.insert(0, _REPO_ROOT)
from totani_helpers.totani_io import lonlat_grids, read_counts_and_ebounds  # noqa: E402

from shared.paths import REPO_ROOT as REPO_DIR  # path-fix 2026-06  # noqa: E402

CONVERTER_PATH = os.path.join(_HERE, "build_iem_grid_template.py")

DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
GRID_DIR = os.path.join(DATA_DIR, "galprop_iem_grid")
DEFAULT_OUTDIR = os.path.join(DATA_DIR, "processed", "templates")
DEFAULT_COUNTS = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
# galprop_download is a sibling toolkit, not part of this repo; its
# manifest.tsv is read only for the archive-size completeness check
# (fetching itself is out of scope here, per the task).
DEFAULT_MANIFEST = os.path.abspath(
    os.path.join(REPO_DIR, "..", "..", "galprop_download", "manifest.tsv")
)

ANCHOR_MODEL = "SL_Z6_R30_T150_C2"

# The model-to-tag contract. Fixed by the task; do not invent alternatives.
MODEL_TAGS = OrderedDict([
    ("SL_Z6_R30_T150_C2", "iem_sl_z6"),        # anchor
    ("SL_Z4_R30_T150_C2", "iem_sl_z4"),
    ("SL_Z8_R30_T150_C2", "iem_sl_z8"),
    ("SL_Z10_R30_T150_C2", "iem_sl_z10"),
    ("SS_Z6_R30_T150_C2", "iem_ss_z6"),
    ("SY_Z6_R30_T150_C2", "iem_sy_z6"),
    ("SO_Z6_R30_T150_C2", "iem_so_z6"),
    ("SS_Z4_R30_T150_C2", "iem_ss_z4"),
    ("SS_Z10_R30_T150_C2", "iem_ss_z10"),
    ("SY_Z4_R30_T150_C2", "iem_sy_z4"),
    ("SY_Z10_R30_T150_C2", "iem_sy_z10"),
    ("SL_Z6_R30_T100000_C2", "iem_sl_z6_tinf"),
    ("SL_Z6_R30_T150_C5", "iem_sl_z6_c5"),
    ("SL_Z6_R20_T150_C2", "iem_sl_z6_r20"),
])

# Exact prefixes the converter itself matches on (find_component() in
# build_iem_grid_template.py) -- reused here so this driver's notion of
# "the three needed files" can never drift from the converter's.
COMPONENT_PREFIXES = OrderedDict([
    ("pi0", "pi0_decay_healpix_54_"),
    ("bremss", "bremss_healpix_54_"),
    ("ics", "ics_isotropic_healpix_54_"),
])

# Forbidden substrings that would silently mis-bind a label in the
# pipeline's sign/PSF selectors (totani_bounds negative_keys, PSF
# selectors) -- defence in depth on top of the fixed contract above.
_FORBIDDEN_TAG_SUBSTRINGS = ("nfw", "fb_neg")


def parse_model_name(model_name):
    """Split e.g. 'SL_Z6_R30_T150_C2' into its S/Z/R/T/C tokens, or None
    if it does not match the grid's naming convention."""
    import re
    m = re.match(r"^(S[SLYO])_Z(\d+)_R(\d+)_T(\d+)_C(\d+)$", model_name)
    if not m:
        return None
    source, z, r, t, c = m.groups()
    return {"source": source, "z": int(z), "R": int(r), "T": int(t), "C": int(c)}


def dir_has_all_components(d):
    """True if directory d contains exactly one file per required prefix."""
    if not os.path.isdir(d):
        return False
    try:
        names = os.listdir(d)
    except OSError:
        return False
    for prefix in COMPONENT_PREFIXES.values():
        matches = [n for n in names if n.startswith(prefix) and n.endswith(".gz")]
        if len(matches) != 1:
            return False
    return True


def candidate_dirs_for_model(model_name, grid_dir):
    """Directories to check, in order, for an already-extracted model.
    Handles the anchor's non-standard layout (files directly in
    galprop_iem_grid/OUTPUT/, not a per-model subdirectory) and the
    historical short-form directories already on disk for the Z-only
    variants (SL_Z4/OUTPUT etc, dropping the R30_T150_C2 tokens shared
    with the anchor)."""
    if model_name == ANCHOR_MODEL:
        return [os.path.join(grid_dir, "OUTPUT")]
    cands = [
        os.path.join(grid_dir, model_name, "OUTPUT"),
        os.path.join(grid_dir, model_name),
    ]
    m = parse_model_name(model_name)
    if m:
        short = f"{m['source']}_Z{m['z']}"
        cands.append(os.path.join(grid_dir, short, "OUTPUT"))
        cands.append(os.path.join(grid_dir, short))
    return cands


def locate_extracted_dir(model_name, grid_dir):
    """Find an already-extracted directory for this model, checking only
    the known-layout candidates (candidate_dirs_for_model): the anchor's
    non-standard OUTPUT/ location, an exact model_name/OUTPUT directory
    (what this driver itself creates), or the historical short
    source_Z<n> form already on disk for the SL Z-variants.

    Deliberately does NOT fall back to scanning every directory under
    grid_dir for "anything with the right three filenames" -- models
    sharing a z_h (e.g. SS_Z6, SY_Z6, SO_Z6, SL_Z6_R30_T100000_C2 all
    have z=6, same as the anchor) would then risk matching each other's
    already-extracted directory on a same-z coincidence, which is exactly
    the silent-mislabelling failure mode this driver exists to prevent.
    A model with no directory matching one of the explicit candidates is
    reported as MISSING and left to the .tar path (or genuinely absent)
    rather than guessed at.
    """
    for c in candidate_dirs_for_model(model_name, grid_dir):
        if dir_has_all_components(c):
            return c
    return None


def load_manifest(path):
    """Read galprop_download's manifest.tsv (name<TAB>bytes) if present.
    Returns {} if the file is missing -- the manifest check is then simply
    skipped in favour of the stability probe and tar -tf."""
    sizes = {}
    if not path or not os.path.exists(path):
        return sizes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            name, size_str = parts
            try:
                sizes[name] = int(size_str)
            except ValueError:
                continue
    return sizes


def check_tar_completeness(tar_path, manifest):
    """Three-part completeness test, in increasing cost: manifest size
    match, a one-second stability probe (catches a tar still growing),
    and a structural tar -tf read. Never deletes or modifies the tar.
    Returns (ok: bool, message: str)."""
    name = os.path.basename(tar_path)
    size = os.path.getsize(tar_path)

    if manifest and name in manifest:
        expected = manifest[name]
        if size != expected:
            return False, (
                f"size {size} bytes != manifest expected {expected} bytes "
                "(likely still downloading or truncated)"
            )

    size_before = os.path.getsize(tar_path)
    time.sleep(1.0)
    size_after = os.path.getsize(tar_path)
    if size_before != size_after:
        return False, f"size changed within 1s ({size_before} -> {size_after}); still being written"

    try:
        subprocess.run(["tar", "-tf", tar_path], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)
        return False, f"tar -tf failed (corrupt/truncated archive): {stderr[:300]}"
    except FileNotFoundError:
        return False, "tar command not found"

    return True, "size and structure both check out"


def list_component_members(tar_path):
    """List the tar's members and pick out exactly one per required
    prefix, matching the converter's own find_component() strictness."""
    out = subprocess.run(
        ["tar", "-tf", tar_path], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    result = {}
    for key, prefix in COMPONENT_PREFIXES.items():
        matches = [m for m in out if os.path.basename(m).startswith(prefix) and m.endswith(".gz")]
        if len(matches) != 1:
            raise RuntimeError(
                f"{tar_path}: expected exactly one member starting with '{prefix}', found {matches}"
            )
        result[key] = matches[0]
    return result


def extract_components(tar_path, dest_root, members):
    """Extract only the three needed members (never the full 119-member
    archive) into dest_root, preserving the OUTPUT/ layout the tar itself
    uses. Returns the resulting OUTPUT directory."""
    os.makedirs(dest_root, exist_ok=True)
    subprocess.run(["tar", "-xf", tar_path, "-C", dest_root] + list(members.values()), check=True)
    out_dir = os.path.join(dest_root, "OUTPUT")
    if dir_has_all_components(out_dir):
        return out_dir
    if dir_has_all_components(dest_root):
        return dest_root
    raise RuntimeError(f"Extraction from {tar_path} did not produce the expected files under {out_dir}")


def template_paths(tag, outdir):
    return {
        "gas_dnde": os.path.join(outdir, f"gas_{tag}_dnde.fits"),
        "gas_mu": os.path.join(outdir, f"mu_gas_{tag}_counts.fits"),
        "ics_dnde": os.path.join(outdir, f"ics_{tag}_dnde.fits"),
        "ics_mu": os.path.join(outdir, f"mu_ics_{tag}_counts.fits"),
    }


def all_valid(paths, nE):
    """A template set counts as valid, reusable output if every one of
    the four files exists, opens, has the expected number of energy
    planes, and contains finite, non-all-zero data."""
    for p in paths.values():
        if not os.path.exists(p):
            return False
        try:
            with fits.open(p) as h:
                data = h[0].data
                if data is None or data.shape[0] != nE:
                    return False
                if not np.all(np.isfinite(data)):
                    return False
                if not np.any(data):
                    return False
        except Exception:
            return False
    return True


def physics_sanity(mu_ics_path):
    """ICS total counts (whole map, matching the converter's own printed
    'total counts' figure exactly) and a central-concentration ratio:
    ICS counts at 10 <= |b| < 25 deg over 25 <= |b| <= 60 deg, both
    restricted to |l| <= 60 deg -- a higher ratio means flux more
    concentrated toward the plane, as expected for a smaller CR halo
    height z_h."""
    with fits.open(mu_ics_path) as h:
        data = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    ics_total = float(np.nansum(data))

    w = WCS(hdr).celestial
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    lon2d, lat2d = lonlat_grids(w, ny, nx)
    inner = (np.abs(lon2d) <= 60) & (np.abs(lat2d) >= 10) & (np.abs(lat2d) < 25)
    outer = (np.abs(lon2d) <= 60) & (np.abs(lat2d) >= 25) & (np.abs(lat2d) <= 60)

    spatial = np.nansum(data, axis=0)
    inner_counts = float(np.nansum(spatial[inner]))
    outer_counts = float(np.nansum(spatial[outer]))
    ratio = inner_counts / outer_counts if outer_counts else float("nan")
    return ics_total, ratio


def parse_stage0_agreement(stdout_text):
    """Pull every 'median frac diff = +X.XXX%' figure out of the
    converter's --compare-gas/--compare-ics stdout, for the regression
    guard and for the report."""
    import re
    values = [float(x) for x in re.findall(r"median frac diff\s*=\s*([+-]?\d+\.\d+)%", stdout_text)]
    if not values:
        return None
    return {
        "n_bins": len(values),
        "min_pct": min(values),
        "max_pct": max(values),
        "max_abs_pct": max(abs(v) for v in values),
    }


def process_model(model_name, tag, args, ctx):
    result = {
        "model": model_name, "tag": tag, "status": "", "detail": "",
        "ics_total": None, "conc_ratio": None, "stage0": None,
    }

    assert tag.startswith("iem_"), f"tag '{tag}' does not follow the iem_ family convention"
    low = tag.lower()
    for bad in _FORBIDDEN_TAG_SUBSTRINGS:
        assert bad not in low, (
            f"tag '{tag}' contains forbidden substring '{bad}' -- would be silently "
            "mis-bound by the sign/PSF substring selectors"
        )

    paths = template_paths(tag, ctx["outdir"])
    if all_valid(paths, ctx["nE"]) and not args.force:
        result["status"] = "SKIP (valid templates already exist)"
        try:
            result["ics_total"], result["conc_ratio"] = physics_sanity(paths["ics_mu"])
        except Exception as e:  # pragma: no cover - reporting path only
            result["detail"] = f"sanity calc failed: {e}"
        return result

    model_dir = locate_extracted_dir(model_name, ctx["grid_dir"])
    created_dir = None

    if model_dir is None:
        tar_path = os.path.join(ctx["grid_dir"], f"{model_name}.tar")
        if not os.path.exists(tar_path):
            result["status"] = "MISSING (no extracted directory and no .tar found)"
            return result
        if args.dry_run:
            ok, msg = check_tar_completeness(tar_path, ctx["manifest"])
            result["status"] = "DRY-RUN: would extract from tar" if ok else f"NOT READY: {msg}"
            return result
        ok, msg = check_tar_completeness(tar_path, ctx["manifest"])
        if not ok:
            result["status"] = f"SKIP (tar not ready: {msg})"
            return result
        members = list_component_members(tar_path)
        dest_root = os.path.join(ctx["grid_dir"], model_name)
        pre_existing = os.path.isdir(dest_root)
        model_dir = extract_components(tar_path, dest_root, members)
        if not pre_existing:
            created_dir = dest_root
        result["detail"] += f"extracted three maps from {os.path.basename(tar_path)}; "
    else:
        result["detail"] += f"used already-extracted {model_dir}; "

    if args.dry_run:
        result["status"] = "DRY-RUN: would build"
        return result

    cmd = [sys.executable, CONVERTER_PATH, "--model-dir", model_dir, "--tag", tag]
    if model_name == ANCHOR_MODEL:
        cmd += ["--compare-gas", ctx["baseline_gas"], "--compare-ics", ctx["baseline_ics"]]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    if proc.returncode != 0:
        result["status"] = "FAILED (converter exited non-zero)"
        result["detail"] += (proc.stderr or "")[-2000:]
        return result

    result["status"] = "BUILT"

    if model_name == ANCHOR_MODEL:
        agreement = parse_stage0_agreement(proc.stdout)
        result["stage0"] = agreement
        if agreement is not None and agreement["max_abs_pct"] > args.stage0_threshold_pct:
            raise SystemExit(
                "STAGE 0 REGRESSION: anchor gas/ics agreement against the existing baseline "
                f"reached {agreement['max_abs_pct']:.3f}% median fractional difference in a bin, "
                f"exceeding the {args.stage0_threshold_pct:.1f}% guard (previously 0.00-0.03% ICS, "
                "0.04-0.08% gas). This means either the converter or the anchor identification is "
                "wrong, and every variant built on the same converter would be wrong the same way. "
                "Stopping before building anything else."
            )

    if created_dir is not None and not args.keep_tar_extract:
        shutil.rmtree(created_dir, ignore_errors=True)
        result["detail"] += f"removed extracted intermediate {created_dir}; "

    try:
        result["ics_total"], result["conc_ratio"] = physics_sanity(paths["ics_mu"])
    except Exception as e:  # pragma: no cover - reporting path only
        result["detail"] += f"sanity calc failed: {e}"

    return result


def print_summary_table(results):
    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    header = f"{'Model':<24}{'Tag':<20}{'Status':<40}{'ICS total counts':>18}{'Conc. ratio':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        ics = f"{r['ics_total']:.3e}" if r["ics_total"] is not None else "-"
        ratio = f"{r['conc_ratio']:.4f}" if r["conc_ratio"] is not None else "-"
        status = r["status"][:38]
        print(f"{r['model']:<24}{r['tag']:<20}{status:<40}{ics:>18}{ratio:>14}")
    print("-" * len(header))
    print("Conc. ratio = ICS counts (10<=|b|<25 deg) / ICS counts (25<=|b|<=60 deg), both |l|<=60 deg.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid-dir", default=GRID_DIR,
                     help=f"Directory holding per-model tars/extracted OUTPUT dirs (default: {GRID_DIR}).")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                     help=f"Template output directory, passed straight to the converter (default: {DEFAULT_OUTDIR}).")
    ap.add_argument("--counts", default=DEFAULT_COUNTS,
                     help="Counts cube used only to read off the number of analysis energy bins, "
                          "for the template validity check.")
    ap.add_argument("--baseline-gas", default=os.path.join(DEFAULT_OUTDIR, "gas_dnde.fits"),
                     help="Existing baseline gas_dnde.fits for the anchor's Stage 0 comparison.")
    ap.add_argument("--baseline-ics", default=os.path.join(DEFAULT_OUTDIR, "ics_dnde.fits"),
                     help="Existing baseline ics_dnde.fits for the anchor's Stage 0 comparison.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                     help="galprop_download's manifest.tsv, for the tar-completeness size check "
                          "(optional; the stability probe and tar -tf run regardless).")
    ap.add_argument("--only", nargs="+", default=None, metavar="MODEL",
                     help="Restrict to these models, given as either the model name "
                          "(e.g. SL_Z4_R30_T150_C2) or the tag (e.g. iem_sl_z4).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report what would happen for each model without extracting, "
                          "building, or deleting anything.")
    ap.add_argument("--force", action="store_true",
                     help="Rebuild even if valid templates already exist.")
    ap.add_argument("--keep-tar-extract", action="store_true",
                     help="Keep the per-model OUTPUT directory extracted from a tar after a "
                          "successful build (default: remove it, but only if this run created it "
                          "-- pre-existing extracted directories are never touched).")
    ap.add_argument("--stage0-threshold-pct", type=float, default=5.0,
                     help="Fail loudly if the anchor's median fractional agreement in any bin "
                          "exceeds this percentage (default: 5.0).")
    args = ap.parse_args()

    selection = None
    if args.only:
        selection = {s.strip().lower() for s in args.only}

    _, _, _, _, Ectr_mev, _ = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)

    ctx = {
        "grid_dir": args.grid_dir,
        "outdir": args.outdir,
        "manifest": load_manifest(args.manifest),
        "baseline_gas": args.baseline_gas,
        "baseline_ics": args.baseline_ics,
        "nE": nE,
    }
    if not ctx["manifest"]:
        print(f"Note: manifest not loaded from {args.manifest} -- relying on the stability "
              "probe and tar -tf alone for tar-completeness checks.")

    os.makedirs(ctx["outdir"], exist_ok=True)

    print(f"Grid dir:   {ctx['grid_dir']}")
    print(f"Output dir: {ctx['outdir']}")
    print(f"Analysis energy bins: {nE}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}"
          f"{'  (force rebuild)' if args.force else ''}")
    print()

    results = []
    for model_name, tag in MODEL_TAGS.items():
        if selection is not None:
            if model_name.lower() not in selection and tag.lower() not in selection:
                continue
        print(f"--- {model_name}  ->  {tag} " + "-" * max(1, 60 - len(model_name) - len(tag)))
        r = process_model(model_name, tag, args, ctx)
        print(f"  status: {r['status']}")
        if r.get("detail"):
            print(f"  detail: {r['detail']}")
        if r.get("stdout"):
            print(r["stdout"].rstrip())
        if r.get("stderr"):
            sys.stderr.write(r["stderr"])
        if r.get("stage0") is not None:
            s = r["stage0"]
            print(f"  Stage 0 agreement (anchor vs baseline): {s['n_bins']} bin comparisons, "
                  f"median frac diff range [{s['min_pct']:+.3f}%, {s['max_pct']:+.3f}%], "
                  f"max |median| = {s['max_abs_pct']:.3f}%")
        results.append(r)
        print()

    print_summary_table(results)


if __name__ == "__main__":
    main()
