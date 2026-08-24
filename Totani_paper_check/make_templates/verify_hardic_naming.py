#!/usr/bin/env python3
r"""
verify_hardic_naming.py
========================
Stand-alone safety test for the hardic_<tag> label family (hard
inverse-Compton injected-population templates).  Run this before wiring a
new hardic label into any fit config.

Why this exists
----------------
The pipeline's ``negative_keys`` (sign-bound selector, ``totani_bounds``,
``Totani_reanalysis/mcmc/mcmc_helper.py:262-284``) and ``psf_labels``
(PSF-convolution selector, ``_selected_for_psf``,
``Totani_reanalysis/mcmc/run_global_from_config.py:158-164``) both match
by **case-insensitive substring**, not exact string equality.  A hardic
label that happens to *contain* ``"nfw"`` or ``"fb_neg"`` would silently
become sign-unconstrained (able to absorb flux like the halo, defeating
the whole point of the test); a label containing ``"gas"``, ``"ics"``,
``"loopA"`` or ``"loopB"`` would be silently swept into the *wrong*
PSF-convolution group.  This is the single most dangerous failure mode for
a new template label in this pipeline, so it gets its own explicit,
re-runnable test rather than a one-off manual check.

This script re-implements the exact matching logic of both call sites
(rather than importing them, so it has no dependency on the rest of the
pipeline being importable) and checks every label in
``build_hardic_template.py``'s tag family against every matcher used
anywhere in the pipeline.

Usage
-----
    python verify_hardic_naming.py            # exit 0 if all checks pass
    python verify_hardic_naming.py --verbose  # print every check, not just failures
"""

from __future__ import annotations

import argparse
import sys

# The hardic tag family this task's build_hardic_template.py /
# write_global_systematics_configs.py::HARDIC_TAGS produces.
HARDIC_TAGS = ["burst_s", "burst_m", "burst_l", "cont", "adv", "cmbonly_burst_m"]
HARDIC_LABELS = [f"hardic_{tag}" for tag in HARDIC_TAGS]

# Matchers actually used by the pipeline (verified directly from the source):
NEGATIVE_KEYS = ["nfw", "fb_neg"]  # mcmc_helper.py:totani_bounds default;
                                    # gce_global_base.json; run_global_fit.py
BASELINE_PSF_SELECTORS = ["nfw", "gas", "ics", "loopA", "loopB"]  # gce_global_base.json
HARDIC_PSF_SELECTORS = BASELINE_PSF_SELECTORS + ["hardic"]  # what the
                                                              # cr_injection variants use
EXACT_INIT_KEYS = ["ps", "gas", "ics", "iso", "isotropic", "point_sources", "pointsources"]
# _build_global_init / totani_init_from_mu_counts match these keys by
# EXACT equality (not substring) -- included here as a negative control:
# a hardic label should never accidentally equal one of these.
HALO_COLUMN_KEY = "nfw"  # plot_global_systematics.py / propagate_systematics.py
                          # halo-column parsers key on this substring.


def _substring_match(label: str, selectors) -> list:
    """Case-insensitive substring match, exactly mirroring
    totani_bounds / _selected_for_psf's `sel_l in lab_l` logic."""
    lab_l = label.lower()
    return [s for s in selectors if str(s).lower() in lab_l]


def check_label(label: str, verbose: bool) -> list:
    """Returns a list of failure strings (empty if the label is safe)."""
    failures = []

    neg_hits = _substring_match(label, NEGATIVE_KEYS)
    if neg_hits:
        failures.append(
            f"'{label}' matches negative_keys {neg_hits} -> would become "
            f"SIGN-UNCONSTRAINED (hardic must be positive-only by construction)"
        )
    elif verbose:
        print(f"  [OK] '{label}' does not match negative_keys {NEGATIVE_KEYS}")

    baseline_psf_hits = _substring_match(label, BASELINE_PSF_SELECTORS)
    if baseline_psf_hits:
        failures.append(
            f"'{label}' matches the BASELINE psf_labels selectors {baseline_psf_hits} "
            f"even before 'hardic' is added -- would be auto-PSF-selected regardless "
            f"of intent (confusing at minimum)"
        )
    elif verbose:
        print(f"  [OK] '{label}' does not match baseline psf_labels {BASELINE_PSF_SELECTORS}")

    hardic_psf_hits = _substring_match(label, HARDIC_PSF_SELECTORS)
    if "hardic" not in [h.lower() for h in hardic_psf_hits]:
        failures.append(
            f"'{label}' is NOT matched by the 'hardic' PSF selector -- the "
            f"PSF-convolved template mu_{label}_psf_counts.fits would never be "
            f"auto-built by _build_psf_templates"
        )
    elif verbose:
        print(f"  [OK] '{label}' is matched by the 'hardic' PSF selector")

    if label.lower() in [k.lower() for k in EXACT_INIT_KEYS]:
        failures.append(
            f"'{label}' EXACTLY equals an init key in {EXACT_INIT_KEYS} -- would start "
            f"at theta=1 instead of theta=0 (would start from the wrong initial value)"
        )
    elif verbose:
        print(f"  [OK] '{label}' does not exactly equal any exact-match init key")

    halo_col_hits = _substring_match(label, [HALO_COLUMN_KEY])
    if halo_col_hits:
        failures.append(
            f"'{label}' matches the halo-column parser key '{HALO_COLUMN_KEY}' -- would "
            f"be misidentified as the halo column by plot_global_systematics.py / "
            f"propagate_systematics.py"
        )
    elif verbose:
        print(f"  [OK] '{label}' does not match the halo-column parser key '{HALO_COLUMN_KEY}'")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="Print every check, not just failures.")
    ap.add_argument("--label", action="append", default=None,
                    help="Check additional label(s) beyond the built-in hardic tag family "
                         "(repeatable).")
    args = ap.parse_args()

    labels = list(HARDIC_LABELS)
    if args.label:
        labels += list(args.label)

    print(f"Checking {len(labels)} label(s) against {len(NEGATIVE_KEYS)} negative_keys, "
          f"{len(BASELINE_PSF_SELECTORS)} baseline psf_labels, and the halo-column key...")
    print()

    n_fail = 0
    for label in labels:
        print(f"Label: {label}")
        failures = check_label(label, args.verbose)
        if failures:
            n_fail += len(failures)
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  all checks passed")
        print()

    if n_fail:
        print(f"{n_fail} FAILURE(S) -- fix the naming before wiring this label into a fit config.")
        return 1
    print(f"All {len(labels)} label(s) are safe against every substring matcher in the pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
