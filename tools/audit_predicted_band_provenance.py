#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tools/audit_predicted_band_provenance.py — operator-facing predicted_band audit.

Per operator NON-NEGOTIABLE 2026-05-17 (META-FIX Catalog #324): every
``predicted_band`` declared in a ``.omx/operator_authorize_recipes/*.yaml``
must attest to whether the underlying Tier-C density evidence was
measured pre-training (random-init) or post-training. Empirical bug
class anchor — C6 IBPS 22× miss (call_id ``fc-01KRW353MJJ9A6QW8H99QWZEMH``,
2026-05-17) — proved that random-init Tier-C density produces
predicted_bands that EMPIRICALLY MISS the actual smoke score by orders
of magnitude. Sister #835's Assumption-Adversary verbatim warned;
#836 empirically falsified.

Usage::

    # Default audit (scans all substrate recipes; prints summary)
    .venv/bin/python tools/audit_predicted_band_provenance.py

    # Write JSON report:
    .venv/bin/python tools/audit_predicted_band_provenance.py \\
        --report-out .omx/state/predicted_band_audit_$(date -u +%Y%m%dT%H%M%SZ).json

    # CI-friendly: exit non-zero on any FAIL verdict
    .venv/bin/python tools/audit_predicted_band_provenance.py --strict

    # Custom recipe glob
    .venv/bin/python tools/audit_predicted_band_provenance.py \\
        --recipe-glob '.omx/operator_authorize_recipes/substrate_z*.yaml'

Exit codes:
  0 — all recipes clean (no FAIL verdicts)
  1 — at least one recipe is FAIL AND --strict is set
  2 — CLI error (bad args; missing canonical helper)

Sister of:
  * Catalog #323 ``tools/audit_provenance_compliance.py`` — same operator-
    audit pattern for the persisted-artifact surface; #324 is the
    predicted_band sub-surface.
  * Catalog #321 ``check_no_phantom_wyner_ziv_savings_from_research_sidecar``
    — same META-class at the Wyner-Ziv deliverability surface.

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow CLI
contract, human-readable summary, machine-readable JSON when requested.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from tac.optimization.tier_c_density_post_training_validator import (
        RecipeAuditVerdict,
        VALIDATION_STATUS_PHANTOM_RANDOM_INIT,
        VALIDATION_STATUS_VALIDATED_POST_TRAINING,
        VALIDATION_STATUS_PENDING_POST_TRAINING,
        VALIDATION_STATUS_OPERATOR_WAIVED,
        validate_recipe_predicted_band,
    )
except ImportError as exc:
    sys.stderr.write(
        f"[audit_predicted_band_provenance] FATAL: canonical helper "
        f"`tac.optimization.tier_c_density_post_training_validator` unavailable: {exc}\n"
        f"Per Catalog #324 + CLAUDE.md 'Bugs must be permanently fixed' fail-closed-on-import "
        f"discipline (sister Catalog #279), refusing to audit without the canonical contract.\n"
    )
    sys.exit(2)


SCHEMA_VERSION: str = "predicted_band_audit_v1_20260517"

DEFAULT_RECIPE_GLOB: str = ".omx/operator_authorize_recipes/*.yaml"


def _verdict_to_dict(verdict: RecipeAuditVerdict) -> dict:
    """Render an audit verdict as a JSON-serializable dict."""
    d = dataclasses.asdict(verdict)
    # Normalize tuple fields to lists for JSON
    d["blockers"] = list(verdict.blockers)
    if verdict.detected_predicted_band is not None:
        d["detected_predicted_band"] = list(verdict.detected_predicted_band)
    return d


def audit_recipes(recipe_glob: str = DEFAULT_RECIPE_GLOB) -> list[RecipeAuditVerdict]:
    """Audit every recipe matching the glob; return per-recipe verdicts."""
    paths = sorted(glob.glob(recipe_glob))
    return [validate_recipe_predicted_band(p) for p in paths]


def render_summary(verdicts: list[RecipeAuditVerdict]) -> str:
    """Render a human-readable summary table."""
    in_scope = [v for v in verdicts if v.has_predicted_band]
    out_of_scope = [v for v in verdicts if not v.has_predicted_band]
    pass_count = sum(1 for v in in_scope if v.is_valid)
    fail_count = sum(1 for v in in_scope if not v.is_valid)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Catalog #324 PREDICTED_BAND PROVENANCE AUDIT")
    lines.append("=" * 80)
    lines.append(f"Schema version: {SCHEMA_VERSION}")
    lines.append(f"Recipes scanned: {len(verdicts)}")
    lines.append(f"  In-scope (declared predicted_band): {len(in_scope)}")
    lines.append(f"  Out-of-scope (no predicted_band): {len(out_of_scope)}")
    lines.append(f"  PASS: {pass_count}")
    lines.append(f"  FAIL: {fail_count}")
    lines.append("")

    if in_scope:
        lines.append("Per-recipe verdicts:")
        lines.append("-" * 80)
        lines.append(f"{'Verdict':6} | {'Status':35} | Recipe")
        lines.append("-" * 80)
        for v in in_scope:
            flag = "PASS" if v.is_valid else "FAIL"
            name = Path(v.recipe_path).name
            lines.append(f"{flag:6} | {v.validation_status:35} | {name[:75]}")

    if fail_count > 0:
        lines.append("")
        lines.append("FAIL details (first blocker per recipe):")
        lines.append("-" * 80)
        for v in in_scope:
            if not v.is_valid:
                name = Path(v.recipe_path).name
                first_blocker = v.blockers[0] if v.blockers else "(no blocker recorded)"
                lines.append(f"- {name}:")
                # Wrap blocker text
                blocker_lines = []
                current = ""
                for word in first_blocker.split():
                    if len(current) + len(word) + 1 < 75:
                        current = (current + " " + word).strip()
                    else:
                        blocker_lines.append(current)
                        current = word
                if current:
                    blocker_lines.append(current)
                for bl in blocker_lines:
                    lines.append(f"    {bl}")

    lines.append("")
    lines.append("CLAUDE.md non-negotiables honored:")
    lines.append("  * Catalog #324: predicted_band post-training Tier-C validation")
    lines.append("  * Catalog #229: premise verification before edit")
    lines.append("  * Catalog #287: empirical-claim-tag")
    lines.append("  * Catalog #303: cargo-cult audit per assumption")
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit substrate recipes for predicted_band provenance per Catalog #324.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--recipe-glob",
        default=DEFAULT_RECIPE_GLOB,
        help=f"Glob pattern for recipe files (default: {DEFAULT_RECIPE_GLOB!r})",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="If set, write JSON report to this path (in addition to printing summary).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero (rc=1) if any in-scope recipe has FAIL verdict.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary output (still writes JSON if requested).",
    )

    args = parser.parse_args()

    verdicts = audit_recipes(recipe_glob=args.recipe_glob)
    in_scope = [v for v in verdicts if v.has_predicted_band]
    fail_count = sum(1 for v in in_scope if not v.is_valid)

    if not args.quiet:
        print(render_summary(verdicts))

    if args.report_out:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "recipe_glob": args.recipe_glob,
            "recipes_scanned": len(verdicts),
            "in_scope_count": len(in_scope),
            "out_of_scope_count": sum(1 for v in verdicts if not v.has_predicted_band),
            "pass_count": sum(1 for v in in_scope if v.is_valid),
            "fail_count": fail_count,
            "per_recipe_verdicts": [_verdict_to_dict(v) for v in verdicts],
            "anchor_note": (
                "Catalog #324 was landed in response to the C6 IBPS 22× miss "
                "(call_id fc-01KRW353MJJ9A6QW8H99QWZEMH, 2026-05-17). Sister "
                "#835's Assumption-Adversary verbatim warned that "
                "random-init Tier-C density provides only an UPPER bound on "
                "disconfirmation; #836 empirically falsified the prediction. "
                "Catalog #324 is the structural extinction at the recipe-emit surface."
            ),
        }
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True))
        if not args.quiet:
            print(f"\nJSON report written to: {args.report_out}")

    if args.strict and fail_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
