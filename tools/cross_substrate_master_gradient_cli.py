#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""META-LIFT-1 cross-substrate master-gradient analyzer CLI.

Per the 11th standing directive ORDER discipline: this is the operator-facing
sister of :mod:`tac.cross_substrate_master_gradient_analyzer`. Loads
canonical master-gradient anchors from
``.omx/state/master_gradient_anchors.jsonl`` (one row per substrate
archive), invokes the canonical analyzer, and emits ranked cross-substrate
byte-saving opportunities (human-readable summary or JSON).

Usage:
    .venv/bin/python tools/cross_substrate_master_gradient_cli.py
    .venv/bin/python tools/cross_substrate_master_gradient_cli.py --top-n 8
    .venv/bin/python tools/cross_substrate_master_gradient_cli.py --per-axis seg
    .venv/bin/python tools/cross_substrate_master_gradient_cli.py --json

Exit codes:
    0  CLEAN — analysis emitted (stdout)
    1  NO_MASTER_GRADIENT_ANCHORS — no authoritative anchors found
    2  CLI error (invalid arguments / unrecognized flag value)

Per Catalog #341 routing markers: this CLI's output is OBSERVABILITY-ONLY
by construction. Promotion of any opportunity to a contest score signal
REQUIRES paired-CUDA empirical anchor per CLAUDE.md "Submission auth eval
— BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Sister of:
  - ``tools/extract_master_gradient.py`` (canonical extractor — produces
    the per-substrate anchors this CLI consumes)
  - ``tools/list_canonical_equations.py`` (canonical equations registry)
  - ``tools/cathedral_autopilot_autonomous_loop.py`` (downstream ranker
    consumer via the sister cathedral consumer)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.cross_substrate_master_gradient_analyzer import (  # noqa: E402
    VALID_AXIS_LABELS,
    analyze_cross_substrate_master_gradients,
    append_analysis_locked,
)
from tac.master_gradient import (  # noqa: E402
    is_authoritative_contest_axis_anchor,
    is_usable_planning_anchor,
    load_anchors_lenient,
)


def _load_substrate_inputs(
    ledger_path: Path | None = None,
    *,
    include_non_authoritative: bool = False,
) -> list[dict]:
    """Load per-substrate master-gradient anchors via canonical API.

    Per Catalog #230 sister-disjoint: this function READS the canonical
    ledger via :func:`tac.master_gradient.load_anchors_lenient` and does
    NOT duplicate producer logic. Only AGGREGATE-tensor-kind anchors are
    selected (per :func:`tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`
    canonical contract).

    Per Catalog #327 contest-axis custody: by default only usable planning
    anchors are included. Diagnostic/advisory rows may be usable planning
    signal, but they still carry ``is_authoritative=False`` unless they have
    actual contest-CPU/contest-CUDA custody.

    Returns a list of dicts in the canonical
    :func:`tac.cross_substrate_master_gradient_analyzer.analyze_cross_substrate_master_gradients`
    input shape.
    """
    anchors = load_anchors_lenient(ledger_path)

    # Group by archive_sha256 — keep the most-recent authoritative aggregate
    # anchor per substrate (one row per substrate enters the analysis).
    per_archive: dict[str, dict] = {}
    for anchor in anchors:
        # AGGREGATE tensor kind = (N_bytes, 3) shape per
        # master_gradient_consumers.AGGREGATE_GRADIENT_TENSOR_KIND.
        # Schema heuristic: the canonical ledger row has 'gradient_array_path'
        # pointing to an .npy with shape (N_bytes, 3); we infer aggregate
        # from the absence of 'per_pair' marker in the path OR explicit kind.
        tensor_kind = anchor.get("tensor_kind", "aggregate")
        if tensor_kind != "aggregate":
            # Default canonical ledger has aggregate-tensor entries without
            # explicit kind; per_pair anchors set tensor_kind="per_pair".
            continue

        usable_planning_anchor = is_usable_planning_anchor(anchor)
        if not include_non_authoritative and not usable_planning_anchor:
            continue

        archive_sha = anchor.get("archive_sha256", "")
        if not archive_sha:
            continue

        existing = per_archive.get(archive_sha)
        if existing is not None:
            # Keep most-recent measurement.
            if anchor.get("measurement_utc", "") > existing.get("measurement_utc", ""):
                per_archive[archive_sha] = anchor
        else:
            per_archive[archive_sha] = anchor

    substrate_inputs: list[dict] = []
    for archive_sha, anchor in per_archive.items():
        gradient_path = Path(anchor["gradient_array_path"])
        if not gradient_path.is_absolute():
            gradient_path = REPO_ROOT / gradient_path

        if not gradient_path.exists():
            # Skip silently — the sidecar may have been gc'd; the analyzer
            # cannot project without the actual gradient tensor.
            continue

        substrate_inputs.append(
            {
                "gradient_array": lambda gp=gradient_path: np.load(gp),
                "archive_sha256": archive_sha,
                "measurement_axis": anchor.get("measurement_axis", "[unknown]"),
                "measurement_hardware": anchor.get("measurement_hardware", "unknown"),
                "measurement_call_id": anchor.get("measurement_call_id", "unknown"),
                "is_authoritative": is_authoritative_contest_axis_anchor(anchor),
            }
        )

    return substrate_inputs


def _format_human_summary(analysis_dict: dict) -> str:
    """Format the analysis as a human-readable summary."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("META-LIFT-1 CROSS-SUBSTRATE MASTER-GRADIENT ANALYSIS")
    lines.append("=" * 78)
    lines.append(f"analysis_id:                 {analysis_dict['analysis_id']}")
    lines.append(f"measurement_utc:             {analysis_dict['measurement_utc']}")
    lines.append(f"axis_tag:                    {analysis_dict['axis_tag']}")
    lines.append(f"evidence_grade:              {analysis_dict['evidence_grade']}")
    lines.append(f"score_claim:                 {analysis_dict['score_claim']}")
    lines.append(f"promotable:                  {analysis_dict['promotable']}")
    lines.append(f"canonical_equation_id:       {analysis_dict['canonical_equation_id']}")
    lines.append(f"canonical_equation_status:   {analysis_dict['canonical_equation_status']}")
    lines.append(f"target_axes:                 {analysis_dict['target_axes']}")
    lines.append("")
    lines.append(
        f"cauchy_schwarz_aggregate_upper_bound: {analysis_dict['cauchy_schwarz_aggregate_upper_bound']:.6f}"
    )
    lines.append("  (unit-perturbation bound; SUM_i ||∇S_i||_2 across substrates)")
    lines.append("")
    lines.append(f"substrate_rows ({len(analysis_dict['substrate_rows'])}):")
    for row in analysis_dict["substrate_rows"]:
        sha_short = row["archive_sha256"][:12]
        auth_marker = "*" if row["is_authoritative"] else " "
        lines.append(
            f"  {auth_marker} {sha_short}  axis={row['measurement_axis']:<24} hw={row['measurement_hardware']:<40}"
        )
        lines.append(
            f"    n_bytes={row['n_bytes']:>8d}  agg_L2={row['aggregate_gradient_l2_norm']:.6f}  "
            f"agg_per_byte_leverage={row['aggregate_per_byte_leverage']:.6f}"
        )
    lines.append("")
    lines.append(f"ranked_opportunities ({len(analysis_dict['ranked_opportunities'])}):")
    for opp in analysis_dict["ranked_opportunities"]:
        sha_short = opp["archive_sha256"][:12]
        auth_marker = "*" if opp["is_authoritative"] else " "
        lines.append(
            f"  #{opp['rank']:>3d} {auth_marker} {sha_short} axis={opp['axis']:<5}  "
            f"per_byte_leverage={opp['per_byte_leverage']:.6f}  "
            f"CS_unit_bound={opp['cauchy_schwarz_unit_perturbation_upper_bound']:.6f}"
        )
    lines.append("")
    lines.append("Per Catalog #341 + CLAUDE.md 'Apples-to-apples evidence discipline':")
    lines.append("  EVERY opportunity is observability-only. Promotion REQUIRES paired-CUDA")
    lines.append("  empirical anchor per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA'.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_substrate_master_gradient_cli",
        description=(
            "META-LIFT-1 cross-substrate master-gradient analyzer CLI. "
            "Loads per-substrate authoritative anchors from "
            ".omx/state/master_gradient_anchors.jsonl, runs the canonical "
            "Cauchy-Schwarz + per-axis Taylor analysis, and emits ranked "
            "byte-saving opportunities (human or JSON)."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=16,
        help="Number of ranked opportunities to emit (default: 16).",
    )
    parser.add_argument(
        "--per-axis",
        choices=sorted(VALID_AXIS_LABELS),
        default=None,
        help="Restrict ranking to a single axis (seg / pose / rate); default: all 3.",
    )
    parser.add_argument(
        "--top-k-per-axis",
        type=int,
        default=64,
        help="K for the per-substrate top-K byte ranking (default: 64).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human summary.",
    )
    parser.add_argument(
        "--persist-to-ledger",
        action="store_true",
        help=(
            "Append the analysis row to "
            ".omx/state/cross_substrate_master_gradient_analyses.jsonl "
            "(default: do not persist; CLI is observability-only)."
        ),
    )
    parser.add_argument(
        "--include-non-authoritative",
        action="store_true",
        help=(
            "Include rows that fail the usable-planning-anchor filter. "
            "Diagnostic/advisory opportunities still carry "
            "is_authoritative=False unless they have contest-axis custody."
        ),
    )
    parser.add_argument(
        "--ledger-path",
        type=str,
        default=None,
        help=(
            "Override path to .omx/state/master_gradient_anchors.jsonl "
            "(default: canonical repo location)."
        ),
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse calls sys.exit(2) for invalid args; preserve canonical rc.
        return int(exc.code) if exc.code is not None else 2

    if args.top_n < 1:
        sys.stderr.write("ERROR: --top-n must be >= 1\n")
        return 2
    if args.top_k_per_axis < 1:
        sys.stderr.write("ERROR: --top-k-per-axis must be >= 1\n")
        return 2

    ledger_path = Path(args.ledger_path) if args.ledger_path else None

    substrate_inputs = _load_substrate_inputs(
        ledger_path,
        include_non_authoritative=args.include_non_authoritative,
    )

    if not substrate_inputs:
        sys.stderr.write(
            "ERROR: no aggregate master-gradient anchors found in canonical "
            "ledger at .omx/state/master_gradient_anchors.jsonl "
            "(use tools/extract_master_gradient.py to populate the ledger).\n"
        )
        return 1

    target_axes = (args.per_axis,) if args.per_axis else ("seg", "pose", "rate")

    try:
        analysis = analyze_cross_substrate_master_gradients(
            substrate_inputs,
            target_axes=target_axes,
            top_k_per_axis=args.top_k_per_axis,
            top_n_opportunities=args.top_n,
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: analyzer raised: {exc}\n")
        return 2

    if args.persist_to_ledger:
        append_analysis_locked(analysis)

    analysis_dict = analysis.as_dict()
    if args.json:
        sys.stdout.write(json.dumps(analysis_dict, sort_keys=True, indent=2, allow_nan=False))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_human_summary(analysis_dict))
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
