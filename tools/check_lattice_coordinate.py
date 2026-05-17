#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator CLI: query the canonical lattice-state ledger.

Per canonical Path 2 LATTICE OF CLASS-SHIFTS framework + operator binding
constraint 2026-05-16 verbatim *"Remember we need outside nerv-family too"*.

Usage examples
──────────────

    # Coverage summary (rule counts + outside-NeRV count + horizon distribution)
    tools/check_lattice_coordinate.py --list-coverage

    # Show all substrates targeting a specific rule
    tools/check_lattice_coordinate.py --rule rule_2_nullspace_split_pr95_paradigm

    # Show coordinate for a specific substrate
    tools/check_lattice_coordinate.py --substrate nscs01

    # Show all OUTSIDE-NeRV-family substrates (operator constraint enforcement)
    tools/check_lattice_coordinate.py --list-outside-nerv

    # Show currently UNCOVERED lattice rules
    tools/check_lattice_coordinate.py --list-uncovered-rules

    # Machine-readable JSON output
    tools/check_lattice_coordinate.py --list-coverage --json

Exit codes
──────────
    0  — query succeeded; no operator action required
    1  — query succeeded; operator-action-required signal (e.g. uncovered rule
         in scope; outside-NeRV count below recommended 3-4 threshold)
    2  — argument or runtime error

Sister of Catalog #245 ``tools/harvest_modal_calls.py`` + Catalog #313
``tools/check_predecessor_probe_outcome.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.lattice_state_ledger import (
    LATTICE_STATE_LEDGER_PATH,
    NERV_FAMILY_ARCHITECTURAL_CLASSES,
    VALID_LATTICE_RULES,
    compute_coverage_report,
    latest_node_state,
    load_nodes,
    query_by_architectural_class,
    query_by_rule,
    query_by_substrate,
    query_outside_nerv_family,
    query_uncovered_rules,
)


OUTSIDE_NERV_RECOMMENDED_MINIMUM = 3


def _render_text_coverage(report) -> str:
    lines: list[str] = []
    lines.append("LATTICE COORDINATE COVERAGE REPORT")
    lines.append("=" * 60)
    lines.append(f"Total registered nodes: {report.total_nodes}")
    lines.append("")
    lines.append("--- Per-rule counts ---")
    for rule in sorted(VALID_LATTICE_RULES):
        count = report.rule_counts.get(rule, 0)
        marker = " (UNCOVERED!)" if rule in report.uncovered_rules else ""
        lines.append(f"  {rule}: {count}{marker}")
    lines.append("")
    lines.append("--- Per-horizon counts ---")
    for horizon, count in sorted(report.horizon_counts.items()):
        lines.append(f"  {horizon}: {count}")
    lines.append("")
    lines.append("--- NeRV-family vs OUTSIDE-NeRV ---")
    lines.append(f"  NeRV-family: {report.nerv_family_count}")
    lines.append(f"  Outside-NeRV: {report.outside_nerv_count}")
    if report.outside_nerv_count < OUTSIDE_NERV_RECOMMENDED_MINIMUM:
        lines.append(
            f"  WARNING: outside-NeRV count ({report.outside_nerv_count}) is below "
            f"the operator-recommended minimum ({OUTSIDE_NERV_RECOMMENDED_MINIMUM})."
        )
    lines.append("")
    lines.append("--- Status distribution ---")
    lines.append(f"  In-flight (lifted; not yet dispatched): {report.in_flight_count}")
    lines.append(f"  Dispatched (evidence landed): {report.dispatched_count}")
    lines.append(f"  Deferred: {report.deferred_count}")
    lines.append("")
    if report.uncovered_rules:
        lines.append("UNCOVERED RULES (operator-action recommended):")
        for rule in report.uncovered_rules:
            lines.append(f"  - {rule}")
    else:
        lines.append("All canonical rules have at least one active coverage anchor.")
    return "\n".join(lines)


def _render_text_nodes(nodes: list[dict]) -> str:
    if not nodes:
        return "(no matching nodes)"
    lines: list[str] = []
    for n in nodes:
        nid = n.get("lattice_node_id", "?")
        rule = n.get("lattice_rule", "?")
        horizon = n.get("horizon_class", "?")
        arch = n.get("architectural_class", "?")
        status = n.get("status", "?")
        score = n.get("evidence_score")
        axis = n.get("evidence_score_axis")
        score_part = ""
        if score is not None:
            score_part = f" | score={score} [{axis}]" if axis else f" | score={score}"
        lines.append(
            f"  {nid}: rule={rule} | horizon={horizon} | arch={arch} | status={status}{score_part}"
        )
    return "\n".join(lines)


def _render_text_single_node(node: dict) -> str:
    lines: list[str] = []
    lines.append(f"lattice_node_id: {node.get('lattice_node_id', '?')}")
    lines.append(f"substrate: {node.get('substrate', '?')}")
    lines.append(f"lattice_rule: {node.get('lattice_rule', '?')}")
    lines.append(f"horizon_class: {node.get('horizon_class', '?')}")
    lines.append(f"architectural_class: {node.get('architectural_class', '?')}")
    lines.append(f"status: {node.get('status', '?')}")
    lines.append(
        f"paradigm_vs_implementation_classification: {node.get('paradigm_vs_implementation_classification', '?')}"
    )
    score = node.get("evidence_score")
    axis = node.get("evidence_score_axis")
    if score is not None:
        lines.append(f"evidence_score: {score} [{axis or 'unspecified axis'}]")
    if node.get("evidence_artifact_path"):
        lines.append(f"evidence_artifact_path: {node['evidence_artifact_path']}")
    if node.get("recipe_path"):
        lines.append(f"recipe_path: {node['recipe_path']}")
    if node.get("trainer_path"):
        lines.append(f"trainer_path: {node['trainer_path']}")
    if node.get("lane_id"):
        lines.append(f"lane_id: {node['lane_id']}")
    if node.get("notes"):
        lines.append(f"notes: {node['notes']}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the canonical lattice-state ledger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help="Override the default ledger path (default: .omx/state/lattice_state.jsonl).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--substrate",
        type=str,
        help="Show the current coordinate for a specific substrate (lattice_node_id OR substrate id).",
    )
    group.add_argument(
        "--rule",
        type=str,
        choices=sorted(VALID_LATTICE_RULES),
        help="Show all substrates targeting a specific lattice rule.",
    )
    group.add_argument(
        "--architectural-class",
        type=str,
        help="Show all substrates of a specific architectural class.",
    )
    group.add_argument(
        "--list-coverage",
        action="store_true",
        help="Show the lattice-rule coverage summary + outside-NeRV count.",
    )
    group.add_argument(
        "--list-uncovered-rules",
        action="store_true",
        help="Show canonical rules currently with no in-flight or dispatched coverage.",
    )
    group.add_argument(
        "--list-outside-nerv",
        action="store_true",
        help="Show all substrates NOT in the NeRV family (operator binding constraint surface).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path = args.ledger_path or LATTICE_STATE_LEDGER_PATH
    if not path.exists():
        msg = f"lattice-state ledger not found at {path}"
        if args.json:
            print(json.dumps({"error": msg, "exit_code": 2}))
        else:
            print(msg, file=sys.stderr)
        return 2

    if args.substrate:
        # Try by lattice_node_id first, then by substrate
        node = latest_node_state(args.substrate, path=path)
        if node is None:
            sub_rows = query_by_substrate(args.substrate, path=path)
            if sub_rows:
                node = sub_rows[-1]
        if node is None:
            msg = f"no lattice coordinate found for substrate {args.substrate!r}"
            if args.json:
                print(json.dumps({"error": msg, "exit_code": 1}))
            else:
                print(msg, file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(node, sort_keys=True, indent=2))
        else:
            print(_render_text_single_node(node))
        return 0

    if args.rule:
        nodes = query_by_rule(args.rule, path=path, latest_only=True)
        if args.json:
            print(json.dumps({"rule": args.rule, "nodes": nodes}, sort_keys=True, indent=2))
        else:
            print(f"Substrates targeting {args.rule}:")
            print(_render_text_nodes(nodes))
        return 0 if nodes else 1

    if args.architectural_class:
        nodes = query_by_architectural_class(args.architectural_class, path=path, latest_only=True)
        if args.json:
            print(json.dumps({"architectural_class": args.architectural_class, "nodes": nodes}, sort_keys=True, indent=2))
        else:
            print(f"Substrates of class {args.architectural_class}:")
            print(_render_text_nodes(nodes))
        return 0 if nodes else 1

    if args.list_coverage:
        report = compute_coverage_report(path=path)
        if args.json:
            print(
                json.dumps(
                    {
                        "total_nodes": report.total_nodes,
                        "rule_counts": report.rule_counts,
                        "uncovered_rules": report.uncovered_rules,
                        "nerv_family_count": report.nerv_family_count,
                        "outside_nerv_count": report.outside_nerv_count,
                        "horizon_counts": report.horizon_counts,
                        "architectural_class_counts": report.architectural_class_counts,
                        "in_flight_count": report.in_flight_count,
                        "dispatched_count": report.dispatched_count,
                        "deferred_count": report.deferred_count,
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
        else:
            print(_render_text_coverage(report))
        # Operator-action signal: uncovered rules OR outside-NeRV below recommended
        if (
            report.uncovered_rules
            or report.outside_nerv_count < OUTSIDE_NERV_RECOMMENDED_MINIMUM
        ):
            return 1
        return 0

    if args.list_uncovered_rules:
        uncovered = query_uncovered_rules(path=path)
        if args.json:
            print(json.dumps({"uncovered_rules": uncovered}, sort_keys=True, indent=2))
        else:
            if uncovered:
                print("Uncovered canonical rules:")
                for r in uncovered:
                    print(f"  - {r}")
            else:
                print("All canonical rules have at least one active coverage anchor.")
        return 1 if uncovered else 0

    if args.list_outside_nerv:
        nodes = query_outside_nerv_family(path=path)
        if args.json:
            print(
                json.dumps(
                    {
                        "outside_nerv_family_substrates": nodes,
                        "count": len(nodes),
                        "recommended_minimum": OUTSIDE_NERV_RECOMMENDED_MINIMUM,
                        "nerv_family_architectural_classes": sorted(
                            NERV_FAMILY_ARCHITECTURAL_CLASSES
                        ),
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
        else:
            print(f"OUTSIDE-NeRV-family substrates ({len(nodes)}):")
            print(_render_text_nodes(nodes))
            if len(nodes) < OUTSIDE_NERV_RECOMMENDED_MINIMUM:
                print(
                    f"\nWARNING: outside-NeRV count ({len(nodes)}) below operator-recommended "
                    f"minimum ({OUTSIDE_NERV_RECOMMENDED_MINIMUM}). The operator's 2026-05-16 "
                    f"binding constraint requires at least 3-4 outside-NeRV-family substrates "
                    f"in any K-measurement Wave-N plan."
                )
        return 1 if len(nodes) < OUTSIDE_NERV_RECOMMENDED_MINIMUM else 0

    parser.error("no action specified")
    return 2


if __name__ == "__main__":
    sys.exit(main())
