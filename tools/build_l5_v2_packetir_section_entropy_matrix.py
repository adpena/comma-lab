#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the L5 v2 PacketIR section-entropy matrix from real PR106 archives.

The matrix is a planning artifact. It profiles real PacketIR sections and, when
requested, emits charged PCR1 prototype rows so unpriced context-entropy floors
cannot masquerade as score movement. It never builds submission archives,
dispatches jobs, or claims score/promotability.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_candidate_matrix import (  # noqa: E402
    DEFAULT_PR106_PACKETIR_CANDIDATES,
)
from tac.packet_compiler.pr106_context_recode import (  # noqa: E402
    DEFAULT_CONTEXT_ORDERS,
    TARGETABLE_INNER_SECTIONS,
    build_pr106_context_recode_report,
    encode_context_recode_section,
    load_pr106_context_source_from_archive,
)
from tac.repo_io import json_text  # noqa: E402

SCHEMA = "l5_v2_packetir_section_entropy_matrix_v1"
TOOL = "tools/build_l5_v2_packetir_section_entropy_matrix.py"
DEFAULT_JSON_OUT = (
    REPO_ROOT / ".omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.json"
)
DEFAULT_MD_OUT = (
    REPO_ROOT / ".omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.md"
)
DISPATCH_BLOCKERS = (
    "planning_only_packetir_section_entropy_matrix",
    "prototype_runtime_decoder_not_integrated",
    "full_frame_same_runtime_parity_missing",
    "exact_cuda_auth_eval_missing",
    "contest_auth_eval_adjudication_missing",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="candidate id to include; repeatable. Defaults to every known PR106 candidate.",
    )
    parser.add_argument(
        "--orders",
        default="0,1,2,3,4",
        help="comma-separated context floor orders",
    )
    parser.add_argument(
        "--prototype-orders",
        default="2",
        help="comma-separated context orders for charged PCR1 prototype rows",
    )
    parser.add_argument(
        "--build-prototypes",
        action="store_true",
        help="emit charged prototype rows for targetable sections and --prototype-orders",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args(argv)


def _parse_orders(text: str) -> tuple[int, ...]:
    try:
        orders = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("orders must be comma-separated integers") from exc
    if not orders:
        raise argparse.ArgumentTypeError("orders must be nonempty")
    if any(order < 0 for order in orders):
        raise argparse.ArgumentTypeError("orders must be nonnegative")
    return orders


def build_matrix(
    *,
    candidate_ids: set[str] | None = None,
    context_orders: tuple[int, ...] = DEFAULT_CONTEXT_ORDERS,
    prototype_orders: tuple[int, ...] = (2,),
    build_prototypes: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for spec in DEFAULT_PR106_PACKETIR_CANDIDATES:
        if candidate_ids is not None and spec.candidate_id not in candidate_ids:
            continue
        rows.append(
            _candidate_row(
                spec,
                context_orders=context_orders,
                prototype_orders=prototype_orders,
                build_prototypes=build_prototypes,
            )
        )
    if candidate_ids is not None:
        found = {str(row["candidate_id"]) for row in rows}
        for candidate_id in sorted(candidate_ids - found):
            blockers.append(f"candidate_id_not_known:{candidate_id}")

    existing_rows = [row for row in rows if row.get("status") == "profiled"]
    prototype_rows = [
        proto
        for row in existing_rows
        for proto in row.get("prototype_rows", [])
        if isinstance(proto, dict)
    ]
    rate_positive = [
        proto for proto in prototype_rows if proto.get("delta_bytes_vs_source_section", 1) < 0
    ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "context_orders": list(context_orders),
        "prototype_orders": list(prototype_orders),
        "prototype_rows_built": bool(build_prototypes),
        "candidate_count": len(rows),
        "profiled_candidate_count": len(existing_rows),
        "prototype_row_count": len(prototype_rows),
        "rate_positive_prototype_row_count": len(rate_positive),
        "best_rate_positive_prototype": min(
            rate_positive,
            key=lambda row: int(row["delta_bytes_vs_source_section"]),
            default=None,
        ),
        "matrix_blockers": blockers,
        "rows": rows,
    }


def _candidate_row(
    spec: Any,
    *,
    context_orders: tuple[int, ...],
    prototype_orders: tuple[int, ...],
    build_prototypes: bool,
) -> dict[str, Any]:
    archive_path = REPO_ROOT / spec.archive_path
    base: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "expected_format_id": spec.expected_format_id,
        "archive_path": spec.archive_path,
        "runtime_consumption_path": spec.runtime_consumption_path,
        "notes": spec.notes,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if not archive_path.is_file():
        return {
            **base,
            "status": "missing_archive",
            "blockers": [f"archive_missing:{spec.archive_path}"],
        }
    try:
        source = load_pr106_context_source_from_archive(archive_path)
        result = build_pr106_context_recode_report(
            source,
            target_section="auto",
            context_order=prototype_orders[0] if prototype_orders else 2,
            context_orders=context_orders,
            build_prototype=False,
        )
    except Exception as exc:
        return {
            **base,
            "status": "profile_failed",
            "blockers": [f"profile_failed:{type(exc).__name__}:{exc}"],
        }

    report = result.report
    section_rows = []
    for profile in report.get("section_context_profiles", []):
        if not isinstance(profile, dict):
            continue
        section_rows.append(_section_summary(profile))

    prototype_rows: list[dict[str, Any]] = []
    if build_prototypes:
        for section_name in TARGETABLE_INNER_SECTIONS:
            section = source.section(section_name)
            for order in prototype_orders:
                prototype_rows.append(
                    encode_context_recode_section(
                        section.name,
                        section.data,
                        context_order=order,
                    ).manifest()
                )

    return {
        **base,
        "status": "profiled",
        "source": report.get("source"),
        "sections": section_rows,
        "selected_target": report.get("selected_target"),
        "prototype_rows": prototype_rows,
        "rate_positive_prototype_rows": [
            row for row in prototype_rows if row.get("delta_bytes_vs_source_section", 1) < 0
        ],
        "blockers": list(DISPATCH_BLOCKERS),
    }


def _section_summary(profile: dict[str, Any]) -> dict[str, Any]:
    floors = profile.get("floors")
    floor_rows = floors if isinstance(floors, list) else []
    return {
        "section_name": profile.get("section_name"),
        "role": profile.get("role"),
        "targetable": profile.get("targetable") is True,
        "current_bytes": profile.get("current_bytes"),
        "sha256": profile.get("sha256"),
        "best_high_order_context_order": profile.get("best_high_order_context_order"),
        "best_high_order_floor_bytes": profile.get("best_high_order_floor_bytes"),
        "best_high_order_delta_vs_current_bytes": (
            profile.get("best_high_order_delta_vs_current_bytes")
        ),
        "floors": floor_rows,
        "limitations": profile.get("limitations"),
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# L5 v2 PacketIR Section Entropy Matrix",
        "",
        f"- planning_only: `{str(matrix['planning_only']).lower()}`",
        f"- score_claim: `{str(matrix['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(matrix['ready_for_exact_eval_dispatch']).lower()}`",
        f"- profiled_candidate_count: `{matrix['profiled_candidate_count']}`",
        f"- prototype_row_count: `{matrix['prototype_row_count']}`",
        f"- rate_positive_prototype_row_count: `{matrix['rate_positive_prototype_row_count']}`",
        f"- dispatch_blockers: `{', '.join(matrix['dispatch_blockers'])}`",
        "",
        "## Candidate Rows",
        "",
        "| candidate | status | format | best target | floor delta | best prototype delta |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in matrix.get("rows", []):
        selected = row.get("selected_target") if isinstance(row, dict) else {}
        if not isinstance(selected, dict):
            selected = {}
        prototype_rows = row.get("prototype_rows") if isinstance(row, dict) else []
        best_proto = min(
            (proto for proto in prototype_rows if isinstance(proto, dict)),
            key=lambda proto: int(proto.get("delta_bytes_vs_source_section", 10**18)),
            default={},
        )
        lines.append(
            "| {candidate} | `{status}` | `{fmt}` | `{target}` | {floor_delta} | {proto_delta} |".format(
                candidate=row.get("candidate_id"),
                status=row.get("status"),
                fmt=row.get("expected_format_id"),
                target=selected.get("section_name", ""),
                floor_delta=selected.get("best_high_order_delta_vs_current_bytes", ""),
                proto_delta=best_proto.get("delta_bytes_vs_source_section", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context_orders = _parse_orders(args.orders)
    prototype_orders = _parse_orders(args.prototype_orders)
    candidate_ids = set(args.candidate_id) if args.candidate_id else None
    matrix = build_matrix(
        candidate_ids=candidate_ids,
        context_orders=context_orders,
        prototype_orders=prototype_orders,
        build_prototypes=args.build_prototypes,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(matrix), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(matrix), encoding="utf-8")
    print(f"wrote {args.json_out}")
    if args.md_out:
        print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
