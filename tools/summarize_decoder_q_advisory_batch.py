#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a decision packet for OP3-V3/FEC6 decoder-q advisory batches."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score_delta(row: dict[str, Any], baseline_score: float | None) -> float | None:
    advisory = row.get("advisory_eval")
    if not isinstance(advisory, dict):
        return None
    if row.get("delta_vs_baseline_score") is not None:
        return float(row["delta_vs_baseline_score"])
    if baseline_score is None or advisory.get("canonical_score") is None:
        return None
    return float(advisory["canonical_score"]) - float(baseline_score)


def _candidate_mutation(row: dict[str, Any]) -> dict[str, Any] | None:
    manifest = row.get("mutation_manifest")
    if not isinstance(manifest, dict):
        return None
    mutation_row = manifest.get("mutation_row")
    if not isinstance(mutation_row, dict):
        return None
    mutation = mutation_row.get("mutation")
    return mutation if isinstance(mutation, dict) else None


def _archive_custody(row: dict[str, Any]) -> dict[str, Any]:
    extract = row.get("archive_extract")
    if not isinstance(extract, dict):
        return {"passed": False, "reason": "missing_archive_extract_record"}
    passed = (
        extract.get("member") == "x"
        and int(extract.get("compress_type", -1)) == 0
        and int(extract.get("member_bytes", -1)) == 178417
    )
    return {
        "passed": passed,
        "member": extract.get("member"),
        "compress_type": extract.get("compress_type"),
        "member_bytes": extract.get("member_bytes"),
        "member_sha256": extract.get("member_sha256"),
        "archive_zip_bytes": extract.get("archive_zip_bytes"),
        "archive_zip_sha256": extract.get("archive_zip_sha256"),
    }


def _fixed_length_custody(row: dict[str, Any]) -> dict[str, Any]:
    manifest = row.get("mutation_manifest")
    if not isinstance(manifest, dict):
        return {"passed": False, "reason": "missing_mutation_manifest"}
    mutation_row = manifest.get("mutation_row")
    if not isinstance(mutation_row, dict):
        # Cumulative waterbucket candidates store the fixed-length fields at
        # manifest top-level instead of under a single mutation_row.
        mutation_row = manifest
    passed = (
        bool(mutation_row.get("fixed_length_runtime_compatible"))
        and int(mutation_row.get("source_decoder_len", -1)) == 162164
        and int(mutation_row.get("mutated_decoder_len", -1)) == 162164
        and int(mutation_row.get("length_delta", 99)) == 0
        and int(manifest.get("archive_zip_bytes", -1)) == 178517
        and int(manifest.get("archive_bin_bytes", -1)) == 178417
    )
    return {
        "passed": passed,
        "fixed_length_runtime_compatible": mutation_row.get("fixed_length_runtime_compatible"),
        "source_decoder_len": mutation_row.get("source_decoder_len"),
        "mutated_decoder_len": mutation_row.get("mutated_decoder_len"),
        "length_delta": mutation_row.get("length_delta"),
        "archive_zip_bytes": manifest.get("archive_zip_bytes"),
        "archive_bin_bytes": manifest.get("archive_bin_bytes"),
        "mutated_decoder_sha256": mutation_row.get("mutated_decoder_sha256"),
    }


def _inflate_custody(row: dict[str, Any]) -> dict[str, Any]:
    comparison = row.get("raw_comparison")
    if not isinstance(comparison, dict):
        return {"passed": False, "reason": "missing_raw_comparison"}
    passed = bool(comparison.get("passed_visible_change")) and int(comparison.get("changed_frame_count", 0)) > 0
    return {
        "passed": passed,
        "candidate_raw_sha256": comparison.get("candidate_raw_sha256"),
        "baseline_raw_sha256": comparison.get("baseline_raw_sha256"),
        "changed_frame_count": comparison.get("changed_frame_count"),
        "byte_delta_summary": comparison.get("byte_delta_summary"),
    }


def _candidate_rows(
    advisory_summary: dict[str, Any] | None,
    inflate_controls: dict[str, Any] | None,
    baseline_score: float | None,
) -> list[dict[str, Any]]:
    if advisory_summary is not None:
        source_rows = advisory_summary.get("candidates", [])
        source_kind = "advisory_summary"
    elif inflate_controls is not None:
        source_rows = inflate_controls.get("candidates", [])
        source_kind = "inflate_controls"
    else:
        source_rows = []
        source_kind = "none"
    out = []
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        advisory = row.get("advisory_eval") if isinstance(row.get("advisory_eval"), dict) else None
        delta = _score_delta(row, baseline_score)
        out.append(
            {
                "candidate_id": row.get("candidate_id"),
                "source_kind": source_kind,
                "mutation": _candidate_mutation(row),
                "fixed_length_custody": _fixed_length_custody(row),
                "archive_custody": _archive_custody(row),
                "inflate_custody": _inflate_custody(row),
                "advisory": (
                    {
                        "returncode": advisory.get("returncode"),
                        "canonical_score": advisory.get("canonical_score"),
                        "delta_vs_baseline_score": delta,
                        "avg_posenet_dist": advisory.get("avg_posenet_dist"),
                        "avg_segnet_dist": advisory.get("avg_segnet_dist"),
                        "rate_unscaled": advisory.get("rate_unscaled"),
                        "archive_size_bytes": advisory.get("archive_size_bytes"),
                        "raw_sha256": advisory.get("raw", {}).get("sha256")
                        if isinstance(advisory.get("raw"), dict)
                        else None,
                        "archive_sha256": advisory.get("archive", {}).get("sha256")
                        if isinstance(advisory.get("archive"), dict)
                        else None,
                        "blockers": advisory.get("blockers"),
                    }
                    if advisory is not None
                    else None
                ),
                "blockers": row.get("blockers"),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    out.sort(
        key=lambda item: (
            float("inf")
            if item.get("advisory") is None
            or item["advisory"].get("delta_vs_baseline_score") is None
            else float(item["advisory"]["delta_vs_baseline_score"]),
            str(item.get("candidate_id")),
        )
    )
    return out


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    feasibility = _read_json(args.feasibility.resolve())
    targeted_feasibility = _read_json(args.targeted_feasibility.resolve()) if args.targeted_feasibility else None
    inflate_controls = _read_json(args.inflate_controls.resolve()) if args.inflate_controls and args.inflate_controls.is_file() else None
    advisory_summary = _read_json(args.advisory_summary.resolve()) if args.advisory_summary and args.advisory_summary.is_file() else None
    next_selection = _read_json(args.next_selection.resolve()) if args.next_selection and args.next_selection.is_file() else None
    rows = _candidate_rows(advisory_summary, inflate_controls, args.baseline_score)
    advisory_success = [
        row
        for row in rows
        if isinstance(row.get("advisory"), dict) and row["advisory"].get("returncode") == 0
    ]
    improved = [
        row
        for row in advisory_success
        if row["advisory"].get("delta_vs_baseline_score") is not None
        and float(row["advisory"]["delta_vs_baseline_score"]) < 0.0
    ]
    best = advisory_success[0] if advisory_success else None
    all_custody_passed = all(
        bool(row["fixed_length_custody"]["passed"])
        and bool(row["archive_custody"]["passed"])
        and bool(row["inflate_custody"]["passed"])
        for row in rows
    ) if rows else False
    if advisory_summary is None:
        recommendation = "await_advisory_batch"
    elif improved and all_custody_passed:
        recommendation = "prepare_best_for_exact_cuda_dispatch_review"
    elif advisory_success and all_custody_passed:
        recommendation = "do_not_dispatch_exact_eval__widen_search"
    elif advisory_success:
        recommendation = "repair_decision_packet_custody_before_more_search"
    else:
        recommendation = "repair_advisory_batch_before_more_search"

    return {
        "schema": "op3v3_decoder_q_advisory_decision_packet_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/summarize_decoder_q_advisory_batch.py",
        "inputs": {
            "feasibility": str(args.feasibility.resolve()),
            "targeted_feasibility": str(args.targeted_feasibility.resolve()) if args.targeted_feasibility else None,
            "inflate_controls": str(args.inflate_controls.resolve()) if args.inflate_controls else None,
            "advisory_summary": str(args.advisory_summary.resolve()) if args.advisory_summary else None,
            "next_selection": str(args.next_selection.resolve()) if args.next_selection else None,
            "baseline_score": args.baseline_score,
        },
        "feasibility_summary": feasibility.get("summary"),
        "targeted_feasibility_summary": targeted_feasibility.get("summary") if targeted_feasibility else None,
        "inflate_controls_summary": inflate_controls.get("summary") if inflate_controls else None,
        "advisory_summary": advisory_summary.get("summary") if advisory_summary else None,
        "next_selection_summary": next_selection.get("summary") if next_selection else None,
        "decision": {
            "recommendation": recommendation,
            "all_current_candidate_custody_passed": all_custody_passed,
            "advisory_success_count": len(advisory_success),
            "improved_count": len(improved),
            "best_candidate_id": best.get("candidate_id") if best else None,
            "best_delta_vs_baseline_score": best.get("advisory", {}).get("delta_vs_baseline_score") if best else None,
            "best_score": best.get("advisory", {}).get("canonical_score") if best else None,
        },
        "candidates_ranked": rows,
        "next_queue": next_selection.get("queue", []) if next_selection else [],
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": "Decision packet only. Exact CUDA dispatch still requires lane claim and contest_auth_eval custody.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--targeted-feasibility", type=Path)
    parser.add_argument("--inflate-controls", type=Path)
    parser.add_argument("--advisory-summary", type=Path)
    parser.add_argument("--next-selection", type=Path)
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_packet(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommendation": payload["decision"]["recommendation"],
                "advisory_success_count": payload["decision"]["advisory_success_count"],
                "improved_count": payload["decision"]["improved_count"],
                "best_candidate_id": payload["decision"]["best_candidate_id"],
                "best_delta_vs_baseline_score": payload["decision"]["best_delta_vs_baseline_score"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
