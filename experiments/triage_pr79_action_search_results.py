# SPDX-License-Identifier: MIT
"""Triage recovered PR79 action-search artifacts before exact-eval dispatch.

This is a custody helper, not a scorer. It converts a recovered Modal result
directory into a deterministic archive matrix with SHA-256, byte deltas, and
the exact component improvement still required to beat a target score. Exact
CUDA auth eval remains the only score truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCORE_DENOMINATOR_BYTES = 37_545_489
RATE_SCORE_WEIGHT = 25.0
# Current A++ T4 frontier after the lossless PR79 S2 action repack.
DEFAULT_FRONTIER_SCORE = 0.31453355357318635
DEFAULT_FRONTIER_BYTES = 277_321
DEFAULT_TARGET_SCORE = 0.31
DEFAULT_FRONTIER_SHA256 = "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
ACTION_SEARCH_ARCHIVE_NAMES = {"probe_archive.zip", "archive_optimized.zip"}
BASE_ARCHIVE_NAMES = {"archive.zip"}


class TriageError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TriageError(f"{path} must contain a JSON object")
    return payload


def _find_json_near(path: Path, names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in names:
        payload = _load_json(path.parent / name)
        if payload is not None:
            return payload
    return None


def _nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _manifest_archive_identity(manifest: dict[str, Any] | None, archive_name: str) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {"has_declared_bytes": False, "has_declared_sha256": False, "matches_actual": False}
    archive_block = manifest.get("output_archive")
    if not isinstance(archive_block, dict):
        archive_block = manifest.get(archive_name)
    declared_bytes = None
    declared_sha = None
    if isinstance(archive_block, dict):
        declared_bytes = archive_block.get("bytes")
        declared_sha = archive_block.get("sha256")
    declared_bytes = declared_bytes if declared_bytes is not None else manifest.get("candidate_archive_bytes")
    declared_sha = declared_sha or manifest.get("candidate_archive_sha256")
    return {
        "declared_bytes": declared_bytes,
        "declared_sha256": declared_sha,
        "has_declared_bytes": isinstance(declared_bytes, int),
        "has_declared_sha256": isinstance(declared_sha, str) and len(declared_sha) == 64,
    }


def _s2_packing_or_reason(manifest: dict[str, Any] | None) -> dict[str, Any]:
    stream_packing = _nested(manifest, "stream_packing")
    action_codec = _nested(stream_packing, "action_codec")
    absent_reason = _nested(manifest, "s2_packing_absent_reason")
    has_s2 = isinstance(action_codec, str) and "S2" in action_codec
    has_reason = isinstance(absent_reason, str) and bool(absent_reason.strip())
    return {
        "action_codec": action_codec,
        "has_s2_packing": has_s2,
        "s2_packing_absent_reason": absent_reason,
        "s2_packing_or_explicit_reason": has_s2 or has_reason,
    }


def _no_op_guard(manifest: dict[str, Any] | None) -> dict[str, Any]:
    no_op = _nested(manifest, "no_op_detection")
    status = _nested(no_op, "status")
    present = isinstance(no_op, dict) and isinstance(status, str) and bool(status)
    non_noop = present and status != "byte_noop"
    archive_equal = bool(_nested(no_op, "archive_sha_equal_to_source"))
    payload_equal = bool(_nested(no_op, "payload_sha_equal_to_source"))
    return {
        "no_op_detection_present": present,
        "no_op_status": status,
        "passes": bool(non_noop and not archive_equal and not payload_equal),
    }


def _accounting_guard(manifest: dict[str, Any] | None) -> dict[str, Any]:
    accounting = _nested(manifest, "action_record_accounting")
    duplicate_accounting = _nested(accounting, "duplicate_pair_tile_accounting")
    record_order = _nested(accounting, "record_order")
    parity_requirement = _nested(accounting, "raw_output_parity_requirement")
    has_duplicate_accounting = isinstance(duplicate_accounting, dict)
    order_changed = bool(_nested(record_order, "encoder_reorders_records"))
    duplicate_count = _nested(duplicate_accounting, "duplicate_pair_tile_record_count")
    duplicates_present = isinstance(duplicate_count, int) and duplicate_count > 0
    parity_required = order_changed or duplicates_present
    has_parity_requirement = isinstance(parity_requirement, dict) and isinstance(
        parity_requirement.get("required"),
        bool,
    )
    return {
        "duplicate_pair_tile_accounting_present": has_duplicate_accounting,
        "order_changed": order_changed,
        "duplicate_pair_tile_record_count": duplicate_count,
        "raw_output_parity_requirement_present": has_parity_requirement,
        "raw_output_parity_requirement_required": parity_required,
        "passes": bool(has_duplicate_accounting and (not parity_required or has_parity_requirement)),
    }


def _break_even_guard(manifest: dict[str, Any] | None) -> dict[str, Any]:
    math_block = _nested(manifest, "break_even_math")
    vs_pr79 = _nested(math_block, "versus_pr79")
    vs_s2 = _nested(math_block, "versus_pr79_s2")
    return {
        "break_even_math_present": isinstance(math_block, dict),
        "versus_pr79_present": isinstance(vs_pr79, dict),
        "versus_pr79_s2_present": isinstance(vs_s2, dict),
        "passes": isinstance(vs_pr79, dict) and isinstance(vs_s2, dict),
    }


def _repo_rel(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def candidate_record(
    archive: Path,
    *,
    root: Path,
    frontier_score: float,
    frontier_bytes: int,
    frontier_sha256: str,
    target_score: float,
) -> dict[str, Any]:
    archive = archive.resolve()
    size = archive.stat().st_size
    sha = sha256_file(archive)
    rate_delta = RATE_SCORE_WEIGHT * (size - frontier_bytes) / SCORE_DENOMINATOR_BYTES
    projected_score_if_same_components = frontier_score + rate_delta
    improvement_needed_for_target = max(0.0, projected_score_if_same_components - target_score)
    improvement_needed_for_frontier = max(0.0, projected_score_if_same_components - frontier_score)
    is_base_name = archive.name in BASE_ARCHIVE_NAMES
    is_known_candidate_name = archive.name in ACTION_SEARCH_ARCHIVE_NAMES
    manifest = _find_json_near(archive, ("manifest.json", "summary.json", "probe_archive_manifest.json"))
    manifest_score_claim = manifest.get("score_claim") if isinstance(manifest, dict) else None
    source_sha = None
    if isinstance(manifest, dict):
        base = manifest.get("base_archive")
        if isinstance(base, dict):
            source_sha = base.get("sha256")
        source_sha = source_sha or manifest.get("source_archive_sha256")
    no_op_vs_frontier = sha == frontier_sha256 or source_sha == sha
    archive_identity = _manifest_archive_identity(manifest, archive.name)
    archive_identity["matches_actual"] = (
        archive_identity.get("declared_bytes") == size
        and archive_identity.get("declared_sha256") == sha
    )
    s2_guard = _s2_packing_or_reason(manifest)
    noop_guard = _no_op_guard(manifest)
    accounting_guard = _accounting_guard(manifest)
    break_even_guard = _break_even_guard(manifest)
    guard_results = {
        "deterministic_archive_identity": bool(
            archive_identity["has_declared_bytes"]
            and archive_identity["has_declared_sha256"]
            and archive_identity["matches_actual"]
        ),
        "s2_packing_or_explicit_reason": bool(s2_guard["s2_packing_or_explicit_reason"]),
        "no_op_detection": bool(noop_guard["passes"]),
        "duplicate_pair_tile_accounting": bool(accounting_guard["passes"]),
        "break_even_math_vs_pr79_and_s2": bool(break_even_guard["passes"]),
    }
    failed_guards = [name for name, ok in guard_results.items() if not ok]
    dispatchable_after_parity_gate = (
        is_known_candidate_name
        and not is_base_name
        and not no_op_vs_frontier
        and manifest_score_claim is False
        and not failed_guards
    )
    return {
        "archive_path": _repo_rel(archive, root=root),
        "archive_bytes": size,
        "archive_sha256": sha,
        "archive_name": archive.name,
        "byte_delta_vs_frontier": size - frontier_bytes,
        "rate_score_delta_vs_frontier": rate_delta,
        "projected_score_if_components_unchanged": projected_score_if_same_components,
        "score_improvement_needed_to_beat_frontier": improvement_needed_for_frontier,
        "score_improvement_needed_to_beat_target": improvement_needed_for_target,
        "break_even_math": {
            "rate_score_weight": RATE_SCORE_WEIGHT,
            "score_denominator_bytes": SCORE_DENOMINATOR_BYTES,
            "versus_pr79_s2_frontier": {
                "archive_byte_delta": size - frontier_bytes,
                "rate_score_delta": rate_delta,
                "required_component_improvement_to_beat_frontier": improvement_needed_for_frontier,
                "required_component_improvement_to_beat_target": improvement_needed_for_target,
            },
        },
        "target_score": target_score,
        "frontier": {
            "score": frontier_score,
            "archive_bytes": frontier_bytes,
            "archive_sha256": frontier_sha256,
        },
        "score_claim": False,
        "evidence_grade": "empirical_modal_artifact_triage_only",
        "manifest_score_claim": manifest_score_claim,
        "source_archive_sha256_from_manifest": source_sha,
        "no_op_vs_frontier": no_op_vs_frontier,
        "preflight_guards": {
            "archive_identity": archive_identity,
            "s2_packing": s2_guard,
            "no_op_detection": noop_guard,
            "action_record_accounting": accounting_guard,
            "break_even_math": break_even_guard,
            "failed": failed_guards,
            "passed": not failed_guards,
        },
        "dispatchable_after_parity_gate": dispatchable_after_parity_gate,
        "next_gate": (
            "run raw-output/action parity, claim lane, then exact T4 CUDA auth eval"
            if dispatchable_after_parity_gate
            else "do not dispatch until all PR79 action-search preflight guards pass"
        ),
    }


def triage_results(
    result_dir: Path,
    *,
    output_json: Path,
    frontier_score: float = DEFAULT_FRONTIER_SCORE,
    frontier_bytes: int = DEFAULT_FRONTIER_BYTES,
    frontier_sha256: str = DEFAULT_FRONTIER_SHA256,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict[str, Any]:
    if not result_dir.exists():
        raise TriageError(f"result_dir does not exist: {result_dir}")
    archives = sorted(
        p for p in result_dir.rglob("*.zip")
        if p.is_file() and (p.name in ACTION_SEARCH_ARCHIVE_NAMES or p.name in BASE_ARCHIVE_NAMES)
    )
    records = [
        candidate_record(
            archive,
            root=Path.cwd(),
            frontier_score=frontier_score,
            frontier_bytes=frontier_bytes,
            frontier_sha256=frontier_sha256,
            target_score=target_score,
        )
        for archive in archives
    ]
    records.sort(
        key=lambda row: (
            not row["dispatchable_after_parity_gate"],
            row["score_improvement_needed_to_beat_target"],
            row["archive_bytes"],
            row["archive_path"],
        )
    )
    summary = {
        "schema": "pr79_action_search_result_triage_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "result_dir": _repo_rel(result_dir, root=Path.cwd()),
        "candidate_count": len(records),
        "dispatchable_after_parity_gate_count": sum(
            1 for row in records if row["dispatchable_after_parity_gate"]
        ),
        "candidates": records,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via exact CUDA auth eval"
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frontier-score", type=float, default=DEFAULT_FRONTIER_SCORE)
    parser.add_argument("--frontier-archive-bytes", type=int, default=DEFAULT_FRONTIER_BYTES)
    parser.add_argument("--frontier-archive-sha256", default=DEFAULT_FRONTIER_SHA256)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    args = parser.parse_args()
    summary = triage_results(
        args.result_dir,
        output_json=args.output_json,
        frontier_score=args.frontier_score,
        frontier_bytes=args.frontier_archive_bytes,
        frontier_sha256=args.frontier_archive_sha256,
        target_score=args.target_score,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
