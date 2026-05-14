#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a fail-closed Omega-OPT linear-stack packet manifest.

This CLI emits the operator-facing packet scaffold for the Omega-OPT linear
stack. It can optionally import the nested ``omega_opt_linear_stack`` claim
from a planning report, but it never promotes that prediction into score
evidence unless the exact A++ archive/runtime/eval anchor fields are supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.omega_opt_linear_stack_packet import (  # noqa: E402
    LINEAR_STACK_CLAIM_ID,
    build_linear_stack_packet_manifest,
    canonical_json_sha256,
    has_exact_linear_stack_anchor,
    linear_stack_packet_status,
    validate_linear_stack_packet_manifest,
)

_MISSING = object()
_GATED_FLAG_NAMES: tuple[str, ...] = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "promotion_allowed",
    "dispatchable",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _walk_objects(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Mapping[str, Any]]]:
    found: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        found.append((path, value))
        for key, item in value.items():
            found.extend(_walk_objects(item, (*path, key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_walk_objects(item, (*path, index)))
    return found


def _json_pointer(path: Sequence[Any]) -> str:
    if not path:
        return ""
    parts = []
    for part in path:
        text = str(part).replace("~", "~0").replace("/", "~1")
        parts.append(text)
    return "/" + "/".join(parts)


def _lookup_path(data: Mapping[str, Any], path: Sequence[Any]) -> Any:
    current: Any = data
    for part in path:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)) and isinstance(part, int):
            if part < 0 or part >= len(current):
                return _MISSING
            current = current[part]
        else:
            return _MISSING
    return current


def _extract_source_claim(plan: Mapping[str, Any]) -> tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any] | None]:
    matches = [
        (path, obj)
        for path, obj in _walk_objects(plan)
        if obj.get("claim_id") == LINEAR_STACK_CLAIM_ID
    ]
    if not matches:
        raise SystemExit(f"{LINEAR_STACK_CLAIM_ID!r} claim not found in source plan")
    if len(matches) > 1:
        pointers = ", ".join(_json_pointer(path) for path, _obj in matches)
        raise SystemExit(f"ambiguous {LINEAR_STACK_CLAIM_ID!r} claims in source plan: {pointers}")

    claim_path, claim = matches[0]
    parent = None
    if len(claim_path) >= 2:
        candidate = _lookup_path(plan, claim_path[:-2])
        if isinstance(candidate, Mapping):
            parent = candidate
    return claim_path, claim, parent


def _source_claim_metadata(
    *,
    source_plan: Path,
    claim_path: Sequence[Any],
    claim: Mapping[str, Any],
    parent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    copied_fields = (
        "claim_id",
        "label",
        "predicted_score",
        "score_classification",
        "current_anchor_status",
        "missing_anchors",
        "next_buildable_1to1_test",
        "reactivation_criteria",
        "notes",
        "requires_exact_1to1_anchor",
        *_GATED_FLAG_NAMES,
    )
    normalized = {field: claim[field] for field in copied_fields if field in claim}
    return {
        "source_plan_path": source_plan.as_posix(),
        "json_pointer": _json_pointer(claim_path),
        "source_claim_sha256": canonical_json_sha256(dict(claim)),
        "parent_evidence_grade": parent.get("evidence_grade") if parent is not None else None,
        "normalized_claim": normalized,
    }


def _source_evidence_grade(source_metadata: Mapping[str, Any] | None) -> str | None:
    if source_metadata is None:
        return None
    grade = source_metadata.get("parent_evidence_grade")
    return str(grade) if grade not in (None, "") else None


def _source_claim_field(source_metadata: Mapping[str, Any] | None, name: str) -> Any:
    if source_metadata is None:
        return _MISSING
    claim = source_metadata.get("normalized_claim")
    if isinstance(claim, Mapping) and name in claim:
        return claim[name]
    return _MISSING


def _requested_gated_flags(args: argparse.Namespace) -> dict[str, bool]:
    return {name: bool(getattr(args, name)) for name in _GATED_FLAG_NAMES}


def _refresh_manifest_hash_and_status(manifest: dict[str, Any]) -> None:
    manifest["promotion_status"] = linear_stack_packet_status(manifest)
    manifest["blockers"] = manifest["promotion_status"]["blockers"]
    manifest["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def materialize_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_metadata: dict[str, Any] | None = None
    if args.source_plan is not None:
        plan = _load_json_object(args.source_plan)
        claim_path, claim, parent = _extract_source_claim(plan)
        source_metadata = _source_claim_metadata(
            source_plan=args.source_plan,
            claim_path=claim_path,
            claim=claim,
            parent=parent,
        )

    evidence_grade = args.evidence_grade or _source_evidence_grade(source_metadata) or "prediction"
    evidence_semantics = args.evidence_semantics or "omega_opt_linear_stack_packet_scaffold_no_score"
    prototype_id = args.prototype_id
    if prototype_id is None:
        prototype_id = "omega_opt_linear_stack_pr101_1to1_scaffold"

    manifest = build_linear_stack_packet_manifest(
        prototype_id=prototype_id,
        archive_path=args.archive_path,
        archive_bytes=args.archive_bytes,
        archive_sha256=args.archive_sha256,
        runtime_packet_path=args.runtime_packet_path,
        inflate_path=args.inflate_path,
        evidence_grade=evidence_grade,
        evidence_semantics=evidence_semantics,
        contest_auth_eval_json=args.contest_auth_eval_json,
        one_to_one_anchor_artifact=args.one_to_one_anchor_artifact,
    )

    if source_metadata is not None:
        manifest["source_claim"] = source_metadata
        source_predicted_score = _source_claim_field(source_metadata, "predicted_score")
        if isinstance(source_predicted_score, int | float) and not isinstance(source_predicted_score, bool):
            manifest["predicted_score"] = float(source_predicted_score)
        source_score_classification = _source_claim_field(source_metadata, "score_classification")
        if isinstance(source_score_classification, str) and source_score_classification.strip():
            manifest["score_classification"] = source_score_classification

    requested_flags = _requested_gated_flags(args)
    exact_anchor_complete = has_exact_linear_stack_anchor(manifest)
    suppressed_flags = sorted(name for name, requested in requested_flags.items() if requested and not exact_anchor_complete)
    for name, requested in requested_flags.items():
        manifest[name] = bool(requested and exact_anchor_complete)
    if suppressed_flags:
        manifest["suppressed_positive_intents"] = {
            "reason": "exact_a_plus_plus_linear_stack_anchor_incomplete",
            "fields": suppressed_flags,
        }

    _refresh_manifest_hash_and_status(manifest)
    findings = validate_linear_stack_packet_manifest(manifest)
    if findings:
        manifest["materialization_findings"] = findings
        _refresh_manifest_hash_and_status(manifest)
    return manifest


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Manifest JSON output. Defaults to stdout.")
    parser.add_argument(
        "--source-plan",
        type=Path,
        help="Optional planning report containing a nested omega_opt_linear_stack claim.",
    )
    parser.add_argument("--prototype-id", help="Stable prototype id for the packet scaffold.")
    parser.add_argument("--archive-path", help="Exact archive.zip path, when available.")
    parser.add_argument("--archive-bytes", type=int, help="Exact archive.zip byte count, when available.")
    parser.add_argument("--archive-sha256", help="Exact archive.zip SHA-256, when available.")
    parser.add_argument("--runtime-packet-path", help="Runtime packet manifest/path consumed by inflate.")
    parser.add_argument("--inflate-path", help="inflate.sh path used for exact eval.")
    parser.add_argument(
        "--evidence-grade",
        help="Evidence grade for the packet. A++ is required for exact-anchor completion.",
    )
    parser.add_argument("--evidence-semantics", help="Evidence semantics string for the manifest.")
    parser.add_argument("--contest-auth-eval-json", help="Full-sample exact CUDA contest_auth_eval.json path.")
    parser.add_argument("--one-to-one-anchor-artifact", help="1:1 archive/runtime/eval anchor artifact path.")
    parser.add_argument("--score-claim", action="store_true", help="Request score_claim=true if exact A++ anchor is complete.")
    parser.add_argument(
        "--promotion-eligible",
        action="store_true",
        help="Request promotion_eligible=true if exact A++ anchor is complete.",
    )
    parser.add_argument(
        "--rank-or-kill-eligible",
        action="store_true",
        help="Request rank_or_kill_eligible=true if exact A++ anchor is complete.",
    )
    parser.add_argument(
        "--ready-for-exact-eval-dispatch",
        action="store_true",
        help="Request ready_for_exact_eval_dispatch=true if exact A++ anchor is complete.",
    )
    parser.add_argument(
        "--promotion-allowed",
        action="store_true",
        help="Request promotion_allowed=true if exact A++ anchor is complete.",
    )
    parser.add_argument("--dispatchable", action="store_true", help="Request dispatchable=true if exact A++ anchor is complete.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_manifest(args)
    text = json_text(manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
