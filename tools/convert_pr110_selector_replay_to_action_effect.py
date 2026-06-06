#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert a PR110 selector replay summary row into ActionEffect JSONL.

This is a thin adapter over ``ActionEffect.from_pr110_selector_row``.  It does
not run inflate, score frames, infer contest authority, or join unrelated
frontier-rate rows.  The input summary must already contain the replay axis,
baseline distortion/bytes, candidate distortion/bytes, and a matching
candidate manifest.  The output row is non-promotable and carries the measured
authority string from the replay axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import (  # noqa: E402
    ActionEffect,
    NormalizationScope,
    append_action_effect,
    exact_delta_score,
    validate_action_effect_payload,
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_path(value: str | None, *, base: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    nearby = (base.parent / path).resolve(strict=False)
    if nearby.exists():
        return nearby
    return (REPO_ROOT / path).resolve(strict=False)


def _select_summary_row(summary: Mapping[str, Any], candidate_id: str | None) -> Mapping[str, Any]:
    rows = summary.get("rows")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        candidates.extend(row for row in rows if isinstance(row, Mapping))
    for key in ("best_completed", "best_candidate"):
        row = summary.get(key)
        if isinstance(row, Mapping):
            candidates.append(row)
    if not candidates:
        raise ValueError("PR110 selector summary has no replay rows")
    if candidate_id is None:
        return candidates[0]
    for row in candidates:
        if str(row.get("candidate_id") or "") == candidate_id:
            return row
    raise ValueError(f"candidate_id not found in PR110 selector summary: {candidate_id}")


def _manifest_for_row(
    row: Mapping[str, Any],
    *,
    summary_path: Path,
    manifest_override: Path | None,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_override
    if manifest_path is None:
        manifest_path = _resolve_path(str(row.get("manifest_path") or ""), base=summary_path)
    if manifest_path is None or not manifest_path.is_file():
        raise ValueError(f"candidate manifest missing for selector replay row: {manifest_path}")
    manifest = _load_json_object(manifest_path, label="candidate manifest")
    row_candidate_id = str(row.get("candidate_id") or "")
    manifest_candidate_id = str(manifest.get("candidate_id") or "")
    if row_candidate_id and manifest_candidate_id and row_candidate_id != manifest_candidate_id:
        raise ValueError(
            "candidate manifest does not match replay row: "
            f"summary={row_candidate_id} manifest={manifest_candidate_id}"
        )
    return manifest_path, manifest


def build_pr110_selector_replay_action_effect(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    candidate_id: str | None = None,
    manifest_path: Path | None = None,
    consumer: str = "action_effect_commutator_ledger",
) -> ActionEffect:
    """Build one non-promotional PR110 selector replay ActionEffect."""

    baseline = summary.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("PR110 selector summary missing baseline object")
    row = _select_summary_row(summary, candidate_id)
    manifest_file, manifest = _manifest_for_row(row, summary_path=summary_path, manifest_override=manifest_path)

    axis = str(row.get("axis") or baseline.get("axis") or "").strip()
    if not axis:
        raise ValueError("PR110 selector replay row missing authority axis")
    old_bytes = _required_int(baseline, "archive_bytes")
    new_bytes = _required_int(row, "archive_bytes")
    old_d_seg = _required_float(baseline, "avg_segnet_dist")
    new_d_seg = _required_float(row, "avg_segnet_dist")
    old_d_pose = _required_float(baseline, "avg_posenet_dist")
    new_d_pose = _required_float(row, "avg_posenet_dist")

    archive = dict(_mapping(manifest.get("archive")) or _mapping(row.get("archive")) or {})
    if not archive:
        archive = {
            "bytes": new_bytes,
            "delta_bytes_vs_source_archive": new_bytes - old_bytes,
            "sha256": row.get("archive_sha256"),
        }
    source_archive = _mapping(manifest.get("source_archive"))
    if source_archive and source_archive.get("bytes") is not None:
        old_bytes = int(source_archive["bytes"])

    merged: dict[str, Any] = {
        "candidate_id": str(row.get("candidate_id") or manifest.get("candidate_id") or "").strip(),
        "family": "pr110",
        "action_kind": "selector_replay",
        "authority": f"{axis} pr110_selector_replay",
        "normalization_scope": NormalizationScope.FULL_VIDEO_EQUIV_ESTIMATE.value,
        "selected_pairs": row.get("selected_pairs") or _mapping(manifest.get("selection")).get("selected_pairs"),
        "archive": archive,
        "old_archive_bytes": old_bytes,
        "new_archive_bytes": new_bytes,
        "old_d_seg": old_d_seg,
        "new_d_seg": new_d_seg,
        "old_d_pose": old_d_pose,
        "new_d_pose": new_d_pose,
        "official_inflate_control_passed": bool(
            row.get("official_inflate_returncode") == 0
            or manifest.get("official_inflate_control") is True
            or manifest.get("official_inflate_control_passed") is True
        ),
        "restore_state_passed": row.get("locality_control_passed"),
        "artifact_ref": manifest_file.as_posix(),
        "summary_path": summary_path.as_posix(),
        "archive_sha256": row.get("archive_sha256") or archive.get("sha256"),
        "taint_status": "unknown",
    }
    if not merged["candidate_id"]:
        raise ValueError("PR110 selector replay row missing candidate_id")

    effect = ActionEffect.from_pr110_selector_row(merged, consumer=consumer)
    expected = exact_delta_score(old_d_seg, new_d_seg, old_d_pose, new_d_pose, old_bytes, new_bytes)
    if effect.delta_score_total != expected:
        raise ValueError(
            "PR110 selector replay ActionEffect did not use shared exact_delta_score: "
            f"effect={effect.delta_score_total!r} expected={expected!r}"
        )
    return effect


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"required integer field missing: {key}")
    return value


def _required_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"required numeric field missing: {key}")
    return float(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="PR110 selector replay summary JSON.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="ActionEffect JSONL ledger to append.")
    parser.add_argument("--candidate-id", default=None, help="Candidate id to select; defaults to first replay row.")
    parser.add_argument("--manifest", default=None, type=Path, help="Optional matching candidate manifest override.")
    parser.add_argument("--consumer", default="action_effect_commutator_ledger")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary_path = args.summary.resolve(strict=False)
        summary = _load_json_object(summary_path, label="summary")
        effect = build_pr110_selector_replay_action_effect(
            summary,
            summary_path=summary_path,
            candidate_id=args.candidate_id,
            manifest_path=args.manifest.resolve(strict=False) if args.manifest is not None else None,
            consumer=args.consumer,
        )
        record = append_action_effect(effect, args.output_jsonl)
        validation = validate_action_effect_payload(record)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: could not convert PR110 selector replay to ActionEffect: {exc}", file=sys.stderr)
        return 2

    summary_payload = {
        "schema": "tac.pr110_selector_replay_action_effect_conversion.v1",
        "summary_path": args.summary.as_posix(),
        "output_jsonl": args.output_jsonl.as_posix(),
        "action_id": record["action_id"],
        "authority": record["authority"],
        "delta_score_total": record["delta_score_total"],
        "delta_bytes": record["delta_bytes"],
        "value_per_byte": record["value_per_byte"],
        "validation": validation,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", end="")
    return 0 if validation.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
