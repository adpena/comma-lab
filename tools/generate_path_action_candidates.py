#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate path-coded ActionEffect candidates from hard-region NPZ inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import append_action_effect  # noqa: E402
from tac.analysis.path_action_producer import (  # noqa: E402
    PATH_ACTION_PRODUCER_SCHEMA,
    build_path_action_candidates_from_arrays,
    build_pose_temporal_path_candidates_from_arrays,
    build_selector_temporal_path_candidates_from_rows,
)

DEFAULT_SSD_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            if "Extra data" not in str(exc):
                raise
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _default_output_dir() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_SSD_ROOT / f"path_action_producer_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-region-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--rdp-epsilon", type=float, default=2.0)
    parser.add_argument("--base-archive-bytes", type=int, default=0)
    parser.add_argument("--old-d-seg", type=float, default=0.0)
    parser.add_argument("--old-d-pose", type=float, default=0.0)
    parser.add_argument("--pose-trajectory-json", type=Path, default=None)
    parser.add_argument("--selector-rows-json", type=Path, default=None)
    args = parser.parse_args()

    npz_path = args.hard_region_npz.expanduser().resolve(strict=True)
    out_dir = (args.output_dir or _default_output_dir()).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with np.load(npz_path, allow_pickle=False) as data:
        results.append(build_path_action_candidates_from_arrays(
            target_labels_bhw=data["target_labels_bhw"],
            candidate_argmax_bhw=data["candidate_argmax_bhw"],
            target_margin_bhw=data["target_margin_bhw"] if "target_margin_bhw" in data.files else None,
            pair_indices=data["pair_indices"] if "pair_indices" in data.files else None,
            target_class=args.target_class,
            batch_index=int(args.batch_index),
            rdp_epsilon=float(args.rdp_epsilon),
            base_archive_bytes=int(args.base_archive_bytes),
            old_d_seg=float(args.old_d_seg),
            old_d_pose=float(args.old_d_pose),
        ))
    if args.pose_trajectory_json is not None:
        pose_path = args.pose_trajectory_json.expanduser().resolve(strict=True)
        pose_payload = _load_json_or_jsonl(pose_path)
        if not isinstance(pose_payload, dict):
            raise SystemExit("--pose-trajectory-json must contain a JSON object")
        results.append(
            build_pose_temporal_path_candidates_from_arrays(
                pair_indices=pose_payload["pair_indices"],
                pose_residuals=pose_payload.get("pose_residuals"),
                pose_action_profile=pose_payload.get("pose_action_profile"),
                rdp_epsilon=float(pose_payload.get("rdp_epsilon", args.rdp_epsilon)),
                base_archive_bytes=int(pose_payload.get("base_archive_bytes", args.base_archive_bytes)),
                old_d_seg=pose_payload.get("old_d_seg", args.old_d_seg),
                new_d_seg=pose_payload.get("new_d_seg", args.old_d_seg),
                old_d_pose=pose_payload.get("old_d_pose", args.old_d_pose),
                new_d_pose=pose_payload.get("new_d_pose", args.old_d_pose),
            )
        )
    if args.selector_rows_json is not None:
        selector_path = args.selector_rows_json.expanduser().resolve(strict=True)
        selector_payload = _load_json_or_jsonl(selector_path)
        selector_base_bytes = int(args.base_archive_bytes)
        if isinstance(selector_payload, dict):
            selector_base_bytes = int(selector_payload.get("base_archive_bytes", selector_base_bytes))
        results.append(
            build_selector_temporal_path_candidates_from_rows(
                selector_payload,
                rdp_epsilon=0.0,
                base_archive_bytes=selector_base_bytes,
                old_d_seg=float(args.old_d_seg),
                new_d_seg=float(args.old_d_seg),
                old_d_pose=float(args.old_d_pose),
                new_d_pose=float(args.old_d_pose),
            )
        )

    effects = [effect for result in results for effect in result.get("action_effects", [])]
    rows = [effect.as_dict() for effect in effects]
    action_rows_path = out_dir / "action_effect_rows.jsonl"
    action_rows_path.unlink(missing_ok=True)
    for effect in effects:
        append_action_effect(effect, action_rows_path)
    candidate_queue = [dict(row) for result in results for row in result.get("candidate_queue", [])]
    candidate_queue_path = out_dir / "candidate_queue.jsonl"
    path_candidates_path = out_dir / "path_action_candidates.jsonl"
    _write_jsonl(candidate_queue_path, candidate_queue)
    path_candidates = [dict(row) for result in results for row in result.get("path_action_candidates", [])]
    _write_jsonl(path_candidates_path, path_candidates)
    comparisons = [result.get("comparison", {}) for result in results if result.get("comparison")]
    blockers = [blocker for result in results for blocker in result.get("blockers", [])]

    summary = {
        "schema": PATH_ACTION_PRODUCER_SCHEMA,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "hard_region_npz": npz_path.as_posix(),
        "output_dir": out_dir.as_posix(),
        "action_effect_rows_path": action_rows_path.as_posix(),
        "candidate_queue_path": candidate_queue_path.as_posix(),
        "path_action_candidates_path": path_candidates_path.as_posix(),
        "action_effect_row_count": len(rows),
        "candidate_queue_row_count": len(candidate_queue),
        "path_action_candidate_count": len(path_candidates),
        "path_action_kinds": sorted({str(row.get("action_kind")) for row in path_candidates}),
        "comparison": comparisons[0] if len(comparisons) == 1 else comparisons,
        "blockers": blockers,
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _write_json(out_dir / "summary.json", summary)
    next_blocker = out_dir / "next_blocker.md"
    first_blocker = "path_action_parseback_survival_missing"
    next_blocker.write_text(
        "\n".join(
            [
                "# PathActionProducer",
                "",
                f"- hard_region_npz: `{npz_path.as_posix()}`",
                f"- action_effect_rows: `{len(rows)}`",
                f"- candidate_queue_rows: `{len(candidate_queue)}`",
                "",
                "## First Blocker",
                "",
                f"`{first_blocker}`",
                "",
                "Path rows are archive-support/action candidates only. They do not clear launch, score, rank, or promotion without same-action receiver proof, parse-back, and inflate survival.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
