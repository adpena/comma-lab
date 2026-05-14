#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan PR79/C102 pose-action interaction atoms from existing evidence.

This is a planning-only analyzer. It consumes exact component traces plus
public qpose-family byte/action/pose profiles and emits machine-readable atom
rankings. It does not build archives, dispatch GPU work, or claim score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_pr79_c102_pose_action_interactions.py"
SCHEMA = "pr79_c102_pose_action_interaction_policy_v1"

RATE_DENOMINATOR_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / RATE_DENOMINATOR_BYTES
TARGET_SCORE = 0.31
PR79_FRONTIER_SCORE = 0.31457805357318636
PR79_FRONTIER_BYTES = 277_388
PR79_FRONTIER_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr79_c102_pose_action_interactions_20260503_codex"
)
DEFAULT_PR79_TRACE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_pr79_minp_v2_public_replay_t4_20260503T1615Z/component_trace.json"
)
DEFAULT_C102_TRACE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/"
    "component_trace.json"
)
DEFAULT_FAMILY_PROFILE = (
    REPO_ROOT / "experiments/results/pr79_archive_binary_forensics_20260503_worker/family_profile.json"
)
DEFAULT_ACTION_DIFF_CSV = REPO_ROOT / (
    "experiments/results/pr79_archive_binary_forensics_20260503_worker/"
    "action_record_multiset_union_diff.csv"
)
DEFAULT_PR79_ACTION_CSV = REPO_ROOT / (
    "experiments/results/pr79_archive_binary_forensics_20260503_worker/action_records_pr79.csv"
)
DEFAULT_C102_ACTION_CSV = REPO_ROOT / (
    "experiments/results/pr79_archive_binary_forensics_20260503_worker/action_records_c102.csv"
)
DEFAULT_POSE_DIFF_CSV = REPO_ROOT / (
    "experiments/results/pr79_archive_binary_forensics_20260503_worker/"
    "pose_qp1_q0_word_diff.csv"
)


@dataclass(frozen=True)
class ActionRecord:
    index: int
    pair_index: int
    tile_id: int
    action_id: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.pair_index, self.tile_id, self.action_id)

    @property
    def atom_id(self) -> str:
        return f"pair{self.pair_index:03d}_tile{self.tile_id:03d}_act{self.action_id:03d}"


@dataclass(frozen=True)
class TraceSample:
    pair_index: int
    frame_indices: tuple[int, ...]
    video_name: str | None
    seg_score: float
    pose_score: float
    combined_score: float
    segnet_dist: float | None
    posenet_dist: float | None


@dataclass(frozen=True)
class LoadedTrace:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str | None
    score: float
    seg_score: float
    pose_score: float
    samples: dict[int, TraceSample]


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _finite(value: Any, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


def _load_trace(path: Path, *, label: str) -> LoadedTrace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"{path} has no component-trace samples")
    samples: dict[int, TraceSample] = {}
    for raw in raw_samples:
        pair = int(raw["pair_index"])
        if pair in samples:
            raise ValueError(f"{path} duplicate pair_index={pair}")
        frames = raw.get("frame_indices") or ()
        samples[pair] = TraceSample(
            pair_index=pair,
            frame_indices=tuple(int(item) for item in frames),
            video_name=raw.get("video_name"),
            seg_score=_finite(raw["score_seg_contribution_exact"], field="seg score"),
            pose_score=_finite(
                raw["score_pose_contribution_first_order"],
                field="pose score",
            ),
            combined_score=_finite(
                raw["score_combined_contribution_first_order"],
                field="combined score",
            ),
            segnet_dist=(
                _finite(raw["segnet_dist"], field="segnet_dist")
                if raw.get("segnet_dist") is not None
                else None
            ),
            posenet_dist=(
                _finite(raw["posenet_dist"], field="posenet_dist")
                if raw.get("posenet_dist") is not None
                else None
            ),
        )
    trace_inputs = payload.get("trace_inputs") if isinstance(payload, Mapping) else None
    archive_sha = trace_inputs.get("archive_sha256") if isinstance(trace_inputs, Mapping) else None
    return LoadedTrace(
        label=label,
        path=path,
        archive_bytes=int(payload["archive_size_bytes"]),
        archive_sha256=str(archive_sha) if archive_sha else None,
        score=_finite(
            payload["score_recomputed_from_components"],
            field="score_recomputed_from_components",
        ),
        seg_score=_finite(payload["score_seg_contribution"], field="score seg"),
        pose_score=_finite(payload["score_pose_contribution"], field="score pose"),
        samples=samples,
    )


def _pair_delta_rows(pr79: LoadedTrace, c102: LoadedTrace) -> dict[int, dict[str, Any]]:
    missing = sorted(set(c102.samples) - set(pr79.samples))
    if missing:
        raise ValueError(f"PR79 trace missing C102 pairs: {missing[:8]}")
    rows: dict[int, dict[str, Any]] = {}
    for pair, c102_sample in c102.samples.items():
        pr79_sample = pr79.samples[pair]
        seg_excess = pr79_sample.seg_score - c102_sample.seg_score
        pose_excess = pr79_sample.pose_score - c102_sample.pose_score
        combined_excess = seg_excess + pose_excess
        rows[pair] = {
            "pair_index": pair,
            "frame_indices": list(pr79_sample.frame_indices),
            "video_name": pr79_sample.video_name,
            "pr79_seg_score": pr79_sample.seg_score,
            "pr79_pose_score": pr79_sample.pose_score,
            "pr79_combined_score": pr79_sample.combined_score,
            "c102_seg_score": c102_sample.seg_score,
            "c102_pose_score": c102_sample.pose_score,
            "c102_combined_score": c102_sample.combined_score,
            "seg_excess_vs_c102": seg_excess,
            "pose_excess_vs_c102": pose_excess,
            "combined_excess_vs_c102": combined_excess,
            "component_repair_benefit_proxy": max(0.0, combined_excess),
            "pose_repair_benefit_proxy": max(0.0, pose_excess),
            "seg_repair_benefit_proxy": max(0.0, seg_excess),
        }
    return rows


def _load_action_records(path: Path) -> list[ActionRecord]:
    records: list[ActionRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"index", "pair", "tile", "action"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} missing required columns {sorted(required)}")
        for raw in reader:
            records.append(
                ActionRecord(
                    index=int(raw["index"]),
                    pair_index=int(raw["pair"]),
                    tile_id=int(raw["tile"]),
                    action_id=int(raw["action"]),
                )
            )
    return records


def _load_action_diff(path: Path) -> dict[tuple[int, int, int], dict[str, int]]:
    out: dict[tuple[int, int, int], dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            key = (int(raw["pair"]), int(raw["tile"]), int(raw["action"]))
            out[key] = {
                key_name: int(value)
                for key_name, value in raw.items()
                if key_name.startswith("count_") and value not in (None, "")
            }
    return out


def _load_pose_diff_rows(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = int(raw["row"])
            c102 = int(raw["c102"]) if raw.get("c102") not in (None, "") else None
            pr79 = int(raw["pr79"]) if raw.get("pr79") not in (None, "") else None
            all_equal = str(raw.get("all_equal", "")).lower() == "true"
            out[row] = {
                "row": row,
                "c102_q0": c102,
                "pr79_q0": pr79,
                "q0_delta_pr79_minus_c102": (
                    pr79 - c102 if pr79 is not None and c102 is not None else None
                ),
                "all_equal": all_equal,
                "differs_pr79_vs_c102": (not all_equal) and pr79 != c102,
            }
    return out


def _family_stream_summary(family_profile: Path) -> dict[str, Any]:
    payload = json.loads(family_profile.read_text(encoding="utf-8"))
    archives = {archive["label"]: archive for archive in payload.get("archives", [])}
    def stream_bytes(label: str, stream_name: str) -> int:
        return int(archives[label]["decoded_streams"][stream_name]["charged_bytes"])

    return {
        "path": _repo_rel(family_profile),
        "sha256": _sha256_file(family_profile),
        "schema": payload.get("schema"),
        "score_claim": payload.get("score_claim"),
        "archives": {
            label: {
                "archive_bytes": int(archive["archive"]["bytes"]),
                "archive_sha256": archive["archive"]["sha256"],
                "action_charged_bytes": int(
                    archive["decoded_streams"]["seg_tile_actions.bin"]["charged_bytes"]
                ),
                "pose_charged_bytes": int(
                    archive["decoded_streams"]["optimized_poses.qp1"]["charged_bytes"]
                ),
                "action_record_count": int(archive["actions"]["record_count"]),
                "unique_action_pair_count": int(archive["actions"]["unique_pair_count"]),
            }
            for label, archive in sorted(archives.items())
        },
        "action_stream_delta_pr79_vs_c102": stream_bytes("pr79", "seg_tile_actions.bin")
        - stream_bytes("c102", "seg_tile_actions.bin"),
        "pose_stream_delta_pr79_vs_c102": stream_bytes("pr79", "optimized_poses.qp1")
        - stream_bytes("c102", "optimized_poses.qp1"),
        "action_pairwise_diffs": payload.get("action_pairwise_diffs", {}),
        "pose_diffs": payload.get("pose_diffs", {}),
    }


def _break_even(byte_delta_vs_pr79: float, *, benefit_proxy: float = 0.0) -> dict[str, Any]:
    rate_delta = byte_delta_vs_pr79 * RATE_SCORE_PER_BYTE
    score_after_rate = PR79_FRONTIER_SCORE + rate_delta
    required = max(0.0, score_after_rate - TARGET_SCORE)
    return {
        "target_score": TARGET_SCORE,
        "frontier_score": PR79_FRONTIER_SCORE,
        "byte_delta_vs_pr79": byte_delta_vs_pr79,
        "rate_score_delta_vs_pr79": rate_delta,
        "score_if_components_unchanged": score_after_rate,
        "required_component_gain_to_reach_target": required,
        "component_benefit_proxy": benefit_proxy,
        "benefit_minus_required_component_gain": benefit_proxy - required,
        "target_equivalent_bytes_remaining": (
            math.ceil(max(0.0, required - benefit_proxy) / RATE_SCORE_PER_BYTE)
        ),
    }


def _count_by_pair(records: Iterable[ActionRecord]) -> dict[int, int]:
    out: dict[int, int] = {}
    for record in records:
        out[record.pair_index] = out.get(record.pair_index, 0) + 1
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _action_atom_rows(
    *,
    pr79_records: Sequence[ActionRecord],
    c102_records: Sequence[ActionRecord],
    action_diff: Mapping[tuple[int, int, int], Mapping[str, int]],
    pair_rows: Mapping[int, Mapping[str, Any]],
    action_extra_byte_proxy: float,
) -> list[dict[str, Any]]:
    c102_keys = {record.key for record in c102_records}
    pr79_only_by_pair = _count_by_pair(
        record for record in pr79_records if record.key not in c102_keys
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for record in pr79_records:
        if record.key in seen:
            continue
        seen.add(record.key)
        pair = pair_rows.get(record.pair_index)
        if pair is None:
            continue
        counts = action_diff.get(record.key, {})
        no_op = record.key in c102_keys
        share_count = max(1, pr79_only_by_pair.get(record.pair_index, 0))
        benefit = 0.0 if no_op else float(pair["component_repair_benefit_proxy"]) / share_count
        pose_benefit = 0.0 if no_op else float(pair["pose_repair_benefit_proxy"]) / share_count
        seg_benefit = 0.0 if no_op else float(pair["seg_repair_benefit_proxy"]) / share_count
        byte_delta = 0.0 if no_op else -action_extra_byte_proxy
        row = {
            "atom_id": f"action_{record.atom_id}_pr79_to_c102_context",
            "atom_type": "action",
            "pair_index": record.pair_index,
            "frame_indices": pair["frame_indices"],
            "tile_id": record.tile_id,
            "action_id": record.action_id,
            "pose_atom_id": None,
            "byte_cost_proxy": byte_delta,
            "component_benefit_proxy": benefit,
            "pose_benefit_proxy": pose_benefit,
            "seg_benefit_proxy": seg_benefit,
            "break_even_score": _break_even(byte_delta, benefit_proxy=benefit),
            "confidence": 0.55 if (not no_op and benefit > 0.0) else 0.35,
            "no_op": no_op,
            "no_op_status": "exact_c102_action_duplicate" if no_op else "pr79_only_action_record",
            "count_c102": int(counts.get("count_c102", 0)),
            "count_pr75": int(counts.get("count_pr75", 0)),
            "count_pr77": int(counts.get("count_pr77", 0)),
            "count_pr79": int(counts.get("count_pr79", 0)),
            "recommended_archive_builder_inputs": (
                {
                    "builder": "experiments/build_pr79_action_subset_candidates.py",
                    "policy_hint": "drop_or_replace_pr79_only_pose_toxic_records",
                    "record": {
                        "pair_index": record.pair_index,
                        "tile_id": record.tile_id,
                        "action_id": record.action_id,
                    },
                    "dispatch_after_build": False,
                }
                if not no_op and benefit > 0.0
                else None
            ),
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            bool(item["no_op"]),
            -float(item["component_benefit_proxy"]),
            float(item["byte_cost_proxy"]),
            item["pair_index"],
            item["tile_id"],
            item["action_id"],
        ),
    )


def _pose_atom_rows(
    *,
    pose_diff_rows: Mapping[int, Mapping[str, Any]],
    pair_rows: Mapping[int, Mapping[str, Any]],
    pose_extra_byte_proxy: float,
) -> list[dict[str, Any]]:
    differing = [row for row in pose_diff_rows.values() if row["differs_pr79_vs_c102"]]
    share = pose_extra_byte_proxy / max(1, len(differing))
    rows: list[dict[str, Any]] = []
    for raw in differing:
        pair = pair_rows.get(int(raw["row"]))
        if pair is None:
            continue
        benefit = float(pair["pose_repair_benefit_proxy"])
        byte_delta = -share
        atom_id = f"pose_qp1_row{int(raw['row']):03d}_pr79q0_{raw['pr79_q0']}_to_c102q0_{raw['c102_q0']}"
        rows.append(
            {
                "atom_id": atom_id,
                "atom_type": "pose",
                "pair_index": int(raw["row"]),
                "frame_indices": pair["frame_indices"],
                "tile_id": None,
                "action_id": None,
                "pose_atom_id": atom_id,
                "byte_cost_proxy": byte_delta,
                "component_benefit_proxy": benefit,
                "pose_benefit_proxy": benefit,
                "seg_benefit_proxy": 0.0,
                "break_even_score": _break_even(byte_delta, benefit_proxy=benefit),
                "confidence": 0.7 if benefit > 0.0 else 0.45,
                "no_op": False,
                "no_op_status": "qp1_q0_row_differs_pr79_vs_c102",
                "q0_delta_pr79_minus_c102": raw["q0_delta_pr79_minus_c102"],
                "recommended_archive_builder_inputs": (
                    {
                        "builder": "pose-stream builder needed",
                        "policy_hint": "replace_or_refit_pr79_qp1_rows_toward_c102",
                        "row": int(raw["row"]),
                        "dispatch_after_build": False,
                    }
                    if benefit > 0.0
                    else None
                ),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -float(item["component_benefit_proxy"]),
            abs(float(item.get("q0_delta_pr79_minus_c102") or 0)),
            item["pair_index"],
        ),
    )


def _pair_atom_rows(
    *,
    pair_rows: Mapping[int, Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    pose_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    action_counts = _count_by_pair(
        ActionRecord(
            index=idx,
            pair_index=int(row["pair_index"]),
            tile_id=int(row["tile_id"] or 0),
            action_id=int(row["action_id"] or 0),
        )
        for idx, row in enumerate(action_rows)
        if not row["no_op"]
    )
    pose_pairs = {int(row["pair_index"]) for row in pose_rows}
    rows: list[dict[str, Any]] = []
    for pair, raw in pair_rows.items():
        benefit = float(raw["component_repair_benefit_proxy"])
        action_count = int(action_counts.get(pair, 0))
        has_pose = pair in pose_pairs
        no_op = action_count == 0 and not has_pose
        atom_id = f"pair{pair:03d}_frames_{'-'.join(map(str, raw['frame_indices'])) or 'unknown'}"
        rows.append(
            {
                "atom_id": atom_id,
                "atom_type": "pair_frame",
                "pair_index": pair,
                "frame_indices": raw["frame_indices"],
                "tile_id": None,
                "action_id": None,
                "pose_atom_id": f"pose_qp1_row{pair:03d}" if has_pose else None,
                "byte_cost_proxy": 0.0,
                "component_benefit_proxy": benefit,
                "pose_benefit_proxy": raw["pose_repair_benefit_proxy"],
                "seg_benefit_proxy": raw["seg_repair_benefit_proxy"],
                "break_even_score": _break_even(0.0, benefit_proxy=benefit),
                "confidence": 0.85 if benefit > 0.0 else 0.5,
                "no_op": no_op,
                "no_op_status": (
                    "no_pr79_action_or_pose_delta_for_pair"
                    if no_op
                    else "pair_has_pr79_action_or_pose_delta"
                ),
                "pr79_only_action_record_count": action_count,
                "pose_qp1_differs": has_pose,
                "recommended_archive_builder_inputs": None,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -float(item["component_benefit_proxy"]),
            -float(item["pose_benefit_proxy"]),
            -int(item["pr79_only_action_record_count"]),
            item["pair_index"],
        ),
    )


def _interaction_rows(
    *,
    action_rows: Sequence[Mapping[str, Any]],
    pose_rows: Sequence[Mapping[str, Any]],
    pair_rows: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pose_by_pair = {int(row["pair_index"]): row for row in pose_rows}
    interactions: list[dict[str, Any]] = []
    for action in action_rows:
        if action["no_op"]:
            continue
        pair = int(action["pair_index"])
        pose = pose_by_pair.get(pair)
        if pose is None:
            continue
        pair_row = pair_rows[pair]
        benefit = max(
            float(pair_row["component_repair_benefit_proxy"]),
            float(action["component_benefit_proxy"]) + float(pose["component_benefit_proxy"]),
        )
        byte_delta = float(action["byte_cost_proxy"]) + float(pose["byte_cost_proxy"])
        atom_id = (
            f"interaction_pair{pair:03d}_tile{int(action['tile_id']):03d}_"
            f"act{int(action['action_id']):03d}_poseq0"
        )
        interactions.append(
            {
                "atom_id": atom_id,
                "atom_type": "pose_action_interaction",
                "pair_index": pair,
                "frame_indices": action["frame_indices"],
                "tile_id": action["tile_id"],
                "action_id": action["action_id"],
                "pose_atom_id": pose["atom_id"],
                "byte_cost_proxy": byte_delta,
                "component_benefit_proxy": benefit,
                "pose_benefit_proxy": pair_row["pose_repair_benefit_proxy"],
                "seg_benefit_proxy": pair_row["seg_repair_benefit_proxy"],
                "break_even_score": _break_even(byte_delta, benefit_proxy=benefit),
                "confidence": min(float(action["confidence"]), float(pose["confidence"]), 0.6),
                "no_op": False,
                "no_op_status": "pr79_only_action_and_qp1_row_differs",
                "recommended_archive_builder_inputs": {
                    "builder": "compose action subset builder plus pose-row/refit builder",
                    "policy_hint": "screen coupled pose-action repair atom locally before any eval",
                    "action_record": {
                        "pair_index": pair,
                        "tile_id": action["tile_id"],
                        "action_id": action["action_id"],
                    },
                    "pose_row": pair,
                    "dispatch_after_build": False,
                },
            }
        )
    return sorted(
        interactions,
        key=lambda item: (
            -float(item["component_benefit_proxy"]),
            float(item["byte_cost_proxy"]),
            item["pair_index"],
            item["tile_id"],
            item["action_id"],
        ),
    )


def _policy_rows(
    *,
    action_rows: Sequence[Mapping[str, Any]],
    pose_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    policies: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = [
        (
            "pr79_pose_action_top16_interactions",
            "Top coupled PR79-only action plus QP1-row repair atoms.",
            interaction_rows[:16],
        ),
        (
            "pr79_pose_row_top32_repair",
            "Top PR79 pose-excess QP1 rows, independent of action attribution.",
            pose_rows[:32],
        ),
        (
            "pr79_action_drop_top64_pose_toxic",
            "Top PR79-only action records on pairs where PR79 is worse than C102.",
            [row for row in action_rows if not row["no_op"] and row["component_benefit_proxy"] > 0.0][:64],
        ),
    ]
    out: list[dict[str, Any]] = []
    for policy_id, rationale, atoms in policies:
        byte_delta = sum(float(atom["byte_cost_proxy"]) for atom in atoms)
        benefit = sum(float(atom["component_benefit_proxy"]) for atom in atoms)
        out.append(
            {
                "candidate_policy_id": policy_id,
                "rationale": rationale,
                "atom_ids": [str(atom["atom_id"]) for atom in atoms],
                "records_selected_count": len(atoms),
                "charged_byte_proxy": {
                    "byte_delta_vs_pr79": byte_delta,
                    "method": "summed_stream_delta_proxy_not_builder_bytes",
                },
                "expected_component_benefit_proxy": benefit,
                "break_even_vs_0_31": _break_even(byte_delta, benefit_proxy=benefit),
                "dispatchable": False,
                "dispatchable_reason": (
                    "planning-only: proxy atoms require local archive construction, "
                    "raw-output/parity checks, lane claim, and exact CUDA auth eval before any score use"
                ),
                "no_op_status": "empty_policy" if not atoms else "non_noop_proxy_atoms",
                "recommended_archive_builder_inputs": {
                    "action_records": [
                        {
                            "pair_index": atom["pair_index"],
                            "tile_id": atom.get("tile_id"),
                            "action_id": atom.get("action_id"),
                        }
                        for atom in atoms
                        if atom["atom_type"] in {"action", "pose_action_interaction"}
                    ],
                    "pose_rows": [
                        atom["pair_index"]
                        for atom in atoms
                        if atom["atom_type"] in {"pose", "pose_action_interaction"}
                    ],
                    "score_claim": False,
                    "dispatch_after_build": False,
                },
            }
        )
    return sorted(
        out,
        key=lambda row: (
            float(row["expected_component_benefit_proxy"])
            - float(row["break_even_vs_0_31"]["required_component_gain_to_reach_target"]),
            -int(row["records_selected_count"]),
        ),
        reverse=True,
    )


def build_plan(
    *,
    output_dir: Path,
    pr79_trace_path: Path = DEFAULT_PR79_TRACE,
    c102_trace_path: Path = DEFAULT_C102_TRACE,
    family_profile_path: Path = DEFAULT_FAMILY_PROFILE,
    action_diff_csv: Path = DEFAULT_ACTION_DIFF_CSV,
    pr79_action_csv: Path = DEFAULT_PR79_ACTION_CSV,
    c102_action_csv: Path = DEFAULT_C102_ACTION_CSV,
    pose_diff_csv: Path = DEFAULT_POSE_DIFF_CSV,
    top_k: int = 80,
) -> dict[str, Any]:
    pr79 = _load_trace(pr79_trace_path, label="pr79_exact_t4")
    c102 = _load_trace(c102_trace_path, label="c102_exact_t4")
    pair_rows = _pair_delta_rows(pr79, c102)
    family = _family_stream_summary(family_profile_path)
    pr79_records = _load_action_records(pr79_action_csv)
    c102_records = _load_action_records(c102_action_csv)
    action_diff = _load_action_diff(action_diff_csv)
    pose_diff = _load_pose_diff_rows(pose_diff_csv)

    action_pairwise = family.get("action_pairwise_diffs", {}).get("c102_vs_pr79", {})
    pr79_only_count = max(1, int(action_pairwise.get("right_only_record_count", 0)))
    action_extra = max(0.0, float(family["action_stream_delta_pr79_vs_c102"])) / pr79_only_count
    pose_diff_count = max(
        1,
        sum(1 for row in pose_diff.values() if row["differs_pr79_vs_c102"]),
    )
    pose_extra = max(0.0, float(family["pose_stream_delta_pr79_vs_c102"])) / pose_diff_count

    action_atoms = _action_atom_rows(
        pr79_records=pr79_records,
        c102_records=c102_records,
        action_diff=action_diff,
        pair_rows=pair_rows,
        action_extra_byte_proxy=action_extra,
    )
    pose_atoms = _pose_atom_rows(
        pose_diff_rows=pose_diff,
        pair_rows=pair_rows,
        pose_extra_byte_proxy=pose_extra,
    )
    pair_atoms = _pair_atom_rows(
        pair_rows=pair_rows,
        action_rows=action_atoms,
        pose_rows=pose_atoms,
    )
    interaction_atoms = _interaction_rows(
        action_rows=action_atoms,
        pose_rows=pose_atoms,
        pair_rows=pair_rows,
    )
    policies = _policy_rows(
        action_rows=action_atoms,
        pose_rows=pose_atoms,
        interaction_rows=interaction_atoms,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_exact_trace_planning_no_score_claim",
        "no_remote_dispatch_performed": True,
        "target_score": TARGET_SCORE,
        "frontier": {
            "label": "PR79 exact T4",
            "score": PR79_FRONTIER_SCORE,
            "archive_bytes": PR79_FRONTIER_BYTES,
            "archive_sha256": PR79_FRONTIER_SHA256,
            "component_trace_score": pr79.score,
            "component_trace_path": _repo_rel(pr79_trace_path),
        },
        "reference": {
            "label": "C102 exact T4",
            "score": c102.score,
            "archive_bytes": c102.archive_bytes,
            "archive_sha256": c102.archive_sha256,
            "component_trace_path": _repo_rel(c102_trace_path),
        },
        "global_delta_pr79_minus_c102": {
            "archive_bytes": pr79.archive_bytes - c102.archive_bytes,
            "score": pr79.score - c102.score,
            "seg_score": pr79.seg_score - c102.seg_score,
            "pose_score": pr79.pose_score - c102.pose_score,
            "rate_score": (pr79.archive_bytes - c102.archive_bytes) * RATE_SCORE_PER_BYTE,
        },
        "source_inputs": {
            "family_profile": family,
            "action_diff_csv": _repo_rel(action_diff_csv),
            "pr79_action_csv": _repo_rel(pr79_action_csv),
            "c102_action_csv": _repo_rel(c102_action_csv),
            "pose_diff_csv": _repo_rel(pose_diff_csv),
        },
        "byte_proxy_model": {
            "action_extra_byte_proxy_per_pr79_only_record": action_extra,
            "pose_extra_byte_proxy_per_qp1_diff_row": pose_extra,
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "note": "Byte costs are stream-delta proxies for planning, not builder output bytes.",
        },
        "dispatch_decision": {
            "exact_eval_justified": False,
            "next_command_if_yes": None,
            "reason": (
                "planning-only atom ranking; recommended inputs must first become byte-closed "
                "archives with local parity/no-op guards and lane-claim compliance"
            ),
        },
        "ranked_pair_atoms": pair_atoms[:top_k],
        "ranked_action_atoms": action_atoms[:top_k],
        "ranked_pose_atoms": pose_atoms[:top_k],
        "ranked_interaction_atoms": interaction_atoms[:top_k],
        "ranked_policy_inputs": policies,
        "atom_counts": {
            "pair_atoms": len(pair_atoms),
            "action_atoms": len(action_atoms),
            "pose_atoms": len(pose_atoms),
            "interaction_atoms": len(interaction_atoms),
            "policies": len(policies),
        },
    }
    _write_json(output_dir / "ranked_pose_action_atoms.json", payload)
    csv_fields = [
        "atom_id",
        "atom_type",
        "pair_index",
        "frame_indices",
        "tile_id",
        "action_id",
        "pose_atom_id",
        "byte_cost_proxy",
        "component_benefit_proxy",
        "pose_benefit_proxy",
        "seg_benefit_proxy",
        "confidence",
        "no_op",
        "no_op_status",
    ]
    _write_csv(output_dir / "ranked_pair_atoms.csv", pair_atoms[:top_k], csv_fields)
    _write_csv(output_dir / "ranked_action_atoms.csv", action_atoms[:top_k], csv_fields)
    _write_csv(output_dir / "ranked_pose_atoms.csv", pose_atoms[:top_k], csv_fields)
    _write_csv(
        output_dir / "ranked_interaction_atoms.csv",
        interaction_atoms[:top_k],
        csv_fields,
    )
    _write_csv(
        output_dir / "ranked_policy_inputs.csv",
        policies,
        [
            "candidate_policy_id",
            "records_selected_count",
            "expected_component_benefit_proxy",
            "no_op_status",
            "dispatchable",
            "dispatchable_reason",
        ],
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pr79-trace", type=Path, default=DEFAULT_PR79_TRACE)
    parser.add_argument("--c102-trace", type=Path, default=DEFAULT_C102_TRACE)
    parser.add_argument("--family-profile", type=Path, default=DEFAULT_FAMILY_PROFILE)
    parser.add_argument("--action-diff-csv", type=Path, default=DEFAULT_ACTION_DIFF_CSV)
    parser.add_argument("--pr79-action-csv", type=Path, default=DEFAULT_PR79_ACTION_CSV)
    parser.add_argument("--c102-action-csv", type=Path, default=DEFAULT_C102_ACTION_CSV)
    parser.add_argument("--pose-diff-csv", type=Path, default=DEFAULT_POSE_DIFF_CSV)
    parser.add_argument("--top-k", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_plan(
        output_dir=args.output_dir,
        pr79_trace_path=args.pr79_trace,
        c102_trace_path=args.c102_trace,
        family_profile_path=args.family_profile,
        action_diff_csv=args.action_diff_csv,
        pr79_action_csv=args.pr79_action_csv,
        c102_action_csv=args.c102_action_csv,
        pose_diff_csv=args.pose_diff_csv,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_dir / "ranked_pose_action_atoms.json"),
                "score_claim": payload["score_claim"],
                "top_interaction_atom": (
                    payload["ranked_interaction_atoms"][0]["atom_id"]
                    if payload["ranked_interaction_atoms"]
                    else None
                ),
                "top_policy": payload["ranked_policy_inputs"][0]["candidate_policy_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
