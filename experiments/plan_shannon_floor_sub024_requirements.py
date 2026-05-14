#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan byte and distortion requirements for sub-0.24 frontier targets.

This is a control-plane planner, not score evidence. It consumes exact CUDA
eval JSON plus an empirical archive byte profile, then writes the rate-
distortion constraints that any sub-0.30/sub-0.24 lane must satisfy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_WEIGHT = 25.0
SCHEMA = "shannon_floor_sub024_requirements_v1"
TOOL = "experiments/plan_shannon_floor_sub024_requirements.py"
DEFAULT_EVAL_JSON = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    / "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/"
    / "contest_auth_eval.adjudicated.json"
)
DEFAULT_BYTE_PROFILE = (
    REPO_ROOT
    / "experiments/results/c067_archive_byte_accounting_20260502/"
    / "archive_byte_accounting.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/shannon_floor_sub024_requirements_20260502"
DEFAULT_TARGETS = (0.30, 0.24, 0.20)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rate_score_for_bytes(archive_bytes: int) -> float:
    return RATE_WEIGHT * float(archive_bytes) / float(ORIGINAL_VIDEO_BYTES)


def bytes_for_rate_score(rate_score: float) -> int:
    return int(math.floor(rate_score * float(ORIGINAL_VIDEO_BYTES) / RATE_WEIGHT))


def _float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    return None


def exact_component_record(eval_json: dict[str, Any]) -> dict[str, Any]:
    archive_bytes = int(eval_json["archive_size_bytes"])
    seg_dist = float(eval_json["avg_segnet_dist"])
    pose_dist = float(eval_json["avg_posenet_dist"])
    seg_score = _float(eval_json, "score_seg_contribution")
    if seg_score is None:
        seg_score = 100.0 * seg_dist
    pose_score = _float(eval_json, "score_pose_contribution")
    if pose_score is None:
        pose_score = math.sqrt(10.0 * pose_dist)
    reported_rate_score = _float(eval_json, "score_rate_contribution")
    rate_score = rate_score_for_bytes(archive_bytes)
    exact_score = _float(eval_json, "score_recomputed_from_components", "final_score")
    if exact_score is None:
        exact_score = seg_score + pose_score + rate_score
    distortion_score = seg_score + pose_score
    return {
        "score": exact_score,
        "archive_bytes": archive_bytes,
        "seg_dist": seg_dist,
        "pose_dist": pose_dist,
        "seg_score": seg_score,
        "pose_score": pose_score,
        "distortion_score": distortion_score,
        "rate_score": rate_score,
        "reported_rate_score": reported_rate_score,
        "reported_rate_score_delta": None if reported_rate_score is None else rate_score - reported_rate_score,
        "no_distortion_floor_at_current_bytes": rate_score,
        "archive_sha256": eval_json.get("provenance", {}).get("archive_sha256"),
        "gpu_model": eval_json.get("provenance", {}).get("gpu_model"),
        "gpu_t4_match": eval_json.get("provenance", {}).get("gpu_t4_match"),
        "n_samples": eval_json.get("n_samples"),
    }


def target_requirements(component: dict[str, Any], target: float) -> dict[str, Any]:
    archive_bytes = int(component["archive_bytes"])
    distortion_score = float(component["distortion_score"])
    rate_score = float(component["rate_score"])
    required_distortion_at_current_bytes = target - rate_score
    distortion_reduction_needed_at_current_bytes = distortion_score - required_distortion_at_current_bytes
    unchanged_distortion_rate_budget = target - distortion_score
    if unchanged_distortion_rate_budget >= 0.0:
        max_bytes_if_distortion_unchanged = bytes_for_rate_score(unchanged_distortion_rate_budget)
        bytes_to_remove_if_distortion_unchanged = max(0, archive_bytes - max_bytes_if_distortion_unchanged)
    else:
        max_bytes_if_distortion_unchanged = None
        bytes_to_remove_if_distortion_unchanged = None
    return {
        "target_score": target,
        "current_gap": float(component["score"]) - target,
        "required_distortion_score_at_current_bytes": required_distortion_at_current_bytes,
        "distortion_reduction_needed_at_current_bytes": distortion_reduction_needed_at_current_bytes,
        "feasible_by_distortion_only_at_current_bytes": required_distortion_at_current_bytes >= 0.0,
        "max_archive_bytes_if_distortion_unchanged": max_bytes_if_distortion_unchanged,
        "bytes_to_remove_if_distortion_unchanged": bytes_to_remove_if_distortion_unchanged,
        "feasible_by_rate_only_at_current_distortion": max_bytes_if_distortion_unchanged is not None,
    }


def _stream_map(byte_profile: dict[str, Any]) -> dict[str, int]:
    streams: dict[str, int] = {}
    for stream in byte_profile.get("streams", []):
        name = stream.get("name") or stream.get("stream")
        if name:
            streams[str(name)] = int(stream["encoded_bytes"])
    return streams


def stream_budget_scenarios(
    component: dict[str, Any],
    byte_profile: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    streams = _stream_map(byte_profile)
    mask_bytes = streams.get("masks.mkv")
    renderer_bytes = streams.get("renderer.bin")
    pose_bytes = streams.get("optimized_poses.bin")
    scenarios: list[dict[str, Any]] = []
    for requirement in requirements:
        bytes_to_remove = requirement["bytes_to_remove_if_distortion_unchanged"]
        target = requirement["target_score"]
        if bytes_to_remove is None:
            scenarios.append(
                {
                    "target_score": target,
                    "scenario": "rate_only_impossible_at_current_distortion",
                    "reason": "current distortion score already exceeds target",
                }
            )
            continue
        if mask_bytes is not None:
            scenarios.append(
                {
                    "target_score": target,
                    "scenario": "mask_only_at_current_distortion",
                    "required_mask_savings_bytes": bytes_to_remove,
                    "target_mask_bytes": max(0, mask_bytes - bytes_to_remove),
                    "required_mask_savings_fraction": bytes_to_remove / mask_bytes,
                    "stream_can_cover": bytes_to_remove <= mask_bytes,
                }
            )
        if renderer_bytes is not None:
            scenarios.append(
                {
                    "target_score": target,
                    "scenario": "renderer_only_at_current_distortion",
                    "required_renderer_savings_bytes": bytes_to_remove,
                    "target_renderer_bytes": max(0, renderer_bytes - bytes_to_remove),
                    "required_renderer_savings_fraction": bytes_to_remove / renderer_bytes,
                    "stream_can_cover": bytes_to_remove <= renderer_bytes,
                }
            )
        if mask_bytes is not None and renderer_bytes is not None:
            for renderer_target in (20_000, 11_000):
                renderer_savings = max(0, renderer_bytes - renderer_target)
                remaining_mask_savings = max(0, bytes_to_remove - renderer_savings)
                scenarios.append(
                    {
                        "target_score": target,
                        "scenario": f"renderer_to_{renderer_target}_then_mask",
                        "renderer_target_bytes": renderer_target,
                        "renderer_savings_bytes": renderer_savings,
                        "remaining_mask_savings_bytes": remaining_mask_savings,
                        "target_mask_bytes": max(0, mask_bytes - remaining_mask_savings),
                        "remaining_mask_savings_fraction": remaining_mask_savings / mask_bytes,
                        "stream_can_cover": remaining_mask_savings <= mask_bytes,
                    }
                )
        if pose_bytes is not None:
            scenarios.append(
                {
                    "target_score": target,
                    "scenario": "pose_stream_upper_bound",
                    "pose_stream_bytes": pose_bytes,
                    "pose_stream_score_span": rate_score_for_bytes(pose_bytes),
                    "can_matter_alone": bytes_to_remove <= pose_bytes,
                }
            )
    return scenarios


def decision_record(component: dict[str, Any], requirements: list[dict[str, Any]]) -> dict[str, Any]:
    sub024 = next(item for item in requirements if abs(item["target_score"] - 0.24) < 1e-12)
    return {
        "verdict": "sub_024_requires_large_representation_change",
        "why": (
            "At C067 distortion, sub-0.24 requires roughly six figures of archive-byte removal; "
            "at C067 bytes, it requires a large SegNet/PoseNet distortion reduction. "
            "Pose/SJ-KL micro-lanes are useful polish only unless fused into a bigger payload."
        ),
        "sub024_bytes_to_remove_if_distortion_unchanged": sub024["bytes_to_remove_if_distortion_unchanged"],
        "sub024_distortion_reduction_needed_at_current_bytes": sub024[
            "distortion_reduction_needed_at_current_bytes"
        ],
        "nonnegotiable_next_lanes": [
            {
                "lane": "geometry_preserving_mask_topology_or_learned_decoder",
                "reason": "masks.mkv is the only single stream large enough to carry sub-0.24 alone",
                "minimum_evidence": "exact CUDA archive eval after deterministic archive build",
            },
            {
                "lane": "real_non_surrogate_trained_renderer_self_compression",
                "reason": "renderer-to-11KB can buy about 45KB but cannot reach sub-0.24 alone",
                "minimum_evidence": "non-surrogate export preflight plus exact CUDA archive eval",
            },
            {
                "lane": "stacked_mask_renderer_distortion_waterfill",
                "reason": "sub-0.24 likely needs mask bytes plus renderer bytes plus small positive atoms",
                "minimum_evidence": "stacked archive exact eval; additive deltas are not score claims",
            },
        ],
    }


def build_plan(eval_json_path: Path, byte_profile_path: Path, targets: tuple[float, ...]) -> dict[str, Any]:
    eval_json = _load_json(eval_json_path)
    byte_profile = _load_json(byte_profile_path)
    component = exact_component_record(eval_json)
    requirements = [target_requirements(component, target) for target in targets]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": "derivation_plus_exact_frontier_inputs",
        "score_truth_boundary": "derived from exact CUDA eval inputs; this planner itself is not a score claim",
        "inputs": {
            "eval_json": str(eval_json_path),
            "eval_json_sha256": _sha256_file(eval_json_path),
            "byte_profile": str(byte_profile_path),
            "byte_profile_sha256": _sha256_file(byte_profile_path),
        },
        "constants": {
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_weight": RATE_WEIGHT,
        },
        "frontier": component,
        "target_requirements": requirements,
        "stream_budget_scenarios": stream_budget_scenarios(component, byte_profile, requirements),
        "decision": decision_record(component, requirements),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--byte-profile", type=Path, default=DEFAULT_BYTE_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = tuple(float(target) for target in args.targets)
    if not targets:
        raise SystemExit("--targets must contain at least one target score")
    plan = build_plan(args.eval_json, args.byte_profile, targets)
    output_path = args.output_dir / "shannon_floor_sub024_requirements.json"
    _json_dump(output_path, plan)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
