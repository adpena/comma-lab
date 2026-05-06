#!/usr/bin/env python3
"""Plan C067 micro AV1 mask reencode trust-region candidates.

This is a deterministic planning and byte-screen tool.  It reads a C067-style
component trace, an explicit protected-pair list, or both, then emits
trust-region reencode policies for local archive-building experiments.  It does
not load scorers, run GPU eval, build archives, or dispatch remote jobs.

Broad whole-mask CRF replacement is intentionally refused because the current
forensic record shows coarse mask AV1/CRF replacement can collapse PoseNet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "c067_micro_mask_reencode_plan_v1"
TOOL = "experiments/plan_c067_micro_mask_reencode.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)

C067_FRONTIER_SCORE = 0.31561703078448233
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_FRONTIER_ARCHIVE_SHA256 = "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
C067_MASK_STREAM_BYTES = 219_472
C067_MASK_MEMBER = "masks.mkv"

DEFAULT_TARGET_SAVINGS_BYTES = (5_000, 8_000, 12_000)
DEFAULT_PROBE_CRFS = (48, 50, 52, 54, 56)
DEFAULT_PROTECTED_CLASS_IDS = (1, 2, 3, 4)
DEFAULT_MASK_FRAME_COUNT = 600
DEFAULT_MASK_WIDTH = 512
DEFAULT_MASK_HEIGHT = 384
DEFAULT_TOP_POSE_PAIRS = 32
DEFAULT_TOP_SEG_PAIRS = 32
DEFAULT_TOP_COMBINED_PAIRS = 48
DEFAULT_MICRO_MAX_SAVINGS_BYTES = 12_000


class PlannerError(ValueError):
    """Raised when the requested plan would violate the micro trust-region contract."""


@dataclass(frozen=True)
class RegionSpec:
    """A rectangular decoded-mask region in 512x384 scorer mask coordinates."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    def with_frames(self, frames: tuple[int, ...]) -> dict[str, Any]:
        """Return the JSON shape consumed by protected-mask reencode policies."""

        return {
            "name": self.name,
            "x0": int(self.x0),
            "y0": int(self.y0),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "frames": [int(frame) for frame in frames],
        }


@dataclass(frozen=True)
class PairRecord:
    """Ranked protection evidence for one contest sample pair."""

    pair_index: int
    rank_score: float
    sources: tuple[str, ...]
    frame_indices: tuple[int, ...]
    posenet_dist: float | None = None
    segnet_dist: float | None = None
    combined_contribution: float | None = None
    explicit: bool = False

    def to_json(self) -> dict[str, Any]:
        """Serialize with deterministic key/value types."""

        return {
            "pair_index": int(self.pair_index),
            "rank_score": round(float(self.rank_score), 12),
            "sources": list(self.sources),
            "frame_indices": [int(value) for value in self.frame_indices],
            "posenet_dist": self.posenet_dist,
            "segnet_dist": self.segnet_dist,
            "score_combined_contribution_first_order": self.combined_contribution,
            "explicit": bool(self.explicit),
        }


@dataclass
class _PairAccumulator:
    pair_index: int
    score: float = 0.0
    sources: set[str] | None = None
    frame_indices: set[int] | None = None
    posenet_dist: float | None = None
    segnet_dist: float | None = None
    combined_contribution: float | None = None
    explicit: bool = False

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = set()
        if self.frame_indices is None:
            self.frame_indices = set()

    def add_sample(self, sample: dict[str, Any], *, source: str, score: float, explicit: bool = False) -> None:
        """Accumulate ranked evidence from one trace sample."""

        self.score += float(score)
        self.explicit = self.explicit or explicit
        self.sources.add(source)
        for frame in _sample_frame_indices(sample, pair_index=self.pair_index):
            self.frame_indices.add(frame)
        self.posenet_dist = _max_optional(self.posenet_dist, _optional_float(sample.get("posenet_dist")))
        self.segnet_dist = _max_optional(self.segnet_dist, _optional_float(sample.get("segnet_dist")))
        self.combined_contribution = _max_optional(
            self.combined_contribution,
            _optional_float(sample.get("score_combined_contribution_first_order")),
        )

    def record(self) -> PairRecord:
        """Freeze the accumulated signal for deterministic ranking/output."""

        return PairRecord(
            pair_index=self.pair_index,
            rank_score=self.score,
            sources=tuple(sorted(self.sources or ())),
            frame_indices=tuple(sorted(self.frame_indices or ())),
            posenet_dist=self.posenet_dist,
            segnet_dist=self.segnet_dist,
            combined_contribution=self.combined_contribution,
            explicit=self.explicit,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _max_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def _parse_int_set(value: str | None, *, field: str) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    parsed: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_text, sep, end_text = token.partition("-")
            if not sep or not start_text.isdigit() or not end_text.isdigit():
                raise argparse.ArgumentTypeError(f"{field} entries must be integers or ranges")
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"{field} range has end before start: {token!r}")
            parsed.update(range(start, end + 1))
        else:
            try:
                parsed.add(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"{field} entries must be integers or ranges") from exc
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{field} entries must be nonnegative")
    return tuple(sorted(parsed))


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = _parse_int_set(value, field="positive_ints")
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _parse_class_ids(value: str | None) -> tuple[int, ...]:
    parsed = _parse_int_set(value, field="protected_class_ids")
    if not parsed:
        return DEFAULT_PROTECTED_CLASS_IDS
    invalid = [class_id for class_id in parsed if class_id < 0 or class_id > 4]
    if invalid:
        raise argparse.ArgumentTypeError(f"protected class ids must be in [0,4], got {invalid}")
    return parsed


def _parse_region(value: str) -> RegionSpec:
    name = "custom_region"
    region_text = value
    if ":" in value and not value.split(":", 1)[0].strip().isdigit():
        name, region_text = value.split(":", 1)
        name = name.strip() or "custom_region"
    parts = [part.strip() for part in region_text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be x0,y0,x1,y1 or name:x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("region coordinates must be integers") from exc
    return RegionSpec(name=name, x0=x0, y0=y0, x1=x1, y1=y1)


def _default_regions() -> tuple[RegionSpec, ...]:
    return (
        RegionSpec(name="horizon_lane_band", x0=0, y0=96, x1=512, y1=224),
        RegionSpec(name="foveal_road_center", x0=144, y0=112, x1=368, y1=288),
        RegionSpec(name="ego_lower_road", x0=96, y0=208, x1=416, y1=384),
    )


def _validate_regions(regions: tuple[RegionSpec, ...], *, width: int, height: int) -> None:
    for region in regions:
        if not region.name:
            raise PlannerError("region names must be nonempty")
        if not (0 <= region.x0 < region.x1 <= width and 0 <= region.y0 < region.y1 <= height):
            raise PlannerError(
                f"region {region.name!r} is outside mask bounds "
                f"{width}x{height}: {(region.x0, region.y0, region.x1, region.y1)}"
            )


def _trace_input_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "component_trace",
        "path": str(path.resolve()),
        "sha256": _sha256_file(path.resolve()),
        "schema_version": payload.get("schema_version"),
        "evidence_grade": payload.get("evidence_grade"),
        "score_claim": payload.get("score_claim"),
        "n_samples": payload.get("n_samples"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
    }


def _sample_pair_index(sample: dict[str, Any]) -> int | None:
    for key in ("pair_index", "video_pair_index", "sample_index"):
        value = sample.get(key)
        if isinstance(value, int) and value >= 0:
            return int(value)
    frame_start = sample.get("frame_start")
    if isinstance(frame_start, int) and frame_start >= 0:
        return int(frame_start) // 2
    frames = sample.get("frame_indices")
    if isinstance(frames, list) and frames and isinstance(frames[0], int) and frames[0] >= 0:
        return int(frames[0]) // 2
    return None


def _sample_frame_indices(sample: dict[str, Any], *, pair_index: int) -> tuple[int, ...]:
    frames = sample.get("frame_indices")
    if isinstance(frames, list):
        parsed = tuple(sorted({int(frame) for frame in frames if isinstance(frame, int) and frame >= 0}))
        if parsed:
            return parsed
    frame_start = sample.get("frame_start")
    if isinstance(frame_start, int) and frame_start >= 0:
        return (int(frame_start), int(frame_start) + 1)
    return (2 * int(pair_index), 2 * int(pair_index) + 1)


def _get_trace_samples(payload: dict[str, Any], key: str, limit: int | None) -> list[dict[str, Any]]:
    values = payload.get(key)
    if not isinstance(values, list):
        return []
    samples = [item for item in values if isinstance(item, dict)]
    if limit is None:
        return samples
    return samples[: max(0, int(limit))]


def _rank_bonus(rank: int, total: int, *, channel_weight: float) -> float:
    if total <= 0:
        return 0.0
    return channel_weight * (1.0 + (total - rank) / total)


def _sample_component_bonus(sample: dict[str, Any]) -> float:
    pose = _optional_float(sample.get("posenet_dist")) or 0.0
    seg = _optional_float(sample.get("segnet_dist")) or 0.0
    combined = _optional_float(sample.get("score_combined_contribution_first_order")) or 0.0
    pose_contrib = _optional_float(sample.get("score_pose_contribution_first_order")) or 0.0
    seg_contrib = _optional_float(sample.get("score_seg_contribution_exact")) or 0.0
    return 2_500.0 * combined + 1_500.0 * pose_contrib + 1_000.0 * seg_contrib + 15.0 * pose + 8.0 * seg


def _add_trace_sample(
    accumulators: dict[int, _PairAccumulator],
    sample: dict[str, Any],
    *,
    source: str,
    score: float,
) -> None:
    pair_index = _sample_pair_index(sample)
    if pair_index is None:
        return
    item = accumulators.setdefault(pair_index, _PairAccumulator(pair_index=pair_index))
    item.add_sample(sample, source=source, score=score)


def _load_component_trace_pairs(
    path: Path,
    *,
    top_pose_pairs: int,
    top_seg_pairs: int,
    top_combined_pairs: int,
    min_pose_dist: float | None,
    min_seg_dist: float | None,
    min_combined_contribution: float | None,
) -> tuple[list[PairRecord], dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    accumulators: dict[int, _PairAccumulator] = {}
    ranked_sources = (
        ("top_pose_samples", top_pose_pairs, 3.0),
        ("top_seg_samples", top_seg_pairs, 2.5),
        ("top_combined_samples", top_combined_pairs, 3.5),
    )
    for key, limit, weight in ranked_sources:
        samples = _get_trace_samples(payload, key, limit)
        for rank, sample in enumerate(samples):
            _add_trace_sample(
                accumulators,
                sample,
                source=key,
                score=_rank_bonus(rank, len(samples), channel_weight=weight) + _sample_component_bonus(sample),
            )

    threshold_samples = _get_trace_samples(payload, "samples", None)
    for sample in threshold_samples:
        pose = _optional_float(sample.get("posenet_dist"))
        seg = _optional_float(sample.get("segnet_dist"))
        combined = _optional_float(sample.get("score_combined_contribution_first_order"))
        keep = False
        if min_pose_dist is not None and pose is not None and pose >= min_pose_dist:
            keep = True
        if min_seg_dist is not None and seg is not None and seg >= min_seg_dist:
            keep = True
        if min_combined_contribution is not None and combined is not None and combined >= min_combined_contribution:
            keep = True
        if keep:
            _add_trace_sample(
                accumulators,
                sample,
                source="threshold_samples",
                score=1.0 + _sample_component_bonus(sample),
            )

    records = _sort_pair_records(item.record() for item in accumulators.values())
    return records, _trace_input_record(path, payload)


def _explicit_pairs_from_json(path: Path) -> tuple[int, ...]:
    payload = _read_json(path)
    values: Any
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = None
        for key in ("protected_pair_indices", "hard_pair_indices", "pair_indices", "pairs"):
            if key in payload:
                values = payload[key]
                break
        if values is None:
            raise PlannerError(f"{path} must contain protected_pair_indices, hard_pair_indices, pair_indices, or pairs")
    else:
        raise PlannerError(f"{path} must contain a JSON list or object")
    if not isinstance(values, list):
        raise PlannerError(f"{path} protected pair payload must be a list")
    pairs: set[int] = set()
    for index, value in enumerate(values):
        if not isinstance(value, int) or value < 0:
            raise PlannerError(f"{path} protected pair at index {index} must be a nonnegative integer")
        pairs.add(int(value))
    return tuple(sorted(pairs))


def _sort_pair_records(records: Iterable[PairRecord]) -> list[PairRecord]:
    return sorted(records, key=lambda item: (-item.rank_score, item.pair_index))


def _merge_pair_records(records: Iterable[PairRecord], explicit_pairs: tuple[int, ...]) -> list[PairRecord]:
    accumulators: dict[int, _PairAccumulator] = {}
    for record in records:
        item = accumulators.setdefault(record.pair_index, _PairAccumulator(pair_index=record.pair_index))
        item.score += record.rank_score
        item.explicit = item.explicit or record.explicit
        item.sources.update(record.sources)
        item.frame_indices.update(record.frame_indices)
        item.posenet_dist = _max_optional(item.posenet_dist, record.posenet_dist)
        item.segnet_dist = _max_optional(item.segnet_dist, record.segnet_dist)
        item.combined_contribution = _max_optional(item.combined_contribution, record.combined_contribution)
    for pair_index in explicit_pairs:
        item = accumulators.setdefault(pair_index, _PairAccumulator(pair_index=pair_index))
        item.explicit = True
        item.score += 10_000.0
        item.sources.add("explicit_protected_pairs")
        item.frame_indices.update((2 * pair_index, 2 * pair_index + 1))
    return _sort_pair_records(item.record() for item in accumulators.values())


def _select_pairs_for_band(
    ranked_pairs: list[PairRecord],
    *,
    target_savings: int,
    max_target_savings: int,
) -> tuple[PairRecord, ...]:
    if not ranked_pairs:
        return ()
    explicit = [item for item in ranked_pairs if item.explicit]
    ranked_nonexplicit = [item for item in ranked_pairs if not item.explicit]
    if max_target_savings <= 0:
        fraction = 1.0
    else:
        fraction = 1.0 - 0.35 * max(0.0, min(1.0, target_savings / max_target_savings))
    cap = max(len(explicit), int(math.ceil(len(ranked_pairs) * fraction)))
    selected = explicit + ranked_nonexplicit[: max(0, cap - len(explicit))]
    return tuple(_sort_pair_records(selected))


def _mask_frames_for_pair(
    record: PairRecord,
    *,
    mask_frame_mode: str,
    frame_radius: int,
    mask_frame_count: int | None,
) -> tuple[int, ...]:
    if mask_frame_mode == "pair-index":
        base_frames = (record.pair_index,)
    elif mask_frame_mode == "video-frame":
        base_frames = record.frame_indices or (2 * record.pair_index, 2 * record.pair_index + 1)
    else:
        raise PlannerError(f"unsupported mask_frame_mode {mask_frame_mode!r}")
    frames: set[int] = set()
    for frame in base_frames:
        for expanded in range(int(frame) - frame_radius, int(frame) + frame_radius + 1):
            if expanded < 0:
                continue
            if mask_frame_count is not None and expanded >= mask_frame_count:
                continue
            frames.add(expanded)
    return tuple(sorted(frames))


def _protected_frames_for_selection(
    selected_pairs: tuple[PairRecord, ...],
    *,
    mask_frame_mode: str,
    frame_radius: int,
    mask_frame_count: int | None,
    explicit_mask_frames: tuple[int, ...],
) -> tuple[int, ...]:
    frames: set[int] = set()
    for record in selected_pairs:
        frames.update(
            _mask_frames_for_pair(
                record,
                mask_frame_mode=mask_frame_mode,
                frame_radius=frame_radius,
                mask_frame_count=mask_frame_count,
            )
        )
    for frame in explicit_mask_frames:
        for expanded in range(int(frame) - frame_radius, int(frame) + frame_radius + 1):
            if expanded < 0:
                continue
            if mask_frame_count is not None and expanded >= mask_frame_count:
                continue
            frames.add(expanded)
    return tuple(sorted(frames))


def _frame_preview(frames: tuple[int, ...], *, limit: int = 20) -> dict[str, Any]:
    return {
        "count": len(frames),
        "first": [int(value) for value in frames[:limit]],
        "last": [int(value) for value in frames[-limit:]] if len(frames) > limit else [],
    }


def _target_crf_hint(target_savings: int, target_savings_values: tuple[int, ...], probe_crfs: tuple[int, ...]) -> int:
    if not probe_crfs:
        return 0
    ordered_targets = tuple(sorted(target_savings_values))
    try:
        target_index = ordered_targets.index(target_savings)
    except ValueError:
        target_index = min(len(ordered_targets) - 1, max(0, round(len(ordered_targets) / 2)))
    crf_index = min(len(probe_crfs) - 1, target_index + max(0, len(probe_crfs) // 2 - len(ordered_targets) // 2))
    return int(probe_crfs[crf_index])


def _builder_policy(
    *,
    policy_id: str,
    selected_pairs: tuple[PairRecord, ...],
    protected_frames: tuple[int, ...],
    protected_class_ids: tuple[int, ...],
    boundary_dilation: int,
    regions: tuple[RegionSpec, ...],
    emit_global_class_protection: bool,
) -> dict[str, Any]:
    horizon_regions = [region.with_frames(protected_frames) for region in regions if "horizon" in region.name]
    foveal_regions = [region.with_frames(protected_frames) for region in regions if "foveal" in region.name]
    ego_regions = [
        region.with_frames(protected_frames)
        for region in regions
        if "ego" in region.name or ("horizon" not in region.name and "foveal" not in region.name)
    ]
    return {
        "label": policy_id,
        "hard_pair_indices": [],
        "hard_frame_indices": [int(frame) for frame in protected_frames],
        "class_ids": [int(value) for value in protected_class_ids] if emit_global_class_protection else [],
        "boundary_dilation": int(boundary_dilation),
        "horizon_bands": horizon_regions,
        "foveal_boxes": foveal_regions,
        "ego_boxes": ego_regions,
        "planner_semantics": {
            "protected_pair_indices": [int(item.pair_index) for item in selected_pairs],
            "protected_class_ids": [int(value) for value in protected_class_ids],
            "class_protection_scope": (
                "global_builder_class_ids"
                if emit_global_class_protection
                else "classes are protected through selected hard frames and protected regions; "
                "global class-id protection is omitted to keep the plan micro-scoped"
            ),
        },
    }


def _candidate_config(
    *,
    target_savings: int,
    selected_pairs: tuple[PairRecord, ...],
    protected_frames: tuple[int, ...],
    protected_class_ids: tuple[int, ...],
    regions: tuple[RegionSpec, ...],
    boundary_dilation: int,
    savings_tolerance: int,
    base_archive_bytes: int,
    mask_stream_bytes: int,
    probe_crfs: tuple[int, ...],
    target_savings_values: tuple[int, ...],
    output_json: Path,
    write_policy_jsons: bool,
    emit_global_class_protection: bool,
    base_archive: Path | None,
    builder_output_dir: Path | None,
) -> dict[str, Any]:
    policy_id = f"c067_micro_av1_mask_reencode_save{target_savings // 1000:02d}k"
    savings_min = max(1, int(target_savings) - int(savings_tolerance))
    savings_max = int(target_savings) + int(savings_tolerance)
    builder_policy = _builder_policy(
        policy_id=policy_id,
        selected_pairs=selected_pairs,
        protected_frames=protected_frames,
        protected_class_ids=protected_class_ids,
        boundary_dilation=boundary_dilation,
        regions=regions,
        emit_global_class_protection=emit_global_class_protection,
    )
    policy_json_path = str(output_json.with_name(f"{policy_id}.policy.json"))
    if write_policy_jsons:
        _write_json(Path(policy_json_path), builder_policy)
    command: list[str] | None = None
    if base_archive is not None and builder_output_dir is not None:
        command = [
            "python",
            "experiments/build_protected_mask_reencode_candidate.py",
            "--base-archive",
            str(base_archive),
            "--policy-json",
            policy_json_path,
            "--output-dir",
            str(builder_output_dir / policy_id),
            "--crf",
            str(_target_crf_hint(target_savings, target_savings_values, probe_crfs)),
            "--protection-iterations",
            "1",
        ]
    return {
        "policy_id": policy_id,
        "candidate_family": "micro_av1_mask_reencode_trust_region",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "planning_only",
        "target_savings_bytes": int(target_savings),
        "byte_screen": {
            "baseline_archive_bytes": int(base_archive_bytes),
            "baseline_mask_stream_bytes": int(mask_stream_bytes),
            "target_archive_bytes": int(base_archive_bytes) - int(target_savings),
            "target_mask_stream_bytes": int(mask_stream_bytes) - int(target_savings),
            "acceptable_savings_min_bytes": int(savings_min),
            "acceptable_savings_max_bytes": int(savings_max),
            "acceptable_archive_byte_range": [
                int(base_archive_bytes) - int(savings_max),
                int(base_archive_bytes) - int(savings_min),
            ],
            "acceptable_mask_stream_byte_range": [
                int(mask_stream_bytes) - int(savings_max),
                int(mask_stream_bytes) - int(savings_min),
            ],
            "rate_score_improvement_at_target": round(LAMBDA_RATE * int(target_savings), 12),
            "component_damage_budget_before_score_regression": round(LAMBDA_RATE * int(target_savings), 12),
            "reject_if_archive_bytes_not_in_range": True,
        },
        "av1_probe": {
            "probe_crfs": [int(value) for value in probe_crfs],
            "first_probe_crf_hint": _target_crf_hint(target_savings, target_savings_values, probe_crfs),
            "svtav1_params": "enable-restoration=0:enable-cdef=0:lp=1",
            "broad_crf_replacement_allowed": False,
            "must_use_protected_preencode_composite": True,
        },
        "protection": {
            "protected_pair_count": len(selected_pairs),
            "protected_pair_indices": [int(item.pair_index) for item in sorted(selected_pairs, key=lambda item: item.pair_index)],
            "protected_pairs_ranked_preview": [item.to_json() for item in selected_pairs[:24]],
            "protected_mask_frames": [int(frame) for frame in protected_frames],
            "protected_mask_frame_preview": _frame_preview(protected_frames),
            "protected_class_ids": [int(value) for value in protected_class_ids],
            "protected_regions": [region.with_frames(protected_frames) for region in regions],
            "boundary_dilation": int(boundary_dilation),
        },
        "builder_policy_json_path": policy_json_path,
        "builder_policy_json_written": bool(write_policy_jsons),
        "builder_policy_json": builder_policy,
        "local_builder_command_if_reviewed": command if write_policy_jsons else None,
        "local_builder_command_template_if_policy_json_written": command,
        "required_acceptance_gates": [
            "candidate manifest score_claim=false and promotion_eligible=false",
            "archive members preserved except mask payload",
            "candidate measured savings falls inside this byte band",
            "protected-region decoded agreement is 1.0 before any exact eval dispatch",
            "no remote GPU dispatch from this planning output",
        ],
    }


def _extract_archive_bytes(payload: dict[str, Any]) -> int | None:
    archive = payload.get("archive")
    if isinstance(archive, dict):
        size = archive.get("size_bytes", archive.get("archive_size_bytes"))
        if isinstance(size, int):
            return int(size)
    for key in ("archive_size_bytes", "archive_bytes", "size_bytes"):
        size = payload.get(key)
        if isinstance(size, int):
            return int(size)
    return None


def _extract_mask_bytes(payload: dict[str, Any]) -> int | None:
    candidate = payload.get("candidate_mask_stream")
    if isinstance(candidate, dict):
        size = candidate.get("size_bytes", candidate.get("mask_stream_bytes"))
        if isinstance(size, int):
            return int(size)
    for key in ("mask_stream_bytes", "candidate_mask_stream_bytes"):
        size = payload.get(key)
        if isinstance(size, int):
            return int(size)
    return None


def _screen_measured_candidate(
    path: Path,
    *,
    base_archive_bytes: int,
    target_savings_values: tuple[int, ...],
    savings_tolerance: int,
    micro_max_savings_bytes: int,
) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    archive_bytes = _extract_archive_bytes(payload)
    mask_bytes = _extract_mask_bytes(payload)
    savings = None if archive_bytes is None else int(base_archive_bytes) - int(archive_bytes)
    reject_reasons: list[str] = []
    if payload.get("score_claim") is True:
        reject_reasons.append("measured_candidate_claims_score")
    if archive_bytes is None:
        reject_reasons.append("missing_archive_size_bytes")
    elif savings is not None and savings <= 0:
        reject_reasons.append("no_archive_byte_saving")
    if savings is not None and savings > micro_max_savings_bytes + savings_tolerance:
        reject_reasons.append("exceeds_micro_savings_cap_broad_reencode_risk")
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        reject_reasons.append("missing_trust_region_policy")
    elif not (
        policy.get("hard_frame_indices")
        or policy.get("hard_pair_indices")
        or policy.get("horizon_bands")
        or policy.get("foveal_boxes")
        or policy.get("ego_boxes")
    ):
        reject_reasons.append("policy_has_no_protected_frames_or_regions")
    nearest_band = None
    if savings is not None and target_savings_values:
        nearest_band = min(target_savings_values, key=lambda target: abs(int(target) - int(savings)))
        if abs(int(nearest_band) - int(savings)) > savings_tolerance:
            reject_reasons.append("outside_target_savings_bands")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path.resolve()),
        "score_claim": payload.get("score_claim"),
        "archive_size_bytes": archive_bytes,
        "mask_stream_size_bytes": mask_bytes,
        "archive_savings_bytes": savings,
        "nearest_target_savings_bytes": nearest_band,
        "accepted_for_local_byte_screen": not reject_reasons,
        "reject_reasons": reject_reasons,
    }


def _validate_micro_contract(
    *,
    candidate_family: str,
    target_savings_values: tuple[int, ...],
    micro_max_savings_bytes: int,
    protected_pair_count: int,
) -> None:
    if candidate_family != "micro-av1-trust-region":
        raise PlannerError(
            "broad CRF replacement is refused for C067: coarse mask AV1/CRF replacement has "
            "collapsed PoseNet in current forensics; use candidate_family=micro-av1-trust-region"
        )
    if not target_savings_values:
        raise PlannerError("at least one target savings band is required")
    too_large = [value for value in target_savings_values if value > micro_max_savings_bytes]
    if too_large:
        raise PlannerError(
            f"target savings {too_large} exceed the micro trust-region cap "
            f"{micro_max_savings_bytes}; broad replacement is refused"
        )
    if protected_pair_count <= 0:
        raise PlannerError("no protected pairs were found; pass --component-trace or --protected-pairs")


def build_plan(
    *,
    output_json: Path,
    component_traces: list[Path] | None = None,
    protected_pairs: tuple[int, ...] = (),
    protected_pairs_jsons: list[Path] | None = None,
    protected_mask_frames: tuple[int, ...] = (),
    target_savings_bytes: tuple[int, ...] = DEFAULT_TARGET_SAVINGS_BYTES,
    savings_tolerance_bytes: int = 512,
    probe_crfs: tuple[int, ...] = DEFAULT_PROBE_CRFS,
    protected_class_ids: tuple[int, ...] = DEFAULT_PROTECTED_CLASS_IDS,
    regions: tuple[RegionSpec, ...] | None = None,
    top_pose_pairs: int = DEFAULT_TOP_POSE_PAIRS,
    top_seg_pairs: int = DEFAULT_TOP_SEG_PAIRS,
    top_combined_pairs: int = DEFAULT_TOP_COMBINED_PAIRS,
    min_pose_dist: float | None = None,
    min_seg_dist: float | None = None,
    min_combined_contribution: float | None = None,
    mask_frame_mode: str = "pair-index",
    mask_frame_count: int | None = DEFAULT_MASK_FRAME_COUNT,
    mask_width: int = DEFAULT_MASK_WIDTH,
    mask_height: int = DEFAULT_MASK_HEIGHT,
    frame_radius: int = 1,
    boundary_dilation: int = 1,
    micro_max_savings_bytes: int = DEFAULT_MICRO_MAX_SAVINGS_BYTES,
    candidate_family: str = "micro-av1-trust-region",
    base_archive_bytes: int = C067_FRONTIER_ARCHIVE_BYTES,
    base_archive_sha256: str = C067_FRONTIER_ARCHIVE_SHA256,
    base_score: float = C067_FRONTIER_SCORE,
    mask_stream_bytes: int = C067_MASK_STREAM_BYTES,
    mask_member: str = C067_MASK_MEMBER,
    base_archive: Path | None = None,
    builder_output_dir: Path | None = None,
    measured_candidate_jsons: list[Path] | None = None,
    write_policy_jsons: bool = False,
    emit_global_class_protection: bool = False,
) -> dict[str, Any]:
    """Build and write a deterministic C067 micro mask reencode plan."""

    if savings_tolerance_bytes < 0:
        raise PlannerError("savings_tolerance_bytes must be nonnegative")
    if frame_radius < 0:
        raise PlannerError("frame_radius must be nonnegative")
    if boundary_dilation < 0:
        raise PlannerError("boundary_dilation must be nonnegative")
    if mask_frame_count is not None and mask_frame_count <= 0:
        raise PlannerError("mask_frame_count must be positive when supplied")
    if base_archive_bytes <= 0 or mask_stream_bytes <= 0:
        raise PlannerError("base archive and mask stream bytes must be positive")
    if any(crf < 0 or crf > 63 for crf in probe_crfs):
        raise PlannerError("probe CRFs must be in [0,63]")
    target_savings_values = tuple(sorted(set(int(value) for value in target_savings_bytes)))
    probe_crf_values = tuple(sorted(set(int(value) for value in probe_crfs)))
    region_values = tuple(regions if regions is not None else _default_regions())
    _validate_regions(region_values, width=mask_width, height=mask_height)

    trace_inputs: list[dict[str, Any]] = []
    trace_records: list[PairRecord] = []
    for trace_path in component_traces or []:
        records, input_record = _load_component_trace_pairs(
            trace_path,
            top_pose_pairs=top_pose_pairs,
            top_seg_pairs=top_seg_pairs,
            top_combined_pairs=top_combined_pairs,
            min_pose_dist=min_pose_dist,
            min_seg_dist=min_seg_dist,
            min_combined_contribution=min_combined_contribution,
        )
        trace_records.extend(records)
        trace_inputs.append(input_record)

    explicit_pairs = set(protected_pairs)
    protected_pair_inputs: list[dict[str, Any]] = []
    for path in protected_pairs_jsons or []:
        values = _explicit_pairs_from_json(path)
        explicit_pairs.update(values)
        protected_pair_inputs.append(
            {
                "kind": "protected_pairs_json",
                "path": str(path.resolve()),
                "sha256": _sha256_file(path.resolve()),
                "pair_count": len(values),
            }
        )
    ranked_pairs = _merge_pair_records(trace_records, tuple(sorted(explicit_pairs)))
    _validate_micro_contract(
        candidate_family=candidate_family,
        target_savings_values=target_savings_values,
        micro_max_savings_bytes=micro_max_savings_bytes,
        protected_pair_count=len(ranked_pairs),
    )

    candidate_configs: list[dict[str, Any]] = []
    max_target = max(target_savings_values)
    for target_savings in target_savings_values:
        band_radius = max(0, frame_radius + (1 if target_savings == min(target_savings_values) else 0))
        if target_savings == max_target:
            band_radius = max(0, band_radius - 1)
        selected_pairs = _select_pairs_for_band(
            ranked_pairs,
            target_savings=target_savings,
            max_target_savings=max_target,
        )
        protected_frames_for_band = _protected_frames_for_selection(
            selected_pairs,
            mask_frame_mode=mask_frame_mode,
            frame_radius=band_radius,
            mask_frame_count=mask_frame_count,
            explicit_mask_frames=protected_mask_frames,
        )
        candidate_configs.append(
            _candidate_config(
                target_savings=target_savings,
                selected_pairs=selected_pairs,
                protected_frames=protected_frames_for_band,
                protected_class_ids=protected_class_ids,
                regions=region_values,
                boundary_dilation=boundary_dilation,
                savings_tolerance=savings_tolerance_bytes,
                base_archive_bytes=base_archive_bytes,
                mask_stream_bytes=mask_stream_bytes,
                probe_crfs=probe_crf_values,
                target_savings_values=target_savings_values,
                output_json=output_json,
                write_policy_jsons=write_policy_jsons,
                emit_global_class_protection=emit_global_class_protection,
                base_archive=base_archive,
                builder_output_dir=builder_output_dir,
            )
        )

    measured_screen = [
        _screen_measured_candidate(
            path,
            base_archive_bytes=base_archive_bytes,
            target_savings_values=target_savings_values,
            savings_tolerance=savings_tolerance_bytes,
            micro_max_savings_bytes=micro_max_savings_bytes,
        )
        for path in measured_candidate_jsons or []
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": "planning_only",
        "cuda_jobs_launched": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "frontier": {
            "name": "C067",
            "score": float(base_score),
            "archive_bytes": int(base_archive_bytes),
            "archive_sha256": str(base_archive_sha256),
            "mask_member": mask_member,
            "mask_stream_bytes": int(mask_stream_bytes),
            "rate_score_per_byte": LAMBDA_RATE,
        },
        "configuration": {
            "candidate_family": candidate_family,
            "target_savings_bytes": [int(value) for value in target_savings_values],
            "savings_tolerance_bytes": int(savings_tolerance_bytes),
            "probe_crfs": [int(value) for value in probe_crf_values],
            "top_pose_pairs": int(top_pose_pairs),
            "top_seg_pairs": int(top_seg_pairs),
            "top_combined_pairs": int(top_combined_pairs),
            "min_pose_dist": min_pose_dist,
            "min_seg_dist": min_seg_dist,
            "min_combined_contribution": min_combined_contribution,
            "mask_frame_mode": mask_frame_mode,
            "mask_frame_count": mask_frame_count,
            "mask_width": int(mask_width),
            "mask_height": int(mask_height),
            "frame_radius": int(frame_radius),
            "boundary_dilation": int(boundary_dilation),
            "micro_max_savings_bytes": int(micro_max_savings_bytes),
            "write_policy_jsons": bool(write_policy_jsons),
            "emit_global_class_protection": bool(emit_global_class_protection),
        },
        "refused_candidate_families": [
            {
                "family": "broad_whole_mask_crf_replacement",
                "status": "refused",
                "reason": (
                    "C067/PR67 coarse mask AV1/CRF replacement and CMG3A-style broad mutations have "
                    "collapsed PoseNet in current forensics; this tool only emits micro trust-region plans."
                ),
            }
        ],
        "inputs": {
            "component_traces": trace_inputs,
            "protected_pair_inputs": protected_pair_inputs,
            "inline_protected_pair_indices": [int(value) for value in protected_pairs],
            "inline_protected_mask_frames": [int(value) for value in protected_mask_frames],
        },
        "protected_pair_ranking": {
            "pair_count": len(ranked_pairs),
            "ranked_pairs": [record.to_json() for record in ranked_pairs],
        },
        "candidate_configs": candidate_configs,
        "measured_candidate_byte_screen": measured_screen,
        "required_next_steps": [
            "Review candidate policy JSON before local archive building.",
            "Build locally only; this planner does not authorize GPU dispatch.",
            "Reject any candidate outside its byte band or without protected-region decoded agreement.",
            "Before any remote exact eval, claim the lane with tools/claim_lane_dispatch.py claim.",
        ],
    }
    _write_json(output_json, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--component-trace", type=Path, action="append", default=[])
    parser.add_argument("--protected-pairs", type=lambda value: _parse_int_set(value, field="protected_pairs"), default=())
    parser.add_argument("--protected-pairs-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--protected-mask-frames",
        type=lambda value: _parse_int_set(value, field="protected_mask_frames"),
        default=(),
    )
    parser.add_argument("--target-savings-bytes", type=_parse_positive_ints, default=DEFAULT_TARGET_SAVINGS_BYTES)
    parser.add_argument("--savings-tolerance-bytes", type=int, default=512)
    parser.add_argument("--probe-crfs", type=_parse_positive_ints, default=DEFAULT_PROBE_CRFS)
    parser.add_argument("--protected-class-ids", type=_parse_class_ids, default=DEFAULT_PROTECTED_CLASS_IDS)
    parser.add_argument("--region", type=_parse_region, action="append", default=None)
    parser.add_argument("--top-pose-pairs", type=int, default=DEFAULT_TOP_POSE_PAIRS)
    parser.add_argument("--top-seg-pairs", type=int, default=DEFAULT_TOP_SEG_PAIRS)
    parser.add_argument("--top-combined-pairs", type=int, default=DEFAULT_TOP_COMBINED_PAIRS)
    parser.add_argument("--min-pose-dist", type=float, default=None)
    parser.add_argument("--min-seg-dist", type=float, default=None)
    parser.add_argument("--min-combined-contribution", type=float, default=None)
    parser.add_argument("--mask-frame-mode", choices=("pair-index", "video-frame"), default="pair-index")
    parser.add_argument("--mask-frame-count", type=int, default=DEFAULT_MASK_FRAME_COUNT)
    parser.add_argument("--mask-width", type=int, default=DEFAULT_MASK_WIDTH)
    parser.add_argument("--mask-height", type=int, default=DEFAULT_MASK_HEIGHT)
    parser.add_argument("--frame-radius", type=int, default=1)
    parser.add_argument("--boundary-dilation", type=int, default=1)
    parser.add_argument("--micro-max-savings-bytes", type=int, default=DEFAULT_MICRO_MAX_SAVINGS_BYTES)
    parser.add_argument(
        "--candidate-family",
        choices=("micro-av1-trust-region", "broad-crf-replacement"),
        default="micro-av1-trust-region",
    )
    parser.add_argument("--base-archive-bytes", type=int, default=C067_FRONTIER_ARCHIVE_BYTES)
    parser.add_argument("--base-archive-sha256", default=C067_FRONTIER_ARCHIVE_SHA256)
    parser.add_argument("--base-score", type=float, default=C067_FRONTIER_SCORE)
    parser.add_argument("--mask-stream-bytes", type=int, default=C067_MASK_STREAM_BYTES)
    parser.add_argument("--mask-member", default=C067_MASK_MEMBER)
    parser.add_argument("--base-archive", type=Path, default=None)
    parser.add_argument("--builder-output-dir", type=Path, default=None)
    parser.add_argument("--measured-candidate-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--write-policy-jsons",
        action="store_true",
        help="Write adjacent *.policy.json files referenced by local builder command templates.",
    )
    parser.add_argument(
        "--emit-global-class-protection",
        action="store_true",
        help="Emit builder class_ids globally. Default omits this because it is usually too broad for micro plans.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_plan(
        output_json=args.output_json,
        component_traces=args.component_trace,
        protected_pairs=args.protected_pairs,
        protected_pairs_jsons=args.protected_pairs_json,
        protected_mask_frames=args.protected_mask_frames,
        target_savings_bytes=args.target_savings_bytes,
        savings_tolerance_bytes=args.savings_tolerance_bytes,
        probe_crfs=args.probe_crfs,
        protected_class_ids=args.protected_class_ids,
        regions=tuple(args.region) if args.region is not None else None,
        top_pose_pairs=args.top_pose_pairs,
        top_seg_pairs=args.top_seg_pairs,
        top_combined_pairs=args.top_combined_pairs,
        min_pose_dist=args.min_pose_dist,
        min_seg_dist=args.min_seg_dist,
        min_combined_contribution=args.min_combined_contribution,
        mask_frame_mode=args.mask_frame_mode,
        mask_frame_count=args.mask_frame_count,
        mask_width=args.mask_width,
        mask_height=args.mask_height,
        frame_radius=args.frame_radius,
        boundary_dilation=args.boundary_dilation,
        micro_max_savings_bytes=args.micro_max_savings_bytes,
        candidate_family=args.candidate_family,
        base_archive_bytes=args.base_archive_bytes,
        base_archive_sha256=args.base_archive_sha256,
        base_score=args.base_score,
        mask_stream_bytes=args.mask_stream_bytes,
        mask_member=args.mask_member,
        base_archive=args.base_archive,
        builder_output_dir=args.builder_output_dir,
        measured_candidate_jsons=args.measured_candidate_json,
        write_policy_jsons=args.write_policy_jsons,
        emit_global_class_protection=args.emit_global_class_protection,
    )
    first_policy = payload["candidate_configs"][0]["policy_id"] if payload["candidate_configs"] else "none"
    print(
        "[c067-micro-mask-reencode-plan] "
        f"wrote {args.output_json} policies={len(payload['candidate_configs'])} "
        f"protected_pairs={payload['protected_pair_ranking']['pair_count']} "
        f"first_policy={first_policy} score_claim=false remote_jobs_dispatched=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
