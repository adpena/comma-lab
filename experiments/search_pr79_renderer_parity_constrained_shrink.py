#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Search PR79 renderer shrink candidates with parity-before-archive gates.

This is a local-only screen.  It applies small tensor-local renderer
transforms, compares sampled render outputs against the source runtime before
building an archive, and only emits archive candidates after strict
renderer-transplant pose-safety passes.  It never dispatches GPU work and makes
no score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_renderer_shrink_candidate as shrink_builder  # noqa: E402
from experiments import preflight_renderer_transplant_pose_safety as pose_safety  # noqa: E402
from experiments import search_renderer_parity_shrink_candidate as base_search  # noqa: E402
from tac.quantizr_qzs3_codec import (  # noqa: E402
    decode_qzs3_state_dict,
    encode_qzs3_state_dict,
)


SCHEMA = "pr79_renderer_parity_constrained_shrink_search_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
FRONTIER_SCORE = 0.31457805357318636
TARGET_SCORE = 0.314
SOURCE_ARCHIVE_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr79_minp_v2_public_replay_t4_20260503T1615Z/archive.zip"
)
DEFAULT_SOURCE_EVIDENCE = DEFAULT_SOURCE_ARCHIVE.with_name("contest_auth_eval.json")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/pr79_renderer_parity_constrained_shrink_20260503_codex"
)


@dataclass(frozen=True)
class TransformSpec:
    kind: str
    prefix: str | None
    value: float | int

    @property
    def candidate_id(self) -> str:
        if self.kind == "qzs3_reblock":
            return f"qzs3_reblock_b{int(self.value):04d}"
        if self.kind == "zero_fp4_prefix":
            value = f"{float(self.value):.5f}".rstrip("0").rstrip(".")
            return _slug(f"zero_fp4_{self.prefix}_{value}")
        raise ValueError(f"unsupported transform kind: {self.kind!r}")

    @property
    def spec(self) -> str:
        if self.kind == "qzs3_reblock":
            return f"qzs3-reblock:{int(self.value)}"
        if self.kind == "zero_fp4_prefix":
            return f"zero-fp4-prefix:{self.prefix}:{float(self.value):.6g}"
        raise ValueError(f"unsupported transform kind: {self.kind!r}")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "candidate"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _file_meta(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    meta: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            meta["json_parseable"] = False
        else:
            meta["json_parseable"] = True
            meta["summary"] = {
                key: payload[key]
                for key in (
                    "canonical_score",
                    "score_recomputed_from_components",
                    "avg_posenet_dist",
                    "avg_segnet_dist",
                    "archive_size_bytes",
                    "n_samples",
                )
                if key in payload
            }
    return meta


def parse_transform_spec(raw: str) -> TransformSpec:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) == 2 and parts[0] == "qzs3-reblock":
        block_size = int(parts[1])
        if block_size <= 0 or block_size > 4096:
            raise ValueError(f"block size must be in [1, 4096], got {block_size}")
        return TransformSpec("qzs3_reblock", None, block_size)
    if len(parts) == 3 and parts[0] == "zero-fp4-prefix":
        prefix = parts[1]
        if not prefix or "/" in prefix or "\\" in prefix or prefix.startswith("."):
            raise ValueError(f"unsafe or empty tensor prefix: {prefix!r}")
        threshold = float(parts[2])
        if threshold < 0.0 or threshold > 1.0 or not math.isfinite(threshold):
            raise ValueError(f"zero threshold must be finite and in [0, 1], got {threshold}")
        return TransformSpec("zero_fp4_prefix", prefix, threshold)
    raise ValueError(
        "transform spec must be qzs3-reblock:<block_size> or "
        "zero-fp4-prefix:<prefix>:<threshold>"
    )


def default_transform_specs() -> tuple[TransformSpec, ...]:
    raw_specs = (
        "qzs3-reblock:40",
        "qzs3-reblock:48",
        "qzs3-reblock:64",
        "zero-fp4-prefix:frame1_head:0.02",
        "zero-fp4-prefix:frame1_head:0.03",
        "zero-fp4-prefix:frame1_head:0.04",
        "zero-fp4-prefix:frame2_head.pre:0.01",
        "zero-fp4-prefix:frame2_head.pre:0.02",
        "zero-fp4-prefix:frame2_head.block2:0.01",
        "zero-fp4-prefix:frame2_head.block2:0.02",
    )
    return tuple(parse_transform_spec(raw) for raw in raw_specs)


def _source_context(source_archive: Path, *, brotli_quality: int) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    source_bytes = source_archive.read_bytes()
    source_members, source_packaging = shrink_builder.blockfp.extract_runtime_members(
        source_archive,
    )
    if shrink_builder.RENDERER_MEMBER not in source_members:
        raise ValueError("source archive missing renderer.bin")
    if shrink_builder.MASK_MEMBER not in source_members:
        raise ValueError("source archive missing masks.mkv")
    if not any(name in source_members for name in shrink_builder.POSE_MEMBERS):
        raise ValueError("source archive missing optimized pose payload")
    source_renderer = source_members[shrink_builder.RENDERER_MEMBER]
    if not source_renderer.startswith(b"QZS3"):
        raise ValueError(f"source renderer must be QZS3, got {source_renderer[:4]!r}")
    pr75_slices = shrink_builder.load_pr75_slices_for_renderer_shrink(
        source_archive,
        source_members,
        brotli_quality=brotli_quality,
    )
    if pr75_slices is None:
        raise ValueError("PR79 parity-constrained shrink requires a PR75-style p payload")
    return {
        "source_archive": source_archive,
        "source_bytes": source_bytes,
        "source_sha256": _sha256_bytes(source_bytes),
        "source_members": source_members,
        "source_packaging": source_packaging,
        "source_renderer": source_renderer,
        "source_state": decode_qzs3_state_dict(source_renderer, device="cpu"),
        "source_block_size": int.from_bytes(source_renderer[4:6], "little"),
        "pr75_slices": pr75_slices,
        "non_renderer_members": {
            name: payload
            for name, payload in source_members.items()
            if name != shrink_builder.RENDERER_MEMBER
        },
    }


def _encode_transform_renderer(
    *,
    source_state: dict[str, torch.Tensor],
    source_renderer: bytes,
    source_block_size: int,
    transform: TransformSpec,
) -> tuple[bytes, dict[str, Any]]:
    if transform.kind == "qzs3_reblock":
        block_size = int(transform.value)
        renderer = encode_qzs3_state_dict(source_state, block_size=block_size)
        decode_qzs3_state_dict(renderer, device="cpu")
        return renderer, {
            "transform_family": "qzs3_global_reblock",
            "wire_format": "QZS3",
            "source_block_size": source_block_size,
            "output_block_size": block_size,
            "renderer_changed": renderer != source_renderer,
        }
    if transform.kind == "zero_fp4_prefix":
        transformed, meta = base_search._apply_zero_fp4_prefix(
            source_state,
            prefix=str(transform.prefix),
            threshold_fraction=float(transform.value),
        )
        renderer = encode_qzs3_state_dict(transformed, block_size=source_block_size)
        decode_qzs3_state_dict(renderer, device="cpu")
        return renderer, {
            "wire_format": "QZS3",
            "source_block_size": source_block_size,
            "output_block_size": source_block_size,
            "renderer_changed": renderer != source_renderer,
            **meta,
        }
    raise ValueError(f"unsupported transform kind: {transform.kind!r}")


def _pair_count_for_state(state: dict[str, Any]) -> int:
    masks = state["masks"]
    poses = state["poses"]
    pair_count = int(poses.shape[0])
    if bool(getattr(masks, "_half_frame_only", False)):
        return min(pair_count, int(masks.shape[0]))
    return min(pair_count, int(masks.shape[0]) // 2)


def _compare_against_source_frames(
    *,
    candidate_state: dict[str, Any],
    pair_indices: list[int],
    source_frames: Any,
    source_output_summary: dict[str, Any],
    pair_count_available: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    candidate_frames = pose_safety._render_pair_batch(
        renderer=candidate_state["renderer"],
        masks=candidate_state["masks"],
        poses=candidate_state["poses"],
        pair_indices=pair_indices,
    )
    aggregate = pose_safety.compare_frame_batches(
        source_frames,
        candidate_frames,
        max_mean_abs_delta=max_mean_abs_delta,
        max_rms_delta=max_rms_delta,
        max_max_abs_delta=max_max_abs_delta,
    )
    per_pair = []
    for offset, pair_index in enumerate(pair_indices):
        comparison = pose_safety.compare_frame_batches(
            source_frames[offset : offset + 1],
            candidate_frames[offset : offset + 1],
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
        per_pair.append({"pair_index": pair_index, **comparison})
    return {
        "ok": bool(aggregate.get("ok")),
        "failure_class": None if aggregate.get("ok") else "render_output_parity_unsafe",
        "pair_count_available": pair_count_available,
        "sampled_pair_indices": pair_indices,
        "source_output_summary": source_output_summary,
        "candidate_output_summary": pose_safety._summarize_frames("candidate", candidate_frames),
        "aggregate": aggregate,
        "per_pair": per_pair,
    }


def _build_pre_archive_parity(
    *,
    context: dict[str, Any],
    renderer: bytes,
    source_state: dict[str, Any],
    source_frames: Any,
    source_output_summary: dict[str, Any],
    pair_indices: list[int],
    pair_count_available: int,
    work_root: Path,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    candidate_members = dict(context["source_members"])
    candidate_members[shrink_builder.RENDERER_MEMBER] = renderer
    candidate_dir = work_root / "candidate_runtime"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_state = pose_safety._load_runtime_state(candidate_dir, candidate_members)
    if list(source_state["masks"].shape) != list(candidate_state["masks"].shape):
        return {
            "ok": False,
            "failure_class": "mask_shape_mismatch",
            "stage": "pre_archive_render_output_parity",
        }
    if list(source_state["poses"].shape) != list(candidate_state["poses"].shape):
        return {
            "ok": False,
            "failure_class": "pose_shape_mismatch",
            "stage": "pre_archive_render_output_parity",
        }
    output_parity = _compare_against_source_frames(
        candidate_state=candidate_state,
        pair_indices=pair_indices,
        source_frames=source_frames,
        source_output_summary=source_output_summary,
        pair_count_available=pair_count_available,
        max_mean_abs_delta=max_mean_abs_delta,
        max_rms_delta=max_rms_delta,
        max_max_abs_delta=max_max_abs_delta,
    )
    return {
        "ok": bool(output_parity.get("ok")),
        "stage": "pre_archive_render_output_parity",
        "failure_class": output_parity.get("failure_class"),
        "output_parity": output_parity,
    }


def _build_temp_archive(
    *,
    context: dict[str, Any],
    transform: TransformSpec,
    renderer: bytes,
    transform_meta: dict[str, Any],
    candidate_dir: Path,
    brotli_quality: int,
    pre_archive_parity: dict[str, Any],
    source_evidence_path: Path | None,
) -> dict[str, Any]:
    payload, payload_meta = shrink_builder._build_pr75_payload(
        context["pr75_slices"],
        renderer_bytes=renderer,
        brotli_quality=brotli_quality,
    )
    archive_path = candidate_dir / "archive.zip"
    shrink_builder._write_single_member_archive(archive_path, payload)
    runtime_unpack = shrink_builder._verify_archive(
        archive_path,
        expected_renderer=renderer,
        expected_non_renderer_members=context["non_renderer_members"],
    )
    archive_bytes = archive_path.stat().st_size
    source_bytes = len(context["source_bytes"])
    delta = archive_bytes - source_bytes
    manifest = {
        "schema": SCHEMA,
        "tool": "experiments/search_pr79_renderer_parity_constrained_shrink.py",
        "candidate_id": transform.candidate_id,
        "transform_spec": transform.spec,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_eval_ready": False,
        "evidence_grade": "empirical_local_preflight_no_score",
        "source_archive": {
            "path": str(context["source_archive"]),
            "bytes": source_bytes,
            "sha256": context["source_sha256"],
            **context["source_packaging"],
        },
        "source_evidence": _file_meta(source_evidence_path),
        "renderer_transform": {
            "source_bytes": len(context["source_renderer"]),
            "source_sha256": _sha256_bytes(context["source_renderer"]),
            "output_bytes": len(renderer),
            "output_sha256": _sha256_bytes(renderer),
            **transform_meta,
        },
        "pre_archive_parity": pre_archive_parity,
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_bytes,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_source_archive": delta,
            "formula_only_rate_delta_vs_source_archive": 25.0 * delta / ORIGINAL_VIDEO_BYTES,
            "frontier_score_if_only_bytes_change": (
                FRONTIER_SCORE + 25.0 * delta / ORIGINAL_VIDEO_BYTES
            ),
            "frontier_score_target": TARGET_SCORE,
        },
        "non_renderer_preservation": {
            "all_non_renderer_members_preserved": True,
            "members": shrink_builder._member_meta(context["non_renderer_members"]),
        },
        "payload": {
            "member_name": shrink_builder.PR75_PAYLOAD_MEMBER,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            **payload_meta,
        },
        "runtime_contract": {
            "byte_closed": True,
            "single_payload_member": True,
            "runtime_unpack_verified": True,
            "runtime_unpack_summary": runtime_unpack,
            "renderer_only_transplant": True,
            "pre_archive_parity_ran_before_archive_build": True,
            "pose_safety_preflight_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
        },
    }
    (candidate_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _run_strict_pose_safety(
    *,
    source_archive: Path,
    candidate_archive: Path,
    output_json: Path,
    max_pairs: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    try:
        report = pose_safety.build_pose_safety_preflight(
            source_archive=source_archive,
            candidate_archive=candidate_archive,
            output_json=output_json,
            max_pairs=max_pairs,
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
    except Exception as exc:
        report = {
            "schema": pose_safety.SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "remote_gpu_dispatch_performed": False,
            "safe_for_exact_eval_dispatch": False,
            "failure_class": "renderer_transplant_pose_safety_exception",
            "fail_closed_reasons": [type(exc).__name__],
            "exception": str(exc),
        }
        output_json.write_bytes(_json_bytes(report))
    return report


def _candidate_row(
    *,
    transform: TransformSpec,
    renderer: bytes | None = None,
    transform_meta: dict[str, Any] | None = None,
    pre_archive_parity: dict[str, Any] | None = None,
    temp_manifest: dict[str, Any] | None = None,
    strict_pose_safety: dict[str, Any] | None = None,
    emitted_dir: Path | None = None,
    failure_class: str | None,
    fail_closed_reasons: list[str] | None = None,
) -> dict[str, Any]:
    archive = None
    output_archive = None
    if emitted_dir is not None and temp_manifest is not None:
        archive = emitted_dir / "archive.zip"
        output_archive = {
            **temp_manifest["output_archive"],
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
        }
    elif temp_manifest is not None:
        output_archive = {
            "not_emitted_reason": failure_class,
            "screened_bytes": temp_manifest["output_archive"]["bytes"],
            "screened_sha256": temp_manifest["output_archive"]["sha256"],
            "delta_bytes_vs_source_archive": temp_manifest["output_archive"][
                "delta_bytes_vs_source_archive"
            ],
            "frontier_score_if_only_bytes_change": temp_manifest["output_archive"][
                "frontier_score_if_only_bytes_change"
            ],
        }
    return {
        "candidate_id": transform.candidate_id,
        "transform_spec": transform.spec,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_eval_ready": bool(
            strict_pose_safety
            and strict_pose_safety.get("safe_for_exact_eval_dispatch")
            and emitted_dir is not None
        ),
        "failure_class": failure_class,
        "fail_closed_reasons": fail_closed_reasons or [],
        "renderer": (
            None
            if renderer is None
            else {
                "bytes": len(renderer),
                "sha256": _sha256_bytes(renderer),
                **(transform_meta or {}),
            }
        ),
        "pre_archive_parity": pre_archive_parity,
        "strict_pose_safety": (
            None
            if strict_pose_safety is None
            else {
                "safe_for_exact_eval_dispatch": bool(
                    strict_pose_safety.get("safe_for_exact_eval_dispatch")
                ),
                "failure_class": strict_pose_safety.get("failure_class"),
                "fail_closed_reasons": strict_pose_safety.get("fail_closed_reasons", []),
                "output_parity": strict_pose_safety.get("output_parity"),
            }
        ),
        "archive": None if archive is None else str(archive),
        "archive_bytes": None if archive is None else archive.stat().st_size,
        "archive_sha256": None if archive is None else _sha256_file(archive),
        "output_archive": output_archive,
        "remote_gpu_dispatch_performed": False,
        "dispatch_recommendation": (
            "exact_eval_ready_no_dispatch"
            if strict_pose_safety
            and strict_pose_safety.get("safe_for_exact_eval_dispatch")
            and emitted_dir is not None
            else "do_not_dispatch"
        ),
    }


def _recommend_dispatch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row.get("exact_eval_ready")]
    if not ready:
        return {
            "recommendation": "do_not_dispatch",
            "reason": (
                "no tensor-local renderer shrink passed pre-archive render parity "
                "and strict archive pose-safety"
            ),
            "claim_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
        }
    best = min(ready, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    return {
        "recommendation": "exact_eval_ready_no_dispatch",
        "reason": (
            "candidate passed local pre-archive parity and strict pose-safety; "
            "exact CUDA auth eval is required before any score claim"
        ),
        "candidate": best,
        "claim_required_before_dispatch": True,
        "claim_tool": "tools/claim_lane_dispatch.py claim ...",
        "remote_gpu_dispatch_performed": False,
    }


def run_search(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    source_evidence_path: Path | None = DEFAULT_SOURCE_EVIDENCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    transform_specs: tuple[TransformSpec, ...] = default_transform_specs(),
    brotli_quality: int = 11,
    max_pairs: int = 5,
    strict_max_pairs: int = pose_safety.DEFAULT_MAX_PAIRS,
    max_mean_abs_delta: float = 0.02,
    max_rms_delta: float = 0.04,
    max_max_abs_delta: float = 0.75,
    strict_max_mean_abs_delta: float = pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    strict_max_rms_delta: float = pose_safety.DEFAULT_MAX_RMS_DELTA,
    strict_max_max_abs_delta: float = pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    require_byte_saving: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    if max_pairs <= 0 or strict_max_pairs <= 0:
        raise ValueError("max pair counts must be positive")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _source_context(source_archive, brotli_quality=brotli_quality)
    source_warning = None
    if (
        context["source_archive"] == DEFAULT_SOURCE_ARCHIVE.resolve()
        and context["source_sha256"] != SOURCE_ARCHIVE_SHA256
    ):
        source_warning = {
            "expected_default_sha256": SOURCE_ARCHIVE_SHA256,
            "actual_sha256": context["source_sha256"],
            "warning": "default PR79 source archive SHA differs from recorded frontier SHA",
        }

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pr79_parity_shrink_") as tmp:
        tmp_root = Path(tmp)
        source_state = pose_safety._load_runtime_state(
            tmp_root / "source_runtime",
            context["source_members"],
        )
        pair_count_available = _pair_count_for_state(source_state)
        pair_indices = pose_safety._select_pair_indices(pair_count_available, max_pairs)
        source_frames = pose_safety._render_pair_batch(
            renderer=source_state["renderer"],
            masks=source_state["masks"],
            poses=source_state["poses"],
            pair_indices=pair_indices,
        )
        source_output_summary = pose_safety._summarize_frames("source", source_frames)

        for transform in transform_specs:
            try:
                renderer, transform_meta = _encode_transform_renderer(
                    source_state=context["source_state"],
                    source_renderer=context["source_renderer"],
                    source_block_size=context["source_block_size"],
                    transform=transform,
                )
            except Exception as exc:
                rows.append(
                    _candidate_row(
                        transform=transform,
                        failure_class="renderer_transform_failed",
                        fail_closed_reasons=[type(exc).__name__, str(exc)],
                    )
                )
                continue
            if not transform_meta.get("renderer_changed"):
                rows.append(
                    _candidate_row(
                        transform=transform,
                        renderer=renderer,
                        transform_meta=transform_meta,
                        failure_class="renderer_payload_unchanged_or_surrogate",
                        fail_closed_reasons=["renderer_payload_unchanged_or_surrogate"],
                    )
                )
                continue

            try:
                pre_archive_parity = _build_pre_archive_parity(
                    context=context,
                    renderer=renderer,
                    source_state=source_state,
                    source_frames=source_frames,
                    source_output_summary=source_output_summary,
                    pair_indices=pair_indices,
                    pair_count_available=pair_count_available,
                    work_root=tmp_root / transform.candidate_id,
                    max_mean_abs_delta=max_mean_abs_delta,
                    max_rms_delta=max_rms_delta,
                    max_max_abs_delta=max_max_abs_delta,
                )
            except Exception as exc:
                rows.append(
                    _candidate_row(
                        transform=transform,
                        renderer=renderer,
                        transform_meta=transform_meta,
                        failure_class="pre_archive_render_parity_exception",
                        fail_closed_reasons=[type(exc).__name__, str(exc)],
                    )
                )
                continue
            if not pre_archive_parity.get("ok"):
                rows.append(
                    _candidate_row(
                        transform=transform,
                        renderer=renderer,
                        transform_meta=transform_meta,
                        pre_archive_parity=pre_archive_parity,
                        failure_class="pre_archive_render_output_parity_unsafe",
                        fail_closed_reasons=[
                            pre_archive_parity.get("failure_class")
                            or "render_output_parity_unsafe"
                        ],
                    )
                )
                continue

            temp_candidate_dir = tmp_root / "archives" / transform.candidate_id
            try:
                manifest = _build_temp_archive(
                    context=context,
                    transform=transform,
                    renderer=renderer,
                    transform_meta=transform_meta,
                    candidate_dir=temp_candidate_dir,
                    brotli_quality=brotli_quality,
                    pre_archive_parity=pre_archive_parity,
                    source_evidence_path=source_evidence_path,
                )
            except Exception as exc:
                rows.append(
                    _candidate_row(
                        transform=transform,
                        renderer=renderer,
                        transform_meta=transform_meta,
                        pre_archive_parity=pre_archive_parity,
                        failure_class="archive_build_or_runtime_unpack_failed",
                        fail_closed_reasons=[type(exc).__name__, str(exc)],
                    )
                )
                continue
            delta = int(manifest["output_archive"]["delta_bytes_vs_source_archive"])
            if require_byte_saving and delta >= 0:
                rows.append(
                    _candidate_row(
                        transform=transform,
                        renderer=renderer,
                        transform_meta=transform_meta,
                        pre_archive_parity=pre_archive_parity,
                        temp_manifest=manifest,
                        failure_class="candidate_archive_not_byte_saving",
                        fail_closed_reasons=["candidate_archive_not_byte_saving"],
                    )
                )
                continue

            strict_json = temp_candidate_dir / "pose_safety_preflight.json"
            strict = _run_strict_pose_safety(
                source_archive=context["source_archive"],
                candidate_archive=Path(manifest["output_archive"]["path"]),
                output_json=strict_json,
                max_pairs=strict_max_pairs,
                max_mean_abs_delta=strict_max_mean_abs_delta,
                max_rms_delta=strict_max_rms_delta,
                max_max_abs_delta=strict_max_max_abs_delta,
            )
            if not strict.get("safe_for_exact_eval_dispatch"):
                rows.append(
                    _candidate_row(
                        transform=transform,
                        renderer=renderer,
                        transform_meta=transform_meta,
                        pre_archive_parity=pre_archive_parity,
                        temp_manifest=manifest,
                        strict_pose_safety=strict,
                        failure_class="strict_pose_safety_failed",
                        fail_closed_reasons=strict.get("fail_closed_reasons", []),
                    )
                )
                continue

            emitted_dir = output_dir / transform.candidate_id
            if emitted_dir.exists():
                shutil.rmtree(emitted_dir)
            shutil.copytree(temp_candidate_dir, emitted_dir)
            emitted_manifest = json.loads((emitted_dir / "build_manifest.json").read_text())
            emitted_manifest["exact_eval_ready"] = True
            emitted_manifest["strict_pose_safety"] = {
                "path": str(emitted_dir / "pose_safety_preflight.json"),
                "safe_for_exact_eval_dispatch": True,
                "failure_class": strict.get("failure_class"),
                "fail_closed_reasons": strict.get("fail_closed_reasons", []),
            }
            (emitted_dir / "build_manifest.json").write_bytes(_json_bytes(emitted_manifest))
            rows.append(
                _candidate_row(
                    transform=transform,
                    renderer=renderer,
                    transform_meta=transform_meta,
                    pre_archive_parity=pre_archive_parity,
                    temp_manifest=emitted_manifest,
                    strict_pose_safety=strict,
                    emitted_dir=emitted_dir,
                    failure_class=None,
                )
            )

    rows = sorted(
        rows,
        key=lambda row: (
            row["archive_bytes"] if row["archive_bytes"] is not None else 10**12,
            row["candidate_id"],
        ),
    )
    recommendation = _recommend_dispatch(rows)
    summary = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "output_dir": str(output_dir),
        "source_archive": {
            "path": str(context["source_archive"]),
            "bytes": len(context["source_bytes"]),
            "sha256": context["source_sha256"],
        },
        "source_sha_warning": source_warning,
        "source_evidence": _file_meta(source_evidence_path),
        "frontier_score": FRONTIER_SCORE,
        "target_score": TARGET_SCORE,
        "byte_delta_needed_to_cross_target_if_components_unchanged": math.floor(
            (TARGET_SCORE - FRONTIER_SCORE) * ORIGINAL_VIDEO_BYTES / 25.0
        ),
        "transform_specs": [transform.spec for transform in transform_specs],
        "pre_archive_thresholds": {
            "max_pairs": max_pairs,
            "max_mean_abs_delta": max_mean_abs_delta,
            "max_rms_delta": max_rms_delta,
            "max_max_abs_delta": max_max_abs_delta,
        },
        "strict_pose_safety_thresholds": {
            "max_pairs": strict_max_pairs,
            "max_mean_abs_delta": strict_max_mean_abs_delta,
            "max_rms_delta": strict_max_rms_delta,
            "max_max_abs_delta": strict_max_max_abs_delta,
        },
        "require_byte_saving": require_byte_saving,
        "candidate_count": len(rows),
        "exact_eval_ready_candidate_count": sum(1 for row in rows if row["exact_eval_ready"]),
        "candidates": rows,
        "dispatch_recommendation": recommendation,
    }
    (output_dir / "candidate_matrix.json").write_bytes(_json_bytes(summary))
    (output_dir / "dispatch_recommendation.json").write_bytes(
        _json_bytes(recommendation)
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--source-evidence-path", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transform",
        action="append",
        default=None,
        help=(
            "Repeatable transform. Supported: qzs3-reblock:<block_size> and "
            "zero-fp4-prefix:<prefix>:<threshold>."
        ),
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--strict-max-pairs", type=int, default=pose_safety.DEFAULT_MAX_PAIRS)
    parser.add_argument("--max-mean-abs-delta", type=float, default=0.02)
    parser.add_argument("--max-rms-delta", type=float, default=0.04)
    parser.add_argument("--max-max-abs-delta", type=float, default=0.75)
    parser.add_argument(
        "--strict-max-mean-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    )
    parser.add_argument(
        "--strict-max-rms-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_RMS_DELTA,
    )
    parser.add_argument(
        "--strict-max-max-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    )
    parser.add_argument("--allow-non-byte-saving", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    transforms = (
        tuple(parse_transform_spec(raw) for raw in args.transform)
        if args.transform
        else default_transform_specs()
    )
    summary = run_search(
        source_archive=args.source_archive,
        source_evidence_path=args.source_evidence_path,
        output_dir=args.output_dir,
        transform_specs=transforms,
        brotli_quality=args.brotli_quality,
        max_pairs=args.max_pairs,
        strict_max_pairs=args.strict_max_pairs,
        max_mean_abs_delta=args.max_mean_abs_delta,
        max_rms_delta=args.max_rms_delta,
        max_max_abs_delta=args.max_max_abs_delta,
        strict_max_mean_abs_delta=args.strict_max_mean_abs_delta,
        strict_max_rms_delta=args.strict_max_rms_delta,
        strict_max_max_abs_delta=args.strict_max_max_abs_delta,
        require_byte_saving=not args.allow_non_byte_saving,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
