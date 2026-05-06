#!/usr/bin/env python3
"""Search PR75-preserving renderer shrink candidates with local parity gates.

This is a local archive-builder and preflight orchestrator. It preserves every
non-renderer logical member from a source archive, applies small structured
renderer transforms, writes deterministic byte-closed archives, and optionally
runs the renderer-transplant pose-safety preflight before any dispatch
recommendation. It does not dispatch remote work and does not make score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_blockfp_c067_archive as blockfp  # noqa: E402
from experiments import build_mixed_qzs_block_candidate as mixed_qzs  # noqa: E402
from experiments import build_renderer_shrink_candidate as pr75_shrink  # noqa: E402
from experiments import preflight_renderer_transplant_pose_safety as pose_safety  # noqa: E402
from tac.quantizr_qzs3_codec import (  # noqa: E402
    _is_fp4_weight_name,
    decode_qzs3_state_dict,
    encode_qzs3_state_dict,
)


SCHEMA = "renderer_parity_shrink_search_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr79_minp_v2_public_replay_t4_20260503T1615Z/archive.zip"
)
DEFAULT_SOURCE_EVIDENCE = DEFAULT_SOURCE_ARCHIVE.with_name("contest_auth_eval.json")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/renderer_parity_shrink_search_20260503_worker"
)
FRONTIER_SCORE = 0.31457805357318636
TARGET_SCORE = 0.314
SOURCE_ARCHIVE_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"


@dataclass(frozen=True)
class TransformSpec:
    """A structured renderer transform to screen."""

    kind: str
    prefix: str
    value: float | int

    @property
    def candidate_id(self) -> str:
        if self.kind == "zero_fp4_prefix":
            value = f"{float(self.value):.5f}".rstrip("0").rstrip(".")
            return _slug(f"zero_fp4_{self.prefix}_{value}")
        if self.kind == "mixed_block_prefix":
            return _slug(f"mixed_block_{self.prefix}_{int(self.value)}")
        raise ValueError(f"unsupported transform kind: {self.kind!r}")

    @property
    def spec(self) -> str:
        if self.kind == "zero_fp4_prefix":
            return f"zero-fp4-prefix:{self.prefix}:{float(self.value):.6g}"
        if self.kind == "mixed_block_prefix":
            return f"mixed-block-prefix:{self.prefix}:{int(self.value)}"
        raise ValueError(f"unsupported transform kind: {self.kind!r}")


class _PrefixBlockPolicy:
    """Small adapter for the existing MQZ1 per-prefix encoder."""

    def __init__(self, prefix: str, block_size: int, *, default_block_size: int) -> None:
        self.name = _slug(f"mixed_block_{prefix}_{block_size}")
        self.spec = f"mixed-block-prefix:{prefix}:{block_size}"
        self.prefix_overrides = ((prefix, int(block_size)),)
        self.default_block_size = int(default_block_size)
        self.exact_evaluable_archive = True
        self.component_awareness = None

    def block_size_for(self, tensor_name: str) -> int:
        for prefix, block_size in self.prefix_overrides:
            if tensor_name == prefix or tensor_name.startswith(prefix + "."):
                return block_size
        return self.default_block_size

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "default_block_size": self.default_block_size,
            "prefix_overrides": [
                {"prefix": prefix, "block_size": block_size}
                for prefix, block_size in self.prefix_overrides
            ],
            "exact_evaluable_archive": self.exact_evaluable_archive,
        }


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
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    meta: dict[str, Any] = {
        "path": str(resolved),
        "exists": True,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if resolved.suffix == ".json":
        try:
            payload = json.loads(resolved.read_text())
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
            provenance = payload.get("provenance")
            if isinstance(provenance, dict):
                for key in ("archive_sha256", "archive_size_bytes", "device", "gpu_model"):
                    if key in provenance:
                        meta["summary"][f"provenance_{key}"] = provenance[key]
    return meta


def parse_transform_spec(raw: str) -> TransformSpec:
    """Parse a transform spec from the CLI grammar."""

    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(
            "transform spec must be zero-fp4-prefix:<prefix>:<threshold> "
            "or mixed-block-prefix:<prefix>:<block_size>"
        )
    kind, prefix, value_raw = parts
    if not prefix or "/" in prefix or "\\" in prefix or prefix.startswith("."):
        raise ValueError(f"unsafe or empty tensor prefix: {prefix!r}")
    if kind == "zero-fp4-prefix":
        threshold = float(value_raw)
        if threshold < 0.0 or threshold > 1.0 or not math.isfinite(threshold):
            raise ValueError(f"zero threshold must be finite and in [0, 1], got {threshold}")
        return TransformSpec("zero_fp4_prefix", prefix, threshold)
    if kind == "mixed-block-prefix":
        block_size = int(value_raw)
        if block_size <= 0 or block_size > 4096:
            raise ValueError(f"block size must be in [1, 4096], got {block_size}")
        return TransformSpec("mixed_block_prefix", prefix, block_size)
    raise ValueError(f"unsupported transform kind: {kind!r}")


def default_transform_specs() -> tuple[TransformSpec, ...]:
    """Return a bounded high-EV local search set."""

    raw_specs = (
        "zero-fp4-prefix:all_fp4:0.075",
        "zero-fp4-prefix:all_fp4:0.10",
        "zero-fp4-prefix:shared_trunk:0.075",
        "zero-fp4-prefix:shared_trunk:0.10",
        "zero-fp4-prefix:frame2_head:0.075",
        "zero-fp4-prefix:frame2_head:0.10",
        "zero-fp4-prefix:frame2_head.block2:0.10",
        "zero-fp4-prefix:frame2_head.pre:0.10",
    )
    return tuple(parse_transform_spec(raw) for raw in raw_specs)


def _matches_prefix(name: str, prefix: str) -> bool:
    return prefix == "all_fp4" or name == prefix or name.startswith(prefix + ".")


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _apply_zero_fp4_prefix(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
    threshold_fraction: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    out = _clone_state(state)
    changed_tensors: list[dict[str, Any]] = []
    total_values = 0
    zeroed_values = 0
    for name, tensor in out.items():
        if not _is_fp4_weight_name(name) or not _matches_prefix(name, prefix):
            continue
        max_abs = float(tensor.abs().max().item())
        total_values += int(tensor.numel())
        if max_abs <= 0.0:
            continue
        mask = tensor.abs() <= threshold_fraction * max_abs
        zeroed = int(mask.sum().item())
        if zeroed:
            tensor[mask] = 0
            zeroed_values += zeroed
            changed_tensors.append(
                {
                    "name": name,
                    "numel": int(tensor.numel()),
                    "zeroed_values": zeroed,
                    "zeroed_fraction": zeroed / float(tensor.numel()),
                    "max_abs": max_abs,
                }
            )
    return out, {
        "transform_family": "qzs3_same_block_fp4_threshold_zero",
        "prefix": prefix,
        "threshold_fraction_of_tensor_max_abs": float(threshold_fraction),
        "changed_tensor_count": len(changed_tensors),
        "eligible_value_count": total_values,
        "zeroed_value_count": zeroed_values,
        "zeroed_fraction_of_eligible": (
            zeroed_values / float(total_values) if total_values else 0.0
        ),
        "changed_tensors": changed_tensors,
    }


def _encode_transform_renderer(
    *,
    source_state: dict[str, torch.Tensor],
    source_block_size: int,
    transform: TransformSpec,
) -> tuple[bytes, dict[str, Any]]:
    if transform.kind == "zero_fp4_prefix":
        transformed_state, transform_meta = _apply_zero_fp4_prefix(
            source_state,
            prefix=transform.prefix,
            threshold_fraction=float(transform.value),
        )
        renderer = encode_qzs3_state_dict(transformed_state, block_size=source_block_size)
        decode_qzs3_state_dict(renderer, device="cpu")
        return renderer, {
            "wire_format": "QZS3",
            "source_block_size": source_block_size,
            "output_block_size": source_block_size,
            **transform_meta,
        }
    if transform.kind == "mixed_block_prefix":
        policy = _PrefixBlockPolicy(
            transform.prefix,
            int(transform.value),
            default_block_size=source_block_size,
        )
        renderer, policy_meta = mixed_qzs.encode_mixed_qzs_block_payload(
            source_state,
            policy,
        )
        return renderer, {
            "wire_format": "MQZ1",
            "source_block_size": source_block_size,
            "block_policy": policy_meta,
        }
    raise ValueError(f"unsupported transform kind: {transform.kind!r}")


def _source_context(source_archive: Path, *, brotli_quality: int) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    source_bytes = source_archive.read_bytes()
    source_members, source_packaging = blockfp.extract_runtime_members(source_archive)
    if pr75_shrink.RENDERER_MEMBER not in source_members:
        raise ValueError("source archive missing renderer.bin")
    source_renderer = source_members[pr75_shrink.RENDERER_MEMBER]
    if not source_renderer.startswith(b"QZS3"):
        raise ValueError(f"source renderer must be QZS3, got {source_renderer[:4]!r}")
    pr75_slices = pr75_shrink.load_pr75_slices_for_renderer_shrink(
        source_archive,
        source_members,
        brotli_quality=brotli_quality,
    )
    if pr75_slices is None:
        raise ValueError("renderer parity shrink search currently requires a PR75 single p payload")
    non_renderer_members = {
        name: payload
        for name, payload in source_members.items()
        if name != pr75_shrink.RENDERER_MEMBER
    }
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
        "non_renderer_members": non_renderer_members,
    }


def _candidate_manifest(
    *,
    context: dict[str, Any],
    transform: TransformSpec,
    renderer: bytes,
    payload: bytes,
    payload_meta: dict[str, Any],
    output_archive: Path,
    transform_meta: dict[str, Any],
    source_evidence_path: Path | None,
    runtime_unpack: dict[str, Any],
) -> dict[str, Any]:
    archive_bytes = output_archive.stat().st_size
    source_archive = context["source_archive"]
    source_bytes_len = len(context["source_bytes"])
    delta = archive_bytes - source_bytes_len
    byte_only_score = FRONTIER_SCORE + 25.0 * delta / ORIGINAL_VIDEO_BYTES
    return {
        "schema": SCHEMA,
        "tool": "experiments/search_renderer_parity_shrink_candidate.py",
        "candidate_id": transform.candidate_id,
        "transform_spec": transform.spec,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_archive_candidate_until_pose_safety_and_exact_cuda",
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_bytes_len,
            "sha256": context["source_sha256"],
            **context["source_packaging"],
        },
        "source_evidence": _file_meta(source_evidence_path),
        "output_archive": {
            "path": str(output_archive),
            "bytes": archive_bytes,
            "sha256": _sha256_file(output_archive),
            "delta_bytes_vs_source_archive": delta,
            "formula_only_rate_delta_vs_source_archive": (
                25.0 * delta / ORIGINAL_VIDEO_BYTES
            ),
            "frontier_score_if_only_bytes_change": byte_only_score,
            "frontier_score_target": TARGET_SCORE,
            "byte_only_crosses_target": byte_only_score < TARGET_SCORE,
        },
        "renderer_transform": {
            "source_bytes": len(context["source_renderer"]),
            "source_sha256": _sha256_bytes(context["source_renderer"]),
            "output_bytes": len(renderer),
            "output_sha256": _sha256_bytes(renderer),
            **transform_meta,
        },
        "non_renderer_preservation": {
            "all_non_renderer_members_preserved": True,
            "members": pr75_shrink._member_meta(context["non_renderer_members"]),
        },
        "payload": {
            "member_name": pr75_shrink.PR75_PAYLOAD_MEMBER,
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
            "pose_safety_preflight_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
            "canonical_score_source_required": (
                "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                "experiments/contest_auth_eval.py --device cuda"
            ),
        },
    }


def build_candidate(
    *,
    context: dict[str, Any],
    transform: TransformSpec,
    candidate_dir: Path,
    brotli_quality: int,
    source_evidence_path: Path | None,
) -> dict[str, Any]:
    renderer, transform_meta = _encode_transform_renderer(
        source_state=context["source_state"],
        source_block_size=context["source_block_size"],
        transform=transform,
    )
    payload, payload_meta = pr75_shrink._build_pr75_payload(
        context["pr75_slices"],
        renderer_bytes=renderer,
        brotli_quality=brotli_quality,
    )
    archive_path = candidate_dir / "archive.zip"
    pr75_shrink._write_single_member_archive(archive_path, payload)
    runtime_unpack = pr75_shrink._verify_archive(
        archive_path,
        expected_renderer=renderer,
        expected_non_renderer_members=context["non_renderer_members"],
    )
    manifest = _candidate_manifest(
        context=context,
        transform=transform,
        renderer=renderer,
        payload=payload,
        payload_meta=payload_meta,
        output_archive=archive_path,
        transform_meta=transform_meta,
        source_evidence_path=source_evidence_path,
        runtime_unpack=runtime_unpack,
    )
    (candidate_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _preflight_candidate(
    *,
    source_archive: Path,
    candidate_manifest: dict[str, Any],
    max_pairs: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    archive_path = Path(candidate_manifest["output_archive"]["path"])
    output_json = archive_path.with_name("pose_safety_preflight.json")
    try:
        report = pose_safety.build_pose_safety_preflight(
            source_archive=source_archive,
            candidate_archive=archive_path,
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
    return {
        "path": str(output_json),
        "safe_for_exact_eval_dispatch": bool(report.get("safe_for_exact_eval_dispatch")),
        "failure_class": report.get("failure_class"),
        "fail_closed_reasons": report.get("fail_closed_reasons", []),
        "output_parity": report.get("output_parity"),
    }


def _summary_candidate_row(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": manifest["candidate_id"],
        "transform_spec": manifest["transform_spec"],
        "archive": manifest["output_archive"]["path"],
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "delta_bytes_vs_source_archive": manifest["output_archive"][
            "delta_bytes_vs_source_archive"
        ],
        "frontier_score_if_only_bytes_change": manifest["output_archive"][
            "frontier_score_if_only_bytes_change"
        ],
        "byte_only_crosses_target": manifest["output_archive"]["byte_only_crosses_target"],
        "renderer_wire_format": manifest["renderer_transform"]["wire_format"],
        "score_claim": False,
        "promotion_eligible": False,
    }


def _recommend_dispatch(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [
        item
        for item in candidates
        if item.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
    ]
    target_safe = [
        item
        for item in safe
        if item["frontier_score_if_only_bytes_change"] < TARGET_SCORE
    ]
    if target_safe:
        best = min(target_safe, key=lambda item: item["archive_bytes"])
        return {
            "recommendation": "exact_eval_ready_no_dispatch",
            "reason": (
                "candidate passed local pose-safety and byte-only rate math crosses "
                f"target {TARGET_SCORE}"
            ),
            "candidate": best,
            "claim_required_before_dispatch": True,
            "pose_safety_preflight_required_before_dispatch": True,
            "claim_tool": "tools/claim_lane_dispatch.py claim ...",
            "remote_gpu_dispatch_performed": False,
        }
    if safe:
        best = min(safe, key=lambda item: item["archive_bytes"])
        return {
            "recommendation": "exact_eval_ready_no_dispatch",
            "reason": (
                "candidate passed local pose-safety and saves bytes; exact CUDA "
                "eval is required for any score claim"
            ),
            "candidate": best,
            "claim_required_before_dispatch": True,
            "pose_safety_preflight_required_before_dispatch": True,
            "claim_tool": "tools/claim_lane_dispatch.py claim ...",
            "remote_gpu_dispatch_performed": False,
        }
    return {
        "recommendation": "do_not_dispatch",
        "reason": "no byte-saving candidate passed local renderer-transplant pose-safety",
        "claim_required_before_dispatch": True,
        "remote_gpu_dispatch_performed": False,
    }


def run_search(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    source_evidence_path: Path | None = DEFAULT_SOURCE_EVIDENCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    transform_specs: tuple[TransformSpec, ...] = default_transform_specs(),
    brotli_quality: int = 11,
    max_preflight_candidates: int = 8,
    preflight_max_pairs: int = pose_safety.DEFAULT_MAX_PAIRS,
    max_mean_abs_delta: float = pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    max_rms_delta: float = pose_safety.DEFAULT_MAX_RMS_DELTA,
    max_max_abs_delta: float = pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    force: bool = False,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    """Build candidates, run local pose-safety on the best byte-savers, and summarize."""

    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _source_context(source_archive, brotli_quality=brotli_quality)
    source_sha = context["source_sha256"]
    source_warning = None
    if source_archive.resolve() == DEFAULT_SOURCE_ARCHIVE.resolve() and source_sha != SOURCE_ARCHIVE_SHA256:
        source_warning = {
            "expected_default_sha256": SOURCE_ARCHIVE_SHA256,
            "actual_sha256": source_sha,
            "warning": "default frontier source archive SHA differs from the recorded frontier SHA",
        }

    manifests: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for transform in transform_specs:
        candidate_dir = output_dir / transform.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest = build_candidate(
                context=context,
                transform=transform,
                candidate_dir=candidate_dir,
                brotli_quality=brotli_quality,
                source_evidence_path=source_evidence_path,
            )
        except Exception as exc:
            rejection = {
                "schema": SCHEMA,
                "candidate_id": transform.candidate_id,
                "transform_spec": transform.spec,
                "build_ok": False,
                "score_claim": False,
                "promotion_eligible": False,
                "remote_gpu_dispatch_performed": False,
                "failure_class": "candidate_build_or_runtime_unpack_failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
            (candidate_dir / "build_rejected.json").write_bytes(_json_bytes(rejection))
            rejected.append(rejection)
            continue
        manifests.append(manifest)

    rows = [_summary_candidate_row(manifest) for manifest in manifests]
    byte_savers = [
        item
        for item in sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
        if item["delta_bytes_vs_source_archive"] < 0
    ]
    preflight_targets = [] if skip_preflight else byte_savers[:max_preflight_candidates]
    preflight_target_ids = {item["candidate_id"] for item in preflight_targets}
    manifest_by_id = {manifest["candidate_id"]: manifest for manifest in manifests}
    for row in rows:
        if row["candidate_id"] not in preflight_target_ids:
            row["pose_safety"] = {
                "preflight_ran": False,
                "safe_for_exact_eval_dispatch": False,
                "failure_class": "pose_safety_preflight_not_run",
            }
            continue
        pose_report = _preflight_candidate(
            source_archive=context["source_archive"],
            candidate_manifest=manifest_by_id[row["candidate_id"]],
            max_pairs=preflight_max_pairs,
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
        row["pose_safety"] = {"preflight_ran": True, **pose_report}

    recommendation = _recommend_dispatch(rows) if rows else {
        "recommendation": "do_not_dispatch",
        "reason": "no candidate archive passed build/runtime-unpack verification",
        "claim_required_before_dispatch": True,
        "remote_gpu_dispatch_performed": False,
    }
    summary = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "output_dir": str(output_dir),
        "source_archive": {
            "path": str(context["source_archive"]),
            "bytes": len(context["source_bytes"]),
            "sha256": source_sha,
        },
        "source_sha_warning": source_warning,
        "source_evidence": _file_meta(source_evidence_path),
        "frontier_score": FRONTIER_SCORE,
        "target_score": TARGET_SCORE,
        "byte_delta_needed_to_cross_target_if_components_unchanged": math.floor(
            (TARGET_SCORE - FRONTIER_SCORE) * ORIGINAL_VIDEO_BYTES / 25.0
        ),
        "transform_specs": [transform.spec for transform in transform_specs],
        "candidate_count": len(rows),
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
        "preflight_thresholds": {
            "max_pairs": preflight_max_pairs,
            "max_mean_abs_delta": max_mean_abs_delta,
            "max_rms_delta": max_rms_delta,
            "max_max_abs_delta": max_max_abs_delta,
        },
        "max_preflight_candidates": max_preflight_candidates,
        "candidates": sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "best_by_archive_bytes": (
            min(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
            if rows
            else None
        ),
        "dispatch_recommendation": recommendation,
    }
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))
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
            "Repeatable transform. Supported: "
            "zero-fp4-prefix:<prefix>:<threshold>, "
            "mixed-block-prefix:<prefix>:<block_size>. Defaults to a bounded "
            "PR75 renderer-parity shrink search."
        ),
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--max-preflight-candidates", type=int, default=8)
    parser.add_argument(
        "--preflight-max-pairs",
        type=int,
        default=pose_safety.DEFAULT_MAX_PAIRS,
    )
    parser.add_argument(
        "--max-mean-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    )
    parser.add_argument(
        "--max-rms-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_RMS_DELTA,
    )
    parser.add_argument(
        "--max-max-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    )
    parser.add_argument("--skip-preflight", action="store_true")
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
        max_preflight_candidates=args.max_preflight_candidates,
        preflight_max_pairs=args.preflight_max_pairs,
        max_mean_abs_delta=args.max_mean_abs_delta,
        max_rms_delta=args.max_rms_delta,
        max_max_abs_delta=args.max_max_abs_delta,
        force=args.force,
        skip_preflight=args.skip_preflight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
