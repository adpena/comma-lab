#!/usr/bin/env python3
"""Build deterministic mixed/local QZS block-allocation candidates.

Mixed per-tensor QZS block allocation uses the charged MQZ1 renderer wire
format.  MQZ1 is decoded by the contest runtime, so these archives are exact-
evaluable, but they remain non-promotable until the exact archive bytes pass
CUDA auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_renderer_packed_payload_archive import (  # noqa: E402
    PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    POSE_QP1_CODEC,
    build_packed_archive,
)
from experiments.repack_quantizr_faithful_qzs3_archive import (  # noqa: E402
    RENDERER_CODEC_QZS3,
    SUBMISSION_LAYOUT_PR64_MASK_FIRST_SINGLE_BLOB,
    _runtime_members_from_source_archive,
    build_submission_archive,
)
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer  # noqa: E402
from tac.quantizr_qzs3_codec import (  # noqa: E402
    MIXED_QZS_MAGIC,
    QZS3_MAGIC,
    _is_bias_name,
    _is_fp4_weight_name,
    _quantize_fp4_blocks,
    _quantize_qv_tensor,
    decode_mixed_qzs_block_state_dict,
    decode_qzs3_state_dict,
    qzs3_qv_specs,
)


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_POLICIES = (
    "component-aware-v1:frame2_pre64",
    "component-aware-v1:frame2_block2_pre64",
    "component-aware-v1:frame2_all64",
)

BASELINE_SAFE_BLOCK_SIZE = 32
COMPONENT_AWARE_AGGRESSIVE_BLOCK_SIZE = 64
COMPONENT_AWARE_HIGH_RISK_PREFIXES = (
    "shared_trunk",
    "frame1_head",
    "pose_mlp",
)
COMPONENT_AWARE_POLICY_OVERRIDES: dict[str, tuple[tuple[str, int], ...]] = {
    "frame2_pre64": (("frame2_head.pre", COMPONENT_AWARE_AGGRESSIVE_BLOCK_SIZE),),
    "frame2_block2_pre64": (
        ("frame2_head.block2", COMPONENT_AWARE_AGGRESSIVE_BLOCK_SIZE),
        ("frame2_head.pre", COMPONENT_AWARE_AGGRESSIVE_BLOCK_SIZE),
    ),
    "frame2_all64": (("frame2_head", COMPONENT_AWARE_AGGRESSIVE_BLOCK_SIZE),),
}


@dataclass(frozen=True)
class BlockPolicy:
    name: str
    spec: str
    default_block_size: int
    prefix_overrides: tuple[tuple[str, int], ...]
    exact_evaluable_archive: bool
    component_awareness: dict[str, Any] | None = None

    def block_size_for(self, tensor_name: str) -> int:
        for prefix, block_size in self.prefix_overrides:
            if tensor_name == prefix or tensor_name.startswith(prefix + "."):
                return block_size
        return self.default_block_size

    def as_json(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "spec": self.spec,
            "default_block_size": self.default_block_size,
            "prefix_overrides": [
                {"prefix": prefix, "block_size": block_size}
                for prefix, block_size in self.prefix_overrides
            ],
            "exact_evaluable_archive": self.exact_evaluable_archive,
        }
        if self.component_awareness is not None:
            payload["component_awareness"] = self.component_awareness
        return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "policy"


def _validate_block_size(block_size: int) -> int:
    block_size = int(block_size)
    if block_size <= 0 or block_size > 4096:
        raise ValueError(f"invalid QZS block size: {block_size}")
    return block_size


def _component_aware_policy(tier: str) -> BlockPolicy:
    if tier not in COMPONENT_AWARE_POLICY_OVERRIDES:
        expected = ", ".join(sorted(COMPONENT_AWARE_POLICY_OVERRIDES))
        raise ValueError(f"unsupported component-aware-v1 tier {tier!r}; expected one of: {expected}")
    overrides = COMPONENT_AWARE_POLICY_OVERRIDES[tier]
    awareness = {
        "schema": "component_aware_mixed_qzs_policy_v1",
        "policy_family": "component-aware-v1",
        "tier": tier,
        "baseline_block_size": BASELINE_SAFE_BLOCK_SIZE,
        "aggressive_block_size": COMPONENT_AWARE_AGGRESSIVE_BLOCK_SIZE,
        "protected_prefixes": [
            {
                "prefix": "shared_trunk",
                "block_size": BASELINE_SAFE_BLOCK_SIZE,
                "reason": (
                    "shared representation feeds both frames; prior MQZ1 default-64 "
                    "candidate showed PoseNet collapse"
                ),
            },
            {
                "prefix": "frame1_head",
                "block_size": BASELINE_SAFE_BLOCK_SIZE,
                "reason": "pose-conditioned frame head; keep at source QZS3 block size",
            },
            {
                "prefix": "pose_mlp",
                "block_size": BASELINE_SAFE_BLOCK_SIZE,
                "reason": "direct pose-conditioning path; never made more aggressive in this policy",
            },
        ],
        "aggressive_prefixes": [
            {
                "prefix": prefix,
                "block_size": block_size,
                "reason": (
                    "frame2-only static head tensor group; lower direct PoseNet risk "
                    "than shared trunk, frame1 FiLM head, or pose MLP"
                ),
            }
            for prefix, block_size in overrides
        ],
        "selection_rule": (
            "default to the exact-evaluated QZS3 b32 block size, protect all "
            "pose-critical prefixes, and shrink only frame2-only tensor groups"
        ),
    }
    return BlockPolicy(
        name=f"component_aware_v1_{tier}",
        spec=f"component-aware-v1:{tier}",
        default_block_size=BASELINE_SAFE_BLOCK_SIZE,
        prefix_overrides=overrides,
        exact_evaluable_archive=True,
        component_awareness=awareness,
    )


def parse_block_policy(spec: str) -> BlockPolicy:
    raw = spec.strip()
    if not raw:
        raise ValueError("empty block policy")
    if raw.startswith("component-aware-v1:"):
        return _component_aware_policy(raw.split(":", 1)[1].strip())
    if raw.startswith("global:"):
        block_size = _validate_block_size(int(raw.split(":", 1)[1]))
        return BlockPolicy(
            name=f"global_b{block_size}",
            spec=raw,
            default_block_size=block_size,
            prefix_overrides=(),
            exact_evaluable_archive=True,
        )
    if not raw.startswith("mixed:"):
        raise ValueError(
            f"unsupported policy {raw!r}; expected global:<block>, "
            "mixed:<default>:prefix=block,..., or component-aware-v1:<tier>"
        )
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"mixed policy must be mixed:<default>:prefix=block,...; got {raw!r}")
    default_block_size = _validate_block_size(int(parts[1]))
    overrides: list[tuple[str, int]] = []
    for item in parts[2].split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"mixed policy override must be prefix=block; got {item!r}")
        prefix, value = item.split("=", 1)
        prefix = prefix.strip()
        if not prefix or "/" in prefix or "\\" in prefix or prefix.startswith("."):
            raise ValueError(f"unsafe or empty tensor prefix in policy: {prefix!r}")
        overrides.append((prefix, _validate_block_size(int(value))))
    if not overrides:
        raise ValueError(f"mixed policy requires at least one prefix override: {raw!r}")
    return BlockPolicy(
        name=_slug(raw.replace(":", "_").replace(",", "_")),
        spec=raw,
        default_block_size=default_block_size,
        prefix_overrides=tuple(overrides),
        exact_evaluable_archive=True,
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    info.extra = b""
    info.comment = b""
    return info


def _write_runtime_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
            zf.writestr(_zip_info(name), members[name])


def _file_meta(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "exists": False,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_path(resolved),
    }


def _json_evidence_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"json_parseable": False}
    summary: dict[str, Any] = {"json_parseable": True}
    for key in (
        "format",
        "schema_version",
        "tool",
        "evidence_grade",
        "device",
        "score_claim",
        "promotion_eligible",
        "exact_evaluable_archive",
        "response_coverage_passed",
        "same_run_zero_baseline",
        "avg_posenet_dist",
        "avg_segnet_dist",
        "score_recomputed_from_components",
        "archive_size_bytes",
        "n_samples",
    ):
        if key in payload:
            summary[key] = payload[key]
    if "provenance" in payload and isinstance(payload["provenance"], dict):
        provenance = payload["provenance"]
        for key in ("archive_sha256", "archive_size_bytes", "device", "gpu_model"):
            if key in provenance:
                summary[f"provenance_{key}"] = provenance[key]
    if isinstance(payload.get("atoms"), list):
        atoms = payload["atoms"]
        summary["atom_count"] = len(atoms)
        summary["top_atom_layers"] = [
            str(atom.get("layer_name"))
            for atom in atoms[:8]
            if isinstance(atom, dict) and atom.get("layer_name") is not None
        ]
    if isinstance(payload.get("points"), list):
        summary["point_count"] = len(payload["points"])
        summary["point_epsilons"] = [
            item.get("epsilon")
            for item in payload["points"]
            if isinstance(item, dict) and "epsilon" in item
        ]
    avg_pose = payload.get("avg_posenet_dist")
    if isinstance(avg_pose, (int, float)):
        summary["pose_collapse_signal"] = float(avg_pose) > 0.05
    return summary


def _evidence_inputs_metadata(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for path in paths:
        meta = _file_meta(path)
        if meta is not None and meta.get("exists") and path.suffix == ".json":
            meta["summary"] = _json_evidence_summary(path.resolve())
        evidence.append(meta if meta is not None else {"path": str(path), "exists": False})
    return evidence


def encode_mixed_qzs_block_payload(
    state: dict[str, torch.Tensor],
    policy: BlockPolicy,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a deterministic byte-screen renderer payload with local blocks."""

    template_state = build_quantizr_faithful_renderer().state_dict()
    if list(state.keys()) != list(template_state.keys()):
        missing = [key for key in template_state if key not in state]
        extra = [key for key in state if key not in template_state]
        raise ValueError(
            "state dict is not JointFrameGenerator-compatible: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    qv_specs = qzs3_qv_specs()
    packed_parts: list[bytes] = []
    scale_parts: list[bytes] = []
    bias_parts: list[bytes] = []
    dense_fp_parts: list[bytes] = []
    fp_weight_parts: list[bytes] = []
    dense_other_parts: list[bytes] = []
    qv_parts: list[bytes] = []
    fp4_tensors: list[dict[str, Any]] = []
    fp4_block_sizes: list[int] = []
    block_counts: dict[str, int] = {}

    for key, tensor in state.items():
        ref = template_state[key]
        if tuple(tensor.shape) != tuple(ref.shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(ref.shape)}")
        if _is_fp4_weight_name(key):
            block_size = policy.block_size_for(key)
            packed, scales = _quantize_fp4_blocks(tensor, block_size)
            packed_parts.append(packed)
            scale_parts.append(scales)
            fp4_block_sizes.append(block_size)
            block_counts[str(block_size)] = block_counts.get(str(block_size), 0) + 1
            fp4_tensors.append(
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "numel": int(tensor.numel()),
                    "block_size": block_size,
                    "packed_bytes": len(packed),
                    "scale_bytes": len(scales),
                }
            )
        elif key.endswith(".weight") and (
            key == "shared_trunk.embedding.weight"
            or key in {"frame1_head.head.weight", "frame2_head.head.weight"}
        ):
            fp_weight_parts.append(tensor.detach().cpu().to(torch.float16).numpy().tobytes())
        elif _is_bias_name(key):
            bias_parts.append(tensor.detach().cpu().to(torch.float16).numpy().tobytes())
        elif key in qv_specs:
            bits, per_row = qv_specs[key]
            qv_parts.append(_quantize_qv_tensor(tensor, bits, per_row))
        elif torch.is_floating_point(tensor):
            dense_fp_parts.append(tensor.detach().cpu().to(torch.float16).numpy().tobytes())
        else:
            dense_other_parts.append(tensor.detach().cpu().numpy().tobytes())

    segments = {
        "packed": b"".join(packed_parts),
        "scales": b"".join(scale_parts),
        "bias": b"".join(bias_parts),
        "dense_fp": b"".join(dense_fp_parts),
        "fp_weight": b"".join(fp_weight_parts),
        "dense_other": b"".join(dense_other_parts),
        "qv": b"".join(qv_parts),
    }
    header = {
        "schema": "mixed_qzs_block_screen_v1",
        "wire_format": "MQZ1",
        "runtime_decoder_available": True,
        "source_codec_family": "QZS3",
        "policy": {
            "name": policy.name,
            "spec": policy.spec,
        },
        "block_size_counts": {key: block_counts[key] for key in sorted(block_counts, key=int)},
        "fp4_tensor_order": "joint_frame_generator_state_dict_fp4_order_v1",
        "fp4_block_sizes": fp4_block_sizes,
        "segment_bytes": {key: len(value) for key, value in segments.items()},
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = (
        MIXED_QZS_MAGIC
        + struct.pack("<I", len(header_bytes))
        + header_bytes
        + b"".join(segments[key] for key in ("packed", "scales", "bias", "dense_fp", "fp_weight", "dense_other", "qv"))
    )
    return payload, {
        **header,
        "policy": policy.as_json(),
        "charged_policy": header["policy"],
        "charged_header_format": "compact_fp4_block_sizes_v1",
        "fp4_tensors": fp4_tensors,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
        "header_bytes": len(header_bytes),
    }


def _zip_members_meta(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        return {
            info.filename: {
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "sha256": _sha256_bytes(zf.read(info)),
            }
            for info in zf.infolist()
            if not info.is_dir()
        }


def _source_metadata(source_archive: Path, source_evidence_path: Path | None) -> dict[str, Any]:
    return {
        "source_archive": str(source_archive.resolve()),
        "source_archive_bytes": source_archive.resolve().stat().st_size,
        "source_archive_sha256": _sha256_path(source_archive.resolve()),
        "source_evidence": _file_meta(source_evidence_path),
    }


def build_mixed_candidate(
    source_archive: Path,
    output_archive: Path,
    *,
    policy: BlockPolicy,
    source_evidence_path: Path | None = None,
    policy_evidence_paths: tuple[Path, ...] = (),
    pose_codec: str = POSE_QP1_CODEC,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    output_archive = output_archive.resolve()
    runtime_members, source_contract = _runtime_members_from_source_archive(source_archive)
    renderer_raw = runtime_members["renderer.bin"]
    if not renderer_raw.startswith(QZS3_MAGIC):
        raise ValueError(
            "mixed QZS block allocation currently requires a QZS3 source renderer; "
            f"got magic {renderer_raw[:4]!r}"
        )
    state = decode_qzs3_state_dict(renderer_raw, device="cpu")
    mixed_renderer, block_meta = encode_mixed_qzs_block_payload(state, policy)
    decode_mixed_qzs_block_state_dict(mixed_renderer, device="cpu")

    with tempfile.TemporaryDirectory(prefix="mixed_qzs_block_") as tmpdir:
        runtime_zip = Path(tmpdir) / "runtime_members.zip"
        _write_runtime_zip(
            runtime_zip,
            {
                "renderer.bin": mixed_renderer,
                "masks.mkv": runtime_members["masks.mkv"],
                "optimized_poses.bin": runtime_members["optimized_poses.bin"],
            },
        )
        packed_meta = build_packed_archive(
            runtime_zip,
            output_archive,
            brotli_quality=brotli_quality,
            pose_codec=pose_codec,
            payload_member_name="p",
            payload_format=PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
        )

    source = _source_metadata(source_archive, source_evidence_path)
    output_bytes = output_archive.stat().st_size
    return {
        "schema_version": 1,
        "tool": "experiments/build_mixed_qzs_block_candidate.py",
        "score_claim": False,
        "promotion_eligible": False,
        "exact_evaluable_archive": True,
        "exact_evaluable_reason": "mixed/local QZS blocks use the MQZ1 runtime decoder; still not promotable without exact CUDA auth eval",
        "evidence_grade": "empirical_byte_screen_only_until_cuda_auth_eval",
        **source,
        "policy_evidence_inputs": _evidence_inputs_metadata(policy_evidence_paths),
        "source_runtime_contract": source_contract,
        "output_archive": str(output_archive),
        "output_archive_bytes": output_bytes,
        "output_archive_sha256": _sha256_path(output_archive),
        "archive_byte_delta": output_bytes - source["source_archive_bytes"],
        "formula_rate_score_delta": 25.0 * float(output_bytes - source["source_archive_bytes"]) / ORIGINAL_VIDEO_BYTES,
        "block_policy": block_meta,
        "renderer": {
            "source_renderer_format": "QZS3",
            "source_renderer_bytes": len(renderer_raw),
            "source_renderer_sha256": _sha256_bytes(renderer_raw),
            "output_renderer_format": "MQZ1",
            "output_renderer_bytes": len(mixed_renderer),
            "output_renderer_sha256": _sha256_bytes(mixed_renderer),
        },
        "packed_payload_stage": packed_meta,
        "archive_members": _zip_members_meta(output_archive),
    }


def build_global_candidate(
    source_archive: Path,
    output_archive: Path,
    *,
    policy: BlockPolicy,
    source_evidence_path: Path | None = None,
    policy_evidence_paths: tuple[Path, ...] = (),
    pose_codec: str = POSE_QP1_CODEC,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    meta = build_submission_archive(
        source_archive,
        output_archive,
        renderer_codec=RENDERER_CODEC_QZS3,
        qzs3_block_size=policy.default_block_size,
        submission_layout=SUBMISSION_LAYOUT_PR64_MASK_FIRST_SINGLE_BLOB,
        pose_codec=pose_codec,
        brotli_quality=brotli_quality,
    )
    meta.update(
        {
            "tool": "experiments/build_mixed_qzs_block_candidate.py",
            "score_claim": False,
            "promotion_eligible": False,
            "exact_evaluable_archive": True,
            "exact_evaluable_reason": "global QZS3 block size uses the existing QZS3 runtime decoder; still not promotable without exact CUDA auth eval",
            "evidence_grade": "empirical_byte_screen_only_until_cuda_auth_eval",
            "source_evidence": _file_meta(source_evidence_path),
            "policy_evidence_inputs": _evidence_inputs_metadata(policy_evidence_paths),
            "block_policy": {
                "schema": "mixed_qzs_block_policy_v1",
                "wire_format": "QZS3",
                "runtime_decoder_available": True,
                "policy": policy.as_json(),
                "block_size_counts": {str(policy.default_block_size): "all_fp4_tensors"},
            },
            "archive_members": _zip_members_meta(output_archive),
        }
    )
    return meta


def build_candidate_for_policy(
    source_archive: Path,
    output_archive: Path,
    *,
    policy: BlockPolicy,
    source_evidence_path: Path | None = None,
    policy_evidence_paths: tuple[Path, ...] = (),
    pose_codec: str = POSE_QP1_CODEC,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    if not policy.prefix_overrides:
        return build_global_candidate(
            source_archive,
            output_archive,
            policy=policy,
            source_evidence_path=source_evidence_path,
            policy_evidence_paths=policy_evidence_paths,
            pose_codec=pose_codec,
            brotli_quality=brotli_quality,
        )
    return build_mixed_candidate(
        source_archive,
        output_archive,
        policy=policy,
        source_evidence_path=source_evidence_path,
        policy_evidence_paths=policy_evidence_paths,
        pose_codec=pose_codec,
        brotli_quality=brotli_quality,
    )


def build_candidates(
    source_archive: Path,
    output_dir: Path,
    *,
    policy_specs: tuple[str, ...] = DEFAULT_POLICIES,
    source_evidence_path: Path | None = None,
    policy_evidence_paths: tuple[Path, ...] = (),
    pose_codec: str = POSE_QP1_CODEC,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for spec in policy_specs:
        policy = parse_block_policy(spec)
        candidate_dir = output_dir / policy.name
        candidate_archive = candidate_dir / "archive.zip"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        meta = build_candidate_for_policy(
            source_archive,
            candidate_archive,
            policy=policy,
            source_evidence_path=source_evidence_path,
            policy_evidence_paths=policy_evidence_paths,
            pose_codec=pose_codec,
            brotli_quality=brotli_quality,
        )
        (candidate_dir / "build_provenance.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        candidates.append(meta)

    summary = {
        "schema_version": 1,
        "tool": "experiments/build_mixed_qzs_block_candidate.py",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_byte_screen_only_until_cuda_auth_eval",
        "output_dir": str(output_dir),
        "source": _source_metadata(source_archive.resolve(), source_evidence_path),
        "policy_evidence_inputs": _evidence_inputs_metadata(policy_evidence_paths),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "policy": item["block_policy"]["policy"],
                "component_awareness": item["block_policy"]["policy"].get("component_awareness"),
                "output_archive": item["output_archive"],
                "output_archive_bytes": item["output_archive_bytes"],
                "output_archive_sha256": item["output_archive_sha256"],
                "archive_byte_delta": item["archive_byte_delta"],
                "exact_evaluable_archive": item["exact_evaluable_archive"],
                "score_claim": item["score_claim"],
                "promotion_eligible": item["promotion_eligible"],
            }
            for item in candidates
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-evidence-path", type=Path, default=None)
    parser.add_argument(
        "--policy-evidence-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Repeatable component/sensitivity evidence JSON to fingerprint in "
            "candidate provenance. Evidence is planning-only; outputs remain non-promotable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/mixed_local_qzs_block_allocation_20260502"),
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "Repeatable block policy. Supported forms: global:<block> or "
            "mixed:<default>:prefix=block,prefix=block, or "
            "component-aware-v1:<tier>. Defaults to component-aware-v1 screens."
        ),
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    args = parser.parse_args(argv)

    summary = build_candidates(
        args.source_archive,
        args.output_dir,
        policy_specs=tuple(args.policy) if args.policy else DEFAULT_POLICIES,
        source_evidence_path=args.source_evidence_path,
        policy_evidence_paths=tuple(args.policy_evidence_path or ()),
        brotli_quality=args.brotli_quality,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
