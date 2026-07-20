# SPDX-License-Identifier: MIT
"""Fail-closed R1b3 producer preflight and typed ``xi[0]`` payload codec.

The R1b2 assembler audits three producer manifests, but an auditable manifest
is not itself proof that the production receiver consumes the appended bytes.
This module re-derives the available real custody and keeps three distinct
questions separate:

* the exact rank-4 SegNet *head* chart versus the still-owed receiver-coordinate
  pullback;
* offline bounded-uint8/full-kernel evidence versus a search-free n600 replay;
* a counted quantized PoseNet coordinate-zero target versus a receiver actuator
  that can realize that target in decoded frames.

No score, candidate archive, or family verdict is emitted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

PAIR_COUNT: Final = 600
BATCH_SIZE: Final = 16
SEGMENTATION_HEIGHT: Final = 384
SEGMENTATION_WIDTH: Final = 512
SEGMENTATION_CLASS_COUNT: Final = 5
MODERATE_MARGIN_FLIPS: Final = 16_319
TIE_TIGHT_FLIPS: Final = 1_607
GAP_FLIPS: Final = 17_926
XI0_SCHEMA: Final = "r1b2_xi0_custody.v1"
XI0_RECEIVER_SCHEMA: Final = "r1b2_xi0_receiver.v1"
XI0_PAYLOAD_SCHEMA: Final = "r1b3_xi0_float16_payload.v1"
XI0_MAGIC: Final = b"XI01"
_XI0_PREFIX: Final = struct.Struct("<4sII")
_XI0_CRC: Final = struct.Struct("<I")

R1B2_EXTENSION_NAMES: Final = (
    "r1b2_manifest.json",
    "boundary_coordinate.bgj",
    "full_kernel_replay.r1k",
    "xi0.xi0",
)


class R1B3ProducerError(ValueError):
    """Malformed, drifted, or falsely promoted producer custody."""


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise R1B3ProducerError("payload is not canonical-JSON encodable") from exc


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1B3ProducerError(f"invalid JSON artifact: {resolved}") from exc
    if not isinstance(value, dict):
        raise R1B3ProducerError(f"JSON artifact is not an object: {resolved}")
    return value, {
        "path": str(resolved),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def audit_rank4_strata(
    stage_dir: Path,
    *,
    pair_count: int = PAIR_COUNT,
    batch_size: int = BATCH_SIZE,
    expected_moderate: int = MODERATE_MARGIN_FLIPS,
    expected_tie: int = TIE_TIGHT_FLIPS,
) -> dict[str, Any]:
    """Re-derive the exact batch-16 winner/rival debt strata from hard-oracle rows."""

    resolved = stage_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise R1B3ProducerError("rank4 stratum stage path is not a directory")
    expected_starts = list(range(0, pair_count, batch_size))
    paths = [resolved / f"batch-{start:04d}.json" for start in expected_starts]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise R1B3ProducerError(f"rank4 stratum batch custody is incomplete: {missing[:3]}")

    moderate = tie = other = total = 0
    per_pair = [
        {
            "pair_index": pair,
            "moderate_margin_1e_3_to_1": 0,
            "tie_tight_lt_1e_3": 0,
            "other": 0,
        }
        for pair in range(pair_count)
    ]
    custody: list[dict[str, Any]] = []
    for path, start in zip(paths, expected_starts, strict=True):
        row, row_custody = _read_json(path)
        stop = min(pair_count, start + batch_size)
        flips = row.get("flips")
        if (
            row.get("schema") != "r2b_hard_oracle_batch.v1"
            or row.get("pair_start") != start
            or row.get("pair_stop") != stop
            or not isinstance(flips, list)
            or row.get("flip_count") != len(flips)
        ):
            raise R1B3ProducerError(f"rank4 hard-oracle batch row drift: {path}")
        for flip in flips:
            if not isinstance(flip, list) or len(flip) != 6:
                raise R1B3ProducerError("rank4 winner/rival flip row is malformed")
            pair, pixel_row, pixel_col, winner, rival, margin = flip
            if (
                isinstance(pair, bool)
                or not isinstance(pair, int)
                or not start <= pair < stop
                or isinstance(pixel_row, bool)
                or not isinstance(pixel_row, int)
                or not 0 <= pixel_row < SEGMENTATION_HEIGHT
                or isinstance(pixel_col, bool)
                or not isinstance(pixel_col, int)
                or not 0 <= pixel_col < SEGMENTATION_WIDTH
                or isinstance(winner, bool)
                or not isinstance(winner, int)
                or not 0 <= winner < SEGMENTATION_CLASS_COUNT
                or isinstance(rival, bool)
                or not isinstance(rival, int)
                or not 0 <= rival < SEGMENTATION_CLASS_COUNT
                or winner == rival
                or isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or float(margin) < 0.0
            ):
                raise R1B3ProducerError("rank4 winner/rival value custody is malformed")
            absolute_margin = abs(float(margin))
            if absolute_margin < 1e-3:
                tie += 1
                per_pair[pair]["tie_tight_lt_1e_3"] += 1
            elif absolute_margin < 1.0:
                moderate += 1
                per_pair[pair]["moderate_margin_1e_3_to_1"] += 1
            else:
                other += 1
                per_pair[pair]["other"] += 1
            total += 1
        custody.append(row_custody)

    expected_total = expected_moderate + expected_tie
    if (moderate, tie, other, total) != (
        expected_moderate,
        expected_tie,
        0,
        expected_total,
    ):
        raise R1B3ProducerError(
            f"rank4 stratum totals drifted: moderate={moderate}, tie={tie}, other={other}, total={total}"
        )
    return {
        "schema": "r1b3_rank4_strata_audit.v1",
        "pair_count": pair_count,
        "batch_size": batch_size,
        "hard_oracle_batch_count": len(paths),
        "moderate_margin_1e_3_to_1": moderate,
        "tie_tight_lt_1e_3": tie,
        "other": other,
        "total": total,
        "per_pair": per_pair,
        "batch_custody": custody,
        "score_claim": False,
    }


def audit_head_rank4(segnet_weights: Path) -> dict[str, Any]:
    """Re-derive the exact centered-head rank from the frozen safetensors bytes."""

    resolved = segnet_weights.expanduser().resolve(strict=True)
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment gate
        raise R1B3ProducerError("safetensors is required for frozen-head custody") from exc
    with safe_open(str(resolved), framework="numpy") as handle:
        weight = np.asarray(handle.get_tensor("segmentation_head.0.weight"), dtype=np.float32)
    if weight.shape[0] != 5:
        raise R1B3ProducerError(f"SegNet head class geometry drifted: {weight.shape}")
    flattened = weight.reshape(5, -1).astype(np.float64)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-6))
    if rank != 4:
        raise R1B3ProducerError(f"frozen SegNet centered-head rank drifted: {rank}")
    rank4 = (u[:, :4] * singular_values[:4]) @ vh[:4]
    pair_norms = {
        f"{left}-{right}": float(np.linalg.norm(flattened[left] - flattened[right]))
        for left in range(5)
        for right in range(left + 1, 5)
    }
    return {
        "schema": "r1b3_frozen_segnet_head_chart.v1",
        "weights": {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        },
        "weight_shape": list(weight.shape),
        "patch_dimension": int(flattened.shape[1]),
        "centered_rank": rank,
        "singular_values": singular_values.tolist(),
        "rank4_reconstruction_max_abs_error_float64": float(np.max(np.abs(centered - rank4))),
        "pair_normal_l2": pair_norms,
        "exact_domain": "penultimate_feature_patch",
        "receiver_coordinate_pullback_status": "ABSENT",
        "score_claim": False,
    }


def _load_gt_poses(cache_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = cache_path.expanduser().resolve(strict=True)
    try:
        with zipfile.ZipFile(resolved, "r") as archive, archive.open("gt_poses.npy", "r") as handle:
            poses = np.load(handle, allow_pickle=False)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise R1B3ProducerError("GT cache lacks a valid gt_poses.npy member") from exc
    array = np.asarray(poses)
    if array.shape != (PAIR_COUNT, 6) or not np.issubdtype(array.dtype, np.floating):
        raise R1B3ProducerError(f"GT pose geometry drifted: {array.shape} {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise R1B3ProducerError("GT pose targets contain non-finite values")
    return array.astype(np.float32), {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "member": "gt_poses.npy",
    }


def encode_xi0_payload(values: np.ndarray) -> bytes:
    """Encode exactly 600 PoseNet coordinate-zero targets as canonical float16 LE."""

    raw = np.asarray(values)
    if raw.shape != (PAIR_COUNT,) or not np.issubdtype(raw.dtype, np.floating):
        raise R1B3ProducerError("xi0 values must be exactly 600 floating-point scalars")
    if not np.all(np.isfinite(raw)):
        raise R1B3ProducerError("xi0 values must be finite")
    quantized = raw.astype("<f2")
    if not np.all(np.isfinite(quantized)):
        raise R1B3ProducerError("xi0 targets do not remain finite in float16")
    body = quantized.tobytes(order="C")
    header = {
        "schema": XI0_PAYLOAD_SCHEMA,
        "pair_count": PAIR_COUNT,
        "coordinate_indices": [0],
        "target_semantics": "source_posenet_output_coordinate0",
        "quantization": "float16_le",
        "body_bytes": len(body),
        "body_sha256": _sha256_bytes(body),
        "score_claim": False,
    }
    header_bytes = _canonical_json(header)
    prefix = _XI0_PREFIX.pack(XI0_MAGIC, len(header_bytes), len(body))
    checksum = _XI0_CRC.pack(zlib.crc32(header_bytes + body) & 0xFFFFFFFF)
    return prefix + header_bytes + body + checksum


def decode_xi0_payload(payload: bytes) -> np.ndarray:
    """Strictly parse the canonical coordinate-zero payload."""

    if not isinstance(payload, bytes) or len(payload) < _XI0_PREFIX.size + _XI0_CRC.size:
        raise R1B3ProducerError("xi0 payload is truncated")
    magic, header_size, body_size = _XI0_PREFIX.unpack_from(payload)
    if magic != XI0_MAGIC:
        raise R1B3ProducerError("xi0 payload magic mismatch")
    expected = _XI0_PREFIX.size + header_size + body_size + _XI0_CRC.size
    if len(payload) != expected:
        raise R1B3ProducerError("xi0 payload length mismatch or trailing bytes")
    header_start = _XI0_PREFIX.size
    body_start = header_start + header_size
    body_end = body_start + body_size
    header_bytes = payload[header_start:body_start]
    body = payload[body_start:body_end]
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1B3ProducerError("xi0 header is not ASCII JSON") from exc
    if not isinstance(header, dict) or _canonical_json(header) != header_bytes:
        raise R1B3ProducerError("xi0 header is not canonical")
    if header != {
        "schema": XI0_PAYLOAD_SCHEMA,
        "pair_count": PAIR_COUNT,
        "coordinate_indices": [0],
        "target_semantics": "source_posenet_output_coordinate0",
        "quantization": "float16_le",
        "body_bytes": PAIR_COUNT * 2,
        "body_sha256": _sha256_bytes(body),
        "score_claim": False,
    }:
        raise R1B3ProducerError("xi0 sealed header values mismatch")
    (stored_crc,) = _XI0_CRC.unpack(payload[body_end:])
    if stored_crc != (zlib.crc32(header_bytes + body) & 0xFFFFFFFF):
        raise R1B3ProducerError("xi0 payload CRC mismatch")
    values = np.frombuffer(body, dtype="<f2").copy()
    if values.shape != (PAIR_COUNT,) or encode_xi0_payload(values) != payload:
        raise R1B3ProducerError("xi0 payload is noncanonical")
    return values


def build_xi0_bundle(cache_path: Path, *, payload_path: Path | None = None) -> dict[str, Any]:
    """Build the real coordinate-zero payload and an R1b2-auditable manifest object."""

    poses, cache_custody = _load_gt_poses(cache_path)
    source = poses[:, 0]
    payload = encode_xi0_payload(source)
    decoded = decode_xi0_payload(payload).astype(np.float32)
    quantization_error = decoded - source
    payload_custody = {
        "path": None if payload_path is None else str(payload_path.expanduser().resolve()),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }
    manifest = {
        "schema": XI0_SCHEMA,
        "pair_count": PAIR_COUNT,
        "score_claim": False,
        "coordinate_indices": [0],
        "receiver_schema": XI0_RECEIVER_SCHEMA,
        "other_coordinates_counted": 0,
        "quantization": "float16_le",
        "payload_path": payload_custody["path"],
        "payload_sha256": payload_custody["sha256"],
    }
    return {
        "schema": "r1b3_xi0_bundle.v1",
        "manifest": manifest,
        "payload": payload_custody,
        "payload_bytes": payload,
        "source_cache": cache_custody,
        "source_coordinate": 0,
        "source_min": float(np.min(source)),
        "source_max": float(np.max(source)),
        "quantization_mse": float(np.mean(quantization_error.astype(np.float64) ** 2)),
        "quantization_max_abs": float(np.max(np.abs(quantization_error))),
        "receiver_actuation_status": "ABSENT",
        "score_claim": False,
    }


def audit_full_kernel_inputs(full_kernel_receipt: Path, r2b_receipt: Path) -> dict[str, Any]:
    """Classify existing exact/coder evidence without promoting it to n600 replay."""

    full, full_custody = _read_json(full_kernel_receipt)
    r2b, r2b_custody = _read_json(r2b_receipt)
    if full.get("schema") != "resize_null_preimage_full_kernel_measurement.v1":
        raise R1B3ProducerError("full-kernel receipt schema mismatch")
    if r2b.get("schema") != "r2b_sparse_target_selection_receipt.v1":
        raise R1B3ProducerError("R2b receipt schema mismatch")
    stream = r2b.get("stream")
    if not isinstance(stream, dict):
        raise R1B3ProducerError("R2b compact stream custody is absent")
    stream_path = Path(str(stream.get("path", ""))).expanduser().resolve(strict=True)
    if stream_path.stat().st_size != stream.get("bytes") or sha256_file(stream_path) != stream.get("sha256"):
        raise R1B3ProducerError("R2b compact stream custody drifted")

    selected_frames = full.get("minimum_description", {}).get("selected_full_kernel_frames")
    frame_rows = full.get("frame_rows")
    measured_frames = len(frame_rows) if isinstance(frame_rows, list) else 0
    blockers: list[str] = []
    if measured_frames != PAIR_COUNT * 2 or selected_frames != PAIR_COUNT * 2:
        blockers.append("R1B3_P2_N600_FULL_KERNEL_MDL_SELECTION_ABSENT")
    if r2b.get("kkt_stop_decisions") != 0:
        raise R1B3ProducerError("R2b KKT admission anchor drifted")
    blockers.append("R1B3_P2_CODER_ADMITTED_NONEMPTY_REPLAY_ABSENT")
    blockers.append("R1B3_P2_ZERO_SEARCH_RECEIVER_REPLAY_PROOF_ABSENT")
    baseline = r2b.get("baseline")
    candidate = r2b.get("candidate")
    curve = r2b.get("curve")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict) or not isinstance(curve, list) or not curve:
        raise R1B3ProducerError("R2b baseline/candidate/curve custody is malformed")
    hard_fixed_flips = baseline.get("flip_count", 0) - candidate.get("flip_count", 0)
    evaluated = r2b.get("candidate_evaluation_decisions")
    scheduled_upper_bound = curve[-1].get("scheduled_recovered_seg_score_upper_bound")
    recovered_score = r2b.get("candidate_recovered_score")
    if (
        isinstance(hard_fixed_flips, bool)
        or not isinstance(hard_fixed_flips, int)
        or hard_fixed_flips < 0
        or isinstance(evaluated, bool)
        or not isinstance(evaluated, int)
        or evaluated <= 0
        or isinstance(scheduled_upper_bound, bool)
        or not isinstance(scheduled_upper_bound, (int, float))
        or not math.isfinite(float(scheduled_upper_bound))
        or float(scheduled_upper_bound) <= 0.0
        or isinstance(recovered_score, bool)
        or not isinstance(recovered_score, (int, float))
        or not math.isfinite(float(recovered_score))
    ):
        raise R1B3ProducerError("R2b realization anchors are malformed")
    return {
        "schema": "r1b3_full_kernel_producer_preflight.v1",
        "full_kernel_receipt": full_custody,
        "r2b_receipt": r2b_custody,
        "measured_full_kernel_frames": measured_frames,
        "selected_full_kernel_frames": selected_frames,
        "r2b_kkt_admitted_decisions": r2b.get("kkt_stop_decisions"),
        "r2b_hard_evaluated_decisions": evaluated,
        "r2b_hard_fixed_flips": hard_fixed_flips,
        "r2b_decision_realization_fraction": hard_fixed_flips / evaluated,
        "r2b_score_realization_fraction": float(recovered_score) / float(scheduled_upper_bound),
        "r2b_candidate_recovered_score": float(recovered_score),
        "r2b_scheduled_recovered_seg_score_upper_bound": float(scheduled_upper_bound),
        "r2b_compact_stream": {
            "path": str(stream_path),
            "bytes": stream_path.stat().st_size,
            "sha256": sha256_file(stream_path),
        },
        "blockers": blockers,
        "verdict_scope": (
            "one full-kernel fixture frame and one R2b fixed-magnitude sparse-decision formulation; "
            "no full-kernel or boundary-carrier family negative"
        ),
        "score_claim": False,
    }


def audit_production_receiver_binding(
    base_decoder: Path,
    production_parser_source: Path,
) -> dict[str, Any]:
    """Require literal consumer custody for every appended R1b2 member."""

    decoder = base_decoder.expanduser().resolve(strict=True)
    parser_source = production_parser_source.expanduser().resolve(strict=True)
    decoder_text = decoder.read_text(encoding="utf-8")
    parser_text = parser_source.read_text(encoding="utf-8")
    decoder_members = {name: name in decoder_text for name in R1B2_EXTENSION_NAMES}
    parser_members = {name: name in parser_text for name in R1B2_EXTENSION_NAMES}
    blockers: list[str] = []
    if not all(decoder_members.values()):
        blockers.append("R1B3_APPENDED_SECTIONS_ABSENT_FROM_PRODUCTION_DECODER")
    if "r1b2_counted_archive.v1" not in parser_text or not all(parser_members.values()):
        blockers.append("R1B3_APPENDED_SECTIONS_REFUSED_BY_PRODUCTION_C2_PARSER")
    literal_binding_present = (
        all(decoder_members.values()) and "r1b2_counted_archive.v1" in parser_text and all(parser_members.values())
    )
    if literal_binding_present:
        blockers.append("R1B3_PRODUCTION_RECEIVER_BEHAVIORAL_PROOF_ABSENT")
    return {
        "schema": "r1b3_production_receiver_binding_audit.v1",
        "base_decoder": {
            "path": str(decoder),
            "bytes": decoder.stat().st_size,
            "sha256": sha256_file(decoder),
            "extension_member_literals": decoder_members,
        },
        "production_parser_source": {
            "path": str(parser_source),
            "bytes": parser_source.stat().st_size,
            "sha256": sha256_file(parser_source),
            "r1b2_schema_literal": "r1b2_counted_archive.v1" in parser_text,
            "extension_member_literals": parser_members,
        },
        "literal_binding_present": literal_binding_present,
        "receiver_bound": False,
        "blockers": blockers,
        "verdict_scope": "current hash-pinned C2 production decoder/parser pair only",
        "score_claim": False,
    }


def build_producer_preflight_receipt(
    *,
    stage_dir: Path,
    segnet_weights: Path,
    full_kernel_receipt: Path,
    r2b_receipt: Path,
    gt_cache: Path,
    base_decoder: Path,
    production_parser_source: Path,
) -> dict[str, Any]:
    """Compose the three producer audits and their exact blocking seam."""

    p1_strata = audit_rank4_strata(stage_dir)
    p1_head = audit_head_rank4(segnet_weights)
    p2 = audit_full_kernel_inputs(full_kernel_receipt, r2b_receipt)
    p3 = build_xi0_bundle(gt_cache)
    receiver = audit_production_receiver_binding(base_decoder, production_parser_source)
    blockers = [
        "R1B3_P1_RECEIVER_COORDINATE_JACOBIAN_AND_REALIZED_SECANT_ABSENT",
        *p2["blockers"],
        "R1B3_P3_XI0_TARGET_TO_DECODED_FRAME_ACTUATOR_ABSENT",
        *receiver["blockers"],
    ]
    return {
        "schema": "r1b3_producer_preflight_receipt.v1",
        "verdict": "PRODUCER_INPUTS_MEASURED_COMPILER_BUNDLES_BLOCKED",
        "verdict_scope": (
            "current R1b2 production control and existing P1/P2/P3 custody only; "
            "no rank4, full-kernel, xi, or boundary-carrier family negative"
        ),
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_mutation": False,
        },
        "p1": {
            "strata": p1_strata,
            "head_chart": p1_head,
            "compiler_manifest_emitted": False,
            "blocker": "R1B3_P1_RECEIVER_COORDINATE_JACOBIAN_AND_REALIZED_SECANT_ABSENT",
        },
        "p2": {
            **p2,
            "compiler_manifest_emitted": False,
        },
        "p3": {key: value for key, value in p3.items() if key != "payload_bytes"},
        "production_receiver": receiver,
        "blockers": list(dict.fromkeys(blockers)),
        "pointer": "0.19108 [contest-CPU] UNMOVED",
    }


__all__ = [
    "R1B3ProducerError",
    "audit_full_kernel_inputs",
    "audit_head_rank4",
    "audit_production_receiver_binding",
    "audit_rank4_strata",
    "build_producer_preflight_receipt",
    "build_xi0_bundle",
    "decode_xi0_payload",
    "encode_xi0_payload",
    "sha256_file",
]
