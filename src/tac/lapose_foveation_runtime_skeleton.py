# SPDX-License-Identifier: MIT
"""Fail-closed runtime skeleton for LA-POSE foveation tuple payloads.

This module is intentionally not a contest decoder. The local LFV1 archive
builder packages it as a charged archive member so archive custody can prove
that the runtime has only archive-contained LFV1 bytes available while runtime
output parity, no-op controls, and exact CUDA auth eval remain blockers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PAYLOAD_MEMBER = "lapose_foveation_tuples.lfv1"
FOVEATION_PARAMS_MEMBER = "foveation_params.bin"
PROOF_MEMBER = "runtime_consumer_proof_skeleton.json"
REQUIRED_MEMBERS = (PAYLOAD_MEMBER, FOVEATION_PARAMS_MEMBER, PROOF_MEMBER)
PAYLOAD_MAGIC = b"LFV1"
FOVEATION_MAGIC = b"HFV1"
HEADER_STRUCT = struct.Struct("<4sHHHH")
FOVEATION_HEADER_STRUCT = struct.Struct("<4sIII")
ROW_STRUCT = struct.Struct("<BHHHHHH")
FOVEATION_ROW_STRUCT = struct.Struct("<fffff")
RUNTIME_PROOF_SKELETON_CONTRACT = "lapose_foveation_runtime_consumer_proof_skeleton_v1"
RUNTIME_EFFECT_CONTROLS_CONTRACT = "lapose_foveation_runtime_effect_controls_v1"
RUNTIME_STRUCTURAL_OUTPUT_CONTRACT = "lapose_foveation_runtime_structural_output_v1"
RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT = "lapose_foveation_scorer_visible_bridge_v1"
LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT = "lapose_lfv1_to_foveation_params_bridge_v1"
UINT16_MAX = 65_535
SCORER_VISIBLE_MEMBER_GROUPS = {
    "mask_or_segmentation_stream": (
        "masks.mkv",
        "grayscale.mkv",
        "masks.alpha4.mkv",
        "masks.amrc",
        "masks.nrv",
        "masks.cdo1",
        "masks.cdo1.xz",
        "masks.cdo1.zlib",
        "masks.cdo1.br",
    ),
    "renderer_or_segmap_runtime": (
        "renderer.bin",
        "renderer_payload.bin",
        "renderer_payload.bin.br",
        "p",
        "segmap_weights.tar.xz",
        "payload.bin",
    ),
    "pose_or_geometry_stream": (
        "optimized_poses.pt",
        "optimized_poses.bin",
        "optimized_poses.qp1",
        "zoom_scalars.bin",
        "zoom_scalars.pt",
        FOVEATION_PARAMS_MEMBER,
    ),
}
PAIR_TO_FRAME_POLICY = "contest_pair_maps_to_frames_2k_and_2k_plus_1"


class RuntimeSkeletonError(RuntimeError):
    """Raised when the charged LFV1 runtime skeleton contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return _sha256_bytes(raw)


def _dequantize_u16(value: int, low: float, high: float) -> float:
    return float(low) + (float(high) - float(low)) * (float(value) / float(UINT16_MAX))


def _default_foveation_row(frame_width: int, frame_height: int) -> tuple[float, float, float, float, float]:
    return (
        0.0,
        max(math.hypot(float(frame_width), float(frame_height)), 1.0),
        1.0,
        float(max(frame_width - 1, 0)) / 2.0,
        float(max(frame_height - 1, 0)) / 2.0,
    )


def _pair_frame_indices(pair_index: int) -> tuple[int, int]:
    """Map one scorer pair index to its two rendered frame indices."""

    if pair_index < 0:
        raise RuntimeSkeletonError("pair_index must be non-negative")
    return 2 * int(pair_index), 2 * int(pair_index) + 1


def lower_lfv1_to_foveation_params(decoded: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Lower decoded LFV1 tuples into a deterministic HFV1 geometry payload.

    This produces an archive-contained scorer-visible geometry member. It does
    not prove that a contest runtime applies the member to output frames.
    """

    frame_width = int(decoded["frame_width"])
    frame_height = int(decoded["frame_height"])
    rows = decoded["rows"]
    n_frames = max(_pair_frame_indices(int(row["pair_index"]))[1] for row in rows) + 1 if rows else 1
    n_frames = max(n_frames, 1)
    radius_high = max(math.hypot(float(frame_width), float(frame_height)), 1.0)
    params = [_default_foveation_row(frame_width, frame_height) for _ in range(n_frames)]
    applied_rows: list[dict[str, Any]] = []
    for row in rows:
        pair_index = int(row["pair_index"])
        target_frame_indices = _pair_frame_indices(pair_index)
        quantized = row["quantized"]
        values = (
            _dequantize_u16(int(quantized["alpha"]), 0.0, 8.0),
            max(_dequantize_u16(int(quantized["radius"]), 0.0, radius_high), 1e-6),
            _dequantize_u16(int(quantized["power"]), 0.0, 8.0),
            _dequantize_u16(int(quantized["origin_x"]), 0.0, float(max(frame_width - 1, 0))),
            _dequantize_u16(int(quantized["origin_y"]), 0.0, float(max(frame_height - 1, 0))),
        )
        for frame_index in target_frame_indices:
            params[frame_index] = values
        applied_rows.append(
            {
                "row_index": int(row["row_index"]),
                "pair_index": pair_index,
                "opcode": int(row["opcode"]),
                "target_member": FOVEATION_PARAMS_MEMBER,
                "target_frame_indices": list(target_frame_indices),
            }
        )
    body = b"".join(FOVEATION_ROW_STRUCT.pack(*values) for values in params)
    raw = FOVEATION_HEADER_STRUCT.pack(
        FOVEATION_MAGIC,
        int(n_frames),
        int(frame_height),
        int(frame_width),
    ) + body
    report = {
        "schema_version": 1,
        "contract": LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_member": PAYLOAD_MEMBER,
        "target_member": FOVEATION_PARAMS_MEMBER,
        "target_wire_format": FOVEATION_MAGIC.decode("ascii"),
        "source_row_count": int(decoded["row_count"]),
        "target_frame_count": int(n_frames),
        "pair_to_frame_policy": PAIR_TO_FRAME_POLICY,
        "image_size": {"height": frame_height, "width": frame_width},
        "duplicate_pair_policy": "last_lfv1_tuple_for_pair_index_wins",
        "applied_rows": applied_rows,
        "output_bytes": len(raw),
        "output_sha256": _sha256_bytes(raw),
        "runtime_output_parity_proven": False,
        "exact_cuda_auth_eval_proven": False,
        "blockers": [
            "lapose_foveation_runtime_output_parity_not_proven",
            "exact_cuda_auth_eval_missing",
        ],
    }
    return raw, report


def build_lfv1_foveation_params_bridge_report(
    payload: bytes,
    foveation_params: bytes,
) -> dict[str, Any]:
    decoded = _decode_lfv1(payload)
    expected_raw, report = lower_lfv1_to_foveation_params(decoded)
    report.update(
        {
            "source_payload_sha256": _sha256_bytes(payload),
            "target_member_sha256": _sha256_bytes(foveation_params),
            "target_member_bytes": len(foveation_params),
            "derived_bytes_match": foveation_params == expected_raw,
            "passed": foveation_params == expected_raw,
        }
    )
    if foveation_params != expected_raw:
        report["blockers"] = [
            *report["blockers"],
            "lapose_foveation_params_member_not_derived_from_lfv1",
        ]
    return report


def _member_group_report(member_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    present = set(member_names)
    return {
        group: {
            "required_any_of": list(candidates),
            "present_members": [name for name in candidates if name in present],
            "present": any(name in present for name in candidates),
        }
        for group, candidates in SCORER_VISIBLE_MEMBER_GROUPS.items()
    }


def _runtime_symbol_names(runtime_consumer_source: str) -> set[str]:
    try:
        tree = ast.parse(runtime_consumer_source)
    except SyntaxError:
        return set()
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            symbols.add(node.name)
    return symbols


def _runtime_source_references(runtime_consumer_source: str) -> dict[str, bool]:
    symbols = _runtime_symbol_names(runtime_consumer_source)
    return {
        "references_lfv1_member": PAYLOAD_MEMBER in runtime_consumer_source,
        "references_lfv1_decoder": "_decode_lfv1" in symbols,
        "references_mask_stream_member": bool(
            symbols
            & {
                "apply_lfv1_to_mask_output",
                "apply_lfv1_to_segmentation_stream",
                "write_lfv1_masks",
            }
        ),
        "references_renderer_runtime_member": bool(
            symbols
            & {
                "apply_lfv1_to_renderer_output",
                "apply_lfv1_to_rgb_frames",
                "write_lfv1_rendered_frames",
            }
        ),
        "references_pose_or_geometry_member": bool(
            symbols
            & {
                "apply_lfv1_to_pose_output",
                "apply_lfv1_to_zoom_scalars",
                "write_lfv1_pose_geometry",
            }
        ),
        "references_foveation_loader": bool(
            symbols
            & {
                "apply_lfv1_to_foveation_params",
                "build_lfv1_foveation_params_bridge_report",
                "lower_lfv1_to_foveation_params",
                "load_foveation_params",
                "functional_hyperbolic_foveation",
            }
        ),
        "references_frame_write_path": "write_scorer_visible_frames" in symbols,
    }


def build_scorer_visible_bridge_report(
    archive_member_names: Sequence[str],
    *,
    runtime_consumer_source: str = "",
) -> dict[str, Any]:
    """Report whether LFV1 has a deterministic path to scorer-visible outputs."""

    names = sorted(str(name) for name in archive_member_names if isinstance(name, str))
    groups = _member_group_report(names)
    source_refs = _runtime_source_references(runtime_consumer_source)
    has_lfv1_payload = PAYLOAD_MEMBER in names
    has_scorer_visible_member_path = (
        groups["mask_or_segmentation_stream"]["present"]
        or groups["renderer_or_segmap_runtime"]["present"]
        or groups["pose_or_geometry_stream"]["present"]
    )
    runtime_references_scorer_visible_path = any(
        source_refs[key]
        for key in (
            "references_mask_stream_member",
            "references_renderer_runtime_member",
            "references_pose_or_geometry_member",
            "references_foveation_loader",
            "references_frame_write_path",
        )
    )
    bridge_path_present = (
        has_lfv1_payload
        and has_scorer_visible_member_path
        and runtime_references_scorer_visible_path
    )
    blockers: list[str] = []
    if not has_lfv1_payload:
        blockers.append("lapose_foveation_lfv1_member_missing")
    if not has_scorer_visible_member_path:
        blockers.append("lapose_foveation_renderer_mask_or_geometry_output_path_missing")
    if not runtime_references_scorer_visible_path:
        blockers.append("lapose_foveation_runtime_does_not_reference_scorer_visible_output_path")
    blockers.append("lapose_foveation_scorer_visible_output_parity_not_proven")

    return {
        "schema_version": 1,
        "contract": RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "passed": False,
        "bridge_path_present": bridge_path_present,
        "archive_member_names": names,
        "archive_member_groups": groups,
        "runtime_source_references": source_refs,
        "required_bridge": {
            "charged_lfv1_tuple_member": PAYLOAD_MEMBER,
            "must_map_lfv1_to": [
                "renderer-conditioned RGB frame output",
                "mask or segmentation stream consumed by inflate",
                "pose, zoom, or foveation geometry consumed by inflate",
            ],
            "must_prove": [
                "LFV1 mutation changes scorer-visible frames or masks",
                "identity/no-op controls are byte-exact",
                "exact CUDA auth eval on exact archive bytes",
            ],
        },
        "blockers": blockers,
        "fail_closed_reason": (
            "LFV1 tuple bytes are lowered to archive-contained foveation geometry when present, "
            "but no packaged contest runtime parity proof shows that geometry changes scorer-visible frames."
        ),
    }


def _required_int(value: Any, *, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeSkeletonError(f"{label} must be an integer")
    if not low <= value <= high:
        raise RuntimeSkeletonError(f"{label} must be in [{low}, {high}]")
    return int(value)


def _decode_lfv1(payload: bytes) -> dict[str, Any]:
    raw = bytes(payload)
    if len(raw) < HEADER_STRUCT.size:
        raise RuntimeSkeletonError("LFV1 payload shorter than header")
    magic, version, row_count, frame_width, frame_height = HEADER_STRUCT.unpack(
        raw[: HEADER_STRUCT.size]
    )
    if magic != PAYLOAD_MAGIC:
        raise RuntimeSkeletonError(f"bad LFV1 magic: {magic!r}")
    expected_bytes = HEADER_STRUCT.size + int(row_count) * ROW_STRUCT.size
    if len(raw) != expected_bytes:
        raise RuntimeSkeletonError(
            f"bad LFV1 payload size: got {len(raw)} bytes, expected {expected_bytes}"
        )

    rows: list[dict[str, Any]] = []
    offset = HEADER_STRUCT.size
    for row_index in range(int(row_count)):
        opcode, pair_index, alpha_q, radius_q, power_q, origin_x_q, origin_y_q = (
            ROW_STRUCT.unpack(raw[offset : offset + ROW_STRUCT.size])
        )
        rows.append(
            {
                "row_index": row_index,
                "byte_offset": offset,
                "opcode": opcode,
                "pair_index": pair_index,
                "quantized": {
                    "alpha": alpha_q,
                    "radius": radius_q,
                    "power": power_q,
                    "origin_x": origin_x_q,
                    "origin_y": origin_y_q,
                },
            }
        )
        offset += ROW_STRUCT.size

    return {
        "magic": magic.decode("ascii"),
        "schema_version": int(version),
        "row_count": int(row_count),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "rows": rows,
    }


def _encode_lfv1(decoded: dict[str, Any]) -> bytes:
    if not isinstance(decoded, dict):
        raise RuntimeSkeletonError("decoded LFV1 payload must be an object")
    if decoded.get("magic") != PAYLOAD_MAGIC.decode("ascii"):
        raise RuntimeSkeletonError("decoded LFV1 magic mismatch")
    version = _required_int(
        decoded.get("schema_version"),
        label="decoded.schema_version",
        low=0,
        high=UINT16_MAX,
    )
    frame_width = _required_int(
        decoded.get("frame_width"),
        label="decoded.frame_width",
        low=0,
        high=UINT16_MAX,
    )
    frame_height = _required_int(
        decoded.get("frame_height"),
        label="decoded.frame_height",
        low=0,
        high=UINT16_MAX,
    )
    rows = decoded.get("rows")
    if not isinstance(rows, list):
        raise RuntimeSkeletonError("decoded LFV1 rows must be a list")
    declared_row_count = _required_int(
        decoded.get("row_count"),
        label="decoded.row_count",
        low=0,
        high=UINT16_MAX,
    )
    if declared_row_count != len(rows):
        raise RuntimeSkeletonError("decoded LFV1 row_count does not match rows length")

    body = bytearray()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeSkeletonError(f"decoded LFV1 row {index} must be an object")
        quantized = row.get("quantized")
        if not isinstance(quantized, dict):
            raise RuntimeSkeletonError(f"decoded LFV1 row {index} missing quantized values")
        body.extend(
            ROW_STRUCT.pack(
                _required_int(row.get("opcode"), label=f"rows[{index}].opcode", low=0, high=255),
                _required_int(
                    row.get("pair_index"),
                    label=f"rows[{index}].pair_index",
                    low=0,
                    high=UINT16_MAX,
                ),
                _required_int(
                    quantized.get("alpha"),
                    label=f"rows[{index}].quantized.alpha",
                    low=0,
                    high=UINT16_MAX,
                ),
                _required_int(
                    quantized.get("radius"),
                    label=f"rows[{index}].quantized.radius",
                    low=0,
                    high=UINT16_MAX,
                ),
                _required_int(
                    quantized.get("power"),
                    label=f"rows[{index}].quantized.power",
                    low=0,
                    high=UINT16_MAX,
                ),
                _required_int(
                    quantized.get("origin_x"),
                    label=f"rows[{index}].quantized.origin_x",
                    low=0,
                    high=UINT16_MAX,
                ),
                _required_int(
                    quantized.get("origin_y"),
                    label=f"rows[{index}].quantized.origin_y",
                    low=0,
                    high=UINT16_MAX,
                ),
            )
        )
    header = HEADER_STRUCT.pack(
        PAYLOAD_MAGIC,
        version,
        len(rows),
        frame_width,
        frame_height,
    )
    return header + bytes(body)


def _structural_output(decoded: dict[str, Any]) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    for row in decoded["rows"]:
        quantized = row["quantized"]
        pair_index = int(row["pair_index"])
        routes.append(
            {
                "pair_index": pair_index,
                "target_frame_indices": list(_pair_frame_indices(pair_index)),
                "opcode": int(row["opcode"]),
                "quantized_alpha": int(quantized["alpha"]),
                "quantized_radius": int(quantized["radius"]),
                "quantized_power": int(quantized["power"]),
                "quantized_origin_x": int(quantized["origin_x"]),
                "quantized_origin_y": int(quantized["origin_y"]),
            }
        )
    payload = {
        "contract": RUNTIME_STRUCTURAL_OUTPUT_CONTRACT,
        "frame_width": int(decoded["frame_width"]),
        "frame_height": int(decoded["frame_height"]),
        "row_count": int(decoded["row_count"]),
        "routes": routes,
    }
    return {
        "contract": RUNTIME_STRUCTURAL_OUTPUT_CONTRACT,
        "frame_width": payload["frame_width"],
        "frame_height": payload["frame_height"],
        "row_count": payload["row_count"],
        "route_count": len(routes),
        "first_route": routes[0] if routes else None,
        "last_route": routes[-1] if routes else None,
        "structural_output_sha256": _canonical_json_sha256(payload),
    }


def _mutate_first_tuple(decoded: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not decoded["rows"]:
        raise RuntimeSkeletonError("LFV1 mutation control requires at least one tuple row")
    mutated = json.loads(json.dumps(decoded, sort_keys=True))
    quantized = mutated["rows"][0]["quantized"]
    old_value = int(quantized["alpha"])
    new_value = old_value + 1 if old_value < UINT16_MAX else old_value - 1
    quantized["alpha"] = new_value
    return mutated, {
        "row_index": 0,
        "field": "quantized.alpha",
        "old_value": old_value,
        "new_value": new_value,
    }


def build_runtime_effect_control_report(payload: bytes) -> dict[str, Any]:
    """Prove LFV1 structural controls without claiming scored output parity."""

    raw = bytes(payload)
    decoded = _decode_lfv1(raw)
    reencoded = _encode_lfv1(decoded)
    structural_output = _structural_output(decoded)
    mutated_decoded, mutation = _mutate_first_tuple(decoded)
    mutated_payload = _encode_lfv1(mutated_decoded)
    mutated_roundtrip = _encode_lfv1(_decode_lfv1(mutated_payload))
    mutated_structural_output = _structural_output(_decode_lfv1(mutated_payload))
    identity_passed = reencoded == raw
    mutation_changed_output = (
        structural_output["structural_output_sha256"]
        != mutated_structural_output["structural_output_sha256"]
    )
    mutation_roundtrip_passed = mutated_roundtrip == mutated_payload
    structural_consumption_passed = (
        identity_passed and mutation_changed_output and mutation_roundtrip_passed
    )

    return {
        "schema_version": 1,
        "runtime_effect_controls_contract": RUNTIME_EFFECT_CONTROLS_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "passed": structural_consumption_passed,
        "payload_member": PAYLOAD_MEMBER,
        "payload_sha256": _sha256_bytes(raw),
        "payload_bytes": len(raw),
        "lfv1_identity_decode_control": {
            "passed": identity_passed,
            "decoded_row_count": decoded["row_count"],
            "source_payload_sha256": _sha256_bytes(raw),
            "reencoded_payload_sha256": _sha256_bytes(reencoded),
            "byte_exact": reencoded == raw,
        },
        "lfv1_tuple_mutation_runtime_output_control": {
            "passed": mutation_changed_output and mutation_roundtrip_passed,
            "mutation": mutation,
            "mutated_payload_sha256": _sha256_bytes(mutated_payload),
            "mutated_payload_bytes": len(mutated_payload),
            "mutated_identity_decode_passed": mutation_roundtrip_passed,
            "source_structural_output_sha256": structural_output[
                "structural_output_sha256"
            ],
            "mutated_structural_output_sha256": mutated_structural_output[
                "structural_output_sha256"
            ],
            "structural_output_changed": mutation_changed_output,
        },
        "runtime_consumes_foveation_tuple_control": {
            "passed": structural_consumption_passed,
            "structural_output_contract": RUNTIME_STRUCTURAL_OUTPUT_CONTRACT,
            "source_structural_output": structural_output,
            "mutated_structural_output": mutated_structural_output,
            "tuple_fields_in_structural_output": [
                "opcode",
                "pair_index",
                "quantized.alpha",
                "quantized.radius",
                "quantized.power",
                "quantized.origin_x",
                "quantized.origin_y",
            ],
        },
        "structural_runtime_consumption": {
            "passed": structural_consumption_passed,
            "meaning": "LFV1 tuple bytes deterministically affect the runtime skeleton structural output digest",
        },
        "scored_runtime_output_parity": {
            "passed": False,
            "meaning": "No scorer-visible frames or masks are reconstructed by this skeleton",
            "blocker": "scored_runtime_output_parity_not_proven",
        },
    }


def _read_proof(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeSkeletonError(f"runtime proof skeleton is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeSkeletonError("runtime proof skeleton must be a JSON object")
    if payload.get("runtime_consumer_proof_skeleton_contract") != RUNTIME_PROOF_SKELETON_CONTRACT:
        raise RuntimeSkeletonError("runtime proof skeleton contract mismatch")
    return payload


def verify_charged_members(archive_root: str | Path) -> dict[str, Any]:
    """Verify LFV1 charged members without loading uncharged sidecars."""

    root = Path(archive_root)
    missing: list[str] = []
    for name in REQUIRED_MEMBERS:
        if not (root / name).is_file():
            missing.append(name)
    if missing:
        raise RuntimeSkeletonError("missing charged runtime member(s): " + ", ".join(missing))

    payload_path = root / PAYLOAD_MEMBER
    foveation_path = root / FOVEATION_PARAMS_MEMBER
    proof_path = root / PROOF_MEMBER
    payload_raw = payload_path.read_bytes()
    foveation_raw = foveation_path.read_bytes()
    decoded = _decode_lfv1(payload_raw)
    runtime_effect_controls = build_runtime_effect_control_report(payload_raw)
    lfv1_foveation_params_bridge = build_lfv1_foveation_params_bridge_report(
        payload_raw,
        foveation_raw,
    )
    proof = _read_proof(proof_path)

    charged_sha = proof.get("charged_member_sha256")
    if not isinstance(charged_sha, dict):
        raise RuntimeSkeletonError("runtime proof missing charged_member_sha256")
    charged_bytes = proof.get("charged_member_bytes")
    if not isinstance(charged_bytes, dict):
        raise RuntimeSkeletonError("runtime proof missing charged_member_bytes")

    payload_sha = _sha256_bytes(payload_raw)
    if charged_sha.get(PAYLOAD_MEMBER) != payload_sha:
        raise RuntimeSkeletonError("LFV1 payload SHA-256 does not match runtime proof")
    if charged_bytes.get(PAYLOAD_MEMBER) != len(payload_raw):
        raise RuntimeSkeletonError("LFV1 payload byte count does not match runtime proof")
    foveation_sha = _sha256_bytes(foveation_raw)
    if charged_sha.get(FOVEATION_PARAMS_MEMBER) != foveation_sha:
        raise RuntimeSkeletonError("foveation_params.bin SHA-256 does not match runtime proof")
    if charged_bytes.get(FOVEATION_PARAMS_MEMBER) != len(foveation_raw):
        raise RuntimeSkeletonError("foveation_params.bin byte count does not match runtime proof")
    if not lfv1_foveation_params_bridge["passed"]:
        raise RuntimeSkeletonError("foveation_params.bin is not derived from LFV1 payload")

    records = [
        {
            "name": PAYLOAD_MEMBER,
            "bytes": len(payload_raw),
            "sha256": payload_sha,
        },
        {
            "name": PROOF_MEMBER,
            "bytes": proof_path.stat().st_size,
            "sha256": _sha256_file(proof_path),
        },
        {
            "name": FOVEATION_PARAMS_MEMBER,
            "bytes": len(foveation_raw),
            "sha256": foveation_sha,
        },
    ]

    runtime_path = Path(__file__).resolve()
    if runtime_path.is_file():
        records.append(
            {
                "name": "runtime_consumer.py",
                "bytes": runtime_path.stat().st_size,
                "sha256": _sha256_file(runtime_path),
            }
        )

    try:
        runtime_source = runtime_path.read_text(encoding="utf-8")
    except OSError:
        runtime_source = ""
    scorer_visible_bridge = build_scorer_visible_bridge_report(
        [path.name for path in root.iterdir() if path.is_file()],
        runtime_consumer_source=runtime_source,
    )

    return {
        "schema_version": 1,
        "kind": "lapose_foveation_runtime_skeleton_member_check",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "charged_members_verified": records,
        "lfv1_payload_decode": decoded,
        "lfv1_foveation_params_bridge": lfv1_foveation_params_bridge,
        "runtime_effect_controls": runtime_effect_controls,
        "scorer_visible_bridge": scorer_visible_bridge,
        "structural_runtime_consumption_proven": runtime_effect_controls[
            "structural_runtime_consumption"
        ]["passed"],
        "runtime_output_parity_proven": False,
        "scored_runtime_output_parity_proven": False,
        "noop_controls_proven": runtime_effect_controls["passed"],
        "lfv1_to_foveation_params_bridge_proven": lfv1_foveation_params_bridge["passed"],
        "exact_cuda_auth_eval_proven": False,
        "dispatch_blockers": [
            "lapose_foveation_runtime_skeleton_not_a_decoder",
            "lapose_foveation_scored_runtime_output_parity_missing",
            *scorer_visible_bridge["blockers"],
            "exact_cuda_auth_eval_missing",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        payload = verify_charged_members(args.archive_root)
    except RuntimeSkeletonError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
