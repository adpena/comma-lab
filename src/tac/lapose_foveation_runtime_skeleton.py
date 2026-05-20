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
RUNTIME_SCORER_VISIBLE_FRAME_WARP_CONTROL_CONTRACT = (
    "lapose_foveation_scorer_visible_frame_warp_control_v1"
)
RUNTIME_SCORER_VISIBLE_BYTE_OUTPUT_CONTROL_CONTRACT = (
    "lapose_foveation_scorer_visible_byte_output_control_v1"
)
LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT = "lapose_lfv1_to_foveation_params_bridge_v1"
UINT16_MAX = 65_535
FRAME_WARP_CONTROL_MAX_PROBE_FRAMES = 4
RGB24_BYTES_FORMAT = "nchw_float_rgb_to_nhwc_uint8_rgb24"
DEFAULT_OFFICIAL_FACADE_CHUNK_FRAMES = 16
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


def _decode_hfv1_foveation_params(raw: bytes) -> dict[str, Any]:
    payload = bytes(raw)
    if len(payload) < FOVEATION_HEADER_STRUCT.size:
        raise RuntimeSkeletonError("HFV1 foveation params shorter than header")
    magic, n_frames, frame_height, frame_width = FOVEATION_HEADER_STRUCT.unpack(
        payload[: FOVEATION_HEADER_STRUCT.size]
    )
    if magic != FOVEATION_MAGIC:
        raise RuntimeSkeletonError(f"bad HFV1 magic: {magic!r}")
    expected_bytes = FOVEATION_HEADER_STRUCT.size + int(n_frames) * FOVEATION_ROW_STRUCT.size
    if len(payload) != expected_bytes:
        raise RuntimeSkeletonError(
            f"bad HFV1 payload size: got {len(payload)} bytes, expected {expected_bytes}"
        )
    if int(n_frames) <= 0:
        raise RuntimeSkeletonError("HFV1 n_frames must be positive")
    if int(frame_height) <= 0 or int(frame_width) <= 0:
        raise RuntimeSkeletonError("HFV1 frame size must be positive")

    rows: list[dict[str, float | int]] = []
    offset = FOVEATION_HEADER_STRUCT.size
    for frame_index in range(int(n_frames)):
        alpha, radius, power, origin_x, origin_y = FOVEATION_ROW_STRUCT.unpack(
            payload[offset : offset + FOVEATION_ROW_STRUCT.size]
        )
        values = (alpha, radius, power, origin_x, origin_y)
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeSkeletonError(f"HFV1 row {frame_index} contains non-finite values")
        rows.append(
            {
                "frame_index": frame_index,
                "alpha": float(alpha),
                "radius": float(radius),
                "power": float(power),
                "origin_x": float(origin_x),
                "origin_y": float(origin_y),
            }
        )
        offset += FOVEATION_ROW_STRUCT.size
    return {
        "magic": FOVEATION_MAGIC.decode("ascii"),
        "n_frames": int(n_frames),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "rows": rows,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
    }


def load_foveation_params(source: bytes | str | Path) -> dict[str, Any]:
    """Load archive-contained HFV1 foveation parameters without scorer access."""

    raw = Path(source).read_bytes() if isinstance(source, str | Path) else bytes(source)
    return _decode_hfv1_foveation_params(raw)


def _import_torch_for_frame_warp() -> Any:
    try:
        import torch
        import torch.nn.functional as torch_functional
    except Exception as exc:  # pragma: no cover - exercised only on runtimes missing torch
        raise RuntimeSkeletonError(f"torch frame-warp dependency unavailable: {exc}") from exc
    return torch, torch_functional


def _hfv1_matrix(params: dict[str, Any], *, dtype: Any, device: Any) -> Any:
    torch, _torch_functional = _import_torch_for_frame_warp()
    rows = params.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeSkeletonError("HFV1 params missing rows")
    matrix = [
        [
            float(row["alpha"]),
            max(float(row["radius"]), 1e-6),
            max(float(row["power"]), 0.0),
            float(row["origin_x"]),
            float(row["origin_y"]),
        ]
        for row in rows
    ]
    return torch.tensor(matrix, dtype=dtype, device=device)


def _normalise_frame_grid(coords: Any, frame_height: int, frame_width: int) -> Any:
    torch, _torch_functional = _import_torch_for_frame_warp()
    x = coords[..., 0]
    y = coords[..., 1]
    gx = torch.zeros_like(x) if frame_width <= 1 else 2.0 * x / float(frame_width - 1) - 1.0
    gy = torch.zeros_like(y) if frame_height <= 1 else 2.0 * y / float(frame_height - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def functional_hyperbolic_foveation(
    rgb_frames: Any,
    foveation_params: bytes | str | Path | dict[str, Any],
    *,
    frame_indices: Sequence[int] | None = None,
) -> Any:
    """Apply archive-contained HFV1 geometry to scorer-visible RGB tensors.

    ``rgb_frames`` must be a torch tensor with shape ``(B, C, H, W)``.  This is
    a local bridge/control primitive, not a contest decoder by itself.
    """

    torch, torch_functional = _import_torch_for_frame_warp()
    if not torch.is_tensor(rgb_frames):
        raise RuntimeSkeletonError("rgb_frames must be a torch.Tensor")
    if rgb_frames.ndim != 4:
        raise RuntimeSkeletonError(
            f"rgb_frames must have shape (B, C, H, W), got {tuple(rgb_frames.shape)}"
        )
    if not rgb_frames.is_floating_point():
        raise RuntimeSkeletonError("rgb_frames must be floating point for deterministic warping")

    params = (
        foveation_params
        if isinstance(foveation_params, dict)
        else load_foveation_params(foveation_params)
    )
    batch, _channels, frame_height, frame_width = rgb_frames.shape
    if (int(params["frame_height"]), int(params["frame_width"])) != (
        int(frame_height),
        int(frame_width),
    ):
        raise RuntimeSkeletonError(
            "HFV1 image size does not match rgb_frames: "
            f"{(params['frame_height'], params['frame_width'])} vs {(frame_height, frame_width)}"
        )
    matrix = _hfv1_matrix(params, dtype=rgb_frames.dtype, device=rgb_frames.device)
    n_frames = int(matrix.shape[0])
    if frame_indices is None:
        idx = torch.arange(batch, dtype=torch.long, device=rgb_frames.device) % n_frames
    else:
        if len(frame_indices) != batch:
            raise RuntimeSkeletonError(
                f"frame_indices must have {batch} entries for batch size {batch}"
            )
        idx = torch.tensor(list(frame_indices), dtype=torch.long, device=rgb_frames.device)
        if bool(torch.any(idx < 0).item()) or bool(torch.any(idx >= n_frames).item()):
            raise RuntimeSkeletonError("frame_indices out of range for HFV1 params")

    selected = matrix[idx]
    alpha = selected[:, 0].abs().view(batch, 1, 1)
    radius_limit = selected[:, 1].view(batch, 1, 1).clamp_min(1e-6)
    power = selected[:, 2].view(batch, 1, 1).clamp_min(0.0)
    origin = selected[:, 3:5].view(batch, 1, 1, 2)

    yy = torch.arange(frame_height, dtype=rgb_frames.dtype, device=rgb_frames.device)
    xx = torch.arange(frame_width, dtype=rgb_frames.dtype, device=rgb_frames.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    delta = coords - origin
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(0.0)
    small_alpha = alpha <= 1e-6
    alpha_safe = torch.where(small_alpha, torch.ones_like(alpha), alpha)
    q_raw = torch.tanh(alpha_safe * distance) / alpha_safe
    q = torch.where(small_alpha.expand_as(q_raw), distance, q_raw)
    scale = torch.where(
        distance > 1e-6,
        q / distance.clamp_min(1e-6),
        torch.ones_like(distance),
    )
    hyperbolic = origin + scale.unsqueeze(-1) * delta
    blend_base = (1.0 - (distance / radius_limit).clamp(min=0.0, max=1.0)).clamp(
        min=0.0,
        max=1.0,
    )
    blend = blend_base.pow(power).unsqueeze(-1)
    mapped = coords + blend * (hyperbolic - coords)
    grid = _normalise_frame_grid(mapped, frame_height, frame_width)
    return torch_functional.grid_sample(
        rgb_frames,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def apply_lfv1_to_rgb_frames(
    rgb_frames: Any,
    foveation_params: bytes | str | Path | dict[str, Any],
    *,
    frame_indices: Sequence[int] | None = None,
) -> Any:
    """Named scorer-visible bridge from LFV1-derived HFV1 params to RGB frames."""

    return functional_hyperbolic_foveation(
        rgb_frames,
        foveation_params,
        frame_indices=frame_indices,
    )


def write_scorer_visible_frames(
    output_path: str | Path,
    rgb_frames: Any,
    foveation_params: bytes | str | Path | dict[str, Any],
    *,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write locally warped frames for bridge controls; still not contest inflate."""

    torch, _torch_functional = _import_torch_for_frame_warp()
    warped = apply_lfv1_to_rgb_frames(
        rgb_frames,
        foveation_params,
        frame_indices=frame_indices,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warped.detach().cpu(), path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "frame_count": int(warped.shape[0]),
        "image_size": {"height": int(warped.shape[2]), "width": int(warped.shape[3])},
    }


def rgb_frames_to_rgb24_bytes(rgb_frames: Any) -> bytes:
    """Convert ``(B, 3, H, W)`` frames into deterministic NHWC RGB24 bytes."""

    torch, _torch_functional = _import_torch_for_frame_warp()
    if not torch.is_tensor(rgb_frames):
        raise RuntimeSkeletonError("rgb_frames must be a torch.Tensor")
    if rgb_frames.ndim != 4:
        raise RuntimeSkeletonError(
            f"rgb_frames must have shape (B, 3, H, W), got {tuple(rgb_frames.shape)}"
        )
    if int(rgb_frames.shape[1]) != 3:
        raise RuntimeSkeletonError(f"rgb_frames must have 3 channels, got {rgb_frames.shape[1]}")
    if rgb_frames.dtype == torch.uint8:
        uint8_frames = rgb_frames.detach().cpu()
    elif rgb_frames.is_floating_point():
        uint8_frames = (
            rgb_frames.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .cpu()
        )
    else:
        raise RuntimeSkeletonError("rgb_frames must be float or uint8")
    return uint8_frames.permute(0, 2, 3, 1).contiguous().numpy().tobytes()


def rgb24_bytes_to_rgb_frames(
    raw: bytes,
    *,
    frame_count: int,
    frame_height: int,
    frame_width: int,
) -> Any:
    """Load deterministic NHWC RGB24 bytes into ``(B, 3, H, W)`` float frames."""

    torch, _torch_functional = _import_torch_for_frame_warp()
    n_frames = int(frame_count)
    height = int(frame_height)
    width = int(frame_width)
    if n_frames <= 0 or height <= 0 or width <= 0:
        raise RuntimeSkeletonError("RGB24 frame_count, height, and width must be positive")
    expected = n_frames * height * width * 3
    payload = bytes(raw)
    if len(payload) != expected:
        raise RuntimeSkeletonError(
            f"bad RGB24 byte count: got {len(payload)} bytes, expected {expected}"
        )
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(
        n_frames,
        height,
        width,
        3,
    )
    return tensor.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0


def apply_lfv1_adapter_to_rgb24_bytes(
    input_rgb24: bytes,
    foveation_params: bytes | str | Path | dict[str, Any],
    *,
    frame_count: int,
    frame_height: int,
    frame_width: int,
    frame_indices: Sequence[int] | None = None,
) -> bytes:
    """Adapter core: RGB24 bytes in, LFV1-derived HFV1 geometry, RGB24 bytes out."""

    frames = rgb24_bytes_to_rgb_frames(
        input_rgb24,
        frame_count=frame_count,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    warped = apply_lfv1_to_rgb_frames(
        frames,
        foveation_params,
        frame_indices=frame_indices,
    )
    return rgb_frames_to_rgb24_bytes(warped)


def _infer_rgb24_frame_count(raw: bytes, *, frame_height: int, frame_width: int) -> int:
    frame_bytes = int(frame_height) * int(frame_width) * 3
    if frame_bytes <= 0:
        raise RuntimeSkeletonError("RGB24 frame byte size must be positive")
    if len(raw) % frame_bytes != 0:
        raise RuntimeSkeletonError(
            f"RGB24 byte count {len(raw)} is not divisible by frame size {frame_bytes}"
        )
    frame_count = len(raw) // frame_bytes
    if frame_count <= 0:
        raise RuntimeSkeletonError("RGB24 payload must contain at least one frame")
    return frame_count


def apply_lfv1_adapter_to_full_rgb24_bytes(
    input_rgb24: bytes,
    foveation_params: bytes | str | Path | dict[str, Any],
    *,
    frame_height: int,
    frame_width: int,
) -> bytes:
    """Apply LFV1/HFV1 params to a full raw stream; frames past HFV1 rows stay identity."""

    torch, _torch_functional = _import_torch_for_frame_warp()
    params = (
        foveation_params
        if isinstance(foveation_params, dict)
        else load_foveation_params(foveation_params)
    )
    frame_count = _infer_rgb24_frame_count(
        input_rgb24,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    frames = rgb24_bytes_to_rgb_frames(
        input_rgb24,
        frame_count=frame_count,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    apply_count = min(frame_count, int(params["n_frames"]))
    if apply_count <= 0:
        return input_rgb24
    warped_head = apply_lfv1_to_rgb_frames(
        frames[:apply_count],
        params,
        frame_indices=list(range(apply_count)),
    )
    if apply_count < frame_count:
        frames = torch.cat([warped_head, frames[apply_count:]], dim=0)
    else:
        frames = warped_head
    return rgb_frames_to_rgb24_bytes(frames)


def _apply_lfv1_to_frame_batch_with_identity_tail(
    frames: Any,
    params: dict[str, Any],
    *,
    frame_start: int,
) -> Any:
    torch, _torch_functional = _import_torch_for_frame_warp()
    if int(frames.shape[0]) <= 0:
        return frames
    n_param_frames = int(params["n_frames"])
    indices = [int(frame_start) + offset for offset in range(int(frames.shape[0]))]
    apply_count = sum(1 for index in indices if index < n_param_frames)
    if apply_count <= 0:
        return frames
    warped = apply_lfv1_to_rgb_frames(
        frames[:apply_count],
        params,
        frame_indices=indices[:apply_count],
    )
    if apply_count < int(frames.shape[0]):
        return torch.cat([warped, frames[apply_count:]], dim=0)
    return warped


def _stream_lfv1_rgb24_file(
    *,
    input_path: Path,
    output_path: Path,
    params: dict[str, Any],
    chunk_frames: int,
) -> dict[str, Any]:
    frame_height = int(params["frame_height"])
    frame_width = int(params["frame_width"])
    frame_bytes = frame_height * frame_width * 3
    frames_per_chunk = max(int(chunk_frames), 1)
    chunk_bytes = frame_bytes * frames_per_chunk
    input_sha = hashlib.sha256()
    output_sha = hashlib.sha256()
    frame_start = 0
    input_total = 0
    output_total = 0
    changed = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as source, output_path.open("wb") as sink:
        while True:
            raw = source.read(chunk_bytes)
            if not raw:
                break
            if len(raw) % frame_bytes != 0:
                raise RuntimeSkeletonError(
                    f"RGB24 chunk from {input_path} is not frame-aligned: {len(raw)} bytes"
                )
            frame_count = len(raw) // frame_bytes
            frames = rgb24_bytes_to_rgb_frames(
                raw,
                frame_count=frame_count,
                frame_height=frame_height,
                frame_width=frame_width,
            )
            warped = _apply_lfv1_to_frame_batch_with_identity_tail(
                frames,
                params,
                frame_start=frame_start,
            )
            out = rgb_frames_to_rgb24_bytes(warped)
            input_sha.update(raw)
            output_sha.update(out)
            sink.write(out)
            input_total += len(raw)
            output_total += len(out)
            changed = changed or raw != out
            frame_start += frame_count
    if frame_start <= 0:
        raise RuntimeSkeletonError(f"base raw fixture contains no frames: {input_path}")
    return {
        "input_bytes": input_total,
        "output_bytes": output_total,
        "input_sha256": input_sha.hexdigest(),
        "output_sha256": output_sha.hexdigest(),
        "frame_count": frame_start,
        "output_changed": changed,
        "chunk_frames": frames_per_chunk,
        "peak_adapter_tensor_frames": frames_per_chunk,
    }


def write_scorer_visible_rgb24(
    output_path: str | Path,
    rgb_frames: Any,
    foveation_params: bytes | str | Path | dict[str, Any],
    *,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write deterministic local RGB24 bytes after applying LFV1-derived geometry."""

    warped = apply_lfv1_to_rgb_frames(
        rgb_frames,
        foveation_params,
        frame_indices=frame_indices,
    )
    raw = rgb_frames_to_rgb24_bytes(warped)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": str(path),
        "format": RGB24_BYTES_FORMAT,
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "frame_count": int(warped.shape[0]),
        "image_size": {"height": int(warped.shape[2]), "width": int(warped.shape[3])},
    }


def _official_base_name(line: str) -> str:
    return line.rsplit(".", 1)[0] if "." in line else line


def _read_official_file_list(file_list: str | Path) -> list[str]:
    path = Path(file_list)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def run_lfv1_official_inflate_facade(
    *,
    archive_root: str | Path,
    output_dir: str | Path,
    file_list: str | Path,
    base_raw_dir: str | Path,
    chunk_frames: int = DEFAULT_OFFICIAL_FACADE_CHUNK_FRAMES,
) -> dict[str, Any]:
    """Official-signature local facade over pre-existing base ``.raw`` outputs.

    This is a bridge harness. It proves the adapter can sit behind the
    challenge's ``inflate.sh archive_dir output_dir file_list`` shape when a
    base raw stream is supplied by a local control. It is not a self-sufficient
    contest decoder.
    """

    root = Path(archive_root)
    out_dir = Path(output_dir)
    base_dir = Path(base_raw_dir)
    params = load_foveation_params(root / FOVEATION_PARAMS_MEMBER)
    frame_height = int(params["frame_height"])
    frame_width = int(params["frame_width"])
    records: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in _read_official_file_list(file_list):
        base = _official_base_name(line)
        src = base_dir / f"{base}.raw"
        dst = out_dir / f"{base}.raw"
        if not src.is_file():
            raise RuntimeSkeletonError(f"base raw fixture missing for official facade: {src}")
        stream = _stream_lfv1_rgb24_file(
            input_path=src,
            output_path=dst,
            params=params,
            chunk_frames=chunk_frames,
        )
        records.append(
            {
                "source_name": line,
                "base": base,
                "input": {
                    "path": str(src),
                    "bytes": stream["input_bytes"],
                    "sha256": stream["input_sha256"],
                },
                "output": {
                    "path": str(dst),
                    "bytes": stream["output_bytes"],
                    "sha256": stream["output_sha256"],
                },
                "frame_count": stream["frame_count"],
                "pair_sensitivity_route_summary": pair_sensitivity_route_summary(
                    range(min(stream["frame_count"], int(params["n_frames"])))
                ),
                "output_changed": stream["output_changed"],
                "chunk_frames": stream["chunk_frames"],
                "peak_adapter_tensor_frames": stream["peak_adapter_tensor_frames"],
            }
        )
    return {
        "schema_version": 1,
        "kind": "lapose_foveation_official_signature_local_facade_run",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_root": str(root),
        "output_dir": str(out_dir),
        "file_list": str(file_list),
        "base_raw_dir": str(base_dir),
        "format": RGB24_BYTES_FORMAT,
        "image_size": {"height": frame_height, "width": frame_width},
        "hardware_contract": {
            "chunked_streaming": True,
            "chunk_frames": max(int(chunk_frames), 1),
            "device_selection": "torch_runtime_default_for_local_control_no_score_claim",
            "peak_adapter_tensor_frames": max(int(chunk_frames), 1),
            "raw_format": "flat_uint8_rgb_nhwc_no_header",
        },
        "research_fidelity_contract": {
            "external_teacher_models_at_inflate": False,
            "hf_v1_transform_family": "radial_hyperbolic_foveation_identity_limit",
            "paper_faithfulness_scope": "Telescope-style geometry control only; LA-Pose/RAFT/etc. are compress-time priors, not inflate-time dependencies",
            "contest_domain_binding": "384x512-or-header-declared dashcam RGB24 raw streams",
            "optimization_stack_scope": [
                "compress_time_pair_frame_sensitivity_weighting",
                "orthogonal_pose_vs_seg_gradient_projection",
                "stagewise_freeze_unfreeze_curriculum",
                "master_gradient_or_atom_level_byte_selection",
            ],
            "posenet_segnet_sensitivity_collapsed": False,
        },
        "records": records,
        "file_count": len(records),
        "all_outputs_written": all(Path(record["output"]["path"]).is_file() for record in records),
        "any_output_changed": any(bool(record["output_changed"]) for record in records),
        "adapter_scope": "official_signature_local_base_raw_facade_not_self_sufficient_contest_decoder",
    }


def run_lfv1_rgb24_inflate_adapter(
    *,
    input_rgb24_path: str | Path,
    output_rgb24_path: str | Path,
    foveation_params: bytes | str | Path | dict[str, Any],
    frame_count: int,
    frame_height: int,
    frame_width: int,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run the local RGB24 adapter path used before contest-inflate integration."""

    input_path = Path(input_rgb24_path)
    output_path = Path(output_rgb24_path)
    input_raw = input_path.read_bytes()
    output_raw = apply_lfv1_adapter_to_rgb24_bytes(
        input_raw,
        foveation_params,
        frame_count=frame_count,
        frame_height=frame_height,
        frame_width=frame_width,
        frame_indices=frame_indices,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_raw)
    return {
        "schema_version": 1,
        "kind": "lapose_foveation_local_rgb24_inflate_adapter_run",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "format": RGB24_BYTES_FORMAT,
        "input": {
            "path": str(input_path),
            "bytes": len(input_raw),
            "sha256": _sha256_bytes(input_raw),
        },
        "output": {
            "path": str(output_path),
            "bytes": len(output_raw),
            "sha256": _sha256_bytes(output_raw),
        },
        "frame_count": int(frame_count),
        "image_size": {"height": int(frame_height), "width": int(frame_width)},
        "frame_indices": list(frame_indices) if frame_indices is not None else None,
        "output_changed": input_raw != output_raw,
        "adapter_scope": "local_rgb24_fixture_not_full_contest_inflate",
    }


def _pack_hfv1_from_rows(
    *,
    n_frames: int,
    frame_height: int,
    frame_width: int,
    rows: Sequence[tuple[float, float, float, float, float]],
) -> bytes:
    if len(rows) != int(n_frames):
        raise RuntimeSkeletonError("HFV1 row count mismatch while packing control payload")
    body = b"".join(FOVEATION_ROW_STRUCT.pack(*values) for values in rows)
    return FOVEATION_HEADER_STRUCT.pack(
        FOVEATION_MAGIC,
        int(n_frames),
        int(frame_height),
        int(frame_width),
    ) + body


def _control_tensor_sha256(tensor: Any) -> str:
    torch, _torch_functional = _import_torch_for_frame_warp()
    raw = (
        tensor.detach()
        .cpu()
        .to(dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes()
    )
    return _sha256_bytes(raw)


def _probe_frame_indices(params: dict[str, Any]) -> list[int]:
    frame_width = int(params["frame_width"])
    frame_height = int(params["frame_height"])
    default = _default_foveation_row(frame_width, frame_height)
    selected: list[int] = []
    for row in params["rows"]:
        values = (
            float(row["alpha"]),
            float(row["radius"]),
            float(row["power"]),
            float(row["origin_x"]),
            float(row["origin_y"]),
        )
        if max(abs(values[index] - default[index]) for index in range(5)) > 1e-5:
            selected.append(int(row["frame_index"]))
        if len(selected) >= FRAME_WARP_CONTROL_MAX_PROBE_FRAMES:
            break
    return selected or [0]


def pair_sensitivity_channel_for_frame(frame_index: int) -> dict[str, Any]:
    """Expose the pair/frame routing needed by PoseNet-vs-SegNet sensitivity logic."""

    index = int(frame_index)
    if index < 0:
        raise RuntimeSkeletonError("frame_index must be non-negative")
    return {
        "frame_index": index,
        "pair_index": index // 2,
        "frame_within_pair": index % 2,
        "segnet_channel": "per_frame_segmentation_sensitivity",
        "posenet_channel": "per_pair_pose_sensitivity",
        "orthogonalization_contract": {
            "segnet_vs_posenet_gradients_must_not_be_collapsed": True,
            "recommended_compress_time_methods": [
                "pcgrad_or_gradient_projection",
                "stage_freeze_renderer_then_geometry",
                "pose_sensitive_frame_pair_waterfill",
                "seg_sensitive_per_frame_mask_or_region_weighting",
                "master_gradient_byte_or_atom_selection",
            ],
            "inflate_time_training_or_scorer_dependency_allowed": False,
        },
        "optimization_note": (
            "PoseNet and SegNet sensitivity are not interchangeable: geometry changes "
            "must be routed with both frame-level segmentation and pair-level pose anchors."
        ),
    }


def pair_sensitivity_routes(frame_indices: Sequence[int]) -> list[dict[str, Any]]:
    return [pair_sensitivity_channel_for_frame(int(frame_index)) for frame_index in frame_indices]


def pair_sensitivity_route_summary(frame_indices: Sequence[int]) -> dict[str, Any]:
    indices = [int(frame_index) for frame_index in frame_indices]
    if not indices:
        return {
            "frame_count": 0,
            "pair_count": 0,
            "first_routes": [],
            "last_routes": [],
            "posenet_segnet_sensitivity_collapsed": False,
        }
    pair_indices = sorted({index // 2 for index in indices})
    return {
        "frame_count": len(indices),
        "pair_count": len(pair_indices),
        "first_routes": pair_sensitivity_routes(indices[:4]),
        "last_routes": pair_sensitivity_routes(indices[-4:]),
        "posenet_segnet_sensitivity_collapsed": False,
        "optimization_note": (
            "Full route list is intentionally summarized for hardware/runtime logs; "
            "the frame-to-pair map is frame_index//2 with frame parity retained."
        ),
    }


def _byte_output_probe_frames(
    *,
    batch: int,
    frame_height: int,
    frame_width: int,
) -> Any:
    torch, _torch_functional = _import_torch_for_frame_warp()
    yy = torch.arange(frame_height, dtype=torch.int64).view(1, frame_height, 1)
    xx = torch.arange(frame_width, dtype=torch.int64).view(1, 1, frame_width)
    batch_offsets = torch.arange(batch, dtype=torch.int64).view(batch, 1, 1)
    checker = ((xx + yy + batch_offsets) % 2).to(dtype=torch.float32)
    inverse = 1.0 - checker
    ramp = ((3 * xx + 5 * yy + 7 * batch_offsets) % 256).to(dtype=torch.float32) / 255.0
    return torch.stack([checker, inverse, ramp], dim=1).contiguous()


def build_scorer_visible_frame_warp_control_report(foveation_params: bytes) -> dict[str, Any]:
    """Prove HFV1 params can deterministically alter scorer-visible RGB pixels."""

    raw = bytes(foveation_params)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "contract": RUNTIME_SCORER_VISIBLE_FRAME_WARP_CONTROL_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "passed": False,
        "foveation_params_sha256": _sha256_bytes(raw),
        "foveation_params_bytes": len(raw),
        "blockers": [],
    }
    try:
        torch, _torch_functional = _import_torch_for_frame_warp()
        params = load_foveation_params(raw)
        frame_height = int(params["frame_height"])
        frame_width = int(params["frame_width"])
        frame_indices = _probe_frame_indices(params)
        batch = len(frame_indices)
        probe = torch.linspace(
            0.0,
            1.0,
            steps=batch * 3 * frame_height * frame_width,
            dtype=torch.float32,
        ).view(batch, 3, frame_height, frame_width)

        real_warp = apply_lfv1_to_rgb_frames(
            probe,
            params,
            frame_indices=frame_indices,
        )
        identity_row = _default_foveation_row(frame_width, frame_height)
        identity_raw = _pack_hfv1_from_rows(
            n_frames=int(params["n_frames"]),
            frame_height=frame_height,
            frame_width=frame_width,
            rows=[identity_row for _ in range(int(params["n_frames"]))],
        )
        identity_warp = apply_lfv1_to_rgb_frames(
            probe,
            identity_raw,
            frame_indices=frame_indices,
        )
        identity_max_abs_delta = float((identity_warp - probe).abs().max().item())
        real_max_abs_delta = float((real_warp - probe).abs().max().item())
        input_sha = _control_tensor_sha256(probe)
        real_sha = _control_tensor_sha256(real_warp)
        identity_sha = _control_tensor_sha256(identity_warp)
        identity_allclose = bool(torch.allclose(identity_warp, probe, atol=1e-5, rtol=1e-5))
        real_warp_changed = real_max_abs_delta > 1e-5 and real_sha != input_sha
        passed = identity_allclose and real_warp_changed
        blockers = [] if passed else ["scorer_visible_frame_warp_control_not_passed"]
        base_report.update(
            {
                "passed": passed,
                "hf_v1_decode": {
                    "passed": True,
                    "target_frame_count": int(params["n_frames"]),
                    "image_size": {"height": frame_height, "width": frame_width},
                },
                "probe_frame_indices": frame_indices,
                "pair_sensitivity_routes": pair_sensitivity_routes(frame_indices),
                "identity_control": {
                    "passed": identity_allclose,
                    "input_sha256": input_sha,
                    "identity_output_sha256": identity_sha,
                    "max_abs_delta": identity_max_abs_delta,
                    "allclose_atol_rtol": 1e-5,
                },
                "nonidentity_control": {
                    "passed": real_warp_changed,
                    "input_sha256": input_sha,
                    "warped_output_sha256": real_sha,
                    "max_abs_delta": real_max_abs_delta,
                    "output_changed": real_sha != input_sha,
                },
                "meaning": (
                    "LFV1-derived HFV1 params are consumed by a deterministic RGB frame warp "
                    "that can change scorer-visible pixels in a local control."
                ),
                "blockers": blockers,
            }
        )
    except Exception as exc:
        base_report["blockers"] = [
            "scorer_visible_frame_warp_control_exception",
            f"{type(exc).__name__}: {exc}",
        ]
    return base_report


def build_scorer_visible_byte_output_control_report(foveation_params: bytes) -> dict[str, Any]:
    """Prove local LFV1 geometry has deterministic byte-output consequences."""

    raw = bytes(foveation_params)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "contract": RUNTIME_SCORER_VISIBLE_BYTE_OUTPUT_CONTROL_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "passed": False,
        "format": RGB24_BYTES_FORMAT,
        "foveation_params_sha256": _sha256_bytes(raw),
        "foveation_params_bytes": len(raw),
        "blockers": [],
    }
    try:
        params = load_foveation_params(raw)
        frame_height = int(params["frame_height"])
        frame_width = int(params["frame_width"])
        frame_indices = _probe_frame_indices(params)
        batch = len(frame_indices)
        probe = _byte_output_probe_frames(
            batch=batch,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        identity_row = _default_foveation_row(frame_width, frame_height)
        identity_raw = _pack_hfv1_from_rows(
            n_frames=int(params["n_frames"]),
            frame_height=frame_height,
            frame_width=frame_width,
            rows=[identity_row for _ in range(int(params["n_frames"]))],
        )
        input_bytes = rgb_frames_to_rgb24_bytes(probe)
        identity_warp = apply_lfv1_to_rgb_frames(
            probe,
            identity_raw,
            frame_indices=frame_indices,
        )
        identity_bytes = rgb_frames_to_rgb24_bytes(identity_warp)
        real_warp = apply_lfv1_to_rgb_frames(
            probe,
            params,
            frame_indices=frame_indices,
        )
        real_bytes = rgb_frames_to_rgb24_bytes(real_warp)
        byte_count = len(input_bytes)
        input_sha = _sha256_bytes(input_bytes)
        identity_sha = _sha256_bytes(identity_bytes)
        real_sha = _sha256_bytes(real_bytes)
        identity_exact = input_bytes == identity_bytes
        real_changed = input_bytes != real_bytes and input_sha != real_sha
        passed = identity_exact and real_changed
        blockers = [] if passed else ["scorer_visible_byte_output_control_not_passed"]
        base_report.update(
            {
                "passed": passed,
                "hf_v1_decode": {
                    "passed": True,
                    "target_frame_count": int(params["n_frames"]),
                    "image_size": {"height": frame_height, "width": frame_width},
                },
                "probe_frame_indices": frame_indices,
                "pair_sensitivity_routes": pair_sensitivity_routes(frame_indices),
                "byte_count": byte_count,
                "identity_byte_output_control": {
                    "passed": identity_exact,
                    "input_sha256": input_sha,
                    "identity_output_sha256": identity_sha,
                    "byte_exact": identity_exact,
                },
                "nonidentity_byte_output_control": {
                    "passed": real_changed,
                    "input_sha256": input_sha,
                    "warped_output_sha256": real_sha,
                    "output_changed": real_changed,
                },
                "meaning": (
                    "LFV1-derived HFV1 params are consumed by a deterministic local "
                    "RGB24 writer: identity params preserve emitted bytes and "
                    "nonidentity params alter emitted bytes."
                ),
                "blockers": blockers,
            }
        )
    except Exception as exc:
        base_report["blockers"] = [
            "scorer_visible_byte_output_control_exception",
            f"{type(exc).__name__}: {exc}",
        ]
    return base_report


def build_inflate_adapter_byte_output_control_report(foveation_params: bytes) -> dict[str, Any]:
    """Exercise the local RGB24 inflate-adapter core without claiming contest parity."""

    raw = bytes(foveation_params)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "lapose_foveation_local_rgb24_inflate_adapter_control_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "passed": False,
        "format": RGB24_BYTES_FORMAT,
        "foveation_params_sha256": _sha256_bytes(raw),
        "foveation_params_bytes": len(raw),
        "blockers": [],
    }
    try:
        params = load_foveation_params(raw)
        frame_height = int(params["frame_height"])
        frame_width = int(params["frame_width"])
        frame_indices = _probe_frame_indices(params)
        batch = len(frame_indices)
        probe = _byte_output_probe_frames(
            batch=batch,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        input_bytes = rgb_frames_to_rgb24_bytes(probe)
        identity_row = _default_foveation_row(frame_width, frame_height)
        identity_raw = _pack_hfv1_from_rows(
            n_frames=int(params["n_frames"]),
            frame_height=frame_height,
            frame_width=frame_width,
            rows=[identity_row for _ in range(int(params["n_frames"]))],
        )
        identity_output = apply_lfv1_adapter_to_rgb24_bytes(
            input_bytes,
            identity_raw,
            frame_count=batch,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_indices=frame_indices,
        )
        real_output = apply_lfv1_adapter_to_rgb24_bytes(
            input_bytes,
            params,
            frame_count=batch,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_indices=frame_indices,
        )
        input_sha = _sha256_bytes(input_bytes)
        identity_sha = _sha256_bytes(identity_output)
        real_sha = _sha256_bytes(real_output)
        identity_exact = input_bytes == identity_output
        real_changed = input_bytes != real_output and input_sha != real_sha
        passed = identity_exact and real_changed
        blockers = [] if passed else ["inflate_adapter_byte_output_control_not_passed"]
        base_report.update(
            {
                "passed": passed,
                "hf_v1_decode": {
                    "passed": True,
                    "target_frame_count": int(params["n_frames"]),
                    "image_size": {"height": frame_height, "width": frame_width},
                },
                "probe_frame_indices": frame_indices,
                "pair_sensitivity_routes": pair_sensitivity_routes(frame_indices),
                "byte_count": len(input_bytes),
                "identity_adapter_control": {
                    "passed": identity_exact,
                    "input_sha256": input_sha,
                    "identity_output_sha256": identity_sha,
                    "byte_exact": identity_exact,
                },
                "nonidentity_adapter_control": {
                    "passed": real_changed,
                    "input_sha256": input_sha,
                    "warped_output_sha256": real_sha,
                    "output_changed": real_changed,
                },
                "adapter_scope": "local_rgb24_fixture_not_full_contest_inflate",
                "meaning": (
                    "The local adapter core accepts RGB24 bytes, applies archive-contained "
                    "HFV1 geometry, and emits deterministic RGB24 bytes with identity and "
                    "nonidentity controls."
                ),
                "blockers": blockers,
            }
        )
    except Exception as exc:
        base_report["blockers"] = [
            "inflate_adapter_byte_output_control_exception",
            f"{type(exc).__name__}: {exc}",
        ]
    return base_report


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
    """Prove LFV1 local controls without claiming scored output parity."""

    raw = bytes(payload)
    decoded = _decode_lfv1(raw)
    reencoded = _encode_lfv1(decoded)
    structural_output = _structural_output(decoded)
    foveation_params, _foveation_bridge = lower_lfv1_to_foveation_params(decoded)
    frame_warp_control = build_scorer_visible_frame_warp_control_report(foveation_params)
    byte_output_control = build_scorer_visible_byte_output_control_report(foveation_params)
    inflate_adapter_control = build_inflate_adapter_byte_output_control_report(foveation_params)
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
    controls_passed = (
        structural_consumption_passed
        and frame_warp_control["passed"]
        and byte_output_control["passed"]
        and inflate_adapter_control["passed"]
    )

    return {
        "schema_version": 1,
        "runtime_effect_controls_contract": RUNTIME_EFFECT_CONTROLS_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "passed": controls_passed,
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
        "scorer_visible_frame_warp_control": frame_warp_control,
        "scorer_visible_byte_output_control": byte_output_control,
        "inflate_adapter_byte_output_control": inflate_adapter_control,
        "scored_runtime_output_parity": {
            "passed": False,
            "local_frame_warp_control_passed": frame_warp_control["passed"],
            "local_byte_output_control_passed": byte_output_control["passed"],
            "local_inflate_adapter_control_passed": inflate_adapter_control["passed"],
            "meaning": (
                "Deterministic local RGB frame warp, RGB24 byte-output, and local "
                "inflate-adapter controls are proven, but no full contest inflate path "
                "reconstructs and writes scorer-visible frames or masks."
            ),
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


def _parse_frame_indices(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    out: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise RuntimeSkeletonError(f"bad frame index: {item!r}") from exc
        if parsed < 0:
            raise RuntimeSkeletonError("frame indices must be non-negative")
        out.append(parsed)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("."))
    parser.add_argument("--input-rgb24", type=Path)
    parser.add_argument("--output-rgb24", type=Path)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frame-indices", type=str)
    parser.add_argument("--official-output-dir", type=Path)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--base-raw-dir", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_OFFICIAL_FACADE_CHUNK_FRAMES)
    args = parser.parse_args(argv)
    try:
        payload = verify_charged_members(args.archive_root)
        adapter_args_present = args.input_rgb24 is not None or args.output_rgb24 is not None
        official_args_present = args.official_output_dir is not None or args.file_list is not None
        if adapter_args_present and official_args_present:
            raise RuntimeSkeletonError("local RGB24 adapter args and official facade args are mutually exclusive")
        if official_args_present:
            missing_official_args = [
                name
                for name, value in (
                    ("--official-output-dir", args.official_output_dir),
                    ("--file-list", args.file_list),
                    ("--base-raw-dir", args.base_raw_dir),
                )
                if value is None
            ]
            if missing_official_args:
                raise RuntimeSkeletonError(
                    "missing required official facade argument(s): "
                    + ", ".join(missing_official_args)
                    + "; LFV1 archive is not a self-sufficient contest decoder"
                )
            facade_report = run_lfv1_official_inflate_facade(
                archive_root=args.archive_root,
                output_dir=args.official_output_dir,
                file_list=args.file_list,
                base_raw_dir=args.base_raw_dir,
                chunk_frames=int(args.chunk_frames),
            )
            payload["official_signature_local_facade_run"] = facade_report
            payload["runtime_adapter_executed"] = True
            print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
            return 0
        if adapter_args_present:
            missing_adapter_args = [
                name
                for name, value in (
                    ("--input-rgb24", args.input_rgb24),
                    ("--output-rgb24", args.output_rgb24),
                    ("--frame-count", args.frame_count),
                    ("--height", args.height),
                    ("--width", args.width),
                )
                if value is None
            ]
            if missing_adapter_args:
                raise RuntimeSkeletonError(
                    "missing required local adapter argument(s): "
                    + ", ".join(missing_adapter_args)
                )
            foveation_path = Path(args.archive_root) / FOVEATION_PARAMS_MEMBER
            adapter_report = run_lfv1_rgb24_inflate_adapter(
                input_rgb24_path=args.input_rgb24,
                output_rgb24_path=args.output_rgb24,
                foveation_params=foveation_path,
                frame_count=int(args.frame_count),
                frame_height=int(args.height),
                frame_width=int(args.width),
                frame_indices=_parse_frame_indices(args.frame_indices),
            )
            payload["local_rgb24_inflate_adapter_run"] = adapter_report
            payload["runtime_adapter_executed"] = True
            print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
            return 0
    except RuntimeSkeletonError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
