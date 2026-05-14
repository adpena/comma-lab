# SPDX-License-Identifier: MIT
"""Planning-only byte payloads for LA-POSE foveation tuple atoms.

The payload here is byte-closed at the local artifact level: selected
``lapose_foveation_transport_atom`` rows are lowered into deterministic bytes
with a recorded SHA-256. It is not archive evidence. The readiness manifest
therefore remains fail-closed until a contest runtime consumes the member,
no-op controls prove the bytes matter, and exact CUDA auth eval runs.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import repo_relative, sha256_bytes, sha256_file

SCHEMA_VERSION = 1
SCHEMA = "lapose_foveation_tuple_payload_readiness_v1"
TOOL = "tac.analysis.lapose_foveation_payload.build_lapose_foveation_tuple_payload_artifact"
PAYLOAD_MAGIC = b"LFV1"
PAYLOAD_MEMBER = "lapose_foveation_tuples.lfv1"
HEADER_STRUCT = struct.Struct("<4sHHHH")
ROW_STRUCT = struct.Struct("<BHHHHHH")
DEFAULT_OPCODE = 1
UINT16_MAX = 65_535
ALPHA_RANGE = (0.0, 8.0)
POWER_RANGE = (0.0, 8.0)
DISPATCH_BLOCKERS = [
    "lapose_foveation_tuple_payload_planning_only",
    "not_archive_consumed_payload",
    "no_runtime_consumer",
    "no_noop_controls",
    "no_exact_cuda_eval",
    "exact_cuda_auth_eval_required_before_score_claim",
]


class LaposeFoveationPayloadError(ValueError):
    """Raised when LA-POSE foveation tuple payload packing fails closed."""


def build_lapose_foveation_tuple_payload_artifact(
    manifest: Mapping[str, Any],
    *,
    payload_path: str | Path,
    repo_root: str | Path,
    selected_atom_ids: Sequence[str] | None = None,
    max_atoms: int | None = None,
    source_manifest_path: str | Path | None = None,
    source_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Write a deterministic local tuple payload and return readiness JSON."""

    payload, pack_manifest = pack_lapose_foveation_tuple_payload(
        manifest,
        selected_atom_ids=selected_atom_ids,
        max_atoms=max_atoms,
    )
    target = Path(payload_path)
    root = Path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    payload_sha256 = sha256_file(target)
    if payload_sha256 != sha256_bytes(payload):
        raise LaposeFoveationPayloadError("written payload SHA-256 does not match in-memory bytes")

    source_path_text = repo_relative(source_manifest_path, root) if source_manifest_path else ""
    return {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "ok": True,
        "evidence_grade": "empirical_payload_custody_planning_only",
        "wire_format": PAYLOAD_MAGIC.decode("ascii"),
        "member": PAYLOAD_MEMBER,
        "path": repo_relative(target, root),
        "bytes": target.stat().st_size,
        "sha256": payload_sha256,
        "byte_delta": target.stat().st_size,
        "source_manifest": {
            "path": source_path_text,
            "sha256": source_manifest_sha256,
            "schema": str(manifest.get("schema") or ""),
            "source": str(manifest.get("source") or ""),
            "record_sha256": str(manifest.get("record_sha256") or ""),
            "atom_count": _non_bool_int(manifest.get("atom_count"), default=len(pack_manifest["selected_atoms"])),
            "score_claim": manifest.get("score_claim", False),
            "ready_for_exact_eval_dispatch": manifest.get("ready_for_exact_eval_dispatch", False),
        },
        "payload": {
            "magic": PAYLOAD_MAGIC.decode("ascii"),
            "header_bytes": HEADER_STRUCT.size,
            "tuple_row_bytes": ROW_STRUCT.size,
            "tuple_body_bytes": ROW_STRUCT.size * len(pack_manifest["selected_atoms"]),
            "row_count": len(pack_manifest["selected_atoms"]),
            "total_bytes": target.stat().st_size,
            "sha256": payload_sha256,
        },
        "byte_closure": {
            "local_payload_bytes_measured": True,
            "local_payload_sha256_measured": True,
            "archive_member_proven": False,
            "archive_consumed_by_runtime": False,
            "noop_controls_ran": False,
            "exact_cuda_auth_eval_ran": False,
        },
        "runtime_contract": {
            "charged_member_required": PAYLOAD_MEMBER,
            "runtime_consumer_required": True,
            "noop_controls_required": True,
            "exact_cuda_auth_eval_required": True,
            "scorer_loads_at_pack_time": False,
        },
        "selection": pack_manifest["selection"],
        "frame_contract": pack_manifest["frame_contract"],
        "quantization": pack_manifest["quantization"],
        "selected_atoms": pack_manifest["selected_atoms"],
        "decoded_tuple_preview": decode_lapose_foveation_tuple_payload(payload),
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


def pack_lapose_foveation_tuple_payload(
    manifest: Mapping[str, Any],
    *,
    selected_atom_ids: Sequence[str] | None = None,
    max_atoms: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Encode selected foveation atom rows into deterministic ``LFV1`` bytes."""

    atoms = _select_atoms(manifest, selected_atom_ids=selected_atom_ids, max_atoms=max_atoms)
    frame_contract = _frame_contract(manifest)
    frame_width = int(frame_contract["width"])
    frame_height = int(frame_contract["height"])
    radius_range = (0.0, math.hypot(float(frame_width), float(frame_height)))
    quantization = {
        "schema": "lapose_foveation_tuple_quantization_v1",
        "scalar_bits": 16,
        "rounding": "floor(value * 65535 + 0.5)",
        "ranges": {
            "alpha": list(ALPHA_RANGE),
            "radius": [round(radius_range[0], 12), round(radius_range[1], 12)],
            "power": list(POWER_RANGE),
            "origin_x": [0.0, float(frame_width - 1)],
            "origin_y": [0.0, float(frame_height - 1)],
        },
    }

    rows: list[dict[str, Any]] = []
    body = bytearray()
    for offset_index, atom in enumerate(sorted(atoms, key=_atom_sort_key)):
        row = _encode_atom_row(
            atom,
            frame_width=frame_width,
            frame_height=frame_height,
            radius_range=radius_range,
            quantization=quantization,
            tuple_offset=HEADER_STRUCT.size + offset_index * ROW_STRUCT.size,
        )
        body.extend(row.pop("_row_bytes"))
        rows.append(row)

    header = HEADER_STRUCT.pack(
        PAYLOAD_MAGIC,
        SCHEMA_VERSION,
        len(rows),
        frame_width,
        frame_height,
    )
    payload = header + bytes(body)
    return payload, {
        "selection": {
            "policy": "select_manifest_rows_then_sort_by_pair_index_for_wire_stability",
            "requested_atom_ids": list(selected_atom_ids or []),
            "max_atoms": max_atoms,
            "selected_atom_count": len(rows),
        },
        "frame_contract": frame_contract,
        "quantization": quantization,
        "selected_atoms": rows,
    }


def decode_lapose_foveation_tuple_payload(payload: bytes) -> dict[str, Any]:
    """Parse ``LFV1`` payload structure without interpreting runtime effects."""

    raw = bytes(payload)
    if len(raw) < HEADER_STRUCT.size:
        raise LaposeFoveationPayloadError("LFV1 payload shorter than header")
    magic, version, row_count, frame_width, frame_height = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
    if magic != PAYLOAD_MAGIC:
        raise LaposeFoveationPayloadError(f"bad LFV1 magic: {magic!r}")
    expected = HEADER_STRUCT.size + int(row_count) * ROW_STRUCT.size
    if len(raw) != expected:
        raise LaposeFoveationPayloadError(f"bad LFV1 payload size: got {len(raw)} bytes, expected {expected}")
    rows = []
    pos = HEADER_STRUCT.size
    for row_index in range(int(row_count)):
        opcode, pair_index, alpha_q, radius_q, power_q, origin_x_q, origin_y_q = ROW_STRUCT.unpack(
            raw[pos : pos + ROW_STRUCT.size]
        )
        rows.append(
            {
                "row_index": row_index,
                "byte_offset": pos,
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
        pos += ROW_STRUCT.size
    return {
        "magic": magic.decode("ascii"),
        "schema_version": int(version),
        "row_count": int(row_count),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "rows": rows,
    }


def _select_atoms(
    manifest: Mapping[str, Any],
    *,
    selected_atom_ids: Sequence[str] | None,
    max_atoms: int | None,
) -> list[Mapping[str, Any]]:
    if manifest.get("score_claim") is True:
        raise LaposeFoveationPayloadError("source manifest score_claim must not be true")
    if manifest.get("ready_for_exact_eval_dispatch") is True:
        raise LaposeFoveationPayloadError("source manifest must not already request exact eval dispatch")
    raw_atoms = manifest.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise LaposeFoveationPayloadError("source manifest must contain nonempty atoms list")
    atoms = []
    for index, atom in enumerate(raw_atoms):
        if not isinstance(atom, Mapping):
            raise LaposeFoveationPayloadError(f"atom {index} is not an object")
        atoms.append(atom)

    requested = [str(atom_id) for atom_id in selected_atom_ids or []]
    if len(set(requested)) != len(requested):
        raise LaposeFoveationPayloadError("selected_atom_ids contains duplicates")
    if requested:
        requested_set = set(requested)
        by_id = {str(atom.get("atom_id") or ""): atom for atom in atoms}
        missing = sorted(atom_id for atom_id in requested_set if atom_id not in by_id)
        if missing:
            raise LaposeFoveationPayloadError(f"selected_atom_ids not found: {', '.join(missing)}")
        atoms = [atom for atom in atoms if str(atom.get("atom_id") or "") in requested_set]

    if max_atoms is not None:
        if max_atoms <= 0:
            raise LaposeFoveationPayloadError("max_atoms must be positive when provided")
        atoms = atoms[:max_atoms]
    if not atoms:
        raise LaposeFoveationPayloadError("no atoms selected")
    _reject_duplicate_pair_indices(atoms)
    return atoms


def _frame_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = manifest.get("frame_contract")
    if not isinstance(contract, Mapping):
        raise LaposeFoveationPayloadError("source manifest missing frame_contract")
    width = _required_u16(contract.get("width"), "frame_contract.width")
    height = _required_u16(contract.get("height"), "frame_contract.height")
    return {
        "width": width,
        "height": height,
        "base_foveal_center": list(contract.get("base_foveal_center") or []),
        "center_gain": list(contract.get("center_gain") or []),
    }


def _encode_atom_row(
    atom: Mapping[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    radius_range: tuple[float, float],
    quantization: Mapping[str, Any],
    tuple_offset: int,
) -> dict[str, Any]:
    if atom.get("score_claim") is True:
        raise LaposeFoveationPayloadError(f"{atom.get('atom_id', '<unknown>')}: atom score_claim must not be true")
    if atom.get("dispatch_attempted") is True:
        raise LaposeFoveationPayloadError(f"{atom.get('atom_id', '<unknown>')}: atom dispatch_attempted must not be true")
    if atom.get("ready_for_exact_eval_dispatch") is True:
        raise LaposeFoveationPayloadError(
            f"{atom.get('atom_id', '<unknown>')}: atom must remain non-dispatchable"
        )
    params = atom.get("foveation_parameters")
    if not isinstance(params, Mapping):
        raise LaposeFoveationPayloadError(f"{atom.get('atom_id', '<unknown>')}: missing foveation_parameters")
    atom_id = str(atom.get("atom_id") or "")
    if not atom_id:
        raise LaposeFoveationPayloadError("atom_id is required")
    pair_value = params.get("pair_index")
    if pair_value is None:
        pair_support = atom.get("pair_support")
        if isinstance(pair_support, Sequence) and pair_support and not isinstance(pair_support, str | bytes):
            pair_value = pair_support[0]
    pair_index = _required_u16(pair_value, f"{atom_id}.pair_index")
    alpha = _finite_float(params.get("alpha"), f"{atom_id}.alpha")
    radius = _finite_float(params.get("radius"), f"{atom_id}.radius")
    power = _finite_float(params.get("power"), f"{atom_id}.power")
    origin_x = _finite_float(params.get("origin_x"), f"{atom_id}.origin_x")
    origin_y = _finite_float(params.get("origin_y"), f"{atom_id}.origin_y")
    ranges = quantization["ranges"]
    encoded = {
        "alpha": _quantize_u16(alpha, ALPHA_RANGE, f"{atom_id}.alpha"),
        "radius": _quantize_u16(radius, radius_range, f"{atom_id}.radius"),
        "power": _quantize_u16(power, POWER_RANGE, f"{atom_id}.power"),
        "origin_x": _quantize_u16(origin_x, (0.0, float(frame_width - 1)), f"{atom_id}.origin_x"),
        "origin_y": _quantize_u16(origin_y, (0.0, float(frame_height - 1)), f"{atom_id}.origin_y"),
    }
    row_bytes = ROW_STRUCT.pack(
        DEFAULT_OPCODE,
        pair_index,
        encoded["alpha"],
        encoded["radius"],
        encoded["power"],
        encoded["origin_x"],
        encoded["origin_y"],
    )
    return {
        "_row_bytes": row_bytes,
        "atom_id": atom_id,
        "pair_index": pair_index,
        "opcode": DEFAULT_OPCODE,
        "tuple_byte_offset": tuple_offset,
        "tuple_bytes": ROW_STRUCT.size,
        "source_parameters": {
            "alpha": round(alpha, 12),
            "radius": round(radius, 12),
            "power": round(power, 12),
            "origin_x": round(origin_x, 12),
            "origin_y": round(origin_y, 12),
        },
        "quantized": encoded,
        "quantization_ranges": ranges,
        "expected_seg_dist_delta": float(atom.get("expected_seg_dist_delta", 0.0)),
        "expected_pose_dist_delta": float(atom.get("expected_pose_dist_delta", 0.0)),
        "confidence": float(atom.get("confidence", 1.0)),
    }


def _reject_duplicate_pair_indices(atoms: Sequence[Mapping[str, Any]]) -> None:
    seen: set[int] = set()
    duplicates: list[int] = []
    for atom in atoms:
        params = atom.get("foveation_parameters") if isinstance(atom.get("foveation_parameters"), Mapping) else {}
        pair_index = _required_u16(params.get("pair_index"), f"{atom.get('atom_id', '<unknown>')}.pair_index")
        if pair_index in seen:
            duplicates.append(pair_index)
        seen.add(pair_index)
    if duplicates:
        joined = ", ".join(str(value) for value in duplicates)
        raise LaposeFoveationPayloadError(f"duplicate foveation pair_index values: {joined}")


def _atom_sort_key(atom: Mapping[str, Any]) -> tuple[int, str]:
    params = atom.get("foveation_parameters") if isinstance(atom.get("foveation_parameters"), Mapping) else {}
    return int(params.get("pair_index", 0)), str(atom.get("atom_id") or "")


def _required_u16(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaposeFoveationPayloadError(f"{label} must be an integer")
    if not 0 <= value <= UINT16_MAX:
        raise LaposeFoveationPayloadError(f"{label} must fit uint16")
    return int(value)


def _non_bool_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LaposeFoveationPayloadError(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise LaposeFoveationPayloadError(f"{label} must be finite")
    return out


def _quantize_u16(value: float, span: tuple[float, float], label: str) -> int:
    low, high = span
    if high <= low:
        raise LaposeFoveationPayloadError(f"{label} quantization range is invalid")
    if value < low or value > high:
        raise LaposeFoveationPayloadError(f"{label}={value} outside quantization range [{low}, {high}]")
    scaled = (value - low) / (high - low)
    return max(0, min(UINT16_MAX, math.floor(scaled * UINT16_MAX + 0.5)))


__all__ = [
    "PAYLOAD_MAGIC",
    "PAYLOAD_MEMBER",
    "SCHEMA",
    "SCHEMA_VERSION",
    "LaposeFoveationPayloadError",
    "build_lapose_foveation_tuple_payload_artifact",
    "decode_lapose_foveation_tuple_payload",
    "pack_lapose_foveation_tuple_payload",
]
