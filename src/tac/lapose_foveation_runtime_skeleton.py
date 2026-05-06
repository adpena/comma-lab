"""Fail-closed runtime skeleton for LA-POSE foveation tuple payloads.

This module is intentionally not a contest decoder. The local LFV1 archive
builder packages it as a charged archive member so archive custody can prove
that the runtime has only archive-contained LFV1 bytes available while runtime
output parity, no-op controls, and exact CUDA auth eval remain blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

PAYLOAD_MEMBER = "lapose_foveation_tuples.lfv1"
PROOF_MEMBER = "runtime_consumer_proof_skeleton.json"
REQUIRED_MEMBERS = (PAYLOAD_MEMBER, PROOF_MEMBER)
PAYLOAD_MAGIC = b"LFV1"
HEADER_STRUCT = struct.Struct("<4sHHHH")
ROW_STRUCT = struct.Struct("<BHHHHHH")
RUNTIME_PROOF_SKELETON_CONTRACT = "lapose_foveation_runtime_consumer_proof_skeleton_v1"
RUNTIME_EFFECT_CONTROLS_CONTRACT = "lapose_foveation_runtime_effect_controls_v1"
RUNTIME_STRUCTURAL_OUTPUT_CONTRACT = "lapose_foveation_runtime_structural_output_v1"
UINT16_MAX = 65_535


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
        routes.append(
            {
                "pair_index": int(row["pair_index"]),
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
    proof_path = root / PROOF_MEMBER
    payload_raw = payload_path.read_bytes()
    decoded = _decode_lfv1(payload_raw)
    runtime_effect_controls = build_runtime_effect_control_report(payload_raw)
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

    return {
        "schema_version": 1,
        "kind": "lapose_foveation_runtime_skeleton_member_check",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "charged_members_verified": records,
        "lfv1_payload_decode": decoded,
        "runtime_effect_controls": runtime_effect_controls,
        "structural_runtime_consumption_proven": runtime_effect_controls[
            "structural_runtime_consumption"
        ]["passed"],
        "runtime_output_parity_proven": False,
        "scored_runtime_output_parity_proven": False,
        "noop_controls_proven": runtime_effect_controls["passed"],
        "exact_cuda_auth_eval_proven": False,
        "dispatch_blockers": [
            "lapose_foveation_runtime_skeleton_not_a_decoder",
            "lapose_foveation_scored_runtime_output_parity_missing",
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
