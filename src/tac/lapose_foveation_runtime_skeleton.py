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
        "runtime_output_parity_proven": False,
        "noop_controls_proven": False,
        "exact_cuda_auth_eval_proven": False,
        "dispatch_blockers": [
            "lapose_foveation_runtime_skeleton_not_a_decoder",
            "lapose_foveation_runtime_output_parity_missing",
            "lapose_foveation_noop_controls_missing",
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
