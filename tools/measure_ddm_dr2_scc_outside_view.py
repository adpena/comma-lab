#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure an exact per-record SDWL1 mode race for the DR2 outside view.

This is a research control, not a witness encoder or candidate archive format.
It deliberately omits the SDWL1 schema and lexicon to give the per-record
challenger a favorable rate bound.  A result is therefore inadmissible as a
composed candidate unless every record separately closes scorer visibility,
sensitivity-priced tolerance, and the descriptive/compact/coder decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.arith_selfcomp_rate_coders import (  # noqa: E402
    decode_spatial_context_arithmetic,
    encode_spatial_context_arithmetic,
)

DEFAULT_INVENTORY_STAGE: Final = Path(
    ".omx/research/ddm_dv2_sdwl1_n600_20260723/stage_10_fact_inventory.json"
)
BASELINE_OUTER_BYTES: Final = 68_464
EXPECTED_SEMANTIC_SHA256: Final = "e7dee11d0fd162470bb206acca3c4667c79100cc41259d5d1ecb293e31e225f3"
EXPECTED_SHAPE: Final = (600, 11, 8)
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
ROW_NAMES: Final = tuple(f"{name} partition_cell" for name in CLASS_NAMES) + tuple(
    f"{name} separatrix" for name in CLASS_NAMES
) + ("Pose pair_screw",)

PACKET_MAGIC: Final = b"RDO1"
PACKET_HEADER: Final = struct.Struct("<4sB")
STREAM_HEADER: Final = struct.Struct("<8sBBI32s")
MODE_IDS: Final = {"track": 2, "re_key": 3}
MODE_NAMES: Final = {value: key for key, value in MODE_IDS.items()}


class MeasurementError(ValueError):
    """Raised when input custody or exact parse-back fails."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic receipt bytes."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish a receipt atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_zlib_decompress(payload: bytes) -> bytes:
    decoder = zlib.decompressobj()
    try:
        restored = decoder.decompress(payload)
        restored += decoder.flush()
    except zlib.error as exc:
        raise MeasurementError("invalid outer zlib stream") from exc
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise MeasurementError("truncated or trailing outer zlib stream")
    return restored


def _semantic_sha256(tensor: np.ndarray) -> str:
    canonical = np.ascontiguousarray(tensor, dtype="<i8")
    return sha256_bytes(canonical.tobytes(order="C"))


def load_inventory(stage_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and verify the frozen SDWL1 inventory stage."""

    try:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid inventory stage {stage_path}") from exc
    if not isinstance(stage, dict) or stage.get("schema") != "sdwl1.fact_inventory_stage.v1":
        raise MeasurementError("unexpected inventory stage schema")
    payload = stage.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
        raise MeasurementError("inventory stage has no governed payload")
    payload_path = stage_path.parent / payload["path"]
    if payload_path.stat().st_size != payload.get("bytes"):
        raise MeasurementError("inventory payload byte count drift")
    if sha256_file(payload_path) != payload.get("sha256"):
        raise MeasurementError("inventory payload SHA-256 drift")
    try:
        tensor = np.load(payload_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise MeasurementError("invalid inventory NPY") from exc
    if tensor.shape != EXPECTED_SHAPE or tensor.dtype != np.dtype("<i8"):
        raise MeasurementError(f"inventory shape/dtype drift: {tensor.shape} {tensor.dtype}")
    semantic_sha256 = _semantic_sha256(tensor)
    if semantic_sha256 != stage.get("semantic_sha256") or semantic_sha256 != EXPECTED_SEMANTIC_SHA256:
        raise MeasurementError("inventory semantic SHA-256 drift")
    return np.ascontiguousarray(tensor), {
        "bytes": payload["bytes"],
        "path": str(payload_path),
        "sha256": payload["sha256"],
        "stage_path": str(stage_path),
        "stage_sha256": sha256_file(stage_path),
    }


def _temporal_delta(record: np.ndarray) -> np.ndarray:
    delta = np.ascontiguousarray(record.copy())
    delta[1:] -= record[:-1]
    return delta


def _decode_mode(payload: bytes, mode: str) -> np.ndarray:
    decoded = decode_spatial_context_arithmetic(payload)
    if mode == "track":
        decoded = np.cumsum(decoded, axis=0, dtype=np.int64)
    return np.ascontiguousarray(decoded, dtype="<i8")


def measure_record(record: np.ndarray, row_index: int) -> tuple[dict[str, Any], bytes]:
    """Race the exact admissible modes for one frozen semantic record."""

    if record.shape != (EXPECTED_SHAPE[0], 1, EXPECTED_SHAPE[2]):
        raise MeasurementError(f"unexpected record shape {record.shape}")
    re_key_payload = encode_spatial_context_arithmetic(record)
    track_payload = encode_spatial_context_arithmetic(_temporal_delta(record))
    candidates = {"track": track_payload, "re_key": re_key_payload}
    for mode, payload in candidates.items():
        if not np.array_equal(_decode_mode(payload, mode), record):
            raise MeasurementError(f"{ROW_NAMES[row_index]} {mode} parse-back mismatch")
    selected_mode = min(candidates, key=lambda mode: (len(candidates[mode]), mode))
    selected_payload = candidates[selected_mode]
    unique_states = int(np.unique(record.reshape(record.shape[0], -1), axis=0).shape[0])
    adjacent_changes = int(np.count_nonzero(np.any(record[1:] != record[:-1], axis=(1, 2))))
    row = {
        "adjacent_changes": adjacent_changes,
        "audit_triple": {
            "scorer_visibility": {
                "status": "UNPROVEN",
                "verdict": "No record-to-RGB receiver or per-DOF scorer-null quotient is present.",
            },
            "sensitivity_priced_tolerance": {
                "status": "UNMEASURED",
                "verdict": "Exact fact parse-back only; no margin/g3-lambda/c1-dual tolerance is applied.",
            },
            "three_layer_decomposition": {
                "coder_gain": {
                    "re_key_bytes": len(re_key_payload),
                    "selected_bytes": len(selected_payload),
                    "selected_mode": selected_mode,
                    "track_bytes": len(track_payload),
                },
                "descriptive_form": ROW_NAMES[row_index],
                "inherently_compact_dof_count": None,
                "status": "INCOMPLETE",
            },
        },
        "available_modes": {
            "re_key": {
                "bytes": len(re_key_payload),
                "definition": "whole-record absolute arithmetic control",
                "parseback_exact": True,
            },
            "static": {
                "admissible_exact": unique_states == 1,
                "bytes": None,
                "reason": "record is not temporally static" if unique_states != 1 else None,
            },
            "track": {
                "bytes": len(track_payload),
                "definition": "first absolute state plus causal deltas",
                "parseback_exact": True,
            },
            "xi_advect": {
                "admissible_exact": False,
                "bytes": None,
                "reason": "no decoder-free xi-to-aggregate-fact transport map or priced side information",
            },
        },
        "row_index": row_index,
        "row_name": ROW_NAMES[row_index],
        "selected_exact_mode": selected_mode,
        "selected_stream_bytes": len(selected_payload),
        "selected_stream_sha256": sha256_bytes(selected_payload),
        "unique_states": unique_states,
    }
    return row, selected_payload


def build_measurement_packet(rows: list[dict[str, Any]], streams: list[bytes]) -> bytes:
    """Build a fully framed measurement envelope, never persisted as a format."""

    if len(rows) != len(streams) or len(rows) != EXPECTED_SHAPE[1]:
        raise MeasurementError("measurement envelope requires all 11 rows")
    parts = [PACKET_HEADER.pack(PACKET_MAGIC, len(rows))]
    for row, stream in zip(rows, streams, strict=True):
        parts.append(_framed_stream(row, stream))
    return b"".join(parts)


def _framed_stream(row: dict[str, Any], stream: bytes) -> bytes:
    row_index = int(row["row_index"])
    mode = str(row["selected_exact_mode"])
    tag = f"ROW{row_index:05d}".encode("ascii")
    return (
        STREAM_HEADER.pack(
            tag,
            row_index,
            MODE_IDS[mode],
            len(stream),
            hashlib.sha256(stream).digest(),
        )
        + stream
    )


def measure_pairwise_redundancy(
    rows: list[dict[str, Any]],
    streams: list[bytes],
) -> dict[str, Any]:
    """Measure every ordered pair with an explicit outer-zlib marginal proxy.

    For ordered pair A -> B, ``bytes(B | A decoded)`` is the marginal zlib9
    byte cost ``bytes(zlib(A || B)) - bytes(zlib(A))`` over the same complete
    per-stream framing.  This is not a substitute for a true conditional
    arithmetic model; it is a deterministic detector for overlap left on the
    table by the current unconditioned streams.
    """

    framed = [_framed_stream(row, stream) for row, stream in zip(rows, streams, strict=True)]
    standalone = [len(zlib.compress(value, level=9)) for value in framed]
    ordered_pairs = []
    for owner_index, owner in enumerate(framed):
        owner_bytes = len(zlib.compress(owner, level=9))
        for target_index, target in enumerate(framed):
            if owner_index == target_index:
                continue
            conditional_bytes = len(zlib.compress(owner + target, level=9)) - owner_bytes
            redundancy_bytes = standalone[target_index] - conditional_bytes
            ordered_pairs.append(
                {
                    "conditioned_on_row": owner_index,
                    "conditioned_on_row_name": rows[owner_index]["row_name"],
                    "conditional_marginal_bytes": conditional_bytes,
                    "redundancy_bytes": redundancy_bytes,
                    "target_row": target_index,
                    "target_row_name": rows[target_index]["row_name"],
                    "target_standalone_bytes": standalone[target_index],
                }
            )
    positive = [row for row in ordered_pairs if row["redundancy_bytes"] > 0]
    return {
        "definition": (
            "ordered A->B outer-zlib9 marginal proxy: bytes(B)-"
            "(bytes(zlib(A||B))-bytes(zlib(A))); complete fixed stream framing included"
        ),
        "interpretation": (
            "positive values detect pairwise overlap, zero does not prove independence, "
            "negative values are framing/model interference and not a credit"
        ),
        "ordered_pair_count": len(ordered_pairs),
        "ordered_pairs": ordered_pairs,
        "positive_overlap_pair_count": len(positive),
        "positive_overlap_total_bytes_nonadditive": sum(row["redundancy_bytes"] for row in positive),
        "maximum_positive_overlap_bytes": max((row["redundancy_bytes"] for row in positive), default=0),
    }


def decode_measurement_packet(packet: bytes) -> np.ndarray:
    """Strictly restore the semantic tensor from the measurement envelope."""

    if len(packet) < PACKET_HEADER.size:
        raise MeasurementError("truncated measurement envelope")
    magic, count = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or count != EXPECTED_SHAPE[1]:
        raise MeasurementError("measurement envelope header drift")
    offset = PACKET_HEADER.size
    restored = np.zeros(EXPECTED_SHAPE, dtype="<i8")
    for expected_index in range(count):
        end = offset + STREAM_HEADER.size
        if end > len(packet):
            raise MeasurementError("truncated stream header")
        tag, row_index, mode_id, size, digest = STREAM_HEADER.unpack_from(packet, offset)
        offset = end
        if tag != f"ROW{expected_index:05d}".encode("ascii") or row_index != expected_index:
            raise MeasurementError("noncanonical stream order")
        if mode_id not in MODE_NAMES:
            raise MeasurementError("unknown record mode")
        end = offset + size
        if end > len(packet):
            raise MeasurementError("truncated record stream")
        stream = packet[offset:end]
        offset = end
        if hashlib.sha256(stream).digest() != digest:
            raise MeasurementError("record stream SHA-256 mismatch")
        record = _decode_mode(stream, MODE_NAMES[mode_id])
        if record.shape != (EXPECTED_SHAPE[0], 1, EXPECTED_SHAPE[2]):
            raise MeasurementError("decoded record shape drift")
        restored[:, expected_index : expected_index + 1, :] = record
    if offset != len(packet):
        raise MeasurementError("trailing measurement-envelope bytes")
    return restored


def measure(stage_path: Path) -> dict[str, Any]:
    """Run the deterministic exact-control race."""

    tensor, inventory_custody = load_inventory(stage_path)
    rows: list[dict[str, Any]] = []
    streams: list[bytes] = []
    for row_index in range(EXPECTED_SHAPE[1]):
        row, stream = measure_record(tensor[:, row_index : row_index + 1, :], row_index)
        rows.append(row)
        streams.append(stream)
    packet = build_measurement_packet(rows, streams)
    outer = zlib.compress(packet, level=9)
    restored_packet = _strict_zlib_decompress(outer)
    restored_tensor = decode_measurement_packet(restored_packet)
    if not np.array_equal(restored_tensor, tensor):
        raise MeasurementError("whole measurement-envelope parse-back mismatch")
    selected_counts = {
        mode: sum(row["selected_exact_mode"] == mode for row in rows)
        for mode in sorted(MODE_IDS)
    }
    pairwise_redundancy = measure_pairwise_redundancy(rows, streams)
    return {
        "authority": "LOCAL_CPU_EXACT_FACT_CONTROL_ONLY",
        "baseline": {
            "bytes": BASELINE_OUTER_BYTES,
            "row_id": "whole_typed_section_causal_delta",
            "source": ".omx/research/ddm_dv2_sdwl1_n600_20260723/receipt.json",
        },
        "global_audit_triple": {
            "scorer_visibility": "FAIL_UNPROVEN",
            "sensitivity_priced_tolerance": "FAIL_EXACT_ONLY",
            "three_layer_decomposition": "FAIL_COMPACT_DOF_UNMEASURED",
            "candidate_admissible": False,
        },
        "non_redundancy_audit": {
            "corrections_are_deltas": {
                "status": "NOT_APPLICABLE",
                "verdict": "This control emits no correction stream.",
            },
            "cross_stream_conditional_coding": {
                "status": "FAIL_UNCONDITIONED_ARITHMETIC_STREAMS",
                "pairwise_redundancy": pairwise_redundancy,
            },
            "dimension_homes": {
                "status": "FAIL_UNMEASURED",
                "verdict": (
                    "All 600 exact per-pair states are coded; no persistent-primitive versus "
                    "innovation/event ownership split has been measured."
                ),
            },
            "single_owner_fact_rule": {
                "status": "PASS_WITHIN_DECLARED_TENSOR_ONLY",
                "verdict": (
                    "Each of the 45,600 declared scalar coordinates has one row owner; ownership "
                    "against a future receiver/correction stream is not established."
                ),
            },
        },
        "implementation": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "tool_path": str(Path(__file__).resolve().relative_to(REPO)),
            "tool_sha256": sha256_file(Path(__file__)),
        },
        "inventory": {
            **inventory_custody,
            "semantic_sha256": _semantic_sha256(tensor),
            "shape": list(tensor.shape),
        },
        "measurement_envelope": {
            "definition": (
                "favorable measurement-only envelope: 11 selected exact streams with fixed row/mode/length/SHA "
                "framing; SDWL1 lexicon and schema omitted"
            ),
            "inner_bytes": len(packet),
            "outer_zlib9_bytes": len(outer),
            "outer_zlib9_sha256": sha256_bytes(outer),
            "parseback_exact": True,
            "semantic_sha256": _semantic_sha256(restored_tensor),
        },
        "result": {
            "byte_delta_vs_68464": len(outer) - BASELINE_OUTER_BYTES,
            "candidate_admissible": False,
            "selected_mode_counts": selected_counts,
            "verdict": "DOMINATED_EXACT_CODER_LAYER_CONTROL",
        },
        "rows": rows,
        "schema": "ddm.dr2.scc_outside_view.mode_race.v1",
        "verdict_scope": (
            "exact frozen SDWL1 fact tensor and favorable measurement framing only; excludes tolerance, "
            "scorer-null quotient, RGB receiver, R survival, evaluator output, archive bytes, and contest score"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-stage", type=Path, default=DEFAULT_INVENTORY_STAGE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.inventory_stage)
    payload = canonical_json_bytes(result)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        atomic_bytes(args.output, payload)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
