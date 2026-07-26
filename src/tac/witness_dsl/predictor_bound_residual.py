# SPDX-License-Identifier: MIT
"""Identity-bound semantic residuals for predictive witness codecs.

A residual is meaningful only in the coordinate system of the exact decoded
predictor it corrects.  This envelope binds an S2 partition-event payload to:

* the counted predictor-program bytes;
* the predictor renderer contract and source identity;
* the exact decoded predictor semantic stream; and
* the exact semantic target stream recovered after applying the events.

The receiver refuses predictor-program, renderer, or semantic-stream swaps
before applying a residual.  This closes the reference-frame bug in which a
mathematically exact C1 residual was considered composable with an unrelated
W_seg predictor.  The target-stream digest is an integrity checksum, but the
event payload itself is a lossless predictor-conditional encoding of the
caller-supplied semantic target.  This packet is therefore a research teacher
and conditional-entropy measurement, not a candidate-admissible payload and
not score evidence.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from tac.optimization.s2_partition_seed import (
    PACKET_SCHEMA as S2_PACKET_SCHEMA,
)
from tac.optimization.s2_partition_seed import (
    SEMANTIC_NAMES,
    PartitionEvent,
    PartitionEventSeed,
    PartitionSeedError,
    apply_partition_seed,
    decode_partition_seed,
    encode_partition_seed,
)

PACKET_SCHEMA: Final = "tac.predictor_bound_partition_residual.v1"
PACKET_MAGIC: Final = b"PBR1"
PACKET_VERSION: Final = 1
PREDICTOR_CONTRACT_MAX_BYTES: Final = 256
_PREFIX: Final = struct.Struct("<4sIII")
_CRC: Final = struct.Struct("<I")
_HEADER_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "residual_codec_schema",
        "predictor_contract_id",
        "predictor_renderer_sha256",
        "predictor_program_bytes",
        "predictor_program_sha256",
        "predictor_semantic_bytes",
        "predictor_semantic_sha256",
        "target_semantic_bytes",
        "target_semantic_sha256",
        "residual_payload_bytes",
        "residual_payload_sha256",
        "n_pairs",
        "height",
        "width",
        "event_count",
        "semantic_names",
        "semantic_class_ids",
    }
)


class PredictorBoundResidualError(ValueError):
    """Fail-closed predictor identity, packet, geometry, or apply error."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PredictorBoundResidualError("header must be finite canonical ASCII JSON") from exc


def _sha256(value: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PredictorBoundResidualError(f"{label} must be a lowercase SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise PredictorBoundResidualError(f"{label} must be a lowercase SHA-256 hex digest") from exc
    if len(decoded) != 32 or value != value.lower():
        raise PredictorBoundResidualError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_contract(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PredictorBoundResidualError("predictor_contract_id must be a non-empty string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PredictorBoundResidualError("predictor_contract_id must be ASCII") from exc
    if len(encoded) > PREDICTOR_CONTRACT_MAX_BYTES:
        raise PredictorBoundResidualError("predictor_contract_id is too long")
    return value


def _semantic_array(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or any(int(dimension) <= 0 for dimension in array.shape):
        raise PredictorBoundResidualError(f"{label} must have pair x height x width geometry")
    if array.dtype.kind not in ("i", "u"):
        raise PredictorBoundResidualError(f"{label} must contain integer class ids")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= len(SEMANTIC_NAMES)):
        raise PredictorBoundResidualError(f"{label} class ids must be in [0,4]")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _semantic_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value, dtype=np.uint8)).cast("B"))


def _semantic_ids(value: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise PredictorBoundResidualError("semantic_class_ids must contain integers")
    if len(ids) != len(SEMANTIC_NAMES) or set(ids) != set(range(len(SEMANTIC_NAMES))):
        raise PredictorBoundResidualError("semantic_class_ids must be a permutation of 0..4")
    return ids


def _events_from_difference(predictor: np.ndarray, target: np.ndarray) -> tuple[PartitionEvent, ...]:
    events: list[PartitionEvent] = []
    for pair in range(predictor.shape[0]):
        rows, cols = np.nonzero(predictor[pair] != target[pair])
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            events.append(
                PartitionEvent(
                    pair=pair,
                    row=row,
                    col=col,
                    target_class=int(target[pair, row, col]),
                    baseline_class=int(predictor[pair, row, col]),
                )
            )
    return tuple(events)


@dataclass(frozen=True)
class PredictorBoundPartitionResidual:
    """Strictly decoded identity-bound residual envelope."""

    header: Mapping[str, Any]
    residual_payload: bytes
    seed: PartitionEventSeed

    @property
    def predictor_program_sha256(self) -> str:
        return str(self.header["predictor_program_sha256"])

    @property
    def predictor_semantic_sha256(self) -> str:
        return str(self.header["predictor_semantic_sha256"])

    @property
    def target_semantic_sha256(self) -> str:
        return str(self.header["target_semantic_sha256"])


def build_predictor_bound_partition_residual(
    *,
    predictor_program: bytes,
    predictor_contract_id: str,
    predictor_renderer_sha256: str,
    predictor_labels: np.ndarray,
    target_labels: np.ndarray,
    semantic_class_ids: Sequence[int] = tuple(range(len(SEMANTIC_NAMES))),
) -> bytes:
    """Build a deterministic counted residual bound to one decoded predictor."""

    if not isinstance(predictor_program, bytes) or not predictor_program:
        raise PredictorBoundResidualError("predictor_program must be non-empty counted bytes")
    contract = _require_contract(predictor_contract_id)
    renderer_sha = _require_sha256(predictor_renderer_sha256, "predictor_renderer_sha256")
    predictor = _semantic_array(predictor_labels, label="predictor_labels")
    target = _semantic_array(target_labels, label="target_labels")
    if predictor.shape != target.shape:
        raise PredictorBoundResidualError("predictor and target semantic geometry differs")
    ids = _semantic_ids(semantic_class_ids)
    n_pairs, height, width = (int(value) for value in predictor.shape)
    seed = PartitionEventSeed(
        n_pairs=n_pairs,
        height=height,
        width=width,
        semantic_class_ids=ids,
        events=_events_from_difference(predictor, target),
    )
    residual = encode_partition_seed(seed)
    semantic_bytes = int(predictor.size)
    header = {
        "schema": PACKET_SCHEMA,
        "version": PACKET_VERSION,
        "residual_codec_schema": S2_PACKET_SCHEMA,
        "predictor_contract_id": contract,
        "predictor_renderer_sha256": renderer_sha,
        "predictor_program_bytes": len(predictor_program),
        "predictor_program_sha256": _sha256(predictor_program),
        "predictor_semantic_bytes": semantic_bytes,
        "predictor_semantic_sha256": _semantic_sha256(predictor),
        "target_semantic_bytes": semantic_bytes,
        "target_semantic_sha256": _semantic_sha256(target),
        "residual_payload_bytes": len(residual),
        "residual_payload_sha256": _sha256(residual),
        "n_pairs": n_pairs,
        "height": height,
        "width": width,
        "event_count": len(seed.events),
        "semantic_names": list(SEMANTIC_NAMES),
        "semantic_class_ids": list(ids),
    }
    header_bytes = _canonical_json(header)
    prefix = _PREFIX.pack(PACKET_MAGIC, PACKET_VERSION, len(header_bytes), len(residual))
    checksum = _CRC.pack(zlib.crc32(header_bytes + residual) & 0xFFFFFFFF)
    return prefix + header_bytes + residual + checksum


def decode_predictor_bound_partition_residual(payload: bytes) -> PredictorBoundPartitionResidual:
    """Strictly parse and cross-check an identity-bound residual packet."""

    if not isinstance(payload, bytes) or len(payload) < _PREFIX.size + _CRC.size:
        raise PredictorBoundResidualError("predictor-bound residual is truncated or not bytes")
    magic, version, header_size, residual_size = _PREFIX.unpack_from(payload)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise PredictorBoundResidualError("predictor-bound residual magic/version mismatch")
    expected = _PREFIX.size + header_size + residual_size + _CRC.size
    if len(payload) != expected:
        raise PredictorBoundResidualError("predictor-bound residual length mismatch or trailing bytes")
    header_start = _PREFIX.size
    residual_start = header_start + header_size
    residual_end = residual_start + residual_size
    header_bytes = payload[header_start:residual_start]
    residual = payload[residual_start:residual_end]
    (stored_crc,) = _CRC.unpack(payload[residual_end:])
    if stored_crc != (zlib.crc32(header_bytes + residual) & 0xFFFFFFFF):
        raise PredictorBoundResidualError("predictor-bound residual CRC mismatch")
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorBoundResidualError("predictor-bound residual header is not ASCII JSON") from exc
    if not isinstance(header, dict) or frozenset(header) != _HEADER_FIELDS:
        raise PredictorBoundResidualError("predictor-bound residual header fields mismatch")
    if _canonical_json(header) != header_bytes:
        raise PredictorBoundResidualError("predictor-bound residual header is not canonical")
    if (
        header["schema"] != PACKET_SCHEMA
        or header["version"] != PACKET_VERSION
        or header["residual_codec_schema"] != S2_PACKET_SCHEMA
    ):
        raise PredictorBoundResidualError("predictor-bound residual schema/version differs")
    _require_contract(header["predictor_contract_id"])
    _require_sha256(header["predictor_renderer_sha256"], "predictor_renderer_sha256")
    _require_sha256(header["predictor_program_sha256"], "predictor_program_sha256")
    _require_sha256(header["predictor_semantic_sha256"], "predictor_semantic_sha256")
    _require_sha256(header["target_semantic_sha256"], "target_semantic_sha256")
    _require_sha256(header["residual_payload_sha256"], "residual_payload_sha256")
    integer_fields = (
        "predictor_program_bytes",
        "predictor_semantic_bytes",
        "target_semantic_bytes",
        "residual_payload_bytes",
        "n_pairs",
        "height",
        "width",
        "event_count",
    )
    if any(isinstance(header[name], bool) or not isinstance(header[name], int) for name in integer_fields):
        raise PredictorBoundResidualError("predictor-bound residual integer field is malformed")
    if any(header[name] <= 0 for name in integer_fields if name != "event_count") or header["event_count"] < 0:
        raise PredictorBoundResidualError("predictor-bound residual sizes/geometry must be positive")
    semantic_bytes = header["n_pairs"] * header["height"] * header["width"]
    if (
        header["semantic_names"] != list(SEMANTIC_NAMES)
        or tuple(header["semantic_class_ids"]) != _semantic_ids(header["semantic_class_ids"])
        or header["predictor_semantic_bytes"] != semantic_bytes
        or header["target_semantic_bytes"] != semantic_bytes
        or header["residual_payload_bytes"] != len(residual)
        or header["residual_payload_sha256"] != _sha256(residual)
    ):
        raise PredictorBoundResidualError("predictor-bound residual header custody differs")
    try:
        seed = decode_partition_seed(residual)
    except PartitionSeedError as exc:
        raise PredictorBoundResidualError("nested residual payload is invalid") from exc
    if (
        seed.n_pairs != header["n_pairs"]
        or seed.height != header["height"]
        or seed.width != header["width"]
        or tuple(seed.semantic_class_ids) != tuple(header["semantic_class_ids"])
        or len(seed.events) != header["event_count"]
        or encode_partition_seed(seed) != residual
    ):
        raise PredictorBoundResidualError("nested residual identity differs from its envelope")
    return PredictorBoundPartitionResidual(header=header, residual_payload=residual, seed=seed)


def apply_predictor_bound_partition_residual(
    payload: bytes,
    *,
    predictor_program: bytes,
    predictor_contract_id: str,
    predictor_renderer_sha256: str,
    predictor_labels: np.ndarray,
) -> np.ndarray:
    """Apply only after exact predictor program, renderer, and cells all match."""

    decoded = decode_predictor_bound_partition_residual(payload)
    header = decoded.header
    predictor = _semantic_array(predictor_labels, label="predictor_labels")
    if (
        not isinstance(predictor_program, bytes)
        or len(predictor_program) != header["predictor_program_bytes"]
        or _sha256(predictor_program) != header["predictor_program_sha256"]
    ):
        raise PredictorBoundResidualError("predictor program identity differs from residual custody")
    if _require_contract(predictor_contract_id) != header["predictor_contract_id"]:
        raise PredictorBoundResidualError("predictor renderer contract differs from residual custody")
    if _require_sha256(predictor_renderer_sha256, "predictor_renderer_sha256") != header["predictor_renderer_sha256"]:
        raise PredictorBoundResidualError("predictor renderer source differs from residual custody")
    if (
        list(predictor.shape) != [header["n_pairs"], header["height"], header["width"]]
        or _semantic_sha256(predictor) != header["predictor_semantic_sha256"]
    ):
        raise PredictorBoundResidualError("decoded predictor semantic stream differs from residual custody")
    try:
        target = apply_partition_seed(predictor, decoded.seed)
    except PartitionSeedError as exc:
        raise PredictorBoundResidualError("residual cannot be applied to the bound predictor") from exc
    if _semantic_sha256(target) != header["target_semantic_sha256"]:
        raise PredictorBoundResidualError("recovered target semantic stream differs from residual custody")
    return target


def packet_accounting(payload: bytes) -> dict[str, Any]:
    """Return exact counted anatomy after strict nested parse-back."""

    decoded = decode_predictor_bound_partition_residual(payload)
    _, _, header_size, residual_size = _PREFIX.unpack_from(payload)
    return {
        "schema": PACKET_SCHEMA,
        "packet_bytes": len(payload),
        "packet_sha256": _sha256(payload),
        "prefix_bytes": _PREFIX.size,
        "header_bytes": header_size,
        "nested_residual_bytes": residual_size,
        "crc_bytes": _CRC.size,
        "event_count": len(decoded.seed.events),
        "predictor_program_sha256": decoded.predictor_program_sha256,
        "predictor_semantic_sha256": decoded.predictor_semantic_sha256,
        "target_semantic_sha256": decoded.target_semantic_sha256,
        "separate_dense_target_table_section_bytes": 0,
        "pbr1_is_target_derived": True,
        "pbr1_target_derived_section_bytes": len(payload),
        "exact_target_semantic_reconstruction": True,
        "candidate_payload_allowed": False,
        "candidate_archive_blocker": "lossless predictor-conditional target-semantic-table encoding",
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "PACKET_MAGIC",
    "PACKET_SCHEMA",
    "PACKET_VERSION",
    "PredictorBoundPartitionResidual",
    "PredictorBoundResidualError",
    "apply_predictor_bound_partition_residual",
    "build_predictor_bound_partition_residual",
    "decode_predictor_bound_partition_residual",
    "packet_accounting",
]
