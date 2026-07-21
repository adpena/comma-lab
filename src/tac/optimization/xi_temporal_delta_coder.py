# SPDX-License-Identifier: MIT
"""Bit-exact xi-keyed temporal coding for the settled LBND lane description.

This module is deliberately narrower than a universal byte-delta wrapper.  It
operates on the already tracked/coherent LBND semantic object, predicts the
next quantized lane row from the previous decoded row through the declared
``(rho_z, rho_x, omega_y)`` projection of a counted composed full screw, and
range-codes the exact innovation through
``tac.shared_pmf_model``.  The innovation is lossless, so a poor predictor can
only cost bytes; it cannot change the decoded lane statistic.

The wire format is self-describing and strict: exact segment sizes and hashes,
an outer digest, canonical JSON, full shared-PMF decode/re-encode identity, and
full xi payload decode/re-encode identity are all checked.  It is a research
codec surface, not yet a standalone S4 receiver codec.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import brotli
import numpy as np

from tac.boundary_math import analytic_lane_render_band as lane_codec
from tac.boundary_math.xi_pose_coder import (
    decode_xi_payload,
    parse_xi_payload,
    quantize_xi,
    serialize_xi_payload,
)
from tac.shared_pmf_model import (
    SharedPMFConfig,
    TensorSymbolStream,
    compress_model,
    decode_rows_with_model,
    decompress_model,
    encode_rows_with_model,
    fit_shared_pmf_model,
    payload_bits_for_assignments,
)

MAGIC: Final = b"XTDL1\x00"
WIRE_VERSION: Final = 1
PREFIX: Final = struct.Struct("<6sHI")
DIGEST_BYTES: Final = 32
XI_CODER: Final = "delta_res"
XI_Q_LEVELS: Final = 4096
CONTEXT_BIN_CANDIDATES: Final = (1, 2, 4, 8)
PMF_SEED: Final = 1234
SCHEMA: Final = "xi_temporal_lane_bundle.v1"
SEMANTIC_HASH_ALGORITHM: Final = "sha256_quantized_lane_grid_metadata_and_bytes.v1"
RECEIVER_STATUS: Final = "REPOSITORY_DECODER_BIT_EXACT_STANDALONE_S4_INTEGRATION_OWED"
XI_REPRESENTATION: Final = "corrected_composed_full_screw_translation_first"
XI_COORDINATE_ORDER: Final = ["rho_x", "rho_y", "rho_z", "omega_x", "omega_y", "omega_z"]
XI_LANE_PROJECTION: Final = {"ds": "rho_z", "dy": "rho_x", "dpsi": "omega_y"}
CONTEXT_ALGORITHM: Final = "stable_rank_quantile_rho_z_from_decoded_counted_xi.v1"
ENTROPY_BACKEND: Final = "tac.shared_pmf_model_over_tac.lossless.range_coder"
SIGNED_MAP: Final = "zigzag_then_self_delimiting_uvarint"
PACK_MODES: Final = ("coherent_slot", "lateral_sort")
INT64_MIN: Final = -(1 << 63)
INT64_MAX: Final = (1 << 63) - 1

Predictor = Literal["identity", "planar3_from_composed_screw"]
PackMode = Literal["coherent_slot", "lateral_sort"]


class XiTemporalDeltaError(ValueError):
    """Malformed input, noncanonical wire bytes, or failed exact reconstruction."""


@dataclass(frozen=True)
class XiTemporalLaneArtifact:
    """Exact wire artifact plus the counted entropy-stage accounting."""

    payload: bytes
    header: dict[str, Any]
    estimated_payload_bytes: int
    model_bytes: int
    range_payload_bytes: int
    xi_payload_bytes: int
    presence_bytes: int


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise XiTemporalDeltaError(f"{name} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise XiTemporalDeltaError(f"{name} keys do not match the wire contract")


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise XiTemporalDeltaError(f"{name} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise XiTemporalDeltaError(f"{name} is below its minimum")
    return value


def _require_float(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise XiTemporalDeltaError(f"{name} must be a finite exact JSON float")
    if positive and value <= 0.0:
        raise XiTemporalDeltaError(f"{name} must be positive")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise XiTemporalDeltaError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise XiTemporalDeltaError(f"{name} must be a SHA-256 hex digest") from exc
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise XiTemporalDeltaError("header is not canonical-JSON encodable") from exc


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise XiTemporalDeltaError("uvarint cannot encode a negative integer")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _read_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    start = offset
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise XiTemporalDeltaError("innovation uvarint is truncated or overlong")
        byte = payload[offset]
        offset += 1
        if shift == 63 and byte & 0xFE:
            raise XiTemporalDeltaError("innovation uvarint exceeds uint64")
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if payload[start:offset] != _uvarint(value):
                raise XiTemporalDeltaError("innovation uvarint is not canonical")
            return value, offset
        shift += 7


def _zigzag_encode(value: int) -> int:
    if not -(1 << 63) <= value < (1 << 63):
        raise XiTemporalDeltaError("innovation is outside signed int64")
    return (value << 1) ^ (value >> 63)


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _context_ids(decoded_xi: np.ndarray, bins: int) -> np.ndarray:
    """Stable rho-z rank bins, derived entirely from the counted decoded xi.

    All candidate bin counts are measured and the smallest shared-PMF model
    plus range payload is selected. Thus the context count is not guessed.
    """

    xi = np.asarray(decoded_xi, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] != 6 or bins < 1 or bins > len(xi):
        raise XiTemporalDeltaError("invalid decoded xi or context-bin count")
    order = np.argsort(xi[:, 2], kind="stable")
    contexts = np.empty(len(xi), dtype=np.int64)
    contexts[order] = np.arange(len(xi), dtype=np.int64) * bins // len(xi)
    return contexts


def _innovation_rows(
    innovation: np.ndarray,
    contexts: np.ndarray,
) -> tuple[list[int], list[TensorSymbolStream]]:
    matrix = np.asarray(innovation, dtype=np.int64)
    if matrix.ndim != 2 or contexts.shape != (matrix.shape[0],):
        raise XiTemporalDeltaError("innovation/context shape mismatch")
    active = sorted(int(value) for value in np.unique(contexts).tolist())
    rows: list[TensorSymbolStream] = []
    for context in active:
        encoded = bytearray()
        for row_index in np.flatnonzero(contexts == context).tolist():
            for value in matrix[row_index].tolist():
                encoded.extend(_uvarint(_zigzag_encode(int(value))))
        if not encoded:
            raise XiTemporalDeltaError("active innovation context encoded no symbols")
        symbols = np.frombuffer(bytes(encoded), dtype=np.uint8).astype(np.int64)
        rows.append(
            TensorSymbolStream(
                name=f"xi_context_{context}",
                symbols=symbols,
                shape=(len(symbols),),
            )
        )
    return active, rows


def _predictor_innovation(
    q_lane: np.ndarray,
    presence: np.ndarray,
    steps_full: np.ndarray,
    decoded_xi: np.ndarray,
    slots: int,
    predictor: Predictor,
) -> np.ndarray:
    if predictor == "identity":
        innovation = np.asarray(q_lane, dtype=np.int64).copy()
        if len(innovation) > 1:
            exact_delta = q_lane[1:].astype(object) - q_lane[:-1].astype(object)
            if np.any(exact_delta < INT64_MIN) or np.any(exact_delta > INT64_MAX):
                raise XiTemporalDeltaError("identity innovation exceeds signed int64")
            innovation[1:] = np.asarray(exact_delta, dtype=np.int64)
        return innovation
    if predictor != "planar3_from_composed_screw":
        raise XiTemporalDeltaError(f"unknown predictor {predictor!r}")
    # Lane coordinates use forward rho_z, lateral rho_x, and yaw omega_y.
    # Quantize through the exact LBND3 closed-loop seam before prediction.
    _qds, _qdy, _qdpsi, ds, dy, dpsi = lane_codec._quantize_ego(decoded_xi[:, 2], decoded_xi[:, 0], decoded_xi[:, 4])
    with np.errstate(invalid="ignore", over="ignore"):
        innovation = lane_codec._predictive_encode(q_lane, presence, steps_full, ds, dy, dpsi, slots)
    inferred_predictor = q_lane.astype(object) - innovation.astype(object)
    if np.any(inferred_predictor < INT64_MIN) or np.any(inferred_predictor > INT64_MAX):
        raise XiTemporalDeltaError("planar predictor innovation exceeds signed int64")
    return innovation


def _predictor_reconstruct(
    innovation: np.ndarray,
    presence: np.ndarray,
    steps_full: np.ndarray,
    decoded_xi: np.ndarray,
    slots: int,
    predictor: Predictor,
) -> np.ndarray:
    if predictor == "identity":
        reconstructed = np.asarray(innovation, dtype=np.int64).copy()
        for row_index in range(1, len(reconstructed)):
            exact_row = reconstructed[row_index - 1].astype(object) + innovation[row_index].astype(object)
            if np.any(exact_row < INT64_MIN) or np.any(exact_row > INT64_MAX):
                raise XiTemporalDeltaError("identity reconstruction exceeds signed int64")
            reconstructed[row_index] = np.asarray(exact_row, dtype=np.int64)
        return reconstructed
    if predictor != "planar3_from_composed_screw":
        raise XiTemporalDeltaError(f"unknown predictor {predictor!r}")
    _qds, _qdy, _qdpsi, ds, dy, dpsi = lane_codec._quantize_ego(decoded_xi[:, 2], decoded_xi[:, 0], decoded_xi[:, 4])
    with np.errstate(invalid="ignore", over="ignore"):
        reconstructed = lane_codec._predictive_decode(innovation, presence, steps_full, ds, dy, dpsi, slots)
    inferred_predictor = reconstructed.astype(object) - innovation.astype(object)
    if np.any(inferred_predictor < INT64_MIN) or np.any(inferred_predictor > INT64_MAX):
        raise XiTemporalDeltaError("planar predictor reconstruction exceeds signed int64")
    return reconstructed


def _render_header(config: Any) -> dict[str, Any]:
    return {
        "softness": float(config.softness),
        "dash_gate": bool(config.dash_gate),
        "dash_forward_max_m": float(config.dash_forward_max_m),
        "v_h": float(config.v_h),
        "cx": None if config.cx is None else float(config.cx),
        "weight": float(config.weight),
        "lane_cls": int(config.lane_cls),
        "lane_rgb_mode": str(config.lane_rgb_mode),
        "u_mask": (
            {
                "source": "witness_margin",
                "tau": float(config.u_mask_tau),
                "eps": float(config.u_mask_eps),
            }
            if config.u_mask_enabled
            else None
        ),
        "geom": {
            "cam_h": float(lane_codec._CAM_H),
            "fx": float(lane_codec._FX),
            "fy": float(lane_codec._FY),
            "seg_h": int(lane_codec._SEG_H),
            "seg_w": int(lane_codec._SEG_W),
        },
    }


def _pack_lane_grid(
    pairs_lines: Sequence[Sequence[Any]],
    base_steps: np.ndarray,
    f_near: float,
    pack_mode: PackMode,
) -> tuple[np.ndarray, np.ndarray]:
    lines = [list(row) for row in pairs_lines]
    if pack_mode == "coherent_slot":
        # Lazy import avoids the analytic-lane <-> tracking import cycle.
        from tac.boundary_math.lane_track_and_smooth import coherent_slot_pack

        packed = coherent_slot_pack(lines, f_near=f_near)
        matrix, presence, slots = packed.M, packed.presence, packed.K
    elif pack_mode == "lateral_sort":
        matrix, presence, slots = lane_codec._pack_pairs_to_matrix(lines, f_near=f_near)
    else:
        raise XiTemporalDeltaError(f"unknown lane pack mode {pack_mode!r}")
    steps = np.asarray(base_steps, dtype=np.float64)
    steps_full = np.tile(steps, slots) if slots else np.zeros(0, dtype=np.float64)
    return lane_codec._quantize_matrix(matrix, steps_full), np.asarray(presence, dtype=bool)


def _validate_quantized_grid(
    q_lane: np.ndarray,
    presence: np.ndarray,
    base_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    q = np.asarray(q_lane)
    if q.dtype.kind not in "iu" or q.ndim != 2:
        raise XiTemporalDeltaError("q_lane must be a two-dimensional integer grid")
    if q.dtype.kind == "u" and q.size and int(q.max()) >= 1 << 63:
        raise XiTemporalDeltaError("q_lane exceeds signed int64")
    q = np.asarray(q, dtype=np.int64)
    p = np.asarray(presence)
    if p.dtype != np.bool_ or p.ndim != 2 or p.shape[0] != q.shape[0]:
        raise XiTemporalDeltaError("presence must be a bool grid covering q_lane pairs")
    slots = int(p.shape[1])
    if q.shape[1] != slots * lane_codec._RD_D_SLOT:
        raise XiTemporalDeltaError("q_lane dimensions do not match the presence slots")
    steps = np.asarray(base_steps, dtype=np.float64)
    if steps.shape != (lane_codec._RD_D_SLOT,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise XiTemporalDeltaError("base_steps must be the finite positive 11-D settled LBND grid")
    return q, p, steps, slots


def semantic_quantized_lane_sha256(
    q_lane: np.ndarray,
    presence: np.ndarray,
    base_steps: np.ndarray,
    f_near: float,
    config: Any,
    *,
    pack_mode: PackMode = "coherent_slot",
) -> str:
    """Hash the exact slot-labelled lattice and every receiver semantic scalar."""

    q, p, steps, slots = _validate_quantized_grid(q_lane, presence, base_steps)
    if pack_mode not in PACK_MODES or not math.isfinite(float(f_near)):
        raise XiTemporalDeltaError("invalid semantic grid metadata")
    metadata = {
        "algorithm": SEMANTIC_HASH_ALGORITHM,
        "pair_count": int(q.shape[0]),
        "slot_count": slots,
        "slot_dims": int(lane_codec._RD_D_SLOT),
        "base_steps": [float(value) for value in steps.tolist()],
        "f_near": float(f_near),
        "pack_mode": pack_mode,
        "render": _render_header(config),
    }
    digest = hashlib.sha256(_canonical_json(metadata))
    digest.update(np.ascontiguousarray(q, dtype="<i8").tobytes())
    digest.update(np.packbits(p.reshape(-1), bitorder="big").tobytes())
    return digest.hexdigest()


def quantized_lane_grid_from_lbnd2(blob: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract the exact slot-labelled ``(Q, presence)`` lattice from LBND2 bytes.

    Unlike a decode-to-``LaneLine`` followed by re-packing, this preserves the
    correspondence slot identity that is itself part of the settled description.
    """

    raw = bytes(blob)
    magic = lane_codec.LANE_BAND_RD_MAGIC
    if not raw.startswith(magic) or len(raw) < len(magic) + 8:
        raise XiTemporalDeltaError("source lane payload is not complete LBND2")
    cursor = len(magic)
    try:
        (header_size,) = struct.unpack_from("<I", raw, cursor)
        cursor += 4
        header_end = cursor + header_size
        if header_size <= 0 or header_end + 4 > len(raw):
            raise XiTemporalDeltaError("LBND2 header is truncated")
        encoded_header = raw[cursor:header_end]
        header = json.loads(encoded_header.decode("utf-8"))
        cursor = header_end
        if _canonical_json(header) != encoded_header:
            raise XiTemporalDeltaError("LBND2 header is not canonical")
        rd = _require_mapping(header.get("rd"), "LBND2 rd")
        pair_count = _require_int(rd.get("n_pairs"), "LBND2 n_pairs", minimum=1)
        slots = _require_int(rd.get("K"), "LBND2 K", minimum=0)
        slot_dims = _require_int(rd.get("d_slot"), "LBND2 d_slot", minimum=1)
        if slot_dims != lane_codec._RD_D_SLOT:
            raise XiTemporalDeltaError("LBND2 slot dimensions drifted")
        steps = np.asarray(rd.get("base_steps"), dtype=np.float64)
        if steps.shape != (slot_dims,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
            raise XiTemporalDeltaError("LBND2 grid is invalid")
        (presence_size,) = struct.unpack_from("<I", raw, cursor)
        cursor += 4
        expected_presence = (pair_count * slots + 7) // 8
        if presence_size != expected_presence or cursor + presence_size > len(raw):
            raise XiTemporalDeltaError("LBND2 presence bitmap length mismatch")
        presence_bytes = raw[cursor : cursor + presence_size]
        cursor += presence_size
        unpacked = np.unpackbits(np.frombuffer(presence_bytes, dtype=np.uint8), bitorder="big")
        if np.any(unpacked[pair_count * slots :]):
            raise XiTemporalDeltaError("LBND2 presence bitmap has nonzero padding")
        presence = unpacked[: pair_count * slots].reshape(pair_count, slots).astype(bool)
        dims = slots * slot_dims
        expected_delta_bytes = pair_count * dims * 4
        if len(raw) - cursor != expected_delta_bytes:
            raise XiTemporalDeltaError("LBND2 delta payload length mismatch")
        zz = np.frombuffer(raw[cursor:], dtype="<u4").reshape(pair_count, dims)
        delta = lane_codec._zigzag_decode(zz)
        q_lane = np.cumsum(delta, axis=0, dtype=np.int64)
    except XiTemporalDeltaError:
        raise
    except (KeyError, TypeError, ValueError, struct.error, IndexError, OverflowError) as exc:
        raise XiTemporalDeltaError("LBND2 lattice parse failed") from exc
    return q_lane, presence, header


def _assemble(
    *,
    header_without_segments: dict[str, Any],
    xi_payload: bytes,
    presence_payload: bytes,
    model_payload: bytes,
    range_payload: bytes,
) -> tuple[bytes, dict[str, Any]]:
    segments = {
        "xi": {"bytes": len(xi_payload), "sha256": _sha(xi_payload)},
        "presence": {"bytes": len(presence_payload), "sha256": _sha(presence_payload)},
        "shared_pmf_model": {"bytes": len(model_payload), "sha256": _sha(model_payload)},
        "range_payload": {"bytes": len(range_payload), "sha256": _sha(range_payload)},
    }
    header = {**header_without_segments, "segments": segments}
    encoded_header = _canonical_json(header)
    body = xi_payload + presence_payload + model_payload + range_payload
    prefix = PREFIX.pack(MAGIC, WIRE_VERSION, len(encoded_header))
    without_digest = prefix + encoded_header + body
    return without_digest + hashlib.sha256(without_digest).digest(), header


def encode_lane_xi_temporal(
    pairs_lines: Sequence[Sequence[Any]],
    config: Any,
    full_xi: np.ndarray,
    *,
    base_steps: np.ndarray,
    f_near: float,
    predictor: Predictor = "planar3_from_composed_screw",
    seed: int = PMF_SEED,
    pack_mode: PackMode = "coherent_slot",
) -> XiTemporalLaneArtifact:
    """Pack a lane description and encode its exact quantized lattice.

    ``coherent_slot`` is the default because slot identity is part of the
    settled S4 description. Measurement code that already owns source LBND2
    bytes should use :func:`encode_quantized_lane_xi_temporal` to avoid any
    decode/re-pack ambiguity.
    """

    lines = [list(row) for row in pairs_lines]
    if not lines:
        raise XiTemporalDeltaError("lines must cover a positive pair count")
    q_lane, presence = _pack_lane_grid(lines, base_steps, f_near, pack_mode)
    return encode_quantized_lane_xi_temporal(
        q_lane,
        presence,
        config,
        full_xi,
        base_steps=base_steps,
        f_near=f_near,
        predictor=predictor,
        seed=seed,
        pack_mode=pack_mode,
    )


def encode_quantized_lane_xi_temporal(
    q_lane: np.ndarray,
    presence: np.ndarray,
    config: Any,
    full_xi: np.ndarray,
    *,
    base_steps: np.ndarray,
    f_near: float,
    predictor: Predictor = "planar3_from_composed_screw",
    seed: int = PMF_SEED,
    pack_mode: PackMode = "coherent_slot",
) -> XiTemporalLaneArtifact:
    """Encode an exact slot-labelled LBND lattice with a counted full xi.

    Every context-bin candidate is encoded through the shared-PMF stack; the
    smallest complete XTDL1 wire is selected by measured bytes with
    deterministic ties.
    """

    q_lane, presence, steps, slots = _validate_quantized_grid(q_lane, presence, base_steps)
    pair_count = int(q_lane.shape[0])
    xi = np.asarray(full_xi, dtype=np.float64)
    if pair_count <= 0 or xi.shape != (pair_count, 6) or not np.all(np.isfinite(xi)):
        raise XiTemporalDeltaError("q_lane and finite full_xi must cover the same positive pair count")
    if pack_mode not in PACK_MODES or not math.isfinite(float(f_near)) or type(seed) is not int:
        raise XiTemporalDeltaError("invalid lane-grid metadata or PMF seed")
    steps_full = np.tile(steps, slots) if slots else np.zeros(0, dtype=np.float64)

    q_xi, xi_scales = quantize_xi(xi, q_levels=XI_Q_LEVELS)
    xi_payload = serialize_xi_payload(q_xi, xi_scales, coder=XI_CODER)
    parsed_q, parsed_scales = parse_xi_payload(xi_payload)
    if not (np.array_equal(parsed_q, q_xi) and np.array_equal(parsed_scales, xi_scales)):
        raise XiTemporalDeltaError("xi payload failed exact quantized-grid roundtrip")
    decoded_xi = decode_xi_payload(xi_payload)
    innovation = _predictor_innovation(q_lane, presence, steps_full, decoded_xi, slots, predictor)

    presence_payload = np.packbits(presence.reshape(-1), bitorder="big").tobytes() if presence.size else b""
    semantic_sha = semantic_quantized_lane_sha256(
        q_lane,
        presence,
        steps,
        f_near,
        config,
        pack_mode=pack_mode,
    )

    def _candidate_header(
        context_bins: int,
        active_contexts: list[int],
        context_counts: list[int],
        estimated_payload_bytes: int,
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "pair_count": pair_count,
            "slot_count": int(slots),
            "slot_dims": int(lane_codec._RD_D_SLOT),
            "predictor": predictor,
            "seed": int(seed),
            "xi": {
                "representation": XI_REPRESENTATION,
                "coordinate_order": XI_COORDINATE_ORDER,
                "coder": XI_CODER,
                "q_levels": XI_Q_LEVELS,
                "lane_projection": XI_LANE_PROJECTION,
            },
            "context": {
                "algorithm": CONTEXT_ALGORITHM,
                "candidate_bins": list(CONTEXT_BIN_CANDIDATES),
                "selected_bins": int(context_bins),
                "active_contexts": active_contexts,
                "frame_counts": context_counts,
                "selection": "minimum_complete_xtdl1_wire_bytes",
            },
            "entropy": {
                "backend": ENTROPY_BACKEND,
                "n_categories": 256,
                "n_models": min(4, len(active_contexts)),
                "estimated_range_payload_bytes": int(estimated_payload_bytes),
                "signed_map": SIGNED_MAP,
            },
            "rd": {
                "base_steps": [float(value) for value in steps.tolist()],
                "f_near": float(f_near),
                "pack_mode": pack_mode,
            },
            "render": _render_header(config),
            "semantic": {"algorithm": SEMANTIC_HASH_ALGORITHM, "grid_sha256": semantic_sha},
            "receiver_status": RECEIVER_STATUS,
        }

    candidates: list[tuple[int, int, bytes, dict[str, Any], bytes, bytes, int]] = []
    for context_bins in CONTEXT_BIN_CANDIDATES:
        if context_bins > pair_count:
            continue
        contexts = _context_ids(decoded_xi, context_bins)
        active, rows = _innovation_rows(innovation, contexts)
        pmf_config = SharedPMFConfig(
            n_models=min(4, len(rows)),
            n_categories=256,
            seed=int(seed),
        )
        # Some candidate initializations transiently evaluate zero-probability
        # floating costs before the positive-frequency repair.  The fitted
        # integer tables are validated below; keep that internal search quiet.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            model = fit_shared_pmf_model(rows, pmf_config)
        model_payload = compress_model(model)
        encoded = encode_rows_with_model(rows, model)
        estimated = int(np.ceil(model.estimated_payload_bits / 8.0))
        context_counts = [int(np.count_nonzero(contexts == value)) for value in active]
        header_without_segments = _candidate_header(context_bins, active, context_counts, estimated)
        candidate_payload, candidate_header = _assemble(
            header_without_segments=header_without_segments,
            xi_payload=xi_payload,
            presence_payload=presence_payload,
            model_payload=model_payload,
            range_payload=encoded.payload_bytes,
        )
        candidates.append(
            (
                len(candidate_payload),
                context_bins,
                candidate_payload,
                candidate_header,
                model_payload,
                encoded.payload_bytes,
                estimated,
            )
        )
    if not candidates:
        raise XiTemporalDeltaError("no shared-PMF context candidate was admissible")
    (
        _candidate_wire_bytes,
        _context_bins,
        payload,
        header,
        model_payload,
        range_payload,
        estimated_payload_bytes,
    ) = min(candidates, key=lambda row: (row[0], row[1]))
    decoded_q, decoded_presence, decoded_header = decode_lane_xi_temporal_grid(payload)
    if (
        decoded_header != header
        or not np.array_equal(decoded_q, q_lane)
        or not np.array_equal(decoded_presence, presence)
    ):
        raise XiTemporalDeltaError("encoded lane artifact failed exact semantic self-check")
    return XiTemporalLaneArtifact(
        payload=payload,
        header=header,
        estimated_payload_bytes=int(estimated_payload_bytes),
        model_bytes=len(model_payload),
        range_payload_bytes=len(range_payload),
        xi_payload_bytes=len(xi_payload),
        presence_bytes=len(presence_payload),
    )


def _validate_header_contract(header: dict[str, Any]) -> None:
    _require_exact_keys(
        header,
        {
            "schema",
            "pair_count",
            "slot_count",
            "slot_dims",
            "predictor",
            "seed",
            "xi",
            "context",
            "entropy",
            "rd",
            "render",
            "semantic",
            "receiver_status",
            "segments",
        },
        "header",
    )
    if header["schema"] != SCHEMA or header["receiver_status"] != RECEIVER_STATUS:
        raise XiTemporalDeltaError("xi-temporal schema or receiver status mismatch")
    _require_int(header["pair_count"], "pair_count", minimum=1)
    _require_int(header["slot_count"], "slot_count", minimum=0)
    if _require_int(header["slot_dims"], "slot_dims", minimum=1) != lane_codec._RD_D_SLOT:
        raise XiTemporalDeltaError("slot_dims drifted from LBND")
    _require_int(header["seed"], "seed")
    if header["predictor"] not in ("identity", "planar3_from_composed_screw"):
        raise XiTemporalDeltaError("predictor declaration is invalid")

    xi = _require_mapping(header["xi"], "xi")
    _require_exact_keys(xi, {"representation", "coordinate_order", "coder", "q_levels", "lane_projection"}, "xi")
    if (
        xi["representation"] != XI_REPRESENTATION
        or xi["coordinate_order"] != XI_COORDINATE_ORDER
        or not all(isinstance(value, str) for value in xi["coordinate_order"])
        or xi["coder"] != XI_CODER
        or _require_int(xi["q_levels"], "xi.q_levels", minimum=1) != XI_Q_LEVELS
        or xi["lane_projection"] != XI_LANE_PROJECTION
    ):
        raise XiTemporalDeltaError("xi decoder contract mismatch")
    projection = _require_mapping(xi["lane_projection"], "xi.lane_projection")
    _require_exact_keys(projection, set(XI_LANE_PROJECTION), "xi.lane_projection")

    context = _require_mapping(header["context"], "context")
    _require_exact_keys(
        context,
        {"algorithm", "candidate_bins", "selected_bins", "active_contexts", "frame_counts", "selection"},
        "context",
    )
    candidates = context["candidate_bins"]
    if (
        context["algorithm"] != CONTEXT_ALGORITHM
        or type(candidates) is not list
        or len(candidates) != len(CONTEXT_BIN_CANDIDATES)
        or any(type(value) is not int for value in candidates)
        or candidates != list(CONTEXT_BIN_CANDIDATES)
        or context["selection"] != "minimum_complete_xtdl1_wire_bytes"
    ):
        raise XiTemporalDeltaError("context decoder contract mismatch")
    selected = _require_int(context["selected_bins"], "context.selected_bins", minimum=1)
    if selected not in CONTEXT_BIN_CANDIDATES:
        raise XiTemporalDeltaError("selected context count is invalid")
    for field in ("active_contexts", "frame_counts"):
        values = context[field]
        if type(values) is not list or any(type(value) is not int for value in values):
            raise XiTemporalDeltaError(f"context.{field} must contain exact JSON integers")
    if len(context["active_contexts"]) != len(context["frame_counts"]):
        raise XiTemporalDeltaError("context row accounting lengths differ")

    entropy = _require_mapping(header["entropy"], "entropy")
    _require_exact_keys(
        entropy,
        {"backend", "n_categories", "n_models", "estimated_range_payload_bytes", "signed_map"},
        "entropy",
    )
    if entropy["backend"] != ENTROPY_BACKEND or entropy["signed_map"] != SIGNED_MAP:
        raise XiTemporalDeltaError("entropy decoder contract mismatch")
    if _require_int(entropy["n_categories"], "entropy.n_categories", minimum=1) != 256:
        raise XiTemporalDeltaError("entropy category count drifted")
    _require_int(entropy["n_models"], "entropy.n_models", minimum=1)
    _require_int(entropy["estimated_range_payload_bytes"], "entropy.estimated_range_payload_bytes", minimum=0)

    rd = _require_mapping(header["rd"], "rd")
    _require_exact_keys(rd, {"base_steps", "f_near", "pack_mode"}, "rd")
    if type(rd["base_steps"]) is not list or len(rd["base_steps"]) != lane_codec._RD_D_SLOT:
        raise XiTemporalDeltaError("rd.base_steps has the wrong shape")
    for index, value in enumerate(rd["base_steps"]):
        _require_float(value, f"rd.base_steps[{index}]", positive=True)
    _require_float(rd["f_near"], "rd.f_near")
    if rd["pack_mode"] not in PACK_MODES:
        raise XiTemporalDeltaError("rd.pack_mode is invalid")

    render = _require_mapping(header["render"], "render")
    _require_exact_keys(
        render,
        {
            "softness",
            "dash_gate",
            "dash_forward_max_m",
            "v_h",
            "cx",
            "weight",
            "lane_cls",
            "lane_rgb_mode",
            "u_mask",
            "geom",
        },
        "render",
    )
    _require_float(render["softness"], "render.softness", positive=True)
    if type(render["dash_gate"]) is not bool:
        raise XiTemporalDeltaError("render.dash_gate must be a JSON boolean")
    _require_float(render["dash_forward_max_m"], "render.dash_forward_max_m")
    _require_float(render["v_h"], "render.v_h")
    if render["cx"] is not None:
        _require_float(render["cx"], "render.cx")
    _require_float(render["weight"], "render.weight")
    _require_int(render["lane_cls"], "render.lane_cls", minimum=0)
    if not isinstance(render["lane_rgb_mode"], str) or not render["lane_rgb_mode"]:
        raise XiTemporalDeltaError("render.lane_rgb_mode must be a nonempty string")
    if render["u_mask"] is not None:
        u_mask = _require_mapping(render["u_mask"], "render.u_mask")
        _require_exact_keys(u_mask, {"source", "tau", "eps"}, "render.u_mask")
        if u_mask["source"] != "witness_margin":
            raise XiTemporalDeltaError("render.u_mask source is invalid")
        _require_float(u_mask["tau"], "render.u_mask.tau")
        _require_float(u_mask["eps"], "render.u_mask.eps", positive=True)
    geom = _require_mapping(render["geom"], "render.geom")
    _require_exact_keys(geom, {"cam_h", "fx", "fy", "seg_h", "seg_w"}, "render.geom")
    for field, expected in (("cam_h", lane_codec._CAM_H), ("fx", lane_codec._FX), ("fy", lane_codec._FY)):
        if _require_float(geom[field], f"render.geom.{field}") != float(expected):
            raise XiTemporalDeltaError("render geometry drifted")
    for field, expected in (("seg_h", lane_codec._SEG_H), ("seg_w", lane_codec._SEG_W)):
        if _require_int(geom[field], f"render.geom.{field}", minimum=1) != int(expected):
            raise XiTemporalDeltaError("render geometry drifted")

    semantic = _require_mapping(header["semantic"], "semantic")
    _require_exact_keys(semantic, {"algorithm", "grid_sha256"}, "semantic")
    if semantic["algorithm"] != SEMANTIC_HASH_ALGORITHM:
        raise XiTemporalDeltaError("semantic hash algorithm mismatch")
    _require_sha256(semantic["grid_sha256"], "semantic.grid_sha256")

    segments = _require_mapping(header["segments"], "segments")
    expected_segment_names = {"xi", "presence", "shared_pmf_model", "range_payload"}
    _require_exact_keys(segments, expected_segment_names, "segments")
    for name in expected_segment_names:
        row = _require_mapping(segments[name], f"segments.{name}")
        _require_exact_keys(row, {"bytes", "sha256"}, f"segments.{name}")
        _require_int(row["bytes"], f"segments.{name}.bytes", minimum=0)
        _require_sha256(row["sha256"], f"segments.{name}.sha256")


def _split_segments(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    raw = bytes(payload)
    if len(raw) < PREFIX.size + DIGEST_BYTES:
        raise XiTemporalDeltaError("xi-temporal lane bundle is truncated")
    if hashlib.sha256(raw[:-DIGEST_BYTES]).digest() != raw[-DIGEST_BYTES:]:
        raise XiTemporalDeltaError("xi-temporal lane bundle outer digest mismatch")
    magic, version, header_size = PREFIX.unpack_from(raw)
    if magic != MAGIC or version != WIRE_VERSION or header_size <= 0:
        raise XiTemporalDeltaError("xi-temporal lane bundle header mismatch")
    header_end = PREFIX.size + header_size
    if header_end > len(raw) - DIGEST_BYTES:
        raise XiTemporalDeltaError("xi-temporal lane header is truncated")
    try:
        header = json.loads(raw[PREFIX.size : header_end].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise XiTemporalDeltaError("xi-temporal lane header JSON is malformed") from exc
    if _canonical_json(header) != raw[PREFIX.size : header_end]:
        raise XiTemporalDeltaError("xi-temporal lane header is not canonical")
    _validate_header_contract(header)

    cursor = header_end
    segments: dict[str, bytes] = {}
    for name in ("xi", "presence", "shared_pmf_model", "range_payload"):
        row = header.get("segments", {}).get(name)
        if not isinstance(row, dict) or not isinstance(row.get("bytes"), int):
            raise XiTemporalDeltaError(f"missing segment accounting for {name}")
        size = int(row["bytes"])
        end = cursor + size
        if size < 0 or end > len(raw) - DIGEST_BYTES:
            raise XiTemporalDeltaError(f"segment {name} length is invalid")
        segment = raw[cursor:end]
        cursor = end
        if _sha(segment) != row.get("sha256"):
            raise XiTemporalDeltaError(f"segment {name} digest mismatch")
        segments[name] = segment
    if cursor != len(raw) - DIGEST_BYTES:
        raise XiTemporalDeltaError("xi-temporal lane bundle has trailing bytes")
    return header, segments


def _decode_lane_xi_temporal_grid(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Implementation of the strict exact-grid decoder."""

    header, segments = _split_segments(payload)
    pair_count = header["pair_count"]
    slots = header["slot_count"]
    slot_dims = header["slot_dims"]
    if pair_count <= 0 or slots < 0 or slot_dims != lane_codec._RD_D_SLOT:
        raise XiTemporalDeltaError("invalid lane dimensions")
    predictor = str(header["predictor"])
    if predictor not in ("identity", "planar3_from_composed_screw"):
        raise XiTemporalDeltaError("invalid predictor declaration")
    q_xi, xi_scales = parse_xi_payload(segments["xi"])
    if q_xi.shape != (pair_count, 6):
        raise XiTemporalDeltaError("xi payload dimensions do not match lane bundle")
    if serialize_xi_payload(q_xi, xi_scales, coder=header["xi"]["coder"]) != segments["xi"]:
        raise XiTemporalDeltaError("xi payload is not byte-canonical")
    decoded_xi = decode_xi_payload(segments["xi"])

    presence_count = pair_count * slots
    expected_presence_bytes = (presence_count + 7) // 8
    if len(segments["presence"]) != expected_presence_bytes:
        raise XiTemporalDeltaError("presence bitmap length mismatch")
    if slots:
        unpacked = np.unpackbits(np.frombuffer(segments["presence"], dtype=np.uint8), bitorder="big")
        if np.any(unpacked[presence_count:]):
            raise XiTemporalDeltaError("presence bitmap has nonzero padding bits")
        presence = unpacked[:presence_count].reshape(pair_count, slots).astype(bool)
    else:
        presence = np.zeros((pair_count, 0), dtype=bool)

    context_bins = header["context"]["selected_bins"]
    contexts = _context_ids(decoded_xi, context_bins)
    active = sorted(int(value) for value in np.unique(contexts).tolist())
    if active != header["context"]["active_contexts"]:
        raise XiTemporalDeltaError("derived xi contexts do not match the header")
    if [int(np.count_nonzero(contexts == value)) for value in active] != header["context"]["frame_counts"]:
        raise XiTemporalDeltaError("xi context frame counts drifted")
    if header["entropy"]["n_models"] != min(4, len(active)):
        raise XiTemporalDeltaError("entropy model count does not match active contexts")

    model = decompress_model(segments["shared_pmf_model"])
    if (
        compress_model(model) != segments["shared_pmf_model"]
        or len(model.tensor_lengths) != len(active)
        or model.config.seed != header["seed"]
        or model.config.n_categories != header["entropy"]["n_categories"]
        or model.config.n_models != header["entropy"]["n_models"]
    ):
        raise XiTemporalDeltaError("shared-PMF model is not canonical for active contexts")
    decoded_rows = decode_rows_with_model(segments["range_payload"], model)
    canonical_rows = [
        TensorSymbolStream(
            name=f"xi_context_{context}",
            symbols=np.asarray(symbols, dtype=np.int64),
            shape=(len(symbols),),
        )
        for context, symbols in zip(active, decoded_rows, strict=True)
    ]
    if encode_rows_with_model(canonical_rows, model).payload_bytes != segments["range_payload"]:
        raise XiTemporalDeltaError("range payload failed byte-identical decode/re-encode")
    counts = np.stack(
        [
            np.bincount(row.symbols.astype(np.int64), minlength=model.config.n_categories).astype(np.float64)
            for row in canonical_rows
        ],
        axis=0,
    )
    estimated_payload_bytes = math.ceil(
        payload_bits_for_assignments(
            counts,
            model.frequencies,
            np.asarray(model.assignments, dtype=np.int64),
            model.config.total_frequency,
        )
        / 8.0
    )
    if header["entropy"]["estimated_range_payload_bytes"] != estimated_payload_bytes:
        raise XiTemporalDeltaError("range payload estimate does not match decoded symbols and model")

    dims = slots * slot_dims
    innovation = np.zeros((pair_count, dims), dtype=np.int64)
    for context, symbols in zip(active, decoded_rows, strict=True):
        row = np.asarray(symbols, dtype=np.uint8).tobytes()
        cursor = 0
        for pair in np.flatnonzero(contexts == context).tolist():
            for dim in range(dims):
                value, cursor = _read_uvarint(row, cursor)
                innovation[pair, dim] = _zigzag_decode(value)
        if cursor != len(row):
            raise XiTemporalDeltaError("decoded innovation context has trailing symbols")

    steps = np.asarray(header["rd"]["base_steps"], dtype=np.float64)
    if steps.shape != (slot_dims,) or np.any(steps <= 0):
        raise XiTemporalDeltaError("invalid decoded LBND grid")
    steps_full = np.tile(steps, slots) if slots else np.zeros(0, dtype=np.float64)
    q_lane = _predictor_reconstruct(
        innovation,
        presence,
        steps_full,
        decoded_xi,
        slots,
        predictor,  # type: ignore[arg-type]
    )
    config = lane_codec.render_config_from_header(header["render"])
    actual_semantic = semantic_quantized_lane_sha256(
        q_lane,
        presence,
        steps,
        header["rd"]["f_near"],
        config,
        pack_mode=header["rd"]["pack_mode"],
    )
    if actual_semantic != header["semantic"]["grid_sha256"]:
        raise XiTemporalDeltaError("decoded quantized Lane grid semantic digest mismatch")
    return q_lane, presence, header


def decode_lane_xi_temporal_grid(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode and validate the exact slot-labelled ``(Q, presence)`` lattice."""

    try:
        return _decode_lane_xi_temporal_grid(payload)
    except XiTemporalDeltaError:
        raise
    except (
        brotli.error,
        zlib.error,
        KeyError,
        TypeError,
        ValueError,
        struct.error,
        IndexError,
        OverflowError,
        EOFError,
    ) as exc:
        raise XiTemporalDeltaError("xi-temporal lane bundle decode failed") from exc


def decode_lane_xi_temporal(payload: bytes) -> tuple[list[list[Any]], dict[str, Any]]:
    """Strict inverse returning dequantized ``LaneLine`` objects for rendering."""

    q_lane, presence, header = decode_lane_xi_temporal_grid(payload)
    slots = header["slot_count"]
    steps = np.asarray(header["rd"]["base_steps"], dtype=np.float64)
    steps_full = np.tile(steps, slots) if slots else np.zeros(0, dtype=np.float64)
    matrix = q_lane.astype(np.float64) * steps_full
    lines = lane_codec._unpack_matrix_to_pairs(matrix, presence, slots)
    return lines, header


def semantic_lane_sha256(
    pairs_lines: Sequence[Sequence[Any]],
    base_steps: np.ndarray,
    f_near: float,
    config: Any | None = None,
    *,
    pack_mode: PackMode = "coherent_slot",
) -> str:
    """Hash a packed LBND-grid statistic without losing coherent slot identity."""

    lines = [list(row) for row in pairs_lines]
    q_lane, presence = _pack_lane_grid(lines, base_steps, f_near, pack_mode)
    return semantic_quantized_lane_sha256(
        q_lane,
        presence,
        base_steps,
        f_near,
        lane_codec.LaneBandRenderConfig() if config is None else config,
        pack_mode=pack_mode,
    )


def inspect_lane_xi_temporal(payload: bytes) -> dict[str, Any]:
    """Return the validated canonical header without exposing unchecked bytes."""

    header, _segments = _split_segments(payload)
    decode_lane_xi_temporal(payload)
    return header


__all__ = [
    "CONTEXT_BIN_CANDIDATES",
    "PMF_SEED",
    "WIRE_VERSION",
    "XiTemporalDeltaError",
    "XiTemporalLaneArtifact",
    "decode_lane_xi_temporal",
    "decode_lane_xi_temporal_grid",
    "encode_lane_xi_temporal",
    "encode_quantized_lane_xi_temporal",
    "inspect_lane_xi_temporal",
    "quantized_lane_grid_from_lbnd2",
    "semantic_lane_sha256",
    "semantic_quantized_lane_sha256",
]
