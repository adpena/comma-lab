# SPDX-License-Identifier: MIT
"""Z5 predictive-coding world-model archive grammar — Z5PCWM1 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (≤200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

Z5PCWM1 extends Z4CR1 by adding a predictor section and replacing the
per-pair-latent section with (latent_init + per-pair-residuals + ego_motion).

Grammar:

::

    MAGIC(4)             b"Z5WM"
    VERSION(1)           u8       schema version (currently 1)
    LATENT_DIM(2)        u16      cfg.latent_dim (e.g. 24)
    EGO_MOTION_DIM(2)    u16      cfg.predictor_ego_motion_dim (e.g. 8)
    NUM_PAIRS(2)         u16      cfg.num_pairs (e.g. 600)
    ENCODER_BLOB_LEN(4)  u32      brotli-compressed encoder state_dict bytes len
    DECODER_BLOB_LEN(4)  u32      brotli-compressed decoder state_dict bytes len
    PREDICTOR_BLOB_LEN(4)u32      brotli-compressed predictor state_dict bytes len
    LATENT_INIT_LEN(4)   u32      int8 z_0 bytes (= latent_dim)
    RESIDUALS_LEN(4)     u32      int8 residuals bytes (= num_pairs * latent_dim)
    EGO_MOTION_LEN(4)    u32      int8 ego_motion bytes (= num_pairs * ego_motion_dim)
    META_BLOB_LEN(4)     u32      sorted-keys JSON utf-8 bytes len
    ENCODER_BLOB         ...      brotli(quality=9) encoder state_dict fp16
    DECODER_BLOB         ...      brotli(quality=9) decoder state_dict fp16
    PREDICTOR_BLOB       ...      brotli(quality=9) predictor state_dict fp16
    LATENT_INIT_BLOB     ...      int8 z_0 row-major (latent_dim,)
    RESIDUALS_BLOB       ...      int8 residuals row-major (num_pairs, latent_dim)
    EGO_MOTION_BLOB      ...      int8 ego_motion row-major (num_pairs, ego_motion_dim)
    META_BLOB            ...      sorted-keys JSON with predictive_coding_world_model_meta tag

Round-trip contract:

    bytes → parse_archive → (encoder_sd, decoder_sd, predictor_sd, latent_init,
                              residuals, ego_motion, meta)
    inverse → pack_archive → bytes

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli;
  fp16 state_dict cast on CPU)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.substrates._shared.numpy_portable_inflate import as_numpy_array

Z5PCWM1_MAGIC: bytes = b"Z5WM"
"""Z5 predictive-coding world-model variant 1 archive magic."""

Z5PCWM1_SCHEMA_VERSION: int = 1
"""Schema version byte."""

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + EGO_MOTION_DIM(2)
#                + NUM_PAIRS(2) + 7 × u32 section lengths
# = 4+1+2+2+2+7*4 = 39 bytes
Z5PCWM1_HEADER_FMT: str = "<4sBHHHIIIIIII"
Z5PCWM1_HEADER_SIZE: int = struct.calcsize(Z5PCWM1_HEADER_FMT)
assert Z5PCWM1_HEADER_SIZE == 39, f"header size invariant: expected 39, got {Z5PCWM1_HEADER_SIZE}"

_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PredictiveCodingArchive:
    """Parsed Z5PCWM1 archive — the inflate-time data contract."""

    encoder_state_dict: dict[str, Any]
    decoder_state_dict: dict[str, Any]
    predictor_state_dict: dict[str, Any]
    latent_init: Any  # (latent_dim,) float32 dequantized
    residuals: Any    # (num_pairs, latent_dim) float32 dequantized
    ego_motion: Any   # (num_pairs, ego_motion_dim) float32 dequantized
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, Any]) -> bytes:
    """Serialize state_dict deterministically (matches Z4 pattern)."""
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = np.ascontiguousarray(as_numpy_array(sd[key]), dtype=np.float16)
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} too long for u16")
        shape = tuple(int(s) for s in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(header_fmt, len(key_bytes), key_bytes, len(shape), *shape)
        parts.append(header)
        parts.append(tensor.tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_numpy_state_dict(blob: bytes) -> dict[str, np.ndarray]:
    """Torch-free deserialize of the Z5PCWM1 length-prefixed fp16 state_dict blob.

    Reads the EXACT same byte format ``_serialize_state_dict`` produces, but
    reconstructs each tensor as an ``np.float32`` ndarray with NO torch import.
    The shipped numpy-portable inflate runtime calls this (8th MLX-first
    standing directive: ``torch`` is FORBIDDEN at decode time).
    """
    raw = brotli.decompress(blob)
    sd: dict[str, np.ndarray] = {}
    pos = 0
    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + ndim * 4 > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + ndim * 4])
        pos += ndim * 4
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor_bytes = numel * 2  # fp16
        if pos + tensor_bytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        arr = np.frombuffer(raw[pos : pos + tensor_bytes], dtype=np.float16).copy()
        sd[key] = arr.reshape(shape).astype(np.float32)
        pos += tensor_bytes
    if pos != len(raw):
        raise ValueError(f"state_dict blob has trailing bytes (pos={pos} len={len(raw)})")
    return sd


def _deserialize_state_dict(blob: bytes) -> dict[str, Any]:
    import importlib

    torch = importlib.import_module("torch")
    raw = brotli.decompress(blob)
    sd: dict[str, Any] = {}
    pos = 0

    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + ndim * 4 > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + ndim * 4])
        pos += ndim * 4
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor_bytes = numel * 2
        if pos + tensor_bytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        arr = np.frombuffer(
            raw[pos : pos + tensor_bytes], dtype=np.float16
        ).reshape(shape).astype(np.float16, copy=True)
        sd[key] = torch.from_numpy(arr)
        pos += tensor_bytes
    if pos != len(raw):
        raise ValueError(f"state_dict blob has trailing bytes (pos={pos} len={len(raw)})")
    return sd


def _quantize_to_int8(t: Any) -> tuple[np.ndarray, float, float]:
    """Quantize a float tensor to int8 via min/max range.

    Per Catalog #161 the degenerate-range case (hi <= lo) clamps to -127
    so the decoded value is exactly ``lo``.
    """
    arr = as_numpy_array(t)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"tensor must be float; got {arr.dtype}")
    f = np.ascontiguousarray(arr, dtype=np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.full(f.shape, -127, dtype=np.int8), 1.0, lo
    scale = (hi - lo) / 254.0
    q_unsigned = np.clip(np.round((f - lo) / scale), 0.0, 254.0)
    q = (q_unsigned - 127.0).astype(np.int8)
    return q, scale, lo


def _dequantize_from_int8(
    q: np.ndarray, scale: float, zero_point: float
) -> np.ndarray:
    q_unsigned = q.astype(np.float32) + 127.0
    return q_unsigned * float(scale) + float(zero_point)


def _make_predictive_coding_world_model_meta_block(
    base_meta: dict[str, object],
    *,
    lambda_residual_entropy: float,
    predictor_num_layers: int,
    identity_predictor: bool,
) -> dict[str, object]:
    """Inject the predictive-coding-world-model provenance tag."""
    meta = dict(base_meta)
    meta["predictive_coding_world_model_meta"] = {
        "lambda_residual_entropy": float(lambda_residual_entropy),
        "predictor_num_layers": int(predictor_num_layers),
        "identity_predictor": bool(identity_predictor),
        "literature_anchor": "Rao-Ballard1999",
        "staircase_step": 3,
        "prediction_band_verdict": {
            "schema": "substrate_archive_prediction_band_verdict_v1",
            "planning_band": [0.155, 0.180],
            "axis": "mixed",
            "evidence_semantics": "planning_prior_only",
            "valid_for_rank_reward": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": [
                "requires_paired_cpu_cuda_axis_plan_before_promotion",
                "requires_empirical_anchor_before_rank_reward",
            ],
        },
    }
    return meta


def pack_archive(
    encoder_state_dict: dict[str, Any],
    decoder_state_dict: dict[str, Any],
    predictor_state_dict: dict[str, Any],
    latent_init: Any,
    residuals: Any,
    ego_motion: Any,
    meta: dict[str, object],
    *,
    schema_version: int = Z5PCWM1_SCHEMA_VERSION,
    lambda_residual_entropy: float = 1.0,
    predictor_num_layers: int = 2,
    identity_predictor: bool = False,
) -> bytes:
    """Serialize trained substrate into Z5PCWM1 0.bin bytes."""
    if schema_version != Z5PCWM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    latent_init_arr = as_numpy_array(latent_init)
    residuals_arr = as_numpy_array(residuals)
    ego_motion_arr = as_numpy_array(ego_motion)
    if latent_init_arr.ndim != 1:
        raise ValueError(
            f"latent_init must be 1-D (latent_dim,); got {tuple(latent_init_arr.shape)}"
        )
    if residuals_arr.ndim != 2:
        raise ValueError(f"residuals must be 2-D; got {tuple(residuals_arr.shape)}")
    if ego_motion_arr.ndim != 2:
        raise ValueError(f"ego_motion must be 2-D; got {tuple(ego_motion_arr.shape)}")

    latent_dim = int(latent_init_arr.shape[0])
    num_pairs = int(residuals_arr.shape[0])
    ego_motion_dim = int(ego_motion_arr.shape[1])
    if int(residuals_arr.shape[1]) != latent_dim:
        raise ValueError(
            f"residuals second dim {residuals_arr.shape[1]} != latent_dim {latent_dim}"
        )
    if int(ego_motion_arr.shape[0]) != num_pairs:
        raise ValueError(
            f"ego_motion first dim {ego_motion_arr.shape[0]} != num_pairs {num_pairs}"
        )
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if ego_motion_dim <= 0 or ego_motion_dim > 0xFFFF:
        raise ValueError(f"ego_motion_dim {ego_motion_dim} out of u16 range")

    q_latent_init, scale_li, zp_li = _quantize_to_int8(latent_init_arr)
    q_residuals, scale_r, zp_r = _quantize_to_int8(residuals_arr)
    q_ego, scale_e, zp_e = _quantize_to_int8(ego_motion_arr)
    latent_init_bytes = np.ascontiguousarray(q_latent_init).tobytes()
    residuals_bytes = np.ascontiguousarray(q_residuals).tobytes()
    ego_motion_bytes = np.ascontiguousarray(q_ego).tobytes()

    encoder_blob = _serialize_state_dict(encoder_state_dict)
    decoder_blob = _serialize_state_dict(decoder_state_dict)
    predictor_blob = _serialize_state_dict(predictor_state_dict)

    meta_with_tag = _make_predictive_coding_world_model_meta_block(
        meta,
        lambda_residual_entropy=lambda_residual_entropy,
        predictor_num_layers=predictor_num_layers,
        identity_predictor=identity_predictor,
    )
    meta_with_tag["_latent_init_scale"] = float(scale_li)
    meta_with_tag["_latent_init_zp"] = float(zp_li)
    meta_with_tag["_residuals_scale"] = float(scale_r)
    meta_with_tag["_residuals_zp"] = float(zp_r)
    meta_with_tag["_ego_motion_scale"] = float(scale_e)
    meta_with_tag["_ego_motion_zp"] = float(zp_e)
    meta_bytes = json.dumps(
        meta_with_tag, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z5PCWM1_HEADER_FMT,
        Z5PCWM1_MAGIC,
        schema_version,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        len(encoder_blob),
        len(decoder_blob),
        len(predictor_blob),
        len(latent_init_bytes),
        len(residuals_bytes),
        len(ego_motion_bytes),
        len(meta_bytes),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + predictor_blob
        + latent_init_bytes
        + residuals_bytes
        + ego_motion_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> PredictiveCodingArchive:
    """Parse Z5PCWM1 0.bin bytes back into all sections."""
    import importlib

    torch = importlib.import_module("torch")
    if len(blob) < Z5PCWM1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {Z5PCWM1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        encoder_len,
        decoder_len,
        predictor_len,
        latent_init_len,
        residuals_len,
        ego_motion_len,
        meta_len,
    ) = struct.unpack(Z5PCWM1_HEADER_FMT, blob[:Z5PCWM1_HEADER_SIZE])
    if magic != Z5PCWM1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {Z5PCWM1_MAGIC!r})")
    if version != Z5PCWM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    if latent_init_len != latent_dim:
        raise ValueError(
            f"latent_init_len {latent_init_len} != latent_dim {latent_dim}"
        )
    expected_residuals = num_pairs * latent_dim
    if residuals_len != expected_residuals:
        raise ValueError(
            f"residuals_len {residuals_len} != num_pairs*latent_dim = {expected_residuals}"
        )
    expected_ego = num_pairs * ego_motion_dim
    if ego_motion_len != expected_ego:
        raise ValueError(
            f"ego_motion_len {ego_motion_len} != num_pairs*ego_motion_dim = {expected_ego}"
        )

    pos = Z5PCWM1_HEADER_SIZE
    end_encoder = pos + encoder_len
    end_decoder = end_encoder + decoder_len
    end_predictor = end_decoder + predictor_len
    end_latent_init = end_predictor + latent_init_len
    end_residuals = end_latent_init + residuals_len
    end_ego = end_residuals + ego_motion_len
    end_meta = end_ego + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_blob = blob[pos:end_encoder]
    decoder_blob = blob[end_encoder:end_decoder]
    predictor_blob = blob[end_decoder:end_predictor]
    latent_init_blob = blob[end_predictor:end_latent_init]
    residuals_blob = blob[end_latent_init:end_residuals]
    ego_motion_blob = blob[end_residuals:end_ego]
    meta_blob = blob[end_ego:end_meta]

    encoder_sd = _deserialize_state_dict(encoder_blob)
    decoder_sd = _deserialize_state_dict(decoder_blob)
    predictor_sd = _deserialize_state_dict(predictor_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    q_li = np.frombuffer(latent_init_blob, dtype=np.int8).copy().reshape(latent_dim)
    q_r = np.frombuffer(residuals_blob, dtype=np.int8).copy().reshape(
        num_pairs, latent_dim
    )
    q_e = np.frombuffer(ego_motion_blob, dtype=np.int8).copy().reshape(
        num_pairs, ego_motion_dim
    )

    scale_li = float(meta.pop("_latent_init_scale"))
    zp_li = float(meta.pop("_latent_init_zp"))
    scale_r = float(meta.pop("_residuals_scale"))
    zp_r = float(meta.pop("_residuals_zp"))
    scale_e = float(meta.pop("_ego_motion_scale"))
    zp_e = float(meta.pop("_ego_motion_zp"))

    latent_init = torch.from_numpy(_dequantize_from_int8(q_li, scale_li, zp_li))
    residuals = torch.from_numpy(_dequantize_from_int8(q_r, scale_r, zp_r))
    ego_motion = torch.from_numpy(_dequantize_from_int8(q_e, scale_e, zp_e))

    return PredictiveCodingArchive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        predictor_state_dict=predictor_sd,
        latent_init=latent_init,
        residuals=residuals,
        ego_motion=ego_motion,
        meta=meta,
        schema_version=int(version),
    )


@dataclass(frozen=True)
class PredictiveCodingArchiveNumpy:
    """Torch-free parsed Z5PCWM1 archive — the numpy-portable inflate-time contract.

    Sister of :class:`PredictiveCodingArchive` but with ``np.ndarray`` weights /
    latents so the shipped inflate runtime needs ONLY numpy + brotli (no torch)
    per the 8th MLX-first standing directive. The ``decoder_state_dict`` +
    ``predictor_state_dict`` are the inflate-time consumers; the encoder is
    provenance-only.
    """

    encoder_state_dict: dict[str, np.ndarray]
    decoder_state_dict: dict[str, np.ndarray]
    predictor_state_dict: dict[str, np.ndarray]
    latent_init: np.ndarray  # (latent_dim,) fp32
    residuals: np.ndarray    # (num_pairs, latent_dim) fp32
    ego_motion: np.ndarray   # (num_pairs, ego_motion_dim) fp32
    meta: dict[str, object]
    schema_version: int


def parse_archive_numpy(blob: bytes) -> PredictiveCodingArchiveNumpy:
    """Torch-free parse of a Z5PCWM1 archive for the numpy-portable inflate.

    Identical section walk to :func:`parse_archive` but reconstructs every
    weight / latent / residual / ego-motion as a numpy array with NO torch
    import. Per the 8th MLX-first directive's bridge contract the shipped
    inflate runtime reads weights via this path so the runtime tree carries
    only numpy + brotli.
    """
    if len(blob) < Z5PCWM1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {Z5PCWM1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        encoder_len,
        decoder_len,
        predictor_len,
        latent_init_len,
        residuals_len,
        ego_motion_len,
        meta_len,
    ) = struct.unpack(Z5PCWM1_HEADER_FMT, blob[:Z5PCWM1_HEADER_SIZE])
    if magic != Z5PCWM1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {Z5PCWM1_MAGIC!r})")
    if version != Z5PCWM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    if latent_init_len != latent_dim:
        raise ValueError(
            f"latent_init_len {latent_init_len} != latent_dim {latent_dim}"
        )
    expected_residuals = num_pairs * latent_dim
    if residuals_len != expected_residuals:
        raise ValueError(
            f"residuals_len {residuals_len} != num_pairs*latent_dim = {expected_residuals}"
        )
    expected_ego = num_pairs * ego_motion_dim
    if ego_motion_len != expected_ego:
        raise ValueError(
            f"ego_motion_len {ego_motion_len} != num_pairs*ego_motion_dim = {expected_ego}"
        )

    pos = Z5PCWM1_HEADER_SIZE
    end_encoder = pos + encoder_len
    end_decoder = end_encoder + decoder_len
    end_predictor = end_decoder + predictor_len
    end_latent_init = end_predictor + latent_init_len
    end_residuals = end_latent_init + residuals_len
    end_ego = end_residuals + ego_motion_len
    end_meta = end_ego + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_sd = _deserialize_numpy_state_dict(blob[pos:end_encoder])
    decoder_sd = _deserialize_numpy_state_dict(blob[end_encoder:end_decoder])
    predictor_sd = _deserialize_numpy_state_dict(blob[end_decoder:end_predictor])
    meta = json.loads(blob[end_ego:end_meta].decode("utf-8"))

    q_li = np.frombuffer(blob[end_predictor:end_latent_init], dtype=np.int8).reshape(latent_dim)
    q_r = np.frombuffer(blob[end_latent_init:end_residuals], dtype=np.int8).reshape(
        num_pairs, latent_dim
    )
    q_e = np.frombuffer(blob[end_residuals:end_ego], dtype=np.int8).reshape(
        num_pairs, ego_motion_dim
    )

    scale_li = float(meta.pop("_latent_init_scale"))
    zp_li = float(meta.pop("_latent_init_zp"))
    scale_r = float(meta.pop("_residuals_scale"))
    zp_r = float(meta.pop("_residuals_zp"))
    scale_e = float(meta.pop("_ego_motion_scale"))
    zp_e = float(meta.pop("_ego_motion_zp"))

    # dequant mirrors _dequantize_from_int8: (q + 127) * scale + zero_point
    latent_init = (q_li.astype(np.float32) + 127.0) * scale_li + zp_li
    residuals = (q_r.astype(np.float32) + 127.0) * scale_r + zp_r
    ego_motion = (q_e.astype(np.float32) + 127.0) * scale_e + zp_e

    return PredictiveCodingArchiveNumpy(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        predictor_state_dict=predictor_sd,
        latent_init=latent_init,
        residuals=residuals,
        ego_motion=ego_motion,
        meta=meta,
        schema_version=int(version),
    )


def parse_z5pcwm1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for Z5PCWM1 (predictive-coding world-model) grammar.

    Canonical section-offset parser for Z5PCWM1 inner-blob bytes. The returned
    mapping is the data contract consumed by:

    - :mod:`tac.analysis.scorer_conditional_mdl` (ScorerConditionalMDLEstimator
      section-aware Tier A density estimation)
    - :mod:`tac.analysis.hnerv_packet_sections` (parser-section manifest
      dispatch — Z5PCWM1 auto-detection by ``b"Z5WM"`` magic prefix)

    Z5PCWM1 extends Z4CR1 by adding a predictor section and replacing the
    per-pair-latent section with (latent_init + residuals + ego_motion). The
    predictor implements Rao-Ballard 1999 predictive-coding world-model.

    Returned sections (Tier A / Tier B targets):

    - ``z5pcwm1_header`` — 39-byte header (control_or_metadata; fixed layout)
    - ``encoder_blob`` — brotli q=9 compressed encoder state_dict
      (training_provenance_only — parsed/loaded for provenance, but not
      score-affecting at inflate because ``frames_for_encoder=None``)
    - ``decoder_blob`` — brotli q=9 compressed decoder state_dict
      (decoder_weight_stream — the inflate-time renderer)
    - ``predictor_blob`` — brotli q=9 compressed predictor state_dict
      (decoder_weight_stream — Rao-Ballard predictive-coding world-model
      that rolls forward the latent state given ego-motion)
    - ``latent_init_blob`` — int8 quantized initial latent z_0 (latent_stream)
    - ``residuals_blob`` — int8 quantized per-pair residuals (latent_stream)
    - ``ego_motion_blob`` — int8 quantized per-pair ego-motion vectors
      (sidecar_or_correction_stream — ego-motion-conditional side info)
    - ``meta_blob`` — sorted-keys JSON with predictive_coding_world_model_meta
      provenance tag + per-section scale/zero_point (control_or_metadata)

    The byte ranges returned here MUST agree with the writer in
    :func:`pack_archive` and with the canonical full-decode parser
    :func:`parse_archive`. The single-source-of-truth for Z5PCWM1 byte layout
    is :data:`Z5PCWM1_HEADER_FMT` + :data:`Z5PCWM1_HEADER_SIZE`.

    Differs from :func:`parse_archive` in that this function returns
    section-offset tuples only (no torch state_dict deserialization). It is
    cheaper and is safe to call with brotli-tampered blobs.

    Raises ``ValueError`` on:

    - short header (< 39 bytes)
    - bad magic (!= ``b"Z5WM"``)
    - unsupported schema version (!= 1)
    - latent_init_len != latent_dim
    - residuals_len != num_pairs * latent_dim
    - ego_motion_len != num_pairs * ego_motion_dim
    - archive size mismatch (declared end_meta != len(archive_bytes)),
      covering BOTH truncated archives and trailing-byte schema drift.
    """
    if len(archive_bytes) < Z5PCWM1_HEADER_SIZE:
        raise ValueError(
            f"z5pcwm1 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {Z5PCWM1_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        encoder_len,
        decoder_len,
        predictor_len,
        latent_init_len,
        residuals_len,
        ego_motion_len,
        meta_len,
    ) = struct.unpack(Z5PCWM1_HEADER_FMT, archive_bytes[:Z5PCWM1_HEADER_SIZE])
    if magic != Z5PCWM1_MAGIC:
        raise ValueError(
            f"z5pcwm1 archive: bad magic {magic!r} (expected {Z5PCWM1_MAGIC!r})"
        )
    if version != Z5PCWM1_SCHEMA_VERSION:
        raise ValueError(
            f"z5pcwm1 archive: unsupported schema version {version} "
            f"(expected {Z5PCWM1_SCHEMA_VERSION})"
        )
    if latent_init_len != int(latent_dim):
        raise ValueError(
            f"z5pcwm1 archive: latent_init_len {latent_init_len} != "
            f"latent_dim {latent_dim}"
        )
    expected_residuals = int(num_pairs) * int(latent_dim)
    if residuals_len != expected_residuals:
        raise ValueError(
            f"z5pcwm1 archive: residuals_len {residuals_len} != "
            f"num_pairs*latent_dim = {expected_residuals}"
        )
    expected_ego = int(num_pairs) * int(ego_motion_dim)
    if ego_motion_len != expected_ego:
        raise ValueError(
            f"z5pcwm1 archive: ego_motion_len {ego_motion_len} != "
            f"num_pairs*ego_motion_dim = {expected_ego}"
        )
    end_header = Z5PCWM1_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_predictor = end_decoder + int(predictor_len)
    end_latent_init = end_predictor + int(latent_init_len)
    end_residuals = end_latent_init + int(residuals_len)
    end_ego = end_residuals + int(ego_motion_len)
    end_meta = end_ego + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"z5pcwm1 archive: archive size {len(archive_bytes)} != expected "
            f"{end_meta} from header"
        )
    return {
        "z5pcwm1_header": (0, Z5PCWM1_HEADER_SIZE),
        "encoder_blob": (end_header, int(encoder_len)),
        "decoder_blob": (end_encoder, int(decoder_len)),
        "predictor_blob": (end_decoder, int(predictor_len)),
        "latent_init_blob": (end_predictor, int(latent_init_len)),
        "residuals_blob": (end_latent_init, int(residuals_len)),
        "ego_motion_blob": (end_residuals, int(ego_motion_len)),
        "meta_blob": (end_ego, int(meta_len)),
    }


# Canonical optimization-role mapping for Z5PCWM1 sections.
#
# Note: the predictor is decoder_weight_stream because in Rao-Ballard
# predictive-coding the predictor IS part of the inference path — it rolls
# latent state forward at every pair, and the inflate consumes its weights.
# residuals + latent_init are latent_stream (predictive-coding residual
# entropy); ego_motion is sidecar (side-information for predictor
# conditioning).
Z5PCWM1_SECTION_ROLES: dict[str, str] = {
    "z5pcwm1_header": "control_or_metadata",
    "encoder_blob": "training_provenance_only",
    "decoder_blob": "decoder_weight_stream",
    "predictor_blob": "decoder_weight_stream",
    "latent_init_blob": "latent_stream",
    "residuals_blob": "latent_stream",
    "ego_motion_blob": "sidecar_or_correction_stream",
    "meta_blob": "control_or_metadata",
}


__all__ = [
    "Z5PCWM1_HEADER_FMT",
    "Z5PCWM1_HEADER_SIZE",
    "Z5PCWM1_MAGIC",
    "Z5PCWM1_SCHEMA_VERSION",
    "Z5PCWM1_SECTION_ROLES",
    "PredictiveCodingArchive",
    "PredictiveCodingArchiveNumpy",
    "pack_archive",
    "parse_archive",
    "parse_archive_numpy",
    "parse_z5pcwm1_archive_bytes",
]
