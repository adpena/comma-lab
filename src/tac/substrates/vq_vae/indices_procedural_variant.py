# SPDX-License-Identifier: MIT
"""VQ-VAE procedural-index residual scaffold.

This module is intentionally separate from
``distillation_procedural_variant.py``. The codebook variant is a direct
replacement surface; the VQ-VAE ``indices_blob`` is a score-affecting address
stream, so the safe L0 design is predictor-plus-residual byte accounting via
``procedural_predictor_plus_residual_correction_savings_v1``.
"""
from __future__ import annotations

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
from dataclasses import dataclass
import struct

from tac.canonical_equations.procedural_predictor_residual_savings import (
    EQUATION_ID as RESIDUAL_EQUATION_ID,
)
from tac.canonical_equations.procedural_predictor_residual_savings import (
    predict_procedural_predictor_plus_residual_correction_savings,
)
from tac.procedural_codebook_generator import DEFAULT_GENERATOR_KIND
from tac.procedural_codebook_generator import derive_codebook_from_seed
from tac.substrates.vq_vae.archive import (
    BROTLI_QUALITY,
    VQV1_HEADER_FMT,
    VQV1_HEADER_SIZE,
    VQV1_MAGIC,
    VQV1_PROCEDURAL_INDICES_SENTINEL,
    VQV1_SCHEMA_VERSION,
)

RESIDUAL_CONTEXT = (
    "vq_vae_indices_blob_residual_correction_on_parser_safe_score_affecting_indices"
)
"""Residual-hybrid context token for the VQ-VAE indices_blob scaffold."""

PROCEDURAL_INDICES_ENVELOPE_VERSION = 1
"""Envelope version for VQPI procedural-index sections."""

_GENERATOR_TAGS: dict[str, int] = {"xorshift": 0, "lcg": 1, "pcg64": 2}
_GENERATOR_BY_TAG: dict[int, str] = {value: key for key, value in _GENERATOR_TAGS.items()}
_ENVELOPE_HEADER_FMT = "<4sBBHI"
_ENVELOPE_HEADER_SIZE = struct.calcsize(_ENVELOPE_HEADER_FMT)


class ProceduralIndicesVariantError(ValueError):
    """Raised when a procedural-index envelope is malformed."""


@dataclass(frozen=True)
class ProceduralIndicesComposition:
    """Typed result for a VQPI procedural-index archive composition."""

    archive_bytes: bytes
    indices_blob: bytes
    original_indices_bytes: int
    predictor_seed_or_code_bytes: int
    residual_stream_bytes: int
    container_overhead_bytes: int
    predicted_rate_accounting: dict[str, object]

    @property
    def replacement_total_bytes(self) -> int:
        """Bytes charged for seed + residual + envelope overhead."""

        return (
            self.predictor_seed_or_code_bytes
            + self.residual_stream_bytes
            + self.container_overhead_bytes
        )

    @property
    def delta_bytes_replacement_minus_original(self) -> int:
        """Positive means the residual scaffold grew the indices section."""

        return self.replacement_total_bytes - self.original_indices_bytes


def _validate_seed(seed_bytes: bytes) -> bytes:
    if not isinstance(seed_bytes, (bytes, bytearray, memoryview)):
        raise ProceduralIndicesVariantError(
            f"seed_bytes must be bytes-like; got {type(seed_bytes).__name__}"
        )
    seed = bytes(seed_bytes)
    if not (8 <= len(seed) <= 256):
        raise ProceduralIndicesVariantError(
            f"seed length {len(seed)} outside residual scaffold range [8, 256]"
        )
    return seed


def _validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 4:
        raise ProceduralIndicesVariantError(f"indices shape must be 4-D; got {shape}")
    if any(int(dim) <= 0 for dim in shape):
        raise ProceduralIndicesVariantError(f"indices shape must be positive; got {shape}")
    return tuple(int(dim) for dim in shape)


def derive_procedural_indices_predictor(
    seed_bytes: bytes,
    *,
    shape: tuple[int, ...],
    codebook_size: int,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> torch.Tensor:
    """Derive deterministic VQ index predictions from an in-archive seed."""

    seed = _validate_seed(seed_bytes)
    index_shape = _validate_shape(shape)
    if codebook_size <= 0 or codebook_size > 65536:
        raise ProceduralIndicesVariantError(
            f"codebook_size {codebook_size} outside (0, 65536]"
        )
    if generator_kind not in _GENERATOR_TAGS:
        raise ProceduralIndicesVariantError(
            f"generator_kind {generator_kind!r} not in {sorted(_GENERATOR_TAGS)}"
        )
    raw = derive_codebook_from_seed(
        seed_bytes=seed,
        output_shape=index_shape,
        dtype=np.uint16,
        generator_kind=generator_kind,
    )
    return torch.from_numpy((raw % int(codebook_size)).astype(np.int64, copy=True))


def encode_procedural_indices_blob(
    indices: torch.Tensor,
    *,
    codebook_size: int,
    seed_bytes: bytes,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
    brotli_quality: int = BROTLI_QUALITY,
) -> bytes:
    """Encode indices as seed-prediction plus brotli-compressed residuals."""

    if indices.dtype not in (torch.int64, torch.int32):
        raise ProceduralIndicesVariantError(f"indices must be integer; got {indices.dtype}")
    shape = _validate_shape(tuple(int(dim) for dim in indices.shape))
    seed = _validate_seed(seed_bytes)
    predicted = derive_procedural_indices_predictor(
        seed,
        shape=shape,
        codebook_size=codebook_size,
        generator_kind=generator_kind,
    )
    indices_cpu = indices.detach().to("cpu", dtype=torch.int64).contiguous()
    if int(indices_cpu.min()) < 0 or int(indices_cpu.max()) >= int(codebook_size):
        raise ProceduralIndicesVariantError("indices out of codebook range")
    residual = ((indices_cpu - predicted) % int(codebook_size)).numpy().astype(
        np.uint16,
        copy=False,
    )
    residual_blob = brotli.compress(residual.tobytes(order="C"), quality=brotli_quality)
    return (
        struct.pack(
            _ENVELOPE_HEADER_FMT,
            VQV1_PROCEDURAL_INDICES_SENTINEL,
            PROCEDURAL_INDICES_ENVELOPE_VERSION,
            _GENERATOR_TAGS[generator_kind],
            len(seed),
            len(residual_blob),
        )
        + seed
        + residual_blob
    )


def decode_procedural_indices_blob(
    blob: bytes,
    *,
    shape: tuple[int, ...],
    codebook_size: int,
) -> torch.Tensor:
    """Decode a VQPI seed-plus-residual indices section exactly."""

    shape = _validate_shape(shape)
    if len(blob) < _ENVELOPE_HEADER_SIZE:
        raise ProceduralIndicesVariantError("procedural indices envelope too short")
    sentinel, version, generator_tag, seed_len, residual_len = struct.unpack(
        _ENVELOPE_HEADER_FMT,
        blob[:_ENVELOPE_HEADER_SIZE],
    )
    if sentinel != VQV1_PROCEDURAL_INDICES_SENTINEL:
        raise ProceduralIndicesVariantError(f"bad procedural indices sentinel: {sentinel!r}")
    if version != PROCEDURAL_INDICES_ENVELOPE_VERSION:
        raise ProceduralIndicesVariantError(
            f"unsupported procedural indices envelope version: {version}"
        )
    try:
        generator_kind = _GENERATOR_BY_TAG[int(generator_tag)]
    except KeyError as exc:
        raise ProceduralIndicesVariantError(
            f"unknown procedural indices generator tag: {generator_tag}"
        ) from exc
    pos = _ENVELOPE_HEADER_SIZE
    seed = _validate_seed(blob[pos : pos + seed_len])
    pos += seed_len
    residual_blob = blob[pos : pos + residual_len]
    pos += residual_len
    if pos != len(blob):
        raise ProceduralIndicesVariantError(
            f"procedural indices envelope has trailing bytes: {len(blob) - pos}"
        )
    residual_raw = brotli.decompress(residual_blob)
    expected_raw_len = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    if len(residual_raw) != expected_raw_len:
        raise ProceduralIndicesVariantError(
            f"residual raw length {len(residual_raw)} != expected {expected_raw_len}"
        )
    residual = torch.from_numpy(
        np.frombuffer(residual_raw, dtype=np.uint16).copy().reshape(shape)
    ).to(torch.int64)
    predicted = derive_procedural_indices_predictor(
        seed,
        shape=shape,
        codebook_size=codebook_size,
        generator_kind=generator_kind,
    )
    return (predicted + residual) % int(codebook_size)


def analyze_procedural_indices_blob(
    indices: torch.Tensor,
    *,
    codebook_size: int,
    seed_bytes: bytes,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> dict[str, object]:
    """Return honest rate-only byte accounting for a VQPI indices scaffold."""

    raw_indices_bytes = int(indices.numel()) * np.dtype(np.int16).itemsize
    blob = encode_procedural_indices_blob(
        indices,
        codebook_size=codebook_size,
        seed_bytes=seed_bytes,
        generator_kind=generator_kind,
    )
    seed_len = len(_validate_seed(seed_bytes))
    residual_len = len(blob) - _ENVELOPE_HEADER_SIZE - seed_len
    accounting = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=raw_indices_bytes,
        predictor_seed_or_code_bytes=seed_len,
        residual_stream_bytes=residual_len,
        container_overhead_bytes=_ENVELOPE_HEADER_SIZE,
        context=RESIDUAL_CONTEXT,
    )
    return {
        "equation_id": RESIDUAL_EQUATION_ID,
        "context": RESIDUAL_CONTEXT,
        "indices_blob_bytes": len(blob),
        "original_indices_bytes": raw_indices_bytes,
        "predictor_seed_or_code_bytes": seed_len,
        "residual_stream_bytes": residual_len,
        "container_overhead_bytes": _ENVELOPE_HEADER_SIZE,
        "predicted_rate_accounting": accounting,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def compose_with_procedural_indices(
    *,
    original_archive_bytes: bytes,
    seed_bytes: bytes,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> ProceduralIndicesComposition:
    """Compose a VQV1 archive with a VQPI procedural indices section."""

    if len(original_archive_bytes) < VQV1_HEADER_SIZE:
        raise ValueError("archive too short")
    (
        magic,
        version,
        codebook_size,
        embedding_dim,
        num_pairs,
        h_grid,
        w_grid,
        decoder_len,
        indices_len,
        meta_len,
    ) = struct.unpack(VQV1_HEADER_FMT, original_archive_bytes[:VQV1_HEADER_SIZE])
    if magic != VQV1_MAGIC or version != VQV1_SCHEMA_VERSION:
        raise ValueError(f"not a VQV1 archive: magic={magic!r} version={version}")
    pos = VQV1_HEADER_SIZE
    decoder_blob = original_archive_bytes[pos : pos + decoder_len]
    pos += decoder_len
    indices_blob = original_archive_bytes[pos : pos + indices_len]
    pos += indices_len
    meta_blob = original_archive_bytes[pos : pos + meta_len]
    pos += meta_len
    if pos != len(original_archive_bytes):
        raise ValueError("archive header lengths do not consume full archive")
    if indices_blob.startswith(VQV1_PROCEDURAL_INDICES_SENTINEL):
        raise ProceduralIndicesVariantError("archive already uses VQPI procedural indices")

    from tac.substrates.vq_vae.archive import _unpack_indices_int16

    shape = (int(num_pairs), 2, int(h_grid), int(w_grid))
    indices = _unpack_indices_int16(indices_blob, shape)
    new_indices_blob = encode_procedural_indices_blob(
        indices,
        codebook_size=int(codebook_size),
        seed_bytes=seed_bytes,
        generator_kind=generator_kind,
    )
    seed_len = len(_validate_seed(seed_bytes))
    residual_len = len(new_indices_blob) - _ENVELOPE_HEADER_SIZE - seed_len
    accounting = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=int(indices_len),
        predictor_seed_or_code_bytes=seed_len,
        residual_stream_bytes=residual_len,
        container_overhead_bytes=_ENVELOPE_HEADER_SIZE,
        context=RESIDUAL_CONTEXT,
    )
    new_header = struct.pack(
        VQV1_HEADER_FMT,
        VQV1_MAGIC,
        VQV1_SCHEMA_VERSION,
        int(codebook_size),
        int(embedding_dim),
        int(num_pairs),
        int(h_grid),
        int(w_grid),
        int(decoder_len),
        len(new_indices_blob),
        int(meta_len),
    )
    archive_bytes = new_header + decoder_blob + new_indices_blob + meta_blob
    return ProceduralIndicesComposition(
        archive_bytes=archive_bytes,
        indices_blob=new_indices_blob,
        original_indices_bytes=int(indices_len),
        predictor_seed_or_code_bytes=seed_len,
        residual_stream_bytes=residual_len,
        container_overhead_bytes=_ENVELOPE_HEADER_SIZE,
        predicted_rate_accounting=accounting,
    )


__all__ = [
    "PROCEDURAL_INDICES_ENVELOPE_VERSION",
    "RESIDUAL_CONTEXT",
    "ProceduralIndicesComposition",
    "ProceduralIndicesVariantError",
    "analyze_procedural_indices_blob",
    "compose_with_procedural_indices",
    "decode_procedural_indices_blob",
    "derive_procedural_indices_predictor",
    "encode_procedural_indices_blob",
]
