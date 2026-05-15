# SPDX-License-Identifier: MIT
"""Z3 v2 latent-replacement archive grammar (council omnibus Decision 3).

v1 (sibling ``archive.py``) was an append-only sidecar over A1 bytes, which
structurally cannot realize the predicted byte savings: bytes can only be
ADDED to A1, never removed. Per the grand council omnibus 2026-05-14 binding
verdict (commit ``7872c9f4b`` Decision 3, 11/11 PROCEED Option B), v2
REPLACES A1's ``latent_blob`` (15387 B at fixed offset) with a Z3-coded
section emitted by the Ballé-2018 scale hyperprior, while preserving the
A1 ``decoder_blob`` (162164 B) and ``sidecar_blob`` (~607 B) verbatim.

A1 wire format (decoded by ``submissions/a1/src/codec.py``)::

    [uint32 LE section_total = 4 + DECODER_BLOB_LEN = 162168]
    [decoder_blob 162164 B]
    [latent_blob 15387 B]                                 <-- v2 REPLACES this
    [sidecar_blob (variable; ~607 B)]

Z3 v2 wire format (decoded by sibling ``inflate_v2.py``)::

    [uint32 LE section_total = 4 + DECODER_BLOB_LEN = 162168]   (verbatim from A1)
    [decoder_blob 162164 B]                                     (verbatim from A1)
    [Z3HV2 header + payload]                                    (NEW; replaces latent_blob)
    [sidecar_blob (variable; ~607 B)]                           (verbatim from A1)

Z3HV2 section structure (the bytes between offset 162168 and the trailing A1 sidecar)::

    magic              : 4 bytes ASCII "Z3V2"   (distinguishes from v1 "Z3H1")
    version            : uint8 (== 2)
    n_pairs            : uint16 LE (== 600 == A1_N_PAIRS)
    hyper_dim          : uint8  (Ballé hyper-latent dim, default 8; diagnostic-only in direct-residual mode)
    latent_dim         : uint8  (== A1_LATENT_DIM == 28)
    int8_w_scale       : float32 LE (4 B; w_int8 / scale = w_real)
    quant_step         : float32 LE (4 B; latent residual quantization Δ)
    min_sigma          : float32 LE (4 B; sigma clamp lower bound)
    max_sigma          : float32 LE (4 B; sigma clamp upper bound)
    factorized_half    : float32 LE (4 B; w_hat factorized prior half-range)
    reserved           : 2 B (== 0)
    weights_blob_len   : uint16 LE (2 B; brotli-compressed int8 MLP weights length; 0 in direct-residual mode)
    weights_blob       : <weights_blob_len> bytes
    w_hat_blob_len     : uint32 LE (4 B; brotli-compressed int8 hyper-latent codes; 0 in direct-residual mode)
    w_hat_blob         : <w_hat_blob_len> bytes  (n_pairs * hyper_dim int8 codes, or empty)
    residual_blob_len  : uint32 LE (4 B; brotli-compressed AC-coded latent residual)
    residual_blob      : <residual_blob_len> bytes  (n_pairs * latent_dim int8 residuals)
    latent_offset_blob : 28 * float32 = 112 B (per-dim centered affine offset)
    latent_scale_blob  : 28 * float32 = 112 B (per-dim scale for re-affine to A1 range)

The trailing 2*4*28=224 bytes encode the per-dim affine `(offset, scale)` so
the inflate path can reconstruct latents in A1's quantized-range space
(``latents = decoded_q.float() * scale + offset``) BEFORE A1's
``apply_latent_sidecar`` runs on top. v2 uses a centered offset because the
residual is signed int8; min-offset sections are version-1 legacy and are
refused rather than supported. Total fixed overhead
beyond the Ballé bits = 32 (header) + 2 + 4 + 4 + 224 = 266 B + the int8
MLP weights blob (~1.8k params * 0.6 brotli ratio ~ 1080 B).

LOC budget: ≤ 350 LOC per HNeRV parity discipline L7. Z3 v2 IS a bolt-on
(A1 weights frozen); substrate-engineering exemption NOT used.

NO score claim. NO promotion. NO exact-eval dispatch from this module.

Per Catalog #146 the inflate runtime that consumes this format is at
``tac.substrates.z3_balle_hyperprior_bolton.inflate_v2``.

Per Catalog #220 (substrate L1 SCAFFOLD must be COMPLETE or RESEARCH-ONLY):
v2 IS the OPERATIONAL latent-replacement contract — bytes are CHANGED at
runtime AND consumed by the inflate path to produce different latents at the
same A1 decoder boundary. The current production-safe mode is direct residual:
hyperprior side-info fields are allowed to be empty and are not authority for
score movement until a real entropy-coded residual decoder consumes them.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import brotli
import torch

from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
)

Z3HV2_MAGIC = b"Z3V2"
Z3HV2_VERSION = 2

# magic(4s) + version(B) + n_pairs(H) + hyper_dim(B) + latent_dim(B)
# + int8_w_scale(f) + quant_step(f) + min_sigma(f) + max_sigma(f)
# + factorized_half(f) + reserved(2s) = 4+1+2+1+1+4+4+4+4+4+2 = 31
Z3HV2_HEADER_STRUCT = struct.Struct("<4sBHBBfffff2s")

# A1 wire-format constants (mirrored from submissions/a1/src/codec.py).
A1_DECODER_BLOB_LEN = 162_164
A1_LATENT_BLOB_LEN = 15_387
A1_SECTION_TOTAL_PREFIX_LEN = 4
A1_DECODER_SECTION_TOTAL = A1_SECTION_TOTAL_PREFIX_LEN + A1_DECODER_BLOB_LEN  # 162168


Z3HV2_BYTE_OFFSET_AFTER_DECODER = A1_DECODER_SECTION_TOTAL  # 162168
Z3HV2_PER_DIM_AFFINE_LEN = 4 * A1_LATENT_DIM * 2  # 224 B (min + scale, fp32)


# Per-line waivers: WEIGHTS_ONLY_FALSE_OK / EXPORT_FORMAT_OK -- this module is
# pure encode/decode; no torch.load and no training-time deserialization.


@dataclass(frozen=True)
class Z3HV2SectionMeta:
    """Decoded Z3HV2 header metadata."""

    n_pairs: int
    hyper_dim: int
    latent_dim: int
    int8_w_scale: float
    quant_step: float
    min_sigma: float
    max_sigma: float
    factorized_half_range: float


@dataclass(frozen=True)
class Z3V2CompositionArchiveContract:
    """Typed authority contract for a Z3 v2 composition payload.

    Contracts authority flags are intentionally duplicated here so callers
    cannot accidentally treat an unverified packet as byte-saving or
    exact-eval-ready without the trainer's smoke + auth-eval pipeline.
    """

    payload_bytes: bytes
    layout: str
    base_archive_bytes: int
    z3v2_section_bytes: int
    a1_latent_blob_bytes_replaced: int
    archive_bytes: int
    byte_saving: bool
    byte_savings_bytes: int
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool
    exact_eval_ready: bool
    result_review_blockers: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        """Return JSON-safe manifest fields for stats/provenance outputs."""
        return {
            "layout": self.layout,
            "base_archive_bytes": self.base_archive_bytes,
            "z3v2_section_bytes": self.z3v2_section_bytes,
            "a1_latent_blob_bytes_replaced": self.a1_latent_blob_bytes_replaced,
            "archive_bytes": self.archive_bytes,
            "byte_saving": self.byte_saving,
            "byte_savings_bytes": self.byte_savings_bytes,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "exact_eval_ready": self.exact_eval_ready,
            "result_review_blockers": list(self.result_review_blockers),
        }


def _validate_a1_decoder_section_header(a1_bytes: bytes) -> None:
    """Require the exact A1 decoder section boundary before Z3V2 splicing."""
    if len(a1_bytes) < A1_SECTION_TOTAL_PREFIX_LEN:
        raise ValueError("a1_bytes too short for A1 section_total prefix")
    (section_total,) = struct.unpack_from("<I", a1_bytes, 0)
    if int(section_total) != A1_DECODER_SECTION_TOTAL:
        raise ValueError(
            "A1 decoder section_total mismatch: "
            f"{int(section_total)} != {A1_DECODER_SECTION_TOTAL}"
        )


def encode_z3hv2_section(
    *,
    hyperprior_weights_int8: bytes,
    w_hat_int8: bytes,
    residual_int8: bytes,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
    hyper_dim: int,
    int8_w_scale: float,
    quant_step: float,
    min_sigma: float,
    max_sigma: float,
    factorized_half_range: float,
    n_pairs: int = A1_N_PAIRS,
    latent_dim: int = A1_LATENT_DIM,
) -> bytes:
    """Pack the Z3HV2 latent-replacement section.

    Args:
        hyperprior_weights_int8: Raw int8 bytes for the hyperprior MLP
            weights (will be brotli-compressed inside the section).
        w_hat_int8: ``n_pairs * hyper_dim`` int8 bytes for the quantized
            hyper-latents (one per pair).
        residual_int8: ``n_pairs * latent_dim`` int8 bytes for the
            quantized latent residuals (one row per pair).
        latent_offset: ``(latent_dim,)`` centered affine offset (fp32).
        latent_scale: ``(latent_dim,)`` per-dim scale (fp32).
        hyper_dim: Hyper-latent dim (e.g. 8).
        int8_w_scale: scale s.t. w_int8 / scale = w_real.
        quant_step: quantization step Δ for the conditional-Gaussian AC coder.
        min_sigma, max_sigma: bounds used at encoding-time.
        factorized_half_range: half-range used by the factorized prior on w_hat.
        n_pairs: Number of pairs (default A1_N_PAIRS = 600).
        latent_dim: Latent dim (default A1_LATENT_DIM = 28).

    Returns:
        Section bytes (header + length-prefixed brotli blobs + per-dim affine).
    """
    if n_pairs != A1_N_PAIRS:
        raise ValueError(f"n_pairs must be {A1_N_PAIRS}; got {n_pairs}")
    if latent_dim != A1_LATENT_DIM:
        raise ValueError(f"latent_dim must be {A1_LATENT_DIM}; got {latent_dim}")
    if hyper_dim <= 0 or hyper_dim > 255:
        raise ValueError(f"hyper_dim must be in [1, 255]; got {hyper_dim}")
    direct_residual_mode = len(hyperprior_weights_int8) == 0 and len(w_hat_int8) == 0
    if (len(hyperprior_weights_int8) == 0) != (len(w_hat_int8) == 0):
        raise ValueError(
            "Z3HV2 requires weights and w_hat to be both present or both empty"
        )
    if not direct_residual_mode and len(w_hat_int8) != n_pairs * hyper_dim:
        raise ValueError(
            f"w_hat_int8 length {len(w_hat_int8)} != n_pairs*hyper_dim "
            f"{n_pairs * hyper_dim}"
        )
    if len(residual_int8) != n_pairs * latent_dim:
        raise ValueError(
            f"residual_int8 length {len(residual_int8)} != n_pairs*latent_dim "
            f"{n_pairs * latent_dim}"
        )
    if float(quant_step) != 1.0:
        raise ValueError(
            "Z3HV2 quant_step must be 1.0 until train/export/inflate all "
            "apply non-unit residual quantization consistently"
        )
    if latent_offset.shape != (latent_dim,) or latent_scale.shape != (latent_dim,):
        raise ValueError(
            f"latent_offset and latent_scale must be ({latent_dim},); got "
            f"{tuple(latent_offset.shape)} and {tuple(latent_scale.shape)}"
        )

    header = Z3HV2_HEADER_STRUCT.pack(
        Z3HV2_MAGIC,
        Z3HV2_VERSION,
        n_pairs,
        hyper_dim,
        latent_dim,
        float(int8_w_scale),
        float(quant_step),
        float(min_sigma),
        float(max_sigma),
        float(factorized_half_range),
        b"\x00\x00",
    )

    weights_compressed = (
        b""
        if direct_residual_mode
        else brotli.compress(hyperprior_weights_int8, quality=11)
    )
    if len(weights_compressed) > 0xFFFF:
        raise ValueError(
            f"hyperprior weights blob too large: {len(weights_compressed)} > 65535"
        )
    w_hat_compressed = (
        b"" if direct_residual_mode else brotli.compress(w_hat_int8, quality=11)
    )
    residual_compressed = brotli.compress(residual_int8, quality=11)

    affine_bytes = (
        latent_offset.detach().cpu().to(torch.float32).numpy().tobytes()
        + latent_scale.detach().cpu().to(torch.float32).numpy().tobytes()
    )
    if len(affine_bytes) != Z3HV2_PER_DIM_AFFINE_LEN:
        raise ValueError(
            f"per-dim affine bytes length {len(affine_bytes)} != "
            f"{Z3HV2_PER_DIM_AFFINE_LEN}"
        )

    return (
        header
        + struct.pack("<H", len(weights_compressed))
        + weights_compressed
        + struct.pack("<I", len(w_hat_compressed))
        + w_hat_compressed
        + struct.pack("<I", len(residual_compressed))
        + residual_compressed
        + affine_bytes
    )


def decode_z3hv2_section(
    data: bytes,
) -> tuple[Z3HV2SectionMeta, bytes, bytes, bytes, torch.Tensor, torch.Tensor, int]:
    """Unpack the Z3HV2 section.

    Returns ``(meta, hyperprior_weights_int8, w_hat_int8, residual_int8,
    latent_offset, latent_scale, section_total_bytes)``.

    Raises ``ValueError`` on bad magic / truncated payload.
    """
    import numpy as np

    if len(data) < Z3HV2_HEADER_STRUCT.size:
        raise ValueError("Z3HV2 section too short for header")
    fields = Z3HV2_HEADER_STRUCT.unpack_from(data, 0)
    magic, version, n_pairs, hyper_dim, latent_dim = fields[:5]
    int8_w_scale, quant_step, min_sigma, max_sigma, factorized_half = fields[5:10]
    if magic != Z3HV2_MAGIC:
        raise ValueError(f"bad Z3HV2 magic: {magic!r}")
    if version != Z3HV2_VERSION:
        raise ValueError(f"unsupported Z3HV2 version: {version}")
    if int(n_pairs) != A1_N_PAIRS:
        raise ValueError(f"Z3HV2 n_pairs {int(n_pairs)} != A1_N_PAIRS {A1_N_PAIRS}")
    if int(latent_dim) != A1_LATENT_DIM:
        raise ValueError(
            f"Z3HV2 latent_dim {int(latent_dim)} != A1_LATENT_DIM {A1_LATENT_DIM}"
        )

    meta = Z3HV2SectionMeta(
        n_pairs=int(n_pairs),
        hyper_dim=int(hyper_dim),
        latent_dim=int(latent_dim),
        int8_w_scale=float(int8_w_scale),
        quant_step=float(quant_step),
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        factorized_half_range=float(factorized_half),
    )

    pos = Z3HV2_HEADER_STRUCT.size
    if pos + 2 > len(data):
        raise ValueError("Z3HV2 truncated before weights length prefix")
    (weights_len,) = struct.unpack_from("<H", data, pos)
    pos += 2
    if pos + weights_len > len(data):
        raise ValueError("Z3HV2 truncated mid-weights blob")
    weights_int8 = (
        b"" if weights_len == 0 else brotli.decompress(data[pos : pos + weights_len])
    )
    pos += weights_len

    if pos + 4 > len(data):
        raise ValueError("Z3HV2 truncated before w_hat length prefix")
    (w_hat_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if pos + w_hat_len > len(data):
        raise ValueError("Z3HV2 truncated mid-w_hat blob")
    w_hat_int8 = b"" if w_hat_len == 0 else brotli.decompress(data[pos : pos + w_hat_len])
    pos += w_hat_len
    if (len(weights_int8) == 0) != (len(w_hat_int8) == 0):
        raise ValueError("Z3HV2 weights/w_hat presence mismatch")
    if len(w_hat_int8) and len(w_hat_int8) != meta.n_pairs * meta.hyper_dim:
        raise ValueError(
            f"Z3HV2 w_hat decoded {len(w_hat_int8)} != "
            f"n_pairs*hyper_dim {meta.n_pairs * meta.hyper_dim}"
        )

    if pos + 4 > len(data):
        raise ValueError("Z3HV2 truncated before residual length prefix")
    (residual_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if pos + residual_len > len(data):
        raise ValueError("Z3HV2 truncated mid-residual blob")
    residual_int8 = brotli.decompress(data[pos : pos + residual_len])
    pos += residual_len
    if len(residual_int8) != meta.n_pairs * meta.latent_dim:
        raise ValueError(
            f"Z3HV2 residual decoded {len(residual_int8)} != "
            f"n_pairs*latent_dim {meta.n_pairs * meta.latent_dim}"
        )

    if pos + Z3HV2_PER_DIM_AFFINE_LEN > len(data):
        raise ValueError("Z3HV2 truncated mid-affine bytes")
    affine_bytes = data[pos : pos + Z3HV2_PER_DIM_AFFINE_LEN]
    pos += Z3HV2_PER_DIM_AFFINE_LEN
    half = Z3HV2_PER_DIM_AFFINE_LEN // 2
    latent_offset = torch.from_numpy(
        np.frombuffer(affine_bytes[:half], dtype=np.float32).copy()
    )
    latent_scale = torch.from_numpy(
        np.frombuffer(affine_bytes[half:], dtype=np.float32).copy()
    )

    return meta, weights_int8, w_hat_int8, residual_int8, latent_offset, latent_scale, pos


def build_z3v2_payload_bytes(
    *,
    a1_bytes: bytes,
    z3hv2_section: bytes,
) -> bytes:
    """Construct the v2 inner payload by REPLACING A1's latent_blob.

    Layout (LE-everywhere)::

        a1_bytes[:A1_DECODER_SECTION_TOTAL]                 (162168 B; verbatim)
        z3hv2_section                                       (NEW; replaces 15387 B)
        a1_bytes[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN:]  (sidecar; verbatim)

    Raises ``ValueError`` if A1 bytes are too short to host the layout.
    """
    if len(a1_bytes) < A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN:
        raise ValueError(
            f"a1_bytes too short for v2 layout: {len(a1_bytes)} < "
            f"{A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN}"
        )
    _validate_a1_decoder_section_header(a1_bytes)
    if z3hv2_section[: len(Z3HV2_MAGIC)] != Z3HV2_MAGIC:
        raise ValueError(
            f"z3hv2_section does not start with magic {Z3HV2_MAGIC!r}"
        )
    decoder_section = a1_bytes[:A1_DECODER_SECTION_TOTAL]
    sidecar_section = a1_bytes[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN :]
    return decoder_section + z3hv2_section + sidecar_section


def split_z3v2_payload_bytes(payload_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    """Split a v2 payload into (decoder_section, z3hv2_section, sidecar_section).

    Returns the three concrete byte slices. Raises ``ValueError`` if the
    payload is not in v2 layout (no Z3HV2 magic at offset 162168).
    """
    if len(payload_bytes) < A1_DECODER_SECTION_TOTAL + Z3HV2_HEADER_STRUCT.size:
        raise ValueError("payload too short for v2 layout")
    _validate_a1_decoder_section_header(payload_bytes)
    decoder_section = payload_bytes[:A1_DECODER_SECTION_TOTAL]
    if (
        payload_bytes[
            A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + len(Z3HV2_MAGIC)
        ]
        != Z3HV2_MAGIC
    ):
        raise ValueError("missing Z3HV2 magic at offset A1_DECODER_SECTION_TOTAL")
    # Decode just to learn the section length; the consumer should call
    # decode_z3hv2_section() to actually use the data.
    _, _, _, _, _, _, section_total_bytes = decode_z3hv2_section(
        payload_bytes[A1_DECODER_SECTION_TOTAL:]
    )
    z3hv2_section = payload_bytes[
        A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + section_total_bytes
    ]
    sidecar_section = payload_bytes[
        A1_DECODER_SECTION_TOTAL + section_total_bytes :
    ]
    return decoder_section, z3hv2_section, sidecar_section


def build_z3v2_composition_archive_contract(
    a1_bytes: bytes,
    z3v2_payload_bytes: bytes,
) -> Z3V2CompositionArchiveContract:
    """Build the fail-closed typed contract for a Z3 v2 inner archive payload.

    A v2 payload IS the operational latent-replacement layout: the byte_saving
    flag reflects whether the Z3HV2 section is smaller than the A1 latent_blob
    it replaces. Per CLAUDE.md "Apples-to-apples evidence discipline", these
    flags are descriptive only — score_claim / promotion_eligible /
    exact_eval_ready remain false until paired CUDA + CPU auth-evals adjudicate.
    """
    if len(z3v2_payload_bytes) < A1_DECODER_SECTION_TOTAL + Z3HV2_HEADER_STRUCT.size:
        raise ValueError("z3v2 payload too short")
    _validate_a1_decoder_section_header(a1_bytes)
    _validate_a1_decoder_section_header(z3v2_payload_bytes)
    if z3v2_payload_bytes[:A1_DECODER_SECTION_TOTAL] != a1_bytes[:A1_DECODER_SECTION_TOTAL]:
        raise ValueError("z3v2 payload decoder section diverges from A1")
    decoder_section, z3hv2_section, sidecar_section = split_z3v2_payload_bytes(
        z3v2_payload_bytes
    )
    a1_sidecar_section = a1_bytes[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN :]
    if sidecar_section != a1_sidecar_section:
        raise ValueError("z3v2 payload sidecar diverges from A1 sidecar")
    z3v2_section_bytes = len(z3hv2_section)
    byte_savings = A1_LATENT_BLOB_LEN - z3v2_section_bytes
    byte_saving = byte_savings > 0
    blockers = (
        "z3v2_score_claim_requires_paired_cuda_cpu_auth_eval",
        "result_review_required_before_promotion",
    )
    return Z3V2CompositionArchiveContract(
        payload_bytes=z3v2_payload_bytes,
        layout="z3v2_latent_replacement",
        base_archive_bytes=len(a1_bytes),
        z3v2_section_bytes=z3v2_section_bytes,
        a1_latent_blob_bytes_replaced=A1_LATENT_BLOB_LEN,
        archive_bytes=len(z3v2_payload_bytes),
        byte_saving=byte_saving,
        byte_savings_bytes=max(byte_savings, 0),
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        exact_eval_ready=False,
        result_review_blockers=blockers,
    )


__all__ = [
    "A1_DECODER_BLOB_LEN",
    "A1_DECODER_SECTION_TOTAL",
    "A1_LATENT_BLOB_LEN",
    "A1_SECTION_TOTAL_PREFIX_LEN",
    "Z3HV2_BYTE_OFFSET_AFTER_DECODER",
    "Z3HV2_HEADER_STRUCT",
    "Z3HV2_MAGIC",
    "Z3HV2_PER_DIM_AFFINE_LEN",
    "Z3HV2_VERSION",
    "Z3HV2SectionMeta",
    "Z3V2CompositionArchiveContract",
    "build_z3v2_composition_archive_contract",
    "build_z3v2_payload_bytes",
    "decode_z3hv2_section",
    "encode_z3hv2_section",
    "split_z3v2_payload_bytes",
]
