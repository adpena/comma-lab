# SPDX-License-Identifier: MIT
"""Z3-G1 entropy-coded v2 archive grammar.

Per `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md` +
the F1 codex finding empirical confirmation: the v1 Z3HV2 wire grammar
ships `hyperprior_weights_int8` + `w_hat_int8` slots empty (b"") in
production-safe direct-residual mode. v2 introduces a NEW magic +
grammar (`Z3G2`) that REPLACES those empty slots with TWO actual
entropy-coded streams:

    sigma_table_blob: brotli-compressed 5x28 int8 sigma table (~300B)
    class_prior_cdf:  5 * uint16 = 10B raw (frequency counts; smoothed)
    class_index_blob: constriction-Huffman encoded 600 class indices (~200-400B)

These bytes ARE consumed by the inflate path (see `inflate_consumer.py`)
to reconstruct per-pair sigmas + class indices and feed them into the
conditional Gaussian AC decoder for the residual.

Per Catalog #220 the score_improvement_mechanism is OPERATIONAL because
the bytes flow through the inflate runtime and produce different inflate
outputs (different class-conditional sigmas → different decoded residual →
different latents → different decoded frames). This is verified
structurally by `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py`.

Z3G2 wire format (the bytes between offset 162168 and the trailing A1 sidecar)::

    magic               : 4 bytes ASCII "Z3G2"   (distinguishes from v1 "Z3V2")
    version             : uint8 (== 1)
    n_pairs             : uint16 LE (== 600 == A1_N_PAIRS)
    num_scorer_classes  : uint8 (== 5)
    latent_dim          : uint8 (== A1_LATENT_DIM == 28)
    int8_sigma_scale    : float32 LE (4 B; sigma_int8 -> sigma_real scale)
    quant_step          : float32 LE (4 B; latent residual quantization Δ)
    min_sigma           : float32 LE (4 B; sigma clamp lower bound)
    max_sigma           : float32 LE (4 B; sigma clamp upper bound)
    reserved            : 2 B (== 0)
    --- Header total = 4 + 1 + 2 + 1 + 1 + 4 + 4 + 4 + 4 + 2 = 27 B
    sigma_table_len     : uint16 LE (2 B; brotli-compressed sigma table length)
    sigma_table_blob    : <sigma_table_len> bytes
    class_prior_blob    : 5 * uint16 LE = 10 B (frequency counts)
    class_index_len     : uint32 LE (4 B; constriction-encoded class index byte length)
    class_index_blob    : <class_index_len> bytes
    residual_blob_len   : uint32 LE (4 B; brotli-compressed AC-coded latent residual)
    residual_blob       : <residual_blob_len> bytes
    latent_offset_blob  : 28 * float32 = 112 B
    latent_scale_blob   : 28 * float32 = 112 B

The trailing 224 B per-dim affine encodes `(offset, scale)` for re-mapping
residual_q back into A1's quantized-range space (same as Z3HV2).

LOC budget: <= 350 LOC per HNeRV parity discipline L7. v2 IS a bolt-on
(A1 weights frozen); substrate-engineering exemption NOT used.

NO score claim. NO promotion. NO exact-eval dispatch from this module.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import brotli
import constriction
import numpy as np
import torch

from tac.substrates.z3_g1_entropy_coded_v2.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
)

# v1 Z3HV2 magic was "Z3V2"; v2 uses "Z3G2" so the inflate path can fork
# on first 4 bytes after offset A1_DECODER_SECTION_TOTAL.
Z3G2_MAGIC = b"Z3G2"
Z3G2_VERSION = 1

# magic(4s) + version(B) + n_pairs(H) + num_scorer_classes(B) + latent_dim(B)
# + int8_sigma_scale(f) + quant_step(f) + min_sigma(f) + max_sigma(f)
# + reserved(2s) = 4+1+2+1+1+4+4+4+4+2 = 27
Z3G2_HEADER_STRUCT = struct.Struct("<4sBHBBffff2s")

# A1 wire-format constants (mirrored from submissions/a1/src/codec.py).
A1_DECODER_BLOB_LEN = 162_164
A1_LATENT_BLOB_LEN = 15_387
A1_SECTION_TOTAL_PREFIX_LEN = 4
A1_DECODER_SECTION_TOTAL = A1_SECTION_TOTAL_PREFIX_LEN + A1_DECODER_BLOB_LEN  # 162168

Z3G2_BYTE_OFFSET_AFTER_DECODER = A1_DECODER_SECTION_TOTAL  # 162168
Z3G2_PER_DIM_AFFINE_LEN = 4 * A1_LATENT_DIM * 2  # 224 B (offset + scale, fp32)
Z3G2_CLASS_PRIOR_BLOB_LEN = G1_NUM_SCORER_CLASSES * 2  # 10 B (5 * uint16 LE)


@dataclass(frozen=True)
class Z3G2EntropyCodedSectionMeta:
    """Decoded Z3G2 header metadata."""

    n_pairs: int
    num_scorer_classes: int
    latent_dim: int
    int8_sigma_scale: float
    quant_step: float
    min_sigma: float
    max_sigma: float


@dataclass(frozen=True)
class Z3G2EntropyCodedCompositionArchiveContract:
    """Typed authority contract for a Z3G2 composition payload.

    Mirrors v1's ``Z3V2CompositionArchiveContract`` but for the v2 wire
    grammar. score_claim / promotion_eligible / exact_eval_ready remain
    false until paired CUDA + CPU auth-evals adjudicate per CLAUDE.md
    "Apples-to-apples evidence discipline".
    """

    payload_bytes: bytes
    layout: str
    base_archive_bytes: int
    z3g2_section_bytes: int
    a1_latent_blob_bytes_replaced: int
    archive_bytes: int
    byte_saving: bool
    byte_savings_bytes: int
    distinguishing_feature_bytes: int
    """Total bytes ACTUALLY shipped that distinguish this packet from Z3 v2.
    Equals (sigma_table_blob_len + class_prior_blob_len + class_index_blob_len).
    For F1-class regression detection: this MUST be > 0 in production-safe
    mode (was 0 in v1 because empty slots → identical packets to Z3 v2)."""
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
            "z3g2_section_bytes": self.z3g2_section_bytes,
            "a1_latent_blob_bytes_replaced": self.a1_latent_blob_bytes_replaced,
            "archive_bytes": self.archive_bytes,
            "byte_saving": self.byte_saving,
            "byte_savings_bytes": self.byte_savings_bytes,
            "distinguishing_feature_bytes": self.distinguishing_feature_bytes,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "exact_eval_ready": self.exact_eval_ready,
            "result_review_blockers": list(self.result_review_blockers),
        }


def _validate_a1_decoder_section_header(a1_bytes: bytes) -> None:
    """Require the exact A1 decoder section boundary before Z3G2 splicing."""
    if len(a1_bytes) < A1_SECTION_TOTAL_PREFIX_LEN:
        raise ValueError("a1_bytes too short for A1 section_total prefix")
    (section_total,) = struct.unpack_from("<I", a1_bytes, 0)
    if int(section_total) != A1_DECODER_SECTION_TOTAL:
        raise ValueError(
            "A1 decoder section_total mismatch: "
            f"{int(section_total)} != {A1_DECODER_SECTION_TOTAL}"
        )


def _encode_class_indices_huffman(
    class_indices_uint8: bytes,
    class_prior_counts: np.ndarray,
) -> bytes:
    """Encode class indices via constriction's Huffman QueueEncoder.

    Uses Huffman coding (not range coding) since constriction.symbol.QueueEncoder
    + huffman.EncoderHuffmanTree is the simplest stable AC API in constriction
    0.4.x. Class entropy is small (5 symbols × 600 pairs) so Huffman near-optimal
    matches arithmetic to within ~5%.

    The encoded bytes are the uint32 word array packed as little-endian bytes
    plus a 4-byte uint32 prefix giving the original class index count (so
    the decoder knows how many symbols to extract, since Huffman is variable-
    length).
    """
    if len(class_indices_uint8) == 0:
        # Encode empty as 4-byte 0 prefix + 0 words.
        return struct.pack("<I", 0)
    counts_f = class_prior_counts.astype(np.float64)
    counts_f = np.maximum(counts_f, 1.0)
    probs = counts_f / counts_f.sum()
    tree = constriction.symbol.huffman.EncoderHuffmanTree(probs)
    encoder = constriction.symbol.QueueEncoder()
    for sym in class_indices_uint8:
        encoder.encode_symbol(int(sym), tree)
    word_arr, _bits = encoder.get_compressed_and_bitrate()
    # Word array is uint32; pack count + words.
    count_prefix = struct.pack("<I", len(class_indices_uint8))
    word_bytes = word_arr.astype("<u4").tobytes()
    return count_prefix + word_bytes


def _decode_class_indices_huffman(
    encoded_bytes: bytes,
    class_prior_counts: np.ndarray,
) -> bytes:
    """Decode class indices via constriction's Huffman QueueDecoder.

    Inverse of ``_encode_class_indices_huffman``; returns the per-pair
    class indices as a uint8 byte string of length n_pairs.
    """
    if len(encoded_bytes) < 4:
        raise ValueError(
            f"class_index_blob too short for count prefix: {len(encoded_bytes)} < 4"
        )
    (n_pairs,) = struct.unpack_from("<I", encoded_bytes, 0)
    if n_pairs == 0:
        return b""
    word_bytes = encoded_bytes[4:]
    if len(word_bytes) % 4 != 0:
        raise ValueError(
            f"class_index_blob word section length {len(word_bytes)} not divisible by 4"
        )
    word_arr = np.frombuffer(word_bytes, dtype="<u4").copy()
    counts_f = class_prior_counts.astype(np.float64)
    counts_f = np.maximum(counts_f, 1.0)
    probs = counts_f / counts_f.sum()
    tree = constriction.symbol.huffman.DecoderHuffmanTree(probs)
    decoder = constriction.symbol.QueueDecoder(word_arr)
    decoded = bytearray()
    for _ in range(int(n_pairs)):
        decoded.append(int(decoder.decode_symbol(tree)) & 0xFF)
    return bytes(decoded)


def encode_z3g2_section(
    *,
    sigma_table_int8: torch.Tensor,
    class_indices_uint8: bytes,
    class_prior_counts: torch.Tensor,
    residual_int8: bytes,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
    int8_sigma_scale: float,
    quant_step: float,
    min_sigma: float,
    max_sigma: float,
    n_pairs: int = A1_N_PAIRS,
    num_scorer_classes: int = G1_NUM_SCORER_CLASSES,
    latent_dim: int = A1_LATENT_DIM,
) -> bytes:
    """Pack the Z3G2 entropy-coded section.

    Args:
        sigma_table_int8: (num_scorer_classes, latent_dim) int8 sigma table.
        class_indices_uint8: ``n_pairs`` raw uint8 class indices.
        class_prior_counts: (num_scorer_classes,) int64 frequency counts.
        residual_int8: ``n_pairs * latent_dim`` int8 latent residuals.
        latent_offset: (latent_dim,) per-dim affine offset (fp32).
        latent_scale: (latent_dim,) per-dim scale (fp32).
        int8_sigma_scale: scale s.t. sigma_int8 / 127 * scale = sigma_real.
        quant_step: quantization step Δ for the conditional-Gaussian AC coder.
        min_sigma, max_sigma: bounds used at encoding-time.
        n_pairs: Number of pairs (default A1_N_PAIRS = 600).
        num_scorer_classes: Number of distinct classes (default 5).
        latent_dim: Latent dim (default A1_LATENT_DIM = 28).

    Returns:
        Section bytes (header + length-prefixed brotli/AC blobs + per-dim affine).
    """
    if n_pairs != A1_N_PAIRS:
        raise ValueError(f"n_pairs must be {A1_N_PAIRS}; got {n_pairs}")
    if latent_dim != A1_LATENT_DIM:
        raise ValueError(f"latent_dim must be {A1_LATENT_DIM}; got {latent_dim}")
    if num_scorer_classes != G1_NUM_SCORER_CLASSES:
        raise ValueError(
            f"num_scorer_classes must be {G1_NUM_SCORER_CLASSES}; got {num_scorer_classes}"
        )
    if sigma_table_int8.shape != (num_scorer_classes, latent_dim):
        raise ValueError(
            f"sigma_table_int8 must be ({num_scorer_classes}, {latent_dim}); "
            f"got {tuple(sigma_table_int8.shape)}"
        )
    if sigma_table_int8.dtype != torch.int8:
        raise ValueError(
            f"sigma_table_int8 must be int8; got {sigma_table_int8.dtype}"
        )
    if len(class_indices_uint8) != n_pairs:
        raise ValueError(
            f"class_indices_uint8 length {len(class_indices_uint8)} != n_pairs {n_pairs}"
        )
    if class_prior_counts.shape != (num_scorer_classes,):
        raise ValueError(
            f"class_prior_counts must be ({num_scorer_classes},); "
            f"got {tuple(class_prior_counts.shape)}"
        )
    if class_prior_counts.dtype not in (torch.int64, torch.int32):
        raise ValueError(
            f"class_prior_counts must be int dtype; got {class_prior_counts.dtype}"
        )
    if class_prior_counts.max().item() > 0xFFFF:
        raise ValueError(
            f"class_prior_counts overflow uint16: max={class_prior_counts.max().item()}"
        )
    if class_prior_counts.min().item() < 0:
        raise ValueError(
            f"class_prior_counts must be non-negative; got min={class_prior_counts.min().item()}"
        )
    if len(residual_int8) != n_pairs * latent_dim:
        raise ValueError(
            f"residual_int8 length {len(residual_int8)} != n_pairs*latent_dim "
            f"{n_pairs * latent_dim}"
        )
    if float(quant_step) != 1.0:
        raise ValueError(
            "Z3G2 quant_step must be 1.0 until train/export/inflate all "
            "apply non-unit residual quantization consistently"
        )
    if latent_offset.shape != (latent_dim,) or latent_scale.shape != (latent_dim,):
        raise ValueError(
            f"latent_offset and latent_scale must be ({latent_dim},); got "
            f"{tuple(latent_offset.shape)} and {tuple(latent_scale.shape)}"
        )

    header = Z3G2_HEADER_STRUCT.pack(
        Z3G2_MAGIC,
        Z3G2_VERSION,
        n_pairs,
        num_scorer_classes,
        latent_dim,
        float(int8_sigma_scale),
        float(quant_step),
        float(min_sigma),
        float(max_sigma),
        b"\x00\x00",
    )

    sigma_table_bytes = sigma_table_int8.detach().cpu().numpy().tobytes()
    sigma_table_compressed = brotli.compress(sigma_table_bytes, quality=11)
    if len(sigma_table_compressed) > 0xFFFF:
        raise ValueError(
            f"sigma_table_blob too large: {len(sigma_table_compressed)} > 65535"
        )

    class_prior_arr = class_prior_counts.detach().cpu().numpy().astype("<u2")
    class_prior_bytes = class_prior_arr.tobytes()
    if len(class_prior_bytes) != Z3G2_CLASS_PRIOR_BLOB_LEN:
        raise ValueError(
            f"class_prior_bytes length {len(class_prior_bytes)} != "
            f"{Z3G2_CLASS_PRIOR_BLOB_LEN}"
        )

    class_index_compressed = _encode_class_indices_huffman(
        class_indices_uint8, class_prior_arr.astype(np.int64)
    )

    residual_compressed = brotli.compress(residual_int8, quality=11)

    affine_bytes = (
        latent_offset.detach().cpu().to(torch.float32).numpy().tobytes()
        + latent_scale.detach().cpu().to(torch.float32).numpy().tobytes()
    )
    if len(affine_bytes) != Z3G2_PER_DIM_AFFINE_LEN:
        raise ValueError(
            f"per-dim affine bytes length {len(affine_bytes)} != "
            f"{Z3G2_PER_DIM_AFFINE_LEN}"
        )

    return (
        header
        + struct.pack("<H", len(sigma_table_compressed))
        + sigma_table_compressed
        + class_prior_bytes
        + struct.pack("<I", len(class_index_compressed))
        + class_index_compressed
        + struct.pack("<I", len(residual_compressed))
        + residual_compressed
        + affine_bytes
    )


def decode_z3g2_section(
    data: bytes,
) -> tuple[
    Z3G2EntropyCodedSectionMeta,
    torch.Tensor,
    bytes,
    torch.Tensor,
    bytes,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Unpack the Z3G2 section.

    Returns ``(meta, sigma_table_int8, class_indices_uint8, class_prior_counts,
    residual_int8, latent_offset, latent_scale, section_total_bytes)``.

    Raises ``ValueError`` on bad magic / truncated payload.
    """
    if len(data) < Z3G2_HEADER_STRUCT.size:
        raise ValueError("Z3G2 section too short for header")
    fields = Z3G2_HEADER_STRUCT.unpack_from(data, 0)
    magic, version, n_pairs, num_scorer_classes, latent_dim = fields[:5]
    int8_sigma_scale, quant_step, min_sigma, max_sigma = fields[5:9]
    if magic != Z3G2_MAGIC:
        raise ValueError(f"bad Z3G2 magic: {magic!r}")
    if version != Z3G2_VERSION:
        raise ValueError(f"unsupported Z3G2 version: {version}")
    if int(n_pairs) != A1_N_PAIRS:
        raise ValueError(f"Z3G2 n_pairs {int(n_pairs)} != A1_N_PAIRS {A1_N_PAIRS}")
    if int(latent_dim) != A1_LATENT_DIM:
        raise ValueError(
            f"Z3G2 latent_dim {int(latent_dim)} != A1_LATENT_DIM {A1_LATENT_DIM}"
        )
    if int(num_scorer_classes) != G1_NUM_SCORER_CLASSES:
        raise ValueError(
            f"Z3G2 num_scorer_classes {int(num_scorer_classes)} != "
            f"G1_NUM_SCORER_CLASSES {G1_NUM_SCORER_CLASSES}"
        )

    meta = Z3G2EntropyCodedSectionMeta(
        n_pairs=int(n_pairs),
        num_scorer_classes=int(num_scorer_classes),
        latent_dim=int(latent_dim),
        int8_sigma_scale=float(int8_sigma_scale),
        quant_step=float(quant_step),
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
    )

    pos = Z3G2_HEADER_STRUCT.size

    if pos + 2 > len(data):
        raise ValueError("Z3G2 truncated before sigma_table length prefix")
    (sigma_table_len,) = struct.unpack_from("<H", data, pos)
    pos += 2
    if pos + sigma_table_len > len(data):
        raise ValueError("Z3G2 truncated mid-sigma_table blob")
    sigma_table_bytes = brotli.decompress(data[pos : pos + sigma_table_len])
    pos += sigma_table_len
    sigma_table_arr = np.frombuffer(sigma_table_bytes, dtype=np.int8).reshape(
        meta.num_scorer_classes, meta.latent_dim
    )
    sigma_table_int8 = torch.from_numpy(sigma_table_arr.copy())

    if pos + Z3G2_CLASS_PRIOR_BLOB_LEN > len(data):
        raise ValueError("Z3G2 truncated mid-class_prior blob")
    class_prior_bytes = data[pos : pos + Z3G2_CLASS_PRIOR_BLOB_LEN]
    pos += Z3G2_CLASS_PRIOR_BLOB_LEN
    class_prior_arr = np.frombuffer(class_prior_bytes, dtype="<u2")
    class_prior_counts = torch.from_numpy(class_prior_arr.astype(np.int64).copy())

    if pos + 4 > len(data):
        raise ValueError("Z3G2 truncated before class_index length prefix")
    (class_index_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if pos + class_index_len > len(data):
        raise ValueError("Z3G2 truncated mid-class_index blob")
    class_indices_uint8 = _decode_class_indices_huffman(
        data[pos : pos + class_index_len],
        class_prior_arr.astype(np.int64),
    )
    pos += class_index_len
    if len(class_indices_uint8) != meta.n_pairs:
        raise ValueError(
            f"Z3G2 class_indices decoded {len(class_indices_uint8)} != n_pairs {meta.n_pairs}"
        )

    if pos + 4 > len(data):
        raise ValueError("Z3G2 truncated before residual length prefix")
    (residual_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if pos + residual_len > len(data):
        raise ValueError("Z3G2 truncated mid-residual blob")
    residual_int8 = brotli.decompress(data[pos : pos + residual_len])
    pos += residual_len
    if len(residual_int8) != meta.n_pairs * meta.latent_dim:
        raise ValueError(
            f"Z3G2 residual decoded {len(residual_int8)} != "
            f"n_pairs*latent_dim {meta.n_pairs * meta.latent_dim}"
        )

    if pos + Z3G2_PER_DIM_AFFINE_LEN > len(data):
        raise ValueError("Z3G2 truncated mid-affine bytes")
    affine_bytes = data[pos : pos + Z3G2_PER_DIM_AFFINE_LEN]
    pos += Z3G2_PER_DIM_AFFINE_LEN
    half = Z3G2_PER_DIM_AFFINE_LEN // 2
    latent_offset = torch.from_numpy(
        np.frombuffer(affine_bytes[:half], dtype=np.float32).copy()
    )
    latent_scale = torch.from_numpy(
        np.frombuffer(affine_bytes[half:], dtype=np.float32).copy()
    )

    return (
        meta,
        sigma_table_int8,
        class_indices_uint8,
        class_prior_counts,
        residual_int8,
        latent_offset,
        latent_scale,
        pos,
    )


def is_z3g2_payload(payload_bytes: bytes) -> bool:
    """True iff the bytes start with the Z3G2 layout (Z3G2 magic at decoder boundary)."""
    if len(payload_bytes) < A1_DECODER_SECTION_TOTAL + len(Z3G2_MAGIC):
        return False
    return (
        payload_bytes[
            A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + len(Z3G2_MAGIC)
        ]
        == Z3G2_MAGIC
    )


def build_z3g2_payload_bytes(
    *,
    a1_bytes: bytes,
    z3g2_section: bytes,
) -> bytes:
    """Construct the v2 inner payload by REPLACING A1's latent_blob.

    Layout (LE-everywhere)::

        a1_bytes[:A1_DECODER_SECTION_TOTAL]                       (162168 B; verbatim)
        z3g2_section                                              (NEW; replaces 15387 B)
        a1_bytes[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN:]  (sidecar; verbatim)

    Raises ``ValueError`` if A1 bytes are too short to host the layout.
    """
    if len(a1_bytes) < A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN:
        raise ValueError(
            f"a1_bytes too short for v2 layout: {len(a1_bytes)} < "
            f"{A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN}"
        )
    _validate_a1_decoder_section_header(a1_bytes)
    if z3g2_section[: len(Z3G2_MAGIC)] != Z3G2_MAGIC:
        raise ValueError(
            f"z3g2_section does not start with magic {Z3G2_MAGIC!r}"
        )
    decoder_section = a1_bytes[:A1_DECODER_SECTION_TOTAL]
    sidecar_section = a1_bytes[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN :]
    return decoder_section + z3g2_section + sidecar_section


def split_z3g2_payload_bytes(payload_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    """Split a v2 payload into (decoder_section, z3g2_section, sidecar_section).

    Raises ``ValueError`` if the payload is not in v2 layout.
    """
    if len(payload_bytes) < A1_DECODER_SECTION_TOTAL + Z3G2_HEADER_STRUCT.size:
        raise ValueError("payload too short for v2 layout")
    _validate_a1_decoder_section_header(payload_bytes)
    decoder_section = payload_bytes[:A1_DECODER_SECTION_TOTAL]
    if (
        payload_bytes[
            A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + len(Z3G2_MAGIC)
        ]
        != Z3G2_MAGIC
    ):
        raise ValueError("missing Z3G2 magic at offset A1_DECODER_SECTION_TOTAL")
    _, _, _, _, _, _, _, section_total_bytes = decode_z3g2_section(
        payload_bytes[A1_DECODER_SECTION_TOTAL:]
    )
    z3g2_section = payload_bytes[
        A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + section_total_bytes
    ]
    sidecar_section = payload_bytes[
        A1_DECODER_SECTION_TOTAL + section_total_bytes :
    ]
    return decoder_section, z3g2_section, sidecar_section


def build_z3g2_composition_archive_contract(
    a1_bytes: bytes,
    z3g2_payload_bytes: bytes,
) -> Z3G2EntropyCodedCompositionArchiveContract:
    """Build the fail-closed typed contract for a Z3G2 inner archive payload.

    Per CLAUDE.md "Apples-to-apples evidence discipline", these flags are
    descriptive only — score_claim / promotion_eligible / exact_eval_ready
    remain false until paired CUDA + CPU auth-evals adjudicate.
    """
    if len(z3g2_payload_bytes) < A1_DECODER_SECTION_TOTAL + Z3G2_HEADER_STRUCT.size:
        raise ValueError("z3g2 payload too short")
    _validate_a1_decoder_section_header(a1_bytes)
    _validate_a1_decoder_section_header(z3g2_payload_bytes)
    if z3g2_payload_bytes[:A1_DECODER_SECTION_TOTAL] != a1_bytes[:A1_DECODER_SECTION_TOTAL]:
        raise ValueError("z3g2 payload decoder section diverges from A1")
    decoder_section, z3g2_section, sidecar_section = split_z3g2_payload_bytes(
        z3g2_payload_bytes
    )
    a1_sidecar_section = a1_bytes[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN :]
    if sidecar_section != a1_sidecar_section:
        raise ValueError("z3g2 payload sidecar diverges from A1 sidecar")
    z3g2_section_bytes = len(z3g2_section)
    byte_savings = A1_LATENT_BLOB_LEN - z3g2_section_bytes
    byte_saving = byte_savings > 0

    # Compute distinguishing-feature byte count (for F1-class regression detection).
    # Re-decode to extract individual blob lengths.
    (
        _,
        sigma_table_t,
        class_indices_b,
        class_prior_t,
        _,
        _,
        _,
        _,
    ) = decode_z3g2_section(z3g2_section)
    # The bytes that distinguish v2 from a Z3 v2 packet (which would have empty
    # sigma_table / class_index slots): sum of sigma_table_compressed + class_prior +
    # class_index_compressed lengths. Re-encode the sigma table to get its compressed
    # length (this is canonical because the encoder is deterministic given inputs).
    sigma_table_bytes = sigma_table_t.detach().cpu().numpy().tobytes()
    sigma_table_compressed_len = len(brotli.compress(sigma_table_bytes, quality=11))
    class_prior_arr = class_prior_t.detach().cpu().numpy().astype("<u2")
    class_index_compressed_len = len(
        _encode_class_indices_huffman(class_indices_b, class_prior_arr.astype(np.int64))
    )
    distinguishing_feature_bytes = (
        sigma_table_compressed_len
        + Z3G2_CLASS_PRIOR_BLOB_LEN
        + class_index_compressed_len
    )

    blockers = (
        "z3g2_score_claim_requires_paired_cuda_cpu_auth_eval",
        "result_review_required_before_promotion",
        "byte_mutation_smoke_must_pass_per_catalog_139_before_promotion",
    )
    return Z3G2EntropyCodedCompositionArchiveContract(
        payload_bytes=z3g2_payload_bytes,
        layout="z3g2_entropy_coded_latent_replacement",
        base_archive_bytes=len(a1_bytes),
        z3g2_section_bytes=z3g2_section_bytes,
        a1_latent_blob_bytes_replaced=A1_LATENT_BLOB_LEN,
        archive_bytes=len(z3g2_payload_bytes),
        byte_saving=byte_saving,
        byte_savings_bytes=max(byte_savings, 0),
        distinguishing_feature_bytes=distinguishing_feature_bytes,
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
    "Z3G2_BYTE_OFFSET_AFTER_DECODER",
    "Z3G2_CLASS_PRIOR_BLOB_LEN",
    "Z3G2_HEADER_STRUCT",
    "Z3G2_MAGIC",
    "Z3G2_PER_DIM_AFFINE_LEN",
    "Z3G2_VERSION",
    "Z3G2EntropyCodedCompositionArchiveContract",
    "Z3G2EntropyCodedSectionMeta",
    "build_z3g2_composition_archive_contract",
    "build_z3g2_payload_bytes",
    "decode_z3g2_section",
    "encode_z3g2_section",
    "is_z3g2_payload",
    "split_z3g2_payload_bytes",
]
