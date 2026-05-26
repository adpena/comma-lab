# SPDX-License-Identifier: MIT
"""BPR1 Variant B-d codec: sign-bitmap + per-pair magnitude (frontier-push).

Per operator NON-NEGOTIABLE 2026-05-26 cascade follow-up to gain_clamp sweep
landing memo (`.omx/research/boostnerv_pr110_gain_clamp_sweep_landed_20260526.md`
commit `8240aceda`). The L1 BPR1 codec's int8 quantizer is empirically
SCALE-INVARIANT — 42 bytes constant across all 9 (gain_clamp, epochs) cells.

ROOT-CAUSE empirical discovery (sweep heatmap re-inspection):
- `clamped_fraction_at_boundary = 1.0000` at ALL 9 cells (100% saturation).
- `residual_blob_uncompressed_bytes = 1,843,200` (50 pairs × 96 × 128 × 3).
- `residual_blob_brotli_bytes      = 14` (ratio 7.6e-6) at ALL 9 cells.
- Mechanism: tanh-bounded residual head + L2 loss → optimizer drives residuals
  to ±tanh-asymptote → clip(±gain_clamp) truncates 100% → int8 quantizer
  produces ±127 uniformly → brotli RLE-collapses to ~14B.

The information content is ENTIRELY in the SIGN bitmap. The int8 quantizer
discards no information — but it also exposes no information because brotli
RLE-collapses the uniform stream.

Variant B-d design (FRONTIER-PUSH per `feedback_pushing_the_frontier_of_
research_on_optimization_algorithms_standing_directive_20260526.md`):

1. Extract sign-bitmap: `sign_bit = (residual_clamped >= 0)` → 1 bit/pixel.
2. Pack 8 sign-bits per byte via numpy `packbits` (big-endian per numpy spec).
3. Per-pair magnitude scalar = gain_clamp value (one fp16 per pair = 2B/pair).
4. Apply brotli quality 9 to the packed sign-bitmap (entropy depends on per-
   pair sign-pattern correlation; sweeps will reveal whether this grows with
   gain_clamp).
5. Sidecar composition:
   [BPR1 header 28B]
   [len_packed_sign_bytes  4B  u32 LE]
   [len_brotli_sign_bytes  4B  u32 LE]
   [brotli(packed_sign_bytes)  variable]
   [per_pair_magnitudes_fp16  NUM_PAIRS × 2B]

Per Catalog #307: this is IMPLEMENTATION-LEVEL response to the empirical
falsification of Variants B-a/b/c (signed-exponent / variable-bit-width /
non-uniform quantization). The PARADIGM (residual-correction-hybrid stacking)
is UNCHANGED.

Per CLAUDE.md "Forbidden premature KILL": if Variant B-d still scale-invariant,
DEFER to Variant C (training-side fix preventing tanh saturation), do NOT kill.

Cross-references:
- Sister L1 BPR1 codec: `src/tac/substrates/boost_nerv_pr110_residual/archive.py`
- Sister gain_clamp sweep landing: `.omx/research/boostnerv_pr110_gain_clamp_sweep_landed_20260526.md`
- Canonical equation #347 registry: `tac.canonical_equations.residual_hybrid_boosting_savings_v1`
- Pre-execution gate report: `.omx/research/boostnerv_bpr1_variant_b_codec_redesign_pre_execution_gate_report_20260526.md`
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli
import numpy as np

# Canonical BPR1 header (mirrors `src/tac/substrates/boost_nerv_pr110_residual/__init__.py`).
BPR1_MAGIC = b"BPR1"
BPR1_HEADER_LEN = 28  # 4 magic + 1 version + 1 rounds + 1 align + 16 sha + 4 blob_len + 1 reserved

# Variant B-d header extensions
VARIANT_B_D_LEN_FIELDS = 8  # len_packed_signs (u32) + len_brotli_signs (u32)
MAGNITUDE_SCALAR_BYTES_PER_PAIR = 2  # fp16


@dataclass(frozen=True)
class VariantBDSidecarManifest:
    """Per-cell sidecar build manifest (sister of L1 sidecar_manifest)."""
    bpr1_magic_hex: str
    num_boosting_rounds: int
    num_pairs: int
    num_pixels_per_pair: int  # H*W*3
    sign_bitmap_packed_bytes: int  # NUM_PAIRS * NUM_PIXELS / 8
    sign_bitmap_brotli_bytes: int
    sign_bitmap_brotli_ratio: float  # brotli/raw_packed
    per_pair_magnitudes_bytes: int
    bpr1_header_bytes: int
    variant_b_d_len_fields_bytes: int
    bpr1_sidecar_total_bytes: int

    def as_dict(self) -> dict:
        return {
            "bpr1_magic_hex": self.bpr1_magic_hex,
            "num_boosting_rounds": self.num_boosting_rounds,
            "num_pairs": self.num_pairs,
            "num_pixels_per_pair": self.num_pixels_per_pair,
            "sign_bitmap_packed_bytes": self.sign_bitmap_packed_bytes,
            "sign_bitmap_brotli_bytes": self.sign_bitmap_brotli_bytes,
            "sign_bitmap_brotli_ratio": self.sign_bitmap_brotli_ratio,
            "per_pair_magnitudes_bytes": self.per_pair_magnitudes_bytes,
            "bpr1_header_bytes": self.bpr1_header_bytes,
            "variant_b_d_len_fields_bytes": self.variant_b_d_len_fields_bytes,
            "bpr1_sidecar_total_bytes": self.bpr1_sidecar_total_bytes,
            "codec_variant": "BPR1_VARIANT_B_D_SIGN_BITMAP",
        }


def build_variant_b_d_sidecar(
    residuals_clamped: np.ndarray,
    pr110_base_sha256_prefix: bytes,
    gain_clamp: float,
    num_boosting_rounds: int = 1,
) -> tuple[bytes, VariantBDSidecarManifest]:
    """Build Variant B-d sidecar bytes.

    Args:
        residuals_clamped: shape (NUM_PAIRS, H, W, 3); float values in [-gain_clamp, +gain_clamp]
            after the residual head's clip(±gain_clamp). Caller is responsible for ensuring
            values are within this range (canonical builder does this before calling).
        pr110_base_sha256_prefix: 16-byte SHA-256 prefix of PR110 base archive (binding token
            per Catalog #139 byte-mutation discipline).
        gain_clamp: per-pair magnitude scalar (encoded as fp16 per pair).
        num_boosting_rounds: BPR1 header field (canonical=1 per L1 landing).

    Returns:
        (sidecar_bytes, manifest)
    """
    if residuals_clamped.ndim != 4:
        raise ValueError(
            f"residuals_clamped must be (NUM_PAIRS, H, W, 3); got shape={residuals_clamped.shape}"
        )
    if len(pr110_base_sha256_prefix) != 16:
        raise ValueError(
            f"pr110_base_sha256_prefix must be 16 bytes; got {len(pr110_base_sha256_prefix)}"
        )
    if not (1 <= num_boosting_rounds <= 4):
        raise ValueError(
            f"num_boosting_rounds must be in [1, 4]; got {num_boosting_rounds}"
        )
    if gain_clamp <= 0:
        raise ValueError(f"gain_clamp must be positive; got {gain_clamp}")

    num_pairs, h, w, c = residuals_clamped.shape
    num_pixels_per_pair = h * w * c

    # Step 1: extract sign bitmap (>= 0 → 1; < 0 → 0)
    # NOTE: residuals exactly at zero map to sign=1 (positive). This is canonical
    # since post-clip residuals at exactly 0.0 contribute zero magnitude either way.
    sign_bits = (residuals_clamped.flatten() >= 0).astype(np.uint8)

    # Step 2: pack 8 sign bits per byte (numpy packbits is big-endian by canonical spec)
    sign_bytes_packed = np.packbits(sign_bits, bitorder="big")
    sign_bytes_packed_bytes = sign_bytes_packed.tobytes()

    # Step 3: brotli q9 compress packed sign bitmap
    sign_bytes_brotli = brotli.compress(sign_bytes_packed_bytes, quality=9)

    # Step 4: per-pair magnitudes as fp16 (one scalar per pair = gain_clamp value)
    # Per Variant B-d design: every pair shares the same magnitude (=gain_clamp). We
    # encode per-pair to permit future per-pair-variable-magnitude extension without
    # breaking the wire format.
    per_pair_magnitudes = np.full(num_pairs, gain_clamp, dtype=np.float16)
    per_pair_magnitudes_bytes = per_pair_magnitudes.tobytes()

    # Step 5: assemble sidecar (sister-identical L1 header byte layout: 4+1+16+4+3 = 28)
    header = (
        BPR1_MAGIC                                            # 4
        + struct.pack("<B", num_boosting_rounds)              # 1 rounds
        + pr110_base_sha256_prefix                            # 16
        + struct.pack("<I", len(sign_bytes_brotli))           # 4 residual_blob_len (sister field; here = brotli-sign-bytes)
        + b"\x00\x00\x00"                                     # 3 reserved tail
    )
    assert len(header) == BPR1_HEADER_LEN, f"header={len(header)} expected={BPR1_HEADER_LEN}"

    # Variant B-d len fields (canonical extension)
    len_fields = (
        struct.pack("<I", len(sign_bytes_packed_bytes))       # uncompressed packed-sign-bytes
        + struct.pack("<I", len(sign_bytes_brotli))           # brotli-compressed sign-bytes
    )
    assert len(len_fields) == VARIANT_B_D_LEN_FIELDS

    sidecar = header + len_fields + sign_bytes_brotli + per_pair_magnitudes_bytes

    sidecar_total = (
        BPR1_HEADER_LEN
        + VARIANT_B_D_LEN_FIELDS
        + len(sign_bytes_brotli)
        + len(per_pair_magnitudes_bytes)
    )
    assert len(sidecar) == sidecar_total

    manifest = VariantBDSidecarManifest(
        bpr1_magic_hex=BPR1_MAGIC.hex(),
        num_boosting_rounds=num_boosting_rounds,
        num_pairs=num_pairs,
        num_pixels_per_pair=num_pixels_per_pair,
        sign_bitmap_packed_bytes=len(sign_bytes_packed_bytes),
        sign_bitmap_brotli_bytes=len(sign_bytes_brotli),
        sign_bitmap_brotli_ratio=len(sign_bytes_brotli) / max(len(sign_bytes_packed_bytes), 1),
        per_pair_magnitudes_bytes=len(per_pair_magnitudes_bytes),
        bpr1_header_bytes=BPR1_HEADER_LEN,
        variant_b_d_len_fields_bytes=VARIANT_B_D_LEN_FIELDS,
        bpr1_sidecar_total_bytes=sidecar_total,
    )
    return sidecar, manifest


def compute_sign_bitmap_entropy_diagnostic(residuals_clamped: np.ndarray) -> dict:
    """Diagnostic helper: per-pair fraction of positive signs (entropy proxy).

    Per the pre-execution gate report KEY HYPOTHESIS: does the trained sign-bitmap
    have higher entropy at larger gain_clamp? Diagnostic surface for the Catalog
    #305 observability requirement + sister Catalog #356 per-axis decomposition
    surface.

    Returns dict with:
        - per_pair_positive_fraction: array[num_pairs] of fraction-positive per pair
        - global_positive_fraction: scalar global fraction-positive
        - per_pair_sign_entropy_bits: array[num_pairs] of binary entropy H(p) per pair
        - global_sign_entropy_bits: scalar global binary entropy
        - num_pairs / num_pixels_per_pair
    """
    if residuals_clamped.ndim != 4:
        raise ValueError(
            f"residuals_clamped must be (NUM_PAIRS, H, W, 3); got shape={residuals_clamped.shape}"
        )
    num_pairs = residuals_clamped.shape[0]
    num_pixels_per_pair = int(np.prod(residuals_clamped.shape[1:]))
    signs = (residuals_clamped >= 0).astype(np.float32)
    per_pair_pos = signs.reshape(num_pairs, -1).mean(axis=1)
    global_pos = float(signs.mean())

    def _binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
        p_arr = np.asarray(p, dtype=np.float64)
        eps = 1e-12
        p_clamped = np.clip(p_arr, eps, 1.0 - eps)
        ent = -p_clamped * np.log2(p_clamped) - (1.0 - p_clamped) * np.log2(1.0 - p_clamped)
        return ent

    per_pair_ent = _binary_entropy(per_pair_pos)
    global_ent = float(_binary_entropy(global_pos))

    return {
        "num_pairs": num_pairs,
        "num_pixels_per_pair": num_pixels_per_pair,
        "per_pair_positive_fraction": per_pair_pos.tolist(),
        "global_positive_fraction": global_pos,
        "per_pair_sign_entropy_bits": per_pair_ent.tolist(),
        "global_sign_entropy_bits": global_ent,
    }
