# SPDX-License-Identifier: MIT
"""Canonical codec name registry + bits-per-sample lookup.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md Section 5.2]
[verified-against: tac.codec.* canonical codec primitives]
[verified-against: Catalog #265 canonical-contract tokens]
"""
from __future__ import annotations


CODEC_BITS_PER_SAMPLE: dict[str, int | None] = {
    # Floating point
    "fp32": 32,
    "fp16": 16,
    "bfloat16": 16,
    "fp4": 4,
    # Signed integer
    "int8": 8,
    "int6": 6,
    "int4": 4,
    "int2": 2,
    # Unsigned integer
    "uint8": 8,
    "uint6": 6,
    "uint4": 4,
    # Generic lossless (variable rate — represented as None; depends on entropy)
    "brotli": None,
    "lzma": None,
    "zstd": None,
    "arithmetic": None,
    "magic_codec": None,
    # Substrate-specific codecs
    "block_fp": 4,                      # Quantizr canonical (1.017 bpw)
    "water_filling": None,              # variable per-tensor
    "wyner_ziv_tier_1": 0,              # zero-cost (delivered from existing)
    "wyner_ziv_tier_2": 1,              # constants (very low cost)
    "wyner_ziv_tier_3": 4,              # waiver-tier (medium cost)
    "balle_entropy_bottleneck": None,   # learned entropy model
}
"""Canonical codec name → bits-per-sample lookup (None = variable rate)."""


CODEC_NAMES: frozenset[str] = frozenset(CODEC_BITS_PER_SAMPLE.keys())
"""Canonical codec name registry."""


def codec_bits_per_sample(codec: str) -> int | None:
    """Return the bits-per-sample for a codec, or None if variable-rate.

    Args:
        codec: codec name from CODEC_NAMES

    Returns:
        int bits per sample (e.g. 16 for fp16, 4 for int4) or None for variable-rate

    Raises:
        ValueError: if codec not in CODEC_NAMES
    """
    if codec not in CODEC_NAMES:
        raise ValueError(f"codec {codec!r} not in canonical CODEC_NAMES")
    return CODEC_BITS_PER_SAMPLE[codec]


def codec_bytes_for_n_samples(codec: str, n_samples: int) -> int:
    """Return the predicted byte count for encoding N samples with the given codec.

    For variable-rate codecs (None bits-per-sample), this returns N (1 byte per sample
    as an upper-bound placeholder; actual count requires empirical measurement).

    Args:
        codec: codec name from CODEC_NAMES
        n_samples: number of samples (bytes) to encode

    Returns:
        predicted byte count
    """
    bits = codec_bits_per_sample(codec)
    if bits is None:
        # Variable-rate placeholder: assume 1 byte per sample as upper bound
        return n_samples
    if bits == 0:
        return 0
    total_bits = n_samples * bits
    return (total_bits + 7) // 8  # round up to whole bytes


__all__ = [
    "CODEC_BITS_PER_SAMPLE",
    "CODEC_NAMES",
    "codec_bits_per_sample",
    "codec_bytes_for_n_samples",
]
