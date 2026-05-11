"""Reusable packet-compiler byte grammar primitives.

This package is intentionally narrow. It exposes only implemented, tested
transducers; broader PacketIR APIs remain roadmap items until their identity
vectors and strict no-op proofs land.
"""

from __future__ import annotations

from tac.packet_compiler.pr101_sidecar_grammar import (
    CenteredDeltaUint8Stream,
    RankedSidecarSchema,
    SplitBrotliStream,
    decode_centered_delta_uint8,
    decode_ranked_no_op_sidecar,
    encode_centered_delta_uint8,
    encode_ranked_no_op_sidecar,
    parse_split_brotli_self_delimiting,
    split_brotli_self_delimiting,
)
from tac.packet_compiler.pr103_arithmetic_coding import (
    AdaptiveBrotliResult,
    MergedRangeStream,
    WeightTensorACSpec,
    adaptive_brotli_param_search,
    decode_latent_hi_arithmetic,
    decode_merged_range_stream,
    encode_latent_hi_arithmetic,
    encode_merged_range_stream,
)

__all__ = [
    "AdaptiveBrotliResult",
    "CenteredDeltaUint8Stream",
    "MergedRangeStream",
    "RankedSidecarSchema",
    "SplitBrotliStream",
    "WeightTensorACSpec",
    "adaptive_brotli_param_search",
    "decode_centered_delta_uint8",
    "decode_latent_hi_arithmetic",
    "decode_merged_range_stream",
    "decode_ranked_no_op_sidecar",
    "encode_centered_delta_uint8",
    "encode_latent_hi_arithmetic",
    "encode_merged_range_stream",
    "encode_ranked_no_op_sidecar",
    "parse_split_brotli_self_delimiting",
    "split_brotli_self_delimiting",
]
