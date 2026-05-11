"""Reusable packet-compiler byte grammar primitives.

This package is intentionally narrow. It exposes only implemented, tested
transducers; broader PacketIR APIs remain roadmap items until their identity
vectors and strict no-op proofs land.
"""

from __future__ import annotations

from tac.packet_compiler.pr81_quantizr import (
    PR81_POS_LEVELS,
    FP4Codebook,
    decode_router_actions,
    encode_router_actions,
    pack_nibbles,
    unpack_nibbles,
)
from tac.packet_compiler.pr84_adaptive_mask import (
    AdaptiveContextSpec,
    decode_adaptive_context_stream,
    encode_adaptive_context_stream,
)
from tac.packet_compiler.pr91_hpac_grammar import (
    MAGIC_QH0,
    MAGIC_QM0,
    QMQHHeader,
    decode_categorical_stream,
    emit_qmqh_header,
    encode_categorical_stream,
    pack_hi_lo_split,
    parse_qmqh_header,
    unpack_hi_lo_split,
)
from tac.packet_compiler.pr92_joint_stream import (
    MAGIC_RMC1,
    MAGIC_RSA1,
    MAGIC_RSB1,
    RMC1Composite,
    RSA1Side,
    RSB1Side,
    pack_rmc1_composite,
    pack_rsa1_side,
    pack_rsb1_side,
    unpack_rmc1_composite,
    unpack_rsa1_side,
    unpack_rsb1_side,
)
from tac.packet_compiler.pr93_pose_codec import (
    MAGIC_MODEL_COMPACT,
    MAGIC_POSE_DV,
    DeltaVarintPoseStream,
    QZMB1Block,
    decode_delta_varint_pose,
    encode_delta_varint_pose,
    pack_qzmb1_block,
    unpack_qzmb1_block,
)
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
    # PR81 — FP4 codebook + ROUTER_ACTION packing
    "FP4Codebook",
    "PR81_POS_LEVELS",
    "decode_router_actions",
    "encode_router_actions",
    "pack_nibbles",
    "unpack_nibbles",
    # PR84 — adaptive-context range coder
    "AdaptiveContextSpec",
    "decode_adaptive_context_stream",
    "encode_adaptive_context_stream",
    # PR91 — universal AC wrapper + QM0/QH0 grammar
    "MAGIC_QH0",
    "MAGIC_QM0",
    "QMQHHeader",
    "decode_categorical_stream",
    "emit_qmqh_header",
    "encode_categorical_stream",
    "pack_hi_lo_split",
    "parse_qmqh_header",
    "unpack_hi_lo_split",
    # PR92 — RMC1 / RSA1 / RSB1 joint-stream grammar
    "MAGIC_RMC1",
    "MAGIC_RSA1",
    "MAGIC_RSB1",
    "RMC1Composite",
    "RSA1Side",
    "RSB1Side",
    "pack_rmc1_composite",
    "pack_rsa1_side",
    "pack_rsb1_side",
    "unpack_rmc1_composite",
    "unpack_rsa1_side",
    "unpack_rsb1_side",
    # PR93 — delta-varint pose codec + QZMB1 grammar
    "DeltaVarintPoseStream",
    "MAGIC_MODEL_COMPACT",
    "MAGIC_POSE_DV",
    "QZMB1Block",
    "decode_delta_varint_pose",
    "encode_delta_varint_pose",
    "pack_qzmb1_block",
    "unpack_qzmb1_block",
    # PR101 — sidecar grammar
    "CenteredDeltaUint8Stream",
    "RankedSidecarSchema",
    "SplitBrotliStream",
    "decode_centered_delta_uint8",
    "decode_ranked_no_op_sidecar",
    "encode_centered_delta_uint8",
    "encode_ranked_no_op_sidecar",
    "parse_split_brotli_self_delimiting",
    "split_brotli_self_delimiting",
    # PR103 — arithmetic coding
    "AdaptiveBrotliResult",
    "MergedRangeStream",
    "WeightTensorACSpec",
    "adaptive_brotli_param_search",
    "decode_latent_hi_arithmetic",
    "decode_merged_range_stream",
    "encode_latent_hi_arithmetic",
    "encode_merged_range_stream",
]
