"""Scorer-conditional JSCC codec primitives.

This package exposes two complementary scorer-conditional codecs:

* :class:`ScorerConditionalHuffmanCoder` (legacy, deterministic Huffman):
  the byte-allocation + Huffman-coding prototype used in packet experiments.
  Encoded packets carry code lengths and bytes only; no score authority.

* :class:`ScorerConditionalEntropyCoder` (SE-4 range-coder MLP):
  the canonical SE-4 implementation per Shannon 1959 Joint Source-Channel
  Coding. Uses a small MLP to predict ``p(symbol | side_state)`` and a
  precision-controlled integer range coder. ~10% rate-term savings predicted
  on HNeRV-family substrates per the source memo.

Both coders refuse to grant score authority; exact eval packet paths remain
the only promotion path. Use the Huffman variant when alphabet=byte (256) and
the side-info is a small categorical/scalar; use the entropy-coder variant
when the side-state is a learned vector and you want ~Shannon-bound rate.

Cross-references
----------------
- Source memo SE-4: ``.omx/research/ancient_elder_polymath_research_20260513.md``
- Source memo era 1 (Shannon era): ``.omx/research/ancient_elder_era_1_shannon_20260513.md``
- Sister unconditional arithmetic coder:
  ``tac.packet_compiler.pr103_arithmetic_coding``
- Sister neural compression (Ballé hyperprior):
  ``tac.packet_compiler.balle_hyperprior``

Lane: ``lane_implement_iglt_ternary_jscc_kc3_canonical_20260513``.
"""

from __future__ import annotations

from tac.codec.jscc.archive_format import (
    JSCCArchiveSection,
    JSCCSectionManifest,
    parse_jscc_section,
    serialize_jscc_section,
)
from tac.codec.jscc.conditional_huffman import (
    CONFORMANCE_VECTORS,
    JSCCCodingContext,
    JSCCEncodedPacket,
    JSCCSection,
    ScorerConditionalHuffmanCoder,
    ScorerConditionalSignal,
    allocate_scorer_conditional_bytes,
)
from tac.codec.jscc.entropy_coder import (
    JSCC_FORMAT_VERSION,
    JSCC_MAGIC,
    ScorerConditionalEntropyCoder,
    ScorerConditionalProbabilityModel,
    decode_jscc_stream,
    encode_jscc_stream,
)

__all__ = [
    "CONFORMANCE_VECTORS",
    "JSCC_FORMAT_VERSION",
    "JSCC_MAGIC",
    "JSCCArchiveSection",
    "JSCCCodingContext",
    "JSCCEncodedPacket",
    "JSCCSection",
    "JSCCSectionManifest",
    "ScorerConditionalEntropyCoder",
    "ScorerConditionalHuffmanCoder",
    "ScorerConditionalProbabilityModel",
    "ScorerConditionalSignal",
    "allocate_scorer_conditional_bytes",
    "decode_jscc_stream",
    "encode_jscc_stream",
    "parse_jscc_section",
    "serialize_jscc_section",
]
