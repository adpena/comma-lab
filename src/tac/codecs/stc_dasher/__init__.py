# SPDX-License-Identifier: MIT
"""STC-Dasher arithmetic-coding maximalism - substrate-agnostic codec scaffold.

Per the Grand Reunion Fields-Grade Symposium 2026-05-15 Phase F binding
verdict (Composite #6, Filler + MacKay + Shannon).

Predicted score impact (rate-axis, substrate-agnostic):
    v1 scaffold: none; rate-negative byte-closed roundtrip/custody artifact.
    post-Viterbi inverse: [-0.010, -0.030] [prediction; first-principles;
                       STC achieves H(W|context) within 0.5%]

Contract surfaces
-----------------
- :class:`STCDasherEncoder`: ``encode(residual_bytes, sigma) -> bytes``
- :class:`STCDasherDecoder`: ``decode(encoded_bytes, sigma) -> bytes``
- :func:`encode_stream` / :func:`decode_stream`: stateless functional API
- :data:`SCAFFOLD_ONLY`: module-level scaffold-discipline constant per
  CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY".

Math: combines Filler-Judas-Fridrich 2011 syndrome-trellis coding (high-
cardinality high-entropy stream rate-shaver) with MacKay 2003 ITILA section 6.6
Dasher arithmetic-coded sparse-signal context model (low-cardinality
sparse-signal rate-shaver). Composition:

    encoded = arithmetic_code( STC_encode(symbols, parity_matrix) )

Math primitives delegate to :mod:`tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism`
which is the council-grade canonical implementation. THIS package is the
substrate-agnostic codec wrapper that turns those primitives into a
roundtrip-correct ``bytes -> bytes`` codec for archive bolt-on.

Scaffold discipline
-------------------
Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
(Catalog #220 sister surface):

- ``SCAFFOLD_ONLY = True`` - set to ``False`` only after smoke validation
  on a real substrate archive lands a measured ``[contest-CUDA]`` /
  ``[contest-CPU]`` anchor with byte-stable encode-decode roundtrip.
- The encode-decode roundtrip is mathematically lossless for the syndrome
  channel (STC syndrome is deterministic mod 2; arithmetic-coded
  syndrome bytes recover the syndrome exactly).
- The full residual-bytes-to-cover-bytes Viterbi inverse is council-gated
  (the symposium spec defers full Viterbi decoding to a follow-up
  subagent; this scaffold ships the syndrome roundtrip and a decoder that
  recovers the syndrome plus the original residual via a side-channel
  envelope so the codec is substrate-agnostic and byte-stable).

Lane: ``lane_stc_dasher_scaffold_v1_20260515``.
Catalog #244 sister (canonical NVML block already enforced for any
remote dispatch wrapper that consumes this codec).
"""
from __future__ import annotations

from tac.codecs.stc_dasher.decoder import (
    STCDasherDecodeError,
    STCDasherDecoder,
    STCDasherDecodeResult,
    decode_stream,
)
from tac.codecs.stc_dasher.encoder import (
    STCDasherEncoder,
    STCDasherEncodeResult,
    encode_stream,
)

__all__ = (
    "SCAFFOLD_ONLY",
    "STC_DASHER_MAGIC",
    "STC_DASHER_SCHEMA_VERSION",
    "STCDasherDecodeError",
    "STCDasherDecodeResult",
    "STCDasherDecoder",
    "STCDasherEncodeResult",
    "STCDasherEncoder",
    "decode_stream",
    "encode_stream",
)

# Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
# (Catalog #220) + "HNeRV / leaderboard-implementation parity discipline"
# lesson 2 (export-first design): scaffold-only until a contest-CUDA
# anchor lands.
SCAFFOLD_ONLY: bool = True

# 4-byte magic identifying STC-Dasher v1 envelope ("STCD" \x01).
STC_DASHER_MAGIC: bytes = b"STCD\x01"

# Schema version of the on-the-wire envelope. Any breaking change MUST
# bump this AND quarantine prior anchors per CLAUDE.md HISTORICAL_PROVENANCE
# discipline (Catalog #110 / #113).
STC_DASHER_SCHEMA_VERSION: int = 1
