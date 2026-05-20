# SPDX-License-Identifier: MIT
"""Reusable helpers for the V8 learned-compression Faiss local smoke path.

The package import stays lightweight so archive/runtime tests can load the
fixture grammar without importing torch. The optional torch-based smoke helper
remains importable from ``tac.substrates.v8_learned_compression_faiss.smoke``.
"""

from tac.substrates.v8_learned_compression_faiss.archive import (
    V8ArchiveError,
    V8ArchiveHeader,
    V8_ARCHIVE_VERSION,
    V8_MAGIC,
    build_raw_frame_archive,
    decode_raw_frame_archive,
    parse_v8_archive,
    parse_v8_header,
)
from tac.substrates.v8_learned_compression_faiss.architecture import (
    V8CategoricalPosteriorConfig,
    build_scale_hyperprior_from_codewords,
    deterministic_categorical_codewords,
    deterministic_rgb_codebook,
)
from tac.substrates.v8_learned_compression_faiss.score_aware_loss import (
    build_score_aware_roundtrip_contract,
)

__all__ = [
    "V8_ARCHIVE_VERSION",
    "V8_MAGIC",
    "V8ArchiveError",
    "V8ArchiveHeader",
    "V8CategoricalPosteriorConfig",
    "build_raw_frame_archive",
    "build_scale_hyperprior_from_codewords",
    "build_score_aware_roundtrip_contract",
    "decode_raw_frame_archive",
    "deterministic_categorical_codewords",
    "deterministic_rgb_codebook",
    "parse_v8_archive",
    "parse_v8_header",
]
