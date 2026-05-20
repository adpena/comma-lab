"""Canonical helpers for archive-seeded procedural codebooks.

These helpers are planning-only primitives. They do not claim score movement,
perform provider dispatch, or load contest scorers.
"""

from __future__ import annotations

from .authority import (
    AuthorityMode,
    LiteralPayloadKind,
    SeedCarrier,
    build_procedural_seed_authority_packet,
    classify_procedural_seed_authority,
)
from .candidate_authority import build_procedural_codebook_candidate_authority
from .hash_seed_codebook_generator import (
    emit_seed,
    expand_seed_to_codebook,
    verify_generator_seed_mutation_smoke,
)
from .null_replacement_plan import (
    NullSpanCandidate,
    build_null_seed_replacement_plan,
    contiguous_runs,
    render_null_seed_replacement_markdown,
)
from .seed_derived_codebook import (
    DEFAULT_GENERATOR_KIND,
    MAX_OUTPUT_BYTES,
    SUPPORTED_GENERATOR_KINDS,
    ProceduralCodebookGeneratorError,
    derive_codebook_from_seed,
    verify_codebook_from_seed,
)
from .weight_derived_codebook_generator import (
    derive_codebook_from_archive_bytes,
    freeze_source_member_sha256,
    verify_no_new_bytes_added,
)

__all__ = [
    "AuthorityMode",
    "DEFAULT_GENERATOR_KIND",
    "LiteralPayloadKind",
    "MAX_OUTPUT_BYTES",
    "NullSpanCandidate",
    "ProceduralCodebookGeneratorError",
    "SUPPORTED_GENERATOR_KINDS",
    "SeedCarrier",
    "build_null_seed_replacement_plan",
    "build_procedural_codebook_candidate_authority",
    "build_procedural_seed_authority_packet",
    "classify_procedural_seed_authority",
    "contiguous_runs",
    "derive_codebook_from_archive_bytes",
    "derive_codebook_from_seed",
    "emit_seed",
    "expand_seed_to_codebook",
    "freeze_source_member_sha256",
    "render_null_seed_replacement_markdown",
    "verify_codebook_from_seed",
    "verify_generator_seed_mutation_smoke",
    "verify_no_new_bytes_added",
]
