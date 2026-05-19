"""Canonical helpers for archive-seeded procedural codebooks.

These helpers are planning-only primitives. They do not claim score movement,
perform provider dispatch, or load contest scorers.
"""

from __future__ import annotations

from .authority import LiteralPayloadKind, SeedCarrier, classify_procedural_seed_authority
from .hash_seed_codebook_generator import (
    emit_seed,
    expand_seed_to_codebook,
    verify_generator_seed_mutation_smoke,
)
from .weight_derived_codebook_generator import (
    derive_codebook_from_archive_bytes,
    freeze_source_member_sha256,
    verify_no_new_bytes_added,
)

__all__ = [
    "LiteralPayloadKind",
    "SeedCarrier",
    "classify_procedural_seed_authority",
    "derive_codebook_from_archive_bytes",
    "emit_seed",
    "expand_seed_to_codebook",
    "freeze_source_member_sha256",
    "verify_generator_seed_mutation_smoke",
    "verify_no_new_bytes_added",
]
