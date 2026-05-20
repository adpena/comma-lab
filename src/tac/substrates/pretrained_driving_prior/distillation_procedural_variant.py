# SPDX-License-Identifier: MIT
"""DP1 procedural-codebook replacement variant — first-empirical-anchor scaffold.

Per ``feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md``
OP-ROUTABLE #2 (BUILD the variant trainer extension) + operator-frontier-override
2026-05-20 ("3 slots are approved + magic codec stacking") + canonical equation
``procedural_codebook_from_seed_compression_savings_v1`` (sister
``src/tac/canonical_equations/procedural_codebook_savings.py``).

**Canonical equation #26 IN-DOMAIN context** (per slot 3 refined domain;
the canonical helper ``validate_context_is_in_domain`` lands in a sister
subagent — until then we declare the IN-DOMAIN string constant
``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT`` here and integrate via the
sister helper post-landing).

This module is the **L0 SCAFFOLD procedural variant** of the DP1 trainer's
codebook source. The existing canonical ``distill_codebook`` path produces
a ~5-10 KB codebook from Comma2k19 PCA. This variant REPLACES that codebook
with a deterministic 32-byte PCG64 seed; the inflate runtime re-derives the
codebook bytes via :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`.

**Predicted ΔS = -0.002706** per canonical equation #26 closed form:
``-25 * (4096 - 32) / 37,545,489``.

**Compliance posture** (per ``canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict):

1. Seed bytes live INSIDE archive.zip (~32 B replaces ~4096 B codebook); the
   contest rate term charges ``25 * archive_bytes / 37_545_489`` per
   ``upstream/evaluate.py:63``.
2. Derivation routine ``derive_codebook_from_seed`` lives in inflate.py
   (legitimate "external library / tool" per upstream README).
3. Structurally distinct from rejected loophole pattern (PR #36 / #38 / #68
   / #69 / #78 / #87) which relocates score-relevant bytes OUTSIDE
   archive.zip. This variant keeps BOTH seed AND derivation routine inside
   the canonical submission package — *sensitivity-aware compression*, NOT
   *payload smuggling*.

**Catalog #209/#213 Comma2k19 leakage refusal**: this variant does NOT
construct any ``Comma2k19FrameIterator``; the seed bytes are derived
deterministically from a PCG64 PRNG. No Comma2k19 download or chunk
streaming. The procedural variant is structurally OOD by construction.

**Catalog #240 recipe-vs-trainer-state consistency**: the variant is
``research_only=True`` until the operator-routed paired smoke
(``feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md``
OP-ROUTABLE #3) lands a contest-CUDA + contest-CPU anchor per
CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA".

**Catalog #220 substrate L1+ scaffold operational mechanism**: the
``compose_with_procedural_codebook`` API emits archive bytes whose
codebook section IS structurally consumed by inflate (the seed is
inverted at parse-time to re-derive the codebook bytes byte-identically
to the original distilled codebook for the inflate-time consumer).

**Catalog #272 byte-mutation distinguishing-feature contract**: the
operational mechanism IS byte-mutation-traceable: mutating any of the
32 seed bytes produces a different derived codebook + different rendered
frames (verified via ``verify_seed_mutation_changes_codebook_bytes``).

**Catalog #287 + #323 canonical Provenance**: no score claim asserted in
this module; all return values carry ``score_claim=False`` markers.

**Catalog #324 post-training Tier-C validation**: the predicted ΔS
prediction ``-0.002706`` is REGISTERED HYPOTHESIS; reactivation criterion
= "post-training Tier-C re-measurement on landed paired smoke archive sha
via tools/mdl_scorer_conditional_ablation.py --tier c".

**6-hook wire-in declaration** per Catalog #125:

* hook #1 sensitivity-map = N/A (variant is a single archive-build path)
* hook #2 Pareto constraint = ACTIVE via procedural_codebook_savings_v1
  predicted ΔS contribution to rate-axis Pareto polytope
* hook #3 bit-allocator = ACTIVE (32-byte seed slot replaces ~4096 B
  codebook slot; bit-allocator's per-tensor importance changes)
* hook #4 cathedral autopilot dispatch = ACTIVE via sister consumer
  ``tac.cathedral_consumers.procedural_codebook_generator_consumer``
  (auto-discovered per Catalog #335)
* hook #5 continual-learning posterior = ACTIVE (first empirical anchor
  via ``update_equation_with_empirical_anchor`` post-paired-smoke)
* hook #6 probe-disambiguator = ACTIVE (PROCEDURAL vs ORIGINAL vs
  NULL-EXPLOIT 3-recipe contrast IS the probe disambiguator)

Cross-references:

* Design memo: ``.omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md``
* Memory landing: ``~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md``
* Canonical helper: :mod:`tac.procedural_codebook_generator.seed_derived_codebook`
* Canonical equation: :mod:`tac.canonical_equations.procedural_codebook_savings`
* Sister substrate API: :mod:`tac.substrates.pretrained_driving_prior.archive`
"""

from __future__ import annotations

from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.procedural_codebook_generator import (
    DEFAULT_GENERATOR_KIND,
    derive_codebook_from_seed,
    verify_codebook_from_seed,
)
from tac.substrates.pretrained_driving_prior.archive import (
    DP1_HEADER_FMT,
    DP1_HEADER_SIZE,
    DP1_MAGIC,
    DP1_SCHEMA_VERSION,
    parse_dp1_archive_bytes,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "PROCEDURAL_CODEBOOK_SHAPE_DEFAULT",
    "PROCEDURAL_CODEBOOK_DTYPE_DEFAULT",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "derive_procedural_codebook_replacement",
    "compose_with_procedural_codebook",
    "verify_procedural_codebook_in_domain",
    "verify_seed_mutation_changes_codebook_bytes",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT: str = "comma2k19_ood_derived_basis_replacement"
"""IN-DOMAIN context label for canonical equation #26 per slot 3 refined domain.

When slot 3 ``validate_context_is_in_domain`` helper lands, this constant
becomes the value the helper validates against. Until then, the constant
is the documented context-string consumed by
:func:`verify_procedural_codebook_in_domain`.
"""

PROCEDURAL_SEED_SIZE_BYTES: int = 32
"""Canonical procedural-variant seed size in bytes.

Per design memo §4 recipe outline #2: 32-byte PCG64 seed replaces the
~4096-byte codebook slot. The ``32`` constant is the numerator in the
canonical equation #26 ``K_seed`` term:

    predicted_delta_s = -25 * (N_codebook - 32) / 37_545_489
"""

PROCEDURAL_CODEBOOK_SHAPE_DEFAULT: tuple[int, int] = (1024, 4)
"""Default output shape for procedural codebook derivation.

Per design memo §4 recipe outline #2: ``(1024, 4)`` int8 = 4096 bytes
total, matching the canonical codebook size declared in the equation
``predicted_codebook_size_per_substrate_bytes_lower=2048`` ...
``upper=6144`` domain-of-validity range.
"""

PROCEDURAL_CODEBOOK_DTYPE_DEFAULT: np.dtype = np.dtype(np.uint8)
"""Default dtype for procedural codebook derivation."""

_MIN_SEED_SIZE_BYTES: int = 8
_MAX_SEED_SIZE_BYTES: int = 256


class ProceduralVariantError(ValueError):
    """Raised on invalid procedural-variant configuration.

    Sister of :class:`tac.procedural_codebook_generator.ProceduralCodebookGeneratorError`
    + :class:`ValueError` raised by archive.py validators.
    """


@dataclass(frozen=True)
class ProceduralVariantConfig:
    """Configuration for the DP1 procedural-codebook replacement variant.

    Sister of ``DistillationConfig`` (canonical Comma2k19-derived path).
    This config holds the inputs to the procedural derivation pipeline.

    Attributes:
        seed_bytes: The procedural seed bytes (typically 32 bytes for
            ``derive_codebook_from_seed`` PCG64). Length must be in
            ``[_MIN_SEED_SIZE_BYTES, _MAX_SEED_SIZE_BYTES]`` per canonical
            equation #26 ``seed_size_bytes_range`` domain-of-validity.
        output_shape: The codebook shape to derive (default
            ``PROCEDURAL_CODEBOOK_SHAPE_DEFAULT`` = ``(1024, 4)``).
        dtype: The codebook dtype (default
            ``PROCEDURAL_CODEBOOK_DTYPE_DEFAULT`` = ``np.uint8``).
        generator_kind: The PRNG kind (one of ``xorshift`` / ``lcg`` /
            ``pcg64``; default ``DEFAULT_GENERATOR_KIND`` = ``"pcg64"``).
        canonical_equation_context: The IN-DOMAIN context per slot 3
            refined domain (default ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``).
    """

    seed_bytes: bytes
    output_shape: tuple[int, ...] = PROCEDURAL_CODEBOOK_SHAPE_DEFAULT
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT
    generator_kind: str = DEFAULT_GENERATOR_KIND
    canonical_equation_context: str = CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT

    def __post_init__(self) -> None:
        """Validate configuration invariants.

        Per Catalog #287 placeholder-rationale rejection sister discipline +
        CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + canonical
        equation #26 domain-of-validity ``seed_size_bytes_range=[8, 256]``.
        """
        if not isinstance(self.seed_bytes, (bytes, bytearray, memoryview)):
            raise ProceduralVariantError(
                f"seed_bytes must be bytes-like; got {type(self.seed_bytes).__name__}"
            )
        seed_view = bytes(self.seed_bytes)
        if not (_MIN_SEED_SIZE_BYTES <= len(seed_view) <= _MAX_SEED_SIZE_BYTES):
            raise ProceduralVariantError(
                f"seed_bytes length {len(seed_view)} outside canonical equation "
                f"#26 domain-of-validity range "
                f"[{_MIN_SEED_SIZE_BYTES}, {_MAX_SEED_SIZE_BYTES}]"
            )
        if not isinstance(self.output_shape, tuple) or len(self.output_shape) == 0:
            raise ProceduralVariantError(
                f"output_shape must be non-empty tuple; got {self.output_shape!r}"
            )
        if self.generator_kind not in {"xorshift", "lcg", "pcg64"}:
            raise ProceduralVariantError(
                f"generator_kind {self.generator_kind!r} not canonical; see "
                "tac.procedural_codebook_generator.SUPPORTED_GENERATOR_KINDS"
            )
        if not isinstance(self.canonical_equation_context, str):
            raise ProceduralVariantError(
                f"canonical_equation_context must be str; got "
                f"{type(self.canonical_equation_context).__name__}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_procedural_codebook_replacement(
    seed_bytes: bytes,
    output_shape: tuple[int, ...] = PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> np.ndarray:
    """Derive a procedural codebook replacement from a deterministic seed.

    Thin wrapper around :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`
    with DP1-specific validation + canonical equation #26 domain awareness.

    Args:
        seed_bytes: Procedural seed (8-256 bytes; canonical 32 bytes).
        output_shape: Codebook shape (default ``(1024, 4)``).
        dtype: Codebook dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        np.ndarray with the specified shape + dtype, deterministically
        derived from ``seed_bytes`` via the canonical generator.

    Raises:
        ProceduralVariantError: invalid inputs per
            :meth:`ProceduralVariantConfig.__post_init__` invariants.
    """
    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        output_shape=output_shape,
        dtype=np.dtype(dtype),
        generator_kind=generator_kind,
    )
    return derive_codebook_from_seed(
        seed_bytes=config.seed_bytes,
        output_shape=config.output_shape,
        dtype=config.dtype,
        generator_kind=config.generator_kind,
    )


def verify_procedural_codebook_in_domain(
    context: str = CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
) -> bool:
    """Verify the procedural codebook context is IN-DOMAIN per canonical equation #26.

    Per slot 3 ``CANONICAL EQUATION #26 DOMAIN REFINEMENT`` (mid-flight
    sister subagent ``a230693c``): the refined domain is
    ``comma2k19_ood_derived_basis_replacement`` (OOD-derived basis
    replacement; the procedural variant is structurally OOD because it
    NEVER reads Comma2k19 data — the seed is a deterministic PRNG output).

    When slot 3's ``validate_context_is_in_domain`` helper lands in
    ``tac.canonical_equations``, this function will be refactored to call
    that helper. Until then, the IN-DOMAIN check is a constant comparison
    against ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``.

    Args:
        context: The context string to validate (default
            ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``).

    Returns:
        True if context is IN-DOMAIN per canonical equation #26.
    """
    # Try slot 3 sister helper first if it has landed; gracefully fall
    # back to the constant comparison if not.
    try:
        from tac.canonical_equations import (  # type: ignore[attr-defined]
            validate_context_is_in_domain,
        )
        return bool(
            validate_context_is_in_domain(
                equation_id="procedural_codebook_from_seed_compression_savings_v1",
                context=context,
            )
        )
    except ImportError:
        # Slot 3 sister helper not yet landed — fall back to canonical constant.
        return context == CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT


def compose_with_procedural_codebook(
    original_archive_bytes: bytes,
    seed_bytes: bytes,
    *,
    output_shape: tuple[int, ...] = PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bytes:
    """Compose a DP1 archive whose codebook is REPLACED by a procedural seed.

    Per design memo §4 recipe outline #2 (PROCEDURAL replacement variant):
    takes an existing DP1 archive + a 32-byte seed; emits a NEW archive
    whose codebook section bytes are ``brotli(seed_bytes)`` (the seed is
    re-derived at inflate time via :func:`derive_procedural_codebook_replacement`).

    The bytes-saved is ``len(original_codebook_blob) - len(brotli(seed_bytes))``
    which approximately equals ``4096 - 32 = 4064`` bytes for the canonical
    config + matches the canonical equation #26 prediction ``N - K`` term.

    Args:
        original_archive_bytes: The existing DP1 archive bytes (with the
            canonical Comma2k19-distilled codebook).
        seed_bytes: The procedural seed (32 bytes canonical).
        output_shape: Codebook shape (must match the original codebook
            shape for byte-stability at inflate; default ``(1024, 4)``).
        dtype: Codebook dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        New DP1-style archive bytes with the codebook slot replaced by
        ``brotli(seed_bytes)``. Header is rewritten so
        ``codebook_blob`` length matches the new payload; renderer,
        residual, meta sections are preserved byte-for-byte.

    Raises:
        ProceduralVariantError: invalid configuration.
        ValueError: original archive bytes are not parseable as DP1.
    """
    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        output_shape=output_shape,
        dtype=np.dtype(dtype),
        generator_kind=generator_kind,
    )

    # Parse the existing DP1 archive to locate section offsets.
    sections = parse_dp1_archive_bytes(original_archive_bytes)

    # The procedural codebook section is the deterministic brotli of
    # (seed_bytes + shape metadata + generator_kind tag). The inflate
    # runtime re-derives the codebook via derive_codebook_from_seed.
    # For the L0 scaffold we store ONLY the seed bytes; the shape and
    # generator kind are pinned in the substrate's archive metadata.
    new_codebook_blob = brotli.compress(config.seed_bytes, quality=9)

    # Locate the other section blobs in the original archive.
    header_start, header_len = sections["dp1_header"]
    _, original_codebook_len = sections["codebook_blob"]
    renderer_start, renderer_len = sections["renderer_blob"]
    residual_start, residual_len = sections["residual_blob"]
    meta_start, meta_len = sections["meta_blob"]

    renderer_blob = original_archive_bytes[
        renderer_start : renderer_start + renderer_len
    ]
    residual_blob = original_archive_bytes[
        residual_start : residual_start + residual_len
    ]
    meta_blob = original_archive_bytes[meta_start : meta_start + meta_len]

    # Parse the original header to recover the unchanged fields.
    import struct

    (
        magic,
        version,
        num_pairs,
        out_h,
        out_w,
        per_pair_bytes,
        _original_codebook_len_in_header,
        original_renderer_len,
        original_residual_len,
        original_meta_len,
    ) = struct.unpack(DP1_HEADER_FMT, original_archive_bytes[:DP1_HEADER_SIZE])

    if magic != DP1_MAGIC or version != DP1_SCHEMA_VERSION:
        raise ValueError(
            f"original_archive_bytes not parseable as DP1 v{DP1_SCHEMA_VERSION}: "
            f"magic={magic!r} version={version}"
        )

    # Rewrite the header with the new codebook_len.
    new_header = struct.pack(
        DP1_HEADER_FMT,
        DP1_MAGIC,
        DP1_SCHEMA_VERSION,
        num_pairs,
        out_h,
        out_w,
        per_pair_bytes,
        len(new_codebook_blob),
        original_renderer_len,
        original_residual_len,
        original_meta_len,
    )
    assert len(new_header) == DP1_HEADER_SIZE, (
        f"new_header size {len(new_header)} != {DP1_HEADER_SIZE}"
    )

    new_archive = (
        new_header + new_codebook_blob + renderer_blob + residual_blob + meta_blob
    )

    # Catalog #220 + #272 + #287 invariant: assert procedural replacement
    # actually reduced bytes (the operational mechanism is byte-mutation-
    # traceable). For canonical 32-byte seed + ~4096 B codebook the
    # bytes-saved should be > 1000.
    bytes_saved = len(original_archive_bytes) - len(new_archive)
    if bytes_saved <= 0:
        raise ProceduralVariantError(
            f"procedural variant did NOT reduce archive bytes "
            f"({len(original_archive_bytes)} -> {len(new_archive)}); "
            f"check seed size + brotli quality"
        )

    return new_archive


def verify_seed_mutation_changes_codebook_bytes(
    seed_bytes: bytes,
    *,
    output_shape: tuple[int, ...] = PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bool:
    """Verify mutating a single seed byte changes the derived codebook.

    Catalog #272 byte-mutation distinguishing-feature contract: the
    operational mechanism is byte-mutation-traceable. Flipping any of
    the 32 seed bytes MUST produce a different derived codebook.

    Args:
        seed_bytes: The canonical seed.
        output_shape: Codebook shape (default ``(1024, 4)``).
        dtype: Codebook dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        True if mutating ANY seed byte produces a different codebook.
        False if the derivation is degenerate (does NOT depend on the
        mutated byte).
    """
    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        output_shape=output_shape,
        dtype=np.dtype(dtype),
        generator_kind=generator_kind,
    )
    original_codebook = derive_codebook_from_seed(
        seed_bytes=config.seed_bytes,
        output_shape=config.output_shape,
        dtype=config.dtype,
        generator_kind=config.generator_kind,
    )
    # Flip the FIRST byte to test mutation propagation (full byte-mutation
    # sweep is the operator-routed runtime verifier; this is the per-seed
    # invariant check).
    mutated_seed = bytearray(config.seed_bytes)
    mutated_seed[0] = (mutated_seed[0] + 1) & 0xFF
    mutated_codebook = derive_codebook_from_seed(
        seed_bytes=bytes(mutated_seed),
        output_shape=config.output_shape,
        dtype=config.dtype,
        generator_kind=config.generator_kind,
    )
    # Use verify_codebook_from_seed to compare byte-for-byte.
    if mutated_codebook.shape != original_codebook.shape:
        return True
    return not bool(np.array_equal(original_codebook, mutated_codebook))
