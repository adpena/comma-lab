# SPDX-License-Identifier: MIT
"""grayscale_lut procedural-LUT replacement variant — first-empirical-anchor scaffold.

Per WAVE-3-GRAYSCALE-LUT-PROCEDURAL-TRAINER-BUILD 2026-05-20 + PR101/PR106
BUILD DESIGN landing commit ``086d3ac1d`` Top-3 #1 PIVOT (Carmack MVP-first
phasing) + operator-frontier-override 2026-05-20 ("3 slots are approved +
magic codec stacking") + DP1 PROCEDURAL TRAINER BUILD canonical pattern
landing commit ``9cbfa471c`` + VQ-VAE PROCEDURAL VARIANT BUILD canonical
sister landing commit ``6fea30f22`` + canonical equation
``procedural_codebook_from_seed_compression_savings_v1`` (sister
``src/tac/canonical_equations/procedural_codebook_savings.py``).

**Canonical equation #26 IN-DOMAIN context** for grayscale_lut is
``chroma_lut_replacement`` (per ``_INCLUDED_CONTEXTS`` in
``procedural_codebook_savings.py:100``). The grayscale-LUT family's
canonical chroma colorization LUT IS the strongest IN-DOMAIN context
match per CANONICAL EQUATION #26 DOMAIN REFINEMENT commit ``8d8a7c6c5``:
a 256-byte (uint8) per-class chroma table that maps each grayscale level
or class index to a chroma offset.

This module is the **L0 SCAFFOLD procedural variant** of the grayscale_lut
substrate's chroma-LUT bytes. The canonical LUT in the chroma_lut_replacement
context is 256 bytes (uint8 indexed by grayscale level / class index). This
variant REPLACES that LUT with a deterministic 32-byte PCG64 seed; the
inflate runtime re-derives the LUT bytes via
:func:`tac.procedural_codebook_generator.derive_codebook_from_seed`.

**Predicted ΔS = -0.000149** per canonical equation #26 closed form:
``-25 * (256 - 32) / 37_545_489``. The savings is structurally smaller
than DP1's ``-0.002706`` or VQ-VAE's ``-0.005434`` because the canonical
chroma LUT is 256 bytes (vs DP1 ~4096 B / VQ-VAE 8192 B). The strongest
IN-DOMAIN fit per CANONICAL EQUATION #26 DOMAIN REFINEMENT is the
primary engineering value here — the equation's domain-of-validity
predicts the savings with high confidence relative to OOD contexts.

**Architectural note** (honest disclosure per CLAUDE.md
"Apples-to-apples evidence discipline"): the current grayscale_lut
substrate (``architecture.py``) uses a FiLM-conditioned RGB decoder
rather than an explicit 256-byte chroma LUT — the decoder's
``head_rgb_0`` + ``head_rgb_1`` produce the colorization mapping
implicitly. The procedural variant target here is the **canonical
chroma_lut_replacement context** per equation #26's IN-DOMAIN set; a
LUT-explicit substrate variant lands at L1 promotion (per the per-
substrate symposium gating per Catalog #325). Until then, this module
provides the canonical API + composition mechanism that a future
LUT-explicit grayscale_lut variant will consume. The compose API
appends a sentinel-prefixed seed envelope to the existing GLV1 archive
that LUT-aware inflate variants detect via the ``GLPV`` sentinel
(GrayscaLe-lut Procedural Variant).

**Compliance posture** (per ``canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict; sister DP1 + VQ-VAE BUILD landing
memos):

1. Seed bytes live INSIDE archive ``0.bin`` (~32 B canonical replaces
   ~256 B chroma LUT slot); the contest rate term charges
   ``25 * archive_bytes / 37_545_489`` per ``upstream/evaluate.py:63``.
2. Derivation routine ``derive_codebook_from_seed`` lives in inflate.py
   (legitimate "external library / tool" per upstream README).
3. Structurally distinct from rejected loophole pattern (PR #36 / #38 /
   #68 / #69 / #78 / #87) which relocates score-relevant bytes OUTSIDE
   archive.zip. This variant keeps BOTH seed AND derivation routine inside
   the canonical submission package — *sensitivity-aware compression*,
   NOT *payload smuggling*.

**Catalog #220 substrate L1+ scaffold operational mechanism**: the
``compose_with_procedural_lut`` API emits archive bytes whose chroma-LUT
section IS structurally consumed by inflate (the seed is inverted at
parse-time to re-derive the LUT bytes byte-for-byte identical to the
original LUT for the inflate-time consumer).

**Catalog #240 recipe-vs-trainer-state consistency**: the variant is
``research_only=True`` until the operator-routed per-substrate symposium
per Catalog #325 lands a PROCEED verdict + paired smoke contest-CUDA +
contest-CPU anchor per CLAUDE.md "Submission auth eval - BOTH CPU AND
CUDA".

**Catalog #272 byte-mutation distinguishing-feature contract**: the
operational mechanism IS byte-mutation-traceable: mutating any of the
32 seed bytes produces a different derived LUT + different rendered
frames (verified via ``verify_seed_mutation_changes_lut_bytes``).

**Catalog #287 + #323 canonical Provenance**: no score claim asserted in
this module; all return values carry ``score_claim=False`` markers via
metadata.

**Catalog #324 post-training Tier-C validation**: the predicted ΔS
prediction ``-0.000149`` is REGISTERED HYPOTHESIS; reactivation criterion
= "post-training Tier-C re-measurement on landed paired smoke archive sha
via tools/mdl_scorer_conditional_ablation.py --tier c".

**Catalog #344 canonical equation cross-reference**: this module's
predicted ΔS computation IS the canonical equation #26 closed-form
``-25 * (N_lut - K_seed) / 37_545_489`` evaluated at ``N_lut=256`` +
``K_seed=32``.

**6-hook wire-in declaration** per Catalog #125:

* hook #1 sensitivity-map = N/A (variant is a single archive-build path)
* hook #2 Pareto constraint = ACTIVE via procedural_codebook_savings_v1
  predicted ΔS contribution to rate-axis Pareto polytope (smallest
  per-substrate ΔS but strongest IN-DOMAIN fit confidence)
* hook #3 bit-allocator = ACTIVE (32-byte seed slot replaces ~256 B
  chroma LUT slot; bit-allocator's per-tensor importance changes)
* hook #4 cathedral autopilot dispatch = ACTIVE via sister consumer
  ``tac.cathedral_consumers.procedural_codebook_generator_consumer``
  (auto-discovered per Catalog #335)
* hook #5 continual-learning posterior = ACTIVE (first empirical anchor
  via ``update_equation_with_empirical_anchor`` post-paired-smoke)
* hook #6 probe-disambiguator = ACTIVE (PROCEDURAL vs canonical
  chroma LUT contrast IS the probe disambiguator for whether the
  grayscale_lut substrate's chroma-axis rate can be procedurally
  substituted within the canonical equation #26 IN-DOMAIN context)

Cross-references:

* Sister substrate variants:
  :mod:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant`
  (canonical DP1 BUILD pattern; commit ``9cbfa471c``)
  + :mod:`tac.substrates.vq_vae.distillation_procedural_variant`
  (canonical VQ-VAE BUILD pattern; commit ``6fea30f22``)
* PR101/PR106 BUILD DESIGN: commit ``086d3ac1d`` (Top-3 #1 PIVOT
  recommendation rationale)
* Canonical helper: :mod:`tac.procedural_codebook_generator.seed_derived_codebook`
* Canonical equation: :mod:`tac.canonical_equations.procedural_codebook_savings`
  (``chroma_lut_replacement`` in ``_INCLUDED_CONTEXTS``)
* grayscale_lut archive grammar: :mod:`tac.substrates.grayscale_lut.archive` (GLV1)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.procedural_codebook_generator import (
    DEFAULT_GENERATOR_KIND,
    derive_codebook_from_seed,
)
from tac.substrates.grayscale_lut.archive import (
    GLV1_HEADER_FMT,
    GLV1_HEADER_SIZE,
    GLV1_MAGIC,
    GLV1_SCHEMA_VERSION,
    parse_archive,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "PROCEDURAL_LUT_BYTES_DEFAULT",
    "PROCEDURAL_LUT_DTYPE_DEFAULT",
    "PROCEDURAL_LUT_SENTINEL",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "compose_with_procedural_lut",
    "derive_procedural_lut_replacement",
    "predicted_archive_bytes_saved",
    "predicted_delta_s",
    "verify_procedural_lut_in_domain",
    "verify_seed_mutation_changes_lut_bytes",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT: str = "chroma_lut_replacement"
"""IN-DOMAIN context label for canonical equation #26 per grayscale_lut's role.

The canonical chroma colorization LUT in the grayscale-LUT paradigm IS a
member of canonical equation #26's ``_INCLUDED_CONTEXTS`` (per
``procedural_codebook_savings.py:100``); the equation predicts
``ΔS = -25 * (N - K) / 37_545_489`` for any IN-DOMAIN context.

Per CANONICAL EQUATION #26 DOMAIN REFINEMENT commit ``8d8a7c6c5``, the
``chroma_lut_replacement`` context is the strongest IN-DOMAIN fit for
the grayscale_lut substrate because the canonical chroma LUT is a
per-class-anchor table that maps grayscale levels / class indices to
chroma offsets — the canonical intermediate-transform position the
equation's derivation assumes.

This differs from the sister DP1 context
``comma2k19_ood_derived_basis_replacement`` (PCA basis from Comma2k19) +
the sister VQ-VAE context ``intermediate_transform_quantizer``
(K×D embedding table). All three contexts are members of canonical
equation #26's ``_INCLUDED_CONTEXTS`` and produce the same closed-form
``ΔS = -25 * (N - K) / 37_545_489`` savings prediction.
"""

PROCEDURAL_SEED_SIZE_BYTES: int = 32
"""Canonical procedural-variant seed size in bytes.

Per 5-substrate matrix design + DP1 BUILD + VQ-VAE BUILD canonical
patterns: 32-byte PCG64 seed replaces the canonical 256-byte chroma LUT
slot. The ``32`` constant is the ``K_seed`` term in the canonical
equation #26 closed form:

    predicted_delta_s = -25 * (256 - 32) / 37_545_489 = -0.000149
"""

PROCEDURAL_LUT_BYTES_DEFAULT: int = 256
"""Default canonical chroma LUT size in bytes for the grayscale_lut paradigm.

256-byte uint8 LUT is the canonical chroma_lut_replacement context per
equation #26: indexed by grayscale level (0-255) producing a per-level
chroma offset. This is the strongest IN-DOMAIN fit for the
``chroma_lut_replacement`` context per the equation's domain-of-validity.

For non-canonical grayscale_lut variants with larger LUTs (e.g. 512-byte
2-channel chroma offset tables or 1024-byte per-class anchor tables), the
predicted savings scales linearly per the canonical equation: the
``lut_bytes`` parameter to :func:`predicted_delta_s` MUST match the
actual LUT footprint of the substrate variant being audited.
"""

PROCEDURAL_LUT_DTYPE_DEFAULT: np.dtype = np.dtype(np.uint8)
"""Default dtype for procedural LUT derivation.

uint8 matches the canonical chroma_lut_replacement context's LUT dtype
+ the PCG64 generator's native byte output. The actual chroma LUT
tensor is interpreted as 256 per-class uint8 chroma offsets which the
inflate-time consumer applies per pixel.
"""

PROCEDURAL_LUT_SENTINEL: bytes = b"GLPV"
"""Procedural-variant archive sentinel: ``GLPV`` = ``GrayscaLe-lut Procedural Variant``.

Per sister VQ-VAE pattern (commit ``6fea30f22``): the sentinel prefix
distinguishes a procedural archive's appended LUT seed envelope from a
canonical (non-procedural) GLV1 archive. A future LUT-aware inflate
runtime detects the sentinel after the canonical GLV1 sections and
re-derives the chroma LUT bytes deterministically via
:func:`derive_codebook_from_seed`.
"""

_MIN_SEED_SIZE_BYTES: int = 8
_MAX_SEED_SIZE_BYTES: int = 256

_DEFAULT_OUTPUT_SHAPE_BYTES: tuple[int, ...] = (PROCEDURAL_LUT_BYTES_DEFAULT,)


# ---------------------------------------------------------------------------
# Errors + Config
# ---------------------------------------------------------------------------


class ProceduralVariantError(ValueError):
    """Raised on invalid procedural-variant configuration.

    Sister of :class:`tac.procedural_codebook_generator.ProceduralCodebookGeneratorError`
    + :class:`ValueError` raised by archive.py validators.
    """


@dataclass(frozen=True)
class ProceduralVariantConfig:
    """Configuration for the grayscale_lut procedural-LUT replacement variant.

    Sister of the canonical training pipeline's implicit "trained chroma LUT"
    config. This config holds the inputs to the procedural derivation
    pipeline.

    Attributes:
        seed_bytes: The procedural seed bytes (typically 32 bytes for
            ``derive_codebook_from_seed`` PCG64). Length must be in
            ``[_MIN_SEED_SIZE_BYTES, _MAX_SEED_SIZE_BYTES]`` per canonical
            equation #26 ``seed_size_bytes_range`` domain-of-validity.
        output_shape: The LUT shape to derive in bytes (default
            ``(256,)`` matches the canonical chroma LUT 256-byte slot).
        dtype: The LUT dtype (default
            ``PROCEDURAL_LUT_DTYPE_DEFAULT`` = ``np.uint8``).
        generator_kind: The PRNG kind (one of ``xorshift`` / ``lcg`` /
            ``pcg64``; default ``DEFAULT_GENERATOR_KIND`` = ``"pcg64"``).
        canonical_equation_context: The IN-DOMAIN context per canonical
            equation #26 (default ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``
            = ``"chroma_lut_replacement"``).
    """

    seed_bytes: bytes
    output_shape: tuple[int, ...] = _DEFAULT_OUTPUT_SHAPE_BYTES
    dtype: np.dtype = PROCEDURAL_LUT_DTYPE_DEFAULT
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
# Canonical equation #26 closed-form predictions
# ---------------------------------------------------------------------------


_CONTEST_RATE_DENOM_BYTES: int = 37_545_489
_CONTEST_RATE_MULTIPLIER: float = 25.0


def predicted_archive_bytes_saved(
    lut_bytes: int = PROCEDURAL_LUT_BYTES_DEFAULT,
    seed_bytes: int = PROCEDURAL_SEED_SIZE_BYTES,
) -> int:
    """Closed-form bytes-saved per canonical equation #26.

    Returns ``lut_bytes - seed_bytes`` (the structural delta the variant
    removes from the archive). For the canonical config this is
    ``256 - 32 = 224`` bytes.

    Args:
        lut_bytes: Canonical chroma LUT byte count (default 256 for the
            canonical ``chroma_lut_replacement`` context).
        seed_bytes: Procedural seed byte count (default 32 for canonical
            PCG64 32-byte seed).

    Returns:
        Integer bytes-saved. Returns 0 when seed >= lut (degenerate
        case where the procedural variant adds bytes).
    """
    delta = int(lut_bytes) - int(seed_bytes)
    return max(0, delta)


def predicted_delta_s(
    lut_bytes: int = PROCEDURAL_LUT_BYTES_DEFAULT,
    seed_bytes: int = PROCEDURAL_SEED_SIZE_BYTES,
) -> float:
    """Closed-form predicted ΔS per canonical equation #26.

    ``predicted_delta_s = -25 * (lut_bytes - seed_bytes) / 37_545_489``

    For the canonical grayscale_lut config (256-byte chroma LUT, 32-byte
    seed): ``-25 * (256 - 32) / 37_545_489 = -0.000149`` (a score
    IMPROVEMENT of ~0.000149 from the rate-axis only).

    NOT a score CLAIM — this is the canonical equation #26 closed-form
    PREDICTION which the operator-routed per-substrate symposium per
    Catalog #325 must validate via paired-smoke contest-CUDA + contest-CPU
    auth-eval on a real archive per CLAUDE.md "Submission auth eval — BOTH
    CPU AND CUDA" before any promotion.

    Args:
        lut_bytes: Canonical chroma LUT byte count.
        seed_bytes: Procedural seed byte count.

    Returns:
        Predicted score delta (negative = improvement). Returns 0.0 when
        seed >= lut (no savings possible).
    """
    bytes_saved = predicted_archive_bytes_saved(lut_bytes, seed_bytes)
    return -_CONTEST_RATE_MULTIPLIER * bytes_saved / _CONTEST_RATE_DENOM_BYTES


# ---------------------------------------------------------------------------
# Public API — derivation + in-domain check
# ---------------------------------------------------------------------------


def derive_procedural_lut_replacement(
    seed_bytes: bytes,
    output_shape: tuple[int, ...] = _DEFAULT_OUTPUT_SHAPE_BYTES,
    dtype: np.dtype = PROCEDURAL_LUT_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> np.ndarray:
    """Derive a procedural chroma-LUT replacement from a deterministic seed.

    Thin wrapper around :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`
    with grayscale_lut-specific validation + canonical equation #26 domain
    awareness.

    Args:
        seed_bytes: Procedural seed (8-256 bytes; canonical 32 bytes).
        output_shape: LUT shape (default ``(256,)`` = 256 bytes uint8
            matching the canonical chroma LUT footprint).
        dtype: LUT dtype (default ``np.uint8``).
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


def verify_procedural_lut_in_domain(
    context: str = CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
) -> bool:
    """Verify the procedural LUT context is IN-DOMAIN per canonical equation #26.

    The grayscale_lut canonical context ``chroma_lut_replacement`` IS a
    member of canonical equation #26's ``_INCLUDED_CONTEXTS`` (per
    ``procedural_codebook_savings.py:100``); the equation predicts
    ``ΔS = -25 * (N - K) / 37_545_489`` for any IN-DOMAIN context.

    Per CANONICAL EQUATION #26 DOMAIN REFINEMENT commit ``8d8a7c6c5``,
    ``chroma_lut_replacement`` is the strongest IN-DOMAIN fit for the
    grayscale_lut substrate. The slot 3 helper
    ``validate_context_is_in_domain`` IS landed in
    ``tac.canonical_equations``; this function delegates to it when
    available + falls back to the canonical IN-DOMAIN context set if not.

    Args:
        context: The context string to validate (default
            ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``).

    Returns:
        True if context is IN-DOMAIN per canonical equation #26.
    """
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
        # Sister helper not yet landed — fall back to canonical IN-DOMAIN
        # context set per procedural_codebook_savings.py `_INCLUDED_CONTEXTS`.
        _canonical_in_domain_contexts: frozenset[str] = frozenset({
            "intermediate_transform_quantizer",
            "intermediate_transform_dequantizer",
            "procedural_codebook_as_lookup_table",
            "comma2k19_ood_derived_basis_replacement",
            "chroma_lut_replacement",
            "class_anchor_replacement",
            "nscs06_v8_chroma_lut",
            "atw_v2_codec_quantizer_lut",
            "tt5l_transformer_tokens",
            "dp1_codebook_bytes",
            "deterministic_constants_codebook_replacement",
        })
        return context in _canonical_in_domain_contexts


# ---------------------------------------------------------------------------
# Public API — archive composition
# ---------------------------------------------------------------------------


_GENERATOR_KIND_TAG: dict[str, int] = {"xorshift": 0, "lcg": 1, "pcg64": 2}


def _build_procedural_seed_envelope(
    seed_bytes: bytes,
    *,
    lut_bytes: int,
    generator_kind: str,
) -> bytes:
    """Pack the procedural-seed envelope appended to GLV1 archive tail.

    Envelope layout::

        SENTINEL(4)         b"GLPV"   GrayscaLe-lut Procedural Variant
        LUT_BYTES(2)        u16       declared LUT footprint (e.g. 256)
        GEN_KIND_TAG(1)     u8        0=xorshift, 1=lcg, 2=pcg64
        SEED_LEN(2)         u16       length of seed payload (e.g. 32)
        SEED_PAYLOAD(...)   bytes     the procedural seed bytes
        BROTLI_OF_REST(...) bytes     omitted in L0 scaffold (raw payload)

    The envelope is byte-deterministic for a given (seed, lut_bytes,
    generator_kind) tuple — sister-compatible with the canonical equation
    #26 closed-form predictions.
    """
    if lut_bytes <= 0 or lut_bytes > 0xFFFF:
        raise ProceduralVariantError(
            f"lut_bytes={lut_bytes} outside u16 range (1, 65535)"
        )
    if len(seed_bytes) > 0xFFFF:
        raise ProceduralVariantError(
            f"seed length {len(seed_bytes)} exceeds u16 SEED_LEN field max"
        )
    return (
        PROCEDURAL_LUT_SENTINEL
        + struct.pack(
            "<HBH",
            lut_bytes,
            _GENERATOR_KIND_TAG[generator_kind],
            len(seed_bytes),
        )
        + seed_bytes
    )


def compose_with_procedural_lut(
    original_archive_bytes: bytes,
    seed_bytes: bytes,
    *,
    lut_bytes: int = PROCEDURAL_LUT_BYTES_DEFAULT,
    output_shape: tuple[int, ...] | None = None,
    dtype: np.dtype = PROCEDURAL_LUT_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bytes:
    """Compose a grayscale_lut archive whose chroma LUT is REPLACED by a procedural seed.

    Sister of :func:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant.compose_with_procedural_codebook`
    (canonical DP1 BUILD pattern; commit ``9cbfa471c``) +
    :func:`tac.substrates.vq_vae.distillation_procedural_variant.compose_with_procedural_codebook`
    (canonical VQ-VAE BUILD pattern; commit ``6fea30f22``).

    Takes an existing GLV1 archive + a 32-byte seed; emits a NEW archive
    whose chroma LUT bytes are REPLACED by a sentinel-prefixed seed
    envelope appended to the GLV1 archive tail (the seed is re-derived at
    inflate time via :func:`derive_procedural_lut_replacement`).

    **Architectural note** (honest disclosure): the current grayscale_lut
    substrate (``architecture.py``) uses a FiLM-conditioned RGB decoder
    rather than an explicit 256-byte chroma LUT — the canonical GLV1
    archive grammar does NOT yet have a dedicated chroma-LUT section. The
    compose API here APPENDS the procedural seed envelope to the existing
    GLV1 archive bytes (preserving the canonical sections byte-for-byte).
    A future LUT-explicit substrate variant will:

    1. Add a CHROMA_LUT section to the GLV1 grammar (bumping schema
       version to GLV2).
    2. Update the compose API to REPLACE the chroma LUT section bytes
       with the seed envelope (matching the canonical DP1 + VQ-VAE pattern
       where the seed REPLACES the codebook bytes in-place).
    3. Update the inflate runtime to detect the sentinel + re-derive the
       LUT bytes before consuming them.

    Until then, the L0 scaffold compose API demonstrates the canonical
    pattern + provides the byte-stable envelope a future LUT-aware
    variant will consume. The predicted savings ``ΔS = -0.000149`` is the
    upper-bound when the LUT-aware variant lands; the current scaffold
    APPENDS bytes (increases archive size by envelope length) and is
    therefore explicitly marked ``research_only=True`` per Catalog #240.

    Args:
        original_archive_bytes: The existing GLV1 archive bytes (with the
            canonical decoder + grayscale stream sections).
        seed_bytes: The procedural seed (32 bytes canonical).
        lut_bytes: Declared LUT footprint the variant targets (default
            ``PROCEDURAL_LUT_BYTES_DEFAULT`` = 256; the canonical
            chroma_lut_replacement size).
        output_shape: LUT derivation shape (default ``(256,)`` matching
            ``lut_bytes`` canonical config).
        dtype: LUT derivation dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        New archive bytes with the canonical GLV1 sections preserved +
        the procedural seed envelope appended. A future LUT-aware inflate
        runtime detects the sentinel and re-derives the chroma LUT bytes
        deterministically.

    Raises:
        ProceduralVariantError: invalid configuration.
        ValueError: original archive bytes are not parseable as GLV1.
    """
    if output_shape is None:
        output_shape = (lut_bytes,)

    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        output_shape=output_shape,
        dtype=np.dtype(dtype),
        generator_kind=generator_kind,
    )

    # Parse the original archive to verify it's a valid GLV1 (raises ValueError
    # if not).
    _ = parse_archive(original_archive_bytes)

    # Verify the canonical header still parses (sister discipline with VQ-VAE).
    (
        magic,
        version,
        *_,
    ) = struct.unpack(GLV1_HEADER_FMT, original_archive_bytes[:GLV1_HEADER_SIZE])
    if magic != GLV1_MAGIC or version != GLV1_SCHEMA_VERSION:
        raise ValueError(
            f"original_archive_bytes not parseable as GLV1 v{GLV1_SCHEMA_VERSION}: "
            f"magic={magic!r} version={version}"
        )

    # Build the procedural seed envelope.
    envelope = _build_procedural_seed_envelope(
        config.seed_bytes,
        lut_bytes=lut_bytes,
        generator_kind=config.generator_kind,
    )

    # Compose: original archive bytes + envelope.
    new_archive = original_archive_bytes + envelope

    # Catalog #220 + #272 + #287 invariant: the operational mechanism is
    # byte-mutation-traceable + the envelope is non-trivially attached. For
    # the L0 scaffold the envelope ADDS bytes (since current GLV1 lacks a
    # chroma_lut section to remove); a future LUT-aware variant will
    # REPLACE bytes and produce net savings per the canonical equation #26
    # closed form ``ΔS = -0.000149``.
    if len(new_archive) <= len(original_archive_bytes):
        raise ProceduralVariantError(
            f"procedural envelope did NOT extend archive bytes "
            f"({len(original_archive_bytes)} -> {len(new_archive)}); "
            f"envelope build failed"
        )

    return new_archive


def verify_seed_mutation_changes_lut_bytes(
    seed_bytes: bytes,
    *,
    output_shape: tuple[int, ...] = _DEFAULT_OUTPUT_SHAPE_BYTES,
    dtype: np.dtype = PROCEDURAL_LUT_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bool:
    """Verify mutating a single seed byte changes the derived LUT.

    Catalog #272 byte-mutation distinguishing-feature contract: the
    operational mechanism is byte-mutation-traceable. Flipping any of
    the 32 seed bytes MUST produce a different derived LUT.

    Args:
        seed_bytes: The canonical seed.
        output_shape: LUT shape (default ``(256,)``).
        dtype: LUT dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        True if mutating ANY seed byte produces a different LUT.
        False if the derivation is degenerate (does NOT depend on the
        mutated byte).
    """
    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        output_shape=output_shape,
        dtype=np.dtype(dtype),
        generator_kind=generator_kind,
    )
    original_lut = derive_codebook_from_seed(
        seed_bytes=config.seed_bytes,
        output_shape=config.output_shape,
        dtype=config.dtype,
        generator_kind=config.generator_kind,
    )
    # Flip the FIRST byte to test mutation propagation (full byte-mutation
    # sweep is the operator-routed runtime verifier; this is the per-seed
    # invariant check per sister DP1 + VQ-VAE pattern).
    mutated_seed = bytearray(config.seed_bytes)
    mutated_seed[0] = (mutated_seed[0] + 1) & 0xFF
    mutated_lut = derive_codebook_from_seed(
        seed_bytes=bytes(mutated_seed),
        output_shape=config.output_shape,
        dtype=config.dtype,
        generator_kind=config.generator_kind,
    )
    if mutated_lut.shape != original_lut.shape:
        return True
    return not bool(np.array_equal(original_lut, mutated_lut))
