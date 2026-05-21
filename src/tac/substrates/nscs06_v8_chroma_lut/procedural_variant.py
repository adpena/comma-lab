# SPDX-License-Identifier: MIT
"""nscs06 v8 chroma-LUT procedural-variant API per canonical equation #26 IN-DOMAIN.

Per WAVE-3-NSCS06-V8-CHROMA-LUT-SUBSTRATE-BUILD 2026-05-21 + sister
PROCEDURAL VARIANT canonical pattern (grayscale_lut commit ``f037d1144`` +
DP1 commit ``9cbfa471c`` + VQ-VAE commit ``6fea30f22``) + canonical equation
``procedural_codebook_from_seed_compression_savings_v1`` (sister
``src/tac/canonical_equations/procedural_codebook_savings.py``).

**Canonical equation #26 IN-DOMAIN context** for nscs06_v8_chroma_lut is
``nscs06_v8_chroma_lut`` (per ``_INCLUDED_CONTEXTS`` in
``procedural_codebook_savings.py:102``). The canonical 4096-byte chroma
LUT is the strongest IN-DOMAIN fit per the equation's domain-of-validity
+ the CASCADE COMPRESSION symposium PRIORITY 3 Daubechies + Mallat
multi-scale partition discovery framing.

**Predicted ΔS = -0.002706** per canonical equation #26 closed form:
``-25 * (4096 - 32) / 37_545_489`` [prediction; canonical-equation-26-grounded;
per-substrate-symposium-pending].

**Architectural note** (honest disclosure per CLAUDE.md "Apples-to-apples
evidence discipline"): the current v8 chroma-LUT substrate ships BOTH
``CH08_SCHEMA_VERSION_INLINE_LUT`` (v1; inline 4096-byte LUT) and
``CH08_SCHEMA_VERSION_PROCEDURAL_SEED`` (v2; 32-byte seed replaces LUT).
The procedural variant in this module wraps the v2 path via the canonical
``derive_codebook_from_seed`` helper. Per the sister grayscale_lut /
DP1 / VQ-VAE canonical pattern, this module ALSO exposes a deferred
``compose_with_procedural_lut`` API for parsing v1 -> emitting v2 archives
once a per-substrate symposium ratifies the variant.

**Compliance posture** (per
``canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict; sister DP1 + VQ-VAE + grayscale_lut
BUILD landing memos):

1. Seed bytes live INSIDE archive ``0.bin`` (CH08 v2 LUT_PAYLOAD slot).
   The contest rate term charges ``25 * archive_bytes / 37_545_489`` per
   ``upstream/evaluate.py:63``.
2. Derivation routine ``derive_codebook_from_seed`` lives in
   ``tac.procedural_codebook_generator`` (legitimate "external library /
   tool" per upstream README).
3. Structurally distinct from rejected loophole pattern (PR #36 / #38 /
   #68 / #69 / #78 / #87) which relocates score-relevant bytes OUTSIDE
   archive.zip. This variant keeps BOTH seed AND derivation routine inside
   the canonical submission package — *sensitivity-aware compression*,
   NOT *payload smuggling*.

**Catalog #220 substrate L1+ scaffold operational mechanism**: the CH08 v2
LUT_PAYLOAD bytes ARE byte-mutation-traceable: mutating any of the 32 seed
bytes produces a different derived LUT + different rendered frames per
:func:`verify_seed_mutation_changes_lut_bytes`.

**Catalog #272 byte-mutation distinguishing-feature contract**: the
operational mechanism IS byte-mutation-traceable; the distinguishing
feature IS the chroma-LUT replacement via procedural seed.

**Catalog #287 + #323 canonical Provenance**: no score claim asserted in
this module. Public APIs return plain Python/numpy values; lane-level
authority is carried by the registry + landing memo, not by return-value
metadata.

**Catalog #324 post-training Tier-C validation**: the predicted ΔS
prediction ``-0.002706`` is REGISTERED HYPOTHESIS; reactivation criterion
= "post-training Tier-C re-measurement on landed paired smoke archive sha
via tools/mdl_scorer_conditional_ablation.py --tier c".

**6-hook wire-in declaration** per Catalog #125:

* hook #1 sensitivity-map = N/A (variant is a single archive-build path)
* hook #2 Pareto constraint = ACTIVE via procedural_codebook_savings_v1
  predicted ΔS contribution to rate-axis Pareto polytope
* hook #3 bit-allocator = PLANNED (32-byte seed slot replaces 4096-byte
  chroma LUT slot only after per-substrate symposium PROCEED verdict)
* hook #4 cathedral autopilot dispatch = ACTIVE via sister consumer
  ``tac.cathedral_consumers.procedural_codebook_generator_consumer``
  (auto-discovered per Catalog #335)
* hook #5 continual-learning posterior = ACTIVE (first empirical anchor
  via ``update_equation_with_empirical_anchor`` post-paired-smoke)
* hook #6 probe-disambiguator = ACTIVE (PROCEDURAL vs canonical chroma
  LUT contrast IS the probe disambiguator for whether the v8 substrate's
  chroma-axis rate can be procedurally substituted within the canonical
  equation #26 IN-DOMAIN context)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac.procedural_codebook_generator import (
    DEFAULT_GENERATOR_KIND,
    derive_codebook_from_seed,
)

from .architecture import (
    CHROMA_LUT_BYTES_DEFAULT,
    CHROMA_LUT_DTYPE_DEFAULT,
    PROCEDURAL_SEED_SIZE_BYTES,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "PROCEDURAL_LUT_SENTINEL",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "derive_procedural_chroma_lut_replacement",
    "predicted_archive_bytes_saved",
    "predicted_delta_s",
    "verify_procedural_lut_in_domain",
    "verify_seed_mutation_changes_lut_bytes",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT: str = "nscs06_v8_chroma_lut"
"""IN-DOMAIN context label per canonical equation #26 ``_INCLUDED_CONTEXTS``.

Per ``src/tac/canonical_equations/procedural_codebook_savings.py:102``
``_INCLUDED_CONTEXTS['nscs06_v8_chroma_lut']``: the canonical chroma LUT
in the v8 substrate IS the strongest IN-DOMAIN fit per the
``CANONICAL EQUATION #26 DOMAIN REFINEMENT`` (commit ``8d8a7c6c5``) +
CASCADE COMPRESSION symposium commit ``d125af6c3`` PRIORITY 3
(Daubechies + Mallat multi-scale partition discovery framing).
"""

PROCEDURAL_LUT_SENTINEL: bytes = b"NV8C"
"""Procedural-variant archive sentinel: ``NV8C`` = ``Nscs06 V8 Chroma``.

Sister of grayscale_lut ``PROCEDURAL_LUT_SENTINEL = b"GLPV"`` + DP1 +
VQ-VAE canonical sentinel pattern. The v8 substrate uses the CH08
schema version byte (``CH08_SCHEMA_VERSION_PROCEDURAL_SEED = 2``) to
distinguish v1-inline vs v2-procedural archives natively; this
sentinel is reserved for cross-format envelope wrapping (e.g.
composing v1 -> v2 in a future tooling pass).
"""

_MIN_SEED_SIZE_BYTES: int = 8
_MAX_SEED_SIZE_BYTES: int = 256

_CONTEST_RATE_DENOM_BYTES: int = 37_545_489
_CONTEST_RATE_MULTIPLIER: float = 25.0


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
    """Configuration for the v8 chroma-LUT procedural-replacement variant.

    Attributes:
        seed_bytes: Procedural seed bytes (typically 32 bytes for PCG64).
            Length must be in ``[_MIN_SEED_SIZE_BYTES, _MAX_SEED_SIZE_BYTES]``
            per canonical equation #26 ``seed_size_bytes_range``.
        chroma_lut_bytes: Declared LUT footprint the variant targets
            (default ``CHROMA_LUT_BYTES_DEFAULT`` = 4096).
        dtype: LUT dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).
        canonical_equation_context: IN-DOMAIN context per canonical equation
            #26 (default ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT`` =
            ``"nscs06_v8_chroma_lut"``).
    """

    seed_bytes: bytes
    chroma_lut_bytes: int = CHROMA_LUT_BYTES_DEFAULT
    dtype: np.dtype = CHROMA_LUT_DTYPE_DEFAULT
    generator_kind: str = DEFAULT_GENERATOR_KIND
    canonical_equation_context: str = CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT

    def __post_init__(self) -> None:
        """Validate canonical invariants.

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
        if self.chroma_lut_bytes <= 0 or self.chroma_lut_bytes > 65535:
            raise ProceduralVariantError(
                f"chroma_lut_bytes {self.chroma_lut_bytes} outside u16 range"
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


def predicted_archive_bytes_saved(
    lut_bytes: int = CHROMA_LUT_BYTES_DEFAULT,
    seed_bytes: int = PROCEDURAL_SEED_SIZE_BYTES,
) -> int:
    """Closed-form bytes-saved per canonical equation #26.

    Returns ``lut_bytes - seed_bytes`` (the structural delta the variant
    removes from the archive). For the canonical config this is
    ``4096 - 32 = 4064`` bytes.

    Returns 0 when ``seed_bytes >= lut_bytes`` (degenerate case where the
    procedural variant would ADD bytes; not allowed per canonical equation
    #26 domain-of-validity).
    """
    delta = int(lut_bytes) - int(seed_bytes)
    return max(0, delta)


def predicted_delta_s(
    lut_bytes: int = CHROMA_LUT_BYTES_DEFAULT,
    seed_bytes: int = PROCEDURAL_SEED_SIZE_BYTES,
) -> float:
    """Closed-form predicted ΔS per canonical equation #26.

    ``predicted_delta_s = -25 * (lut_bytes - seed_bytes) / 37_545_489``

    For the canonical v8 config (4096-byte chroma LUT, 32-byte seed):
    ``-25 * (4096 - 32) / 37_545_489 ≈ -0.002706`` (a score IMPROVEMENT of
    ~0.002706 from the rate-axis only).

    NOT a score CLAIM — this is the canonical equation #26 closed-form
    PREDICTION which the operator-routed per-substrate symposium per
    Catalog #325 must validate via paired-smoke contest-CUDA + contest-CPU
    auth-eval on a real archive per CLAUDE.md "Submission auth eval — BOTH
    CPU AND CUDA" before any promotion.
    """
    bytes_saved = predicted_archive_bytes_saved(lut_bytes, seed_bytes)
    return -_CONTEST_RATE_MULTIPLIER * bytes_saved / _CONTEST_RATE_DENOM_BYTES


# ---------------------------------------------------------------------------
# Derivation + in-domain check
# ---------------------------------------------------------------------------


def derive_procedural_chroma_lut_replacement(
    seed_bytes: bytes,
    *,
    grayscale_levels: int = 16,
    num_segnet_classes: int = 5,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> np.ndarray:
    """Derive the procedural chroma LUT from a deterministic seed.

    Thin wrapper around :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`
    with v8-specific validation + canonical equation #26 domain awareness.

    Args:
        seed_bytes: Procedural seed (8-256 bytes; canonical 32 bytes).
        grayscale_levels: LUT level dimension (default 16).
        num_segnet_classes: LUT class dimension (default 5).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        np.ndarray ``(grayscale_levels, num_segnet_classes, 3)`` uint8 LUT,
        deterministically derived from ``seed_bytes`` via the canonical
        generator.

    Raises:
        ProceduralVariantError: invalid inputs per
            :meth:`ProceduralVariantConfig.__post_init__` invariants.
    """
    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        generator_kind=generator_kind,
    )
    flat = derive_codebook_from_seed(
        seed_bytes=config.seed_bytes,
        output_shape=(grayscale_levels * num_segnet_classes * 3,),
        dtype=CHROMA_LUT_DTYPE_DEFAULT,
        generator_kind=config.generator_kind,
    )
    return flat.reshape(grayscale_levels, num_segnet_classes, 3)


def verify_procedural_lut_in_domain(
    context: str = CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
) -> bool:
    """Verify the procedural LUT context is IN-DOMAIN per canonical equation #26.

    The v8 canonical context ``nscs06_v8_chroma_lut`` IS a member of
    canonical equation #26's ``_INCLUDED_CONTEXTS`` (per
    ``procedural_codebook_savings.py:102``); the equation predicts
    ``ΔS = -25 * (N - K) / 37_545_489`` for any IN-DOMAIN context.

    Args:
        context: The context string to validate (default
            ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``).

    Returns:
        True if context is IN-DOMAIN per canonical equation #26.
    """
    try:
        from tac.canonical_equations.procedural_codebook_savings import (
            validate_context_is_in_domain,
        )

        return bool(
            validate_context_is_in_domain(
                context=context,
                raise_on_excluded=False,
            )
        )
    except ImportError:
        # Fall-back IN-DOMAIN set mirroring procedural_codebook_savings.py
        # `_INCLUDED_CONTEXTS` per sister discipline.
        _canonical_in_domain_contexts: frozenset[str] = frozenset(
            {
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
            }
        )
        return context in _canonical_in_domain_contexts


def verify_seed_mutation_changes_lut_bytes(
    seed_a: bytes,
    seed_b: bytes,
    *,
    grayscale_levels: int = 16,
    num_segnet_classes: int = 5,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bool:
    """Catalog #272 byte-mutation distinguishing-feature contract verifier.

    Mutating any seed byte MUST yield different derived LUT bytes. Returns
    True when ``seed_a != seed_b`` implies ``derived_a != derived_b``.

    Args:
        seed_a, seed_b: Two distinct seed byte strings (must differ in at
            least one byte).
        grayscale_levels, num_segnet_classes: LUT dimensions.
        generator_kind: PRNG kind.

    Returns:
        True if mutating the seed produces different LUT bytes.
    """
    if seed_a == seed_b:
        raise ProceduralVariantError("seed_a and seed_b must differ in at least one byte")
    lut_a = derive_procedural_chroma_lut_replacement(
        seed_a,
        grayscale_levels=grayscale_levels,
        num_segnet_classes=num_segnet_classes,
        generator_kind=generator_kind,
    )
    lut_b = derive_procedural_chroma_lut_replacement(
        seed_b,
        grayscale_levels=grayscale_levels,
        num_segnet_classes=num_segnet_classes,
        generator_kind=generator_kind,
    )
    return not np.array_equal(lut_a, lut_b)
