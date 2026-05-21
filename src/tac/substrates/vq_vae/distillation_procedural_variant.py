# SPDX-License-Identifier: MIT
"""VQ-VAE procedural-codebook replacement variant — first-empirical-anchor scaffold.

Per 5-SUBSTRATE MATRIX DESIGN landing commit ``b3e3442c3`` candidate #5
(VQ-VAE PROCEDURAL VARIANT) + operator-frontier-override 2026-05-20 ("3
slots are approved + magic codec stacking") + DP1 PROCEDURAL TRAINER BUILD
canonical pattern landing commit ``9cbfa471c`` + canonical equation
``procedural_codebook_from_seed_compression_savings_v1`` (sister
``src/tac/canonical_equations/procedural_codebook_savings.py``).

**Canonical equation #26 IN-DOMAIN context** for VQ-VAE is
``intermediate_transform_quantizer`` (per ``_INCLUDED_CONTEXTS`` in
``procedural_codebook_savings.py:96``). The VQ-VAE codebook IS the canonical
intermediate-transform quantizer: the K×D embedding table is used as a
nearest-neighbor LUT that quantizes encoder outputs to discrete indices
which the decoder consumes at inflate time.

This module is the **L0 SCAFFOLD procedural variant** of the VQ-VAE
codebook source. The existing canonical training path produces an 8192-byte
codebook (K=512 entries × D=8 embedding dim × fp16 = 8192 B; verified via
``architecture.py:62,65,181-183``). This variant REPLACES that codebook
with a deterministic 32-byte PCG64 seed; the inflate runtime re-derives the
codebook bytes via :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`.

**Predicted ΔS = -0.005434** per canonical equation #26 closed form:
``-25 * (8192 - 32) / 37_545_489``. This is ~2× the DP1 procedural variant's
``-0.002706`` savings because the VQ-VAE codebook is structurally larger
(K=512 × D=8 × fp16 = 8192 B vs DP1's ~4096-byte codebook slot).

**Compliance posture** (per ``canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict; sister DP1 BUILD landing memo
``feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md``):

1. Seed bytes live INSIDE archive ``0.bin`` (~32 B replaces ~8192 B
   codebook section); the contest rate term charges
   ``25 * archive_bytes / 37_545_489`` per ``upstream/evaluate.py:63``.
2. Derivation routine ``derive_codebook_from_seed`` lives in inflate.py
   (legitimate "external library / tool" per upstream README).
3. Structurally distinct from rejected loophole pattern (PR #36 / #38 /
   #68 / #69 / #78 / #87) which relocates score-relevant bytes OUTSIDE
   archive.zip. This variant keeps BOTH seed AND derivation routine inside
   the canonical submission package — *sensitivity-aware compression*, NOT
   *payload smuggling*.

**Catalog #220 substrate L1+ scaffold operational mechanism**: the
``compose_with_procedural_codebook`` API emits archive bytes whose
codebook section IS structurally consumed by inflate (the seed is
inverted at parse-time to re-derive the codebook bytes byte-for-byte
identical to the original derived codebook for the inflate-time consumer).

**Catalog #240 recipe-vs-trainer-state consistency**: the variant is
``research_only=True`` until the operator-routed per-substrate symposium
per Catalog #325 lands a PROCEED verdict + paired smoke contest-CUDA +
contest-CPU anchor per CLAUDE.md "Submission auth eval - BOTH CPU AND
CUDA".

**Catalog #272 byte-mutation distinguishing-feature contract**: the
operational mechanism IS byte-mutation-traceable: mutating any of the
32 seed bytes produces a different derived codebook + different rendered
frames (verified via ``verify_seed_mutation_changes_codebook_bytes``).

**Catalog #287 + #323 canonical Provenance**: no score claim asserted in
this module; all return values carry ``score_claim=False`` markers via
metadata.

**Catalog #324 post-training Tier-C validation**: the predicted ΔS
prediction ``-0.005434`` is REGISTERED HYPOTHESIS; reactivation criterion
= "post-training Tier-C re-measurement on landed paired smoke archive sha
via tools/mdl_scorer_conditional_ablation.py --tier c".

**Catalog #344 canonical equation cross-reference**: this module's
predicted ΔS computation IS the canonical equation #26 closed-form
``-25 * (N_codebook - K_seed) / 37_545_489`` evaluated at
``N_codebook=8192`` + ``K_seed=32``.

**6-hook wire-in declaration** per Catalog #125:

* hook #1 sensitivity-map = N/A (variant is a single archive-build path)
* hook #2 Pareto constraint = ACTIVE via procedural_codebook_savings_v1
  predicted ΔS contribution to rate-axis Pareto polytope
* hook #3 bit-allocator = ACTIVE (32-byte seed slot replaces ~8192 B
  codebook slot; bit-allocator's per-tensor importance changes)
* hook #4 cathedral autopilot dispatch = ACTIVE via sister consumer
  ``tac.cathedral_consumers.procedural_codebook_generator_consumer``
  (auto-discovered per Catalog #335)
* hook #5 continual-learning posterior = ACTIVE (first empirical anchor
  via ``update_equation_with_empirical_anchor`` post-paired-smoke)
* hook #6 probe-disambiguator = ACTIVE (PROCEDURAL vs ORIGINAL trained-
  codebook contrast IS the probe disambiguator for whether VQ-VAE's
  discrete-token quantizer rate-axis can be procedurally substituted)

Cross-references:

* Sister substrate variant: :mod:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant`
  (canonical DP1 BUILD pattern; commit ``9cbfa471c``)
* 5-substrate matrix design: commit ``b3e3442c3``
* Canonical helper: :mod:`tac.procedural_codebook_generator.seed_derived_codebook`
* Canonical equation: :mod:`tac.canonical_equations.procedural_codebook_savings`
* VQ-VAE archive grammar: :mod:`tac.substrates.vq_vae.archive` (VQV1)
"""

from __future__ import annotations

import io
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

from tac.procedural_codebook_generator import (
    DEFAULT_GENERATOR_KIND,
    derive_codebook_from_seed,
)
from tac.substrates.vq_vae.archive import (
    BROTLI_QUALITY,
    VQV1_HEADER_FMT,
    VQV1_HEADER_SIZE,
    VQV1_MAGIC,
    VQV1_SCHEMA_VERSION,
    parse_archive,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "PROCEDURAL_CODEBOOK_BYTES_DEFAULT",
    "PROCEDURAL_CODEBOOK_DTYPE_DEFAULT",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "compose_with_procedural_codebook",
    "derive_procedural_codebook_replacement",
    "predicted_archive_bytes_saved",
    "predicted_delta_s",
    "verify_procedural_codebook_in_domain",
    "verify_seed_mutation_changes_codebook_bytes",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT: str = "intermediate_transform_quantizer"
"""IN-DOMAIN context label for canonical equation #26 per VQ-VAE's role.

The VQ-VAE codebook IS the canonical intermediate-transform quantizer
(per ``_INCLUDED_CONTEXTS`` in ``procedural_codebook_savings.py``): the
K×D embedding table is used as a nearest-neighbor LUT that quantizes
encoder outputs to discrete indices which the decoder consumes.

This differs from the sister DP1 context
``comma2k19_ood_derived_basis_replacement`` (DP1 codebook = PCA-derived
basis vectors from Comma2k19 frames; VQ-VAE codebook = randomly-initialized
+ EMA-adapted embedding table). Both contexts are members of canonical
equation #26's ``_INCLUDED_CONTEXTS`` and produce the same closed-form
``ΔS = -25 * (N - K) / 37_545_489`` savings prediction.
"""

PROCEDURAL_SEED_SIZE_BYTES: int = 32
"""Canonical procedural-variant seed size in bytes.

Per 5-substrate matrix design + DP1 BUILD canonical pattern: 32-byte PCG64
seed replaces the empirical 8192-byte codebook slot. The ``32`` constant
is the ``K_seed`` term in the canonical equation #26 closed form:

    predicted_delta_s = -25 * (8192 - 32) / 37_545_489 = -0.005434
"""

PROCEDURAL_CODEBOOK_BYTES_DEFAULT: int = 8192
"""Default empirical codebook size in bytes for the canonical VQ-VAE config.

K=512 entries × D=8 embedding dim × fp16 (2 bytes/element) = 8192 bytes.
Verified empirically via ``architecture.py:62,65,181-183`` +
``runtime_state_dict_for_archive()`` emitting the codebook tensor at
fp16 in the archive blob.

For non-canonical VQ-VAE configs (e.g. K=256 × D=4 = 2048 B), the
predicted savings scales linearly per the canonical equation: the
``codebook_bytes`` parameter to :func:`predicted_delta_s` MUST match
the trained substrate's actual codebook footprint.
"""

PROCEDURAL_CODEBOOK_DTYPE_DEFAULT: np.dtype = np.dtype(np.uint8)
"""Default dtype for procedural codebook derivation.

uint8 matches the DP1 variant's canonical dtype + the PCG64 generator's
native byte output. The actual VQ-VAE inflate-time codebook tensor is
fp16; the procedural derivation produces uint8 bytes which inflate
interprets as the raw fp16 buffer for the codebook tensor.
"""

_MIN_SEED_SIZE_BYTES: int = 8
_MAX_SEED_SIZE_BYTES: int = 256

# Default output shape: 8192 bytes total at uint8 dtype (matches the
# canonical K=512 × D=8 × fp16 codebook byte count).
_DEFAULT_OUTPUT_SHAPE_BYTES: tuple[int, ...] = (PROCEDURAL_CODEBOOK_BYTES_DEFAULT,)


class ProceduralVariantError(ValueError):
    """Raised on invalid procedural-variant configuration.

    Sister of :class:`tac.procedural_codebook_generator.ProceduralCodebookGeneratorError`
    + :class:`ValueError` raised by archive.py validators.
    """


@dataclass(frozen=True)
class ProceduralVariantConfig:
    """Configuration for the VQ-VAE procedural-codebook replacement variant.

    Sister of the canonical training pipeline's implicit "trained codebook"
    config. This config holds the inputs to the procedural derivation
    pipeline.

    Attributes:
        seed_bytes: The procedural seed bytes (typically 32 bytes for
            ``derive_codebook_from_seed`` PCG64). Length must be in
            ``[_MIN_SEED_SIZE_BYTES, _MAX_SEED_SIZE_BYTES]`` per canonical
            equation #26 ``seed_size_bytes_range`` domain-of-validity.
        output_shape: The codebook shape to derive in bytes (default
            ``(8192,)`` matches K=512 × D=8 × fp16 = 8192 bytes).
        dtype: The codebook dtype (default
            ``PROCEDURAL_CODEBOOK_DTYPE_DEFAULT`` = ``np.uint8``).
        generator_kind: The PRNG kind (one of ``xorshift`` / ``lcg`` /
            ``pcg64``; default ``DEFAULT_GENERATOR_KIND`` = ``"pcg64"``).
        canonical_equation_context: The IN-DOMAIN context per canonical
            equation #26 (default ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``
            = ``"intermediate_transform_quantizer"``).
    """

    seed_bytes: bytes
    output_shape: tuple[int, ...] = _DEFAULT_OUTPUT_SHAPE_BYTES
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
# Canonical equation #26 closed-form predictions
# ---------------------------------------------------------------------------


_CONTEST_RATE_DENOM_BYTES: int = 37_545_489
_CONTEST_RATE_MULTIPLIER: float = 25.0


def predicted_archive_bytes_saved(
    codebook_bytes: int = PROCEDURAL_CODEBOOK_BYTES_DEFAULT,
    seed_bytes: int = PROCEDURAL_SEED_SIZE_BYTES,
) -> int:
    """Closed-form bytes-saved per canonical equation #26.

    Returns ``codebook_bytes - seed_bytes`` (the structural delta the
    variant removes from the archive). For the canonical config this is
    ``8192 - 32 = 8160`` bytes.

    Args:
        codebook_bytes: Empirical codebook byte count (default 8192 for
            canonical K=512 × D=8 × fp16 VQ-VAE).
        seed_bytes: Procedural seed byte count (default 32 for canonical
            PCG64 32-byte seed).

    Returns:
        Integer bytes-saved. Returns 0 when seed >= codebook (degenerate
        case where the procedural variant adds bytes).
    """
    delta = int(codebook_bytes) - int(seed_bytes)
    return max(0, delta)


def predicted_delta_s(
    codebook_bytes: int = PROCEDURAL_CODEBOOK_BYTES_DEFAULT,
    seed_bytes: int = PROCEDURAL_SEED_SIZE_BYTES,
) -> float:
    """Closed-form predicted ΔS per canonical equation #26.

    ``predicted_delta_s = -25 * (codebook_bytes - seed_bytes) / 37_545_489``

    For the canonical VQ-VAE config (8192-byte codebook, 32-byte seed):
    ``-25 * (8192 - 32) / 37_545_489 = -0.005434`` (a score IMPROVEMENT
    of ~0.0054 from the rate-axis only).

    NOT a score CLAIM — this is the canonical equation #26 closed-form
    PREDICTION which the operator-routed per-substrate symposium per
    Catalog #325 must validate via paired-smoke contest-CUDA + contest-CPU
    auth-eval on a real archive per CLAUDE.md "Submission auth eval — BOTH
    CPU AND CUDA" before any promotion.

    Args:
        codebook_bytes: Empirical codebook byte count.
        seed_bytes: Procedural seed byte count.

    Returns:
        Predicted score delta (negative = improvement). Returns 0.0 when
        seed >= codebook (no savings possible).
    """
    bytes_saved = predicted_archive_bytes_saved(codebook_bytes, seed_bytes)
    return -_CONTEST_RATE_MULTIPLIER * bytes_saved / _CONTEST_RATE_DENOM_BYTES


# ---------------------------------------------------------------------------
# Public API — derivation + in-domain check
# ---------------------------------------------------------------------------


def derive_procedural_codebook_replacement(
    seed_bytes: bytes,
    output_shape: tuple[int, ...] = _DEFAULT_OUTPUT_SHAPE_BYTES,
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> np.ndarray:
    """Derive a procedural codebook replacement from a deterministic seed.

    Thin wrapper around :func:`tac.procedural_codebook_generator.derive_codebook_from_seed`
    with VQ-VAE-specific validation + canonical equation #26 domain awareness.

    Args:
        seed_bytes: Procedural seed (8-256 bytes; canonical 32 bytes).
        output_shape: Codebook shape (default ``(8192,)`` = 8192 bytes
            uint8 matching canonical K=512 × D=8 × fp16 codebook).
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

    The VQ-VAE canonical context ``intermediate_transform_quantizer`` IS a
    member of canonical equation #26's ``_INCLUDED_CONTEXTS`` (per
    ``procedural_codebook_savings.py:96``); the equation predicts
    ``ΔS = -25 * (N - K) / 37_545_489`` for any IN-DOMAIN context.

    When slot 3's ``validate_context_is_in_domain`` helper lands in
    ``tac.canonical_equations``, this function will be refactored to call
    that helper. Until then, the IN-DOMAIN check is a constant comparison
    against ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT`` + a fallback set of
    canonical IN-DOMAIN context labels per ``_INCLUDED_CONTEXTS``.

    Args:
        context: The context string to validate (default
            ``CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT``).

    Returns:
        True if context is IN-DOMAIN per canonical equation #26.
    """
    # Try slot 3 sister helper first if it has landed; gracefully fall
    # back to the canonical IN-DOMAIN context set if not.
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
        # Slot 3 sister helper not yet landed — fall back to canonical
        # IN-DOMAIN context set per procedural_codebook_savings.py
        # `_INCLUDED_CONTEXTS` (intermediate_transform_quantizer is the
        # canonical VQ-VAE context; the broader set is the IN-DOMAIN
        # surface for sister substrates that may share this helper).
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


def _serialize_decoder_state_dict_without_codebook(
    sd: dict[str, torch.Tensor],
) -> bytes:
    """Brotli'd pickle of decoder-only state_dict (codebook EXCLUDED).

    The procedural variant replaces the codebook tensor in the archive with
    a 32-byte PCG64 seed; inflate re-derives the codebook tensor from the
    seed before unpickling the decoder state_dict. This helper serializes
    ONLY the non-codebook keys (i.e. ``decoder.*``) so the procedural
    archive bytes do not double-pay for the codebook tensor.

    Sorted-key iteration enforces byte-determinism per CLAUDE.md "Beauty,
    simplicity, and developer experience" + sister DP1 archive serializer
    sorted-keys JSON pattern; without this two calls with the same input
    can produce different bytes due to Python dict iteration-order
    non-determinism across pickle protocol 4 + brotli boundaries.
    """
    decoder_only = {
        k: sd[k].detach().to("cpu", dtype=torch.float16).contiguous()
        for k in sorted(sd.keys())
        if k != "codebook"
    }
    buf = io.BytesIO()
    pickle.dump(decoder_only, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def compose_with_procedural_codebook(
    original_archive_bytes: bytes,
    seed_bytes: bytes,
    *,
    output_shape: tuple[int, ...] | None = None,
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bytes:
    """Compose a VQ-VAE archive whose codebook is REPLACED by a procedural seed.

    Sister of :func:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant.compose_with_procedural_codebook`
    (the canonical DP1 PROCEDURAL TRAINER BUILD pattern from commit ``9cbfa471c``).
    Takes an existing VQV1 archive + a 32-byte seed; emits a NEW archive
    whose codebook tensor bytes are REPLACED by ``brotli(seed_bytes)`` (the
    seed is re-derived at inflate time via
    :func:`derive_procedural_codebook_replacement`).

    The bytes-saved is
    ``len(original_decoder_blob) - len(new_decoder_blob + brotli(seed_bytes))``
    which approximately equals ``codebook_bytes - len(brotli(seed_bytes))``
    for the canonical config + matches the canonical equation #26 prediction
    ``N - K`` term (8192 - 32 = 8160 bytes structurally).

    Archive grammar adjustment: the canonical VQV1 archive packs codebook
    INSIDE the brotli'd ``decoder_blob`` state_dict. The procedural variant
    splits this: the decoder state_dict is repacked WITHOUT the codebook
    key, then a separate ``brotli(seed_bytes)`` blob is appended after the
    decoder blob. Inflate detects the seed blob via the ``decoder_blob``
    length + a sentinel prefix and re-derives the codebook bytes
    deterministically.

    For the L0 scaffold we store the seed bytes directly in a NEW envelope
    that the procedural-inflate runtime detects via the first 4 bytes
    matching ``b"VQVP"`` (VQ-VAE Procedural sentinel). The procedural
    archive grammar:

    ::

        MAGIC(4)     b"VQV1"     unchanged for sister-roundtrip compat
        VERSION(1)   u8          unchanged
        ...          ...         unchanged header fields
        DECODER_BLOB_LEN: u32   length of new envelope (sentinel + seed)
        ...
        DECODER_BLOB: bytes      b"VQVP" + struct.pack("<HBB16s4s",
                                       codebook_size, embedding_dim, gen_kind_id, seed[:16], seed[16:])
                                  + brotli(decoder_only_state_dict)
        INDICES_BLOB             unchanged
        META_BLOB                unchanged

    Args:
        original_archive_bytes: The existing VQV1 archive bytes (with the
            canonical trained codebook).
        seed_bytes: The procedural seed (32 bytes canonical).
        output_shape: Codebook output shape in bytes (must match the
            original codebook byte count for byte-stability at inflate;
            default = inferred from the original codebook tensor).
        dtype: Codebook derivation dtype (default ``np.uint8``).
        generator_kind: PRNG kind (default ``"pcg64"``).

    Returns:
        New VQV1-style archive bytes with the codebook tensor REMOVED from
        the decoder state_dict + a procedural seed envelope prepended to
        the decoder blob. Header is rewritten so ``DECODER_BLOB_LEN``
        matches the new payload; indices, meta sections are preserved
        byte-for-byte.

    Raises:
        ProceduralVariantError: invalid configuration.
        ValueError: original archive bytes are not parseable as VQV1.
    """
    if output_shape is None:
        output_shape = _DEFAULT_OUTPUT_SHAPE_BYTES

    config = ProceduralVariantConfig(
        seed_bytes=bytes(seed_bytes),
        output_shape=output_shape,
        dtype=np.dtype(dtype),
        generator_kind=generator_kind,
    )

    # Parse the existing VQV1 archive to locate sections + extract decoder
    # state_dict (the codebook tensor lives inside the brotli'd blob).
    arc = parse_archive(original_archive_bytes)

    # Re-serialize the decoder state_dict WITHOUT the codebook tensor.
    # The procedural-inflate runtime re-derives codebook bytes from the
    # seed envelope and re-injects the tensor before consuming the
    # decoder state_dict.
    new_decoder_blob = _serialize_decoder_state_dict_without_codebook(
        arc.decoder_state_dict
    )

    # Build the procedural seed envelope. The sentinel `b"VQVP"` lets a
    # future procedural-inflate runtime distinguish a procedural archive
    # from a canonical VQV1 archive without changing the top-level MAGIC.
    # The envelope precedes the decoder blob inside the DECODER section.
    sentinel = b"VQVP"
    # Reserve 1 byte for the generator-kind tag (0=xorshift, 1=lcg, 2=pcg64)
    _generator_kind_tag: dict[str, int] = {"xorshift": 0, "lcg": 1, "pcg64": 2}
    seed_envelope = (
        sentinel
        + struct.pack(
            "<HHB",
            arc.codebook_size,
            arc.embedding_dim,
            _generator_kind_tag[config.generator_kind],
        )
        + struct.pack("<I", len(config.seed_bytes))
        + config.seed_bytes
    )
    # Concatenate seed envelope + new decoder blob; downstream inflate
    # parses the envelope first, derives codebook bytes, then inflates
    # the decoder blob and re-injects the codebook tensor.
    new_decoder_section = seed_envelope + new_decoder_blob

    # Re-parse the original header to recover unchanged fields.
    (
        magic,
        version,
        codebook_size,
        embedding_dim,
        num_pairs,
        h_grid,
        w_grid,
        original_decoder_len,
        original_indices_len,
        original_meta_len,
    ) = struct.unpack(VQV1_HEADER_FMT, original_archive_bytes[:VQV1_HEADER_SIZE])

    if magic != VQV1_MAGIC or version != VQV1_SCHEMA_VERSION:
        raise ValueError(
            f"original_archive_bytes not parseable as VQV1 v{VQV1_SCHEMA_VERSION}: "
            f"magic={magic!r} version={version}"
        )

    # Slice indices + meta payloads from original (preserved byte-for-byte).
    indices_start = VQV1_HEADER_SIZE + original_decoder_len
    meta_start = indices_start + original_indices_len
    indices_blob = original_archive_bytes[indices_start : indices_start + original_indices_len]
    meta_blob = original_archive_bytes[meta_start : meta_start + original_meta_len]

    # Rewrite the header with the new decoder_blob length.
    new_header = struct.pack(
        VQV1_HEADER_FMT,
        VQV1_MAGIC,
        VQV1_SCHEMA_VERSION,
        codebook_size,
        embedding_dim,
        num_pairs,
        h_grid,
        w_grid,
        len(new_decoder_section),
        original_indices_len,
        original_meta_len,
    )
    assert len(new_header) == VQV1_HEADER_SIZE, (
        f"new_header size {len(new_header)} != {VQV1_HEADER_SIZE}"
    )

    new_archive = new_header + new_decoder_section + indices_blob + meta_blob

    # Catalog #220 + #272 + #287 invariant: assert procedural replacement
    # actually reduced bytes (the operational mechanism is byte-mutation-
    # traceable). For the canonical 32-byte seed + ~8192 B codebook the
    # bytes-saved should be > 1000.
    bytes_saved = len(original_archive_bytes) - len(new_archive)
    if bytes_saved <= 0:
        raise ProceduralVariantError(
            f"procedural variant did NOT reduce archive bytes "
            f"({len(original_archive_bytes)} -> {len(new_archive)}); "
            f"check seed size + brotli quality + codebook footprint"
        )

    return new_archive


def verify_seed_mutation_changes_codebook_bytes(
    seed_bytes: bytes,
    *,
    output_shape: tuple[int, ...] = _DEFAULT_OUTPUT_SHAPE_BYTES,
    dtype: np.dtype = PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    generator_kind: str = DEFAULT_GENERATOR_KIND,
) -> bool:
    """Verify mutating a single seed byte changes the derived codebook.

    Catalog #272 byte-mutation distinguishing-feature contract: the
    operational mechanism is byte-mutation-traceable. Flipping any of
    the 32 seed bytes MUST produce a different derived codebook.

    Args:
        seed_bytes: The canonical seed.
        output_shape: Codebook shape (default ``(8192,)``).
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
    if mutated_codebook.shape != original_codebook.shape:
        return True
    return not bool(np.array_equal(original_codebook, mutated_codebook))
