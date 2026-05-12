"""Composition cell registry: (substrate × primitive × order) tuples.

Per operator directive 2026-05-12 ("stacking and composition on everything"),
this module extends :mod:`tac.optimization.substrate_composition_matrix`
(the pairwise SUBSTRATE × SUBSTRATE matrix) with a NEW second dimension —
**packet-compiler PRIMITIVES** — and enumerates the cross-product as typed
:class:`CompositionCell` rows the cathedral autopilot can rank.

The substrate × substrate matrix already lives at
:mod:`tac.optimization.substrate_composition_matrix`; we re-use its
``SubstrateRow`` / ``SubstrateClass`` / ``ScoreAxis`` taxonomy and its
canonical 18-row inventory. THIS module adds:

- ``PrimitiveRow``: one packet-compiler primitive (PR101 GOLD ×3,
  sign-encoding ×5, schema-elision ×3, magic_codec_dense_streams, brotli,
  lzma) with its archive-grammar metadata.
- ``PRIMITIVE_INVENTORY``: the frozen list of primitive rows.
- ``CompositionCell``: one (substrate × ordered-primitive-list) tuple with
  predicted bytes_delta and score_delta.
- ``primitive_compatibility``: per (substrate_class × primitive_category)
  compatibility matrix (which primitives apply to which substrates).
- Per-cell ordering / mutual-exclusion rules.

**Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md).** Every
:class:`CompositionCell` carries ``score_claim=False``,
``ready_for_exact_eval_dispatch=False``, ``promotion_eligible=False``
until an empirical anchor is posted via :mod:`tac.continual_learning`.
The predicted deltas are derived from primitive metadata + the substrate
matrix's per-substrate predicted band; no number here is an authoritative
measurement.

**Wire-in surface.** :func:`enumerate_cells` (in
:mod:`tac.composition.enumerate_cells`) consumes this module to emit the
autopilot's ranking input. The autopilot reads
``cell.autopilot_candidate_kwargs()`` to construct its
:class:`RankedDispatchCandidate` rows.

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — pairwise SUBSTRATE
  × SUBSTRATE matrix (re-used taxonomy).
- :mod:`tac.optimization.autopilot_dispatch_ranking` — consumer.
- :mod:`tac.packet_compiler.__init__` — primitive symbol surface.
- :mod:`tac.packet_compiler.magic_codec_dense_streams` — primitive bundle
  used in the magic-codec-dense-streams primitive.
- :mod:`tac.packet_compiler.sign_encoding` — sign-encoding 5-strategy
  unified taxonomy.

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
- ``substrate_primitive_composition_cell_registry_v1``
- ``halt_and_ask_default_for_dispatch_recommendations``
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tac.optimization.substrate_composition_matrix import (
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    canonical_substrate_inventory,
)

# Schema constants pinned to v1 so downstream consumers detect schema drift.
SCHEMA_VERSION = "tac_composition_cell_registry_v1"

# Score-claim invariants — hard-coded False per CLAUDE.md "Forbidden score
# claims". A cell promotes ONLY after an empirical anchor lands via
# :mod:`tac.continual_learning`.
PLANNING_ONLY = True
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False


# ── Primitive taxonomy ────────────────────────────────────────────────────


class PrimitiveCategory(str, Enum):
    """Top-level primitive taxonomy class.

    The category governs:
    - Which substrates the primitive applies to (compatibility matrix).
    - Whether the primitive is order-sensitive (e.g., PR101 GOLD storage
      ordering MUST run BEFORE byte-map sign encoding).
    - Whether multiple primitives in the same category stack or are
      mutually exclusive (e.g., choose ONE sign-encoding strategy per
      tensor; brotli and lzma compete on the same byte stream).
    """

    # PR101 GOLD trio (storage order / conv4 perms / byte maps).
    # These are HNeRV-family-specific and order-sensitive.
    PR101_GOLD_STORAGE = "pr101_gold_storage"

    # Sign-encoding 5-strategy unified taxonomy (negzig/zig/twos/off/raw_uint8).
    # One strategy per tensor; mutually exclusive within a tensor.
    SIGN_ENCODING = "sign_encoding"

    # Schema-elision primitives (PR98 CD1, PR100 schema-driven, PR105
    # packed state schema). Mutually exclusive at the decoder-storage slot.
    SCHEMA_ELISION = "schema_elision"

    # Magic codec dense-streams bundle — auto-selecting codec for the
    # 6 dense-stream primitives (sparse RLE/AC/CDU/DV/categorical/lowpass).
    MAGIC_CODEC_DENSE_STREAMS = "magic_codec_dense_streams"

    # Generic byte-stream compressors. brotli is the contest baseline;
    # lzma is the universal-density entropy coder. They compete on a
    # given byte stream (choose one); they ORTHOGONALLY apply to
    # different streams.
    BROTLI = "brotli"
    LZMA = "lzma"


class PrimitiveOrderSensitivity(str, Enum):
    """How the primitive composes with siblings in its category.

    - ``ordered_pipeline``: order matters; primitives in this category
      MUST run in a specific sequence (e.g., PR101 GOLD: storage_order
      → conv4 perms → byte_maps).
    - ``mutually_exclusive``: choose at most one per target stream.
    - ``stackable``: multiple primitives can apply to different streams
      orthogonally (no ordering constraint between streams).
    """

    ORDERED_PIPELINE = "ordered_pipeline"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    STACKABLE = "stackable"


@dataclass(frozen=True)
class PrimitiveRow:
    """One packet-compiler primitive inventory row.

    Every primitive that ships in the composition registry appears
    exactly once. Fields are typed and frozen so downstream consumers
    cannot tamper with the declared compatibility surface.
    """

    primitive_id: str
    name: str
    category: PrimitiveCategory
    order_sensitivity: PrimitiveOrderSensitivity
    # Where in a pipeline does this primitive sit? Lower = earlier.
    # PR101 GOLD storage_order is order_index=0 (first), conv4_perms=1,
    # byte_maps=2 (last). Sign-encoding is order_index=3 (after byte maps
    # but before generic entropy coding). brotli/lzma are order_index=4
    # (last byte-stream-level entropy coding).
    order_index: int
    # Score-axis this primitive primarily affects via byte savings.
    # Per CLAUDE.md "SegNet vs PoseNet importance — operating-point
    # dependent", at PR106 r2 the marginal-value ranking is POSE > SEG > RATE.
    target_axis: ScoreAxis
    # Predicted byte savings band for the primitive applied alone to a
    # typical PR106 r2 substrate byte budget (low, high). NEGATIVE numbers
    # mean savings (bytes_delta < 0 = archive shrinks).
    predicted_bytes_delta_band: tuple[int, int]
    # Predicted score-delta band per CLAUDE.md "Forbidden empirical-claim-
    # without-evidence-tag" — NUMBERS HERE ARE PREDICTIONS, not measurements.
    predicted_score_delta_band: tuple[float, float]
    # Module path / canonical symbol the primitive lives at.
    canonical_module: str
    canonical_symbol: str
    # Which substrate classes the primitive applies to. Empty tuple
    # means "any substrate class" (e.g., brotli is universal).
    applicable_substrate_classes: tuple[SubstrateClass, ...]
    # Within-category ordering for ORDERED_PIPELINE primitives. Empty
    # tuple for non-ordered.
    within_category_order: tuple[str, ...] = ()
    # Documentation / rationale tag.
    notes: str = ""


# ── Canonical primitive inventory (FROZEN; bump SCHEMA_VERSION on change) ──


def canonical_primitive_inventory() -> list[PrimitiveRow]:
    """Return the 13 packet-compiler primitives composition registry covers.

    Inventory (13 primitives, alphabetical-by-category-then-id for stable
    matrix construction):

    PR101 GOLD ×3 (HNeRV-family substrates only, ordered pipeline):
      1. ``pr101_decoder_storage_order`` (order_index=0)
      2. ``pr101_conv4_storage_perms`` (order_index=1)
      3. ``pr101_decoder_byte_maps`` (order_index=2)

    Sign-encoding ×5 (any substrate with int-tensor weights, mutually
    exclusive within a tensor):
      4. ``sign_encoding_negzig``
      5. ``sign_encoding_zig``
      6. ``sign_encoding_twos``
      7. ``sign_encoding_off``
      8. ``sign_encoding_raw_uint8``

    Schema-elision ×3 (HNeRV-family decoder grammar, mutually exclusive):
      9. ``pr98_cd1_compact_format``
      10. ``pr100_schema_driven_decoder``
      11. ``pr105_packed_state_schema``

    Magic-codec dense-streams ×1 (universal byte-stream wrapper):
      12. ``magic_codec_dense_streams``

    Generic byte-stream compressors ×2:
      13. ``brotli`` (contest baseline)
      14. ``lzma`` (universal-density alternative)

    Total: 14 primitives. (PR101 GOLD ×3 + sign-encoding ×5 + schema-
    elision ×3 + magic-codec-dense ×1 + brotli + lzma = 14.)

    Cross-references for each row are in the ``notes`` field.
    """
    return [
        # ── PR101 GOLD trio (ordered pipeline, HNeRV-family only) ──
        PrimitiveRow(
            primitive_id="pr101_decoder_storage_order",
            name="PR101 GOLD: decoder storage order schema",
            category=PrimitiveCategory.PR101_GOLD_STORAGE,
            order_sensitivity=PrimitiveOrderSensitivity.ORDERED_PIPELINE,
            order_index=0,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-800, -100),
            predicted_score_delta_band=(-0.0005, -0.0001),
            canonical_module="tac.packet_compiler.pr101_decoder_storage_order",
            canonical_symbol="DecoderStorageOrderSchema",
            applicable_substrate_classes=(
                SubstrateClass.RENDERER_REPLACEMENT,
            ),
            within_category_order=(
                "pr101_decoder_storage_order",
                "pr101_conv4_storage_perms",
                "pr101_decoder_byte_maps",
            ),
            notes=(
                "PR101 GOLD step 1: tensor storage-order reordering for "
                "downstream stream-partition codecs; HNeRV-family substrates only"
            ),
        ),
        PrimitiveRow(
            primitive_id="pr101_conv4_storage_perms",
            name="PR101 GOLD: conv4 storage permutations",
            category=PrimitiveCategory.PR101_GOLD_STORAGE,
            order_sensitivity=PrimitiveOrderSensitivity.ORDERED_PIPELINE,
            order_index=1,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-600, -100),
            predicted_score_delta_band=(-0.0005, -0.0001),
            canonical_module="tac.packet_compiler.pr101_conv4_storage_perms",
            canonical_symbol="Conv4StoragePermSchema",
            applicable_substrate_classes=(
                SubstrateClass.RENDERER_REPLACEMENT,
            ),
            within_category_order=(
                "pr101_decoder_storage_order",
                "pr101_conv4_storage_perms",
                "pr101_decoder_byte_maps",
            ),
            notes=(
                "PR101 GOLD step 2: per-conv4 axis storage permutations "
                "that hit lower entropy after step 1's storage order"
            ),
        ),
        PrimitiveRow(
            primitive_id="pr101_decoder_byte_maps",
            name="PR101 GOLD: decoder byte maps (sign-encoding strategy)",
            category=PrimitiveCategory.PR101_GOLD_STORAGE,
            order_sensitivity=PrimitiveOrderSensitivity.ORDERED_PIPELINE,
            order_index=2,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-1200, -200),
            predicted_score_delta_band=(-0.0010, -0.0002),
            canonical_module="tac.packet_compiler.pr101_decoder_byte_maps",
            canonical_symbol="DecoderByteMapsSchema",
            applicable_substrate_classes=(
                SubstrateClass.RENDERER_REPLACEMENT,
            ),
            within_category_order=(
                "pr101_decoder_storage_order",
                "pr101_conv4_storage_perms",
                "pr101_decoder_byte_maps",
            ),
            notes=(
                "PR101 GOLD step 3: per-tensor sign-encoding byte map "
                "(consumes sign_encoding family); HNeRV-family only"
            ),
        ),
        # ── Sign-encoding 5 strategies (mutually exclusive per tensor) ──
        PrimitiveRow(
            primitive_id="sign_encoding_negzig",
            name="Sign-encoding: negzig (negate then zig-zag)",
            category=PrimitiveCategory.SIGN_ENCODING,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=3,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-400, -50),
            predicted_score_delta_band=(-0.0004, -0.00005),
            canonical_module="tac.packet_compiler.sign_encoding",
            canonical_symbol="SignEncodingStrategy.NEGZIG",
            applicable_substrate_classes=(),  # universal (any int-tensor).
            notes=(
                "Best for distributions with both signs but skewed negative; "
                "negate-then-zig-zag often produces lower-entropy nibble stream"
            ),
        ),
        PrimitiveRow(
            primitive_id="sign_encoding_zig",
            name="Sign-encoding: zig (canonical zig-zag)",
            category=PrimitiveCategory.SIGN_ENCODING,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=3,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-400, -50),
            predicted_score_delta_band=(-0.0004, -0.00005),
            canonical_module="tac.packet_compiler.sign_encoding",
            canonical_symbol="SignEncodingStrategy.ZIG",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "Canonical zig-zag: -1→1, 1→2, -2→3, 2→4, … bias to small "
                "magnitudes; standard for protobuf/varint pipelines"
            ),
        ),
        PrimitiveRow(
            primitive_id="sign_encoding_twos",
            name="Sign-encoding: twos (two's-complement byte view)",
            category=PrimitiveCategory.SIGN_ENCODING,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=3,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-200, 100),  # may regress on skewed dist.
            predicted_score_delta_band=(-0.0002, 0.0001),
            canonical_module="tac.packet_compiler.sign_encoding",
            canonical_symbol="SignEncodingStrategy.TWOS",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "Two's-complement view of int8 as uint8 — zero-cost transform; "
                "best when entropy coder already handles sign bit via context"
            ),
        ),
        PrimitiveRow(
            primitive_id="sign_encoding_off",
            name="Sign-encoding: off (identity passthrough)",
            category=PrimitiveCategory.SIGN_ENCODING,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=3,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(0, 0),
            predicted_score_delta_band=(0.0, 0.0),
            canonical_module="tac.packet_compiler.sign_encoding",
            canonical_symbol="SignEncodingStrategy.OFF",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "Identity passthrough — baseline for ablation; never improves "
                "over the best strategy but ensures determinism"
            ),
        ),
        PrimitiveRow(
            primitive_id="sign_encoding_raw_uint8",
            name="Sign-encoding: raw_uint8 (reinterpret cast)",
            category=PrimitiveCategory.SIGN_ENCODING,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=3,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-300, 0),
            predicted_score_delta_band=(-0.0003, 0.0),
            canonical_module="tac.packet_compiler.sign_encoding",
            canonical_symbol="SignEncodingStrategy.RAW_UINT8",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "Reinterpret signed int as raw uint8 byte stream; for "
                "already-uint8-quantized weights this is the canonical no-op view"
            ),
        ),
        # ── Schema-elision ×3 (HNeRV-family decoder grammar, MX) ──
        PrimitiveRow(
            primitive_id="pr98_cd1_compact_format",
            name="Schema-elision: PR98 CD1 compact format",
            category=PrimitiveCategory.SCHEMA_ELISION,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=2,  # Sits at decoder-storage-grammar slot.
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-600, -100),
            predicted_score_delta_band=(-0.0005, -0.0001),
            canonical_module="tac.packet_compiler.pr98_cd1_compact_format",
            canonical_symbol="CD1CompactFormat",
            applicable_substrate_classes=(
                SubstrateClass.RENDERER_REPLACEMENT,
            ),
            notes=(
                "PR98 CD1: schema-elision V1; removes per-tensor schema "
                "headers via fixed-format encoding"
            ),
        ),
        PrimitiveRow(
            primitive_id="pr100_schema_driven_decoder",
            name="Schema-elision: PR100 schema-driven decoder storage",
            category=PrimitiveCategory.SCHEMA_ELISION,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=2,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-800, -200),
            predicted_score_delta_band=(-0.0007, -0.0002),
            canonical_module="tac.packet_compiler.pr100_schema_driven_decoder",
            canonical_symbol="SchemaDrivenPayload",
            applicable_substrate_classes=(
                SubstrateClass.RENDERER_REPLACEMENT,
            ),
            notes=(
                "PR100 schema-driven decoder: V2 of schema-elision; emits "
                "per-tensor schema via single archive-level grammar token"
            ),
        ),
        PrimitiveRow(
            primitive_id="pr105_packed_state_schema",
            name="Schema-elision: PR105 packed-state-schema size-sorted",
            category=PrimitiveCategory.SCHEMA_ELISION,
            order_sensitivity=PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE,
            order_index=2,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-500, -50),
            predicted_score_delta_band=(-0.0004, -0.00005),
            canonical_module="tac.packet_compiler.pr105_packed_state_schema",
            canonical_symbol="pack_state_schema_size_sorted",
            applicable_substrate_classes=(
                SubstrateClass.RENDERER_REPLACEMENT,
            ),
            notes=(
                "PR105 packed state-schema with size-sorted entries; "
                "smallest schema-elision but most-universal applicability"
            ),
        ),
        # ── Magic-codec dense-streams ×1 (universal byte-stream auto-select) ──
        PrimitiveRow(
            primitive_id="magic_codec_dense_streams",
            name="Magic codec: dense-streams auto-selector bundle",
            category=PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS,
            order_sensitivity=PrimitiveOrderSensitivity.STACKABLE,
            order_index=4,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-1200, 0),
            predicted_score_delta_band=(-0.0010, 0.0),
            canonical_module="tac.packet_compiler.magic_codec_dense_streams",
            canonical_symbol="select_best_dense_stream_primitive",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "Auto-selects between 6 dense-stream primitives (sparse-RLE, "
                "arithmetic-coefficients, centered-delta-uint8, delta-varint-"
                "pose, categorical-stream, lowpass-luma-residual); "
                "score-claim=false until vendored runtime closure lands"
            ),
        ),
        # ── Generic byte-stream compressors ×2 ──
        PrimitiveRow(
            primitive_id="brotli",
            name="Generic compressor: brotli (contest baseline)",
            category=PrimitiveCategory.BROTLI,
            order_sensitivity=PrimitiveOrderSensitivity.STACKABLE,
            order_index=4,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-1500, 0),
            predicted_score_delta_band=(-0.0012, 0.0),
            canonical_module="brotli",
            canonical_symbol="brotli.compress",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "Contest-baseline byte-stream compressor; orthogonal to "
                "every substrate via byte-stream wrapping"
            ),
        ),
        PrimitiveRow(
            primitive_id="lzma",
            name="Generic compressor: lzma (universal-density entropy coder)",
            category=PrimitiveCategory.LZMA,
            order_sensitivity=PrimitiveOrderSensitivity.STACKABLE,
            order_index=4,
            target_axis=ScoreAxis.RATE,
            predicted_bytes_delta_band=(-1700, 0),
            predicted_score_delta_band=(-0.0014, 0.0),
            canonical_module="lzma",
            canonical_symbol="lzma.compress",
            applicable_substrate_classes=(),  # universal.
            notes=(
                "LZMA / xz universal-density entropy coder; tends to outperform "
                "brotli on small dense streams but at higher decode cost"
            ),
        ),
    ]


# ── Compatibility matrix (substrate_class × primitive_category) ──────────


# Each cell answers: "can this primitive_category compose with substrates
# of this class?" The matrix is consulted by enumerate_cells() to skip
# inapplicable cells (e.g., PR101 GOLD does not apply to RESIDUAL).
_COMPATIBILITY_MATRIX_V1: dict[
    tuple[SubstrateClass, PrimitiveCategory], bool
] = {
    # PR101 GOLD: HNeRV-family substrates (RENDERER_REPLACEMENT) only.
    # The PR101 source landed for hnerv_lc_family; downstream NeRV-family
    # substrates (BlockNeRV/FFNeRV/etc.) are HNeRV-derived enough to
    # reuse the primitive via the storage-order schema.
    (SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.PR101_GOLD_STORAGE): True,
    (SubstrateClass.RESIDUAL, PrimitiveCategory.PR101_GOLD_STORAGE): False,
    (SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.PR101_GOLD_STORAGE): False,
    (SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.PR101_GOLD_STORAGE): False,
    (SubstrateClass.BOLT_ON, PrimitiveCategory.PR101_GOLD_STORAGE): False,
    (SubstrateClass.META_CODEC, PrimitiveCategory.PR101_GOLD_STORAGE): False,
    # Sign-encoding: universal (any substrate with int-tensor weights).
    (SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.SIGN_ENCODING): True,
    (SubstrateClass.RESIDUAL, PrimitiveCategory.SIGN_ENCODING): True,
    (SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.SIGN_ENCODING): True,
    (SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.SIGN_ENCODING): True,
    (SubstrateClass.BOLT_ON, PrimitiveCategory.SIGN_ENCODING): True,
    (SubstrateClass.META_CODEC, PrimitiveCategory.SIGN_ENCODING): False,
    # Schema-elision: HNeRV-family decoder grammar only (RENDERER_REPLACEMENT).
    (SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.SCHEMA_ELISION): True,
    (SubstrateClass.RESIDUAL, PrimitiveCategory.SCHEMA_ELISION): False,
    (SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.SCHEMA_ELISION): False,
    (SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.SCHEMA_ELISION): False,
    (SubstrateClass.BOLT_ON, PrimitiveCategory.SCHEMA_ELISION): False,
    (SubstrateClass.META_CODEC, PrimitiveCategory.SCHEMA_ELISION): False,
    # Magic-codec dense streams: universal byte-stream wrapper.
    # Note: META_CODEC + MAGIC_CODEC_DENSE_STREAMS is self-referential
    # (magic_codec applied to magic_codec); skip.
    (SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS): True,
    (SubstrateClass.RESIDUAL, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS): True,
    (SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS): True,
    (SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS): True,
    (SubstrateClass.BOLT_ON, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS): True,
    (SubstrateClass.META_CODEC, PrimitiveCategory.MAGIC_CODEC_DENSE_STREAMS): False,
    # Brotli: universal byte-stream compressor.
    (SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.BROTLI): True,
    (SubstrateClass.RESIDUAL, PrimitiveCategory.BROTLI): True,
    (SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.BROTLI): True,
    (SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.BROTLI): True,
    (SubstrateClass.BOLT_ON, PrimitiveCategory.BROTLI): True,
    (SubstrateClass.META_CODEC, PrimitiveCategory.BROTLI): True,
    # LZMA: universal byte-stream compressor.
    (SubstrateClass.RENDERER_REPLACEMENT, PrimitiveCategory.LZMA): True,
    (SubstrateClass.RESIDUAL, PrimitiveCategory.LZMA): True,
    (SubstrateClass.POSE_AXIS_SIDECHANNEL, PrimitiveCategory.LZMA): True,
    (SubstrateClass.SELF_COMPRESSION, PrimitiveCategory.LZMA): True,
    (SubstrateClass.BOLT_ON, PrimitiveCategory.LZMA): True,
    (SubstrateClass.META_CODEC, PrimitiveCategory.LZMA): True,
}


def primitive_compatibility(
    substrate_class: SubstrateClass,
    primitive_category: PrimitiveCategory,
) -> bool:
    """Return True if the given primitive_category applies to substrate_class.

    Per CLAUDE.md "FORBIDDEN_PATTERNS / forbidden_score_claim_with_byte_change":
    a (substrate, primitive) pair that returns False here will NEVER be
    enumerated as a :class:`CompositionCell` and thus cannot be promoted
    to dispatch.
    """
    return _COMPATIBILITY_MATRIX_V1.get(
        (substrate_class, primitive_category), False
    )


# ── Composition cell ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompositionCell:
    """One (substrate × ordered primitives × order) composition cell.

    The cell is the autopilot's ranking unit. Each cell carries:

    - ``substrate_id``: a key into ``canonical_substrate_inventory()``.
    - ``primitives``: a tuple of ``(primitive_id, kwargs)`` pairs in
      the order they apply. Mutually-exclusive primitives appear at most
      once per category.
    - ``composition_order``: an explicit ordering token for the pipeline
      ("storage_order → conv4_perms → byte_maps" for PR101 GOLD; or
      "sign_encoding → brotli" for the universal byte-stream path).
    - ``predicted_bytes_delta``: aggregated from primitive metadata.
    - ``predicted_score_delta``: aggregated; ``[predicted; substrate × primitive matrix v1]``.
    - ``compatibility_verdict``: "compatible" / "violates_compatibility_matrix" /
      "violates_ordering" / "mutually_exclusive_collision".

    Per CLAUDE.md "Forbidden score claims" + "Forbidden empirical-claim-
    without-evidence-tag": ``score_claim``, ``promotion_eligible``, and
    ``ready_for_exact_eval_dispatch`` are hard-coded False. The cell
    promotes only after an empirical anchor is posted via
    :mod:`tac.continual_learning`.
    """

    cell_id: str
    substrate_id: str
    substrate_class: SubstrateClass
    primitives: tuple[tuple[str, dict[str, Any]], ...]
    composition_order: tuple[str, ...]
    predicted_bytes_delta: int
    predicted_score_delta: float
    compatibility_verdict: str
    blockers: tuple[str, ...] = ()
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    notes: str = ""

    def primitive_ids(self) -> tuple[str, ...]:
        """Return the primitive_ids in pipeline order."""
        return tuple(p for p, _ in self.primitives)

    def autopilot_candidate_kwargs(self) -> dict[str, Any]:
        """Return kwargs compatible with the autopilot's CandidateRow.

        Cross-ref :class:`tac.optimization.autopilot_dispatch_ranking.RankedDispatchCandidate`
        and :class:`tools.cathedral_autopilot_autonomous_loop.CandidateRow`.
        The mapping is intentionally narrow so cells can flow into the
        existing autopilot ranker without schema drift.
        """
        return {
            "candidate_id": self.cell_id,
            "family": self.substrate_class.value,
            "predicted_score_delta": self.predicted_score_delta,
            "expected_information_gain": abs(self.predicted_score_delta),
            "estimated_dispatch_cost_usd": 0.0,  # set by ranker.
            "blockers": list(self.blockers),
            "notes": self._format_notes(),
        }

    def _format_notes(self) -> str:
        lines = [
            "[predicted; substrate × primitive composition matrix v1]",
            f"substrate_id: {self.substrate_id}",
            f"primitives (in order): {list(self.primitive_ids())}",
            f"composition_order: {list(self.composition_order)}",
            f"predicted_bytes_delta: {self.predicted_bytes_delta}",
            f"compatibility_verdict: {self.compatibility_verdict}",
        ]
        if self.notes:
            lines.append(f"notes: {self.notes}")
        return "\n".join(lines)


# ── Order validation helpers ─────────────────────────────────────────────


def validate_pipeline_ordering(
    primitive_ids: list[str],
    primitives_by_id: dict[str, PrimitiveRow],
) -> tuple[bool, str]:
    """Validate a primitive pipeline against ordering / mutual-exclusion rules.

    Returns ``(ok, rationale)``. If ``ok`` is False, ``rationale`` carries
    the reason the pipeline violates the rules.

    Rules enforced:

    1. **Mutual-exclusion within category**: at most one primitive per
       MUTUALLY_EXCLUSIVE category (e.g., one sign-encoding, one schema-
       elision).
    2. **Ordered-pipeline within-category order**: primitives in an
       ORDERED_PIPELINE category MUST appear in the order declared by
       ``within_category_order``.
    3. **Cross-category order**: primitives with lower ``order_index``
       MUST appear before primitives with higher ``order_index``.
    4. **No duplicates**: a primitive_id appears at most once in the
       pipeline.
    """
    if not primitive_ids:
        return True, "empty pipeline trivially valid"

    if len(set(primitive_ids)) != len(primitive_ids):
        dupes = sorted({p for p in primitive_ids if primitive_ids.count(p) > 1})
        return False, f"duplicate primitive_ids: {dupes}"

    rows: list[PrimitiveRow] = []
    for pid in primitive_ids:
        if pid not in primitives_by_id:
            return False, f"unknown primitive_id: {pid!r}"
        rows.append(primitives_by_id[pid])

    # Rule 3: cross-category order_index monotonic.
    indices = [r.order_index for r in rows]
    if indices != sorted(indices):
        return False, (
            f"cross-category order_index not monotonic: {indices} "
            f"for {primitive_ids}; smaller order_index must appear first"
        )

    # Rule 1: mutual-exclusion within MUTUALLY_EXCLUSIVE categories.
    cat_counts: dict[PrimitiveCategory, list[str]] = {}
    for r in rows:
        cat_counts.setdefault(r.category, []).append(r.primitive_id)
    for cat, members in cat_counts.items():
        if len(members) <= 1:
            continue
        sensitivity = rows[
            [r.primitive_id for r in rows].index(members[0])
        ].order_sensitivity
        if sensitivity == PrimitiveOrderSensitivity.MUTUALLY_EXCLUSIVE:
            return False, (
                f"mutually-exclusive category {cat.value} has multiple "
                f"primitives: {members}; choose one"
            )

    # Rule 2: ordered-pipeline within-category order.
    for cat, members in cat_counts.items():
        sensitivity = rows[
            [r.primitive_id for r in rows].index(members[0])
        ].order_sensitivity
        if sensitivity != PrimitiveOrderSensitivity.ORDERED_PIPELINE:
            continue
        # The within-category order tuple is the same across all rows in
        # this category. Filter the pipeline's primitives in this category
        # and ensure they are a prefix of the declared within_category_order.
        expected_order = rows[
            [r.primitive_id for r in rows].index(members[0])
        ].within_category_order
        observed_order = tuple(members)
        # Observed must be a subsequence of expected; specifically,
        # the indices of each observed within expected must be increasing.
        expected_index = {pid: i for i, pid in enumerate(expected_order)}
        try:
            observed_indices = [expected_index[pid] for pid in observed_order]
        except KeyError as exc:
            return False, (
                f"category {cat.value} pipeline contains primitive not in "
                f"declared within_category_order: {exc.args[0]!r}"
            )
        if observed_indices != sorted(observed_indices):
            return False, (
                f"category {cat.value} pipeline order {observed_order} "
                f"violates declared within_category_order {expected_order}"
            )

    return True, "pipeline ordering valid"


# ── Serialization helpers ────────────────────────────────────────────────


def _primitive_to_dict(p: PrimitiveRow) -> dict[str, Any]:
    d = dataclasses.asdict(p)
    d["category"] = p.category.value
    d["order_sensitivity"] = p.order_sensitivity.value
    d["target_axis"] = p.target_axis.value
    d["applicable_substrate_classes"] = [
        cls.value for cls in p.applicable_substrate_classes
    ]
    d["predicted_bytes_delta_band"] = list(p.predicted_bytes_delta_band)
    d["predicted_score_delta_band"] = list(p.predicted_score_delta_band)
    d["within_category_order"] = list(p.within_category_order)
    return d


def _cell_to_dict(c: CompositionCell) -> dict[str, Any]:
    d = dataclasses.asdict(c)
    d["substrate_class"] = c.substrate_class.value
    d["primitives"] = [
        {"primitive_id": pid, "kwargs": kw} for pid, kw in c.primitives
    ]
    d["composition_order"] = list(c.composition_order)
    d["blockers"] = list(c.blockers)
    return d


def serialize_primitive_inventory() -> list[dict[str, Any]]:
    """JSON-safe serialization of the primitive inventory."""
    return [_primitive_to_dict(p) for p in canonical_primitive_inventory()]


__all__ = [
    "SCHEMA_VERSION",
    "PLANNING_ONLY",
    "SCORE_CLAIM",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "PrimitiveCategory",
    "PrimitiveOrderSensitivity",
    "PrimitiveRow",
    "CompositionCell",
    "canonical_primitive_inventory",
    "primitive_compatibility",
    "validate_pipeline_ordering",
    "serialize_primitive_inventory",
    "_primitive_to_dict",
    "_cell_to_dict",
    # Re-exports for convenience.
    "ScoreAxis",
    "SubstrateClass",
    "SubstrateRow",
    "canonical_substrate_inventory",
]
