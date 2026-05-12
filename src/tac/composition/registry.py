"""Composition cell registry: (substrate x primitive x order) tuples.

Per operator directive 2026-05-12 ("stacking and composition on everything"),
this module extends :mod:`tac.optimization.substrate_composition_matrix`
(the pairwise SUBSTRATE x SUBSTRATE matrix) with a NEW second dimension —
**packet-compiler PRIMITIVES** — and enumerates the cross-product as typed
:class:`CompositionCell` rows the cathedral autopilot can rank.

The substrate x substrate matrix already lives at
:mod:`tac.optimization.substrate_composition_matrix`; we re-use its
``SubstrateRow`` / ``SubstrateClass`` / ``ScoreAxis`` taxonomy and its
canonical 18-row inventory. THIS module adds:

- ``PrimitiveRow``: one packet-compiler primitive (PR101 GOLD x3,
  sign-encoding x5, schema-elision x3, magic_codec_dense_streams, brotli,
  lzma) with its archive-grammar metadata.
- ``PRIMITIVE_INVENTORY``: the frozen list of primitive rows.
- ``CompositionCell``: one (substrate x ordered-primitive-list) tuple with
  predicted bytes_delta and score_delta.
- ``primitive_compatibility``: per (substrate_class x primitive_category)
  compatibility matrix (which primitives apply to which substrates).
- Per-cell ordering / mutual-exclusion rules.

FIX-D enrichments (2026-05-12, per ZZZZZ medium polish audit):

- :class:`SemanticConstraint` on each :class:`PrimitiveRow` declares
  substrates whose own design already implements the primitive's
  benefit (``redundant_with_substrate_ids``) AND substrate properties
  the primitive structurally requires (``expects_substrate_property``).
  Cells populate :attr:`CompositionCell.semantic_compatibility_warning`
  when the (substrate x primitive) pair is formally compatible but
  semantically no-op.
- :class:`RefusedReason` taxonomy makes the 8,975 refused cells
  machine-checkable: ``ORDERING_VIOLATION`` / ``MUTUAL_EXCLUSION`` /
  ``SUBSTRATE_INCOMPATIBLE`` / ``SEMANTIC_INCOMPATIBLE`` /
  ``DEPENDENCY_MISSING``. Each refused cell carries a typed
  :attr:`CompositionCell.refused_reason` next to its free-form blocker.

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
  x SUBSTRATE matrix (re-used taxonomy).
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
from enum import StrEnum
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


class PrimitiveCategory(StrEnum):
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


class PrimitiveOrderSensitivity(StrEnum):
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


class RefusedReason(StrEnum):
    """Machine-checkable taxonomy for refused composition cells.

    Per ZZZZZ stress-test audit 2026-05-12, the 8,975 refused cells
    emitted by :func:`tac.composition.enumerate.enumerate_cells` carry
    only a free-form ``blockers`` rationale. A typed taxonomy makes the
    refusal machine-checkable so downstream consumers (autopilot bridge,
    Pareto solver, sensitivity-map updater) can branch on the refusal
    class without parsing prose.

    Categories
    ----------
    - ``ORDERING_VIOLATION``: primitives are mutually-ordered but the
      pipeline lists them out of declared order (e.g., PR101 GOLD trio
      in wrong sequence).
    - ``MUTUAL_EXCLUSION``: multiple primitives from a
      MUTUALLY_EXCLUSIVE category appear in the pipeline (e.g., two
      sign-encoding strategies, two schema-elision primitives).
    - ``SUBSTRATE_INCOMPATIBLE``: the primitive's category does NOT
      apply to the substrate's class per the compatibility matrix (e.g.,
      PR101 GOLD on a RESIDUAL substrate).
    - ``SEMANTIC_INCOMPATIBLE``: primitives operate on incompatible
      byte regions or the primitive's expected substrate property is
      not present (formally legal but cannot actually compose).
    - ``DEPENDENCY_MISSING``: the primitive requires another primitive
      that is NOT in the pipeline (e.g., PR101 GOLD step 2 without
      step 1's storage-order reordering).
    - ``DUPLICATE_PRIMITIVE``: the same primitive_id appears more than
      once in the pipeline.
    - ``UNKNOWN_PRIMITIVE``: the pipeline references a primitive_id
      that is not in the canonical inventory.
    """

    ORDERING_VIOLATION = "ordering_violation"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    SUBSTRATE_INCOMPATIBLE = "substrate_incompatible"
    SEMANTIC_INCOMPATIBLE = "semantic_incompatible"
    DEPENDENCY_MISSING = "dependency_missing"
    DUPLICATE_PRIMITIVE = "duplicate_primitive"
    UNKNOWN_PRIMITIVE = "unknown_primitive"


@dataclass(frozen=True)
class SemanticConstraint:
    """Per-primitive semantic-composition constraint declaration.

    Per ZZZZZ medium polish 2026-05-12, primitives declare:

    - ``redundant_with_substrate_ids``: substrates whose own design
      already implements this primitive's benefit (applying the
      primitive on top is a semantic no-op). Example:
      ``magic_codec_dense_streams`` primitive applied to the
      ``magic_codec`` substrate is redundant — the substrate IS the
      magic codec.
    - ``expects_substrate_property``: free-form tokens documenting
      what the primitive structurally requires of the substrate (e.g.,
      ``"int_tensor_weights"`` for sign-encoding, ``"ar_coded_latent"``
      for primitives that consume an arithmetic-coded latent stream).
      Used in the cell's
      :attr:`CompositionCell.semantic_compatibility_warning` string.
    - ``incompatible_with_substrate_ids``: hard structural
      incompatibility (the primitive cannot operate on this substrate
      even though the compatibility matrix permits the category).
      Surfaces a :class:`RefusedReason.SEMANTIC_INCOMPATIBLE` cell.
    - ``requires_primitive_ids``: other primitive_ids that MUST be in
      the same pipeline for this primitive to function. Empty tuple
      means the primitive is standalone. Surfaces
      :class:`RefusedReason.DEPENDENCY_MISSING` when violated.

    Frozen and optional: empty tuples mean "no semantic constraint
    declared for this primitive". The enumerator never raises on a
    missing constraint — the field is purely additive metadata.
    """

    redundant_with_substrate_ids: tuple[str, ...] = ()
    expects_substrate_property: tuple[str, ...] = ()
    incompatible_with_substrate_ids: tuple[str, ...] = ()
    requires_primitive_ids: tuple[str, ...] = ()


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
    # Per-primitive semantic-composition constraint. Empty constraint
    # (default) means "no semantic gate beyond the compatibility matrix".
    # Per ZZZZZ medium polish 2026-05-12.
    semantic_constraint: SemanticConstraint = SemanticConstraint()
    # Documentation / rationale tag.
    notes: str = ""


# ── Canonical primitive inventory (FROZEN; bump SCHEMA_VERSION on change) ──


def canonical_primitive_inventory() -> list[PrimitiveRow]:
    """Return the 13 packet-compiler primitives composition registry covers.

    Inventory (13 primitives, alphabetical-by-category-then-id for stable
    matrix construction):

    PR101 GOLD x3 (HNeRV-family substrates only, ordered pipeline):
      1. ``pr101_decoder_storage_order`` (order_index=0)
      2. ``pr101_conv4_storage_perms`` (order_index=1)
      3. ``pr101_decoder_byte_maps`` (order_index=2)

    Sign-encoding x5 (any substrate with int-tensor weights, mutually
    exclusive within a tensor):
      4. ``sign_encoding_negzig``
      5. ``sign_encoding_zig``
      6. ``sign_encoding_twos``
      7. ``sign_encoding_off``
      8. ``sign_encoding_raw_uint8``

    Schema-elision x3 (HNeRV-family decoder grammar, mutually exclusive):
      9. ``pr98_cd1_compact_format``
      10. ``pr100_schema_driven_decoder``
      11. ``pr105_packed_state_schema``

    Magic-codec dense-streams x1 (universal byte-stream wrapper):
      12. ``magic_codec_dense_streams``

    Generic byte-stream compressors x2:
      13. ``brotli`` (contest baseline)
      14. ``lzma`` (universal-density alternative)

    Total: 14 primitives. (PR101 GOLD x3 + sign-encoding x5 + schema-
    elision x3 + magic-codec-dense x1 + brotli + lzma = 14.)

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
            semantic_constraint=SemanticConstraint(
                # step 2 hits the storage-order substrate produced by step 1;
                # without step 1 the conv4 permutation has no consistent
                # tensor layout to permute against.
                requires_primitive_ids=("pr101_decoder_storage_order",),
                expects_substrate_property=("hnerv_family_decoder_layout",),
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
            semantic_constraint=SemanticConstraint(
                # byte-maps overlays on top of conv4 perms; needs both
                # earlier steps to have produced a consistent layout.
                requires_primitive_ids=(
                    "pr101_decoder_storage_order",
                    "pr101_conv4_storage_perms",
                ),
                expects_substrate_property=("hnerv_family_decoder_layout",),
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
            semantic_constraint=SemanticConstraint(
                # Sign-encoding rewrites signed int weights as a more
                # entropy-friendly nibble stream. Substrates whose weights
                # are inherently uint8 categorical / palette tokens gain
                # nothing — they don't carry signed magnitudes.
                expects_substrate_property=("int_tensor_weights",),
                redundant_with_substrate_ids=(
                    "categorical_substrate",
                    "anr_token_renderer_v62",
                ),
            ),
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
            semantic_constraint=SemanticConstraint(
                expects_substrate_property=("int_tensor_weights",),
                redundant_with_substrate_ids=(
                    "categorical_substrate",
                    "anr_token_renderer_v62",
                ),
            ),
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
            semantic_constraint=SemanticConstraint(
                expects_substrate_property=("int_tensor_weights",),
                redundant_with_substrate_ids=(
                    "categorical_substrate",
                    "anr_token_renderer_v62",
                ),
            ),
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
            # Identity passthrough has no semantic constraint by design —
            # it is the ablation baseline and applies everywhere.
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
            semantic_constraint=SemanticConstraint(
                # raw_uint8 is the canonical no-op view for already-uint8
                # categorical substrates — applying it is redundant.
                redundant_with_substrate_ids=(
                    "categorical_substrate",
                    "anr_token_renderer_v62",
                ),
            ),
            notes=(
                "Reinterpret signed int as raw uint8 byte stream; for "
                "already-uint8-quantized weights this is the canonical no-op view"
            ),
        ),
        # ── Schema-elision x3 (HNeRV-family decoder grammar, MX) ──
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
        # ── Magic-codec dense-streams x1 (universal byte-stream auto-select) ──
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
            semantic_constraint=SemanticConstraint(
                # Per ZZZZZ stress-test 2026-05-12: substrates that already
                # ship an arithmetic-coded latent stream gain little from
                # the magic-codec dense-streams auto-selector — the
                # selector's best pick on the substrate's own AR-coded
                # output is the substrate's own AR coder.
                redundant_with_substrate_ids=(
                    "cool_chic_residual",
                    "c3_residual",
                ),
            ),
            notes=(
                "Auto-selects between 6 dense-stream primitives (sparse-RLE, "
                "arithmetic-coefficients, centered-delta-uint8, delta-varint-"
                "pose, categorical-stream, lowpass-luma-residual); "
                "score-claim=false until vendored runtime closure lands"
            ),
        ),
        # ── Generic byte-stream compressors x2 ──
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
            semantic_constraint=SemanticConstraint(
                # Substrates that ship their own AR-coded latent stream
                # (cool_chic, c3) already exhaust the entropy a generic
                # byte-stream coder could find — applying brotli/lzma
                # on top is largely a no-op (residual incompressible).
                redundant_with_substrate_ids=(
                    "cool_chic_residual",
                    "c3_residual",
                ),
            ),
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
            semantic_constraint=SemanticConstraint(
                redundant_with_substrate_ids=(
                    "cool_chic_residual",
                    "c3_residual",
                ),
            ),
            notes=(
                "LZMA / xz universal-density entropy coder; tends to outperform "
                "brotli on small dense streams but at higher decode cost"
            ),
        ),
    ]


# ── Compatibility matrix (substrate_class x primitive_category) ──────────


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
    """One (substrate x ordered primitives x order) composition cell.

    The cell is the autopilot's ranking unit. Each cell carries:

    - ``substrate_id``: a key into ``canonical_substrate_inventory()``.
    - ``primitives``: a tuple of ``(primitive_id, kwargs)`` pairs in
      the order they apply. Mutually-exclusive primitives appear at most
      once per category.
    - ``composition_order``: an explicit ordering token for the pipeline
      ("storage_order → conv4_perms → byte_maps" for PR101 GOLD; or
      "sign_encoding → brotli" for the universal byte-stream path).
    - ``predicted_bytes_delta``: aggregated from primitive metadata.
    - ``predicted_score_delta``: aggregated; ``[predicted; substrate x primitive matrix v1]``.
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
    # FIX-D enrichment 1 (ZZZZZ medium polish 2026-05-12): when the cell
    # is FORMALLY compatible (passes the compatibility matrix AND ordering
    # rules) but one or more primitives is semantically a no-op on the
    # substrate (e.g., magic_codec primitive applied to a substrate that
    # already ships its own AR-coded latent stream), populate a short
    # operator-facing string. ``None`` means "no semantic warning to
    # surface". Refused cells (compatibility_verdict != "compatible*")
    # never set this field — they carry refused_reason instead.
    semantic_compatibility_warning: str | None = None
    # FIX-D enrichment 2 (ZZZZZ medium polish 2026-05-12): when the cell
    # is REFUSED (compatibility_verdict != "compatible*"), the typed
    # taxonomy classifying WHY the cell was refused. ``None`` for
    # compatible cells. The free-form rationale stays in ``blockers``;
    # this field makes the refusal class machine-checkable.
    refused_reason: RefusedReason | None = None

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
            "semantic_compatibility_warning": self.semantic_compatibility_warning,
            "refused_reason": (
                self.refused_reason.value
                if self.refused_reason is not None
                else None
            ),
        }

    def _format_notes(self) -> str:
        lines = [
            "[predicted; substrate x primitive composition matrix v1]",
            f"substrate_id: {self.substrate_id}",
            f"primitives (in order): {list(self.primitive_ids())}",
            f"composition_order: {list(self.composition_order)}",
            f"predicted_bytes_delta: {self.predicted_bytes_delta}",
            f"compatibility_verdict: {self.compatibility_verdict}",
        ]
        if self.semantic_compatibility_warning is not None:
            lines.append(
                f"semantic_compatibility_warning: "
                f"{self.semantic_compatibility_warning}"
            )
        if self.refused_reason is not None:
            lines.append(f"refused_reason: {self.refused_reason.value}")
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
    the reason the pipeline violates the rules. Kept stable for
    backward-compat consumers; new callers should prefer
    :func:`classify_pipeline_violation` which returns the typed
    :class:`RefusedReason` alongside the rationale.
    """
    ok, _reason, rationale = classify_pipeline_violation(
        primitive_ids, primitives_by_id
    )
    return ok, rationale


def classify_pipeline_violation(
    primitive_ids: list[str],
    primitives_by_id: dict[str, PrimitiveRow],
) -> tuple[bool, RefusedReason | None, str]:
    """Validate a primitive pipeline and classify the refusal class.

    Per FIX-D 2026-05-12 (ZZZZZ medium polish), this function pairs the
    free-form rationale with a typed :class:`RefusedReason`. Downstream
    consumers can branch on the refusal class without parsing prose.

    Returns
    -------
    (ok, reason, rationale):
        ``ok`` is True iff the pipeline satisfies all ordering / MX /
        duplicate / unknown-primitive rules. ``reason`` is None when
        ``ok`` is True; otherwise it is the typed refusal class.
        ``rationale`` is the human-readable explanation (same as the
        legacy :func:`validate_pipeline_ordering` return).

    Rules enforced (with typed classification):

    1. **No duplicates** → :attr:`RefusedReason.DUPLICATE_PRIMITIVE`.
    2. **Known primitive_ids** → :attr:`RefusedReason.UNKNOWN_PRIMITIVE`.
    3. **Cross-category order_index monotonic** →
       :attr:`RefusedReason.ORDERING_VIOLATION`.
    4. **Mutual-exclusion within MX categories** →
       :attr:`RefusedReason.MUTUAL_EXCLUSION`.
    5. **Ordered-pipeline within-category order** →
       :attr:`RefusedReason.ORDERING_VIOLATION`.
    """
    if not primitive_ids:
        return True, None, "empty pipeline trivially valid"

    if len(set(primitive_ids)) != len(primitive_ids):
        dupes = sorted({p for p in primitive_ids if primitive_ids.count(p) > 1})
        return (
            False,
            RefusedReason.DUPLICATE_PRIMITIVE,
            f"duplicate primitive_ids: {dupes}",
        )

    rows: list[PrimitiveRow] = []
    for pid in primitive_ids:
        if pid not in primitives_by_id:
            return (
                False,
                RefusedReason.UNKNOWN_PRIMITIVE,
                f"unknown primitive_id: {pid!r}",
            )
        rows.append(primitives_by_id[pid])

    # Rule 3: cross-category order_index monotonic.
    indices = [r.order_index for r in rows]
    if indices != sorted(indices):
        return (
            False,
            RefusedReason.ORDERING_VIOLATION,
            (
                f"cross-category order_index not monotonic: {indices} "
                f"for {primitive_ids}; smaller order_index must appear first"
            ),
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
            return (
                False,
                RefusedReason.MUTUAL_EXCLUSION,
                (
                    f"mutually-exclusive category {cat.value} has multiple "
                    f"primitives: {members}; choose one"
                ),
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
            return (
                False,
                RefusedReason.ORDERING_VIOLATION,
                (
                    f"category {cat.value} pipeline contains primitive not in "
                    f"declared within_category_order: {exc.args[0]!r}"
                ),
            )
        if observed_indices != sorted(observed_indices):
            return (
                False,
                RefusedReason.ORDERING_VIOLATION,
                (
                    f"category {cat.value} pipeline order {observed_order} "
                    f"violates declared within_category_order {expected_order}"
                ),
            )

    return True, None, "pipeline ordering valid"


def compute_semantic_warning(
    substrate_id: str,
    primitives: list[PrimitiveRow],
) -> str | None:
    """Compute a semantic-compatibility warning for a (substrate, pipeline) pair.

    Returns ``None`` when no warning applies. Walks each primitive's
    :class:`SemanticConstraint`:

    - Primitives that declare ``redundant_with_substrate_ids`` containing
      ``substrate_id`` contribute a redundancy warning.
    - Primitives that declare ``incompatible_with_substrate_ids``
      containing ``substrate_id`` contribute a SEMANTIC_INCOMPATIBLE
      caveat (this surface only — the caller decides whether to mark
      the cell refused via :class:`RefusedReason.SEMANTIC_INCOMPATIBLE`).

    Multiple primitives can contribute; the warnings are joined with
    ``" | "``. Per FIX-D 2026-05-12 (ZZZZZ medium polish).
    """
    warnings: list[str] = []
    for p in primitives:
        sc = p.semantic_constraint
        if substrate_id in sc.redundant_with_substrate_ids:
            warnings.append(
                f"primitive {p.primitive_id!r} is semantically redundant on "
                f"substrate {substrate_id!r} (substrate already provides "
                f"this primitive's benefit; applying it is a no-op)"
            )
        if substrate_id in sc.incompatible_with_substrate_ids:
            warnings.append(
                f"primitive {p.primitive_id!r} is semantically incompatible "
                f"with substrate {substrate_id!r} (structural mismatch even "
                f"though compatibility matrix permits the category)"
            )
    if not warnings:
        return None
    return " | ".join(warnings)


def detect_dependency_violation(
    primitive_ids: list[str],
    primitives_by_id: dict[str, PrimitiveRow],
) -> tuple[bool, str | None]:
    """Detect required-primitive dependencies not satisfied by the pipeline.

    Returns ``(ok, rationale)``. ``ok`` is True when every primitive's
    ``semantic_constraint.requires_primitive_ids`` is present in the
    pipeline.

    Caller uses :class:`RefusedReason.DEPENDENCY_MISSING` to classify
    the refusal.
    """
    pipeline_set = set(primitive_ids)
    for pid in primitive_ids:
        row = primitives_by_id.get(pid)
        if row is None:
            continue
        for required in row.semantic_constraint.requires_primitive_ids:
            if required not in pipeline_set:
                return False, (
                    f"primitive {pid!r} requires {required!r} in the same "
                    f"pipeline, but the pipeline is {sorted(pipeline_set)!r}"
                )
    return True, None


def detect_substrate_semantic_incompatibility(
    substrate_id: str,
    primitives: list[PrimitiveRow],
) -> tuple[bool, str | None]:
    """Detect hard substrate x primitive semantic incompatibility.

    Returns ``(ok, rationale)``. ``ok`` is True when no primitive's
    ``semantic_constraint.incompatible_with_substrate_ids`` lists
    ``substrate_id``. When False, the caller classifies the refusal as
    :class:`RefusedReason.SEMANTIC_INCOMPATIBLE`.
    """
    for p in primitives:
        if substrate_id in p.semantic_constraint.incompatible_with_substrate_ids:
            return False, (
                f"primitive {p.primitive_id!r} declares substrate "
                f"{substrate_id!r} as semantically incompatible"
            )
    return True, None


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
    # FIX-D 2026-05-12: ensure SemanticConstraint serializes as JSON-friendly
    # lists rather than the dataclass-as-dict default tuples.
    sc = p.semantic_constraint
    d["semantic_constraint"] = {
        "redundant_with_substrate_ids": list(sc.redundant_with_substrate_ids),
        "expects_substrate_property": list(sc.expects_substrate_property),
        "incompatible_with_substrate_ids": list(
            sc.incompatible_with_substrate_ids
        ),
        "requires_primitive_ids": list(sc.requires_primitive_ids),
    }
    return d


def _cell_to_dict(c: CompositionCell) -> dict[str, Any]:
    d = dataclasses.asdict(c)
    d["substrate_class"] = c.substrate_class.value
    d["primitives"] = [
        {"primitive_id": pid, "kwargs": kw} for pid, kw in c.primitives
    ]
    d["composition_order"] = list(c.composition_order)
    d["blockers"] = list(c.blockers)
    # FIX-D 2026-05-12: serialize RefusedReason as its enum value (string)
    # for JSON friendliness; None passes through unchanged.
    d["refused_reason"] = (
        c.refused_reason.value if c.refused_reason is not None else None
    )
    # semantic_compatibility_warning is already a string-or-None; no
    # transformation needed but call out explicitly for clarity.
    d["semantic_compatibility_warning"] = c.semantic_compatibility_warning
    return d


def serialize_primitive_inventory() -> list[dict[str, Any]]:
    """JSON-safe serialization of the primitive inventory."""
    return [_primitive_to_dict(p) for p in canonical_primitive_inventory()]


__all__ = [
    "PLANNING_ONLY",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCHEMA_VERSION",
    "SCORE_CLAIM",
    "CompositionCell",
    "PrimitiveCategory",
    "PrimitiveOrderSensitivity",
    "PrimitiveRow",
    "RefusedReason",
    "ScoreAxis",
    "SemanticConstraint",
    "SubstrateClass",
    "SubstrateRow",
    "_cell_to_dict",
    "_primitive_to_dict",
    "canonical_primitive_inventory",
    "canonical_substrate_inventory",
    "classify_pipeline_violation",
    "compute_semantic_warning",
    "detect_dependency_violation",
    "detect_substrate_semantic_incompatibility",
    "primitive_compatibility",
    "serialize_primitive_inventory",
    "validate_pipeline_ordering",
]
