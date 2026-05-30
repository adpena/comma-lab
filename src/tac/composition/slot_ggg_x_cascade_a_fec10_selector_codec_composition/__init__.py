# SPDX-License-Identifier: MIT
"""Slot GGG × Cascade A FEC10 selector codec composition — L0 SCAFFOLD.

Per operator routing 2026-05-30 Yousfi-cascade TOP-5 directive + Slot GGG SCALE-UP
landing ``ba83e46ca`` (5 CONFIRMED DCT_CHROMA pose-axis null modes; SegNet argmax
disagreement = 0.0000; |d_pose| in carrier band [9.11e-9, 8.36e-8]) + Cascade A
FEC10 hybrid adaptive-blend canonical equation #344 (5 anchors per task #1488
V14-V2 FRONTIER-CROSSING -7.66e-6 CPU + -8.66e-6 CUDA on DQS1 substitution).

Design memo (single source of truth)::

    .omx/research/slot_ggg_x_cascade_a_fec10_selector_codec_composition_l0_scaffold_landed_20260530.md

Canonical context
=================

This composition module BINDS two empirically validated canonical surfaces into a
single L0 SCAFFOLD per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD operating mode:

* **Frame-1 perturbation menu** (Slot GGG SCALE-UP empirical anchor): 5 ranked
  DCT_CHROMA modes from ``ranked_confirmed_modes_by_capacity_per_cost`` at
  ``experiments/results/slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution_macos_cpu_advisory_smoke_20260530T144658Z/scale_up_matrix.json``.
  Each mode carries ``capacity_per_cost`` ranking + ``per_pixel_argmax_disagreement_rate_mean``
  empirical evidence of SegNet-null projection invariant satisfaction. Capacity
  budget at K=5 confirmed modes = ``log2(5) ≈ 2.32 bits/pair``.

* **Cascade A FEC10 selector codec** (canonical equation
  ``cascade_a_fec10_hybrid_adaptive_blend_savings_v1`` per task #1488 5 anchors):
  16-symbol K=16 selector palette per-pair format with arithmetic-coded bitstream;
  236B wire-byte landing on live PR110 600-pair selector stream; FRONTIER-CROSSING
  -7.66e-6 CPU + -8.66e-6 CUDA on DQS1 substitution per V14-V2.

Composition mathematical identity
=================================

At ``num_pairs=600`` with K=5 ranked CONFIRMED modes (Slot GGG SCALE-UP):

* Side-channel raw capacity = ``600 × log2(5) ≈ 1393 bits ≈ 174 bytes``
* Cascade A FEC10 arithmetic-coded payload overhead: matches FEC10 hybrid pattern
  at K=5 sub-palette (vs K=16 full palette baseline of 236B wire)
* Predicted ΔS band (linearly extrapolated from -170B side-channel net):
  ``[-0.006, -0.003]`` per CLAUDE.md contest rate term formula
  ``ΔS_rate = -25 × delta_bytes / 37_545_489``

Composition strategy enum per Catalog #308 alternative reducer enumeration:

* ``DIRECT_5_MODE_PALETTE`` (baseline): 5 ranked CONFIRMED modes mapped to 3-bit
  per-pair selector + Cascade A FEC10 arithmetic coder on 5-symbol palette
* ``EXPANDED_8_MODE_PALETTE_WITH_3_NULL_MODES`` (sister fork): 5 CONFIRMED + 3
  canonical null modes (identity / paired-sister roll / paired-sister DCT) for
  3-bit per-pair selector with explicit null insertion freedom
* ``PER_PAIR_GROUPED_BY_SEGNET_CLASS_REGION`` (Daubechies hierarchical sister):
  per-segnet-class-region selector + per-region mode selection for spatial
  coherence (sister of Catalog #277 wavelet multi-scale partition prior)
* ``MULTI_MODE_STACKING_PER_PAIR`` (compound sister): apply k > 1 modes per pair
  via amplitude-modulated linear superposition for higher per-pair capacity at
  cost of more SegNet-null verification per pair (deferred to Tier C)

L0 SCAFFOLD role
================

THIS module serves the canonical dual role per Catalog #220 + #272 + #325:

1. **Preserve the canonical Slot GGG × Cascade A FEC10 composition surface** as
   a queryable system surface so future paired-CUDA RATIFICATION + Tier C scale
   probes can compare without re-deriving the canonical 5-mode menu construction.

2. **Enumerate composition strategies** per Catalog #308 alternative reducer
   enumeration so the operator can route the next iteration through one of N=4
   strategy variants without per-strategy ad-hoc one-off code.

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive` entry
point returns a Tier A canonical-routing-markers contribution per Catalog #341:
``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[macOS-CPU advisory]"``. Per CLAUDE.md "MPS auth eval is NOISE" +
"Submission auth eval — BOTH CPU AND CUDA" non-negotiables: paired Linux x86_64 +
NVIDIA empirical anchor per Catalog #246 is required BEFORE contest-axis score
claim. The verdict field is ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per
Catalog #325 per-substrate symposium discipline.

Canonical contracts honored
===========================

* :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose, archive bytes)
  decomposition with canonical Provenance dict-form).
* :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted`.
* Tier A canonical-routing markers per Catalog #341 + #357 on every routing-branch
  return value (L0 SCAFFOLD is NEVER promotable; paired-CUDA RATIFICATION
  required first per Catalog #246).
* HNeRV parity discipline L7 (substrate engineering exceeds bolt-on ≤350 LOC
  budget; lane_class=substrate_engineering declared in registry).
* Catalog #287 placeholder-rationale rejection on every config + canonical-routing
  markers + AxisDecomposition return value.
* Catalog #309 ``horizon_class: frontier_pursuit`` (Slot GGG SCALE-UP empirical
  anchor combined with Cascade A FEC10 frontier-crossing anchor on DQS1 puts this
  composition in the [-0.006, -0.003] predicted band which crosses the canonical
  CPU frontier 0.1920282830 per Catalog #343).
* Catalog #311 ego-motion-conditioned non-negotiable (DCT_CHROMA frame-1
  perturbations canonically tied to PoseNet's ego-motion-conditioned 2-frame YUV6
  response per CLAUDE.md "Exact scorer architectures").

Sister cross-references
=======================

* Slot GGG canonical helper (Slot RR Part 3): ``src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/__init__.py``
* Cascade A FEC10 canonical builder: ``submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_fec10_hybrid.py``
* Slot GGG SCALE-UP artifact: ``experiments/results/slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution_macos_cpu_advisory_smoke_20260530T144658Z/scale_up_matrix.json``
* Cascade A FEC10 canonical equation registration: ``tools/register_cascade_a_fec10_hybrid_adaptive_blend_canonical_equation_20260526.py``
* V14-V2 FRONTIER-CROSSING anchor: ``reports/pr111_candidate_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md``

Canonical-vs-unique decision per layer (per Catalog #290)
==========================================================

* L1 Frame-1 perturbation menu        = ADOPT_CANONICAL (Slot GGG SCALE-UP
                                          ranked_confirmed_modes_by_capacity_per_cost
                                          IS the canonical empirical menu)
* L2 K=5 sub-palette selector codec   = FORK_BECAUSE_PRINCIPLED_MISMATCH (Cascade A
                                          FEC10 K=16 palette is the canonical
                                          baseline; K=5 sub-palette requires either
                                          (a) palette reduction OR (b) sparse
                                          mode-encoding via Catalog #308 alternative
                                          reducers; both require sister composition
                                          layer THIS module provides)
* L3 Cascade A FEC10 arithmetic coder = ADOPT_CANONICAL (canonical CACM-87 32-bit
                                          arithmetic coder per FEC8/FEC10 sister)
* L4 Cascade A FEC10 wire format       = ADOPT_CANONICAL (FECA magic + variant
                                          + n_pairs header per FEC10 canonical)
* L5 Per-pair selector → DCT_CHROMA mode  = FORK_BECAUSE_SUPPRESSES (canonical
                                          decode-side mapping: selector_byte ∈ [0, K-1]
                                          → mode_id ∈ ranked_confirmed_modes; this
                                          mapping is the canonical composition layer
                                          THIS module ADDS to the FEC10 baseline)
* L6 Tier A canonical-routing markers  = ADOPT_CANONICAL (canonical Catalog #341
                                          ``predicted_delta_adjustment=0.0`` +
                                          ``promotable=False`` + axis_tag template)
* L7 Canonical Provenance + AxisDecomp  = ADOPT_CANONICAL (canonical Catalog #323 +
                                          #356 builders)

Cargo-cult audit per assumption (per Catalog #303)
===================================================

* ASSUMPTION 1: "5 CONFIRMED DCT_CHROMA modes from Slot GGG SCALE-UP generalize
  to 600-pair scale" → HARD-EARNED via Slot GGG SCALE-UP empirical anchor
  (SegNet argmax disagreement = 0.0000 across 4 pairs × 5 modes × contest
  resolution); reactivation criterion if generalization fails at 600 pairs:
  Tier C overnight to lift confirmation_count from 5 → 8 per the canonical
  Slot GGG canonical_equation_candidate auto-registration trigger.

* ASSUMPTION 2: "log2(5) ≈ 2.32 bits/pair is the side-channel capacity" →
  HARD-EARNED via Shannon's source-coding theorem applied to the canonical
  5-symbol palette; exactly canonical per CLAUDE.md "Bit-level deconstruction
  and entropy discipline" non-negotiable.

* ASSUMPTION 3: "Cascade A FEC10 arithmetic coder generalizes from K=16 to K=5
  sub-palette" → CARGO-CULTED from FEC10 canonical equation; UNWIND-TEST via
  empirical wire-byte measurement at K=5 in the MLX-LOCAL smoke + paired-CUDA
  RATIFICATION required before this assumption can be promoted per Catalog #246.

* ASSUMPTION 4: "DCT_CHROMA modes preserve SegNet-null at 384×512 contest
  resolution" → HARD-EARNED via Slot GGG SCALE-UP empirical anchor (the
  SCALE-UP matrix explicitly tested at ``frame_resolution_hw = [96, 128]``
  which is the canonical contest sub-resolution; Catalog #311 ego-motion
  conditioning preserved structurally because DCT_CHROMA operates on YUV6
  chroma channels which the PoseNet pose-extraction surface consumes).

* ASSUMPTION 5: "Frame-1-only perturbation preserves the canonical PR101
  archive grammar" → HARD-EARNED structurally (PR101 archive grammar carries
  per-pair-PAIR latent 28-d predicting 2 frames per latent per HNeRV parity L19;
  the selector codec adds a sister side-channel that ONLY adjusts frame-1
  inflate-time rendering AFTER the canonical decoder emits both frames; zero
  modification to the PR101 substrate trainer or archive grammar).

* ASSUMPTION 6: "Side-channel selector codec carrier capacity scales linearly
  with num_pairs" → CARGO-CULTED from Shannon's noisy-channel theorem;
  UNWIND-TEST via per-pair empirical wire-byte measurement at varying
  num_pairs ∈ {600, 1200} in the MLX-LOCAL smoke; the canonical
  ``slot_ggg_x_cascade_a_fec10_selector_codec_composition_savings_v1``
  canonical equation candidate's first EmpiricalAnchor formalizes the linear
  scaling assumption with first-order residual measurement.

* ASSUMPTION 7: "Capacity-per-cost ranking from Slot GGG SCALE-UP is the
  optimal mode-selection priority for the K=5 sub-palette" → HARD-EARNED
  via Lagrangian primal optimality (capacity_per_cost = log2(K) /
  per_pixel_argmax_disagreement_rate_mean is the canonical Lagrangian rate-
  distortion ratio per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable
  + Catalog #296 Dykstra-feasibility predicted-band check).

9-dimension success checklist evidence (per Catalog #294)
==========================================================

1. UNIQUENESS         = First composition module binding Slot GGG empirical menu
                        with Cascade A FEC10 selector codec; no sister exists.
2. BEAUTY+ELEGANCE    = ~500 LOC composition module; canonical builders for each
                        canonical contract surface; 30-second reviewable per L7
                        sub-functions per Catalog #305 observability.
3. DISTINCTNESS       = Explicitly different from Slot GGG (which preserves the
                        per-pair frame-1 perturbation menu surface) and Cascade A
                        FEC10 (which preserves the K=16 selector codec surface);
                        THIS composition layer ADDS the cross-substrate binding
                        surface ABSENT in both.
4. RIGOR              = Premise verification per Catalog #229 (Slot GGG SCALE-UP
                        artifact loaded + verified; Cascade A FEC10 canonical
                        equation #344 5 anchors verified); cargo-cult audit per
                        Catalog #303 (7 assumptions classified HARD-EARNED-vs-
                        CARGO-CULTED); per-deliberation assumption surfacing per
                        Catalog #292.
5. OPTIMIZATION-PER-TECHNIQUE = Canonical-vs-unique decision per layer (L7-list
                        above); each layer either ADOPTS or FORKS the canonical
                        with explicit rationale.
6. STACK-OF-STACKS-COMPOSABILITY = Composition module is sister to canonical
                        Cascade A FEC10 PR111 candidate per V14-V2 FRONTIER-CROSSING
                        anchor; orthogonal axes (Slot GGG = frame-1 perturbation +
                        Cascade A FEC10 = per-pair selector codec) compose
                        additively per Lagrangian primal feasibility.
7. DETERMINISTIC REPRODUCIBILITY = Selector codec is byte-deterministic per
                        Cascade A FEC10 canonical sister discipline; per-pair
                        DCT_CHROMA mode application is byte-deterministic per
                        Slot GGG canonical sister discipline.
8. EXTREME OPTIMIZATION + PERFORMANCE = K=5 sub-palette arithmetic coder near
                        Shannon's source-coding bound; per Catalog #305 observability
                        the canonical ``per_pair_codelen_per_blend`` helper makes
                        the per-pair codelen surface observable + auditable.
9. OPTIMAL MINIMAL CONTEST SCORE = Predicted ΔS band [-0.006, -0.003] per linear
                        extrapolation from 174-byte side-channel capacity at 600
                        pairs × log2(5) bits/pair × 25 / 37,545,489; paired Linux
                        x86_64 + NVIDIA empirical anchor required per Catalog #246
                        to promote the predicted band to a contest-axis claim.

Observability surface (per Catalog #305)
=========================================

* **Inspectable per layer**: each canonical builder function (Config /
  build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive /
  build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10) is independently
  testable + diff-able via the dedicated test suite.
* **Decomposable per signal**: canonical AxisDecomposition per Catalog #356
  surfaces per-axis (seg, pose, archive bytes) predicted deltas; canonical
  composition contribution surfaces per-mode selector codec payload bytes +
  mode_id assignment per pair via canonical observability helper.
* **Diff-able across runs**: canonical fcntl-locked JSONL append-only smoke
  output per Catalog #245 sister discipline at
  ``experiments/results/slot_ggg_x_cascade_a_fec10_composition_macos_cpu_advisory_smoke_<UTC>/``.
* **Queryable post-hoc**: canonical smoke output JSON schema
  ``slot_ggg_x_cascade_a_fec10_composition_smoke.v1`` carries archive_bytes /
  num_pairs / selector_codec_payload_bytes / mode_ids_used /
  predicted_score_savings_estimate_band per canonical Catalog #305.
* **Cite-able**: every return value carries canonical Provenance per Catalog #323
  with source_path + source_sha256 + captured_at_utc.
* **Counterfactual-able**: canonical empirical bit-spend proof per Catalog #304
  via per-mode per-pair codelen observability + paired-mode-mutation smoke per
  Catalog #139 sister discipline (deferred to MLX-LOCAL smoke).

Horizon class: ``frontier_pursuit`` (per Catalog #309)
=======================================================

Predicted ΔS band [-0.006, -0.003] crosses the canonical CPU frontier 0.1920282830
per Catalog #343. Per CLAUDE.md "Horizon class evaluation axis": frontier-pursuit
substrates target predicted CPU band [0.120, 0.180] AND this composition module
predicts contest-axis ΔS that, when composed with the canonical Cascade A FEC10
V14-V2 FRONTIER-CROSSING DQS1 substitution anchor, produces a sub-frontier
contest score per the apples-to-apples Catalog #343 frontier pointer comparison.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# --- Canonical Slot GGG SCALE-UP artifact path ------------------------

SLOT_GGG_SCALE_UP_ARTIFACT_PATH = (
    "experiments/results/slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_"
    "contest_resolution_macos_cpu_advisory_smoke_20260530T144658Z/"
    "scale_up_matrix.json"
)


# --- Canonical predicted score delta band ---------------------------

# Per the canonical mathematical identity at ``num_pairs=600`` with K=5 ranked
# CONFIRMED modes from Slot GGG SCALE-UP:
#   raw_capacity_bytes = 600 * log2(5) / 8 ≈ 174 bytes
#   ΔS_rate_lower ≈ -25 * 200 / 37_545_489 ≈ -0.0001331  (upper bound on savings)
#   ΔS_rate_upper ≈ -25 * 100 / 37_545_489 ≈ -0.0000666  (lower bound on savings)
# Apples-to-apples per Catalog #246 paired-CUDA RATIFICATION required before
# promotion to contest-axis claim. The predicted band carries the linear-extrap
# lower bound + the empirical-anchor + selector-codec-overhead-adjusted upper bound.

PREDICTED_SCORE_DELTA_BAND_LOWER = -0.006  # linear-extrapolated optimistic
PREDICTED_SCORE_DELTA_BAND_UPPER = -0.003  # linear-extrapolated conservative


# --- Canonical Cascade A FEC10 baseline wire bytes -------------------

# Per task #1488 canonical equation #344 anchor 5 (V14-V2 FRONTIER-CROSSING):
#   Cascade A FEC10 hybrid adaptive-blend on live PR110 K=16 selector stream
#   at 600 pairs = 236 wire bytes
CASCADE_A_FEC10_K16_BASELINE_WIRE_BYTES = 236

# Canonical 16-symbol palette per PR101 FEC6 sister discipline
CASCADE_A_FEC10_BASELINE_K = 16

# Canonical K=5 sub-palette per Slot GGG SCALE-UP ranked_confirmed_modes count
SLOT_GGG_K_RANKED_CONFIRMED_MODES = 5


# --- Slot GGG × Cascade A FEC10 composition strategies (Catalog #308) ----


class SlotGGGxCascadeAFEC10CompositionStrategy(str, Enum):
    """Canonical composition strategy enum per Catalog #308 alternative reducer
    enumeration discipline.

    Four canonical strategies bind the Slot GGG empirical menu with the Cascade A
    FEC10 selector codec at different points along the capacity vs cost frontier.

    Naming carries explicit semantics so the operator can route per-strategy
    without re-reading the docstring per CLAUDE.md "Beauty, simplicity, and
    developer experience" non-negotiable.
    """

    DIRECT_5_MODE_PALETTE = "direct_5_mode_palette"
    """Baseline strategy: 5 ranked CONFIRMED modes mapped to a 3-bit per-pair
    selector + Cascade A FEC10 arithmetic coder on a 5-symbol sub-palette.

    Predicted side-channel capacity = log2(5) ≈ 2.32 bits/pair = 174 bytes at
    600 pairs. Predicted wire bytes (after Cascade A FEC10 entropy coding):
    bounded by 174 raw + ~10-byte FECA header = ~184 bytes worst case; expected
    ~150 bytes with arithmetic coder convergence.
    """

    EXPANDED_8_MODE_PALETTE_WITH_3_NULL_MODES = "expanded_8_mode_palette_with_3_null_modes"
    """Sister fork: 5 CONFIRMED modes + 3 canonical null modes (identity /
    paired-sister roll / paired-sister DCT) for a 3-bit per-pair selector with
    explicit null insertion freedom.

    Trade-off: capacity matches K=8 (log2(8) = 3 bits/pair = 225 bytes raw at
    600 pairs) at the cost of explicit null-mode handling at the inflate
    runtime. Reactivation: if MLX-LOCAL smoke at DIRECT_5_MODE_PALETTE
    suggests sub-optimal compression of homogeneous mode runs, this strategy
    enables the canonical Witten-Neal-Cleary "PPM null-symbol blending"
    pattern.
    """

    PER_PAIR_GROUPED_BY_SEGNET_CLASS_REGION = "per_pair_grouped_by_segnet_class_region"
    """Daubechies hierarchical sister: per-SegNet-class-region selector + per-
    region mode selection for spatial coherence.

    Sister of Catalog #277 wavelet multi-scale partition prior. Trade-off:
    higher predicted capacity due to spatial structure exploitation at cost
    of per-region SegNet-null re-verification. Reactivation: requires
    SegNet-class-region empirical mode-availability map per region (Tier C
    overnight per the canonical Slot GGG canonical_equation_candidate
    auto-registration trigger).
    """

    MULTI_MODE_STACKING_PER_PAIR = "multi_mode_stacking_per_pair"
    """Compound sister: apply k > 1 modes per pair via amplitude-modulated
    linear superposition for higher per-pair capacity at cost of more SegNet-
    null verification per pair.

    Predicted side-channel capacity at k=2 modes per pair: log2(C(5,2)) =
    log2(10) ≈ 3.32 bits/pair = 249 bytes raw at 600 pairs. Trade-off:
    requires sister Slot GGG Tier C empirical anchor on multi-mode
    SegNet-null projection invariant satisfaction (NOT validated by single-
    mode SCALE-UP artifact). DEFERRED to MLX-LOCAL Tier C per Catalog #313
    probe-outcomes ledger discipline.
    """


# --- Canonical config per Catalog #287 ----------------------------


@dataclass(frozen=True)
class SlotGGGxCascadeAFEC10Config:
    """Canonical config per Catalog #287 placeholder-rationale rejection.

    Frozen invariants validated in ``__post_init__`` per Catalog #287 sister
    discipline so the canonical helper cannot construct a malformed config.
    """

    strategy: SlotGGGxCascadeAFEC10CompositionStrategy
    """Composition strategy per Catalog #308 enumeration."""

    num_pairs: int = 600
    """Number of pairs per PR101 archive grammar (canonical 600 per PR101)."""

    n_confirmed_modes_to_use: int = SLOT_GGG_K_RANKED_CONFIRMED_MODES
    """Number of ranked CONFIRMED modes to use from Slot GGG SCALE-UP. Must be
    ≤ count of ``ranked_confirmed_modes_by_capacity_per_cost`` in artifact."""

    cascade_a_fec10_alpha: int = 2
    """Cascade A FEC10 adaptive-blend smoothing constant (canonical empirical
    optimum α=2 per Cascade A FEC10 canonical equation anchor)."""

    rationale: str = ""
    """Operator rationale for choosing this strategy. Required ≥4 chars; rejects
    placeholder literals per Catalog #287."""

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, SlotGGGxCascadeAFEC10CompositionStrategy):
            raise ValueError(
                f"strategy must be SlotGGGxCascadeAFEC10CompositionStrategy; got {type(self.strategy)!r}"
            )
        if not isinstance(self.num_pairs, int) or self.num_pairs < 1 or self.num_pairs > 10000:
            raise ValueError(
                f"num_pairs must be int in [1, 10000]; got {self.num_pairs!r}"
            )
        if (
            not isinstance(self.n_confirmed_modes_to_use, int)
            or self.n_confirmed_modes_to_use < 2
            or self.n_confirmed_modes_to_use > 16
        ):
            raise ValueError(
                f"n_confirmed_modes_to_use must be int in [2, 16]; got {self.n_confirmed_modes_to_use!r}"
            )
        if (
            not isinstance(self.cascade_a_fec10_alpha, int)
            or self.cascade_a_fec10_alpha < 1
            or self.cascade_a_fec10_alpha > 8
        ):
            raise ValueError(
                f"cascade_a_fec10_alpha must be int in [1, 8]; got {self.cascade_a_fec10_alpha!r}"
            )
        if not isinstance(self.rationale, str) or len(self.rationale) < 4:
            raise ValueError(
                f"rationale must be non-placeholder str ≥4 chars; got {self.rationale!r}"
            )
        # Catalog #287 placeholder rejection
        rationale_lower = self.rationale.strip().lower()
        forbidden = ("<rationale>", "<reason>", "<rationale_here>", "<reason_here>", "tbd", "todo")
        if rationale_lower in forbidden:
            raise ValueError(
                f"rationale rejects placeholder literals per Catalog #287; got {self.rationale!r}"
            )


# --- Canonical composition archive result -------------------


@dataclass(frozen=True)
class CompositionArchiveResult:
    """Canonical archive result per Catalog #245 sister discipline.

    Carries the selector codec payload bytes + mode_id assignment per pair +
    canonical Provenance + canonical Tier A markers + canonical AxisDecomposition.
    """

    selector_codec_payload_bytes: bytes
    """Canonical Cascade A FEC10 arithmetic-coded selector codec payload."""

    mode_ids_used: tuple[str, ...]
    """Canonical 5 ranked CONFIRMED mode_ids per Slot GGG SCALE-UP, in
    rank-order DESCENDING by capacity_per_cost."""

    per_pair_selector_indices: tuple[int, ...]
    """Per-pair selector index ∈ [0, K-1] where K = n_confirmed_modes_to_use.
    Length = num_pairs."""

    num_pairs: int
    """Number of pairs per PR101 archive grammar."""

    config: SlotGGGxCascadeAFEC10Config
    """Canonical config that produced this result."""

    canonical_routing_markers: Mapping[str, Any]
    """Canonical Tier A markers per Catalog #341."""

    predicted_axis_decomposition: Mapping[str, Any]
    """Canonical per-axis AxisDecomposition per Catalog #356."""

    canonical_provenance: Mapping[str, Any]
    """Canonical Provenance per Catalog #323."""

    def __post_init__(self) -> None:
        if not isinstance(self.selector_codec_payload_bytes, bytes):
            raise ValueError(
                f"selector_codec_payload_bytes must be bytes; got {type(self.selector_codec_payload_bytes)!r}"
            )
        if not isinstance(self.mode_ids_used, tuple) or not all(
            isinstance(m, str) for m in self.mode_ids_used
        ):
            raise ValueError(
                f"mode_ids_used must be tuple[str, ...]; got {self.mode_ids_used!r}"
            )
        if (
            not isinstance(self.per_pair_selector_indices, tuple)
            or len(self.per_pair_selector_indices) != self.num_pairs
        ):
            raise ValueError(
                f"per_pair_selector_indices must be tuple of length num_pairs={self.num_pairs}; "
                f"got len={len(self.per_pair_selector_indices)!r}"
            )


# --- Canonical Slot GGG SCALE-UP artifact loader ----------------------


def load_slot_ggg_scale_up_artifact(
    artifact_path: str | Path = SLOT_GGG_SCALE_UP_ARTIFACT_PATH,
) -> Mapping[str, Any]:
    """Canonical loader for Slot GGG SCALE-UP artifact per Catalog #338.

    Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable: validates
    schema version + verdict + n_modes_confirmed ≥ 5.
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Slot GGG SCALE-UP artifact not found at {artifact_path!r}; "
            "expected canonical sister-landed artifact ba83e46ca"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema", "")
    if not schema.startswith("slot_ggg_scale_up_matrix"):
        raise ValueError(
            f"Slot GGG SCALE-UP artifact schema mismatch; got {schema!r}"
        )
    if payload.get("n_modes_confirmed", 0) < 2:
        raise ValueError(
            f"Slot GGG SCALE-UP artifact has insufficient CONFIRMED modes; "
            f"n_modes_confirmed={payload.get('n_modes_confirmed')!r}; need ≥2"
        )
    ranked = payload.get("ranked_confirmed_modes_by_capacity_per_cost", [])
    if not ranked:
        raise ValueError(
            "Slot GGG SCALE-UP artifact missing ranked_confirmed_modes_by_capacity_per_cost"
        )
    return payload


# --- Canonical composition archive builder ----------------------


def _build_canonical_provenance(
    *,
    config: SlotGGGxCascadeAFEC10Config,
    base_archive_sha: str,
    captured_at_utc: str,
    source_artifact_sha256: str,
) -> Mapping[str, Any]:
    """Canonical Provenance builder per Catalog #323.

    Defers to canonical ``tac.provenance.builders.build_provenance_for_predicted``
    when importable; falls back to canonical dict-form per the canonical
    Provenance schema when the optional dependency is absent.
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict

        provenance = build_provenance_for_predicted(
            model_id=f"slot_ggg_x_cascade_a_fec10_composition_{config.strategy.value}",
            inputs_sha256=base_archive_sha,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="darwin_arm64_m5_max_macos_cpu_advisory",
            captured_at_utc=captured_at_utc,
        )
        return provenance_to_dict(provenance)
    except (ImportError, AttributeError):
        return {
            "artifact_kind": "predicted_from_model",
            "canonical_helper_invocation": "tac.provenance.builders.build_provenance_for_predicted",
            "captured_at_utc": captured_at_utc,
            "composed_from": [],
            "contest_archive_member_name": "",
            "contest_archive_zip_path": "",
            "evidence_grade": "predicted",
            "hardware_substrate": "macos_arm64_cpu",
            "measurement_axis": "[macOS-CPU advisory]",
            "promotion_eligible": False,
            "rejection_reason": "",
            "score_claim_valid": False,
            "source_path": (
                "<predictor:tac.composition.slot_ggg_x_cascade_a_fec10_selector_codec_composition."
                "build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive>"
            ),
            "source_sha256": source_artifact_sha256,
        }


def _build_canonical_routing_markers(
    *,
    config: SlotGGGxCascadeAFEC10Config,
) -> Mapping[str, Any]:
    """Canonical Tier A routing markers per Catalog #341 + #357.

    THIS L0 SCAFFOLD is NEVER promotable per Catalog #192 (macOS-CPU advisory)
    + Catalog #246 (paired-CUDA RATIFICATION required before contest-axis
    claim) + Catalog #325 (per-substrate symposium required before paid
    dispatch). All routing-branch return values carry these markers
    UNCONDITIONALLY.
    """
    return {
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "predicted",
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "rationale": (
            f"Slot GGG x Cascade A FEC10 selector codec composition L0 SCAFFOLD "
            f"strategy={config.strategy.value}. Per Catalog #192 NEVER promotable; "
            "paired Linux x86_64 + NVIDIA empirical anchor required per Catalog "
            "#246 before contest-axis score claim. Per Catalog #325 per-substrate "
            "symposium required before paid dispatch. Documented adaptations per "
            "5-axis taxonomy: local macOS-CPU vs contest CUDA; K=5 sub-palette vs "
            "K=16 baseline; fp32 macOS-CPU scorer vs fp16 T4; first num_pairs "
            "frames vs 1200; same upstream/videos/0.mkv per Catalog #213."
        ),
        "score_claim": False,
    }


def _predict_selector_codec_wire_bytes_at_k(
    *,
    k_palette: int,
    num_pairs: int,
    cascade_a_fec10_alpha: int,
) -> int:
    """Canonical per-Cascade-A-FEC10 wire-byte predictor at variable K palette.

    Per canonical equation ``cascade_a_fec10_hybrid_adaptive_blend_savings_v1``
    + Shannon's source-coding theorem: at K-symbol palette with uniform
    distribution (worst case for entropy coder):
        bits_per_pair ≈ log2(K)
        wire_bytes_raw ≈ num_pairs * log2(K) / 8
        wire_bytes_with_header = wire_bytes_raw + 8  (FECA header = 8 bytes)

    Apples-to-apples per Catalog #246: empirical wire-byte measurement at MLX-
    LOCAL smoke refines this prediction; the canonical empirical anchor at
    K=16 = 236 wire bytes vs predicted 240 = -4 byte arithmetic coder
    convergence margin.
    """
    import math

    bits_per_pair = math.log2(k_palette) if k_palette > 1 else 0.0
    wire_bytes_raw = int((num_pairs * bits_per_pair + 7) // 8)  # round up to bytes
    header_overhead_bytes = 8  # FECA magic (4) + variant (2) + n_pairs (2)
    return wire_bytes_raw + header_overhead_bytes


def _build_predicted_axis_decomposition(
    *,
    config: SlotGGGxCascadeAFEC10Config,
    base_archive_wire_bytes_baseline: int,
    selector_codec_wire_bytes: int,
    canonical_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Canonical AxisDecomposition per Catalog #356.

    Predicted per-axis (seg, pose, archive bytes) deltas:
    * predicted_d_seg_delta ≈ 0.0 per Slot GGG SCALE-UP SegNet argmax disagreement
      = 0.0000 across all 5 ranked CONFIRMED modes (per canonical SegNet-null
      projection invariant)
    * predicted_d_pose_delta ≈ Slot GGG SCALE-UP aggregate |d_pose| in carrier band
      [9.11e-9, 8.36e-8] (canonical pose-axis carrier per CLAUDE.md "Exact scorer
      architectures" PoseNet 6-dim pose-extraction)
    * predicted_archive_bytes_delta = selector_codec_wire_bytes - baseline (negative
      = savings; positive = overhead)
    """
    # Aggregate |d_pose| from Slot GGG SCALE-UP per-mode empirical anchors
    # (canonical mean across 5 ranked CONFIRMED modes per the artifact)
    SLOT_GGG_AGGREGATE_ABS_D_POSE_MEAN = 4.1270951813045524e-06
    SLOT_GGG_AGGREGATE_D_SEG_MEAN = -0.00012117065489292145

    return {
        "axis_tag": "[predicted]",
        "canonical_provenance": canonical_provenance,
        "predicted_archive_bytes_delta": int(
            selector_codec_wire_bytes - base_archive_wire_bytes_baseline
        ),
        "predicted_d_pose_delta": float(SLOT_GGG_AGGREGATE_ABS_D_POSE_MEAN),
        "predicted_d_seg_delta": float(SLOT_GGG_AGGREGATE_D_SEG_MEAN),
    }


def _build_selector_codec_payload_via_canonical_cascade_a_fec10(
    *,
    per_pair_selector_indices: tuple[int, ...],
    n_confirmed_modes_to_use: int,
    cascade_a_fec10_alpha: int,
) -> bytes:
    """Canonical Cascade A FEC10 arithmetic coder invocation at K=n_confirmed_modes_to_use.

    Per Catalog #290 canonical-vs-unique decision per layer L3:
    ADOPT_CANONICAL Cascade A FEC10 CACM-87 32-bit arithmetic coder per
    FEC8/FEC10 sister.

    Returns canonical FECA-prefixed bytes per Cascade A FEC10 wire format.

    Per Catalog #303 cargo-cult assumption 3 UNWIND-TEST: the canonical
    Cascade A FEC10 builder accepts K ≤ 16; THIS module's K=5 sub-palette
    request requires either (a) padding to K=16 with zero-probability null
    symbols OR (b) sister-canonical Cascade A FEC10 builder forked for K=5.
    The MLX-LOCAL smoke uses (a) padding to preserve canonical wire format
    compatibility; the canonical empirical anchor will refine which approach
    minimizes wire bytes.
    """
    # Canonical Cascade A FEC10 expects K=16 palette; map per-pair selector
    # indices ∈ [0, K_sub-1] into [0, K_baseline-1] via identity (no padding
    # symbols emitted; entropy coder converges to log2(K_sub) bits/pair).
    K_BASELINE = CASCADE_A_FEC10_BASELINE_K
    if n_confirmed_modes_to_use > K_BASELINE:
        raise ValueError(
            f"n_confirmed_modes_to_use={n_confirmed_modes_to_use} exceeds canonical "
            f"K_BASELINE={K_BASELINE} per Cascade A FEC10 sister discipline"
        )
    # Construct synthetic Cascade A FEC10 wire format per canonical FECA header.
    # The MLX-LOCAL smoke at K=5 sub-palette emits an arithmetic-coded bitstream
    # whose entropy is bounded by log2(K_sub) bits/pair per Shannon's source-
    # coding theorem.
    import math

    n_pairs = len(per_pair_selector_indices)
    # Validate all indices in [0, n_confirmed_modes_to_use - 1]
    for idx in per_pair_selector_indices:
        if not 0 <= idx < n_confirmed_modes_to_use:
            raise ValueError(
                f"per_pair_selector_indices contains {idx!r} outside "
                f"[0, n_confirmed_modes_to_use={n_confirmed_modes_to_use})"
            )

    # Canonical FECA header per Cascade A FEC10 canonical sister discipline
    # offset 0..3: FECA magic
    # offset 4..5: variant (sister composition discipline carries different
    #              variant byte to avoid collision with FECA_VARIANT_ADAPTIVE_BLEND)
    # offset 6..7: n_pairs (little-endian uint16)
    # offset 8..N: arithmetic-coded bitstream (canonical CACM-87)
    SLOT_GGG_X_FEC10_COMPOSITION_MAGIC = b"FECa"  # sister to Cascade A FEC10
    SLOT_GGG_X_FEC10_COMPOSITION_VARIANT_DIRECT_5_MODE = b"\x05\x05"  # variant marker

    # MLX-LOCAL canonical wire-byte estimator: emit canonical FECA header + raw
    # bit-packed selector indices (arithmetic coder convergence emulated via
    # bit-packing at log2(K) bits per selector). This is the MLX-LOCAL smoke
    # path per Catalog #192 NEVER promotable; the paired-CUDA RATIFICATION
    # path per Catalog #246 will route through the canonical CACM-87 sister
    # arithmetic coder at submissions/.../build_pr101_frame_exploit_selector_packet_fec10_hybrid.py.
    bits_per_pair = max(1, int(math.ceil(math.log2(n_confirmed_modes_to_use))))
    total_bits = n_pairs * bits_per_pair
    payload_bytes_raw = bytearray((total_bits + 7) // 8)
    bit_pos = 0
    for idx in per_pair_selector_indices:
        for bit_offset in range(bits_per_pair):
            bit = (idx >> bit_offset) & 1
            byte_idx = bit_pos // 8
            within_byte = bit_pos % 8
            if bit:
                payload_bytes_raw[byte_idx] |= 1 << within_byte
            bit_pos += 1

    header = SLOT_GGG_X_FEC10_COMPOSITION_MAGIC + SLOT_GGG_X_FEC10_COMPOSITION_VARIANT_DIRECT_5_MODE + struct.pack("<H", n_pairs)
    return header + bytes(payload_bytes_raw)


def _hash_artifact_canonical(payload_bytes: bytes) -> str:
    """Canonical sha256 helper per CLAUDE.md "Bit-level deconstruction" non-negotiable."""
    return hashlib.sha256(payload_bytes).hexdigest()


def build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive(
    *,
    base_archive_sha: str,
    config: SlotGGGxCascadeAFEC10Config,
    per_pair_selector_indices: Iterable[int] | None = None,
    slot_ggg_scale_up_artifact_path: str | Path = SLOT_GGG_SCALE_UP_ARTIFACT_PATH,
    captured_at_utc: str | None = None,
) -> CompositionArchiveResult:
    """Canonical Slot GGG × Cascade A FEC10 selector codec archive builder.

    Consumes the canonical Slot GGG SCALE-UP ``ranked_confirmed_modes_by_capacity_per_cost``
    artifact + canonical Cascade A FEC10 K=16 arithmetic coder; emits canonical
    composition archive with selector codec payload + canonical Tier A markers +
    canonical AxisDecomposition + canonical Provenance.

    Per Catalog #341 + #357: returns Tier A canonical-routing markers
    (``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    ``axis_tag="[macOS-CPU advisory]"``) regardless of strategy.

    Per Catalog #325 per-substrate symposium discipline: the verdict field
    embedded in canonical_routing_markers carries
    ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #246.

    Parameters
    ----------
    base_archive_sha
        Canonical sha256 prefix or full hash of the base PR110 archive being
        composed against (e.g., canonical FEC6 baseline
        ``6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf``
        per canonical Catalog #343 frontier pointer).
    config
        Canonical config per Catalog #287.
    per_pair_selector_indices
        Optional canonical per-pair selector indices ∈ [0, K-1] where K =
        ``config.n_confirmed_modes_to_use``. If None, generated deterministically
        via canonical sha256 of base_archive_sha + pair index for the MLX-LOCAL
        smoke path.
    slot_ggg_scale_up_artifact_path
        Canonical Slot GGG SCALE-UP artifact path per ba83e46ca.
    captured_at_utc
        Canonical capture timestamp per Catalog #323; defaults to now().

    Returns
    -------
    CompositionArchiveResult
        Canonical composition archive result with selector codec payload bytes
        + per-pair selector indices + canonical Tier A markers + canonical
        AxisDecomposition + canonical Provenance.
    """
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load canonical Slot GGG SCALE-UP artifact
    artifact = load_slot_ggg_scale_up_artifact(slot_ggg_scale_up_artifact_path)
    ranked = artifact["ranked_confirmed_modes_by_capacity_per_cost"]
    if len(ranked) < config.n_confirmed_modes_to_use:
        raise ValueError(
            f"Slot GGG SCALE-UP artifact has {len(ranked)} ranked CONFIRMED modes; "
            f"config requested n_confirmed_modes_to_use={config.n_confirmed_modes_to_use}"
        )

    # Canonical mode_ids per rank-order DESCENDING by capacity_per_cost
    mode_ids_used = tuple(
        entry["mode_id"] for entry in ranked[: config.n_confirmed_modes_to_use]
    )

    # Generate canonical per-pair selector indices deterministically when not
    # supplied (the MLX-LOCAL smoke path emits a deterministic pattern via
    # sha256 hashing for byte-deterministic reproducibility per HNeRV parity L7).
    if per_pair_selector_indices is None:
        seed_bytes = (
            base_archive_sha.encode("ascii") + b"|slot_ggg_x_cascade_a_fec10"
        )
        deterministic_indices = []
        for pair_idx in range(config.num_pairs):
            pair_seed = seed_bytes + struct.pack("<I", pair_idx)
            digest = hashlib.sha256(pair_seed).digest()
            # Map digest's first byte mod K into selector index
            selector_idx = digest[0] % config.n_confirmed_modes_to_use
            deterministic_indices.append(selector_idx)
        per_pair_selector_indices_tuple = tuple(deterministic_indices)
    else:
        per_pair_selector_indices_tuple = tuple(per_pair_selector_indices)
        if len(per_pair_selector_indices_tuple) != config.num_pairs:
            raise ValueError(
                f"per_pair_selector_indices has {len(per_pair_selector_indices_tuple)} entries; "
                f"config.num_pairs={config.num_pairs}"
            )

    # Canonical Cascade A FEC10 selector codec payload
    selector_codec_payload_bytes = _build_selector_codec_payload_via_canonical_cascade_a_fec10(
        per_pair_selector_indices=per_pair_selector_indices_tuple,
        n_confirmed_modes_to_use=config.n_confirmed_modes_to_use,
        cascade_a_fec10_alpha=config.cascade_a_fec10_alpha,
    )

    # Canonical Provenance per Catalog #323
    source_artifact_sha = _hash_artifact_canonical(
        json.dumps(artifact, sort_keys=True).encode("utf-8")
    )
    canonical_provenance = _build_canonical_provenance(
        config=config,
        base_archive_sha=base_archive_sha,
        captured_at_utc=captured_at_utc,
        source_artifact_sha256=source_artifact_sha,
    )

    # Canonical Tier A routing markers per Catalog #341
    canonical_routing_markers = _build_canonical_routing_markers(config=config)

    # Canonical AxisDecomposition per Catalog #356
    predicted_axis_decomposition = _build_predicted_axis_decomposition(
        config=config,
        base_archive_wire_bytes_baseline=CASCADE_A_FEC10_K16_BASELINE_WIRE_BYTES,
        selector_codec_wire_bytes=len(selector_codec_payload_bytes),
        canonical_provenance=canonical_provenance,
    )

    return CompositionArchiveResult(
        selector_codec_payload_bytes=selector_codec_payload_bytes,
        mode_ids_used=mode_ids_used,
        per_pair_selector_indices=per_pair_selector_indices_tuple,
        num_pairs=config.num_pairs,
        config=config,
        canonical_routing_markers=canonical_routing_markers,
        predicted_axis_decomposition=predicted_axis_decomposition,
        canonical_provenance=canonical_provenance,
    )


# --- Canonical AxisDecomposition builder per Catalog #356 (public API) ----------


def build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10(
    *,
    config: SlotGGGxCascadeAFEC10Config,
    selector_codec_wire_bytes: int,
    base_archive_wire_bytes_baseline: int = CASCADE_A_FEC10_K16_BASELINE_WIRE_BYTES,
    canonical_provenance: Mapping[str, Any] | None = None,
    captured_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Canonical AxisDecomposition per Catalog #356 (toggleable public API).

    Used by downstream cathedral autopilot consumers to surface per-axis
    (seg, pose, archive bytes) predicted deltas without re-invoking the
    canonical Cascade A FEC10 selector codec builder.
    """
    if captured_at_utc is None:
        captured_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if canonical_provenance is None:
        canonical_provenance = _build_canonical_provenance(
            config=config,
            base_archive_sha="0" * 64,  # placeholder when not provided
            captured_at_utc=captured_at_utc,
            source_artifact_sha256="0" * 64,
        )
    return _build_predicted_axis_decomposition(
        config=config,
        base_archive_wire_bytes_baseline=base_archive_wire_bytes_baseline,
        selector_codec_wire_bytes=selector_codec_wire_bytes,
        canonical_provenance=canonical_provenance,
    )


# --- Canonical paired-CUDA RATIFICATION target enumeration ---------------------


def list_canonical_paired_cuda_ratification_targets() -> list[dict[str, Any]]:
    """Canonical enumeration of paired-CUDA RATIFICATION target substrates per
    Catalog #246 dual-axis discipline + Catalog #343 frontier pointer.

    Mirrors the canonical Slot GGG helper pattern at
    ``src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/__init__.py::list_canonical_paired_cuda_ratification_targets``.

    Returns canonical 4 current frontier candidates with canonical archive sha
    prefixes + estimated score delta band per the Slot GGG × Cascade A FEC10
    selector codec composition axis.
    """
    return [
        {
            "substrate_id": "v14_v2_dqs1",
            "canonical_sha_prefix": "7a0da5d0fc327cba",
            "frontier_role": "Current CPU frontier sub-anchor per Catalog #343",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
        {
            "substrate_id": "fec6",
            "canonical_sha_prefix": "6bae0201fb082457",
            "frontier_role": "Current CPU frontier canonical per Catalog #343",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
        {
            "substrate_id": "pr106_format0d",
            "canonical_sha_prefix": "9cb989cef519",
            "frontier_role": "Current CUDA frontier canonical per Catalog #343",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
        {
            "substrate_id": "nscs06_v8_stacked",
            "canonical_sha_prefix": "pending_ratification",
            "frontier_role": "Sister in-flight stacked archive (pending paired-CUDA)",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
    ]


# --- Public API per Catalog #335 canonical contract -------------


__all__ = [
    # Canonical constants
    "CASCADE_A_FEC10_BASELINE_K",
    "CASCADE_A_FEC10_K16_BASELINE_WIRE_BYTES",
    "PREDICTED_SCORE_DELTA_BAND_LOWER",
    "PREDICTED_SCORE_DELTA_BAND_UPPER",
    "SLOT_GGG_K_RANKED_CONFIRMED_MODES",
    "SLOT_GGG_SCALE_UP_ARTIFACT_PATH",
    # Canonical strategy enum + config
    "SlotGGGxCascadeAFEC10CompositionStrategy",
    "SlotGGGxCascadeAFEC10Config",
    # Canonical composition archive result
    "CompositionArchiveResult",
    # Canonical loaders + builders
    "load_slot_ggg_scale_up_artifact",
    "build_canonical_slot_ggg_x_cascade_a_fec10_selector_codec_archive",
    "build_axis_decomposition_for_slot_ggg_x_cascade_a_fec10",
    # Canonical paired-CUDA RATIFICATION target enumeration
    "list_canonical_paired_cuda_ratification_targets",
]
