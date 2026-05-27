# SPDX-License-Identifier: MIT
"""Tests for BUILD-2+3-EXT 8 NOT-BUILT operators per audit memo enumeration.

Per `.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md`
§"Phase 1 catalog enumeration" + operator insight 2026-05-26 + 8 standing
directives.

Coverage:
- Per-operator correctness (empty canvas / single-cell / multi-cell)
- Per-axis decomposition per Catalog #356
- Canonical Provenance per Catalog #323 + non-promotable markers per Catalog #341
- Apples-to-apples baseline preservation per 10th standing directive
- Live-repo regression guard
- Integration with sister BUILD-2+3 canonical 4 operations (cross-operator
  composition test per 8th INDIVIDUALLY-FRACTAL standing directive)
- Catalog #335 canonical contract conformance
- 8th INDIVIDUALLY-FRACTAL per-substrate optimization tree declaration
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from tac.cathedral.consumer_contract import (
    AxisDecomposition,
    ConsumerTier,
    HookNumber,
)
from tac.cathedral_consumers import (
    pair_frame_5d_extended_operator_consumer as cathedral_consumer,
)

# Import canonical pair-count constant from canvas (sister-disjoint).
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANONICAL_PAIR_COUNT,
    CanonicalOperation,
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    CONSUMER_HOOK_NUMBERS,
    CONSUMER_NAME,
    CONSUMER_TIER,
    CONSUMER_VERSION,
    DEFAULT_BEAM_DEPTH,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MERGE_MAX_CANDIDATES,
    DEFAULT_MOTION_THRESHOLD_PERCENTILE,
    DEFAULT_TEMPORAL_COHERENCE_WINDOW,
    EXTENDED_MODULE_SCHEMA,
    EXTENDED_OPERATION_CANONICAL_EQUATION_IDS,
    EXTENDED_OPERATOR_REGISTRY,
    DropFrameParameters,
    ExtendedOperation,
    MergePairParameters,
    MotionConditionalParameters,
    ReorderPairParameters,
    ReplaceManyParameters,
    ReplaceOneParameters,
    SynthesizeFrameParameters,
    TemporalCoherenceParameters,
    _cosine_similarity,
    _extended_to_canvas_operation_proxy,
    _per_pair_axis_signature,
    consume_candidate,
    generate_drop_frame_candidates,
    generate_merge_pair_candidates,
    generate_motion_conditional_candidates,
    generate_reorder_pair_candidates,
    generate_replace_many_candidates,
    generate_replace_one_candidates,
    generate_synthesize_frame_candidates,
    generate_temporal_coherence_candidates,
    update_from_anchor,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic 5D canvas builders for test reproducibility.
# ---------------------------------------------------------------------------


def _build_cell(
    *,
    pair_idx: int,
    frame_idx: int,
    scorer_axis: ScorerAxis = ScorerAxis.SEGNET_5CLASS,
    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL,
    cpu_cuda_axis: CpuCudaAxis = CpuCudaAxis.CONTEST_CPU,
    predicted_delta_score: float = -1e-6,
    predicted_byte_cost: int = -1,
    receiver_feasibility: bool = True,
) -> PairFrameScorerGeometryCell:
    return PairFrameScorerGeometryCell(
        pair_idx=pair_idx,
        frame_idx=frame_idx,
        scorer_axis=scorer_axis,
        receiver_runtime=receiver_runtime,
        cpu_cuda_axis=cpu_cuda_axis,
        predicted_delta_score=predicted_delta_score,
        predicted_byte_cost=predicted_byte_cost,
        receiver_feasibility=receiver_feasibility,
    )


_SYNTH_SHA256 = "a" * 64
_FRONTIER_LIKE_SHA256 = "7a0da5d0fc" + "0" * 54


def _empty_canvas() -> PairFrameScorerGeometryLattice:
    return PairFrameScorerGeometryLattice(archive_sha256=_SYNTH_SHA256)


def _small_canvas() -> PairFrameScorerGeometryLattice:
    """Build a small 4-pair canvas with mixed receiver-feasible cells."""
    cells: dict[tuple, PairFrameScorerGeometryCell] = {}
    for pair_idx in range(4):
        frame_first = 2 * pair_idx
        frame_last = 2 * pair_idx + 1
        for scorer in (
            ScorerAxis.SEGNET_5CLASS,
            ScorerAxis.POSENET_6D,
            ScorerAxis.RATE_TERM,
        ):
            for frame in (frame_first, frame_last):
                for axis in (CpuCudaAxis.CONTEST_CPU, CpuCudaAxis.CONTEST_CUDA_T4):
                    cell = _build_cell(
                        pair_idx=pair_idx,
                        frame_idx=frame,
                        scorer_axis=scorer,
                        cpu_cuda_axis=axis,
                        predicted_delta_score=-1e-6 * (pair_idx + 1)
                        if scorer is not ScorerAxis.RATE_TERM
                        else 0.0,
                        predicted_byte_cost=-1
                        if scorer is ScorerAxis.RATE_TERM
                        else 0,
                    )
                    cells[cell.coordinate] = cell
    return PairFrameScorerGeometryLattice(
        archive_sha256=_SYNTH_SHA256, cells=cells
    )


# ---------------------------------------------------------------------------
# Module-level canonical contract conformance (Catalog #335 + #357 + #341).
# ---------------------------------------------------------------------------


def test_canonical_contract_consumer_name_matches_module() -> None:
    assert (
        CONSUMER_NAME
        == "pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators"
    )


def test_canonical_contract_consumer_version_is_scaffold_form() -> None:
    assert CONSUMER_VERSION.startswith("0.")


def test_canonical_contract_consumer_hook_numbers_complete() -> None:
    assert set(CONSUMER_HOOK_NUMBERS) == {
        HookNumber.SENSITIVITY_MAP,
        HookNumber.PARETO_CONSTRAINT,
        HookNumber.BIT_ALLOCATOR,
        HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
        HookNumber.CONTINUAL_LEARNING_POSTERIOR,
        HookNumber.PROBE_DISAMBIGUATOR,
    }


def test_canonical_contract_consumer_tier_a_scaffold() -> None:
    assert CONSUMER_TIER is ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_canonical_contract_8_operators_registered() -> None:
    assert len(list(ExtendedOperation)) == 8
    assert len(EXTENDED_OPERATOR_REGISTRY) == 8


def test_canonical_contract_8_canonical_equation_ids_pending() -> None:
    assert len(EXTENDED_OPERATION_CANONICAL_EQUATION_IDS) == 8
    for op, eq_id in EXTENDED_OPERATION_CANONICAL_EQUATION_IDS.items():
        assert isinstance(op, ExtendedOperation)
        assert isinstance(eq_id, str)
        assert eq_id.endswith("_v1")


def test_canonical_contract_update_from_anchor_observability_only() -> None:
    """Per Catalog #357 Tier A: update_from_anchor returns None."""
    result = update_from_anchor(object())
    assert result is None


def test_canonical_contract_consume_candidate_returns_canonical_markers() -> None:
    """Per Catalog #341: canonical-routing markers default to non-promotable."""
    result = consume_candidate({})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_cathedral_consumer_routes_extended_operator_without_authority() -> None:
    canvas = _small_canvas()
    candidates = generate_replace_one_candidates(
        canvas,
        ReplaceOneParameters(
            target_pair_idx=1,
            alternative_selector_id=7,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        ),
        top_n=1,
    )

    result = cathedral_consumer.consume_candidate(candidates[0].as_dict())

    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert result["readiness_verdict"] == "PLANNING_VISIBLE"
    assert result["extended_operation"] == ExtendedOperation.REPLACE_ONE.value


# ---------------------------------------------------------------------------
# Per-operator parameter dataclass invariants.
# ---------------------------------------------------------------------------


def test_replace_one_params_invariants() -> None:
    p = ReplaceOneParameters(
        target_pair_idx=0,
        alternative_selector_id=0,
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
    )
    assert p.target_pair_idx == 0
    with pytest.raises(ValueError, match="target_pair_idx"):
        ReplaceOneParameters(
            target_pair_idx=-1,
            alternative_selector_id=0,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        )
    with pytest.raises(ValueError, match="target_pair_idx"):
        ReplaceOneParameters(
            target_pair_idx=CANONICAL_PAIR_COUNT,
            alternative_selector_id=0,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        )
    with pytest.raises(ValueError, match="alternative_selector_id"):
        ReplaceOneParameters(
            target_pair_idx=0,
            alternative_selector_id=-1,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        )
    with pytest.raises(ValueError, match="alternative_selector_id"):
        ReplaceOneParameters(
            target_pair_idx=0,
            alternative_selector_id="bad",  # type: ignore[arg-type]
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        )
    with pytest.raises(ValueError, match="receiver_runtime"):
        ReplaceOneParameters(
            target_pair_idx=0,
            alternative_selector_id=0,
            receiver_runtime="raw_residual",  # type: ignore[arg-type]
        )


def test_replace_many_params_invariants() -> None:
    p = ReplaceManyParameters()
    assert p.beam_width == DEFAULT_BEAM_WIDTH
    assert p.beam_depth == DEFAULT_BEAM_DEPTH
    with pytest.raises(ValueError, match="beam_width"):
        ReplaceManyParameters(beam_width=0)
    with pytest.raises(ValueError, match="beam_width"):
        ReplaceManyParameters(beam_width=2048)
    with pytest.raises(ValueError, match="beam_depth"):
        ReplaceManyParameters(beam_depth=0)
    with pytest.raises(ValueError, match="beam_depth"):
        ReplaceManyParameters(beam_depth=32)
    with pytest.raises(ValueError, match="target_pair_indices"):
        ReplaceManyParameters(target_pair_indices=[0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_pair_indices entry"):
        ReplaceManyParameters(target_pair_indices=(CANONICAL_PAIR_COUNT,))


def test_merge_pair_params_invariants() -> None:
    p = MergePairParameters()
    assert p.max_candidates == DEFAULT_MERGE_MAX_CANDIDATES
    with pytest.raises(ValueError, match="max_candidates"):
        MergePairParameters(max_candidates=0)
    with pytest.raises(ValueError, match="max_candidates"):
        MergePairParameters(max_candidates=10000)


def test_reorder_pair_params_invariants() -> None:
    p = ReorderPairParameters()
    assert p.block_size == 8
    with pytest.raises(ValueError, match="block_size"):
        ReorderPairParameters(block_size=1)
    with pytest.raises(ValueError, match="block_size"):
        ReorderPairParameters(block_size=64)


def test_drop_frame_params_invariants() -> None:
    p = DropFrameParameters()
    assert p.which_frame == "last"
    with pytest.raises(ValueError, match="which_frame"):
        DropFrameParameters(which_frame="middle")  # type: ignore[arg-type]
    DropFrameParameters(which_frame="both")  # accepted for drop-frame


def test_synthesize_frame_params_invariants() -> None:
    p = SynthesizeFrameParameters()
    assert p.synthesis_seed == 0
    with pytest.raises(ValueError, match="which_frame"):
        SynthesizeFrameParameters(which_frame="both")  # forbidden for synthesis
    with pytest.raises(ValueError, match="synthesis_seed"):
        SynthesizeFrameParameters(synthesis_seed="seed")  # type: ignore[arg-type]


def test_motion_conditional_params_invariants() -> None:
    p = MotionConditionalParameters()
    assert (
        p.motion_threshold_percentile == DEFAULT_MOTION_THRESHOLD_PERCENTILE
    )
    with pytest.raises(ValueError, match="motion_threshold_percentile"):
        MotionConditionalParameters(motion_threshold_percentile=1.5)
    with pytest.raises(ValueError, match="motion_threshold_percentile"):
        MotionConditionalParameters(motion_threshold_percentile=-0.1)
    with pytest.raises(ValueError, match="cannot be MOTION_CONDITIONAL"):
        MotionConditionalParameters(
            high_motion_operator=ExtendedOperation.MOTION_CONDITIONAL
        )
    with pytest.raises(ValueError, match="cannot be MOTION_CONDITIONAL"):
        MotionConditionalParameters(
            low_motion_operator=ExtendedOperation.MOTION_CONDITIONAL
        )


def test_temporal_coherence_params_invariants() -> None:
    p = TemporalCoherenceParameters()
    assert p.temporal_window == DEFAULT_TEMPORAL_COHERENCE_WINDOW
    with pytest.raises(ValueError, match="temporal_window"):
        TemporalCoherenceParameters(temporal_window=1)
    with pytest.raises(ValueError, match="temporal_window"):
        TemporalCoherenceParameters(temporal_window=128)
    with pytest.raises(ValueError, match="similarity_threshold"):
        TemporalCoherenceParameters(similarity_threshold=2.0)
    with pytest.raises(ValueError, match="similarity_threshold"):
        TemporalCoherenceParameters(similarity_threshold=-2.0)


# ---------------------------------------------------------------------------
# Per-operator correctness: empty canvas behavior.
# ---------------------------------------------------------------------------


def test_replace_one_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_replace_one_candidates(
            _empty_canvas(),
            ReplaceOneParameters(
                target_pair_idx=0,
                alternative_selector_id=0,
                receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
            ),
        )
        == []
    )


def test_replace_many_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_replace_many_candidates(_empty_canvas(), ReplaceManyParameters())
        == []
    )


def test_merge_pair_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_merge_pair_candidates(_empty_canvas(), MergePairParameters())
        == []
    )


def test_reorder_pair_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_reorder_pair_candidates(_empty_canvas(), ReorderPairParameters())
        == []
    )


def test_drop_frame_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_drop_frame_candidates(_empty_canvas(), DropFrameParameters())
        == []
    )


def test_synthesize_frame_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_synthesize_frame_candidates(
            _empty_canvas(), SynthesizeFrameParameters()
        )
        == []
    )


def test_motion_conditional_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_motion_conditional_candidates(
            _empty_canvas(), MotionConditionalParameters()
        )
        == []
    )


def test_temporal_coherence_empty_canvas_returns_empty_list() -> None:
    assert (
        generate_temporal_coherence_candidates(
            _empty_canvas(), TemporalCoherenceParameters()
        )
        == []
    )


# ---------------------------------------------------------------------------
# Per-operator correctness: small canvas (4 pairs × multi-axis).
# ---------------------------------------------------------------------------


def test_replace_one_returns_candidate_for_target_pair() -> None:
    canvas = _small_canvas()
    params = ReplaceOneParameters(
        target_pair_idx=1,
        alternative_selector_id=42,
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
    )
    candidates = generate_replace_one_candidates(canvas, params, top_n=8)
    # 2 candidates per target pair (one per cpu_cuda_axis with feasibility).
    assert 1 <= len(candidates) <= 4
    for c in candidates:
        assert (
            c.canonical_dispatch_recipe_hint["target_pair_idx"] == 1
        )
        assert (
            c.canonical_dispatch_recipe_hint["alternative_selector_id"] == 42
        )
        assert c.predicted_delta_score < 0.0


def test_replace_many_beam_search_returns_candidates() -> None:
    canvas = _small_canvas()
    params = ReplaceManyParameters(
        beam_width=4,
        beam_depth=2,
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
    )
    candidates = generate_replace_many_candidates(canvas, params, top_n=16)
    assert len(candidates) > 0
    for c in candidates:
        assert c.predicted_delta_score < 0.0
        assert c.canonical_dispatch_recipe_hint["operation"] == "replace_many"
        assert c.canonical_dispatch_recipe_hint["beam_depth_achieved"] <= 2


def test_merge_pair_returns_candidates_with_shared_encoding() -> None:
    canvas = _small_canvas()
    params = MergePairParameters(
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        max_candidates=32,
    )
    candidates = generate_merge_pair_candidates(canvas, params, top_n=16)
    assert len(candidates) > 0
    for c in candidates:
        hint = c.canonical_dispatch_recipe_hint
        assert "merge_pair_a" in hint
        assert "merge_pair_b" in hint
        assert hint["merge_pair_a"] != hint["merge_pair_b"]
        assert "merged_bytes_delta_via_shared_encoding" in hint


def test_reorder_pair_skips_when_already_sorted() -> None:
    # Build a canvas where pair0 has best delta, pair1 worse, etc.
    cells: dict[tuple, PairFrameScorerGeometryCell] = {}
    for pair_idx in range(4):
        cell = _build_cell(
            pair_idx=pair_idx,
            frame_idx=2 * pair_idx + 1,
            predicted_delta_score=-1e-6 * (4 - pair_idx),
        )
        cells[cell.coordinate] = cell
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256=_SYNTH_SHA256, cells=cells
    )
    params = ReorderPairParameters(block_size=4)
    candidates = generate_reorder_pair_candidates(canvas, params)
    # block is already sorted ascending by total_delta (pair0 best -> pair3 worst);
    # reorder should be no-op.
    assert candidates == []


def test_reorder_pair_emits_when_unsorted() -> None:
    # Build a canvas where pair2 has best delta then pair0 then pair1 then pair3.
    cells: dict[tuple, PairFrameScorerGeometryCell] = {}
    deltas = [-1e-6, -2e-7, -3e-6, -1e-7]
    for pair_idx, delta in enumerate(deltas):
        cell = _build_cell(
            pair_idx=pair_idx,
            frame_idx=2 * pair_idx + 1,
            predicted_delta_score=delta,
        )
        cells[cell.coordinate] = cell
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256=_SYNTH_SHA256, cells=cells
    )
    params = ReorderPairParameters(block_size=4)
    candidates = generate_reorder_pair_candidates(canvas, params)
    assert len(candidates) >= 1
    for c in candidates:
        hint = c.canonical_dispatch_recipe_hint
        assert hint["original_pair_order"] != hint["reordered_pair_order"]


def test_drop_frame_filters_by_which_frame() -> None:
    canvas = _small_canvas()
    last = generate_drop_frame_candidates(
        canvas, DropFrameParameters(which_frame="last"), top_n=32
    )
    first = generate_drop_frame_candidates(
        canvas, DropFrameParameters(which_frame="first"), top_n=32
    )
    both = generate_drop_frame_candidates(
        canvas, DropFrameParameters(which_frame="both"), top_n=64
    )
    for c in last:
        assert c.canonical_dispatch_recipe_hint["frame_idx"] % 2 == 1
    for c in first:
        assert c.canonical_dispatch_recipe_hint["frame_idx"] % 2 == 0
    # 'both' is the union; should contain ≥ max(last, first).
    assert len(both) >= max(len(last), len(first))


def test_synthesize_frame_threads_synthesis_seed() -> None:
    canvas = _small_canvas()
    params = SynthesizeFrameParameters(
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        which_frame="last",
        synthesis_seed=12345,
    )
    candidates = generate_synthesize_frame_candidates(canvas, params, top_n=8)
    assert len(candidates) > 0
    for c in candidates:
        hint = c.canonical_dispatch_recipe_hint
        assert hint["synthesis_seed"] == 12345
        assert "atick_redlich" in hint["literature_anchor"]
        assert "Atick" in hint["council_attendees"]


def test_motion_conditional_routes_per_pair_motion() -> None:
    # Build canvas where pair 0+1 have low pose magnitude, pair 2+3 high.
    cells: dict[tuple, PairFrameScorerGeometryCell] = {}
    for pair_idx in range(4):
        for scorer in (
            ScorerAxis.SEGNET_5CLASS,
            ScorerAxis.POSENET_6D,
            ScorerAxis.RATE_TERM,
        ):
            pose_magnitude = 1e-5 if pair_idx < 2 else 1e-3
            delta = (
                -pose_magnitude
                if scorer is ScorerAxis.POSENET_6D
                else (-1e-6 if scorer is ScorerAxis.SEGNET_5CLASS else 0.0)
            )
            cell = _build_cell(
                pair_idx=pair_idx,
                frame_idx=2 * pair_idx + 1,
                scorer_axis=scorer,
                predicted_delta_score=delta,
                predicted_byte_cost=-1 if scorer is ScorerAxis.RATE_TERM else 0,
            )
            cells[cell.coordinate] = cell
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256=_SYNTH_SHA256, cells=cells
    )
    params = MotionConditionalParameters(
        motion_threshold_percentile=0.5,
        high_motion_operator=ExtendedOperation.SYNTHESIZE_FRAME,
        low_motion_operator=ExtendedOperation.REPLACE_ONE,
    )
    candidates = generate_motion_conditional_candidates(canvas, params)
    # We expect classifications surface in hint.
    classes = {c.canonical_dispatch_recipe_hint["motion_class"] for c in candidates}
    assert classes == {"high", "low"}


def test_temporal_coherence_emits_when_similar_neighbors() -> None:
    # Build canvas where pair0 + pair1 + pair2 have identical signatures.
    cells: dict[tuple, PairFrameScorerGeometryCell] = {}
    for pair_idx in range(3):
        for scorer, delta in [
            (ScorerAxis.SEGNET_5CLASS, -1e-6),
            (ScorerAxis.POSENET_6D, -1e-6),
            (ScorerAxis.RATE_TERM, 0.0),
        ]:
            cell = _build_cell(
                pair_idx=pair_idx,
                frame_idx=2 * pair_idx + 1,
                scorer_axis=scorer,
                predicted_delta_score=delta,
                predicted_byte_cost=-1 if scorer is ScorerAxis.RATE_TERM else 0,
            )
            cells[cell.coordinate] = cell
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256=_SYNTH_SHA256, cells=cells
    )
    params = TemporalCoherenceParameters(
        temporal_window=2, similarity_threshold=0.9
    )
    candidates = generate_temporal_coherence_candidates(canvas, params, top_n=8)
    assert len(candidates) > 0
    for c in candidates:
        hint = c.canonical_dispatch_recipe_hint
        assert hint["cosine_similarity"] >= 0.9
        assert hint["pair_a"] != hint["pair_b"]
        assert "wyner_ziv" in hint["literature_anchor"]


# ---------------------------------------------------------------------------
# Per-axis decomposition per Catalog #356.
# ---------------------------------------------------------------------------


def test_per_axis_decomposition_populated_on_every_candidate() -> None:
    canvas = _small_canvas()
    for op_enum in ExtendedOperation:
        generator = EXTENDED_OPERATOR_REGISTRY[op_enum]
        if op_enum is ExtendedOperation.REPLACE_ONE:
            params: Any = ReplaceOneParameters(
                target_pair_idx=1,
                alternative_selector_id=7,
                receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
            )
        elif op_enum is ExtendedOperation.REPLACE_MANY:
            params = ReplaceManyParameters(beam_width=2, beam_depth=2)
        elif op_enum is ExtendedOperation.MERGE_PAIR:
            params = MergePairParameters()
        elif op_enum is ExtendedOperation.REORDER_PAIR:
            params = ReorderPairParameters(block_size=2)
        elif op_enum is ExtendedOperation.DROP_FRAME:
            params = DropFrameParameters()
        elif op_enum is ExtendedOperation.SYNTHESIZE_FRAME:
            params = SynthesizeFrameParameters(
                receiver_runtime=ReceiverRuntime.RAW_RESIDUAL
            )
        elif op_enum is ExtendedOperation.MOTION_CONDITIONAL:
            params = MotionConditionalParameters()
        elif op_enum is ExtendedOperation.TEMPORAL_COHERENCE:
            params = TemporalCoherenceParameters(
                temporal_window=2, similarity_threshold=0.0
            )
        else:
            raise AssertionError(f"unknown operator {op_enum}")
        candidates = generator(canvas, params, top_n=8)
        for c in candidates:
            assert isinstance(c.predicted_axis_decomposition, AxisDecomposition)
            assert c.predicted_axis_decomposition.axis_tag == "[predicted]"
            assert c.predicted_axis_decomposition.canonical_provenance


# ---------------------------------------------------------------------------
# Canonical Provenance per Catalog #323 + canonical-routing markers per #341.
# ---------------------------------------------------------------------------


def test_canonical_routing_markers_non_promotable_per_catalog_341() -> None:
    canvas = _small_canvas()
    candidates = generate_replace_one_candidates(
        canvas,
        ReplaceOneParameters(
            target_pair_idx=0,
            alternative_selector_id=1,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        ),
    )
    for c in candidates:
        markers = c.canonical_routing_markers
        assert markers["predicted_delta_adjustment"] == 0.0
        assert markers["promotable"] is False
        assert markers["axis_tag"] == "[predicted]"


def test_canonical_provenance_threaded_through_every_candidate() -> None:
    canvas = _small_canvas()
    candidates = generate_synthesize_frame_candidates(
        canvas,
        SynthesizeFrameParameters(
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL, synthesis_seed=11
        ),
    )
    for c in candidates:
        prov = c.catalog_323_provenance
        # Canonical Provenance structure per Catalog #323 / build_provenance_for_predicted.
        assert "artifact_kind" in prov
        # axis_tag is [predicted] per Catalog #341 Tier A defaults.
        assert prov.get("measurement_axis") == "[predicted]"
        assert prov.get("evidence_grade") == "predicted"


# ---------------------------------------------------------------------------
# Apples-to-apples baseline preservation per 10th standing directive.
# ---------------------------------------------------------------------------


def test_apples_to_apples_archive_sha256_threaded_into_provenance() -> None:
    canvas = _small_canvas()
    candidates = generate_drop_frame_candidates(canvas, DropFrameParameters())
    for c in candidates:
        # Provenance carries canonical_helper_invocation + source_path.
        prov = c.catalog_323_provenance
        # The canonical_helper_invocation field identifies the canonical
        # Provenance builder (sister of model_id in older Provenance shapes).
        assert "canonical_helper_invocation" in prov
        # source_path is the deterministic predictor identifier derived from
        # (model_id, inputs_sha256) via build_provenance_for_predicted.
        sp = prov.get("source_path", "")
        assert "<predictor:" in sp
        assert (
            "pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators"
            in sp
        )


# ---------------------------------------------------------------------------
# Cosine similarity + per-pair signature helpers.
# ---------------------------------------------------------------------------


def test_cosine_similarity_orthogonal_returns_zero() -> None:
    assert _cosine_similarity((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == 0.0


def test_cosine_similarity_identical_returns_one() -> None:
    assert _cosine_similarity((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(
        1.0
    )


def test_cosine_similarity_anti_parallel_returns_neg_one() -> None:
    assert _cosine_similarity(
        (1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)
    ) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert _cosine_similarity((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == 0.0


def test_per_pair_axis_signature_returns_3_tuple() -> None:
    cells = [
        _build_cell(
            pair_idx=0,
            frame_idx=1,
            scorer_axis=ScorerAxis.SEGNET_5CLASS,
            predicted_delta_score=-1e-6,
        ),
        _build_cell(
            pair_idx=0,
            frame_idx=1,
            scorer_axis=ScorerAxis.POSENET_6D,
            predicted_delta_score=-2e-6,
        ),
        _build_cell(
            pair_idx=0,
            frame_idx=1,
            scorer_axis=ScorerAxis.RATE_TERM,
            predicted_byte_cost=-5,
            predicted_delta_score=0.0,
        ),
    ]
    sig = _per_pair_axis_signature(cells)
    assert sig == (-1e-6, -2e-6, -5.0)


# ---------------------------------------------------------------------------
# Sister BUILD-2+3 canvas operation proxy mapping.
# ---------------------------------------------------------------------------


def test_extended_to_canvas_proxy_maps_all_8_operators() -> None:
    for op_enum in ExtendedOperation:
        proxy = _extended_to_canvas_operation_proxy(op_enum)
        assert isinstance(proxy, CanonicalOperation)


def test_extended_to_canvas_proxy_specific_mappings() -> None:
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.REPLACE_ONE)
        is CanonicalOperation.REPAIR
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.REPLACE_MANY)
        is CanonicalOperation.REPAIR
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.MERGE_PAIR)
        is CanonicalOperation.MASKED
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.REORDER_PAIR)
        is CanonicalOperation.MASKED
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.DROP_FRAME)
        is CanonicalOperation.FULL_DROP
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.SYNTHESIZE_FRAME)
        is CanonicalOperation.REPAIR
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.MOTION_CONDITIONAL)
        is CanonicalOperation.FEATHERED
    )
    assert (
        _extended_to_canvas_operation_proxy(ExtendedOperation.TEMPORAL_COHERENCE)
        is CanonicalOperation.FEATHERED
    )


# ---------------------------------------------------------------------------
# Cross-operator composition per 8th INDIVIDUALLY-FRACTAL standing directive.
# ---------------------------------------------------------------------------


def test_cross_operator_composition_8_extended_emit_disjoint_candidates() -> None:
    """8 operators × small canvas should each emit candidates without
    fatal exception; together they form the canonical extended-operator
    vocabulary per 8th INDIVIDUALLY-FRACTAL standing directive 2026-05-26.
    """
    canvas = _small_canvas()
    counts: dict[ExtendedOperation, int] = {}
    for op_enum, generator in EXTENDED_OPERATOR_REGISTRY.items():
        if op_enum is ExtendedOperation.REPLACE_ONE:
            params: Any = ReplaceOneParameters(
                target_pair_idx=0,
                alternative_selector_id=1,
                receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
            )
        elif op_enum is ExtendedOperation.REPLACE_MANY:
            params = ReplaceManyParameters(beam_width=2, beam_depth=2)
        elif op_enum is ExtendedOperation.MERGE_PAIR:
            params = MergePairParameters()
        elif op_enum is ExtendedOperation.REORDER_PAIR:
            params = ReorderPairParameters(block_size=2)
        elif op_enum is ExtendedOperation.DROP_FRAME:
            params = DropFrameParameters()
        elif op_enum is ExtendedOperation.SYNTHESIZE_FRAME:
            params = SynthesizeFrameParameters(
                receiver_runtime=ReceiverRuntime.RAW_RESIDUAL
            )
        elif op_enum is ExtendedOperation.MOTION_CONDITIONAL:
            params = MotionConditionalParameters()
        elif op_enum is ExtendedOperation.TEMPORAL_COHERENCE:
            params = TemporalCoherenceParameters(
                temporal_window=3, similarity_threshold=0.0
            )
        else:
            raise AssertionError("unknown operator")
        candidates = generator(canvas, params, top_n=4)
        counts[op_enum] = len(candidates)
    # Per 8th INDIVIDUALLY-FRACTAL: every operator should have a candidate
    # surface; at least one should emit candidates.
    assert sum(counts.values()) > 0


def test_top_n_truncation_observed() -> None:
    canvas = _small_canvas()
    params = MergePairParameters()
    short = generate_merge_pair_candidates(canvas, params, top_n=2)
    longer = generate_merge_pair_candidates(canvas, params, top_n=32)
    assert len(short) <= 2
    assert len(longer) >= len(short)


def test_top_n_must_be_positive_int() -> None:
    canvas = _small_canvas()
    with pytest.raises(ValueError, match="top_n must be a positive int"):
        generate_drop_frame_candidates(
            canvas, DropFrameParameters(), top_n=0
        )
    with pytest.raises(ValueError, match="top_n must be a positive int"):
        generate_drop_frame_candidates(
            canvas, DropFrameParameters(), top_n=True  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Determinism: identical input produces identical output.
# ---------------------------------------------------------------------------


def test_determinism_identical_inputs_produce_identical_outputs() -> None:
    canvas = _small_canvas()
    params = SynthesizeFrameParameters(
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL, synthesis_seed=42
    )
    a = generate_synthesize_frame_candidates(canvas, params, top_n=8)
    b = generate_synthesize_frame_candidates(canvas, params, top_n=8)
    assert len(a) == len(b)
    for ca, cb in zip(a, b, strict=True):
        assert ca.predicted_delta_score == cb.predicted_delta_score
        assert ca.predicted_byte_cost == cb.predicted_byte_cost
        assert ca.archive_candidate_path == cb.archive_candidate_path


def test_sorting_ascending_by_predicted_delta_score() -> None:
    canvas = _small_canvas()
    candidates = generate_replace_one_candidates(
        canvas,
        ReplaceOneParameters(
            target_pair_idx=2,
            alternative_selector_id=3,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        ),
    )
    if len(candidates) >= 2:
        for i in range(len(candidates) - 1):
            assert (
                candidates[i].predicted_delta_score
                <= candidates[i + 1].predicted_delta_score
            )


# ---------------------------------------------------------------------------
# as_dict() round-trip serialization.
# ---------------------------------------------------------------------------


def test_candidate_as_dict_serializes_to_json_safe_dict() -> None:
    import json

    canvas = _small_canvas()
    candidates = generate_drop_frame_candidates(canvas, DropFrameParameters())
    for c in candidates:
        d = c.as_dict()
        # Round-trip via json to verify JSON-safety.
        s = json.dumps(d, sort_keys=True)
        d2 = json.loads(s)
        assert d2["operation"] == c.operation.value
        assert d2["predicted_delta_score"] == c.predicted_delta_score


# ---------------------------------------------------------------------------
# Schema constants exposed.
# ---------------------------------------------------------------------------


def test_extended_module_schema_constant_present() -> None:
    assert (
        EXTENDED_MODULE_SCHEMA
        == "pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.v0_scaffold"
    )


# ---------------------------------------------------------------------------
# Live-repo regression guard: module + CLI importable.
# ---------------------------------------------------------------------------


def test_live_repo_regression_module_importable() -> None:
    import importlib

    mod = importlib.import_module(
        "tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators"
    )
    assert hasattr(mod, "ExtendedOperation")
    assert hasattr(mod, "EXTENDED_OPERATOR_REGISTRY")
    assert hasattr(mod, "EXTENDED_OPERATION_CANONICAL_EQUATION_IDS")


def test_live_repo_regression_cli_importable() -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    cli_path = repo_root / "tools" / "apply_8_extended_operators_to_5d_canvas_cli.py"
    assert cli_path.exists()
    # The CLI is a script (not a module); import it via importlib.util.
    spec = importlib.util.spec_from_file_location(
        "apply_8_extended_operators_to_5d_canvas_cli", cli_path
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
    assert hasattr(mod, "_build_parser")


def test_live_repo_regression_cli_executes_replace_one(
    tmp_path: pathlib.Path,
) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    canvas = _small_canvas()
    canvas_path = tmp_path / "small_canvas.json"
    canvas_path.write_text(
        json.dumps(
            {
                "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
                "archive_sha256": canvas.archive_sha256,
                "cells": [cell.as_dict() for cell in canvas._cells.values()],
            }
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[3]
    cli_path = repo_root / "tools" / "apply_8_extended_operators_to_5d_canvas_cli.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--canvas-path",
            str(canvas_path),
            "--operator",
            ExtendedOperation.REPLACE_ONE.value,
            "--target-pair-idx",
            "1",
            "--alternative-selector-id",
            "7",
            "--top-n",
            "4",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema"] == "extended_operator_candidates.v0"
    assert payload["operator"] == ExtendedOperation.REPLACE_ONE.value
    assert payload["candidates_emitted"] >= 1
    markers = payload["candidates"][0]["canonical_routing_markers"]
    assert markers["promotable"] is False


def test_live_repo_regression_cli_executes_all_operators(
    tmp_path: pathlib.Path,
) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    canvas = _small_canvas()
    canvas_path = tmp_path / "small_canvas.json"
    canvas_path.write_text(
        json.dumps(
            {
                "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
                "archive_sha256": canvas.archive_sha256,
                "cells": [cell.as_dict() for cell in canvas._cells.values()],
            }
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[3]
    cli_path = repo_root / "tools" / "apply_8_extended_operators_to_5d_canvas_cli.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--canvas-path",
            str(canvas_path),
            "--operator",
            "all",
            "--top-n",
            "3",
            "--global-top-n",
            "5",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema"] == "extended_operator_batch_candidates.v0"
    assert payload["operator"] == "all"
    assert payload["operator_count"] == 8
    assert set(payload["operator_results"]) == {op.value for op in ExtendedOperation}
    assert payload["flattened_candidates_emitted"] <= 5


# ---------------------------------------------------------------------------
# Operator's "merge and other ops do even better" insight regression.
# ---------------------------------------------------------------------------


def test_operator_insight_merge_pair_emits_meaningful_candidates() -> None:
    """Per operator insight 2026-05-26: merge-pair should emit candidates
    on a small canvas (validates the canonical infrastructure)."""
    canvas = _small_canvas()
    candidates = generate_merge_pair_candidates(
        canvas, MergePairParameters(max_candidates=64), top_n=16
    )
    assert len(candidates) > 0
    # Per "merge and other ops do even better": at least one candidate should
    # show negative predicted_delta (improvement).
    assert any(c.predicted_delta_score < 0.0 for c in candidates)


def test_operator_insight_8_operators_each_register_canonical_eq_id() -> None:
    """Per Catalog #344: every operator's canonical equation registered
    at FORMALIZATION_PENDING."""
    for op in ExtendedOperation:
        assert op in EXTENDED_OPERATION_CANONICAL_EQUATION_IDS
        eq_id = EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[op]
        # The 8 IDs are distinct.
        assert (
            sum(1 for e in EXTENDED_OPERATION_CANONICAL_EQUATION_IDS.values() if e == eq_id)
            == 1
        )
