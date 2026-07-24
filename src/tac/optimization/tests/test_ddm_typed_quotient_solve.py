from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from tac.optimization.coupled_margin_levelset import CouplingOperator
from tac.optimization.ddm_typed_quotient_solve import (
    CLASS_PAIRS,
    EVIDENCE_AXIS,
    METRIC_COORDINATE_SYSTEM,
    RATE_SCORE_PER_BYTE,
    AlternationStage,
    EffectiveQuantum,
    EvaluationRecursionLevel,
    G4TemporalClass,
    MeasuredScorerGeometry,
    ScorerVisibility,
    TypedBlock,
    TypedQuotientSolveError,
    bounded_exact_metric_sieve,
    generalized_metric_dictionary_update,
    solve_metric_active_block,
    validate_alternation_trace,
    validate_geometry_ladder,
    validate_typed_atlas,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _geometry() -> MeasuredScorerGeometry:
    return MeasuredScorerGeometry(
        metric_id="measured_margin_fisher_pose6_v1",
        coordinate_system=METRIC_COORDINATE_SYSTEM,
        metric_gram=np.array([[2.0, 0.25], [0.25, 1.0]]),
        composite_hessian=np.array([[0.8, -0.1], [-0.1, 0.4]]),
        seg_head_rank=4,
        pose_rank=2,
        evidence_axis=EVIDENCE_AXIS,
        geometry_receipt_sha256=SHA_A,
        composite_r_adjoint_sha256=SHA_B,
        inner_jacobian_sha256=SHA_C,
        pose_quadratic_sha256=SHA_D,
        dual_metric_readback_active=True,
        bregman_binding_active=True,
    )


def _quanta() -> tuple[EffectiveQuantum, ...]:
    return (
        EffectiveQuantum("q0", 1.0, 0.1, (0.1, 0.2, 0.4)),
        EffectiveQuantum("q1", 1.0, 0.2, (0.2, 0.4, 0.8)),
    )


def test_identity_geometry_is_a_refused_control() -> None:
    with pytest.raises(TypedQuotientSolveError, match="identity/Euclidean"):
        MeasuredScorerGeometry(
            metric_id="measured_metric_v1",
            coordinate_system=METRIC_COORDINATE_SYSTEM,
            metric_gram=np.eye(2),
            composite_hessian=np.array([[0.8, 0.1], [0.1, 0.5]]),
            seg_head_rank=4,
            pose_rank=2,
            evidence_axis=EVIDENCE_AXIS,
            geometry_receipt_sha256=SHA_A,
            composite_r_adjoint_sha256=SHA_B,
            inner_jacobian_sha256=SHA_C,
            pose_quadratic_sha256=SHA_D,
            dual_metric_readback_active=True,
            bregman_binding_active=True,
        )


def test_exact_isotropic_composite_hessian_is_not_mislabeled_as_euclidean() -> None:
    geometry = MeasuredScorerGeometry(
        metric_id="measured_margin_fisher_plus_isotropic_composite_r_v1",
        coordinate_system=METRIC_COORDINATE_SYSTEM,
        metric_gram=np.array([[2.0, 0.25], [0.25, 1.0]]),
        composite_hessian=0.5 * np.eye(2),
        seg_head_rank=4,
        pose_rank=2,
        evidence_axis=EVIDENCE_AXIS,
        geometry_receipt_sha256=SHA_A,
        composite_r_adjoint_sha256=SHA_B,
        inner_jacobian_sha256=SHA_C,
        pose_quadratic_sha256=SHA_D,
        dual_metric_readback_active=True,
        bregman_binding_active=True,
    )
    np.testing.assert_allclose(
        geometry.second_order_metric,
        np.array([[2.5, 0.25], [0.25, 1.5]]),
    )


def test_metric_active_kkt_uses_exact_second_order_without_identity_damping() -> None:
    operator = CouplingOperator(
        matrix=np.array([[1.0, 0.0]]),
        margin=np.array([0.0]),
        required_margin=np.array([0.5]),
        targeted_count=1,
        row_labels=("target",),
        dof_labels=("q0", "q1"),
        activation_pattern_sha256=SHA_A,
    )
    geometry = _geometry()
    result = solve_metric_active_block(
        operator,
        geometry=geometry,
        quanta=_quanta(),
        maximum_integer_steps=8,
    )
    assert result.diagnostics.converged is True
    np.testing.assert_allclose(result.hessian, geometry.second_order_metric)
    assert result.step[0] == pytest.approx(0.5)


def test_bounded_exact_sieve_prices_real_coder_and_reports_completion() -> None:
    result = bounded_exact_metric_sieve(
        np.array([0.09, 0.18]),
        geometry=_geometry(),
        quanta=_quanta(),
        integer_radius=1,
        node_limit=9,
        feasible=lambda step: bool(step[0] >= 0.0 and step[1] >= 0.0),
        real_coder_bytes=lambda step: int(np.count_nonzero(step)),
    )
    assert result.search_complete is True
    assert result.evaluated_candidates == 9
    assert result.feasible_candidates == 9
    np.testing.assert_array_equal(result.integer_coefficients, np.array([1, 1]))
    np.testing.assert_allclose(result.realized_step, np.array([0.1, 0.2]))
    error = result.realized_step - np.array([0.09, 0.18])
    assert result.metric_objective == pytest.approx(
        0.5 * error @ _geometry().second_order_metric @ error
        + RATE_SCORE_PER_BYTE * result.counted_bytes
    )

    truncated = bounded_exact_metric_sieve(
        np.array([0.09, 0.18]),
        geometry=_geometry(),
        quanta=_quanta(),
        integer_radius=1,
        node_limit=2,
        feasible=lambda _step: True,
        real_coder_bytes=lambda _step: 1,
    )
    assert truncated.search_complete is False


def test_generalized_dictionary_update_is_metric_orthonormal_and_exact_at_full_rank() -> None:
    samples = np.array([[1.0, 2.0], [-2.0, 1.0], [0.5, -0.25]])
    result = generalized_metric_dictionary_update(samples, geometry=_geometry(), rank=2)
    np.testing.assert_allclose(result.reconstruction, samples, atol=1e-12)
    assert result.weighted_residual_squared == pytest.approx(0.0, abs=1e-24)
    assert result.metric_orthonormality_error < 1e-12
    assert result.method == "MEASURED_METRIC_GENERALIZED_SVD_LS"


def _atlas_blocks() -> tuple[TypedBlock, ...]:
    return tuple(
        TypedBlock(
            block_id=f"pair_{left}_{right}",
            stratum="edge",
            scorer_visibility=ScorerVisibility.SEG,
            temporal_class=G4TemporalClass.STATIC_IN_IMAGE,
            class_pair=(left, right),
            representation_type="SKELETON",
            recursion_level=EvaluationRecursionLevel.LEVEL1_SCORER_INTERNALS,
            measured_flip_mass=index + 1,
            counted_bytes=1,
            parameter_bytes=0,
            exception_bytes=0,
            connection_operator_code_bytes=0,
            amortization_factor=600,
            coder_race_winner="SKELETON",
            pose_serving=False,
            atlas_receipt_sha256=SHA_A,
            coder_race_receipt_sha256=SHA_B,
        )
        for index, (left, right) in enumerate(CLASS_PAIRS)
    )


def test_typed_atlas_requires_measured_all_ten_pair_coverage() -> None:
    masses = {pair: index + 1 for index, pair in enumerate(CLASS_PAIRS)}
    result = validate_typed_atlas(_atlas_blocks(), measured_flip_mass_by_pair=masses)
    assert result.valid is True
    assert result.class_pair_count == len(tuple(itertools.combinations(range(5), 2))) == 10
    assert result.total_measured_flip_mass == 55

    rows = list(_atlas_blocks())
    rows[1] = replace(rows[1], measured_flip_mass=1)
    rows.append(
        replace(
            rows[1],
            block_id=f"{rows[1].block_id}_pose",
            scorer_visibility=ScorerVisibility.POSE,
        )
    )
    split = validate_typed_atlas(rows, measured_flip_mass_by_pair=masses)
    assert split.block_count == 11
    assert split.total_measured_flip_mass == 55

    with pytest.raises(TypedQuotientSolveError, match="all ten"):
        validate_typed_atlas(
            _atlas_blocks()[:-1],
            measured_flip_mass_by_pair=masses,
        )
    with pytest.raises(TypedQuotientSolveError, match="reconcile"):
        validate_typed_atlas(
            (*_atlas_blocks(), replace(_atlas_blocks()[1], block_id="duplicate_mass")),
            measured_flip_mass_by_pair=masses,
        )


def test_gauge_and_connection_byte_laws_fail_closed() -> None:
    with pytest.raises(TypedQuotientSolveError, match="class_pair"):
        replace(_atlas_blocks()[0], class_pair=(False, 1))
    with pytest.raises(TypedQuotientSolveError, match="exactly zero bytes"):
        TypedBlock(
            block_id="bad_gauge",
            stratum="cell",
            scorer_visibility=ScorerVisibility.INVISIBLE_GAUGE,
            temporal_class=G4TemporalClass.TRANSIENT,
            class_pair=(0, 1),
            representation_type="GAUGE",
            recursion_level=EvaluationRecursionLevel.LEVEL1_SCORER_INTERNALS,
            measured_flip_mass=1,
            counted_bytes=1,
            parameter_bytes=0,
            exception_bytes=0,
            connection_operator_code_bytes=0,
            amortization_factor=1,
            coder_race_winner="FIBER",
            pose_serving=False,
            atlas_receipt_sha256=SHA_A,
            coder_race_receipt_sha256=SHA_B,
        )
    with pytest.raises(TypedQuotientSolveError, match="not physical BEV"):
        TypedBlock(
            block_id="xi_proxy",
            stratum="edge",
            scorer_visibility=ScorerVisibility.JOINT,
            temporal_class=G4TemporalClass.STATIC_IN_XI_PROXY,
            class_pair=(0, 1),
            representation_type="CONNECTION",
            recursion_level=EvaluationRecursionLevel.LEVEL2_PAIR_TRAJECTORY,
            measured_flip_mass=1,
            counted_bytes=3,
            parameter_bytes=2,
            exception_bytes=1,
            connection_operator_code_bytes=0,
            amortization_factor=600,
            coder_race_winner="FIBER",
            pose_serving=False,
            atlas_receipt_sha256=SHA_A,
            coder_race_receipt_sha256=SHA_B,
        )


def test_ws1_pose_serving_content_cannot_be_misfiled_in_seg_stream() -> None:
    with pytest.raises(TypedQuotientSolveError, match="pose-serving WS1"):
        TypedBlock(
            block_id="ws1_misfiled_pose_fiber",
            stratum="joint-accepted",
            scorer_visibility=ScorerVisibility.SEG,
            temporal_class=G4TemporalClass.TRANSIENT,
            class_pair=(0, 1),
            representation_type="FIBER",
            recursion_level=EvaluationRecursionLevel.LEVEL1_SCORER_INTERNALS,
            measured_flip_mass=8,
            counted_bytes=1,
            parameter_bytes=0,
            exception_bytes=0,
            connection_operator_code_bytes=0,
            amortization_factor=1,
            coder_race_winner="FIBER",
            pose_serving=True,
            atlas_receipt_sha256=SHA_A,
            coder_race_receipt_sha256=SHA_B,
        )

    block = TypedBlock(
        block_id="ws1_pose_fiber",
        stratum="joint-accepted",
        scorer_visibility=ScorerVisibility.POSE,
        temporal_class=G4TemporalClass.TRANSIENT,
        class_pair=(0, 1),
        representation_type="FIBER",
        recursion_level=EvaluationRecursionLevel.LEVEL1_SCORER_INTERNALS,
        measured_flip_mass=8,
        counted_bytes=1,
        parameter_bytes=0,
        exception_bytes=0,
        connection_operator_code_bytes=0,
        amortization_factor=1,
        coder_race_winner="FIBER",
        pose_serving=True,
        atlas_receipt_sha256=SHA_A,
        coder_race_receipt_sha256=SHA_B,
    )
    assert block.pose_serving is True


def test_alternation_and_geometry_ladders_put_authority_first() -> None:
    assert (
        validate_alternation_trace(
            (
                AlternationStage.ARGMAX_CELL,
                AlternationStage.WITHIN_CELL_LATTICE,
                AlternationStage.REAL_CODER_PRICE,
            )
            * 2,
            pose_tube_active_each_iteration=True,
            real_coder_price_inside_objective=True,
        )
        == 2
    )
    with pytest.raises(TypedQuotientSolveError, match="out of canonical order"):
        validate_alternation_trace(
            (
                AlternationStage.WITHIN_CELL_LATTICE,
                AlternationStage.ARGMAX_CELL,
                AlternationStage.REAL_CODER_PRICE,
            ),
            pose_tube_active_each_iteration=True,
            real_coder_price_inside_objective=True,
        )
    assert validate_geometry_ladder(("MEASURED_SCORER_SECOND_ORDER", "IDENTITY_EUCLIDEAN_CONTROL"))
    with pytest.raises(TypedQuotientSolveError, match="must begin"):
        validate_geometry_ladder(("IDENTITY_EUCLIDEAN_CONTROL",))
