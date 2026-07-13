import math

import pytest

from tac.canonical_equations.manifold_geometry_slots_20260713 import (
    advective_acoustic_metric,
    build_all_manifold_geometry_slot_equations,
    equal_flip_metric_density,
    fisher_pairwise_wall_distance,
    worldsheet_rate_saving,
)


def test_equal_flip_metric_density_is_normalized_and_data_dependent() -> None:
    assert equal_flip_metric_density([0.0, 1.0, 3.0]) == (0.0, 0.25, 0.75)
    with pytest.raises(ValueError, match="positive mass"):
        equal_flip_metric_density([0.0, 0.0])


def test_pairwise_fisher_wall_is_flat_to_first_order_and_monotone() -> None:
    small = 1e-4
    assert fisher_pairwise_wall_distance(small) == pytest.approx(small / 2.0, rel=1e-7)
    assert fisher_pairwise_wall_distance(0.44) > fisher_pairwise_wall_distance(0.12) > 0.0
    assert fisher_pairwise_wall_distance(-0.44) == pytest.approx(fisher_pairwise_wall_distance(0.44))


def test_advective_acoustic_metric_has_lorentzian_determinant() -> None:
    metric = advective_acoustic_metric((2.0, -1.0), 3.0)
    determinant = (
        metric[0][0] * (metric[1][1] * metric[2][2] - metric[1][2] * metric[2][1])
        - metric[0][1] * (metric[1][0] * metric[2][2] - metric[1][2] * metric[2][0])
        + metric[0][2] * (metric[1][0] * metric[2][1] - metric[1][1] * metric[2][0])
    )
    assert determinant == pytest.approx(-9.0)


def test_worldsheet_saving_refuses_misaligned_marks() -> None:
    assert worldsheet_rate_saving([10.0, 10.0], 8.0, [1.0], [2.0], [3.0]) == 6.0
    with pytest.raises(ValueError, match="equal length"):
        worldsheet_rate_saving([10.0], 5.0, [1.0], [], [1.0])


def test_equations_encode_empirical_and_derived_scope_without_score_authority() -> None:
    equations = build_all_manifold_geometry_slot_equations()
    assert {equation.equation_id for equation in equations} == {
        "flip_density_chart_metric_v1",
        "fisher_pairwise_decision_wall_v1",
        "advective_worldsheet_rate_v1",
    }
    by_id = {equation.equation_id: equation for equation in equations}
    assert len(by_id["flip_density_chart_metric_v1"].empirical_anchors) == 1
    assert len(by_id["fisher_pairwise_decision_wall_v1"].empirical_anchors) == 1
    assert by_id["advective_worldsheet_rate_v1"].empirical_anchors == ()
    assert all(not equation.provenance.score_claim_valid for equation in equations)
    assert math.isclose(
        by_id["fisher_pairwise_decision_wall_v1"].predicted_vs_empirical_residual[
            "flat_shadow_relative_error_at_flip_p90"
        ],
        0.007970304445913068,
    )
