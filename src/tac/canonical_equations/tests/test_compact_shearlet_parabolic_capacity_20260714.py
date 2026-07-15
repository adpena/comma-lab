from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.canonical_equations.compact_shearlet_parabolic_capacity_20260714 import (
    EQUATION_ID,
    PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION,
    SELECTION_STATUS,
    SOURCE_MODULE,
    STRUCTURAL_PROOF,
    STRUCTURAL_PROOF_COMPILED_SOURCE_SHA256,
    STRUCTURAL_PROOF_SHA256,
    build_compact_shearlet_parabolic_capacity_v1,
    compact_shearlet_sigma_pair,
    compact_shearlet_structural_certificate_law,
    populate_compact_shearlet_parabolic_capacity_v1,
)
from tac.canonical_equations.registry import (
    get_equation_by_id,
    load_registry_events_lenient,
    query_equations,
)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_parabolic_sigma_pair_derives_width_as_length_squared_before_clamp() -> None:
    w0 = 0.6
    aniso = 1.4
    for scale in range(5):
        sigma_n, sigma_t = compact_shearlet_sigma_pair(
            scale,
            w0=w0,
            width_ratio=2.0,
            aniso=aniso,
            min_sigma=1.0e-8,
        )
        assert sigma_n == pytest.approx((sigma_t / aniso) ** 2 / w0)
    assert compact_shearlet_sigma_pair(100) == pytest.approx((0.02, 0.02))


@pytest.mark.parametrize(
    ("scale", "kwargs"),
    [
        (-1, {}),
        (True, {}),
        (0, {"width_ratio": 1.0}),
        (0, {"aniso": 0.9}),
        (0, {"w0": float("nan")}),
    ],
)
def test_parabolic_sigma_pair_fails_closed(scale, kwargs) -> None:
    with pytest.raises(ValueError):
        compact_shearlet_sigma_pair(scale, **kwargs)


def test_primitive_owned_structural_certificate_passes() -> None:
    certificate = compact_shearlet_structural_certificate_law()
    assert certificate["passes"] is True
    assert certificate["integer_lattice_preserving"] is True
    assert certificate["parabolic_scaling_monotone"] is True
    assert certificate["shear_discrimination_ratio"] > 10.0
    assert certificate["shearlet_envelope_span"] > certificate["fourier_envelope_span"]


def test_equation_separates_primitive_and_compiled_proof_source_custody() -> None:
    assert _sha256(SOURCE_MODULE) == PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION
    assert _sha256(STRUCTURAL_PROOF) == STRUCTURAL_PROOF_SHA256
    assert PRIMITIVE_SOURCE_SHA256_AT_REGISTRATION != (
        STRUCTURAL_PROOF_COMPILED_SOURCE_SHA256
    )
    equation = build_compact_shearlet_parabolic_capacity_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["source_equivalence_claim"] is False
    assert equation.domain_of_validity["selection_metric_status"] == SELECTION_STATUS
    assert equation.domain_of_validity["score_claim"] is False
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert equation.domain_of_validity["dsl_wire_status"].startswith("OWED")
    assert equation.canonical_consumers == ()
    assert equation.empirical_anchors[0].empirical_output["passes"] is True
    assert equation.empirical_anchors[1].inputs["same_source_bytes"] is False
    assert equation.empirical_anchors[1].empirical_output[
        "selection_metric_status"
    ] == SELECTION_STATUS


def test_population_is_explicit_and_uses_registry_api(tmp_path) -> None:
    registry_path = tmp_path / "canonical_equations_registry.jsonl"
    lock_path = tmp_path / "canonical_equations_registry.jsonl.lock"
    assert not registry_path.exists()
    populated = populate_compact_shearlet_parabolic_capacity_v1(
        path=registry_path,
        lock_path=lock_path,
        agent="codex",
        subagent_id="bregman_all_surfaces_504_test",
    )
    assert populated.equation_id == EQUATION_ID
    assert [event["equation_id"] for event in load_registry_events_lenient(registry_path)] == [
        EQUATION_ID
    ]
    assert [equation.equation_id for equation in query_equations(path=registry_path)] == [
        EQUATION_ID
    ]
    assert get_equation_by_id(EQUATION_ID, path=registry_path) is not None


def test_live_registry_surfaces_curvelet_shearlet_and_metric_equations_once() -> None:
    ids = [equation.equation_id for equation in query_equations()]
    expected = (
        "optimal_metric_unification_v1",
        "categorical_fisher_trust_region_winner_rival_v1",
        "windowed_curvelet_parabolic_capacity_v1",
        EQUATION_ID,
    )
    for equation_id in expected:
        assert ids.count(equation_id) == 1
