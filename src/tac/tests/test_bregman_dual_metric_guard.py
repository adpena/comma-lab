from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from tac.witness_dsl import lever_registry
from tac.witness_dsl.bregman_dual_metric_guard import (
    CANONICAL_METRIC_ID,
    BregmanDualMetricAdoptionError,
    BregmanDualMetricBinding,
    canonical_bregman_dual_metric_binding,
    resolve_bregman_dual_metric_adoption,
)


def test_exact_existing_registry_resolution_and_guard_acceptance() -> None:
    assert lever_registry.canonical_metric_ids() == (CANONICAL_METRIC_ID,)
    descriptor = lever_registry.resolve_canonical_metric(CANONICAL_METRIC_ID)
    assert descriptor.binding_artifact == (
        ".omx/research/bregman_v9_all_surfaces_binding_20260714.json"
    )
    adoption = resolve_bregman_dual_metric_adoption(
        [canonical_bregman_dual_metric_binding()]
    )
    assert adoption.registry_entry is descriptor
    assert adoption.binding.metric_id == CANONICAL_METRIC_ID
    assert adoption.binding.fisher_natural_cotangent_geometry == (
        "inverse_hessian_H_inverse"
    )
    assert adoption.binding.fisher_natural_cotangent_solve == "typed_linear_solve"
    assert adoption.binding.fisher_natural_cotangent_solve_elided is False
    assert adoption.binding.dual_euclidean_no_solve_scope == (
        "squared_hessian_H_squared_only"
    )


def test_complete_mapping_adoption_is_accepted() -> None:
    binding = canonical_bregman_dual_metric_binding()
    adoption = resolve_bregman_dual_metric_adoption([asdict(binding)])
    assert adoption.binding == binding


def test_missing_duplicate_and_unknown_bindings_fail_closed() -> None:
    binding = canonical_bregman_dual_metric_binding()
    with pytest.raises(BregmanDualMetricAdoptionError, match="missing"):
        resolve_bregman_dual_metric_adoption([])
    with pytest.raises(BregmanDualMetricAdoptionError, match="exactly once"):
        resolve_bregman_dual_metric_adoption([binding, binding])
    with pytest.raises(BregmanDualMetricAdoptionError, match="unknown"):
        resolve_bregman_dual_metric_adoption(
            [replace(binding, metric_id="unknown_metric_v1")]
        )


def test_incomplete_and_extra_mapping_fields_fail_closed() -> None:
    payload = asdict(canonical_bregman_dual_metric_binding())
    payload.pop("fisher_natural_cotangent_solve")
    with pytest.raises(BregmanDualMetricAdoptionError, match="missing"):
        resolve_bregman_dual_metric_adoption([payload])
    payload = asdict(canonical_bregman_dual_metric_binding())
    payload["shortcut"] = True
    with pytest.raises(BregmanDualMetricAdoptionError, match="unknown"):
        resolve_bregman_dual_metric_adoption([payload])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fisher_natural_cotangent_geometry", "raw_dual_euclidean", "Fisher-natural"),
        ("fisher_natural_cotangent_solve", "no_solve", "typed H\\^-1 linear solve"),
        ("fisher_natural_cotangent_solve_elided", True, "solve_elided"),
        ("fisher_natural_cotangent_solve_elided", 0, "solve_elided"),
        ("dual_euclidean_no_solve_scope", "ordinary_hessian_H", "squared_hessian"),
    ],
)
def test_shortcut_bindings_are_rejected(field: str, value: object, message: str) -> None:
    payload = asdict(canonical_bregman_dual_metric_binding())
    payload[field] = value
    with pytest.raises(BregmanDualMetricAdoptionError, match=message):
        BregmanDualMetricBinding.from_mapping(payload)


def test_duplicate_metric_entries_in_existing_registry_fail_exact_resolution(
    monkeypatch,
) -> None:
    descriptor = lever_registry.resolve_canonical_metric(CANONICAL_METRIC_ID)
    monkeypatch.setattr(
        lever_registry,
        "_CANONICAL_METRICS",
        (descriptor, descriptor),
    )
    with pytest.raises(lever_registry.MetricResolutionError, match="duplicated"):
        lever_registry.resolve_canonical_metric(CANONICAL_METRIC_ID)


def test_missing_binding_artifact_is_rejected_as_orphan(monkeypatch) -> None:
    descriptor = lever_registry.resolve_canonical_metric(CANONICAL_METRIC_ID)
    monkeypatch.setattr(
        lever_registry,
        "_CANONICAL_METRICS",
        (replace(descriptor, binding_artifact=".omx/research/does_not_exist.json"),),
    )
    with pytest.raises(lever_registry.MetricResolutionError, match="orphaned"):
        lever_registry.resolve_canonical_metric(CANONICAL_METRIC_ID)


def test_guard_is_not_a_trainer_lever() -> None:
    binding = canonical_bregman_dual_metric_binding()
    assert not hasattr(binding, "overrides")
    assert CANONICAL_METRIC_ID not in lever_registry.lever_factories()
