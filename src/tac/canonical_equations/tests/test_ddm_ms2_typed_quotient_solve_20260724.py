from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.ddm_ms2_typed_quotient_solve_20260724 import (
    EQUATION_IDS,
    build_ddm_ms2_typed_quotient_equations,
    derivation_edges,
    effective_quantum,
    populate_ddm_ms2_typed_quotient_equations,
    scorer_metric_rate_action,
    skeleton_fiber_coder_race,
    typed_block_exchange_rate,
    visible_quotient_counted_bytes,
)
from tac.canonical_equations.registry import load_registry_events_lenient
from tac.optimization.ddm_typed_quotient_solve import (
    EVIDENCE_AXIS,
    METRIC_COORDINATE_SYSTEM,
    MeasuredScorerGeometry,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _geometry() -> MeasuredScorerGeometry:
    return MeasuredScorerGeometry(
        metric_id="measured_rank4_pose2_metric_v1",
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


def _minimal_receipt() -> dict:
    return {
        "schema": "ddm_ms2_typed_quotient_solve_repo_receipt.v1",
        "authority": {
            "evidence_axis": EVIDENCE_AXIS,
        },
        "measurement": {
            "finished_at_utc": "2026-07-24T03:00:00Z",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def test_callables_enforce_structural_laws() -> None:
    assert visible_quotient_counted_bytes(
        visible_counted_bytes=17,
        gauge_counted_bytes=0,
    ) == 17
    with pytest.raises(ValueError, match="GAUGE"):
        visible_quotient_counted_bytes(
            visible_counted_bytes=17,
            gauge_counted_bytes=1,
        )

    geometry = _geometry()
    delta = np.array([0.5, -0.25])
    expected = 0.5 * delta @ geometry.second_order_metric @ delta
    assert scorer_metric_rate_action(
        delta,
        geometry=geometry,
        counted_bytes=0,
    ) == pytest.approx(expected)
    exchange = typed_block_exchange_rate(
        score_gain=0.5,
        exact_delta_bytes=4,
        kkt_dual=0.25,
        measured_source_sha256=SHA_A,
    )
    assert exchange["score_gain_per_byte"] == pytest.approx(0.125)
    assert effective_quantum(uint8_step=2.0, scorer_sensitivity=0.25) == 0.5
    assert (
        skeleton_fiber_coder_race(
            skeleton_counted_bytes=10,
            fiber_counted_bytes=11,
            semantic_parseback_exact=True,
        )
        == "SKELETON"
    )
    with pytest.raises(ValueError, match="parse-back"):
        skeleton_fiber_coder_race(
            skeleton_counted_bytes=10,
            fiber_counted_bytes=9,
            semantic_parseback_exact=False,
        )


def test_builders_expose_importable_callables_and_derivation_edges(tmp_path: Path) -> None:
    source = tmp_path / "receipt.json"
    source.write_text(json.dumps(_minimal_receipt()), encoding="utf-8")
    provenance = build_provenance_for_research_sidecar(
        source,
        reactivation_criteria="test-only measured custody",
        measurement_axis=EVIDENCE_AXIS,
        hardware_substrate="darwin_arm64_cpu_torch",
        captured_at_utc="2026-07-24T03:00:00Z",
    )
    equations = build_ddm_ms2_typed_quotient_equations(
        provenance=provenance,
        calibration_utc="2026-07-24T03:00:00Z",
    )
    assert tuple(row.equation_id for row in equations) == EQUATION_IDS
    assert len(derivation_edges()) == 9
    for equation in equations:
        module_name, callable_name = equation.python_callable_module_path.split(":")
        assert callable(getattr(importlib.import_module(module_name), callable_name))
        assert equation.domain_of_validity["score_claim"] is False
        assert "identity-Euclidean verdicts" in equation.domain_of_validity["excluded"]
        assert "pf2r.metric_active_three_formulation_rerun" in (
            equation.canonical_consumers
        )
        assert equation.domain_of_validity["pf2r_blocker_id"] == (
            "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE"
        )


def test_population_is_locked_explicit_and_complete(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_minimal_receipt()), encoding="utf-8")
    registry = tmp_path / "canonical_equations_registry.jsonl"
    lock = tmp_path / "canonical_equations_registry.jsonl.lock"
    equations = populate_ddm_ms2_typed_quotient_equations(
        receipt_path=receipt,
        path=registry,
        lock_path=lock,
        agent="codex",
        subagent_id="ddm_ms2_test",
    )
    assert tuple(row.equation_id for row in equations) == EQUATION_IDS
    events = load_registry_events_lenient(registry)
    assert [event["equation_id"] for event in events] == list(EQUATION_IDS)


def test_live_registry_contains_each_ms2_law_exactly_once() -> None:
    events = load_registry_events_lenient(
        Path(".omx/state/canonical_equations_registry.jsonl")
    )
    for equation_id in EQUATION_IDS:
        matching = [
            event for event in events if event.get("equation_id") == equation_id
        ]
        assert [event["event_type"] for event in matching] == [
            "registered",
            "domain_refined",
            "domain_refined",
        ]
        assert matching[0]["equation_payload"]["provenance"]["source_sha256"] == (
            "9b17c5108e4b8d5a517ecb66276fc0e78162e54b53a9f4d819a48286989b98b6"
        )
        current_domain = matching[-1]["equation_payload"]["domain_of_validity"]
        assert current_domain["current_custody_receipt_sha256"] == (
            "04060edf9834b661f12a9794e50ceadf7dd4ab114baf55a15555537abc71e419"
        )
        assert "pf2r.metric_active_three_formulation_rerun" in (
            current_domain["named_downstream_consumers"]
        )
