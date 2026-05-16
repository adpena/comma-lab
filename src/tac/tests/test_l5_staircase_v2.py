# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.l5_staircase_v2 import (
    PREDICTED_DELTA_BAND,
    SUBJECT_ID,
    L5V2GateEvidence,
    l5_v2_dispatch_readiness,
    l5_v2_prediction_band_payload,
    l5_v2_prediction_band_verdict,
    l5_v2_required_gates,
    l5_v2_research_basis_ids,
    l5_v2_staircase_steps,
)
from tac.optimization.l5_v2_probe_disambiguator import L5V2_PROBE_TOOL_PATH
from tac.optimization.substrate_composition_matrix import (
    build_composition_matrix,
    per_substrate_pareto_rows,
)


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _valid_gate_evidence() -> dict[str, L5V2GateEvidence]:
    return {
        gate.gate_id: L5V2GateEvidence(
            gate_id=gate.gate_id,
            artifact_path=(
                f"experiments/results/time_traveler_l5_v2/{gate.gate_id}.json"
            ),
            artifact_sha256=_sha(index + 1),
            predicate_id=f"l5_v2_{gate.gate_id}_predicate",
            predicate_passed=True,
            evidence_grade="contest_artifact",
        )
        for index, gate in enumerate(l5_v2_required_gates())
    }


def test_l5_v2_research_basis_is_explicit_and_canonical() -> None:
    ids = l5_v2_research_basis_ids()

    assert ids == (
        "rao_ballard_1999",
        "friston_free_energy_2010",
        "dreamerv3_2023",
        "ha_schmidhuber_world_models_2018",
        "rissanen_mdl_1978",
        "mackay_itila_2003",
        "tishby_information_bottleneck_1999",
        "tishby_zaslavsky_2015",
        "balle_hyperprior_2018",
        "hnerv_2023",
    )


def test_l5_v2_staircase_steps_are_ordered_and_fail_closed() -> None:
    steps = l5_v2_staircase_steps()
    gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}

    assert [step.step_id for step in steps] == [
        "l5v2_00_source_and_alias_custody",
        "l5v2_01_sideinfo_consumption_proof",
        "l5v2_02_probe_disambiguator",
        "l5v2_03_paired_axis_anchor",
        "l5v2_04_stack_of_stacks_candidate",
    ]
    for step in steps:
        assert step.research_basis_ids
        assert step.dispatch_allowed is False
        assert step.promotion_eligible is False
        assert set(step.required_gate_ids) <= gate_ids

    probe_step = next(step for step in steps if step.step_id == "l5v2_02_probe_disambiguator")
    assert probe_step.deliverable_surface == L5V2_PROBE_TOOL_PATH
    assert Path(L5V2_PROBE_TOOL_PATH).is_file()


def test_l5_v2_prediction_band_is_source_backed_but_rank_blocked() -> None:
    payload = l5_v2_prediction_band_payload()
    verdict = l5_v2_prediction_band_verdict()

    assert payload["subject_id"] == SUBJECT_ID
    assert (payload["low"], payload["high"]) == PREDICTED_DELTA_BAND
    assert payload["score_claim"] is False
    assert payload["planning_only"] is True
    assert payload["band_source"]["research_basis_ids"] == l5_v2_research_basis_ids()
    assert verdict["valid_for_dispatch_planning"] is True
    assert verdict["valid_for_rank_reward"] is False
    assert "prediction_band_baseline_missing" in verdict["blockers"]
    assert "prediction_band_empirical_anchor_missing" in verdict["blockers"]
    assert "prediction_band_research_basis_missing" not in verdict["blockers"]


def test_l5_v2_dispatch_readiness_requires_artifact_evidence_not_booleans() -> None:
    blocked = l5_v2_dispatch_readiness()
    all_gate_ids = {gate.gate_id for gate in l5_v2_required_gates()}
    boolean_only = l5_v2_dispatch_readiness(dict.fromkeys(all_gate_ids, True))

    assert blocked["ready_for_dispatch"] is False
    assert blocked["all_gate_claims_satisfied"] is False
    assert blocked["all_gate_evidence_valid"] is False
    assert blocked["promotion_eligible"] is False
    assert {
        "requires_byte_closed_temporal_sideinfo_consumption_proof",
        "requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock",
        "requires_paired_cpu_cuda_axis_plan_before_promotion",
        "requires_l5_v2_empirical_anchor",
    } <= set(blocked["blockers"])
    assert any(
        blocker.startswith("l5_v2_gate_evidence_missing:")
        for blocker in blocked["blockers"]
    )
    assert boolean_only["all_gate_claims_satisfied"] is True
    assert boolean_only["all_gate_evidence_valid"] is False
    assert boolean_only["ready_for_dispatch"] is False
    assert boolean_only["promotion_eligible"] is False
    assert boolean_only["score_claim"] is False
    assert all(gate["claimed_satisfied"] is True for gate in boolean_only["gates"])
    assert all(gate["evidence_valid"] is False for gate in boolean_only["gates"])


def test_l5_v2_dispatch_readiness_accepts_valid_gate_evidence() -> None:
    ready = l5_v2_dispatch_readiness(gate_evidence=_valid_gate_evidence())

    assert ready["all_gate_claims_satisfied"] is True
    assert ready["all_gate_evidence_valid"] is True
    assert ready["ready_for_dispatch"] is True
    assert ready["blockers"] == []
    assert ready["promotion_eligible"] is False
    assert ready["score_claim"] is False
    assert all(gate["evidence_valid"] is True for gate in ready["gates"])


@pytest.mark.parametrize(
    ("field", "value", "expected_blocker"),
    [
        ("artifact_path", "", "l5_v2_gate_artifact_path_missing:"),
        ("artifact_path", "/tmp/l5v2.json", "l5_v2_gate_artifact_path_transient:"),
        ("artifact_sha256", "not-a-sha", "l5_v2_gate_artifact_sha256_invalid:"),
        ("predicate_id", "", "l5_v2_gate_predicate_id_missing:"),
        ("predicate_passed", False, "l5_v2_gate_predicate_failed:"),
    ],
)
def test_l5_v2_dispatch_readiness_rejects_bad_gate_evidence(
    field: str,
    value: object,
    expected_blocker: str,
) -> None:
    evidence = {
        gate_id: gate_evidence.__dict__.copy()
        for gate_id, gate_evidence in _valid_gate_evidence().items()
    }
    first_gate_id = next(iter(evidence))
    evidence[first_gate_id][field] = value

    readiness = l5_v2_dispatch_readiness(gate_evidence=evidence)

    assert readiness["all_gate_claims_satisfied"] is False
    assert readiness["all_gate_evidence_valid"] is False
    assert readiness["ready_for_dispatch"] is False
    assert f"{expected_blocker}{first_gate_id}" in readiness["blockers"]


def test_l5_v2_prediction_band_flows_into_composition_row() -> None:
    rows = {
        row.substrate_id: row
        for row in per_substrate_pareto_rows(matrix=build_composition_matrix())
    }
    row = rows[SUBJECT_ID]

    assert row.prediction_band is not None
    assert row.prediction_band["band_id"] == "time_traveler_l5_v2_delta_prior_20260516"
    assert row.prediction_band_verdict is not None
    assert row.prediction_band_verdict["valid_for_rank_reward"] is False
    assert "prediction_band_baseline_missing" in row.dispatch_blockers
    assert "prediction_band_empirical_anchor_missing" in row.dispatch_blockers
