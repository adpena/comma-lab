# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools import probe_ddm_a1_column_generated_correction as probe
from tools.probe_ddm_a1_column_generated_correction import (
    FIXED_BUDGETS,
    FIXED_CODER_ENTRANTS,
    FIXED_FAMILIES,
    FIXED_SELECTORS,
    DDMA1ColumnGeneratedCorrectionConfigV1,
    _blocked_equal_byte_rows,
    _pricing_history,
    _producer_custody,
    evaluate_source_closure,
)


def _config() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema": "DDMA1ColumnGeneratedCorrectionConfigV1",
        "run_id": "fixture-run",
        "source_v12_receipt": "v12.json",
        "source_v12_receipt_sha256": digest,
        "source_v12_archive": "v12.zip",
        "source_v12_archive_sha256": digest,
        "grammar_receipt": "g1.json",
        "grammar_receipt_sha256": digest,
        "source_v15_receipt": "v15.json",
        "source_v15_receipt_sha256": digest,
        "source_v15_archive": "v15.zip",
        "source_v15_archive_sha256": digest,
        "source_v16_receipt": "v16.json",
        "source_v16_receipt_sha256": digest,
        "pair_start": 448,
        "pair_count": 64,
        "column_families": list(FIXED_FAMILIES),
        "pricing_metric": "realized_joint_objective_reduced_cost",
        "selection_modes": list(FIXED_SELECTORS),
        "coder_entrants": list(FIXED_CODER_ENTRANTS),
        "coder_comparison_rule": "matched_realized_d_seg_minimum_exact_bytes",
        "maximum_pricing_rounds": 3,
        "maximum_new_columns_per_round": 64,
        "added_byte_budgets": list(FIXED_BUDGETS),
        "exact_replay_after_each_selected_set": True,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
    }


def test_typed_config_accepts_only_preregistered_surface() -> None:
    config = DDMA1ColumnGeneratedCorrectionConfigV1.model_validate(_config())
    assert config.added_byte_budgets == FIXED_BUDGETS
    assert config.coder_entrants == FIXED_CODER_ENTRANTS
    changed = _config()
    changed["added_byte_budgets"] = [1, 2, 3, 4]
    with pytest.raises(ValidationError, match="rungs"):
        DDMA1ColumnGeneratedCorrectionConfigV1.model_validate(changed)


def test_typed_config_rejects_dropped_structured_coder_entrant() -> None:
    changed = _config()
    changed["coder_entrants"] = ["unstructured_explicit_indices"]
    with pytest.raises(ValidationError, match="coder entrants"):
        DDMA1ColumnGeneratedCorrectionConfigV1.model_validate(changed)


def test_producer_custody_is_content_bound() -> None:
    custody = _producer_custody()
    assert len(custody) == 4
    assert all(row["bytes"] > 0 for row in custody)
    assert all(len(row["sha256"]) == 64 for row in custody)
    assert custody[0]["path"] == "tools/probe_ddm_a1_column_generated_correction.py"


def test_existing_receipt_revalidation_refuses_stale_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DDMA1ColumnGeneratedCorrectionConfigV1.model_validate(_config())
    custody = _producer_custody()
    receipt = {
        "schema": probe.SCHEMA,
        "verdict": probe.VERDICT,
        "typed_config_sha256": config.typed_config_hash(),
        "producer_custody": custody,
    }
    monkeypatch.setattr(probe, "_bound_bytes", lambda *_args: b"bound")
    probe._revalidate_existing_receipt(receipt, config)
    stale = dict(receipt)
    stale["producer_custody"] = [{**custody[0], "sha256": "0" * 64}, *custody[1:]]
    with pytest.raises(probe.DirectDescriptionError, match="producer custody"):
        probe._revalidate_existing_receipt(stale, config)


def test_source_closure_names_all_noncomparable_surfaces() -> None:
    blockers = evaluate_source_closure(
        v12={
            "schema": "direct_description_v12_obligation_drain_receipt.v1",
            "score_claim": False,
            "pointer_moved": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        },
        grammar={
            "schema": "ddm_g1_grammar_induction_compact_receipt.v1",
            "candidate_archive": False,
            "coverage_projection": {
                "receiver_closed": False,
                "pose_measured": False,
            },
            "verdict_scope": "mask-only",
        },
        v15={
            "schema": "ddm_v15_scorer_solved_template_receipt.v1",
            "score_claim": False,
            "inherited_control": {"d_seg": "0.027470296224"},
        },
        v16={
            "schema": "ddm_v16_coupled_joint_solve_receipt.v1",
            "conditionals": {"linearization_invalid": True},
            "fork": {"case": "C"},
        },
        v12_has_realization_profile=False,
        hybrid_compile_error="v13 PREDICT productions cannot be mixed",
    )
    codes = {row["code"] for row in blockers}
    assert codes == {
        "V12_CONTROL_NOT_CAMERA_UINT8_R",
        "G1_COORDINATES_NOT_RECEIVER_CLOSED",
        "V15_TEMPLATE_DOF_NOT_MEASURED_AT_V12_OPERATING_POINT",
        "V16_LINEARIZATION_INVALID_AT_SOURCE",
        "NO_COMMON_HYBRID_ARCHIVE_SCHEMA",
    }


def test_blocked_rows_cannot_trigger_falsifier_by_shape() -> None:
    rows = _blocked_equal_byte_rows()
    pricing = _pricing_history()
    assert [row["added_byte_budget"] for row in rows] == list(FIXED_BUDGETS)
    assert all(row["generated_vocabulary"]["d_seg"] is None for row in rows)
    assert all(row["exact_replay_complete"] is False for row in rows)
    assert all(row["complete"] is False for row in pricing)
    assert all(row["negative_reduced_cost_count"] is None for row in pricing)
