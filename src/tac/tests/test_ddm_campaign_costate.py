"""Adversarial tests for the shared DDM campaign SENSE/DECIDE state."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from tac.ddm_campaign_costate import (  # noqa: E402
    BLOCKER_SCHEMA,
    CLASS_E_DIMENSIONS,
    DECISION_SCHEMA,
    DYNAMIC_POLICY_SCHEMA,
    METRIC_ROW_SCHEMA,
    PLATEAU_FORKS,
    SENSE_ROW_SCHEMA,
    VERDICT_SCHEMA,
    _lambda_ranker_state,
    build_campaign_costate,
    campaign_consumer_view,
    derive_noise_alarm,
    derive_top_k_from_evaluator_bands,
    load_campaign_sources,
    route_plateau,
    validate_realized_verdict,
)
from tac.ddm_campaign_evidence_join import METRIC_ID  # noqa: E402
from tac.ddm_costate_law import RATE_BREAK_EVEN_SCORE_PER_BYTE  # noqa: E402
from tac.ddm_costate_organ import build_live_ddm_costate  # noqa: E402
from tac.witness_dsl.lever_registry import campaign_activation_nag  # noqa: E402


def _raw_verdict(
    verdict_id: str,
    *,
    plateau_type: str,
    score_after: float,
    noise: float,
    band: tuple[float, float],
) -> dict:
    telemetry = {
        **{f"Q{index}": {"status": "fixture"} for index in range(1, 8)},
        "lever_engage": ["fixture"],
        "term_inert": [],
        "liveness": {
            "accepted_batch_fraction": 1.0,
            "weights_stepped": True,
            "frozen": False,
        },
        "S_before": 10.0,
        "S_after": score_after,
        "counted_bytes_before": 100,
        "counted_bytes_after": 103,
        "measured_work": 1.0,
        "g_S": 10.0 - score_after,
        "g_L": -3.0,
        "delta_S_per_wall_clock_hour": (score_after - 10.0) / (18.0 / 3600.0),
        "plateau_type": plateau_type,
        "pose_gate_state": "PASS",
        "noise_sample_delta_S": noise,
        "noise_regime_id": "fixture_receiver_state_v1",
        "evaluator_band_low": band[0],
        "evaluator_band_high": band[1],
        "plateau_residual": {
            "residual_type": plateau_type,
            "metric_id": "exact_composite_R_rank4_Fisher_plus_pose6_quadratic",
            "value": abs(score_after - 10.0),
            "units": "delta_S",
        },
        "candidate_evaluator_bands": [
            {
                "candidate_id": verdict_id,
                "delta_S": score_after - 10.0,
                "evaluator_band": [band[0], band[1]],
            },
            {
                "candidate_id": verdict_id + "_outside",
                "delta_S": band[1] + 1.0,
                "evaluator_band": [band[1] + 0.5, band[1] + 1.5],
            },
        ],
        "alarm_familywise_alpha": 0.05,
        "joint_null_energy": 1.0,
        "seg_only_energy": 2.0,
        "pose_only_energy": 3.0,
        "joint_visible_energy": 4.0,
        "projector_rejected_energy": 5.0,
        "temporal_flicker": 0.25,
        "clip_stationarity": 0.9,
        "delta_bytes_per_step": 3,
        "dribble_rate": 0.5,
    }
    return {
        "schema": VERDICT_SCHEMA,
        "event_id": verdict_id,
        "identity": {"event": "fixture"},
        "telemetry": telemetry,
        "accepted": score_after < 10.0,
        "rollback_reason": None,
    }


def _validated(raw: dict) -> dict:
    return validate_realized_verdict(
        raw,
        source_path=".omx/research/fixture.jsonl",
        source_sha256="a" * 64,
    )


def _campaign(*, verdicts: list[dict] | None = None) -> dict:
    return build_campaign_costate(
        repo_root=REPO,
        verdicts=verdicts,
    )


def test_current_campaign_is_one_hash_lineaged_advisory_truth() -> None:
    state = _campaign()
    assert state["maturity"] == "_dev"
    assert state["actuation"] == "NONE"
    assert state["score_claim"] is False
    assert state["main_landing_review_required"] is True
    assert len(state["state_digest"]) == 64
    assert all(
        row["status"] == "CONTENT_HASH_VERIFIED"
        for row in state["source_lineage"]["sources"].values()
    )

    views = [
        campaign_consumer_view(state, name)
        for name in ("digest", "dashboard", "duty_queue", "activation_nag")
    ]
    assert {view["state_digest"] for view in views} == {state["state_digest"]}
    assert campaign_activation_nag(state)["state_digest"] == state["state_digest"]
    assert views[0]["campaign_evidence"]["v19_receiver_closed_join_status"] == (
        "MEASURED_600_OF_600"
    )
    assert views[1]["campaign_evidence"]["rd1_dimension_evidence_status"] == (
        "MEASURED_162_AMORTIZED_HOMES_(108_SHARED,54_PER_FRAME)_AND_"
        "162_UINT8_HISTOGRAMS"
    )
    ranker = state["metric_state"]["lambda_ranker"]
    assert ranker["selected_model"]["candidate_id"] == (
        "factorized_ms4d_interactions"
    )
    assert ranker["selected_model"]["metrics"]["heldout_only"] is True
    assert ranker["selected_model"]["metrics"]["ndcg_at_4"] == pytest.approx(1.0)
    assert ranker["selected_model"]["metrics"]["spearman_rho"] == pytest.approx(
        0.8607149751465011
    )
    assert ranker["admission_gate"]["duty_ranking_upgrade_eligible"] is True
    assert ranker["pair_precision"]["unranked_precision_owed"] == 585
    assert ranker["pair_precision"]["pair_duty_ranking_status"] == (
        "BLOCKED_INCOMPLETE_FISHER_PRECISION"
    )
    assert all(
        row["actuation"] == "NONE"
        for row in ranker["top_heldout_diagnostics_nonactionable"]
    )
    assert all(view["lambda_ranker"] == ranker for view in views)


def test_lambda_ranker_receipt_tamper_fails_before_consumer_views() -> None:
    sources, payloads = load_campaign_sources(REPO)
    tampered = json.loads(json.dumps(payloads["co3_lambda_ranker"]))
    tampered["selected_model"]["metrics"]["ndcg_at_4"] = 0.0
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        _lambda_ranker_state(tampered, sources["co3_lambda_ranker"])


def test_all_366_class_e_rows_stand_even_before_j8f() -> None:
    state = _campaign()
    rows = state["sense"]["standing_rows"]
    assert {row["row_id"] for row in rows} == {
        name for name, _units, _contract in CLASS_E_DIMENSIONS
    }
    assert len(rows) == 9
    assert all(row["schema"] == SENSE_ROW_SCHEMA for row in rows)
    assert all(row["status"] == "AWAITING_J8F_MEASUREMENT" for row in rows)
    assert all(row["value"] is None for row in rows)
    assert state["consumers"]["activation_nag"]["unmeasured_sense_rows"] == 9
    assert "BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY" in {
        row["blocker_id"] for row in state["blockers"]
    }
    assert "BLOCKED_PAIR_LEVEL_MS4D_FISHER_PRECISION_585" in {
        row["blocker_id"] for row in state["blockers"]
    }


def test_metric_rows_are_scoped_and_bucket_prices_remain_blocked() -> None:
    state = _campaign()
    metric = state["metric_state"]
    assert metric["scorer_metric"] == METRIC_ID
    rows = metric["aggregate_scalarization_controls"]
    assert len(rows) == 3
    assert all(row["schema"] == METRIC_ROW_SCHEMA for row in rows)
    assert metric["bucket_exchange_rate_status"] == (
        "EVIDENCE_MEASURED_162_OF_162; "
        "PRICING_PENDING_MS2R_0_OF_162_ACTIONABLE"
    )
    assert metric["v19_receiver_closed_join_status"] == "MEASURED_600_OF_600"
    assert metric["rd1_dimension_evidence_status"] == (
        "MEASURED_162_AMORTIZED_HOMES_(108_SHARED,54_PER_FRAME)_AND_"
        "162_UINT8_HISTOGRAMS"
    )
    bucket_rows = metric["bucket_exchange_rates"]
    assert len(bucket_rows) == 162
    assert all(row["schema"] == METRIC_ROW_SCHEMA for row in bucket_rows)
    assert all(row["lambda_score_per_byte"] is None for row in bucket_rows)
    assert all(row["actionable_for_train_decision"] is False for row in bucket_rows)
    assert all(
        row["evidence_status"]
        == "MEASURED_RECEIVER_CLOSED_AMORTIZED_HOME_AND_HISTOGRAM"
        for row in bucket_rows
    )
    assert sum(row["byte_home_scope"] != "per_frame" for row in bucket_rows) == 108
    assert sum(row["byte_home_scope"] == "per_frame" for row in bucket_rows) == 54
    assert all(row["byte_home_k"] >= 1.0 for row in bucket_rows)
    assert all(row["amortized_bytes_per_frame"] >= 0.0 for row in bucket_rows)
    assert all(len(row["receiver_uint8_abs_step_histogram"]) == 256 for row in bucket_rows)
    assert all(
        row["status"]
        == "DERIVED_FROM_EV1_FRESH_N600_ENDPOINTS_NONADDITIVE_CONTROL"
        for row in rows
    )
    assert rows[0]["lambda_score_per_byte"] == pytest.approx(
        RATE_BREAK_EVEN_SCORE_PER_BYTE
        - rows[0]["lambda_distortion_reduction_per_byte"]
    )
    assert "BLOCKED_RD1_CANDIDATE_DELTA_G4_DIMENSION_BYTE_HOME" not in {
        row["blocker_id"] for row in state["blockers"]
    }
    assert all(row["schema"] == BLOCKER_SCHEMA for row in state["blockers"])
    assert state["decide"]["plateau_route"]["schema"] == DECISION_SCHEMA
    assert state["dynamic_policy"]["schema"] == DYNAMIC_POLICY_SCHEMA


def test_v19_gap_is_closed_by_exact_receiver_rows_and_shared_rate_home() -> None:
    state = _campaign()
    assert not any(
        row["blocker_id"].startswith("BLOCKED_RECEIVER_CLOSED_V19_EVIDENCE_JOIN_")
        for row in state["blockers"]
    )
    evidence = state["source_lineage"]["sources"]["ev1_campaign_evidence_join"]
    assert evidence["status"] == "CONTENT_HASH_VERIFIED"
    assert state["consumers"]["duty_queue"]["rows"][-1] == {
        "duty": "MS2R_TOLERANCE_CAPPED_DIMENSION_PRICING",
        "reason": (
            "EV1 measured 162/162 exclusive homes (108 shared across frames) "
            "and receiver histograms; "
            "ms2r owns the 0/162 priced solve"
        ),
        "actuation": "NONE",
        "rank": 3,
    }
    assert state["consumers"]["duty_queue"]["rows"][1] == {
        "duty": "CO3_LAMBDA_RANKER_FISHER_PRECISION_CLOSURE",
        "reason": (
            "held-out NDCG@4=1 admits the ranker, but "
            "585/600 pair-level Fisher intervals remain owed"
        ),
        "actuation": "NONE",
        "rank": 2,
    }


@pytest.mark.parametrize(
    ("plateau_type", "fork_id", "formulation"),
    [
        (row["plateau_type"], row["fork_id"], row["formulation"])
        for row in PLATEAU_FORKS
    ],
)
def test_every_feed603_plateau_routes_exactly(
    plateau_type: str,
    fork_id: str,
    formulation: str,
) -> None:
    decision = route_plateau(plateau_type)
    assert decision["fork_id"] == fork_id
    assert decision["formulation"] == formulation
    assert decision["actuation"] == "NONE"


def test_unknown_or_absent_plateau_never_fires() -> None:
    assert route_plateau(None)["status"] == "AWAITING_MEASURED_PLATEAU"
    unknown = route_plateau("made_up")
    assert unknown["status"] == "BLOCKED_UNKNOWN_PLATEAU_TYPE"
    assert unknown["fork_id"] is None


def test_dynamic_alarm_is_derived_from_measured_sigma_and_preregistered_alpha() -> None:
    blocked = derive_noise_alarm([0.1], familywise_alpha=0.05)
    assert blocked["threshold_abs_delta_S"] is None
    derived = derive_noise_alarm([-0.2, -0.1, 0.1, 0.2], familywise_alpha=0.05)
    assert derived["status"] == "DERIVED_FROM_MEASURED_NOISE_FLOOR"
    assert derived["k"] > 0.0
    assert derived["threshold_abs_delta_S"] == pytest.approx(
        derived["k"] * derived["sigma_delta_S"]
    )
    with pytest.raises(ValueError, match="familywise_alpha"):
        derive_noise_alarm([0.0, 0.1], familywise_alpha=1.0)


def test_top_k_is_band_overlap_not_a_literal() -> None:
    result = derive_top_k_from_evaluator_bands(
        [
            {"candidate_id": "best", "delta_S": -0.3, "evaluator_band": [-0.5, -0.1]},
            {"candidate_id": "tie", "delta_S": -0.2, "evaluator_band": [-0.2, 0.0]},
            {"candidate_id": "separate", "delta_S": 0.4, "evaluator_band": [0.2, 0.6]},
        ]
    )
    assert result["top_k"] == 2
    assert result["candidate_ids"] == ["best", "tie"]


def test_verdict_requires_every_366_dimension_and_exact_byte_identity() -> None:
    raw = _raw_verdict(
        "v0",
        plateau_type="GRAMMAR_UNREACHABLE",
        score_after=9.5,
        noise=-0.01,
        band=(-0.6, -0.4),
    )
    verdict = _validated(raw)
    assert verdict["delta_S"] == pytest.approx(-0.5)
    assert verdict["delta_S_per_wall_clock_hour"] == pytest.approx(-100.0)
    assert verdict["dimensions"]["joint_visible_energy"] == 4.0

    missing = json.loads(json.dumps(raw))
    missing["telemetry"].pop("projector_rejected_energy")
    with pytest.raises(ValueError, match="projector_rejected_energy"):
        _validated(missing)
    wrong_bytes = json.loads(json.dumps(raw))
    wrong_bytes["telemetry"]["delta_bytes_per_step"] = 2
    with pytest.raises(ValueError, match="counted-byte identity"):
        _validated(wrong_bytes)


def test_realized_verdict_populates_sense_and_routes_plateau() -> None:
    rows = [
        _validated(
            _raw_verdict(
                "v0",
                plateau_type="DESCENDING",
                score_after=9.8,
                noise=-0.01,
                band=(-0.3, -0.1),
            )
        ),
        _validated(
            _raw_verdict(
                "v1",
                plateau_type="GRAMMAR_UNREACHABLE",
                score_after=9.7,
                noise=0.01,
                band=(-0.4, -0.2),
            )
        ),
    ]
    state = _campaign(verdicts=rows)
    assert state["sense"]["verdict_count"] == 2
    assert all(row["status"] == "MEASURED_J8F_REALIZED" for row in state["sense"]["standing_rows"])
    assert state["decide"]["plateau_route"]["fork_id"] == "F5"
    assert state["decide"]["plateau_route"]["trigger_evidence"]["units"] == "delta_S"
    assert state["consumers"]["duty_queue"]["rows"][0]["fork_id"] == "F5"
    assert state["dynamic_policy"]["noise_alarm"]["status"] == (
        "DERIVED_FROM_MEASURED_NOISE_FLOOR"
    )
    assert state["dynamic_policy"]["top_k"]["top_k"] == 1


def test_organ_digest_dashboard_and_digest_tool_share_campaign_digest() -> None:
    organ = build_live_ddm_costate(repo_root=REPO)
    state_digest = organ["campaign"]["state_digest"]

    import costate_digest
    import dashboard_server

    dashboard_server._DDM_CAMPAIGN_CACHE.clear()
    dashboard = dashboard_server._read_ddm_campaign()
    assert dashboard["state_digest"] == state_digest
    lines, data = costate_digest.build_digest(include_fm=False)
    assert data["ddm_campaign"]["state_digest"] == state_digest
    assert data["ddm_campaign_activation_nag"]["state_digest"] == state_digest
    assert data["duty_to_measure"]["state_digest"] == state_digest
    assert dashboard["lambda_ranker"] == data["ddm_campaign"]["lambda_ranker"]
    assert data["ddm_campaign_activation_nag"]["lambda_ranker_admission"][
        "passed"
    ] is True
    assert data["duty_to_measure"]["lambda_ranker"]["pair_precision"][
        "unranked_precision_owed"
    ] == 585
    assert any(line.startswith("DDM-campaign:") for line in lines)


def test_no_launcher_provider_or_subprocess_in_campaign_module() -> None:
    source = (REPO / "src/tac/ddm_campaign_costate.py").read_text()
    tree_words = set(source.replace("(", " ").replace(")", " ").split())
    assert "subprocess" not in tree_words
    assert "modal" not in tree_words
    assert "launcher" not in tree_words


def test_dashboard_snapshot_exposes_global_campaign_without_mutation() -> None:
    import dashboard_server

    dashboard_server._DDM_CAMPAIGN_CACHE.clear()
    with pytest.MonkeyPatch.context() as monkeypatch:
        # The campaign path is independent of the legacy run-local shadow file.
        monkeypatch.setenv("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))
        view = dashboard_server._read_ddm_campaign()
    assert view["ok"] is True
    assert view["actuation"] == "NONE"
    assert view["campaign_evidence"]["v19_receiver_closed_join_status"] == (
        "MEASURED_600_OF_600"
    )
    json.dumps(view, sort_keys=True, allow_nan=False)


def test_dashboard_campaign_cache_is_receipt_signature_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dashboard_server

    import tac.ddm_costate_organ as organ_module

    dashboard_server._DDM_CAMPAIGN_CACHE.clear()
    first = dashboard_server._read_ddm_campaign()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unchanged receipt signature must not rebuild the organ")

    monkeypatch.setattr(organ_module, "build_live_ddm_costate", forbidden)
    second = dashboard_server._read_ddm_campaign()
    assert second["state_digest"] == first["state_digest"]
