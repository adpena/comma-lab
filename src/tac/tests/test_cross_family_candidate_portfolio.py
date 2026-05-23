# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.cross_family_candidate_portfolio import (
    CrossFamilyCandidatePortfolioError,
    build_cross_family_candidate_portfolio,
)
from tac.repo_io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _mlx_selection() -> dict[str, object]:
    return {
        "schema": "mlx_effective_spend_triage_candidate_selection.v1",
        **_false_authority(),
        "selected_rows": [
            {
                "schema": "mlx_effective_spend_triage_candidate_row.v1",
                **_false_authority(),
                "candidate_id": "mlx_scorer_response:window:501:502",
                "family": "mlx_decoder_q",
                "rank": 1,
                "observed_delta_vs_baseline_score": -0.002,
                "projected_full_video_delta_vs_baseline_score": -0.00002,
                "full_video_denominator": 600,
                "normalized_full_video_scorer_gain_vs_baseline": 0.00003,
                "normalized_full_video_byte_budget_margin_vs_break_even": 25.0,
                "predicted_delta_vs_baseline_score": 0.0004,
                "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
                "selection_basis": "normalized_full_video_mlx_singleton_response_gain",
                "pair_indices": [501, 502],
                "byte_budget_margin_vs_break_even": 2500.0,
                "requires_exact_auth_eval_before_score_claim": True,
            }
        ],
    }


def _pairset_acquisition() -> dict[str, object]:
    return {
        "schema": "decoder_q_pairset_acquisition.v1",
        **_false_authority(),
        "dispatch_attempted": False,
        "candidates": [
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_diversity_k002",
                "acquisition_rank": 1,
                "selector_kind": "diversity_spaced",
                "selected_pair_count": 2,
                "selected_pair_indices": [26, 588],
                "payload_bytes": 14,
                "rate_delta": 0.00001,
                "predicted_score_mean": 0.192028948816,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            }
        ],
    }


def _hfv2_manifest(archive: Path) -> dict[str, object]:
    archive.write_bytes(b"byte closed hfv2 archive")
    return {
        "schema": "hfv1_to_hfv2_sparse_sidecar_candidate_v1",
        **_false_authority(),
        "output_submission_archive": archive.as_posix(),
        "output_archive_bytes": archive.stat().st_size,
        "output_archive_sha256": sha256_file(archive),
        "rate_delta_vs_baseline_archive": 0.0003,
        "row_parity_exact": True,
        "sparse_pair_count": 16,
        "dispatch_blockers": [
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
        ],
        "target_modes": ["contest_exact_eval"],
    }


def _contest_observation(
    tmp_path: Path,
    *,
    score: float = 0.193,
    candidate_id: str = "pairset_diversity_k002",
    archive_char: str = "a",
    raw_char: str = "b",
    selected_pair_indices: list[int] | None = None,
    segnet_delta: float = 0.0001,
    posenet_delta: float = 0.0,
    rate_delta: float = -0.00001,
    axis: str = "contest_cpu",
    evidence_grade: str = "contest-CPU",
    evidence_tag: str = "[contest-CPU]",
) -> dict[str, object]:
    source = tmp_path / f"{candidate_id}_{axis}_{archive_char}_contest_auth_eval.json"
    archive_sha = archive_char * 64
    raw_sha = raw_char * 64
    source.write_text(
        json.dumps(
            {
                "score_axis": axis,
                "evidence_grade": evidence_grade,
                "score_claim_valid": True,
                "canonical_score": score,
                "provenance": {
                    "archive_sha256": archive_sha,
                    "inflated_output_manifest": {
                        "payload": {"aggregate_sha256": raw_sha}
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    row: dict[str, object] = {
        "schema": "mlx_dynamic_sweep_observation.v1",
        "candidate_id": candidate_id,
        "sweep_config_id": "contest_cpu_exact_candidate",
        "optimization_pass_id": "exact_cpu_calibration",
        "family": "decoder_q_pairset_diversity",
        "observed_axis": axis,
        "evidence_tag": evidence_tag,
        "evidence_grade": evidence_grade,
        "observed_score_or_delta": score,
        "archive_sha256": archive_sha,
        "runtime_sha256": "c" * 64,
        "raw_output_or_cache_sha256": raw_sha,
        "component_deltas": {
            "segnet_delta": segnet_delta,
            "posenet_delta": posenet_delta,
            "rate_delta": rate_delta,
        },
        "source_artifact_path": source.as_posix(),
    }
    if selected_pair_indices is not None:
        row["selected_pair_indices"] = selected_pair_indices
    return row


def _pairset_acquisition_with_component_candidates() -> dict[str, object]:
    base_pairs = [101, 327, 371, 376]
    return {
        "schema": "decoder_q_pairset_acquisition.v1",
        **_false_authority(),
        "dispatch_attempted": False,
        "candidates": [
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_drop_one_rank002_pair0327",
                "acquisition_rank": 1,
                "selector_kind": "drop_one_from_best",
                "selected_pair_count": 3,
                "selected_pair_indices": [101, 371, 376],
                "payload_bytes": 40,
                "rate_delta": 0.00001,
                "acquisition_score": 0.5,
                "acquisition_operation": {
                    "op": "drop_one",
                    "dropped_pair_index": 327,
                    "dropped_pair_rank": 2,
                },
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_drop_one_rank003_pair0371",
                "acquisition_rank": 2,
                "selector_kind": "drop_one_from_best",
                "selected_pair_count": 3,
                "selected_pair_indices": [101, 327, 376],
                "payload_bytes": 40,
                "rate_delta": 0.00001,
                "acquisition_score": 0.5,
                "acquisition_operation": {
                    "op": "drop_one",
                    "dropped_pair_index": 371,
                    "dropped_pair_rank": 3,
                },
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_drop_one_rank004_pair0376",
                "acquisition_rank": 3,
                "selector_kind": "drop_one_from_best",
                "selected_pair_count": 3,
                "selected_pair_indices": [101, 327, 371],
                "payload_bytes": 40,
                "rate_delta": 0.00001,
                "acquisition_score": 0.5,
                "acquisition_operation": {
                    "op": "drop_one",
                    "dropped_pair_index": 376,
                    "dropped_pair_rank": 4,
                },
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_prefix_k004",
                "acquisition_rank": 4,
                "selector_kind": "prefix_variant",
                "selected_pair_count": 4,
                "selected_pair_indices": base_pairs,
                "payload_bytes": 41,
                "rate_delta": 0.000011,
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
        ],
    }


def _pairset_acquisition_with_response_candidates() -> dict[str, object]:
    return {
        "schema": "decoder_q_pairset_acquisition.v1",
        **_false_authority(),
        "dispatch_attempted": False,
        "candidates": [
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_diversity_k002",
                "acquisition_rank": 1,
                "selector_kind": "diversity_spaced",
                "selected_pair_count": 2,
                "selected_pair_indices": [26, 588],
                "payload_bytes": 14,
                "rate_delta": 0.00001,
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_diversity_k004",
                "acquisition_rank": 2,
                "selector_kind": "diversity_spaced",
                "selected_pair_count": 4,
                "selected_pair_indices": [26, 109, 501, 588],
                "payload_bytes": 17,
                "rate_delta": 0.000011,
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_diversity_k008",
                "acquisition_rank": 3,
                "selector_kind": "diversity_spaced",
                "selected_pair_count": 8,
                "selected_pair_indices": [26, 109, 229, 296, 378, 459, 501, 588],
                "payload_bytes": 21,
                "rate_delta": 0.000012,
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_prefix_k008",
                "acquisition_rank": 4,
                "selector_kind": "prefix_variant",
                "selected_pair_count": 8,
                "selected_pair_indices": [59, 68, 98, 109, 134, 376, 479, 501],
                "payload_bytes": 21,
                "rate_delta": 0.000012,
                "predicted_score_mean": 0.195,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            },
        ],
    }


def test_portfolio_fuses_mlx_pairset_and_outside_class_without_authority(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.192051316881,
        mlx_selections=[_mlx_selection()],
        pairset_acquisitions=[_pairset_acquisition()],
        hfv2_manifests=[_hfv2_manifest(tmp_path / "archive.zip")],
        top_k=8,
    )

    assert portfolio["schema"] == "cross_family_candidate_portfolio.v1"
    assert portfolio["score_claim"] is False
    assert portfolio["ready_for_exact_eval_dispatch"] is False
    assert portfolio["cross_family_policy"]["outside_class_allowed"] is True
    assert portfolio["portfolio_summary"]["candidate_count_before_top_k"] == 3
    assert portfolio["portfolio_summary"]["operator_action_candidate_count"] == 3
    assert portfolio["portfolio_summary"]["recommended_next_candidate_id"] == (
        "pairset_diversity_k002"
    )
    assert portfolio["portfolio_summary"]["recommended_next_action"] == (
        "materialize_pairset_archive_and_run_local_controls"
    )
    assert portfolio["portfolio_summary"]["candidate_archive_custody_ready_count"] == 1
    assert portfolio["portfolio_summary"]["source_counts"] == {
        "decoder_q_pairset_acquisition": 1,
        "hfv2_sparse_sidecar_manifest": 1,
        "mlx_effective_spend_triage_selection": 1,
    }
    assert {row["family_id"] for row in portfolio["ranked_rows"]} == {
        "decoder_q_selective_dqs1",
        "hfv2_sparse_sidecar",
        "mlx_decoder_q",
    }
    assert all(row["score_claim"] is False for row in portfolio["ranked_rows"])
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in portfolio["ranked_rows"]
    )
    assert all(
        "portfolio_planning_only_requires_separate_lane_claim"
        in row["dispatch_blockers"]
        for row in portfolio["ranked_rows"]
    )
    assert portfolio["operator_action_rows"][0]["candidate_id"] == "pairset_diversity_k002"
    assert portfolio["operator_action_rows"][0]["operator_action_rank"] == 1
    hfv2 = next(row for row in portfolio["ranked_rows"] if row["family_id"] == "hfv2_sparse_sidecar")
    assert "exact_contest_cpu_eval_missing" in hfv2["dispatch_blockers"]


def test_portfolio_demotes_already_observed_exact_axis_candidate(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.192051316881,
        mlx_selections=[_mlx_selection()],
        pairset_acquisitions=[_pairset_acquisition()],
        hfv2_manifests=[_hfv2_manifest(tmp_path / "archive.zip")],
        observations=[_contest_observation(tmp_path)],
        incumbent_scores_by_axis={"contest_cpu": 0.192051316881},
        top_k=8,
    )

    assert portfolio["portfolio_summary"]["observation_row_count"] == 1
    assert portfolio["portfolio_summary"]["observed_candidate_count"] == 1
    assert portfolio["portfolio_summary"]["recommended_next_candidate_id"] != (
        "pairset_diversity_k002"
    )
    assert portfolio["operator_action_rows"][0]["source_kind"] == (
        "mlx_effective_spend_triage_selection"
    )
    pairset = next(
        row
        for row in portfolio["operator_action_rows"]
        if row["candidate_id"] == "pairset_diversity_k002"
    )
    assert pairset["source_metadata"]["observation_feedback"]["status"] == (
        "observed_exact_axis_regressed_vs_axis_baseline"
    )
    assert (
        "candidate_already_observed_contest_cpu_do_not_repeat_same_axis"
        in pairset["dispatch_blockers"]
    )
    assert (
        "candidate_observed_contest_cpu_regressed_vs_axis_baseline"
        in pairset["dispatch_blockers"]
    )


def test_portfolio_does_not_compare_cpu_observation_to_cuda_incumbent(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.191,
        mlx_selections=[_mlx_selection()],
        pairset_acquisitions=[_pairset_acquisition()],
        observations=[_contest_observation(tmp_path, score=0.193)],
        top_k=8,
    )

    pairset = next(
        row
        for row in portfolio["operator_action_rows"]
        if row["candidate_id"] == "pairset_diversity_k002"
    )
    feedback = pairset["source_metadata"]["observation_feedback"]
    assert feedback["status"] == "observed_exact_axis_without_axis_baseline"
    assert feedback["axis_baseline_available"] is False
    assert "observed_delta_vs_axis_baseline" not in feedback
    assert (
        "candidate_observed_contest_cpu_regressed_vs_axis_baseline"
        not in pairset["dispatch_blockers"]
    )


def test_portfolio_uses_exact_pairset_observations_as_planning_prior(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.195,
        pairset_acquisitions=[_pairset_acquisition_with_response_candidates()],
        observations=[
            _contest_observation(
                tmp_path,
                candidate_id="pairset_diversity_k002",
                score=0.193,
                archive_char="a",
                raw_char="b",
                selected_pair_indices=[26, 588],
            ),
            _contest_observation(
                tmp_path,
                candidate_id="pairset_diversity_k004",
                score=0.1928,
                archive_char="d",
                raw_char="e",
                selected_pair_indices=[26, 109, 501, 588],
            ),
        ],
        incumbent_scores_by_axis={"contest_cpu": 0.1927},
        top_k=8,
    )

    model = portfolio["observation_feedback"]["pairset_observation_response_model"]
    assert model["active"] is True
    assert model["axis"] == "contest_cpu"
    assert model["training_row_count"] == 2
    assert model["observed_selected_pair_counts"] == [2, 4]
    assert model["active_selector_kinds"] == ["diversity_spaced"]
    assert model["regression_only_cap_active"] is True
    assert model["score_claim"] is False

    unobserved = next(
        row
        for row in portfolio["ranked_rows"]
        if row["candidate_id"] == "pairset_diversity_k008"
    )
    row_model = unobserved["source_metadata"]["observation_response_model"]
    assert row_model["active"] is True
    assert unobserved["prediction_source"] == (
        "exact_pairset_observation_response_model_planning_prior"
    )
    assert unobserved["predicted_score_mean"] < 0.195
    assert unobserved["predicted_score_mean"] >= 0.1928
    assert unobserved["predicted_score_variance"] >= 2.5e-9
    assert unobserved["score_claim"] is False
    assert unobserved["ready_for_exact_eval_dispatch"] is False
    prefix = next(
        row
        for row in portfolio["ranked_rows"]
        if row["candidate_id"] == "pairset_prefix_k008"
    )
    assert "prediction_source" not in prefix
    prefix_model = prefix["source_metadata"]["observation_response_model"]
    assert prefix_model["active"] is False
    assert prefix_model["inactive_reason"] == "no_exact_observations_for_selector_kind"


def test_pairset_response_model_requires_selected_pair_identity(tmp_path: Path) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.195,
        pairset_acquisitions=[_pairset_acquisition_with_response_candidates()],
        observations=[
            _contest_observation(
                tmp_path,
                candidate_id="pairset_diversity_k002",
                score=0.193,
                archive_char="a",
                raw_char="b",
            ),
            _contest_observation(
                tmp_path,
                candidate_id="pairset_diversity_k004",
                score=0.1928,
                archive_char="d",
                raw_char="e",
            ),
        ],
        incumbent_scores_by_axis={"contest_cpu": 0.1927},
        top_k=8,
    )

    model = portfolio["observation_feedback"]["pairset_observation_response_model"]
    assert model["active"] is False
    assert model["identity_counts"]["selected_pair_indices_missing_observation_count"] == 2
    assert model["inactive_reason"] == "need_two_distinct_selected_pair_counts_per_selector_kind"
    unobserved = next(
        row
        for row in portfolio["ranked_rows"]
        if row["candidate_id"] == "pairset_diversity_k008"
    )
    assert "prediction_source" not in unobserved


def test_portfolio_builds_component_marginal_model_and_axis_transfer_diagnostics(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.226190435402,
        pairset_acquisitions=[_pairset_acquisition_with_component_candidates()],
        observations=[
            _contest_observation(
                tmp_path,
                candidate_id="pairset_drop_one_rank002_pair0327",
                score=0.19202928,
                archive_char="a",
                raw_char="b",
                selected_pair_indices=[101, 371, 376],
                segnet_delta=0.000001,
                rate_delta=-0.00000066585895312,
            ),
            _contest_observation(
                tmp_path,
                candidate_id="pairset_drop_one_rank003_pair0371",
                score=0.19202828,
                archive_char="d",
                raw_char="e",
                selected_pair_indices=[101, 327, 376],
                segnet_delta=0.0,
                rate_delta=-0.00000066585895312,
            ),
            _contest_observation(
                tmp_path,
                candidate_id="pairset_drop_one_rank004_pair0376",
                score=0.19202928,
                archive_char="1",
                raw_char="2",
                selected_pair_indices=[101, 327, 371],
                segnet_delta=0.000001,
                rate_delta=-0.00000066585895312,
            ),
            _contest_observation(
                tmp_path,
                candidate_id="pairset_drop_one_rank003_pair0371",
                score=0.22619177,
                archive_char="f",
                raw_char="0",
                selected_pair_indices=[101, 327, 376],
                segnet_delta=0.000002,
                rate_delta=-0.00000066585895312,
                axis="contest_cuda",
                evidence_grade="contest-CUDA",
                evidence_tag="[contest-CUDA]",
            ),
        ],
        incumbent_scores_by_axis={
            "contest_cpu": 0.192028948816,
            "contest_cuda": 0.226190435402,
        },
        top_k=8,
    )

    model = portfolio["observation_feedback"]["pairset_component_marginal_model"]
    assert model["active"] is True
    assert model["score_claim"] is False
    assert model["ready_for_exact_eval_dispatch"] is False
    assert model["canonical_signal_refs"]["xray_primitives"][0] == (
        "pairset_component_marginal"
    )
    assert (
        "pairset_component_marginal_score_decomposition_v1"
        in model["canonical_signal_refs"]["canonical_equations"]
    )
    assert model["axes"] == ["contest_cpu", "contest_cuda"]
    cpu_model = model["axis_models"]["contest_cpu"]
    assert cpu_model["safe_drop_pair_indices"] == [371]
    assert cpu_model["protected_drop_pair_indices"] == [327, 376]
    cpu_payload = cpu_model["drop_one_pair_marginals"][0][
        "component_score_delta_payload"
    ]
    assert cpu_payload["schema"] == "pairset_component_marginal_score_delta.v1"
    assert cpu_payload["score_claim"] is False
    transfer = model["cross_axis_transfer_diagnostics"][0]
    assert transfer["candidate_id"] == "pairset_drop_one_rank003_pair0371"
    assert transfer["transfer_status"] == "cpu_improves_cuda_regresses"
    assert transfer["score_claim"] is False

    unobserved = next(
        row
        for row in portfolio["ranked_rows"]
        if row["candidate_id"] == "pairset_drop_one_rank004_pair0376"
    )
    feedback = unobserved["source_metadata"]["component_marginal_model"]
    assert feedback["active"] is True
    nearest_cpu = feedback["nearest_drop_one_evidence_by_axis"]["contest_cpu"]
    assert nearest_cpu["source_pair_index"] == 376
    assert nearest_cpu["source_component_marginal_status"] == (
        "scorer_penalty_exceeds_rate_credit"
    )
    action_prior = feedback["component_marginal_action_prior"]
    assert action_prior["schema"] == "pairset_component_marginal_action_prior.v1"
    assert action_prior["primary_axis"] == "contest_cpu"
    assert action_prior["primary_axis_expected_net_component_delta"] > 0.0
    assert action_prior["primary_axis_expected_component_marginal_status"] == (
        "scorer_penalty_expected_to_exceed_rate_credit"
    )
    assert "scorer_penalty_exceeds_rate_credit" in (
        action_prior["axis_priors"]["contest_cpu"][
            "same_pair_observed_component_marginal_statuses"
        ]
    )
    assert "component_marginal_exact_axis_protected_pair" in (
        action_prior["planning_blockers"]
    )
    assert "component_marginal_exact_axis_protected_pair" in (
        unobserved["source_dispatch_blockers"]
    )
    assert (
        "tac.master_gradient_consumers.per_pair_difficulty_atlas"
        in feedback["canonical_signal_refs"]["master_gradient_consumers"]
    )
    assert feedback["score_claim"] is False
    assert (
        portfolio["operator_action_rows"][0]["candidate_id"]
        != "pairset_drop_one_rank004_pair0376"
    )


def test_portfolio_preserves_custody_readiness_as_advisory_only(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.2,
        hfv2_manifests=[_hfv2_manifest(tmp_path / "archive.zip")],
    )
    row = portfolio["ranked_rows"][0]

    assert row["exact_archive_custody"]["verified"] is True
    assert row["bayesian_ready_for_exact_eval_dispatch"] is True
    assert row["exact_archive_custody_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "auth_axis_gate_required_before_dispatch" in row["dispatch_blockers"]
    assert "exact_contest_cuda_eval_missing" in row["dispatch_blockers"]


def test_portfolio_rejects_authoritative_source_rows() -> None:
    selection = _mlx_selection()
    selection["selected_rows"][0]["score_claim"] = True  # type: ignore[index]
    with pytest.raises(CrossFamilyCandidatePortfolioError, match="score_claim"):
        build_cross_family_candidate_portfolio(
            incumbent_score=0.2,
            mlx_selections=[selection],
        )


def test_cross_family_portfolio_cli_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    mlx_path = tmp_path / "mlx_selection.json"
    pairset_path = tmp_path / "pairset.json"
    hfv2_path = tmp_path / "hfv2_manifest.json"
    json_out = tmp_path / "portfolio.json"
    md_out = tmp_path / "portfolio.md"
    mlx_path.write_text(json.dumps(_mlx_selection(), sort_keys=True), encoding="utf-8")
    pairset_path.write_text(
        json.dumps(_pairset_acquisition(), sort_keys=True),
        encoding="utf-8",
    )
    hfv2_path.write_text(
        json.dumps(_hfv2_manifest(tmp_path / "archive.zip"), sort_keys=True),
        encoding="utf-8",
    )

    first = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_cross_family_candidate_portfolio.py"),
            "--incumbent-score",
            "0.192051316881",
            "--mlx-selection",
            str(mlx_path),
            "--pairset-acquisition",
            str(pairset_path),
            "--hfv2-manifest",
            str(hfv2_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    first_text = json_out.read_text(encoding="utf-8")
    second = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_cross_family_candidate_portfolio.py"),
            "--incumbent-score",
            "0.192051316881",
            "--mlx-selection",
            str(mlx_path),
            "--pairset-acquisition",
            str(pairset_path),
            "--hfv2-manifest",
            str(hfv2_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(first.stdout)["score_claim"] is False
    assert json.loads(second.stdout)["ready_for_exact_eval_dispatch"] is False
    assert json_out.read_text(encoding="utf-8") == first_text
    assert "Cross-Family Candidate Portfolio" in md_out.read_text(encoding="utf-8")
