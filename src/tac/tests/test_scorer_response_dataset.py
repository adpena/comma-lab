from __future__ import annotations

import json
import math

from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES
from tac.optimization.scorer_response_dataset import (
    RATE_SCORE_PER_BYTE,
    ResponseBaseline,
    ScorerResponseDatasetError,
    build_magic_codec_seed_boundary,
    build_null_byte_priority_weights,
    build_next_probe_plan,
    build_response_dataset,
    normalize_legacy_response_dataset_authority,
    render_next_probe_plan_markdown,
    render_markdown,
)


def _advisory(score: float, archive_bytes: int, pose: float, seg: float) -> dict:
    return {
        "canonical_score": score,
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "axis": "[macOS-CPU advisory test]",
        "archive": {"sha256": "a" * 64, "bytes": archive_bytes},
        "raw": {"sha256": "b" * 64},
    }


def test_build_response_dataset_normalizes_single_candidate(tmp_path) -> None:
    path = tmp_path / "scorer_gradient.json"
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "summary": {
            "component": "pose",
            "pair_indices": [7],
            "changed_pixel_count": 3,
            "changed_byte_count": 8,
            "packed_bytes": 12,
            "delta_vs_baseline_score": 0.1,
        },
        "candidate": {
            "advisory_eval": _advisory(1.25, 110, 0.004, 0.010),
            "inputs": {"target_raw_sha256": "c" * 64},
            "plan": {"selected_gain_sum": 5.5, "n_kept": 3},
            "local_pair_evals": [
                {
                    "delta": {"pose_dist_delta": -0.25, "seg_dist_delta": 0.0},
                    "worse_or_null": False,
                }
            ],
        },
        "authority": {"score_claim": False, "promotion_blockers": ["advisory"]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    baseline = ResponseBaseline(score=1.0, archive_bytes=100)

    dataset = build_response_dataset([path], baseline=baseline)

    assert dataset["score_claim"] is False
    assert dataset["rank_or_kill_eligible"] is False
    assert dataset["promotable"] is False
    assert dataset["authority"]["rank_or_kill_eligible"] is False
    assert dataset["authority"]["promotable"] is False
    assert dataset["summary"]["row_count"] == 1
    row = dataset["rows"][0]
    assert row["rank_or_kill_eligible"] is False
    assert row["promotable"] is False
    assert row["family"] == "scorer_gradient_sparse_residual"
    assert row["delta_vs_baseline_score"] == 0.25
    expected_rate_delta = 25.0 * 10.0 / CONTEST_UNCOMPRESSED_BYTES
    assert math.isclose(row["rate_delta_vs_baseline"], expected_rate_delta)
    assert math.isclose(row["scorer_delta_vs_baseline"], 0.25 - expected_rate_delta)
    assert row["added_archive_bytes"] == 10
    assert math.isclose(row["required_scorer_gain_for_added_bytes"], expected_rate_delta)
    assert row["observed_scorer_gain_vs_baseline"] == 0.0
    assert math.isclose(row["scorer_gain_shortfall_to_break_even"], expected_rate_delta)
    assert row["break_even_added_bytes_from_scorer_gain"] is None
    assert row["byte_budget_margin_vs_break_even"] is None
    assert row["local_pose_delta_sum"] == -0.25
    assert row["target_raw_sha256"] == "c" * 64
    assert row["holdout_fold"] in {0, 1, 2, 3, 4}


def _current_response_dataset_payload() -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return {
        "schema": "scorer_response_dataset.v1",
        "producer": "test",
        **false_authority,
        "authority": {
            **false_authority,
            "evidence_grade": "macOS-CPU advisory response dataset",
        },
        "summary": {"row_count": 1},
        "rows": [
            {
                "schema": "scorer_response_row.v1",
                "row_id": "row-a",
                **false_authority,
                "authority_source_score_claim": False,
                "advisory_score_report_derived": 1.0,
                "delta_vs_baseline_score": 0.0,
            }
        ],
    }


def test_normalize_legacy_response_dataset_authority_backfills_extended_fields_only() -> None:
    payload = _current_response_dataset_payload()
    payload.pop("rank_or_kill_eligible")
    payload.pop("promotable")
    payload["authority"].pop("rank_or_kill_eligible")
    payload["authority"].pop("promotable")
    payload["rows"][0].pop("rank_or_kill_eligible")
    payload["rows"][0].pop("promotable")

    normalized = normalize_legacy_response_dataset_authority(
        payload,
        source_label="historical_pr110",
    )

    assert normalized["rank_or_kill_eligible"] is False
    assert normalized["promotable"] is False
    assert normalized["authority"]["rank_or_kill_eligible"] is False
    assert normalized["rows"][0]["promotable"] is False
    metadata = normalized["authority_normalization"]
    assert metadata["score_claim"] is False
    assert metadata["source_label"] == "historical_pr110"
    assert metadata["backfilled_missing_false_field_count"] == 6
    assert {
        (item["label"], item["field"])
        for item in metadata["backfilled_missing_false_fields"]
    } == {
        ("scorer-response dataset", "rank_or_kill_eligible"),
        ("scorer-response dataset", "promotable"),
        ("scorer-response dataset authority", "rank_or_kill_eligible"),
        ("scorer-response dataset authority", "promotable"),
        ("scorer-response row 0", "rank_or_kill_eligible"),
        ("scorer-response row 0", "promotable"),
    }


def test_normalize_legacy_response_dataset_authority_refuses_core_or_source_ambiguity() -> None:
    missing_core = _current_response_dataset_payload()
    missing_core.pop("score_claim")
    try:
        normalize_legacy_response_dataset_authority(missing_core)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing core authority rejection")

    source_claim = _current_response_dataset_payload()
    source_claim["rows"][0]["authority_source_score_claim"] = "true"
    try:
        normalize_legacy_response_dataset_authority(source_claim)
    except ScorerResponseDatasetError as exc:
        assert "authority_source_score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected source score-claim rejection")


def test_build_response_dataset_normalizes_candidate_list_and_correlations(tmp_path) -> None:
    path = tmp_path / "op3_summary.json"
    payload = {
        "schema": "op3v3_decoder_q_advisory_batch.v1",
        "producer": "tools/run_decoder_q_candidate_advisory_batch.py",
        "candidates": [
            {
                "candidate_id": f"c{i}",
                "advisory_eval": _advisory(1.0 + i * 0.1, 100 + i, 0.001 + i * 0.001, 0.01),
            }
            for i in range(4)
        ],
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=1.0, archive_bytes=100),
    )

    assert dataset["summary"]["row_count"] == 4
    assert dataset["summary"]["family_counts"] == {"decoder_q": 4}
    assert dataset["feature_correlations"]
    md = render_markdown(dataset)
    assert "Scorer Response Dataset" in md
    assert "`decoder_q`: 4" in md


def test_response_dataset_computes_break_even_bytes_for_scorer_gain(tmp_path) -> None:
    path = tmp_path / "gain.json"
    score = 1.0 - RATE_SCORE_PER_BYTE
    payload = {
        "schema": "sparse_residual_oracle_smoke.v1",
        "candidate": {
            "advisory_eval": _advisory(score, 102, 0.001, 0.01),
            "plan": {"packed_bytes": 2},
        },
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))
    row = dataset["rows"][0]

    assert row["added_archive_bytes"] == 2
    assert math.isclose(row["rate_delta_vs_baseline"], 2.0 * RATE_SCORE_PER_BYTE)
    assert math.isclose(row["scorer_delta_vs_baseline"], -3.0 * RATE_SCORE_PER_BYTE)
    assert math.isclose(row["observed_scorer_gain_vs_baseline"], 3.0 * RATE_SCORE_PER_BYTE)
    assert math.isclose(row["required_scorer_gain_for_added_bytes"], 2.0 * RATE_SCORE_PER_BYTE)
    assert row["scorer_gain_shortfall_to_break_even"] == 0.0
    assert math.isclose(row["break_even_added_bytes_from_scorer_gain"], 3.0)
    assert math.isclose(row["byte_budget_margin_vs_break_even"], 1.0)
    assert dataset["summary"]["best_byte_budget_margin"]["byte_budget_margin_vs_break_even"] > 0


def test_next_probe_plan_blocks_overbudget_coordinate_residual(tmp_path) -> None:
    path = tmp_path / "overbudget.json"
    score = 1.0 + RATE_SCORE_PER_BYTE
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(score, 103, 0.001, 0.01)},
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    plan = build_next_probe_plan(dataset)

    assert plan["score_claim"] is False
    assert plan["prohibitions"][0]["rule"] == "do_not_widen_coordinate_sparse_residual_sidecar"
    assert plan["probes"][0]["probe_id"] == "ll_byte_neutral_decoder_q_response_model"
    assert "Next-Probe" in render_next_probe_plan_markdown(plan)


def _null_byte_matrix() -> dict:
    return {
        "schema": "null_byte_master_gradient_probe_matrix_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "n_anchors_probed_ok": 2,
        "top5_replacement_candidates": [
            {
                "substrate_label": "smaller_null_budget",
                "codec_family": "hnerv_family",
                "scored_archive_sha256": "b" * 64,
                "axis": "[contest-CUDA]",
                "anchor_index": 2,
                "n_null_bytes": 128,
                "null_fraction": 0.1,
                "predicted_delta_s_per_seed_budget": {"K=16": -0.0001},
            },
            {
                "substrate_label": "larger_null_budget",
                "codec_family": "hnerv_family",
                "scored_archive_sha256": "a" * 64,
                "axis": "[contest-CUDA]",
                "anchor_index": 1,
                "n_null_bytes": 256,
                "null_fraction": 0.2,
                "predicted_delta_s_per_seed_budget": {"K=16": -0.0002},
            },
        ],
    }


def _pair4_seed_boundary_smoke() -> dict:
    return {
        "smoke_label": "wave_3_magic_codec_pair_4_procedural_seed_orthogonality_smoke",
        "smoke_pair_id": "pair_4_magic_codec_x_procedural_codebook_seed_bytes",
        "cascade_verdict": "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "n_canonical_reversible_ordering_rows": 30,
        "n_canonical_reversible_ordering_rows_raw_seed_dominates": 30,
        "min_canonical_reversible_best_nonraw_delta_vs_raw_bytes": 4,
        "ordering_dimension": {
            "reversible_free_orderings": ["identity", "reverse"],
            "non_free_control_orderings": ["sorted_ascending"],
        },
        "codec_dimensions": {"raw_seed": True, "brotli_q11_seed_bytes": True},
    }


def test_magic_codec_seed_boundary_normalizes_pair4_smoke() -> None:
    boundary = build_magic_codec_seed_boundary(_pair4_seed_boundary_smoke())

    assert boundary["schema"] == "ll_magic_codec_seed_boundary.v1"
    assert boundary["score_claim"] is False
    assert boundary["score_claim_valid"] is False
    assert boundary["rank_or_kill_eligible"] is False
    assert boundary["promotable"] is False
    assert boundary["boundary_validated_raw_seed_dominates"] is True
    assert boundary["n_canonical_reversible_ordering_rows"] == 30
    assert boundary["min_canonical_reversible_best_nonraw_delta_vs_raw_bytes"] == 4


def test_magic_codec_seed_boundary_rejects_promotional_smoke() -> None:
    bad = _pair4_seed_boundary_smoke()
    bad["promotion_eligible"] = True
    try:
        build_magic_codec_seed_boundary(bad)
    except ScorerResponseDatasetError as exc:
        assert "promotion_eligible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected promotional boundary rejection")

    score_claim = _pair4_seed_boundary_smoke()
    score_claim["score_claim"] = True
    try:
        build_magic_codec_seed_boundary(score_claim)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected score-claim boundary rejection")


def test_null_byte_priority_weights_sort_by_predicted_delta() -> None:
    weights = build_null_byte_priority_weights(_null_byte_matrix())

    assert weights["score_claim"] is False
    assert weights["rank_or_kill_eligible"] is False
    assert weights["promotable"] is False
    assert weights["summary"]["candidate_count"] == 2
    assert weights["priority_rows"][0]["substrate_label"] == "larger_null_budget"
    assert weights["priority_rows"][0]["priority_weight"] == 0.0002
    assert weights["priority_rows"][0]["priority_weight_units"] == "absolute_predicted_score_delta"
    assert 0.0 < weights["priority_rows"][0]["ll_sampling_weight"] < 1.0


def test_null_byte_priority_weights_reject_legacy_missing_false_authority_keys_by_default() -> None:
    matrix = _null_byte_matrix()
    matrix.pop("promotion_eligible")
    matrix.pop("rank_or_kill_eligible")

    try:
        build_null_byte_priority_weights(matrix)
    except ScorerResponseDatasetError as exc:
        assert "must be explicit false" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing authority rejection")


def test_null_byte_priority_weights_accept_legacy_missing_false_authority_keys_with_flag() -> None:
    matrix = _null_byte_matrix()
    matrix.pop("promotion_eligible")
    matrix.pop("rank_or_kill_eligible")
    matrix.pop("ready_for_exact_eval_dispatch")

    weights = build_null_byte_priority_weights(
        matrix,
        allow_legacy_missing_authority=True,
    )

    assert weights["score_claim"] is False
    assert weights["promotion_eligible"] is False
    assert weights["ready_for_exact_eval_dispatch"] is False
    assert set(weights["legacy_missing_authority_fields_accepted"]) == {
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    }


def test_next_probe_plan_consumes_null_byte_matrix_as_first_probe(tmp_path) -> None:
    path = tmp_path / "overbudget.json"
    score = 1.0 + RATE_SCORE_PER_BYTE
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(score, 103, 0.001, 0.01)},
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    plan = build_next_probe_plan(dataset, null_byte_matrix=_null_byte_matrix())

    assert plan["null_byte_priority_weights"]["schema"] == "ll_null_byte_priority_weights.v1"
    assert plan["probes"][0]["probe_id"] == "ll_null_byte_procedural_codebook_candidates"
    assert plan["probes"][0]["priority"] == 1
    assert plan["probes"][1]["probe_id"] == "ll_byte_neutral_decoder_q_response_model"
    assert "CandidateModificationSpec" in plan["probes"][0]["acceptance_gate"]
    assert "Null-Byte Matrix Priority" in render_next_probe_plan_markdown(plan)


def test_next_probe_plan_consumes_pair4_seed_boundary_as_prohibition(tmp_path) -> None:
    path = tmp_path / "overbudget.json"
    score = 1.0 + RATE_SCORE_PER_BYTE
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(score, 103, 0.001, 0.01)},
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    plan = build_next_probe_plan(
        dataset,
        null_byte_matrix=_null_byte_matrix(),
        magic_codec_seed_boundary_smoke=_pair4_seed_boundary_smoke(),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in rules
    assert plan["magic_codec_seed_boundary"]["boundary_validated_raw_seed_dominates"] is True
    assert "keep seeds raw" in plan["probes"][0]["rationale"]
    rendered = render_next_probe_plan_markdown(plan)
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in rendered


def test_null_byte_matrix_fail_closed_on_promotional_or_missing_k() -> None:
    bad = _null_byte_matrix()
    bad["score_claim"] = True
    try:
        build_null_byte_priority_weights(bad)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected promotional matrix rejection")

    missing_authority = _null_byte_matrix()
    missing_authority.pop("ready_for_exact_eval_dispatch")
    try:
        build_next_probe_plan(
            {"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []},
            null_byte_matrix=missing_authority,
        )
    except ScorerResponseDatasetError as exc:
        assert "ready_for_exact_eval_dispatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing ready_for_exact_eval_dispatch rejection")

    missing_k = _null_byte_matrix()
    missing_k["top5_replacement_candidates"][0]["predicted_delta_s_per_seed_budget"] = {"K=32": -0.1}
    try:
        build_null_byte_priority_weights(missing_k)
    except ScorerResponseDatasetError as exc:
        assert "K=16" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing K rejection")


def test_build_response_dataset_skips_non_advisory_rows(tmp_path) -> None:
    path = tmp_path / "skip.json"
    path.write_text(
        json.dumps({"candidate": {"advisory_eval": {"skipped": True, "reason": "local_veto"}}}),
        encoding="utf-8",
    )

    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    assert dataset["summary"]["row_count"] == 0
    assert dataset["skipped"][0]["reason"].endswith("no usable advisory row")
