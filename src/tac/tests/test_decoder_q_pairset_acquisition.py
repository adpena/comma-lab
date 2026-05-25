# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.decoder_q_pairset_acquisition import (
    CANDIDATE_SCHEMA,
    EUREKA_EXPANSION_SCHEMA,
    FALSE_ACQUISITION_AUTHORITY,
    SCHEMA,
    DecoderQPairsetAcquisitionError,
    build_decoder_q_pairset_acquisition_plan,
)

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


def _exact_estimate(predicted_score: float) -> dict[str, object]:
    return {
        "schema": "decoder_q_selective_selector_exact_cpu_calibrated_estimate.v1",
        **_false_authority(),
        "predicted_score": predicted_score,
        "predicted_delta_vs_base": predicted_score - 0.1921,
        "allowed_use": "non_authoritative_selector_ranking_only",
    }


def _selector_pareto() -> dict[str, object]:
    return {
        "schema": "decoder_q_selective_selector_pareto.v1",
        **_false_authority(),
        "summary": {"recommended_selector_id": "best_top4"},
        "candidates": [
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "best_top4",
                "selector_kind": "top_rank_prefix",
                "selector_rank": 1,
                "rank_order_pair_indices": [10, 20, 30, 40],
                "selected_pair_indices": [10, 20, 30, 40],
                "selected_pair_count": 4,
                "payload_bytes": 14,
                "predicted_score_mean": 0.1919,
                "exact_cpu_calibrated_estimate": _exact_estimate(0.1919),
            },
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "lower_locality",
                "selector_kind": "full_minus_one",
                "selector_rank": 2,
                "rank_order_pair_indices": [10, 25, 30, 50],
                "selected_pair_indices": [10, 25, 30, 50],
                "selected_pair_count": 4,
                "payload_bytes": 14,
                "predicted_score_mean": 0.1923,
                "exact_cpu_calibrated_estimate": _exact_estimate(0.1923),
            },
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "lower_spread",
                "selector_kind": "diversity_probe",
                "selector_rank": 3,
                "rank_order_pair_indices": [0, 199, 399, 599],
                "selected_pair_indices": [0, 199, 399, 599],
                "selected_pair_count": 4,
                "payload_bytes": 14,
                "predicted_score_mean": 0.193,
                "exact_cpu_calibrated_estimate": _exact_estimate(0.193),
            },
        ],
    }


def _dqs1_observation(candidate_id: str, *, score_delta: float = -0.0001) -> dict[str, object]:
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        **_false_authority(),
        "candidate_id": candidate_id,
        "source_schema": "dqs1_local_first_harvest.v1",
        "sweep_config_id": "dqs1_local_first_macos_cpu_advisory",
        "optimization_pass_id": "local_cpu_advisory_harvest",
        "family": "decoder_q_pairset_drop",
        "observed_axis": "macos_cpu_advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "observed_score_or_delta": 0.1919,
        "score_delta_vs_baseline": score_delta,
        "archive_byte_delta_vs_baseline": -4,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "raw_output_or_cache_sha256": "c" * 64,
        "component_deltas": {
            "segnet_delta": -0.0001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00002,
        },
    }


def _large_selector_pareto(pair_count: int = 32) -> dict[str, object]:
    pairs = list(range(pair_count))
    return {
        "schema": "decoder_q_selective_selector_pareto.v1",
        **_false_authority(),
        "summary": {"recommended_selector_id": "best_top32"},
        "candidates": [
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "best_top32",
                "selector_kind": "top_rank_prefix",
                "selector_rank": 1,
                "rank_order_pair_indices": pairs,
                "selected_pair_indices": pairs,
                "selected_pair_count": len(pairs),
                "payload_bytes": 41,
                "predicted_score_mean": 0.19203,
                "exact_cpu_calibrated_estimate": _exact_estimate(0.19203),
            },
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "spread_alt",
                "selector_kind": "diversity_probe",
                "selector_rank": 2,
                "rank_order_pair_indices": list(range(0, pair_count * 2, 2)),
                "selected_pair_indices": list(range(0, pair_count * 2, 2)),
                "selected_pair_count": pair_count,
                "payload_bytes": 41,
                "predicted_score_mean": 0.19204,
                "exact_cpu_calibrated_estimate": _exact_estimate(0.19204),
            },
        ],
    }


def _small_plan() -> dict[str, object]:
    return build_decoder_q_pairset_acquisition_plan(
        _selector_pareto(),
        prefix_ks=[1, 2, 4],
        diversity_ks=[2, 4],
        max_drop_two=10,
        max_swap_in=3,
    )


def test_pairset_acquisition_schema_and_false_authority() -> None:
    plan = _small_plan()

    assert plan["schema"] == SCHEMA
    assert plan["candidate_generation_only"] is True
    assert plan["local_only"] is True
    for key, value in FALSE_ACQUISITION_AUTHORITY.items():
        assert plan[key] is value

    kinds = {row["selector_kind"] for row in plan["candidates"]}  # type: ignore[index]
    assert kinds == {
        "diversity_spaced",
        "drop_one_from_best",
        "drop_two_from_best",
        "prefix_variant",
        "swap_in_alternative",
    }
    for row in plan["candidates"]:  # type: ignore[index]
        assert row["schema"] == CANDIDATE_SCHEMA
        assert row["candidate_generation_only"] is True
        assert row["local_only"] is True
        for key, value in FALSE_ACQUISITION_AUTHORITY.items():
            assert row[key] is value


def test_pairset_acquisition_generates_deterministic_candidates() -> None:
    first = _small_plan()
    second = _small_plan()

    first_rows = [
        (
            row["acquisition_id"],
            row["selector_kind"],
            row["selected_pair_indices"],
            row["payload_bytes"],
            row["acquisition_score"],
            row["diversity_score"],
        )
        for row in first["candidates"]  # type: ignore[index]
    ]
    second_rows = [
        (
            row["acquisition_id"],
            row["selector_kind"],
            row["selected_pair_indices"],
            row["payload_bytes"],
            row["acquisition_score"],
            row["diversity_score"],
        )
        for row in second["candidates"]  # type: ignore[index]
    ]
    assert first_rows == second_rows
    assert any(row[0] == "pairset_prefix_k004" for row in first_rows)
    assert any(row[1] == "drop_two_from_best" for row in first_rows)
    assert any(row[1] == "swap_in_alternative" for row in first_rows)


def test_pairset_acquisition_defaults_include_dense_response_tail() -> None:
    plan = build_decoder_q_pairset_acquisition_plan(
        _large_selector_pareto(),
        max_drop_two=0,
        max_swap_in=0,
        include_drop_one=False,
    )

    assert plan["selection_policy"]["prefix_ks"] == [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        26,
        28,
        30,
        31,
        32,
    ]
    assert plan["selection_policy"]["diversity_ks"] == [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        26,
        28,
        30,
        31,
        32,
    ]
    acquisition_ids = {
        row["acquisition_id"]
        for row in plan["candidates"]  # type: ignore[index]
    }
    assert "pairset_prefix_k026" in acquisition_ids
    assert "pairset_prefix_k028" in acquisition_ids
    assert "pairset_diversity_k030" in acquisition_ids
    assert (
        plan["selection_policy"]["default_k_policy"]
        == "coarse_global_sweep_plus_dense_tail_for_observation_response_interpolation"
    )


def test_pairset_acquisition_eureka_expands_into_bounded_drop_many() -> None:
    eureka_planning = {
        "schema": "frontier_rate_attack_local_cpu_eureka_discovery.v1",
        "active": True,
        "signal_count": 2,
        "planner_hints": [
            {
                "schema": "frontier_rate_attack_local_cpu_eureka_planner_hint.v1",
                "hint_id": "dqs1_expand_beyond_drop_two_near_boundary",
                "pairset_acquisition_profile": {
                    "schema": (
                        "frontier_rate_attack_local_cpu_eureka_pairset_acquisition_profile.v1"
                    ),
                    "active": True,
                    "max_drop_many": 5,
                    "drop_many_counts": [3, 4],
                    **FALSE_ACQUISITION_AUTHORITY,
                },
                **FALSE_ACQUISITION_AUTHORITY,
            }
        ],
        **FALSE_ACQUISITION_AUTHORITY,
    }

    plan = build_decoder_q_pairset_acquisition_plan(
        _large_selector_pareto(),
        max_drop_two=0,
        max_swap_in=0,
        include_drop_one=False,
        prefix_ks=[32],
        diversity_ks=[32],
        eureka_planning=eureka_planning,
    )

    expansion = plan["selection_policy"]["eureka_expansion"]
    assert expansion["schema"] == EUREKA_EXPANSION_SCHEMA
    assert expansion["active"] is True
    assert expansion["drop_many_candidate_generation_active"] is True
    assert "pair" in expansion["levels_considered"]
    assert "full_video" in expansion["levels_considered"]
    assert plan["selection_policy"]["drop_many_counts"] == [3, 4]
    assert plan["selection_policy"]["max_drop_many"] == 5
    assert plan["summary"]["drop_many_candidate_count"] == 5

    drop_many = [
        row
        for row in plan["candidates"]  # type: ignore[index]
        if row["selector_kind"] == "drop_many_beam_pairwise_interaction_waterfill"
    ]
    assert drop_many
    assert all(row["acquisition_id"].startswith("pairset_drop_many_") for row in drop_many)
    assert all(row["acquisition_operation"]["op"] == "drop_many" for row in drop_many)
    assert any(
        row["distortion_repair_budget_from_rate_savings"]["active"] is True
        for row in drop_many
    )
    best_budget = next(
        row["distortion_repair_budget_from_rate_savings"]
        for row in drop_many
        if row["distortion_repair_budget_from_rate_savings"]["active"] is True
    )
    assert best_budget["saved_bytes_vs_source_selector"] > 0
    assert best_budget["score_budget"] > 0.0
    assert best_budget["segnet_distortion_budget_at_fixed_pose"] == pytest.approx(
        best_budget["score_budget"] / 100.0
    )
    assert best_budget["posenet_score_term_budget_at_fixed_seg"] == pytest.approx(
        best_budget["score_budget"]
    )
    assert all(
        "dqs1_expand_beyond_drop_two_near_boundary"
        in row["acquisition_operation"]["eureka_planner_hint_ids"]
        for row in drop_many
    )
    assert all(row["score_claim"] is False for row in drop_many)


def test_pairset_acquisition_suppresses_observed_dqs1_candidates() -> None:
    plan = build_decoder_q_pairset_acquisition_plan(
        _selector_pareto(),
        prefix_ks=[1, 2, 4],
        diversity_ks=[2, 4],
        max_drop_two=10,
        max_swap_in=3,
        dqs1_observations=(
            _dqs1_observation("pairset_prefix_k004"),
            _dqs1_observation("pairset_drop_one_rank001_pair0010", score_delta=0.0002),
        ),
    )

    ids = [row["acquisition_id"] for row in plan["candidates"]]  # type: ignore[index]
    assert "pairset_prefix_k004" not in ids
    assert "pairset_drop_one_rank001_pair0010" not in ids
    assert plan["summary"]["observed_dqs1_candidate_count"] == 2
    assert plan["summary"]["suppressed_observed_candidate_count"] == 2
    assert plan["summary"]["candidate_count"] == plan["summary"]["unfiltered_candidate_count"] - 2
    policy = plan["selection_policy"]["observation_skip"]
    assert policy["schema"] == "dqs1_pairset_acquisition_observation_skip.v1"
    assert policy["active"] is True
    assert policy["score_claim"] is False
    assert policy["ready_for_exact_eval_dispatch"] is False
    assert [row["acquisition_rank"] for row in plan["candidates"]] == list(  # type: ignore[index]
        range(1, len(plan["candidates"]) + 1)  # type: ignore[arg-type]
    )


def test_pairset_acquisition_replay_can_include_observed_dqs1_candidates() -> None:
    plan = build_decoder_q_pairset_acquisition_plan(
        _selector_pareto(),
        prefix_ks=[1, 2, 4],
        diversity_ks=[2, 4],
        max_drop_two=10,
        max_swap_in=3,
        dqs1_observations=(_dqs1_observation("pairset_prefix_k004"),),
        include_observed_candidates=True,
    )

    ids = {row["acquisition_id"] for row in plan["candidates"]}  # type: ignore[index]
    assert "pairset_prefix_k004" in ids
    assert plan["summary"]["observed_dqs1_candidate_count"] == 1
    assert plan["summary"]["suppressed_observed_candidate_count"] == 0
    assert plan["selection_policy"]["observation_skip"]["active"] is False


def test_pairset_acquisition_preserves_selector_metadata_and_valid_pairs() -> None:
    plan = _small_plan()

    prefix = next(
        row
        for row in plan["candidates"]  # type: ignore[index]
        if row["acquisition_id"] == "pairset_prefix_k004"
    )
    assert prefix["selected_pair_indices"] == [10, 20, 30, 40]
    assert prefix["source_selector_kind"] == "top_rank_prefix"
    assert prefix["payload_byte_estimate"] == prefix["payload_bytes"]
    assert prefix["source_selector_payload_bytes"] == 14
    assert (
        prefix["payload_bytes_delta_vs_source_selector"]
        == prefix["payload_bytes"] - 14
    )
    assert prefix["distortion_repair_budget_from_rate_savings"]["active"] is False
    assert prefix["predicted_score_mean"] == pytest.approx(0.1919)
    assert prefix["predicted_score_scope"] == "candidate_specific"
    assert prefix["exact_cpu_calibrated_estimate"]["predicted_score"] == pytest.approx(0.1919)
    assert prefix["exact_cpu_calibrated_estimate_scope"] == "candidate_specific"
    assert prefix["exact_cpu_calibrated_estimate"]["score_claim"] is False

    drop_one = next(
        row
        for row in plan["candidates"]  # type: ignore[index]
        if row["selector_kind"] == "drop_one_from_best"
    )
    assert drop_one["predicted_score_scope"] == "source_selector_scope_not_child_candidate"
    assert "exact_cpu_calibrated_estimate" not in drop_one
    assert drop_one["source_selector_exact_cpu_calibrated_estimate"]["score_claim"] is False
    assert (
        drop_one["source_selector_exact_cpu_calibrated_estimate_scope"]
        == "source_selector_scope_not_child_candidate"
    )

    for row in plan["candidates"]:  # type: ignore[index]
        pairs = row["selected_pair_indices"]
        assert pairs == sorted(pairs)
        assert len(pairs) == len(set(pairs))
        assert all(0 <= pair < 600 for pair in pairs)
        assert row["selected_pair_count"] == len(pairs)
        assert row["payload_bytes"] > 0
        assert isinstance(row["acquisition_score"], float)
        assert isinstance(row["diversity_score"], float)


def test_pairset_acquisition_rejects_authority_and_bad_pairs() -> None:
    authoritative = _selector_pareto()
    authoritative["candidates"][0]["score_claim"] = True  # type: ignore[index]
    with pytest.raises(DecoderQPairsetAcquisitionError, match="score_claim"):
        build_decoder_q_pairset_acquisition_plan(authoritative)

    duplicate_pairs = _selector_pareto()
    duplicate_pairs["candidates"][0]["selected_pair_indices"] = [10, 10]  # type: ignore[index]
    with pytest.raises(DecoderQPairsetAcquisitionError, match="duplicates"):
        build_decoder_q_pairset_acquisition_plan(duplicate_pairs)

    authoritative_observation = _dqs1_observation("pairset_prefix_k004")
    authoritative_observation["score_claim"] = True
    with pytest.raises(DecoderQPairsetAcquisitionError, match="score_claim"):
        build_decoder_q_pairset_acquisition_plan(
            _selector_pareto(),
            dqs1_observations=(authoritative_observation,),
        )


def test_pairset_acquisition_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    selector_path = tmp_path / "selector_pareto.json"
    observations_path = tmp_path / "dqs1_observations.jsonl"
    eureka_path = tmp_path / "eureka_planning.json"
    json_out = tmp_path / "pairset_acquisition.json"
    md_out = tmp_path / "pairset_acquisition.md"
    selector_path.write_text(json.dumps(_selector_pareto()), encoding="utf-8")
    observations_path.write_text(
        json.dumps(_dqs1_observation("pairset_prefix_k004"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eureka_path.write_text(
        json.dumps(
            {
                "schema": "frontier_rate_attack_local_cpu_eureka_discovery.v1",
                "active": True,
                "planner_hints": [
                    {
                        "hint_id": "dqs1_expand_beyond_drop_two_near_boundary",
                        **FALSE_ACQUISITION_AUTHORITY,
                    }
                ],
                **FALSE_ACQUISITION_AUTHORITY,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_decoder_q_pairset_acquisition.py"),
            "--selector-pareto",
            str(selector_path),
            "--prefix-ks",
            "1,4",
            "--diversity-ks",
            "2,4",
            "--max-drop-two",
            "3",
            "--drop-many-counts",
            "3",
            "--max-drop-many",
            "2",
            "--max-swap-in",
            "2",
            "--eureka-planning-json",
            str(eureka_path),
            "--dqs1-observation-jsonl",
            str(observations_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["score_claim"] is False
    assert stdout_payload["promotion_eligible"] is False
    assert stdout_payload["ready_for_exact_eval_dispatch"] is False
    assert stdout_payload["dispatch_attempted"] is False
    assert stdout_payload["suppressed_observed_candidate_count"] == 1
    assert stdout_payload["drop_many_candidate_count"] == 1
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["summary"]["candidate_count"] > 0
    assert payload["summary"]["drop_many_candidate_count"] == 1
    assert "pairset_prefix_k004" not in {
        row["acquisition_id"] for row in payload["candidates"]
    }
    markdown = md_out.read_text(encoding="utf-8")
    assert "Decoder-Q Pair-Set Acquisition Plan" in markdown
    assert "planning-only local pair-set acquisition" in markdown
