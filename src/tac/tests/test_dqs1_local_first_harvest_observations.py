# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.dqs1_local_first_harvest_observations import (
    DQS1LocalHarvestObservationError,
    build_harvest_observation_summary,
    build_observation_rows_from_harvests,
    write_observation_jsonl,
)
from tac.optimization.macos_cpu_advisory_signal import EVIDENCE_GRADE, EVIDENCE_TAG
from tac.optimization.mlx_dynamic_sweep_observations import load_observation_rows
from tac.optimization.pairset_component_marginal import rate_delta_for_archive_byte_delta
from tools.build_dqs1_local_first_harvest_observations import _expand_harvest_inputs


def _sha(char: str) -> str:
    return char * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _advisory(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    raw_sha256: str,
    runtime_sha256: str,
    seg: float,
    pose: float = 0.017,
) -> dict:
    rate = rate_delta_for_archive_byte_delta(archive_size_bytes)
    score = seg + pose + rate
    return {
        "schema_version": "contest_auth_eval_result.v1",
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "score_seg_contribution": seg,
        "score_pose_contribution": pose,
        "score_rate_contribution": rate,
        "archive_size_bytes": archive_size_bytes,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "provenance": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": runtime_sha256,
            },
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": raw_sha256,
                },
            },
        },
    }


def test_build_rows_from_harvests_preserves_false_authority_and_component_deltas(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    baseline_path = tmp_path / "baseline.json"
    advisory_path = tmp_path / "candidate" / "local_cpu_advisory.json"
    harvest_path = tmp_path / ".omx" / "research" / "harvest.json"
    acquisition_path = tmp_path / "pairset_acquisition.json"
    baseline_size = 178_560
    candidate_size = 178_559

    _write_json(
        baseline_path,
        _advisory(
            archive_sha256=_sha("a"),
            archive_size_bytes=178_592,
            raw_sha256=_sha("b"),
            runtime_sha256=_sha("c"),
            seg=0.055988,
        ),
    )
    _write_json(
        advisory_path,
        _advisory(
            archive_sha256=_sha("d"),
            archive_size_bytes=candidate_size,
            raw_sha256=_sha("e"),
            runtime_sha256=_sha("f"),
            seg=0.055989,
        ),
    )
    _write_json(
        harvest_path,
        {
            "schema": "dqs1_local_first_harvest.v1",
            "candidate_id": "pairset_drop_one_rank009_pair0459",
            "candidate_archive_sha256": _sha("d"),
            "local_cpu_advisory_path": str(advisory_path),
            "local_score": 0.123,
            "projected_contest_score": 0.122,
            "conservative_projected_contest_score": 0.124,
            "recommended_action": "observe_only",
            "eureka_trigger": False,
            "eureka_margin": -1e-6,
            "authority": "false_authority_dqs1_local_first_harvest",
            "harvested_at_utc": "20260523T114957Z",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
            "dispatch_blockers": ["exact_cpu_cuda_auth_eval_required"],
        },
    )
    _write_json(
        acquisition_path,
        {
            "schema": "decoder_q_pairset_acquisition.v1",
            "candidates": [
                {
                    "acquisition_id": "pairset_drop_one_rank009_pair0459",
                    "selector_id": "pairset_drop_one_rank009_pair0459",
                    "selector_kind": "drop_one_from_best",
                    "selected_pair_indices": [26, 59, 68],
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 9,
                        "dropped_pair_index": 459,
                    },
                }
            ],
        },
    )

    rows = build_observation_rows_from_harvests(
        [harvest_path],
        repo_root=repo_root,
        pairset_acquisition_path=acquisition_path,
        baseline_advisory_path=baseline_path,
        baseline_archive_size_bytes=baseline_size,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "pairset_drop_one_rank009_pair0459"
    assert row["observed_axis"] == "macos_cpu_advisory"
    assert row["evidence_grade"] == EVIDENCE_GRADE
    assert row["evidence_tag"] == EVIDENCE_TAG
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["family"] == "decoder_q_pairset_drop_one"
    assert row["selected_pair_indices"] == [26, 59, 68]
    assert row["observed_at_utc"] == "2026-05-23T11:49:57Z"
    assert row["segnet_delta"] == pytest.approx(0.000001)
    assert row["rate_delta"] == pytest.approx(
        rate_delta_for_archive_byte_delta(candidate_size - baseline_size)
    )
    assert row["score_delta_vs_baseline"] == pytest.approx(
        row["segnet_delta"] + row["posenet_delta"] + row["rate_delta"]
    )
    assert row["baseline_archive_size_bytes"] == baseline_size
    assert row["archive_byte_delta_vs_baseline"] == -1


def test_write_rows_and_summary_validate_with_dynamic_observation_loader(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    baseline_path = tmp_path / "baseline.json"
    advisory_path = tmp_path / "candidate" / "local_cpu_advisory.json"
    harvest_path = tmp_path / "harvest.json"
    acquisition_path = tmp_path / "pairset_acquisition.json"
    output = tmp_path / "observations.jsonl"

    _write_json(
        baseline_path,
        _advisory(
            archive_sha256=_sha("1"),
            archive_size_bytes=178_592,
            raw_sha256=_sha("2"),
            runtime_sha256=_sha("3"),
            seg=0.055988,
        ),
    )
    _write_json(
        advisory_path,
        _advisory(
            archive_sha256=_sha("4"),
            archive_size_bytes=178_558,
            raw_sha256=_sha("5"),
            runtime_sha256=_sha("6"),
            seg=0.055989,
        ),
    )
    _write_json(
        harvest_path,
        {
            "schema": "dqs1_local_first_harvest.v1",
            "candidate_id": "pairset_drop_two_r029_017_p0259_0242",
            "candidate_archive_sha256": _sha("4"),
            "local_cpu_advisory_path": str(advisory_path),
            "harvested_at_utc": "20260523T114957Z",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    )
    _write_json(
        acquisition_path,
        {
            "candidates": [
                {
                    "acquisition_id": "pairset_drop_two_r029_017_p0259_0242",
                    "selector_kind": "drop_two_from_best",
                    "selected_pair_indices": [26, 59],
                    "acquisition_operation": {
                        "op": "drop_two",
                        "dropped_pair_ranks": [17, 29],
                        "dropped_pair_indices": [242, 259],
                    },
                }
            ],
        },
    )

    rows = build_observation_rows_from_harvests(
        [harvest_path],
        repo_root=repo_root,
        pairset_acquisition_path=acquisition_path,
        baseline_advisory_path=baseline_path,
        baseline_archive_size_bytes=178_560,
    )
    write_observation_jsonl(rows, output_path=output)
    loaded = load_observation_rows(output)
    summary = build_harvest_observation_summary(
        loaded,
        jsonl_path=output,
        repo_root=repo_root,
    )

    assert loaded == rows
    assert summary["schema"] == "dqs1_local_first_harvest_observations.v1"
    assert summary["row_count"] == 1
    assert summary["best_local_advisory"]["candidate_id"] == (
        "pairset_drop_two_r029_017_p0259_0242"
    )
    assert summary["score_claim"] is False


def test_missing_pairset_acquisition_identity_fails_closed(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    advisory_path = tmp_path / "candidate" / "local_cpu_advisory.json"
    harvest_path = tmp_path / "harvest.json"
    acquisition_path = tmp_path / "pairset_acquisition.json"

    _write_json(
        baseline_path,
        _advisory(
            archive_sha256=_sha("a"),
            archive_size_bytes=178_592,
            raw_sha256=_sha("b"),
            runtime_sha256=_sha("c"),
            seg=0.055988,
        ),
    )
    _write_json(
        advisory_path,
        _advisory(
            archive_sha256=_sha("d"),
            archive_size_bytes=178_559,
            raw_sha256=_sha("e"),
            runtime_sha256=_sha("f"),
            seg=0.055989,
        ),
    )
    _write_json(
        harvest_path,
        {
            "schema": "dqs1_local_first_harvest.v1",
            "candidate_id": "unknown_candidate",
            "candidate_archive_sha256": _sha("d"),
            "local_cpu_advisory_path": str(advisory_path),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    )
    _write_json(acquisition_path, {"candidates": []})

    with pytest.raises(
        DQS1LocalHarvestObservationError,
        match="no pairset acquisition candidate",
    ):
        build_observation_rows_from_harvests(
            [harvest_path],
            repo_root=tmp_path,
            pairset_acquisition_path=acquisition_path,
            baseline_advisory_path=baseline_path,
            baseline_archive_size_bytes=178_560,
        )


def test_harvest_input_glob_ignores_generated_observation_artifacts(tmp_path: Path) -> None:
    harvest = tmp_path / "dqs1_local_first_harvest_pairset_drop_two_20260523T000000Z.json"
    generated_summary = (
        tmp_path / "dqs1_local_first_harvest_observations_20260523T000000Z.summary.json"
    )
    harvest.write_text("{}", encoding="utf-8")
    generated_summary.write_text("{}", encoding="utf-8")

    paths = _expand_harvest_inputs([str(tmp_path / "dqs1_local_first_harvest_*.json")])

    assert paths == [harvest]
