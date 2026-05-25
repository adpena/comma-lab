# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from comma_lab.artifact_retention import sha256_file
from tools import run_dqs1_local_first_autopilot as autopilot


def _local_advisory_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "score_recomputed_from_components": 0.192010,
        "evidence_grade": "macOS-CPU advisory",
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "n_samples": 600,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    payload.update(overrides)
    return payload


def _write_certified_local_cpu_scratch(candidate: Path, *, mutate_manifest: bool = False) -> None:
    work = candidate / "local_cpu_advisory_work"
    inflated = work / "inflated"
    extracted = work / "extracted"
    inflated.mkdir(parents=True)
    extracted.mkdir(parents=True)
    (inflated / "0.raw").write_bytes(b"raw")
    (extracted / "archive.zip").write_bytes(b"zip")
    (work / "archive.zip").write_bytes(b"zip")
    (work / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                **_local_advisory_payload(),
                "promotable": False,
            }
        ),
        encoding="utf-8",
    )
    raw_sha = "0" * 64 if mutate_manifest else sha256_file(inflated / "0.raw")
    (work / "inflated_outputs_manifest.json").write_text(
        json.dumps({"payload": {"files": [{"path": "0.raw", "sha256": raw_sha}]}}),
        encoding="utf-8",
    )
    (work / "provenance.json").write_text(json.dumps({"command": ["inflate"]}), encoding="utf-8")


def test_post_harvest_retention_moves_certified_candidate_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    candidate = repo / "results" / "materialized" / "good_candidate"
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    _write_certified_local_cpu_scratch(candidate)

    result = autopilot._execute_candidate_artifact_retention(
        candidate_root=candidate,
        candidate_id="good_candidate",
        stamp="20260523T000000Z",
        action="move",
        cold_store_roots=[cold_store],
        min_bytes=1,
        include_mlx_cache=False,
        repo_root=repo,
    )

    assert result["candidate_count"] == 2
    assert result["blocked_candidate_count"] == 0
    assert result["executed_count"] == 2
    assert not (candidate / "local_cpu_advisory_work" / "inflated").exists()
    assert not (candidate / "local_cpu_advisory_work" / "extracted").exists()
    assert (cold_store / "results" / "materialized" / "good_candidate").is_dir()
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert payload["schema"] == "dqs1_local_first_artifact_retention.v1"
    assert payload["execution"]["cold_store_contract"]["write_probe_passed"] is True
    assert payload["score_claim"] is False
    assert Path(str(result["journal_path"])).is_file()


def test_post_harvest_retention_blocks_uncertified_raw_without_deleting(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    candidate = repo / "results" / "materialized" / "bad_candidate"
    cold_store = tmp_path / "cold"
    cold_store.mkdir()
    _write_certified_local_cpu_scratch(candidate, mutate_manifest=True)

    with pytest.raises(autopilot.ExperimentQueueError, match="artifact retention blocked"):
        autopilot._execute_candidate_artifact_retention(
            candidate_root=candidate,
            candidate_id="bad_candidate",
            stamp="20260523T000000Z",
            action="move",
            cold_store_roots=[cold_store],
            min_bytes=1,
            include_mlx_cache=False,
            repo_root=repo,
        )

    assert (candidate / "local_cpu_advisory_work" / "inflated" / "0.raw").is_file()
    artifact = repo / ".omx/research/dqs1_artifact_retention_bad_candidate_20260523T000000Z.json"
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["plan"]["blocked_candidate_count"] == 1


def test_free_disk_probe_accepts_uncreated_external_results_root(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "external" / "pact_experiments" / "frontier_cycle"

    free_gb = autopilot._free_disk_gb(nested)

    assert free_gb > 0


def test_succeeded_candidate_ids_survive_active_batch_tail() -> None:
    queue = {
        "experiments": [
            {"id": "candidate_done"},
            {"id": "candidate_queued"},
            {"id": "candidate_partial"},
        ],
    }
    summary = {
        "steps": [
            {
                "experiment_id": "candidate_done",
                "step_id": "materialize",
                "status": "succeeded",
            },
            {
                "experiment_id": "candidate_done",
                "step_id": "local_cpu_advisory",
                "status": "succeeded",
            },
            {
                "experiment_id": "candidate_queued",
                "step_id": "materialize",
                "status": "queued",
            },
            {
                "experiment_id": "candidate_partial",
                "step_id": "materialize",
                "status": "succeeded",
            },
            {
                "experiment_id": "candidate_partial",
                "step_id": "local_cpu_advisory",
                "status": "queued",
            },
        ]
    }

    assert autopilot._succeeded_candidate_ids(
        queue,
        summary,
        already_harvested=set(),
    ) == ["candidate_done"]
    assert (
        autopilot._succeeded_candidate_ids(
            queue,
            summary,
            already_harvested={"candidate_done"},
        )
        == []
    )
