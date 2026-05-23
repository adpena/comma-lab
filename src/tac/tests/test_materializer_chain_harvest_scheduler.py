# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
)
from comma_lab.scheduler.experiment_queue import (
    QUEUE_SCHEMA,
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
)
from comma_lab.scheduler.materializer_chain_harvest import (
    HARVEST_SCHEMA,
    harvest_materializer_chain_manifests,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "harvest_materializer_chain_candidates.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_bytes(path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _chain_manifest(
    external_root: Path,
    *,
    authority_overrides: dict[str, object] | None = None,
) -> Path:
    source_archive = _write_bytes(
        external_root / "source" / "archive.zip",
        b"source archive bytes",
    )
    candidate_archive = _write_bytes(
        external_root / "candidate" / "archive.zip",
        b"candidate",
    )
    artifact = _write_json(
        external_root / "candidate" / "candidate_manifest.json",
        {"schema": "fixture_materializer_artifact_v1"},
    )
    artifact_record = _artifact_record(artifact)
    payload: dict[str, object] = {
        "schema": CHAIN_SCHEMA,
        "candidate_id": "external_materializer_candidate",
        "lane_id": "fixture_materializer_chain_harvest",
        "source_archive": source_archive,
        "source_archive_sha256": source_archive["sha256"],
        "source_archive_bytes": source_archive["bytes"],
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "serialized_archive_delta": build_serialized_archive_delta_contract(
            source_archive=source_archive,
            candidate_archive=candidate_archive,
            require_realized_saving=True,
        ),
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_proof_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "readiness_blockers": ["exact_cuda_auth_eval_missing"],
        "dispatch_blockers": [
            "byte_range_entropy_recode_chain_is_not_dispatch_authorization",
            "exact_cuda_auth_eval_missing",
        ],
        "artifacts": {"candidate_manifest": artifact_record},
        "chain_steps": [
            {
                "step_id": "materialize_candidate",
                "status": "succeeded",
                "artifact": artifact_record,
            }
        ],
        "next_required_gates": ["contest_auth_eval"],
        **_false_authority(),
    }
    if authority_overrides:
        payload.update(authority_overrides)
    return _write_json(external_root / CHAIN_MANIFEST_NAME, payload)


def _work_queue(repo: Path, chain_manifest: Path) -> Path:
    payload = {
        "schema": MATERIALIZER_WORK_QUEUE_SCHEMA,
        "tool": "fixture",
        "row_count": 1,
        "executable_row_count": 1,
        "blocked_row_count": 0,
        "rows": [
            {
                "schema": "byte_shaving_materializer_work_row.v1",
                "work_id": "Materializer Work Fixture",
                "work_rank": 1,
                "backlog_key": "fixture",
                "unit_kind": "byte_range",
                "operation_family": "entropy_recode",
                "target_kind": "byte_range_entropy_recode_v1",
                "resource_kind": "local_cpu",
                "executable": True,
                "postconditions": [
                    {
                        "type": "materializer_chain_complete",
                        "path": str(chain_manifest),
                        "schema": CHAIN_SCHEMA,
                    }
                ],
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }
    return _write_json(repo / "materializer_work_queue.json", payload)


def _state_path(repo: Path, *, status: str) -> Path:
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": "materializer_execution_queue",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
            "experiments": [
                {
                    "id": "materializer_work_fixture",
                    "steps": [
                        {
                            "id": MATERIALIZER_EXECUTION_STEP_ID,
                            "command": [sys.executable, "-c", "print('fixture')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                }
            ],
        }
    )
    state = repo / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = ?, attempts = 1, last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                status,
                json.dumps({"fixture": True}),
                "materializer_execution_queue",
                "materializer_work_fixture",
                MATERIALIZER_EXECUTION_STEP_ID,
            ),
        )
        conn.commit()
    return state


def test_harvest_work_queue_chain_manifest_into_source_queue(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "VertigoDataTier"
    chain = _chain_manifest(external)
    work_queue = _work_queue(repo, chain)
    state = _state_path(repo, status="succeeded")

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        experiment_queue_state_path=state,
        experiment_queue_id="materializer_execution_queue",
    )

    report = result["report"]
    queue = result["source_queue"]
    row = queue["top_k"][0]
    assert report["schema"] == HARVEST_SCHEMA
    assert report["accepted_manifest_count"] == 1
    assert report["source_queue_candidate_count"] == 1
    assert report["source_queue_dispatch_ready_count"] == 0
    assert report["rows"][0]["state_rows"][0]["status"] == "succeeded"
    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "external_materializer_candidate"
    assert row["archive_candidate_verified"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert str(external) in row["candidate_archive_path"]


def test_harvest_requires_succeeded_state_for_work_queue_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    work_queue = _work_queue(repo, chain)
    state = _state_path(repo, status="failed")

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        experiment_queue_state_path=state,
        experiment_queue_id="materializer_execution_queue",
    )

    assert result["report"]["accepted_manifest_count"] == 0
    assert result["source_queue"]["n_candidates"] == 0
    assert result["report"]["rows"][0]["blockers"] == [
        "experiment_queue_state_not_succeeded:Materializer Work Fixture:failed"
    ]


def test_harvest_validates_manifest_even_when_state_succeeded(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(
        tmp_path / "external",
        authority_overrides={"score_claim": True},
    )
    work_queue = _work_queue(repo, chain)
    state = _state_path(repo, status="succeeded")

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        experiment_queue_state_path=state,
        experiment_queue_id="materializer_execution_queue",
    )

    assert result["report"]["accepted_manifest_count"] == 0
    assert result["source_queue"]["n_candidates"] == 0
    assert "score_claim=truthy" in result["report"]["rows"][0]["blockers"][0]


def test_harvest_cli_writes_report_and_source_queue(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    report_out = repo / "harvest_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo-root",
            str(repo),
            "--chain-manifest",
            str(chain),
            "--source-queue-out",
            str(source_queue_out),
            "--report-out",
            str(report_out),
            "--require-accepted",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source_queue = json.loads(source_queue_out.read_text(encoding="utf-8"))
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert source_queue["n_candidates"] == 1
    assert source_queue["dispatch_ready_count"] == 0
    assert report["accepted_manifest_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
