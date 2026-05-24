# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

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
    EXACT_READINESS_BRIDGE_SCHEMA,
    HARVEST_SCHEMA,
    harvest_materializer_chain_manifests,
    run_exact_readiness_bridge_for_harvested_queue,
)
from comma_lab.scheduler.materializer_chain_harvest import (
    write_json as write_harvest_json,
)
from comma_lab.scheduler.materializer_exact_eval_dispatch_plan import (
    DISPATCH_PLAN_SCHEMA,
    build_materializer_exact_eval_dispatch_plan,
)
from comma_lab.scheduler.materializer_exact_eval_dispatch_plan import (
    write_json as write_dispatch_plan_json,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import truthy_authority_field_violations
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "harvest_materializer_chain_candidates.py"
DISPATCH_PLAN_TOOL = REPO_ROOT / "tools" / "build_materializer_exact_eval_dispatch_plan.py"


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


def _write_zip_archive(path: Path, member: str, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(info, data, compress_type=zipfile.ZIP_STORED)
    return _artifact_record(path)


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


def _write_runtime_bound_pr101_proof(
    submission: Path,
    *,
    archive_sha: str,
    stale_inflate_sh_sha: bool = False,
) -> Path:
    inflate_sh_sha = hashlib.sha256((submission / "inflate.sh").read_bytes()).hexdigest()
    if stale_inflate_sh_sha:
        inflate_sh_sha = "0" * 64 if inflate_sh_sha != "0" * 64 else "1" * 64
    inflate_py_sha = hashlib.sha256((submission / "inflate.py").read_bytes()).hexdigest()
    manifest_path = _write_json(
        submission / "runtime_packet_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_packet_v1",
            "packet_dir": str(submission),
            "runtime_custody": {
                "runtime_files": [
                    {"relpath": "inflate.sh", "sha256": inflate_sh_sha},
                    {"relpath": "inflate.py", "sha256": inflate_py_sha},
                ],
            },
        },
    )
    return _write_json(
        submission / "runtime_consumption_proof.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
            "proof_kind": "fixture_runtime_bound_pr101_proof",
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "packet_dir": str(submission),
            "runtime_consumption_proven_for_supported_bias_params": True,
            "inflate_sh_routes_to_packet_inflate_py": True,
            "archive_unchanged_proof": {
                "archive_sha256": archive_sha,
            },
            "inflate_wrapper_route_proof": {
                "wrapper_invoked_packet_inflate_py": True,
                "inflate_sh_sha256": inflate_sh_sha,
                "packet_inflate_py_sha256": inflate_py_sha,
            },
            "inflate_static_bias_patch_proof": {
                "inflate_sha256": inflate_py_sha,
            },
            "inflate_runtime_bias_logic_proof": {
                "packet_inflate_function_executed": True,
                "inflate_py_sha256": inflate_py_sha,
            },
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
    )


def _exact_ready_chain_manifest(
    repo: Path,
    *,
    stale_runtime_proof: bool = False,
) -> Path:
    source_archive = _write_zip_archive(
        repo / "source" / "archive.zip",
        "0.bin",
        b"source archive payload that is deliberately larger",
    )
    submission = repo / "submission"
    candidate_archive = _write_zip_archive(
        submission / "archive.zip",
        "0.bin",
        b"candidate",
    )
    inflate = submission / "inflate.sh"
    inflate_py = submission / "inflate.py"
    inflate_py.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(b'')\n",
        encoding="utf-8",
    )
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "python \"$SCRIPT_DIR/inflate.py\" \"$1\" \"$2\"\n",
        encoding="utf-8",
    )
    inflate.chmod(inflate.stat().st_mode | 0o100)
    (submission / "report.txt").write_text(
        "candidate archive ready for exact eval\n",
        encoding="utf-8",
    )
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": candidate_archive["sha256"],
            "candidate_archive_bytes": candidate_archive["bytes"],
            "candidate_archive": {"member_name": "0.bin"},
        },
    )
    runtime_proof = _write_runtime_bound_pr101_proof(
        submission,
        archive_sha=str(candidate_archive["sha256"]),
        stale_inflate_sh_sha=stale_runtime_proof,
    )
    (repo / "upstream").mkdir(parents=True, exist_ok=True)
    (repo / "upstream" / "evaluate.py").write_text("# fixture\n", encoding="utf-8")
    artifact = _write_json(
        submission / "candidate_manifest.json",
        {"schema": "fixture_materializer_artifact_v1"},
    )
    artifact_record = _artifact_record(artifact)
    payload: dict[str, object] = {
        "schema": CHAIN_SCHEMA,
        "candidate_id": "exact_ready_materializer_candidate",
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
        "readiness_blockers": [],
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
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": runtime_proof.relative_to(repo).as_posix(),
        **_false_authority(),
    }
    return _write_json(repo / "chain" / CHAIN_MANIFEST_NAME, payload)


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


def test_exact_readiness_bridge_promotes_valid_harvested_source_queue(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["schema"] == EXACT_READINESS_BRIDGE_SCHEMA
    assert bridge["ready_candidate_count"] == 1
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(bridge) == []
    row = bridge["rows"][0]
    assert row["candidate_id"] == "exact_ready_materializer_candidate"
    assert row["readiness_verdict"] == "exact_ready_queue_written"
    assert row["exact_ready_queue_written"] is True
    exact_ready_queue = json.loads(
        (repo / row["exact_ready_queue_path"]).read_text(encoding="utf-8")
    )
    assert exact_ready_queue["dispatch_ready_count"] == 1
    promoted = exact_ready_queue["dispatch_ready"][0]
    assert promoted["ready_for_exact_eval_dispatch"] is True
    assert promoted["score_claim"] is False
    assert promoted["dispatch_claim_required_before_gpu_or_remote_eval"] is True


def test_materializer_dispatch_plan_dry_run_does_not_write_claim(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    dispatch_queue = result["experiment_queue"]
    assert plan["schema"] == DISPATCH_PLAN_SCHEMA
    assert plan["authorized_candidate_count"] == 1
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(plan) == []
    steps = dispatch_queue["experiments"][0]["steps"]
    claim_command = steps[0]["command"]
    dispatch_command = steps[1]["command"]
    assert steps[0]["id"] == "claim_lane_dispatch"
    assert "--dry-run" in claim_command
    assert "--dry-run" in dispatch_command
    assert "--claim-policy" not in dispatch_command


def test_materializer_dispatch_plan_execute_requires_active_claim_for_dispatch(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        dispatch_mode="execute",
        allow_paid_dispatch_queue=True,
        provider="lightning",
        label_prefix="fixture_materializer_exact_eval",
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    dispatch_queue = result["experiment_queue"]
    assert plan["authorized_candidate_count"] == 1
    assert "execute_dispatch_queue_created_requires_operator_review" in plan["plan_blockers"]
    steps = dispatch_queue["experiments"][0]["steps"]
    claim_command = steps[0]["command"]
    dispatch_command = steps[1]["command"]
    job_id = plan["rows"][0]["dispatch_job_id"]
    assert "--dry-run" not in claim_command
    assert "--claim-policy" in dispatch_command
    assert "require_active_claim" in dispatch_command
    assert "--required-claim-platform" in dispatch_command
    assert "lightning" in dispatch_command
    assert "--required-claim-instance-job-id" in dispatch_command
    assert job_id in dispatch_command


def test_materializer_dispatch_plan_freezes_queue_when_cost_cap_exceeded(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
        estimated_cost_per_dispatch=0.30,
        max_total_cost=0.01,
    )

    plan = result["plan"]
    dispatch_queue = result["experiment_queue"]
    assert plan["authorized_candidate_count"] == 1
    assert plan["experiment_count"] == 0
    assert plan["hard_plan_blockers"] == ["estimated_total_cost_exceeds_cap:0.30>0.01"]
    assert dispatch_queue["experiments"][0]["id"] == (
        "frozen_materializer_exact_eval_dispatch"
    )
    assert dispatch_queue["experiments"][0]["steps"][0]["id"] == "noop"


def test_exact_readiness_bridge_writes_blocked_report_without_ready_queue(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["ready_candidate_count"] == 0
    assert bridge["blocked_candidate_count"] == 1
    row = bridge["rows"][0]
    assert row["exact_ready_queue_path"] is None
    assert row["blockers"]
    assert (repo / row["exact_readiness_report_path"]).is_file()


def test_exact_readiness_bridge_blocks_stale_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo, stale_runtime_proof=True)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["ready_candidate_count"] == 0
    blockers = bridge["rows"][0]["blockers"]
    assert "runtime_consumption_proof_inflate_sh_sha_mismatch" in blockers
    assert bridge["rows"][0]["exact_ready_queue_path"] is None


def test_exact_readiness_bridge_refuses_unallowlisted_source_blocker_clear(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    with pytest.raises(
        Exception,
        match="exact_readiness_extra_source_blocker_not_allowlisted:"
        "candidate_archive_missing",
    ):
        run_exact_readiness_bridge_for_harvested_queue(
            repo_root=repo,
            source_queue_path=source_queue_out,
            exact_readiness_out_dir=repo / "exact_readiness",
            allow_source_blockers=["candidate_archive_missing"],
            operator_override_reason="fixture",
            active_floor_archive_bytes=None,
        )


def test_exact_readiness_bridge_refuses_duplicate_candidate_ids(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    queue = harvest_result["source_queue"]
    queue["top_k"].append(dict(queue["top_k"][0]))
    _write_json(source_queue_out, queue)

    with pytest.raises(
        Exception,
        match="exact_readiness_duplicate_candidate_id:"
        "exact_ready_materializer_candidate",
    ):
        run_exact_readiness_bridge_for_harvested_queue(
            repo_root=repo,
            source_queue_path=source_queue_out,
            exact_readiness_out_dir=repo / "exact_readiness",
            active_floor_archive_bytes=None,
        )


def test_harvest_cli_can_run_explicit_exact_readiness_bridge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    report_out = repo / "harvest_report.json"
    bridge_report_out = repo / "bridge_report.json"

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
            "--exact-readiness-out-dir",
            str(repo / "exact_readiness"),
            "--exact-readiness-bridge-report-out",
            str(bridge_report_out),
            "--require-accepted",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    bridge = json.loads(bridge_report_out.read_text(encoding="utf-8"))
    assert bridge["schema"] == EXACT_READINESS_BRIDGE_SCHEMA
    assert bridge["candidate_count"] == 1
    assert bridge["ready_candidate_count"] == 0
    assert "exact-readiness bridge: ready=0/1" in completed.stdout


def test_harvest_cli_refuses_require_ready_without_bridge_dir(tmp_path: Path) -> None:
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
            "--exact-readiness-require-ready",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--exact-readiness-out-dir is required" in completed.stderr


def test_dispatch_plan_builds_claim_then_dry_run_queue_from_bridge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    queue = result["experiment_queue"]
    assert plan["schema"] == DISPATCH_PLAN_SCHEMA
    assert plan["authorized_candidate_count"] == 1
    assert plan["blocked_candidate_count"] == 0
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(plan) == []
    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["controls"]["mode"] == "paused"
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["candidate_id"] == "exact_ready_materializer_candidate"
    claim_step, dispatch_step = experiment["steps"]
    assert claim_step["id"] == "claim_lane_dispatch"
    assert "tools/claim_lane_dispatch.py" in claim_step["command"]
    assert dispatch_step["requires"] == ["claim_lane_dispatch"]
    assert dispatch_step["id"] == "dispatch_exact_eval_dry_run"
    assert "tools/parallel_dispatch_top_k.py" in dispatch_step["command"]
    assert "--dry-run" in dispatch_step["command"]


def test_dispatch_plan_execute_mode_requires_explicit_paid_queue_flag(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    exact_ready_queue = repo / "exact_ready_queue.json"
    _write_json(
        exact_ready_queue,
        {
            "schema": "optimizer_candidate_queue_v1",
            "dispatch_ready_count": 0,
            "dispatch_ready": [],
            "top_k": [],
        },
    )

    with pytest.raises(
        Exception,
        match="execute_dispatch_queue_requires_allow_paid_dispatch_queue",
    ):
        build_materializer_exact_eval_dispatch_plan(
            repo_root=repo,
            exact_ready_queue_paths=[exact_ready_queue],
            dispatch_mode="execute",
        )


def test_materializer_handoff_json_writes_refuse_overwrite_by_default(
    tmp_path: Path,
) -> None:
    harvest_path = tmp_path / "harvest.json"
    dispatch_path = tmp_path / "dispatch.json"

    write_harvest_json(harvest_path, {"value": 1})
    write_dispatch_plan_json(dispatch_path, {"value": 1})

    with pytest.raises(Exception, match="refusing_to_overwrite_json"):
        write_harvest_json(harvest_path, {"value": 2})
    with pytest.raises(Exception, match="refusing_to_overwrite_json"):
        write_dispatch_plan_json(dispatch_path, {"value": 2})

    write_harvest_json(harvest_path, {"value": 2}, overwrite=True)
    write_dispatch_plan_json(dispatch_path, {"value": 2}, overwrite=True)

    assert json.loads(harvest_path.read_text(encoding="utf-8"))["value"] == 2
    assert json.loads(dispatch_path.read_text(encoding="utf-8"))["value"] == 2


def test_dispatch_plan_cli_preserves_default_active_floor_guards(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_materializer_exact_eval_dispatch_plan_fixture",
        DISPATCH_PLAN_TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    def fake_build_materializer_exact_eval_dispatch_plan(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "plan": {
                "authorized_candidate_count": 0,
                "blocked_candidate_count": 0,
                "dispatch_mode": "dry_run",
            },
            "experiment_queue": {"schema": QUEUE_SCHEMA, "experiments": []},
        }

    module.build_materializer_exact_eval_dispatch_plan = (
        fake_build_materializer_exact_eval_dispatch_plan
    )
    module.write_json = lambda path, payload, **kwargs: None

    rc = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--exact-ready-queue",
            str(tmp_path / "exact_ready_queue.json"),
            "--dispatch-plan-out",
            str(tmp_path / "dispatch_plan.json"),
            "--experiment-queue-out",
            str(tmp_path / "dispatch_queue.json"),
        ]
    )

    assert rc == 0
    assert (
        captured["active_floor_archive_bytes"]
        == module.ACTIVE_FLOOR_ARCHIVE_BYTES
    )
    assert captured["active_floor_score"] == module.ACTIVE_FLOOR_SCORE


def test_dispatch_plan_cli_writes_paused_dry_run_queue_from_bridge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)
    plan_out = repo / "dispatch_plan.json"
    queue_out = repo / "dispatch_queue.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(DISPATCH_PLAN_TOOL),
            "--repo-root",
            str(repo),
            "--bridge-report",
            str(bridge_path),
            "--dispatch-plan-out",
            str(plan_out),
            "--experiment-queue-out",
            str(queue_out),
            "--active-floor-archive-bytes",
            "0",
            "--require-authorized",
            "--allow-above-active-floor-dispatch",
            "--operator-override-reason",
            "fixture",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    queue = json.loads(queue_out.read_text(encoding="utf-8"))
    assert plan["authorized_candidate_count"] == 1
    assert plan["dispatch_mode"] == "dry_run"
    assert queue["controls"]["mode"] == "paused"
    assert "authorized=1 blocked=0" in completed.stdout
