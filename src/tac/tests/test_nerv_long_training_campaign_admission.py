# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import normalize_queue_definition
from tac.analysis.nerv_long_training_campaign_admission import (
    ADMISSION_SCHEMA,
    DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS,
    build_nerv_long_training_campaign_execution_admission,
)
from tac.analysis.nerv_long_training_campaign_plan import (
    build_nerv_long_training_campaign_plan,
)
from tac.analysis.pr95_distortion_practices_guard import (
    AXIS_TRACE_CONTRACT_SCHEMA,
    POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA,
    PRACTICE_DAG_SCHEMA,
    SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA,
    STAGE_DAG_SCHEMA,
)
from tac.cathedral_consumers.nerv_long_training_campaign_consumer import consume_candidate

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nerv_long_training_campaign_admission_builds_storage_gated_queue(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["schema"] == ADMISSION_SCHEMA
    assert admission["experiment_queue_ready"] is True
    assert admission["local_mlx_execution_ready"] is True
    assert admission["admitted_experiment_count"] == 1
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False
    assert admission["blockers"] == []
    queue = normalize_queue_definition(admission["experiment_queue"])
    assert queue["queue_id"] == "nerv_manifest_pinned_long_training_local_mlx_admission.v1"
    assert queue["experiments"][0]["id"] == "nerv_campaign_storage_preflight"
    selected = queue["experiments"][1]
    assert selected["steps"][0]["requires"] == [
        "nerv_campaign_storage_preflight.proactive_cleanup"
    ]
    assert selected["steps"][0]["resources"]["kind"] == "local_mlx"
    assert selected["steps"][0]["timeout_seconds"] == (
        DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS
    )
    command = selected["steps"][0]["command"]
    output_dir = Path(command[command.index("--output-dir") + 1])
    artifact_paths = set(selected["steps"][0]["telemetry"]["artifact_paths"])
    assert (
        output_dir / "compact_renderer_mlx_spine_runner_report.json"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "compact_renderer_mlx_spine_runner_startup.json"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl"
    ).as_posix() in artifact_paths
    assert (
        output_dir / "hi_nerv_mlx_training" / "nerv_crux_trace_rows.json"
    ).as_posix() in artifact_paths
    json_postcondition_paths = {
        condition["path"]
        for condition in selected["steps"][0]["postconditions"]
        if condition["type"].startswith("json_")
    }
    assert json_postcondition_paths == {
        (output_dir / "compact_renderer_mlx_spine_runner_report.json").as_posix()
    }
    assert output_dir.as_posix() not in json_postcondition_paths
    assert selected["metadata"]["human_visual_fidelity_relevance"] == (
        "irrelevant_unless_scorer_causal"
    )
    source_row = selected["metadata"]["source_selected_row"]
    axis_contract = _source_row_contract(
        source_row,
        "pr95_distortion_axis_trace_contract",
    )
    assert axis_contract["schema"] == AXIS_TRACE_CONTRACT_SCHEMA
    assert axis_contract["required_axes"] == [
        "live_forward",
        "fakequant_forward",
        "archive_parseback",
        "inflate_replay",
        "official_evaluate_py",
    ]
    pose_contract = _source_row_contract(
        source_row,
        "pr95_posenet_marginal_telemetry_contract",
    )
    assert pose_contract["schema"] == POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA
    assert pose_contract["pose_marginal_formula"] == "5/sqrt(10*d_pose)"
    actuator_contract = _source_row_contract(
        source_row,
        "pr95_scorer_atom_actuator_contract",
    )
    assert actuator_contract["schema"] == SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA
    assert "pair_local_film_or_latent_adapter" in actuator_contract[
        "family_actuators"
    ]
    row_guard = admission["selected_rows"][0]["pr95_distortion_practices_guard"]
    assert row_guard["schema"] == "pr95_distortion_practices_guard.v1"
    assert row_guard["launch_allowed"] is True
    assert row_guard["blockers"] == []
    practice_rows = {row["practice_id"]: row for row in row_guard["practice_rows"]}
    assert practice_rows["archive_parseback_distortion_axis_trace"]["observed"] is True
    assert row_guard["practice_dag"]["schema"] == PRACTICE_DAG_SCHEMA
    assert row_guard["practice_dag"]["all_nodes_green"] is True
    assert row_guard["optimization_stage_dag"]["schema"] == STAGE_DAG_SCHEMA
    assert row_guard["optimization_stage_dag"][
        "all_required_stage_signals_observed"
    ] is True
    assert admission["pr95_distortion_source_inventory"]["source_ready"] is True


def test_nerv_long_training_campaign_admission_blocks_without_active_claim(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    claims = tmp_path / "claims.md"
    claims.write_text("# empty\n", encoding="utf-8")

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["experiment_queue"] is None
    assert admission["admitted_experiment_count"] == 0
    assert "active_lane_claim_missing_or_terminal" in admission["blockers"]
    assert admission["score_claim"] is False


def test_nerv_long_training_campaign_admission_blocks_non_ssd_output(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "local_disk")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert "selected_row_output_dir_not_on_allowed_ssd_tier" in admission["blockers"]


def test_nerv_long_training_campaign_admission_blocks_existing_output_artifacts(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    selected = verdict["selected_local_mlx_experiments"][0]
    command = selected["command"]
    out_dir = Path(command[command.index("--output-dir") + 1])
    telemetry = out_dir / "hi_nerv_mlx_training" / "telemetry.jsonl"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text('{"epoch": 1}\n', encoding="utf-8")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["admitted_experiment_count"] == 0
    assert "selected_row_output_dir_contains_prior_training_artifacts" in admission[
        "blockers"
    ]
    row = admission["selected_rows"][0]
    assert telemetry.as_posix() in row["existing_output_artifact_paths"]
    assert row["admitted"] is False


def test_nerv_long_training_campaign_admission_blocks_active_local_mlx_process(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        active_local_mlx_processes=(
            {
                "pid": 12345,
                "ppid": 1,
                "stat": "R",
                "etime": "00:12",
                "command": (
                    "python tools/run_compact_renderer_mlx_spine_runner.py "
                    "--execute-family hi_nerv --output-dir /Volumes/VertigoDataTier/pact/live"
                ),
            },
        ),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["local_mlx_execution_ready"] is False
    assert admission["admitted_experiment_count"] == 0
    assert "active_local_mlx_training_process_present" in admission["blockers"]
    assert admission["active_local_mlx_process_count"] == 1
    assert admission["active_local_mlx_processes"][0]["pid"] == 12345
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False


def test_nerv_long_training_campaign_admission_blocks_missing_pr95_distortion_practice(
    tmp_path: Path,
) -> None:
    verdict = _runnable_verdict(tmp_path / "ssd")
    selected = verdict["selected_local_mlx_experiments"][0]
    command = selected["command"]
    command[command.index("--pose-distillation-weight") + 1] = "0"
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    admission = build_nerv_long_training_campaign_execution_admission(
        verdict,
        repo_root=tmp_path,
        active_claims_path=claims,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
        limit=1,
        storage_expected_bytes_per_row=1024,
        storage_reserve_free_gb=0.0,
        allowed_output_roots=(tmp_path / "ssd",),
        now_utc="2026-06-02T18:40:00Z",
    )

    assert admission["experiment_queue_ready"] is False
    assert admission["admitted_experiment_count"] == 0
    assert (
        "hi_nerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing"
        in admission["blockers"]
    )
    row_guard = admission["selected_rows"][0]["pr95_distortion_practices_guard"]
    assert row_guard["launch_allowed"] is False
    assert "hi_nerv_pr95_distortion_dual_component_real_scorer_pressure_missing" in row_guard[
        "blockers"
    ]


def test_nerv_long_training_campaign_admission_cli_writes_artifacts(
    tmp_path: Path,
) -> None:
    verdict_path = tmp_path / "verdict.json"
    out_json = tmp_path / "admission.json"
    out_md = tmp_path / "admission.md"
    out_queue = tmp_path / "queue.json"
    verdict_path.write_text(
        json.dumps(_runnable_verdict(tmp_path / "ssd")),
        encoding="utf-8",
    )
    claims = _claims_file(
        tmp_path,
        lane_id="lane_nerv_local_mlx",
        instance_job_id="job_first",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/build_nerv_long_training_campaign_execution_admission.py"),
            "--consumer-verdict",
            str(verdict_path),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--lane-id",
            "lane_nerv_local_mlx",
            "--instance-job-id",
            "job_first",
            "--active-claims-path",
            str(claims),
            "--storage-expected-bytes-per-row",
            "1024",
            "--storage-reserve-free-gb",
            "0",
            "--local-mlx-timeout-seconds",
            "777",
            "--allowed-output-root",
            str(tmp_path / "ssd"),
            "--skip-active-local-mlx-process-scan",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["experiment_queue_ready"] is True
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == ADMISSION_SCHEMA
    assert payload["score_claim"] is False
    assert out_md.read_text(encoding="utf-8").startswith(
        "# NeRV Long-Training Campaign Execution Admission"
    )
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue["schema"] == "experiment_queue.v1"
    selected = queue["experiments"][1]
    assert selected["steps"][0]["timeout_seconds"] == 777


def _campaign_plan(output_root: Path) -> dict:
    return build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget={
            "schema": "nerv_modelsize_budget.v1",
            "selected_candidates": [
                {
                    "schema": "hinerv_modelsize_candidate.v1",
                    "family": "hi_nerv",
                    "candidate_id": "hinerv_tiny",
                    "num_pairs": 600,
                    "hard_byte_ceiling": 178_000,
                    "decoder_codec": "int4_mixed",
                    "nominal_total_payload_bytes": 120_000,
                    "nominal_under_ceiling": True,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        snerv_modelsize_budget={
            "schema": "snerv_modelsize_budget.v1",
            "selected_candidates": [
                {
                    "schema": "snerv_modelsize_candidate.v1",
                    "family": "snerv",
                    "candidate_id": "snerv_tiny",
                    "num_pairs": 600,
                    "hard_byte_ceiling": 178_000,
                    "decoder_payload_codec": "int4_symmetric",
                    "nominal_total_payload_bytes": 160_000,
                    "nominal_under_ceiling": True,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        optimizer_kinds=("adamw",),
        epochs=16,
        batch_pairs=4,
        learning_rate=3.0e-4,
        output_root=output_root,
        max_candidates_per_family=1,
    )


def _source_row_contract(source_row: dict, key: str) -> dict:
    direct = source_row.get(key)
    if isinstance(direct, dict):
        return direct
    launch = source_row.get("launch_authority_contract")
    if isinstance(launch, dict) and isinstance(launch.get(key), dict):
        return launch[key]
    metadata = source_row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get(key), dict):
        return metadata[key]
    raise KeyError(key)


def _runnable_verdict(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    queue = json.loads(json.dumps(_campaign_plan(output_root)["experiment_queue"]))
    hi = next(row for row in queue["experiments"] if row["family"] == "hi_nerv")
    hi["status"] = "queued"
    hi["blocked"] = False
    contract = hi["launch_authority_contract"]
    contract["queue_status_is_local_mlx_plan"] = True
    contract["queue_status_is_runnable_plan"] = True
    contract["queue_launch_blockers"] = []
    contract["queue_status_is_receiver_proof"] = False
    contract["queue_status_is_cpu_replay_proof"] = False
    contract["queue_status_is_exact_eval_authority"] = False
    gate = hi["score_lowering_gate"]
    gate["local_mlx_executable"] = True
    gate["prelaunch_allowed"] = True
    gate["cpu_replay_ready"] = False
    gate["exact_gate_ready"] = False
    return dict(consume_candidate(queue))


def _claims_file(tmp_path: Path, *, lane_id: str, instance_job_id: str) -> Path:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                (
                    f"| 2026-06-02T18:34:58Z | codex:gpt-5 | {lane_id} | "
                    f"local_mlx | {instance_job_id} | 2026-06-03T00:34:58Z | "
                    "active_local_mlx_queue_first_row | test claim |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return claims
