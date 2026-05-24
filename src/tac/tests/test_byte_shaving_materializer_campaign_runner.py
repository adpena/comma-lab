# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
)
from tools import run_byte_shaving_materializer_campaign as runner


def test_materializer_campaign_runner_builds_queue_owned_followup_command(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--queue-id",
            "materializer_campaign_fixture",
            "--materializer-resource-concurrency",
            "local_mlx=2",
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(tmp_path / "campaign" / "work"),
            "--storage-workload-subdir",
            "work",
            "--proactive-cleanup-root",
            "experiments/results",
            "--proactive-cleanup-cold-store-root",
            str(tmp_path / "cold_store"),
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert command[:2] == [
        runner.sys.executable,
        "tools/build_byte_shaving_campaign_queue.py",
    ]
    assert "--include-materializer-exact-readiness-followup" in command
    assert "--materializer-execution-queue-out" in command
    assert "--materializer-resource-concurrency" in command
    assert "local_mlx=2" in command
    assert "--include-materializer-scheduler-preflight" in command
    assert "--materializer-scheduler-proactive-cleanup-execute" in command
    assert "--materializer-scheduler-proactive-cleanup-cold-store-root" in command
    assert str(tmp_path / "cold_store") in command
    assert "--dispatch-mode" not in command
    assert "--allow-paid-dispatch-queue" not in command


def test_materializer_campaign_runner_can_auto_generate_contexts_from_artifact_map(
    tmp_path: Path,
) -> None:
    artifact_map = tmp_path / "artifact_map.json"
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-artifact-map",
            str(artifact_map),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--materializer-contexts-fail-if-blocked",
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert "--materializer-artifact-map" in command
    assert str(artifact_map) in command
    assert "--materializer-contexts-out" in command
    assert str(tmp_path / "campaign" / "materializer_contexts.json") in command
    assert "--materializer-context-default-output-root" in command
    assert str(tmp_path / "campaign" / "materializer_outputs") in command
    assert "--materializer-contexts-fail-if-blocked" in command
    assert "--materializer-contexts" not in command


def test_materializer_campaign_runner_can_generate_inverse_scorer_artifact_map(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    action = tmp_path / "inverse_action.json"
    template = tmp_path / "template.zip"
    source_inflate = tmp_path / "source_inflate"
    candidate_inflate = tmp_path / "candidate_inflate"
    action.write_text("{}", encoding="utf-8")
    template.write_bytes(b"zip fixture")
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--inverse-scorer-action-functional",
            str(action),
            "--inverse-scorer-candidate-archive-template",
            str(template),
            "--inverse-scorer-raw-contest-video-digest",
            "f" * 64,
            "--inverse-scorer-atom-id",
            "inverse_surface_pair0007",
            "--inverse-scorer-selected-limit",
            "2",
            "--inverse-scorer-chain-output-dir",
            str(run_dir / "inverse_cell_chain"),
            "--inverse-scorer-source-inflate-output-dir",
            str(source_inflate),
            "--inverse-scorer-candidate-inflate-output-dir",
            str(candidate_inflate),
            "--inverse-scorer-fail-if-inflate-parity-blocked",
            "--run-dir",
            str(run_dir),
        ]
    )

    generated = runner._write_generated_materializer_artifact_map(
        args,
        run_dir=run_dir,
        generated_action_functional_path=None,
    )

    assert generated == run_dir / "materializer_artifact_map.json"
    payload = json.loads(generated.read_text(encoding="utf-8"))
    context = payload["artifacts"][runner.INVERSE_SCORER_CELL_TARGET_KIND]
    assert context["candidate_archive_template"] == str(template)
    assert context["inverse_action_functional"] == str(action)
    assert context["raw_contest_video_digest"] == "f" * 64
    assert context["atom_ids"] == ["inverse_surface_pair0007"]
    assert context["selected_limit"] == 2
    assert context["chain_output_dir"] == str(run_dir / "inverse_cell_chain")
    assert context["source_inflate_output_dir"] == str(source_inflate)
    assert context["candidate_inflate_output_dir"] == str(candidate_inflate)
    assert context["fail_if_inflate_parity_blocked"] is True
    assert context["score_claim"] is False

    command = runner._build_queue_command(
        args,
        run_dir=run_dir,
        plan_path=tmp_path / "plan.json",
        generated_materializer_artifact_map=generated,
    )

    assert "--materializer-artifact-map" in command
    assert str(generated) in command
    assert "--materializer-contexts-out" in command
    assert str(run_dir / "materializer_contexts.json") in command


def test_materializer_campaign_runner_generated_artifact_map_uses_generated_action(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    template = tmp_path / "template.zip"
    generated_action = run_dir / "inverse_steganalysis_action_functional.json"
    template.write_bytes(b"zip fixture")
    generated_action.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--scorer-response",
            str(tmp_path / "scorer_response.json"),
            "--inverse-scorer-candidate-archive-template",
            str(template),
            "--inverse-scorer-raw-contest-video-digest",
            "a" * 64,
            "--run-dir",
            str(run_dir),
        ]
    )

    generated = runner._write_generated_materializer_artifact_map(
        args,
        run_dir=run_dir,
        generated_action_functional_path=generated_action,
    )

    payload = json.loads(generated.read_text(encoding="utf-8"))
    context = payload["artifacts"][runner.INVERSE_SCORER_CELL_TARGET_KIND]
    assert context["inverse_action_functional"] == str(generated_action)
    assert context["candidate_archive_template"] == str(template)
    assert context["raw_contest_video_digest"] == "a" * 64


def test_materializer_campaign_runner_rejects_auto_artifact_map_with_contexts(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="auto artifact-map flags"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--materializer-contexts",
                str(tmp_path / "contexts.json"),
                "--inverse-scorer-candidate-archive-template",
                str(tmp_path / "template.zip"),
                "--inverse-scorer-raw-contest-video-digest",
                "f" * 64,
            ]
        )


def test_materializer_campaign_runner_uses_policy_cold_store_default_for_move_preflight(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(tmp_path / "campaign" / "work"),
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert "--include-materializer-scheduler-preflight" in command
    assert "--materializer-scheduler-proactive-cleanup-execute" in command
    assert "--materializer-scheduler-proactive-cleanup-cold-store-root" not in command


def test_materializer_campaign_runner_emits_staircase_artifacts(tmp_path: Path) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "campaign_staircase_fixture",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
            "experiments": [
                {
                    "id": "candidate",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "materialize",
                            "command": ["python", "-c", "print('ok')"],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {"type": "path_exists", "path": str(tmp_path / "done.json")}
                            ],
                        }
                    ],
                }
            ],
        }
    )
    queue_path = tmp_path / "queue.json"
    state_path = tmp_path / "queue.sqlite"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--emit-staircase-plan",
            "--staircase-resource-pool",
            "sshbox:local_cpu=1,memory_gb=8,disk_gb=8,"
            "executor=ssh_experiment_queue,ssh_target=user@sshbox,"
            "remote_repo_root=/remote/pact",
        ]
    )
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()

    result = runner._build_staircase_artifacts(
        args,
        run_dir=run_dir,
        execution_queue=queue_path,
        state_path=state_path,
        queue=queue,
    )

    assert result["selected_count"] == 1
    plan = json.loads((run_dir / "staircase_dispatch_plan.json").read_text(encoding="utf-8"))
    task = plan["dask_task_specs"][0]
    assert task["machine"]["executor"] == "ssh_experiment_queue"
    assert task["machine"]["remote_repo_root"] == "/remote/pact"
    assert task["queue_state_writeback"]["required"] is True
    assert plan["score_claim"] is False


def test_materializer_campaign_runner_builds_ssh_dry_run_command(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--staircase-ssh-dry-run",
            "--staircase-ssh-machine-id",
            "sshbox",
            "--staircase-ssh-remote-repo-root",
            "sshbox=/remote/pact",
        ]
    )

    command = runner._ssh_executor_dry_run_command(
        args,
        execution_queue=tmp_path / "queue.json",
        state_path=tmp_path / "queue.sqlite",
        staircase_plan_path=tmp_path / "plan.staircase.json",
        run_dir=tmp_path / "campaign",
    )

    assert command[:2] == [
        runner.sys.executable,
        "tools/run_staircase_ssh_executor.py",
    ]
    assert "--execute" not in command
    assert "--machine-id" in command
    assert "sshbox" in command
    assert "--remote-repo-root" in command
    assert "sshbox=/remote/pact" in command


def test_materializer_campaign_runner_builds_ssh_execute_command_with_artifact_pullback(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--staircase-ssh-execute",
            "--staircase-ssh-max-steps",
            "3",
            "--staircase-ssh-machine-id",
            "sshbox",
            "--staircase-ssh-remote-repo-root",
            "sshbox=/remote/pact",
            "--staircase-ssh-artifact-path-map",
            f"{tmp_path / 'campaign'}=/remote/campaign",
            "--staircase-ssh-rsync-binary",
            "rsync-fixture",
            "--staircase-ssh-artifact-pull-timeout-seconds",
            "42",
        ]
    )

    command = runner._ssh_executor_command(
        args,
        execution_queue=tmp_path / "queue.json",
        state_path=tmp_path / "queue.sqlite",
        staircase_plan_path=tmp_path / "plan.staircase.json",
        run_dir=tmp_path / "campaign",
        execute=True,
    )

    assert "--execute" in command
    assert "--max-steps" in command
    assert "3" in command
    assert "--require-artifact-mobility" in command
    assert "--artifact-path-map" in command
    assert f"{tmp_path / 'campaign'}=/remote/campaign" in command
    assert "--rsync-binary" in command
    assert "rsync-fixture" in command
    assert "--artifact-pull-timeout-seconds" in command
    assert "42" in command
    assert any("staircase_ssh_executor_execute.json" in part for part in command)


def test_materializer_campaign_runner_generates_plan_from_high_level_sources(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer_response.json"
    mlx_batch = tmp_path / "mlx_acquisition_batch.json"
    action_path = tmp_path / "campaign" / "inverse_steganalysis_action_functional.json"
    plan_path = tmp_path / "campaign" / "byte_shaving_campaign_plan.json"
    scorer.write_text("{}", encoding="utf-8")
    mlx_batch.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--scorer-response",
            str(scorer),
            "--mlx-acquisition-batch",
            str(mlx_batch),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "high_level_fixture",
            "--candidate-id",
            "candidate_a",
            "--total-byte-budget",
            "64",
            "--campaign-plan-max-k",
            "3",
            "--queue-id",
            "high_level_materializer_queue",
        ]
    )

    action_command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )
    plan_command = runner._build_campaign_plan_command(
        args,
        action_functional_path=action_path,
        run_dir=tmp_path / "campaign",
    )
    queue_command = runner._build_queue_command(
        args,
        run_dir=tmp_path / "campaign",
        plan_path=plan_path,
    )

    assert action_command[:2] == [
        runner.sys.executable,
        "tools/build_inverse_steganalysis_action_functional.py",
    ]
    assert "--scorer-response" in action_command
    assert str(scorer) in action_command
    assert "--mlx-acquisition-batch" in action_command
    assert str(mlx_batch) in action_command
    assert "--total-byte-budget" in action_command
    assert "64" in action_command
    assert "--candidate-id" in action_command
    assert "candidate_a" in action_command
    assert plan_command[:2] == [
        runner.sys.executable,
        "tools/plan_byte_shaving_campaign.py",
    ]
    assert "--from-inverse-action-functional" in plan_command
    assert "--campaign-id" in plan_command
    assert "high_level_fixture" in plan_command
    assert "--max-k" in plan_command
    assert "3" in plan_command
    assert "--plan" in queue_command
    assert str(plan_path) in queue_command


def test_materializer_campaign_runner_treats_mlx_acquisition_batch_as_action_source(
    tmp_path: Path,
) -> None:
    mlx_batch = tmp_path / "mlx_acquisition_batch.json"
    mlx_batch.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--mlx-acquisition-batch",
            str(mlx_batch),
            "--run-dir",
            str(tmp_path / "campaign"),
        ]
    )

    assert runner._action_source_count(args) == 1
    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )

    assert "--mlx-acquisition-batch" in command
    assert str(mlx_batch) in command


def test_materializer_campaign_runner_forwards_family_byte_shaving_sources(
    tmp_path: Path,
) -> None:
    surface = tmp_path / "family_surface.json"
    campaign_plan = tmp_path / "family_campaign_plan.json"
    surface.write_text("{}", encoding="utf-8")
    campaign_plan.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--byte-shaving-signal-surface",
            str(surface),
            "--byte-shaving-campaign-plan",
            str(campaign_plan),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "family_mix_runner",
        ]
    )

    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )

    assert runner._action_source_count(args) == 2
    assert "--byte-shaving-signal-surface" in command
    assert str(surface) in command
    assert "--byte-shaving-campaign-plan" in command
    assert str(campaign_plan) in command
    assert "--candidate-id" not in command


def test_materializer_campaign_runner_builds_mlx_batch_from_selection_inline(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "mlx_selection.json"
    selection.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "campaign"
    args = runner.parse_args(
        [
            "--mlx-effective-spend-triage-selection",
            str(selection),
            "--run-dir",
            str(run_dir),
            "--mlx-acquisition-set-size",
            "4",
            "--mlx-acquisition-limit",
            "8",
            "--overwrite-output",
        ]
    )

    batch_commands = runner._build_mlx_acquisition_batch_commands(
        args,
        run_dir=run_dir,
    )
    assert len(batch_commands) == 1
    batch_path, batch_command = batch_commands[0]
    assert batch_path == run_dir / "mlx_acquisition_batch_0000.json"
    assert batch_command[:2] == [
        runner.sys.executable,
        "tools/build_mlx_acquisition_batch.py",
    ]
    assert "--mlx-effective-spend-triage-selection" in batch_command
    assert str(selection) in batch_command
    assert "--set-size" in batch_command
    assert "4" in batch_command
    assert "--limit" in batch_command
    assert "8" in batch_command
    assert "--allow-overwrite" in batch_command

    action_command = runner._build_action_functional_command(
        args,
        run_dir=run_dir,
        generated_mlx_acquisition_batches=[batch_path],
    )
    assert "--mlx-acquisition-batch" in action_command
    assert str(batch_path) in action_command
    assert "--mlx-effective-spend-triage-selection" not in action_command


def test_materializer_campaign_runner_can_preserve_direct_mlx_selection_mode(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "mlx_selection.json"
    selection.write_text("{}", encoding="utf-8")
    args = runner.parse_args(
        [
            "--mlx-effective-spend-triage-selection",
            str(selection),
            "--mlx-effective-spend-triage-selection-mode",
            "direct",
            "--run-dir",
            str(tmp_path / "campaign"),
        ]
    )

    assert runner._build_mlx_acquisition_batch_commands(
        args,
        run_dir=tmp_path / "campaign",
    ) == []
    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )
    assert "--mlx-effective-spend-triage-selection" in command
    assert str(selection) in command


def test_materializer_campaign_runner_loads_file_driven_run_config(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer_response.json"
    selection = tmp_path / "mlx_selection.json"
    config_path = tmp_path / "rate_attack_config.json"
    scorer.write_text("{}", encoding="utf-8")
    selection.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "schema": runner.RUN_CONFIG_SCHEMA,
                "args": {
                    "scorer_response": [str(scorer)],
                    "mlx_effective_spend_triage_selection": [str(selection)],
                    "run_dir": str(tmp_path / "campaign"),
                    "campaign_id": "configured_final_rate_attack",
                    "candidate_id": "candidate_from_config",
                    "total_byte_budget": 96,
                    "campaign_plan_max_k": 4,
                    "queue_id": "configured_materializer_queue",
                    "mlx_acquisition_set_size": 3,
                    "materializer_resource_concurrency": ["local_cpu=2"],
                    "emit_staircase_plan": True,
                },
            }
        ),
        encoding="utf-8",
    )

    args = runner.parse_args(["--run-config", str(config_path)])

    assert args.run_config == config_path
    assert args.scorer_response == [str(scorer)]
    assert args.mlx_effective_spend_triage_selection == [str(selection)]
    assert args.run_dir == tmp_path / "campaign"
    assert args.campaign_id == "configured_final_rate_attack"
    assert args.candidate_id == "candidate_from_config"
    assert args.total_byte_budget == 96
    assert args.campaign_plan_max_k == 4
    assert args.queue_id == "configured_materializer_queue"
    assert args.mlx_acquisition_set_size == 3
    assert args.materializer_resource_concurrency == ["local_cpu=2"]
    assert args.emit_staircase_plan is True

    command = runner._build_action_functional_command(
        args,
        run_dir=tmp_path / "campaign",
    )
    assert "--scorer-response" in command
    assert str(scorer) in command
    assert "--candidate-id" in command
    assert "candidate_from_config" in command


def test_materializer_campaign_runner_cli_scalar_overrides_run_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "rate_attack_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": runner.RUN_CONFIG_SCHEMA,
                "args": {
                    "atom": [str(tmp_path / "atom.json")],
                    "run_dir": str(tmp_path / "campaign_from_config"),
                    "campaign_id": "configured_campaign",
                    "candidate_id": "configured_candidate",
                    "total_byte_budget": 96,
                },
            }
        ),
        encoding="utf-8",
    )

    args = runner.parse_args(
        [
            "--run-config",
            str(config_path),
            "--candidate-id",
            "cli_candidate",
            "--total-byte-budget",
            "32",
        ]
    )

    assert args.atom == [str(tmp_path / "atom.json")]
    assert args.campaign_id == "configured_campaign"
    assert args.candidate_id == "cli_candidate"
    assert args.total_byte_budget == 32


def test_materializer_campaign_runner_rejects_unknown_run_config_key(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "rate_attack_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": runner.RUN_CONFIG_SCHEMA,
                "args": {"definitely_not_a_runner_field": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unknown run config key"):
        runner.parse_args(["--run-config", str(config_path)])


def test_materializer_campaign_runner_rejects_missing_plan_and_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="provide --plan or high-level action sources"):
        runner.main(["--run-dir", str(tmp_path / "campaign")])


def test_materializer_campaign_runner_rejects_mixed_plan_and_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--scorer-response",
                str(tmp_path / "scorer_response.json"),
                "--run-dir",
                str(tmp_path / "campaign"),
            ]
        )


def test_materializer_campaign_runner_rejects_mixed_local_and_ssh_execute(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="cannot target the same queue run"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--run-dir",
                str(tmp_path / "campaign"),
                "--execute",
                "--staircase-ssh-execute",
                "--staircase-ssh-artifact-path-map",
                f"{tmp_path / 'campaign'}=/remote/campaign",
            ]
        )


def test_materializer_campaign_runner_rejects_ssh_execute_without_artifact_mobility(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="requires --staircase-ssh-artifact-path-map"):
        runner.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--run-dir",
                str(tmp_path / "campaign"),
                "--staircase-ssh-execute",
            ]
        )


def test_materializer_campaign_runner_rejects_bad_resource_concurrency() -> None:
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=0"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=two"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu"])


def test_materializer_campaign_runner_rejects_bad_ssh_remote_root_mapping() -> None:
    with pytest.raises(SystemExit):
        runner._parse_remote_repo_roots(["sshbox"])


def test_materializer_campaign_runner_rejects_bad_artifact_path_mapping() -> None:
    with pytest.raises(SystemExit):
        runner._parse_artifact_path_maps(["/local-only"])


def test_materializer_campaign_runner_requires_json_from_ssh_dry_run() -> None:
    with pytest.raises(SystemExit, match="did not emit a JSON object"):
        runner._require_json_stdout(
            runner.CommandResult(
                command=["ssh-dry-run"],
                returncode=0,
                stdout="not-json",
                stderr="",
                elapsed_seconds=0.0,
            ),
            label="staircase SSH executor dry-run",
        )
    with pytest.raises(SystemExit, match="failed"):
        runner._require_json_stdout(
            runner.CommandResult(
                command=["ssh-dry-run"],
                returncode=2,
                stdout="",
                stderr="bad plan",
                elapsed_seconds=0.0,
            ),
            label="staircase SSH executor dry-run",
        )
