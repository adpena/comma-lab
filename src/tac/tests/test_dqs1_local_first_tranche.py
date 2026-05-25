# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_dqs1_local_first_tranche as tranche


def test_tranche_parse_args_accepts_post_training_candidate_inputs() -> None:
    args = tranche.parse_args(
        [
            "--rounds",
            "3",
            "--candidate-json",
            "experiments/results/promising_run/candidates.json",
            "--mlx-selection",
            "experiments/results/promising_run/mlx_selection.json",
            "--hfv2-manifest",
            "experiments/results/promising_run/hfv_manifest.json",
            "--family-beliefs",
            ".omx/research/family_beliefs.json",
        ]
    )

    assert args.rounds == 3
    assert args.candidate_json == ["experiments/results/promising_run/candidates.json"]
    assert args.mlx_selection == ["experiments/results/promising_run/mlx_selection.json"]
    assert args.hfv2_manifest == ["experiments/results/promising_run/hfv_manifest.json"]
    assert args.family_beliefs == ".omx/research/family_beliefs.json"
    assert args.queue_candidate_limit == 2
    assert args.storage_waterfall is True
    assert args.allow_local_storage_tier is False
    assert args.proactive_cleanup is True
    assert args.proactive_cleanup_action == "move"
    assert args.retention_action == "move"
    assert args.local_cpu_concurrency == 2
    assert args.local_io_concurrency == 2
    assert args.include_mlx_local_advisory_debug is False
    assert args.mlx_device == "gpu"
    assert args.refresh_pairset_acquisition is True
    assert args.refresh_pairset_max_drop_two == 512
    assert args.refresh_pairset_max_swap_in == 32
    assert args.refresh_pairset_geometry_lattice is True
    assert args.refresh_pairset_geometry_frame_pair_curriculum == "latest"
    assert args.refresh_pairset_geometry_pair_component_xray == []
    assert args.refresh_pairset_geometry_drop_counts == "3,4,6,8,12,16"
    assert args.refresh_pairset_geometry_max_requests == 32


def test_tranche_portfolio_directory_names_include_harvest_count() -> None:
    assert (
        tranche._portfolio_dir_name("20260523T130000Z", 17)
        == "20260523T130000Z_full_drop_two_local_harvest17"
    )


def test_tranche_portfolio_falls_back_when_pairset_model_inactive(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True) -> tranche.CommandResult:
        calls.append(command)
        if len(calls) == 1:
            return tranche.CommandResult(
                command=command,
                returncode=2,
                stdout="",
                stderr=tranche.INACTIVE_PAIRSET_OBSERVATION_MODEL_MARKER,
                elapsed_seconds=0.01,
            )
        return tranche.CommandResult(
            command=command,
            returncode=0,
            stdout='{"ok": true}',
            stderr="",
            elapsed_seconds=0.02,
        )

    monkeypatch.setattr(tranche, "_run", fake_run)

    result, used_fallback = tranche._run_portfolio_with_exploratory_fallback(
        ["planner", "--require-active-pairset-observation-model", "--json-out", "x.json"]
    )

    assert result.returncode == 0
    assert used_fallback is True
    assert len(calls) == 2
    assert "--require-active-pairset-observation-model" not in calls[1]


def test_tranche_portfolio_fallback_keeps_non_model_failures_strict(
    monkeypatch,
) -> None:
    def fake_run(command: list[str], *, check: bool = True) -> tranche.CommandResult:
        return tranche.CommandResult(
            command=command,
            returncode=2,
            stdout="not-json",
            stderr="different failure",
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(tranche, "_run", fake_run)

    with pytest.raises(RuntimeError, match="command failed"):
        tranche._run_portfolio_with_exploratory_fallback(
            ["planner", "--require-active-pairset-observation-model"]
        )


def test_tranche_parse_args_accepts_tiered_cold_store_roots() -> None:
    args = tranche.parse_args(
        [
            "--retention-action",
            "move",
            "--retention-cold-store-root",
            "/Volumes/VertigoDataTier/pact/cold_store",
            "--retention-cold-store-root",
            "/Volumes/APDataStore/pact/cold_store",
            "--proactive-cleanup-min-bytes",
            "2GiB",
        ]
    )

    assert args.retention_cold_store_root == [
        "/Volumes/VertigoDataTier/pact/cold_store",
        "/Volumes/APDataStore/pact/cold_store",
    ]
    assert args.proactive_cleanup_min_bytes == 2 * 1024**3


def test_tranche_defaults_to_storage_working_queue() -> None:
    args = tranche.parse_args([])

    assert args.storage_working_queue is True
    assert args.initial_action_summary == "latest"


def test_tranche_queue_build_args_preserve_storage_queue_and_completed_roots() -> None:
    args = tranche.parse_args(
        [
            "--storage-tier",
            "vertigo=/Volumes/VertigoDataTier/pact",
            "--storage-tier",
            "ap=/Volumes/APDataStore/pact",
            "--proactive-cleanup-root",
            "experiments/results",
            "--proactive-cleanup-min-bytes",
            "2GiB",
        ]
    )

    built = tranche._queue_build_common_args(
        args,
        results_root=tranche.Path("/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first"),
    )

    assert "--include-scheduler-preflight" in built
    assert built.count("--completed-results-root") == 3
    assert "/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first" in built
    assert "/Volumes/APDataStore/pact/experiments/results/dqs1_local_first" in built
    expected_root = built[built.index("--scheduler-storage-expected-workload-root") + 1]
    assert expected_root == "/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first"


def test_tranche_queue_build_args_forward_mlx_local_advisory_controls() -> None:
    args = tranche.parse_args(
        [
            "--no-storage-waterfall",
            "--include-mlx-local-advisory-debug",
            "--allow-large-mlx-cache",
            "--mlx-reference-cache-dir",
            "reference/full600",
            "--mlx-device",
            "cpu",
            "--mlx-batch-pairs",
            "1",
            "--mlx-cache-batch-pairs",
            "4",
            "--skip-mlx-retention-plan",
        ]
    )

    built = tranche._queue_build_common_args(
        args,
        results_root=tranche.Path("experiments/results/dqs1_local_first"),
    )

    assert "--include-mlx-local-advisory-debug" in built
    assert "--allow-large-mlx-cache" in built
    assert built[built.index("--mlx-reference-cache-dir") + 1] == "reference/full600"
    assert built[built.index("--mlx-device") + 1] == "cpu"
    assert built[built.index("--mlx-cache-batch-pairs") + 1] == "4"
    assert "--skip-mlx-retention-plan" in built


def test_tranche_queue_build_args_forward_materializer_feedback() -> None:
    args = tranche.parse_args(
        [
            "--no-storage-waterfall",
            "--materializer-feedback",
            "experiments/results/header_elide/sweep.json",
            "--materializer-feedback",
            "experiments/results/recompress/sweep.json",
            "--dqs1-observation-jsonl",
            ".omx/research/dqs1_local_first_harvest_observations_prior.jsonl",
            "--include-observed-dqs1-candidate",
        ]
    )

    built = tranche._queue_build_common_args(
        args,
        results_root=tranche.Path("experiments/results/dqs1_local_first"),
        dqs1_observation_jsonl=(
            tranche.Path(".omx/research/dqs1_local_first_harvest_observations_round.jsonl"),
        ),
    )

    assert built.count("--materializer-feedback") == 2
    first = built.index("--materializer-feedback")
    second = built.index("--materializer-feedback", first + 1)
    assert built[first + 1] == "experiments/results/header_elide/sweep.json"
    assert built[second + 1] == "experiments/results/recompress/sweep.json"
    assert built.count("--dqs1-observation-jsonl") == 2
    prior = built.index("--dqs1-observation-jsonl")
    current = built.index("--dqs1-observation-jsonl", prior + 1)
    assert built[prior + 1] == (
        ".omx/research/dqs1_local_first_harvest_observations_prior.jsonl"
    )
    assert built[current + 1] == (
        ".omx/research/dqs1_local_first_harvest_observations_round.jsonl"
    )
    assert "--include-observed-dqs1-candidate" in built


def test_tranche_refresh_pairset_acquisition_uses_prior_and_round_observations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selector_root = tmp_path / "selector_pareto"
    selector_root.mkdir(parents=True)
    selector = selector_root / "latest_selector_pareto.json"
    selector.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "refreshed_pairset"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True) -> tranche.CommandResult:
        calls.append(command)
        return tranche.CommandResult(
            command=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "candidate_count": 5,
                    "unfiltered_candidate_count": 8,
                    "suppressed_observed_candidate_count": 3,
                }
            ),
            stderr="",
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(tranche, "_run", fake_run)
    args = tranche.parse_args(
        [
            "--pairset-acquisition-refresh-root",
            str(output_root),
            "--dqs1-observation-jsonl",
            "prior.jsonl",
            "--include-observed-dqs1-candidate",
            "--refresh-pairset-max-drop-two",
            "7",
            "--refresh-pairset-max-swap-in",
            "3",
            "--no-refresh-pairset-geometry-lattice",
        ]
    )

    json_out, payload = tranche._refresh_pairset_acquisition(
        args=args,
        pairset_acquisition_root=tmp_path,
        refresh_stamp="20260525T010203Z",
        dqs1_observation_jsonl=(Path("round.jsonl"),),
    )

    assert json_out == output_root / "dqs1_pairset_acquisition_observation_filtered_20260525T010203Z.json"
    assert payload["candidate_count"] == 5
    assert payload["unfiltered_candidate_count"] == 8
    assert payload["suppressed_observed_candidate_count"] == 3
    command = calls[0]
    assert command[command.index("--selector-pareto") + 1] == str(selector)
    assert command[command.index("--max-drop-two") + 1] == "7"
    assert command[command.index("--max-swap-in") + 1] == "3"
    assert command.count("--dqs1-observation-jsonl") == 2
    assert "prior.jsonl" in command
    assert "round.jsonl" in command
    assert "--include-observed-dqs1-candidate" in command
    assert payload["pair_frame_geometry_lattice"]["active"] is False


def test_tranche_refresh_pairset_acquisition_binds_geometry_lattice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selector_root = tmp_path / "selector_pareto"
    selector_root.mkdir(parents=True)
    selector = selector_root / "latest_selector_pareto.json"
    selector.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "refreshed_pairset"
    curriculum = tmp_path / "ll_frame_pair_curriculum.json"
    xray = tmp_path / "pair_component_xray.json"
    curriculum.write_text("{}", encoding="utf-8")
    xray.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True) -> tranche.CommandResult:
        calls.append(command)
        if command[1] == "tools/build_pair_frame_scorer_geometry_lattice.py":
            stdout = {
                "queue_executable_request_count": 2,
                "geometry_coverage": 1.0,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        elif "--pair-frame-geometry-lattice-json" in command:
            stdout = {
                "candidate_count": 7,
                "unfiltered_candidate_count": 10,
                "suppressed_observed_candidate_count": 0,
                "pair_frame_geometry_candidate_count": 2,
            }
        else:
            stdout = {
                "candidate_count": 5,
                "unfiltered_candidate_count": 8,
                "suppressed_observed_candidate_count": 0,
                "pair_frame_geometry_candidate_count": 0,
            }
        return tranche.CommandResult(
            command=command,
            returncode=0,
            stdout=json.dumps(stdout),
            stderr="",
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(tranche, "_run", fake_run)
    args = tranche.parse_args(
        [
            "--pairset-acquisition-refresh-root",
            str(output_root),
            "--refresh-pairset-geometry-frame-pair-curriculum",
            str(curriculum),
            "--refresh-pairset-geometry-pair-component-xray",
            str(xray),
            "--refresh-pairset-geometry-drop-counts",
            "3,4",
            "--refresh-pairset-geometry-max-requests",
            "9",
        ]
    )

    json_out, payload = tranche._refresh_pairset_acquisition(
        args=args,
        pairset_acquisition_root=tmp_path,
        refresh_stamp="20260525T020304Z",
        dqs1_observation_jsonl=(),
    )

    assert json_out == output_root / "dqs1_pairset_acquisition_observation_filtered_20260525T020304Z.json"
    assert len(calls) == 3
    base_command, geometry_command, final_command = calls
    assert base_command[base_command.index("--json-out") + 1].endswith(".base.json")
    assert geometry_command[1] == "tools/build_pair_frame_scorer_geometry_lattice.py"
    assert geometry_command[geometry_command.index("--pairset-acquisition") + 1].endswith(
        ".base.json"
    )
    assert geometry_command[geometry_command.index("--frame-pair-curriculum") + 1] == str(
        curriculum
    )
    assert geometry_command[geometry_command.index("--pair-component-xray") + 1] == str(
        xray
    )
    assert geometry_command[geometry_command.index("--drop-counts") + 1] == "3,4"
    assert geometry_command[geometry_command.index("--max-requests") + 1] == "9"
    assert "--pair-frame-geometry-lattice-json" in final_command
    lattice_arg = final_command[final_command.index("--pair-frame-geometry-lattice-json") + 1]
    assert lattice_arg.endswith("pair_frame_scorer_geometry_lattice_20260525T020304Z.json")
    assert payload["stdout"]["pair_frame_geometry_candidate_count"] == 2
    geometry = payload["pair_frame_geometry_lattice"]
    assert geometry["active"] is True
    assert geometry["queue_executable_request_count"] == 2
    assert geometry["geometry_coverage"] == 1.0
    assert geometry["frame_pair_curriculum"] == str(curriculum)
    assert geometry["pair_component_xrays"] == [str(xray)]


def test_tranche_proactive_cleanup_move_is_not_gated_by_mlx_cache_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    cold = tmp_path / "cold"
    captured: dict[str, list[str]] = {}

    def fake_run(command: list[str]) -> object:
        captured["command"] = command
        output_path = tranche.Path(command[command.index("--json-output") + 1])
        if not output_path.is_absolute():
            output_path = tmp_path / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "plan": {
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                }
            ),
            encoding="utf-8",
        )

        class Result:
            elapsed_seconds = 0.01

        return Result()

    monkeypatch.setattr(tranche, "_run", fake_run)
    monkeypatch.setattr(tranche, "REPO_ROOT", tmp_path)
    args = tranche.parse_args(
        [
            "--proactive-cleanup-root",
            str(root),
            "--retention-cold-store-root",
            str(cold),
            "--proactive-cleanup-min-bytes",
            "1",
        ]
    )

    payload = tranche._run_proactive_cleanup(args=args, stamp="20260523T000000Z", round_index=0)

    command = captured["command"]
    assert payload is not None
    assert "--cold-store-root" in command
    assert str(cold) in command
    assert "--cold-store-reserve-gb" in command
    assert "--execute" not in command
    output_path = tmp_path / ".omx/research/dqs1_proactive_artifact_retention_20260523T000000Z_round000.json"
    disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert disk_payload["schema"] == "dqs1_local_first_proactive_cleanup.v1"
    assert disk_payload["round_index"] == 0
    assert disk_payload["command"] == command
    assert disk_payload["ready_for_exact_eval_dispatch"] is False
