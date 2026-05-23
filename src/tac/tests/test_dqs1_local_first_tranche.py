# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

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


def test_tranche_portfolio_directory_names_include_harvest_count() -> None:
    assert (
        tranche._portfolio_dir_name("20260523T130000Z", 17)
        == "20260523T130000Z_full_drop_two_local_harvest17"
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
