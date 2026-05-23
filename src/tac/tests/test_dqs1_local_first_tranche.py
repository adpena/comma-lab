# SPDX-License-Identifier: MIT
from __future__ import annotations

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
    assert args.queue_candidate_limit == 1
    assert args.storage_waterfall is True
    assert args.allow_local_storage_tier is False
    assert args.proactive_cleanup is True
    assert args.proactive_cleanup_action == "move"


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
