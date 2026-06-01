# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tac.substrates.hprc.archive_candidate as hprc_candidate  # noqa: E402
from tac.optimization.archive_bound_candidate_runtime_bridge import (  # noqa: E402
    build_archive_bound_candidate_runtime_package,
)
from tac.substrates.hprc.batch_profile_compare import (  # noqa: E402
    HPRC_MLX_BATCH_PROFILE_COMPARISON_SCHEMA,
    compare_hprc_mlx_batch_profiles,
)
from tac.substrates.hprc.campaign import (  # noqa: E402
    HPRC_CAMPAIGN_MANIFEST_SCHEMA,
    HPRC_EXACT_READINESS_REFUSAL_SCHEMA,
    materialize_minimal_hprc_campaign,
)
from tac.substrates.hprc.exact_gate import (  # noqa: E402
    HPRC_INCREMENTAL_EXACT_GATE_BRIDGE_SCHEMA,
    build_hprc_incremental_exact_gate_bridge,
)
from tac.substrates.hprc.incremental_pair_response import (  # noqa: E402
    HPRC_INCREMENTAL_PAIR_RESPONSE_SCHEMA,
    build_hprc_incremental_pair_response_report,
)
from tac.substrates.hprc.incremental_runner_execution import (  # noqa: E402
    HPRC_INCREMENTAL_RUNNER_EXECUTION_COMPARISON_SCHEMA,
    HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA,
    HPRC_INCREMENTAL_RUNNER_EXECUTION_SCHEMA,
    build_hprc_incremental_runner_execution_report,
    compare_hprc_incremental_runner_execution_reports,
    prepare_hprc_incremental_runner_execution,
)
from tac.substrates.hprc.learned_receiver import (  # noqa: E402
    build_compact_receiver_packet_from_lowres_frames,
)
from tac.substrates.hprc.pair_scoped_residual_harvest import (  # noqa: E402
    HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_HARVEST_SCHEMA,
    build_pair_scoped_residual_runner_harvest,
)
from tac.substrates.hprc.pair_scoped_residual_runner import (  # noqa: E402
    HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_PLAN_SCHEMA,
    build_pair_scoped_residual_bounded_runner_plan,
)
from tools import build_hprc_pair_scoped_residual_bounded_runner as pair_runner_tool  # noqa: E402
from tools import compare_hprc_mlx_batch_profiles as batch_compare_tool  # noqa: E402
from tools import harvest_hprc_pair_scoped_residual_runner as harvest_runner_tool  # noqa: E402
from tools import package_hprc_minimal_candidate as hprc_tool  # noqa: E402


def _fake_emit_runtime_package(**kwargs):
    proof = {
        "schema": kwargs["proof_schema"],
        "proof_path": "receiver_proof/hprc_receiver_proof.json",
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "inflate_argv": ["inflate.sh", "archive_dir", "out", "file_list"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return build_archive_bound_candidate_runtime_package(
        adapter_id=kwargs["adapter_id"],
        candidate_family=kwargs["candidate_family"],
        candidate_id_prefix=kwargs["candidate_id_prefix"],
        transform_kind=kwargs["transform_kind"],
        archive_zip_path=kwargs["archive_zip_path"],
        archive_sha256=kwargs["archive_sha256"],
        archive_bytes=kwargs["archive_bytes"],
        submission_dir=kwargs["submission_dir"],
        output_dir=kwargs["output_dir"],
        repo_root=kwargs["repo_root"],
        receiver_proof=proof,
        receiver_contract_kind=kwargs["receiver_contract_kind"],
        runtime_adapter_manifest_extra=kwargs["runtime_adapter_manifest_extra"],
        candidate_row_schema=kwargs["candidate_row_schema"],
        wrapper_schema=kwargs["wrapper_schema"],
        input_artifacts=kwargs["input_artifacts"],
        extra_blockers=kwargs["extra_blockers"],
        mlx_triage_argv=kwargs["mlx_triage_argv"],
    )


def test_hprc_campaign_emits_refusal_and_resolution_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        _fake_emit_runtime_package,
    )
    storage_plan_path = tmp_path / "hprc_storage_plan.json"
    storage_plan_path.write_text(json.dumps({"schema": "test_storage_plan.v1"}))

    result = materialize_minimal_hprc_campaign(
        repo_root=repo,
        output_dir=tmp_path / "explicit",
        run_id="unit_hprc_campaign",
        storage_plan_path=storage_plan_path,
    )

    output_dir = Path(result.output_dir)
    assert output_dir == tmp_path / "explicit"
    assert Path(result.archive_zip_path).is_file()
    assert result.score_claim is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.storage_plan_path == storage_plan_path.as_posix()

    refusal = json.loads(Path(result.exact_readiness_refusal_path).read_text())
    assert refusal["schema"] == HPRC_EXACT_READINESS_REFUSAL_SCHEMA
    assert refusal["ready"] is False
    assert "trained_receiver_export_missing" in refusal["blockers"]
    assert "contest_resolution_contract_not_proven_by_full_frame_inflate" in refusal["blockers"]
    assert refusal["promotion_eligible"] is False

    manifest = json.loads(Path(result.campaign_manifest_path).read_text())
    assert manifest["schema"] == HPRC_CAMPAIGN_MANIFEST_SCHEMA
    assert manifest["storage_plan_path"] == storage_plan_path.as_posix()
    assert manifest["phase_status"]["receiver_scaffold"] == "runnable_non_promotable"
    assert manifest["phase_status"]["trained_receiver"] == "missing"
    assert manifest["phase_status"]["resolution_contract"] == "declared_not_proven"
    assert manifest["resolution_contract"]["contest_output"]["width"] == 1164
    assert manifest["resolution_contract"]["contest_output"]["height"] == 874
    assert manifest["resolution_contract"]["contest_output"]["pair_count"] == 600
    assert manifest["resolution_contract"]["scorer_preprocess"]["width"] == 512
    assert manifest["resolution_contract"]["scorer_preprocess"]["height"] == 384
    assert manifest["resolution_contract"]["posenet"]["frames_per_sample"] == 2
    assert manifest["queue_next_actions"][0]["id"] == "hprc_v1_train_export_archive"
    assert (
        manifest["queue_next_actions"][0]["status"]
        == "ready_via_hprc_compact_receiver_long_training_adapter"
    )
    assert manifest["campaign_taxonomy"]["score_claim"] is False


def test_hprc_campaign_explicit_output_dir_skips_storage_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        _fake_emit_runtime_package,
    )

    result = materialize_minimal_hprc_campaign(
        repo_root=tmp_path,
        output_dir=tmp_path / "explicit",
        run_id="explicit",
    )

    assert result.storage_plan_path is None
    assert Path(result.archive_bound_package_path).is_file()
    manifest = json.loads(Path(result.campaign_manifest_path).read_text())
    assert manifest["storage_plan_path"] is None


def test_hprc_packaging_cli_uses_storage_waterfall(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    fast = tmp_path / "fast"
    repo.mkdir()
    fast.mkdir()
    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        _fake_emit_runtime_package,
    )

    exit_code = hprc_tool.main(
        [
            "--repo-root",
            repo.as_posix(),
            "--run-id",
            "unit_hprc_cli",
            "--storage-tier",
            f"fast={fast}",
            "--storage-reserve-free-gb",
            "0",
            "--storage-expected-bytes",
            "0",
            "--allow-local-output-dir",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = fast / hprc_tool.DEFAULT_HPRC_WORKLOAD_SUBDIR / "unit_hprc_cli"
    assert Path(payload["output_dir"]) == output_dir
    storage_plan = json.loads((output_dir / "hprc_storage_plan.json").read_text())
    assert storage_plan["schema"] == hprc_tool.HPRC_STORAGE_PLAN_SCHEMA
    assert storage_plan["storage_plan"]["selected_tier"] == "fast"
    assert storage_plan["score_claim"] is False
    manifest = json.loads(Path(payload["campaign_manifest_path"]).read_text())
    assert manifest["storage_plan_path"] == (output_dir / "hprc_storage_plan.json").as_posix()


def test_hprc_pair_scoped_residual_runner_plan_emits_executable_rows(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate_dir = repo / "candidate"
    candidate_dir.mkdir()
    pair_plan = repo / "pair_plan.json"
    pair_plan.write_text(
        json.dumps(
            {
                "schema": "hprc_scorer_ranked_residual_shrink_backlog.v1",
                "pair_scoped_residual_candidate_rows": [
                    {
                        "source_variant_id": "residual_transform_threshold_abs_le_3",
                        "residual_transform": "threshold_abs_le_pairs=3@0,2-4",
                        "threshold_abs_le": 3,
                        "selected_pair_count": 4,
                        "protected_pair_count": 596,
                        "estimated_archive_bytes_removed_vs_baseline": 4000,
                        "estimated_delta_nonrate_pair_local_sum": -1.25,
                        "estimated_delta_rate_score": -0.01,
                        "pair_ranges": [[0, 0], [2, 4]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_profile = repo / "baseline_profile.json"
    baseline_profile.write_text("{}\n", encoding="utf-8")

    plan = build_pair_scoped_residual_bounded_runner_plan(
        pair_plan_path=pair_plan,
        reuse_baseline_profile_path=baseline_profile,
        candidate_dir=candidate_dir,
        output_dir=repo / "runner",
        repo_root=repo,
        max_candidates=1,
        max_pairs=600,
    )

    assert plan["schema"] == HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_PLAN_SCHEMA
    assert plan["baseline_reuse_required"] is True
    assert plan["score_claim"] is False
    row = plan["runner_rows"][0]
    assert row["residual_transform"] == "threshold_abs_le_pairs=3@0,2-4"
    assert row["candidate_id"].startswith("hprc-threshold-abs-le-pairs-")
    assert "--reuse-baseline-profile" in row["profile_command_argv"]
    assert "--residual-transforms" in row["profile_command_argv"]
    assert "--scorer-batch-pairs" in row["profile_command_argv"]
    assert "--pair-ranges" in row["incremental_response_command_argv"]
    assert "0,2-4" in row["incremental_response_command_argv"]
    assert plan["runner_policy"]["primary_execution"].startswith(
        "incremental_pair_response_first"
    )
    assert row["incremental_first_execution"]["tool"].endswith(
        "execute_hprc_pair_scoped_incremental_runner.py"
    )
    assert row["incremental_first_execution"]["full_candidate_profile_required"] is False
    assert row["incremental_first_execution"]["materializes_archive_zip"] is True
    assert row["expected_incremental_response_report"].endswith(
        "hprc_incremental_pair_response_report.json"
    )
    assert row["scorer_batch_pairs"] == 1
    assert row["receiver_proof_followup"]["required"] is True


def test_hprc_pair_scoped_residual_runner_cli_writes_plan(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate_dir = repo / "candidate"
    candidate_dir.mkdir()
    pair_plan = repo / "pair_plan.json"
    pair_plan.write_text(
        json.dumps(
            {
                "pair_scoped_residual_candidate_rows": [
                    {
                        "residual_transform": "threshold_abs_le_pairs=2@1",
                        "estimated_archive_bytes_removed_vs_baseline": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_profile = repo / "baseline_profile.json"
    baseline_profile.write_text("{}\n", encoding="utf-8")

    exit_code = pair_runner_tool.main(
        [
            "--repo-root",
            repo.as_posix(),
            "--pair-plan",
            pair_plan.as_posix(),
            "--reuse-baseline-profile",
            baseline_profile.as_posix(),
            "--candidate-dir",
            candidate_dir.as_posix(),
            "--output-dir",
            (repo / "runner").as_posix(),
        ]
    )

    assert exit_code == 0
    plan = json.loads(
        (repo / "runner" / "hprc_pair_scoped_residual_bounded_runner_plan.json").read_text()
    )
    assert len(plan["runner_rows"]) == 1
    assert plan["runner_rows"][0]["baseline_reuse_required"] is True


def test_hprc_pair_scoped_residual_runner_allows_batched_research_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate_dir = repo / "candidate"
    candidate_dir.mkdir()
    pair_plan = repo / "pair_plan.json"
    pair_plan.write_text(
        json.dumps(
            {
                "pair_scoped_residual_candidate_rows": [
                    {
                        "residual_transform": "threshold_abs_le_pairs=2@1",
                        "estimated_archive_bytes_removed_vs_baseline": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_profile = repo / "baseline_profile.json"
    baseline_profile.write_text("{}\n", encoding="utf-8")

    plan = build_pair_scoped_residual_bounded_runner_plan(
        pair_plan_path=pair_plan,
        reuse_baseline_profile_path=baseline_profile,
        candidate_dir=candidate_dir,
        output_dir=repo / "runner",
        repo_root=repo,
        scorer_batch_pairs=8,
        allow_batch_shape_research_signal=True,
    )

    row = plan["runner_rows"][0]
    assert row["scorer_batch_pairs"] == 8
    assert row["batch_shape_research_signal"] is True
    assert "--allow-batch-shape-research-signal" in row["profile_command_argv"]
    assert "--allow-batch-shape-research-signal" in row["incremental_response_command_argv"]


def test_hprc_pair_scoped_residual_harvest_binds_receiver_proof_by_sha(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output_dir = repo / "runner"
    profile_dir = output_dir / "candidate-a"
    profile_dir.mkdir(parents=True)
    runner_plan = repo / "runner_plan.json"
    runner_plan.write_text(
        json.dumps(
            {
                "runner_rows": [
                    {
                        "candidate_id": "candidate-a",
                        "expected_profile_report": (
                            profile_dir / "hprc_mlx_component_neutralization_profile.json"
                        ).as_posix(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "hprc_mlx_component_neutralization_profile.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 12.5,
                "baseline_reuse": {"enabled": True},
                "scorer_batch_pairs": 1,
                "batch_shape_research_signal": False,
                "variant_rows": [
                    {"variant_id": "baseline"},
                    {
                        "variant_id": "residual_transform_threshold_abs_le_pairs_3_x",
                        "archive_zip_path": "candidate/archive.zip",
                        "archive_zip_bytes": 859923,
                        "archive_zip_sha256": "a" * 64,
                        "hprc_0bin_path": "candidate/0.bin",
                        "hprc_0bin_sha256": "b" * 64,
                    },
                ],
                "section_value_rows": [
                    {"variant_id": "baseline"},
                    {
                        "variant_id": "residual_transform_threshold_abs_le_pairs_3_x",
                        "delta_nonrate_score": -1.0,
                        "delta_rate_score": -0.2,
                        "delta_total_mlx_score_advisory": -1.2,
                        "archive_bytes_removed_vs_baseline": 303937,
                        "marginal_status": "cut_candidate_distortion_nonworse",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "artifact_retention_plan.json").write_text(
        json.dumps(
            {
                "plan": {
                    "candidates": [],
                    "blocked_candidates": [
                        {"bytes": 10, "blockers": ["mlx_cache_identity_audit_stamp_missing"]}
                    ],
                    "total_reclaimable_bytes": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    proof_dir = repo / "proof" / "receiver_proof"
    proof_dir.mkdir(parents=True)
    (proof_dir / "hprc_receiver_proof.json").write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_ready": True,
                "receiver_output_sha256": "c" * 64,
                "receiver_output_bytes": 3662409600,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    harvest = build_pair_scoped_residual_runner_harvest(
        runner_plan_path=runner_plan,
        candidate_id="candidate-a",
        proof_roots=[repo / "proof"],
        repo_root=repo,
    )

    assert harvest["schema"] == HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_HARVEST_SCHEMA
    assert harvest["archive"]["sha256"] == "a" * 64
    assert harvest["receiver_proof_binding"]["status"] == "linked_by_archive_sha256"
    assert harvest["cleanup"]["status"] == "blocked"
    assert harvest["ready_for_exact_eval_dispatch"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in harvest["exact_axis_gate"]["blockers"]


def test_hprc_pair_scoped_residual_harvest_cli_writes_output(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    profile_dir = repo / "runner" / "candidate-a"
    profile_dir.mkdir(parents=True)
    runner_plan = repo / "runner_plan.json"
    runner_plan.write_text(
        json.dumps(
            {
                "runner_rows": [
                    {
                        "candidate_id": "candidate-a",
                        "expected_profile_report": (
                            profile_dir / "hprc_mlx_component_neutralization_profile.json"
                        ).as_posix(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "hprc_mlx_component_neutralization_profile.json").write_text(
        json.dumps(
            {
                "variant_rows": [
                    {"variant_id": "baseline"},
                    {
                        "variant_id": "candidate",
                        "archive_zip_path": "archive.zip",
                        "archive_zip_bytes": 1,
                        "archive_zip_sha256": "d" * 64,
                    },
                ],
                "section_value_rows": [
                    {"variant_id": "baseline"},
                    {"variant_id": "candidate"},
                ],
            }
        ),
        encoding="utf-8",
    )

    output = repo / "harvest.json"
    exit_code = harvest_runner_tool.main(
        [
            "--repo-root",
            repo.as_posix(),
            "--runner-plan",
            runner_plan.as_posix(),
            "--candidate-id",
            "candidate-a",
            "--output",
            output.as_posix(),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["receiver_proof_binding"]["status"] == "missing"


def test_hprc_mlx_batch_profile_comparison_reports_drift_and_speed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    singleton_profile, batched_profile = _write_batch_compare_fixture(repo)

    comparison = compare_hprc_mlx_batch_profiles(
        singleton_profile_path=singleton_profile,
        batched_profile_path=batched_profile,
        repo_root=repo,
    )

    assert comparison["schema"] == HPRC_MLX_BATCH_PROFILE_COMPARISON_SCHEMA
    assert comparison["wall_clock"]["singleton_scored_variant_count"] == 1
    assert comparison["wall_clock"]["batched_scored_variant_count"] == 2
    assert comparison["max_abs_response_drift"] > 0
    assert comparison["max_abs_delta_drift"] > 0
    assert comparison["ready_for_exact_eval_dispatch"] is False


def test_hprc_mlx_batch_profile_comparison_cli_writes_output(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    singleton_profile, batched_profile = _write_batch_compare_fixture(repo)
    output = repo / "comparison.json"

    exit_code = batch_compare_tool.main(
        [
            "--repo-root",
            repo.as_posix(),
            "--singleton-profile",
            singleton_profile.as_posix(),
            "--batched-profile",
            batched_profile.as_posix(),
            "--output",
            output.as_posix(),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["schema"] == HPRC_MLX_BATCH_PROFILE_COMPARISON_SCHEMA
    assert payload["score_claim"] is False


def test_hprc_incremental_pair_response_patches_changed_pairs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    baseline_components = repo / "baseline_components"
    candidate_components = repo / "candidate_components"
    candidate_cache = repo / "candidate_cache"
    baseline_components.mkdir()
    candidate_components.mkdir()
    candidate_cache.mkdir()
    np.save(baseline_components / "pose.npy", np.asarray([0.10, 0.20, 0.30]))
    np.save(baseline_components / "seg.npy", np.asarray([0.01, 0.02, 0.03]))
    np.save(candidate_components / "pose.npy", np.asarray([0.25, 0.35]))
    np.save(candidate_components / "seg.npy", np.asarray([0.025, 0.035]))
    np.save(candidate_cache / "pair_indices.npy", np.asarray([[2, 3], [4, 5]]))
    baseline_response = repo / "baseline_response.json"
    candidate_response = repo / "candidate_response.json"
    baseline_response.write_text(
        json.dumps(_response_with_components(baseline_components)),
        encoding="utf-8",
    )
    candidate_response.write_text(
        json.dumps(_response_with_components(candidate_components)),
        encoding="utf-8",
    )
    profile = repo / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "variant_rows": [
                    {
                        "variant_id": "baseline",
                        "archive_zip_bytes": 1000,
                        "mlx_response": baseline_response.as_posix(),
                    },
                    {
                        "variant_id": "candidate",
                        "archive_zip_bytes": 800,
                        "mlx_response": candidate_response.as_posix(),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    materialization_report = repo / "cache_report.json"
    materialization_report.write_text(
        json.dumps(
            {
                "cached_pair_count": 2,
                "hprc_direct_cache_report": {
                    "selected_pair_ranges": [[1, 2]],
                    "pair_index_scope": "explicit_pair_ranges",
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hprc_incremental_pair_response_report(
        profile_path=profile,
        candidate_variant_id="candidate",
        candidate_response_path=candidate_response,
        candidate_cache_dir=candidate_cache,
        materialization_report_path=materialization_report,
        repo_root=repo,
    )

    assert report["schema"] == HPRC_INCREMENTAL_PAIR_RESPONSE_SCHEMA
    assert report["changed_pair_rows"] == [1, 2]
    assert report["full_video_pair_count"] == 3
    assert report["archive_bytes_removed_vs_baseline"] == 200
    assert report["delta_avg_posenet_dist"] > 0
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hprc_incremental_runner_execution_prepares_synthetic_profile(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate_dir = repo / "candidate"
    candidate_dir.mkdir()
    frames = np.zeros((4, 8, 10, 3), dtype=np.float32)
    frames[:, :, :, 0] = np.arange(4, dtype=np.float32)[:, None, None] * 11
    (candidate_dir / "0.bin").write_bytes(
        build_compact_receiver_packet_from_lowres_frames(
            frames,
            basis_count=2,
            residual_grid_h=2,
            residual_grid_w=3,
        )
    )
    baseline_components = repo / "baseline_components"
    baseline_components.mkdir()
    np.save(baseline_components / "pose.npy", np.asarray([0.1, 0.2]))
    np.save(baseline_components / "seg.npy", np.asarray([0.01, 0.02]))
    baseline_response = repo / "baseline_response.json"
    baseline_response.write_text(
        json.dumps(_response_with_components(baseline_components)),
        encoding="utf-8",
    )
    baseline_profile = repo / "baseline_profile.json"
    baseline_profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "reference_cache_dir": (repo / "reference_cache").as_posix(),
                "max_pairs": 2,
                "window_pairs": 1,
                "variant_rows": [
                    {
                        "variant_id": "baseline",
                        "archive_zip_bytes": 1000,
                        "hprc_0bin_sha256": "b" * 64,
                        "mlx_response": baseline_response.as_posix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner_plan = repo / "runner_plan.json"
    runner_plan.write_text(
        json.dumps(
            {
                "candidate_dir": candidate_dir.as_posix(),
                "reuse_baseline_profile_path": baseline_profile.as_posix(),
                "runner_rows": [
                    {
                        "candidate_id": "candidate-a",
                        "residual_transform": "threshold_abs_le_pairs=3@1",
                        "pair_ranges": [[1, 1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prep = prepare_hprc_incremental_runner_execution(
        runner_plan_path=runner_plan,
        candidate_id="candidate-a",
        output_dir=repo / "incremental_exec",
        repo_root=repo,
    )

    synthetic = json.loads(Path(prep["synthetic_profile_path"]).read_text())
    assert prep["schema"] == HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA
    assert prep["pair_ranges_arg"] == "1"
    assert Path(prep["archive"]["path"]).is_file()
    assert synthetic["profile_kind"] == "hprc_incremental_pair_scoped_synthetic_profile.v1"
    assert synthetic["variant_rows"][1]["variant_id"].startswith(
        "residual_transform_threshold_abs_le_pairs_3_"
    )
    assert "--pair-ranges" in prep["incremental_command_argv"]
    assert prep["score_claim"] is False


def test_hprc_incremental_runner_execution_report_preserves_cleanup_blocker(
    tmp_path: Path,
) -> None:
    incremental = tmp_path / "incremental.json"
    incremental.write_text(
        json.dumps(
            {
                "changed_pair_rows": [1, 2],
                "full_video_pair_count": 3,
                "archive_bytes_removed_vs_baseline": 200,
                "delta_total_mlx_score_advisory": -0.5,
                "delta_avg_posenet_dist": 0.01,
                "delta_avg_segnet_dist": -0.001,
            }
        ),
        encoding="utf-8",
    )
    retention = tmp_path / "retention.json"
    retention.write_text(
        json.dumps(
            {
                "plan": {
                    "candidates": [],
                    "blocked_candidates": [
                        {
                            "bytes": 123,
                            "blockers": ["mlx_cache_identity_audit_stamp_missing"],
                        }
                    ],
                    "total_reclaimable_bytes": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_hprc_incremental_runner_execution_report(
        prep={
            "schema": HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA,
            "candidate_id": "candidate-a",
            "candidate_variant_id": "candidate",
            "residual_transform": "threshold_abs_le_pairs=3@1",
            "archive": {"sha256": "a" * 64, "bytes": 10},
            "synthetic_profile_path": "profile.json",
            "incremental_command_argv": ["python", "tool.py"],
        },
        incremental_report_path=incremental,
        retention_plan_path=retention,
    )

    assert report["schema"] == HPRC_INCREMENTAL_RUNNER_EXECUTION_SCHEMA
    assert report["incremental_summary"]["changed_pair_count"] == 2
    assert report["receiver_proof_binding"]["status"] == "missing"
    assert report["cleanup"]["status"] == "blocked"
    assert "uncertified_mlx_cache_retained_cleanup_blocker" in report["exact_axis_gate"]["blockers"]
    assert report["score_claim"] is False


def test_hprc_incremental_runner_execution_report_binds_receiver_proof(
    tmp_path: Path,
) -> None:
    incremental = tmp_path / "incremental.json"
    incremental.write_text(
        json.dumps(
            {
                "changed_pair_rows": [1],
                "full_video_pair_count": 3,
                "archive_bytes_removed_vs_baseline": 200,
                "delta_total_mlx_score_advisory": -0.5,
            }
        ),
        encoding="utf-8",
    )
    proof_dir = tmp_path / "proof" / "receiver_proof"
    proof_dir.mkdir(parents=True)
    (proof_dir / "hprc_receiver_proof.json").write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_ready": True,
                "receiver_output_sha256": "b" * 64,
                "receiver_output_bytes": 123,
            }
        ),
        encoding="utf-8",
    )

    report = build_hprc_incremental_runner_execution_report(
        prep={
            "schema": HPRC_INCREMENTAL_RUNNER_EXECUTION_PREP_SCHEMA,
            "candidate_id": "candidate-a",
            "candidate_variant_id": "candidate",
            "residual_transform": "threshold_abs_le_pairs=3@1",
            "archive": {"sha256": "a" * 64, "bytes": 10},
            "synthetic_profile_path": "profile.json",
            "incremental_command_argv": ["python", "tool.py"],
        },
        incremental_report_path=incremental,
        proof_roots=[tmp_path / "proof"],
    )

    assert report["receiver_proof_binding"]["status"] == "linked_by_archive_sha256"
    assert "receiver_proof_missing_for_incremental_runner_candidate_sha" not in report[
        "exact_axis_gate"
    ]["blockers"]


def test_hprc_incremental_runner_execution_comparison_keeps_slow_batch_research_only(
    tmp_path: Path,
) -> None:
    reference = _write_incremental_execution_fixture(
        tmp_path / "singleton",
        scorer_batch_pairs=1,
        elapsed_seconds=100.0,
        cache_elapsed_seconds=10.0,
        delta_total=-1.0,
        delta_pose=-2.0,
        delta_seg=-0.001,
    )
    challenger = _write_incremental_execution_fixture(
        tmp_path / "batch8",
        scorer_batch_pairs=8,
        elapsed_seconds=99.5,
        cache_elapsed_seconds=9.0,
        delta_total=-1.000001,
        delta_pose=-2.000001,
        delta_seg=-0.001000001,
    )

    comparison = compare_hprc_incremental_runner_execution_reports(
        reference_report_path=reference,
        challenger_report_path=challenger,
    )

    assert comparison["schema"] == HPRC_INCREMENTAL_RUNNER_EXECUTION_COMPARISON_SCHEMA
    assert comparison["drift"]["within_tolerance"] is True
    assert comparison["challenger"]["batch_shape_research_signal"] is True
    assert comparison["default_execution_recommendation"]["mode"] == "reference"
    assert comparison["score_claim"] is False


def test_hprc_incremental_exact_gate_bridge_blocks_uncertified_cleanup(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    hprc_0bin = tmp_path / "0.bin"
    hprc_0bin.write_bytes(b"hprc")
    proof_dir = tmp_path / "proof"
    proof_dir.mkdir()
    proof = proof_dir / "hprc_receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "archive_sha256": _sha256(archive),
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_ready": True,
                "receiver_output_sha256": "b" * 64,
                "receiver_output_bytes": 123,
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "execution_report.json"
    report.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-a",
                "candidate_variant_id": "variant-a",
                "archive": {
                    "path": archive.as_posix(),
                    "bytes": archive.stat().st_size,
                    "sha256": _sha256(archive),
                    "hprc_0bin_path": hprc_0bin.as_posix(),
                    "hprc_0bin_sha256": _sha256(hprc_0bin),
                },
                "receiver_proof_binding": {
                    "status": "linked_by_archive_sha256",
                    "proof_path": proof.as_posix(),
                    "archive_sha256": _sha256(archive),
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_ready": True,
                    "receiver_output_sha256": "b" * 64,
                    "receiver_output_bytes": 123,
                    "blockers": [],
                },
                "cleanup": {
                    "status": "blocked",
                    "blocked_bytes_retained": 100,
                    "blockers": ["mlx_cache_identity_audit_stamp_missing"],
                },
                "incremental_summary": {
                    "delta_total_mlx_score_advisory": -1.0,
                },
                "exact_axis_gate": {
                    "ready_for_exact_eval_dispatch": False,
                    "blockers": [
                        "contest_cpu_cuda_exact_eval_not_executed",
                        "mlx_local_response_is_advisory_not_score_authority",
                        "uncertified_mlx_cache_retained_cleanup_blocker",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    bridge = build_hprc_incremental_exact_gate_bridge(
        execution_report_path=report,
        repo_root=tmp_path,
    )

    assert bridge["schema"] == HPRC_INCREMENTAL_EXACT_GATE_BRIDGE_SCHEMA
    assert bridge["archive_custody"]["verified"] is True
    assert bridge["receiver_proof_custody"]["verified"] is True
    assert bridge["cleanup_custody"]["verified"] is False
    assert bridge["exact_packet"]["packet_kind"] == "blocked_exact_packet"
    assert bridge["exact_packet"]["dispatchable_after_lane_claim"] is False
    assert bridge["exact_packet"]["preclaim_blockers"] == [
        "mlx_cache_identity_audit_stamp_missing",
        "cleanup_status_blocked",
    ]
    assert bridge["exact_dispatch_plan"]["dispatchable_after_lane_claim"] is False
    assert bridge["exact_dispatch_plan"]["source_packet_kind"] == "blocked_exact_packet"
    assert "mlx_cache_identity_audit_stamp_missing" in bridge["exact_axis_gate"]["blockers"]
    assert "uncertified_mlx_cache_retained_cleanup_blocker" not in bridge[
        "exact_axis_gate"
    ]["blockers"]
    assert bridge["score_claim"] is False


def test_hprc_incremental_exact_gate_bridge_emits_dispatchable_packet(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    hprc_0bin = tmp_path / "0.bin"
    hprc_0bin.write_bytes(b"hprc")
    proof = tmp_path / "hprc_receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "archive_sha256": _sha256(archive),
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_ready": True,
                "receiver_output_sha256": "b" * 64,
                "receiver_output_bytes": 123,
            }
        ),
        encoding="utf-8",
    )
    retention_plan = tmp_path / "artifact_retention_plan.json"
    retention_plan.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "execution_report.json"
    report.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-a",
                "candidate_variant_id": "variant-a",
                "archive": {
                    "path": archive.as_posix(),
                    "bytes": archive.stat().st_size,
                    "sha256": _sha256(archive),
                    "hprc_0bin_path": hprc_0bin.as_posix(),
                    "hprc_0bin_sha256": _sha256(hprc_0bin),
                },
                "receiver_proof_binding": {
                    "status": "linked_by_archive_sha256",
                    "proof_path": proof.as_posix(),
                    "archive_sha256": _sha256(archive),
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_ready": True,
                    "receiver_output_sha256": "b" * 64,
                    "receiver_output_bytes": 123,
                    "blockers": [],
                },
                "cleanup": {
                    "status": "planned",
                    "plan_path": retention_plan.as_posix(),
                    "blocked_bytes_retained": 0,
                    "reclaimable_bytes": 100,
                    "blockers": [],
                },
                "incremental_summary": {
                    "delta_total_mlx_score_advisory": -1.0,
                },
                "exact_axis_gate": {
                    "ready_for_exact_eval_dispatch": False,
                    "blockers": [
                        "contest_cpu_cuda_exact_eval_not_executed",
                        "mlx_local_response_is_advisory_not_score_authority",
                        "receiver_proof_missing_for_incremental_runner_candidate_sha",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    bridge = build_hprc_incremental_exact_gate_bridge(
        execution_report_path=report,
        repo_root=tmp_path,
    )

    assert bridge["archive_custody"]["verified"] is True
    assert bridge["receiver_proof_custody"]["verified"] is True
    assert bridge["cleanup_custody"]["verified"] is True
    assert bridge["exact_packet"]["packet_kind"] == "dispatchable_exact_packet"
    assert bridge["exact_packet"]["preclaim_blockers"] == []
    assert bridge["exact_packet"]["dispatchable_after_lane_claim"] is True
    assert bridge["exact_dispatch_plan"]["dispatchable_after_lane_claim"] is True
    assert bridge["exact_dispatch_plan"]["source_packet_kind"] == "dispatchable_exact_packet"
    assert bridge["ready_for_exact_eval_dispatch"] is True
    assert "receiver_proof_missing_for_incremental_runner_candidate_sha" not in bridge[
        "exact_axis_gate"
    ]["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in bridge["exact_axis_gate"][
        "blockers"
    ]
    assert bridge["score_claim"] is False


def _write_batch_compare_fixture(repo: Path) -> tuple[Path, Path]:
    singleton_dir = repo / "singleton"
    batched_dir = repo / "batched"
    singleton_dir.mkdir()
    batched_dir.mkdir()
    singleton_responses = _write_compare_responses(singleton_dir, score_offset=0.0)
    batched_responses = _write_compare_responses(batched_dir, score_offset=0.03)
    singleton_profile = singleton_dir / "profile.json"
    batched_profile = batched_dir / "profile.json"
    singleton_profile.write_text(
        json.dumps(
            _compare_profile_payload(
                responses=singleton_responses,
                scorer_batch_pairs=1,
                batch_shape_research_signal=False,
                baseline_reuse_enabled=True,
                elapsed_seconds=100.0,
                delta_total=-1.2,
            )
        ),
        encoding="utf-8",
    )
    batched_profile.write_text(
        json.dumps(
            _compare_profile_payload(
                responses=batched_responses,
                scorer_batch_pairs=8,
                batch_shape_research_signal=True,
                baseline_reuse_enabled=False,
                elapsed_seconds=120.0,
                delta_total=-1.17,
            )
        ),
        encoding="utf-8",
    )
    return singleton_profile, batched_profile


def _write_incremental_execution_fixture(
    root: Path,
    *,
    scorer_batch_pairs: int,
    elapsed_seconds: float,
    cache_elapsed_seconds: float,
    delta_total: float,
    delta_pose: float,
    delta_seg: float,
) -> Path:
    root.mkdir(parents=True)
    incremental = root / "hprc_incremental_pair_response_report.json"
    incremental.write_text(
        json.dumps(
            {
                "changed_pair_rows": [1, 2],
                "scorer_batch_pairs": scorer_batch_pairs,
                "batch_shape_research_signal": scorer_batch_pairs != 1,
                "delta_total_mlx_score_advisory": delta_total,
                "delta_avg_posenet_dist": delta_pose,
                "delta_avg_segnet_dist": delta_seg,
            }
        ),
        encoding="utf-8",
    )
    (root / "mlx_incremental_cache_report.json").write_text(
        json.dumps({"elapsed_seconds": cache_elapsed_seconds}),
        encoding="utf-8",
    )
    report = root / "hprc_incremental_runner_execution_report.json"
    report.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-a",
                "archive": {"sha256": "a" * 64, "bytes": 10},
                "incremental_report_path": incremental.as_posix(),
                "incremental_command": {"elapsed_seconds": elapsed_seconds},
                "incremental_summary": {
                    "delta_total_mlx_score_advisory": delta_total,
                    "delta_avg_posenet_dist": delta_pose,
                    "delta_avg_segnet_dist": delta_seg,
                },
                "cleanup": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )
    return report


def _write_compare_responses(root: Path, *, score_offset: float) -> dict[str, Path]:
    responses: dict[str, Path] = {}
    for variant_id, archive_bytes, base_score in (
        ("baseline", 1000, 10.0),
        ("candidate", 900, 8.8),
    ):
        path = root / f"{variant_id}.json"
        path.write_text(
            json.dumps(
                {
                    "avg_segnet_dist": 0.01 + score_offset,
                    "avg_posenet_dist": 0.02 + score_offset,
                    "canonical_score": base_score + score_offset,
                    "score_rate_contribution": archive_bytes / 1000.0,
                }
            ),
            encoding="utf-8",
        )
        responses[variant_id] = path
    return responses


def _response_with_components(component_dir: Path) -> dict:
    return {
        "components": {
            "artifacts": {
                "posenet_distortion": {
                    "path": (component_dir / "pose.npy").as_posix(),
                },
                "segnet_distortion": {
                    "path": (component_dir / "seg.npy").as_posix(),
                },
            }
        }
    }


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _compare_profile_payload(
    *,
    responses: dict[str, Path],
    scorer_batch_pairs: int,
    batch_shape_research_signal: bool,
    baseline_reuse_enabled: bool,
    elapsed_seconds: float,
    delta_total: float,
) -> dict:
    return {
        "schema": "hprc_mlx_component_neutralization_profile.v1",
        "max_pairs": 600,
        "reference_cache_dir": "/reference/cache",
        "elapsed_seconds": elapsed_seconds,
        "scorer_batch_pairs": scorer_batch_pairs,
        "batch_shape_research_signal": batch_shape_research_signal,
        "baseline_reuse": {"enabled": baseline_reuse_enabled},
        "variant_rows": [
            {
                "variant_id": "baseline",
                "archive_zip_sha256": "a" * 64,
                "hprc_0bin_sha256": "b" * 64,
                "mlx_response": responses["baseline"].as_posix(),
            },
            {
                "variant_id": "candidate",
                "archive_zip_sha256": "c" * 64,
                "hprc_0bin_sha256": "d" * 64,
                "mlx_response": responses["candidate"].as_posix(),
            },
        ],
        "section_value_rows": [
            {"variant_id": "baseline"},
            {
                "variant_id": "candidate",
                "archive_bytes_removed_vs_baseline": 100,
                "delta_nonrate_score": -1.0,
                "delta_rate_score": -0.2,
                "delta_total_mlx_score_advisory": delta_total,
                "delta_avg_posenet_dist": 0.001,
                "delta_avg_segnet_dist": -0.002,
            },
        ],
    }
