# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tac.substrates.hprc.archive_candidate as hprc_candidate  # noqa: E402
from tac.optimization.archive_bound_candidate_runtime_bridge import (  # noqa: E402
    build_archive_bound_candidate_runtime_package,
)
from tac.substrates.hprc.campaign import (  # noqa: E402
    HPRC_CAMPAIGN_MANIFEST_SCHEMA,
    HPRC_EXACT_READINESS_REFUSAL_SCHEMA,
    materialize_minimal_hprc_campaign,
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
