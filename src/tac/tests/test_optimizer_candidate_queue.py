from __future__ import annotations

import json
from pathlib import Path

from tac.optimizer.candidate_queue import QUEUE_SCHEMA, build_candidate_queue
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_a1_rollup_merges_m5_ranking_without_dispatch_overclaim(tmp_path: Path) -> None:
    repo = tmp_path
    manifest_a = _write_json(
        repo / "experiments/results/coord/v_a/build_manifest.json",
        {
            "archive_path": "experiments/results/coord/v_a/submission_dir/archive.zip",
            "archive_sha256": "a" * 64,
            "archive_size_bytes": 178262,
            "archive_unchanged_from_a1": True,
            "inflate_py_sha256_new": "b" * 64,
            "runtime_smoke_checked": False,
        },
    )
    manifest_b = _write_json(
        repo / "experiments/results/coord/v_b/build_manifest.json",
        {
            "archive_path": "experiments/results/coord/v_b/submission_dir/archive.zip",
            "archive_sha256": "c" * 64,
            "archive_size_bytes": 178262,
            "archive_unchanged_from_a1": True,
            "inflate_py_sha256_new": "d" * 64,
            "runtime_smoke_checked": False,
        },
    )
    rollup = _write_json(
        repo / "experiments/results/coord/rollup.json",
        {
            "schema_version": "constrained_coord_search_rollup_v1",
            "lane_id": "lane_pr101_bias_constrained_coord_search",
            "evidence_grade": "[predicted; constrained coord search on A1 substrate]",
            "variants": [
                {
                    "variant_id": "v_a",
                    "coords": {"c0_0": -1.0, "c0_2": -1.0, "c1_1": -1.0},
                    "build_manifest_relpath": manifest_a.relative_to(repo).as_posix(),
                    "submission_name": "v_a",
                    "inflate_py_sha256": "b" * 64,
                    "archive_sha256": "a" * 64,
                },
                {
                    "variant_id": "v_b",
                    "coords": {"c0_0": -1.0, "c0_2": 0.0, "c1_1": -1.0},
                    "build_manifest_relpath": manifest_b.relative_to(repo).as_posix(),
                    "submission_name": "v_b",
                    "inflate_py_sha256": "d" * 64,
                    "archive_sha256": "c" * 64,
                },
            ],
        },
    )
    m5 = _write_json(
        repo / "experiments/results/m5/sweep_manifest.json",
        {
            "schema_version": 1,
            "tool": "tools/sweep_m5max_hnerv_cluster.py",
            "summary": {
                "operator_decision_queue": [
                    {
                        "candidate_id": "v_a",
                        "macos_cpu_score": 0.1930,
                        "predicted_contest_cpu_gha": 0.19299,
                        "tag": "[macOS-CPU calibrated]",
                    },
                    {
                        "candidate_id": "v_b",
                        "macos_cpu_score": 0.1928,
                        "predicted_contest_cpu_gha": 0.19279,
                        "tag": "[macOS-CPU calibrated]",
                    },
                ]
            },
        },
    )

    queue = build_candidate_queue([rollup, m5], repo_root=repo, top_k=2)

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["dispatch_ready_count"] == 0
    assert [row["candidate_id"] for row in queue["top_k"]] == ["v_b", "v_a"]
    best = queue["top_k"][0]
    assert best["ready_for_exact_eval_dispatch"] is False
    assert best["score_claim"] is False
    assert best["archive_path"] == "experiments/results/coord/v_b/submission_dir/archive.zip"
    assert best["candidate_archive_sha256"] == "c" * 64
    assert best["predicted_contest_cpu_gha"] == 0.19279
    assert "macos_cpu_is_not_contest_cuda_evidence" in best["dispatch_blockers"]
    assert "requires_exact_eval_readiness_gate" in best["dispatch_blockers"]


def test_codec_search_report_rows_are_payload_planning_not_archive_dispatch(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "optuna_search_report.json",
        {
            "schema": "codec_op_optuna_search_report_v1",
            "tool": "tools/codec_op_optuna_search.py",
            "op_module": "fixture.module",
            "op_class": "FixtureCodecOp",
            "evidence_grade": "[CPU-prep+optuna_tpe]",
            "evidence_semantics": "cpu_codec_op_search_forensic",
            "all_evaluations": [
                {
                    "eval_idx": 0,
                    "params": {"quality": 11},
                    "bytes_out": 120,
                    "reconstruction_rms": 0.0,
                    "fitness": 120.0,
                    "pareto_frontier": True,
                    "materialized_payload_path": "payloads/eval_00000.section",
                    "materialized_payload_sha256": "e" * 64,
                    "materialized_payload_bytes": 120,
                    "materialized_payload_contract": "raw_codecop_encode_blob",
                },
                {
                    "eval_idx": 1,
                    "params": {"quality": 1},
                    "bytes_out": -1,
                    "fitness": None,
                    "error": "RuntimeError: fixture",
                },
            ],
        },
    )

    queue = build_candidate_queue([report], repo_root=tmp_path, top_k=2)

    best = queue["top_k"][0]
    assert best["candidate_id"] == "fixturecodecop_eval_00000"
    assert best["candidate_substream_bytes"] == 120
    assert "archive_path" not in best
    assert best["ready_for_exact_eval_dispatch"] is False
    assert best["score_affecting_payload_changed"] is False
    assert "codec_op_payload_not_archive_zip" in best["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in best["dispatch_blockers"]

    failed = queue["top_k"][1]
    assert failed["candidate_id"] == "fixturecodecop_eval_00001"
    assert "optimizer_eval_failed" in failed["dispatch_blockers"]


def test_predicted_param_sweep_manifest_is_forced_non_dispatchable(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "codec_sweep_manifest.json",
        {
            "schema_version": "codec_op_param_sweep_manifest.v1",
            "evidence_semantics": "cpu_substrate_predicted_band",
            "candidates": [
                {
                    "candidate_id": "unsafe_source_claim",
                    "predicted_score": 0.1,
                    "ready_for_exact_eval_dispatch": True,
                    "score_claim": True,
                    "op_params": {"q": 11},
                }
            ],
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["evidence_semantics"] == "cpu_substrate_predicted_band"
    assert "predicted_score_is_not_score_evidence" in row["dispatch_blockers"]


def test_kaggle_proxy_manifest_becomes_canonical_non_dispatchable_queue_row(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "proxy_sweep_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_sweep_v1",
            "optimizer": "cmaes",
            "optimizer_status": "cmaes_style_stdlib",
            "evidence_semantics": "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval",
            "dispatch_blockers": [
                "kaggle_proxy_substrate_not_contest_exact_eval",
                "no_archive_zip_emitted",
            ],
            "best_candidate": {
                "candidate_id": "proxy_cmaes_0007",
                "trial_index": 7,
                "optimizer": "cmaes",
                "optimizer_status": "cmaes_style_stdlib",
                "params": {"delta_scale": 0.01, "bias_r": -1.0},
                "proxy_objective": 0.192851,
                "proxy_components": {"anchor_proximity": 0.0},
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "proxy_only": True,
            },
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_id"] == "proxy_cmaes_0007"
    assert row["rank_score"] == 0.192851
    assert row["rank_score_field"] == "proxy_objective"
    assert row["target_modes"] == ["contest_exact_eval_planning"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert "kaggle_proxy_substrate_not_contest_exact_eval" in row["dispatch_blockers"]
    assert "kaggle_proxy_output_requires_archive_builder_promotion" in row["dispatch_blockers"]
    assert validate_proxy_candidate(row) == []
    assert queue["dispatch_ready_count"] == 0


def test_queue_sanitizes_non_finite_legacy_telemetry(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "meta_lagrangian_report.json",
        {
            "schema": "meta_lagrangian_search_v1",
            "evidence_semantics": "local_proxy_prediction_forensic",
            "top_k_forensic": [
                {
                    "candidate_id": "legacy_proxy",
                    "archive_bytes": 159973,
                    "proxy_score": 0.1927,
                    "rank_key": float("inf"),
                }
            ],
        },
    )

    queue = build_candidate_queue([report], repo_root=tmp_path)

    assert queue["top_k"][0]["rank_key"] is None
    json.dumps(queue, allow_nan=False)


def test_concrete_eval_queue_outranks_lower_numeric_proxy_score(tmp_path: Path) -> None:
    rollup_manifest = _write_json(
        tmp_path / "experiments/results/coord/v_real/build_manifest.json",
        {
            "archive_path": "experiments/results/coord/v_real/submission_dir/archive.zip",
            "archive_sha256": "f" * 64,
            "archive_size_bytes": 178262,
            "inflate_py_sha256_new": "1" * 64,
        },
    )
    rollup = _write_json(
        tmp_path / "experiments/results/coord/rollup.json",
        {
            "schema_version": "constrained_coord_search_rollup_v1",
            "variants": [
                {
                    "variant_id": "v_real",
                    "build_manifest_relpath": rollup_manifest.relative_to(tmp_path).as_posix(),
                }
            ],
        },
    )
    m5 = _write_json(
        tmp_path / "experiments/results/m5/sweep_manifest.json",
        {
            "tool": "tools/sweep_m5max_hnerv_cluster.py",
            "summary": {
                "operator_decision_queue": [
                    {
                        "candidate_id": "v_real",
                        "predicted_contest_cpu_gha": 0.193,
                        "macos_cpu_score": 0.193,
                    }
                ]
            },
        },
    )
    proxy = _write_json(
        tmp_path / "reports/meta_lagrangian.json",
        {
            "schema": "meta_lagrangian_search_v1",
            "evidence_semantics": "local_proxy_prediction_forensic",
            "top_k_forensic": [
                {
                    "candidate_id": "lower_proxy_only",
                    "proxy_score": 0.100,
                    "archive_path": None,
                }
            ],
        },
    )

    queue = build_candidate_queue([rollup, m5, proxy], repo_root=tmp_path)

    assert [row["candidate_id"] for row in queue["top_k"]] == [
        "v_real",
        "lower_proxy_only",
    ]
