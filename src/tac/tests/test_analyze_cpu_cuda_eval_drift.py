from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool(name: str):
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analyze_pair_decomposes_pr102_cpu_cuda_gap() -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    pr_row = {
        "pr": 102,
        "title": "fixture",
        "url": "https://example.test/pr/102",
        "eval_comments": [
            {
                "device": "cuda",
                "created_at": "2026-05-04T16:58:03Z",
                "pose": 0.00017347,
                "seg": 0.00067565,
                "archive_bytes": 178_981,
            },
            {
                "device": "cpu",
                "created_at": "2026-05-05T17:05:18Z",
                "pose": 0.00003460,
                "seg": 0.00057599,
                "archive_bytes": 178_981,
            },
        ],
    }

    pair = mod.analyze_pair(pr_row)

    assert pair is not None
    assert pair["same_archive_bytes"] is True
    assert pair["cuda"]["score"] > pair["cpu"]["score"]
    assert abs(pair["ratios"]["pose_distortion_cuda_over_cpu"] - 5.013583815) < 1e-6
    assert abs(pair["ratios"]["seg_distortion_cuda_over_cpu"] - 1.173023837) < 1e-6
    assert 0.68 < pair["gaps_cuda_minus_cpu"]["pose_gap_share"] < 0.72


def test_build_analysis_preserves_non_promotable_mechanism_boundary() -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    scorecard = {
        "schema": "public_pr_eval_comment_scorecard.v1",
        "rows": [
            {
                "pr": 100,
                "title": "fixture",
                "url": "https://example.test/pr/100",
                "eval_comments": [
                    {
                        "device": "cuda",
                        "created_at": "2026-05-04T00:00:00Z",
                        "pose": 0.00017198,
                        "seg": 0.00067623,
                        "archive_bytes": 178_981,
                    },
                    {
                        "device": "cpu",
                        "created_at": "2026-05-05T00:00:00Z",
                        "pose": 0.00003443,
                        "seg": 0.00057654,
                        "archive_bytes": 178_981,
                    },
                ],
            }
        ],
    }

    analysis = mod.build_analysis(scorecard)

    assert analysis["paired_pr_count"] == 1
    assert analysis["input_schema"] == "public_pr_eval_comment_scorecard.v1"
    assert analysis["score_claim"] is False
    assert analysis["promotion_eligible"] is False
    assert analysis["rank_or_kill_eligible"] is False
    assert analysis["mechanism_claim_proven"] is False
    assert abs(analysis["summary"]["pose_tau_from_public_cpu_comments"] - 0.00586770824) < 1e-12


def _exact_payload(
    *,
    device: str,
    score: float,
    pose: float,
    seg: float,
    raw_sha: str,
) -> dict:
    return {
        "score_recomputed_from_components": score,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 185_578,
        "runtime_content_tree_sha256": "b" * 64,
        "device": device,
        "n_samples": 600,
        "evidence_grade": "contest-CPU" if device == "cpu" else "A++",
        "gpu_t4_match": device == "cuda",
        "hardware": (
            "github-actions-ubuntu-latest-x86_64"
            if device == "cpu"
            else "Tesla T4"
        ),
        "provenance": {
            "device": device,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_t4_match": device == "cuda",
            "inflated_output_manifest": {
                "path": f"{device}/inflated_outputs_manifest.json",
                "sha256": f"{device}-manifest",
                "payload": {
                    "aggregate_sha256": raw_sha,
                    "raw_file_count": 1,
                    "total_bytes": 603_979_776,
                },
            },
        },
    }


def test_analyze_exact_pair_classifies_different_raw_outputs(tmp_path: Path) -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    cpu_json.write_text(
        json.dumps(
            _exact_payload(
                device="cpu",
                score=0.2296576634626332,
                pose=0.000164,
                seg=0.00065592,
                raw_sha="1" * 64,
            )
        ),
        encoding="utf-8",
    )
    cuda_json.write_text(
        json.dumps(
            _exact_payload(
                device="cuda",
                score=0.20898305277982338,
                pose=0.00003360,
                seg=0.00067084,
                raw_sha="2" * 64,
            )
        ),
        encoding="utf-8",
    )

    pair = mod.analyze_exact_pair(cpu_json, cuda_json)

    assert pair["valid_for_mechanism_analysis"] is True
    assert pair["same_archive_sha256"] is True
    assert pair["same_runtime_tree_sha256"] is True
    assert pair["raw_output_pairing_status"] == "different_inflated_outputs"
    assert pair["mechanism_class"] == "different_raw_outputs_runtime_or_inflate_drift"
    assert pair["gaps_cuda_minus_cpu"]["score"] < 0
    assert pair["gaps_cuda_minus_cpu"]["pose_term"] < 0
    assert pair["gaps_cuda_minus_cpu"]["seg_term"] > 0
    assert pair["score_claim"] is False


def test_analyze_exact_pair_keeps_score_pair_but_blocks_mechanism_without_raw_outputs(
    tmp_path: Path,
) -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    cpu_payload = _exact_payload(
        device="cpu",
        score=0.2296576634626332,
        pose=0.000164,
        seg=0.00065592,
        raw_sha="1" * 64,
    )
    cuda_payload = _exact_payload(
        device="cuda",
        score=0.20898305277982338,
        pose=0.00003360,
        seg=0.00067084,
        raw_sha="2" * 64,
    )
    cpu_payload["provenance"].pop("inflated_output_manifest")
    cuda_payload["provenance"].pop("inflated_output_manifest")
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    cpu_json.write_text(json.dumps(cpu_payload), encoding="utf-8")
    cuda_json.write_text(json.dumps(cuda_payload), encoding="utf-8")

    pair = mod.analyze_exact_pair(cpu_json, cuda_json)

    assert pair["valid_for_pair_score_analysis"] is True
    assert pair["valid_for_mechanism_analysis"] is False
    assert pair["mechanism_class"] == "same_archive_runtime_raw_outputs_unmeasured"
    assert pair["mechanism_blockers"] == ["raw_output_manifest_missing"]
    assert pair["blockers"] == []


def test_analyze_exact_pair_preserves_axis_scores_when_pair_custody_incomplete(
    tmp_path: Path,
) -> None:
    mod = _load_tool("analyze_cpu_cuda_eval_drift")
    cpu_payload = _exact_payload(
        device="cpu",
        score=0.19284757743677347,
        pose=0.00003286,
        seg=0.00056023,
        raw_sha="1" * 64,
    )
    cuda_payload = _exact_payload(
        device="cuda",
        score=0.2263520234784395,
        pose=0.00017103,
        seg=0.00066299,
        raw_sha="2" * 64,
    )
    cpu_payload.pop("runtime_content_tree_sha256")
    cpu_payload["provenance"].pop("inflated_output_manifest")
    cuda_payload["provenance"].pop("inflated_output_manifest")
    cpu_json = tmp_path / "cpu.json"
    cuda_json = tmp_path / "cuda.json"
    cpu_json.write_text(json.dumps(cpu_payload), encoding="utf-8")
    cuda_json.write_text(json.dumps(cuda_payload), encoding="utf-8")

    pair = mod.analyze_exact_pair(cpu_json, cuda_json)

    assert pair["valid_individual_axis_scores"] is True
    assert pair["valid_same_archive_axis_score_pair"] is True
    assert pair["valid_for_pair_score_analysis"] is False
    assert pair["valid_for_mechanism_analysis"] is False
    assert pair["individual_axis_blockers"] == []
    assert "cpu_runtime_tree_sha256_missing" in pair["blockers"]
