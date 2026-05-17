# SPDX-License-Identifier: MIT
"""Tests for the Rule #6 substrate score-response probe."""

from __future__ import annotations

import importlib.util
import json
import pathlib

from tac.exact_eval_custody import contest_score
from tac.scorer_response_probe import (
    VERDICT_BLOCKED_CONTROL_MISMATCH,
    VERDICT_BLOCKED_CUSTODY,
    VERDICT_NO_MEASURABLE_RESPONSE,
    VERDICT_RATE_ONLY_IMPROVEMENT,
    VERDICT_SCORER_RESPONSE_POSITIVE,
    VERDICT_SCORER_RESPONSE_PRESENT_RATE_NEGATIVE,
    VERDICT_SCORE_REGRESSION,
    compare_score_response,
    normalize_score_response_mapping,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _evidence(
    *,
    seg: float = 0.00070,
    pose: float = 0.00003,
    bytes_: int = 180_000,
    axis: str = "contest_cpu",
    runtime_sha: str = SHA_A,
    archive_sha: str = SHA_B,
) -> dict[str, object]:
    return {
        "axis": axis,
        "archive_sha256": archive_sha,
        "runtime_tree_sha256": runtime_sha,
        "n_samples": 600,
        "archive_bytes": bytes_,
        "seg_dist": seg,
        "pose_dist": pose,
        "score": contest_score(seg, pose, bytes_),
        "hardware": "Ubuntu Linux x86_64 CPU",
        "inflate_device": "cpu",
        "eval_device": "cpu",
        "auth_eval_command": "python upstream/evaluate.py --device cpu",
        "log_path": "experiments/results/example/contest_auth_eval.log",
    }


def test_positive_scorer_response_requires_total_and_scorer_improvement() -> None:
    baseline = _evidence()
    candidate = _evidence(seg=0.00068, pose=0.000028, bytes_=180_000)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        min_total_improvement=0.001,
        min_scorer_term_improvement=0.0005,
    )
    assert report.verdict == VERDICT_SCORER_RESPONSE_POSITIVE
    assert report.total_delta is not None and report.total_delta < 0.0
    assert report.scorer_term_delta is not None and report.scorer_term_delta < 0.0


def test_scorer_response_present_rate_negative_classifies_byte_overpay() -> None:
    baseline = _evidence(bytes_=180_000)
    candidate = _evidence(seg=0.00068, pose=0.000028, bytes_=220_000)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        min_total_improvement=0.001,
        min_scorer_term_improvement=0.0005,
    )
    assert report.verdict == VERDICT_SCORER_RESPONSE_PRESENT_RATE_NEGATIVE
    assert report.scorer_term_delta is not None and report.scorer_term_delta < 0.0
    assert report.rate_term_delta is not None and report.rate_term_delta > 0.0


def test_rate_only_improvement_is_not_scorer_response() -> None:
    baseline = _evidence(bytes_=190_000)
    candidate = _evidence(bytes_=180_000)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        min_total_improvement=0.001,
        min_scorer_term_improvement=0.0005,
    )
    assert report.verdict == VERDICT_RATE_ONLY_IMPROVEMENT
    assert report.scorer_term_delta == 0.0


def test_near_tie_is_no_measurable_response() -> None:
    baseline = _evidence()
    candidate = _evidence(seg=0.000699, pose=0.00003, bytes_=180_000)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        min_total_improvement=0.001,
        min_scorer_term_improvement=0.0005,
    )
    assert report.verdict == VERDICT_NO_MEASURABLE_RESPONSE


def test_regression_classifies_positive_total_delta() -> None:
    baseline = _evidence()
    candidate = _evidence(seg=0.00072, pose=0.000035, bytes_=180_000)
    report = compare_score_response(baseline=baseline, candidate=candidate)
    assert report.verdict == VERDICT_SCORE_REGRESSION
    assert report.total_delta is not None and report.total_delta > 0.0


def test_ablation_mode_requires_runtime_match() -> None:
    baseline = _evidence(runtime_sha=SHA_A)
    candidate = _evidence(runtime_sha=SHA_C)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        mode="ablation",
    )
    assert report.verdict == VERDICT_BLOCKED_CONTROL_MISMATCH
    assert "runtime_tree_mismatch" in report.blockers


def test_candidate_mode_allows_runtime_mismatch() -> None:
    baseline = _evidence(runtime_sha=SHA_A)
    candidate = _evidence(seg=0.00068, pose=0.000028, runtime_sha=SHA_C)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        mode="candidate",
    )
    assert report.verdict == VERDICT_SCORER_RESPONSE_POSITIVE


def test_strict_custody_blocks_missing_hardware() -> None:
    baseline = _evidence()
    candidate = _evidence()
    candidate.pop("hardware")
    report = compare_score_response(baseline=baseline, candidate=candidate)
    assert report.verdict == VERDICT_BLOCKED_CUSTODY
    assert "candidate:hardware_missing" in report.blockers


def test_missing_axis_blocks_even_without_expected_axis() -> None:
    baseline = _evidence()
    candidate = _evidence(seg=0.00068, pose=0.000028)
    baseline.pop("axis")
    candidate.pop("axis")
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        strict_exact_custody=False,
    )
    assert report.verdict == VERDICT_BLOCKED_CUSTODY
    assert "baseline:axis_missing" in report.blockers
    assert "candidate:axis_missing" in report.blockers


def test_normalizes_bracketed_axis_labels_without_promoting_advisory() -> None:
    contest = normalize_score_response_mapping({"axis": "[contest-CPU]"})
    advisory = normalize_score_response_mapping({"axis": "[macOS-CPU advisory]"})
    assert contest["axis"] == "contest_cpu"
    assert advisory["axis"] == "macos_cpu_advisory"


def test_relaxed_custody_allows_score_field_only_probe() -> None:
    baseline = _evidence()
    candidate = _evidence(seg=0.00068, pose=0.000028)
    for key in ("hardware", "inflate_device", "eval_device", "auth_eval_command", "log_path"):
        baseline.pop(key)
        candidate.pop(key)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        strict_exact_custody=False,
    )
    assert report.verdict == VERDICT_SCORER_RESPONSE_POSITIVE


def test_json_report_is_stable_shape() -> None:
    report = compare_score_response(
        baseline=_evidence(),
        candidate=_evidence(seg=0.00068, pose=0.000028),
    )
    payload = report.to_json_dict()
    assert payload["schema"] == "substrate_score_response_probe_v1"
    assert payload["deltas"]["total_delta"] == report.total_delta
    assert payload["baseline"]["axis"] == "contest_cpu"


def test_normalizes_contest_auth_eval_schema() -> None:
    payload = {
        "score_axis": "cpu_advisory",
        "canonical_score": contest_score(0.00056, 0.000032, 178_262),
        "avg_segnet_dist": 0.00056,
        "avg_posenet_dist": 0.000032,
        "archive_size_bytes": 178_262,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": SHA_B,
            "device": "cpu",
            "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cpu"],
            "inflate_runtime_manifest": {"runtime_tree_sha256": SHA_A},
        },
    }
    normalized = normalize_score_response_mapping(payload)
    assert normalized["axis"] == "macos_cpu_advisory"
    assert normalized["archive_sha256"] == SHA_B
    assert normalized["runtime_tree_sha256"] == SHA_A
    assert normalized["archive_bytes"] == 178_262
    assert normalized["seg_dist"] == 0.00056
    assert normalized["pose_dist"] == 0.000032


def test_relaxed_probe_accepts_contest_auth_eval_schema() -> None:
    baseline = {
        "score_axis": "cpu_advisory",
        "canonical_score": contest_score(0.00056, 0.000032, 178_262),
        "avg_segnet_dist": 0.00056,
        "avg_posenet_dist": 0.000032,
        "archive_size_bytes": 178_262,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": SHA_B,
            "device": "cpu",
            "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cpu"],
            "inflate_runtime_manifest": {"runtime_tree_sha256": SHA_A},
        },
    }
    candidate = {
        **baseline,
        "canonical_score": contest_score(0.00056, 0.000032, 178_258),
        "archive_size_bytes": 178_258,
    }
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        mode="ablation",
        strict_exact_custody=False,
        min_total_improvement=0.000001,
        min_scorer_term_improvement=0.000001,
    )
    assert report.verdict == VERDICT_RATE_ONLY_IMPROVEMENT
    assert report.baseline is not None
    assert report.baseline.axis == "macos_cpu_advisory"


def test_normalizes_exact_cuda_result_review_schema() -> None:
    payload = {
        "score_axis": "contest_cuda",
        "canonical_score": contest_score(0.0006426, 0.00003236, 187_209),
        "score_recomputation": {
            "archive_bytes": 187_209,
            "avg_segnet_dist": 0.0006426,
            "avg_posenet_dist": 0.00003236,
            "recomputed_score": contest_score(0.0006426, 0.00003236, 187_209),
        },
        "custody": {
            "archive_sha256": SHA_B,
            "command": [
                "/workspace/pact/experiments/contest_auth_eval.py",
                "--device",
                "cuda",
                "--inflate-device",
                "auto",
            ],
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "n_samples": 600,
        },
        "runtime_custody": {
            "runtime_tree_sha256": SHA_A,
            "inflated_output_aggregate_sha256": SHA_C,
            "inflated_output_manifest_sha256": "d" * 64,
        },
    }
    normalized = normalize_score_response_mapping(payload)
    assert normalized["axis"] == "contest_cuda"
    assert normalized["archive_sha256"] == SHA_B
    assert normalized["runtime_tree_sha256"] == SHA_A
    assert normalized["archive_bytes"] == 187_209
    assert normalized["seg_dist"] == 0.0006426
    assert normalized["pose_dist"] == 0.00003236
    assert normalized["hardware"] == "Tesla T4 cuda"
    assert normalized["inflate_device"] == "auto"
    assert normalized["eval_device"] == "cuda"
    assert normalized["raw_output_aggregate_sha256"] == SHA_C


def test_relaxed_probe_accepts_exact_cuda_result_review_schema() -> None:
    baseline = {
        "score_axis": "contest_cuda",
        "score_recomputation": {
            "archive_bytes": 187_209,
            "avg_segnet_dist": 0.0006426,
            "avg_posenet_dist": 0.00003236,
            "recomputed_score": contest_score(0.0006426, 0.00003236, 187_209),
        },
        "custody": {
            "archive_sha256": SHA_B,
            "command": ["experiments/contest_auth_eval.py", "--device", "cuda"],
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "n_samples": 600,
        },
        "runtime_custody": {"runtime_tree_sha256": SHA_A},
    }
    candidate = {
        **baseline,
        "score_recomputation": {
            "archive_bytes": 186_380,
            "avg_segnet_dist": 0.0006426,
            "avg_posenet_dist": 0.00003236,
            "recomputed_score": contest_score(0.0006426, 0.00003236, 186_380),
        },
    }
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        mode="candidate",
        strict_exact_custody=False,
        min_total_improvement=0.000001,
        min_scorer_term_improvement=0.000001,
    )
    assert report.verdict == VERDICT_RATE_ONLY_IMPROVEMENT
    assert report.baseline is not None
    assert report.baseline.axis == "contest_cuda"


def _load_cli_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "probe_substrate_score_response.py"
    spec = importlib.util.spec_from_file_location("probe_substrate_score_response", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    cli = _load_cli_module()
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_json = tmp_path / "out" / "probe.json"
    output_md = tmp_path / "out" / "probe.md"
    baseline_path.write_text(json.dumps(_evidence()), encoding="utf-8")
    candidate_path.write_text(
        json.dumps(_evidence(seg=0.00068, pose=0.000028)),
        encoding="utf-8",
    )
    rc = cli.main(
        [
            "--baseline-json",
            str(baseline_path),
            "--candidate-json",
            str(candidate_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--axis",
            "contest_cpu",
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == VERDICT_SCORER_RESPONSE_POSITIVE
    assert "SCORER_RESPONSE_POSITIVE" in output_md.read_text(encoding="utf-8")
