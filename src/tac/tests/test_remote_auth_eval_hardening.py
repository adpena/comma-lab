# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    _scan_remote_distillation_promotion_provenance,
    _scan_remote_lane_auth_eval_fragile_parse,
    check_launch_retry_wrapper_singleflight_and_signal_safe,
    check_modal_cpu_auth_eval_is_advisory_only,
    check_modal_recovery_cli_guidance_current,
    check_remote_distillation_promotion_provenance,
    check_remote_lane_auth_eval_json_adjudication,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODAL_RECOVER = REPO_ROOT / "experiments" / "modal_recover_lane.py"


def _load_adjudicator():
    path = REPO_ROOT / "scripts" / "adjudicate_contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("_adjudicate_contest_auth_eval", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modal_recover_preserves_artifacts_while_wiring_cost_band_anchor() -> None:
    text = MODAL_RECOVER.read_text()

    assert "append_modal_training_cost_anchor" in text
    assert "append_modal_training_terminal_claim" in text
    assert "cost_band_anchor_append_error.json" in text
    assert "modal_training_terminal_claim_error.json" in text
    assert "cost-band anchor appended" in text
    assert "cost-band anchor skipped" in text
    assert "terminal claim appended" in text
    assert "failed_modal_training_result_cache_expired" in text
    assert "score_claim" in text
    assert "promotion_eligible" in text


def _source_manifest_payload(files: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "tool": "test",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _file_manifest_entry(repo_root: Path, rel: str) -> dict[str, object]:
    raw = (repo_root / rel).read_bytes()
    return {
        "path": rel,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_adjudicator_uses_recomputed_json_score_not_human_formula(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"deterministic archive bytes")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 1.04,
        "score_recomputed_from_components": 1.0440481283330025,
        "avg_posenet_dist": 0.0034602,
        "avg_segnet_dist": 0.0040083,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "gpu_t4_match": False,
        },
    }))

    provenance = tmp_path / "provenance.json"
    result_copy = tmp_path / "RESULT_JSON"
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=str(result_copy),
        baseline_score=1.05,
        baseline_archive_bytes=694074,
        predicted_band=[1.04, 1.05],
        hard_kill_above=1.05,
        delta_key="score_delta_vs_lane_g_v3",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
    )

    result = adjudicator.adjudicate(args)

    assert result["score_recomputed"] == pytest.approx(1.0440481283330025)
    assert result["score_recomputed"] != 100.0
    assert result["regression_triggered"] is False
    assert result["evidence_grade"] == "A score-grade"

    prov = json.loads(provenance.read_text())
    assert prov["contest_cuda_score_recomputed"] == pytest.approx(1.0440481283330025)
    assert prov["contest_cuda_score_reported_rounded"] == pytest.approx(1.04)
    assert prov["contest_cuda_avg_posenet_dist"] == pytest.approx(0.0034602)
    assert prov["contest_cuda_avg_segnet_dist"] == pytest.approx(0.0040083)
    assert prov["component_gates"] == []
    assert prov["component_gate_triggered"] is False
    assert prov["lane_status"] == "IN_PREDICTED_BAND"
    assert prov["regression_triggered"] is False
    assert prov["regression_scope"] == "measured_implementation_config_only_pending_review"
    assert prov["promotion_eligible"] is False
    assert prov["scientific_score_eligible"] is True
    assert prov["hardware_promotion_gate_triggered"] is True
    assert prov["paper_claim_grade"] == "A score-grade; T4/equivalent promotion required"
    assert prov["allowed_use"] == [
        "diagnostic_score_screen",
        "requires_t4_confirmation",
        "no_promotion",
    ]
    assert "hard_kill_triggered" not in prov
    assert prov["contest_cuda_gpu_t4_match"] is False
    assert prov["contest_equivalent_hardware"] is False


def test_adjudicator_records_source_manifest_runtime_closure(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    repo = tmp_path / "repo"
    runtime = repo / "experiments/public_runtime_adapters/pr104"
    external = repo / "experiments/results/public_pr104/source/submissions/qhnerv_ft_best"
    runtime.mkdir(parents=True)
    external.mkdir(parents=True)
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n")
    (external / "inflate.py").write_text("print('inflate')\n")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    source_manifest = tmp_path / "source_manifest.json"
    files = [
        _file_manifest_entry(repo, "experiments/public_runtime_adapters/pr104/inflate.sh"),
        _file_manifest_entry(
            repo,
            "experiments/results/public_pr104/source/submissions/qhnerv_ft_best/inflate.py",
        ),
    ]
    source_manifest.write_text(json.dumps(_source_manifest_payload(files)) + "\n")
    manifest_sha = json.loads(source_manifest.read_text())["manifest_sha256"]

    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(
        json.dumps(
            {
                "final_score": 0.5,
                "score_recomputed_from_components": 0.5,
                "avg_posenet_dist": 0.003,
                "avg_segnet_dist": 0.004,
                "archive_size_bytes": archive.stat().st_size,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": archive_sha,
                    "device": "cuda",
                    "gpu_t4_match": True,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "b" * 64,
                        "files": [
                            {
                                "repo_relative_path": "experiments/public_runtime_adapters/pr104/inflate.sh",
                                "bytes": files[0]["bytes"],
                                "sha256": files[0]["sha256"],
                            }
                        ],
                        "external_dependency_roots": [
                            {
                                "repo_relative_root": (
                                    "experiments/results/public_pr104/source/submissions/"
                                    "qhnerv_ft_best"
                                ),
                                "exists": True,
                                "files": [
                                    {
                                        "repo_relative_path": (
                                            "experiments/results/public_pr104/source/submissions/"
                                            "qhnerv_ft_best/inflate.py"
                                        ),
                                        "bytes": files[1]["bytes"],
                                        "sha256": files[1]["sha256"],
                                    }
                                ],
                            }
                        ],
                    },
                },
            }
        )
    )
    (tmp_path / "lightning_queue_metadata.json").write_text(
        json.dumps(
            {
                "queue_metadata": {
                    "source_manifest": str(source_manifest),
                    "source_manifest_sha256": manifest_sha,
                }
            }
        )
        + "\n"
    )
    (tmp_path / "eval_provenance.json").write_text(
        json.dumps(json.loads(contest_json.read_text())["provenance"]) + "\n"
    )

    result = adjudicator.adjudicate(
        argparse.Namespace(
            contest_json=str(contest_json),
            provenance=str(tmp_path / "adjudication_provenance.json"),
            archive=str(archive),
            result_copy=str(tmp_path / "contest_auth_eval.adjudicated.json"),
            baseline_score=0.6,
            baseline_archive_bytes=None,
            predicted_band=[0.4, 0.6],
            hard_kill_above=None,
            regression_threshold=0.7,
            delta_key="score_delta_vs_baseline",
            required_device="cuda",
            required_samples=600,
            max_sane_score=10.0,
            max_posenet_dist=None,
            max_segnet_dist=None,
            baseline_posenet_dist=None,
            baseline_segnet_dist=None,
            max_posenet_relative=None,
            max_segnet_relative=None,
            component_reference_label="baseline",
        )
    )

    prov = json.loads((tmp_path / "adjudication_provenance.json").read_text())
    assert result["promotion_eligible"] is True
    assert prov["source_manifest_sha256"] == manifest_sha
    assert prov["source_manifest_runtime_closure"]["verified"] is True
    assert prov["source_manifest_runtime_closure"]["checked_runtime_file_count"] == 2
    assert prov["source_manifest_runtime_closure_gate_triggered"] is False


def test_adjudicator_gates_promotion_on_source_manifest_runtime_sha_mismatch(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    repo = tmp_path / "repo"
    runtime = repo / "experiments/public_runtime_adapters/pr104"
    runtime.mkdir(parents=True)
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n")
    source_manifest = tmp_path / "source_manifest.json"
    files = [_file_manifest_entry(repo, "experiments/public_runtime_adapters/pr104/inflate.sh")]
    source_manifest.write_text(json.dumps(_source_manifest_payload(files)) + "\n")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(
        json.dumps(
            {
                "final_score": 0.5,
                "score_recomputed_from_components": 0.5,
                "avg_posenet_dist": 0.003,
                "avg_segnet_dist": 0.004,
                "archive_size_bytes": archive.stat().st_size,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": archive_sha,
                    "device": "cuda",
                    "gpu_t4_match": True,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "c" * 64,
                        "files": [
                            {
                                "repo_relative_path": "experiments/public_runtime_adapters/pr104/inflate.sh",
                                "bytes": files[0]["bytes"],
                                "sha256": "d" * 64,
                            }
                        ],
                        "external_dependency_roots": [],
                    },
                },
            }
        )
    )
    manifest_sha = json.loads(source_manifest.read_text())["manifest_sha256"]
    (tmp_path / "lightning_queue_metadata.json").write_text(
        json.dumps(
            {
                "queue_metadata": {
                    "source_manifest": str(source_manifest),
                    "source_manifest_sha256": manifest_sha,
                }
            }
        )
        + "\n"
    )

    result = adjudicator.adjudicate(
        argparse.Namespace(
            contest_json=str(contest_json),
            provenance=str(tmp_path / "adjudication_provenance.json"),
            archive=str(archive),
            result_copy=str(tmp_path / "contest_auth_eval.adjudicated.json"),
            baseline_score=0.6,
            baseline_archive_bytes=None,
            predicted_band=[0.4, 0.6],
            hard_kill_above=None,
            regression_threshold=0.7,
            delta_key="score_delta_vs_baseline",
            required_device="cuda",
            required_samples=600,
            max_sane_score=10.0,
            max_posenet_dist=None,
            max_segnet_dist=None,
            baseline_posenet_dist=None,
            baseline_segnet_dist=None,
            max_posenet_relative=None,
            max_segnet_relative=None,
            component_reference_label="baseline",
        )
    )

    prov = json.loads((tmp_path / "adjudication_provenance.json").read_text())
    assert result["promotion_eligible"] is False
    assert result["source_manifest_runtime_closure_gate_triggered"] is True
    assert prov["source_manifest_runtime_closure_gate_triggered"] is True
    assert "SOURCE_MANIFEST_CLOSURE_REVIEW_REQUIRED" in prov["lane_status"]


def test_adjudicator_regression_threshold_is_delta_vs_baseline(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"frontier archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.32,
        "score_recomputed_from_components": 0.3247176275031171,
        "avg_posenet_dist": 0.00052391,
        "avg_segnet_dist": 0.00061261,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
        },
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(tmp_path / "provenance.json"),
        archive=str(archive),
        result_copy=None,
        baseline_score=0.32518843312932477,
        baseline_archive_bytes=287573,
        predicted_band=[0.324, 0.326],
        regression_threshold=0.01,
        delta_key="score_delta_vs_public_pr63",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=None,
        baseline_segnet_dist=None,
        max_posenet_relative=None,
        max_segnet_relative=None,
        component_reference_label="public_pr63_qpose14",
    )

    result = adjudicator.adjudicate(args)

    assert result["lane_status"] == "IN_PREDICTED_BAND"
    assert result["regression_triggered"] is False
    assert result["promotion_eligible"] is True
    prov = json.loads(Path(args.provenance).read_text())
    assert prov["regression_threshold_mode"] == "delta_vs_baseline"
    assert prov["score_delta_vs_public_pr63"] == pytest.approx(-0.00047080562620765987)


def test_adjudicator_writes_forensic_artifacts_on_component_gate_violation(
    tmp_path: Path,
) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.0030,
        "avg_segnet_dist": 0.0041,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    provenance = tmp_path / "adjudication_provenance.json"
    result_copy = tmp_path / "contest_auth_eval.adjudicated.json"
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=str(result_copy),
        baseline_score=1.05,
        baseline_archive_bytes=686635,
        predicted_band=[0.40, 0.60],
        regression_threshold=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=None,
        baseline_segnet_dist=0.00400656,
        max_posenet_relative=None,
        max_segnet_relative=1.002,
        component_reference_label="pfp16_a_plus_plus_t4",
    )

    result = adjudicator.adjudicate(args)

    assert result["component_gate_triggered"] is True
    assert result["lane_status"] == "COMPONENT_GATE_REVIEW_REQUIRED"
    assert result_copy.is_file()
    assert provenance.is_file()
    prov = json.loads(provenance.read_text())
    assert prov["component_gate_triggered"] is True
    assert prov["lane_status"] == "COMPONENT_GATE_REVIEW_REQUIRED"
    assert prov["promotion_eligible"] is False
    assert prov["paper_claim_grade"] == "A-negative scoped forensic"
    assert prov["allowed_use"] == ["forensic", "no_rank_frontier", "no_promotion"]
    assert prov["component_gate_violations"][0]["reason"] == "relative_component_gate"
    copied = json.loads(result_copy.read_text())
    assert copied["promotion_eligible"] is False
    assert copied["score_claim_valid"] is False
    assert copied["score_claim"] is False
    assert copied["evidence_grade"] == "A-negative scoped forensic"
    assert copied["paper_claim_grade"] == "A-negative scoped forensic"
    assert copied["allowed_use"] == ["forensic", "no_rank_frontier", "no_promotion"]
    assert copied["lane_status"] == "COMPONENT_GATE_REVIEW_REQUIRED"
    assert copied["component_gate_triggered"] is True
    assert copied["adjudication"]["promotion_eligible"] is False


def test_adjudicator_writes_forensic_artifacts_on_sane_score_gate_violation(
    tmp_path: Path,
) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 12.5,
        "score_recomputed_from_components": 12.5,
        "avg_posenet_dist": 10.0,
        "avg_segnet_dist": 0.05,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    provenance = tmp_path / "adjudication_provenance.json"
    result_copy = tmp_path / "contest_auth_eval.adjudicated.json"
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=str(result_copy),
        baseline_score=0.315,
        baseline_archive_bytes=276214,
        predicted_band=[0.2, 2.5],
        regression_threshold=0.2,
        delta_key="score_delta_vs_frontier",
        required_device="cuda",
        required_samples=600,
        max_sane_score=5.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=None,
        baseline_segnet_dist=None,
        max_posenet_relative=None,
        max_segnet_relative=None,
        component_reference_label="frontier",
    )

    result = adjudicator.adjudicate(args)

    assert result["sane_score_gate_triggered"] is True
    assert result["promotion_eligible"] is False
    assert result["paper_claim_grade"] == "A-negative scoped forensic"
    assert result_copy.is_file()
    assert provenance.is_file()
    prov = json.loads(provenance.read_text())
    assert prov["sane_score_gate_triggered"] is True
    assert prov["sane_score_gate_violation"]["reason"] == "sane_score_gate"
    assert prov["lane_status"] == "REGRESSION_AND_SANE_SCORE_REVIEW_REQUIRED"
    copied = json.loads(result_copy.read_text())
    assert copied["promotion_eligible"] is False
    assert copied["score_claim_valid"] is False
    assert copied["evidence_grade"] == "A-negative scoped forensic"
    assert copied["lane_status"] == "REGRESSION_AND_SANE_SCORE_REVIEW_REQUIRED"
    assert copied["adjudication"]["sane_score_gate_triggered"] is True


def test_adjudicator_result_copy_demotes_regressions(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.337,
        "score_recomputed_from_components": 0.3371617511972341,
        "avg_posenet_dist": 0.0005,
        "avg_segnet_dist": 0.0006,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "promotion_eligible": True,
        "score_claim_valid": True,
        "evidence_grade": "A++",
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    provenance = tmp_path / "adjudication_provenance.json"
    result_copy = tmp_path / "contest_auth_eval.adjudicated.json"
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=str(result_copy),
        baseline_score=0.204543,
        baseline_archive_bytes=186131,
        predicted_band=[0.20, 0.27],
        regression_threshold=0.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=1.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=None,
        baseline_segnet_dist=None,
        max_posenet_relative=None,
        max_segnet_relative=None,
        component_reference_label="frontier",
    )

    result = adjudicator.adjudicate(args)

    assert result["regression_triggered"] is True
    assert result["promotion_eligible"] is False
    prov = json.loads(provenance.read_text())
    copied = json.loads(result_copy.read_text())
    assert prov["paper_claim_grade"] == "A-negative scoped forensic"
    assert copied["promotion_eligible"] is False
    assert copied["score_claim_valid"] is False
    assert copied["score_claim"] is False
    assert copied["rank_or_kill_eligible"] is False
    assert copied["evidence_grade"] == "A-negative scoped forensic"
    assert copied["paper_claim_grade"] == "A-negative scoped forensic"
    assert copied["lane_status"] == "REGRESSION_REVIEW_REQUIRED"
    assert copied["regression_triggered"] is True
    assert copied["allowed_use"] == ["forensic", "no_rank_frontier", "no_promotion"]
    assert copied["adjudication"]["promotion_eligible"] is False


def test_adjudicator_cli_component_gate_fails_closed_by_default(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.0030,
        "avg_segnet_dist": 0.0041,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "adjudicate_contest_auth_eval.py"),
            "--contest-json",
            str(contest_json),
            "--provenance",
            str(tmp_path / "adjudication_provenance.json"),
            "--archive",
            str(archive),
            "--result-copy",
            str(tmp_path / "contest_auth_eval.adjudicated.json"),
            "--baseline-score",
            "1.05",
            "--predicted-band",
            "0.40",
            "0.60",
            "--regression-threshold",
            "1.05",
            "--baseline-segnet-dist",
            "0.00400656",
            "--max-segnet-relative",
            "1.002",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 2
    assert "FATAL: component gate violation" in result.stdout
    assert (tmp_path / "adjudication_provenance.json").is_file()
    assert (tmp_path / "contest_auth_eval.adjudicated.json").is_file()


def test_adjudicator_cli_can_mark_component_gate_as_forensic_success(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.0030,
        "avg_segnet_dist": 0.0041,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "adjudicate_contest_auth_eval.py"),
            "--contest-json",
            str(contest_json),
            "--provenance",
            str(tmp_path / "adjudication_provenance.json"),
            "--archive",
            str(archive),
            "--result-copy",
            str(tmp_path / "contest_auth_eval.adjudicated.json"),
            "--baseline-score",
            "1.05",
            "--predicted-band",
            "0.40",
            "0.60",
            "--regression-threshold",
            "1.05",
            "--baseline-segnet-dist",
            "0.00400656",
            "--max-segnet-relative",
            "1.002",
            "--allow-component-gate-forensic-success",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "COMPONENT_GATE_FORENSIC_SUCCESS=1" in result.stdout
    assert "PROMOTION_ELIGIBLE=0" in result.stdout
    assert "PAPER_CLAIM_GRADE=A-negative scoped forensic" in result.stdout
    provenance = json.loads((tmp_path / "adjudication_provenance.json").read_text())
    assert provenance["component_gate_triggered"] is True
    assert provenance["lane_status"] == "COMPONENT_GATE_REVIEW_REQUIRED"
    assert provenance["promotion_eligible"] is False
    assert provenance["allowed_use"] == ["forensic", "no_rank_frontier", "no_promotion"]


def test_adjudicator_rejects_non_cuda_evidence(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 1.04,
        "score_recomputed_from_components": 1.044,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cpu",
            "gpu_t4_match": False,
        },
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(tmp_path / "provenance.json"),
        archive=str(archive),
        result_copy=None,
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[1.04, 1.05],
        hard_kill_above=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
    )
    with pytest.raises(SystemExit, match="expected 'cuda'"):
        adjudicator.adjudicate(args)


@pytest.mark.parametrize(
    ("gate_arg", "match"),
    [
        ("max_posenet_dist", "avg_posenet_dist"),
        ("max_segnet_dist", "avg_segnet_dist"),
    ],
)
def test_adjudicator_rejects_high_components_even_when_score_parses(
    tmp_path: Path,
    gate_arg: str,
    match: str,
) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.25,
        "avg_segnet_dist": 0.20,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    gate_kwargs = {
        "max_posenet_dist": None,
        "max_segnet_dist": None,
        "baseline_posenet_dist": None,
        "baseline_segnet_dist": None,
        "max_posenet_relative": None,
        "max_segnet_relative": None,
        "component_reference_label": "frontier",
    }
    gate_kwargs[gate_arg] = 0.01
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(tmp_path / "provenance.json"),
        archive=str(archive),
        result_copy=str(tmp_path / "result.json"),
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[0.40, 0.60],
        regression_threshold=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        **gate_kwargs,
    )

    result = adjudicator.adjudicate(args)
    assert result["component_gate_triggered"] is True
    assert any(gate["metric"] == match for gate in result["component_gates"])
    assert (tmp_path / "result.json").exists()


def test_adjudicator_rejects_relative_component_gate_violation(tmp_path: Path) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.006,
        "avg_segnet_dist": 0.009,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(tmp_path / "provenance.json"),
        archive=str(archive),
        result_copy=None,
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[0.40, 0.60],
        regression_threshold=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=0.003,
        baseline_segnet_dist=None,
        max_posenet_relative=1.5,
        max_segnet_relative=None,
        component_reference_label="frontier",
    )

    result = adjudicator.adjudicate(args)
    assert result["component_gate_triggered"] is True
    assert result["component_gate_violations"][0]["reason"] == "relative_component_gate"


def test_adjudicator_blocks_distillation_active_promotion_without_policy(
    tmp_path: Path,
) -> None:
    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.0030,
        "avg_segnet_dist": 0.0040,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "lane_script": "scripts/remote_lane_g_v3_corrected_kl_weight.sh",
        "kl_distill_weight": 0.002,
        "kl_distill_temperature": 2.0,
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=None,
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[0.40, 0.60],
        regression_threshold=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=0.003,
        baseline_segnet_dist=0.004,
        max_posenet_relative=1.01,
        max_segnet_relative=1.01,
        component_reference_label="frontier",
    )

    result = adjudicator.adjudicate(args)

    assert result["distillation_policy_active"] is True
    assert result["distillation_policy_gate_triggered"] is True
    assert result["promotion_eligible"] is False
    assert result["lane_status"] == "DISTILLATION_POLICY_REVIEW_REQUIRED"
    reasons = {v["reason"] for v in result["distillation_policy_gate_violations"]}
    assert "missing_distillation_policy_v1" in reasons
    assert "missing_distillation_policy_sha256" in reasons
    prov = json.loads(provenance.read_text())
    assert prov["contest_cuda_archive_sha256"] == archive_sha
    assert prov["contest_cuda_archive_bytes"] == archive.stat().st_size
    assert prov["contest_cuda_device"] == "cuda"
    assert prov["distillation_policy_gate_triggered"] is True
    assert prov["allowed_use"] == ["forensic", "no_rank_frontier", "no_promotion"]


def test_adjudicator_allows_distillation_active_promotion_with_policy_sha_and_gates(
    tmp_path: Path,
) -> None:
    from tac.kl_config import distillation_policy_sha256, normalize_distillation_policy

    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.0030,
        "avg_segnet_dist": 0.0040,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    policy_obj = normalize_distillation_policy({
        "family": "segnet_aux_kl",
        "weight": 0.002,
        "temperature": 2.0,
        "promotion_eligible": True,
        "eval_roundtrip": True,
    })
    policy = policy_obj.to_provenance()
    policy_sha = distillation_policy_sha256(policy_obj)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "lane_script": "scripts/remote_lane_g_v3_corrected_kl_weight.sh",
        "kl_distill_weight": 0.002,
        "kl_distill_temperature": 2.0,
        "distillation_policy": policy,
        "distillation_policy_sha256": policy_sha,
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=None,
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[0.40, 0.60],
        regression_threshold=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=0.003,
        baseline_segnet_dist=0.004,
        max_posenet_relative=1.01,
        max_segnet_relative=1.01,
        component_reference_label="frontier",
    )

    result = adjudicator.adjudicate(args)

    assert result["distillation_policy_active"] is True
    assert result["distillation_policy_gate_triggered"] is False
    assert result["distillation_policy_sha256"] == policy_sha
    assert result["promotion_eligible"] is True
    assert {gate["component"] for gate in result["component_gates"]} == {"posenet", "segnet"}


def test_adjudicator_rejects_distillation_policy_sha_mismatch(tmp_path: Path) -> None:
    from tac.kl_config import normalize_distillation_policy

    adjudicator = _load_adjudicator()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(json.dumps({
        "final_score": 0.50,
        "score_recomputed_from_components": 0.50,
        "avg_posenet_dist": 0.0030,
        "avg_segnet_dist": 0.0040,
        "archive_size_bytes": archive.stat().st_size,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }))
    policy = normalize_distillation_policy({
        "family": "segnet_aux_kl",
        "weight": 0.002,
        "temperature": 2.0,
        "promotion_eligible": True,
    }).to_provenance()
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "kl_distill_weight": 0.002,
        "distillation_policy": policy,
        "distillation_policy_sha256": "0" * 64,
    }))
    args = argparse.Namespace(
        contest_json=str(contest_json),
        provenance=str(provenance),
        archive=str(archive),
        result_copy=None,
        baseline_score=1.05,
        baseline_archive_bytes=None,
        predicted_band=[0.40, 0.60],
        regression_threshold=1.05,
        delta_key="score_delta_vs_baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=0.003,
        baseline_segnet_dist=0.004,
        max_posenet_relative=1.01,
        max_segnet_relative=1.01,
        component_reference_label="frontier",
    )

    result = adjudicator.adjudicate(args)

    assert result["distillation_policy_gate_triggered"] is True
    reasons = {v["reason"] for v in result["distillation_policy_gate_violations"]}
    assert "distillation_policy_sha256_mismatch" in reasons


def test_preflight_catches_human_score_regex(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "python experiments/contest_auth_eval.py 2>&1 | tee auth_eval.log\n"
        "python - <<'PY'\n"
        "import re\n"
        "m = re.search(r'final[_ ]?score[\\s:=]+([0-9.]+)', open('auth_eval.log').read())\n"
        "PY\n"
    )
    violations = _scan_remote_lane_auth_eval_fragile_parse(bad, root)
    assert violations
    assert any("final[_ ]?score" in v for v in violations)


def test_preflight_catches_last_json_object_scrape(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "python experiments/contest_auth_eval.py 2>&1 | tee auth_eval.log\n"
        "grep -Eo '\\{.*\\}' auth_eval.log | tail -1 > RESULT_JSON\n"
    )
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(violations) == 1
    with pytest.raises(MetaBugViolation):
        check_remote_lane_auth_eval_json_adjudication(
            repo_root=root, strict=True, verbose=False,
        )


def test_preflight_catches_contest_cuda_without_kept_work_dir(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "log(){ echo \"$*\"; }\n"
        "python experiments/contest_auth_eval.py --device cuda 2>&1 | tee auth_eval.log\n"
        "log 'LANE_DONE [contest-CUDA]'\n"
    )
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=root, strict=False, verbose=False,
    )
    assert any("--keep-work-dir" in v and "--work-dir" in v for v in violations)


def test_preflight_catches_auth_eval_device_under_contest_cuda(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "LOG_DIR=logs\n"
        "python experiments/contest_auth_eval.py \\\n"
        "  --device \"${AUTH_EVAL_DEVICE:-cuda}\" \\\n"
        "  --keep-work-dir \\\n"
        "  --work-dir \"$LOG_DIR/eval_work\" 2>&1 | tee \"$LOG_DIR/auth_eval.log\"\n"
        "echo 'LANE_DONE [contest-CUDA]'\n"
    )
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=root, strict=False, verbose=False,
    )
    assert any("AUTH_EVAL_DEVICE" in v for v in violations)


def test_preflight_allows_guarded_auth_eval_device_advisory_escape(tmp_path: Path) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    good = scripts / "remote_lane_guarded.sh"
    good.write_text(
        "#!/bin/bash\n"
        "LOG_DIR=logs\n"
        "AUTH_DEVICE=\"${AUTH_EVAL_DEVICE:-cuda}\"\n"
        "if [ \"$AUTH_DEVICE\" != \"cuda\" ] && [ \"${ALLOW_NON_CUDA_EVAL:-0}\" != \"1\" ]; then exit 2; fi\n"
        "python experiments/contest_auth_eval.py \\\n"
        "  --device \"$AUTH_DEVICE\" \\\n"
        "  --keep-work-dir \\\n"
        "  --work-dir \"$LOG_DIR/eval_work\" 2>&1 | tee \"$LOG_DIR/auth_eval.log\"\n"
        "test -f \"$LOG_DIR/eval_work/contest_auth_eval.json\"\n"
        "echo 'LANE_DONE [contest-CUDA]'\n"
    )
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=root, strict=False, verbose=False,
    )
    assert violations == []


def test_preflight_catches_distillation_promotion_path_without_policy_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    bad = scripts / "remote_lane_bad_kl.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "PROVENANCE=provenance.json\n"
        "python experiments/contest_auth_eval.py --device cuda --keep-work-dir --work-dir eval_work\n"
        "python scripts/adjudicate_contest_auth_eval.py \\\n"
        "  --contest-json eval_work/contest_auth_eval.json \\\n"
        "  --provenance \"$PROVENANCE\" \\\n"
        "  --archive archive.zip \\\n"
        "  --baseline-score 1.05 \\\n"
        "  --predicted-band 0.9 1.1 \\\n"
        "  --regression-threshold 1.2\n"
        "echo \"PROMOTION_ELIGIBLE=$PROMOTION_ELIGIBLE\"\n"
        "python - <<'PY'\n"
        "prov = {'kl_distill_weight': 0.002}\n"
        "PY\n"
    )
    violations = _scan_remote_distillation_promotion_provenance(bad, root)
    assert any("distillation_policy" in v for v in violations)
    assert any("--baseline-posenet-dist" in v for v in violations)
    with pytest.raises(MetaBugViolation):
        check_remote_distillation_promotion_provenance(
            repo_root=root, strict=True, verbose=False,
        )


def test_preflight_allows_distillation_promotion_path_with_policy_sha_and_gates(
    tmp_path: Path,
) -> None:
    root = tmp_path
    scripts = root / "scripts"
    scripts.mkdir()
    good = scripts / "remote_lane_good_kl.sh"
    good.write_text(
        "#!/bin/bash\n"
        "PROVENANCE=provenance.json\n"
        "python - <<'PY'\n"
        "prov = {\n"
        "  'kl_distill_weight': 0.002,\n"
        "  'distillation_policy': {'format': 'distillation_policy_v1'},\n"
        "  'distillation_policy_sha256': 'abc',\n"
        "}\n"
        "PY\n"
        "python experiments/contest_auth_eval.py \\\n"
        "  --archive archive.zip \\\n"
        "  --inflate-sh submissions/robust_current/inflate.sh \\\n"
        "  --upstream-dir upstream \\\n"
        "  --device cuda \\\n"
        "  --keep-work-dir \\\n"
        "  --work-dir eval_work\n"
        "python scripts/adjudicate_contest_auth_eval.py \\\n"
        "  --contest-json eval_work/contest_auth_eval.json \\\n"
        "  --provenance \"$PROVENANCE\" \\\n"
        "  --archive archive.zip \\\n"
        "  --baseline-score 1.05 \\\n"
        "  --predicted-band 0.9 1.1 \\\n"
        "  --regression-threshold 1.2 \\\n"
        "  --baseline-posenet-dist 0.003 \\\n"
        "  --baseline-segnet-dist 0.004 \\\n"
        "  --max-posenet-relative 1.01 \\\n"
        "  --max-segnet-relative 1.01\n"
        "echo \"ARCHIVE_SHA256=$ARCHIVE_SHA256 ARCHIVE_BYTES=$ARCHIVE_BYTES\"\n"
        "echo \"PROMOTION_ELIGIBLE=$PROMOTION_ELIGIBLE\"\n"
    )

    assert check_remote_distillation_promotion_provenance(
        repo_root=root, strict=True, verbose=False,
    ) == []


def test_live_remote_distillation_promotion_preflight_passes_current_scripts() -> None:
    violations = check_remote_distillation_promotion_provenance(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_live_remote_lane_scripts_avoid_fragile_auth_eval_parsers() -> None:
    violations = check_remote_lane_auth_eval_json_adjudication(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_launch_retry_wrapper_preflight_self_protection() -> None:
    violations = check_launch_retry_wrapper_singleflight_and_signal_safe(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_modal_recovery_cli_guidance_current() -> None:
    violations = check_modal_recovery_cli_guidance_current(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_modal_cpu_auth_eval_preflight_live_wrapper_advisory_only() -> None:
    violations = check_modal_cpu_auth_eval_is_advisory_only(
        repo_root=REPO_ROOT, strict=False, verbose=False,
    )
    assert violations == []


def test_modal_cpu_auth_eval_preflight_catches_stale_score_truth_wording(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "modal_train_lane.py").write_text(
        "\"\"\"Output: auto-extracted auth score.\"\"\"\n"
        "env = {\n"
        "    \"AUTH_EVAL_DEVICE\": \"cpu\",\n"
        "}\n"
        "# uses --device cpu, which produces identical scores\n"
    )
    (experiments / "modal_recover_lane.py").write_text(
        "def recover_one():\n"
        "    print('=== AUTH SCORE: 1.04 ===')\n"
    )

    violations = check_modal_cpu_auth_eval_is_advisory_only(
        repo_root=tmp_path, strict=False, verbose=False,
    )

    assert any("identical scores" in v for v in violations)
    assert any("ADVISORY AUTH SCORE" in v for v in violations)
    with pytest.raises(MetaBugViolation, match="MODAL CPU AUTH-EVAL"):
        check_modal_cpu_auth_eval_is_advisory_only(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_modal_result_dir_label_parser_preserves_lane_prefix() -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert (
        mod.label_from_modal_result_dir(
            "lane_lane_g_v3_owv3_fisher_smoke_20260430_codex_modal"  # FAKE_LANE_OK:test fixture for Modal label parsing
        )
        == "lane_g_v3_owv3_fisher_smoke_20260430_codex"  # FAKE_LANE_OK:test fixture for Modal label parsing
    )
    assert (
        mod.label_from_modal_result_dir(
            "lane_uniward_v8_modal"  # FAKE_LANE_OK:test fixture for Modal label parsing
        )
        == "uniward_v8"
    )


def test_modal_recover_direct_call_id_derives_original_dispatch_label_from_log() -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    label = "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch_20260514T012029Z__smoke__100ep"
    result = {
        "artifacts": {
            f"modal_lane_{label}.log": b"log",
            "lane_pr95plus_results/provenance.json": json.dumps(
                {"dispatch_instance_job_id": label}
            ).encode("utf-8"),
        }
    }

    assert mod.label_from_modal_result(result) == label


def test_modal_recover_direct_call_id_falls_back_to_provenance_label() -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    label = "lane_preserve_full_label_with_underscores"
    result = {
        "artifacts": {
            "lane_pr95plus_results/provenance.json": json.dumps(
                {"dispatch_instance_job_id": label}
            ).encode("utf-8"),
        }
    }

    assert mod.label_from_modal_result(result) == label


def test_modal_recover_closes_terminal_claim_on_artifact_materialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    terminal_calls: list[dict[str, object]] = []

    def fake_terminal_claim(**kwargs):
        terminal_calls.append(kwargs)
        return {"appended": True, "status": kwargs.get("status")}

    monkeypatch.setattr(mod, "_append_terminal_claim_if_requested", fake_terminal_claim)

    class OutputExpiredError(Exception):
        pass

    class FunctionCall:
        @classmethod
        def from_id(cls, call_id: str):
            assert call_id == "fc-test"
            return cls()

        def get(self, timeout: float = 0):
            return {
                "returncode": 0,
                "artifacts": {
                    ".": b"not a file",
                    "run.log": b"would otherwise be valid",
                },
            }

    fake_modal = types.SimpleNamespace(
        FunctionCall=FunctionCall,
        exception=types.SimpleNamespace(OutputExpiredError=OutputExpiredError),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    rc = mod.recover_one("demo", "fc-test")

    out_dir = tmp_path / "experiments" / "results" / "lane_demo_modal"
    summary = json.loads((out_dir / "modal_recover_summary.json").read_text())
    assert rc == 5
    assert summary["status"] == "invalid_artifacts"
    assert summary["score_claim"] is False
    assert terminal_calls
    assert terminal_calls[0]["status"] == "failed_modal_training_invalid_artifacts"


def test_modal_recover_terminal_ledger_metadata_uses_recovered_lane_id(
    tmp_path: Path,
) -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_dir = tmp_path / "lane_job_modal"
    out_dir.mkdir()
    (out_dir / "modal_metadata.json").write_text(
        json.dumps(
            {
                "call_id": "fc-recovered",
                "lane_id": "lane_dp1_canonical",
                "label": "dp1_timestamped_label",
            }
        ),
        encoding="utf-8",
    )

    metadata = mod.modal_terminal_ledger_metadata(
        out_dir=out_dir,
        call_id="fc-fallback",
        label="timestamped_label",
    )

    assert metadata["call_id"] == "fc-recovered"
    assert metadata["lane_id"] == "lane_dp1_canonical"
    assert metadata["label"] == "dp1_timestamped_label"
    assert metadata["platform"] == "modal"


def test_modal_recover_terminal_ledger_metadata_falls_back_without_metadata(
    tmp_path: Path,
) -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    metadata = mod.modal_terminal_ledger_metadata(
        out_dir=tmp_path,
        call_id="fc-fallback",
        label="timestamped_label",
    )

    assert metadata["call_id"] == "fc-fallback"
    assert metadata["lane_id"] == "timestamped_label"
    assert metadata["label"] == "timestamped_label"


def test_modal_recover_labels_non_cuda_scores_advisory() -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lines = mod.auth_score_summary_lines(
        {
            "score_recomputed_from_components": 1.2345,
            "final_score": 1.23,
            "avg_posenet_dist": 0.004,
            "avg_segnet_dist": 0.005,
            "rate": 0.01,
            "provenance": {"device": "cpu"},
        },
        label="lane_demo",
        source="eval_work/contest_auth_eval.json",
    )

    assert lines is not None
    text = "\n".join(lines)
    assert "ADVISORY AUTH SCORE" in text
    assert "NON-PROMOTABLE, device=cpu" in text
    assert "1.2345" in text
    assert "--device cuda" in text


def test_modal_recover_labels_cuda_scores_as_unadjudicated_non_promotable() -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lines = mod.auth_score_summary_lines(
        {
            "score_recomputed_from_components": 1.044048,
            "provenance": {"device": "cuda"},
        },
        label="lane_demo",
        source="eval_work/contest_auth_eval.json",
    )

    assert lines is not None
    text = "\n".join(lines)
    assert "CUDA AUTH SCORE" in text
    assert "UNADJUDICATED, NON-PROMOTABLE" in text
    assert "Adjudication required" in text
    assert "ADVISORY AUTH SCORE" not in text


def test_modal_recover_running_guidance_uses_supported_commands(capsys) -> None:
    path = REPO_ROOT / "experiments" / "modal_recover_lane.py"
    spec = importlib.util.spec_from_file_location("_modal_recover_lane", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.print_still_running_guidance("fc-test", label="lane_demo")
    out = capsys.readouterr().out

    assert "modal call get" not in out
    assert ".venv/bin/python experiments/modal_recover_lane.py --label lane_demo" in out
    assert ".venv/bin/modal app list" in out
    assert ".venv/bin/modal app logs <app-id>" in out


def test_launch_retry_timeouts_cover_launcher_poll_windows(monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    assert launcher.PHASE2_WAIT_TIMEOUT_SECONDS >= 480
    assert launcher.PHASE2_SCP_TIMEOUT_SECONDS >= 600
    assert launcher.PHASE2_EXTRACT_TIMEOUT_SECONDS >= 120
    assert launcher.PHASE2_LAUNCH_TIMEOUT_SECONDS > 240

    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_run_stage(cmd, timeout=300):
        calls.append((tuple(cmd), timeout))
        stage = cmd[2]
        if stage == "phase1":
            return 0, "INSTANCE_ID=123\n"
        if stage in {"phase2-wait", "phase2-scp", "phase2-extract"}:
            return 0, "ok\n"
        if stage == "phase2-launch":
            return 124, "TIMEOUT after 420s"
        raise AssertionError(cmd)

    destroyed: list[int] = []
    monkeypatch.setattr(launcher, "run_stage", fake_run_stage)
    monkeypatch.setattr(launcher, "destroy", destroyed.append)
    args = types.SimpleNamespace(
        lane_script="scripts/remote_lane_pfp16_stack.sh",
        label="lane_pfp16",
        max_dph=0.40,
        predicted_band=[1.04, 1.05],
        estimated_cost=0.50,
        allow_existing_label_prefix=True,
    )

    status, iid, log = launcher.attempt_dispatch(args, attempt=1)

    assert status == "unknown"
    assert iid == 123
    assert destroyed == []
    assert "UNKNOWN_REMOTE_STATE" in log
    assert calls[-1][1] == launcher.PHASE2_LAUNCH_TIMEOUT_SECONDS


def test_launch_retry_refuses_duplicate_live_label_prefix(monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    monkeypatch.setattr(
        launcher,
        "live_instances_with_label_prefix",
        lambda _label: ([{
            "id": 35905118,
            "label": "lane_sa_segmap_clone_2026-04-30_codex_a3",
            "actual_status": "running",
            "ssh_host": "ssh6.vast.ai",
            "ssh_port": 25118,
            "dph_total": 0.253,
        }], None),
    )
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(launcher, "run_stage", lambda cmd, timeout=300: calls.append(tuple(cmd)) or (0, "INSTANCE_ID=1\n"))
    args = types.SimpleNamespace(
        lane_script="scripts/remote_lane_sa_segmap_clone.sh",
        label="lane_sa_segmap_clone_2026-04-30_codex",
        max_dph=0.40,
        predicted_band=[0.40, 0.55],
        estimated_cost=1.00,
        allow_existing_label_prefix=False,
    )

    status, iid, log = launcher.attempt_dispatch(args, attempt=1)

    assert status == "unknown"
    assert iid == 35905118
    assert calls == []
    assert "UNKNOWN_EXISTING_LABEL_PREFIX" in log
    assert "No duplicate retry launched" in log


def test_lane12_retraining_gate_blocks_non_lane12_training_script(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    script = tmp_path / "scripts" / "remote_lane_new_training.sh"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env bash\n"
        ".venv/bin/python experiments/train_renderer.py --profile smoke\n"
    )

    violations = launcher.lane12_retraining_gate_violations(
        lane_script=script,
        label="lane_new_training_2026-04-30",
        repo_root=tmp_path,
        clearance_path=tmp_path / "missing_lane12_clearance.json",
    )

    assert violations
    assert "blocked until Lane 12/Alpha" in violations[0]
    assert "missing Lane 12 L2 clearance packet" in violations[1]


def test_lane12_retraining_gate_allows_build_only_non_training_script(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    script = tmp_path / "scripts" / "remote_lane_pfp16_stack.sh"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env bash\n"
        ".venv/bin/python experiments/build_lane_g_v3_pfp16_stack.py\n"
        ".venv/bin/python experiments/contest_auth_eval.py --device cuda\n"
    )

    assert launcher.lane12_retraining_gate_violations(
        lane_script=script,
        label="lane_pfp16_stack_2026-04-30",
        repo_root=tmp_path,
        clearance_path=tmp_path / "missing_lane12_clearance.json",
    ) == []


def test_lane12_retraining_gate_allows_lane12_itself(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    script = tmp_path / "scripts" / "remote_lane_nerv.sh"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env bash\n"
        ".venv/bin/python experiments/train_nerv_mask.py --device cuda\n"
    )

    assert launcher.lane12_retraining_gate_violations(
        lane_script=script,
        label="lane_12_nerv_2026-04-30_r1",
        repo_root=tmp_path,
        clearance_path=tmp_path / "missing_lane12_clearance.json",
    ) == []


def test_lane12_retraining_gate_accepts_explicit_clearance_packet(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    script = tmp_path / "scripts" / "remote_lane_new_training.sh"
    clearance = tmp_path / "lane12_nerv_l2_clearance.json"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env bash\n"
        ".venv/bin/python experiments/train_renderer.py --profile smoke\n"
    )
    clearance.write_text(json.dumps({
        "lane_id": "lane_12_nerv_mask_codec",
        "cleared_for_retraining_unblock": True,
        "lane12_l2": True,
        "geometry_gate_passed": True,
        "grand_council_clean_passes": 3,
        "evidence": ".omx/research/lane12_l2_packet.md",
    }))

    assert launcher.lane12_retraining_gate_violations(
        lane_script=script,
        label="lane_new_training_2026-04-30",
        repo_root=tmp_path,
        clearance_path=clearance,
    ) == []


def test_launch_retry_refuses_timestamped_logical_duplicate(monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    assert (
        launcher.logical_lane_key("lane_19_logit_margin_2026-04-30_q1_20260430T211406Z")
        == "lane_19_logit_margin_2026-04-30"
    )
    assert (
        launcher.logical_lane_key("lane_19_logit_margin_2026-04-30_q1c_20260430T211553Z_a1")
        == "lane_19_logit_margin_2026-04-30"
    )

    monkeypatch.setattr(
        launcher,
        "live_instances_with_label_prefix",
        lambda _label: ([{
            "id": 35925374,
            "label": "lane_19_logit_margin_2026-04-30_q1_20260430T211406Z_a1",
            "actual_status": "running",
            "ssh_host": "ssh8.vast.ai",
            "ssh_port": 15374,
            "dph_total": 0.206,
        }], None),
    )
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        launcher,
        "run_stage",
        lambda cmd, timeout=300: calls.append(tuple(cmd)) or (0, "INSTANCE_ID=1\n"),
    )
    args = types.SimpleNamespace(
        lane_script="scripts/remote_lane_19_logit_margin.sh",
        label="lane_19_logit_margin_2026-04-30_q1c_20260430T211553Z",
        max_dph=0.40,
        predicted_band=[0.75, 1.05],
        estimated_cost=1.50,
        allow_existing_label_prefix=False,
    )

    status, iid, log = launcher.attempt_dispatch(args, attempt=1)

    assert status == "unknown"
    assert iid == 35925374
    assert calls == []
    assert "UNKNOWN_EXISTING_LABEL_PREFIX" in log


def test_live_instance_guard_matches_timestamped_logical_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    class Proc:
        returncode = 0
        stderr = ""
        stdout = json.dumps([
            {
                "id": 35925374,
                "label": "lane_19_logit_margin_2026-04-30_q1_20260430T211406Z_a1",
                "actual_status": "running",
                "ssh_host": "ssh8.vast.ai",
                "ssh_port": 15374,
                "dph_total": 0.206,
            }
        ])

    fake_vastai = tmp_path / "vastai"
    fake_vastai.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "VASTAI", fake_vastai)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: Proc())

    matches, error = launcher.live_instances_with_label_prefix(
        "lane_19_logit_margin_2026-04-30_q1c_20260430T211553Z"
    )

    assert error is None
    assert [m["id"] for m in matches] == [35925374]


def test_dispatch_hold_blocks_quarantined_logical_label(tmp_path: Path, monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    holds = tmp_path / "dispatch_holds.json"
    holds.write_text(json.dumps({
        "holds": [
            {
                "logical_key": "lane_19_logit_margin_2026-04-30",
                "reason": "forensic hold pending deterministic archive/adjudication repair",
            }
        ]
    }))
    monkeypatch.setattr(launcher, "DISPATCH_HOLDS_PATH", holds)

    hold = launcher.dispatch_hold_for_label(
        "lane_19_logit_margin_2026-04-30_q1d_20260430T212704Z"
    )

    assert hold is not None
    assert "forensic hold" in hold["reason"]


def test_launch_retry_run_stage_timeout_kills_stage_group() -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    rc, out = launcher.run_stage(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=1,
    )

    assert rc == 124
    assert "TIMEOUT after 1s" in out


def test_launch_retry_signal_cleanup_tolerates_killpg_permission_error(monkeypatch) -> None:
    path = REPO_ROOT / "scripts" / "launch_lane_with_retry.py"
    spec = importlib.util.spec_from_file_location("_launch_lane_with_retry", path)
    assert spec and spec.loader
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    class Proc:
        pid = 12345

        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    proc = Proc()
    monkeypatch.setattr(launcher.os, "getpgid", lambda _pid: (_ for _ in ()).throw(PermissionError()))

    launcher._kill_process_group(proc)

    assert proc.terminated is True
