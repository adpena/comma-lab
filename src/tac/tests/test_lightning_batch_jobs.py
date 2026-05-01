from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tac.deploy.lightning.batch_jobs import (
    ARTIFACT_DALI_BOOTSTRAP,
    ARTIFACT_DALI_REQUIREMENTS,
    ARTIFACT_COMPONENT_RESPONSE_INPUTS,
    ARTIFACT_COMPONENT_RESPONSE_SUMMARY,
    ARTIFACT_COMPONENT_RESPONSE_VALIDATION,
    ARTIFACT_COMPONENT_SENSITIVITY_INPUTS,
    ARTIFACT_COMPONENT_SENSITIVITY_RUN,
    ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY,
    ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION,
    COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES,
    ARTIFACT_VALIDATION,
    CANONICAL_ARTIFACT_FILES,
    COMPONENT_SENSITIVITY_CURVE_FILES,
    COMPONENT_SENSITIVITY_MAP_FILES,
    COMPONENT_RESPONSE_CURVE_FILES,
    ARTIFACT_RUNNER_PREFLIGHT,
    ARTIFACT_SUPPLY_CHAIN_SCAN,
    ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
    LightningAdjudicationSpec,
    LightningBatchJobsClient,
    LightningBatchJobSpec,
    archive_identity,
    default_exact_eval_output_dir,
    diagnostic_component_sensitivity_command,
    exact_cuda_eval_command,
    lightning_sdk_artifact_path,
    lightning_sdk_job_name,
    lightning_sdk_persisted_studio_output_dir,
    make_diagnostic_component_sensitivity_spec,
    make_exact_eval_spec,
    make_official_component_response_spec,
    mirror_local_component_sensitivity_artifact_dir,
    mirror_local_artifact_dir,
    official_component_response_command,
    validate_local_component_sensitivity_artifact_dir,
    validate_local_component_response_artifact_dir,
    validate_local_artifact_dir,
)
import tac.deploy.lightning.batch_jobs as lightning_batch_jobs


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "scripts" / "launch_lightning_batch_job.py"
EXPECTED_SHA = "a" * 64
EXPECTED_BYTES = 123


def _load_lightning_cli_module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("launch_lightning_batch_job_under_test", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO_ROOT = tmp_path
    return module


def _expected_archive_kwargs(marker: str = "a", size: int = 123) -> dict[str, object]:
    return {
        "expected_archive_sha256": marker * 64,
        "expected_archive_size_bytes": size,
    }


def _adjudication() -> LightningAdjudicationSpec:
    return LightningAdjudicationSpec(
        baseline_score=1.2,
        predicted_band_low=1.0,
        predicted_band_high=1.4,
        regression_threshold=1.6,
        baseline_archive_size_bytes=100,
        max_posenet_dist=0.006,
        max_segnet_dist=0.004,
        baseline_posenet_dist=0.003,
        baseline_segnet_dist=0.002,
        max_posenet_relative=1.5,
        max_segnet_relative=1.2,
        component_reference_label="frontier",
    )


def _write_artifact_dir(root: Path, *, adjudication: bool = True) -> tuple[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
    archive.write_bytes(b"fake archive bytes")
    identity = archive_identity(archive)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "job_name": "artifact-job",
        "role": "exact_cuda_eval",
        "expected_archive_sha256": identity["archive_sha256"],
        "expected_archive_size_bytes": identity["archive_size_bytes"],
        "queue_metadata": {"lane": "pfp16"},
        "adjudication": None,
        "score_source": "contest_auth_eval.json:score_recomputed_from_components",
        "status_source": "lightning_sdk_job_attributes",
    }
    if adjudication:
        metadata["adjudication"] = {
            "provenance_name": "adjudication_provenance.json",
            "result_copy_name": "contest_auth_eval.adjudicated.json",
        }
    (root / "lightning_queue_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 1.25,
                "final_score": 1.25,
                "avg_posenet_dist": 0.003,
                "avg_segnet_dist": 0.004,
                "n_samples": 600,
                "archive_size_bytes": identity["archive_size_bytes"],
                "provenance": {
                    "device": "cuda",
                    "archive_sha256": identity["archive_sha256"],
                    "gpu_model": "T4",
                    "gpu_t4_match": True,
                },
            },
            indent=2,
        )
        + "\n"
    )
    (root / "auth_eval.log").write_text("deceptive human log score: 9.99\n")
    (root / "report.txt").write_text("report text is not parsed for score\n")
    (root / "eval_provenance.json").write_text(
        json.dumps({"device": "cuda", "archive_sha256": identity["archive_sha256"]}) + "\n"
    )
    (root / ARTIFACT_DALI_REQUIREMENTS).write_text(
        "https://pypi.nvidia.com/nvidia-dali-cuda130/nvidia_dali_cuda130-1.52.0-py3-none-manylinux_2_28_x86_64.whl "
        "--hash=sha256:37369fb30e9c66f710b29836688c90abc36793bbe757cd3ad699fac76ba07119\n"
    )
    (root / ARTIFACT_DALI_BOOTSTRAP).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_exact_eval_dali_bootstrap",
                "required_dali_version": "1.52.0",
                "bootstrap_action": "already_exact",
                "selected_package": "nvidia-dali-cuda130",
                "selected_requirement": "nvidia-dali-cuda130==1.52.0",
                "selected_wheels": [
                    {
                        "name": "nvidia-dali-cuda130",
                        "version": "1.52.0",
                        "url": "https://pypi.nvidia.com/nvidia-dali-cuda130/nvidia_dali_cuda130-1.52.0-py3-none-manylinux_2_28_x86_64.whl",
                        "sha256": "37369fb30e9c66f710b29836688c90abc36793bbe757cd3ad699fac76ba07119",
                    }
                ],
                "final_probe": {
                    "dali_version": "1.52.0",
                    "nvidia_dali_fn_module": "nvidia.dali.fn",
                    "installed_distributions": {
                        "nvidia-dali-cuda120": None,
                        "nvidia-dali-cuda130": "1.52.0",
                    },
                },
                "final_probe_violations": [],
            },
            indent=2,
        )
        + "\n"
    )
    (root / ARTIFACT_RUNNER_PREFLIGHT).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_exact_eval_runner_preflight",
                "cuda_available": True,
                "device_count": 1,
                "device_name": "T4",
                "nvidia_dali_fn_module": "nvidia.dali.fn",
            }
        )
        + "\n"
    )
    supply_scan = {
        "schema_version": 1,
        "tool": "scripts/scan_lightning_supply_chain.py",
        "status": "OK",
        "strict": True,
        "violation_count": 0,
        "violations": [],
        "package_versions": {"lightning": None, "lightning-sdk": "2026.4.10"},
    }
    (root / ARTIFACT_SUPPLY_CHAIN_SCAN_PRE).write_text(json.dumps(supply_scan) + "\n")
    (root / ARTIFACT_SUPPLY_CHAIN_SCAN).write_text(json.dumps(supply_scan) + "\n")
    if adjudication:
        contest_payload = json.loads((root / "contest_auth_eval.json").read_text())
        (root / "contest_auth_eval.adjudicated.json").write_text(
            json.dumps(contest_payload, indent=2, sort_keys=True) + "\n"
        )
        (root / "adjudication_provenance.json").write_text(
            json.dumps(
                {
                    "contest_cuda_archive_sha256": identity["archive_sha256"],
                    "contest_cuda_archive_bytes": identity["archive_size_bytes"],
                    "contest_cuda_score_source": "contest_auth_eval.json:score_recomputed_from_components",
                    "contest_cuda_device": "cuda",
                },
                indent=2,
            )
            + "\n"
        )
    return identity["archive_sha256"], identity["archive_size_bytes"]


def _write_component_response_artifact_dir(root: Path, *, passed: bool = True) -> tuple[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    baseline_sha = "b" * 64
    baseline_bytes = 456
    metadata: dict[str, object] = {
        "schema_version": 1,
        "job_name": "component-response-job",
        "role": "official_component_response",
        "expected_archive_sha256": baseline_sha,
        "expected_archive_size_bytes": baseline_bytes,
        "expected_baseline_archive_sha256": baseline_sha,
        "expected_baseline_archive_size_bytes": baseline_bytes,
        "queue_metadata": {"lane": "official_component_response"},
        "adjudication": None,
        "score_source": "official_component_response_summary.json:contest_auth_eval_json_components",
        "status_source": "lightning_sdk_job_attributes",
    }
    (root / "lightning_queue_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / ARTIFACT_COMPONENT_RESPONSE_INPUTS).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_official_component_response_input_preflight",
                "baseline_archive": {
                    "path": "/repo/baseline.zip",
                    "bytes": baseline_bytes,
                    "sha256": baseline_sha,
                    "zip_member_count": 2,
                },
                "baseline_contest_auth_eval_json": None,
                "perturbation_plan": {
                    "path": "/repo/plan.json",
                    "bytes": 123,
                    "sha256": "c" * 64,
                },
                "point_count": 3,
                "nonzero_point_count": 2,
                "points": [
                    {"index": 0, "epsilon": -1.0, "archive": {"path": "/repo/m.zip", "bytes": 111, "sha256": "d" * 64}},
                    {"index": 1, "epsilon": 0.0, "archive": {"path": "/repo/baseline.zip", "bytes": baseline_bytes, "sha256": baseline_sha}},
                    {"index": 2, "epsilon": 1.0, "archive": {"path": "/repo/p.zip", "bytes": 112, "sha256": "e" * 64}},
                ],
            },
            indent=2,
        )
        + "\n"
    )
    supply_scan = {
        "schema_version": 1,
        "tool": "scripts/scan_lightning_supply_chain.py",
        "status": "OK",
        "strict": True,
        "violation_count": 0,
        "violations": [],
        "package_versions": {"lightning": None, "lightning-sdk": "2026.4.10"},
    }
    (root / ARTIFACT_SUPPLY_CHAIN_SCAN_PRE).write_text(json.dumps(supply_scan) + "\n")
    (root / ARTIFACT_SUPPLY_CHAIN_SCAN).write_text(json.dumps(supply_scan) + "\n")
    (root / ARTIFACT_DALI_REQUIREMENTS).write_text(
        "https://pypi.nvidia.com/nvidia-dali-cuda130/nvidia_dali_cuda130-1.52.0-py3-none-manylinux_2_28_x86_64.whl "
        "--hash=sha256:37369fb30e9c66f710b29836688c90abc36793bbe757cd3ad699fac76ba07119\n"
    )
    (root / ARTIFACT_DALI_BOOTSTRAP).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_exact_eval_dali_bootstrap",
                "required_dali_version": "1.52.0",
                "bootstrap_action": "already_exact",
                "selected_package": "nvidia-dali-cuda130",
                "selected_requirement": "nvidia-dali-cuda130==1.52.0",
                "selected_wheels": [
                    {
                        "name": "nvidia-dali-cuda130",
                        "version": "1.52.0",
                        "url": "https://pypi.nvidia.com/nvidia-dali-cuda130/nvidia_dali_cuda130-1.52.0-py3-none-manylinux_2_28_x86_64.whl",
                        "sha256": "37369fb30e9c66f710b29836688c90abc36793bbe757cd3ad699fac76ba07119",
                    }
                ],
                "final_probe": {
                    "dali_version": "1.52.0",
                    "nvidia_dali_fn_module": "nvidia.dali.fn",
                    "installed_distributions": {
                        "nvidia-dali-cuda120": None,
                        "nvidia-dali-cuda130": "1.52.0",
                    },
                },
                "final_probe_violations": [],
            },
            indent=2,
        )
        + "\n"
    )
    (root / ARTIFACT_RUNNER_PREFLIGHT).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_exact_eval_runner_preflight",
                "cuda_available": True,
                "device_count": 1,
                "device_name": "T4",
                "gpu_t4_match": True,
                "nvidia_dali_fn_module": "nvidia.dali.fn",
            }
        )
        + "\n"
    )
    response_curve_paths = {}
    for component in ("posenet", "segnet", "combined"):
        curve_name = f"{component}_official_response_curve.json"
        response_curve_paths[component] = str(root / curve_name)
        (root / curve_name).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": "official_component_response_curves_v1",
                    "tool": "experiments/profile_component_sensitivity_official.py",
                    "component": component,
                    "device": "cuda",
                    "official_component_response": True,
                    "canonical_scorer_path": True,
                    "component_response_path": "archive_zip_inflate_sh_upstream_evaluate_py",
                    "promotion_eligible": passed,
                    "passed": passed,
                    "points": [
                        {"epsilon": -1.0, "value": 0.9, "baseline": 1.0, "delta": -0.1},
                        {"epsilon": 0.0, "value": 1.0, "baseline": 1.0, "delta": 0.0},
                        {"epsilon": 1.0, "value": 1.1, "baseline": 1.0, "delta": 0.1},
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    (root / ARTIFACT_COMPONENT_RESPONSE_SUMMARY).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "official_component_response_summary_v1",
                "tool": "experiments/profile_component_sensitivity_official.py",
                "device": "cuda",
                "promotion_eligible": passed,
                "baseline_archive": {
                    "path": "/repo/baseline.zip",
                    "bytes": baseline_bytes,
                    "sha256": baseline_sha,
                },
                "perturbation_plan": {
                    "path": "/repo/plan.json",
                    "bytes": 123,
                    "sha256": "c" * 64,
                },
                "response_curve_paths": response_curve_paths,
                "points": [],
            },
            indent=2,
        )
        + "\n"
    )
    return baseline_sha, baseline_bytes


def test_exact_cuda_eval_command_is_json_and_cuda_only() -> None:
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/artifacts/archive.zip",
        upstream_dir="/upstream",
        output_dir="/out",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    assert "experiments/contest_auth_eval.py" in command
    assert "--device cuda" in command
    assert "contest_auth_eval.json" in command
    assert "score_recomputed_from_components" in command
    assert "auth_eval.log" in command
    assert "scripts/scan_lightning_supply_chain.py" in command
    assert "--quiet --strict" in command
    assert command.count("scripts/scan_lightning_supply_chain.py") == 2
    assert "rm -rf /out/eval_work /out/uv_project_env" in command
    assert "export UV_PROJECT_ENVIRONMENT=/out/uv_project_env" in command
    assert "export UV_LINK_MODE=${UV_LINK_MODE:-copy}" in command
    assert "LIGHTNING_VENV_LOCK=.omx/state/lightning_exact_eval_venv.lock" in command
    assert "FATAL: timed out waiting for Lightning exact-eval venv lock" in command
    assert "nvidia-dali-cuda130==1.52.0" in command
    assert "nvidia-dali-cuda120==1.52.0" in command
    assert "https://pypi.nvidia.com/nvidia-dali-cuda130/" in command
    assert "37369fb30e9c66f710b29836688c90abc36793bbe757cd3ad699fac76ba07119" in command
    assert "--require-hashes" in command
    assert "--no-deps" in command
    assert "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" in command
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in command
    assert "nvidia.dali.fn" in command
    assert "nvidia_dali_fn_module" in command
    assert ARTIFACT_DALI_BOOTSTRAP in command
    assert "lightning_dali_requirements.txt" in command
    assert ARTIFACT_RUNNER_PREFLIGHT in command
    assert ARTIFACT_SUPPLY_CHAIN_SCAN in command
    assert "--index-url" not in command
    assert "--extra-index-url" not in command
    assert "grep" not in command
    assert "re.search" not in command
    assert "expected_sha = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'" in command
    assert "expected_sha = null" not in command


def test_official_component_response_command_is_cuda_and_supply_chain_guarded() -> None:
    command = official_component_response_command(
        repo_dir="/repo",
        baseline_archive_path="/repo/baseline.zip",
        perturbation_plan_path="/repo/plan.json",
        upstream_dir="/repo/upstream",
        output_dir="/out",
        expected_baseline_archive_sha256="b" * 64,
        expected_baseline_archive_size_bytes=456,
        baseline_contest_auth_eval_json="/repo/baseline_contest_auth_eval.json",
        allow_directional=True,
        require_passed=True,
    )

    assert "experiments/profile_component_sensitivity_official.py" in command
    assert "--baseline-archive /repo/baseline.zip" in command
    assert "--perturbation-plan /repo/plan.json" in command
    assert "--baseline-contest-auth-eval-json /repo/baseline_contest_auth_eval.json" in command
    assert "--device cuda" in command
    assert "--allow-directional" in command
    assert "--require-passed" in command
    assert "LIGHTNING_COMPONENT_RESPONSE_INPUT_PREFLIGHT_OK" in command
    assert "LIGHTNING_COMPONENT_RESPONSE_CLEANUP_OK" in command
    assert "scripts/scan_lightning_supply_chain.py" in command
    assert command.count("scripts/scan_lightning_supply_chain.py") == 2
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in command
    assert "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" in command
    assert "--require-hashes" in command
    assert "--no-deps" in command
    assert "validate-component-response-artifacts" in command
    assert ARTIFACT_COMPONENT_RESPONSE_INPUTS in command
    assert ARTIFACT_COMPONENT_RESPONSE_SUMMARY in command
    assert ARTIFACT_COMPONENT_RESPONSE_VALIDATION in command
    for curve_name in COMPONENT_RESPONSE_CURVE_FILES:
        assert curve_name in command
    assert "--index-url" not in command
    assert "--extra-index-url" not in command


def test_diagnostic_component_sensitivity_command_extracts_archive_and_is_non_promotable() -> None:
    command = diagnostic_component_sensitivity_command(
        repo_dir="/repo",
        baseline_archive_path="/repo/baseline.zip",
        upstream_dir="/repo/upstream",
        output_dir="/out",
        expected_baseline_archive_sha256="b" * 64,
        expected_baseline_archive_size_bytes=456,
        pair_weights_path="/repo/weights.pt",
        response_epsilons="-0.001,0.0,0.001",
    )

    assert "experiments/profile_component_sensitivity.py" in command
    assert "--checkpoint /out/extracted/renderer.bin" in command
    assert "--masks-mkv /out/extracted/masks.mkv" in command
    assert "--poses /out/extracted/optimized_poses.bin" in command
    assert "--video /repo/upstream/videos/0.mkv" in command
    assert "--upstream /repo/upstream" in command
    assert "--device cuda" in command
    assert "--pair-weights /repo/weights.pt" in command
    assert "--all-pairs" not in command
    assert "--manifest-output" not in command
    assert "zip-slip member in baseline archive" in command
    assert "duplicate zip member in baseline archive" in command
    assert "hidden/resource-fork zip member" in command
    assert "renderer.bin" in command
    assert "masks.mkv" in command
    assert "optimized_poses.bin" in command
    assert "LIGHTNING_DIAGNOSTIC_COMPONENT_SENSITIVITY_INPUT_PREFLIGHT_OK" in command
    assert "LIGHTNING_DIAGNOSTIC_COMPONENT_SENSITIVITY_RUN_METADATA_OK" in command
    assert "LIGHTNING_DIAGNOSTIC_COMPONENT_SENSITIVITY_ARTIFACTS_OK" in command
    assert "promotion_eligible': False" in command
    assert "score_claim': False" in command
    assert "scripts/scan_lightning_supply_chain.py" in command
    assert command.count("scripts/scan_lightning_supply_chain.py") == 2
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in command
    assert "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" in command
    assert "--require-hashes" in command
    assert "--no-deps" in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_INPUTS in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_RUN in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION in command
    for name in COMPONENT_SENSITIVITY_MAP_FILES + COMPONENT_SENSITIVITY_CURVE_FILES:
        assert name in command
    assert "--index-url" not in command
    assert "--extra-index-url" not in command


def test_diagnostic_component_sensitivity_spec_is_diagnostic_and_cuda() -> None:
    spec = make_diagnostic_component_sensitivity_spec(
        name="component_sensitivity",
        baseline_archive_path="/repo/baseline.zip",
        repo_dir="/repo",
        upstream_dir="/repo/upstream",
        studio="pact",
        expected_baseline_archive_sha256="b" * 64,
        expected_baseline_archive_size_bytes=456,
        queue_metadata={"lane": "beta_diagnostic"},
    )
    assert spec.machine == "T4"
    assert spec.interruptible is False
    assert spec.reuse_snapshot is False
    assert spec.role == "diagnostic_component_sensitivity"
    assert spec.remote_output_dir == "/repo/experiments/results/lightning_batch/component_sensitivity"
    assert spec.queue_metadata["lane"] == "beta_diagnostic"
    assert "score_claim" in spec.command
    assert "--all-pairs" in spec.command
    spec.validate()


def test_component_response_spec_is_fail_closed() -> None:
    spec = make_official_component_response_spec(
        name="component_response",
        baseline_archive_path="/repo/baseline.zip",
        perturbation_plan_path="/repo/plan.json",
        repo_dir="/repo",
        upstream_dir="/repo/upstream",
        studio="pact",
        expected_baseline_archive_sha256="b" * 64,
        expected_baseline_archive_size_bytes=456,
        require_passed=True,
    )
    assert spec.machine == "T4"
    assert spec.interruptible is False
    assert spec.reuse_snapshot is False
    assert spec.role == "official_component_response"
    assert spec.remote_output_dir == "/repo/experiments/results/lightning_batch/component_response"
    spec.validate()


def test_component_response_dry_run_validates_manifest_closure_when_manifest_supplied(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    baseline = tmp_path / "baseline.zip"
    variant = tmp_path / "archives" / "point_001_eps_p1.zip"
    variant.parent.mkdir(parents=True)
    baseline.write_bytes(b"baseline")
    variant.write_bytes(b"variant")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "points": [
                    {"index": 0, "epsilon": 0.0},
                    {"index": 1, "epsilon": 1.0, "archive": "archives/point_001_eps_p1.zip"},
                ],
            }
        )
    )
    manifest = tmp_path / "source_manifest.json"
    base_files = [
        {"path": "baseline.zip"},
        {"path": "plan.json"},
    ]
    manifest.write_text(json.dumps({"files": base_files}))
    args = argparse.Namespace(
        dry_run=True,
        source_manifest=str(manifest),
        local_perturbation_plan=str(plan),
        baseline_archive="/repo/baseline.zip",
        perturbation_plan="/repo/plan.json",
        repo_dir="/repo",
        baseline_contest_auth_eval_json=None,
    )

    with pytest.raises(SystemExit, match="staged source manifest does not include"):
        module._validate_component_response_submit_inputs(args)

    manifest.write_text(json.dumps({"files": [*base_files, {"path": "archives/point_001_eps_p1.zip"}]}))
    module._validate_component_response_submit_inputs(args)


def test_component_response_submit_rejects_nonportable_plan_point_paths(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "points": [
                    {"index": 0, "epsilon": 0.0},
                    {
                        "index": 1,
                        "epsilon": 1.0,
                        "archive": str(tmp_path / "archives" / "point_001_eps_p1.zip"),
                    },
                ],
            }
        )
    )
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "baseline.zip"}, {"path": "plan.json"}]}))
    args = argparse.Namespace(
        dry_run=True,
        source_manifest=str(manifest),
        local_perturbation_plan=str(plan),
        baseline_archive="/repo/baseline.zip",
        perturbation_plan="/repo/plan.json",
        repo_dir="/repo",
        baseline_contest_auth_eval_json=None,
    )

    with pytest.raises(SystemExit, match="points\\[1\\]\\.archive must be relative"):
        module._validate_component_response_submit_inputs(args)


def test_component_response_explicit_baseline_json_overrides_plan_absolute_metadata(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "baseline_contest_auth_eval_json": str(tmp_path / "stale_local_eval.json"),
                "points": [
                    {"index": 0, "epsilon": 0.0},
                    {"index": 1, "epsilon": 1.0, "archive": "archives/point_001_eps_p1.zip"},
                ],
            }
        )
    )
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "baseline.zip"},
                    {"path": "plan.json"},
                    {"path": "archives/point_001_eps_p1.zip"},
                    {"path": "baseline_eval.json"},
                ]
            }
        )
    )
    args = argparse.Namespace(
        dry_run=True,
        source_manifest=str(manifest),
        local_perturbation_plan=str(plan),
        baseline_archive="/repo/baseline.zip",
        perturbation_plan="/repo/plan.json",
        repo_dir="/repo",
        baseline_contest_auth_eval_json="/repo/baseline_eval.json",
    )

    module._validate_component_response_submit_inputs(args)


def test_component_response_manifest_rejects_traversal_entries(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "points": [
                    {"index": 0, "epsilon": 0.0},
                    {"index": 1, "epsilon": 1.0, "archive": "archives/point_001_eps_p1.zip"},
                ],
            }
        )
    )
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "baseline.zip"},
                    {"path": "plan.json"},
                    {"path": "../archives/point_001_eps_p1.zip"},
                ]
            }
        )
    )
    args = argparse.Namespace(
        dry_run=True,
        source_manifest=str(manifest),
        local_perturbation_plan=str(plan),
        baseline_archive="/repo/baseline.zip",
        perturbation_plan="/repo/plan.json",
        repo_dir="/repo",
        baseline_contest_auth_eval_json=None,
    )

    with pytest.raises(SystemExit, match="path traversal"):
        module._validate_component_response_submit_inputs(args)


def test_component_response_explicit_baseline_json_does_not_allow_absolute_point_eval_json(
    tmp_path: Path,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "official_component_response_plan_v1",
                "baseline_contest_auth_eval_json": str(tmp_path / "stale_local_eval.json"),
                "points": [
                    {"index": 0, "epsilon": 0.0},
                    {
                        "index": 1,
                        "epsilon": 1.0,
                        "archive": "archives/point_001_eps_p1.zip",
                        "contest_auth_eval_json": str(tmp_path / "point_001_eval.json"),
                    },
                ],
            }
        )
    )
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "baseline.zip"},
                    {"path": "plan.json"},
                    {"path": "archives/point_001_eps_p1.zip"},
                    {"path": "baseline_eval.json"},
                ]
            }
        )
    )
    args = argparse.Namespace(
        dry_run=True,
        source_manifest=str(manifest),
        local_perturbation_plan=str(plan),
        baseline_archive="/repo/baseline.zip",
        perturbation_plan="/repo/plan.json",
        repo_dir="/repo",
        baseline_contest_auth_eval_json="/repo/baseline_eval.json",
    )

    with pytest.raises(SystemExit, match="points\\[1\\]\\.contest_auth_eval_json must be relative"):
        module._validate_component_response_submit_inputs(args)


def test_component_response_dry_run_rejects_half_enabled_manifest_validation(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(
        dry_run=True,
        source_manifest=str(tmp_path / "source_manifest.json"),
        local_perturbation_plan=None,
    )

    with pytest.raises(SystemExit, match="requires both --source-manifest and --local-perturbation-plan"):
        module._validate_component_response_submit_inputs(args)


def test_component_sensitivity_submit_validates_staged_input_closure(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "baseline.zip"}]}))
    args = argparse.Namespace(
        dry_run=False,
        source_manifest=str(manifest),
        baseline_archive="/repo/baseline.zip",
        repo_dir="/repo",
        upstream_dir="/repo/upstream",
        video=None,
        pair_weights="/repo/weights.pt",
    )

    with pytest.raises(SystemExit, match="staged source manifest does not include"):
        module._validate_component_sensitivity_submit_inputs(args)

    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "baseline.zip"},
                    {"path": "upstream/videos/0.mkv"},
                    {"path": "weights.pt"},
                ]
            }
        )
    )
    module._validate_component_sensitivity_submit_inputs(args)


def test_exact_eval_submit_requires_source_manifest_and_archive_closure(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=None,
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
    )
    with pytest.raises(SystemExit, match="requires --source-manifest"):
        module._validate_exact_eval_submit_inputs(args)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "other.zip"}]}) + "\n")
    args.source_manifest = str(manifest)
    with pytest.raises(SystemExit, match="does not include archive artifact"):
        module._validate_exact_eval_submit_inputs(args)

    manifest.write_text(json.dumps({"files": [{"path": "archive.zip"}]}) + "\n")
    module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_submit_requires_metadata_baseline_json_closure(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "archive.zip"}]}) + "\n")
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=["baseline_json=baseline/contest_auth_eval.json"],
    )

    with pytest.raises(SystemExit, match="metadata baseline_json artifact"):
        module._validate_exact_eval_submit_inputs(args)

    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "archive.zip"},
                    {"path": "baseline/contest_auth_eval.json"},
                ]
            }
        )
        + "\n"
    )
    module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_manifest_rejects_traversal_entries(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "../archive.zip"}]}))
    args = argparse.Namespace(
        dry_run=False,
        studio="lossy-compression-challenge",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
    )

    with pytest.raises(SystemExit, match="path traversal"):
        module._validate_exact_eval_submit_inputs(args)


def test_non_dry_run_studio_submit_requires_remote_preflight(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        remote_preflight_ssh_target=None,
        allow_skip_remote_preflight_reason=None,
    )
    with pytest.raises(SystemExit, match="requires --remote-preflight-ssh-target"):
        module._require_remote_preflight_for_submit(args, role="exact-eval")

    args.allow_skip_remote_preflight_reason = "externally attested image-backed repro path"
    module._require_remote_preflight_for_submit(args, role="exact-eval")


def test_component_response_ssh_harvest_requires_state_or_override(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(
        job_name=None,
        remote_artifact_dir="/teamspace/jobs/manual/artifacts/pact/out",
        allow_manual_artifact_dir=False,
        override_reason=None,
    )
    with pytest.raises(SystemExit, match="must be state-derived"):
        module._require_manual_artifact_override(args, role="component-response")

    args.allow_manual_artifact_dir = True
    args.override_reason = "terminal forensic recovery from known SDK artifact path"
    assert module._require_manual_artifact_override(args, role="component-response") == args.override_reason


def test_exact_cuda_eval_command_wires_expected_archive_and_adjudication() -> None:
    adjudication = LightningAdjudicationSpec(
        baseline_score=1.2,
        predicted_band_low=1.0,
        predicted_band_high=1.4,
        regression_threshold=1.6,
        baseline_archive_size_bytes=100,
        max_posenet_dist=0.006,
        max_segnet_dist=0.004,
        baseline_posenet_dist=0.003,
        baseline_segnet_dist=0.002,
        max_posenet_relative=1.5,
        max_segnet_relative=1.2,
        component_reference_label="frontier",
    )
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/repo/archive.zip",
        upstream_dir="/repo/upstream",
        output_dir="/out",
        expected_archive_sha256="a" * 64,
        expected_archive_size_bytes=123,
        queue_metadata={"lane": "pfp16"},
        adjudication=adjudication,
    )
    assert "LIGHTNING_ARCHIVE_PREFLIGHT_JSON_OK" in command
    assert "expected_archive_sha256" in command
    assert "cp /repo/archive.zip /out/archive.zip" in command
    assert "--archive /out/archive.zip" in command
    assert "scripts/adjudicate_contest_auth_eval.py" in command
    assert "--baseline-score 1.2" in command
    assert "--baseline-archive-bytes 100" in command
    assert "--max-posenet-dist 0.006" in command
    assert "--max-segnet-dist 0.004" in command
    assert "--baseline-posenet-dist 0.003" in command
    assert "--baseline-segnet-dist 0.002" in command
    assert "--max-posenet-relative 1.5" in command
    assert "--max-segnet-relative 1.2" in command
    assert "--component-reference-label frontier" in command
    assert "--allow-component-gate-forensic-success" in command
    assert "adjudication_provenance.json" in command


def test_exact_cuda_eval_command_can_fail_job_on_component_gate() -> None:
    adjudication = LightningAdjudicationSpec(
        baseline_score=1.2,
        predicted_band_low=1.0,
        predicted_band_high=1.4,
        regression_threshold=1.6,
        allow_component_gate_forensic_success=False,
    )
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/repo/archive.zip",
        upstream_dir="/repo/upstream",
        output_dir="/out",
        adjudication=adjudication,
        **_expected_archive_kwargs(),
    )

    assert "scripts/adjudicate_contest_auth_eval.py" in command
    assert "--allow-component-gate-forensic-success" not in command


def test_exact_eval_spec_is_fail_closed() -> None:
    spec = make_exact_eval_spec(
        name="pfp16-eval",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    assert spec.machine == "T4"
    assert spec.interruptible is False
    assert spec.reuse_snapshot is False
    assert spec.role == "exact_cuda_eval"
    spec.validate()


def test_exact_eval_spec_records_expected_archive_and_queue_metadata() -> None:
    spec = make_exact_eval_spec(
        name="pfp16-eval",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        expected_archive_sha256="b" * 64,
        expected_archive_size_bytes=456,
        queue_metadata={"lane": "pfp16", "operator": "codex"},
        local_artifact_dir="/local/artifacts/pfp16-eval",
        adjudication=_adjudication(),
    )
    payload = spec.asdict()
    assert payload["expected_archive_sha256"] == "b" * 64
    assert payload["expected_archive_size_bytes"] == 456
    assert payload["queue_metadata"]["lane"] == "pfp16"
    assert payload["local_artifact_dir"].endswith("pfp16-eval")


def test_exact_eval_default_output_dir_is_writable_studio_workspace_path() -> None:
    spec = make_exact_eval_spec(
        name="job_with_underscores",
        archive_path="/repo/archive.zip",
        repo_dir="/teamspace/studios/this_studio/pact",
        upstream_dir="/upstream",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    sdk_name = lightning_sdk_job_name("job_with_underscores")
    assert sdk_name == "job-with-underscores"
    assert spec.remote_output_dir == default_exact_eval_output_dir(
        repo_dir="/teamspace/studios/this_studio/pact",
        job_name="job_with_underscores",
    )
    assert spec.remote_output_dir in spec.command
    assert "/teamspace/jobs/" not in spec.command
    assert lightning_sdk_artifact_path("job_with_underscores") == (
        f"/teamspace/jobs/{sdk_name}/artifacts"
    )
    assert lightning_sdk_persisted_studio_output_dir(
        sdk_artifact_path="/teamspace/jobs/job-with-underscores/artifacts",
        remote_output_dir="/teamspace/studios/this_studio/pact/experiments/out",
    ) == "/teamspace/jobs/job-with-underscores/artifacts/pact/experiments/out"


def test_exact_eval_rejects_read_only_sdk_artifact_view_as_output_dir() -> None:
    with pytest.raises(ValueError, match="read-only"):
        make_exact_eval_spec(
            name="bad-output",
            archive_path="/repo/archive.zip",
            repo_dir="/repo",
            upstream_dir="/upstream",
            output_dir="/teamspace/jobs/bad-output/artifacts",
            adjudication=_adjudication(),
            **_expected_archive_kwargs(),
        )


def test_exact_eval_rejects_partial_expected_archive_identity() -> None:
    with pytest.raises(ValueError, match="provided together"):
        make_exact_eval_spec(
            name="bad-expected",
            archive_path="/repo/archive.zip",
            repo_dir="/repo",
            upstream_dir="/upstream",
            expected_archive_sha256="c" * 64,
        )


def test_adjudication_relative_gate_requires_component_reference() -> None:
    spec = LightningAdjudicationSpec(
        baseline_score=1.2,
        predicted_band_low=1.0,
        predicted_band_high=1.4,
        regression_threshold=1.6,
        max_posenet_relative=1.05,
    )
    with pytest.raises(ValueError, match="baseline_posenet_dist"):
        spec.validate()


def test_exact_eval_rejects_interruptible() -> None:
    spec = LightningBatchJobSpec(
        name="bad",
        machine="T4",
        command="python experiments/contest_auth_eval.py --device cuda && cp contest_auth_eval.json .",
        interruptible=True,
        role="exact_cuda_eval",
        expected_archive_sha256="a" * 64,
        expected_archive_size_bytes=123,
        adjudication=_adjudication(),
    )
    with pytest.raises(ValueError, match="interruptible"):
        spec.validate()


def test_exact_eval_rejects_missing_runner_preflight() -> None:
    spec = LightningBatchJobSpec(
        name="bad",
        machine="T4",
        command="python experiments/contest_auth_eval.py --device cuda && cp contest_auth_eval.json .",
        role="exact_cuda_eval",
        expected_archive_sha256="a" * 64,
        expected_archive_size_bytes=123,
        adjudication=_adjudication(),
    )
    with pytest.raises(ValueError, match="supply-chain scan"):
        spec.validate()


def test_dry_run_records_without_sdk_call(tmp_path: Path) -> None:
    spec = make_exact_eval_spec(
        name="dry",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    record = client.submit(spec, dry_run=True)
    assert record["status"] == "DRY_RUN"
    assert record["spec"]["name"] == "dry"
    assert record["queue"]["command_sha256"]
    assert record["queue"]["queue_name"] == "official_lightning_batch_jobs"
    assert record["queue"]["remote_output_dir"] == "/repo/experiments/results/lightning_batch/dry"
    assert record["queue"]["sdk_artifact_path"] == "/teamspace/jobs/dry/artifacts"
    assert json.loads((tmp_path / "jobs.json").read_text())[0]["status"] == "DRY_RUN"


def test_submit_records_official_job_fields(tmp_path: Path) -> None:
    class FakeJob:
        name = "submitted"
        status = "running"
        link = "https://lightning.ai/jobs/submitted"
        snapshot_path = "/teamspace/jobs/submitted/snapshot"
        artifact_path = "/teamspace/jobs/submitted/artifacts"
        machine = "T4"
        total_cost = 0.0
        calls: list[dict] = []

        @classmethod
        def run(cls, **kwargs):
            cls.calls.append(kwargs)
            return cls()

    spec = make_exact_eval_spec(
        name="submitted",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json", job_cls=FakeJob)
    record = client.submit(spec)

    assert FakeJob.calls[0]["name"] == "submitted"
    assert FakeJob.calls[0]["machine"] == "T4"
    assert FakeJob.calls[0]["interruptible"] is False
    assert record["status"] == "SUBMITTED"
    assert record["job"]["snapshot_path"].endswith("/snapshot")
    assert record["job"]["artifact_path"].endswith("/artifacts")
    assert record["job"]["source"] == "lightning_sdk_job_attributes"
    assert record["queue"]["command_sha256"]


def test_refresh_status_uses_job_attributes_not_logs(tmp_path: Path) -> None:
    class FakeJob:
        name = "dry"
        status = "completed"
        link = "https://lightning.ai/jobs/dry"
        snapshot_path = "/snapshot"
        artifact_path = "/artifacts"
        machine = "T4"
        total_cost = 0.25

        @property
        def logs(self):
            raise AssertionError("status refresh must not read logs")

    spec = make_exact_eval_spec(
        name="dry",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)
    refreshed = client.refresh_status_from_job(job_name="dry", job=FakeJob())
    assert refreshed["status"] == "completed"
    assert refreshed["job"]["source"] == "lightning_sdk_job_attributes"
    assert refreshed["status_history"][-1]["source"] == "lightning_sdk_job_attributes"


def test_batch_job_cli_refresh_status_infers_sdk_context_from_state(tmp_path: Path) -> None:
    state_path = tmp_path / "jobs.json"
    spec = make_exact_eval_spec(
        name="job_with_underscores",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        teamspace="comma-lab",
        user="adpena",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=state_path)
    client.submit(spec, dry_run=True)
    records = json.loads(state_path.read_text())
    records[0]["job"] = {"name": "custom-sdk-job"}
    state_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    (tmp_path / "lightning_sdk.py").write_text(
        """
class Job:
    def __init__(self, *, name, teamspace=None, org=None, user=None):
        if name != "custom-sdk-job":
            raise RuntimeError(f"unexpected name {name!r}")
        if teamspace != "comma-lab":
            raise RuntimeError(f"unexpected teamspace {teamspace!r}")
        if user != "adpena":
            raise RuntimeError(f"unexpected user {user!r}")
        self.name = name
        self.status = "completed"
        self.link = "https://lightning.ai/jobs/custom-sdk-job"
        self.snapshot_path = "/teamspace/jobs/custom-sdk-job/snapshot"
        self.artifact_path = "/teamspace/jobs/custom-sdk-job/artifacts"
        self.machine = "g4dn.2xlarge"
        self.total_cost = 0.42
""".lstrip()
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT / "src")])
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "refresh-status",
            "--state-path",
            str(state_path),
            "--job-name",
            "job_with_underscores",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["job"]["name"] == "custom-sdk-job"
    assert payload["job"]["total_cost"] == 0.42


def test_batch_job_cli_refresh_status_all_skips_dry_runs(tmp_path: Path) -> None:
    state_path = tmp_path / "jobs.json"

    class SubmittedJob:
        def __init__(self, name: str):
            self.name = name
            self.status = "submitted"
            self.link = f"https://lightning.ai/jobs/{name}"
            self.snapshot_path = f"/teamspace/jobs/{name}/snapshot"
            self.artifact_path = f"/teamspace/jobs/{name}/artifacts"
            self.machine = "T4"
            self.total_cost = 0.0

        @classmethod
        def run(cls, **kwargs):
            return cls(kwargs["name"])

    client = LightningBatchJobsClient(state_path=state_path, job_cls=SubmittedJob)
    for name in ("job_one", "job_two", "dry_job"):
        spec = make_exact_eval_spec(
            name=name,
            archive_path="/repo/archive.zip",
            repo_dir="/repo",
            upstream_dir="/upstream",
            studio="pact",
            teamspace="comma-lab",
            user="adpena",
            adjudication=_adjudication(),
            **_expected_archive_kwargs(),
        )
        client.submit(spec, dry_run=(name == "dry_job"))

    (tmp_path / "lightning_sdk.py").write_text(
        """
class Job:
    def __init__(self, *, name, teamspace=None, org=None, user=None):
        self.name = name
        self.status = "completed"
        self.link = f"https://lightning.ai/jobs/{name}"
        self.snapshot_path = f"/teamspace/jobs/{name}/snapshot"
        self.artifact_path = f"/teamspace/jobs/{name}/artifacts"
        self.machine = "g4dn.2xlarge"
        self.total_cost = 0.25
        if teamspace != "comma-lab":
            raise RuntimeError(f"unexpected teamspace {teamspace!r}")
        if user != "adpena":
            raise RuntimeError(f"unexpected user {user!r}")
""".lstrip()
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT / "src")])
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "refresh-status",
            "--state-path",
            str(state_path),
            "--all",
            "--fail-on-error",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["failure_count"] == 0
    assert payload["refreshed_count"] == 2
    assert payload["skipped_count"] == 1
    assert {row["job"]["name"] for row in payload["results"]} == {"job_one", "job_two"}
    assert payload["skipped"] == [{"job_name": "dry_job", "reason": "dry_run"}]


def test_validate_local_artifact_dir_uses_json_not_logs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    expected_sha, expected_bytes = _write_artifact_dir(artifact_dir)

    result = validate_local_artifact_dir(
        artifact_dir,
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
    )

    assert result["archive_sha256"] == expected_sha
    assert result["archive_size_bytes"] == expected_bytes
    assert result["score_recomputed_from_components"] == 1.25
    assert result["score_source"] == "contest_auth_eval.json:score_recomputed_from_components"
    assert result["supply_chain_pre"]["status"] == "OK"
    assert result["supply_chain_post"]["status"] == "OK"
    assert result["dali_bootstrap"]["final_probe_violations"] == []
    assert result["runner_preflight"]["cuda_available"] is True


def test_validate_local_artifact_dir_rejects_failed_supply_chain_scan(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir)
    scan_path = artifact_dir / ARTIFACT_SUPPLY_CHAIN_SCAN
    payload = json.loads(scan_path.read_text())
    payload["status"] = "FAIL"
    payload["violation_count"] = 1
    payload["violations"] = ["lightning==2.6.3"]
    scan_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="supply-chain scan"):
        validate_local_artifact_dir(artifact_dir)


def test_validate_local_artifact_dir_rejects_bad_dali_bootstrap(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir)
    bootstrap_path = artifact_dir / ARTIFACT_DALI_BOOTSTRAP
    payload = json.loads(bootstrap_path.read_text())
    payload["final_probe_violations"] = ["wrong DALI"]
    bootstrap_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="DALI bootstrap"):
        validate_local_artifact_dir(artifact_dir)


def test_validate_local_artifact_dir_rejects_non_cuda_runner_preflight(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir)
    preflight_path = artifact_dir / ARTIFACT_RUNNER_PREFLIGHT
    payload = json.loads(preflight_path.read_text())
    payload["cuda_available"] = False
    preflight_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="runner preflight"):
        validate_local_artifact_dir(artifact_dir)


def test_validate_local_artifact_dir_rejects_missing_expected_archive_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir)
    metadata_path = artifact_dir / "lightning_queue_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["expected_archive_sha256"] = None
    payload["expected_archive_size_bytes"] = None
    metadata_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="expected archive"):
        validate_local_artifact_dir(artifact_dir)


def test_validate_local_artifact_dir_rejects_missing_adjudication_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir, adjudication=False)

    with pytest.raises(ValueError, match="adjudication"):
        validate_local_artifact_dir(artifact_dir)


def test_mirror_local_artifact_dir_validates_mirror_and_writes_report(tmp_path: Path) -> None:
    source = tmp_path / "source"
    mirror = tmp_path / "mirror"
    expected_sha, expected_bytes = _write_artifact_dir(source, adjudication=True)

    result = mirror_local_artifact_dir(
        source,
        mirror,
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
        require_adjudication=True,
    )

    assert result["adjudication_provenance"]["contest_cuda_archive_sha256"] == expected_sha
    assert result["promotion_eligible"] is True
    validation = json.loads((mirror / ARTIFACT_VALIDATION).read_text())
    assert validation["archive_size_bytes"] == expected_bytes


def test_validate_local_artifact_dir_reports_failed_component_gate_as_non_promotable(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir, adjudication=True)
    adjudication_path = artifact_dir / "adjudication_provenance.json"
    payload = json.loads(adjudication_path.read_text())
    payload.update(
        {
            "lane_status": "COMPONENT_GATE_REVIEW_REQUIRED",
            "component_gate_triggered": True,
            "component_gate_violations": [{"component": "segnet"}],
            "regression_triggered": False,
        }
    )
    adjudication_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    result = validate_local_artifact_dir(artifact_dir, require_adjudication=True)

    assert result["adjudication_lane_status"] == "COMPONENT_GATE_REVIEW_REQUIRED"
    assert result["adjudication_component_gate_triggered"] is True
    assert result["promotion_eligible"] is False


def test_validate_local_component_response_artifact_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "component_response"
    expected_sha, expected_bytes = _write_component_response_artifact_dir(artifact_dir, passed=True)

    result = validate_local_component_response_artifact_dir(
        artifact_dir,
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
        require_passed=True,
    )

    assert result["role"] == "official_component_response"
    assert result["baseline_archive_sha256"] == expected_sha
    assert result["baseline_archive_size_bytes"] == expected_bytes
    assert result["point_count"] == 3
    assert result["nonzero_point_count"] == 2
    assert result["promotion_eligible"] is True
    assert result["supply_chain_pre"]["status"] == "OK"
    assert result["supply_chain_post"]["status"] == "OK"
    assert result["runner_preflight"]["cuda_available"] is True
    assert set(result["curves"]) == {"posenet", "segnet", "combined"}


def test_validate_local_component_response_artifact_dir_rejects_failed_gates(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "component_response"
    _write_component_response_artifact_dir(artifact_dir, passed=False)

    with pytest.raises(ValueError, match="official component-response gates did not pass"):
        validate_local_component_response_artifact_dir(artifact_dir, require_passed=True)


def test_harvest_local_artifacts_attaches_validation_to_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    expected_sha, expected_bytes = _write_artifact_dir(artifact_dir, adjudication=True)
    spec = make_exact_eval_spec(
        name="dry",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
        adjudication=_adjudication(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    validation = client.harvest_local_artifacts(
        job_name="dry",
        artifact_dir=artifact_dir,
        require_adjudication=True,
    )

    assert validation["archive_sha256"] == expected_sha
    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "HARVESTED"
    assert record["harvests"][0]["archive_size_bytes"] == expected_bytes


def test_harvest_ssh_artifacts_mirrors_validates_and_records_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_source = tmp_path / "remote_source"
    expected_sha, expected_bytes = _write_artifact_dir(remote_source, adjudication=True)
    mirror = tmp_path / "mirror"
    calls: list[list[str]] = []
    persisted_remote = "/teamspace/jobs/dry/artifacts/pact/remote/out"

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ssh" and args[-1].startswith("test -d "):
            assert "BatchMode=yes" in args
            assert "PasswordAuthentication=no" in args
            assert "KbdInteractiveAuthentication=no" in args
            assert "ConnectTimeout=15" in args
            assert args[-2] == "lightning-host"
            assert args[-1] == f"test -d {persisted_remote}"
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "ssh" and args[-1].startswith("test -f "):
            name = args[-1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                args,
                0 if (remote_source / name).is_file() else 1,
                stdout="",
                stderr="",
            )
        if args[0] == "scp":
            assert "BatchMode=yes" in args
            assert "PasswordAuthentication=no" in args
            assert "KbdInteractiveAuthentication=no" in args
            assert "ConnectTimeout=15" in args
            assert "-p" in args
            assert args[-2].startswith(f"lightning-host:{persisted_remote}/")
            name = args[-2].rsplit("/", 1)[-1]
            assert name in CANONICAL_ARTIFACT_FILES or name == "auth_eval.log"
            shutil.copy2(remote_source / name, Path(args[-1]))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(lightning_batch_jobs.subprocess, "run", fake_run)
    spec = make_exact_eval_spec(
        name="dry",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        output_dir="/teamspace/studios/this_studio/pact/remote/out",
        local_artifact_dir=str(mirror),
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
        adjudication=_adjudication(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    validation = client.harvest_ssh_artifacts(
        job_name="dry",
        ssh_target="lightning-host",
        require_adjudication=True,
    )

    assert calls[0][0] == "ssh"
    assert calls[1][0] == "scp"
    assert validation["archive_sha256"] == expected_sha
    assert validation["ssh_source"]["remote_dir"] == persisted_remote
    assert "eval_work" not in validation["ssh_source"]["copied_files"]
    assert (mirror / ARTIFACT_VALIDATION).is_file()
    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "HARVESTED"
    assert record["status_history"][-1]["source"] == "ssh_artifact_validation"
    assert record["harvests"][0]["ssh_source"]["ssh_target"] == "lightning-host"


def test_harvest_ssh_component_response_artifacts_uses_state_artifact_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_source = tmp_path / "remote_component_response"
    expected_sha, expected_bytes = _write_component_response_artifact_dir(remote_source, passed=True)
    mirror = tmp_path / "mirror"
    calls: list[list[str]] = []
    persisted_remote = "/teamspace/jobs/component-response/artifacts/pact/remote/component"

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ssh" and args[-1].startswith("test -d "):
            assert "BatchMode=yes" in args
            assert "PasswordAuthentication=no" in args
            assert "KbdInteractiveAuthentication=no" in args
            assert "ConnectTimeout=15" in args
            assert args[-2] == "lightning-host"
            assert args[-1] == f"test -d {persisted_remote}"
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "ssh" and args[-1].startswith("test -f "):
            name = args[-1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                args,
                0 if (remote_source / name).is_file() else 1,
                stdout="",
                stderr="",
            )
        if args[0] == "ssh" and "find evals" in args[-1]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "scp":
            assert "BatchMode=yes" in args
            assert "PasswordAuthentication=no" in args
            assert "KbdInteractiveAuthentication=no" in args
            assert "ConnectTimeout=15" in args
            assert "-p" in args
            assert args[-2].startswith(f"lightning-host:{persisted_remote}/")
            name = args[-2].rsplit("/", 1)[-1]
            shutil.copy2(remote_source / name, Path(args[-1]))
            if name == ARTIFACT_COMPONENT_RESPONSE_VALIDATION:
                Path(args[-1]).chmod(0o444)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(lightning_batch_jobs.subprocess, "run", fake_run)
    spec = make_official_component_response_spec(
        name="component_response",
        baseline_archive_path="/repo/baseline.zip",
        perturbation_plan_path="/repo/plan.json",
        repo_dir="/repo",
        upstream_dir="/upstream",
        output_dir="/teamspace/studios/this_studio/pact/remote/component",
        local_artifact_dir=str(mirror),
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    validation = client.harvest_ssh_component_response_artifacts(
        job_name="component_response",
        ssh_target="lightning-host",
        require_passed=True,
    )

    assert calls[0][0] == "ssh"
    assert calls[1][0] == "scp"
    assert validation["baseline_archive_sha256"] == expected_sha
    assert validation["ssh_source"]["remote_dir"] == persisted_remote
    assert (mirror / ARTIFACT_COMPONENT_RESPONSE_VALIDATION).is_file()
    assert os.access(mirror / ARTIFACT_COMPONENT_RESPONSE_VALIDATION, os.W_OK)
    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "HARVESTED"
    assert record["status_history"][-1]["source"] == "ssh_component_response_artifact_validation"


def test_direct_ssh_harvest_rejects_bare_lightning_host(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not bare ssh\\.lightning\\.ai"):
        lightning_batch_jobs.mirror_ssh_artifact_dir(
            ssh_target="ssh.lightning.ai",
            remote_dir="/teamspace/jobs/run/artifacts/pact/out",
            mirror_dir=tmp_path / "mirror",
        )


def test_sdk_import_disables_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTNING_DISABLE_VERSION_CHECK", raising=False)

    class FakeJob:
        @classmethod
        def run(cls, **kwargs):
            return cls()

    class FakeLightningSdkModule:
        Job = FakeJob

    monkeypatch.setitem(sys.modules, "lightning_sdk", FakeLightningSdkModule())
    returned = LightningBatchJobsClient._import_job_cls()
    assert returned is FakeJob
    assert os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] == "1"


def test_batch_job_cli_sets_lightning_version_check_disable_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIGHTNING_DISABLE_VERSION_CHECK", raising=False)
    import runpy

    runpy.run_path(str(CLI), run_name="__lightning_cli_import_probe__")

    assert os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] == "1"


def test_lightning_doctor_payload_fails_on_supply_chain_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_lightning_cli_module(tmp_path)
    monkeypatch.setattr(
        mod,
        "_run_local_supply_chain_scan",
        lambda: {
            "ok": False,
            "returncode": 1,
            "payload": {"status": "FAIL", "violations": ["lightning==2.6.3"]},
        },
    )
    monkeypatch.setattr(mod, "_lightning_package_versions", lambda: {"lightning": None})

    payload = mod._doctor_payload(
        argparse.Namespace(
            ssh_target=None,
            ssh_bin="ssh",
            ssh_connect_timeout=1,
            require_ssh=False,
            remote_supply_chain=True,
            require_remote_supply_chain=False,
            repo_dir="/teamspace/studios/this_studio/pact",
            python_bin=".venv/bin/python",
            run_id="doctor_test",
            teamspace=None,
            org=None,
            user=None,
            machine_inventory=True,
            require_machine_inventory=False,
            cloud_account=[],
            machine=None,
            gpu_only=True,
        )
    )

    assert payload["status"] == "FAIL"
    assert payload["failed_checks"] == ["local_supply_chain"]
    assert payload["checks"]["ssh_auth"]["status"] == "skipped"


def test_lightning_doctor_payload_passes_with_required_skips_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_lightning_cli_module(tmp_path)
    monkeypatch.setattr(
        mod,
        "_run_local_supply_chain_scan",
        lambda: {"ok": True, "returncode": 0, "payload": {"status": "OK"}},
    )
    monkeypatch.setattr(mod, "_lightning_package_versions", lambda: {"lightning": None})

    payload = mod._doctor_payload(
        argparse.Namespace(
            ssh_target=None,
            ssh_bin="ssh",
            ssh_connect_timeout=1,
            require_ssh=False,
            remote_supply_chain=True,
            require_remote_supply_chain=False,
            repo_dir="/teamspace/studios/this_studio/pact",
            python_bin=".venv/bin/python",
            run_id="doctor_test",
            teamspace=None,
            org=None,
            user=None,
            machine_inventory=True,
            require_machine_inventory=False,
            cloud_account=[],
            machine=None,
            gpu_only=True,
        )
    )

    assert payload["status"] == "OK"
    assert payload["checks"]["local_supply_chain"]["ok"] is True
    assert payload["checks"]["remote_supply_chain"]["status"] == "skipped"


def test_machine_rows_filter_provider_instance_names_without_sdk_enum() -> None:
    module = _load_lightning_cli_module(REPO_ROOT)
    rows = [
        {
            "family": "T4",
            "instance_type": "g4dn.2xlarge",
            "name": "g4dn.2xlarge",
            "slug": "lit-t4-1",
        },
        {
            "family": "T4",
            "instance_type": "g4dn.xlarge",
            "name": "g4dn.xlarge",
            "slug": "lit-t4-1-small",
        },
    ]

    assert module._is_sdk_machine_filter("T4") is True
    assert module._is_sdk_machine_filter("g4dn.2xlarge") is False
    assert module._filter_machine_rows(
        rows,
        machine="g4dn.2xlarge",
        sdk_filtered=False,
    ) == [rows[0]]
    assert module._filter_machine_rows(rows, machine="T4", sdk_filtered=True) == rows


def test_batch_job_cli_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
            "--state-path",
            str(tmp_path / "jobs.json"),
            "--job-name",
            "cli-dry",
            "--archive",
            "/repo/archive.zip",
            "--repo-dir",
            "/repo",
            "--upstream-dir",
            "/upstream",
            "--studio",
            "pact",
            "--expected-archive-sha256",
            "d" * 64,
            "--expected-archive-size-bytes",
            "789",
            "--queue-metadata",
            "lane=pfp16",
            "--adjudicate",
            "--baseline-score",
            "1.2",
            "--predicted-band",
            "1.0",
            "1.4",
            "--regression-threshold",
            "1.6",
            "--max-posenet-dist",
            "0.006",
            "--max-segnet-dist",
            "0.004",
            "--reference-posenet-dist",
            "0.003",
            "--reference-segnet-dist",
            "0.002",
            "--max-posenet-relative",
            "1.5",
            "--max-segnet-relative",
            "1.2",
            "--component-reference-label",
            "frontier",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN"
    assert payload["spec"]["role"] == "exact_cuda_eval"
    assert payload["queue"]["expected_archive_sha256"] == "d" * 64
    assert payload["queue"]["expected_archive_size_bytes"] == 789
    assert payload["queue"]["queue_metadata"]["lane"] == "pfp16"
    assert "scripts/adjudicate_contest_auth_eval.py" in payload["spec"]["command"]
    assert payload["spec"]["adjudication"]["max_posenet_dist"] == 0.006
    assert payload["spec"]["adjudication"]["max_segnet_dist"] == 0.004
    assert payload["spec"]["adjudication"]["baseline_posenet_dist"] == 0.003
    assert payload["spec"]["adjudication"]["baseline_segnet_dist"] == 0.002
    assert payload["spec"]["adjudication"]["max_posenet_relative"] == 1.5
    assert payload["spec"]["adjudication"]["max_segnet_relative"] == 1.2
    assert payload["spec"]["adjudication"]["component_reference_label"] == "frontier"
    assert payload["spec"]["adjudication"]["allow_component_gate_forensic_success"] is True
    assert "--allow-component-gate-forensic-success" in payload["spec"]["command"]


def test_batch_job_cli_can_force_component_gate_job_failure(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
            "--state-path",
            str(tmp_path / "jobs.json"),
            "--job-name",
            "cli-dry",
            "--archive",
            "/repo/archive.zip",
            "--repo-dir",
            "/repo",
            "--upstream-dir",
            "/upstream",
            "--studio",
            "pact",
            "--expected-archive-sha256",
            "d" * 64,
            "--expected-archive-size-bytes",
            "789",
            "--adjudicate",
            "--baseline-score",
            "1.2",
            "--predicted-band",
            "1.0",
            "1.4",
            "--regression-threshold",
            "1.6",
            "--fail-job-on-component-gate",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["spec"]["adjudication"]["allow_component_gate_forensic_success"] is False
    assert "--allow-component-gate-forensic-success" not in payload["spec"]["command"]


def test_batch_job_cli_component_sensitivity_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "component-sensitivity",
            "--state-path",
            str(tmp_path / "jobs.json"),
            "--job-name",
            "sens-dry",
            "--baseline-archive",
            "/repo/baseline.zip",
            "--repo-dir",
            "/repo",
            "--upstream-dir",
            "/repo/upstream",
            "--studio",
            "pact",
            "--expected-baseline-archive-sha256",
            "e" * 64,
            "--expected-baseline-archive-size-bytes",
            "456",
            "--queue-metadata",
            "lane=beta_diagnostic",
            "--pair-batch",
            "4",
            "--response-top-k",
            "8",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN"
    assert payload["spec"]["role"] == "diagnostic_component_sensitivity"
    assert payload["queue"]["expected_archive_sha256"] == "e" * 64
    assert payload["queue"]["expected_archive_size_bytes"] == 456
    assert payload["queue"]["queue_metadata"]["lane"] == "beta_diagnostic"
    command = payload["spec"]["command"]
    assert "experiments/profile_component_sensitivity.py" in command
    assert "--device cuda" in command
    assert "--all-pairs" in command
    assert "--pair-batch 4" in command
    assert "--response-top-k 8" in command
    assert "promotion_eligible" in command
    assert "score_claim" in command


def test_batch_job_cli_has_refresh_status_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "refresh-status",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "--sdk-job-name" in result.stdout
    assert "--teamspace" in result.stdout


def test_batch_job_cli_has_harvest_ssh_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "harvest-ssh",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "--ssh-target" in result.stdout
    assert "--remote-artifact-dir" in result.stdout
    assert "--require-adjudication" in result.stdout
    assert "--ssh-connect-timeout" in result.stdout


def test_batch_job_cli_has_stateful_component_response_harvest_and_remote_preflight(
    tmp_path: Path,
) -> None:
    component = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "component-response",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert component.returncode == 0, component.stderr
    assert "--remote-preflight-ssh-target" in component.stdout

    sensitivity = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "component-sensitivity",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert sensitivity.returncode == 0, sensitivity.stderr
    assert "--baseline-archive" in sensitivity.stdout
    assert "--pair-weights" in sensitivity.stdout
    assert "--remote-preflight-ssh-target" in sensitivity.stdout

    harvest = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "harvest-component-response-ssh",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert harvest.returncode == 0, harvest.stderr
    assert "--job-name" in harvest.stdout
    assert "--state-path" in harvest.stdout
    assert "--remote-artifact-dir" in harvest.stdout


def test_remote_supply_chain_preflight_runs_strict_scan_before_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    calls: list[list[str]] = []

    def fake_auth_ready(*args: object, **kwargs: object) -> None:
        calls.append(["auth"])

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        assert args[0] == "ssh"
        assert args[1:7] == [
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
        ]
        assert args[-2] == "lightning-host"
        assert "scripts/scan_lightning_supply_chain.py" in args[-1]
        assert "--quiet --strict" in args[-1]
        assert ".omx/state/lightning_batch_pre_submit_supply_chain_component_response.json" in args[-1]
        return subprocess.CompletedProcess(args, 0, stdout='{"status":"OK"}\n', stderr="")

    monkeypatch.setattr(module, "_ensure_ssh_auth_ready", fake_auth_ready)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_remote_supply_chain_preflight(
        ssh_target="lightning-host",
        job_name="component_response",
        repo_dir="/teamspace/studios/this_studio/pact",
        python_bin=".venv/bin/python",
    )

    assert calls[0] == ["auth"]
    assert calls[1][0] == "ssh"


def test_batch_job_cli_ssh_preflight_reports_identity_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    key = tmp_path / "lightning_rsa"
    pub = tmp_path / "lightning_rsa.pub"
    key.write_text("private placeholder\n")
    pub.write_text("public placeholder\n")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=(
                    "host lightning-pact\n"
                    "hostname ssh.lightning.ai\n"
                    "user s_example\n"
                    f"identityfile {key}\n"
                    "identitiesonly yes\n"
                    "stricthostkeychecking accept-new\n"
                ),
                stderr="",
            )
        if args[0] == "ssh" and args[-1] == "true":
            return subprocess.CompletedProcess(
                args,
                255,
                stdout="",
                stderr="s_example@ssh.lightning.ai: Permission denied (publickey).\n",
            )
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as excinfo:
        module._ensure_ssh_auth_ready("lightning-pact", connect_timeout=5)

    message = str(excinfo.value)
    assert "Permission denied (publickey)" in message
    assert str(key.resolve()) in message
    assert str(pub.resolve()) in message
    assert "Lightning Studio/account SSH keys" in message
    assert "bare lightning CLI" in message


def test_batch_job_cli_ssh_preflight_rejects_bare_provider_host(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)

    with pytest.raises(SystemExit, match="bare ssh.lightning.ai"):
        module._ensure_ssh_auth_ready("ssh.lightning.ai")


def test_batch_job_cli_ssh_preflight_rejects_disabled_host_key_checking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    key = tmp_path / "lightning_rsa"
    pub = tmp_path / "lightning_rsa.pub"
    key.write_text("private placeholder\n")
    pub.write_text("public placeholder\n")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=(
                    "host lightning-pact\n"
                    "hostname ssh.lightning.ai\n"
                    "user s_example\n"
                    f"identityfile {key}\n"
                    "identitiesonly yes\n"
                    "stricthostkeychecking false\n"
                ),
                stderr="",
            )
        if args[0] == "ssh" and args[-1] == "true":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as excinfo:
        module._ensure_ssh_auth_ready("lightning-pact")

    assert "StrictHostKeyChecking is disabled" in str(excinfo.value)


def test_batch_job_cli_has_list_machines_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "list-machines",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "--cloud-account" in result.stdout
    assert "--gpu-only" in result.stdout
