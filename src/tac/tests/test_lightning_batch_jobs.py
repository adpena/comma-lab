# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import pytest
import torch

import tac.deploy.lightning.batch_jobs as lightning_batch_jobs
from tac.deploy.lightning.batch_jobs import (
    ARTIFACT_COMPONENT_RESPONSE_INPUTS,
    ARTIFACT_COMPONENT_RESPONSE_SUMMARY,
    ARTIFACT_COMPONENT_RESPONSE_VALIDATION,
    ARTIFACT_COMPONENT_SENSITIVITY_INPUTS,
    ARTIFACT_COMPONENT_SENSITIVITY_RUN,
    ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY,
    ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION,
    ARTIFACT_DALI_BOOTSTRAP,
    ARTIFACT_DALI_REQUIREMENTS,
    ARTIFACT_INFLATE_RUNTIME_BOOTSTRAP,
    ARTIFACT_INFLATE_RUNTIME_STATIC_PREFLIGHT,
    ARTIFACT_INFRA_FAILURE,
    ARTIFACT_METADATA,
    ARTIFACT_RUNNER_PREFLIGHT,
    ARTIFACT_SUPPLY_CHAIN_SCAN,
    ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
    ARTIFACT_VALIDATION,
    CANONICAL_ARTIFACT_FILES,
    COMPONENT_RESPONSE_CURVE_FILES,
    COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES,
    COMPONENT_SENSITIVITY_CURVE_FILES,
    COMPONENT_SENSITIVITY_HOLDOUT_MAP_FILES,
    COMPONENT_SENSITIVITY_MAP_FILES,
    LIGHTNING_EMPTY_ARTIFACT_INFRA_TERMINAL_CLASS,
    LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
    LightningAdjudicationSpec,
    LightningBatchJobsClient,
    LightningBatchJobSpec,
    LightningStudioCloudAccountMismatchError,
    archive_identity,
    default_exact_eval_local_artifact_dir,
    default_exact_eval_output_dir,
    diagnostic_component_sensitivity_command,
    exact_cuda_eval_command,
    lightning_sdk_artifact_path,
    lightning_sdk_job_name,
    lightning_sdk_persisted_studio_output_dir,
    make_diagnostic_component_sensitivity_spec,
    make_exact_eval_spec,
    make_official_component_response_spec,
    mirror_local_artifact_dir,
    mirror_local_component_sensitivity_artifact_dir,
    official_component_response_command,
    validate_local_artifact_dir,
    validate_local_component_response_artifact_dir,
    validate_local_component_sensitivity_artifact_dir,
    validate_studio_machine_class_pair,
)
from tac.sensitivity_map import save_sensitivity_map

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "scripts" / "launch_lightning_batch_job.py"
EXPECTED_SHA = "a" * 64
EXPECTED_BYTES = 123


def _manifest_entry(path: str, *, payload: bytes | None = None) -> dict[str, object]:
    raw = payload if payload is not None else path.encode("utf-8")
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _repo_manifest_entry(repo_root: Path, rel: str) -> dict[str, object]:
    path = repo_root / rel
    if path.is_file():
        return _manifest_entry(rel, payload=path.read_bytes())
    return _manifest_entry(rel)


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


def _adjudication(required_device: str = "cuda") -> LightningAdjudicationSpec:
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
        required_device=required_device,
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
                "score_axis": "contest_cuda",
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


def _write_cpu_artifact_dir(root: Path, *, promotion_eligible: bool = False) -> tuple[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
    archive.write_bytes(b"fake cpu archive bytes")
    identity = archive_identity(archive)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "job_name": "artifact-cpu-job",
        "role": "exact_cpu_eval",
        "expected_archive_sha256": identity["archive_sha256"],
        "expected_archive_size_bytes": identity["archive_size_bytes"],
        "queue_metadata": {"lane": "tt5l"},
        "adjudication": {
            "required_device": "cpu",
            "required_samples": 600,
            "provenance_name": "adjudication_provenance.json",
            "result_copy_name": "contest_auth_eval.adjudicated.json",
        },
        "score_source": "contest_auth_eval.json:score_recomputed_from_components",
        "status_source": "lightning_sdk_job_attributes",
    }
    (root / "lightning_queue_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    contest_payload = {
        "score_recomputed_from_components": 1.25,
        "final_score": 1.25,
        "avg_posenet_dist": 0.003,
        "avg_segnet_dist": 0.004,
        "n_samples": 600,
        "archive_size_bytes": identity["archive_size_bytes"],
        "score_axis": "contest_cpu",
        "evidence_grade": "contest-CPU",
        "score_claim_valid": True,
        "cpu_leaderboard_reproduction_eligible": True,
        "provenance": {
            "device": "cpu",
            "archive_sha256": identity["archive_sha256"],
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
    }
    (root / "contest_auth_eval.json").write_text(
        json.dumps(contest_payload, indent=2, sort_keys=True) + "\n"
    )
    (root / "auth_eval.log").write_text("cpu axis score\n")
    (root / "report.txt").write_text("cpu report text\n")
    (root / "eval_provenance.json").write_text(
        json.dumps({"device": "cpu", "archive_sha256": identity["archive_sha256"]}) + "\n"
    )
    (root / ARTIFACT_RUNNER_PREFLIGHT).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_exact_eval_runner_preflight",
                "requested_device": "cpu",
                "torch_import_ok": True,
                "platform_system": "Linux",
                "platform_machine": "x86_64",
                "cpu_contest_platform": True,
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
    (root / "contest_auth_eval.adjudicated.json").write_text(
        json.dumps(contest_payload, indent=2, sort_keys=True) + "\n"
    )
    (root / "adjudication_provenance.json").write_text(
        json.dumps(
            {
                "score_axis": "contest_cpu",
                "contest_cpu_archive_sha256": identity["archive_sha256"],
                "contest_cpu_archive_bytes": identity["archive_size_bytes"],
                "contest_cpu_score_source": "contest_auth_eval.json:score_recomputed_from_components",
                "contest_cpu_device": "cpu",
                "score_tag": "[contest-CPU]",
                "promotion_eligible": promotion_eligible,
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


def _write_component_sensitivity_artifact_dir(
    root: Path,
    *,
    sensitivity_source: str = "fisher_proxy",
) -> tuple[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    direct_fd = sensitivity_source == "direct_renderer_cuda_finite_difference_component_response"
    baseline_sha = "b" * 64
    baseline_bytes = 456
    metadata: dict[str, object] = {
        "schema_version": 1,
        "job_name": "component-sensitivity-job",
        "role": "diagnostic_component_sensitivity",
        "expected_archive_sha256": baseline_sha,
        "expected_archive_size_bytes": baseline_bytes,
        "expected_baseline_archive_sha256": baseline_sha,
        "expected_baseline_archive_size_bytes": baseline_bytes,
        "queue_metadata": {"lane": "component_sensitivity"},
        "adjudication": None,
        "score_claim": False,
        "promotion_eligible": False,
        "score_source": "none:diagnostic_component_sensitivity_non_promotable",
        "status_source": "lightning_sdk_job_attributes",
    }
    (root / "lightning_queue_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / ARTIFACT_COMPONENT_SENSITIVITY_INPUTS).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_diagnostic_component_sensitivity_input_preflight",
                "baseline_archive": {
                    "path": "/repo/baseline.zip",
                    "bytes": baseline_bytes,
                    "sha256": baseline_sha,
                    "zip_member_count": 3,
                },
                "extracted_dir": "/out/extracted",
                "extracted_members": {
                    "renderer.bin": {"path": "/out/extracted/renderer.bin", "bytes": 111, "sha256": "c" * 64},
                    "masks.mkv": {"path": "/out/extracted/masks.mkv", "bytes": 222, "sha256": "d" * 64},
                    "optimized_poses.bin": {
                        "path": "/out/extracted/optimized_poses.bin",
                        "bytes": 333,
                        "sha256": "e" * 64,
                    },
                },
                "score_claim": False,
                "promotion_eligible": False,
                "diagnostic": True,
            },
            indent=2,
        )
        + "\n"
    )
    (root / ARTIFACT_COMPONENT_SENSITIVITY_RUN).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "lightning_diagnostic_component_sensitivity_run_metadata",
                "role": "diagnostic_component_sensitivity",
                "profile_argv": [
                    ".venv/bin/python",
                    "-u",
                    "experiments/profile_component_sensitivity.py",
                    "--device",
                    "cuda",
                    *(
                        [
                            "--promotion-finite-difference",
                            "--finite-difference-epsilon",
                            "0.001",
                            "--all-pairs",
                        ]
                        if direct_fd
                        else []
                    ),
                ],
                "diagnostic": True,
                "score_claim": False,
                "promotion_eligible": False,
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
    if direct_fd:
        calibration_pairs = [
            {"video": 0, "pair_index": i, "t": 2 * i, "t1": 2 * i + 1}
            for i in range(480)
        ]
        holdout_pairs = [
            {"video": 0, "pair_index": i, "t": 2 * i, "t1": 2 * i + 1}
            for i in range(480, 600)
        ]
    else:
        calibration_pairs = [{"video": 0, "pair_index": 0, "t": 0, "t1": 1}]
        holdout_pairs = [{"video": 0, "pair_index": 1, "t": 2, "t1": 3}]
    (root / "sample_plan.json").write_text(
        json.dumps(
            {
                "split_seed": 20260430,
                "split_hash": "s" * 64,
                "calibration_pairs": calibration_pairs,
                "holdout_pairs": holdout_pairs,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "stability.json").write_text(
        json.dumps({"passed": False, "component_passed": {"posenet": False, "segnet": False, "combined": False}})
        + "\n"
    )
    (root / "perturbation_basis_v1.json").write_text(json.dumps({"format": "perturbation_basis_v1"}) + "\n")
    response_curve_paths = {}
    map_paths = {}
    for component in ("posenet", "segnet", "combined"):
        map_name = f"{component}_sensitivity_map.pt"
        holdout_map_name = f"{component}_holdout_sensitivity_map.pt"
        curve_name = f"{component}_response_curve.json"
        map_metadata = {
            "device": "cuda",
            "component": component,
            "scorer_target": component,
            "score_claim": False,
            "promotion_eligible": False,
            "official_component_response": False,
            "canonical_scorer_path": False,
            "sensitivity_source": sensitivity_source,
            "finite_difference_shard": None,
        }
        save_sensitivity_map(
            root / map_name,
            {"renderer.layer.weight": torch.ones(2)},
            metadata=map_metadata,
        )
        save_sensitivity_map(
            root / holdout_map_name,
            {"renderer.layer.weight": torch.ones(2)},
            metadata={**map_metadata, "split": "holdout"},
        )
        map_paths[component] = str(root / map_name)
        response_curve_paths[component] = str(root / curve_name)
        (root / curve_name).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "component": component,
                    "device": "cuda",
                    "score_claim": False,
                    "official_component_response": False,
                    "canonical_scorer_path": False,
                    "component_response_path": "direct_renderer_tensor_inprocess_scorer",
                    "promotion_eligible": False,
                    "sensitivity_source": sensitivity_source,
                    "points": [
                        {"epsilon": -0.001, "value": 0.9, "baseline": 1.0, "delta": -0.1},
                        {"epsilon": 0.0, "value": 1.0, "baseline": 1.0, "delta": 0.0},
                        {"epsilon": 0.001, "value": 1.1, "baseline": 1.0, "delta": 0.1},
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    (root / ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "experiments/profile_component_sensitivity.py",
                "device": "cuda",
                "evidence_grade": (
                    "diagnostic_cuda_direct_renderer_finite_difference"
                    if direct_fd
                    else "diagnostic_cuda"
                ),
                "sensitivity_source": sensitivity_source,
                "promotion_requested": direct_fd,
                "finite_difference_epsilon": 0.001 if direct_fd else None,
                "finite_difference_shard": None,
                "finite_difference_merge": None,
                "certification_handoff_eligible": direct_fd,
                "score_claim": False,
                "official_component_response": False,
                "canonical_scorer_path": False,
                "component_response_path": "direct_renderer_tensor_inprocess_scorer",
                "promotion_eligible": False,
                "map_paths": map_paths,
                "response_curve_paths": response_curve_paths,
                "n_pairs_total": 600,
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
    ensure_uv_idx = command.index("scripts/ensure_remote_uv.sh --symlink-system")
    dali_idx = command.index("LIGHTNING_VENV_LOCK=.omx/state/lightning_exact_eval_venv.lock")
    assert ensure_uv_idx < dali_idx
    assert 'export PATH="$(dirname "$UV_BIN"):$PATH"' in command
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
    assert "LIGHTNING_INFLATE_RUNTIME_BOOTSTRAP_OK" in command
    assert "LIGHTNING_INFLATE_RUNTIME_STATIC_PREFLIGHT_OK" in command
    assert "pr81_router_actions" in command
    assert "nvidia.dali.fn" in command
    assert "nvidia_dali_fn_module" in command
    assert ARTIFACT_DALI_BOOTSTRAP in command
    assert "lightning_dali_requirements.txt" in command
    assert ARTIFACT_RUNNER_PREFLIGHT in command
    assert ARTIFACT_INFLATE_RUNTIME_BOOTSTRAP in command
    assert ARTIFACT_INFLATE_RUNTIME_STATIC_PREFLIGHT in command
    assert ARTIFACT_SUPPLY_CHAIN_SCAN in command
    assert "--index-url" not in command
    assert "--extra-index-url" not in command
    assert "grep" not in command
    assert "re.search" not in command
    assert "expected_sha = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'" in command
    assert "expected_sha = null" not in command


def test_exact_cuda_eval_command_installs_declared_external_inflate_deps() -> None:
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/artifacts/archive.zip",
        upstream_dir="/upstream",
        output_dir="/out",
        adjudication=_adjudication(),
        env={
            "INFLATE_TORCH_SPEC": "torch==2.5.1+cu124",
            "INFLATE_TORCHVISION_SPEC": "torchvision==0.20.1+cu124",
            "INFLATE_BROTLI_SPEC": "brotli==1.2.0",
            "INFLATE_AV_SPEC": "av==17.0.1",
        },
        **_expected_archive_kwargs(),
    )

    assert "lightning_exact_eval_inflate_runtime_bootstrap" in command
    assert "torch==2.5.1+cu124" in command
    assert "torchvision==0.20.1+cu124" in command
    assert "brotli==1.2.0" in command
    assert "av==17.0.1" in command
    assert "uv, 'pip', 'install'" in command
    assert "export PATH=$(dirname .venv/bin/python):$PATH" in command
    assert command.index("LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK") < command.index(
        "LIGHTNING_INFLATE_RUNTIME_BOOTSTRAP_OK"
    )
    assert command.index("LIGHTNING_INFLATE_RUNTIME_BOOTSTRAP_OK") < command.index(
        "experiments/contest_auth_eval.py"
    )


def test_exact_cpu_eval_command_is_axis_explicit_and_not_cuda_preflighted() -> None:
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/artifacts/archive.zip",
        upstream_dir="/upstream",
        output_dir="/out",
        adjudication=_adjudication(required_device="cpu"),
        eval_device="cpu",
        **_expected_archive_kwargs(),
    )

    assert "experiments/contest_auth_eval.py" in command
    assert "--device cpu" in command
    assert "--device cuda" not in command
    assert "LIGHTNING_EXACT_CPU_EVAL_JSON_OK" in command
    assert "payload.get('score_axis') == 'contest_cpu'" in command
    assert "LIGHTNING_RUNNER_CPU_PREFLIGHT_OK" in command
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in command
    assert "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" not in command
    assert "export INFLATE_REQUIRE_CUDA=1" not in command
    assert "--required-device cpu" in command
    assert "scripts/scan_lightning_supply_chain.py" in command
    assert "LIGHTNING_INFLATE_RUNTIME_STATIC_PREFLIGHT_OK" in command


def test_exact_cuda_eval_command_can_emit_component_trace() -> None:
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/artifacts/archive.zip",
        upstream_dir="/upstream",
        output_dir="/out",
        adjudication=_adjudication(),
        component_trace=True,
        component_trace_top_k=13,
        **_expected_archive_kwargs(),
    )

    assert "experiments/contest_component_trace.py" in command
    assert "--submission-dir /out/eval_work" in command
    assert "--contest-auth-eval-json /out/contest_auth_eval.json" in command
    assert "--output-json /out/component_trace.json" in command
    assert "--top-k 13" in command
    assert "LIGHTNING_COMPONENT_TRACE_STATUS" in command
    assert "component_trace_status.json" in command
    assert "diagnostic_unavailable_or_invalid" in command
    assert "component_trace_rc=${PIPESTATUS[0]}" in command
    assert command.index("experiments/contest_auth_eval.py") < command.index(
        "experiments/contest_component_trace.py"
    )
    assert command.index("experiments/contest_component_trace.py") < command.index(
        "scripts/adjudicate_contest_auth_eval.py"
    )


def test_exact_cpu_eval_rejects_component_trace() -> None:
    with pytest.raises(ValueError, match="component_trace is CUDA-only"):
        exact_cuda_eval_command(
            repo_dir="/repo",
            archive_path="/artifacts/archive.zip",
            upstream_dir="/upstream",
            output_dir="/out",
            adjudication=_adjudication(required_device="cpu"),
            component_trace=True,
            eval_device="cpu",
            **_expected_archive_kwargs(),
        )


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
    ensure_uv_idx = command.index("scripts/ensure_remote_uv.sh --symlink-system")
    dali_idx = command.index("LIGHTNING_VENV_LOCK=.omx/state/lightning_exact_eval_venv.lock")
    assert ensure_uv_idx < dali_idx
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
    assert "--response-epsilons=-0.001,0.0,0.001" in command
    assert "--response-epsilons -0.001,0.0,0.001" not in command
    profile_line = next(
        line for line in command.splitlines()
        if line.startswith(".venv/bin/python -u experiments/profile_component_sensitivity.py")
    )
    assert "--all-pairs" not in profile_line
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
    ensure_uv_idx = command.index("scripts/ensure_remote_uv.sh --symlink-system")
    dali_idx = command.index("LIGHTNING_VENV_LOCK=.omx/state/lightning_exact_eval_venv.lock")
    assert ensure_uv_idx < dali_idx
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in command
    assert "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" in command
    assert "--require-hashes" in command
    assert "--no-deps" in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_INPUTS in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_RUN in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY in command
    assert ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION in command
    for name in COMPONENT_SENSITIVITY_MAP_FILES + COMPONENT_SENSITIVITY_HOLDOUT_MAP_FILES + COMPONENT_SENSITIVITY_CURVE_FILES:
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


def test_diagnostic_component_sensitivity_command_can_request_direct_finite_difference() -> None:
    command = diagnostic_component_sensitivity_command(
        repo_dir="/repo",
        baseline_archive_path="/repo/baseline.zip",
        upstream_dir="/repo/upstream",
        output_dir="/out",
        expected_baseline_archive_sha256="b" * 64,
        expected_baseline_archive_size_bytes=456,
        promotion_finite_difference=True,
        finite_difference_epsilon=0.002,
        finite_difference_shard_index=3,
        finite_difference_shard_count=16,
    )

    assert "--promotion-finite-difference" in command
    assert "--finite-difference-epsilon 0.002" in command
    assert "--finite-difference-shard-index 3" in command
    assert "--finite-difference-shard-count 16" in command
    assert "--all-pairs" in command
    assert "--pair-weights" not in command


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
        env=[],
        machine="g6e.4xlarge",
    )
    with pytest.raises(SystemExit, match="requires --source-manifest"):
        module._validate_exact_eval_submit_inputs(args)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [_manifest_entry("other.zip")]}) + "\n")
    args.source_manifest = str(manifest)
    with pytest.raises(SystemExit, match="does not include archive artifact"):
        module._validate_exact_eval_submit_inputs(args)

    manifest.write_text(json.dumps({"files": [_manifest_entry("archive.zip")]}) + "\n")
    with pytest.raises(SystemExit, match="inflate runtime closure"):
        module._validate_exact_eval_submit_inputs(args)

    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                ]
            }
        )
        + "\n"
    )
    module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_submit_requires_source_manifest_entry_byte_sha(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "manifest.json"
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
        env=[],
        machine="g6e.4xlarge",
    )
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "archive.zip"},
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="lacks integer bytes"):
        module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_submit_rejects_source_manifest_byte_sha_drift(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"actual archive bytes")
    manifest = tmp_path / "manifest.json"
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
        env=[],
        machine="g6e.4xlarge",
    )
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip", payload=b"stale archive bytes"),
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="source manifest byte/SHA mismatch"):
        module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_submit_requires_nested_external_inflate_runtime(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    runtime = tmp_path / "experiments/results/candidate/submission_dir"
    (runtime / "src").mkdir(parents=True)
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n")
    (runtime / "inflate.py").write_text("from src.model import X\n")
    (runtime / "src" / "model.py").write_text("X = 1\n")
    (runtime / "src" / "codec.py").write_text("Y = 2\n")
    (runtime / "__pycache__").mkdir()
    (runtime / "__pycache__" / "inflate.cpython.pyc").write_bytes(b"cache")
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(tmp_path / "manifest.json"),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
        env=[],
        machine="g6e.4xlarge",
        inflate_sh="/repo/experiments/results/candidate/submission_dir/inflate.sh",
    )
    manifest = Path(args.source_manifest)
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/inflate.sh"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/inflate.py"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/src/model.py"),
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="inflate runtime closure"):
        module._validate_exact_eval_submit_inputs(args)

    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/inflate.sh"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/inflate.py"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/src/model.py"),
                    _repo_manifest_entry(tmp_path, "experiments/results/candidate/submission_dir/src/codec.py"),
                ]
            }
        )
        + "\n"
    )
    module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_queue_metadata_binds_source_manifest_runtime_closure(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                ]
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        cmd="exact-eval",
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
        env=[],
        machine="g6e.4xlarge",
        inflate_sh="submissions/robust_current/inflate.sh",
    )

    metadata = module._queue_metadata_from_args(args)

    assert metadata["source_manifest_sha256"]
    closure = metadata["source_manifest_runtime_closure"]
    assert closure["archive"]["path"] == "archive.zip"
    assert closure["runtime_file_count"] == 2
    assert closure["source_manifest_sha256"] == metadata["source_manifest_sha256"]
    assert metadata["source_manifest_runtime_closure_sha256"] == closure["closure_sha256"]


def test_exact_eval_manifest_dispatch_gate_blocks_renderer_stack_without_pose_safety(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    archive = tmp_path / "experiments/results/candidate/archive.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"placeholder")
    (archive.parent / "manifest.json").write_text(
        json.dumps(
            {
                "exact_eval_dispatch_gate": {
                    "required": True,
                    "status": "missing_pose_safety_report",
                    "safe_for_exact_eval_dispatch": False,
                    "blockers": ["missing renderer transplant pose-safety preflight"],
                }
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        archive="/repo/experiments/results/candidate/archive.zip",
        repo_dir="/repo",
    )

    with pytest.raises(SystemExit, match="exact_eval_dispatch_gate"):
        module._validate_archive_manifest_dispatch_gate(args)

    (archive.parent / "manifest.json").write_text(
        json.dumps(
            {
                "exact_eval_dispatch_gate": {
                    "required": True,
                    "status": "pass",
                    "safe_for_exact_eval_dispatch": True,
                    "blockers": [],
                }
            }
        )
        + "\n"
    )
    module._validate_archive_manifest_dispatch_gate(args)


def test_exact_eval_submit_requires_metadata_baseline_json_closure(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                ]
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=["baseline_json=baseline/contest_auth_eval.json"],
        env=[],
        machine="g6e.4xlarge",
    )

    with pytest.raises(SystemExit, match="metadata baseline_json artifact"):
        module._validate_exact_eval_submit_inputs(args)

    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                    _manifest_entry("baseline/contest_auth_eval.json"),
                ]
            }
        )
        + "\n"
    )
    module._validate_exact_eval_submit_inputs(args)


def test_exact_eval_submit_rejects_symbolic_non_t4_studio_machine(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(dry_run=False, studio="pact", machine="RTXP_6000", cloud_account=None)

    with pytest.raises(SystemExit, match="symbolic accelerator 'RTXP_6000'"):
        module._validate_studio_machine_for_submit(args)

    args.machine = "g7e.4xlarge"
    module._validate_studio_machine_for_submit(args)

    args.machine = "T4"
    module._validate_studio_machine_for_submit(args)

    args.machine = "T4_X_4"
    module._validate_studio_machine_for_submit(args)


def test_exact_eval_submit_rejects_unsupported_concrete_studio_machine(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(dry_run=False, studio="pact", machine="g4dn.4xlarge", cloud_account=None)

    with pytest.raises(SystemExit, match="unsupported Lightning Studio machine/class pair"):
        module._validate_studio_machine_for_submit(args)

    for machine in ("g4dn.xlarge", "g4dn.2xlarge", "g4dn.12xlarge", "g6e.4xlarge"):
        args.machine = machine
        module._validate_studio_machine_for_submit(args)

    with pytest.raises(ValueError, match="g4dn\\.4xlarge"):
        validate_studio_machine_class_pair("g4dn.4xlarge")


def test_exact_eval_submit_rejects_gcp_machine_without_cloud_account(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(dry_run=False, studio="pact", machine="n1-standard-8", cloud_account=None)

    with pytest.raises(SystemExit, match="requires an explicit --cloud-account"):
        module._validate_studio_machine_for_submit(args)

    args.cloud_account = "gcp-lightning-public-prod"
    module._validate_studio_machine_for_submit(args)


def test_exact_eval_submit_requires_cdo1_manifest_pair_basis(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "source_manifest.json"
    archive_manifest = tmp_path / "candidate" / "manifest.json"
    archive_manifest.parent.mkdir(parents=True)
    archive_manifest.write_text(
        json.dumps(
            {
                "schema": "c067_decoded_delta_overlay_candidate_v1",
                "cdo1_overlay": {
                    "selected_pair_indices": [79, 153],
                },
            }
        )
        + "\n"
    )

    def write_source_manifest() -> None:
        manifest.write_text(
            json.dumps(
                {
                    "files": [
                        _manifest_entry("archive.zip"),
                        _manifest_entry("submissions/robust_current/inflate.sh"),
                        _manifest_entry("submissions/robust_current/config.env"),
                        _repo_manifest_entry(tmp_path, "candidate/manifest.json"),
                    ]
                }
            )
            + "\n"
        )

    write_source_manifest()
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[
            "archive_manifest=candidate/manifest.json",
            "pair_index_basis=half_frame_pair_index",
        ],
        env=[],
        machine="g6e.4xlarge",
    )

    with pytest.raises(SystemExit, match=r"cdo1_overlay\.pair_index_basis"):
        module._validate_exact_eval_submit_inputs(args)

    payload = json.loads(archive_manifest.read_text())
    payload["cdo1_overlay"]["pair_index_basis"] = "video_frame_pair_index"
    archive_manifest.write_text(json.dumps(payload) + "\n")
    write_source_manifest()
    with pytest.raises(SystemExit, match="does not match archive manifest"):
        module._validate_exact_eval_submit_inputs(args)

    payload["cdo1_overlay"]["pair_index_basis"] = "half_frame_pair_index"
    archive_manifest.write_text(json.dumps(payload) + "\n")
    write_source_manifest()
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
        env=[],
        machine="g6e.4xlarge",
    )

    with pytest.raises(SystemExit, match="path traversal"):
        module._validate_exact_eval_submit_inputs(args)


def test_t4_exact_eval_submit_requires_driver_compatible_torch_pin(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    _manifest_entry("archive.zip"),
                    _manifest_entry("submissions/robust_current/inflate.sh"),
                    _manifest_entry("submissions/robust_current/config.env"),
                ]
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        source_manifest=str(manifest),
        archive="/repo/archive.zip",
        repo_dir="/repo",
        queue_metadata=[],
        env=[],
        machine="g4dn.xlarge",
    )
    with pytest.raises(SystemExit, match="INFLATE_TORCH_SPEC"):
        module._validate_exact_eval_submit_inputs(args)

    args.env = ["INFLATE_TORCH_SPEC=torch==2.5.1+cu124"]
    with pytest.raises(SystemExit, match="INFLATE_TORCHVISION_SPEC"):
        module._validate_exact_eval_submit_inputs(args)

    args.env = [
        "INFLATE_TORCH_SPEC=torch==2.5.1+cu124",
        "INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124",
    ]
    with pytest.raises(SystemExit, match="UV_EXTRA_INDEX_URL"):
        module._validate_exact_eval_submit_inputs(args)

    args.env = [
        "INFLATE_TORCH_SPEC=torch==2.5.1+cu124",
        "INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124",
        "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124",
        "UV_INDEX_STRATEGY=unsafe-best-match",
    ]
    module._validate_exact_eval_submit_inputs(args)


def test_non_dry_run_studio_submit_requires_remote_preflight(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        remote_preflight_ssh_target=None,
        allow_skip_remote_preflight_reason=None,
    )
    with pytest.raises(SystemExit, match="missing --remote-preflight-ssh-target"):
        module._require_remote_preflight_for_submit(args, role="exact-eval")

    args.allow_skip_remote_preflight_reason = "externally attested image-backed repro path"
    module._require_remote_preflight_for_submit(args, role="exact-eval")

    args.allow_skip_remote_preflight_reason = None
    args.remote_preflight_ssh_target = "ssh.lightning.ai"
    with pytest.raises(SystemExit, match="bare ssh\\.lightning\\.ai"):
        module._require_remote_preflight_for_submit(args, role="exact-eval")


def test_non_dry_run_studio_submit_with_teamspace_requires_user_or_org(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    args = argparse.Namespace(
        dry_run=False,
        studio=None,
        image=None,
        teamspace=None,
        org=None,
        user=None,
    )
    with pytest.raises(SystemExit, match="requires explicit --studio or --image"):
        module._require_lightning_identity_for_studio_submit(args, role="exact-eval")

    args.image = "ghcr.io/example/exact-eval:latest"
    module._require_lightning_identity_for_studio_submit(args, role="exact-eval")

    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        image=None,
        teamspace=None,
        org=None,
        user=None,
    )
    with pytest.raises(SystemExit, match="requires --teamspace"):
        module._require_lightning_identity_for_studio_submit(args, role="exact-eval")

    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        image=None,
        teamspace="comma-lab",
        org=None,
        user=None,
    )
    with pytest.raises(SystemExit, match="requires --user or --org"):
        module._require_lightning_identity_for_studio_submit(args, role="exact-eval")

    args.user = "adpena"
    module._require_lightning_identity_for_studio_submit(args, role="exact-eval")

    args.user = None
    args.org = "comma-ai"
    module._require_lightning_identity_for_studio_submit(args, role="exact-eval")


def test_cli_submit_helper_exits_on_studio_cloud_account_mismatch(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)

    class FakeClient:
        def submit(self, spec, *, dry_run: bool = False):
            raise LightningStudioCloudAccountMismatchError(
                job_name="h100_nebius_submit",
                studio="aws-studio",
                cloud_account="nebius-h100-prod",
                machine="H100",
                original_error_type="ValueError",
                original_message=(
                    "Studio cloud account does not match provided cloud account. "
                    "Can only run jobs with Studio envs in the same cloud account."
                ),
            )

    with pytest.raises(SystemExit) as excinfo:
        module._submit_lightning_or_exit(FakeClient(), object(), dry_run=False)

    message = str(excinfo.value)
    assert "terminal_class=studio_cloud_account_namespace_mismatch" in message
    assert "use a Studio/env in that cloud account" in message
    assert "SDK said ValueError" in message


def test_non_dry_run_studio_submit_requires_active_dispatch_claim(tmp_path: Path) -> None:
    module = _load_lightning_cli_module(tmp_path)
    claims = tmp_path / "active_lane_dispatch_claims.md"
    args = argparse.Namespace(
        dry_run=False,
        studio="pact",
        queue_metadata=["lane=lane_line_search_pose_refinement"],
        dispatch_lane_id=None,
        dispatch_claims_path=str(claims),
        allow_missing_dispatch_claim_reason=None,
        job_name="exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z",
    )

    with pytest.raises(SystemExit, match="missing active dispatch claim"):
        module._require_dispatch_claim_for_submit(args, role="exact-eval")

    claims.write_text(
        "# Active lane dispatch claims -- test\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-02T00:00:00Z | codex:test | lane_line_search_pose_refinement | lightning | exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z | 2026-05-02T01:00Z | eval | active |\n"
    )
    module._require_dispatch_claim_for_submit(args, role="exact-eval")

    claims.write_text(
        "# Active lane dispatch claims -- test\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-02T00:00:00Z | codex:test | lane_line_search_pose_refinement | lightning | exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z | 2026-05-02T01:00Z | completed_score=0.32 | terminal |\n"
    )
    with pytest.raises(SystemExit, match="missing active dispatch claim"):
        module._require_dispatch_claim_for_submit(args, role="exact-eval")

    args.allow_missing_dispatch_claim_reason = "manual operator recovery during provider outage"
    module._require_dispatch_claim_for_submit(args, role="exact-eval")
    metadata = module._queue_metadata_from_args(args)
    assert metadata["dispatch_claim_skip_reason"] == "manual operator recovery during provider outage"

    from tools.claim_lane_dispatch import TERMINAL_PREFIXES

    for prefix in TERMINAL_PREFIXES:
        assert module._dispatch_claim_status_is_terminal(f"{prefix}test") is True
    assert module._dispatch_claim_status_is_terminal("active_exact_eval") is False

    claims.write_text(
        "# Active lane dispatch claims -- test\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-02T00:30:00Z | codex:test | lane_line_search_pose_refinement | lightning | exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z | 2026-05-02T01:00Z | stopped_duplicate_same_archive | terminal |\n"
        "| 2026-05-02T00:00:00Z | codex:test | lane_line_search_pose_refinement | lightning | exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z | 2026-05-02T01:00Z | eval | stale active row |\n"
    )
    args.allow_missing_dispatch_claim_reason = None
    with pytest.raises(SystemExit, match="missing active dispatch claim"):
        module._require_dispatch_claim_for_submit(args, role="exact-eval")


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
    assert "--allow-component-gate-forensic-success" not in command
    assert "--allow-sane-score-forensic-success" not in command
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
    assert "--allow-sane-score-forensic-success" not in command


def test_exact_cuda_eval_command_can_fail_job_on_sane_score_gate() -> None:
    adjudication = LightningAdjudicationSpec(
        baseline_score=1.2,
        predicted_band_low=1.0,
        predicted_band_high=1.4,
        regression_threshold=1.6,
        allow_sane_score_forensic_success=False,
    )
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/repo/archive.zip",
        upstream_dir="/upstream",
        output_dir="/out",
        adjudication=adjudication,
        **_expected_archive_kwargs(),
    )

    assert "scripts/adjudicate_contest_auth_eval.py" in command
    assert "--allow-component-gate-forensic-success" not in command
    assert "--allow-sane-score-forensic-success" not in command


def test_exact_cuda_eval_command_requires_explicit_forensic_success() -> None:
    adjudication = LightningAdjudicationSpec(
        baseline_score=1.2,
        predicted_band_low=1.0,
        predicted_band_high=1.4,
        regression_threshold=1.6,
        allow_component_gate_forensic_success=True,
        allow_sane_score_forensic_success=True,
    )
    command = exact_cuda_eval_command(
        repo_dir="/repo",
        archive_path="/repo/archive.zip",
        upstream_dir="/upstream",
        output_dir="/out",
        adjudication=adjudication,
        **_expected_archive_kwargs(),
    )

    assert "--allow-component-gate-forensic-success" in command
    assert "--allow-sane-score-forensic-success" in command


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


def test_exact_cpu_eval_spec_records_separate_role_and_axis() -> None:
    spec = make_exact_eval_spec(
        name="pfp16-cpu-eval",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(required_device="cpu"),
        eval_device="cpu",
        **_expected_archive_kwargs(),
    )

    assert spec.role == "exact_cpu_eval"
    assert spec.adjudication is not None
    assert spec.adjudication.required_device == "cpu"
    assert "--device cpu" in spec.command
    assert "--device cuda" not in spec.command
    assert "LIGHTNING_RUNNER_CPU_PREFLIGHT_OK" in spec.command
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in spec.command
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


def test_exact_eval_spec_defaults_local_artifact_dir_for_harvest() -> None:
    spec = make_exact_eval_spec(
        name="exact_eval_missing_local_dir_regression",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        expected_archive_sha256="b" * 64,
        expected_archive_size_bytes=456,
        adjudication=_adjudication(),
    )
    assert spec.local_artifact_dir == default_exact_eval_local_artifact_dir(
        job_name="exact_eval_missing_local_dir_regression"
    )
    assert spec.asdict()["local_artifact_dir"] == (
        "experiments/results/lightning_batch/exact_eval_missing_local_dir_regression"
    )


def test_exact_eval_command_can_gate_runtime_tree_hash() -> None:
    spec = make_exact_eval_spec(
        name="runtime-hash-eval",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        expected_archive_sha256="b" * 64,
        expected_archive_size_bytes=456,
        queue_metadata={"expected_runtime_tree_sha256": "c" * 64},
        adjudication=_adjudication(),
    )
    assert "--expected-runtime-tree-sha256 " + "c" * 64 in spec.command


def test_exact_eval_default_output_dir_is_writable_studio_workspace_path() -> None:
    spec = make_exact_eval_spec(
        name="job_with_underscores",
        archive_path="/repo/archive.zip",
        repo_dir="/teamspace/studios/this_studio/pact",
        upstream_dir="/upstream",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    sdk_name = lightning_sdk_job_name("job_with_underscores_T130313Z")
    assert sdk_name == "job-with-underscores-t130313z"
    assert spec.remote_output_dir == default_exact_eval_output_dir(
        repo_dir="/teamspace/studios/this_studio/pact",
        job_name="job_with_underscores",
    )
    assert spec.remote_output_dir in spec.command
    assert "/teamspace/jobs/" not in spec.command
    assert lightning_sdk_artifact_path("job_with_underscores_T130313Z") == (
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


def test_generic_lightning_job_rejects_invalid_python_stdin_heredoc() -> None:
    spec = LightningBatchJobSpec(
        name="bad_heredoc",
        machine="g7e.4xlarge",
        command=(
            "set -euo pipefail\n"
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "Path('x.json').write_text('unterminated\n"
            "')\n"
            "PY\n"
        ),
        role="training",
    )
    with pytest.raises(ValueError, match="embedded Python heredoc"):
        spec.validate()


def test_generic_lightning_job_accepts_valid_python_stdin_heredoc() -> None:
    spec = LightningBatchJobSpec(
        name="good_heredoc",
        machine="g7e.4xlarge",
        command=(
            "set -euo pipefail\n"
            ".venv/bin/python - \"$RUN/out.json\" <<'PY2'\n"
            "import json, pathlib, sys\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'ok': True}) + '\\n')\n"
            "PY2\n"
        ),
        role="training",
    )
    spec.validate()


def test_alpha_geo0_exact_eval_role_allows_generated_archive_identity() -> None:
    spec = LightningBatchJobSpec(
        name="alpha",
        machine="g4dn.xlarge",
        command=(
            "scripts/scan_lightning_supply_chain.py "
            "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK "
            "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK "
            "--require-hashes --no-deps "
            "experiments/alpha_geo0_pose_regen.py --device cuda "
            "contest_auth_eval.json"
        ),
        role="alpha_geo0_exact_eval",
        adjudication=_adjudication(),
    )
    spec.validate()


def test_alpha_geo0_exact_eval_role_rejects_generic_command() -> None:
    spec = LightningBatchJobSpec(
        name="alpha",
        machine="g4dn.xlarge",
        command=(
            "scripts/scan_lightning_supply_chain.py "
            "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK "
            "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK "
            "--require-hashes --no-deps "
            "experiments/contest_auth_eval.py --device cuda "
            "contest_auth_eval.json"
        ),
        role="alpha_geo0_exact_eval",
        adjudication=_adjudication(),
    )
    with pytest.raises(ValueError, match="alpha_geo0_pose_regen"):
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


def test_state_record_waits_on_cross_process_file_lock(tmp_path: Path) -> None:
    fcntl = pytest.importorskip("fcntl")
    state_path = tmp_path / "jobs.json"
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tac.deploy.lightning.batch_jobs import LightningBatchJobsClient\n"
        "client = LightningBatchJobsClient(state_path=Path(sys.argv[1]))\n"
        "client.record({'schema_version': 2, 'status': 'CHILD', 'spec': {'name': 'child'}})\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(state_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        time.sleep(0.35)
        assert proc.poll() is None
        assert not state_path.exists()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)
        lock_file.close()

    assert stdout == ""
    assert stderr == ""
    records = json.loads(state_path.read_text(encoding="utf-8"))
    assert records == [{"schema_version": 2, "spec": {"name": "child"}, "status": "CHILD"}]


def test_state_record_concurrent_subprocess_appends_preserve_json(tmp_path: Path) -> None:
    pytest.importorskip("fcntl")
    state_path = tmp_path / "jobs.json"
    code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from tac.deploy.lightning.batch_jobs import LightningBatchJobsClient\n"
        "state = Path(sys.argv[1])\n"
        "prefix = sys.argv[2]\n"
        "count = int(sys.argv[3])\n"
        "client = LightningBatchJobsClient(state_path=state)\n"
        "for idx in range(count):\n"
        "    client.record({'schema_version': 2, 'status': 'APPENDED', 'spec': {'name': f'{prefix}-{idx}'}})\n"
        "    time.sleep(0.005)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(state_path), f"writer{writer}", "8"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for writer in range(5)
    ]

    for proc in processes:
        stdout, stderr = proc.communicate(timeout=20)
        assert proc.returncode == 0, stderr
        assert stdout == ""

    records = json.loads(state_path.read_text(encoding="utf-8"))
    names = [record["spec"]["name"] for record in records]
    assert len(records) == 40
    assert len(set(names)) == 40
    assert all(record["status"] == "APPENDED" for record in records)


def test_dry_run_rejects_unsupported_studio_machine_before_sdk(tmp_path: Path) -> None:
    class ExplodingJob:
        @classmethod
        def run(cls, **kwargs):
            raise AssertionError("SDK should not be called")

    spec = LightningBatchJobSpec(
        name="bad-machine-dry-run",
        machine="g4dn.4xlarge",
        command="echo ok",
        studio="pact",
        role="generic",
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json", job_cls=ExplodingJob)

    with pytest.raises(ValueError, match="unsupported Lightning Studio machine/class pair"):
        client.submit(spec, dry_run=True)


def test_submit_records_official_job_fields(tmp_path: Path) -> None:
    class FakeJob:
        name = "submitted"
        status = "running"
        link = "https://lightning.ai/jobs/submitted"
        snapshot_path = "/teamspace/jobs/submitted/snapshot"
        artifact_path = "/teamspace/jobs/submitted/artifacts"
        machine = "T4"
        total_cost = 0.0
        calls: ClassVar[list[dict]] = []

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
        cloud_account="gcp-fast",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json", job_cls=FakeJob)
    record = client.submit(spec)

    assert FakeJob.calls[0]["name"] == "submitted"
    assert FakeJob.calls[0]["machine"] == "T4"
    assert FakeJob.calls[0]["cloud_account"] == "gcp-fast"
    assert FakeJob.calls[0]["interruptible"] is False
    assert record["status"] == "SUBMITTED"
    assert record["job"]["snapshot_path"].endswith("/snapshot")
    assert record["job"]["artifact_path"].endswith("/artifacts")
    assert record["job"]["source"] == "lightning_sdk_job_attributes"
    assert record["queue"]["cloud_account"] == "gcp-fast"
    assert record["queue"]["command_sha256"]


def test_submit_failure_records_queue_record_before_reraising(tmp_path: Path) -> None:
    class FailingJob:
        @classmethod
        def run(cls, **kwargs):
            raise ValueError("teamspace unavailable")

    spec = make_exact_eval_spec(
        name="submit_fail",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        teamspace="missing",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    state_path = tmp_path / "jobs.json"
    client = LightningBatchJobsClient(state_path=state_path, job_cls=FailingJob)

    with pytest.raises(ValueError, match="teamspace unavailable"):
        client.submit(spec)

    records = json.loads(state_path.read_text())
    assert len(records) == 1
    assert records[0]["status"] == "SUBMIT_FAILED"
    assert records[0]["queue"]["job_name"] == "submit_fail"
    assert records[0]["submit_error"] == {
        "type": "ValueError",
        "message": "teamspace unavailable",
    }


def test_submit_wraps_studio_cloud_account_mismatch_with_terminal_class(tmp_path: Path) -> None:
    class FailingJob:
        @classmethod
        def run(cls, **kwargs):
            raise ValueError(
                "Studio cloud account does not match provided cloud account. "
                "Can only run jobs with Studio envs in the same cloud account."
            )

    spec = make_exact_eval_spec(
        name="submit_cloud_namespace_mismatch",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact-studio",
        cloud_account="nebius-h100-prod",
        machine="g6e.4xlarge",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    state_path = tmp_path / "jobs.json"
    client = LightningBatchJobsClient(state_path=state_path, job_cls=FailingJob)

    with pytest.raises(LightningStudioCloudAccountMismatchError) as excinfo:
        client.submit(spec)

    message = str(excinfo.value)
    assert "terminal_class=studio_cloud_account_namespace_mismatch" in message
    assert "studio='pact-studio'" in message
    assert "cloud_account='nebius-h100-prod'" in message
    assert "omit --cloud-account" in message

    records = json.loads(state_path.read_text())
    assert records[0]["status"] == "SUBMIT_FAILED"
    assert records[0]["terminal_class"] == "studio_cloud_account_namespace_mismatch"
    assert records[0]["submit_error"]["terminal_class"] == "studio_cloud_account_namespace_mismatch"
    assert records[0]["submit_error"]["type"] == "ValueError"


def test_submit_default_cloud_account_keeps_sdk_failure_behavior(tmp_path: Path) -> None:
    class FailingJob:
        @classmethod
        def run(cls, **kwargs):
            raise ValueError(
                "Studio cloud account does not match provided cloud account. "
                "Can only run jobs with Studio envs in the same cloud account."
            )

    spec = make_exact_eval_spec(
        name="submit_default_cloud_account",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact-studio",
        cloud_account=None,
        machine="g6e.4xlarge",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    state_path = tmp_path / "jobs.json"
    client = LightningBatchJobsClient(state_path=state_path, job_cls=FailingJob)

    with pytest.raises(ValueError, match="Studio cloud account does not match"):
        client.submit(spec)

    records = json.loads(state_path.read_text())
    assert "terminal_class" not in records[0]
    assert records[0]["submit_error"] == {
        "type": "ValueError",
        "message": (
            "Studio cloud account does not match provided cloud account. "
            "Can only run jobs with Studio envs in the same cloud account."
        ),
    }


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


def test_refresh_status_records_status_regression_anomaly(tmp_path: Path) -> None:
    class FakeJob:
        name = "volatile"
        link = "https://lightning.ai/jobs/volatile"
        snapshot_path = "/snapshot"
        artifact_path = "/artifacts"
        machine = "T4"
        total_cost = 0.25

        def __init__(self, status: str) -> None:
            self.status = status

    spec = make_exact_eval_spec(
        name="volatile",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)
    records = json.loads((tmp_path / "jobs.json").read_text())
    records[0]["dry_run"] = False
    (tmp_path / "jobs.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    client.refresh_status_from_job(job_name="volatile", job=FakeJob("Running"))
    refreshed = client.refresh_status_from_job(job_name="volatile", job=FakeJob("Pending"))

    assert refreshed["status"] == "REMOTE_STATUS_RECONCILIATION_REQUIRED"
    assert refreshed["remote_observed_status"] == "Pending"
    assert refreshed["remote_status_accepted"] is False
    assert refreshed["status_reconciliation_required"] is True
    assert refreshed["identity_confidence"] == "name_only"
    assert refreshed["identity_reconciliation_required"] is True
    anomaly = refreshed["status_anomalies"][-1]
    assert anomaly["type"] == "nonterminal_status_regression"
    assert anomaly["previous_status"] == "Running"
    assert anomaly["current_status"] == "Pending"
    assert anomaly["accepted_status"] == "REMOTE_STATUS_RECONCILIATION_REQUIRED"
    assert anomaly["status_history_index"] == len(refreshed["status_history"]) - 1
    assert refreshed["status_history"][-1]["anomaly"] == anomaly
    assert refreshed["status_history"][-1]["observed_status"] == "Pending"
    assert refreshed["status_history"][-1]["accepted_status"] == "REMOTE_STATUS_RECONCILIATION_REQUIRED"
    assert refreshed["status_history"][-1]["job_snapshot"]["status"] == "Pending"


def test_refresh_status_backfills_existing_status_history_anomalies(tmp_path: Path) -> None:
    class FakeJob:
        name = "volatile"
        status = "Failed"
        link = "https://lightning.ai/jobs/volatile"
        snapshot_path = "/snapshot"
        artifact_path = "/artifacts"
        machine = "T4"
        total_cost = 0.25

    spec = make_exact_eval_spec(
        name="volatile",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        studio="pact",
        adjudication=_adjudication(),
        **_expected_archive_kwargs(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)
    records = json.loads((tmp_path / "jobs.json").read_text())
    records[0]["dry_run"] = False
    records[0]["status"] = "Pending"
    records[0]["status_history"] = [
        {"recorded_at_utc": "2026-05-01T00:00:00Z", "status": "SUBMITTED"},
        {
            "recorded_at_utc": "2026-05-01T00:01:00Z",
            "status": "Pending",
            "source": "lightning_sdk_job_attributes",
        },
        {
            "recorded_at_utc": "2026-05-01T00:02:00Z",
            "status": "Running",
            "source": "lightning_sdk_job_attributes",
        },
        {
            "recorded_at_utc": "2026-05-01T00:03:00Z",
            "status": "Pending",
            "source": "lightning_sdk_job_attributes",
        },
    ]
    (tmp_path / "jobs.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    refreshed = client.refresh_status_from_job(job_name="volatile", job=FakeJob())

    anomaly = refreshed["status_anomalies"][0]
    assert anomaly["type"] == "nonterminal_status_regression"
    assert anomaly["previous_status"] == "Running"
    assert anomaly["current_status"] == "Pending"
    assert anomaly["status_history_index"] == 3
    assert refreshed["status_history"][3]["anomaly"] == anomaly
    assert refreshed["status_reconciliation_required"] is True
    assert refreshed["status"] == "Failed"
    assert refreshed["remote_status_accepted"] is True


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


def test_batch_job_cli_refresh_status_all_fails_on_status_anomaly(tmp_path: Path) -> None:
    state_path = tmp_path / "jobs.json"
    spec = make_exact_eval_spec(
        name="volatile",
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
    records[0]["dry_run"] = False
    records[0]["status"] = "Running"
    records[0]["status_history"] = [
        {
            "recorded_at_utc": "2026-05-01T00:00:00Z",
            "status": "Running",
            "source": "lightning_sdk_job_attributes",
        }
    ]
    state_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    (tmp_path / "lightning_sdk.py").write_text(
        """
class Job:
    def __init__(self, *, name, teamspace=None, org=None, user=None):
        self.name = name
        self.status = "Pending"
        self.link = f"https://lightning.ai/jobs/{name}"
        self.snapshot_path = f"/teamspace/jobs/{name}/snapshot"
        self.artifact_path = f"/teamspace/jobs/{name}/artifacts"
        self.machine = "g4dn.2xlarge"
        self.total_cost = 0.25
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
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["failure_count"] == 1
    assert payload["failures"][0]["error_type"] == "StatusReconciliationRequired"
    assert payload["results"][0]["status"] == "REMOTE_STATUS_RECONCILIATION_REQUIRED"


def test_batch_job_cli_stop_infers_sdk_context_and_records_request(tmp_path: Path) -> None:
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
    records[0]["dry_run"] = False
    records[0]["status"] = "Running"
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
        self.status = "Running"
        self.link = "https://lightning.ai/jobs/custom-sdk-job"
        self.snapshot_path = "/teamspace/jobs/custom-sdk-job/snapshot"
        self.artifact_path = "/teamspace/jobs/custom-sdk-job/artifacts"
        self.machine = "g4dn.2xlarge"
        self.total_cost = 0.25

    def stop(self):
        self.status = "stopped"
""".lstrip()
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT / "src")])
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "stop",
            "--state-path",
            str(state_path),
            "--job-name",
            "job_with_underscores",
            "--reason",
            "unit-test cleanup",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["stop_request"]["stop_returned"] is True
    assert payload["stop_request"]["sdk_job_name"] == "custom-sdk-job"
    assert payload["record"]["status"] == "stopped"

    records = json.loads(state_path.read_text())
    assert records[0]["stop_requests"][0]["reason"] == "unit-test cleanup"
    assert records[0]["status_history"][-1]["observed_status"] == "stopped"


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


def test_validate_local_artifact_dir_accepts_exact_cpu_axis_as_non_promotional(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts-cpu"
    expected_sha, expected_bytes = _write_cpu_artifact_dir(artifact_dir)

    result = validate_local_artifact_dir(
        artifact_dir,
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
        require_adjudication=True,
    )

    assert result["score_axis"] == "contest_cpu"
    assert result["device"] == "cpu"
    assert result["promotion_eligible"] is False
    assert result["dali_bootstrap"] is None
    assert result["adjudication_provenance"]["contest_cpu_archive_sha256"] == expected_sha


def test_validate_local_artifact_dir_rejects_promotional_exact_cpu_adjudication(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts-cpu-promotional"
    expected_sha, expected_bytes = _write_cpu_artifact_dir(
        artifact_dir,
        promotion_eligible=True,
    )

    with pytest.raises(ValueError, match="CPU adjudication must not be promotion_eligible"):
        validate_local_artifact_dir(
            artifact_dir,
            expected_archive_sha256=expected_sha,
            expected_archive_size_bytes=expected_bytes,
            require_adjudication=True,
        )


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


def test_validate_local_artifact_dir_accepts_component_trace_status_without_trace(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir, adjudication=True)
    (artifact_dir / "component_trace_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "exact_cuda_eval_command.component_trace_status",
                "score_claim": False,
                "evidence_grade": "diagnostic_component_trace_status",
                "component_trace_exit_code": 1,
                "status": "diagnostic_unavailable_or_invalid",
                "errors": ["component trace command exited 1"],
                "summary": {},
            },
            indent=2,
        )
        + "\n"
    )

    result = validate_local_artifact_dir(artifact_dir, require_adjudication=True)

    assert result["component_trace"] is None
    assert result["component_trace_status"]["status"] == "diagnostic_unavailable_or_invalid"
    assert result["promotion_eligible"] is True


def test_validate_local_artifact_dir_uses_adjudicated_hardware_promotion_gate(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_dir(artifact_dir, adjudication=True)
    contest_path = artifact_dir / "contest_auth_eval.json"
    contest = json.loads(contest_path.read_text())
    contest["provenance"]["gpu_model"] = "NVIDIA RTX PRO 6000"
    contest["provenance"]["gpu_t4_match"] = False
    contest_path.write_text(json.dumps(contest, indent=2, sort_keys=True) + "\n")
    (artifact_dir / "contest_auth_eval.adjudicated.json").write_text(
        json.dumps(contest, indent=2, sort_keys=True) + "\n"
    )
    adjudication_path = artifact_dir / "adjudication_provenance.json"
    adjudication = json.loads(adjudication_path.read_text())
    adjudication.update(
        {
            "contest_cuda_gpu_model": "NVIDIA RTX PRO 6000",
            "contest_cuda_gpu_t4_match": False,
            "contest_equivalent_hardware": False,
            "promotion_eligible": False,
        }
    )
    adjudication_path.write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n"
    )

    result = validate_local_artifact_dir(artifact_dir, require_adjudication=True)

    assert result["gpu_t4_match"] is False
    assert result["adjudication_provenance"]["promotion_eligible"] is False
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


def test_validate_local_component_response_requires_external_baseline_repro_gate(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "component_response"
    _write_component_response_artifact_dir(artifact_dir, passed=True)
    summary_path = artifact_dir / ARTIFACT_COMPONENT_RESPONSE_SUMMARY
    summary = json.loads(summary_path.read_text())
    summary["external_baseline_contest_auth_eval_json"] = {
        "path": "/repo/baseline_eval.json",
        "bytes": 123,
        "sha256": "f" * 64,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    result = validate_local_component_response_artifact_dir(artifact_dir)

    assert result["external_baseline_repro_required"] is True
    assert result["promotion_eligible"] is False
    assert result["failed_components"] == ["combined", "posenet", "segnet"]
    with pytest.raises(ValueError, match="official component-response gates did not pass"):
        validate_local_component_response_artifact_dir(artifact_dir, require_passed=True)


def test_validate_local_component_sensitivity_artifact_dir_is_non_promotable(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "component_sensitivity"
    expected_sha, expected_bytes = _write_component_sensitivity_artifact_dir(artifact_dir)

    result = validate_local_component_sensitivity_artifact_dir(
        artifact_dir,
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
    )

    assert result["role"] == "diagnostic_component_sensitivity"
    assert result["baseline_archive_sha256"] == expected_sha
    assert result["baseline_archive_size_bytes"] == expected_bytes
    assert result["device"] == "cuda"
    assert result["promotion_eligible"] is False
    assert result["score_claim"] is False
    assert result["planning_eligible"] is True
    assert result["certification_handoff_eligible"] is False
    assert result["certification_candidate"] is False
    assert result["score_source"] == "none:diagnostic_component_sensitivity_non_promotable"
    assert set(result["curves"]) == {"posenet", "segnet", "combined"}


def test_validate_local_component_sensitivity_accepts_direct_fd_planning_artifact(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "component_sensitivity_fd"
    expected_sha, expected_bytes = _write_component_sensitivity_artifact_dir(
        artifact_dir,
        sensitivity_source="direct_renderer_cuda_finite_difference_component_response",
    )

    result = validate_local_component_sensitivity_artifact_dir(
        artifact_dir,
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
    )

    assert result["sensitivity_source"] == "direct_renderer_cuda_finite_difference_component_response"
    assert result["planning_eligible"] is True
    assert result["certification_handoff_eligible"] is True
    assert result["certification_candidate"] is True
    assert result["promotion_eligible"] is False
    assert result["score_claim"] is False
    assert result["summary"]["promotion_requested"] is True
    assert "--promotion-finite-difference" in result["run_metadata"]["profile_argv"]


def test_validate_local_component_sensitivity_rejects_unknown_source(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "component_sensitivity_unknown_source"
    _write_component_sensitivity_artifact_dir(artifact_dir, sensitivity_source="unknown_proxy")

    with pytest.raises(ValueError, match="sensitivity_source must be one of"):
        validate_local_component_sensitivity_artifact_dir(artifact_dir)


@pytest.mark.parametrize(
    ("curve_key", "bad_value", "message"),
    [
        ("score_claim", True, "score_claim=false"),
        ("promotion_eligible", True, "promotion_eligible=false"),
        ("official_component_response", True, "must not claim official_component_response"),
        ("canonical_scorer_path", True, "must not claim canonical_scorer_path"),
        ("sensitivity_source", "other_source", "sensitivity_source does not match summary"),
    ],
)
def test_validate_local_component_sensitivity_rejects_promotable_curve_claims(
    tmp_path: Path,
    curve_key: str,
    bad_value: object,
    message: str,
) -> None:
    artifact_dir = tmp_path / f"component_sensitivity_bad_{curve_key}"
    _write_component_sensitivity_artifact_dir(artifact_dir)
    curve_path = artifact_dir / "posenet_response_curve.json"
    curve = json.loads(curve_path.read_text())
    curve[curve_key] = bad_value
    curve_path.write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=message):
        validate_local_component_sensitivity_artifact_dir(artifact_dir)


@pytest.mark.parametrize(
    ("map_key", "bad_value", "message"),
    [
        ("score_claim", True, "score_claim=false"),
        ("promotion_eligible", True, "promotion_eligible=false"),
        ("official_component_response", True, "must not claim official_component_response"),
        ("canonical_scorer_path", True, "must not claim canonical_scorer_path"),
        ("sensitivity_source", "other_source", "sensitivity_source does not match summary"),
    ],
)
def test_validate_local_component_sensitivity_rejects_promotable_map_metadata(
    tmp_path: Path,
    map_key: str,
    bad_value: object,
    message: str,
) -> None:
    artifact_dir = tmp_path / f"component_sensitivity_bad_map_{map_key}"
    _write_component_sensitivity_artifact_dir(artifact_dir)
    map_path = artifact_dir / "posenet_sensitivity_map.pt"
    payload = torch.load(map_path, map_location="cpu", weights_only=False)
    payload["metadata"][map_key] = bad_value
    torch.save(payload, map_path)

    with pytest.raises(ValueError, match=message):
        validate_local_component_sensitivity_artifact_dir(artifact_dir)


def test_mirror_local_component_sensitivity_artifact_dir_copies_compact_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_component_sensitivity"
    mirror = tmp_path / "mirror_component_sensitivity"
    expected_sha, expected_bytes = _write_component_sensitivity_artifact_dir(source)

    result = mirror_local_component_sensitivity_artifact_dir(
        source,
        mirror,
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
    )

    assert result["baseline_archive_sha256"] == expected_sha
    assert result["copied_files"] == list(COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES)
    assert (mirror / ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION).is_file()


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
        if args[0] == "ssh" and args[-1].startswith("find "):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="\n".join(f"{persisted_remote}/{name}" for name in CANONICAL_ARTIFACT_FILES) + "\n",
                stderr="",
            )
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
            assert name in CANONICAL_ARTIFACT_FILES or name in {
                "auth_eval.log",
                "component_trace.json",
                "component_trace.log",
                "component_trace_status.json",
            }
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
    records = json.loads((tmp_path / "jobs.json").read_text())
    records[0]["status"] = "ARTIFACT_INFRA_FAILURE"
    records[0]["terminal_class"] = LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    records[0]["artifact_failures"] = [
        {"status": "ARTIFACT_INFRA_FAILURE", "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS}
    ]
    (tmp_path / "jobs.json").write_text(json.dumps(records, indent=2) + "\n")

    validation = client.harvest_ssh_artifacts(
        job_name="dry",
        ssh_target="lightning-host",
        require_adjudication=True,
    )

    assert calls[0][0] == "ssh"
    assert calls[1][0] == "ssh"
    assert calls[1][-1].startswith("find ")
    assert calls[2][0] == "ssh"
    assert calls[2][-1].startswith("test -d ")
    assert any(call[0] == "scp" for call in calls[3:])
    assert validation["archive_sha256"] == expected_sha
    assert validation["ssh_source"]["remote_dir"] == persisted_remote
    assert "eval_work" not in validation["ssh_source"]["copied_files"]
    assert (mirror / ARTIFACT_VALIDATION).is_file()
    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "HARVESTED"
    assert record["status_history"][-1]["source"] == "ssh_artifact_validation"
    assert record["harvests"][0]["ssh_source"]["ssh_target"] == "lightning-host"


def test_harvest_ssh_artifacts_records_empty_remote_dir_as_infra_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = tmp_path / "mirror"
    persisted_remote = "/teamspace/jobs/dry/artifacts/pact/remote/out"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ssh" and args[-1].startswith("test -d "):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "ssh" and args[-1].startswith("find "):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "scp":
            raise AssertionError("empty artifact dirs must be classified before scp")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(lightning_batch_jobs.subprocess, "run", fake_run)
    spec = make_exact_eval_spec(
        name="dry",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        output_dir="/teamspace/studios/this_studio/pact/remote/out",
        local_artifact_dir=str(mirror),
        expected_archive_sha256=EXPECTED_SHA,
        expected_archive_size_bytes=EXPECTED_BYTES,
        adjudication=_adjudication(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    diagnostic = client.harvest_ssh_artifacts(
        job_name="dry",
        ssh_target="lightning-host",
        require_adjudication=True,
    )

    assert [call[0] for call in calls] == ["ssh", "ssh"]
    assert diagnostic["status"] == "ARTIFACT_INFRA_FAILURE"
    assert diagnostic["terminal_class"] == LIGHTNING_EMPTY_ARTIFACT_INFRA_TERMINAL_CLASS
    assert diagnostic["score_claim"] is False
    assert diagnostic["method_evidence"] is False
    assert diagnostic["promotion_eligible"] is False
    assert diagnostic["score_source"] == "none:empty_lightning_artifact_dir"
    assert diagnostic["ssh_source"]["remote_dir"] == persisted_remote
    assert diagnostic["expected_archive_sha256"] == EXPECTED_SHA
    assert (mirror / ARTIFACT_INFRA_FAILURE).is_file()

    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "ARTIFACT_INFRA_FAILURE"
    assert record["terminal_class"] == LIGHTNING_EMPTY_ARTIFACT_INFRA_TERMINAL_CLASS
    assert "harvests" not in record
    assert record["artifact_failures"][0]["terminal_class"] == LIGHTNING_EMPTY_ARTIFACT_INFRA_TERMINAL_CLASS
    assert record["status_history"][-1]["source"] == "ssh_artifact_preharvest_classification"


def test_harvest_ssh_artifacts_reports_missing_remote_dir_as_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = tmp_path / "mirror"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ssh" and args[-1].startswith("test -d "):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="missing")
        if args[0] == "scp":
            raise AssertionError("not-ready artifact dirs must not be mirrored")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(lightning_batch_jobs.subprocess, "run", fake_run)
    spec = make_exact_eval_spec(
        name="dry",
        archive_path="/repo/archive.zip",
        repo_dir="/repo",
        upstream_dir="/upstream",
        output_dir="/teamspace/studios/this_studio/pact/remote/out",
        local_artifact_dir=str(mirror),
        expected_archive_sha256=EXPECTED_SHA,
        expected_archive_size_bytes=EXPECTED_BYTES,
        adjudication=_adjudication(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    diagnostic = client.harvest_ssh_artifacts(
        job_name="dry",
        ssh_target="lightning-host",
        require_adjudication=True,
    )

    assert [call[0] for call in calls] == ["ssh"]
    assert diagnostic["status"] == "ARTIFACT_NOT_READY"
    assert diagnostic["score_claim"] is False
    assert diagnostic["method_evidence"] is False
    assert diagnostic["recommended_action"].startswith("Refresh job status")
    assert not mirror.exists()

    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "DRY_RUN"
    assert "artifact_failures" not in record
    assert "harvests" not in record


def test_harvest_ssh_artifacts_records_partial_missing_score_json_as_infra_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = tmp_path / "mirror"
    remote_source = tmp_path / "remote"
    remote_source.mkdir()
    (remote_source / "archive.zip").write_bytes(b"partial archive bytes")
    identity = archive_identity(remote_source / "archive.zip")
    (remote_source / ARTIFACT_METADATA).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "exact_cuda_eval",
                "expected_archive_sha256": identity["archive_sha256"],
                "expected_archive_size_bytes": identity["archive_size_bytes"],
            }
        )
        + "\n"
    )
    (remote_source / "auth_eval.log").write_text("NameError before contest_auth_eval.json\n")
    persisted_remote = "/teamspace/jobs/dry/artifacts/pact/remote/out"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ssh" and args[-1].startswith("test -d "):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "ssh" and args[-1].startswith("find "):
            stdout = "\n".join(
                f"{persisted_remote}/{path.name}" for path in sorted(remote_source.iterdir())
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout + "\n", stderr="")
        if args[0] == "ssh" and args[-1].startswith("test -f "):
            name = args[-1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                args,
                0 if (remote_source / name).is_file() else 1,
                stdout="",
                stderr="",
            )
        if args[0] == "scp":
            name = args[-2].rsplit("/", 1)[-1]
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
        expected_archive_sha256=identity["archive_sha256"],
        expected_archive_size_bytes=identity["archive_size_bytes"],
        adjudication=_adjudication(),
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    diagnostic = client.harvest_ssh_artifacts(
        job_name="dry",
        ssh_target="lightning-host",
        require_adjudication=True,
    )

    assert diagnostic["status"] == "ARTIFACT_INFRA_FAILURE"
    assert diagnostic["terminal_class"] == LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    assert diagnostic["score_claim"] is False
    assert diagnostic["method_evidence"] is False
    assert diagnostic["archive_identity"]["archive_sha256"] == identity["archive_sha256"]
    assert "contest_auth_eval.json" in diagnostic["missing_required_files"]
    assert (mirror / "archive.zip").is_file()
    assert (mirror / "auth_eval.log").is_file()
    assert (mirror / ARTIFACT_INFRA_FAILURE).is_file()
    assert (mirror / ARTIFACT_VALIDATION).is_file()

    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "ARTIFACT_INFRA_FAILURE"
    assert record["terminal_class"] == LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    assert "harvests" not in record
    assert record["artifact_failures"][0]["terminal_class"] == LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    assert record["status_history"][-1]["source"] == "ssh_artifact_partial_failure_classification"


@pytest.mark.parametrize(
    ("log_text", "terminal_class"),
    [
        (
            "Archive validator rejected renderer_payload.bin.zst: extension not in whitelist\n",
            "archive_validator_whitelist_block",
        ),
        (
            "contest_auth_eval: inflate.sh returned non-zero exit status 1\ninflate returncode=1\n",
            "inflate_returncode_failure",
        ),
        (
            "PR86 constriction HPAC decode failed: invalid entropy model for stream 7\n",
            "pr86_constriction_hpac_invalid_entropy_model",
        ),
    ],
)
def test_launch_cli_refines_missing_score_json_from_auth_eval_log_signatures(
    tmp_path: Path,
    log_text: str,
    terminal_class: str,
) -> None:
    mod = _load_lightning_cli_module(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "auth_eval.log").write_text(log_text)
    diagnostic = {
        "status": "ARTIFACT_INFRA_FAILURE",
        "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
        "missing_required_files": ["contest_auth_eval.json", "report.txt"],
    }

    refined = mod.refine_exact_eval_missing_json_failure(
        diagnostic,
        artifact_dir=artifact_dir,
    )

    assert refined["terminal_class"] == terminal_class
    assert refined["refined_from_terminal_class"] == LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    assert refined["score_claim"] is False
    assert refined["method_evidence"] is False
    assert refined["promotion_eligible"] is False
    assert refined["score_source"].startswith("none:")
    assert "auth_eval_log_snippet_tail" in refined


def test_launch_cli_keeps_missing_artifacts_bucket_when_auth_eval_log_is_unmatched(
    tmp_path: Path,
) -> None:
    mod = _load_lightning_cli_module(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "auth_eval.log").write_text("NameError before contest_auth_eval.json\n")
    diagnostic = {
        "status": "ARTIFACT_INFRA_FAILURE",
        "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
        "missing_required_files": ["contest_auth_eval.json"],
    }

    refined = mod.refine_exact_eval_missing_json_failure(
        diagnostic,
        artifact_dir=artifact_dir,
    )

    assert refined["terminal_class"] == LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    assert refined["auth_eval_log_classification"] == "missing_artifacts"
    assert refined["score_source"] == "none:missing_contest_auth_eval_json"


def test_launch_cli_persists_refined_harvest_failure_to_mirror_and_state(
    tmp_path: Path,
) -> None:
    mod = _load_lightning_cli_module(tmp_path)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    state_path = tmp_path / "jobs.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "spec": {"name": "dry"},
                    "status": "ARTIFACT_INFRA_FAILURE",
                    "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
                    "artifact_failures": [
                        {
                            "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
                        }
                    ],
                    "status_history": [
                        {
                            "status": "ARTIFACT_INFRA_FAILURE",
                            "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
                            "source": "ssh_artifact_partial_failure_classification",
                        }
                    ],
                }
            ],
            indent=2,
        )
        + "\n"
    )
    refined = {
        "status": "ARTIFACT_INFRA_FAILURE",
        "terminal_class": "archive_validator_whitelist_block",
        "ssh_source": {"mirror_dir": str(mirror)},
    }
    args = argparse.Namespace(job_name="dry", state_path=str(state_path), mirror_dir=None)

    mod._persist_harvest_failure_refinement(args=args, refined=refined)

    for name in (ARTIFACT_INFRA_FAILURE, ARTIFACT_VALIDATION):
        payload = json.loads((mirror / name).read_text())
        assert payload["terminal_class"] == "archive_validator_whitelist_block"
    record = json.loads(state_path.read_text())[0]
    assert record["terminal_class"] == "archive_validator_whitelist_block"
    assert record["artifact_failures"][0]["terminal_class"] == "archive_validator_whitelist_block"
    assert record["status_history"][-1]["terminal_class"] == "archive_validator_whitelist_block"


def test_harvest_ssh_artifacts_recovers_missing_adjudication_copy_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = tmp_path / "mirror"
    remote_source = tmp_path / "remote"
    expected_sha, expected_bytes = _write_artifact_dir(remote_source, adjudication=True)
    (remote_source / "adjudication_provenance.json").unlink()
    (remote_source / "contest_auth_eval.adjudicated.json").unlink()
    (remote_source / "adjudication.log").write_text(
        "SCORE_RECOMPUTED=1.25\n"
        "EVIDENCE_GRADE=A++ contest T4\n"
        "PROMOTION_ELIGIBLE=1\n"
    )
    persisted_remote = "/teamspace/jobs/dry/artifacts/pact/remote/out"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[0] == "ssh" and args[-1].startswith("test -d "):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "ssh" and args[-1].startswith("find "):
            stdout = "\n".join(
                f"{persisted_remote}/{path.name}" for path in sorted(remote_source.iterdir())
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout + "\n", stderr="")
        if args[0] == "ssh" and args[-1].startswith("test -f "):
            name = args[-1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                args,
                0 if (remote_source / name).is_file() else 1,
                stdout="",
                stderr="",
            )
        if args[0] == "scp":
            name = args[-2].rsplit("/", 1)[-1]
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

    assert validation["archive_sha256"] == expected_sha
    assert validation["promotion_eligible"] is True
    assert validation["adjudication_provenance"]["artifact_recovery"] == (
        "reconstructed_missing_adjudication_artifacts_from_metadata"
    )
    assert (mirror / "adjudication_provenance.json").is_file()
    assert (mirror / "contest_auth_eval.adjudicated.json").is_file()
    assert not (mirror / ARTIFACT_INFRA_FAILURE).exists()
    assert "adjudication_provenance.json:reconstructed" in validation["ssh_source"]["copied_files"]
    assert "contest_auth_eval.adjudicated.json:reconstructed" in validation["ssh_source"]["copied_files"]

    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "HARVESTED"
    assert "terminal_class" not in record
    assert record["status_history"][-1]["source"] == "ssh_artifact_validation"


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


def test_harvest_ssh_component_sensitivity_artifacts_uses_state_artifact_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_source = tmp_path / "remote_component_sensitivity"
    expected_sha, expected_bytes = _write_component_sensitivity_artifact_dir(remote_source)
    mirror = tmp_path / "mirror"
    calls: list[list[str]] = []
    persisted_remote = "/teamspace/jobs/component-sensitivity/artifacts/pact/remote/sensitivity"

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
            shutil.copy2(remote_source / name, Path(args[-1]))
            if name == ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION:
                Path(args[-1]).chmod(0o444)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(lightning_batch_jobs.subprocess, "run", fake_run)
    spec = make_diagnostic_component_sensitivity_spec(
        name="component_sensitivity",
        baseline_archive_path="/repo/baseline.zip",
        repo_dir="/repo",
        upstream_dir="/repo/upstream",
        output_dir="/teamspace/studios/this_studio/pact/remote/sensitivity",
        local_artifact_dir=str(mirror),
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
    )
    client = LightningBatchJobsClient(state_path=tmp_path / "jobs.json")
    client.submit(spec, dry_run=True)

    validation = client.harvest_ssh_component_sensitivity_artifacts(
        job_name="component_sensitivity",
        ssh_target="lightning-host",
    )

    assert calls[0][0] == "ssh"
    assert calls[1][0] == "scp"
    assert validation["baseline_archive_sha256"] == expected_sha
    assert validation["promotion_eligible"] is False
    assert validation["ssh_source"]["remote_dir"] == persisted_remote
    assert (mirror / ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION).is_file()
    assert os.access(mirror / ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION, os.W_OK)
    record = json.loads((tmp_path / "jobs.json").read_text())[0]
    assert record["status"] == "HARVESTED"
    assert record["status_history"][-1]["source"] == "ssh_component_sensitivity_artifact_validation"


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
    assert "--remote-preflight-ssh-target" in payload["checks"]["argparse_surface"]["remote_related_options"]
    assert "--ssh-target" in payload["checks"]["argparse_surface"]["remote_related_options"]


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
            "--component-trace",
            "--component-trace-top-k",
            "17",
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
    assert payload["spec"]["adjudication"]["allow_component_gate_forensic_success"] is False
    assert payload["spec"]["adjudication"]["allow_sane_score_forensic_success"] is False
    assert "--allow-component-gate-forensic-success" not in payload["spec"]["command"]
    assert "--allow-sane-score-forensic-success" not in payload["spec"]["command"]
    assert "experiments/contest_component_trace.py" in payload["spec"]["command"]
    assert "--top-k 17" in payload["spec"]["command"]
    assert payload["submit_readiness"]["ok"] is False
    assert "missing --remote-preflight-ssh-target" in payload["submit_readiness"]["blockers"][0]


def test_batch_job_cli_dry_run_accepts_explicit_cpu_axis(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
            "--state-path",
            str(tmp_path / "jobs.json"),
            "--job-name",
            "cli-cpu-dry",
            "--archive",
            "/repo/archive.zip",
            "--repo-dir",
            "/repo",
            "--upstream-dir",
            "/upstream",
            "--studio",
            "pact",
            "--expected-archive-sha256",
            "e" * 64,
            "--expected-archive-size-bytes",
            "790",
            "--adjudicate",
            "--eval-device",
            "cpu",
            "--baseline-score",
            "1.2",
            "--predicted-band",
            "1.0",
            "1.4",
            "--regression-threshold",
            "1.6",
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
    assert payload["spec"]["role"] == "exact_cpu_eval"
    assert payload["spec"]["adjudication"]["required_device"] == "cpu"
    assert "--device cpu" in payload["spec"]["command"]
    assert "--device cuda" not in payload["spec"]["command"]
    assert "LIGHTNING_RUNNER_CPU_PREFLIGHT_OK" in payload["spec"]["command"]
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in payload["spec"]["command"]


def test_batch_job_cli_unknown_remote_flag_reports_real_argparse_surface(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
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
            "--remote",
            "lightning-pact",
            "--adjudicate",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--remote: --remote-preflight-ssh-target" in result.stderr
    assert "Strict argparse rejected unknown option" in result.stderr


def test_batch_job_cli_disallows_abbreviated_remote_preflight_flag(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
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
            "--remote-preflight-target",
            "lightning-pact",
            "--adjudicate",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--remote-preflight-target: --remote-preflight-ssh-target" in result.stderr


def test_batch_job_cli_rejects_adjudicator_only_flags_with_specific_hint(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
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
            "--required-device",
            "cuda",
            "--required-samples",
            "600",
            "--adjudicate",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--required-device: belongs to scripts/adjudicate_contest_auth_eval.py" in result.stderr
    assert "--required-samples: belongs to scripts/adjudicate_contest_auth_eval.py" in result.stderr


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
    assert payload["spec"]["adjudication"]["allow_sane_score_forensic_success"] is False
    assert "--allow-component-gate-forensic-success" not in payload["spec"]["command"]
    assert "--allow-sane-score-forensic-success" not in payload["spec"]["command"]


def test_batch_job_cli_requires_explicit_forensic_success_opt_in(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "exact-eval",
            "--state-path",
            str(tmp_path / "jobs.json"),
            "--job-name",
            "cli-dry-forensic",
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
            "--allow-component-gate-forensic-success",
            "--allow-sane-score-forensic-success",
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
    assert payload["spec"]["adjudication"]["allow_component_gate_forensic_success"] is True
    assert payload["spec"]["adjudication"]["allow_sane_score_forensic_success"] is True
    assert "--allow-component-gate-forensic-success" in payload["spec"]["command"]
    assert "--allow-sane-score-forensic-success" in payload["spec"]["command"]


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
            "--promotion-finite-difference",
            "--finite-difference-epsilon",
            "0.002",
            "--finite-difference-shard-index",
            "2",
            "--finite-difference-shard-count",
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
    assert "--promotion-finite-difference" in command
    assert "--finite-difference-epsilon 0.002" in command
    assert "--finite-difference-shard-index 2" in command
    assert "--finite-difference-shard-count 8" in command
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

    harvest_sensitivity = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "harvest-component-sensitivity-ssh",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=tmp_path,
    )
    assert harvest_sensitivity.returncode == 0, harvest_sensitivity.stderr
    assert "--job-name" in harvest_sensitivity.stdout
    assert "--state-path" in harvest_sensitivity.stdout
    assert "--remote-artifact-dir" in harvest_sensitivity.stdout


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
        assert "BatchMode=yes" in args
        assert "PasswordAuthentication=no" in args
        assert "KbdInteractiveAuthentication=no" in args
        assert "ServerAliveInterval=15" in args
        assert "ServerAliveCountMax=4" in args
        assert "TCPKeepAlive=yes" in args
        assert "ConnectionAttempts=3" in args
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


def test_remote_supply_chain_preflight_retries_transient_ssh_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_lightning_cli_module(tmp_path)
    calls: list[list[str]] = []

    def fake_auth_ready(*args: object, **kwargs: object) -> None:
        calls.append(["auth"])

    def fake_sleep(_delay: float) -> None:
        calls.append(["sleep"])

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if len(calls) == 2:
            return subprocess.CompletedProcess(
                args,
                255,
                stdout="",
                stderr="kex_exchange_identification: read: Connection reset by peer\n",
            )
        return subprocess.CompletedProcess(args, 0, stdout='{"status":"OK"}\n', stderr="")

    monkeypatch.setattr(module, "_ensure_ssh_auth_ready", fake_auth_ready)
    monkeypatch.setattr(module.time, "sleep", fake_sleep)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_remote_supply_chain_preflight(
        ssh_target="lightning-host",
        job_name="component_response",
        repo_dir="/teamspace/studios/this_studio/pact",
        python_bin=".venv/bin/python",
    )

    ssh_calls = [call for call in calls if call and call[0] == "ssh"]
    assert len(ssh_calls) == 2
    assert ["sleep"] in calls


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

    with pytest.raises(SystemExit, match=r"bare ssh\.lightning\.ai"):
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
