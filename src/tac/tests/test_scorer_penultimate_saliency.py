# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from tac.analysis.scorer_penultimate_saliency import (
    DEFAULT_HOOK_TARGETS,
    PenultimateHookTarget,
    build_proxy_safe_manifest,
    build_synthetic_scorer_models,
    build_synthetic_smoke_manifest,
    resolve_module,
    run_penultimate_saliency_probe,
    validate_penultimate_saliency_manifest,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate

REPO_ROOT = Path(__file__).resolve().parents[3]


def _hook_row(manifest: dict, target_id: str) -> dict:
    for row in manifest["hook_rows"]:
        if row["target_id"] == target_id:
            return row
    raise AssertionError(f"missing hook row {target_id}")


def test_synthetic_smoke_manifest_is_proxy_safe_and_analysis_only() -> None:
    manifest = build_synthetic_smoke_manifest(seed=11, repo_root=REPO_ROOT)

    assert manifest["schema"] == "tac_scorer_penultimate_saliency.v1"
    assert manifest["analysis_time_only"] is True
    assert manifest["inflate_time_use_allowed"] is False
    assert manifest["inflate_time_scorer_load_allowed"] is False
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["research_only"] is True
    assert manifest["gpu_used"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["required_hook_modules_available"] is True
    assert manifest["required_activations_seen"] is True
    assert manifest["gradient_support_available"] is True
    assert manifest["saliency_ready_for_proxy_planning"] is True
    assert validate_penultimate_saliency_manifest(manifest) == []

    for target_id in ("posenet_summarizer", "segnet_decoder"):
        row = _hook_row(manifest, target_id)
        assert row["hook_registerable"] is True
        assert row["activation_seen"] is True
        assert row["activation_l2_total"] > 0.0
        assert row["gradient_available"] is True
        assert row["gradient_l2_total"] > 0.0
        assert row["score_claim"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert validate_proxy_candidate(row) == []


def test_synthetic_smoke_is_deterministic_for_norms() -> None:
    first = build_synthetic_smoke_manifest(seed=123, repo_root=REPO_ROOT)
    second = build_synthetic_smoke_manifest(seed=123, repo_root=REPO_ROOT)

    for target in ("posenet_summarizer", "posenet_hydra_resblock", "segnet_decoder"):
        a = _hook_row(first, target)
        b = _hook_row(second, target)
        assert math.isclose(a["activation_l2_total"], b["activation_l2_total"], rel_tol=0.0, abs_tol=0.0)
        assert math.isclose(a["gradient_l2_total"], b["gradient_l2_total"], rel_tol=0.0, abs_tol=0.0)


def test_probe_records_missing_required_hook_without_score_authority() -> None:
    models = {"posenet": build_synthetic_scorer_models(seed=5)["posenet"]}
    inputs = {"posenet": torch.randn(2, 8)}
    target = PenultimateHookTarget(
        target_id="missing_pose_feature",
        scorer_id="posenet",
        module_path="not_a_real_module",
        required=True,
        saliency_role="pose_penultimate_feature_saliency",
        expected_feature="missing for test",
    )

    probe = run_penultimate_saliency_probe(models, inputs, targets=(target,))
    manifest = build_proxy_safe_manifest(probe, repo_root=REPO_ROOT, mode="unit_missing_hook", seed=5)

    row = _hook_row(manifest, "missing_pose_feature")
    assert row["status"] == "module_missing"
    assert row["module_present"] is False
    assert row["hook_registerable"] is False
    assert row["activation_seen"] is False
    assert manifest["required_hook_modules_available"] is False
    assert manifest["required_activations_seen"] is False
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert validate_penultimate_saliency_manifest(manifest) == []


def test_probe_removes_forward_hooks_after_run() -> None:
    models = build_synthetic_scorer_models(seed=7)
    inputs = {
        "posenet": torch.randn(2, 8),
        "segnet": torch.randn(2, 3, 8, 8),
    }

    run_penultimate_saliency_probe(models, inputs, targets=DEFAULT_HOOK_TARGETS)

    for target in DEFAULT_HOOK_TARGETS:
        module = resolve_module(models[target.scorer_id], target.module_path)
        assert module is not None
        assert len(module._forward_hooks) == 0


def test_cli_writes_strict_proxy_safe_manifest(tmp_path: Path) -> None:
    output = tmp_path / "penultimate_saliency.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/probe_scorer_penultimate_saliency.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(output),
            "--seed",
            "19",
            "--strict",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(output.read_text())
    stdout_payload = json.loads(proc.stdout)
    assert payload["manifest_validation"]["passed"] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert stdout_payload["seed"] == 19


def test_probe_can_capture_tuple_outputs() -> None:
    class TupleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.summarizer = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor):
            y = self.summarizer(x)
            return y, {"pose": y[:, :2]}

    target = PenultimateHookTarget(
        target_id="tuple_summarizer",
        scorer_id="posenet",
        module_path="summarizer",
        required=True,
        saliency_role="pose_penultimate_feature_saliency",
        expected_feature="tuple output support",
    )
    model = TupleModel()
    probe = run_penultimate_saliency_probe(
        {"posenet": model},
        {"posenet": torch.randn(3, 4)},
        targets=(target,),
    )

    row = probe["hook_rows"][0]
    assert row["activation_seen"] is True
    assert row["gradient_available"] is True
    assert row["gradient_l2_total"] > 0.0
