# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_PATH = (
    REPO_ROOT
    / "submitted_jobs"
    / "training_dinov3_cooperative_receiver_anchor_20260518T140408Z.py"
)


def _load_job_module():
    spec = importlib.util.spec_from_file_location("dinov3_anchor_job", JOB_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RegisterTinyDino:
    num_prefix_tokens = 5


def test_submitted_dinov3_job_uses_256px_contract_and_tool_axis_labels(monkeypatch) -> None:
    job = _load_job_module()
    monkeypatch.setattr(job.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(job.platform, "system", lambda: "Darwin")

    device, evidence_grade = job.resolve_anchor_device()

    assert job.DINOV3_INPUT_SIZE == 256
    assert device == "cpu"
    assert evidence_grade == "[macOS-CPU advisory frozen-anchor]"
    assert "contest-CPU" not in evidence_grade


def test_submitted_dinov3_job_strips_register_tokens_from_patch_grid() -> None:
    job = _load_job_module()
    tokens = torch.randn(2, 1 + 4 + 256, 8)

    cls_tokens, patch_tokens = job.split_dinov3_tokens(tokens, _RegisterTinyDino())

    assert cls_tokens.shape == (2, 8)
    assert patch_tokens.shape == (2, 256, 8)


def test_submitted_dinov3_job_fallback_detects_square_patch_grid_after_registers() -> None:
    job = _load_job_module()
    tokens = torch.randn(1, 1 + 4 + 256, 8)

    prefix_count = job.infer_vit_prefix_token_count(tokens, object())

    assert prefix_count == 5
