# SPDX-License-Identifier: MIT
"""Behavioral tests for exact input-costate injection and its negative controls."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.boundary_math.segnet_gradient_replacement import (
    array_content_sha256,
    costate_injection_loss_torch,
    evaluate_teacher_step,
    measure_costate_agreement,
    relative_frame_displacement,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHA_A = "a" * 64
SHA_E = "e" * 64


def _step_check(**losses):
    return evaluate_teacher_step(
        objective_context_fingerprint=SHA_A,
        anchor_frame=np.zeros((2, 2)),
        candidate_frame=np.ones((2, 2)) * 0.8,
        reference_frame=np.ones((2, 2)) * 0.7,
        provider_custody_sha256=SHA_E,
        evaluated_at_step=7,
        **losses,
    )


def _direct_and_injected_gradients(costate_transform=lambda value: value):
    theta = torch.tensor([0.31, -0.47, 0.19], dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor(
        [[0.2, -0.3, 0.7], [0.5, 0.9, -0.4], [-0.8, 0.1, 0.6], [0.4, -0.2, 0.3]],
        dtype=torch.float64,
    )
    frame = (torch.sin(matrix @ theta) + 0.1 * (matrix @ theta).square()).reshape(2, 2)
    mixed = torch.stack((frame[0, 0] + frame[1, 1], frame[0, 1] * frame[1, 0]))
    teacher_loss = torch.logsumexp(mixed, dim=0) + 0.3 * frame.square().mean()
    exact_costate = torch.autograd.grad(teacher_loss, frame, retain_graph=True)[0]
    direct = torch.autograd.grad(teacher_loss, theta, retain_graph=True)[0]
    injected_loss = costate_injection_loss_torch(
        frame, costate_transform(exact_costate)
    )
    injected = torch.autograd.grad(injected_loss, theta)[0]
    return direct, injected


def test_exact_teacher_input_costate_reproduces_parameter_gradient() -> None:
    direct, injected = _direct_and_injected_gradients()
    assert torch.equal(direct, injected)
    metrics = measure_costate_agreement(direct.detach(), injected.detach())
    assert metrics.valid
    assert metrics.cosine_similarity == pytest.approx(1.0, abs=1e-15)
    assert metrics.relative_l2_error == pytest.approx(0.0, abs=1e-15)
    assert metrics.norm_ratio == pytest.approx(1.0, abs=1e-15)


def test_wrong_costate_negative_control_does_not_reproduce_gradient() -> None:
    direct, wrong = _direct_and_injected_gradients(lambda value: -value)
    assert not torch.allclose(direct, wrong)
    metrics = measure_costate_agreement(direct.detach(), wrong.detach())
    assert metrics.valid
    assert metrics.cosine_similarity == pytest.approx(-1.0, abs=1e-15)
    assert metrics.relative_l2_error == pytest.approx(2.0, abs=1e-15)


def test_costate_metrics_support_margin_annulus_mask() -> None:
    teacher = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 1.0
    candidate = teacher.copy()
    candidate[:, 1, :] *= -1.0
    annulus = np.array(
        [[True, False, False, True], [True, False, True, False], [False, True, False, True]]
    )
    # (C,H/W-like) mask expands over leading batch axis through ordinary broadcasting.
    metrics = measure_costate_agreement(teacher, candidate, mask=annulus)
    assert metrics.valid
    assert metrics.compared_elements == int(np.broadcast_to(annulus, teacher.shape).sum())
    assert metrics.relative_l2_error > 0.0


@pytest.mark.parametrize(
    ("teacher", "candidate", "reason"),
    [
        (np.ones((2, 3)), np.ones((3, 2)), "shape mismatch"),
        (np.ones((2, 3)), np.full((2, 3), np.nan), "nonfinite"),
        (np.zeros((2, 3)), np.zeros((2, 3)), "zero"),
    ],
)
def test_costate_metrics_fail_closed_on_invalid_inputs(
    teacher: np.ndarray, candidate: np.ndarray, reason: str
) -> None:
    metrics = measure_costate_agreement(teacher, candidate)
    assert not metrics.valid
    assert any(reason in item for item in metrics.reasons)


def test_teacher_step_check_requires_descent_and_bounded_regret() -> None:
    good = _step_check(current_loss=1.0, candidate_loss=0.8, reference_loss=0.7)
    assert good.passes(max_regret=0.11)
    assert not good.passes(max_regret=0.09)
    worse = _step_check(current_loss=1.0, candidate_loss=1.01, reference_loss=0.7)
    assert not worse.passes(max_regret=1.0)
    nonfinite = _step_check(
        current_loss=1.0, candidate_loss=float("nan"), reference_loss=0.7
    )
    assert not nonfinite.passes(max_regret=1.0)

    with pytest.raises(ValueError, match="evaluated_at_step"):
        evaluate_teacher_step(
            current_loss=1.0,
            candidate_loss=0.8,
            reference_loss=0.7,
            objective_context_fingerprint=SHA_A,
            anchor_frame=np.zeros((2, 2)),
            candidate_frame=np.ones((2, 2)) * 0.8,
            reference_frame=np.ones((2, 2)) * 0.7,
            provider_custody_sha256=SHA_E,
            evaluated_at_step=True,
        )


def test_array_content_hash_binds_dtype_shape_and_bytes() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert array_content_sha256(array) == array_content_sha256(array.copy())
    assert array_content_sha256(array) != array_content_sha256(array.reshape(3, 2))
    assert array_content_sha256(array) != array_content_sha256(array.astype(np.float64))
    changed = array.copy()
    changed[0, 0] = 9.0
    assert array_content_sha256(array) != array_content_sha256(changed)


def test_frame_trust_radius_is_relative_and_fail_closed() -> None:
    anchor = np.ones((2, 3), dtype=np.float64)
    current = anchor * 1.1
    assert relative_frame_displacement(anchor, current) == pytest.approx(0.1)
    assert relative_frame_displacement(anchor, np.ones((3, 2))) == float("inf")


def test_import_is_mlx_lazy_on_headless_consumers() -> None:
    code = (
        "import sys; "
        "import tac.boundary_math.segnet_gradient_replacement; "
        "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_probe_writes_behavioral_receipt(tmp_path: Path) -> None:
    output = tmp_path / "costate_receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/probe_segnet_costate_injection.py"),
            "--output",
            str(output),
            "--seed",
            "20260712",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text())
    proof = receipt["synthetic_proof"]
    assert proof["proof_pass"] is True
    assert proof["direct_vs_exact_costate_injected"]["max_absolute_error"] <= 1e-12
    assert proof["direct_vs_exact_costate_injected"]["cosine_similarity"] >= 1.0 - 1e-12
    assert proof["negative_control"]["failed_as_required"] is True
    assert proof["negative_control"]["direct_vs_wrong_costate_injected"][
        "cosine_similarity"
    ] <= -0.99
    assert receipt["score_claim"] is False
    assert len(receipt["git_head"]) == 40
    assert len(
        receipt["source_sha256s"]["tools/probe_segnet_costate_injection.py"]
    ) == 64
    assert receipt["command_argv"]
    assert receipt["config"]["seed"] == 20260712
