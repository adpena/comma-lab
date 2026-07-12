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
import torch.nn as nn

from tac.boundary_math.segnet_gradient_replacement import (
    array_content_sha256,
    capture_yopo_first_layer_bank,
    costate_injection_loss_torch,
    evaluate_teacher_step,
    load_yopo_first_layer_bank,
    measure_costate_agreement,
    relative_frame_displacement,
    write_yopo_first_layer_bank,
    yopo_first_layer_costate_torch,
    yopo_first_layer_split_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHA_A = "a" * 64
SHA_E = "e" * 64


class _YopoFeatureInfo:
    def __init__(self) -> None:
        self.info = [
            {"module": "blocks.0"},
            {"module": "blocks.1"},
            {"module": "blocks.2"},
            {"module": "blocks.4"},
            {"module": "blocks.6"},
        ]


class _YopoEncoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(1, 2, 1, bias=False)
        self.bn1 = nn.Identity()
        self.blocks = nn.ModuleList([nn.Tanh() for _ in range(7)])
        self._stage_out_idx = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}
        self.feature_info = _YopoFeatureInfo()


class _YopoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _YopoEncoderModel()


class _YopoSegNet(nn.Module):
    def __init__(self, *, bypass_cut: bool = False) -> None:
        super().__init__()
        self.encoder = _YopoEncoder()
        self.tail = nn.Conv2d(2, 1, 1, bias=False)
        self.bypass_cut = bypass_cut

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        model = self.encoder.model
        x = model.conv_stem(frame)
        x = model.bn1(x)
        cut = model.blocks[0](x)
        x = x + 0.0 * cut if self.bypass_cut else cut
        for block in model.blocks[1:]:
            x = block(x)
        return self.tail(x)


def _yopo_model(*, bypass_cut: bool = False) -> _YopoSegNet:
    torch.manual_seed(701)
    model = _YopoSegNet(bypass_cut=bypass_cut).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _yopo_loss(logits: torch.Tensor) -> torch.Tensor:
    return logits.square().mean() + 0.1 * torch.logsumexp(logits.flatten(), dim=0)


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
    injected_loss = costate_injection_loss_torch(frame, costate_transform(exact_costate))
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
    annulus = np.array([[True, False, False, True], [True, False, True, False], [False, True, False, True]])
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
def test_costate_metrics_fail_closed_on_invalid_inputs(teacher: np.ndarray, candidate: np.ndarray, reason: str) -> None:
    metrics = measure_costate_agreement(teacher, candidate)
    assert not metrics.valid
    assert any(reason in item for item in metrics.reasons)


def test_teacher_step_check_requires_descent_and_bounded_regret() -> None:
    good = _step_check(current_loss=1.0, candidate_loss=0.8, reference_loss=0.7)
    assert good.passes(max_regret=0.11)
    assert not good.passes(max_regret=0.09)
    worse = _step_check(current_loss=1.0, candidate_loss=1.01, reference_loss=0.7)
    assert not worse.passes(max_regret=1.0)
    nonfinite = _step_check(current_loss=1.0, candidate_loss=float("nan"), reference_loss=0.7)
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
    assert proof["negative_control"]["direct_vs_wrong_costate_injected"]["cosine_similarity"] <= -0.99
    assert receipt["score_claim"] is False
    assert len(receipt["git_head"]) == 40
    assert len(receipt["source_sha256s"]["tools/probe_segnet_costate_injection.py"]) == 64
    assert receipt["command_argv"]
    assert receipt["config"]["seed"] == 20260712


def test_yopo_first_layer_bank_exact_current_frame_chain_rule(tmp_path: Path) -> None:
    model = _yopo_model()
    anchor = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    model = model.to(dtype=torch.float64)
    bank, exact = capture_yopo_first_layer_bank(
        segnet=model,
        anchor_frame=anchor,
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=3,
    )
    path = tmp_path / "p1.npz"
    bank_sha = write_yopo_first_layer_bank(path, bank)
    candidate, metadata = yopo_first_layer_costate_torch(
        segnet=model,
        current_frame=anchor,
        bank_path=path,
        expected_bank_sha256=bank_sha,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        current_step=3,
        expected_split_identity_sha256=yopo_first_layer_split_identity(model),
        expected_anchor_frame_sha256=bank.anchor_frame_sha256,
        expected_source_step=3,
        max_staleness_steps=0,
    )
    assert torch.allclose(candidate, exact, atol=1e-12, rtol=1e-12)
    assert metadata["source_step"] == 3
    assert metadata["current_step"] == 3
    assert metadata["split_module_path"] == "encoder.model.blocks[0]"


def test_yopo_provider_rejects_same_step_different_frame_and_unfrozen_scorer(tmp_path: Path) -> None:
    model = _yopo_model().to(dtype=torch.float64)
    anchor = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="eval mode"):
        capture_yopo_first_layer_bank(
            segnet=model.train(),
            anchor_frame=anchor,
            teacher_loss_fn=_yopo_loss,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            evaluated_at_step=3,
        )
    model.eval()
    model.tail.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        capture_yopo_first_layer_bank(
            segnet=model,
            anchor_frame=anchor,
            teacher_loss_fn=_yopo_loss,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            evaluated_at_step=3,
        )
    model.tail.weight.requires_grad_(False)
    bank, _ = capture_yopo_first_layer_bank(
        segnet=model,
        anchor_frame=anchor,
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=3,
    )
    path = tmp_path / "bank.npz"
    bank_sha = write_yopo_first_layer_bank(path, bank)
    with pytest.raises(ValueError, match="same-step current frame"):
        yopo_first_layer_costate_torch(
            segnet=model,
            current_frame=anchor + 0.25,
            bank_path=path,
            expected_bank_sha256=bank_sha,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            current_step=3,
            expected_split_identity_sha256=bank.split_identity_sha256,
            expected_anchor_frame_sha256=bank.anchor_frame_sha256,
            expected_source_step=3,
            max_staleness_steps=0,
        )


def test_yopo_provider_rejects_batchnorm_train_mode_and_state_mutation() -> None:
    model = _yopo_model().to(dtype=torch.float64)
    batch_norm = nn.BatchNorm2d(2, dtype=torch.float64)
    model.encoder.model.bn1 = batch_norm
    for parameter in batch_norm.parameters():
        parameter.requires_grad_(False)
    batch_norm.train()
    with pytest.raises(ValueError, match="eval mode"):
        capture_yopo_first_layer_bank(
            segnet=model,
            anchor_frame=torch.ones((1, 1, 3, 3), dtype=torch.float64),
            teacher_loss_fn=_yopo_loss,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            evaluated_at_step=0,
        )
    model.eval()

    def mutate_running_mean(_module, _inputs, _output) -> None:
        with torch.no_grad():
            batch_norm.running_mean.add_(1.0)

    handle = batch_norm.register_forward_hook(mutate_running_mean)
    try:
        with pytest.raises(ValueError, match="teacher forward/backward"):
            capture_yopo_first_layer_bank(
                segnet=model,
                anchor_frame=torch.ones((1, 1, 3, 3), dtype=torch.float64),
                teacher_loss_fn=_yopo_loss,
                objective_context_fingerprint=SHA_A,
                scorer_fingerprint=SHA_E,
                evaluated_at_step=0,
            )
    finally:
        handle.remove()


def test_yopo_provider_rejects_refresh_canary_and_current_prefix_state_mutation(tmp_path: Path) -> None:
    model = _yopo_model().to(dtype=torch.float64)
    block0 = model.encoder.model.blocks[0]
    block0.register_buffer("yopo_mutation_marker", torch.zeros((), dtype=torch.float64))
    calls = 0

    def mutate_on_canary(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            with torch.no_grad():
                block0.yopo_mutation_marker.add_(1.0)

    handle = block0.register_forward_hook(mutate_on_canary)
    try:
        with pytest.raises(ValueError, match="refresh prefix canary"):
            capture_yopo_first_layer_bank(
                segnet=model,
                anchor_frame=torch.ones((1, 1, 3, 3), dtype=torch.float64),
                teacher_loss_fn=_yopo_loss,
                objective_context_fingerprint=SHA_A,
                scorer_fingerprint=SHA_E,
                evaluated_at_step=0,
            )
    finally:
        handle.remove()

    model = _yopo_model().to(dtype=torch.float64)
    block0 = model.encoder.model.blocks[0]
    block0.register_buffer("yopo_mutation_marker", torch.zeros((), dtype=torch.float64))
    anchor = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    bank, _ = capture_yopo_first_layer_bank(
        segnet=model,
        anchor_frame=anchor,
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=0,
    )
    path = tmp_path / "bank.npz"
    bank_sha = write_yopo_first_layer_bank(path, bank)

    def mutate_prefix_vjp(_module, _inputs, _output) -> None:
        with torch.no_grad():
            block0.yopo_mutation_marker.add_(1.0)

    handle = block0.register_forward_hook(mutate_prefix_vjp)
    try:
        with pytest.raises(ValueError, match="current prefix VJP"):
            yopo_first_layer_costate_torch(
                segnet=model,
                current_frame=anchor,
                bank_path=path,
                expected_bank_sha256=bank_sha,
                objective_context_fingerprint=SHA_A,
                scorer_fingerprint=SHA_E,
                current_step=0,
                expected_split_identity_sha256=bank.split_identity_sha256,
                expected_anchor_frame_sha256=bank.anchor_frame_sha256,
                expected_source_step=0,
                max_staleness_steps=0,
            )
    finally:
        handle.remove()


def test_yopo_first_layer_bank_rejects_wrong_or_mutated_bank(tmp_path: Path) -> None:
    model = _yopo_model().to(dtype=torch.float64)
    anchor = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    bank, exact = capture_yopo_first_layer_bank(
        segnet=model,
        anchor_frame=anchor,
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=1,
    )
    wrong = type(bank)(
        p1=-bank.p1,
        objective_context_fingerprint=bank.objective_context_fingerprint,
        scorer_fingerprint=bank.scorer_fingerprint,
        anchor_frame_sha256=bank.anchor_frame_sha256,
        split_identity_sha256=bank.split_identity_sha256,
        live_segnet_state_sha256=bank.live_segnet_state_sha256,
        source_step=bank.source_step,
    )
    wrong_path = tmp_path / "wrong.npz"
    wrong_sha = write_yopo_first_layer_bank(wrong_path, wrong)
    candidate, _ = yopo_first_layer_costate_torch(
        segnet=model,
        current_frame=anchor,
        bank_path=wrong_path,
        expected_bank_sha256=wrong_sha,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        current_step=1,
        expected_split_identity_sha256=yopo_first_layer_split_identity(model),
        expected_anchor_frame_sha256=bank.anchor_frame_sha256,
        expected_source_step=1,
        max_staleness_steps=0,
    )
    assert not torch.allclose(candidate, exact)
    wrong_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        load_yopo_first_layer_bank(
            wrong_path,
            expected_bank_sha256=wrong_sha,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            expected_split_identity_sha256=bank.split_identity_sha256,
        )


@pytest.mark.parametrize("field", ["objective", "scorer", "anchor", "split", "step"])
def test_yopo_bank_rejects_changed_bindings(tmp_path: Path, field: str) -> None:
    model = _yopo_model().to(dtype=torch.float64)
    anchor = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    bank, _ = capture_yopo_first_layer_bank(
        segnet=model,
        anchor_frame=anchor,
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=2,
    )
    path = tmp_path / "bank.npz"
    bank_sha = write_yopo_first_layer_bank(path, bank)
    kwargs = {
        "expected_bank_sha256": bank_sha,
        "objective_context_fingerprint": SHA_A,
        "scorer_fingerprint": SHA_E,
        "expected_split_identity_sha256": bank.split_identity_sha256,
        "expected_anchor_frame_sha256": bank.anchor_frame_sha256,
        "expected_source_step": 2,
    }
    if field == "objective":
        kwargs["objective_context_fingerprint"] = "b" * 64
    if field == "scorer":
        kwargs["scorer_fingerprint"] = "c" * 64
    if field == "anchor":
        kwargs["expected_anchor_frame_sha256"] = "d" * 64
    if field == "split":
        kwargs["expected_split_identity_sha256"] = "e" * 64
    if field == "step":
        kwargs["expected_source_step"] = 7
    with pytest.raises(ValueError, match="mismatch"):
        load_yopo_first_layer_bank(path, **kwargs)


def test_yopo_topology_rejects_bypassed_or_nonfinite_cut(tmp_path: Path) -> None:
    model = _yopo_model().to(dtype=torch.float64)
    model.encoder.model._stage_out_idx = {1: 1}
    with pytest.raises(ValueError, match="topology mismatch"):
        yopo_first_layer_split_identity(model)
    model = _yopo_model().to(dtype=torch.float64)
    with pytest.raises(ValueError, match="must be finite"):
        capture_yopo_first_layer_bank(
            segnet=model,
            anchor_frame=torch.full((1, 1, 3, 3), float("nan"), dtype=torch.float64),
            teacher_loss_fn=_yopo_loss,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            evaluated_at_step=0,
        )
    valid = _yopo_model().to(dtype=torch.float64)
    finite_bank, _ = capture_yopo_first_layer_bank(
        segnet=valid,
        anchor_frame=torch.ones((1, 1, 3, 3), dtype=torch.float64),
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=0,
    )
    nonfinite_bank = type(finite_bank)(
        p1=np.full_like(finite_bank.p1, np.nan),
        objective_context_fingerprint=finite_bank.objective_context_fingerprint,
        scorer_fingerprint=finite_bank.scorer_fingerprint,
        anchor_frame_sha256=finite_bank.anchor_frame_sha256,
        split_identity_sha256=finite_bank.split_identity_sha256,
        live_segnet_state_sha256=finite_bank.live_segnet_state_sha256,
        source_step=finite_bank.source_step,
    )
    with pytest.raises(ValueError, match="p1 must be a finite"):
        write_yopo_first_layer_bank(tmp_path / "nonfinite.npz", nonfinite_bank)


def test_yopo_provider_rejects_live_state_drift_and_stale_or_future_bank(tmp_path: Path) -> None:
    model = _yopo_model().to(dtype=torch.float64)
    anchor = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    bank, _ = capture_yopo_first_layer_bank(
        segnet=model,
        anchor_frame=anchor,
        teacher_loss_fn=_yopo_loss,
        objective_context_fingerprint=SHA_A,
        scorer_fingerprint=SHA_E,
        evaluated_at_step=4,
    )
    path = tmp_path / "bank.npz"
    bank_sha = write_yopo_first_layer_bank(path, bank)
    common = {
        "segnet": model,
        "current_frame": anchor,
        "bank_path": path,
        "expected_bank_sha256": bank_sha,
        "objective_context_fingerprint": SHA_A,
        "scorer_fingerprint": SHA_E,
        "expected_split_identity_sha256": bank.split_identity_sha256,
        "expected_anchor_frame_sha256": bank.anchor_frame_sha256,
        "expected_source_step": 4,
        "max_staleness_steps": 1,
    }
    with pytest.raises(ValueError, match="future"):
        yopo_first_layer_costate_torch(**common, current_step=3)
    with pytest.raises(ValueError, match="stale"):
        yopo_first_layer_costate_torch(**common, current_step=6)
    with torch.no_grad():
        model.encoder.model.conv_stem.weight.add_(0.01)
    with pytest.raises(ValueError, match="live SegNet state changed"):
        yopo_first_layer_costate_torch(**common, current_step=4)
    bypass = _yopo_model(bypass_cut=True).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="refresh cut canary failed"):
        capture_yopo_first_layer_bank(
            segnet=bypass,
            anchor_frame=torch.ones((1, 1, 3, 3), dtype=torch.float64),
            teacher_loss_fn=_yopo_loss,
            objective_context_fingerprint=SHA_A,
            scorer_fingerprint=SHA_E,
            evaluated_at_step=0,
        )
