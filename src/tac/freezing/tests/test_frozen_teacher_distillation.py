# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.frozen_teacher_distillation``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.freezing.frozen_teacher_distillation import (
    FrozenTeacherDistillationConfig,
    FrozenTeacherDistillationReport,
    build_frozen_teacher_from_state_dict,
    frozen_teacher_distillation_loss,
)


def test_default_config_temperature_is_hinton_canonical():
    """Default T=2.0 per CLAUDE.md Quantizr intelligence + Hinton 2014."""
    cfg = FrozenTeacherDistillationConfig()
    assert cfg.temperature == 2.0
    assert cfg.reduction == "batchmean"


def test_temperature_zero_rejected():
    """Temperature must be positive."""
    cfg = FrozenTeacherDistillationConfig(temperature=0.0)
    s = torch.randn(2, 5)
    t = torch.randn(2, 5)
    with pytest.raises(ValueError):
        frozen_teacher_distillation_loss(s, t, config=cfg)


def test_temperature_negative_rejected():
    """Temperature must be > 0."""
    cfg = FrozenTeacherDistillationConfig(temperature=-1.0)
    s = torch.randn(2, 5)
    t = torch.randn(2, 5)
    with pytest.raises(ValueError):
        frozen_teacher_distillation_loss(s, t, config=cfg)


def test_kl_zero_when_student_equals_teacher():
    """KL(p||p) = 0 when student logits exactly match teacher logits."""
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    loss, report = frozen_teacher_distillation_loss(logits, logits)
    assert isinstance(report, FrozenTeacherDistillationReport)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
    assert report.loss_value == pytest.approx(0.0, abs=1e-6)


def test_kl_positive_when_student_differs_from_teacher():
    """KL > 0 when the distributions differ."""
    torch.manual_seed(0)
    s = torch.randn(4, 5)
    t = torch.randn(4, 5) * 5.0  # Very different distribution.
    loss, _ = frozen_teacher_distillation_loss(s, t)
    assert float(loss.detach()) > 0.0


def test_t_squared_scaling_applied():
    """Loss scales with T^2 per Hinton 2015 gradient normalization."""
    torch.manual_seed(0)
    s = torch.randn(8, 5)
    t = torch.randn(8, 5) * 3.0
    cfg1 = FrozenTeacherDistillationConfig(temperature=1.0)
    cfg2 = FrozenTeacherDistillationConfig(temperature=2.0)
    loss1, _ = frozen_teacher_distillation_loss(s, t, config=cfg1)
    loss2, _ = frozen_teacher_distillation_loss(s, t, config=cfg2)
    # Manually replicate without T^2:
    log_p2 = F.log_softmax(s / 2.0, dim=-1)
    q2 = F.softmax(t / 2.0, dim=-1)
    raw_kl_2 = F.kl_div(log_p2, q2, reduction="batchmean")
    assert float(loss2) == pytest.approx(float(raw_kl_2) * 4.0, rel=1e-4)


def test_gradient_flows_to_student_not_teacher():
    """``loss.backward()`` produces gradient on the student logits only."""
    s = torch.randn(4, 5, requires_grad=True)
    t = torch.randn(4, 5, requires_grad=True)
    loss, _ = frozen_teacher_distillation_loss(s, t)
    loss.backward()
    assert s.grad is not None
    # Teacher is .detach()ed inside the helper so teacher.grad is None.
    assert t.grad is None


def test_report_records_temperature_and_reduction():
    """Report carries the config back so provenance is machine-readable."""
    cfg = FrozenTeacherDistillationConfig(temperature=3.0, reduction="sum")
    s = torch.randn(2, 5)
    t = torch.randn(2, 5)
    _, report = frozen_teacher_distillation_loss(s, t, config=cfg)
    assert report.temperature == 3.0
    assert report.reduction == "sum"


def test_build_frozen_teacher_from_state_dict_freezes():
    """The helper instantiates, loads, freezes, and evals the teacher."""
    torch.manual_seed(0)
    src = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))
    src.train()
    sd = src.state_dict()

    def factory():
        return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3))

    teacher = build_frozen_teacher_from_state_dict(factory, sd)
    # Frozen.
    for p in teacher.parameters():
        assert p.requires_grad is False
    # Eval mode.
    assert teacher.training is False
    # State dict matches source.
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, sd[k])


def test_build_frozen_teacher_to_device_cpu():
    """``device='cpu'`` produces a CPU-resident teacher."""
    src = nn.Linear(4, 2)
    sd = src.state_dict()
    teacher = build_frozen_teacher_from_state_dict(
        lambda: nn.Linear(4, 2), sd, device="cpu"
    )
    for p in teacher.parameters():
        assert p.device.type == "cpu"


def test_end_to_end_a1_teacher_distillation_pattern():
    """End-to-end: A1-style frozen teacher + trainable student + distillation loss.

    This mirrors the canonical T4 Priority 1 BOLT-ON-on-A1 wiring per
    Hinton's grand-council position: bolt-on student initialized from A1
    weights + trained with KL-T=2.0 from A1 frozen teacher.
    """
    torch.manual_seed(0)
    a1_state = {
        "weight": torch.randn(5, 8),
        "bias": torch.randn(5),
    }

    def factory():
        return nn.Linear(8, 5)

    teacher = build_frozen_teacher_from_state_dict(factory, a1_state)

    # Student is initialized from the same A1 weights (per Hinton; this is
    # the knowledge-preservation trick).
    student = nn.Linear(8, 5)
    student.load_state_dict(a1_state)

    x = torch.randn(4, 8)
    s_logits = student(x)
    with torch.no_grad():
        t_logits = teacher(x)
    loss, report = frozen_teacher_distillation_loss(s_logits, t_logits)
    # At construction, student == teacher so KL == 0.
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-5)
    # Backward + step would not move student (zero gradient at the fixed point).
    loss.backward()
    # Student gradient is exactly zero (or near-zero numerically).
    for p in student.parameters():
        assert p.grad is not None
        assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-5)
