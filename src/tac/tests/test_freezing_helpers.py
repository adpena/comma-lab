# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from tac.freezing import (
    LoRARendererAdapter,
    SWACheckpointAverager,
    ScorerNotFrozenError,
    apply_pose_gradient_stop_after_warmstart,
    build_frozen_teacher_from_state_dict,
    ema_freeze_at_eval_snapshot_restore,
    ensure_compress_time_scorer_freeze,
    extract_lottery_ticket,
    freeze_module_parameters,
    frozen_teacher_distillation_loss,
)


def test_freeze_module_parameters_is_idempotent_and_evals() -> None:
    module = nn.Linear(3, 2)

    first = freeze_module_parameters(module, name="scorer")
    second = freeze_module_parameters(module, name="scorer")

    assert first.trainable_before == 8
    assert first.trainable_after == 0
    assert second.trainable_before == 0
    assert not module.training
    assert all(not param.requires_grad for param in module.parameters())


def test_ensure_compress_time_scorer_freeze_raises_when_trainable() -> None:
    module = nn.Linear(3, 2)

    with pytest.raises(ScorerNotFrozenError):
        ensure_compress_time_scorer_freeze(module, names=("segnet",))

    freeze_module_parameters(module, name="segnet")
    report = ensure_compress_time_scorer_freeze(module, names=("segnet",))
    assert report[0].name == "segnet"


def test_pose_gradient_stop_after_warmstart_waits_then_freezes() -> None:
    module = nn.Linear(2, 2)

    before = apply_pose_gradient_stop_after_warmstart(
        module,
        current_epoch=1,
        warmstart_epochs=2,
    )
    after = apply_pose_gradient_stop_after_warmstart(
        module,
        current_epoch=2,
        warmstart_epochs=2,
    )

    assert before.stopped is False
    assert before.trainable_after > 0
    assert after.stopped is True
    assert after.trainable_after == 0


def test_lora_renderer_adapter_freezes_base_and_starts_identity() -> None:
    base = nn.Linear(4, 3)
    x = torch.randn(5, 4)
    expected = base(x).detach()

    adapter = LoRARendererAdapter(base, rank=2, alpha=4.0)

    assert torch.allclose(adapter(x), expected)
    assert all(not param.requires_grad for param in adapter.base.parameters())
    assert adapter.a.requires_grad
    assert adapter.b.requires_grad
    assert adapter.report().adapter_parameters == 2 * 4 + 3 * 2


def test_frozen_teacher_builder_and_distillation_loss() -> None:
    torch.manual_seed(1)
    source = nn.Linear(3, 2)
    teacher = build_frozen_teacher_from_state_dict(
        lambda: nn.Linear(3, 2),
        source.state_dict(),
    )
    student_logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    teacher_logits = teacher(torch.ones(1, 3))

    loss, report = frozen_teacher_distillation_loss(student_logits, teacher_logits)
    loss.backward()

    assert loss.ndim == 0
    assert student_logits.grad is not None
    assert report.temperature == 2.0
    assert all(not param.requires_grad for param in teacher.parameters())


def test_swa_checkpoint_averager_applies_mean_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    averager = SWACheckpointAverager()
    with torch.no_grad():
        model.weight.fill_(1.0)
    averager.update(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    report = averager.update(model)
    with torch.no_grad():
        model.weight.fill_(9.0)

    averager.apply_to(model)

    assert report.snapshot_count == 2
    assert torch.allclose(model.weight, torch.tensor([[2.0]]))


def test_lottery_ticket_extracts_boolean_masks() -> None:
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    ticket = extract_lottery_ticket(model, keep_fraction=0.5)

    mask = ticket.masks["weight"]
    assert mask.dtype == torch.bool
    assert ticket.total_parameters == 4
    assert ticket.kept_parameters == 2
    assert mask.tolist() == [[False, False, True, True]]


def test_ema_freeze_at_eval_snapshot_restore_restores_live_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.train()
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema_state = {"weight": torch.tensor([[5.0]])}

    with ema_freeze_at_eval_snapshot_restore(model, ema_state) as snapshot:
        assert snapshot.tensor_count == 1
        assert snapshot.training_before is True
        assert not model.training
        assert torch.allclose(model.weight, torch.tensor([[5.0]]))

    assert model.training
    assert torch.allclose(model.weight, torch.tensor([[1.0]]))
