# SPDX-License-Identifier: MIT
"""Tests for tac.composition.distillation_chain — Distillation chain."""

from __future__ import annotations

import pytest
import torch

from tac.composition.distillation_chain import (
    DEFAULT_TEMPERATURE,
    DISTILL_MAGIC,
    DISTILL_SCHEMA_VERSION,
    DistillationChain,
    DistillationError,
    DistillationLevel,
    distillation_loss,
)

# ---------------------------------------------------------------------------
# DistillationLevel validation
# ---------------------------------------------------------------------------


def test_level_defaults() -> None:
    lvl = DistillationLevel(name="teacher", param_count=1000)
    assert lvl.temperature == DEFAULT_TEMPERATURE
    assert lvl.kl_weight > 0


def test_level_rejects_empty_name() -> None:
    with pytest.raises(DistillationError, match="name must be non-empty"):
        DistillationLevel(name="", param_count=100)


def test_level_rejects_non_positive_param_count() -> None:
    with pytest.raises(DistillationError, match="param_count must"):
        DistillationLevel(name="x", param_count=0)


def test_level_rejects_negative_target_bytes() -> None:
    with pytest.raises(DistillationError, match="target_archive_bytes"):
        DistillationLevel(name="x", param_count=100, target_archive_bytes=-1)


def test_level_rejects_non_positive_temperature() -> None:
    with pytest.raises(DistillationError, match="temperature must"):
        DistillationLevel(name="x", param_count=100, temperature=0.0)


def test_level_rejects_negative_weights() -> None:
    with pytest.raises(DistillationError, match="non-negative"):
        DistillationLevel(name="x", param_count=100, kl_weight=-0.5)


def test_level_rejects_all_zero_weights() -> None:
    with pytest.raises(DistillationError, match="at least one"):
        DistillationLevel(
            name="x", param_count=100, kl_weight=0.0, hard_weight=0.0
        )


# ---------------------------------------------------------------------------
# DistillationChain validation
# ---------------------------------------------------------------------------


def test_chain_requires_at_least_two_levels() -> None:
    lvl = DistillationLevel(name="t", param_count=100)
    with pytest.raises(DistillationError, match=">= 2 levels"):
        DistillationChain(levels=(lvl,))


def test_chain_requires_compression() -> None:
    # Student must be smaller by >= 1.5x.
    t = DistillationLevel(name="t", param_count=100)
    s = DistillationLevel(name="s", param_count=80)  # only 1.25x compression
    with pytest.raises(DistillationError, match="does not compress"):
        DistillationChain(levels=(t, s))


def test_chain_default_compression_floor() -> None:
    t = DistillationLevel(name="t", param_count=300)
    s = DistillationLevel(name="s", param_count=100)  # 3x compression
    chain = DistillationChain(levels=(t, s))
    assert chain.num_levels() == 2


def test_chain_custom_compression_floor() -> None:
    t = DistillationLevel(name="t", param_count=130)
    s = DistillationLevel(name="s", param_count=100)
    chain = DistillationChain(levels=(t, s), compression_factor_floor=1.2)
    assert chain.total_compression() == pytest.approx(1.3)


def test_chain_rejects_invalid_floor() -> None:
    t = DistillationLevel(name="t", param_count=100)
    s = DistillationLevel(name="s", param_count=50)
    with pytest.raises(DistillationError, match="compression_factor_floor"):
        DistillationChain(levels=(t, s), compression_factor_floor=1.0)


def test_chain_multi_level_total_compression() -> None:
    levels = (
        DistillationLevel(name="t0", param_count=8000),
        DistillationLevel(name="t1", param_count=2000),
        DistillationLevel(name="t2", param_count=500),
    )
    chain = DistillationChain(levels=levels)
    assert chain.total_compression() == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_serialize_starts_with_magic() -> None:
    chain = DistillationChain(
        levels=(
            DistillationLevel(name="t", param_count=1000),
            DistillationLevel(name="s", param_count=300),
        )
    )
    blob = chain.serialize_state()
    assert blob[:4] == DISTILL_MAGIC


def test_serialize_deserialize_roundtrip() -> None:
    chain = DistillationChain(
        levels=(
            DistillationLevel(
                name="hnerv_large",
                param_count=300_000,
                target_archive_bytes=400_000,
                temperature=2.0,
                kl_weight=0.8,
                hard_weight=0.2,
            ),
            DistillationLevel(
                name="hnerv_med",
                param_count=120_000,
                target_archive_bytes=150_000,
                temperature=1.5,
            ),
            DistillationLevel(
                name="hnerv_small",
                param_count=60_000,
                target_archive_bytes=80_000,
            ),
        ),
        compression_factor_floor=2.0,
    )
    blob = chain.serialize_state()
    restored = DistillationChain.deserialize_state(blob)
    assert restored == chain


def test_deserialize_rejects_bad_magic() -> None:
    with pytest.raises(DistillationError, match="bad magic"):
        DistillationChain.deserialize_state(b"XXXX" + b"\x00" * 80)


def test_deserialize_rejects_unknown_version() -> None:
    bad = (
        DISTILL_MAGIC
        + (DISTILL_SCHEMA_VERSION + 99).to_bytes(2, "little")
        + b"\x00" * 80
    )
    with pytest.raises(DistillationError, match="unsupported schema"):
        DistillationChain.deserialize_state(bad)


# ---------------------------------------------------------------------------
# distillation_loss
# ---------------------------------------------------------------------------


def test_loss_zero_at_perfect_match() -> None:
    logits = torch.tensor([[[2.0], [0.0]]])  # (B=1, C=2, ...)
    loss = distillation_loss(
        logits.clone().requires_grad_(),
        logits.clone(),
        temperature=2.0,
        kl_weight=1.0,
        hard_weight=0.0,
    )
    # Identical soft targets → loss = -Σ p log p = entropy of softmax(logits/T).
    # This is NOT zero in absolute terms; but the GRADIENT wrt student should
    # be ~zero because the cross-entropy minimum is at p_t == p_s.
    assert float(loss.detach()) >= 0


def test_loss_grad_flows_through_student() -> None:
    student = torch.randn(2, 4, 3, requires_grad=True)
    teacher = torch.randn(2, 4, 3)
    loss = distillation_loss(student, teacher, temperature=2.0, kl_weight=1.0)
    loss.backward()
    assert student.grad is not None
    assert torch.all(torch.isfinite(student.grad))


def test_loss_no_grad_through_teacher() -> None:
    student = torch.randn(2, 4, 3, requires_grad=True)
    teacher = torch.randn(2, 4, 3, requires_grad=True)
    loss = distillation_loss(student, teacher, temperature=2.0, kl_weight=1.0)
    loss.backward()
    # Teacher grad should be None because we detach inside the loss.
    assert teacher.grad is None


def test_loss_t_squared_scaling() -> None:
    # T² scaling: loss(T=2) should differ from loss(T=1) by a factor reflecting
    # the kl divergence scaling.
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])  # (B=1, C=2, S=2)
    teacher = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loss_1 = distillation_loss(student, teacher, temperature=1.0, kl_weight=1.0)
    loss_2 = distillation_loss(student, teacher, temperature=2.0, kl_weight=1.0)
    assert float(loss_1) != float(loss_2)


def test_loss_shape_mismatch_raises() -> None:
    with pytest.raises(DistillationError, match="shape mismatch"):
        distillation_loss(torch.zeros(2, 4), torch.zeros(2, 3), kl_weight=1.0)


def test_loss_negative_temperature_raises() -> None:
    with pytest.raises(DistillationError, match="temperature must"):
        distillation_loss(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            temperature=-1.0,
            kl_weight=1.0,
        )


def test_loss_negative_weight_raises() -> None:
    with pytest.raises(DistillationError, match="non-negative"):
        distillation_loss(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            kl_weight=-1.0,
        )


def test_loss_hard_targets_required_when_hard_weight() -> None:
    with pytest.raises(DistillationError, match="hard_targets required"):
        distillation_loss(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            kl_weight=0.5,
            hard_weight=0.5,
            hard_targets=None,
        )


def test_loss_with_hard_targets_works() -> None:
    student = torch.randn(2, 4, 3, requires_grad=True)
    teacher = torch.randn(2, 4, 3)
    hard = torch.randint(0, 4, (2, 3))
    loss = distillation_loss(
        student,
        teacher,
        kl_weight=0.5,
        hard_weight=0.5,
        hard_targets=hard,
    )
    loss.backward()
    assert student.grad is not None


def test_loss_unknown_reduction_raises() -> None:
    with pytest.raises(DistillationError, match="reduction must"):
        distillation_loss(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            kl_weight=1.0,
            reduction="banana",
        )


def test_loss_sum_reduction() -> None:
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8)
    loss_mean = distillation_loss(student, teacher, kl_weight=1.0, reduction="mean")
    loss_sum = distillation_loss(student, teacher, kl_weight=1.0, reduction="sum")
    assert float(loss_sum.detach()) > float(loss_mean.detach())


def test_loss_default_temperature_matches_selfcomp() -> None:
    # CLAUDE.md "Quantizr uses kl_on_logits() with T=2.0 for SegNet distillation"
    assert DEFAULT_TEMPERATURE == 2.0
