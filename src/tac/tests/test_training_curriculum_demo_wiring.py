# SPDX-License-Identifier: MIT
"""Tests for tac.training_curriculum demo wiring modules.

Lane: lane_pausing_exploits_wave_20260517
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.training_curriculum import (
    DiagnosticCheckpoint,
    GreedyModelSoup,
    StageScheduler,
    UniformModelSoup,
    pause_and_capture,
)
from tac.training_curriculum.demo_nscs01_wiring import (
    NSCS01CurriculumRecipe,
    build_nscs01_canonical_curriculum,
    head0_grad_norm_metric_fn,
    head1_grad_norm_metric_fn,
    nscs01_diagnostic_metric_fns,
)
from tac.training_curriculum.demo_nscs03_wiring import (
    NSCS03SoupRecipe,
    apply_nscs03_soup_recipe,
    nscs03_per_subnet_l2_norms,
)


# ----- NSCS01 demo wiring -----


class _NSCS01Stub(nn.Module):
    """Minimal stub mirroring NSCS01's frame_0_head + frame_1_head shape."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_0_head = nn.Linear(4, 2)
        self.frame_1_head = nn.Linear(4, 2)
        self.latents = nn.Parameter(torch.zeros(8, 16))


def test_nscs01_canonical_curriculum_structure() -> None:
    sched = build_nscs01_canonical_curriculum()
    assert sched.total_epochs == 1000
    stages = sched.stages
    assert len(stages) == 3
    assert stages[0].name == "anchor"
    assert stages[0].epochs == 100
    assert stages[0].loss_key == "pixel_only_both_heads"
    assert stages[0].optimizer_state_policy == "reset"
    assert stages[1].name == "joint"
    assert stages[1].epochs == 600
    assert stages[1].loss_key == "pixel_plus_split_scorer_lagrangian"
    assert stages[1].optimizer_state_policy == "inherit_lr_reset"
    assert stages[2].name == "distill"
    assert stages[2].epochs == 300
    assert "kl_distill" in stages[2].loss_key
    assert stages[2].lr_multiplier == 0.1
    assert stages[2].optimizer_state_policy == "inherit"


def test_nscs01_curriculum_transitions_at_expected_epochs() -> None:
    sched = build_nscs01_canonical_curriculum()
    assert sched.is_transition_epoch(0) is False
    assert sched.is_transition_epoch(100) is True  # anchor → joint
    assert sched.is_transition_epoch(700) is True  # joint → distill
    assert sched.is_transition_epoch(99) is False
    assert sched.is_transition_epoch(101) is False

    t1 = sched.transition_at_epoch(100)
    assert t1.from_stage_name == "anchor"
    assert t1.to_stage_name == "joint"
    assert "loss_swapped" in t1.action_keys
    assert "scheduler_reset" in t1.action_keys

    t2 = sched.transition_at_epoch(700)
    assert t2.from_stage_name == "joint"
    assert t2.to_stage_name == "distill"


def test_head0_metric_fn_returns_positive() -> None:
    m = _NSCS01Stub()
    # Default Linear has nonzero weight initialization
    n = head0_grad_norm_metric_fn(m)
    assert n > 0


def test_head1_metric_fn_returns_positive() -> None:
    m = _NSCS01Stub()
    n = head1_grad_norm_metric_fn(m)
    assert n > 0


def test_head_metric_fns_zero_for_missing_attr() -> None:
    m = nn.Linear(4, 2)  # no frame_0_head/frame_1_head
    assert head0_grad_norm_metric_fn(m) == 0.0
    assert head1_grad_norm_metric_fn(m) == 0.0


def test_nscs01_diagnostic_metric_fns_complete_registry() -> None:
    registry = nscs01_diagnostic_metric_fns()
    assert "head0_l2_norm" in registry
    assert "head1_l2_norm" in registry
    for name, (fn, axis, rationale) in registry.items():
        assert callable(fn)
        assert axis == "diagnostic"
        assert rationale  # non-empty


def test_nscs01_recipe_validates_total_epochs() -> None:
    NSCS01CurriculumRecipe()  # default OK
    with pytest.raises(ValueError):
        NSCS01CurriculumRecipe(total_epochs=500)


def test_nscs01_pause_and_capture_end_to_end(tmp_path: Path) -> None:
    """Demonstrate the full demo wiring: build curriculum, pause at
    transitions, capture diagnostic metrics."""
    m = _NSCS01Stub()
    sched = build_nscs01_canonical_curriculum()
    transition_epochs = [100, 700, sched.total_epochs - 1]
    checkpoints: list[DiagnosticCheckpoint] = []
    for ep in transition_epochs:
        # NOTE: epoch 999 (final) is not a transition per the scheduler API,
        # but we capture there too as the recipe specifies.
        ckpt = pause_and_capture(
            m,
            epoch=ep,
            output_dir=tmp_path,
            substrate_id="nscs01_demo",
            metric_fns=nscs01_diagnostic_metric_fns(),
            notes=f"demo pause at epoch {ep}",
            utc_iso=f"2026-05-17T12:00:0{ep % 10}Z",
        )
        checkpoints.append(ckpt)

    assert len(checkpoints) == 3
    for ckpt in checkpoints:
        assert len(ckpt.metrics) == 2
        metric_names = {m.name for m in ckpt.metrics}
        assert metric_names == {"head0_l2_norm", "head1_l2_norm"}
        assert Path(ckpt.state_dict_path).exists()


# ----- NSCS03 demo wiring -----


def test_nscs03_soup_recipe_validates() -> None:
    NSCS03SoupRecipe()  # default OK
    with pytest.raises(ValueError):
        NSCS03SoupRecipe(lambda_r_grid=())
    with pytest.raises(ValueError):
        NSCS03SoupRecipe(lambda_r_grid=(0.0, 0.1))
    with pytest.raises(ValueError):
        NSCS03SoupRecipe(soup_kind="bogus")
    with pytest.raises(ValueError):
        NSCS03SoupRecipe(held_out_axis="macOS-CPU advisory")


def test_nscs03_uniform_soup_recipe_end_to_end() -> None:
    recipe = NSCS03SoupRecipe(
        lambda_r_grid=(0.01, 0.05), soup_kind="uniform"
    )
    ckpts = {
        0.01: {"w": torch.tensor([1.0, 2.0])},
        0.05: {"w": torch.tensor([3.0, 4.0])},
    }
    result = apply_nscs03_soup_recipe(
        recipe=recipe, per_lambda_checkpoints=ckpts
    )
    assert result.num_checkpoints_in_soup == 2
    assert torch.allclose(result.soup_state_dict["w"], torch.tensor([2.0, 3.0]))


def test_nscs03_greedy_soup_recipe_end_to_end() -> None:
    recipe = NSCS03SoupRecipe(
        lambda_r_grid=(0.01, 0.05, 0.10), soup_kind="greedy"
    )
    ckpts = {
        0.01: {"w": torch.tensor([1.0])},
        0.05: {"w": torch.tensor([2.0])},
        0.10: {"w": torch.tensor([100.0])},  # poison
    }

    def metric(sd: dict[str, torch.Tensor]) -> float:
        return abs(float(sd["w"].item()) - 1.5)

    result = apply_nscs03_soup_recipe(
        recipe=recipe,
        per_lambda_checkpoints=ckpts,
        held_out_metric_fn=metric,
    )
    # Poison λ_R=0.10 should be rejected by greedy filter
    assert "lambda_r_0.1000" not in result.checkpoint_keys_kept
    assert result.num_checkpoints_in_soup == 2


def test_nscs03_recipe_refuses_lambda_mismatch() -> None:
    recipe = NSCS03SoupRecipe(
        lambda_r_grid=(0.01, 0.05), soup_kind="uniform"
    )
    with pytest.raises(ValueError):
        apply_nscs03_soup_recipe(
            recipe=recipe,
            per_lambda_checkpoints={0.01: {"w": torch.tensor([1.0])}},
        )


def test_nscs03_recipe_refuses_greedy_without_metric() -> None:
    recipe = NSCS03SoupRecipe(soup_kind="greedy")
    ckpts = {lr: {"w": torch.tensor([1.0])} for lr in recipe.lambda_r_grid}
    with pytest.raises(ValueError):
        apply_nscs03_soup_recipe(
            recipe=recipe, per_lambda_checkpoints=ckpts
        )


def test_nscs03_per_subnet_l2_norms_picks_up_subnet_prefixes() -> None:
    state = {
        "g_a.weight": torch.tensor([1.0, 0.0]),
        "g_s.weight": torch.tensor([0.0, 1.0]),
        "h_a.weight": torch.tensor([3.0, 4.0]),
        "h_s.weight": torch.tensor([1.0]),
        "entropy_bottleneck.weight": torch.tensor([2.0]),
        "other.weight": torch.tensor([99.0]),  # not in any subnet
    }
    norms = nscs03_per_subnet_l2_norms(state)
    assert norms["g_a"] == pytest.approx(1.0)
    assert norms["g_s"] == pytest.approx(1.0)
    assert norms["h_a"] == pytest.approx(5.0)
    assert norms["h_s"] == pytest.approx(1.0)
    assert norms["entropy_bottleneck"] == pytest.approx(2.0)
    assert "other" not in norms


def test_nscs03_per_subnet_l2_norms_missing_subnet_returns_zero() -> None:
    state = {"x.weight": torch.tensor([1.0])}
    norms = nscs03_per_subnet_l2_norms(state, sub_net_prefixes=("g_a",))
    assert norms["g_a"] == 0.0
