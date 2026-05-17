# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.training_curriculum.quantizr_5_stage_staircase`.

Op-routable #5 from ``.omx/research/cpu_frontier_master_gradient_campaign_plan_
20260517.md`` §1.3. Covers:

* QuantizrFiveStageStaircase dataclass invariants
* StaircaseStage validation (canonical-set names, non-empty fields,
  max_epochs sentinel, frozen-dataclass mutation rejection)
* Canonical factory ``from_quantizr_canonical`` returns the empirically-
  validated 5-stage schedule
* Stage scheduling arithmetic delegates correctly to
  ``multi_stage_curriculum.StageScheduler``
* Transition records carry weight-domain action_keys
  (``bn_stats_frozen`` / ``lsq_inserted`` / ``fp4_fakequant_inserted`` /
  ``param_group_frozen:*`` / ``ema_shadow_promoted_to_inference``)
* ``freeze_bn_stats`` idempotency on PyTorch BatchNorm modules
* ``freeze_param_groups`` prefix-matching + idempotency
* ``apply_ema_shadow_to_inference`` snapshot+restore semantics via real
  :class:`tac.training.EMA` instance
* JSON-serializable ``as_dict()`` representation for design-memo cite
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn

from tac.training_curriculum.multi_stage_curriculum import StageScheduler
from tac.training_curriculum.quantizr_5_stage_staircase import (
    QUANTIZR_CANONICAL_STAGES,
    QUANTIZR_DEFAULT_EPOCHS,
    QuantizrFiveStageStaircase,
    QuantizrStaircaseError,
    StageTransitionRecord,
    StaircaseStage,
    apply_ema_shadow_to_inference,
    freeze_bn_stats,
    freeze_param_groups,
)

# ──────────────────────────────────────────────────────────────────────
# StaircaseStage validation
# ──────────────────────────────────────────────────────────────────────


def _canonical_anchor_kwargs() -> dict:
    return {
        "name": "anchor",
        "epochs": 100,
        "active_loss_terms": frozenset({"pixel"}),
        "frozen_param_groups": frozenset(),
        "ema_shadow_groups": frozenset({"all"}),
        "convergence_metric": "val_pixel_loss",
        "convergence_threshold": 0.05,
        "notes": "canonical anchor",
    }


def test_staircase_stage_canonical_construct() -> None:
    stage = StaircaseStage(**_canonical_anchor_kwargs())
    assert stage.name == "anchor"
    assert stage.epochs == 100
    assert stage.max_epochs == 100  # sentinel resolved to epochs


def test_staircase_stage_name_must_be_in_canonical_set() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["name"] = "warmup"  # not in canonical set
    with pytest.raises(QuantizrStaircaseError, match="canonical set"):
        StaircaseStage(**kwargs)


def test_staircase_stage_epochs_must_be_positive() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["epochs"] = 0
    with pytest.raises(QuantizrStaircaseError, match="epochs=0"):
        StaircaseStage(**kwargs)


def test_staircase_stage_max_epochs_sentinel_resolves_to_epochs() -> None:
    stage = StaircaseStage(**_canonical_anchor_kwargs())
    assert stage.max_epochs == stage.epochs


def test_staircase_stage_max_epochs_explicit_override() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["max_epochs"] = 150
    stage = StaircaseStage(**kwargs)
    assert stage.max_epochs == 150


def test_staircase_stage_max_epochs_below_epochs_rejected() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["max_epochs"] = 50  # < epochs=100
    with pytest.raises(QuantizrStaircaseError, match="max_epochs=50"):
        StaircaseStage(**kwargs)


def test_staircase_stage_active_loss_terms_must_be_nonempty() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["active_loss_terms"] = frozenset()
    with pytest.raises(QuantizrStaircaseError, match="active_loss_terms"):
        StaircaseStage(**kwargs)


def test_staircase_stage_notes_must_be_nonempty() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["notes"] = "   "  # whitespace-only
    with pytest.raises(QuantizrStaircaseError, match="notes"):
        StaircaseStage(**kwargs)


def test_staircase_stage_convergence_metric_must_be_nonempty() -> None:
    kwargs = _canonical_anchor_kwargs()
    kwargs["convergence_metric"] = ""
    with pytest.raises(QuantizrStaircaseError, match="convergence_metric"):
        StaircaseStage(**kwargs)


def test_staircase_stage_is_frozen() -> None:
    stage = StaircaseStage(**_canonical_anchor_kwargs())
    with pytest.raises(FrozenInstanceError):
        stage.epochs = 200  # type: ignore[misc]


def test_staircase_stage_as_dict_keys() -> None:
    stage = StaircaseStage(**_canonical_anchor_kwargs())
    out = stage.as_dict()
    assert set(out.keys()) == {
        "name",
        "epochs",
        "max_epochs",
        "active_loss_terms",
        "frozen_param_groups",
        "ema_shadow_groups",
        "convergence_metric",
        "convergence_threshold",
        "insert_lsq",
        "insert_fp4_fakequant",
        "promote_ema_to_inference",
        "notes",
    }
    # frozenset → sorted list for JSON serializability
    assert out["active_loss_terms"] == ["pixel"]
    assert out["frozen_param_groups"] == []


def test_staircase_stage_as_curriculum_stage_projects_correctly() -> None:
    stage = StaircaseStage(**_canonical_anchor_kwargs())
    generic = stage.as_curriculum_stage()
    assert generic.name == "anchor"
    assert generic.epochs == 100
    assert generic.loss_key == "pixel"
    # Anchor canonically uses "reset" policy
    assert generic.optimizer_state_policy == "reset"


def test_staircase_stage_as_curriculum_stage_qat_uses_lr_reset() -> None:
    stage = StaircaseStage(
        name="qat",
        epochs=150,
        active_loss_terms=frozenset({"pixel", "scorer"}),
        frozen_param_groups=frozenset({"bn_stats"}),
        ema_shadow_groups=frozenset({"all"}),
        convergence_metric="val_score_combined",
        convergence_threshold=0.33,
        insert_lsq=True,
        insert_fp4_fakequant=True,
        notes="qat stage with LSQ + FP4",
    )
    generic = stage.as_curriculum_stage()
    assert generic.optimizer_state_policy == "inherit_lr_reset"


def test_staircase_stage_as_curriculum_stage_finetune_inherits() -> None:
    stage = StaircaseStage(
        name="finetune",
        epochs=200,
        active_loss_terms=frozenset({"pixel", "kl"}),
        frozen_param_groups=frozenset(),
        ema_shadow_groups=frozenset({"all"}),
        convergence_metric="val_segnet_distortion",
        convergence_threshold=0.005,
        notes="finetune stage",
    )
    generic = stage.as_curriculum_stage()
    assert generic.optimizer_state_policy == "inherit"


# ──────────────────────────────────────────────────────────────────────
# QuantizrFiveStageStaircase canonical factory
# ──────────────────────────────────────────────────────────────────────


def test_canonical_factory_returns_5_stages_in_order() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    assert len(staircase.stages) == 5
    assert tuple(s.name for s in staircase.stages) == QUANTIZR_CANONICAL_STAGES


def test_canonical_factory_total_epochs_matches_default_budget() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    expected = sum(QUANTIZR_DEFAULT_EPOCHS.values())
    assert staircase.total_epochs == expected
    assert staircase.total_epochs == 700  # 100+200+200+150+50


def test_canonical_factory_stage_specific_primitives() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    stages = {s.name: s for s in staircase.stages}

    # anchor: no freezes, no LSQ, no FP4, no EMA-promote
    assert stages["anchor"].frozen_param_groups == frozenset()
    assert not stages["anchor"].insert_lsq
    assert not stages["anchor"].insert_fp4_fakequant
    assert not stages["anchor"].promote_ema_to_inference

    # finetune: adds KL distill loss term
    assert "kl_distill_segnet_T2" in stages["finetune"].active_loss_terms

    # joint: adds posenet
    assert "posenet" in stages["joint"].active_loss_terms

    # qat: BN frozen + LSQ + FP4 inserted
    assert "bn_stats" in stages["qat"].frozen_param_groups
    assert stages["qat"].insert_lsq
    assert stages["qat"].insert_fp4_fakequant
    assert not stages["qat"].promote_ema_to_inference

    # final: all but pose-axis frozen + EMA promoted, ema_shadow_groups empty
    assert "bn_stats" in stages["final"].frozen_param_groups
    assert "renderer_trunk" in stages["final"].frozen_param_groups
    assert "renderer_heads" in stages["final"].frozen_param_groups
    assert stages["final"].promote_ema_to_inference
    assert stages["final"].ema_shadow_groups == frozenset()


def test_canonical_factory_epoch_budget_override_partial() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical(
        epoch_budget={"qat": 50, "final": 25}
    )
    stages = {s.name: s for s in staircase.stages}
    assert stages["qat"].epochs == 50
    assert stages["final"].epochs == 25
    # other stages retain defaults
    assert stages["anchor"].epochs == QUANTIZR_DEFAULT_EPOCHS["anchor"]


def test_canonical_factory_unknown_stage_in_override_rejected() -> None:
    with pytest.raises(QuantizrStaircaseError, match="not in canonical"):
        QuantizrFiveStageStaircase.from_quantizr_canonical(
            epoch_budget={"warmup": 50}
        )


def test_staircase_requires_exactly_5_stages() -> None:
    only_4 = QuantizrFiveStageStaircase.from_quantizr_canonical().stages[:4]
    with pytest.raises(QuantizrStaircaseError, match="exactly 5"):
        QuantizrFiveStageStaircase(stages=only_4)


def test_staircase_requires_canonical_order() -> None:
    canonical = QuantizrFiveStageStaircase.from_quantizr_canonical().stages
    swapped = (canonical[1], canonical[0], *canonical[2:])  # finetune before anchor
    with pytest.raises(QuantizrStaircaseError, match="in order"):
        QuantizrFiveStageStaircase(stages=swapped)


# ──────────────────────────────────────────────────────────────────────
# Stage scheduling delegates to multi_stage_curriculum.StageScheduler
# ──────────────────────────────────────────────────────────────────────


def test_scheduler_property_returns_stagescheduler() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    assert isinstance(staircase.scheduler, StageScheduler)


def test_quantizr_staircase_is_package_discoverable() -> None:
    from tac.training_curriculum import (
        QuantizrFiveStageStaircase as ExportedQuantizrFiveStageStaircase,
    )

    assert ExportedQuantizrFiveStageStaircase is QuantizrFiveStageStaircase


def test_stage_for_epoch_canonical_boundaries() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    # Default budgets: anchor[0,100), finetune[100,300), joint[300,500), qat[500,650), final[650,700)
    assert staircase.stage_for_epoch(0).name == "anchor"
    assert staircase.stage_for_epoch(99).name == "anchor"
    assert staircase.stage_for_epoch(100).name == "finetune"
    assert staircase.stage_for_epoch(299).name == "finetune"
    assert staircase.stage_for_epoch(300).name == "joint"
    assert staircase.stage_for_epoch(499).name == "joint"
    assert staircase.stage_for_epoch(500).name == "qat"
    assert staircase.stage_for_epoch(649).name == "qat"
    assert staircase.stage_for_epoch(650).name == "final"
    assert staircase.stage_for_epoch(699).name == "final"


def test_stage_for_epoch_out_of_range_raises() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    with pytest.raises(QuantizrStaircaseError):
        staircase.stage_for_epoch(700)  # one past the end


def test_is_transition_epoch_at_stage_boundaries() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    # First epoch of each non-first stage is a transition
    assert not staircase.is_transition_epoch(0)
    assert staircase.is_transition_epoch(100)  # anchor -> finetune
    assert staircase.is_transition_epoch(300)  # finetune -> joint
    assert staircase.is_transition_epoch(500)  # joint -> qat
    assert staircase.is_transition_epoch(650)  # qat -> final
    # Non-boundary epochs are not transitions
    assert not staircase.is_transition_epoch(50)
    assert not staircase.is_transition_epoch(450)


# ──────────────────────────────────────────────────────────────────────
# Transition records carry weight-domain action_keys
# ──────────────────────────────────────────────────────────────────────


def test_qat_transition_emits_bn_lsq_fp4_action_keys() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    transition = staircase.transition_at_epoch(500)  # joint -> qat
    assert isinstance(transition, StageTransitionRecord)
    assert transition.from_stage_name == "joint"
    assert transition.to_stage_name == "qat"
    assert "bn_stats_frozen" in transition.action_keys
    assert "lsq_inserted" in transition.action_keys
    assert "fp4_fakequant_inserted" in transition.action_keys


def test_final_transition_emits_ema_promote_and_param_freeze() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    transition = staircase.transition_at_epoch(650)  # qat -> final
    assert transition.from_stage_name == "qat"
    assert transition.to_stage_name == "final"
    assert "ema_shadow_promoted_to_inference" in transition.action_keys
    assert "param_group_frozen:renderer_trunk" in transition.action_keys
    assert "param_group_frozen:renderer_heads" in transition.action_keys
    # bn_stats also frozen at final stage (inherited from qat)
    assert "bn_stats_frozen" in transition.action_keys


def test_anchor_to_finetune_transition_has_no_weight_domain_actions() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    transition = staircase.transition_at_epoch(100)  # anchor -> finetune
    # finetune has no freezes / no LSQ / no FP4 / no EMA-promote
    assert "bn_stats_frozen" not in transition.action_keys
    assert "lsq_inserted" not in transition.action_keys
    assert "fp4_fakequant_inserted" not in transition.action_keys
    assert "ema_shadow_promoted_to_inference" not in transition.action_keys
    # But loss-swap action from generic scheduler is preserved
    assert "loss_swapped" in transition.action_keys


def test_transition_at_non_transition_epoch_raises() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    with pytest.raises(QuantizrStaircaseError):
        staircase.transition_at_epoch(50)  # mid-anchor; not a transition


# ──────────────────────────────────────────────────────────────────────
# Convergence criterion lookup
# ──────────────────────────────────────────────────────────────────────


def test_convergence_criterion_for_qat() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    metric, threshold = staircase.convergence_criterion_for_stage("qat")
    assert metric == "val_score_combined"
    assert threshold == pytest.approx(0.33)


def test_convergence_criterion_unknown_stage_raises() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    with pytest.raises(QuantizrStaircaseError, match="canonical set"):
        staircase.convergence_criterion_for_stage("warmup")


# ──────────────────────────────────────────────────────────────────────
# as_dict() JSON-serializable representation
# ──────────────────────────────────────────────────────────────────────


def test_staircase_as_dict_top_level_keys() -> None:
    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    out = staircase.as_dict()
    assert out["staircase_schema_version"] == "quantizr_five_stage_v1"
    assert out["total_epochs"] == 700
    assert len(out["stages"]) == 5
    assert "PR55" in out["canonical_anchor"]


def test_staircase_as_dict_is_json_serializable() -> None:
    import json

    staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
    payload = staircase.as_dict()
    # Round-trip through JSON to prove all values are serializable
    s = json.dumps(payload)
    parsed = json.loads(s)
    assert parsed["total_epochs"] == 700


# ──────────────────────────────────────────────────────────────────────
# freeze_bn_stats — PyTorch BatchNorm idempotency
# ──────────────────────────────────────────────────────────────────────


class _TinyConvWithBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.head = nn.Linear(16, 4)


def test_freeze_bn_stats_counts_bn_modules() -> None:
    model = _TinyConvWithBN()
    model.train()  # ensure BN is in training mode initially
    assert model.bn1.training
    assert model.bn2.training
    count = freeze_bn_stats(model)
    assert count == 2
    # After freeze, BN modules are in eval mode (running stats frozen)
    assert not model.bn1.training
    assert not model.bn2.training


def test_freeze_bn_stats_idempotent() -> None:
    model = _TinyConvWithBN()
    model.train()
    freeze_bn_stats(model)
    # Calling again is a no-op (BN modules already in eval mode)
    count2 = freeze_bn_stats(model)
    assert count2 == 2  # still counts; just no state change
    assert not model.bn1.training
    assert not model.bn2.training


def test_freeze_bn_stats_no_bn_modules_returns_zero() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    assert freeze_bn_stats(model) == 0


# ──────────────────────────────────────────────────────────────────────
# freeze_param_groups — prefix-matching
# ──────────────────────────────────────────────────────────────────────


def test_freeze_param_groups_freezes_matching_prefix() -> None:
    model = _TinyConvWithBN()
    # All params initially require grad
    assert all(p.requires_grad for p in model.parameters())
    count = freeze_param_groups(model, ("conv1", "bn1"))
    # conv1.weight, conv1.bias, bn1.weight, bn1.bias = 4 params
    assert count == 4
    # conv1 + bn1 frozen
    assert not model.conv1.weight.requires_grad
    assert not model.bn1.weight.requires_grad
    # conv2 + head remain trainable
    assert model.conv2.weight.requires_grad
    assert model.head.weight.requires_grad


def test_freeze_param_groups_idempotent() -> None:
    model = _TinyConvWithBN()
    freeze_param_groups(model, ("conv1",))
    # Calling again returns 0 because all matching params are already frozen
    count2 = freeze_param_groups(model, ("conv1",))
    assert count2 == 0


def test_freeze_param_groups_empty_iterable_no_op() -> None:
    model = _TinyConvWithBN()
    assert freeze_param_groups(model, ()) == 0
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_param_groups_no_match_returns_zero() -> None:
    model = _TinyConvWithBN()
    assert freeze_param_groups(model, ("nonexistent_prefix",)) == 0
    assert all(p.requires_grad for p in model.parameters())


# ──────────────────────────────────────────────────────────────────────
# apply_ema_shadow_to_inference — snapshot+restore semantics
# ──────────────────────────────────────────────────────────────────────


def test_apply_ema_shadow_promotes_shadow_to_model() -> None:
    from tac.training import EMA

    model = _TinyConvWithBN()
    ema = EMA(model, decay=0.997)
    # Take a snapshot of the initial weights
    initial_weight = model.conv1.weight.detach().clone()
    # Mutate model weights (simulating training)
    with torch.no_grad():
        model.conv1.weight.add_(torch.randn_like(model.conv1.weight))
    # EMA shadow has NOT been updated, so it still holds initial weights
    # Promote shadow -> inference
    apply_ema_shadow_to_inference(model, ema)
    # Model now holds shadow (= initial weights)
    assert torch.allclose(model.conv1.weight, initial_weight)


def test_apply_ema_shadow_rejects_non_ema_object() -> None:
    model = _TinyConvWithBN()

    class _NoApply:
        pass

    with pytest.raises(QuantizrStaircaseError, match=r"no \.apply method"):
        apply_ema_shadow_to_inference(model, _NoApply())
