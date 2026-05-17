# SPDX-License-Identifier: MIT
"""Quantizr 5-stage staircase — empirically-validated training schedule helper.

Op-routable #5 from ``.omx/research/cpu_frontier_master_gradient_campaign_plan_
20260517.md`` §1.3. Per HNeRV parity discipline L7 + Hotz's revision (operator-
approved 2026-05-17): ship Quantizr-specific FIRST as the canonical helper;
generalize later only when a SECOND substrate demonstrably needs a different
schedule.

Quantizr's PR #55 (0.33 contest-CUDA) used a 5-stage training pipeline per
``project_quantizr_full_intel_20260421.md`` and
``project_quantizr_definitive_binary_analysis.md``:

1. **Anchor** — pixel-loss warmup, EMA active, BN trainable
2. **Anchor boost / finetune** — SegNet KL distillation T=2.0 + hard mining
3. **Pose finetune** — PoseNet loss path active
4. **Joint / QAT** — combined scorer objective + quantization-aware training
5. **Micro / final** — low-LR polish with EMA inference promotion

The empirical evidence anchoring this schedule is the 0.33 [contest-CUDA]
score on Quantizr's archive (299,970 bytes; FiLM+DSConv 88K params; FP4 +
Brotli + AV1 mask packet). This is historical architecture/training evidence,
not current 0.192-frontier authority. PR101 gold (0.193) used a related
multi-stage bolt-on discipline; the canonical choice points (BN freeze cadence,
EMA shadow groups, LSQ activation, FP4 fakequant insertion) are treated here as
training-practice priors that still require byte-closed trainer adoption before
any score-moving claim.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Stage scheduling arithmetic → ADOPT canonical
  :class:`tac.training_curriculum.multi_stage_curriculum.StageScheduler`.
  No reason to fork the integer epoch accounting.
* Stage-specific transition primitives → UNIQUE to Quantizr substrate
  (BN freeze, EMA shadow groups, LSQ activation, FP4 fakequant insertion);
  generic ``multi_stage_curriculum`` cannot express these because they are
  weight-domain transformations, not loss-domain swaps.
* EMA decay → ADOPT canonical 0.997 from :class:`tac.training.EMA`.
* LSQ step size init → ADOPT canonical :class:`tac.quantization.LSQScale`.
* FP4 codebook → ADOPT canonical ``fake_quant_fp4`` /
  :class:`tac.fp4_quantize.FakeQuantFP4` primitives (per FP4 hardware
  disclosure discipline).

Cargo-cult audit per assumption
───────────────────────────────
* "5 stages is empirically optimal" — HARD-EARNED only for PR55's measured
  architecture/training packet (0.33 [contest-CUDA]).
  CARGO-CULTED if applied without justification to substrates with materially
  different architectures (e.g., HNeRV-family had a different bolt-on cadence).
  Future second-substrate adopter must justify the 5-stage choice in its
  design memo per Catalog #303.
* "BN should be frozen at QAT stage entry" — HARD-EARNED per standard QAT
  practice (Jacob et al. 2018 §4); BN running stats drift under fake-quantized
  weights otherwise.
* "FP4 fakequant inserted only at stage 4" — HARD-EARNED at PR55; inserting
  earlier (e.g., stage 2 or 3) destabilizes pixel + scorer convergence.
* "EMA shadow promoted to inference at stage 5" — CANONICAL across all training
  paths per the EMA non-negotiable in CLAUDE.md; not specific to Quantizr.

## Observability surface

* Inspectable per layer — every stage carries an `active_loss_terms` set,
  `frozen_param_groups` set, `ema_shadow_groups` set, `convergence_metric` +
  `convergence_threshold` so the operator can decompose per-stage behavior.
* Decomposable per signal — stage transitions emit a `StageTransitionRecord`
  with action_keys so the operator sees exactly which primitives fired.
* Diff-able across runs — the canonical 5-stage factory returns a deterministic
  configuration; two runs of the same substrate produce byte-identical
  schedules.
* Queryable post-hoc — `QuantizrFiveStageStaircase.as_dict()` emits JSON-
  serializable representation for archive build manifest + design memo cite.
* Cite-able — every stage carries `notes` field with literature anchor
  (e.g., "Quantizr PR55 §4.2 + Jacob QAT 2018 §4").
* Counterfactual-able — operator can fork the canonical factory and modify
  per-stage settings to probe alternative schedules; the factory is a plain
  dataclass.

## 9-dimension success checklist evidence

1. **UNIQUENESS** — Quantizr-specific factory; generic
   `multi_stage_curriculum.StageScheduler` is the sister helper. Two distinct
   roles; not within-class refinement.
2. **BEAUTY + ELEGANCE** — ~150 LOC; PR101-style 30-sec-reviewable.
3. **DISTINCTNESS** — explicit per-stage BN/EMA/LSQ/FP4 primitives that the
   generic curriculum helper cannot express (it operates on loss-domain only).
4. **RIGOR** — empirical anchor (PR55 0.33); literature anchor (Jacob 2018);
   premise verified pre-edit against the canonical `EMA`, `LSQScale`, and FP4
   primitive sources in `src/tac/training.py`, `src/tac/quantization.py`, and
   `src/tac/fp4_quantize.py`.
5. **OPTIMIZATION PER TECHNIQUE** — every per-stage primitive routes through
   the canonical helper (no reimplementation of EMA / LSQ / FP4).
6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal to architecture (any
   substrate trainer with FiLM+DSConv can adopt); orthogonal to archive
   grammar (the stages are weight-side transformations; archive grammar is
   inflate-side).
7. **DETERMINISTIC REPRODUCIBILITY** — the canonical factory is a pure
   function of operator-supplied epoch counts; the returned dataclass is
   frozen + JSON-serializable.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — no runtime overhead beyond
   integer arithmetic per stage transition.
9. **OPTIMAL MINIMAL CONTEST SCORE** — PR55 empirical anchor 0.33
   [contest-CUDA]. Sub-0.30 reachable per Quantizr's own assessment "sub
   0.30 is possible just by sweeping conv dims" — the 5-stage schedule is
   not a score claim here; a byte-closed trainer adoption must prove whether
   this schedule moves the current frontier.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tac.training_curriculum.multi_stage_curriculum import (
    CurriculumStage,
    CurriculumStageBudgetError,
    StageScheduler,
)

__all__ = [
    "QUANTIZR_CANONICAL_STAGES",
    "QUANTIZR_DEFAULT_EPOCHS",
    "QuantizrFiveStageStaircase",
    "QuantizrStaircaseError",
    "StageTransitionRecord",
    "StaircaseStage",
    "apply_ema_shadow_to_inference",
    "freeze_bn_stats",
    "freeze_param_groups",
]


# Canonical stage names per PR55 deobfuscation (`project_quantizr_full_intel_20260421.md`).
QUANTIZR_CANONICAL_STAGES: tuple[str, ...] = (
    "anchor",
    "finetune",
    "joint",
    "qat",
    "final",
)

# Default local epoch budget for this reusable helper. PR55's historical packet
# used a longer schedule; this 700-epoch preset is an operator-adjustable budget,
# not a faithful PR55 reproduction claim.
QUANTIZR_DEFAULT_EPOCHS: dict[str, int] = {
    "anchor": 100,
    "finetune": 200,
    "joint": 200,
    "qat": 150,
    "final": 50,
}


class QuantizrStaircaseError(ValueError):
    """Raised when the staircase config violates the canonical contract."""


@dataclass(frozen=True)
class StaircaseStage:
    """One stage of the Quantizr 5-stage staircase with weight-domain primitives.

    Extends :class:`CurriculumStage` with the per-stage primitives that the
    generic curriculum helper cannot express (BN freeze, EMA shadow groups,
    LSQ activation, FP4 fakequant insertion).

    Args:
        name: Stage name; MUST be one of :data:`QUANTIZR_CANONICAL_STAGES`.
        epochs: Integer epoch budget for this stage (>= 1).
        active_loss_terms: Frozenset of loss term identifiers active in this
            stage (e.g. ``frozenset({"pixel"})`` for anchor;
            ``frozenset({"pixel", "kl_distill_segnet_T2"})`` for finetune).
        frozen_param_groups: Frozenset of param group identifiers frozen in
            this stage (e.g. ``frozenset({"bn_stats"})`` for qat;
            ``frozenset({"bn_stats", "renderer_trunk", "renderer_heads"})``
            for final, leaving only pose-axis trainable).
        ema_shadow_groups: Frozenset of param group identifiers tracked by
            EMA shadow in this stage (canonical ``frozenset({"all"})`` for
            stages 1-4; ``frozenset()`` for stage 5 where EMA shadow is
            promoted to inference weights and no further EMA updates apply).
        convergence_metric: Operator-readable metric identifier used to detect
            stage convergence (e.g. ``"val_pixel_loss"`` /
            ``"val_pose_distortion"`` / ``"val_score_combined"``).
        convergence_threshold: Threshold value for convergence_metric (e.g.
            ``0.05`` for ``val_pixel_loss``). The trainer SHOULD transition to
            the next stage when the metric drops below this threshold OR when
            ``epochs`` are exhausted, whichever comes first.
        max_epochs: Hard cap on stage epochs (>= epochs). Defaults to
            ``epochs`` (no early-stop slack); operator can pass larger value
            to allow stage to run longer if convergence is slow.
        insert_lsq: Whether to insert :class:`tac.quantization.LSQScale` on
            this stage's entry transition. Default ``False``; True only for
            ``qat`` stage in the canonical factory.
        insert_fp4_fakequant: Whether to insert FP4 fakequant on this stage's
            entry transition. Default ``False``; True only for ``qat`` stage
            in the canonical factory.
        promote_ema_to_inference: Whether to promote EMA shadow to inference
            weights at this stage's entry transition. Default ``False``;
            True only for ``final`` stage in the canonical factory.
        notes: Operator-readable rationale; required per CLAUDE.md
            "Comment-only contracts are FORBIDDEN".
    """

    name: str
    epochs: int
    active_loss_terms: frozenset[str]
    frozen_param_groups: frozenset[str]
    ema_shadow_groups: frozenset[str]
    convergence_metric: str
    convergence_threshold: float
    max_epochs: int = 0  # 0 means "default to epochs"; resolved in __post_init__
    insert_lsq: bool = False
    insert_fp4_fakequant: bool = False
    promote_ema_to_inference: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.name not in QUANTIZR_CANONICAL_STAGES:
            raise QuantizrStaircaseError(
                f"StaircaseStage.name={self.name!r} not in canonical set "
                f"{QUANTIZR_CANONICAL_STAGES}; use generic "
                "tac.training_curriculum.multi_stage_curriculum.CurriculumStage "
                "for non-Quantizr substrates"
            )
        if self.epochs < 1:
            raise QuantizrStaircaseError(
                f"StaircaseStage(name={self.name!r}).epochs={self.epochs} must be >= 1"
            )
        # Resolve max_epochs sentinel; object.__setattr__ used because dataclass is frozen.
        if self.max_epochs == 0:
            object.__setattr__(self, "max_epochs", self.epochs)
        elif self.max_epochs < self.epochs:
            raise QuantizrStaircaseError(
                f"StaircaseStage(name={self.name!r}).max_epochs={self.max_epochs} "
                f"must be >= epochs={self.epochs}"
            )
        if not self.active_loss_terms:
            raise QuantizrStaircaseError(
                f"StaircaseStage(name={self.name!r}).active_loss_terms must be non-empty"
            )
        if not self.convergence_metric or not self.convergence_metric.strip():
            raise QuantizrStaircaseError(
                f"StaircaseStage(name={self.name!r}).convergence_metric must be non-empty"
            )
        if not self.notes or not self.notes.strip():
            raise QuantizrStaircaseError(
                f"StaircaseStage(name={self.name!r}).notes must be non-empty "
                "per CLAUDE.md 'Comment-only contracts are FORBIDDEN'"
            )

    def as_curriculum_stage(self, *, lr_multiplier: float = 1.0) -> CurriculumStage:
        """Project this StaircaseStage to a generic CurriculumStage.

        Loss-domain swap is reified as a single ``loss_key`` token deterministic
        in ``active_loss_terms`` so the generic StageScheduler can drive stage
        scheduling arithmetic. Weight-domain primitives (BN/EMA/LSQ/FP4) are
        NOT in CurriculumStage's contract; they remain on StaircaseStage.
        """
        loss_key = "+".join(sorted(self.active_loss_terms))
        # Optimizer-state policy is canonical "inherit" for stages 2-5 (preserves
        # Adam momentum across stage transitions per Karpathy nanoGPT default);
        # "reset" for anchor stage 1 only (warmup from cold).
        policy = "reset" if self.name == "anchor" else "inherit"
        # qat stage canonically resets LR for the LSQ warmup schedule.
        if self.name == "qat":
            policy = "inherit_lr_reset"
        return CurriculumStage(
            name=self.name,
            epochs=self.epochs,
            loss_key=loss_key,
            lr_multiplier=lr_multiplier,
            optimizer_state_policy=policy,
            notes=self.notes,
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for design-memo + manifest cite."""
        return {
            "name": self.name,
            "epochs": self.epochs,
            "max_epochs": self.max_epochs,
            "active_loss_terms": sorted(self.active_loss_terms),
            "frozen_param_groups": sorted(self.frozen_param_groups),
            "ema_shadow_groups": sorted(self.ema_shadow_groups),
            "convergence_metric": self.convergence_metric,
            "convergence_threshold": self.convergence_threshold,
            "insert_lsq": self.insert_lsq,
            "insert_fp4_fakequant": self.insert_fp4_fakequant,
            "promote_ema_to_inference": self.promote_ema_to_inference,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class StageTransitionRecord:
    """One stage→stage transition record with weight-domain action_keys.

    Sister of :class:`tac.training_curriculum.multi_stage_curriculum.
    StageTransition` (which only carries loss-domain action_keys). The
    Quantizr staircase emits this richer record at every transition so the
    operator can audit BN-freeze / LSQ-insert / FP4-insert / EMA-promote
    primitives fired correctly.
    """

    from_stage_name: str
    to_stage_name: str
    epoch: int
    action_keys: tuple[str, ...]


@dataclass(frozen=True)
class QuantizrFiveStageStaircase:
    """The Quantizr 5-stage staircase as a single dataclass.

    Composes :class:`StaircaseStage` × 5 + a wrapped
    :class:`StageScheduler` so the operator can use the same epoch-driven
    interface as the generic curriculum, while gaining the weight-domain
    primitives for BN/EMA/LSQ/FP4.

    Usage::

        staircase = QuantizrFiveStageStaircase.from_quantizr_canonical()
        for epoch in range(staircase.total_epochs):
            stage = staircase.stage_for_epoch(epoch)
            # apply per-stage loss terms via stage.active_loss_terms
            # ...
            if staircase.is_transition_epoch(epoch):
                transition = staircase.transition_at_epoch(epoch)
                # transition.action_keys carries BN_FROZEN / LSQ_INSERTED /
                # FP4_INSERTED / EMA_PROMOTED tokens
                # ...
    """

    stages: tuple[StaircaseStage, ...]

    def __post_init__(self) -> None:
        if len(self.stages) != 5:
            raise QuantizrStaircaseError(
                f"QuantizrFiveStageStaircase requires exactly 5 stages; got {len(self.stages)}"
            )
        names = tuple(s.name for s in self.stages)
        if names != QUANTIZR_CANONICAL_STAGES:
            raise QuantizrStaircaseError(
                f"QuantizrFiveStageStaircase stage names must be {QUANTIZR_CANONICAL_STAGES} "
                f"in order; got {names}"
            )

    @classmethod
    def from_quantizr_canonical(
        cls,
        *,
        epoch_budget: dict[str, int] | None = None,
    ) -> QuantizrFiveStageStaircase:
        """Return the empirically-validated 5-stage configuration per PR55.

        Args:
            epoch_budget: Optional override of per-stage epoch counts. Keys
                MUST be a subset of :data:`QUANTIZR_CANONICAL_STAGES`;
                missing keys default to :data:`QUANTIZR_DEFAULT_EPOCHS`.
                Operator override is allowed because the canonical 700-epoch
                budget is from PR55's wall-clock budget, not a theoretical
                optimum; substrates with smaller models may converge faster.
        """
        budget = dict(QUANTIZR_DEFAULT_EPOCHS)
        if epoch_budget is not None:
            unknown = set(epoch_budget) - set(QUANTIZR_CANONICAL_STAGES)
            if unknown:
                raise QuantizrStaircaseError(
                    f"epoch_budget keys {sorted(unknown)} not in canonical "
                    f"stage set {QUANTIZR_CANONICAL_STAGES}"
                )
            budget.update(epoch_budget)

        stages = (
            StaircaseStage(
                name="anchor",
                epochs=budget["anchor"],
                active_loss_terms=frozenset({"pixel"}),
                frozen_param_groups=frozenset(),
                ema_shadow_groups=frozenset({"all"}),
                convergence_metric="val_pixel_loss",
                convergence_threshold=0.05,
                notes=(
                    "PR55 stage 1: pixel-loss only warmup; EMA(0.997) on all "
                    "params; BN trainable. Canonical baseline before scorer "
                    "terms enter."
                ),
            ),
            StaircaseStage(
                name="finetune",
                epochs=budget["finetune"],
                active_loss_terms=frozenset({"pixel", "kl_distill_segnet_T2"}),
                frozen_param_groups=frozenset(),
                ema_shadow_groups=frozenset({"all"}),
                convergence_metric="val_segnet_distortion",
                convergence_threshold=0.005,
                notes=(
                    "PR55 stage 2: + SegNet KL distillation T=2.0. Routes "
                    "through tac.losses.kl_distill_segnet_only canonical "
                    "helper. EMA + BN trainable."
                ),
            ),
            StaircaseStage(
                name="joint",
                epochs=budget["joint"],
                active_loss_terms=frozenset({"pixel", "kl_distill_segnet_T2", "posenet"}),
                frozen_param_groups=frozenset(),
                ema_shadow_groups=frozenset({"all"}),
                convergence_metric="val_score_combined",
                convergence_threshold=0.30,
                notes=(
                    "PR55 stage 3: + PoseNet loss (full scorer). EMA + BN "
                    "trainable. Convergence target ~0.30 [contest-CUDA] "
                    "before QAT entry."
                ),
            ),
            StaircaseStage(
                name="qat",
                epochs=budget["qat"],
                active_loss_terms=frozenset({"pixel", "kl_distill_segnet_T2", "posenet"}),
                frozen_param_groups=frozenset({"bn_stats"}),
                ema_shadow_groups=frozenset({"all"}),
                convergence_metric="val_score_combined",
                convergence_threshold=0.33,
                insert_lsq=True,
                insert_fp4_fakequant=True,
                notes=(
                    "PR55 stage 4: BN frozen (Jacob 2018 §4 QAT discipline); "
                    "FP4 fakequant inserted on Conv2d weights via "
                    "tac.fp4_quantize.fake_quant_fp4 (per FP4 hardware "
                    "disclosure discipline). LSQ step size enabled via "
                    "tac.quantization.apply_lsq. EMA preserved across LSQ "
                    "insertion."
                ),
            ),
            StaircaseStage(
                name="final",
                epochs=budget["final"],
                active_loss_terms=frozenset({"pixel", "kl_distill_segnet_T2", "posenet"}),
                frozen_param_groups=frozenset({"bn_stats", "renderer_trunk", "renderer_heads"}),
                ema_shadow_groups=frozenset(),
                convergence_metric="val_score_combined",
                convergence_threshold=0.33,
                promote_ema_to_inference=True,
                notes=(
                    "PR55 stage 5: EMA shadow promoted to inference weights "
                    "(per CLAUDE.md EMA non-negotiable); all but pose-axis "
                    "frozen for final pose-TTO. Stage 5 is short (~50 ep) "
                    "because it is a polish pass, not a full retrain."
                ),
            ),
        )
        return cls(stages=stages)

    @property
    def total_epochs(self) -> int:
        return sum(s.epochs for s in self.stages)

    @property
    def scheduler(self) -> StageScheduler:
        """Wrapped generic StageScheduler for epoch arithmetic.

        Lazily constructed because StageScheduler validates uniqueness of
        names + epochs >= 1 — these invariants are already enforced by
        StaircaseStage.__post_init__ so the wrap is a thin projection.
        """
        return StageScheduler(
            tuple(s.as_curriculum_stage() for s in self.stages)
        )

    def stage_for_epoch(self, epoch: int) -> StaircaseStage:
        """Return the StaircaseStage that owns ``epoch``."""
        try:
            generic = self.scheduler.stage_for_epoch(epoch)
        except CurriculumStageBudgetError as exc:
            raise QuantizrStaircaseError(str(exc)) from exc
        # Lookup by name (unique per StaircaseStage.__post_init__ guarantee).
        for stage in self.stages:
            if stage.name == generic.name:
                return stage
        # Unreachable per the name-uniqueness invariant.
        raise QuantizrStaircaseError(
            f"epoch={epoch} resolved to generic stage {generic.name!r} "
            "with no matching StaircaseStage (invariant violation)"
        )

    def is_transition_epoch(self, epoch: int) -> bool:
        return self.scheduler.is_transition_epoch(epoch)

    def transition_at_epoch(self, epoch: int) -> StageTransitionRecord:
        """Return the StageTransitionRecord at ``epoch`` with weight-domain action_keys."""
        try:
            generic = self.scheduler.transition_at_epoch(epoch)
        except CurriculumStageBudgetError as exc:
            raise QuantizrStaircaseError(str(exc)) from exc
        to_stage = next(s for s in self.stages if s.name == generic.to_stage_name)
        actions: list[str] = list(generic.action_keys)
        # Append weight-domain primitives in canonical evaluation order.
        if "bn_stats" in to_stage.frozen_param_groups:
            actions.append("bn_stats_frozen")
        if to_stage.insert_lsq:
            actions.append("lsq_inserted")
        if to_stage.insert_fp4_fakequant:
            actions.append("fp4_fakequant_inserted")
        # Param-group freezes beyond bn_stats.
        for group in sorted(to_stage.frozen_param_groups):
            if group == "bn_stats":
                continue
            actions.append(f"param_group_frozen:{group}")
        if to_stage.promote_ema_to_inference:
            actions.append("ema_shadow_promoted_to_inference")
        return StageTransitionRecord(
            from_stage_name=generic.from_stage_name,
            to_stage_name=generic.to_stage_name,
            epoch=epoch,
            action_keys=tuple(actions),
        )

    def convergence_criterion_for_stage(self, name: str) -> tuple[str, float]:
        """Return ``(convergence_metric, convergence_threshold)`` for ``name``."""
        for stage in self.stages:
            if stage.name == name:
                return stage.convergence_metric, stage.convergence_threshold
        raise QuantizrStaircaseError(
            f"stage name={name!r} not in canonical set {QUANTIZR_CANONICAL_STAGES}"
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for design-memo + manifest cite."""
        return {
            "staircase_schema_version": "quantizr_five_stage_v1",
            "total_epochs": self.total_epochs,
            "stages": [s.as_dict() for s in self.stages],
            "canonical_anchor": (
                "PR55 0.33 [contest-CUDA] per project_quantizr_full_intel_20260421.md; "
                "5-stage schedule extracted from deobfuscated inflate.py + commit history"
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# Per-stage transition primitives (called by trainer at transition epochs)
# ──────────────────────────────────────────────────────────────────────


def freeze_bn_stats(model: Any) -> int:
    """Freeze all BatchNorm running_mean / running_var on ``model``.

    Idempotent: calling twice is a no-op. Returns the count of BN modules
    frozen (informational; the operator can log this at the transition to
    verify the primitive fired).

    The freeze is achieved by ``module.eval()`` on every BatchNorm[1d|2d|3d]
    submodule. This matches the canonical Jacob 2018 §4 QAT discipline:
    BN running stats must be frozen before fakequant inserts so the stats
    don't drift under quantized-weight forward passes.

    Args:
        model: ``torch.nn.Module`` (typed as ``Any`` to keep the helper
            torch-free at import time per the existing
            ``tac.training_curriculum.multi_stage_curriculum`` pattern).

    Returns:
        Count of BN modules frozen.
    """
    # Lazy import; torch is heavy and not all callers need this primitive.
    import torch.nn as nn

    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def freeze_param_groups(model: Any, group_names: Iterable[str]) -> int:
    """Freeze parameters whose qualified name starts with one of ``group_names``.

    Sets ``param.requires_grad = False`` on every parameter matching the
    prefix filter. Returns the count of parameters frozen.

    The match semantics intentionally use prefix-match (not regex / glob) so
    the operator can pass simple group names like ``"renderer_trunk"`` and
    have every ``renderer_trunk.*`` parameter freeze, without needing to
    spell out individual layer names. This matches the canonical PR101
    naming convention where modules carry stable prefixes.

    Args:
        model: ``torch.nn.Module``.
        group_names: Iterable of parameter-name prefixes to freeze (e.g.
            ``("renderer_trunk", "renderer_heads")``).

    Returns:
        Count of parameters frozen.
    """
    prefixes = tuple(group_names)
    if not prefixes:
        return 0
    count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes) and param.requires_grad:
            param.requires_grad = False
            count += 1
    return count


def apply_ema_shadow_to_inference(model: Any, ema: Any) -> None:
    """Promote EMA shadow weights to ``model``'s inference weights.

    Wraps the canonical :meth:`tac.training.EMA.apply` to make the intent
    explicit at the call site. The operator may call this at the
    ``final``-stage transition (per :attr:`StaircaseStage.promote_ema_to_inference`)
    or at any other inference checkpoint per the EMA non-negotiable in
    CLAUDE.md.

    Args:
        model: ``torch.nn.Module``.
        ema: :class:`tac.training.EMA` instance.
    """
    if not hasattr(ema, "apply"):
        raise QuantizrStaircaseError(
            f"ema object {type(ema).__name__} has no .apply method; "
            "expected tac.training.EMA or compatible interface"
        )
    ema.apply(model)
