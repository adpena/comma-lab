# SPDX-License-Identifier: MIT
"""The 1:1 MLX port of PR95's 8-stage training curriculum (L14) — the FLEET fix.

The capstone (and the whole substrate fleet) had been training with a FIXED
single-stage recipe (just ``ce_seg_loss`` + a guessed fixed Muon LR), and floored
at ``d_seg ~ 0.008-0.012``. PR95 reached ``d_seg 5.6e-4`` ONLY via the 8-stage
curriculum below: a sequence of seg-loss families, an INT8 QAT stage, a coder-aware
C1a entropy regularizer (L16), a Gaussian sigma-noise schedule (L17), and a final
Muon-finetune stage (L15). This module is the EXACT per-stage spec extracted from
the proven PR95 torch reference (``submissions/hnerv_muon/src/stages/stage{1..8}*.py``)
plus a reusable runner that drives ANY score-aware trainer through the curriculum.

Source of truth (torch), per stage (epochs, seg_loss, sigma, c1a_lambda, qat, opt/lr):

  ===  ====================  ========  ====================  =====  ==========  ====  ============
  #    name                  epochs    seg_loss              sigma  c1a_lambda  qat   opt / lr
  ===  ====================  ========  ====================  =====  ==========  ====  ============
  1    stage1_v328_ce        3000      ce_seg_loss           0.2*   0.0         no    AdamW 1e-3
  2    stage2_v331_softplus  5650      tau_softplus(0.3)     0.2*   0.0         no    AdamW 1e-3
  3    stage3_v332_smooth    1500      smooth_disagree(0.3)  0.2*   0.0         no    AdamW 1e-4
  4    stage4_v332_qat       500       smooth_disagree(0.3)  0.2*   0.0         YES   AdamW 1e-4
  5    stage5_c1a_l7         9000      l7_softplus(0.3)      0.2    0.01        YES   AdamW 3e-5
  6    stage6_lambda_sweep   2000      l7_softplus(0.3)      0.2    0.02        YES   AdamW 3e-5
  7    stage7_sigma_sweep    3000      l7_softplus(0.3)      0.1    0.02        YES   AdamW 3e-5
  8    stage8_muon_finetune  5000      l7_softplus(0.3)      0.1    0.02        YES   Muon 2e-4 +
                                                                                      AdamW 1e-5
  ===  ====================  ========  ====================  =====  ==========  ====  ============

  ``sigma*`` (stages 1-4): in torch, ``cat_sigma`` is carried in the config but is
  UNUSED while ``cat_lambda == 0`` (the C1a term is not added). So the sigma-noise
  *injection* (L17) is only active where ``cat_lambda > 0`` AND the runner is told to
  apply the sigma WEIGHT-noise schedule. The L17 weight-noise schedule and the C1a
  ``cat_sigma`` are two distinct uses of the same ``sigma`` symbol in PR95 — see the
  ``sigma_weight_noise`` field below, which is the per-stage L17 weight-noise scale
  (0.2 in stages 5-6, 0.1 in stages 7-8 by default, mirroring the C1a sigma schedule
  and the L17 lesson text). Stages 1-4 default to ``sigma_weight_noise=0.0`` because
  the torch reference does NOT inject weight noise there (it only carries the unused
  ``cat_sigma``).

Optimizer schedule (per ``.omx/research/tilde_optimizers_*.md`` #77 vs PR95 canonical):
  - ``pr95_adamw_then_muon`` — FAITHFUL to PR95: AdamW for stages 1-7, Muon (hidden
    convs) + AdamW (stem/rgb/latents) for stage 8 ONLY. This is what PR95 ACTUALLY
    used (``stage8_muon_finetune.py`` is the only stage with ``use_muon=True``).
  - ``muon_throughout`` — our #77 DEVIATION: Muon from stage 1. The inert-loop audit
    found AdamW + 100%-clip stalled the early stages on our smaller MLX basis; #77
    runs Muon throughout. Both are SELECTABLE so the optimizer-correctness audit can
    A/B them — neither is silently hardcoded.

The runner is backend-agnostic over a tiny ``CurriculumTrainerProtocol`` (the two
methods the curriculum needs: ``configure_stage`` + ``run_stage_epochs``), so it
drives BOTH :class:`tac.mlx_pr95_port.mlx_trainer.MlxScoreAwareTrainer` (the PR95
reference loop) AND :class:`tac.capstone_vq_nerv.capstone_trainer.CapstoneTrainer`
(the FiLM-pose capstone) without duplicating the inner loop.

Authority: ``[macOS-MLX research-signal]`` — non-promotable per Catalog #192; a
contest score requires ``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

# Optimizer-schedule selector values (exposed for CLI + audit A/B).
OPTIMIZER_SCHEDULE_PR95 = "pr95_adamw_then_muon"
OPTIMIZER_SCHEDULE_MUON_THROUGHOUT = "muon_throughout"
OPTIMIZER_SCHEDULES = (OPTIMIZER_SCHEDULE_PR95, OPTIMIZER_SCHEDULE_MUON_THROUGHOUT)


@dataclass(frozen=True)
class StageSpec:
    """One PR95 curriculum stage's exact spec (the source-of-truth tuple).

    Every field is the value extracted verbatim from the torch
    ``stages/stage{n}*.py`` ``make_config`` (cross-referenced in the module
    docstring table). ``use_muon`` here is the PR95-CANONICAL value (only stage 8
    is True); the runner OVERRIDES it when the ``muon_throughout`` schedule is
    selected (see :func:`resolve_use_muon`).
    """

    name: str
    epochs: int
    seg_loss_form: str
    # tau is fixed at 0.3 for stages 2-8 (stage 1 CE ignores tau).
    cat_lambda: float  # C1a entropy weight (L16); 0.0 disables the C1a term.
    cat_sigma: float  # C1a soft-histogram bandwidth (L16) — used only if cat_lambda>0.
    sigma_weight_noise: float  # L17 additive weight-noise scale; 0.0 disables.
    use_qat: bool  # L14 stage-4+ INT8 fake-quant in the forward.
    use_muon_canonical: bool  # PR95-canonical optimizer (only stage 8 True).
    adamw_lr: float
    muon_lr: float
    # PR95 constants, identical across all 8 stages (kept explicit for fidelity):
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    latent_lr_mult: float = 10.0
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    grad_clip_muon: float = 1.0
    muon_weight_decay: float = 0.0


# ---------------------------------------------------------------------------
# The canonical PR95 8-stage curriculum (exact torch spec).
# ---------------------------------------------------------------------------
# epochs are the FULL canonical PR95 counts (stage5-8 use PR95's "our extension"
# counts where the torch make_config default differs from the docstring's
# "default canonical" — we take the make_config default, which is what train.py
# actually runs). A scale factor is applied by build_pr95_8stage_curriculum so a
# short local smoke/run can proportionally compress the schedule without losing
# the stage STRUCTURE (the ordering + loss/sigma/lambda/qat/opt transitions).
_PR95_8STAGE: tuple[StageSpec, ...] = (
    StageSpec(
        name="stage1_v328_ce", epochs=3000, seg_loss_form="ce_seg_loss",
        cat_lambda=0.0, cat_sigma=0.2, sigma_weight_noise=0.0,
        use_qat=False, use_muon_canonical=False, adamw_lr=1e-3, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage2_v331_softplus", epochs=5650, seg_loss_form="tau_softplus_seg_loss",
        cat_lambda=0.0, cat_sigma=0.2, sigma_weight_noise=0.0,
        use_qat=False, use_muon_canonical=False, adamw_lr=1e-3, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage3_v332_smooth", epochs=1500, seg_loss_form="smooth_disagreement_seg_loss",
        cat_lambda=0.0, cat_sigma=0.2, sigma_weight_noise=0.0,
        use_qat=False, use_muon_canonical=False, adamw_lr=1e-4, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage4_v332_qat", epochs=500, seg_loss_form="smooth_disagreement_seg_loss",
        cat_lambda=0.0, cat_sigma=0.2, sigma_weight_noise=0.0,
        use_qat=True, use_muon_canonical=False, adamw_lr=1e-4, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage5_c1a_l7", epochs=9000, seg_loss_form="l7_softplus_seg_loss",
        cat_lambda=0.01, cat_sigma=0.2, sigma_weight_noise=0.2,
        use_qat=True, use_muon_canonical=False, adamw_lr=3e-5, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage6_lambda_sweep", epochs=2000, seg_loss_form="l7_softplus_seg_loss",
        cat_lambda=0.02, cat_sigma=0.2, sigma_weight_noise=0.2,
        use_qat=True, use_muon_canonical=False, adamw_lr=3e-5, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage7_sigma_sweep", epochs=3000, seg_loss_form="l7_softplus_seg_loss",
        cat_lambda=0.02, cat_sigma=0.1, sigma_weight_noise=0.1,
        use_qat=True, use_muon_canonical=False, adamw_lr=3e-5, muon_lr=2e-4,
    ),
    StageSpec(
        name="stage8_muon_finetune", epochs=5000, seg_loss_form="l7_softplus_seg_loss",
        cat_lambda=0.02, cat_sigma=0.1, sigma_weight_noise=0.1,
        use_qat=True, use_muon_canonical=True, adamw_lr=1e-5, muon_lr=2e-4,
        muon_weight_decay=5e-4,  # researcher #24 idea 1 (Chen-Li-Liu 2506.15054).
    ),
)

CURRICULA: dict[str, tuple[StageSpec, ...]] = {
    "pr95_8stage": _PR95_8STAGE,
}


def resolve_use_muon(spec: StageSpec, optimizer_schedule: str) -> bool:
    """Resolve whether THIS stage uses Muon, given the optimizer schedule.

    - ``pr95_adamw_then_muon`` (faithful): use the PR95-canonical value
      (``use_muon_canonical`` — only stage 8 True).
    - ``muon_throughout`` (#77 deviation): always True.
    """
    if optimizer_schedule == OPTIMIZER_SCHEDULE_PR95:
        return spec.use_muon_canonical
    if optimizer_schedule == OPTIMIZER_SCHEDULE_MUON_THROUGHOUT:
        return True
    raise ValueError(
        f"unknown optimizer_schedule {optimizer_schedule!r}; "
        f"known: {OPTIMIZER_SCHEDULES}"
    )


def build_pr95_8stage_curriculum(
    *,
    epoch_scale: float = 1.0,
    total_epochs: int | None = None,
    min_epochs_per_stage: int = 1,
) -> tuple[StageSpec, ...]:
    """Return the PR95 8-stage curriculum, optionally compressed for a local run.

    The canonical PR95 schedule is ~29,650 epochs (~50 GPU-hr). For a local MLX
    run we proportionally compress every stage's epoch count while preserving the
    stage STRUCTURE (the ordering and the loss/sigma/lambda/qat/opt transitions —
    which is what breaks the d_seg floor; the absolute epoch counts only set how
    long each phase runs).

    Args:
        epoch_scale: multiply every canonical epoch count by this factor.
        total_epochs: if set, OVERRIDES ``epoch_scale`` to spread exactly this many
            epochs across the 8 stages proportionally to the canonical counts.
        min_epochs_per_stage: floor so no stage degenerates to 0 epochs.

    Returns:
        The 8-stage tuple with adjusted ``epochs`` (all other fields unchanged).
    """
    base = CURRICULA["pr95_8stage"]
    canonical_total = sum(s.epochs for s in base)
    if total_epochs is not None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        scale = total_epochs / canonical_total
    else:
        if epoch_scale <= 0:
            raise ValueError("epoch_scale must be positive")
        scale = epoch_scale
    return tuple(
        replace(s, epochs=max(min_epochs_per_stage, round(s.epochs * scale)))
        for s in base
    )


@runtime_checkable
class CurriculumTrainerProtocol(Protocol):
    """The minimal trainer surface the curriculum runner drives.

    Any score-aware trainer that implements these two methods can be driven
    through the PR95 curriculum. Both the PR95-reference ``MlxScoreAwareTrainer``
    and the FiLM-pose ``CapstoneTrainer`` implement them.
    """

    def configure_stage(self, spec: StageSpec, *, optimizer_schedule: str) -> None:
        """Switch the trainer to this stage: set the bridge seg-loss, rebuild the
        optimizer config (LR + Muon-vs-AdamW per the resolved schedule), and arm the
        per-stage QAT / C1a / sigma-noise mechanisms. Optimizer STATE (Muon/AdamW
        momentum/EMA buffers) is preserved across stages (PR95 inter-stage transitions
        resume weights + carry the cosine schedule)."""
        ...

    def run_stage_epochs(self, spec: StageSpec) -> dict[str, Any]:
        """Run ``spec.epochs`` epochs of the (already-configured) stage and return a
        per-stage summary (at least ``{"stage", "epochs", "d_seg_best", "d_seg_final"}``)."""
        ...


@dataclass
class CurriculumResult:
    """Outcome of a full curriculum run."""

    optimizer_schedule: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    d_seg_initial: float | None = None
    d_seg_final: float | None = None
    d_seg_best: float | None = None


def run_curriculum(
    trainer: CurriculumTrainerProtocol,
    stages: tuple[StageSpec, ...],
    *,
    optimizer_schedule: str = OPTIMIZER_SCHEDULE_MUON_THROUGHOUT,
    on_stage_done: Any = None,
) -> CurriculumResult:
    """Drive ``trainer`` through the ordered ``stages`` (the FLEET-reusable runner).

    For each stage: ``configure_stage`` (switch loss/opt/QAT/C1a/sigma) then
    ``run_stage_epochs`` (run the stage's epochs). Weights + optimizer state carry
    across stages (the trainer holds them). Returns the per-stage summaries plus the
    overall d_seg descent.

    Args:
        trainer: a :class:`CurriculumTrainerProtocol` implementer.
        stages: ordered stage specs (e.g. :func:`build_pr95_8stage_curriculum`).
        optimizer_schedule: ``pr95_adamw_then_muon`` or ``muon_throughout``.
        on_stage_done: optional callback ``(stage_index, spec, summary) -> None`` for
            live logging / checkpointing between stages.
    """
    if optimizer_schedule not in OPTIMIZER_SCHEDULES:
        raise ValueError(
            f"unknown optimizer_schedule {optimizer_schedule!r}; known: {OPTIMIZER_SCHEDULES}"
        )
    result = CurriculumResult(optimizer_schedule=optimizer_schedule)
    d_seg_best = float("inf")
    for i, spec in enumerate(stages):
        trainer.configure_stage(spec, optimizer_schedule=optimizer_schedule)
        summary = trainer.run_stage_epochs(spec)
        summary = dict(summary)
        summary["stage_index"] = i
        summary["stage"] = spec.name
        summary["use_muon"] = resolve_use_muon(spec, optimizer_schedule)
        result.stages.append(summary)
        if result.d_seg_initial is None and summary.get("d_seg_initial") is not None:
            result.d_seg_initial = summary["d_seg_initial"]
        sb = summary.get("d_seg_best")
        if sb is not None:
            d_seg_best = min(d_seg_best, sb)
        if on_stage_done is not None:
            on_stage_done(i, spec, summary)
    if result.stages:
        result.d_seg_final = result.stages[-1].get("d_seg_final")
    if d_seg_best != float("inf"):
        result.d_seg_best = d_seg_best
    return result


__all__ = [
    "CURRICULA",
    "OPTIMIZER_SCHEDULES",
    "OPTIMIZER_SCHEDULE_MUON_THROUGHOUT",
    "OPTIMIZER_SCHEDULE_PR95",
    "CurriculumResult",
    "CurriculumTrainerProtocol",
    "StageSpec",
    "build_pr95_8stage_curriculum",
    "resolve_use_muon",
    "run_curriculum",
]
