# SPDX-License-Identifier: MIT
"""Canonical pausing / curriculum exploits for substrate training.

This package operationalizes the PAUSING-EXPLOITS-WAVE 2026-05-17 directive:
research + design + implement pause-and-{diagnose,swap-loss,quantize,distill,
soup-average} exploits channeling Karpathy + Apple/MLX + Hugging Face PEFT +
Kaggle + ggerganov/llama.cpp + hyperscale Chinchilla + academic literature.

A1's 0.19285 [contest-CPU] frontier IS itself a pause-and-diagnose pattern:
`tools/build_a1_inflate_time_bias_correction_sweep.py` pauses an already-
trained PR95-paradigm substrate AT inflate time and applies bias corrections
to head-output bytes that the trainer never saw. This package generalizes that
empirically-validated exploit to ANY substrate + adds the Karpathy multi-stage
curriculum + Wortsman model soup + Polyak averaging + pause-quantize +
pause-distill + pause-to-swap-loss patterns.

Why "pausing exploits"?
─────────────────────────
The discipline-orthodox training loop is a single-pass continuous optimization
that runs from epoch 0 to epoch N with one loss, one optimizer, one schedule.
Every pause/swap/freeze/checkpoint-interpolation is an exploit of the
observation that the optimal training trajectory IS NOT continuous in this
sense — the empirical winners (PR101 gold via 5-stage Quantizr pipeline; A1's
inflate-time bias correction; Wortsman model soup; Polyak/SWA averaging) all
involve discrete pauses where you (a) inspect state, (b) modify the loss or
the loop or the weights or the archive, (c) resume.

Helper surface (see individual module docstrings):

All ten primitives listed below are importable in this landing. Score claims
remain disabled until a concrete archive candidate produces matched
scorer-response and exact-eval evidence.

* :mod:`tac.training_curriculum.pause_and_diagnose` — canonical pause-and-
  instrument checkpoint with deep state snapshot + diagnostic metrics; allows
  post-hoc inspection AND post-hoc correction without retraining (generalizes
  A1's inflate-time bias correction exploit).
* :mod:`tac.training_curriculum.multi_stage_curriculum` — Karpathy-style
  staged curriculum scheduler (e.g. L1 pixel loss → L2 + scorer loss → KL-T=2.0
  distillation per Hinton); stage transitions are explicit, declarative, and
  operator-routable.
* :mod:`tac.training_curriculum.quantizr_5_stage_staircase` — Quantizr-specific
  5-stage staircase helper with explicit BN/EMA/LSQ/FP4 transition records.
  This is a training-practice scaffold, not current-frontier authority until a
  byte-closed trainer adoption produces exact scorer evidence.
* :mod:`tac.training_curriculum.model_soup_averaging` — Wortsman et al 2022
  "Model soups: averaging weights of multiple fine-tuned models" greedy and
  uniform soup variants on top of EMA snapshots; preserves SegNet/PoseNet
  forward compatibility.
* :mod:`tac.training_curriculum.swa_polyak_averaging` — Polyak (1990) running
  averaging + Izmailov-Podoprikhin-Polyak-Vyas-Wilson SWA (2018) variants
  scoped to the late training phase per the Kaggle reference pattern;
  documented as a generalization of the existing :class:`tac.training.SWA`
  helper to be EMA-compatible from byte zero.
* :mod:`tac.training_curriculum.pause_to_swap_loss` — pause training, swap
  loss function, resume; enables curriculum without restart and avoids the
  default "monolithic loss with weighted terms" anti-pattern that suppresses
  per-stage signal per the operator's 2026-05-15 standing directive on
  UNIQUE-AND-COMPLETE-PER-METHOD operating mode.
* :mod:`tac.training_curriculum.a1_pattern_inflate_time_bias_correction` —
  generalization of A1's inflate-time bias correction pattern to ANY substrate:
  pause-extract-correct-rebuild archive without re-running training. Documents
  the empirically-validated exploit per `feedback_a1_inflate_bias_sweep_exact_
  cpu_review_20260509_codex.md`.
* :mod:`tac.training_curriculum.pause_quantize_finetune` — pause-then-AWQ/
  GPTQ-style post-train quantization (ggerganov llama.cpp pattern) + resume
  fine-tune; sister of :mod:`tac.quantization` but at the pause-and-finetune
  surface.
* :mod:`tac.training_curriculum.pause_distill_resume` — pause-extract-teacher
  (frozen anchor checkpoint) + train student with KL-on-logits T=2.0 (Hinton +
  Quantizr canonical pattern) + resume teacher with student knowledge;
  REQUIRED primitive for T4 SYMPOSIUM Priority 1 BOLT-ON-on-A1 wave per
  Hinton's verbatim recommendation.
* :mod:`tac.training_curriculum.early_stopping_with_resume` — Hugging Face
  Trainer-style early stopping with patience + resume from best checkpoint;
  generalization of A1's "best EMA checkpoint" pattern already present in
  :class:`tac.training.Trainer`.

Cross-decision input to T4 Priority 1 BOLT-ON-on-A1 wave
────────────────────────────────────────────────────────
Every BOLT-ON-on-A1 lane (Ballé hyperprior / PR101 entropy stack / VQ-codebook)
consumes ≥3 of these helpers:

1. `pause_and_diagnose.checkpoint_with_metrics` — pause-A1 at frontier-score
   to extract teacher weights for distillation.
2. `pause_distill_resume.kl_on_logits_distillation` — train BOLT-ON student
   with KL-on-logits T=2.0 from A1 frozen teacher (Hinton verbatim).
3. `multi_stage_curriculum.StageScheduler` — Stage 1 = warmup BOLT-ON head
   with frozen A1 backbone; Stage 2 = unfreeze + joint fine-tune.
4. `a1_pattern_inflate_time_bias_correction.GeneralizedInflateBiasCorrector` —
   if BOLT-ON's empirical-CPU score regresses by < 0.005, apply inflate-time
   bias correction sweep over the BOLT-ON sidecar before promotion claim.
5. `swa_polyak_averaging.SWAScheduler` — late-phase SWA averaging applied to
   the BOLT-ON head's EMA shadow for wider quantization-friendly minima.

Apples-to-apples evidence discipline
────────────────────────────────────
Per CLAUDE.md, every speedup / score-improvement claim in this package's
module docstrings is tagged `[derived]` (first-principles) /
`[literature-extrapolation]` (paper citation) / `[would-need-empirical]`
(no internal anchor yet). NO score claims at the function-return surface;
that's the job of `experiments/train_substrate_*.py` + `auth_eval_renderer.py`.

Cargo-cult audit per Catalog #303
─────────────────────────────────
Every helper is interrogated for HARD-EARNED vs CARGO-CULTED inheritance
from the source literature. See `.omx/research/pausing_exploits_design_and_
implementation_landed_20260517.md` § "Cargo-cult audit per assumption" for
the per-helper classification.

Voice attribution
─────────────────
* Karpathy nanoGPT — pause-to-tune-LR + pause-to-instrument at smallest viable
  scale + "let compute speak"; informs `multi_stage_curriculum` + `pause_and_
  diagnose`.
* Apple/MLX — pause-and-convert-to-ANE checkpoint export pattern; informs
  `pause_and_diagnose.export_for_mlx_inference` (deferred — Phase 2; not
  contest-critical).
* Hugging Face PEFT — pause-and-switch-adapter; pretrain → SFT → RLHF stages;
  informs `pause_to_swap_loss` + `multi_stage_curriculum.StageTransition`.
* Kaggle competition — pseudo-labeling pause; cross-validation pause; per-fold
  checkpoint extraction; informs `model_soup_averaging` (Kaggle ensembling).
* ggerganov llama.cpp — pause-and-AWQ/GPTQ pattern; informs `pause_quantize_
  finetune`.
* Hyperscale Chinchilla (Hoffmann et al 2022) — compute-optimal training
  schedule; informs `multi_stage_curriculum`'s default stage-budget allocation.
* Academic (Wortsman 2022 / Polyak 1990 / Izmailov 2018 / IMP Frankle-Carbin
  2018 / Hinton 2014) — informs each respective helper's mathematical
  foundation.
* A1 anchor (2026-05-09 GHA CPU 0.19285) — informs `a1_pattern_inflate_time_
  bias_correction`; the ONLY empirically-validated pausing exploit in this
  package as of landing.
"""

from __future__ import annotations

from tac.training_curriculum.a1_pattern_inflate_time_bias_correction import (
    A1PatternBiasCorrectionPlan,
    GeneralizedInflateBiasCorrector,
    InflateBiasCorrectionError,
    InflateBiasCorrectionVerdict,
)
from tac.training_curriculum.early_stopping_with_resume import (
    EarlyStoppingState,
    EarlyStoppingTracker,
    ResumeFromBestCheckpoint,
)
from tac.training_curriculum.model_soup_averaging import (
    GreedyModelSoup,
    ModelSoupError,
    ModelSoupResult,
    UniformModelSoup,
)
from tac.training_curriculum.multi_stage_curriculum import (
    CurriculumStage,
    CurriculumStageBudgetError,
    StageScheduler,
    StageTransition,
)
from tac.training_curriculum.master_gradient_pair_weights import (
    MasterGradientPairWeights,
    derive_master_gradient_pair_weights,
    load_master_gradient_pair_weights_for_archive,
)
from tac.training_curriculum.pause_and_diagnose import (
    DiagnosticCheckpoint,
    DiagnosticMetric,
    PauseAndDiagnoseError,
    pause_and_capture,
)
from tac.training_curriculum.pause_distill_resume import (
    DistillationConfig,
    DistillationError,
    kl_on_logits_distillation,
    teacher_student_pair,
)
from tac.training_curriculum.pause_quantize_finetune import (
    PauseQuantizeFinetuneError,
    PauseQuantizePlan,
    apply_pause_quantize_finetune_plan,
)
from tac.training_curriculum.pause_to_swap_loss import (
    LossSwap,
    LossSwapError,
    swap_loss_at_pause,
)
from tac.training_curriculum.per_pair_master_gradient_wire_in import (
    TrainingCurriculumPerPairWireInOutcome,
    compose_training_curriculum_per_pair_wire_in,
)
from tac.training_curriculum.quantizr_5_stage_staircase import (
    QUANTIZR_CANONICAL_STAGES,
    QUANTIZR_DEFAULT_EPOCHS,
    QuantizrFiveStageStaircase,
    QuantizrStaircaseError,
    StaircaseStage,
    apply_ema_shadow_to_inference,
    freeze_bn_stats,
    freeze_param_groups,
)
from tac.training_curriculum.quantizr_5_stage_staircase import (
    StageTransitionRecord as QuantizrStageTransitionRecord,
)
from tac.training_curriculum.swa_polyak_averaging import (
    PolyakAverager,
    SWAScheduler,
    SWASchedulerError,
)

IMPLEMENTED_MODULES: tuple[str, ...] = (
    "a1_pattern_inflate_time_bias_correction",
    "early_stopping_with_resume",
    "model_soup_averaging",
    "multi_stage_curriculum",
    "master_gradient_pair_weights",
    "pause_and_diagnose",
    "pause_distill_resume",
    "pause_quantize_finetune",
    "pause_to_swap_loss",
    "quantizr_5_stage_staircase",
    "swa_polyak_averaging",
)

DEFERRED_MODULES: tuple[str, ...] = ()

DEFERRED_RATIONALE = (
    "No module-level deferrals remain in the 2026-05-17 pausing/curriculum "
    "package. Score claims remain disabled until concrete archive candidates "
    "produce matched scorer-response and exact-eval evidence."
)

__all__ = [
    "DEFERRED_MODULES",
    "DEFERRED_RATIONALE",
    "IMPLEMENTED_MODULES",
    "QUANTIZR_CANONICAL_STAGES",
    "QUANTIZR_DEFAULT_EPOCHS",
    "A1PatternBiasCorrectionPlan",
    "CurriculumStage",
    "CurriculumStageBudgetError",
    "DiagnosticCheckpoint",
    "DiagnosticMetric",
    "DistillationConfig",
    "DistillationError",
    "EarlyStoppingState",
    "EarlyStoppingTracker",
    "GeneralizedInflateBiasCorrector",
    "GreedyModelSoup",
    "InflateBiasCorrectionError",
    "InflateBiasCorrectionVerdict",
    "LossSwap",
    "LossSwapError",
    "MasterGradientPairWeights",
    "ModelSoupError",
    "ModelSoupResult",
    "PauseAndDiagnoseError",
    "PauseQuantizeFinetuneError",
    "PauseQuantizePlan",
    "PolyakAverager",
    "QuantizrFiveStageStaircase",
    "QuantizrStageTransitionRecord",
    "QuantizrStaircaseError",
    "ResumeFromBestCheckpoint",
    "SWAScheduler",
    "SWASchedulerError",
    "StageScheduler",
    "StageTransition",
    "StaircaseStage",
    "TrainingCurriculumPerPairWireInOutcome",
    "UniformModelSoup",
    "apply_ema_shadow_to_inference",
    "apply_pause_quantize_finetune_plan",
    "compose_training_curriculum_per_pair_wire_in",
    "derive_master_gradient_pair_weights",
    "freeze_bn_stats",
    "freeze_param_groups",
    "kl_on_logits_distillation",
    "load_master_gradient_pair_weights_for_archive",
    "pause_and_capture",
    "swap_loss_at_pause",
    "teacher_student_pair",
]
