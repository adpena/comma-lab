# SPDX-License-Identifier: MIT
"""Pause-quantize-finetune — ggerganov llama.cpp + AWQ/GPTQ pattern.

The canonical "train to convergence → pause → apply post-train quantization
→ resume short fine-tune to recover lost accuracy" pattern. ggerganov's
llama.cpp + GGUF format established this as the dominant pattern for shipping
quantized inference of LLMs; the same pattern applies to contest substrates
where the archive is the deployment artifact.

Two canonical forms:

1. **Static post-train quantization (PTQ)**: pause, apply per-channel /
   per-tensor INT8 quantization, do NOT fine-tune. The simplest form;
   degrades accuracy but is free.
2. **Quantization-aware fine-tuning (QAT-FT)**: pause, apply quantization,
   ENABLE STE (straight-through estimator) on quantization rounding,
   fine-tune for a few epochs. Recovers most of the accuracy degradation;
   ~5-20% of original training cost.

This module is the PLANNING + WIRING surface; the actual quantization
arithmetic lives in :mod:`tac.quantization` (which has FakeQuantSTE,
Uint8STE, FakeQuantFP4 already; the LSQ step-size learning lives in
:mod:`tac.training`). This module exposes a single :class:`PauseQuantizePlan`
dataclass + :func:`apply_pause_quantize_finetune_plan` orchestrator.

`[literature-extrapolation]` claims:
- AWQ (Activation-Aware Weight Quantization, Lin et al 2023): scaled weight-
  quantization that preserves activation-magnitude-aware channels at full
  precision; reported ~1-2% accuracy loss vs FP16 at INT4.
- GPTQ (Frantar et al 2022): one-shot post-train quantization that approximates
  the Hessian inverse via Gauss-Newton; reported similar accuracy loss to AWQ.
- ggerganov GGUF: canonical INT4/INT5/INT8 quantization format for LLM
  inference; established the pause-quantize-deploy pattern.

`[derived]` claims:
- Pause cost: O(state_dict_bytes) deep-clone for rollback if fine-tune
  diverges; ~200KB for 100k-param substrates at fp16.
- Fine-tune cost: typically 5-20% of original training epoch budget per
  empirical PTQ vs QAT-FT literature.

Cargo-cult audit per assumption
───────────────────────────────
* "Post-train quantization always degrades accuracy" — HARD-EARNED for
  arbitrary networks; CARGO-CULTED for networks specifically trained with
  QAT-aware techniques (where the final-epoch weights are ALREADY
  quantization-friendly). Empirical: Quantizr 0.33 [contest-CUDA] achieved
  this via INTEGRATED QAT in stages 4-5 of its pipeline; the post-train
  PTQ step was effectively a no-op because the QAT had already converged
  to a quantization-friendly basin.
* "INT4 is universally fine for substrates < 100k params" — CARGO-CULTED;
  empirical from `feedback_three_lossy_anchors_show_rel_err_squared_
  objective_falsified_20260508.md`: at rms ≥ 0.04, the rel_err² objective
  is falsified (the network's score-axis sensitivity is non-linear in
  weight perturbation). For small substrates, INT4 may exceed this
  threshold; INT6 or INT8 may be required.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Quantization arithmetic → ADOPT canonical (:mod:`tac.quantization`).
* Fine-tune loop → DOCUMENTED FORK (caller supplies their substrate's
  training loop; we expose a plan dataclass + orchestrator that wraps
  the caller's loop).
* Rollback policy → ADOPT canonical (snapshot + restore pattern from
  :class:`tac.training.Trainer`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

QuantizationKind = Literal[
    "int4_per_channel",
    "int8_per_channel",
    "int8_per_tensor",
    "fp4_per_channel",
]

QuantizationMode = Literal["ptq", "qat_ft"]


class PauseQuantizeFinetuneError(RuntimeError):
    """Raised when pause-quantize-finetune invariants are violated."""


@dataclass(frozen=True)
class PauseQuantizePlan:
    """Plan for pause-quantize-finetune.

    Args:
        quantization_kind: One of ``"int4_per_channel"`` /
            ``"int8_per_channel"`` / ``"int8_per_tensor"`` /
            ``"fp4_per_channel"``.
        mode: ``"ptq"`` (no fine-tune) or ``"qat_ft"`` (with fine-tune).
        finetune_epochs: Epoch budget for fine-tune (only meaningful if
            ``mode == "qat_ft"``; must be > 0 in that case).
        finetune_lr_multiplier: LR multiplier vs. original training LR
            (Quantizr canonical 0.1; default 0.1).
        accuracy_recovery_threshold: If validation metric AFTER fine-tune
            is WORSE than the pre-quantization metric by more than
            ``accuracy_recovery_threshold``, rollback to the pre-quantization
            checkpoint. Default ``0.005`` matches A1's empirical noise floor.
        minimize_metric: True if lower validation metric is better (contest
            scorer semantics); default True.
        rationale: Operator-readable rationale (rejected if empty).
    """

    quantization_kind: QuantizationKind
    mode: QuantizationMode
    finetune_epochs: int = 0
    finetune_lr_multiplier: float = 0.1
    accuracy_recovery_threshold: float = 0.005
    minimize_metric: bool = True
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.quantization_kind not in {
            "int4_per_channel",
            "int8_per_channel",
            "int8_per_tensor",
            "fp4_per_channel",
        }:
            raise PauseQuantizeFinetuneError(
                f"quantization_kind={self.quantization_kind!r} not in canonical set"
            )
        if self.mode not in {"ptq", "qat_ft"}:
            raise PauseQuantizeFinetuneError(
                f"mode={self.mode!r} not in canonical set"
            )
        if self.mode == "qat_ft" and self.finetune_epochs < 1:
            raise PauseQuantizeFinetuneError(
                "mode='qat_ft' requires finetune_epochs >= 1"
            )
        if self.mode == "ptq" and self.finetune_epochs != 0:
            raise PauseQuantizeFinetuneError(
                "mode='ptq' requires finetune_epochs == 0"
            )
        if self.finetune_lr_multiplier <= 0:
            raise PauseQuantizeFinetuneError(
                f"finetune_lr_multiplier={self.finetune_lr_multiplier} must be > 0"
            )
        if self.accuracy_recovery_threshold <= 0:
            raise PauseQuantizeFinetuneError(
                f"accuracy_recovery_threshold={self.accuracy_recovery_threshold} "
                "must be > 0"
            )
        if not self.rationale or not self.rationale.strip():
            raise PauseQuantizeFinetuneError(
                "rationale must be non-empty per CLAUDE.md 'Comment-only "
                "contracts are FORBIDDEN'"
            )


def apply_pause_quantize_finetune_plan(
    *,
    model: nn.Module,
    plan: PauseQuantizePlan,
    pre_quantize_validate_fn: Callable[[nn.Module], float],
    post_quantize_validate_fn: Callable[[nn.Module], float],
    quantize_fn: Callable[[nn.Module, QuantizationKind], None],
    finetune_fn: Callable[[nn.Module, int, float], float] | None = None,
    base_lr: float = 1e-3,
) -> dict[str, Any]:
    """Orchestrate pause → quantize → optionally fine-tune → validate or rollback.

    Args:
        model: Live :class:`nn.Module` whose state will be quantized in-place.
        plan: :class:`PauseQuantizePlan`.
        pre_quantize_validate_fn: Callable that returns a scalar metric
            BEFORE quantization. Used to compute the rollback threshold.
        post_quantize_validate_fn: Callable that returns a scalar metric
            AFTER quantization (and after fine-tune if applicable).
        quantize_fn: Callable that mutates ``model`` in-place to apply
            ``plan.quantization_kind``. Substrate-specific; the canonical
            arithmetic primitives live in :mod:`tac.quantization`.
        finetune_fn: REQUIRED if ``plan.mode == "qat_ft"``. Callable that
            takes ``(model, epochs, lr)`` and returns the post-fine-tune
            validation metric. Substrate-specific.
        base_lr: Base LR; fine-tune uses ``base_lr * plan.finetune_lr_
            multiplier``.

    Returns:
        Dict with keys:
            * ``pre_quantize_metric``
            * ``post_quantize_metric``
            * ``post_finetune_metric`` (or ``None`` if PTQ-only)
            * ``rollback_invoked`` (bool)
            * ``final_metric``
            * ``final_state_kind`` ("pre_quantize_rollback" or
              "post_quantize" or "post_finetune")

    Raises:
        :class:`PauseQuantizeFinetuneError` on validation failure.
    """
    if plan.mode == "qat_ft" and finetune_fn is None:
        raise PauseQuantizeFinetuneError(
            "plan.mode='qat_ft' requires finetune_fn"
        )

    # 1. Snapshot pre-quantization state for rollback.
    pre_quantize_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    pre_quantize_metric = float(pre_quantize_validate_fn(model))

    # 2. Quantize (in-place per quantize_fn contract).
    quantize_fn(model, plan.quantization_kind)
    post_quantize_metric = float(post_quantize_validate_fn(model))

    # 3. Optionally fine-tune.
    post_finetune_metric: float | None = None
    if plan.mode == "qat_ft":
        assert finetune_fn is not None  # narrowed above
        finetune_lr = base_lr * plan.finetune_lr_multiplier
        post_finetune_metric = float(
            finetune_fn(model, plan.finetune_epochs, finetune_lr)
        )

    final_metric_post_pipeline = (
        post_finetune_metric
        if post_finetune_metric is not None
        else post_quantize_metric
    )

    # 4. Compare to threshold and rollback if regression too large.
    if plan.minimize_metric:
        regression = final_metric_post_pipeline - pre_quantize_metric
    else:
        regression = pre_quantize_metric - final_metric_post_pipeline

    rollback = regression > plan.accuracy_recovery_threshold
    final_state_kind: str
    final_metric: float
    if rollback:
        model.load_state_dict(pre_quantize_state)
        final_state_kind = "pre_quantize_rollback"
        final_metric = pre_quantize_metric
    elif post_finetune_metric is not None:
        final_state_kind = "post_finetune"
        final_metric = post_finetune_metric
    else:
        final_state_kind = "post_quantize"
        final_metric = post_quantize_metric

    return {
        "pre_quantize_metric": pre_quantize_metric,
        "post_quantize_metric": post_quantize_metric,
        "post_finetune_metric": post_finetune_metric,
        "rollback_invoked": rollback,
        "final_metric": final_metric,
        "final_state_kind": final_state_kind,
        "quantization_kind": plan.quantization_kind,
        "mode": plan.mode,
    }
