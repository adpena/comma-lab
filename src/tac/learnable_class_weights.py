# SPDX-License-Identifier: MIT
"""Lane PS-V2 — LEARNABLE per-class SegNet weights via Lagrangian on
per-class distortion equalisation.

Lane PS-V1 hard-coded ``--segnet-class-weights "1,5,5,1,1"``: classes 1
(lane markings) and 2 (undrivable, including sky) are 5x boosted; classes
0 (road), 3 (movable), and 4 (my_car) keep weight 1. The numbers are
heuristic; the real scoring formula is per-pixel argmax disagreement averaged
across classes, so the optimal weighting depends on the per-class error
distribution which shifts during training.

Lane PS-V2 makes the per-class weights LEARNABLE. The parameterisation is
softmax (so weights sum to 1 and stay positive) over a 5-vector of raw
logits. The Lagrangian objective drives the per-class distortion variance
toward zero — i.e. pushes the optimiser to spend its budget on the
classes where the score is currently bottlenecked.

Math:

    weights_c = softmax(raw_c)             ∀ c ∈ [0..4]
    sum_c(weights_c) = 1   (softmax constraint)
    distortion_c = mean over pixels in class c of |pred ≠ gt|

    Lagrangian objective:
        L_total = sum_c weights_c * loss_c              # weighted loss
                + λ_var * Var(distortion_c)             # equalisation

The variance penalty drives the weights up on bottleneck classes (high
distortion) and down on solved classes (low distortion). At the optimum
all per-class distortions are equal — Pareto-optimal under the score
formula.

Why softmax (not softplus): the per-class weights only matter as RATIOS
(scaling all by α multiplies the weighted loss by α, which is absorbed
by the optimiser's learning rate). Softmax pins the scale and keeps the
weights interpretable as probabilities — operator can read ``weights[1]
= 0.42`` as "lane mark gets 42% of the per-class loss budget".

CLAUDE.md compliance:
- Pure PyTorch. CUDA-required at the call site (caller decides device).
- Tests: gradient flows through raw logits; softmax keeps weights >= 0;
  variance penalty drives equalisation; saved+loaded weights match.
- No global state, no MPS/CPU fallback.
- Mirrors the LearnableBitDepth/LearnableSaliencyThreshold pattern.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnableClassWeights",
    "compute_class_weight_dual_update",
    "compute_class_weight_equalisation_penalty",
    "compute_class_weighted_primal_loss",
    "save_learnable_class_weights",
    "load_learnable_class_weights",
]


class LearnableClassWeights(nn.Module):
    """Per-class dual multipliers for SegNet distortion pressure.

    Args:
        num_classes: number of SegNet classes (5 for the comma SegNet).
        warm_start: optional ``(num_classes,)`` non-negative tensor to
            initialise the weights. ``None`` ⇒ uniform (all = 1/num_classes).
            Common warm-start: the Lane PS-V1 ``[1, 5, 5, 1, 1]`` vector
            re-normalised to sum=1.

    ``base_weights`` and ``lambda_class`` are buffers, not Parameters.
    Optimizers must not update them by backprop. The only mutating path is
    :meth:`dual_update`, which applies
    ``λ_c <- max(0, λ_c + η * (distortion_c - target))``.
    """

    def __init__(
        self,
        num_classes: int = 5,
        *,
        warm_start: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if warm_start is None:
            base = torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
        else:
            if not isinstance(warm_start, torch.Tensor):
                raise TypeError(
                    f"warm_start must be a torch.Tensor, got "
                    f"{type(warm_start).__name__}"
                )
            if warm_start.ndim != 1 or warm_start.shape[0] != num_classes:
                raise ValueError(
                    f"warm_start shape {tuple(warm_start.shape)} != "
                    f"({num_classes},)"
                )
            if (warm_start < 0).any():
                raise ValueError("warm_start must be non-negative")
            ws = warm_start.detach().to(torch.float32).clamp_min(1e-9)
            base = ws / ws.sum()
        self.register_buffer("base_weights", base.clone())
        self.register_buffer("lambda_class", torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer(
            "running_target_distortion",
            torch.tensor(float("nan"), dtype=torch.float32),
        )
        self.register_buffer("dual_step", torch.tensor(0, dtype=torch.long))
        self.num_classes = int(num_classes)

    # ── Accessors ────────────────────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """Return detached base weights amplified by ``1 + λ_c``."""
        return self.base_weights * (1.0 + self.lambda_class)

    def weights(self) -> torch.Tensor:
        """Alias for ``forward()`` — explicit call site."""
        return self.forward()

    def lambdas(self) -> torch.Tensor:
        """Return the nonnegative class dual multipliers λ_c."""
        return self.lambda_class

    def csv(self) -> str:
        """Compatibility: emit a CSV string in the same format as the
        ``--segnet-class-weights`` flag.

        Round 13 (I-1): the emitted vector is L1-renormalised to sum
        to 1.0. The raw ``self.weights() = base_weights * (1 + λ_c)``
        sums to ``1 + sum(base_weights * λ_c)`` after dual updates,
        which can exceed 1; downstream consumers that read the CSV as
        a probability distribution would otherwise see an invalid
        normalisation. This guarantees the long-standing docstring
        contract that the values sum to 1.
        """
        raw = self.weights().detach().cpu()
        total = raw.sum().clamp_min(1e-12)
        w = (raw / total).tolist()
        return ",".join(f"{v:.6f}" for v in w)

    def state_for_save(self) -> dict:
        return {
            "schema_version": 2,
            "module": "tac.learnable_class_weights.LearnableClassWeights",
            "num_classes": self.num_classes,
            "base_weights": self.base_weights.detach().cpu().clone(),
            "lambda_class": self.lambda_class.detach().cpu().clone(),
            "weights": self.weights().detach().cpu().clone(),
            "running_target_distortion": self.running_target_distortion.detach().cpu().clone(),
            "dual_step": self.dual_step.detach().cpu().clone(),
        }

    @torch.no_grad()
    def dual_update(
        self,
        per_class_distortion: torch.Tensor,
        *,
        eta: float,
        target: torch.Tensor | float | None = None,
        running_mean_momentum: float = 0.99,
    ) -> torch.Tensor:
        """Apply projected dual ascent to class lambdas."""
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if not 0.0 <= running_mean_momentum < 1.0:
            raise ValueError(
                f"running_mean_momentum must be in [0, 1), got {running_mean_momentum}"
            )
        distortion = per_class_distortion.detach().to(
            device=self.lambda_class.device,
            dtype=self.lambda_class.dtype,
        )
        _validate_distortion_shape(self, distortion)

        if target is None:
            target_t = distortion.mean()
        else:
            target_t = torch.as_tensor(
                target, device=distortion.device, dtype=distortion.dtype,
            )
        self.lambda_class.add_(float(eta) * (distortion - target_t))
        self.lambda_class.clamp_(min=0.0)
        if bool(torch.isnan(self.running_target_distortion)):
            running = target_t
        else:
            running = (
                float(running_mean_momentum) * self.running_target_distortion.to(
                    distortion.device, distortion.dtype
                )
                + (1.0 - float(running_mean_momentum)) * target_t
            )
        self.running_target_distortion.copy_(running.detach())
        self.dual_step.add_(1)
        return self.lambda_class


def _validate_distortion_shape(
    class_weights: LearnableClassWeights,
    per_class_distortion: torch.Tensor,
) -> None:
    if per_class_distortion.shape != (class_weights.num_classes,):
        raise ValueError(
            f"per_class_distortion shape {tuple(per_class_distortion.shape)} != "
            f"({class_weights.num_classes},)"
        )
    if not torch.isfinite(per_class_distortion).all():
        raise ValueError(
            "per_class_distortion contains NaN/Inf — refuse to compute "
            "dual update against a corrupt distortion vector."
        )


def compute_class_weight_dual_update(
    class_weights: LearnableClassWeights,
    per_class_distortion: torch.Tensor,
    *,
    eta: float,
    target: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Functional wrapper for :meth:`LearnableClassWeights.dual_update`."""
    return class_weights.dual_update(
        per_class_distortion,
        eta=eta,
        target=target,
    )


def compute_class_weighted_primal_loss(
    class_weights: LearnableClassWeights,
    per_class_loss: torch.Tensor,
) -> torch.Tensor:
    """Primal loss using detached multipliers ``(1 + λ_c) * weight_c``."""
    if per_class_loss.shape != (class_weights.num_classes,):
        raise ValueError(
            f"per_class_loss shape {tuple(per_class_loss.shape)} != "
            f"({class_weights.num_classes},)"
        )
    multiplier = class_weights().detach().to(
        device=per_class_loss.device,
        dtype=per_class_loss.dtype,
    )
    return (multiplier * per_class_loss).sum()


def compute_class_weight_equalisation_penalty(
    class_weights: LearnableClassWeights,
    per_class_distortion: torch.Tensor,
    *,
    lambda_var: float = 1.0,
) -> torch.Tensor:
    """Retired compatibility shim.

    Round 10 replaced variance minimisation with projected dual ascent.
    This function validates inputs and returns a zero-valued tensor so old
    imports do not reintroduce backprop-updated class weights.
    """
    del lambda_var
    distortion = per_class_distortion.detach().to(
        device=class_weights.lambda_class.device,
        dtype=class_weights.lambda_class.dtype,
    )
    _validate_distortion_shape(class_weights, distortion)
    return class_weights.lambda_class.sum() * 0.0


def save_learnable_class_weights(
    module: LearnableClassWeights,
    path: str | Path,
) -> Path:
    """Serialise to ``path`` as a torch dict."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_for_save(), p)
    return p


def load_learnable_class_weights(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LearnableClassWeights:
    """Inverse of ``save_learnable_class_weights``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    # Loader-format-safety: this loader handles ONLY pytorch pickles, never
    # renderer .bin files. Verify magic bytes to prevent the 2026-04-26
    # DEN-V2 bug pattern.
    with open(p, "rb") as _f:
        _magic = _f.read(4)
    if _magic in (b"FP4A", b"ASYM", b"DPSM", b"I4LZ", b"CCh1", b"C3R1", b"SCv1", b"OMG1"):
        raise ValueError(
            f"learnable-class-weights file {p} has renderer magic {_magic!r} — "
            f"expected pytorch pickle. Wrong file?"
        )
    # WEIGHTS_ONLY_FALSE_ALLOWED: the magic-byte guard above rejects every
    # known renderer .bin format (FP4A/ASYM/DPSM/I4LZ/CCh1/C3R1/SCv1/OMG1)
    # BEFORE we hit torch.load, so the DEN-V2 pickle-on-renderer-bin crash
    # cannot fire here. The state dict we serialise contains plain tensors
    # + python ints/floats (no custom classes), so weights_only=False is
    # only required because the legacy snapshot dict layout pre-dates
    # PyTorch's safer weights_only path. Safe per check_loader_format_safety.
    state = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(
            f"{p} is not a learnable-class-weights snapshot "
            f"(got {type(state).__name__})"
        )
    schema_version = state.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError(
            f"{p} schema_version={state.get('schema_version')} not in (1, 2)"
        )
    if state.get("module") != "tac.learnable_class_weights.LearnableClassWeights":
        raise ValueError(
            f"{p} module={state.get('module')!r} != "
            "tac.learnable_class_weights.LearnableClassWeights"
        )
    nc = int(state["num_classes"])
    cw = LearnableClassWeights(nc)
    if schema_version == 2:
        cw.base_weights.copy_(state["base_weights"].to(cw.base_weights.dtype))
        cw.lambda_class.copy_(state["lambda_class"].to(cw.lambda_class.dtype))
        if "running_target_distortion" in state:
            cw.running_target_distortion.copy_(
                state["running_target_distortion"].to(
                    cw.running_target_distortion.dtype
                )
            )
        if "dual_step" in state:
            cw.dual_step.copy_(state["dual_step"].to(cw.dual_step.dtype))
    else:
        if "raw_logits" in state:
            base = F.softmax(state["raw_logits"].to(torch.float32), dim=0)
        elif "weights" in state:
            weights = state["weights"].to(torch.float32).clamp_min(1e-9)
            base = weights / weights.sum()
        else:
            raise ValueError(f"{p} v1 snapshot has neither 'raw_logits' nor 'weights'")
        cw.base_weights.copy_(base.to(cw.base_weights.dtype))
    return cw
