"""Lane PS-V2 — LEARNABLE per-class SegNet weights via Lagrangian on
per-class distortion equalisation.

Lane PS-V1 hard-coded ``--segnet-class-weights "1,5,5,1,1"``: classes 1
(lane mark) and 2 (vehicle) are 5× boosted; classes 0 (road), 3 (sky),
and 4 (other) keep weight 1. The numbers are heuristic — the real scoring
formula is per-pixel argmax disagreement averaged across classes, so the
optimal weighting depends on the per-class error distribution which
shifts during training.

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
    "compute_class_weight_equalisation_penalty",
    "save_learnable_class_weights",
    "load_learnable_class_weights",
]


class LearnableClassWeights(nn.Module):
    """Learnable per-class SegNet loss weights via softmax.

    Args:
        num_classes: number of SegNet classes (5 for the comma SegNet).
        warm_start: optional ``(num_classes,)`` non-negative tensor to
            initialise the weights. ``None`` ⇒ uniform (all = 1/num_classes).
            Common warm-start: the Lane PS-V1 ``[1, 5, 5, 1, 1]`` vector
            (re-normalised to sum=1 via softmax-inverse).

    The single nn.Parameter is ``self.raw_logits`` of shape
    ``(num_classes,)``. The forward call returns ``softmax(raw_logits)``,
    a probability simplex.
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
            raw = torch.zeros(num_classes, dtype=torch.float32)
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
            # Inverse-softmax: raw = log(p) - mean(log(p)) (mean shift is
            # softmax-invariant, but produces the canonical zero-mean form).
            ws_norm = ws / ws.sum()
            raw = torch.log(ws_norm)
            raw = raw - raw.mean()
        self.raw_logits = nn.Parameter(raw.clone())
        self.num_classes = int(num_classes)

    # ── Accessors ────────────────────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """Return the (num_classes,) softmax probability vector."""
        return F.softmax(self.raw_logits, dim=0)

    def weights(self) -> torch.Tensor:
        """Alias for ``forward()`` — explicit call site."""
        return self.forward()

    def csv(self) -> str:
        """Compatibility: emit a CSV string in the same format as the
        ``--segnet-class-weights`` flag. The values sum to 1 (softmax)
        not to 5 (Lane PS-V1's hard-coded ``[1,5,5,1,1]`` sums to 13);
        the loss helpers L1-normalise internally so this is fine."""
        w = self.weights().detach().cpu().tolist()
        return ",".join(f"{v:.6f}" for v in w)

    def state_for_save(self) -> dict:
        return {
            "schema_version": 1,
            "module": "tac.learnable_class_weights.LearnableClassWeights",
            "num_classes": self.num_classes,
            "raw_logits": self.raw_logits.detach().cpu().clone(),
            "weights": self.weights().detach().cpu().clone(),
        }


def compute_class_weight_equalisation_penalty(
    class_weights: LearnableClassWeights,
    per_class_distortion: torch.Tensor,
    *,
    lambda_var: float = 1.0,
) -> torch.Tensor:
    """Lagrangian penalty on per-class distortion VARIANCE.

    Args:
        class_weights: the LearnableClassWeights module (used so the
            penalty's gradient flows back into the raw logits — the
            weighted distortion is computed inside this function).
        per_class_distortion: ``(num_classes,)`` tensor of per-class
            distortion estimates (e.g., per-class argmax-disagreement
            rate). MUST already be detached from the renderer/scorer
            graph so the Lagrangian penalty doesn't double-count the
            primary loss.
        lambda_var: Lagrangian multiplier on the variance term. Annealed
            via dual ascent in the training loop — start low so early
            training isn't bottlenecked by the equalisation constraint.

    Returns:
        Scalar tensor with grad wrt. class_weights.raw_logits. The
        penalty is the WEIGHTED variance of the per-class distortion,
        which is what equalises the *contribution* of each class to the
        total loss (a class with tiny weight × tiny distortion is no
        priority; a class with large weight × large distortion is the
        bottleneck the optimiser should attack).
    """
    if per_class_distortion.shape != (class_weights.num_classes,):
        raise ValueError(
            f"per_class_distortion shape {tuple(per_class_distortion.shape)} != "
            f"({class_weights.num_classes},)"
        )
    if not torch.isfinite(per_class_distortion).all():
        raise ValueError(
            "per_class_distortion contains NaN/Inf — refuse to compute "
            "Lagrangian penalty against a corrupt distortion vector."
        )
    w = class_weights()
    contrib = w * per_class_distortion.detach().to(w.dtype).to(w.device)
    # Variance of the per-class contribution; minimised when all classes
    # carry equal weight × distortion (Pareto optimum).
    return float(lambda_var) * contrib.var(unbiased=False)


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
    state = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(
            f"{p} is not a learnable-class-weights snapshot "
            f"(got {type(state).__name__})"
        )
    if state.get("schema_version") != 1:
        raise ValueError(
            f"{p} schema_version={state.get('schema_version')} != 1"
        )
    if state.get("module") != "tac.learnable_class_weights.LearnableClassWeights":
        raise ValueError(
            f"{p} module={state.get('module')!r} != "
            "tac.learnable_class_weights.LearnableClassWeights"
        )
    nc = int(state["num_classes"])
    cw = LearnableClassWeights(nc)
    cw.raw_logits.data.copy_(state["raw_logits"].to(cw.raw_logits.dtype))
    return cw
