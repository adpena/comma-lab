# SPDX-License-Identifier: MIT
"""Witness training-collapse stability levers (AMBER deep-unroll unblock; #146/#211).

DIAGNOSIS (SETTLED in the DAG, FEED-amber-unblock — do NOT re-diagnose). Deep-unroll / per-pair
(batch=1) witness training diverges to a DEAD-SATURATION LOCK: d_seg drops cleanly, then the loss
SPIKES (57 -> 1070) and FREEZES at EXACTLY ``1070.0802`` (6-decimal-place identical across steps) =
OPTIMIZER DIVERGENCE (the sigmoid rails -> gradient -> 0 -> AdamW step -> 0; the number is FINITE, not
NaN). Four ranked destabilizers:

1. **No grad clipping** — the global grad norm is unbounded.
2. **sqrt-pose-eps gradient blowup** — the score-domain pose term ``sqrt(10*d_pose + eps)`` has
   gradient coefficient ``d/d(pose) sqrt(10*pose+eps) = 5/sqrt(10*pose+eps)``, which -> ``5/sqrt(eps)``
   for EASY (near-zero-pose) pairs. At ``eps=1e-8`` that is **5e4**; a per-pair (batch=1) update EXPOSES
   it with no averaging.
3. **w_seg=100** = 100x the seg learning-rate (interacts with #1/#2).
4. **lr 2e-3 x 600 steps/ep** — a large effective step.

The contrast that did NOT collapse: raw-loss torch n48 (w_seg=1, no sqrt). So the sqrt-blowup x 100x x
no-clip is the destabilizer.

STATE (2026-07-09, MEASURED via source inspection of the live trainer
``experiments/train_levelset_witness_realized_through_R_mlx.py``): destabilizers #1 and #2 are ALREADY
tamed in the incumbent default config — ``--grad-clip`` defaults to ``1.0`` (``mlx.optimizers.
clip_grad_norm`` wired at the opt step) and ``--pose-eps`` defaults to ``1e-2`` (coefficient bound 50,
DOWN from the 5e4 the diagnosis saw at 1e-8). So the primary collapse cure is IN PLACE on the incumbent.

This module is the FORMALIZATION + ERGONOMIC + COMPOSITION layer on top of that:
  * the pose-eps <-> max-grad-coefficient LAW that CONNECTS destabilizers #1/#2 (a real invertible
    identity, not a heuristic) — reusable + tested + registered as a canonical equation;
  * the ``pose_grad_coeff_max`` reparametrization ("bound the coefficient to <= C" -> the derived eps
    floor) so an operator can specify the STABILITY BUDGET directly instead of the eps;
  * the ``amber`` PRESET that composes the collapse cures TIGHTER for the batch=1 deep-unroll arm
    (grad-clip 0.5 + coeff-max 25 + per-group-grad-clip);
  * the stage-boundary loss-weight GUARD (SPEC_v75 §8-C: loss weights change at STAGE boundaries only,
    never per-step).

DEFAULT byte-identical (#205-safe): ``stability_preset='none'`` AND ``pose_grad_coeff_max<=0`` return
the incumbent ``pose_eps`` / ``grad_clip`` / ``per_group_grad_clip`` UNCHANGED — no override, no
trajectory change, the live run resumes byte-identical.

means != ends: this module BUILDS the stability mechanism + PROVES (unit-level, $0) that the cures cap
the coefficient / grad norm on a SYNTHETIC reproduction of the blowup. It does NOT claim the AMBER arm
un-collapses at n600 — that is a SEPARATE operator-GO heavy launch (un-collapse A/B OWED). Pointer
0.19110 UNMOVED.

Law: ``tac.canonical_equations.witness_pose_grad_coeff_stability_20260709``.
DSL leg: ``tac.witness_dsl.curriculum_dsl.WitnessStability``.
Consumer: ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (post-parse resolution).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "SCORE_POSE_CONST",
    "StabilityViolation",
    "StabilityPreset",
    "StabilityConfig",
    "AMBER",
    "PRESETS",
    "max_pose_grad_coeff",
    "pose_eps_floor_for_coeff_max",
    "resolve_effective_pose_eps",
    "resolve_stability_config",
    "assert_loss_weights_stage_boundary_only",
    "per_param_normalize_grads",
    "overflow_state_penalty",
]

# The multiplicative constant in the contest score's pose term ``sqrt(10 * d_pose)`` (upstream
# evaluate.py). The gradient of ``sqrt(a*p + eps)`` w.r.t. ``p`` is ``a / (2*sqrt(a*p + eps))``; with
# a = SCORE_POSE_CONST the coefficient at p=0 is ``a / (2*sqrt(eps)) = 5/sqrt(eps)``.
SCORE_POSE_CONST: float = 10.0


class StabilityViolation(ValueError):
    """Raised by :func:`assert_loss_weights_stage_boundary_only` when a loss weight changes mid-stage
    (SPEC_v75 §8-C: loss weights change at STAGE boundaries only, never per-step)."""


# ---------------------------------------------------------------------------
# The pose-eps <-> max-grad-coefficient law (exact, invertible)
# ---------------------------------------------------------------------------
def max_pose_grad_coeff(pose_eps: float, score_pose_const: float = SCORE_POSE_CONST) -> float:
    """Maximum gradient COEFFICIENT of the score-domain pose term ``sqrt(a*p + eps)`` over ``p >= 0``.

    ``d/dp sqrt(a*p + eps) = a / (2*sqrt(a*p + eps))`` is monotone-DECREASING in ``p``, so its supremum
    is at ``p = 0``: ``a / (2*sqrt(eps))``. With ``a = 10`` this is ``5/sqrt(eps)`` — the 5e4-at-1e-8
    blowup the diagnosis identified.

    Raises on ``pose_eps <= 0`` (the coefficient is unbounded / the term is non-differentiable at 0).
    """
    a = float(score_pose_const)
    e = float(pose_eps)
    if e <= 0.0:
        raise ValueError(f"pose_eps must be > 0 (coefficient unbounded at eps<=0), got {e!r}")
    if a <= 0.0:
        raise ValueError(f"score_pose_const must be > 0, got {a!r}")
    return a / (2.0 * math.sqrt(e))


def pose_eps_floor_for_coeff_max(
    coeff_max: float, score_pose_const: float = SCORE_POSE_CONST,
) -> float:
    """The smallest ``pose_eps`` whose max pose gradient coefficient is ``<= coeff_max`` — the INVERSE
    of :func:`max_pose_grad_coeff`.

    From ``a / (2*sqrt(eps)) <= C`` -> ``sqrt(eps) >= a/(2C)`` -> ``eps >= (a/(2C))**2``. With ``a=10``
    this is ``(5/C)**2`` (e.g. C=50 -> 1e-2, C=25 -> 4e-2). Raises on ``coeff_max <= 0``.
    """
    a = float(score_pose_const)
    c = float(coeff_max)
    if c <= 0.0:
        raise ValueError(f"coeff_max must be > 0, got {c!r}")
    if a <= 0.0:
        raise ValueError(f"score_pose_const must be > 0, got {a!r}")
    return (a / (2.0 * c)) ** 2


def resolve_effective_pose_eps(
    pose_eps: float,
    pose_grad_coeff_max: float = 0.0,
    score_pose_const: float = SCORE_POSE_CONST,
) -> float:
    """The effective ``pose_eps`` after applying an OPTIONAL coefficient bound.

    ``pose_grad_coeff_max <= 0`` (DEFAULT) => returns ``pose_eps`` UNCHANGED (byte-identical). Otherwise
    returns ``max(pose_eps, pose_eps_floor_for_coeff_max(coeff_max))`` — the bound can only RAISE the eps
    (tighten stability), never lower it, so it can never DESTABILISE a config that is already safer.

    NOTE (honest): for the ``sqrt(a*p + eps)`` functional form, bounding the gradient coefficient is
    MATHEMATICALLY IDENTICAL to raising the eps floor (there is no separate non-biasing clamp for this
    term); the trade is a negligible forward-loss bias of ``eps`` inside the sqrt (at eps=4e-2 the pose
    term shifts by <= 0.2 in sqrt-units only for d_pose << 4e-3, i.e. only where the term is already
    tiny). The GLOBAL ``--grad-clip`` handles the total-loss norm; this handles the per-term coefficient.
    """
    if float(pose_grad_coeff_max) <= 0.0:
        return float(pose_eps)
    floor = pose_eps_floor_for_coeff_max(pose_grad_coeff_max, score_pose_const)
    return max(float(pose_eps), floor)


# ---------------------------------------------------------------------------
# Stability presets (composed cures) + resolved config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StabilityPreset:
    """A named bundle of the collapse cures for a training arm.

    ``grad_clip`` = global grad-norm clip budget; ``pose_grad_coeff_max`` = the pose coefficient bound
    (-> effective pose_eps floor); ``per_group_grad_clip`` = the C4-confound cure (clip each param
    GROUP independently so a volatile regularizer gradient cannot throttle the seg/pose gradient on
    other groups via the shared 1/gnorm scale)."""

    name: str
    grad_clip: float
    pose_grad_coeff_max: float
    per_group_grad_clip: bool


# AMBER = the batch=1 deep-unroll arm tightening. TIGHTER than the incumbent (grad-clip 1.0 / pose-eps
# 1e-2 => coeff 50), because per-pair updates have NO averaging so a single easy-pair coefficient is
# fully exposed: grad-clip 0.5 (halve the step budget), coeff-max 25 (eps floor 4e-2, half the
# incumbent coefficient), per-group clip ON (isolate regularizer volatility). These are the values the
# diagnosis implies for the divergence-prone arm; the un-collapse A/B that MEASURES them is operator-GO.
AMBER = StabilityPreset(
    name="amber", grad_clip=0.5, pose_grad_coeff_max=25.0, per_group_grad_clip=True,
)

PRESETS: Mapping[str, "StabilityPreset | None"] = {"none": None, "amber": AMBER}


@dataclass(frozen=True)
class StabilityConfig:
    """The resolved stability knobs the trainer applies + a provenance record.

    ``changed`` is False exactly when nothing moved from the incoming incumbent values (the
    byte-identity contract). ``provenance()`` is an observability row (score-neutral)."""

    grad_clip: float
    effective_pose_eps: float
    incoming_pose_eps: float
    pose_grad_coeff_max: float
    per_group_grad_clip: bool
    preset: str
    max_pose_grad_coeff_effective: float
    changed: bool
    _notes: tuple[str, ...] = field(default_factory=tuple)

    def provenance(self) -> dict:
        return {
            "preset": self.preset,
            "grad_clip": self.grad_clip,
            "incoming_pose_eps": self.incoming_pose_eps,
            "effective_pose_eps": self.effective_pose_eps,
            "pose_grad_coeff_max": self.pose_grad_coeff_max,
            "per_group_grad_clip": self.per_group_grad_clip,
            "max_pose_grad_coeff_effective": self.max_pose_grad_coeff_effective,
            "changed": self.changed,
            "notes": list(self._notes),
        }


def resolve_stability_config(
    *,
    grad_clip: float,
    pose_eps: float,
    pose_grad_coeff_max: float = 0.0,
    stability_preset: str = "none",
    per_group_grad_clip: bool = False,
    score_pose_const: float = SCORE_POSE_CONST,
) -> StabilityConfig:
    """Compose the stability knobs.

    Contract:
      * ``stability_preset='none'`` AND ``pose_grad_coeff_max<=0`` => everything UNCHANGED (byte-
        identical incumbent; ``changed=False``). This is the #205-safe default path.
      * ``pose_grad_coeff_max>0`` (with any preset) => raise the effective pose_eps to the coefficient
        floor (an EXPLICIT coeff bound always wins over a preset's).
      * a named preset (currently 'amber') => apply its grad_clip / per_group / coeff-max, but an
        explicit ``pose_grad_coeff_max`` overrides the preset's coeff-max.
    """
    preset_name = str(stability_preset or "none")
    if preset_name not in PRESETS:
        raise ValueError(f"unknown stability_preset {preset_name!r} (known: {sorted(PRESETS)})")
    preset = PRESETS[preset_name]

    in_grad_clip = float(grad_clip)
    in_pose_eps = float(pose_eps)
    in_per_group = bool(per_group_grad_clip)
    in_coeff_max = float(pose_grad_coeff_max)

    out_grad_clip = in_grad_clip
    out_per_group = in_per_group
    coeff_max = in_coeff_max
    notes: list[str] = []

    if preset is not None:
        out_grad_clip = float(preset.grad_clip)
        out_per_group = bool(preset.per_group_grad_clip)
        # explicit coeff-max on the CLI (>0) wins over the preset's default coeff-max
        if in_coeff_max <= 0.0:
            coeff_max = float(preset.pose_grad_coeff_max)
        notes.append(f"preset:{preset_name}")

    effective_pose_eps = resolve_effective_pose_eps(in_pose_eps, coeff_max, score_pose_const)
    if effective_pose_eps != in_pose_eps:
        notes.append(
            f"pose_eps {in_pose_eps:g}->{effective_pose_eps:g} "
            f"(coeff_max<={coeff_max:g})"
        )
    max_coeff = max_pose_grad_coeff(effective_pose_eps, score_pose_const)

    changed = (
        out_grad_clip != in_grad_clip
        or out_per_group != in_per_group
        or effective_pose_eps != in_pose_eps
    )
    return StabilityConfig(
        grad_clip=out_grad_clip,
        effective_pose_eps=effective_pose_eps,
        incoming_pose_eps=in_pose_eps,
        pose_grad_coeff_max=coeff_max,
        per_group_grad_clip=out_per_group,
        preset=preset_name,
        max_pose_grad_coeff_effective=max_coeff,
        changed=changed,
        _notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Cells2Pixels NCA deep-unroll stabilizers (SIGGRAPH'26; memo
# .omx/research/cells2pixels_deepdive_bridge_20260709.md) — opt-in companions
# ---------------------------------------------------------------------------
def per_param_normalize_grads(grads, *, tree_map, leaf_norm, eps: float = 1e-8):
    """Cells2Pixels per-PARAMETER gradient normalization: ``g_p <- g_p / (||g_p|| + eps)`` per tensor.

    Scale-FREE (every parameter tensor contributes a UNIT-norm direction), a stronger deep-unroll NCA
    stabilizer than a single global-norm clip and a candidate BETTER PRIMARY for the batch=1 arm. The
    ``tree_map`` (maps a leaf fn over the grad tree) and ``leaf_norm`` (tensor -> scalar norm) are
    INJECTED so this module stays framework-free (no mlx/torch import) and unit-testable with a plain
    dict tree + numpy norm; the trainer passes ``mlx.utils.tree_map`` + an mlx L2 norm.

    CAVEAT (document + A/B, per the memo): per-param normalization ALTERS the seg-vs-pose gradient
    SCALE ratio (each head's tensors are individually unit-normed), so it is NOT proven neutral for our
    joint objective — an OWED trajectory A/B, not a byte-identical change. Only fire it on the
    divergence-prone deep-unroll arm; default 'none' keeps the incumbent grad path byte-identical.
    """
    e = float(eps)
    return tree_map(lambda g: g / (leaf_norm(g) + e), grads)


def overflow_state_penalty(state_abs_max, clamp: float = 1.0):
    """Cells2Pixels OVERFLOW / state-clamp penalty (their config weight ~100): a hinge penalty on the
    NCA CELL-STATE magnitude exceeding ``clamp`` — ``mean(relu(|state| - clamp))`` — that directly
    prevents the unbounded-state -> sigmoid-rail DEAD-LOCK this fix targets (we currently LACK it).

    ``state_abs_max`` is ``|state|`` (any array-like with ``+``/``-``/``max``-with-0/``mean`` semantics;
    the trainer passes the witness's PRE-activation state magnitude). Pure + framework-agnostic: works
    on a numpy array (returns a numpy scalar) or an mlx array (returns a lazy mlx scalar) since it uses
    only ``max(x - clamp, 0)`` + ``mean``. Degeneracy: ``|state| <= clamp`` everywhere => penalty 0.

    WIRE-IN STATUS (honest, NO-FAKE): this is the READY-TO-WIRE primitive; the faithful consumer needs
    the witness's PRE-sigmoid state exposed at the loss boundary (a loss-construction edit deferred
    while two sisters edit that surface). NAMED in DAG FEED-collapsefix as the deep-unroll companion
    OWED wire-in; a --overflow-loss-weight flag lands WITH the state exposure (never a no-op flag)."""
    import numpy as _np

    arr = state_abs_max
    if isinstance(arr, _np.ndarray) or _np.isscalar(arr):
        a = _np.asarray(arr, dtype=_np.float64)
        return _np.mean(_np.maximum(a - float(clamp), 0.0))
    # array-protocol (e.g. mlx): use the object's own ops (relu via a max-with-0 pattern)
    over = arr - float(clamp)
    zero = over * 0.0
    try:  # mlx / numpy-like both expose .mean via a module fn; fall back to python
        import mlx.core as _mx  # noqa: PLC0415

        if type(arr).__module__.startswith("mlx"):
            return _mx.mean(_mx.maximum(over, zero))
    except Exception:  # pragma: no cover - mlx optional
        pass
    return (over if over > zero else zero)


# ---------------------------------------------------------------------------
# Stage-boundary loss-weight guard (SPEC_v75 §8-C)
# ---------------------------------------------------------------------------
def assert_loss_weights_stage_boundary_only(
    weight_events: "list[tuple[int, float, float]] | tuple[tuple[int, float, float], ...]",
    stage_boundary_epochs: "set[int] | frozenset[int] | tuple[int, ...] | list[int]",
) -> None:
    """Assert the loss weights ``(w_seg, w_pose)`` change ONLY at stage boundaries (SPEC_v75 §8-C).

    ``weight_events`` = a chronologically-ordered list of ``(epoch, w_seg, w_pose)`` observations (one
    per epoch, or one per change). ``stage_boundary_epochs`` = the epochs at which a stage transition
    is ALLOWED to change the weights (the curriculum boundaries + the initial epoch).

    Raises :class:`StabilityViolation` on the FIRST weight change whose epoch is not a stage boundary.
    A run that holds ``w_seg``/``w_pose`` constant (the incumbent) passes trivially. Pure + $0.
    """
    boundaries = set(int(e) for e in stage_boundary_epochs)
    prev: "tuple[float, float] | None" = None
    prev_epoch: "int | None" = None
    for epoch, w_seg, w_pose in weight_events:
        cur = (float(w_seg), float(w_pose))
        ep = int(epoch)
        if prev is not None and cur != prev and ep not in boundaries:
            raise StabilityViolation(
                f"loss weights changed at epoch {ep} ({prev} -> {cur}) which is NOT a stage boundary "
                f"(allowed: {sorted(boundaries)}); SPEC_v75 §8-C forbids per-step weight changes "
                f"(prev change-free since epoch {prev_epoch})"
            )
        prev = cur
        prev_epoch = ep
