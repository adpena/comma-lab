# SPDX-License-Identifier: MIT
"""Lever A — the EQUIMARGINAL pose-weight controller (the WEIGHT analogue of the APGC cadence).

Design memo: ``.omx/research/sophisticated_pose_treatment_design_20260616T222900Z.md``.

THE PRINCIPLE (the score calculus made into an online controller). The driver minimises the per-pair
Lagrangian ``L = w_seg·seg_l(F) + w_pose·pose_l(F)`` where ``F`` is the SHARED decoded frame tensor.
Crucially ``pose_l = sqrt(10·pose_mse + 1e-12)`` is **already the contest pose-term in score units**
(``upstream/evaluate.py:92`` + ``tac.score_composition``: ``S = 100·d_seg + sqrt(10·d_pose) + 25·rate``),
so the per-axis frame cotangents the split-by-head backward ALREADY computes —
``cot_seg = ∂(w_seg·seg_l)/∂F`` and ``cot_pose = ∂(w_pose·pose_l)/∂F`` — carry the per-axis SCORE PULL on
the shared frame. The sqrt bakes the marginal ``∂pose_l/∂pose_mse = 5/sqrt(10·pose_mse)`` into ``cot_pose``
by the chain rule, so as ``d_pose`` falls the pose pull rises automatically — the operating-point-dependent
``∂S/∂d_pose`` is measured, not assumed.

THE EQUIMARGINAL RULE. Two axes are at the score-optimal balance when one marginal step moves the SCORE by
the same amount on each axis. The measurable, NO-FAKE version: balance the per-axis cotangent norms,
``‖cot_pose‖ ≈ ρ·‖cot_seg‖`` with ``ρ`` the desired pose:seg score-marginal ratio (default ``ρ = 1.0``).
Because the pose term is already in score units and ``w_seg`` calibrates the seg surrogate's score units,
balancing the cotangent norms balances the per-step score pull.

THE CONTROLLER (closed-loop, EMA-smoothed, deadbanded, clamped — mirrors ``_adaptive_do_pose``'s structure
but cadences the WEIGHT not the schedule):

    ratio_t    = ‖cot_pose‖ / max(‖cot_seg‖, eps)          # measured at the CURRENT w_pose
    ratio_ema  = decay·ratio_ema + (1−decay)·ratio_t       # smoothed
    if |ratio_ema − ρ| > tol·ρ:                            # outside the deadband
        w_pose ← clamp(w_pose · (ρ / ratio_ema), w_pose0·lo, w_pose0·hi)

``ratio_t`` is measured at the cotangent norms that were computed WITH the current ``w_pose`` already folded
in (the backward weights each head's loss before computing its leaf grad). So ``ratio_t = w_pose·r_unit /
(w_seg·s_unit)`` where ``r_unit``/``s_unit`` are the unit-weight axis norms. The multiplicative correction
``w_pose·(ρ/ratio_ema)`` drives ``ratio`` toward ``ρ`` (a fixed point: at ``ratio_ema == ρ`` the multiplier
is 1.0). The clamp to ``[lo, hi]·w_pose0`` bounds the controller so it can never starve seg or blow up pose.

WHY THIS IS THE SCORE-OPTIMAL WEIGHT (not a hand-set constant). It TRACKS ``5/sqrt(10·d_pose)``: as d_pose
falls, ``∂pose_l/∂pose_mse`` rises → ``‖cot_pose‖`` rises → measured ``ratio_ema`` rises → the controller
AUTO-LOWERS ``w_pose`` to hold the equimarginal point. A hand-set constant cannot follow this curve. The
oomph refinement stages scale ``w_seg`` up 1.5× with NO matching pose adjustment; this controller RESTORES
the balance the oomph overlay breaks (the warm-throttled d_pose-drift regression source).

DEFAULT-OFF / BYTE-IDENTICAL. The driver only consults this controller when
``cfg.pose_equimarginal_enabled`` is True. OFF → ``w_pose`` is the unmodified stage value every epoch →
byte-identical gradient. The controller itself holds only a float EMA + a step counter; it is checkpointed
so a resume CONTINUES the same trajectory.

AUTHORITY. The cotangent norms are TELEMETRY in score units; the controller's ``w_pose`` choice is a training
hyperparameter, not a score claim. The in-loop d_pose remains ``[contest-CPU advisory]`` NON-PROMOTABLE until
the byte-closed archive is run through ``upstream/evaluate.py``.

NO-FAKE (class 1/2). :func:`equimarginal_pose_weight_multiplier` is a pure deterministic function of the two
measured cotangent norms + the controller state; a test feeds known norms and asserts the multiplier moves
``w_pose`` toward ``ρ`` and respects the clamp + deadband, and that the OFF / ρ-matched cases are exact
no-ops (multiplier == 1.0). It does NOT return canonical markers without doing the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EquimarginalPoseWeightController",
    "EquimarginalPoseWeightState",
    "equimarginal_pose_weight_multiplier",
]

_EPS = 1e-12


@dataclass
class EquimarginalPoseWeightState:
    """Checkpointable controller state (a smoothed ratio + a step counter).

    ``ratio_ema is None`` means "not yet seeded" — the first measured ``ratio_t`` seeds it directly (no
    decay blend on the first sample, so a single-epoch run still reacts). ``steps`` counts the epochs the
    controller has been consulted (telemetry + resume continuity)."""

    ratio_ema: float | None = None
    steps: int = 0

    def to_dict(self) -> dict[str, float | int | None]:
        return {"ratio_ema": self.ratio_ema, "steps": int(self.steps)}

    @classmethod
    def from_dict(cls, d: dict | None) -> EquimarginalPoseWeightState:
        if not d:
            return cls()
        re = d.get("ratio_ema", None)
        return cls(
            ratio_ema=(None if re is None else float(re)),
            steps=int(d.get("steps", 0)),
        )


def equimarginal_pose_weight_multiplier(
    cot_seg_norm: float,
    cot_pose_norm: float,
    state: EquimarginalPoseWeightState,
    *,
    rho: float,
    decay: float,
    tol: float,
    clamp_lo: float,
    clamp_hi: float,
) -> float:
    """Return the multiplicative correction ``m`` to apply to the CURRENT ``w_pose`` so the next epoch's
    per-axis cotangent-norm ratio moves toward ``rho``. MUTATES ``state.ratio_ema`` + ``state.steps``.

    ``cot_seg_norm`` / ``cot_pose_norm``: the L2 norms of the per-axis frame cotangents measured THIS epoch
        (WITH the current ``w_pose``/``w_seg`` already folded in by the backward).
    ``rho``: target pose:seg score-marginal ratio (1.0 = true equimarginal; <1 biases toward seg).
    ``decay``: EMA decay for ``ratio_ema`` (e.g. 0.9). ``tol``: deadband half-width as a FRACTION of ``rho``
        (e.g. 0.15 → no correction while ``ratio_ema ∈ [rho·(1−tol), rho·(1+tol)]``).
    ``clamp_lo`` / ``clamp_hi``: the multiplier is clamped to ``[clamp_lo, clamp_hi]`` so a single epoch can
        never move ``w_pose`` by more than this factor (the caller additionally clamps the ACCUMULATED
        ``w_pose`` to ``[lo, hi]·w_pose0``).

    Returns ``m`` with ``new_w_pose = clamp(w_pose · m, ...)`` done by the caller. ``m == 1.0`` exactly when
    inside the deadband or when ``ratio_ema == rho`` (a true no-op). FAIL-SOFT: if ``cot_seg_norm`` is ~0
    (degenerate — no seg pull) the ratio is treated as +inf → the controller drives ``w_pose`` DOWN by
    ``clamp_lo`` (do not amplify pose against a vanishing seg signal); if ``cot_pose_norm`` is ~0 it drives
    UP by ``clamp_hi`` (pose pull vanished, restore it). Both are bounded by the clamp."""
    s = max(float(cot_seg_norm), 0.0)
    p = max(float(cot_pose_norm), 0.0)
    rho = float(rho)
    # measured ratio this epoch (pose pull / seg pull), with fail-soft on degenerate norms.
    if s <= _EPS:  # noqa: SIM108 — explicit branch is clearer than a nested ternary here
        ratio_t = float("inf") if p > _EPS else rho  # no seg pull: inf (drive w_pose down); both ~0: no-op
    else:
        ratio_t = p / s
    # EMA blend (seed directly on the first sample so a 1-epoch run reacts).
    if state.ratio_ema is None or not _finite(state.ratio_ema):
        ratio_ema = ratio_t
    elif not _finite(ratio_t):
        ratio_ema = ratio_t  # an inf measurement propagates (degenerate seg signal)
    else:
        d = float(decay)
        ratio_ema = d * state.ratio_ema + (1.0 - d) * ratio_t
    state.ratio_ema = ratio_ema
    state.steps += 1
    # deadband: no correction while the smoothed ratio is within tol·rho of rho.
    if _finite(ratio_ema):
        if rho <= 0.0:
            return 1.0  # rho<=0 disables the correction (defensive; the caller validates rho>0)
        if abs(ratio_ema - rho) <= tol * rho:
            return 1.0
        if ratio_ema <= _EPS:  # noqa: SIM108 — explicit branch documents the vanished-pose case
            m = clamp_hi  # pose pull vanished → restore it (bounded)
        else:
            m = rho / ratio_ema
    else:
        # ratio_ema == +inf (seg pull vanished) → drive w_pose DOWN (do not amplify pose).
        m = clamp_lo
    return float(min(max(m, float(clamp_lo)), float(clamp_hi)))


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


@dataclass
class EquimarginalPoseWeightController:
    """Stateful wrapper around :func:`equimarginal_pose_weight_multiplier` that accumulates the effective
    ``w_pose`` across epochs and applies the ACCUMULATED clamp to ``[lo, hi]·w_pose0``.

    Usage in the driver (default-OFF guarded by ``cfg.pose_equimarginal_enabled``)::

        ctrl = EquimarginalPoseWeightController(rho=cfg.pose_equimarginal_rho, ...)
        # per epoch, after the seg+pose cotangents are computed:
        w_pose_eff = ctrl.update(cot_seg_norm, cot_pose_norm, w_pose_base=spec.pose_weight)

    The controller is checkpointed via ``state_dict()`` / ``load_state_dict()`` so a resume continues the
    same ``w_pose`` trajectory (else it snaps back to ``w_pose0`` and the balance jolts)."""

    rho: float = 1.0
    decay: float = 0.9
    tol: float = 0.15
    bound_lo: float = 0.25  # accumulated w_pose floor as a fraction of w_pose0
    bound_hi: float = 4.0  # accumulated w_pose ceiling as a fraction of w_pose0
    step_clamp_lo: float = 0.5  # per-epoch multiplier floor (bounds the single-step move)
    step_clamp_hi: float = 2.0  # per-epoch multiplier ceiling
    state: EquimarginalPoseWeightState = None  # type: ignore[assignment]
    # the running effective w_pose as a FRACTION of w_pose0 (1.0 == unmodified). Persisted.
    _w_pose_frac: float = 1.0

    def __post_init__(self) -> None:
        if not (self.rho > 0.0):
            raise ValueError(f"rho must be > 0 (got {self.rho})")
        if not (0.0 <= self.decay < 1.0):
            raise ValueError(f"decay must be in [0,1) (got {self.decay})")
        if not (self.tol > 0.0):
            raise ValueError(f"tol must be > 0 (got {self.tol})")
        if not (0.0 < self.bound_lo <= 1.0 <= self.bound_hi):
            raise ValueError(
                f"require 0 < bound_lo <= 1 <= bound_hi (got {self.bound_lo}, {self.bound_hi})"
            )
        if not (0.0 < self.step_clamp_lo <= 1.0 <= self.step_clamp_hi):
            raise ValueError(
                f"require 0 < step_clamp_lo <= 1 <= step_clamp_hi (got "
                f"{self.step_clamp_lo}, {self.step_clamp_hi})"
            )
        if self.state is None:
            self.state = EquimarginalPoseWeightState()

    def update(self, cot_seg_norm: float, cot_pose_norm: float, *, w_pose_base: float) -> float:
        """Advance the controller one epoch and return the EFFECTIVE ``w_pose`` to use this epoch.

        ``w_pose_base`` is the stage's base ``pose_weight`` (``w_pose0``). The returned value is
        ``w_pose0 · clamp(_w_pose_frac · step_multiplier, bound_lo, bound_hi)``. At a matched ratio the
        step multiplier is 1.0 → the effective weight is the accumulated fraction (stable fixed point)."""
        m = equimarginal_pose_weight_multiplier(
            cot_seg_norm,
            cot_pose_norm,
            self.state,
            rho=self.rho,
            decay=self.decay,
            tol=self.tol,
            clamp_lo=self.step_clamp_lo,
            clamp_hi=self.step_clamp_hi,
        )
        new_frac = self._w_pose_frac * m
        new_frac = min(max(new_frac, self.bound_lo), self.bound_hi)
        self._w_pose_frac = float(new_frac)
        return float(w_pose_base) * self._w_pose_frac

    @property
    def w_pose_frac(self) -> float:
        return self._w_pose_frac

    @property
    def ratio_ema(self) -> float | None:
        return self.state.ratio_ema

    def telemetry(self, *, cot_seg_norm: float, cot_pose_norm: float, w_pose_eff: float) -> dict:
        """A compact observability row (#305 facet 1/2/4): the measured per-axis pulls + the controller
        state + the effective weight, all in score-pull units. Caller logs this per epoch."""
        return {
            "cot_seg_norm": float(cot_seg_norm),
            "cot_pose_norm": float(cot_pose_norm),
            "ratio_ema": (None if self.state.ratio_ema is None else float(self.state.ratio_ema)),
            "rho_target": float(self.rho),
            "w_pose_frac": float(self._w_pose_frac),
            "w_pose_effective": float(w_pose_eff),
            "steps": int(self.state.steps),
        }

    def state_dict(self) -> dict:
        return {"ratio_ema": self.state.ratio_ema, "steps": int(self.state.steps),
                "w_pose_frac": float(self._w_pose_frac)}

    def load_state_dict(self, d: dict | None) -> None:
        if not d:
            return
        self.state = EquimarginalPoseWeightState.from_dict(d)
        self._w_pose_frac = float(d.get("w_pose_frac", 1.0))
