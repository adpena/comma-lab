# SPDX-License-Identifier: MIT
"""The A/B DECOUPLING SCREEN — matched-compute CONTROL arm spec + kill-criterion evaluator.

increment-1a items 3 + 4 (SYNTHESIS_v2_v8 §A.4 / §B / F1). 1a is an A/B: the decoupled
per-class-field arm vs a MATCHED-COMPUTE shared-head CONTROL arm, both measured PAINT-FREE
the SAME way (composite/head argmax → MASK d_seg vs L\\*). The control arm IS the MEASURED
baseline (measured IN-RUN, never borrowed from run-1's 0.312 pre-actuation birth arm).

**Pre-registered KILL (F1):** the decoupled arm's mask d_seg must beat the control's by
> ``delta_mask``. Both arms paint-free ⇒ the flat-paint 0.0064 confound is EXCLUDED by
construction. ``delta_mask`` defaults to the ``delta_R`` through-R proxy (0.0196) until the
mask-level noise floor is MEASURED (recess R7); the verdict carries that provenance so a
proxy floor is never silently promoted to a measured one.

NO-FAKE / n600: the evaluator REFUSES to emit a verdict if EITHER arm is missing, is a toy
(``n_frames != 600``), or was measured at a different frame count — a partition that cannot
be measured at n600 cannot falsify anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tac.inc1a_harness.mask_dseg_meter import MaskDsegResult

# The through-R uint8 noise floor (reports/delta_R_noise_floor.json:delta_R, p95 over the
# annulus). Used as a CONSERVATIVE proxy for delta_mask until R7 measures the mask-level
# floor. NOT a measured mask-level floor — carried with that caveat welded on.
DELTA_R_PROXY = 0.019590163230895963

# Verdict labels (the three pre-registered outcomes; item 4).
VERDICT_CONFIRMED = "DECOUPLING-CONFIRMED"
VERDICT_KILLED = "KILLED-at-delta_mask"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE-below-floor"
VERDICT_REFUSED = "REFUSED-arm-missing-or-toy"


class DecouplingScreenError(ValueError):
    """Raised on malformed A/B inputs (NO silent verdict on a toy)."""


# ---------------------------------------------------------------------------
# item 3 — the matched-compute CONTROL-arm spec (the DSL authoring path P7 compiles)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ControlArmSpec:
    """Spec for the matched-compute shared-head CONTROL arm of the 1a A/B.

    The control is the SAME trunk as the decoupled arm but with ONE shared 5-class head
    (not per-class decoupled fields), sized so its total param count matches the decoupled
    arm's within ``param_tolerance``, trained on the SAME seed/epoch/curriculum budget. This
    makes the A/B a fair test of the DECOUPLING (∂φ_c/∂θ_{c'}=0) rather than of capacity or
    compute. This spec is the AUTHORING path; P7 compiles the exact typed ``WitnessProgram``
    + DSL ``Lever`` factory against the live architecture (this harness never edits the DSL).
    """

    decoupled_param_count: int  # MEASURED trunk+heads param count of the decoupled arm
    param_tolerance: float = 0.05  # |P_ctrl - P_dec| / P_dec must be <= this (matching rule)
    seed: int = 0  # SAME seed as the decoupled arm
    epochs: int = 0  # SAME training budget as the decoupled arm (0 = inherit from arm cfg)
    curriculum: str = "match_decoupled"  # SAME curriculum schedule
    head: str = "shared_5class"  # ONE shared head (vs the decoupled per-class fields)
    matching_provenance: str = (
        "MATCHED-COMPUTE: shared-head control sized so total params match the decoupled "
        "arm within param_tolerance (adjust shared-head width); SAME seed/epochs/curriculum. "
        "The A/B isolates DECOUPLING, not capacity/compute. Actual sizing at P7 compile."
    )

    def target_param_window(self) -> tuple[int, int]:
        """The admissible [lo, hi] control param count that satisfies the matching rule."""
        p = int(self.decoupled_param_count)
        tol = float(self.param_tolerance)
        return (int(round(p * (1.0 - tol))), int(round(p * (1.0 + tol))))

    def params_match(self, control_param_count: int) -> bool:
        """Does an actual control param count satisfy the matching rule?"""
        if int(self.decoupled_param_count) <= 0:
            raise DecouplingScreenError("decoupled_param_count must be > 0 to check matching")
        rel = abs(int(control_param_count) - int(self.decoupled_param_count)) / float(
            self.decoupled_param_count
        )
        return rel <= float(self.param_tolerance)

    def to_config_dict(self) -> dict:
        """Emit the control-arm config the DSL authoring path consumes (P7 compiles it)."""
        lo, hi = self.target_param_window()
        return {
            "arm": "control_shared_head",
            "head": self.head,
            "target_param_count": int(self.decoupled_param_count),
            "param_window": [lo, hi],
            "param_tolerance": float(self.param_tolerance),
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "curriculum": self.curriculum,
            "measure": "composite_head argmax MASK d_seg vs L* (gt_n600.npz)",
            "paint_free": True,
            "matching_provenance": self.matching_provenance,
            "geometric_home": "N/A (control = a single shared head, no per-class decomposition)",
        }


def matched_control_spec(
    decoupled_param_count: int, *, param_tolerance: float = 0.05, seed: int = 0, epochs: int = 0,
) -> ControlArmSpec:
    """Build the matched-compute :class:`ControlArmSpec` (the fair-A/B baseline authoring path)."""
    if int(decoupled_param_count) <= 0:
        raise DecouplingScreenError("decoupled_param_count must be > 0")
    return ControlArmSpec(
        decoupled_param_count=int(decoupled_param_count),
        param_tolerance=float(param_tolerance),
        seed=int(seed),
        epochs=int(epochs),
    )


# ---------------------------------------------------------------------------
# item 4 — the kill-criterion evaluator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmResult:
    """One arm's measured mask d_seg, wrapped for the kill evaluator (name + toy flag)."""

    name: str  # "decoupled" | "control"
    n_frames: int
    agg_dseg: float
    per_class_dseg: dict[str, float] = field(default_factory=dict)
    is_toy: bool = False  # set True if the arm was a smoke/subset, never a verdict input

    @classmethod
    def from_meter(cls, name: str, res: MaskDsegResult, *, is_toy: bool | None = None) -> ArmResult:
        toy = (not res.is_n600) if is_toy is None else bool(is_toy)
        return cls(
            name=str(name),
            n_frames=int(res.n_frames),
            agg_dseg=float(res.agg_dseg),
            per_class_dseg=dict(res.per_class_dseg),
            is_toy=bool(toy),
        )


@dataclass
class KillVerdict:
    """The pre-registered A/B verdict."""

    verdict: str  # one of VERDICT_*
    passes_preregistered_gate: bool  # True ONLY for DECOUPLING-CONFIRMED (proceed to 1b)
    improvement: float  # control.agg_dseg - decoupled.agg_dseg (positive = decoupled better)
    delta_mask: float
    delta_mask_provenance: str
    decoupled_dseg: float = float("nan")
    control_dseg: float = float("nan")
    reason: str = ""


def evaluate_kill(
    decoupled: ArmResult,
    control: ArmResult,
    *,
    delta_mask: float = DELTA_R_PROXY,
    delta_mask_provenance: str = "delta_R through-R proxy (0.0196; NOT a measured mask floor — recess R7)",
) -> KillVerdict:
    """Emit the pre-registered A/B verdict {CONFIRMED / KILLED / INCONCLUSIVE / REFUSED}.

    ``improvement = control.agg_dseg - decoupled.agg_dseg`` (lower d_seg is better):
      * ``improvement > delta_mask``   -> **DECOUPLING-CONFIRMED** (beats control by > floor;
        the ONLY outcome that PASSES the pre-registered gate and proceeds to 1b).
      * ``improvement < -delta_mask``  -> **KILLED-at-delta_mask** (decoupled is WORSE than a
        shared head by > floor -> the decoupling FORMULATION is falsified, NOT the paradigm).
      * ``|improvement| <= delta_mask`` -> **INCONCLUSIVE-below-floor** (indistinguishable at
        the floor; underpowered -> measure the mask floor (R7) / more data; NOT a clean kill).

    REFUSES (NO-FAKE / n600): if either arm is missing, ``is_toy``, or not measured at 600
    frames, no verdict is emitted (``REFUSED-arm-missing-or-toy``) — a partition that cannot
    be measured at n600 cannot falsify anything.
    """

    for arm in (decoupled, control):
        if arm is None:
            return KillVerdict(
                verdict=VERDICT_REFUSED, passes_preregistered_gate=False,
                improvement=float("nan"), delta_mask=float(delta_mask),
                delta_mask_provenance=delta_mask_provenance,
                reason="an arm is missing",
            )
    problems = []
    for arm in (decoupled, control):
        if arm.is_toy:
            problems.append(f"{arm.name} is a toy (subset/smoke), never a verdict input")
        if int(arm.n_frames) != 600:
            problems.append(f"{arm.name} measured at N={arm.n_frames} != 600 (n600 discipline)")
    if problems:
        return KillVerdict(
            verdict=VERDICT_REFUSED, passes_preregistered_gate=False,
            improvement=float("nan"), delta_mask=float(delta_mask),
            delta_mask_provenance=delta_mask_provenance,
            decoupled_dseg=float(decoupled.agg_dseg), control_dseg=float(control.agg_dseg),
            reason="; ".join(problems),
        )

    improvement = float(control.agg_dseg) - float(decoupled.agg_dseg)
    dm = float(delta_mask)
    if improvement > dm:
        v, gate, reason = (
            VERDICT_CONFIRMED, True,
            f"decoupled beats control by {improvement:.6f} > delta_mask {dm:.6f}",
        )
    elif improvement < -dm:
        v, gate, reason = (
            VERDICT_KILLED, False,
            f"decoupled WORSE than control by {-improvement:.6f} > delta_mask {dm:.6f} "
            f"-> decoupling FORMULATION falsified (verdict_scope: FORMULATION, NOT paradigm)",
        )
    else:
        v, gate, reason = (
            VERDICT_INCONCLUSIVE, False,
            f"|improvement| {abs(improvement):.6f} <= delta_mask {dm:.6f} -> "
            f"indistinguishable at the floor; measure the mask floor (R7) / more data",
        )
    return KillVerdict(
        verdict=v, passes_preregistered_gate=gate, improvement=improvement,
        delta_mask=dm, delta_mask_provenance=delta_mask_provenance,
        decoupled_dseg=float(decoupled.agg_dseg), control_dseg=float(control.agg_dseg),
        reason=reason,
    )
