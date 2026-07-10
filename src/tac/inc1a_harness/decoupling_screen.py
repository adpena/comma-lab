# SPDX-License-Identifier: MIT
"""The A/B DECOUPLING SCREEN — matched-compute CONTROL arm spec + kill-criterion evaluator.

increment-1a items 3 + 4 (SYNTHESIS_v2_v8 §A.4 / §B / F1). 1a is an A/B: the decoupled
per-class-field arm vs a MATCHED-COMPUTE shared-head CONTROL arm, both measured PAINT-FREE
the SAME way (composite/head argmax → MASK d_seg vs L\\*). The control arm IS the MEASURED
baseline (measured IN-RUN, never borrowed from run-1's 0.312 pre-actuation birth arm).

**Pre-registered KILL (F1):** the decoupled arm's mask d_seg must beat the control's by
> ``delta_mask``. Both arms paint-free ⇒ the flat-paint 0.0064 confound is EXCLUDED by
construction. The OPERATIVE ``delta_mask`` = ``max(frame-sampling floor, in-run seed
spread)`` (F-P5-P9-1 + F-P5-2): the frame-sampling component is R7-MEASURED (3.46e-6, n600);
the DOMINANT component is training SEED variance, which is NOT $0-measurable and MUST be
measured in-run from >=3 control-arm seed replicates. When replicates exist, ``seed_spread``
is REQUIRED — a kill fired on a floor that dropped a measurable seed component would be
firing on within-seed noise (the eightfold-P2 failure). The verdict carries the floor's
provenance so a proxy floor is never silently promoted to a measured one.

**RETIRED (F-P5-P9-1 / P4_recess R7):** the old default was ``delta_R`` (0.0196), the
through-R uint8 floor borrowed as a ``delta_mask`` proxy — a ~5600x UNIT category error
(margin-perturbation magnitude at n96 through-R, NOT a mask-level d_seg rate at n600). At
0.0196 the screen was DECISION-INERT: no realistic decoupling improvement (0.001-0.01)
could clear it, so it could never fire CONFIRMED or KILLED. It is retired below and must
NOT be reintroduced as the kill floor.

NO-FAKE / n600: the evaluator REFUSES to emit a verdict if EITHER arm is missing, is a toy
(``n_frames != 600``), was measured at a different frame count, OR the seed component is
measurable (>=2 replicates) but ``seed_spread`` was not supplied — a partition that cannot
be measured at n600, or a floor that dropped a measurable component, cannot falsify anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tac.inc1a_harness.mask_dseg_meter import MaskDsegResult

# R7 MEASURED mask-level frame-sampling noise floor (P4_recess_20260709 R7: SEM of the
# 600-frame mean of the per-frame mask flip fraction under a small margin perturbation, n600;
# bootstrap-agreeing). value-provenance: MEASURED-ANCHOR (this gt_n600 cache; INSTANCE-scope).
# This is the frame-sampling LOWER BOUND on the 1a kill margin; the OPERATIVE floor ADDS the
# in-run SEED-variance component via ``operative_delta_mask`` (P2 seed honesty).
DELTA_MASK_FRAME_SAMPLING_FLOOR = 3.46e-6

# RETIRED (F-P5-P9-1): the delta_R through-R uint8 floor, formerly the delta_mask DEFAULT.
# Kept ONLY as a documented historical reference (a ~5600x UNIT category error vs the mask
# floor above). NEVER a default; NEVER the kill floor. Importing this to seed a kill is a bug.
DELTA_R_PROXY_RETIRED = 0.019590163230895963

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
# the operative delta_mask floor (F-P5-P9-1 + F-P5-2): max(frame-sampling, seed spread)
# ---------------------------------------------------------------------------
def operative_delta_mask(
    *,
    seed_spread: float | None = None,
    n_seed_replicates: int = 1,
    frame_sampling_floor: float = DELTA_MASK_FRAME_SAMPLING_FLOOR,
) -> float:
    """The operative 1a kill floor = ``max(frame-sampling floor, in-run seed spread)``.

    P2 SEED-VARIANCE honesty (F-P5-2): the DOMINANT floor component is the training-seed
    variance of the trained decoupled/control fields, which is NOT $0-measurable and MUST be
    measured in-run from the control arm's OWN seed replicates. When ``n_seed_replicates >= 2``
    the seed component is measurable, so ``seed_spread`` is REQUIRED — a kill fired on a floor
    that dropped a measurable seed component would be firing on within-seed noise (the
    eightfold-P2 failure). With a single seed the seed component is UNMEASURED and the floor
    is the R7 frame-sampling LOWER BOUND only (INSTANCE-level; a Δ near it is not a clean
    verdict). Raises :class:`DecouplingScreenError` on the under-specified (replicates-without-
    spread) case and on negative inputs.
    """
    if float(frame_sampling_floor) < 0.0:
        raise DecouplingScreenError("frame_sampling_floor must be >= 0")
    if int(n_seed_replicates) >= 2 and seed_spread is None:
        raise DecouplingScreenError(
            "seed_spread REQUIRED once >=2 seed replicates exist (F-P5-2 / P2 seed honesty): "
            "the seed-variance floor component is measurable in-run and must not be dropped"
        )
    if seed_spread is None:
        return float(frame_sampling_floor)
    if float(seed_spread) < 0.0:
        raise DecouplingScreenError("seed_spread must be >= 0")
    return max(float(frame_sampling_floor), float(seed_spread))


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
    delta_mask: float | None = None,
    seed_spread: float | None = None,
    n_seed_replicates: int = 1,
    frame_sampling_floor: float = DELTA_MASK_FRAME_SAMPLING_FLOOR,
    delta_mask_provenance: str | None = None,
) -> KillVerdict:
    """Emit the pre-registered A/B verdict {CONFIRMED / KILLED / INCONCLUSIVE / REFUSED}.

    The operative kill floor (F-P5-P9-1 + F-P5-2) is ``max(frame_sampling_floor, seed_spread)``
    (see :func:`operative_delta_mask`), computed here UNLESS ``delta_mask`` is passed explicitly
    (an override, e.g. for a sensitivity sweep). ``improvement = control.agg_dseg -
    decoupled.agg_dseg`` (lower d_seg is better):
      * ``improvement > delta_mask``   -> **DECOUPLING-CONFIRMED** (beats control by > floor;
        the ONLY outcome that PASSES the pre-registered gate and proceeds to 1b).
      * ``improvement < -delta_mask``  -> **KILLED-at-delta_mask** (decoupled is WORSE than a
        shared head by > floor -> the decoupling FORMULATION is falsified, NOT the paradigm).
      * ``|improvement| <= delta_mask`` -> **INCONCLUSIVE-below-floor** (indistinguishable at
        the floor; underpowered -> measure the seed spread / more data; NOT a clean kill).

    REFUSES (NO-FAKE / n600): if either arm is missing, ``is_toy``, or not measured at 600
    frames — a partition that cannot be measured at n600 cannot falsify anything. ALSO refuses
    (F-P5-2) if the seed component is measurable (``n_seed_replicates >= 2``) but ``seed_spread``
    was not supplied — a floor that silently drops a measurable component cannot ground a kill.
    """

    # Compute the operative floor first (a pre-registration concern independent of the arms).
    if delta_mask is None:
        try:
            dm_value = operative_delta_mask(
                seed_spread=seed_spread,
                n_seed_replicates=int(n_seed_replicates),
                frame_sampling_floor=float(frame_sampling_floor),
            )
        except DecouplingScreenError as exc:
            return KillVerdict(
                verdict=VERDICT_REFUSED, passes_preregistered_gate=False,
                improvement=float("nan"), delta_mask=float("nan"),
                delta_mask_provenance=f"REFUSED: under-specified floor ({exc})",
                reason=f"delta_mask floor under-specified (F-P5-2 / P2 seed honesty): {exc}",
            )
        if delta_mask_provenance is None:
            if seed_spread is not None:
                delta_mask_provenance = (
                    f"operative = max(frame-sampling floor {float(frame_sampling_floor):.2e} "
                    f"[R7 MEASURED-ANCHOR], seed spread {float(seed_spread):.2e} "
                    f"[in-run, {int(n_seed_replicates)} replicates])"
                )
            else:
                delta_mask_provenance = (
                    f"frame-sampling LOWER BOUND {float(frame_sampling_floor):.2e} "
                    f"[R7 MEASURED-ANCHOR]; seed component UNMEASURED (single seed) "
                    f"-> INSTANCE-level floor"
                )
    else:
        dm_value = float(delta_mask)
        if delta_mask_provenance is None:
            delta_mask_provenance = "explicit delta_mask override (caller-supplied)"

    delta_mask = dm_value

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
