"""ddm_ds1 — the cheap-to-shrink objective for the WD3 score-aware renderer.

The campaign's live objective minimizes distortion at ONE quantization allocation.
This module adds the second-order property: distortion that stays low as the
allocator removes bytes. It is the objective form of `dD/dB`.

WHAT THIS IS
------------
An explicit weighted sum of the SAME score-aware loss `D`, evaluated at an ordered
set of progressively cheaper allocations::

    L = w_0 * D(a_0) + w_1 * D(a_1) + ... + w_k * D(a_k)

`a_0` is the shipped allocation; `a_1..a_k` are cheaper rungs derived by re-running
the trainer's OWN discrete waterfill at looser predicted-error ceilings. Reference
form: Matryoshka Representation Learning (Kusupati et al. 2022) supplies the
explicit-weighted-sum-over-a-declared-ordered-set; Universally Slimmable Networks
(Yu & Huang 2019) supplies the sandwich rule (widest + narrowest + n sampled) and
FjORD (Horvath et al. 2021) the sampled-rung estimator.

DELTAS DECLARED (per the CHARTER-TIME OPTIMAL-FORM LAW)
-------------------------------------------------------
SCOPE (legal, no mechanism loss):
  * `mode="sampled"` evaluates ONE rung per step instead of all k+1. The objective
    is IDENTICAL IN EXPECTATION and every evaluation is exact-in-D; only the
    gradient's variance changes. This is FjORD's estimator, not a proxy.
  * `k` is small (1-2 rungs) in the smoke.

MECHANISM: none. `D` is the real score-aware loss through the real frozen scorers,
and the perturbation is the trainer's real packet quantizer. NOT a surrogate.

WHY NOT THE CHEAPER TERMS (derivation, see the memo for the full race):
  * Penalizing the allocator's own `total_error()` proxy directly is GOODHART: that
    proxy is `sum((w - q(w)) * g)**2`, which is FIRST-ORDER in the training gradient
    `g` and therefore collapses toward 0 as `g -> 0` at convergence, for a sharp
    minimum exactly as much as for a flat one. Optimizing it teaches the model to
    report a low predicted error, not to survive quantization.
  * Sharpness-aware minimization's `eps* = rho * grad/||grad||` is a LINEARIZED,
    isotropic stand-in for a perturbation we can apply EXACTLY. Substituting it here
    would be a mechanism reduction.
  * Hessian-aware sensitivity (HAWQ) is the correct THEORY for why the first-order
    proxy is blind, but an approximation of a quantity we can evaluate exactly.

NOT NESTED — a precision point. The WD3 quantizer re-derives a per-group fp16 scale
at every bit depth (`quantize_tensor_groups`), so the b and b-1 grids are NOT nested.
This is an ORDERED set of allocations, not a nested prefix set; we borrow Matryoshka's
FORM, not its nesting property. (The PR130 HPAC `bit_depth` clip intervals DO nest,
but that object is losslessly coded and therefore carries no distortion at all.)

SWITCHABLE NORMALIZATION IS NOT NEEDED. Slimmable networks require switchable
BatchNorm because per-configuration activation statistics diverge. The WD3 student
uses `nn.GroupNorm` (`ddm_wd2_student_receiver.py:89,108,126`), which holds NO running
statistics and normalizes per sample. There is nothing to switch.

RESUMABILITY (P0). The sampler is STATELESS BY CONSTRUCTION: the rung for a step is a
pure deterministic function of `(seed, step)`. No new state is registered, so an
existing checkpoint schema cannot silently restart it.

DEFAULT-OFF. `DEFAULT_CONFIG` is inert. When inert, `apply` returns the caller's own
loss object unchanged -- not `loss + 0.0` -- so the autograd graph and the emitted
bytes are identical.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

__all__ = [
    "DEFAULT_CONFIG",
    "CheapToShrinkConfig",
    "DS1Error",
    "RungLadder",
    "apply",
    "derive_rung_ladder",
    "is_inert",
    "rungs_for_step",
    "select_rung_for_step",
]

TensorT = TypeVar("TensorT")

# Reference-form default: Matryoshka/slimmable both weight every configuration
# equally. Any other weighting is a departure and must be declared.
REFERENCE_UNIFORM_WEIGHT = 1.0

# The allocator's ceiling knob is the ONLY budget dial the apparatus exposes; a
# looser ceiling yields a cheaper allocation. These are the declared rung offsets.
#
#   "off"      inert; provably cannot change a byte.
#   "sandwich" base + exactly ONE cheapest rung -- the two-end reference form
#              (Yu & Huang's widest+narrowest). 1 extra scorer pass.
#   "all"      base + EVERY declared rung, full deterministic weighted sum
#              (Matryoshka's literal form). k extra scorer passes.
#   "sampled"  base + ONE rung sampled per step and rescaled; unbiased for "all"
#              (FjORD's estimator). 1 extra scorer pass regardless of k.
#
# "sandwich" is pinned to exactly one cheaper rung on purpose: with k>1 a
# two-end schedule would never evaluate the middle rungs, so their declared
# weights would sit unused -- a silently orphaned lever. Use "all" or "sampled".
ADMITTED_MODES = ("off", "sandwich", "all", "sampled")
SINGLE_RUNG_MODES = ("sandwich", "sampled")


class DS1Error(RuntimeError):
    """Fail-closed error for the cheap-to-shrink objective."""


class Waterfill(Protocol):
    """The trainer's existing discrete waterfill, injected rather than imported."""

    def __call__(self, *, maximum_predicted_error: float) -> Any: ...


@dataclass(frozen=True)
class CheapToShrinkConfig:
    """Declared shape of the cheap-to-shrink term. Frozen and validated."""

    mode: str = "off"
    # Multipliers on the trainer's own `maximum_predicted_error` ceiling. Each
    # entry > 1.0 loosens the ceiling and so buys a strictly cheaper allocation.
    ceiling_multipliers: tuple[float, ...] = ()
    # Relative weight of each cheaper rung. Empty => reference uniform weighting.
    rung_weights: tuple[float, ...] = ()
    # Weight of the shipped rung a_0. Reference form keeps it at 1.0.
    base_weight: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ADMITTED_MODES:
            raise DS1Error(f"mode must be one of {ADMITTED_MODES}, got {self.mode!r}")
        if self.base_weight < 0.0:
            raise DS1Error("base_weight is negative")
        if self.mode == "off":
            if self.ceiling_multipliers or self.rung_weights:
                raise DS1Error("mode='off' must declare no rungs and no weights")
            return
        if not self.ceiling_multipliers:
            raise DS1Error(f"mode={self.mode!r} requires at least one ceiling multiplier")
        if any(not (multiplier > 1.0) for multiplier in self.ceiling_multipliers):
            raise DS1Error("every ceiling multiplier must be > 1.0 (a looser ceiling is a cheaper rung)")
        if list(self.ceiling_multipliers) != sorted(self.ceiling_multipliers):
            raise DS1Error("ceiling multipliers must be ordered cheapest-last")
        if len(set(self.ceiling_multipliers)) != len(self.ceiling_multipliers):
            raise DS1Error("ceiling multipliers must be distinct")
        if self.rung_weights and len(self.rung_weights) != len(self.ceiling_multipliers):
            raise DS1Error("rung_weights length differs from ceiling_multipliers length")
        if any(weight < 0.0 for weight in self.rung_weights):
            raise DS1Error("rung weights must be non-negative")
        if self.base_weight == 0.0:
            raise DS1Error(
                "base_weight=0 abandons the SHIPPED allocation and trains only the cheap rungs; "
                "that is not a cheap-to-shrink objective. Use mode='off' to disable the lever."
            )
        if self.mode == "sandwich" and len(self.ceiling_multipliers) != 1:
            raise DS1Error(
                "mode='sandwich' is the two-end form and admits exactly one cheaper rung; "
                "declaring more would leave the middle rungs untrained. Use 'all' or 'sampled'."
            )

    @property
    def weights(self) -> tuple[float, ...]:
        """Per-rung weights, defaulting to the reference uniform weighting."""

        if self.rung_weights:
            return self.rung_weights
        return tuple(REFERENCE_UNIFORM_WEIGHT for _ in self.ceiling_multipliers)

    def provenance(self) -> dict[str, Any]:
        """Machine-readable declaration for the run receipt."""

        return {
            "lever": "ddm_ds1_cheap_to_shrink",
            "mode": self.mode,
            "ceiling_multipliers": list(self.ceiling_multipliers),
            "rung_weights": list(self.weights),
            "base_weight": self.base_weight,
            "seed": self.seed,
            "inert": is_inert(self),
            "reference_form": "matryoshka_weighted_sum_over_ordered_configuration_set",
            "reference_citations": [
                "Kusupati et al. 2022 Matryoshka Representation Learning (weighted sum over declared set)",
                "Yu & Huang 2019 Universally Slimmable Networks (sandwich rule)",
                "Horvath et al. 2021 FjORD (sampled-configuration estimator)",
            ],
            "distortion_surface": "wd3_score_native_objective_through_frozen_scorers",
            "perturbation_operator": "wd3_receiver_packet_quantizer_exact_not_surrogate",
        }


DEFAULT_CONFIG = CheapToShrinkConfig()


def is_inert(config: CheapToShrinkConfig) -> bool:
    """True when the lever provably cannot change a single emitted byte."""

    if config.mode == "off":
        return True
    if not config.ceiling_multipliers:
        return True
    return all(weight == 0.0 for weight in config.weights)


@dataclass(frozen=True)
class RungLadder:
    """The ordered set of allocations the objective is summed over."""

    base_allocation: Any
    cheaper_allocations: tuple[Any, ...] = ()
    ceilings: tuple[float, ...] = ()
    base_ceiling: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return 1 + len(self.cheaper_allocations)


def derive_rung_ladder(
    *,
    base_allocation: Any,
    base_ceiling: float,
    waterfill: Waterfill,
    config: CheapToShrinkConfig,
    byte_cost: Callable[[Any], int] | None = None,
) -> RungLadder:
    """Build the ordered rung set by re-running the trainer's OWN waterfill.

    No new allocation policy is invented: each cheaper rung is what the existing
    discrete waterfill returns when its predicted-error ceiling is loosened.
    """

    if base_ceiling <= 0.0:
        raise DS1Error("base predicted-error ceiling must be positive")
    if is_inert(config):
        return RungLadder(base_allocation=base_allocation, base_ceiling=base_ceiling)

    ceilings: list[float] = []
    allocations: list[Any] = []
    for multiplier in config.ceiling_multipliers:
        ceiling = base_ceiling * multiplier
        allocations.append(waterfill(maximum_predicted_error=ceiling))
        ceilings.append(ceiling)

    # Whether the cheaper-rung check actually ran must be VISIBLE in the receipt.
    # A silently skipped check is indistinguishable from a passed one.
    diagnostics: dict[str, Any] = {
        "ceilings": list(ceilings),
        "byte_cost_checked": byte_cost is not None,
    }
    if byte_cost is not None:
        base_bytes = int(byte_cost(base_allocation))
        rung_bytes = [int(byte_cost(allocation)) for allocation in allocations]
        diagnostics["base_bytes"] = base_bytes
        diagnostics["rung_bytes"] = rung_bytes
        diagnostics["rung_byte_savings"] = [base_bytes - value for value in rung_bytes]
        # A rung that saves nothing cannot teach the model anything about shrinking.
        if any(saving <= 0 for saving in diagnostics["rung_byte_savings"]):
            raise DS1Error(
                "a declared rung is not cheaper than the shipped allocation; "
                f"base={base_bytes} rungs={rung_bytes}"
            )

    return RungLadder(
        base_allocation=base_allocation,
        cheaper_allocations=tuple(allocations),
        ceilings=tuple(ceilings),
        base_ceiling=base_ceiling,
        diagnostics=diagnostics,
    )


def rungs_for_step(config: CheapToShrinkConfig, step: int, rung_count: int) -> tuple[int, ...]:
    """Every cheaper rung the caller must evaluate at `step`. The general API.

    Stateless by construction: a pure function of (seed, step). Nothing to
    checkpoint, so resume cannot silently restart the schedule.
    """

    if rung_count <= 0 or is_inert(config):
        return ()
    if config.mode == "all":
        return tuple(range(rung_count))
    selected = select_rung_for_step(config, step, rung_count)
    return () if selected is None else (selected,)


def select_rung_for_step(config: CheapToShrinkConfig, step: int, rung_count: int) -> int | None:
    """The single cheaper rung to evaluate at `step`, for the single-rung modes.

    Refuses mode='all', which has no single selection -- call `rungs_for_step`.
    """

    if rung_count <= 0 or is_inert(config):
        return None
    if config.mode == "sandwich":
        # The "narrowest" end of the sandwich; the config pins rung_count to 1.
        return rung_count - 1
    if config.mode == "sampled":
        if step < 0:
            raise DS1Error("step must be non-negative")
        digest = hashlib.sha256(f"{config.seed}:{step}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % rung_count
    raise DS1Error(f"mode={config.mode!r} has no single rung; call rungs_for_step instead")


def apply(
    *,
    base_loss: TensorT,
    rung_losses: Sequence[tuple[int, TensorT]],
    config: CheapToShrinkConfig,
) -> tuple[TensorT, dict[str, Any]]:
    """Compose the cheap-to-shrink total.

    `rung_losses` are `(rung_index, D_at_that_rung)` pairs already evaluated by the
    caller through the real scorer graph. Returns `(total, telemetry)`.

    When inert, returns the caller's own `base_loss` OBJECT unchanged so that the
    autograd graph and every emitted byte are identical.
    """

    if is_inert(config):
        if rung_losses:
            raise DS1Error("inert config must not be given rung losses")
        return base_loss, {"ds1_active": False, "ds1_rungs_evaluated": 0}

    weights = config.weights
    total = base_loss * config.base_weight if config.base_weight != 1.0 else base_loss
    telemetry: dict[str, Any] = {
        "ds1_active": True,
        "ds1_rungs_evaluated": len(rung_losses),
        "ds1_rung_index": [index for index, _ in rung_losses],
    }
    if config.mode == "sampled" and len(rung_losses) > 1:
        raise DS1Error("sampled mode evaluates exactly one rung per step")

    contributions: list[float] = []
    for index, rung_loss in rung_losses:
        if not 0 <= index < len(weights):
            raise DS1Error(f"rung index {index} is outside the declared ladder")
        weight = weights[index]
        # The sampled estimator is unbiased for the full sum only if each sampled
        # rung is rescaled by the number of rungs it stands in for.
        scale = weight * (len(weights) if config.mode == "sampled" else 1.0)
        total = total + rung_loss * scale
        contributions.append(scale)
    telemetry["ds1_rung_scale"] = contributions
    return total, telemetry
