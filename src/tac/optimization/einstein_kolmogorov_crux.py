"""Pure score arithmetic and bounded projected middle-point DSPSA transitions.

The recursive real iterate is projected into the configured box after every
update.  This is therefore a bounded projected middle-point DSPSA variant, not
an exact implementation of Wang's unprojected recursion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

SCORE_BYTES_NORMALIZER = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / SCORE_BYTES_NORMALIZER


def score(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    if d_seg < 0.0 or d_pose < 0.0 or archive_bytes < 0:
        raise ValueError("score inputs must be non-negative")
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + RATE_SCORE_PER_BYTE * archive_bytes


def score_delta(*, before: tuple[float, float, int], after: tuple[float, float, int]) -> float:
    return score(d_seg=after[0], d_pose=after[1], archive_bytes=after[2]) - score(
        d_seg=before[0], d_pose=before[1], archive_bytes=before[2]
    )


def admit_candidate(
    *,
    before: tuple[float, float, int],
    after: tuple[float, float, int],
    before_mismatches: int | None = None,
    after_mismatches: int | None = None,
) -> bool:
    if before[2] == after[2] and before_mismatches is not None and after_mismatches is not None:
        return after_mismatches < before_mismatches
    return score_delta(before=before, after=after) < 0.0


def marginal_beats_waterline(*, non_rate_score_improvement: float, added_bytes: int) -> bool:
    if added_bytes <= 0:
        raise ValueError("added_bytes must be positive for a rate-bearing admission")
    return non_rate_score_improvement / added_bytes > RATE_SCORE_PER_BYTE


def project_uint8(values: Sequence[int | float], *, lower: int = 0, upper: int = 255) -> tuple[int, ...]:
    if not 0 <= lower <= upper <= 255:
        raise ValueError("projection bounds must be uint8")
    return tuple(max(lower, min(upper, round(value))) for value in values)


def coordinate_candidates(current: Sequence[int], *, lower: int = 0, upper: int = 255) -> tuple[tuple[int, ...], ...]:
    current = project_uint8(current, lower=lower, upper=upper)
    out: list[tuple[int, ...]] = []
    for index, value in enumerate(current):
        for delta in (-1, 1):
            candidate = list(current)
            candidate[index] = max(lower, min(upper, value + delta))
            if tuple(candidate) != current:
                out.append(tuple(candidate))
    return tuple(out)


def dspsa_perturbation(*, seed: int, iteration: int, dimension: int) -> tuple[int, ...]:
    if seed < 0 or iteration < 0 or dimension <= 0:
        raise ValueError("invalid DSPSA perturbation domain")
    stream = hashlib.shake_256(f"{seed}:{iteration}:{dimension}".encode("ascii")).digest(dimension)
    return tuple(1 if byte & 1 else -1 for byte in stream)


def middle_point(theta: Sequence[float], *, lower: int = 0, upper: int = 255) -> tuple[float, ...]:
    """Return the projected half-integer centre used by this bounded variant."""
    if not 0 <= lower < upper <= 255:
        raise ValueError("invalid projected middle-point bounds")
    return tuple(float(math.floor(max(lower + 0.5, min(upper - 0.5, value))) + 0.5) for value in theta)


def project_theta(theta: Sequence[float], *, lower: int = 0, upper: int = 255) -> tuple[float, ...]:
    """Project the real DSPSA iterate; the middle point is only for evaluation."""
    if not 0 <= lower < upper <= 255:
        raise ValueError("invalid real-theta bounds")
    return tuple(float(max(lower + 0.5, min(upper - 0.5, value))) for value in theta)


def wang_corners(
    theta: Sequence[float], perturbation: Sequence[int], *, lower: int = 0, upper: int = 255
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return opposite corners around the bounded projected middle point.

    The historical function name is retained for API compatibility.  Projection
    of the recursive iterate means the surrounding algorithm is not claimed to
    be exact Wang DSPSA.
    """
    if len(theta) != len(perturbation) or any(sign not in {-1, 1} for sign in perturbation):
        raise ValueError("invalid middle-point perturbation")
    center = middle_point(theta, lower=lower, upper=upper)
    plus = tuple(round(value + sign / 2.0) for value, sign in zip(center, perturbation, strict=True))
    minus = tuple(round(value - sign / 2.0) for value, sign in zip(center, perturbation, strict=True))
    if any(abs(left - right) != 1 for left, right in zip(plus, minus, strict=True)):
        raise AssertionError("middle-point corners must differ by exactly one grid unit")
    return plus, minus


@dataclass(frozen=True)
class DSPSAState:
    """Immutable state for bounded projected middle-point DSPSA."""

    theta: tuple[float, ...]
    best: tuple[int, ...]
    best_objective: float
    iteration: int
    seed: int
    config_fingerprint: str
    calibrated_a: float | None = None
    last_objective_plus: float | None = None
    last_objective_minus: float | None = None

    def __post_init__(self) -> None:
        if not self.theta or len(self.theta) != len(self.best):
            raise ValueError("DSPSA theta and best vectors must have the same non-zero dimension")
        if any(not math.isfinite(value) or value < 0.5 or value > 254.5 for value in self.theta):
            raise ValueError("DSPSA theta must stay within the uint8 middle-point domain")
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255 for value in self.best):
            raise ValueError("DSPSA best vector must be uint8 integers")
        if self.iteration < 0 or self.seed < 0 or not self.config_fingerprint:
            raise ValueError("invalid DSPSA state metadata")
        if not math.isfinite(self.best_objective):
            raise ValueError("DSPSA best objective must be finite")
        if self.calibrated_a is not None and (not math.isfinite(self.calibrated_a) or self.calibrated_a <= 0.0):
            raise ValueError("calibrated_a must be positive when present")
        for value in (self.last_objective_plus, self.last_objective_minus):
            if value is not None and not math.isfinite(value):
                raise ValueError("DSPSA corner objective evidence must be finite")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["theta"] = list(self.theta)
        payload["best"] = list(self.best)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        config_fingerprint: str,
        lower: int = 0,
        upper: int = 255,
    ) -> DSPSAState:
        payload: dict[str, Any] = json.loads(raw)
        if payload.get("config_fingerprint") != config_fingerprint:
            raise ValueError("resume fingerprint mismatch")
        payload["theta"] = tuple(payload["theta"])
        payload["best"] = tuple(payload["best"])
        state = cls(**payload)
        state.require_bounds(lower=lower, upper=upper)
        return state

    def require_bounds(self, *, lower: int, upper: int) -> None:
        """Refuse an incumbent or real iterate outside the configured box."""
        if not 0 <= lower < upper <= 255:
            raise ValueError("invalid DSPSA state bounds")
        theta_lower, theta_upper = lower + 0.5, upper - 0.5
        if any(value < theta_lower or value > theta_upper for value in self.theta):
            raise ValueError("DSPSA theta violates configured bounds")
        if any(value < lower or value > upper for value in self.best):
            raise ValueError("DSPSA incumbent violates configured bounds")


def wang_dspsa_step(
    state: DSPSAState,
    *,
    objective_plus: float,
    objective_minus: float,
    target_first_displacement: float,
    gain_alpha: float,
    A: int,
    lower: int = 0,
    upper: int = 255,
) -> DSPSAState:
    """Advance bounded projected middle-point DSPSA by two measured corners.

    The recursive ``theta`` update is projected into ``[lower+0.5,
    upper-0.5]``.  This projected recursion intentionally does not claim exact
    Wang semantics.
    """
    if target_first_displacement <= 0.0 or not 0.5 < gain_alpha <= 1.0 or A < 1:
        raise ValueError("invalid DSPSA gain schedule")
    state.require_bounds(lower=lower, upper=upper)
    signs = dspsa_perturbation(seed=state.seed, iteration=state.iteration, dimension=len(state.theta))
    plus, minus = wang_corners(state.theta, signs, lower=lower, upper=upper)
    gradient = tuple((objective_plus - objective_minus) * sign for sign in signs)
    nonzero = sorted(abs(value) for value in gradient if value != 0.0)
    calibrated_a = state.calibrated_a
    if calibrated_a is None and nonzero:
        median = nonzero[len(nonzero) // 2]
        calibrated_a = target_first_displacement * ((A + state.iteration + 1) ** gain_alpha) / median
    if calibrated_a is None:
        theta = project_theta(state.theta, lower=lower, upper=upper)
    else:
        gain = calibrated_a / ((A + state.iteration + 1) ** gain_alpha)
        theta = project_theta(
            [value - gain * grad for value, grad in zip(state.theta, gradient, strict=True)], lower=lower, upper=upper
        )
    best, best_objective = state.best, state.best_objective
    for corner, objective in ((plus, objective_plus), (minus, objective_minus)):
        if objective < best_objective:
            best, best_objective = corner, objective
    next_state = DSPSAState(
        theta=theta,
        best=best,
        best_objective=best_objective,
        iteration=state.iteration + 1,
        seed=state.seed,
        config_fingerprint=state.config_fingerprint,
        calibrated_a=calibrated_a,
        last_objective_plus=objective_plus,
        last_objective_minus=objective_minus,
    )
    next_state.require_bounds(lower=lower, upper=upper)
    return next_state
