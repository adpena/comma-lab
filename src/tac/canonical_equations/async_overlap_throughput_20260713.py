# SPDX-License-Identifier: MIT
"""Fail-closed throughput laws for overlapped verdict work and D-A probes.

These equations are MEANS-only.  They do not assign score authority to a wall
clock measurement and they deliberately distinguish three quantities that the
throughput corpus previously conflated:

* no cadence miss (the worker finished before the next scheduled verdict),
* no exposed async tail (``max(0, T_async - T_train_concurrent) == 0``), and
* no contention (``T_train_concurrent == T_train_solo``).

Only the first can be established from an async-only run log.  The third needs
a matched solo/concurrent A/B.  D-A's backward probe is also inclusive of the
forward needed by autodiff, so its incremental VJP is a difference, never a
second disjoint timer that may be summed with the forward.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

EQUATION_ID = "async_overlap_and_inclusive_vjp_throughput_v1"
ASYNC_WALL_LAW = "T_window=max(T_train_concurrent,T_async)"
CONTENTION_LAW = "delta_contention=T_train_concurrent/T_train_solo-1"
INCREMENTAL_VJP_LAW = "T_vjp_incremental=T_backward_inclusive-T_forward"
VERDICT_SCOPE = (
    "wall-clock MEANS accounting on a matched host/config only; no transfer to "
    "contest-CPU/CUDA, score, archive, fidelity, or promotion"
)
REQ_R = (
    "re-run a matched solo/concurrent A/B when host load, frozen-scorer build, "
    "thread binding, MLX build, batch geometry, or verdict cadence changes"
)


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class AsyncOverlapAccounting:
    """Receipt-ready output for one matched solo/concurrent timing window."""

    train_solo_s: float
    train_concurrent_s: float
    async_service_s: float
    current_window_s: float
    no_contention_counterfactual_window_s: float
    exposed_async_tail_s: float
    contention_penalty_fraction: float
    contention_free_speedup_x: float
    evidence_label: str = "DERIVED_FROM_MATCHED_MEASUREMENTS"
    score_claim: bool = False
    pointer_moved: bool = False
    verdict_scope: str = VERDICT_SCOPE
    req_r: str = REQ_R

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def account_async_overlap(
    *, train_solo_s: float, train_concurrent_s: float, async_service_s: float
) -> AsyncOverlapAccounting:
    """Compose a matched A/B without declaring hidden work free.

    ``train_concurrent_s`` must be measured while the async service is active;
    an async-only log cannot supply ``train_solo_s`` and therefore cannot call
    this function honestly.
    """

    solo = _finite_positive(train_solo_s, "train_solo_s")
    concurrent = _finite_positive(train_concurrent_s, "train_concurrent_s")
    service = _finite_nonnegative(async_service_s, "async_service_s")
    current = max(concurrent, service)
    counterfactual = max(solo, service)
    return AsyncOverlapAccounting(
        train_solo_s=solo,
        train_concurrent_s=concurrent,
        async_service_s=service,
        current_window_s=current,
        no_contention_counterfactual_window_s=counterfactual,
        exposed_async_tail_s=max(0.0, service - concurrent),
        contention_penalty_fraction=concurrent / solo - 1.0,
        contention_free_speedup_x=current / counterfactual,
    )


def derive_incremental_vjp_s(*, forward_s: float, backward_inclusive_s: float) -> float:
    """Derive VJP-only time from D-A's inclusive autodiff observation.

    A negative difference is not clipped to zero: it means order/noise/cache
    effects exceed the putative increment and the probe cannot resolve the
    split at that sample size.
    """

    forward = _finite_nonnegative(forward_s, "forward_s")
    inclusive = _finite_nonnegative(backward_inclusive_s, "backward_inclusive_s")
    if inclusive < forward:
        raise ValueError(
            "backward_inclusive_s is smaller than forward_s; incremental VJP is unresolved"
        )
    return inclusive - forward


def async_only_identifiability(*, cadence_miss_count: int) -> dict[str, object]:
    """State exactly what an async-only log identifies.

    Completion timestamps and worker durations can prove cadence misses, but
    without a solo arm they cannot identify CPU/memory contention.
    """

    if (
        isinstance(cadence_miss_count, bool)
        or not isinstance(cadence_miss_count, int)
        or cadence_miss_count < 0
    ):
        raise ValueError("cadence_miss_count must be a non-negative integer")
    misses = cadence_miss_count
    return {
        "cadence_miss_count": misses,
        "no_cadence_miss_measured": misses == 0,
        "contention_identified": False,
        "contention_penalty_fraction": None,
        "evidence_label": "MEASURED_CADENCE_EVENTS__CONTENTION_UNMEASURED",
        "verdict_scope": VERDICT_SCOPE,
        "req_r": REQ_R,
    }


__all__ = [
    "ASYNC_WALL_LAW",
    "CONTENTION_LAW",
    "EQUATION_ID",
    "INCREMENTAL_VJP_LAW",
    "REQ_R",
    "VERDICT_SCOPE",
    "AsyncOverlapAccounting",
    "account_async_overlap",
    "async_only_identifiability",
    "derive_incremental_vjp_s",
]
