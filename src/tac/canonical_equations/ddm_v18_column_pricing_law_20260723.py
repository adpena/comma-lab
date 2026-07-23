# SPDX-License-Identifier: MIT
"""Canonical pricing law for DDM v18 generated correction columns.

For a restricted master with singleton measured objective coefficients ``c``,
exact coder bytes ``b``, conflict incidence ``A``, byte dual ``y_b <= 0``,
and conflict duals ``y_k <= 0``, the proposal reduced cost is

``r_j = c_j - b_j y_b - sum_k A[k,j] y_k``.

This law ranks/proposes columns only.  Non-additive set value is decided by
receiver-closed exact replay, never by ``sum(c_j)``.  The formulation falsifier
requires three complete rounds with no negative exact-priced column plus four
global exact-replay rows that do not beat the v12 control.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def ddm_column_reduced_cost(
    *,
    singleton_objective_delta: float,
    exact_coder_bytes: int,
    byte_dual_marginal: float,
    conflict_dual_marginals: Mapping[str, float],
    active_conflict_keys: tuple[str, ...],
) -> float:
    """Return ``c_j - a_j^T y`` for the bounded DDM restricted master."""

    values = (
        singleton_objective_delta,
        byte_dual_marginal,
        *conflict_dual_marginals.values(),
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pricing inputs must be finite")
    if (
        isinstance(exact_coder_bytes, bool)
        or not isinstance(exact_coder_bytes, int)
        or exact_coder_bytes <= 0
    ):
        raise ValueError("exact_coder_bytes must be positive")
    if len(set(active_conflict_keys)) != len(active_conflict_keys):
        raise ValueError("active_conflict_keys must be unique")
    if byte_dual_marginal > 0.0 or any(value > 0.0 for value in conflict_dual_marginals.values()):
        raise ValueError("<= constraint dual marginals must be nonpositive")
    penalty = exact_coder_bytes * float(byte_dual_marginal)
    penalty += sum(float(conflict_dual_marginals.get(key, 0.0)) for key in active_conflict_keys)
    return float(singleton_objective_delta) - penalty


__all__ = ["ddm_column_reduced_cost"]
