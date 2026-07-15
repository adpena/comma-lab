# SPDX-License-Identifier: MIT
"""Single-owner V9 hosc endpoint law.

V6's ``10.0`` is a clock-replica pin and V7's ``3.177`` is the value frozen at
its event boundary.  Neither is the endpoint of the V9 step-native continuation.
For V9 the interface width of ``tanh(beta*sin(u))`` scales as ``beta**-1``.
Three dyadic continuation refinements therefore give the explicit homotopy

    beta: 1 -> 2 -> 4 -> 8.

The final ``8`` is a DERIVED schedule value, not an efficacy measurement.  A
byte-closed A/B still owns whether this continuation improves the contest score.
"""
from __future__ import annotations

import math

from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    InputRef,
    LawRef,
    resolve,
)

EQUATION_ID = "v9_hosc_beta_endpoint_v1"
V9_HOSC_BETA_START = 1.0
V9_HOSC_DYADIC_REFINEMENTS = 3


def v9_hosc_beta_endpoint(beta_start: float, dyadic_refinements: int) -> float:
    """Return ``beta_start * 2**dyadic_refinements`` with strict domains."""

    start = float(beta_start)
    refinements = int(dyadic_refinements)
    if not math.isfinite(start) or start <= 0.0:
        raise ValueError(f"beta_start must be finite and > 0, got {beta_start!r}")
    if refinements != dyadic_refinements or refinements < 1:
        raise ValueError(
            "dyadic_refinements must be a positive integer, "
            f"got {dyadic_refinements!r}"
        )
    return start * (2.0**refinements)


def v9_hosc_beta_endpoint_lawref() -> LawRef:
    """Return the only LawRef allowed to own V9's hosc endpoint."""

    return LawRef(
        equation_id=EQUATION_ID,
        inputs={
            "beta_start": InputRef.literal(
                V9_HOSC_BETA_START,
                "V9 smooth-side continuation start; gradients remain live",
            ),
            "dyadic_refinements": InputRef.literal(
                V9_HOSC_DYADIC_REFINEMENTS,
                "three interface-width halvings: beta 1 -> 2 -> 4 -> 8",
            ),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )


def resolve_v9_hosc_beta_endpoint(*, repo_root=None):
    """Resolve the V9 endpoint and return its machine-readable custody row."""

    resolved = resolve(v9_hosc_beta_endpoint_lawref(), repo_root=repo_root)
    expected = v9_hosc_beta_endpoint(
        V9_HOSC_BETA_START, V9_HOSC_DYADIC_REFINEMENTS
    )
    if float(resolved.value) != expected:
        raise ValueError(
            "V9 hosc endpoint LawRef diverged from its defining equation: "
            f"resolved={resolved.value!r}, expected={expected!r}"
        )
    return resolved


__all__ = [
    "EQUATION_ID",
    "V9_HOSC_BETA_START",
    "V9_HOSC_DYADIC_REFINEMENTS",
    "resolve_v9_hosc_beta_endpoint",
    "v9_hosc_beta_endpoint",
    "v9_hosc_beta_endpoint_lawref",
]
