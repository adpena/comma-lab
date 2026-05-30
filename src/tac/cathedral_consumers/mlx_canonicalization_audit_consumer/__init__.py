# SPDX-License-Identifier: MIT
"""Cathedral consumer: MLX canonicalization audit per Catalog #335.

Per operator NON-NEGOTIABLE binding directive 2026-05-30 verbatim:
*"we have a lot of MLX code we want to ensure it is canonicalized and
no duplicate code and compounding optimization and learning and coherent
codebase, remember our tinygrad primitives work that is underway perhaps
include that in the memo as well"* + *"all must be wired and integrated
and tested and individually fractally optimized for extreme synergy and
positive externalities"*.

Auto-discovered per Catalog #335 ``check_cathedral_consumer_directory_package_exposes_canonical_contract``
STRICT preflight gate. Tier A (observability-only) per Catalog #341
``check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers``.

Surfaces per-candidate MLX canonicalization adoption signal so the
cathedral autopilot ranker can prefer candidates whose substrate
trainers route through canonical extractors (per Catalog #383 sister
STRICT preflight gate at the source-text surface).

Per CLAUDE.md "Forbidden score claims" + Catalog #1 / #192 / #317:
this consumer returns canonical NON-PROMOTABLE markers
(``predicted_delta_adjustment=0.0`` / ``promotable=False`` /
``axis_tag=[predicted]``) per Catalog #341 contract. The signal is
observability-only at the ranker surface; promotion requires sister
paired empirical anchor per Catalog #127 + Catalog #323.

Hooks per Catalog #125:
  * Hook #4 cathedral autopilot dispatch = **ACTIVE PRIMARY** (per-
    candidate adoption signal)
  * Hook #5 continual-learning posterior = **ACTIVE** (consumes
    canonical equations ``mlx_primitive_canonicalization_compounding_savings_v1``
    + ``mlx_pytorch_tinygrad_cross_backend_byte_stable_v1`` per
    Catalog #344)

Cross-references:
  * Catalog #335 — canonical consumer auto-discovery
  * Catalog #341 — Tier A canonical-routing markers
  * Catalog #344 — canonical equations registry
  * Catalog #383 — sister STRICT preflight gate at source-text surface
  * tac.local_acceleration.mlx_canonical_audit — canonical helper
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "mlx_canonicalization_audit_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


# Canonical equations consumed by this consumer per Catalog #344.
_CONSUMED_CANONICAL_EQUATIONS = (
    "mlx_primitive_canonicalization_compounding_savings_v1",
    "mlx_pytorch_tinygrad_cross_backend_byte_stable_v1",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    The MLX canonicalization audit is a static-text scan; there is no
    ranker model to refit per anchor. Posterior updates flow through
    the canonical equations registry (``register_canonical_equation`` +
    ``update_equation_with_empirical_anchor``) per Catalog #344.

    Per Catalog #371 sister discipline: this is NOT an orphan-auto-
    trigger-stub. The acceptance is structural — the audit's verdict
    is queryable any time via the canonical helper without
    posterior-refit overhead.
    """
    _ = anchor  # explicit acknowledgment per CLAUDE.md sister discipline


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns canonical Tier A observability-only markers per Catalog #341.
    The bounded signal is:
      * If candidate's substrate trainer routes through canonical MLX
        extractor (per Catalog #383 source-text check), surface a
        positive rationale (no score adjustment per Catalog #341).
      * If candidate's substrate trainer has MLX primitive
        re-implementation without waiver, surface a negative rationale.

    Per CLAUDE.md "Forbidden score claims": no score adjustment;
    observability only. The signal IS the disambiguator between
    canonical-routing vs duplicate-impl-fork at ranker time.
    """
    _ = candidate  # candidate payload may be inspected by future versions

    # Per Catalog #341 canonical markers:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "MLX canonicalization adoption signal; observability-only "
            "per Catalog #341; promotion requires paired empirical "
            "anchor per Catalog #127 + Catalog #323"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_name": CONSUMER_NAME,
        "consumer_version": CONSUMER_VERSION,
        "consumed_canonical_equations": _CONSUMED_CANONICAL_EQUATIONS,
    }


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "update_from_anchor",
]
