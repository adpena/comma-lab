# SPDX-License-Identifier: MIT
"""Cathedral consumer surfacing canonical framework-agnostic backend choice.

Per Catalog #335 canonical cathedral consumer contract + operator
NON-NEGOTIABLE META directive 2026-05-28: ``framework-agnostic primitives``
need a downstream cathedral consumer so the autopilot ranker can disambiguate
substrate candidates by their declared framework choice.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125 6-hook wire-in:
this consumer is the canonical surface that ranks candidates by their
backend tag (MLX-LOCAL non-promotable per Catalog #192/#317 vs PyTorch
contest-resolution promotable per Catalog #205 sister) so the autopilot
loop can correctly route per-candidate dispatch decisions.

Per Catalog #341 Tier A canonical-routing markers: this consumer is
observability-only — ``predicted_delta_adjustment=0.0`` /
``promotable=False`` / ``axis_tag="[predicted]"`` for every routing branch.
The canonical framework choice IS the disambiguator (hook #6
probe-disambiguator); the score signal flows through other consumers that
incorporate empirical anchors.

Per Catalog #357 dual-tier consumer architecture: this consumer is Tier A
(``TIER_A_OBSERVABILITY_ONLY``); it does NOT contribute to score-grade
ranking adjustments per the canonical contract.

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE (annotate candidates per backend)
  * #6 probe-disambiguator — ACTIVE PRIMARY (framework backend IS the
    canonical disambiguator between MLX-LOCAL non-promotable vs PyTorch
    contest-resolution promotable per Catalog #192/#246/#317)
  * #1, #2, #3, #5 — N/A (observability-only consumer; sister consumers
    cover sensitivity-map / Pareto / bit-allocator / continual-learning)

Cross-references:
  * Catalog #205 — sister at inflate-time device-selection surface
  * Catalog #335 — canonical Protocol contract this consumer satisfies
  * Catalog #341 — Tier A canonical routing markers
  * Catalog #357 — dual-tier consumer architecture (Tier A pinned)
  * Catalog #344 — canonical equations registry sister
    (framework_agnostic_backend_abstraction_compounding_v1 registered alongside)
  * tac.framework_agnostic — upstream canonical helper package
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import ConsumerTier, HookNumber


CONSUMER_NAME = "framework_agnostic_lookup_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY


# Canonical promotion-vs-non-promotion classification per Catalog #205 sister
# discipline + Catalog #192 (macOS-CPU advisory non-promotable) + Catalog
# #317 (one-arg local-MPS vs Modal dispatch switch).
_PROMOTABLE_BACKEND_VALUES: frozenset[str] = frozenset({"pytorch"})
_NON_PROMOTABLE_BACKEND_VALUES: frozenset[str] = frozenset({"mlx", "tinygrad"})
_DIAGNOSTIC_BACKEND_VALUES: frozenset[str] = frozenset({"numpy"})


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Reference implementation: NO-OP. The canonical framework choice does
    not require posterior refit per anchor; the auto-discovery loop's
    sister consumers (canonical_equations / canonical_anti_patterns) handle
    the cross-substrate posterior updates that THIS consumer surfaces as
    observability annotations.
    """
    _ = anchor  # explicit acknowledgment


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with framework backend disambiguation.

    Inspects the candidate's payload for backend-declarative tokens
    (``framework_backend`` / ``framework`` / ``backend`` keys, or
    ``trainer_path`` / ``recipe`` strings containing ``_mlx_local`` /
    ``_pytorch`` / ``_modal_t4`` / etc.) and surfaces the canonical
    promotion-vs-non-promotion classification per Catalog #205 sister.

    Returns ``predicted_delta_adjustment=0.0`` always per Catalog #341 Tier A
    routing markers — the framework choice does NOT directly adjust score
    predictions; it is the canonical disambiguator (hook #6) between
    promotable contest-grade candidates and non-promotable
    research-signal candidates.
    """
    backend_token = _infer_backend_token(candidate)
    routing_class = _classify_routing(backend_token)

    rationale = (
        f"framework_backend={backend_token or 'unknown'} → "
        f"routing_class={routing_class} per Catalog #205 sister discipline. "
        "Observability-only annotation per Catalog #341 Tier A."
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "framework_backend": backend_token,
        "routing_class": routing_class,
    }


def _infer_backend_token(candidate: Mapping[str, Any]) -> str | None:
    """Best-effort backend inference from candidate's declarative keys + paths.

    Inspection order:
      1. Explicit ``framework_backend`` / ``framework`` / ``backend`` key
      2. ``trainer_path`` / ``recipe`` string contains ``_mlx_local`` → mlx
      3. ``trainer_path`` / ``recipe`` string contains ``_pytorch`` / ``_modal_t4``
         / ``_a100`` / ``_4090`` / ``_h100`` / ``_l40s`` → pytorch (CUDA contest)
      4. None (unknown; caller's downstream consumer may disambiguate further)
    """
    for explicit_key in ("framework_backend", "framework", "backend"):
        if explicit_key in candidate:
            raw = str(candidate[explicit_key]).lower().strip()
            if raw in _PROMOTABLE_BACKEND_VALUES:
                return raw
            if raw in _NON_PROMOTABLE_BACKEND_VALUES:
                return raw
            if raw in _DIAGNOSTIC_BACKEND_VALUES:
                return raw
            if raw:
                return raw

    # Path-based inference per canonical V3 substrate naming convention.
    for path_key in ("trainer_path", "recipe", "lane_script", "lane_id"):
        if path_key not in candidate:
            continue
        path_str = str(candidate[path_key]).lower()
        if "_mlx_local" in path_str or "_mlx_" in path_str:
            return "mlx"
        if any(
            token in path_str
            for token in (
                "_modal_t4",
                "_modal_a100",
                "_modal_a10g",
                "_modal_l40s",
                "_modal_h100",
                "_vastai_4090",
                "_lightning_a100",
                "_pytorch",
            )
        ):
            return "pytorch"

    return None


def _classify_routing(backend_token: str | None) -> str:
    """Classify routing class per Catalog #205 sister discipline.

    Returns:
        ``"promotable_contest_resolution"`` — PyTorch CUDA per Catalog #205
        ``"non_promotable_research_signal"`` — MLX / tinygrad per Catalog #192/#317
        ``"diagnostic_reference"`` — numpy (canonical bridge oracle)
        ``"unknown"`` — backend token not inferable
    """
    if backend_token is None:
        return "unknown"
    if backend_token in _PROMOTABLE_BACKEND_VALUES:
        return "promotable_contest_resolution"
    if backend_token in _NON_PROMOTABLE_BACKEND_VALUES:
        return "non_promotable_research_signal"
    if backend_token in _DIAGNOSTIC_BACKEND_VALUES:
        return "diagnostic_reference"
    return "unknown"
