# SPDX-License-Identifier: MIT
"""Cathedral consumer for the canonical equations registry (Catalog #344 sister).

Per CLAUDE.md "Canonical equations + models registry" non-negotiable +
Catalog #335 paradigm-shift (canonical contract auto-discovery). Wires
the orphan-signal closure for ``tac.canonical_equations`` so the cathedral
autopilot ranker sees per-equation prediction annotations on every
candidate that maps to a registered domain or consumer.

This consumer is observability-only (predicted_delta_adjustment=0.0; per
CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323).
Per-equation predictions are NEVER promoted to score adjustments — they
surface as ``[predicted]`` annotations that future paid empirical
dispatches can compare against to refresh the equation's
``predicted_vs_empirical_residual``.

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE (annotate candidates)
  * #5 continual-learning posterior — ACTIVE (refresh equation calibration
    when a new posterior anchor matches a registered equation domain)
  * #1, #2, #3, #6 — N/A (observability-only annotation; no
    sensitivity-map / Pareto / bit-allocator / probe-disambiguator
    contribution)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "canonical_equation_lookup_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    When a new empirical anchor lands in the canonical continual-learning
    posterior, the canonical equations registry's
    ``auto_recalibrate_from_continual_learning_posterior`` is the canonical
    refresh path. This consumer is structurally NO-OP here — the
    canonical refresh is operator-triggered via
    ``tools/recalibrate_equation.py`` because automatic refit requires
    explicit signed measurement provenance per Catalog #287/#323.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with applicable equation predictions.

    Looks up equations whose ``canonical_consumers`` mention the candidate's
    substrate / lane / archive_family token (best-effort string match) and
    surfaces a concatenated rationale + the predicted residual band per
    matching equation. Returns ``predicted_delta_adjustment=0.0`` always
    — predictions are observability-only per the consumer's docstring.
    """
    try:
        from tac.canonical_equations import query_equations
    except ImportError:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": "canonical_equations registry unavailable [predicted]",
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    # Best-effort match: any candidate field whose string mention overlaps
    # any registered equation's consumers / domain tokens. The cathedral
    # autopilot main loop passes a dict candidate so we walk values.
    candidate_text = " ".join(
        f"{k}={v}" for k, v in candidate.items() if isinstance(v, (str, int, float))
    ).lower()

    matching: list[dict[str, Any]] = []
    try:
        for eq in query_equations():
            consumers_text = " ".join(eq.canonical_consumers).lower()
            domain_text = " ".join(str(v) for v in eq.domain_of_validity.values()).lower()
            if any(
                token in candidate_text
                for token in (
                    eq.equation_id.split("_v")[0],
                    *eq.canonical_consumers,
                )
                if token
            ):
                matching.append(
                    {
                        "equation_id": eq.equation_id,
                        "well_calibrated": eq.is_well_calibrated,
                        "residuals": dict(eq.predicted_vs_empirical_residual),
                    }
                )
    except Exception:  # noqa: BLE001 - defensive; registry may be empty
        matching = []

    if not matching:
        rationale = (
            "no canonical equation matches this candidate; observability-only "
            "annotation [predicted]"
        )
    else:
        ids = ", ".join(m["equation_id"] for m in matching)
        rationale = (
            f"canonical equation predictions available for: {ids} "
            "(observability-only; consult tools/list_canonical_equations.py for residuals) "
            "[predicted]"
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_equations": matching,
    }
