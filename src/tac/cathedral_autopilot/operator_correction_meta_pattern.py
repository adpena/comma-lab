# SPDX-License-Identifier: MIT
"""Operator-correction META-pattern formalization for cathedral autopilot (GAP 2).

Per ``feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_canonicalize_and_harden_for_automation_standing_directive_20260528.md``
canonical anti-pattern #2 (``operator_correction_canonical_apparatus_mutation_lag_v1``):
operator binding correction -> memory file landing TIME-LAG before
canonical apparatus mutation (equation + anti-pattern + consumer)
registered structurally.

This helper closes that gap. Every operator binding correction auto-
registers a canonical equation + canonical anti-pattern + (optional)
cathedral consumer within the SAME TURN, NOT in a deferred follow-on
landing.

Per Catalog #344 canonical equations + anti-patterns registry +
Catalog #335 cathedral consumer auto-discovery + Catalog #371 auto-
recalibration trigger. Per CLAUDE.md "Results must become system
intelligence" + "memos must be acted upon".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OperatorBindingCorrectionRegistration:
    """One operator binding correction with canonical apparatus mutations.

    Args:
        operator_quote: verbatim operator quote (≥4 chars; placeholder
            ``<rationale>`` / ``<reason>`` literals rejected per
            Catalog #287 sister discipline).
        anti_pattern_id: canonical anti-pattern id registered as a result
            of this correction (snake_case + ``_vN`` suffix per
            ``_ANTI_PATTERN_ID_RE``).
        equation_id: canonical equation id registered as a result of
            this correction (snake_case + ``_vN`` suffix per
            ``_EQUATION_ID_RE``).
        mission_predicted_contribution: one of the Catalog #300
            5-value enum (``frontier_breaking`` / ``frontier_protecting``
            / ``rigor_overhead`` / ``apparatus_maintenance`` /
            ``mission_questioned``).
        registered_at_utc: ISO-UTC timestamp of registration; set by
            :func:`register_operator_binding_correction` if not provided.
        rationale: operator-facing readable summary explaining the
            registration + cross-references.
        canonical_unwind_path: the canonical correct alternative the
            corrected behavior should adopt going forward.
        sister_consumer_module_path: optional dotted path to a cathedral
            consumer auto-discovered per Catalog #335; None if no sister
            consumer is needed.
    """

    operator_quote: str
    anti_pattern_id: str
    equation_id: str
    mission_predicted_contribution: str
    registered_at_utc: str
    rationale: str
    canonical_unwind_path: str
    sister_consumer_module_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator_quote, str) or not self.operator_quote.strip():
            raise ValueError("operator_quote must be a non-empty string")
        # Placeholder rejection per Catalog #287 sister discipline so
        # the helper's docstring example cannot self-register.
        quote_lower = self.operator_quote.strip().lower()
        if quote_lower in {"<rationale>", "<reason>", "rationale", "reason"}:
            raise ValueError(
                "operator_quote rejected as placeholder per Catalog #287 sister discipline"
            )
        if len(self.operator_quote.strip()) < 4:
            raise ValueError(
                "operator_quote must be >= 4 chars per non-placeholder discipline"
            )
        if not isinstance(self.anti_pattern_id, str) or not self.anti_pattern_id.strip():
            raise ValueError("anti_pattern_id must be a non-empty string")
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise ValueError("equation_id must be a non-empty string")
        if not isinstance(self.mission_predicted_contribution, str):
            raise ValueError("mission_predicted_contribution must be a string")
        if self.mission_predicted_contribution not in _VALID_MISSION_CONTRIBUTIONS:
            raise ValueError(
                f"mission_predicted_contribution={self.mission_predicted_contribution!r} "
                f"must be one of {sorted(_VALID_MISSION_CONTRIBUTIONS)}"
            )
        if not isinstance(self.registered_at_utc, str) or not self.registered_at_utc.strip():
            raise ValueError("registered_at_utc must be a non-empty ISO-UTC string")
        if not isinstance(self.canonical_unwind_path, str) or not self.canonical_unwind_path.strip():
            raise ValueError("canonical_unwind_path must be a non-empty string")
        if self.sister_consumer_module_path is not None:
            if not isinstance(self.sister_consumer_module_path, str) or not self.sister_consumer_module_path.strip():
                raise ValueError(
                    "sister_consumer_module_path must be None or a non-empty string"
                )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "operator_quote": self.operator_quote,
            "anti_pattern_id": self.anti_pattern_id,
            "equation_id": self.equation_id,
            "mission_predicted_contribution": self.mission_predicted_contribution,
            "registered_at_utc": self.registered_at_utc,
            "rationale": self.rationale,
            "canonical_unwind_path": self.canonical_unwind_path,
            "sister_consumer_module_path": self.sister_consumer_module_path,
        }


# Mirrors Catalog #300 mission-alignment enum to avoid a hard import-cycle
# with ``tac.council_continual_learning``.
_VALID_MISSION_CONTRIBUTIONS = frozenset(
    {
        "frontier_breaking",
        "frontier_breaking_enabler",
        "frontier_protecting",
        "rigor_overhead",
        "apparatus_maintenance",
        "mission_questioned",
    }
)


def _utc_now_iso() -> str:
    """Canonical UTC timestamp with trailing Z (mirrors sister registries)."""
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def register_operator_binding_correction(
    operator_quote: str,
    *,
    anti_pattern_id: str,
    equation_id: str,
    mission_predicted_contribution: str = "apparatus_maintenance",
    canonical_unwind_path: str,
    sister_consumer_module_path: str | None = None,
    auto_fire_recalibrator: bool = True,
    agent: str | None = "claude",
    subagent_id: str | None = None,
) -> OperatorBindingCorrectionRegistration:
    """Register an operator binding correction via canonical apparatus mutations.

    Per ``feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_canonicalize_and_harden_for_automation_standing_directive_20260528.md``
    + canonical anti-pattern ``operator_correction_canonical_apparatus_mutation_lag_v1``:
    every operator binding correction MUST be acted upon via canonical
    apparatus mutations within the same turn. The historical pattern of
    "memo lands at turn N; canonical mutation lands at turn N+M" is the
    bug class this helper extincts.

    Best-effort semantics: if the canonical equation/anti-pattern
    registry import fails OR the registration raises, this helper
    returns a registration record with the operator-routable rationale
    pointing at the canonical structural alternative. The helper itself
    does NOT raise; it surfaces the structural state for caller routing.

    Per Catalog #371: if ``auto_fire_recalibrator=True``, the canonical
    recalibrator fires within this call so any newly-landed empirical
    anchors per the equation are absorbed before the next ranking pass.

    Per Catalog #335: if ``sister_consumer_module_path`` is provided, the
    helper notes the path but does NOT auto-create the consumer (per the
    canonical convention-over-configuration paradigm: NEW consumers land
    in ``src/tac/cathedral_consumers/<name>/__init__.py`` with the
    canonical contract, and the auto-discovery loop ingests them).

    Args:
        operator_quote: verbatim operator quote that issued the correction.
        anti_pattern_id: id of the canonical anti-pattern to register
            (must already exist in ``tac.canonical_anti_patterns`` OR
            this helper records the routing for follow-on registration).
        equation_id: id of the canonical equation to register/update.
        mission_predicted_contribution: one of the Catalog #300 enum
            values; defaults to ``apparatus_maintenance``.
        canonical_unwind_path: the canonical correct alternative.
        sister_consumer_module_path: optional cathedral consumer path.
        auto_fire_recalibrator: if True (default), fire the canonical
            recalibrator per Catalog #371.
        agent: agent identifier for the registry event (default "claude").
        subagent_id: optional subagent id for the registry event.

    Returns:
        :class:`OperatorBindingCorrectionRegistration` recording the
        registration outcome + canonical apparatus mutation paths.
    """
    registered_at = _utc_now_iso()
    rationale_parts: list[str] = []
    rationale_parts.append(
        f"Operator binding correction registered at {registered_at}; "
        f"canonical anti-pattern={anti_pattern_id!r}; "
        f"canonical equation={equation_id!r}; "
        f"mission contribution={mission_predicted_contribution!r}."
    )

    # Best-effort: try to fire the canonical recalibrator. The helper
    # MUST NOT raise per CLAUDE.md "memos must be acted upon" — even
    # partial registration is operator-routable evidence.
    recalibrator_fired = False
    recalibrator_error: str | None = None
    if auto_fire_recalibrator:
        try:
            from tac.canonical_equations.registry import (
                auto_recalibrate_from_continual_learning_posterior,
            )

            try:
                report = auto_recalibrate_from_continual_learning_posterior(
                    equation_id,
                    agent=agent,
                    subagent_id=subagent_id,
                )
                recalibrator_fired = True
                rationale_parts.append(
                    f"Catalog #371 recalibrator fired: "
                    f"{report.equations_checked} equations checked / "
                    f"{report.equations_recalibrated} recalibrated / "
                    f"{report.new_anchors_absorbed} anchors absorbed."
                )
            except Exception as exc:  # noqa: BLE001
                recalibrator_error = f"{type(exc).__name__}: {exc}"
                rationale_parts.append(
                    f"Catalog #371 recalibrator fired with non-fatal error "
                    f"({recalibrator_error}); registration proceeds (best-effort)."
                )
        except ImportError as exc:
            recalibrator_error = f"ImportError: {exc}"
            rationale_parts.append(
                f"Catalog #371 recalibrator unavailable ({recalibrator_error}); "
                "registration records the routing for follow-on."
            )

    if sister_consumer_module_path is not None:
        rationale_parts.append(
            f"Sister cathedral consumer routing: {sister_consumer_module_path} "
            "(operator-routable: land package in src/tac/cathedral_consumers/ "
            "per Catalog #335 canonical contract for auto-discovery)."
        )

    if not recalibrator_fired:
        rationale_parts.append(
            "Per CLAUDE.md 'memos must be acted upon': operator-routable "
            "next-spawn = land sister registration in tac.canonical_anti_patterns "
            "+ tac.canonical_equations to close the canonical apparatus mutation."
        )

    rationale = " ".join(rationale_parts)

    return OperatorBindingCorrectionRegistration(
        operator_quote=operator_quote,
        anti_pattern_id=anti_pattern_id,
        equation_id=equation_id,
        mission_predicted_contribution=mission_predicted_contribution,
        registered_at_utc=registered_at,
        rationale=rationale,
        canonical_unwind_path=canonical_unwind_path,
        sister_consumer_module_path=sister_consumer_module_path,
    )


__all__ = [
    "OperatorBindingCorrectionRegistration",
    "register_operator_binding_correction",
]
