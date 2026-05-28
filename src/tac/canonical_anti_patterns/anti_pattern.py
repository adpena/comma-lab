# SPDX-License-Identifier: MIT
"""Canonical anti-patterns + empirical falsifications — typed contract.

Per operator NON-NEGOTIABLE 2026-05-28 verbatim: *"learning anti-patterns
is upser important too for compounding continual learning, like the
canonical equations bu netgative and a higher layer of abstraction"*.

Sister of ``tac.canonical_equations.equation`` (POSITIVE registry). The
``AntiPattern`` dataclass encodes a CLASS-level forbidden pattern that
recurs across substrates/atoms/elements/components plus its falsification
history. ``EmpiricalFalsification`` rows capture single empirical
confirmation/refutation measurements so the auto-recalibrator can refresh
the falsification_band without re-reading prose memos.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + "Results must become
system intelligence" non-negotiables: anti-patterns are NEGATIVE constraints
on the Pareto polytope feasibility set. Where canonical equations ADD
predicted score deltas (positive feasibility regions), anti-patterns
EXCLUDE regions where known compounding/diagnosis/provenance failures
manifest empirically. The cathedral autopilot ranker + the per-axis
Dykstra solver (Slot 1 in flight) consume both registries to steer
next-cycle attack direction.

Cross-references:
  * ``tac.canonical_anti_patterns.registry`` — fcntl-locked JSONL persistence.
  * ``tac.canonical_anti_patterns.builtins`` — initial 12 anti-pattern population.
  * ``tac.canonical_anti_patterns.pattern_matcher`` — stack-spec → matched anti-patterns.
  * ``tac.canonical_equations.equation`` — sister POSITIVE registry.
  * ``tac.provenance`` — Catalog #323 canonical Provenance contract.
  * CLAUDE.md "FORBIDDEN PATTERNS" section (canonical source of initial population).

Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
dataclasses with explicit invariants; no hidden state; every field is
either machine-readable JSON or a small dotted callable path. Mirrors
``CanonicalEquation`` shape exactly so future readers see the
positive/negative symmetry at a glance.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any, Mapping

from tac.provenance.contract import Provenance


CANONICAL_ANTI_PATTERN_SCHEMA_VERSION = "canonical_anti_pattern_v1_20260528"
"""Pinned schema version. Bump only via explicit migration landing."""

# Canonical anti_pattern_id pattern: lowercase snake_case with trailing _vN.
_ANTI_PATTERN_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")

# Canonical falsification_id pattern (no _vN suffix; per-measurement).
_FALSIFICATION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Canonical recalibration trigger taxonomy (sister of canonical_equations).
RECALIBRATE_ON_NEW_FALSIFICATIONS = "when_3+_new_empirical_falsifications_in_domain"
RECALIBRATE_ON_SEVERITY_DRIFT = "when_severity_drift_exceeds_2x"
RECALIBRATE_ON_OPERATOR = "when_operator_invokes_recalibrate_anti_pattern"
RECALIBRATE_NEVER_AUTO = "never_auto_operator_only"

VALID_RECALIBRATION_TRIGGERS = frozenset(
    {
        RECALIBRATE_ON_NEW_FALSIFICATIONS,
        RECALIBRATE_ON_SEVERITY_DRIFT,
        RECALIBRATE_ON_OPERATOR,
        RECALIBRATE_NEVER_AUTO,
    }
)

# Canonical paradigm_class taxonomy.
PARADIGM_COMPOUNDING_ORDER = "compounding_order_anti_pattern"
PARADIGM_QUANTIZATION = "quantization_anti_pattern"
PARADIGM_DIAGNOSIS = "diagnosis_anti_pattern"
PARADIGM_PROVENANCE = "provenance_anti_pattern"
PARADIGM_DATA_SOURCE = "data_source_anti_pattern"
PARADIGM_OBSERVABILITY = "observability_anti_pattern"
PARADIGM_RIGOR_LOSS = "rigor_loss_anti_pattern"
PARADIGM_PREMATURE_KILL = "premature_kill_anti_pattern"

VALID_PARADIGM_CLASSES = frozenset(
    {
        PARADIGM_COMPOUNDING_ORDER,
        PARADIGM_QUANTIZATION,
        PARADIGM_DIAGNOSIS,
        PARADIGM_PROVENANCE,
        PARADIGM_DATA_SOURCE,
        PARADIGM_OBSERVABILITY,
        PARADIGM_RIGOR_LOSS,
        PARADIGM_PREMATURE_KILL,
    }
)

# Canonical severity taxonomy.
SEVERITY_CRITICAL = "critical_paradigm_blocker"
SEVERITY_HIGH = "high_compound_corruption"
SEVERITY_MEDIUM = "medium_substrate_regression"
SEVERITY_LOW = "low_implementation_inefficiency"

VALID_SEVERITIES = frozenset(
    {SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW}
)

# Per-falsification severity_observed (lower-cased; observation, not classification).
SEVERITY_OBSERVED_CRITICAL = "critical"
SEVERITY_OBSERVED_HIGH = "high"
SEVERITY_OBSERVED_MEDIUM = "medium"
SEVERITY_OBSERVED_LOW = "low"

VALID_OBSERVED_SEVERITIES = frozenset(
    {
        SEVERITY_OBSERVED_CRITICAL,
        SEVERITY_OBSERVED_HIGH,
        SEVERITY_OBSERVED_MEDIUM,
        SEVERITY_OBSERVED_LOW,
    }
)

# Canonical incident_classification taxonomy (per Catalog #307).
INCIDENT_PARADIGM_LEVEL_FALSIFICATION = "paradigm_level_falsification_of_anti_pattern"
INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION = (
    "implementation_level_confirmation_of_anti_pattern"
)
INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE = "ratification_of_anti_pattern_at_new_substrate"
INCIDENT_EDGE_CASE_PARTIAL = "edge_case_partial_manifestation"

VALID_INCIDENT_CLASSIFICATIONS = frozenset(
    {
        INCIDENT_PARADIGM_LEVEL_FALSIFICATION,
        INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
        INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        INCIDENT_EDGE_CASE_PARTIAL,
    }
)


class InvalidAntiPatternError(ValueError):
    """Raised when AntiPattern / EmpiricalFalsification violates invariants.

    Mirrors ``InvalidEquationError`` from the sister canonical_equations
    package. Every field-level contract is enforced in ``__post_init__``
    (not docstring-only) per CLAUDE.md "Comment-only contracts are
    FORBIDDEN" so the construction surface refuses bad inputs at the source.
    """


def _utc_now_iso() -> str:
    """Canonical UTC timestamp with trailing Z."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_iso_utc(value: str, field_name: str) -> None:
    """Refuse non-ISO-8601 UTC timestamps."""
    if not isinstance(value, str) or not value:
        raise InvalidAntiPatternError(
            f"{field_name} must be a non-empty ISO-UTC string"
        )
    try:
        if value.endswith("Z"):
            _dt.datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidAntiPatternError(
            f"{field_name}={value!r} is not valid ISO-8601 UTC: {exc}"
        ) from exc


@dataclass(frozen=True)
class EmpiricalFalsification:
    """One empirical confirmation/refutation attached to an AntiPattern.

    Mirrors ``EmpiricalAnchor`` from the sister canonical_equations package
    but at the NEGATIVE-registry surface: rather than measuring how close
    a predicted band lands to empirical, this measures how strongly the
    forbidden pattern manifested in a specific incident.

    Per Catalog #287 (empirical-claim-evidence-tag) + Catalog #323
    (canonical Provenance umbrella): every falsification MUST carry a
    Provenance so downstream consumers know which axis (`[contest-CUDA]` /
    `[contest-CPU]` / `[predicted]` / `[MPS-PROXY]` / etc.) the
    empirical_output was measured on. Per Catalog #127/#192: macOS-CPU
    advisory + MPS proxy + research-only grades are non-promotable by
    construction (the Provenance contract enforces this at construction
    time).

    Per Catalog #307 (paradigm-vs-implementation-falsification): every
    falsification declares ``incident_classification`` so future agents
    can distinguish IMPLEMENTATION-LEVEL confirmation (the specific stack
    matched the anti-pattern) from PARADIGM-LEVEL falsification (a sister
    incident proves the anti-pattern itself was over-classified).
    """

    anti_pattern_id: str
    falsification_id: str
    measurement_method: str
    empirical_artifact_path: str
    empirical_output: Mapping[str, Any]
    falsification_residual: float | None
    captured_at_utc: str
    canonical_provenance: Provenance
    incident_classification: str
    severity_observed: str
    operator_routable_unwind_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.anti_pattern_id, str) or not self.anti_pattern_id.strip():
            raise InvalidAntiPatternError("anti_pattern_id must be a non-empty string")
        # NOTE: cross-registry reference check (this id matches a registered
        # AntiPattern) is enforced by the registry layer, not here, because
        # __post_init__ should not depend on filesystem state.
        if not _ANTI_PATTERN_ID_RE.match(self.anti_pattern_id):
            raise InvalidAntiPatternError(
                f"anti_pattern_id={self.anti_pattern_id!r} must match snake_case_vN pattern"
            )
        if not isinstance(self.falsification_id, str):
            raise InvalidAntiPatternError("falsification_id must be a string")
        if not _FALSIFICATION_ID_RE.match(self.falsification_id):
            raise InvalidAntiPatternError(
                f"falsification_id={self.falsification_id!r} must match snake_case pattern"
            )
        if any(c in self.falsification_id for c in ("\n", "\t", "\x1f")):
            raise InvalidAntiPatternError(
                "falsification_id must not contain newlines/tabs/0x1f"
            )
        if not isinstance(self.measurement_method, str) or not self.measurement_method.strip():
            raise InvalidAntiPatternError("measurement_method must be a non-empty string")
        if not isinstance(self.empirical_artifact_path, str) or not self.empirical_artifact_path.strip():
            raise InvalidAntiPatternError(
                "empirical_artifact_path must be a non-empty string (real path "
                "or canonical reference like 'commit:<sha>' / 'catalog:#NNN')"
            )
        if not isinstance(self.empirical_output, Mapping):
            raise InvalidAntiPatternError("empirical_output must be a mapping")
        if self.falsification_residual is not None:
            if not isinstance(self.falsification_residual, (int, float)):
                raise InvalidAntiPatternError("falsification_residual must be numeric or None")
            if self.falsification_residual != self.falsification_residual:  # NaN
                raise InvalidAntiPatternError("falsification_residual must not be NaN")
            if self.falsification_residual < 0:
                raise InvalidAntiPatternError(
                    "falsification_residual must be >= 0 (normalized magnitude)"
                )
        _require_iso_utc(self.captured_at_utc, "captured_at_utc")
        if not isinstance(self.canonical_provenance, Provenance):
            raise InvalidAntiPatternError(
                f"canonical_provenance must be a tac.provenance.Provenance, "
                f"got {type(self.canonical_provenance).__name__}"
            )
        if self.incident_classification not in VALID_INCIDENT_CLASSIFICATIONS:
            raise InvalidAntiPatternError(
                f"incident_classification={self.incident_classification!r} must be one of "
                f"{sorted(VALID_INCIDENT_CLASSIFICATIONS)!r}"
            )
        if self.severity_observed not in VALID_OBSERVED_SEVERITIES:
            raise InvalidAntiPatternError(
                f"severity_observed={self.severity_observed!r} must be one of "
                f"{sorted(VALID_OBSERVED_SEVERITIES)!r}"
            )
        if not isinstance(self.operator_routable_unwind_path, str) or not self.operator_routable_unwind_path.strip():
            raise InvalidAntiPatternError(
                "operator_routable_unwind_path must be a non-empty string"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict (Provenance flattened via its own as_dict)."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "anti_pattern_id": self.anti_pattern_id,
            "falsification_id": self.falsification_id,
            "measurement_method": self.measurement_method,
            "empirical_artifact_path": self.empirical_artifact_path,
            "empirical_output": dict(self.empirical_output),
            "falsification_residual": (
                float(self.falsification_residual)
                if self.falsification_residual is not None
                else None
            ),
            "captured_at_utc": self.captured_at_utc,
            "canonical_provenance": provenance_to_dict(self.canonical_provenance),
            "incident_classification": self.incident_classification,
            "severity_observed": self.severity_observed,
            "operator_routable_unwind_path": self.operator_routable_unwind_path,
        }


@dataclass(frozen=True)
class AntiPattern:
    """One canonical CLASS-level forbidden pattern + falsification history.

    Every anti-pattern MUST declare:
      * ``anti_pattern_id`` (snake_case + trailing version slug per regex)
      * ``description`` (operator-facing readable explanation)
      * ``forbidden_pattern_predicate`` (mathematical / source-text / artifact-level signature)
      * ``falsification_band`` (range where the anti-pattern empirically MANIFESTS)
      * ``recurrence_conditions`` (situations that trigger manifestation)
      * ``canonical_source_anchor`` (CLAUDE.md FORBIDDEN_PATTERNS section / Catalog # / sister memo)
      * ``canonical_unwind_path`` (the canonical correct alternative)
      * ``canonical_producers`` (where the anti-pattern manifests in repo)
      * ``canonical_consumers`` (who consults this anti-pattern)
      * ``paradigm_class`` (one of VALID_PARADIGM_CLASSES)
      * ``severity`` (one of VALID_SEVERITIES)
      * ``provenance`` (Catalog #323 canonical Provenance)
      * ``empirical_falsifications`` (tuple; may be empty for design-only)
      * ``last_recalibration_utc`` + ``next_recalibration_trigger``

    Per CLAUDE.md "Subagent coherence-by-default": an anti-pattern with NO
    producers AND NO consumers is an orphan and refused. This is the
    structural extinction of the "anti-pattern in CLAUDE.md prose with no
    machine-readable consumer" failure mode the operator flagged.

    Per CLAUDE.md "Forbidden premature KILL": an AntiPattern is NOT a KILL
    verdict on the technique whose stack matches it. It is a CLASS of
    recurrences to AVOID, with a canonical_unwind_path that routes the
    operator to the correct alternative. The cathedral autopilot ranker
    consumes this to STEER next-cycle attack direction, not to retire
    research lanes.
    """

    anti_pattern_id: str
    description: str
    forbidden_pattern_predicate: str
    falsification_band: Mapping[str, float]
    recurrence_conditions: tuple[str, ...]
    canonical_source_anchor: str
    canonical_unwind_path: str
    canonical_producers: tuple[str, ...]
    canonical_consumers: tuple[str, ...]
    paradigm_class: str
    severity: str
    provenance: Provenance
    empirical_falsifications: tuple[EmpiricalFalsification, ...]
    last_recalibration_utc: str
    next_recalibration_trigger: str
    schema_version: str = CANONICAL_ANTI_PATTERN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.anti_pattern_id, str):
            raise InvalidAntiPatternError("anti_pattern_id must be a string")
        if not _ANTI_PATTERN_ID_RE.match(self.anti_pattern_id):
            raise InvalidAntiPatternError(
                f"anti_pattern_id={self.anti_pattern_id!r} must match snake_case_vN pattern"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise InvalidAntiPatternError("description must be a non-empty string")
        if len(self.description) > 600:
            raise InvalidAntiPatternError(
                f"description length={len(self.description)} exceeds 600-char limit; "
                "move detail to the anti-pattern memo"
            )
        if not isinstance(self.forbidden_pattern_predicate, str) or not self.forbidden_pattern_predicate.strip():
            raise InvalidAntiPatternError(
                "forbidden_pattern_predicate must be a non-empty string"
            )
        if not isinstance(self.falsification_band, Mapping):
            raise InvalidAntiPatternError("falsification_band must be a mapping")
        if not self.falsification_band:
            raise InvalidAntiPatternError("falsification_band must be non-empty")
        for k, v in self.falsification_band.items():
            if not isinstance(v, (int, float)):
                raise InvalidAntiPatternError(
                    f"falsification_band[{k!r}]={v!r} must be numeric"
                )
            if v != v:  # NaN
                raise InvalidAntiPatternError(
                    f"falsification_band[{k!r}] must not be NaN"
                )
        if not isinstance(self.recurrence_conditions, tuple):
            raise InvalidAntiPatternError("recurrence_conditions must be a tuple (frozen)")
        if not self.recurrence_conditions:
            raise InvalidAntiPatternError("recurrence_conditions must be non-empty")
        for i, condition in enumerate(self.recurrence_conditions):
            if not isinstance(condition, str) or not condition.strip():
                raise InvalidAntiPatternError(
                    f"recurrence_conditions[{i}] must be a non-empty string"
                )
        if not isinstance(self.canonical_source_anchor, str) or not self.canonical_source_anchor.strip():
            raise InvalidAntiPatternError("canonical_source_anchor must be a non-empty string")
        if not isinstance(self.canonical_unwind_path, str) or not self.canonical_unwind_path.strip():
            raise InvalidAntiPatternError("canonical_unwind_path must be a non-empty string")
        if not isinstance(self.canonical_producers, tuple):
            raise InvalidAntiPatternError("canonical_producers must be a tuple")
        if not isinstance(self.canonical_consumers, tuple):
            raise InvalidAntiPatternError("canonical_consumers must be a tuple")
        # Producer→consumer audit: orphan anti-patterns are refused (sister
        # of CanonicalEquation invariant).
        if not self.canonical_consumers and not self.canonical_producers:
            raise InvalidAntiPatternError(
                f"anti_pattern_id={self.anti_pattern_id!r} has empty "
                "canonical_consumers AND canonical_producers — orphan "
                "anti-patterns are forbidden per operator NON-NEGOTIABLE "
                "2026-05-28. Declare at least one consumer module-path "
                "(helper that consults this anti-pattern) OR producer "
                "module-path (helper that surfaces empirical falsifications)."
            )
        if self.paradigm_class not in VALID_PARADIGM_CLASSES:
            raise InvalidAntiPatternError(
                f"paradigm_class={self.paradigm_class!r} must be one of "
                f"{sorted(VALID_PARADIGM_CLASSES)!r}"
            )
        if self.severity not in VALID_SEVERITIES:
            raise InvalidAntiPatternError(
                f"severity={self.severity!r} must be one of {sorted(VALID_SEVERITIES)!r}"
            )
        if not isinstance(self.provenance, Provenance):
            raise InvalidAntiPatternError(
                f"provenance must be a tac.provenance.Provenance, got "
                f"{type(self.provenance).__name__}"
            )
        if not isinstance(self.empirical_falsifications, tuple):
            raise InvalidAntiPatternError("empirical_falsifications must be a tuple (frozen)")
        for i, fals in enumerate(self.empirical_falsifications):
            if not isinstance(fals, EmpiricalFalsification):
                raise InvalidAntiPatternError(
                    f"empirical_falsifications[{i}] must be EmpiricalFalsification, "
                    f"got {type(fals).__name__}"
                )
            if fals.anti_pattern_id != self.anti_pattern_id:
                raise InvalidAntiPatternError(
                    f"empirical_falsifications[{i}].anti_pattern_id="
                    f"{fals.anti_pattern_id!r} does not match parent "
                    f"anti_pattern_id={self.anti_pattern_id!r}"
                )
        _require_iso_utc(self.last_recalibration_utc, "last_recalibration_utc")
        if self.next_recalibration_trigger not in VALID_RECALIBRATION_TRIGGERS:
            raise InvalidAntiPatternError(
                f"next_recalibration_trigger={self.next_recalibration_trigger!r} "
                f"must be one of {sorted(VALID_RECALIBRATION_TRIGGERS)!r}"
            )
        if self.schema_version != CANONICAL_ANTI_PATTERN_SCHEMA_VERSION:
            raise InvalidAntiPatternError(
                f"schema_version={self.schema_version!r} != canonical "
                f"{CANONICAL_ANTI_PATTERN_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict (falsifications + provenance flattened)."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "schema_version": self.schema_version,
            "anti_pattern_id": self.anti_pattern_id,
            "description": self.description,
            "forbidden_pattern_predicate": self.forbidden_pattern_predicate,
            "falsification_band": dict(self.falsification_band),
            "recurrence_conditions": list(self.recurrence_conditions),
            "canonical_source_anchor": self.canonical_source_anchor,
            "canonical_unwind_path": self.canonical_unwind_path,
            "canonical_producers": list(self.canonical_producers),
            "canonical_consumers": list(self.canonical_consumers),
            "paradigm_class": self.paradigm_class,
            "severity": self.severity,
            "provenance": provenance_to_dict(self.provenance),
            "empirical_falsifications": [
                f.to_dict() for f in self.empirical_falsifications
            ],
            "last_recalibration_utc": self.last_recalibration_utc,
            "next_recalibration_trigger": self.next_recalibration_trigger,
        }

    def with_new_falsification(
        self, falsification: EmpiricalFalsification
    ) -> "AntiPattern":
        """Return a new AntiPattern with the falsification appended (frozen-safe).

        Also bumps ``last_recalibration_utc``. The original anti-pattern is
        not mutated (dataclass is frozen). Mirrors
        ``CanonicalEquation.with_new_anchor`` sister API exactly.
        """
        if not isinstance(falsification, EmpiricalFalsification):
            raise InvalidAntiPatternError(
                f"with_new_falsification expected EmpiricalFalsification, "
                f"got {type(falsification).__name__}"
            )
        if falsification.anti_pattern_id != self.anti_pattern_id:
            raise InvalidAntiPatternError(
                f"falsification.anti_pattern_id={falsification.anti_pattern_id!r} "
                f"does not match parent anti_pattern_id={self.anti_pattern_id!r}"
            )
        new_falsifications = self.empirical_falsifications + (falsification,)
        from dataclasses import replace

        return replace(
            self,
            empirical_falsifications=new_falsifications,
            last_recalibration_utc=_utc_now_iso(),
        )

    @property
    def recurrence_count(self) -> int:
        """Total number of EmpiricalFalsifications recorded for this anti-pattern."""
        return len(self.empirical_falsifications)

    @property
    def is_actively_recurring(self) -> bool:
        """True iff >=2 empirical falsifications recorded (canonical 'recurring' threshold).

        Per operator META directive: an anti-pattern with 0 or 1 falsifications
        is design-only (the canonical source anchor proved the class exists
        once). 2+ falsifications means the class has DEMONSTRATED recurrence
        across at least two distinct empirical incidents and is therefore
        priority-ordered for cathedral autopilot ranker exclusion.
        """
        return self.recurrence_count >= 2


__all__ = [
    "CANONICAL_ANTI_PATTERN_SCHEMA_VERSION",
    "INCIDENT_EDGE_CASE_PARTIAL",
    "INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION",
    "INCIDENT_PARADIGM_LEVEL_FALSIFICATION",
    "INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE",
    "PARADIGM_COMPOUNDING_ORDER",
    "PARADIGM_DATA_SOURCE",
    "PARADIGM_DIAGNOSIS",
    "PARADIGM_OBSERVABILITY",
    "PARADIGM_PREMATURE_KILL",
    "PARADIGM_PROVENANCE",
    "PARADIGM_QUANTIZATION",
    "PARADIGM_RIGOR_LOSS",
    "RECALIBRATE_NEVER_AUTO",
    "RECALIBRATE_ON_NEW_FALSIFICATIONS",
    "RECALIBRATE_ON_OPERATOR",
    "RECALIBRATE_ON_SEVERITY_DRIFT",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_OBSERVED_CRITICAL",
    "SEVERITY_OBSERVED_HIGH",
    "SEVERITY_OBSERVED_LOW",
    "SEVERITY_OBSERVED_MEDIUM",
    "VALID_INCIDENT_CLASSIFICATIONS",
    "VALID_OBSERVED_SEVERITIES",
    "VALID_PARADIGM_CLASSES",
    "VALID_RECALIBRATION_TRIGGERS",
    "VALID_SEVERITIES",
    "AntiPattern",
    "EmpiricalFalsification",
    "InvalidAntiPatternError",
]
