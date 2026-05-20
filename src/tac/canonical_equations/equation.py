# SPDX-License-Identifier: MIT
"""Canonical equations + empirical anchors — typed contract.

Per operator NON-NEGOTIABLE 2026-05-19 verbatim: *"we need to formalize all
of this and canonicalize and operationalize because I am afraid we are
learning but if we don't have systems of equations and models and such we
are just gaining tribal knowledge"*.

The CanonicalEquation dataclass encodes one mathematical predictor plus
its calibration history. EmpiricalAnchor rows capture a single
predicted-vs-empirical measurement so future agents can audit residuals
and trigger auto-recalibration without re-reading prose memos.

Cross-references:
  * ``tac.canonical_equations.registry`` — fcntl-locked JSONL persistence.
  * ``tac.canonical_equations.builtins`` — initial population helpers.
  * ``tac.provenance`` — Catalog #323 canonical Provenance contract.
  * CLAUDE.md "Meta-Lagrangian/Pareto solver" + "Canonical equations + models registry"
    non-negotiable sections.

Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
dataclasses with explicit invariants; no hidden state; every field is
either machine-readable JSON or a small dotted callable path.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from tac.provenance.contract import Provenance


CANONICAL_EQUATION_SCHEMA_VERSION = "canonical_equation_v1_20260519"
"""Pinned schema version. Bump only via explicit migration landing."""

# Canonical equation_id pattern: lowercase snake_case with trailing _vN.
_EQUATION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")

# Canonical callable path: dotted module path + optional ":callable" suffix.
_CALLABLE_PATH_RE = re.compile(r"^[a-z_][a-z0-9_.]*(?::[a-z_][a-z0-9_]*)?$")

# Recalibration trigger taxonomy — operator-facing readable tokens.
RECALIBRATE_ON_NEW_ANCHORS = "when_3+_new_empirical_anchors_in_domain"
RECALIBRATE_ON_RESIDUAL_DRIFT = "when_residual_drift_exceeds_2x"
RECALIBRATE_ON_PARAMETER_REFIT = "when_operator_invokes_recalibrate_equation"
RECALIBRATE_NEVER_AUTO = "never_auto_operator_only"

VALID_RECALIBRATION_TRIGGERS = frozenset(
    {
        RECALIBRATE_ON_NEW_ANCHORS,
        RECALIBRATE_ON_RESIDUAL_DRIFT,
        RECALIBRATE_ON_PARAMETER_REFIT,
        RECALIBRATE_NEVER_AUTO,
    }
)


class InvalidEquationError(ValueError):
    """Raised when a CanonicalEquation or EmpiricalAnchor violates invariants.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": every field-level
    contract is enforced in ``__post_init__`` (not docstring-only) so the
    construction surface refuses bad inputs at the source.
    """


class DomainOfValidityViolation(ValueError):
    """Raised when a context is explicitly excluded from an equation's domain.

    Per CLAUDE.md "Canonical equations + models registry" non-negotiable +
    Catalog #344 sister discipline: every canonical equation MAY declare
    ``domain_of_validity_included`` + ``domain_of_validity_excluded`` lists
    inside its ``domain_of_validity`` mapping. Callers that invoke an
    equation's predictor in an EXCLUDED context (e.g., the DWT detail-
    subband empirical vindication 2026-05-20: KL=1.638 nats / 3.28σ proves
    direct procedural-codebook substitution on DWT detail subbands corrupts
    inverse DWT) MUST be refused at the source rather than silently
    producing a phantom prediction that downstream cathedral autopilot /
    Pareto solver / continual-learning consumers would absorb.

    Sister of :class:`InvalidEquationError` at the per-invocation surface
    (``InvalidEquationError`` is per-construction; this exception is
    per-invocation in a specific context).
    """


def _utc_now_iso() -> str:
    """Canonical UTC timestamp with trailing Z."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_iso_utc(value: str, field_name: str) -> None:
    """Refuse non-ISO-8601 UTC timestamps; matches per-helper convention."""
    if not isinstance(value, str) or not value:
        raise InvalidEquationError(f"{field_name} must be a non-empty ISO-UTC string")
    # Accept either trailing Z or +00:00 form.
    try:
        if value.endswith("Z"):
            _dt.datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidEquationError(
            f"{field_name}={value!r} is not valid ISO-8601 UTC: {exc}"
        ) from exc


@dataclass(frozen=True)
class EmpiricalAnchor:
    """One predicted-vs-empirical measurement attached to a CanonicalEquation.

    The anchor encodes (inputs, predicted_output, empirical_output, residual)
    as a typed row so auto-recalibration can refit equation parameters via
    least-squares OR the operator can audit per-axis residuals via the
    canonical CLI ``tools/list_canonical_equations.py``.

    Per Catalog #287 (empirical-claim-evidence-tag) + Catalog #323
    (canonical Provenance umbrella): every anchor MUST carry a Provenance
    so downstream consumers know which axis (`[contest-CUDA]` /
    `[contest-CPU]` / `[predicted]` / `MPS-research-signal` / etc.) the
    empirical_output was measured on.
    """

    anchor_id: str
    measurement_utc: str
    inputs: Mapping[str, Any]
    predicted_output: Any
    empirical_output: Any
    residual: float
    source_artifact: str
    measurement_method: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or not self.anchor_id.strip():
            raise InvalidEquationError("anchor_id must be a non-empty string")
        if any(c in self.anchor_id for c in ("\n", "\t", "\x1f")):
            raise InvalidEquationError("anchor_id must not contain newlines/tabs/0x1f")
        _require_iso_utc(self.measurement_utc, "measurement_utc")
        if not isinstance(self.inputs, Mapping):
            raise InvalidEquationError("inputs must be a mapping")
        if not isinstance(self.residual, (int, float)):
            raise InvalidEquationError("residual must be numeric")
        if self.residual != self.residual:  # NaN check
            raise InvalidEquationError("residual must not be NaN")
        if self.residual < 0:
            raise InvalidEquationError("residual must be >= 0 (normalized magnitude)")
        if not isinstance(self.source_artifact, str) or not self.source_artifact.strip():
            raise InvalidEquationError("source_artifact must be a non-empty string")
        if not isinstance(self.measurement_method, str) or not self.measurement_method.strip():
            raise InvalidEquationError("measurement_method must be a non-empty string")
        if not isinstance(self.provenance, Provenance):
            raise InvalidEquationError(
                f"provenance must be a tac.provenance.Provenance, got {type(self.provenance).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (Provenance flattened via its own as_dict)."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "anchor_id": self.anchor_id,
            "measurement_utc": self.measurement_utc,
            "inputs": dict(self.inputs),
            "predicted_output": self.predicted_output,
            "empirical_output": self.empirical_output,
            "residual": float(self.residual),
            "source_artifact": self.source_artifact,
            "measurement_method": self.measurement_method,
            "provenance": provenance_to_dict(self.provenance),
        }


@dataclass(frozen=True)
class CanonicalEquation:
    """One canonical mathematical predictor + calibration history.

    Every equation MUST declare:
      * ``equation_id`` (snake_case + trailing version slug per regex)
      * ``one_line_summary`` (operator-facing readable)
      * ``latex_form`` (for documentation / paper writeups)
      * ``python_callable_module_path`` (dotted module path; optional ``:callable``)
      * ``domain_of_validity`` (input type/range manifest)
      * ``units_in`` + ``units_out`` (per-field unit tokens for dimensional audit)
      * ``empirical_anchors`` (list[EmpiricalAnchor]; may be empty for design-only)
      * ``next_recalibration_trigger`` (one of VALID_RECALIBRATION_TRIGGERS)
      * ``canonical_consumers`` + ``canonical_producers`` (dotted-path lists; for
        producer→consumer audit per CLAUDE.md "Subagent coherence-by-default")
      * ``provenance`` (Catalog #323 canonical Provenance)

    Per CLAUDE.md "Apples-to-apples evidence discipline": the
    ``predicted_vs_empirical_residual`` dict maps an axis token (e.g.
    "tinyrenderer" or "segnet_class_pending") to the latest normalized
    residual for that axis. Auto-recalibration refreshes this dict.
    """

    equation_id: str
    name: str
    one_line_summary: str
    latex_form: str
    python_callable_module_path: str
    domain_of_validity: Mapping[str, Any]
    units_in: Mapping[str, str]
    units_out: Mapping[str, str]
    empirical_anchors: tuple[EmpiricalAnchor, ...]
    predicted_vs_empirical_residual: Mapping[str, float]
    last_calibration_utc: str
    next_recalibration_trigger: str
    canonical_consumers: tuple[str, ...]
    canonical_producers: tuple[str, ...]
    provenance: Provenance
    schema_version: str = CANONICAL_EQUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.equation_id, str):
            raise InvalidEquationError("equation_id must be a string")
        if not _EQUATION_ID_RE.match(self.equation_id):
            raise InvalidEquationError(
                f"equation_id={self.equation_id!r} must match snake_case_vN pattern "
                "(e.g. 'mps_drift_architecture_class_dependent_v1')"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidEquationError("name must be a non-empty string")
        if not isinstance(self.one_line_summary, str) or not self.one_line_summary.strip():
            raise InvalidEquationError("one_line_summary must be a non-empty string")
        if len(self.one_line_summary) > 200:
            raise InvalidEquationError(
                f"one_line_summary length={len(self.one_line_summary)} exceeds 200-char limit; "
                "move detail to the equation memo"
            )
        if not isinstance(self.latex_form, str) or not self.latex_form.strip():
            raise InvalidEquationError("latex_form must be a non-empty string")
        if not isinstance(self.python_callable_module_path, str):
            raise InvalidEquationError("python_callable_module_path must be a string")
        if not _CALLABLE_PATH_RE.match(self.python_callable_module_path):
            raise InvalidEquationError(
                f"python_callable_module_path={self.python_callable_module_path!r} must be "
                "dotted-module-path (optional :callable suffix); e.g. "
                "'tac.mps_diagnostic.drift_predictor:predict_drift'"
            )
        if not isinstance(self.domain_of_validity, Mapping):
            raise InvalidEquationError("domain_of_validity must be a mapping")
        if not isinstance(self.units_in, Mapping):
            raise InvalidEquationError("units_in must be a mapping")
        if not isinstance(self.units_out, Mapping):
            raise InvalidEquationError("units_out must be a mapping")
        if not isinstance(self.empirical_anchors, tuple):
            raise InvalidEquationError("empirical_anchors must be a tuple (frozen)")
        for i, anchor in enumerate(self.empirical_anchors):
            if not isinstance(anchor, EmpiricalAnchor):
                raise InvalidEquationError(
                    f"empirical_anchors[{i}] must be EmpiricalAnchor, "
                    f"got {type(anchor).__name__}"
                )
        if not isinstance(self.predicted_vs_empirical_residual, Mapping):
            raise InvalidEquationError("predicted_vs_empirical_residual must be a mapping")
        for axis, residual in self.predicted_vs_empirical_residual.items():
            if not isinstance(residual, (int, float)):
                raise InvalidEquationError(
                    f"predicted_vs_empirical_residual[{axis!r}] must be numeric"
                )
            if residual < 0:
                raise InvalidEquationError(
                    f"predicted_vs_empirical_residual[{axis!r}]={residual} must be >= 0"
                )
        _require_iso_utc(self.last_calibration_utc, "last_calibration_utc")
        if self.next_recalibration_trigger not in VALID_RECALIBRATION_TRIGGERS:
            raise InvalidEquationError(
                f"next_recalibration_trigger={self.next_recalibration_trigger!r} must be one of "
                f"{sorted(VALID_RECALIBRATION_TRIGGERS)!r}"
            )
        if not isinstance(self.canonical_consumers, tuple):
            raise InvalidEquationError("canonical_consumers must be a tuple")
        if not isinstance(self.canonical_producers, tuple):
            raise InvalidEquationError("canonical_producers must be a tuple")
        # Producer→consumer audit: an equation with NO producers AND NO
        # consumers is an orphan and refused. This is the structural
        # extinction of the "tribal knowledge with no machine-readable
        # consumer" failure mode the operator flagged.
        if not self.canonical_consumers and not self.canonical_producers:
            raise InvalidEquationError(
                f"equation_id={self.equation_id!r} has empty canonical_consumers AND "
                "canonical_producers — orphan equations are forbidden per "
                "operator NON-NEGOTIABLE 2026-05-19. Declare at least one "
                "consumer module-path (helper that reads this equation) OR "
                "producer module-path (helper that emits empirical anchors)."
            )
        if not isinstance(self.provenance, Provenance):
            raise InvalidEquationError(
                f"provenance must be a tac.provenance.Provenance, got "
                f"{type(self.provenance).__name__}"
            )
        if self.schema_version != CANONICAL_EQUATION_SCHEMA_VERSION:
            raise InvalidEquationError(
                f"schema_version={self.schema_version!r} != canonical "
                f"{CANONICAL_EQUATION_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict (anchors + provenance flattened)."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "schema_version": self.schema_version,
            "equation_id": self.equation_id,
            "name": self.name,
            "one_line_summary": self.one_line_summary,
            "latex_form": self.latex_form,
            "python_callable_module_path": self.python_callable_module_path,
            "domain_of_validity": dict(self.domain_of_validity),
            "units_in": dict(self.units_in),
            "units_out": dict(self.units_out),
            "empirical_anchors": [a.to_dict() for a in self.empirical_anchors],
            "predicted_vs_empirical_residual": dict(self.predicted_vs_empirical_residual),
            "last_calibration_utc": self.last_calibration_utc,
            "next_recalibration_trigger": self.next_recalibration_trigger,
            "canonical_consumers": list(self.canonical_consumers),
            "canonical_producers": list(self.canonical_producers),
            "provenance": provenance_to_dict(self.provenance),
        }

    def with_new_anchor(self, anchor: EmpiricalAnchor) -> "CanonicalEquation":
        """Return a new CanonicalEquation with the anchor appended (frozen-safe).

        Also refreshes ``predicted_vs_empirical_residual`` for the anchor's
        ``measurement_method`` axis token + bumps ``last_calibration_utc``.
        The original equation is not mutated (dataclass is frozen).
        """
        if not isinstance(anchor, EmpiricalAnchor):
            raise InvalidEquationError(
                f"with_new_anchor expected EmpiricalAnchor, got {type(anchor).__name__}"
            )
        new_anchors = self.empirical_anchors + (anchor,)
        new_residuals = dict(self.predicted_vs_empirical_residual)
        new_residuals[anchor.measurement_method] = anchor.residual
        # Build a fresh frozen copy via dataclasses.replace semantics.
        from dataclasses import replace

        return replace(
            self,
            empirical_anchors=new_anchors,
            predicted_vs_empirical_residual=new_residuals,
            last_calibration_utc=_utc_now_iso(),
        )

    @property
    def is_well_calibrated(self) -> bool:
        """True iff every per-axis residual is below the 2.0 universal threshold.

        2.0 = "predicted within 2x of empirical" (canonical readable threshold
        per the operator-facing dashboard). Lower-resolution equations may
        still be well-calibrated per their domain; higher-resolution
        equations needing tighter tolerances should track per-axis bands.
        """
        if not self.predicted_vs_empirical_residual:
            # No anchors yet — not falsified, not confirmed. Treat as not
            # well-calibrated (operator-visible cue to land first anchor).
            return False
        return all(r < 2.0 for r in self.predicted_vs_empirical_residual.values())


__all__ = [
    "CANONICAL_EQUATION_SCHEMA_VERSION",
    "RECALIBRATE_ON_NEW_ANCHORS",
    "RECALIBRATE_ON_RESIDUAL_DRIFT",
    "RECALIBRATE_ON_PARAMETER_REFIT",
    "RECALIBRATE_NEVER_AUTO",
    "VALID_RECALIBRATION_TRIGGERS",
    "InvalidEquationError",
    "DomainOfValidityViolation",
    "EmpiricalAnchor",
    "CanonicalEquation",
]
