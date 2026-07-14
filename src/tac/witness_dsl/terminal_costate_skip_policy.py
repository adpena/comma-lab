# SPDX-License-Identifier: MIT
"""Typed, fail-closed DSL policy for the measured #396 terminal costate skip.

The admitted formulation is deliberately narrow: a post-training exact-metric
accept/reject search over the pinned n600 objective.  It is not an SPSA/ES
gradient estimator and it does not imply a multiplicative training-loop speedup.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.through_r.terminal_costate_skip import (
    TerminalAction,
    TerminalMethod,
    TerminalReceiptIdentity,
    decide_terminal_costate_skip,
)
from tac.witness_dsl.curriculum_dsl import Lever

POLICY_NAME = "terminal_exact_metric_costate_skip_396"
RECEIPT_SCHEMA = "p0_terminal_costate_skip_handoff.v1"
SOURCE_SCHEMA = "p0_terminal_costate_skip_source_snapshot.v1"
FIXTURE_SCHEMA = "p0_terminal_costate_skip_fixture_snapshot.v1"
REQUIRED_N_PAIRS = 600
CANONICAL_RECEIPT_PATH = ".omx/research/p0_terminal_costate_skip_handoff_20260713.json"
CANONICAL_RECEIPT_SHA256 = "17574857da5ff862e520140977e988197962f009d6870d23fe3071c398112a9c"
MEASURED_DELTA_S = -8.229284904293088e-06
MEASURED_FUNCTION_EVALS = 64
MEASURED_WINNER = "one_plus_one_es"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REPO = Path(__file__).resolve().parents[3]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _repo_file(entry: Any, *, label: str) -> tuple[Path | None, tuple[str, ...]]:
    """Resolve and verify one small durable repo-relative custody entry."""

    if not isinstance(entry, dict):
        return None, (f"{label} custody is missing",)
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        return None, (f"{label} path is missing",)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None, (f"{label} path escapes the repository",)
    path = (_REPO / candidate).resolve()
    try:
        path.relative_to(_REPO.resolve())
    except ValueError:
        return None, (f"{label} path escapes the repository",)
    try:
        raw = path.read_bytes()
    except OSError:
        return path, (f"{label} bytes are unavailable",)
    errors: list[str] = []
    if isinstance(entry.get("bytes"), bool) or entry.get("bytes") != len(raw):
        errors.append(f"{label} byte count mismatch")
    if entry.get("sha256") != _sha256_bytes(raw):
        errors.append(f"{label} sha256 mismatch")
    return path, tuple(errors)


def _read_object(path: Path, *, label: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, (f"{label} is not valid JSON",)
    if not isinstance(value, dict):
        return None, (f"{label} is not a JSON object",)
    return value, ()


@dataclass(frozen=True)
class TerminalCostateSkipEvidence:
    """Trusted identity; authorization re-reads all committed compact sources."""

    path: str
    expected_sha256: str

    @classmethod
    def canonical(cls) -> TerminalCostateSkipEvidence:
        return cls(CANONICAL_RECEIPT_PATH, CANONICAL_RECEIPT_SHA256)

    def _wrapper(self) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
        if not _valid_sha256(self.expected_sha256):
            return None, ("trusted terminal handoff sha256 is invalid",)
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            return None, ("terminal handoff path must be repo-relative",)
        absolute = (_REPO / path).resolve()
        try:
            absolute.relative_to(_REPO.resolve())
            raw = absolute.read_bytes()
        except (ValueError, OSError):
            return None, ("terminal handoff bytes are unavailable",)
        if _sha256_bytes(raw) != self.expected_sha256:
            return None, ("terminal handoff sha256 mismatch",)
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, ("terminal handoff is not valid JSON",)
        if not isinstance(payload, dict):
            return None, ("terminal handoff is not a JSON object",)
        return payload, ()

    def validation_errors(self) -> tuple[str, ...]:
        payload, errors_tuple = self._wrapper()
        errors = list(errors_tuple)
        if payload is None:
            return tuple(errors)

        required_values = {
            "schema": RECEIPT_SCHEMA,
            "status": "completed",
            "n_pairs": REQUIRED_N_PAIRS,
            "method": TerminalMethod.EXACT_METRIC_MC_396.value,
        }
        for field, expected in required_values.items():
            if payload.get(field) != expected:
                errors.append(f"terminal handoff {field} mismatch")

        objective = payload.get("objective")
        scorer = payload.get("scorer_custody")
        if not isinstance(objective, dict) or _canonical_sha256(objective) != payload.get(
            "objective_sha256"
        ):
            errors.append("terminal objective binding mismatch")
        if not isinstance(scorer, dict) or _canonical_sha256(scorer) != payload.get(
            "scorer_sha256"
        ):
            errors.append("terminal scorer binding mismatch")

        measured = payload.get("measured")
        if not isinstance(measured, dict):
            errors.append("terminal measurement is missing")
        else:
            if measured.get("winner") != MEASURED_WINNER:
                errors.append("terminal measured winner mismatch")
            if measured.get("n_pairs_composed_per_objective") != REQUIRED_N_PAIRS:
                errors.append("terminal objective is not composed over n600")
            if measured.get("exact_function_evaluations_search") != MEASURED_FUNCTION_EVALS:
                errors.append("terminal function-evaluation budget mismatch")
            delta_s = measured.get("delta_s")
            if not isinstance(delta_s, (int, float)) or not math.isclose(
                float(delta_s), MEASURED_DELTA_S, rel_tol=0.0, abs_tol=1e-18
            ):
                errors.append("terminal measured delta_s mismatch")

        derived = payload.get("derived")
        if not isinstance(derived, dict):
            errors.append("terminal derived accounting is missing")
        else:
            if derived.get("terminal_route_costate_calls") != 0:
                errors.append("terminal route is not costate-free")
            if derived.get("terminal_route_teacher_cost_reduction_fraction") != 1.0:
                errors.append("terminal route skip fraction mismatch")
            if derived.get("bulk_training_teacher_cost_reduction_fraction") != (
                "UNQUANTIFIED_NOT_COMPOSABLE"
            ):
                errors.append("terminal result improperly claims bulk training savings")

        authority = payload.get("authority")
        if not isinstance(authority, dict) or any(
            authority.get(field) is not False
            for field in ("score_claim", "promotion_eligible", "pointer_moved")
        ):
            errors.append("terminal handoff carries false authority")
        if (
            not isinstance(authority, dict)
            or authority.get("candidate_bytes_revalidated_at_authorization") is not False
            or not isinstance(authority.get("candidate_custody_scope"), str)
        ):
            errors.append("terminal candidate non-revalidation scope is missing")

        source_path, source_errors = _repo_file(
            payload.get("source_receipt_custody"), label="terminal source receipt"
        )
        errors.extend(source_errors)
        fixture_path, fixture_errors = _repo_file(
            payload.get("fixture_manifest_custody"), label="terminal fixture manifest"
        )
        errors.extend(fixture_errors)
        candidate = payload.get("candidate_archive_custody")
        if (
            not isinstance(candidate, dict)
            or candidate.get("bytes") != 177169
            or not _valid_sha256(candidate.get("sha256"))
            or not isinstance(candidate.get("path"), str)
        ):
            errors.append("terminal candidate archive custody is invalid")
        harness = payload.get("harness_custody")
        if not isinstance(harness, list) or len(harness) != 2:
            errors.append("terminal harness custody must name two source files")
        else:
            for index, entry in enumerate(harness):
                _path, custody_errors = _repo_file(entry, label=f"terminal harness {index}")
                errors.extend(custody_errors)

        source: dict[str, Any] | None = None
        if source_path is not None and not source_errors:
            source, source_json_errors = _read_object(source_path, label="terminal source receipt")
            errors.extend(source_json_errors)
        fixture: dict[str, Any] | None = None
        if fixture_path is not None and not fixture_errors:
            fixture, fixture_json_errors = _read_object(
                fixture_path, label="terminal fixture manifest"
            )
            errors.extend(fixture_json_errors)
        if source is not None:
            if source.get("schema") != SOURCE_SCHEMA:
                errors.append("terminal source receipt schema mismatch")
            if source.get("original_schema") != "ugc_terminal_polish_ab_receipt.v2":
                errors.append("terminal original source schema mismatch")
            if source.get("n_pairs_authority") != REQUIRED_N_PAIRS:
                errors.append("terminal source receipt lacks n600 authority")
            if source.get("measured_stochastic_winner") != MEASURED_WINNER:
                errors.append("terminal source winner mismatch")
            if source.get("paid_dispatch") is not False or source.get("score_claim") is not False:
                errors.append("terminal source receipt carries false authority")
            winning = source.get("winning_arm")
            if not isinstance(winning, dict):
                errors.append("terminal source winning arm is missing")
            else:
                if winning.get("function_evals_search") != MEASURED_FUNCTION_EVALS:
                    errors.append("terminal source search budget mismatch")
                if not math.isclose(
                    float(winning.get("delta_s", math.inf)),
                    MEASURED_DELTA_S,
                    rel_tol=0.0,
                    abs_tol=1e-18,
                ):
                    errors.append("terminal source delta_s mismatch")
                verification = winning.get("verification")
                if not isinstance(verification, dict) or any(
                    verification.get(field) != 0.0
                    for field in ("seg_cell_maxabs", "pose_cell_maxabs", "residual")
                ):
                    errors.append("terminal source exact-cell verification mismatch")
            source_fixture = source.get("fixture")
            candidate_value = candidate if isinstance(candidate, dict) else {}
            scorer_value = scorer if isinstance(scorer, dict) else {}
            if not isinstance(source_fixture, dict):
                errors.append("terminal source fixture snapshot is missing")
            else:
                if source_fixture.get("archive_sha256") != candidate_value.get("sha256"):
                    errors.append("terminal source archive binding mismatch")
                if source_fixture.get("base_archive_bytes") != candidate_value.get("bytes"):
                    errors.append("terminal source archive byte binding mismatch")
                if source_fixture.get("gt_cache_sha256") != scorer_value.get("gt_cache_sha256"):
                    errors.append("terminal source GT-cache binding mismatch")
        if fixture is not None:
            if fixture.get("schema") != FIXTURE_SCHEMA:
                errors.append("terminal fixture snapshot schema mismatch")
            archive = candidate if isinstance(candidate, dict) else {}
            scorer_value = scorer if isinstance(scorer, dict) else {}
            if fixture.get("archive_sha256") != archive.get("sha256"):
                errors.append("terminal fixture archive binding mismatch")
            if fixture.get("base_archive_bytes") != archive.get("bytes"):
                errors.append("terminal fixture byte binding mismatch")
            if fixture.get("gt_cache_sha256") != scorer_value.get("gt_cache_sha256"):
                errors.append("terminal fixture GT-cache binding mismatch")
            if fixture.get("score_claim") is not False or fixture.get("promotable") is not False:
                errors.append("terminal fixture carries false authority")
        return tuple(errors)

    def receipt_identity(self) -> TerminalReceiptIdentity:
        """Return a through-R identity only after the strong wrapper gate passes."""

        errors = self.validation_errors()
        if errors:
            raise ValueError("terminal evidence refused: " + "; ".join(errors))
        return TerminalReceiptIdentity.from_path(
            _REPO / self.path,
            expected_sha256=self.expected_sha256,
        )


@dataclass(frozen=True)
class TerminalCostateSkipPolicy:
    """Default-off policy; evidence admission is distinct from live activation."""

    enabled: bool = False
    evidence: TerminalCostateSkipEvidence | None = None
    provider_current: bool = True

    def measurement_errors(self) -> tuple[str, ...]:
        if self.evidence is None:
            return ("terminal n600 handoff evidence is missing",)
        return self.evidence.validation_errors()

    def compile_contract(self) -> dict[str, Any]:
        measurement_errors = self.measurement_errors()
        activation_errors = list(measurement_errors)
        if not self.enabled:
            activation_errors.append("terminal costate-skip policy is default-off")
        if not self.provider_current:
            activation_errors.append("#396 exact-metric provider is unavailable")
        action = TerminalAction.EXACT_METRIC_MC_ORDINARY_ROUTE
        if not measurement_errors and self.evidence is not None:
            identity = self.evidence.receipt_identity()
            action = decide_terminal_costate_skip(
                method=TerminalMethod.EXACT_METRIC_MC_396,
                receipt=identity,
                expected_receipt=identity,
                expected_receipt_sha256=self.evidence.expected_sha256,
                expected_dimension_certificate_sha256=None,
            ).action
            # Re-run the recursive source gate after the pure decision so a
            # source/fixture replacement in the validation-to-action window
            # cannot leave an admitted contract behind.
            post_action_errors = self.evidence.validation_errors()
            if post_action_errors:
                measurement_errors = tuple(
                    dict.fromkeys((*measurement_errors, *post_action_errors))
                )
                activation_errors = list(measurement_errors)
                if not self.enabled:
                    activation_errors.append("terminal costate-skip policy is default-off")
                if not self.provider_current:
                    activation_errors.append("#396 exact-metric provider is unavailable")
                action = TerminalAction.EXACT_METRIC_MC_ORDINARY_ROUTE
        return {
            "policy": POLICY_NAME,
            "method": TerminalMethod.EXACT_METRIC_MC_396.value,
            "measurement_admitted": not measurement_errors,
            "measurement_errors": list(measurement_errors),
            "terminal_action": action.value,
            "terminal_route_activation_admitted": not activation_errors,
            "terminal_route_activation_errors": activation_errors,
            "live_trainer_argv": [],
            "bulk_training_teacher_cost_reduction": "UNQUANTIFIED_NOT_COMPOSABLE",
            "candidate_bytes_revalidated_at_authorization": False,
            "candidate_custody_scope": (
                "committed snapshots bind original path/bytes/SHA; restore and explicitly audit "
                "original candidate bytes before any remeasurement"
            ),
            "score_claim": False,
        }


def terminal_exact_metric_costate_skip_lever(
    policy: TerminalCostateSkipPolicy | None = None,
) -> Lever:
    """Return the named argv-inert DSL leg; callers must consume the policy API."""

    compiled = (policy or TerminalCostateSkipPolicy()).compile_contract()
    measurement = "ADMITTED" if compiled["measurement_admitted"] else "REFUSED"
    activation = "ADMITTED" if compiled["terminal_route_activation_admitted"] else "REFUSED"
    return Lever(
        name=POLICY_NAME,
        overrides={},
        epochs_delta=0,
        notes=(
            f"argv-inert terminal policy; n600 measurement={measurement}; activation={activation}; "
            "exact-metric #396 only; candidate bytes are snapshot-bound but not rehashed at "
            "authorization; SPSA/ES and bulk training savings are not admitted"
        ),
    )


__all__ = [
    "CANONICAL_RECEIPT_PATH",
    "CANONICAL_RECEIPT_SHA256",
    "POLICY_NAME",
    "TerminalCostateSkipEvidence",
    "TerminalCostateSkipPolicy",
    "terminal_exact_metric_costate_skip_lever",
]
