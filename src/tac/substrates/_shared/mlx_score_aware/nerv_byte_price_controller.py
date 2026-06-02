# SPDX-License-Identifier: MIT
"""Contest byte-price admission control for compact NeRV sections.

This module is a planning/control primitive. It prices existing carrier
sections and proposed residual/sidecar bytes against the official contest rate
term, while preserving false-authority blockers instead of promoting local or
proxy evidence into score authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from typing import Any

NERV_BYTE_PRICE_CONTROLLER_SCHEMA = "compact_nerv_byte_price_controller.v1"
CONTEST_RATE_POINTS = 25.0
FULL_CONTEST_SAMPLE_COUNT = 600

FALSE_AUTHORITY_FLAGS: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

ADMIT = "admit"
CUT = "cut"
PROTECT = "protect"
RETRAIN = "retrain"
DEMOTE = "demote"

EXISTING_SECTION_CUT = "existing_section_cut"
NEW_RESIDUAL_OR_SIDECAR = "new_residual_or_sidecar"

_ROW_KEYS = (
    "section_value_rows",
    "section_rows",
    "admission_rows",
    "decision_rows",
    "rows",
    "sections",
)
_SECTION_KEYS = (
    "section_id",
    "section_name",
    "section",
    "name",
    "neutralized_section",
    "component",
)
_BYTE_KEYS = (
    "bytes",
    "section_bytes",
    "payload_bytes",
    "candidate_bytes",
    "archive_bytes_removed_vs_baseline",
    "bytes_removed",
    "archive_delta_bytes",
    "delta_bytes",
    "byte_delta",
    "candidate_delta_bytes",
)
_DELTA_NONRATE_KEYS = (
    "delta_nonrate_score",
    "candidate_delta_nonrate_score",
    "measured_delta_nonrate_score",
    "measured_delta_nonrate",
    "nonrate_score_delta",
    "delta_distortion_score",
)
_DELTA_TOTAL_KEYS = (
    "delta_total_score",
    "delta_total_mlx_score_advisory",
    "candidate_delta_total_score",
    "objective_delta",
)
_DELTA_RATE_KEYS = (
    "delta_rate_score",
    "candidate_delta_rate_score",
    "rate_score_delta",
)
_ARCHIVE_SHA_KEYS = (
    "archive_sha256",
    "archive_sha",
    "archive_zip_sha256",
    "candidate_archive_sha256",
    "candidate_archive_sha",
    "source_archive_sha256",
)
_CANDIDATE_KEYS = (
    "candidate_id",
    "candidate",
    "archive_id",
    "variant_id",
    "row_id",
    "id",
)
_AXIS_KEYS = (
    "axis_tag",
    "axis_label",
    "axis",
    "score_axis",
    "evidence_axis",
    "evidence_grade",
    "hardware_axis",
    "substrate_axis",
)
_ADVISORY_MARKERS = (
    "advisory",
    "proxy",
    "macos",
    "mps",
    "mlx",
    "planning/control",
    "research-signal",
    "research_signal",
    "local",
)
_RECEIVER_PROOF_GOOD = (
    "satisfied",
    "valid",
    "passed",
    "receiver_proof_satisfied",
    "receiver_proof_valid",
    "runtime_consumption_proof_passed",
    "runtime_consumption_proof_ready",
)


class NervBytePriceControllerError(ValueError):
    """Raised when section rows cannot be interpreted."""


@dataclass(frozen=True)
class ContestBytePrice:
    """Resolved contest byte price and fail-closed fallback state."""

    score_per_byte: float | None
    original_video_bytes: int | None
    source: str
    blockers: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            self.score_per_byte is not None
            and self.original_video_bytes is not None
            and self.original_video_bytes > 0
            and isfinite(float(self.score_per_byte))
        )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "score_per_byte": self.score_per_byte,
            "original_video_bytes": self.original_video_bytes,
            "source": self.source,
            "available": self.available,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class SectionAdmissionInput:
    """Normalized section-value row consumed by the admission controller."""

    row_id: str
    section_id: str
    row_kind: str
    byte_delta: int | None
    section_bytes: int | None
    delta_nonrate_score: float | None
    family: str | None
    scope: str | None
    candidate_id: str | None
    archive_sha256: str | None
    axis_labels: tuple[str, ...]
    receiver_proof_status: str
    full_video_coverage: bool
    source: dict[str, Any]
    blockers: tuple[str, ...]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "section_id": self.section_id,
            "row_kind": self.row_kind,
            "byte_delta": self.byte_delta,
            "section_bytes": self.section_bytes,
            "delta_nonrate_score": self.delta_nonrate_score,
            "family": self.family,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "archive_sha256": self.archive_sha256,
            "axis_labels": list(self.axis_labels),
            "receiver_proof_status": self.receiver_proof_status,
            "full_video_coverage": self.full_video_coverage,
            "blockers": list(self.blockers),
            "source": self.source,
        }


@dataclass(frozen=True)
class SectionAdmissionDecision:
    """One contest byte-price decision row."""

    row_id: str
    section_id: str
    row_kind: str
    decision: str
    economic_decision: str
    byte_delta: int | None
    section_bytes: int | None
    delta_nonrate_score: float | None
    delta_rate_score: float | None
    delta_total_score: float | None
    family: str | None
    scope: str | None
    candidate_id: str | None
    archive_sha256: str | None
    axis_labels: tuple[str, ...]
    receiver_proof_status: str
    full_video_coverage: bool
    blockers: tuple[str, ...]
    source: dict[str, Any]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "section_id": self.section_id,
            "row_kind": self.row_kind,
            "decision": self.decision,
            "economic_decision": self.economic_decision,
            "byte_delta": self.byte_delta,
            "section_bytes": self.section_bytes,
            "delta_nonrate_score": self.delta_nonrate_score,
            "delta_rate_score": self.delta_rate_score,
            "delta_total_score": self.delta_total_score,
            "strict_residual_admission_rule": (
                "new residual/sidecar bytes admitted only when "
                "delta_total_score < 0 and no fail-closed blockers are present"
            ),
            "family": self.family,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "archive_sha256": self.archive_sha256,
            "axis_labels": list(self.axis_labels),
            "receiver_proof_status": self.receiver_proof_status,
            "full_video_coverage": self.full_video_coverage,
            "blockers": list(self.blockers),
            "source": self.source,
            **FALSE_AUTHORITY_FLAGS,
        }


def resolve_contest_byte_price() -> ContestBytePrice:
    """Resolve the official contest byte price from existing repo constants.

    The preferred source is ``modelsize_budget_plan.CONTEST_BYTE_PRICE_SCORE``.
    If that is unavailable, the fallback recomputes ``25 / ORIGINAL_VIDEO_BYTES``.
    Missing ``ORIGINAL_VIDEO_BYTES`` is a blocker, not a guessed constant.
    """

    try:
        module = import_module(
            "tac.substrates._shared.mlx_score_aware.modelsize_budget_plan"
        )
    except (ImportError, AttributeError):
        module = None
    if module is not None:
        price = _finite_float_or_none(getattr(module, "CONTEST_BYTE_PRICE_SCORE", None))
        original = _positive_int_or_none(getattr(module, "ORIGINAL_VIDEO_BYTES", None))
        if price is not None and original is not None:
            return ContestBytePrice(
                score_per_byte=price,
                original_video_bytes=original,
                source=(
                    "tac.substrates._shared.mlx_score_aware."
                    "modelsize_budget_plan.CONTEST_BYTE_PRICE_SCORE"
                ),
            )

    try:
        auth_schema = import_module("tac.auth_eval_schema")
    except (ImportError, AttributeError):
        auth_schema = None
    original = (
        None
        if auth_schema is None
        else _positive_int_or_none(getattr(auth_schema, "ORIGINAL_VIDEO_BYTES", None))
    )
    if original is None:
        return ContestBytePrice(
            score_per_byte=None,
            original_video_bytes=None,
            source="unresolved",
            blockers=("contest_byte_price_unavailable_original_video_bytes_missing",),
        )
    return ContestBytePrice(
        score_per_byte=CONTEST_RATE_POINTS / float(original),
        original_video_bytes=original,
        source="tac.auth_eval_schema.ORIGINAL_VIDEO_BYTES",
    )


def build_nerv_byte_price_plan(
    section_value_rows_or_artifact: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    baseline_id: str | None = None,
    byte_price: ContestBytePrice | None = None,
) -> dict[str, Any]:
    """Build a false-authority NeRV section admission plan."""

    price = byte_price or resolve_contest_byte_price()
    artifact_context, rows = _artifact_context_and_rows(section_value_rows_or_artifact)
    normalized = [
        _normalize_section_row(
            row,
            index=index,
            artifact_context=artifact_context,
            candidate_id=candidate_id,
        )
        for index, row in enumerate(rows)
    ]
    decisions = [
        _decision_for_row(row, price=price) for row in normalized
    ]
    decision_rows = [decision.as_jsonable() for decision in decisions]
    blockers = _ordered_unique(
        [
            *price.blockers,
            *_string_list(artifact_context.get("blockers")),
            *[
                blocker
                for decision in decisions
                for blocker in decision.blockers
            ],
        ]
    )
    counts = {
        decision: sum(1 for row in decisions if row.decision == decision)
        for decision in (PROTECT, CUT, RETRAIN, DEMOTE, ADMIT)
    }
    return {
        "schema": NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
        "candidate_id": candidate_id or artifact_context.get("candidate_id"),
        "baseline_id": baseline_id,
        "source_schema": artifact_context.get("schema"),
        "contest_byte_price": price.as_jsonable(),
        "selection_rule": (
            "delta_total_score = delta_nonrate_score + delta_rate_score; "
            "existing sections are cut only when their removal lowers total "
            "score; new residual/sidecar bytes are admitted only when "
            "delta_total_score < 0"
        ),
        "input_row_count": len(normalized),
        "decision_counts": counts,
        "full_video_coverage": all(row.full_video_coverage for row in normalized)
        if normalized
        else False,
        "decision_rows": decision_rows,
        "admitted_section_ids": _ids_for_decision(decisions, ADMIT),
        "cut_section_ids": _ids_for_decision(decisions, CUT),
        "protected_section_ids": _ids_for_decision(decisions, PROTECT),
        "retrain_section_ids": _ids_for_decision(decisions, RETRAIN),
        "demoted_section_ids": _ids_for_decision(decisions, DEMOTE),
        "blockers": blockers,
        **FALSE_AUTHORITY_FLAGS,
    }


def build_nerv_byte_price_plan_from_iterable(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_id: str | None = None,
    baseline_id: str | None = None,
    byte_price: ContestBytePrice | None = None,
) -> dict[str, Any]:
    """Build a plan from any iterable of section-value rows."""

    return build_nerv_byte_price_plan(
        list(rows),
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        byte_price=byte_price,
    )


def _decision_for_row(
    row: SectionAdmissionInput,
    *,
    price: ContestBytePrice,
) -> SectionAdmissionDecision:
    blockers = [*row.blockers, *price.blockers]
    delta_rate = None
    delta_total = None
    if not price.available:
        blockers.append("contest_byte_price_missing_fail_closed")
    if row.byte_delta is None:
        blockers.append("section_byte_delta_missing")
    elif price.available:
        delta_rate = float(row.byte_delta) * float(price.score_per_byte)
    if row.delta_nonrate_score is None:
        blockers.append("delta_nonrate_score_missing")
    if row.delta_nonrate_score is not None and delta_rate is not None:
        delta_total = float(row.delta_nonrate_score) + float(delta_rate)

    economic_decision = _economic_decision(row, delta_total=delta_total)
    final_blockers = tuple(_ordered_unique(blockers))
    decision = DEMOTE if final_blockers else economic_decision
    return SectionAdmissionDecision(
        row_id=row.row_id,
        section_id=row.section_id,
        row_kind=row.row_kind,
        decision=decision,
        economic_decision=economic_decision,
        byte_delta=row.byte_delta,
        section_bytes=row.section_bytes,
        delta_nonrate_score=row.delta_nonrate_score,
        delta_rate_score=delta_rate,
        delta_total_score=delta_total,
        family=row.family,
        scope=row.scope,
        candidate_id=row.candidate_id,
        archive_sha256=row.archive_sha256,
        axis_labels=row.axis_labels,
        receiver_proof_status=row.receiver_proof_status,
        full_video_coverage=row.full_video_coverage,
        blockers=final_blockers,
        source=row.source,
    )


def _economic_decision(
    row: SectionAdmissionInput,
    *,
    delta_total: float | None,
) -> str:
    if delta_total is None or row.delta_nonrate_score is None:
        return DEMOTE
    if row.row_kind == NEW_RESIDUAL_OR_SIDECAR:
        if delta_total < 0.0:
            return ADMIT
        if row.delta_nonrate_score < 0.0:
            return RETRAIN
        return DEMOTE
    if delta_total < 0.0:
        return CUT
    return PROTECT


def _normalize_section_row(
    row: Mapping[str, Any],
    *,
    index: int,
    artifact_context: Mapping[str, Any],
    candidate_id: str | None,
) -> SectionAdmissionInput:
    source = dict(row)
    section_id = _first_string(row, _SECTION_KEYS) or f"section_{index:04d}"
    row_id = _first_string(row, ("row_id", "id", "variant_id")) or section_id
    row_kind = _row_kind(row, section_id=section_id)
    byte_delta = _byte_delta(row, row_kind=row_kind)
    section_bytes = _section_bytes(row, byte_delta=byte_delta)
    delta_nonrate = _delta_nonrate_score(row)
    archive_sha = _archive_sha256(row) or _archive_sha256(artifact_context)
    axis_labels = _axis_labels(row, artifact_context)
    receiver_status = _receiver_proof_status(row, artifact_context)
    full_video = _full_video_coverage(row, artifact_context)
    blockers = _row_blockers(
        row,
        artifact_context=artifact_context,
        archive_sha256=archive_sha,
        axis_labels=axis_labels,
        receiver_proof_status=receiver_status,
        full_video_coverage=full_video,
    )
    return SectionAdmissionInput(
        row_id=row_id,
        section_id=section_id,
        row_kind=row_kind,
        byte_delta=byte_delta,
        section_bytes=section_bytes,
        delta_nonrate_score=delta_nonrate,
        family=_first_string(row, ("family", "carrier_family"))
        or _first_string(artifact_context, ("family", "carrier_family")),
        scope=_first_string(row, ("scope", "section_scope", "coverage_scope"))
        or _first_string(artifact_context, ("scope", "coverage_scope")),
        candidate_id=(
            candidate_id
            or _first_string(row, _CANDIDATE_KEYS)
            or _first_string(artifact_context, _CANDIDATE_KEYS)
        ),
        archive_sha256=archive_sha,
        axis_labels=axis_labels,
        receiver_proof_status=receiver_status,
        full_video_coverage=full_video,
        source=source,
        blockers=tuple(blockers),
    )


def _artifact_context_and_rows(
    rows_or_artifact: Iterable[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if isinstance(rows_or_artifact, Mapping):
        artifact = dict(rows_or_artifact)
        for key in _ROW_KEYS:
            rows = artifact.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                return artifact, [row for row in rows if isinstance(row, Mapping)]
        return artifact, [artifact]
    return {}, [row for row in rows_or_artifact if isinstance(row, Mapping)]


def _row_kind(row: Mapping[str, Any], *, section_id: str) -> str:
    explicit = _first_string(
        row,
        (
            "row_kind",
            "section_admission_kind",
            "admission_kind",
            "byte_role",
            "section_role",
        ),
    )
    if explicit:
        lowered = explicit.lower()
        if "residual" in lowered or "sidecar" in lowered or "add" in lowered:
            return NEW_RESIDUAL_OR_SIDECAR
        if "cut" in lowered or "remove" in lowered or "existing" in lowered:
            return EXISTING_SECTION_CUT
    lowered_section = section_id.lower()
    if "residual" in lowered_section or "sidecar" in lowered_section:
        removed = _positive_int_or_none(row.get("archive_bytes_removed_vs_baseline"))
        if removed is None or removed <= 0:
            return NEW_RESIDUAL_OR_SIDECAR
    return EXISTING_SECTION_CUT


def _byte_delta(row: Mapping[str, Any], *, row_kind: str) -> int | None:
    explicit = _integer_or_none(
        _first_present(
            row,
            (
                "byte_delta",
                "delta_bytes",
                "archive_delta_bytes",
                "candidate_delta_bytes",
            ),
        )
    )
    if explicit is not None:
        return explicit
    if row_kind == EXISTING_SECTION_CUT:
        removed = _positive_int_or_none(
            _first_present(row, ("archive_bytes_removed_vs_baseline", "bytes_removed"))
        )
        if removed is not None:
            return -removed
        section_bytes = _positive_int_or_none(
            _first_present(row, ("section_bytes", "bytes", "payload_bytes"))
        )
        return None if section_bytes is None else -section_bytes
    added = _positive_int_or_none(
        _first_present(
            row,
            (
                "candidate_bytes",
                "section_bytes",
                "bytes",
                "payload_bytes",
                "residual_bytes",
                "sidecar_bytes",
            ),
        )
    )
    return added


def _section_bytes(row: Mapping[str, Any], *, byte_delta: int | None) -> int | None:
    value = _positive_int_or_none(_first_present(row, _BYTE_KEYS))
    if value is not None:
        return value
    if byte_delta is not None:
        return abs(int(byte_delta))
    return None


def _delta_nonrate_score(row: Mapping[str, Any]) -> float | None:
    value = _finite_float_or_none(_first_present(row, _DELTA_NONRATE_KEYS))
    if value is not None:
        return value
    total = _finite_float_or_none(_first_present(row, _DELTA_TOTAL_KEYS))
    rate = _finite_float_or_none(_first_present(row, _DELTA_RATE_KEYS))
    if total is not None and rate is not None:
        return float(total) - float(rate)
    return None


def _row_blockers(
    row: Mapping[str, Any],
    *,
    artifact_context: Mapping[str, Any],
    archive_sha256: str | None,
    axis_labels: Sequence[str],
    receiver_proof_status: str,
    full_video_coverage: bool,
) -> list[str]:
    blockers = [
        *_string_list(artifact_context.get("blockers")),
        *_string_list(row.get("blockers")),
    ]
    lowered_axes = " ".join(axis_labels).lower()
    lowered_blockers = " ".join(blockers).lower()
    if not axis_labels:
        blockers.append("axis_label_missing")
    if any(marker in lowered_axes for marker in _ADVISORY_MARKERS) or any(
        marker in lowered_blockers for marker in ("advisory", "proxy")
    ):
        blockers.append("advisory_or_proxy_axis_not_promotion_authority")
    if not archive_sha256:
        blockers.append("missing_archive_sha256")
    if receiver_proof_status.lower() not in _RECEIVER_PROOF_GOOD:
        blockers.append("receiver_proof_not_satisfied")
    if not full_video_coverage:
        blockers.append("full_video_coverage_missing")
    return _ordered_unique(blockers)


def _receiver_proof_status(
    row: Mapping[str, Any],
    artifact_context: Mapping[str, Any],
) -> str:
    for mapping in (row, artifact_context):
        value = _first_string(
            mapping,
            (
                "receiver_proof_status",
                "receiver_contract_status",
                "runtime_consumption_proof_status",
            ),
        )
        if value:
            return value
        if mapping.get("receiver_proof_valid") is True:
            return "receiver_proof_valid"
        if mapping.get("runtime_consumption_proof_passed") is True:
            return "runtime_consumption_proof_passed"
        if mapping.get("runtime_consumption_proof_ready") is True:
            return "runtime_consumption_proof_ready"
        proof = mapping.get("receiver_proof")
        if isinstance(proof, Mapping):
            if proof.get("valid") is True or proof.get("proof_valid") is True:
                return "receiver_proof_valid"
            if proof.get("runtime_consumption_proof_ready") is True:
                return "runtime_consumption_proof_ready"
            if proof.get("runtime_consumption_proof_passed") is True:
                return "runtime_consumption_proof_passed"
            value = _first_string(proof, ("status", "receiver_proof_status"))
            if value:
                return value
        proof = mapping.get("runtime_consumption_proof")
        if isinstance(proof, Mapping):
            if proof.get("runtime_consumption_proof_ready") is True:
                return "runtime_consumption_proof_ready"
            if proof.get("runtime_consumption_proof_passed") is True:
                return "runtime_consumption_proof_passed"
            if proof.get("ready_for_exact_eval_runtime") is True:
                return "runtime_consumption_proof_ready"
            value = _first_string(
                proof,
                (
                    "status",
                    "receiver_proof_status",
                    "runtime_consumption_proof_status",
                ),
            )
            if value:
                return value
    return "missing"


def _full_video_coverage(
    row: Mapping[str, Any],
    artifact_context: Mapping[str, Any],
) -> bool:
    for mapping in (row, artifact_context):
        explicit = mapping.get("full_video_coverage")
        if isinstance(explicit, bool):
            return explicit
        full_video = mapping.get("full_video")
        if isinstance(full_video, bool):
            return full_video
        if isinstance(full_video, str):
            return full_video.lower() in {"executed", "true", "full", "full_video"}
        scope_status = mapping.get("scope_status")
        if isinstance(scope_status, Mapping):
            scope_full = scope_status.get("full_video")
            if isinstance(scope_full, str):
                return scope_full.lower() == "executed"
        for key in (
            "n_samples",
            "num_samples",
            "evaluated_pairs",
            "actual_pairs",
            "decoded_pairs",
            "scored_pairs",
            "num_pairs",
        ):
            count = _positive_int_or_none(mapping.get(key))
            if count is not None:
                return count >= FULL_CONTEST_SAMPLE_COUNT
    return False


def _axis_labels(
    row: Mapping[str, Any],
    artifact_context: Mapping[str, Any],
) -> tuple[str, ...]:
    values: list[str] = []
    for mapping in (artifact_context, row):
        for key in _AXIS_KEYS:
            value = mapping.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values.extend(str(item) for item in value if str(item))
            elif value:
                values.append(str(value))
    if not values:
        values.extend(_inferred_axis_labels(row, artifact_context))
    return tuple(_ordered_unique(values))


def _inferred_axis_labels(
    row: Mapping[str, Any],
    artifact_context: Mapping[str, Any],
) -> tuple[str, ...]:
    text = " ".join(
        [
            str(row.get("schema") or ""),
            str(artifact_context.get("schema") or ""),
            str(artifact_context.get("source_schema") or ""),
            " ".join(_string_list(row.get("blockers"))),
            " ".join(_string_list(artifact_context.get("blockers"))),
        ]
    ).lower()
    if (
        "mlx_local_response_is_advisory_not_score_authority" in text
        or "mlx_component_neutralization_profile" in text
        or "mlx_section_value_profile" in text
        or "pact_nerv_selector" in text
    ):
        return ("[macOS-MLX research-signal]",)
    return ()


def _archive_sha256(mapping: Mapping[str, Any]) -> str | None:
    direct = _first_string(mapping, _ARCHIVE_SHA_KEYS)
    if direct:
        return direct
    for key in ("candidate_archive", "archive", "source_archive"):
        nested = mapping.get(key)
        if isinstance(nested, Mapping):
            value = _first_string(nested, ("sha256", "sha", "archive_sha256"))
            if value:
                return value
    return None


def _ids_for_decision(
    decisions: Sequence[SectionAdmissionDecision],
    decision: str,
) -> list[str]:
    return [row.section_id for row in decisions if row.decision == decision]


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _first_string(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value = _first_present(mapping, keys)
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _positive_int_or_none(value: Any) -> int | None:
    parsed = _integer_or_none(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _integer_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    try:
        if float(value) != float(parsed):
            return None
    except (TypeError, ValueError):
        return None
    return parsed


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return []


__all__ = [
    "ADMIT",
    "CUT",
    "DEMOTE",
    "EXISTING_SECTION_CUT",
    "FALSE_AUTHORITY_FLAGS",
    "FULL_CONTEST_SAMPLE_COUNT",
    "NERV_BYTE_PRICE_CONTROLLER_SCHEMA",
    "NEW_RESIDUAL_OR_SIDECAR",
    "PROTECT",
    "RETRAIN",
    "ContestBytePrice",
    "NervBytePriceControllerError",
    "SectionAdmissionDecision",
    "SectionAdmissionInput",
    "build_nerv_byte_price_plan",
    "build_nerv_byte_price_plan_from_iterable",
    "resolve_contest_byte_price",
]
