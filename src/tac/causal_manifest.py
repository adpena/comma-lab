"""Typed causal/transition manifest for witness training and the costate organ.

The manifest is score-neutral observability.  It records what was observed; it does not
grant Markov sufficiency, causal identification, support, or score authority.  FORE and HCM
consumers therefore fail closed unless explicit transition, coverage, apparatus, and positive-
control custody is present.

Persistence uses :func:`tac.jsonl_store.append_locked_jsonl`, the repository's canonical
fcntl-locked append-only JSONL writer.  The public import surface is intentionally this single
module so launch/config composers and later readers do not grow parallel schemas.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from tac.jsonl_store import append_locked_jsonl

SCHEMA_ID = "pact.causal_manifest.v1"
MANIFEST_FILENAME = "causal_manifest.jsonl"
NON_PROMOTABLE_AXIS = "[observability-only] NON-PROMOTABLE"

ROW_RUN_MANIFEST = "run_manifest"
ROW_BOUNDARY = "boundary"
ROW_TRANSITION = "transition"
ROW_EXPLORATION_DECISION = "exploration_decision"
ROW_COVERAGE_RECEIPT = "coverage_receipt"
ROW_EVENT_MARK = "event_mark"

_ROW_KINDS = frozenset(
    {
        ROW_RUN_MANIFEST,
        ROW_BOUNDARY,
        ROW_TRANSITION,
        ROW_EXPLORATION_DECISION,
        ROW_COVERAGE_RECEIPT,
        ROW_EVENT_MARK,
    }
)
_DIGEST_HEX_LENGTHS = {"sha256": 64, "git_sha": 40, "git_tree": 40}
_BOUNDARY_PRIORITY = {"baseline": 0, "verdict": 2, "checkpoint": 4, "stage": 6, "final": 8}
_CANONICAL_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

EVENT_FAMILY_KINDS: dict[str, frozenset[str]] = {
    "topology": frozenset(
        {
            "component_birth",
            "component_death",
            "merge",
            "split",
            "hole_birth",
            "hole_death",
            "junction_incidence_change",
        }
    ),
    "chart": frozenset(
        {
            "atlas_transition",
            "stabilizer_change",
            "admissible_arrow_change",
            "occlusion",
            "disocclusion",
            "nonrigid_residual",
            "clamp_cell_change",
            "relu_cell_change",
            "argmax_cell_change",
        }
    ),
    "receiver_lattice": frozenset(
        {
            "resize_cell_crossing",
            "uint8_rounding_crossing",
            "subpixel_phase_wrap",
            "sampled_connectivity_change",
        }
    ),
}
EVENT_FAMILY_PRIORITY = {"topology": 0, "chart": 1, "receiver_lattice": 2}


class CausalManifestError(ValueError):
    """Base error for invalid causal-manifest rows or files."""


class CausalManifestConflictError(CausalManifestError):
    """Raised when an append would reinterpret an existing immutable identifier."""


def utc_now() -> str:
    """Return a stable UTC timestamp for append-only rows."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalManifestError(f"{name} must be a non-empty string")
    return value


def _require_canonical_utc(value: str, name: str) -> str:
    _require_text(value, name)
    if not _CANONICAL_UTC_RE.fullmatch(value):
        raise CausalManifestError(f"{name} must be canonical UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CausalManifestError(f"{name} is not a valid UTC timestamp") from exc
    return value


def _finite_or_none(value: float | None, name: str) -> float | None:
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
        raise CausalManifestError(f"{name} must be finite or None")
    return None if value is None else float(value)


def _jsonable(value: Any) -> Any:
    """Convert common config values to deterministic JSON without inventing content."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CausalManifestError("configuration values may not contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        scalar = scalar_item()
        if scalar is not value:
            return _jsonable(scalar)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    raise CausalManifestError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a value canonically for exact treatment/config custody."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    """SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash an existing file without loading it into memory."""

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class JsonField:
    """One immutable named JSON value; the canonical JSON bytes are the stored value."""

    name: str
    value_json: str

    def __post_init__(self) -> None:
        _require_text(self.name, "JsonField.name")
        try:
            parsed = json.loads(self.value_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CausalManifestError(f"JsonField {self.name!r} has invalid JSON") from exc
        if canonical_json(parsed) != self.value_json:
            raise CausalManifestError(f"JsonField {self.name!r} must use canonical JSON")

    @classmethod
    def from_value(cls, name: str, value: Any) -> JsonField:
        return cls(name=str(name), value_json=canonical_json(value))

    @property
    def value(self) -> Any:
        return json.loads(self.value_json)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value_json": self.value_json}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> JsonField:
        return cls(name=str(row["name"]), value_json=str(row["value_json"]))


def freeze_fields(values: Mapping[str, Any] | None = None) -> tuple[JsonField, ...]:
    """Freeze a mapping into sorted immutable fields."""

    values = values or {}
    return tuple(JsonField.from_value(str(k), v) for k, v in sorted(values.items(), key=lambda item: str(item[0])))


def thaw_fields(fields: Sequence[JsonField]) -> dict[str, Any]:
    return {item.name: item.value for item in fields}


def _validate_fields(fields: Sequence[JsonField], name: str) -> None:
    names = [item.name for item in fields]
    if names != sorted(names) or len(set(names)) != len(names):
        raise CausalManifestError(f"{name} must have unique lexicographically sorted names")


@dataclass(frozen=True)
class DigestRef:
    """Typed content digest.  Git object ids are not mislabeled as SHA-256."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        expected = _DIGEST_HEX_LENGTHS.get(self.algorithm)
        if expected is None:
            raise CausalManifestError(f"unsupported digest algorithm {self.algorithm!r}")
        if len(self.value) != expected or any(c not in "0123456789abcdef" for c in self.value.lower()):
            raise CausalManifestError(f"{self.algorithm} digest must be {expected} hexadecimal characters")
        object.__setattr__(self, "value", self.value.lower())

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> DigestRef:
        return cls(algorithm=str(row["algorithm"]), value=str(row["value"]))


@dataclass(frozen=True)
class ArtifactRef:
    """Artifact custody with an exact digest or an explicit unavailable reason."""

    role: str
    uri: str
    digest: DigestRef | None = None
    digest_status: str = "captured"  # captured | unavailable | deferred_cost
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.role, "ArtifactRef.role")
        _require_text(self.uri, "ArtifactRef.uri")
        if self.digest_status not in {"captured", "unavailable", "deferred_cost"}:
            raise CausalManifestError("ArtifactRef.digest_status is invalid")
        if self.digest_status == "captured" and self.digest is None:
            raise CausalManifestError("captured artifacts require a digest")
        if self.digest_status != "captured" and self.digest is not None:
            raise CausalManifestError("non-captured artifacts may not carry a digest")
        if self.digest_status != "captured":
            _require_text(self.unavailable_reason or "", "ArtifactRef.unavailable_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "uri": self.uri,
            "digest": None if self.digest is None else self.digest.to_dict(),
            "digest_status": self.digest_status,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ArtifactRef:
        digest = row.get("digest")
        return cls(
            role=str(row["role"]),
            uri=str(row["uri"]),
            digest=None if digest is None else DigestRef.from_dict(digest),
            digest_status=str(row.get("digest_status", "captured")),
            unavailable_reason=row.get("unavailable_reason"),
        )


@dataclass(frozen=True)
class StagePlanEntry:
    name: str
    start_epoch: int | None
    trigger: str

    def __post_init__(self) -> None:
        _require_text(self.name, "StagePlanEntry.name")
        _require_text(self.trigger, "StagePlanEntry.trigger")
        if self.start_epoch is not None and self.start_epoch < 0:
            raise CausalManifestError("StagePlanEntry.start_epoch must be >= 0 or None")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start_epoch": self.start_epoch, "trigger": self.trigger}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> StagePlanEntry:
        start = row.get("start_epoch")
        return cls(name=str(row["name"]), start_epoch=None if start is None else int(start), trigger=str(row["trigger"]))


@dataclass(frozen=True)
class RunTreatmentManifest:
    row_id: str
    run_id: str
    treatment_vector: tuple[JsonField, ...]
    treatment_sha256: str
    base_checkpoint: ArtifactRef
    seed: int
    machine: str
    backend: str
    axis: str
    data_order: tuple[JsonField, ...]
    data_order_sha256: str
    stage_plan: tuple[StagePlanEntry, ...]
    scorer_artifacts: tuple[ArtifactRef, ...]
    cache_artifacts: tuple[ArtifactRef, ...]
    created_at_utc: str
    schema_id: str = SCHEMA_ID
    row_kind: str = ROW_RUN_MANIFEST
    score_neutral: bool = True
    promotable: bool = False

    def __post_init__(self) -> None:
        _validate_row_header(self.schema_id, self.row_kind, self.row_id, self.run_id, ROW_RUN_MANIFEST)
        _validate_fields(self.treatment_vector, "treatment_vector")
        _validate_fields(self.data_order, "data_order")
        if canonical_sha256(thaw_fields(self.treatment_vector)) != self.treatment_sha256:
            raise CausalManifestError("treatment_sha256 does not match treatment_vector")
        if canonical_sha256(thaw_fields(self.data_order)) != self.data_order_sha256:
            raise CausalManifestError("data_order_sha256 does not match data_order")
        _validate_sha256(self.treatment_sha256, "treatment_sha256")
        _validate_sha256(self.data_order_sha256, "data_order_sha256")
        _require_text(self.machine, "machine")
        _require_text(self.backend, "backend")
        _require_text(self.axis, "axis")
        _require_text(self.created_at_utc, "created_at_utc")
        if not isinstance(self.seed, int):
            raise CausalManifestError("seed must be an int")
        if not self.stage_plan:
            raise CausalManifestError("stage_plan must contain at least one declared stage")
        if not self.scorer_artifacts or not self.cache_artifacts:
            raise CausalManifestError("scorer_artifacts and cache_artifacts must not be empty")
        if not self.score_neutral or self.promotable:
            raise CausalManifestError("causal manifest rows are score-neutral and non-promotable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "row_kind": self.row_kind,
            "row_id": self.row_id,
            "run_id": self.run_id,
            "treatment_vector": [item.to_dict() for item in self.treatment_vector],
            "treatment_sha256": self.treatment_sha256,
            "base_checkpoint": self.base_checkpoint.to_dict(),
            "seed": self.seed,
            "machine": self.machine,
            "backend": self.backend,
            "axis": self.axis,
            "data_order": [item.to_dict() for item in self.data_order],
            "data_order_sha256": self.data_order_sha256,
            "stage_plan": [item.to_dict() for item in self.stage_plan],
            "scorer_artifacts": [item.to_dict() for item in self.scorer_artifacts],
            "cache_artifacts": [item.to_dict() for item in self.cache_artifacts],
            "created_at_utc": self.created_at_utc,
            "score_neutral": self.score_neutral,
            "promotable": self.promotable,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> RunTreatmentManifest:
        return cls(
            row_id=str(row["row_id"]),
            run_id=str(row["run_id"]),
            treatment_vector=tuple(JsonField.from_dict(item) for item in row["treatment_vector"]),
            treatment_sha256=str(row["treatment_sha256"]),
            base_checkpoint=ArtifactRef.from_dict(row["base_checkpoint"]),
            seed=int(row["seed"]),
            machine=str(row["machine"]),
            backend=str(row["backend"]),
            axis=str(row["axis"]),
            data_order=tuple(JsonField.from_dict(item) for item in row["data_order"]),
            data_order_sha256=str(row["data_order_sha256"]),
            stage_plan=tuple(StagePlanEntry.from_dict(item) for item in row["stage_plan"]),
            scorer_artifacts=tuple(ArtifactRef.from_dict(item) for item in row["scorer_artifacts"]),
            cache_artifacts=tuple(ArtifactRef.from_dict(item) for item in row["cache_artifacts"]),
            created_at_utc=str(row["created_at_utc"]),
            schema_id=str(row["schema_id"]),
            row_kind=str(row["row_kind"]),
            score_neutral=bool(row.get("score_neutral", True)),
            promotable=bool(row.get("promotable", False)),
        )


@dataclass(frozen=True)
class LossTerm:
    name: str
    weight: float
    value: float

    def __post_init__(self) -> None:
        _require_text(self.name, "LossTerm.name")
        _finite_or_none(self.weight, "LossTerm.weight")
        _finite_or_none(self.value, "LossTerm.value")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "value": self.value}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> LossTerm:
        return cls(name=str(row["name"]), weight=float(row["weight"]), value=float(row["value"]))


@dataclass(frozen=True)
class ApparatusState:
    weights_stepped: bool | None = None
    accepted_fraction: float | None = None
    guard_path: str | None = None
    measurement_mode: str = "observational"
    positive_control: str | None = None
    total_loss: float | None = None
    loss_terms: tuple[LossTerm, ...] = ()
    negative_controls: tuple[JsonField, ...] = ()

    def __post_init__(self) -> None:
        if self.accepted_fraction is not None and not 0.0 <= float(self.accepted_fraction) <= 1.0:
            raise CausalManifestError("accepted_fraction must be in [0, 1]")
        _finite_or_none(self.total_loss, "total_loss")
        _require_text(self.measurement_mode, "measurement_mode")
        _validate_fields(self.negative_controls, "negative_controls")
        names = [item.name for item in self.loss_terms]
        if len(names) != len(set(names)):
            raise CausalManifestError("loss_terms names must be unique")

    def negative_control(self, name: str) -> float | None:
        value = thaw_fields(self.negative_controls).get(name)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None

    def closure_residual(self) -> float | None:
        if self.total_loss is None or not self.loss_terms:
            return None
        return float(self.total_loss) - sum(float(item.weight) * float(item.value) for item in self.loss_terms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_stepped": self.weights_stepped,
            "accepted_fraction": self.accepted_fraction,
            "guard_path": self.guard_path,
            "measurement_mode": self.measurement_mode,
            "positive_control": self.positive_control,
            "total_loss": self.total_loss,
            "loss_terms": [item.to_dict() for item in self.loss_terms],
            "negative_controls": [item.to_dict() for item in self.negative_controls],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ApparatusState:
        return cls(
            weights_stepped=row.get("weights_stepped"),
            accepted_fraction=row.get("accepted_fraction"),
            guard_path=row.get("guard_path"),
            measurement_mode=str(row.get("measurement_mode", "observational")),
            positive_control=row.get("positive_control"),
            total_loss=row.get("total_loss"),
            loss_terms=tuple(LossTerm.from_dict(item) for item in row.get("loss_terms", ())),
            negative_controls=tuple(JsonField.from_dict(item) for item in row.get("negative_controls", ())),
        )


@dataclass(frozen=True)
class RealizedOutcome:
    observed: bool
    through_r: bool
    d_seg: float | None = None
    d_pose: float | None = None
    archive_bytes: int | None = None
    implied_score: float | None = None
    d_seg_by_class: tuple[float, ...] = ()
    axis: str = NON_PROMOTABLE_AXIS
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("d_seg", "d_pose", "implied_score"):
            _finite_or_none(getattr(self, name), name)
        for value in self.d_seg_by_class:
            _finite_or_none(value, "d_seg_by_class")
        if self.archive_bytes is not None and self.archive_bytes < 0:
            raise CausalManifestError("archive_bytes must be >= 0")
        _require_text(self.axis, "RealizedOutcome.axis")
        if self.observed:
            if not self.through_r:
                raise CausalManifestError("observed causal-manifest outcomes must be realized through R")
            if (
                self.d_seg is None
                and self.d_pose is None
                and self.archive_bytes is None
                and self.implied_score is None
                and not self.d_seg_by_class
            ):
                raise CausalManifestError("observed outcomes require at least one measured value")
        else:
            _require_text(self.missing_reason or "", "RealizedOutcome.missing_reason")
            if (
                any(value is not None for value in (self.d_seg, self.d_pose, self.archive_bytes, self.implied_score))
                or self.d_seg_by_class
            ):
                raise CausalManifestError("unobserved outcomes may not carry measured values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "through_r": self.through_r,
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "archive_bytes": self.archive_bytes,
            "implied_score": self.implied_score,
            "d_seg_by_class": list(self.d_seg_by_class),
            "axis": self.axis,
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> RealizedOutcome:
        return cls(
            observed=bool(row["observed"]),
            through_r=bool(row["through_r"]),
            d_seg=row.get("d_seg"),
            d_pose=row.get("d_pose"),
            archive_bytes=row.get("archive_bytes"),
            implied_score=row.get("implied_score"),
            d_seg_by_class=tuple(float(v) for v in row.get("d_seg_by_class", ())),
            axis=str(row.get("axis", NON_PROMOTABLE_AXIS)),
            missing_reason=row.get("missing_reason"),
        )


@dataclass(frozen=True)
class StateSummary:
    boundary_id: str
    sequence_index: int
    boundary_kind: str
    epoch: int
    stage: str
    policy_sha256: str
    data_order_cursor: int | None
    telemetry_history_sha256: str | None
    telemetry_history_rows: int
    checkpoint: ArtifactRef | None
    resume_state_sha256: str | None
    rng_state_sha256: str | None
    controller_state_sha256: str | None
    apparatus: ApparatusState
    outcome: RealizedOutcome
    observed_at_utc: str

    def __post_init__(self) -> None:
        _require_text(self.boundary_id, "boundary_id")
        _require_text(self.boundary_kind, "boundary_kind")
        _require_text(self.stage, "stage")
        _require_text(self.observed_at_utc, "observed_at_utc")
        if self.sequence_index < 0 or self.epoch < 0:
            raise CausalManifestError("sequence_index and epoch must be >= 0")
        if self.data_order_cursor is not None and self.data_order_cursor < 0:
            raise CausalManifestError("data_order_cursor must be >= 0 or None")
        if self.telemetry_history_rows < 0:
            raise CausalManifestError("telemetry_history_rows must be >= 0")
        _validate_sha256(self.policy_sha256, "policy_sha256")
        for name in (
            "telemetry_history_sha256",
            "resume_state_sha256",
            "rng_state_sha256",
            "controller_state_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_sha256(value, name)

    @property
    def state_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_state_sha=False))

    def to_dict(self, *, include_state_sha: bool = True) -> dict[str, Any]:
        row = {
            "boundary_id": self.boundary_id,
            "sequence_index": self.sequence_index,
            "boundary_kind": self.boundary_kind,
            "epoch": self.epoch,
            "stage": self.stage,
            "policy_sha256": self.policy_sha256,
            "data_order_cursor": self.data_order_cursor,
            "telemetry_history_sha256": self.telemetry_history_sha256,
            "telemetry_history_rows": self.telemetry_history_rows,
            "checkpoint": None if self.checkpoint is None else self.checkpoint.to_dict(),
            "resume_state_sha256": self.resume_state_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "controller_state_sha256": self.controller_state_sha256,
            "apparatus": self.apparatus.to_dict(),
            "outcome": self.outcome.to_dict(),
            "observed_at_utc": self.observed_at_utc,
        }
        if include_state_sha:
            row["state_sha256"] = self.state_sha256
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> StateSummary:
        checkpoint = row.get("checkpoint")
        state = cls(
            boundary_id=str(row["boundary_id"]),
            sequence_index=int(row["sequence_index"]),
            boundary_kind=str(row["boundary_kind"]),
            epoch=int(row["epoch"]),
            stage=str(row["stage"]),
            policy_sha256=str(row["policy_sha256"]),
            data_order_cursor=row.get("data_order_cursor"),
            telemetry_history_sha256=row.get("telemetry_history_sha256"),
            telemetry_history_rows=int(row.get("telemetry_history_rows", 0)),
            checkpoint=None if checkpoint is None else ArtifactRef.from_dict(checkpoint),
            resume_state_sha256=row.get("resume_state_sha256"),
            rng_state_sha256=row.get("rng_state_sha256"),
            controller_state_sha256=row.get("controller_state_sha256"),
            apparatus=ApparatusState.from_dict(row["apparatus"]),
            outcome=RealizedOutcome.from_dict(row["outcome"]),
            observed_at_utc=str(row["observed_at_utc"]),
        )
        claimed = row.get("state_sha256")
        if claimed is not None and claimed != state.state_sha256:
            raise CausalManifestError(f"state_sha256 mismatch for {state.boundary_id!r}")
        return state


@dataclass(frozen=True)
class ActionSummary:
    action_id: str
    action_type: str
    arm_id: str
    policy_id: str
    policy_sha256: str
    parameters: tuple[JsonField, ...] = ()

    def __post_init__(self) -> None:
        for name in ("action_id", "action_type", "arm_id", "policy_id"):
            _require_text(getattr(self, name), name)
        _validate_sha256(self.policy_sha256, "ActionSummary.policy_sha256")
        _validate_fields(self.parameters, "ActionSummary.parameters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "arm_id": self.arm_id,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "parameters": [item.to_dict() for item in self.parameters],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ActionSummary:
        return cls(
            action_id=str(row["action_id"]),
            action_type=str(row["action_type"]),
            arm_id=str(row["arm_id"]),
            policy_id=str(row["policy_id"]),
            policy_sha256=str(row["policy_sha256"]),
            parameters=tuple(JsonField.from_dict(item) for item in row.get("parameters", ())),
        )


@dataclass(frozen=True)
class RewardObservation:
    estimand_id: str
    observed: bool
    value: float | None
    components: tuple[JsonField, ...] = ()
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.estimand_id, "RewardObservation.estimand_id")
        _validate_fields(self.components, "RewardObservation.components")
        _finite_or_none(self.value, "RewardObservation.value")
        if self.observed and self.value is None:
            raise CausalManifestError("observed rewards require a value")
        if not self.observed:
            if self.value is not None:
                raise CausalManifestError("unobserved rewards may not carry a value")
            _require_text(self.missing_reason or "", "RewardObservation.missing_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimand_id": self.estimand_id,
            "observed": self.observed,
            "value": self.value,
            "components": [item.to_dict() for item in self.components],
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> RewardObservation:
        return cls(
            estimand_id=str(row["estimand_id"]),
            observed=bool(row["observed"]),
            value=row.get("value"),
            components=tuple(JsonField.from_dict(item) for item in row.get("components", ())),
            missing_reason=row.get("missing_reason"),
        )


@dataclass(frozen=True)
class BoundaryObservationRow:
    row_id: str
    run_id: str
    state: StateSummary
    emitted_at_utc: str
    schema_id: str = SCHEMA_ID
    row_kind: str = ROW_BOUNDARY

    def __post_init__(self) -> None:
        _validate_row_header(self.schema_id, self.row_kind, self.row_id, self.run_id, ROW_BOUNDARY)
        _require_text(self.emitted_at_utc, "emitted_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "row_kind": self.row_kind,
            "row_id": self.row_id,
            "run_id": self.run_id,
            "state": self.state.to_dict(),
            "emitted_at_utc": self.emitted_at_utc,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> BoundaryObservationRow:
        return cls(
            row_id=str(row["row_id"]),
            run_id=str(row["run_id"]),
            state=StateSummary.from_dict(row["state"]),
            emitted_at_utc=str(row["emitted_at_utc"]),
            schema_id=str(row["schema_id"]),
            row_kind=str(row["row_kind"]),
        )


@dataclass(frozen=True)
class TransitionRow:
    row_id: str
    transition_id: str
    run_id: str
    pair_id: str | None
    state: StateSummary
    action: ActionSummary
    reward: RewardObservation
    next_state: StateSummary
    emitted_at_utc: str
    schema_id: str = SCHEMA_ID
    row_kind: str = ROW_TRANSITION

    def __post_init__(self) -> None:
        _validate_row_header(self.schema_id, self.row_kind, self.row_id, self.run_id, ROW_TRANSITION)
        _require_text(self.transition_id, "transition_id")
        _require_text(self.emitted_at_utc, "emitted_at_utc")
        if self.next_state.sequence_index <= self.state.sequence_index:
            raise CausalManifestError("transition next_state must be strictly later than state")
        if self.next_state.epoch < self.state.epoch:
            raise CausalManifestError("transition next_state epoch may not move backward")
        if self.action.policy_sha256 != self.next_state.policy_sha256:
            raise CausalManifestError("action policy digest must match next_state policy digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "row_kind": self.row_kind,
            "row_id": self.row_id,
            "transition_id": self.transition_id,
            "run_id": self.run_id,
            "pair_id": self.pair_id,
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward.to_dict(),
            "next_state": self.next_state.to_dict(),
            "emitted_at_utc": self.emitted_at_utc,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> TransitionRow:
        return cls(
            row_id=str(row["row_id"]),
            transition_id=str(row["transition_id"]),
            run_id=str(row["run_id"]),
            pair_id=row.get("pair_id"),
            state=StateSummary.from_dict(row["state"]),
            action=ActionSummary.from_dict(row["action"]),
            reward=RewardObservation.from_dict(row["reward"]),
            next_state=StateSummary.from_dict(row["next_state"]),
            emitted_at_utc=str(row["emitted_at_utc"]),
            schema_id=str(row["schema_id"]),
            row_kind=str(row["row_kind"]),
        )


@dataclass(frozen=True)
class ArmPropensity:
    arm_id: str
    propensity: float

    def __post_init__(self) -> None:
        _require_text(self.arm_id, "ArmPropensity.arm_id")
        if not isinstance(self.propensity, (int, float)) or not math.isfinite(float(self.propensity)):
            raise CausalManifestError("propensity must be finite")
        if not 0.0 <= float(self.propensity) <= 1.0:
            raise CausalManifestError("propensity must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "propensity": self.propensity}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ArmPropensity:
        return cls(arm_id=str(row["arm_id"]), propensity=float(row["propensity"]))


@dataclass(frozen=True)
class ExplorationDecisionRow:
    row_id: str
    decision_id: str
    run_id: str
    state: StateSummary
    chosen_arm: str
    arm_propensities: tuple[ArmPropensity, ...]
    policy_id: str
    policy_sha256: str
    policy_mode: str  # deterministic | randomized
    exploration_hook: str  # disabled_pending_operator_go | externally_authorized
    executed: bool
    actuation: str
    random_seed: int | None
    random_draw: float | None
    emitted_at_utc: str
    schema_id: str = SCHEMA_ID
    row_kind: str = ROW_EXPLORATION_DECISION

    def __post_init__(self) -> None:
        _validate_row_header(self.schema_id, self.row_kind, self.row_id, self.run_id, ROW_EXPLORATION_DECISION)
        for name in ("decision_id", "chosen_arm", "policy_id", "exploration_hook", "actuation", "emitted_at_utc"):
            _require_text(getattr(self, name), name)
        _validate_sha256(self.policy_sha256, "policy_sha256")
        if self.state.policy_sha256 != self.policy_sha256:
            raise CausalManifestError("decision policy digest must match its state policy digest")
        if self.policy_mode not in {"deterministic", "randomized"}:
            raise CausalManifestError("policy_mode must be deterministic or randomized")
        if self.exploration_hook not in {"disabled_pending_operator_go", "externally_authorized"}:
            raise CausalManifestError("exploration_hook is not a registered authorization state")
        if self.executed == (self.actuation == "NONE"):
            raise CausalManifestError("executed decisions require actuation; advisory decisions require NONE")
        arms = [item.arm_id for item in self.arm_propensities]
        if not arms or len(arms) != len(set(arms)):
            raise CausalManifestError("arm_propensities must contain unique alternatives")
        if self.chosen_arm not in arms:
            raise CausalManifestError("chosen_arm must occur in arm_propensities")
        total = sum(float(item.propensity) for item in self.arm_propensities)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise CausalManifestError("arm propensities must sum to one")
        chosen_p = next(item.propensity for item in self.arm_propensities if item.arm_id == self.chosen_arm)
        if chosen_p <= 0.0:
            raise CausalManifestError("chosen_arm must have positive propensity")
        if self.policy_mode == "deterministic":
            if chosen_p != 1.0 or any(item.propensity != 0.0 for item in self.arm_propensities if item.arm_id != self.chosen_arm):
                raise CausalManifestError("deterministic decisions require propensity one for the chosen arm")
            if self.random_seed is not None or self.random_draw is not None:
                raise CausalManifestError("deterministic decisions may not claim a random draw")
        else:
            if self.exploration_hook != "externally_authorized":
                raise CausalManifestError("randomized decisions require an externally_authorized hook")
            if self.random_seed is None or self.random_draw is None:
                raise CausalManifestError("randomized decisions require the actual seed and draw")
            if not isinstance(self.random_seed, int):
                raise CausalManifestError("random_seed must be an int")
            if not 0.0 <= float(self.random_draw) < 1.0:
                raise CausalManifestError("random_draw must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "row_kind": self.row_kind,
            "row_id": self.row_id,
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "state": self.state.to_dict(),
            "chosen_arm": self.chosen_arm,
            "arm_propensities": [item.to_dict() for item in self.arm_propensities],
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "policy_mode": self.policy_mode,
            "exploration_hook": self.exploration_hook,
            "executed": self.executed,
            "actuation": self.actuation,
            "random_seed": self.random_seed,
            "random_draw": self.random_draw,
            "emitted_at_utc": self.emitted_at_utc,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ExplorationDecisionRow:
        return cls(
            row_id=str(row["row_id"]),
            decision_id=str(row["decision_id"]),
            run_id=str(row["run_id"]),
            state=StateSummary.from_dict(row["state"]),
            chosen_arm=str(row["chosen_arm"]),
            arm_propensities=tuple(ArmPropensity.from_dict(item) for item in row["arm_propensities"]),
            policy_id=str(row["policy_id"]),
            policy_sha256=str(row["policy_sha256"]),
            policy_mode=str(row["policy_mode"]),
            exploration_hook=str(row["exploration_hook"]),
            executed=bool(row["executed"]),
            actuation=str(row["actuation"]),
            random_seed=row.get("random_seed"),
            random_draw=row.get("random_draw"),
            emitted_at_utc=str(row["emitted_at_utc"]),
            schema_id=str(row["schema_id"]),
            row_kind=str(row["row_kind"]),
        )


@dataclass(frozen=True)
class ActionSupport:
    arm_id: str
    minimum_propensity: float

    def __post_init__(self) -> None:
        _require_text(self.arm_id, "ActionSupport.arm_id")
        if not 0.0 < float(self.minimum_propensity) <= 1.0:
            raise CausalManifestError("minimum_propensity must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "minimum_propensity": self.minimum_propensity}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ActionSupport:
        return cls(arm_id=str(row["arm_id"]), minimum_propensity=float(row["minimum_propensity"]))


@dataclass(frozen=True)
class CoverageReceiptRow:
    row_id: str
    receipt_id: str
    run_id: str
    target_policy_id: str
    target_policy_sha256: str
    working_support_id: str
    initial_state_covered: bool
    one_step_target_covered: bool
    action_support: tuple[ActionSupport, ...]
    assessment_method: str
    evidence: tuple[str, ...]
    verdict_scope: str
    emitted_at_utc: str
    schema_id: str = SCHEMA_ID
    row_kind: str = ROW_COVERAGE_RECEIPT

    def __post_init__(self) -> None:
        _validate_row_header(self.schema_id, self.row_kind, self.row_id, self.run_id, ROW_COVERAGE_RECEIPT)
        for name in (
            "receipt_id",
            "target_policy_id",
            "working_support_id",
            "assessment_method",
            "verdict_scope",
            "emitted_at_utc",
        ):
            _require_text(getattr(self, name), name)
        _validate_sha256(self.target_policy_sha256, "target_policy_sha256")
        arms = [item.arm_id for item in self.action_support]
        if len(arms) != len(set(arms)):
            raise CausalManifestError("coverage action_support arms must be unique")
        if not self.evidence:
            raise CausalManifestError("coverage receipts require evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "row_kind": self.row_kind,
            "row_id": self.row_id,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "target_policy_id": self.target_policy_id,
            "target_policy_sha256": self.target_policy_sha256,
            "working_support_id": self.working_support_id,
            "initial_state_covered": self.initial_state_covered,
            "one_step_target_covered": self.one_step_target_covered,
            "action_support": [item.to_dict() for item in self.action_support],
            "assessment_method": self.assessment_method,
            "evidence": list(self.evidence),
            "verdict_scope": self.verdict_scope,
            "emitted_at_utc": self.emitted_at_utc,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> CoverageReceiptRow:
        return cls(
            row_id=str(row["row_id"]),
            receipt_id=str(row["receipt_id"]),
            run_id=str(row["run_id"]),
            target_policy_id=str(row["target_policy_id"]),
            target_policy_sha256=str(row["target_policy_sha256"]),
            working_support_id=str(row["working_support_id"]),
            initial_state_covered=bool(row["initial_state_covered"]),
            one_step_target_covered=bool(row["one_step_target_covered"]),
            action_support=tuple(ActionSupport.from_dict(item) for item in row.get("action_support", ())),
            assessment_method=str(row["assessment_method"]),
            evidence=tuple(str(item) for item in row["evidence"]),
            verdict_scope=str(row["verdict_scope"]),
            emitted_at_utc=str(row["emitted_at_utc"]),
            schema_id=str(row["schema_id"]),
            row_kind=str(row["row_kind"]),
        )


def _require_sorted_unique(values: Sequence[Any], name: str) -> None:
    if tuple(values) != tuple(sorted(values)) or len(set(values)) != len(values):
        raise CausalManifestError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class ClassEdgeMark:
    winner_class: int
    other_class: int
    directed: bool
    junction_classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("winner_class", "other_class"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
                raise CausalManifestError(f"ClassEdgeMark.{name} must be an int in [0,4]")
        if self.winner_class == self.other_class:
            raise CausalManifestError("ClassEdgeMark requires two distinct classes")
        _require_sorted_unique(self.junction_classes, "ClassEdgeMark.junction_classes")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4
            for value in self.junction_classes
        ):
            raise CausalManifestError("junction_classes values must be ints in [0,4]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_class": self.winner_class,
            "other_class": self.other_class,
            "directed": self.directed,
            "junction_classes": list(self.junction_classes),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ClassEdgeMark:
        return cls(
            winner_class=int(row["winner_class"]),
            other_class=int(row["other_class"]),
            directed=bool(row["directed"]),
            junction_classes=tuple(int(value) for value in row.get("junction_classes", ())),
        )


@dataclass(frozen=True)
class SpacetimeMark:
    coordinate_system: str
    y: float
    x: float
    time_fraction: float
    support_y0: float
    support_x0: float
    support_y1: float
    support_x1: float
    chart_id: str
    location_quantizer_id: str

    def __post_init__(self) -> None:
        if self.coordinate_system not in {"scorer_grid", "camera_grid", "latent_chart"}:
            raise CausalManifestError("SpacetimeMark.coordinate_system is invalid")
        for name in (
            "y",
            "x",
            "time_fraction",
            "support_y0",
            "support_x0",
            "support_y1",
            "support_x1",
        ):
            if _finite_or_none(getattr(self, name), f"SpacetimeMark.{name}") is None:
                raise CausalManifestError(f"SpacetimeMark.{name} must be finite")
        if not 0.0 <= float(self.time_fraction) <= 1.0:
            raise CausalManifestError("SpacetimeMark.time_fraction must be in [0,1]")
        if not self.support_y0 <= self.y <= self.support_y1:
            raise CausalManifestError("SpacetimeMark.y must lie inside the support box")
        if not self.support_x0 <= self.x <= self.support_x1:
            raise CausalManifestError("SpacetimeMark.x must lie inside the support box")
        _require_text(self.chart_id, "SpacetimeMark.chart_id")
        _require_text(self.location_quantizer_id, "SpacetimeMark.location_quantizer_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "coordinate_system",
                "y",
                "x",
                "time_fraction",
                "support_y0",
                "support_x0",
                "support_y1",
                "support_x1",
                "chart_id",
                "location_quantizer_id",
            )
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> SpacetimeMark:
        return cls(
            coordinate_system=str(row["coordinate_system"]),
            y=float(row["y"]),
            x=float(row["x"]),
            time_fraction=float(row["time_fraction"]),
            support_y0=float(row["support_y0"]),
            support_x0=float(row["support_x0"]),
            support_y1=float(row["support_y1"]),
            support_x1=float(row["support_x1"]),
            chart_id=str(row["chart_id"]),
            location_quantizer_id=str(row["location_quantizer_id"]),
        )


@dataclass(frozen=True)
class IncidenceMark:
    before_component_ids: tuple[str, ...] = ()
    after_component_ids: tuple[str, ...] = ()
    before_junction_ids: tuple[str, ...] = ()
    after_junction_ids: tuple[str, ...] = ()
    parent_child_edges: tuple[tuple[str, str], ...] = ()
    incidence_before_sha256: DigestRef | None = None
    incidence_after_sha256: DigestRef | None = None
    attachment_rule_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "before_component_ids",
            "after_component_ids",
            "before_junction_ids",
            "after_junction_ids",
            "parent_child_edges",
        ):
            values = getattr(self, name)
            _require_sorted_unique(values, f"IncidenceMark.{name}")
        for edge in self.parent_child_edges:
            if len(edge) != 2 or not all(isinstance(item, str) and item for item in edge):
                raise CausalManifestError("parent_child_edges must contain nonempty string pairs")
        _require_text(self.attachment_rule_id, "IncidenceMark.attachment_rule_id")
        if not any(
            (
                self.before_component_ids,
                self.after_component_ids,
                self.before_junction_ids,
                self.after_junction_ids,
                self.parent_child_edges,
                self.incidence_before_sha256,
                self.incidence_after_sha256,
            )
        ):
            raise CausalManifestError("IncidenceMark rejects count-only/attachment-free events")

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_component_ids": list(self.before_component_ids),
            "after_component_ids": list(self.after_component_ids),
            "before_junction_ids": list(self.before_junction_ids),
            "after_junction_ids": list(self.after_junction_ids),
            "parent_child_edges": [list(edge) for edge in self.parent_child_edges],
            "incidence_before_sha256": (
                None if self.incidence_before_sha256 is None else self.incidence_before_sha256.to_dict()
            ),
            "incidence_after_sha256": (
                None if self.incidence_after_sha256 is None else self.incidence_after_sha256.to_dict()
            ),
            "attachment_rule_id": self.attachment_rule_id,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> IncidenceMark:
        before_digest = row.get("incidence_before_sha256")
        after_digest = row.get("incidence_after_sha256")
        return cls(
            before_component_ids=tuple(str(value) for value in row.get("before_component_ids", ())),
            after_component_ids=tuple(str(value) for value in row.get("after_component_ids", ())),
            before_junction_ids=tuple(str(value) for value in row.get("before_junction_ids", ())),
            after_junction_ids=tuple(str(value) for value in row.get("after_junction_ids", ())),
            parent_child_edges=tuple(
                (str(edge[0]), str(edge[1])) for edge in row.get("parent_child_edges", ())
            ),
            incidence_before_sha256=(
                None if before_digest is None else DigestRef.from_dict(before_digest)
            ),
            incidence_after_sha256=(
                None if after_digest is None else DigestRef.from_dict(after_digest)
            ),
            attachment_rule_id=str(row["attachment_rule_id"]),
        )


@dataclass(frozen=True)
class StratumMark:
    topology_signature_id: str
    orbit_stabilizer_chart_id: str
    activation_chart_id: str
    receiver_phase_cell_id: str

    def __post_init__(self) -> None:
        for name in (
            "topology_signature_id",
            "orbit_stabilizer_chart_id",
            "activation_chart_id",
            "receiver_phase_cell_id",
        ):
            _require_text(getattr(self, name), f"StratumMark.{name}")

    def to_dict(self) -> dict[str, str]:
        return {
            "topology_signature_id": self.topology_signature_id,
            "orbit_stabilizer_chart_id": self.orbit_stabilizer_chart_id,
            "activation_chart_id": self.activation_chart_id,
            "receiver_phase_cell_id": self.receiver_phase_cell_id,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> StratumMark:
        return cls(**{name: str(row[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ReceiverStateMark:
    R_operator_id: str
    uint8_rounding_id: str
    xi_quantizer_id: str
    xi_symbol: tuple[int, ...]
    phase_symbol: int | None
    prediction_chart_id: str

    def __post_init__(self) -> None:
        for name in (
            "R_operator_id",
            "uint8_rounding_id",
            "xi_quantizer_id",
            "prediction_chart_id",
        ):
            _require_text(getattr(self, name), f"ReceiverStateMark.{name}")
        _require_sorted_unique(self.xi_symbol, "ReceiverStateMark.xi_symbol")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.xi_symbol):
            raise CausalManifestError("ReceiverStateMark.xi_symbol must contain ints")
        if self.phase_symbol is not None and (
            not isinstance(self.phase_symbol, int) or isinstance(self.phase_symbol, bool)
        ):
            raise CausalManifestError("ReceiverStateMark.phase_symbol must be int or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "R_operator_id": self.R_operator_id,
            "uint8_rounding_id": self.uint8_rounding_id,
            "xi_quantizer_id": self.xi_quantizer_id,
            "xi_symbol": list(self.xi_symbol),
            "phase_symbol": self.phase_symbol,
            "prediction_chart_id": self.prediction_chart_id,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ReceiverStateMark:
        return cls(
            R_operator_id=str(row["R_operator_id"]),
            uint8_rounding_id=str(row["uint8_rounding_id"]),
            xi_quantizer_id=str(row["xi_quantizer_id"]),
            xi_symbol=tuple(int(value) for value in row["xi_symbol"]),
            phase_symbol=None if row.get("phase_symbol") is None else int(row["phase_symbol"]),
            prediction_chart_id=str(row["prediction_chart_id"]),
        )


def _detector_family(detector: str) -> str:
    token = _require_text(detector, "detectors_matched item")
    prefix = token.split(":", 1)[0]
    if prefix in EVENT_FAMILY_KINDS:
        return prefix
    for family, kinds in EVENT_FAMILY_KINDS.items():
        if token in kinds:
            return family
    raise CausalManifestError(
        f"detector {detector!r} must be family-prefixed or a registered event kind"
    )


@dataclass(frozen=True)
class EventMarkRow:
    event_id: str
    run_id: str
    stage_id: str
    checkpoint_id: str | None
    pair_index: int
    frame_from: int
    frame_to: int
    observed_at_utc: str
    authority_axis: str
    family: str
    kind: str
    detectors_matched: tuple[str, ...]
    class_edge: ClassEdgeMark
    location: SpacetimeMark
    attachment: IncidenceMark
    stratum_before: StratumMark
    stratum_after: StratumMark
    receiver_state: ReceiverStateMark
    evidence: tuple[ArtifactRef, ...]
    receiver_derivable: bool
    public_derivation_ref: ArtifactRef | None
    resume_key: str
    notes: tuple[JsonField, ...] = ()
    schema_id: str = SCHEMA_ID
    row_kind: str = ROW_EVENT_MARK

    @property
    def row_id(self) -> str:
        return f"event:{self.event_id}"

    def __post_init__(self) -> None:
        _validate_row_header(self.schema_id, self.row_kind, self.row_id, self.run_id, ROW_EVENT_MARK)
        _validate_sha256(self.event_id, "EventMarkRow.event_id")
        _require_text(self.stage_id, "EventMarkRow.stage_id")
        if self.checkpoint_id is not None:
            _require_text(self.checkpoint_id, "EventMarkRow.checkpoint_id")
        if self.pair_index < 0 or self.frame_from < 0 or self.frame_to <= self.frame_from:
            raise CausalManifestError(
                "EventMarkRow requires pair_index/frame_from >=0 and frame_to > frame_from"
            )
        _require_canonical_utc(self.observed_at_utc, "EventMarkRow.observed_at_utc")
        if self.authority_axis != NON_PROMOTABLE_AXIS:
            raise CausalManifestError(
                f"EventMarkRow.authority_axis must equal {NON_PROMOTABLE_AXIS!r}"
            )
        if self.family not in EVENT_FAMILY_KINDS:
            raise CausalManifestError(f"unknown event family {self.family!r}")
        if self.kind not in EVENT_FAMILY_KINDS[self.family]:
            raise CausalManifestError(
                f"event kind {self.kind!r} is not valid for family {self.family!r}"
            )
        if not self.detectors_matched:
            raise CausalManifestError("EventMarkRow.detectors_matched must be nonempty")
        _require_sorted_unique(self.detectors_matched, "EventMarkRow.detectors_matched")
        selected_family = min(
            (_detector_family(item) for item in self.detectors_matched),
            key=EVENT_FAMILY_PRIORITY.__getitem__,
        )
        if selected_family != self.family:
            raise CausalManifestError(
                f"event priority requires family={selected_family!r}, got {self.family!r}"
            )
        evidence_keys = tuple(canonical_json(item.to_dict()) for item in self.evidence)
        _require_sorted_unique(evidence_keys, "EventMarkRow.evidence")
        _validate_fields(self.notes, "EventMarkRow.notes")
        if self.receiver_derivable != (self.public_derivation_ref is not None):
            raise CausalManifestError(
                "receiver_derivable requires exactly one public_derivation_ref; "
                "non-derivable marks must leave it null"
            )
        expected_id = canonical_sha256(self.identity_payload())
        if self.event_id != expected_id:
            raise CausalManifestError(
                f"EventMarkRow.event_id mismatch: expected {expected_id}, got {self.event_id}"
            )
        expected_resume = (
            f"{self.run_id}/{self.stage_id}/{self.pair_index}/{self.frame_to}/{self.event_id}"
        )
        if self.resume_key != expected_resume:
            raise CausalManifestError("EventMarkRow.resume_key does not match canonical cursor")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "pair_index": self.pair_index,
            "frame_from": self.frame_from,
            "frame_to": self.frame_to,
            "family": self.family,
            "kind": self.kind,
            "class_edge": self.class_edge.to_dict(),
            "location": self.location.to_dict(),
            "attachment": self.attachment.to_dict(),
        }

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        stage_id: str,
        checkpoint_id: str | None,
        pair_index: int,
        frame_from: int,
        frame_to: int,
        observed_at_utc: str,
        family: str,
        kind: str,
        detectors_matched: Sequence[str],
        class_edge: ClassEdgeMark,
        location: SpacetimeMark,
        attachment: IncidenceMark,
        stratum_before: StratumMark,
        stratum_after: StratumMark,
        receiver_state: ReceiverStateMark,
        evidence: Sequence[ArtifactRef],
        receiver_derivable: bool,
        public_derivation_ref: ArtifactRef | None,
        notes: Mapping[str, Any] | None = None,
    ) -> EventMarkRow:
        detectors = tuple(sorted({str(item) for item in detectors_matched}))
        sorted_evidence = tuple(
            sorted(evidence, key=lambda item: canonical_json(item.to_dict()))
        )
        identity = {
            "run_id": run_id,
            "stage_id": stage_id,
            "pair_index": int(pair_index),
            "frame_from": int(frame_from),
            "frame_to": int(frame_to),
            "family": family,
            "kind": kind,
            "class_edge": class_edge.to_dict(),
            "location": location.to_dict(),
            "attachment": attachment.to_dict(),
        }
        event_id = canonical_sha256(identity)
        resume_key = f"{run_id}/{stage_id}/{int(pair_index)}/{int(frame_to)}/{event_id}"
        return cls(
            event_id=event_id,
            run_id=run_id,
            stage_id=stage_id,
            checkpoint_id=checkpoint_id,
            pair_index=int(pair_index),
            frame_from=int(frame_from),
            frame_to=int(frame_to),
            observed_at_utc=observed_at_utc,
            authority_axis=NON_PROMOTABLE_AXIS,
            family=family,
            kind=kind,
            detectors_matched=detectors,
            class_edge=class_edge,
            location=location,
            attachment=attachment,
            stratum_before=stratum_before,
            stratum_after=stratum_after,
            receiver_state=receiver_state,
            evidence=sorted_evidence,
            receiver_derivable=bool(receiver_derivable),
            public_derivation_ref=public_derivation_ref,
            resume_key=resume_key,
            notes=freeze_fields(notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "row_kind": self.row_kind,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "checkpoint_id": self.checkpoint_id,
            "pair_index": self.pair_index,
            "frame_from": self.frame_from,
            "frame_to": self.frame_to,
            "observed_at_utc": self.observed_at_utc,
            "authority_axis": self.authority_axis,
            "family": self.family,
            "kind": self.kind,
            "detectors_matched": list(self.detectors_matched),
            "class_edge": self.class_edge.to_dict(),
            "location": self.location.to_dict(),
            "attachment": self.attachment.to_dict(),
            "stratum_before": self.stratum_before.to_dict(),
            "stratum_after": self.stratum_after.to_dict(),
            "receiver_state": self.receiver_state.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "receiver_derivable": self.receiver_derivable,
            "public_derivation_ref": (
                None if self.public_derivation_ref is None else self.public_derivation_ref.to_dict()
            ),
            "resume_key": self.resume_key,
            "notes": [item.to_dict() for item in self.notes],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> EventMarkRow:
        forbidden = {"score", "score_claim", "promotable", "promotion_eligible"} & set(row)
        if forbidden:
            raise CausalManifestError(
                f"EventMarkRow rejects score/promotion fields: {sorted(forbidden)}"
            )
        public_ref = row.get("public_derivation_ref")
        return cls(
            event_id=str(row["event_id"]),
            run_id=str(row["run_id"]),
            stage_id=str(row["stage_id"]),
            checkpoint_id=row.get("checkpoint_id"),
            pair_index=int(row["pair_index"]),
            frame_from=int(row["frame_from"]),
            frame_to=int(row["frame_to"]),
            observed_at_utc=str(row["observed_at_utc"]),
            authority_axis=str(row["authority_axis"]),
            family=str(row["family"]),
            kind=str(row["kind"]),
            detectors_matched=tuple(str(value) for value in row["detectors_matched"]),
            class_edge=ClassEdgeMark.from_dict(row["class_edge"]),
            location=SpacetimeMark.from_dict(row["location"]),
            attachment=IncidenceMark.from_dict(row["attachment"]),
            stratum_before=StratumMark.from_dict(row["stratum_before"]),
            stratum_after=StratumMark.from_dict(row["stratum_after"]),
            receiver_state=ReceiverStateMark.from_dict(row["receiver_state"]),
            evidence=tuple(ArtifactRef.from_dict(item) for item in row["evidence"]),
            receiver_derivable=bool(row["receiver_derivable"]),
            public_derivation_ref=(
                None if public_ref is None else ArtifactRef.from_dict(public_ref)
            ),
            resume_key=str(row["resume_key"]),
            notes=tuple(JsonField.from_dict(item) for item in row.get("notes", ())),
            schema_id=str(row["schema_id"]),
            row_kind=str(row["row_kind"]),
        )


CausalRow: TypeAlias = (
    RunTreatmentManifest
    | BoundaryObservationRow
    | TransitionRow
    | ExplorationDecisionRow
    | CoverageReceiptRow
    | EventMarkRow
)


def _validate_row_header(schema_id: str, row_kind: str, row_id: str, run_id: str, expected_kind: str) -> None:
    if schema_id != SCHEMA_ID:
        raise CausalManifestError(f"unsupported schema_id {schema_id!r}")
    if row_kind != expected_kind or row_kind not in _ROW_KINDS:
        raise CausalManifestError(f"invalid row_kind {row_kind!r}; expected {expected_kind!r}")
    _require_text(row_id, "row_id")
    _require_text(run_id, "run_id")


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise CausalManifestError(f"{name} must be a 64-character SHA-256 hex digest")


def row_from_dict(row: Mapping[str, Any]) -> CausalRow:
    """Parse and validate one row, failing closed on unknown versions/kinds."""

    if row.get("schema_id") != SCHEMA_ID:
        raise CausalManifestError(f"unsupported schema_id {row.get('schema_id')!r}")
    kind = row.get("row_kind")
    parser = {
        ROW_RUN_MANIFEST: RunTreatmentManifest.from_dict,
        ROW_BOUNDARY: BoundaryObservationRow.from_dict,
        ROW_TRANSITION: TransitionRow.from_dict,
        ROW_EXPLORATION_DECISION: ExplorationDecisionRow.from_dict,
        ROW_COVERAGE_RECEIPT: CoverageReceiptRow.from_dict,
        ROW_EVENT_MARK: EventMarkRow.from_dict,
    }.get(kind)
    if parser is None:
        raise CausalManifestError(f"unknown causal-manifest row_kind {kind!r}")
    return parser(row)


def append_causal_row(path: str | Path, row: CausalRow) -> Path:
    """Validate then append one typed row via the canonical locked writer."""

    if not isinstance(
        row,
        (
            RunTreatmentManifest,
            BoundaryObservationRow,
            TransitionRow,
            ExplorationDecisionRow,
            CoverageReceiptRow,
            EventMarkRow,
        ),
    ):
        raise TypeError(f"unsupported causal row type {type(row).__name__}")
    target = Path(path)
    append_locked_jsonl(target, row.to_dict())
    return target


def load_causal_manifest(path: str | Path, *, strict: bool = True) -> tuple[CausalRow, ...]:
    """Load typed rows; strict mode rejects corrupt lines and duplicate immutable row ids."""

    target = Path(path)
    if not target.exists():
        return ()
    rows: list[CausalRow] = []
    seen: dict[str, str] = {}
    with target.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                parsed = row_from_dict(raw)
                encoded = canonical_json(parsed.to_dict())
                if parsed.row_id in seen:
                    if seen[parsed.row_id] != encoded:
                        raise CausalManifestConflictError(f"row_id {parsed.row_id!r} changed content")
                    if strict:
                        raise CausalManifestConflictError(f"duplicate row_id {parsed.row_id!r}")
                    continue
                seen[parsed.row_id] = encoded
                rows.append(parsed)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if strict:
                    raise CausalManifestError(f"invalid causal manifest line {line_number}: {exc}") from exc
    return tuple(rows)


def boundary_sequence_index(epoch: int, boundary_kind: str, *, ordinal: int = 0) -> int:
    """Deterministic causal ordering key; async late arrivals remain detectable."""

    if epoch < 0 or not 0 <= ordinal < 1000:
        raise CausalManifestError("epoch and ordinal must be >= 0")
    priority = _BOUNDARY_PRIORITY.get(boundary_kind, 9)
    return int(epoch) * 10_000 + priority * 1_000 + int(ordinal)


class CausalManifestWriter:
    """Thread-serialized, disk-resumable append facade for one run.

    Every boundary is retained.  A transition is emitted only when the new boundary is strictly
    later than the latest causal boundary.  Thus a late async verdict is not falsely described as
    ``Z -> Z'`` backward in time; it remains an honest boundary observation and FORE support refuses
    the missing transition.
    """

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = _require_text(run_id, "run_id")
        self._lock = threading.Lock()
        self._rows = list(load_causal_manifest(self.path, strict=True))
        self._by_id = {row.row_id: canonical_json(row.to_dict()) for row in self._rows}
        states = [row.state for row in self._rows if isinstance(row, BoundaryObservationRow) and row.run_id == run_id]
        self._latest_state = max(states, key=lambda state: state.sequence_index, default=None)

    def _append_idempotent(self, row: CausalRow) -> bool:
        encoded = canonical_json(row.to_dict())
        prior = self._by_id.get(row.row_id)
        if prior is not None:
            if prior != encoded:
                raise CausalManifestConflictError(f"row_id {row.row_id!r} already has different content")
            return False
        append_causal_row(self.path, row)
        self._rows.append(row)
        self._by_id[row.row_id] = encoded
        return True

    def ensure_run_manifest(self, manifest: RunTreatmentManifest) -> bool:
        if manifest.run_id != self.run_id:
            raise CausalManifestConflictError("writer run_id differs from manifest run_id")
        with self._lock:
            existing = [row for row in self._rows if isinstance(row, RunTreatmentManifest) and row.run_id == self.run_id]
            if existing and canonical_json(existing[0].to_dict()) != canonical_json(manifest.to_dict()):
                raise CausalManifestConflictError("run treatment manifest changed across resume")
            return self._append_idempotent(manifest)

    @property
    def run_manifest(self) -> RunTreatmentManifest | None:
        """Original immutable treatment manifest, if this run has emitted one."""

        with self._lock:
            return next(
                (
                    row
                    for row in self._rows
                    if isinstance(row, RunTreatmentManifest) and row.run_id == self.run_id
                ),
                None,
            )

    def record_boundary(
        self,
        state: StateSummary,
        *,
        action: ActionSummary,
        reward: RewardObservation,
        pair_id: str | None = None,
    ) -> TransitionRow | None:
        if action.policy_sha256 != state.policy_sha256:
            raise CausalManifestConflictError("boundary action and state policy digests differ")
        boundary = BoundaryObservationRow(
            row_id=f"boundary:{self.run_id}:{state.boundary_id}",
            run_id=self.run_id,
            state=state,
            emitted_at_utc=utc_now(),
        )
        with self._lock:
            self._append_idempotent(boundary)
            prior = self._latest_state
            if prior is None or state.sequence_index <= prior.sequence_index or state.epoch < prior.epoch:
                if prior is None or state.sequence_index > prior.sequence_index:
                    self._latest_state = state
                return None
            transition_id = f"{prior.boundary_id}->{state.boundary_id}"
            transition = TransitionRow(
                row_id=f"transition:{self.run_id}:{transition_id}",
                transition_id=transition_id,
                run_id=self.run_id,
                pair_id=pair_id,
                state=prior,
                action=action,
                reward=reward,
                next_state=state,
                emitted_at_utc=utc_now(),
            )
            self._append_idempotent(transition)
            self._latest_state = state
            return transition

    def record_decision(self, decision: ExplorationDecisionRow) -> bool:
        if decision.run_id != self.run_id:
            raise CausalManifestConflictError("writer run_id differs from decision run_id")
        with self._lock:
            return self._append_idempotent(decision)

    def record_coverage(self, receipt: CoverageReceiptRow) -> bool:
        if receipt.run_id != self.run_id:
            raise CausalManifestConflictError("writer run_id differs from coverage run_id")
        with self._lock:
            return self._append_idempotent(receipt)

    def record_event_mark(self, event: EventMarkRow) -> bool:
        """Append an immutable D39 event mark, idempotently across resume."""

        if event.run_id != self.run_id:
            raise CausalManifestConflictError("writer run_id differs from event-mark run_id")
        with self._lock:
            return self._append_idempotent(event)


@dataclass(frozen=True)
class FORESupportReport:
    admissible: bool
    status: str
    n_run_manifests: int
    n_transitions: int
    n_observed_rewards: int
    n_decisions: int
    blockers: tuple[str, ...]
    evidence: tuple[str, ...]
    verdict_scope: str = "FORMULATION x CACHE SUPPORT"


def check_fore_support(
    rows_or_path: Iterable[CausalRow] | str | Path,
    *,
    target_policy_sha256: str,
    target_arms: Sequence[str],
) -> FORESupportReport:
    """Fail-closed structural admission for FORE round-3/round-4 caches."""

    _validate_sha256(target_policy_sha256, "target_policy_sha256")
    rows = _coerce_rows(rows_or_path)
    manifests = [row for row in rows if isinstance(row, RunTreatmentManifest)]
    transitions = [row for row in rows if isinstance(row, TransitionRow)]
    decisions = [row for row in rows if isinstance(row, ExplorationDecisionRow)]
    receipts = [
        row
        for row in rows
        if isinstance(row, CoverageReceiptRow) and row.target_policy_sha256 == target_policy_sha256
    ]
    blockers: list[str] = []
    if not manifests:
        blockers.append("missing_run_treatment_manifest")
    if not transitions:
        blockers.append("missing_state_action_reward_successor_transitions")
    observed = [row for row in transitions if row.reward.observed]
    if transitions and len(observed) != len(transitions):
        blockers.append("transition_rewards_not_fully_observed")
    if transitions and any(row.action.policy_sha256 != target_policy_sha256 for row in transitions):
        blockers.append("transition_policy_hash_mismatch")
    receipt = receipts[-1] if receipts else None
    if receipt is None:
        blockers.append("missing_explicit_coverage_receipt")
        supported: dict[str, float] = {}
    else:
        if not receipt.initial_state_covered:
            blockers.append("initial_state_coverage_not_established")
        if not receipt.one_step_target_covered:
            blockers.append("one_step_target_coverage_not_established")
        supported = {item.arm_id: item.minimum_propensity for item in receipt.action_support}
    for arm in target_arms:
        if supported.get(arm, 0.0) <= 0.0:
            blockers.append(f"target_action_not_supported:{arm}")
    executed = [row for row in decisions if row.executed and row.policy_sha256 == target_policy_sha256]
    if target_arms and not executed:
        blockers.append("no_executed_decision_rows_for_target_policy")
    elif executed:
        positive = {
            prop.arm_id
            for row in executed
            for prop in row.arm_propensities
            if prop.propensity > 0.0
        }
        for arm in target_arms:
            if arm not in positive:
                blockers.append(f"no_positive_logged_propensity:{arm}")
    unique = tuple(dict.fromkeys(blockers))
    return FORESupportReport(
        admissible=not unique,
        status="ADMISSIBLE_STRUCTURAL_INPUT" if not unique else "NOT_IDENTIFIED",
        n_run_manifests=len(manifests),
        n_transitions=len(transitions),
        n_observed_rewards=len(observed),
        n_decisions=len(decisions),
        blockers=unique,
        evidence=tuple(row.row_id for row in (*manifests, *transitions, *receipts)),
    )


@dataclass(frozen=True)
class HCMNegativeControlMoment:
    name: str
    mean_moment: float
    whole_run_se: float
    standardized_abs: float


@dataclass(frozen=True)
class HCML4ResidualReport:
    status: str
    fired: bool
    unconfounded_certificate: bool
    n_transitions: int
    n_runs: int
    closure_max_abs: float | None
    positive_control_seen: bool
    positive_control_triggered: bool
    moments: tuple[HCMNegativeControlMoment, ...]
    blockers: tuple[str, ...]
    verdict_scope: str = "OBSERVATIONAL GRAPH FALSIFICATION"


def hcm_l4_residual_check(
    rows_or_path: Iterable[CausalRow] | str | Path,
    *,
    cross_fitted_predictions: Mapping[str, float],
    negative_control_names: Sequence[str],
    closure_tolerance: float = 1e-8,
    moment_z_threshold: float = 3.0,
    positive_control_threshold: float = 1e-6,
) -> HCML4ResidualReport:
    """Read-only HCM-L4 residual/term-closure skeleton for equations (2)--(5).

    Quiet output explicitly is *not* an unconfounded certificate.  Rows are grouped by whole run;
    callers must supply predictions produced out of fold.  The function never fits a model or
    invents a p-value.
    """

    rows = _coerce_rows(rows_or_path)
    manifests = [row for row in rows if isinstance(row, RunTreatmentManifest)]
    transitions = [row for row in rows if isinstance(row, TransitionRow)]
    if not transitions:
        return HCML4ResidualReport(
            status="NO_ROWS",
            fired=False,
            unconfounded_certificate=False,
            n_transitions=0,
            n_runs=0,
            closure_max_abs=None,
            positive_control_seen=False,
            positive_control_triggered=False,
            moments=(),
            blockers=("no_transition_rows",),
        )
    blockers: list[str] = []
    if not negative_control_names:
        blockers.append("no_preregistered_negative_controls")
    manifest_by_run = {row.run_id: row for row in manifests}
    for run_id in sorted({row.run_id for row in transitions}):
        manifest = manifest_by_run.get(run_id)
        if manifest is None:
            blockers.append(f"missing_run_treatment_manifest:{run_id}")
            continue
        for artifact in (*manifest.scorer_artifacts, *manifest.cache_artifacts):
            if artifact.digest_status != "captured" or artifact.digest is None:
                blockers.append(f"missing_source_hash:{run_id}:{artifact.role}")
    residuals: dict[str, float] = {}
    closure: list[float] = []
    for row in transitions:
        if row.pair_id in {None, "", "__aggregate_all_pairs__"}:
            blockers.append(f"missing_pair_outcome_custody:{row.row_id}")
        if not row.reward.observed or row.reward.value is None:
            blockers.append(f"unobserved_reward:{row.row_id}")
            continue
        pred = cross_fitted_predictions.get(row.row_id)
        if pred is None or not isinstance(pred, (int, float)) or not math.isfinite(float(pred)):
            blockers.append(f"missing_cross_fitted_prediction:{row.row_id}")
            continue
        residuals[row.row_id] = float(row.reward.value) - float(pred)
        closure_value = row.next_state.apparatus.closure_residual()
        if closure_value is None:
            blockers.append(f"missing_loss_closure:{row.row_id}")
        else:
            closure.append(abs(float(closure_value)))
    run_ids = sorted({row.run_id for row in transitions})
    if len(run_ids) < 2:
        blockers.append("whole_run_calibration_requires_at_least_two_runs")
    closure_max = max(closure) if closure else None
    if closure_max is not None and closure_max > closure_tolerance:
        blockers.append(f"loss_term_closure_failed:{closure_max:.6g}")
    positive_rows = [row for row in transitions if row.next_state.apparatus.positive_control == "frozen_no_update"]
    positive_seen = bool(positive_rows)
    if not positive_seen:
        blockers.append("missing_frozen_no_update_positive_control")
    positive_triggered = any(
        abs(residuals.get(row.row_id, 0.0)) > positive_control_threshold for row in positive_rows
    )
    if positive_seen and not positive_triggered:
        blockers.append("positive_control_did_not_trigger")

    moments: list[HCMNegativeControlMoment] = []
    for name in negative_control_names:
        per_run: list[float] = []
        for run_id in run_ids:
            values: list[float] = []
            for row in transitions:
                if row.run_id != run_id or row.row_id not in residuals:
                    continue
                control = row.next_state.apparatus.negative_control(name)
                if control is not None:
                    values.append(residuals[row.row_id] * control)
            if values:
                per_run.append(sum(values) / len(values))
        if len(per_run) < 2:
            blockers.append(f"negative_control_missing_whole_run_support:{name}")
            continue
        mean = sum(per_run) / len(per_run)
        variance = sum((value - mean) ** 2 for value in per_run) / (len(per_run) - 1)
        se = math.sqrt(variance / len(per_run))
        z = abs(mean) / se if se > 0.0 else (float("inf") if mean != 0.0 else 0.0)
        moments.append(HCMNegativeControlMoment(name=name, mean_moment=mean, whole_run_se=se, standardized_abs=z))

    invalid_prefixes = (
        "unobserved_reward:",
        "missing_cross_fitted_prediction:",
        "missing_loss_closure:",
        "missing_pair_outcome_custody:",
        "missing_run_treatment_manifest:",
        "missing_source_hash:",
        "no_preregistered_negative_controls",
        "whole_run_calibration_",
        "missing_frozen_no_update_",
        "negative_control_missing_",
    )
    invalid = any(item.startswith(invalid_prefixes) for item in blockers)
    refused = invalid or any(item.startswith("loss_term_closure_failed:") for item in blockers) or (
        positive_seen and not positive_triggered
    )
    moment_fired = any(item.standardized_abs >= moment_z_threshold for item in moments)
    if invalid:
        status = "INVALID_INPUT"
    elif refused:
        status = "REFUSED_APPARATUS"
    elif moment_fired:
        status = "FIRED_GRAPH_FALSIFICATION"
    else:
        status = "QUIET_NOT_CERTIFIED"
    return HCML4ResidualReport(
        status=status,
        fired=bool(moment_fired or refused),
        unconfounded_certificate=False,
        n_transitions=len(transitions),
        n_runs=len(run_ids),
        closure_max_abs=closure_max,
        positive_control_seen=positive_seen,
        positive_control_triggered=positive_triggered,
        moments=tuple(moments),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _coerce_rows(rows_or_path: Iterable[CausalRow] | str | Path) -> tuple[CausalRow, ...]:
    if isinstance(rows_or_path, (str, Path)):
        return load_causal_manifest(rows_or_path, strict=True)
    return tuple(rows_or_path)


def unavailable_artifact(role: str, uri: str, reason: str, *, deferred_cost: bool = False) -> ArtifactRef:
    """Construct explicit missing-custody metadata without placeholder strings."""

    return ArtifactRef(
        role=role,
        uri=uri,
        digest=None,
        digest_status="deferred_cost" if deferred_cost else "unavailable",
        unavailable_reason=reason,
    )


__all__ = [
    "EVENT_FAMILY_KINDS",
    "EVENT_FAMILY_PRIORITY",
    "MANIFEST_FILENAME",
    "NON_PROMOTABLE_AXIS",
    "SCHEMA_ID",
    "ActionSummary",
    "ActionSupport",
    "ApparatusState",
    "ArmPropensity",
    "ArtifactRef",
    "BoundaryObservationRow",
    "CausalManifestConflictError",
    "CausalManifestError",
    "CausalManifestWriter",
    "ClassEdgeMark",
    "CoverageReceiptRow",
    "DigestRef",
    "EventMarkRow",
    "ExplorationDecisionRow",
    "FORESupportReport",
    "HCML4ResidualReport",
    "HCMNegativeControlMoment",
    "IncidenceMark",
    "JsonField",
    "LossTerm",
    "RealizedOutcome",
    "ReceiverStateMark",
    "RewardObservation",
    "RunTreatmentManifest",
    "SpacetimeMark",
    "StagePlanEntry",
    "StateSummary",
    "StratumMark",
    "TransitionRow",
    "append_causal_row",
    "boundary_sequence_index",
    "canonical_json",
    "canonical_sha256",
    "check_fore_support",
    "freeze_fields",
    "hcm_l4_residual_check",
    "load_causal_manifest",
    "row_from_dict",
    "sha256_file",
    "thaw_fields",
    "unavailable_artifact",
    "utc_now",
]
