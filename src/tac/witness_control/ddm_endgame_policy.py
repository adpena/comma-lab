# SPDX-License-Identifier: MIT
"""Pure, typed DDM endgame stage-exit policy.

The policy compares measured score gain per measured wall time at one exact
operating point.  It deliberately does not turn NCDE, saddle/grokking labels,
or a v17 validity ratio into an actuator.  Those values are serialized as
advisory metadata only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

ARCHIVE_DENOMINATOR = 37_545_489
BYTE_SCORE_WEIGHT = 25.0
BYTE_SCORE_PRICE = BYTE_SCORE_WEIGHT / ARCHIVE_DENOMINATOR
OFFICIAL_DISPLAYED_BAR = 0.172
SUB015_BAR = 0.15
POLICY_SCHEMA = "ddm_endgame_policy.v1"
DECISION_SCHEMA = "ddm_endgame_policy.decision.v1"
ARITHMETIC_SCHEMA = "ddm_eg1_policy_arithmetic.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCORE_TOLERANCE = 1e-12
_OPERATING_POINT_PAYLOAD_KEYS = frozenset(
    {
        "stage_id",
        "checkpoint_sha256",
        "archive_sha256",
        "receiver_sha256",
        "d_seg",
        "per_class_d_seg",
        "d_pose",
        "archive_bytes",
        "score",
        "n_pairs",
        "hard_parsed",
        "receiver_realized",
        "topology_stable",
        "transitions_pending",
        "topology_signature",
        "evidence_axis",
        "verdict_scope",
    }
)
_ACTION_QUOTE_PAYLOAD_KEYS = frozenset(
    {
        "quote_id",
        "kind",
        "parent_checkpoint_sha256",
        "parent_archive_sha256",
        "parent_receiver_sha256",
        "n_pairs",
        "gain_lower",
        "gain_upper",
        "wall_seconds_lower",
        "wall_seconds_upper",
        "rate_lower",
        "rate_upper",
        "measured",
        "hard_parsed",
        "receiver_realized",
        "admissible",
        "verdict_scope",
        "evidence_axis",
        "endpoint_d_seg",
        "endpoint_d_pose",
        "endpoint_archive_bytes",
        "candidate_evaluations",
    }
)


class EndgamePolicyError(ValueError):
    """Malformed policy input or inconsistent same-object evidence."""


class ActionKind(StrEnum):
    """Actions whose measured economics can be compared at a stage exit."""

    TRAIN_WINDOW = "TRAIN_WINDOW"
    SEG_GN = "SEG_GN"
    QDBS = "QDBS"
    TERMINAL_POSE = "TERMINAL_POSE"


class DecisionAction(StrEnum):
    """Fail-closed result of the pure stage-exit policy."""

    CONTINUE_BOUNDED_WINDOW = "CONTINUE_BOUNDED_WINDOW"
    MEASURE_FINISHER_QUOTE = "MEASURE_FINISHER_QUOTE"
    HANDOFF_SEG_GN = "HANDOFF_SEG_GN"
    HANDOFF_QDBS = "HANDOFF_QDBS"
    HANDOFF_TERMINAL_POSE = "HANDOFF_TERMINAL_POSE"
    R6_EXACT_EVAL = "R6_EXACT_EVAL"
    REFUSE_INSUFFICIENT_EVIDENCE = "REFUSE_INSUFFICIENT_EVIDENCE"


class TrajectoryRegime(StrEnum):
    """Advisory #216/#475 trajectory label; never an actuator by itself."""

    UNKNOWN = "UNKNOWN"
    SADDLE_STAIRCASE = "SADDLE_STAIRCASE"
    SMOOTH_POWERLAW = "SMOOTH_POWERLAW"
    FIXED_QUADRATIC_TERMINAL = "FIXED_QUADRATIC_TERMINAL"


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EndgamePolicyError(f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: object) -> str:
    text = _require_nonempty(name, value)
    if _SHA256_RE.fullmatch(text) is None:
        raise EndgamePolicyError(f"{name} must be a lowercase 64-hex SHA-256")
    return text


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise EndgamePolicyError(f"{name} must be a bool")
    return value


def _require_finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise EndgamePolicyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EndgamePolicyError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise EndgamePolicyError(f"{name} must be a finite number")
    if minimum is not None and number < minimum:
        raise EndgamePolicyError(f"{name} must be >= {minimum}")
    return number


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EndgamePolicyError(f"{name} must be an integer >= {minimum}")
    return value


def _reject_unexpected_keys(name: str, payload: Mapping[object, Any], allowed: Collection[str]) -> None:
    unexpected = [key for key in payload if key not in allowed]
    if unexpected:
        rendered = ", ".join(sorted(repr(key) for key in unexpected))
        raise EndgamePolicyError(f"{name} contains unexpected keys: {rendered}")


def contest_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """Return the exact contest objective decomposition for one measured row."""

    seg = _require_finite("d_seg", d_seg, minimum=0.0)
    pose = _require_finite("d_pose", d_pose, minimum=0.0)
    size = _require_int("archive_bytes", archive_bytes)
    return 100.0 * seg + math.sqrt(10.0 * pose) + BYTE_SCORE_PRICE * size


def strict_integer_byte_ceiling(target: float, d_seg: float, d_pose: float) -> int | None:
    """Largest integer B satisfying ``contest_score(d_seg, d_pose, B) < target``.

    ``None`` means the distortion terms already meet or exceed the target at
    zero bytes.
    """

    bar = _require_finite("target", target, minimum=0.0)
    distortion = contest_score(d_seg, d_pose, 0)
    raw = (bar - distortion) / BYTE_SCORE_PRICE
    if raw <= 0.0:
        return None
    ceiling = max(0, math.ceil(raw) - 1)
    while ceiling >= 0 and contest_score(d_seg, d_pose, ceiling) >= bar:
        ceiling -= 1
    while contest_score(d_seg, d_pose, ceiling + 1) < bar:
        ceiling += 1
    return ceiling


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Exact stage-exit state whose identity every quote must bind."""

    stage_id: str
    checkpoint_sha256: str
    archive_sha256: str
    receiver_sha256: str
    d_seg: float
    per_class_d_seg: tuple[float, float, float, float, float]
    d_pose: float
    archive_bytes: int
    n_pairs: int
    hard_parsed: bool
    receiver_realized: bool
    topology_stable: bool
    transitions_pending: bool
    topology_signature: str
    evidence_axis: str
    verdict_scope: str

    def __post_init__(self) -> None:
        _require_nonempty("stage_id", self.stage_id)
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        _require_sha256("archive_sha256", self.archive_sha256)
        _require_sha256("receiver_sha256", self.receiver_sha256)
        _require_finite("d_seg", self.d_seg, minimum=0.0)
        if not isinstance(self.per_class_d_seg, tuple) or len(self.per_class_d_seg) != 5:
            raise EndgamePolicyError("per_class_d_seg must be a canonical five-value tuple")
        for class_id, class_d_seg in enumerate(self.per_class_d_seg):
            _require_finite(f"per_class_d_seg[{class_id}]", class_d_seg, minimum=0.0)
        _require_finite("d_pose", self.d_pose, minimum=0.0)
        _require_int("archive_bytes", self.archive_bytes)
        _require_int("n_pairs", self.n_pairs, minimum=1)
        _require_bool("hard_parsed", self.hard_parsed)
        _require_bool("receiver_realized", self.receiver_realized)
        _require_bool("topology_stable", self.topology_stable)
        _require_bool("transitions_pending", self.transitions_pending)
        _require_nonempty("topology_signature", self.topology_signature)
        _require_nonempty("evidence_axis", self.evidence_axis)
        _require_nonempty("verdict_scope", self.verdict_scope)

    @property
    def score(self) -> float:
        return contest_score(self.d_seg, self.d_pose, self.archive_bytes)

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.checkpoint_sha256, self.archive_sha256, self.receiver_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "archive_sha256": self.archive_sha256,
            "receiver_sha256": self.receiver_sha256,
            "d_seg": self.d_seg,
            "per_class_d_seg": list(self.per_class_d_seg),
            "d_pose": self.d_pose,
            "archive_bytes": self.archive_bytes,
            "score": self.score,
            "n_pairs": self.n_pairs,
            "hard_parsed": self.hard_parsed,
            "receiver_realized": self.receiver_realized,
            "topology_stable": self.topology_stable,
            "transitions_pending": self.transitions_pending,
            "topology_signature": self.topology_signature,
            "evidence_axis": self.evidence_axis,
            "verdict_scope": self.verdict_scope,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OperatingPoint:
        if not isinstance(payload, Mapping):
            raise EndgamePolicyError("operating_point must be a mapping")
        _reject_unexpected_keys("operating_point", payload, _OPERATING_POINT_PAYLOAD_KEYS)
        point = cls(
            stage_id=payload["stage_id"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            archive_sha256=payload["archive_sha256"],
            receiver_sha256=payload["receiver_sha256"],
            d_seg=payload["d_seg"],
            per_class_d_seg=tuple(payload["per_class_d_seg"]),
            d_pose=payload["d_pose"],
            archive_bytes=payload["archive_bytes"],
            n_pairs=payload["n_pairs"],
            hard_parsed=payload["hard_parsed"],
            receiver_realized=payload["receiver_realized"],
            topology_stable=payload["topology_stable"],
            transitions_pending=payload["transitions_pending"],
            topology_signature=payload["topology_signature"],
            evidence_axis=payload["evidence_axis"],
            verdict_scope=payload["verdict_scope"],
        )
        if "score" in payload and not math.isclose(
            _require_finite("operating_point.score", payload["score"]),
            point.score,
            rel_tol=0.0,
            abs_tol=_SCORE_TOLERANCE,
        ):
            raise EndgamePolicyError("serialized operating_point.score does not re-derive")
        return point


@dataclass(frozen=True, slots=True)
class ActionQuote:
    """Measured or bounded economics for one action on one exact parent."""

    quote_id: str
    kind: ActionKind
    parent_checkpoint_sha256: str
    parent_archive_sha256: str
    parent_receiver_sha256: str
    n_pairs: int
    gain_lower: float
    gain_upper: float
    wall_seconds_lower: float
    wall_seconds_upper: float
    measured: bool
    hard_parsed: bool
    receiver_realized: bool
    admissible: bool
    verdict_scope: str
    evidence_axis: str
    endpoint_d_seg: float | None = None
    endpoint_d_pose: float | None = None
    endpoint_archive_bytes: int | None = None
    candidate_evaluations: int = 0

    def __post_init__(self) -> None:
        _require_nonempty("quote_id", self.quote_id)
        if not isinstance(self.kind, ActionKind):
            raise EndgamePolicyError("kind must be an ActionKind")
        _require_sha256("parent_checkpoint_sha256", self.parent_checkpoint_sha256)
        _require_sha256("parent_archive_sha256", self.parent_archive_sha256)
        _require_sha256("parent_receiver_sha256", self.parent_receiver_sha256)
        _require_int("n_pairs", self.n_pairs, minimum=1)
        lower = _require_finite("gain_lower", self.gain_lower)
        upper = _require_finite("gain_upper", self.gain_upper)
        if lower > upper:
            raise EndgamePolicyError("gain_lower must be <= gain_upper")
        wall_lower = _require_finite("wall_seconds_lower", self.wall_seconds_lower, minimum=0.0)
        wall_upper = _require_finite("wall_seconds_upper", self.wall_seconds_upper, minimum=0.0)
        if wall_lower <= 0.0 or wall_lower > wall_upper:
            raise EndgamePolicyError("wall seconds must satisfy 0 < lower <= upper")
        _require_bool("measured", self.measured)
        _require_bool("hard_parsed", self.hard_parsed)
        _require_bool("receiver_realized", self.receiver_realized)
        _require_bool("admissible", self.admissible)
        _require_nonempty("verdict_scope", self.verdict_scope)
        _require_nonempty("evidence_axis", self.evidence_axis)
        _require_int("candidate_evaluations", self.candidate_evaluations)
        endpoints = (self.endpoint_d_seg, self.endpoint_d_pose, self.endpoint_archive_bytes)
        if any(value is None for value in endpoints) and not all(value is None for value in endpoints):
            raise EndgamePolicyError("endpoint d_seg/d_pose/archive_bytes must be all present or all absent")
        if self.endpoint_d_seg is not None:
            contest_score(
                self.endpoint_d_seg,
                self.endpoint_d_pose,  # type: ignore[arg-type]
                self.endpoint_archive_bytes,  # type: ignore[arg-type]
            )

    @property
    def parent_identity(self) -> tuple[str, str, str]:
        return (
            self.parent_checkpoint_sha256,
            self.parent_archive_sha256,
            self.parent_receiver_sha256,
        )

    @property
    def rate_lower(self) -> float:
        return self.gain_lower / self.wall_seconds_upper

    @property
    def rate_upper(self) -> float:
        return self.gain_upper / self.wall_seconds_lower

    @property
    def endpoint_complete(self) -> bool:
        return self.endpoint_d_seg is not None

    @property
    def endpoint_score(self) -> float | None:
        if not self.endpoint_complete:
            return None
        return contest_score(
            self.endpoint_d_seg,  # type: ignore[arg-type]
            self.endpoint_d_pose,  # type: ignore[arg-type]
            self.endpoint_archive_bytes,  # type: ignore[arg-type]
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "quote_id": self.quote_id,
            "kind": self.kind.value,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_archive_sha256": self.parent_archive_sha256,
            "parent_receiver_sha256": self.parent_receiver_sha256,
            "n_pairs": self.n_pairs,
            "gain_lower": self.gain_lower,
            "gain_upper": self.gain_upper,
            "wall_seconds_lower": self.wall_seconds_lower,
            "wall_seconds_upper": self.wall_seconds_upper,
            "rate_lower": self.rate_lower,
            "rate_upper": self.rate_upper,
            "measured": self.measured,
            "hard_parsed": self.hard_parsed,
            "receiver_realized": self.receiver_realized,
            "admissible": self.admissible,
            "verdict_scope": self.verdict_scope,
            "evidence_axis": self.evidence_axis,
            "endpoint_d_seg": self.endpoint_d_seg,
            "endpoint_d_pose": self.endpoint_d_pose,
            "endpoint_archive_bytes": self.endpoint_archive_bytes,
            "candidate_evaluations": self.candidate_evaluations,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ActionQuote:
        if not isinstance(payload, Mapping):
            raise EndgamePolicyError("action quote must be a mapping")
        _reject_unexpected_keys("action quote", payload, _ACTION_QUOTE_PAYLOAD_KEYS)
        quote = cls(
            quote_id=payload["quote_id"],
            kind=ActionKind(payload["kind"]),
            parent_checkpoint_sha256=payload["parent_checkpoint_sha256"],
            parent_archive_sha256=payload["parent_archive_sha256"],
            parent_receiver_sha256=payload["parent_receiver_sha256"],
            n_pairs=payload["n_pairs"],
            gain_lower=payload["gain_lower"],
            gain_upper=payload["gain_upper"],
            wall_seconds_lower=payload["wall_seconds_lower"],
            wall_seconds_upper=payload["wall_seconds_upper"],
            measured=payload["measured"],
            hard_parsed=payload["hard_parsed"],
            receiver_realized=payload["receiver_realized"],
            admissible=payload["admissible"],
            verdict_scope=payload["verdict_scope"],
            evidence_axis=payload["evidence_axis"],
            endpoint_d_seg=payload.get("endpoint_d_seg"),
            endpoint_d_pose=payload.get("endpoint_d_pose"),
            endpoint_archive_bytes=payload.get("endpoint_archive_bytes"),
            candidate_evaluations=payload.get("candidate_evaluations", 0),
        )
        for key, actual in (("rate_lower", quote.rate_lower), ("rate_upper", quote.rate_upper)):
            if key in payload and not math.isclose(
                _require_finite(f"quote.{key}", payload[key]),
                actual,
                rel_tol=0.0,
                abs_tol=_SCORE_TOLERANCE,
            ):
                raise EndgamePolicyError(f"serialized quote.{key} does not re-derive")
        return quote


@dataclass(frozen=True, slots=True)
class AdvisorySignals:
    """Serialized forecast/radius/regime metadata that cannot actuate."""

    ncde_fire: bool | None = None
    ncde_fit_r2: float | None = None
    trajectory_regime: TrajectoryRegime = TrajectoryRegime.UNKNOWN
    grokking_classification: str | None = None
    v17_rho: float | None = None
    v17_radius_update: str | None = None

    def __post_init__(self) -> None:
        if self.ncde_fire is not None:
            _require_bool("ncde_fire", self.ncde_fire)
        if self.ncde_fit_r2 is not None:
            _require_finite("ncde_fit_r2", self.ncde_fit_r2)
        if not isinstance(self.trajectory_regime, TrajectoryRegime):
            raise EndgamePolicyError("trajectory_regime must be a TrajectoryRegime")
        if self.grokking_classification is not None:
            _require_nonempty("grokking_classification", self.grokking_classification)
        if self.v17_rho is not None:
            _require_finite("v17_rho", self.v17_rho)
        if self.v17_radius_update is not None:
            _require_nonempty("v17_radius_update", self.v17_radius_update)

    def to_payload(self) -> dict[str, Any]:
        return {
            "ncde_fire": self.ncde_fire,
            "ncde_fit_r2": self.ncde_fit_r2,
            "trajectory_regime": self.trajectory_regime.value,
            "grokking_classification": self.grokking_classification,
            "v17_rho": self.v17_rho,
            "v17_radius_update": self.v17_radius_update,
            "actuation": "NONE",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> AdvisorySignals:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise EndgamePolicyError("advisory_signals must be a mapping")
        if payload.get("actuation", "NONE") != "NONE":
            raise EndgamePolicyError("advisory signals cannot carry actuation")
        return cls(
            ncde_fire=payload.get("ncde_fire"),
            ncde_fit_r2=payload.get("ncde_fit_r2"),
            trajectory_regime=TrajectoryRegime(payload.get("trajectory_regime", "UNKNOWN")),
            grokking_classification=payload.get("grokking_classification"),
            v17_rho=payload.get("v17_rho"),
            v17_radius_update=payload.get("v17_radius_update"),
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """Resume-serializable policy result."""

    action: DecisionAction
    selected_quote_id: str | None
    operating_score: float
    target_score: float
    target_gap: float
    train_rate_lower: float | None
    train_rate_upper: float | None
    selected_finisher_rate_lower: float | None
    selected_finisher_rate_upper: float | None
    reason_codes: tuple[str, ...]
    verdict_scope: str
    advisory_signals: AdvisorySignals

    def __post_init__(self) -> None:
        if not isinstance(self.action, DecisionAction):
            raise EndgamePolicyError("action must be a DecisionAction")
        if self.selected_quote_id is not None:
            _require_nonempty("selected_quote_id", self.selected_quote_id)
        _require_finite("operating_score", self.operating_score, minimum=0.0)
        _require_finite("target_score", self.target_score, minimum=0.0)
        _require_finite("target_gap", self.target_gap, minimum=0.0)
        for name, value in (
            ("train_rate_lower", self.train_rate_lower),
            ("train_rate_upper", self.train_rate_upper),
            ("selected_finisher_rate_lower", self.selected_finisher_rate_lower),
            ("selected_finisher_rate_upper", self.selected_finisher_rate_upper),
        ):
            if value is not None:
                _require_finite(name, value)
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise EndgamePolicyError("reason_codes must be a non-empty tuple")
        for reason in self.reason_codes:
            _require_nonempty("reason_code", reason)
        _require_nonempty("verdict_scope", self.verdict_scope)
        if not isinstance(self.advisory_signals, AdvisorySignals):
            raise EndgamePolicyError("advisory_signals must be AdvisorySignals")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "action": self.action.value,
            "selected_quote_id": self.selected_quote_id,
            "operating_score": self.operating_score,
            "target_score": self.target_score,
            "target_gap": self.target_gap,
            "train_rate_lower": self.train_rate_lower,
            "train_rate_upper": self.train_rate_upper,
            "selected_finisher_rate_lower": self.selected_finisher_rate_lower,
            "selected_finisher_rate_upper": self.selected_finisher_rate_upper,
            "reason_codes": list(self.reason_codes),
            "verdict_scope": self.verdict_scope,
            "advisory_signals": self.advisory_signals.to_payload(),
            "research_only": True,
            "score_claim": False,
            "pointer_moved": False,
            "actuation": "ADVISORY_DECISION_ONLY",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Decision:
        if not isinstance(payload, Mapping) or payload.get("schema") != DECISION_SCHEMA:
            raise EndgamePolicyError("decision schema mismatch")
        if payload.get("research_only") is not True:
            raise EndgamePolicyError("decision must preserve research_only=true")
        if payload.get("score_claim") is not False:
            raise EndgamePolicyError("decision must preserve score_claim=false")
        if payload.get("pointer_moved") is not False:
            raise EndgamePolicyError("decision must preserve pointer_moved=false")
        if payload.get("actuation") != "ADVISORY_DECISION_ONLY":
            raise EndgamePolicyError("decision actuation firewall drifted")
        return cls(
            action=DecisionAction(payload["action"]),
            selected_quote_id=payload.get("selected_quote_id"),
            operating_score=payload["operating_score"],
            target_score=payload["target_score"],
            target_gap=payload["target_gap"],
            train_rate_lower=payload.get("train_rate_lower"),
            train_rate_upper=payload.get("train_rate_upper"),
            selected_finisher_rate_lower=payload.get("selected_finisher_rate_lower"),
            selected_finisher_rate_upper=payload.get("selected_finisher_rate_upper"),
            reason_codes=tuple(payload["reason_codes"]),
            verdict_scope=payload["verdict_scope"],
            advisory_signals=AdvisorySignals.from_payload(payload.get("advisory_signals")),
        )


def _decision(
    *,
    action: DecisionAction,
    point: OperatingPoint,
    target: float,
    signals: AdvisorySignals,
    reasons: Sequence[str],
    train: ActionQuote | None = None,
    finisher: ActionQuote | None = None,
) -> Decision:
    return Decision(
        action=action,
        selected_quote_id=None if finisher is None else finisher.quote_id,
        operating_score=point.score,
        target_score=target,
        target_gap=max(0.0, point.score - target),
        train_rate_lower=None if train is None else train.rate_lower,
        train_rate_upper=None if train is None else train.rate_upper,
        selected_finisher_rate_lower=None if finisher is None else finisher.rate_lower,
        selected_finisher_rate_upper=None if finisher is None else finisher.rate_upper,
        reason_codes=tuple(reasons),
        verdict_scope=(
            "INSTANCE: one exact parent identity and supplied same-parent measured quote intervals; "
            "no transfer across checkpoints, archives, receivers, vehicles, or authority axes"
        ),
        advisory_signals=signals,
    )


def _validate_quote_against_point(point: OperatingPoint, quote: ActionQuote) -> float | None:
    if quote.parent_identity != point.identity:
        raise EndgamePolicyError(f"quote {quote.quote_id!r} parent identity does not match the operating point")
    if quote.n_pairs != point.n_pairs:
        raise EndgamePolicyError(f"quote {quote.quote_id!r} n_pairs does not match the operating point")
    if quote.evidence_axis != point.evidence_axis:
        raise EndgamePolicyError(f"quote {quote.quote_id!r} evidence_axis does not match the operating point")
    endpoint_score = quote.endpoint_score
    if endpoint_score is None:
        return None
    exact_gain = point.score - endpoint_score
    if exact_gain < quote.gain_lower - _SCORE_TOLERANCE or exact_gain > quote.gain_upper + _SCORE_TOLERANCE:
        raise EndgamePolicyError(f"quote {quote.quote_id!r} gain interval does not contain its exact endpoint gain")
    return exact_gain


def decide_endgame_policy(
    point: OperatingPoint,
    quotes: Sequence[ActionQuote],
    *,
    target_score: float = OFFICIAL_DISPLAYED_BAR,
    advisory_signals: AdvisorySignals | None = None,
) -> Decision:
    """Return a deterministic advisory decision from same-parent measured economics.

    A finisher handoff requires its conservative gain-rate lower bound to be
    strictly greater than the measured training quote's gain-rate upper bound.
    Overlap is unresolved evidence, not a tie-break opportunity.
    """

    if not isinstance(point, OperatingPoint):
        raise EndgamePolicyError("point must be an OperatingPoint")
    signals = AdvisorySignals() if advisory_signals is None else advisory_signals
    if not isinstance(signals, AdvisorySignals):
        raise EndgamePolicyError("advisory_signals must be AdvisorySignals")
    target = _require_finite("target_score", target_score, minimum=0.0)
    rows = tuple(quotes)
    if not all(isinstance(row, ActionQuote) for row in rows):
        raise EndgamePolicyError("quotes must contain only ActionQuote values")
    ids = [row.quote_id for row in rows]
    if len(ids) != len(set(ids)):
        raise EndgamePolicyError("quote_id values must be unique")
    exact_gain_by_id = {row.quote_id: _validate_quote_against_point(point, row) for row in rows}

    if point.n_pairs != 600 or not point.hard_parsed or not point.receiver_realized:
        return _decision(
            action=DecisionAction.REFUSE_INSUFFICIENT_EVIDENCE,
            point=point,
            target=target,
            signals=signals,
            reasons=("OPERATING_POINT_NOT_N600_HARD_PARSED_RECEIVER_REALIZED",),
        )
    if point.score < target:
        return _decision(
            action=DecisionAction.R6_EXACT_EVAL,
            point=point,
            target=target,
            signals=signals,
            reasons=("ADVISORY_OPERATING_POINT_STRICTLY_BELOW_TARGET_REQUIRES_R6_EXACT_EVAL",),
        )

    train_rows = [row for row in rows if row.kind is ActionKind.TRAIN_WINDOW]
    if len(train_rows) > 1:
        raise EndgamePolicyError("at most one TRAIN_WINDOW quote is allowed")
    if not train_rows:
        return _decision(
            action=DecisionAction.MEASURE_FINISHER_QUOTE,
            point=point,
            target=target,
            signals=signals,
            reasons=("MISSING_SAME_PARENT_MEASURED_TRAINING_QUOTE",),
        )
    train = train_rows[0]
    train_exact_gain = exact_gain_by_id[train.quote_id]
    train_usable = (
        train.measured
        and train.hard_parsed
        and train.receiver_realized
        and train.admissible
        and train.endpoint_complete
        and train_exact_gain is not None
        and train_exact_gain > 0.0
        and train.gain_lower > 0.0
    )
    if not train_usable:
        return _decision(
            action=DecisionAction.MEASURE_FINISHER_QUOTE,
            point=point,
            target=target,
            signals=signals,
            reasons=("TRAINING_QUOTE_MISSING_POSITIVE_MEASURED_HARD_REALIZED_EXACT_ENDPOINT",),
            train=train,
        )

    finisher_rows = [row for row in rows if row.kind is not ActionKind.TRAIN_WINDOW]
    if not finisher_rows:
        return _decision(
            action=DecisionAction.CONTINUE_BOUNDED_WINDOW,
            point=point,
            target=target,
            signals=signals,
            reasons=("POSITIVE_MEASURED_TRAINING_QUOTE_AND_NO_COMPETING_FINISHER_QUOTE",),
            train=train,
        )

    eligible: list[ActionQuote] = []
    incomplete_price = False
    stage_blocked = False
    for row in finisher_rows:
        exact_gain = exact_gain_by_id[row.quote_id]
        complete = (
            row.measured
            and row.hard_parsed
            and row.receiver_realized
            and row.admissible
            and row.endpoint_complete
            and exact_gain is not None
            and exact_gain > 0.0
            and row.gain_lower > 0.0
        )
        if not complete:
            incomplete_price = True
            continue
        if row.kind in {ActionKind.SEG_GN, ActionKind.TERMINAL_POSE} and (
            not point.topology_stable or point.transitions_pending
        ):
            stage_blocked = True
            continue
        eligible.append(row)

    if incomplete_price:
        return _decision(
            action=DecisionAction.MEASURE_FINISHER_QUOTE,
            point=point,
            target=target,
            signals=signals,
            reasons=("FINISHER_QUOTE_MISSING_HARD_PARSED_REALIZED_ADMISSIBLE_PRICE",),
            train=train,
        )
    if not eligible:
        reason = (
            "GN_OR_POSE_BLOCKED_UNTIL_TOPOLOGY_STABLE_AND_TRANSITIONS_COMPLETE"
            if stage_blocked
            else "NO_ADMISSIBLE_POSITIVE_FINISHER_QUOTE"
        )
        return _decision(
            action=DecisionAction.CONTINUE_BOUNDED_WINDOW,
            point=point,
            target=target,
            signals=signals,
            reasons=(reason,),
            train=train,
        )

    strict_dominators = [row for row in eligible if row.rate_lower > train.rate_upper]
    if strict_dominators:
        selected = min(strict_dominators, key=lambda row: (-row.rate_lower, row.quote_id))
        action = {
            ActionKind.SEG_GN: DecisionAction.HANDOFF_SEG_GN,
            ActionKind.QDBS: DecisionAction.HANDOFF_QDBS,
            ActionKind.TERMINAL_POSE: DecisionAction.HANDOFF_TERMINAL_POSE,
        }[selected.kind]
        return _decision(
            action=action,
            point=point,
            target=target,
            signals=signals,
            reasons=("FINISHER_GAIN_RATE_LOWER_STRICTLY_EXCEEDS_TRAIN_UPPER",),
            train=train,
            finisher=selected,
        )

    best_finisher_upper = max(row.rate_upper for row in eligible)
    if train.rate_lower > best_finisher_upper:
        return _decision(
            action=DecisionAction.CONTINUE_BOUNDED_WINDOW,
            point=point,
            target=target,
            signals=signals,
            reasons=("TRAIN_GAIN_RATE_LOWER_STRICTLY_EXCEEDS_ALL_FINISHER_UPPERS",),
            train=train,
        )
    selected = min(eligible, key=lambda row: (-row.rate_upper, row.quote_id))
    return _decision(
        action=DecisionAction.MEASURE_FINISHER_QUOTE,
        point=point,
        target=target,
        signals=signals,
        reasons=("TRAIN_AND_FINISHER_GAIN_RATE_INTERVALS_OVERLAP",),
        train=train,
        finisher=selected,
    )


def _corner(d_seg: float, d_pose: float, archive_bytes: int | float) -> dict[str, Any]:
    size = _require_finite("corner.archive_bytes", archive_bytes, minimum=0.0)
    score = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + BYTE_SCORE_PRICE * size
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "seg_term": 100.0 * d_seg,
        "pose_term": math.sqrt(10.0 * d_pose),
        "rate_term": BYTE_SCORE_PRICE * size,
        "score": score,
    }


def build_endgame_arithmetic_receipt() -> dict[str, Any]:
    """Build the deterministic E2 arithmetic receipt from committed constants."""

    qdbs_full_verdicts = 49  # 48 candidates plus one shared base.
    r1_d_pose = 0.001610
    r1_at_seg_3e4_zero_bytes = contest_score(3e-4, r1_d_pose, 0)
    return {
        "schema": ARITHMETIC_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "competitive_target": {
            "official_displayed": OFFICIAL_DISPLAYED_BAR,
            "sub015": SUB015_BAR,
            "note": "0.172 is the conservative displayed comparator; only an exact official row is authority",
        },
        "score_law": {
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "archive_denominator": ARCHIVE_DENOMINATOR,
            "byte_score_price": BYTE_SCORE_PRICE,
            "binary_kib_score_price": BYTE_SCORE_PRICE * 1024,
            "binary_64kib_score_price": BYTE_SCORE_PRICE * 65536,
        },
        "configuration_corners": {
            "tr1_optimistic_decimal_149k": _corner(2.97e-4, 2.33e-5, 149_000),
            "tr1_spec_mid": _corner(3e-4, 2.33e-5, 196_000),
            "tr1_spec_mid_dseg_5e4": _corner(5e-4, 2.33e-5, 196_000),
            "tr1_banked_pose_fallback": _corner(5e-4, r1_d_pose, 201_000),
            "pp1_conditional": _corner(3e-4, 2.33e-5, 215_616.5),
        },
        "strict_integer_byte_ceilings": {
            "dseg_3e4_dpose_2p33e5": {
                "official_displayed_0p172": strict_integer_byte_ceiling(OFFICIAL_DISPLAYED_BAR, 3e-4, 2.33e-5),
                "sub015": strict_integer_byte_ceiling(SUB015_BAR, 3e-4, 2.33e-5),
            },
            "dseg_5e4_dpose_2p33e5": {
                "official_displayed_0p172": strict_integer_byte_ceiling(OFFICIAL_DISPLAYED_BAR, 5e-4, 2.33e-5),
                "sub015": strict_integer_byte_ceiling(SUB015_BAR, 5e-4, 2.33e-5),
            },
            "dseg_0_banked_r1_pose": {
                "official_displayed_0p172": strict_integer_byte_ceiling(OFFICIAL_DISPLAYED_BAR, 0.0, r1_d_pose),
                "sub015": strict_integer_byte_ceiling(SUB015_BAR, 0.0, r1_d_pose),
            },
            "dseg_3e4_banked_r1_pose": {
                "official_displayed_0p172": strict_integer_byte_ceiling(OFFICIAL_DISPLAYED_BAR, 3e-4, r1_d_pose),
                "sub015": strict_integer_byte_ceiling(SUB015_BAR, 3e-4, r1_d_pose),
            },
        },
        "current_t2": {
            "d_seg": 0.013833,
            "archive_bytes": 534_597,
            "pose_lower_bound": 0.0,
            "score_lower_bound": contest_score(0.013833, 0.0, 534_597),
            "policy_without_same_parent_quotes": DecisionAction.MEASURE_FINISHER_QUOTE.value,
            "conditional_train_only_action": DecisionAction.CONTINUE_BOUNDED_WINDOW.value,
            "conditional_note": (
                "CONTINUE requires a positive measured hard receiver-realized training quote; "
                "the tracked T2 endpoint alone is not such a next-window quote"
            ),
        },
        "r1_banked_pose": {
            "d_pose": r1_d_pose,
            "counted_section_bytes": 7_195,
            "pose_term": math.sqrt(10.0 * r1_d_pose),
            "score_at_dseg_3e4_zero_bytes": r1_at_seg_3e4_zero_bytes,
            "sub015_possible_at_dseg_3e4": r1_at_seg_3e4_zero_bytes < SUB015_BAR,
        },
        "qdbs_cost_quote": {
            "status": "DERIVED_FROM_MEASURED_FULL_VERDICT_TIMES_NOT_MEASURED_QDBS_RUNTIME",
            "qdbs_candidates": 24,
            "random_controls": 24,
            "candidate_evaluations_max": 48,
            "shared_base_verdicts": 1,
            "total_full_verdicts": qdbs_full_verdicts,
            "full_verdict_seconds_lower": 423,
            "full_verdict_seconds_upper": 514,
            "total_seconds_lower": qdbs_full_verdicts * 423,
            "total_seconds_upper": qdbs_full_verdicts * 514,
            "gain": None,
        },
        "terminal_pose_measured_quote": {
            "scope": "PC2 exact ws4 W_joint-step50 parent only; no transfer to TR1",
            "accepted_steps": 16,
            "candidate_evaluations": 192,
            "elapsed_seconds": 1275.2549629998393,
            "delta_d_seg": 0.0001900566948784717,
            "delta_d_pose": -2.1452689763775084,
            "delta_archive_bytes": 23,
            "delta_score": -0.24750113405601581,
            "gain_score_per_hour": 0.24750113405601581 / 1275.2549629998393 * 3600.0,
            "tr1_2kb_dpose_2p33e5_price_status": "UNMEASURED_ON_OUR_VEHICLE",
        },
        "source_receipts": [
            ".omx/research/ddm_tb1_renderer_build_20260728.md",
            ".omx/research/SPEC_tr1_trained_partition_renderer_20260728.md",
            ".omx/research/ddm_eu1_sol_ultra_eureka_hunt_20260728.md",
            ".omx/research/ddm_pc2_pose_descent_smoke_result_20260725.json",
            ".omx/research/r1_dxi_shippability_byteclose_20260708.md",
            ".omx/research/basin_finisher_head_solve_probe_measured_20260707.md",
        ],
        "generator": "tools/derive_ddm_endgame_policy.py",
    }


__all__ = [
    "ARCHIVE_DENOMINATOR",
    "ARITHMETIC_SCHEMA",
    "BYTE_SCORE_PRICE",
    "OFFICIAL_DISPLAYED_BAR",
    "POLICY_SCHEMA",
    "SUB015_BAR",
    "ActionKind",
    "ActionQuote",
    "AdvisorySignals",
    "Decision",
    "DecisionAction",
    "EndgamePolicyError",
    "OperatingPoint",
    "TrajectoryRegime",
    "build_endgame_arithmetic_receipt",
    "contest_score",
    "decide_endgame_policy",
    "strict_integer_byte_ceiling",
]
