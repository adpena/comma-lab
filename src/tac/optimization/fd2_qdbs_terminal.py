# SPDX-License-Identifier: MIT
"""Bounded description-level hard-oracle search for the FD2 QDBS row.

This is the description-coordinate analogue of
``uint8_lattice_feasibility.repair_with_hard_oracle``.  Smooth scorer signals
may order a precommitted proposal set, but they never admit a move.  Admission
uses only the fresh:

    compile -> parse back -> receiver consumption -> realized joint action

chain supplied by the caller.

The EU1 row is intentionally fixed: sixteen signed singleton proposals, eight
grouped proposals, and twenty-four matched integer-random controls.  All 48
candidates are evaluated against one shared base.  A bounded stale rehearsal
can exercise the mechanism.  Self-attested full-n600 custody can identify only
a candidate requiring an external governor; this module never grants handoff
or promotion authority.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

SCORER_SINGLETON_COUNT = 16
SCORER_GROUP_COUNT = 8
RANDOM_CONTROL_COUNT = 24
MAX_CANDIDATE_EVALUATIONS = 48
FULL_N600_SAMPLE_COUNT = 600
FULL_N600_AUTHORITY_MARKER = "FULL_N600_JOINT_ZIP_ACTION"
STALE_REHEARSAL_AUTHORITY_MARKER = "STALE_BOUNDED_MECHANISM_ONLY"
_RESUME_SCHEMA = "ddm_fd2_qdbs_resume.v3"
_EVALUATION_KEY_SCHEMA = "ddm_fd2_qdbs_evaluation_idempotency.v1"
_RECORD_PENDING = "PENDING_EVALUATION"
_RECORD_COMPLETE = "COMPLETE"


class FD2QDBSError(ValueError):
    """Fail-closed malformed schedule, callback, or authority receipt."""


class QDBSAuthorityMode(StrEnum):
    PRODUCTION_FULL_N600 = "PRODUCTION_FULL_N600"
    STALE_REHEARSAL = "STALE_REHEARSAL"


class ContestAxis(StrEnum):
    CONTEST_CPU = "[contest-CPU]"
    CONTEST_CUDA = "[contest-CUDA]"


class ProposalClass(StrEnum):
    SCORER_SINGLETON = "SCORER_SINGLETON"
    SCORER_GROUP = "SCORER_GROUP"
    RANDOM_CONTROL = "RANDOM_CONTROL"


class QDBSStatus(StrEnum):
    REQUIRES_EXTERNAL_GOVERNOR = "REQUIRES_EXTERNAL_GOVERNOR"
    NO_STRICT_IMPROVEMENT = "NO_STRICT_IMPROVEMENT"
    REHEARSAL_NONPROMOTABLE = "REHEARSAL_NONPROMOTABLE"


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FD2QDBSError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise FD2QDBSError(f"{name} must be >= {minimum}")
    return result


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FD2QDBSError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise FD2QDBSError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise FD2QDBSError(f"{name} must be >= {minimum}")
    return result


def _immutable_i64(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in ("i", "u"):
        raise FD2QDBSError(f"{name} must be a nonempty one-dimensional integer array")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise FD2QDBSError(f"{name} exceeds int64")
    contiguous = np.ascontiguousarray(raw, dtype=np.int64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.int64).reshape(contiguous.shape)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FD2QDBSError(f"{name} must be a nonempty stripped string")
    return value


def _sha256_text(value: object, name: str) -> str:
    digest = _text(value, name)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise FD2QDBSError(f"{name} must be lowercase SHA-256")
    return digest


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _payload_digest(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _theta_digest(theta: np.ndarray) -> str:
    return sha256(np.asarray(theta, dtype=">i8").tobytes()).hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FD2QDBSError(f"{label} ancestry is unreadable") from exc
        if stat.S_ISLNK(mode):
            raise FD2QDBSError(f"{label} cannot contain a symlink ancestor")


def _read_bytes_nofollow(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FD2QDBSError(f"{label} is unreadable or symlinked") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FD2QDBSError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ProductionCustody:
    """Typed self-attested custody for one exact contest-axis verdict family."""

    parent_checkpoint_sha256: str
    parent_archive_sha256: str
    parent_archive_bytes: int
    compiler_sha256: str
    receiver_sha256: str
    evaluator_sha256: str
    evaluation_protocol_sha256: str
    upstream_sha256: str
    n_pairs: int
    axis: ContestAxis
    command: str
    hardware: str

    def __post_init__(self) -> None:
        for name in (
            "parent_checkpoint_sha256",
            "parent_archive_sha256",
            "compiler_sha256",
            "receiver_sha256",
            "evaluator_sha256",
            "evaluation_protocol_sha256",
            "upstream_sha256",
        ):
            object.__setattr__(self, name, _sha256_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "parent_archive_bytes",
            _integer(self.parent_archive_bytes, "custody.parent_archive_bytes", minimum=1),
        )
        object.__setattr__(self, "n_pairs", _integer(self.n_pairs, "custody.n_pairs", minimum=1))
        if self.n_pairs != FULL_N600_SAMPLE_COUNT:
            raise FD2QDBSError("custody.n_pairs must be exactly 600")
        if not isinstance(self.axis, ContestAxis):
            raise FD2QDBSError("custody.axis must be ContestAxis")
        object.__setattr__(self, "command", _text(self.command, "custody.command"))
        object.__setattr__(self, "hardware", _text(self.hardware, "custody.hardware"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_archive_sha256": self.parent_archive_sha256,
            "parent_archive_bytes": self.parent_archive_bytes,
            "compiler_sha256": self.compiler_sha256,
            "receiver_sha256": self.receiver_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "upstream_sha256": self.upstream_sha256,
            "n_pairs": self.n_pairs,
            "axis": self.axis.value,
            "command": self.command,
            "hardware": self.hardware,
            "verification_status": self.verification_status,
        }

    @property
    def verification_status(self) -> str:
        return "SELF_ATTESTED_REQUIRES_EXTERNAL_GOVERNOR"

    @property
    def digest(self) -> str:
        return _payload_digest(self.to_payload())


@dataclass(frozen=True, order=True)
class DescriptionDelta:
    """One signed unit move in the realized description-coordinate lattice."""

    index: int
    delta: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _integer(self.index, "delta.index", minimum=0))
        delta = _integer(self.delta, "delta.delta")
        if delta not in (-1, 1):
            raise FD2QDBSError("delta.delta must be exactly -1 or +1")
        object.__setattr__(self, "delta", delta)

    def to_payload(self) -> dict[str, int]:
        return {"index": self.index, "delta": self.delta}


@dataclass(frozen=True)
class DescriptionProposal:
    """A precommitted description edit; scorer metadata is ranking-only."""

    identity: str
    proposal_class: ProposalClass
    deltas: tuple[DescriptionDelta, ...]
    signal_label: str
    signal_value: float
    matched_scorer_identity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _text(self.identity, "proposal.identity"))
        if not isinstance(self.proposal_class, ProposalClass):
            raise FD2QDBSError("proposal.proposal_class must be ProposalClass")
        if not isinstance(self.deltas, tuple) or not self.deltas:
            raise FD2QDBSError("proposal.deltas must be a nonempty tuple")
        if any(not isinstance(delta, DescriptionDelta) for delta in self.deltas):
            raise FD2QDBSError("proposal.deltas must contain DescriptionDelta values")
        if tuple(sorted(self.deltas)) != self.deltas:
            raise FD2QDBSError("proposal.deltas must be in canonical sorted order")
        indices = tuple(delta.index for delta in self.deltas)
        if len(set(indices)) != len(indices):
            raise FD2QDBSError("proposal cannot edit one coordinate more than once")
        object.__setattr__(self, "signal_label", _text(self.signal_label, "proposal.signal_label"))
        object.__setattr__(self, "signal_value", _finite(self.signal_value, "proposal.signal_value"))
        if self.proposal_class is ProposalClass.SCORER_SINGLETON:
            if len(self.deltas) != 1 or self.matched_scorer_identity is not None:
                raise FD2QDBSError("scorer singleton must contain one unmatched delta")
        elif self.proposal_class is ProposalClass.SCORER_GROUP:
            if len(self.deltas) < 2 or self.matched_scorer_identity is not None:
                raise FD2QDBSError("scorer group must contain at least two unmatched deltas")
        else:
            if self.matched_scorer_identity is None:
                raise FD2QDBSError("random control must identify its matched scorer proposal")
            object.__setattr__(
                self,
                "matched_scorer_identity",
                _text(self.matched_scorer_identity, "proposal.matched_scorer_identity"),
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "proposal_class": self.proposal_class.value,
            "deltas": [delta.to_payload() for delta in self.deltas],
            "signal_label": self.signal_label,
            "signal_value": self.signal_value,
            "matched_scorer_identity": self.matched_scorer_identity,
        }


@dataclass(frozen=True)
class QDBSCandidateSchedule:
    seed: int
    coordinate_count: int
    active_indices: tuple[int, ...]
    scorer_proposals: tuple[DescriptionProposal, ...]
    random_controls: tuple[DescriptionProposal, ...]
    schedule_sha256: str

    @property
    def candidates(self) -> tuple[DescriptionProposal, ...]:
        return self.scorer_proposals + self.random_controls

    @property
    def active_indices_sha256(self) -> str:
        return _payload_digest(list(self.active_indices))

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "coordinate_count": self.coordinate_count,
            "active_indices": list(self.active_indices),
            "active_indices_sha256": self.active_indices_sha256,
            "schedule_sha256": self.schedule_sha256,
            "scorer_proposals": [proposal.to_payload() for proposal in self.scorer_proposals],
            "random_controls": [proposal.to_payload() for proposal in self.random_controls],
        }


@dataclass(frozen=True)
class ParsedDescriptionCandidate:
    """Exact parse-back evidence returned by the production parser callback."""

    realized_theta: np.ndarray
    archive_sha256: str
    exact_parseback: bool
    value: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "realized_theta", _immutable_i64(self.realized_theta, "parsed.realized_theta"))
        digest = _text(self.archive_sha256, "parsed.archive_sha256")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise FD2QDBSError("parsed.archive_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "archive_sha256", digest)
        if type(self.exact_parseback) is not bool:
            raise FD2QDBSError("parsed.exact_parseback must be bool")


@dataclass(frozen=True)
class ConsumedDescriptionCandidate:
    """Receiver-consumption evidence for the exact parsed archive."""

    realized_theta: np.ndarray
    archive_sha256: str
    exact_consumption: bool
    value: object = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "realized_theta",
            _immutable_i64(self.realized_theta, "consumed.realized_theta"),
        )
        digest = _text(self.archive_sha256, "consumed.archive_sha256")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise FD2QDBSError("consumed.archive_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "archive_sha256", digest)
        if type(self.exact_consumption) is not bool:
            raise FD2QDBSError("consumed.exact_consumption must be bool")


@dataclass(frozen=True)
class RealizedJointAction:
    """Hard evaluator result; the contest action is recomputed, never trusted."""

    d_seg: float
    d_pose: float
    archive_sha256: str
    archive_bytes: int
    sample_count: int
    authority_marker: str
    custody_digest: str | None
    evaluation_idempotency_key: str
    realized: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "d_seg", _finite(self.d_seg, "action.d_seg", minimum=0.0))
        object.__setattr__(self, "d_pose", _finite(self.d_pose, "action.d_pose", minimum=0.0))
        object.__setattr__(
            self,
            "archive_sha256",
            _sha256_text(self.archive_sha256, "action.archive_sha256"),
        )
        object.__setattr__(self, "archive_bytes", _integer(self.archive_bytes, "action.archive_bytes", minimum=1))
        object.__setattr__(self, "sample_count", _integer(self.sample_count, "action.sample_count", minimum=1))
        object.__setattr__(self, "authority_marker", _text(self.authority_marker, "action.authority_marker"))
        if self.custody_digest is not None:
            object.__setattr__(
                self,
                "custody_digest",
                _sha256_text(self.custody_digest, "action.custody_digest"),
            )
        object.__setattr__(
            self,
            "evaluation_idempotency_key",
            _sha256_text(
                self.evaluation_idempotency_key,
                "action.evaluation_idempotency_key",
            ),
        )
        if type(self.realized) is not bool:
            raise FD2QDBSError("action.realized must be bool")

    @property
    def action(self) -> float:
        return 100.0 * self.d_seg + math.sqrt(10.0 * self.d_pose) + 25.0 * self.archive_bytes / 37_545_489.0

    @property
    def full_n600(self) -> bool:
        return (
            self.realized
            and self.sample_count == FULL_N600_SAMPLE_COUNT
            and self.authority_marker == FULL_N600_AUTHORITY_MARKER
            and self.custody_digest is not None
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "sample_count": self.sample_count,
            "authority_marker": self.authority_marker,
            "custody_digest": self.custody_digest,
            "evaluation_idempotency_key": self.evaluation_idempotency_key,
            "realized": self.realized,
            "joint_action": self.action,
            "full_n600": self.full_n600,
        }

    @classmethod
    def from_payload(cls, payload: object) -> RealizedJointAction:
        if not isinstance(payload, dict):
            raise FD2QDBSError("resume action payload must be an object")
        allowed = {
            "d_seg",
            "d_pose",
            "archive_sha256",
            "archive_bytes",
            "sample_count",
            "authority_marker",
            "custody_digest",
            "evaluation_idempotency_key",
            "realized",
            "joint_action",
            "full_n600",
        }
        if set(payload) != allowed:
            raise FD2QDBSError("resume action fields differ")
        action = cls(
            d_seg=payload["d_seg"],
            d_pose=payload["d_pose"],
            archive_sha256=payload["archive_sha256"],
            archive_bytes=payload["archive_bytes"],
            sample_count=payload["sample_count"],
            authority_marker=payload["authority_marker"],
            custody_digest=payload["custody_digest"],
            evaluation_idempotency_key=payload["evaluation_idempotency_key"],
            realized=payload["realized"],
        )
        if payload.get("joint_action") != action.action or payload.get("full_n600") is not action.full_n600:
            raise FD2QDBSError("resume action derived fields do not re-derive")
        return action


@dataclass(frozen=True)
class DescriptionHardOracleCallbacks:
    """The four exact stages required for every base or candidate verdict."""

    compile_archive: Callable[[np.ndarray, DescriptionProposal | None], bytes]
    parse_archive: Callable[[bytes], ParsedDescriptionCandidate]
    consume_archive: Callable[[ParsedDescriptionCandidate], ConsumedDescriptionCandidate]
    evaluate_joint_action_idempotent: Callable[[ConsumedDescriptionCandidate, str], RealizedJointAction]
    compiler_sha256: str | None = None
    receiver_sha256: str | None = None
    evaluator_sha256: str | None = None
    evaluation_protocol_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "compile_archive",
            "parse_archive",
            "consume_archive",
            "evaluate_joint_action_idempotent",
        ):
            if not callable(getattr(self, name)):
                raise FD2QDBSError(f"callbacks.{name} must be callable")
        for name in (
            "compiler_sha256",
            "receiver_sha256",
            "evaluator_sha256",
            "evaluation_protocol_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256_text(value, f"callbacks.{name}"))


@dataclass(frozen=True)
class QDBSCandidateTrace:
    ordinal: int
    identity: str
    proposal_class: ProposalClass
    archive_sha256: str
    action: RealizedJointAction
    delta_vs_base: float
    strict_realized_improvement: bool
    governed_handoff_eligible: bool

    def __post_init__(self) -> None:
        if self.governed_handoff_eligible is not False:
            raise FD2QDBSError("QDBS candidate traces cannot authorize governed handoff")

    def to_payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "identity": self.identity,
            "proposal_class": self.proposal_class.value,
            "archive_sha256": self.archive_sha256,
            "action": self.action.to_payload(),
            "delta_vs_base": self.delta_vs_base,
            "strict_realized_improvement": self.strict_realized_improvement,
            "governed_handoff_eligible": self.governed_handoff_eligible,
        }


@dataclass(frozen=True)
class QDBSTerminalResult:
    authority_mode: QDBSAuthorityMode
    production_custody: ProductionCustody | None
    status: QDBSStatus
    schedule: QDBSCandidateSchedule
    base_archive_sha256: str
    base_action: RealizedJointAction
    traces: tuple[QDBSCandidateTrace, ...]
    best_strict_improvement_identity: str | None
    governed_handoff_identity: str | None
    candidate_evaluations: int
    shared_base_evaluations: int
    resume_records_reused: int
    resume_records_written: int
    resume_ledger_path: str | None
    resume_ledger_sha256: str | None

    def __post_init__(self) -> None:
        if self.governed_handoff_identity is not None:
            raise FD2QDBSError("QDBS terminal result cannot authorize governed handoff")
        if any(trace.governed_handoff_eligible for trace in self.traces):
            raise FD2QDBSError("QDBS terminal result contains a handoff-authorizing trace")

    @property
    def promotion_allowed(self) -> bool:
        return False

    @property
    def governed_handoff_eligible(self) -> bool:
        return False

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "ddm_fd2_qdbs_terminal.v2",
            "authority_mode": self.authority_mode.value,
            "production_custody": (None if self.production_custody is None else self.production_custody.to_payload()),
            "production_custody_digest": (None if self.production_custody is None else self.production_custody.digest),
            "status": self.status.value,
            "schedule": self.schedule.to_payload(),
            "base_archive_sha256": self.base_archive_sha256,
            "base_action": self.base_action.to_payload(),
            "traces": [trace.to_payload() for trace in self.traces],
            "best_strict_improvement_identity": self.best_strict_improvement_identity,
            "governed_handoff_identity": self.governed_handoff_identity,
            "governed_handoff_eligible": self.governed_handoff_eligible,
            "candidate_evaluations": self.candidate_evaluations,
            "shared_base_evaluations": self.shared_base_evaluations,
            "resume_records_reused": self.resume_records_reused,
            "resume_records_written": self.resume_records_written,
            "resume_ledger_path": self.resume_ledger_path,
            "resume_ledger_sha256": self.resume_ledger_sha256,
            "outer_governor_required": True,
            "external_governor_blocker": (
                "Production custody identifies artifacts and callbacks by self-attested "
                "hashes; this module does not independently verify the contest execution."
            ),
            "promotion_allowed": self.promotion_allowed,
            "score_claim": False,
            "pointer_moved": False,
        }


def precommit_qdbs_schedule(
    signed_singletons: Sequence[DescriptionProposal],
    grouped_proposals: Sequence[DescriptionProposal],
    *,
    coordinate_count: int,
    active_indices: Sequence[int],
    seed: int,
) -> QDBSCandidateSchedule:
    """Freeze the exact 24+24 schedule before any hard evaluation occurs."""

    coordinate_count = _integer(coordinate_count, "coordinate_count", minimum=2)
    seed = _integer(seed, "seed", minimum=0)
    active = tuple(_integer(index, "active_indices value", minimum=0) for index in active_indices)
    if not active or active != tuple(sorted(set(active))):
        raise FD2QDBSError("active_indices must be nonempty, unique, and sorted")
    if active[-1] >= coordinate_count:
        raise FD2QDBSError("active_indices contains an out-of-bounds coordinate")
    active_set = frozenset(active)
    singletons = tuple(signed_singletons)
    groups = tuple(grouped_proposals)
    if len(singletons) != SCORER_SINGLETON_COUNT or any(
        proposal.proposal_class is not ProposalClass.SCORER_SINGLETON for proposal in singletons
    ):
        raise FD2QDBSError("QDBS requires exactly 16 scorer signed singletons")
    if len(groups) != SCORER_GROUP_COUNT or any(
        proposal.proposal_class is not ProposalClass.SCORER_GROUP for proposal in groups
    ):
        raise FD2QDBSError("QDBS requires exactly 8 scorer grouped proposals")
    scorer = singletons + groups
    identities = tuple(proposal.identity for proposal in scorer)
    if len(set(identities)) != len(identities):
        raise FD2QDBSError("scorer proposal identities must be unique")
    scorer_delta_sets = tuple(tuple((delta.index, delta.delta) for delta in proposal.deltas) for proposal in scorer)
    if len(set(scorer_delta_sets)) != len(scorer_delta_sets):
        raise FD2QDBSError("scorer proposal delta sets must be distinct")
    for proposal in scorer:
        if any(delta.index >= coordinate_count for delta in proposal.deltas):
            raise FD2QDBSError("scorer proposal coordinate is out of bounds")
        if any(delta.index not in active_set for delta in proposal.deltas):
            raise FD2QDBSError("scorer proposal coordinate is outside active_indices")
        if len(proposal.deltas) > len(active):
            raise FD2QDBSError("proposal is wider than the active coordinate set")

    rng = np.random.default_rng(seed)
    controls: list[DescriptionProposal] = []
    used_delta_sets = set(scorer_delta_sets)
    for ordinal, proposal in enumerate(scorer):
        width = len(proposal.deltas)
        chosen: tuple[DescriptionDelta, ...] | None = None
        for _ in range(128):
            indices = rng.choice(np.asarray(active, dtype=np.int64), size=width, replace=False)
            trial = tuple(
                sorted(
                    DescriptionDelta(int(index), source_delta.delta)
                    for index, source_delta in zip(indices, proposal.deltas, strict=True)
                )
            )
            key = tuple((delta.index, delta.delta) for delta in trial)
            if key not in used_delta_sets:
                chosen = trial
                used_delta_sets.add(key)
                break
        if chosen is None:
            raise FD2QDBSError("cannot precommit a unique matched random control")
        identity_material = {
            "ordinal": ordinal,
            "matched": proposal.identity,
            "deltas": [delta.to_payload() for delta in chosen],
            "seed": seed,
        }
        identity_hash = sha256(_canonical_json(identity_material)).hexdigest()[:16]
        controls.append(
            DescriptionProposal(
                identity=f"random_control_{ordinal:02d}_{identity_hash}",
                proposal_class=ProposalClass.RANDOM_CONTROL,
                deltas=chosen,
                signal_label="precommitted_matched_integer_random",
                signal_value=proposal.signal_value,
                matched_scorer_identity=proposal.identity,
            )
        )

    if len(controls) != RANDOM_CONTROL_COUNT:
        raise AssertionError("internal QDBS control count differs")
    all_identities = ("__base__", *identities, *(control.identity for control in controls))
    if len(all_identities) != MAX_CANDIDATE_EVALUATIONS + 1 or len(set(all_identities)) != len(all_identities):
        raise FD2QDBSError("all QDBS base/candidate identities must be unique and reserved")
    unsigned_payload = {
        "seed": seed,
        "coordinate_count": coordinate_count,
        "active_indices": list(active),
        "scorer_proposals": [proposal.to_payload() for proposal in scorer],
        "random_controls": [proposal.to_payload() for proposal in controls],
    }
    return QDBSCandidateSchedule(
        seed=seed,
        coordinate_count=coordinate_count,
        active_indices=active,
        scorer_proposals=scorer,
        random_controls=tuple(controls),
        schedule_sha256=sha256(_canonical_json(unsigned_payload)).hexdigest(),
    )


def apply_description_proposal(base_theta: np.ndarray, proposal: DescriptionProposal) -> np.ndarray:
    """Apply one precommitted unit proposal without float coercion."""

    base = _immutable_i64(base_theta, "base_theta")
    candidate = np.array(base, dtype=np.int64, copy=True)
    for delta in proposal.deltas:
        if delta.index >= candidate.size:
            raise FD2QDBSError("proposal coordinate is out of bounds")
        value = int(candidate[delta.index])
        if delta.delta > 0 and value == np.iinfo(np.int64).max:
            raise FD2QDBSError("description coordinate overflows int64")
        if delta.delta < 0 and value == np.iinfo(np.int64).min:
            raise FD2QDBSError("description coordinate underflows int64")
        candidate[delta.index] = value + delta.delta
    return candidate


def _validate_action_binding(
    action: RealizedJointAction,
    *,
    archive_sha256: str,
    archive_bytes: int,
    evaluation_idempotency_key: str,
    authority_mode: QDBSAuthorityMode,
    production_custody: ProductionCustody | None,
) -> None:
    if not action.realized:
        raise FD2QDBSError("joint-action callback is not a realized verdict")
    if action.archive_sha256 != archive_sha256:
        raise FD2QDBSError("joint-action archive SHA differs from compiled bytes")
    if action.archive_bytes != archive_bytes:
        raise FD2QDBSError("joint-action archive byte count differs from compiled bytes")
    if action.evaluation_idempotency_key != evaluation_idempotency_key:
        raise FD2QDBSError("joint-action idempotency key differs")
    if authority_mode is QDBSAuthorityMode.PRODUCTION_FULL_N600:
        if production_custody is None:
            raise FD2QDBSError("production QDBS requires typed production custody")
        if not action.full_n600:
            raise FD2QDBSError("production QDBS requires a full-n600 authority verdict")
        if action.custody_digest != production_custody.digest:
            raise FD2QDBSError("production verdict custody digest differs")


def _validate_callback_custody(
    callbacks: DescriptionHardOracleCallbacks,
    production_custody: ProductionCustody,
) -> None:
    expected = {
        "compiler_sha256": production_custody.compiler_sha256,
        "receiver_sha256": production_custody.receiver_sha256,
        "evaluator_sha256": production_custody.evaluator_sha256,
        "evaluation_protocol_sha256": production_custody.evaluation_protocol_sha256,
    }
    for name, digest in expected.items():
        if getattr(callbacks, name) != digest:
            raise FD2QDBSError(f"production callback {name} differs from custody")


def _resume_binding(
    base_theta: np.ndarray,
    schedule: QDBSCandidateSchedule,
    production_custody: ProductionCustody,
    callbacks: DescriptionHardOracleCallbacks,
) -> dict[str, str]:
    return {
        "base_theta_sha256": _theta_digest(base_theta),
        "schedule_sha256": schedule.schedule_sha256,
        "active_indices_sha256": schedule.active_indices_sha256,
        "production_custody_digest": production_custody.digest,
        "compiler_sha256": _sha256_text(callbacks.compiler_sha256, "callbacks.compiler_sha256"),
        "receiver_sha256": _sha256_text(callbacks.receiver_sha256, "callbacks.receiver_sha256"),
        "evaluator_sha256": _sha256_text(callbacks.evaluator_sha256, "callbacks.evaluator_sha256"),
        "evaluation_protocol_sha256": _sha256_text(
            callbacks.evaluation_protocol_sha256,
            "callbacks.evaluation_protocol_sha256",
        ),
        "authority_mode": QDBSAuthorityMode.PRODUCTION_FULL_N600.value,
    }


def _evaluation_idempotency_key(
    *,
    binding: dict[str, str] | None,
    schedule: QDBSCandidateSchedule,
    authority_mode: QDBSAuthorityMode,
    ordinal: int,
    identity: str,
    theta: np.ndarray,
    archive_sha256: str,
    archive_bytes: int,
) -> str:
    return _payload_digest(
        {
            "schema": _EVALUATION_KEY_SCHEMA,
            "resume_binding_sha256": (None if binding is None else _payload_digest(binding)),
            "schedule_sha256": schedule.schedule_sha256,
            "authority_mode": authority_mode.value,
            "ordinal": ordinal,
            "identity": identity,
            "theta_sha256": _theta_digest(theta),
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
        }
    )


def _resume_artifact_directory(ledger_path: Path) -> Path:
    return ledger_path.with_name(f"{ledger_path.name}.archives")


def _archive_artifact_name(ordinal: int, archive_sha256: str) -> str:
    prefix = "base" if ordinal == -1 else f"candidate_{ordinal:02d}"
    return f"{prefix}_{_sha256_text(archive_sha256, 'archive artifact SHA')}.bin"


def _atomic_write_archive_artifact(
    ledger_path: Path,
    *,
    ordinal: int,
    archive_sha256: str,
    compiled: bytes,
) -> str:
    if sha256(compiled).hexdigest() != archive_sha256:
        raise FD2QDBSError("archive artifact bytes differ from compiled SHA")
    directory = _resume_artifact_directory(ledger_path)
    _reject_symlink_components(directory.parent, "resume archive directory")
    directory.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(directory, "resume archive directory")
    if not directory.is_dir():
        raise FD2QDBSError("resume archive directory is not a directory")
    name = _archive_artifact_name(ordinal, archive_sha256)
    path = directory / name
    if path.is_symlink():
        raise FD2QDBSError("resume archive artifact cannot be a symlink")
    if path.exists():
        if not path.is_file():
            raise FD2QDBSError("resume archive artifact is not a file")
        existing = _read_bytes_nofollow(path, "resume archive artifact")
        if existing != compiled:
            raise FD2QDBSError("resume archive artifact differs from compiled bytes")
        return name
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=f".{name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(compiled)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_symlink_components(directory, "resume archive directory")
        if path.is_symlink():
            raise FD2QDBSError("resume archive artifact cannot be a symlink")
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return name


def _verify_archive_artifact(
    ledger_path: Path,
    *,
    ordinal: int,
    archive_sha256: str,
    archive_bytes: int,
    artifact_name: object,
) -> bytes:
    expected_name = _archive_artifact_name(ordinal, archive_sha256)
    if artifact_name != expected_name:
        raise FD2QDBSError("resume archive artifact name differs")
    directory = _resume_artifact_directory(ledger_path)
    _reject_symlink_components(directory, "resume archive directory")
    if not directory.is_dir():
        raise FD2QDBSError("resume archive directory is unavailable or symlinked")
    path = directory / expected_name
    if not path.is_file():
        raise FD2QDBSError("resume archive artifact is unavailable or symlinked")
    compiled = _read_bytes_nofollow(path, "resume archive artifact")
    if len(compiled) != archive_bytes:
        raise FD2QDBSError("resume archive artifact byte count differs")
    if sha256(compiled).hexdigest() != archive_sha256:
        raise FD2QDBSError("resume archive artifact SHA differs")
    return compiled


def _resume_record(
    *,
    ordinal: int,
    identity: str,
    theta: np.ndarray,
    archive_sha256: str,
    archive_bytes: int,
    archive_artifact_name: str,
    evaluation_idempotency_key: str,
    action: RealizedJointAction | None,
    previous_record_sha256: str | None,
) -> dict[str, Any]:
    evaluation_idempotency_key = _sha256_text(
        evaluation_idempotency_key,
        "resume evaluation idempotency key",
    )
    body = {
        "ordinal": ordinal,
        "identity": identity,
        "theta_sha256": _theta_digest(theta),
        "archive_sha256": archive_sha256,
        "archive_bytes": _integer(
            archive_bytes,
            "resume archive bytes",
            minimum=1,
        ),
        "archive_artifact_name": archive_artifact_name,
        "evaluation_idempotency_key": evaluation_idempotency_key,
        "state": _RECORD_PENDING if action is None else _RECORD_COMPLETE,
        "action": None if action is None else action.to_payload(),
        "previous_record_sha256": previous_record_sha256,
    }
    return {**body, "record_sha256": _payload_digest(body)}


def _ledger_payload(binding: dict[str, str], records: list[dict[str, Any]]) -> dict[str, Any]:
    body = {
        "schema": _RESUME_SCHEMA,
        "binding": binding,
        "binding_sha256": _payload_digest(binding),
        "records": records,
    }
    return {**body, "ledger_sha256": _payload_digest(body)}


def _atomic_write_ledger(
    path: Path,
    binding: dict[str, str],
    records: list[dict[str, Any]],
) -> str:
    payload = _ledger_payload(binding, records)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    _reject_symlink_components(path.parent, "resume ledger path")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, "resume ledger path")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_symlink_components(path, "resume ledger path")
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return str(payload["ledger_sha256"])


def _load_resume_ledger(
    path: Path,
    binding: dict[str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    _reject_symlink_components(path, "resume ledger path")
    if not path.exists():
        return [], None
    if not path.is_file():
        raise FD2QDBSError("resume ledger path is not a file")
    try:
        payload = json.loads(_read_bytes_nofollow(path, "resume ledger path").decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FD2QDBSError("resume ledger is unreadable or malformed") from exc
    if not isinstance(payload, dict):
        raise FD2QDBSError("resume ledger root must be an object")
    allowed = {"schema", "binding", "binding_sha256", "records", "ledger_sha256"}
    if set(payload) != allowed:
        raise FD2QDBSError("resume ledger fields differ")
    if payload["schema"] != _RESUME_SCHEMA or payload["binding"] != binding:
        raise FD2QDBSError("resume ledger binding differs")
    if payload["binding_sha256"] != _payload_digest(binding):
        raise FD2QDBSError("resume ledger binding checksum differs")
    body = {key: payload[key] for key in ("schema", "binding", "binding_sha256", "records")}
    if payload["ledger_sha256"] != _payload_digest(body):
        raise FD2QDBSError("resume ledger checksum differs")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or len(raw_records) > MAX_CANDIDATE_EVALUATIONS + 1:
        raise FD2QDBSError("resume ledger record count differs")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for position, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise FD2QDBSError("resume ledger record must be an object")
        record_allowed = {
            "ordinal",
            "identity",
            "theta_sha256",
            "archive_sha256",
            "archive_bytes",
            "archive_artifact_name",
            "evaluation_idempotency_key",
            "state",
            "action",
            "previous_record_sha256",
            "record_sha256",
        }
        if set(raw_record) != record_allowed:
            raise FD2QDBSError("resume ledger record fields differ")
        ordinal = _integer(raw_record["ordinal"], "resume ordinal")
        if ordinal != position - 1:
            raise FD2QDBSError("resume ledger ordinals are not a contiguous prefix")
        if raw_record["previous_record_sha256"] != previous:
            raise FD2QDBSError("resume ledger hash chain differs")
        record_body = {key: raw_record[key] for key in record_allowed if key != "record_sha256"}
        record_sha = _sha256_text(raw_record["record_sha256"], "resume record checksum")
        if record_sha != _payload_digest(record_body):
            raise FD2QDBSError("resume ledger record checksum differs")
        _text(raw_record["identity"], "resume record identity")
        _sha256_text(raw_record["theta_sha256"], "resume theta SHA")
        archive_sha = _sha256_text(raw_record["archive_sha256"], "resume archive SHA")
        archive_bytes = _integer(
            raw_record["archive_bytes"],
            "resume archive bytes",
            minimum=1,
        )
        evaluation_key = _sha256_text(
            raw_record["evaluation_idempotency_key"],
            "resume evaluation idempotency key",
        )
        state = raw_record["state"]
        if state == _RECORD_PENDING:
            if position != len(raw_records) - 1 or raw_record["action"] is not None:
                raise FD2QDBSError("resume pending intent must be the final actionless record")
        elif state == _RECORD_COMPLETE:
            action = RealizedJointAction.from_payload(raw_record["action"])
            if action.archive_sha256 != archive_sha:
                raise FD2QDBSError("resume action archive SHA differs from record")
            if action.archive_bytes != archive_bytes:
                raise FD2QDBSError("resume action archive bytes differ from record")
            if action.evaluation_idempotency_key != evaluation_key:
                raise FD2QDBSError("resume action idempotency key differs from record")
        else:
            raise FD2QDBSError("resume record state differs")
        _verify_archive_artifact(
            path,
            ordinal=ordinal,
            archive_sha256=archive_sha,
            archive_bytes=archive_bytes,
            artifact_name=raw_record["archive_artifact_name"],
        )
        previous = record_sha
        records.append(raw_record)
    return records, str(payload["ledger_sha256"])


@contextmanager
def _exclusive_resume_lock(path: Path) -> Iterator[None]:
    _reject_symlink_components(path.parent, "resume ledger path")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, "resume ledger path")
    lock_path = path.with_name(f"{path.name}.lock")
    _reject_symlink_components(lock_path, "resume ledger lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise FD2QDBSError("resume ledger lock is unavailable or symlinked") from exc
    with os.fdopen(descriptor, "a+b") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise FD2QDBSError("resume ledger lock is not a regular file")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_and_consume_archive(
    compiled: bytes,
    theta: np.ndarray,
    callbacks: DescriptionHardOracleCallbacks,
) -> tuple[str, ConsumedDescriptionCandidate]:
    archive_sha = sha256(compiled).hexdigest()
    parsed = callbacks.parse_archive(compiled)
    if not isinstance(parsed, ParsedDescriptionCandidate):
        raise FD2QDBSError("parse callback must return ParsedDescriptionCandidate")
    if not parsed.exact_parseback:
        raise FD2QDBSError("parse callback did not certify exact parse-back")
    if parsed.archive_sha256 != archive_sha:
        raise FD2QDBSError("parse-back archive SHA differs from compiled bytes")
    if not np.array_equal(parsed.realized_theta, theta):
        raise FD2QDBSError("parse-back realized description differs from candidate")
    consumed = callbacks.consume_archive(parsed)
    if not isinstance(consumed, ConsumedDescriptionCandidate):
        raise FD2QDBSError("consume callback must return ConsumedDescriptionCandidate")
    if not consumed.exact_consumption:
        raise FD2QDBSError("receiver did not certify exact consumption")
    if consumed.archive_sha256 != archive_sha:
        raise FD2QDBSError("consumed archive SHA differs from compiled bytes")
    if not np.array_equal(consumed.realized_theta, theta):
        raise FD2QDBSError("receiver-consumed description differs from candidate")
    return archive_sha, consumed


def _prepare_hard_candidate(
    theta: np.ndarray,
    proposal: DescriptionProposal | None,
    callbacks: DescriptionHardOracleCallbacks,
) -> tuple[bytes, str, ConsumedDescriptionCandidate]:
    compiled = callbacks.compile_archive(theta, proposal)
    if not isinstance(compiled, bytes) or not compiled:
        raise FD2QDBSError("compile callback must return nonempty bytes")
    archive_sha, consumed = _parse_and_consume_archive(compiled, theta, callbacks)
    return compiled, archive_sha, consumed


def _evaluate_prepared_candidate(
    consumed: ConsumedDescriptionCandidate,
    evaluation_idempotency_key: str,
    callbacks: DescriptionHardOracleCallbacks,
    *,
    archive_sha256: str,
    archive_bytes: int,
    authority_mode: QDBSAuthorityMode,
    production_custody: ProductionCustody | None,
) -> RealizedJointAction:
    action = callbacks.evaluate_joint_action_idempotent(
        consumed,
        evaluation_idempotency_key,
    )
    if not isinstance(action, RealizedJointAction):
        raise FD2QDBSError("idempotent joint-action callback must return RealizedJointAction")
    _validate_action_binding(
        action,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        evaluation_idempotency_key=evaluation_idempotency_key,
        authority_mode=authority_mode,
        production_custody=production_custody,
    )
    return action


def run_fd2_qdbs_terminal(
    base_theta: np.ndarray,
    signed_singletons: Sequence[DescriptionProposal],
    grouped_proposals: Sequence[DescriptionProposal],
    callbacks: DescriptionHardOracleCallbacks,
    *,
    active_indices: Sequence[int],
    seed: int,
    authority_mode: QDBSAuthorityMode,
    production_custody: ProductionCustody | None = None,
    resume_ledger_path: Path | None = None,
) -> QDBSTerminalResult:
    """Evaluate the fixed row without granting handoff or promotion authority."""

    if not isinstance(authority_mode, QDBSAuthorityMode):
        raise FD2QDBSError("authority_mode must be QDBSAuthorityMode")
    if not isinstance(callbacks, DescriptionHardOracleCallbacks):
        raise FD2QDBSError("callbacks must be DescriptionHardOracleCallbacks")
    if authority_mode is QDBSAuthorityMode.PRODUCTION_FULL_N600:
        if not isinstance(production_custody, ProductionCustody):
            raise FD2QDBSError("production QDBS requires typed production custody")
        if not isinstance(resume_ledger_path, Path):
            raise FD2QDBSError("production QDBS requires a resume ledger Path")
        if not resume_ledger_path.is_absolute():
            raise FD2QDBSError("production QDBS resume ledger path must be absolute")
        _validate_callback_custody(callbacks, production_custody)
    elif production_custody is not None or resume_ledger_path is not None:
        raise FD2QDBSError("stale rehearsal cannot carry production custody or resume ledger")
    base = _immutable_i64(base_theta, "base_theta")
    schedule = precommit_qdbs_schedule(
        signed_singletons,
        grouped_proposals,
        coordinate_count=base.size,
        active_indices=active_indices,
        seed=seed,
    )
    if len(schedule.candidates) != MAX_CANDIDATE_EVALUATIONS:
        raise AssertionError("internal QDBS candidate budget differs")

    def execute() -> QDBSTerminalResult:
        binding: dict[str, str] | None = None
        records: list[dict[str, Any]] = []
        ledger_sha: str | None = None
        if production_custody is not None:
            binding = _resume_binding(base, schedule, production_custody, callbacks)
            assert resume_ledger_path is not None
            records, ledger_sha = _load_resume_ledger(resume_ledger_path, binding)
        reused = 0
        written = 0

        def validate_base_parent(
            ordinal: int,
            archive_sha: str,
            archive_bytes: int,
        ) -> None:
            if (
                ordinal == -1
                and production_custody is not None
                and (
                    archive_sha != production_custody.parent_archive_sha256
                    or archive_bytes != production_custody.parent_archive_bytes
                )
            ):
                raise FD2QDBSError("production base archive differs from custody parent")

        def evaluate_or_resume(
            ordinal: int,
            identity: str,
            theta: np.ndarray,
            proposal: DescriptionProposal | None,
        ) -> tuple[str, RealizedJointAction]:
            nonlocal ledger_sha, reused, written
            position = ordinal + 1
            if binding is not None and position < len(records):
                record = records[position]
                if record["identity"] != identity or record["theta_sha256"] != _theta_digest(theta):
                    raise FD2QDBSError("resume record candidate identity differs")
                archive_sha = _sha256_text(record["archive_sha256"], "resume archive SHA")
                archive_bytes = _integer(
                    record["archive_bytes"],
                    "resume archive bytes",
                    minimum=1,
                )
                evaluation_key = _sha256_text(
                    record["evaluation_idempotency_key"],
                    "resume evaluation idempotency key",
                )
                expected_key = _evaluation_idempotency_key(
                    binding=binding,
                    schedule=schedule,
                    authority_mode=authority_mode,
                    ordinal=ordinal,
                    identity=identity,
                    theta=theta,
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                )
                if evaluation_key != expected_key:
                    raise FD2QDBSError("resume evaluation idempotency key does not re-derive")
                validate_base_parent(ordinal, archive_sha, archive_bytes)
                if record["state"] == _RECORD_COMPLETE:
                    action = RealizedJointAction.from_payload(record["action"])
                    _validate_action_binding(
                        action,
                        archive_sha256=archive_sha,
                        archive_bytes=archive_bytes,
                        evaluation_idempotency_key=evaluation_key,
                        authority_mode=authority_mode,
                        production_custody=production_custody,
                    )
                    reused += 1
                    return archive_sha, action
                if record["state"] != _RECORD_PENDING:
                    raise FD2QDBSError("resume record state differs")
                assert resume_ledger_path is not None
                compiled = _verify_archive_artifact(
                    resume_ledger_path,
                    ordinal=ordinal,
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    artifact_name=record["archive_artifact_name"],
                )
                parsed_sha, consumed = _parse_and_consume_archive(
                    compiled,
                    theta,
                    callbacks,
                )
                if parsed_sha != archive_sha:
                    raise FD2QDBSError("pending archive SHA differs after parse-back")
                action = _evaluate_prepared_candidate(
                    consumed,
                    evaluation_key,
                    callbacks,
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    authority_mode=authority_mode,
                    production_custody=production_custody,
                )
                records[position] = _resume_record(
                    ordinal=ordinal,
                    identity=identity,
                    theta=theta,
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    archive_artifact_name=record["archive_artifact_name"],
                    evaluation_idempotency_key=evaluation_key,
                    action=action,
                    previous_record_sha256=record["previous_record_sha256"],
                )
                ledger_sha = _atomic_write_ledger(
                    resume_ledger_path,
                    binding,
                    records,
                )
                written += 1
                return archive_sha, action
            if binding is not None and position != len(records):
                raise FD2QDBSError("resume ledger is not a contiguous evaluation prefix")
            compiled, archive_sha, consumed = _prepare_hard_candidate(
                theta,
                proposal,
                callbacks,
            )
            archive_bytes = len(compiled)
            validate_base_parent(ordinal, archive_sha, archive_bytes)
            evaluation_key = _evaluation_idempotency_key(
                binding=binding,
                schedule=schedule,
                authority_mode=authority_mode,
                ordinal=ordinal,
                identity=identity,
                theta=theta,
                archive_sha256=archive_sha,
                archive_bytes=archive_bytes,
            )
            if binding is not None:
                assert resume_ledger_path is not None
                artifact_name = _atomic_write_archive_artifact(
                    resume_ledger_path,
                    ordinal=ordinal,
                    archive_sha256=archive_sha,
                    compiled=compiled,
                )
                previous = None if not records else str(records[-1]["record_sha256"])
                records.append(
                    _resume_record(
                        ordinal=ordinal,
                        identity=identity,
                        theta=theta,
                        archive_sha256=archive_sha,
                        archive_bytes=archive_bytes,
                        archive_artifact_name=artifact_name,
                        evaluation_idempotency_key=evaluation_key,
                        action=None,
                        previous_record_sha256=previous,
                    )
                )
                ledger_sha = _atomic_write_ledger(resume_ledger_path, binding, records)
            action = _evaluate_prepared_candidate(
                consumed,
                evaluation_key,
                callbacks,
                archive_sha256=archive_sha,
                archive_bytes=archive_bytes,
                authority_mode=authority_mode,
                production_custody=production_custody,
            )
            if binding is not None:
                assert resume_ledger_path is not None
                pending = records[-1]
                records[-1] = _resume_record(
                    ordinal=ordinal,
                    identity=identity,
                    theta=theta,
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    archive_artifact_name=pending["archive_artifact_name"],
                    evaluation_idempotency_key=evaluation_key,
                    action=action,
                    previous_record_sha256=pending["previous_record_sha256"],
                )
                ledger_sha = _atomic_write_ledger(resume_ledger_path, binding, records)
                written += 1
            return archive_sha, action

        base_sha, base_action = evaluate_or_resume(-1, "__base__", base, None)
        traces: list[QDBSCandidateTrace] = []
        best_trace: QDBSCandidateTrace | None = None
        for ordinal, proposal in enumerate(schedule.candidates):
            candidate = apply_description_proposal(base, proposal)
            archive_sha, action = evaluate_or_resume(ordinal, proposal.identity, candidate, proposal)
            delta = action.action - base_action.action
            strict = action.action < base_action.action
            trace = QDBSCandidateTrace(
                ordinal=ordinal,
                identity=proposal.identity,
                proposal_class=proposal.proposal_class,
                archive_sha256=archive_sha,
                action=action,
                delta_vs_base=delta,
                strict_realized_improvement=strict,
                governed_handoff_eligible=False,
            )
            traces.append(trace)
            if strict and (best_trace is None or action.action < best_trace.action.action):
                best_trace = trace

        if len(traces) != MAX_CANDIDATE_EVALUATIONS:
            raise AssertionError("QDBS candidate budget differs")
        if authority_mode is QDBSAuthorityMode.STALE_REHEARSAL:
            status = QDBSStatus.REHEARSAL_NONPROMOTABLE
            governed_handoff_identity = None
        elif best_trace is None:
            status = QDBSStatus.NO_STRICT_IMPROVEMENT
            governed_handoff_identity = None
        else:
            status = QDBSStatus.REQUIRES_EXTERNAL_GOVERNOR
            governed_handoff_identity = None

        return QDBSTerminalResult(
            authority_mode=authority_mode,
            production_custody=production_custody,
            status=status,
            schedule=schedule,
            base_archive_sha256=base_sha,
            base_action=base_action,
            traces=tuple(traces),
            best_strict_improvement_identity=None if best_trace is None else best_trace.identity,
            governed_handoff_identity=governed_handoff_identity,
            candidate_evaluations=len(traces),
            shared_base_evaluations=1,
            resume_records_reused=reused,
            resume_records_written=written,
            resume_ledger_path=(None if resume_ledger_path is None else str(resume_ledger_path)),
            resume_ledger_sha256=ledger_sha,
        )

    if resume_ledger_path is None:
        return execute()
    with _exclusive_resume_lock(resume_ledger_path):
        return execute()


__all__ = [
    "FULL_N600_AUTHORITY_MARKER",
    "FULL_N600_SAMPLE_COUNT",
    "MAX_CANDIDATE_EVALUATIONS",
    "RANDOM_CONTROL_COUNT",
    "SCORER_GROUP_COUNT",
    "SCORER_SINGLETON_COUNT",
    "STALE_REHEARSAL_AUTHORITY_MARKER",
    "ConsumedDescriptionCandidate",
    "ContestAxis",
    "DescriptionDelta",
    "DescriptionHardOracleCallbacks",
    "DescriptionProposal",
    "FD2QDBSError",
    "ParsedDescriptionCandidate",
    "ProductionCustody",
    "ProposalClass",
    "QDBSAuthorityMode",
    "QDBSCandidateSchedule",
    "QDBSCandidateTrace",
    "QDBSStatus",
    "QDBSTerminalResult",
    "RealizedJointAction",
    "apply_description_proposal",
    "precommit_qdbs_schedule",
    "run_fd2_qdbs_terminal",
]
