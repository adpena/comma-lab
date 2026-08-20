# SPDX-License-Identifier: MIT
"""Proof-byte verifier for finite-horizon continuation bounds.

This module closes one narrow but load-bearing false-authority surface: a
favorable scalar plus a syntactically valid SHA-256 is not a subtree bound.
Only a canonical, self-hashed proof envelope whose exact terminal-leaf set
matches an independently supplied generator-manifest root can construct a
``VerifiedContinuationCertificateV1``.

The verifier is structural and ``research_only``.  It neither runs nor stands
in for the public evaluator.  Exact terminal scores reproduce the pinned
evaluator's binary64 operation order from receipt-bound component bits.
Primitive bounds fail closed until a typed interval-theorem verifier exists.
Generator-manifest and evidence dependency bytes are reopened, strictly
parsed, and semantically joined instead of treating SHA strings as proofs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, cast

from tac.contest_score import (
    POSE_WEIGHT,
    RATE_WEIGHT,
    SEG_WEIGHT,
    UNCOMPRESSED_SIZE_BYTES,
)

SCHEMA: Final = "tac.verified_continuation_certificate.v1"
PROOF_BODY_SCHEMA: Final = "tac.verified_continuation_proof_body.v1"
PROOF_ENVELOPE_SCHEMA: Final = "tac.verified_continuation_proof_envelope.v1"
BASE_STOP_ACTION_ID: Final = "BASE_STOP"
GENERATOR_DOMAIN_MANIFEST_SCHEMA: Final = "tac.generator_domain_manifest.v1"
FINITE_MEASUREMENT_RECEIPT_SCHEMA: Final = "tac.continuation_finite_measurement_receipt.v1"
HARD_CONSTRAINT_RECEIPT_SCHEMA: Final = "tac.continuation_hard_constraint_receipt.v1"
PINNED_UPSTREAM_EVALUATE_SHA256: Final = "7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b"

AuthorityMode = Literal[
    "RESEARCH_ADVISORY",
    "PRODUCTION_CONTEST_CPU",
    "PRODUCTION_CONTEST_CUDA",
]
DerivationMethod = Literal[
    "FINITE_TERMINAL_LEAF_ENUMERATION",
    "PER_LEAF_CONSERVATIVE_PRIMITIVE_INTERVALS",
    "BASE_STOP_EXACT_BASE",
]
VerifierStatus = Literal[
    "VERIFIED_FINITE_TERMINAL_ENUMERATION",
    "VERIFIED_CONSERVATIVE_PRIMITIVE_INTERVALS",
    "VERIFIED_BASE_STOP_EXACT_SCORE",
]

FINITE_ROUNDING_POLICY: Final = "EXACT_FINITE_ENUMERATION_CANONICAL_SCORE_V1"
PRIMITIVE_ROUNDING_POLICY: Final = "UNADMITTED_UNTIL_TYPED_INTERVAL_THEOREM_V1"
BASE_STOP_ROUNDING_POLICY: Final = "EXACT_BASE_CANONICAL_SCORE_V1"

_AUTHORITY_MODES: Final = frozenset(
    {
        "RESEARCH_ADVISORY",
        "PRODUCTION_CONTEST_CPU",
        "PRODUCTION_CONTEST_CUDA",
    }
)
_DERIVATION_METHODS: Final = frozenset(
    {
        "FINITE_TERMINAL_LEAF_ENUMERATION",
        "PER_LEAF_CONSERVATIVE_PRIMITIVE_INTERVALS",
        "BASE_STOP_EXACT_BASE",
    }
)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_MAX_PROOF_INTEGER: Final = (1 << 63) - 1

_ENVELOPE_FIELDS: Final = {
    "schema",
    "proof_body",
    "proof_body_sha256",
}
_BODY_FIELDS: Final = {
    "schema",
    "derivation_method",
    "exact_base_identity_sha256",
    "epoch_id",
    "generator_domain_manifest_sha256",
    "subtree_root_sha256",
    "subtree_cardinality",
    "first_action_id",
    "support_hyperedge_sha256",
    "continuation_equivalence_sha256",
    "horizon",
    "authority_mode",
    "axis",
    "hard_constraint_contract_sha256",
    "score_implementation_sha256",
    "covered_terminal_leaf_ids",
    "proof_dependency_sha256s",
    "outward_rounding_policy",
    "finite_terminal_leaves",
    "primitive_interval_leaves",
}
_FINITE_LEAF_FIELDS: Final = {
    "terminal_leaf_id",
    "terminal_endpoint_identity_sha256",
    "action_path",
    "d_seg",
    "d_pose",
    "archive_nbytes",
    "measurement_receipt_sha256",
    "hard_constraint_receipt_sha256",
}
_PRIMITIVE_LEAF_FIELDS: Final = {
    "terminal_leaf_id",
    "d_seg_lower",
    "d_seg_upper",
    "d_pose_lower",
    "d_pose_upper",
    "archive_nbytes_lower",
    "archive_nbytes_upper",
    "interval_derivation_receipt_sha256",
    "leaf_coverage_receipt_sha256",
}
_GENERATOR_MANIFEST_FIELDS: Final = {
    "schema",
    "manifest_id",
    "epoch_id",
    "exact_base_identity_sha256",
    "generator_program_sha256",
    "enumeration_receipt_sha256",
    "parameter_domain_manifest_sha256",
    "chronology_domain_manifest_sha256",
    "deterministic_seed",
    "horizon",
    "maximum_interaction_order",
    "families",
    "scales",
    "terminal_leaves",
    "generated_leaf_cardinality",
    "generated_leaf_merkle_root_sha256",
    "generated_terminal_leaf_id_root_sha256",
}
_GENERATOR_LEAF_FIELDS: Final = {
    "terminal_leaf_id",
    "first_action_id",
    "action_path",
    "support_hyperedge",
    "parent_action_ids",
    "parameter_identity_sha256",
    "chronology_context_sha256",
    "exact_base_identity_sha256",
    "universe_epoch_id",
}
_SUPPORT_CELL_FIELDS: Final = {"family", "scale"}
_ACTION_FAMILIES: Final = (
    "PRUNE",
    "MERGE",
    "MACRO_EVICT",
    "MIGRATE",
    "INVERSE_REPAIR",
    "FIT_QUOTIENT",
    "TRAIN_QUOTIENT",
    "REQUANTIZE_STORAGE",
    "JOINT_DESCENT",
)
_ACTION_SCALES: Final = ("MICRO", "MESO", "MACRO", "SYSTEM")
_FINITE_MEASUREMENT_RECEIPT_FIELDS: Final = {
    "schema",
    "receipt_role",
    "exact_base_identity_sha256",
    "epoch_id",
    "generator_domain_manifest_sha256",
    "terminal_leaf_id",
    "terminal_endpoint_identity_sha256",
    "action_path",
    "first_action_id",
    "support_hyperedge_sha256",
    "continuation_equivalence_sha256",
    "horizon",
    "authority_mode",
    "axis",
    "score_implementation_sha256",
    "upstream_evaluate_sha256",
    "d_seg_binary64_hex",
    "d_pose_binary64_hex",
    "archive_nbytes",
    "archive_sha256",
    "realized_output_sha256",
    "evaluator_measurement_receipt_sha256",
    "public_auth_eval_receipt_sha256",
    "research_only",
}
_HARD_CONSTRAINT_RECEIPT_FIELDS: Final = {
    "schema",
    "receipt_role",
    "exact_base_identity_sha256",
    "epoch_id",
    "generator_domain_manifest_sha256",
    "terminal_leaf_id",
    "terminal_endpoint_identity_sha256",
    "action_path",
    "first_action_id",
    "support_hyperedge_sha256",
    "continuation_equivalence_sha256",
    "horizon",
    "authority_mode",
    "axis",
    "hard_constraint_contract_sha256",
    "decode_constraint_receipt_sha256",
    "hard_constraints_satisfied",
    "blocker_codes",
    "research_only",
}


class ContinuationCertificateVerificationError(ValueError):
    """Raised before a caller-authored continuation claim gains authority."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ContinuationCertificateVerificationError(
            "continuation proof must be finite canonical ASCII JSON"
        ) from exc


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContinuationCertificateVerificationError(f"continuation proof repeats JSON key {key!r}")
        result[key] = value
    return result


def _decode_canonical_json(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise ContinuationCertificateVerificationError("proof payload must be exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContinuationCertificateVerificationError("proof payload is not strict ASCII JSON") from exc
    if _canonical_json(value) != payload:
        raise ContinuationCertificateVerificationError("proof payload changed on canonical parse and re-emit")
    return value


def _sha256_bytes(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise ContinuationCertificateVerificationError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _sha256_canonical(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _require_exact_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ContinuationCertificateVerificationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContinuationCertificateVerificationError(
            f"{label} fields differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return cast("dict[str, Any]", value)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ContinuationCertificateVerificationError(f"{label} must be canonical lowercase SHA-256")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ContinuationCertificateVerificationError(f"{label} must be a bounded identifier")
    return value


def _require_axis(value: object) -> str:
    if type(value) is not str or not value or len(value) > 256 or "\x00" in value:
        raise ContinuationCertificateVerificationError("axis must be a nonempty bounded string without NUL")
    return value


def _require_finite_nonnegative(value: object, label: str) -> float:
    if type(value) is not float:
        raise ContinuationCertificateVerificationError(f"{label} must be a canonical binary64 JSON number")
    result = value
    if not math.isfinite(result) or result < 0.0:
        raise ContinuationCertificateVerificationError(f"{label} left its finite nonnegative domain")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise ContinuationCertificateVerificationError(f"{label} may not encode negative zero")
    return result


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1 or value > _MAX_PROOF_INTEGER:
        raise ContinuationCertificateVerificationError(f"{label} must be a positive signed-64-bit int")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_PROOF_INTEGER:
        raise ContinuationCertificateVerificationError(f"{label} must be a nonnegative signed-64-bit int")
    return value


def canonical_score_implementation_sha256() -> str:
    """Identity of the pinned evaluator's exact binary64 operation DAG."""

    return _sha256_canonical(
        {
            "addition_order": "((100*d_seg)+sqrt(d_pose*10))+(25*rate)",
            "formula": "rate=archive_nbytes/N;score=100*d_seg+sqrt(d_pose*10)+25*rate",
            "numeric_semantics": "CPython IEEE754 binary64, one operation per Python AST node",
            "pose_weight": POSE_WEIGHT,
            "rate_weight": RATE_WEIGHT,
            "reference_uncompressed_nbytes": UNCOMPRESSED_SIZE_BYTES,
            "schema": "tac.upstream_evaluate_binary64_operation_dag.v1",
            "seg_weight": SEG_WEIGHT,
            "upstream_evaluate_sha256": PINNED_UPSTREAM_EVALUATE_SHA256,
            "upstream_source_line": "score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate",
        }
    )


CANONICAL_SCORE_IMPLEMENTATION_SHA256: Final = canonical_score_implementation_sha256()


def upstream_binary64_contest_score(d_seg: float, d_pose: float, archive_nbytes: int) -> float:
    """Reproduce pinned ``upstream/evaluate.py`` operation order exactly."""

    seg = _require_finite_nonnegative(d_seg, "score d_seg")
    pose = _require_finite_nonnegative(d_pose, "score d_pose")
    if seg > 1.0:
        raise ContinuationCertificateVerificationError("score d_seg exceeds disagreement-rate domain")
    size = _require_positive_int(archive_nbytes, "score archive bytes")
    rate = size / UNCOMPRESSED_SIZE_BYTES
    return 100 * seg + math.sqrt(pose * 10) + 25 * rate


def _require_authority_axis(authority_mode: object, axis: object) -> tuple[AuthorityMode, str]:
    """Cross-check a DECLARED authority mode against its DECLARED axis string.

    Catalog #127 direction note (adjudicated 2026-08-01, task #852). This
    predicate can only REFUSE; it never grants authority. It answers exactly one
    question — "is the declared mode consistent with the declared axis?" — and
    raises on inconsistency, so a disagreement between this code and the gate
    errs toward refusing a legitimate certificate, never toward admitting an
    illegitimate one. The advisory branch below is affirmatively anti-laundering:
    a research row carrying a contest tag anywhere in its axis is refused.

    RESIDUAL CONDITION, deliberately recorded at the site rather than in a memo:
    this leg does NOT validate hardware_substrate, and this dataclass carries no
    substrate or archive_sha256 field to validate against — so a
    ``validate_custody`` call here would have nothing to validate. What makes the
    PRODUCTION_* branches unreachable today is the hard research-only admission
    gate in ``_verify_receipt_common`` ("G38 admission is research-only until a
    governed production receipt adapter exists"). WHOEVER LANDS THAT ADAPTER must
    add the substrate leg here (or route the receipt through
    ``validate_exact_eval_evidence``) in the same landing, because at that moment
    a PRODUCTION_CONTEST_CPU certificate declaring ``[contest-CPU]`` from a macOS
    host would pass this function — which is precisely the codex round-2 HIGH 2
    bug class Catalog #127 exists to prevent.
    """
    if type(authority_mode) is not str or authority_mode not in _AUTHORITY_MODES:
        raise ContinuationCertificateVerificationError("authority mode is unsupported")
    normalized_axis = _require_axis(axis)
    if authority_mode == "PRODUCTION_CONTEST_CPU":
        valid = normalized_axis.startswith("[contest-CPU]")  # CUSTODY_VALIDATOR_OK:refusal-only axis/mode consistency leg; substrate leg owed by the future production receipt adapter (see docstring), production admission hard-gated research-only in _verify_receipt_common
    elif authority_mode == "PRODUCTION_CONTEST_CUDA":
        valid = normalized_axis.startswith("[contest-CUDA]")  # CUSTODY_VALIDATOR_OK:refusal-only axis/mode consistency leg; substrate leg owed by the future production receipt adapter (see docstring), production admission hard-gated research-only in _verify_receipt_common
    else:
        valid = (
            (
                normalized_axis.startswith("[macOS-CPU advisory]")
                or normalized_axis.startswith("[macOS-MLX research-signal]")
                or normalized_axis.startswith("[research-only")
            )
            and "[contest-CPU]" not in normalized_axis
            and "[contest-CUDA]" not in normalized_axis
        )
    if not valid:
        raise ContinuationCertificateVerificationError("authority mode and hardware axis are inconsistent")
    return cast("AuthorityMode", authority_mode), normalized_axis


def terminal_leaf_set_root_sha256(terminal_leaf_ids: Sequence[str]) -> str:
    """Merkle-root one exact, sorted set of generator terminal-leaf IDs.

    IDs are sorted before hashing, duplicates are forbidden, and cardinality is
    mixed into the final root.  The helper is public so the generator manifest
    and the verifier use exactly the same set-partition identity.
    """

    if isinstance(terminal_leaf_ids, (str, bytes)) or not isinstance(terminal_leaf_ids, Sequence):
        raise ContinuationCertificateVerificationError("terminal leaf IDs must be a finite sequence")
    raw = tuple(terminal_leaf_ids)
    if not raw:
        raise ContinuationCertificateVerificationError("terminal leaf IDs must be a nonempty unique set")
    for leaf_id in raw:
        _require_identifier(leaf_id, "terminal leaf ID")
    ordered = tuple(sorted(raw))
    if len(set(ordered)) != len(ordered):
        raise ContinuationCertificateVerificationError("terminal leaf IDs must be a nonempty unique set")
    level = [
        hashlib.sha256(b"TAC-CONTINUATION-TERMINAL-LEAF-V1\0" + leaf_id.encode("ascii")).digest() for leaf_id in ordered
    ]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"TAC-CONTINUATION-TERMINAL-NODE-V1\0" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return hashlib.sha256(
        b"TAC-CONTINUATION-TERMINAL-ROOT-V1\0" + len(ordered).to_bytes(8, "big") + level[0]
    ).hexdigest()


def base_stop_terminal_leaf_id(*, exact_base_identity_sha256: str, epoch_id: str) -> str:
    """Derive the controller-owned terminal-leaf ID for the exact base."""

    _require_sha256(exact_base_identity_sha256, "base-stop exact base identity")
    _require_identifier(epoch_id, "base-stop epoch")
    return "base_stop." + _sha256_canonical(
        {
            "epoch_id": epoch_id,
            "exact_base_identity_sha256": exact_base_identity_sha256,
            "schema": "tac.base_stop_terminal_leaf.v1",
        }
    )


def base_stop_support_hyperedge_sha256(*, exact_base_identity_sha256: str, epoch_id: str) -> str:
    """Derive the empty-support identity for the controller-owned stop branch."""

    _require_sha256(exact_base_identity_sha256, "base-stop exact base identity")
    _require_identifier(epoch_id, "base-stop epoch")
    return _sha256_canonical(
        {
            "epoch_id": epoch_id,
            "exact_base_identity_sha256": exact_base_identity_sha256,
            "members": [],
            "schema": "tac.base_stop_support_hyperedge.v1",
        }
    )


@dataclass(frozen=True, slots=True)
class ContinuationProofBindingV1:
    """Foreign-key bindings supplied both by proof bytes and controller context."""

    exact_base_identity_sha256: str
    epoch_id: str
    generator_domain_manifest_sha256: str
    first_action_id: str
    support_hyperedge_sha256: str
    continuation_equivalence_sha256: str
    horizon: int
    authority_mode: AuthorityMode
    axis: str
    hard_constraint_contract_sha256: str
    score_implementation_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.exact_base_identity_sha256, "binding exact base identity")
        _require_identifier(self.epoch_id, "binding epoch")
        _require_sha256(self.generator_domain_manifest_sha256, "binding generator-domain manifest")
        _require_identifier(self.first_action_id, "binding first action")
        _require_sha256(self.support_hyperedge_sha256, "binding support hyperedge")
        _require_sha256(self.continuation_equivalence_sha256, "binding continuation equivalence")
        _require_positive_int(self.horizon, "binding horizon")
        _require_authority_axis(self.authority_mode, self.axis)
        _require_sha256(self.hard_constraint_contract_sha256, "binding hard-constraint contract")
        _require_sha256(self.score_implementation_sha256, "binding score implementation")
        if self.score_implementation_sha256 != CANONICAL_SCORE_IMPLEMENTATION_SHA256:
            raise ContinuationCertificateVerificationError(
                "binding score implementation is not the canonical contest-score arithmetic"
            )


@dataclass(frozen=True, slots=True)
class ContinuationVerificationContextV1:
    """Independently derived controller expectations for one proof envelope."""

    binding: ContinuationProofBindingV1
    expected_subtree_root_sha256: str
    expected_subtree_cardinality: int
    expected_covered_terminal_leaf_ids: tuple[str, ...]
    expected_dependency_sha256s: tuple[str, ...]
    expected_proof_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.binding) is not ContinuationProofBindingV1:
            raise ContinuationCertificateVerificationError("verification context requires a typed proof binding")
        _require_sha256(self.expected_subtree_root_sha256, "expected subtree root")
        _require_positive_int(self.expected_subtree_cardinality, "expected subtree cardinality")
        if type(self.expected_covered_terminal_leaf_ids) is not tuple or not self.expected_covered_terminal_leaf_ids:
            raise ContinuationCertificateVerificationError(
                "verification context requires exact covered terminal leaf IDs"
            )
        for leaf_id in self.expected_covered_terminal_leaf_ids:
            _require_identifier(leaf_id, "expected covered terminal leaf ID")
        if self.expected_covered_terminal_leaf_ids != tuple(sorted(set(self.expected_covered_terminal_leaf_ids))):
            raise ContinuationCertificateVerificationError(
                "expected covered terminal leaf IDs must be sorted and unique"
            )
        if (
            len(self.expected_covered_terminal_leaf_ids) != self.expected_subtree_cardinality
            or terminal_leaf_set_root_sha256(self.expected_covered_terminal_leaf_ids)
            != self.expected_subtree_root_sha256
        ):
            raise ContinuationCertificateVerificationError(
                "expected leaf partition does not recompute expected subtree root/cardinality"
            )
        if type(self.expected_dependency_sha256s) is not tuple or not self.expected_dependency_sha256s:
            raise ContinuationCertificateVerificationError("verification context requires exact proof dependencies")
        for dependency in self.expected_dependency_sha256s:
            _require_sha256(dependency, "expected proof dependency")
        if self.expected_dependency_sha256s != tuple(sorted(set(self.expected_dependency_sha256s))):
            raise ContinuationCertificateVerificationError("expected proof dependencies must be sorted and unique")
        if self.expected_proof_sha256 is not None:
            _require_sha256(self.expected_proof_sha256, "expected proof identity")


@dataclass(frozen=True, slots=True)
class FiniteTerminalLeafEvidenceV1:
    """Exact score components and receipts for one enumerated terminal leaf."""

    terminal_leaf_id: str
    terminal_endpoint_identity_sha256: str
    action_path: tuple[str, ...]
    d_seg: float
    d_pose: float
    archive_nbytes: int
    measurement_receipt_sha256: str
    hard_constraint_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.terminal_leaf_id, "finite terminal leaf ID")
        _require_sha256(self.terminal_endpoint_identity_sha256, "finite terminal endpoint identity")
        if type(self.action_path) is not tuple or not self.action_path:
            raise ContinuationCertificateVerificationError("finite terminal action path must be a nonempty tuple")
        for action_id in self.action_path:
            _require_identifier(action_id, "finite terminal action-path member")
        d_seg = _require_finite_nonnegative(self.d_seg, "finite terminal d_seg")
        if d_seg > 1.0:
            raise ContinuationCertificateVerificationError("finite terminal d_seg exceeds disagreement-rate domain")
        _require_finite_nonnegative(self.d_pose, "finite terminal d_pose")
        _require_positive_int(self.archive_nbytes, "finite terminal archive bytes")
        _require_sha256(self.measurement_receipt_sha256, "finite measurement receipt")
        _require_sha256(self.hard_constraint_receipt_sha256, "finite hard-constraint receipt")

    @property
    def score(self) -> float:
        return upstream_binary64_contest_score(self.d_seg, self.d_pose, self.archive_nbytes)


@dataclass(frozen=True, slots=True)
class ConservativePrimitiveIntervalLeafEvidenceV1:
    """Per-leaf primitive lower/upper intervals from a separate derivation receipt."""

    terminal_leaf_id: str
    d_seg_lower: float
    d_seg_upper: float
    d_pose_lower: float
    d_pose_upper: float
    archive_nbytes_lower: int
    archive_nbytes_upper: int
    interval_derivation_receipt_sha256: str
    leaf_coverage_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.terminal_leaf_id, "primitive terminal leaf ID")
        d_seg_lower = _require_finite_nonnegative(self.d_seg_lower, "primitive d_seg lower")
        d_seg_upper = _require_finite_nonnegative(self.d_seg_upper, "primitive d_seg upper")
        if d_seg_lower > d_seg_upper or d_seg_upper > 1.0:
            raise ContinuationCertificateVerificationError("primitive d_seg interval is invalid")
        d_pose_lower = _require_finite_nonnegative(self.d_pose_lower, "primitive d_pose lower")
        d_pose_upper = _require_finite_nonnegative(self.d_pose_upper, "primitive d_pose upper")
        if d_pose_lower > d_pose_upper:
            raise ContinuationCertificateVerificationError("primitive d_pose interval is invalid")
        archive_lower = _require_nonnegative_int(self.archive_nbytes_lower, "primitive archive-byte lower")
        archive_upper = _require_nonnegative_int(self.archive_nbytes_upper, "primitive archive-byte upper")
        if archive_lower > archive_upper:
            raise ContinuationCertificateVerificationError("primitive archive-byte interval is invalid")
        _require_sha256(
            self.interval_derivation_receipt_sha256,
            "primitive interval derivation receipt",
        )
        _require_sha256(self.leaf_coverage_receipt_sha256, "primitive leaf-coverage receipt")

    @property
    def conservative_score_lower_bound(self) -> float:
        raise ContinuationCertificateVerificationError(
            "primitive interval admission is disabled until a typed interval theorem is independently verified"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class VerifiedContinuationCertificateV1:
    """Verifier result; consumers must reverify bytes or assert its registry fingerprint."""

    exact_base_identity_sha256: str
    epoch_id: str
    generator_domain_manifest_sha256: str
    subtree_root_sha256: str
    subtree_cardinality: int
    first_action_id: str
    support_hyperedge_sha256: str
    continuation_equivalence_sha256: str
    horizon: int
    authority_mode: AuthorityMode
    axis: str
    hard_constraint_contract_sha256: str
    score_implementation_sha256: str
    terminal_lower_bound: float
    feasible_terminal_upper_bound: float | None
    feasible_terminal_action_path: tuple[str, ...]
    feasible_terminal_endpoint_identity_sha256: str | None
    covered_terminal_leaf_ids: tuple[str, ...]
    covered_terminal_leaf_root_sha256: str
    proof_dependency_sha256s: tuple[str, ...]
    derivation_method: DerivationMethod
    outward_rounding_policy: str
    proof_sha256: str
    verifier_status: VerifierStatus
    research_only: bool

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ContinuationCertificateVerificationError(
            "VerifiedContinuationCertificateV1 has no public constructor; verify proof bytes"
        )


_REGISTERED_CERTIFICATE_FINGERPRINTS = weakref.WeakKeyDictionary()


def _certificate_fingerprint(certificate: VerifiedContinuationCertificateV1) -> str:
    if type(certificate) is not VerifiedContinuationCertificateV1:
        raise ContinuationCertificateVerificationError("continuation certificate has the wrong runtime type")
    try:
        return _sha256_canonical(asdict(certificate))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContinuationCertificateVerificationError("continuation certificate fields are malformed") from exc


def assert_registered_verified_certificate(certificate: VerifiedContinuationCertificateV1) -> None:
    """Reject forged or post-verification-mutated in-process certificate objects."""

    fingerprint = _certificate_fingerprint(certificate)
    if _REGISTERED_CERTIFICATE_FINGERPRINTS.get(certificate) != fingerprint:
        raise ContinuationCertificateVerificationError(
            "certificate is forged, unregistered, or mutated; reverify its proof and dependency bytes"
        )


def _binding_payload(binding: ContinuationProofBindingV1) -> dict[str, object]:
    return asdict(binding)


def _finite_leaf_payload(row: FiniteTerminalLeafEvidenceV1) -> dict[str, object]:
    return {
        "action_path": list(row.action_path),
        "archive_nbytes": row.archive_nbytes,
        "d_pose": row.d_pose,
        "d_seg": row.d_seg,
        "hard_constraint_receipt_sha256": row.hard_constraint_receipt_sha256,
        "measurement_receipt_sha256": row.measurement_receipt_sha256,
        "terminal_endpoint_identity_sha256": row.terminal_endpoint_identity_sha256,
        "terminal_leaf_id": row.terminal_leaf_id,
    }


def _primitive_leaf_payload(
    row: ConservativePrimitiveIntervalLeafEvidenceV1,
) -> dict[str, object]:
    return asdict(row)


def _proof_bytes(
    *,
    binding: ContinuationProofBindingV1,
    derivation_method: DerivationMethod,
    outward_rounding_policy: str,
    finite_leaves: tuple[FiniteTerminalLeafEvidenceV1, ...],
    primitive_leaves: tuple[ConservativePrimitiveIntervalLeafEvidenceV1, ...],
) -> bytes:
    if type(derivation_method) is not str or derivation_method not in _DERIVATION_METHODS:
        raise ContinuationCertificateVerificationError("proof derivation method is unsupported")
    if derivation_method in (
        "FINITE_TERMINAL_LEAF_ENUMERATION",
        "BASE_STOP_EXACT_BASE",
    ):
        if not finite_leaves or primitive_leaves:
            raise ContinuationCertificateVerificationError(
                "finite/base-stop proofs require only finite terminal evidence"
            )
        leaf_ids = tuple(sorted(row.terminal_leaf_id for row in finite_leaves))
        dependencies = tuple(
            sorted(
                {
                    dependency
                    for row in finite_leaves
                    for dependency in (
                        row.measurement_receipt_sha256,
                        row.hard_constraint_receipt_sha256,
                    )
                }
            )
        )
    else:
        if not primitive_leaves or finite_leaves:
            raise ContinuationCertificateVerificationError("primitive proofs require only per-leaf primitive intervals")
        leaf_ids = tuple(sorted(row.terminal_leaf_id for row in primitive_leaves))
        dependencies = tuple(
            sorted(
                {
                    dependency
                    for row in primitive_leaves
                    for dependency in (
                        row.interval_derivation_receipt_sha256,
                        row.leaf_coverage_receipt_sha256,
                    )
                }
            )
        )
    root = terminal_leaf_set_root_sha256(leaf_ids)
    body = {
        **_binding_payload(binding),
        "covered_terminal_leaf_ids": list(leaf_ids),
        "derivation_method": derivation_method,
        "finite_terminal_leaves": [
            _finite_leaf_payload(row) for row in sorted(finite_leaves, key=lambda item: item.terminal_leaf_id)
        ],
        "outward_rounding_policy": outward_rounding_policy,
        "primitive_interval_leaves": [
            _primitive_leaf_payload(row) for row in sorted(primitive_leaves, key=lambda item: item.terminal_leaf_id)
        ],
        "proof_dependency_sha256s": list(dependencies),
        "schema": PROOF_BODY_SCHEMA,
        "subtree_cardinality": len(leaf_ids),
        "subtree_root_sha256": root,
    }
    body_sha256 = _sha256_canonical(body)
    return _canonical_json(
        {
            "proof_body": body,
            "proof_body_sha256": body_sha256,
            "schema": PROOF_ENVELOPE_SCHEMA,
        }
    )


def build_finite_terminal_leaf_proof_bytes(
    *,
    binding: ContinuationProofBindingV1,
    leaves: Sequence[FiniteTerminalLeafEvidenceV1],
) -> bytes:
    """Build canonical proof bytes; this does not construct a certificate."""

    if type(binding) is not ContinuationProofBindingV1:
        raise ContinuationCertificateVerificationError("finite proof requires typed binding")
    if isinstance(leaves, (str, bytes)) or not isinstance(leaves, Sequence):
        raise ContinuationCertificateVerificationError("finite leaves must be a sequence")
    typed = tuple(leaves)
    if any(type(row) is not FiniteTerminalLeafEvidenceV1 for row in typed):
        raise ContinuationCertificateVerificationError("finite proof contains an untyped terminal leaf")
    return _proof_bytes(
        binding=binding,
        derivation_method="FINITE_TERMINAL_LEAF_ENUMERATION",
        outward_rounding_policy=FINITE_ROUNDING_POLICY,
        finite_leaves=typed,
        primitive_leaves=(),
    )


def build_conservative_primitive_interval_proof_bytes(
    *,
    binding: ContinuationProofBindingV1,
    leaves: Sequence[ConservativePrimitiveIntervalLeafEvidenceV1],
) -> bytes:
    """Build exhaustive per-leaf interval proof bytes without claiming verification."""

    if type(binding) is not ContinuationProofBindingV1:
        raise ContinuationCertificateVerificationError("primitive proof requires typed binding")
    if isinstance(leaves, (str, bytes)) or not isinstance(leaves, Sequence):
        raise ContinuationCertificateVerificationError("primitive leaves must be a sequence")
    typed = tuple(leaves)
    if any(type(row) is not ConservativePrimitiveIntervalLeafEvidenceV1 for row in typed):
        raise ContinuationCertificateVerificationError("primitive proof contains an untyped interval leaf")
    return _proof_bytes(
        binding=binding,
        derivation_method="PER_LEAF_CONSERVATIVE_PRIMITIVE_INTERVALS",
        outward_rounding_policy=PRIMITIVE_ROUNDING_POLICY,
        finite_leaves=(),
        primitive_leaves=typed,
    )


def build_base_stop_proof_bytes(
    *,
    binding: ContinuationProofBindingV1,
    base_endpoint_identity_sha256: str,
    d_seg: float,
    d_pose: float,
    archive_nbytes: int,
    measurement_receipt_sha256: str,
    hard_constraint_receipt_sha256: str,
) -> bytes:
    """Build proof bytes for the exact feasible base as a nonoptional stop branch."""

    if type(binding) is not ContinuationProofBindingV1:
        raise ContinuationCertificateVerificationError("base-stop proof requires typed binding")
    expected_support = base_stop_support_hyperedge_sha256(
        exact_base_identity_sha256=binding.exact_base_identity_sha256,
        epoch_id=binding.epoch_id,
    )
    if binding.first_action_id != BASE_STOP_ACTION_ID or binding.support_hyperedge_sha256 != expected_support:
        raise ContinuationCertificateVerificationError(
            "base-stop binding must use the controller-owned action and empty support"
        )
    leaf = FiniteTerminalLeafEvidenceV1(
        terminal_leaf_id=base_stop_terminal_leaf_id(
            exact_base_identity_sha256=binding.exact_base_identity_sha256,
            epoch_id=binding.epoch_id,
        ),
        terminal_endpoint_identity_sha256=base_endpoint_identity_sha256,
        action_path=(BASE_STOP_ACTION_ID,),
        d_seg=d_seg,
        d_pose=d_pose,
        archive_nbytes=archive_nbytes,
        measurement_receipt_sha256=measurement_receipt_sha256,
        hard_constraint_receipt_sha256=hard_constraint_receipt_sha256,
    )
    return _proof_bytes(
        binding=binding,
        derivation_method="BASE_STOP_EXACT_BASE",
        outward_rounding_policy=BASE_STOP_ROUNDING_POLICY,
        finite_leaves=(leaf,),
        primitive_leaves=(),
    )


def _parse_finite_leaf(value: object) -> FiniteTerminalLeafEvidenceV1:
    item = _require_exact_fields(value, _FINITE_LEAF_FIELDS, "finite terminal leaf")
    action_path = item["action_path"]
    if type(action_path) is not list:
        raise ContinuationCertificateVerificationError("finite terminal action path must serialize as a list")
    return FiniteTerminalLeafEvidenceV1(
        terminal_leaf_id=item["terminal_leaf_id"],
        terminal_endpoint_identity_sha256=item["terminal_endpoint_identity_sha256"],
        action_path=tuple(action_path),
        d_seg=item["d_seg"],
        d_pose=item["d_pose"],
        archive_nbytes=item["archive_nbytes"],
        measurement_receipt_sha256=item["measurement_receipt_sha256"],
        hard_constraint_receipt_sha256=item["hard_constraint_receipt_sha256"],
    )


def _parse_primitive_leaf(
    value: object,
) -> ConservativePrimitiveIntervalLeafEvidenceV1:
    item = _require_exact_fields(value, _PRIMITIVE_LEAF_FIELDS, "primitive interval leaf")
    return ConservativePrimitiveIntervalLeafEvidenceV1(**item)


def _parse_binding(body: Mapping[str, object]) -> ContinuationProofBindingV1:
    authority_mode = body["authority_mode"]
    if type(authority_mode) is not str or authority_mode not in _AUTHORITY_MODES:
        raise ContinuationCertificateVerificationError("proof authority mode is unsupported")
    return ContinuationProofBindingV1(
        exact_base_identity_sha256=_require_sha256(body["exact_base_identity_sha256"], "proof exact base identity"),
        epoch_id=_require_identifier(body["epoch_id"], "proof epoch"),
        generator_domain_manifest_sha256=_require_sha256(
            body["generator_domain_manifest_sha256"], "proof generator-domain manifest"
        ),
        first_action_id=_require_identifier(body["first_action_id"], "proof first action"),
        support_hyperedge_sha256=_require_sha256(body["support_hyperedge_sha256"], "proof support hyperedge"),
        continuation_equivalence_sha256=_require_sha256(
            body["continuation_equivalence_sha256"], "proof continuation equivalence"
        ),
        horizon=_require_positive_int(body["horizon"], "proof horizon"),
        authority_mode=cast("AuthorityMode", authority_mode),
        axis=_require_axis(body["axis"]),
        hard_constraint_contract_sha256=_require_sha256(
            body["hard_constraint_contract_sha256"], "proof hard-constraint contract"
        ),
        score_implementation_sha256=_require_sha256(body["score_implementation_sha256"], "proof score implementation"),
    )


@dataclass(frozen=True, slots=True)
class _ManifestLeafBindingV1:
    terminal_leaf_id: str
    first_action_id: str
    action_path: tuple[str, ...]
    support_hyperedge_sha256: str


def _parse_generator_domain_manifest(
    payload: bytes,
    *,
    binding: ContinuationProofBindingV1,
) -> dict[str, _ManifestLeafBindingV1]:
    manifest = _require_exact_fields(
        _decode_canonical_json(payload),
        _GENERATOR_MANIFEST_FIELDS,
        "generator-domain manifest",
    )
    if manifest["schema"] != GENERATOR_DOMAIN_MANIFEST_SCHEMA:
        raise ContinuationCertificateVerificationError("generator-domain manifest schema drifted")
    if _sha256_bytes(payload) != binding.generator_domain_manifest_sha256:
        raise ContinuationCertificateVerificationError("generator-domain manifest bytes differ from bound identity")
    if (
        manifest["exact_base_identity_sha256"] != binding.exact_base_identity_sha256
        or manifest["epoch_id"] != binding.epoch_id
        or manifest["horizon"] != binding.horizon
    ):
        raise ContinuationCertificateVerificationError("generator-domain manifest left bound base/epoch/horizon")
    _require_identifier(manifest["manifest_id"], "generator manifest ID")
    _require_identifier(manifest["epoch_id"], "generator manifest epoch")
    for field_name in (
        "exact_base_identity_sha256",
        "generator_program_sha256",
        "enumeration_receipt_sha256",
        "parameter_domain_manifest_sha256",
        "chronology_domain_manifest_sha256",
        "generated_leaf_merkle_root_sha256",
    ):
        _require_sha256(manifest[field_name], f"generator manifest {field_name}")
    _require_nonnegative_int(manifest["deterministic_seed"], "generator deterministic seed")
    maximum_interaction_order = _require_positive_int(
        manifest["maximum_interaction_order"], "generator maximum interaction order"
    )
    _require_positive_int(manifest["horizon"], "generator horizon")
    if manifest["families"] != list(_ACTION_FAMILIES) or manifest["scales"] != list(_ACTION_SCALES):
        raise ContinuationCertificateVerificationError(
            "generator-domain manifest lost the full family/scale vocabulary"
        )
    serialized_leaves = manifest["terminal_leaves"]
    if type(serialized_leaves) is not list:
        raise ContinuationCertificateVerificationError("generator terminal leaves must serialize as a list")
    parsed: dict[str, _ManifestLeafBindingV1] = {}
    leaf_identity_rows: list[dict[str, str]] = []
    previous_id: str | None = None
    for serialized_leaf in serialized_leaves:
        leaf = _require_exact_fields(serialized_leaf, _GENERATOR_LEAF_FIELDS, "generated terminal leaf")
        terminal_leaf_id = _require_identifier(leaf["terminal_leaf_id"], "generated terminal leaf ID")
        first_action_id = _require_identifier(leaf["first_action_id"], "generated first action")
        if terminal_leaf_id == BASE_STOP_ACTION_ID or first_action_id == BASE_STOP_ACTION_ID:
            raise ContinuationCertificateVerificationError(
                "generator manifest may not claim controller-owned BASE_STOP"
            )
        action_path_value = leaf["action_path"]
        parents_value = leaf["parent_action_ids"]
        support_value = leaf["support_hyperedge"]
        if type(action_path_value) is not list or not action_path_value:
            raise ContinuationCertificateVerificationError("generated action path must be a nonempty list")
        action_path = tuple(_require_identifier(item, "generated action path member") for item in action_path_value)
        if action_path[0] != first_action_id or len(action_path) > binding.horizon:
            raise ContinuationCertificateVerificationError("generated leaf left its first action or horizon")
        if type(parents_value) is not list or tuple(parents_value) != action_path[:-1]:
            raise ContinuationCertificateVerificationError("generated leaf parent path drifted")
        if type(support_value) is not list or not support_value:
            raise ContinuationCertificateVerificationError("generated support hyperedge must be nonempty")
        support_keys: list[str] = []
        support_order: list[tuple[str, str]] = []
        for serialized_cell in support_value:
            cell = _require_exact_fields(serialized_cell, _SUPPORT_CELL_FIELDS, "generated support cell")
            family = cell["family"]
            scale = cell["scale"]
            if (
                type(family) is not str
                or type(scale) is not str
                or family not in _ACTION_FAMILIES
                or scale not in _ACTION_SCALES
            ):
                raise ContinuationCertificateVerificationError("generated support cell left the fixed vocabulary")
            support_order.append((family, scale))
            support_keys.append(f"{family}:{scale}")
        if support_order != sorted(set(support_order)) or len(support_order) > maximum_interaction_order:
            raise ContinuationCertificateVerificationError("generated support hyperedge is noncanonical or too large")
        support_sha256 = _sha256_canonical({"schema": "tac.action_support_hyperedge.v1", "members": support_keys})
        for field_name in ("parameter_identity_sha256", "chronology_context_sha256"):
            _require_sha256(leaf[field_name], f"generated leaf {field_name}")
        if (
            leaf["exact_base_identity_sha256"] != binding.exact_base_identity_sha256
            or leaf["universe_epoch_id"] != binding.epoch_id
        ):
            raise ContinuationCertificateVerificationError("generated leaf left manifest base or epoch")
        if previous_id is not None and terminal_leaf_id <= previous_id:
            raise ContinuationCertificateVerificationError("generator terminal leaves are not sorted and unique")
        previous_id = terminal_leaf_id
        parsed[terminal_leaf_id] = _ManifestLeafBindingV1(
            terminal_leaf_id=terminal_leaf_id,
            first_action_id=first_action_id,
            action_path=action_path,
            support_hyperedge_sha256=support_sha256,
        )
        leaf_identity_rows.append(
            {
                "terminal_leaf_id": terminal_leaf_id,
                "identity_sha256": _sha256_canonical({"schema": "tac.generated_terminal_leaf.v1", **leaf}),
            }
        )
    generated_cardinality = _require_nonnegative_int(
        manifest["generated_leaf_cardinality"], "generator leaf cardinality"
    )
    if generated_cardinality != len(parsed):
        raise ContinuationCertificateVerificationError("generator leaf cardinality is not derived from manifest leaves")
    expected_merkle = _sha256_canonical({"schema": "tac.generated_terminal_leaf_set.v1", "leaves": leaf_identity_rows})
    if manifest["generated_leaf_merkle_root_sha256"] != expected_merkle:
        raise ContinuationCertificateVerificationError("generator leaf descriptor Merkle root is invalid")
    expected_id_root = None if not parsed else terminal_leaf_set_root_sha256(tuple(parsed))
    if expected_id_root is not None:
        _require_sha256(manifest["generated_terminal_leaf_id_root_sha256"], "generator leaf ID root")
    if manifest["generated_terminal_leaf_id_root_sha256"] != expected_id_root:
        raise ContinuationCertificateVerificationError("generator terminal-leaf ID root is invalid")
    return parsed


def _require_binary64_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise ContinuationCertificateVerificationError(f"{label} must be canonical binary64 hex")
    try:
        parsed = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise ContinuationCertificateVerificationError(f"{label} is not binary64 hex") from exc
    if parsed.hex() != value:
        raise ContinuationCertificateVerificationError(f"{label} is not canonical binary64 hex")
    return _require_finite_nonnegative(parsed, label)


def _verify_receipt_common(
    receipt: Mapping[str, object],
    *,
    row: FiniteTerminalLeafEvidenceV1,
    binding: ContinuationProofBindingV1,
) -> None:
    action_path = receipt["action_path"]
    if type(action_path) is not list:
        raise ContinuationCertificateVerificationError("terminal receipt action path must serialize as a list")
    expected = (
        receipt["exact_base_identity_sha256"] == binding.exact_base_identity_sha256
        and receipt["epoch_id"] == binding.epoch_id
        and receipt["generator_domain_manifest_sha256"] == binding.generator_domain_manifest_sha256
        and receipt["terminal_leaf_id"] == row.terminal_leaf_id
        and receipt["terminal_endpoint_identity_sha256"] == row.terminal_endpoint_identity_sha256
        and tuple(action_path) == row.action_path
        and receipt["first_action_id"] == binding.first_action_id
        and receipt["support_hyperedge_sha256"] == binding.support_hyperedge_sha256
        and receipt["continuation_equivalence_sha256"] == binding.continuation_equivalence_sha256
        and receipt["horizon"] == binding.horizon
        and receipt["authority_mode"] == binding.authority_mode
        and receipt["axis"] == binding.axis
    )
    if not expected:
        raise ContinuationCertificateVerificationError("terminal receipt foreign keys differ from proof branch")
    for field_name in (
        "exact_base_identity_sha256",
        "generator_domain_manifest_sha256",
        "terminal_endpoint_identity_sha256",
        "support_hyperedge_sha256",
        "continuation_equivalence_sha256",
    ):
        _require_sha256(receipt[field_name], f"terminal receipt {field_name}")
    _require_identifier(receipt["epoch_id"], "terminal receipt epoch")
    _require_identifier(receipt["terminal_leaf_id"], "terminal receipt leaf")
    _require_identifier(receipt["first_action_id"], "terminal receipt first action")
    for action_id in action_path:
        _require_identifier(action_id, "terminal receipt action path member")
    _require_positive_int(receipt["horizon"], "terminal receipt horizon")
    _require_authority_axis(receipt["authority_mode"], receipt["axis"])
    if receipt["research_only"] is not True or binding.authority_mode != "RESEARCH_ADVISORY":
        raise ContinuationCertificateVerificationError(
            "G38 admission is research-only until a governed production receipt adapter exists"
        )


def _verify_finite_receipts(
    rows: tuple[FiniteTerminalLeafEvidenceV1, ...],
    *,
    binding: ContinuationProofBindingV1,
    dependency_payloads_by_sha256: Mapping[str, bytes],
    derivation_method: DerivationMethod,
) -> None:
    expected_measurement_role = (
        "G36_VERIFIED_WHOLE_OBJECT_BASE"
        if derivation_method == "BASE_STOP_EXACT_BASE"
        else "G36_VERIFIED_WHOLE_OBJECT_ACTION_ENDPOINT"
    )
    for row in rows:
        measurement = _require_exact_fields(
            _decode_canonical_json(dependency_payloads_by_sha256[row.measurement_receipt_sha256]),
            _FINITE_MEASUREMENT_RECEIPT_FIELDS,
            "finite terminal measurement receipt",
        )
        if (
            measurement["schema"] != FINITE_MEASUREMENT_RECEIPT_SCHEMA
            or measurement["receipt_role"] != expected_measurement_role
        ):
            raise ContinuationCertificateVerificationError("finite measurement receipt schema or role drifted")
        _verify_receipt_common(measurement, row=row, binding=binding)
        if (
            measurement["score_implementation_sha256"] != binding.score_implementation_sha256
            or measurement["upstream_evaluate_sha256"] != PINNED_UPSTREAM_EVALUATE_SHA256
        ):
            raise ContinuationCertificateVerificationError("finite measurement receipt score implementation drifted")
        d_seg = _require_binary64_hex(measurement["d_seg_binary64_hex"], "receipt d_seg")
        d_pose = _require_binary64_hex(measurement["d_pose_binary64_hex"], "receipt d_pose")
        if float(d_seg).hex() != float(row.d_seg).hex() or float(d_pose).hex() != float(row.d_pose).hex():
            raise ContinuationCertificateVerificationError("proof components differ from receipt-bound binary64 bits")
        if measurement["archive_nbytes"] != row.archive_nbytes:
            raise ContinuationCertificateVerificationError("proof archive bytes differ from measurement receipt")
        for field_name in (
            "score_implementation_sha256",
            "upstream_evaluate_sha256",
            "archive_sha256",
            "realized_output_sha256",
            "evaluator_measurement_receipt_sha256",
            "public_auth_eval_receipt_sha256",
        ):
            _require_sha256(measurement[field_name], f"finite measurement receipt {field_name}")
        _require_positive_int(measurement["archive_nbytes"], "finite measurement receipt archive bytes")

        constraints = _require_exact_fields(
            _decode_canonical_json(dependency_payloads_by_sha256[row.hard_constraint_receipt_sha256]),
            _HARD_CONSTRAINT_RECEIPT_FIELDS,
            "finite terminal hard-constraint receipt",
        )
        if (
            constraints["schema"] != HARD_CONSTRAINT_RECEIPT_SCHEMA
            or constraints["receipt_role"] != "G36_VERIFIED_DECODE_CONSTRAINTS"
        ):
            raise ContinuationCertificateVerificationError("hard-constraint receipt schema or role drifted")
        _verify_receipt_common(constraints, row=row, binding=binding)
        if constraints["hard_constraint_contract_sha256"] != binding.hard_constraint_contract_sha256:
            raise ContinuationCertificateVerificationError("hard-constraint receipt contract drifted")
        _require_sha256(constraints["hard_constraint_contract_sha256"], "hard-constraint receipt contract")
        _require_sha256(constraints["decode_constraint_receipt_sha256"], "decode constraint receipt")
        if constraints["hard_constraints_satisfied"] is not True or constraints["blocker_codes"] != []:
            raise ContinuationCertificateVerificationError("terminal endpoint has unresolved hard constraints")


def _verify_dependency_payloads(
    dependency_sha256s: tuple[str, ...],
    dependency_payloads_by_sha256: Mapping[str, bytes],
) -> None:
    if not isinstance(dependency_payloads_by_sha256, Mapping):
        raise ContinuationCertificateVerificationError("dependency payloads must be a SHA-to-bytes mapping")
    if set(dependency_payloads_by_sha256) != set(dependency_sha256s):
        raise ContinuationCertificateVerificationError(
            "reopened dependency payload set differs from proof dependencies"
        )
    for expected_sha256 in dependency_sha256s:
        _require_sha256(expected_sha256, "proof dependency")
        payload = dependency_payloads_by_sha256[expected_sha256]
        if type(payload) is not bytes or not payload:
            raise ContinuationCertificateVerificationError("proof dependency payload must be nonempty exact bytes")
        if _sha256_bytes(payload) != expected_sha256:
            raise ContinuationCertificateVerificationError("reopened dependency payload SHA-256 mismatch")


def verify_continuation_proof_bytes(
    proof_bytes: bytes,
    *,
    context: ContinuationVerificationContextV1,
    generator_domain_manifest_payload_bytes: bytes,
    dependency_payloads_by_sha256: Mapping[str, bytes],
) -> VerifiedContinuationCertificateV1:
    """Reopen manifest/proof/typed-receipt bytes and derive one research bound."""

    if type(context) is not ContinuationVerificationContextV1:
        raise ContinuationCertificateVerificationError("continuation verification requires a typed independent context")
    envelope = _require_exact_fields(_decode_canonical_json(proof_bytes), _ENVELOPE_FIELDS, "proof envelope")
    if envelope["schema"] != PROOF_ENVELOPE_SCHEMA:
        raise ContinuationCertificateVerificationError("proof envelope schema drifted")
    body = _require_exact_fields(envelope["proof_body"], _BODY_FIELDS, "proof body")
    if body["schema"] != PROOF_BODY_SCHEMA:
        raise ContinuationCertificateVerificationError("proof body schema drifted")
    expected_body_sha256 = _sha256_canonical(body)
    if envelope["proof_body_sha256"] != expected_body_sha256:
        raise ContinuationCertificateVerificationError("proof body changed after its self-hash was sealed")
    proof_sha256 = _sha256_bytes(proof_bytes)
    if context.expected_proof_sha256 is not None and proof_sha256 != context.expected_proof_sha256:
        raise ContinuationCertificateVerificationError(
            "proof bytes differ from the independently expected proof identity"
        )

    binding = _parse_binding(body)
    if binding != context.binding:
        raise ContinuationCertificateVerificationError("proof foreign keys differ from independent controller context")
    if binding.authority_mode != "RESEARCH_ADVISORY":
        raise ContinuationCertificateVerificationError(
            "production continuation admission is disabled until a governed production receipt adapter exists"
        )
    manifest_leaves = _parse_generator_domain_manifest(
        generator_domain_manifest_payload_bytes,
        binding=binding,
    )
    derivation_method = body["derivation_method"]
    if type(derivation_method) is not str or derivation_method not in _DERIVATION_METHODS:
        raise ContinuationCertificateVerificationError("proof derivation method is unsupported")
    cardinality = _require_positive_int(body["subtree_cardinality"], "proof subtree cardinality")
    subtree_root = _require_sha256(body["subtree_root_sha256"], "proof subtree root")
    if cardinality != context.expected_subtree_cardinality or subtree_root != context.expected_subtree_root_sha256:
        raise ContinuationCertificateVerificationError(
            "proof subtree root/cardinality differ from generator-domain manifest"
        )
    serialized_leaf_ids = body["covered_terminal_leaf_ids"]
    if type(serialized_leaf_ids) is not list:
        raise ContinuationCertificateVerificationError("covered terminal leaf IDs must serialize as a list")
    covered_leaf_ids = tuple(serialized_leaf_ids)
    for leaf_id in covered_leaf_ids:
        _require_identifier(leaf_id, "covered terminal leaf ID")
    if covered_leaf_ids != tuple(sorted(set(covered_leaf_ids))):
        raise ContinuationCertificateVerificationError("covered terminal leaf IDs must be sorted and unique")
    if len(covered_leaf_ids) != cardinality:
        raise ContinuationCertificateVerificationError("covered terminal leaf count differs from subtree cardinality")
    if covered_leaf_ids != context.expected_covered_terminal_leaf_ids:
        raise ContinuationCertificateVerificationError(
            "covered terminal leaf partition differs from independent controller context"
        )
    recomputed_root = terminal_leaf_set_root_sha256(covered_leaf_ids)
    if recomputed_root != subtree_root:
        raise ContinuationCertificateVerificationError(
            "covered terminal leaves do not recompute the manifest subtree root"
        )

    if derivation_method != "BASE_STOP_EXACT_BASE":
        if not set(covered_leaf_ids).issubset(manifest_leaves):
            raise ContinuationCertificateVerificationError("proof names a terminal leaf absent from reopened manifest")
        branch_leaves = tuple(manifest_leaves[leaf_id] for leaf_id in covered_leaf_ids)
        if any(
            leaf.first_action_id != binding.first_action_id
            or leaf.support_hyperedge_sha256 != binding.support_hyperedge_sha256
            for leaf in branch_leaves
        ):
            raise ContinuationCertificateVerificationError(
                "proof subtree crosses manifest first-action or support-hyperedge branches"
            )

    serialized_dependencies = body["proof_dependency_sha256s"]
    if type(serialized_dependencies) is not list:
        raise ContinuationCertificateVerificationError("proof dependencies must serialize as a list")
    dependencies = tuple(serialized_dependencies)
    for dependency in dependencies:
        _require_sha256(dependency, "proof dependency")
    if dependencies != tuple(sorted(set(dependencies))):
        raise ContinuationCertificateVerificationError("proof dependencies must be sorted and unique")
    if dependencies != context.expected_dependency_sha256s:
        raise ContinuationCertificateVerificationError("proof dependencies differ from independent controller context")
    _verify_dependency_payloads(dependencies, dependency_payloads_by_sha256)

    serialized_finite = body["finite_terminal_leaves"]
    serialized_primitive = body["primitive_interval_leaves"]
    if type(serialized_finite) is not list or type(serialized_primitive) is not list:
        raise ContinuationCertificateVerificationError("proof evidence collections must serialize as lists")
    finite_leaves = tuple(_parse_finite_leaf(row) for row in serialized_finite)
    primitive_leaves = tuple(_parse_primitive_leaf(row) for row in serialized_primitive)

    feasible_upper: float | None
    feasible_path: tuple[str, ...]
    feasible_endpoint: str | None
    if derivation_method in (
        "FINITE_TERMINAL_LEAF_ENUMERATION",
        "BASE_STOP_EXACT_BASE",
    ):
        expected_policy = (
            BASE_STOP_ROUNDING_POLICY if derivation_method == "BASE_STOP_EXACT_BASE" else FINITE_ROUNDING_POLICY
        )
        if not finite_leaves or primitive_leaves:
            raise ContinuationCertificateVerificationError("finite/base-stop derivation has the wrong evidence kind")
        finite_ids = tuple(row.terminal_leaf_id for row in finite_leaves)
        if finite_ids != covered_leaf_ids:
            raise ContinuationCertificateVerificationError("finite evidence rows differ from covered terminal leaves")
        if len({row.action_path for row in finite_leaves}) != len(finite_leaves):
            raise ContinuationCertificateVerificationError("finite proof repeats a terminal action path")
        for row in finite_leaves:
            if row.action_path[0] != binding.first_action_id or len(row.action_path) > binding.horizon:
                raise ContinuationCertificateVerificationError("finite terminal path left its first action or horizon")
            if derivation_method != "BASE_STOP_EXACT_BASE":
                manifest_leaf = manifest_leaves[row.terminal_leaf_id]
                if row.action_path != manifest_leaf.action_path:
                    raise ContinuationCertificateVerificationError(
                        "finite terminal action path differs from reopened generator manifest"
                    )
        referenced_dependencies = {
            dependency
            for row in finite_leaves
            for dependency in (
                row.measurement_receipt_sha256,
                row.hard_constraint_receipt_sha256,
            )
        }
        if referenced_dependencies != set(dependencies):
            raise ContinuationCertificateVerificationError(
                "finite evidence dependency set differs from reopened dependencies"
            )
        _verify_finite_receipts(
            finite_leaves,
            binding=binding,
            dependency_payloads_by_sha256=dependency_payloads_by_sha256,
            derivation_method=cast("DerivationMethod", derivation_method),
        )
        best = min(
            finite_leaves,
            key=lambda row: (
                row.score,
                row.terminal_endpoint_identity_sha256,
                row.action_path,
                row.terminal_leaf_id,
            ),
        )
        terminal_lower = best.score
        feasible_upper = best.score
        feasible_path = best.action_path
        feasible_endpoint = best.terminal_endpoint_identity_sha256
        if derivation_method == "BASE_STOP_EXACT_BASE":
            expected_leaf_id = base_stop_terminal_leaf_id(
                exact_base_identity_sha256=binding.exact_base_identity_sha256,
                epoch_id=binding.epoch_id,
            )
            if (
                binding.first_action_id != BASE_STOP_ACTION_ID
                or binding.support_hyperedge_sha256
                != base_stop_support_hyperedge_sha256(
                    exact_base_identity_sha256=binding.exact_base_identity_sha256,
                    epoch_id=binding.epoch_id,
                )
                or covered_leaf_ids != (expected_leaf_id,)
                or best.action_path != (BASE_STOP_ACTION_ID,)
            ):
                raise ContinuationCertificateVerificationError(
                    "base-stop proof is not the controller-owned exact-base singleton"
                )
            verifier_status: VerifierStatus = "VERIFIED_BASE_STOP_EXACT_SCORE"
        else:
            if binding.first_action_id == BASE_STOP_ACTION_ID:
                raise ContinuationCertificateVerificationError(
                    "ordinary finite proof may not impersonate controller-owned BASE_STOP"
                )
            verifier_status = "VERIFIED_FINITE_TERMINAL_ENUMERATION"
    else:
        expected_policy = PRIMITIVE_ROUNDING_POLICY
        if not primitive_leaves or finite_leaves:
            raise ContinuationCertificateVerificationError("primitive derivation has the wrong evidence kind")
        primitive_ids = tuple(row.terminal_leaf_id for row in primitive_leaves)
        if primitive_ids != covered_leaf_ids:
            raise ContinuationCertificateVerificationError(
                "primitive evidence rows differ from covered terminal leaves"
            )
        referenced_dependencies = {
            dependency
            for row in primitive_leaves
            for dependency in (
                row.interval_derivation_receipt_sha256,
                row.leaf_coverage_receipt_sha256,
            )
        }
        if referenced_dependencies != set(dependencies):
            raise ContinuationCertificateVerificationError(
                "primitive evidence dependency set differs from reopened dependencies"
            )
        raise ContinuationCertificateVerificationError(
            "primitive interval proof is structurally bound but unadmitted: typed theorem verification is unavailable"
        )

    outward_rounding_policy = body["outward_rounding_policy"]
    if outward_rounding_policy != expected_policy:
        raise ContinuationCertificateVerificationError(
            "proof outward-rounding policy differs from its derivation method"
        )
    if not math.isfinite(terminal_lower):
        raise ContinuationCertificateVerificationError("derived terminal lower bound is not finite")

    certificate_values: dict[str, object] = {
        "exact_base_identity_sha256": binding.exact_base_identity_sha256,
        "epoch_id": binding.epoch_id,
        "generator_domain_manifest_sha256": binding.generator_domain_manifest_sha256,
        "subtree_root_sha256": subtree_root,
        "subtree_cardinality": cardinality,
        "first_action_id": binding.first_action_id,
        "support_hyperedge_sha256": binding.support_hyperedge_sha256,
        "continuation_equivalence_sha256": binding.continuation_equivalence_sha256,
        "horizon": binding.horizon,
        "authority_mode": binding.authority_mode,
        "axis": binding.axis,
        "hard_constraint_contract_sha256": binding.hard_constraint_contract_sha256,
        "score_implementation_sha256": binding.score_implementation_sha256,
        "terminal_lower_bound": terminal_lower,
        "feasible_terminal_upper_bound": feasible_upper,
        "feasible_terminal_action_path": feasible_path,
        "feasible_terminal_endpoint_identity_sha256": feasible_endpoint,
        "covered_terminal_leaf_ids": covered_leaf_ids,
        "covered_terminal_leaf_root_sha256": recomputed_root,
        "proof_dependency_sha256s": dependencies,
        "derivation_method": derivation_method,
        "outward_rounding_policy": outward_rounding_policy,
        "proof_sha256": proof_sha256,
        "verifier_status": verifier_status,
        "research_only": True,
    }
    if set(certificate_values) != set(VerifiedContinuationCertificateV1.__dataclass_fields__):
        raise ContinuationCertificateVerificationError("internal certificate field set drifted")
    certificate = object.__new__(VerifiedContinuationCertificateV1)
    for name, value in certificate_values.items():
        object.__setattr__(certificate, name, value)
    _REGISTERED_CERTIFICATE_FINGERPRINTS[certificate] = _certificate_fingerprint(certificate)
    return certificate


def verify_base_stop_proof_bytes(
    proof_bytes: bytes,
    *,
    context: ContinuationVerificationContextV1,
    generator_domain_manifest_payload_bytes: bytes,
    dependency_payloads_by_sha256: Mapping[str, bytes],
) -> VerifiedContinuationCertificateV1:
    """Verify the nonoptional exact-base incumbent and refuse ordinary branches."""

    certificate = verify_continuation_proof_bytes(
        proof_bytes,
        context=context,
        generator_domain_manifest_payload_bytes=generator_domain_manifest_payload_bytes,
        dependency_payloads_by_sha256=dependency_payloads_by_sha256,
    )
    if certificate.verifier_status != "VERIFIED_BASE_STOP_EXACT_SCORE":
        raise ContinuationCertificateVerificationError("base-stop verifier received an ordinary continuation proof")
    return certificate


def reverify_continuation_certificate(
    certificate: VerifiedContinuationCertificateV1,
    proof_bytes: bytes,
    *,
    context: ContinuationVerificationContextV1,
    generator_domain_manifest_payload_bytes: bytes,
    dependency_payloads_by_sha256: Mapping[str, bytes],
) -> VerifiedContinuationCertificateV1:
    """Reverify all durable bytes at an authority-consuming boundary.

    Python objects are not cryptographic capabilities.  This function is the
    durable boundary: it reconstructs a fresh registered result and refuses if
    any field of the supplied object differs.
    """

    fresh = verify_continuation_proof_bytes(
        proof_bytes,
        context=context,
        generator_domain_manifest_payload_bytes=generator_domain_manifest_payload_bytes,
        dependency_payloads_by_sha256=dependency_payloads_by_sha256,
    )
    try:
        same = asdict(certificate) == asdict(fresh)
    except (AttributeError, TypeError) as exc:
        raise ContinuationCertificateVerificationError("supplied certificate object is malformed") from exc
    if not same:
        raise ContinuationCertificateVerificationError(
            "supplied certificate differs from reverified manifest/proof/receipt bytes"
        )
    return fresh


__all__ = [
    "BASE_STOP_ACTION_ID",
    "BASE_STOP_ROUNDING_POLICY",
    "CANONICAL_SCORE_IMPLEMENTATION_SHA256",
    "FINITE_MEASUREMENT_RECEIPT_SCHEMA",
    "FINITE_ROUNDING_POLICY",
    "GENERATOR_DOMAIN_MANIFEST_SCHEMA",
    "HARD_CONSTRAINT_RECEIPT_SCHEMA",
    "PINNED_UPSTREAM_EVALUATE_SHA256",
    "PRIMITIVE_ROUNDING_POLICY",
    "PROOF_BODY_SCHEMA",
    "PROOF_ENVELOPE_SCHEMA",
    "SCHEMA",
    "ConservativePrimitiveIntervalLeafEvidenceV1",
    "ContinuationCertificateVerificationError",
    "ContinuationProofBindingV1",
    "ContinuationVerificationContextV1",
    "FiniteTerminalLeafEvidenceV1",
    "VerifiedContinuationCertificateV1",
    "assert_registered_verified_certificate",
    "base_stop_support_hyperedge_sha256",
    "base_stop_terminal_leaf_id",
    "build_base_stop_proof_bytes",
    "build_conservative_primitive_interval_proof_bytes",
    "build_finite_terminal_leaf_proof_bytes",
    "canonical_score_implementation_sha256",
    "reverify_continuation_certificate",
    "terminal_leaf_set_root_sha256",
    "upstream_binary64_contest_score",
    "verify_base_stop_proof_bytes",
    "verify_continuation_proof_bytes",
]
