# SPDX-License-Identifier: MIT
"""Proof-carrying finite-horizon control for the selected-solution codec.

Costates, local derivatives, and proposal allocators are acquisition functions;
they never decide a mutation.  The decision unit here is a complete terminal
state reached through an ordered action path on one exact base.  The controller
prices the terminal nonlinear contest score, groups terminals by first action,
keeps ``BASE_STOP`` as an exact incumbent, and commits a first action only when
its feasible terminal upper bound is strictly below every competing branch's
verified lower bound.

The generator manifest is also load-bearing.  It declares the full fixed
family/scale vocabulary and an exact finite set of atomic and higher-order
terminal leaves.  Family-by-scale occupancy remains audit telemetry; it cannot
establish search closure.  Equal present evaluator cells are quotiented only
when continuation and proof-dependency state also agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from tac.contest_score import (
    compute_contest_score,
    pose_term,
    rate_term,
    seg_term,
)
from tac.score_geometry import score_sublevel_audit, score_transition_audit
from tac.witness_control.verified_continuation_certificate_v1 import (
    CANONICAL_SCORE_IMPLEMENTATION_SHA256,
    ContinuationVerificationContextV1,
    VerifiedContinuationCertificateV1,
    reverify_continuation_certificate,
    terminal_leaf_set_root_sha256,
)

SCHEMA: Final = "tac.taskspace_receding_horizon_decision.v1"

AuthorityMode = Literal[
    "RESEARCH_ADVISORY",
    "PRODUCTION_CONTEST_CPU",
    "PRODUCTION_CONTEST_CUDA",
]
ActionFamily = Literal[
    "PRUNE",
    "MERGE",
    "MACRO_EVICT",
    "MIGRATE",
    "INVERSE_REPAIR",
    "FIT_QUOTIENT",
    "TRAIN_QUOTIENT",
    "REQUANTIZE_STORAGE",
    "JOINT_DESCENT",
]
ActionScale = Literal["MICRO", "MESO", "MACRO", "SYSTEM"]
CoverageStatus = Literal[
    "GENERATED_MANIFEST_ACTIONS",
    "CERTIFIED_DOMINATED",
    "EXPLICITLY_BLOCKED",
    "EMPTY_IN_MANIFEST",
]

ACTION_FAMILIES: Final = (
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
ACTION_SCALES: Final = ("MICRO", "MESO", "MACRO", "SYSTEM")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
BASE_STOP_ACTION_ID: Final = "BASE_STOP"


class TaskspaceRecedingHorizonError(ValueError):
    """Raised before a decision when the universe loses exact-base custody."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TaskspaceRecedingHorizonError("value is not finite canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TaskspaceRecedingHorizonError(f"{label} must be canonical lowercase SHA-256")
    return value


def _require_id(value: object, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise TaskspaceRecedingHorizonError(f"{label} must be a bounded identifier")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskspaceRecedingHorizonError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise TaskspaceRecedingHorizonError(f"{label} left its finite physical domain")
    return result


def _score(d_seg: float, d_pose: float, archive_nbytes: int) -> float:
    if d_seg < 0.0 or d_pose < 0.0 or type(archive_nbytes) is not int or archive_nbytes < 1:
        raise TaskspaceRecedingHorizonError("score arguments left their physical domain")
    return compute_contest_score(d_seg, d_pose, archive_nbytes)


@dataclass(frozen=True, slots=True)
class WholeObjectBaseV1:
    epoch_id: str
    authority_mode: AuthorityMode
    axis: str
    archive_sha256: str
    archive_nbytes: int
    realized_output_sha256: str
    evaluator_cell_identity_sha256: str
    evaluator_output_cell_ledger_sha256: str
    continuation_equivalence_sha256: str
    continuation_equivalence_receipt_sha256: str
    decoder_runtime_sha256: str
    payload_placement_manifest_sha256: str
    measurement_receipt_sha256: str
    public_auth_eval_receipt_sha256: str
    dynamic_pointer_artifact_sha256: str
    dynamic_pointer_artifact_nbytes: int
    dynamic_target_score: float
    d_seg: float
    d_pose: float
    decode_constraints: DecodeConstraintEvidenceV1

    def __post_init__(self) -> None:
        _require_id(self.epoch_id, "base.epoch_id")
        if self.authority_mode not in (
            "RESEARCH_ADVISORY",
            "PRODUCTION_CONTEST_CPU",
            "PRODUCTION_CONTEST_CUDA",
        ):
            raise TaskspaceRecedingHorizonError("base authority mode is unsupported")
        if type(self.axis) is not str or not self.axis:
            raise TaskspaceRecedingHorizonError("base axis is empty")
        for field_name in (
            "archive_sha256",
            "realized_output_sha256",
            "evaluator_cell_identity_sha256",
            "evaluator_output_cell_ledger_sha256",
            "continuation_equivalence_sha256",
            "continuation_equivalence_receipt_sha256",
            "decoder_runtime_sha256",
            "payload_placement_manifest_sha256",
            "measurement_receipt_sha256",
            "public_auth_eval_receipt_sha256",
            "dynamic_pointer_artifact_sha256",
        ):
            _require_sha(getattr(self, field_name), f"base.{field_name}")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise TaskspaceRecedingHorizonError("base archive_nbytes must be positive int")
        if type(self.dynamic_pointer_artifact_nbytes) is not int or self.dynamic_pointer_artifact_nbytes < 1:
            raise TaskspaceRecedingHorizonError("base pointer bytes must be positive int")
        _finite(self.dynamic_target_score, "base.dynamic_target_score", nonnegative=True)
        _finite(self.d_seg, "base.d_seg", nonnegative=True)
        _finite(self.d_pose, "base.d_pose", nonnegative=True)
        if type(self.decode_constraints) is not DecodeConstraintEvidenceV1:
            raise TaskspaceRecedingHorizonError("base decode constraints require exact typed evidence")
        blockers = self.decode_constraints.blockers()
        if blockers:
            raise TaskspaceRecedingHorizonError(
                "base is not a feasible public-workflow incumbent: " + ",".join(blockers)
            )

    @property
    def score(self) -> float:
        return _score(self.d_seg, self.d_pose, self.archive_nbytes)

    @property
    def identity_sha256(self) -> str:
        return _sha256({"schema": "tac.taskspace_whole_object_base.v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class DecodeConstraintEvidenceV1:
    receipt_sha256: str
    measurement_scope: Literal["PUBLIC_EVALUATE_SH_WHOLE_WORKFLOW"]
    measured_wall_seconds: float
    wall_seconds_limit: float
    measured_peak_rss_bytes: int
    peak_rss_limit_bytes: int
    double_decode_bit_identical: bool
    public_inflate_video_closed: bool
    recursive_auth_eval_closed: bool
    hidden_data_absence_proved: bool
    network_access_absence_proved: bool
    source_video_access_absence_proved: bool
    scorer_access_absence_proved: bool
    deterministic_portability_proved: bool
    video_specific_state_fully_counted: bool

    def __post_init__(self) -> None:
        _require_sha(self.receipt_sha256, "decode constraint receipt")
        if self.measurement_scope != "PUBLIC_EVALUATE_SH_WHOLE_WORKFLOW":
            raise TaskspaceRecedingHorizonError(
                "wall-time authority must cover the complete upstream/evaluate.sh workflow"
            )
        wall = _finite(self.measured_wall_seconds, "decode measured wall seconds", nonnegative=True)
        limit = _finite(self.wall_seconds_limit, "decode wall seconds limit", nonnegative=True)
        if limit <= 0.0 or wall < 0.0:
            raise TaskspaceRecedingHorizonError("decode wall-time contract is invalid")
        for field_name in ("measured_peak_rss_bytes", "peak_rss_limit_bytes"):
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) < 0:
                raise TaskspaceRecedingHorizonError(f"decode {field_name} must be nonnegative int")
        for field_name in (
            "double_decode_bit_identical",
            "public_inflate_video_closed",
            "recursive_auth_eval_closed",
            "hidden_data_absence_proved",
            "network_access_absence_proved",
            "source_video_access_absence_proved",
            "scorer_access_absence_proved",
            "deterministic_portability_proved",
            "video_specific_state_fully_counted",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TaskspaceRecedingHorizonError(f"decode {field_name} must be bool")

    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.measured_wall_seconds > self.wall_seconds_limit:
            blockers.append("OFFICIAL_TOTAL_WORKFLOW_WALL_TIME_EXCEEDED")
        if self.measured_peak_rss_bytes > self.peak_rss_limit_bytes:
            blockers.append("DECODE_PEAK_RSS_EXCEEDED")
        for field_name, code in (
            ("double_decode_bit_identical", "DOUBLE_DECODE_MISMATCH"),
            ("public_inflate_video_closed", "PUBLIC_INFLATE_VIDEO_NOT_CLOSED"),
            ("recursive_auth_eval_closed", "RECURSIVE_AUTH_EVAL_NOT_CLOSED"),
            ("hidden_data_absence_proved", "HIDDEN_DATA_ABSENCE_UNPROVED"),
            ("network_access_absence_proved", "NETWORK_ABSENCE_UNPROVED"),
            ("source_video_access_absence_proved", "SOURCE_VIDEO_ABSENCE_UNPROVED"),
            ("scorer_access_absence_proved", "SCORER_ABSENCE_UNPROVED"),
            ("deterministic_portability_proved", "DETERMINISTIC_PORTABILITY_UNPROVED"),
            ("video_specific_state_fully_counted", "VIDEO_SPECIFIC_STATE_NOT_FULLY_COUNTED"),
        ):
            if getattr(self, field_name) is not True:
                blockers.append(code)
        return tuple(blockers)


@dataclass(frozen=True, slots=True, order=True)
class ActionSupportCellV1:
    """One cell in an atomic or higher-order action's support hyperedge."""

    family: ActionFamily
    scale: ActionScale

    def __post_init__(self) -> None:
        if self.family not in ACTION_FAMILIES:
            raise TaskspaceRecedingHorizonError("action support family left the closed vocabulary")
        if self.scale not in ACTION_SCALES:
            raise TaskspaceRecedingHorizonError("action support scale left the closed vocabulary")

    @property
    def key(self) -> str:
        return f"{self.family}:{self.scale}"


def action_support_hyperedge_sha256(
    support_hyperedge: Sequence[ActionSupportCellV1],
) -> str:
    """Canonical identity shared by manifests and continuation certificates."""

    if isinstance(support_hyperedge, (str, bytes)) or not isinstance(support_hyperedge, Sequence):
        raise TaskspaceRecedingHorizonError("support hyperedge must be a finite sequence")
    cells = tuple(support_hyperedge)
    if not cells or any(type(cell) is not ActionSupportCellV1 for cell in cells):
        raise TaskspaceRecedingHorizonError("support hyperedge must contain typed cells")
    if cells != tuple(sorted(set(cells))):
        raise TaskspaceRecedingHorizonError("support hyperedge must be canonical and unique")
    return _sha256(
        {
            "schema": "tac.action_support_hyperedge.v1",
            "members": [cell.key for cell in cells],
        }
    )


@dataclass(frozen=True, slots=True)
class GeneratedTerminalLeafV1:
    """One exact leaf of the finite generator-domain action tree.

    ``terminal_leaf_id`` names a leaf, while ``first_action_id`` is the only
    mutation a receding-horizon controller may commit.  The support hyperedge
    prevents a sign-reversing joint bundle from being laundered into one scalar
    family/scale bucket.
    """

    terminal_leaf_id: str
    first_action_id: str
    action_path: tuple[str, ...]
    support_hyperedge: tuple[ActionSupportCellV1, ...]
    parent_action_ids: tuple[str, ...]
    parameter_identity_sha256: str
    chronology_context_sha256: str
    exact_base_identity_sha256: str
    universe_epoch_id: str

    def __post_init__(self) -> None:
        _require_id(self.terminal_leaf_id, "generated leaf terminal_leaf_id")
        _require_id(self.first_action_id, "generated leaf first_action_id")
        _require_id(self.universe_epoch_id, "generated leaf universe_epoch_id")
        if self.first_action_id == BASE_STOP_ACTION_ID or self.terminal_leaf_id == BASE_STOP_ACTION_ID:
            raise TaskspaceRecedingHorizonError("BASE_STOP is controller-owned and may not be generated")
        if type(self.action_path) is not tuple or not self.action_path:
            raise TaskspaceRecedingHorizonError("generated leaf action_path must be a nonempty tuple")
        for action_id in self.action_path:
            _require_id(action_id, "generated leaf action_path entry")
        if self.action_path[0] != self.first_action_id:
            raise TaskspaceRecedingHorizonError("generated leaf first action differs from its ordered path")
        if type(self.parent_action_ids) is not tuple or self.parent_action_ids != self.action_path[:-1]:
            raise TaskspaceRecedingHorizonError("generated leaf parents must exactly equal the ordered path prefix")
        if type(self.support_hyperedge) is not tuple or not self.support_hyperedge:
            raise TaskspaceRecedingHorizonError("generated leaf support hyperedge must be nonempty")
        if any(type(cell) is not ActionSupportCellV1 for cell in self.support_hyperedge):
            raise TaskspaceRecedingHorizonError("generated leaf support contains an untyped cell")
        canonical_support = tuple(sorted(set(self.support_hyperedge)))
        if self.support_hyperedge != canonical_support:
            raise TaskspaceRecedingHorizonError("generated leaf support must be unique and canonically ordered")
        for field_name in (
            "parameter_identity_sha256",
            "chronology_context_sha256",
            "exact_base_identity_sha256",
        ):
            _require_sha(getattr(self, field_name), f"generated leaf {field_name}")

    @property
    def interaction_order(self) -> int:
        return len(self.support_hyperedge)

    @property
    def support_hyperedge_sha256(self) -> str:
        return action_support_hyperedge_sha256(self.support_hyperedge)

    @property
    def identity_sha256(self) -> str:
        return _sha256({"schema": "tac.generated_terminal_leaf.v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class GeneratorDomainManifestV1:
    """Exact finite action-tree manifest over the fixed full vocabulary.

    This closes a *declared finite generator domain*, not the unknowable set of
    every codec transformation that could ever be invented.  That scope is
    emitted in every decision so manifest closure cannot become a global
    optimality claim.
    """

    manifest_id: str
    epoch_id: str
    exact_base_identity_sha256: str
    generator_program_sha256: str
    enumeration_receipt_sha256: str
    parameter_domain_manifest_sha256: str
    chronology_domain_manifest_sha256: str
    deterministic_seed: int
    horizon: int
    maximum_interaction_order: int
    families: tuple[ActionFamily, ...]
    scales: tuple[ActionScale, ...]
    terminal_leaves: tuple[GeneratedTerminalLeafV1, ...]

    def __post_init__(self) -> None:
        _require_id(self.manifest_id, "generator manifest_id")
        _require_id(self.epoch_id, "generator epoch_id")
        for field_name in (
            "exact_base_identity_sha256",
            "generator_program_sha256",
            "enumeration_receipt_sha256",
            "parameter_domain_manifest_sha256",
            "chronology_domain_manifest_sha256",
        ):
            _require_sha(getattr(self, field_name), f"generator {field_name}")
        if type(self.deterministic_seed) is not int or self.deterministic_seed < 0:
            raise TaskspaceRecedingHorizonError("generator deterministic seed must be nonnegative int")
        if type(self.horizon) is not int or self.horizon < 1:
            raise TaskspaceRecedingHorizonError("generator horizon must be positive int")
        if (
            type(self.maximum_interaction_order) is not int
            or self.maximum_interaction_order < 1
            or self.maximum_interaction_order > len(ACTION_FAMILIES) * len(ACTION_SCALES)
        ):
            raise TaskspaceRecedingHorizonError("generator maximum interaction order is invalid")
        if self.families != ACTION_FAMILIES or self.scales != ACTION_SCALES:
            raise TaskspaceRecedingHorizonError(
                "generator manifest must declare the exact full action family and scale vocabulary"
            )
        if type(self.terminal_leaves) is not tuple:
            raise TaskspaceRecedingHorizonError("generator terminal leaves must be a tuple")
        if any(type(leaf) is not GeneratedTerminalLeafV1 for leaf in self.terminal_leaves):
            raise TaskspaceRecedingHorizonError("generator manifest contains an untyped leaf")
        leaf_ids = [leaf.terminal_leaf_id for leaf in self.terminal_leaves]
        if len(set(leaf_ids)) != len(leaf_ids):
            raise TaskspaceRecedingHorizonError("generator manifest repeats a terminal leaf ID")
        if tuple(sorted(self.terminal_leaves, key=lambda leaf: leaf.terminal_leaf_id)) != self.terminal_leaves:
            raise TaskspaceRecedingHorizonError("generator terminal leaves must be canonically ordered")
        for leaf in self.terminal_leaves:
            if (
                leaf.exact_base_identity_sha256 != self.exact_base_identity_sha256
                or leaf.universe_epoch_id != self.epoch_id
            ):
                raise TaskspaceRecedingHorizonError("generated leaf left the manifest base or epoch")
            if len(leaf.action_path) > self.horizon:
                raise TaskspaceRecedingHorizonError("generated leaf exceeds the finite control horizon")
            if leaf.interaction_order > self.maximum_interaction_order:
                raise TaskspaceRecedingHorizonError("generated leaf exceeds the interaction closure order")

    @property
    def generated_leaf_cardinality(self) -> int:
        return len(self.terminal_leaves)

    @property
    def generated_leaf_merkle_root_sha256(self) -> str:
        return _sha256(
            {
                "schema": "tac.generated_terminal_leaf_set.v1",
                "leaves": [
                    {
                        "terminal_leaf_id": leaf.terminal_leaf_id,
                        "identity_sha256": leaf.identity_sha256,
                    }
                    for leaf in self.terminal_leaves
                ],
            }
        )

    @property
    def generated_terminal_leaf_id_root_sha256(self) -> str | None:
        if not self.terminal_leaves:
            return None
        return terminal_leaf_set_root_sha256(tuple(leaf.terminal_leaf_id for leaf in self.terminal_leaves))

    @property
    def canonical_payload_bytes(self) -> bytes:
        """Exact bytes reopened by the continuation-certificate verifier."""

        return _canonical_json(
            {
                "schema": "tac.generator_domain_manifest.v1",
                **asdict(self),
                "generated_leaf_cardinality": self.generated_leaf_cardinality,
                "generated_leaf_merkle_root_sha256": self.generated_leaf_merkle_root_sha256,
                "generated_terminal_leaf_id_root_sha256": (self.generated_terminal_leaf_id_root_sha256),
            }
        )

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class WholeObjectActionEndpointV1:
    generated_leaf: GeneratedTerminalLeafV1
    authority_mode: AuthorityMode
    axis: str
    archive_sha256: str
    archive_nbytes: int
    realized_output_sha256: str
    evaluator_cell_identity_sha256: str
    evaluator_output_cell_ledger_sha256: str
    continuation_equivalence_sha256: str
    continuation_equivalence_receipt_sha256: str
    decoder_runtime_sha256: str
    payload_placement_manifest_sha256: str
    measurement_receipt_sha256: str
    public_auth_eval_receipt_sha256: str
    d_seg: float
    d_pose: float
    decode_constraints: DecodeConstraintEvidenceV1
    proposal_evidence_sha256: str
    interaction_hypergraph_sha256: str

    def __post_init__(self) -> None:
        if type(self.generated_leaf) is not GeneratedTerminalLeafV1:
            raise TaskspaceRecedingHorizonError("action endpoint requires a typed generated leaf")
        if self.authority_mode not in (
            "RESEARCH_ADVISORY",
            "PRODUCTION_CONTEST_CPU",
            "PRODUCTION_CONTEST_CUDA",
        ):
            raise TaskspaceRecedingHorizonError("action authority mode is unsupported")
        if type(self.axis) is not str or not self.axis:
            raise TaskspaceRecedingHorizonError("action axis is empty")
        for field_name in (
            "archive_sha256",
            "realized_output_sha256",
            "evaluator_cell_identity_sha256",
            "evaluator_output_cell_ledger_sha256",
            "continuation_equivalence_sha256",
            "continuation_equivalence_receipt_sha256",
            "decoder_runtime_sha256",
            "payload_placement_manifest_sha256",
            "measurement_receipt_sha256",
            "public_auth_eval_receipt_sha256",
            "proposal_evidence_sha256",
            "interaction_hypergraph_sha256",
        ):
            _require_sha(getattr(self, field_name), f"action.{field_name}")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise TaskspaceRecedingHorizonError("action archive_nbytes must be positive int")
        _finite(self.d_seg, "action.d_seg", nonnegative=True)
        _finite(self.d_pose, "action.d_pose", nonnegative=True)
        if type(self.decode_constraints) is not DecodeConstraintEvidenceV1:
            raise TaskspaceRecedingHorizonError("action decode constraints require exact typed evidence")

    @property
    def action_id(self) -> str:
        return self.generated_leaf.terminal_leaf_id

    @property
    def first_action_id(self) -> str:
        return self.generated_leaf.first_action_id

    @property
    def exact_base_identity_sha256(self) -> str:
        return self.generated_leaf.exact_base_identity_sha256

    @property
    def universe_epoch_id(self) -> str:
        return self.generated_leaf.universe_epoch_id

    @property
    def score(self) -> float:
        return _score(self.d_seg, self.d_pose, self.archive_nbytes)

    @property
    def endpoint_identity_sha256(self) -> str:
        return _sha256({"schema": "tac.taskspace_whole_object_action_endpoint.v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class ActionFamilyScaleCoverageV1:
    """Audit projection of the manifest; never a universe-completeness proof."""

    family: ActionFamily
    scale: ActionScale
    status: CoverageStatus
    action_ids: tuple[str, ...]
    evidence_sha256: str
    blocks_global_selection: bool
    verdict_scope: str
    optimistic_score_lower_bound: float | None = None

    def __post_init__(self) -> None:
        if self.family not in ACTION_FAMILIES:
            raise TaskspaceRecedingHorizonError("coverage family left the closed vocabulary")
        if self.scale not in ACTION_SCALES:
            raise TaskspaceRecedingHorizonError("coverage scale left the closed vocabulary")
        if self.status not in (
            "GENERATED_MANIFEST_ACTIONS",
            "EXPLICITLY_BLOCKED",
            "EMPTY_IN_MANIFEST",
        ):
            if self.status == "CERTIFIED_DOMINATED":
                raise TaskspaceRecedingHorizonError(
                    "CERTIFIED_DOMINATED requires a reopened typed verifier proof; scalar plus SHA is forbidden"
                )
            raise TaskspaceRecedingHorizonError("coverage status is unsupported")
        if type(self.action_ids) is not tuple or len(set(self.action_ids)) != len(self.action_ids):
            raise TaskspaceRecedingHorizonError("coverage action_ids must be a unique tuple")
        for action_id in self.action_ids:
            _require_id(action_id, "coverage action_id")
        _require_sha(self.evidence_sha256, "coverage evidence")
        if type(self.blocks_global_selection) is not bool:
            raise TaskspaceRecedingHorizonError("coverage blocks_global_selection must be bool")
        if type(self.verdict_scope) is not str or not self.verdict_scope:
            raise TaskspaceRecedingHorizonError("coverage verdict scope is empty")
        if self.status == "GENERATED_MANIFEST_ACTIONS" and not self.action_ids:
            raise TaskspaceRecedingHorizonError("generated coverage requires manifest action ids")
        if self.status != "GENERATED_MANIFEST_ACTIONS" and self.action_ids:
            raise TaskspaceRecedingHorizonError("blocked/empty coverage may not name manifest actions")
        if self.optimistic_score_lower_bound is not None:
            raise TaskspaceRecedingHorizonError(
                "caller-authored optimistic score bounds are forbidden; provide a verified typed proof"
            )
        if self.status == "EXPLICITLY_BLOCKED" and not self.blocks_global_selection:
            raise TaskspaceRecedingHorizonError("explicitly blocked coverage must block global selection")
        if self.status in ("GENERATED_MANIFEST_ACTIONS", "EMPTY_IN_MANIFEST") and self.blocks_global_selection:
            raise TaskspaceRecedingHorizonError("generated or empty coverage cannot block selection")


@dataclass(frozen=True, slots=True)
class ContinuationProofDependencyPayloadV1:
    """One exact dependency blob reopened at every bound-consuming boundary."""

    sha256: str
    payload_bytes: bytes

    def __post_init__(self) -> None:
        _require_sha(self.sha256, "continuation dependency")
        if type(self.payload_bytes) is not bytes or not self.payload_bytes:
            raise TaskspaceRecedingHorizonError("continuation dependency payload must be nonempty exact bytes")
        if hashlib.sha256(self.payload_bytes).hexdigest() != self.sha256:
            raise TaskspaceRecedingHorizonError("continuation dependency payload differs from its SHA-256")


@dataclass(frozen=True, slots=True)
class ContinuationCertificateEvidenceBundleV1:
    """Durable proof material required to consume one G38 certificate.

    The certificate object is only a cached verifier result.  It cannot grant
    authority by itself: ``reverify`` reconstructs a fresh result from the
    exact proof, manifest, independent context, and dependency bytes.
    """

    certificate: VerifiedContinuationCertificateV1
    proof_payload_bytes: bytes
    verification_context: ContinuationVerificationContextV1
    generator_domain_manifest_payload_bytes: bytes
    dependency_payloads: tuple[ContinuationProofDependencyPayloadV1, ...]

    def __post_init__(self) -> None:
        if type(self.certificate) is not VerifiedContinuationCertificateV1:
            raise TaskspaceRecedingHorizonError("continuation evidence requires a sealed G38 certificate result")
        if type(self.proof_payload_bytes) is not bytes or not self.proof_payload_bytes:
            raise TaskspaceRecedingHorizonError("continuation proof payload must be nonempty exact bytes")
        if type(self.verification_context) is not ContinuationVerificationContextV1:
            raise TaskspaceRecedingHorizonError("continuation evidence requires typed independent verification context")
        if (
            type(self.generator_domain_manifest_payload_bytes) is not bytes
            or not self.generator_domain_manifest_payload_bytes
        ):
            raise TaskspaceRecedingHorizonError("continuation generator manifest payload must be nonempty exact bytes")
        if type(self.dependency_payloads) is not tuple or not self.dependency_payloads:
            raise TaskspaceRecedingHorizonError("continuation evidence requires exact dependency payloads")
        if any(type(row) is not ContinuationProofDependencyPayloadV1 for row in self.dependency_payloads):
            raise TaskspaceRecedingHorizonError("continuation evidence contains an untyped dependency payload")
        identities = tuple(row.sha256 for row in self.dependency_payloads)
        if identities != tuple(sorted(set(identities))):
            raise TaskspaceRecedingHorizonError(
                "continuation dependency payloads must be canonically ordered and unique"
            )

    @property
    def dependency_payloads_by_sha256(self) -> dict[str, bytes]:
        return {row.sha256: row.payload_bytes for row in self.dependency_payloads}

    def reverify(self) -> VerifiedContinuationCertificateV1:
        return reverify_continuation_certificate(
            self.certificate,
            self.proof_payload_bytes,
            context=self.verification_context,
            generator_domain_manifest_payload_bytes=(self.generator_domain_manifest_payload_bytes),
            dependency_payloads_by_sha256=self.dependency_payloads_by_sha256,
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {
                "schema": "tac.continuation_certificate_evidence_bundle.v1",
                "certificate": asdict(self.certificate),
                "proof_payload_sha256": hashlib.sha256(self.proof_payload_bytes).hexdigest(),
                "verification_context": asdict(self.verification_context),
                "generator_domain_manifest_payload_sha256": hashlib.sha256(
                    self.generator_domain_manifest_payload_bytes
                ).hexdigest(),
                "dependency_payload_sha256s": [row.sha256 for row in self.dependency_payloads],
            }
        )


@dataclass(frozen=True, slots=True)
class CompleteActionUniverseV1:
    universe_id: str
    generator_manifest: GeneratorDomainManifestV1
    coverage: tuple[ActionFamilyScaleCoverageV1, ...]
    endpoints: tuple[WholeObjectActionEndpointV1, ...]
    continuation_evidence_bundles: tuple[ContinuationCertificateEvidenceBundleV1, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.universe_id, "universe.universe_id")
        if type(self.generator_manifest) is not GeneratorDomainManifestV1:
            raise TaskspaceRecedingHorizonError("universe requires an exact typed generator manifest")
        if (
            type(self.coverage) is not tuple
            or type(self.endpoints) is not tuple
            or type(self.continuation_evidence_bundles) is not tuple
        ):
            raise TaskspaceRecedingHorizonError("universe coverage/endpoints/continuation evidence must be tuples")
        if any(type(row) is not ActionFamilyScaleCoverageV1 for row in self.coverage):
            raise TaskspaceRecedingHorizonError("universe coverage has an untyped row")
        if any(type(row) is not WholeObjectActionEndpointV1 for row in self.endpoints):
            raise TaskspaceRecedingHorizonError("universe endpoints have an untyped row")
        if any(type(row) is not ContinuationCertificateEvidenceBundleV1 for row in self.continuation_evidence_bundles):
            raise TaskspaceRecedingHorizonError(
                "bare continuation certificates are forbidden; provide durable proof-byte evidence bundles"
            )
        reverified_certificates = self.reverified_continuation_certificates
        expected_cells = {(family, scale) for family in ACTION_FAMILIES for scale in ACTION_SCALES}
        coverage_by_cell = {(row.family, row.scale): row for row in self.coverage}
        if len(coverage_by_cell) != len(self.coverage) or set(coverage_by_cell) != expected_cells:
            raise TaskspaceRecedingHorizonError(
                "universe audit projection must contain the exact full family x scale vocabulary"
            )
        endpoint_by_id = {row.action_id: row for row in self.endpoints}
        if len(endpoint_by_id) != len(self.endpoints):
            raise TaskspaceRecedingHorizonError("universe repeats an action endpoint")
        manifest_by_id = {leaf.terminal_leaf_id: leaf for leaf in self.generator_manifest.terminal_leaves}
        measured_ids = set(endpoint_by_id)
        certificate_ids: set[str] = set()
        for certificate in reverified_certificates:
            covered = set(certificate.covered_terminal_leaf_ids)
            if certificate_ids & covered or measured_ids & covered:
                raise TaskspaceRecedingHorizonError(
                    "measured endpoints and verified certificate subtrees must be disjoint"
                )
            certificate_ids.update(covered)
        if measured_ids | certificate_ids != set(manifest_by_id):
            raise TaskspaceRecedingHorizonError(
                "measured leaves plus verified certificate subtrees differ from the generator manifest leaf set"
            )
        for action_id, endpoint in endpoint_by_id.items():
            if endpoint.generated_leaf != manifest_by_id[action_id]:
                raise TaskspaceRecedingHorizonError("endpoint generated-leaf descriptor differs from the manifest")
        for certificate in reverified_certificates:
            leaves = [manifest_by_id[leaf_id] for leaf_id in certificate.covered_terminal_leaf_ids]
            if (
                certificate.exact_base_identity_sha256 != self.exact_base_identity_sha256
                or certificate.epoch_id != self.epoch_id
                or certificate.generator_domain_manifest_sha256 != self.generator_manifest.identity_sha256
                or certificate.horizon != self.generator_manifest.horizon
                or certificate.score_implementation_sha256 != CANONICAL_SCORE_IMPLEMENTATION_SHA256
            ):
                raise TaskspaceRecedingHorizonError(
                    "verified continuation certificate left its exact manifest/base/horizon/score binding"
                )
            if any(leaf.first_action_id != certificate.first_action_id for leaf in leaves):
                raise TaskspaceRecedingHorizonError("verified continuation certificate crosses first-action branches")
            support_identities = {leaf.support_hyperedge_sha256 for leaf in leaves}
            if support_identities != {certificate.support_hyperedge_sha256}:
                raise TaskspaceRecedingHorizonError("verified continuation certificate crosses support hyperedges")
            if certificate.subtree_cardinality != len(
                leaves
            ) or certificate.subtree_root_sha256 != terminal_leaf_set_root_sha256(
                certificate.covered_terminal_leaf_ids
            ):
                raise TaskspaceRecedingHorizonError(
                    "verified continuation certificate subtree partition changed after verification"
                )
        for cell, row in coverage_by_cell.items():
            expected_ids = tuple(
                sorted(
                    leaf.terminal_leaf_id
                    for leaf in self.generator_manifest.terminal_leaves
                    if ActionSupportCellV1(*cell) in leaf.support_hyperedge
                )
            )
            if expected_ids:
                if row.status != "GENERATED_MANIFEST_ACTIONS" or tuple(sorted(row.action_ids)) != expected_ids:
                    raise TaskspaceRecedingHorizonError(
                        "coverage projection lost or laundered a generated support-hyperedge leaf"
                    )
            elif row.status == "GENERATED_MANIFEST_ACTIONS" or row.action_ids:
                raise TaskspaceRecedingHorizonError(
                    "coverage projection invented a leaf outside the generator support hypergraph"
                )
        for row in self.endpoints:
            if (
                row.exact_base_identity_sha256 != self.exact_base_identity_sha256
                or row.universe_epoch_id != self.epoch_id
            ):
                raise TaskspaceRecedingHorizonError("endpoint left its exact generator base or epoch")

    @property
    def epoch_id(self) -> str:
        return self.generator_manifest.epoch_id

    @property
    def exact_base_identity_sha256(self) -> str:
        return self.generator_manifest.exact_base_identity_sha256

    @property
    def generator_program_sha256(self) -> str:
        return self.generator_manifest.generator_program_sha256

    @property
    def enumeration_receipt_sha256(self) -> str:
        return self.generator_manifest.enumeration_receipt_sha256

    @property
    def required_families(self) -> tuple[ActionFamily, ...]:
        return self.generator_manifest.families

    @property
    def required_scales(self) -> tuple[ActionScale, ...]:
        return self.generator_manifest.scales

    @property
    def reverified_continuation_certificates(self) -> tuple[VerifiedContinuationCertificateV1, ...]:
        """Reopen every durable proof input before exposing certificate values."""

        if type(self.continuation_evidence_bundles) is not tuple or any(
            type(bundle) is not ContinuationCertificateEvidenceBundleV1 for bundle in self.continuation_evidence_bundles
        ):
            raise TaskspaceRecedingHorizonError(
                "bare continuation certificates are forbidden; provide durable proof-byte evidence bundles"
            )
        for bundle in self.continuation_evidence_bundles:
            if bundle.generator_domain_manifest_payload_bytes != self.generator_manifest.canonical_payload_bytes:
                raise TaskspaceRecedingHorizonError(
                    "continuation evidence manifest bytes differ from the exact universe manifest"
                )
        return tuple(bundle.reverify() for bundle in self.continuation_evidence_bundles)

    @property
    def enumeration_complete(self) -> Literal[True]:
        """Every manifest leaf is measured or covered by reverified proof bytes."""

        _ = self.reverified_continuation_certificates
        return True

    @property
    def cross_scale_scan_complete(self) -> Literal[True]:
        """The fixed full family/scale vocabulary is present as audit telemetry."""

        return True

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            {
                "schema": "tac.taskspace_complete_action_universe.v1",
                "universe_id": self.universe_id,
                "epoch_id": self.epoch_id,
                "exact_base_identity_sha256": self.exact_base_identity_sha256,
                "generator_manifest_identity_sha256": self.generator_manifest.identity_sha256,
                "generated_leaf_merkle_root_sha256": (self.generator_manifest.generated_leaf_merkle_root_sha256),
                "generated_leaf_cardinality": self.generator_manifest.generated_leaf_cardinality,
                "coverage": [
                    {
                        **asdict(row),
                        "action_ids": sorted(row.action_ids),
                    }
                    for row in sorted(self.coverage, key=lambda item: (item.family, item.scale))
                ],
                "endpoints": [asdict(row) for row in sorted(self.endpoints, key=lambda item: item.action_id)],
                "continuation_evidence_bundles": [
                    {
                        "identity_sha256": bundle.identity_sha256,
                        "certificate": asdict(certificate),
                    }
                    for bundle, certificate in sorted(
                        zip(
                            self.continuation_evidence_bundles,
                            self.reverified_continuation_certificates,
                            strict=True,
                        ),
                        key=lambda item: (
                            item[1].first_action_id,
                            item[1].continuation_equivalence_sha256,
                            item[1].subtree_root_sha256,
                        ),
                    )
                ],
                "finite_manifest_leaf_equality_closed": True,
                "family_scale_projection_is_load_bearing": False,
            }
        )


def _endpoint_row(endpoint: WholeObjectActionEndpointV1, base: WholeObjectBaseV1) -> dict[str, Any]:
    seg_before = seg_term(base.d_seg)
    pose_before = pose_term(base.d_pose)
    rate_before = rate_term(base.archive_nbytes)
    seg_after = seg_term(endpoint.d_seg)
    pose_after = pose_term(endpoint.d_pose)
    rate_after = rate_term(endpoint.archive_nbytes)
    blockers = endpoint.decode_constraints.blockers()
    sublevel = score_sublevel_audit(
        target_score=base.dynamic_target_score,
        d_seg=endpoint.d_seg,
        d_pose=endpoint.d_pose,
        archive_bytes=endpoint.archive_nbytes,
    )
    transition = score_transition_audit(
        target_score=base.dynamic_target_score,
        before_d_seg=base.d_seg,
        before_d_pose=base.d_pose,
        before_archive_bytes=base.archive_nbytes,
        after_d_seg=endpoint.d_seg,
        after_d_pose=endpoint.d_pose,
        after_archive_bytes=endpoint.archive_nbytes,
    )
    return {
        "terminal_leaf_id": endpoint.action_id,
        "action_id": endpoint.action_id,
        "first_action_id": endpoint.first_action_id,
        "action_path": list(endpoint.generated_leaf.action_path),
        "parent_action_ids": list(endpoint.generated_leaf.parent_action_ids),
        "support_hyperedge": [cell.key for cell in endpoint.generated_leaf.support_hyperedge],
        "interaction_order": endpoint.generated_leaf.interaction_order,
        "generated_leaf_identity_sha256": endpoint.generated_leaf.identity_sha256,
        "chronology_context_sha256": endpoint.generated_leaf.chronology_context_sha256,
        "family_scale_scalar_label_used_for_closure": False,
        "authority_mode": endpoint.authority_mode,
        "axis": endpoint.axis,
        "endpoint_identity_sha256": endpoint.endpoint_identity_sha256,
        "archive_sha256": endpoint.archive_sha256,
        "archive_nbytes": endpoint.archive_nbytes,
        "realized_output_sha256": endpoint.realized_output_sha256,
        "evaluator_cell_identity_sha256": endpoint.evaluator_cell_identity_sha256,
        "evaluator_output_cell_ledger_sha256": endpoint.evaluator_output_cell_ledger_sha256,
        "continuation_equivalence_sha256": endpoint.continuation_equivalence_sha256,
        "continuation_equivalence_receipt_sha256": (endpoint.continuation_equivalence_receipt_sha256),
        "proof_dependency_identity_sha256": _sha256(
            {
                "schema": "tac.taskspace_terminal_proof_dependencies.v1",
                "evaluator_output_cell_ledger_sha256": endpoint.evaluator_output_cell_ledger_sha256,
                "continuation_equivalence_receipt_sha256": (endpoint.continuation_equivalence_receipt_sha256),
                "decoder_runtime_sha256": endpoint.decoder_runtime_sha256,
                "payload_placement_manifest_sha256": endpoint.payload_placement_manifest_sha256,
                "measurement_receipt_sha256": endpoint.measurement_receipt_sha256,
                "public_auth_eval_receipt_sha256": endpoint.public_auth_eval_receipt_sha256,
            }
        ),
        "measurement_receipt_sha256": endpoint.measurement_receipt_sha256,
        "public_auth_eval_receipt_sha256": endpoint.public_auth_eval_receipt_sha256,
        "payload_placement_manifest_sha256": endpoint.payload_placement_manifest_sha256,
        "decode_constraint_receipt_sha256": endpoint.decode_constraints.receipt_sha256,
        "hard_constraint_blockers": list(blockers),
        "hard_constraints_passed": not blockers,
        "d_seg": endpoint.d_seg,
        "d_pose": endpoint.d_pose,
        "score": endpoint.score,
        "score_delta": endpoint.score - base.score,
        "finite_action_costate": {
            "seg_component_delta": seg_after - seg_before,
            "pose_component_delta": pose_after - pose_before,
            "rate_component_delta": rate_after - rate_before,
            "joint_score_delta": transition.exact_score_delta,
            "linearized_score_delta_at_before": transition.linearized_score_delta_at_before,
            "nonlinear_remainder": transition.nonlinear_remainder,
            "changed_axes": list(transition.changed_axes),
            "jointly_coupled_transition": transition.jointly_coupled_transition,
            "derivative_claim": False,
            "additive_prediction_used_for_verdict": False,
        },
        "coupled_target_sublevel": asdict(sublevel),
        "inside_dynamic_target_sublevel": sublevel.inside_strict_sublevel,
    }


def _functional_quotient(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Quotient only present, continuation, and proof-dependency equivalent states."""

    by_full_key: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    by_cell: dict[str, set[tuple[str, str]]] = {}
    cell_measurements: dict[str, tuple[object, object]] = {}
    for row in rows:
        cell = str(row["evaluator_cell_identity_sha256"])
        continuation = str(row["continuation_equivalence_sha256"])
        proof_dependencies = str(row["proof_dependency_identity_sha256"])
        measurement = (row["d_seg"], row["d_pose"])
        if cell in cell_measurements and cell_measurements[cell] != measurement:
            raise TaskspaceRecedingHorizonError(
                "equal evaluator-cell identities disagree on realized Seg/Pose measurements"
            )
        cell_measurements[cell] = measurement
        by_full_key.setdefault((cell, continuation, proof_dependencies), []).append(row)
        by_cell.setdefault(cell, set()).add((continuation, proof_dependencies))

    representatives: list[dict[str, Any]] = []
    quotient_classes: list[dict[str, Any]] = []
    for (cell, continuation, proof_dependencies), members in sorted(by_full_key.items()):
        ordered = sorted(
            members,
            key=lambda row: (
                row["score"],
                row["archive_nbytes"],
                row["action_id"],
            ),
        )
        representative = dict(ordered[0])
        representatives.append(representative)
        quotient_classes.append(
            {
                "evaluator_cell_identity_sha256": cell,
                "continuation_equivalence_sha256": continuation,
                "proof_dependency_identity_sha256": proof_dependencies,
                "member_action_ids": [row["action_id"] for row in ordered],
                "representative_action_id": representative["action_id"],
                "dominated_action_ids": [row["action_id"] for row in ordered[1:]],
                "quotient_safe": True,
            }
        )
    preserved_future_distinctions = [
        {
            "evaluator_cell_identity_sha256": cell,
            "continuation_and_proof_dependency_identities": [
                {
                    "continuation_equivalence_sha256": continuation,
                    "proof_dependency_identity_sha256": proof_dependencies,
                }
                for continuation, proof_dependencies in sorted(continuations)
            ],
            "reason": (
                "same present evaluator cells but different reachable future action spaces or proof dependencies"
            ),
        }
        for cell, continuations in sorted(by_cell.items())
        if len(continuations) > 1
    ]
    return representatives, quotient_classes, preserved_future_distinctions


def select_receding_horizon_action_v1(
    base: WholeObjectBaseV1,
    universe: CompleteActionUniverseV1,
) -> dict[str, Any]:
    """Commit at most one first action under exact finite-horizon dominance."""

    if type(base) is not WholeObjectBaseV1 or type(universe) is not CompleteActionUniverseV1:
        raise TaskspaceRecedingHorizonError("controller inputs require exact typed base and universe")
    if universe.exact_base_identity_sha256 != base.identity_sha256 or universe.epoch_id != base.epoch_id:
        raise TaskspaceRecedingHorizonError("action universe is stale or bound to another exact base")
    continuation_certificates = universe.reverified_continuation_certificates
    if any(
        endpoint.authority_mode != base.authority_mode or endpoint.axis != base.axis for endpoint in universe.endpoints
    ):
        raise TaskspaceRecedingHorizonError("action endpoint authority axis differs from its exact base")
    if any(
        certificate.authority_mode != base.authority_mode
        or certificate.axis != base.axis
        or certificate.hard_constraint_contract_sha256 != base.decode_constraints.receipt_sha256
        for certificate in continuation_certificates
    ):
        raise TaskspaceRecedingHorizonError(
            "verified continuation certificate authority axis or hard-constraint contract differs from its base"
        )

    ordered_endpoints = sorted(universe.endpoints, key=lambda row: row.action_id)
    endpoint_rows = [_endpoint_row(endpoint, base) for endpoint in ordered_endpoints]
    hard_admissible = [row for row in endpoint_rows if row["hard_constraints_passed"]]
    quotient_rows, quotient_classes, preserved_future_distinctions = _functional_quotient(hard_admissible)
    blocking_coverage_cells = sorted(
        (row.family, row.scale) for row in universe.coverage if row.blocks_global_selection
    )
    blocking_coverage = sorted({family for family, _scale in blocking_coverage_cells})
    finite_manifest_closed = not blocking_coverage_cells

    manifest_by_id = {leaf.terminal_leaf_id: leaf for leaf in universe.generator_manifest.terminal_leaves}
    value_pieces_by_first_action: dict[str, list[dict[str, Any]]] = {}
    for row in hard_admissible:
        value_pieces_by_first_action.setdefault(str(row["first_action_id"]), []).append(
            {
                "source_kind": "EXACT_MEASURED_WHOLE_OBJECT_ENDPOINT",
                "source_identity_sha256": row["endpoint_identity_sha256"],
                "continuation_equivalence_sha256": row["continuation_equivalence_sha256"],
                "proof_dependency_identity_sha256": row["proof_dependency_identity_sha256"],
                "terminal_lower_bound": float(row["score"]),
                "feasible_terminal_upper_bound": float(row["score"]),
                "best_terminal": dict(row),
                "covered_terminal_leaf_ids": [row["terminal_leaf_id"]],
                "bounds_equal": True,
                "verifier_status": "EXACT_MEASURED_ENDPOINT",
            }
        )
    for certificate in continuation_certificates:
        feasible_leaf_id: str | None = None
        if certificate.feasible_terminal_action_path:
            path_matches = [
                leaf.terminal_leaf_id
                for leaf in (manifest_by_id[leaf_id] for leaf_id in certificate.covered_terminal_leaf_ids)
                if leaf.action_path == certificate.feasible_terminal_action_path
            ]
            if len(path_matches) != 1:
                raise TaskspaceRecedingHorizonError(
                    "verified feasible terminal path does not identify exactly one manifest leaf"
                )
            feasible_leaf_id = path_matches[0]
        certificate_terminal = (
            None
            if certificate.feasible_terminal_upper_bound is None
            else {
                "terminal_leaf_id": feasible_leaf_id,
                "action_id": feasible_leaf_id,
                "first_action_id": certificate.first_action_id,
                "action_path": list(certificate.feasible_terminal_action_path),
                "endpoint_identity_sha256": (certificate.feasible_terminal_endpoint_identity_sha256),
                "continuation_equivalence_sha256": (certificate.continuation_equivalence_sha256),
                "proof_dependency_identity_sha256": _sha256(
                    {
                        "schema": "tac.verified_continuation_proof_dependencies.v1",
                        "proof_sha256": certificate.proof_sha256,
                        "proof_dependency_sha256s": list(certificate.proof_dependency_sha256s),
                    }
                ),
                "score": certificate.feasible_terminal_upper_bound,
                "inside_dynamic_target_sublevel": (
                    certificate.feasible_terminal_upper_bound < base.dynamic_target_score
                ),
                "source_kind": "VERIFIED_CONTINUATION_CERTIFICATE",
                "certificate_proof_sha256": certificate.proof_sha256,
            }
        )
        value_pieces_by_first_action.setdefault(certificate.first_action_id, []).append(
            {
                "source_kind": "VERIFIED_CONTINUATION_CERTIFICATE",
                "source_identity_sha256": certificate.proof_sha256,
                "continuation_equivalence_sha256": (certificate.continuation_equivalence_sha256),
                "proof_dependency_identity_sha256": _sha256(
                    {
                        "schema": "tac.verified_continuation_proof_dependencies.v1",
                        "proof_sha256": certificate.proof_sha256,
                        "proof_dependency_sha256s": list(certificate.proof_dependency_sha256s),
                    }
                ),
                "terminal_lower_bound": certificate.terminal_lower_bound,
                "feasible_terminal_upper_bound": (certificate.feasible_terminal_upper_bound),
                "best_terminal": certificate_terminal,
                "covered_terminal_leaf_ids": list(certificate.covered_terminal_leaf_ids),
                "bounds_equal": (certificate.feasible_terminal_upper_bound == certificate.terminal_lower_bound),
                "verifier_status": certificate.verifier_status,
                "outward_rounding_policy": certificate.outward_rounding_policy,
            }
        )

    manifest_leaves_by_first_action: dict[str, list[GeneratedTerminalLeafV1]] = {}
    for leaf in universe.generator_manifest.terminal_leaves:
        manifest_leaves_by_first_action.setdefault(leaf.first_action_id, []).append(leaf)
    first_action_branches: list[dict[str, Any]] = []
    for first_action_id, manifest_leaves in sorted(manifest_leaves_by_first_action.items()):
        pieces = value_pieces_by_first_action.get(first_action_id, [])
        lower_bounds = [float(piece["terminal_lower_bound"]) for piece in pieces]
        upper_pieces = [piece for piece in pieces if piece["feasible_terminal_upper_bound"] is not None]
        best_upper_piece = (
            None
            if not upper_pieces
            else min(
                upper_pieces,
                key=lambda piece: (
                    piece["feasible_terminal_upper_bound"],
                    piece["source_identity_sha256"],
                ),
            )
        )
        continuation_values: list[dict[str, Any]] = []
        by_continuation: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for piece in pieces:
            key = (
                str(piece["continuation_equivalence_sha256"]),
                str(piece["proof_dependency_identity_sha256"]),
            )
            by_continuation.setdefault(key, []).append(piece)
        for (continuation, proof_dependencies), continuation_pieces in sorted(by_continuation.items()):
            continuation_upper_pieces = [
                piece for piece in continuation_pieces if piece["feasible_terminal_upper_bound"] is not None
            ]
            continuation_values.append(
                {
                    "continuation_equivalence_sha256": continuation,
                    "proof_dependency_identity_sha256": proof_dependencies,
                    "terminal_lower_bound": min(float(piece["terminal_lower_bound"]) for piece in continuation_pieces),
                    "feasible_terminal_upper_bound": (
                        None
                        if not continuation_upper_pieces
                        else min(float(piece["feasible_terminal_upper_bound"]) for piece in continuation_upper_pieces)
                    ),
                    "covered_terminal_leaf_ids": sorted(
                        leaf_id for piece in continuation_pieces for leaf_id in piece["covered_terminal_leaf_ids"]
                    ),
                    "value_sources": continuation_pieces,
                }
            )
        first_action_branches.append(
            {
                "first_action_id": first_action_id,
                "terminal_lower_bound": min(lower_bounds) if lower_bounds else None,
                "feasible_terminal_upper_bound": (
                    None if best_upper_piece is None else float(best_upper_piece["feasible_terminal_upper_bound"])
                ),
                "best_terminal_leaf_id": (
                    None
                    if best_upper_piece is None or best_upper_piece["best_terminal"] is None
                    else best_upper_piece["best_terminal"]["terminal_leaf_id"]
                ),
                "best_terminal": (None if best_upper_piece is None else best_upper_piece["best_terminal"]),
                "continuation_values": continuation_values,
                "manifest_terminal_leaf_ids": sorted(leaf.terminal_leaf_id for leaf in manifest_leaves),
                "finite_terminal_value_exact": bool(pieces) and all(piece["bounds_equal"] for piece in pieces),
                "all_generated_terminals_proved_hard_infeasible": not pieces,
            }
        )

    base_stop = {
        "first_action_id": BASE_STOP_ACTION_ID,
        "terminal_lower_bound": base.score,
        "feasible_terminal_upper_bound": base.score,
        "best_terminal_leaf_id": None,
        "best_terminal": None,
        "continuation_values": [
            {
                "continuation_equivalence_sha256": base.continuation_equivalence_sha256,
                "proof_dependency_identity_sha256": _sha256(
                    {
                        "schema": "tac.taskspace_base_stop_proof_dependencies.v1",
                        "archive_sha256": base.archive_sha256,
                        "evaluator_output_cell_ledger_sha256": (base.evaluator_output_cell_ledger_sha256),
                        "continuation_equivalence_receipt_sha256": (base.continuation_equivalence_receipt_sha256),
                        "decoder_runtime_sha256": base.decoder_runtime_sha256,
                        "payload_placement_manifest_sha256": (base.payload_placement_manifest_sha256),
                        "measurement_receipt_sha256": base.measurement_receipt_sha256,
                        "public_auth_eval_receipt_sha256": base.public_auth_eval_receipt_sha256,
                    }
                ),
                "terminal_lower_bound": base.score,
                "feasible_terminal_upper_bound": base.score,
                "bounds_equal_because_base_is_exact_feasible_incumbent": True,
            }
        ],
        "manifest_terminal_leaf_ids": [],
        "finite_terminal_value_exact": True,
        "controller_owned": True,
    }

    def _terminal_equivalence_key(branch: Mapping[str, Any]) -> tuple[object, ...] | None:
        terminal = branch.get("best_terminal")
        required = (
            "archive_sha256",
            "archive_nbytes",
            "realized_output_sha256",
            "evaluator_cell_identity_sha256",
            "continuation_equivalence_sha256",
            "proof_dependency_identity_sha256",
            "d_seg",
            "d_pose",
        )
        if not isinstance(terminal, Mapping) or any(key not in terminal for key in required):
            return None
        return (
            terminal["archive_sha256"],
            terminal["archive_nbytes"],
            terminal["realized_output_sha256"],
            terminal["evaluator_cell_identity_sha256"],
            terminal["continuation_equivalence_sha256"],
            terminal["proof_dependency_identity_sha256"],
            terminal["d_seg"],
            terminal["d_pose"],
        )

    improving_branches = [
        branch
        for branch in first_action_branches
        if branch["feasible_terminal_upper_bound"] is not None
        and float(branch["feasible_terminal_upper_bound"]) < base.score
    ]
    improving_branches.sort(
        key=lambda branch: (
            branch["feasible_terminal_upper_bound"],
            branch["first_action_id"],
        )
    )
    selected_branch: dict[str, Any] | None = None
    dominance_blockers: list[str] = []
    equivalent_tie_first_actions: list[str] = []
    open_below_base_first_actions = sorted(
        str(branch["first_action_id"])
        for branch in first_action_branches
        if branch["terminal_lower_bound"] is not None
        and float(branch["terminal_lower_bound"]) < base.score
        and (
            branch["feasible_terminal_upper_bound"] is None
            or float(branch["feasible_terminal_upper_bound"]) >= base.score
        )
    )
    if finite_manifest_closed and improving_branches:
        candidate = improving_branches[0]
        candidate_upper = float(candidate["feasible_terminal_upper_bound"])
        candidate_equivalence = _terminal_equivalence_key(candidate)
        for competitor in [base_stop, *first_action_branches]:
            if competitor["first_action_id"] == candidate["first_action_id"]:
                continue
            if competitor["terminal_lower_bound"] is None:
                continue
            competitor_lower = float(competitor["terminal_lower_bound"])
            if candidate_upper < competitor_lower:
                continue
            if (
                candidate_upper == competitor_lower
                and candidate_equivalence is not None
                and candidate_equivalence == _terminal_equivalence_key(competitor)
            ):
                equivalent_tie_first_actions.append(str(competitor["first_action_id"]))
                continue
            dominance_blockers.append(str(competitor["first_action_id"]))
        if not dominance_blockers:
            selected_branch = candidate

    selected = None if selected_branch is None else selected_branch["best_terminal"]
    if universe.generator_manifest.generated_leaf_cardinality == 0:
        verdict = "BLOCKED_EMPTY_GENERATOR_MANIFEST"
    elif not finite_manifest_closed:
        verdict = "BLOCKED_GENERATOR_DOMAIN_NOT_CLOSED"
    elif dominance_blockers or (selected is None and improving_branches):
        verdict = "BLOCKED_CONTINUATION_DOMINANCE_UNPROVED"
    elif open_below_base_first_actions:
        verdict = "BLOCKED_FEASIBLE_TERMINAL_UPPER_UNAVAILABLE"
    elif selected is None:
        verdict = "SCOPED_FINITE_GENERATOR_FIXED_POINT"
    elif base.authority_mode in ("PRODUCTION_CONTEST_CPU", "PRODUCTION_CONTEST_CUDA"):
        verdict = "SELECT_FIRST_ACTION_FROM_STRICT_FINITE_HORIZON_DOMINANCE_THEN_REQUIRE_PAIRED_CONTEST_REPLAY"
    else:
        verdict = "ADVISORY_FINITE_HORIZON_FIRST_ACTION_NO_PRODUCTION_COMMIT"

    selected_terminal_id = None if selected is None else selected["terminal_leaf_id"]
    selected_first_action_id = None if selected_branch is None else selected_branch["first_action_id"]
    stale_local_marginals = (
        [] if selected is None else [leaf.terminal_leaf_id for leaf in universe.generator_manifest.terminal_leaves]
    )
    retained_nonselected_evidence = (
        []
        if selected is None
        else [
            leaf.terminal_leaf_id
            for leaf in universe.generator_manifest.terminal_leaves
            if leaf.terminal_leaf_id != selected_terminal_id
        ]
    )
    dynamic_target_crossed = bool(selected is not None and selected["inside_dynamic_target_sublevel"])
    output = {
        "schema": SCHEMA,
        "base": {
            "identity_sha256": base.identity_sha256,
            "epoch_id": base.epoch_id,
            "authority_mode": base.authority_mode,
            "axis": base.axis,
            "archive_sha256": base.archive_sha256,
            "archive_nbytes": base.archive_nbytes,
            "d_seg": base.d_seg,
            "d_pose": base.d_pose,
            "score": base.score,
            "dynamic_target_score": base.dynamic_target_score,
            "dynamic_pointer_artifact_sha256": base.dynamic_pointer_artifact_sha256,
            "decode_constraint_receipt_sha256": base.decode_constraints.receipt_sha256,
        },
        "action_universe": {
            "identity_sha256": universe.identity_sha256,
            "generator_manifest_identity_sha256": universe.generator_manifest.identity_sha256,
            "generated_leaf_merkle_root_sha256": (universe.generator_manifest.generated_leaf_merkle_root_sha256),
            "generated_leaf_cardinality": universe.generator_manifest.generated_leaf_cardinality,
            "finite_control_horizon": universe.generator_manifest.horizon,
            "maximum_interaction_order": (universe.generator_manifest.maximum_interaction_order),
            "universe_id": universe.universe_id,
            "epoch_id": universe.epoch_id,
            "enumeration_complete": universe.enumeration_complete,
            "cross_scale_scan_complete": universe.cross_scale_scan_complete,
            "blocking_coverage_families": blocking_coverage,
            "blocking_coverage_cells": [f"{family}:{scale}" for family, scale in blocking_coverage_cells],
            "finite_manifest_selection_closed": finite_manifest_closed,
            "global_selection_closed": False,
            "global_optimality_claim": False,
            "selection_scope": "exact finite generator manifest on one exact base",
            "required_families": sorted(universe.required_families),
            "required_scales": sorted(universe.required_scales),
            "endpoint_count": len(endpoint_rows),
            "hard_admissible_endpoint_count": len(hard_admissible),
            "verified_continuation_certificate_count": len(continuation_certificates),
            "caller_order_used": False,
            "base_stop_incumbent_score": base.score,
            "family_scale_projection_is_load_bearing": False,
            "caller_scalar_sha_branch_bounds_accepted": False,
        },
        "functional_quotient": {
            "identity_measure": ("evaluator_cells_x_continuation_equivalence_x_proof_dependencies"),
            "present_score_equivalence_alone_is_insufficient": True,
            "classes": quotient_classes,
            "preserved_future_distinctions": preserved_future_distinctions,
            "representative_count": len(quotient_rows),
        },
        "finite_horizon_value": {
            "horizon": universe.generator_manifest.horizon,
            "base_stop": base_stop,
            "first_action_branches": first_action_branches,
            "terminal_value_function": "min exact nonlinear terminal score over each first-action branch",
            "lower_bounds_verified": True,
            "feasible_upper_bounds_verified_when_present": True,
            "all_bounds_exact": all(
                branch["finite_terminal_value_exact"] or branch["all_generated_terminals_proved_hard_infeasible"]
                for branch in first_action_branches
            ),
            "strict_dominance_required": True,
            "dominance_blocking_first_action_ids": sorted(dominance_blockers),
            "open_below_base_without_improving_upper_first_action_ids": (open_below_base_first_actions),
            "equivalent_successor_tie_first_action_ids": sorted(equivalent_tie_first_actions),
        },
        "endpoint_measurements": endpoint_rows,
        "decision": {
            "verdict": verdict,
            "selected_action_id": selected_first_action_id,
            "selected_first_action_id": selected_first_action_id,
            "selected_terminal_leaf_id": selected_terminal_id,
            "selected_terminal_endpoint": selected,
            "one_action_maximum": True,
            "globally_lowest_exact_endpoint_selected": False,
            "strict_finite_horizon_dominance_proved": selected is not None,
            "base_stop_strictly_beaten": selected is not None,
            "independent_seg_pose_rate_thresholds_used": False,
            "nonlinear_whole_object_score_is_verdict_authority": True,
            "hard_decode_constraints_added_to_score": False,
            "search_state_transition_recommended": selected is not None,
            "production_commit_authorized": False,
            "pointer_promotion_authorized": False,
            "same_archive_contest_cpu_cuda_pair_owed": selected is not None,
            "dynamic_target_crossed": dynamic_target_crossed,
            "frontier_candidate_requires_same_archive_contest_replay": dynamic_target_crossed,
        },
        "receding_horizon_transition": {
            "stale_local_marginal_action_ids": stale_local_marginals,
            "retained_nonselected_exact_terminal_evidence_ids": (retained_nonselected_evidence),
            "nonselected_exact_branch_receipts_discarded": False,
            "all_unselected_marginals_invalid_after_commit": selected is not None,
            "next_epoch_required": selected is not None,
            "regenerate_complete_action_universe_from_selected_exact_base": selected is not None,
            "reuse_current_marginals_forbidden": True,
        },
        "fixed_point_scope": (
            {
                "scope": "only this exact base, enumerated families/scales, decoder constraints, and target snapshot",
                "family_level_negative_claimed": False,
            }
            if verdict == "SCOPED_FINITE_GENERATOR_FIXED_POINT"
            else None
        ),
        "triality": {
            "dsl": (
                "full-vocabulary finite generator manifest with typed action paths, support "
                "hyperedges, exact terminal leaves, and BASE_STOP"
            ),
            "dag": (
                "base -> generate atomic+joint tree -> rebuild terminal leaves -> public double "
                "decode -> exact terminal value -> strict first-action dominance -> commit one -> "
                "invalidate local derivatives -> regenerate"
            ),
            "equation": (
                "commit a only if U_H(a)<S(BASE_STOP) and U_H(a)<min_{b!=a}L_H(b); "
                "S=100*d_seg+sqrt(10*d_pose)+25*bytes/37545489"
            ),
        },
        "truth": {
            "score_claim": False,
            "pointer_mutation_performed": False,
            "archive_write_or_dispatch_performed": False,
            "decision_only": True,
        },
    }
    output["decision_receipt_sha256"] = _sha256(output)
    if json.loads(_canonical_json(output)) != output:
        raise TaskspaceRecedingHorizonError("decision canonical round trip changed output")
    return output


__all__ = [
    "ACTION_FAMILIES",
    "ACTION_SCALES",
    "BASE_STOP_ACTION_ID",
    "SCHEMA",
    "ActionFamilyScaleCoverageV1",
    "ActionSupportCellV1",
    "CompleteActionUniverseV1",
    "ContinuationCertificateEvidenceBundleV1",
    "ContinuationProofDependencyPayloadV1",
    "DecodeConstraintEvidenceV1",
    "GeneratedTerminalLeafV1",
    "GeneratorDomainManifestV1",
    "TaskspaceRecedingHorizonError",
    "WholeObjectActionEndpointV1",
    "WholeObjectBaseV1",
    "action_support_hyperedge_sha256",
    "select_receding_horizon_action_v1",
]
