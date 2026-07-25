# SPDX-License-Identifier: MIT
"""Fail-closed J11 opening-proposal decomposition custody audit.

J11 may only call a component ``pose-null`` or ``seg-null`` when the measured
scorer metric is joined to the exact receiver coordinate that is being
projected.  A metric on PoseNet output space and a metric on SegNet's rank-4
head are not, by themselves, receiver-coordinate Jacobians.

This module authenticates the proposed inputs and emits a typed refusal when
those actuator joins are absent.  It never estimates a missing Jacobian,
materializes a candidate, invokes a scorer, or changes the pure-priced
acceptance rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from tac.optimization.ddm_metric_custody_bundle import (
    ComponentId,
    load_metric_custody_bundle,
)

if TYPE_CHECKING:
    from tac.optimization.direct_description_joint_descent import (
        DirectDescriptionJointDescentTypedConfigV1,
    )

CONFIG_SCHEMA: Final = "ddm_j11_opening_proposal_decomposition_config.v1"
RECEIPT_SCHEMA: Final = "ddm_j11_opening_proposal_decomposition_refusal.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
SOURCE_ARCHIVE_SHA256: Final = "2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"
SOURCE_ARCHIVE_BYTES: Final = 138_813
SOURCE_BASELINE_D_SEG: Final = 0.06974277072482639
SOURCE_BASELINE_D_POSE: Final = 35.49982080959101
SEALED_OPENING_PROPOSALS: Final = (
    "worldsheet_joint_active_x_+1",
    "worldsheet_joint_active_x_-1",
    "worldsheet_joint_active_y_-1",
    "local_exact_gradient",
)
POSE_ACTUATOR_BLOCKER: Final = "POSE_RECEIVER_COORDINATE_JACOBIAN_AND_PROPOSAL_FOREIGN_KEY_ABSENT"
SEG_ACTUATOR_BLOCKER: Final = "SEG_RANK4_RECEIVER_COORDINATE_INNER_JACOBIAN_AND_PROPOSAL_FOREIGN_KEY_ABSENT"
RANGE_A_BLOCKER: Final = "RANGE_A_PROJECTOR_IS_RESIZE_GAUGE_CANONICALIZER_NOT_SEG_NULL_PROJECTOR"
PC1_REBASE_BLOCKER: Final = "PC1_ACTIVE_ZERO_HOME_IS_NOT_SOURCE_PRESERVING_AT_WJOINT_STEP50"


class J11ProposalDecompositionError(ValueError):
    """Malformed, stale, or falsely promoted J11 proposal-source custody."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise J11ProposalDecompositionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise J11ProposalDecompositionError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class BoundArtifactV1:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_payload(cls, value: Mapping[str, Any], *, label: str) -> BoundArtifactV1:
        if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
            raise J11ProposalDecompositionError(f"{label} binding keys differ")
        path = value["path"]
        byte_count = value["bytes"]
        digest = value["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise J11ProposalDecompositionError(f"{label} binding is malformed")
        return cls(path=path, bytes=byte_count, sha256=digest)

    def resolve(self, repository_root: Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else repository_root / path

    def read(self, repository_root: Path) -> bytes:
        path = self.resolve(repository_root)
        if not path.is_file() or path.is_symlink():
            raise J11ProposalDecompositionError(f"bound regular file is unavailable: {path}")
        payload = path.read_bytes()
        actual = _sha256(payload)
        if len(payload) != self.bytes or actual != self.sha256:
            raise J11ProposalDecompositionError(
                f"bound file custody differs: {path}: {len(payload)}/{actual} != {self.bytes}/{self.sha256}"
            )
        return payload

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class J11ProposalDecompositionConfigV1:
    config_path: Path
    lane_id: str
    run_id: str
    output_path: str
    source_artifacts: Mapping[str, BoundArtifactV1]

    @property
    def repository_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def from_path(cls, path: str | Path) -> J11ProposalDecompositionConfigV1:
        config_path = Path(path).resolve(strict=True)
        value = _read_json_bytes(config_path.read_bytes(), label="J11 typed config")
        required = {
            "schema",
            "lane_id",
            "run_id",
            "output_path",
            "source_artifacts",
            "pair_count",
            "verdict_batch",
            "source_baseline",
            "acceptance_rule",
            "research_only",
            "score_claim",
            "promotion_eligible",
            "pointer",
            "pointer_moved",
            "main_review_required",
        }
        if set(value) != required or value.get("schema") != CONFIG_SCHEMA:
            raise J11ProposalDecompositionError("J11 typed config keys/schema differ")
        fixed = {
            "pair_count": 600,
            "verdict_batch": 32,
            "acceptance_rule": "strict_realized_joint_delta_s_lt_zero",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "main_review_required": True,
        }
        drift = {key: (value.get(key), expected) for key, expected in fixed.items() if value.get(key) != expected}
        if drift:
            raise J11ProposalDecompositionError(f"J11 false-authority/execution contract differs: {drift}")
        baseline = value.get("source_baseline")
        if not isinstance(baseline, Mapping) or baseline != {
            "archive_bytes": SOURCE_ARCHIVE_BYTES,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "d_seg": SOURCE_BASELINE_D_SEG,
            "d_pose": SOURCE_BASELINE_D_POSE,
        }:
            raise J11ProposalDecompositionError("J11 source baseline differs")
        expected_artifacts = {
            "authority",
            "j10_ticket",
            "j10_full_run_receipt",
            "metric_bundle",
            "pc2_receipt",
            "range_a_projector",
            "source_archive",
            *(f"proposal_{proposal_id}" for proposal_id in SEALED_OPENING_PROPOSALS),
        }
        source_artifacts = value.get("source_artifacts")
        if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != expected_artifacts:
            raise J11ProposalDecompositionError("J11 source-artifact set differs")
        lane_id = value.get("lane_id")
        run_id = value.get("run_id")
        output_path = value.get("output_path")
        if not all(isinstance(item, str) and item for item in (lane_id, run_id, output_path)):
            raise J11ProposalDecompositionError("J11 lane/run/output identity differs")
        return cls(
            config_path=config_path,
            lane_id=str(lane_id),
            run_id=str(run_id),
            output_path=str(output_path),
            source_artifacts={
                name: BoundArtifactV1.from_payload(binding, label=name) for name, binding in source_artifacts.items()
            },
        )

    def validate_all_bindings(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                **binding.to_payload(),
                "resolved_path": str(binding.resolve(self.repository_root)),
                "validated": bool(binding.read(self.repository_root)),
            }
            for name, binding in sorted(self.source_artifacts.items())
        }


def derive_authority_blockers(
    *,
    pose_data: Mapping[str, Any],
    seg_data: Mapping[str, Any],
    range_a_source: str,
    pc2_receipt: Mapping[str, Any],
    source_baseline: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Re-derive which joins are missing without estimating their values."""

    rows = pose_data.get("rows")
    if (
        pose_data.get("schema") != "ddm_pose_metric_custody.v1"
        or pose_data.get("metric_surface") != "EXACT_POSENET_OUTPUT_MSE_QUADRATIC"
        or pose_data.get("output_dimension") != 6
        or pose_data.get("pair_count") != 600
        or pose_data.get("scorer_batch_size") != 32
        or not isinstance(rows, list)
        or len(rows) != 600
    ):
        raise J11ProposalDecompositionError("Pose metric is not the strict n600 output-space bundle")
    forbidden_pose_promotions = {
        "receiver_coordinate_jacobian",
        "receiver_coordinate_ids",
        "proposal_foreign_keys",
        "j5_coordinate_jacobian",
    }
    pose_fields_present = sorted(forbidden_pose_promotions.intersection(pose_data))
    if pose_fields_present:
        raise J11ProposalDecompositionError(f"Pose bundle acquired unreviewed actuator fields: {pose_fields_present}")
    if (
        seg_data.get("schema") != "ddm_seg_metric_custody.direct_scorer_intrinsic.v2"
        or seg_data.get("metric_mode") != "DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT"
        or seg_data.get("head_rank") != 4
        or seg_data.get("pair_count") != 600
        or seg_data.get("scorer_batch_size") != 32
    ):
        raise J11ProposalDecompositionError("Seg metric is not the strict direct no-actuator bundle")
    direct_blocks = seg_data.get("direct_blocks")
    if not isinstance(direct_blocks, list) or any(
        row.get("secant_status") != "NOT_APPLICABLE_DIRECT_SCORER_INTRINSIC_NO_ACTUATOR"
        for row in direct_blocks
        if isinstance(row, Mapping)
    ):
        raise J11ProposalDecompositionError("Seg direct-block actuator boundary differs")
    if (
        "Project camera-space ``frames`` onto ``range(A)``" not in range_a_source
        or "A(PX)=A(X)" not in range_a_source
        or "SegNet-null" in range_a_source
    ):
        raise J11ProposalDecompositionError("#580 projector source no longer proves the range(A) invariant")
    if (
        pc2_receipt.get("schema") != "ddm_pc2_pose_descent_smoke_receipt.v1"
        or pc2_receipt.get("parent", {}).get("archive_sha256") != SOURCE_ARCHIVE_SHA256
        or pc2_receipt.get("parent", {}).get("archive_bytes") != SOURCE_ARCHIVE_BYTES
        or pc2_receipt.get("verdict") != "PC1_DESCENT_MEASURED_NET_JOINT_NEGATIVE"
    ):
        raise J11ProposalDecompositionError("PC2 receipt does not bind the J11 source")
    exact_zero = pc2_receipt.get("exact_verdicts", {}).get("0")
    if not isinstance(exact_zero, Mapping):
        raise J11ProposalDecompositionError("PC2 exact active-zero verdict is absent")
    rebase = {
        "source_archive_bytes": int(source_baseline["archive_bytes"]),
        "active_zero_archive_bytes": int(exact_zero["archive_bytes"]),
        "delta_archive_bytes": int(exact_zero["archive_bytes"]) - int(source_baseline["archive_bytes"]),
        "source_d_seg": float(source_baseline["d_seg"]),
        "active_zero_d_seg": float(exact_zero["d_seg"]),
        "delta_d_seg": float(exact_zero["d_seg"]) - float(source_baseline["d_seg"]),
        "source_d_pose": float(source_baseline["d_pose"]),
        "active_zero_d_pose": float(exact_zero["d_pose"]),
        "delta_d_pose": float(exact_zero["d_pose"]) - float(source_baseline["d_pose"]),
        "source_preserving": (
            int(exact_zero["archive_bytes"]) == int(source_baseline["archive_bytes"])
            and math.isclose(
                float(exact_zero["d_seg"]),
                float(source_baseline["d_seg"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                float(exact_zero["d_pose"]),
                float(source_baseline["d_pose"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ),
        "epistemic_status": "DERIVED_FROM_TWO_EXACT_N600_MEASURED_RECEIPTS",
    }
    blockers = [POSE_ACTUATOR_BLOCKER, SEG_ACTUATOR_BLOCKER, RANGE_A_BLOCKER]
    if not rebase["source_preserving"]:
        blockers.append(PC1_REBASE_BLOCKER)
    return tuple(blockers), rebase


def _original_proposal_rows(
    *,
    baseline: Mapping[str, Any],
    proposal_verdicts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from tac.optimization.pure_priced_realized_objective import (
        RealizedObjectiveState,
        pure_priced_realized_delta,
    )

    reference = RealizedObjectiveState(
        d_seg=float(baseline["d_seg"]),
        d_pose=float(baseline["d_pose"]),
        archive_bytes=int(baseline["archive_bytes"]),
    )
    rows: list[dict[str, Any]] = []
    if set(proposal_verdicts) != set(SEALED_OPENING_PROPOSALS):
        raise J11ProposalDecompositionError("J10 proposal verdict set differs")
    for proposal_id in SEALED_OPENING_PROPOSALS:
        verdict = proposal_verdicts[proposal_id]
        if (
            verdict.get("schema") != "ddm_joint_descent_chunked_stage_verdict.v1"
            or verdict.get("proposal_source") != proposal_id
            or verdict.get("proposal_multiplier") != 32.0
            or verdict.get("warm_start_realized_admitted") is not False
            or verdict.get("score_claim") is not False
        ):
            raise J11ProposalDecompositionError(f"J10 proposal verdict differs: {proposal_id}")
        recomputed = pure_priced_realized_delta(
            reference,
            RealizedObjectiveState(
                d_seg=float(verdict["d_seg"]),
                d_pose=float(verdict["d_pose"]),
                archive_bytes=int(verdict["archive_bytes"]),
            ),
        )
        recorded = verdict.get("pure_priced_delta")
        if not isinstance(recorded, Mapping):
            raise J11ProposalDecompositionError(f"J10 proposal lacks pure pricing: {proposal_id}")
        for field in ("seg_term", "pose_term", "rate_term", "joint_delta"):
            if not math.isclose(
                float(recorded[field]),
                float(getattr(recomputed, field)),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise J11ProposalDecompositionError(
                    f"J10 proposal pure-price arithmetic differs: {proposal_id}/{field}"
                )
        rows.append(
            {
                "proposal_id": proposal_id,
                "archive_sha256": verdict["archive_sha256"],
                "archive_bytes": int(verdict["archive_bytes"]),
                "d_seg": float(verdict["d_seg"]),
                "d_pose": float(verdict["d_pose"]),
                "seg_term": recomputed.seg_term,
                "pose_term": recomputed.pose_term,
                "rate_term": recomputed.rate_term,
                "joint_delta": recomputed.joint_delta,
                "accepted": recomputed.accepted,
                "epistemic_status": "MEASURED_EXACT_N600_BATCH32_RECOMPUTED_UNCHANGED_RULE",
            }
        )
    return rows


def blocked_component_tables(
    proposal_ids: Sequence[str],
    *,
    blockers: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return explicit NULL-priced rows; never coerce an absent candidate to zero."""

    singles: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for proposal_id in proposal_ids:
        singles.extend(
            [
                {
                    "component_id": f"{proposal_id}::pose_null_seg",
                    "source_proposal_id": proposal_id,
                    "component_kind": "pose-null_seg",
                    "materialized": False,
                    "receiver_parseback_exact": None,
                    "d_seg": None,
                    "d_pose": None,
                    "archive_bytes": None,
                    "joint_delta": None,
                    "n600_evidence": False,
                    "blockers": list(blockers),
                },
                {
                    "component_id": f"{proposal_id}::seg_null_pose",
                    "source_proposal_id": proposal_id,
                    "component_kind": "seg-null_pose",
                    "materialized": False,
                    "receiver_parseback_exact": None,
                    "d_seg": None,
                    "d_pose": None,
                    "archive_bytes": None,
                    "joint_delta": None,
                    "n600_evidence": False,
                    "blockers": list(blockers),
                },
            ]
        )
        pairs.append(
            {
                "composite_id": f"{proposal_id}::pose_null_seg+pc1_pose_coordinate",
                "source_proposal_id": proposal_id,
                "materialized": False,
                "receiver_parseback_exact": None,
                "d_seg": None,
                "d_pose": None,
                "archive_bytes": None,
                "joint_delta": None,
                "n600_evidence": False,
                "blockers": list(blockers),
            }
        )
    return singles, pairs


def build_refusal_receipt(
    *,
    typed_descent: DirectDescriptionJointDescentTypedConfigV1,
    audit_config_path: str | Path,
) -> dict[str, Any]:
    """Authenticate all J11 inputs and emit the scoped precondition refusal."""

    config = J11ProposalDecompositionConfigV1.from_path(audit_config_path)
    bindings = config.validate_all_bindings()
    root = config.repository_root
    bound_ticket = config.source_artifacts["j10_ticket"].resolve(root).resolve()
    if Path(typed_descent.ticket_path).resolve() != bound_ticket:
        raise J11ProposalDecompositionError("typed J10 consumer and audit ticket path differ")
    if (
        typed_descent.source_archive_sha256 != SOURCE_ARCHIVE_SHA256
        or typed_descent.source_archive_bytes != SOURCE_ARCHIVE_BYTES
        or typed_descent.source_baseline_d_seg != SOURCE_BASELINE_D_SEG
        or typed_descent.source_baseline_d_pose != SOURCE_BASELINE_D_POSE
    ):
        raise J11ProposalDecompositionError("typed J10 source baseline differs")
    reform = typed_descent.full_run_schedule.warm_start_reform if typed_descent.full_run_schedule else None
    if reform is None or reform.opening_candidate_ids != SEALED_OPENING_PROPOSALS:
        raise J11ProposalDecompositionError("typed J10 sealed opening proposal order differs")
    j10 = _read_json_bytes(
        config.source_artifacts["j10_full_run_receipt"].read(root),
        label="J10 full-run receipt",
    )
    baseline = j10.get("baseline_verdict")
    if (
        j10.get("schema") != "ddm_joint_descent_full_run_receipt.v1"
        or j10.get("campaign_blocker") != "BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER"
        or not isinstance(baseline, Mapping)
        or baseline.get("archive_sha256") != SOURCE_ARCHIVE_SHA256
        or baseline.get("archive_bytes") != SOURCE_ARCHIVE_BYTES
        or baseline.get("d_seg") != SOURCE_BASELINE_D_SEG
        or baseline.get("d_pose") != SOURCE_BASELINE_D_POSE
    ):
        raise J11ProposalDecompositionError("J10 exact opening blocker custody differs")
    proposal_verdicts = {
        proposal_id: _read_json_bytes(
            config.source_artifacts[f"proposal_{proposal_id}"].read(root),
            label=f"J10 proposal {proposal_id}",
        )
        for proposal_id in SEALED_OPENING_PROPOSALS
    }
    original_rows = _original_proposal_rows(
        baseline=baseline,
        proposal_verdicts=proposal_verdicts,
    )
    bundle_path = config.source_artifacts["metric_bundle"].resolve(root)
    bundle = load_metric_custody_bundle(
        bundle_path,
        repository_root=root,
        require_complete=True,
    )
    pose_artifact = bundle.components[ComponentId.POSE_METRIC].data_artifact
    seg_artifact = bundle.components[ComponentId.SEG_METRIC].data_artifact
    if pose_artifact is None or seg_artifact is None:
        raise J11ProposalDecompositionError("COMPLETE bundle omitted metric data")
    pose_data = _read_json_bytes(
        pose_artifact.revalidate(repository_root=root).read_bytes(),
        label="MS4 Pose data",
    )
    seg_data = _read_json_bytes(
        seg_artifact.revalidate(repository_root=root).read_bytes(),
        label="MS4 Seg data",
    )
    pc2 = _read_json_bytes(
        config.source_artifacts["pc2_receipt"].read(root),
        label="PC2 receipt",
    )
    range_a_source = config.source_artifacts["range_a_projector"].read(root).decode("utf-8")
    blockers, pc1_rebase = derive_authority_blockers(
        pose_data=pose_data,
        seg_data=seg_data,
        range_a_source=range_a_source,
        pc2_receipt=pc2,
        source_baseline=baseline,
    )
    singles, pairs = blocked_component_tables(
        SEALED_OPENING_PROPOSALS,
        blockers=blockers,
    )
    aggregate_slope = next(
        (row for row in pc2.get("slope_rows", ()) if isinstance(row, Mapping) and row.get("aggregate") is True),
        None,
    )
    if not isinstance(aggregate_slope, Mapping) or not math.isclose(
        float(aggregate_slope.get("observed_pose_to_seg_regression_ratio", math.nan)),
        14.023295441931698,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise J11ProposalDecompositionError("PC2 aggregate slope custody differs")
    return {
        "schema": RECEIPT_SCHEMA,
        "lane_id": config.lane_id,
        "run_id": config.run_id,
        "typed_descent_config_hash": typed_descent.typed_config_hash(),
        "source_bindings": bindings,
        "source_baseline": {
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "archive_bytes": SOURCE_ARCHIVE_BYTES,
            "d_seg": SOURCE_BASELINE_D_SEG,
            "d_pose": SOURCE_BASELINE_D_POSE,
            "evidence_axis": EVIDENCE_AXIS,
        },
        "unchanged_acceptance_rule": {
            "function": "tac.optimization.pure_priced_realized_objective.pure_priced_realized_delta",
            "admit_iff": "realized_joint_delta_s_lt_zero",
            "reweighted": False,
            "component_gate_added": False,
        },
        "original_sealed_proposals": original_rows,
        "metric_bundle": {
            "bundle_id": bundle.bundle_id,
            "status": bundle.status.value,
            "loader_required_complete": True,
            "pose_metric_surface": pose_data["metric_surface"],
            "pose_coordinate_domain": "POSENET_OUTPUT_6_ONLY",
            "pose_receiver_coordinate_jacobian_present": False,
            "seg_metric_mode": seg_data["metric_mode"],
            "seg_coordinate_domain": seg_data["head_chart"]["coordinate_domain"],
            "seg_receiver_coordinate_inner_jacobian_present": False,
        },
        "range_a_projector": {
            "projector": "P_range(A)",
            "invariant": "A(PX)=A(X)",
            "removes": "ker(A)_resize_invisible_energy",
            "seg_null_authority": False,
        },
        "pc1_composition_rebase": {
            **pc1_rebase,
            "measured_ratio_in_non_preserving_active_home": float(
                aggregate_slope["observed_pose_to_seg_regression_ratio"]
            ),
            "ratio_transfer_authorized": False,
        },
        "authority_blockers": list(blockers),
        "component_pricing_table": singles,
        "composed_pair_pricing_table": pairs,
        "candidate_counts": {
            "sealed_originals": len(SEALED_OPENING_PROPOSALS),
            "required_single_components": 2 * len(SEALED_OPENING_PROPOSALS),
            "materialized_single_components": 0,
            "required_composed_pairs": len(SEALED_OPENING_PROPOSALS),
            "materialized_composed_pairs": 0,
        },
        "bounded_smoke": {
            "ran": False,
            "n600_scorer_invoked": False,
            "live_ema_dual_verdict_rows": 0,
            "reason": "NO_LAWFUL_RECEIVER_REALIZABLE_COMPONENT_OR_COMPOSITE_PROPOSAL",
            "campaign_launched": False,
        },
        "reseal_state": {
            "tools/reseal_ddm_j7_366_ticket.py": "NOT_RUN",
            "reason": "NO_ADMITTED_OPENING_STEP; READY/FIRE reseal would be false authority",
            "ready_to_fire_under_standing_go": False,
            "fire_authority": "MAIN_ONLY_AFTER_FUTURE_ADMITTED_STEP_AND_RESEAL",
        },
        "verdict": "BLOCKED_J11_PROPOSAL_DECOMPOSITION_CUSTODY_PRECONDITION",
        "verdict_scope": (
            "PRECONDITION/APPARATUS: the sealed J10 proposal decomposition is not "
            "materializable from the named bundle/projector/PC2 inputs. This is not a "
            "FORMULATION negative on a correctly joined decomposition."
        ),
        "named_residual_obstruction": (
            "receiver-coordinate scorer Jacobians and proposal foreign keys are absent; "
            "PC1 active-zero composition is not source-preserving"
        ),
        "next_reformulation": (
            "measure SHA-bound per-proposal J_pose and rank4-inner-J on exact J5 receiver "
            "secants, land a source-preserving PC1 zero-home adapter, then project, "
            "integer-realize, parse back, and exact-n600 price all singles and composites"
        ),
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_review_required": True,
    }


__all__ = [
    "CONFIG_SCHEMA",
    "PC1_REBASE_BLOCKER",
    "POSE_ACTUATOR_BLOCKER",
    "RANGE_A_BLOCKER",
    "RECEIPT_SCHEMA",
    "SEALED_OPENING_PROPOSALS",
    "SEG_ACTUATOR_BLOCKER",
    "BoundArtifactV1",
    "J11ProposalDecompositionConfigV1",
    "J11ProposalDecompositionError",
    "blocked_component_tables",
    "build_refusal_receipt",
    "derive_authority_blockers",
]
