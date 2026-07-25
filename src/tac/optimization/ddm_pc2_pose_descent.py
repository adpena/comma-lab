# SPDX-License-Identifier: MIT
"""Typed laws for the bounded PC1 solved-plane pose-descent smoke.

The smoke deliberately uses the exact score-domain objective

    100 d_seg + sqrt(10 d_pose) + 25 B / 37_545_489

with a static pose weight of one.  It therefore must not also enable the
``PoseMarginalWeightLaw``: that law is only correct for a raw-d_pose loss and
would square the contest marginal when composed with the score-domain term.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_SCHEMA = "ddm_pc2_pose_descent_smoke_config.v1"
CHECKPOINT_SCHEMA = "ddm_pc2_pose_descent_checkpoint.v1"
VERDICT_SCHEMA = "ddm_pc2_pose_descent_n600_verdict.v1"
RECEIPT_SCHEMA = "ddm_pc2_pose_descent_smoke_receipt.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
SOURCE_BYTES = 37_545_489
CRITICAL_RATIO = 4.1215446777965665
TARGET_D_POSE = 2.94e-5
POSE_AXES = ("tx", "ty", "tz", "rx", "ry", "rz")


class PC2PoseDescentError(ValueError):
    """Raised when typed custody or bounded-smoke semantics differ."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class BoundArtifactV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def resolve(self, repo_root: Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else repo_root / path

    def validate_bytes(self, repo_root: Path) -> bytes:
        path = self.resolve(repo_root)
        payload = path.read_bytes()
        if len(payload) != self.bytes or sha256_bytes(payload) != self.sha256:
            raise PC2PoseDescentError(f"bound artifact custody differs: {path}")
        return payload


class PC2PoseDescentConfigV1(BaseModel):
    """Fail-closed typed config for one local advisory smoke."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["ddm_pc2_pose_descent_smoke_config.v1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str
    lane_id: Literal["ddm_pc2_pose_descent_smoke"]
    delegation_checkpoint_key: Literal["codex_delegate:ddm_pc2_pose_descent_smoke:20260725T121448Z"]
    own_python: str
    output_root: str
    source_artifacts: dict[str, BoundArtifactV1]
    upstream_root: str
    pair_count: Literal[600] = 600
    train_batch: Literal[4] = 4
    verdict_batch: Literal[32] = 32
    torch_threads: Literal[4] = 4
    target_accepted_steps: int = Field(ge=10, le=20)
    exact_verdict_steps: tuple[int, ...]
    proposal_quanta: tuple[int, ...]
    maximum_candidate_evaluations: int = Field(gt=0)
    seed: Literal[0] = 0
    score_domain_loss: bool
    pose_marginal_weight_law: bool
    pose_objective_weight: float
    critical_ratio: float
    target_d_pose: float
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer: Literal["0.1910828242 [contest-CPU]"] = POINTER
    main_review_required: Literal[True] = True

    @model_validator(mode="after")
    def _validate_semantics(self) -> PC2PoseDescentConfigV1:
        expected_artifacts = {
            "authority",
            "j7_precedent",
            "menu1_config",
            "pc1_admission",
            "target_cache",
            "w_joint_step50",
            "ws4_arbitration",
        }
        if set(self.source_artifacts) != expected_artifacts:
            raise ValueError("PC2 source-artifact set differs")
        if self.score_domain_loss == self.pose_marginal_weight_law:
            raise ValueError("exactly one of score_domain_loss and PoseMarginalWeightLaw must be active")
        if not self.score_domain_loss or self.pose_marginal_weight_law or self.pose_objective_weight != 1.0:
            raise ValueError("PC2 binds score-domain loss with static w_pose=1; marginal-law composition is refused")
        if not math.isclose(self.critical_ratio, CRITICAL_RATIO, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("PC2 critical ratio differs from the registered WS1 law")
        if not math.isclose(self.target_d_pose, TARGET_D_POSE, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("PC2 target d_pose differs from the optimal-start card")
        if (
            not self.exact_verdict_steps
            or self.exact_verdict_steps[0] != 0
            or self.exact_verdict_steps[-1] != self.target_accepted_steps
            or tuple(sorted(set(self.exact_verdict_steps))) != self.exact_verdict_steps
        ):
            raise ValueError("exact verdict steps must be sorted, unique, and include 0 and the horizon")
        if (
            not self.proposal_quanta
            or any(isinstance(value, bool) or value <= 0 for value in self.proposal_quanta)
            or tuple(sorted(self.proposal_quanta, reverse=True)) != self.proposal_quanta
        ):
            raise ValueError("proposal quanta must be positive and coarse-to-fine")
        return self

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def typed_hash(self) -> str:
        return sha256_bytes(canonical_bytes(self.to_payload()))

    def validate_all_bindings(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "path": artifact.path,
                "bytes": len(artifact.validate_bytes(self.repo_root)),
                "sha256": artifact.sha256,
            }
            for name, artifact in sorted(self.source_artifacts.items())
        }

    @classmethod
    def from_path(cls, path: str | Path) -> PC2PoseDescentConfigV1:
        source = Path(path)
        payload = source.read_bytes()
        try:
            result = cls.model_validate_json(payload, strict=True)
        except Exception as exc:
            raise PC2PoseDescentError("typed PC2 config validation failed") from exc
        if payload != canonical_bytes(result.to_payload()):
            raise PC2PoseDescentError("typed PC2 config is not canonical JSON")
        return result


def bit_reversal_knot_order(knot_count: int) -> tuple[int, ...]:
    """Cover a power-of-two knot horizon coarse-to-fine without a tuned order."""

    if isinstance(knot_count, bool) or knot_count < 2 or knot_count & (knot_count - 1):
        raise PC2PoseDescentError("knot count must be a power of two")
    width = int(math.log2(knot_count))
    return tuple(int(f"{index:0{width}b}"[::-1], 2) for index in range(knot_count))


def four_pair_batch_for_knot(
    knot_id: int,
    *,
    knot_count: int = 32,
    pair_count: int = 600,
) -> tuple[int, int, int, int]:
    """Choose four exact receiver pairs centered on one spline knot."""

    if not 0 <= knot_id < knot_count or pair_count < 4:
        raise PC2PoseDescentError("knot or pair geometry differs")
    center = round(knot_id * (pair_count - 1) / (knot_count - 1))
    start = min(max(center - 1, 0), pair_count - 4)
    return tuple(range(start, start + 4))  # type: ignore[return-value]


def score_domain_action(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    values = (float(d_seg), float(d_pose))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise PC2PoseDescentError("score-domain distortions must be finite and nonnegative")
    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise PC2PoseDescentError("archive bytes must be a positive integer")
    return 100.0 * values[0] + math.sqrt(10.0 * values[1]) + 25.0 * archive_bytes / SOURCE_BYTES


def select_realized_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1.0e-12,
) -> Mapping[str, Any] | None:
    """Select a strict pose-and-joint improving exact mini-batch proposal."""

    admissible = [
        row
        for row in rows
        if float(row["pose_delta"]) < -tolerance
        and float(row["joint_delta"]) < -tolerance
        and bool(row["receiver_visible"])
    ]
    if not admissible:
        return None
    return min(
        admissible,
        key=lambda row: (
            float(row["joint_delta"]),
            float(row["pose_delta"]),
            float(row["seg_delta"]),
            int(row["archive_bytes"]),
            str(row["coordinate_id"]),
        ),
    )


def realized_slope_row(
    *,
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    accepted_steps: int,
    critical_ratio: float = CRITICAL_RATIO,
) -> dict[str, Any]:
    """Measure score-term slopes and the pose/Seg-collateral ratio."""

    if accepted_steps <= 0:
        raise PC2PoseDescentError("slope window needs a positive accepted-step count")
    start_seg = 100.0 * float(start["d_seg"])
    end_seg = 100.0 * float(end["d_seg"])
    start_pose = math.sqrt(10.0 * float(start["d_pose"]))
    end_pose = math.sqrt(10.0 * float(end["d_pose"]))
    seg_delta = (end_seg - start_seg) / accepted_steps
    pose_progress = (start_pose - end_pose) / accepted_steps
    seg_regression = max(seg_delta, 0.0)
    observed_ratio = (
        math.inf if pose_progress > 0.0 and seg_regression <= 1.0e-15 else pose_progress / max(seg_regression, 1.0e-15)
    )
    net_joint_delta = float(end["advisory_action"]) - float(start["advisory_action"])
    return {
        "accepted_steps": accepted_steps,
        "d_pose_delta_per_step": (float(end["d_pose"]) - float(start["d_pose"])) / accepted_steps,
        "d_seg_delta_per_step": (float(end["d_seg"]) - float(start["d_seg"])) / accepted_steps,
        "pose_term_progress_per_step": pose_progress,
        "seg_term_delta_per_step": seg_delta,
        "seg_regression_per_step": seg_regression,
        "observed_pose_to_seg_regression_ratio": observed_ratio,
        "critical_ratio": critical_ratio,
        "ratio_clears_critical": observed_ratio >= critical_ratio,
        "net_joint_delta": net_joint_delta,
        "net_joint_delta_per_step": net_joint_delta / accepted_steps,
    }


def constant_slope_horizon(
    *,
    start_d_pose: float,
    end_d_pose: float,
    accepted_steps: int,
    target_d_pose: float = TARGET_D_POSE,
) -> dict[str, Any]:
    """Honest linear extrapolation of the measured d_pose window."""

    progress = (float(start_d_pose) - float(end_d_pose)) / accepted_steps
    if progress <= 0.0:
        additional_steps = math.inf
        total_steps = math.inf
    else:
        additional_steps = max(
            (float(end_d_pose) - float(target_d_pose)) / progress,
            0.0,
        )
        total_steps = accepted_steps + additional_steps
    return {
        "method": "DERIVED_CONSTANT_REALIZED_DPOSE_SLOPE_OUTSIDE_WINDOW",
        "measured_d_pose_progress_per_step": progress,
        "target_d_pose": target_d_pose,
        "additional_steps_from_window_end": additional_steps,
        "total_steps_from_window_start": total_steps,
        "warning": (
            "This is a constant-slope extrapolation, not a convergence law; "
            "quantization, curvature, and photometric saturation are unmodeled."
        ),
    }


def fork_verdict(*, start: Mapping[str, Any], end: Mapping[str, Any]) -> tuple[str, str]:
    pose_descends = float(end["d_pose"]) < float(start["d_pose"])
    joint_negative = float(end["advisory_action"]) < float(start["advisory_action"])
    if pose_descends and joint_negative:
        return (
            "PC1_DESCENT_MEASURED_NET_JOINT_NEGATIVE",
            "INSTANCE: bounded PC1 score-domain coordinate descent from exact ws4 W_joint-step50",
        )
    if not pose_descends:
        return (
            "PC1_SOLVED_PLANE_POSE_ILLEGIBLE_FORMULATION_STOP",
            "FORMULATION: PC1 one-depth ground-plus-Movable smooth-xi receiver under this bounded exact-secant smoke",
        )
    return (
        "PC1_POSE_DESCENDS_BUT_JOINT_NOT_NEGATIVE_FORMULATION_STOP",
        "FORMULATION: PC1 one-depth ground-plus-Movable smooth-xi receiver under this bounded exact-secant smoke",
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "CRITICAL_RATIO",
    "EVIDENCE_AXIS",
    "POINTER",
    "POSE_AXES",
    "RECEIPT_SCHEMA",
    "TARGET_D_POSE",
    "VERDICT_SCHEMA",
    "PC2PoseDescentConfigV1",
    "PC2PoseDescentError",
    "bit_reversal_knot_order",
    "canonical_bytes",
    "constant_slope_horizon",
    "fork_verdict",
    "four_pair_batch_for_knot",
    "realized_slope_row",
    "score_domain_action",
    "select_realized_candidate",
    "sha256_bytes",
]
