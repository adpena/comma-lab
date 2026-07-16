# SPDX-License-Identifier: MIT
"""Fail-closed transfer law for the literal-curvelet equal-ZIP A/B.

This law does not contain a score row.  It turns a future, fully custodied
receipt into an instance-scoped verdict while keeping the curvelet family open.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

EQUATION_ID = "curvelet_equal_archive_transfer_v1"
RECEIPT_SCHEMA = "curvelet_equal_archive_transfer.v1"
CONTROL_FAMILY = "legacy_fourier_ab_control"
TREATMENT_FAMILY = "literal_polar_curvelet"
AUTHORITY_AXES = frozenset({"contest-CPU", "contest-CUDA"})


class CurveletTransferReceiptError(ValueError):
    """A receipt cannot authorize an equal-archive transfer verdict."""


@dataclass(frozen=True)
class CurveletTransferVerdict:
    status: str
    verdict_scope: str
    family_verdict: str
    authority_axis: str
    equal_archive_bytes: int
    control_d_seg: float
    treatment_d_seg: float
    delta_d_seg_treatment_minus_control: float
    control_d_pose: float
    treatment_d_pose: float
    delta_pose_score_term: float
    delta_nonrate_score: float
    dseg_transfer: bool
    pose_nonregression: bool
    pointer_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CurveletTransferReceiptError(f"{name} must be a mapping")
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise CurveletTransferReceiptError(f"{name} must be a finite nonnegative number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CurveletTransferReceiptError(
            f"{name} must be a finite nonnegative number"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise CurveletTransferReceiptError(f"{name} must be a finite nonnegative number")
    return numeric


def _sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CurveletTransferReceiptError(f"{name} must be a lowercase SHA-256")
    return digest


def _measurement(
    row: Any,
    *,
    name: str,
    expected_family: str,
    expected_archive_sha256: str,
    expected_archive_bytes: int,
    expected_output_tree_sha256: str,
) -> tuple[Mapping[str, Any], float, float]:
    item = _mapping(row, name=name)
    if item.get("family") != expected_family:
        raise CurveletTransferReceiptError(f"{name}.family does not identify the preregistered arm")
    if item.get("archive_sha256") != expected_archive_sha256:
        raise CurveletTransferReceiptError(f"{name}.archive_sha256 is not bound to the matched ZIP")
    if item.get("archive_bytes") != expected_archive_bytes:
        raise CurveletTransferReceiptError(f"{name}.archive_bytes is not the equal-ZIP byte count")
    required = {
        "n_pairs": 600,
        "n_samples": 600,
        "scorer_batch_size": 32,
        "through_r": True,
        "official_evaluator": True,
        "parse_back": True,
    }
    drift = {key: (item.get(key), value) for key, value in required.items() if item.get(key) != value}
    if drift:
        raise CurveletTransferReceiptError(f"{name} full-n600 evaluator custody drift: {drift}")
    if item.get("output_tree_sha256") != expected_output_tree_sha256:
        raise CurveletTransferReceiptError(f"{name}.output_tree_sha256 lost inflated-frame custody")
    for field in (
        "upstream_snapshot_sha256",
        "runtime_sha256",
        "checkpoint_sha256",
        "evaluate_report_sha256",
        "segnet_weights_sha256",
        "posenet_weights_sha256",
    ):
        _sha256(item.get(field), name=f"{name}.{field}")
    git_sha = str(item.get("git_sha", ""))
    if len(git_sha) not in {40, 64} or any(char not in "0123456789abcdef" for char in git_sha):
        raise CurveletTransferReceiptError(f"{name}.git_sha must be a 40- or 64-hex revision")
    for field in ("measurement_utc", "hardware_substrate", "torch_version", "device"):
        if not isinstance(item.get(field), str) or not str(item[field]).strip():
            raise CurveletTransferReceiptError(f"{name}.{field} must be a nonempty custody string")
    if not str(item["measurement_utc"]).endswith("Z"):
        raise CurveletTransferReceiptError(f"{name}.measurement_utc must be UTC Z-form")
    argv = item.get("evaluator_argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(token, str) and token for token in argv)
        or not any(token.endswith("upstream/evaluate.py") for token in argv)
    ):
        raise CurveletTransferReceiptError(
            f"{name}.evaluator_argv must be a nonempty upstream/evaluate.py argv"
        )
    seed = item.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CurveletTransferReceiptError(f"{name}.seed must be a nonnegative integer")
    d_seg = _finite_nonnegative(item.get("d_seg"), name=f"{name}.d_seg")
    d_pose = _finite_nonnegative(item.get("d_pose"), name=f"{name}.d_pose")
    return item, d_seg, d_pose


def evaluate_curvelet_equal_archive_transfer(
    receipt: Mapping[str, Any],
) -> CurveletTransferVerdict:
    r"""Evaluate the preregistered equal-rate transfer law.

    With matched counted bytes, the rate term cancels exactly:

    ``Delta S = 100 Delta d_seg + sqrt(10 d_pose,t) - sqrt(10 d_pose,c)``.

    The result is deliberately INSTANCE-scoped.  A loss cannot become a
    curvelet-family NO-GO, and even a win does not move the score pointer.
    """

    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("lawref") != EQUATION_ID:
        raise CurveletTransferReceiptError("transfer receipt schema/LawRef mismatch")
    if receipt.get("equal_budget_receipt_verified") is not True:
        raise CurveletTransferReceiptError("equal-budget archive receipt was not re-derived")
    if receipt.get("output_trees_preserved") is not True:
        raise CurveletTransferReceiptError("matched ZIP inflate outputs were not byte-identical")

    budget = _mapping(receipt.get("equal_budget"), name="equal_budget")
    if budget.get("equal_archive_bytes") is not True:
        raise CurveletTransferReceiptError("equal-budget receipt does not assert exact equality")
    target_bytes = budget.get("target_archive_bytes")
    if isinstance(target_bytes, bool) or not isinstance(target_bytes, int) or target_bytes <= 0:
        raise CurveletTransferReceiptError("equal-budget target_archive_bytes must be positive")
    control_budget = _mapping(budget.get("left"), name="equal_budget.left")
    treatment_budget = _mapping(budget.get("right"), name="equal_budget.right")
    if (
        control_budget.get("matched_archive_bytes") != target_bytes
        or treatment_budget.get("matched_archive_bytes") != target_bytes
    ):
        raise CurveletTransferReceiptError("equal-budget arm byte counts differ")
    tree_custody = _mapping(receipt.get("output_tree_custody"), name="output_tree_custody")
    control_tree_sha = _sha256(
        tree_custody.get("control_sha256"), name="output_tree_custody.control_sha256"
    )
    treatment_tree_sha = _sha256(
        tree_custody.get("treatment_sha256"), name="output_tree_custody.treatment_sha256"
    )

    measurements = _mapping(receipt.get("measurements"), name="measurements")
    control, control_d_seg, control_d_pose = _measurement(
        measurements.get("control"),
        name="measurements.control",
        expected_family=CONTROL_FAMILY,
        expected_archive_sha256=str(control_budget.get("matched_archive_sha256")),
        expected_archive_bytes=target_bytes,
        expected_output_tree_sha256=control_tree_sha,
    )
    treatment, treatment_d_seg, treatment_d_pose = _measurement(
        measurements.get("treatment"),
        name="measurements.treatment",
        expected_family=TREATMENT_FAMILY,
        expected_archive_sha256=str(treatment_budget.get("matched_archive_sha256")),
        expected_archive_bytes=target_bytes,
        expected_output_tree_sha256=treatment_tree_sha,
    )
    if control.get("axis") != treatment.get("axis"):
        raise CurveletTransferReceiptError("control/treatment evidence axes differ")
    axis = str(control.get("axis"))
    if treatment.get("basis_program_sha256") != receipt.get("basis_program_sha256"):
        raise CurveletTransferReceiptError("treatment measurement lost BasisProgramConfig custody")
    program_sha = str(receipt.get("basis_program_sha256", ""))
    if len(program_sha) != 64 or any(char not in "0123456789abcdef" for char in program_sha):
        raise CurveletTransferReceiptError("basis_program_sha256 must be a lowercase SHA-256")

    delta_d_seg = treatment_d_seg - control_d_seg
    delta_pose_score = math.sqrt(10.0 * treatment_d_pose) - math.sqrt(10.0 * control_d_pose)
    delta_nonrate_score = 100.0 * delta_d_seg + delta_pose_score
    dseg_transfer = delta_d_seg < 0.0
    pose_nonregression = treatment_d_pose <= control_d_pose
    authority = axis in AUTHORITY_AXES
    if not authority:
        status = "ADVISORY_ONLY_NO_PROMOTION"
    elif dseg_transfer and pose_nonregression and delta_nonrate_score < 0.0:
        status = "MEASURED_TRANSFER_FORMULATION_INSTANCE"
    else:
        status = "MEASURED_NO_TRANSFER_FORMULATION_INSTANCE"
    return CurveletTransferVerdict(
        status=status,
        verdict_scope=(
            "FORMULATION-INSTANCE: literal polar curvelet program hash versus legacy Fourier "
            "control at one exact matched-ZIP full-n600 evaluator cell"
        ),
        family_verdict="OPEN",
        authority_axis=axis,
        equal_archive_bytes=target_bytes,
        control_d_seg=control_d_seg,
        treatment_d_seg=treatment_d_seg,
        delta_d_seg_treatment_minus_control=delta_d_seg,
        control_d_pose=control_d_pose,
        treatment_d_pose=treatment_d_pose,
        delta_pose_score_term=delta_pose_score,
        delta_nonrate_score=delta_nonrate_score,
        dseg_transfer=dseg_transfer,
        pose_nonregression=pose_nonregression,
    )


__all__ = [
    "AUTHORITY_AXES",
    "CONTROL_FAMILY",
    "EQUATION_ID",
    "RECEIPT_SCHEMA",
    "TREATMENT_FAMILY",
    "CurveletTransferReceiptError",
    "CurveletTransferVerdict",
    "evaluate_curvelet_equal_archive_transfer",
]
