# SPDX-License-Identifier: MIT
"""Canonical shared-A seg/pose coupling law for Task #538."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar
from tac.witness_dsl.shared_resize_joint_coupling_policy import (
    AXIS,
    SCHEMA,
    validate_measurement_receipt,
)

EQUATION_ID = "shared_resize_joint_coupling_through_a_v1"
NON_RESOLVING_DISPLAY_ALIAS = "shared_resize_joint_coupling_through_A_v1"
MEMO = ".omx/research/completeness_coupling_joint_solve_20260718.md"
PENDING_UTC = "2026-07-18T00:00:00Z"


def pose_score_marginal(d_pose: float) -> float:
    """Return d sqrt(10*d_pose) / d d_pose = 5/sqrt(10*d_pose)."""

    value = float(d_pose)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("d_pose must be finite and strictly positive")
    return 5.0 / math.sqrt(10.0 * value)


def normalized_overlap(g_seg_norm2: float, cross: float, g_pose_norm2: float) -> float:
    """Cosine overlap from the three independent entries of a 2x2 Gram."""

    ss, sp, pp = float(g_seg_norm2), float(cross), float(g_pose_norm2)
    if not all(math.isfinite(v) for v in (ss, sp, pp)):
        raise ValueError("Gram entries must be finite")
    if ss < 0.0 or pp < 0.0:
        raise ValueError("Gram diagonal entries must be non-negative")
    if ss == 0.0 or pp == 0.0:
        return 0.0
    result = sp / math.sqrt(ss * pp)
    if abs(result) > 1.0 + 1e-7:
        raise ValueError("Gram cross entry violates Cauchy-Schwarz")
    return max(-1.0, min(1.0, result))


def joint_costate_coefficients(d_pose_baseline: float) -> dict[str, float]:
    """The score-derived joint coefficients; no tuneable coupling constant."""

    return {"lambda_seg": 100.0, "lambda_pose": pose_score_marginal(d_pose_baseline)}


def smooth_coupling_summary(
    raw_gram_2x2: Sequence[Sequence[float]], *, d_pose_baseline: float
) -> dict[str, Any]:
    """Price a smooth two-row Gram in score units and expose its overlap."""

    if len(raw_gram_2x2) != 2 or any(len(row) != 2 for row in raw_gram_2x2):
        raise ValueError("raw_gram_2x2 must be 2x2")
    matrix = [[float(v) for v in row] for row in raw_gram_2x2]
    if not all(math.isfinite(v) for row in matrix for v in row):
        raise ValueError("raw_gram_2x2 must be finite")
    scale = max(1.0, *(abs(v) for row in matrix for v in row))
    if abs(matrix[0][1] - matrix[1][0]) > 1e-8 * scale:
        raise ValueError("raw_gram_2x2 must be symmetric")
    overlap = normalized_overlap(matrix[0][0], matrix[0][1], matrix[1][1])
    coeff = joint_costate_coefficients(d_pose_baseline)
    lambdas = (coeff["lambda_seg"], coeff["lambda_pose"])
    priced = [
        [lambdas[i] * matrix[i][j] * lambdas[j] for j in range(2)]
        for i in range(2)
    ]
    return {
        "raw_gram_2x2": matrix,
        "score_priced_gram_2x2": priced,
        "normalized_overlap": overlap,
        "joint_costate_coefficients": coeff,
        "joint_costate_norm2": sum(priced[i][j] for i in range(2) for j in range(2)),
    }


def _stable_file_hash_and_size(path: str | Path) -> tuple[str, int]:
    artifact = Path(path)
    if not artifact.is_file():
        raise ValueError(f"bound receipt artifact does not exist: {artifact}")
    before = artifact.stat()
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    after = artifact.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"bound receipt artifact changed while hashing: {artifact}")
    return digest.hexdigest(), before.st_size


def _checkpoint_payload_from_npz(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    before = checkpoint.stat()
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            keys = sorted(
                member[:-4]
                for member in archive.namelist()
                if member.endswith(".npy") and "/" not in member
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"bound checkpoint is not a readable NPZ: {checkpoint}") from exc
    after = checkpoint.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"bound checkpoint changed while inspecting keys: {checkpoint}")
    carrier_keys = sorted(
        key
        for key in keys
        if "pose_carrier." in key
        or key.startswith("__cfg_pose_carrier")
        or key.startswith("__pose_carrier")
    )
    if carrier_keys:
        raise ValueError(f"bound checkpoint contains forbidden carrier keys: {carrier_keys}")
    key_digest = hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "carrier_absent": True,
        "base_inr_only": True,
        "detected_carrier_keys": [],
        "checkpoint_key_count": len(keys),
        "checkpoint_key_manifest_sha256": key_digest,
    }


def load_bound_measurement_receipt(receipt_path: str | Path) -> dict[str, Any]:
    """Reread and bind a receipt plus every scorer/input artifact it names."""

    path = Path(receipt_path)
    if not path.is_file():
        raise ValueError(f"measurement receipt does not exist: {path}")
    before = path.stat()
    try:
        raw_receipt = path.read_bytes()
        payload = json.loads(raw_receipt.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"measurement receipt cannot be reread as JSON: {path}") from exc
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"measurement receipt changed while reading: {path}")
    receipt_sha256 = hashlib.sha256(raw_receipt).hexdigest()
    validated = validate_measurement_receipt(payload)
    custody = validated["input_custody"]
    execution = validated["execution_custody"]
    artifact_rows = {
        "checkpoint": (
            custody["checkpoint_path"],
            custody["checkpoint_sha256"],
            execution["input_bytes"]["checkpoint"],
        ),
        "gt_cache": (
            custody["gt_cache_path"],
            custody["gt_cache_sha256"],
            execution["input_bytes"]["gt_cache"],
        ),
        "segnet": (
            custody["segnet_path"],
            custody["segnet_sha256"],
            execution["input_bytes"]["segnet"],
        ),
        "posenet": (
            custody["posenet_path"],
            custody["posenet_sha256"],
            execution["input_bytes"]["posenet"],
        ),
    }
    for name in ("modules_py", "frame_utils_py", "evaluate_py"):
        artifact_rows[name] = (
            execution["upstream_source_paths"][name],
            execution["upstream_source_sha256"][name],
            execution["input_bytes"][name],
        )
    verified: dict[str, dict[str, Any]] = {}
    for name, (artifact_path, expected_hash, expected_bytes) in artifact_rows.items():
        actual_hash, actual_bytes = _stable_file_hash_and_size(artifact_path)
        if actual_hash != expected_hash or actual_bytes != expected_bytes:
            raise ValueError(
                f"bound receipt artifact custody mismatch for {name}: "
                f"sha256={actual_hash}, bytes={actual_bytes}"
            )
        verified[name] = {
            "path": str(Path(artifact_path).resolve()),
            "sha256": actual_hash,
            "bytes": actual_bytes,
        }
    actual_checkpoint_payload = _checkpoint_payload_from_npz(custody["checkpoint_path"])
    if actual_checkpoint_payload != custody["checkpoint_payload"]:
        raise ValueError("bound checkpoint key/carrier custody differs from the receipt")
    return {
        "receipt": validated,
        "receipt_path": str(path.resolve()),
        "receipt_sha256": receipt_sha256,
        "verified_artifacts": verified,
    }


def _anchor_from_receipt_path(receipt_path: str | Path) -> EmpiricalAnchor:
    bound = load_bound_measurement_receipt(receipt_path)
    validated = bound["receipt"]
    if validated["evidence_status"] != "MEASURED_ADVISORY_SUBSET":
        raise ValueError("liveness-only receipts cannot populate an empirical anchor")
    smooth = validated["smooth_coupling"]
    central = validated["actual_response"]["by_support_fraction"][0][
        "central_secant_response_2x2"
    ]
    primary = smooth["shared_frame1"]
    smooth_cos = float(primary["normalized_overlap"])
    actual_cos = normalized_overlap(
        sum(float(central[r][0]) ** 2 for r in range(2)),
        sum(float(central[r][0]) * float(central[r][1]) for r in range(2)),
        sum(float(central[r][1]) ** 2 for r in range(2)),
    )
    shared_a = validated["shared_A"]
    if shared_a.get("seg_pose_operator_identical") is not True:
        raise ValueError("shared-A operator identity was not parity-verified")
    if shared_a.get("seg_preprocess_tensor_equal") is not True:
        raise ValueError("shared-A Seg preprocess tensor equality was not parity-verified")
    observed_yuv6_max_abs = float(shared_a.get("pose_yuv6_clone_max_abs", math.nan))
    if not math.isfinite(observed_yuv6_max_abs) or observed_yuv6_max_abs < 0.0:
        raise ValueError("shared-A Pose YUV6 clone max_abs must be finite and non-negative")
    expected_yuv6_max_abs = 0.0
    comparison_contract = "NONCOMMENSURATE_NO_CROSS_SURFACE_RESIDUAL"
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=bound["receipt_path"],
        reactivation_criteria=(
            "rerun on a new real n600-trained EMA checkpoint; subset advisory evidence never "
            "promotes or moves the frontier pointer"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macos_arm64",
        captured_at_utc=str(validated["captured_at_utc"]),
    )
    return EmpiricalAnchor(
        anchor_id=(
            f"shared_a_parity_and_coupling_diagnostics_subset_n"
            f"{validated['sample']['n_of_600']}_20260718"
        ),
        measurement_utc=str(validated["captured_at_utc"]),
        inputs={
            "checkpoint_sha256": validated["input_custody"]["checkpoint_sha256"],
            "gt_cache_sha256": validated["input_custody"]["gt_cache_sha256"],
            "pair_ids": list(validated["sample"]["pair_ids"]),
            "shared_A": dict(validated["shared_A"]),
            "axis": AXIS,
            "measurement_receipt_sha256": bound["receipt_sha256"],
            "verified_input_artifacts": bound["verified_artifacts"],
        },
        predicted_output={
            "shared_A_parity_expected": {
                "seg_pose_operator_identical": True,
                "seg_preprocess_tensor_equal": True,
                "pose_yuv6_clone_max_abs": expected_yuv6_max_abs,
            },
            "comparison_contract": comparison_contract,
            "claim": (
                "the exact shared-forward Seg/Pose preprocessing path is parity-verified; "
                "the coupling diagnostics are complementary and noncommensurate"
            ),
        },
        empirical_output={
            "shared_A_parity": {
                "seg_pose_operator_identical": shared_a["seg_pose_operator_identical"],
                "seg_preprocess_tensor_equal": shared_a["seg_preprocess_tensor_equal"],
                "pose_yuv6_clone_max_abs": observed_yuv6_max_abs,
            },
            "comparison_contract": comparison_contract,
            "smooth_input_gradient_overlap_surrogate": smooth_cos,
            "finite_lattice_response_column_overlap_unpriced": actual_cos,
            "shared_frame1_raw_gram_2x2": primary["raw_gram_2x2"],
            "shared_frame1_score_priced_gram_2x2": primary[
                "score_priced_gram_2x2"
            ],
            "full_pair_context": smooth["full_pair_context"],
            "actual_first_support_central_response_2x2": central,
            "evidence_status": validated["evidence_status"],
            "verdict_scope": validated["verdict_scope"],
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=abs(observed_yuv6_max_abs - expected_yuv6_max_abs),
        source_artifact=bound["receipt_path"],
        measurement_method=(
            "exact shared-forward parity verification plus complementary B1 smooth-input-"
            "gradient and B32 finite-lattice coupling diagnostics; no cross-surface "
            "calibration residual"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_shared_resize_joint_coupling_through_A_v1(
    *,
    measurement_receipt: Mapping[str, Any] | None = None,
    measurement_receipt_path: str | Path | None = None,
) -> CanonicalEquation:
    """Build pending structural law or a subset-labeled empirical specialization."""

    anchors: tuple[EmpiricalAnchor, ...] = ()
    last_calibration = PENDING_UTC
    residuals: dict[str, float] = {}
    evidence_status = "ASSUMED_AWAITING_VERIFICATION"
    source_path: str | Path = MEMO
    unbound_mapping_supplied = measurement_receipt is not None
    if measurement_receipt_path is not None:
        if measurement_receipt is not None:
            raise ValueError("pass a receipt path only; arbitrary mappings cannot bind an anchor")
        anchor = _anchor_from_receipt_path(measurement_receipt_path)
        anchors = (anchor,)
        last_calibration = anchor.measurement_utc
        residuals = {"shared_A_YUV6_parity_max_abs": anchor.residual}
        evidence_status = "VERIFIED_VIA_EMPIRICAL_ANCHOR_SUBSET_ONLY"
        source_path = anchor.source_artifact
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=source_path,
        reactivation_criteria=(
            "land a real n>=8 stride-subset receipt from a n600-trained EMA checkpoint; "
            "contest promotion additionally requires exact archive evaluation"
        ),
        measurement_axis=AXIS if anchors else "[research-signal]",
        hardware_substrate="macos_arm64" if anchors else "unknown",
        captured_at_utc=last_calibration,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Joint SegNet/PoseNet costate coupling through the shared resize A",
        one_line_summary=(
            "J=[d ell_seg/d render; d d_pose/d render], G=JJ^T, and "
            "q_joint=100*g_seg+5/sqrt(10*d_pose)*g_pose through one shared resize A."
        ),
        latex_form=(
            r"J_A=\begin{bmatrix}\nabla_r\ell_{seg}(A r)\\\nabla_r d_{pose}(A r)\end{bmatrix},\ "
            r"G_A=J_AJ_A^\top,\ q=100\,g_{seg}+\frac{5}{\sqrt{10d_{pose}}}g_{pose}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.shared_resize_joint_coupling_20260718:"
            "smooth_coupling_summary"
        ),
        domain_of_validity={
            "non_resolving_display_alias": NON_RESOLVING_DISPLAY_ALIAS,
            "display_alias_support": "NON_RESOLVING_DISPLAY_ALIAS",
            "shared_A_structural_authority": (
                "VERIFIED_VIA_SOURCE_INSPECTION: upstream PoseNet.preprocess_input and "
                "SegNet.preprocess_input both use torch bilinear 874x1164->384x512"
            ),
            "seg_smooth_row": "winner-rival zero-margin hinge; not exact discontinuous d_seg",
            "actual_authority": (
                "B32_DUPLICATE_LAST_SUBSET_ADVISORY uint8 one-LSB paired finite secants; "
                "not native/full-n600 comparable"
            ),
            "empirical_verification_status": evidence_status,
            "subset_evidence_only": bool(anchors),
            "unbound_mapping_supplied_pending_only": unbound_mapping_supplied,
            "receipt_schema": SCHEMA,
            "axis": AXIS if anchors else "pending",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": (
                "structural shared-A law plus checkpoint-and-subset instance when anchored; "
                "never a family negative or frontier claim"
            ),
        },
        units_in={
            "render": "camera_rgb_uint8_lattice_with_fp32_local_vjp",
            "d_pose_baseline": "PoseNet_first_six_MSE",
        },
        units_out={
            "raw_gram": "loss_gradient_squared_per_camera_RGB_unit_squared",
            "score_priced_gram": "score_gradient_squared_per_camera_RGB_unit_squared",
            "normalized_overlap": "dimensionless",
        },
        empirical_anchors=anchors,
        predicted_vs_empirical_residual=residuals,
        last_calibration_utc=last_calibration,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.shared_resize_joint_coupling_policy",
            "future v10 completeness compiler gate (not live-integrated)",
        ),
        canonical_producers=(
            "experiments.measure_shared_resize_seg_pose_coupling_20260718",
        ),
        provenance=provenance,
    )


def populate_shared_resize_joint_coupling_equation(
    *,
    path=None,
    lock_path=None,
    measurement_receipt: Mapping[str, Any] | None = None,
    measurement_receipt_path: str | Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append a registration to the supplied registry (tests use a temp JSONL)."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_shared_resize_joint_coupling_through_A_v1(
        measurement_receipt=measurement_receipt,
        measurement_receipt_path=measurement_receipt_path,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "NON_RESOLVING_DISPLAY_ALIAS",
    "build_shared_resize_joint_coupling_through_A_v1",
    "joint_costate_coefficients",
    "load_bound_measurement_receipt",
    "normalized_overlap",
    "populate_shared_resize_joint_coupling_equation",
    "pose_score_marginal",
    "smooth_coupling_summary",
]
