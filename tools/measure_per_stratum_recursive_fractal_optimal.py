#!/usr/bin/env python3
"""Fail-closed custody audit for the per-stratum recursive treatment table.

The tool reads settled receipts and existing archive bytes. It does not decode,
score, train, estimate missing payload sizes, or turn a diagnostic into a score.
Absent parser-consumed V9 sections therefore remain JSON ``null``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BEV_ROOT = Path("/Volumes/VertigoDataTier/pact/evidence/bev_staticity_v2_20260721/canonical_v1")
M1_ROOT = Path("/Volumes/VertigoDataTier/pact/evidence/m1_byteclose_20260721")
S4_ROOT = Path("/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721")
M1_ARCHIVE = M1_ROOT / "m1_candidate_archive.zip"
M1_BUILD = M1_ROOT / "build_receipt.json"
M1_DECODE_1 = M1_ROOT / "decode_1/receipt.json"
M1_DECODE_2 = M1_ROOT / "decode_2/receipt.json"
M1_HARNESS = M1_ROOT / "exact_candidate_harness_20260721.json"
M1_DECOMPOSITION = M1_ROOT / "hard_oracle_decomposition_20260721.json"
M1_PROGRESS = M1_ROOT / "hard_oracle_decomposition_progress.jsonl"
S4_ARCHIVE = S4_ROOT / "canonical_s4_20260721/archive.zip"
S4_BUILD = S4_ROOT / "canonical_s4_20260721/build_receipt.json"
S4_MEASUREMENT = S4_ROOT / "measurement_s4_20260721/measurement_receipt.json"
S4_PARITY = S4_ROOT / "measurement_s4_20260721/parity_checkpoint.json"
S4_ADVISORY = S4_ROOT / "measurement_s4_20260721/advisory_eval_checkpoint.json"
PRODUCTION_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/per_stratum_recursive_fractal_20260721/"
    "per_stratum_recursive_fractal_optimal_receipt.json"
)

EXPECTED_SHA256 = {
    "bev_n64": "94a7d7b5635e04d5da6f22e1d4f2e5b8d170a9dc95923e3835b9421aedb8bbba",
    "bev_n600": "c3ec847ba5ca43246f01af12f7bd650b14aba2784eb1878c29c16f8a4469ab96",
    "m1_archive": "a386a854e2483f839191f6c9da781f60b49774b71830b9baccee259be85edf8c",
    "m1_build": "0ef9bd6061ef6cd288b5ab8c140b04b57268d8ee01ae0d61a4250168045b75e2",
    "m1_decode_1": "53781b33fe99051babfebf3afc62801d28059a58a501f35e5ec93f20c60f97ba",
    "m1_decode_2": "e0179759e55c22cddfdc7da85f6ff8b5f9584f49d2c3caef469a26d905fc8fff",
    "m1_harness": "20d01dac12d8d96c7e20dca44aad1079c9c43e4a5ff92789214a3510faa0ba17",
    "m1_decomposition": "c15423c5316c61297cc7dd1f15df7168d83e46724f8326d2fbfbfac9b214cca5",
    "m1_progress": "6202f02f388522d087d6981428ba9f680d8add30ed4c6a182399b3e58777f653",
    "s4_archive": "d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed",
    "s4_build": "f9f2f9b63ea5c1b1dd3972752012e0a9279aca705064f1a8fa89c231bd51f590",
    "s4_measurement": "244d7b8fa695068755cd64a47572d58f2212f215e287e5d1ee6e6182384ea428",
    "s4_parity": "7e97c60a3676fc25634cda19186b3a197dabb077e9bc60124c90e9831ac8c2f6",
    "s4_advisory": "164d057ff5aca89664f10bc12f3e3585c654db659b34147407ea31d5b6b2a086",
}

REPO_INPUT_SHA256 = {
    ".omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json": (
        "38b1f5d5475037e360ce13f5aed7ae114d9e3c4834e7bffe388f0fb748fc5089"
    ),
    ".omx/research/recursive_fractal_optimal_representation_v9_measurement_receipt_20260714.json": (
        "225f8db81b5a0607f227fa34a97d13e51c5c0e664375eb5e8bf5248b1d0ea60b"
    ),
    ".omx/research/recursive_fractal_optimal_representation_v9_build_spec_20260714.md": (
        "e4e3435e2b87ce93e56df18d499475af8a80f5ca842c1d257bfadadf0bb96283"
    ),
    ".omx/research/c2_perclass_stratum_carrier_taxonomy_20260716.md": (
        "8ae65930b160bb0d68b10f6f078c5a78c6319ecbec2c8ae55641470ac305dc97"
    ),
    "src/tac/canonical_equations/perclass_stratum_carrier_taxonomy_20260716.py": (
        "33b465c185e0b69cc8499b506d4711c5de44c743efa89db83fcdb02bf3b96ca3"
    ),
    ".omx/research/SPEC_v8_perclass_decomposition_20260708.md": (
        "74417253b351f25185106d150fa67dae2b3357aeb33faed982e9b4756e2c4e72"
    ),
    "src/tac/boundary_math/warp_real_luma_frame0.py": (
        "e291a7355eecb542c9146b8a75fc1c3e0e44e003e15fc23d54c5a0b91864a03f"
    ),
    "src/tac/calibrated_geometry.py": ("884efda60849b932f1500463c7a0bfb1dcab6f3e5428207fccf718389bd74e78"),
}

V9_REQUIRED_MODULES = (
    "src/tac/boundary_math/decision_carrier_bundle.py",
    "src/tac/boundary_math/decision_palette_chroma.py",
    "src/tac/witness_dsl/decision_carrier_policy.py",
    "tools/probe_appearance_task_rank_n600.py",
)
V9_RECEIPT_REL = ".omx/research/recursive_fractal_optimal_representation_v9_measurement_receipt_20260714.json"
FRAME_COUNT = 600
SOURCE_BYTES = 37_545_489
# Full-precision ceil-minus-one crossing corrected by the Task #603 PRIMARY
# custody audit.  154,600 was a stale displayed approximation.
CAP_BYTES = 154_524
EXPECTED_POINTER = "0.1910828242"
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_NAME_FROM_M1 = {"Undriv": "Undrivable"}
RESIDUAL_DSEG = 0.01328

C2_BUCKETS = {
    "Road": (("edge", 0.004684), ("near", 0.001586)),
    "Lane": (("edge", 0.001098),),
    "Undrivable": (("edge+near", 0.000585),),
    "Movable": (("far", 0.003222), ("near", 0.001101), ("edge", 0.000614)),
    "MyCar": (("edge", 0.000140),),
}

TREATMENTS: dict[str, dict[str, Any]] = {
    "Road": {
        "v8_carrier": "shared Road+Undrivable bulk-boundary field; spend on separatrices",
        "coordinate_frame": "ground/BEV conditional on independent calibration rotation custody",
        "basis": "bulk field plus localized tangent atoms on incident separatrices",
        "temporal": "one translation-first se(3) xi plus receiver-proven sparse correction",
        "quantization": "Fisher-margin reverse-waterfill; shallow side chosen per class pair",
        "boundary": "shared Road-Lane, Road-Undrivable, Road-MyCar, Road-Movable annuli",
        "composition": "edge-centric generator; no duplicated sampled boundary",
    },
    "Lane": {
        "v8_carrier": "analytic ground-frame band prior (1-2 KB; not a composed byte row)",
        "coordinate_frame": "ground curve chart with independent calibration owed",
        "basis": "analytic curve/band plus dash grammar; repo compression abstraction",
        "temporal": "xi transport plus dash phase and sparse curvature/fork correction",
        "quantization": "thin-class high precision with uint8/resize survival",
        "boundary": "localized along-tangent 4-8 render-pixel atoms",
        "composition": "sampled-curve correction; not an OpenPilot-native polynomial claim",
    },
    "Undrivable": {
        "v8_carrier": "shared Road+Undrivable bulk-boundary field",
        "coordinate_frame": "image/horizon chart plus shared ground edge",
        "basis": "low-frequency region and one shared Road boundary generator",
        "temporal": "slow horizon knots carried by shared xi",
        "quantization": "low interior precision; precision on the Road-side shallow edge",
        "boundary": "horizon annulus has one unique home shared with Road",
        "composition": "no double charge for the shared separatrix",
    },
    "Movable": {
        "v8_carrier": "sparse islands plus the measured two-regime boundary law",
        "coordinate_frame": "per-object frame",
        "basis": "object-cell generators plus localized object-border atoms",
        "temporal": "per-object track plus xi; far-bucket persistence 0.865",
        "quantization": "both sides fragile; per-object Fisher-margin waterfill",
        "boundary": "one-sided border contrast plus sparse birth/death correction",
        "composition": "store object generators, not sampled object boundaries",
    },
    "MyCar": {
        "v8_carrier": "static bottom-connected hood mask prior (0.1-0.5 KB)",
        "coordinate_frame": "ego-image frame",
        "basis": "static mask plus localized rim correction",
        "temporal": "store once; sparse rim correction only",
        "quantization": "lowest post-seed debt; image-coordinate quantization",
        "boundary": "single static rim with one unique home",
        "composition": "bottom-connected component; no per-frame duplicate",
    },
}

V9_DIMENSIONS = {
    "pixel": "generator/tie/xi/Pose6/chroma DecisionCarrierBundle; RGB only at scorer boundary",
    "class": "five edge-centric carriers with merge-diff-correct reconciliation",
    "boundary": "smooth interior plus localized curvelet/shearlet annulus partition",
    "frame": "decision keyframe plus deterministic warp plus sparse correction",
    "pair": "one xi screw; Pose6 tangent only when xi is insufficient",
    "epoch": "metric annealing at stage boundaries only",
    "chroma": "decision palette plus luma-null sparse chroma correction",
    "scale": "multiresolution partition with unique lowest-sufficient home",
    "frequency": "smooth interior plus localized 4-8 render-pixel tangent atoms",
}


class CustodyError(RuntimeError):
    """Evidence cannot support the requested verdict."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise CustodyError(f"missing input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CustodyError(message)


def _read_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_schema: str | None = None,
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"hash drift: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CustodyError(f"invalid JSON: {path}") from exc
    _require(isinstance(payload, dict), f"schema drift (object required): {path}")
    if expected_schema is not None:
        _require(payload.get("schema") == expected_schema, f"schema drift: {path}")
    return payload, digest


def _number(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"schema drift ({label} must be numeric)",
    )
    return float(value)


def _integer(value: Any, label: str) -> int:
    _require(not isinstance(value, bool) and isinstance(value, int), f"schema drift ({label})")
    return int(value)


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        _require(isinstance(value, Mapping) and key in value, f"schema drift at {'.'.join(path)}")
        value = value[key]
    return value


def _matrix(value: Any, rows: int, cols: int, label: str) -> list[list[float]]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), label)
    _require(len(value) == rows, label)
    output: list[list[float]] = []
    for row in value:
        _require(isinstance(row, Sequence) and not isinstance(row, (str, bytes)), label)
        _require(len(row) == cols, label)
        output.append([_number(item, label) for item in row])
    return output


def _vector(value: Any, size: int, label: str) -> list[float]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), label)
    _require(len(value) == size, label)
    return [_number(item, label) for item in value]


def audit_rotation_stages(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit absolute 4x4 rotations and translation-first twist rotations."""
    _require(len(stages) == FRAME_COUNT, f"non-{FRAME_COUNT} stage count")
    max_r_deviation = 0.0
    max_absolute_translation = 0.0
    max_twist_omega = 0.0
    nonidentity_rotation_frames = 0
    nonzero_cross_omega = 0
    nonzero_within_omega = 0
    for expected_frame, stage in enumerate(stages):
        _require(stage.get("frame") in (None, expected_frame), "stage frame ordering drift")
        pose_value = stage.get("absolute_f1_pose")
        if isinstance(pose_value, Mapping):  # bounded fixture compatibility
            rotation = _matrix(
                pose_value.get("rotation_matrix", pose_value.get("R")),
                3,
                3,
                "absolute_f1_pose rotation",
            )
            translation = [0.0, 0.0, 0.0]
        else:
            pose = _matrix(pose_value, 4, 4, "absolute_f1_pose must be 4x4")
            _require(pose[3] == [0.0, 0.0, 0.0, 1.0], "invalid homogeneous last row")
            rotation = [row[:3] for row in pose[:3]]
            translation = [row[3] for row in pose[:3]]
        deviation = max(abs(rotation[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
        cross = _vector(stage.get("calibrated_cross_xi"), 6, "calibrated_cross_xi")
        within = _vector(stage.get("calibrated_within_xi"), 6, "calibrated_within_xi")
        cross_omega = math.sqrt(sum(value * value for value in cross[3:]))
        within_omega = math.sqrt(sum(value * value for value in within[3:]))
        max_r_deviation = max(max_r_deviation, deviation)
        max_absolute_translation = max(
            max_absolute_translation,
            math.sqrt(sum(value * value for value in translation)),
        )
        max_twist_omega = max(max_twist_omega, cross_omega, within_omega)
        nonidentity_rotation_frames += int(deviation > 0.0)
        nonzero_cross_omega += int(cross_omega > 0.0)
        nonzero_within_omega += int(within_omega > 0.0)
    nonzero_transitions = nonzero_cross_omega + nonzero_within_omega
    return {
        "stage_count": FRAME_COUNT,
        "absolute_pose_schema": "4x4 homogeneous transform",
        "twist_convention": "translation-first [rho,omega]",
        "max_rotation_matrix_deviation_from_identity": max_r_deviation,
        "nonidentity_absolute_rotation_frames": nonidentity_rotation_frames,
        "max_absolute_translation_magnitude": max_absolute_translation,
        "max_twist_rotation_vector_magnitude": max_twist_omega,
        "nonzero_cross_rotation_transitions": nonzero_cross_omega,
        "nonzero_within_rotation_transitions": nonzero_within_omega,
        "nonzero_rotation_transition_vectors": nonzero_transitions,
        "observed_pixel_homography_count": 0,
        "rotation_observation_verdict": (
            "IDENTITY_ONLY_STORED_ROTATION"
            if nonidentity_rotation_frames == 0 and nonzero_transitions == 0
            else "NONZERO_STORED_ROTATION"
        ),
        "calibration_explained_fraction": None,
        "genuine_geometry_fraction": None,
        "causal_R_t_verdict": "UNIDENTIFIABLE_FROM_CURRENT_CUSTODY",
    }


def vp_pixel(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float]:
    return cx + fx * math.tan(yaw), cy - fy * math.tan(pitch) / math.cos(yaw)


def _pitch_delta_for_upward_pixels(
    pixels: float,
    *,
    fy: float,
    cy: float,
    horizon: float,
) -> float:
    nominal = math.atan((cy - horizon) / fy)
    return math.atan((cy - horizon + abs(pixels)) / fy) - nominal


def vp_sensitivity(
    *,
    road_p50_pixels: float,
    lane_p50_pixels: float,
    fx: float = 400.3,
    fy: float = 399.5,
    cx: float = 256.0,
    cy: float = 192.0,
    horizon: float = 174.0,
) -> dict[str, Any]:
    nominal_pitch = math.atan((cy - horizon) / fy)
    nominal_u, nominal_v = vp_pixel(fx, fy, cx, cy, nominal_pitch, 0.0)
    pitch_threshold = math.radians(4.0)
    yaw_threshold = math.radians(2.0)
    corner_u, corner_v = vp_pixel(
        fx,
        fy,
        cx,
        cy,
        nominal_pitch + pitch_threshold,
        yaw_threshold,
    )
    threshold_du = abs(corner_u - nominal_u)
    threshold_dv = abs(corner_v - nominal_v)
    validity_pitch = (-0.09074, 0.17)
    validity_yaw = (-0.06912, 0.06912)
    equivalents: dict[str, dict[str, float]] = {}
    for name, pixels in (("Road", road_p50_pixels), ("Lane", lane_p50_pixels)):
        pitch_delta = _pitch_delta_for_upward_pixels(
            pixels,
            fy=fy,
            cy=cy,
            horizon=horizon,
        )
        yaw_delta = math.atan(abs(pixels) / fx)
        equivalents[name] = {
            "measured_p50_pixels": pixels,
            "pitch_delta_rad": pitch_delta,
            "pitch_delta_deg": math.degrees(pitch_delta),
            "yaw_delta_rad": yaw_delta,
            "yaw_delta_deg": math.degrees(yaw_delta),
        }
    return {
        "claim_kind": "DERIVED_SENSITIVITY_NOT_CAUSAL_ATTRIBUTION",
        "formula": "u=cx+fx*tan(yaw); v=cy-fy*tan(pitch)/cos(yaw)",
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "reconciled_horizon_pixels": horizon,
        "nominal_pitch_rad": nominal_pitch,
        "nominal_pitch_deg": math.degrees(nominal_pitch),
        "nominal_vanishing_point": {"u": nominal_u, "v": nominal_v},
        "spread_thresholds": {"pitch_deg": 4.0, "yaw_deg": 2.0},
        "threshold_corner_displacement_pixels": {
            "horizontal": threshold_du,
            "vertical": threshold_dv,
            "euclidean": math.hypot(threshold_du, threshold_dv),
        },
        "validity_windows_rad": {
            "pitch": list(validity_pitch),
            "yaw": list(validity_yaw),
        },
        "minimum_one_axis_angular_equivalents": equivalents,
        "calibration_explained_fraction": None,
        "genuine_geometry_fraction": None,
        "verdict": "UNIDENTIFIABLE_FROM_CURRENT_CUSTODY",
    }


def scorer_k_identity_canary(
    decompose: Callable[[Any], Any],
    *,
    identity_input: Any | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    result = decompose(identity if identity_input is None else identity_input)
    if isinstance(result, Mapping):
        rotation, translation, pose = result.get("R"), result.get("t"), result.get("pose")
    else:
        rotation = getattr(result, "R", None)
        translation = getattr(result, "t", None)
        pose = getattr(result, "pose", None)
    if hasattr(rotation, "tolist"):
        rotation = rotation.tolist()
    if hasattr(translation, "tolist"):
        translation = translation.tolist()
    if hasattr(pose, "tolist"):
        pose = pose.tolist()
    rotation_matrix = _matrix(rotation, 3, 3, "canary rotation")
    translation_vector = _vector(translation, 3, "canary translation")
    pose_vector = [0.0] * 6 if pose is None else _vector(pose, 6, "canary pose")
    rotation_error = max(abs(rotation_matrix[i][j] - identity[i][j]) for i in range(3) for j in range(3))
    translation_error = max(abs(value) for value in translation_vector)
    pose_error = max(abs(value) for value in pose_vector)
    _require(
        max(rotation_error, translation_error, pose_error) <= tolerance,
        "scorer-K identity homography canary failed",
    )
    return {
        "K": {"fx": 400.3, "fy": 399.5, "pp": [256.0, 192.0]},
        "working_shape": {"width": 512, "height": 384},
        "api": "CalibratedGeometry.homography_to_pose(return_decomposition=True)",
        "identity_homography": True,
        "max_rotation_error": rotation_error,
        "max_translation_error": translation_error,
        "max_pose_error": pose_error,
        "defaults_used": False,
        "verdict": "PASS_IDENTITY_CANARY_ONLY",
    }


def calibrated_geometry_identity_canary() -> dict[str, Any]:
    """Exercise the real API with explicit scorer-resolution intrinsics."""
    try:
        import torch

        from tac.calibrated_geometry import CalibratedGeometry
    except ImportError as exc:
        raise CustodyError("missing CalibratedGeometry/torch authority surface") from exc
    geometry = CalibratedGeometry(
        fx=400.3,
        fy=399.5,
        pp=(256.0, 192.0),
        width=512,
        height=384,
        device="cpu",
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)
    return scorer_k_identity_canary(
        lambda homography: geometry.homography_to_pose(
            homography,
            return_decomposition=True,
        ),
        identity_input=identity,
    )


def cap_accounting(cap_bytes: int = CAP_BYTES) -> dict[str, Any]:
    rate = Fraction(25 * cap_bytes, SOURCE_BYTES)
    remaining = Fraction(3, 20) - rate
    return {
        "cap_bytes": cap_bytes,
        "source_bytes": SOURCE_BYTES,
        "archive_to_source_ratio": float(Fraction(cap_bytes, SOURCE_BYTES)),
        "rate_component": float(rate),
        "rate_component_exact": f"{rate.numerator}/{rate.denominator}",
        "remaining_sub_0_15_distortion_budget": float(remaining),
        "remaining_sub_0_15_distortion_budget_exact": (f"{remaining.numerator}/{remaining.denominator}"),
    }


def _archive_accounting(archive_bytes: int) -> dict[str, Any]:
    cap_ratio = Fraction(archive_bytes, CAP_BYTES)
    rate = Fraction(25 * archive_bytes, SOURCE_BYTES)
    return {
        "archive_bytes": archive_bytes,
        "cap_ratio": float(cap_ratio),
        "cap_ratio_exact": f"{cap_ratio.numerator}/{cap_ratio.denominator}",
        "rate_component": float(rate),
        "rate_component_exact": f"{rate.numerator}/{rate.denominator}",
    }


def per_class_accounting(
    rows: Sequence[Mapping[str, Any]],
    total_pixels: int,
) -> dict[str, Any]:
    _require(total_pixels > 0, "total pixels must be positive")
    _require(len(rows) == len(CLASS_NAMES), "exactly five class rows required")
    normalized: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        raw_name = str(row.get("class_name"))
        name = CLASS_NAME_FROM_M1.get(raw_name, raw_name)
        _require(name in CLASS_NAMES and name not in normalized, f"invalid class row: {raw_name}")
        normalized[name] = row
    _require(set(normalized) == set(CLASS_NAMES), "five required class rows missing")
    output: list[dict[str, Any]] = []
    mismatch_total = 0
    gt_total = 0
    for name in CLASS_NAMES:
        row = normalized[name]
        gt_pixels = _integer(row.get("gt_pixels"), f"{name}.gt_pixels")
        mismatch_pixels = _integer(row.get("mismatch_pixels"), f"{name}.mismatch_pixels")
        _require(0 <= mismatch_pixels <= gt_pixels, f"invalid hard-oracle row: {name}")
        gt_total += gt_pixels
        mismatch_total += mismatch_pixels
        contribution = Fraction(mismatch_pixels, total_pixels)
        treatment = TREATMENTS[name]
        buckets = [
            {
                "stratum": stratum,
                "d_seg_contribution": value,
                "residual_weight": value / RESIDUAL_DSEG,
                "scope": "MEASURED_N600_PALETTE_VEHICLE",
            }
            for stratum, value in C2_BUCKETS[name]
        ]
        output.append(
            {
                "class_name": name,
                "claim_kind": "SETTLED_RECALL_TREATMENT_PLUS_MEASURED_M1_CONTROL",
                "v8_carrier": treatment["v8_carrier"],
                "v9_dimensional_treatment": {
                    key: treatment[key]
                    for key in (
                        "coordinate_frame",
                        "basis",
                        "temporal",
                        "quantization",
                        "boundary",
                        "composition",
                    )
                },
                "c2_residual_buckets": buckets,
                "c2_residual_d_seg_sum": sum(item[1] for item in C2_BUCKETS[name]),
                "m1_hard_oracle": {
                    "gt_pixels": gt_pixels,
                    "mismatch_pixels": mismatch_pixels,
                    "conditional_error": mismatch_pixels / gt_pixels,
                    "d_seg_contribution": float(contribution),
                    "d_seg_contribution_exact": (f"{contribution.numerator}/{contribution.denominator}"),
                },
                "measured_unique_home_bytes": None,
                "unique_home_byte_blocker": ("NO_PARSER_CONSUMED_SECTION_WITH_UNIQUE_HOME_ATTRIBUTION"),
                "verdict_scope": "CURRENT_COMPOSITION_CUSTODY_ONLY_FAMILY_OPEN",
            }
        )
    _require(gt_total == total_pixels, "per-class GT pixels do not cover scorer grid")
    aggregate = Fraction(mismatch_total, total_pixels)
    return {
        "rows": output,
        "aggregate_gt_pixels": gt_total,
        "aggregate_mismatch_pixels": mismatch_total,
        "aggregate_d_seg": float(aggregate),
        "aggregate_d_seg_exact": {
            "numerator": mismatch_total,
            "denominator": total_pixels,
            "reduced": f"{aggregate.numerator}/{aggregate.denominator}",
        },
        "nonclass_saddle_debt": {
            "d_seg": 0.000100,
            "weight": 0.000100 / RESIDUAL_DSEG,
            "home": "incident edge packets; not a sixth class carrier",
        },
        "unassigned_c2_remainder": 0.000150,
    }


def _validate_pointer(repo_root: Path) -> dict[str, Any]:
    report = repo_root / "reports/latest.md"
    text = report.read_text(encoding="utf-8")
    match = re.search(
        r"\|\s*\*\*`\[contest-CPU Linux x86_64\]`\*\*\s*\|\s*\*\*([0-9.]+)\*\*",
        text,
    )
    _require(match is not None, "canonical contest-CPU pointer row missing")
    score = match.group(1)
    _require(score == EXPECTED_POINTER, f"pointer mutation detected: {score}")
    return {
        "source": "reports/latest.md canonical frontier table",
        "source_sha256": sha256_file(report),
        "axis": "[contest-CPU Linux x86_64]",
        "score": score,
        "delta": None,
        "mutated": False,
    }


def _validate_repo_inputs(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, expected in REPO_INPUT_SHA256.items():
        actual = sha256_file(repo_root / relative)
        _require(actual == expected, f"hash drift: {relative}")
        hashes[relative] = actual
    return hashes


def _validate_bev(
    bev_root: Path,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, str]]]:
    n64_path = bev_root / "receipt_n64.json"
    n600_path = bev_root / "receipt_n600.json"
    n64, n64_hash = _read_json(
        n64_path,
        expected_sha256=EXPECTED_SHA256["bev_n64"],
        expected_schema="bev_staticity_developability_probe.v2",
    )
    n600, n600_hash = _read_json(
        n600_path,
        expected_sha256=EXPECTED_SHA256["bev_n600"],
        expected_schema="bev_staticity_developability_probe.v2",
    )
    for payload, scale in ((n64, "n64"), (n600, "n600")):
        _require(payload.get("scale") == scale, f"BEV scale drift: {scale}")
        _require(_nested(payload, ("config", "seed")) == 1234, "BEV seed drift")
        authority = _nested(payload, ("authority",))
        _require(authority.get("score_claim") is False, "BEV score authority drift")
        _require(authority.get("pointer_moved") is False, "BEV pointer mutation")
    calibration = _nested(
        n600,
        ("D0_hood_positive_control", "absolute_trajectory", "calibration"),
    )
    _require(calibration == {"pitch_rad": -0.05, "s_r": 0.0, "s_t": -0.00143}, "calibration drift")
    geometry = _nested(n600, ("config", "geometry"))
    intrinsics = geometry.get("intrinsics")
    _require(
        intrinsics == {"cx": 256.0, "cy": 192.0, "fx": 400.3, "fy": 399.5},
        "scorer intrinsics drift",
    )
    _require(geometry.get("v_horizon") == 174.0, "horizon drift")
    _require(geometry.get("camera_height_m") == 1.22, "camera-height drift")
    stage_paths = sorted((bev_root / "measurement_stages_n600").glob("frame_*.json"))
    _require(len(stage_paths) == FRAME_COUNT, "non-600 persisted stage count")
    stages: list[dict[str, Any]] = []
    stage_bindings: list[dict[str, str]] = []
    stage_manifest = hashlib.sha256()
    for path in stage_paths:
        stage, digest = _read_json(path)
        relative = path.relative_to(bev_root).as_posix()
        stages.append(stage)
        stage_bindings.append({"path": relative, "sha256": digest})
        stage_manifest.update(f"{relative}\0{digest}\n".encode())
    rotation = audit_rotation_stages(stages)
    road = _nested(n600, ("D1_BEV_staticity", "Road"))
    lane = _nested(n600, ("D1_BEV_staticity", "Lane"))

    def residual_summary(payload: Mapping[str, Any]) -> dict[str, float]:
        residual = _nested(payload, ("ruling_reconstruction_residual_px",))
        return {
            "p50_pixels": _number(residual.get("p50"), "p50"),
            "p90_pixels": _number(residual.get("p90"), "p90"),
            "fraction_at_or_below_1px": _number(
                payload.get("static_fraction_at_1px_floor"),
                "static fraction",
            ),
        }

    def hood_summary(payload: Mapping[str, Any]) -> dict[str, float]:
        summary = _nested(
            payload,
            ("D0_hood_positive_control", "hood", "temporal_boundary_summary"),
        )
        residual = _nested(summary, ("ruling_reconstruction_residual_px",))
        return {
            "p50_pixels": _number(residual.get("p50"), "hood p50"),
            "p90_pixels": _number(residual.get("p90"), "hood p90"),
            "fraction_at_or_below_1px": _number(
                summary.get("developable_fraction_at_noise_floor"),
                "hood static fraction",
            ),
        }

    road_summary = residual_summary(road)
    lane_summary = residual_summary(lane)
    result = {
        "claim_kind": "MEASURED_RECEIPT_REAUDIT",
        "n64_hood_positive_control": hood_summary(n64),
        "n600_hood_positive_control": hood_summary(n600),
        "n600_road": road_summary,
        "n600_lane": lane_summary,
        "calibration": calibration,
        "geometry": {
            "intrinsics": intrinsics,
            "v_horizon": 174.0,
            "camera_height_m": 1.22,
        },
        "rotation_custody": rotation,
        "calibrated_geometry_identity_canary": calibrated_geometry_identity_canary(),
        "openpilot_vanishing_point_sensitivity": vp_sensitivity(
            road_p50_pixels=road_summary["p50_pixels"],
            lane_p50_pixels=lane_summary["p50_pixels"],
        ),
        "stage_manifest": {
            "entry_count": len(stage_bindings),
            "sha256": stage_manifest.hexdigest(),
            "entries": stage_bindings,
        },
        "verdict": "UNIDENTIFIABLE_FROM_CURRENT_CUSTODY",
        "verdict_scope": ("exact translation-only BEV-v2 custody; independently calibrated ground charts open"),
    }
    return result, {str(n64_path): n64_hash, str(n600_path): n600_hash}, stage_bindings


def _validate_m1() -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    archive_hash = sha256_file(M1_ARCHIVE)
    _require(archive_hash == EXPECTED_SHA256["m1_archive"], "M1 archive hash drift")
    _require(M1_ARCHIVE.stat().st_size == 90_566, "M1 archive size drift")
    build, build_hash = _read_json(
        M1_BUILD,
        expected_sha256=EXPECTED_SHA256["m1_build"],
        expected_schema="c2_integer_plane_counted_archive.v1",
    )
    _require(build.get("archive_sha256") == archive_hash, "M1 build/archive hash drift")
    _require(build.get("archive_bytes") == 90_566, "M1 build/archive size drift")
    _require(build.get("accounted_archive_bytes") == 90_566, "M1 build accounting drift")
    _require(build.get("score_claim") is False, "M1 build false score authority")
    decode_receipts: list[tuple[dict[str, Any], str]] = []
    for path, digest_key in (
        (M1_DECODE_1, "m1_decode_1"),
        (M1_DECODE_2, "m1_decode_2"),
    ):
        decode_receipts.append(
            _read_json(
                path,
                expected_sha256=EXPECTED_SHA256[digest_key],
                expected_schema="c2_integer_plane_byte_close_receipt.v1",
            )
        )
    for decode, _ in decode_receipts:
        _require(decode.get("archive_sha256") == archive_hash, "M1 decode/archive drift")
        _require(decode.get("archive_bytes_full") == 90_566, "M1 decode size drift")
        _require(decode.get("logical_pair_count") == FRAME_COUNT, "M1 decode pair-count drift")
        _require(decode.get("factor2_exact") is True, "M1 factor-2 closure drift")
        _require(decode.get("numpy_decode_equal") is True, "M1 NumPy parity drift")
    decoded_raw_hashes = {decode.get("decoded_raw_sha256") for decode, _ in decode_receipts}
    _require(len(decoded_raw_hashes) == 1 and None not in decoded_raw_hashes, "M1 double-decode drift")
    harness, harness_hash = _read_json(
        M1_HARNESS,
        expected_sha256=EXPECTED_SHA256["m1_harness"],
        expected_schema="r1b_boundary_generator_n600_measurement.v1",
    )
    decomposition, decomposition_hash = _read_json(
        M1_DECOMPOSITION,
        expected_sha256=EXPECTED_SHA256["m1_decomposition"],
        expected_schema="r1b_n600_hard_oracle_decomposition.v1",
    )
    authority = _nested(harness, ("authority",))
    _require(authority.get("hard_cpu_torch") is True, "M1 hard CPU authority drift")
    _require(authority.get("through_r") is True, "M1 through-R authority drift")
    _require(authority.get("contest_score_claim") is False, "M1 false score authority")
    row = _nested(harness, ("row",))
    _require(row.get("seed") == 1234, "M1 seed drift")
    _require(row.get("pair_count") == 600 and row.get("batch_count") == 38, "M1 n600 custody drift")
    _require(row.get("d_pose") == 127.36588287353516, "M1 d_pose drift")
    _require(row.get("d_seg") == 0.003515794640406966, "M1 official d_seg drift")
    _require(
        _nested(harness, ("decode", "decoded_raw_sha256")) == next(iter(decoded_raw_hashes)),
        "M1 harness/direct-decode stream drift",
    )
    aggregate = _nested(decomposition, ("measurement", "aggregate"))
    integer = _nested(decomposition, ("measurement", "integer_accounting"))
    _require(aggregate.get("pair_count") == 600 and aggregate.get("batch_count") == 38, "M1 decomposition count drift")
    _require(aggregate.get("d_pose_official_float32") == row.get("d_pose"), "M1 Pose crosscheck drift")
    total_pixels = _integer(integer.get("total_segnet_pixels"), "M1 total pixels")
    mismatch = _integer(integer.get("mismatch_pixels"), "M1 mismatch pixels")
    exact_dseg = Fraction(mismatch, total_pixels)
    _require(float(exact_dseg) == aggregate.get("d_seg_exact_argmax_rational"), "M1 exact d_seg drift")
    progress_hash = sha256_file(M1_PROGRESS)
    _require(progress_hash == EXPECTED_SHA256["m1_progress"], "M1 progress hash drift")
    try:
        progress_rows = [json.loads(line) for line in M1_PROGRESS.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise CustodyError("invalid M1 checkpoint JSONL") from exc
    _require(len(progress_rows) == 38, "M1 checkpoint count drift")
    for index, checkpoint in enumerate(progress_rows, start=1):
        _require(
            checkpoint.get("schema") == "r1b_n600_hard_oracle_decomposition.v1.checkpoint.v1",
            "M1 checkpoint schema drift",
        )
        _require(checkpoint.get("batch_count") == index, "M1 checkpoint ordering drift")
    final_checkpoint = progress_rows[-1]
    _require(final_checkpoint.get("sample_count") == FRAME_COUNT, "M1 final checkpoint sample drift")
    _require(final_checkpoint.get("mismatch_pixels") == mismatch, "M1 final checkpoint mismatch drift")
    raw_per_class = _nested(decomposition, ("measurement", "per_class"))
    class_rows = [
        {
            "class_name": name,
            "gt_pixels": _integer(values.get("gt_pixels"), f"{name}.gt_pixels"),
            "mismatch_pixels": _integer(
                values.get("mismatch_pixels"),
                f"{name}.mismatch_pixels",
            ),
        }
        for name, values in raw_per_class.items()
    ]
    table = per_class_accounting(class_rows, total_pixels)
    _require(table["aggregate_mismatch_pixels"] == mismatch, "M1 class mismatch sum drift")
    result = {
        **_archive_accounting(90_566),
        "archive_sha256": archive_hash,
        "axis": authority.get("axis"),
        "receiver_closed": True,
        "through_r": True,
        "hard_cpu_torch": True,
        "seed": 1234,
        "pair_count": 600,
        "batch_count": 38,
        "decoded_raw_sha256": next(iter(decoded_raw_hashes)),
        "double_decode_byte_identical": True,
        "preserved_checkpoint_count": len(progress_rows),
        "d_seg_official_float32": row.get("d_seg"),
        "d_seg_exact_argmax_rational": float(exact_dseg),
        "d_seg_exact_fraction": f"{exact_dseg.numerator}/{exact_dseg.denominator}",
        "d_pose_official_float32": row.get("d_pose"),
        "mismatch_pixels": mismatch,
        "total_segnet_pixels": total_pixels,
        "full_richness_exact": False,
        "verdict": "MEASURED_CONTROL_FAILS_FULL_SCORER_CELL_CLOSURE",
        "verdict_scope": "one exact M1 receiver output; no family negative or contest score",
    }
    hashes = {
        str(M1_ARCHIVE): archive_hash,
        str(M1_BUILD): build_hash,
        str(M1_DECODE_1): decode_receipts[0][1],
        str(M1_DECODE_2): decode_receipts[1][1],
        str(M1_HARNESS): harness_hash,
        str(M1_DECOMPOSITION): decomposition_hash,
        str(M1_PROGRESS): progress_hash,
    }
    return result, hashes, table["rows"]


def _validate_s4() -> tuple[dict[str, Any], dict[str, str]]:
    archive_hash = sha256_file(S4_ARCHIVE)
    _require(archive_hash == EXPECTED_SHA256["s4_archive"], "S4 archive hash drift")
    _require(S4_ARCHIVE.stat().st_size == 451_191, "S4 archive size drift")
    build, build_hash = _read_json(
        S4_BUILD,
        expected_sha256=EXPECTED_SHA256["s4_build"],
        expected_schema="s4_archive_build_receipt.v1",
    )
    build_archive = _nested(build, ("archive",))
    _require(build_archive.get("sha256") == archive_hash, "S4 build/archive hash drift")
    _require(build_archive.get("bytes") == 451_191, "S4 build/archive size drift")
    parity, parity_hash = _read_json(
        S4_PARITY,
        expected_sha256=EXPECTED_SHA256["s4_parity"],
        expected_schema="s4_parity_checkpoint.v1",
    )
    parity_rows = parity.get("rows")
    _require(isinstance(parity_rows, list), "S4 parity rows drift")
    full_rows = [row for row in parity_rows if isinstance(row, Mapping) and row.get("pairs") == FRAME_COUNT]
    _require(len(full_rows) == 1, "S4 n600 parity row drift")
    full_parity = full_rows[0]
    _require(full_parity.get("byte_exact_parity") is True, "S4 n600 repo parity drift")
    _require(
        full_parity.get("standalone_double_decode_deterministic") is True,
        "S4 n600 double-decode drift",
    )
    parity_streams = {
        _nested(full_parity, (surface, "stream_sha256"))
        for surface in ("repo_native", "standalone_first", "standalone_second")
    }
    _require(len(parity_streams) == 1 and None not in parity_streams, "S4 n600 stream drift")
    advisory_checkpoint, advisory_hash = _read_json(
        S4_ADVISORY,
        expected_sha256=EXPECTED_SHA256["s4_advisory"],
        expected_schema="s4_advisory_eval_checkpoint.v1",
    )
    _require(advisory_checkpoint.get("passed") is True, "S4 advisory checkpoint incomplete")
    _require(advisory_checkpoint.get("archive_sha256") == archive_hash, "S4 advisory/archive drift")
    receipt, receipt_hash = _read_json(
        S4_MEASUREMENT,
        expected_sha256=EXPECTED_SHA256["s4_measurement"],
        expected_schema="s4_archive_composer_measurement.v1",
    )
    _require(receipt.get("research_only") is True, "S4 research authority drift")
    _require(receipt.get("promotion_eligible") is False, "S4 promotion authority drift")
    advisory = _nested(receipt, ("advisory_eval",))
    _require(advisory.get("passed") is True, "S4 advisory eval incomplete")
    _require(advisory.get("archive_sha256") == archive_hash, "S4 receipt/archive drift")
    _require(advisory == advisory_checkpoint, "S4 embedded/direct advisory checkpoint drift")
    evaluate = _nested(advisory, ("stages", "evaluate"))
    measured = _nested(evaluate, ("measured",))
    _require(evaluate.get("status") == "complete", "S4 n600 eval incomplete")
    _require(measured.get("d_seg") == 0.60198647, "S4 d_seg drift")
    _require(measured.get("d_pose") == 163.11865234, "S4 d_pose drift")
    cleanup = _nested(advisory, ("stages", "cleanup"))
    _require(cleanup.get("deleted_after_success") is True, "S4 cleanup custody drift")
    result = {
        **_archive_accounting(451_191),
        "archive_sha256": archive_hash,
        "axis": advisory.get("axis"),
        "receiver_closed": True,
        "deterministic_n600_decode_sha256": next(iter(parity_streams)),
        "repo_standalone_n600_byte_identical": True,
        "d_seg": measured.get("d_seg"),
        "d_pose": measured.get("d_pose"),
        "full_richness_exact": False,
        "verdict": "MEASURED_CONTROL_FAILS_RATE_AND_SCORER_CELLS",
        "verdict_scope": "one exact S4 receiver output; no family negative or contest score",
    }
    return result, {
        str(S4_ARCHIVE): archive_hash,
        str(S4_BUILD): build_hash,
        str(S4_PARITY): parity_hash,
        str(S4_ADVISORY): advisory_hash,
        str(S4_MEASUREMENT): receipt_hash,
    }


def requested_v9_row(
    repo_root: Path = REPO_ROOT,
    receiver_rate_custody_path: Path | None = None,
) -> dict[str, Any]:
    receipt, _ = _read_json(
        repo_root / V9_RECEIPT_REL,
        expected_sha256=REPO_INPUT_SHA256[V9_RECEIPT_REL],
        expected_schema="recursive_fractal_optimal_representation_v9_measurement_receipt.v1",
    )
    authority = _nested(receipt, ("authority",))
    _require(authority.get("score_claim_valid") is False, "V9 authority drift")
    module_presence = {relative: (repo_root / relative).is_file() for relative in V9_REQUIRED_MODULES}
    missing: list[str] = []
    if authority.get("candidate_archive_present") is not True:
        missing.append("candidate_archive")
    if not all(module_presence.values()):
        missing.append("encoder_parser_receiver_modules")
    if authority.get("full_n600_measured") is not True:
        missing.append("n600_decode_and_full_scorer_receipt")
    if authority.get("exact_evaluator_called") is not True:
        missing.append("exact_evaluator_receipt")
    missing.extend(("parser_consumed_section_registry", "unique_home_byte_attribution"))
    dimensions: dict[str, dict[str, Any]] = {
        name: {
            "settled_optimal_form": form,
            "exact_composed_bytes": None,
            "claim_kind": "SETTLED_RECALL_FORM_NO_VERDICT_BYTES",
        }
        for name, form in V9_DIMENSIONS.items()
    }
    row = {
        "name": "requested_v9_composition",
        "module_presence": module_presence,
        "historical_65172_byte_diagnostic": _nested(
            receipt,
            ("rate_diagnostic", "baseline_current_receiver_0bin"),
        ),
        "diagnostic_is_archive": False,
        "total_archive_bytes": None,
        "dimension_bytes": dict.fromkeys(V9_DIMENSIONS),
        "per_stratum_bytes": dict.fromkeys(CLASS_NAMES),
        "dimensions": dimensions,
        "missing_required_custody": missing,
        "verdict": "NO_VERDICT_RECEIVER_RATE_CUSTODY",
        "verdict_scope": "current checkout and #503 custody; representation family open",
    }
    if receiver_rate_custody_path is None:
        return row
    from tac.optimization.direct_description_minimizer import (
        DirectDescriptionError,
        validate_receiver_rate_custody,
    )

    try:
        custody = validate_receiver_rate_custody(receiver_rate_custody_path)
    except DirectDescriptionError as exc:
        raise CustodyError(f"invalid direct-description receiver-rate custody: {exc}") from exc
    missing = [] if custody["exact_evaluator_called"] else ["exact_evaluator_receipt"]
    row.update(
        {
            "total_archive_bytes": custody["archive_bytes"],
            "archive_sha256": custody["archive_sha256"],
            "dimension_bytes": custody["dimension_bytes"],
            "per_stratum_bytes": custody["per_stratum_bytes"],
            "dimensions": {
                name: {
                    **dimensions[name],
                    "exact_composed_bytes": custody["dimension_bytes"][name],
                    "claim_kind": "MEASURED_PARSER_CONSUMED_UNIQUE_HOME_CUSTODY",
                }
                for name in V9_DIMENSIONS
            },
            "receiver_rate_custody": custody,
            "missing_required_custody": missing,
            "verdict": (
                "RECEIVER_RATE_CUSTODY_AND_EXACT_EVAL_PRESENT"
                if not missing
                else "RECEIVER_RATE_CUSTODY_PRESENT_EXACT_EVAL_OWED"
            ),
            "verdict_scope": (
                "one fresh exact A(z) candidate with parser-consumed unique-home bytes; "
                "no family claim or pointer movement"
            ),
        }
    )
    return row


def _storage_preflight(output: Path) -> dict[str, Any]:
    _require(
        str(output).startswith("/Volumes/VertigoDataTier/pact/"),
        "full production receipt must use the primary SSD tier",
    )
    existing = output.parent
    while not existing.exists():
        _require(existing != existing.parent, "no existing output ancestor")
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    required = 1 << 20
    _require(usage.free >= required, "insufficient receipt output space")
    return {
        "tier": "/Volumes/VertigoDataTier/pact",
        "checked_existing_ancestor": str(existing),
        "required_free_bytes": required,
        "observed_free_bytes_at_least_required": True,
        "observed_free_bytes_redacted_for_determinism": True,
        "passed": True,
    }


def build_receipt(
    *,
    repo_root: Path = REPO_ROOT,
    bev_root: Path = BEV_ROOT,
    storage_preflight: Mapping[str, Any] | None = None,
    receiver_rate_custody_path: Path | None = None,
) -> dict[str, Any]:
    repo_hashes = _validate_repo_inputs(repo_root)
    pointer = _validate_pointer(repo_root)
    bev, bev_hashes, _ = _validate_bev(bev_root)
    m1, m1_hashes, treatment_rows = _validate_m1()
    s4, s4_hashes = _validate_s4()
    v9 = requested_v9_row(repo_root, receiver_rate_custody_path)
    receiver_rate_present = v9["verdict"] != "NO_VERDICT_RECEIVER_RATE_CUSTODY"
    exact_eval_present = v9["verdict"] == "RECEIVER_RATE_CUSTODY_AND_EXACT_EVAL_PRESENT"
    return {
        "schema": "per_stratum_recursive_fractal_optimal_receipt.v2",
        "authority": {
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "axis": "[macOS-CPU advisory] controls plus DERIVED sensitivity",
            "main_landing_review_required": True,
        },
        "pointer": pointer,
        "input_sha256": {**repo_hashes, **bev_hashes, **m1_hashes, **s4_hashes},
        "cap_accounting": cap_accounting(),
        "road_lane_ground_frame": bev,
        "controls": {"M1": m1, "S4": s4},
        "per_stratum_treatment_table": treatment_rows,
        "c2_taxonomy": {
            "residual_d_seg": RESIDUAL_DSEG,
            "class_bucket_sum": sum(value for values in C2_BUCKETS.values() for _, value in values),
            "saddle_d_seg": 0.000100,
            "everything_else_d_seg": 0.000150,
            "total_crosscheck": sum(value for values in C2_BUCKETS.values() for _, value in values)
            + 0.000100
            + 0.000150,
            "scope": "MEASURED_N600_PALETTE_VEHICLE; treatment forms SETTLED_RECALL",
        },
        "requested_v9": v9,
        "openpilot_representation_correction": (
            "v0.9.7/current lane outputs are four sampled 33-point (x,y,z) curves "
            "with probability/std metadata and two sampled road edges; the repo "
            "LaneLine polynomial is a compression abstraction, not an OpenPilot-native carrier"
        ),
        "storage_preflight": None if storage_preflight is None else dict(storage_preflight),
        "disk_hygiene": {
            "bulk_bytes_created": 0,
            "raw_decode_or_scorer_scratch_created": False,
            "cleanup_required": False,
            "inputs_read_only": True,
        },
        "verdict": (
            "CANDIDATE_RECEIVER_RATE_CUSTODY_PRESENT"
            if exact_eval_present
            else ("NO_VERDICT_EXACT_EVALUATOR_CUSTODY" if receiver_rate_present else "NO_VERDICT_RECEIVER_RATE_CUSTODY")
        ),
        "verdict_scope": (
            "current V9 composition custody only; v8/v9 carrier families and independently "
            "calibrated ground-frame treatments remain open"
        ),
    }


def compact_receipt(full_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the 600-entry stage list while retaining its deterministic binding."""
    compact = json.loads(json.dumps(full_receipt, sort_keys=True, allow_nan=False))
    manifest = compact["road_lane_ground_frame"]["stage_manifest"]
    entries = manifest.pop("entries")
    manifest["entries_omitted_from_compact_receipt"] = len(entries)
    manifest["full_receipt_required_for_per_file_bindings"] = True
    return compact


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRODUCTION_RECEIPT)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--bev-root", type=Path, default=BEV_ROOT)
    parser.add_argument(
        "--receiver-rate-custody",
        type=Path,
        help="typed fresh A(z) parser-consumption/unique-home receipt (never a control payload sum)",
    )
    args = parser.parse_args(argv)
    try:
        preflight = _storage_preflight(args.output)
        receipt = build_receipt(
            repo_root=args.repo_root,
            bev_root=args.bev_root,
            storage_preflight=preflight,
            receiver_rate_custody_path=args.receiver_rate_custody,
        )
        _atomic_json(args.output, receipt)
        if args.compact_output is not None:
            _atomic_json(args.compact_output, compact_receipt(receipt))
    except CustodyError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
