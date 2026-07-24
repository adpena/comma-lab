# SPDX-License-Identifier: MIT
"""Targeted DM4 cures for the ten named DM2 realization rows.

This is a bounded measurement harness, not a trainer or descent loop.  It
constructs a fixed candidate menu from the genuine #502 literal-curvelet and
compact-shearlet frames, orders that menu with the exact frozen-SegNet
Fisher/margin pullback, corrects the ordering with receiver-closed secants, and
admits a row only when its exact DM1 L4 semantic record survives R/uint8.

For the five DM2 pose-harm rows, the harness measures a six-output PoseNet
secant in RGB coefficient coordinates and evaluates a fixed SE(3)-null/QCQP
candidate family.  All selected writes are then composed across the complete
25-row demand set and Seg/Pose/rate are freshly remeasured.  The result is a
research-only macOS-CPU advisory upper bound; it never calls ``evaluate.py`` or
emits a candidate archive.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tac.boundary_math.compact_shearlet_frame import compact_shearlet_feats
from tac.boundary_math.localized_basis_frames import (
    ATOM_SPEC_SHA256 as LITERAL_CURVELET_ATOM_SPEC_SHA256,
)
from tac.boundary_math.localized_basis_frames import (
    inclusive_grid_coords,
    localized_basis_features_grid_numpy,
)
from tac.optimization.ddm_dm1_solved_value_pricing import (
    _class_id,
    _class_pair_from_bucket,
    _event_support,
    _winner_symbols,
    canonical_json_bytes,
    checked_json,
    sha256_bytes,
)
from tac.optimization.ddm_dm2_l3_realization_race import (
    AXIS,
    FRAME_INDEX,
    POINTER,
    SCORER_HW,
    DM2RealizationError,
    RGBDeltaRecord,
    _expected_record,
    _pose_forward,
    _score_delta,
    _seg_forward,
    candidate_scorer_plane,
    decode_joint_rgb_records,
    encode_joint_rgb_records,
    price_rgb_raw,
)
from tac.optimization.ddm_dm2_l3_realization_race import (
    _bound_inputs as bind_dm2_inputs,
)
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import (
    SolveDiffMiningConfigV1,
    _load_production_inputs,
    _open_production_inputs,
    realize_solve_camera,
    sha256_file,
)
from tac.witness_dsl.basis_control import genuine_frame_compact_shearlet_config

SCHEMA = "ddm_dm4_targeted_realization_cures.v1"
ROW_SCHEMA = "ddm_dm4_targeted_realization_row.v1"
CONFIG_SCHEMA = "ddm_dm4_targeted_realization_cures_config.v1"
CHECKPOINT_SCHEMA = "ddm_dm4_targeted_realization_checkpoint.v1"
GLOBAL_TAIL_ROWS = (1, 6, 15, 19, 24)
POSE_HARM_ROWS = (5, 10, 11, 12, 23)
TARGETED_ROWS = tuple(sorted((*GLOBAL_TAIL_ROWS, *POSE_HARM_ROWS)))
TARGETED_PAIRS = (16, 38, 55, 60, 90, 327, 446, 523)
CURVELET_FAMILY = "literal_polar_curvelet"
SHEARLET_FAMILY = "compact_shearlet"
STEM_STRIDE = 2
NAIVE_MENU_LABEL = "[naive-menu upper bound]"
_REPO = Path(__file__).resolve().parents[3]


class DM4RealizationError(DM2RealizationError):
    """Raised on DM4 custody, geometry, parseback, or authority drift."""


@dataclass(frozen=True)
class PairState:
    pair_id: int
    base_planes: np.ndarray
    target_planes: np.ndarray
    base_camera: np.ndarray
    target_camera: np.ndarray
    base_logits: np.ndarray
    target_logits: np.ndarray
    base_pose: np.ndarray
    gt_pose: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class FrameLibrary:
    envelopes: Mapping[str, np.ndarray]
    atom_counts: Mapping[str, int]
    custody: Mapping[str, Any]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DM4RealizationError(f"refusing to overwrite unequal artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_config(path: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    if config.get("schema") != CONFIG_SCHEMA:
        raise DM4RealizationError("DM4 config schema differs")
    required = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
        "torch_threads": 4,
        "row_count": 25,
        "global_tail_rows": list(GLOBAL_TAIL_ROWS),
        "pose_harm_rows": list(POSE_HARM_ROWS),
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise DM4RealizationError("DM4 false-authority or fixed-row contract differs")
    if config.get("frame_families") != [CURVELET_FAMILY, SHEARLET_FAMILY]:
        raise DM4RealizationError("DM4 genuine-frame family contract differs")
    recursive = config.get("scorer_recursive_support", {})
    if (
        recursive.get("stem_stride") != STEM_STRIDE
        or recursive.get("erf_r50_pixels") != 85.0
        or recursive.get("energy_fractions") != [0.5, 0.9]
        or recursive.get("operator_directive_utc") != "2026-07-24T14:45:16Z"
        or "never disk radii" not in recursive.get("write_support_rule", "")
    ):
        raise DM4RealizationError("DM4 scorer-recursive support contract differs")
    return config, raw


def _bound_inputs(
    config: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    SolveDiffMiningConfigV1,
]:
    for path_key, sha_key in (
        ("authority_file", "authority_sha256"),
        ("dm2_config_path", "dm2_config_sha256"),
        ("dm2_receipt_path", "dm2_receipt_sha256"),
        ("genuine_frame_proof_path", "genuine_frame_proof_sha256"),
        ("genuine_frame_receiver_path", "genuine_frame_receiver_sha256"),
        ("literal_curvelet_source_path", "literal_curvelet_source_sha256"),
        ("compact_shearlet_source_path", "compact_shearlet_source_sha256"),
    ):
        if sha256_file(config[path_key]) != config[sha_key]:
            raise DM4RealizationError(f"{path_key} SHA-256 mismatch")
    recursive = config["scorer_recursive_support"]
    for path_key, sha_key in (
        ("recursive_factorization_path", "recursive_factorization_sha256"),
        ("exact_factorization_path", "exact_factorization_sha256"),
        ("at1_findings_path", "at1_findings_sha256"),
        ("at1_receipt_path", "at1_receipt_sha256"),
        ("sn1_findings_path", "sn1_findings_sha256"),
        ("sn1_receipt_path", "sn1_receipt_sha256"),
        ("fr1_corrected_jacobian_config_path", "fr1_corrected_jacobian_config_sha256"),
    ):
        if sha256_file(recursive[path_key]) != recursive[sha_key]:
            raise DM4RealizationError(f"scorer_recursive_support.{path_key} SHA-256 mismatch")
    dm2_config = checked_json(config["dm2_config_path"], config["dm2_config_sha256"])
    dm2 = checked_json(config["dm2_receipt_path"], config["dm2_receipt_sha256"])
    proof = checked_json(config["genuine_frame_proof_path"], config["genuine_frame_proof_sha256"])
    receiver = checked_json(
        config["genuine_frame_receiver_path"],
        config["genuine_frame_receiver_sha256"],
    )
    if (
        dm2.get("schema") != "ddm_dm2_l3_realization_race.v1"
        or dm2.get("row_count") != 25
        or len(dm2.get("rows", ())) != 25
        or dm2.get("aggregate", {}).get("semantic_records_joint_exact_after_composition") is not True
        or dm2.get("score_claim") is not False
        or dm2.get("pointer_moved") is not False
    ):
        raise DM4RealizationError("DM2 receipt authority contract differs")
    dm1, index_receipt, source_config = bind_dm2_inputs(dm2_config)
    if (
        proof.get("schema") != "genuine_frame_nterm_probe.v4"
        or proof.get("status") != "COMPLETE"
        or proof.get("score_claim") is not False
        or proof.get("pointer_moved") is not False
    ):
        raise DM4RealizationError("genuine-frame structural proof contract differs")
    for family in ("windowed_curvelet", SHEARLET_FAMILY):
        row = proof.get("structural_proof", {}).get(family, {})
        if (
            row.get("proof", {}).get("passed") is not True
            or row.get("proof", {}).get("spatially_localized") is not True
            or row.get("metadata", {}).get("feature_width") != 80
        ):
            raise DM4RealizationError(f"#502 {family} structural proof is not admissible")
    measurement = receiver.get("receiver_measurement", {})
    if (
        receiver.get("score_claim") is not False
        or receiver.get("pointer_moved") is not False
        or measurement.get("config", {}).get("batch_size") != 32
        or "REALIZED-through-R" not in measurement.get("authority_label", "")
    ):
        raise DM4RealizationError("#502 receiver advisory custody differs")
    return dm2_config, dm2, dm1, index_receipt, source_config


def _frame_library(config: Mapping[str, Any]) -> FrameLibrary:
    coords = inclusive_grid_coords(*SCORER_HW)
    literal = localized_basis_features_grid_numpy(*SCORER_HW)
    if literal.shape != (SCORER_HW[0] * SCORER_HW[1], 80):
        raise DM4RealizationError("literal curvelet feature geometry differs")
    literal_envelopes = np.abs(literal[:, 4:]).astype(np.float32, copy=False)
    shearlet_config = genuine_frame_compact_shearlet_config()
    shearlet = compact_shearlet_feats(coords, shearlet_config)
    if shearlet.shape != (SCORER_HW[0] * SCORER_HW[1], 80):
        raise DM4RealizationError("compact shearlet feature geometry differs")
    half = shearlet.shape[1] // 2
    shearlet_envelopes = np.sqrt(
        np.square(shearlet[:, :half], dtype=np.float64) + np.square(shearlet[:, half:], dtype=np.float64)
    ).astype(np.float32)
    if (
        not np.all(np.isfinite(literal_envelopes))
        or not np.all(np.isfinite(shearlet_envelopes))
        or not np.any(literal_envelopes > 0)
        or not np.any(shearlet_envelopes > 0)
    ):
        raise DM4RealizationError("genuine-frame envelopes are empty or nonfinite")
    custody = {
        CURVELET_FAMILY: {
            "source_path": config["literal_curvelet_source_path"],
            "source_sha256": config["literal_curvelet_source_sha256"],
            "atom_spec_sha256": LITERAL_CURVELET_ATOM_SPEC_SHA256,
            "directional_atom_count": literal_envelopes.shape[1],
            "spatially_localized": True,
            "fourier_disguise": False,
        },
        SHEARLET_FAMILY: {
            "source_path": config["compact_shearlet_source_path"],
            "source_sha256": config["compact_shearlet_source_sha256"],
            "directional_atom_count": shearlet_envelopes.shape[1],
            "spatially_localized": True,
            "shear_steered": True,
            "fourier_disguise": False,
        },
    }
    return FrameLibrary(
        envelopes={
            CURVELET_FAMILY: literal_envelopes,
            SHEARLET_FAMILY: shearlet_envelopes,
        },
        atom_counts={
            CURVELET_FAMILY: literal_envelopes.shape[1],
            SHEARLET_FAMILY: shearlet_envelopes.shape[1],
        },
        custody=custody,
    )


def _load_pair_state(
    *,
    pair_id: int,
    context: Any,
    source_config: SolveDiffMiningConfigV1,
    kernel: FullResizeKernel,
    segnet: Any,
    posenet: Any,
) -> PairState:
    chunk = _load_production_inputs(context, source_config, [pair_id], kernel)
    base_planes = chunk.predictor_planes[0]
    target_planes = chunk.solved_planes[0]
    base_camera = np.stack([realize_solve_camera(base_planes[index], kernel) for index in range(2)])
    target_camera = np.stack([realize_solve_camera(target_planes[index], kernel) for index in range(2)])
    return PairState(
        pair_id=pair_id,
        base_planes=base_planes,
        target_planes=target_planes,
        base_camera=base_camera,
        target_camera=target_camera,
        base_logits=_seg_forward(segnet, base_camera),
        target_logits=_seg_forward(segnet, target_camera),
        base_pose=_pose_forward(posenet, base_camera),
        gt_pose=np.asarray(chunk.poses[0], dtype=np.float64),
        labels=np.asarray(chunk.labels[0], dtype=np.uint8),
    )


def _row_support(
    *,
    row: Mapping[str, Any],
    event_index: Any,
    index_receipt: Mapping[str, Any],
) -> np.ndarray:
    bucket_id = str(row["bucket_id"])
    array_key = index_receipt.get("bucket_arrays", {}).get(bucket_id)
    if not isinstance(array_key, str):
        raise DM4RealizationError(f"PF2 receipt lacks bucket mapping for {bucket_id}")
    support = _event_support(event_index, bucket_id, array_key, int(row["pair_id"]))
    digest = sha256(np.asarray(support, dtype="<u4").tobytes()).hexdigest()
    if digest != row["support"]["sha256_uint32le"]:
        raise DM4RealizationError("PF2 support SHA differs from DM1")
    return np.asarray(support, dtype=np.uint32)


def _semantic_context(
    row: Mapping[str, Any], state: PairState, support: np.ndarray
) -> tuple[int, int, tuple[bytes, bytes], np.ndarray, np.ndarray]:
    left_name, right_name = _class_pair_from_bucket(str(row["bucket_id"]))
    left, right = _class_id(left_name), _class_id(right_name)
    expected = _expected_record(row, state.target_logits, support)
    target_classes = np.argmax(state.target_logits.reshape(5, -1)[:, support], axis=0).astype(np.int64)
    base_flat = state.base_logits.reshape(5, -1)[:, support]
    base_margins = []
    fisher = []
    for column, target_class in enumerate(target_classes):
        values = base_flat[:, column]
        rival = float(np.max(np.delete(values, int(target_class))))
        margin = float(values[int(target_class)] - rival)
        base_margins.append(margin)
        fisher.append(0.5 / math.cosh(0.5 * margin) ** 2)
    return (
        left,
        right,
        expected,
        target_classes,
        np.asarray(fisher, dtype=np.float64),
    )


def _margin_objective(
    logits: np.ndarray,
    support: np.ndarray,
    target_classes: np.ndarray,
    fisher_weights: np.ndarray,
) -> float:
    flat = np.asarray(logits, dtype=np.float64).reshape(5, -1)
    total = 0.0
    for column, (site, target_class) in enumerate(zip(support, target_classes, strict=True)):
        values = flat[:, int(site)]
        rival = float(np.max(np.delete(values, int(target_class))))
        total += float(fisher_weights[column]) * (float(values[int(target_class)]) - rival)
    return total


def _fisher_margin_pullback(
    *,
    segnet: Any,
    state: PairState,
    support: np.ndarray,
    target_classes: np.ndarray,
    kernel: FullResizeKernel,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    camera = torch.from_numpy(state.base_camera[None]).permute(0, 1, 4, 2, 3).float().requires_grad_(True)
    logits = segnet(segnet.preprocess_input(camera))[0]
    flat = logits.reshape(5, -1)
    objective = torch.zeros((), dtype=logits.dtype)
    base_margins = []
    fisher_weights = []
    for site, target_class in zip(support, target_classes, strict=True):
        target = int(target_class)
        values = flat[:, int(site)]
        rival = torch.cat((values[:target], values[target + 1 :])).max()
        margin = values[target] - rival
        fisher = 0.5 / torch.cosh(0.5 * margin.detach()) ** 2
        objective = objective + fisher * margin
        base_margins.append(float(margin.detach().cpu()))
        fisher_weights.append(float(fisher.detach().cpu()))
    objective.backward()
    if camera.grad is None:
        raise DM4RealizationError("SegNet Fisher pullback produced no gradient")
    camera_gradient = camera.grad[0, FRAME_INDEX].permute(1, 2, 0).detach().cpu().numpy()
    row_indices = np.asarray([item.indices for item in kernel.operator.row_supports], dtype=np.intp)
    col_indices = np.asarray([item.indices for item in kernel.operator.col_supports], dtype=np.intp)
    plane_gradient = camera_gradient[
        row_indices[:, None, :, None],
        col_indices[None, :, None, :],
        :,
    ].sum(axis=(2, 3))
    if plane_gradient.shape != (*SCORER_HW, 3) or not np.all(np.isfinite(plane_gradient)):
        raise DM4RealizationError("projected SegNet pullback geometry differs")
    return plane_gradient.astype(np.float64), {
        "metric": "rank4 target-vs-runner SegNet head margin on categorical Fisher base",
        "fisher_formula": "0.5*sech^2(margin/2)",
        "support_count": len(support),
        "base_margin_min": min(base_margins),
        "base_margin_max": max(base_margins),
        "fisher_weight_min": min(fisher_weights),
        "fisher_weight_max": max(fisher_weights),
        "camera_gradient_l2": float(np.linalg.norm(camera_gradient)),
        "projected_plane_gradient_l2": float(np.linalg.norm(plane_gradient)),
        "projected_input_adjoint": ("exact sum over the canonical disjoint factor2 preimage taps"),
        "euclidean_control_only": True,
    }


def _stem_blocks_from_support(support: np.ndarray) -> np.ndarray:
    height, width = SCORER_HW
    flat = np.asarray(support, dtype=np.int64)
    if flat.ndim != 1 or flat.size == 0 or np.any(flat < 0) or np.any(flat >= height * width):
        raise DM4RealizationError("semantic support differs from scorer geometry")
    y, x = np.divmod(flat, width)
    block_width = width // STEM_STRIDE
    return np.unique((y // STEM_STRIDE) * block_width + x // STEM_STRIDE).astype(np.uint32)


def _stem_lattice_mask(block_indices: Sequence[int]) -> np.ndarray:
    height, width = SCORER_HW
    block_height = height // STEM_STRIDE
    block_width = width // STEM_STRIDE
    indices = np.asarray(tuple(int(value) for value in block_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= block_height * block_width):
        raise DM4RealizationError("stored stem-lattice block indices differ")
    blocks = np.zeros((block_height, block_width), dtype=bool)
    blocks.reshape(-1)[indices] = True
    return np.repeat(np.repeat(blocks, STEM_STRIDE, axis=0), STEM_STRIDE, axis=1)


def _stem_erf_universe(support: np.ndarray, erf_r50_pixels: float) -> np.ndarray:
    if not math.isfinite(erf_r50_pixels) or erf_r50_pixels <= 0:
        raise DM4RealizationError("ERF r50 must be positive and finite")
    height, width = SCORER_HW
    block_height = height // STEM_STRIDE
    block_width = width // STEM_STRIDE
    block_radius = math.ceil(erf_r50_pixels / STEM_STRIDE)
    universe = np.zeros((block_height, block_width), dtype=bool)
    seed_blocks = _stem_blocks_from_support(support)
    for block in seed_blocks:
        y, x = divmod(int(block), block_width)
        universe[
            max(0, y - block_radius) : min(block_height, y + block_radius + 1),
            max(0, x - block_radius) : min(block_width, x + block_radius + 1),
        ] = True
    return universe


def _scorer_recursive_write_support(
    *,
    plane_gradient: np.ndarray,
    support: np.ndarray,
    energy_fraction: float,
    erf_r50_pixels: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive content-selected write support on the scorer's stem lattice.

    The exact shared-resize adjoint supplies the plane gradient.  Energy is
    pooled only on the frozen SegNet stride-2 stem lattice, then ranked inside
    the registered empirical ERF-r50 rectangle around the demanded sites.
    No Euclidean disk or history heuristic participates.
    """

    if plane_gradient.shape != (*SCORER_HW, 3) or not np.all(np.isfinite(plane_gradient)):
        raise DM4RealizationError("scorer-recursive gradient geometry differs")
    if not 0.0 < energy_fraction <= 1.0:
        raise DM4RealizationError("scorer-recursive energy fraction differs")
    height, width = SCORER_HW
    block_height = height // STEM_STRIDE
    block_width = width // STEM_STRIDE
    energy = (
        np.square(plane_gradient, dtype=np.float64)
        .reshape(
            block_height,
            STEM_STRIDE,
            block_width,
            STEM_STRIDE,
            3,
        )
        .sum(axis=(1, 3, 4))
    )
    universe = _stem_erf_universe(support, erf_r50_pixels)
    mandatory = _stem_blocks_from_support(support)
    eligible = np.flatnonzero(universe.reshape(-1))
    total_energy = float(energy.reshape(-1)[eligible].sum(dtype=np.float64))
    target_energy = energy_fraction * total_energy
    selected: set[int] = {int(value) for value in mandatory}
    captured = float(energy.reshape(-1)[mandatory].sum(dtype=np.float64))
    ranked = sorted(
        (int(value) for value in eligible if int(value) not in selected),
        key=lambda value: (-float(energy.reshape(-1)[value]), value),
    )
    for value in ranked:
        if captured >= target_energy:
            break
        selected.add(value)
        captured += float(energy.reshape(-1)[value])
    selected_array = np.asarray(sorted(selected), dtype="<u4")
    mask = _stem_lattice_mask(selected_array)
    return mask, {
        "schema": "ddm_dm4_scorer_recursive_write_support.v1",
        "construction": (
            "exact shared-resize adjoint Fisher energy -> measured ERF-r50 "
            "rectangle -> stride-2 SegNet stem lattice ranking"
        ),
        "energy_fraction": energy_fraction,
        "erf_r50_pixels": erf_r50_pixels,
        "erf_r50_epistemic_label": "MEASURED_REGISTERED_MEDIAN_ROUNDED",
        "stem_stride": STEM_STRIDE,
        "eligible_stem_blocks": int(eligible.size),
        "selected_stem_blocks": int(selected_array.size),
        "selected_scorer_cells": int(np.count_nonzero(mask)),
        "captured_energy_fraction": captured / total_energy if total_energy > 0.0 else 1.0,
        "stem_block_indices": selected_array.tolist(),
        "stem_block_indices_sha256_uint32le": sha256(selected_array.tobytes()).hexdigest(),
        "support_seed_blocks": int(mandatory.size),
        "support_rule": "scorer-recursive; no disks, global writes, or history",
    }


def _stem_erf_write_support(
    support: np.ndarray,
    *,
    erf_r50_pixels: float,
    multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise DM4RealizationError("ERF repair multiplier differs")
    effective_r50 = erf_r50_pixels * multiplier
    universe = _stem_erf_universe(support, effective_r50)
    indices = np.flatnonzero(universe.reshape(-1)).astype("<u4")
    mask = _stem_lattice_mask(indices)
    return mask, {
        "schema": "ddm_dm4_stem_erf_repair_support.v1",
        "erf_r50_pixels": erf_r50_pixels,
        "multiplier": multiplier,
        "effective_r50_pixels": effective_r50,
        "stem_stride": STEM_STRIDE,
        "stem_block_count": int(indices.size),
        "stem_block_indices_sha256_uint32le": sha256(indices.tobytes()).hexdigest(),
        "support_rule": "measured ERF rectangle lifted through stride-2 stem lattice; no disk radius",
    }


def _mask_from_descriptor(
    descriptor: Mapping[str, Any],
    frames: FrameLibrary,
    support: np.ndarray,
) -> np.ndarray:
    family = str(descriptor["family"])
    envelopes = frames.envelopes[family]
    threshold = float(descriptor["threshold_fraction"])
    atoms = tuple(int(value) for value in descriptor["atom_indices"])
    if not atoms or any(index < 0 or index >= envelopes.shape[1] for index in atoms):
        raise DM4RealizationError("frame-mask atom indices differ")
    frame_mask = np.zeros(SCORER_HW[0] * SCORER_HW[1], dtype=bool)
    for atom_index in atoms:
        envelope = envelopes[:, atom_index]
        cutoff = threshold * float(np.max(envelope, initial=0.0))
        frame_mask |= envelope >= cutoff
    lattice_mask = _stem_lattice_mask(descriptor["scorer_recursive_write_support"]["stem_block_indices"]).reshape(-1)
    mask = frame_mask & lattice_mask
    mask[support] = True
    return mask.reshape(SCORER_HW)


def _plane_from_descriptor(
    *,
    descriptor: Mapping[str, Any],
    state: PairState,
    support: np.ndarray,
    frames: FrameLibrary,
    old_selected: Mapping[str, Any],
) -> np.ndarray:
    mechanism = str(descriptor["mechanism"])
    if mechanism == "dm2_control":
        return candidate_scorer_plane(
            state.base_planes[FRAME_INDEX],
            state.target_planes[FRAME_INDEX],
            support,
            scope=str(old_selected["scope"]),
            radius=old_selected["radius"],
            quantum=old_selected["quantum"],
        )
    if mechanism == "scorer_recursive_target":
        mask = _stem_lattice_mask(descriptor["scorer_recursive_write_support"]["stem_block_indices"])
        base = state.base_planes[FRAME_INDEX]
        delta = state.target_planes[FRAME_INDEX].astype(np.int16) - base.astype(np.int16)
        quantum = descriptor.get("quantum")
        if quantum is not None:
            delta = np.clip(delta, -int(quantum), int(quantum))
        updated = base.astype(np.int16)
        updated[mask] += delta[mask]
        return np.clip(updated, 0, 255).astype(np.uint8)
    if mechanism in {"frame_target_secant_qp", "frame_fisher_adjoint"}:
        mask = _mask_from_descriptor(descriptor, frames, support)
        base = state.base_planes[FRAME_INDEX]
        if mechanism == "frame_target_secant_qp":
            output = base.copy()
            delta = state.target_planes[FRAME_INDEX].astype(np.int16) - base.astype(np.int16)
            quantum = descriptor.get("quantum")
            if quantum is not None:
                delta = np.clip(delta, -int(quantum), int(quantum))
            updated = base.astype(np.int16)
            updated[mask] += delta[mask]
            return np.clip(updated, 0, 255).astype(np.uint8)
        signed_direction = np.asarray(descriptor["signed_direction"], dtype=np.int8)
        if signed_direction.shape != (3,) or np.any(~np.isin(signed_direction, (-1, 0, 1))):
            raise DM4RealizationError("frame-adjoint signed direction differs")
        quantum = int(descriptor["quantum"])
        updated = base.astype(np.int16)
        updated[mask] += quantum * signed_direction[None, :]
        return np.clip(updated, 0, 255).astype(np.uint8)
    if mechanism == "pose_se3_null":
        parent = _plane_from_descriptor(
            descriptor=descriptor["parent"],
            state=state,
            support=support,
            frames=frames,
            old_selected=old_selected,
        )
        delta = parent.astype(np.int16) - state.base_planes[FRAME_INDEX].astype(np.int16)
        coefficients = np.asarray(descriptor["rgb_coefficients"], dtype=np.float64)
        if (
            coefficients.shape != (3,)
            or not np.all(np.isfinite(coefficients))
            or np.any(coefficients < 0.0)
            or np.any(coefficients > 1.0)
        ):
            raise DM4RealizationError("pose-null RGB coefficients differ")
        output = state.base_planes[FRAME_INDEX].astype(np.float64)
        output += delta.astype(np.float64) * coefficients[None, None, :]
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)
    raise DM4RealizationError(f"unknown DM4 mechanism {mechanism!r}")


def _evaluate_candidate(
    *,
    row: Mapping[str, Any],
    state: PairState,
    support: np.ndarray,
    descriptor: Mapping[str, Any],
    plane: np.ndarray,
    segnet: Any,
    posenet: Any,
    config: Mapping[str, Any],
    left: int,
    right: int,
    expected: tuple[bytes, bytes],
    target_classes: np.ndarray,
    fisher_weights: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    camera = state.base_camera.copy()
    camera[FRAME_INDEX] = realize_solve_camera(plane, FullResizeKernel.build())
    logits = _seg_forward(segnet, camera)
    observed = _winner_symbols(logits, support, left, right)[:2]
    semantic_exact = observed == expected
    record = RGBDeltaRecord.from_frames(int(row["pair_id"]), state.base_camera[FRAME_INDEX], camera[FRAME_INDEX])
    raw = record.encode()
    prices, winner = price_rgb_raw(raw)
    candidate_cells = np.argmax(logits, axis=0).astype(np.uint8)
    base_cells = np.argmax(state.base_logits, axis=0).astype(np.uint8)
    labels = state.labels
    support_mask = np.zeros(SCORER_HW, dtype=bool)
    support_mask.reshape(-1)[support] = True
    outside = ~support_mask
    base_correct = base_cells == labels
    candidate_correct = candidate_cells == labels
    changed = candidate_cells != base_cells
    pose = _pose_forward(posenet, camera)
    base_pose_sse = float(np.square(state.base_pose - state.gt_pose).sum(dtype=np.float64))
    pose_sse = float(np.square(pose - state.gt_pose).sum(dtype=np.float64))
    delta_pose_sse = pose_sse - base_pose_sse
    delta_errors = int(np.count_nonzero(~candidate_correct) - np.count_nonzero(~base_correct))
    off_target_delta_errors = int(
        np.count_nonzero(outside & ~candidate_correct) - np.count_nonzero(outside & ~base_correct)
    )
    exact_bytes = int(prices[winner]["container_bytes"])
    score = _score_delta(
        delta_errors=delta_errors,
        delta_pose_sse=delta_pose_sse,
        realized_bytes=exact_bytes,
        config=config,
    )
    collateral_seg_score = 100.0 * off_target_delta_errors / int(config["global_seg_sites"])
    collateral_score = collateral_seg_score + score["pose_score_delta"]
    collateral_bytes = max(0.0, collateral_score) * int(config["source_video_bytes"]) / 25.0
    pose_safety_bytes = max(0.0, score["pose_score_delta"]) * int(config["source_video_bytes"]) / 25.0
    delta = camera[FRAME_INDEX].astype(np.int16) - state.base_camera[FRAME_INDEX].astype(np.int16)
    outcome = {
        "schema": ROW_SCHEMA,
        "row_index": int(row["row_index"]),
        "pair_id": int(row["pair_id"]),
        "bucket_id": row["bucket_id"],
        "stream_type": row["adjudicated_typed_home"]["type"],
        "stratum": row["stratum"],
        "semantic_bytes_dm1": int(row["exact_counted_bytes"]),
        "semantic_record_sha256": row["semantic_record"]["raw_sha256"],
        "support_count_n": len(support),
        "support_sha256_uint32le": row["support"]["sha256_uint32le"],
        "selected_candidate": dict(descriptor),
        "semantic_record_exact": semantic_exact,
        "observed_winners_sha256": sha256(observed[0]).hexdigest(),
        "observed_margin_relations_sha256": sha256(observed[1]).hexdigest(),
        "fisher_margin_objective": _margin_objective(logits, support, target_classes, fisher_weights),
        "rgb_record": {
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "parseback_exact": record.apply(state.base_camera[FRAME_INDEX]).tobytes() == camera[FRAME_INDEX].tobytes(),
            "prices": prices,
            "winning_codec": winner,
            "exact_counted_bytes": exact_bytes,
            "changed_rgb_pixels": len(record.flat_indices),
            "changed_channel_values": int(np.count_nonzero(delta)),
            "l1_rgb_delta": int(np.abs(delta).sum(dtype=np.int64)),
            "l2_rgb_delta_euclidean_control": float(np.sqrt(np.square(delta, dtype=np.float64).sum())),
        },
        "seg": {
            "delta_errors_exact_argmax": delta_errors,
            "delta_d_seg": score["delta_d_seg"],
            "metric": "exact frozen SegNet last-frame argmax contest units",
            "aiming_metric": "rank4 margin/categorical-Fisher pullback",
            "euclidean_is_control_only": True,
        },
        "pose": {
            "base_pose6": state.base_pose.tolist(),
            "candidate_pose6": pose.tolist(),
            "gt_pose6": state.gt_pose.tolist(),
            "delta_xi6": (pose - state.base_pose).tolist(),
            "base_pair_mse": base_pose_sse / 6.0,
            "candidate_pair_mse": pose_sse / 6.0,
            "delta_pair_mse": delta_pose_sse / 6.0,
            "pose_nonharm": delta_pose_sse <= 0.0,
            "metric": "exact frozen PoseNet first-six-output MSE",
            "safety_price_bytes_at_rate_dual": pose_safety_bytes,
        },
        "collateral": {
            "off_target_argmax_flips": int(np.count_nonzero(outside & changed)),
            "harmful_off_target_flips": int(np.count_nonzero(outside & base_correct & ~candidate_correct)),
            "helpful_off_target_flips": int(np.count_nonzero(outside & ~base_correct & candidate_correct)),
            "neutral_wrong_to_wrong_off_target_flips": int(
                np.count_nonzero(outside & changed & ~base_correct & ~candidate_correct)
            ),
            "off_target_delta_errors": off_target_delta_errors,
            "seg_score_delta": collateral_seg_score,
            "pose_score_delta": score["pose_score_delta"],
            "joint_collateral_score_delta": collateral_score,
            "positive_collateral_byte_equivalent_at_rate_dual": collateral_bytes,
        },
        "joint_score_accounting": score,
        "effective_realized_plus_positive_collateral_bytes": (exact_bytes + collateral_bytes),
        "realization_status": (
            "SUCCESS_EXACT_L4_RECORD_THROUGH_L3_RGB" if semantic_exact else "FAILED_EXACT_L4_RECORD"
        ),
        "evidence_axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
    }
    return outcome, logits


def _candidate_rank(outcome: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(outcome["semantic_record_exact"]),
        float(outcome["effective_realized_plus_positive_collateral_bytes"]),
        int(outcome["rgb_record"]["exact_counted_bytes"]),
        int(outcome["rgb_record"]["changed_rgb_pixels"]),
        str(outcome["selected_candidate"]["candidate_id"]),
    )


def _measure_global_tail(
    *,
    row: Mapping[str, Any],
    state: PairState,
    support: np.ndarray,
    old_row: Mapping[str, Any],
    frames: FrameLibrary,
    kernel: FullResizeKernel,
    segnet: Any,
    posenet: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    left, right, expected, target_classes, fisher_weights = _semantic_context(row, state, support)
    plane_gradient, pullback = _fisher_margin_pullback(
        segnet=segnet,
        state=state,
        support=support,
        target_classes=target_classes,
        kernel=kernel,
    )
    recursive_config = config["scorer_recursive_support"]
    recursive_supports = []
    for energy_fraction in recursive_config["energy_fractions"]:
        _mask, support_receipt = _scorer_recursive_write_support(
            plane_gradient=plane_gradient,
            support=support,
            energy_fraction=float(energy_fraction),
            erf_r50_pixels=float(recursive_config["erf_r50_pixels"]),
        )
        recursive_supports.append(support_receipt)
    base_objective = _margin_objective(state.base_logits, support, target_classes, fisher_weights)
    target_delta = state.target_planes[FRAME_INDEX].astype(np.int16) - state.base_planes[FRAME_INDEX].astype(np.int16)
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    probes: list[dict[str, Any]] = []
    seen_planes: set[str] = set()

    def evaluate(descriptor: dict[str, Any], plane: np.ndarray) -> dict[str, Any]:
        digest = sha256(plane.tobytes()).hexdigest()
        if digest in seen_planes:
            return {
                "candidate_id": descriptor["candidate_id"],
                "status": "DUPLICATE_PLANE_SKIPPED",
                "plane_sha256": digest,
            }
        seen_planes.add(digest)
        outcome, _logits = _evaluate_candidate(
            row=row,
            state=state,
            support=support,
            descriptor=descriptor,
            plane=plane,
            segnet=segnet,
            posenet=posenet,
            config=config,
            left=left,
            right=right,
            expected=expected,
            target_classes=target_classes,
            fisher_weights=fisher_weights,
        )
        candidates.append((outcome, plane))
        probe = {
            "candidate_id": descriptor["candidate_id"],
            "mechanism": descriptor["mechanism"],
            "semantic_record_exact": outcome["semantic_record_exact"],
            "exact_counted_bytes": outcome["rgb_record"]["exact_counted_bytes"],
            "fisher_margin_objective": outcome["fisher_margin_objective"],
            "exact_secant_gain": outcome["fisher_margin_objective"] - base_objective,
            "delta_errors_exact_argmax": outcome["seg"]["delta_errors_exact_argmax"],
            "delta_pose_mse": outcome["pose"]["delta_pair_mse"],
            "pose_nonharm": outcome["pose"]["pose_nonharm"],
            "plane_sha256": digest,
        }
        probes.append(probe)
        return probe

    old_selected = old_row["selected_candidate"]
    control_descriptor = {
        "candidate_id": f"dm2_control_{old_selected['candidate_id']}",
        "mechanism": "dm2_control",
        "scope_class": old_selected["scope"],
        "application_stage": old_selected["application_stage"],
        "quantization_policy": old_selected["quantization_policy"],
        "authority_label": NAIVE_MENU_LABEL,
        "scorer_recursive_replacement": (
            "exact resize-adjoint Fisher energy x measured ERF-r50 x stride-2 stem lattice x genuine #502 frame support"
        ),
    }
    control_plane = _plane_from_descriptor(
        descriptor=control_descriptor,
        state=state,
        support=support,
        frames=frames,
        old_selected=old_selected,
    )
    evaluate(control_descriptor, control_plane)

    thresholds = tuple(float(value) for value in config["frame_threshold_fractions"])
    preselect = int(config["frame_preselect_atoms"])
    union_sizes = tuple(int(value) for value in config["frame_union_sizes"])
    quantum_ladder = tuple(int(value) for value in config["fixed_quantum_ladder"])
    adjoint_quantum_ladder = tuple(int(value) for value in config["adjoint_quantum_ladder"])
    signed_global = np.sign(np.sum(plane_gradient, axis=(0, 1))).astype(np.int8)
    if not np.any(signed_global):
        signed_global = np.sign(np.sum(np.abs(plane_gradient), axis=(0, 1))).astype(np.int8)

    for family in (CURVELET_FAMILY, SHEARLET_FAMILY):
        envelopes = frames.envelopes[family]
        for threshold, support_receipt in (
            (threshold, support_receipt) for threshold in thresholds for support_receipt in recursive_supports
        ):
            energy_fraction = float(support_receipt["energy_fraction"])
            scored: list[tuple[float, float, int, int]] = []
            masks: dict[int, np.ndarray] = {}
            for atom_index in range(envelopes.shape[1]):
                descriptor = {
                    "family": family,
                    "threshold_fraction": threshold,
                    "atom_indices": [atom_index],
                    "scorer_recursive_write_support": support_receipt,
                }
                mask = _mask_from_descriptor(descriptor, frames, support)
                masks[atom_index] = mask
                first_order = float(np.sum(plane_gradient * target_delta.astype(np.float64) * mask[:, :, None]))
                adjoint_mass = float(np.sum(np.abs(plane_gradient) * mask[:, :, None]))
                changed = max(
                    1,
                    int(np.count_nonzero(target_delta.astype(np.int16) * mask[:, :, None])),
                )
                scored.append(
                    (
                        first_order / changed,
                        adjoint_mass / changed,
                        atom_index,
                        changed,
                    )
                )
            by_target = sorted(scored, key=lambda item: (item[0], item[1], -item[2]), reverse=True)[:preselect]
            by_adjoint = sorted(scored, key=lambda item: (item[1], item[0], -item[2]), reverse=True)[:preselect]
            selected_atoms = sorted({item[2] for item in (*by_target, *by_adjoint)})
            secants: list[tuple[float, int]] = []
            for atom_index in selected_atoms:
                descriptor = {
                    "candidate_id": (f"{family}_erf{energy_fraction:g}_t{threshold:g}_atom{atom_index}_target"),
                    "mechanism": "frame_target_secant_qp",
                    "family": family,
                    "threshold_fraction": threshold,
                    "atom_indices": [atom_index],
                    "scorer_recursive_write_support": support_receipt,
                    "quantum": None,
                    "scope_class": "local",
                    "application_stage": (
                        "genuine #502 frame support at scorer plane -> canonical factor2 preimage -> exact R/uint8"
                    ),
                    "quantization_policy": "exact target substitution for secant",
                }
                plane = _plane_from_descriptor(
                    descriptor=descriptor,
                    state=state,
                    support=support,
                    frames=frames,
                    old_selected=old_selected,
                )
                probe = evaluate(descriptor, plane)
                if probe.get("status") == "DUPLICATE_PLANE_SKIPPED":
                    continue
                secants.append(
                    (
                        float(probe["exact_secant_gain"]) / max(1, int(probe["exact_counted_bytes"])),
                        atom_index,
                    )
                )
            exact_order = [
                atom_index for _gain, atom_index in sorted(secants, key=lambda item: (item[0], -item[1]), reverse=True)
            ]
            for union_size in union_sizes:
                atom_indices = exact_order[: min(union_size, len(exact_order))]
                if not atom_indices:
                    continue
                base_descriptor = {
                    "family": family,
                    "threshold_fraction": threshold,
                    "atom_indices": atom_indices,
                    "scorer_recursive_write_support": support_receipt,
                    "scope_class": "local",
                    "application_stage": (
                        "exact resize-adjoint Fisher energy -> measured ERF-r50 "
                        "stride-2 stem blocks -> genuine #502 frame support -> "
                        "canonical factor2 preimage -> exact R/uint8"
                    ),
                }
                exact_descriptor = {
                    **base_descriptor,
                    "candidate_id": (f"{family}_erf{energy_fraction:g}_t{threshold:g}_k{len(atom_indices)}_target"),
                    "mechanism": "frame_target_secant_qp",
                    "quantum": None,
                    "quantization_policy": ("exact target substitution after corrected secant ordering"),
                }
                exact_plane = _plane_from_descriptor(
                    descriptor=exact_descriptor,
                    state=state,
                    support=support,
                    frames=frames,
                    old_selected=old_selected,
                )
                exact_probe = evaluate(exact_descriptor, exact_plane)
                if exact_probe.get("semantic_record_exact") is True:
                    for quantum in quantum_ladder:
                        descriptor = {
                            **base_descriptor,
                            "candidate_id": (
                                f"{family}_erf{energy_fraction:g}_t{threshold:g}_k{len(atom_indices)}_target_q{quantum}"
                            ),
                            "mechanism": "frame_target_secant_qp",
                            "quantum": quantum,
                            "quantization_policy": ("fixed integer quantum; 1D hard-crossing QP menu"),
                        }
                        plane = _plane_from_descriptor(
                            descriptor=descriptor,
                            state=state,
                            support=support,
                            frames=frames,
                            old_selected=old_selected,
                        )
                        probe = evaluate(descriptor, plane)
                        if probe.get("semantic_record_exact") is True:
                            break
                for quantum in adjoint_quantum_ladder:
                    descriptor = {
                        **base_descriptor,
                        "candidate_id": (
                            f"{family}_erf{energy_fraction:g}_t{threshold:g}_k{len(atom_indices)}_adjoint_q{quantum}"
                        ),
                        "mechanism": "frame_fisher_adjoint",
                        "quantum": quantum,
                        "signed_direction": signed_global.tolist(),
                        "quantization_policy": ("fixed signed uint8 quantum on measured Fisher pullback"),
                    }
                    plane = _plane_from_descriptor(
                        descriptor=descriptor,
                        state=state,
                        support=support,
                        frames=frames,
                        old_selected=old_selected,
                    )
                    probe = evaluate(descriptor, plane)
                    if probe.get("semantic_record_exact") is True:
                        break
            all_descriptor = {
                "candidate_id": f"{family}_erf{energy_fraction:g}_t{threshold:g}_all_target",
                "mechanism": "frame_target_secant_qp",
                "family": family,
                "threshold_fraction": threshold,
                "atom_indices": list(range(envelopes.shape[1])),
                "scorer_recursive_write_support": support_receipt,
                "quantum": None,
                "scope_class": "local",
                "application_stage": (
                    "measured ERF-r50 stride-2 stem blocks intersect union of "
                    "every genuine #502 localized frame atom -> canonical "
                    "factor2 preimage -> exact R/uint8"
                ),
                "quantization_policy": "exact target substitution scorer-recursive broad control",
            }
            all_plane = _plane_from_descriptor(
                descriptor=all_descriptor,
                state=state,
                support=support,
                frames=frames,
                old_selected=old_selected,
            )
            evaluate(all_descriptor, all_plane)

    exact = [(outcome, plane) for outcome, plane in candidates if outcome["semantic_record_exact"]]
    if not exact:
        raise DM4RealizationError(f"DM2 positive control disappeared for global row {row['row_index']}")
    selected_outcome, selected_plane = min(exact, key=lambda item: _candidate_rank(item[0]))
    selected_outcome["search"] = {
        "candidate_count": len(probes),
        "semantic_success_count": sum(bool(probe.get("semantic_record_exact")) for probe in probes),
        "pullback": pullback,
        "scorer_recursive_write_supports": recursive_supports,
        "base_fisher_margin_objective": base_objective,
        "target_copy_first_order_gain": float(np.sum(plane_gradient * target_delta.astype(np.float64))),
        "frame_families": list(frames.envelopes),
        "genuine_frame_custody": frames.custody,
        "probes": probes,
        "old_dm2_candidate_id": old_selected["candidate_id"],
        "verdict_scope": (
            "INSTANCE x exact row x exact resize-adjoint/Fisher/ERF/stem-lattice "
            "x fixed #502 frame/secant/quantum menu; "
            "not a global minimum-preimage or frame-family verdict"
        ),
    }
    improved = int(selected_outcome["rgb_record"]["exact_counted_bytes"]) < int(
        old_row["rgb_record"]["exact_counted_bytes"]
    )
    selected_outcome["cure_disposition"] = (
        "LOCALIZED_GLOBAL_TAIL_CURED"
        if improved and selected_outcome["selected_candidate"]["scope_class"] == "local"
        else "GLOBAL_TAIL_NOT_CURED_WITHIN_FIXED_MENU"
    )
    selected_outcome["first_rung"] = (
        "Admit this localized frame-supported write to pair-level composition and "
        "remeasure exact non-telescoping survival."
        if selected_outcome["cure_disposition"] == "LOCALIZED_GLOBAL_TAIL_CURED"
        else "Residual is true-global only for this corrected-Jacobian plus #502 "
        "frame instance; next measure one receiver-closed low-rank color/normal "
        "factor on the same row without widening the family verdict."
    )
    return selected_outcome, selected_outcome["selected_candidate"], selected_plane


def _measure_pose_harm(
    *,
    row: Mapping[str, Any],
    state: PairState,
    support: np.ndarray,
    old_row: Mapping[str, Any],
    frames: FrameLibrary,
    kernel: FullResizeKernel,
    segnet: Any,
    posenet: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    left, right, expected, target_classes, fisher_weights = _semantic_context(row, state, support)
    old_selected = old_row["selected_candidate"]
    plane_gradient, pullback = _fisher_margin_pullback(
        segnet=segnet,
        state=state,
        support=support,
        target_classes=target_classes,
        kernel=kernel,
    )
    recursive_config = config["scorer_recursive_support"]
    recursive_supports = []
    for energy_fraction in recursive_config["energy_fractions"]:
        _mask, support_receipt = _scorer_recursive_write_support(
            plane_gradient=plane_gradient,
            support=support,
            energy_fraction=float(energy_fraction),
            erf_r50_pixels=float(recursive_config["erf_r50_pixels"]),
        )
        recursive_supports.append(support_receipt)
    control_descriptor = {
        "candidate_id": f"dm2_control_{old_selected['candidate_id']}",
        "mechanism": "dm2_control",
        "scope_class": old_selected["scope"],
        "application_stage": old_selected["application_stage"],
        "quantization_policy": old_selected["quantization_policy"],
        "authority_label": NAIVE_MENU_LABEL,
        "scorer_recursive_replacement": (
            "exact resize-adjoint Fisher energy x measured ERF-r50 x stride-2 stem lattice before the Pose xi6 secant"
        ),
    }
    control_plane = _plane_from_descriptor(
        descriptor=control_descriptor,
        state=state,
        support=support,
        frames=frames,
        old_selected=old_selected,
    )
    control_outcome, _ = _evaluate_candidate(
        row=row,
        state=state,
        support=support,
        descriptor=control_descriptor,
        plane=control_plane,
        segnet=segnet,
        posenet=posenet,
        config=config,
        left=left,
        right=right,
        expected=expected,
        target_classes=target_classes,
        fisher_weights=fisher_weights,
    )
    parent_candidates: list[tuple[dict[str, Any], dict[str, Any], np.ndarray]] = [
        (control_outcome, control_descriptor, control_plane)
    ]
    parent_quanta: list[int | None] = [None]
    if old_selected.get("quantum") is not None:
        parent_quanta.append(int(old_selected["quantum"]))
    for support_receipt in recursive_supports:
        energy_fraction = float(support_receipt["energy_fraction"])
        for quantum in parent_quanta:
            descriptor = {
                "candidate_id": (
                    f"scorer_recursive_erf{energy_fraction:g}_target"
                    if quantum is None
                    else f"scorer_recursive_erf{energy_fraction:g}_target_q{quantum}"
                ),
                "mechanism": "scorer_recursive_target",
                "scorer_recursive_write_support": support_receipt,
                "quantum": quantum,
                "scope_class": "local",
                "application_stage": (
                    "exact resize-adjoint Fisher energy -> measured ERF-r50 "
                    "stride-2 stem blocks -> target secant -> canonical "
                    "factor2 preimage -> exact R/uint8"
                ),
                "quantization_policy": (
                    "exact target substitution on scorer-recursive stem support"
                    if quantum is None
                    else "fixed integer quantum on scorer-recursive stem support"
                ),
            }
            plane = _plane_from_descriptor(
                descriptor=descriptor,
                state=state,
                support=support,
                frames=frames,
                old_selected=old_selected,
            )
            outcome, _ = _evaluate_candidate(
                row=row,
                state=state,
                support=support,
                descriptor=descriptor,
                plane=plane,
                segnet=segnet,
                posenet=posenet,
                config=config,
                left=left,
                right=right,
                expected=expected,
                target_classes=target_classes,
                fisher_weights=fisher_weights,
            )
            parent_candidates.append((outcome, descriptor, plane))
    exact_recursive_parents = [
        item
        for item in parent_candidates
        if item[0]["semantic_record_exact"] and item[1]["mechanism"] == "scorer_recursive_target"
    ]
    if exact_recursive_parents:
        _parent_outcome, parent_descriptor, parent_plane = min(
            exact_recursive_parents,
            key=lambda item: _candidate_rank(item[0]),
        )
        parent_authority = "SCORER_RECURSIVE_PARENT"
    else:
        _parent_outcome, parent_descriptor, parent_plane = parent_candidates[0]
        parent_authority = f"{NAIVE_MENU_LABEL} NO_EXACT_SCORER_RECURSIVE_PARENT"
    candidates: list[tuple[dict[str, Any], np.ndarray]] = [
        (outcome, plane) for outcome, _descriptor, plane in parent_candidates
    ]
    probes: list[dict[str, Any]] = []
    secant_columns = []
    component_receipts = []
    for channel in range(3):
        component = state.base_planes[FRAME_INDEX].copy()
        component[:, :, channel] = parent_plane[:, :, channel]
        camera = state.base_camera.copy()
        camera[FRAME_INDEX] = realize_solve_camera(component, kernel)
        pose = _pose_forward(posenet, camera)
        secant = pose - state.base_pose
        secant_columns.append(secant)
        component_receipts.append(
            {
                "channel": channel,
                "delta_xi6": secant.tolist(),
                "delta_xi6_l2": float(np.linalg.norm(secant)),
                "scorer_plane_sha256": sha256(component.tobytes()).hexdigest(),
            }
        )
    jacobian = np.stack(secant_columns, axis=1)
    error = state.base_pose - state.gt_pose
    ones = np.ones(3, dtype=np.float64)
    projector = np.eye(3) - np.linalg.pinv(jacobian) @ jacobian
    null_coefficients = np.clip(projector @ ones, 0.0, 1.0)
    coefficient_rows: list[tuple[str, np.ndarray]] = [("svd_null", null_coefficients)]
    for ridge in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        lhs = jacobian.T @ jacobian + ridge * np.eye(3)
        rhs = -jacobian.T @ error + ridge * ones
        fitted = np.clip(np.linalg.solve(lhs, rhs), 0.0, 1.0)
        coefficient_rows.append((f"qcqp_ridge_{ridge:g}", fitted))
        for blend in (0.25, 0.5, 0.75):
            coefficient_rows.append(
                (
                    f"qcqp_ridge_{ridge:g}_blend_{blend:g}",
                    np.clip(blend * ones + (1.0 - blend) * fitted, 0.0, 1.0),
                )
            )
    unique: dict[tuple[float, float, float], tuple[str, np.ndarray]] = {}
    for candidate_id, coefficients in coefficient_rows:
        key = tuple(float(value) for value in np.round(coefficients, 8))
        unique.setdefault(key, (candidate_id, coefficients))
    for candidate_id, coefficients in unique.values():
        descriptor = {
            "candidate_id": f"{parent_descriptor['candidate_id']}__{candidate_id}",
            "mechanism": "pose_se3_null",
            "parent": parent_descriptor,
            "rgb_coefficients": coefficients.tolist(),
            "scope_class": parent_descriptor["scope_class"],
            "application_stage": (
                "exact resize-adjoint/ERF/stem parent -> measured RGB-channel "
                "PoseNet xi6 secant -> fixed SVD/QCQP coefficient family -> "
                "canonical factor2 preimage -> exact R/uint8"
            ),
            "quantization_policy": "round-to-nearest exact uint8 after bounded [0,1] coefficients",
        }
        plane = _plane_from_descriptor(
            descriptor=descriptor,
            state=state,
            support=support,
            frames=frames,
            old_selected=old_selected,
        )
        outcome, _ = _evaluate_candidate(
            row=row,
            state=state,
            support=support,
            descriptor=descriptor,
            plane=plane,
            segnet=segnet,
            posenet=posenet,
            config=config,
            left=left,
            right=right,
            expected=expected,
            target_classes=target_classes,
            fisher_weights=fisher_weights,
        )
        predicted_delta_xi = jacobian @ coefficients
        actual_delta_xi = np.asarray(outcome["pose"]["delta_xi6"], dtype=np.float64)
        outcome["pose_null_receipt"] = {
            "predicted_delta_xi6": predicted_delta_xi.tolist(),
            "actual_delta_xi6": actual_delta_xi.tolist(),
            "secant_correction_l2": float(np.linalg.norm(actual_delta_xi - predicted_delta_xi)),
            "projected_null_residual_l2": float(np.linalg.norm(jacobian @ coefficients)),
        }
        candidates.append((outcome, plane))
        probes.append(
            {
                "candidate_id": candidate_id,
                "full_candidate_id": descriptor["candidate_id"],
                "rgb_coefficients": coefficients.tolist(),
                "semantic_record_exact": outcome["semantic_record_exact"],
                "pose_nonharm": outcome["pose"]["pose_nonharm"],
                "delta_pose_mse": outcome["pose"]["delta_pair_mse"],
                "safety_price_bytes_at_rate_dual": outcome["pose"]["safety_price_bytes_at_rate_dual"],
                "exact_counted_bytes": outcome["rgb_record"]["exact_counted_bytes"],
                "secant_correction_l2": outcome["pose_null_receipt"]["secant_correction_l2"],
            }
        )
    exact = [(outcome, plane) for outcome, plane in candidates if outcome["semantic_record_exact"]]
    if not exact:
        raise DM4RealizationError(f"DM2 positive control disappeared for pose row {row['row_index']}")
    nonharm = [(outcome, plane) for outcome, plane in exact if outcome["pose"]["pose_nonharm"]]
    selection_pool = nonharm if nonharm else exact
    selected_outcome, selected_plane = min(selection_pool, key=lambda item: _candidate_rank(item[0]))
    selected_outcome["search"] = {
        "candidate_count": len(candidates),
        "semantic_success_count": len(exact),
        "pose_nonharm_success_count": len(nonharm),
        "pose_se3_secant_matrix_6x3": jacobian.tolist(),
        "pose_se3_secant_rank": int(np.linalg.matrix_rank(jacobian)),
        "pose_se3_secant_singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist(),
        "component_receipts": component_receipts,
        "pullback": pullback,
        "scorer_recursive_write_supports": recursive_supports,
        "pose_parent_candidate_count": len(parent_candidates),
        "pose_parent_authority": parent_authority,
        "pose_parent_selected_candidate_id": parent_descriptor["candidate_id"],
        "pose_parent_probes": [
            {
                "candidate_id": descriptor["candidate_id"],
                "mechanism": descriptor["mechanism"],
                "authority_label": descriptor.get("authority_label"),
                "semantic_record_exact": outcome["semantic_record_exact"],
                "exact_counted_bytes": outcome["rgb_record"]["exact_counted_bytes"],
                "delta_pose_mse": outcome["pose"]["delta_pair_mse"],
            }
            for outcome, descriptor, _plane in parent_candidates
        ],
        "svd_null_coefficients": null_coefficients.tolist(),
        "probes": probes,
        "old_dm2_candidate_id": old_selected["candidate_id"],
        "verdict_scope": (
            "INSTANCE x exact row x exact resize-adjoint/ERF/stem parent x "
            "measured RGB-to-xi6 secant and fixed SVD/QCQP family; not a "
            "global Pose-nullspace certificate"
        ),
    }
    old_harm = float(old_row["pose"]["delta_pair_mse"]) > 0.0
    cured = old_harm and bool(selected_outcome["pose"]["pose_nonharm"])
    selected_outcome["cure_disposition"] = (
        "POSE_HARM_CURED_EXACT_L4" if cured else "POSE_HARM_PRICED_NOT_CURED_WITHIN_FIXED_MENU"
    )
    selected_outcome["first_rung"] = (
        "Admit this exact pose-nonharm RGB coefficient row to pair-level "
        "composition and remeasure non-telescoping survival."
        if cured
        else "Retain the exact safety price; next measure a spatially factorized "
        "xi6 secant on this same row because the global RGB-channel secant did "
        "not expose an exact pose-nonharm member."
    )
    return selected_outcome, selected_outcome["selected_candidate"], selected_plane


def _reconstruct_old_plane(
    *,
    old_row: Mapping[str, Any],
    row: Mapping[str, Any],
    state: PairState,
    support: np.ndarray,
) -> np.ndarray:
    selected = old_row["selected_candidate"]
    return candidate_scorer_plane(
        state.base_planes[FRAME_INDEX],
        state.target_planes[FRAME_INDEX],
        support,
        scope=str(selected["scope"]),
        radius=selected["radius"],
        quantum=selected["quantum"],
    )


def _all_pair_semantics_exact(
    *,
    logits: np.ndarray,
    pair_rows: Sequence[tuple[Mapping[str, Any], np.ndarray]],
    state: PairState,
) -> tuple[bool, list[int]]:
    failed = []
    for row, support in pair_rows:
        left_name, right_name = _class_pair_from_bucket(str(row["bucket_id"]))
        left, right = _class_id(left_name), _class_id(right_name)
        expected = _expected_record(row, state.target_logits, support)
        observed = _winner_symbols(logits, support, left, right)[:2]
        if observed != expected:
            failed.append(int(row["row_index"]))
    return not failed, failed


def _compose(
    *,
    dm1: Mapping[str, Any],
    dm2: Mapping[str, Any],
    states: Mapping[int, PairState],
    supports: Mapping[int, np.ndarray],
    selected_planes: Mapping[int, np.ndarray],
    kernel: FullResizeKernel,
    segnet: Any,
    posenet: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[RGBDeltaRecord]]:
    by_pair: dict[int, list[Mapping[str, Any]]] = {}
    for row in dm1["rows"]:
        by_pair.setdefault(int(row["pair_id"]), []).append(row)
    records = []
    pair_receipts = []
    total_delta_errors = 0
    total_off_target_delta_errors = 0
    total_delta_pose_sse = 0.0
    conflict_pairs = []
    fallback_pairs = []
    for pair_id, rows in sorted(by_pair.items()):
        state = states[pair_id]
        progress = np.zeros_like(state.base_planes[FRAME_INDEX], dtype=np.int16)
        plane = state.base_planes[FRAME_INDEX].copy()
        union_support = np.zeros(SCORER_HW, dtype=bool)
        pair_rows = []
        for row in rows:
            row_index = int(row["row_index"])
            candidate = selected_planes[row_index]
            movement = candidate.astype(np.int16) - state.base_planes[FRAME_INDEX].astype(np.int16)
            take = np.abs(movement) > np.abs(progress)
            plane[take] = candidate[take]
            progress[take] = movement[take]
            support = supports[row_index]
            union_support.reshape(-1)[support] = True
            pair_rows.append((row, support))
        camera = state.base_camera.copy()
        camera[FRAME_INDEX] = realize_solve_camera(plane, kernel)
        logits = _seg_forward(segnet, camera)
        initial_exact, initial_failed = _all_pair_semantics_exact(logits=logits, pair_rows=pair_rows, state=state)
        composition_mode = "farthest_same_direction_union"
        repair_support_receipt = None
        if not initial_exact:
            conflict_pairs.append(pair_id)
            combined_support = np.unique(np.concatenate([support for _row, support in pair_rows])).astype(np.uint32)
            recursive_config = config["scorer_recursive_support"]
            for multiplier in tuple(float(value) for value in config["conflict_repair_erf_multipliers"]):
                repaired = plane.copy()
                repair_mask, candidate_repair_receipt = _stem_erf_write_support(
                    combined_support,
                    erf_r50_pixels=float(recursive_config["erf_r50_pixels"]),
                    multiplier=multiplier,
                )
                repaired[repair_mask] = state.target_planes[FRAME_INDEX][repair_mask]
                repaired_camera = state.base_camera.copy()
                repaired_camera[FRAME_INDEX] = realize_solve_camera(repaired, kernel)
                repaired_logits = _seg_forward(segnet, repaired_camera)
                repaired_exact, _failed = _all_pair_semantics_exact(
                    logits=repaired_logits, pair_rows=pair_rows, state=state
                )
                if repaired_exact:
                    plane = repaired
                    camera = repaired_camera
                    logits = repaired_logits
                    repair_support_receipt = candidate_repair_receipt
                    composition_mode = "scorer_recursive_erf_stem_repair_after_union_conflict"
                    break
            else:
                plane = state.target_planes[FRAME_INDEX].copy()
                camera = state.target_camera.copy()
                logits = state.target_logits
                composition_mode = "full_solved_target_positive_control_after_targeted_repair_failure"
                fallback_pairs.append(pair_id)
        final_exact, final_failed = _all_pair_semantics_exact(logits=logits, pair_rows=pair_rows, state=state)
        if not final_exact:
            raise DM4RealizationError(f"pair {pair_id} failed exact semantics after positive control: {final_failed}")
        base_cells = np.argmax(state.base_logits, axis=0)
        candidate_cells = np.argmax(logits, axis=0)
        base_errors = base_cells != state.labels
        candidate_errors = candidate_cells != state.labels
        delta_errors = int(np.count_nonzero(candidate_errors) - np.count_nonzero(base_errors))
        off_target = ~union_support
        off_target_delta = int(
            np.count_nonzero(off_target & candidate_errors) - np.count_nonzero(off_target & base_errors)
        )
        pose = _pose_forward(posenet, camera)
        pose_delta_sse = float(
            np.square(pose - state.gt_pose).sum(dtype=np.float64)
            - np.square(state.base_pose - state.gt_pose).sum(dtype=np.float64)
        )
        record = RGBDeltaRecord.from_frames(pair_id, state.base_camera[FRAME_INDEX], camera[FRAME_INDEX])
        if record.apply(state.base_camera[FRAME_INDEX]).tobytes() != camera[FRAME_INDEX].tobytes():
            raise DM4RealizationError("joint RGB record parseback differs")
        records.append(record)
        total_delta_errors += delta_errors
        total_off_target_delta_errors += off_target_delta
        total_delta_pose_sse += pose_delta_sse
        pair_receipts.append(
            {
                "pair_id": pair_id,
                "row_indices": [int(row["row_index"]) for row in rows],
                "composition_mode": composition_mode,
                "initial_union_semantic_records_exact": initial_exact,
                "initial_failed_row_indices": initial_failed,
                "semantic_records_exact": final_exact,
                "repair_support": repair_support_receipt,
                "delta_errors": delta_errors,
                "off_target_delta_errors": off_target_delta,
                "delta_pose_sse_6d": pose_delta_sse,
                "changed_rgb_pixels": len(record.flat_indices),
                "rgb_record_raw_sha256": sha256_bytes(record.encode()),
            }
        )
    joint_raw = encode_joint_rgb_records(records)
    ordered = tuple(sorted(records, key=lambda item: (item.pair_id, item.frame_index)))
    if decode_joint_rgb_records(joint_raw) != ordered:
        raise DM4RealizationError("joint RGB record parseback differs")
    prices, winner = price_rgb_raw(joint_raw)
    joint_bytes = int(prices[winner]["container_bytes"])
    score = _score_delta(
        delta_errors=total_delta_errors,
        delta_pose_sse=total_delta_pose_sse,
        realized_bytes=joint_bytes,
        config=config,
    )
    collateral_seg = 100.0 * total_off_target_delta_errors / int(config["global_seg_sites"])
    collateral_score = collateral_seg + score["pose_score_delta"]
    collateral_bytes = max(0.0, collateral_score) * int(config["source_video_bytes"]) / 25.0
    semantic_bytes = int(config["semantic_joint_bytes"])
    old_ratio = float(dm2["aggregate"]["ratio"]["effective_bytes_per_semantic_byte"])
    effective = joint_bytes + collateral_bytes
    return {
        "semantic_records_joint_exact_after_composition": True,
        "semantic_bytes_dm1_joint": semantic_bytes,
        "realized_rgb_joint": {
            "raw_bytes": len(joint_raw),
            "raw_sha256": sha256_bytes(joint_raw),
            "record_count": len(records),
            "parseback_exact": True,
            "prices": prices,
            "winning_codec": winner,
            "exact_counted_bytes": joint_bytes,
        },
        "collateral": {
            "off_target_delta_errors": total_off_target_delta_errors,
            "seg_score_delta": collateral_seg,
            "pose_score_delta": score["pose_score_delta"],
            "joint_collateral_score_delta": collateral_score,
            "positive_collateral_byte_equivalent_at_rate_dual": collateral_bytes,
        },
        "joint_score_accounting": score,
        "ratio": {
            "realized_bytes_per_semantic_byte": joint_bytes / semantic_bytes,
            "effective_realized_plus_positive_collateral_bytes": effective,
            "effective_bytes_per_semantic_byte": effective / semantic_bytes,
            "old_dm2_effective_bytes_per_semantic_byte": old_ratio,
            "ratio_delta_vs_dm2": effective / semantic_bytes - old_ratio,
            "ratio_fraction_of_dm2": (effective / semantic_bytes) / old_ratio,
            "bound_status": (
                "CONSTRUCTIVE_UPPER_BOUND_WITH_FULL_TARGET_FALLBACK"
                if fallback_pairs
                else "MEASURED_TARGETED_CURE_COMPOSITION"
            ),
        },
        "pair_rows": pair_receipts,
        "union_conflict_pair_ids": conflict_pairs,
        "union_conflict_pair_count": len(conflict_pairs),
        "fallback_pair_ids": fallback_pairs,
        "fallback_pair_count": len(fallback_pairs),
        "old_dm2_union_conflict_pair_ids": dm2["aggregate"]["fallback_pair_ids"],
        "non_telescope_policy": (
            "all 25 selected writes were jointly composed and Seg/Pose were "
            "freshly remeasured; no independent row delta was summed"
        ),
    }, records


def _decomposition(
    *,
    rows: Sequence[Mapping[str, Any]],
    old_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old_by_index = {int(row["row_index"]): row for row in old_rows}
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    residual_global = []
    for row in rows:
        row_index = int(row["row_index"])
        old = old_by_index[row_index]
        stream = str(row["stream_type"])
        scope = str(row["selected_candidate"].get("scope_class", "local"))
        old_harm = float(old["pose"]["delta_pair_mse"]) > 0.0
        new_nonharm = bool(row["pose"]["pose_nonharm"])
        if old_harm and new_nonharm:
            pose_class = "pose_cured"
        elif new_nonharm:
            pose_class = "pose_safe"
        else:
            pose_class = "pose_priced"
        key = (stream, scope, pose_class)
        cell = cells.setdefault(
            key,
            {
                "stream_type": stream,
                "scope": scope,
                "pose_disposition": pose_class,
                "row_count": 0,
                "semantic_bytes_independent_sum": 0,
                "realized_bytes_independent_sum": 0,
                "effective_bytes_independent_sum": 0.0,
                "row_indices": [],
            },
        )
        cell["row_count"] += 1
        cell["semantic_bytes_independent_sum"] += int(row["semantic_bytes_dm1"])
        cell["realized_bytes_independent_sum"] += int(row["rgb_record"]["exact_counted_bytes"])
        cell["effective_bytes_independent_sum"] += float(row["effective_realized_plus_positive_collateral_bytes"])
        cell["row_indices"].append(row_index)
        if scope == "global":
            residual_global.append(
                {
                    "row_index": row_index,
                    "pair_id": row["pair_id"],
                    "bucket_id": row["bucket_id"],
                    "mechanism_instance": row["selected_candidate"]["candidate_id"],
                    "mechanism": row["selected_candidate"]["mechanism"],
                    "verdict_scope": (
                        "INSTANCE x exact row x fixed candidate menu; not a "
                        "curvelet, shearlet, corrected-Jacobian, or pose-null "
                        "family negative"
                    ),
                    "next_measurement": row["first_rung"],
                }
            )
    output = {}
    for key, cell in sorted(cells.items()):
        semantic = cell["semantic_bytes_independent_sum"]
        cell["realized_per_semantic_ratio_of_sums"] = cell["realized_bytes_independent_sum"] / semantic
        cell["effective_per_semantic_ratio_of_sums"] = cell["effective_bytes_independent_sum"] / semantic
        output["|".join(key)] = cell
    return output, residual_global


def _implementation_custody() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        _REPO / "tools/measure_ddm_dm4_targeted_realization_cures.py",
        _REPO / "src/tac/optimization/ddm_dm2_l3_realization_race.py",
        _REPO / "src/tac/boundary_math/localized_basis_frames.py",
        _REPO / "src/tac/boundary_math/compact_shearlet_frame.py",
        _REPO / "src/tac/scorer.py",
    )
    output = {}
    for path in paths:
        if not path.is_file():
            raise DM4RealizationError(f"implementation custody path absent: {path}")
        output[str(path.relative_to(_REPO))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def materialize(config_path: str | Path, output_dir: str | Path) -> Mapping[str, Any]:
    """Run/resume the bounded ten-row cures and fresh 25-row composition."""

    config, config_raw = _read_config(config_path)
    dm2_config, dm2, dm1, index_receipt, source_config = _bound_inputs(config)
    runtime_config = {**dm2_config, **config}
    kernel = FullResizeKernel.build()
    context = _open_production_inputs(source_config)

    import torch

    from tac.scorer import load_default_scorers

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    posenet, segnet = load_default_scorers(config["upstream_dir"], device="cpu")

    frames = _frame_library(config)
    states = {
        pair_id: _load_pair_state(
            pair_id=pair_id,
            context=context,
            source_config=source_config,
            kernel=kernel,
            segnet=segnet,
            posenet=posenet,
        )
        for pair_id in sorted({int(row["pair_id"]) for row in dm1["rows"]})
    }
    root = Path(output_dir)
    stage = root / "stage_checkpoints"
    config_sha = sha256_bytes(config_raw)
    implementation_sha = sha256_file(Path(__file__).resolve())
    old_by_index = {int(row["row_index"]): row for row in dm2["rows"]}
    supports: dict[int, np.ndarray] = {}
    targeted_outcomes: dict[int, Mapping[str, Any]] = {}
    selected_descriptors: dict[int, Mapping[str, Any]] = {}
    selected_planes: dict[int, np.ndarray] = {}

    with np.load(config["pf2_event_index_path"], allow_pickle=False) as event_index:
        for row in dm1["rows"]:
            row_index = int(row["row_index"])
            support = _row_support(row=row, event_index=event_index, index_receipt=index_receipt)
            supports[row_index] = support
            state = states[int(row["pair_id"])]
            if row_index not in TARGETED_ROWS:
                selected_planes[row_index] = _reconstruct_old_plane(
                    old_row=old_by_index[row_index],
                    row=row,
                    state=state,
                    support=support,
                )
                continue
            checkpoint_path = stage / f"row_{row_index:02d}.json"
            if checkpoint_path.is_file():
                checkpoint = json.loads(checkpoint_path.read_bytes())
                if (
                    checkpoint.get("schema") != CHECKPOINT_SCHEMA
                    or checkpoint.get("typed_config_sha256") != config_sha
                    or checkpoint.get("implementation_sha256") != implementation_sha
                    or checkpoint.get("row", {}).get("row_index") != row_index
                ):
                    raise DM4RealizationError(f"targeted row checkpoint custody differs: {checkpoint_path}")
                outcome = checkpoint["row"]
                descriptor = outcome["selected_candidate"]
                plane = _plane_from_descriptor(
                    descriptor=descriptor,
                    state=state,
                    support=support,
                    frames=frames,
                    old_selected=old_by_index[row_index]["selected_candidate"],
                )
            elif row_index in GLOBAL_TAIL_ROWS:
                outcome, descriptor, plane = _measure_global_tail(
                    row=row,
                    state=state,
                    support=support,
                    old_row=old_by_index[row_index],
                    frames=frames,
                    kernel=kernel,
                    segnet=segnet,
                    posenet=posenet,
                    config=runtime_config,
                )
                _atomic_write(
                    checkpoint_path,
                    canonical_json_bytes(
                        {
                            "schema": CHECKPOINT_SCHEMA,
                            "typed_config_sha256": config_sha,
                            "implementation_sha256": implementation_sha,
                            "row": outcome,
                        }
                    ),
                )
            else:
                outcome, descriptor, plane = _measure_pose_harm(
                    row=row,
                    state=state,
                    support=support,
                    old_row=old_by_index[row_index],
                    frames=frames,
                    kernel=kernel,
                    segnet=segnet,
                    posenet=posenet,
                    config=runtime_config,
                )
                _atomic_write(
                    checkpoint_path,
                    canonical_json_bytes(
                        {
                            "schema": CHECKPOINT_SCHEMA,
                            "typed_config_sha256": config_sha,
                            "implementation_sha256": implementation_sha,
                            "row": outcome,
                        }
                    ),
                )
            targeted_outcomes[row_index] = outcome
            selected_descriptors[row_index] = descriptor
            selected_planes[row_index] = plane

    composed_rows = []
    old_new_table = []
    for row_index in range(25):
        old = old_by_index[row_index]
        if row_index in targeted_outcomes:
            new = dict(targeted_outcomes[row_index])
        else:
            new = {
                **old,
                "semantic_record_exact": True,
                "selected_candidate": {
                    **old["selected_candidate"],
                    "mechanism": "dm2_unchanged",
                    "scope_class": old["selected_candidate"]["scope"],
                },
                "seg": {
                    "delta_errors_exact_argmax": round(
                        float(old["joint_score_accounting"]["delta_d_seg"]) * int(dm2_config["global_seg_sites"])
                    ),
                    "delta_d_seg": old["joint_score_accounting"]["delta_d_seg"],
                    "metric": "exact frozen SegNet last-frame argmax contest units",
                    "aiming_metric": "DM2 unchanged",
                    "euclidean_is_control_only": True,
                },
                "effective_realized_plus_positive_collateral_bytes": old["ratio"][
                    "effective_realized_plus_positive_collateral_bytes"
                ],
                "cure_disposition": "UNCHANGED_NOT_TARGETED",
            }
        composed_rows.append(new)
        old_new_table.append(
            {
                "row_index": row_index,
                "pair_id": new["pair_id"],
                "bucket_id": new["bucket_id"],
                "stream_type": new["stream_type"],
                "targeted": row_index in TARGETED_ROWS,
                "cure_disposition": new.get("cure_disposition", "UNCHANGED_NOT_TARGETED"),
                "old_candidate_id": old["selected_candidate"]["candidate_id"],
                "new_candidate_id": new["selected_candidate"]["candidate_id"],
                "old_scope": old["selected_candidate"]["scope"],
                "new_scope": new["selected_candidate"].get("scope_class", new["selected_candidate"].get("scope")),
                "old_realized_bytes": old["rgb_record"]["exact_counted_bytes"],
                "new_realized_bytes": new["rgb_record"]["exact_counted_bytes"],
                "delta_realized_bytes": int(new["rgb_record"]["exact_counted_bytes"])
                - int(old["rgb_record"]["exact_counted_bytes"]),
                "old_delta_pose_mse": old["pose"]["delta_pair_mse"],
                "new_delta_pose_mse": new["pose"]["delta_pair_mse"],
                "old_pose_nonharm": old["pose"]["pose_nonharm"],
                "new_pose_nonharm": new["pose"]["pose_nonharm"],
                "old_joint_score_delta": old["joint_score_accounting"]["joint_score_delta"],
                "new_joint_score_delta": new["joint_score_accounting"]["joint_score_delta"],
                "next_measurement": new["first_rung"],
            }
        )

    aggregate, _records = _compose(
        dm1=dm1,
        dm2=dm2,
        states=states,
        supports=supports,
        selected_planes=selected_planes,
        kernel=kernel,
        segnet=segnet,
        posenet=posenet,
        config=runtime_config,
    )
    decomposition, residual_global = _decomposition(rows=composed_rows, old_rows=dm2["rows"])
    result = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "source_commit": config["source_commit"],
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "row_count": len(composed_rows),
        "targeted_row_indices": list(TARGETED_ROWS),
        "global_tail_row_indices": list(GLOBAL_TAIL_ROWS),
        "pose_harm_row_indices": list(POSE_HARM_ROWS),
        "rows": composed_rows,
        "old_new_row_table": old_new_table,
        "aggregate": aggregate,
        "decomposition": {
            "SKELETON_FIBER_x_local_global_x_pose": decomposition,
            "residual_true_global_rows": residual_global,
            "residual_true_global_count": len(residual_global),
            "scope_note": (
                "true-global is mechanism-instance scoped to the fixed "
                "Fisher/secant/#502/pose-null menu; it is not a family negative"
            ),
        },
        "custody": {
            "authority_file": config["authority_file"],
            "authority_sha256": config["authority_sha256"],
            "dm2_config_path": config["dm2_config_path"],
            "dm2_config_sha256": config["dm2_config_sha256"],
            "dm2_receipt_path": config["dm2_receipt_path"],
            "dm2_receipt_sha256": config["dm2_receipt_sha256"],
            "dm1_receipt_path": dm2_config["dm1_receipt_path"],
            "dm1_receipt_sha256": dm2_config["dm1_receipt_sha256"],
            "pf2_event_index_path": config["pf2_event_index_path"],
            "pf2_event_index_sha256": config["pf2_event_index_sha256"],
            "segnet_weights_sha256": config["segnet_weights_sha256"],
            "posenet_weights_sha256": config["posenet_weights_sha256"],
            "genuine_frame_proof_path": config["genuine_frame_proof_path"],
            "genuine_frame_proof_sha256": config["genuine_frame_proof_sha256"],
            "genuine_frame_receiver_path": config["genuine_frame_receiver_path"],
            "genuine_frame_receiver_sha256": config["genuine_frame_receiver_sha256"],
            "genuine_frame_library": frames.custody,
            "scorer_recursive_support": config["scorer_recursive_support"],
            "implementation": _implementation_custody(),
            "torch_threads": 4,
            "deterministic_algorithms": True,
            "seed": int(config["seed"]),
            "checkpoint_policy": (
                "one atomic preserved JSON checkpoint per targeted row; selected "
                "plane deterministically reconstructed from typed descriptor"
            ),
            "streaming_policy": ("scorer/camera planes streamed in memory; no plane stockpile emitted"),
        },
        "verdict_scope": (
            "INSTANCE x SHA-bound 25-row DM1 demand set x fixed DM4 "
            "Fisher/secant/#502/SE3 candidate menu. Constructive upper bound only; "
            "no minimum-preimage, family, score, promotion, or frontier verdict."
        ),
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
    }
    _atomic_write(
        root / "ddm_dm4_targeted_realization_cures_receipt.json",
        canonical_json_bytes(result),
    )
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "GLOBAL_TAIL_ROWS",
    "POSE_HARM_ROWS",
    "SCHEMA",
    "TARGETED_ROWS",
    "DM4RealizationError",
    "materialize",
]
