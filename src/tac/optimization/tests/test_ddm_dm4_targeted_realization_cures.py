# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np

from tac.optimization.ddm_dm4_targeted_realization_cures import (
    CURVELET_FAMILY,
    SCORER_HW,
    STEM_STRIDE,
    FrameLibrary,
    PairState,
    _mask_from_descriptor,
    _plane_from_descriptor,
    _read_config,
    _scorer_recursive_write_support,
    _stem_blocks_from_support,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[4] / ".omx/research/configs/ddm_dm4_targeted_realization_cures_20260724.json"
    )


def test_config_binds_late_scorer_recursive_directive() -> None:
    config, _raw = _read_config(_config_path())
    recursive = config["scorer_recursive_support"]
    assert recursive["operator_directive_utc"] == "2026-07-24T14:45:16Z"
    assert recursive["stem_stride"] == STEM_STRIDE
    assert recursive["erf_r50_pixels"] == 85.0
    assert "never disk radii" in recursive["write_support_rule"]


def test_support_sites_collapse_to_stride2_stem_block() -> None:
    width = SCORER_HW[1]
    support = np.asarray([0, 1, width, width + 1], dtype=np.uint32)
    assert _stem_blocks_from_support(support).tolist() == [0]


def test_recursive_support_rejects_high_energy_outside_erf_rectangle() -> None:
    height, width = SCORER_HW
    center_y, center_x = 100, 100
    support = np.asarray([center_y * width + center_x], dtype=np.uint32)
    gradient = np.zeros((height, width, 3), dtype=np.float64)
    gradient[102:104, 102:104] = 2.0
    gradient[300:302, 300:302] = 1000.0
    mask, receipt = _scorer_recursive_write_support(
        plane_gradient=gradient,
        support=support,
        energy_fraction=0.5,
        erf_r50_pixels=8.0,
    )
    assert mask[center_y, center_x]
    assert mask[102, 102]
    assert not mask[300, 300]
    assert receipt["stem_stride"] == 2
    assert receipt["support_rule"].startswith("scorer-recursive")
    assert receipt["selected_scorer_cells"] % 4 == 0


def test_frame_mask_is_intersection_with_stored_stem_support() -> None:
    height, width = SCORER_HW
    support = np.asarray([0], dtype=np.uint32)
    frames = FrameLibrary(
        envelopes={CURVELET_FAMILY: np.ones((height * width, 1), dtype=np.float32)},
        atom_counts={CURVELET_FAMILY: 1},
        custody={},
    )
    descriptor = {
        "family": CURVELET_FAMILY,
        "threshold_fraction": 0.5,
        "atom_indices": [0],
        "scorer_recursive_write_support": {"stem_block_indices": [0]},
    }
    mask = _mask_from_descriptor(descriptor, frames, support)
    assert np.count_nonzero(mask) == 4
    assert np.all(mask[:2, :2])


def test_recursive_target_plane_writes_only_stored_stem_blocks() -> None:
    height, width = SCORER_HW
    base_planes = np.zeros((2, height, width, 3), dtype=np.uint8)
    target_planes = np.full_like(base_planes, 9)
    empty_camera = np.zeros((2, 1, 1, 3), dtype=np.uint8)
    state = PairState(
        pair_id=0,
        base_planes=base_planes,
        target_planes=target_planes,
        base_camera=empty_camera,
        target_camera=empty_camera,
        base_logits=np.zeros((5, height, width), dtype=np.float32),
        target_logits=np.zeros((5, height, width), dtype=np.float32),
        base_pose=np.zeros(6, dtype=np.float64),
        gt_pose=np.zeros(6, dtype=np.float64),
        labels=np.zeros((height, width), dtype=np.uint8),
    )
    descriptor = {
        "mechanism": "scorer_recursive_target",
        "scorer_recursive_write_support": {"stem_block_indices": [0]},
        "quantum": None,
    }
    plane = _plane_from_descriptor(
        descriptor=descriptor,
        state=state,
        support=np.asarray([0], dtype=np.uint32),
        frames=FrameLibrary(envelopes={}, atom_counts={}, custody={}),
        old_selected={"scope": "local", "radius": 0, "quantum": None},
    )
    assert np.all(plane[:2, :2] == 9)
    assert np.count_nonzero(plane) == 12
    assert not np.any(plane[2:, 2:])
