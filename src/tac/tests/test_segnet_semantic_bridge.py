# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np

from tac.analysis.segnet_semantic_bridge import (
    FALSE_AUTHORITY,
    SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES,
    SemanticBridgeConfig,
    build_segnet_semantic_bridge,
    build_segnet_semantic_surface_arrays,
    crammer_singer_hinge_for_targets,
    top2_class_indices,
    write_segnet_semantic_surface_npz,
)


def _logits_from_labels(
    labels: np.ndarray,
    *,
    top2: np.ndarray | None = None,
    high: float = 4.0,
    runner: float = 2.0,
) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int64)
    n, h, w = label_arr.shape
    logits = np.full((n, 5, h, w), -5.0, dtype=np.float64)
    if top2 is None:
        top2 = (label_arr + 1) % 5
    for sample in range(n):
        for y in range(h):
            for x in range(w):
                target = int(label_arr[sample, y, x])
                runner_up = int(top2[sample, y, x])
                logits[sample, target, y, x] = high
                if runner_up != target:
                    logits[sample, runner_up, y, x] = runner
    return logits


def test_crammer_singer_hinge_zero_iff_target_wins_by_margin() -> None:
    targets = np.array([[[0, 1]]], dtype=np.int64)
    logits = np.array(
        [
            [
                [[3.0, 0.0]],
                [[1.0, 2.0]],
                [[0.0, 3.5]],
                [[-1.0, -1.0]],
                [[-2.0, -2.0]],
            ]
        ],
        dtype=np.float64,
    )

    hinge = crammer_singer_hinge_for_targets(logits, targets, margin=0.5)

    np.testing.assert_allclose(hinge[0, 0, 0], 0.0)
    np.testing.assert_allclose(hinge[0, 0, 1], 2.0)


def test_bridge_tracks_top1_top2_spread_and_false_authority() -> None:
    source_labels = np.array(
        [
            [
                [0, 0, 1, 2],
                [3, 4, 1, 2],
            ]
        ],
        dtype=np.int64,
    )
    source_top2 = np.array(
        [
            [
                [1, 1, 2, 3],
                [4, 0, 3, 3],
            ]
        ],
        dtype=np.int64,
    )
    candidate_labels = np.array(
        [
            [
                [0, 1, 4, 2],
                [3, 0, 1, 3],
            ]
        ],
        dtype=np.int64,
    )
    source_logits = _logits_from_labels(source_labels, top2=source_top2)
    candidate_logits = _logits_from_labels(candidate_labels)

    bridge = build_segnet_semantic_bridge(
        source_logits=source_logits,
        candidate_logits=candidate_logits,
        config=SemanticBridgeConfig(
            candidate_id="unit_candidate",
            generalization_mode="mixed",
            boundary_dilation=1,
            low_margin_threshold=2.5,
            hinge_margin=0.5,
        ),
        sample_ids=[42],
        pair_component_rows={
            42: {
                "pose_dist": 0.1,
                "seg_dist": 0.2,
                "component_score_no_rate": 0.3,
            }
        },
    )

    assert bridge["schema"] == "segnet_semantic_bridge.v1"
    for key, value in FALSE_AUTHORITY.items():
        assert bridge[key] is value
    assert bridge["generalization_mode"] == "mixed"
    assert bridge["contest_overfit_policy"]["must_not_be_rebranded_as_fleet_ready"]
    assert bridge["summary"]["wrong_pixels"] == 4
    assert bridge["summary"]["error_is_top1_top2_flip_pixels"] == 3
    assert bridge["summary"]["error_is_out_of_pair_spread_pixels"] == 1
    np.testing.assert_allclose(
        bridge["summary"]["error_is_out_of_pair_spread_fraction"],
        0.25,
    )
    assert bridge["recommended_training"]["segnet_distillation_objective"] == (
        "boundary_argmax_hinge"
    )
    assert "out_of_pair" in bridge["recommended_training"]["teacher_loss_verdict"]
    assert bridge["sample_rows"][0]["pair_idx"] == 42
    assert bridge["sample_rows"][0]["pair_component_context"]["pose_dist"] == 0.1
    assert bridge["dominant_error_pairs_real_world"][0]["real_world_read"]


def test_bridge_exposes_all_parallel_generalization_lanes() -> None:
    labels = np.array([[[0, 1], [2, 3]]], dtype=np.int64)
    source_logits = _logits_from_labels(labels)
    candidate_logits = _logits_from_labels((labels + 1) % 5)

    bridge = build_segnet_semantic_bridge(
        source_logits=source_logits,
        candidate_logits=candidate_logits,
        config=SemanticBridgeConfig(
            candidate_id="fleet_candidate",
            generalization_mode="fleet_adaptable",
        ),
    )

    assert top2_class_indices(source_logits).shape == labels.shape
    modes = {row["generalization_mode"] for row in bridge["executable_backlog"]}
    assert modes == {"contest_fixed_dataset", "mixed", "fleet_adaptable"}
    families = {row["family_id"] for row in bridge["executable_backlog"]}
    assert {
        "deterministic_boundary_repair",
        "deterministic_boundary_postfilter",
        "mlx_lora_or_dora_boundary_adapter",
    }.issubset(families)
    assert (
        bridge["contest_overfit_policy"]["allowed_for_contest_fixed_dataset"]
        is False
    )
    contest_rows = [
        row
        for row in bridge["executable_backlog"]
        if row["generalization_mode"] == "contest_fixed_dataset"
    ]
    assert contest_rows
    assert all(
        row["enqueueable_under_requested_generalization_mode"] is False
        for row in contest_rows
    )
    assert all(row["compatibility_blockers"] for row in contest_rows)


def test_surface_npz_writer_is_canonical_repair_input(tmp_path: Path) -> None:
    labels = np.array([[[0, 1], [2, 3]]], dtype=np.int64)
    source_logits = _logits_from_labels(labels)
    candidate_logits = _logits_from_labels((labels + 1) % 5)

    arrays = build_segnet_semantic_surface_arrays(
        source_logits=source_logits,
        candidate_logits=candidate_logits,
        sample_ids=[99],
        boundary_dilation=1,
        hinge_margin=0.5,
    )

    assert tuple(arrays) == SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES
    assert arrays["source_argmax"].dtype == np.uint8
    assert arrays["candidate_argmax"].shape == labels.shape
    assert arrays["wrong_mask"].sum() == labels.size
    assert arrays["hinge_map"].dtype == np.float32
    assert arrays["sample_ids"].tolist() == [99]

    surface_path = tmp_path / "bridge.semantic_surfaces.npz"
    record = write_segnet_semantic_surface_npz(
        source_logits=source_logits,
        candidate_logits=candidate_logits,
        sample_ids=[99],
        boundary_dilation=1,
        hinge_margin=0.5,
        path=surface_path,
    )

    assert record["arrays"] == list(SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES)
    assert record["array_shapes"]["hinge_map"] == list(labels.shape)
    assert record["ready_for_exact_eval_dispatch"] is False
    with np.load(surface_path) as loaded:
        assert set(loaded.files) == set(SEGNET_SEMANTIC_SURFACE_ARRAY_NAMES)
        np.testing.assert_array_equal(loaded["wrong_mask"], arrays["wrong_mask"])
