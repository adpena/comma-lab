# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tac.analysis.receiver_replay_scorer_hard_regions as hard_regions
from tac.analysis.receiver_replay_scorer_hard_regions import (
    ReceiverReplayHardRegionError,
    build_hard_region_recon_pixel_weight,
    build_receiver_replay_scorer_hard_region_report,
    build_segnet_argmax_arrays_from_cache_dirs,
    load_argmax_array,
    write_hard_region_recon_pixel_weight_artifact,
)


def test_receiver_replay_hard_regions_builds_confusion_records() -> None:
    reference = np.array(
        [
            [
                [0, 0, 1],
                [1, 1, 2],
                [2, 2, 2],
            ]
        ],
        dtype=np.int16,
    )
    candidate = np.array(
        [
            [
                [0, 1, 1],
                [1, 2, 2],
                [2, 0, 2],
            ]
        ],
        dtype=np.int16,
    )

    report = build_receiver_replay_scorer_hard_region_report(
        candidate_argmax=candidate,
        reference_argmax=reference,
        top_components=0,
    )

    assert report["confusion_matrix"][:3] == [
        [1, 1, 0, 0, 0],
        [0, 2, 1, 0, 0],
        [1, 0, 3, 0, 0],
    ]
    assert report["mismatch_pixels"] == 3
    assert report["argmax_disagreement_rate"] == pytest.approx(3 / 9)
    assert report["segnet_score_contribution"] == pytest.approx(100 * 3 / 9)
    assert [
        (row["target_class"], row["predicted_class"], row["pixel_count"])
        for row in report["hard_region_records"]
    ] == [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_receiver_replay_hard_regions_ranks_connected_components() -> None:
    reference = np.zeros((1, 5, 5), dtype=np.int16)
    candidate = np.zeros((1, 5, 5), dtype=np.int16)
    candidate[0, 0:2, 0:2] = 2
    candidate[0, 3:5, 3] = 3
    candidate[0, 4, 0] = 3

    report = build_receiver_replay_scorer_hard_region_report(
        candidate_argmax=candidate,
        reference_argmax=reference,
        top_components=4,
    )

    components = report["top_connected_components"]
    assert [component["pixel_count"] for component in components] == [4, 2, 1]
    assert components[0]["target_class"] == 0
    assert components[0]["predicted_class"] == 2
    assert components[0]["bbox_y0x0y1x1_exclusive"] == [0, 0, 2, 2]
    assert components[1]["predicted_class"] == 3
    assert components[1]["bbox_y0x0y1x1_exclusive"] == [3, 3, 5, 4]
    assert components[2]["bbox_y0x0y1x1_exclusive"] == [4, 0, 5, 1]


def test_receiver_replay_hard_regions_prices_score_weighted_mass_and_pose_context() -> None:
    reference = np.zeros((1, 2, 2), dtype=np.int16)
    candidate = np.array([[[0, 1], [0, 0]]], dtype=np.int16)

    report = build_receiver_replay_scorer_hard_region_report(
        candidate_argmax=candidate,
        reference_argmax=reference,
        pair_indices=np.array([[10, 11]], dtype=np.int64),
        posenet_distortion=np.array([9.0], dtype=np.float32),
        segnet_distortion=np.array([0.25], dtype=np.float32),
        top_components=1,
    )

    hard = report["hard_region_records"][0]
    assert hard["source_frame_pair"] == [10, 11]
    assert hard["target_class"] == 0
    assert hard["predicted_class"] == 1
    assert hard["score_weighted_unsolved_mass"] == pytest.approx(25.0)
    assert hard["pair_score_weighted_unsolved_mass"] == pytest.approx(25.0)
    assert hard["target_mass_pixels"] == 4
    assert hard["pixel_fraction_of_target_region"] == pytest.approx(0.25)
    assert hard["posenet_distortion"] == pytest.approx(9.0)
    assert hard["pose_marginal_score_contribution"] == pytest.approx(9.0 * 5.0 / np.sqrt(90.0))
    assert hard["scorer_response_segnet_distortion"] == pytest.approx(0.25)
    assert report["per_pair"][0]["argmax_pair_segnet_distortion"] == pytest.approx(0.25)
    assert report["per_pair"][0]["scorer_response_segnet_distortion"] == pytest.approx(0.25)


def test_receiver_replay_hard_regions_fail_closed_on_bad_inputs(tmp_path) -> None:
    with pytest.raises(ReceiverReplayHardRegionError, match="shapes must match"):
        build_receiver_replay_scorer_hard_region_report(
            candidate_argmax=np.zeros((1, 2, 2), dtype=np.int16),
            reference_argmax=np.zeros((1, 2, 3), dtype=np.int16),
        )

    with pytest.raises(ReceiverReplayHardRegionError, match="negative class ids"):
        build_receiver_replay_scorer_hard_region_report(
            candidate_argmax=np.array([[[-1]]], dtype=np.int16),
            reference_argmax=np.zeros((1, 1, 1), dtype=np.int16),
        )

    with pytest.raises(FileNotFoundError, match="argmax array does not exist"):
        load_argmax_array(tmp_path / "missing.npy")


def test_receiver_replay_cache_miner_accepts_nhwc_but_runs_nchw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    np.save(candidate / "segnet_last_rgb.npy", np.zeros((2, 4, 5, 3), dtype=np.float32))
    np.save(reference / "segnet_last_rgb.npy", np.zeros((2, 4, 5, 3), dtype=np.float32))
    np.save(candidate / "pair_indices.npy", np.array([[0, 1], [2, 3]], dtype=np.int64))
    seen_shapes: list[tuple[int, ...]] = []

    def fake_logits_builder(*, upstream_dir: Path, device: str):
        assert upstream_dir == tmp_path / "upstream"
        assert device == "cpu"

        def logits_fn(x_nchw: np.ndarray) -> np.ndarray:
            seen_shapes.append(tuple(int(v) for v in x_nchw.shape))
            logits = np.zeros(
                (x_nchw.shape[0], 5, x_nchw.shape[2], x_nchw.shape[3]),
                dtype=np.float32,
            )
            logits[:, 2] = 1.0
            return logits

        return logits_fn

    monkeypatch.setattr(hard_regions, "_build_real_mlx_segnet_logits_fn", fake_logits_builder)

    candidate_argmax, reference_argmax, pair_indices, source = (
        build_segnet_argmax_arrays_from_cache_dirs(
            candidate_cache_dir=candidate,
            reference_cache_dir=reference,
            upstream_dir=tmp_path / "upstream",
            batch_frames=1,
            device="cpu",
        )
    )

    assert seen_shapes == [(1, 3, 4, 5), (1, 3, 4, 5), (1, 3, 4, 5), (1, 3, 4, 5)]
    assert candidate_argmax.shape == (2, 4, 5)
    assert reference_argmax.shape == (2, 4, 5)
    assert np.all(candidate_argmax == 2)
    assert pair_indices.tolist() == [[0, 1], [2, 3]]
    assert source["candidate_cache_layout"] == "NHWC_to_NCHW"
    assert source["reference_cache_layout"] == "NHWC_to_NCHW"


def test_receiver_replay_cache_miner_rejects_unknown_rgb_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    np.save(candidate / "segnet_last_rgb.npy", np.zeros((1, 4, 5, 4), dtype=np.float32))
    np.save(reference / "segnet_last_rgb.npy", np.zeros((1, 4, 5, 4), dtype=np.float32))
    np.save(candidate / "pair_indices.npy", np.array([[0, 1]], dtype=np.int64))
    monkeypatch.setattr(
        hard_regions,
        "_build_real_mlx_segnet_logits_fn",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not build SegNet")),
    )

    with pytest.raises(ReceiverReplayHardRegionError, match="RGB channel dimension 3"):
        build_segnet_argmax_arrays_from_cache_dirs(
            candidate_cache_dir=candidate,
            reference_cache_dir=reference,
            upstream_dir=tmp_path / "upstream",
        )


def test_hard_region_report_builds_trainer_recon_weight_surface() -> None:
    reference = np.zeros((2, 4, 4), dtype=np.int16)
    candidate = np.zeros((2, 4, 4), dtype=np.int16)
    candidate[0, 0:2, 0:2] = 2
    candidate[1, 3, 3] = 3
    report = build_receiver_replay_scorer_hard_region_report(
        candidate_argmax=candidate,
        reference_argmax=reference,
        top_components=8,
    )

    weight, metadata = build_hard_region_recon_pixel_weight(
        report,
        output_height=8,
        output_width=8,
        base_weight=1.0,
        score_gain=2.0,
        component_gain=1.0,
        normalize="none",
    )

    assert weight.shape == (2, 2, 8, 8, 1)
    assert metadata["schema"] == "receiver_replay_hard_region_recon_pixel_weight.v1"
    assert metadata["target_frame_index"] == 1
    assert metadata["applied_hard_region_records"] == 2
    assert metadata["applied_component_bboxes"] == 2
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    # SegNet frame 1 receives localized pressure; frame 0 remains baseline.
    assert float(weight[0, 0, :, :, 0].max()) == pytest.approx(1.0)
    assert float(weight[0, 1, 0:4, 0:4, 0].mean()) > 1.0
    assert float(weight[0, 1, 4:, 4:, 0].mean()) == pytest.approx(1.0)
    assert float(weight[1, 1, 6:8, 6:8, 0].mean()) > 1.0


def test_hard_region_recon_weight_artifact_has_custody(tmp_path: Path) -> None:
    reference = np.zeros((1, 2, 2), dtype=np.int16)
    candidate = np.array([[[0, 1], [0, 0]]], dtype=np.int16)
    report = build_receiver_replay_scorer_hard_region_report(
        candidate_argmax=candidate,
        reference_argmax=reference,
        top_components=2,
    )

    manifest = write_hard_region_recon_pixel_weight_artifact(
        report=report,
        output_dir=tmp_path / "hard_region_weight",
        output_height=4,
        output_width=4,
    )

    weight_path = Path(manifest["weight_path"])
    manifest_path = Path(manifest["manifest_path"])
    assert weight_path.is_file()
    assert manifest_path.is_file()
    loaded = np.load(weight_path)["weight"]
    assert loaded.shape == (1, 2, 4, 4, 1)
    assert manifest["schema"] == (
        "receiver_replay_hard_region_recon_pixel_weight_manifest.v1"
    )
    assert manifest["metadata"]["applied_hard_region_records"] == 1
    assert manifest["consumption"]["training_arg"] == "--recon-pixel-weight-path"
    assert manifest["consumption"]["auto_discovery_eligible"] is False
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
