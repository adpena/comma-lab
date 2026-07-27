from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tac.optimization.taskspace_exact_coarse_costates_v2 import (
    compute_batch_exact_coarse_costates_v2,
)
from tac.optimization.taskspace_projected_population_costates_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PopulationScorePointV1,
)

H = "a" * 64


class _ToyPoseNet:
    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def __call__(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        summary = value.mean(dim=(1, 2, 3, 4), keepdim=False)[:, None]
        return {"pose": summary.repeat(1, 12)}


class _GradDriftSegNet:
    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value[:, -1],
            size=(SCORER_HEIGHT, SCORER_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        intensity = value.mean(dim=1, keepdim=True)
        zero = intensity * 0.0
        # Candidate forward has grad enabled; authority custody below is a
        # separately supplied inference result.
        drift = 1.0e-6 if torch.is_grad_enabled() else -1.0e-6
        impossible = torch.full_like(zero, -1000.0)
        return torch.cat(
            (
                zero,
                intensity,
                zero + drift,
                impossible,
                impossible,
            ),
            dim=1,
        )


def _score_point() -> PopulationScorePointV1:
    return PopulationScorePointV1(
        global_mean_pose_dist=163.06130981,
        sample_count=600,
        archive_bytes=129_392,
        archive_sha256=H,
    )


def test_v2_keeps_inference_cells_authoritative_and_annotates_grad_tie() -> None:
    candidate = np.zeros(
        (1, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8,
    )
    target = np.full_like(candidate, 255)
    current_cells = np.zeros((1, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    target_cells = np.ones_like(current_cells)
    result = compute_batch_exact_coarse_costates_v2(
        candidate_pairs_hwc=candidate,
        target_pairs_hwc=target,
        expected_target_cells=target_cells,
        expected_current_cells=current_cells,
        authority_target_cells=target_cells,
        authority_current_cells=current_cells,
        pair_ids=(100,),
        posenet=_ToyPoseNet(),
        segnet=_GradDriftSegNet(),
        device="cpu",
        score_point=_score_point(),
    )
    assert result.costates.base_mismatch_count == SCORER_HEIGHT * SCORER_WIDTH
    assert result.current_drift.mismatch_cell_count == SCORER_HEIGHT * SCORER_WIDTH
    assert result.current_drift.mismatch_pair_ids == (100,)
    assert result.current_drift.minimum_top_two_margin_at_drift == pytest.approx(1.0e-6)
    assert result.target_drift.mismatch_cell_count == 0
    assert result.pose_drift.maximum_abs_current_delta == 0.0
    assert result.pose_drift.maximum_abs_target_delta == 0.0
    assert result.drift_dict()["authority_cells_drive_exact_replay"] is True
    assert result.drift_dict()["authority_pose_targets_and_base_mse_drive_exact_replay"] is True


def test_v2_refuses_authority_cells_that_differ_from_expected_custody() -> None:
    candidate = np.zeros(
        (1, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8,
    )
    expected = np.zeros((1, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    wrong_authority = np.ones_like(expected)
    with pytest.raises(
        ValueError,
        match="current scorer-authority inference cells differ",
    ):
        compute_batch_exact_coarse_costates_v2(
            candidate_pairs_hwc=candidate,
            target_pairs_hwc=candidate,
            expected_target_cells=expected,
            expected_current_cells=expected,
            authority_target_cells=expected,
            authority_current_cells=wrong_authority,
            pair_ids=(0,),
            posenet=_ToyPoseNet(),
            segnet=_GradDriftSegNet(),
            device="cpu",
            score_point=_score_point(),
        )
