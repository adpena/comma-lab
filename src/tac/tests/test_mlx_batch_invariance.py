# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.local_acceleration.mlx_batch_invariance import (
    FAIL_VERDICT,
    PASS_VERDICT,
    MLXBatchInvarianceThresholds,
    build_batch_invariance_manifest_from_outputs,
    concatenate_distortion_outputs,
)

REPO = Path(__file__).resolve().parents[3]


def _outputs(*, pose_delta: float = 0.0, seg_delta: float = 0.0) -> tuple[dict, dict]:
    singleton = {
        "posenet": {"pose": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        "segnet": np.array(
            [
                [
                    [[2.0, 0.5], [0.1, 1.0]],
                    [[1.0, 1.5], [0.2, 0.9]],
                ],
                [
                    [[0.4, 2.0], [1.1, 0.3]],
                    [[1.4, 1.0], [0.1, 1.3]],
                ],
            ],
            dtype=np.float32,
        ),
    }
    batched = {
        "posenet": {"pose": singleton["posenet"]["pose"].copy()},
        "segnet": singleton["segnet"].copy(),
    }
    batched["posenet"]["pose"][1, 1] += np.float32(pose_delta)
    batched["segnet"][0, 1, 0, 0] += np.float32(seg_delta)
    return batched, singleton


def test_batch_invariance_manifest_passes_exact_outputs() -> None:
    batched, singleton = _outputs()

    manifest = build_batch_invariance_manifest_from_outputs(
        batched_outputs=batched,
        singleton_outputs=singleton,
        device_type="cpu",
        start_pair=16,
        batch_pairs=2,
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["deltas"]["segnet_argmax_diff_pixels"] == 0


def test_batch_invariance_manifest_fails_pose_and_segnet_drift() -> None:
    batched, singleton = _outputs(pose_delta=0.01, seg_delta=2.0)

    manifest = build_batch_invariance_manifest_from_outputs(
        batched_outputs=batched,
        singleton_outputs=singleton,
        thresholds=MLXBatchInvarianceThresholds(
            max_posenet_output_abs_delta=1.0e-4,
            max_segnet_logit_abs_delta=1.0e-3,
            max_segnet_argmax_diff_pixels=0,
        ),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert "singletons_or_smaller_batches_only" in manifest["allowed_use"]
    assert any(
        blocker.startswith("posenet_output_abs_delta_exceeds_threshold")
        for blocker in manifest["blockers"]
    )
    assert any(
        blocker.startswith("segnet_logit_abs_delta_exceeds_threshold")
        for blocker in manifest["blockers"]
    )
    assert any(
        blocker.startswith("segnet_argmax_diff_pixels_exceeds_threshold")
        for blocker in manifest["blockers"]
    )


def test_concatenate_distortion_outputs_preserves_order() -> None:
    outputs = [
        {
            "posenet": {"pose": np.array([[1.0, 2.0]], dtype=np.float32)},
            "segnet": np.zeros((1, 2, 1, 1), dtype=np.float32),
        },
        {
            "posenet": {"pose": np.array([[3.0, 4.0]], dtype=np.float32)},
            "segnet": np.ones((1, 2, 1, 1), dtype=np.float32),
        },
    ]

    merged = concatenate_distortion_outputs(outputs)

    assert merged["posenet"]["pose"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert merged["segnet"].shape == (2, 2, 1, 1)


def test_batch_invariance_cli_rejects_singleton_batch(tmp_path: Path) -> None:
    out_path = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_batch_invariance.py"),
            "--cache-dir",
            str(tmp_path),
            "--output",
            str(out_path),
            "--batch-pairs",
            "1",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "batch_pairs must be >= 2" in completed.stderr
    assert not out_path.exists()
