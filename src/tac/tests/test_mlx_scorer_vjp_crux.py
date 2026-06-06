# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file

from tac.local_acceleration.mlx_scorer_vjp_crux import (
    FALSE_AUTHORITY,
    POSENET_YUV6_VJP_PARITY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _comparison_rows,
    _stats,
    build_posenet_yuv6_vjp_parity_record,
)


def test_vjp_crux_stats_and_ratio_blocker_detect_metal_explosion() -> None:
    stats = _stats(np.array([1.0, -2.0], dtype=np.float32))
    assert stats["finite_fraction"] == 1.0
    assert stats["abs_max"] == 2.0

    rows = _comparison_rows(
        branch="seg",
        branch_result={
            "torch_cpu": {
                "posenet_yuv6_pair_grad": _stats(np.zeros((1,), dtype=np.float32)),
                "segnet_last_rgb_grad": _stats(np.array([1.0e-4], dtype=np.float32)),
            },
            "mlx_cpu": {
                "posenet_yuv6_pair_grad": _stats(np.zeros((1,), dtype=np.float32)),
                "segnet_last_rgb_grad": _stats(np.array([1.1e-4], dtype=np.float32)),
            },
            "mlx_gpu": {
                "posenet_yuv6_pair_grad": _stats(np.zeros((1,), dtype=np.float32)),
                "segnet_last_rgb_grad": _stats(np.array([1.0e21], dtype=np.float32)),
            },
        },
        max_abs_ratio_warn=1.0e3,
    )

    seg_row = next(row for row in rows if row["tensor"] == "segnet_last_rgb_grad")
    assert seg_row["mlx_cpu_to_torch_abs_max_ratio"] < 2.0
    assert seg_row["mlx_gpu_to_torch_abs_max_ratio"] > 1.0e20
    assert len(seg_row["blockers"]) == 1
    assert seg_row["blockers"][0].startswith(
        "seg_mlx_gpu_segnet_last_rgb_grad_abs_max_ratio_exceeds:"
    )


def test_vjp_crux_schema_and_false_authority_constants() -> None:
    assert SCHEMA_VERSION == "mlx_scorer_vjp_crux.v1"
    assert FALSE_AUTHORITY["score_claim"] is False
    assert FALSE_AUTHORITY["promotion_eligible"] is False


def test_pose_branch_vjp_record_is_fail_closed_on_mlx_cpu_tiny_yuv6() -> None:
    pytest.importorskip("mlx.core")
    upstream = Path("upstream").resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import modules  # type: ignore

    posenet = modules.PoseNet().eval()
    posenet.load_state_dict(load_file(modules.posenet_sd_path))
    rng = np.random.default_rng(113)
    reference = rng.normal(0.0, 1.0, size=(1, 12, 64, 80)).astype(np.float32)
    candidate = reference + rng.normal(0.0, 0.01, size=reference.shape).astype(
        np.float32
    )

    record = build_posenet_yuv6_vjp_parity_record(
        torch_posenet=posenet,
        reference_yuv6_pair_nchw=reference,
        candidate_yuv6_pair_nchw=candidate,
        full_video_pair_count=1,
        full_video_d_pose=0.01,
        device_type="cpu",
    )

    assert record["schema_version"] == POSENET_YUV6_VJP_PARITY_SCHEMA_VERSION
    assert record["passed"] is (not record["blockers"])
    assert record["verdict"] in {
        "PASS_POSENET_YUV6_VJP_PARITY",
        "FAIL_POSENET_YUV6_VJP_PARITY",
    }
    assert record["score_claim"] is False
    assert record["promotion_eligible"] is False
    assert record["torch_cpu"]["candidate_grad_stats"]["abs_max"] > 0.0
    assert record["mlx_cpu"]["candidate_grad_stats"]["abs_max"] > 0.0
    if record["passed"]:
        assert record["comparison"]["candidate_pose_delta"]["abs_max"] < 2.0e-3
        assert record["comparison"]["candidate_grad_delta"]["abs_max"] < 2.0e-4
    else:
        assert any("posenet_yuv6" in blocker for blocker in record["blockers"])
        assert record["ready_for_exact_eval_dispatch"] is False


def test_pose_branch_vjp_parity_record_can_require_measured_pass_stability() -> None:
    pytest.importorskip("mlx.core")
    upstream = Path("upstream").resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import modules  # type: ignore

    rng = np.random.default_rng(113)
    reference = rng.normal(0.0, 1.0, size=(1, 12, 64, 80)).astype(np.float32)
    candidate = reference + rng.normal(0.0, 0.01, size=reference.shape).astype(
        np.float32
    )
    posenet = modules.PoseNet().eval()
    posenet.load_state_dict(load_file(modules.posenet_sd_path))
    record = build_posenet_yuv6_vjp_parity_record(
        torch_posenet=posenet,
        reference_yuv6_pair_nchw=reference,
        candidate_yuv6_pair_nchw=candidate,
        full_video_pair_count=1,
        full_video_d_pose=0.01,
        device_type="cpu",
        measured_passes=2,
    )

    assert record["measured_passes"] == 2
    assert record["mlx_cpu"]["measured_passes"] == 2
    assert "measured_loss_spread" in record["mlx_cpu"]
    assert "measured_candidate_grad_spread_stats" in record["mlx_cpu"]
    assert record["passed"] is (not record["blockers"])
