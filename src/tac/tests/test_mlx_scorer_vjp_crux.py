# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.local_acceleration.mlx_scorer_vjp_crux import (
    FALSE_AUTHORITY,
    SCHEMA_VERSION,
    _comparison_rows,
    _stats,
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
