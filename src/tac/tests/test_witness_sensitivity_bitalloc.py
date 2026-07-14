# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

from tac.witness_sensitivity_bitalloc import (
    BITS_TO_LEVEL,
    build_rd_table,
    classify_response_rows,
    distortion_score_delta,
    repeat_noise_floor,
    solve_measured_reverse_waterfill,
)


def _curves():
    baseline = {"archive_bytes": 1000, "d_seg": 0.01, "d_pose": 0.04}
    rows = []
    for tensor, scale in (("a", 1.0), ("b", 0.001)):
        for bits in range(8, 1, -1):
            saved = (8 - bits) * (50 if tensor == "a" else 100)
            dist = scale * (8 - bits) * 1e-4
            rows.append(
                {
                    "tensor": tensor,
                    "bits": bits,
                    "label": f"{tensor}:int{bits}",
                    "archive_bytes": baseline["archive_bytes"] - saved,
                    "d_seg": baseline["d_seg"] + dist,
                    "d_pose": baseline["d_pose"],
                }
            )
    return baseline, rows


def test_exact_nonlinear_pose_delta() -> None:
    got = distortion_score_delta(0.02, 0.09, 0.01, 0.04)
    assert got == 1.0 + math.sqrt(0.9) - math.sqrt(0.4)


def test_repeat_floor_and_zero_invariance_are_componentwise() -> None:
    base, rows = _curves()
    floor = repeat_noise_floor(
        {"d_seg": 0.01, "d_pose": 0.04},
        {"d_seg": 0.010000001, "d_pose": 0.040000001},
    )
    classified = classify_response_rows(rows, baseline=base, noise_floor=floor)
    int8 = [r for r in classified if r["bits"] == 8]
    assert all(r["zero_invariant_within_repeat_floor"] for r in int8)
    assert all(not r["zero_invariant_within_repeat_floor"] for r in classified if r["bits"] < 8)


def test_classifier_accepts_nonprecision_ablation_rows() -> None:
    base, rows = _curves()
    rows.extend(
        [
            {
                "tensor": "a",
                "bits": None,
                "label": "a:zero",
                "archive_bytes": 500,
                "d_seg": 0.5,
                "d_pose": 0.04,
            },
            {
                "tensor": "a",
                "bits": None,
                "label": "a:mean",
                "archive_bytes": 501,
                "d_seg": 0.4,
                "d_pose": 0.04,
            },
        ]
    )
    classified = classify_response_rows(
        rows, baseline=base, noise_floor={"d_seg": 0.0, "d_pose": 0.0}
    )
    assert {row["label"] for row in classified if row["bits"] is None} == {
        "a:zero",
        "a:mean",
    }


def test_rd_table_uses_real_qmax_levels() -> None:
    base, rows = _curves()
    classified = classify_response_rows(
        rows, baseline=base, noise_floor={"d_seg": 0.0, "d_pose": 0.0}
    )
    table = build_rd_table(classified, base)
    assert set(table["a"]) == set(BITS_TO_LEVEL.values())
    assert table["a"][127] == (0.0, 0.0)


def test_measured_waterfill_is_nonuniform_and_kkt_checked() -> None:
    base, rows = _curves()
    classified = classify_response_rows(
        rows, baseline=base, noise_floor={"d_seg": 0.0, "d_pose": 0.0}
    )
    solved = solve_measured_reverse_waterfill(classified, base)
    assert solved["kkt_holds"] is True
    assert solved["nbits"]["b"] < solved["nbits"]["a"]
    assert solved["n_coarsened"] >= 1
