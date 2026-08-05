# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import ddm_st2_scorer_native_student as st2


def _fake_capacity_prior() -> dict:
    return {
        "source_path": "fake",
        "source_sha256": "0" * 64,
        "top_by_road_lane_stratum": {
            "boundary_static": {"top_channels": [0]},
            "boundary_transient": {"top_channels": [6]},
            "cell_static": {"top_channels": [12]},
            "cell_transient": {"top_channels": [15]},
        },
    }


def test_fisher_trace_decreases_with_margin() -> None:
    values = st2.fisher_trace_from_margin(np.asarray([0.0, 1.0, 4.0], dtype=np.float32))
    assert np.isclose(values[0], 0.5)
    assert values[0] > values[1] > values[2]


def test_scorer_native_payload_roundtrip() -> None:
    qlogits = np.asarray([1, -2, 300], dtype="<i2")
    raw = st2.scorer_native_payload(
        qlogits,
        bucket_count=3,
        qscale=256.0,
        bins=st2.FeatureBins(),
        capacity_prior=_fake_capacity_prior(),
    )
    header, decoded = st2.decode_scorer_native_payload(raw)
    assert header["schema"] == "ddm_st2_scorer_native_bucket_student_payload.v1"
    assert header["feature_schema"]["road_lane_head_norm"] == st2.ROAD_LANE_HEAD_NORM
    np.testing.assert_array_equal(decoded, qlogits)


def test_capacity_prior_loader_extracts_road_lane_strata(tmp_path: Path) -> None:
    rows = []
    for bucket_id, support in (
        ("road_lane__boundary__static_in_image", 10),
        ("road_lane__boundary__transient", 11),
        ("road_lane__cell__static_in_image", 12),
        ("road_lane__cell__transient", 13),
    ):
        rows.append(
            {
                "bucket_id": bucket_id,
                "class_pair": "0-1",
                "class_pair_names": "Road--Lane",
                "capacity_share": [0.1, 0.7, 0.2],
                "capacity_per_channel": [1.0, 7.0, 2.0],
                "support_pixel_count": support,
            }
        )
    table = tmp_path / "hope_per_stratum_capacity_table.json"
    table.write_text(json.dumps({"schema": "test", "strata": rows}), encoding="utf-8")
    prior = st2.load_road_lane_capacity_prior(table)
    assert set(prior["top_by_road_lane_stratum"]) == {
        "boundary_static",
        "boundary_transient",
        "cell_static",
        "cell_transient",
    }
    assert prior["top_by_road_lane_stratum"]["boundary_static"]["top_channels"][0] == 1


def test_feature_codes_change_with_margin_frequency_and_boundary_state() -> None:
    current = np.zeros((1, st2.SEG_H, st2.SEG_W), dtype=np.uint8)
    current[:, 10, 10] = st2.ROAD
    current[:, 10, 11] = st2.LANE
    margins = np.ones((1, st2.SEG_H, st2.SEG_W), dtype=np.float16)
    margins[:, 10, 10] = np.float16(0.02)
    margins[:, 20, 20] = np.float16(2.0)
    road_lane_freq = np.zeros((st2.SEG_H, st2.SEG_W), dtype=np.uint16)
    all_freq = np.zeros((st2.SEG_H, st2.SEG_W), dtype=np.uint16)
    road_lane_freq[10, 10] = 5
    all_freq[10, 10] = 9
    pairs = np.asarray([0, 0], dtype=np.int16)
    y = np.asarray([10, 20], dtype=np.int16)
    x = np.asarray([10, 20], dtype=np.int16)
    codes = st2.scorer_native_feature_codes(
        current=current,
        margins=margins,
        pairs=pairs,
        y=y,
        x=x,
        road_lane_frequency=road_lane_freq,
        all_flip_frequency=all_freq,
        capacity_prior=_fake_capacity_prior(),
        bucket_count=4096,
    )
    assert codes.shape == (2,)
    assert codes[0] != codes[1]
