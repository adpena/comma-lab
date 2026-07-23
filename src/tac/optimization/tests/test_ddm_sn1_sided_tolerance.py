# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from tac.optimization.ddm_sn1_sided_tolerance import (
    ORDERED_PAIR_COUNT,
    TEMPORAL_STRATA,
    SidedToleranceError,
    build_header,
    build_sided_rows,
    export_e1_bounds,
    export_jsonl,
    orientation,
    parse_jsonl,
    price_e2_sided_update,
    validate_row,
)


def _samples() -> tuple[dict[str, list[float]], dict[str, float]]:
    margins: dict[str, list[float]] = {}
    norms: dict[str, float] = {}
    for winner in range(5):
        for rival in range(5):
            if winner == rival:
                continue
            key = orientation(winner, rival)
            margins[key] = [0.1 + 0.01 * winner, 0.2 + 0.02 * rival, 0.4]
            norms[key] = 2.0 + 0.1 * min(winner, rival) + 0.1 * max(winner, rival)
    return margins, norms


def _header():
    return build_header(
        source_video_sha256="1" * 64,
        segnet_weights_sha256="2" * 64,
        upstream_modules_sha256="3" * 64,
        telemetry_sha256="4" * 64,
    )


def test_full_ordered_matrix_roundtrips_canonical_jsonl() -> None:
    margins, norms = _samples()
    rows = [
        row
        for stratum in TEMPORAL_STRATA
        for row in build_sided_rows(
            temporal_stratum=stratum,
            margins_by_orientation=margins,
            pair_norms_by_orientation=norms,
        )
    ]
    payload = export_jsonl(_header(), rows)
    header, parsed = parse_jsonl(payload)
    assert header.ordered_pair_count == ORDERED_PAIR_COUNT
    assert len(parsed) == ORDERED_PAIR_COUNT * len(TEMPORAL_STRATA)
    assert export_jsonl(header, parsed) == payload


def test_reverse_sides_remain_directional_and_export_signed_e1_bounds() -> None:
    margins, norms = _samples()
    margins["Road->Lane"] = [0.02, 0.04, 0.06]
    margins["Lane->Road"] = [0.8, 1.0, 1.2]
    rows = {
        row.orientation: row
        for row in build_sided_rows(
            temporal_stratum="n600_full",
            margins_by_orientation=margins,
            pair_norms_by_orientation=norms,
        )
    }
    road_lane = rows["Road->Lane"]
    lane_road = rows["Lane->Road"]
    assert road_lane.inner_tolerance_d2 != lane_road.inner_tolerance_d2
    assert road_lane.outer_tolerance_d2 == lane_road.inner_tolerance_d2
    bounds = export_e1_bounds(road_lane)
    assert bounds["inner_signed_bound"] < 0
    assert bounds["outer_signed_bound"] > 0


def test_e2_prices_inner_and_outer_with_independent_multipliers() -> None:
    margins, norms = _samples()
    row = build_sided_rows(
        temporal_stratum="n600_full",
        margins_by_orientation=margins,
        pair_norms_by_orientation=norms,
    )[0]
    priced = price_e2_sided_update(
        row,
        realized_inner_excess_d2=0.1,
        realized_outer_excess_d2=0.2,
        lambda_inner_seg=3.0,
        lambda_outer_seg=7.0,
        pose_objective_delta=0.01,
        lambda_pose=2.0,
        delta_archive_bytes=4,
        lambda_byte=0.5,
    )
    assert priced["inner_price"] == pytest.approx(0.3)
    assert priced["outer_price"] == pytest.approx(1.4)
    assert priced["reduced_cost"] == pytest.approx(3.72)
    assert priced["asymmetric_seg_prices"] is True


def test_strict_row_validator_rejects_direction_identity_drift() -> None:
    margins, norms = _samples()
    row = build_sided_rows(
        temporal_stratum="n600_full",
        margins_by_orientation=margins,
        pair_norms_by_orientation=norms,
    )[0]
    malformed = asdict(replace(row, reverse_orientation="Road->Lane"))
    with pytest.raises(SidedToleranceError, match="identity"):
        validate_row(malformed)
