from __future__ import annotations

import pytest

from tac.local_acceleration.tile_halo_exactness import (
    cadence_from_temporal_iou,
    derive_exact_tile_halo_contract,
    derive_receptive_field_rows,
)


def test_frozen_b2_unet_local_halo_and_global_dependency_are_full_frame() -> None:
    rows = derive_receptive_field_rows()
    assert [row.receptive_field_px for row in rows] == [
        3,
        11,
        31,
        111,
        223,
        479,
        1055,
        1183,
        1247,
        1279,
        1311,
        1311,
        1311,
        1311,
    ]
    assert rows[-1].local_halo_px == 685
    assert rows[-1].global_se_blocks_seen == 23
    contract = derive_exact_tile_halo_contract()
    assert contract.exact_dependency == "FULL_FRAME_GLOBAL"
    assert contract.exact_source_area_fraction == 1.0
    assert contract.ideal_exact_speedup_upper_bound == 1.0
    assert contract.verdict == "NO_GO"


def test_temporal_iou_cadence_uses_preregistered_90pct_survival() -> None:
    assert cadence_from_temporal_iou(0.263) == 1  # Lane
    assert cadence_from_temporal_iou(0.903) == 1  # Movable
    assert cadence_from_temporal_iou(0.955) == 2  # Road
    assert cadence_from_temporal_iou(0.995) == 21  # Undrivable
    assert cadence_from_temporal_iou(0.994) == 17  # MyCar outside static core


@pytest.mark.parametrize("iou", [0.0, 1.0, -0.1, 1.1])
def test_temporal_iou_cadence_fails_closed(iou: float) -> None:
    with pytest.raises(ValueError):
        cadence_from_temporal_iou(iou)
