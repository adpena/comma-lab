from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_dm2_l3_realization_race import (
    CAMERA_HW,
    SCORER_HW,
    DM2RealizationError,
    RGBDeltaRecord,
    candidate_scorer_plane,
    decode_coded_rgb,
    decode_joint_rgb_records,
    dilated_support_mask,
    encode_coded_rgb,
    encode_joint_rgb_records,
    price_rgb_raw,
)
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import realize_solve_camera


def test_rgb_delta_record_and_real_coders_parse_back_exactly() -> None:
    base = np.zeros((*CAMERA_HW, 3), dtype=np.uint8)
    candidate = base.copy()
    candidate[4, 5] = (1, 2, 3)
    candidate[800, 1000] = (255, 9, 8)
    record = RGBDeltaRecord.from_frames(14, base, candidate)
    parsed = RGBDeltaRecord.decode(record.encode())
    assert parsed == record
    assert np.array_equal(parsed.apply(base), candidate)

    prices, winner = price_rgb_raw(record.encode())
    assert winner in prices
    assert all(row["parseback_exact"] for row in prices.values())
    for codec in prices:
        encoded = encode_coded_rgb(record.encode(), codec)
        assert decode_coded_rgb(encoded) == (codec, record.encode())


def test_joint_rgb_record_rejects_duplicate_pair_frame_keys() -> None:
    base = np.zeros((*CAMERA_HW, 3), dtype=np.uint8)
    candidate = base.copy()
    candidate[1, 1] = (1, 0, 0)
    left = RGBDeltaRecord.from_frames(1, base, candidate)
    right = RGBDeltaRecord.from_frames(2, base, candidate)
    payload = encode_joint_rgb_records((right, left))
    assert decode_joint_rgb_records(payload) == (left, right)
    with pytest.raises(DM2RealizationError, match="duplicate"):
        encode_joint_rgb_records((left, left))


def test_local_fixed_quantum_candidate_is_exactly_realizable_through_R() -> None:
    base = np.full((*SCORER_HW, 3), 100, dtype=np.uint8)
    target = base.copy()
    target[10, 20] = (180, 20, 101)
    support = np.asarray([10 * SCORER_HW[1] + 20], dtype=np.uint32)
    candidate = candidate_scorer_plane(
        base,
        target,
        support,
        scope="local",
        radius=0,
        quantum=8,
    )
    assert tuple(candidate[10, 20]) == (108, 92, 101)
    assert np.count_nonzero(candidate != base) == 3

    kernel = FullResizeKernel.build()
    camera = realize_solve_camera(candidate, kernel)
    verification = kernel.operator.verify_factor2_uint8(camera, candidate)
    assert verification.numerator_exact is True
    assert verification.certified_exact is True


def test_support_dilation_and_candidate_contract_fail_closed() -> None:
    support = np.asarray([0], dtype=np.uint32)
    mask = dilated_support_mask(support, 2)
    assert int(mask.sum()) == 9
    base = np.zeros((*SCORER_HW, 3), dtype=np.uint8)
    with pytest.raises(DM2RealizationError, match="requires a support radius"):
        candidate_scorer_plane(
            base,
            base,
            support,
            scope="local",
            radius=None,
            quantum=None,
        )
    with pytest.raises(DM2RealizationError, match="integer in"):
        candidate_scorer_plane(
            base,
            base,
            support,
            scope="global",
            radius=None,
            quantum=0,
        )
