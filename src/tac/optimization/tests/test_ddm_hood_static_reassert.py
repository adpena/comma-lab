# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_hood_static_reassert import (
    HoodStaticReassertError,
    class_transition_rows,
    decode_stored_support,
    derive_hood_supports,
    encode_stored_support,
    expand_support_to_camera,
    reassert_frame1,
)


@pytest.mark.parametrize("shape", [(7, 9), (3, 7, 9)])
def test_support_payload_roundtrips_exactly(shape: tuple[int, ...]) -> None:
    source = (np.arange(np.prod(shape)).reshape(shape) % 5) == 0
    payload = encode_stored_support(source)
    assert np.array_equal(decode_stored_support(payload), source)
    assert encode_stored_support(decode_stored_support(payload)) == payload


def test_reassert_is_frame1_only_and_support_local() -> None:
    base = np.zeros((2, 2, 4, 5, 3), dtype=np.uint8)
    base[:, 1] = 17
    winner = base.copy()
    winner[:, 1] = 99
    support = np.zeros((2, 4, 5), dtype=bool)
    support[:, 2:, 1:4] = True
    result = reassert_frame1(
        winner_camera=winner, base_camera=base, camera_support=support
    )
    assert np.array_equal(result[:, 0], base[:, 0])
    assert np.all(result[:, 1][support] == 17)
    assert np.all(result[:, 1][~support] == 99)


def test_reassert_refuses_a_parent_that_changed_frame0() -> None:
    base = np.zeros((1, 2, 2, 2, 3), dtype=np.uint8)
    winner = base.copy()
    winner[:, 0] = 1
    with pytest.raises(HoodStaticReassertError, match="frame 0"):
        reassert_frame1(
            winner_camera=winner,
            base_camera=base,
            camera_support=np.ones((1, 2, 2), dtype=bool),
        )


def test_expand_support_uses_nearest_cell_mapping() -> None:
    support = np.array([[True, False], [False, True]])
    expanded = expand_support_to_camera(
        support, batch_size=2, camera_hw=(4, 4)
    )
    assert expanded.shape == (2, 4, 4)
    assert expanded[0, :2, :2].all()
    assert expanded[0, :2, 2:].sum() == 0
    assert expanded[0, 2:, 2:].all()
    assert np.array_equal(expanded[0], expanded[1])


def test_class_transition_rows_separates_recovery_from_introduction() -> None:
    target = np.array([[[0, 0], [1, 1]]], dtype=np.uint8)
    before = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
    after = np.array([[[0, 1], [1, 1]]], dtype=np.uint8)
    rows = class_transition_rows(
        before=before, after=after, target=target, class_names={0: "Road", 1: "MyCar"}
    )
    assert rows["Road"]["errors_corrected"] == 1
    assert rows["Road"]["errors_introduced"] == 1
    assert rows["MyCar"]["errors_corrected"] == 1
    assert rows["MyCar"]["errors_introduced"] == 0


def test_support_derivation_does_not_hardcode_the_hood_class() -> None:
    cells = np.zeros((600, 8, 8), dtype=np.uint8)
    cells[:, :4] = 1
    cells[:, 6:] = 2
    cells[:, 4:6] = np.arange(600, dtype=np.uint16)[:, None, None] % 2
    supports = derive_hood_supports(cells)
    assert supports.hood_class == 2
    assert supports.static[6:].all()
    assert not supports.static[:6].any()
