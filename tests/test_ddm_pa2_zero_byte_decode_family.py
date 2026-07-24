# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_pa2_zero_byte_decode_family import (
    BLOCKERS,
    PA2Member,
    PA2TransformError,
    apply_member,
    apply_stack,
    blind_zero_fill,
    estimate_xihat,
    family_inventory,
    scorer_resize,
    spatial_stride2_stem_residual,
    temporal_xihat,
)
from tac.through_r.blind_coordinate import build_blind_mask


def _camera(batch: int = 1) -> np.ndarray:
    rows = np.arange(874, dtype=np.uint16)[:, None, None]
    cols = np.arange(1164, dtype=np.uint16)[None, :, None]
    channels = np.arange(3, dtype=np.uint16)[None, None, :]
    frame0 = ((rows + cols + channels * 17) % 256).astype(np.uint8)
    frame1 = ((rows + cols * 2 + channels * 29) % 256).astype(np.uint8)
    pair = np.stack((frame0, frame1))
    return np.repeat(pair[None], batch, axis=0)


def test_inventory_is_typed_and_zero_byte_boundary_is_explicit() -> None:
    value = family_inventory()
    assert value["schema"] == "ddm_pa2_zero_byte_decode_family.v1"
    assert value["score_claim"] is False
    assert len(value["executable_members"]) == 4
    assert {row["member"] for row in value["blocked_members"]} == {
        PA2Member.GAUGE_ORBIT.value,
        PA2Member.RANK4_CLASS_TONE.value,
    }
    assert all("COUNTED" in row["counted_if_supplied"] for row in value["blocked_members"])


def test_blind_zero_fill_preserves_exact_resized_scorer_input() -> None:
    source = _camera()
    output = blind_zero_fill(source)
    mask = build_blind_mask().mask
    assert np.all(output[:, :, mask, :] == 0)
    assert np.array_equal(output[:, :, ~mask, :], source[:, :, ~mask, :])
    assert np.array_equal(
        scorer_resize(output).cpu().numpy(),
        scorer_resize(source).cpu().numpy(),
    )


def test_spatial_member_changes_only_frame1_and_is_deterministic() -> None:
    source = _camera()
    first = spatial_stride2_stem_residual(source)
    second = spatial_stride2_stem_residual(source)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 0], source[:, 0])
    assert np.count_nonzero(first[:, 1] != source[:, 1]) > 0


def test_xihat_is_zero_for_identical_frames() -> None:
    source = _camera()
    source[:, 1] = source[:, 0]
    row, col = estimate_xihat(scorer_resize(source))
    assert row.tolist() == [0]
    assert col.tolist() == [0]
    assert np.array_equal(temporal_xihat(source, target_frame=0), source)
    assert np.array_equal(temporal_xihat(source, target_frame=1), source)


def test_temporal_frame_ownership_and_stack_order() -> None:
    source = _camera()
    frame0 = temporal_xihat(source, target_frame=0)
    frame1 = temporal_xihat(source, target_frame=1)
    assert np.array_equal(frame0[:, 1], source[:, 1])
    assert np.array_equal(frame1[:, 0], source[:, 0])
    stack = apply_stack(
        source,
        (
            PA2Member.TEMPORAL_XIHAT_FRAME0,
            PA2Member.BLIND_ZERO_FILL,
        ),
    )
    assert np.array_equal(
        stack,
        apply_member(frame0, PA2Member.BLIND_ZERO_FILL),
    )


@pytest.mark.parametrize("blocker", BLOCKERS)
def test_blocked_member_fails_closed(blocker: object) -> None:
    with pytest.raises(PA2TransformError, match="COUNTED"):
        apply_member(_camera(), blocker.member)  # type: ignore[attr-defined]


def test_wrong_geometry_and_target_frame_fail_closed() -> None:
    with pytest.raises(PA2TransformError, match="uint8"):
        blind_zero_fill(np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
    with pytest.raises(PA2TransformError, match="target_frame"):
        temporal_xihat(_camera(), target_frame=2)
