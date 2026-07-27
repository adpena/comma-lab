# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
)
from tac.witness_dsl.taskspace_g76_base_preferred_exact_numerator_overlay_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    FRAME_COUNT,
    G76ExactNumeratorOverlayError,
    G76ExactNumeratorOverlayResultV1,
    parse_g76_exact_numerator_overlay_receipt,
    project_base_preferred_exact_numerator_overlay,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )


def _camera_pair() -> np.ndarray:
    pair = np.empty(
        (FRAME_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS),
        dtype=np.uint8,
    )
    for frame_index in range(FRAME_COUNT):
        for channel in range(CHANNELS):
            pair[frame_index, :, :, channel] = 31 + 29 * frame_index + 7 * channel
    return pair


def _one_channel_donor(
    base: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    donor = base.copy()
    frame_index = 1
    scorer_row = 13
    scorer_col = 17
    channel = 1
    operator = _operator()
    row_support = operator.row_supports[scorer_row]
    col_support = operator.col_supports[scorer_col]
    donor[frame_index][
        np.ix_(
            row_support.indices,
            col_support.indices,
            (channel,),
        )
    ] = np.asarray((91, 103, 127, 149), dtype=np.uint8).reshape(
        2,
        2,
        1,
    )
    return donor, (frame_index, scorer_row, scorer_col, channel)


def test_channelwise_projection_preserves_base_and_exact_donor_numerator() -> None:
    base = _camera_pair()
    donor, (frame_index, scorer_row, scorer_col, channel) = _one_channel_donor(base)
    operator = _operator()
    base_numerators = [operator.apply_numerators(base[index])[0] for index in range(FRAME_COUNT)]
    donor_numerators = [operator.apply_numerators(donor[index])[0] for index in range(FRAME_COUNT)]

    result = project_base_preferred_exact_numerator_overlay(
        base_camera_pair=base,
        donor_camera_pair=donor,
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
    )
    output_numerators = [operator.apply_numerators(result.camera_pair[index])[0] for index in range(FRAME_COUNT)]

    np.testing.assert_array_equal(result.camera_pair[0], base[0])
    np.testing.assert_array_equal(output_numerators[0], base_numerators[0])
    assert (
        output_numerators[frame_index][
            scorer_row,
            scorer_col,
            channel,
        ]
        == donor_numerators[frame_index][
            scorer_row,
            scorer_col,
            channel,
        ]
    )
    changed = np.zeros_like(
        donor_numerators[frame_index],
        dtype=np.bool_,
    )
    changed[scorer_row, scorer_col, channel] = True
    np.testing.assert_array_equal(
        output_numerators[frame_index][~changed],
        base_numerators[frame_index][~changed],
    )
    np.testing.assert_array_equal(
        result.camera_pair[~result.owned_camera_mask],
        base[~result.owned_camera_mask],
    )
    assert np.count_nonzero(result.owned_camera_mask) == 4
    assert np.count_nonzero(result.owned_camera_mask[:, :, :, 0]) == 0
    assert np.count_nonzero(result.owned_camera_mask[:, :, :, 2]) == 0
    assert result.receipt.owned_scorer_values == 1
    assert result.receipt.owned_scorer_cells == 1
    assert result.receipt.changed_numerator_values == 1
    assert (
        result.receipt.base_preferred_torch_exact_blocks
        + result.receipt.solver_budget_fallback_blocks
        + result.receipt.torch_parity_fallback_blocks
        == 1
    )
    assert result.receipt.actually_changed_camera_values <= 4


def test_native_fractional_numerator_and_receipt_parse_back_survive() -> None:
    base = _camera_pair()
    donor, (frame_index, scorer_row, scorer_col, channel) = _one_channel_donor(base)
    operator = _operator()
    donor_numerators, denominator = operator.apply_numerators(donor[frame_index])
    assert (
        int(
            donor_numerators[
                scorer_row,
                scorer_col,
                channel,
            ]
        )
        % denominator
        != 0
    )

    first = project_base_preferred_exact_numerator_overlay(
        base_camera_pair=base,
        donor_camera_pair=donor,
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
    )
    second = project_base_preferred_exact_numerator_overlay(
        base_camera_pair=base,
        donor_camera_pair=donor,
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
    )
    np.testing.assert_array_equal(first.camera_pair, second.camera_pair)
    assert first.receipt == second.receipt
    payload = first.receipt.to_receipt_bytes()
    assert parse_g76_exact_numerator_overlay_receipt(payload) == first.receipt


def test_integer_nullspace_difference_falls_back_for_live_torch_parity() -> None:
    base = np.full(
        (FRAME_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS),
        128,
        dtype=np.uint8,
    )
    donor = base.copy()
    operator = _operator()
    frame_index = 1
    scorer_row = 13
    scorer_col = 0
    channel = 0
    row_support = operator.row_supports[scorer_row]
    col_support = operator.col_supports[scorer_col]
    index = np.ix_(
        row_support.indices,
        col_support.indices,
        (channel,),
    )
    donor_values = donor[frame_index][index].reshape(-1).astype(np.int16)
    donor_values[0] += 29
    donor_values[2] -= 99
    donor[frame_index][index] = donor_values.astype(np.uint8).reshape(
        2,
        2,
        1,
    )
    base_num, denominator = operator.apply_numerators(base[frame_index])
    donor_num, donor_denominator = operator.apply_numerators(donor[frame_index])
    assert donor_denominator == denominator
    assert base_num[scorer_row, scorer_col, channel] == donor_num[scorer_row, scorer_col, channel]

    result = project_base_preferred_exact_numerator_overlay(
        base_camera_pair=base,
        donor_camera_pair=donor,
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
    )
    assert result.receipt.owned_scorer_values == 1
    assert result.receipt.changed_numerator_values == 0
    assert result.receipt.torch_parity_fallback_blocks == 1
    assert result.receipt.base_preferred_torch_exact_blocks == 0
    assert (
        result.receipt.donor_selected_torch_scorer_input_sha256
        == result.receipt.output_selected_torch_scorer_input_sha256
    )
    np.testing.assert_array_equal(result.camera_pair[0], base[0])
    np.testing.assert_array_equal(
        result.camera_pair[frame_index][index],
        donor[frame_index][index],
    )


def test_projection_refuses_noop_and_non_enum_selector() -> None:
    base = _camera_pair()
    with pytest.raises(
        G76ExactNumeratorOverlayError,
        match="changed no scorer-owned camera support",
    ):
        project_base_preferred_exact_numerator_overlay(
            base_camera_pair=base,
            donor_camera_pair=base.copy(),
            frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        )
    donor, _ = _one_channel_donor(base)
    with pytest.raises(
        G76ExactNumeratorOverlayError,
        match="exact SelectedPreimageFrameSelectorV1",
    ):
        project_base_preferred_exact_numerator_overlay(
            base_camera_pair=base,
            donor_camera_pair=donor,
            frame_selector="Y1",  # type: ignore[arg-type]
        )


def test_receipt_parser_refuses_duplicate_and_authority_drift() -> None:
    base = _camera_pair()
    donor, _ = _one_channel_donor(base)
    result = project_base_preferred_exact_numerator_overlay(
        base_camera_pair=base,
        donor_camera_pair=donor,
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
    )
    payload = result.receipt.to_receipt_bytes()
    duplicate = payload[:-1] + b',"score_claim":false}'
    with pytest.raises(
        G76ExactNumeratorOverlayError,
        match="repeats key",
    ):
        parse_g76_exact_numerator_overlay_receipt(duplicate)

    drifted = json.loads(payload)
    drifted["pose_claim"] = True
    drifted_payload = json.dumps(
        drifted,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    with pytest.raises(
        G76ExactNumeratorOverlayError,
        match="authority boundary differs",
    ):
        parse_g76_exact_numerator_overlay_receipt(drifted_payload)


def test_result_refuses_same_count_wrong_ownership_mask() -> None:
    base = _camera_pair()
    donor, _ = _one_channel_donor(base)
    result = project_base_preferred_exact_numerator_overlay(
        base_camera_pair=base,
        donor_camera_pair=donor,
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
    )
    wrong = np.zeros_like(result.owned_camera_mask)
    wrong.reshape(-1)[: result.receipt.owned_camera_values] = True
    assert not np.array_equal(wrong, result.owned_camera_mask)
    with pytest.raises(
        G76ExactNumeratorOverlayError,
        match="ownership mask hash differs",
    ):
        G76ExactNumeratorOverlayResultV1(
            camera_pair=result.camera_pair,
            owned_camera_mask=wrong,
            receipt=result.receipt,
        )
