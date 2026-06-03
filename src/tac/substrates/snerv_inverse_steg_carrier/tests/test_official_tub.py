# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier import (
    OFFICIAL_SNERV_T_TUB_SCHEMA,
    OfficialTubError,
    official_output2_fusion_shape,
    prepare_official_tub_graph_inputs,
)


def test_official_tub_graph_inputs_match_haar_lowpass_contract() -> None:
    current = np.array(
        [[[0.0, 2.0, 4.0, 6.0], [8.0, 10.0, 12.0, 14.0]]],
        dtype=np.float64,
    )
    previous = current + 2.0
    next_frame = current + 4.0

    out = prepare_official_tub_graph_inputs(current, previous, next_frame)

    assert out.schema == OFFICIAL_SNERV_T_TUB_SCHEMA
    assert out.score_claim is False
    assert out.promotion_eligible is False
    expected_lf_current = np.array([[[[10.0, 18.0]]]], dtype=np.float64)
    np.testing.assert_allclose(out.lf_triplet[0:1], expected_lf_current)
    normalized = out.normalized_lf
    np.testing.assert_allclose(out.current_lf, normalized[0:1])
    np.testing.assert_allclose(
        out.prev_lowpass_over_2,
        (normalized[0:1] + normalized[1:2]) / (2.0 * np.sqrt(2.0)),
    )
    np.testing.assert_allclose(
        out.next_lowpass_over_2,
        (normalized[0:1] + normalized[2:3]) / (2.0 * np.sqrt(2.0)),
    )
    metadata = out.as_jsonable_metadata()
    assert metadata["shape_metadata"]["temporal_encoder_input_count"] == 2
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False


def test_official_tub_output2_fusion_shape_matches_source_split_concat_shuffle() -> None:
    shape = official_output2_fusion_shape(
        (1, 12, 4, 5),
        fc_hw=(2, 3),
        decoder_output_shape=(2, 18, 4, 5),
    )

    assert shape.emb_ch == 6
    assert shape.prev_half_shape == (1, 6, 4, 5)
    assert shape.next_half_shape == (1, 6, 4, 5)
    assert shape.decoder_input_shape == (2, 6, 4, 5)
    assert shape.fused_output2_shape == (2, 3, 8, 15)


def test_official_tub_rejects_non_source_inputs() -> None:
    frame = np.zeros((1, 3, 4), dtype=np.float64)
    with pytest.raises(OfficialTubError, match="spatial dims must be even"):
        prepare_official_tub_graph_inputs(frame, frame, frame)
    with pytest.raises(OfficialTubError, match="requires non-constant LF"):
        prepare_official_tub_graph_inputs(
            np.zeros((1, 4, 4)),
            np.zeros((1, 4, 4)),
            np.zeros((1, 4, 4)),
        )
    with pytest.raises(OfficialTubError, match="even temporal channels"):
        official_output2_fusion_shape((1, 5, 2, 2))
