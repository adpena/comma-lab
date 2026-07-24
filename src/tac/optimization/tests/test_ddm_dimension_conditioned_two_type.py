# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_dimension_conditioned_two_type import (
    EVENT_SENTINEL,
    IDENTICAL_CONTENT_CODER_CONTROL,
    IDENTITY_EUCLIDEAN_CONTROL,
    METRIC_ACTIVE_SCORER_GEOMETRY,
    DimensionConditionedTwoTypeError,
    decode_flat_event_content,
    decode_temporal_event_skeleton,
    encode_flat_event_content,
    encode_temporal_event_skeleton,
    moment_constrained_hood_projection,
    race_event_coders,
    real_decode,
    resolve_formulation_metric_disposition,
)


def _events() -> np.ndarray:
    value = np.full((7, 4, 5), EVENT_SENTINEL, dtype=np.uint8)
    value[[0, 2, 3, 6], 1, 2] = 1
    value[[1, 5], 3, 4] = 13
    value[4, 0, 0] = 20
    return value


def test_event_skeleton_and_flat_are_exact_equal_content() -> None:
    source = _events()
    program = encode_temporal_event_skeleton(source)
    flat = encode_flat_event_content(source)
    assert np.array_equal(decode_temporal_event_skeleton(program), source)
    assert np.array_equal(decode_flat_event_content(flat), source)
    race = race_event_coders(source)
    assert race.event_count == 7
    assert real_decode(race.program_coded.payload) == race.program_raw
    assert real_decode(race.flat_coded.payload) == race.flat_raw


def test_event_skeleton_refuses_nonflip_and_trailing_bytes() -> None:
    source = _events()
    source[0, 0, 0] = 0
    with pytest.raises(DimensionConditionedTwoTypeError, match="non-flip"):
        encode_temporal_event_skeleton(source)
    with pytest.raises(DimensionConditionedTwoTypeError, match="trailing"):
        decode_temporal_event_skeleton(
            encode_temporal_event_skeleton(_events()) + b"x"
        )


def test_moment_projection_preserves_frame0_and_outside_support() -> None:
    rng = np.random.default_rng(1234)
    base = rng.integers(0, 256, size=(2, 2, 6, 8, 3), dtype=np.uint8)
    winner = base.copy()
    winner[:, 1, 2:6, 1:7] = np.clip(
        winner[:, 1, 2:6, 1:7].astype(np.int16) + 30, 0, 255
    ).astype(np.uint8)
    support = np.zeros((2, 6, 8), dtype=bool)
    support[:, 2:6, 1:7] = True
    result = moment_constrained_hood_projection(
        base_camera=base,
        winner_camera=winner,
        camera_support=support,
        alpha=0.75,
    )
    assert np.array_equal(result[:, 0], base[:, 0])
    assert np.array_equal(result[:, 1][~support], winner[:, 1][~support])


def test_moment_projection_rejects_frame0_drift() -> None:
    base = np.zeros((1, 2, 2, 2, 3), dtype=np.uint8)
    winner = base.copy()
    winner[:, 0] = 1
    with pytest.raises(DimensionConditionedTwoTypeError, match="frame 0"):
        moment_constrained_hood_projection(
            base_camera=base,
            winner_camera=winner,
            camera_support=np.ones((1, 2, 2), dtype=bool),
        )


def test_metric_law_excludes_identity_control_from_verdicts() -> None:
    disposition = resolve_formulation_metric_disposition(
        IDENTITY_EUCLIDEAN_CONTROL,
        identical_content_proven=False,
    )
    assert not disposition.verdict_eligible
    assert not disposition.waterfill_eligible


def test_metric_law_allows_measured_geometry_and_exact_content_control() -> None:
    active = resolve_formulation_metric_disposition(
        METRIC_ACTIVE_SCORER_GEOMETRY,
        identical_content_proven=False,
    )
    coder = resolve_formulation_metric_disposition(
        IDENTICAL_CONTENT_CODER_CONTROL,
        identical_content_proven=True,
    )
    assert active.verdict_eligible and active.waterfill_eligible
    assert coder.verdict_eligible and not coder.waterfill_eligible
    with pytest.raises(
        DimensionConditionedTwoTypeError, match="identical-content"
    ):
        resolve_formulation_metric_disposition(
            IDENTICAL_CONTENT_CODER_CONTROL,
            identical_content_proven=False,
        )
