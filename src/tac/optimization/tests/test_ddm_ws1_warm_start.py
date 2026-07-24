# SPDX-License-Identifier: MIT
"""Receiver and J5-custody tests for materialized WS1 warm starts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_realized_flip_menu import (
    encode_local_statistics,
    encode_temporal_affine,
)
from tac.optimization.ddm_ws1_warm_start import (
    W_JOINT,
    W_SEG,
    compile_ws1_warm_start_archive,
    parse_ws1_warm_start_archive,
    receive_joint_descent_archive,
)
from tac.optimization.direct_description_joint_descent import lift_v15_archive
from tac.optimization.direct_description_minimizer import DirectDescriptionError

pytestmark = pytest.mark.timeout(180)

REPO = Path(__file__).resolve().parents[4]
BASE = (
    REPO
    / ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/"
    "ddm_v19c_final_n600.zip.receipt-bytes"
)


def _payload(candidate: str) -> bytes:
    if candidate == W_SEG:
        return encode_temporal_affine(
            np.ones((16, 3), dtype=np.float16),
            np.zeros((16, 3), dtype=np.float16),
        )
    return encode_local_statistics(
        np.ones((5, 16, 3), dtype=np.float16),
        np.zeros((5, 16, 3), dtype=np.float16),
    )


@pytest.mark.parametrize(
    ("candidate", "payload_bytes"),
    ((W_SEG, 204), (W_JOINT, 974)),
)
def test_ws1_archive_exact_price_parseback_and_j5_reemit(
    candidate: str,
    payload_bytes: int,
) -> None:
    base = BASE.read_bytes()
    payload = _payload(candidate)
    archive = compile_ws1_warm_start_archive(
        base,
        candidate=candidate,  # type: ignore[arg-type]
        payload=payload,
    )
    assert len(payload) == payload_bytes
    assert len(archive) == len(base) + payload_bytes
    parsed = parse_ws1_warm_start_archive(archive)
    assert parsed.exact_reemit() == archive
    assert parsed.base_archive == base
    assert parsed.payload == payload
    assert parsed.custody["scorer_weights_present"] is False
    assert parsed.custody["ground_truth_argmax_present"] is False

    lift = lift_v15_archive(archive)
    assert len(lift.parameter_names) == 368
    assert lift.exact_reemit() == archive
    received = receive_joint_descent_archive(archive)
    assert received.render_camera_pairs((0,)).shape == (1, 2, 874, 1164, 3)


def test_ws1_j5_nested_rewrap_preserves_payload_and_changes_carrier() -> None:
    archive = compile_ws1_warm_start_archive(
        BASE.read_bytes(),
        candidate=W_SEG,
        payload=_payload(W_SEG),
    )
    lift = lift_v15_archive(archive)
    lane_archive = lift.lane_seed_archive()
    parsed_before = parse_ws1_warm_start_archive(archive)
    parsed_after = parse_ws1_warm_start_archive(lane_archive)
    assert parsed_after.payload == parsed_before.payload
    assert parsed_after.base_archive != parsed_before.base_archive
    assert parsed_after.carrier_archive != parsed_before.carrier_archive
    assert receive_joint_descent_archive(lane_archive).render_camera_pairs((0,)).shape == (
        1,
        2,
        874,
        1164,
        3,
    )


def test_ws1_archive_refuses_truncated_or_ambiguous_suffix() -> None:
    base = BASE.read_bytes()
    with pytest.raises(DirectDescriptionError, match="unambiguous"):
        parse_ws1_warm_start_archive(base)
    valid = compile_ws1_warm_start_archive(
        base,
        candidate=W_SEG,
        payload=_payload(W_SEG),
    )
    with pytest.raises(DirectDescriptionError):
        parse_ws1_warm_start_archive(valid[:-1])
