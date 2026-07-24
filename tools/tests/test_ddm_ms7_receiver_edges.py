from __future__ import annotations

import math

import pytest

from tac.canonical_equations.ddm_dynamic_quantum_calibration_20260724 import (
    dynamic_quantum_calibration,
)
from tac.optimization.ddm_ms7_receiver_edges import (
    MS7ReceiverEdgesError,
    build_r0_reach_table,
    decode_coded_receiver_object,
    race_same_receiver_object,
)


def _sources() -> dict[str, object]:
    return {"test": True}


def _direct() -> dict[str, object]:
    return {
        "schema": "ddm_seg_metric_custody.direct_scorer_intrinsic.v2",
        "direct_blocks": [
            {
                "pair_id": index,
                "bucket_id": f"bucket_{index}",
                "support_count": index + 1,
            }
            for index in range(25)
        ],
    }


def _dm4() -> dict[str, object]:
    return {
        "schema": "ddm_dm4_targeted_realization_cures.v1",
        "row_count": 25,
        "rows": [
            {
                "pair_id": index,
                "bucket_id": f"bucket_{index}",
                "rgb_record": {"exact_counted_bytes": 100 + index, "parseback_exact": True},
            }
            for index in range(25)
        ],
    }


def _atlas() -> dict[int, dict[str, object]]:
    return {
        index: {
            "segmentation": {"flip_count": 1000},
            "score_mass": {"distortion_score_mass": 0.1},
        }
        for index in range(25)
    }


def test_dynamic_quantum_snaps_and_fails_closed_outside_radius() -> None:
    inside = dynamic_quantum_calibration(
        composite_r_gain=0.3,
        realized_uint8_deadzone=1.0,
        lattice=(1, 2, 4, 8, 16),
        validity_radius=2,
    )
    assert inside["unsnapped_k_star"] == 2
    assert inside["selected_k_star"] == 2
    outside = dynamic_quantum_calibration(
        composite_r_gain=0.02,
        realized_uint8_deadzone=1.0,
        lattice=(1, 2, 4, 8, 16),
        validity_radius=8,
    )
    assert outside["predicted_k_star"] is None
    assert outside["selected_k_star"] is None
    assert outside["status"] == "NULL_LATTICE_EXHAUSTED"


def test_r0_exact_join_and_null_prices() -> None:
    receipt = build_r0_reach_table(
        direct_metric=_direct(),
        dm4_receipt=_dm4(),
        atlas=_atlas(),
        sources=_sources(),
    )
    assert receipt["row_count"] == 25
    assert receipt["rows"][0]["event_mass"] == pytest.approx(0.001)
    assert receipt["rows"][0]["reach_prices"]["R1_DYNAMIC_EXISTING_COORDINATE_BYTES"] is None
    assert receipt["rows"][0]["reach_prices"]["R2_T_RESIDUAL_BYTES"] is None
    assert math.isfinite(receipt["rows"][0]["flip_weighted_S_leverage"])


def test_r0_refuses_identity_drift() -> None:
    dm4 = _dm4()
    dm4["rows"][0]["bucket_id"] = "wrong"  # type: ignore[index]
    with pytest.raises(MS7ReceiverEdgesError, match="identities differ"):
        build_r0_reach_table(
            direct_metric=_direct(),
            dm4_receipt=dm4,
            atlas=_atlas(),
            sources=_sources(),
        )


def test_same_object_coder_race_is_real_and_exact() -> None:
    raw = (b"receiver-object-" * 1024) + bytes(range(256))
    race, frames = race_same_receiver_object(raw)
    assert race["same_object_raw_sha256"]
    assert race["winner"]["framed_bytes"] <= len(raw)
    assert {row["codec"] for row in race["rows"]} == {
        "RAW_COMPACT",
        "ZLIB9",
        "RAW_LZMA1",
        "ORDER1_CONTEXT_ARITHMETIC",
        "E4_BROTLI_Q11",
        "CONSTRICTION_ORDER1_CONTEXT_ANS",
        "ZSTD19_TRAINED_DICTIONARY",
        "G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT",
    }
    for codec, frame in frames.items():
        if codec != "RAW_COMPACT":
            assert decode_coded_receiver_object(frame) == raw
    g4 = next(row for row in race["rows"] if row["codec"] == "G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT")
    assert g4["available"] is False
    assert g4["framed_bytes"] is None
