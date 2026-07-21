# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.predictor_r2_missdelta import (
    PredictorR2Error,
    analyze_frame,
    apply_refinement_policy,
    decode_boundary_delta,
    decode_shape_blobs,
    encode_boundary_delta,
    encode_shape_blobs,
    fit_refinement_policy,
    parse_refinement_policy,
    validate_resume_config,
)


def shifted_boundary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.zeros((8, 10), dtype=np.uint8)
    predicted[:, 5:] = 2
    target = predicted.copy()
    target[2:6, 4] = 2
    strata = np.zeros_like(predicted)
    strata[3:5] = 1
    return predicted, target, strata


def test_decomposition_is_exclusive_exhaustive_and_stratified() -> None:
    predicted, target, strata = shifted_boundary()
    target[0:2, 0:2] = 3
    target[7, 0] = 4
    summary, kinds = analyze_frame(predicted, target, strata)
    misses = predicted != target
    assert np.count_nonzero(kinds != 255) == np.count_nonzero(misses)
    assert set(np.unique(kinds[misses])).issubset({0, 1, 2})
    assert int(np.asarray(summary["counts"]).sum()) == int(np.count_nonzero(misses))
    assert int(summary["event_count"]) == int(np.count_nonzero(kinds == 0))
    assert np.count_nonzero(kinds == 1) == 4
    assert np.count_nonzero(kinds == 2) == 1


def test_boundary_delta_is_exact_canonical_and_fail_closed() -> None:
    predicted, target, strata = shifted_boundary()
    blob, receipt = encode_boundary_delta([predicted], [target], [strata])
    decoded = decode_boundary_delta(blob, [predicted])
    assert np.array_equal(decoded[0], target)
    assert receipt["decode_verified_exact"] is True
    assert receipt["reencode_verified_byte_identical"] is True
    assert receipt["event_count"] == 4
    with pytest.raises(PredictorR2Error, match="length mismatch"):
        decode_boundary_delta(blob + b"x", [predicted])
    damaged = bytearray(blob)
    damaged[-1] ^= 1
    with pytest.raises(PredictorR2Error, match="checksum"):
        decode_boundary_delta(bytes(damaged), [predicted])
    with pytest.raises(PredictorR2Error, match="geometry mismatch"):
        decode_boundary_delta(blob, [predicted[:, :-1]])


def test_boundary_delta_selection_corrects_only_selected_class() -> None:
    predicted, target, strata = shifted_boundary()
    target[2:6, 5] = 1
    blob, receipt = encode_boundary_delta(
        [predicted], [target], [strata], selection=(2, None)
    )
    decoded = decode_boundary_delta(blob, [predicted])[0]
    assert receipt["event_count"] == 4
    assert np.all(decoded[2:6, 4] == 2)
    assert np.all(decoded[2:6, 5] == 2)


def test_shape_sidecar_exactly_replays_only_admitted_components() -> None:
    predicted = np.zeros((9, 9), dtype=np.uint8)
    target = predicted.copy()
    target[1:3, 1:3] = 3
    target[7, 7] = 4
    kinds = np.full_like(predicted, 255)
    kinds[1:3, 1:3] = 1
    kinds[7, 7] = 2
    blob, receipt = encode_shape_blobs([predicted], [target], [kinds])
    decoded = decode_shape_blobs(blob, [predicted])[0]
    assert receipt["pixel_count"] == 4
    assert np.all(decoded[1:3, 1:3] == 3)
    assert decoded[7, 7] == 0
    assert receipt["reencode_verified_byte_identical"] is True
    with pytest.raises(PredictorR2Error, match="length mismatch"):
        decode_shape_blobs(blob + b"x", [predicted])


def test_n64_refinement_policy_roundtrips_and_improves_training_split() -> None:
    predicted, target, _ = shifted_boundary()
    payload, fit = fit_refinement_policy([predicted], [target], target_class=2)
    assert tuple(parse_refinement_policy(payload))
    refined = apply_refinement_policy(predicted, payload)
    before = int(np.count_nonzero(predicted != target))
    after = int(np.count_nonzero(refined != target))
    assert after < before
    assert fit["policy_bytes"] == len(payload)
    damaged = bytearray(payload)
    damaged[-1] ^= 1
    with pytest.raises(PredictorR2Error, match="checksum"):
        parse_refinement_policy(bytes(damaged))


def test_resume_manifest_refuses_config_drift() -> None:
    validate_resume_config({"config_sha256": "same"}, "same")
    with pytest.raises(PredictorR2Error, match="config drift"):
        validate_resume_config({"config_sha256": "old"}, "new")
