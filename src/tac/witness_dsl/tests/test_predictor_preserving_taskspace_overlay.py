# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayError,
    overlay_g_on_predictor_camera_y1,
    parse_predictor_preserving_overlay_receipt,
)


def _base() -> np.ndarray:
    payload = hashlib.shake_256(b"predictor-preserving-overlay-base").digest(874 * 1164 * 3)
    return np.frombuffer(payload, dtype=np.uint8).reshape(1, 874, 1164, 3).copy()


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (12, 24, 36),
            (48, 60, 72),
            (84, 96, 108),
            (120, 132, 144),
            (156, 168, 180),
        )
    )


def _labels() -> tuple[np.ndarray, np.ndarray]:
    predictor = np.zeros((1, 384, 512), dtype=np.uint8)
    corrected = predictor.copy()
    corrected[0, 11, 13] = 1
    corrected[0, 200, 300] = 4
    return predictor, corrected


def _decoded(corrected: np.ndarray) -> DecodedGenerativeCorrectionV1:
    return DecodedGenerativeCorrectionV1(
        labels=corrected,
        realization_profile=_profile(),
        correction_packet_sha256="a" * 64,
    )


def test_overlay_preserves_every_unowned_camera_value_and_scorer_numerator() -> None:
    base = _base()
    predictor, corrected = _labels()
    result = overlay_g_on_predictor_camera_y1(base, predictor, _decoded(corrected))
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    base_num, denominator = operator.apply_numerators(base[0])
    result_num, result_denominator = operator.apply_numerators(result.camera_y1[0])
    changed = result.changed_label_mask[0]
    target = _decoded(corrected).paint_rgb()[0]

    assert result_denominator == denominator == result.receipt.scorer_denominator
    assert np.array_equal(result.camera_y1[0][~result.owned_camera_mask[0]], base[0][~result.owned_camera_mask[0]])
    assert np.array_equal(result_num[~changed], base_num[~changed])
    assert np.array_equal(result_num[changed], target[changed].astype(np.int64) * denominator)
    assert result.receipt.changed_scorer_cells == 2
    assert result.receipt.predictor_rgb_preserved_outside_g is True
    assert result.receipt.full_frame_palette_repaint is False
    assert result.receipt.dense_predictor_rgb_serialized is False
    assert result.receipt.scorer_invoked is False
    assert result.receipt.score_claim is False
    assert result.receipt.candidate_claim is False
    assert result.receipt.research_only is True


def test_overlay_changes_only_four_disjoint_camera_taps_per_changed_cell() -> None:
    base = _base()
    predictor, corrected = _labels()
    result = overlay_g_on_predictor_camera_y1(base, predictor, _decoded(corrected))

    assert np.count_nonzero(result.owned_camera_mask) == 8
    assert result.receipt.owned_camera_values == 24
    assert 1 <= result.receipt.actually_changed_camera_values <= 24
    assert np.count_nonzero(result.camera_y1[0] != base[0]) <= 24
    assert result.receipt.camera_values_total - result.receipt.owned_camera_values > 3_000_000


def test_repeat_overlay_and_receipt_parse_back_are_exact() -> None:
    base = _base()
    predictor, corrected = _labels()
    first = overlay_g_on_predictor_camera_y1(base, predictor, _decoded(corrected))
    second = overlay_g_on_predictor_camera_y1(base, predictor, _decoded(corrected))
    payload = first.receipt.to_receipt_bytes()

    assert first.receipt == second.receipt
    assert np.array_equal(first.camera_y1, second.camera_y1)
    assert parse_predictor_preserving_overlay_receipt(payload) == first.receipt
    assert first.receipt.receipt_sha256 == hashlib.sha256(payload).hexdigest()
    with pytest.raises(PredictorPreservingOverlayError, match="not canonical JSON"):
        parse_predictor_preserving_overlay_receipt(payload + b"\n")


def test_no_changed_semantic_cells_fails_before_repainting() -> None:
    base = _base()
    predictor, _corrected = _labels()
    with pytest.raises(PredictorPreservingOverlayError, match="changed no semantic cells"):
        overlay_g_on_predictor_camera_y1(base, predictor, _decoded(predictor.copy()))


def test_bad_camera_and_label_abis_fail_closed() -> None:
    base = _base()
    predictor, corrected = _labels()
    with pytest.raises(PredictorPreservingOverlayError, match="camera Y1"):
        overlay_g_on_predictor_camera_y1(base.astype(np.float32), predictor, _decoded(corrected))
    with pytest.raises(PredictorPreservingOverlayError, match="predictor labels"):
        overlay_g_on_predictor_camera_y1(base, predictor.astype(np.int16), _decoded(corrected))


def test_receipt_numeric_and_truth_type_smuggling_is_refused() -> None:
    predictor, corrected = _labels()
    receipt = overlay_g_on_predictor_camera_y1(_base(), predictor, _decoded(corrected)).receipt
    body = json.loads(receipt.to_receipt_bytes())
    body["changed_scorer_cells"] = 2.0
    forged_numeric = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PredictorPreservingOverlayError, match="exact integers"):
        parse_predictor_preserving_overlay_receipt(forged_numeric)

    with pytest.raises(PredictorPreservingOverlayError, match="truth labels"):
        replace(receipt, candidate_claim=True)
