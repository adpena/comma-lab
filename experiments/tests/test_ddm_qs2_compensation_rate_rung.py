from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_qs2_compensation_overlay_runtime as overlay
from experiments import ddm_qs2_compensation_rate_rung as qs2
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _measured_six_pair_deltas() -> tuple[list[int], np.ndarray]:
    pairs = [105, 176, 178, 517, 523, 532]
    deltas = np.asarray(
        [
            [3, 0, -1, 0, -1, 0, -1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, -2, 0, -2, 0, 0, -1, 0, 0, 0],
            [0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 2, -1, 1, -1, 0, 0, -1, 2, 0],
            [0, 0, 0, 0, -1, 2, 4, -1, 0, 0, -1, 0],
        ],
        dtype=np.int32,
    )
    return pairs, deltas


def test_measured_six_pair_overlay_roundtrips_in_31_bytes() -> None:
    pairs, deltas = _measured_six_pair_deltas()
    payload = overlay.encode_compensation_overlay(pairs, deltas)
    decoded_pairs, decoded_deltas = overlay.decode_compensation_overlay(payload)
    assert len(payload) == 31
    np.testing.assert_array_equal(decoded_pairs, pairs)
    np.testing.assert_array_equal(decoded_deltas, deltas)


def test_overlay_applies_real_deltas_and_preserves_unselected_rows() -> None:
    pairs, deltas = _measured_six_pair_deltas()
    base = np.arange(600 * 12, dtype=np.int32).reshape(600, 12) % 1000
    payload = overlay.encode_compensation_overlay(pairs, deltas)
    actual = overlay.apply_compensation_overlay(base, payload)
    expected = base.copy()
    expected[np.asarray(pairs)] += deltas
    np.testing.assert_array_equal(actual, expected)
    untouched = sorted(set(range(600)) - set(pairs))
    np.testing.assert_array_equal(actual[untouched], base[untouched])


def test_overlay_rejects_nonzero_padding_and_zero_alias() -> None:
    pairs, deltas = _measured_six_pair_deltas()
    payload = bytearray(overlay.encode_compensation_overlay(pairs, deltas))
    payload[-1] |= 1
    with pytest.raises(overlay.CompensationOverlayError, match="padding"):
        overlay.decode_compensation_overlay(bytes(payload))
    aliased = deltas.copy()
    aliased[0, 0] = 0
    encoded = overlay.encode_compensation_overlay(pairs, aliased)
    decoded_pairs, decoded = overlay.decode_compensation_overlay(encoded)
    np.testing.assert_array_equal(decoded_pairs, pairs)
    assert decoded[0, 0] == 0


def test_selector_split_preserves_exact_selector_and_overlay() -> None:
    pairs, deltas = _measured_six_pair_deltas()
    payload = overlay.encode_compensation_overlay(pairs, deltas)
    selector = bytes.fromhex("46304531010100000000")
    # count=1 over n600 needs two rank bytes and one label byte.
    assert overlay.selector_payload_bytes(selector) == len(selector)
    selected, compensation = overlay.split_selector_compensation(selector + payload)
    assert selected == selector
    assert compensation == payload


def test_deadzone_quantization_is_monotone_in_support() -> None:
    _, deltas = _measured_six_pair_deltas()
    counts = [np.count_nonzero(qs2.deadzone_quantize(deltas, step)) for step in (1, 2, 3, 4)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 24
    assert counts[-1] < counts[0]


def test_admission_uses_complete_component_arithmetic() -> None:
    winning = qs2.admission({"archive_delta_bytes_vs_cp135": 31})
    assert winning["complete_delta_s"] == pytest.approx(
        qs2.SEG_DELTA_S_QS1
        + qs2.POSE_DELTA_S_QS1
        + 31 * qs2.RATE_S_PER_BYTE
    )
    assert winning["admitted"] is True
    losing = qs2.admission({"archive_delta_bytes_vs_cp135": 77})
    assert losing["admitted"] is False


def test_output_override_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(qs2.QS2Error, match="governed SSD store"):
        qs2.run(tmp_path)


def test_re1t_request_preserves_pose_unknown_transport_contract() -> None:
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": "run",
        "resume_from": "run",
        "retain_pose_vectors": True,
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "score_claim": False,
        "promotion_eligible": False,
        "inputs": {
            "candidate_archive.zip": {},
            "candidate_runtime.zip": {},
            "POSE_SCREEN_RESULT.json": {},
        },
    }
    qs2.validate_re1t_request(request)
    request["pose_unmeasured"] = False
    with pytest.raises(qs2.QS2Error, match="RE1T worker contract"):
        qs2.validate_re1t_request(request)


def test_qs2_runners_do_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=qs2.REPO,
        roots=[
            Path("experiments/ddm_qs2_compensation_overlay_runtime.py"),
            Path("experiments/ddm_qs2_compensation_rate_rung.py"),
        ],
    )
    assert findings == []
