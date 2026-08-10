from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_sv3_unmeasured_semantic_screen import (
    compare_rgb24_raws,
    compare_states,
    parse_sm3state,
    screen,
)


def _state_wire(state: dict[str, np.ndarray]) -> bytes:
    output = bytearray(b"SM3STATE\x01")
    output.extend(struct.pack("<I", len(state)))
    for name, value in state.items():
        array = np.asarray(value, dtype="<f4")
        encoded_name = name.encode()
        output.extend(struct.pack("<H", len(encoded_name)))
        output.extend(encoded_name)
        output.extend(struct.pack("<B", array.ndim))
        output.extend(struct.pack(f"<{array.ndim}I", *array.shape))
        output.extend(struct.pack("<Q", array.nbytes))
        output.extend(array.tobytes())
    return bytes(output)


def test_sm3state_parse_and_weight_alert(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sm3state"
    state_path.write_bytes(
        _state_wire(
            {
                "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "bias": np.array([0.5, -0.5], dtype=np.float32),
            }
        )
    )
    base = parse_sm3state(state_path)
    candidate = {name: value.copy() for name, value in base.items()}
    candidate["matrix"][0, 0] = 2.0
    result = compare_states(base, candidate, relative_l2_alert=0.05)
    assert result["tensor_denominator"] == 2
    assert result["changed_values"] == 1
    assert result["alert_fired"] is True
    assert result["alerting_tensors"] == ["matrix"]


def test_raw_screen_separates_frame_parity_and_fires_control(tmp_path: Path) -> None:
    frame_bytes = 6
    base_path = tmp_path / "base.raw"
    candidate_path = tmp_path / "candidate.raw"
    base_path.write_bytes(bytes([10] * frame_bytes * 4))
    candidate = bytearray(base_path.read_bytes())
    candidate[frame_bytes : 2 * frame_bytes] = bytes([30] * frame_bytes)
    candidate[3 * frame_bytes : 4 * frame_bytes] = bytes([20] * frame_bytes)
    candidate_path.write_bytes(candidate)
    result = compare_rgb24_raws(
        base_path,
        candidate_path,
        frame_bytes=frame_bytes,
        frame_count=4,
        catastrophic_mean_absolute=5.0,
        catastrophic_changed_rgb_fraction=0.5,
    )
    even = result["parity"]["even_pose_carrier"]
    odd = result["parity"]["odd_semantic"]
    assert even["byte_identical"] is True
    assert odd["mean_absolute_delta"] == 15.0
    assert odd["changed_rgb_pixel_fraction"] == 1.0
    assert result["catastrophic_alert_fired"] is True
    assert result["catastrophic_parities"] == ["odd_semantic"]


def test_screen_reuses_only_byte_identical_checkpointed_candidate(tmp_path: Path) -> None:
    base_state = tmp_path / "base.sm3state"
    candidate_state = tmp_path / "candidate.sm3state"
    base_state.write_bytes(_state_wire({"weight": np.array([1.0, 2.0], dtype=np.float32)}))
    candidate_state.write_bytes(
        _state_wire({"weight": np.array([10.0, 20.0], dtype=np.float32)})
    )
    base_raw = tmp_path / "base.raw"
    candidate_raw = tmp_path / "candidate.raw"
    base_raw.write_bytes(bytes([0] * 12))
    candidate_raw.write_bytes(bytes([20] * 12))
    kwargs = {
        "base_state_path": base_state,
        "candidate_states": {"control": candidate_state},
        "base_raw_path": base_raw,
        "candidate_raws": {"control": candidate_raw},
        "positive_control": "control",
        "frame_bytes": 6,
        "frame_count": 2,
        "relative_l2_alert": 0.05,
        "catastrophic_mean_absolute": 5.0,
        "catastrophic_changed_rgb_fraction": 0.5,
    }
    first = screen(**kwargs)
    resumed = screen(
        **kwargs,
        completed_candidates=first["candidates"],
    )
    assert resumed["candidates"] == first["candidates"]
    candidate_raw.write_bytes(bytes([21] * 12))
    with pytest.raises(ValueError, match="resumed RAW payload changed"):
        screen(**kwargs, completed_candidates=first["candidates"])
