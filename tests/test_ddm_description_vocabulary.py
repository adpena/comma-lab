from __future__ import annotations

import numpy as np

from tac.optimization.ddm_description_vocabulary import (
    HEIGHT,
    WIDTH,
    decode_boundary_worldsheet_spline,
    decode_derivation,
    decode_joint_ground_vocabulary,
    decode_persistent_level_set,
    decode_turning_angle_curves,
    encode_derivation,
    encode_joint_ground_vocabulary,
    encode_persistent_level_set,
    fit_boundary_worldsheet_spline,
    fit_persistent_level_set,
    fit_turning_angle_curves,
    inspect_coded_derivation,
)


def _labels() -> np.ndarray:
    labels = np.full((5, HEIGHT, WIDTH), 2, dtype=np.uint8)
    rows = np.arange(HEIGHT)[:, None]
    columns = np.arange(WIDTH)[None, :]
    for pair in range(labels.shape[0]):
        horizon = 145 + pair * 2 + (columns - WIDTH // 2) ** 2 // 9000
        labels[pair][rows >= horizon] = 0
        labels[pair, 240:265, 220 + pair : 255 + pair] = 3
        labels[pair, 330:, 180:330] = 4
    return labels


def test_real_coder_envelope_roundtrip_and_joint_stream() -> None:
    raw = (b"typed-description-string-" * 400) + bytes(range(64))
    first = encode_derivation("persistent_level_set", raw)
    second = encode_derivation("boundary_worldsheet_spline", raw[::-1])
    assert decode_derivation(first.envelope) == ("persistent_level_set", raw)
    assert inspect_coded_derivation(first.envelope) == first
    assert decode_derivation(second.envelope) == (
        "boundary_worldsheet_spline",
        raw[::-1],
    )
    joint = encode_joint_ground_vocabulary([first, second])
    kind, decoded = decode_derivation(joint.envelope)
    assert kind == "joint_ground_vocabulary"
    assert decoded.startswith(b"DVJG1")
    assert joint.counted_bytes == len(joint.envelope)
    assert decode_joint_ground_vocabulary(joint.envelope) == (first, second)


def test_persistent_level_set_roundtrip() -> None:
    labels = _labels()
    field = fit_persistent_level_set(labels)
    payload = encode_persistent_level_set(field)
    assert np.array_equal(decode_persistent_level_set(payload.envelope), field)
    assert field.shape == (HEIGHT, WIDTH)


def test_boundary_worldsheet_spline_is_typed_and_deterministic() -> None:
    labels = _labels()
    payload, rendered, metadata = fit_boundary_worldsheet_spline(
        labels,
        temporal_stride=2,
        horizontal_stride=32,
    )
    decoded, decoded_metadata = decode_boundary_worldsheet_spline(payload.envelope)
    assert np.array_equal(decoded, rendered)
    assert decoded_metadata == metadata
    assert rendered.shape == (labels.shape[0], HEIGHT, WIDTH)
    assert rendered.dtype == np.bool_
    assert 0 < np.count_nonzero(rendered) < rendered.size


def test_turning_angle_curve_is_typed_and_deterministic() -> None:
    labels = _labels()
    payload, rendered, metadata = fit_turning_angle_curves(
        labels,
        epsilon_pixels=2.0,
        heading_bins=32,
    )
    decoded, decoded_metadata = decode_turning_angle_curves(payload.envelope)
    assert np.array_equal(decoded, rendered)
    assert decoded_metadata == metadata
    assert metadata.contour_count >= labels.shape[0]
    assert metadata.segment_count >= metadata.contour_count * 3
    target = labels == 0
    intersection = np.count_nonzero(rendered & target)
    union = np.count_nonzero(rendered | target)
    assert intersection / union > 0.80
