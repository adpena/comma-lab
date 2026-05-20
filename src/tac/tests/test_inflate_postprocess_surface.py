from __future__ import annotations

import numpy as np

from tac.optimization.inflate_postprocess_surface import (
    PostprocessSpec,
    RawVideoShape,
    apply_postprocess,
    builtin_specs,
    plan_payload,
    postprocess_spec_from_dict,
)


def _write_raw(path, array: np.ndarray) -> None:
    path.write_bytes(array.astype(np.uint8).tobytes())


def test_channel_bias_only_changes_selected_frames(tmp_path) -> None:
    shape = RawVideoShape(frames=4, height=2, width=3, channels=3)
    source = np.full((4, 2, 3, 3), 10, dtype=np.uint8)
    raw = tmp_path / "in.raw"
    out = tmp_path / "out.raw"
    _write_raw(raw, source)

    result = apply_postprocess(
        input_raw=raw,
        output_raw=out,
        spec=PostprocessSpec(
            spec_id="odd_minus_one",
            kind="channel_bias",
            frame_selector="odd",
            channel_deltas=(-1, -1, -1),
        ),
        shape=shape,
    )

    decoded = np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(source.shape)
    assert np.array_equal(decoded[0], source[0])
    assert np.array_equal(decoded[2], source[2])
    assert np.all(decoded[1] == 9)
    assert np.all(decoded[3] == 9)
    assert result.changed_frame_count == 2
    assert result.changed_byte_count == 2 * 2 * 3 * 3
    assert result.max_abs_delta == 1
    assert result.passed_visible_change is True


def test_channel_bias_saturates_uint8_bounds(tmp_path) -> None:
    shape = RawVideoShape(frames=1, height=1, width=2, channels=3)
    source = np.array([[[[0, 1, 255], [254, 255, 0]]]], dtype=np.uint8)
    raw = tmp_path / "in.raw"
    out = tmp_path / "out.raw"
    _write_raw(raw, source)

    apply_postprocess(
        input_raw=raw,
        output_raw=out,
        spec=PostprocessSpec(
            spec_id="sat",
            kind="channel_bias",
            channel_deltas=(-1, 1, 1),
        ),
        shape=shape,
    )

    decoded = np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(source.shape)
    assert decoded.tolist() == [[[[0, 2, 255], [253, 255, 1]]]]


def test_channel_bias_can_target_explicit_frame_indices(tmp_path) -> None:
    shape = RawVideoShape(frames=4, height=1, width=1, channels=3)
    source = np.full((4, 1, 1, 3), 10, dtype=np.uint8)
    raw = tmp_path / "in.raw"
    out = tmp_path / "out.raw"
    _write_raw(raw, source)

    result = apply_postprocess(
        input_raw=raw,
        output_raw=out,
        spec=PostprocessSpec(
            spec_id="frames_1_3_minus_one",
            kind="channel_bias",
            frame_indices=(1, 3),
            channel_deltas=(-1, -1, -1),
        ),
        shape=shape,
    )

    decoded = np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(source.shape)
    assert decoded[:, 0, 0, 0].tolist() == [10, 9, 10, 9]
    assert result.changed_frame_count == 2
    assert result.changed_byte_count == 6


def test_postprocess_spec_from_dict_roundtrips_frame_indices() -> None:
    spec = postprocess_spec_from_dict(
        {
            "spec_id": "frame_15_bias",
            "kind": "channel_bias",
            "frame_indices": [15],
            "channel_deltas": [-1, -1, -1],
        }
    )

    assert spec.frame_indices == (15,)
    assert spec.selected(15) is True
    assert spec.selected(14) is False


def test_temporal_blend_uses_neighbor_average(tmp_path) -> None:
    shape = RawVideoShape(frames=3, height=1, width=1, channels=3)
    source = np.array(
        [[[[0, 0, 0]]], [[[100, 100, 100]]], [[[160, 160, 160]]]],
        dtype=np.uint8,
    )
    raw = tmp_path / "in.raw"
    out = tmp_path / "out.raw"
    _write_raw(raw, source)

    apply_postprocess(
        input_raw=raw,
        output_raw=out,
        spec=PostprocessSpec(
            spec_id="blend",
            kind="temporal_blend",
            frame_selector="odd",
            alpha_num=1,
            alpha_den=2,
        ),
        shape=shape,
    )

    decoded = np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(source.shape)
    assert decoded[0].tolist() == [[[0, 0, 0]]]
    assert decoded[1].tolist() == [[[90, 90, 90]]]
    assert decoded[2].tolist() == [[[160, 160, 160]]]


def test_plan_payload_is_advisory_only() -> None:
    payload = plan_payload()
    assert payload["schema"] == "inflate_postprocess_surface_plan.v1"
    assert payload["authority"]["score_claim"] is False
    assert payload["authority"]["promotion_eligible"] is False
    assert "odd_luma_bias_m1" in {row["spec_id"] for row in payload["builtin_specs"]}
    assert "odd_luma_bias_p1" in builtin_specs()
