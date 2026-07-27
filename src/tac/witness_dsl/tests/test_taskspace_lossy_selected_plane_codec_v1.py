from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import taskspace_lossy_selected_plane_codec_v1 as codec


def _config(*, pair_count: int = 2, pairs_per_segment: int = 2, segment_count: int = 1) -> dict:
    return {
        "schema": codec.CONFIG_SCHEMA,
        "research_only": True,
        "historical_c1_encoder_oracle_only": True,
        "candidate_lineage_allowed": False,
        "pair_count": pair_count,
        "pairs_per_segment": pairs_per_segment,
        "segment_count": segment_count,
        "geometry": {"height": 64, "width": 64, "channels": 3},
        "rate_ceiling_bytes": codec.RATE_CEILING_BYTES,
        "source": {
            "archive_path": "/never/read/in/unit/test.zip",
            "archive_sha256": codec.HISTORICAL_C1_ARCHIVE_SHA256,
            "member_name": codec.HISTORICAL_C1_MEMBER,
            "member_sha256": codec.HISTORICAL_C1_MEMBER_SHA256,
        },
        "codec": {
            "ffmpeg_executable": "ffmpeg",
            "encoder": "libsvtav1",
            "container": "ivf",
            "input_pixel_format": "rgb24",
            "encoded_pixel_format": "yuv420p",
            "decoded_pixel_format": "rgb24",
            "frame_rate": 20,
            "preset": 13,
            "threads": 1,
            "color_range": "pc",
            "colorspace": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
        },
        "endpoints": [
            {
                "name": "unit",
                "direct_bitrate_bps": 100_000,
                "base_bitrate_bps": 50_000,
                "enhancement_bitrate_bps": 50_000,
            }
        ],
        "required_free_bytes": 1,
        "test_only_small_fixture": True,
    }


def _source(config: dict) -> codec.SourcePlanes:
    rng = np.random.default_rng(20260726)
    frame1 = rng.integers(
        0,
        256,
        size=(
            config["pair_count"],
            config["geometry"]["height"],
            config["geometry"]["width"],
            3,
        ),
        dtype=np.uint8,
    )
    offset = rng.integers(-20, 21, size=frame1.shape, dtype=np.int16)
    frame0 = np.clip(frame1.astype(np.int16) + offset, 0, 255).astype(np.uint8)
    return codec.SourcePlanes(
        frame0=np.ascontiguousarray(frame0),
        frame1=np.ascontiguousarray(frame1),
        archive_sha256=codec.HISTORICAL_C1_ARCHIVE_SHA256,
        member_sha256=codec.HISTORICAL_C1_MEMBER_SHA256,
        frame0_sha256=codec.sha256_array(frame0),
        frame1_sha256=codec.sha256_array(frame1),
    )


def test_conditional_transform_is_typed_uint8_and_bounded() -> None:
    rng = np.random.default_rng(7)
    frame1 = rng.integers(0, 256, size=(3, 8, 10, 3), dtype=np.uint8)
    frame0 = rng.integers(0, 256, size=(3, 8, 10, 3), dtype=np.uint8)
    enhancement = codec.conditional_enhancement(frame0, frame1)
    reconstructed = codec.reconstruct_frame0(frame1, enhancement)
    assert enhancement.dtype == np.uint8
    assert reconstructed.dtype == np.uint8
    assert enhancement.shape == frame0.shape
    assert reconstructed.shape == frame0.shape
    assert np.abs(reconstructed.astype(np.int16) - frame0.astype(np.int16)).max() <= 1


def test_interleave_preserves_pair_order() -> None:
    frame0 = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
    frame1 = np.flip(frame0, axis=2).copy()
    frames = codec.direct_interleave(frame0, frame1)
    reopened0, reopened1 = codec.direct_deinterleave(frames)
    assert np.array_equal(reopened0, frame0)
    assert np.array_equal(reopened1, frame1)
    assert np.array_equal(frames[0], frame0[0])
    assert np.array_equal(frames[1], frame1[0])


def test_config_refuses_candidate_lineage_and_non_n600(tmp_path: Path) -> None:
    value = _config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert codec.load_config(path)["test_only_small_fixture"] is True

    value["candidate_lineage_allowed"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(codec.LossySelectedPlaneCodecError, match="candidate lineage"):
        codec.load_config(path)

    value = _config()
    value["test_only_small_fixture"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(codec.LossySelectedPlaneCodecError, match="600 pairs"):
        codec.load_config(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_ffmpeg_segment_resume_and_exact_operating_point_bundles(tmp_path: Path) -> None:
    config = _config()
    source = _source(config)
    spec = codec.codec_spec(config)
    ffmpeg_receipt = codec.ffmpeg_version(spec)

    for arm in codec.ARMS:
        for segment_index in range(config["segment_count"]):
            first = codec.run_segment(
                config,
                source,
                output_root=tmp_path,
                endpoint_name="unit",
                arm=arm,
                segment_index=segment_index,
                ffmpeg_receipt=ffmpeg_receipt,
            )
            resumed = codec.run_segment(
                config,
                source,
                output_root=tmp_path,
                endpoint_name="unit",
                arm=arm,
                segment_index=segment_index,
                ffmpeg_receipt=ffmpeg_receipt,
            )
            assert first == resumed
            assert first["double_decode_identical"] is True
            assert first["pair_start"] == segment_index * config["pairs_per_segment"]
            assert first["pair_stop"] == (segment_index + 1) * config["pairs_per_segment"]
        bundle = codec.build_arm_bundle(config, output_root=tmp_path, endpoint_name="unit", arm=arm)
        bundle_path = Path(bundle["bundle_path"])
        assert bundle_path.name.endswith(".diagnostic.zip")
        assert bundle_path.name != "archive.zip"
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == bundle["bundle_sha256"]
        assert bundle["bundle_bytes"] == bundle_path.stat().st_size
        assert bundle["operating_points"][0]["action_id"] == "full"
        assert all(row["exact_bundle_bytes"] > 0 for row in bundle["operating_points"])
        if arm == codec.LAYERED_ARM:
            action_ids = {row["action_id"] for row in bundle["operating_points"]}
            assert "enhancement_off_all" in action_ids
            assert "evict_enhancement_segment_00" in action_ids
            assert "evict_base_segment_00" in action_ids
            assert "evict_entire_segment_00" in action_ids


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_svtav1_explicit_lp1_repeats_identical_encoder_bytes(tmp_path: Path) -> None:
    config = _config()
    source = _source(config)
    frames = codec.direct_interleave(source.frame0, source.frame1)
    spec = codec.codec_spec(config)
    left = tmp_path / "left.ivf"
    right = tmp_path / "right.ivf"
    left_receipt = codec.encode_stream(frames, spec=spec, bitrate_bps=100_000, output_path=left)
    right_receipt = codec.encode_stream(frames, spec=spec, bitrate_bps=100_000, output_path=right)
    assert any("lp=1" in token for token in left_receipt["argv"])
    assert left_receipt["sha256"] == right_receipt["sha256"]
    assert left.read_bytes() == right.read_bytes()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("encoder", "container", "encoded_pixel_format", "parameter_token"),
    [
        ("libx265", "hevc", "yuv444p", "wpp=0"),
        ("libx264rgb", "h264", "rgb24", "sliced-threads=0"),
    ],
)
def test_chroma_faithful_encoder_contracts_are_real_and_deterministic(
    tmp_path: Path,
    encoder: str,
    container: str,
    encoded_pixel_format: str,
    parameter_token: str,
) -> None:
    config = _config()
    config["codec"] = {
        **config["codec"],
        "encoder": encoder,
        "container": container,
        "encoded_pixel_format": encoded_pixel_format,
        "preset": "medium",
    }
    config_path = tmp_path / f"{encoder}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    loaded = codec.load_config(config_path)
    source = _source(config)
    frames = codec.direct_interleave(source.frame0, source.frame1)
    spec = codec.codec_spec(loaded)
    left = tmp_path / f"left.{container}"
    right = tmp_path / f"right.{container}"
    left_receipt = codec.encode_stream(frames, spec=spec, bitrate_bps=100_000, output_path=left)
    right_receipt = codec.encode_stream(frames, spec=spec, bitrate_bps=100_000, output_path=right)
    assert any(parameter_token in token for token in left_receipt["argv"])
    assert left_receipt["sha256"] == right_receipt["sha256"]
    decoded_left, decoded_left_receipt = codec.decode_stream(
        left,
        spec=spec,
        expected_frame_count=len(frames),
        height=frames.shape[1],
        width=frames.shape[2],
    )
    decoded_right, decoded_right_receipt = codec.decode_stream(
        right,
        spec=spec,
        expected_frame_count=len(frames),
        height=frames.shape[1],
        width=frames.shape[2],
    )
    assert decoded_left_receipt["decoded_sha256"] == decoded_right_receipt["decoded_sha256"]
    assert np.array_equal(decoded_left, decoded_right)
