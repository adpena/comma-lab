from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.taskspace_fresh_selected_plane_codec_v1 import (
    CONFIG_SCHEMA,
    FreshSelectedPlaneCodecError,
    decode_stream_pyav,
    iter_validated_stages,
    resolve_ffmpeg_binary_identity,
    run_full_experiment,
    validate_config,
)


@dataclass(frozen=True)
class _Stage:
    pair_range: tuple[int, int]
    pair_ids: np.ndarray
    y0_u8: np.ndarray
    y1_u8: np.ndarray
    target_labels_u8: np.ndarray
    gt_poses_f32: np.ndarray


class _Provider:
    def __init__(self, stages: list[_Stage]) -> None:
        self._stages = stages

    def iter_stages(self, *, max_pairs: int = 120):
        for stage in self._stages:
            assert stage.pair_range[1] - stage.pair_range[0] <= max_pairs
            yield stage


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": CONFIG_SCHEMA,
        "research_only": True,
        "candidate_lineage_allowed": True,
        "historical_payload_reused": False,
        "pair_count": 4,
        "pairs_per_stage": 2,
        "stage_count": 2,
        "geometry": {"height": 16, "width": 16, "channels": 3},
        "representation": {
            "mode": "DIRECT_TASK_LAYERED",
            "program_residual_layered_available": False,
            "program_residual_layered_blocker": "fresh semantic predictor/base bytes are not supplied",
        },
        "operand_provider": {
            "aggregate_receipt_path": str(tmp_path / "fresh_operand_receipt.json"),
            "aggregate_receipt_sha256": "1" * 64,
        },
        "codec": {
            **resolve_ffmpeg_binary_identity("ffmpeg"),
            "encoder": "libx264rgb",
            "container": "h264",
            "input_pixel_format": "rgb24",
            "encoded_pixel_format": "rgb24",
            "decoded_pixel_format": "rgb24",
            "frame_rate": 20,
            "preset": "ultrafast",
            "threads": 1,
            "color_range": "pc",
            "colorspace": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
        },
        "endpoint": {
            "name": "fixture",
            "base_bitrate_bps": 80_000,
            "enhancement_bitrate_bps": 80_000,
        },
        "required_free_bytes": 1,
        "test_only_small_fixture": True,
    }


def _provider() -> _Provider:
    rng = np.random.default_rng(7)
    y0 = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
    y1 = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
    labels = rng.integers(0, 5, size=(4, 16, 16), dtype=np.uint8)
    poses = rng.normal(size=(4, 6)).astype(np.float32)
    return _Provider(
        [
            _Stage((start, start + 2), np.arange(start, start + 2), y0[start : start + 2], y1[start : start + 2], labels[start : start + 2], poses[start : start + 2])
            for start in (0, 2)
        ]
    )


def test_config_rejects_historical_input_and_program_residual_overclaim(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["operand_provider"] = {
        "aggregate_receipt_path": "/Volumes/VertigoDataTier/pact/evidence/c1_old/receipt.json",
        "aggregate_receipt_sha256": "1" * 64,
    }
    with pytest.raises(FreshSelectedPlaneCodecError, match="forbidden historical"):
        validate_config(config)

    config = _config(tmp_path)
    config["representation"] = {
        "mode": "DIRECT_TASK_LAYERED",
        "program_residual_layered_available": True,
        "program_residual_layered_blocker": "fresh semantic base is present",
    }
    with pytest.raises(FreshSelectedPlaneCodecError, match="cannot be claimed"):
        validate_config(config)


def test_provider_requires_exact_chronological_cover_and_current_targets(tmp_path: Path) -> None:
    config = validate_config(_config(tmp_path))
    stages = list(iter_validated_stages(_provider(), config))
    assert [row[0] for row in stages] == [(0, 2), (2, 4)]
    assert all(row[3].dtype == np.uint8 and row[4].dtype == np.float32 for row in stages)

    provider = _provider()
    object.__setattr__(provider._stages[1], "pair_ids", np.array([3, 2]))
    with pytest.raises(FreshSelectedPlaneCodecError, match="canonical chronological"):
        list(iter_validated_stages(provider, config))


def test_full_fixture_recode_is_two_long_streams_and_pyav_authoritative(tmp_path: Path) -> None:
    config = validate_config(_config(tmp_path))
    result = run_full_experiment(
        config,
        _provider(),
        output_root=tmp_path / "output",
        repo_root=Path(__file__).resolve().parents[4],
    )
    assert result["pair_count"] == 4
    assert result["stage_count"] == 2
    assert result["representation_mode"] == "DIRECT_TASK_LAYERED"
    assert result["program_residual_layered"]["v15_composition_claim"] is False
    assert result["pointer_delta"] == "UNMOVED"
    assert result["dynamic_frontier"]["target_score"] < 0.18
    assert result["public_decode"]["authority"] == "public-runtime parse-back"
    assert result["upstream_pyav_lock"]["version"] == "17.0.0"
    assert result["pose_custody"] == "SEALED_SOURCE_CACHE_ADVISORY_ONLY"
    assert result["pose_authority"] is False

    final = result["counted_stream_bundle"]
    assert len(final["manifest"]["streams"]) == 2
    assert {row["frame_count"] for row in final["manifest"]["streams"]} == {4}
    assert {
        tuple(row["actual_native_pixel_formats"])
        for row in final["manifest"]["streams"]
    } == {("gbrp",)}
    assert {
        tuple(row["rgb_conversion_paths"])
        for row in final["manifest"]["streams"]
    } == {("native-gbrp-plane-extraction-and-rgb-reorder.v1",)}
    assert final["manifest"]["pose_authority"] is False
    assert final["manifest"]["public_raw_contract"]["realization_helper"].endswith(
        "realize_factor2_uint8_scorer_plane"
    )
    assert final["manifest"]["public_raw_contract"]["chronological_order"] == (
        "pair_id ascending, frame0 from Y0 then frame1 from Y1"
    )
    stream_path = tmp_path / "output" / "final" / "y1_base.h264"
    frames, receipt = decode_stream_pyav(stream_path, expected_frame_count=4, height=16, width=16)
    assert frames.shape == (4, 16, 16, 3)
    assert receipt["decoder"]["module"] == "av"

    resumed = run_full_experiment(
        config,
        _provider(),
        output_root=tmp_path / "output",
        repo_root=Path(__file__).resolve().parents[4],
    )
    assert resumed["counted_stream_bundle"]["sha256"] == final["sha256"]
