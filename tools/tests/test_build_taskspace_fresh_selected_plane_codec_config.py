from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "build_taskspace_fresh_selected_plane_codec_config.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_taskspace_fresh_selected_plane_codec_config",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("contract", "encoder", "container", "pixel_format"),
    [
        ("x264rgb", "libx264rgb", "h264", "rgb24"),
        ("x265-444", "libx265", "hevc", "yuv444p"),
    ],
)
def test_build_config_seals_explicit_production_contract(
    tmp_path: Path,
    contract: str,
    encoder: str,
    container: str,
    pixel_format: str,
) -> None:
    module = _load_tool()
    config = module.build_config(
        operand_receipt=tmp_path / "fresh_operand_receipt.json",
        operand_receipt_sha256="1" * 64,
        encoder_contract=contract,
        preset="slow",
        endpoint_name="full_n600",
        base_bitrate_bps=20_000,
        enhancement_bitrate_bps=24_000,
        required_free_bytes=16_000_000_000,
    )
    assert config["pair_count"] == 600
    assert config["codec"]["encoder"] == encoder
    assert config["codec"]["container"] == container
    assert config["codec"]["encoded_pixel_format"] == pixel_format
    assert Path(config["codec"]["ffmpeg_executable"]).is_absolute()
    assert config["codec"]["ffmpeg_executable_bytes"] > 0
    assert len(config["codec"]["ffmpeg_executable_sha256"]) == 64
    assert config["codec"]["ffmpeg_version_line"].startswith("ffmpeg version ")
    assert config["representation"]["program_residual_layered_available"] is False


def test_parser_requires_measured_rate_allocation_and_exact_receipt() -> None:
    module = _load_tool()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--operand-receipt",
                "/fresh/aggregate_receipt.json",
                "--encoder-contract",
                "x264rgb",
            ]
        )
