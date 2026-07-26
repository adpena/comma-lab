from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "run_taskspace_fresh_selected_plane_codec_n600.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_taskspace_fresh_selected_plane_codec_n600", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_requires_typed_config_and_ssd_output() -> None:
    module = _load_tool()
    args = module.build_parser().parse_args(
        [
            "--config",
            "/config.json",
            "--output-root",
            "/Volumes/VertigoDataTier/pact/evidence/g52-fresh",
        ]
    )
    assert args.config == Path("/config.json")
    assert args.output_root == Path("/Volumes/VertigoDataTier/pact/evidence/g52-fresh")
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--output-root", "/tmp/g52"])


def test_runner_binds_governor_fresh_loader_and_no_historical_source() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")
    assert 'assert_governed_admission("taskspace_fresh_selected_plane_codec_n600")' in source
    assert "FreshScorerPlaneOperandLoaderV1.open" in source
    assert "historical_c1" not in source.lower()
    assert '"public_decode_authority": "PyAV"' in source
