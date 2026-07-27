from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "run_taskspace_program_residual_n600.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "run_taskspace_program_residual_n600",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_requires_typed_config_and_resume_root() -> None:
    module = _load_tool()
    args = module.build_parser().parse_args(
        [
            "--config",
            "/config.json",
            "--resume-from",
            "/Volumes/VertigoDataTier/pact/g63",
        ]
    )
    assert args.config == Path("/config.json")
    assert args.resume_from == Path("/Volumes/VertigoDataTier/pact/g63")
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--config", "/config.json"])


def test_runner_is_governed_and_terminal_refusal_is_explicit() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")
    assert 'assert_governed_admission("taskspace_program_residual_n600")' in source
    assert "return 2" in source
    assert "run_structural_producer" in source
    assert "upstream/evaluate.py" not in source
    assert "compute_contest_score" not in source
