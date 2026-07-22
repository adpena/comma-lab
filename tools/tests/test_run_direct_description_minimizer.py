from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

from tac.optimization.direct_description_minimizer import build_direct_description_owner

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools/run_direct_description_minimizer.py"
OWNER_PATH = (
    REPO_ROOT
    / ".omx/research/direct_description_minimizer_owner_bundle_603_20260721T225631Z.json"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_direct_description_minimizer", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_prints_draft_and_optimize_refuses(tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    owner_path = tmp_path / "owner.json"
    owner_path.write_text(json.dumps(build_direct_description_owner()))
    common = [
        "--owner-manifest",
        str(owner_path),
        "--execution-allowed",
        "false",
    ]
    assert tool.main([*common, "--mode", "preflight"]) == 2
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "DRAFT_DO_NOT_FIRE"
    assert output["spawn_permitted"] is False
    assert "PREFLIGHT_REFUSE" in captured.err
    assert tool.main([*common, "--mode", "optimize"]) == 2
    assert "DRAFT_DO_NOT_FIRE" in capsys.readouterr().err


def test_true_or_operator_go_cannot_override_primary(tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    owner_path = tmp_path / "owner.json"
    owner_path.write_text(json.dumps(build_direct_description_owner()))
    assert (
        tool.main(
            [
                "--owner-manifest",
                str(owner_path),
                "--mode",
                "preflight",
                "--execution-allowed",
                "true",
            ]
        )
        == 2
    )
    assert "only compiles" in capsys.readouterr().err
    go = tmp_path / "go.json"
    go.write_text("{}")
    assert (
        tool.main(
            [
                "--owner-manifest",
                str(owner_path),
                "--mode",
                "preflight",
                "--execution-allowed",
                "false",
                "--operator-go",
                str(go),
            ]
        )
        == 2
    )
    assert "cannot supersede" in capsys.readouterr().err


def test_checked_in_owner_is_exact_compiler_output_and_refuses(capsys) -> None:
    checked_in = json.loads(OWNER_PATH.read_text(encoding="utf-8"))
    assert checked_in == build_direct_description_owner()
    tool = _load_tool()
    assert (
        tool.main(
            [
                "--owner-manifest",
                str(OWNER_PATH),
                "--mode",
                "preflight",
                "--execution-allowed",
                "false",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "PREFLIGHT_REFUSE" in captured.err


def test_exact_compiled_argv_bootstraps_repo_python_and_refuses() -> None:
    owner = json.loads(OWNER_PATH.read_text(encoding="utf-8"))
    completed = subprocess.run(
        owner["consumer_argv"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "PREFLIGHT_REFUSE" in completed.stderr
