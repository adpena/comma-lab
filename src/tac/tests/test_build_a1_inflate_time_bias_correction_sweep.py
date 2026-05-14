# SPDX-License-Identifier: MIT
"""Tests for the A1 inflate-time bias-correction sweep builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_a1_inflate_time_bias_correction_sweep.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("a1_bias_sweep_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_inflate_py_replaces_existing_bias_block() -> None:
    tool = load_tool()
    template = "\n".join(
        [
            "def inflate():",
            "    with torch.inference_mode():",
            "        for i in range(1):",
            "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)",
            "            up[:, 0, 0].sub_(1.0)",
            "            up[:, 0, 2].sub_(1.0)",
            "            up[:, 1, 1].sub_(1.0)",
            "            frames = (",
            "                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)",
            "            )",
            "",
        ]
    )

    out = tool.build_inflate_py(template, ["up[:, 0, 0].add_(1.0)"])

    assert "up[:, 0, 0].add_(1.0)" in out
    assert "up[:, 0, 2].sub_(1.0)" not in out
    assert "up[:, 1, 1].sub_(1.0)" not in out
    assert "            frames = (" in out


def test_build_inflate_py_can_emit_no_bias_control() -> None:
    tool = load_tool()
    template = "\n".join(
        [
            "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)",
            "            up[:, 0, 0].sub_(1.0)",
            "            frames = (",
            "                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)",
            "            )",
            "",
        ]
    )

    out = tool.build_inflate_py(template, [])

    assert "sub_(1.0)" not in out
    assert "            frames = (" in out


def test_manifest_path_allows_repo_relative_and_external_tmp_paths(tmp_path: Path) -> None:
    tool = load_tool()

    assert tool.manifest_path(REPO_ROOT / "tools" / "x.py") == "tools/x.py"
    assert tool.manifest_path(tmp_path / "x.py") == str(tmp_path / "x.py")


def test_write_variant_outside_repo_records_absolute_paths(tmp_path: Path) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists() or not tool.PR101_INFLATE.exists():
        pytest.skip("A1/PR101 forensic artifacts are not present in this checkout")

    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    manifest = tool.write_variant(
        tool.VARIANTS[1],
        tmp_path,
        "TEST",
        template,
    )

    assert manifest["archive_path"].startswith(str(tmp_path))
    assert manifest["inflate_py_path"].startswith(str(tmp_path))
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["runtime_smoke_checked"] is False
    assert manifest["dispatch_blockers"]
    assert manifest["archive_sha256"] == tool.A1_EXPECTED_ARCHIVE_SHA
    inflate_sh = (
        tmp_path
        / "a1_bias_correction_sweep_v1_pr101_baseline_TEST"
        / "submission_dir"
        / "inflate.sh"
    )
    assert inflate_sh.stat().st_mode & 0o111
