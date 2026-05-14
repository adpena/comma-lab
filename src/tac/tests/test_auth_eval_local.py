# SPDX-License-Identifier: MIT
"""Tests for tools/auth_eval_local.py.

These tests verify the local-eval wrapper without invoking GPU. Subprocess
calls to contest_auth_eval.py are mocked out; archive packing is exercised
end-to-end with byte-level assertions.
"""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "auth_eval_local.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "auth_eval_local_mod", str(TOOL_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["auth_eval_local_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_module_imports_clean(mod):
    assert hasattr(mod, "run_local_eval")
    assert hasattr(mod, "_build_archive_from_dir")
    assert hasattr(mod, "_detect_archive_members")


def test_detect_archive_members_finds_top_level(mod, tmp_path):
    (tmp_path / "renderer.bin").write_bytes(b"r")
    (tmp_path / "masks.mkv").write_bytes(b"m")
    (tmp_path / "optimized_poses.pt").write_bytes(b"p")
    members = mod._detect_archive_members(tmp_path)
    names = sorted(m.name for m in members)
    assert names == ["masks.mkv", "optimized_poses.pt", "renderer.bin"]


def test_detect_archive_members_finds_nested(mod, tmp_path):
    nested = tmp_path / "workspace"
    nested.mkdir()
    (nested / "renderer.bin").write_bytes(b"r")
    members = mod._detect_archive_members(tmp_path)
    assert any(m.name == "renderer.bin" for m in members)


def test_build_archive_from_dir_creates_deterministic_zip(mod, tmp_path):
    (tmp_path / "renderer.bin").write_bytes(b"hello")
    (tmp_path / "masks.mkv").write_bytes(b"world")
    out = tmp_path / "out.zip"
    mod._build_archive_from_dir(tmp_path, out)
    assert out.exists()
    # Verify the zip can be opened and contains both members.
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
        assert names == ["masks.mkv", "renderer.bin"]
        # Verify content roundtrips.
        assert zf.read("renderer.bin") == b"hello"
        assert zf.read("masks.mkv") == b"world"
        # Determinism check: every entry has the fixed 1980-01-01 mtime.
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_build_archive_from_dir_byte_deterministic_across_runs(mod, tmp_path):
    """Two builds of the same directory produce byte-identical zips."""
    (tmp_path / "renderer.bin").write_bytes(b"hello")
    (tmp_path / "masks.mkv").write_bytes(b"world")
    out1 = tmp_path / "out1.zip"
    out2 = tmp_path / "out2.zip"
    mod._build_archive_from_dir(tmp_path, out1)
    mod._build_archive_from_dir(tmp_path, out2)
    assert out1.read_bytes() == out2.read_bytes(), \
        "deterministic zip builder must produce byte-identical output"


def test_build_archive_from_dir_raises_on_empty_dir(mod, tmp_path):
    """Directory with no archive members is a hard error (not silent skip)."""
    with pytest.raises(SystemExit):
        mod._build_archive_from_dir(tmp_path, tmp_path / "out.zip")


def test_run_local_eval_emits_advisory_banner_on_cpu(mod, tmp_path, capsys):
    """device != cuda must emit a stderr WARNING banner."""
    fake_archive = tmp_path / "archive.zip"
    fake_archive.write_bytes(b"not a real zip")
    fake_inflate = tmp_path / "inflate.sh"
    fake_inflate.write_text("#!/bin/sh")
    (tmp_path / "config.env").write_text("PYTHON_INFLATE=renderer\n")
    fake_upstream = tmp_path / "upstream"
    fake_upstream.mkdir()
    (fake_upstream / "videos").mkdir()
    (fake_upstream / "videos" / "0.mkv").write_bytes(b"x")
    (fake_upstream / "evaluate.py").write_text("# stub")
    fake_video_names = tmp_path / "names.txt"
    fake_video_names.write_text("0.mkv\n")

    with patch.object(mod, "DEFAULT_GT_VIDEO", fake_upstream / "videos" / "0.mkv"):
        with patch("subprocess.run") as srun:
            srun.return_value.returncode = 0
            mod.run_local_eval(
                archive=fake_archive,
                inflate_sh=fake_inflate,
                upstream_dir=fake_upstream,
                video_names_file=fake_video_names,
                device="cpu",
            )
    err = capsys.readouterr().err
    assert "ADVISORY ONLY" in err
    assert "device=cpu" in err


def test_run_local_eval_hard_errors_on_missing_gt_video(mod, tmp_path):
    """Without 0.mkv we must FATAL out, not silent-fail downstream."""
    fake_archive = tmp_path / "archive.zip"
    fake_archive.write_bytes(b"x")
    with patch.object(mod, "DEFAULT_GT_VIDEO", tmp_path / "missing.mkv"):
        with pytest.raises(SystemExit):
            mod.run_local_eval(archive=fake_archive)


def test_run_local_eval_hard_errors_on_missing_config_env(mod, tmp_path):
    """Missing config.env triggers F5 guard with FATAL exit."""
    fake_archive = tmp_path / "archive.zip"
    fake_archive.write_bytes(b"x")
    fake_inflate = tmp_path / "submission" / "inflate.sh"
    fake_inflate.parent.mkdir()
    fake_inflate.write_text("#!/bin/sh")
    fake_upstream = tmp_path / "upstream"
    fake_upstream.mkdir()
    (fake_upstream / "videos").mkdir()
    (fake_upstream / "videos" / "0.mkv").write_bytes(b"x")
    with patch.object(mod, "DEFAULT_GT_VIDEO", fake_upstream / "videos" / "0.mkv"):
        with pytest.raises(SystemExit) as exc:
            mod.run_local_eval(
                archive=fake_archive,
                inflate_sh=fake_inflate,
                upstream_dir=fake_upstream,
            )
        assert "config.env" in str(exc.value)


def test_main_archive_dir_path_packs_and_invokes(mod, tmp_path):
    """--archive-dir must build a zip and pass it to contest_auth_eval."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "renderer.bin").write_bytes(b"r")
    (src_dir / "masks.mkv").write_bytes(b"m")
    out = tmp_path / "archive_out.zip"
    upstream = tmp_path / "upstream"
    (upstream / "videos").mkdir(parents=True)
    (upstream / "videos" / "0.mkv").write_bytes(b"x")
    (upstream / "evaluate.py").write_text("# stub")
    inflate = tmp_path / "submission" / "inflate.sh"
    inflate.parent.mkdir()
    inflate.write_text("#!/bin/sh")
    (inflate.parent / "config.env").write_text("PYTHON_INFLATE=renderer\n")
    names = tmp_path / "names.txt"
    names.write_text("0.mkv\n")

    captured: dict[str, list[str]] = {}

    def fake_subprocess_run(cmd, check=False):
        captured["cmd"] = cmd

        class R:
            returncode = 0
        return R()

    with patch.object(mod, "DEFAULT_GT_VIDEO", upstream / "videos" / "0.mkv"):
        with patch("subprocess.run", side_effect=fake_subprocess_run):
            rc = mod.main([
                "--archive-dir", str(src_dir),
                "--archive-out", str(out),
                "--inflate-sh", str(inflate),
                "--upstream-dir", str(upstream),
                "--video-names-file", str(names),
                "--device", "cuda",
            ])
    assert rc == 0
    assert out.exists(), "--archive-out should write the built zip"
    # Verify subprocess saw the right --archive arg.
    assert "--archive" in captured["cmd"]
    arch_idx = captured["cmd"].index("--archive")
    assert captured["cmd"][arch_idx + 1] == str(out)
