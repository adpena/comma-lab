# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "build_pr85_final_bias_stack_candidates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_pr85_final_bias_stack_candidates", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_stored_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def test_builds_two_member_deterministic_stack(tmp_path: Path) -> None:
    mod = _load_module()
    source = tmp_path / "source" / "archive.zip"
    fb = tmp_path / "fb" / "archive.zip"
    out = tmp_path / "out"
    _write_stored_zip(source, {"x": b"source-x"})
    _write_stored_zip(fb, {"x": b"ignored", "fb": bytes(range(100)) * 3})

    summary = mod.build_candidates(input_archives=[source], fb_archive=fb, out_dir=out)

    candidate = out / "source_fb" / "archive.zip"
    assert summary["candidate_count"] == 1
    with zipfile.ZipFile(candidate, "r") as zf:
        infos = zf.infolist()
        assert [info.filename for info in infos] == ["x", "fb"]
        assert [info.date_time for info in infos] == [(1980, 1, 1, 0, 0, 0)] * 2
        assert [info.compress_type for info in infos] == [zipfile.ZIP_STORED, zipfile.ZIP_STORED]
        assert zf.read("x") == b"source-x"
        assert zf.read("fb") == bytes(range(100)) * 3

    manifest = json.loads((out / "source_fb" / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["charged_fb_bytes"] == 300
    assert manifest["candidate"]["members"][1]["name"] == "fb"


def test_rejects_wrong_fb_length(tmp_path: Path) -> None:
    mod = _load_module()
    source = tmp_path / "source" / "archive.zip"
    fb = tmp_path / "fb" / "archive.zip"
    _write_stored_zip(source, {"x": b"source-x"})
    _write_stored_zip(fb, {"fb": b"too short"})

    try:
        mod.build_candidates(input_archives=[source], fb_archive=fb, out_dir=tmp_path / "out")
    except ValueError as exc:
        assert "300 bytes" in str(exc)
    else:
        raise AssertionError("expected wrong fb length to fail closed")


def test_rejects_missing_x_member(tmp_path: Path) -> None:
    mod = _load_module()
    source = tmp_path / "source" / "archive.zip"
    fb = tmp_path / "fb" / "archive.zip"
    _write_stored_zip(source, {"not_x": b"payload"})
    _write_stored_zip(fb, {"fb": bytes(300)})

    try:
        mod.build_candidates(input_archives=[source], fb_archive=fb, out_dir=tmp_path / "out")
    except ValueError as exc:
        assert "required member 'x'" in str(exc)
    else:
        raise AssertionError("expected missing x member to fail closed")
