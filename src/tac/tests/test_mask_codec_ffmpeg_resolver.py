from __future__ import annotations

from pathlib import Path

import pytest


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | 0o111)


def test_ffmpeg_resolver_honors_path_name_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tac import mask_codec

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / "ffmpeg"
    _write_executable(ffmpeg, "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("TAC_FFMPEG", "ffmpeg")

    assert mask_codec._ffmpeg_binary() == str(ffmpeg)


def test_ffmpeg_resolver_skips_broken_upstream_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tac import mask_codec

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _write_executable(upstream / "ffmpeg-new", "#!/bin/sh\necho missing libSvtAv1Enc >&2\nexit 127\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / "ffmpeg"
    _write_executable(ffmpeg, "#!/bin/sh\nexit 0\n")

    monkeypatch.delenv("TAC_FFMPEG", raising=False)
    monkeypatch.setenv("TAC_UPSTREAM_DIR", str(upstream))
    monkeypatch.setenv("PATH", str(bin_dir))

    assert mask_codec._ffmpeg_binary() == str(ffmpeg)


def test_ffmpeg_resolver_fails_closed_on_bad_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tac import mask_codec

    bad = tmp_path / "bad-ffmpeg"
    _write_executable(bad, "#!/bin/sh\nexit 127\n")
    monkeypatch.setenv("TAC_FFMPEG", str(bad))

    with pytest.raises(RuntimeError, match="TAC_FFMPEG"):
        mask_codec._ffmpeg_binary()
