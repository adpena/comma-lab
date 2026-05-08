"""Regression tests for `_resolve_pr101_codec_dir` in `tools/build_pr101_finetuned_archive.py`.

PR101 intake clones (`experiments/results/public_pr101_intake_*/source/...`) place
codec.py + model.py at `submissions/hnerv_ft_microcodec/src/`, NOT at the source
dir root. The Modal A1 dispatcher (`experiments/modal_phase_a1_score_gradient_pr101.py`)
ships the entire intake source dir, so the build stage MUST resolve the nested
layout. The original `_stage_submission_dir` impl (commit pre-fix) only checked
the flat layout, causing rc=1 on Modal call_id `fc-01KR4TVY14SWW0VN07XT1B4Y2Q`
after a successful 200-epoch training run (1825 GPU-seconds wasted).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_SCRIPT_PATH = _REPO_ROOT / "tools" / "build_pr101_finetuned_archive.py"
_HARVEST_SCRIPT_PATH = _REPO_ROOT / "tools" / "harvest_modal_calls.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "build_pr101_finetuned_archive", _BUILD_SCRIPT_PATH,
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_resolve_codec_dir_flat_layout(tmp_path: Path) -> None:
    """Flat layout: codec.py + model.py at root of pr101_source_dir."""
    mod = _load_build_module()
    (tmp_path / "codec.py").write_text("# codec\n")
    (tmp_path / "model.py").write_text("# model\n")
    resolved = mod._resolve_pr101_codec_dir(tmp_path)
    assert resolved == tmp_path


def test_resolve_codec_dir_nested_pr101_intake_layout(tmp_path: Path) -> None:
    """Canonical PR101 intake layout: nested under submissions/hnerv_ft_microcodec/src/."""
    mod = _load_build_module()
    nested = tmp_path / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("# codec\n")
    (nested / "model.py").write_text("# model\n")
    resolved = mod._resolve_pr101_codec_dir(tmp_path)
    assert resolved == nested


def test_resolve_codec_dir_missing_raises_systemexit(tmp_path: Path) -> None:
    """Empty pr101_source_dir must raise SystemExit naming both attempted layouts."""
    mod = _load_build_module()
    with pytest.raises(SystemExit) as exc_info:
        mod._resolve_pr101_codec_dir(tmp_path)
    msg = str(exc_info.value)
    assert "FATAL" in msg
    assert "submissions/hnerv_ft_microcodec/src" in msg or "submissions" in msg


def test_resolve_codec_dir_partial_flat_falls_through_to_missing(tmp_path: Path) -> None:
    """Flat codec.py without model.py at root must NOT count as flat layout."""
    mod = _load_build_module()
    (tmp_path / "codec.py").write_text("# codec\n")
    # model.py missing → flat layout rejected; nested also missing
    with pytest.raises(SystemExit):
        mod._resolve_pr101_codec_dir(tmp_path)


def test_resolve_codec_dir_partial_nested_falls_through_to_missing(tmp_path: Path) -> None:
    """Nested codec.py without model.py must NOT count as nested layout."""
    mod = _load_build_module()
    nested = tmp_path / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("# codec\n")
    # nested model.py missing → rejected
    with pytest.raises(SystemExit):
        mod._resolve_pr101_codec_dir(tmp_path)


def test_resolve_codec_dir_prefers_flat_when_both_present(tmp_path: Path) -> None:
    """If a directory has both flat and nested codec.py, flat takes precedence
    (caller passed pr101_source_dir at the right level on purpose).
    """
    mod = _load_build_module()
    (tmp_path / "codec.py").write_text("# flat\n")
    (tmp_path / "model.py").write_text("# flat\n")
    nested = tmp_path / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("# nested\n")
    (nested / "model.py").write_text("# nested\n")
    resolved = mod._resolve_pr101_codec_dir(tmp_path)
    assert resolved == tmp_path


def test_harvest_modal_calls_handles_none_elapsed_and_stdout_tail() -> None:
    text = _HARVEST_SCRIPT_PATH.read_text()

    assert "elapsed_raw = result.get(\"elapsed_seconds\")" in text
    assert "isinstance(elapsed_raw, (int, float))" in text
    assert "result.get(\"stdout_tail\", \"\") or \"\"" in text
