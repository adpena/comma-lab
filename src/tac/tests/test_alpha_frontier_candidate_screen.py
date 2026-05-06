"""Tests for the local Alpha frontier candidate screen."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_frontier_candidate_screen.py"
SPEC = importlib.util.spec_from_file_location("alpha_frontier_candidate_screen", MODULE_PATH)
assert SPEC is not None
screen = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def _safe_archive(tmp_path: Path, mask_bytes: bytes = b"mask-bytes") -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
    return archive


def _tiny_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 2, 2, 1],
                [3, 3, 4, 4],
                [3, 0, 4, 4],
            ],
            [
                [0, 1, 1, 1],
                [0, 2, 2, 2],
                [3, 3, 4, 4],
                [3, 0, 0, 4],
            ],
        ],
        dtype=torch.int64,
    )


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def _base_report(tmp_path: Path) -> dict[str, Any]:
    archive = _safe_archive(tmp_path)
    member_data, source_meta = screen._read_archive_member(archive, "masks.mkv")
    config = screen.ScreenConfig(candidates=("av1",), include_raw_stats=True)
    return screen._build_screen_report_from_masks(
        masks=_tiny_masks(),
        member_data=member_data,
        source_meta=source_meta,
        config=config,
        artifact_dir=tmp_path / "artifacts",
        command=["alpha_frontier_candidate_screen.py", "--unit-test"],
    )


def test_read_archive_member_rejects_requested_traversal(tmp_path: Path) -> None:
    archive = _safe_archive(tmp_path)

    with pytest.raises(ValueError, match="unsafe archive member path"):
        screen._read_archive_member(archive, "../masks.mkv")


def test_read_archive_member_rejects_zip_slip_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("../escape", b"bad")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        screen._read_archive_member(archive, "masks.mkv")


def test_read_archive_member_rejects_hidden_sidecar(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("._masks.mkv", b"sidecar")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        screen._read_archive_member(archive, "masks.mkv")


def test_screen_report_shape_schema_and_deterministic_json(tmp_path: Path) -> None:
    report_a = _base_report(tmp_path)
    report_b = _base_report(tmp_path)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
    assert report_a["schema"] == "alpha_frontier_candidate_screen_v1"
    assert report_a["evidence_grade"] == "empirical"
    assert report_a["source"]["decoded_masks"]["shape"] == [2, 4, 4]
    assert report_a["source"]["screened_masks"]["class_histogram"] == {
        "0": 8,
        "1": 6,
        "2": 5,
        "3": 6,
        "4": 7,
    }
    assert len(report_a["candidates"]) == 1
    candidate = report_a["candidates"][0]
    assert candidate["name"] == "av1_archive_member"
    assert candidate["status"] == "ok"
    assert candidate["agreement_metrics"]["argmax_agreement"] == 1.0
    assert candidate["byte_metrics"]["bytes_delta_vs_av1_member"] == 0


def test_ffmpeg_resolver_skips_broken_upstream_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _write_executable(
        upstream / "ffmpeg-new",
        "#!/bin/sh\nprintf 'broken upstream ffmpeg\\n' >&2\nexit 127\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    good_ffmpeg = _write_executable(
        bin_dir / "ffmpeg",
        "#!/bin/sh\nprintf 'ffmpeg version fake\\n'\nexit 0\n",
    )

    monkeypatch.delenv("TAC_FFMPEG", raising=False)
    monkeypatch.setenv("TAC_UPSTREAM_DIR", str(upstream))
    monkeypatch.setenv("PATH", str(bin_dir))

    assert screen._resolve_ffmpeg_binary() == str(good_ffmpeg)


def test_ffmpeg_resolver_fails_closed_for_bad_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_ffmpeg = _write_executable(
        tmp_path / "bad-ffmpeg",
        "#!/bin/sh\nprintf 'bad explicit ffmpeg\\n' >&2\nexit 1\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "ffmpeg", "#!/bin/sh\nprintf 'ffmpeg version fake\\n'\nexit 0\n")

    monkeypatch.setenv("TAC_FFMPEG", str(bad_ffmpeg))
    monkeypatch.setenv("PATH", str(bin_dir))

    with pytest.raises(RuntimeError, match="not a usable ffmpeg"):
        screen._resolve_ffmpeg_binary()


def test_screen_report_cannot_promote_or_claim_score(tmp_path: Path) -> None:
    report = _base_report(tmp_path)

    def walk(node: Any) -> list[dict[str, Any]]:
        if isinstance(node, dict):
            out = [node]
            for value in node.values():
                out.extend(walk(value))
            return out
        if isinstance(node, list):
            out = []
            for value in node:
                out.extend(walk(value))
            return out
        return []

    for item in walk(report):
        if "score_claim" in item:
            assert item["score_claim"] is False
        if "promotion_eligible" in item:
            assert item["promotion_eligible"] is False
        if "evidence_grade" in item:
            assert item["evidence_grade"] == "empirical"

    assert report["local_screen_only"] is True
    assert report["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in report["canonical_score_source_required"]
