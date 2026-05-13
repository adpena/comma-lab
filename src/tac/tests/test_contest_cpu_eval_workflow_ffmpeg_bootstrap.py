from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_contest_cpu_eval_workflow_uses_canonical_parity_ffmpeg_bootstrap() -> None:
    workflow = (REPO_ROOT / ".github/workflows/contest_cpu_eval.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/ensure_parity_ffmpeg.sh" in workflow
    assert "SELECTED_FFMPEG=" in workflow
    assert "--env-file \"${GITHUB_ENV}\"" in workflow
    assert "--path-file \"${GITHUB_PATH}\"" in workflow
    assert "grep -q in_primaries" in workflow
    assert "apt-get install -y ffmpeg" not in workflow


def test_parity_ffmpeg_bootstrap_preserves_required_color_contract() -> None:
    script = (REPO_ROOT / "scripts/ensure_parity_ffmpeg.sh").read_text(
        encoding="utf-8"
    )
    for token in (
        "in_range",
        "out_range",
        "in_color_matrix",
        "in_primaries",
        "in_transfer",
    ):
        assert token in script
    assert "BtbN/FFmpeg-Builds" in script
    assert "FFMPEG_BIN" in script
