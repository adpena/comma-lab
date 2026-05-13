from __future__ import annotations

import subprocess
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
    assert "--allow-missing-contract" in workflow
    assert "apt-get install -y ffmpeg" not in workflow


def test_contest_cpu_eval_workflow_fetches_pinned_upstream_snapshot() -> None:
    workflow = (REPO_ROOT / ".github/workflows/contest_cpu_eval.yml").read_text(
        encoding="utf-8"
    )
    assert "UPSTREAM_REPO_URL" in workflow
    assert "UPSTREAM_COMMIT" in workflow
    assert "git clone --no-tags --depth 1" in workflow
    assert "git -C upstream checkout --detach" in workflow
    assert "11ad728f563d8970929e8947a1cf6124ee6303e4" in workflow


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
    assert "--allow-missing-contract" in script
    assert "tail -3 >&2" in script


def test_renderer_inflate_mode_does_not_require_ffmpeg_for_empty_filelist(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "archive"
    inflated_dir = tmp_path / "inflated"
    runtime_dir = tmp_path / "runtime"
    bin_dir = tmp_path / "bin"
    archive_dir.mkdir()
    inflated_dir.mkdir()
    runtime_dir.mkdir()
    bin_dir.mkdir()
    (runtime_dir / "config.env").write_text("PYTHON_INFLATE=renderer\n", encoding="utf-8")
    (runtime_dir / "inflate.sh").write_text(
        (REPO_ROOT / "submissions/robust_current/inflate.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (runtime_dir / "inflate.sh").chmod(0o755)
    (tmp_path / "file_list.txt").write_text("", encoding="utf-8")
    uv = bin_dir / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)

    result = subprocess.run(
        [
            str(runtime_dir / "inflate.sh"),
            str(archive_dir),
            str(inflated_dir),
            str(tmp_path / "file_list.txt"),
        ],
        check=False,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "CONFIG_ENV_PATH": str(runtime_dir / "config.env"),
            "UV_BIN": "uv",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "skipping ffmpeg color-contract gate" in result.stderr
