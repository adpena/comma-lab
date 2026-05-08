"""Regression test for the generated unified-winners ``inflate.sh``.

Codex HIGH finding #1 (2026-05-08): the previous wrapper used a 2-arg
``python inflate.py "$1" "$2"`` invocation, but the contest auth evaluator
calls inflate scripts as ``inflate.sh <data_dir> <output_dir> <file_list>``.
The 2-arg wrapper would have passed the extracted archive *directory* as
``src.bin`` and ignored the file list entirely, failing exact eval before
producing any ``.raw`` outputs.

This test exercises the new 3-arg ``inflate.sh`` end-to-end with a stub
``inflate.py`` that simply copies bytes. We verify that:

  * the script accepts ``<data_dir> <output_dir> <file_list>``,
  * it iterates the file list,
  * it locates the source via ``${DATA_DIR}/x`` (PR101 single-member
    convention) AND falls back to ``${DATA_DIR}/${BASE}.bin``,
  * it produces ``${OUTPUT_DIR}/${BASE}.raw`` for each line.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from tools.pr101_unified_winners_stack_empirical import (
    INFLATE_SH_CANONICAL,
)


_STUB_INFLATE_PY = """#!/usr/bin/env python
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
print(f"stub-inflate copied {src} -> {dst}")
"""


def _stage_inflate_assets(submission_dir: Path) -> None:
    submission_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = submission_dir / "inflate.sh"
    inflate_py = submission_dir / "inflate.py"
    inflate_sh.write_text(INFLATE_SH_CANONICAL, encoding="utf-8")
    inflate_py.write_text(_STUB_INFLATE_PY, encoding="utf-8")
    inflate_sh.chmod(inflate_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    inflate_py.chmod(inflate_py.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_inflate_sh_accepts_three_canonical_args_and_emits_raw_outputs(
    tmp_path: Path,
) -> None:
    submission_dir = tmp_path / "submission"
    _stage_inflate_assets(submission_dir)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # PR101 single-member convention: archive contents live at DATA_DIR/x.
    archive_payload = b"unified-winners-stack canonical archive payload bytes"
    (data_dir / "x").write_bytes(archive_payload)

    output_dir = tmp_path / "out"
    file_list_path = tmp_path / "file_list.txt"
    # The auth-eval file_list lists members like "0.bin" (or any "<base>.bin").
    file_list_path.write_text("0.bin\n", encoding="utf-8")

    env = os.environ.copy()
    # Force the host-python fallback path; we don't want to invoke uv during
    # unit tests. The inflate.sh should still run because it falls back to
    # plain ``python`` when UV_BIN is not executable.
    env["UV_BIN"] = "/nonexistent/uv-binary-do-not-find"

    result = subprocess.run(
        [
            "bash",
            str(submission_dir / "inflate.sh"),
            str(data_dir),
            str(output_dir),
            str(file_list_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"inflate.sh failed: rc={result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )

    # Canonical output: OUTPUT_DIR/0.raw
    out_path = output_dir / "0.raw"
    assert out_path.is_file(), f"expected {out_path} to exist; out_dir contents: {list(output_dir.iterdir()) if output_dir.exists() else 'missing'}"
    assert out_path.read_bytes() == archive_payload


def test_inflate_sh_falls_back_to_base_bin_when_x_missing(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    _stage_inflate_assets(submission_dir)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # No 'x' member; only an explicit "<base>.bin" file. Fallback path.
    base_bin_payload = b"explicit base.bin payload"
    (data_dir / "0.bin").write_bytes(base_bin_payload)

    output_dir = tmp_path / "out"
    file_list_path = tmp_path / "file_list.txt"
    file_list_path.write_text("0.bin\n", encoding="utf-8")

    env = os.environ.copy()
    env["UV_BIN"] = "/nonexistent/uv-binary-do-not-find"

    result = subprocess.run(
        [
            "bash",
            str(submission_dir / "inflate.sh"),
            str(data_dir),
            str(output_dir),
            str(file_list_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"inflate.sh fallback failed: rc={result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    out_path = output_dir / "0.raw"
    assert out_path.is_file()
    assert out_path.read_bytes() == base_bin_payload


def test_inflate_sh_fails_when_required_arg_missing(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    _stage_inflate_assets(submission_dir)

    # Omit the file_list arg; the canonical contract requires three.
    result = subprocess.run(
        [
            "bash",
            str(submission_dir / "inflate.sh"),
            str(tmp_path / "data"),
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "file list required" in (result.stderr or result.stdout)


def test_inflate_sh_text_contains_three_arg_contract() -> None:
    """Static guard: the canonical wrapper text must invoke the three-arg
    contract with the file-list iteration loop.

    This catches regression to the previous 2-arg ``$1``/``$2`` wrapper at
    the source level even if the integration tests above are skipped.
    """

    text = INFLATE_SH_CANONICAL
    assert "DATA_DIR=" in text
    assert "OUTPUT_DIR=" in text
    assert "FILE_LIST=" in text
    assert "while IFS= read -r line" in text
    assert "${BASE}.raw" in text
    # Negative guard: the two-arg form must not survive.
    assert 'exec python "$HERE/inflate.py" "$1" "$2"' not in text
