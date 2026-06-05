# SPDX-License-Identifier: MIT
"""Legacy HiNeRV trainer archive wrapper regressions."""

from __future__ import annotations

import zipfile
from pathlib import Path

from experiments import train_substrate_hi_nerv as trainer


def test_legacy_hinerv_archive_builder_keeps_runtime_outside_zip(
    tmp_path: Path,
) -> None:
    submission_dir = tmp_path / "submission"
    payload = b"HIV1" + b"\x00" * 37

    trainer._write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(payload)
    trainer._build_archive_zip(
        tmp_path / "archive.zip",
        bin_bytes=payload,
        submission_dir=submission_dir,
    )

    with zipfile.ZipFile(tmp_path / "archive.zip") as zf:
        assert zf.namelist() == ["x"]
        assert zf.read("x") == payload
        assert "inflate.py" not in zf.namelist()
        assert "inflate.sh" not in zf.namelist()
    assert (submission_dir / "inflate.py").is_file()
    inflate_source = (submission_dir / "inflate.py").read_text(encoding="utf-8")
    assert "archive_dir / 'x'" in inflate_source
    assert "archive_dir / '0.bin'" in inflate_source
