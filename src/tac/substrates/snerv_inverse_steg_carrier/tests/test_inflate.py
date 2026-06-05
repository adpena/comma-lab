# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV SNAR1 inflate runtime."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
)
from tac.substrates.snerv_inverse_steg_carrier.inflate import (
    CAMERA_HW,
    SnervInflateError,
    _read_archive_bytes,
    inflate_one_video,
    snerv_frames_to_raw_bytes,
    write_snerv_frames_to_raw,
)
from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (
    build_snerv_receiver_archive_proof,
)


def test_inflate_one_video_writes_camera_raw_from_full_frame_packet(tmp_path: Path) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(16, 24),
        full_frame_packet=True,
    )
    raw_path = tmp_path / "0.raw"

    inflate_one_video(archive.packet, raw_path)
    decoded_frames = decode_snerv_archive_frames(archive.packet)
    expected_raw = snerv_frames_to_raw_bytes(decoded_frames)

    assert raw_path.read_bytes() == expected_raw
    assert raw_path.stat().st_size == 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3


def test_snerv_frames_to_raw_rejects_bad_shape() -> None:
    with pytest.raises(SnervInflateError, match="expected frames"):
        snerv_frames_to_raw_bytes(np.zeros((2, 3, 16, 24), dtype=np.float32))


def test_write_snerv_frames_to_raw_chunks_resize_without_byte_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.inflate as inflate_mod

    frames = np.arange(5 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(5, 2, 3, 4, 5)
    max_resize_batch = 0
    original_resize = inflate_mod._resize_nchw_bilinear

    def wrapped_resize(arr: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
        nonlocal max_resize_batch
        max_resize_batch = max(max_resize_batch, int(arr.shape[0]))
        return original_resize(arr, out_hw=out_hw)

    monkeypatch.setattr(inflate_mod, "CAMERA_HW", (4, 5))
    monkeypatch.setattr(inflate_mod, "_resize_nchw_bilinear", wrapped_resize)
    expected = inflate_mod.snerv_frames_to_raw_bytes(frames, pair_chunk_count=99)
    max_resize_batch = 0
    out = io.BytesIO()

    frame_count = write_snerv_frames_to_raw(out, frames, pair_chunk_count=2)

    assert frame_count == 10
    assert out.getvalue() == expected
    assert max_resize_batch <= 4


def test_write_snerv_frames_to_raw_rejects_invalid_chunk_count() -> None:
    frames = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)

    with pytest.raises(SnervInflateError, match="pair_chunk_count"):
        write_snerv_frames_to_raw(io.BytesIO(), frames, pair_chunk_count=0)


def test_inflate_cli_rejects_unsafe_file_list(tmp_path: Path) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(16, 24),
        full_frame_packet=True,
    )
    archive_dir = tmp_path / "archive"
    out_dir = tmp_path / "out"
    archive_dir.mkdir()
    (archive_dir / "0.bin").write_bytes(archive.packet)
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("../escape.mkv\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tac.substrates.snerv_inverse_steg_carrier.inflate",
            str(archive_dir),
            str(out_dir),
            str(file_list),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsafe file_list" in result.stderr


def test_inflate_cli_accepts_byte_minimal_x_member(tmp_path: Path) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(16, 24),
        full_frame_packet=True,
    )
    archive_dir = tmp_path / "archive"
    out_dir = tmp_path / "out"
    archive_dir.mkdir()
    (archive_dir / "x").write_bytes(archive.packet)
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tac.substrates.snerv_inverse_steg_carrier.inflate",
            str(archive_dir),
            str(out_dir),
            str(file_list),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "0.raw").stat().st_size == 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3


def test_inflate_archive_member_reader_rejects_ambiguous_x_and_0bin(
    tmp_path: Path,
) -> None:
    (tmp_path / "x").write_bytes(b"x")
    (tmp_path / "0.bin").write_bytes(b"zero")

    with pytest.raises(SnervInflateError, match="expected exactly one"):
        _read_archive_bytes(tmp_path)


def test_inflate_module_imports_no_torch_or_scorer() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.inflate as inflate_mod

    with open(inflate_mod.__file__) as f:
        src = f.read()
    assert "import torch" not in src
    assert "load_score_exact_scorers" not in src
