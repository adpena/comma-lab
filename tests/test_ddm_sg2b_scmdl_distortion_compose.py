from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_sg2b_scmdl_distortion_compose as sg2b


def write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def test_derive_changed_sites_counts_exact_denominator(tmp_path: Path) -> None:
    base = tmp_path / "base.u8"
    candidate = tmp_path / "candidate.u8"
    write_bytes(base, bytes([0, 1, 2, 3, 4, 0]))
    write_bytes(candidate, bytes([0, 2, 2, 1, 4, 4]))
    assert sg2b.derive_changed_sites(base, candidate, expected_bytes=6) == 3


def test_derive_changed_sites_refuses_wrong_denominator(tmp_path: Path) -> None:
    base = tmp_path / "base.u8"
    candidate = tmp_path / "candidate.u8"
    write_bytes(base, b"\x00\x01")
    write_bytes(candidate, b"\x00\x01")
    with pytest.raises(sg2b.Sg2bError, match="denominator"):
        sg2b.derive_changed_sites(base, candidate, expected_bytes=3)


def test_materialize_overlay_roundtrips_dense_field(tmp_path: Path) -> None:
    shape = (3, 2, 2)
    base_array = np.zeros(shape, dtype=np.uint8)
    candidate_array = base_array.copy()
    candidate_array[1, 0, 1] = 3
    candidate_array[2, 1, 0] = 4
    base = tmp_path / "base.u8"
    candidate = tmp_path / "candidate.u8"
    base.write_bytes(base_array.tobytes())
    candidate.write_bytes(candidate_array.tobytes())
    overlay = tmp_path / "overlay.npz"
    receipt = sg2b.materialize_overlay(base, candidate, overlay, shape=shape)
    assert receipt["active_pairs"] == [1, 2]
    assert receipt["parseback_field_sha256"] == hashlib.sha256(candidate_array.tobytes()).hexdigest()
    assert receipt["parseback_matches_dense_field"] is True


def test_materialize_overlay_refuses_different_existing_payload(tmp_path: Path) -> None:
    shape = (1, 1, 2)
    base = tmp_path / "base.u8"
    candidate = tmp_path / "candidate.u8"
    base.write_bytes(b"\x00\x00")
    candidate.write_bytes(b"\x00\x01")
    overlay = tmp_path / "overlay.npz"
    overlay.write_bytes(b"different")
    with pytest.raises(sg2b.Sg2bError, match="overwrite"):
        sg2b.materialize_overlay(base, candidate, overlay, shape=shape)


def test_read_rx1_header_accepts_exact_container(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    header = struct.pack("<4sBBBBHHH", b"RX1M", 1, 2, 0, 0b11010, 13_515, 30_856, 22_010)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("p", header + b"payload")
    assert sg2b.read_rx1_header(archive_path) == sg2b.RX1_EXPECTED


def test_read_rx1_header_refuses_substitute_container(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    header = struct.pack("<4sBBBBHHH", b"F24S", 1, 2, 0, 0b11010, 13_515, 30_856, 22_010)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("p", header + b"payload")
    with pytest.raises(sg2b.Sg2bError, match="header drift"):
        sg2b.read_rx1_header(archive_path)


def test_batch_refuses_foreground_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SG2B_DETACHED_BATCH", raising=False)
    with pytest.raises(sg2b.Sg2bError, match="detached"):
        sg2b.stage_batch()


def test_runtime_path_preserves_explicit_null_runtime_name() -> None:
    assert sg2b.runtime_path(sg2b.BY_KEY["p00"]) == sg2b.BASE_RUNTIME
    assert sg2b.runtime_path(sg2b.BY_KEY["p01"]) == sg2b.STORE / "runtimes" / "p01"


def test_verify_runtime_surface_refuses_missing_source(tmp_path: Path) -> None:
    with pytest.raises(sg2b.Sg2bError, match="missing pinned source"):
        sg2b.verify_runtime_surface(tmp_path)
