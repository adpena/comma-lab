from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import tools.measure_dense_quotient_section as dense_measurement
from tools.measure_dense_quotient_section import MEMBER_NAME, measure_dense_section


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_measure_dense_section_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4096
    source = tmp_path / "field.npy"
    source.write_bytes(payload)

    first = measure_dense_section(
        source=source,
        output_zip=tmp_path / "first.zip",
        receipt=tmp_path / "first.json",
        required_source_sha256=_sha(payload),
        command_argv=("pytest", "measure_dense_section", "first"),
    )
    second = measure_dense_section(
        source=source,
        output_zip=tmp_path / "second.zip",
        receipt=tmp_path / "second.json",
        required_source_sha256=_sha(payload),
        command_argv=("pytest", "measure_dense_section", "second"),
    )
    assert first["zip_sha256"] == second["zip_sha256"]
    assert first["zip_bytes"] == second["zip_bytes"]
    assert first["source_sha256"] == _sha(payload)
    assert first["through_r_authority"] is False
    assert first["producer_custody"]["git_head"]
    assert first["producer_custody"]["zlib"]["runtime_version"]
    assert first["producer_custody"]["command_argv"][-1] == "first"
    assert json.loads((tmp_path / "first.json").read_text())["zip_crc_test"] == "PASS"
    with zipfile.ZipFile(tmp_path / "first.zip", "r") as archive:
        assert archive.namelist() == [MEMBER_NAME]
        assert archive.read(MEMBER_NAME) == payload


def test_measure_dense_section_refuses_drift_and_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "field.npy"
    source.write_bytes(b"real field")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        measure_dense_section(
            source=source,
            output_zip=tmp_path / "field.zip",
            receipt=tmp_path / "field.json",
            required_source_sha256="0" * 64,
            command_argv=("pytest", "measure_dense_section", "drift"),
        )

    output = tmp_path / "existing.zip"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite refused"):
        measure_dense_section(
            source=source,
            output_zip=output,
            receipt=tmp_path / "existing.json",
            required_source_sha256=None,
            command_argv=("pytest", "measure_dense_section", "overwrite"),
        )


def test_measure_dense_section_removes_uncertified_output_on_receipt_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "field.npy"
    source.write_bytes(bytes(range(64)) * 2048)
    output = tmp_path / "field.zip"

    def _fail_receipt(_path: Path, _value: dict) -> None:
        raise OSError("injected receipt failure")

    monkeypatch.setattr(dense_measurement, "_atomic_json", _fail_receipt)
    with pytest.raises(OSError, match="injected receipt failure"):
        measure_dense_section(
            source=source,
            output_zip=output,
            receipt=tmp_path / "field.json",
            required_source_sha256=None,
            command_argv=("pytest", "measure_dense_section", "receipt-failure"),
        )
    assert not output.exists()
