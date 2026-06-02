# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.analysis.source_marker_scan import read_python_source_for_marker_scan


def test_marker_scan_ignores_comments_and_docstrings(tmp_path: Path) -> None:
    source = tmp_path / "fake_parity.py"
    source.write_text(
        '''
"""SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF in module prose."""

# SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF in a comment.

def fake():
    """SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF in function prose."""
    return "not_the_marker"
''',
        encoding="utf-8",
    )

    scanned = read_python_source_for_marker_scan(source)

    assert "SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF" not in scanned
    assert "not_the_marker" in scanned


def test_marker_scan_preserves_executable_constants_and_identifiers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real_parity.py"
    source.write_text(
        '''
SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF = "receiver_closed"

def SNERV_LF_PAYLOAD_INTN_CODEC_PROOF():
    return "codec proof"
''',
        encoding="utf-8",
    )

    scanned = read_python_source_for_marker_scan(source)

    assert "SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF" in scanned
    assert "SNERV_LF_PAYLOAD_INTN_CODEC_PROOF" in scanned


def test_marker_scan_directory_excludes_named_files(tmp_path: Path) -> None:
    (tmp_path / "kept.py").write_text(
        'REAL_RUNTIME_PROOF = "present"\n',
        encoding="utf-8",
    )
    (tmp_path / "excluded.py").write_text(
        'EXCLUDED_PROOF = "present"\n',
        encoding="utf-8",
    )

    scanned = read_python_source_for_marker_scan(
        tmp_path,
        exclude_names=("excluded.py",),
    )

    assert "REAL_RUNTIME_PROOF" in scanned
    assert "EXCLUDED_PROOF" not in scanned
