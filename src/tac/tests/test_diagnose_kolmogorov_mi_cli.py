# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_kolmogorov_mi_cli_outputs_json_for_files(tmp_path: Path) -> None:
    section = tmp_path / "section.bin"
    signal = tmp_path / "signal.bin"
    section.write_bytes((b"abcd" * 128) + (b"zz" * 64))
    signal.write_bytes(b"abcd" * 128)

    result = subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "tools" / "diagnose_kolmogorov_mi.py"),
            "--section",
            f"payload={section}",
            "--signal",
            f"seg_signal={signal}",
            "--compressor",
            "zlib",
        ],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(result.stdout)
    record = data["records"][0]

    assert data["score_claim"] is False
    assert data["promotion_eligible"] is False
    assert data["ready_for_exact_eval_dispatch"] is False
    assert data["evidence_grade"] == "proxy_diagnostic_kolmogorov_mi"
    assert data["proxy"] is True
    assert data["proxy_only"] is True
    assert data["operator_metadata"]["operator_visible"] is True
    assert "runbook" in data["operator_metadata"]
    assert record["section"] == "payload"
    assert record["signal"] == "seg_signal"
    assert record["algorithmic_mi_bytes"] > 0
    assert 0.0 <= record["normalized_compression_distance"] <= 2.0
    assert record["ready_for_exact_eval_dispatch"] is False
    assert record["evidence_grade"] == "proxy_diagnostic_kolmogorov_mi"
    assert record["proxy"] is True


def test_kolmogorov_mi_cli_reads_zip_members(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    signal = tmp_path / "signal.bin"
    signal.write_bytes(b"mask-mask-mask" * 20)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"mask-mask-mask" * 20)

    result = subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "tools" / "diagnose_kolmogorov_mi.py"),
            "--section",
            f"member={archive}::0.bin",
            "--signal",
            f"signal={signal}",
        ],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(result.stdout)

    assert data["records"][0]["section_source"].endswith("archive.zip::0.bin")
    assert data["records"][0]["raw_section_bytes"] == len(b"mask-mask-mask" * 20)
    assert data["records"][0]["ready_for_exact_eval_dispatch"] is False
