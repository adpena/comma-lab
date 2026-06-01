# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tac.substrates.hprc.runtime_closure_repair import (
    HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA,
    repair_embedded_runtime_zip_closure,
)


def test_repair_embedded_runtime_zip_closure_adds_missing_member(
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"payload")
        zf.writestr("inflate.sh", b"#!/usr/bin/env bash\n")
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "repaired.zip"
    report_path = tmp_path / "repair.json"

    report = repair_embedded_runtime_zip_closure(
        source_archive=source,
        output_archive=output,
        add_members={"src/pkg/helper.py": helper},
        report_path=report_path,
    )

    assert report["schema"] == HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA
    assert report["runtime_closure_repair_ready_for_receiver_proof"] is True
    with zipfile.ZipFile(output) as zf:
        assert zf.read("0.bin") == b"payload"
        assert zf.read("src/pkg/helper.py") == b"VALUE = 1\n"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["output_archive"]["sha256"] == report["output_archive"]["sha256"]
    assert persisted["score_claim"] is False


def test_repair_embedded_runtime_zip_closure_refuses_existing_member(
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"payload")
    helper = tmp_path / "helper.bin"
    helper.write_bytes(b"replacement")

    report = repair_embedded_runtime_zip_closure(
        source_archive=source,
        output_archive=tmp_path / "repaired.zip",
        add_members={"0.bin": helper},
        report_path=tmp_path / "repair.json",
    )

    assert report["runtime_closure_repair_ready_for_receiver_proof"] is False
    assert report["blockers"] == ["member_already_present:0.bin"]


def test_repair_embedded_runtime_zip_closure_replaces_existing_member(
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("inflate.py", b"old")
    replacement = tmp_path / "inflate.py"
    replacement.write_bytes(b"new")

    report = repair_embedded_runtime_zip_closure(
        source_archive=source,
        output_archive=tmp_path / "repaired.zip",
        add_members={},
        replace_members={"inflate.py": replacement},
        report_path=tmp_path / "repair.json",
    )

    assert report["runtime_closure_repair_ready_for_receiver_proof"] is True
    assert report["replaced_members"][0]["old_sha256"] != report["replaced_members"][0]["sha256"]
    with zipfile.ZipFile(tmp_path / "repaired.zip") as zf:
        assert zf.read("inflate.py") == b"new"
