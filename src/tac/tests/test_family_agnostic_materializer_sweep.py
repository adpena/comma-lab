# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tools.run_family_agnostic_materializer_sweep import (
    OBSERVATION_SCHEMA,
    SWEEP_SCHEMA,
    build_materializer_empirical_sweep,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_zip(path: Path, *, payload: bytes, header_overhead: bool) -> None:
    info = zipfile.ZipInfo("payload.bin")
    info.compress_type = zipfile.ZIP_STORED
    if header_overhead:
        info.extra = b"\x7f\x7f\x04\x00abcd"
        info.comment = b"deterministic member comment"
    with zipfile.ZipFile(path, "w") as zf:
        if header_overhead:
            zf.comment = b"deterministic archive comment"
        zf.writestr(info, payload)


def _write_multi_member_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.comment = b"archive comment"
        for name, payload in {
            "renderer.bin": b"A" * 256,
            "weights.bin": b"B" * 256,
        }.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.extra = b"\x7f\x7f\x04\x00abcd"
            info.comment = f"{name} comment".encode()
            zf.writestr(info, payload)


def test_materializer_empirical_sweep_summarizes_rate_positive_and_zero(
    tmp_path: Path,
) -> None:
    positive = tmp_path / "header_overhead.zip"
    zero = tmp_path / "plain.zip"
    _write_zip(positive, payload=b"A" * 256, header_overhead=True)
    _write_zip(zero, payload=b"B" * 256, header_overhead=False)

    payload = build_materializer_empirical_sweep(
        target_kind="packet_member_zip_header_elide_v1",
        archives=[f"positive={positive}", f"zero={zero}"],
        output_dir=tmp_path / "sweep",
    )

    assert payload["schema"] == SWEEP_SCHEMA
    assert payload["target_kind"] == "packet_member_zip_header_elide_v1"
    assert payload["observation_count"] == 2
    assert payload["rate_positive_count"] == 1
    assert payload["rate_nonpositive_count"] == 1
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    rows = {row["archive_label"]: row for row in payload["observations"]}
    assert rows["positive"]["schema"] == OBSERVATION_SCHEMA
    assert rows["positive"]["axis"] == "[local-materializer-proof]"
    assert rows["positive"]["portability_contract"]["requires_gpu"] is False
    assert rows["positive"]["saved_bytes"] > 0
    assert rows["positive"]["observed_score_gain"] > 0
    assert rows["positive"]["observed_rate_gain"] == rows["positive"]["observed_score_gain"]
    assert rows["positive"]["rate_positive"] is True
    assert rows["positive"]["receiver_contract_satisfied"] is True
    assert rows["positive"]["recommended_planner_action"] == (
        "keep_rate_positive_candidate_for_inflate_parity_gate"
    )
    assert rows["zero"]["saved_bytes"] <= 0
    assert rows["zero"]["observed_score_gain"] == 0.0
    assert rows["zero"]["rate_positive"] is False
    assert "candidate_not_rate_positive" in rows["zero"]["readiness_blockers"]
    assert rows["zero"]["recommended_planner_action"] == (
        "demote_matching_archive_class_for_header_elide"
    )
    assert all(row["score_claim"] is False for row in rows.values())


def test_materializer_empirical_sweep_can_record_all_member_header_elide(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "multi.zip"
    _write_multi_member_zip(archive)

    payload = build_materializer_empirical_sweep(
        target_kind="packet_member_zip_header_elide_v1",
        archives=[f"multi={archive}"],
        output_dir=tmp_path / "sweep",
        all_members=True,
    )

    row = payload["observations"][0]
    assert row["rate_positive"] is True
    assert row["portability_contract"]["requires_gpu"] is False
    assert row["portability_contract"]["deterministic_surface"] == (
        "python_stdlib_raw_zip32_wire_rewrite"
    )
    assert row["selected_member_names"] == ["renderer.bin", "weights.bin"]
    assert row["selection_scope"] == "all_members"
    assert row["selected_elision"]["elided_header_bytes"] > 0
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_materializer_empirical_sweep_cli_writes_json_and_jsonl(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "header_overhead.zip"
    output_dir = tmp_path / "sweep"
    output_json = tmp_path / "sweep.json"
    output_jsonl = tmp_path / "observations.jsonl"
    _write_zip(archive, payload=b"A" * 256, header_overhead=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer_sweep.py"),
            "--archive",
            f"fixture={archive}",
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--observation-jsonl",
            str(output_jsonl),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert stdout_payload["schema"] == SWEEP_SCHEMA
    assert persisted["schema"] == SWEEP_SCHEMA
    assert rows[0]["schema"] == OBSERVATION_SCHEMA
    assert rows[0]["source_archive_path"] == str(archive.resolve())
    assert rows[0]["rate_positive"] is True
    assert Path(rows[0]["manifest_path"]).is_file()
    assert Path(rows[0]["candidate_archive_path"]).is_file()
    assert stdout_payload["score_claim"] is False
    assert stdout_payload["ready_for_exact_eval_dispatch"] is False
