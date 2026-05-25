# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import brotli
import numpy as np

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


def _write_recompress_zip(path: Path) -> None:
    payload = (b"ABCD" * 4096) + (b"\x00" * 4096)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "payload.bin",
            payload,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        )


def _write_merge_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in {
            "renderer.bin": b"A" * 256,
            "masks.mkv": b"B" * 128,
        }.items():
            zf.writestr(name, payload, compress_type=zipfile.ZIP_STORED)


def _write_dfl1_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in {
            "renderer.bin": b"A" * 512,
            "masks.mkv": b"B" * 384,
            "optimized_poses.pt": b"C" * 256,
        }.items():
            zf.writestr(
                name,
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _write_tensor_zip(path: Path) -> None:
    rows = np.arange(256, dtype=np.float32).reshape(256, 1)
    cols = np.linspace(0.25, 2.0, 256, dtype=np.float32).reshape(1, 256)
    matrix = rows @ cols
    payload = BytesIO()
    np.save(payload, matrix, allow_pickle=False)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("weights.npy", payload.getvalue(), compress_type=zipfile.ZIP_STORED)


def _write_section_recode_zip(path: Path) -> dict[str, object]:
    raw = (b"abcdef0123456789" * 4096) + (b"A" * 8192)
    compressed = brotli.compress(raw, quality=1)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("payload.bin", compressed, compress_type=zipfile.ZIP_STORED)
    return {
        "schema": "archive_section_manifest.v1",
        "member_name": "payload.bin",
        "sections": [
            {
                "name": "payload_brotli",
                "index": 0,
                "offset": 0,
                "length": len(compressed),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            }
        ],
    }


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


def test_materializer_empirical_sweep_supports_packet_member_recompress(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recompress.zip"
    _write_recompress_zip(archive)

    payload = build_materializer_empirical_sweep(
        target_kind="packet_member_recompress_v1",
        archives=[f"recompress={archive}"],
        output_dir=tmp_path / "sweep",
        member_name="payload.bin",
        zip_compression_methods=("deflated",),
        zip_compresslevels=(9,),
    )

    row = payload["observations"][0]
    assert payload["materializer_id"] == "packet_member_recompress_adapter"
    assert payload["planner_feedback"]["target_kind"] == "packet_member_recompress_v1"
    assert row["selected_materialization_key"] == "selected_compression"
    assert row["selected_compression"]["compression_method"] == "deflated"
    assert row["saved_bytes"] > 0
    assert row["observed_rate_gain"] > 0
    assert row["observed_score_gain"] > 0
    assert row["receiver_contract_satisfied"] is True
    assert row["score_claim"] is False
    assert Path(row["candidate_archive_path"]).is_file()


def test_materializer_empirical_sweep_supports_packet_member_merge(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "merge.zip"
    _write_merge_zip(archive)

    payload = build_materializer_empirical_sweep(
        target_kind="packet_member_merge_v1",
        archives=[f"merge={archive}"],
        output_dir=tmp_path / "sweep",
        member_names=("renderer.bin", "masks.mkv"),
        merged_member_name="p",
    )

    row = payload["observations"][0]
    assert payload["materializer_id"] == "packet_member_merge_adapter"
    assert payload["planner_feedback"]["target_kind"] == "packet_member_merge_v1"
    assert row["selected_materialization_key"] == "selected_merge"
    assert row["selected_member_names"] == ["renderer.bin", "masks.mkv"]
    assert row["rate_positive"] is True
    assert row["receiver_contract_satisfied"] is False
    assert "packet_member_merge_receiver_contract_not_satisfied" in (
        row["readiness_blockers"]
    )
    assert row["score_claim"] is False
    assert Path(row["candidate_archive_path"]).is_file()


def test_materializer_empirical_sweep_supports_renderer_payload_dfl1(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dfl1.zip"
    _write_dfl1_zip(archive)

    payload = build_materializer_empirical_sweep(
        target_kind="renderer_payload_dfl1_v1",
        archives=[f"dfl1={archive}"],
        output_dir=tmp_path / "sweep",
    )

    row = payload["observations"][0]
    assert payload["materializer_id"] == "renderer_payload_dfl1_adapter"
    assert payload["planner_feedback"]["target_kind"] == "renderer_payload_dfl1_v1"
    assert row["selected_materialization_key"] == "selected_payload"
    assert row["selected_member_names"] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.pt",
    ]
    assert row["rate_positive"] is True
    assert row["receiver_contract_satisfied"] is False
    assert "renderer_payload_dfl1_full_frame_inflate_parity_missing" in (
        row["readiness_blockers"]
    )
    assert row["score_claim"] is False
    assert Path(row["candidate_archive_path"]).is_file()


def test_materializer_empirical_sweep_supports_tensor_factorize(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "tensor.zip"
    _write_tensor_zip(archive)

    payload = build_materializer_empirical_sweep(
        target_kind="tensor_factorize_v1",
        archives=[f"tensor={archive}"],
        output_dir=tmp_path / "sweep",
        tensor_manifest={"member_name": "weights.npy"},
        factorization_contract={
            "rank": 1,
            "max_abs_error_tolerance": 0.01,
            "max_relative_error_tolerance": 0.01,
            "cooperative_receiver_id": "fixture_tensor_receiver",
            "receiver_adapter_kind": "numpy_low_rank_reconstruction",
        },
    )

    row = payload["observations"][0]
    assert payload["materializer_id"] == "tensor_factorize_adapter"
    assert payload["planner_feedback"]["target_kind"] == "tensor_factorize_v1"
    assert row["selected_materialization_key"] == "factorization"
    assert row["factorization"]["rank"] == 1
    assert row["saved_bytes"] > 0
    assert row["rate_positive"] is True
    assert row["receiver_contract_satisfied"] is True
    assert row["recommended_planner_action"] == (
        "keep_rate_positive_candidate_for_inflate_parity_gate"
    )
    assert row["score_claim"] is False
    assert Path(row["manifest_path"]).is_file()


def test_materializer_empirical_sweep_supports_archive_section_entropy_recode(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "section_recode.zip"
    section_manifest = _write_section_recode_zip(archive)

    payload = build_materializer_empirical_sweep(
        target_kind="archive_section_entropy_recode_v1",
        archives=[f"section={archive}"],
        output_dir=tmp_path / "sweep",
        section_manifest=section_manifest,
        section_names=("payload_brotli",),
        brotli_qualities=(5, 9),
    )

    row = payload["observations"][0]
    assert payload["materializer_id"] == "archive_section_entropy_recode_adapter"
    assert payload["planner_feedback"]["target_kind"] == (
        "archive_section_entropy_recode_v1"
    )
    assert row["selected_materialization_key"] == "section_recode"
    assert row["section_recode"]["selected_section_names"] == ["payload_brotli"]
    assert row["saved_bytes"] > 0
    assert row["rate_positive"] is True
    assert row["receiver_contract_satisfied"] is False
    assert row["observed_score_gain"] == 0.0
    assert "section_length_changed_requires_runtime_consumption_proof" in (
        row["readiness_blockers"]
    )
    assert row["recommended_planner_action"] == (
        "repair_receiver_contract_before_exact_readiness"
    )
    assert row["score_claim"] is False
    assert Path(row["candidate_archive_path"]).is_file()


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
