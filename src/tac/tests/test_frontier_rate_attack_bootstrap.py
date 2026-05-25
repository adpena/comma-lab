# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import pytest

from comma_lab.scheduler.byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from comma_lab.scheduler.frontier_rate_attack_bootstrap import (
    BOOTSTRAP_SCHEMA,
    DERIVED_SECTION_MANIFEST_BATCH_SCHEMA,
    FrontierRateAttackBootstrapError,
    archive_record,
    build_frontier_rate_attack_payloads,
    derive_archive_section_recode_manifests,
    parse_archive_spec,
    resolve_current_frontier_archive,
)
from tac.repo_io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]


AUTHORITY_KEYS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _write_archive(path: Path, *, member_name: str = "renderer.bin") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, b"frontier-bytes")
    return path


def _write_stored_archive(path: Path, *, member_name: str, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, payload)
    return path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key in AUTHORITY_KEYS:
        assert payload[key] is False


def test_frontier_bootstrap_builds_queue_with_experiment_metadata(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip")
    record = archive_record(
        label="frontier",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_unit",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        ),
        include_optional_target_blockers=True,
        local_cpu_concurrency=2,
    )

    bootstrap = payloads["bootstrap"]
    queue = payloads["queue"]
    backlog = payloads["backlog"]
    contexts = payloads["contexts"]
    _assert_false_authority(record)
    _assert_false_authority(bootstrap)
    assert bootstrap["schema"] == BOOTSTRAP_SCHEMA
    assert bootstrap["executable_target_count"] == 2
    assert bootstrap["experiment_count"] == 2
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"] == {"local_cpu": 2}
    assert backlog["backlog_row_count"] == 2
    assert len(contexts["rows"]) == 2

    omitted = {row["target_kind"]: row for row in bootstrap["target_omissions"]}
    assert ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND in omitted
    assert TENSOR_FACTORIZE_TARGET_KIND in omitted
    assert omitted[ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]["score_claim"] is False

    for experiment in queue["experiments"]:
        metadata = experiment["metadata"]["frontier_rate_attack_bootstrap"]
        assert metadata["schema"] == BOOTSTRAP_SCHEMA
        assert metadata["archive_labels"] == ["frontier"]
        assert metadata["target_kinds"] == [
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        ]
        assert metadata["score_claim"] is False
        assert metadata["ready_for_exact_eval_dispatch"] is False


def test_frontier_bootstrap_cli_writes_valid_queue(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "archive.zip")
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_final_rate_attack_queue.py",
            "--no-current-frontier",
            "--archive",
            f"frontier={archive_path}",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_final_rate_attack_unit",
            "--target-kind",
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
            "--no-optional-target-blockers",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    queue_path = output_dir / "experiment_queue.json"
    bootstrap_path = output_dir / "frontier_rate_attack_bootstrap.json"
    assert queue_path.exists()
    assert bootstrap_path.exists()
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert bootstrap["archive_count"] == 1
    assert bootstrap["archives"][0]["sha256"] == sha256_file(archive_path)

    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_bootstrap_derives_per_archive_section_manifest_target(
    tmp_path: Path,
) -> None:
    decoder_section = brotli.compress(b"decoder payload" * 16, quality=3)
    tail_section = brotli.compress(b"latent payload" * 16, quality=3)
    payload = bytes([0xFF]) + len(decoder_section).to_bytes(3, "little") + decoder_section + tail_section
    archive_path = _write_stored_archive(tmp_path / "archive.zip", member_name="x", payload=payload)
    record = archive_record(
        label="pr106_like",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    derived = derive_archive_section_recode_manifests(
        archive_records=[record],
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
    )

    assert derived["schema"] == DERIVED_SECTION_MANIFEST_BATCH_SCHEMA
    assert derived["ready_manifest_count"] == 1
    row = derived["rows"][0]
    assert row["score_claim"] is False
    assert row["ready_for_materializer_target"] is True
    assert row["selected_section_names"] == [
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_sections",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,),
        include_optional_target_blockers=False,
        section_manifest_by_archive_label={"pr106_like": row["section_manifest_path"]},
        section_names_by_archive_label={"pr106_like": tuple(row["selected_section_names"])},
    )

    bootstrap = payloads["bootstrap"]
    contexts = payloads["contexts"]
    assert bootstrap["executable_target_kinds"] == [ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND]
    assert bootstrap["target_omissions"] == []
    context = contexts["rows"][0]["context"]
    assert context["sweep_archive_specs"] == [f"pr106_like={record['path']}"]
    assert context["section_manifest"] == row["section_manifest_path"]
    assert context["section_names"] == [
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    queue_path = tmp_path / "queue.json"
    _write_json(queue_path, payloads["queue"])
    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_bootstrap_omits_section_target_without_brotli_sections(
    tmp_path: Path,
) -> None:
    payload = bytes([0xFF]) + (7).to_bytes(3, "little") + b"decoder" + b"tail"
    archive_path = _write_stored_archive(tmp_path / "archive.zip", member_name="x", payload=payload)
    record = archive_record(
        label="opaque_pr106",
        archive_path=archive_path,
        repo_root=tmp_path,
        source_kind="unit_test",
    )
    derived = derive_archive_section_recode_manifests(
        archive_records=[record],
        output_dir=tmp_path / "manifests",
        repo_root=tmp_path,
    )

    assert derived["ready_manifest_count"] == 0
    row = derived["rows"][0]
    assert row["ready_for_materializer_target"] is False
    assert "section_manifest_has_no_brotli_decompressible_sections" in row["blockers"]

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_sections_blocked",
        archive_records=[record],
        results_root=tmp_path / "results",
        target_kinds=(
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        ),
        include_optional_target_blockers=False,
        section_manifest_by_archive_label={"opaque_pr106": row["section_manifest_path"]},
        section_names_by_archive_label={"opaque_pr106": tuple(row["selected_section_names"])},
    )

    omissions = payloads["bootstrap"]["target_omissions"]
    assert omissions[0]["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert omissions[0]["archive_label"] == "opaque_pr106"
    assert (
        "archive_section_entropy_recode_requires_brotli_decompressible_section"
        in omissions[0]["blockers"]
    )
    assert payloads["bootstrap"]["executable_target_kinds"] == [
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND
    ]


def test_frontier_bootstrap_refuses_shared_section_manifest_for_multi_archive_sweep(
    tmp_path: Path,
) -> None:
    first = _write_stored_archive(
        tmp_path / "first.zip",
        member_name="x",
        payload=b"\xff\x01\x00\x00a",
    )
    second = _write_stored_archive(
        tmp_path / "second.zip",
        member_name="x",
        payload=b"\xff\x01\x00\x00b",
    )
    records = [
        archive_record(
            label="first",
            archive_path=first,
            repo_root=tmp_path,
            source_kind="unit_test",
        ),
        archive_record(
            label="second",
            archive_path=second,
            repo_root=tmp_path,
            source_kind="unit_test",
        ),
    ]
    manifest = tmp_path / "shared_section_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "unit_test_section_manifest.v1",
            "member": {"name": "x"},
            "sections": [
                {
                    "name": "one_byte",
                    "offset": 4,
                    "length": 1,
                    "sha256": "unused",
                }
            ],
        },
    )

    payloads = build_frontier_rate_attack_payloads(
        repo_root=tmp_path,
        queue_id="frontier_final_rate_attack_shared_section_refusal",
        archive_records=records,
        results_root=tmp_path / "results",
        target_kinds=(
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        ),
        include_optional_target_blockers=False,
        section_manifest=manifest.as_posix(),
        section_names=("one_byte",),
    )

    assert payloads["bootstrap"]["executable_target_kinds"] == [
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND
    ]
    omissions = payloads["bootstrap"]["target_omissions"]
    assert omissions[0]["target_kind"] == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
    assert omissions[0]["archive_labels"] == ["first", "second"]
    assert omissions[0]["score_claim"] is False
    assert omissions[0]["ready_for_exact_eval_dispatch"] is False
    assert (
        "archive_section_entropy_recode_requires_per_archive_section_manifest_for_multi_archive_sweep"
        in omissions[0]["blockers"]
    )


def test_frontier_bootstrap_parse_archive_spec_checks_sha_and_members(
    tmp_path: Path,
) -> None:
    archive_path = _write_archive(tmp_path / "candidate.zip", member_name="masks.mkv")

    record = parse_archive_spec(f"candidate={archive_path}", repo_root=tmp_path)

    assert record["label"] == "candidate"
    assert record["sha256"] == sha256_file(archive_path)
    assert record["zip_member_count"] == 1
    assert record["zip_members"][0]["name"] == "masks.mkv"
    _assert_false_authority(record)


def test_resolve_current_frontier_archive_from_auth_request(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path / "experiments" / "results" / "candidate" / "submission_dir" / "archive.zip"
    )
    digest = sha256_file(archive)
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "score": 0.123,
                "evidence_grade": "[contest-CPU]",
                "hardware_substrate": "linux_x86_64_cpu",
                "measured_at_utc": "2026-05-25T00:00:00Z",
                "extra": {"archive_bytes": archive.stat().st_size},
            }
        },
    )
    request_path = (
        tmp_path
        / "experiments"
        / "results"
        / "modal_auth_eval_cpu"
        / "job"
        / "modal_cpu_auth_eval_local_request.json"
    )
    _write_json(
        request_path,
        {
            "archive_path": archive.as_posix(),
            "archive_sha256": digest,
            "archive_size_bytes": archive.stat().st_size,
        },
    )

    resolution = resolve_current_frontier_archive(
        repo_root=tmp_path,
        pointer_path=pointer_path,
        frontier_axis="contest_cpu",
    )

    assert resolution["archive_sha256"] == digest
    assert resolution["archive_record"]["path"] == (
        "experiments/results/candidate/submission_dir/archive.zip"
    )
    assert resolution["match"]["request_path"] == (
        "experiments/results/modal_auth_eval_cpu/job/modal_cpu_auth_eval_local_request.json"
    )


def test_resolve_current_frontier_archive_fails_closed_on_duplicate_matches(
    tmp_path: Path,
) -> None:
    first = _write_archive(
        tmp_path / "experiments" / "results" / "candidate_a" / "submission_dir" / "archive.zip"
    )
    second = _write_archive(
        tmp_path / "experiments" / "results" / "candidate_b" / "submission_dir" / "archive.zip"
    )
    digest = sha256_file(first)
    assert sha256_file(second) == digest
    pointer_path = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    _write_json(
        pointer_path,
        {
            "our_local_frontier_contest_cpu": {
                "archive_sha256": digest,
                "extra": {"archive_bytes": first.stat().st_size},
            }
        },
    )
    for name, archive in (("a", first), ("b", second)):
        _write_json(
            tmp_path
            / "experiments"
            / "results"
            / "modal_auth_eval_cpu"
            / name
            / "request.json",
            {
                "archive_path": archive.as_posix(),
                "archive_sha256": digest,
                "archive_size_bytes": archive.stat().st_size,
            },
        )

    with pytest.raises(FrontierRateAttackBootstrapError, match="ambiguous"):
        resolve_current_frontier_archive(
            repo_root=tmp_path,
            pointer_path=pointer_path,
            frontier_axis="contest_cpu",
        )
