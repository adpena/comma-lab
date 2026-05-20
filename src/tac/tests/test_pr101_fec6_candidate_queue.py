# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from tac.packet_compiler.pr101_fec6_candidate_queue import (
    PR101_FEC6_BYTE_ACCOUNTING_SCHEMA,
    PR101_FEC6_CANDIDATE_QUEUE_SCHEMA,
    build_pr101_fec6_packetir_candidate_queue,
    render_pr101_fec6_packetir_candidate_queue_markdown,
)
from tac.packet_compiler.pr101_fec6_packetir import FEC6_FIXED_K16_CODE_BITS
from tac.repo_io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_pr101_fec6_packetir_candidate_queue.py"
FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir/archive.zip"
)
FEC6_ARCHIVE_SHA256 = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_pr101_fec6_packetir_candidate_queue",
        TOOL_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fec6_selector(codes: list[int]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    payload = bytes(
        int((bits + ("0" * ((-len(bits)) % 8)))[index : index + 8], 2)
        for index in range(0, len(bits + ("0" * ((-len(bits)) % 8))), 8)
    )
    return b"FEC6" + len(codes).to_bytes(2, "little") + payload


def _fp11_member(source: bytes = b"source-pr101") -> bytes:
    selector = _fec6_selector([0, 2, 7, 13])
    return (
        b"FP11"
        + len(source).to_bytes(4, "little")
        + source
        + len(selector).to_bytes(2, "little")
        + selector
    )


def _single_member_archive(tmp_path: Path, payload: bytes, *, name: str = "x") -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    return archive


def _operator_manifest(path: Path, *, codes: list[int]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "tac_fec6_selector_operator_space_v1",
                "source_archive": {"codes": codes},
                "top_bit_saving_rows": [
                    {
                        "operator_id": "pair_1_to_0",
                        "pair": 1,
                        "current_code": 2,
                        "candidate_code": 0,
                        "selector_code_bit_delta": -1,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_candidate_queue_accounts_bytes_and_stays_nonpromotional(
    tmp_path: Path,
) -> None:
    """Per codex adversarial review 2026-05-19 F1: queue without runtime
    consumption proof MUST surface ``runtime_byte_consumption_noop_detector_missing``
    at top-level blockers AND keep ``runtime_consumption_proven=False``.
    Pre-fix the v1 contract conflated parser_byte_accounting_passed with
    runtime_consumed_byte_accounting_passed.
    """

    archive = _single_member_archive(tmp_path, _fp11_member())
    manifest = _operator_manifest(tmp_path / "operator.json", codes=[0, 2, 7, 13])

    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=archive,
        operator_space_manifest_path=manifest,
        expected_archive_sha256=sha256_file(archive),
    )

    assert queue["schema"] == PR101_FEC6_CANDIDATE_QUEUE_SCHEMA
    # F1 v2 contract: top-level blockers MUST surface missing runtime proof.
    assert "runtime_byte_consumption_noop_detector_missing" in queue["blockers"]
    assert queue["score_claim"] is False
    assert queue["promotion_eligible"] is False
    assert queue["ready_for_exact_eval_dispatch"] is False
    assert queue["byte_accounting"]["schema"] == PR101_FEC6_BYTE_ACCOUNTING_SCHEMA
    # Parser-domain truth still holds — primary section lengths sum to member bytes.
    assert queue["byte_accounting"]["all_payload_bytes_accounted"] is True
    assert queue["byte_accounting"]["parser_byte_accounting_passed"] is True
    # Runtime-authority defaults False because no Catalog #105 no-op proof attached.
    assert queue["byte_accounting"]["runtime_consumption_proven"] is False
    assert (
        queue["byte_accounting"]["runtime_consumed_byte_accounting_passed"] is False
    )
    assert queue["candidate_count"] == len(queue["candidates"])
    assert queue["operator_candidate_count"] == 1
    candidate_ids = {candidate["candidate_id"] for candidate in queue["candidates"]}
    assert "pr101_sidecar_only_runtime_probe" in candidate_ids
    assert "pr101_latent_plus_sidecar_runtime_adapter_probe" in candidate_ids
    assert all(candidate["score_claim"] is False for candidate in queue["candidates"])
    assert all(candidate["consumer_surfaces"] for candidate in queue["candidates"])


def test_candidate_queue_flips_runtime_proven_when_proof_supplied(
    tmp_path: Path,
) -> None:
    """Supplying a valid Catalog #105 no-op proof flips runtime_consumption_proven=True
    AND removes the missing-noop-detector blocker from top-level + byte_accounting.
    """

    member_payload = _fp11_member()
    source_len = len(b"\x02source-pr101")
    archive = _single_member_archive(tmp_path, member_payload)
    proof_path = tmp_path / "noop_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema_version": "deterministic_no_op_proof.v1",
                "mode": "identity",
                "no_op_detector_passed": True,
                "runtime_bytes_consumed": len(member_payload),
                "runtime_consumption_proof_source": "inflate_smoke",
                "consumed_section_names": [
                    "source_pr101_payload",
                    "selector_fec6_payload",
                ],
                "consumed_byte_ranges": [
                    {
                        "section_name": "source_pr101_payload",
                        "range": [8, 8 + source_len],
                    },
                    {
                        "section_name": "selector_fec6_payload",
                        "range": [8 + source_len + 2, len(member_payload)],
                    },
                ],
            }
        )
    )

    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=archive,
        expected_archive_sha256=sha256_file(archive),
        runtime_consumption_proof_path=proof_path,
    )

    assert queue["byte_accounting"]["runtime_consumption_proven"] is True
    assert queue["byte_accounting"]["runtime_consumed_section_names"] == [
        "selector_fec6_payload",
        "source_pr101_payload",
    ]
    assert (
        queue["byte_accounting"]["runtime_consumed_byte_accounting_passed"] is True
    )
    assert (
        "runtime_byte_consumption_noop_detector_missing" not in queue["blockers"]
    )
    # Score authority defaults False — runtime proof only flips byte-consumption
    # authority, not promotion eligibility (paired contest exact eval still required).
    assert queue["score_claim"] is False
    assert queue["ready_for_exact_eval_dispatch"] is False


def test_candidate_queue_rejects_proof_with_byte_count_mismatch(
    tmp_path: Path,
) -> None:
    """Runtime-proof bytes_consumed mismatch keeps runtime_consumption_proven=False."""

    member_payload = _fp11_member()
    archive = _single_member_archive(tmp_path, member_payload)
    proof_path = tmp_path / "noop_proof_bad.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema_version": "deterministic_no_op_proof.v1",
                "mode": "identity",
                "no_op_detector_passed": True,
                "runtime_bytes_consumed": len(member_payload) - 1,
            }
        )
    )

    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=archive,
        runtime_consumption_proof_path=proof_path,
    )

    assert queue["byte_accounting"]["runtime_consumption_proven"] is False
    assert (
        "runtime_consumption_proof_bytes_consumed_mismatch" in queue["blockers"]
    )


def test_candidate_queue_fails_closed_on_operator_source_mismatch(
    tmp_path: Path,
) -> None:
    archive = _single_member_archive(tmp_path, _fp11_member())
    manifest = _operator_manifest(tmp_path / "operator.json", codes=[0, 0, 7, 13])

    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=archive,
        operator_space_manifest_path=manifest,
        expected_archive_sha256=sha256_file(archive),
    )

    assert "operator_manifest_source_codes_mismatch" in queue["blockers"]
    operator_candidate = queue["candidates"][-1]
    assert operator_candidate["current_code_matches_parsed_selector"] is True
    assert queue["score_claim"] is False


def test_candidate_queue_markdown_names_consumers(tmp_path: Path) -> None:
    archive = _single_member_archive(tmp_path, _fp11_member())
    queue = build_pr101_fec6_packetir_candidate_queue(archive_path=archive)

    markdown = render_pr101_fec6_packetir_candidate_queue_markdown(queue)

    assert "PR101/FEC6 PacketIR Candidate Queue" in markdown
    assert "tac.cathedral_consumers.packetir_candidate_queue_consumer" in markdown
    assert "fec6_selector_entropy_recode_probe" in markdown


def test_candidate_queue_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """Without --runtime-consumption-proof the CLI exits rc=3 (proof-missing,
    queue itself well-formed) per codex adversarial review 2026-05-19 F1."""

    archive = _single_member_archive(tmp_path, _fp11_member())
    manifest = _operator_manifest(tmp_path / "operator.json", codes=[0, 2, 7, 13])
    out_json = tmp_path / "queue.json"
    out_md = tmp_path / "queue.md"
    tool = _load_tool()

    rc = tool.main(
        [
            "--archive",
            str(archive),
            "--operator-space-manifest",
            str(manifest),
            "--expected-archive-sha256",
            sha256_file(archive),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 3  # runtime-proof missing is its own exit code
    queue = json.loads(out_json.read_text(encoding="utf-8"))
    assert queue["schema"] == PR101_FEC6_CANDIDATE_QUEUE_SCHEMA
    assert (
        "runtime_byte_consumption_noop_detector_missing" in queue["blockers"]
    )
    assert out_md.read_text(encoding="utf-8").startswith(
        "# PR101/FEC6 PacketIR Candidate Queue"
    )


def test_candidate_queue_cli_returns_rc0_with_runtime_proof(tmp_path: Path) -> None:
    """With --runtime-consumption-proof supplied + valid + matching operator
    manifest the CLI exits rc=0."""

    member_payload = _fp11_member()
    archive = _single_member_archive(tmp_path, member_payload)
    # Synthetic operator manifest with matching codes so no operator-mismatch
    # blocker fires; default-path manifest belongs to the real PR101 archive.
    manifest = _operator_manifest(tmp_path / "operator.json", codes=[0, 2, 7, 13])
    out_json = tmp_path / "queue.json"
    out_md = tmp_path / "queue.md"
    proof_path = tmp_path / "noop_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema_version": "deterministic_no_op_proof.v1",
                "mode": "identity",
                "no_op_detector_passed": True,
                "runtime_bytes_consumed": len(member_payload),
            }
        )
    )
    tool = _load_tool()
    rc = tool.main(
        [
            "--archive",
            str(archive),
            "--operator-space-manifest",
            str(manifest),
            "--expected-archive-sha256",
            sha256_file(archive),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--runtime-consumption-proof",
            str(proof_path),
        ]
    )

    assert rc == 0
    queue = json.loads(out_json.read_text(encoding="utf-8"))
    assert queue["byte_accounting"]["runtime_consumption_proven"] is True
    assert queue["blockers"] == []


def test_real_pr101_fec6_candidate_queue_if_present() -> None:
    if not FEC6_ARCHIVE.exists():
        return

    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=FEC6_ARCHIVE,
        expected_archive_sha256=FEC6_ARCHIVE_SHA256,
    )

    assert queue["archive_sha256"] == FEC6_ARCHIVE_SHA256
    assert queue["member_payload_bytes"] == 178_417
    assert queue["selector"]["n_pairs"] == 600
    assert queue["selector"]["selector_index_bytes"] == 243
    assert queue["byte_accounting"]["all_payload_bytes_accounted"] is True
