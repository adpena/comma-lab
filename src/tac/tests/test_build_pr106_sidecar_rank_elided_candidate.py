# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_PR101_RANKED_SCHEMA,
    PR106SidecarPacket,
    StoredZipMember,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    encode_pr101_ranked_sidecar_payload,
    sha256_hex,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_pr106_sidecar_rank_elided_candidate.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_pr106_sidecar_rank_elided_candidate",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_arrays() -> tuple[np.ndarray, np.ndarray]:
    dims = np.full(
        PR106_PR101_RANKED_SCHEMA.n_pairs,
        PR106_PR101_RANKED_SCHEMA.no_op_sentinel,
        dtype=np.uint8,
    )
    deltas = np.zeros(PR106_PR101_RANKED_SCHEMA.n_pairs, dtype=np.int8)
    vocab = np.asarray(PR106_PR101_RANKED_SCHEMA.deltas, dtype=np.int8)
    for out_index, pair_index in enumerate(range(0, PR106_PR101_RANKED_SCHEMA.n_pairs, 3)):
        dims[pair_index] = (out_index * 5) % PR106_PR101_RANKED_SCHEMA.n_dims
        deltas[pair_index] = vocab[out_index % len(vocab)]
    return dims, deltas


def _fixture_archive(path: Path, *, member_name: str = "x") -> bytes:
    dims, deltas = _fixture_arrays()
    sidecar_payload, framing_meta = encode_pr101_ranked_sidecar_payload(dims, deltas)
    packet = PR106SidecarPacket(
        format_id=PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
        pr106_bytes=b"fixture-pr106-payload" * 3,
        sidecar_payload=sidecar_payload,
        framing_meta=framing_meta,
    )
    member = StoredZipMember(
        name=member_name,
        payload=emit_pr106_sidecar_packet(packet),
        date_time=(1980, 1, 1, 0, 0, 0),
        external_attr=0o644 << 16,
        create_system=3,
        flag_bits=0,
        comment=b"",
        extra=b"",
    )
    archive_bytes = emit_single_stored_member_archive(member)
    path.write_bytes(archive_bytes)
    return archive_bytes


def test_rank_elided_sidecar_roundtrips_fixture_arrays() -> None:
    tool = _load_tool()
    dims, deltas = _fixture_arrays()

    candidate = tool.build_rank_elided_sidecar(dims, deltas)
    decoded_dims, decoded_deltas = tool.decode_rank_elided_sidecar_payload(
        candidate.rank_elided_payload,
        candidate.rank_elided_meta,
    )

    assert candidate.elided_length_rank_blob == b"\x00"
    assert candidate.source_rank_bytes == 1
    assert candidate.rank_elided_charged_bytes == candidate.source_charged_bytes - 2
    assert np.array_equal(decoded_dims, dims)
    assert np.array_equal(decoded_deltas, deltas)


def test_rank_elided_candidate_manifest_is_no_score_and_non_dispatchable(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    source_archive = tmp_path / "source.zip"
    source_archive_bytes = _fixture_archive(source_archive)

    report = tool.build_rank_elided_candidate_report(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        source_label="fixture_pr106_r2_pr101",
        expected_source_sha256=sha256_hex(source_archive_bytes),
    )

    candidate_archive = Path(report["candidate_archive"]["path"])
    manifest = Path(report["manifest_path"])
    markdown = Path(report["markdown_path"])
    proof = report["packet_ir_consumed_byte_proof"]

    assert candidate_archive.exists()
    assert manifest.exists()
    assert markdown.exists()
    assert report["research_only"] is True
    assert report["score_claim"] is False
    assert report["contest_axis_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["source_archive_sha256"] == sha256_hex(source_archive_bytes)
    assert report["source_archive_bytes"] == len(source_archive_bytes)
    assert report["candidate_archive_sha256"] == report["candidate_archive"]["sha256"]
    assert report["candidate_archive_bytes"] == report["candidate_archive"]["bytes"]
    assert report["candidate_archive_byte_delta"] == -4
    assert report["candidate_diff_audit"] == {"blockers": [], "total_byte_delta": -4}
    assert report["byte_closed_archive_emitted"] is True
    assert report["byte_closed_for_existing_runtime"] is False
    assert report["packet_ir_decoder_implemented"] is True
    assert report["existing_scored_runtime_decoder_implemented"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["exact_next_dispatch_command"] is None
    assert "runtime_decoder_missing_for_format_0x04" in report["dispatch_blockers"]
    assert (
        "existing_scored_runtime_does_not_consume_format_0x04"
        in report["dispatch_blockers"]
    )
    assert report["source_packet"]["source_pr101_canonical_payload_matches_packet"] is True
    assert report["candidate_packet"]["format_id"] == "0x04"
    assert report["candidate_packet"]["packet_ir_decoder_implemented"] is True
    assert report["candidate_packet"]["runtime_decoder_implemented"] is False
    assert report["candidate_packet"]["runtime_decoder_scope"] == (
        "packet_ir_local_parser_only_not_existing_scored_runtime"
    )
    assert report["candidate_archive"]["byte_delta_vs_source_archive"] == -4
    assert report["candidate_member"]["payload_byte_delta_vs_source_member"] == -4
    assert report["candidate_packet"]["charged_sidecar_byte_delta_vs_source_packet"] == -2
    assert report["rank_elision"]["elided_length_rank_blob_bytes"] == 1
    assert report["rank_elision"]["elided_rank_bytes_meta_field_bytes"] == 1
    assert report["rank_elision"]["elided_sidecar_len_prefix_bytes"] == 2
    assert report["rank_elision"]["total_packet_payload_savings_bytes"] == 4
    assert report["semantic_equivalence"]["rank_elided_decodes_to_source_dim_delta"] is True
    assert proof["all_payload_bytes_accounted"] is True
    assert proof["runtime_consumption_claim"] is False
    assert proof["parsed_reemit_identity"] is True


def test_rank_elided_packet_parser_rejects_non_candidate_format() -> None:
    tool = _load_tool()

    try:
        tool.parse_rank_elided_packet(b"\xfe\x02" + b"\x00" * 12)
    except ValueError as exc:
        assert "format mismatch" in str(exc)
    else:  # pragma: no cover - defensive failure clarity
        raise AssertionError("format-0x04 parser accepted a non-candidate format")
