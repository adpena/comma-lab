from __future__ import annotations

from pathlib import Path

import pytest

from tac.packet_compiler import (
    dumps_runtime_consumption_manifest,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    load_pr106_sidecar_runtime,
    mutate_pr106_sidecar_semantic_correction,
    parse_pr106_sidecar_packet,
    prove_pr106_same_runtime_full_frame_parity,
    prove_pr106_sidecar_runtime_decode_consumption,
    read_single_stored_member_archive,
    runtime_sidecar_correction_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_R2_ARCHIVE = REPO_ROOT / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_RUNTIME = REPO_ROOT / "submissions/pr106_latent_sidecar_r2"
PR106_R2_PR101_ARCHIVE = (
    REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)
PR106_R2_PR101_RUNTIME = REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar"


@pytest.mark.parametrize(
    ("archive_path", "runtime_dir", "format_id"),
    [
        (PR106_R2_ARCHIVE, PR106_R2_RUNTIME, "0x01"),
        (PR106_R2_PR101_ARCHIVE, PR106_R2_PR101_RUNTIME, "0x02"),
    ],
)
def test_pr106_sidecar_runtime_decode_consumption_proof_is_nonpromotable(
    archive_path: Path,
    runtime_dir: Path,
    format_id: str,
) -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive_path,
        runtime_dir=runtime_dir,
    )

    assert manifest["schema"] == "pr106_sidecar_runtime_decode_consumption_proof_v1"
    assert manifest["archive_member_name"] == "0.bin"
    assert manifest["format_id"] == format_id
    assert manifest["payload_sha256_changed"] is True
    assert manifest["inner_pr106_payload_sha256_unchanged"] is True
    assert manifest["sidecar_payload_sha256_changed"] is True
    assert manifest["runtime_semantic_digest_changed"] is True
    assert manifest["runtime_corrected_latents_digest_changed"] is True
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False

    source_digest = manifest["source_runtime_correction_digest"]
    mutated_digest = manifest["mutated_runtime_correction_digest"]
    assert isinstance(source_digest, dict)
    assert isinstance(mutated_digest, dict)
    assert source_digest["format_id"] == format_id
    assert mutated_digest["format_id"] == format_id
    assert source_digest["n_pairs"] == 600
    assert mutated_digest["n_pairs"] == 600
    assert source_digest["combined_sha256"] != mutated_digest["combined_sha256"]
    assert source_digest["source_latents_sha256"] == mutated_digest["source_latents_sha256"]
    assert source_digest["corrected_latents_sha256"] != mutated_digest[
        "corrected_latents_sha256"
    ]
    assert source_digest["latents_changed_by_sidecar"] is True
    assert mutated_digest["latents_changed_by_sidecar"] is True
    assert "score_claim" in dumps_runtime_consumption_manifest(manifest)


def test_pr106_runtime_decode_consumption_autodetects_x_member(
    tmp_path: Path,
) -> None:
    source_member = read_single_stored_member_archive(PR106_R2_ARCHIVE.read_bytes())
    archive = tmp_path / "x_member.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(source_member)(
            name="x",
            payload=source_member.payload,
            date_time=source_member.date_time,
            external_attr=source_member.external_attr,
            create_system=source_member.create_system,
            flag_bits=source_member.flag_bits,
            comment=source_member.comment,
            extra=source_member.extra,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_RUNTIME,
    )

    assert manifest["archive_member_name"] == "x"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["score_claim"] is False


def test_pr106_pr101_grammar_runtime_consumes_framing_meta_fail_closed() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    assert packet.framing_meta is not None
    mutated_meta = bytearray(packet.framing_meta)
    mutated_meta[0] ^= 0x01
    mutated_payload = emit_pr106_sidecar_packet(
        type(packet)(
            format_id=packet.format_id,
            pr106_bytes=packet.pr106_bytes,
            sidecar_payload=packet.sidecar_payload,
            framing_meta=bytes(mutated_meta),
        )
    )
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    baseline_digest = runtime_sidecar_correction_digest(runtime, member.payload)

    try:
        mutated_digest = runtime_sidecar_correction_digest(runtime, mutated_payload)
    except (ValueError, IndexError):
        return
    assert mutated_digest["combined_sha256"] != baseline_digest["combined_sha256"]


def test_pr106_pr101_runtime_accepts_legacy_brotli_format_for_same_runtime_pairing() -> None:
    """The 0x02 runtime can decode 0x01 payloads; compare only in same runtime."""
    source_member = read_single_stored_member_archive(PR106_R2_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(source_member.payload)
    legacy_payload = emit_pr106_sidecar_packet(source_packet)
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    digest = runtime_sidecar_correction_digest(runtime, legacy_payload)
    grammar_digest = runtime_sidecar_correction_digest(runtime, grammar_member.payload)

    assert digest["format_id"] == "0x01"
    assert grammar_digest["format_id"] == "0x02"
    assert digest["n_pairs"] == 600
    assert digest["combined_sha256"] == grammar_digest["combined_sha256"]
    assert digest["corrected_latents_sha256"] == grammar_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_same_runtime_streaming_prefix_parity_is_nonpromotable() -> None:
    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=PR106_R2_ARCHIVE,
        candidate_archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert manifest["schema"] == "pr106_same_runtime_streaming_frame_parity_v1"
    assert manifest["proof_scope"] == "same_runtime_streaming_prefix_hash"
    assert manifest["streaming_output_sha256_equal"] is True
    assert manifest["streaming_output_total_bytes_equal"] is True
    assert manifest["prefix_parity_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    source = manifest["source"]
    candidate = manifest["candidate"]
    assert isinstance(source, dict)
    assert isinstance(candidate, dict)
    assert source["full_frame_digest"] is False
    assert candidate["full_frame_digest"] is False
    assert source["total_frames"] == candidate["total_frames"] == 2
    assert source["streaming_raw_sha256"] == candidate["streaming_raw_sha256"]


def test_pr106_same_runtime_streaming_prefix_autodetects_x_members(
    tmp_path: Path,
) -> None:
    source_member = read_single_stored_member_archive(PR106_R2_ARCHIVE.read_bytes())
    candidate_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    source_archive = tmp_path / "source_x.zip"
    candidate_archive = tmp_path / "candidate_x.zip"
    source_archive.write_bytes(
        emit_single_stored_member_archive(type(source_member)(
            name="x",
            payload=source_member.payload,
            date_time=source_member.date_time,
            external_attr=source_member.external_attr,
            create_system=source_member.create_system,
            flag_bits=source_member.flag_bits,
            comment=source_member.comment,
            extra=source_member.extra,
        ))
    )
    candidate_archive.write_bytes(
        emit_single_stored_member_archive(type(candidate_member)(
            name="x",
            payload=candidate_member.payload,
            date_time=candidate_member.date_time,
            external_attr=candidate_member.external_attr,
            create_system=candidate_member.create_system,
            flag_bits=candidate_member.flag_bits,
            comment=candidate_member.comment,
            extra=candidate_member.extra,
        ))
    )

    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=source_archive,
        candidate_archive_path=candidate_archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert manifest["source_archive"]["member_name"] == "x"
    assert manifest["candidate_archive"]["member_name"] == "x"
    assert manifest["prefix_parity_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is False


def test_pr106_same_runtime_streaming_prefix_detects_semantic_sidecar_change(
    tmp_path: Path,
) -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    mutated_packet, _ = mutate_pr106_sidecar_semantic_correction(packet, pair_index=0)
    mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
    mutated_archive = tmp_path / "mutated.zip"
    mutated_archive.write_bytes(
        emit_single_stored_member_archive(type(member)(
            name=member.name,
            payload=mutated_payload,
            date_time=member.date_time,
            external_attr=member.external_attr,
            create_system=member.create_system,
            flag_bits=member.flag_bits,
            comment=member.comment,
            extra=member.extra,
        ))
    )
    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=PR106_R2_PR101_ARCHIVE,
        candidate_archive_path=mutated_archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert manifest["streaming_output_sha256_equal"] is False
    assert manifest["streaming_output_total_bytes_equal"] is True
    assert manifest["prefix_parity_claim"] is False
    assert manifest["full_frame_inflate_output_parity_claim"] is False
