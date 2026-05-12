from __future__ import annotations

from pathlib import Path

import pytest

from tac.packet_compiler import (
    dumps_runtime_consumption_manifest,
    emit_pr106_sidecar_packet,
    load_pr106_sidecar_runtime,
    parse_pr106_sidecar_packet,
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
    assert manifest["format_id"] == format_id
    assert manifest["payload_sha256_changed"] is True
    assert manifest["inner_pr106_payload_sha256_unchanged"] is True
    assert manifest["sidecar_payload_sha256_changed"] is True
    assert manifest["runtime_semantic_digest_changed"] is True
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
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
    assert "score_claim" in dumps_runtime_consumption_manifest(manifest)


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
