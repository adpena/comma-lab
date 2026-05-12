from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.packet_compiler import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    parse_pr106_sidecar_packet,
    pr106_sidecar_manifest,
    read_single_stored_member_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_R2_ARCHIVE = REPO_ROOT / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_PR101_ARCHIVE = (
    REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    ("archive_path", "format_id", "sidecar_kind", "expected_archive_sha"),
    [
        (
            PR106_R2_ARCHIVE,
            PR106_SIDECAR_FORMAT_BROTLI,
            "brotli_dim_delta",
            "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f",
        ),
        (
            PR106_R2_PR101_ARCHIVE,
            PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
            "pr101_ranked_no_op",
            "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383",
        ),
    ],
)
def test_pr106_sidecar_packet_ir_identity_on_release_archives(
    archive_path: Path,
    format_id: int,
    sidecar_kind: str,
    expected_archive_sha: str,
) -> None:
    archive_bytes = archive_path.read_bytes()
    assert _sha(archive_bytes) == expected_archive_sha

    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    emitted_payload = emit_pr106_sidecar_packet(packet)
    emitted_archive = emit_single_stored_member_archive(member)
    manifest = pr106_sidecar_manifest(packet, archive_sha256=_sha(archive_bytes))

    assert packet.format_id == format_id
    assert packet.sidecar_kind == sidecar_kind
    assert emitted_payload == member.payload
    assert emitted_archive == archive_bytes
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_sha256"] == expected_archive_sha
    assert manifest["emitted_payload_sha256"] == _sha(member.payload)


def test_pr106_sidecar_packet_rejects_unknown_format_id() -> None:
    archive_bytes = PR106_R2_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    mutated = bytearray(member.payload)
    mutated[1] = 0x99

    with pytest.raises(ValueError, match="unsupported PR106 sidecar format_id"):
        parse_pr106_sidecar_packet(bytes(mutated))


def test_pr106_sidecar_packet_rejects_pr101_missing_framing_meta() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)

    with pytest.raises(ValueError, match="requires framing_meta"):
        emit_pr106_sidecar_packet(
            type(packet)(
                format_id=packet.format_id,
                pr106_bytes=packet.pr106_bytes,
                sidecar_payload=packet.sidecar_payload,
                framing_meta=None,
            )
        )
