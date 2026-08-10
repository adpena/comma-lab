from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from experiments import ddm_hp3_hpac_section_and_zip_frame as hp3
from tac.payload_retention_gate import check_no_measure_and_discard_payload
from tac.pr130_runtime.ddm_hp3_runtime.hp3_codec import (
    FRAME_COUNT,
    factor_frame_embedding,
    pack_monolithic_checkpoint,
    pack_token_chunks,
    restore_ihs1,
    unpack_monolithic_checkpoint,
    unpack_token_chunks,
)


def test_hp31_frame_delta_round_trip_is_byte_exact() -> None:
    frame_offset = 32
    body = bytearray(b"IHS1" + bytes(range(28)))
    body.extend((index * 17) % 256 for index in range(FRAME_COUNT * 8))
    body.extend(b"tail")
    source = bytes(body)
    factored = factor_frame_embedding(source, frame_offset)
    assert factored.startswith(b"HP31")
    assert restore_ihs1(factored) == source


def test_hpt1_round_trip_and_exact_consumption() -> None:
    chunk_frames = 120
    count = (FRAME_COUNT + chunk_frames - 1) // chunk_frames
    chunks = tuple((index.to_bytes(4, "little") or b"\0\0\0\0") for index in range(count))
    envelope = pack_token_chunks(chunks, chunk_frames=chunk_frames)
    parsed = unpack_token_chunks(envelope)
    assert parsed.frame_count == FRAME_COUNT
    assert parsed.chunk_frames == chunk_frames
    assert parsed.chunks == chunks
    with pytest.raises(ValueError, match="trailing"):
        unpack_token_chunks(envelope + b"junk")


def test_hpm1_round_trip_preserves_seek_state_and_range_payload() -> None:
    range_payload = bytes(range(16))
    payload = pack_monolithic_checkpoint(range_payload, position=2, state=(17, 23))
    parsed = unpack_monolithic_checkpoint(payload)
    assert parsed.position == 2
    assert parsed.state == (17, 23)
    assert parsed.range_payload == range_payload


def test_zip_breakdown_proves_100_byte_stored_floor(tmp_path: Path) -> None:
    payload = b"real member bytes" * 100
    archive = tmp_path / "archive.zip"
    hp3.write_zip(archive, payload, compression=zipfile.ZIP_STORED)
    breakdown = hp3.zip_breakdown(archive.read_bytes())
    assert archive.stat().st_size - len(payload) == 100
    assert breakdown == {
        "local_header": 30,
        "local_filename": 1,
        "local_extra": 0,
        "member_data": len(payload),
        "central_header": 46,
        "central_filename": 1,
        "central_extra": 0,
        "central_comment": 0,
        "eocd": 22,
        "zip_comment": 0,
    }


def test_real_hpac_candidate_models_have_exact_control(tmp_path: Path) -> None:
    packer, _ = hp3.configure_sources()
    candidates, decomposition = hp3.build_candidate_models(tmp_path, packer)
    by_name = {candidate.name: candidate for candidate in candidates}
    control = by_name["control_ihs1"]
    assert len(control.raw) == hp3.HPAC_RAW_BYTES
    assert hp3.sha256_bytes(control.raw) == hp3.HPAC_RAW_SHA256
    assert restore_ihs1(by_name["factor_frame_delta"].raw) == control.raw
    assert by_name["requant_frame_embed_step2"].changed_values > 0
    assert by_name["prune_weight_abs1"].changed_values > 0
    assert decomposition["magic_bytes"] == 4
    assert decomposition["frame_embed_bytes"] == 4_800
    assert decomposition["raw_hpac_bytes"] == hp3.HPAC_RAW_BYTES


def test_real_control_archive_rebuild_is_byte_identical(tmp_path: Path) -> None:
    packer, _ = hp3.configure_sources()
    candidates, _ = hp3.build_candidate_models(tmp_path, packer)
    control = next(candidate for candidate in candidates if candidate.name == "control_ihs1")
    _, semantic_pose, _, _ = hp3.split_base()
    target = tmp_path / "retained/candidates/control_ihs1/tokens.range"
    hp3.retain_payload(target, hp3.CANONICAL_RANGE.read_bytes())
    result = hp3.build_archive_candidate(tmp_path, control, semantic_pose, hp3.CANONICAL_RANGE.read_bytes())
    assert result["archive"]["bytes"] == hp3.BASE_BYTES
    assert result["archive"]["sha256"] == hp3.BASE_SHA256
    assert result["delta_bytes_vs_exact_base"] == 0


def test_new_python_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_hp3_hpac_section_and_zip_frame.py",
            "src/tac/pr130_runtime/ddm_hp3_runtime",
            "src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py",
        ),
    )
    assert findings == []
