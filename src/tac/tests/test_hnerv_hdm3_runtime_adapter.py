from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli
import pytest

from tac.hnerv_decoder_recode import PACKED_STATE_SCHEMA, encode_hdm3_q_brotli_split_fixture
from tac.hnerv_hdm3_runtime_adapter import (
    HnervHdm3RuntimeAdapterError,
    restore_hdm3_file_to_legacy_brotli,
    restore_hdm3_payload_to_legacy_brotli,
)
from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    sha256_bytes,
    write_stored_single_member_zip,
)

REPO = Path(__file__).resolve().parents[3]


def test_restore_hdm3_payload_to_legacy_brotli_preserves_raw_decoder_and_latents() -> None:
    raw = _synthetic_decoder_raw()
    hdm3, _stats = encode_hdm3_q_brotli_split_fixture(
        _parsed_decoder_from_legacy_brotli(brotli.compress(raw, quality=0))
    )
    latents = brotli.compress(b"latent-sidecar" * 200, quality=5)
    payload = PackedHnervPayload(
        header=b"\xff\x00\x00\x00",
        decoder_packed_brotli=hdm3,
        latents_and_sidecar_brotli=latents,
    ).to_bytes()

    restored, proof = restore_hdm3_payload_to_legacy_brotli(payload, require_hdm3=True)

    restored_packed = parse_ff_packed_brotli_hnerv(restored)
    assert restored != payload
    assert not restored_packed.decoder_packed_brotli.startswith(b"HDM3")
    assert brotli.decompress(restored_packed.decoder_packed_brotli) == raw
    assert restored_packed.latents_and_sidecar_brotli == latents
    assert proof["mode"] == "hdm3_restored_to_legacy_brotli"
    assert proof["decoder_raw_equal"] is True
    assert proof["latents_and_sidecar_preserved"] is True
    assert proof["ready_for_public_runtime_inflate"] is True
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_restore_legacy_payload_passthrough_is_explicit() -> None:
    raw = _synthetic_decoder_raw()
    legacy = brotli.compress(raw, quality=3)
    payload = PackedHnervPayload(
        header=b"\xff\x00\x00\x00",
        decoder_packed_brotli=legacy,
        latents_and_sidecar_brotli=b"latents",
    ).to_bytes()

    restored, proof = restore_hdm3_payload_to_legacy_brotli(payload)

    assert restored == payload
    assert proof["mode"] == "legacy_brotli_passthrough"
    assert proof["payload_changed"] is False
    assert proof["decoder_raw_sha256"] == sha256_bytes(raw)


def test_restore_unknown_decoder_fails_closed() -> None:
    payload = PackedHnervPayload(
        header=b"\xff\x00\x00\x00",
        decoder_packed_brotli=b"NOPE-not-brotli",
        latents_and_sidecar_brotli=b"latents",
    ).to_bytes()

    with pytest.raises(HnervHdm3RuntimeAdapterError, match="neither HDM3 nor legacy Brotli"):
        restore_hdm3_payload_to_legacy_brotli(payload)


def test_restore_file_cli_writes_payload_and_proof(tmp_path: Path) -> None:
    raw = _synthetic_decoder_raw()
    hdm3, _stats = encode_hdm3_q_brotli_split_fixture(
        _parsed_decoder_from_legacy_brotli(brotli.compress(raw, quality=0))
    )
    source = tmp_path / "x"
    output = tmp_path / "x.legacy"
    proof_path = tmp_path / "proof.json"
    source.write_bytes(
        PackedHnervPayload(
            header=b"\xff\x00\x00\x00",
            decoder_packed_brotli=hdm3,
            latents_and_sidecar_brotli=b"latents",
        ).to_bytes()
    )

    proof = restore_hdm3_file_to_legacy_brotli(
        input_path=source,
        output_path=output,
        json_out=proof_path,
        require_hdm3=True,
    )

    assert output.exists()
    assert proof_path.exists()
    assert json.loads(proof_path.read_text(encoding="utf-8"))["output_payload_sha256"] == proof[
        "output_payload_sha256"
    ]


def test_prove_hnerv_hdm3_runtime_adapter_tool_writes_archive_parity_manifest(
    tmp_path: Path,
) -> None:
    raw = _synthetic_decoder_raw()
    source_decoder = brotli.compress(raw, quality=10)
    parsed = _parsed_decoder_from_legacy_brotli(source_decoder)
    hdm3, _stats = encode_hdm3_q_brotli_split_fixture(parsed)
    latents = brotli.compress(b"latents" * 100, quality=5)
    source_archive = tmp_path / "source.zip"
    candidate_archive = tmp_path / "candidate.zip"
    write_stored_single_member_zip(
        source_archive,
        member_name="x",
        payload=PackedHnervPayload(
            header=b"\xff\x00\x00\x00",
            decoder_packed_brotli=source_decoder,
            latents_and_sidecar_brotli=latents,
        ).to_bytes(),
    )
    write_stored_single_member_zip(
        candidate_archive,
        member_name="x",
        payload=PackedHnervPayload(
            header=b"\xff\x00\x00\x00",
            decoder_packed_brotli=hdm3,
            latents_and_sidecar_brotli=latents,
        ).to_bytes(),
    )
    json_out = tmp_path / "runtime_adapter_proof.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "prove_hnerv_hdm3_runtime_adapter.py"),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    proof = json.loads(json_out.read_text(encoding="utf-8"))
    assert proof["candidate_decoder_section_is_hdm3"] is True
    assert proof["decoder_raw_matches_source"] is True
    assert proof["latents_and_sidecar_match_source"] is True
    assert proof["runtime_adapter_integrated"] is True
    assert proof["restored_payload_matches_source"] is True
    assert proof["restored_decoder_section_matches_source"] is True
    assert proof["inflate_output_parity_proven_by_payload_identity"] is True
    assert proof["ready_for_public_runtime_inflate"] is True
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert "exact_inflate_output_parity_missing" not in proof["remaining_dispatch_blockers"]


def _parsed_decoder_from_legacy_brotli(decoder_brotli: bytes):
    from tac.hnerv_decoder_recode import parse_packed_decoder_brotli

    return parse_packed_decoder_brotli(decoder_brotli)


def _synthetic_decoder_raw() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = 1
        for dim in shape:
            count *= dim
        pattern = bytes(((index + i // 7) % 23) for i in range(128))
        repeats, remainder = divmod(count, len(pattern))
        q_parts.append(pattern * repeats + pattern[:remainder])
        scale_parts.append((index + 1).to_bytes(4, "little"))
    return b"".join(q_parts) + b"".join(scale_parts)
