from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path("experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_single_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        assert [info.filename for info in infos] == ["0.bin"]
        assert infos[0].compress_type == zipfile.ZIP_STORED
        return zf.read(infos[0])


def test_pr103_ac_candidate_accounts_for_header_update_and_removed_word() -> None:
    if not (ROOT / "manifest.json").is_file():
        pytest.skip("local-only PR103 hidden-gem candidate artifact is not materialized")

    manifest = json.loads((ROOT / "manifest.json").read_text())
    source_archive = Path(manifest["source_archive"]["path"])
    candidate_archive = Path(manifest["candidate_archive"]["path"])

    source_payload = _read_single_member(source_archive)
    candidate_payload = _read_single_member(candidate_archive)

    assert _sha256(source_payload) == manifest["source_archive"]["payload_sha256"]
    assert _sha256(candidate_payload) == manifest["candidate_archive"]["payload_sha256"]
    assert len(source_payload) - len(candidate_payload) == 4

    mutation = manifest["mutation"]
    removed_offset = int(mutation["payload_offset"])
    assert source_payload[removed_offset : removed_offset + 4].hex() == mutation["removed_bytes_hex"]

    header = manifest["packed_header_update_proof"]
    header_start = int(header["payload_offset_start"])
    header_end = int(header["payload_offset_end"])
    assert source_payload[header_start:header_end].hex() == header["source_header_bytes_hex"]
    assert candidate_payload[header_start:header_end].hex() == header["candidate_header_bytes_hex"]
    assert int.from_bytes(source_payload[header_start:header_end], "little") == header["source_decoder_len"]
    assert int.from_bytes(candidate_payload[header_start:header_end], "little") == header["candidate_decoder_len"]

    expected = bytearray(source_payload)
    expected[header_start:header_end] = int(header["candidate_decoder_len"]).to_bytes(3, "little")
    del expected[removed_offset : removed_offset + 4]
    assert candidate_payload == bytes(expected)
    assert candidate_payload != source_payload[:removed_offset] + source_payload[removed_offset + 4 :]
