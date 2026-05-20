# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tac.packet_compiler.pr101_fec6_candidate_queue import (
    build_pr101_fec6_packetir_candidate_queue,
)
from tac.packet_compiler.pr101_fec6_packetir import FEC6_FIXED_K16_CODE_BITS
from tac.packet_compiler.pr101_fec6_runtime_consumption import (
    PR101_FEC6_RUNTIME_CONSUMPTION_PROOF_FAMILY,
    PR101_FEC6_RUNTIME_CONSUMPTION_SCHEMA_VERSION,
    prove_pr101_fec6_runtime_consumption,
)
from tac.repo_io import sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "prove_pr101_fec6_runtime_consumption.py"
REAL_FEC6_DIR = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir"
)
REAL_FEC6_ARCHIVE = REAL_FEC6_DIR / "archive.zip"
REAL_FEC6_ARCHIVE_SHA256 = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "prove_pr101_fec6_runtime_consumption",
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
    padded = bits + ("0" * ((-len(bits)) % 8))
    payload = bytes(
        int(padded[index : index + 8], 2) for index in range(0, len(padded), 8)
    )
    return b"FEC6" + len(codes).to_bytes(2, "little") + payload


def _fp11_member(source: bytes = b"\x02source-pr101") -> bytes:
    selector = _fec6_selector([0, 2, 7, 13])
    return (
        b"FP11"
        + len(source).to_bytes(4, "little")
        + source
        + len(selector).to_bytes(2, "little")
        + selector
    )


def _single_member_archive(tmp_path: Path, payload: bytes) -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    return archive


def _synthetic_runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    src = runtime / "src"
    src.mkdir(parents=True)
    (src / "codec.py").write_text("# synthetic codec\n", encoding="utf-8")
    (src / "frame_selector.py").write_text("# synthetic selector\n", encoding="utf-8")
    (runtime / "inflate.py").write_text(
        """
import struct
import torch

FEC6_FIXED_K16_CODE_BITS = (
    "00", "1100", "01", "111010", "11010", "111011", "111100", "100",
    "111101", "11011", "1111110", "111110", "11111110", "101", "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE = {bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)}

def unpack_fec6_fixed_huffman_codes(payload, *, n_pairs):
    codes = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise ValueError("truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        if prefix in FEC6_FIXED_K16_DECODE:
            codes.append(FEC6_FIXED_K16_DECODE[prefix])
            prefix = ""
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise ValueError("non-zero padding")
    return codes

def unpack_compact_selector_codes(selector_payload):
    if selector_payload[:4] != b"FEC6":
        raise ValueError("bad selector")
    n_pairs = struct.unpack_from("<H", selector_payload, 4)[0]
    codes = unpack_fec6_fixed_huffman_codes(selector_payload[6:], n_pairs=n_pairs)
    specs = tuple(("identity", (), 0) for _ in FEC6_FIXED_K16_CODE_BITS)
    return codes, specs

def parse_pr101_frame_selector_archive(bin_bytes):
    if len(bin_bytes) < 10:
        raise ValueError("truncated")
    if bin_bytes[:4] != b"FP11":
        raise ValueError("bad magic")
    pos = 4
    source_len = struct.unpack_from("<I", bin_bytes, pos)[0]
    pos += 4
    source_payload = bin_bytes[pos:pos + source_len]
    pos += source_len
    if len(source_payload) != source_len:
        raise ValueError("source truncated")
    selector_len = struct.unpack_from("<H", bin_bytes, pos)[0]
    pos += 2
    selector_payload = bin_bytes[pos:pos + selector_len]
    pos += selector_len
    if len(selector_payload) != selector_len:
        raise ValueError("selector truncated")
    if pos != len(bin_bytes):
        raise ValueError("trailing bytes")
    codes, specs = unpack_compact_selector_codes(selector_payload)
    return source_payload, "compact", codes, specs

def parse_archive(archive_bytes):
    if not archive_bytes or archive_bytes[0] == 0:
        raise ValueError("bad synthetic source")
    value = float(archive_bytes[0])
    return {"w": torch.tensor([value])}, torch.tensor([value, len(archive_bytes)]), {"n_pairs": 4}
""".lstrip(),
        encoding="utf-8",
    )
    return runtime


def test_runtime_consumption_proof_passes_and_flips_candidate_queue(
    tmp_path: Path,
) -> None:
    member_payload = _fp11_member()
    archive = _single_member_archive(tmp_path, member_payload)
    runtime = _synthetic_runtime(tmp_path)

    proof = prove_pr101_fec6_runtime_consumption(
        archive_path=archive,
        runtime_dir=runtime,
        expected_archive_sha256=sha256_file(archive),
        repo_root=tmp_path,
    )
    proof_path = tmp_path / "proof.json"
    write_json(proof_path, proof)

    assert proof["schema_version"] == PR101_FEC6_RUNTIME_CONSUMPTION_SCHEMA_VERSION
    assert proof["proof_family"] == PR101_FEC6_RUNTIME_CONSUMPTION_PROOF_FAMILY
    assert proof["no_op_detector_passed"] is True
    assert proof["runtime_bytes_consumed"] == len(member_payload)
    assert "source_pr101_payload" in proof["consumed_section_names"]
    assert "selector_fec6_payload" in proof["consumed_section_names"]
    assert proof["score_claim"] is False
    assert proof["promotion_eligible"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert all(probe["passed"] is True for probe in proof["mutation_probes"])

    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=archive,
        expected_archive_sha256=sha256_file(archive),
        runtime_consumption_proof_path=proof_path,
    )
    assert queue["byte_accounting"]["runtime_consumption_proven"] is True
    assert queue["blockers"] == []
    assert queue["score_claim"] is False


def test_runtime_consumption_cli_writes_proof(tmp_path: Path) -> None:
    archive = _single_member_archive(tmp_path, _fp11_member())
    runtime = _synthetic_runtime(tmp_path)
    out_json = tmp_path / "runtime_proof.json"
    tool = _load_tool()

    rc = tool.main(
        [
            "--archive",
            str(archive),
            "--runtime-dir",
            str(runtime),
            "--expected-archive-sha256",
            sha256_file(archive),
            "--output-json",
            str(out_json),
        ]
    )

    proof = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert proof["no_op_detector_passed"] is True
    assert proof["runtime_consumption_proof_path"] == str(out_json)
    assert "runtime_consumption_proof_sha256" not in proof


def test_real_pr101_fec6_runtime_consumption_proof_if_present() -> None:
    if not REAL_FEC6_ARCHIVE.exists():
        pytest.skip(
            f"missing local PR101/FEC6 archive: {REAL_FEC6_ARCHIVE.relative_to(REPO_ROOT)}"
        )

    proof = prove_pr101_fec6_runtime_consumption(
        archive_path=REAL_FEC6_ARCHIVE,
        runtime_dir=REAL_FEC6_DIR,
        expected_archive_sha256=REAL_FEC6_ARCHIVE_SHA256,
        repo_root=REPO_ROOT,
    )

    assert proof["archive_sha256"] == REAL_FEC6_ARCHIVE_SHA256
    assert proof["member_name"] == "x"
    assert proof["member_payload_bytes"] == 178_417
    assert proof["runtime_bytes_consumed"] == 178_417
    assert proof["runtime_parse"]["source_len"] == 178_158
    assert proof["runtime_parse"]["selector_len"] == 249
    assert proof["runtime_parse"]["selector_code_count"] == 600
    assert proof["runtime_parse"]["selector_specs_count"] == 16
    assert proof["no_op_detector_passed"] is True
    assert proof["section_mutation_probe_count"] == 2
    assert "pr101_latent_blob" in proof["consumed_section_names"]
    assert "pr101_sidecar_blob" in proof["consumed_section_names"]
    ranges = {row["section_name"]: row["range"] for row in proof["consumed_byte_ranges"]}
    assert ranges["pr101_latent_blob"] == [162_172, 177_559]
    assert ranges["pr101_sidecar_blob"] == [177_559, 178_166]
    assert proof["score_claim"] is False
