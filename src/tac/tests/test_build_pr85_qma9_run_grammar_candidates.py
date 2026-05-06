from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

from tac.pr85_bundle import FIXED_V5_LENGTHS, SEGMENT_ORDER, pack_pr85_bundle
from tac.qma9_run_grammar import (
    decode_qrg1_run_grammar,
    encode_qrg1_run_grammar,
    sha256_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_pr85_qma9_run_grammar_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_qma9_run_grammar_candidates_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qma9_payload(*, frame_count: int, width: int, height: int, bitstream_bytes: int) -> bytes:
    return struct.pack("<4sIIII", b"QMA9", frame_count, width, height, bitstream_bytes) + (
        b"s" * bitstream_bytes
    )


def _write_pr85_like_archive(path: Path, *, mask: bytes) -> None:
    segments = {name: (name.encode("ascii") + b"-payload") for name in SEGMENT_ORDER}
    segments["mask"] = mask
    segments["bias"] = b"B" * FIXED_V5_LENGTHS["bias"]
    segments["region"] = b"R" * FIXED_V5_LENGTHS["region"]
    raw = pack_pr85_bundle(segments, header_mode="v5")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, raw)


def test_qrg1_row_run_roundtrip_preserves_tokens() -> None:
    raw = bytes([0] * 8 + [1] * 8 + [2] * 16 + [3] * 32)
    encoded = encode_qrg1_run_grammar(raw, frame_count=1, width=4, height=16, mode="row_rle_zlib9")
    decoded = decode_qrg1_run_grammar(encoded.payload)

    assert decoded.data == raw
    assert decoded.sha256 == sha256_bytes(raw)
    assert decoded.header.magic == "QRG1"
    assert encoded.stats["rle_rows"] == 4


def test_byte_positive_qrg1_candidate_is_runtime_locked(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0] * 64 + [1] * 64 + [2] * 64 + [3] * 64)
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(raw)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(
        archive,
        mask=_qma9_payload(frame_count=1, width=8, height=32, bitstream_bytes=700),
    )

    summary = script.build_run_grammar_candidates(
        archive=archive,
        token_source=token_source,
        out_dir=tmp_path / "out",
        modes=("row_rle_zlib9",),
        expected_archive_sha256=None,
        expected_archive_bytes=None,
        expected_qma9_bytes=None,
        expected_qma9_sha256=None,
        expected_token_sha256=sha256_bytes(raw),
        expected_token_bytes=len(raw),
    )

    assert summary["planning_only"] is True
    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["dispatch_unlocked"] is False
    assert summary["byte_positive_candidate_count"] == 1
    assert summary["runtime_supported_byte_positive_candidate_count"] == 0
    assert summary["best_byte_positive_candidate"]["payload_magic"] == "QRG1"
    assert summary["best_byte_positive_candidate"]["runtime_supported"] is False
    assert summary["best_byte_positive_candidate"]["archive"] is None
    assert summary["best_byte_positive_candidate"]["token_parity"]["verified"] is True
    assert "byte_positive_qrg1_candidate_runtime_unsupported" in summary["blockers"]
    assert summary["dispatch_gate"]["safe_for_remote_dispatch"] is False
    assert (tmp_path / "out" / "candidate_summary.json").is_file()


def test_no_byte_positive_qrg1_candidate_records_blockers(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 1, 2, 3, 4, 0, 1, 2])
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(raw)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(
        archive,
        mask=_qma9_payload(frame_count=1, width=2, height=4, bitstream_bytes=1),
    )

    summary = script.build_run_grammar_candidates(
        archive=archive,
        token_source=token_source,
        out_dir=tmp_path / "out",
        modes=("row_rle_zlib9",),
        expected_archive_sha256=None,
        expected_archive_bytes=None,
        expected_qma9_bytes=None,
        expected_qma9_sha256=None,
        expected_token_sha256=sha256_bytes(raw),
        expected_token_bytes=len(raw),
    )

    assert summary["byte_positive_candidate_count"] == 0
    assert summary["best_byte_positive_candidate"] is None
    assert summary["fail_closed"]["emitted"] is True
    assert "no_byte_positive_qrg1_row_run_candidate" in summary["blockers"]
    assert "qma9_adaptive9bin_beats_screened_row_run_payloads" in summary["blockers"]
    assert summary["dispatch_gate"]["exact_eval_dispatch_allowed"] is False
