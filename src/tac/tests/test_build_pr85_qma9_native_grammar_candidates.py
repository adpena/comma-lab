# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from tac.qma9_range_mask_contract import decode_qma9_mask, encode_qma9_mask, sha256_bytes


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_pr85_qma9_native_grammar_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_qma9_native_grammar_candidates_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _write_pr85_like_archive(path: Path, *, mask: bytes) -> bytes:
    segments = {
        "mask": mask,
        "model": b"model-payload",
        "pose": b"pose-payload",
        "post": b"post-payload",
        "shift": b"shift-payload",
        "frac": b"frac-payload",
        "frac2": b"frac2-payload",
        "frac3": b"frac3-payload",
        "bias": b"b" * 223,
        "region": b"r" * 273,
        "randmulti": b"randmulti-tail",
    }
    header = b"".join(
        _u24(len(segments[name]))
        for name in ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3")
    )
    order = _load_script().parse_pr85_bundle.__globals__["SEGMENT_ORDER"]
    bundle = header + b"".join(segments[name] for name in order)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, bundle)
    return bundle


def _inflate_declared_bitstream(qma9: bytes, extra_zero_bytes: int) -> bytes:
    magic, frames, width, height, bitstream_bytes = struct.unpack_from("<4sIIII", qma9, 0)
    assert magic == b"QMA9"
    return (
        struct.pack("<4sIIII", magic, frames, width, height, bitstream_bytes + extra_zero_bytes)
        + qma9[20:]
        + (b"\x00" * extra_zero_bytes)
    )


def test_synthetic_trailing_zero_bitstream_trim_builds_byte_win_archive(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 1, 1, 2, 2, 3, 4, 4, 0, 0, 1, 1] * 2)
    qma9 = encode_qma9_mask(raw, frame_count=2, width=4, height=3)
    bloated_qma9 = _inflate_declared_bitstream(qma9, extra_zero_bytes=7)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(archive, mask=bloated_qma9)
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(raw)

    summary = script.build_native_grammar_candidates(
        archive=archive,
        token_source=token_source,
        out_dir=tmp_path / "out",
        expected_archive_sha256=None,
        expected_archive_bytes=None,
        expected_token_sha256=sha256_bytes(raw),
        expected_token_bytes=len(raw),
        decode_implementation="python",
        cpp_decoder=tmp_path / "missing.cpp",
        decode_timeout_seconds=5,
        max_prefix_trim_bytes=2,
        run_grammar_summary=tmp_path / "missing_run_summary.json",
        alt_grammar_summary=tmp_path / "missing_alt_summary.json",
        mode_sweep_summary=tmp_path / "missing_mode_sweep.json",
        macro_prior_dir=tmp_path / "missing_macro_prior",
    )

    assert summary["planning_only"] is True
    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["candidate_count"] >= 1
    assert summary["best_byte_delta"] == -7
    candidate = summary["best_candidate"]
    assert candidate["candidate_id"] == "qma9_declared_bitstream_trim7"
    assert candidate["runtime_compatibility"]["decoded_token_sha_matches_source"] is True
    assert candidate["archive_delta_bytes_vs_pr85"] == -7
    assert candidate["safe_for_remote_dispatch"] is False
    assert summary["family_resolution"]["family_id"] == "qma9_native_run_grammar_or_table_reduction"
    assert summary["family_resolution"]["status"] == "candidate_archives_built_local_only"
    assert candidate["archive"]["path"] in summary["family_resolution"]["candidate_archive_paths"]

    candidate_archive = REPO_ROOT / candidate["archive"]["path"]
    with zipfile.ZipFile(candidate_archive, "r") as zf:
        assert [info.filename for info in zf.infolist() if not info.is_dir()] == ["x"]
        x_payload = zf.read("x")
    rebuilt = script.parse_pr85_bundle(x_payload)
    decoded = decode_qma9_mask(rebuilt.segments["mask"])
    assert decoded.sha256 == sha256_bytes(raw)


def test_tight_qma9_stream_emits_no_candidate_and_dispatch_gate(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1] * 2)
    qma9 = encode_qma9_mask(raw, frame_count=2, width=4, height=3)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(archive, mask=qma9)
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(raw)

    summary = script.build_native_grammar_candidates(
        archive=archive,
        token_source=token_source,
        out_dir=tmp_path / "out",
        expected_archive_sha256=None,
        expected_archive_bytes=None,
        expected_token_sha256=sha256_bytes(raw),
        expected_token_bytes=len(raw),
        decode_implementation="python",
        cpp_decoder=tmp_path / "missing.cpp",
        decode_timeout_seconds=5,
        max_prefix_trim_bytes=2,
        run_grammar_summary=tmp_path / "missing_run_summary.json",
        alt_grammar_summary=tmp_path / "missing_alt_summary.json",
        mode_sweep_summary=tmp_path / "missing_mode_sweep.json",
        macro_prior_dir=tmp_path / "missing_macro_prior",
    )

    assert summary["candidate_count"] == 0
    assert summary["best_byte_delta"] is None
    assert summary["dispatch_gate"]["safe_for_remote_dispatch"] is False
    assert "source_qma9_segment_has_no_bytes_after_declared_bitstream" in summary["blockers"]
    assert "source_qma9_declared_bitstream_has_no_trailing_zero_bytes" in summary["blockers"]
    assert "no_byte_positive_runtime_supported_qma9_native_grammar_candidate" in summary["blockers"]
    assert "no_deterministic_byte_closed_pr85_qma9_native_run_or_table_candidate" in summary["blockers"]
    assert summary["family_resolution"]["status"] == (
        "fail_closed_no_byte_positive_runtime_supported_or_screened_run_table_candidate"
    )
    assert summary["family_resolution"]["minimal_missing_implementation"]
    prefix_results = [
        row for row in summary["screen_results"] if row["screen"] == "decode_proven_nonzero_suffix_byte_trim"
    ]
    assert prefix_results
    assert all("decoded_token_sha_mismatch" in row["rejection_reasons"] for row in prefix_results)
    assert (tmp_path / "out" / "candidate_summary.json").is_file()


def test_token_source_mismatch_fails_closed_before_candidate_archive(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes([0, 1, 2, 3, 4, 0] * 4)
    qma9 = encode_qma9_mask(raw, frame_count=2, width=4, height=3)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(archive, mask=_inflate_declared_bitstream(qma9, extra_zero_bytes=3))
    wrong_tokens = bytes([4, 3, 2, 1, 0, 4] * 4)
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(wrong_tokens)

    with pytest.raises(script.NativeGrammarCandidateError, match="source QMA9 decodes to token sha256"):
        script.build_native_grammar_candidates(
            archive=archive,
            token_source=token_source,
            out_dir=tmp_path / "out",
            expected_archive_sha256=None,
            expected_archive_bytes=None,
            expected_token_sha256=sha256_bytes(wrong_tokens),
            expected_token_bytes=len(wrong_tokens),
            decode_implementation="python",
            cpp_decoder=tmp_path / "missing.cpp",
            decode_timeout_seconds=5,
            max_prefix_trim_bytes=2,
            run_grammar_summary=tmp_path / "missing_run_summary.json",
            alt_grammar_summary=tmp_path / "missing_alt_summary.json",
            mode_sweep_summary=tmp_path / "missing_mode_sweep.json",
            macro_prior_dir=tmp_path / "missing_macro_prior",
        )

    assert not (tmp_path / "out" / "qma9_declared_bitstream_trim3" / "archive.zip").exists()
