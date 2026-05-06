from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

from tac.pr85_bundle import FIXED_V5_LENGTHS, SEGMENT_ORDER, pack_pr85_bundle
from tac.qma9_range_mask_contract import sha256_bytes


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_pr85_qma9_alt_grammar_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_qma9_alt_grammar_candidates_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qma9_payload(*, bitstream_bytes: int = 20) -> bytes:
    return struct.pack("<4sIIII", b"QMA9", 1, 2, 3, bitstream_bytes) + (b"s" * bitstream_bytes)


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


def _write_fake_codec(path: Path) -> None:
    path.write_text(
        r'''
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void put_u32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 255));
  out.push_back(static_cast<uint8_t>((value >> 8) & 255));
  out.push_back(static_cast<uint8_t>((value >> 16) & 255));
  out.push_back(static_cast<uint8_t>((value >> 24) & 255));
}

int main(int argc, char** argv) {
  if (argc != 7) return 2;
  std::vector<uint8_t> raw = read_file(argv[1]);
  int frames = std::stoi(argv[2]);
  int width = std::stoi(argv[3]);
  int height = std::stoi(argv[4]);
  std::string out_path = argv[5];
  std::string mode = argv[6];
  std::string magic = mode == "adaptive9bin" ? "QMA9" : "QMA8";
  uint32_t bitstream_bytes = mode == "fake_short" ? 10u : 30u;
  std::vector<uint8_t> payload;
  payload.insert(payload.end(), magic.begin(), magic.end());
  put_u32(payload, static_cast<uint32_t>(frames));
  put_u32(payload, static_cast<uint32_t>(width));
  put_u32(payload, static_cast<uint32_t>(height));
  put_u32(payload, bitstream_bytes);
  payload.insert(payload.end(), bitstream_bytes, static_cast<uint8_t>(mode[0]));
  std::ofstream out(out_path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  std::cout << "{\"mode\":\"" << mode << "\",\"raw_bytes\":" << raw.size()
            << ",\"bitstream_bytes\":" << bitstream_bytes
            << ",\"packed_bytes\":" << payload.size()
            << ",\"model_bytes\":20}" << std::endl;
  return 0;
}
''',
        encoding="utf-8",
    )


def test_byte_positive_alt_payload_is_runtime_locked(tmp_path: Path) -> None:
    script = _load_script()
    tokens = bytes([0, 1, 2, 3, 4, 0])
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(tokens)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(archive, mask=_qma9_payload(bitstream_bytes=20))
    fake_codec = tmp_path / "fake_codec.cpp"
    live_runtime = tmp_path / "live_runtime.cpp"
    _write_fake_codec(fake_codec)
    live_runtime.write_text("// live runtime placeholder\n", encoding="utf-8")

    summary = script.build_alt_grammar_candidates(
        archive=archive,
        token_source=token_source,
        out_dir=tmp_path / "out",
        replay_codec_cpp=fake_codec,
        live_runtime_cpp=live_runtime,
        modes=("adaptive9bin", "fake_short"),
        expected_archive_sha256=None,
        expected_archive_bytes=None,
        expected_token_sha256=sha256_bytes(tokens),
        expected_token_bytes=len(tokens),
        timeout_seconds_per_mode=10,
        native_summary=tmp_path / "missing_native.json",
        opportunity_matrix=tmp_path / "missing_matrix.json",
    )

    assert summary["planning_only"] is True
    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["dispatch_unlocked"] is False
    assert summary["byte_positive_candidate_count"] == 1
    assert summary["runtime_supported_byte_positive_candidate_count"] == 0
    assert summary["best_byte_positive_candidate"]["mode"] == "fake_short"
    assert summary["best_byte_positive_candidate"]["payload_magic"] == "QMA8"
    assert summary["best_byte_positive_candidate"]["archive"] is None
    assert summary["best_byte_positive_candidate"]["runtime_custody_contract"]["live_runtime_supported"] is False
    assert "byte_positive_payload_requires_runtime_edit_before_archive_dispatch" in summary["best_byte_positive_candidate"]["archive_rejection_reasons"]
    assert summary["dispatch_gate"]["safe_for_remote_dispatch"] is False
    assert (tmp_path / "out" / "candidate_summary.json").is_file()


def test_no_byte_positive_alt_emits_fail_closed_economics(tmp_path: Path) -> None:
    script = _load_script()
    tokens = bytes([0, 1, 2, 3, 4, 0])
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(tokens)
    archive = tmp_path / "archive.zip"
    _write_pr85_like_archive(archive, mask=_qma9_payload(bitstream_bytes=20))
    fake_codec = tmp_path / "fake_codec.cpp"
    live_runtime = tmp_path / "live_runtime.cpp"
    _write_fake_codec(fake_codec)
    live_runtime.write_text("// live runtime placeholder\n", encoding="utf-8")

    summary = script.build_alt_grammar_candidates(
        archive=archive,
        token_source=token_source,
        out_dir=tmp_path / "out",
        replay_codec_cpp=fake_codec,
        live_runtime_cpp=live_runtime,
        modes=("fake_long",),
        expected_archive_sha256=None,
        expected_archive_bytes=None,
        expected_token_sha256=sha256_bytes(tokens),
        expected_token_bytes=len(tokens),
        timeout_seconds_per_mode=10,
        native_summary=tmp_path / "missing_native.json",
        opportunity_matrix=tmp_path / "missing_matrix.json",
    )

    assert summary["byte_positive_candidate_count"] == 0
    assert summary["fail_closed"]["emitted"] is True
    assert summary["fail_closed"]["source_qma9_payload_bytes"] == 40
    assert summary["fail_closed"]["best_alt_delta_bytes_vs_source_qma9"] == 10
    assert "no_byte_positive_alt_grammar_candidate" in summary["blockers"]
    assert "qma9_adaptive9bin_remains_smallest_observed_full_stream" in summary["blockers"]
