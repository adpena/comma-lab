from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_decoder_recode import (
    PACKED_STATE_SCHEMA,
    build_structural_recode_profile,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv, write_stored_single_member_zip

REPO = Path(__file__).resolve().parents[3]


def test_parse_packed_decoder_brotli_roundtrips_raw() -> None:
    raw = _synthetic_decoder_raw()
    parsed = parse_packed_decoder_brotli(brotli.compress(raw, quality=5))

    assert parsed.to_raw() == raw
    assert len(parsed.records) == len(PACKED_STATE_SCHEMA)
    assert len(parsed.scale_stream) == 4 * len(PACKED_STATE_SCHEMA)


def test_structural_recode_profile_is_planning_only_and_raw_equal() -> None:
    packed = parse_ff_packed_brotli_hnerv(_packed_payload(brotli.compress(_synthetic_decoder_raw(), quality=5)))
    profile = build_structural_recode_profile(
        packed,
        source_label="fixture",
        source_archive_sha256="a" * 64,
    )

    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["ready_for_archive_preflight"] is False
    assert profile["record_count"] == len(PACKED_STATE_SCHEMA)
    assert all(row["raw_equal"] is True for row in profile["variants"])
    entropy = profile["entropy_summary"]
    assert entropy["score_claim"] is False
    assert entropy["q_stream_symbols"] == profile["q_stream_bytes"]
    assert entropy["global_q_entropy_floor_bytes"] <= profile["q_stream_bytes"]
    assert entropy["global_q_entropy_floor_plus_raw_scales_bytes"] >= entropy["global_q_entropy_floor_bytes"]
    assert entropy["current_static_model_interpretation"] in {
        "zero_order_q_symbol_floor_loses_to_current_brotli",
        "zero_order_q_symbol_floor_has_byte_headroom",
    }
    aq_global = next(
        row for row in profile["variants"] if row["variant"] == "aq_global_q_stream_plus_raw_scales"
    )
    assert "byte_gap_vs_global_q_entropy_floor_plus_raw_scales" in aq_global


def test_profile_hnerv_decoder_structural_recode_cli(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(
        archive,
        member_name="x",
        payload=_packed_payload(brotli.compress(_synthetic_decoder_raw(), quality=5)),
    )
    json_out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "profile_hnerv_decoder_structural_recode.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "fixture",
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["source_label"] == "fixture"
    assert payload["score_claim"] is False
    assert payload["best_variant"]["raw_equal"] is True


def _synthetic_decoder_raw() -> bytes:
    q_parts = []
    scale_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = 1
        for dim in shape:
            count *= dim
        q_parts.append(bytes((i + index) % 256 for i in range(count)))
        scale_parts.append((index + 1).to_bytes(4, "little"))
    return b"".join(q_parts) + b"".join(scale_parts)


def _packed_payload(decoder_brotli: bytes) -> bytes:
    latents = brotli.compress(b"latents", quality=5)
    return bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents
