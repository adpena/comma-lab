# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
import struct
import subprocess
import sys
from pathlib import Path

import brotli
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import pr106_entropy_floor_probe as probe  # type: ignore  # noqa: E402

from tac.hnerv_decoder_recode import (  # noqa: E402
    PACKED_STATE_SCHEMA,
    decode_hdm7_q_brotli_len_elided_fixture,
    encode_hdm4_q_brotli_split_fixture,
    encode_hdm7_q_brotli_len_elided_fixture,
    encode_hdm8_q_brotli_recipe_elided_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import read_packed_archive_view, write_stored_single_member_zip  # noqa: E402
from tac.packet_compiler.pr106_fixed_latent_recode import (  # noqa: E402
    encode_hlm1_fixed_latents_from_brotli,
)
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106SidecarPacket,
    emit_pr106_sidecar_packet,
)


def test_markov_floors_capture_deterministic_alternation() -> None:
    stream = probe.SymbolStream(
        "alternating",
        np.array([0, 1] * 64, dtype=np.uint8),
        2,
        "fixture",
    )

    rows = probe.floor_rows_for_group([stream], current_storage_bytes=128)
    identity = next(row for row in rows if row["transform"] == "identity")

    assert identity["iid_floor_bytes"] > identity["markov1_floor_bytes"]
    assert identity["markov2_floor_bytes"] <= identity["markov1_floor_bytes"]
    assert identity["markov1_floor_delta_vs_current_storage_bytes"] < 0
    assert identity["model_complexity"]["markov2_contexts_unpriced"] > 0


def test_zero0_split_preserves_symbol_count_accounting() -> None:
    stream = probe.SymbolStream(
        "s",
        np.array([0, 0, 5, 7, 0, 9], dtype=np.uint8),
        256,
        "fixture",
    )

    transformed = probe.transform_zero0_nonzero_value([stream])

    assert [item.name for item in transformed] == [
        "s:is_zero0",
        "s:nonzero_value_minus1",
    ]
    assert transformed[0].symbols.size == 6
    assert transformed[1].symbols.tolist() == [4, 6, 8]
    assert transformed[1].n_categories == 255


def test_synthetic_pr106_archive_report_has_custody_and_no_score_claim(tmp_path: Path) -> None:
    decoder_raw = _synthetic_decoder_raw()
    decoder_brotli = brotli.compress(decoder_raw, quality=5)
    latents_raw = _synthetic_fixed_latents_raw()
    latents_brotli = brotli.compress(latents_raw, quality=5)
    payload = b"\xff" + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)

    report = probe.build_report_from_archive(
        archive,
        pr101_reference_archive_bytes=178_258,
        active_floor_archive_bytes=185_578,
        active_floor_label="fixture_active_floor",
    )

    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["source"]["member_name"] == "0.bin"
    assert report["source"]["payload_magic"] == "ff_packed_hnerv"
    assert report["source"]["decoder_raw_bytes"] == len(decoder_raw)
    assert report["source"]["latents_raw_bytes"] == len(latents_raw)
    assert {group["group"] for group in report["groups"]} == {
        "decoder_q_zz_plus_f32_scales",
        "fixed_latents_delta_zz_plus_fp16_meta",
        "decoded_payload_sections_without_ff_header",
    }
    decoder_group = next(
        group for group in report["groups"] if group["group"] == "decoder_q_zz_plus_f32_scales"
    )
    assert decoder_group["current_storage_bytes"] == len(decoder_brotli)
    assert decoder_group["source_stream_count"] == len(PACKED_STATE_SCHEMA) + 1
    assert decoder_group["best_markov2_floor_bytes"] is not None
    assert decoder_group["floors"][0]["model_complexity"]["markov2_edges_unpriced"] > 0
    claim = report["adversarial_claim_check"]
    assert claim["verdict"] == (
        "pr101_only_not_transferable_to_pr106_without_pr106_specific_codec_and_exact_eval"
    )


def test_pr106_sidecar_wrapper_archive_is_unwrapped_with_outer_custody(tmp_path: Path) -> None:
    inner_payload = _synthetic_payload()
    sidecar_payload = b"\x01\x02\x03\x04"
    framing_meta = b"\x00\x00\x00\x00\x00\x00"
    wrapper_payload = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
            pr106_bytes=inner_payload,
            sidecar_payload=sidecar_payload,
            framing_meta=framing_meta,
        )
    )
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=wrapper_payload)

    report = probe.build_report_from_archive(
        archive,
        pr101_reference_archive_bytes=None,
        active_floor_archive_bytes=None,
        active_floor_label=None,
    )

    assert report["score_claim"] is False
    assert report["source"]["outer_payload_magic"] == "pr106_sidecar_wrapper"
    assert report["source"]["payload_magic"] == "ff_packed_hnerv"
    assert report["source"]["outer_payload_bytes"] == len(wrapper_payload)
    assert report["source"]["payload_bytes"] == len(inner_payload)
    assert report["source"]["sidecar_format_id"] == PR106_SIDECAR_FORMAT_PR101_GRAMMAR
    assert report["source"]["sidecar_payload_bytes"] == len(sidecar_payload)
    assert report["source"]["framing_meta_bytes"] == len(framing_meta)
    assert report["source"]["wrapper_unwrapped_for_entropy_model"] is True


def test_hdm4_decoder_section_is_decoded_for_entropy_probe(tmp_path: Path) -> None:
    raw = _synthetic_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=5)
    parsed = parse_packed_decoder_brotli(decoder_brotli)
    decoder_hdm4, _stats = encode_hdm4_q_brotli_split_fixture(parsed)
    latents_brotli = brotli.compress(_synthetic_fixed_latents_raw(), quality=5)
    payload = b"\xff" + len(decoder_hdm4).to_bytes(3, "little") + decoder_hdm4 + latents_brotli
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)

    report = probe.build_report_from_archive(
        archive,
        pr101_reference_archive_bytes=None,
        active_floor_archive_bytes=None,
        active_floor_label=None,
    )

    assert report["source"]["decoder_section_codec"] == "hdm4_q_brotli_split"
    assert report["source"]["decoder_raw_bytes"] == len(raw)
    assert report["groups"][0]["current_storage_label"] == "decoder_section_encoded"
    assert report["score_claim"] is False


def test_hdm7_decoder_section_is_decoded_for_entropy_probe(tmp_path: Path) -> None:
    raw = _synthetic_decoder_raw()
    decoder_brotli = brotli.compress(raw, quality=5)
    parsed = parse_packed_decoder_brotli(decoder_brotli)
    decoder_hdm7, _stats = encode_hdm7_q_brotli_len_elided_fixture(parsed)
    latents_brotli = brotli.compress(_synthetic_fixed_latents_raw(), quality=5)
    payload = b"\xff" + len(decoder_hdm7).to_bytes(3, "little") + decoder_hdm7 + latents_brotli
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)

    report = probe.build_report_from_archive(
        archive,
        pr101_reference_archive_bytes=None,
        active_floor_archive_bytes=None,
        active_floor_label=None,
    )

    assert report["source"]["decoder_section_codec"] == "hdm7_q_brotli_len_elided_split"
    assert report["source"]["decoder_raw_bytes"] == len(raw)
    assert report["groups"][0]["current_storage_label"] == "decoder_section_encoded"
    assert report["score_claim"] is False


def test_hdm8_decoder_section_is_decoded_for_entropy_probe(tmp_path: Path) -> None:
    hdm7_archive = (
        REPO
        / "experiments/results/pr106_r2_hdm6_hlm2_hdm7_candidate_20260514_codex/"
        "pr106_r2_hdm6_hlm2_xmember_hdm7_archive_candidate.zip"
    )
    if not hdm7_archive.exists():
        pytest.skip("HDM7 exact-CUDA candidate artifact is not present in this checkout")
    hdm7_view = read_packed_archive_view(hdm7_archive)
    parsed = decode_hdm7_q_brotli_len_elided_fixture(hdm7_view.packed.decoder_packed_brotli)
    decoder_hdm8, _stats = encode_hdm8_q_brotli_recipe_elided_fixture(parsed)
    payload = (
        b"\xff"
        + len(decoder_hdm8).to_bytes(3, "little")
        + decoder_hdm8
        + hdm7_view.packed.latents_and_sidecar_brotli
    )
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)

    report = probe.build_report_from_archive(
        archive,
        pr101_reference_archive_bytes=None,
        active_floor_archive_bytes=None,
        active_floor_label=None,
    )

    assert report["source"]["decoder_section_codec"] == "hdm8_q_brotli_fixed_lengths_split"
    assert report["source"]["decoder_raw_bytes"] == len(parsed.to_raw())
    assert report["groups"][0]["current_storage_label"] == "decoder_section_encoded"
    assert report["score_claim"] is False


def test_hlm1_latent_section_is_decoded_for_entropy_probe(tmp_path: Path) -> None:
    decoder_brotli = brotli.compress(_synthetic_decoder_raw(), quality=5)
    latents_brotli = brotli.compress(_synthetic_fixed_latents_raw(), quality=5)
    latents_hlm1 = encode_hlm1_fixed_latents_from_brotli(
        latents_brotli,
        brotli_candidates=((5, 16),),
    ).payload
    payload = b"\xff" + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_hlm1
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)

    report = probe.build_report_from_archive(
        archive,
        pr101_reference_archive_bytes=None,
        active_floor_archive_bytes=None,
        active_floor_label=None,
    )

    assert report["source"]["latents_section_codec"] == "hlm1_sparse_hi_delta_positions"
    assert report["source"]["latents_raw_bytes"] == len(_synthetic_fixed_latents_raw())
    assert report["score_claim"] is False


def test_pr106_entropy_floor_probe_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    payload_path = tmp_path / "0.bin"
    payload_path.write_bytes(payload)
    json_out = tmp_path / "probe.json"
    md_out = tmp_path / "probe.md"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "pr106_entropy_floor_probe.py"),
            "--payload-bin",
            str(payload_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--pr101-reference-archive-bytes",
            "178258",
        ],
        check=True,
        text=True,
    )

    assert json_out.exists()
    assert md_out.exists()
    text = md_out.read_text(encoding="utf-8")
    assert "PR106 Entropy Floor Probe" in text
    assert "Adversarial Claim Check" in text


def _synthetic_payload() -> bytes:
    decoder_brotli = brotli.compress(_synthetic_decoder_raw(), quality=5)
    latents_brotli = brotli.compress(_synthetic_fixed_latents_raw(), quality=5)
    return b"\xff" + len(decoder_brotli).to_bytes(3, "little") + decoder_brotli + latents_brotli


def _synthetic_decoder_raw() -> bytes:
    q_parts = []
    for index, (_name, shape) in enumerate(PACKED_STATE_SCHEMA):
        count = math.prod(shape)
        values = (np.arange(count, dtype=np.uint32) + index) % 251
        q_parts.append(values.astype(np.uint8).tobytes())
    scales = b"".join(
        struct.pack("<f", 1.0 + index / 100.0)
        for index in range(len(PACKED_STATE_SCHEMA))
    )
    return b"".join(q_parts) + scales


def _synthetic_fixed_latents_raw() -> bytes:
    total = probe.PR106_LATENT_N * probe.PR106_LATENT_D
    lo = (np.arange(total, dtype=np.uint32) % 251).astype(np.uint8).tobytes()
    mins = np.zeros(probe.PR106_LATENT_D, dtype=np.float16).tobytes()
    scales = np.ones(probe.PR106_LATENT_D, dtype=np.float16).tobytes()
    hi = np.zeros(total, dtype=np.uint8).tobytes()
    raw = lo + mins + scales + hi
    assert len(raw) == probe.PR106_FIXED_LATENT_RAW_BYTES
    return raw
