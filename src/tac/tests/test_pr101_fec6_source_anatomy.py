# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tac.packet_compiler.pr101_fec6_packetir import FEC6_FIXED_K16_CODE_BITS
from tac.packet_compiler.pr101_fec6_source_anatomy import (
    PR101_DECODER_BLOB_LEN,
    PR101_LATENT_BLOB_LEN,
    SCHEMA,
    build_source_anatomy_sections,
    profile_pr101_fec6_source_payload_anatomy,
    render_source_anatomy_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "profile_pr101_fec6_source_payload_anatomy.py"
FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip"
)
NULL_INDICES = REPO_ROOT / "experiments/results/null_byte_probe_20260520T220614Z/null_byte_indices.npy"


def _load_tool():
    spec = importlib.util.spec_from_file_location("profile_pr101_fec6_source_payload_anatomy", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fec6_selector(codes: list[int]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    pad = (-len(bits)) % 8
    padded = bits + ("0" * pad)
    return b"FEC6" + len(codes).to_bytes(2, "little") + bytes(
        int(padded[index : index + 8], 2) for index in range(0, len(padded), 8)
    )


def _synthetic_archive(tmp_path: Path) -> Path:
    decoder = b"\x00" * PR101_DECODER_BLOB_LEN
    latent = b"\x01" * PR101_LATENT_BLOB_LEN
    sidecar = b"\x02" * 7
    source = decoder + latent + sidecar
    selector = _fec6_selector([0, 2, 7, 13])
    member = b"FP11" + len(source).to_bytes(4, "little") + source
    member += len(selector).to_bytes(2, "little") + selector
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, member, compress_type=zipfile.ZIP_STORED)
    return archive


def test_synthetic_source_anatomy_sections(tmp_path: Path) -> None:
    archive = _synthetic_archive(tmp_path)

    profile = profile_pr101_fec6_source_payload_anatomy(
        archive_path=archive,
        null_indices=np.array([0, 1, 8 + PR101_DECODER_BLOB_LEN], dtype=np.int64),
        include_magic=False,
    )
    sections = {row["name"]: row for row in profile["sections"]}

    assert profile["schema"] == SCHEMA
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["null_indices"]["coordinate_space"] == "fec6_member_payload_byte_offsets_zero_based_half_open"
    assert profile["null_indices"]["all_indices_covered_by_semantic_sections"] is True
    assert sections["pr101_decoder_blob"]["range"] == [8, 8 + PR101_DECODER_BLOB_LEN]
    assert sections["pr101_latent_blob"]["range"] == [
        8 + PR101_DECODER_BLOB_LEN,
        8 + PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN,
    ]
    assert sections["pr101_latent_blob"]["null_coverage"]["n_null_bytes"] == 1


def test_build_source_anatomy_sections_rejects_short_source() -> None:
    short_packet = SimpleNamespace(
        source_pr101_payload=b"\x00" * (PR101_DECODER_BLOB_LEN - 1),
        source_len_u32le=(PR101_DECODER_BLOB_LEN - 1).to_bytes(4, "little"),
        selector_len_u16le=(0).to_bytes(2, "little"),
        selector_fec6_payload=b"",
    )

    try:
        build_source_anatomy_sections(short_packet)
    except ValueError as exc:
        assert "shorter than PR101 decoder+latent constants" in str(exc)
    else:
        raise AssertionError("expected short PR101 source payload to be rejected")


def test_null_indices_rejects_out_of_member_payload_coordinate_space(tmp_path: Path) -> None:
    archive = _synthetic_archive(tmp_path)

    try:
        profile_pr101_fec6_source_payload_anatomy(
            archive_path=archive,
            null_indices=np.array([999_999_999], dtype=np.int64),
            include_magic=False,
        )
    except ValueError as exc:
        assert "member-payload byte offsets" in str(exc)
    else:
        raise AssertionError("expected out-of-range null index to be rejected")


def test_real_fec6_source_anatomy_if_present() -> None:
    if not FEC6_ARCHIVE.exists() or not NULL_INDICES.exists():
        return

    profile = profile_pr101_fec6_source_payload_anatomy(
        archive_path=FEC6_ARCHIVE,
        null_indices=np.load(NULL_INDICES),
        include_magic=True,
    )
    sections = {row["name"]: row for row in profile["sections"]}
    latent = profile["latent_typed_runtime_adapter_probe"]

    assert sections["pr101_latent_blob"]["length_bytes"] == 15_387
    assert sections["pr101_latent_blob"]["null_coverage"]["null_fraction_within_section"] == 1.0
    assert sections["pr101_sidecar_blob"]["length_bytes"] == 607
    assert sections["pr101_sidecar_blob"]["null_coverage"]["null_fraction_within_section"] == 1.0
    assert sections["selector_fec6_payload"]["null_coverage"]["n_null_bytes"] == 249
    assert profile["null_indices"]["coordinate_space"] == "fec6_member_payload_byte_offsets_zero_based_half_open"
    assert profile["null_indices"]["all_indices_covered_by_semantic_sections"] is True
    assert profile["rank1_null_run_decomposition"]["range"] == [162_171, 178_417]
    for row in latent["typed_candidates"]:
        assert row["score_claim"] is False
        assert row["score_claim_valid"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
    magic_rows = [
        row
        for row in latent["typed_candidates"]
        if row["candidate_id"] == "latent_delta_matrix_magic_codec"
    ]
    assert magic_rows[0]["lossless_roundtrip_verified"] is False
    assert magic_rows[0]["byte_count_comparison_role"] == "diagnostic_only_not_lossless_alternative"
    assert latent["best_typed_delta_vs_current_latent_blob"] >= 0
    assert profile["ranked_next_targets"][0]["target_id"] == "latent_blob_plus_sidecar_semantic_null_span"
    assert profile["cascade_relevance"]["score_claim"] is False
    assert "newly produced streams" in profile["cascade_relevance"]["summary"]
    assert "next artifact path" in profile["interpretation"]


def test_markdown_and_cli_smoke(tmp_path: Path) -> None:
    archive = _synthetic_archive(tmp_path)
    profile = profile_pr101_fec6_source_payload_anatomy(
        archive_path=archive,
        include_magic=False,
    )
    md = render_source_anatomy_markdown(profile)
    assert "# PR101/FEC6 Source Payload Anatomy" in md
    assert "Score claim valid" in md
    assert "Ready for exact eval dispatch" in md
    assert "Axis tag" in md
    assert "`pr101_latent_blob`" in md
    assert "Cascade Relevance" in md

    out_json = tmp_path / "profile.json"
    out_md = tmp_path / "profile.md"
    tool = _load_tool()
    rc = tool.main(
        [
            "--archive",
            str(archive),
            "--no-magic",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["schema"] == SCHEMA
    assert "Semantic Sections" in out_md.read_text(encoding="utf-8")
