# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import tac.packet_compiler.pr106_runtime_consumption as runtime_consumption_mod
import tac.substrates._shared.inflate_runtime as repo_inflate_runtime
from tac.packet_compiler import (
    PR106_PACKET_IR_SECTION_HASH_DOMAIN,
    PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    build_pr106_sidecar_recode_candidate_packet,
    decode_pr106_sidecar_packet_dim_delta,
    dumps_runtime_consumption_manifest,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    load_pr106_sidecar_runtime,
    lossless_pr106_sidecar_recode_candidates,
    mutate_pr106_sidecar_semantic_correction,
    parse_pr106_sidecar_packet,
    pr106_runtime_source_manifest,
    prove_pr106_same_runtime_full_frame_parity,
    prove_pr106_sidecar_runtime_decode_consumption,
    read_single_stored_member_archive,
    runtime_framing_meta_consumption_probe,
    runtime_full_frame_streaming_digest,
    runtime_sidecar_correction_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_R2_ARCHIVE = REPO_ROOT / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_RUNTIME = REPO_ROOT / "submissions/pr106_latent_sidecar_r2"
PR106_R2_SHA = "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f"
PR106_R2_RUNTIME_TREE_SHA = "5cb9cc16fa2fae0f3887a6890edd71acf0f89e9cf809cadf22e02d60154e93dc"
PR106_R2_PR101_ARCHIVE = (
    REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)
PR106_R2_PR101_RUNTIME = REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar"
PR106_R2_PR101_SHA = "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383"
PR106_R2_PR101_RUNTIME_TREE_SHA = "373f19a1a892cf21c432d4949312cc788f4d4d23c02f2c1ca0cb3e666fc5c4bc"
PR106_HDM8_FORMAT07_ARCHIVE = (
    REPO_ROOT / "src/tac/tests/fixtures/pr106_hdm8_format07.archive.zip"
)
PR106_FORMAT0D_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip"
)
PR106_FORMAT0D_SHA = "9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4"
PR106_FORMAT04_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/"
    / "pr106_sidecar_rank_elided_format04_candidate.zip"
)
RUNTIME_CONSUMPTION_TOOL = REPO_ROOT / "tools" / "prove_pr106_sidecar_runtime_consumption.py"


@pytest.mark.parametrize(
    ("archive_path", "runtime_dir", "format_id"),
    [
        (PR106_R2_ARCHIVE, PR106_R2_RUNTIME, "0x01"),
        (PR106_R2_PR101_ARCHIVE, PR106_R2_PR101_RUNTIME, "0x02"),
    ],
)
def test_pr106_sidecar_runtime_decode_consumption_proof_is_nonpromotable(
    archive_path: Path,
    runtime_dir: Path,
    format_id: str,
) -> None:
    expected_sha = PR106_R2_PR101_SHA if format_id == "0x02" else PR106_R2_SHA
    expected_runtime_tree = (
        PR106_R2_PR101_RUNTIME_TREE_SHA if format_id == "0x02" else PR106_R2_RUNTIME_TREE_SHA
    )
    expected_runtime_content_tree = pr106_runtime_source_manifest(runtime_dir)[
        "runtime_content_tree_sha256"
    ]
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive_path,
        runtime_dir=runtime_dir,
        expected_archive_sha256=expected_sha,
        expected_runtime_source_tree_sha256=expected_runtime_tree,
        expected_runtime_content_tree_sha256=expected_runtime_content_tree,
    )

    assert manifest["schema"] == "pr106_sidecar_runtime_decode_consumption_proof_v1"
    assert manifest["archive_member_name"] == "0.bin"
    assert manifest["archive"]["expected_sha256_matches"] is True
    assert manifest["archive"]["expected_member_name"] is None
    assert manifest["archive"]["expected_member_name_matches"] is None
    runtime_manifest = manifest["runtime_source_manifest"]
    assert isinstance(runtime_manifest, dict)
    assert runtime_manifest["runtime_source_tree_sha256"] == expected_runtime_tree
    assert runtime_manifest["runtime_content_tree_sha256"] == expected_runtime_content_tree
    assert runtime_manifest["expected_runtime_source_tree_sha256"] == expected_runtime_tree
    assert runtime_manifest["expected_runtime_source_tree_sha256_matches"] is True
    assert (
        runtime_manifest["expected_runtime_content_tree_sha256"]
        == expected_runtime_content_tree
    )
    assert runtime_manifest["expected_runtime_content_tree_sha256_matches"] is True
    assert manifest["format_id"] == format_id
    assert manifest["payload_sha256_changed"] is True
    assert manifest["inner_pr106_payload_sha256_unchanged"] is True
    assert manifest["sidecar_payload_sha256_changed"] is True
    assert manifest["parser_consumed_byte_accounting_passed"] is True
    assert manifest["packet_ir_consumed_byte_accounting_passed"] is True
    assert manifest["source_packet_ir_consumed_byte_proof"][
        "all_payload_bytes_accounted"
    ] is True
    assert manifest["mutated_packet_ir_consumed_byte_proof"][
        "all_payload_bytes_accounted"
    ] is True
    assert manifest["runtime_semantic_digest_changed"] is True
    assert manifest["runtime_corrected_latents_digest_changed"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    expected_framing = True if format_id == "0x02" else None
    assert consumed_sections["framing_meta"] is expected_framing
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["proof_not_score"] is True
    assert manifest["evidence_axis"] == "runtime-sidecar-decode-local-no-score"
    assert manifest["device_axis_label"] == "local-runtime-decode-no-full-frame"
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["blockers"] == []
    archive = manifest["archive"]
    assert isinstance(archive, dict)
    assert archive["member_name"] == "0.bin"

    source_digest = manifest["source_runtime_correction_digest"]
    mutated_digest = manifest["mutated_runtime_correction_digest"]
    assert isinstance(source_digest, dict)
    assert isinstance(mutated_digest, dict)
    assert source_digest["format_id"] == format_id
    assert mutated_digest["format_id"] == format_id
    assert source_digest["n_pairs"] == 600
    assert mutated_digest["n_pairs"] == 600
    assert source_digest["combined_sha256"] != mutated_digest["combined_sha256"]
    assert source_digest["source_latents_sha256"] == mutated_digest["source_latents_sha256"]
    assert source_digest["corrected_latents_sha256"] != mutated_digest[
        "corrected_latents_sha256"
    ]
    assert source_digest["latents_changed_by_sidecar"] is True
    assert mutated_digest["latents_changed_by_sidecar"] is True
    assert "score_claim" in dumps_runtime_consumption_manifest(manifest)


def test_pr106_sidecar_runtime_decode_consumption_fails_closed_on_sha_mismatch() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256="0" * 64,
    )

    assert manifest["schema"] == "pr106_sidecar_runtime_decode_consumption_proof_v1"
    assert manifest["archive"]["expected_sha256_matches"] is False
    assert manifest["blockers"] == ["expected_archive_sha256_mismatch"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["runtime_sidecar_apply_consumption_claim"] is False
    assert manifest["packet_ir_consumed_byte_accounting_passed"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["proof_not_score"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_pr106_sidecar_runtime_decode_consumption_fails_closed_on_runtime_tree_mismatch() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256=PR106_R2_PR101_SHA,
        expected_runtime_source_tree_sha256="0" * 64,
    )

    runtime_manifest = manifest["runtime_source_manifest"]
    assert isinstance(runtime_manifest, dict)
    assert runtime_manifest["expected_runtime_source_tree_sha256_matches"] is False
    assert manifest["blockers"] == ["expected_runtime_source_tree_sha256_mismatch"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["runtime_sidecar_apply_consumption_claim"] is False
    assert manifest["score_claim"] is False


def test_pr106_sidecar_runtime_decode_consumption_fails_closed_on_runtime_content_tree_mismatch() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256=PR106_R2_PR101_SHA,
        expected_runtime_content_tree_sha256="0" * 64,
    )

    runtime_manifest = manifest["runtime_source_manifest"]
    assert isinstance(runtime_manifest, dict)
    assert runtime_manifest["expected_runtime_content_tree_sha256_matches"] is False
    assert manifest["blockers"] == ["expected_runtime_content_tree_sha256_mismatch"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["runtime_sidecar_apply_consumption_claim"] is False
    assert manifest["score_claim"] is False


def test_pr106_sidecar_runtime_decode_consumption_fails_closed_on_malformed_sha() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256="not-a-sha",
    )

    assert manifest["schema"] == "pr106_sidecar_runtime_decode_consumption_proof_v1"
    assert manifest["archive"]["expected_sha256_well_formed"] is False
    assert manifest["blockers"] == ["expected_archive_sha256_malformed"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["runtime_sidecar_apply_consumption_claim"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["proof_not_score"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_pr106_sidecar_runtime_decode_consumption_fails_closed_on_malformed_runtime_tree_sha() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256=PR106_R2_PR101_SHA,
        expected_runtime_source_tree_sha256="not-a-sha",
    )

    runtime_manifest = manifest["runtime_source_manifest"]
    assert isinstance(runtime_manifest, dict)
    assert runtime_manifest["expected_runtime_source_tree_sha256_well_formed"] is False
    assert manifest["blockers"] == ["expected_runtime_source_tree_sha256_malformed"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["score_claim"] is False


def test_pr106_sidecar_runtime_decode_consumption_fails_closed_on_malformed_runtime_content_tree_sha() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256=PR106_R2_PR101_SHA,
        expected_runtime_content_tree_sha256="not-a-sha",
    )

    runtime_manifest = manifest["runtime_source_manifest"]
    assert isinstance(runtime_manifest, dict)
    assert runtime_manifest["expected_runtime_content_tree_sha256_well_formed"] is False
    assert manifest["blockers"] == ["expected_runtime_content_tree_sha256_malformed"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["score_claim"] is False


def test_pr106_sidecar_runtime_consumption_tool_fails_closed_on_sha_mismatch(
    tmp_path: Path,
) -> None:
    out = tmp_path / "runtime_consumption.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNTIME_CONSUMPTION_TOOL),
            "--archive",
            str(PR106_R2_PR101_ARCHIVE),
            "--runtime-dir",
            str(PR106_R2_PR101_RUNTIME),
            "--expected-archive-sha256",
            "0" * 64,
            "--output-json",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["blockers"] == ["expected_archive_sha256_mismatch"]
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["proof_not_score"] is True


def test_pr106_runtime_decode_consumption_autodetects_x_member(
    tmp_path: Path,
) -> None:
    source_member = read_single_stored_member_archive(PR106_R2_ARCHIVE.read_bytes())
    archive = tmp_path / "x_member.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(source_member)(
            name="x",
            payload=source_member.payload,
            date_time=source_member.date_time,
            external_attr=source_member.external_attr,
            create_system=source_member.create_system,
            flag_bits=source_member.flag_bits,
            comment=source_member.comment,
            extra=source_member.extra,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_RUNTIME,
    )

    assert manifest["archive_member_name"] == "x"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["score_claim"] is False


def test_pr106_runtime_source_manifest_is_deterministic_and_runtime_bound() -> None:
    manifest_a = pr106_runtime_source_manifest(PR106_R2_PR101_RUNTIME)
    manifest_b = pr106_runtime_source_manifest(PR106_R2_PR101_RUNTIME)

    assert manifest_a == manifest_b
    assert manifest_a["schema"] == "pr106_runtime_source_manifest_v1"
    assert manifest_a["runtime_source_tree_sha256"] == PR106_R2_PR101_RUNTIME_TREE_SHA
    assert isinstance(manifest_a["runtime_content_tree_sha256"], str)
    assert len(manifest_a["runtime_content_tree_sha256"]) == 64
    assert manifest_a["required_files"] == [
        "inflate.sh",
        "inflate.py",
        "src/codec.py",
        "src/model.py",
    ]
    files = manifest_a["files"]
    assert isinstance(files, list)
    paths = [item["path"] for item in files]
    assert "inflate.sh" in paths
    assert "inflate.py" in paths
    assert "src/pr101_grammar.py" in paths
    inflate_sh = next(item for item in files if item["path"] == "inflate.sh")
    assert inflate_sh["mode"] == "0755"


def test_pr106_runtime_source_manifest_covers_vendored_tac_helpers() -> None:
    manifest = pr106_runtime_source_manifest(PR106_R2_RUNTIME)

    assert manifest["runtime_source_tree_sha256"] == PR106_R2_RUNTIME_TREE_SHA
    files = manifest["files"]
    assert isinstance(files, list)
    paths = [item["path"] for item in files]
    assert "src/tac/__init__.py" in paths
    assert "src/tac/substrates/__init__.py" in paths
    assert "src/tac/substrates/_shared/__init__.py" in paths
    assert "src/tac/substrates/_shared/inflate_runtime.py" in paths


def test_pr106_runtime_import_context_prefers_vendored_tac_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_if_repo_helper_used(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("runtime proof imported repository tac helper")

    monkeypatch.setenv("PACT_INFLATE_DEVICE", "cpu")
    monkeypatch.setattr(
        repo_inflate_runtime,
        "select_inflate_device",
        _raise_if_repo_helper_used,
    )

    runtime = load_pr106_sidecar_runtime(PR106_R2_RUNTIME)

    assert str(runtime.select_inflate_device()) == "cpu"


def test_pr106_runtime_source_manifest_requires_inflate_sh(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    shutil.copytree(PR106_R2_PR101_RUNTIME, runtime)
    (runtime / "inflate.sh").unlink()

    with pytest.raises(ValueError, match=r"inflate.sh"):
        pr106_runtime_source_manifest(runtime)


def test_pr106_pr101_grammar_runtime_consumes_framing_meta_fail_closed() -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    baseline_digest = runtime_sidecar_correction_digest(runtime, member.payload)
    probe = runtime_framing_meta_consumption_probe(runtime, member.payload, baseline_digest)

    assert packet.framing_meta is not None
    assert probe["section"] == "framing_meta"
    assert probe["format_id"] == "0x02"
    assert probe["applicable"] is True
    assert probe["runtime_consumption_claim"] is True
    assert probe["observation"] in {
        "runtime_rejected_mutated_framing_meta",
        "runtime_digest_changed",
    }


def test_pr106_pr101_runtime_accepts_legacy_brotli_format_for_same_runtime_pairing() -> None:
    """The 0x02 runtime can decode 0x01 payloads; compare only in same runtime."""
    source_member = read_single_stored_member_archive(PR106_R2_ARCHIVE.read_bytes())
    source_packet = parse_pr106_sidecar_packet(source_member.payload)
    legacy_payload = emit_pr106_sidecar_packet(source_packet)
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    digest = runtime_sidecar_correction_digest(runtime, legacy_payload)
    grammar_digest = runtime_sidecar_correction_digest(runtime, grammar_member.payload)

    assert digest["format_id"] == "0x01"
    assert grammar_digest["format_id"] == "0x02"
    assert digest["n_pairs"] == 600
    assert digest["combined_sha256"] == grammar_digest["combined_sha256"]
    assert digest["corrected_latents_sha256"] == grammar_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_runtime_consumption_fails_closed_for_format04_runtime_gap() -> None:
    """Regression: PacketIR format 0x04 parsed/emitted correctly but could
    not be mutation-probed, so runtime-consumption proof regeneration halted
    before rebuilding the PR106 candidate matrix. The current PR101 runtime
    does not consume 0x04, so this must return a blocker manifest instead of
    raising and aborting the whole regeneration.
    """
    if not PR106_FORMAT04_ARCHIVE.is_file():
        pytest.skip(f"live format 0x04 archive missing: {PR106_FORMAT04_ARCHIVE}")

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_FORMAT04_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x04"
    assert manifest["packet_ir_consumed_byte_accounting_passed"] is True
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["runtime_sidecar_apply_consumption_claim"] is False
    assert manifest["runtime_all_score_affecting_sections_consumed"] is False
    mutation = manifest["mutation"]
    assert mutation["format_id"] == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED
    assert mutation["section_name"] == "sidecar_payload"
    assert manifest["blockers"] == ["runtime_sidecar_decode_exception:ValueError"]
    assert manifest["score_claim"] is False


def _fixed_meta_rank_elided_member_payload() -> bytes:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    grammar_packet = parse_pr106_sidecar_packet(grammar_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(grammar_packet)
    fixed = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_fixed_meta_rank_elided_sidecar_format_0x05"]
    fixed_packet = build_pr106_sidecar_recode_candidate_packet(grammar_packet, fixed)
    assert fixed_packet.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED
    return emit_pr106_sidecar_packet(fixed_packet)


def _implicit_len_fixed_meta_rank_elided_member_payload() -> bytes:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    grammar_packet = parse_pr106_sidecar_packet(grammar_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(grammar_packet)
    implicit = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06"]
    implicit_packet = build_pr106_sidecar_recode_candidate_packet(grammar_packet, implicit)
    assert (
        implicit_packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    return emit_pr106_sidecar_packet(implicit_packet)


def _headerless_implicit_len_fixed_meta_rank_elided_member_payload() -> bytes:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    grammar_packet = parse_pr106_sidecar_packet(grammar_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(grammar_packet)
    headerless = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07"]
    headerless_packet = build_pr106_sidecar_recode_candidate_packet(
        grammar_packet,
        headerless,
    )
    assert (
        headerless_packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    return emit_pr106_sidecar_packet(headerless_packet)


def _hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_member_payload() -> bytes:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    hdm8_packet = parse_pr106_sidecar_packet(hdm8_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(hdm8_packet)
    inner_headerless = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x08"]
    inner_headerless_packet = build_pr106_sidecar_recode_candidate_packet(
        hdm8_packet,
        inner_headerless,
    )
    assert (
        inner_headerless_packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    )
    return emit_pr106_sidecar_packet(inner_headerless_packet)


def _hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_member_payload() -> bytes:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    hdm8_packet = parse_pr106_sidecar_packet(hdm8_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(hdm8_packet)
    inner_headerless = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09"]
    inner_headerless_packet = build_pr106_sidecar_recode_candidate_packet(
        hdm8_packet,
        inner_headerless,
    )
    assert (
        inner_headerless_packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    )
    return emit_pr106_sidecar_packet(inner_headerless_packet)


def _hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_member_payload() -> bytes:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    hdm8_packet = parse_pr106_sidecar_packet(hdm8_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(hdm8_packet)
    magicless = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }["pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b"]
    magicless_packet = build_pr106_sidecar_recode_candidate_packet(
        hdm8_packet,
        magicless,
    )
    assert (
        magicless_packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    return emit_pr106_sidecar_packet(magicless_packet)


def _hdm9_hlm3_magicless_exact_radix_member_payload() -> bytes:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    hdm8_packet = parse_pr106_sidecar_packet(hdm8_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(hdm8_packet)
    exact_radix = {
        candidate.name: candidate
        for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
    }[
        "pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c"
    ]
    exact_radix_packet = build_pr106_sidecar_recode_candidate_packet(
        hdm8_packet,
        exact_radix,
    )
    assert (
        exact_radix_packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    )
    return emit_pr106_sidecar_packet(exact_radix_packet)


def test_pr106_pr101_runtime_accepts_fixed_meta_rank_elided_format() -> None:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    fixed_payload = _fixed_meta_rank_elided_member_payload()

    fixed_digest = runtime_sidecar_correction_digest(runtime, fixed_payload)
    grammar_digest = runtime_sidecar_correction_digest(runtime, grammar_member.payload)

    assert fixed_digest["format_id"] == "0x05"
    assert grammar_digest["format_id"] == "0x02"
    assert fixed_digest["combined_sha256"] == grammar_digest["combined_sha256"]
    assert fixed_digest["corrected_latents_sha256"] == grammar_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_pr101_runtime_accepts_implicit_len_fixed_meta_rank_elided_format() -> None:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    implicit_payload = _implicit_len_fixed_meta_rank_elided_member_payload()

    implicit_digest = runtime_sidecar_correction_digest(runtime, implicit_payload)
    grammar_digest = runtime_sidecar_correction_digest(runtime, grammar_member.payload)

    assert implicit_digest["format_id"] == "0x06"
    assert grammar_digest["format_id"] == "0x02"
    assert implicit_digest["combined_sha256"] == grammar_digest["combined_sha256"]
    assert implicit_digest["corrected_latents_sha256"] == grammar_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_pr101_runtime_accepts_headerless_implicit_len_fixed_meta_rank_elided_format() -> None:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    headerless_payload = _headerless_implicit_len_fixed_meta_rank_elided_member_payload()

    headerless_digest = runtime_sidecar_correction_digest(runtime, headerless_payload)
    grammar_digest = runtime_sidecar_correction_digest(runtime, grammar_member.payload)

    assert headerless_digest["format_id"] == "0x07"
    assert grammar_digest["format_id"] == "0x02"
    assert headerless_digest["combined_sha256"] == grammar_digest["combined_sha256"]
    assert headerless_digest["corrected_latents_sha256"] == grammar_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_pr101_runtime_accepts_hdm8_hlm2_inner_headerless_format() -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    inner_headerless_payload = _hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_member_payload()

    inner_headerless_digest = runtime_sidecar_correction_digest(
        runtime,
        inner_headerless_payload,
    )
    headerless_digest = runtime_sidecar_correction_digest(runtime, hdm8_member.payload)

    assert inner_headerless_digest["format_id"] == "0x08"
    assert headerless_digest["format_id"] == "0x07"
    assert inner_headerless_digest["combined_sha256"] == headerless_digest["combined_sha256"]
    assert inner_headerless_digest["corrected_latents_sha256"] == headerless_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_pr101_runtime_accepts_hdm9_hlm2_inner_headerless_format() -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    inner_headerless_payload = _hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_member_payload()

    inner_headerless_digest = runtime_sidecar_correction_digest(
        runtime,
        inner_headerless_payload,
    )
    headerless_digest = runtime_sidecar_correction_digest(runtime, hdm8_member.payload)

    assert inner_headerless_digest["format_id"] == "0x09"
    assert headerless_digest["format_id"] == "0x07"
    assert inner_headerless_digest["combined_sha256"] == headerless_digest["combined_sha256"]
    assert inner_headerless_digest["corrected_latents_sha256"] == headerless_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_pr101_runtime_consumption_proves_fixed_meta_rank_elided_sidecar(
    tmp_path: Path,
) -> None:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    fixed_payload = _fixed_meta_rank_elided_member_payload()
    archive = tmp_path / "fixed_meta_rank_elided.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(grammar_member)(
            name=grammar_member.name,
            payload=fixed_payload,
            date_time=grammar_member.date_time,
            external_attr=grammar_member.external_attr,
            create_system=grammar_member.create_system,
            flag_bits=grammar_member.flag_bits,
            comment=grammar_member.comment,
            extra=grammar_member.extra,
            archive_comment=grammar_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x05"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["score_claim"] is False


def test_pr106_pr101_runtime_consumption_proves_implicit_len_fixed_meta_rank_elided_sidecar(
    tmp_path: Path,
) -> None:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    implicit_payload = _implicit_len_fixed_meta_rank_elided_member_payload()
    archive = tmp_path / "implicit_len_fixed_meta_rank_elided.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(grammar_member)(
            name=grammar_member.name,
            payload=implicit_payload,
            date_time=grammar_member.date_time,
            external_attr=grammar_member.external_attr,
            create_system=grammar_member.create_system,
            flag_bits=grammar_member.flag_bits,
            comment=grammar_member.comment,
            extra=grammar_member.extra,
            archive_comment=grammar_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x06"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["score_claim"] is False


def test_pr106_pr101_runtime_consumption_proves_headerless_implicit_len_sidecar(
    tmp_path: Path,
) -> None:
    grammar_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    headerless_payload = _headerless_implicit_len_fixed_meta_rank_elided_member_payload()
    archive = tmp_path / "headerless_implicit_len_fixed_meta_rank_elided.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(grammar_member)(
            name=grammar_member.name,
            payload=headerless_payload,
            date_time=grammar_member.date_time,
            external_attr=grammar_member.external_attr,
            create_system=grammar_member.create_system,
            flag_bits=grammar_member.flag_bits,
            comment=grammar_member.comment,
            extra=grammar_member.extra,
            archive_comment=grammar_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x07"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["score_claim"] is False


def test_pr106_pr101_runtime_consumption_proves_hdm8_hlm2_inner_headerless_sidecar(
    tmp_path: Path,
) -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    inner_headerless_payload = _hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_member_payload()
    archive = tmp_path / "hdm8_hlm2_inner_headerless_fixed_meta_rank_elided.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(hdm8_member)(
            name=hdm8_member.name,
            payload=inner_headerless_payload,
            date_time=hdm8_member.date_time,
            external_attr=hdm8_member.external_attr,
            create_system=hdm8_member.create_system,
            flag_bits=hdm8_member.flag_bits,
            comment=hdm8_member.comment,
            extra=hdm8_member.extra,
            archive_comment=hdm8_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x08"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["score_claim"] is False


def test_pr106_pr101_runtime_consumption_proves_hdm9_hlm2_inner_headerless_sidecar(
    tmp_path: Path,
) -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    inner_headerless_payload = _hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_member_payload()
    archive = tmp_path / "hdm9_hlm2_inner_headerless_fixed_meta_rank_elided.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(hdm8_member)(
            name=hdm8_member.name,
            payload=inner_headerless_payload,
            date_time=hdm8_member.date_time,
            external_attr=hdm8_member.external_attr,
            create_system=hdm8_member.create_system,
            flag_bits=hdm8_member.flag_bits,
            comment=hdm8_member.comment,
            extra=hdm8_member.extra,
            archive_comment=hdm8_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x09"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["score_claim"] is False


def test_pr106_pr101_runtime_consumption_proves_hdm9_hlm3_magicless_sidecar(
    tmp_path: Path,
) -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    magicless_payload = _hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_member_payload()
    archive = tmp_path / "hdm9_hlm3_magicless_fixed_meta_noop_rank_elided.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(hdm8_member)(
            name=hdm8_member.name,
            payload=magicless_payload,
            date_time=hdm8_member.date_time,
            external_attr=hdm8_member.external_attr,
            create_system=hdm8_member.create_system,
            flag_bits=hdm8_member.flag_bits,
            comment=hdm8_member.comment,
            extra=hdm8_member.extra,
            archive_comment=hdm8_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["format_id"] == "0x0B"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["inner_pr106_payload_sha256_unchanged"] is True
    assert manifest["score_claim"] is False


def test_pr106_pr101_runtime_accepts_hdm9_hlm3_magicless_exact_radix_format() -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    runtime = load_pr106_sidecar_runtime(PR106_R2_PR101_RUNTIME)
    exact_radix_payload = _hdm9_hlm3_magicless_exact_radix_member_payload()

    exact_radix_digest = runtime_sidecar_correction_digest(runtime, exact_radix_payload)
    headerless_digest = runtime_sidecar_correction_digest(runtime, hdm8_member.payload)

    assert exact_radix_digest["format_id"] == "0x0C"
    assert headerless_digest["format_id"] == "0x07"
    assert exact_radix_digest["combined_sha256"] == headerless_digest["combined_sha256"]
    assert exact_radix_digest["corrected_latents_sha256"] == headerless_digest[
        "corrected_latents_sha256"
    ]


def test_pr106_pr101_runtime_consumption_proves_hdm9_hlm3_magicless_exact_radix_sidecar(
    tmp_path: Path,
) -> None:
    hdm8_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_ARCHIVE.read_bytes())
    exact_radix_payload = _hdm9_hlm3_magicless_exact_radix_member_payload()
    archive = tmp_path / "hdm9_hlm3_magicless_exact_radix.zip"
    archive.write_bytes(
        emit_single_stored_member_archive(type(hdm8_member)(
            name="x",
            payload=exact_radix_payload,
            date_time=hdm8_member.date_time,
            external_attr=hdm8_member.external_attr,
            create_system=hdm8_member.create_system,
            flag_bits=hdm8_member.flag_bits,
            comment=hdm8_member.comment,
            extra=hdm8_member.extra,
            archive_comment=hdm8_member.archive_comment,
        ))
    )

    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
    )

    assert manifest["archive_member_name"] == "x"
    assert manifest["format_id"] == "0x0C"
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections["pr106_payload"] is True
    assert consumed_sections["sidecar_payload"] is True
    assert consumed_sections["framing_meta"] is None
    assert manifest["inner_pr106_payload_sha256_unchanged"] is True
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False


def test_pr106_runtime_consumption_proves_format0d_stacked_sidecars() -> None:
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_FORMAT0D_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256=PR106_FORMAT0D_SHA,
    )

    assert manifest["format_id"] == "0x0D"
    assert manifest["archive_member_name"] == "x"
    assert manifest["blockers"] == []
    assert manifest["runtime_sidecar_decode_consumption_claim"] is True
    assert manifest["runtime_sidecar_apply_consumption_claim"] is True
    assert manifest["runtime_all_score_affecting_sections_consumed"] is True
    consumed_sections = manifest["runtime_consumed_score_affecting_sections"]
    assert consumed_sections == {
        "pr106_payload": True,
        "base_format0c_sidecar_payload": True,
        "extra_pr101_ranked_no_op_payload": True,
        "extra_framing_meta": True,
    }
    assert manifest["runtime_apply_order"] == [
        "base_format0c_corrections",
        "extra_pr101_ranked_no_op_corrections",
    ]
    identities = manifest["runtime_consumed_score_affecting_section_identities"]
    assert isinstance(identities, list)
    assert [item["name"] for item in identities] == [
        "pr106_payload",
        "base_format0c_sidecar_payload",
        "extra_pr101_ranked_no_op_payload",
        "extra_framing_meta",
    ]
    source_sections = {
        row["name"]: row for row in manifest["source_packet_ir_consumed_byte_proof"]["sections"]
    }
    for identity in identities:
        source_section = source_sections[identity["name"]]
        assert identity["sha256"] == source_section["sha256"]
        assert identity["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        assert source_section["hash_domain"] == PR106_PACKET_IR_SECTION_HASH_DOMAIN
        assert identity["identity_source"] == (
            "packet_ir_consumed_byte_proof_filtered_by_runtime_probe"
        )
        assert identity["runtime_consumption_evidence"] == (
            "runtime_section_mutation_probe"
        )
        assert identity["bytes"] == source_section["bytes"]
        assert identity["consumed"] is True
    source_digest = manifest["source_runtime_correction_digest"]
    mutated_digest = manifest["mutated_runtime_correction_digest"]
    assert isinstance(source_digest, dict)
    assert isinstance(mutated_digest, dict)
    assert source_digest["n_passes"] == mutated_digest["n_passes"] == 2
    source_passes = source_digest["correction_passes"]
    assert isinstance(source_passes, list)
    assert [item["section_name"] for item in source_passes] == [
        "base_format0c_sidecar_payload",
        "extra_pr101_ranked_no_op_payload",
    ]
    probe = manifest["runtime_framing_meta_consumption_probe"]
    assert isinstance(probe, dict)
    assert probe["section"] == "extra_framing_meta"
    assert probe["runtime_consumption_claim"] is True
    section_probes = manifest["runtime_section_consumption_probes"]
    assert set(section_probes) == {
        "pr106_payload",
        "base_format0c_sidecar_payload",
        "extra_pr101_ranked_no_op_payload",
        "extra_framing_meta",
    }
    assert section_probes["base_format0c_sidecar_payload"][
        "runtime_consumption_claim"
    ] is True
    assert section_probes["extra_pr101_ranked_no_op_payload"][
        "runtime_consumption_claim"
    ] is True
    assert section_probes["base_format0c_sidecar_payload"]["mutation"][
        "section_name"
    ] == "base_format0c_sidecar_payload"
    assert section_probes["extra_pr101_ranked_no_op_payload"]["mutation"][
        "section_name"
    ] == "extra_pr101_ranked_no_op_payload"
    assert manifest["mutation"]["section_name"] == "extra_pr101_ranked_no_op_payload"
    assert manifest["sidecar_payload_sha256_changed"] is False
    assert manifest["extra_sidecar_payload_sha256_changed"] is True
    assert manifest["extra_framing_meta_sha256_changed"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False


def test_pr106_runtime_consumption_rejects_format0d_without_base_section_probe(
    monkeypatch,
) -> None:
    real_probe = runtime_consumption_mod.runtime_sidecar_section_consumption_probes

    def fake_probe(runtime_module, member_payload, baseline_digest):
        probes = real_probe(runtime_module, member_payload, baseline_digest)
        probes["base_format0c_sidecar_payload"] = {
            **probes["base_format0c_sidecar_payload"],
            "runtime_consumption_claim": False,
            "observation": "test_runtime_ignored_base_stream",
        }
        return probes

    monkeypatch.setattr(
        runtime_consumption_mod,
        "runtime_sidecar_section_consumption_probes",
        fake_probe,
    )

    manifest = runtime_consumption_mod.prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=PR106_FORMAT0D_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        expected_archive_sha256=PR106_FORMAT0D_SHA,
    )

    assert (
        "runtime_base_format0c_sidecar_payload_consumption_not_proven"
        in manifest["blockers"]
    )
    assert manifest["runtime_all_score_affecting_sections_consumed"] is False
    assert manifest["runtime_sidecar_decode_consumption_claim"] is False
    assert manifest["runtime_consumed_score_affecting_sections"][
        "base_format0c_sidecar_payload"
    ] is False
    assert manifest["runtime_consumed_score_affecting_sections"][
        "extra_pr101_ranked_no_op_payload"
    ] is True


class _ReturningSidecarDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        base_channels: int,
        eval_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.eval_size = eval_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        eval_h, eval_w = self.eval_size
        value = latents[:, :1].reshape(latents.shape[0], 1, 1, 1, 1)
        return value.expand(latents.shape[0], 2, 3, eval_h, eval_w)


class _ReturningSidecarRuntime:
    torch = torch
    F = torch.nn.functional
    HNeRVDecoder = _ReturningSidecarDecoder
    CAMERA_H = 2
    CAMERA_W = 2
    DEFAULT_BATCH_PAIRS = 1

    def __init__(self) -> None:
        self.corrections_called = False

    def parse_sidecar_archive(self, payload: bytes) -> tuple[bytes, bytes]:
        return b"inner-pr106", payload

    def decode_sidecar_corrections(self, blob: bytes) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(blob, bytes)
        return np.array([0], dtype=np.uint8), np.array([1], dtype=np.int8)

    def parse_packed_archive(
        self,
        payload: bytes,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, object]]:
        assert isinstance(payload, bytes)
        return (
            {},
            torch.zeros((1, 1), dtype=torch.float32),
            {
                "latent_dim": 1,
                "base_channels": 1,
                "eval_size": (2, 2),
                "n_pairs": 1,
            },
        )

    def apply_sidecar_corrections(
        self,
        latents: torch.Tensor,
        dim_arr: np.ndarray,
        delta_q_arr: np.ndarray,
    ) -> torch.Tensor:
        self.corrections_called = True
        assert int(dim_arr[0]) == 0
        corrected = latents.clone()
        corrected[:, 0] += float(delta_q_arr[0])
        return corrected


def test_pr106_streaming_digest_consumes_returned_sidecar_latents() -> None:
    runtime = _ReturningSidecarRuntime()

    digest = runtime_full_frame_streaming_digest(
        runtime,
        b"fake-sidecar",
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert runtime.corrections_called is True
    assert digest["source_latents_sha256"] != digest["corrected_latents_sha256"]
    assert digest["latents_changed_by_sidecar"] is True
    assert digest["total_frames"] == 2
    assert digest["total_bytes"] == 24
    assert digest["score_claim"] is False


def test_pr106_same_runtime_streaming_prefix_parity_is_nonpromotable() -> None:
    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=PR106_R2_ARCHIVE,
        candidate_archive_path=PR106_R2_PR101_ARCHIVE,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert manifest["schema"] == "pr106_same_runtime_streaming_frame_parity_v1"
    assert manifest["proof_scope"] == "same_runtime_streaming_prefix_hash"
    assert manifest["streaming_output_sha256_equal"] is True
    assert manifest["streaming_output_total_bytes_equal"] is True
    assert manifest["prefix_parity_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is False
    assert manifest["contest_axis_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    source = manifest["source"]
    candidate = manifest["candidate"]
    assert isinstance(source, dict)
    assert isinstance(candidate, dict)
    assert source["full_frame_digest"] is False
    assert candidate["full_frame_digest"] is False
    assert source["total_frames"] == candidate["total_frames"] == 2
    assert source["streaming_raw_sha256"] == candidate["streaming_raw_sha256"]


def test_pr106_same_runtime_streaming_prefix_autodetects_x_members(
    tmp_path: Path,
) -> None:
    source_member = read_single_stored_member_archive(PR106_R2_ARCHIVE.read_bytes())
    candidate_member = read_single_stored_member_archive(PR106_R2_PR101_ARCHIVE.read_bytes())
    source_archive = tmp_path / "source_x.zip"
    candidate_archive = tmp_path / "candidate_x.zip"
    source_archive.write_bytes(
        emit_single_stored_member_archive(type(source_member)(
            name="x",
            payload=source_member.payload,
            date_time=source_member.date_time,
            external_attr=source_member.external_attr,
            create_system=source_member.create_system,
            flag_bits=source_member.flag_bits,
            comment=source_member.comment,
            extra=source_member.extra,
        ))
    )
    candidate_archive.write_bytes(
        emit_single_stored_member_archive(type(candidate_member)(
            name="x",
            payload=candidate_member.payload,
            date_time=candidate_member.date_time,
            external_attr=candidate_member.external_attr,
            create_system=candidate_member.create_system,
            flag_bits=candidate_member.flag_bits,
            comment=candidate_member.comment,
            extra=candidate_member.extra,
        ))
    )

    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=source_archive,
        candidate_archive_path=candidate_archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert manifest["source_archive"]["member_name"] == "x"
    assert manifest["candidate_archive"]["member_name"] == "x"
    assert manifest["prefix_parity_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is False


def test_pr106_same_runtime_streaming_prefix_detects_semantic_sidecar_change(
    tmp_path: Path,
) -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    mutated_packet, _ = mutate_pr106_sidecar_semantic_correction(packet, pair_index=0)
    mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
    mutated_archive = tmp_path / "mutated.zip"
    mutated_archive.write_bytes(
        emit_single_stored_member_archive(type(member)(
            name=member.name,
            payload=mutated_payload,
            date_time=member.date_time,
            external_attr=member.external_attr,
            create_system=member.create_system,
            flag_bits=member.flag_bits,
            comment=member.comment,
            extra=member.extra,
        ))
    )
    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=PR106_R2_PR101_ARCHIVE,
        candidate_archive_path=mutated_archive,
        runtime_dir=PR106_R2_PR101_RUNTIME,
        device="cpu",
        batch_pairs=1,
        max_pairs=1,
    )

    assert manifest["streaming_output_sha256_equal"] is False
    assert manifest["streaming_output_total_bytes_equal"] is True
    assert manifest["prefix_parity_claim"] is False
    assert manifest["full_frame_inflate_output_parity_claim"] is False
