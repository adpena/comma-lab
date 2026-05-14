# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import brotli
import numpy as np
import pytest

from tac.hnerv_hlm1_archive_candidate import build_hlm1_latent_archive_candidate
from tac.hnerv_lowlevel_packer import read_packed_archive_view
from tac.packet_compiler.pr106_fixed_latent_recode import (
    HLM1_MAGIC,
    HLM2_MAGIC,
    HLM3_MAGIC,
    PR106_FIXED_LATENT_D,
    PR106_FIXED_LATENT_TOTAL,
    PR106FixedLatentRecodeError,
    decode_hlm3_fixed_latent_raw,
    decode_pr106_fixed_latent_raw,
    encode_hlm1_fixed_latents_from_brotli,
    encode_hlm2_fixed_latents_from_brotli,
    encode_hlm3_fixed_latents_from_brotli,
    split_pr106_fixed_latent_raw,
)
from tac.packet_compiler.pr106_hlm1_runtime_consumption import (
    prove_pr106_hlm1_runtime_consumption,
    prove_pr106_hlm_runtime_consumption,
)
from tac.packet_compiler.pr106_runtime_consumption import load_pr106_runtime_codec

REPO = Path(__file__).resolve().parents[3]
PR106_R2_PR101_ARCHIVE = (
    REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar" / "archive.zip"
)
PR106_R2_HDM4_HLM1_ARCHIVE = (
    REPO
    / "experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/"
    "pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip"
)
PR106_R2_PR101_CODEC = (
    REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar" / "src" / "codec.py"
)
PR106_R2_PR101_INFLATE_SH = (
    REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar" / "inflate.sh"
)


def test_hlm1_recode_roundtrips_tracked_pr106_fixed_latents() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    source = view.packed.latents_and_sidecar_brotli
    source_raw = brotli.decompress(source)

    recode = encode_hlm1_fixed_latents_from_brotli(source)

    assert recode.payload.startswith(HLM1_MAGIC)
    assert recode.rate_positive is True
    assert recode.byte_delta < 0
    assert recode.hi_nonzero_symbols == 2
    assert recode.hi_nonzero_symbol_count == 73
    assert decode_pr106_fixed_latent_raw(recode.payload) == source_raw
    assert recode.to_manifest()["score_claim"] is False
    assert recode.to_manifest()["raw_roundtrip_equal"] is True


def test_hlm1_recode_makes_byte_smaller_archive_without_touching_decoder(
    tmp_path: Path,
) -> None:
    source_view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    recode = encode_hlm1_fixed_latents_from_brotli(
        source_view.packed.latents_and_sidecar_brotli
    )
    candidate_packed = dataclasses.replace(
        source_view.packed,
        latents_and_sidecar_brotli=recode.payload,
    )
    candidate_payload = source_view.emit_payload(candidate_packed)
    candidate_archive = tmp_path / "candidate.zip"
    source_view.write_archive(candidate_archive, candidate_payload)

    candidate_view = read_packed_archive_view(candidate_archive)

    assert candidate_archive.stat().st_size < PR106_R2_PR101_ARCHIVE.stat().st_size
    assert candidate_view.packed.decoder_packed_brotli == source_view.packed.decoder_packed_brotli
    assert candidate_view.packed.latents_and_sidecar_brotli.startswith(HLM1_MAGIC)
    assert (
        decode_pr106_fixed_latent_raw(candidate_view.packed.latents_and_sidecar_brotli)
        == brotli.decompress(source_view.packed.latents_and_sidecar_brotli)
    )


def test_hlm2_recode_derives_hi_metadata_and_saves_four_bytes_over_hlm1() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    source = view.packed.latents_and_sidecar_brotli
    source_raw = brotli.decompress(source)

    hlm1 = encode_hlm1_fixed_latents_from_brotli(source)
    hlm2 = encode_hlm2_fixed_latents_from_brotli(source)

    assert hlm2.payload.startswith(HLM2_MAGIC)
    assert hlm2.codec == "pr106_fixed_latent_hlm2"
    assert hlm2.packet_overhead_bytes == 6
    assert hlm2.candidate_bytes == hlm1.candidate_bytes - 4
    assert hlm2.byte_delta == hlm1.byte_delta - 4
    assert hlm2.hi_nonzero_symbol_count == hlm1.hi_nonzero_symbol_count == 73
    assert decode_pr106_fixed_latent_raw(hlm2.payload) == source_raw
    assert hlm2.to_manifest()["raw_roundtrip_equal"] is True


def test_hlm3_range_recode_roundtrips_tracked_pr106_fixed_latents() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    source = view.packed.latents_and_sidecar_brotli
    source_raw = brotli.decompress(source)

    recode = encode_hlm3_fixed_latents_from_brotli(source)
    manifest = recode.to_manifest()

    assert recode.payload.startswith(HLM3_MAGIC)
    assert recode.codec == "pr106_fixed_latent_hlm3"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["raw_roundtrip_equal"] is True
    assert manifest["hi_codec"] == "pr103_range_coded_hi_bytes"
    assert manifest["hi_arithmetic_bytes"] > 0
    assert manifest["hi_histogram_bytes"] == 256
    assert decode_hlm3_fixed_latent_raw(recode.payload) == source_raw
    assert decode_pr106_fixed_latent_raw(recode.payload) == source_raw


def test_hlm3_range_recode_accepts_dense_non_binary_hi_stream() -> None:
    lo = np.arange(PR106_FIXED_LATENT_TOTAL, dtype=np.uint16).astype(np.uint8).tobytes()
    meta = np.zeros(PR106_FIXED_LATENT_D * 2, dtype=np.float16).tobytes()
    hi = (np.arange(PR106_FIXED_LATENT_TOTAL, dtype=np.uint16) % 17).astype(np.uint8).tobytes()
    source = brotli.compress(lo + meta + hi, quality=5)

    recode = encode_hlm3_fixed_latents_from_brotli(
        source,
        brotli_candidates=((5, 16),),
    )

    assert recode.payload.startswith(HLM3_MAGIC)
    assert recode.hi_codec == "pr103_range_coded_hi_bytes"
    assert recode.hi_nonzero_symbols == 17
    assert recode.hi_nonzero_symbol_count > 0
    assert decode_hlm3_fixed_latent_raw(recode.payload) == lo + meta + hi
    with pytest.raises(PR106FixedLatentRecodeError, match="binary hi symbols"):
        encode_hlm1_fixed_latents_from_brotli(source, brotli_candidates=((5, 16),))


def test_hlm2_archive_candidate_builder_recodes_existing_hlm1_source(
    tmp_path: Path,
) -> None:
    source_view = read_packed_archive_view(PR106_R2_HDM4_HLM1_ARCHIVE)

    manifest = build_hlm1_latent_archive_candidate(
        source_archive=PR106_R2_HDM4_HLM1_ARCHIVE,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 HDM4 HLM1",
        latent_codec="hlm2",
        repo_root=REPO,
    )

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)

    assert manifest["score_claim"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["source_latents_section_codec"] == "hlm1"
    assert manifest["target_latents_section_codec"] == "hlm2"
    assert manifest["candidate_transform_kind"] == "hlm2_fixed_latent_recode"
    assert manifest["candidate_latents_section_byte_delta"] == -4
    assert manifest["candidate_archive_byte_delta"] == -4
    assert candidate_archive.stat().st_size == PR106_R2_HDM4_HLM1_ARCHIVE.stat().st_size - 4
    assert candidate_view.packed.decoder_packed_brotli == source_view.packed.decoder_packed_brotli
    assert candidate_view.packed.latents_and_sidecar_brotli.startswith(HLM2_MAGIC)
    assert (
        decode_pr106_fixed_latent_raw(candidate_view.packed.latents_and_sidecar_brotli)
        == decode_pr106_fixed_latent_raw(source_view.packed.latents_and_sidecar_brotli)
    )
    assert (tmp_path / "out" / "hlm2_latent_archive_candidate_manifest.json").exists()


def test_hlm3_archive_candidate_builder_is_byte_closed_but_fail_closed(
    tmp_path: Path,
) -> None:
    source_view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)

    manifest = build_hlm1_latent_archive_candidate(
        source_archive=PR106_R2_PR101_ARCHIVE,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 PR101 grammar",
        latent_codec="hlm3",
        allow_rate_regression=True,
        repo_root=REPO,
    )

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["lane_id"] == "hnerv_hlm3_pr103_range_hi_probe"
    assert manifest["target_latents_section_codec"] == "hlm3"
    assert manifest["candidate_transform_kind"] == "hlm3_fixed_latent_recode"
    assert manifest["decoder_section_preserved"] is True
    assert manifest["hlm3_recode"]["raw_roundtrip_equal"] is True
    assert manifest["hlm3_recode"]["hi_codec"] == "pr103_range_coded_hi_bytes"
    assert "hlm3_inflate_runtime_decoder_missing" in manifest["archive_build_blockers"]
    assert candidate_view.packed.decoder_packed_brotli == source_view.packed.decoder_packed_brotli
    assert candidate_view.packed.latents_and_sidecar_brotli.startswith(HLM3_MAGIC)
    assert (
        decode_pr106_fixed_latent_raw(candidate_view.packed.latents_and_sidecar_brotli)
        == brotli.decompress(source_view.packed.latents_and_sidecar_brotli)
    )
    assert (tmp_path / "out" / "hlm3_latent_archive_candidate_manifest.json").exists()


def test_hlm1_archive_candidate_builder_writes_nonpromotable_manifest(
    tmp_path: Path,
) -> None:
    manifest = build_hlm1_latent_archive_candidate(
        source_archive=PR106_R2_PR101_ARCHIVE,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 PR101 grammar",
        repo_root=REPO,
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_latents_section_byte_delta"] < 0
    assert manifest["candidate_archive_byte_delta"] < 0
    assert manifest["decoder_section_preserved"] is True
    assert manifest["hlm1_recode"]["raw_roundtrip_equal"] is True
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]
    assert (tmp_path / "out" / "hlm1_latent_archive_candidate_manifest.json").exists()


def test_hlm1_archive_candidate_builder_can_emit_x_member_rate_repack(
    tmp_path: Path,
) -> None:
    source_view = read_packed_archive_view(PR106_R2_HDM4_HLM1_ARCHIVE)

    manifest = build_hlm1_latent_archive_candidate(
        source_archive=PR106_R2_HDM4_HLM1_ARCHIVE,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 HDM4 HLM1",
        candidate_member_name="x",
        repo_root=REPO,
    )

    candidate_archive = Path(manifest["candidate_archive_path"])
    candidate_view = read_packed_archive_view(candidate_archive)

    assert manifest["score_claim"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_transform_kind"] == "zip_member_rate_only_repack"
    assert manifest["source_member_name"] == "0.bin"
    assert manifest["candidate_member_name"] == "x"
    assert manifest["candidate_member_name_changed"] is True
    assert manifest["member_payload_sha256_unchanged"] is True
    assert manifest["member_payload_bytes_unchanged"] is True
    assert manifest["candidate_latents_section_byte_delta"] == 0
    assert manifest["candidate_archive_byte_delta"] == -8
    assert candidate_archive.stat().st_size == PR106_R2_HDM4_HLM1_ARCHIVE.stat().st_size - 8
    assert candidate_view.archive.member_name == "x"
    assert candidate_view.archive.payload == source_view.archive.payload


def test_hlm1_archive_candidate_cli_accepts_x_member_name(tmp_path: Path) -> None:
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_pr106_hlm1_latent_candidate.py"),
            "--source-archive",
            str(PR106_R2_HDM4_HLM1_ARCHIVE),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106 R2 HDM4 HLM1",
            "--candidate-member-name",
            "x",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready_for_archive_preflight"] is True
    assert payload["candidate_member_name"] == "x"
    assert payload["candidate_archive_byte_delta"] == -8


def test_hlm_archive_candidate_cli_accepts_hlm2_codec(tmp_path: Path) -> None:
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_pr106_hlm1_latent_candidate.py"),
            "--source-archive",
            str(PR106_R2_HDM4_HLM1_ARCHIVE),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106 R2 HDM4 HLM1",
            "--latent-codec",
            "hlm2",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready_for_archive_preflight"] is True
    assert payload["target_latents_section_codec"] == "hlm2"
    assert payload["candidate_archive_byte_delta"] == -4


def test_hlm_archive_candidate_cli_accepts_hlm3_probe_and_fails_closed(
    tmp_path: Path,
) -> None:
    json_out = tmp_path / "result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_pr106_hlm1_latent_candidate.py"),
            "--source-archive",
            str(PR106_R2_PR101_ARCHIVE),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106 R2 PR101 grammar",
            "--latent-codec",
            "hlm3",
            "--allow-rate-regression",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=False,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert result.returncode == 2
    assert payload["target_latents_section_codec"] == "hlm3"
    assert payload["ready_for_archive_preflight"] is False
    assert payload["score_claim"] is False
    assert "hlm3_inflate_runtime_decoder_missing" in payload["archive_build_blockers"]


def test_pr106_inflate_sh_uses_x_payload_and_refuses_ambiguous_payloads(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "src").mkdir()
    shutil.copy2(PR106_R2_PR101_INFLATE_SH, runtime / "inflate.sh")
    (runtime / "inflate.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_text(Path(sys.argv[1]).name, encoding='utf-8')\n",
        encoding="utf-8",
    )
    (runtime / "inflate.sh").chmod(0o755)
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0.mp4\n", encoding="utf-8")
    (data / "x").write_bytes(b"payload")
    env = dict(os.environ)
    env["PYTHON_BIN"] = sys.executable

    subprocess.run(
        [str(runtime / "inflate.sh"), str(data), str(out), str(file_list)],
        check=True,
        text=True,
        env=env,
    )
    assert (out / "0.raw").read_text(encoding="utf-8") == "x"

    (data / "0.bin").write_bytes(b"payload")
    ambiguous = subprocess.run(
        [str(runtime / "inflate.sh"), str(data), str(out), str(file_list)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert ambiguous.returncode != 0
    assert "ambiguous archive payloads" in ambiguous.stderr


def test_hlm1_runtime_consumption_proof_is_specific_to_fixed_latents(
    tmp_path: Path,
) -> None:
    manifest = build_hlm1_latent_archive_candidate(
        source_archive=PR106_R2_PR101_ARCHIVE,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 PR101 grammar",
        repo_root=REPO,
    )

    proof = prove_pr106_hlm1_runtime_consumption(
        archive_path=Path(manifest["candidate_archive_path"]),
        runtime_dir=REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar",
        repo_root=REPO,
    )

    assert proof["runtime_hlm1_decode_consumption_claim"] is True
    assert proof["runtime_hlm1_decode_matches_canonical"] is True
    assert proof["runtime_hlm1_valid_mutation_changes_raw"] is True
    assert proof["score_claim"] is False
    assert proof["full_frame_inflate_output_parity_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_hlm1_archive_candidate_cli_writes_manifest(tmp_path: Path) -> None:
    json_out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_pr106_hlm1_latent_candidate.py"),
            "--source-archive",
            str(PR106_R2_PR101_ARCHIVE),
            "--output-dir",
            str(tmp_path / "out"),
            "--source-label",
            "PR106 R2 PR101 grammar",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready_for_archive_preflight"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert Path(payload["candidate_archive_path"]).exists()


def test_submission_runtime_decodes_hlm1_fixed_latents() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    recode = encode_hlm1_fixed_latents_from_brotli(view.packed.latents_and_sidecar_brotli)
    runtime = _load_runtime_codec()

    assert runtime.decode_fixed_latents_raw(recode.payload) == brotli.decompress(
        view.packed.latents_and_sidecar_brotli
    )


def test_submission_runtime_decodes_hlm2_fixed_latents() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    recode = encode_hlm2_fixed_latents_from_brotli(view.packed.latents_and_sidecar_brotli)
    runtime = _load_runtime_codec()

    assert runtime.decode_fixed_latents_raw(recode.payload) == brotli.decompress(
        view.packed.latents_and_sidecar_brotli
    )


def test_hlm2_runtime_consumption_proof_is_codec_specific(tmp_path: Path) -> None:
    manifest = build_hlm1_latent_archive_candidate(
        source_archive=PR106_R2_HDM4_HLM1_ARCHIVE,
        output_dir=tmp_path / "out",
        source_label="PR106 R2 HDM4 HLM1",
        latent_codec="hlm2",
        repo_root=REPO,
    )

    proof = prove_pr106_hlm_runtime_consumption(
        archive_path=Path(manifest["candidate_archive_path"]),
        runtime_dir=REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar",
        repo_root=REPO,
    )

    assert proof["latent_section_codec"] == "hlm2"
    assert proof["runtime_hlm_decode_consumption_claim"] is True
    assert proof["runtime_hlm_decode_matches_canonical"] is True
    assert proof["runtime_hlm_valid_mutation_changes_raw"] is True
    assert proof["score_claim"] is False
    assert proof["full_frame_inflate_output_parity_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_runtime_consumption_loader_does_not_write_pycache(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    shutil.copytree(REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar", runtime)
    for cache_dir in runtime.rglob("__pycache__"):
        shutil.rmtree(cache_dir)

    module = load_pr106_runtime_codec(runtime)

    assert hasattr(module, "decode_fixed_latents_raw")
    assert not list(runtime.rglob("__pycache__"))
    assert not list(runtime.rglob("*.pyc"))


def test_hlm1_recode_rejects_non_binary_hi_stream() -> None:
    view = read_packed_archive_view(PR106_R2_PR101_ARCHIVE)
    raw = bytearray(brotli.decompress(view.packed.latents_and_sidecar_brotli))
    lo, meta, hi = split_pr106_fixed_latent_raw(bytes(raw))
    hi_mut = bytearray(hi)
    hi_mut[0] = 2
    malformed = brotli.compress(lo + meta + bytes(hi_mut), quality=5)

    with pytest.raises(PR106FixedLatentRecodeError, match="binary hi symbols"):
        encode_hlm1_fixed_latents_from_brotli(malformed)


def _load_runtime_codec():
    spec = importlib.util.spec_from_file_location(
        "pr106_pr101_runtime_codec",
        PR106_R2_PR101_CODEC,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
