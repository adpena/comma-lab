from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import brotli
import numpy as np
import pytest

from tac.hnerv_lowlevel_packer import read_packed_archive_view
from tac.hnerv_hlm1_archive_candidate import build_hlm1_latent_archive_candidate
from tac.packet_compiler.pr106_fixed_latent_recode import (
    HLM1_MAGIC,
    PR106FixedLatentRecodeError,
    decode_pr106_fixed_latent_raw,
    encode_hlm1_fixed_latents_from_brotli,
    split_pr106_fixed_latent_raw,
)
from tac.packet_compiler.pr106_hlm1_runtime_consumption import (
    prove_pr106_hlm1_runtime_consumption,
)

REPO = Path(__file__).resolve().parents[3]
PR106_R2_PR101_ARCHIVE = (
    REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar" / "archive.zip"
)
PR106_R2_PR101_CODEC = (
    REPO / "submissions" / "pr106_latent_sidecar_r2_pr101_grammar" / "src" / "codec.py"
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
