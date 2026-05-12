"""Tests for ``compile_phase1_packet(packet_compiler_transforms=...)``.

This is the integration surface added by the PR101/PR103 reusable primitives
landing. The Phase 1 compiler does NOT mutate archive bytes itself; this flag
only records the upstream-trainer-applied transforms in
``build_manifest.json::packet_compiler_transforms`` for downstream audit and
future native-port routing. Default behaviour (flag absent) is unchanged.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tac.phase1_packet_compiler import (
    PACKET_COMPILER_TRANSFORMS,
    Phase1PacketCompilerError,
    compile_phase1_packet,
)
from tac.repo_io import sha256_file

# ── Minimal packet helpers (mirrors _write_synthetic_packet) ────────────────


def _write_minimal_inflate_sh(packet_dir: Path) -> None:
    text = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'DATA_DIR="${1:?missing $1 archive_dir}"\n'
        'OUTPUT_DIR="${2:?missing $2 output_dir}"\n'
        'FILE_LIST="${3:?missing $3 file_list}"\n'
        'exec uv run --with brotli --with torch '
        '"$(dirname "$0")/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )
    p = packet_dir / "inflate.sh"
    p.write_text(text, encoding="utf-8")
    p.chmod(0o755)


def _write_minimal_inflate_py(packet_dir: Path) -> None:
    text = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "archive_dir = Path(sys.argv[1])\n"
        "output_dir = Path(sys.argv[2])\n"
        "file_list = Path(sys.argv[3])\n"
        "x_bytes = (archive_dir / 'x').read_bytes()\n"
        "for line in file_list.read_text().splitlines():\n"
        "    name = line.strip()\n"
        "    if not name:\n"
        "        continue\n"
        "    dst = output_dir / (name.rsplit('.', 1)[0] + '.raw')\n"
        "    dst.write_bytes(x_bytes)\n"
    )
    (packet_dir / "inflate.py").write_text(text, encoding="utf-8")


def _write_synthetic_archive(
    packet_dir: Path, *, payload: bytes = b"PR101_PR103_TRANSFORM_PAYLOAD"
) -> Path:
    archive_path = packet_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(filename="x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        zf.writestr(info, payload)
    return archive_path


def _write_synthetic_packet(tmp_path: Path) -> Path:
    packet_dir = tmp_path / "syn"
    packet_dir.mkdir()
    (packet_dir / "src").mkdir()
    (packet_dir / "src" / "codec.py").write_text("# stub\n", encoding="utf-8")
    _write_synthetic_archive(packet_dir)
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)
    return packet_dir


def _write_transform_proof(packet_dir: Path, transforms: list[str]) -> None:
    archive_sha = sha256_file(packet_dir / "archive.zip")
    proof = {
        "schema": "tac.packet_compiler.transform_proof.v1",
        "proof_kind": "upstream_trainer_materialized_bytes",
        "packet_compiler_transforms": transforms,
        "archive_sha256": archive_sha,
        "score_claim": False,
        "runtime_consumption_proof": True,
        "producer": "test_phase1_packet_compiler_packet_compiler_transforms",
        "transform_evidence": [
            {
                "transform": transform,
                "target_member": "x",
                "input_sha256": f"{index + 1:064x}",
                "output_sha256": archive_sha,
                "byte_delta": -(index + 1),
                "changed_bytes_count": index + 1,
            }
            for index, transform in enumerate(transforms)
        ],
    }
    (packet_dir / "packet_compiler_transform_proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8"
    )


# ── Default behaviour preserved ─────────────────────────────────────────────


def test_default_no_transforms_records_empty_list(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_sha = sha256_file(packet_dir / "archive.zip")
    out_dir = tmp_path / "out_default"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="optimize",
        score_affecting_payload_changed=True,
        baseline_archive_sha256="0" * 64,  # forced delta-asserter
        baseline_archive_size_bytes=999_999,
    )
    # Read the manifest the compiler emitted.
    manifest = json.loads((out_dir / "build_manifest.json").read_text(encoding="utf-8"))
    assert "packet_compiler_transforms" in manifest
    assert manifest["packet_compiler_transforms"] == []
    # Default behaviour: bytes copied through (per Phase 1 compiler's
    # "trainer owns bytes" contract); SHA differs from baseline by design.
    assert result.archive_sha256 == pre_sha


def test_records_declared_transforms_in_manifest(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    transforms = [
        "pr101_ranked_no_op_sidecar",
        "pr103_merged_range_stream",
        "pr103_adaptive_brotli_param_search",
    ]
    _write_transform_proof(packet_dir, transforms)
    out_dir = tmp_path / "out_with_transforms"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="optimize",
        score_affecting_payload_changed=True,
        baseline_archive_sha256="0" * 64,
        baseline_archive_size_bytes=999_999,
        packet_compiler_transforms=transforms,
    )
    manifest = json.loads((out_dir / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["packet_compiler_transforms"] == transforms
    assert manifest["packet_compiler_transform_proof"]["status"] == "materialization_proof_present"
    assert manifest["packet_compiler_transform_proof"]["score_claim"] is False
    assert result.score_claim is False
    assert result.promotion_eligible is False


# ── Validation ──────────────────────────────────────────────────────────────


def test_rejects_unknown_transform_token(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out_bad"
    with pytest.raises(Phase1PacketCompilerError, match="unknown packet_compiler_transforms"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=999_999,
            packet_compiler_transforms=["pr999_not_a_real_transform"],
        )


def test_rejects_declared_transforms_without_materialization_proof(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out_missing_proof"
    with pytest.raises(Phase1PacketCompilerError, match="materialization proof"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=999_999,
            packet_compiler_transforms=["pr101_ranked_no_op_sidecar"],
        )


def test_rejects_transform_proof_archive_sha_mismatch(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    _write_transform_proof(packet_dir, ["pr101_ranked_no_op_sidecar"])
    proof_path = packet_dir / "packet_compiler_transform_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["archive_sha256"] = "f" * 64
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    out_dir = tmp_path / "out_bad_proof"
    with pytest.raises(Phase1PacketCompilerError, match="archive_sha256"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=999_999,
            packet_compiler_transforms=["pr101_ranked_no_op_sidecar"],
        )


def test_rejects_transform_proof_without_runtime_consumption(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    _write_transform_proof(packet_dir, ["pr101_ranked_no_op_sidecar"])
    proof_path = packet_dir / "packet_compiler_transform_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["runtime_consumption_proof"] = False
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    out_dir = tmp_path / "out_unconsumed_proof"
    with pytest.raises(Phase1PacketCompilerError, match="runtime_consumption_proof=true"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=999_999,
            packet_compiler_transforms=["pr101_ranked_no_op_sidecar"],
        )


def test_rejects_transform_proof_without_per_transform_evidence(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    _write_transform_proof(packet_dir, ["pr101_ranked_no_op_sidecar"])
    proof_path = packet_dir / "packet_compiler_transform_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("transform_evidence")
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    out_dir = tmp_path / "out_missing_transform_evidence"
    with pytest.raises(Phase1PacketCompilerError, match="transform_evidence"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=999_999,
            packet_compiler_transforms=["pr101_ranked_no_op_sidecar"],
        )


def test_rejects_transforms_in_identity_mode(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out_identity"
    with pytest.raises(Phase1PacketCompilerError, match="only be declared in optimize"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
            packet_compiler_transforms=["pr101_ranked_no_op_sidecar"],
        )


def test_rejects_transforms_in_canonicalize_mode(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out_canon"
    with pytest.raises(Phase1PacketCompilerError, match="only be declared in optimize"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="canonicalize",
            packet_compiler_transforms=["pr101_ranked_no_op_sidecar"],
        )


def test_known_transform_tokens_match_packet_compiler_module() -> None:
    """The declared token set must cover every public packet_compiler primitive."""
    expected = {
        # PR101 — sidecar grammar
        "pr101_ranked_no_op_sidecar",
        "pr101_centered_delta_uint8_lzma",
        "pr101_split_brotli_self_delimiting",
        # PR103 — arithmetic coding
        "pr103_merged_range_stream",
        "pr103_latent_hi_arithmetic",
        "pr103_adaptive_brotli_param_search",
        # PR81 — Quantizr FP4 codebook + ROUTER_ACTION (2026-05-11)
        "pr81_fp4_codebook",
        "pr81_router_action",
        # PR84 — adaptive-context range coder (2026-05-11)
        "pr84_adaptive_mask_context",
        # PR91 — universal AC wrapper + QMQH grammar (2026-05-11)
        "pr91_arithmetic_coder_constriction",
        "pr91_qmqh_grammar",
        # PR92 — RMC1 joint-stream meta-codec (2026-05-11)
        "pr92_rmc_joint_stream",
        # PR93 — delta-varint pose + QZMB1 (2026-05-11)
        "pr93_delta_varint_pose",
        "pr93_qzmb_qzpdv_grammar",
        # PR63 — qpose14 uint16-view int16 + single-zip-member packed payload (2026-05-12)
        "pr63_qpose14_uint16_view_int16",
        "pr63_qpose14_packed_payload",
        # PR64 — unified-brotli pose-velocity-only codec (2026-05-12)
        "pr64_unified_brotli_pose_velocity",
        # PR65 — PQ12 12-bit / 3-byte / 2-value packed pose codec (2026-05-12)
        "pr65_pq12_pose",
        # PR105 — kitchen_sink packed-state-schema size-sorted helper (2026-05-12)
        "pr105_packed_state_schema_size_sorted",
        # PR101 GOLD — 3 newly-ported primitives (2026-05-12)
        "pr101_decoder_storage_order",
        "pr101_conv4_storage_perms",
        "pr101_decoder_byte_maps",
        # PR93 — lowpass-luma residual codec (2026-05-11 punchlist cleanup)
        "pr93_lowpass_luma_residual",
        # PR97 — H3 wire-format grammar (2026-05-11 punchlist cleanup)
        "pr97_length_prefixed_sections",
        "pr97_tile_band_streams",
        # Sparse PacketIR codec — closes O's L2 wire-format ceiling (2026-05-11)
        "sparse_rle_of_zeros",
        "sparse_arithmetic_coefficients",
        "sparse_temporal_subsampled",
        # Magic codec — per-stream auto-selector + meta-codec dispatch (2026-05-11)
        "magic_codec_auto_select",
        # Magic codec dense streams — per-stream brotli/lzma/magic_classic bundle (2026-05-12)
        "magic_codec_dense_streams",
        # Sign-encoding 5-strategy unified taxonomy (2026-05-12)
        "sign_encode_negzig",
        "sign_encode_zig",
        "sign_encode_twos",
        "sign_encode_off",
        "sign_encode_raw_uint8",
        # Schema-elision V1+V2 (2026-05-12)
        "pr98_cd1_compact_architecture_ordered_decoder_format",
        "pr100_schema_driven_decoder_storage_grammar",
        # CompressAI reference adapters (2026-05-12)
        "compressai_factorized_prior",
        "compressai_balle_hyperprior",
        "compressai_cheng2020",
    }
    assert set(PACKET_COMPILER_TRANSFORMS) == expected


# ── Identity mode unchanged ─────────────────────────────────────────────────


def test_identity_mode_still_byte_identical_after_integration(tmp_path: Path) -> None:
    """Regression guard: identity mode must remain byte-for-byte after the
    packet_compiler_transforms parameter was added.
    """
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_sha = sha256_file(packet_dir / "archive.zip")
    out_dir = tmp_path / "out_identity_unchanged"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    post_sha = sha256_file(out_dir / "archive.zip")
    assert result.archive_sha256 == pre_sha
    assert post_sha == pre_sha
    manifest = json.loads((out_dir / "build_manifest.json").read_text(encoding="utf-8"))
    # The new key may be absent (identity mode never sets it) but if present it must be empty.
    assert manifest.get("packet_compiler_transforms", []) == []
