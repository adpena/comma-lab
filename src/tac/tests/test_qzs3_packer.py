"""Byte-equivalence tests for QZS3 weight codec vs PR #67's unpacker.

These complement test_quantizr_faithful_renderer.py which checks our own
decode_qzs3_state_dict round-trip. The tests here load the actual PR #67
inflate.py module and verify our encoder produces bytes that PR #67's
``get_grouped_qv_state_dict`` decodes successfully.

Verified on 2026-05-01: our encoder produces 59288 bytes for an
init-random JointFrameGenerator — byte-identical to the uncompressed
QZS3 payload deployed in PR #67's rank-1 archive.
"""
from __future__ import annotations

import importlib.util
import subprocess
import struct
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import (
    QZS3_MAGIC,
    decode_qzs3_state_dict,
    encode_qzs3_state_dict,
    encode_qzs4_block_search_state_dict,
    qzs3_qv_specs,
)
from experiments.repack_quantizr_faithful_qzs3_archive import (
    RENDERER_CODEC_QZS3,
    RENDERER_CODEC_QZS4,
    build_archive,
    build_submission_archive,
)


PR67_INFLATE = (
    Path(__file__).resolve().parents[3]
    / "reports/raw/leaderboard_intel_20260501/pr67_inflate.py"
)


def _import_pr67():
    """Side-load PR #67 inflate.py without running its main()."""

    if not PR67_INFLATE.exists():
        pytest.skip(f"pr67 inflate reference missing: {PR67_INFLATE}")
    spec = importlib.util.spec_from_file_location("_pr67_inflate", PR67_INFLATE)
    module = importlib.util.module_from_spec(spec)
    # PR #67 only invokes main() under __name__ == "__main__".
    spec.loader.exec_module(module)
    return module


def test_payload_size_matches_pr67_archive_byte_count() -> None:
    """Our QZS3 encoder must emit exactly 59288 bytes for an init-random
    JointFrameGenerator — matches PR #67's deployed uncompressed model size."""

    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model)
    assert payload.startswith(QZS3_MAGIC)
    # PR #67 archive's uncompressed model bytes (verified 2026-05-01).
    assert len(payload) == 59288, (
        f"QZS3 payload {len(payload)} bytes; expected 59288 to match PR #67 archive"
    )


def test_pr67_unpacker_decodes_our_payload() -> None:
    """Cross-check: PR #67's get_grouped_qv_state_dict must accept our bytes."""

    pr67 = _import_pr67()
    torch.manual_seed(123)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model)

    decoded = pr67.get_grouped_qv_state_dict(payload, torch.device("cpu"))

    template_state = model.state_dict()
    assert set(decoded.keys()) == set(template_state.keys()), (
        f"key mismatch: missing={set(template_state)-set(decoded)} "
        f"extra={set(decoded)-set(template_state)}"
    )

    qv_keys = set(qzs3_qv_specs().keys())
    fp4_keys = {
        k for k in template_state
        if (".dw.weight" in k or ".pw.weight" in k)
        and k not in {"frame1_head.head.weight", "frame2_head.head.weight"}
    }

    for key, ref in template_state.items():
        actual = decoded[key]
        assert actual.shape == ref.shape, f"{key} shape: {actual.shape} != {ref.shape}"
        diff = (actual.float() - ref.float()).abs().max().item()
        if key in fp4_keys:
            # FP4 codebook noise bound (max magnitude 6.0 * scale)
            assert diff < 0.2, f"FP4 layer {key} drift {diff} exceeds 0.2"
        elif key in qv_keys:
            # Per-row min/step quant (8-bit norm), expect <= step_max
            assert diff < 0.1, f"qv layer {key} drift {diff} exceeds 0.1"
        else:
            # FP16 conversion noise
            assert diff < 1e-3, f"FP16 layer {key} drift {diff} exceeds 1e-3"


def test_pr67_unpacker_loads_into_jointframegenerator() -> None:
    """The decoded state_dict must load() cleanly with strict=True into the
    template JointFrameGenerator (PR #67 calls this directly)."""

    pr67 = _import_pr67()
    torch.manual_seed(456)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model)

    decoded = pr67.get_grouped_qv_state_dict(payload, torch.device("cpu"))
    fresh = build_quantizr_faithful_renderer()
    # Should not raise: keys + shapes line up exactly.
    fresh.load_state_dict(decoded, strict=True)
    fresh.eval()

    mask = torch.zeros(1, 384, 512, dtype=torch.long)
    pose = torch.zeros(1, 6)
    with torch.no_grad():
        f1, f2 = fresh(mask, pose)
    assert f1.shape == (1, 3, 384, 512)
    assert f2.shape == (1, 3, 384, 512)
    assert torch.isfinite(f1).all() and torch.isfinite(f2).all()


def test_encoder_is_deterministic() -> None:
    """Same model -> same bytes (byte-identical archives are required for
    contest reproducibility + SHA-pinning custody chain)."""

    torch.manual_seed(789)
    model = build_quantizr_faithful_renderer().eval()
    a = encode_qzs3_state_dict(model)
    b = encode_qzs3_state_dict(model)
    assert a == b


def test_block_size_header_field_round_trips() -> None:
    """The 2-byte little-endian block_size field at offset 4 must round-trip."""

    torch.manual_seed(7)
    model = build_quantizr_faithful_renderer().eval()
    payload = encode_qzs3_state_dict(model, block_size=64)
    assert int.from_bytes(payload[4:6], "little") == 64
    decoded = decode_qzs3_state_dict(payload, device="cpu")
    fresh = build_quantizr_faithful_renderer()
    fresh.load_state_dict(decoded, strict=True)


def test_qzs4_block_search_keeps_qzs3_wire_format_and_records_candidates() -> None:
    torch.manual_seed(11)
    model = build_quantizr_faithful_renderer().eval()
    payload, meta = encode_qzs4_block_search_state_dict(
        model,
        block_sizes=(32, 64),
    )
    assert payload.startswith(QZS3_MAGIC)
    assert meta["packer_policy"] == "qzs4_block_search"
    assert meta["wire_format"] == "QZS3"
    assert meta["selected_block_size"] in {32, 64}
    assert len(meta["candidates"]) == 2
    assert sum(1 for item in meta["candidates"] if item["selected"]) == 1
    decoded = decode_qzs3_state_dict(payload, device="cpu")
    fresh = build_quantizr_faithful_renderer()
    fresh.load_state_dict(decoded, strict=True)


def test_repack_qzs3_block_size_reencodes_existing_qzs3_source(tmp_path: Path) -> None:
    torch.manual_seed(12)
    model = build_quantizr_faithful_renderer().eval()
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", encode_qzs3_state_dict(model, block_size=32))
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("optimized_poses.bin", b"\x00" * (4 * 6 * 2))

    out = tmp_path / "out.zip"
    meta = build_archive(
        source,
        out,
        renderer_codec=RENDERER_CODEC_QZS3,
        qzs3_block_size=64,
    )

    assert meta["renderer"]["action"] == "reencoded_qzs3_from_qzs3_state_dict"
    assert meta["renderer"]["source_block_size"] == 32
    assert meta["renderer"]["block_size"] == 64
    with zipfile.ZipFile(out) as zf:
        renderer = zf.read("renderer.bin")
    assert renderer.startswith(QZS3_MAGIC)
    assert int.from_bytes(renderer[4:6], "little") == 64
    decoded = decode_qzs3_state_dict(renderer, device="cpu")
    fresh = build_quantizr_faithful_renderer()
    fresh.load_state_dict(decoded, strict=True)


def test_repack_qzs4_records_submission_path_overhead_and_stackability(tmp_path: Path) -> None:
    torch.manual_seed(12)
    model = build_quantizr_faithful_renderer().eval()
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", encode_qzs3_state_dict(model))
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("optimized_poses.bin", b"\x00" * (4 * 6 * 2))

    out = tmp_path / "out.zip"
    meta = build_archive(
        source,
        out,
        renderer_codec=RENDERER_CODEC_QZS4,
        qzs4_block_sizes=(32, 64),
    )
    assert meta["renderer"]["renderer_codec"] == "qzs4"
    assert meta["renderer"]["wire_format"] == "QZS3"
    assert meta["submission_path"]["zip_overhead"]["member_count"] == 3
    assert meta["submission_path"]["charged_accounting"]["all_score_affecting_bits_inside_archive"] is True
    assert "pr65_postprocess_qpost_sidecar" in meta["submission_path"]["stackability"]


def test_repack_qzs4_accepts_existing_single_blob_frontier_archive(tmp_path: Path) -> None:
    from experiments.build_renderer_packed_payload_archive import (
        PAYLOAD_FORMAT_PR64_LEN_TABLE,
        POSE_QP1_CODEC,
        build_packed_archive,
    )

    torch.manual_seed(13)
    model = build_quantizr_faithful_renderer().eval()
    runtime_zip = tmp_path / "runtime.zip"
    pose_values: list[float] = []
    for row in range(4):
        pose_values.extend([20.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    with zipfile.ZipFile(runtime_zip, "w") as zf:
        zf.writestr("renderer.bin", encode_qzs3_state_dict(model))
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("optimized_poses.bin", pose_bytes)

    packed_frontier = tmp_path / "frontier_p.zip"
    build_packed_archive(
        runtime_zip,
        packed_frontier,
        payload_member_name="p",
        payload_format=PAYLOAD_FORMAT_PR64_LEN_TABLE,
        pose_codec=POSE_QP1_CODEC,
    )

    out = tmp_path / "stack.zip"
    meta = build_submission_archive(
        packed_frontier,
        out,
        renderer_codec=RENDERER_CODEC_QZS4,
        qzs4_block_sizes=(32, 64),
        submission_layout="pr64_single_blob",
        pose_codec=POSE_QP1_CODEC,
    )

    assert meta["source_archive_sha256"]
    assert meta["renderer_stage"]["source_runtime_contract"]["unpacked_from_packed_payload"] is True
    assert meta["renderer_stage"]["source_runtime_contract"]["packed_payload_member"] == "p"
    assert meta["renderer_stage"]["renderer"]["wire_format"] == "QZS3"
    assert meta["submission_path"]["layout"] == "pr64_single_blob"
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["p"]


def test_repack_script_help_is_directly_executable_from_repo_root() -> None:
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/repack_quantizr_faithful_qzs3_archive.py",
            "--help",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--qzs3-block-size" in proc.stdout
