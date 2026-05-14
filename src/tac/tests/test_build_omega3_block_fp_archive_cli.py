# SPDX-License-Identifier: MIT
"""Smoke test for ``experiments/build_omega3_block_fp_archive.py``.

Synthesizes a minimal QFAI-format source archive (JFG state-dict) on disk,
runs the CLI end-to-end, and asserts the manifest schema is correct and
the output archive is decompressible through the BFJ1 codec path.

Strict-scorer-rule: pure CPU torch math; no SegNet/PoseNet ever loaded.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "experiments" / "build_omega3_block_fp_archive.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "build_omega3_block_fp_archive_under_test",
        CLI_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_qfai_renderer_blob(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Build a QFAI binary blob from a state-dict for testing.

    Matches the QFAI layout the CLI's ``_load_jfg_state_dict_from_blob``
    expects.
    """
    import io as _io
    header = {
        "num_classes": 5,
        "pose_dim": 6,
        "cond_dim": 48,
        "depth_mult": 1,
    }
    header_b = json.dumps(header).encode("utf-8")
    body_buf = _io.BytesIO()
    torch.save(state_dict, body_buf)
    body = body_buf.getvalue()
    parts: list[bytes] = []
    parts.append(b"QFAI")
    parts.append(struct.pack("<I", len(header_b)))
    parts.append(header_b)
    parts.append(body)
    return b"".join(parts)


def _make_synthetic_jfg_state_dict() -> dict[str, torch.Tensor]:
    """Build a minimal JFG-shaped state-dict mirroring shared-trunk +
    pose_mlp + frame head structure. Roughly 6K params.
    """
    torch.manual_seed(11)
    # Tensor names matching tac.quantizr_faithful_renderer.JointFrameGenerator.
    sd: dict[str, torch.Tensor] = {}
    sd["shared_trunk.embedding.weight"] = torch.randn(5, 6) * 0.1
    sd["shared_trunk.stem_conv.dw.weight"] = torch.randn(8, 1, 3, 3) * 0.05
    sd["shared_trunk.stem_conv.pw.weight"] = torch.randn(56, 8, 1, 1) * 0.05
    sd["shared_trunk.stem_conv.pw.bias"] = torch.randn(56) * 0.01
    sd["shared_trunk.stem_conv.norm.weight"] = torch.randn(56) * 0.01
    sd["shared_trunk.stem_conv.norm.bias"] = torch.randn(56) * 0.01
    sd["pose_mlp.0.weight"] = torch.randn(48, 6) * 0.1
    sd["pose_mlp.0.bias"] = torch.randn(48) * 0.01
    sd["pose_mlp.2.weight"] = torch.randn(48, 48) * 0.1
    sd["pose_mlp.2.bias"] = torch.randn(48) * 0.01
    sd["frame1_head.block1.film_proj.weight"] = torch.randn(112, 48) * 0.1
    sd["frame1_head.block1.film_proj.bias"] = torch.randn(112) * 0.01
    sd["frame1_head.head.weight"] = torch.randn(3, 52, 1, 1) * 0.05
    sd["frame1_head.head.bias"] = torch.randn(3) * 0.01
    return sd


def test_cli_smoke_on_synthetic_jfg(tmp_path: Path) -> None:
    """End-to-end: build a synthetic source archive, run the CLI,
    verify manifest schema + output archive integrity."""
    sd = _make_synthetic_jfg_state_dict()
    renderer_blob = _make_qfai_renderer_blob(sd)
    src_archive = tmp_path / "source_archive.zip"
    with zipfile.ZipFile(src_archive, "w") as zf:
        zf.writestr("renderer.bin", renderer_blob)
        zf.writestr("masks.mkv", b"\x00\x00\x00\x18ftypisom")  # placeholder
        zf.writestr("optimized_poses.pt", b"placeholder pose bytes")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        str(CLI_PATH),
        "--source-archive",
        str(src_archive),
        "--output-dir",
        str(out_dir),
        "--block-size",
        "64",
        "--film-block-size",
        "32",
        "--validate-film-mse-threshold",
        "1e-1",  # generous; small synthetic FiLM layer should pass
        "--max-bytes",
        "1000000",
        "--lzma-preset",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        pytest.fail(
            f"CLI exited with {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    archive_path = out_dir / "archive.zip"
    manifest_path = out_dir / "manifest.json"
    assert archive_path.exists(), "output archive missing"
    assert manifest_path.exists(), "manifest.json missing"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Required schema fields.
    for key in (
        "timestamp_utc",
        "source_archive_path",
        "source_archive_sha256",
        "source_archive_bytes",
        "source_renderer_member",
        "source_renderer_bytes",
        "output_archive_path",
        "output_archive_sha256",
        "output_archive_bytes",
        "output_renderer_bytes",
        "byte_savings",
        "config",
        "arch_meta",
        "film_validation",
        "layer_count",
        "layer_protected_count",
        "layer_blockfp_count",
        "score_claim",
        "score_evidence_grade",
        "predicted_rate_delta",
    ):
        assert key in manifest, f"missing manifest key: {key}"

    # Score-tagging discipline: NEVER claim contest-CUDA.
    assert manifest["score_claim"] is False
    assert manifest["score_evidence_grade"] == "predicted"
    assert manifest["predicted_rate_delta"]["tag"] == "[predicted]"

    # Config sanity.
    assert manifest["config"]["bfj1_magic"] == "BFJ1"
    assert manifest["config"]["bfj1_version"] == 1
    assert manifest["config"]["protect_film_layers"] is True
    # FiLM patterns by name match the codec defaults.
    assert "film" in manifest["config"]["protect_patterns"]

    # Architecture inferred from QFAI header.
    assert manifest["arch_meta"]["container"] == "QFAI"
    assert manifest["arch_meta"]["num_classes"] == 5

    # FiLM validation must have been run on the FiLM layers.
    film_layers = [r for r in manifest["film_validation"]
                   if "film" in r["layer_name"]]
    assert len(film_layers) >= 1, "expected at least one FiLM-validated layer"
    for r in manifest["film_validation"]:
        assert r["passed"] is True
        assert r["roundtrip_mse"] >= 0.0
        assert r["effective_bpw"] >= 0.0

    # Output archive structure.
    with zipfile.ZipFile(archive_path, "r") as out_zf:
        out_names = out_zf.namelist()
        assert "renderer.bin" in out_names
        # Other members byte-for-byte preserved.
        assert "masks.mkv" in out_names
        assert "optimized_poses.pt" in out_names
        new_renderer_blob = out_zf.read("renderer.bin")
        assert new_renderer_blob[:4] == b"BFJ1"
        # Mask + pose preserved exactly.
        with zipfile.ZipFile(src_archive, "r") as src_zf:
            assert out_zf.read("masks.mkv") == src_zf.read("masks.mkv")
            assert out_zf.read("optimized_poses.pt") == src_zf.read(
                "optimized_poses.pt"
            )

    # Round-trip: BFJ1 -> state_dict.
    from tac.block_fp_jfg import decompress_jfg_block_fp
    restored = decompress_jfg_block_fp(new_renderer_blob)
    assert set(restored.keys()) == set(sd.keys())


def test_cli_kills_on_film_mse_violation(tmp_path: Path) -> None:
    """When the FiLM MSE threshold is set extremely low, the CLI must
    exit nonzero and surface the FiLM-layer name in the error message."""
    sd = _make_synthetic_jfg_state_dict()
    renderer_blob = _make_qfai_renderer_blob(sd)
    src_archive = tmp_path / "source.zip"
    with zipfile.ZipFile(src_archive, "w") as zf:
        zf.writestr("renderer.bin", renderer_blob)

    out_dir = tmp_path / "out_kill"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        str(CLI_PATH),
        "--source-archive",
        str(src_archive),
        "--output-dir",
        str(out_dir),
        "--block-size",
        "64",
        "--validate-film-mse-threshold",
        "1e-30",  # impossible - guaranteed kill
        "--max-bytes",
        "1000000",
        "--lzma-preset",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0, "expected nonzero exit on FiLM kill"
    combined = result.stdout + result.stderr
    assert "FiLM-layer validation FAILED" in combined or "FiLM" in combined


def test_loader_accepts_frontier_jfg_wire_formats_and_rejects_unknown_magic() -> None:
    """Omega-3 source intake must cover actual JFG frontier payloads."""
    from tac.block_fp_jfg import (
        BlockFPConfig,
        compress_jfg_block_fp,
        quantize_jfg_block_fp,
    )
    from tac.qbf1_renderer_codec import pack_qbf1_state_dict
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
    from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

    module = _load_cli_module()
    torch.manual_seed(12)
    model = build_quantizr_faithful_renderer().eval()
    state = model.state_dict()
    payloads = {
        "QZS3": encode_qzs3_state_dict(model),
        "QBF1": pack_qbf1_state_dict(state, block_size=32),
        "BFJ1": compress_jfg_block_fp(
            quantize_jfg_block_fp(state, BlockFPConfig(block_size=32, lzma_preset=1)),
            lzma_preset=1,
        ),
    }

    for expected_container, payload in payloads.items():
        restored, meta = module._load_jfg_state_dict_from_blob(
            payload,
            device=torch.device("cpu"),
        )
        assert meta["container"] == expected_container
        assert set(restored) == set(state)

    with pytest.raises(ValueError, match="unsupported JFG renderer container magic"):
        module._load_jfg_state_dict_from_blob(
            b"NOPEnot-a-supported-jfg-payload",
            device=torch.device("cpu"),
        )


def test_source_member_detection_rejects_pr106_style_hnerv_zero_bin(
    tmp_path: Path,
) -> None:
    module = _load_cli_module()
    source_archive = tmp_path / "hnerv_like.zip"
    with zipfile.ZipFile(source_archive, "w") as zf:
        zf.writestr("0.bin", b"HNeRV-packed-payload-not-jfg")
        zf.writestr("masks.mkv", b"mask")

    with (
        zipfile.ZipFile(source_archive, "r") as zf,
        pytest.raises(ValueError, match="no supported JFG renderer member"),
    ):
        module._identify_source_renderer_member(zf)


def test_builder_rejects_non_default_depth_mult_before_archive_write(
    tmp_path: Path,
) -> None:
    sd = _make_synthetic_jfg_state_dict()
    renderer_blob = _make_qfai_renderer_blob(sd)
    header_len = struct.unpack("<I", renderer_blob[4:8])[0]
    header = json.loads(renderer_blob[8:8 + header_len].decode("utf-8"))
    header["depth_mult"] = 2
    header_b = json.dumps(header).encode("utf-8")
    rewritten_blob = b"".join(
        [
            b"QFAI",
            struct.pack("<I", len(header_b)),
            header_b,
            renderer_blob[8 + header_len:],
        ]
    )
    source_archive = tmp_path / "source_depth2.zip"
    with zipfile.ZipFile(source_archive, "w") as zf:
        zf.writestr("renderer.bin", rewritten_blob)

    out_dir = tmp_path / "out_depth2"
    cmd = [
        sys.executable,
        str(CLI_PATH),
        "--source-archive",
        str(source_archive),
        "--output-dir",
        str(out_dir),
        "--lzma-preset",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "depth_mult=1" in (result.stdout + result.stderr)
    assert not (out_dir / "archive.zip").exists()
