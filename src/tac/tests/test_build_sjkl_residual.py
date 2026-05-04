"""Tests for experiments/build_sjkl_residual.py — recovery + smoke verification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDER_PATH = REPO_ROOT / "experiments" / "build_sjkl_residual.py"
spec = importlib.util.spec_from_file_location("build_sjkl_residual", _BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
sys.modules["build_sjkl_residual"] = builder
spec.loader.exec_module(builder)

from tac.sjkl_basis import decode_full_sjkl_payload, decode_sjkl_alpha_block  # noqa: E402

import brotli  # noqa: E402
import numpy as np  # noqa: E402


def _run_cpu_stub_build(tmp_path, *, rank=4, n_pairs=16, alpha_bits=4, basis_quant_bits=6, seed=42, max_bytes=32768):
    """Helper: run the full build pipeline in CPU stub mode and return (sjkl_path, manifest)."""
    out_dir = tmp_path / "sjkl_smoke"
    cfg = builder.BuildConfig(
        pair_tensor_manifest=Path("/dev/null"),
        output_dir=out_dir,
        device="cpu",
        rank=rank,
        n_pairs=n_pairs,
        alpha_bits=alpha_bits,
        basis_quant_bits=basis_quant_bits,
        max_bytes=max_bytes,
        allow_cpu_stub=True,
        seed=seed,
    )
    manifest = builder.build_sjkl_residual(cfg)
    return out_dir / "sjkl.bin", manifest


def test_cpu_stub_build_emits_sjkl_bin(tmp_path):
    sjkl_path, manifest = _run_cpu_stub_build(tmp_path)
    assert sjkl_path.is_file()
    assert sjkl_path.stat().st_size == manifest["sjkl_bin_bytes"]
    assert manifest["score_claim"] is False
    assert manifest["device"] == "cpu"


def test_emitted_sjkl_bin_runtime_decodable(tmp_path):
    """Critical: the sjkl.bin we write must be decodable by the runtime parser
    (decode_full_sjkl_payload + decode_sjkl_alpha_block)."""
    sjkl_path, manifest = _run_cpu_stub_build(tmp_path, rank=4, n_pairs=16, alpha_bits=4)
    payload = sjkl_path.read_bytes()

    basis, meta = decode_full_sjkl_payload(payload)
    assert basis.rank == 4
    assert basis.dim == 256  # from CPU stub default

    # alpha block is brotli-compressed in our payload
    decompressed = brotli.decompress(meta["alpha_block_raw_bytes"])
    alpha = decode_sjkl_alpha_block(decompressed)
    assert alpha["alpha_block_format"] == "sparse_bitpacked_v2"
    assert alpha["qs"].shape == (16, 4)
    assert alpha["alpha_bits"] == 4
    assert alpha["pair_indices"] is not None
    assert alpha["pair_indices"].shape == (16,)


def test_cpu_stub_requires_allow_flag(tmp_path):
    """Per CLAUDE.md FORBIDDEN PATTERNS: --device cpu must be opt-in via --allow-cpu-stub."""
    cfg = builder.BuildConfig(
        pair_tensor_manifest=Path("/dev/null"),
        output_dir=tmp_path / "x",
        device="cpu",
        rank=4, n_pairs=4, alpha_bits=4, basis_quant_bits=6, max_bytes=32768,
        allow_cpu_stub=False, seed=0,
    )
    with pytest.raises(SystemExit, match="requires --allow-cpu-stub"):
        builder.build_sjkl_residual(cfg)


def test_size_cap_enforced(tmp_path):
    """max_bytes hard cap fails loud rather than silently producing oversized payload."""
    cfg = builder.BuildConfig(
        pair_tensor_manifest=Path("/dev/null"),
        output_dir=tmp_path / "x",
        device="cpu",
        rank=8, n_pairs=64, alpha_bits=8, basis_quant_bits=8,
        max_bytes=100,  # absurdly small to force failure
        allow_cpu_stub=True, seed=0,
    )
    with pytest.raises(SystemExit, match="size .* > max_bytes"):
        builder.build_sjkl_residual(cfg)


def test_manifest_records_score_claim_false(tmp_path):
    """Score claim discipline: until contest_auth_eval.json lands, score_claim=false."""
    sjkl_path, manifest = _run_cpu_stub_build(tmp_path)
    manifest_path = sjkl_path.parent / "sjkl_manifest.json"
    persisted = json.loads(manifest_path.read_text())
    assert persisted["score_claim"] is False
    assert "advisory" in persisted["tag"].lower() or "empirical" in persisted["tag"].lower()


def test_quantize_alpha_per_pair_basic():
    alpha = np.array([[0.0, 0.5, 1.0], [-0.2, 0.3, 0.4]], dtype=np.float32)
    qs, mins, steps = builder._quantize_alpha_per_pair(alpha, alpha_bits=4)
    assert qs.shape == (2, 3)
    assert qs.dtype == np.uint8
    qmax = 15
    assert int(qs.min()) >= 0 and int(qs.max()) <= qmax
    # Reconstruct and verify within step granularity
    recon = mins[:, None] + qs.astype(np.float32) * steps[:, None]
    assert np.allclose(recon, alpha, atol=steps.max() + 1e-6)


def test_select_pairs_no_duplicates():
    indices = builder._select_pairs(n_total=600, n_select=16, seed=42)
    assert len(set(indices.tolist())) == 16
    assert indices.dtype == np.uint16
    assert all(0 <= i < 600 for i in indices)


def test_select_pairs_rejects_oversize():
    with pytest.raises(ValueError, match="cannot exceed"):
        builder._select_pairs(n_total=10, n_select=20, seed=0)


def test_cli_argv_parses():
    parser = builder.build_parser()
    args = parser.parse_args([
        "--output-dir", "/tmp/xyz",
        "--device", "cpu",
        "--rank", "8",
        "--n-pairs", "32",
        "--alpha-bits", "6",
        "--basis-quant-bits", "8",
        "--allow-cpu-stub",
        "--seed", "123",
    ])
    assert args.rank == 8
    assert args.n_pairs == 32
    assert args.alpha_bits == 6
    assert args.basis_quant_bits == 8
    assert args.allow_cpu_stub is True
    assert args.seed == 123
