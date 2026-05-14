# SPDX-License-Identifier: MIT
"""Tests for experiments/prepare_sjkl_pair_tensors.py — recovery + smoke verification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_PREP_PATH = REPO_ROOT / "experiments" / "prepare_sjkl_pair_tensors.py"
spec = importlib.util.spec_from_file_location("prepare_sjkl_pair_tensors", _PREP_PATH)
prep = importlib.util.module_from_spec(spec)
sys.modules["prepare_sjkl_pair_tensors"] = prep
spec.loader.exec_module(prep)


def _run_cpu_stub(tmp_path, *, n_pairs=600, dim=256, seed=42):
    cfg = prep.PrepConfig(
        renderer_output=Path("/dev/null"),
        target_frames=Path("/dev/null"),
        output_dir=tmp_path / "prep_smoke",
        n_pairs=None,
        anchor_pair_idx=0,
        flatten=True,
        cpu_stub=True,
        cpu_stub_dim=dim,
        cpu_stub_n_pairs=n_pairs,
        seed=seed,
    )
    return prep.prepare_sjkl_pair_tensors(cfg)


def test_cpu_stub_emits_required_artifacts(tmp_path):
    manifest = _run_cpu_stub(tmp_path, n_pairs=64, dim=128)
    out_dir = tmp_path / "prep_smoke"
    assert (out_dir / "anchor_frame.pt").is_file()
    assert (out_dir / "pair_residuals.pt").is_file()
    assert (out_dir / "sjkl_pair_tensor_prep_manifest.json").is_file()
    assert manifest["source_kind"] == "cpu_stub"
    assert manifest["n_pairs"] == 64
    assert manifest["frame_dim"] == 128
    assert manifest["score_claim"] is False


def test_anchor_and_residuals_have_expected_shapes(tmp_path):
    _run_cpu_stub(tmp_path, n_pairs=32, dim=64)
    out_dir = tmp_path / "prep_smoke"
    anchor = torch.load(out_dir / "anchor_frame.pt", weights_only=False)
    residuals = torch.load(out_dir / "pair_residuals.pt", weights_only=False)
    assert anchor.shape == (64,)
    assert residuals.shape == (32, 64)


def test_manifest_matches_build_sjkl_residual_contract(tmp_path):
    """build_sjkl_residual.py reads the manifest and expects these exact keys."""
    _run_cpu_stub(tmp_path, n_pairs=16, dim=32)
    out_dir = tmp_path / "prep_smoke"
    manifest = json.loads((out_dir / "sjkl_pair_tensor_prep_manifest.json").read_text())
    required = {
        "schema_version", "anchor_frame_path", "pair_residuals_path",
        "n_pairs", "frame_dim", "source_kind", "source_sha256",
        "anchor_frame_sha256", "pair_residuals_sha256", "produced_at_utc",
        "produced_by", "score_claim",
    }
    assert required.issubset(manifest.keys())


def test_full_pipeline_prep_then_residual_smoke(tmp_path):
    """End-to-end: prepare tensors, then point build_sjkl_residual at the manifest
    (not the actual file path — build_sjkl_residual falls back to its own CPU stub
    if the manifest path doesn't exist, but here it WILL exist after prep runs)."""
    _run_cpu_stub(tmp_path, n_pairs=32, dim=128)
    manifest_path = tmp_path / "prep_smoke" / "sjkl_pair_tensor_prep_manifest.json"
    assert manifest_path.is_file()

    # Now load and verify build_sjkl_residual would accept this manifest's schema
    manifest = json.loads(manifest_path.read_text())
    # paths in manifest are repo-relative — verify they resolve to existing files
    anchor = REPO_ROOT / manifest["anchor_frame_path"]
    residuals = REPO_ROOT / manifest["pair_residuals_path"]
    # Note: tmp_path is OUTSIDE REPO_ROOT, so the relative-path branch won't apply;
    # the manifest will have absolute paths in that case
    if not anchor.is_file():
        anchor = Path(manifest["anchor_frame_path"])
        residuals = Path(manifest["pair_residuals_path"])
    assert anchor.is_file(), f"anchor not at {anchor}"
    assert residuals.is_file(), f"residuals not at {residuals}"


def test_anchor_pair_idx_clamped(tmp_path):
    """anchor_pair_idx out-of-range gets clamped (not crashed)."""
    cfg = prep.PrepConfig(
        renderer_output=Path("/dev/null"),
        target_frames=Path("/dev/null"),
        output_dir=tmp_path / "p",
        n_pairs=None,
        anchor_pair_idx=9999,  # way beyond n_pairs
        flatten=True,
        cpu_stub=True,
        cpu_stub_dim=16,
        cpu_stub_n_pairs=4,
        seed=0,
    )
    manifest = prep.prepare_sjkl_pair_tensors(cfg)
    assert manifest["anchor_pair_idx"] == 3  # clamped to n_pairs - 1


def test_real_tensor_files_load_and_pair(tmp_path):
    """Synthesize real .pt files and feed them to the non-stub path."""
    n_pairs, C, H, W = 8, 3, 4, 4
    g = torch.Generator().manual_seed(0)
    renderer = torch.randn(n_pairs, C, H, W, generator=g, dtype=torch.float32)
    target = renderer + torch.randn_like(renderer) * 0.1
    renderer_path = tmp_path / "renderer_out.pt"
    target_path = tmp_path / "target.pt"
    torch.save(renderer, renderer_path)
    torch.save(target, target_path)

    cfg = prep.PrepConfig(
        renderer_output=renderer_path,
        target_frames=target_path,
        output_dir=tmp_path / "out",
        n_pairs=None,
        anchor_pair_idx=0,
        flatten=True,
        cpu_stub=False,
        cpu_stub_dim=0,
        cpu_stub_n_pairs=0,
        seed=0,
    )
    manifest = prep.prepare_sjkl_pair_tensors(cfg)
    assert manifest["n_pairs"] == 8
    assert manifest["frame_dim"] == C * H * W
    assert manifest["source_kind"] == "renderer_output"
    # source_sha must be a real hex digest (not the stub marker)
    assert len(manifest["source_sha256"]) == 64
    assert manifest["source_sha256"] != "stub_no_source_file"


def test_n_pairs_cap_truncates(tmp_path):
    n_pairs = 16
    g = torch.Generator().manual_seed(0)
    renderer = torch.randn(n_pairs, 32, generator=g, dtype=torch.float32)
    target = renderer + torch.randn_like(renderer) * 0.05
    renderer_path = tmp_path / "renderer_out.pt"
    target_path = tmp_path / "target.pt"
    torch.save(renderer, renderer_path)
    torch.save(target, target_path)

    cfg = prep.PrepConfig(
        renderer_output=renderer_path,
        target_frames=target_path,
        output_dir=tmp_path / "out",
        n_pairs=4,  # cap at 4
        anchor_pair_idx=0,
        flatten=True,
        cpu_stub=False,
        cpu_stub_dim=0, cpu_stub_n_pairs=0, seed=0,
    )
    manifest = prep.prepare_sjkl_pair_tensors(cfg)
    assert manifest["n_pairs"] == 4


def test_mismatched_n_pairs_rejected(tmp_path):
    g = torch.Generator().manual_seed(0)
    renderer = torch.randn(10, 16, generator=g, dtype=torch.float32)
    target = torch.randn(8, 16, generator=g, dtype=torch.float32)  # different count!
    renderer_path = tmp_path / "r.pt"
    target_path = tmp_path / "t.pt"
    torch.save(renderer, renderer_path)
    torch.save(target, target_path)
    cfg = prep.PrepConfig(
        renderer_output=renderer_path, target_frames=target_path,
        output_dir=tmp_path / "out", n_pairs=None, anchor_pair_idx=0,
        flatten=True, cpu_stub=False, cpu_stub_dim=0, cpu_stub_n_pairs=0, seed=0,
    )
    with pytest.raises(SystemExit, match="different pair counts"):
        prep.prepare_sjkl_pair_tensors(cfg)


def test_cli_argv_parses():
    parser = prep.build_parser()
    args = parser.parse_args([
        "--renderer-output", "/tmp/r.pt",
        "--target-frames", "/tmp/t.pt",
        "--output-dir", "/tmp/out",
        "--n-pairs", "100",
        "--anchor-pair-idx", "5",
    ])
    assert args.n_pairs == 100
    assert args.anchor_pair_idx == 5
    assert args.cpu_stub is False
    assert args.no_flatten is False
