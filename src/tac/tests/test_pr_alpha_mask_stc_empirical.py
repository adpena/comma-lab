"""Smoke tests for ``tools/pr_alpha_mask_stc_empirical.py``.

The full empirical needs the canonical ``masks.mkv``; here we synthesize a
small mask stream, run the pipeline at low constraint height, and assert the
manifest carries the right custody flags.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    """Import the tool module by path so it works whether or not ``tools/`` is
    on the import path."""
    spec = importlib.util.spec_from_file_location(
        "pr_alpha_mask_stc_empirical",
        REPO_ROOT / "tools" / "pr_alpha_mask_stc_empirical.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr_alpha_mask_stc_empirical"] = module
    spec.loader.exec_module(module)
    return module


def test_encode_lzma_raw_int8_returns_positive_bytes() -> None:
    tool = _load_tool()
    deltas = np.array([0, 1, -1, 0, 0, 1, 1, -1, 0, 0], dtype=np.int8)
    n = tool.encode_lzma_raw_int8(deltas)
    assert isinstance(n, int)
    assert n > 0


def test_encode_stc_zero_message_returns_dict_with_required_keys() -> None:
    tool = _load_tool()
    rng = np.random.default_rng(7)
    deltas = rng.choice([-1, 0, 1], size=128).astype(np.int8)
    res = tool.encode_stc_zero_message(deltas, constraint_height=4, block_size=32)
    for key in (
        "lzma_compressed_bytes",
        "raw_stego_bytes",
        "total_flips_soz",
        "total_flips_sign",
        "n_blocks",
        "block_size",
        "embedding_distortion",
    ):
        assert key in res, f"missing key {key}"
    assert res["lzma_compressed_bytes"] > 0
    assert res["raw_stego_bytes"] == deltas.size
    assert res["block_size"] == 32


def test_run_experiment_manifest_carries_custody_flags(tmp_path: Path) -> None:
    """Synthesize a tiny mask cube, run the experiment, assert the manifest
    has the non-negotiable custody flags set correctly per CLAUDE.md."""
    tool = _load_tool()
    # Build a tiny synthetic masks.mkv-equivalent: write directly via the
    # canonical mask-video encoder if available; else the test skips.
    try:
        from tac.mask_codec import encode_masks_to_video, decode_masks_auto  # noqa: F401
    except Exception:
        pytest.skip("tac.mask_codec encode/decode round-trip unavailable")

    # 5-class masks of shape (8, 12, 16) — small but enough to exercise
    # consecutive-frame deltas.
    rng = np.random.default_rng(2026)
    masks = rng.integers(0, 5, size=(8, 12, 16), dtype=np.int64)
    masks_path = tmp_path / "masks.mkv"
    try:
        encode_masks_to_video(masks, str(masks_path))
    except Exception:
        # Encoder might require larger frames; skip gracefully.
        pytest.skip("encode_masks_to_video rejected synthetic input")

    if not masks_path.is_file() or masks_path.stat().st_size == 0:
        pytest.skip("synthetic masks.mkv was not produced")

    manifest = tool.run_experiment(
        masks_path, tmp_path / "scratch", constraint_height=4, block_size=32
    )

    # Custody flags — non-negotiable per CLAUDE.md
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["family_falsified"] is False
    assert (
        manifest["falsification_scope"] == "filler_stc_ternary_mask_delta_only"
    )
    assert manifest["evidence_grade"].startswith("[CPU-prep faithful Filler-STC")
    assert "headline" in manifest
    assert manifest["stc_compressed_bytes"] > 0


def test_module_has_evidence_grade_constant() -> None:
    tool = _load_tool()
    assert (
        tool.EVIDENCE_GRADE == "[CPU-prep faithful Filler-STC PARADIGM-α test]"
    )
    assert tool.SCHEMA_VERSION == "pr_alpha_mask_stc_empirical.v1"
