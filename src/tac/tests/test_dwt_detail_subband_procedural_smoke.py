# SPDX-License-Identifier: MIT
"""Tests for ``tools/run_dwt_detail_subband_procedural_smoke.py``.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #344 + Catalog #272
+ Catalog #192. Sister of ``src/tac/tests/test_check_209_dp1_contest_video_leakage_caller_check.py``
(disjoint scope: DP1 contest-video-leakage at caller surface; this smoke is
the DWT detail-subband procedural distributional validation surface).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "run_dwt_detail_subband_procedural_smoke.py"

# Import the tool module as a runtime module (tools/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "run_dwt_detail_subband_procedural_smoke", TOOL_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["run_dwt_detail_subband_procedural_smoke"] = _module
_spec.loader.exec_module(_module)


def test_tool_module_imports_canonical_surfaces():
    """The smoke script must expose the canonical pipeline surface."""
    for name in (
        "decode_frame_from_video",
        "compute_dwt_detail_subbands",
        "normalize_to_int8_distribution",
        "kl_divergence_int8",
        "wasserstein1_int8",
        "derive_seed_for_subband",
        "run_smoke",
        "append_first_empirical_anchor",
        "emit_markdown_report",
        "main",
    ):
        assert hasattr(_module, name), f"smoke module missing canonical surface {name!r}"


def test_normalize_int8_distribution_handles_flat_subband():
    """Degenerate flat subband must produce all-zero int8 (no division by zero)."""
    flat = np.ones((16, 16), dtype=np.float32) * 7.0
    out = _module.normalize_to_int8_distribution(flat)
    assert out.dtype == np.int8
    assert out.shape == flat.shape
    assert int(np.count_nonzero(out)) == 0


def test_normalize_int8_distribution_clips_to_int8_range():
    """Output MUST fit in int8 [-128, 127]."""
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal(size=(64, 64)).astype(np.float32) * 100.0
    out = _module.normalize_to_int8_distribution(arr)
    assert out.dtype == np.int8
    assert int(out.min()) >= -128
    assert int(out.max()) <= 127


def test_kl_divergence_int8_self_is_zero():
    """KL(p || p) = 0 (after Laplace smoothing tiny positive but ≪ 0.01)."""
    rng = np.random.default_rng(seed=43)
    arr = rng.integers(-128, 128, size=(1024,), dtype=np.int8)
    kl = _module.kl_divergence_int8(arr, arr)
    assert kl >= 0.0
    assert kl < 0.01


def test_kl_divergence_int8_disjoint_supports_positive():
    """Two disjoint int8 distributions yield positive KL."""
    a = np.full((1024,), -100, dtype=np.int8)
    b = np.full((1024,), 100, dtype=np.int8)
    kl = _module.kl_divergence_int8(a, b)
    assert kl > 0.0


def test_wasserstein1_int8_self_is_zero():
    """W1(p, p) = 0."""
    rng = np.random.default_rng(seed=44)
    arr = rng.integers(-128, 128, size=(1024,), dtype=np.int8)
    w1 = _module.wasserstein1_int8(arr, arr)
    assert w1 == 0.0 or w1 < 1e-9


def test_derive_seed_for_subband_is_deterministic_and_distinct_per_name():
    """Same (base_seed, name) → same 32-byte digest; different names → distinct."""
    base = b"abcdefgh"
    s_lh = _module.derive_seed_for_subband("LH", base)
    s_lh_again = _module.derive_seed_for_subband("LH", base)
    s_hl = _module.derive_seed_for_subband("HL", base)
    assert s_lh == s_lh_again
    assert len(s_lh) == 32
    assert s_lh != s_hl


def test_compute_dwt_detail_subbands_returns_3_subbands():
    """pywt.wavedec2 level=2 must return LH, HL, HH at the coarsest level."""
    rng = np.random.default_rng(seed=45)
    y = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    subbands = _module.compute_dwt_detail_subbands(y, "haar", 2)
    assert set(subbands.keys()) == {"LH", "HL", "HH"}
    for name, arr in subbands.items():
        # At level=2 the coarsest detail subbands are 32x32 (128/4)
        assert arr.shape == (32, 32), f"{name} shape {arr.shape} != (32, 32)"


@pytest.mark.skipif(
    not (REPO_ROOT / "upstream" / "videos" / "0.mkv").exists(),
    reason="requires upstream/videos/0.mkv (skip in OSS clone / CI without video)",
)
def test_run_smoke_end_to_end_emits_canonical_keys(tmp_path):
    """End-to-end smoke must emit every canonical-result-dict key."""
    base_seed = hashlib.sha256(b"test_seed_for_smoke").digest()
    result = _module.run_smoke(
        video_path=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        frame_index=300,
        base_seed_bytes=base_seed,
        wavelet="haar",
        dwt_level=2,
        generator_kind="pcg64",
    )
    # Canonical custody triple per Catalog #127 + #192 + #323
    assert result["axis_tag"] == "[macOS-CPU advisory]"
    assert result["hardware_substrate"] == "darwin_arm64_m5_max_macos_cpu_advisory"
    assert result["evidence_grade"] == "local_cpu_smoke_advisory"
    assert result["promotion_eligible"] is False
    assert result["score_claim_valid"] is False

    # Canonical pipeline outputs
    assert result["wavelet"] == "haar"
    assert result["dwt_level"] == 2
    assert result["generator_kind"] == "pcg64"
    assert set(result["per_subband"].keys()) == {"LH", "HL", "HH"}

    # Each subband must have all metrics
    for name, per in result["per_subband"].items():
        for key in (
            "shape",
            "n_pixels",
            "kl_divergence_empirical_vs_synthetic_nats",
            "wasserstein1_empirical_vs_synthetic",
            "byte_mutation_smoke_kl_synthetic_vs_mutated_nats",
            "byte_mutation_smoke_byte_differs_count",
            "byte_mutation_smoke_verdict_seed_sensitive",
        ):
            assert key in per, f"per_subband[{name}] missing {key}"

    # Aggregate metrics
    assert result["aggregate_kl_divergence_nats_mean"] >= 0.0
    assert result["canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"] in (
        "HARD-EARNED",
        "CARGO-CULTED",
    )

    # Byte-mutation smoke verdict
    assert isinstance(
        result["byte_mutation_smoke_aggregate_seed_sensitive_all_subbands"], bool
    )

    # Equation linkage
    assert (
        result["canonical_equation_id"]
        == "procedural_codebook_from_seed_compression_savings_v1"
    )


def test_main_with_skip_anchor_append_writes_artifacts(tmp_path):
    """CLI smoke (skip canonical equation append for test hygiene) must write JSON + MD."""
    if not (REPO_ROOT / "upstream" / "videos" / "0.mkv").exists():
        pytest.skip("requires upstream/videos/0.mkv")
    out_dir = tmp_path / "test_smoke_out"
    rc = _module.main(
        [
            "--frame-index",
            "300",
            "--output-dir",
            str(out_dir),
            "--skip-canonical-equation-append",
        ]
    )
    assert rc == 0
    assert (out_dir / "smoke_result.json").exists()
    assert (out_dir / "smoke_result.md").exists()
    payload = json.loads((out_dir / "smoke_result.json").read_text())
    assert payload["promotion_eligible"] is False
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
