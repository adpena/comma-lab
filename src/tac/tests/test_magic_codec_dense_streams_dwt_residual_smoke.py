# SPDX-License-Identifier: MIT
"""Tests for ``tools/run_magic_codec_dense_streams_dwt_residual_smoke.py``.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #344 + Catalog #272
+ Catalog #192 + the MAGIC CODEC × CASCADE STACKING ANALYSIS memo
``.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md``.

Sister of ``src/tac/tests/test_dwt_detail_subband_procedural_smoke.py``
(disjoint scope: yesterday's smoke measures distributional fit under H0
direct substitution; THIS smoke measures byte budget under rescue-path
procedural-predictor + dense-stream residual encoding).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "run_magic_codec_dense_streams_dwt_residual_smoke.py"


def _load_tool_module():
    """Import the tool module as a runtime module (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "run_magic_codec_dense_streams_dwt_residual_smoke", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_tool_module()


def test_tool_module_imports_canonical_surfaces():
    """Catalog #229 PV: every canonical helper the smoke depends on is importable."""
    assert hasattr(SMOKE, "run_smoke")
    assert hasattr(SMOKE, "append_second_empirical_anchor")
    assert hasattr(SMOKE, "compute_residuals_int8")
    assert hasattr(SMOKE, "encode_config_a_baseline_brotli")
    assert hasattr(SMOKE, "encode_config_c_procedural_plus_dense_streams")
    assert hasattr(SMOKE, "PAIR_1_PREDICTED_DELTA_S")
    assert hasattr(SMOKE, "CANONICAL_RATE_MULTIPLIER")
    assert hasattr(SMOKE, "CANONICAL_RATE_DENOM_BYTES")


def test_canonical_constants_match_contest_formula():
    """The smoke uses the canonical contest formula constants (Catalog #344 sister)."""
    assert SMOKE.CANONICAL_RATE_MULTIPLIER == 25.0
    assert SMOKE.CANONICAL_RATE_DENOM_BYTES == 37_545_489
    # Pair #1 prediction from stacking analysis memo §7
    assert SMOKE.PAIR_1_PREDICTED_DELTA_S == -0.00200
    assert SMOKE.PAIR_1_COMPOSITION_ALPHA_ESTIMATE == 0.8


def test_compute_residuals_int8_round_trip_self_is_zero():
    """Residual of empirical against itself is all-zero."""
    arr = np.array([1, -2, 3, -4, 5], dtype=np.int8)
    residuals = SMOKE.compute_residuals_int8(arr, arr)
    assert residuals.dtype == np.int8
    assert residuals.shape == arr.shape
    assert int(np.abs(residuals).sum()) == 0


def test_compute_residuals_int8_rejects_dtype_mismatch():
    """Catalog #229 type discipline: residual inputs must be int8."""
    a = np.array([1, 2, 3], dtype=np.int8)
    b = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(ValueError, match="int8"):
        SMOKE.compute_residuals_int8(a, b)


def test_compute_residuals_int8_rejects_shape_mismatch():
    """Catalog #229 type discipline: residual inputs must be same shape."""
    a = np.array([1, 2, 3], dtype=np.int8)
    b = np.array([[1, 2], [3, 4]], dtype=np.int8)
    with pytest.raises(ValueError, match="shape mismatch"):
        SMOKE.compute_residuals_int8(a, b)


def test_compute_residuals_int8_clips_overflow_to_int8_range():
    """Residual clipping preserves int8 dtype (saturating, not wrapping)."""
    a = np.array([127, 127, 127], dtype=np.int8)
    b = np.array([-128, -128, -128], dtype=np.int8)
    residuals = SMOKE.compute_residuals_int8(a, b)
    assert residuals.dtype == np.int8
    # 127 - (-128) = 255 → clipped to 127
    assert (residuals == 127).all()


def test_encode_config_a_baseline_brotli_returns_byte_count():
    """Configuration A direct empirical brotli baseline emits a positive byte count."""
    arr = np.array([1, 2, 3, 4, 5] * 100, dtype=np.int8)
    n_bytes = SMOKE.encode_config_a_baseline_brotli(arr)
    assert isinstance(n_bytes, int)
    assert n_bytes > 0
    # Brotli should compress 500 repetitive bytes well below the raw 500
    assert n_bytes < 500


def test_encode_config_c_returns_total_bytes_plus_selection_log():
    """Configuration C emits total_bytes + selection_log with per-codec candidates."""
    residuals = np.random.RandomState(42).randint(-50, 50, size=200, dtype=np.int8)
    seed = b"\x00" * 32
    total_bytes, selection_log = SMOKE.encode_config_c_procedural_plus_dense_streams(
        residuals_int8=residuals,
        seed_bytes=seed,
        subband_name="LH_test",
    )
    assert isinstance(total_bytes, int)
    assert total_bytes > 0
    assert isinstance(selection_log, dict)
    assert selection_log["seed_bytes_len"] == 32
    assert selection_log["dense_streams_payload_len"] > 0
    assert total_bytes == 32 + selection_log["dense_streams_payload_len"]
    assert "selected_codec_name" in selection_log
    assert selection_log["selected_codec_name"] in {"brotli", "lzma", "magic_codec_classic"}
    # 3 codec candidates (brotli + lzma + magic_classic; latter refused without hint)
    assert isinstance(selection_log["per_codec_candidates"], list)
    assert len(selection_log["per_codec_candidates"]) >= 2


def test_derive_seed_for_subband_distinct_per_subband():
    """Per-subband seed derivation is deterministic and distinct per subband name."""
    base = b"\x01" * 32
    s_lh = SMOKE.derive_seed_for_subband("LH", base)
    s_hl = SMOKE.derive_seed_for_subband("HL", base)
    s_hh = SMOKE.derive_seed_for_subband("HH", base)
    assert len(s_lh) == 32
    assert len(s_hl) == 32
    assert len(s_hh) == 32
    assert s_lh != s_hl
    assert s_lh != s_hh
    assert s_hl != s_hh
    # Determinism: re-derive returns same bytes
    assert SMOKE.derive_seed_for_subband("LH", base) == s_lh


def test_run_smoke_end_to_end_emits_canonical_keys():
    """End-to-end smoke run on real contest video emits all canonical custody fields."""
    video = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    base_seed = hashlib.sha256(b"test_seed_pair_1_dwt_residual").digest()
    result = SMOKE.run_smoke(
        video_path=video,
        frame_index=300,
        base_seed_bytes=base_seed,
        wavelet="haar",
        dwt_level=2,
        generator_kind="pcg64",
    )
    # Custody triple per Catalog #127 + #192 + #323
    assert result["axis_tag"] == "[macOS-CPU advisory]"
    assert result["hardware_substrate"] == "darwin_arm64_m5_max_macos_cpu_advisory"
    assert result["evidence_grade"] == "local_cpu_smoke_advisory"
    assert result["promotion_eligible"] is False
    assert result["score_claim_valid"] is False
    assert result["score_claim_axis"] is None
    # 3-way comparison fields
    assert "aggregate_config_a_baseline_brotli_bytes" in result
    assert "aggregate_config_b_procedural_only_bytes" in result
    assert "aggregate_config_c_procedural_plus_dense_streams_bytes" in result
    assert "aggregate_bytes_saved_c_vs_a" in result
    assert "empirical_delta_s" in result
    assert "predicted_delta_s_pair_1" in result
    assert result["predicted_delta_s_pair_1"] == -0.00200
    # All 3 detail subbands present
    assert set(result["per_subband"].keys()) == {"LH", "HL", "HH"}
    # Canonical equation linkage
    assert result["canonical_equation_id"] == "procedural_codebook_from_seed_compression_savings_v1"
    assert result["canonical_equation_in_domain_context"] == (
        "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands"
    )


def test_main_with_skip_anchor_append_writes_artifacts(tmp_path):
    """`--skip-canonical-equation-append` emits JSON + MD without touching registry."""
    video = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video.exists():
        pytest.skip("upstream/videos/0.mkv not present")
    rc = SMOKE.main(
        [
            "--video-path",
            str(video),
            "--frame-index",
            "100",  # small frame index for quick smoke
            "--output-dir",
            str(tmp_path),
            "--skip-canonical-equation-append",
        ]
    )
    assert rc == 0
    assert (tmp_path / "smoke_result.json").exists()
    assert (tmp_path / "smoke_result.md").exists()
    payload = json.loads((tmp_path / "smoke_result.json").read_text())
    # Re-verify canonical custody on the persisted artifact
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["evidence_grade"] == "local_cpu_smoke_advisory"
    # Verdict is one of the canonical taxonomy
    assert payload["canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"] in {
        "HARD-EARNED",
        "CARGO-CULTED",
    }
    assert payload["dwt_bind_rescue_path_verdict"] in {
        "DWT_BIND_RESCUE_PATH_VALIDATED_PROCEED_TO_PAIR_2",
        "PARTIAL_RESCUE_NET_SAVINGS_BUT_OUTSIDE_PREDICTED_BAND",
        "DWT_BIND_RESCUE_PATH_FALSIFIED_PIVOT_TO_PAIR_2_NULL_BYTE_RESIDUALS",
        "INDETERMINATE_REQUIRES_PAIRED_LINUX_X86_64_VERIFICATION",
    }


def test_main_refuses_missing_video():
    """CLI fails fast on missing video file (rc=2 per UNIX convention)."""
    rc = SMOKE.main(
        [
            "--video-path",
            "/nonexistent/video.mkv",
            "--skip-canonical-equation-append",
        ]
    )
    assert rc == 2
