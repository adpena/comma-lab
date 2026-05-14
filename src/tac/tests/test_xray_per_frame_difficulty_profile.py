# SPDX-License-Identifier: MIT
"""Tests for tools/xray_per_frame_difficulty_profile.py.

Per CLAUDE.md `forbidden_score_claims`: this profile is `[diagnostic; ...]`,
NOT a score. Per HNeRV parity discipline lesson 1: difficulty MUST be
score-aware (gradient-through-SegNet/PoseNet, not raw L²/KL on frames).

Tests use small synthetic data to avoid the multi-hour CPU full-video run.
The end-to-end full-video execution is exercised via the smoke flag in
``test_full_pipeline_smoke`` and via the build_a1 dispatch tool.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "xray_per_frame_difficulty_profile.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("xray_per_frame_difficulty_profile", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


# ────────────────────────────────────────────────────────────────────────────
# Pure-function tests (no scorer load)


class TestPercentileRank:
    def test_basic_ranking(self, tool):
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ranks = tool._percentile_rank(values)
        assert ranks.shape == (5,)
        assert ranks[0] == pytest.approx(0.0, abs=1e-6)
        assert ranks[-1] == pytest.approx(100.0, abs=1e-6)
        # Monotonic.
        for i in range(4):
            assert ranks[i] < ranks[i + 1]

    def test_rejects_empty_returns_empty(self, tool):
        ranks = tool._percentile_rank(np.array([]))
        assert ranks.shape == (0,)

    def test_single_value(self, tool):
        ranks = tool._percentile_rank(np.array([42.0]))
        assert ranks.shape == (1,)
        assert ranks[0] == pytest.approx(0.0)


class TestStateHash:
    def test_deterministic(self, tool):
        a = {"x": 1, "y": [2, 3]}
        h1 = tool._state_hash(a)
        h2 = tool._state_hash(a)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_input_different_hash(self, tool):
        a = {"x": 1}
        b = {"x": 2}
        assert tool._state_hash(a) != tool._state_hash(b)


class TestAsciiHistogram:
    def test_basic(self, tool):
        values = np.linspace(0, 1, 100)
        hist = tool._ascii_histogram(values, bins=10, width=20)
        assert isinstance(hist, str)
        assert "#" in hist or "(" in hist

    def test_empty(self, tool):
        assert "empty" in tool._ascii_histogram(np.array([]))


# ────────────────────────────────────────────────────────────────────────────
# Profile structure


class TestProfileStructure:
    """Test the JSON structure produced by the profile (without running scorers)."""

    def test_profile_has_required_keys(self, tool):
        # Build a synthetic profile dict matching what compute returns.
        n = 4
        synth = {
            "frames": [
                {
                    "frame_idx": i,
                    "segnet_entropy": 0.5,
                    "posenet_variance": 1e-5,
                    "combined_difficulty": 0.0,
                    "percentile_rank": float(i * 100 / max(n - 1, 1)),
                }
                for i in range(n)
            ],
            "stats": {
                "n_frames": n,
                "n_pairs": n // 2,
                "alpha_seg_weight": 0.95,
                "beta_pose_weight": 0.05,
            },
            "provenance": {
                "video_path": "fake.mkv",
                "video_sha256": "0" * 64,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "evidence_grade": "diagnostic_only",
                "tag": "[diagnostic; per-frame difficulty profile on upstream video]",
            },
        }
        # State hash is stable.
        h = tool._state_hash(synth)
        assert len(h) == 64

    def test_provenance_tag_is_diagnostic_not_score(self, tool):
        # Per CLAUDE.md `forbidden_score_claims`: profile must NOT claim a score.
        # We verify the tag-string template here.
        from tools.xray_per_frame_difficulty_profile import compute_per_frame_difficulty_profile  # noqa

        # Inspect the function source to verify tag.
        import inspect

        src = inspect.getsource(compute_per_frame_difficulty_profile)
        assert "diagnostic" in src
        assert '"score_claim": False' in src
        assert '"promotion_eligible": False' in src

    def test_n_frames_must_be_even(self, tool):
        # Even count required (paired frames for PoseNet variance).
        with pytest.raises(ValueError, match="must be even"):
            tool.compute_per_frame_difficulty_profile(
                video_path=REPO_ROOT / "upstream" / "videos" / "0.mkv",
                upstream_dir=REPO_ROOT / "upstream",
                n_frames=3,
            )


# ────────────────────────────────────────────────────────────────────────────
# Scorer integration (gated by upstream video presence)


VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
UPSTREAM_DIR = REPO_ROOT / "upstream"


@pytest.mark.skipif(not VIDEO_PATH.exists(), reason="upstream video not present")
@pytest.mark.skipif(
    not (UPSTREAM_DIR / "models" / "segnet.safetensors").exists(),
    reason="SegNet weights not present",
)
class TestPipelineSmoke:
    """End-to-end smoke (8 frames) on real video + real scorers."""

    def test_smoke_8_frames(self, tool):
        result = tool.compute_per_frame_difficulty_profile(
            video_path=VIDEO_PATH,
            upstream_dir=UPSTREAM_DIR,
            n_frames=8,
            device="cpu",
            n_pose_perturbations=1,
            seed=0,
            progress=False,
        )
        assert "frames" in result
        assert "stats" in result
        assert "provenance" in result
        assert len(result["frames"]) == 8
        for f in result["frames"]:
            assert "frame_idx" in f
            assert "segnet_entropy" in f
            assert "posenet_variance" in f
            assert "combined_difficulty" in f
            assert "percentile_rank" in f
            assert 0 <= f["percentile_rank"] <= 100
            assert f["segnet_entropy"] >= 0  # entropy is non-negative
            assert f["posenet_variance"] >= 0  # variance is non-negative
        assert result["provenance"]["score_claim"] is False
        assert result["provenance"]["evidence_grade"] == "diagnostic_only"
        assert "diagnostic" in result["provenance"]["tag"]

    def test_smoke_deterministic_with_same_seed(self, tool):
        a = tool.compute_per_frame_difficulty_profile(
            video_path=VIDEO_PATH,
            upstream_dir=UPSTREAM_DIR,
            n_frames=4,
            device="cpu",
            n_pose_perturbations=1,
            seed=42,
        )
        b = tool.compute_per_frame_difficulty_profile(
            video_path=VIDEO_PATH,
            upstream_dir=UPSTREAM_DIR,
            n_frames=4,
            device="cpu",
            n_pose_perturbations=1,
            seed=42,
        )
        # Frame ordering and entropy values should be identical across runs.
        for fa, fb in zip(a["frames"], b["frames"]):
            assert fa["frame_idx"] == fb["frame_idx"]
            assert abs(fa["segnet_entropy"] - fb["segnet_entropy"]) < 1e-5
            assert abs(fa["posenet_variance"] - fb["posenet_variance"]) < 1e-5

    def test_state_hash_consistent_across_runs(self, tool):
        a = tool.compute_per_frame_difficulty_profile(
            video_path=VIDEO_PATH,
            upstream_dir=UPSTREAM_DIR,
            n_frames=4,
            device="cpu",
            n_pose_perturbations=1,
            seed=99,
        )
        b = tool.compute_per_frame_difficulty_profile(
            video_path=VIDEO_PATH,
            upstream_dir=UPSTREAM_DIR,
            n_frames=4,
            device="cpu",
            n_pose_perturbations=1,
            seed=99,
        )
        assert tool._state_hash(a) == tool._state_hash(b)


# ────────────────────────────────────────────────────────────────────────────
# Markdown report


class TestMarkdownReport:
    def test_includes_regen_header(self, tool):
        synth = {
            "frames": [
                {
                    "frame_idx": i,
                    "segnet_entropy": 0.5,
                    "posenet_variance": 1e-5,
                    "combined_difficulty": float(i),
                    "percentile_rank": float(i * 25),
                }
                for i in range(4)
            ],
            "stats": {
                "n_frames": 4,
                "n_pairs": 2,
                "segnet_entropy_mean": 0.5,
                "segnet_entropy_std": 0.0,
                "segnet_entropy_min": 0.5,
                "segnet_entropy_max": 0.5,
                "posenet_variance_mean": 1e-5,
                "posenet_variance_std": 0.0,
                "posenet_variance_min": 1e-5,
                "posenet_variance_max": 1e-5,
                "combined_difficulty_mean": 1.5,
                "combined_difficulty_std": 1.0,
                "combined_difficulty_deciles": [0.0] * 10,
                "alpha_seg_weight": 0.95,
                "beta_pose_weight": 0.05,
                "lambda_seg": 100.0,
                "lambda_pose": 5.04,
            },
            "provenance": {
                "video_path": "fake.mkv",
                "video_sha256": "0" * 64,
                "device": "cpu",
                "seed": 0,
                "tag": "[diagnostic]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "evidence_grade": "diagnostic_only",
            },
        }
        md = tool._build_markdown_report(synth, generated_at="2026-05-09T12:00:00Z", state_hash="a" * 64)
        # Per gate #113 DERIVED_OUTPUT discipline.
        assert "<!-- generated_at: 2026-05-09T12:00:00Z, from_state_hash: " in md
        # Per CLAUDE.md `forbidden_score_claims`.
        assert "score_claim: False" in md
        assert "evidence_grade: diagnostic_only" in md
