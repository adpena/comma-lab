"""Tests for tools/build_a1_frame_conditional_codec_variant.py +
tools/probe_frame_conditional_quantization_disambiguator.py.

Per HNeRV parity discipline lesson 11 (no-op detector): every variant must
prove per-decile bytes change rendered output (i.e., bytes differ from A1).

Per CLAUDE.md `forbidden_score_claims`: every manifest must carry
``score_claim=False`` and ``ready_for_exact_eval_dispatch=False``.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_TOOL = REPO_ROOT / "tools" / "build_a1_frame_conditional_codec_variant.py"
PROBE_TOOL = REPO_ROOT / "tools" / "probe_frame_conditional_quantization_disambiguator.py"


def _load_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_tool():
    return _load_tool(BUILD_TOOL, "build_a1_frame_conditional_codec_variant")


@pytest.fixture(scope="module")
def probe_tool():
    return _load_tool(PROBE_TOOL, "probe_frame_conditional_quantization_disambiguator")


@pytest.fixture
def fake_a1_archive_bytes() -> bytes:
    """Minimal A1-shaped archive with a single 'x' member."""
    payload = (np.random.default_rng(0).bytes(8000))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("x")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, payload)
    return buf.getvalue()


@pytest.fixture
def fake_difficulty_profile_dict() -> dict:
    rng = np.random.default_rng(1)
    return {
        "frames": [
            {
                "frame_idx": i,
                "segnet_entropy": float(rng.uniform(0, 2)),
                "posenet_variance": float(rng.uniform(0, 1e-4)),
                "combined_difficulty": float(rng.normal()),
                "percentile_rank": float(i * 100 / 99),
            }
            for i in range(100)  # 50 pairs
        ],
        "stats": {},
        "provenance": {},
    }


# ────────────────────────────────────────────────────────────────────────────
# Pure-function tests


class TestLoadDifficultyProfile:
    def test_loads_valid_profile(self, build_tool, fake_difficulty_profile_dict, tmp_path):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps(fake_difficulty_profile_dict))
        out = build_tool.load_difficulty_profile(json_path)
        assert isinstance(out, dict)
        assert len(out) == 100
        assert all(isinstance(k, int) for k in out.keys())
        assert all(isinstance(v, float) for v in out.values())

    def test_rejects_missing_frames_key(self, build_tool, tmp_path):
        json_path = tmp_path / "p.json"
        json_path.write_text(json.dumps({"stats": {}}))
        with pytest.raises(ValueError, match="missing 'frames'"):
            build_tool.load_difficulty_profile(json_path)


class TestReducePerPair:
    def test_basic_reduce(self, build_tool):
        per_frame = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        per_pair = build_tool.reduce_per_frame_to_per_pair(per_frame, n_pairs=2)
        assert per_pair == {0: 1.5, 1: 3.5}

    def test_rejects_too_few_frames(self, build_tool):
        per_frame = {0: 1.0, 1: 2.0}
        with pytest.raises(ValueError, match="need 2.*pairs"):
            build_tool.reduce_per_frame_to_per_pair(per_frame, n_pairs=3)


class TestSynthesizeA1LatentsProxy:
    def test_deterministic(self, build_tool, fake_a1_archive_bytes):
        a = build_tool.synthesize_a1_latents_proxy(fake_a1_archive_bytes, n_pairs=10, latent_dim=28)
        b = build_tool.synthesize_a1_latents_proxy(fake_a1_archive_bytes, n_pairs=10, latent_dim=28)
        np.testing.assert_array_equal(a, b)

    def test_shape_correct(self, build_tool, fake_a1_archive_bytes):
        out = build_tool.synthesize_a1_latents_proxy(fake_a1_archive_bytes, n_pairs=5, latent_dim=12)
        assert out.shape == (5, 12)
        assert out.dtype == np.float32

    def test_different_seeds_differ(self, build_tool, fake_a1_archive_bytes):
        a = build_tool.synthesize_a1_latents_proxy(fake_a1_archive_bytes, n_pairs=10, latent_dim=28, seed=0)
        b = build_tool.synthesize_a1_latents_proxy(fake_a1_archive_bytes, n_pairs=10, latent_dim=28, seed=1)
        assert not np.array_equal(a, b)


# ────────────────────────────────────────────────────────────────────────────
# Variant build


class TestBuildVariant:
    def test_basic_build(self, build_tool, fake_a1_archive_bytes, tmp_path):
        per_frame = {i: float(i) for i in range(100)}
        per_pair = build_tool.reduce_per_frame_to_per_pair(per_frame, n_pairs=50)
        variant = {
            "tag": "V_test",
            "strategy": "per-decile-tied",
            "bit_budget_per_decile": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "rationale": "test variant",
        }
        out_dir = tmp_path / "v"
        manifest = build_tool.build_variant(
            a1_archive_bytes=fake_a1_archive_bytes,
            a1_payload=b"\x00" * 8000,
            difficulty_profile=per_pair,
            variant_spec=variant,
            output_dir=out_dir,
            n_pairs=50,
            latent_dim=28,
            seed=0,
        )
        # File outputs.
        assert (out_dir / "candidate_archive.zip").exists()
        assert (out_dir / "build_manifest.json").exists()
        assert (out_dir / "no_op_proof.json").exists()
        # Manifest checks.
        assert manifest["score_claim"] is False
        assert manifest["ready_for_exact_eval_dispatch"] is False
        assert manifest["evidence_grade"] == "byte_anchor_proxy"
        assert manifest["lane_class"] == "substrate_engineering"
        assert manifest["candidate_archive_bytes"] > 0
        # 8-field declaration present.
        for field in (
            "archive_grammar",
            "parser_section_manifest",
            "inflate_runtime_loc_budget",
            "runtime_dep_closure",
            "export_format",
            "score_aware_loss",
            "bolt_on_loc_budget",
            "no_op_detector_planned",
        ):
            assert field in manifest, f"missing field: {field}"

    def test_no_op_detector_passes_for_non_uniform_variant(self, build_tool, fake_a1_archive_bytes, tmp_path):
        per_frame = {i: float(i) for i in range(100)}
        per_pair = build_tool.reduce_per_frame_to_per_pair(per_frame, n_pairs=50)
        variant = {
            "tag": "V_aggressive",
            "strategy": "per-decile-tied",
            "bit_budget_per_decile": [2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
            "rationale": "aggressive skew",
        }
        out_dir = tmp_path / "v"
        manifest = build_tool.build_variant(
            a1_archive_bytes=fake_a1_archive_bytes,
            a1_payload=b"\x00" * 8000,
            difficulty_profile=per_pair,
            variant_spec=variant,
            output_dir=out_dir,
            n_pairs=50,
            latent_dim=28,
            seed=0,
        )
        # The no-op detector is asking: did the bytes change vs A1? Yes,
        # because we re-encoded with a totally different codec.
        proof = json.loads((out_dir / "no_op_proof.json").read_text())
        assert proof["payload_changed"] is True
        assert proof["no_op_detector_passed"] is True
        assert manifest["score_affecting_payload_changed"] is True

    def test_dispatch_blockers_named(self, build_tool, fake_a1_archive_bytes, tmp_path):
        per_frame = {i: float(i) for i in range(100)}
        per_pair = build_tool.reduce_per_frame_to_per_pair(per_frame, n_pairs=50)
        variant = {
            "tag": "V_test",
            "strategy": "per-decile-tied",
            "bit_budget_per_decile": [4] * 10,
            "rationale": "test",
        }
        manifest = build_tool.build_variant(
            a1_archive_bytes=fake_a1_archive_bytes,
            a1_payload=b"\x00" * 8000,
            difficulty_profile=per_pair,
            variant_spec=variant,
            output_dir=tmp_path / "v",
            n_pairs=50,
            latent_dim=28,
            seed=0,
        )
        # Per CLAUDE.md kill-as-last-resort: byte-anchor variants are NOT
        # dispatchable until real-archive empirical lands.
        assert "byte_anchor_proxy_only_no_real_a1_latents" in manifest["dispatch_blockers"]


class TestDispatchPlan:
    def test_dispatch_plan_has_three_stages(self, build_tool, tmp_path):
        manifests = [
            {"variant_tag": "V1", "candidate_archive_bytes": 1000},
            {"variant_tag": "V2", "candidate_archive_bytes": 1100},
        ]
        plan = build_tool.emit_dispatch_plan(output_root=tmp_path, variant_manifests=manifests)
        assert plan["n_variants"] == 2
        stages = plan["stages"]
        assert len(stages) == 3
        assert stages[0]["stage"] == "1_m5_max_parallel_sweep"
        assert stages[1]["stage"] == "2_gha_contest_cpu_dispatch_top_winners"
        assert stages[2]["stage"] == "3_contest_cuda_validation_top_1_to_2"
        # Substrates correctly tagged.
        assert "macOS-CPU" in stages[0]["substrate"]
        assert "contest-CPU" in stages[1]["substrate"]
        assert "contest-CUDA" in stages[2]["substrate"]

    def test_dispatch_plan_predicts_cost(self, build_tool, tmp_path):
        manifests = [
            {"variant_tag": f"V{i}", "candidate_archive_bytes": 1000 + i}
            for i in range(5)
        ]
        plan = build_tool.emit_dispatch_plan(output_root=tmp_path, variant_manifests=manifests)
        # Expected: 0.20 * 2 (contest-CUDA) = 0.40.
        assert plan["predicted_total_cost_dollars"] == pytest.approx(0.40, abs=0.01)


# ────────────────────────────────────────────────────────────────────────────
# Probe disambiguator


class TestProbeDisambiguator:
    def test_probe_basic(self, probe_tool, fake_a1_archive_bytes, tmp_path):
        per_pair = {i: float(i) for i in range(50)}
        latents = np.random.default_rng(0).normal(size=(50, 28)).astype(np.float32)
        result = probe_tool.probe_strategy(
            strategy="per-decile-tied",
            bit_budget_per_decile=(4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
            difficulty_profile=per_pair,
            latents=latents,
            a1_archive_bytes=8000,
        )
        assert result["strategy"] == "per-decile-tied"
        assert result["encoded_bytes"] > 0
        assert result["estimated_matches_actual"] is True
        assert result["encode_determinism_ok"] is True
        assert 0 <= result["reconstruction_rel_err_max"] <= 1
        assert isinstance(result["encoded_sha256"], str)
        assert len(result["encoded_sha256"]) == 64

    def test_arbitrate_returns_winner(self, probe_tool):
        probes = [
            {"strategy": "uniform", "archive_bytes_delta_vs_a1": -100, "reconstruction_rel_err_max": 0.05},
            {"strategy": "per-frame", "archive_bytes_delta_vs_a1": -120, "reconstruction_rel_err_max": 0.10},
            {"strategy": "per-decile-tied", "archive_bytes_delta_vs_a1": -110, "reconstruction_rel_err_max": 0.06},
        ]
        result = probe_tool.arbitrate(probes)
        # uniform: -100 + 50*0.05 = -97.5
        # per-frame: -120 + 50*0.10 = -115
        # per-decile-tied: -110 + 50*0.06 = -107
        # Winner = per-frame (lowest).
        assert result["winner"] == "per-frame"
        assert result["ranked_strategies"][0] == "per-frame"
        assert "diagnostic" in result["tag"]

    def test_arbitrate_emits_method_explanation(self, probe_tool):
        result = probe_tool.arbitrate([
            {"strategy": "uniform", "archive_bytes_delta_vs_a1": 0, "reconstruction_rel_err_max": 0.0}
        ])
        assert "byte" in result["arbitration_method"].lower()
        assert "rel_err" in result["arbitration_method"]


# ────────────────────────────────────────────────────────────────────────────
# Module integration


class TestModuleIntegration:
    def test_default_variants_catalog_size(self, build_tool):
        assert len(build_tool.DEFAULT_VARIANTS) == 5

    def test_each_default_variant_well_formed(self, build_tool):
        for v in build_tool.DEFAULT_VARIANTS:
            assert "tag" in v
            assert "strategy" in v
            assert "bit_budget_per_decile" in v
            assert "rationale" in v
            assert v["strategy"] in ("uniform", "per-frame", "per-decile-tied")
            assert len(v["bit_budget_per_decile"]) == 10
            for q in v["bit_budget_per_decile"]:
                assert 1 <= q <= 8
