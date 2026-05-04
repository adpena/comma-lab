from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments/profile_pr95_hnerv_muon_intake.py"
INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr95_intake_20260504_codex"
ARCHIVE = INTAKE_DIR / "archive.zip"
SOURCE_DIR = INTAKE_DIR / "pr95_src/submissions/hnerv_muon"
STATIC_INTAKE = INTAKE_DIR / "pr95_static_intake.json"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_pr95_hnerv_muon_intake", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_pr95_intake() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in (ARCHIVE, SOURCE_DIR) if not path.exists()]
    if missing:
        pytest.skip(f"PR95 HNeRV/Muon intake fixture missing: {', '.join(missing)}")


def test_pr95_hnerv_muon_profile_parses_archive_source_and_readiness():
    require_pr95_intake()
    module = load_module()

    profile = module.build_profile(ARCHIVE, SOURCE_DIR, STATIC_INTAKE)

    assert profile["schema"] == "pr95_hnerv_muon_static_intake_profile_v1"
    assert profile["archive_anatomy"]["members"][0]["name"] == "0.bin"
    assert profile["archive_anatomy"]["archive_bytes"] == ARCHIVE.stat().st_size
    assert profile["hnerv_muon_blob"]["member_format"] == "hnerv_muon_codec_v1"
    assert profile["hnerv_muon_blob"]["decoder"]["total_params"] > 200_000
    assert profile["hnerv_muon_blob"]["decoder"]["muon_partition_params"] > 0
    assert profile["hnerv_muon_blob"]["latents"]["latent_dim"] == 28
    assert profile["source_intake"]["model_defaults"]["base_channels"] == 36
    assert len(profile["source_intake"]["training_stages"]) == 8
    assert profile["dispatch_readiness"]["ready_for_dispatch"] is False
    assert profile["dispatch_readiness"]["fail_closed"] is True
    assert any("Replay exact eval" in item for item in profile["dispatch_readiness"]["required_before_score_claims"])


def test_pr95_hnerv_muon_hooks_include_prior_repo_gems():
    require_pr95_intake()
    module = load_module()

    profile = module.build_profile(ARCHIVE, SOURCE_DIR, STATIC_INTAKE)
    hooks = [item["hook"] for item in profile["immediate_improvement_hypotheses"]]

    assert hooks[:5] == [
        "RAFT/ego-motion/foveation latent bases",
        "Cool-Chic/C3/wavelet residual bases",
        "Fridrich/Lagrangian hard-pair weighting",
        "engineered corrections and pixel-diff sparse atoms",
        "self-compression entropy objectives",
    ]


def test_pr95_hnerv_muon_score_terms_recompute_static_inputs():
    require_pr95_intake()
    if not STATIC_INTAKE.exists():
        pytest.skip("PR95 static intake JSON missing; cannot test score-term recomputation")
    module = load_module()

    profile = module.build_profile(ARCHIVE, SOURCE_DIR, STATIC_INTAKE)
    score_terms = profile["score_term_math"]["score_terms_from_static_intake"]

    assert score_terms is not None
    assert score_terms["matches_recorded_recomputed_score"] is True
    recomputed = score_terms["recomputed"]
    assert recomputed["seg_component"] > 0
    assert recomputed["pose_component"] > 0
    assert recomputed["rate_component"] > 0
