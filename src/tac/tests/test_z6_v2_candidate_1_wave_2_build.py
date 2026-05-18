# SPDX-License-Identifier: MIT
"""Tests for Z6-v2 Candidate 1 Wave 2 BUILD trainer extension + recipe.

Per Phase 3 council §9 binding spec + Path B BUILD design memo §4.1 + the
Z6-v2 Candidate 1 Wave 2 BUILD landing 2026-05-17.

Coverage:
- Trainer argparse flag wiring (6 new flags)
- TIER_1_OPERATOR_REQUIRED_FLAGS extended per Catalog #151 + #168 AnnAssign
- _resolve_predictor_architecture canonical mapping
- Identity-predictor disambiguator at SAME archive bytes (Catalog #229
  paired-control marker; Council Revision #2)
- Recipe schema validation
- Catalog #270 dispatch optimization protocol PASS verdict
- Provenance contract emission
- Sister regression: Z6-v1 still works backward-compat at default depth=1
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_PATH = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/"
    "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml"
)
TRAINER_PATH = REPO_ROOT / "experiments/train_substrate_time_traveler_l5_z6.py"


# ===========================================================================
# Predictor-architecture resolver
# ===========================================================================


def test_resolver_single_layer_film_75k_returns_depth_1() -> None:
    """Default architecture preserves Z6-v1 backward compat."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    depth, hidden = z6._resolve_predictor_architecture("single_layer_film_75k")
    assert depth == 1
    assert hidden == 64


def test_resolver_multi_layer_film_depth_3_300k_returns_depth_3_hidden_96() -> None:
    """Council §9 binding spec: depth=3 hidden_dim=96 → ~300K total params."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    depth, hidden = z6._resolve_predictor_architecture("multi_layer_film_depth_3_300k")
    assert depth == 3
    assert hidden == 96


def test_resolver_unknown_architecture_raises_systemexit() -> None:
    """Unknown architecture rejected with SystemExit (Catalog #151 + #229)."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    with pytest.raises(SystemExit):
        z6._resolve_predictor_architecture("not_a_real_arch")


# ===========================================================================
# Argparse flag wiring (Phase 3 council §9 spec)
# ===========================================================================


def test_argparse_accepts_predictor_architecture_flag() -> None:
    """--predictor-architecture multi_layer_film_depth_3_300k parses."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args([
        "--output-dir", "/tmp/_test",
        "--predictor-architecture", "multi_layer_film_depth_3_300k",
    ])
    assert args.predictor_architecture == "multi_layer_film_depth_3_300k"


def test_argparse_default_predictor_architecture_is_z6_v1() -> None:
    """Default --predictor-architecture preserves Z6-v1 backward compat."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.predictor_architecture == "single_layer_film_75k"


def test_argparse_accepts_disambiguator_flag() -> None:
    """--emit-identity-predictor-disambiguator-archive parses as store_true."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args([
        "--output-dir", "/tmp/_test",
        "--emit-identity-predictor-disambiguator-archive",
    ])
    assert args.emit_identity_predictor_disambiguator_archive is True


def test_argparse_disambiguator_decision_criterion_delta_s_default() -> None:
    """Default decision threshold matches Council Revision #3 binding ΔS = 0.005."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.paired_control_disambiguator_decision_criterion_delta_s == 0.005


def test_argparse_predictor_param_count_target_default() -> None:
    """Default --predictor-param-count-target=300_000 per Council binding ceiling."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.predictor_param_count_target == 300_000


def test_argparse_paired_control_initialization_default() -> None:
    """Default paired-control marker matches Catalog #229 canonical."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert (
        args.enable_paired_control_initialization
        == "shared_modules_seed_order_matched_v2"
    )


def test_argparse_ego_source_default() -> None:
    """Default ego-source preserves Z6-v1 PoseNet projection per Atick Revision #6."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.ego_source == "posenet_projection"


# ===========================================================================
# TIER_1_OPERATOR_REQUIRED_FLAGS manifest (Catalog #151 + #168 AnnAssign)
# ===========================================================================


def test_tier1_required_flags_contains_all_6_wave_2_build_flags() -> None:
    """Phase 3 council §9 required_flags surfaced in TIER_1 manifest."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    required = {
        "--predictor-architecture",
        "--predictor-param-count-target",
        "--ego-source",
        "--enable-paired-control-initialization",
        "--emit-identity-predictor-disambiguator-archive",
        "--paired-control-disambiguator-decision-criterion-delta-s",
    }
    flags = set(z6.TIER_1_OPERATOR_REQUIRED_FLAGS.keys())
    missing = required - flags
    assert not missing, f"missing TIER_1 manifest entries: {missing}"


def test_tier1_required_flags_uses_annassign_per_catalog_168() -> None:
    """The dict is declared with type annotation (AST AnnAssign per Catalog #168)."""
    import ast
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found_annassign = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id == "TIER_1_OPERATOR_REQUIRED_FLAGS"
            ):
                found_annassign = True
                break
    assert found_annassign, (
        "TIER_1_OPERATOR_REQUIRED_FLAGS must be declared as ast.AnnAssign per "
        "Catalog #168 (bare ast.Assign is silently skipped by the AST walker)"
    )


# ===========================================================================
# Recipe schema validation
# ===========================================================================


def test_recipe_yaml_file_exists() -> None:
    assert RECIPE_PATH.is_file(), f"recipe missing at {RECIPE_PATH}"


def test_recipe_yaml_loads_and_has_required_fields() -> None:
    """Per Phase 3 council §9 spec: recipe declares all required canonical fields."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    # Identity + lifecycle
    assert recipe["name"] == "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch"
    assert recipe["lane_id"].startswith("lane_z6_v2_candidate_1_wave_2_build")
    # Per Catalog #240 + CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
    # RESEARCH-ONLY": Wave 2 BUILD recipe IS non-dispatchable until operator
    # sign-off on Phase 3 council PROCEED_WITH_REVISIONS verdict.
    assert recipe["research_only"] is True
    assert recipe["dispatch_enabled"] is False
    # Per Catalog #270 dispatch optimization protocol Tier 2/3
    assert recipe["platform"] == "modal"
    assert recipe["min_vram_gb"] >= 14
    assert recipe["min_smoke_gpu"] == "T4"
    assert recipe["video_input_strategy"] == "per_dispatch_local_copy"
    assert recipe["pyav_decode_strategy"] == "cpu_thread_async_upload"
    assert isinstance(recipe["target_modes"], list) and recipe["target_modes"]
    assert recipe["canary_status"] == "post_canary_dependent"
    assert recipe["canary_dependency"] == "time_traveler_l5_z6"
    # Catalog #167 smoke-before-full
    assert recipe["smoke_before_full"] is True
    # Catalog #309 horizon_class
    assert recipe["horizon_class"] == "frontier_pursuit"
    # Catalog #220 + #272 distinguishing-feature integration contract
    assert recipe["distinguishing_feature_name"] == "multi_layer_film_predictor_depth_3"
    assert recipe["distinguishing_bytes_path"] == "predictor_blob"
    # Predicted band matches Path B design memo §4.1 + Phase 3 council §5
    assert recipe["predicted_band"] == [0.13, 0.17]
    # Cost band Wave 2 envelope
    cost_band = recipe["cost_band"]
    assert cost_band["epochs"] == 100
    assert cost_band["predicted_cost_usd"] == 3.0


def test_recipe_yaml_declares_all_6_tier1_required_flags() -> None:
    """Per Catalog #151: operator_required_flags must enumerate every Wave 2 flag."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    required = recipe["operator_required_flags"]["TIER_1_OPERATOR_REQUIRED_FLAGS"]
    expected = {
        "--enable-paired-control-initialization",
        "--emit-identity-predictor-disambiguator-archive",
        "--predictor-architecture",
        "--predictor-param-count-target",
        "--ego-source",
        "--paired-control-disambiguator-decision-criterion-delta-s",
    }
    missing = expected - set(required)
    assert not missing, f"recipe operator_required_flags missing: {missing}"


def test_recipe_yaml_env_overrides_thread_all_tier1_required_flags() -> None:
    """Per Catalog #151 env→CLI ladder + Catalog #152 required-input file."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    env = recipe["env_overrides"]
    for required_env in (
        "Z6_PREDICTOR_ARCHITECTURE",
        "Z6_PREDICTOR_PARAM_COUNT_TARGET",
        "Z6_EGO_SOURCE",
        "Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION",
        "Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE",
        "Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S",
    ):
        assert required_env in env, f"env_overrides missing {required_env}"
    # The canonical Wave 2 architecture is multi_layer_film_depth_3_300k
    assert env["Z6_PREDICTOR_ARCHITECTURE"] == "multi_layer_film_depth_3_300k"


def test_recipe_yaml_dispatch_blockers_enforce_pre_dispatch_gate() -> None:
    """Wave 2 BUILD recipe MUST list all 4 dispatch_blockers per Phase 3 §17."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    blockers = recipe["dispatch_blockers"]
    assert any("operator_sign_off" in b for b in blockers), (
        "missing operator_sign_off blocker (Phase 3 §17 op-routable #1)"
    )
    assert any("c6_ibps_outcome" in b for b in blockers), (
        "missing c6_ibps_outcome blocker (Phase 3 §9 conditional)"
    )
    assert any("catalog_167_smoke_before_full" in b for b in blockers), (
        "missing Catalog #167 smoke-before-full blocker"
    )
    assert any("paired_cpu_cuda_empirical_anchor" in b for b in blockers), (
        "missing paired CPU/CUDA blocker per Catalog #220"
    )


# ===========================================================================
# Catalog #270 dispatch optimization protocol PASS
# ===========================================================================


def test_recipe_passes_catalog_270_dispatch_optimization_protocol() -> None:
    """Per CLAUDE.md "Production-hardened dispatch optimization protocol" non-negotiable:
    Tier 1 + Tier 2 + Tier 3 all PASS; 0 blockers; overall_pass: true."""
    tool = REPO_ROOT / "tools/canonical_dispatch_optimization_protocol.py"
    if not tool.is_file():
        pytest.skip(f"protocol tool missing at {tool}")
    result = subprocess.run(
        [
            sys.executable, str(tool),
            "--trainer", str(TRAINER_PATH),
            "--recipe", "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch",
            "--json",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, f"protocol returned rc={result.returncode}; stderr={result.stderr}"
    verdict = json.loads(result.stdout)
    assert verdict["overall_pass"] is True, (
        f"Catalog #270 protocol FAILED for Z6-v2 Candidate 1 Wave 2 recipe; "
        f"blockers: {verdict.get('blockers', [])}"
    )
    # Per-tier zero-blockers contract
    assert verdict["tier1"]["blockers"] == []
    assert verdict["tier2"]["blockers"] == []
    assert verdict["tier3"]["blockers"] == []


# ===========================================================================
# Identity-predictor disambiguator at SAME archive bytes (Council Revision #2)
# ===========================================================================


def test_smoke_disambiguator_emits_two_archives_at_same_bytes(tmp_path: Path) -> None:
    """Per Phase 3 council Revision #2 + Catalog #229 paired-control marker:
    --emit-identity-predictor-disambiguator-archive emits BOTH archives so the
    Wave 2 disambiguator probe can compute ΔS = full_FiLM_score - identity_score
    at the SAME archive bytes (same encoder + decoder + latent_init + residuals
    + ego_motion; ONLY identity_predictor=True changes)."""
    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--output-dir", str(tmp_path),
            "--smoke",
            "--device", "cpu",
            "--predictor-architecture", "multi_layer_film_depth_3_300k",
            "--emit-identity-predictor-disambiguator-archive",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"smoke rc={result.returncode}; stderr={result.stderr[-500:]}"
    )
    # Both archives present
    full_archive = tmp_path / "0.bin"
    disambiguator_archive = tmp_path / "0_identity_predictor_disambiguator.bin"
    assert full_archive.is_file(), "full-predictor archive 0.bin missing"
    assert disambiguator_archive.is_file(), (
        "identity-predictor disambiguator archive missing"
    )
    # Stats record the disambiguator emission
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["emit_identity_predictor_disambiguator_archive"] is True
    assert stats["identity_predictor_disambiguator_archive_bytes"] is not None
    assert stats["identity_predictor_disambiguator_archive_bytes"] > 0
    # Council Revision #3 decision criterion ΔS = 0.005
    assert stats["paired_control_disambiguator_decision_criterion_delta_s"] == 0.005
    # Wave 2 BUILD architecture recorded
    assert stats["predictor_architecture"] == "multi_layer_film_depth_3_300k"
    assert stats["predictor_depth"] == 3


def test_smoke_default_does_not_emit_disambiguator(tmp_path: Path) -> None:
    """Backward compat: omitting --emit-identity-predictor-disambiguator-archive
    preserves Z6-v1 single-archive behavior."""
    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--output-dir", str(tmp_path),
            "--smoke",
            "--device", "cpu",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"smoke rc={result.returncode}; stderr={result.stderr}"
    full_archive = tmp_path / "0.bin"
    disambiguator_archive = tmp_path / "0_identity_predictor_disambiguator.bin"
    assert full_archive.is_file()
    assert not disambiguator_archive.is_file(), (
        "disambiguator emitted without --emit-identity-predictor-disambiguator-archive"
    )


# ===========================================================================
# Sister regression: Z6-v1 backward compat
# ===========================================================================


def test_z6_v1_smoke_still_works_at_default_depth_1(tmp_path: Path) -> None:
    """Default --predictor-architecture=single_layer_film_75k preserves Z6-v1.

    Per CLAUDE.md "Subagent coherence-by-default": the Wave 2 BUILD extension
    MUST NOT break the Z6-v1 dispatch path (sister `lane_time_traveler_l5_z6_l1_
    scaffold_substrate_build_20260516` still depends on the default behavior).
    """
    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--output-dir", str(tmp_path),
            "--smoke",
            "--device", "cpu",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"Z6-v1 default smoke regressed; rc={result.returncode}; "
        f"stderr={result.stderr[-500:]}"
    )
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["predictor_depth"] == 1
    assert stats["predictor_architecture"] == "single_layer_film_75k"
    assert stats["emit_identity_predictor_disambiguator_archive"] is False
