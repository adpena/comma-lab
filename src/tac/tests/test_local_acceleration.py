# SPDX-License-Identifier: MIT
"""Tests for tac.local_acceleration package.

Per CLAUDE.md "Tier 1 engineering" + Carmack MVP-first 5-step discipline:
test surface covers canonical contract invariants (non-promotable triple,
classification correctness, MLX availability check, Provenance markers)
without dispatching paid GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.local_acceleration import (
    EVIDENCE_GRADE_MLX,
    EVIDENCE_GRADE_METAL,
    EVIDENCE_TAG_MLX,
    EVIDENCE_TAG_METAL,
    SCHEMA_VERSION,
)
from tac.local_acceleration.mlx_integration import (
    MLX_AVAILABLE,
    MLXTrainingResult,
    build_mlx_training_result,
    is_mlx_available,
    mlx_smoke_test,
)
from tac.local_acceleration.routability_audit import (
    M5_MAX_UNIFIED_MEMORY_BYTES,
    M5_MAX_USABLE_WORKING_SET_BYTES,
    MLX_TRAINABLE_VRAM_CEILING_GB,
    MPS_TRAINABLE_VRAM_CEILING_GB,
    PAID_VRAM_CEILING_GB,
    SubstrateRoutabilityClass,
    SubstrateRoutabilityVerdict,
    audit_all_substrate_recipes,
    classify_recipe_routability,
    verdict_summary_text,
    write_audit_manifest,
)


class TestPackageContract:
    """Canonical contract invariants for the local_acceleration package."""

    def test_schema_version_pinned(self):
        assert SCHEMA_VERSION == "local_acceleration.v1"

    def test_evidence_grade_mlx_canonical(self):
        assert EVIDENCE_GRADE_MLX == "macOS-MLX-research-signal"
        assert EVIDENCE_TAG_MLX == "[macOS-MLX research-signal]"

    def test_evidence_grade_metal_canonical(self):
        assert EVIDENCE_GRADE_METAL == "macOS-Metal-research-signal"
        assert EVIDENCE_TAG_METAL == "[macOS-Metal research-signal]"


class TestMLXIntegration:
    """MLX framework integration scaffold tests."""

    def test_is_mlx_available_returns_bool(self):
        result = is_mlx_available()
        assert isinstance(result, bool)

    def test_mlx_available_constant_matches_runtime(self):
        # On Apple Silicon with MLX installed, MLX_AVAILABLE matches is_mlx_available().
        assert MLX_AVAILABLE == is_mlx_available()

    def test_mlx_smoke_test_returns_dict(self):
        result = mlx_smoke_test()
        assert isinstance(result, dict)
        assert "available" in result

    def test_build_mlx_training_result_non_promotable_by_default(self):
        result = build_mlx_training_result(
            substrate_id="test_substrate",
            run_id="r1",
            epochs_completed=10,
            final_proxy_loss=0.5,
            wall_seconds=12.3,
        )
        # Per Catalog #1/#192/#317: non-promotable triple is FROZEN.
        assert result.score_claim is False
        assert result.promotion_eligible is False
        assert result.ready_for_exact_eval_dispatch is False
        assert result.evidence_grade == EVIDENCE_GRADE_MLX
        assert result.evidence_tag == EVIDENCE_TAG_MLX

    def test_build_mlx_training_result_carries_canonical_blockers(self):
        result = build_mlx_training_result(
            substrate_id="test_substrate",
            run_id="r1",
            epochs_completed=10,
            final_proxy_loss=0.5,
            wall_seconds=12.3,
        )
        assert len(result.blockers) >= 4
        joined = " ".join(result.blockers)
        assert "not_cuda_auth_eval" in joined
        assert "not_a_11_contest_compliant_axis" in joined

    def test_build_mlx_training_result_validates_inputs(self):
        with pytest.raises(ValueError, match="substrate_id"):
            build_mlx_training_result(
                substrate_id="",
                run_id="r1",
                epochs_completed=10,
                final_proxy_loss=0.5,
                wall_seconds=12.3,
            )
        with pytest.raises(ValueError, match="run_id"):
            build_mlx_training_result(
                substrate_id="test",
                run_id="",
                epochs_completed=10,
                final_proxy_loss=0.5,
                wall_seconds=12.3,
            )
        with pytest.raises(ValueError, match="epochs_completed"):
            build_mlx_training_result(
                substrate_id="test",
                run_id="r1",
                epochs_completed=-1,
                final_proxy_loss=0.5,
                wall_seconds=12.3,
            )
        with pytest.raises(ValueError, match="wall_seconds"):
            build_mlx_training_result(
                substrate_id="test",
                run_id="r1",
                epochs_completed=10,
                final_proxy_loss=0.5,
                wall_seconds=-1.0,
            )

    def test_mlx_training_result_as_dict_roundtrip(self):
        result = build_mlx_training_result(
            substrate_id="atw_codec_v1",
            run_id="r1",
            epochs_completed=50,
            final_proxy_loss=0.123,
            wall_seconds=45.6,
            archive_bytes_estimate=12345,
        )
        d = result.as_dict()
        assert d["substrate_id"] == "atw_codec_v1"
        assert d["epochs_completed"] == 50
        assert d["final_proxy_loss"] == 0.123
        assert d["score_claim"] is False
        assert d["promotion_eligible"] is False


class TestRoutabilityAudit:
    """Per-substrate routability classification tests."""

    def test_constants_pinned(self):
        assert M5_MAX_UNIFIED_MEMORY_BYTES == 137_438_953_472
        assert M5_MAX_USABLE_WORKING_SET_BYTES == 115_448_725_504
        assert MLX_TRAINABLE_VRAM_CEILING_GB == 16
        assert MPS_TRAINABLE_VRAM_CEILING_GB == 40
        assert PAID_VRAM_CEILING_GB == 80

    def test_classification_classes_canonical(self):
        classes = SubstrateRoutabilityClass.all_classes()
        assert "LOCAL_MLX_TRAINABLE" in classes
        assert "LOCAL_MPS_TRAINABLE" in classes
        assert "LOCAL_CPU_PROXY" in classes
        assert "PAID_ONLY" in classes
        assert "UNKNOWN" in classes

    def test_verdict_canonical_provenance(self):
        v = SubstrateRoutabilityVerdict(
            substrate_id="test",
            recipe_path="/tmp/test.yaml",
            classification=SubstrateRoutabilityClass.LOCAL_MLX_TRAINABLE,
            min_vram_gb_declared=16,
            rationale="test",
            estimated_cost_compression_usd=3.0,
        )
        # Canonical Provenance triple frozen per Catalog #287/#323.
        assert v.evidence_grade == "local-routability-audit-advisory"
        assert v.promotable is False
        assert v.score_claim is False
        # Blockers reference Catalog #192/#317.
        joined = " ".join(v.blockers)
        assert "catalog_192_317" in joined.lower()

    def test_classify_recipe_routability_missing_file(self, tmp_path: Path):
        verdict = classify_recipe_routability(tmp_path / "nonexistent.yaml")
        assert verdict.classification == SubstrateRoutabilityClass.UNKNOWN

    def test_classify_recipe_local_mlx_trainable(self, tmp_path: Path):
        recipe = tmp_path / "substrate_test_modal_t4_dispatch.yaml"
        recipe.write_text(
            "platform: modal\nmin_vram_gb: 16\ndispatch_enabled: true\n"
        )
        verdict = classify_recipe_routability(recipe)
        assert verdict.classification == SubstrateRoutabilityClass.LOCAL_MLX_TRAINABLE
        assert verdict.min_vram_gb_declared == 16
        assert verdict.estimated_cost_compression_usd > 0

    def test_classify_recipe_local_mps_trainable(self, tmp_path: Path):
        recipe = tmp_path / "substrate_test_modal_a100_dispatch.yaml"
        recipe.write_text(
            "platform: modal\nmin_vram_gb: 40\ndispatch_enabled: true\n"
        )
        verdict = classify_recipe_routability(recipe)
        assert verdict.classification == SubstrateRoutabilityClass.LOCAL_MPS_TRAINABLE
        assert verdict.min_vram_gb_declared == 40

    def test_classify_recipe_paid_only_high_vram(self, tmp_path: Path):
        recipe = tmp_path / "substrate_test_modal_h100_dispatch.yaml"
        recipe.write_text(
            "platform: modal\nmin_vram_gb: 80\ndispatch_enabled: true\n"
        )
        verdict = classify_recipe_routability(recipe)
        # 80GB exceeds MPS comfort (40GB ceiling).
        assert verdict.classification == SubstrateRoutabilityClass.PAID_ONLY

    def test_classify_recipe_unknown_no_vram(self, tmp_path: Path):
        recipe = tmp_path / "substrate_test_modal_t4_dispatch.yaml"
        recipe.write_text("platform: modal\ndispatch_enabled: true\n")
        verdict = classify_recipe_routability(recipe)
        assert verdict.classification == SubstrateRoutabilityClass.UNKNOWN
        assert verdict.min_vram_gb_declared is None

    def test_classify_recipe_mlx_incompatible_falls_back_to_mps(
        self, tmp_path: Path
    ):
        recipe = tmp_path / "substrate_test_modal_t4_dispatch.yaml"
        recipe.write_text(
            "platform: modal\nmin_vram_gb: 16\ndispatch_enabled: true\n"
            "# uses NVDEC fn.experimental.inputs.video\n"
        )
        trainer = tmp_path / "trainer.py"
        trainer.write_text("import torch.distributed\nimport NVDEC_thing\n")
        verdict = classify_recipe_routability(recipe, trainer)
        # 16GB fits MPS, MLX-incompatible primitives → MPS fallback.
        assert verdict.classification == SubstrateRoutabilityClass.LOCAL_MPS_TRAINABLE

    def test_audit_all_substrate_recipes_empty_dir(self, tmp_path: Path):
        # No .omx/operator_authorize_recipes/ → empty list.
        verdicts = audit_all_substrate_recipes(tmp_path)
        assert verdicts == []

    def test_audit_all_substrate_recipes_skips_dup_substrate(
        self, tmp_path: Path
    ):
        recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
        recipes_dir.mkdir(parents=True)
        for gpu in ["t4", "a100", "h100"]:
            (recipes_dir / f"substrate_test_modal_{gpu}_dispatch.yaml").write_text(
                f"platform: modal\nmin_vram_gb: 16\n"
            )
        verdicts = audit_all_substrate_recipes(tmp_path)
        # Dedup by canonical substrate_id → 1 verdict.
        assert len(verdicts) == 1

    def test_verdict_summary_text_empty(self):
        text = verdict_summary_text([])
        assert "No substrate recipes scanned" in text

    def test_verdict_summary_text_includes_non_promotable_disclaimer(
        self, tmp_path: Path
    ):
        recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "substrate_test_modal_t4_dispatch.yaml").write_text(
            "platform: modal\nmin_vram_gb: 16\n"
        )
        verdicts = audit_all_substrate_recipes(tmp_path)
        text = verdict_summary_text(verdicts)
        assert "NON-PROMOTABLE" in text
        assert "Catalog #1/#192/#317" in text

    def test_write_audit_manifest_creates_file(self, tmp_path: Path):
        verdict = SubstrateRoutabilityVerdict(
            substrate_id="test",
            recipe_path="/tmp/test.yaml",
            classification=SubstrateRoutabilityClass.LOCAL_MLX_TRAINABLE,
            min_vram_gb_declared=16,
            rationale="test",
            estimated_cost_compression_usd=3.0,
        )
        out = tmp_path / "manifest.json"
        write_audit_manifest([verdict], out)
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["schema_version"] == "local_leverage_routability_audit.v1"
        assert len(payload["verdicts"]) == 1
        assert payload["verdicts"][0]["promotable"] is False


class TestLiveRepoSmoke:
    """Live-repo regression guards (run against actual recipes)."""

    def test_live_repo_audit_finds_substrates(self):
        repo_root = Path(__file__).resolve().parents[3]
        recipes_dir = repo_root / ".omx" / "operator_authorize_recipes"
        if not recipes_dir.exists():
            pytest.skip("Live recipes dir not present (CI environment)")
        verdicts = audit_all_substrate_recipes(repo_root)
        # Per audit 2026-05-21: ~75 substrate trainers, ~98 recipes scanned.
        assert len(verdicts) >= 50, (
            f"Expected >= 50 substrate verdicts; got {len(verdicts)}. "
            "Has the recipe corpus shrunk dramatically?"
        )

    def test_live_repo_audit_has_local_routable_majority(self):
        repo_root = Path(__file__).resolve().parents[3]
        recipes_dir = repo_root / ".omx" / "operator_authorize_recipes"
        if not recipes_dir.exists():
            pytest.skip("Live recipes dir not present (CI environment)")
        verdicts = audit_all_substrate_recipes(repo_root)
        local_routable_count = sum(
            1
            for v in verdicts
            if v.classification
            in (
                SubstrateRoutabilityClass.LOCAL_MLX_TRAINABLE,
                SubstrateRoutabilityClass.LOCAL_MPS_TRAINABLE,
                SubstrateRoutabilityClass.LOCAL_CPU_PROXY,
            )
        )
        # Per audit 2026-05-21: 97 of 98 substrates classify as local-routable
        # (61 MLX + 36 MPS); only 1 UNKNOWN (mlx_mask_renderer no min_vram).
        # Sanity guard: >= 80% are local-routable.
        if len(verdicts) > 0:
            assert local_routable_count >= len(verdicts) * 0.8
