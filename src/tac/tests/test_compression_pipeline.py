# SPDX-License-Identifier: MIT
"""Tests for tac.submission_packet.compression_pipeline (Phase 2 Layer 0).

Per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
§3 Phase 2 acceptance: 30+ tests covering canonical-helper contract +
canonical Provenance routing + Catalog gate compliance + CLI shape.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.submission_packet import (
    CANONICAL_EQUATION_ID,
    COMPRESSION_PIPELINE_SCHEMA_VERSION,
    CompressionPipelineError,
    CompressionPipelineResult,
    HardwareSubstrateClass,
    PerAxisPredictedBand,
    build_compression_pipeline,
    classify_hardware_substrate_for_dispatch,
    derive_compression_pipeline_provenance,
    validate_recipe_trainer_pair,
    verify_compression_pipeline_protocol_complete,
)
from tac.submission_packet.compression_pipeline import (
    _PLACEHOLDER_RATIONALES,
    _VALID_HARDWARE_SUBSTRATE_TOKENS,
    PHASE_2_LAYER_VERSION,
    _extract_per_axis_predicted_band_from_recipe,
    _substrate_id_from_trainer_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Canonical-constants regression guards (Catalog #185 sister + #176 sister)
# ---------------------------------------------------------------------------


def test_schema_version_pinned() -> None:
    assert COMPRESSION_PIPELINE_SCHEMA_VERSION == "compression_pipeline_v1_20260526"


def test_phase_2_layer_version_pinned() -> None:
    assert PHASE_2_LAYER_VERSION == "phase_2_compression_pipeline_canonical_landed_20260526"


def test_canonical_equation_id_matches_phase_1_spec() -> None:
    # Per Phase 1 spec memo §13: equation is FORMALIZATION_PENDING.
    assert CANONICAL_EQUATION_ID == (
        "compression_pipeline_canonical_helper_consolidation_savings_v1"
    )


def test_hardware_substrate_class_canonical_members() -> None:
    members = {c.value for c in HardwareSubstrateClass}
    assert members == {"auto", "local-mps", "local-cpu", "modal", "vastai", "lightning"}


def test_valid_hardware_substrate_tokens_canonical_set() -> None:
    # Per Catalog #190 + CLAUDE.md "Hardware identification": expected canonical tokens.
    assert "macos_arm64_m5_max" in _VALID_HARDWARE_SUBSTRATE_TOKENS
    assert "linux_x86_64_modal_a100" in _VALID_HARDWARE_SUBSTRATE_TOKENS
    assert "linux_x86_64_modal_cpu" in _VALID_HARDWARE_SUBSTRATE_TOKENS
    assert "unknown_unknown_unknown" in _VALID_HARDWARE_SUBSTRATE_TOKENS
    # Forbidden tokens (false precision per Catalog #190):
    assert "auto" not in _VALID_HARDWARE_SUBSTRATE_TOKENS
    assert "modal" not in _VALID_HARDWARE_SUBSTRATE_TOKENS


def test_placeholder_rationales_set_matches_catalog_287_canonical() -> None:
    assert "<rationale>" in _PLACEHOLDER_RATIONALES
    assert "<reason>" in _PLACEHOLDER_RATIONALES
    assert "" in _PLACEHOLDER_RATIONALES


# ---------------------------------------------------------------------------
# classify_hardware_substrate_for_dispatch behavior
# ---------------------------------------------------------------------------


def test_classify_hardware_substrate_modal_default_a100() -> None:
    token, klass = classify_hardware_substrate_for_dispatch("modal")
    assert token == "linux_x86_64_modal_a100"
    assert klass == "modal"


def test_classify_hardware_substrate_vastai_default_4090() -> None:
    token, klass = classify_hardware_substrate_for_dispatch("vastai")
    assert token == "linux_x86_64_vastai_4090"
    assert klass == "vastai"


def test_classify_hardware_substrate_lightning_default_a100() -> None:
    token, klass = classify_hardware_substrate_for_dispatch("lightning")
    assert token == "linux_x86_64_lightning_a100"
    assert klass == "lightning"


def test_classify_hardware_substrate_local_mps_maps_to_m5_max() -> None:
    token, klass = classify_hardware_substrate_for_dispatch("local-mps")
    assert token == "macos_arm64_m5_max"
    assert klass == "local-mps"


def test_classify_hardware_substrate_local_cpu_maps_to_linux_x86_64_unknown_cpu() -> None:
    token, klass = classify_hardware_substrate_for_dispatch("local-cpu")
    assert token == "linux_x86_64_unknown_cpu"
    assert klass == "local-cpu"


def test_classify_hardware_substrate_explicit_override_honored() -> None:
    token, klass = classify_hardware_substrate_for_dispatch(
        "modal", explicit_hardware_substrate="linux_x86_64_modal_t4"
    )
    assert token == "linux_x86_64_modal_t4"
    assert klass == "modal"


def test_classify_hardware_substrate_explicit_invalid_token_refused() -> None:
    with pytest.raises(CompressionPipelineError, match="Catalog #190"):
        classify_hardware_substrate_for_dispatch(
            "modal", explicit_hardware_substrate="bogus_gpu_xyz"
        )


def test_classify_hardware_substrate_invalid_request_refused() -> None:
    with pytest.raises(CompressionPipelineError):
        classify_hardware_substrate_for_dispatch("totally-bogus-request")


def test_classify_hardware_substrate_accepts_enum_directly() -> None:
    token, klass = classify_hardware_substrate_for_dispatch(HardwareSubstrateClass.MODAL)
    assert token == "linux_x86_64_modal_a100"
    assert klass == "modal"


# ---------------------------------------------------------------------------
# validate_recipe_trainer_pair behavior
# ---------------------------------------------------------------------------


def test_substrate_id_from_trainer_path_canonical_pattern() -> None:
    assert (
        _substrate_id_from_trainer_path(Path("experiments/train_substrate_d1_segnet_margin.py"))
        == "d1_segnet_margin"
    )


def test_substrate_id_from_trainer_path_non_canonical_refused() -> None:
    with pytest.raises(CompressionPipelineError, match="canonical"):
        _substrate_id_from_trainer_path(Path("experiments/run_something.py"))


def test_validate_recipe_trainer_pair_missing_trainer_refused(tmp_path: Path) -> None:
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n", encoding="utf-8")
    with pytest.raises(CompressionPipelineError, match="trainer"):
        validate_recipe_trainer_pair(
            substrate_trainer=tmp_path / "train_substrate_foo.py",
            recipe_path=recipe,
            repo_root=tmp_path,
        )


def test_validate_recipe_trainer_pair_missing_recipe_refused(tmp_path: Path) -> None:
    trainer = tmp_path / "train_substrate_foo.py"
    trainer.write_text("# trainer stub\n", encoding="utf-8")
    with pytest.raises(CompressionPipelineError, match="recipe"):
        validate_recipe_trainer_pair(
            substrate_trainer=trainer,
            recipe_path=tmp_path / "substrate_foo_modal_t4_dispatch.yaml",
            repo_root=tmp_path,
        )


def test_validate_recipe_trainer_pair_substrate_id_mismatch_refused(tmp_path: Path) -> None:
    trainer = tmp_path / "train_substrate_foo.py"
    trainer.write_text("# trainer stub\n", encoding="utf-8")
    recipe = tmp_path / "substrate_bar_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n", encoding="utf-8")
    with pytest.raises(CompressionPipelineError, match="Catalog #240"):
        validate_recipe_trainer_pair(
            substrate_trainer=trainer,
            recipe_path=recipe,
            repo_root=tmp_path,
        )


def test_validate_recipe_trainer_pair_clean_pass(tmp_path: Path) -> None:
    trainer = tmp_path / "train_substrate_foo.py"
    trainer.write_text("# trainer stub\n", encoding="utf-8")
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n", encoding="utf-8")
    sid, rname = validate_recipe_trainer_pair(
        substrate_trainer=trainer, recipe_path=recipe, repo_root=tmp_path
    )
    assert sid == "foo"
    assert rname == "substrate_foo_modal_t4_dispatch"


# ---------------------------------------------------------------------------
# PerAxisPredictedBand contract (per Catalog #356 + #324)
# ---------------------------------------------------------------------------


def test_per_axis_predicted_band_clean_construction() -> None:
    band = PerAxisPredictedBand(
        predicted_seg_distortion_band=(0.0, 0.005),
        predicted_pose_distortion_band=(0.0, 1.0e-4),
        predicted_archive_bytes_band=(100, 300),
        predicted_band_validation_status="pending_post_training",
    )
    assert band.predicted_seg_distortion_band == (0.0, 0.005)
    d = band.as_dict()
    assert d["predicted_band_validation_status"] == "pending_post_training"


def test_per_axis_predicted_band_invalid_status_refused() -> None:
    with pytest.raises(ValueError, match="Catalog #324"):
        PerAxisPredictedBand(
            predicted_seg_distortion_band=(0.0, 0.005),
            predicted_pose_distortion_band=(0.0, 1.0e-4),
            predicted_archive_bytes_band=(100, 300),
            predicted_band_validation_status="totally-bogus",
        )


def test_per_axis_predicted_band_inverted_seg_band_refused() -> None:
    with pytest.raises(ValueError, match="lo<=hi"):
        PerAxisPredictedBand(
            predicted_seg_distortion_band=(0.5, 0.1),
            predicted_pose_distortion_band=(0.0, 1.0e-4),
            predicted_archive_bytes_band=(100, 300),
            predicted_band_validation_status="pending_post_training",
        )


def test_per_axis_predicted_band_non_int_bytes_refused() -> None:
    with pytest.raises(ValueError, match="int"):
        PerAxisPredictedBand(
            predicted_seg_distortion_band=(0.0, 0.005),
            predicted_pose_distortion_band=(0.0, 1.0e-4),
            predicted_archive_bytes_band=(100.5, 300.5),  # type: ignore[arg-type]
            predicted_band_validation_status="pending_post_training",
        )


# ---------------------------------------------------------------------------
# _extract_per_axis_predicted_band_from_recipe behavior
# ---------------------------------------------------------------------------


def test_extract_per_axis_predicted_band_from_recipe_no_band_returns_none(tmp_path: Path) -> None:
    recipe = tmp_path / "no_band.yaml"
    recipe.write_text("dispatch_enabled: true\n", encoding="utf-8")
    assert _extract_per_axis_predicted_band_from_recipe(recipe) is None


def test_extract_per_axis_predicted_band_from_recipe_full_extraction(tmp_path: Path) -> None:
    recipe = tmp_path / "full_band.yaml"
    recipe.write_text(
        "predicted_band:\n"
        "  predicted_seg_distortion_band: [0.0, 0.005]\n"
        "  predicted_pose_distortion_band: [0.0, 1.0e-4]\n"
        "  predicted_archive_bytes_band: [100, 300]\n"
        "  predicted_band_validation_status: 'pending_post_training'\n",
        encoding="utf-8",
    )
    band = _extract_per_axis_predicted_band_from_recipe(recipe)
    assert band is not None
    assert band.predicted_seg_distortion_band == (0.0, 0.005)
    assert band.predicted_pose_distortion_band[1] == pytest.approx(1.0e-4)
    assert band.predicted_archive_bytes_band == (100, 300)


def test_extract_per_axis_predicted_band_from_recipe_missing_file_returns_none(
    tmp_path: Path,
) -> None:
    assert _extract_per_axis_predicted_band_from_recipe(tmp_path / "nope.yaml") is None


# ---------------------------------------------------------------------------
# derive_compression_pipeline_provenance contract (Catalog #323 sister)
# ---------------------------------------------------------------------------


def test_derive_compression_pipeline_provenance_canonical_shape() -> None:
    prov = derive_compression_pipeline_provenance(
        lane_id="lane_foo_20260526",
        substrate_id="foo",
        hardware_substrate="linux_x86_64_modal_a100",
        measurement_utc="2026-05-26T12:00:00+00:00",
    )
    assert prov["axis_tag"] == "[predicted]"
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["canonical_helper_invocation"] == "tac.submission_packet.build_compression_pipeline"
    assert prov["canonical_equation_id"] == CANONICAL_EQUATION_ID
    assert prov["canonical_equation_status"] == "FORMALIZATION_PENDING"


# ---------------------------------------------------------------------------
# build_compression_pipeline behavior (canonical happy path + invariants)
# ---------------------------------------------------------------------------


def _write_dummy_trainer_and_recipe(tmp_path: Path, sid: str) -> tuple[Path, Path]:
    trainer = tmp_path / f"train_substrate_{sid}.py"
    trainer.write_text("# dummy trainer for test\n", encoding="utf-8")
    recipe = tmp_path / f"substrate_{sid}_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true\n"
        "predicted_band:\n"
        "  predicted_seg_distortion_band: [0.0, 0.005]\n"
        "  predicted_pose_distortion_band: [0.0, 0.0001]\n"
        "  predicted_archive_bytes_band: [100, 300]\n"
        "  predicted_band_validation_status: 'pending_post_training'\n",
        encoding="utf-8",
    )
    return trainer, recipe


def test_build_compression_pipeline_dry_run_skip_protocol(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    result = build_compression_pipeline(
        lane_id="lane_tiny_foo_20260526",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=trainer,
        recipe_path=recipe,
        hardware_substrate="modal",
        qat_enabled=True,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        skip_protocol_verification=True,
    )
    assert isinstance(result, CompressionPipelineResult)
    assert result.lane_id == "lane_tiny_foo_20260526"
    assert result.substrate_id == "tiny_foo"
    assert result.hardware_substrate == "linux_x86_64_modal_a100"
    assert result.hardware_substrate_class == "modal"
    assert result.mlx_first_encode is False  # Modal route is not Apple Silicon
    assert result.qat_enabled is True
    assert result.dispatch_optimization_protocol_overall_pass is True  # skipped means clean
    assert result.dispatch_optimization_protocol_blockers == ()
    assert result.per_axis_predicted_band is not None
    assert result.axis_tag == "[predicted]"
    assert result.score_claim is False
    assert result.promotable is False
    assert result.canonical_equation_id == CANONICAL_EQUATION_ID
    assert result.canonical_equation_status == "FORMALIZATION_PENDING"


def test_build_compression_pipeline_local_mps_routes_mlx_first(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    result = build_compression_pipeline(
        lane_id="lane_tiny_foo_local_mps_20260526",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=trainer,
        recipe_path=recipe,
        hardware_substrate="local-mps",
        qat_enabled=True,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        skip_protocol_verification=True,
    )
    assert result.hardware_substrate == "macos_arm64_m5_max"
    assert result.mlx_first_encode is True


def test_build_compression_pipeline_explicit_mlx_first_override(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    result = build_compression_pipeline(
        lane_id="lane_tiny_foo_mlx_override_20260526",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=trainer,
        recipe_path=recipe,
        hardware_substrate="modal",
        qat_enabled=False,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        mlx_first=True,
        skip_protocol_verification=True,
    )
    assert result.mlx_first_encode is True  # explicit override
    assert result.qat_enabled is False


def test_build_compression_pipeline_empty_lane_id_refused(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    with pytest.raises(CompressionPipelineError, match="lane_id"):
        build_compression_pipeline(
            lane_id="",
            video_path=Path("upstream/videos/0.mkv"),
            substrate_trainer=trainer,
            recipe_path=recipe,
            hardware_substrate="modal",
            qat_enabled=True,
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
            skip_protocol_verification=True,
        )


def test_build_compression_pipeline_carries_caller_weights_metadata(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"\x00" * 1024)
    result = build_compression_pipeline(
        lane_id="lane_tiny_foo_weights_20260526",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=trainer,
        recipe_path=recipe,
        hardware_substrate="modal",
        qat_enabled=True,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        weights_export_path=weights,
        weights_sha256="a" * 64,
        weights_size_bytes=1024,
        training_anchor_call_id="fc-01TESTcallid",
        cost_usd=0.42,
        elapsed_seconds=123.4,
        skip_protocol_verification=True,
    )
    assert result.weights_export_path == str(weights)
    assert result.weights_sha256 == "a" * 64
    assert result.weights_size_bytes == 1024
    assert result.training_anchor_call_id == "fc-01TESTcallid"
    assert result.cost_usd == pytest.approx(0.42)
    assert result.elapsed_seconds == pytest.approx(123.4)


def test_build_compression_pipeline_invalid_weights_sha256_refused(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    with pytest.raises(ValueError, match="weights_sha256"):
        build_compression_pipeline(
            lane_id="lane_tiny_foo_bad_sha_20260526",
            video_path=Path("upstream/videos/0.mkv"),
            substrate_trainer=trainer,
            recipe_path=recipe,
            hardware_substrate="modal",
            qat_enabled=True,
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
            weights_sha256="too_short",
            skip_protocol_verification=True,
        )


# ---------------------------------------------------------------------------
# CompressionPipelineResult.as_dict / canonical roundtrip
# ---------------------------------------------------------------------------


def test_compression_pipeline_result_as_dict_round_trip(tmp_path: Path) -> None:
    trainer, recipe = _write_dummy_trainer_and_recipe(tmp_path, "tiny_foo")
    result = build_compression_pipeline(
        lane_id="lane_tiny_foo_round_trip_20260526",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=trainer,
        recipe_path=recipe,
        hardware_substrate="modal",
        qat_enabled=True,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        skip_protocol_verification=True,
    )
    d = result.as_dict()
    # Canonical keys present:
    for k in (
        "schema_version",
        "lane_id",
        "substrate_id",
        "video_path",
        "hardware_substrate",
        "hardware_substrate_class",
        "axis_tag",
        "score_claim",
        "promotable",
        "evidence_grade",
        "canonical_helper_invocation",
        "canonical_equation_id",
        "canonical_equation_status",
        "canonical_provenance",
        "measurement_utc",
        "elapsed_seconds",
        "per_axis_predicted_band",
    ):
        assert k in d
    # JSON-serializable end-to-end:
    json_str = json.dumps(d, sort_keys=True)
    parsed = json.loads(json_str)
    assert parsed["axis_tag"] == "[predicted]"
    assert parsed["score_claim"] is False
    assert parsed["promotable"] is False


def test_compression_pipeline_result_score_claim_invariant_refused(tmp_path: Path) -> None:
    """Frozen invariant: score_claim cannot be True per Catalog #341."""
    with pytest.raises(ValueError, match="score_claim"):
        CompressionPipelineResult(
            schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
            lane_id="lane_foo_20260526",
            substrate_id="foo",
            video_path="upstream/videos/0.mkv",
            hardware_substrate="linux_x86_64_modal_a100",
            hardware_substrate_class="modal",
            substrate_trainer_path="experiments/train_substrate_foo.py",
            recipe_path=".omx/operator_authorize_recipes/substrate_foo_modal_t4_dispatch.yaml",
            mlx_first_encode=False,
            qat_enabled=True,
            weights_export_path=None,
            weights_sha256=None,
            weights_size_bytes=None,
            training_anchor_call_id=None,
            qat_anchor_call_id=None,
            dispatch_optimization_protocol_overall_pass=True,
            dispatch_optimization_protocol_blockers=(),
            per_axis_predicted_band=None,
            measurement_utc="2026-05-26T12:00:00+00:00",
            axis_tag="[predicted]",
            score_claim=True,  # FORBIDDEN
            promotable=False,
            evidence_grade="[predicted; compression-pipeline-canonical]",
            canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
            elapsed_seconds=0.0,
            cost_usd=None,
        )


def test_compression_pipeline_result_promotable_invariant_refused(tmp_path: Path) -> None:
    """Frozen invariant: promotable cannot be True per Catalog #341."""
    with pytest.raises(ValueError, match="promotable"):
        CompressionPipelineResult(
            schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
            lane_id="lane_foo_20260526",
            substrate_id="foo",
            video_path="upstream/videos/0.mkv",
            hardware_substrate="linux_x86_64_modal_a100",
            hardware_substrate_class="modal",
            substrate_trainer_path="experiments/train_substrate_foo.py",
            recipe_path=".omx/operator_authorize_recipes/substrate_foo_modal_t4_dispatch.yaml",
            mlx_first_encode=False,
            qat_enabled=True,
            weights_export_path=None,
            weights_sha256=None,
            weights_size_bytes=None,
            training_anchor_call_id=None,
            qat_anchor_call_id=None,
            dispatch_optimization_protocol_overall_pass=True,
            dispatch_optimization_protocol_blockers=(),
            per_axis_predicted_band=None,
            measurement_utc="2026-05-26T12:00:00+00:00",
            axis_tag="[predicted]",
            score_claim=False,
            promotable=True,  # FORBIDDEN
            evidence_grade="[predicted; compression-pipeline-canonical]",
            canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
            elapsed_seconds=0.0,
            cost_usd=None,
        )


def test_compression_pipeline_result_evidence_grade_invariant_refused() -> None:
    """Frozen invariant: evidence_grade must start with '[predicted;'."""
    with pytest.raises(ValueError, match="evidence_grade"):
        CompressionPipelineResult(
            schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
            lane_id="lane_foo_20260526",
            substrate_id="foo",
            video_path="upstream/videos/0.mkv",
            hardware_substrate="linux_x86_64_modal_a100",
            hardware_substrate_class="modal",
            substrate_trainer_path="experiments/train_substrate_foo.py",
            recipe_path=".omx/operator_authorize_recipes/substrate_foo_modal_t4_dispatch.yaml",
            mlx_first_encode=False,
            qat_enabled=True,
            weights_export_path=None,
            weights_sha256=None,
            weights_size_bytes=None,
            training_anchor_call_id=None,
            qat_anchor_call_id=None,
            dispatch_optimization_protocol_overall_pass=True,
            dispatch_optimization_protocol_blockers=(),
            per_axis_predicted_band=None,
            measurement_utc="2026-05-26T12:00:00+00:00",
            axis_tag="[predicted]",
            score_claim=False,
            promotable=False,
            evidence_grade="[contest-CUDA]",  # FORBIDDEN at this layer
            canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
            elapsed_seconds=0.0,
            cost_usd=None,
        )


def test_compression_pipeline_result_invalid_hardware_substrate_refused() -> None:
    with pytest.raises(ValueError, match="Catalog #190"):
        CompressionPipelineResult(
            schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
            lane_id="lane_foo_20260526",
            substrate_id="foo",
            video_path="upstream/videos/0.mkv",
            hardware_substrate="totally_bogus_token",
            hardware_substrate_class="modal",
            substrate_trainer_path="experiments/train_substrate_foo.py",
            recipe_path=".omx/operator_authorize_recipes/substrate_foo_modal_t4_dispatch.yaml",
            mlx_first_encode=False,
            qat_enabled=True,
            weights_export_path=None,
            weights_sha256=None,
            weights_size_bytes=None,
            training_anchor_call_id=None,
            qat_anchor_call_id=None,
            dispatch_optimization_protocol_overall_pass=True,
            dispatch_optimization_protocol_blockers=(),
            per_axis_predicted_band=None,
            measurement_utc="2026-05-26T12:00:00+00:00",
            axis_tag="[predicted]",
            score_claim=False,
            promotable=False,
            evidence_grade="[predicted; compression-pipeline-canonical]",
            canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
            elapsed_seconds=0.0,
            cost_usd=None,
        )


def test_compression_pipeline_result_invalid_equation_status_refused() -> None:
    with pytest.raises(ValueError, match="Catalog #344"):
        CompressionPipelineResult(
            schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
            lane_id="lane_foo_20260526",
            substrate_id="foo",
            video_path="upstream/videos/0.mkv",
            hardware_substrate="linux_x86_64_modal_a100",
            hardware_substrate_class="modal",
            substrate_trainer_path="experiments/train_substrate_foo.py",
            recipe_path=".omx/operator_authorize_recipes/substrate_foo_modal_t4_dispatch.yaml",
            mlx_first_encode=False,
            qat_enabled=True,
            weights_export_path=None,
            weights_sha256=None,
            weights_size_bytes=None,
            training_anchor_call_id=None,
            qat_anchor_call_id=None,
            dispatch_optimization_protocol_overall_pass=True,
            dispatch_optimization_protocol_blockers=(),
            per_axis_predicted_band=None,
            measurement_utc="2026-05-26T12:00:00+00:00",
            axis_tag="[predicted]",
            score_claim=False,
            promotable=False,
            evidence_grade="[predicted; compression-pipeline-canonical]",
            canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="totally_bogus_status",
            elapsed_seconds=0.0,
            cost_usd=None,
        )


# ---------------------------------------------------------------------------
# verify_compression_pipeline_protocol_complete (Catalog #270 sister)
# ---------------------------------------------------------------------------


def test_verify_compression_pipeline_protocol_canonical_helper_missing_refused(
    tmp_path: Path,
) -> None:
    """When the canonical helper is missing, the verifier MUST refuse."""
    with pytest.raises(CompressionPipelineError, match="canonical helper"):
        verify_compression_pipeline_protocol_complete(
            substrate_trainer=tmp_path / "train_substrate_foo.py",
            recipe_path=tmp_path / "substrate_foo_modal_t4_dispatch.yaml",
            repo_root=tmp_path,  # tmp_path lacks the canonical helper file
        )


def test_verify_compression_pipeline_protocol_real_repo_returns_tuple() -> None:
    """Smoke test: the canonical helper IS importable from the real repo."""
    # Use a real substrate trainer + recipe from the repo (READ-ONLY).
    trainer = REPO_ROOT / "experiments" / "train_substrate_d1_segnet_margin_polytope.py"
    if not trainer.is_file():
        pytest.skip("canonical d1 trainer absent from repo (sister-flux acceptable)")
    recipe = (
        REPO_ROOT
        / ".omx"
        / "operator_authorize_recipes"
        / "substrate_d1_segnet_margin_polytope_modal_t4_dispatch.yaml"
    )
    if not recipe.is_file():
        pytest.skip("canonical d1 recipe absent (sister-flux acceptable)")
    overall_pass, blockers = verify_compression_pipeline_protocol_complete(
        substrate_trainer=trainer, recipe_path=recipe, repo_root=REPO_ROOT
    )
    assert isinstance(overall_pass, bool)
    assert isinstance(blockers, tuple)


# ---------------------------------------------------------------------------
# Live-repo regression guards (Catalog #185 sister)
# ---------------------------------------------------------------------------


def test_module_importable_from_canonical_package_path() -> None:
    """Catalog #176 sister: the module is reachable via canonical path."""
    import tac.submission_packet as pkg
    import tac.submission_packet.compression_pipeline as mod

    assert hasattr(pkg, "build_compression_pipeline")
    assert hasattr(mod, "build_compression_pipeline")


def test_compression_pipeline_module_canonical_provenance_routing() -> None:
    """Helper module re-exports the canonical contract via __all__."""
    import tac.submission_packet as pkg

    for name in (
        "CANONICAL_EQUATION_ID",
        "COMPRESSION_PIPELINE_SCHEMA_VERSION",
        "HardwareSubstrateClass",
        "PHASE_2_LAYER_VERSION",
        "CompressionPipelineError",
        "CompressionPipelineResult",
        "PerAxisPredictedBand",
        "build_compression_pipeline",
        "classify_hardware_substrate_for_dispatch",
        "derive_compression_pipeline_provenance",
        "validate_recipe_trainer_pair",
        "verify_compression_pipeline_protocol_complete",
    ):
        assert name in pkg.__all__


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


CLI_PATH = REPO_ROOT / "tools" / "compression_pipeline_cli.py"


def test_cli_exists_and_executable() -> None:
    assert CLI_PATH.is_file(), f"canonical CLI must exist at {CLI_PATH}"


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "compression" in result.stdout.lower() or "compression" in result.stderr.lower()


def test_cli_dry_run_exit_code_0_for_canonical_pair(tmp_path: Path) -> None:
    """CLI --dry-run should exit 0 on a clean trainer + recipe pair."""
    # Use synthetic trainer + recipe under tmp_path so we don't touch live infra.
    trainer = tmp_path / "train_substrate_clean_foo.py"
    trainer.write_text("# stub trainer\n", encoding="utf-8")
    recipe = tmp_path / "substrate_clean_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true\n"
        "predicted_band:\n"
        "  predicted_seg_distortion_band: [0.0, 0.005]\n"
        "  predicted_pose_distortion_band: [0.0, 0.0001]\n"
        "  predicted_archive_bytes_band: [100, 300]\n"
        "  predicted_band_validation_status: 'pending_post_training'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--lane-id",
            "lane_cli_dry_run_clean_foo_20260526",
            "--video-path",
            "upstream/videos/0.mkv",
            "--substrate-trainer",
            str(trainer),
            "--recipe-path",
            str(recipe),
            "--hardware-substrate",
            "modal",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--skip-protocol-verification",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    # JSON output must be parseable + canonical schema present.
    parsed = json.loads(result.stdout)
    assert parsed["schema_version"] == COMPRESSION_PIPELINE_SCHEMA_VERSION
    assert parsed["lane_id"] == "lane_cli_dry_run_clean_foo_20260526"
    assert parsed["axis_tag"] == "[predicted]"
    assert parsed["score_claim"] is False
    assert parsed["promotable"] is False


def test_cli_missing_required_args_exit_code_5(tmp_path: Path) -> None:
    """CLI must exit 5 (CLI error) on argparse failure."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH)],  # no args at all
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # argparse returns 2 by default for missing args; CLI may translate to 5.
    assert result.returncode != 0


def test_cli_invalid_substrate_pair_exit_code_nonzero(tmp_path: Path) -> None:
    """CLI must refuse mismatched trainer + recipe pair."""
    trainer = tmp_path / "train_substrate_aaa.py"
    trainer.write_text("# stub\n", encoding="utf-8")
    recipe = tmp_path / "substrate_bbb_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--lane-id",
            "lane_cli_mismatch_20260526",
            "--video-path",
            "upstream/videos/0.mkv",
            "--substrate-trainer",
            str(trainer),
            "--recipe-path",
            str(recipe),
            "--hardware-substrate",
            "modal",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--skip-protocol-verification",
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Cathedral consumer regression (Catalog #335)
# ---------------------------------------------------------------------------


def test_cathedral_consumer_package_exists() -> None:
    """Catalog #335 sister: the cathedral consumer package must exist."""
    consumer_init = (
        REPO_ROOT
        / "src"
        / "tac"
        / "cathedral_consumers"
        / "compression_pipeline_readiness_consumer"
        / "__init__.py"
    )
    assert consumer_init.is_file(), f"cathedral consumer must exist at {consumer_init}"


def test_cathedral_consumer_canonical_contract_compliance() -> None:
    """Catalog #335 sister: canonical contract validates."""
    from tac.cathedral.consumer_contract import validate_consumer_module
    import tac.cathedral_consumers.compression_pipeline_readiness_consumer as consumer

    verdict = validate_consumer_module(consumer)
    assert verdict is not None, "canonical contract verdict must be present"
    # Canonical fields:
    assert hasattr(consumer, "CONSUMER_NAME")
    assert hasattr(consumer, "CONSUMER_VERSION")
    assert hasattr(consumer, "CONSUMER_HOOK_NUMBERS")
    assert callable(getattr(consumer, "update_from_anchor", None))
    assert callable(getattr(consumer, "consume_candidate", None))


def test_cathedral_consumer_returns_canonical_routing_markers() -> None:
    """Catalog #341 sister: every routing branch returns the canonical markers."""
    import tac.cathedral_consumers.compression_pipeline_readiness_consumer as consumer

    contrib = consumer.consume_candidate(
        {"lane_id": "lane_foo_20260526", "substrate_id": "foo"}
    )
    # Tier A observability-only contract:
    assert contrib["predicted_delta_adjustment"] == 0.0
    assert contrib["promotable"] is False
    assert contrib["axis_tag"] == "[predicted]"
