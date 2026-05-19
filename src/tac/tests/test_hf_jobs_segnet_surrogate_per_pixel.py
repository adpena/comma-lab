# SPDX-License-Identifier: MIT
"""Tests for HF Jobs SegNet surrogate per-pixel mIoU sister lane.

Covers:
- Per-pixel mIoU helper correctness (incl. degenerate / empty cases)
- Per-class IoU vector shape + NaN handling for absent classes
- Argmax disagreement rate formulation matches contest distortion
- Bbox prompt extraction from synthetic 5-class GT mask
- SAM2 dispatcher script_args canonical contract
- Recipe schema validation
- Symposium frontmatter Catalog #300 compliance
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_SCRIPT = REPO_ROOT / "experiments" / "hf_jobs_segnet_surrogate_distillation_per_pixel.py"
DISPATCHER_SCRIPT = REPO_ROOT / "tools" / "dispatch_hf_jobs_segnet_surrogate_per_pixel.py"
RECIPE_YAML = (
    REPO_ROOT
    / ".omx"
    / "operator_authorize_recipes"
    / "substrate_hf_jobs_segnet_surrogate_distillation_per_pixel_t4_dispatch.yaml"
)
SYMPOSIUM_MEMO = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "council_t1_hf_jobs_segnet_surrogate_distillation_per_pixel_symposium_20260519.md"
)


def _load_module_from_path(path: Path, mod_name: str):
    """Load a module from filesystem path via importlib (PEP 451)."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def training_module():
    """Load the training script as a module to test its pure helpers.

    The training script guards heavy deps (transformers / monai / trackio /
    datasets) behind a try/except + _HEAVY_DEPS_AVAILABLE sentinel so pure
    helpers (compute_per_pixel_miou + sister) are importable without HF
    Jobs worker dependencies. HF Jobs worker has all heavy deps via PEP
    723 inline metadata at the top of the script.
    """
    return _load_module_from_path(TRAINING_SCRIPT, "hf_jobs_segnet_per_pixel_module")


# --------------------------------------------------------------------------
# Per-pixel mIoU helper correctness
# --------------------------------------------------------------------------


def test_per_pixel_miou_perfect_match():
    """When pred == gt, mIoU == 1.0."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_pixel_miou,
    )

    mask = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3]], dtype=np.int64)
    miou = compute_per_pixel_miou(mask, mask, num_classes=5)
    assert miou == pytest.approx(1.0)


def test_per_pixel_miou_complete_disagreement():
    """When every pixel disagrees, mIoU = 0.0."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_pixel_miou,
    )

    pred = np.array([[0, 0], [0, 0]], dtype=np.int64)
    gt = np.array([[1, 1], [1, 1]], dtype=np.int64)
    miou = compute_per_pixel_miou(pred, gt, num_classes=5)
    # Both class 0 (pred only) and class 1 (gt only) have intersection=0,
    # union>0, IoU=0 for each. Mean = 0.
    assert miou == pytest.approx(0.0)


def test_per_pixel_miou_partial_agreement():
    """50% agreement on single class produces canonical mIoU."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_pixel_miou,
    )

    pred = np.array([[0, 0, 1, 1]], dtype=np.int64)
    gt = np.array([[0, 1, 0, 1]], dtype=np.int64)
    # Class 0: pred={0,0,?,?}=2 vs gt={0,?,0,?}=2; intersection={(0,0)}=1; union={(0,0),(0,1),(1,0)}=3; IoU=1/3
    # Class 1: pred={?,?,1,1}=2 vs gt={?,1,?,1}=2; intersection={(1,1)}=1; union=3; IoU=1/3
    # mIoU = (1/3 + 1/3) / 2 = 1/3
    miou = compute_per_pixel_miou(pred, gt, num_classes=5)
    assert miou == pytest.approx(1 / 3)


def test_per_pixel_miou_excludes_absent_classes():
    """Classes absent from both pred + gt are excluded from mean per canonical convention."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_pixel_miou,
    )

    # Only class 0 + class 1 appear; classes 2/3/4 absent
    pred = np.array([[0, 1]], dtype=np.int64)
    gt = np.array([[0, 1]], dtype=np.int64)
    # mIoU should compute over present classes only (mean of 1.0, 1.0)
    miou = compute_per_pixel_miou(pred, gt, num_classes=5)
    assert miou == pytest.approx(1.0)


def test_per_pixel_miou_shape_mismatch_raises():
    """Shape mismatch between pred and gt raises ValueError."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_pixel_miou,
    )

    pred = np.zeros((2, 2), dtype=np.int64)
    gt = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        compute_per_pixel_miou(pred, gt, num_classes=5)


# --------------------------------------------------------------------------
# Per-class IoU vector correctness
# --------------------------------------------------------------------------


def test_per_class_iou_vector_shape():
    """Returns (num_classes,) float array."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_class_iou,
    )

    pred = np.array([[0, 1, 2]], dtype=np.int64)
    gt = np.array([[0, 1, 2]], dtype=np.int64)
    per_class = compute_per_class_iou(pred, gt, num_classes=5)
    assert per_class.shape == (5,)
    assert per_class.dtype == np.float64


def test_per_class_iou_nan_for_absent_classes():
    """NaN where class is absent from both pred + gt."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_class_iou,
    )

    pred = np.array([[0]], dtype=np.int64)
    gt = np.array([[0]], dtype=np.int64)
    per_class = compute_per_class_iou(pred, gt, num_classes=5)
    # Class 0 present; classes 1/2/3/4 absent
    assert per_class[0] == pytest.approx(1.0)
    for c in range(1, 5):
        assert np.isnan(per_class[c])


def test_per_class_iou_per_class_correctness():
    """Per-class IoU values are correct."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_class_iou,
    )

    pred = np.array([[0, 0, 1, 1]], dtype=np.int64)
    gt = np.array([[0, 1, 0, 1]], dtype=np.int64)
    per_class = compute_per_class_iou(pred, gt, num_classes=5)
    # Class 0: IoU = 1/3 (per test_per_pixel_miou_partial_agreement)
    # Class 1: IoU = 1/3
    assert per_class[0] == pytest.approx(1 / 3)
    assert per_class[1] == pytest.approx(1 / 3)
    assert np.isnan(per_class[2])


# --------------------------------------------------------------------------
# Argmax disagreement rate (contest-axis-parity metric)
# --------------------------------------------------------------------------


def test_argmax_disagreement_rate_perfect_agreement():
    """Perfect agreement = 0.0 disagreement rate."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_argmax_disagreement_rate,
    )

    mask = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    rate = compute_argmax_disagreement_rate(mask, mask)
    assert rate == pytest.approx(0.0)


def test_argmax_disagreement_rate_complete_disagreement():
    """Every-pixel disagreement = 1.0."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_argmax_disagreement_rate,
    )

    pred = np.zeros((3, 3), dtype=np.int64)
    gt = np.ones((3, 3), dtype=np.int64)
    rate = compute_argmax_disagreement_rate(pred, gt)
    assert rate == pytest.approx(1.0)


def test_argmax_disagreement_rate_matches_contest_distortion_formula():
    """Verify formula matches contest distortion (argmax(pred) != gt).mean()."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_argmax_disagreement_rate,
    )

    np.random.seed(42)
    pred = np.random.randint(0, 5, size=(10, 10), dtype=np.int64)
    gt = np.random.randint(0, 5, size=(10, 10), dtype=np.int64)
    expected = float((pred != gt).mean())
    actual = compute_argmax_disagreement_rate(pred, gt)
    assert actual == pytest.approx(expected)


# --------------------------------------------------------------------------
# Bbox prompt extraction from 5-class GT mask
# --------------------------------------------------------------------------


def test_extract_bbox_canonical_centered_square():
    """Centered foreground square produces canonical bbox."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        extract_bbox_prompt_from_gt_mask,
    )

    mask = np.zeros((10, 10), dtype=np.int64)
    mask[3:7, 3:7] = 1
    bbox = extract_bbox_prompt_from_gt_mask(mask, num_classes=5)
    assert bbox == [3, 3, 7, 7]


def test_extract_bbox_empty_mask_returns_whole_image():
    """All-zero mask returns whole-image bbox fallback."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        extract_bbox_prompt_from_gt_mask,
    )

    mask = np.zeros((20, 30), dtype=np.int64)
    bbox = extract_bbox_prompt_from_gt_mask(mask, num_classes=5)
    assert bbox == [0, 0, 30, 20]


def test_extract_bbox_largest_component_wins():
    """Largest connected component bbox is selected when multiple exist."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        extract_bbox_prompt_from_gt_mask,
    )

    mask = np.zeros((20, 20), dtype=np.int64)
    # Small component (class 1, 2x2)
    mask[0:2, 0:2] = 1
    # Large component (class 2, 8x8)
    mask[5:13, 5:13] = 2
    bbox = extract_bbox_prompt_from_gt_mask(mask, num_classes=5)
    # Largest component is the 8x8 at [5:13, 5:13]
    assert bbox == [5, 5, 13, 13]


def test_extract_bbox_3d_mask_squeezes_to_2d():
    """3D mask with trailing single channel is squeezed."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        extract_bbox_prompt_from_gt_mask,
    )

    mask = np.zeros((10, 10, 1), dtype=np.int64)
    mask[3:7, 3:7, 0] = 1
    bbox = extract_bbox_prompt_from_gt_mask(mask, num_classes=5)
    assert bbox == [3, 3, 7, 7]


# --------------------------------------------------------------------------
# SAM2 dispatcher script_args canonical contract
# --------------------------------------------------------------------------


def test_dispatcher_module_loads():
    """Dispatcher script imports cleanly via importlib."""
    module = _load_module_from_path(DISPATCHER_SCRIPT, "dispatch_per_pixel_module")
    assert hasattr(module, "build_sam2_per_pixel_script_args")
    assert hasattr(module, "SAM2_DEFAULT_MODEL")
    assert hasattr(module, "SAM2_DEFAULT_HUB_MODEL_REPO")


def test_dispatcher_script_args_canonical_flags():
    """SAM2 dispatcher emits all required canonical flags per plugin directive #4."""
    module = _load_module_from_path(DISPATCHER_SCRIPT, "dispatch_per_pixel_module_2")
    args = module.build_sam2_per_pixel_script_args(
        hub_dataset_repo="test/dataset",
        hub_model_repo="test/model",
    )
    # PRIMARY: eval_mean_iou is the metric for best model (honors Contrarian VETO)
    assert "--metric_for_best_model" in args
    idx = args.index("--metric_for_best_model")
    assert args[idx + 1] == "eval_mean_iou"
    # Plugin directive #4: bbox prompt_type
    assert "--prompt_type" in args
    idx = args.index("--prompt_type")
    assert args[idx + 1] == "bbox"
    # Plugin directive #4: --remove_unused_columns False (per-pixel mask + bbox must persist)
    assert "--remove_unused_columns" in args
    idx = args.index("--remove_unused_columns")
    assert args[idx + 1] == "False"
    # Plugin directive #4: --dataloader_pin_memory False (SAM2 input_boxes fail pin_memory)
    assert "--dataloader_pin_memory" in args
    idx = args.index("--dataloader_pin_memory")
    assert args[idx + 1] == "False"


def test_dispatcher_default_model_is_sam2_tiny():
    """Default SAM2 model is the canonical sam2.1-hiera-tiny per symposium Section 8."""
    module = _load_module_from_path(DISPATCHER_SCRIPT, "dispatch_per_pixel_module_3")
    assert module.SAM2_DEFAULT_MODEL == "facebook/sam2.1-hiera-tiny"
    assert module.SAM2_DEFAULT_HUB_MODEL_REPO == "adpena/comma-segnet-surrogate-sam2-tiny-per-pixel"
    assert module.SAM2_DEFAULT_NUM_EPOCHS == 30
    assert module.SAM2_DEFAULT_METRIC == "eval_mean_iou"


# --------------------------------------------------------------------------
# Recipe schema validation
# --------------------------------------------------------------------------


def test_recipe_yaml_exists_and_parses():
    """Recipe YAML exists + parses + carries canonical fields."""
    import yaml
    assert RECIPE_YAML.exists(), f"Recipe missing: {RECIPE_YAML}"
    recipe = yaml.safe_load(RECIPE_YAML.read_text())
    assert recipe["schema_version"] == 1
    assert recipe["name"] == "substrate_hf_jobs_segnet_surrogate_distillation_per_pixel_t4_dispatch"
    assert recipe["lane_id"] == "lane_hf_jobs_segnet_surrogate_distillation_per_pixel_20260519"
    assert recipe["platform"] == "hf_jobs"
    assert recipe["dispatch_kind"] == "tool"


def test_recipe_research_only_at_landing():
    """Recipe is research_only + dispatch_enabled=false at landing per CLAUDE.md."""
    import yaml
    recipe = yaml.safe_load(RECIPE_YAML.read_text())
    assert recipe["research_only"] is True
    assert recipe["dispatch_enabled"] is False
    assert "dispatch_blockers" in recipe
    assert len(recipe["dispatch_blockers"]) >= 1


def test_recipe_predicted_band_pending_post_training():
    """Per Catalog #324 predicted_band has pending_post_training validation status."""
    import yaml
    recipe = yaml.safe_load(RECIPE_YAML.read_text())
    assert recipe["predicted_band_validation_status"] == "pending_post_training"
    assert "predicted_band_reactivation_criteria" in recipe


def test_recipe_min_smoke_gpu_t4():
    """min_smoke_gpu = T4 per slot 7 symposium Section 8."""
    import yaml
    recipe = yaml.safe_load(RECIPE_YAML.read_text())
    assert recipe["min_smoke_gpu"] == "T4"
    assert recipe["min_vram_gb"] == 16


def test_recipe_council_symposium_cited():
    """Recipe cites the per-substrate symposium memo per Catalog #325."""
    import yaml
    recipe = yaml.safe_load(RECIPE_YAML.read_text())
    assert "council_symposium_memo" in recipe
    assert recipe["council_symposium_memo"] == str(
        SYMPOSIUM_MEMO.relative_to(REPO_ROOT)
    )
    assert recipe["council_symposium_verdict"] == "PROCEED"


def test_recipe_solver_wire_in_hooks_complete():
    """6-hook wire-in declaration per Catalog #125."""
    import yaml
    recipe = yaml.safe_load(RECIPE_YAML.read_text())
    assert "solver_wire_in_hooks" in recipe
    hooks = recipe["solver_wire_in_hooks"]
    for hook in (
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
        "cathedral_autopilot_dispatch",
        "continual_learning_posterior",
        "probe_disambiguator",
    ):
        assert hook in hooks, f"Missing hook: {hook}"


# --------------------------------------------------------------------------
# Symposium memo frontmatter Catalog #300 compliance
# --------------------------------------------------------------------------


def test_symposium_memo_exists():
    """Symposium memo exists at canonical path."""
    assert SYMPOSIUM_MEMO.exists(), f"Symposium memo missing: {SYMPOSIUM_MEMO}"


def test_symposium_frontmatter_catalog_300_compliance():
    """Symposium frontmatter carries v2 fields per Catalog #300."""
    import yaml
    content = SYMPOSIUM_MEMO.read_text()
    assert content.startswith("---\n")
    end_idx = content.find("\n---\n", 4)
    assert end_idx > 0
    frontmatter = yaml.safe_load(content[4:end_idx])
    required_fields = (
        "council_tier",
        "council_attendees",
        "council_quorum_met",
        "council_verdict",
        "council_dissent",
        "council_assumption_adversary_verdict",
        "council_decisions_recorded",
        "council_predicted_mission_contribution",
        "council_override_invoked",
    )
    for field in required_fields:
        assert field in frontmatter, f"Missing required v2 field: {field}"


def test_symposium_verdict_is_proceed():
    """Symposium PROCEED verdict (Contrarian VETO from sister lane is structurally honored)."""
    import yaml
    content = SYMPOSIUM_MEMO.read_text()
    end_idx = content.find("\n---\n", 4)
    frontmatter = yaml.safe_load(content[4:end_idx])
    assert frontmatter["council_verdict"] == "PROCEED"
    # All 6 attendees present (sextet pact)
    assert len(frontmatter["council_attendees"]) == 6
    assert "Contrarian" in frontmatter["council_attendees"]
    assert "Assumption-Adversary" in frontmatter["council_attendees"]


def test_symposium_assumption_adversary_classifies_assumptions():
    """Assumption-Adversary verdict surfaces assumption classifications per Catalog #292."""
    import yaml
    content = SYMPOSIUM_MEMO.read_text()
    end_idx = content.find("\n---\n", 4)
    frontmatter = yaml.safe_load(content[4:end_idx])
    verdicts = frontmatter["council_assumption_adversary_verdict"]
    assert len(verdicts) >= 1
    for v in verdicts:
        assert "assumption" in v
        assert "classification" in v
        assert "rationale" in v
        assert v["classification"] in (
            "HARD-EARNED", "CARGO-CULTED", "HARD-EARNED-WITH-CAVEAT",
            "CARGO-CULTED-EMPIRICALLY-FALSIFIED",
            "HARD-EARNED-EMPIRICALLY-VERIFIED",
        )


# --------------------------------------------------------------------------
# Training script structural compliance
# --------------------------------------------------------------------------


def test_training_script_exists():
    """Training script exists at canonical path."""
    assert TRAINING_SCRIPT.exists(), f"Training script missing: {TRAINING_SCRIPT}"


def test_training_script_has_pep_723_inline_metadata():
    """Per plugin directive #3: PEP 723 inline metadata with heavy deps."""
    content = TRAINING_SCRIPT.read_text()
    assert content.startswith("# /// script\n")
    assert "# dependencies = [" in content
    # Verify SAM2-specific heavy deps present
    assert "transformers" in content
    assert "monai" in content  # for DiceCE loss
    assert "scipy" in content  # for bbox extraction
    assert "trackio" in content


def test_training_script_imports_pure_helpers():
    """The 3 pure helpers + bbox extraction are importable from the training script."""
    from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel import (
        compute_per_pixel_miou,
        compute_per_class_iou,
        compute_argmax_disagreement_rate,
        extract_bbox_prompt_from_gt_mask,
    )
    assert callable(compute_per_pixel_miou)
    assert callable(compute_per_class_iou)
    assert callable(compute_argmax_disagreement_rate)
    assert callable(extract_bbox_prompt_from_gt_mask)
