# SPDX-License-Identifier: MIT
"""Tests for handoff P5 CPU/CUDA xray tools.

Covers `tools/cpu_cuda_xray_segnet_layer_drift.py`,
`tools/cpu_cuda_xray_posenet_layer_drift.py`, and
`tools/cpu_cuda_xray_loader_drift.py` orchestration logic.

Per CLAUDE.md non-negotiables:
  - all outputs tagged `[diagnostic-not-score]`
  - score_claim=False, promotion_eligible=False, ready_for_exact_eval_dispatch=False
  - macOS-CPU records on paired comparisons trigger a mixed-substrate advisory
  - NaN L2 in fingerprint-mode fingerprint-proxy fallback
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import cpu_cuda_xray_segnet_layer_drift as seg_xray  # noqa: E402
import cpu_cuda_xray_posenet_layer_drift as pose_xray  # noqa: E402
import cpu_cuda_xray_loader_drift as loader_xray  # noqa: E402


# ── SegNet layer-drift tool ──────────────────────────────────────────────


def test_segnet_localize_first_divergence_argmax_path():
    rows = [
        {"layer_name": "encoder.blocks.0", "module_type": "Conv2d",
         "l2_relative_error": 1e-5, "max_abs_error": 0.0, "rank_top1_disagreement": None},
        {"layer_name": "encoder.blocks.1", "module_type": "Conv2d",
         "l2_relative_error": 1e-5, "max_abs_error": 0.0, "rank_top1_disagreement": None},
        {"layer_name": "segmentation_head.0", "module_type": "Conv2d",
         "l2_relative_error": 1e-3, "max_abs_error": 0.1,
         "rank_top1_disagreement": 0.02},
    ]
    out = seg_xray._localize_first_divergence(rows, threshold=1e-2)
    assert out["first_argmax_divergence"] is not None
    assert out["first_argmax_divergence"]["layer_name"] == "segmentation_head.0"
    assert out["first_l2_relative_exceedance"] is None  # 1e-3 < 1e-2


def test_segnet_localize_first_divergence_l2_path():
    rows = [
        {"layer_name": "encoder.blocks.0", "module_type": "Conv2d",
         "l2_relative_error": 1e-5, "max_abs_error": 0.0, "rank_top1_disagreement": None},
        {"layer_name": "encoder.blocks.5", "module_type": "Conv2d",
         "l2_relative_error": 5e-2, "max_abs_error": 0.05,
         "rank_top1_disagreement": None},
    ]
    out = seg_xray._localize_first_divergence(rows, threshold=1e-2)
    assert out["first_l2_relative_exceedance"] is not None
    assert out["first_l2_relative_exceedance"]["layer_name"] == "encoder.blocks.5"


def test_segnet_final_logits_summary_picks_segmentation_head():
    rows = [
        {"layer_name": "encoder.blocks.0", "module_type": "Conv2d",
         "l2_relative_error": 0.0, "rank_top1_disagreement": None},
        {"layer_name": "segmentation_head", "module_type": "SegmentationHead",
         "l2_relative_error": 1e-3, "max_abs_error": 0.05,
         "mean_abs_error": 0.001, "kl_divergence": None,
         "rank_top1_disagreement": None},
        {"layer_name": "segmentation_head.0", "module_type": "Conv2d",
         "l2_relative_error": 5e-4, "rank_top1_disagreement": None},
    ]
    out = seg_xray._segnet_final_logits_summary(rows)
    assert out["available"] is True
    # Should pick the SegmentationHead container, not the inner Conv2d
    assert out["layer_name"] == "segmentation_head"
    # Sister: when only `.0` exists, fallback
    rows2 = [
        {"layer_name": "segmentation_head.0", "module_type": "Conv2d",
         "l2_relative_error": 5e-4, "rank_top1_disagreement": None,
         "max_abs_error": 0.0, "mean_abs_error": 0.0, "kl_divergence": None},
    ]
    out2 = seg_xray._segnet_final_logits_summary(rows2)
    assert out2["available"] is True
    assert out2["layer_name"] == "segmentation_head.0"


def test_segnet_final_logits_summary_unavailable_when_no_head():
    rows = [
        {"layer_name": "encoder.blocks.0", "module_type": "Conv2d",
         "l2_relative_error": 0.0, "rank_top1_disagreement": None},
    ]
    out = seg_xray._segnet_final_logits_summary(rows)
    assert out["available"] is False


def test_segnet_summarize_stage_compounding_groups_by_stage():
    rows = [
        {"layer_name": "encoder.blocks.0.conv1", "module_type": "Conv2d",
         "l2_relative_error": 0.1, "fingerprint_only_l2_proxy": None},
        {"layer_name": "encoder.blocks.0.conv2", "module_type": "Conv2d",
         "l2_relative_error": 0.1, "fingerprint_only_l2_proxy": None},
        {"layer_name": "encoder.blocks.1.conv1", "module_type": "Conv2d",
         "l2_relative_error": 0.2, "fingerprint_only_l2_proxy": None},
        {"layer_name": "decoder.blocks.0", "module_type": "Conv2d",
         "l2_relative_error": 0.05, "fingerprint_only_l2_proxy": None},
    ]
    out = seg_xray._summarize_stage_compounding(rows)
    stages = {row["stage_key"]: row for row in out["by_stage"]}
    assert "encoder.blocks.0" in stages
    assert "encoder.blocks.1" in stages
    # encoder.blocks.0 has 2 layers with ε=0.1 each → (1.1)*(1.1) = 1.21
    assert pytest.approx(stages["encoder.blocks.0"]["compound_factor"], rel=1e-6) == 1.21
    # fingerprint source on this row set since fingerprint_only_l2_proxy is None
    # but l2_relative_error is real → "full_tensor"
    assert "full_tensor" in stages["encoder.blocks.0"]["eps_sources"]


def test_segnet_summarize_stage_compounding_fingerprint_fallback():
    rows = [
        {"layer_name": "encoder.blocks.0.conv1", "module_type": "Conv2d",
         "l2_relative_error": float("nan"), "fingerprint_only_l2_proxy": 0.05},
    ]
    out = seg_xray._summarize_stage_compounding(rows)
    stages = {row["stage_key"]: row for row in out["by_stage"]}
    assert stages["encoder.blocks.0"]["eps_sources"] == ["fingerprint_proxy"]
    assert pytest.approx(stages["encoder.blocks.0"]["compound_factor"], rel=1e-6) == 1.05


def test_segnet_detect_capture_host_macos():
    h = seg_xray._detect_capture_host()
    assert "platform" in h
    assert "evidence_grade_qualifier" in h
    # On macOS this is the path we exercise locally
    if h["is_macos_darwin"]:
        assert h["contest_compliant_cpu_substrate"] is False
        assert h["evidence_grade_qualifier"] == "macos_cpu_advisory_only"
    if h["is_linux_x86_64"]:
        assert h["contest_compliant_cpu_substrate"] is True
        assert h["evidence_grade_qualifier"] == "contest_cpu"


def test_segnet_build_report_non_promotable_fields():
    report = seg_xray._build_report(
        mode="cpu_only",
        cpu_record_path=Path("/nonexistent/cpu.pt"),
        cuda_record_path=None,
        shared_input_tensor=None,
        frame_pair_idx=0,
        drift_rows=None,
        threshold=1e-2,
        label="t",
        cpu_capture_host=None,
    )
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["evidence_grade"] == "diagnostic_not_score"
    assert report["tag"] == "[diagnostic-not-score]"
    # cpu_only with no rows ⇒ pending stub
    assert report["pending_cuda_capture"] is True


def test_segnet_build_report_paired_macos_triggers_advisory():
    report = seg_xray._build_report(
        mode="paired",
        cpu_record_path=Path("/nonexistent/cpu.pt"),
        cuda_record_path=Path("/nonexistent/cuda.pt"),
        shared_input_tensor=None,
        frame_pair_idx=0,
        drift_rows=[],
        threshold=1e-2,
        label="t",
        cpu_capture_host={
            "platform": "macOS-26.4-arm64",
            "system": "Darwin",
            "machine": "arm64",
            "is_macos_darwin": True,
            "is_linux_x86_64": False,
            "contest_compliant_cpu_substrate": False,
            "evidence_grade_qualifier": "macos_cpu_advisory_only",
        },
    )
    assert "mixed_substrate_advisory" in report
    assert "macos" in report["mixed_substrate_advisory"].lower() or "macOS" in report["mixed_substrate_advisory"]


def test_segnet_build_report_paired_linux_x86_no_advisory():
    report = seg_xray._build_report(
        mode="paired",
        cpu_record_path=Path("/nonexistent/cpu.pt"),
        cuda_record_path=Path("/nonexistent/cuda.pt"),
        shared_input_tensor=None,
        frame_pair_idx=0,
        drift_rows=[],
        threshold=1e-2,
        label="t",
        cpu_capture_host={
            "platform": "Linux-6.0.0-1018-azure-x86_64-with-glibc2.35",
            "system": "Linux",
            "machine": "x86_64",
            "is_macos_darwin": False,
            "is_linux_x86_64": True,
            "contest_compliant_cpu_substrate": True,
            "evidence_grade_qualifier": "contest_cpu",
        },
    )
    assert "mixed_substrate_advisory" not in report


# ── PoseNet layer-drift tool ─────────────────────────────────────────────


def test_posenet_fastvit_compounding_blocks_detected():
    rows = [
        {"layer_name": "vision.stem.0", "module_type": "Conv2d",
         "l2_relative_error": 0.01, "fingerprint_only_l2_proxy": None},
        {"layer_name": "vision.stages.0.blocks.0", "module_type": "RepMixerBlock",
         "l2_relative_error": 0.10, "fingerprint_only_l2_proxy": None},
        {"layer_name": "vision.stages.0.blocks.1", "module_type": "RepMixerBlock",
         "l2_relative_error": 0.10, "fingerprint_only_l2_proxy": None},
        # nested sub-modules of a block — counted in by_stage but not by_block
        {"layer_name": "vision.stages.0.blocks.0.token_mixer", "module_type": "RepMixer",
         "l2_relative_error": 0.05, "fingerprint_only_l2_proxy": None},
        {"layer_name": "hydra.final_layer.pose", "module_type": "Linear",
         "l2_relative_error": 0.20, "fingerprint_only_l2_proxy": None,
         "max_abs_error": 0.1, "mean_abs_error": 0.05, "kl_divergence": None,
         "rank_top1_disagreement": None},
    ]
    out = pose_xray._fastvit_compounding(rows)
    block_keys = {row["block_key"] for row in out["by_fastvit_block"]}
    assert "vision.stages.0.blocks.0" in block_keys
    assert "vision.stages.0.blocks.1" in block_keys
    # 2 detected blocks with ε=0.10 each → (1.10)*(1.10) = 1.21 across block-rows;
    # but fastvit_all_blocks includes ALL fastvit-block-named rows (incl nested ones).
    fb = out["fastvit_all_blocks"]
    assert fb["num_blocks_total"] >= 2
    # hydra is a separate stage
    stage_keys = {row["stage_key"] for row in out["by_stage"]}
    assert "hydra" in stage_keys


def test_posenet_localize_first_divergence_l2_path():
    rows = [
        {"layer_name": "vision.stem.0", "module_type": "Conv2d",
         "l2_relative_error": 1e-5, "max_abs_error": 0.0},
        {"layer_name": "vision.stages.0.blocks.0", "module_type": "RepMixerBlock",
         "l2_relative_error": 0.5, "max_abs_error": 0.4},
        {"layer_name": "hydra.final_layer.pose", "module_type": "Linear",
         "l2_relative_error": 0.8, "max_abs_error": 0.6,
         "mean_abs_error": 0.3, "kl_divergence": None,
         "rank_top1_disagreement": None},
    ]
    out = pose_xray._localize_first_divergence(rows, threshold=1e-2)
    assert out["first_l2_relative_exceedance"]["layer_name"] == "vision.stages.0.blocks.0"
    assert out["hydra_pose_head_layer"]["layer_name"] == "hydra.final_layer.pose"


def test_posenet_detect_capture_host_macos():
    h = pose_xray._detect_capture_host()
    assert h["evidence_grade_qualifier"] in {"contest_cpu", "macos_cpu_advisory_only"}


# ── Loader drift tool ────────────────────────────────────────────────────


def test_loader_synthesize_byte_identical():
    probe_report = {
        "comparison_available": True,
        "comparison_rows": [
            {"comparison": {"shape_match": True, "max_abs_lsb": 0.0,
                            "mean_abs_lsb": 0.0, "rms_abs_lsb": 0.0,
                            "nonzero_fraction": 0.0}},
        ],
    }
    out = loader_xray._synthesize_loader_drift_attribution(probe_report)
    assert out["loader_drift_measured"] is True
    assert out["loader_class"] == "byte_identical"
    assert out["max_abs_lsb_across_batches"] == 0.0


def test_loader_synthesize_single_lsb_drift():
    probe_report = {
        "comparison_available": True,
        "comparison_rows": [
            {"comparison": {"shape_match": True, "max_abs_lsb": 1.0,
                            "mean_abs_lsb": 0.5, "rms_abs_lsb": 0.6,
                            "nonzero_fraction": 0.5}},
        ],
    }
    out = loader_xray._synthesize_loader_drift_attribution(probe_report)
    assert out["loader_class"] == "single_lsb_drift"


def test_loader_synthesize_small_multi_lsb_drift():
    probe_report = {
        "comparison_available": True,
        "comparison_rows": [
            {"comparison": {"shape_match": True, "max_abs_lsb": 5.0,
                            "mean_abs_lsb": 2.0, "rms_abs_lsb": 2.5,
                            "nonzero_fraction": 0.8}},
        ],
    }
    out = loader_xray._synthesize_loader_drift_attribution(probe_report)
    assert out["loader_class"] == "small_multi_lsb_drift"


def test_loader_synthesize_large_drift():
    probe_report = {
        "comparison_available": True,
        "comparison_rows": [
            {"comparison": {"shape_match": True, "max_abs_lsb": 64.0,
                            "mean_abs_lsb": 20.0, "rms_abs_lsb": 25.0,
                            "nonzero_fraction": 1.0}},
        ],
    }
    out = loader_xray._synthesize_loader_drift_attribution(probe_report)
    assert out["loader_class"] == "large_drift"
    assert "decoder-path difference" in out["interpretation"].lower()


def test_loader_synthesize_unavailable():
    probe_report = {
        "comparison_available": False,
        "comparison_unavailable_class": "missing_prerequisite",
        "comparison_unavailable_reason": "cuda_dali_runtime_available=not_checked",
    }
    out = loader_xray._synthesize_loader_drift_attribution(probe_report)
    assert out["loader_drift_measured"] is False
    assert "not measured" in out["interpretation"].lower()


def test_loader_dispatch_plan_non_promotable():
    plan = loader_xray._dispatch_plan_for_cuda_dali_cells()
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    # remote command must reference the canonical probe tool
    cmd = plan["remote_command"]
    assert any("probe_eval_loader_drift" in part for part in cmd)
