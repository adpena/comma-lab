from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "experiments" / "sweep_owv3_byte_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_test_sweep_owv3_byte_plan", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_frontier_grid_is_deterministic_and_valid() -> None:
    mod = _load_module()

    grid = mod.resolve_grid(preset="frontier")

    assert grid[0] == mod.CandidateKnobs(
        bit_budget_ratio=0.70,
        protect_threshold=0.0010,
        aggressive_threshold=1e-5,
        fallback_action="keep_asym",
    )
    assert grid == mod.resolve_grid(preset="frontier")
    assert len(grid) == (
        len(mod.PRESETS["frontier"]["bit_budget_ratios"])
        * len(mod.PRESETS["frontier"]["protect_thresholds"])
        * len(mod.PRESETS["frontier"]["aggressive_thresholds"])
    )


def test_grid_rejects_invalid_threshold_order() -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match="aggressive_threshold must be <"):
        mod.resolve_grid(
            preset="baseline",
            protect_thresholds=(1e-5,),
            aggressive_thresholds=(1e-5,),
        )


def test_candidate_id_is_stable_and_filename_safe() -> None:
    mod = _load_module()

    cid = mod.candidate_id(
        7,
        mod.CandidateKnobs(
            bit_budget_ratio=0.69,
            protect_threshold=0.0014,
            aggressive_threshold=1e-5,
        ),
    )

    assert cid == "owv3_0007_bbr0p69_protect0p0014_aggr1em05"
    assert all(ch.isalnum() or ch == "_" for ch in cid)


def test_classify_candidate_marks_only_byte_feasibility() -> None:
    mod = _load_module()

    feasible = mod.classify_candidate(
        archive_bytes=99,
        frontier_bytes=100,
        lane_g_v3_bytes=120,
    )
    blocked = mod.classify_candidate(
        archive_bytes=101,
        frontier_bytes=100,
        lane_g_v3_bytes=120,
    )

    assert feasible["byte_feasible_vs_frontier"] is True
    assert feasible["selection_status"] == "byte_feasible_pending_cuda_auth_eval"
    assert feasible["score_status"] == "not_evaluated_cuda_auth_required"
    assert blocked["byte_feasible_vs_frontier"] is False
    assert blocked["selection_status"] == "byte_infeasible_vs_frontier"


def test_manifest_only_byte_plan_is_planning_only(tmp_path: Path) -> None:
    mod = _load_module()
    sensitivity = tmp_path / "owv3_sensitivity_map.pt"
    sensitivity.write_bytes(b"fake sensitivity map custody")

    manifest = mod.build_component_sensitivity_byte_plan_manifest(
        sensitivity_map=sensitivity,
        preset="baseline",
        bit_budget_ratios=(0.7,),
        protect_thresholds=(0.001,),
        aggressive_thresholds=(1e-5,),
        fallback_action="keep_asym",
        frontier_comparator_bytes=12345,
        frontier_comparator_sha256="a" * 64,
        allow_non_authoritative=True,
        limit=1,
    )

    assert manifest["format"] == "component_sensitivity_byte_plan_manifest_v1"
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["sensitivity_map"]["exists"] is True
    assert manifest["sensitivity_map"]["bytes"] == len(b"fake sensitivity map custody")
    assert manifest["frontier_comparator"]["archive_bytes"] == 12345
    assert manifest["grid"]["candidate_count"] == 1
    assert manifest["candidates"][0]["archive_status"] == "not_built_manifest_only"
    assert manifest["fallback_action_accounting"]["keep_asym"]["promotion_eligible_before_eval"] is True
    assert "manifest_only_no_archive_bytes" in manifest["dispatch_blockers"]


def test_manifest_only_records_missing_or_non_authoritative_sensitivity(tmp_path: Path) -> None:
    mod = _load_module()

    manifest = mod.build_component_sensitivity_byte_plan_manifest(
        sensitivity_map=tmp_path / "missing.pt",
        preset="baseline",
        fallback_action="diagnostic_fp16",
        allow_non_authoritative=False,
        limit=1,
    )

    assert manifest["sensitivity_map"]["exists"] is False
    assert manifest["fallback_action_accounting"]["diagnostic_fp16"][
        "promotion_eligible_before_eval"
    ] is False
    assert "missing_sensitivity_map" in manifest["dispatch_blockers"]
    assert "authoritative_cuda_sensitivity_required" in manifest["dispatch_blockers"]


def test_select_best_byte_candidate_uses_closest_under_frontier() -> None:
    mod = _load_module()

    rows = [
        {
            "candidate_id": "too_large",
            "build_status": "ok",
            "knobs": {
                "bit_budget_ratio": 0.7,
                "protect_threshold": 0.001,
                "aggressive_threshold": 1e-5,
            },
            "archive": {"size_bytes": 101},
            "byte_classification": {
                "frontier_bytes": 100,
                "byte_feasible_vs_frontier": False,
            },
        },
        {
            "candidate_id": "smaller",
            "build_status": "ok",
            "knobs": {
                "bit_budget_ratio": 0.7,
                "protect_threshold": 0.002,
                "aggressive_threshold": 1e-5,
            },
            "archive": {"size_bytes": 95},
            "byte_classification": {
                "frontier_bytes": 100,
                "byte_feasible_vs_frontier": True,
            },
        },
        {
            "candidate_id": "closest",
            "build_status": "ok",
            "knobs": {
                "bit_budget_ratio": 0.69,
                "protect_threshold": 0.0014,
                "aggressive_threshold": 1e-5,
            },
            "archive": {"size_bytes": 99},
            "byte_classification": {
                "frontier_bytes": 100,
                "byte_feasible_vs_frontier": True,
            },
        },
    ]

    assert mod.select_best_byte_candidate(rows)["candidate_id"] == "closest"


def test_select_r5_segnet_conservative_candidates_prefers_small_safe_move() -> None:
    mod = _load_module()

    def row(
        cid: str,
        *,
        low: int,
        bbr: float,
        delta: int,
        ok: bool = True,
        promotion_eligible: bool = True,
        fallback_action: str = "keep_asym",
        diagnostic_layers: int = 0,
    ) -> dict:
        return {
            "candidate_id": cid,
            "build_status": "ok" if ok else "error",
            "knobs": {"bit_budget_ratio": bbr},
            "byte_classification": {
                "byte_feasible_vs_frontier": delta <= 0,
                "frontier_delta_bytes": delta,
            },
            "owv3_byte_plan": {
                "action_counts": {
                    "diagnostic_fp16_layers": diagnostic_layers,
                    "owv2_low_bit_channels": low,
                },
                "fallback_action": fallback_action,
                "promotion_eligible": promotion_eligible,
            },
        }

    rows = [
        row("r4_failed", low=65, bbr=0.69, delta=-78),
        row("diagnostic_fp16", low=61, bbr=0.69, delta=-10, promotion_eligible=False),
        row("wrong_fallback", low=61, bbr=0.69, delta=-11, fallback_action="diagnostic_fp16", diagnostic_layers=1),
        row("not_conservative", low=66, bbr=0.70, delta=-400),
        row("too_large", low=62, bbr=0.69, delta=12),
        row("too_aggressive_bits", low=63, bbr=0.50, delta=-6000),
        row("minimal_safe", low=62, bbr=0.67, delta=-167),
        row("same_reduction_lower_bbr", low=62, bbr=0.65, delta=-20),
        row("more_conservative", low=58, bbr=0.65, delta=-104),
    ]

    ranked = mod.select_r5_segnet_conservative_candidates(
        rows,
        reference_candidate_id="r4_failed",
        limit=3,
    )

    assert [item["candidate_id"] for item in ranked] == [
        "minimal_safe",
        "same_reduction_lower_bbr",
        "more_conservative",
    ]
    first = ranked[0]["r5_paired_calibration"]
    assert first["reference_candidate_id"] == "r4_failed"
    assert first["owv2_low_bit_channel_reduction"] == 3
    assert first["min_allowed_bit_budget_ratio"] == pytest.approx(0.64)
    assert first["score_status"] == "not_evaluated_cuda_auth_required"


def test_select_r5_segnet_conservative_candidates_requires_reference() -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match="reference candidate not found"):
        mod.select_r5_segnet_conservative_candidates(
            [],
            reference_candidate_id="missing",
        )


def test_select_r6_segnet_conservative_candidates_uses_failed_r5_reference() -> None:
    mod = _load_module()

    def row(
        cid: str,
        *,
        low: int,
        bbr: float,
        delta: int,
        promotion_eligible: bool = True,
    ) -> dict:
        return {
            "candidate_id": cid,
            "build_status": "ok",
            "knobs": {"bit_budget_ratio": bbr},
            "byte_classification": {
                "byte_feasible_vs_frontier": delta <= 0,
                "frontier_delta_bytes": delta,
            },
            "owv3_byte_plan": {
                "action_counts": {
                    "diagnostic_fp16_layers": 0,
                    "owv2_low_bit_channels": low,
                },
                "fallback_action": "keep_asym",
                "promotion_eligible": promotion_eligible,
            },
        }

    rows = [
        row("r5_failed", low=62, bbr=0.67, delta=-167),
        row("same_low_bits", low=62, bbr=0.66, delta=-445),
        row("closest_r6", low=58, bbr=0.65, delta=-104),
        row("lower_bbr_r6", low=58, bbr=0.64, delta=-407),
        row("too_much_bbr_drop", low=47, bbr=0.50, delta=-1942),
        row("not_promotion_eligible", low=58, bbr=0.66, delta=-200, promotion_eligible=False),
    ]

    ranked = mod.select_r6_segnet_conservative_candidates(
        rows,
        reference_candidate_id="r5_failed",
        limit=3,
        failed_exact_reference={"candidate_id": "r5_failed", "lane_status": "COMPONENT_GATE_REVIEW_REQUIRED"},
    )

    assert [item["candidate_id"] for item in ranked] == [
        "closest_r6",
        "lower_bbr_r6",
    ]
    first = ranked[0]["r6_paired_calibration"]
    assert first["reference_candidate_id"] == "r5_failed"
    assert first["reference_owv2_low_bit_channels"] == 62
    assert first["candidate_owv2_low_bit_channels"] == 58
    assert first["owv2_low_bit_channel_reduction"] == 4
    assert first["failed_r5_exact_cuda_t4_reference"]["lane_status"] == (
        "COMPONENT_GATE_REVIEW_REQUIRED"
    )
    assert first["promotion_eligible_before_eval"] is False
    assert first["score_status"] == "not_evaluated_cuda_auth_required"


def test_select_r7_pose_balanced_candidates_blocks_lower_bbr_after_r6_pose_fail() -> None:
    mod = _load_module()

    def row(
        cid: str,
        *,
        low: int,
        bbr: float,
        delta: int,
        promotion_eligible: bool = True,
    ) -> dict:
        return {
            "candidate_id": cid,
            "build_status": "ok",
            "knobs": {"bit_budget_ratio": bbr},
            "byte_classification": {
                "byte_feasible_vs_frontier": delta <= 0,
                "frontier_delta_bytes": delta,
            },
            "owv3_byte_plan": {
                "action_counts": {
                    "diagnostic_fp16_layers": 0,
                    "owv2_low_bit_channels": low,
                },
                "fallback_action": "keep_asym",
                "promotion_eligible": promotion_eligible,
            },
        }

    rows = [
        row("r6_failed", low=58, bbr=0.65, delta=-104),
        row("more_protection_but_lower_bbr", low=55, bbr=0.63, delta=-177),
        row("less_protection_higher_bbr", low=62, bbr=0.67, delta=-167),
        row("not_promotion_eligible", low=55, bbr=0.66, delta=-100, promotion_eligible=False),
    ]

    ranked = mod.select_r7_pose_balanced_candidates(
        rows,
        reference_candidate_id="r6_failed",
        limit=3,
        failed_exact_reference={
            "candidate_id": "r6_failed",
            "lane_status": "REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED",
        },
    )

    assert ranked == []


def test_select_r7_pose_balanced_candidates_prefers_pose_safe_budget_restore() -> None:
    mod = _load_module()

    def row(cid: str, *, low: int, bbr: float, delta: int) -> dict:
        return {
            "candidate_id": cid,
            "build_status": "ok",
            "knobs": {"bit_budget_ratio": bbr},
            "byte_classification": {
                "byte_feasible_vs_frontier": delta <= 0,
                "frontier_delta_bytes": delta,
            },
            "owv3_byte_plan": {
                "action_counts": {
                    "diagnostic_fp16_layers": 0,
                    "owv2_low_bit_channels": low,
                },
                "fallback_action": "keep_asym",
                "promotion_eligible": True,
            },
        }

    rows = [
        row("r6_failed", low=58, bbr=0.65, delta=-104),
        row("same_protection_higher_bbr", low=58, bbr=0.66, delta=-12),
        row("more_protection_same_bbr", low=55, bbr=0.65, delta=-60),
        row("lower_bbr_more_protection", low=52, bbr=0.64, delta=-400),
        row("too_large", low=55, bbr=0.66, delta=8),
    ]

    ranked = mod.select_r7_pose_balanced_candidates(
        rows,
        reference_candidate_id="r6_failed",
        limit=3,
        failed_exact_reference={
            "candidate_id": "r6_failed",
            "lane_status": "REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED",
        },
    )

    assert [item["candidate_id"] for item in ranked] == [
        "same_protection_higher_bbr",
        "more_protection_same_bbr",
    ]
    first = ranked[0]["r7_pose_balanced_calibration"]
    assert first["reference_candidate_id"] == "r6_failed"
    assert first["reference_owv2_low_bit_channels"] == 58
    assert first["candidate_owv2_low_bit_channels"] == 58
    assert first["min_bit_budget_ratio"] == pytest.approx(0.65)
    assert first["failed_r6_exact_cuda_t4_reference"]["lane_status"] == (
        "REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED"
    )
    assert first["promotion_eligible_before_eval"] is False
    assert first["score_status"] == "not_evaluated_cuda_auth_required"


def test_r5_queue_entry_requires_paired_pfp16_and_strict_component_gates(tmp_path: Path) -> None:
    mod = _load_module()

    row = {
        "candidate_id": "owv3_0047_bbr0p67_protect0p00135_aggr1em05",
        "archive": {
            "sha256": "1" * 64,
            "size_bytes": 686468,
        },
        "r5_paired_calibration": {
            "reference_candidate_id": "owv3_0018_bbr0p69_protect0p0014_aggr1em05",
            "owv2_low_bit_channel_reduction": 3,
        },
    }

    queue = mod.build_r5_exact_eval_queue_entry(
        row,
        archive_path="experiments/results/r5/archive.zip",
        output_dir=tmp_path,
        baseline_archive_bytes=686635,
    )

    gates = queue["promotion_gates"]
    assert gates["paired_pfp16_calibration_required"] is True
    assert gates["required_device"] == "cuda"
    assert gates["required_gpu_t4_match"] is True
    assert gates["required_samples"] == 600
    adjudicate = queue["adjudication_command_template_after_paired_pfp16"]
    lightning = queue["lightning_exact_eval_command_template_after_paired_pfp16"]
    assert "--device" in queue["contest_auth_eval_command"]
    assert "cuda" in queue["contest_auth_eval_command"]
    assert "--baseline-segnet-dist" in adjudicate
    assert "<paired_pfp16_avg_segnet_dist>" in adjudicate
    assert "--max-segnet-relative" in adjudicate
    assert "1.002" in adjudicate
    assert "--max-posenet-relative" in adjudicate
    assert "--machine" in lightning
    assert "T4" in lightning
    assert "--expected-archive-sha256" in lightning
    assert "1" * 64 in lightning


def test_r6_queue_entry_uses_r6_lane_and_calibration_key(tmp_path: Path) -> None:
    mod = _load_module()

    row = {
        "candidate_id": "owv3_0076_bbr0p65_protect0p0013_aggr1em05",
        "archive": {
            "sha256": "3" * 64,
            "size_bytes": 686531,
        },
        "r6_paired_calibration": {
            "reference_candidate_id": "owv3_0047_bbr0p67_protect0p00135_aggr1em05",
            "owv2_low_bit_channel_reduction": 4,
        },
    }

    queue = mod.build_r5_exact_eval_queue_entry(
        row,
        archive_path="experiments/results/r6/archive.zip",
        output_dir=tmp_path,
        baseline_archive_bytes=686635,
        lane="owv3_r6_segnet_conservative_after_failed_r5",
        calibration_key="r6_paired_calibration",
    )

    assert queue["r6_paired_calibration"]["reference_candidate_id"].startswith("owv3_0047")
    assert "r5_paired_calibration" not in queue
    lightning = queue["lightning_exact_eval_command_template_after_paired_pfp16"]
    assert "lane=owv3_r6_segnet_conservative_after_failed_r5" in lightning
    assert "3" * 64 in lightning


def test_r5_queue_entry_rejects_missing_paired_calibration(tmp_path: Path) -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match="r5_paired_calibration"):
        mod.build_r5_exact_eval_queue_entry(
            {"candidate_id": "owv3_missing", "archive": {"sha256": "2" * 64, "size_bytes": 1}},
            archive_path="archive.zip",
            output_dir=tmp_path,
            baseline_archive_bytes=686635,
        )
