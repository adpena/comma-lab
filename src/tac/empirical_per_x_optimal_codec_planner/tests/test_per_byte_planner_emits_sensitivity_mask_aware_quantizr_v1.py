# SPDX-License-Identifier: MIT
"""Tier 4 canonical demonstration: the planner emits sensitivity_mask_aware_quantizr_v1.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md Section 6]
[verified-against: .omx/research/master_gradient_xray_fields_medal_research_wave_20260518.md Section 4.1]

The Fields-Medal subagent proposed:
    sensitivity_mask_aware_quantizr_v1 (top 2% fp16 / next 5% int8 / next 20% int6 / remaining 73% int4)
    predicted union ΔS [-0.018, -0.005] → [0.174, 0.187]

This test verifies the per-X planner emits EXACTLY this design when given the
canonical fec6 archive + operator's exact codec menu + 300KB byte budget.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.empirical_per_x_optimal_codec_planner import (
    PerXCodecAssignmentPlan,
    plan_per_byte_for_archive_via_sensitivity_quantiles,
    plan_per_byte_from_master_gradient,
)


FEC6_ARCHIVE_SHA256 = "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
FEC6_NPY_PATH = ".omx/state/master_gradient_fec6_contest_cpu_scorer_macos_host_advisory_20260517.npy"
FEC6_N_BYTES = 178417


def _make_synthetic_repo(tmp_path: Path, *, n_bytes: int = FEC6_N_BYTES) -> Path:
    """Create a synthetic repo with a master_gradient anchor + .npy sidecar."""
    state_dir = tmp_path / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic per-byte gradient: deterministic + reproducible
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal((n_bytes, 3)).astype(np.float32) * 1e-6
    # Inject high-leverage top-2% bytes (sparse-aware)
    n_top = int(n_bytes * 0.02)
    arr[:n_top] *= 50.0  # make top 2% have much higher sensitivity

    npy_path = state_dir / "test_grad.npy"
    np.save(npy_path, arr)

    # Anchor JSONL
    anchor = {
        "archive_sha256": "test_archive_sha256_" + "f" * 44,
        "operating_point": {"d_seg": 5.6e-4, "d_pose": 3.286e-5, "rate": 4.748e-3, "score": 0.19285},
        "gradient_array_path": ".omx/state/test_grad.npy",
        "n_bytes": n_bytes,
        "measurement_method": "synthetic_test",
        "measurement_axis": "[predicted]",
        "measurement_hardware": "synthetic_test",
        "measurement_call_id": "test",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "pareto_facets": [],
        "rashomon_disagreement_score": None,
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "n_pairs": None,
        "schema_version": "master_gradient_anchor_v1",
    }
    jsonl_path = state_dir / "master_gradient_anchors.jsonl"
    jsonl_path.write_text(json.dumps(anchor) + "\n")

    return tmp_path


def test_planner_emits_canonical_4_class_assignment(tmp_path: Path) -> None:
    """The planner should emit exactly 4 sensitivity classes matching the v1 design."""
    repo = _make_synthetic_repo(tmp_path)
    plan = plan_per_byte_from_master_gradient(
        archive_sha256="test_archive_sha256_" + "f" * 44,
        codec_menu=("fp16", "int8", "int6", "int4"),
        byte_budget=300_000,
        sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
        repo_root=repo,
    )

    assert isinstance(plan, PerXCodecAssignmentPlan)
    assert plan.x_granularity == "byte"
    assert plan.codec_menu == ("fp16", "int8", "int6", "int4")
    assert plan.evidence_grade == "predicted"
    assert plan.byte_budget == 300_000

    # Class summary verifies the canonical 4-class structure
    summary = plan.class_summary()
    assert "top_2pct" in summary, f"missing top_2pct class; got {list(summary)}"
    assert "top_5pct" in summary, f"missing top_5pct class; got {list(summary)}"
    assert "top_20pct" in summary, f"missing top_20pct class; got {list(summary)}"
    assert "tail" in summary, f"missing tail class; got {list(summary)}"

    # Codec assignments per class
    assert summary["top_2pct"]["codec"] == "fp16"
    assert summary["top_5pct"]["codec"] == "int8"
    assert summary["top_20pct"]["codec"] == "int6"
    assert summary["tail"]["codec"] == "int4"

    # Class sizes (cumulative quantile partition)
    n = FEC6_N_BYTES
    expected_top_2 = int(n * 0.02)
    expected_top_5 = int(n * 0.05) - int(n * 0.02)
    expected_top_20 = int(n * 0.20) - int(n * 0.05)
    expected_tail = n - int(n * 0.20)
    assert summary["top_2pct"]["n_bytes"] == expected_top_2, (
        f"top_2pct count: got {summary['top_2pct']['n_bytes']}, expected {expected_top_2}"
    )
    assert summary["top_5pct"]["n_bytes"] == expected_top_5
    assert summary["top_20pct"]["n_bytes"] == expected_top_20
    assert summary["tail"]["n_bytes"] == expected_tail


def test_planner_assigns_highest_sensitivity_to_highest_precision(tmp_path: Path) -> None:
    """Top 2% bytes (highest sensitivity) MUST be assigned fp16, not int4."""
    repo = _make_synthetic_repo(tmp_path)
    plan = plan_per_byte_from_master_gradient(
        archive_sha256="test_archive_sha256_" + "f" * 44,
        codec_menu=("fp16", "int8", "int6", "int4"),
        byte_budget=300_000,
        sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
        repo_root=repo,
    )
    # Sort assignments by sensitivity descending
    sorted_assignments = sorted(plan.assignments, key=lambda r: -r.sensitivity_score)
    n_top_2pct = int(FEC6_N_BYTES * 0.02)
    # Top n_top_2pct must all be fp16
    for r in sorted_assignments[:n_top_2pct]:
        assert r.chosen_codec == "fp16", (
            f"top-sensitivity byte {r.x_index} (sens={r.sensitivity_score:.3e}) "
            f"assigned to {r.chosen_codec!r} instead of fp16"
        )
    # Bottom 80% must all be int4
    n_tail = FEC6_N_BYTES - int(FEC6_N_BYTES * 0.20)
    for r in sorted_assignments[-n_tail:]:
        assert r.chosen_codec == "int4", (
            f"tail byte {r.x_index} (sens={r.sensitivity_score:.3e}) "
            f"assigned to {r.chosen_codec!r} instead of int4"
        )


def test_planner_predicted_bytes_within_budget(tmp_path: Path) -> None:
    """The emitted plan should fit within byte_budget (it's a 'sensitivity-mask-aware' compression)."""
    repo = _make_synthetic_repo(tmp_path)
    plan = plan_per_byte_from_master_gradient(
        archive_sha256="test_archive_sha256_" + "f" * 44,
        codec_menu=("fp16", "int8", "int6", "int4"),
        byte_budget=300_000,
        sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
        repo_root=repo,
    )
    # Total predicted bytes:
    #   top 2% (3568) → fp16 = 16 bits → 7136 bytes
    #   top 3% (5352) → int8 = 8 bits → 5352 bytes
    #   top 15% (26763) → int6 = 6 bits → ceil(26763*6/8) = 20073 bytes
    #   tail 80% (142734) → int4 = 4 bits → ceil(142734*4/8) = 71367 bytes
    #   TOTAL = ~103928 bytes (under 300000)
    # NOTE: counted via codec_bytes_for_n_samples which rounds up to whole bytes per class
    assert plan.total_predicted_bytes < 200_000, (
        f"per-byte plan emitted total_predicted_bytes={plan.total_predicted_bytes}; "
        f"expected ~103928 for sensitivity_mask_aware_quantizr_v1 design"
    )
    assert plan.total_predicted_bytes_within_budget is True


def test_planner_emits_provenance_per_catalog_323(tmp_path: Path) -> None:
    """Every emitted plan must carry canonical Provenance per Catalog #323."""
    repo = _make_synthetic_repo(tmp_path)
    plan = plan_per_byte_from_master_gradient(
        archive_sha256="test_archive_sha256_" + "f" * 44,
        codec_menu=("fp16", "int8", "int6", "int4"),
        byte_budget=300_000,
        sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
        repo_root=repo,
    )
    # Provenance required fields
    assert "kind" in plan.provenance
    assert "source_artifact_path" in plan.provenance
    assert "captured_at_utc" in plan.provenance
    assert plan.provenance["kind"] == "predicted_from_master_gradient"
    assert plan.provenance["score_claim"] is False
    assert plan.provenance["promotion_eligible"] is False
    assert plan.provenance["evidence_grade"] == "predicted"


def test_planner_uses_advisory_correction_not_false_contest_axis_anchor(
    tmp_path: Path,
) -> None:
    """Per-X planning may use advisory gradients, but not stale contest-axis labels."""
    repo = _make_synthetic_repo(tmp_path, n_bytes=100)
    ledger = repo / ".omx/state/master_gradient_anchors.jsonl"
    correction = json.loads(ledger.read_text().splitlines()[0])
    stale = {
        **correction,
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_per_parameter_projected_8pair_subset",
        "measurement_utc": "2026-05-18T00:00:00Z",
    }
    correction.update(
        {
            "measurement_axis": "[macOS-CPU advisory]",
            "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
            "measurement_method": "autograd_per_parameter_projected_8pair_subset_axis_correction",
            "measurement_utc": "2026-05-18T01:00:00Z",
        }
    )
    ledger.write_text(json.dumps(stale) + "\n" + json.dumps(correction) + "\n")

    plan = plan_per_byte_from_master_gradient(
        archive_sha256="test_archive_sha256_" + "f" * 44,
        codec_menu=("fp16", "int8", "int6", "int4"),
        byte_budget=300_000,
        sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
        repo_root=repo,
    )

    assert plan.measurement_axis == "[macOS-CPU advisory]"
    assert plan.provenance["measurement_axis"] == "[macOS-CPU advisory]"


def test_planner_rejects_only_false_contest_axis_anchor(tmp_path: Path) -> None:
    repo = _make_synthetic_repo(tmp_path, n_bytes=100)
    ledger = repo / ".omx/state/master_gradient_anchors.jsonl"
    stale = json.loads(ledger.read_text().splitlines()[0])
    stale.update(
        {
            "measurement_axis": "[contest-CUDA]",
            "measurement_hardware": "darwin_arm64_m5_max_macos_mps_advisory",
            "measurement_method": "autograd_per_parameter_projected_8pair_subset",
            "measurement_utc": "2026-05-18T00:00:00Z",
        }
    )
    ledger.write_text(json.dumps(stale) + "\n")

    from tac.empirical_per_x_optimal_codec_planner import PlannerError

    with pytest.raises(PlannerError, match="no master_gradient anchor"):
        plan_per_byte_from_master_gradient(
            archive_sha256="test_archive_sha256_" + "f" * 44,
            codec_menu=("fp16", "int8", "int6", "int4"),
            byte_budget=300_000,
            sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
            repo_root=repo,
        )


def test_planner_rejects_invalid_codec_menu(tmp_path: Path) -> None:
    """Codec menu must contain only canonical codec names."""
    repo = _make_synthetic_repo(tmp_path)
    from tac.empirical_per_x_optimal_codec_planner import PlannerError
    with pytest.raises(PlannerError, match="not in canonical CODEC_NAMES"):
        plan_per_byte_from_master_gradient(
            archive_sha256="test_archive_sha256_" + "f" * 44,
            codec_menu=("fp16", "bogus_codec"),
            byte_budget=300_000,
            sensitivity_threshold_quantiles=(0.5, 1.0),
            repo_root=repo,
        )


def test_planner_rejects_mismatched_lengths(tmp_path: Path) -> None:
    """codec_menu and sensitivity_threshold_quantiles must have same length."""
    repo = _make_synthetic_repo(tmp_path)
    from tac.empirical_per_x_optimal_codec_planner import PlannerError
    with pytest.raises(PlannerError, match="must equal len"):
        plan_per_byte_from_master_gradient(
            archive_sha256="test_archive_sha256_" + "f" * 44,
            codec_menu=("fp16", "int8", "int4"),
            byte_budget=300_000,
            sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),  # 4 vs 3 codecs
            repo_root=repo,
        )


def test_planner_rejects_quantiles_not_ending_at_1(tmp_path: Path) -> None:
    """sensitivity_threshold_quantiles[-1] must equal 1.0 (complete partition)."""
    repo = _make_synthetic_repo(tmp_path)
    from tac.empirical_per_x_optimal_codec_planner import PlannerError
    with pytest.raises(PlannerError, match=r"must equal 1.0"):
        plan_per_byte_from_master_gradient(
            archive_sha256="test_archive_sha256_" + "f" * 44,
            codec_menu=("fp16", "int8", "int6", "int4"),
            byte_budget=300_000,
            sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 0.90),  # ends at 0.9
            repo_root=repo,
        )


def test_planner_rejects_quantiles_not_monotonic(tmp_path: Path) -> None:
    """sensitivity_threshold_quantiles must be monotonically increasing."""
    repo = _make_synthetic_repo(tmp_path)
    from tac.empirical_per_x_optimal_codec_planner import PlannerError
    with pytest.raises(PlannerError, match="monotonically increasing"):
        plan_per_byte_from_master_gradient(
            archive_sha256="test_archive_sha256_" + "f" * 44,
            codec_menu=("fp16", "int8", "int6", "int4"),
            byte_budget=300_000,
            sensitivity_threshold_quantiles=(0.20, 0.05, 0.02, 1.00),  # not monotonic
            repo_root=repo,
        )


def test_planner_raises_on_missing_anchor(tmp_path: Path) -> None:
    """If no master_gradient anchor exists, raise PlannerError with actionable message."""
    from tac.empirical_per_x_optimal_codec_planner import PlannerError
    with pytest.raises(PlannerError) as excinfo:
        plan_per_byte_from_master_gradient(
            archive_sha256="nonexistent_archive",
            codec_menu=("fp16", "int8", "int6", "int4"),
            byte_budget=300_000,
            sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
            repo_root=tmp_path,  # empty repo
        )
    message = str(excinfo.value)
    assert "no master_gradient_anchors.jsonl" in message
    assert "--target local-cpu" not in message
    assert "--axis '[macOS-CPU advisory]'" in message
    assert "--device cpu" in message
    assert "--inflate-py <submission_dir/inflate.py>" in message


def test_planner_canonical_alias_equivalent() -> None:
    """plan_per_byte_for_archive_via_sensitivity_quantiles is canonical alias."""
    assert (
        plan_per_byte_for_archive_via_sensitivity_quantiles
        is plan_per_byte_from_master_gradient
    )


@pytest.mark.skipif(
    not Path("/Users/adpena/Projects/pact/.omx/state/master_gradient_fec6_contest_cpu_scorer_macos_host_advisory_20260517.npy").exists(),
    reason="fec6 master gradient .npy not present (this is the live-repo regression test)",
)
def test_planner_on_live_fec6_emits_sensitivity_mask_aware_quantizr_v1() -> None:
    """LIVE REGRESSION: planner output on actual fec6 archive matches v1 design.

    This is the Tier 4 deliverable test from the design memo Section 6.

    Per master_gradient_anchors.jsonl the fec6 archive is sha256
    f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd
    n_bytes 178417.

    The plan must:
    1. emit ~178417 assignments
    2. top 2% (3568) bytes → fp16
    3. next 3% (5352) bytes → int8
    4. next 15% (26763) bytes → int6
    5. tail 80% (142734) bytes → int4
    6. total_predicted_bytes ≤ 300_000
    """
    repo = Path("/Users/adpena/Projects/pact")
    plan = plan_per_byte_from_master_gradient(
        archive_sha256=FEC6_ARCHIVE_SHA256,
        codec_menu=("fp16", "int8", "int6", "int4"),
        byte_budget=300_000,
        sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
        repo_root=repo,
    )

    # Exact n_bytes from canonical anchor
    assert len(plan.assignments) == FEC6_N_BYTES, (
        f"expected {FEC6_N_BYTES} assignments, got {len(plan.assignments)}"
    )

    summary = plan.class_summary()
    expected_top_2 = int(FEC6_N_BYTES * 0.02)  # 3568
    expected_top_5 = int(FEC6_N_BYTES * 0.05) - int(FEC6_N_BYTES * 0.02)  # 5352
    expected_top_20 = int(FEC6_N_BYTES * 0.20) - int(FEC6_N_BYTES * 0.05)  # 26763
    expected_tail = FEC6_N_BYTES - int(FEC6_N_BYTES * 0.20)  # 142734
    assert summary["top_2pct"]["n_bytes"] == expected_top_2
    assert summary["top_5pct"]["n_bytes"] == expected_top_5
    assert summary["top_20pct"]["n_bytes"] == expected_top_20
    assert summary["tail"]["n_bytes"] == expected_tail

    # All emitted plan fits in 300KB budget
    assert plan.total_predicted_bytes <= 300_000
    assert plan.total_predicted_bytes_within_budget

    # Provenance carries canonical Catalog #323 tokens
    assert plan.provenance["kind"] == "predicted_from_master_gradient"
    assert plan.provenance["score_claim"] is False
    assert plan.provenance["evidence_grade"] == "predicted"

    # Measurement axis is diagnostic/advisory unless regenerated with full
    # contest-axis custody; the old local subset row must not survive as
    # authoritative [contest-CPU].
    assert plan.measurement_axis == "[macOS-CPU advisory]"
    assert plan.provenance["promotion_eligible"] is False
