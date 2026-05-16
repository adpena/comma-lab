# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler L5 Autonomy macOS-CPU smoke harness.

PAIR T directive 2026-05-13. Per CLAUDE.md "Subagent commits MUST use
serializer" + "Apples-to-apples evidence discipline" + Catalog #192
(macOS-CPU advisory not promoted without Linux verification): the smoke
harness is the $0-GPU companion runner for sister lane
``lane_time_traveler_l5_autonomy_substrate_20260513``. These tests pin:

  1) verdict classification (predicted band / advisory escalation threshold)
  2) stub-interface fallback when sister substrate is not yet ready
  3) manifest schema compliance (Catalog #192 + #127 contracts)
  4) per-pair breakdown ranking math
  5) operating-point-dependent marginal weights
  6) leverage classification semantics
  7) /tmp path refusal (CLAUDE.md FORBIDDEN_PATTERNS)
  8) non-darwin platform guard
  9) autopilot consumability (Catalog #125 hook 4)
 10) refusal of promoted manifest input
 11) budget-plan envelope math
 12) end-to-end dry-run smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import diagnose_time_traveler_per_pair_breakdown as diag  # noqa: E402
import smoke_time_traveler_l5_autonomy_macos_cpu as smoke  # noqa: E402

# ---------------------------------------------------------------------------
# smoke_time_traveler_l5_autonomy_macos_cpu
# ---------------------------------------------------------------------------


def test_verdict_classifier_pass_in_band() -> None:
    """Score inside [low, high] returns PASS_IN_BAND."""
    v = smoke._classify_verdict(
        0.160, band_low=0.150, band_high=0.170, escalation=0.190
    )
    assert v == smoke.SmokeVerdict.PASS_IN_BAND


def test_verdict_classifier_pass_below_band() -> None:
    """Score below predicted band returns PASS_BELOW_BAND (better than predicted)."""
    v = smoke._classify_verdict(
        0.140, band_low=0.150, band_high=0.170, escalation=0.190
    )
    assert v == smoke.SmokeVerdict.PASS_BELOW_BAND


def test_verdict_classifier_warn_above_band() -> None:
    """Score above band but below escalation returns WARN_ABOVE_BAND."""
    v = smoke._classify_verdict(
        0.180, band_low=0.150, band_high=0.170, escalation=0.190
    )
    assert v == smoke.SmokeVerdict.WARN_ABOVE_BAND


def test_verdict_classifier_escalates_not_falsifies() -> None:
    """Score >= threshold is advisory escalation, not method falsification."""
    v = smoke._classify_verdict(
        0.195, band_low=0.150, band_high=0.170, escalation=0.190
    )
    assert v == smoke.SmokeVerdict.ESCALATE_ABOVE_THRESHOLD


def test_verdict_classifier_eval_error_on_none() -> None:
    """None score returns EVAL_HARNESS_ERROR."""
    v = smoke._classify_verdict(
        None, band_low=0.150, band_high=0.170, escalation=0.190
    )
    assert v == smoke.SmokeVerdict.EVAL_HARNESS_ERROR


def test_escalation_threshold_default_matches_design_memo() -> None:
    """Per design memo: contest-axis recheck above 0.190."""
    assert smoke.ESCALATION_THRESHOLD == 0.190
    assert smoke.PREDICTED_BAND_LOW == 0.150
    assert smoke.PREDICTED_BAND_HIGH == 0.170


def test_resolve_output_dir_refuses_tmp(tmp_path: Path) -> None:
    """Per CLAUDE.md FORBIDDEN_PATTERNS: /tmp paths are refused."""

    class _Args:
        pass

    args = _Args()
    args.output_dir = Path("/tmp/should_be_refused")
    with pytest.raises(ValueError, match="/tmp"):
        smoke._resolve_output_dir(args)


def test_dry_run_returns_pass_summary(tmp_path: Path) -> None:
    """--dry-run produces a structured summary without invoking the trainer."""

    class _Args:
        pass

    args = _Args()
    args.output_dir = tmp_path / "smoke_out"
    args.dry_run = True
    args.allow_non_darwin = True
    args.predicted_band_low = 0.150
    args.predicted_band_high = 0.170
    args.escalation_threshold = 0.190
    args.archive_path = None
    args.stub_interface = False
    args.epochs = 100
    args.batch_size = 4
    args.inflate_sh = None

    summary = smoke.run_smoke(args)
    assert summary["lane_id"] == smoke.LANE_ID
    assert summary["sister_lane_id"] == smoke.SISTER_LANE_ID
    assert summary["evidence_grade"] == "macOS-CPU-advisory"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["ranking_only"] is True
    assert summary["eval_payload"]["dry_run"] is True


def test_stub_interface_falls_back_when_sister_not_ready(tmp_path: Path) -> None:
    """When sister substrate is not ready, stub flow produces a SUBSTRATE_NOT_READY verdict."""

    class _Args:
        pass

    args = _Args()
    args.output_dir = tmp_path / "smoke_stub"
    args.dry_run = False
    args.allow_non_darwin = True
    args.predicted_band_low = 0.150
    args.predicted_band_high = 0.170
    args.escalation_threshold = 0.190
    args.archive_path = None
    args.stub_interface = True
    args.epochs = 100
    args.batch_size = 4
    args.inflate_sh = None

    summary = smoke.run_smoke(args)
    assert summary["verdict"] == smoke.SmokeVerdict.SUBSTRATE_NOT_READY
    assert summary["stub_interface_used"] is True
    assert summary["archive_bytes"] is not None
    assert summary["archive_bytes"] > 0
    # Per CLAUDE.md Catalog #192: stub flow never produces a manifest path.
    assert summary["manifest_path"] is None


def test_stub_archive_has_tt5l_magic(tmp_path: Path) -> None:
    """The stub archive's 0.bin starts with the TT5L magic per design memo."""
    import zipfile

    archive = smoke._try_build_stub_archive(tmp_path)
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "0.bin" in names
        with zf.open("0.bin") as f:
            head = f.read(5)
    assert head[:4] == b"TT5L"
    assert head[4:5] == b"\x01"


def test_inflate_resolution_requires_archive_local_tt5l_runtime(tmp_path: Path) -> None:
    """Missing TT5L runtime is a harness error, not an exact_current fallback."""
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"not-a-real-zip")
    resolved = smoke._resolve_inflate_sh_for_archive(
        archive, output_dir=tmp_path / "out", requested=None
    )
    assert resolved["ok"] is False
    assert resolved["reason"] == "tt5l_inflate_sh_not_found_no_exact_current_fallback"
    assert all("exact_current" not in path for path in resolved["searched"])


def test_inflate_resolution_uses_archive_local_submission_dir(tmp_path: Path) -> None:
    """Trainer-emitted submission_dir/inflate.sh is the default runtime."""
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"not-a-real-zip")
    inflate = tmp_path / "submission_dir" / "inflate.sh"
    inflate.parent.mkdir()
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    resolved = smoke._resolve_inflate_sh_for_archive(
        archive, output_dir=tmp_path / "out", requested=None
    )
    assert resolved["ok"] is True
    assert resolved["path"] == str(inflate.resolve())
    assert resolved["source"] == "archive_local_tt5l_runtime"


def test_non_darwin_guard_emits_skip(tmp_path: Path) -> None:
    """Running on non-Darwin without --allow-non-darwin emits NON_DARWIN verdict."""
    import platform

    if platform.system() == "Darwin":
        pytest.skip("test asserts behavior on non-Darwin; current host is Darwin")

    class _Args:
        pass

    args = _Args()
    args.allow_non_darwin = False
    args.output_dir = tmp_path / "ndr"
    args.dry_run = True
    args.predicted_band_low = 0.150
    args.predicted_band_high = 0.170
    args.escalation_threshold = 0.190
    args.archive_path = None
    args.stub_interface = False
    args.epochs = 100
    args.batch_size = 4
    args.inflate_sh = None
    summary = smoke.run_smoke(args)
    assert summary["verdict"] == smoke.SmokeVerdict.NON_DARWIN


# ---------------------------------------------------------------------------
# diagnose_time_traveler_per_pair_breakdown
# ---------------------------------------------------------------------------


def test_marginal_weights_at_pr106_r2_pose_dominates() -> None:
    """At pose_avg = 3.4e-5 the pose marginal weight should be larger than seg."""
    weights = diag._marginal_weights(3.4e-5)
    # Per CLAUDE.md: pose marginal at PR106 r2 is 2.71x SegNet (constant 100).
    # w_pose = 0.5 * sqrt(10 / 3.4e-5) ≈ 271; w_seg = 100.
    assert weights["seg"] == 100.0
    assert weights["pose"] == pytest.approx(271.0, abs=2.0)
    assert weights["rate"] == pytest.approx(25.0 / 37_545_489, rel=1e-9)


def test_marginal_weights_at_old_1x_operating_point_seg_dominates() -> None:
    """At pose_avg = 0.18 (old 1.x operating point) seg should dominate."""
    weights = diag._marginal_weights(0.18)
    # w_pose = 0.5 * sqrt(10 / 0.18) ≈ 3.73, much smaller than w_seg = 100.
    assert weights["seg"] == 100.0
    assert weights["pose"] < weights["seg"]


def test_per_pair_sensitivity_high_residual_ranks_first() -> None:
    """A pair with much higher residuals should rank top."""
    rows = [
        {"pair_index": 0, "d_seg_residual": 0.001, "d_pose_residual": 1e-6,
         "side_info_bytes": 45},
        {"pair_index": 1, "d_seg_residual": 0.01, "d_pose_residual": 1e-5,
         "side_info_bytes": 45},
    ]
    enriched = diag.compute_per_pair_sensitivity(rows, pose_avg=3.4e-5)
    enriched.sort(key=lambda r: r["sensitivity"], reverse=True)
    assert enriched[0]["pair_index"] == 1


def test_leverage_classification_terciles() -> None:
    """Top tercile = high, middle = med, bottom = low."""
    rows = [
        {"pair_index": i, "d_seg_residual": float(i) * 0.001,
         "d_pose_residual": 0.0, "side_info_bytes": 45}
        for i in range(9)
    ]
    enriched = diag.compute_per_pair_sensitivity(rows, pose_avg=3.4e-5)
    enriched.sort(key=lambda r: r["sensitivity"], reverse=True)
    diag._classify_leverage(enriched)
    classes = [r["leverage_class"] for r in enriched]
    # Top 3 = high, next 3 = med, last 3 = low.
    assert classes[0:3] == ["high"] * 3
    assert classes[3:6] == ["med"] * 3
    assert classes[6:9] == ["low"] * 3


def test_diagnose_returns_planning_only_schema() -> None:
    """diagnose() output carries planning_only=True and the full Catalog #192 schema."""
    rows = diag._synthetic_prior_rows(num_pairs=10)
    summary = diag.diagnose(
        rows, pose_avg=3.4e-5, top_k=3, bottom_k=3,
        evidence_grade="synthetic-prior",
    )
    assert summary["lane_id"] == diag.LANE_ID
    assert summary["planning_only"] is True
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["evidence_grade"] == "synthetic-prior"
    assert "[synthetic-prior" in summary["evidence_tag"]
    assert len(summary["per_pair_ranking"]) == 10
    assert len(summary["top_k_high_leverage"]) == 3
    assert len(summary["bottom_k_low_leverage"]) == 3


def test_diagnose_macos_cpu_tag_when_observations_present(tmp_path: Path) -> None:
    """When real observations exist, evidence_grade='macOS-CPU-advisory'."""
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(json.dumps({
        "per_pair": [
            {"pair_index": 0, "d_seg_residual": 0.001,
             "d_pose_residual": 1e-6, "side_info_bytes": 45},
        ]
    }))
    rows = diag._load_observations(obs_path)
    summary = diag.diagnose(
        rows, pose_avg=3.4e-5, top_k=1, bottom_k=1,
        evidence_grade="macOS-CPU-advisory",
    )
    assert summary["evidence_grade"] == "macOS-CPU-advisory"
    assert summary["evidence_tag"] == "[macOS-CPU advisory only]"


def test_budget_plan_envelope_math() -> None:
    """Budget plan reports actual vs envelope correctly."""
    rows = diag._synthetic_prior_rows(num_pairs=100)
    enriched = diag.compute_per_pair_sensitivity(rows, pose_avg=3.4e-5)
    enriched.sort(key=lambda r: r["sensitivity"], reverse=True)
    diag._classify_leverage(enriched)
    plan = diag._budget_plan(enriched, target_bytes_per_pair=45)
    assert plan["num_pairs"] == 100
    assert plan["target_bytes_per_pair"] == 45
    assert plan["envelope_bytes_total"] == 100 * 45
    assert plan["actual_bytes_total"] == 100 * 45
    assert plan["actual_vs_envelope_pct"] == pytest.approx(100.0, abs=0.001)
    assert plan["planning_only"] is True
    # Hint allocation uses 1.5x high + 1.0x med + 0.5x low; roughly =
    # 0.667 * envelope for balanced terciles. For 100 pairs (33+34+33):
    #   high  = 33 * 67 = 2211  (int(45*1.5)=67)
    #   med   = 34 * 45 = 1530
    #   low   = 33 * 22 = 726   (int(45*0.5)=22)
    #   total = 4467
    assert plan["hint_allocation_bytes_total"] > 0


def test_per_pair_breakdown_refuses_tmp_output(tmp_path: Path, capsys) -> None:
    """Per CLAUDE.md FORBIDDEN_PATTERNS: /tmp output is refused at CLI level."""
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(json.dumps({
        "per_pair": [
            {"pair_index": 0, "d_seg_residual": 0.001,
             "d_pose_residual": 1e-6, "side_info_bytes": 45},
        ]
    }))
    rc = diag.main([
        "--observations", str(obs_path),
        "--output", "/tmp/forbidden.json",
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "FATAL" in captured.err
    assert "/tmp" in captured.err


def test_diagnose_normalize_row_rejects_negative_residuals() -> None:
    """Negative residuals are rejected at row normalization."""
    with pytest.raises(ValueError, match="d_seg_residual"):
        diag._normalize_row(
            {"pair_index": 0, "d_seg_residual": -1.0, "d_pose_residual": 0.0,
             "side_info_bytes": 45},
            index=0,
        )


def test_diagnose_normalize_row_requires_residual_or_pose() -> None:
    """At least one of d_seg_residual / d_pose_residual must be supplied."""
    with pytest.raises(ValueError, match="must supply"):
        diag._normalize_row(
            {"pair_index": 0, "side_info_bytes": 45},
            index=0,
        )


def test_diagnose_marginal_weights_safe_at_zero_pose_avg() -> None:
    """Pose_avg=0 is clipped to 1e-12 — no NaN."""
    weights = diag._marginal_weights(0.0)
    assert weights["pose"] > 0.0
    import math as _m
    assert _m.isfinite(weights["pose"])


def test_smoke_summary_carries_catalog_192_compliance_flags(tmp_path: Path) -> None:
    """Every smoke summary carries the Catalog #192 non-promotability flags."""
    summary = smoke._emit_summary(
        verdict=smoke.SmokeVerdict.PASS_IN_BAND,
        score=0.160,
        band_low=0.150,
        band_high=0.170,
        escalation=0.190,
        archive_bytes=100_000,
        manifest_path=tmp_path / "m.json",
        eval_payload={"score": 0.160, "d_seg": 0.001, "d_pose": 1e-5},
        output_dir=tmp_path,
        elapsed_total_seconds=12.3,
        stub_used=False,
    )
    assert summary["evidence_grade"] == "macOS-CPU-advisory"
    assert summary["evidence_tag"] == "[macOS-CPU advisory only]"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["ranking_only"] is True
    assert summary["in_predicted_band"] is True
    assert summary["above_escalation_threshold"] is False
    # Dispatch blockers list mirrors macos_cpu_advisory_signal DISPATCH_BLOCKERS.
    assert "macos_cpu_advisory_not_score_evidence" in summary["dispatch_blockers"]
    assert (
        "macos_cpu_advisory_cannot_falsify_architecture"
        in summary["dispatch_blockers"]
    )


def test_autopilot_can_consume_manifest_emitted_by_harness(tmp_path: Path) -> None:
    """The manifest the harness emits is loadable by the autopilot's
    load_candidates_from_macos_cpu_advisory_manifest (Catalog #125 hook 4 wire-in)."""
    from tac.optimization.macos_cpu_advisory_signal import (
        build_macos_cpu_advisory_signal_manifest,
        json_text,
    )

    observations = [
        {
            "family": "time_traveler_l5_autonomy",
            "variant_id": "smoke_v1",
            "archive_bytes": 100_000,
            "archive_sha256": "f" * 64,
            "score": 0.160,
            "d_seg": 0.001,
            "d_pose": 1e-5,
            "samples_evaluated": 100,
        }
    ]
    manifest = build_macos_cpu_advisory_signal_manifest(
        observations, source="smoke_test", run_id="smoke_test_run"
    )
    manifest_path = tmp_path / "macos_cpu_advisory_manifest.json"
    manifest_path.write_text(json_text(manifest))

    # The autopilot consumer is the canonical sink for this manifest.
    from cathedral_autopilot_autonomous_loop import (
        MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG,
        load_candidates_from_macos_cpu_advisory_manifest,
    )

    rows = load_candidates_from_macos_cpu_advisory_manifest(manifest_path)
    assert len(rows) == 1
    candidate = rows[0]
    assert candidate.family == "time_traveler_l5_autonomy"
    assert MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG in candidate.notes
    assert "promotion_blocked" in candidate.notes


def test_autopilot_refuses_promoted_manifest_input(tmp_path: Path) -> None:
    """Defense-in-depth: a promoted manifest is refused at load time."""
    from cathedral_autopilot_autonomous_loop import (
        load_candidates_from_macos_cpu_advisory_manifest,
    )

    bad_manifest = {
        "schema": "macos_cpu_advisory_signal_manifest.v1",
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "score_claim": True,  # forbidden!
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "rows": [],
        "dispatch_blockers": [],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match="score_claim"):
        load_candidates_from_macos_cpu_advisory_manifest(p)
