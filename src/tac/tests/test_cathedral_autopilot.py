"""Tests for tools/cathedral_autopilot.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "cathedral_autopilot.py"


def _load_autopilot():
    spec = importlib.util.spec_from_file_location("cathedral_autopilot", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cathedral_autopilot"] = module
    spec.loader.exec_module(module)
    return module


def test_pr106_frontier_targets_arch_techniques() -> None:
    """At PR106 frontier with 0.155 target, top-3 should be architecture
    techniques (not encoder) because architecture has 5-10x more headroom."""
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178144,
        target_score=0.155, label="pr106_frontier",
    )
    arch_names = {t["name"] for t in plan.arch_technique_ranking}
    top_3_names = [t["name"] for t in plan.recommended_top_3]
    # All top-3 should be from architecture catalog
    assert all(n in arch_names for n in top_3_names), (
        f"top-3 should all be architecture techniques at sub-0.20 target; "
        f"got {top_3_names}"
    )


def test_already_at_target_emits_note() -> None:
    """If current score <= target, autopilot says ALREADY AT TARGET."""
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=0.0, d_pose=0.0, archive_bytes=100,  # ~0.0 score
        target_score=0.5,
        label="trivial",
    )
    notes_text = " ".join(plan.notes)
    assert "ALREADY AT TARGET" in notes_text


def test_no_target_no_gap_analysis() -> None:
    """Without target_score, the gap analysis section is empty."""
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178144,
    )
    assert plan.target_score_gap_analysis == {} or "target_score" not in plan.target_score_gap_analysis


def test_score_decomposition_sums_to_total() -> None:
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=0.001, d_pose=1e-4, archive_bytes=200_000,
    )
    decomp = plan.score_geometry["decomposition"]
    total = decomp["seg_term"] + decomp["pose_term"] + decomp["rate_term"]
    assert abs(total - plan.operator_state["current_score"]) < 1e-9


def test_encoder_technique_ranking_sorted_by_score_delta() -> None:
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178144,
    )
    deltas = [r["predicted_score_delta"] for r in plan.encoder_technique_ranking]
    assert deltas == sorted(deltas, reverse=True), (
        f"encoder techniques should be sorted by score delta desc; got {deltas}"
    )


def test_techniques_at_or_above_baseline_get_zero_delta() -> None:
    """Techniques whose predicted bytes >= current archive_bytes give 0 delta."""
    autopilot = _load_autopilot()
    # Use a tiny archive so brotli-default (178144) is HIGHER than current
    plan = autopilot.build_plan(
        d_seg=0.0, d_pose=0.0, archive_bytes=50_000,
    )
    # brotli_optuna_default is 178144 > 50000 -> no improvement possible
    brotli_row = next(
        r for r in plan.encoder_technique_ranking
        if r["name"] == "brotli_optuna_default"
    )
    assert brotli_row["predicted_score_delta"] == 0.0


def test_operating_regime_pose_dominates_at_pr106_frontier() -> None:
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178144,
    )
    assert plan.score_geometry["operating_regime"]["pose_dominates"] is True
    assert plan.score_geometry["operating_regime"]["seg_dominates"] is False


def test_legacy_regime_is_seg_dominated() -> None:
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=0.001, d_pose=0.18, archive_bytes=300_000,
    )
    assert plan.score_geometry["operating_regime"]["seg_dominates"] is True


def test_target_score_gap_uses_inverse_curves() -> None:
    """The target-score gap analysis should produce feasible pose-only and
    bytes-only paths when the target is reachable in either dimension."""
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=1e-4, d_pose=5e-5, archive_bytes=160_000,
        target_score=0.190,
    )
    gap = plan.target_score_gap_analysis
    assert gap["target_score"] == 0.190
    # Both paths should be feasible at this gentle gap
    assert gap["pose_only_feasible"]
    assert gap["bytes_only_feasible"]


def test_plan_summary_renders_without_error() -> None:
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178144,
        target_score=0.155,
    )
    text = autopilot._render_plan_summary(plan)
    assert "Cathedral Autopilot Plan" in text
    assert "TOP-3 RECOMMENDED ACTIONS" in text


def test_plan_from_pareto_round_trip(tmp_path: Path) -> None:
    """Build a synthetic 3-axis Pareto JSON, run autopilot from-pareto."""
    autopilot = _load_autopilot()
    pareto_json = tmp_path / "pareto.json"
    pareto_json.write_text(json.dumps({
        "candidates": [
            {"label": "frontier", "d_seg": 6.7e-4, "d_pose": 3.4e-5, "archive_bytes": 178144},
            {"label": "legacy", "d_seg": 0.001, "d_pose": 0.18, "archive_bytes": 300_000},
        ],
    }), encoding="utf-8")
    output = tmp_path / "plans.json"
    rc = autopilot.main([
        "plan-from-pareto",
        "--pareto-json", str(pareto_json),
        "--target-score", "0.155",
        "--output", str(output),
    ])
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_plans"] == 2
    assert payload["target_score"] == 0.155


def test_feedback_loop_overrides_predicted_with_empirical(tmp_path: Path) -> None:
    """Median over evidence rows replaces the catalog's predicted bytes."""
    autopilot = _load_autopilot()
    evidence = [
        autopilot.TechniqueEvidence(
            technique="brotli_optuna_default",
            empirical_archive_bytes=170_000,
            score_claim=True,
            promotion_eligible=True,
            rank_or_kill_eligible=True,
            ready_for_exact_eval_dispatch=True,
            source="[contest-CUDA] reports/run_a.json",
        ),
        autopilot.TechniqueEvidence(
            technique="brotli_optuna_default",
            empirical_archive_bytes=171_000,
            score_claim=True,
            promotion_eligible=True,
            rank_or_kill_eligible=True,
            ready_for_exact_eval_dispatch=True,
            source="[contest-CUDA] reports/run_b.json",
        ),
        autopilot.TechniqueEvidence(
            technique="brotli_optuna_default",
            empirical_archive_bytes=169_500,
            score_claim=True,
            promotion_eligible=True,
            rank_or_kill_eligible=True,
            ready_for_exact_eval_dispatch=True,
            source="[contest-CUDA] reports/run_c.json",
        ),
    ]
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144,
        prior_evidence=evidence,
    )
    brotli_row = next(
        r for r in plan.encoder_technique_ranking
        if r["name"] == "brotli_optuna_default"
    )
    # Median of [169500, 170000, 171000] is 170000
    assert brotli_row["predicted_archive_bytes"] == 170_000
    assert brotli_row["empirical_anchor_n"] == 3
    assert brotli_row["catalog_prior_bytes"] == 178_144
    assert "empirical-anchor-N3" in brotli_row["evidence_grade"]
    assert brotli_row["empirical_anchor_promotable"] is True


def test_feedback_loop_preserves_non_promotable_cpu_anchor() -> None:
    """CPU/MPS byte anchors can inform planning but must stay non-promotable."""
    autopilot = _load_autopilot()
    evidence = [
        autopilot.TechniqueEvidence(
            technique="lossy_int4_quantization",
            empirical_archive_bytes=100_799,
            evidence_semantics="cpu_lossy_int4_quantization_byte_anchor_no_decode_no_score",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            dispatch_blockers=["missing_exact_cuda_auth_eval"],
            source="[CPU-prep empirical] experiments/results/a1/lossy_int4_manifest.json",
        ),
    ]
    updated = autopilot.update_catalog_from_evidence(
        autopilot.ARCH_TECHNIQUES,
        evidence,
    )
    int4_row = next(r for r in updated if r["name"] == "lossy_int4_quantization")

    assert int4_row["predicted_archive_bytes"] == 100_799
    assert int4_row["empirical_anchor_promotable"] is False
    assert int4_row["empirical_anchor_score_claim"] is False
    assert int4_row["ready_for_exact_eval_dispatch"] is False
    assert int4_row["rank_or_kill_eligible"] is False
    assert "planning-only" in int4_row["evidence_grade"]
    assert "missing_exact_cuda_auth_eval" in int4_row["dispatch_blockers"]


def test_feedback_loop_treats_missing_custody_flags_as_planning_only() -> None:
    """Rows without explicit exact-eval custody can anchor planning, not promotion."""
    autopilot = _load_autopilot()
    evidence = [
        autopilot.TechniqueEvidence(
            technique="tiny_nn_pmf_predictor",
            empirical_archive_bytes=178_779,
            source="[CPU-prep] reports/pr101_tiny_nn_pmf_smoke.json",
        ),
    ]
    updated = autopilot.update_catalog_from_evidence(
        autopilot.ENCODER_TECHNIQUES,
        evidence,
    )
    row = next(r for r in updated if r["name"] == "tiny_nn_pmf_predictor")

    assert row["predicted_archive_bytes"] == 178_779
    assert row["empirical_anchor_promotable"] is False
    assert row["empirical_anchor_score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["rank_or_kill_eligible"] is False
    assert "planning-only" in row["evidence_grade"]
    assert (
        "empirical_anchor_not_promotable_without_explicit_exact_eval_custody"
        in row["dispatch_blockers"]
    )


def test_high_signal_filter_drops_low_delta_techniques() -> None:
    """min_score_delta=0.01 should drop techniques with predicted gain below."""
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144,
        min_score_delta=0.01,
    )
    for row in plan.encoder_technique_ranking:
        assert (
            row["predicted_score_delta"] >= 0.01
            or row is plan.encoder_technique_ranking[0]
        )


def test_high_signal_filter_always_keeps_top_one() -> None:
    """If filter would drop everything, top-1 is preserved (never empty)."""
    autopilot = _load_autopilot()
    plan = autopilot.build_plan(
        d_seg=6.7e-4, d_pose=3.4e-5, archive_bytes=178_144,
        min_score_delta=999.0,
    )
    assert len(plan.encoder_technique_ranking) >= 1
    assert len(plan.arch_technique_ranking) >= 1


def test_evidence_update_subcommand_produces_anchored_catalog(tmp_path: Path) -> None:
    """The CLI evidence-update subcommand should emit a catalog with anchors."""
    autopilot = _load_autopilot()
    ev_path = tmp_path / "evidence.jsonl"
    ev_path.write_text(
        json.dumps({
            "technique": "compressai_balle_hyperprior",
            "empirical_archive_bytes": 162_000,
            "source": "[contest-CUDA]",
        }) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "anchored.json"
    rc = autopilot.main([
        "evidence-update",
        "--prior-evidence", str(ev_path),
        "--output", str(out_path),
        "--catalog", "encoder",
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["n_evidence_rows"] == 1
    balle = next(
        t for t in payload["encoder_catalog"]
        if t["name"] == "compressai_balle_hyperprior"
    )
    assert balle["predicted_archive_bytes"] == 162_000
    assert balle["empirical_anchor_n"] == 1
