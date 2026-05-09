"""Regression tests for tools/probe_seg_loss_surrogate_disambiguator.py.

Pins:

* JSON schema is stable across runs (key set + types).
* Verdict logic respects the operator-defined thresholds:
  - cos sim ≥ 0.7 (max) → PRUNE
  - cos sim ≤ 0.3 (mean) AND incremental Δ ≤ -0.003 → KEEP
  - in-band → ENSEMBLE
  - high regime variance (≥ 0.20) → DEFER
* Predicted Δ score is signed correctly (negative = improvement).
* Per-regime cos sim records are present and meaningful.
* Recommendation engine prefers KEEP > ENSEMBLE > singleton baseline.
* Compositions never claim ``[contest-CUDA]`` / ``[contest-CPU]`` (only
  ``[predicted; ...]`` per CLAUDE.md "Forbidden score claims").
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_probe_module():
    """Import tools/probe_seg_loss_surrogate_disambiguator.py as a module."""
    src_path = REPO / "tools" / "probe_seg_loss_surrogate_disambiguator.py"
    spec = importlib.util.spec_from_file_location(
        "probe_seg_loss_surrogate_disambiguator", src_path
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    # Register in sys.modules BEFORE exec_module so that dataclasses defined in
    # the module can resolve their own __module__ via sys.modules lookup
    # (otherwise dataclass field annotation resolution crashes with
    # ``AttributeError: 'NoneType' object has no attribute '__dict__'``).
    sys.modules["probe_seg_loss_surrogate_disambiguator"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe_mod():
    return _load_probe_module()


@pytest.fixture(scope="module")
def report(probe_mod):
    """Run the probe once at module scope (it's CPU but ~5s); reuse for all tests."""
    return probe_mod.probe(seed=20260509, sinkhorn_blur=0.05, sinkhorn_iters=20, n_repeats=2)


def test_report_has_required_top_level_keys(report):
    expected = {
        "schema_version",
        "generated_at_utc",
        "tool",
        "inputs",
        "calibration",
        "per_regime",
        "overall_pair_cos_sim",
        "decision_matrix",
        "recommendation",
        "sub_additivity_summary",
        "claude_md_compliance",
    }
    assert expected.issubset(report.keys())


def test_schema_version_is_v1(report):
    assert report["schema_version"] == "probe_seg_loss_surrogate_disambiguator/v1"


def test_calibration_uses_R_seg_HNERV(report, probe_mod):
    cal = report["calibration"]
    # Must reflect the canonical HNeRV R_seg ≈ 1.17 from
    # cuda_cpu_axis_calibration.
    assert cal["R_seg_HNERV"] == probe_mod.R_SEG_HNERV
    assert cal["R_pose_HNERV"] == probe_mod.R_POSE_HNERV
    assert cal["score_gradient_dS_d_dseg_cpu"] == 100.0


def test_no_contest_axis_tag_emitted(report):
    """CLAUDE.md "Forbidden score claims": never claim contest-CUDA/CPU."""
    blob = json.dumps(report)
    assert "[contest-CUDA]" not in blob
    assert "[contest-CPU]" not in blob
    assert "[predicted;" in blob


def test_per_regime_records_are_present(report):
    expected_regimes = {
        "interior",
        "near_boundary",
        "sharp_disagreement",
        "soft_disagreement",
        "class_mass_swap",
    }
    assert set(report["per_regime"].keys()) == expected_regimes
    for regime_name, r in report["per_regime"].items():
        assert "per_surrogate" in r
        assert "pair_cos_sim" in r
        assert "triple_mean_cos_sim" in r
        assert {"T7_fisher_rao", "T8_sinkhorn", "T11_lovasz"}.issubset(
            r["per_surrogate"].keys()
        )


def test_overall_pair_cos_sim_in_canonical_range(report):
    """Cosine similarity is in [-1, 1] by definition."""
    for label, v in report["overall_pair_cos_sim"].items():
        assert -1.0 <= v["mean_cos_sim"] <= 1.0
        assert -1.0 <= v["min_cos_sim"] <= 1.0
        assert -1.0 <= v["max_cos_sim"] <= 1.0
        assert v["min_cos_sim"] <= v["mean_cos_sim"] <= v["max_cos_sim"]
        assert "regime_mean_std" in v
        assert v["regime_mean_std"] >= 0.0


def test_decision_matrix_has_all_compositions(report):
    """Decision matrix must contain all 7 compositions (3 single + 3 pair + 1 triple)."""
    compositions = [tuple(r["composition"]) for r in report["decision_matrix"]]
    expected = [
        ("T7_fisher_rao",),
        ("T8_sinkhorn",),
        ("T11_lovasz",),
        ("T7_fisher_rao", "T8_sinkhorn"),
        ("T7_fisher_rao", "T11_lovasz"),
        ("T8_sinkhorn", "T11_lovasz"),
        ("T7_fisher_rao", "T8_sinkhorn", "T11_lovasz"),
    ]
    assert sorted(compositions) == sorted(expected)


def test_decision_matrix_verdicts_in_canonical_set(report):
    """Verdicts must be in the operator-defined canonical set; KILL is forbidden."""
    canonical = {"BASELINE", "SINGLE_SURROGATE", "KEEP", "PRUNE", "ENSEMBLE", "DEFER"}
    for r in report["decision_matrix"]:
        assert r["verdict"] in canonical
        # CLAUDE.md kill-as-last-resort:
        assert r["verdict"] not in {"KILL", "FALSIFIED", "DEAD", "RETIRED"}


def test_predicted_delta_is_signed_correctly(probe_mod):
    """Single surrogate baselines must predict NEGATIVE delta (improvement)."""
    rep = probe_mod.probe(seed=20260509, sinkhorn_blur=0.05, sinkhorn_iters=20, n_repeats=2)
    singletons = [
        r for r in rep["decision_matrix"] if len(r["composition"]) == 1
    ]
    for r in singletons:
        assert r["predicted_delta_score"] < 0.0, (
            f"singleton {r['composition']} predicted Δ {r['predicted_delta_score']} >= 0"
        )


def test_verdict_keep_when_cos_low_and_delta_below_threshold(probe_mod):
    """KEEP requires mean_cos ≤ 0.3 AND incremental Δ ≤ -0.003 AND regime variance < 0.20."""
    # Synthetic cos-sim pairs: T7-T8 = 0.1 (low), no regime variance.
    cos_pairs = {
        frozenset(("T7_fisher_rao", "T8_sinkhorn")): 0.1,
    }
    verdict = probe_mod._verdict_for_composition(
        ("T7_fisher_rao", "T8_sinkhorn"),
        cos_pairs,
        predicted_delta=-0.10,
        delta_baseline=-0.05,
        cos_variance_across_regimes=0.05,  # low variance
    )
    assert verdict == "KEEP"


def test_verdict_prune_when_cos_above_threshold(probe_mod):
    """PRUNE when max cos sim ≥ 0.7."""
    cos_pairs = {
        frozenset(("T7_fisher_rao", "T8_sinkhorn")): 0.85,
    }
    verdict = probe_mod._verdict_for_composition(
        ("T7_fisher_rao", "T8_sinkhorn"),
        cos_pairs,
        predicted_delta=-0.10,
        delta_baseline=-0.05,
        cos_variance_across_regimes=0.05,
    )
    assert verdict == "PRUNE"


def test_verdict_defer_when_regime_variance_high(probe_mod):
    """DEFER overrides any other verdict when regime variance ≥ 0.20."""
    cos_pairs = {
        frozenset(("T7_fisher_rao", "T8_sinkhorn")): 0.1,
    }
    verdict = probe_mod._verdict_for_composition(
        ("T7_fisher_rao", "T8_sinkhorn"),
        cos_pairs,
        predicted_delta=-0.10,
        delta_baseline=-0.05,
        cos_variance_across_regimes=0.25,  # high variance
    )
    assert verdict == "DEFER"


def test_verdict_prune_when_predicted_regression(probe_mod):
    """PRUNE when delta is positive (predicted regression)."""
    cos_pairs = {
        frozenset(("T7_fisher_rao", "T8_sinkhorn")): 0.1,
    }
    verdict = probe_mod._verdict_for_composition(
        ("T7_fisher_rao", "T8_sinkhorn"),
        cos_pairs,
        predicted_delta=+0.005,  # WORSE
        delta_baseline=-0.05,
        cos_variance_across_regimes=0.05,
    )
    assert verdict == "PRUNE"


def test_recommendation_falls_back_to_singleton_when_all_defer(report):
    """Per CLAUDE.md kill-as-last-resort: when all multi-surrogate compositions
    are DEFER (high regime variance), the recommendation must fall back to the
    best singleton, NOT silently pick the most-negative-delta composition."""
    rec = report["recommendation"]
    multi_surr_rows = [
        r for r in report["decision_matrix"] if len(r["composition"]) > 1
    ]
    if all(r["verdict"] == "DEFER" for r in multi_surr_rows):
        assert len(rec["best_composition"]) == 1, (
            f"recommendation should be a singleton when all multi-surrogate "
            f"compositions are DEFER; got {rec['best_composition']}"
        )
        assert rec["verdict"] in {"BASELINE", "SINGLE_SURROGATE"}


def test_sub_additivity_summary_pose_axis_attack_signal(report):
    """Per operator brief: if max pairwise cos sim ≥ 0.3, T20 (KL pose-axis)
    is recommended over additional seg surrogates."""
    summary = report["sub_additivity_summary"]
    if summary["max_pairwise_cos_sim"] >= 0.3:
        assert summary["pose_axis_attack_recommended"] is True
    else:
        assert summary["pose_axis_attack_recommended"] is False


def test_per_regime_means_present_for_each_pair(report):
    """Each pair's overall record must include per-regime means dict."""
    for label, v in report["overall_pair_cos_sim"].items():
        assert "per_regime_means" in v
        # All 5 regimes must have a mean.
        assert set(v["per_regime_means"].keys()) == {
            "interior",
            "near_boundary",
            "sharp_disagreement",
            "soft_disagreement",
            "class_mass_swap",
        }


def test_main_writes_json_file(probe_mod, tmp_path):
    """The CLI main() must write a parseable JSON to --output."""
    out = tmp_path / "probe_results.json"
    rc = probe_mod.main(
        [
            "--output", str(out),
            "--seed", "42",
            "--n-repeats", "2",
        ]
    )
    assert rc == 0
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert parsed["schema_version"] == "probe_seg_loss_surrogate_disambiguator/v1"
    assert "decision_matrix" in parsed


def test_claude_md_compliance_block_present(report):
    """The compliance block must explicitly tag every CLAUDE.md non-negotiable
    we honor."""
    cm = report["claude_md_compliance"]
    assert cm["no_gpu_dispatch"] is True
    assert cm["no_contest_cuda_or_cpu_score_emitted"] is True
    assert cm["all_score_estimates_tagged_predicted"] is True
    assert cm["kill_as_last_resort_honored"] is True
    assert cm["verdicts_use_DEFERRED_or_PRUNE_not_KILL"] is True


def test_lovasz_runs_on_simplex_input_without_dtype_error(probe_mod):
    """Regression: T11 needed an internal fp32 cast to avoid a torch.dot dtype
    mismatch when fed fp64 inputs from this probe. Test it directly."""
    import torch

    gen = torch.Generator().manual_seed(123)
    pred, gt = probe_mod._build_interior(gen)
    assert pred.dtype == torch.float64
    g = probe_mod._gradient_wrt_pred(
        "T11_lovasz",
        pred,
        gt,
        sinkhorn_blur=0.05,
        sinkhorn_iters=20,
    )
    assert g.grad_norm > 0.0
    assert torch.isfinite(g.grad_flat).all()


def test_surrogate_gradients_decrease_their_own_loss(probe_mod):
    """Each surrogate gradient must point downhill on a simplex projection step."""
    import torch

    gen = torch.Generator().manual_seed(321)
    pred, gt = probe_mod._build_soft_disagreement(gen)

    for surrogate in probe_mod.SURROGATES:
        before = probe_mod._gradient_wrt_pred(
            surrogate,
            pred,
            gt,
            sinkhorn_blur=0.05,
            sinkhorn_iters=20,
        )
        updated = pred - 0.01 * before.grad_flat.reshape_as(pred)
        updated = updated.clamp_min(1e-7)
        updated = updated / updated.sum(dim=1, keepdim=True)
        after = probe_mod._gradient_wrt_pred(
            surrogate,
            updated,
            gt,
            sinkhorn_blur=0.05,
            sinkhorn_iters=20,
        )
        loss_before = before.loss
        loss_after = after.loss
        assert loss_after < loss_before, surrogate


def test_zero_norm_gradient_returns_nan_not_zero(probe_mod):
    """Hotz R2: zero-norm gradient cosine sim is UNDEFINED, not orthogonal.

    A pair with both gradients zero must return NaN (not 0.0) so the aggregator
    can filter it rather than silently averaging garbage.
    """
    import math

    import torch

    a = probe_mod.SurrogateGradient(
        name="A",
        loss=0.0,
        grad_norm=0.0,
        grad_flat=torch.zeros(10, dtype=torch.float64),
    )
    b = probe_mod.SurrogateGradient(
        name="B",
        loss=0.0,
        grad_norm=0.0,
        grad_flat=torch.zeros(10, dtype=torch.float64),
    )
    cos = a.cosine_similarity(b)
    assert math.isnan(cos)


def test_recommendation_calls_out_defer_count(report):
    """Quantizr R3: rationale must explicitly include the number of DEFER
    compositions when the singleton recommendation is triggered by regime
    variance."""
    rec = report["recommendation"]
    assert "defer_count_among_multi_surrogate" in rec
    assert "n_multi_surrogate_compositions" in rec


def test_seed_determinism(probe_mod):
    """Same seed → identical pairwise cos sim mean (within fp64 noise floor)."""
    r1 = probe_mod.probe(seed=42, sinkhorn_blur=0.05, sinkhorn_iters=20, n_repeats=2)
    r2 = probe_mod.probe(seed=42, sinkhorn_blur=0.05, sinkhorn_iters=20, n_repeats=2)
    for label in r1["overall_pair_cos_sim"]:
        a = r1["overall_pair_cos_sim"][label]["mean_cos_sim"]
        b = r2["overall_pair_cos_sim"][label]["mean_cos_sim"]
        assert abs(a - b) < 1e-9, f"non-deterministic for {label}: {a} vs {b}"
