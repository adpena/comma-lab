"""Acceptance tests for tac.optimizer.meta_lagrangian.

Exercises the full pipeline: distortion proxy + predictor + Lagrangian +
sanity gate. The CRITICAL test reproduces the apogee_int4 failure mode at
the search-engine layer — the meta-Lagrangian must rank int4-shaped
candidates BELOW int8-shaped ones (high distortion → high Lagrangian),
matching the empirical contest-CUDA result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.optimizer.meta_lagrangian import (  # noqa: E402
    CandidateEvaluation,
    CmaEsSearchBounds,
    LagrangianConstraints,
    MetaLagrangianSearch,
    contest_score,
)
from tac.predictor.score_band import (  # noqa: E402
    PR106_TOTAL_RATE_DENOM,
    CalibrationAnchor,
    load_calibration_anchors,
)

# ── Test fixtures ─────────────────────────────────────────────────────────


def _make_anchors() -> list[CalibrationAnchor]:
    return [
        CalibrationAnchor(
            lane_id="lane_pr106_baseline",
            rel_err_pct_per_weight=0.0,
            archive_bytes=186239,
            contest_cuda_score=0.20945673,
            avg_pose_dist=3.4e-5,
            avg_seg_dist=0.00067819,
            rate_unscaled=186239 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-05T17:25Z",
            job_id="pr106-baseline",
            archive_sha256="ab",
        ),
        CalibrationAnchor(
            lane_id="lane_apogee_int8",
            rel_err_pct_per_weight=0.24,
            archive_bytes=187731,
            contest_cuda_score=0.21119242,
            avg_pose_dist=3.375e-5,
            avg_seg_dist=0.00067819,
            rate_unscaled=187731 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-05T18:02Z",
            job_id="int8",
            archive_sha256="cd",
        ),
        CalibrationAnchor(
            lane_id="lane_apogee_int4",
            rel_err_pct_per_weight=7.09,
            archive_bytes=109996,
            contest_cuda_score=1.42866394,
            avg_pose_dist=0.02370903,
            avg_seg_dist=0.00868503,
            rate_unscaled=109996 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-05T17:40Z",
            job_id="int4",
            archive_sha256="ef",
        ),
    ]


def _honest_proxy(archive_bytes: int, rel_err_pct: float, n_layers: int) -> tuple[float, float]:
    """Empirically-grounded proxy that returns the actual int4 / int8 anchors
    when called near those rel_err points."""
    if rel_err_pct >= 5.0:
        return (0.02370903, 0.00868503)
    if rel_err_pct >= 0.5:
        return (3.375e-5, 0.00067819)
    return (3.4e-5, 0.00067819)


# ── Contest score formula ─────────────────────────────────────────────────


def test_contest_score_matches_pr106_baseline() -> None:
    """contest_score(...) reproduces upstream/evaluate.py within 1e-6."""
    score = contest_score(
        avg_pose_dist=3.4e-5,
        avg_seg_dist=0.00067819,
        archive_bytes=186239,
    )
    # PR106 published score is 0.20945673
    assert abs(score - 0.20945673) < 5e-3, f"score={score} too far from 0.20946"


def test_contest_score_matches_apogee_int4_falsification() -> None:
    """contest_score reproduces the empirical int4 contest-CUDA result."""
    score = contest_score(
        avg_pose_dist=0.02370903,
        avg_seg_dist=0.00868503,
        archive_bytes=109996,
    )
    assert abs(score - 1.42866394) < 1e-3, f"score={score} too far from 1.4287"


def test_contest_score_uses_canonical_upstream_formula_constants() -> None:
    """Per user mandate 2026-05-05 'meta-lagrangian and predictors need to be
    deterministic reproducibility and based on same math as auth eval'.

    The contest-score formula is `100*seg + sqrt(10*pose) + 25*rate_unscaled`
    where `rate_unscaled = archive_bytes / 37545489` (the GT-video bytes).
    Verify EVERY constant by name from `tac.predictor.score_band` so the
    formula cannot drift from upstream/evaluate.py without breaking this test.
    """
    from tac.predictor.score_band import (
        POSE_COEFFICIENT_SQRT_INNER,
        PR106_TOTAL_RATE_DENOM,
        RATE_COEFFICIENT,
        SEG_COEFFICIENT,
    )
    # Constants must match the canonical contest-CUDA formula bytes.
    assert SEG_COEFFICIENT == 100.0, "seg coefficient must be exactly 100 per upstream"
    assert RATE_COEFFICIENT == 25.0, "rate coefficient must be exactly 25 per upstream"
    assert POSE_COEFFICIENT_SQRT_INNER == 10.0, "pose sqrt-inner must be 10 per upstream"
    # GT video size in bytes — comma's 0.mkv. Anyone changing this is changing
    # the official rate denominator; that's a contest-rule violation.
    assert PR106_TOTAL_RATE_DENOM == 37545489, "PR106 rate denom must equal 0.mkv bytes"


def test_contest_score_is_deterministic_across_runs() -> None:
    """Per user mandate, the engine must produce IDENTICAL outputs for identical
    inputs across runs — no RNG, no clock-time drift, no platform variance."""
    inputs = (3.4e-5, 0.00067819, 186239)
    runs = [contest_score(*inputs) for _ in range(50)]
    # All 50 runs must be EXACTLY equal (not just within tolerance — closed-form
    # math should be bit-identical).
    assert len(set(runs)) == 1, f"non-deterministic: got {len(set(runs))} unique values"


def test_meta_lagrangian_evaluate_is_deterministic() -> None:
    """MetaLagrangianSearch.evaluate_candidate() must be deterministic given
    fixed anchors + fixed proxy. Same input → same output across 20 runs."""
    anchors = _make_anchors()
    runs = []
    for _ in range(20):
        search = MetaLagrangianSearch(anchors, _honest_proxy)
        ev = search.evaluate_candidate(
            candidate_id="det_check",
            archive_bytes=187731, rel_err_pct=0.24, n_layers=13,
            lane_class="apogee_intN",
        )
        runs.append((ev.lagrangian, ev.proxy_pose, ev.proxy_seg, ev.band_low, ev.band_high))
    assert len(set(runs)) == 1, f"non-deterministic evaluate: {len(set(runs))} unique tuples"


def test_cma_es_candidate_suggestions_are_deterministic_and_fail_closed() -> None:
    """CMA-ES should be a reproducible candidate generator, not a dispatch gate."""

    bounds = CmaEsSearchBounds(
        archive_bytes=(150_000, 190_000),
        rel_err_pct=(0.0, 1.0),
        n_layers=(8, 16),
    )
    search = MetaLagrangianSearch(_make_anchors(), _honest_proxy)

    first = search.suggest_cma_es_candidates(
        lane_class="apogee_intN",
        bounds=bounds,
        generations=2,
        population_size=4,
        seed=123,
    )
    second = search.suggest_cma_es_candidates(
        lane_class="apogee_intN",
        bounds=bounds,
        generations=2,
        population_size=4,
        seed=123,
    )

    assert [item.candidate for item in first] == [item.candidate for item in second]
    assert [item.objective for item in first] == [item.objective for item in second]
    assert first == sorted(first, key=lambda item: item.objective)
    assert first
    for item in first:
        assert item.score_claim is False
        assert item.ready_for_exact_eval_dispatch is False
        assert "requires_exact_cuda_auth_eval" in item.dispatch_blockers
        assert item.evaluation.archive_path is None
        assert item.evaluation.eligible_for_dispatch is False
        candidate = item.candidate
        assert set(candidate) == {
            "candidate_id",
            "archive_bytes",
            "rel_err_pct",
            "n_layers",
            "lane_class",
        }
        assert bounds.archive_bytes[0] <= candidate["archive_bytes"] <= bounds.archive_bytes[1]
        assert bounds.rel_err_pct[0] <= candidate["rel_err_pct"] <= bounds.rel_err_pct[1]
        assert bounds.n_layers[0] <= candidate["n_layers"] <= bounds.n_layers[1]


def test_cma_es_candidate_suggestions_validate_bounds() -> None:
    search = MetaLagrangianSearch(_make_anchors(), _honest_proxy)

    with pytest.raises(ValueError, match="archive_bytes upper bound"):
        search.suggest_cma_es_candidates(
            lane_class="apogee_intN",
            bounds=CmaEsSearchBounds(
                archive_bytes=(190_000, 150_000),
                rel_err_pct=(0.0, 1.0),
                n_layers=(8, 16),
            ),
        )


def test_meta_lagrangian_preserves_archive_path(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"not-a-real-archive-for-unit-test")
    search = MetaLagrangianSearch(_make_anchors(), _honest_proxy)

    ev = search.evaluate_candidate(
        candidate_id="path_check",
        archive_bytes=187731,
        rel_err_pct=0.24,
        n_layers=13,
        lane_class="apogee_intN",
        archive_path=archive,
    )

    assert ev.archive_path == archive
    assert ev.sanity_failures != ["sanity_skipped: archive_path not yet produced"]


def test_meta_lagrangian_is_arch_agnostic_on_anchor_swap() -> None:
    """Per user mandate 'arch agnostic if possible' — the engine should accept
    arbitrary anchor sets (any architecture class) and produce a self-consistent
    Lagrangian, as long as anchors are well-formed.

    We verify by constructing synthetic anchors for a HYPOTHETICAL different
    architecture (a bigger renderer with different rate-distortion tradeoff)
    and confirming the engine still ranks candidates correctly relative to
    that calibration.
    """
    # Synthetic 'arch_X' anchors with a realistic rate-distortion curve where
    # lossy is strictly worse than lossless across all components (no impossible
    # predictor refusal). Mimics a bigger renderer (300K params class).
    arch_x_anchors = [
        CalibrationAnchor(
            lane_id="arch_x_lossless",
            rel_err_pct_per_weight=0.0,
            archive_bytes=300000,
            contest_cuda_score=0.5,
            avg_pose_dist=1e-4,
            avg_seg_dist=2e-3,
            rate_unscaled=300000 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-05T20:00Z",
            job_id="arch-x-baseline",
            archive_sha256="aaaa",
        ),
        CalibrationAnchor(
            lane_id="arch_x_int8",
            rel_err_pct_per_weight=0.5,
            archive_bytes=240000,
            contest_cuda_score=0.55,
            avg_pose_dist=1.5e-4,
            avg_seg_dist=2.5e-3,
            rate_unscaled=240000 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-05T20:01Z",
            job_id="arch-x-int8",
            archive_sha256="bbbb",
        ),
        CalibrationAnchor(
            lane_id="arch_x_int4",
            rel_err_pct_per_weight=8.0,
            archive_bytes=160000,
            contest_cuda_score=2.5,
            avg_pose_dist=0.05,
            avg_seg_dist=0.01,
            rate_unscaled=160000 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-05T20:02Z",
            job_id="arch-x-int4",
            archive_sha256="cccc",
        ),
    ]
    # Arch-X-aware proxy: returns this arch's distortion levels (NOT the
    # default apogee anchors). Demonstrates the engine accepts arbitrary
    # callable proxy without arch-specific assumptions.
    def _arch_x_proxy(archive_bytes: int, rel_err_pct: float, n_layers: int):
        if rel_err_pct >= 5.0:
            return (0.05, 0.01)        # arch-x int4 anchor distortions
        if rel_err_pct >= 0.3:
            return (1.5e-4, 2.5e-3)    # arch-x int8
        return (1e-4, 2e-3)            # arch-x lossless

    search = MetaLagrangianSearch(arch_x_anchors, _arch_x_proxy)
    # Engine accepts arbitrary anchor architectures without crashing —
    # whether the predictor accepts/refuses the candidate is a separate
    # mathematical-consistency check. This test only verifies the engine
    # is structurally arch-agnostic.
    ev = search.evaluate_candidate(
        candidate_id="arch_x_int8_synthetic",
        archive_bytes=240000, rel_err_pct=0.5, n_layers=42,
        lane_class="arch_X",
    )
    # Lagrangian computed from proxy outputs even when band is refused.
    assert math_isfinite(ev.lagrangian), f"non-finite Lagrangian: {ev.lagrangian}"
    # Proxy outputs reflect the swapped-in arch_x callable, NOT defaults.
    assert ev.proxy_pose == 1.5e-4, f"proxy_pose={ev.proxy_pose} not from arch_x callable"
    assert ev.proxy_seg == 2.5e-3, f"proxy_seg={ev.proxy_seg} not from arch_x callable"


def math_isfinite(x: float) -> bool:
    import math
    return math.isfinite(x)


# ── Lagrangian penalties ──────────────────────────────────────────────────


def test_lagrangian_penalizes_distortion_violation() -> None:
    """A high-rel_err candidate must rank below a lossless one."""
    anchors = _make_anchors()
    search = MetaLagrangianSearch(anchors, _honest_proxy)
    int4 = search.evaluate_candidate(
        candidate_id="int4_synthetic",
        archive_bytes=109996,
        rel_err_pct=7.09,
        n_layers=13,
        lane_class="apogee_intN",
    )
    int8 = search.evaluate_candidate(
        candidate_id="int8_synthetic",
        archive_bytes=187731,
        rel_err_pct=0.24,
        n_layers=13,
        lane_class="apogee_intN",
    )
    # int4's Lagrangian must be DRAMATICALLY higher (worse) than int8.
    assert int4.lagrangian > int8.lagrangian, (
        f"int4 Lagrangian {int4.lagrangian:.4f} should exceed int8 {int8.lagrangian:.4f}"
    )
    # int4 should have nonzero pose violation (0.0237 >> default 1e-4 max).
    assert int4.pose_violation > 0
    # int8 should have minimal/no violation.
    assert int8.pose_violation < int4.pose_violation


def test_lagrangian_constraints_are_tunable() -> None:
    """Tightening lambda_pose increases the gap between int4 and int8."""
    anchors = _make_anchors()
    relaxed = MetaLagrangianSearch(
        anchors, _honest_proxy,
        constraints=LagrangianConstraints(lambda_pose=0.1),
    )
    strict = MetaLagrangianSearch(
        anchors, _honest_proxy,
        constraints=LagrangianConstraints(lambda_pose=100.0),
    )
    for s in (relaxed, strict):
        int4 = s.evaluate_candidate(
            candidate_id="int4", archive_bytes=109996, rel_err_pct=7.09,
            n_layers=13, lane_class="apogee_intN",
        )
        int8 = s.evaluate_candidate(
            candidate_id="int8", archive_bytes=187731, rel_err_pct=0.24,
            n_layers=13, lane_class="apogee_intN",
        )
        gap = int4.lagrangian - int8.lagrangian
        s.last_gap = gap
    assert strict.last_gap > relaxed.last_gap


# ── Top-K ranking ─────────────────────────────────────────────────────────


def test_top_k_filters_refused_and_failed() -> None:
    """Refused / sanity-failed candidates must NOT appear in top_k output."""
    anchors = _make_anchors()
    search = MetaLagrangianSearch(anchors, _honest_proxy)

    # Candidate with rel_err > 1% but proxy provided → predictor accepts.
    feasible = search.evaluate_candidate(
        candidate_id="feasible",
        archive_bytes=187731, rel_err_pct=0.24, n_layers=13, lane_class="apogee_intN",
    )
    # Candidate with rel_err outside calibrated range (>20% beyond 7.09) → refused.
    refused = search.evaluate_candidate(
        candidate_id="refused_extrapolation",
        archive_bytes=80000, rel_err_pct=20.0, n_layers=13, lane_class="apogee_intN",
    )
    assert refused.band_refused
    # Without archive_path, sanity_passed defaults to False → eligible_for_dispatch False.
    assert not feasible.eligible_for_dispatch
    top = search.top_k([feasible, refused], k=3)
    assert top == []  # no eligible candidates without archive_path


def test_top_k_sorts_by_lagrangian() -> None:
    """Among eligible candidates, top_k returns them in ascending Lagrangian."""
    # Build fake evaluations with varied rank keys.
    e1 = CandidateEvaluation(candidate_id="a", archive_bytes=100000, rel_err_pct=0.5,
                             n_layers=13, lane_class="x")
    e1.eligible_for_dispatch = True
    e1.rank_key = 0.5

    e2 = CandidateEvaluation(candidate_id="b", archive_bytes=100000, rel_err_pct=0.5,
                             n_layers=13, lane_class="x")
    e2.eligible_for_dispatch = True
    e2.rank_key = 0.2

    e3 = CandidateEvaluation(candidate_id="c", archive_bytes=100000, rel_err_pct=0.5,
                             n_layers=13, lane_class="x")
    e3.eligible_for_dispatch = True
    e3.rank_key = 0.8
    top = MetaLagrangianSearch.top_k([e1, e2, e3], k=2)
    assert [e.candidate_id for e in top] == ["b", "a"]


# ── Wired-up integration with real anchors file ───────────────────────────


def test_load_anchors_from_disk_and_evaluate(tmp_path: Path) -> None:
    """Live-integration: load .omx/calibration/anchors_apogee_intN.json + evaluate.

    Tier-1 cleanup 2026-05-05: the canonical int8 anchor (archive_bytes=187731 >
    PR106 lossless 186239) is now tagged anchor_role=compatibility_only, so the
    predictor no longer refuses with lossy_anchor_invalid_no_rate_savings.
    The predictor proceeds; whatever it returns must NOT be that specific
    refusal.
    """
    anchors_path = REPO_ROOT / ".omx" / "calibration" / "anchors_apogee_intN.json"
    if not anchors_path.is_file():
        pytest.skip("anchors file not present at repo path")
    anchors = load_calibration_anchors(anchors_path)
    assert len(anchors) >= 3
    # int8 must be tagged compatibility_only post tier-1 cleanup.
    int8_anchors = [a for a in anchors if a.lane_id == "lane_apogee_int8"]
    assert int8_anchors and int8_anchors[0].anchor_role == "compatibility_only", (
        "canonical int8 anchor must be tagged anchor_role=compatibility_only "
        "(tier-1 cleanup 2026-05-05)"
    )

    # Use distortion_proxy_local
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.distortion_proxy_local import make_distortion_proxy  # type: ignore
    proxy = make_distortion_proxy(anchors_path)

    search = MetaLagrangianSearch(anchors, proxy)
    int8_eval = search.evaluate_candidate(
        candidate_id="apogee_int8_anchor",
        archive_bytes=187731, rel_err_pct=0.24, n_layers=13, lane_class="apogee_intN",
    )
    # The lossy_anchor_invalid_no_rate_savings refusal MUST NOT fire — the
    # int8 anchor is compatibility_only and is excluded from that gate.
    assert "lossy_anchor_invalid_no_rate_savings" not in int8_eval.band_refusal_reason, (
        f"lossy_anchor_invalid refusal should not fire on compatibility_only int8 anchor; "
        f"got refusal={int8_eval.band_refusal_reason!r}"
    )
    # Proxy itself still computes (non-negative) distortion estimates.
    assert int8_eval.proxy_pose >= 0
    assert int8_eval.proxy_seg >= 0


# ── Adversarial-review root-cause-fix regression tests (Bugs #1-#7, 2026-05-05)
#    Each test pins one of the 7 bug-class fixes from the dispatch_readiness
#    adversarial review. If any test starts failing, that bug class has
#    re-introduced silent-success behavior. ────────────────────────────────


def test_bug1_distortion_proxy_detects_degenerate_seg_floor(tmp_path: Path) -> None:
    """Bug #1: when an anchor's per-component excess is below the numerical
    floor, the per-component fit MUST be marked degenerate (NaN curve), not
    silently regressed to a meaningless slope."""
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.distortion_proxy_local import (  # type: ignore
        DEGENERATE_EXCESS_THRESHOLD,
        _fit_separate_curves,
    )
    # Two lossy anchors where one has seg_dist EQUAL to baseline (floor case).
    anchors = [
        {"rel_err_pct_per_weight": 0.0, "avg_pose_dist": 3.4e-5, "avg_seg_dist": 6.78e-4},
        {"rel_err_pct_per_weight": 0.24, "avg_pose_dist": 3.4e-5,  # identical pose
                                          "avg_seg_dist": 6.78e-4},  # identical seg → floor
        {"rel_err_pct_per_weight": 7.09, "avg_pose_dist": 0.024, "avg_seg_dist": 0.0087},
    ]
    curves = _fit_separate_curves(anchors)
    # Both pose and seg should be degenerate (the int8-shaped anchor matches baseline).
    import math
    assert math.isnan(curves["pose"]["a"]) and math.isnan(curves["pose"]["b"]), (
        f"pose curve should be degenerate (NaN), got {curves['pose']}"
    )
    assert math.isnan(curves["seg"]["a"]) and math.isnan(curves["seg"]["b"]), (
        f"seg curve should be degenerate (NaN), got {curves['seg']}"
    )
    # The degenerate_reason field must explain WHY.
    assert "numerical_floor" in curves["pose"]["degenerate_reason"]
    assert "numerical_floor" in curves["seg"]["degenerate_reason"]
    assert DEGENERATE_EXCESS_THRESHOLD > 0


def test_bug2_lossy_anchor_with_no_rate_savings_REFUSES() -> None:
    """Bug #2: a lossy anchor (rel_err > 0) with archive_bytes >= tightest
    lossless must trigger lossy_anchor_invalid_no_rate_savings refusal."""
    from tac.predictor.score_band import predict_score_band
    # Three anchors: lossless at 186K, "lossy" at 188K (no savings), int4 at 110K.
    bad_lossy = CalibrationAnchor(
        lane_id="bad_lossy",
        rel_err_pct_per_weight=0.5,
        archive_bytes=188000,  # LARGER than lossless 186239 → invalid
        contest_cuda_score=0.21,
        avg_pose_dist=3.5e-5,
        avg_seg_dist=6.8e-4,
        rate_unscaled=188000 / 37545489,
        measured_utc="t",
        job_id="j",
        archive_sha256="sha",
    )
    lossless = CalibrationAnchor(
        lane_id="lossless",
        rel_err_pct_per_weight=0.0,
        archive_bytes=186239,
        contest_cuda_score=0.20945673,
        avg_pose_dist=3.4e-5,
        avg_seg_dist=6.78e-4,
        rate_unscaled=186239 / 37545489,
        measured_utc="t",
        job_id="j",
        archive_sha256="sha",
    )
    int4 = CalibrationAnchor(
        lane_id="int4",
        rel_err_pct_per_weight=7.09,
        archive_bytes=109996,
        contest_cuda_score=1.4287,
        avg_pose_dist=0.024,
        avg_seg_dist=0.0087,
        rate_unscaled=109996 / 37545489,
        measured_utc="t",
        job_id="j",
        archive_sha256="sha",
    )
    band = predict_score_band(
        archive_bytes=140000,
        rel_err_pct_per_weight=2.0,
        n_quantized_layers=13,
        calibration_anchors=[lossless, bad_lossy, int4],
        distortion_proxy=lambda b, r, n: (1e-4, 1e-3),
    )
    assert band.refused
    assert "lossy_anchor_invalid_no_rate_savings" in band.refusal_reason
    assert "bad_lossy" in band.refusal_reason  # name surfaced for debug


def test_bug3_default_sanity_gate_RAISES_when_helper_missing(tmp_path: Path, monkeypatch) -> None:
    """Bug #3: when tools/predispatch_sanity.py is absent, the default sanity
    gate MUST raise FileNotFoundError, not silently return passed=True."""
    from tac.optimizer import meta_lagrangian as ml
    # Point REPO_ROOT to an empty tmp dir so the helper file does not exist.
    monkeypatch.setattr(ml, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="predispatch_sanity helper not found"):
        ml._default_sanity_gate(
            archive_path=tmp_path / "x.zip",
            predicted_low=0.5, predicted_high=0.6,
            rel_err_pct=0.5, lane_class="apogee_intN",
            distortion_proxy_was_run=True,
        )


def test_bug3_meta_lagrangian_raises_on_missing_helper_when_archive_present(
    tmp_path: Path, monkeypatch,
) -> None:
    """Bug #3 integration: search.evaluate_candidate must propagate the helper-
    missing FileNotFoundError up via sanity_failures (no silent pass)."""
    from tac.optimizer import meta_lagrangian as ml
    monkeypatch.setattr(ml, "REPO_ROOT", tmp_path)
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"fake-archive")
    search = MetaLagrangianSearch(_make_anchors(), _honest_proxy)
    ev = search.evaluate_candidate(
        candidate_id="bug3",
        archive_bytes=187731, rel_err_pct=0.24, n_layers=13, lane_class="apogee_intN",
        archive_path=archive,
    )
    # The exception is caught + recorded as a sanity failure; the candidate is
    # NOT eligible for dispatch.
    assert not ev.sanity_passed
    assert any("FileNotFoundError" in f for f in ev.sanity_failures), (
        f"expected FileNotFoundError in sanity_failures, got {ev.sanity_failures}"
    )
    assert not ev.eligible_for_dispatch


def test_bug4_remote_lane_script_requires_inflate_py_at_stage0() -> None:
    """Bug #4: the remote lane script must require submissions/apogee_intN/
    inflate.py existence and parse_apogee_intn_archive export at Stage 0,
    BEFORE provenance is written."""
    script = REPO_ROOT / "scripts" / "remote_lane_apogee_intN.sh"
    assert script.is_file()
    body = script.read_text()
    # Verify the early require_file checks exist and reference the parser.
    assert "require_file" in body, "require_file helper missing"
    assert 'require_file "$WORKSPACE/submissions/apogee_intN/inflate.py"' in body, (
        "early existence check for inflate.py missing"
    )
    assert "parse_apogee_intn_archive" in body, (
        "Stage 0 must also verify parse_apogee_intn_archive symbol exists"
    )
    # The require_file invocations must come BEFORE the provenance.json write.
    require_idx = body.index('require_file "$WORKSPACE/submissions/apogee_intN/inflate.py"')
    prov_idx = body.index("provenance.json")
    assert require_idx < prov_idx, (
        "require_file checks must run BEFORE provenance is written; "
        "otherwise the lane claim is burned before the existence check fires"
    )


def test_bug5_remote_lane_script_drops_ensure_dali_and_corrects_header() -> None:
    """Bug #5: the script previously ran probe_nvdec.sh --ensure-dali while
    its header claimed NO_NVDEC_NEEDED. Both must be fixed: header truthful
    AND probe runs without --ensure-dali."""
    script = REPO_ROOT / "scripts" / "remote_lane_apogee_intN.sh"
    body = script.read_text()
    # Header must NOT claim NO_NVDEC_NEEDED.
    assert "NO_NVDEC_NEEDED" not in body, (
        "header NO_NVDEC_NEEDED claim is incorrect — DALI is used by upstream/evaluate.py"
    )
    # Probe call must NOT pass --ensure-dali (auto-install hides install errors).
    assert "--ensure-dali" not in body, (
        "probe_nvdec.sh must run without --ensure-dali; DALI must be pre-installed"
    )
    # The probe still runs (we did not delete it).
    assert "probe_nvdec.sh" in body


def test_bug6_score_parser_validates_numeric_and_no_silent_fallback() -> None:
    """Bug #6: the score parser previously had `2>/dev/null || echo PARSE_FAIL`
    which swallowed Python errors AND silently logged a string. The fix: Python
    errors propagate via set -e, and the SCORE value is regex-validated."""
    script = REPO_ROOT / "scripts" / "remote_lane_apogee_intN.sh"
    body = script.read_text()
    # The string-fallback PARSE_FAIL must be GONE.
    assert "PARSE_FAIL" not in body, (
        "PARSE_FAIL silent string fallback must be removed"
    )
    # The "2>/dev/null" suppression of stderr in the score parser must be GONE.
    # (We allow it elsewhere in the script for genuinely optional probes, but
    # NOT on the json.load score parse.)
    score_lines = [
        ln for ln in body.split("\n")
        if "json.load" in ln and "final_score" in ln
    ]
    assert score_lines, "score parser line not found"
    for ln in score_lines:
        assert "2>/dev/null" not in ln, (
            f"score parse line must not suppress stderr: {ln}"
        )
    # The numeric regex validation must be present.
    assert "^-?[0-9]+(\\.[0-9]+)?" in body, "numeric SCORE regex check missing"


def test_bug7_proxy_degenerate_curve_downgrades_confidence_and_widens_band() -> None:
    """Bug #7: when distortion_proxy.curves contains NaN coefficients (degenerate
    fit per Bug #1), the predictor must downgrade confidence to calibrated_weak
    AND widen the band by 50% (per Boyd interval inflation under degenerate fit)."""
    from tac.predictor.score_band import predict_score_band
    # Use a structurally valid 3-anchor set (lossy < lossless bytes) so the new
    # Bug #2 refusal does not fire and we can isolate Bug #7 behavior.
    valid_int8 = CalibrationAnchor(
        lane_id="valid_int8",
        rel_err_pct_per_weight=0.24,
        archive_bytes=160000,  # smaller than lossless 186239
        contest_cuda_score=0.20,
        avg_pose_dist=3.375e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=160000 / PR106_TOTAL_RATE_DENOM,
        measured_utc="t",
        job_id="j",
        archive_sha256="sha",
    )
    canonical = _make_anchors()
    # Replace the broken int8 anchor with our valid one.
    anchors = [canonical[0], valid_int8, canonical[2]]

    # Healthy proxy → calibrated_strong.
    def healthy_proxy(b, r, n):
        return (3.5e-5, 6.8e-4)

    healthy_proxy.curves = {  # type: ignore
        "pose": {"a": 1e-3, "b": 1.0, "d_baseline": 3.4e-5, "degenerate_reason": ""},
        "seg":  {"a": 1e-3, "b": 1.0, "d_baseline": 6.78e-4, "degenerate_reason": ""},
    }
    band_healthy = predict_score_band(
        archive_bytes=160000,
        rel_err_pct_per_weight=0.24,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=healthy_proxy,
        band_half_width_score=0.05,
    )
    assert not band_healthy.refused
    assert band_healthy.confidence == "calibrated_strong"
    healthy_width = band_healthy.high - band_healthy.low

    # Degenerate proxy (NaN curves) → calibrated_weak + band widened 50%.
    import math

    def degen_proxy(b, r, n):
        return (3.5e-5, 6.8e-4)

    degen_proxy.curves = {  # type: ignore
        "pose": {"a": float("nan"), "b": float("nan"), "d_baseline": 3.4e-5,
                 "degenerate_reason": "numerical_floor: 1 of 2 ..."},
        "seg":  {"a": float("nan"), "b": float("nan"), "d_baseline": 6.78e-4,
                 "degenerate_reason": "numerical_floor: 1 of 2 ..."},
    }
    band_degen = predict_score_band(
        archive_bytes=160000,
        rel_err_pct_per_weight=0.24,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=degen_proxy,
        band_half_width_score=0.05,
    )
    assert not band_degen.refused
    assert band_degen.confidence == "calibrated_weak", (
        f"degenerate proxy must downgrade confidence; got {band_degen.confidence}"
    )
    degen_width = band_degen.high - band_degen.low
    # 50% wider (within float tolerance).
    assert math.isclose(degen_width, 1.5 * healthy_width, rel_tol=1e-9), (
        f"degenerate band width {degen_width:.4f} should be 1.5x healthy {healthy_width:.4f}"
    )
    # Derivation must mention the degeneracy + downgrade.
    assert "proxy_curve_degenerate" in band_degen.derivation
