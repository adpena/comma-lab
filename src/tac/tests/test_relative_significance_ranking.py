"""Tests for the RELATIVE-SIGNIFICANCE fold into the costate duty-to-measure ranking
(tac.witness_dsl.activation_ledger + tools/costate_digest.py).

The recurring bug (`relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS`): a lever's
ΔS was judged against the EYEBALL, anchoring on absolute magnitude, so a small-but-near-goal-significant
lever (#169 horizon-margin, ΔS 0.012-0.024) got orphaned. The fold makes the apparatus compute
`rel_sig = est_delta_s / (s_current − s_target)` — the fraction of the REMAINING descent to sub-0.15 —
and rank the duty-to-measure queue by it, reading s_current from the LIVE pointer (never hardcoded).

All $0 apparatus / score-neutral (read/rank/log only). Pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tac.witness_dsl import activation_ledger as al

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))


@pytest.fixture
def sig(tmp_path):
    return tmp_path / "lever_relative_significance.jsonl"


@pytest.fixture
def led(tmp_path):
    return tmp_path / "lever_activation_ledger.jsonl"


@pytest.fixture
def pointer(tmp_path):
    p = tmp_path / "pointer.json"
    p.write_text(json.dumps({"our_local_frontier_contest_cpu": {"score": 0.19110}}))
    return p


# ─────────────────────────── the metric (the anti-dismissal proof) ───────────────────────────
def test_metric_near_goal_amplification_the_169_case():
    """#169 ΔS 0.018 looks WEAK vs the full score (9.4%) but is HIGH vs remaining descent (43.8%).
    This flip from 'looks weak absolutely' to 'ranks high relatively' is the exact recurring bug."""
    vs_full_score = 0.018 / 0.19110               # the eyeball framing that orphaned it
    rel_sig = al.relative_significance(0.018, 0.19110, 0.15)
    assert vs_full_score < 0.10                    # < 10% of the score -> "polish"
    assert rel_sig > 0.40                          # > 40% of the remaining fight -> significant
    assert rel_sig > 4.0 * vs_full_score           # the near-goal denominator reweights it ~4.6x


def test_metric_rises_as_current_approaches_target():
    """For a FIXED est, rel-sig RISES as s_current -> s_target: a small ΔS becomes MORE significant
    near the goal, not negligible. This is the correction the recurring dismissal violated."""
    far = al.relative_significance(0.018, 0.30, 0.15)
    near = al.relative_significance(0.018, 0.19110, 0.15)
    assert near > far


def test_metric_none_when_uncomputable():
    assert al.relative_significance(None, 0.19110, 0.15) is None      # no estimate
    assert al.relative_significance(0.018, None, 0.15) is None        # no pointer
    assert al.relative_significance(0.018, 0.14, 0.15) is None        # at/below goal -> undefined
    assert al.relative_significance(0.018, 0.15, 0.15) is None        # zero remaining gap


def test_metric_monotone_in_est_for_fixed_gap():
    a = al.relative_significance(0.010, 0.19110, 0.15)
    b = al.relative_significance(0.020, 0.19110, 0.15)
    assert b > a


# ─────────────────────────── store persistence / latest-wins ───────────────────────────
def test_store_roundtrip_and_latest_wins(sig):
    al.record_relative_significance("L1", 0.010, label="ESTIMATED", source_anchor="memo#1",
                                    axis="d_seg", path=sig)
    al.record_relative_significance("L1", 0.025, label="MEASURED", source_anchor="memo#2",
                                    axis="d_seg", path=sig)   # supersedes
    m = al._read_significance(sig)
    assert m["L1"]["est_delta_s"] == 0.025 and m["L1"]["delta_s_label"] == "MEASURED"
    assert m["L1"]["source_anchor"] == "memo#2"


def test_store_skips_corrupt_lines(sig):
    al.record_relative_significance("A", 0.01, label="ESTIMATED", source_anchor="s", axis="rate", path=sig)
    with open(sig, "a") as f:
        f.write("{bad json\n\n")
    al.record_relative_significance("B", 0.02, label="ESTIMATED", source_anchor="s", axis="rate", path=sig)
    assert set(al._read_significance(sig).keys()) == {"A", "B"}


def test_store_unmeasured_row_allows_none_est(sig):
    al.record_relative_significance("owe_estimate", None, label="UNMEASURED",
                                    source_anchor="s", axis="d_seg", path=sig)
    assert al._read_significance(sig)["owe_estimate"]["est_delta_s"] is None


def test_store_validation():
    with pytest.raises(ValueError):
        al.record_relative_significance("", 0.01, label="MEASURED", source_anchor="s", axis="d_seg")
    with pytest.raises(ValueError):  # bad label
        al.record_relative_significance("L", 0.01, label="GUESS", source_anchor="s", axis="d_seg")
    with pytest.raises(ValueError):  # bad axis
        al.record_relative_significance("L", 0.01, label="MEASURED", source_anchor="s", axis="bytes")
    with pytest.raises(ValueError):  # negative ΔS magnitude
        al.record_relative_significance("L", -0.01, label="MEASURED", source_anchor="s", axis="d_seg")
    with pytest.raises(ValueError):  # MEASURED requires a number
        al.record_relative_significance("L", None, label="MEASURED", source_anchor="s", axis="d_seg")
    with pytest.raises(ValueError):  # source required (NO-FAKE)
        al.record_relative_significance("L", 0.01, label="MEASURED", source_anchor="", axis="d_seg")


# ─────────────────────────── pointer read (NOT hardcoded) ───────────────────────────
def test_read_pointer_s_from_json(pointer):
    assert al.read_pointer_s(pointer) == pytest.approx(0.19110)


def test_read_pointer_s_missing_returns_none(tmp_path):
    assert al.read_pointer_s(tmp_path / "nope.json") is None


def test_ranked_reads_pointer_not_hardcoded(sig, led, pointer):
    al.record_relative_significance("horizon_169", 0.018, label="MEASURED", source_anchor="s",
                                    axis="d_seg", path=sig)
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer)
    assert rows[0]["lever"] == "horizon_169"
    assert rows[0]["s_current"] == pytest.approx(0.19110)
    # rel_sig used the pointer's 0.19110, not any literal
    assert rows[0]["rel_sig"] == pytest.approx(0.018 / (0.19110 - 0.15))


def test_ranked_pointer_unavailable_degrades_to_est_order(sig, led, tmp_path):
    al.record_relative_significance("big", 0.05, label="ESTIMATED", source_anchor="s", axis="rate", path=sig)
    al.record_relative_significance("small", 0.01, label="ESTIMATED", source_anchor="s", axis="rate", path=sig)
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig,
                                     pointer_path=tmp_path / "absent.json")
    # no pointer -> rel_sig None for all -> falls back to est_delta_s DESC (biggest ΔS first, still ranked)
    assert [r["lever"] for r in rows] == ["big", "small"]
    assert all(r["rel_sig"] is None for r in rows)


# ─────────────────────────── the ranking (ordering flips vs absolute/alphabetical) ───────────────────────────
def test_ranked_orders_by_relsig_not_name(sig, led, pointer):
    """Alphabetical-first 'aaa' with a SMALL est must sort BELOW 'zzz' with a bigger est — the fold
    removes the alphabetical fallback that let a name outrank value."""
    al.record_relative_significance("aaa_small", 0.005, label="ESTIMATED", source_anchor="s",
                                    axis="d_seg", path=sig)
    al.record_relative_significance("zzz_big", 0.030, label="ESTIMATED", source_anchor="s",
                                    axis="d_seg", path=sig)
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer)
    order = [r["lever"] for r in rows]
    assert order.index("zzz_big") < order.index("aaa_small")


def test_ranked_estimated_levers_outrank_unknowns(sig, led, pointer):
    """A registered owed lever with NO ΔS row (duty-to-estimate) sorts BELOW any lever with an estimate."""
    al.record_relative_significance("has_est", 0.001, label="ESTIMATED", source_anchor="s",
                                    axis="rate", path=sig)
    known = ("has_est", "no_est_lever")   # no_est_lever owed but unestimated
    rows = al.duty_to_measure_ranked(known=known, path=led, sig_path=sig, pointer_path=pointer)
    order = [r["lever"] for r in rows]
    assert order.index("has_est") < order.index("no_est_lever")
    no_est = next(r for r in rows if r["lever"] == "no_est_lever")
    assert no_est["rel_sig"] is None and no_est["in_duty_queue"] is True


def test_ranked_s_target_is_parameterized(sig, led, pointer):
    al.record_relative_significance("L", 0.018, label="MEASURED", source_anchor="s", axis="d_seg", path=sig)
    r15 = al.duty_to_measure_ranked(s_target=0.15, known=(), path=led, sig_path=sig, pointer_path=pointer)
    r16 = al.duty_to_measure_ranked(s_target=0.16, known=(), path=led, sig_path=sig, pointer_path=pointer)
    # smaller remaining gap (target 0.16) -> larger fraction of remaining descent
    assert r16[0]["rel_sig"] > r15[0]["rel_sig"]
    assert r15[0]["s_target"] == 0.15 and r16[0]["s_target"] == 0.16


def test_ranked_unbuilt_finding_included_as_missing_wire(sig, led, pointer):
    """A store finding that is NOT a registered lever is INCLUDED (an orphan is often a missing wire)."""
    al.record_relative_significance("unbuilt_finding_169", 0.02, label="MEASURED", source_anchor="s",
                                    axis="d_seg", path=sig)
    rows = al.duty_to_measure_ranked(known=(), path=led, sig_path=sig, pointer_path=pointer)
    fr = next(r for r in rows if r["lever"] == "unbuilt_finding_169")
    assert fr["registered"] is False and fr["activation_state"] == "not-registered"
    assert fr["rel_sig_pct"] is not None


def test_ranked_rel_sig_dseg_only_for_dseg_axis(sig, led, pointer):
    al.record_relative_significance("seg_lever", 0.018, label="MEASURED", source_anchor="s",
                                    axis="d_seg", path=sig)
    al.record_relative_significance("rate_lever", 0.018, label="MEASURED", source_anchor="s",
                                    axis="rate", path=sig)
    rows = {r["lever"]: r for r in al.duty_to_measure_ranked(known=(), path=led, sig_path=sig,
                                                             pointer_path=pointer)}
    # Δd_seg = ΔS/100 = 0.00018; /target 0.0009 = 0.2
    assert rows["seg_lever"]["rel_sig_dseg"] == pytest.approx((0.018 / 100.0) / al.TARGET_D_SEG)
    assert rows["rate_lever"]["rel_sig_dseg"] is None


def test_ranked_registered_measured_lever_dropped_from_duty(sig, led, pointer):
    """A registered lever already MEASURED (drained from duty_to_measure) and with no store row does
    not appear; but if it has a store row it is still surfaced (value memory), just not owed."""
    al.record_activation("done_lever", al.EVENT_FIRED, run_ref="/r", path=led)
    al.record_activation("done_lever", al.EVENT_MEASURED, run_ref="/r", path=led)
    rows = al.duty_to_measure_ranked(known=("done_lever", "owed_lever"), path=led,
                                     sig_path=sig, pointer_path=pointer)
    levers = {r["lever"] for r in rows}
    assert "done_lever" not in levers   # measured + no store row -> gone
    assert "owed_lever" in levers


# ─────────────────────────── committed seeder (reproducible source; store is gitignored) ───────────────────────────
def test_seeder_populates_and_ranks_169_121_at_top(sig, led, pointer):
    """The committed seeder is the reproducible source of truth (the .omx/state store is gitignored).
    After seeding, #121 and #169 rank at the top by % of remaining descent."""
    import seed_lever_relative_significance as seeder
    n = seeder.seed(path=sig, reset=True)
    assert n == len(seeder.SEED_ROWS)
    rows = al.duty_to_measure_ranked(known=("StepNativeActivation", "seg_down_weight_274"),
                                     path=led, sig_path=sig, pointer_path=pointer)
    top = [r["lever"] for r in rows[:3]]
    assert top[0] == "d_seg_aware_taper_121"      # 73% of remaining descent
    assert top[1] == "horizon_weighted_margin_169"  # 43.8%
    # StepNativeActivation is a registered lever -> the value-join lands it too
    assert "StepNativeActivation" in [r["lever"] for r in rows]
    # seg_down_weight_274 is UNMEASURED -> surfaced as duty-to-estimate (rel_sig None), sorts after ests
    d274 = next(r for r in rows if r["lever"] == "seg_down_weight_274")
    assert d274["rel_sig"] is None


def test_seeder_reset_is_idempotent(sig):
    import seed_lever_relative_significance as seeder
    seeder.seed(path=sig, reset=True)
    seeder.seed(path=sig, reset=True)   # reset -> no accretion
    m = al._read_significance(sig)
    assert len(m) == len(seeder.SEED_ROWS)  # latest-wins + reset keeps exactly the seed set


# ─────────────────────────── digest render (pure formatter — no gitignored-state dependency) ───────────────────────────
def test_format_duty_line_shows_pct_and_markers():
    import costate_digest as cd
    ranked = [
        {"lever": "d_seg_aware_taper_121", "rel_sig_pct": 73.0, "est_delta_s": 0.03,
         "registered": False, "activation_state": "not-registered", "in_duty_queue": False,
         "s_current": 0.19110, "s_target": 0.15},
        {"lever": "StepNativeActivation", "rel_sig_pct": 31.6, "est_delta_s": 0.013,
         "registered": True, "activation_state": "never-fired", "in_duty_queue": True,
         "s_current": 0.19110, "s_target": 0.15},
        {"lever": "seg_down_weight_274", "rel_sig_pct": None, "est_delta_s": None,
         "registered": True, "activation_state": "never-fired", "in_duty_queue": True,
         "s_current": 0.19110, "s_target": 0.15},
    ]
    line = cd.format_duty_to_measure_line(ranked)
    assert "% of remaining descent" in line
    assert "pointer 0.19110→0.15" in line
    assert "d_seg_aware_taper_121~ 73%" in line          # ~ = unbuilt finding
    assert "StepNativeActivation* 31.6%" in line         # * = never-fired registered
    assert "seg_down_weight_274? ?%" in line             # ? = est-owed


def test_format_duty_line_empty():
    import costate_digest as cd
    line = cd.format_duty_to_measure_line([])
    assert "duty-to-measure (0 owed" in line and "pointer unavailable" in line


def test_digest_section_and_build_never_raise():
    """The live section reads canonical paths; it must always return a well-formed line + never crash
    the digest (fail-open), regardless of whether the gitignored store is present."""
    import costate_digest as cd
    line, _data = cd.section_duty_to_measure()
    assert "duty-to-measure" in line
    lines, data = cd.build_digest()
    assert lines and "duty_to_measure" in data


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
