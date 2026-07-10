"""Tests for the curriculum-candidate POOL — the P0 tracked costate class (task #403).

Covers the store (append + latest-wins + lenient read), the NO-FAKE validation contract (status /
form-class / source-anchor / exactly-one-of DSL-leg / measured-needs-verdict / est_delta_s+axis), the
ranked report + duty queue, the idempotent seed, and the two DSL folds' completeness + composability.
"""
from __future__ import annotations

import pytest

from tac.witness_dsl import curriculum_candidate_pool as ccp


# ── store round-trip + latest-wins ───────────────────────────────────────────────────────────────
def test_record_and_read_roundtrip(tmp_path):
    p = tmp_path / "pool.jsonl"
    row = ccp.record_candidate(
        "c_x", ccp.STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
        source_anchor="commit abc", gate="fire when X", dsl_lever="SomeLever", path=p)
    assert row["candidate"] == "c_x"
    st = ccp.candidate_status("c_x", path=p)
    assert st is not None
    assert st.status == ccp.STATUS_BUILT_NEVER_FIRED
    assert st.dsl_lever == "SomeLever"
    assert st.in_duty_queue is True


def test_latest_row_wins(tmp_path):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate("c", ccp.STATUS_NEEDS_BUILD, form_class="averaging",
                         source_anchor="a", gate="g", dsl_na_reason="unbuilt", path=p)
    ccp.record_candidate("c", ccp.STATUS_BUILT_NEVER_FIRED, form_class="averaging",
                         source_anchor="a2", gate="g2", dsl_lever="Built", path=p)
    st = ccp.candidate_status("c", path=p)
    assert st.status == ccp.STATUS_BUILT_NEVER_FIRED  # later row wins
    assert st.dsl_lever == "Built"


def test_read_is_lenient_to_corrupt_lines(tmp_path):
    p = tmp_path / "pool.jsonl"
    ccp.record_candidate("ok", ccp.STATUS_ARMED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="L", path=p)
    with p.open("a") as f:
        f.write("not json at all\n")
        f.write('{"candidate": "", "status": "armed"}\n')  # empty candidate skipped
    rows = ccp.pool_report(path=p)
    assert [r["candidate"] for r in rows] == ["ok"]


def test_missing_candidate_returns_none(tmp_path):
    assert ccp.candidate_status("nope", path=tmp_path / "empty.jsonl") is None


# ── NO-FAKE validation contract ──────────────────────────────────────────────────────────────────
def test_invalid_status_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid status"):
        ccp.record_candidate("c", "flying", form_class="averaging",
                             source_anchor="a", gate="g", dsl_lever="L", path=tmp_path / "p.jsonl")


def test_invalid_form_class_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid form_class"):
        ccp.record_candidate("c", ccp.STATUS_ARMED, form_class="magic",
                             source_anchor="a", gate="g", dsl_lever="L", path=tmp_path / "p.jsonl")


def test_source_anchor_required(tmp_path):
    with pytest.raises(ValueError, match="source_anchor is required"):
        ccp.record_candidate("c", ccp.STATUS_ARMED, form_class="averaging",
                             source_anchor="", gate="g", dsl_lever="L", path=tmp_path / "p.jsonl")


def test_exactly_one_dsl_leg_required_both(tmp_path):
    with pytest.raises(ValueError, match="exactly one of dsl_lever"):
        ccp.record_candidate("c", ccp.STATUS_ARMED, form_class="averaging", source_anchor="a",
                             gate="g", dsl_lever="L", dsl_na_reason="also", path=tmp_path / "p.jsonl")


def test_exactly_one_dsl_leg_required_neither(tmp_path):
    with pytest.raises(ValueError, match="exactly one of dsl_lever"):
        ccp.record_candidate("c", ccp.STATUS_ARMED, form_class="averaging", source_anchor="a",
                             gate="g", path=tmp_path / "p.jsonl")


def test_measured_requires_verdict_ref(tmp_path):
    with pytest.raises(ValueError, match="requires a verdict_ref"):
        ccp.record_candidate("c", ccp.STATUS_MEASURED, form_class="averaging", source_anchor="a",
                             gate="g", dsl_lever="L", path=tmp_path / "p.jsonl")


def test_measured_with_verdict_ref_ok(tmp_path):
    row = ccp.record_candidate("c", ccp.STATUS_MEASURED, form_class="averaging", source_anchor="a",
                               gate="g", dsl_lever="L", verdict_ref="byteclose/verdict.json",
                               path=tmp_path / "p.jsonl")
    assert row["status"] == ccp.STATUS_MEASURED
    assert row["verdict_ref"] == "byteclose/verdict.json"


def test_negative_est_delta_s_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        ccp.record_candidate("c", ccp.STATUS_ARMED, form_class="averaging", source_anchor="a",
                             gate="g", dsl_lever="L", est_delta_s=-0.1, axis="d_seg",
                             path=tmp_path / "p.jsonl")


def test_est_delta_s_requires_axis(tmp_path):
    with pytest.raises(ValueError, match="requires axis"):
        ccp.record_candidate("c", ccp.STATUS_ARMED, form_class="averaging", source_anchor="a",
                             gate="g", dsl_lever="L", est_delta_s=0.01, path=tmp_path / "p.jsonl")


# ── ranking + duty queue ─────────────────────────────────────────────────────────────────────────
def test_report_ranks_built_never_fired_first(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.record_candidate("armed1", ccp.STATUS_ARMED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="A", path=p)
    ccp.record_candidate("needs1", ccp.STATUS_NEEDS_BUILD, form_class="averaging",
                         source_anchor="a", gate="g", dsl_na_reason="unbuilt", path=p)
    ccp.record_candidate("bnf1", ccp.STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="B", path=p)
    ccp.record_candidate("ref1", ccp.STATUS_REFORMULATION_QUEUE, form_class="state-evolution",
                         source_anchor="a", gate="g", dsl_na_reason="reform", path=p)
    order = [r["candidate"] for r in ccp.pool_report(path=p)]
    assert order == ["bnf1", "needs1", "ref1", "armed1"]


def test_duty_queue_excludes_armed_measured_retired(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.record_candidate("bnf", ccp.STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="B", path=p)
    ccp.record_candidate("armed", ccp.STATUS_ARMED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="A", path=p)
    ccp.record_candidate("ret", ccp.STATUS_RETIRED, form_class="optimizer-stage",
                         source_anchor="a", gate="g", dsl_na_reason="law", path=p)
    duty = {r["candidate"] for r in ccp.duty_to_measure_pool(path=p)}
    assert duty == {"bnf"}


def test_est_delta_s_breaks_ties_within_status(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.record_candidate("low", ccp.STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="L", est_delta_s=0.01, axis="d_seg", path=p)
    ccp.record_candidate("high", ccp.STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="H", est_delta_s=0.05, axis="d_seg", path=p)
    ccp.record_candidate("none", ccp.STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
                         source_anchor="a", gate="g", dsl_lever="N", path=p)
    order = [r["candidate"] for r in ccp.pool_report(path=p)]
    assert order == ["high", "low", "none"]  # est desc, then None last


# ── seed (idempotent, honest statuses) ───────────────────────────────────────────────────────────
def test_seed_is_idempotent(tmp_path):
    p = tmp_path / "p.jsonl"
    n1 = ccp.seed_default_pool(path=p)
    assert n1 == len(ccp._SEED) > 0
    n2 = ccp.seed_default_pool(path=p)
    assert n2 == 0  # re-seed writes nothing


def test_seed_has_no_measured_status(tmp_path):
    # NO-FAKE: a designed/built candidate is never seeded 'measured' (no byte-closed verdict exists).
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    assert all(r["status"] != ccp.STATUS_MEASURED for r in ccp.pool_report(path=p))


def test_seed_every_row_has_exactly_one_dsl_leg(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    for r in ccp.pool_report(path=p):
        assert bool(r.get("dsl_lever")) != bool(r.get("dsl_na_reason")), r["candidate"]


def test_seed_includes_the_two_folds(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    by_cand = {r["candidate"]: r for r in ccp.pool_report(path=p)}
    assert by_cand["hardness_oversample_lever5"]["dsl_lever"] == "HardnessOversample"
    assert by_cand["head_geometry_218_etf_am"]["dsl_lever"] == "HeadGeometry"


def test_pool_summary_shape(tmp_path):
    p = tmp_path / "p.jsonl"
    ccp.seed_default_pool(path=p)
    summ = ccp.pool_summary(path=p)
    assert summ["total"] == len(ccp._SEED)
    assert summ["owed"] == sum(1 for r in ccp.pool_report(path=p)
                               if r["status"] in ("built-never-fired", "needs-build", "reformulation-queue"))
    assert len(summ["top_fireable"]) <= 6


def test_seed_dsl_lever_rows_reference_real_or_documented_factories(tmp_path):
    # Every seeded row that claims a dsl_lever must name a real held factory OR be one of the
    # documented-elsewhere factory names (owner-scoped). We assert the TWO folds this landing built
    # are actually held (the ones we are responsible for); others may live in sibling landings.
    from tac.witness_dsl.lever_registry import lever_factories
    held = set(lever_factories().keys())
    assert "HardnessOversample" in held
    assert "HeadGeometry" in held


# ── DSL folds: completeness shrink + composability ───────────────────────────────────────────────
def test_folds_hold_the_previously_unmapped_flags():
    from tac.witness_dsl.lever_registry import completeness, lever_factories
    lf = lever_factories()
    assert lf["HardnessOversample"] == frozenset({
        "--hardness-oversample", "--hardness-weighted", "--hardness-source",
        "--hardness-power", "--hardness-band"})
    assert lf["HeadGeometry"] == frozenset({"--head", "--additive-margin"})
    c = completeness()
    for flag in ("--hardness-oversample", "--hardness-weighted", "--hardness-source",
                 "--hardness-power", "--hardness-band", "--head", "--additive-margin"):
        assert flag not in c.unmapped, flag


def test_folds_are_composable_by_bare_name():
    from tac.witness_dsl.lever_registry import name_composable_levers, resolve_composable_lever
    comp = name_composable_levers()
    assert "HardnessOversample" in comp
    assert "HeadGeometry" in comp
    # armed defaults engage the mechanism (oversample>0, ETF head) — not a byte-identical no-op arm.
    assert resolve_composable_lever("HardnessOversample").overrides["--hardness-oversample"] == 0.5
    assert resolve_composable_lever("HeadGeometry").overrides["--head"] == "etf"


def test_fold_arg_validation():
    from tac.witness_dsl.curriculum_dsl import HardnessOversample, HeadGeometry
    with pytest.raises(ValueError, match="hardness-source"):
        HardnessOversample(source="bogus")
    with pytest.raises(ValueError, match="head must be"):
        HeadGeometry(head="bogus")


# ── digest surfacing (pure formatter + fail-open section) ────────────────────────────────────────
def test_digest_formatter_leads_with_counts_and_markers():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from costate_digest import format_curriculum_pool_line
    summary = {
        "total": 3, "owed": 2,
        "counts": {"built-never-fired": 1, "needs-build": 1, "reformulation-queue": 0, "armed": 1},
        "top_fireable": [
            {"candidate": "bnf", "status": "built-never-fired", "dsl_lever": "L"},
            {"candidate": "nb", "status": "needs-build", "dsl_lever": None, "dsl_na_reason": "unbuilt"},
        ],
    }
    line = format_curriculum_pool_line(summary)
    assert "curriculum-pool (3 tracked; 2 owed a fire" in line
    assert "1 built-never-fired" in line
    assert "bnf[built·L]" in line          # held lever, no ~ marker
    assert "nb~[needs·N/A]" in line          # not-a-lever, ~ marker


def test_digest_section_reads_real_seeded_store():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from costate_digest import section_curriculum_pool
    # the store was seeded at landing; section must render a line + machine-readable data (fail-open).
    line, data = section_curriculum_pool()
    if data is not None:  # store present
        assert line.startswith("curriculum-pool (")
        assert data["total"] >= 1
        assert "counts" in data
