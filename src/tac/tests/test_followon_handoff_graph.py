"""Tests for the HANDOFF GRAPH surface of :mod:`tac.followon_ledger` (ddm_oh1).

Every test here pins a defect that was MEASURED on real corpus text, not an invented edge case.
Four of them pin defects found in this arm's OWN round-1 review, after the code was written and
passing an earlier draft of these tests:

  * multi-line scoping   -- both real-world controls MISS under line scoping (the shipped
                            ``extract_followons`` cannot see either instance for this reason)
  * ``handoff`` heading  -- ambient CUSTODY-section vocabulary in this corpus, not owed work
  * table rows           -- ``| n64 | 61,087 | ... |`` under a "…handoff" heading read as debt
  * untracked paths      -- invisible to a commit-touch channel BY CONSTRUCTION

The discipline the whole file enforces is one-directional: the join may be wrong by returning
UNVERIFIABLE, and may never be wrong by returning ORPHANED. A false ORPHANED manufactures debt out
of a gap in the instrument, which is the failure ``ddm_fo1`` measured in its own first run.
"""

from __future__ import annotations

from datetime import date

import pytest

from tac.followon_ledger import (
    ADVANCED,
    HANDOFF_FRAME_RX,
    LIVE,
    ORPHANED,
    OWED_HEADING_RX,
    STRATUM_FRAME,
    STRATUM_HEADING,
    TARGET_ARM,
    TARGET_PATH,
    TARGET_TASK,
    UNVERIFIABLE,
    Handoff,
    HandoffTarget,
    SuccessorIndex,
    audit_handoffs,
    classify_handoff,
    extract_handoffs,
    extract_targets,
    handoff_join_canary,
    iter_markdown_units,
    real_arm_slugs,
    summarise_handoffs,
    task_row_text,
)

TODAY = date(2026, 8, 2)
ARMS = frozenset({"sv2", "os1", "pfs1", "sv1", "n64", "c1"})


def _memo(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _index(touched: dict[str, date], *, available=True, tracked=None, since=None):
    return SuccessorIndex(
        touched=touched,
        available=available,
        since=since,
        reason="test",
        tracked=frozenset(tracked) if tracked is not None else frozenset(),
    )


def _row(**kw):
    base = {
        "memo": "m_20260701.md", "anchor": "L1", "line_no": 1, "text": "hand off",
        "memo_date": date(2026, 7, 1), "stratum": STRATUM_FRAME, "heading": "NEXT-IF-RESUMED",
        "targets": (),
    }
    base.update(kw)
    return Handoff(**base)


# The canary's positive control must be present in any index an `audit_handoffs` test supplies,
# otherwise the audit correctly REFUSES and the test measures the refusal instead of its subject.
_CANARY_TOUCH = {"src/tac/followon_ledger.py": date(2026, 8, 1)}


# ---------------------------------------------------------------------------
# The structural defect: line scoping cannot see either real instance.
# ---------------------------------------------------------------------------

# Verbatim shape of `ddm_sv2_...20260802.md:350-353` -- the numbered item whose TARGET (`#539`) and
# whose VERB ("hand off") are on DIFFERENT physical lines.
_SV2_SHAPE = """## 5. WHAT THIS OWES NEXT

2. Wire the **registered** power-diagram form (`affine_head_to_power_diagram`,
   `power_diagram_witness.py:477`) into the describe loop — **~127 file-hits of argmax reformulation
   exist and none is wired here.** #539 (`Build the POWER-DIAGRAM witness parametrization`, in_progress)
   is the natural donor; this arm should hand off to it rather than fork a parallel surface.
3. Emit per-element survival as typed JSONL from the **existing** pt1 path — not a new instrument.
"""

# Verbatim shape of `ddm_os1_...20260802.md:203-205`.
_OS1_SHAPE = """## 5. What I did NOT do, and why

* **No cure landed at pfs1.** The census is the finding; the fix (split the fused exit, emit
  `stop_reason`, then free the ladder) is unreviewed new code on a live-chain solver and is
  STAGED, not taken — sv1's own discipline, and the fix changes the shipped solve.
"""


def test_no_single_line_carries_both_halves_of_the_sv2_instance():
    """sv2 is the STRUCTURAL case: no predicate can see it one line at a time.

    An earlier draft of this test asserted the same of BOTH instances and FAILED, which is how the
    overclaim in the module comment was caught. os1 is deliberately excluded — see the companion
    test below, which pins the opposite fact about it.
    """
    both = [
        ln for ln in _SV2_SHAPE.splitlines()
        if HANDOFF_FRAME_RX.search(ln) and extract_targets(ln, ARMS)
    ]
    assert both == [], f"a single line carried both halves: {both}"


def test_the_os1_instance_is_a_predicate_failure_not_a_scoping_one():
    """os1 fits on one line; it is invisible to the shipped extractor because ACTION∧CHEAP misses it."""
    from tac.followon_ledger import ACTION_RX, CHEAP_RX

    one_liner = [
        ln for ln in _OS1_SHAPE.splitlines()
        if HANDOFF_FRAME_RX.search(ln) and extract_targets(ln, ARMS)
    ]
    assert one_liner, "os1 should be visible to the handoff predicate on a single line"
    assert not any(ACTION_RX.search(ln) and CHEAP_RX.search(ln) for ln in _OS1_SHAPE.splitlines())


def test_item_scoping_joins_the_lines_that_line_scoping_separates():
    for shape in (_SV2_SHAPE, _OS1_SHAPE):
        hits = [
            u for _, u, _ in iter_markdown_units(shape)
            if HANDOFF_FRAME_RX.search(u) and extract_targets(u, ARMS)
        ]
        assert hits, f"item scoping still cannot see the handoff in:\n{shape}"


def test_sibling_bullets_do_not_merge_into_one_unit():
    """If bullets merged, a target in one item could pair with a verb in an unrelated neighbour."""
    body = "* first item names src/tac/aaa.py and nothing else at all here\n" \
           "* second item says this arm should hand off the work to somebody\n"
    units = [u for _, u, _ in iter_markdown_units(body)]
    assert not any("aaa.py" in u and "hand off" in u for u in units)


def test_extract_handoffs_finds_the_sv2_instance(tmp_path):
    _memo(tmp_path, "ddm_sv2_rebase_20260802.md", _SV2_SHAPE)
    rows, ledger = extract_handoffs(tmp_path, arms=ARMS)
    assert ledger.examined == 1
    tasks = {t.key for r in rows for t in r.targets if t.kind == TARGET_TASK}
    assert "#539" in tasks


def test_extract_handoffs_finds_the_os1_instance(tmp_path):
    _memo(tmp_path, "ddm_os1_census_20260802.md", _OS1_SHAPE)
    rows, _ = extract_handoffs(tmp_path, arms=ARMS)
    arms_named = {t.key for r in rows for t in r.targets if t.kind == TARGET_ARM}
    assert "pfs1" in arms_named


def test_a_runaway_block_cannot_swallow_a_document(tmp_path):
    """max_lines caps a unit so unrelated sentences cannot manufacture co-occurrence."""
    body = "\n".join(["prose line about nothing in particular"] * 40)
    units = list(iter_markdown_units(body, max_lines=4))
    assert units and all(u.count("prose line") <= 4 for _, u, _ in units)


# ---------------------------------------------------------------------------
# Target extraction: the #829 collision class.
# ---------------------------------------------------------------------------


def test_arm_target_requires_membership_in_the_real_arm_corpus():
    unit = "the work is handed off to zz9 and to pfs1 for the solver cure, per the census above"
    keys = {t.key for t in extract_targets(unit, ARMS) if t.kind == TARGET_ARM}
    assert "pfs1" in keys
    assert "zz9" not in keys, "a slug with no ddm_zz9_* artifact must never be admitted"


def test_real_arm_slugs_is_derived_from_artifacts_not_hardcoded():
    slugs = real_arm_slugs()
    assert len(slugs) > 50, "the live corpus should yield many arms"
    assert "fo1" in slugs and "pfs1" in slugs


def test_targets_are_deduplicated_and_keep_their_kind():
    unit = "hand off #539 and #539 and src/tac/x.py to pfs1, per the natural donor argument above"
    tgts = extract_targets(unit, ARMS)
    assert len(tgts) == len({(t.kind, t.key) for t in tgts})
    assert {t.kind for t in tgts} == {TARGET_TASK, TARGET_PATH, TARGET_ARM}


# ---------------------------------------------------------------------------
# The two precision corrections measured in round-1 review.
# ---------------------------------------------------------------------------


def test_handoff_is_not_an_owed_heading_word():
    """MEASURED: in this corpus "handoff" names CUSTODY sections, not owed work."""
    for ambient in (
        "Serializer and exact-hash handoff",
        "Verification, triality, and handoff",
        "Triality handoff",
        "Blocker delta and MAIN handoff",
    ):
        assert not OWED_HEADING_RX.search(ambient), ambient
    for real in ("NEXT-IF-RESUMED", "WHAT THIS OWES NEXT", "What I did NOT do", "OWED triality legs"):
        assert OWED_HEADING_RX.search(real), real


def test_a_data_table_row_cannot_qualify_on_heading_context_alone(tmp_path):
    """The measured false ORPHANED: a measurement table under a "…handoff" heading."""
    _memo(tmp_path, "ddm_v19b_feed_20260723.md", (
        "## Scale ladder and c1 handoff — NEXT-IF-RESUMED\n\n"
        "| n64 | 61,087 | -0.001589775085 | -0.000453630121 | 2,344 | -0.157473 |\n"
    ))
    rows, _ = extract_handoffs(tmp_path, arms=ARMS)
    assert rows == (), f"a pure data row was read as a handoff: {[r.text for r in rows]}"


def test_a_table_row_still_qualifies_when_it_carries_an_explicit_frame(tmp_path):
    """Table-formatted owed queues are real; they are held to the stricter FRAME bar, not dropped."""
    _memo(tmp_path, "ddm_dq_ledger_20260729.md", (
        "## Deferral queue\n\n"
        "| QA20 | the vae1 R7 BB-ANS successor should be run by pfs1 | BLOCKED | open |\n"
    ))
    rows, _ = extract_handoffs(tmp_path, arms=ARMS)
    assert len(rows) == 1 and rows[0].stratum == STRATUM_FRAME


def test_the_stratum_that_fired_travels_with_the_row(tmp_path):
    _memo(tmp_path, "m_20260701.md", (
        "## NEXT-IF-RESUMED\n\nSomebody must finish wiring src/tac/witness_dsl/thing.py before the gate.\n"
    ))
    rows, _ = extract_handoffs(tmp_path, arms=ARMS)
    assert len(rows) == 1 and rows[0].stratum == STRATUM_HEADING


# ---------------------------------------------------------------------------
# The join. ORPHANED must be EARNED; every gap degrades to UNVERIFIABLE.
# ---------------------------------------------------------------------------


def test_activity_after_the_naming_date_is_advanced_not_done():
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({"a/b.py": date(2026, 7, 20)}, tracked=["a/b.py"]),
        today=TODAY,
    )
    assert v.verdict == ADVANCED
    assert "never proof" in v.reason


def test_activity_BEFORE_the_naming_date_does_not_clear_the_edge():
    """A commit that predates the handoff cannot be the successor to it."""
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({"a/b.py": date(2026, 6, 1)}, tracked=["a/b.py"]),
        today=TODAY,
    )
    assert v.verdict == ORPHANED


def test_a_young_edge_is_live_never_orphaned():
    v = classify_handoff(
        _row(memo_date=date(2026, 8, 2), targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({"z/z.py": date(2026, 7, 1)}, tracked=["a/b.py"]),
        today=TODAY,
    )
    assert v.verdict == LIVE


def test_both_real_instances_are_live_because_they_were_named_today():
    """Re-derived, against the seed's own premise: age decides, and both are 0 days old."""
    for memo in ("ddm_sv2_rebase_20260802.md", "ddm_os1_census_20260802.md"):
        v = classify_handoff(
            _row(memo=memo, memo_date=date(2026, 8, 2),
                 targets=(HandoffTarget(TARGET_TASK, "#539"),)),
            _index({}, available=False), today=TODAY,
        )
        assert v.verdict == LIVE


def test_an_unreadable_git_channel_never_yields_orphaned():
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({}, available=False), today=TODAY,
    )
    assert v.verdict == UNVERIFIABLE and "could not look" in v.reason


def test_an_untracked_path_never_yields_orphaned():
    """A commit-touch channel is blind to untracked files BY CONSTRUCTION."""
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_PATH, "experiments/gitignored.py"),)),
        _index({"a/b.py": date(2026, 7, 9)}, tracked=["a/b.py"]),
        today=TODAY,
    )
    assert v.verdict == UNVERIFIABLE and "UNTRACKED" in v.reason


def test_a_tracked_untouched_path_is_orphaned():
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({"other.py": date(2026, 7, 9)}, tracked=["a/b.py", "other.py"]),
        today=TODAY,
    )
    assert v.verdict == ORPHANED


def test_undetermined_tracking_does_not_suppress_the_verdict():
    """Empty `tracked` means "not determined" and must not be read as "untracked"."""
    idx = _index({"other.py": date(2026, 7, 9)})
    assert idx.tracks("a/b.py") is None
    v = classify_handoff(_row(targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)), idx, today=TODAY)
    assert v.verdict == ORPHANED


def test_a_bare_task_id_with_no_ledger_is_unverifiable_never_orphaned():
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_TASK, "#539"),)),
        _index({"a/b.py": date(2026, 7, 9)}, tracked=["a/b.py"]),
        today=TODAY,
    )
    assert v.verdict == UNVERIFIABLE and not v.joinable


def test_a_supplied_task_ledger_makes_a_task_target_decidable():
    idx = _index({"a/b.py": date(2026, 7, 9)}, tracked=["a/b.py"])
    row = _row(targets=(HandoffTarget(TARGET_TASK, "#539"),))
    assert classify_handoff(
        row, idx, today=TODAY, closed_task_ids=frozenset({"539"})
    ).verdict == ADVANCED
    assert classify_handoff(
        row, idx, today=TODAY, closed_task_ids=frozenset({"111"})
    ).verdict == ORPHANED


def test_a_row_predating_the_indexed_window_is_unverifiable():
    """Absence of evidence outside the scanned window is a scan bound, never a finding."""
    v = classify_handoff(
        _row(memo_date=date(2026, 5, 1), targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({"other.py": date(2026, 7, 9)}, tracked=["a/b.py", "other.py"],
               since=date(2026, 7, 1)),
        today=TODAY,
    )
    assert v.verdict == UNVERIFIABLE and "scan bound" in v.reason


def test_evidence_inside_the_window_still_clears_a_row_that_predates_it():
    """The scan-bound guard must not suppress real activity that WAS observed."""
    v = classify_handoff(
        _row(memo_date=date(2026, 5, 1), targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({"a/b.py": date(2026, 7, 9)}, tracked=["a/b.py"], since=date(2026, 7, 1)),
        today=TODAY,
    )
    assert v.verdict == ADVANCED


def test_a_memo_with_no_date_is_unverifiable():
    v = classify_handoff(
        _row(memo_date=None, targets=(HandoffTarget(TARGET_PATH, "a/b.py"),)),
        _index({}, tracked=["a/b.py"]), today=TODAY,
    )
    assert v.verdict == UNVERIFIABLE and not v.joinable


def test_the_arm_channel_matches_on_basename_not_directory():
    """A directory named after an arm must not vote for it; only files named after it do."""
    idx = _index({"experiments/results/ddm_sv2_run/out.txt": date(2026, 7, 20)},
                 tracked=["experiments/results/ddm_sv2_run/out.txt"])
    v = classify_handoff(
        _row(targets=(HandoffTarget(TARGET_ARM, "sv2"),)), idx, today=TODAY)
    assert v.verdict == ORPHANED


# ---------------------------------------------------------------------------
# Canary, scope, and ordering.
# ---------------------------------------------------------------------------


def test_canary_fails_when_the_channel_is_dead():
    ok, note = handoff_join_canary(_index({}, available=False), today=TODAY)
    assert not ok and "POSITIVE CONTROL FAILED" in note


def test_audit_refuses_to_return_rows_when_the_canary_fails(tmp_path):
    _memo(tmp_path, "m_20260701.md", "## NEXT-IF-RESUMED\n\nhand off src/tac/a.py to somebody now.\n")
    paired, scope, note = audit_handoffs(
        tmp_path, index=_index({}, available=False), today=TODAY)
    assert paired == [] and scope.examined == 0 and "REFUSED" in scope.note


def test_scope_reports_the_denominator_and_is_vacuous_on_an_empty_scope(tmp_path):
    (tmp_path / "only_20260101.md").write_text("nothing here at all, just prose", encoding="utf-8")
    _rows, ledger = extract_handoffs(tmp_path, since=date(2026, 7, 1), arms=ARMS)
    assert ledger.population == 1 and ledger.declared == 0
    assert ledger.is_vacuous, "a filter that empties the scope must not read as a clean pass"


def test_rows_come_back_worst_first(tmp_path):
    _memo(tmp_path, "m_20260701.md", (
        "## NEXT-IF-RESUMED\n\n"
        "Finish the wiring in src/tac/orph.py before anything else happens here.\n\n"
        "## NEXT-IF-RESUMED\n\n"
        "Also finish the wiring in src/tac/adv.py before anything else happens here.\n"
    ))
    idx = _index({"src/tac/adv.py": date(2026, 7, 20), **_CANARY_TOUCH},
                 tracked=["src/tac/adv.py", "src/tac/orph.py"])
    paired, _scope, _n = audit_handoffs(tmp_path, index=idx, today=TODAY)
    verdicts = [v.verdict for _r, v in paired]
    assert verdicts and verdicts[0] == ORPHANED
    hist = summarise_handoffs(paired)
    assert hist[ORPHANED] >= 1 and hist[ADVANCED] >= 1


def test_row_id_is_memo_scoped():
    a = _row(memo="a.md", anchor="L7")
    b = _row(memo="b.md", anchor="L7")
    assert a.row_id != b.row_id and a.row_id.startswith("a.md#")


# ---------------------------------------------------------------------------
# The field-alias fix: the silent 100%-UNKNOWN vacuity on native harness rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        {"subject": "land `wr1_kneeA_realized_gate_receipt.json`"},
        {"description": "land `wr1_kneeA_realized_gate_receipt.json`"},
        {"title": "land `wr1_kneeA_realized_gate_receipt.json`"},
    ],
)
def test_task_rows_join_under_harness_key_names_and_canonical_ones(row):
    """MEASURED: the live harness emits subject/description; neither was read before this fix."""
    assert "wr1_kneeA_realized_gate_receipt.json" in task_row_text(row)


def test_task_row_text_is_empty_only_when_the_row_really_has_no_text():
    assert task_row_text({"unrelated_key": "x"}).strip() == ""
