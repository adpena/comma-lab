"""Tests for :mod:`tac.followon_ledger`, including the MEASURED control set.

The control set (``test_control_set_*``) is the part that decides whether the instrument is real.
It is six follow-ons whose true execution status was established BY HAND from primary artifacts
(SSD run logs, receipts, git commits) by an independent verifier, not from any memo's
self-description -- three of the six memos describe their own status WRONGLY.

Per design philosophy P4 ("no meter without a canary") a new measurement surface ships with a
positive AND a negative control before its readings count.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tac.followon_ledger import (
    EXECUTED,
    ORPHANED,
    STAGED,
    UNKNOWN,
    VALID_VERDICTS,
    ExecutionCorpus,
    FollowOn,
    _candidate_tokens,
    _distinctive_tokens,
    build_doc_frequency,
    classify_execution,
    extract_followons,
    memo_date,
    summarise,
)
from tac.scope_ledger import VACUOUS


# --------------------------------------------------------------------------- helpers
def _memo(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _corpus(names: set[str]) -> ExecutionCorpus:
    return ExecutionCorpus(produced_names=frozenset(names), produced_paths={})


def _row(text: str, tokens: tuple[str, ...]) -> FollowOn:
    return FollowOn(memo="m.md", anchor="L1", line_no=1, text=text,
                    memo_date=date(2026, 7, 25), tokens=tokens)


# --------------------------------------------------------------------------- extraction
def test_extraction_requires_the_conjunction_not_either_half():
    """Bare ``$0`` fires 12,684x repo-wide; a bare next-action word is likewise ambient."""
    df = build_doc_frequency([])
    cheap_only = "This whole arm was $0 and ran entirely on the local machine today."
    action_only = "A follow-on measurement is owed here before anyone draws a verdict on it."
    both = "A $0 scorer-free follow-on is owed: run `experiments/foo_probe.py` on the streams."
    from tac.followon_ledger import ACTION_RX, CHEAP_RX

    assert CHEAP_RX.search(cheap_only) and not ACTION_RX.search(cheap_only)
    assert ACTION_RX.search(action_only) and not CHEAP_RX.search(action_only)
    assert ACTION_RX.search(both) and CHEAP_RX.search(both)
    assert _distinctive_tokens(both, df | {"experiments/foo_probe.py": 1}, max_docs=8)


def test_extraction_reports_the_denominator_and_never_says_pass_on_an_empty_scope(tmp_path):
    """The vacuity genus: an empty scope must not be reportable as a clean run."""
    _memo(tmp_path, "unrelated_20260725.md", "nothing to see here at all, no markers present\n")
    rows, led = extract_followons(tmp_path, since=date(2030, 1, 1))
    assert rows == ()
    assert led.verdict == VACUOUS
    assert "PASS" not in led.render()
    assert led.population == 1


def test_extraction_finds_a_real_followon_with_memo_scoped_identity(tmp_path):
    _memo(
        tmp_path,
        "ddm_zz1_thing_20260725.md",
        "# head\n\nsome prose\n"
        "A $0 scorer-free follow-on is owed: run `experiments/zz1_probe.py` and land its receipt.\n",
    )
    rows, led = extract_followons(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row.row_id == "ddm_zz1_thing_20260725.md#L4"
    assert row.memo_date == date(2026, 7, 25)
    assert led.examined == 1


def test_anchor_is_a_locator_never_a_search_key(tmp_path):
    """#829 bug class: short row ids collide repo-wide, so they anchor but never join."""
    _memo(
        tmp_path,
        "ledger_20260725.md",
        "| QA08 | a $0 re-race is owed on `experiments/qa08_thing.py` before any verdict |\n",
    )
    rows, _ = extract_followons(tmp_path)
    assert rows[0].anchor.startswith("QA08@L")
    assert "QA08" not in rows[0].tokens  # the label anchors; it is never a join token


# --------------------------------------------------------------------------- tokens
def test_join_tokens_are_artifact_shaped_only():
    """Rare-but-incidental identifiers were MEASURED to mis-join; only filenames may key it."""
    text = "re-race owed, $0: see prev_coloc and ddm_sb1_20260729 and `experiments/real_thing.py`"
    toks = _candidate_tokens(text)
    assert "experiments/real_thing.py" in toks
    assert "prev_coloc" not in toks
    assert "ddm_sb1_20260729" not in toks


def test_jsonl_is_not_truncated_to_json():
    """Regex alternation is first-match-wins: ``json`` before ``jsonl`` silently truncated every
    ``.jsonl`` to a filename that exists nowhere, yielding confident FALSE ORPHANED verdicts."""
    toks = _candidate_tokens("cached counterfactual d2_ep_solve.partial.jsonl, 600 rows")
    assert "d2_ep_solve.partial.jsonl" in toks
    assert "d2_ep_solve.partial.json" not in toks


def test_partial_glob_captures_are_rejected():
    """``ddm_*_receipt.json`` must not yield the orphan basename ``receipt.json``."""
    assert not any(t.startswith("_") for t in _candidate_tokens("see ddm_pb1_*_receipt_2026.json"))


# --------------------------------------------------------------------------- the join
def test_output_present_is_executed():
    row = _row("owed, $0: land `zz1_receipt.json`", ("zz1_receipt.json",))
    v = classify_execution(row, _corpus({"zz1_receipt.json"}))
    assert v.verdict == EXECUTED


def test_output_missing_is_orphaned():
    row = _row("owed, $0: land `zz1_receipt.json`", ("zz1_receipt.json",))
    v = classify_execution(row, _corpus(set()))
    assert v.verdict == ORPHANED


def test_runner_present_but_no_output_is_STAGED_never_executed():
    """The heart of the class: a runner existing proves it was BUILT, never that it RAN."""
    row = _row("staged, $0, one command: `experiments/stage_zz1.sh`", ("experiments/stage_zz1.sh",))
    v = classify_execution(row, _corpus({"stage_zz1.sh"}))
    assert v.verdict == STAGED
    assert v.verdict != EXECUTED


def test_unjoinable_row_is_UNKNOWN_and_says_so():
    """The third bucket. A detector that cannot say 'I don't know' fabricates."""
    row = _row("a $0 follow-on is owed on the tail pairs", ())
    v = classify_execution(row, _corpus(set()))
    assert v.verdict == UNKNOWN
    assert v.joinable is False


def test_basename_matching_survives_a_pathful_token():
    row = _row("owed, $0: `a/b/c/zz1_receipt.json`", ("a/b/c/zz1_receipt.json",))
    assert classify_execution(row, _corpus({"zz1_receipt.json"})).verdict == EXECUTED


def test_every_verdict_is_declared():
    for v in (EXECUTED, ORPHANED, STAGED, UNKNOWN):
        assert v in VALID_VERDICTS


# --------------------------------------------------------------------------- mutation guard
def test_a_broken_join_would_fail_these_tests():
    """Guard against the 'tests verify constants not behavior' fake (NO-FAKE class 2).

    If ``classify_execution`` were replaced by a constant, these two rows -- identical except for
    what exists on disk -- would return the same verdict. They must not.
    """
    row = _row("owed, $0: land `zz1_receipt.json`", ("zz1_receipt.json",))
    assert (
        classify_execution(row, _corpus({"zz1_receipt.json"})).verdict
        != classify_execution(row, _corpus(set())).verdict
    )


def test_summarise_reports_all_buckets_even_at_zero():
    assert summarise([]) == {ORPHANED: 0, STAGED: 0, UNKNOWN: 0, EXECUTED: 0}


def test_memo_date_parses_and_tolerates_absence(tmp_path):
    assert memo_date(Path("ddm_x_20260725.md")) == date(2026, 7, 25)
    assert memo_date(Path("no_date_here.md")) is None
    assert memo_date(Path("bad_20261345.md")) is None  # month 13 -> None, never a crash


def test_unreadable_memo_does_not_abort_the_sweep(tmp_path):
    """Read total-but-LOUD. One bad file cost this campaign 100% of fused recall once."""
    _memo(tmp_path, "ok_20260725.md",
          "A $0 follow-on is owed: run `experiments/ok_probe.py` now.\n")
    bad = tmp_path / "bad_20260725.md"
    bad.write_bytes(b"\xff\xfe\x00 a $0 follow-on is owed on `experiments/x.py`\n")
    rows, led = extract_followons(tmp_path)
    assert led.examined == 2  # both counted; neither crashed the walk
    assert any(r.memo == "ok_20260725.md" for r in rows)


# --------------------------------------------------------------------------- CONTROL SET
# Ground truth established by hand from primary artifacts (SSD logs / receipts / git), NOT from
# any memo's self-description. Three of these six memos state their own status incorrectly.
CONTROL_SET = [
    # (label, followon text as written, tokens, artifacts on disk, expected verdict)
    (
        "wr1-QA08-re-race",
        "a QA08 re-race on the dropped streams is a $0 staged follow-on; "
        "receipt `tw1_state_dependence_receipt.json`",
        ("tw1_state_dependence_receipt.json",),
        {"tw1_state_dependence_receipt.json"},
        EXECUTED,
    ),
    (
        "826-cell-drop50",
        "#826 cell_drop50 is byte-closed but never evaluated; owed: `d1_eval_receipt.json`",
        ("d1_eval_receipt.json",),
        {"d1_eval_receipt.json"},
        EXECUTED,
    ),
    (
        "ba31-B3-split",
        "the position/label split is $0 and scorer-free; owed: "
        "`ddm_dc1_label_price_n600_20260801.json`",
        ("ddm_dc1_label_price_n600_20260801.json",),
        {"ddm_dc1_label_price_n600_20260801.json"},
        EXECUTED,
    ),
    (
        "v4d-realized-gate",
        "the staged gate is one command and not yet fired: `experiments/stage_v4d_realized_gate.sh`",
        ("experiments/stage_v4d_realized_gate.sh",),
        {"stage_v4d_realized_gate.sh"},
        STAGED,  # runner-only row: correctly NOT decidable, routed to the adjudicate-first bucket
    ),
    (
        "ck1-composed-gate",
        "built and staged, $0 one command, never fired: `experiments/stage_ck1_composed_gate.sh`",
        ("experiments/stage_ck1_composed_gate.sh",),
        {"stage_ck1_composed_gate.sh"},
        STAGED,  # the ONE true orphan of the six -- and STAGED is where it must land
    ),
    (
        "ck1-gate-output",
        "owed, $0: the composed gate receipt `ck1_composed_gate_receipt.json`",
        ("ck1_composed_gate_receipt.json",),
        set(),
        ORPHANED,
    ),
]


@pytest.mark.parametrize("label,text,tokens,on_disk,expected", CONTROL_SET)
def test_control_set_verdicts(label, text, tokens, on_disk, expected):
    row = _row(text, tokens)
    assert classify_execution(row, _corpus(on_disk)).verdict == expected, label


def test_unreadable_artifact_tier_degrades_orphaned_to_unknown():
    """The vacuity genus applied to the join's own evidence base.

    MEASURED: wired into the costate digest without the SSD tiers, the join reported 5 ORPHANED
    rows whose receipts were on the mounted volume all along. When a declared tier cannot be read,
    "not produced" and "I could not look" are the same observation, and the honest symbol is
    UNKNOWN.
    """
    row = _row("owed, $0: land `zz1_receipt.json`", ("zz1_receipt.json",))
    blind = ExecutionCorpus(
        produced_names=frozenset(), produced_paths={},
        missing_tiers=("/Volumes/VertigoDataTier/pact",),
    )
    v = classify_execution(row, blind)
    assert v.verdict == UNKNOWN
    assert blind.artifact_scope_complete is False
    # ...and with the tier readable, the SAME row is a real orphan. Without this pair the test
    # above would pass on a classifier that returned UNKNOWN unconditionally.
    assert classify_execution(row, _corpus(set())).verdict == ORPHANED


def test_cache_never_serves_a_narrowed_scope_from_a_wider_index(tmp_path):
    """Round-2 self-review catch: the on-disk index is keyed to nothing, so a caller that narrowed
    the scope must build its own — otherwise it is handed outputs its own scope cannot see."""
    # Warm the default (SSD-inclusive) cache, then build a scope that EXCLUDES the SSD tiers.
    ExecutionCorpus.build(cache_ttl_s=6 * 3600.0)
    narrow = ExecutionCorpus.build(tmp_path, extra_receipt_dirs=(), cache_ttl_s=6 * 3600.0)
    # An SSD-only artifact must NOT appear in a scope that did not include the SSD.
    assert "kneeA_gate_run.log" not in narrow.produced_names
    assert narrow.missing_tiers == ()  # it declared no tiers, so none are missing


def test_indexed_suffixes_exactly_cover_what_the_join_can_decide():
    """A suffix a token may END in but the corpus does not INDEX would read as missing-on-disk
    when it was merely never looked at — a false ORPHANED with no symptom."""
    from tac.followon_ledger import _INDEXED_SUFFIXES, _OUTPUT_SUFFIXES, _RUNNER_SUFFIXES

    assert frozenset(_OUTPUT_SUFFIXES + _RUNNER_SUFFIXES) == _INDEXED_SUFFIXES
    for suffix in _INDEXED_SUFFIXES:
        assert _candidate_tokens(f"see `experiments/zz1_thing{suffix}` for it")


def test_control_set_never_calls_a_real_orphan_executed():
    """The damaging direction. A false EXECUTED buries live debt; a false UNKNOWN only costs a
    human glance. The instrument is therefore tuned to be conservative, and this asserts it."""
    for label, text, tokens, on_disk, expected in CONTROL_SET:
        got = classify_execution(_row(text, tokens), _corpus(on_disk)).verdict
        if expected in (ORPHANED, STAGED):
            assert got != EXECUTED, f"{label}: live debt reported as done"
