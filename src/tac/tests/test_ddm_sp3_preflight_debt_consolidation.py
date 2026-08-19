# SPDX-License-Identifier: MIT
"""ddm_sp3 — the three preflight-debt consolidations, with executed controls.

Covers:

1. **Catalog #348 CLASS-POPULATION scope extension** (meta-bug M1). A landing
   memo that records a fixed defect must state that defect class's population.
2. **``tac.fp16_floor_guard`` correctness repairs** found while reviewing the
   gate ``ddm_sp2`` built but never committed: a false positive on the canonical
   next-statement cure, a SILENT total blindness on ``#``-inside-a-string, and a
   missing waiver path on a STRICT gate.
3. **Catalog #344 registry round-trip repair** (task #1149): the durable
   canonical-equations registry must be JSON-round-trip exact on every row.

Every control here is executed, and each protection is pinned by BOTH a positive
control (the defect is caught) and a negative control (the cure scans clear), so
none of them can pass vacuously.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.fp16_floor_guard import (
    scan_text_for_fp16_destroyed_floors,
    waiver_rationale_is_substantive,
)
from tac.preflight import (
    _CHECK_348_CLASS_POP_CUTOFF,
    _check_348_class_population_line_violations,
    check_new_gate_landing_includes_retroactive_sweep_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# A memo body that CLAIMS A LANDED FIX (the trigger) but measures no population.
_FIX_MEMO_NO_POPULATION = """# ddm_zz9 — a fix batch

Row 1 premise VERIFIED. The defect was **FIXED** at the incident site and a
regression test landed beside it as the second landing.
"""

_POPULATION_BLOCK = """
### CLASS-POPULATION

```
live sites found ...... 12
live sites fixed ...... 12
```
"""


def _write_memo(root: Path, name: str, body: str) -> Path:
    research = root / ".omx" / "research"
    research.mkdir(parents=True, exist_ok=True)
    memo = research / name
    memo.write_text(body, encoding="utf-8")
    return memo


def _post_cutoff_name(stem: str = "ddm_zz9_fix_batch") -> str:
    """A memo name dated ON the forward-binding cutoff, so the gate scans it."""
    return f"{stem}_{_CHECK_348_CLASS_POP_CUTOFF}.md"


# ---------------------------------------------------------------------------
# 1. Catalog #348 CLASS-POPULATION scope extension
# ---------------------------------------------------------------------------


def test_positive_control_fix_memo_without_population_is_caught(tmp_path: Path):
    _write_memo(tmp_path, _post_cutoff_name(), _FIX_MEMO_NO_POPULATION)
    violations, scanned = _check_348_class_population_line_violations(tmp_path)
    assert scanned == 1, "the memo must be SCANNED, else the pass is vacuous"
    assert len(violations) == 1
    assert "CLASS-POPULATION" in violations[0]


def test_negative_control_the_cure_scans_clear(tmp_path: Path):
    _write_memo(
        tmp_path, _post_cutoff_name(), _FIX_MEMO_NO_POPULATION + _POPULATION_BLOCK
    )
    violations, scanned = _check_348_class_population_line_violations(tmp_path)
    assert scanned == 1
    assert violations == []


def test_a_live_count_table_also_satisfies_the_discipline(tmp_path: Path):
    """ddm_sp2 stated its census as a live-count table, not the literal header."""
    body = _FIX_MEMO_NO_POPULATION + "\n| state | sites |\n| live count | 30 |\n"
    _write_memo(tmp_path, _post_cutoff_name(), body)
    violations, _ = _check_348_class_population_line_violations(tmp_path)
    assert violations == []


def test_the_word_alone_does_not_satisfy_the_gauge(tmp_path: Path):
    """The detector must ZERO on its own cure: prose without a NUMBER is not a census.

    If merely writing "CLASS-POPULATION" cleared the gate, the gate would measure
    instrumentation rather than reality.
    """
    body = _FIX_MEMO_NO_POPULATION + "\n### CLASS-POPULATION\n\nnot measured yet.\n"
    _write_memo(tmp_path, _post_cutoff_name(), body)
    violations, _ = _check_348_class_population_line_violations(tmp_path)
    assert len(violations) == 1


def test_substantive_waiver_is_accepted(tmp_path: Path):
    body = (
        _FIX_MEMO_NO_POPULATION
        + "\n# CLASS_POPULATION_WAIVED: single-site typo in a frozen custody snapshot\n"
    )
    _write_memo(tmp_path, _post_cutoff_name(), body)
    violations, _ = _check_348_class_population_line_violations(tmp_path)
    assert violations == []


def test_placeholder_waiver_is_rejected(tmp_path: Path):
    body = _FIX_MEMO_NO_POPULATION + "\n# CLASS_POPULATION_WAIVED: <reason>\n"
    _write_memo(tmp_path, _post_cutoff_name(), body)
    violations, _ = _check_348_class_population_line_violations(tmp_path)
    assert len(violations) == 1, "placeholder rationales must not self-waive"


def test_the_gate_binds_forward_and_does_not_backfill(tmp_path: Path):
    """A memo dated BEFORE the cutoff is never scanned -- no retroactive flood."""
    _write_memo(tmp_path, "ddm_zz9_fix_batch_20260101.md", _FIX_MEMO_NO_POPULATION)
    violations, scanned = _check_348_class_population_line_violations(tmp_path)
    assert (violations, scanned) == ([], 0)


def test_a_memo_that_records_no_fix_is_not_scanned(tmp_path: Path):
    body = "# ddm_zz9 — a measurement\n\nMeasured d_seg at 0.0042 over n600.\n"
    _write_memo(tmp_path, _post_cutoff_name(), body)
    violations, scanned = _check_348_class_population_line_violations(tmp_path)
    assert (violations, scanned) == ([], 0)


def test_missing_research_dir_is_not_an_error(tmp_path: Path):
    assert _check_348_class_population_line_violations(tmp_path) == ([], 0)


def test_extension_rows_are_returned_but_never_raise(tmp_path: Path):
    """WARN-ONLY, pinned: the rows reach the caller and never the exception."""
    _write_memo(tmp_path, _post_cutoff_name(), _FIX_MEMO_NO_POPULATION)
    rows = check_new_gate_landing_includes_retroactive_sweep_evidence(
        strict=True, repo_root=tmp_path
    )
    assert any("CLASS-POPULATION" in row for row in rows), (
        "the extension's finding must still be REPORTED to the caller"
    )


# ---------------------------------------------------------------------------
# 2. tac.fp16_floor_guard — the three repairs
# ---------------------------------------------------------------------------

_PRE_FIX_SAME_STATEMENT = (
    "sc = ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)\n"
)
_PRE_FIX_CROSS_STATEMENT = (
    "scale = max(max_abs, 1e-8) / 127.0\n"
    "scale_fp16 = torch.tensor([scale], dtype=torch.float16)\n"
)


def test_positive_control_both_pre_fix_shapes_are_caught():
    assert scan_text_for_fp16_destroyed_floors(_PRE_FIX_SAME_STATEMENT, "p.py")
    assert scan_text_for_fp16_destroyed_floors(_PRE_FIX_CROSS_STATEMENT, "p.py")


def test_the_cure_written_on_the_next_statement_is_not_a_violation():
    """REPAIR 1. The canonical cure is naturally written across two statements.

    Before this repair the detector demanded the re-floor on the SAME line, so a
    correctly cured site was reported by a STRICT gate -- telling the engineer to
    do what they had already done.
    """
    cured = (
        "scale = max(max_abs, 1e-8) / 127.0\n"
        "sf = torch.tensor([scale], dtype=torch.float16)\n"
        "sf = sf.clamp_min(_FP16_MIN_POSITIVE)\n"
    )
    assert scan_text_for_fp16_destroyed_floors(cured, "p.py") == []


def test_a_hash_inside_a_string_no_longer_blinds_the_whole_file():
    """REPAIR 2, the severe one: a SILENT total blindness in a STRICT gate.

    ``line.find("#")`` truncated at a ``#`` inside a string literal, which
    unbalanced the brackets, which made the statement splitter return NOTHING for
    the rest of the file. The file then scanned clean AND added 0 to the
    denominator, so the vacuity guard could not see it either.
    """
    source = 'label = f("a#b(")\n' + _PRE_FIX_SAME_STATEMENT
    assert scan_text_for_fp16_destroyed_floors(source, "p.py"), (
        "a violation after a hash-in-string must still be found"
    )


def test_prose_quoting_the_bug_is_not_read_as_code():
    doc = '"""Doc: x.clamp(min=1e-8).to(torch.float16) is the bug."""\ny = 1\n'
    assert scan_text_for_fp16_destroyed_floors(doc, "p.py") == []


def test_substantive_fp16_waiver_accepted_and_placeholder_rejected():
    """REPAIR 3: a STRICT gate with no escape hatch had no waiver path at all."""
    waived = (
        "sc = t.clamp(min=1e-8).to(torch.float16)"
        "  # FP16_POSTCAST_FLOOR_OK: re-floored by the C++ writer on read-back\n"
    )
    placeholder = (
        "sc = t.clamp(min=1e-8).to(torch.float16)  # FP16_POSTCAST_FLOOR_OK: <reason>\n"
    )
    assert scan_text_for_fp16_destroyed_floors(waived, "p.py") == []
    assert scan_text_for_fp16_destroyed_floors(placeholder, "p.py")


@pytest.mark.parametrize(
    "rationale,expected",
    [
        ("re-floored by the C++ writer on read-back", True),
        ("<reason>", False),
        ("tbd", False),
        ("n/a", False),
        ("abc", False),
    ],
)
def test_waiver_rationale_substantiveness(rationale: str, expected: bool):
    assert waiver_rationale_is_substantive(rationale) is expected


# NOTE: the repo-wide sweep and the fp16 subnormal-boundary constant are already
# pinned by `test_fp16_scale_floor_guard.py` (ddm_fx4 / ddm_sp2, commit
# 98f24b3379). They are deliberately NOT repeated here -- one detector and one
# owner per assertion, per the split-bank discipline. This file only adds the
# regression coverage that suite does not have: the three ddm_sp3 repairs.


# ---------------------------------------------------------------------------
# 3. Catalog #330 string-literal awareness (ddm_fx3 debt item 1)
# ---------------------------------------------------------------------------

_TOKEN = "FunctionCall.from_id"


def _is_code(source: str, lineno: int) -> bool:
    from tac.preflight import _check_330_code_lines

    return _TOKEN in _check_330_code_lines(source)[lineno - 1]


def test_real_harvester_call_is_still_read_as_code():
    """Negative control: the cure must not blind the gate to real offenders."""
    source = "import modal\nresult = modal.functions.FunctionCall.from_id(cid).get()\n"
    assert _is_code(source, 2)


def test_docstring_continuation_is_no_longer_falsely_flagged():
    """The exact ddm_fx3 failure: a docstring CONTINUATION line starting with code.

    The old guard asked "does this line START with a quote?", so a continuation
    line beginning with ``print(`` read as live code even though it sits inside a
    docstring -- while a sibling hit inside an f-string was skipped. The detector
    disagreed with itself on the same file.
    """
    source = (
        '"""Recovery notes.\n'
        f"print(modal.functions.{_TOKEN}(cid).get())\n"
        '"""\n'
        "x = 1\n"
    )
    assert not _is_code(source, 2)


def test_single_line_fstring_recovery_command_is_not_a_harvester():
    """Measured: the weaker sanitise-only form added 3 false positives here."""
    source = f'msg = f"Recover: modal.{_TOKEN}(\'{{cid}}\').get(timeout=30)"\n'
    assert not _is_code(source, 1)


def test_comment_mentioning_the_call_is_not_code():
    assert not _is_code(f"# see modal.{_TOKEN}(cid).get()\nx = 1\n", 1)


def test_check_330_remains_green_on_the_live_repo():
    """This gate is STRICT: the rewire must not introduce a single new row."""
    from tac.preflight import check_modal_harvesters_record_call_id_outcome

    violations = check_modal_harvesters_record_call_id_outcome(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert violations == [], "\n  ".join(violations)


def test_blank_all_strings_is_opt_in_so_fp16_behaviour_is_unchanged():
    from tac.fp16_floor_guard import neutralize_prose

    source = 'x = "keeps float16 text"\n'
    assert "float16" in neutralize_prose(source)[0]
    assert "float16" not in neutralize_prose(source, blank_all_strings=True)[0]


# ---------------------------------------------------------------------------
# 4. Catalog #344 registry round-trip repair (task #1149)
# ---------------------------------------------------------------------------


def test_durable_registry_is_roundtrip_exact_on_every_row():
    """Task #1149: an anchor carrying keys the canonical model does not model.

    The two offending keys were written by an in-place edit of a historical row
    (commit ``5ab6506630``) and were dropped on every read, so no consumer ever
    saw them. Removing exactly those inert bytes left the reconstructed equation
    byte-identical while clearing the gate.
    """
    from tac.canonical_equations.registry import (
        audit_empirical_anchor_roundtrip_fidelity,
    )

    registry = REPO_ROOT / ".omx/state/canonical_equations_registry.jsonl"
    if not registry.is_file():
        pytest.skip("durable registry not present in this checkout")
    defects = audit_empirical_anchor_roundtrip_fidelity(registry)
    assert defects == (), "\n  ".join(
        f"line {d.registry_line} equation={d.equation_id} anchor={d.anchor_index} "
        f"at {', '.join(d.changed_json_paths[:5])}"
        for d in defects
    )
