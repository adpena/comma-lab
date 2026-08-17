# SPDX-License-Identifier: MIT
"""Controls for the payload-write-order gate (ddm_pl1, second landing).

The FIX for ddm_lr1/A2 landed in ddm_rg1b (60aefac081): the trainer now writes
its ``result`` JSON before the final checkpoint, refuses a directory at
``--save``/``--out`` at parse time, and reports ``best_step`` /
``improved_over_init``.  That fix is guarded by a per-module test that STRING-
MATCHES two literal call spellings in one function.  This is the repo-wide
gate: it derives bulk-vs-record roles from the primitives a helper actually
calls, so a rename cannot disarm it, and it sees the other 11,015 modules the
per-module test cannot.

The gate lands WARN-ONLY at a MEASURED live count of 10 over 11,016 modules.
It is not strict, and the tests below say so rather than pretending otherwise.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tac.confound_gates as cg
from tac.confound_gates import (
    CONFOUND_GATES,
    MIN_POSITIVE_CONTROL_COVERAGE,
    POSITIVE_CONTROLS,
    check_no_bulk_write_strands_the_ready_record,
    payload_write_order_population,
    positive_control_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_HELPERS = '''
import json
import torch
from pathlib import Path

def _atomic_torch_save(payload, path):
    torch.save(payload, path)

def _atomic_write_json(payload, path):
    Path(path).write_text(json.dumps(payload))
'''

# The EXACT ddm_lr1/A2 shape: record built, bulk save scheduled ahead of it.
_STRANDED = '''
def finalize(model, args, history):
    result = {"verdict": "PASS", "seg": 1.0, "history": history}
    _atomic_torch_save({"sd": model}, args.save){waiver}
    _atomic_write_json(result, args.out)
    return result
'''

# CURE 1 (what ddm_rg1b actually did): cheap-and-irreplaceable goes first.
_RECORD_FIRST = '''
def finalize(model, args, history):
    result = {"verdict": "PASS", "seg": 1.0, "history": history}
    _atomic_write_json(result, args.out)
    _atomic_torch_save({"sd": model}, args.save)
    return result
'''

# CURE 2: the fragile save cannot strand the record if it is guarded.
_GUARDED = '''
def finalize(model, args, history):
    result = {"verdict": "PASS", "seg": 1.0, "history": history}
    try:
        _atomic_torch_save({"sd": model}, args.save)
    except OSError:
        pass
    _atomic_write_json(result, args.out)
    return result
'''

# The common Python spelling the first draft of the gate could not see.
_WITH_OPEN = '''
import json

def finalize(model, args, history):
    result = {"verdict": "PASS", "seg": 1.0, "history": history}
    _atomic_torch_save({"sd": model}, args.save)
    with open(args.out, "w") as handle:
        json.dump(result, handle)
    return result
'''


def _plant(root: Path, body: str, *, waiver: str = "") -> None:
    # `.replace`, never `.format`: the fixtures contain dict literals, and
    # str.format reads `{"verdict"` as a field name.
    target = root / "src" / "tac" / "planted_module.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_HELPERS + body.replace("{waiver}", waiver))


def _run(root: Path) -> list[str]:
    return check_no_bulk_write_strands_the_ready_record(
        repo_root=root, strict=False, verbose=False
    )


# ---------------------------------------------------------------------------
# POSITIVE CONTROL -- the gate must be able to FAIL
# ---------------------------------------------------------------------------
def test_positive_control_the_stranded_record_is_flagged(tmp_path):
    _plant(tmp_path, _STRANDED)
    found = _run(tmp_path)
    assert len(found) == 1, found
    assert "src/tac/planted_module.py:" in found[0]
    assert "'result'" in found[0]
    assert "bulk payload write runs BEFORE" in found[0]


def test_positive_control_with_open_json_dump_is_flagged(tmp_path):
    """`with open(...) as fh: json.dump(...)` is the ordinary spelling.

    The first draft excluded every compound statement and missed this, which
    the registered positive control caught before landing.
    """
    _plant(tmp_path, _WITH_OPEN)
    found = _run(tmp_path)
    assert len(found) == 1, found


def test_the_gate_raises_in_strict_mode_on_a_planted_violation(tmp_path):
    from tac.preflight import PreflightError

    _plant(tmp_path, _STRANDED)
    with pytest.raises(PreflightError):
        check_no_bulk_write_strands_the_ready_record(
            repo_root=tmp_path, strict=True, verbose=False
        )


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS -- both cures pass
# ---------------------------------------------------------------------------
def test_record_written_first_passes(tmp_path):
    _plant(tmp_path, _RECORD_FIRST)
    assert _run(tmp_path) == []


def test_guarded_bulk_write_passes(tmp_path):
    """A save inside try/except cannot strand the record: the handler runs."""
    _plant(tmp_path, _GUARDED)
    assert _run(tmp_path) == []


def test_an_empty_tree_is_clean(tmp_path):
    (tmp_path / "src").mkdir()
    assert _run(tmp_path) == []


# ---------------------------------------------------------------------------
# PREDICATE BOUNDARIES -- the two measured false positives stay excluded
# ---------------------------------------------------------------------------
def test_a_small_dict_is_a_closure_box_not_a_record(tmp_path):
    """MEASURED FP: `_tail_cycle_start_epoch = {"v": None}` in the levelset
    trainer produced three hits before the key-count floor."""
    _plant(
        tmp_path,
        '''
def finalize(model, args):
    box = {"v": None}
    _atomic_torch_save({"sd": model}, args.save)
    _atomic_write_json(box, args.out)
'''.replace("{waiver}", ""),
    )
    assert _run(tmp_path) == []


def test_a_dict_of_paths_used_as_a_subscript_receiver_is_not_a_record(tmp_path):
    """MEASURED FP: experiments/tests/test_ddm_cx2_trace_evaluate.py builds a
    dict of FIXTURE PATHS; the dict is the receiver, never the serialised
    object.  The record must be what is being written, not a name on the line.
    """
    _plant(
        tmp_path,
        '''
def build(tmp):
    deps = {"a": tmp / "a", "b": tmp / "b", "c": tmp / "c"}
    deps["b"].write_bytes(b"payload")
    deps["a"].write_text("not the record")
'''.replace("{waiver}", ""),
    )
    assert _run(tmp_path) == []


def test_printing_the_result_is_not_persisting_it(tmp_path):
    """`print(json.dumps(result))` after the save is stdout, not an artifact.

    The CURED trainer does exactly this, and flagging it would have made the
    gate refuse its own fix.
    """
    _plant(
        tmp_path,
        '''
import json

def finalize(model, args, history):
    result = {"verdict": "PASS", "seg": 1.0, "history": history}
    _atomic_write_json(result, args.out)
    _atomic_torch_save({"sd": model}, args.save)
    print(json.dumps(result, indent=2))
'''.replace("{waiver}", ""),
    )
    assert _run(tmp_path) == []


def test_a_conditional_write_is_not_a_sibling(tmp_path):
    """`if`/`for` branch; a write under a condition is not straight-line."""
    _plant(
        tmp_path,
        '''
def finalize(model, args, history):
    result = {"verdict": "PASS", "seg": 1.0, "history": history}
    _atomic_torch_save({"sd": model}, args.save)
    if args.emit:
        _atomic_write_json(result, args.out)
'''.replace("{waiver}", ""),
    )
    assert _run(tmp_path) == []


def test_helper_role_is_derived_from_primitives_not_from_its_name(tmp_path):
    """Rename both helpers to nonsense; the roles must survive."""
    renamed = (_HELPERS + _STRANDED.replace("{waiver}", "")).replace(
        "_atomic_torch_save", "zzz_one"
    ).replace("_atomic_write_json", "zzz_two")
    target = tmp_path / "src" / "tac" / "planted_module.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(renamed)
    found = _run(tmp_path)
    assert len(found) == 1, found


# ---------------------------------------------------------------------------
# WAIVER -- present with a real rationale; placeholders rejected
# ---------------------------------------------------------------------------
def test_waiver_with_a_real_rationale_suppresses(tmp_path):
    _plant(
        tmp_path,
        _STRANDED,
        waiver="  # PAYLOAD_WRITE_ORDER_OK: the manifest records the saved artifact's real sha256",
    )
    assert _run(tmp_path) == []


@pytest.mark.parametrize("placeholder", ["<rationale>", "<reason>", ""])
def test_placeholder_rationales_do_not_waive(tmp_path, placeholder):
    _plant(tmp_path, _STRANDED, waiver=f"  # PAYLOAD_WRITE_ORDER_OK:{placeholder}")
    assert len(_run(tmp_path)) == 1


# ---------------------------------------------------------------------------
# REGISTRATION -- proven, not asserted
# ---------------------------------------------------------------------------
def test_the_gate_is_registered_in_the_confound_catalog():
    assert check_no_bulk_write_strands_the_ready_record in CONFOUND_GATES


def test_preflight_iterates_the_confound_catalog():
    """The wire-in is a loop over CONFOUND_GATES, so membership IS the wire-in."""
    source = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text()
    assert "from tac.confound_gates import CONFOUND_GATES as _CONFOUND_GATES" in source
    assert "for _confound_gate in _CONFOUND_GATES:" in source


def test_the_gate_is_warn_only_in_preflight_while_its_live_count_is_nonzero():
    """Honest wiring: it must NOT be in the strict set until the sites are cured."""
    source = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text()
    strict_block = source.split("_CONFOUND_STRICT = {", 1)[1].split("}", 1)[0]
    assert "check_no_bulk_write_strands_the_ready_record" not in strict_block


def test_the_gate_carries_a_live_positive_control():
    covered = {control.gate for control in POSITIVE_CONTROLS}
    assert "check_no_bulk_write_strands_the_ready_record" in covered


def test_positive_control_coverage_floor_was_ratcheted():
    coverage = positive_control_coverage()
    assert int(coverage["covered"]) >= MIN_POSITIVE_CONTROL_COVERAGE


# ---------------------------------------------------------------------------
# DENOMINATOR -- a count without its population is not a measurement
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_live_population_reports_its_denominator():
    population = payload_write_order_population()
    assert population["modules_examined"] > 10_000, population["modules_examined"]
    assert population["live_count"] == len(population["violations"])
    # "could not analyse" is reported, never folded into the cleared count.
    assert isinstance(population["unparsed"], list)


def test_the_docstring_states_the_measured_live_count_and_the_flip_condition():
    doc = check_no_bulk_write_strands_the_ready_record.__doc__ or ""
    assert "11,016" in doc
    assert "STRICT-FLIP CONDITION" in doc


def test_the_prefilter_is_a_superset_of_the_bulk_predicate():
    """The speed prefilter must never be able to hide a real bulk write.

    If someone teaches `_pl1_primitive_role` a new bulk spelling and forgets the
    token list, the gate goes silently blind on every module that uses only the
    new spelling. This pins the two lists together.
    """
    for attr in [*cg._BULK_ATTRS, "write_bytes"]:
        assert f".{attr}(" in cg._PL1_BULK_TOKENS, attr


def test_a_module_with_no_bulk_spelling_is_still_counted_as_examined(tmp_path):
    target = tmp_path / "src" / "tac" / "inert.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def f():\n    return {'a': 1, 'b': 2, 'c': 3}\n")
    population = payload_write_order_population(tmp_path)
    assert population["modules_examined"] == 1
    assert population["live_count"] == 0


def test_the_gate_module_parses_and_exposes_the_population_helper():
    module = REPO_ROOT / "src" / "tac" / "confound_gates.py"
    ast.parse(module.read_text())
    assert callable(payload_write_order_population)
