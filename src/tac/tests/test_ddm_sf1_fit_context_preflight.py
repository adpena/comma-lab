# SPDX-License-Identifier: MIT
"""Behaviour tests for the ddm_sf1 fit-context producer gate.

Pins the four properties that make the gate worth having: it REFUSES an
unstamped producer, it ACCEPTS a stamped one, a placeholder waiver does NOT buy
silence, and an EMPTY scope refuses rather than passing (the vacuity genus --
an empty scan and a clean scan must not emit the same symbol).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.ddm_sf1_fit_context_preflight import (
    check_solved_coefficients_stamp_fit_context,
    classify,
    scan,
    waiver_of,
)

# A minimal source that trips every in-scope predicate: emits rows, performs a
# fit, and carries both a solved coefficient pair and a partner coordinate.
_UNSTAMPED = '''
def _refit_ab(x):
    return 1.0, 0.0

def main():
    a_q, b_q = _refit_ab(0)
    rec = {"pair": 0, "a": a_q, "b": b_q, "beta_mag": 0.5, "p": [1.0]}
    fj.write(json.dumps(rec) + "\\n")
'''

_STAMPED = _UNSTAMPED.replace(
    '"p": [1.0]}',
    '"p": [1.0], "fitted_against": stamp_fit_context('
    'coefficient="ab", partners={"beta": 0.0})}',
)


def _mkrepo(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_unstamped_producer_is_refused(tmp_path: Path) -> None:
    root = _mkrepo(tmp_path, "solver.py", _UNSTAMPED)
    res = scan(root, roots=("experiments",))
    assert res["in_scope"] == 1
    assert res["refused"] == 1
    assert res["ok"] is False
    with pytest.raises(ValueError, match="REFUSED"):
        check_solved_coefficients_stamp_fit_context(root, roots=("experiments",))


def test_stamped_producer_passes(tmp_path: Path) -> None:
    root = _mkrepo(tmp_path, "solver.py", _STAMPED)
    res = check_solved_coefficients_stamp_fit_context(root, roots=("experiments",))
    assert res["in_scope"] == 1 and res["refused"] == 0 and res["ok"] is True


def test_empty_scope_is_vacuous_not_a_pass(tmp_path: Path) -> None:
    """The genus this repo has already been bitten by: skip == green."""
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "experiments" / "unrelated.py").write_text("x = 1\n")
    res = scan(tmp_path, roots=("experiments",))
    assert res["in_scope"] == 0
    assert res["vacuous"] is True
    assert res["ok"] is False  # NOT a pass
    with pytest.raises(ValueError, match="VACUOUS"):
        check_solved_coefficients_stamp_fit_context(tmp_path, roots=("experiments",))


def test_substantive_waiver_excuses_but_placeholder_does_not(tmp_path: Path) -> None:
    ok = _UNSTAMPED + "\n# FIT_CONTEXT_STAMP_WAIVED: re-emitter only, solves nothing\n"
    assert waiver_of(ok) == "re-emitter only, solves nothing"
    for bad in ("<rationale>", "TBD", "n/a", "reason", ""):
        assert waiver_of(f"# FIT_CONTEXT_STAMP_WAIVED:{bad}\n") is None
    root = _mkrepo(tmp_path, "solver.py", ok)
    res = check_solved_coefficients_stamp_fit_context(root, roots=("experiments",))
    assert res["waived"] == 1 and res["refused"] == 0


def test_out_of_scope_files_are_not_swept_in() -> None:
    """A row emitter that does not solve has no fit context to stamp."""
    r = classify(Path("x.py"),
                 'rec = {"a": 1, "b": 2, "p": 3}\nfj.write(json.dumps(rec))')
    assert r["in_scope"] is False  # no fit token
    r2 = classify(Path("x.py"),
                  'def _refit_ab(): ...\nrec = {"a": 1, "b": 2}\n'
                  'fj.write(json.dumps(rec))')
    assert r2["in_scope"] is False  # solved coeffs but no partner key


def test_live_repo_population_is_stamped_and_non_vacuous() -> None:
    """The live green case: the gate has a real subject and it is satisfied.

    Guards both failure directions at once -- a regression that unstamps a
    producer, and a rename that empties the scope and would otherwise read as a
    clean pass.
    """
    root = Path(__file__).resolve().parents[3]
    res = scan(root)
    assert res["vacuous"] is False, "gate lost its subject -- scope is empty"
    assert res["in_scope"] >= 2, res
    assert res["refused"] == 0, res["refused_paths"]
