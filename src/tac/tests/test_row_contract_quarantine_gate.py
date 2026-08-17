# SPDX-License-Identifier: MIT
"""Controls for the row-contract-quarantine gate (two-landing half of #1081).

The FIX isolated the row in ``codex_to_claude_inbox``; this is the gate that
refuses the class's return.  It lands STRICT because the live count is
MEASURED zero across 7,532 modules under ``src/`` -- not asserted, scanned.

The predicate is MECHANISM-based: an exception counts as "row-contract" when a
``validate``-named function RAISES it.  Keying on the exception's own spelling
would repeat the name-keying mistake the GT-lineage census measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.confound_gates import check_no_row_contract_error_quarantines_the_ledger

_LEDGER = '''
import json, shutil
from dataclasses import replace
from pathlib import Path

class RowBad(Exception):
    pass

def _validate_row(row):
    if row.get("status") != "open":
        raise RowBad("bad status")

{body}
'''

_QUARANTINES = '''
def load(p):
    rows = []
    try:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            _validate_row(r)
            rows.append(r)
    except (json.JSONDecodeError, RowBad) as exc:{waiver}
        shutil.move(str(p), str(p) + ".corrupt")
        raise RuntimeError("quarantined") from exc
    return rows
'''

_ISOLATES = '''
def load(p):
    rows, broken = [], {}
    for line in Path(p).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            shutil.move(str(p), str(p) + ".corrupt")   # bytes: quarantine STAYS legal
            raise RuntimeError("corrupt") from exc
        try:
            _validate_row(r)
        except RowBad as exc:
            broken[r.get("event_id")] = str(exc)       # row: ISOLATED
            continue
        rows.append(r)
    return rows
'''

_DATACLASS_REPLACE = '''
def recode(packet):
    try:
        _validate_row(packet.row)
    except RowBad:
        packet = replace(packet, row={"status": "open"})   # a COPY, not a move
    return packet
'''


def _repo(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "ledger.py").write_text(_LEDGER.format(body=body), encoding="utf-8")
    return tmp_path


def test_quarantining_on_a_row_contract_error_is_refused(tmp_path: Path) -> None:
    """THE POSITIVE CONTROL: the gate must be able to FAIL."""
    repo = _repo(tmp_path, _QUARANTINES.format(waiver=""))
    v = check_no_row_contract_error_quarantines_the_ledger(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(v) == 1, v
    assert "RowBad" in v[0] and "shutil.move" in v[0]


def test_strict_mode_raises(tmp_path: Path) -> None:
    from tac.preflight import PreflightError

    repo = _repo(tmp_path, _QUARANTINES.format(waiver=""))
    with pytest.raises(PreflightError):
        check_no_row_contract_error_quarantines_the_ledger(repo_root=repo, strict=True)


def test_row_isolation_passes(tmp_path: Path) -> None:
    """The CURED shape -- and file corruption still quarantining -- is clean."""
    repo = _repo(tmp_path, _ISOLATES)
    assert (
        check_no_row_contract_error_quarantines_the_ledger(
            repo_root=repo, strict=True, verbose=False
        )
        == []
    )


def test_dataclasses_replace_is_not_a_file_move(tmp_path: Path) -> None:
    """The measured false positive: ``replace(packet, ...)`` copies a dataclass.

    Caught at ``pr106_sidecar_packet.py:3965`` by READING the flagged site
    instead of trusting the count.
    """
    repo = _repo(tmp_path, _DATACLASS_REPLACE)
    assert (
        check_no_row_contract_error_quarantines_the_ledger(
            repo_root=repo, strict=True, verbose=False
        )
        == []
    )


def test_waiver_is_respected(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        _QUARANTINES.format(waiver="  # ROW_CONTRACT_QUARANTINE_OK:per-key state machine"),
    )
    assert (
        check_no_row_contract_error_quarantines_the_ledger(
            repo_root=repo, strict=True, verbose=False
        )
        == []
    )


def test_live_repo_count_is_zero() -> None:
    """The claim that licenses STRICT-from-byte-one, re-derived not asserted."""
    assert check_no_row_contract_error_quarantines_the_ledger(strict=True, verbose=False) == []


def test_gate_is_registered() -> None:
    from tac.confound_gates import CONFOUND_GATES

    assert check_no_row_contract_error_quarantines_the_ledger in CONFOUND_GATES
