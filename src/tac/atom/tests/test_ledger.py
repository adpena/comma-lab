# SPDX-License-Identifier: MIT
"""Tests for the canonical fcntl-locked JSONL ledger."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from tac.atom.atom import Atom
from tac.atom.builders import build_arbitrary_value_atom
from tac.atom.ledger import (
    AtomLedgerCorruptError,
    append_atom,
    append_atoms_batch,
    load_atoms_strict,
    query_by_kind,
    query_by_min_predicted_impact,
    query_by_resolution_path,
)
from tac.atom.types import AtomKind, ResolutionPath


def _sample_atom(atom_id: str) -> Atom:
    return build_arbitrary_value_atom(
        atom_id=atom_id,
        file_path="f.py",
        current_value=1,
        predicted_replacement=2,
        resolution_path=ResolutionPath.EXPERIMENTAL,
        predicted_ev_delta_s=(-0.005, -0.001),
        cost_envelope_usd=1.0,
        literature_citation="cit",
    )


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """Redirect ATOM_LEDGER_PATH / LOCK to tmp for isolation."""
    tmp_jsonl = tmp_path / "atom_ledger.jsonl"
    tmp_lock = tmp_path / ".atom_ledger.lock"
    monkeypatch.setattr("tac.atom.ledger.ATOM_LEDGER_PATH", tmp_jsonl)
    monkeypatch.setattr("tac.atom.ledger.ATOM_LEDGER_LOCK", tmp_lock)
    return tmp_jsonl


class TestAppendAtom:
    def test_append_one_row(self, tmp_ledger: Path) -> None:
        atom = _sample_atom("a1")
        row = append_atom(atom)
        assert row["event_type"] == "registered"
        assert tmp_ledger.is_file()
        with tmp_ledger.open("r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["atom"]["atom_id"] == "a1"

    def test_rejects_non_atom(self, tmp_ledger: Path) -> None:
        with pytest.raises(TypeError, match="Atom"):
            append_atom({"not": "an atom"})  # type: ignore[arg-type]

    def test_appends_preserve_order(self, tmp_ledger: Path) -> None:
        for i in range(5):
            append_atom(_sample_atom(f"a{i}"))
        with tmp_ledger.open("r") as f:
            ids = [json.loads(line)["atom"]["atom_id"] for line in f]
        assert ids == ["a0", "a1", "a2", "a3", "a4"]


class TestAppendAtomsBatch:
    def test_batch_one_lock_acquisition(self, tmp_ledger: Path) -> None:
        atoms = [_sample_atom(f"b{i}") for i in range(10)]
        rows = append_atoms_batch(atoms)
        assert len(rows) == 10
        with tmp_ledger.open("r") as f:
            assert sum(1 for _ in f) == 10

    def test_empty_batch_no_op(self, tmp_ledger: Path) -> None:
        rows = append_atoms_batch([])
        assert rows == []
        assert not tmp_ledger.is_file()


class TestLoadAtomsStrict:
    def test_empty_when_missing(self, tmp_ledger: Path) -> None:
        rows = load_atoms_strict(tmp_ledger)
        assert rows == []

    def test_round_trip(self, tmp_ledger: Path) -> None:
        for i in range(3):
            append_atom(_sample_atom(f"c{i}"))
        rows = load_atoms_strict(tmp_ledger)
        assert len(rows) == 3
        assert [r["atom"]["atom_id"] for r in rows] == ["c0", "c1", "c2"]

    def test_raises_on_corrupt(self, tmp_ledger: Path) -> None:
        tmp_ledger.parent.mkdir(parents=True, exist_ok=True)
        tmp_ledger.write_text("not valid json\n")
        with pytest.raises(AtomLedgerCorruptError, match="JSON parse error"):
            load_atoms_strict(tmp_ledger)
        # File should be quarantined
        assert not tmp_ledger.is_file()
        quarantine = list(tmp_ledger.parent.glob(f"{tmp_ledger.stem}.jsonl.corrupt.*"))
        assert quarantine


class TestQueryHelpers:
    def test_query_by_kind(self, tmp_ledger: Path) -> None:
        a = _sample_atom("q1")
        append_atom(a)
        # Build a different-kind atom
        from tac.atom.builders import build_premise_verification_atom

        pv = build_premise_verification_atom(
            atom_id="pv1",
            premise="x",
            verified=True,
            verification_method="importlib",
        )
        append_atom(pv)
        arbitrary = query_by_kind(AtomKind.ARBITRARY_VALUE, path=tmp_ledger)
        premise = query_by_kind(AtomKind.PREMISE_VERIFICATION, path=tmp_ledger)
        assert len(arbitrary) == 1
        assert len(premise) == 1

    def test_query_by_resolution_path(self, tmp_ledger: Path) -> None:
        append_atom(_sample_atom("r1"))
        rows = query_by_resolution_path(ResolutionPath.EXPERIMENTAL, path=tmp_ledger)
        assert len(rows) == 1
        rows = query_by_resolution_path(ResolutionPath.FORMULA, path=tmp_ledger)
        assert len(rows) == 0

    def test_query_by_min_predicted_impact(self, tmp_ledger: Path) -> None:
        # Sample atom has predicted_impact_lower = -0.005
        append_atom(_sample_atom("imp1"))
        # >= -0.01 should match
        rows = query_by_min_predicted_impact(lower_bound_geq=-0.01, path=tmp_ledger)
        assert len(rows) == 1
        # >= 0.0 should not match (lower is -0.005 < 0)
        rows = query_by_min_predicted_impact(lower_bound_geq=0.0, path=tmp_ledger)
        assert len(rows) == 0


def _spawn_worker_append(tmp_jsonl: str, tmp_lock: str, worker_idx: int, n_rows: int) -> int:
    """Worker process for the spawn-pool stress test."""
    import importlib

    # Monkeypatch the module-level constants in the spawned process
    ledger_mod = importlib.import_module("tac.atom.ledger")
    ledger_mod.ATOM_LEDGER_PATH = Path(tmp_jsonl)
    ledger_mod.ATOM_LEDGER_LOCK = Path(tmp_lock)
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath

    written = 0
    for i in range(n_rows):
        atom = build_arbitrary_value_atom(
            atom_id=f"w{worker_idx}_row{i}",
            file_path="f",
            current_value=1,
            predicted_replacement=2,
            resolution_path=ResolutionPath.EXPERIMENTAL,
            predicted_ev_delta_s=(-0.001, -0.0001),
            cost_envelope_usd=0.0,
            literature_citation="cit",
        )
        ledger_mod.append_atom(atom)
        written += 1
    return written


class TestConcurrentAppendSafety:
    def test_4proc_spawn_pool_20_rows(self, tmp_ledger: Path) -> None:
        """4 procs x 5 rows = 20 total atoms; all must land without corruption."""
        ctx = multiprocessing.get_context("spawn")
        from tac.atom.ledger import ATOM_LEDGER_LOCK as patched_lock

        with ctx.Pool(processes=4) as pool:
            results = pool.starmap(
                _spawn_worker_append,
                [
                    (str(tmp_ledger), str(patched_lock), i, 5)
                    for i in range(4)
                ],
            )
        assert sum(results) == 20
        with tmp_ledger.open("r") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 20
        # All atom_ids unique
        ids = {r["atom"]["atom_id"] for r in rows}
        assert len(ids) == 20
