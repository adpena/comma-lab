"""Tests for ``tools.meta_lagrangian_atom_ledger_adapter``."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "meta_lagrangian_atom_ledger_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "meta_lagrangian_atom_ledger_adapter", path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ledger(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_read_ledger_handles_missing_file(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    assert mod.read_atom_ledger(tmp_path / "nope.jsonl") == []


def test_read_ledger_skips_blank_and_corrupt_lines(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        "\n"
        '{"phase": 1, "contest_cuda_score": 0.18}\n'
        "{not valid json\n"
        "\n"
        '{"phase": 2, "contest_cuda_score": 0.15}\n'
    )
    records = mod.read_atom_ledger(path)
    assert len(records) == 2
    assert records[0]["phase"] == 1
    assert records[1]["phase"] == 2


def test_record_to_atom_extracts_fields() -> None:
    mod = _load_tool_module()
    rec = {
        "phase": 2,
        "substrate_label": "PR101 (gold)",
        "contest_cuda_score": 0.193,
        "archive_bytes": 178_258,
        "evidence_grade": "[contest-CUDA]",
        "cathedral_op": "Op1+Op2",
        "archive_sha256": "deadbeef",
        "notes": "test",
    }
    atom = mod.record_to_atom(rec, idx=0)
    assert atom.score == 0.193
    assert atom.rate_bytes == 178_258
    assert atom.evidence_grade == "[contest-CUDA]"
    assert atom.archive_sha256 == "deadbeef"


def test_record_to_atom_handles_none_score() -> None:
    mod = _load_tool_module()
    rec = {
        "phase": 1, "substrate_label": "test",
        "contest_cuda_score": None, "archive_bytes": None,
    }
    atom = mod.record_to_atom(rec, idx=0)
    assert atom.score is None
    assert atom.rate_bytes is None


def test_pareto_filter_removes_strictly_dominated(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    atoms = [
        # (atom_id, op, sub, rate, score, grade, sha, notes)
        mod.MetaLagrangianAtom("a1", "op", "s", 100, 0.20, "[contest-CUDA]", None, ""),
        mod.MetaLagrangianAtom("a2", "op", "s", 100, 0.21, "[contest-CUDA]", None, ""),  # dominated by a1
        mod.MetaLagrangianAtom("a3", "op", "s", 80,  0.22, "[contest-CUDA]", None, ""),  # not dominated (smaller rate)
        mod.MetaLagrangianAtom("a4", "op", "s", 50,  0.30, "[contest-CUDA]", None, ""),  # not dominated (smallest rate)
    ]
    nd = mod.filter_pareto_non_dominated(atoms)
    nd_ids = {a.atom_id for a in nd}
    assert "a1" in nd_ids
    assert "a2" not in nd_ids  # strictly dominated by a1
    assert "a3" in nd_ids
    assert "a4" in nd_ids


def test_pareto_filter_excludes_atoms_with_missing_data(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    atoms = [
        mod.MetaLagrangianAtom("complete", "op", "s", 100, 0.20, "[contest-CUDA]", None, ""),
        mod.MetaLagrangianAtom("no_score", "op", "s", 100, None, "[CPU-prep]", None, ""),
        mod.MetaLagrangianAtom("no_rate", "op", "s", None, 0.20, "[contest-CUDA]", None, ""),
    ]
    nd = mod.filter_pareto_non_dominated(atoms)
    ids = {a.atom_id for a in nd}
    assert "complete" in ids
    assert "no_score" not in ids
    assert "no_rate" not in ids


def test_emit_meta_lagrangian_input_writes_json(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [
        {"phase": 1, "substrate_label": "PR101", "contest_cuda_score": 0.193,
         "archive_bytes": 178_258, "evidence_grade": "[contest-CUDA]",
         "cathedral_op": "Op1+Op2"},
        {"phase": 2, "substrate_label": "PR101+RAFT", "contest_cuda_score": 0.180,
         "archive_bytes": 175_000, "evidence_grade": "[contest-CUDA]",
         "cathedral_op": "Op1+Op2+RAFT"},
    ])
    atoms = mod.ledger_to_atoms(ledger)
    output = tmp_path / "atoms.json"
    payload = mod.emit_meta_lagrangian_input(atoms, output, pareto_only=False)
    assert payload["n_atoms"] == 2
    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded["score_claim"] is False
    assert loaded["evidence_grade"] == "[empirical]"
    assert len(loaded["atoms"]) == 2


def test_emit_with_pareto_only_filters(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [
        # a1 dominates a2 (same rate, lower score)
        {"phase": 1, "substrate_label": "a1", "contest_cuda_score": 0.193, "archive_bytes": 178_000, "cathedral_op": "x"},
        {"phase": 1, "substrate_label": "a2", "contest_cuda_score": 0.200, "archive_bytes": 178_000, "cathedral_op": "x"},
    ])
    atoms = mod.ledger_to_atoms(ledger)
    output = tmp_path / "pareto.json"
    payload = mod.emit_meta_lagrangian_input(atoms, output, pareto_only=True)
    assert payload["pareto_only"] is True
    assert payload["n_atoms"] == 1


def test_ledger_to_atoms_real_repo() -> None:
    """Real repo's ledger should be readable (may be empty)."""
    mod = _load_tool_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    ledger = repo_root / "experiments/results/bilevel_atom_ledger.jsonl"
    atoms = mod.ledger_to_atoms(ledger)
    # Just verify it doesn't crash; ledger may be empty on fresh checkout.
    assert isinstance(atoms, list)
