# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.atom.types import AtomKind, ResolutionPath
from tac.optimization.optimizer_guided_candidate_generation import (
    generate_candidate_queue,
    load_profile,
)
from tac.optimization.optimizer_signal_atoms import (
    OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA,
    OptimizerSignalAtomError,
    build_atoms_from_optimizer_signal_source,
    build_optimizer_signal_atom_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "materialize_optimizer_signal_atoms.py"


def _pr95_queue() -> dict[str, object]:
    return generate_candidate_queue(
        profile=load_profile("pr95_hnerv_muon_training_smoke"),
        optimizer="cmaes",
        max_candidates=8,
        top_k=3,
        seed=20260522,
    )


def test_optimizer_guided_queue_materializes_proxy_only_atoms() -> None:
    queue = _pr95_queue()

    atoms = build_atoms_from_optimizer_signal_source(
        queue,
        source_path="experiments/results/unit/optimizer_queue.json",
    )

    assert len(atoms) == 3
    atom = atoms[0]
    assert atom.kind == AtomKind.META_LAGRANGIAN
    assert atom.resolution_path == ResolutionPath.LEARNED
    assert atom.predicted_impact_delta_s_lower == 0.0
    assert atom.predicted_impact_delta_s_upper == 0.0
    assert set(atom.wired_hooks) == {
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
        "cathedral_autopilot_dispatch",
        "continual_learning_posterior",
        "probe_disambiguator",
    }
    assert atom.provenance["evidence_grade"] == "predicted"
    assert atom.provenance["promotion_eligible"] is False
    assert atom.provenance["score_claim_valid"] is False
    assert atom.metadata["candidate_family"] == "pr95_hnerv_muon_optimizer_recipe_smoke"
    assert atom.metadata["representation_family"] == "hnerv"
    assert atom.metadata["substrate_family"] == "nerv_family"
    assert atom.metadata["score_impact_band_status"] == (
        "neutral_until_exact_auth_or_calibrated_posterior"
    )
    assert atom.metadata["false_authority"]["score_claim"] is False
    assert atom.metadata["false_authority"]["ready_for_exact_eval_dispatch"] is False
    assert atom.metadata["parameter_group_fingerprint_sha256"]
    assert atom.metadata["proxy_objective"] is not None
    assert atom.metadata["rank_score_field"] == "proxy_objective_not_score"


def test_optimizer_signal_atom_ledger_includes_meta_lagrangian_projection() -> None:
    queue = _pr95_queue()

    ledger = build_optimizer_signal_atom_ledger(queue, source_path="queue.json", max_atoms=2)

    assert ledger["schema"] == OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA
    assert ledger["atom_count"] == 2
    assert ledger["score_claim"] is False
    assert ledger["promotion_eligible"] is False
    assert ledger["ready_for_exact_eval_dispatch"] is False
    assert len(ledger["atoms"]) == 2
    assert len(ledger["meta_lagrangian_atoms"]) == 2
    first_projection = ledger["meta_lagrangian_atoms"][0]
    assert first_projection["family"] == "atom_kind:meta_lagrangian"
    assert first_projection["family_group"] == "resolution_path:learned"
    assert first_projection["expected_score_delta"] == 0.0


def test_truthy_proxy_authority_is_rejected() -> None:
    queue = _pr95_queue()
    top_k = queue["top_k"]
    assert isinstance(top_k, list)
    row = dict(top_k[0])
    row["score_claim"] = True
    queue["top_k"] = [row]

    with pytest.raises(OptimizerSignalAtomError, match="proxy contract violations"):
        build_atoms_from_optimizer_signal_source(queue)


def test_materialize_optimizer_signal_atoms_cli(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    json_out = tmp_path / "atoms.json"
    jsonl_out = tmp_path / "atoms.jsonl"
    queue_path.write_text(
        json.dumps(_pr95_queue(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--candidate-queue",
            str(queue_path),
            "--json-out",
            str(json_out),
            "--jsonl-out",
            str(jsonl_out),
            "--max-atoms",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "count=2" in proc.stdout
    assert "score_claim=false" in proc.stdout
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == OPTIMIZER_SIGNAL_ATOM_LEDGER_SCHEMA
    assert payload["atom_count"] == 2
    lines = jsonl_out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "meta_lagrangian"
    assert first["resolution_path"] == "learned"
