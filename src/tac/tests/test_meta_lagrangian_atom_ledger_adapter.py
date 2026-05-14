# SPDX-License-Identifier: MIT
"""Tests for ``tools.meta_lagrangian_atom_ledger_adapter``."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from tac.optimization.meta_lagrangian_allocator import build_atom_ledger
from tac.optimization.meta_lagrangian_ledger_adapter import (
    adapt_artifact_to_atoms,
    search_candidates_from_atoms,
)
from tac.optimization.mps_research_signal import build_mps_research_signal_manifest


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "meta_lagrangian_atom_ledger_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "meta_lagrangian_atom_ledger_adapter",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_ledger(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _load_search_cli_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "meta_lagrangian_search_cli.py"
    spec = importlib.util.spec_from_file_location(
        "meta_lagrangian_search_cli",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_canonical_adapter_ingests_bilevel_jsonl_into_field_atoms(
    tmp_path: pathlib.Path,
) -> None:
    ledger_path = tmp_path / "bilevel_atom_ledger.jsonl"
    _write_ledger(
        ledger_path,
        [
            {
                "phase": 2,
                "substrate_label": "synthetic 600-frame poses",
                "substrate_path": "src/tac/codec_pipeline_raft_pose.py",
                "contest_cuda_score": None,
                "evidence_grade": "[CPU-prep]",
                "archive_bytes": 7262,
                "archive_sha256": None,
                "cathedral_op": "Op_RAFTPoseStream",
                "dispatch_blockers": ["requires_raft_flow_extraction"],
                "notes": "CPU-only prep row",
            }
        ],
    )

    result = adapt_artifact_to_atoms(ledger_path, repo_root=tmp_path)
    atoms = list(result.atoms)

    assert result.source_format == "bilevel_atom_ledger_jsonl"
    assert result.score_claim is False
    assert result.ready_for_exact_eval_dispatch is False
    assert len(atoms) == 1
    atom = atoms[0]
    assert atom["evidence_grade"] == "[CPU-prep]"
    assert atom["archive_bytes"] == 7262
    assert atom["source_artifact_bytes"] == 7262
    assert atom["score_claim"] is False
    assert atom["byte_delta"] == 0
    assert "missing_byte_delta_vs_anchor" in atom["dispatch_blockers"]
    assert "requires_raft_flow_extraction" in atom["dispatch_blockers"]

    field_ledger = build_atom_ledger(
        atoms,
        base_pose_dist=0.01,
        source=str(ledger_path),
    )
    row = field_ledger["rows"][0]
    assert row["score_claim"] is False
    assert row["evidence_grade"] == "[CPU-prep]"
    assert row["archive_bytes"] == 7262
    assert row["source_artifact_bytes"] == 7262
    assert "requires_raft_flow_extraction" in row["source_dispatch_blockers"]
    assert "requires_raft_flow_extraction" in row["dispatch_blockers"]
    assert "requires_exact_cuda_auth_eval" in row["exact_dispatch_blockers"]["blockers"]


def test_candidate_packet_manifest_adapts_selected_target_atom_template(
    tmp_path: pathlib.Path,
) -> None:
    packet_path = tmp_path / "candidate_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "score_claim": False,
                "score_evidence_grade": "invalid",
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": ["requires_lane_dispatch_claim_before_gpu"],
                "missing_artifacts": ["candidate_archive_manifest_with_member_sha256s"],
                "audit_summary": {"total_actual_bytes": 221381},
                "selected_target": {
                    "row": {
                        "label": "public_pr106:hdc2",
                        "actual_bytes": 221381,
                        "target_bytes": 40840,
                        "dispatch_blockers": ["requires_roundtrip_decode_validation"],
                        "meta_lagrangian_atom_export": {
                            "export_blockers": [
                                "planning_target_not_byte_closed_candidate",
                                "missing_archive_manifest_path",
                            ],
                            "atom_template": {
                                "atom_id": "public_pr106_hdc2:known_model_overhead",
                                "family": "hnerv_known_model_overhead",
                                "family_group": "hnerv_rate_equivalent_recode",
                                "pareto_scope": "hnerv_rate_equivalent_recode:public_pr106_hdc2",
                                "byte_delta": -40840,
                                "confidence": 0.0,
                                "evidence_grade": (
                                    "invalid_planning_target_until_byte_equivalent_candidate"
                                ),
                                "expected_seg_dist_delta": 0.0,
                                "expected_pose_dist_delta": 0.0,
                                "raw_equal": False,
                                "score_claim": False,
                                "target_bytes": 40840,
                                "interaction_assumptions": [
                                    "rate_only_decoded_output_equivalence_required"
                                ],
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = adapt_artifact_to_atoms(packet_path, repo_root=tmp_path)
    atom = result.atoms[0]

    assert result.source_format == "candidate_packet_manifest_json"
    assert atom["byte_delta"] == -40840
    assert atom["source_artifact_bytes"] == 221381
    assert atom["target_bytes"] == 40840
    assert atom["score_claim"] is False
    assert "requires_roundtrip_decode_validation" in atom["dispatch_blockers"]
    assert "planning_target_not_byte_closed_candidate" in atom["dispatch_blockers"]
    assert search_candidates_from_atoms(result.atoms) == []

    field_ledger = build_atom_ledger(
        result.atoms,
        base_pose_dist=0.01,
        source=str(packet_path),
    )
    row = field_ledger["rows"][0]
    assert row["score_claim"] is False
    assert row["source_artifact_bytes"] == 221381
    assert row["evidence_grade"] == (
        "invalid_planning_target_until_byte_equivalent_candidate"
    )
    assert "planning_target_not_byte_closed_candidate" in row["dispatch_blockers"]


def test_search_cli_reuses_candidates_json_for_jsonl_adapter(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_search_cli_module()
    ledger_path = tmp_path / "bilevel_atom_ledger.jsonl"
    _write_ledger(
        ledger_path,
        [
            {
                "phase": 8,
                "candidate_id": "jsonl_projectable",
                "evidence_grade": "[CPU-prep]",
                "archive_bytes": 185000,
                "archive_sha256": "a" * 64,
                "byte_delta": -10,
                "rel_err_pct": 0.25,
                "n_layers": 13,
                "cathedral_op": "Op_Test",
            }
        ],
    )

    candidates, adapter_report = mod._load_candidates_from_path(ledger_path)

    assert adapter_report["source_format"] == "bilevel_atom_ledger_jsonl"
    assert adapter_report["score_claim"] is False
    assert adapter_report["ready_for_exact_eval_dispatch"] is False
    assert adapter_report["atom_count"] == 1
    assert candidates == [
        {
            "candidate_id": "jsonl_projectable",
            "archive_bytes": 185000,
            "rel_err_pct": 0.25,
            "n_layers": 13,
            "lane_class": "bilevel_atom_ledger",
        }
    ]


def test_mps_research_signal_manifest_adapts_but_never_projects_search_candidates(
    tmp_path: pathlib.Path,
) -> None:
    manifest_path = tmp_path / "mps_manifest.json"
    manifest = build_mps_research_signal_manifest(
        [
            {
                "family": "arch_shrink",
                "variant_id": "epoch_010",
                "device": "mps",
                "archive_bytes": 120_000,
                "proxy_loss": 0.2,
                "rel_err_pct": 0.1,
                "n_layers": 4,
            }
        ],
        source="fixture",
        run_id="fixture",
        anchor_d_seg=0.0007,
        anchor_d_pose=0.00003,
        anchor_archive_bytes=180_000,
    )
    # Simulate a malformed producer trying to make the proxy atom projectable.
    manifest["meta_lagrangian_atoms"][0].update({
        "archive_bytes": 120_000,
        "rel_err_pct": 0.1,
        "n_layers": 4,
        "rankable": True,
        "ready_for_exact_eval_dispatch": True,
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = adapt_artifact_to_atoms(manifest_path, repo_root=tmp_path)
    atom = result.atoms[0]

    assert result.source_format == "mps_research_signal_manifest_json"
    assert atom["proxy_row"] is True
    assert atom["rankable"] is False
    assert atom["ready_for_exact_eval_dispatch"] is False
    assert "mps_research_signal_not_search_candidate" in atom["dispatch_blockers"]
    assert search_candidates_from_atoms(result.atoms) == []


def test_search_candidates_from_atoms_skips_proxy_and_explicit_nonrankable_rows() -> None:
    atoms = [
        {
            "atom_id": "proxy",
            "archive_bytes": 1,
            "rel_err_pct": 0.1,
            "n_layers": 1,
            "proxy_row": True,
        },
        {
            "atom_id": "nonrankable",
            "archive_bytes": 2,
            "rel_err_pct": 0.2,
            "n_layers": 2,
            "rankable": False,
        },
        {
            "atom_id": "planning",
            "archive_bytes": 3,
            "rel_err_pct": 0.3,
            "n_layers": 3,
            "lane_class": "safe_planning",
        },
    ]

    assert search_candidates_from_atoms(atoms) == [
        {
            "candidate_id": "planning",
            "archive_bytes": 3,
            "rel_err_pct": 0.3,
            "n_layers": 3,
            "lane_class": "safe_planning",
        }
    ]


def test_read_ledger_handles_missing_file(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()

    assert mod.read_atom_ledger(tmp_path / "nope.jsonl") == []


def test_read_ledger_fails_closed_on_corrupt_jsonl(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"phase": 1}\n{not valid json\n')

    with pytest.raises(mod.LedgerFormatError, match="invalid JSONL record"):
        mod.read_atom_ledger(path)


def test_read_ledger_can_explicitly_skip_corrupt_lines(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"phase": 1}\n{not valid json\n{"phase": 2}\n')

    records = mod.read_atom_ledger(path, strict=False)

    assert [record["phase"] for record in records] == [1, 2]


def test_cpu_prep_row_preserves_context_but_is_fail_closed() -> None:
    mod = _load_tool_module()
    record = {
        "phase": 1,
        "substrate_label": "PR101 (gold)",
        "substrate_score_anchor": 0.193,
        "contest_cuda_score": None,
        "archive_bytes": None,
        "archive_sha256": None,
        "evidence_grade": "[CPU-prep]",
        "cathedral_op": "Op1+Op2",
        "notes": "awaiting GPU",
    }

    atom = mod.record_to_atom(record, idx=0)

    assert atom.phase == 1
    assert atom.substrate_label == "PR101 (gold)"
    assert atom.evidence_grade == "[CPU-prep]"
    assert atom.score is None
    assert atom.score_delta_vs_substrate_anchor is None
    assert atom.byte_delta_vs_substrate_anchor is None
    assert atom.archive_custody["has_archive_bytes"] is False
    assert atom.fail_closed is True
    assert atom.ready_for_meta_lagrangian_search is False
    assert "missing_archive_bytes" in atom.blockers


def test_complete_row_emits_strict_search_candidate(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive-bytes")
    digest = mod._sha256_file(archive)
    record = {
        "phase": 2,
        "candidate_id": "phase2_ready",
        "substrate_label": "PR101 (gold)",
        "substrate_score_anchor": 0.200,
        "substrate_archive_bytes": 120,
        "contest_cuda_score": 0.190,
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest,
        "evidence_grade": "[contest-CUDA]",
        "cathedral_op": "Op1+Op2",
        "rel_err_pct": 0.25,
        "n_layers": 13,
        "lane_class": "apogee_intN",
    }

    atom = mod.record_to_atom(record, idx=0, repo_root=tmp_path)
    candidate = atom.to_search_candidate()

    assert atom.fail_closed is False
    assert atom.ready_for_meta_lagrangian_search is True
    assert atom.ready_for_exact_eval_dispatch is False
    assert atom.score_claim is False
    assert atom.score_delta_vs_substrate_anchor == pytest.approx(-0.010)
    assert atom.byte_delta_vs_substrate_anchor == archive.stat().st_size - 120
    assert set(candidate) == {
        "candidate_id",
        "archive_bytes",
        "rel_err_pct",
        "n_layers",
        "lane_class",
        "archive_path",
    }
    assert candidate["candidate_id"] == "phase2_ready"
    assert candidate["archive_bytes"] == archive.stat().st_size
    assert candidate["rel_err_pct"] == 0.25
    assert candidate["n_layers"] == 13


def test_archive_custody_mismatch_blocks_candidate(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive-bytes")
    record = {
        "phase": 2,
        "substrate_label": "PR101",
        "contest_cuda_score": 0.190,
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size + 1,
        "archive_sha256": "a" * 64,
        "evidence_grade": "[contest-CUDA]",
        "rel_err_pct": 0.25,
        "n_layers": 13,
    }

    atom = mod.record_to_atom(record, idx=0, repo_root=tmp_path)

    assert atom.fail_closed is True
    assert atom.ready_for_meta_lagrangian_search is False
    assert "archive_custody_bytes_mismatch" in atom.blockers
    assert "archive_custody_sha256_mismatch" in atom.blockers


def test_exact_evidence_grade_without_archive_sha_is_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive-bytes")
    record = {
        "phase": 2,
        "substrate_label": "PR101",
        "contest_cuda_score": 0.190,
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": None,
        "evidence_grade": "[contest-CUDA]",
        "rel_err_pct": 0.25,
        "n_layers": 13,
    }

    atom = mod.record_to_atom(record, idx=0, repo_root=tmp_path)

    assert atom.fail_closed is True
    assert atom.ready_for_meta_lagrangian_search is False
    assert "missing_archive_sha256_for_evidence_grade" in atom.blockers
    assert "missing_archive_custody_sha256" in atom.blockers


def test_emit_writes_rows_and_strict_candidates(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"ready")
    ready_record = {
        "phase": 2,
        "candidate_id": "ready",
        "substrate_label": "PR101",
        "contest_cuda_score": 0.190,
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": mod._sha256_file(archive),
        "evidence_grade": "[contest-CUDA]",
        "rel_err_pct": 0.25,
        "n_layers": 13,
        "lane_class": "apogee_intN",
    }
    blocked_record = {
        "phase": 1,
        "substrate_label": "PR101",
        "contest_cuda_score": None,
        "archive_bytes": None,
        "archive_sha256": None,
        "evidence_grade": "[CPU-prep]",
    }
    atoms = [
        mod.record_to_atom(ready_record, idx=0, repo_root=tmp_path),
        mod.record_to_atom(blocked_record, idx=1, repo_root=tmp_path),
    ]
    rows_output = tmp_path / "rows.json"
    candidates_output = tmp_path / "candidates.json"

    payload = mod.emit_meta_lagrangian_input(
        atoms,
        rows_output,
        candidates_output_path=candidates_output,
    )
    candidates = json.loads(candidates_output.read_text())
    rows_payload = json.loads(rows_output.read_text())

    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["n_rows"] == 2
    assert payload["ready_candidate_count"] == 1
    assert payload["fail_closed_count"] == 1
    assert candidates == payload["candidates_for_meta_lagrangian_search_cli"]
    assert candidates == [
        {
            "candidate_id": "ready",
            "archive_bytes": archive.stat().st_size,
            "rel_err_pct": 0.25,
            "n_layers": 13,
            "lane_class": "apogee_intN",
            "archive_path": str(archive),
        }
    ]
    assert rows_payload["rows"][1]["fail_closed"] is True


def test_pareto_filter_removes_strictly_dominated() -> None:
    mod = _load_tool_module()
    atoms = [
        mod.MetaLagrangianAtom("a1", "op", "s", 100, 0.20, "[contest-CUDA]", None, ""),
        mod.MetaLagrangianAtom("a2", "op", "s", 100, 0.21, "[contest-CUDA]", None, ""),
        mod.MetaLagrangianAtom("a3", "op", "s", 80, 0.22, "[contest-CUDA]", None, ""),
        mod.MetaLagrangianAtom("a4", "op", "s", 50, 0.30, "[contest-CUDA]", None, ""),
    ]

    nd_ids = {atom.atom_id for atom in mod.filter_pareto_non_dominated(atoms)}

    assert nd_ids == {"a1", "a3", "a4"}


def test_ledger_to_atoms_real_repo() -> None:
    mod = _load_tool_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    ledger = repo_root / "experiments/results/bilevel_atom_ledger.jsonl"

    atoms = mod.ledger_to_atoms(ledger, repo_root=repo_root)

    assert isinstance(atoms, list)
