"""Tests for ``tools.codec_op_entropy_floor_to_ledger``."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

from tac.optimization.meta_lagrangian_allocator import build_atom_ledger


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "codec_op_entropy_floor_to_ledger.py"
    spec = importlib.util.spec_from_file_location(
        "codec_op_entropy_floor_to_ledger",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_provable_floor_report_emits_planning_only_atoms(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    report_path = tmp_path / "pr101_provable_optimal_floor.json"
    _write_json(
        report_path,
        {
            "schema": "pr101_compression_floor_ladder.v2",
            "tool": "tools/pr101_provable_optimal_floor.py",
            "input_state_dict": "state.pt",
            "input_state_dict_sha256": "b" * 64,
            "evidence_grade": "derivation",
            "evidence_semantics": "cpu_model_class_entropy_floor_ladder",
            "score_claim": False,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": ["no_actual_context_coder_bitstream"],
            "empirical_encoders": [
                {
                    "name": "brotli_optuna_optimum",
                    "bytes_archive": 178_144,
                    "bytes_payload": 162_050,
                }
            ],
            "provable_floors": [
                {
                    "name": "markov2_per_tensor",
                    "bits": 784_096.36,
                    "bytes_archive": 114_107,
                    "bytes_payload": 98_013,
                    "model_table_overhead_included": False,
                },
                {
                    "name": "iid_joint_pooled",
                    "bits": 1_492_670.43,
                    "bytes_archive": 202_678,
                    "bytes_payload": 186_584,
                    "model_table_overhead_included": False,
                },
            ],
        },
    )

    rows = mod.build_atom_rows([report_path], repo_root=tmp_path)

    assert len(rows) == 2
    markov2 = next(row for row in rows if row["entropy_floor_name"] == "markov2_per_tensor")
    assert markov2["family"] == "codec_op_provable_entropy_floor"
    assert markov2["pareto_scope"] == "pr101_decoder_entropy_floor:bbbbbbbbbbbb"
    assert markov2["byte_delta"] == -64_037
    assert markov2["confidence"] == 0.20
    assert markov2["evidence_grade"] == "derivation_planning"
    assert markov2["planning_only"] is True
    assert markov2["proxy_row"] is True
    assert markov2["score_claim"] is False
    assert markov2["promotion_eligible"] is False
    assert markov2["promotable"] is False
    assert markov2["dispatchable"] is False
    assert markov2["ready_for_exact_eval_dispatch"] is False
    assert markov2["readiness"]["promotion_eligible"] is False
    assert markov2["readiness"]["ready_for_exact_eval_dispatch"] is False
    assert "model_table_overhead_not_charged" in markov2["dispatch_blockers"]
    assert "requires_exact_cuda_auth_eval" in markov2["dispatch_blockers"]
    assert "not_promotion_eligible" in markov2["promotion_blockers"]

    field_ledger = build_atom_ledger(
        rows,
        base_pose_dist=0.000003389640351909648,
        source=str(report_path),
    )
    field_row = next(
        row for row in field_ledger["rows"] if row["atom_id"] == markov2["atom_id"]
    )
    assert field_ledger["ready_for_exact_eval_dispatch"] is False
    assert field_row["proxy_row"] is True
    assert field_row["pareto_eligible"] is False
    assert field_row["dispatchable"] is False
    assert field_row["field_selection_ready_for_exact_eval_dispatch"] is False
    assert field_row["exact_dispatch_blockers"]["ready_for_exact_eval_dispatch"] is False
    assert "proxy_row_not_dispatchable" in field_row["dispatch_blockers"]
    assert "requires_byte_closed_archive" in field_row["dispatch_blockers"]


def test_context_transform_report_refuses_source_readiness_flags(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    report_path = tmp_path / "pr101_context_transform_floor_probe.json"
    _write_json(
        report_path,
        {
            "schema": "pr101_context_transform_floor_probe.v1",
            "tool": "tools/pr101_context_transform_floor_probe.py",
            "input_state_dict": "state.pt",
            "input_state_dict_sha256": "c" * 64,
            "evidence_grade": "derivation",
            "evidence_semantics": "cpu_invertible_transform_entropy_floor_probe",
            "score_claim": True,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "ready_for_exact_eval_dispatch": True,
            "dispatch_blockers": ["entropy_floor_probe_only"],
            "comparison_brotli_optuna_archive_bytes": 178_144,
            "transforms": [
                {
                    "transform": "delta_mod255",
                    "invertible_fixed_transform": True,
                    "metadata_bytes_charged": 0,
                    "n_streams": 28,
                    "n_symbols_total": 228_958,
                    "iid_archive_bytes": 215_378,
                    "iid_payload_bytes": 199_284,
                    "markov1_archive_bytes": 186_595,
                    "markov1_payload_bytes": 170_501,
                    "markov2_archive_bytes": 71_029,
                    "markov2_payload_bytes": 54_935,
                    "delta_markov2_archive_vs_brotli_optuna": -107_115,
                }
            ],
        },
    )

    rows = mod.build_atom_rows([report_path], repo_root=tmp_path)

    assert len(rows) == 3
    markov2 = next(row for row in rows if row["entropy_model"] == "markov2")
    assert markov2["family"] == "codec_op_context_transform_entropy_floor"
    assert markov2["pareto_scope"] == "pr101_context_transform_entropy_floor:cccccccccccc"
    assert markov2["byte_delta"] == -107_115
    assert markov2["context_transform"] == "delta_mod255"
    assert markov2["ready_for_exact_eval_dispatch"] is False
    assert markov2["dispatchable"] is False
    assert markov2["score_claim"] is False
    assert markov2["planning_only"] is True
    assert markov2["promotion_eligible"] is False
    assert "source_ready_for_exact_eval_dispatch_true_refused_by_adapter" in markov2["dispatch_blockers"]
    assert "source_score_claim_true_refused_by_adapter" in markov2["dispatch_blockers"]
    assert "source_charged_bits_changed_true_refused_by_adapter" in markov2["dispatch_blockers"]
    assert "source_score_affecting_payload_changed_true_refused_by_adapter" in markov2["dispatch_blockers"]


def test_cli_writes_jsonl_atoms_json_and_summary(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    report_path = tmp_path / "floor.json"
    output_path = tmp_path / "atoms.jsonl"
    atoms_json_path = tmp_path / "atoms.json"
    summary_path = tmp_path / "summary.json"
    ledger_md_path = tmp_path / "ledger.md"
    _write_json(
        report_path,
        {
            "schema": "pr101_compression_floor_ladder.v2",
            "evidence_grade": "derivation",
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "charged_bits_changed": False,
            "score_affecting_payload_changed": False,
            "comparison_brotli_optuna_archive_bytes": 178_144,
            "input_state_dict_sha256": "d" * 64,
            "provable_floors": [
                {
                    "name": "iid_per_tensor",
                    "bits": 1.0,
                    "bytes_archive": 175_916,
                    "bytes_payload": 159_822,
                    "model_table_overhead_included": False,
                }
            ],
        },
    )

    rc = mod.main(
        [
            "--input",
            str(report_path),
            "--output",
            str(output_path),
            "--atoms-json-output",
            str(atoms_json_path),
            "--summary-output",
            str(summary_path),
            "--ledger-md-output",
            str(ledger_md_path),
        ]
    )

    assert rc == 0
    jsonl_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    atoms_json_rows = json.loads(atoms_json_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert jsonl_rows == atoms_json_rows
    assert jsonl_rows[0]["ready_for_exact_eval_dispatch"] is False
    assert jsonl_rows[0]["promotion_eligible"] is False
    assert jsonl_rows[0]["planning_only"] is True
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["planning_only"] is True
    assert summary["promotion_eligible"] is False
    assert summary["negative_byte_delta_atom_count"] == 1
    assert summary["policy"]["fail_closed"] is True
    assert summary["policy"]["promotion_eligible_atom_count"] == 0
    assert summary["best_negative_byte_deltas"][0]["byte_delta"] == -2228
    assert "promotion-eligible rows: `0`" in ledger_md_path.read_text(encoding="utf-8")


def test_current_pr101_reports_emit_fail_closed_repo_local_policy_rows() -> None:
    mod = _load_tool_module()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    report_paths = [
        repo_root / "reports" / "pr101_provable_optimal_floor.json",
        repo_root / "reports" / "pr101_context_transform_floor_probe.json",
    ]

    rows = mod.build_atom_rows(report_paths, repo_root=repo_root)
    summary = mod.build_summary(
        rows,
        input_paths=report_paths,
        output_path=repo_root
        / "experiments"
        / "results"
        / "codec_op_entropy_floor_ledger_20260507_worker_g"
        / "codec_op_entropy_floor_atoms.jsonl",
        repo_root=repo_root,
    )

    assert len(rows) == 26
    assert summary["atom_count"] == 26
    assert summary["negative_byte_delta_atom_count"] == 15
    assert [report["atom_count"] for report in summary["input_reports"]] == [5, 21]
    assert [report["negative_byte_delta_atom_count"] for report in summary["input_reports"]] == [4, 11]
    assert summary["policy"]["planning_only_atom_count"] == 26
    assert summary["policy"]["proxy_row_count"] == 26
    assert summary["policy"]["score_claim_atom_count"] == 0
    assert summary["policy"]["dispatchable_atom_count"] == 0
    assert summary["policy"]["ready_for_exact_eval_dispatch_atom_count"] == 0
    assert summary["policy"]["promotion_eligible_atom_count"] == 0
    assert summary["policy"]["fail_closed"] is True
    assert all(row["planning_only"] is True for row in rows)
    assert all(row["proxy_row"] is True for row in rows)
    assert all(row["score_claim"] is False for row in rows)
    assert all(row["dispatchable"] is False for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)
    assert all(row["promotion_eligible"] is False for row in rows)
    assert all(row["promotable"] is False for row in rows)

    best = summary["best_negative_byte_deltas"][0]
    assert best["atom_id"] == "codec_op_entropy_floor:pr101_context_transform_floor_probe:delta_mod255:markov2"
    assert best["byte_delta"] == -107_115
    assert best["target_archive_bytes_estimate"] == 71_029
