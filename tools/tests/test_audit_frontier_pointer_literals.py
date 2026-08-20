from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_frontier_pointer_literals as audit


def _pointer(path: Path, score: float = 0.172) -> Path:
    fetched = "2026-07-25T00:00:00+00:00"
    public = {
        "rank": 1,
        "score": score,
        "name": "fixture leader",
        "pr_number": 130,
        "pr_url": "https://example.test/130",
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "canonical_frontier_pointer_v1_20260519",
                "our_local_frontier_contest_cpu": None,
                "our_local_frontier_contest_cuda": None,
                "submitted_pr_number_for_current_frontier": None,
                "upstream_leaderboard_snapshot": {
                    "source": "official_leaderboard",
                    "fetched_at_utc": fetched,
                    "fetch_status": "ok",
                    "entries": [public],
                    "best_entry": public,
                    "entry_count": 1,
                    "score_precision": "official_display",
                },
                "upstream_leaderboard_snapshot_at_utc": fetched,
                "last_refreshed_utc": fetched,
                "auto_update_on_dispatch_completion": True,
                "pointer_refresh_command": "fixture",
                "refresh_provenance": {"kind": "test"},
                "effective_frontier": {
                    "score": score,
                    "axis": "official_leaderboard",
                    "source": "upstream_official_leaderboard",
                    "source_kind": "external_public_leaderboard_target",
                    "leaderboard_rank": 1,
                    "submission_name": "fixture leader",
                    "pr_number": 130,
                    "pr_url": "https://example.test/130",
                    "snapshot_at_utc": fetched,
                    "evidence_grade": "[official-leaderboard display]",
                    "score_precision": "official_display",
                    "custody": "external target only; no local archive authority implied",
                    "selection_rule": (
                        "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, "
                        "upstream_official_leaderboard.best_entry)"
                    ),
                    "role": "competitive_score_to_beat",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_audit_separates_executable_assignment_and_text(tmp_path: Path) -> None:
    root = tmp_path / "code"
    root.mkdir()
    source = root / "probe.py"
    source.write_text(
        'FRONTIER_SCORE = 0.1910828242\nNOTE = "pointer 0.19110 unmoved"\nOTHER = 0.2\n',
        encoding="utf-8",
    )
    result = audit.build_audit(roots=[root], pointer_path=_pointer(tmp_path / "pointer.json"))
    assert result["executable_pointer_literal_count"] == 1
    assert result["retired_pointer_text_count"] == 2
    assert result["executable_pointer_literals"][0]["name"] == "FRONTIER_SCORE"


def test_strict_path_fails_only_when_selected_path_has_findings(tmp_path: Path) -> None:
    root = tmp_path / "code"
    root.mkdir()
    stale = root / "stale.py"
    clean = root / "clean.py"
    stale.write_text("POINTER = 0.191\n", encoding="utf-8")
    clean.write_text("def load_pointer():\n    return object()\n", encoding="utf-8")
    result = audit.build_audit(
        roots=[root],
        pointer_path=_pointer(tmp_path / "pointer.json"),
        strict_paths=[clean],
    )
    assert result["strict_violation_count"] == 0
    result = audit.build_audit(
        roots=[root],
        pointer_path=tmp_path / "pointer.json",
        strict_paths=[stale],
    )
    assert result["strict_violation_count"] == 1


def test_current_v10_measurement_path_is_pointer_literal_clean() -> None:
    target = audit.REPO / "tools/measure_v10_free_predictor_floor.py"
    result = audit.build_audit(
        roots=[target],
        pointer_path=audit.DEFAULT_POINTER,
        strict_paths=[target],
    )
    assert result["strict_violation_count"] == 0


def test_screw_reach_gate_consumes_dynamic_pointer_without_competitive_literal() -> None:
    target = audit.REPO / "tools/measure_screw_reach_through_R.py"
    result = audit.build_audit(
        roots=[target],
        pointer_path=audit.DEFAULT_POINTER,
        strict_paths=[target],
    )
    assert result["competitive_pointer_literal_count"] == 0
    assert result["strict_violation_count"] == 0


def test_taskspace_measurement_gates_consume_dynamic_pointer_without_literals() -> None:
    targets = [
        audit.REPO / "experiments/feedy_byteclosed_exact_row_probe.py",
        audit.REPO / "experiments/measure_symbolic_topological_partition_mdl.py",
    ]
    result = audit.build_audit(
        roots=targets,
        pointer_path=audit.DEFAULT_POINTER,
        strict_paths=targets,
    )
    assert result["competitive_pointer_literal_count"] == 0
    assert result["strict_violation_count"] == 0


def test_audit_catches_attribute_mapping_defaults_and_comparison_aliases(tmp_path: Path) -> None:
    root = tmp_path / "code"
    root.mkdir()
    source = root / "forms.py"
    source.write_text(
        "\n".join(
            (
                "settings.frontier_score = 0.181",
                'settings["target_score"] = 0.182',
                'CONFIG = {"target_score": 0.183}',
                'FRONTIER = {"score": 0.184}',
                "def beats(value, target_score=0.185, *, score_to_beat=0.186):",
                "    return value < target_score",
                "if candidate_score < 0.187:",
                "    pass",
                "UNRELATED_TIMEOUT = 0.188",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit.build_audit(
        roots=[root],
        pointer_path=_pointer(tmp_path / "pointer.json"),
        strict_paths=[source],
    )
    findings = result["executable_pointer_literals"]
    assert {(row["kind"], row["value"]) for row in findings} == {
        ("assignment", 0.181),
        ("assignment", 0.182),
        ("mapping_value", 0.183),
        ("mapping_value", 0.184),
        ("function_default", 0.185),
        ("keyword_default", 0.186),
        ("comparison_threshold", 0.187),
    }
    assert result["strict_violation_count"] == 1


def test_pointer_loader_refuses_wrong_schema_and_fabricated_effective_winner(tmp_path: Path) -> None:
    path = _pointer(tmp_path / "pointer.json")
    payload = json.loads(path.read_text())
    payload["schema_version"] = "lookalike-v1"
    path.write_text(json.dumps(payload))
    with pytest.raises(audit.PointerLiteralAuditError, match="schema version"):
        audit.build_audit(roots=[tmp_path], pointer_path=path)

    path = _pointer(path)
    payload = json.loads(path.read_text())
    payload["effective_frontier"]["score"] = 0.171
    path.write_text(json.dumps(payload))
    with pytest.raises(audit.PointerLiteralAuditError, match="fabricated score"):
        audit.build_audit(roots=[tmp_path], pointer_path=path)

    path = _pointer(path)
    payload = json.loads(path.read_text())
    payload["effective_frontier"]["submission_name"] = "fabricated winner"
    path.write_text(json.dumps(payload))
    with pytest.raises(audit.PointerLiteralAuditError, match="fabricated submission_name"):
        audit.build_audit(roots=[tmp_path], pointer_path=path)

    path = _pointer(path)
    payload = json.loads(path.read_text())
    payload["effective_frontier"]["archive_sha256"] = "f" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(audit.PointerLiteralAuditError, match="fabricated or missing fields"):
        audit.build_audit(roots=[tmp_path], pointer_path=path)


def test_strict_gate_ignores_noncompetitive_frontier_symbols(tmp_path: Path) -> None:
    root = tmp_path / "code"
    root.mkdir()
    source = root / "topology.py"
    source.write_text("MAX_FRONTIER_STATES = 4096\nFRONTIER_DILATION = 0.5\n", encoding="utf-8")
    result = audit.build_audit(
        roots=[root],
        pointer_path=_pointer(tmp_path / "pointer.json"),
        strict_paths=[source],
    )
    assert result["executable_pointer_literal_count"] == 2
    assert result["competitive_pointer_literal_count"] == 0
    assert result["strict_violation_count"] == 0
    assert len(result["receipt_sha256"]) == 64
