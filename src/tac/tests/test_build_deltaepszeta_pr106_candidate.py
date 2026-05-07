"""Tests for ``experiments.build_deltaepszeta_pr106_candidate``."""

from __future__ import annotations

import importlib.util
import json
import pathlib
from zipfile import ZipFile

import pytest


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    tool_path = repo_root / "experiments" / "build_deltaepszeta_pr106_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "build_deltaepszeta_pr106_candidate",
        tool_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool module at {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_targets(tmp_path: pathlib.Path, *, zero_pressure: bool = False) -> pathlib.Path:
    rows = [
        {
            "idx": 0,
            "name": "tensor_a",
            "n_symbols": 80,
            "H0_bits": 6.0,
            "H1_bits": 5.0,
            "H2_bits": 2.0,
            "headroom_bits": 0.0 if zero_pressure else 4.0,
            "prize_bytes": 0.0 if zero_pressure else 40.0,
            "loss_weight_normalized": 0.0 if zero_pressure else 0.8,
            "brotli_bytes": 90,
            "ac_bytes": 80,
            "in_pr103_ac_set": True,
        },
        {
            "idx": 1,
            "name": "tensor_b",
            "n_symbols": 40,
            "H0_bits": 5.0,
            "H1_bits": 4.0,
            "H2_bits": 3.0,
            "headroom_bits": 0.0 if zero_pressure else 2.0,
            "prize_bytes": 0.0 if zero_pressure else 10.0,
            "loss_weight_normalized": 0.0 if zero_pressure else 0.2,
            "brotli_bytes": 30,
            "ac_bytes": None,
            "in_pr103_ac_set": False,
        },
    ]
    payload = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "[empirical]",
        "started_at_utc": "2026-05-07T19:00:00Z",
        "tool": "tools/build_deltaepszeta_training_targets",
        "source_shannon_analysis": "experiments/results/synthetic/per_tensor_shannon.json",
        "substrate": "synthetic-pr106-state-dict.pt",
        "n_tensors": len(rows),
        "summary": {
            "total_prize_bytes": 0 if zero_pressure else 50,
            "total_n_symbols": 120,
            "total_brotli_bytes": 120,
            "total_shannon_floor_h0_bytes": 110,
            "total_shannon_floor_h2_bytes": 60,
            "ratio_h2_over_h0": 0.545,
        },
        "per_tensor": rows,
        "top10_by_prize_bytes": [
            {
                "idx": row["idx"],
                "name": row["name"],
                "prize_bytes": row["prize_bytes"],
                "headroom_bits": row["headroom_bits"],
                "loss_weight": row["loss_weight_normalized"],
            }
            for row in rows
        ],
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_plan_is_non_scoring_and_fail_closed_without_payload(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    targets = _write_targets(tmp_path)
    renderer = tmp_path / "renderer_payload.bin"
    renderer.write_bytes(b"baseline-pr106-payload")

    plan = mod.build_plan(
        targets_json=targets,
        renderer_input=renderer,
        top_k=1,
        created_at_utc="2026-05-07T19:15:00Z",
    )

    assert plan["score_claim"] is False
    assert plan["score_fields"]["score_formula_applied"] is False
    assert plan["targets"]["loaded"] is True
    assert plan["selected_top_tensors"][0]["name"] == "tensor_a"
    assert plan["h2_pressure_plan"]["nonzero_h2_pressure"] is True
    assert (
        plan["h2_pressure_plan"]["shannon_h2_loss_api"]
        == "tac.shannon_h2_loss.shannon_h2_loss"
    )
    assert "CodecPipelineAwareTrainingCallback" in plan["h2_pressure_plan"][
        "training_byte_telemetry_api"
    ]
    assert plan["byte_change_requirement"]["satisfied"] is False
    assert plan["readiness"]["fail_closed"] is True
    assert plan["readiness"]["ready_for_byte_closed_candidate"] is False
    assert plan["readiness"]["ready_for_exact_eval_dispatch"] is False
    assert plan["readiness"]["blockers"] == [
        "missing_trained_payload",
        "missing_candidate_archive",
    ]


def test_zero_pressure_targets_raise(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    targets = _write_targets(tmp_path, zero_pressure=True)
    renderer = tmp_path / "renderer_payload.bin"
    renderer.write_bytes(b"baseline")

    with pytest.raises(ValueError, match="no positive H0-H2 prize bytes"):
        mod.build_plan(
            targets_json=targets,
            renderer_input=renderer,
            created_at_utc="2026-05-07T19:15:00Z",
        )


def test_trained_payload_must_differ_from_renderer_input(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    targets = _write_targets(tmp_path)
    renderer = tmp_path / "renderer_payload.bin"
    renderer.write_bytes(b"same-bytes")
    trained = tmp_path / "trained_payload.bin"
    trained.write_bytes(b"same-bytes")

    plan = mod.build_plan(
        targets_json=targets,
        renderer_input=renderer,
        trained_payload=trained,
        created_at_utc="2026-05-07T19:15:00Z",
    )

    assert plan["byte_change_requirement"][
        "trained_payload_differs_from_renderer_input"
    ] is False
    assert "trained_payload_matches_renderer_input" in plan["readiness"]["blockers"]
    assert "missing_candidate_archive" in plan["readiness"]["blockers"]
    assert plan["readiness"]["ready_for_byte_closed_candidate"] is False


def test_distinct_payload_and_safe_archive_pass_byte_closed_gate(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    targets = _write_targets(tmp_path)
    renderer = tmp_path / "renderer_payload.bin"
    renderer.write_bytes(b"baseline")
    trained = tmp_path / "trained_payload.bin"
    trained.write_bytes(b"trained-different")
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"trained-different")

    plan = mod.build_plan(
        targets_json=targets,
        renderer_input=renderer,
        trained_payload=trained,
        candidate_archive=archive,
        top_k=2,
        created_at_utc="2026-05-07T19:15:00Z",
    )

    assert plan["byte_change_requirement"]["satisfied"] is True
    assert plan["byte_change_requirement"][
        "candidate_archive_contains_trained_payload_sha"
    ] is True
    assert plan["readiness"]["fail_closed"] is False
    assert plan["readiness"]["ready_for_byte_closed_candidate"] is True
    assert plan["readiness"]["ready_for_exact_eval_dispatch"] is False
    zip_manifest = plan["manifest"]["candidate_archive_zip_manifest"]
    assert zip_manifest["safe_for_candidate_review"] is True
    assert zip_manifest["members"][0]["name"] == "x"


def test_archive_must_contain_trained_payload_sha(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    targets = _write_targets(tmp_path)
    renderer = tmp_path / "renderer_payload.bin"
    renderer.write_bytes(b"baseline")
    trained = tmp_path / "trained_payload.bin"
    trained.write_bytes(b"trained-different")
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"stale-baseline-member")

    plan = mod.build_plan(
        targets_json=targets,
        renderer_input=renderer,
        trained_payload=trained,
        candidate_archive=archive,
        created_at_utc="2026-05-07T19:15:00Z",
    )

    assert plan["byte_change_requirement"]["satisfied"] is False
    assert plan["byte_change_requirement"][
        "candidate_archive_contains_trained_payload_sha"
    ] is False
    assert (
        "candidate_archive_does_not_contain_trained_payload_sha"
        in plan["readiness"]["blockers"]
    )
    assert plan["readiness"]["ready_for_byte_closed_candidate"] is False


def test_plan_json_is_deterministic_with_fixed_timestamp(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    targets = _write_targets(tmp_path)
    renderer = tmp_path / "renderer_payload.bin"
    renderer.write_bytes(b"baseline-pr106-payload")
    out_a = tmp_path / "a" / "candidate_plan.json"
    out_b = tmp_path / "b" / "candidate_plan.json"

    plan_a = mod.build_plan(
        targets_json=targets,
        renderer_input=renderer,
        top_k=2,
        created_at_utc="2026-05-07T19:15:00Z",
    )
    plan_b = mod.build_plan(
        targets_json=targets,
        renderer_input=renderer,
        top_k=2,
        created_at_utc="2026-05-07T19:15:00Z",
    )
    mod.write_plan(plan_a, out_a)
    mod.write_plan(plan_b, out_b)

    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")
