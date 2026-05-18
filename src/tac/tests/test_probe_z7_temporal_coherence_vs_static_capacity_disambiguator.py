# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = (
    REPO_ROOT
    / "tools"
    / "probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("z7_disambiguator_tool", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _score(seg: float, pose: float, archive_bytes: int) -> float:
    return 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * archive_bytes / 37_545_489.0


def _eval_payload(seg: float, pose: float, archive_bytes: int) -> dict[str, object]:
    return {
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_bytes,
        "archive_sha256": "a" * 64,
        "n_samples": 600,
        "score_axis": "contest_cuda",
        "score_recomputed_from_components": _score(seg, pose, archive_bytes),
        "score_claim_valid": True,
    }


def test_z7_disambiguator_plan_is_fail_closed() -> None:
    tool = _load_tool()
    payload = tool.build_plan_payload()

    assert payload["schema"] == tool.SCHEMA
    assert payload["verdict"] == "pending_paired_exact_eval_json"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert (
        "z7_full_main_proxy_export_smoke_not_score_authority"
        in payload["blockers"]
    )
    assert (
        "z7_proxy_trained_packet_not_score_aware_or_auth_eval_validated"
        in payload["blockers"]
    )
    assert payload["decision_rule"]["same_archive_bytes_required"] is True


def test_z7_disambiguator_recurrent_win_requires_same_bytes_and_axis(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    recurrent_path = tmp_path / "recurrent.json"
    static_path = tmp_path / "static.json"
    recurrent_path.write_text(
        json.dumps(_eval_payload(0.0010, 0.0010, 200_000)),
        encoding="utf-8",
    )
    static_path.write_text(
        json.dumps(_eval_payload(0.0011, 0.0010, 200_000)),
        encoding="utf-8",
    )

    payload = tool.evaluate_exact_eval_pair(recurrent_path, static_path)

    assert payload["verdict"] == "z7_recurrent_temporal_coherence_win"
    assert payload["preferred_mode"] == "z7_recurrent_temporal_coherence"
    assert payload["deltas"]["static_minus_recurrent_score"] == pytest.approx(0.01)
    assert payload["comparability"] == {
        "same_score_axis": True,
        "same_n_samples": True,
        "same_archive_bytes": True,
    }
    assert payload["score_claim"] is False
    assert payload["blockers"] == []


def test_z7_disambiguator_blocks_unequal_archive_bytes(tmp_path: Path) -> None:
    tool = _load_tool()
    recurrent_path = tmp_path / "recurrent.json"
    static_path = tmp_path / "static.json"
    recurrent_path.write_text(
        json.dumps(_eval_payload(0.0010, 0.0010, 199_000)),
        encoding="utf-8",
    )
    static_path.write_text(
        json.dumps(_eval_payload(0.0011, 0.0010, 200_000)),
        encoding="utf-8",
    )

    payload = tool.evaluate_exact_eval_pair(recurrent_path, static_path)

    assert payload["verdict"] == "blocked_paired_exact_eval_not_comparable"
    assert "same_archive_bytes_required" in payload["blockers"]
    assert payload["comparability"]["same_archive_bytes"] is False


def test_z7_disambiguator_blocks_formula_mismatch(tmp_path: Path) -> None:
    tool = _load_tool()
    recurrent_path = tmp_path / "recurrent.json"
    static_path = tmp_path / "static.json"
    bad = _eval_payload(0.0010, 0.0010, 200_000)
    bad["score_recomputed_from_components"] = 999.0
    recurrent_path.write_text(json.dumps(bad), encoding="utf-8")
    static_path.write_text(
        json.dumps(_eval_payload(0.0011, 0.0010, 200_000)),
        encoding="utf-8",
    )

    payload = tool.evaluate_exact_eval_pair(recurrent_path, static_path)

    assert payload["verdict"] == "blocked_paired_exact_eval_not_comparable"
    assert (
        "z7_recurrent_temporal_coherence:reported_score_mismatches_recomputed_formula"
        in payload["blockers"]
    )


def test_z7_disambiguator_blocks_invalid_source_score_claim(tmp_path: Path) -> None:
    tool = _load_tool()
    recurrent_path = tmp_path / "recurrent.json"
    static_path = tmp_path / "static.json"
    invalid_recurrent = _eval_payload(0.0010, 0.0010, 200_000)
    invalid_recurrent["score_claim_valid"] = False
    recurrent_path.write_text(json.dumps(invalid_recurrent), encoding="utf-8")
    static_path.write_text(
        json.dumps(_eval_payload(0.0011, 0.0010, 200_000)),
        encoding="utf-8",
    )

    payload = tool.evaluate_exact_eval_pair(recurrent_path, static_path)

    assert payload["verdict"] == "blocked_paired_exact_eval_not_comparable"
    assert (
        "z7_recurrent_temporal_coherence:score_claim_valid_missing_or_false"
        in payload["blockers"]
    )
