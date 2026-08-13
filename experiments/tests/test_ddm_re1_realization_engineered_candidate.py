from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments import ddm_re1_realization_engineered_candidate as re1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _row(
    proposal_id: str,
    flips: int,
    pose: float,
    pair: int = 0,
    *,
    eligible: bool = True,
    ordinal: int | None = None,
) -> dict[str, object]:
    resolved_ordinal = int(proposal_id[4:8]) if ordinal is None and proposal_id.startswith("ec1_") else (ordinal or 0)
    return {
        "proposal_id": proposal_id,
        "ordinal": resolved_ordinal,
        "pair": pair,
        "delta_flips_candidate_minus_base": flips,
        "delta_d_pose_global_n600": pose,
        "site_count": 1,
        "downstream_selection_eligible": eligible,
    }


def test_singleton_projection_uses_nonlinear_pose_term() -> None:
    row = _row("x", -2, 7.758156579932347e-10)
    expected = 100.0 * -2 / re1.TOTAL_SEG_SITES + (
        math.sqrt(10.0 * (re1.PROJECTION_BASE_D_POSE + 7.758156579932347e-10))
        - math.sqrt(10.0 * re1.PROJECTION_BASE_D_POSE)
    )
    assert re1.singleton_component_projection_delta(row) == expected
    assert expected < 0.0


def test_derive_round_plan_builds_nested_distinct_pair_rounds() -> None:
    rows = [
        _row(re1.CP5V_EVENT_IDS[0], -2, 7.758156579932347e-10, 96),
        _row(re1.CP5V_EVENT_IDS[1], -1, 6.218442251520092e-10, 96),
        _row(re1.CP5V_EVENT_IDS[2], -1, 8.389119964597517e-10, 7),
        _row(re1.CP5V_EVENT_IDS[3], -1, 2.106638993742369e-9, 73),
        _row(re1.CP5V_EVENT_IDS[4], -1, 2.1961514566934712e-9, 7),
        _row(
            "ec1_0120_463b0cb756b2",
            0,
            -1.8564765495815926e-10,
            73,
            eligible=False,
        ),
    ]
    plan = re1.derive_round_plan(rows)
    assert plan["rounds"][0]["proposal_ids"] == [re1.CP5V_EVENT_IDS[0]]
    assert plan["rounds"][1]["proposal_ids"] == [
        re1.CP5V_EVENT_IDS[0],
        re1.CP5V_EVENT_IDS[2],
        "ec1_0120_463b0cb756b2",
    ]
    assert plan["rounds"][0]["max_delta_bytes_on_projection_surface"] == 1
    assert plan["rounds"][1]["max_delta_bytes_on_projection_surface"] == 2
    assert math.isclose(
        plan["rounds"][1]["projected_seg_pose_delta_s"],
        -1.682074784345662e-6,
        rel_tol=0.0,
        abs_tol=1e-18,
    )
    assert plan["rounds"][1]["projected_seg_pose_delta_s"] < plan["rounds"][0]["projected_seg_pose_delta_s"]
    assert plan["cp5v_composition_residual"]["sign_resolved"] is False
    assert plan["cp5v_composition_residual"]["load_bearing"] is False
    assert plan["projection_surface"]["axis_status"] == "BLOCKED_AXIS_MISMATCH"
    assert plan["projection_is_not_pointer_comparable"] is True
    assert plan["projection_is_not_acceptance"] is True


def test_scorer_queue_has_one_owner_store_and_trigger(tmp_path) -> None:
    round_2 = {
        "round": "round_01_singleton_best",
        "archive": {"path": "/candidate/archive.zip", "bytes": 1, "sha256": "a" * 64},
        "runtime": "/candidate/runtime",
        "runtime_tree": {
            "tree": {"tree_sha256": "b" * 64, "file_count": 25},
            "inflate_sh": {"path": "/candidate/inflate.sh", "bytes": 1, "sha256": "c" * 64},
            "inflate_py": {"path": "/candidate/inflate.py", "bytes": 1, "sha256": "d" * 64},
        },
        "runtime_tree_receipt": {
            "path": "/candidate/70_RUNTIME_TREE.json",
            "bytes": 1,
            "sha256": "e" * 64,
        },
    }
    queue = re1.scorer_queue(round_2, tmp_path)
    assert queue["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
    assert queue["owner"] == "MAIN sole scorer-lane router (pending acceptance)"
    assert queue["consumer_store"].endswith("full_n600_exact/round_01_singleton_best")
    assert queue["fire_trigger"]
    assert queue["job_count"] == 1
    assert queue["max_chunk_size"] == 120
    assert queue["live_queue_ingested"] is False
    assert queue["main_hot_state_row_present"] is False
    assert queue["t4_promotion_fire_order"] is False
    assert queue["candidate_runtime"]["tree_sha256"] == "b" * 64
    assert queue["score_claim"] is False


def test_finalize_routes_best_rate_adjusted_projection(tmp_path: Path) -> None:
    source = {"path": "/source.py", "bytes": 1, "sha256": "f" * 64}
    re1.jo1.atomic_json(
        tmp_path / "05_PREFLIGHT.json",
        {
            "payload_producing_runner_source": source,
            "receipt_refresh_runner_source": source,
        },
    )

    def round_result(name: str, projected: float, archive_sha: str) -> dict[str, object]:
        return {
            "round": name,
            "archive": {"path": f"/{name}.zip", "bytes": 1, "sha256": archive_sha * 64},
            "runtime": f"/{name}/runtime",
            "runtime_tree": {
                "tree": {"tree_sha256": archive_sha * 64, "file_count": 25},
                "inflate_sh": {"path": "/inflate.sh", "bytes": 1, "sha256": "a" * 64},
                "inflate_py": {"path": "/inflate.py", "bytes": 1, "sha256": "b" * 64},
            },
            "runtime_tree_receipt": {"path": "/tree.json", "bytes": 1, "sha256": "c" * 64},
            "projected_delta_s_on_alternate_component_surface_plus_real_rate": projected,
        }

    rounds = [
        round_result("round_01_singleton_best", -1.2e-6, "1"),
        round_result("round_02_distinct_pair_stack", -1.0e-6, "2"),
    ]
    result = re1.finalize(tmp_path, rounds)
    assert result["best_unscored_proposal_round"] == "round_01_singleton_best"
    assert result["status"].endswith("FIRE_ORDER_PENDING_MAIN_INGESTION")
    queue = json.loads((tmp_path / "SCORER_QUEUE.json").read_text())
    assert queue["candidate_round"] == "round_01_singleton_best"


def test_round_rows_retain_deterministic_ordinals() -> None:
    rows = [
        _row(re1.CP5V_EVENT_IDS[0], -2, 7.758156579932347e-10, 96),
        _row(re1.CP5V_EVENT_IDS[1], -1, 6.218442251520092e-10, 96),
        _row(re1.CP5V_EVENT_IDS[2], -1, 8.389119964597517e-10, 7),
        _row(re1.CP5V_EVENT_IDS[3], -1, 2.106638993742369e-9, 73),
        _row(re1.CP5V_EVENT_IDS[4], -1, 2.1961514566934712e-9, 7),
        _row("ec1_0120_463b0cb756b2", 0, -1.8564765495815926e-10, 73),
    ]
    plan = re1.derive_round_plan(rows)
    assert [row["ordinal"] for row in plan["rounds"][1]["singleton_rows"]] == [164, 4, 120]


def test_content_addressed_retention_refuses_existing_drift(tmp_path: Path) -> None:
    first = re1.retain_content_addressed_bytes(tmp_path, "runner.py", b"version-one")
    second = re1.retain_content_addressed_bytes(tmp_path, "runner.py", b"version-one")
    assert first == second
    Path(first["path"]).write_bytes(b"corrupt")
    with pytest.raises(re1.RE1Error, match="content-addressed artifact drifted"):
        re1.retain_content_addressed_bytes(tmp_path, "runner.py", b"version-one")


def test_runtime_tree_rejects_extra_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    for root in (source, runtime):
        root.mkdir()
        (root / "inflate.sh").write_text("#!/bin/sh\n")
        (root / "inflate.py").write_text("pass\n")
        (root / "archive.zip").write_bytes(b"payload")
    record = re1.runtime_tree_record(runtime, source)
    assert record["tree"]["file_count"] == 3
    (runtime / "untracked.py").write_text("pass\n")
    with pytest.raises(re1.RE1Error, match="runtime file set drifted"):
        re1.runtime_tree_record(runtime, source)


def test_file_record_tamper_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"kept")
    record = re1.jo1.file_record(path)
    path.write_bytes(b"tampered")
    with pytest.raises(re1.RE1Error, match="failed custody"):
        re1._require_record(record)


def test_re1_python_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_re1_realization_engineered_candidate.py",
            "experiments/tests/test_ddm_re1_realization_engineered_candidate.py",
        ),
    )
    assert findings == []
