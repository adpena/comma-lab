# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

from tac.analysis.nerv_receiver_closed_modelsize_ladder import (
    SCHEMA,
    build_nerv_receiver_closed_modelsize_ladder,
)


def test_receiver_closed_modelsize_ladder_selects_receiver_byte_budget() -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            _row("tiny", 0.03, 24, 20_000, 0.240, proof=True),
            _row("small", 0.06, 48, 40_000, 0.205, proof=True),
            _row("medium", 0.11, 80, 80_000, 0.206, proof=True),
        ],
        carrier_id="snerv",
    )

    assert payload["schema"] == SCHEMA
    assert payload["status"] == "receiver_closed_modelsize_ladder_ready"
    assert payload["receiver_closed_row_count"] == 3
    assert payload["budget_row_count"] == 3
    assert payload["receiver_closed_selected_archive_bytes"] == 40_000
    assert payload["receiver_closed_selected_modelsize_mparams"] == 0.06
    assert payload["receiver_closed_selected_fc_dim"] == 48
    assert payload["modelsize_budget_plan"]["status"] == (
        "receiver_closed_modelsize_budget_selected"
    )
    assert payload["ready_for_carrier_training_plan"] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_advisory_or_projected_rows_do_not_open_receiver_ladder() -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                "row_id": "projected",
                "modelsize_mparams": 0.03,
                "fc_dim": 24,
                "projected_archive_bytes_600pair": 20_000,
                "nonrate_score": 0.240,
                "lower_bound_only": True,
            },
            _row("zip_without_proof", 0.06, 48, 40_000, 0.205, proof=False),
        ],
        carrier_id="snerv",
    )

    assert payload["status"] == "receiver_closed_modelsize_ladder_blocked"
    assert payload["receiver_closed_row_count"] == 0
    assert payload["modelsize_budget_plan"]["status"] == (
        "advisory_or_projected_modelsize_budget_selected"
    )
    assert payload["ready_for_carrier_training_plan"] is False
    assert "receiver_closed_modelsize_ladder_not_ready" in payload["blockers"]
    assert any("receiver_closed_byte_proof_missing" in b for b in payload["blockers"])
    assert any("projected_archive_bytes_not_receiver_closed" in b for b in payload["blockers"])


def test_missing_modelsize_and_fc_dim_blocks_budget_row() -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                "row_id": "bytes_only",
                "archive_bytes": 20_000,
                "nonrate_score": 0.220,
                "receiver_proof_passed": True,
            }
        ],
        carrier_id="hinerv",
    )

    assert payload["budget_row_count"] == 0
    assert payload["receiver_closed_row_count"] == 0
    assert payload["ready_for_carrier_training_plan"] is False
    assert any("modelsize_or_fc_dim_missing" in b for b in payload["blockers"])
    assert "no_rows_eligible_for_modelsize_budget_plan" in payload["blockers"]


def test_mixed_family_rows_are_not_absorbed_into_carrier_ladder() -> None:
    hnerv = _row("hnerv_row", 0.03, 24, 20_000, 0.220, proof=True)
    hnerv["solved_budget"] = {"family": "hnerv"}
    snerv = _row("snerv_row", 0.06, 48, 40_000, 0.205, proof=True)
    snerv["family"] = "snerv"

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [hnerv, snerv],
        carrier_id="snerv",
    )

    rows = {row["row_id"]: row for row in payload["normalized_rows"]}
    assert "carrier_family_mismatch" in rows["hnerv_row"]["blockers"]
    assert rows["snerv_row"]["receiver_closed_modelsize_row"] is True
    assert payload["budget_row_count"] == 1
    assert payload["receiver_closed_row_count"] == 1
    assert payload["ready_for_carrier_training_plan"] is False
    mismatch_records = [
        record
        for record in payload["blocker_records"]
        if record["reason"] == "carrier_family_mismatch"
    ]
    assert mismatch_records == [
        {
            "scope": "row",
            "row_id": "hnerv_row",
            "reason": "carrier_family_mismatch",
            "carrier_id": "snerv",
            "source_family": "hnerv",
            "source_family_key": "hnerv",
            "accepted_carrier_aliases": ["snerv", "snerv_t", "snervt"],
        }
    ]


def test_hinerv_ladder_rejects_hnerv_rows_as_distinct_family() -> None:
    hnerv = _row("hnerv_row", 0.03, 24, 20_000, 0.220, proof=True)
    hnerv["family"] = "hnerv"

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [hnerv],
        carrier_id="hinerv",
    )

    row = payload["normalized_rows"][0]
    assert row["source_family"] == "hnerv"
    assert row["source_family_key"] == "hnerv"
    assert "carrier_family_mismatch" in row["blockers"]
    assert payload["budget_row_count"] == 0
    assert payload["receiver_closed_row_count"] == 0
    assert payload["ready_for_carrier_training_plan"] is False
    mismatch_records = [
        record
        for record in payload["blocker_records"]
        if record["reason"] == "carrier_family_mismatch"
    ]
    assert mismatch_records == [
        {
            "scope": "row",
            "row_id": "hnerv_row",
            "reason": "carrier_family_mismatch",
            "carrier_id": "hinerv",
            "source_family": "hnerv",
            "source_family_key": "hnerv",
            "accepted_carrier_aliases": ["hi-nerv", "hi_nerv", "hinerv"],
        }
    ]


def test_archive_path_bytes_and_sha_can_back_receiver_closed_row(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"snerv-archive-bytes")
    expected_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                "row_id": "path_backed",
                "modelsize_mparams": 0.03,
                "fc_dim": 24,
                "archive_path": "candidate.zip",
                "nonrate_score": 0.240,
                "receiver_contract_satisfied": True,
            },
            _row("measured", 0.06, 48, 40_000, 0.205, proof=True),
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    row = {r["row_id"]: r for r in payload["normalized_rows"]}["path_backed"]
    assert row["archive_bytes"] == len(b"snerv-archive-bytes")
    assert row["archive_sha256"] == expected_sha
    assert row["receiver_closed_modelsize_row"] is True
    assert payload["receiver_closed_row_count"] == 2
    assert payload["modelsize_budget_plan"]["status"] == (
        "receiver_closed_modelsize_budget_selected"
    )


def _row(
    row_id: str,
    modelsize: float,
    fc_dim: int,
    archive_bytes: int,
    nonrate_score: float,
    *,
    proof: bool,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "modelsize_mparams": modelsize,
        "fc_dim": fc_dim,
        "archive_bytes": archive_bytes,
        "nonrate_score": nonrate_score,
        "receiver_proof_passed": proof,
    }
