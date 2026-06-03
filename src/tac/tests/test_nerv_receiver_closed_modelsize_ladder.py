# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

from tac.analysis.nerv_receiver_closed_modelsize_ladder import (
    SCHEMA,
    build_nerv_receiver_closed_modelsize_ladder,
)


def test_receiver_closed_modelsize_ladder_selects_receiver_byte_budget(
    tmp_path: Path,
) -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            _row("tiny", 0.03, 24, 20_000, 0.240, proof=True, tmp_path=tmp_path),
            _row("small", 0.06, 48, 40_000, 0.205, proof=True, tmp_path=tmp_path),
            _row("medium", 0.11, 80, 80_000, 0.206, proof=True, tmp_path=tmp_path),
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
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
                "axis_tag": "[contest-CPU]",
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


def test_advisory_nonrate_score_does_not_open_receiver_ladder(tmp_path: Path) -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                "row_id": "tiny_advisory",
                "axis_tag": "[contest-CPU]",
                "modelsize_mparams": 0.03,
                "fc_dim": 24,
                "archive_bytes": 20_000,
                "archive_sha256": _row_archive_sha("tiny_advisory", 20_000),
                "nonrate_score_advisory": 0.240,
                "receiver_proof_passed": True,
                "receiver_closed": True,
                "byte_closed_receiver_proof": True,
                **_proof_identity_fields(
                    tmp_path,
                    "tiny_advisory",
                    20_000,
                    _row_archive_sha("tiny_advisory", 20_000),
                ),
            },
            {
                "row_id": "small_advisory",
                "axis_tag": "[contest-CPU]",
                "modelsize_mparams": 0.06,
                "fc_dim": 48,
                "archive_bytes": 40_000,
                "archive_sha256": _row_archive_sha("small_advisory", 40_000),
                "nonrate_score_advisory": 0.205,
                "receiver_proof_passed": True,
                "receiver_closed": True,
                "byte_closed_receiver_proof": True,
                **_proof_identity_fields(
                    tmp_path,
                    "small_advisory",
                    40_000,
                    _row_archive_sha("small_advisory", 40_000),
                ),
            },
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    rows = {row["row_id"]: row for row in payload["normalized_rows"]}
    assert payload["status"] == "receiver_closed_modelsize_ladder_blocked"
    assert payload["budget_row_count"] == 0
    assert payload["receiver_closed_row_count"] == 0
    assert payload["modelsize_budget_plan"]["receiver_closed_points"] == []
    assert rows["tiny_advisory"]["nonrate_score_key"] == "nonrate_score_advisory"
    assert rows["tiny_advisory"]["nonrate_score_evidence_kind"] == "advisory"
    assert "advisory_nonrate_score_not_receiver_closed" in rows["tiny_advisory"][
        "blockers"
    ]
    assert payload["ready_for_carrier_training_plan"] is False


def test_missing_axis_does_not_open_receiver_ladder(tmp_path: Path) -> None:
    row = _row("tiny", 0.03, 24, 20_000, 0.240, proof=True, tmp_path=tmp_path)
    row.pop("axis_tag")

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [row],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    normalized = payload["normalized_rows"][0]
    assert payload["status"] == "receiver_closed_modelsize_ladder_blocked"
    assert payload["budget_row_count"] == 0
    assert payload["receiver_closed_row_count"] == 0
    assert normalized["source_axis_receiver_closed_authority"] is False
    assert "source_axis_not_receiver_closed_contest_authority" in normalized[
        "blockers"
    ]
    assert payload["ready_for_carrier_training_plan"] is False


def test_true_authority_flags_block_receiver_modelsize_rows(tmp_path: Path) -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                **_row("tiny", 0.03, 24, 20_000, 0.240, proof=True, tmp_path=tmp_path),
                "promotion_eligible": True,
            },
            _row("small", 0.06, 48, 40_000, 0.205, proof=True, tmp_path=tmp_path),
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    rows = {row["row_id"]: row for row in payload["normalized_rows"]}
    assert rows["tiny"]["receiver_closed_modelsize_row"] is False
    assert "source_authority_flag_true:promotion_eligible" in rows["tiny"][
        "blockers"
    ]
    assert payload["receiver_closed_row_count"] == 1
    assert payload["ready_for_carrier_training_plan"] is False


def test_missing_modelsize_and_fc_dim_blocks_budget_row() -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                "row_id": "bytes_only",
                "axis_tag": "[contest-CPU]",
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


def test_boolean_only_receiver_proof_does_not_open_modelsize_ladder() -> None:
    payload = build_nerv_receiver_closed_modelsize_ladder(
        [
            {
                **_row("tiny", 0.03, 24, 20_000, 0.240, proof=False),
                "receiver_proof_passed": True,
                "receiver_closed": True,
                "byte_closed_receiver_proof": True,
            },
            {
                **_row("small", 0.06, 48, 40_000, 0.205, proof=False),
                "receiver_proof_passed": True,
                "receiver_closed": True,
                "byte_closed_receiver_proof": True,
            },
        ],
        carrier_id="snerv",
    )

    rows = {row["row_id"]: row for row in payload["normalized_rows"]}
    assert payload["status"] == "receiver_closed_modelsize_ladder_blocked"
    assert payload["receiver_closed_row_count"] == 0
    assert payload["budget_row_count"] == 2
    assert payload["modelsize_budget_plan"]["status"] == (
        "advisory_or_projected_modelsize_budget_selected"
    )
    assert payload["ready_for_carrier_training_plan"] is False
    assert rows["tiny"]["receiver_proof_identity_bound"] is False
    assert "receiver_proof_identity_missing" in rows["tiny"]["blockers"]
    assert "receiver_proof_path_missing" in rows["tiny"]["blockers"]


def test_file_backed_receiver_proof_without_parent_booleans_opens_modelsize_ladder(
    tmp_path: Path,
) -> None:
    tiny = _row("tiny", 0.03, 24, 20_000, 0.240, proof=False)
    tiny.update(
        _proof_identity_fields(
            tmp_path,
            "tiny",
            20_000,
            str(tiny["archive_sha256"]),
        )
    )
    small = _row("small", 0.06, 48, 40_000, 0.205, proof=False)
    small.update(
        _proof_identity_fields(
            tmp_path,
            "small",
            40_000,
            str(small["archive_sha256"]),
        )
    )

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [tiny, small],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    rows = {row["row_id"]: row for row in payload["normalized_rows"]}
    assert payload["status"] == "receiver_closed_modelsize_ladder_ready"
    assert payload["receiver_closed_row_count"] == 2
    assert rows["tiny"]["receiver_proof_identity_bound"] is True
    assert rows["tiny"]["receiver_proof_passed"] is True


def test_mixed_family_rows_are_not_absorbed_into_carrier_ladder(tmp_path: Path) -> None:
    hnerv = _row(
        "hnerv_row", 0.03, 24, 20_000, 0.220, proof=True, tmp_path=tmp_path
    )
    hnerv["solved_budget"] = {"family": "hnerv"}
    snerv = _row(
        "snerv_row", 0.06, 48, 40_000, 0.205, proof=True, tmp_path=tmp_path
    )
    snerv["family"] = "snerv"

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [hnerv, snerv],
        carrier_id="snerv",
        repo_root=tmp_path,
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


def test_hinerv_ladder_rejects_hnerv_rows_as_distinct_family(tmp_path: Path) -> None:
    hnerv = _row(
        "hnerv_row", 0.03, 24, 20_000, 0.220, proof=True, tmp_path=tmp_path
    )
    hnerv["family"] = "hnerv"

    payload = build_nerv_receiver_closed_modelsize_ladder(
        [hnerv],
        carrier_id="hinerv",
        repo_root=tmp_path,
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
                "axis_tag": "[contest-CPU]",
                "modelsize_mparams": 0.03,
                "fc_dim": 24,
                "num_pairs": 600,
                "archive_path": "candidate.zip",
                "nonrate_score": 0.240,
                "receiver_proof_passed": True,
                "receiver_closed": True,
                "byte_closed_receiver_proof": True,
                **_proof_identity_fields(
                    tmp_path,
                    "path_backed",
                    archive.stat().st_size,
                    expected_sha,
                ),
            },
            _row("measured", 0.06, 48, 40_000, 0.205, proof=True, tmp_path=tmp_path),
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
    tmp_path: Path | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": row_id,
        "axis_tag": "[contest-CPU]",
        "modelsize_mparams": modelsize,
        "fc_dim": fc_dim,
        "num_pairs": 600,
        "archive_bytes": archive_bytes,
        "archive_sha256": _row_archive_sha(row_id, archive_bytes),
        "nonrate_score": nonrate_score,
        "receiver_proof_passed": proof,
    }
    if proof and tmp_path is not None:
        row["receiver_closed"] = True
        row["byte_closed_receiver_proof"] = True
        row.update(
            _proof_identity_fields(
                tmp_path,
                row_id,
                archive_bytes,
                _row_archive_sha(row_id, archive_bytes),
            )
        )
    return row


def _proof_identity_fields(
    tmp_path: Path,
    row_id: str,
    archive_bytes: int,
    archive_sha256: str,
) -> dict[str, object]:
    proof = tmp_path / f"{row_id}.receiver_proof.json"
    proof.write_text(
        (
            '{"schema":"snerv_inverse_steg_generated_receiver_proof.v1",'
            '"receiver_contract_satisfied":true,'
            '"runtime_consumption_proof_ready":true,'
            '"runtime_consumption_proof_passed":true,'
            f'"archive_bytes":{archive_bytes},'
            f'"archive_sha256":"{archive_sha256}",'
            '"receiver_output_bytes":123,'
            '"expected_receiver_output_bytes":123,'
            '"blockers":[]}\n'
        ),
        encoding="utf-8",
    )
    return {
        "receiver_proof_path": proof.as_posix(),
        "receiver_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    }


def _row_archive_sha(row_id: str, archive_bytes: int) -> str:
    return hashlib.sha256(f"{row_id}:{archive_bytes}".encode("ascii")).hexdigest()
