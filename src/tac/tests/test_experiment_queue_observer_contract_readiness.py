from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue_observer import (
    _materializer_payload_revalidation,
    _materializer_queue_row_allows_deferred_runtime_identity,
    _optimizer_candidate_queue_materializer_row,
    _path_artifact_record,
)
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    archive_bound_candidate_contract_fields_for_row,
)
from tac.optimization.serialized_archive_economics import (
    SERIALIZED_ARCHIVE_DELTA_SCHEMA,
)


def _write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _contract_row(tmp_path: Path) -> dict[str, Any]:
    candidate = _write_bytes(tmp_path / "out" / "archive.zip", b"candidate-archive")
    source = _write_bytes(tmp_path / "in" / "source.zip", b"source-archive")
    row: dict[str, Any] = {
        "candidate_id": "observer-contract-fixture",
        "target_kind": "archive_range_ans_recode_v1",
        "materializer_id": "range_ans_materializer",
        "receiver_contract_kind": "decode_only_receiver",
        "receiver_contract_satisfied": False,
        "readiness_blockers": [
            "receiver_contract_not_satisfied",
            "runtime_adapter_expected_tree_sha_missing",
        ],
        "candidate_archive_path": candidate["path"],
        "candidate_archive_sha256": candidate["sha256"],
        "candidate_archive_bytes": candidate["bytes"],
        "source_archive_path": source["path"],
        "source_archive_sha256": source["sha256"],
        "source_archive_bytes": source["bytes"],
        "byte_closed_candidate_materialized": True,
        "candidate_archive_materialized": True,
        "runtime_consumption_proof_ready": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "serialized_archive_delta": {
            "schema": SERIALIZED_ARCHIVE_DELTA_SCHEMA,
            "status": "realized_saving",
            "realized_saved_bytes": 1,
            "savings_realized": True,
            "source_archive_bytes": source["bytes"],
            "candidate_archive_bytes": candidate["bytes"],
            "selected_materialization_key": "archive_range_ans_recode_v1",
        },
    }
    row.update(
        archive_bound_candidate_contract_fields_for_row(
            row,
            repo_root=tmp_path,
            family_id="observer_contract_test",
            candidate_chain_id="observer-contract-fixture",
            selected_transform_kind="archive_range_ans_recode_v1",
        )
    )
    return row


def test_contract_row_allows_expected_deferred_runtime_identity(tmp_path: Path) -> None:
    row = _contract_row(tmp_path)

    assert _materializer_queue_row_allows_deferred_runtime_identity(
        row,
        ["runtime_tree_sha256_missing"],
    )

    materializer = _optimizer_candidate_queue_materializer_row(row, row_index=3)
    assert materializer is not None
    assert materializer["archive_bound_candidate_contract_valid"] is True
    assert materializer["ready_for_exact_eval_dispatch"] is False
    assert materializer["archive_bound_candidate_contract"]["schema"]


def test_observer_rejects_truthy_raw_exact_ready_next_to_contract(
    tmp_path: Path,
) -> None:
    row = _contract_row(tmp_path)
    row["ready_for_exact_eval_dispatch"] = True

    assert not _materializer_queue_row_allows_deferred_runtime_identity(
        row,
        ["runtime_tree_sha256_missing"],
    )

    materializer = _optimizer_candidate_queue_materializer_row(row, row_index=0)
    assert materializer is not None
    assert materializer["archive_bound_candidate_contract_valid"] is False
    assert materializer["ready_for_exact_eval_dispatch"] is False
    assert any(
        "ready_for_exact_eval_dispatch" in blocker
        for blocker in materializer["archive_bound_candidate_contract_blockers"]
    )


def test_observer_rejects_stale_duplicate_contract_readiness(
    tmp_path: Path,
) -> None:
    row = _contract_row(tmp_path)
    row["archive_bound_candidate_ready"] = True

    materializer = _optimizer_candidate_queue_materializer_row(row, row_index=0)
    assert materializer is not None
    assert materializer["archive_bound_candidate_contract_valid"] is False
    assert any(
        "archive_bound_contract_stale_duplicate_field:archive_bound_candidate_ready"
        in blocker
        for blocker in materializer["archive_bound_candidate_contract_blockers"]
    )

    revalidation = _materializer_payload_revalidation(
        row,
        repo_root=tmp_path,
        context="observer_contract_test",
    )
    assert revalidation["archive_bound_candidate_contract_valid"] is False
    assert any(
        "archive_bound_contract_stale_duplicate_field:archive_bound_candidate_ready"
        in blocker
        for blocker in revalidation["archive_bound_candidate_contract_blockers"]
    )


def test_observer_extracts_nested_pr95_adapter_package_contract(tmp_path: Path) -> None:
    row = _contract_row(tmp_path)
    package = {
        "schema": ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
        "candidate_family": "pr95_mlx_hnerv",
        "candidate_row_count": 1,
        "ready_contract_count": 0,
        "receiver_proof_gate_passed_count": 0,
        "candidate_rows": [row],
        "archive_bound_candidate_contract_surfaces": [
            row["archive_bound_candidate_contract_surface"]
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    path = tmp_path / "pr95_package_report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pr95_mlx_pytorch_state_dict_to_contest_archive.v1",
                "archive_bound_candidate_adapter_package": package,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    record = _path_artifact_record(path, repo_root=tmp_path)

    assert record["pr95_mlx_package_report"] is True
    assert record["archive_bound_candidate_contract_valid"] is True
    assert record["archive_bound_candidate_contract_key"] == row[
        "archive_bound_candidate_contract"
    ]["contract_key"]
    assert record["candidate_archive"]["sha256"] == row["candidate_archive_sha256"]
    assert record["archive_bound_candidate_adapter_package_candidate_count"] == 1
