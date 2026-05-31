# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    archive_bound_candidate_contracts_from_payload,
)
from tac.optimization.archive_bound_candidate_contract_audit import (
    audit_archive_bound_candidate_contracts,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_archive_bound_contract_audit_accepts_valid_contract_surface(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "candidate_package.json",
        {
            "archive_bound_candidate_contract": {
                "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
                "candidate_archive": {
                    "path": "candidate.zip",
                    "sha256": "a" * 64,
                    "bytes": 123,
                },
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            }
        },
    )

    result = audit_archive_bound_candidate_contracts(
        [tmp_path],
        repo_root=tmp_path,
    )

    assert result.passed is True
    assert result.contract_surface_count == 1
    assert result.valid_contract_surface_count == 1
    assert result.migration_required_findings == ()


def test_archive_bound_contract_audit_flags_stale_duplicate_contract_field(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "stale_exact_ready_candidate.json",
        {
            "candidate_id": "stale",
            "ready_for_exact_eval_dispatch": True,
            "archive_bound_candidate_contract": {
                "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            },
        },
    )

    result = audit_archive_bound_candidate_contracts(
        [tmp_path],
        repo_root=tmp_path,
    )

    assert result.passed is False
    assert len(result.blocking_findings) == 1
    assert result.blocking_findings[0].code == (
        "archive_bound_candidate_contract_invalid"
    )
    assert "ready_for_exact_eval_dispatch" in result.blocking_findings[0].message


def test_archive_bound_contract_audit_surfaces_missing_contract_migration(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "pr95_archive_candidate.json",
        {
            "candidate_id": "pr95_candidate",
            "candidate_archive_path": "candidate.zip",
            "candidate_archive_sha256": "b" * 64,
            "candidate_archive_bytes": 456,
            "runtime_consumption_proof_status": "present",
        },
    )

    result = audit_archive_bound_candidate_contracts(
        [tmp_path],
        repo_root=tmp_path,
    )

    assert result.passed is True
    assert len(result.migration_required_findings) == 1
    assert result.migration_required_findings[0].code == (
        "archive_like_candidate_payload_missing_shared_contract"
    )


def test_archive_bound_contract_audit_can_ignore_untracked_scratch(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    tracked = _write_json(
        tmp_path / "tracked_contract.json",
        {
            "archive_bound_candidate_contract": {
                "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
            }
        },
    )
    subprocess.run(["git", "add", tracked.name], cwd=tmp_path, check=True)
    _write_json(
        tmp_path / "ignored_scratch.json",
        {
            "candidate_id": "scratch",
            "candidate_archive_sha256": "c" * 64,
            "candidate_archive_path": "scratch.zip",
        },
    )

    result = audit_archive_bound_candidate_contracts(
        [tmp_path],
        repo_root=tmp_path,
        tracked_only=True,
    )

    assert result.passed is True
    assert result.paths_scanned == 1
    assert result.migration_required_findings == ()


def test_archive_bound_contract_path_compare_preserves_dot_omx_prefix(
    tmp_path: Path,
) -> None:
    row = {
        "candidate_archive": {
            "path": str(tmp_path / ".omx/research/run/archive.zip"),
            "sha256": "d" * 64,
            "bytes": 123,
        },
        "archive_bound_candidate_contract": {
            "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
            "candidate_archive": {
                "path": ".omx/research/run/archive.zip",
                "sha256": "d" * 64,
                "bytes": 123,
            },
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
        },
    }

    assert archive_bound_candidate_contracts_from_payload(row)
