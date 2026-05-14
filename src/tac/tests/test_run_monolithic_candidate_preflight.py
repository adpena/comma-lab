# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "run_monolithic_candidate_preflight.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("run_monolithic_candidate_preflight", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_monolithic_candidate_preflight"] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(
    *,
    ready: bool,
    dispatch_blockers: list[str] | None = None,
    archive_bytes: int = 123,
) -> dict:
    return {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "unit-monolithic",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": ready,
        "candidate_archive": {
            "path": "candidate.zip",
            "bytes": archive_bytes,
            "sha256": "a" * 64,
        },
        "monolithic_layout": {
            "grammar": "pr106_ff_packed_hnerv",
            "new_member_sha256": "b" * 64,
            "sections": [
                {
                    "name": "ff_header",
                    "changed": False,
                    "old_sha256": "1" * 64,
                    "new_sha256": "1" * 64,
                    "old_len": 4,
                    "new_len": 4,
                },
                {
                    "name": "decoder_packed_brotli",
                    "changed": True,
                    "old_sha256": "2" * 64,
                    "new_sha256": "c" * 64,
                    "old_len": 10,
                    "new_len": 9,
                },
            ],
        },
        "replacements": [
            {
                "section_name": "decoder_packed_brotli",
                "new_sha256": "c" * 64,
            }
        ],
        "runtime_parity": {
            "declared": ready,
            "ready_flag": ready,
        },
        "lane_claim": {
            "declared": ready,
            "active": ready,
            "lane_id": "unit_lane" if ready else "",
            "instance_job_id": "unit_job" if ready else "",
            "claim_status": "active_dispatching" if ready else "",
        },
        "dispatch_blockers": [] if dispatch_blockers is None else dispatch_blockers,
        "promotion_blockers": ["contest_cuda_auth_eval_missing"],
    }


def _runtime_proof(tmp_path: Path, *, section_sha: str = "c" * 64) -> Path:
    return _write_json(
        tmp_path / "runtime_proof.json",
        {
            "schema": "tac_runtime_consumption_proof_v1",
            "ready_for_exact_eval_runtime": True,
            "candidate_archive_sha256": "a" * 64,
            "rebuilt_member_sha256": "b" * 64,
            "new_member_sha256": "b" * 64,
            "changed_sections": {
                "decoder_packed_brotli": section_sha,
            },
            "command_sha256": "d" * 64,
            "log_sha256": "e" * 64,
            "score_claim": False,
            "blockers": [],
        },
    )


def _lane_claim(tmp_path: Path) -> Path:
    row = (
        "| 2026-05-08T00:05:00Z | codex | unit_lane | lightning | unit_job |  | "
        "active_dispatching | unit |"
    )
    claims_path = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims_path.parent.mkdir(parents=True)
    claims_path.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                row,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return _write_json(
        tmp_path / "lane_claim.json",
        {
            "schema": "tac_active_lane_claim_json_v1",
            "active": True,
            "lane_id": "unit_lane",
            "instance_job_id": "unit_job",
            "claim_status": "active_dispatching",
            "claims_path": ".omx/state/active_lane_dispatch_claims.md",
            "claimed_with": ".venv/bin/python tools/claim_lane_dispatch.py claim",
            "claim_row_sha256": hashlib.sha256(row.encode("utf-8")).hexdigest(),
            "blockers": [],
        },
    )


def test_missing_runtime_proof_and_claim_blockers_are_reported(tmp_path: Path) -> None:
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        _manifest(
            ready=False,
            dispatch_blockers=[
                "runtime_consumption_proof_missing",
                "active_lane_claim_missing",
            ],
        ),
    )

    payload = helper.build_preflight(manifest_path)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["runtime_proof"] == {"provided": False, "path": None}
    assert payload["lane_claim"] == {"provided": False, "path": None}
    assert "runtime_consumption_proof_missing" in payload["blockers"]
    assert "active_lane_claim_missing" in payload["blockers"]
    assert payload["dispatch_attempted"] is False
    assert payload["archive_mutation_attempted"] is False
    assert payload["omx_state_touched"] is False
    assert payload["score_claim"] is False


def test_ready_manifest_without_external_proofs_fails_closed(tmp_path: Path) -> None:
    manifest_path = _write_json(tmp_path / "manifest.json", _manifest(ready=True))

    first = helper.build_preflight(manifest_path)
    second = helper.build_preflight(manifest_path)

    assert first == second
    assert first["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in first["blockers"]
    assert "active_lane_claim_missing" in first["blockers"]
    assert first["candidate_manifest"]["ready_for_exact_eval_dispatch"] is True


def test_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(ready=True)
    manifest["schema"] = "wrong_schema"
    manifest_path = _write_json(tmp_path / "manifest.json", manifest)

    payload = helper.build_preflight(manifest_path)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "candidate_manifest_schema_mismatch" in payload["blockers"]
    assert payload["candidate_manifest"]["schema"] == "wrong_schema"


def test_optional_runtime_proof_and_lane_claim_are_bound_to_ready_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_json(tmp_path / "manifest.json", _manifest(ready=True))
    runtime_proof = _runtime_proof(tmp_path)
    lane_claim = _lane_claim(tmp_path)

    payload = helper.build_preflight(
        manifest_path,
        runtime_proof_json=runtime_proof,
        lane_claim_json=lane_claim,
    )

    assert payload["ready_for_exact_eval_dispatch"] is True
    assert payload["blockers"] == []
    assert payload["runtime_proof"]["candidate_archive_sha256_bound"] is True
    assert payload["runtime_proof"]["new_member_sha256_bound"] is True
    assert payload["runtime_proof"]["changed_sections_bound"] == ["decoder_packed_brotli"]
    assert payload["lane_claim"]["active"] is True
    assert payload["closure_gate"]["ready_for_exact_eval_dispatch"] is True


def test_rate_only_manifest_above_current_floor_is_not_dispatchable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        _manifest(ready=True, archive_bytes=186_079),
    )

    payload = helper.build_preflight(
        manifest_path,
        runtime_proof_json=_runtime_proof(tmp_path),
        lane_claim_json=_lane_claim(tmp_path),
    )

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert (
        "rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578"
        in payload["blockers"]
    )


def test_cli_writes_deterministic_summary_and_fail_if_not_ready(tmp_path: Path) -> None:
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        _manifest(ready=False, dispatch_blockers=["runtime_consumption_proof_missing"]),
    )
    out = tmp_path / "summary.json"

    rc = helper.main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--json-out",
            str(out),
        ]
    )
    fail_rc = helper.main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--fail-if-not-ready",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert fail_rc == 1
    assert payload["schema"] == "tac_monolithic_candidate_preflight_v1"
    assert payload["ready_for_exact_eval_dispatch"] is False
