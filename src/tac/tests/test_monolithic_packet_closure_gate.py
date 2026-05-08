from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from tac.monolithic_packet_closure_gate import build_monolithic_packet_closure_gate

if TYPE_CHECKING:
    import pytest


def _manifest(*, archive_bytes: int = 123, changed: bool = True) -> dict:
    old_sha = "1" * 64
    new_sha = "2" * 64
    return {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "unit-monolithic",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": True,
        "dispatch_blockers": [],
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
                    "old_sha256": "3" * 64,
                    "new_sha256": "3" * 64,
                    "old_len": 4,
                    "new_len": 4,
                },
                {
                    "name": "decoder_packed_brotli",
                    "changed": changed,
                    "old_sha256": old_sha,
                    "new_sha256": new_sha if changed else old_sha,
                    "old_len": 10,
                    "new_len": 9 if changed else 10,
                },
            ],
        },
        "replacements": [
            {
                "section_name": "decoder_packed_brotli",
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "old_bytes": 10,
                "new_bytes": 9,
            }
        ],
    }


def _runtime_proof(*, section_sha: str = "2" * 64) -> dict:
    return {
        "schema": "tac_runtime_consumption_proof_v1",
        "ready_for_exact_eval_runtime": True,
        "candidate_archive_sha256": "a" * 64,
        "rebuilt_member_sha256": "b" * 64,
        "new_member_sha256": "b" * 64,
        "changed_sections": {
            "decoder_packed_brotli": section_sha,
        },
        "command_sha256": "c" * 64,
        "log_sha256": "d" * 64,
        "score_claim": False,
        "blockers": [],
    }


def _active_claim(tmp_path: Path) -> dict:
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
    return {
        "schema": "tac_active_lane_claim_json_v1",
        "active": True,
        "lane_id": "unit_lane",
        "instance_job_id": "unit_job",
        "claim_status": "active_dispatching",
        "claims_path": ".omx/state/active_lane_dispatch_claims.md",
        "claimed_with": ".venv/bin/python tools/claim_lane_dispatch.py claim",
        "claim_row_sha256": hashlib.sha256(row.encode("utf-8")).hexdigest(),
        "blockers": [],
    }


def test_closure_gate_opens_with_section_runtime_claim_and_below_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    gate = build_monolithic_packet_closure_gate(
        _manifest(),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(tmp_path),
    )

    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is True
    assert gate["logical_mutation"]["changed_sections"][0]["section_name"] == "decoder_packed_brotli"


def test_parser_proven_logical_mutation_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    gate = build_monolithic_packet_closure_gate(
        _manifest(changed=False),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(tmp_path),
    )

    assert gate["ready_for_exact_eval_dispatch"] is False
    assert "parser_proven_logical_section_mutation_missing" in gate["blockers"]


def test_runtime_proof_must_bind_changed_logical_section_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    gate = build_monolithic_packet_closure_gate(
        _manifest(),
        runtime_proof=_runtime_proof(section_sha="4" * 64),
        lane_claim=_active_claim(tmp_path),
    )

    assert gate["ready_for_exact_eval_dispatch"] is False
    assert "runtime_proof_changed_section_mismatch:decoder_packed_brotli" in gate["blockers"]


def test_missing_claim_blocks_dispatch_but_not_dry_run_closure() -> None:
    real_gate = build_monolithic_packet_closure_gate(
        _manifest(),
        runtime_proof=_runtime_proof(),
    )
    dry_gate = build_monolithic_packet_closure_gate(
        _manifest(),
        runtime_proof=_runtime_proof(),
        dry_run=True,
    )

    assert real_gate["ready_for_exact_eval_dispatch"] is False
    assert "active_lane_claim_missing" in real_gate["blockers"]
    assert dry_gate["closure_gate_passed"] is True
    assert dry_gate["ready_for_exact_eval_dispatch"] is False
    assert dry_gate["dry_run_not_dispatch_authorization"] is True


def test_rate_only_candidate_above_active_floor_is_not_dispatchable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    gate = build_monolithic_packet_closure_gate(
        _manifest(archive_bytes=186_079),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(tmp_path),
    )

    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["rate_only_floor"]["declared_rate_only"] is True
    assert (
        "rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578"
        in gate["blockers"]
    )


def test_closure_gate_cli_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.check_monolithic_packet_closure_gate import main

    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    runtime_path = tmp_path / "runtime.json"
    claim_path = tmp_path / "claim.json"
    out = tmp_path / "gate.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime_path.write_text(json.dumps(_runtime_proof()), encoding="utf-8")
    claim_path.write_text(json.dumps(_active_claim(tmp_path)), encoding="utf-8")

    rc = main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--runtime-proof-json",
            str(runtime_path),
            "--lane-claim-json",
            str(claim_path),
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["schema"] == "tac_monolithic_packet_closure_gate_v1"
    assert payload["ready_for_exact_eval_dispatch"] is True
