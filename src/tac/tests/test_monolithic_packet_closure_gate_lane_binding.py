# SPDX-License-Identifier: MIT
"""Regression tests for monolithic-packet closure-gate lane binding.

Codex HIGH finding #2 (2026-05-08): the previous gate accepted any active
canonical claim — it never compared ``lane_id`` / ``instance_job_id``
against the candidate's intended dispatch. A valid active claim for a
DIFFERENT lane could therefore satisfy the lane-claim section and make
``ready_for_exact_eval_dispatch=True`` for this candidate, defeating the
Level-2 dispatch-custody guard and hiding duplicate / misattributed GPU
work.

This module exercises the new binding contract:

  * ``expected_lane_id`` / ``expected_instance_job_id`` keyword args
    on ``build_monolithic_packet_closure_gate``,
  * the ``candidate_manifest["lane_claim"]`` fallback,
  * the ``--expected-lane-id`` / ``--expected-instance-job-id`` CLI flags
    on ``tools/check_monolithic_packet_closure_gate.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.monolithic_packet_closure_gate import build_monolithic_packet_closure_gate


def _manifest(
    *,
    lane_id: str | None = "expected_lane",
    instance_job_id: str | None = "expected_job",
) -> dict:
    old_sha = "1" * 64
    new_sha = "2" * 64
    manifest: dict = {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "unit-lane-binding",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": True,
        "dispatch_blockers": [],
        "candidate_archive": {
            "path": "candidate.zip",
            "bytes": 123,
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
                    "changed": True,
                    "old_sha256": old_sha,
                    "new_sha256": new_sha,
                    "old_len": 10,
                    "new_len": 9,
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
    binding: dict[str, str] = {}
    if lane_id is not None:
        binding["lane_id"] = lane_id
    if instance_job_id is not None:
        binding["instance_job_id"] = instance_job_id
    if binding:
        manifest["lane_claim"] = binding
    return manifest


def _runtime_proof() -> dict:
    return {
        "schema": "tac_runtime_consumption_proof_v1",
        "ready_for_exact_eval_runtime": True,
        "candidate_archive_sha256": "a" * 64,
        "rebuilt_member_sha256": "b" * 64,
        "new_member_sha256": "b" * 64,
        "changed_sections": {
            "decoder_packed_brotli": "2" * 64,
        },
        "command_sha256": "c" * 64,
        "log_sha256": "d" * 64,
        "score_claim": False,
        "blockers": [],
    }


def _active_claim(
    tmp_path: Path,
    *,
    lane_id: str,
    instance_job_id: str,
) -> dict:
    row = (
        f"| 2026-05-08T00:05:00Z | codex | {lane_id} | lightning | "
        f"{instance_job_id} |  | active_dispatching | unit |"
    )
    claims_path = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims_path.parent.mkdir(parents=True, exist_ok=True)
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
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "claim_status": "active_dispatching",
        "claims_path": ".omx/state/active_lane_dispatch_claims.md",
        "claimed_with": ".venv/bin/python tools/claim_lane_dispatch.py claim",
        "claim_row_sha256": hashlib.sha256(row.encode("utf-8")).hexdigest(),
        "blockers": [],
    }


def test_lane_binding_kwargs_pass_when_claim_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs match the supplied claim -> gate opens."""

    monkeypatch.chdir(tmp_path)
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id=None, instance_job_id=None),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(
            tmp_path, lane_id="expected_lane", instance_job_id="expected_job"
        ),
        expected_lane_id="expected_lane",
        expected_instance_job_id="expected_job",
    )
    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is True
    assert gate["lane_claim"]["lane_id_bound"] is True
    assert gate["lane_claim"]["instance_job_id_bound"] is True
    assert "lane_claim_lane_id_mismatch" not in gate["blockers"]
    assert "lane_claim_instance_job_id_mismatch" not in gate["blockers"]


def test_lane_binding_via_manifest_lane_claim_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest's ``lane_claim`` block supplies the expected binding."""

    monkeypatch.chdir(tmp_path)
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id="expected_lane", instance_job_id="expected_job"),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(
            tmp_path, lane_id="expected_lane", instance_job_id="expected_job"
        ),
    )
    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is True
    assert gate["lane_claim"]["expected_lane_id"] == "expected_lane"
    assert gate["lane_claim"]["expected_instance_job_id"] == "expected_job"


def test_wrong_lane_claim_blocks_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid active claim for the WRONG lane MUST fail the gate.

    This is the security-critical regression: pre-fix, this configuration
    returned ``ready_for_exact_eval_dispatch=True`` because the gate did
    not bind the claim to the candidate's intended lane.
    """

    monkeypatch.chdir(tmp_path)
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id="expected_lane", instance_job_id="expected_job"),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(
            tmp_path,
            lane_id="WRONG_LANE",
            instance_job_id="expected_job",
        ),
    )
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["closure_gate_passed"] is False
    assert "lane_claim_lane_id_mismatch" in gate["blockers"]
    assert gate["lane_claim"]["lane_id_bound"] is False


def test_wrong_instance_job_id_claim_blocks_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong instance_job_id (correct lane) MUST fail the gate."""

    monkeypatch.chdir(tmp_path)
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id="expected_lane", instance_job_id="expected_job"),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(
            tmp_path,
            lane_id="expected_lane",
            instance_job_id="WRONG_JOB",
        ),
    )
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert "lane_claim_instance_job_id_mismatch" in gate["blockers"]
    assert gate["lane_claim"]["instance_job_id_bound"] is False


def test_explicit_kwargs_override_manifest_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs take precedence over manifest's ``lane_claim`` field."""

    monkeypatch.chdir(tmp_path)
    # Manifest says one thing, but explicit kwargs say another and that
    # is what the supplied claim must match.
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id="manifest_lane", instance_job_id="manifest_job"),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(
            tmp_path, lane_id="kwarg_lane", instance_job_id="kwarg_job"
        ),
        expected_lane_id="kwarg_lane",
        expected_instance_job_id="kwarg_job",
    )
    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is True


def test_no_binding_anywhere_blocks_readiness_when_claim_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a claim is supplied but no expected binding exists anywhere,
    the gate must refuse readiness — we don't know what to compare to."""

    monkeypatch.chdir(tmp_path)
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id=None, instance_job_id=None),
        runtime_proof=_runtime_proof(),
        lane_claim=_active_claim(
            tmp_path, lane_id="any_lane", instance_job_id="any_job"
        ),
    )
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert "expected_lane_id_missing" in gate["blockers"]
    assert "expected_instance_job_id_missing" in gate["blockers"]


def test_dry_run_without_claim_does_not_require_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry_run review with no claim has no dispatch authorization, so
    binding cannot misattribute GPU spend; closure may pass."""

    monkeypatch.chdir(tmp_path)
    gate = build_monolithic_packet_closure_gate(
        _manifest(lane_id=None, instance_job_id=None),
        runtime_proof=_runtime_proof(),
        lane_claim=None,
        dry_run=True,
    )
    assert gate["closure_gate_passed"] is True
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert gate["dry_run_not_dispatch_authorization"] is True


def test_check_cli_expected_lane_flags_block_wrong_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ``--expected-lane-id`` CLI flag blocks a wrong-lane claim."""

    from tools.check_monolithic_packet_closure_gate import main

    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    runtime_path = tmp_path / "runtime.json"
    claim_path = tmp_path / "claim.json"
    out = tmp_path / "gate.json"
    # Manifest does NOT carry a lane_claim binding; CLI supplies expected.
    manifest_path.write_text(
        json.dumps(_manifest(lane_id=None, instance_job_id=None)), encoding="utf-8"
    )
    runtime_path.write_text(json.dumps(_runtime_proof()), encoding="utf-8")
    claim_path.write_text(
        json.dumps(
            _active_claim(
                tmp_path, lane_id="WRONG_LANE", instance_job_id="expected_job"
            )
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--runtime-proof-json",
            str(runtime_path),
            "--lane-claim-json",
            str(claim_path),
            "--expected-lane-id",
            "expected_lane",
            "--expected-instance-job-id",
            "expected_job",
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    # Wrong-lane claim must block readiness AND make CLI exit non-zero.
    assert rc == 1
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "lane_claim_lane_id_mismatch" in payload["blockers"]
    assert payload["cli_expected_lane_id"] == "expected_lane"
    assert payload["cli_expected_instance_job_id"] == "expected_job"


def test_check_cli_expected_lane_flags_pass_when_claim_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: matching ``--expected-*`` flags + matching claim opens the gate."""

    from tools.check_monolithic_packet_closure_gate import main

    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    runtime_path = tmp_path / "runtime.json"
    claim_path = tmp_path / "claim.json"
    out = tmp_path / "gate.json"
    manifest_path.write_text(
        json.dumps(_manifest(lane_id=None, instance_job_id=None)), encoding="utf-8"
    )
    runtime_path.write_text(json.dumps(_runtime_proof()), encoding="utf-8")
    claim_path.write_text(
        json.dumps(
            _active_claim(
                tmp_path, lane_id="expected_lane", instance_job_id="expected_job"
            )
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--runtime-proof-json",
            str(runtime_path),
            "--lane-claim-json",
            str(claim_path),
            "--expected-lane-id",
            "expected_lane",
            "--expected-instance-job-id",
            "expected_job",
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["ready_for_exact_eval_dispatch"] is True
