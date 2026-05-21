from __future__ import annotations

from pathlib import Path

import tools.plan_hfv_sidecar_frontier_decision as planner


def _candidate(candidate_id: str = "hfv9_magic_explicit_row") -> planner.CandidateAudit:
    return planner.CandidateAudit(
        candidate_id=candidate_id,
        compliance_profile="row_archive_contained_magic_identified",
        archive="archive.zip",
        archive_bytes=178553,
        archive_sha256="a" * 64,
        payload_bytes=36,
        payload_sha256="b" * 64,
        bytes_delta_vs_baseline_archive=291,
        rate_delta_vs_baseline_archive=0.0,
        output_submission_dir="submission_dir",
        output_inflate_py_sha256="c" * 64,
        manifest_path="manifest.json",
        manifest_sha256="d" * 64,
        parity_path="parity.json",
        parity_sha256="e" * 64,
        dispatch_plan_path="paired_dispatch_plan.json",
        dispatch_plan_sha256="f" * 64,
        zip_single_stored_member_x=True,
        zip_member_bytes=178417,
        zip_extra_fields_absent=True,
        shell_parity_cmp_equal=True,
        shell_parity_output_sha256_match=True,
        shell_parity_raw_sha256="0" * 64,
        paired_dispatch_plan_ready=True,
        paired_dispatch_pair_group_id="pair_hfv9",
        paired_dispatch_required_axes=["contest_cuda", "contest_cpu"],
        paired_dispatch_lanes={
            "contest_cpu": "hfv9_exact_eval_contest_cpu",
            "contest_cuda": "hfv9_exact_eval_contest_cuda",
        },
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        policy_rank=1,
        policy_blockers=[],
        audit_errors=[],
    )


def test_unrelated_active_claim_warns_but_does_not_block(monkeypatch) -> None:
    monkeypatch.setattr(planner, "_audit_candidate", lambda _spec: _candidate())
    packet = planner.build_packet(
        active_claim_count=None,
        active_claims=[
            {
                "lane_id": "unrelated_selfcomp_lane",
                "instance_job_id": "fc-live",
                "status": "active_dispatch",
            }
        ],
    )

    assert packet["dispatch_blocked"] is False
    assert packet["dispatch_blockers"] == []
    assert len(packet["unrelated_active_dispatch_claims"]) == 1
    assert packet["same_lane_dispatch_conflicts"] == []
    assert "unrelated active dispatch claims present" in packet["nonblocking_dispatch_claim_note"]


def test_same_lane_active_claim_blocks_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(planner, "_audit_candidate", lambda _spec: _candidate())
    packet = planner.build_packet(
        active_claim_count=None,
        active_claims=[
            {
                "lane_id": "hfv9_exact_eval_contest_cuda",
                "instance_job_id": "hfv9-cuda",
                "status": "active_modal_auth_eval_spawned",
            }
        ],
    )

    assert packet["dispatch_blocked"] is True
    assert packet["dispatch_blockers"] == [
        "same_lane_active_dispatch_claim_present:hfv9_exact_eval_contest_cuda"
    ]
    assert len(packet["same_lane_dispatch_conflicts"]) == 1


def test_legacy_active_count_without_claim_details_still_blocks(monkeypatch) -> None:
    monkeypatch.setattr(planner, "_audit_candidate", lambda _spec: _candidate())
    packet = planner.build_packet(active_claim_count=1)

    assert packet["dispatch_blocked"] is True
    assert packet["dispatch_blockers"] == ["active_dispatch_claims_present:1"]
    assert packet["active_dispatch_claims_known"] is False


def test_active_claim_rows_ignores_terminal_latest(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-21T20:00:00Z | codex | hfv9_exact_eval_contest_cuda | modal | job |  | completed_modal_auth_eval_recovered | done |",
                "| 2026-05-21T19:00:00Z | codex | hfv9_exact_eval_contest_cuda | modal | job |  | active_modal_auth_eval_spawned | old |",
                "| 2026-05-21T19:00:00Z | codex | unrelated | modal | live |  | active_dispatch | live |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = planner._active_claim_rows(claims)

    assert [row["lane_id"] for row in rows] == ["unrelated"]
