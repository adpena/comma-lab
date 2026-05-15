# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO_ROOT / "tools" / "backfill_terminal_claim_evidence.py"
    spec = importlib.util.spec_from_file_location("backfill_terminal_claim_evidence", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backfill_terminal_claim_evidence_skips_exact_cuda_and_covers_no_cost(tmp_path: Path) -> None:
    mod = _load_module()
    claims = tmp_path / "active_lane_dispatch_claims.md"
    evidence = tmp_path / "cathedral_autopilot_evidence.jsonl"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| 2026-05-15T00:22:25Z | codex | exact_lane | modal | exact_job |  | completed_contest_cuda_modal_auth_eval_recovered | exact result needs review packet |",
                "| 2026-05-15T00:19:37Z | codex | lane_z3_balle_hyperprior_bolton_recover_20260514 | modal | substrate_z3_modal__smoke__100ep |  | stale_superseded_sigurg_killed_pre_modal_spawn_no_cost | killed before provider spawn |",
                "| 2026-05-14T20:00:00Z | codex | unrelated_lane | modal | job |  | completed_other | unrelated |",
                "| 2026-05-14T19:00:00Z | codex | lane_z3_balle_hyperprior_bolton_recover_20260514 | modal | substrate_z3_active |  | active_modal_training | still active |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mod.missing_rows(
        claims_path=claims,
        evidence_path=evidence,
        earliest_timestamp_utc="2026-05-13T00:00:00Z",
        exact_eval_earliest_timestamp_utc="2026-05-14T06:00:00Z",
    )

    assert [row.instance_job_id for row in rows] == ["substrate_z3_modal__smoke__100ep"]
    payload = mod.evidence_row(rows[0], claims_path=claims)
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["covered_terminal_claims"] == [
        {
            "lane_id": "lane_z3_balle_hyperprior_bolton_recover_20260514",
            "instance_job_id": "substrate_z3_modal__smoke__100ep",
            "status": "stale_superseded_sigurg_killed_pre_modal_spawn_no_cost",
        }
    ]


def test_backfill_terminal_claim_evidence_is_idempotent_against_existing_coverage(tmp_path: Path) -> None:
    mod = _load_module()
    claims = tmp_path / "active_lane_dispatch_claims.md"
    evidence = tmp_path / "cathedral_autopilot_evidence.jsonl"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| 2026-05-15T00:19:37Z | codex | lane_d4_wyner_ziv_frame_0_substrate_20260514 | modal | substrate_d4_modal__smoke__100ep |  | stale_superseded_mount_upload_race_no_modal_spend | no modal spend |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "covered_terminal_claims": [
                    {
                        "lane_id": "lane_d4_wyner_ziv_frame_0_substrate_20260514",
                        "instance_job_id": "substrate_d4_modal__smoke__100ep",
                        "status": "stale_superseded_mount_upload_race_no_modal_spend",
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mod.missing_rows(
        claims_path=claims,
        evidence_path=evidence,
        earliest_timestamp_utc="2026-05-13T00:00:00Z",
        exact_eval_earliest_timestamp_utc="2026-05-14T06:00:00Z",
    )

    assert rows == []
