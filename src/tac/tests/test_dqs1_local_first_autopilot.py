# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools import run_dqs1_local_first_autopilot as autopilot


def _local_advisory_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "score_recomputed_from_components": 0.192010,
        "evidence_grade": "macOS-CPU advisory",
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "n_samples": 600,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    payload.update(overrides)
    return payload


def test_cleanup_completed_scratch_requires_valid_local_advisory(tmp_path: Path) -> None:
    results = tmp_path / "results"
    good = results / "materialized" / "good_candidate"
    bad = results / "materialized" / "bad_candidate"
    for root in (good, bad):
        inflated = root / "local_cpu_advisory_work" / "inflated"
        extracted = root / "local_cpu_advisory_work" / "extracted"
        inflated.mkdir(parents=True)
        extracted.mkdir(parents=True)
        (inflated / "0.raw").write_bytes(b"raw")
        (extracted / "archive.zip").write_bytes(b"zip")
    (good / "local_cpu_advisory.json").write_text(
        json.dumps(_local_advisory_payload()),
        encoding="utf-8",
    )
    (bad / "local_cpu_advisory.json").write_text(
        json.dumps(_local_advisory_payload(score_claim=True)),
        encoding="utf-8",
    )

    cleanup = autopilot._cleanup_completed_local_cpu_scratch(
        results_root=results,
        stamp="20260523T000000Z",
    )

    assert cleanup["schema"] == "dqs1_local_first_scratch_cleanup.v1"
    assert cleanup["deleted_path_count"] == 2
    assert not (good / "local_cpu_advisory_work" / "inflated").exists()
    assert not (good / "local_cpu_advisory_work" / "extracted").exists()
    assert (bad / "local_cpu_advisory_work" / "inflated").exists()
    assert any("local_cpu_advisory_contract_blocked" in row["reason"] for row in cleanup["skipped"])
