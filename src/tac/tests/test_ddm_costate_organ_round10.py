"""Round-10 current-campaign re-adjudication tests."""
from __future__ import annotations

from pathlib import Path

from tac.ddm_costate_organ import (
    REPO,
    _open_gate_ownership_scan,
    _round10_recursive_leverage,
    build_live_ddm_costate,
    consumed_evidence_registry,
    digest_lines,
)


def test_round10_re_adjudicates_every_live_open_row() -> None:
    legacy = _open_gate_ownership_scan(REPO)
    result = _round10_recursive_leverage(REPO, legacy)
    assert result["legacy_open_denominator"] == 28
    assert result["re_adjudicated_count"] == 28
    assert not any(
        row["round10_disposition"] == "UNRESOLVED_FAIL_CLOSED"
        for row in result["re_adjudications"]
    )
    assert all(row["current_owner"] and row["consumer_store"] for row in result["re_adjudications"])
    assert result["actuation"] == "NONE"
    assert result["maturity"] == "_dev"


def test_round10_ranks_current_dcc1_successors_not_tr1() -> None:
    result = _round10_recursive_leverage(REPO, _open_gate_ownership_scan(REPO))
    duties = [row["duty"] for row in result["duty_ranking"]]
    assert duties == [
        "CCS1_FIXED_X_CAUSAL_GM_RATE_RUNG",
        "QX_QBT_TARGET_OVERWRITE_GRAMMAR",
        "QBW1_QBMIX_CAUSAL_QUOTIENT_RENDERER",
        "RB1_EXACT_CHANGED_OBJECT_RENDERER",
    ]
    assert not any("TR1" in duty for duty in duties)


def test_round10_fail_closed_when_current_sources_are_absent(tmp_path: Path) -> None:
    result = _round10_recursive_leverage(
        tmp_path,
        {"open_gate_count": 1, "open_gate_unfired_rows": [{"row_id": "QZ99"}]},
    )
    assert result["available"] is False
    assert result["re_adjudications"][0]["round10_disposition"] == "UNRESOLVED_FAIL_CLOSED"


def test_round10_sources_are_registered_and_digest_visible() -> None:
    registry = consumed_evidence_registry()
    assert ".omx/research/ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md" in registry["paths"]
    report = build_live_ddm_costate(repo_root=REPO)
    assert report["round10"]["available"] is True
    assert "DDM-round10[current AFR1/DCC1]: re-adjudicated=28/28" in "\n".join(
        digest_lines(report)
    )

