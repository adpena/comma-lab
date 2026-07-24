"""Tests for the live DDM costate organ and its no-fake scheduling laws."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from tac.ddm_costate_law import (  # noqa: E402
    ddm_joint_costate,
    gauss_southwell_validity_score,
    realized_pair_distortion_delta,
)
from tac.ddm_costate_organ import (  # noqa: E402
    DdmCostateCheckpoint,
    build_live_ddm_costate,
    discover_sources,
    rank_scheduler_blocks,
    register_ddm_costate_checkpoint,
)
from tac.witness_control.resume_registry import ResumeRegistry  # noqa: E402


def test_lambda_d2_exact_product_and_guards() -> None:
    assert ddm_joint_costate(2.0, 0.5, 0.25, 0.1, 0.8) == pytest.approx(0.02)
    assert gauss_southwell_validity_score(0.2, 0.4) == pytest.approx(0.08)
    with pytest.raises(ValueError, match="visibility"):
        ddm_joint_costate(1.0, 1.1, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="exact_gap"):
        ddm_joint_costate(-1.0, 1.0, 1.0, 1.0, 1.0)


def test_pair_delta_excludes_unallocated_shared_rate() -> None:
    delta = realized_pair_distortion_delta(
        d_seg_before=0.03,
        d_seg_after=0.029,
        d_pose_before=160.0,
        d_pose_after=160.0,
    )
    assert delta == pytest.approx(-0.1)


def test_scheduler_topology_freeing_coarse_and_gauss_southwell() -> None:
    blocks = [
        {
            "block_id": "blocked",
            "dependencies": ["later"],
            "frees_bytes": True,
            "coarse_level": 0,
            "lambda_abs": 100.0,
            "validity_radius": 1.0,
        },
        {
            "block_id": "spend_coarse",
            "dependencies": ["root"],
            "frees_bytes": False,
            "coarse_level": 0,
            "lambda_abs": 100.0,
            "validity_radius": 1.0,
        },
        {
            "block_id": "free_first",
            "dependencies": ["root"],
            "frees_bytes": True,
            "coarse_level": 9,
            "lambda_abs": 0.001,
            "validity_radius": 0.1,
        },
        {
            "block_id": "same_scale_low_gs",
            "dependencies": ["root"],
            "frees_bytes": False,
            "coarse_level": 2,
            "lambda_abs": 0.2,
            "validity_radius": 0.5,
        },
        {
            "block_id": "same_scale_high_gs",
            "dependencies": ["root"],
            "frees_bytes": False,
            "coarse_level": 2,
            "lambda_abs": 0.4,
            "validity_radius": 0.5,
        },
    ]
    ranked = rank_scheduler_blocks(blocks, completed={"root"})
    assert [row["block_id"] for row in ranked] == [
        "free_first",
        "spend_coarse",
        "same_scale_high_gs",
        "same_scale_low_gs",
    ]
    assert ranked[2]["gauss_southwell_validity"] == pytest.approx(0.2)


def test_checkpoint_registers_and_roundtrips() -> None:
    registry = ResumeRegistry()
    original = DdmCostateCheckpoint({"g3": "abc"}, ["source_custody"], cycle=4)
    register_ddm_costate_checkpoint(registry, original)
    assert registry.names == ["ddm_live_costate_advisory"]
    arrays = original.state_arrays("__ddmcostate_")

    restored = DdmCostateCheckpoint({}, [])
    assert restored.restore_from_cfg("__ddmcostate_", arrays)
    assert restored.to_dict() == original.to_dict()


def test_current_live_fleet_pair_site_lambda_and_duties() -> None:
    report = build_live_ddm_costate()
    assert report["available"] is True
    assert report["source_custody"]["quarantined_20260717_run_consulted"] is False
    assert all("20260717" not in row["path"] for row in report["sources"].values() if row["available"])
    assert report["live"]["reach"]["road_described_fraction"] == pytest.approx(0.7053524530023775)
    assert report["live"]["box"]["inside_c1_byte_box"] is True
    assert len(report["lambda"]["primitive_rows"]) >= 8
    oracle_row = report["source_custody"]["scorer_value_oracle_rate_row"]
    assert oracle_row["row"] == "rate (archive bytes only)"
    assert oracle_row["freshness"] == "FRESH"
    assert oracle_row["lineage"][0]["fresh"] is True
    assert report["live"]["fleet"]["scorer_value_oracle_coverage"] == {
        "WRAPPED": 21,
        "TYPED-GAP": 0,
    }

    if report["source_custody"]["g3_full_atlas"]["status"] == "VERIFIED":
        # ev1 N600_EXACT_COMPLETE join (2026-07-24): 600 pair rows / 3,000 site
        # rows replaced the historical 8/40 subset. Backtest quality DROPPED on
        # the full join (rho 0.903->0.748, ndcg 0.927->0.196) - the old values
        # were 8-pair-subset optimism; routed as FEED-603 subset-overfit signal.
        assert len(report["lambda"]["pair_rows"]) == 600
        assert len(report["lambda"]["site_rows"]) == 3000
        backtest = report["lambda"]["backtest"]
        assert backtest["spearman_rho"] == pytest.approx(0.7476669456024575)
        assert backtest["ndcg_at_4"] == pytest.approx(0.19557065696692438)
        selected = report["campaign"]["metric_state"]["lambda_ranker"][
            "selected_model"
        ]
        assert selected["candidate_id"] == "factorized_ms4d_interactions"
        assert selected["metrics"]["heldout_only"] is True
        assert selected["metrics"]["ndcg_at_4"] == pytest.approx(1.0)
        assert selected["metrics"]["spearman_rho"] == pytest.approx(
            0.8607149751465011
        )

    duties = report["duties"]
    assert duties["legacy_authority_snapshot_rows_retained"] == 115
    assert duties["retention_status"] == "AT_LEAST_115_RETAINED"
    assert [row["duty"] for row in duties["live_ranked"]] == [
        "J_paint",
        "R6_rehearsal",
        "DDM_iteration_curves",
    ]
    assert report["instruments"]["ncde"]["status"] == "INVALID_FOR_LIVE_DDM"
    assert report["instruments"]["pose_gate"]["legacy_degenerate_guard"] == ("PRESERVED_FAIL_TO_BANKED_R1")


def test_live_import_does_not_initialize_legacy_witness_package() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import tac.ddm_costate_organ; print('tac.witness_control' in sys.modules)"),
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert proc.stdout.strip() == "False"


def _write_dv1_receipt(path: Path, *, run_id: str, score_claim: bool = False) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "ddm_description_vocabulary_receipt.v1",
                "run_id": run_id,
                "score_claim": score_claim,
                "execution_allowed": False,
                "research_only": True,
                "promotion_eligible": False,
            }
        )
    )


def test_source_discovery_uses_latest_run_id_and_refuses_authority_drift(tmp_path: Path) -> None:
    research = tmp_path / ".omx" / "research"
    older = "ddm_dv1_description_vocabulary_n600_20260723T010000Z"
    newer = "ddm_dv1_description_vocabulary_n600_20260723T020000Z"
    _write_dv1_receipt(research / older / "receipt.json", run_id=older)
    _write_dv1_receipt(research / newer / "receipt.json", run_id=newer)
    assert discover_sources(tmp_path)["dv1"]["run_id"] == newer

    bad = "ddm_dv1_description_vocabulary_n600_20260723T030000Z"
    _write_dv1_receipt(
        research / bad / "receipt.json",
        run_id=bad,
        score_claim=True,
    )
    with pytest.raises(ValueError, match="authority firewall drift"):
        discover_sources(tmp_path)


def test_resume_refuses_changed_source_hash() -> None:
    report = build_live_ddm_costate(repo_root=REPO)
    stale = json.loads(json.dumps(report["resume_state"]))
    stale["source_hashes"]["g3"] = "0" * 64
    with pytest.raises(ValueError, match="resume source hashes are stale"):
        build_live_ddm_costate(repo_root=REPO, resume_state=stale)


def test_costate_digest_does_not_touch_legacy_run_when_ddm_is_live(monkeypatch) -> None:
    import costate_digest as digest

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy witness surface must not run")

    monkeypatch.setattr(digest, "section_live_run", forbidden)
    monkeypatch.setattr(digest, "section_curriculum_pool", forbidden)
    monkeypatch.setattr(digest, "section_costate_organ", forbidden)
    lines, data = digest.build_digest(include_fm=False)
    assert any(line.startswith("DDM-LIVE reach=") for line in lines)
    assert data["live_run"]["legacy_lookup_performed"] is False
    assert data["costate_organ_v2"]["status"] == "DOMINATED_STALE"
    assert data["curriculum_pool"]["status"] == "DOMINATED_STALE"
