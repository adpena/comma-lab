from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.meta_lagrangian_allocator import build_atom_ledger, expected_atom_score_delta
from tac.optimization.mps_research_signal import (
    MPSResearchSignalError,
    build_mps_research_signal_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def _observations() -> list[dict[str, object]]:
    return [
        {
            "family": "arch_shrink",
            "curve_id": "width_x0_4",
            "variant_id": "epoch_000",
            "device": "mps",
            "archive_bytes": 100_000,
            "d_seg_proxy": 0.00090,
            "d_pose_proxy": 0.000050,
            "proxy_loss": 0.55,
            "params": {"width_mult": 0.4, "epoch": 0},
        },
        {
            "family": "arch_shrink",
            "curve_id": "width_x0_4",
            "variant_id": "epoch_010",
            "device": "mps",
            "archive_bytes": 120_000,
            "d_seg_proxy": 0.00072,
            "d_pose_proxy": 0.000031,
            "proxy_loss": 0.20,
            "params": {"width_mult": 0.4, "epoch": 10},
        },
        {
            "family": "arch_shrink",
            "curve_id": "width_x0_4",
            "variant_id": "epoch_020",
            "device": "mps",
            "archive_bytes": 180_000,
            "d_seg_proxy": 0.00070,
            "d_pose_proxy": 0.000030,
            "proxy_loss": 0.18,
            "params": {"width_mult": 0.4, "epoch": 20},
        },
    ]


def test_mps_research_signal_manifest_is_non_promotable_and_curve_shaped() -> None:
    manifest = build_mps_research_signal_manifest(
        _observations(),
        source="fixture",
        run_id="fixture_mps",
        anchor_d_seg=0.00070,
        anchor_d_pose=0.000030,
        anchor_archive_bytes=180_000,
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["evidence_grade"] == "MPS-research-signal"
    assert "candidate_generation_prior" in manifest["device_contract"]["allowed_uses"]
    assert "promotion" in manifest["device_contract"]["forbidden_uses"]
    assert manifest["curve_count"] == 1
    curve = manifest["curves"][0]
    assert curve["metric"] == "proxy_loss"
    assert curve["flattening_detected"] is True
    assert curve["lowest_proxy_observation"]["candidate_generation_only"] is True
    assert len(manifest["meta_lagrangian_atoms"]) == 3
    assert all(atom["proxy_row"] is True for atom in manifest["meta_lagrangian_atoms"])
    assert all(atom["rank_or_kill_eligible"] is False for atom in manifest["meta_lagrangian_atoms"])
    assert all(atom["ready_for_exact_eval_dispatch"] is False for atom in manifest["meta_lagrangian_atoms"])


def test_mps_atoms_remain_proxy_only_in_meta_lagrangian() -> None:
    manifest = build_mps_research_signal_manifest(
        _observations(),
        source="fixture",
        run_id="fixture_mps",
        anchor_d_seg=0.00070,
        anchor_d_pose=0.000030,
        anchor_archive_bytes=180_000,
    )
    ledger = build_atom_ledger(
        manifest["meta_lagrangian_atoms"],
        base_pose_dist=0.000030,
        source="fixture",
    )

    assert ledger["score_claim"] is False
    assert ledger["score_evidence_rankable_count"] == 0
    assert ledger["pareto_eligible_count"] == 0
    assert all(row["proxy_row"] is True for row in ledger["rows"])
    assert all(row["rankable"] is False for row in ledger["rows"])
    assert all(row["planning_priority_rankable"] is False for row in ledger["rows"])
    assert all("proxy_row_not_dispatchable" in row["dispatch_blockers"] for row in ledger["rows"])


def test_mps_evidence_grade_is_proxy_even_if_source_forgets_proxy_flag() -> None:
    row = expected_atom_score_delta(
        {
            "atom_id": "forgotten_proxy_flag",
            "family": "mps_curve",
            "byte_delta": -100,
            "confidence": 0.5,
            "evidence_grade": "MPS-research-signal",
            "raw_equal": True,
            "interaction_assumptions": ["mps_curve_shape_only"],
        },
        base_pose_dist=0.01,
    )

    assert row["proxy_row"] is True
    assert row["score_evidence_rankable"] is False
    assert row["planning_priority_rankable"] is False
    assert "planning_or_proxy_atom_not_score_evidence" in row["score_evidence_contract"]["blockers"]


def test_rejects_non_mps_device_and_partial_anchor() -> None:
    bad = [{**_observations()[0], "device": "cuda"}]
    with pytest.raises(MPSResearchSignalError, match="device must be mps"):
        build_mps_research_signal_manifest(bad, source="fixture", run_id="bad")

    with pytest.raises(MPSResearchSignalError, match="must be supplied together"):
        build_mps_research_signal_manifest(
            _observations(),
            source="fixture",
            run_id="bad_anchor",
            anchor_d_seg=0.1,
        )


def test_mps_research_signal_cli_writes_manifest_and_fail_closed_ledger(tmp_path: Path) -> None:
    observations = tmp_path / "observations.json"
    observations.write_text(json.dumps(_observations(), sort_keys=True), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "atom_ledger.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mps_research_signal_manifest.py"),
            "--observations",
            str(observations),
            "--output",
            str(manifest_path),
            "--atom-ledger-output",
            str(ledger_path),
            "--run-id",
            "fixture_cli",
            "--anchor-d-seg",
            "0.00070",
            "--anchor-d-pose",
            "0.000030",
            "--anchor-archive-bytes",
            "180000",
        ],
        check=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert manifest["evidence_grade"] == "MPS-research-signal"
    assert manifest["score_claim"] is False
    assert ledger["score_evidence_rankable_count"] == 0
    assert ledger["pareto_eligible_count"] == 0
