# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.optimization.postdecode_selector_waterfill import (
    build_postdecode_selector_waterfill_plan,
)

REPO = Path(__file__).resolve().parents[3]


def _sweep() -> dict:
    return {
        "axis": "macOS-MPS advisory",
        "archive_bytes": 1000,
        "archive_sha256": "a" * 64,
        "n_pairs": 3,
        "modes": [
            {
                "mode": "none",
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.001,
                "score_proxy": 0.2,
                "pair_posenet_dist": [0.01, 0.01, 0.01],
                "pair_segnet_dist": [0.001, 0.001, 0.001],
            },
            {
                "mode": "even_tile_chroma:3",
                "avg_posenet_dist": 0.008,
                "avg_segnet_dist": 0.001,
                "score_proxy": 0.18,
                "pair_posenet_dist": [0.008, 0.006, 0.007],
                "pair_segnet_dist": [0.001, 0.001, 0.001],
            },
            {
                "mode": "even_rgb_bias:-2,1,1",
                "avg_posenet_dist": 0.009,
                "avg_segnet_dist": 0.002,
                "score_proxy": 0.3,
                "pair_posenet_dist": [0.012, 0.006, 0.009],
                "pair_segnet_dist": [0.002, 0.002, 0.002],
            },
        ],
    }


def test_postdecode_selector_plan_builds_proxy_atoms_and_ledger() -> None:
    plan = build_postdecode_selector_waterfill_plan(
        _sweep(),
        selector_byte_delta=8,
        evidence_source_path="fixture.json",
    )

    assert plan["schema"] == "postdecode_selector_waterfill_plan_v1"
    assert plan["track_id"] == "frame0_postdecode_selector"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["atom_count"] == 3
    selector = next(atom for atom in plan["atoms"] if atom["atom_id"].startswith("postdecode_selector"))
    assert selector["byte_delta"] == 8
    assert selector["selector_indices"] == [1, 1, 1]
    assert selector["selected_mode_counts"] == {"even_tile_chroma:3": 3}
    assert "selector_bytes_charged_before_exact_eval" in selector["interaction_assumptions"]
    assert "fes1_cpu_mps_to_cuda_scorer_device_split_confirmed" in selector["dispatch_blockers"]
    ledger = plan["atom_ledger"]
    assert ledger["score_claim"] is False
    assert ledger["ready_for_exact_eval_dispatch"] is False
    assert ledger["proxy_row_count"] == 3
    assert ledger["family_group_counts"] == {"frame0_postdecode_selector": 3}


def test_postdecode_selector_cuda_axis_rankability_uses_exact_whitelist() -> None:
    good_sweep = {**_sweep(), "axis": "modal-t4-cuda-proxy-prefix"}
    good_plan = build_postdecode_selector_waterfill_plan(
        good_sweep,
        selector_byte_delta=8,
        evidence_source_path="cuda_proxy_fixture.json",
    )

    assert all(atom["rankable"] is True for atom in good_plan["atoms"])
    assert all(atom["axis_transfer_status"] == "cuda_proxy_source" for atom in good_plan["atoms"])

    for axis in (
        "requires_exact_cuda_auth_eval_before_score_use",
        "local-mps-to-cuda-uncalibrated",
    ):
        plan = build_postdecode_selector_waterfill_plan(
            {**_sweep(), "axis": axis},
            selector_byte_delta=8,
            evidence_source_path="unsafe_axis_fixture.json",
        )

        assert all(atom["rankable"] is False for atom in plan["atoms"])
        assert all(
            any(
                blocker in atom["dispatch_blockers"]
                for blocker in (
                    "unknown_proxy_axis_requires_cuda_confirmation",
                    "mps_to_cuda_transfer_uncalibrated",
                )
            )
            for atom in plan["atoms"]
        )


def test_postdecode_selector_atom_ledger_cli(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.json"
    output_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    sweep_path.write_text(json.dumps(_sweep()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_postdecode_selector_atom_ledger.py"),
            "--sweep-json",
            str(sweep_path),
            "--output",
            str(output_path),
            "--atom-ledger-output",
            str(ledger_path),
            "--selector-byte-delta",
            "8",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(output_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert plan["atom_ledger"]["schema_version"] == ledger["schema_version"]
    assert ledger["proxy_rows_dispatchable"] is False
