# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import ddm_mx1_pr130_semantic_renderer as mx1


def test_mx1_mem_probe_leaves_launch_ticket_bytes_untouched(tmp_path: Path, monkeypatch) -> None:
    ticket_path = tmp_path / "launch_ticket_v4_fire_guarded.json"
    ticket = {
        "schema": "rr11_regression_ticket",
        "argv_n32_arm_cap": ["--input-cache", "gt_seg_cache.pt", "--target-cache", "gt_seg_cache.pt"],
        "argv_n32_arm_veh": ["--input-cache", "tq1c_seg_cache.pt", "--target-cache", "gt_seg_cache.pt"],
    }
    ticket_path.write_text(json.dumps(ticket, indent=2, sort_keys=True) + "\n")
    before = ticket_path.read_bytes()

    monkeypatch.setattr(mx1, "mlx_device_probe", lambda *, device: {"status": "blocked"})
    monkeypatch.setattr(
        mx1,
        "run_mem_probe",
        lambda args: {
            "schema": "ddm_mx1_mem_probe.v1",
            "status": "passed",
            "receipt_path": str(tmp_path / "mem_probe_receipt.json"),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mem-probe",
            "--run-dir",
            str(tmp_path / "probe_run"),
            "--out",
            str(tmp_path / "mem_probe_result.json"),
            "--launch-ticket-path",
            str(ticket_path),
        ],
    )

    mx1.main()

    assert ticket_path.read_bytes() == before
    reparsed = json.loads(ticket_path.read_text())
    assert "tq1c_seg_cache.pt" in reparsed["argv_n32_arm_veh"]
