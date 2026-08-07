# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from experiments import ddm_mx1_pr130_semantic_renderer as mx1


def _write_seg_cache(path: Path) -> torch.Tensor:
    seg = torch.arange(600 * 2 * 3, dtype=torch.int16).reshape(600, 2, 3)
    torch.save({"seg": seg}, path)
    return seg


def _args(tmp_path: Path) -> Namespace:
    input_cache = tmp_path / "input.pt"
    target_cache = tmp_path / "target.pt"
    init = tmp_path / "init.pt"
    input_cache.write_bytes(b"input")
    target_cache.write_bytes(b"target")
    init.write_bytes(b"init")
    return Namespace(
        mode="probe",
        input_cache=input_cache,
        target_cache=target_cache,
        init=init,
        run_dir=tmp_path / "run",
        pairs=32,
        steps=6000,
        lr=2e-7,
        seed=20260806,
        ce_fraction=0.0,
        softplus_fraction=-999.0,
        train_exact_path=False,
        scorer="upstream",
        device="gpu",
        bits=4,
        float_warmup_steps=0,
        eval_every=250,
        checkpoint_every=250,
        mem_budget_gb=12.5,
        mem_probe_steps=3,
        resume_from=None,
        out=tmp_path / "result.json",
    )


def test_load_selected_seg_tokens_matches_full_load_slice(tmp_path: Path) -> None:
    cache = tmp_path / "cache.pt"
    full = _write_seg_cache(cache)
    pair_ids = [0, 7, 31, 599]

    selected, meta = mx1._load_selected_seg_tokens(cache, pair_ids)

    assert torch.equal(selected, full[pair_ids].long())
    assert selected.is_contiguous()
    assert selected.dtype == torch.long
    assert meta["selected_pair_count"] == len(pair_ids)
    assert meta["full_shape_seen"] == [600, 2, 3]


def test_run_mem_probe_writes_peak_receipt_schema(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)

    class FakeMx:
        @staticmethod
        def get_active_memory() -> int:
            return 128

    def fake_run_mlx_train(probe_args, *, memory_probe):
        assert probe_args.steps == 3
        memory_probe.sample("start", mx=FakeMx())
        memory_probe.sample("after_train_step_000003", mx=FakeMx())
        return {
            "schema": "ddm_mx1_mlx_train.v1",
            "status": "passed",
            "steps": probe_args.steps,
            "seconds_per_step": 0.25,
            "stage_checkpoint": str(probe_args.run_dir / "mlx_stage_step000003.npz"),
            "latest_checkpoint": str(probe_args.run_dir / "mlx.latest.npz"),
            "latest_checkpoint_sha256": "0" * 64,
            "load_memory_peak": memory_probe.peak(),
        }

    monkeypatch.setattr(mx1, "run_mlx_train", fake_run_mlx_train)

    result = mx1.run_mem_probe(args)

    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    assert result["status"] == "passed"
    assert receipt["schema"] == mx1.MEM_PROBE_RECEIPT_SCHEMA
    assert receipt["metal_fire_clearance"] is True
    assert receipt["requested_training_steps"] == 3
    assert receipt["peak"]["sample_count"] >= 2
    assert receipt["samples"][-1]["stage"] == "after_train_step_000003"


def test_launch_ticket_requires_mem_probe_and_sequential_scheduling(tmp_path: Path) -> None:
    args = _args(tmp_path)

    ticket = mx1.launch_ticket(args, smoke=None, mlx_probe={"status": "blocked"})

    assert ticket["schema"] == "ddm_mx1_row1_launch_ticket.v2_two_arm"
    assert ticket["mem_probe_receipt_required"] is True
    assert ticket["mem_probe_receipt_path"].endswith("mem_probe_receipt.json")
    assert ticket["scheduling"].startswith("SEQUENTIAL")
    assert "argv_n32" not in ticket
    for key in (
        "argv_n32_arm_cap",
        "argv_n32_arm_veh",
        "argv_n120_arm_cap",
        "argv_n120_arm_veh",
    ):
        assert key in ticket
        assert ticket[key][:2] == [".venv/bin/python", "tools/safe_run.py"]
        assert "--projected-gib" in ticket[key]
        assert "--" in ticket[key]
        assert "--mem-budget-gb" in ticket[key]
        assert "12.5" in ticket[key]
    assert ticket["mem_probe_command"][:4] == [
        ".venv/bin/python",
        "experiments/ddm_mx1_pr130_semantic_renderer.py",
        "--mode",
        "mem-probe",
    ]
    assert "--mem-probe-steps" in ticket["mem_probe_command"]
    assert ticket["safe_run_projection"]["schema"] == "ddm_mx1_row1_safe_run_projection.v1"
    assert ticket["safe_run_projection"]["projected_gib"] >= mx1.METAL_UNKNOWN_MARGIN_GIB
    assert ticket["fire_protocol"]["rr8_f1_refuse_condition"] == "pgrep rc>=2 AND ps rc!=0"


def test_mx1_heavy_mode_refuses_raw_when_enforced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.delenv("TAC_GOVERNED_ADMISSION", raising=False)
    monkeypatch.delenv("TAC_ADMISSION_BYPASS_OK", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--out",
            str(tmp_path / "raw_refused.json"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        mx1.main()

    assert excinfo.value.code == 7


def test_mx1_heavy_mode_governed_env_passes_guard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.setenv("TAC_GOVERNED_ADMISSION", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "torch-smoke",
            "--out",
            str(tmp_path / "governed.json"),
        ],
    )
    monkeypatch.setattr(mx1, "mlx_device_probe", lambda *, device: {"status": "blocked"})
    monkeypatch.setattr(
        mx1,
        "run_torch_smoke",
        lambda args: {"status": "passed", "seconds_per_step": 0.001},
    )
    monkeypatch.setattr(
        mx1,
        "launch_ticket",
        lambda args, smoke, mlx_probe: {"schema": "test_ticket"},
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(mx1, "write_json", lambda path, payload: written.update(payload))

    mx1.main()

    assert written["torch_smoke"]["status"] == "passed"


def test_mx1_light_probe_mode_ungated_when_enforced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.delenv("TAC_GOVERNED_ADMISSION", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "probe",
            "--out",
            str(tmp_path / "probe.json"),
        ],
    )
    monkeypatch.setattr(mx1, "mlx_device_probe", lambda *, device: {"status": "blocked"})
    monkeypatch.setattr(
        mx1,
        "launch_ticket",
        lambda args, smoke, mlx_probe: {"schema": "test_ticket"},
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(mx1, "write_json", lambda path, payload: written.update(payload))

    mx1.main()

    assert written["mode"] == "probe"
