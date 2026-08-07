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
        allow_soft_mem_limit=False,
        fire_guard_verdict=None,
        launch_ticket_path=tmp_path / "launch_ticket.json",
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
        memory_probe.install_software_budget(
            {
                "software_cap_required": True,
                "software_budget_bytes": int(2 * mx1.GIB),
            }
        )
        memory_probe.sample_and_check("start", mx=FakeMx())
        memory_probe.sample_and_check("after_train_step_000001", mx=FakeMx())
        memory_probe.sample_and_check("after_train_step_000002", mx=FakeMx())
        memory_probe.sample_and_check("after_train_step_000003", mx=FakeMx())
        return {
            "schema": "ddm_mx1_mlx_train.v1",
            "status": "passed",
            "steps": probe_args.steps,
            "seconds_per_step": 0.25,
            "memory_limits": {
                "enforcement": "software_stage_step_cap",
                "software_cap_required": True,
                "software_cap_installed": True,
                "software_budget_bytes": int(2 * mx1.GIB),
                "hard_limit_required": False,
                "hard_limit_satisfied": False,
                "calls": [{"target": "set_memory_limit", "status": "applied", "hard_limit": False}],
            },
            "software_budget": memory_probe.budget_summary(),
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
    assert receipt["memory_limits"]["enforcement"] == "software_stage_step_cap"
    assert receipt["software_budget"]["check_count"] >= 3
    assert receipt["host"]["node"]
    assert receipt["requested_training_steps"] == 3
    assert receipt["peak"]["sample_count"] >= 2
    assert receipt["samples"][-1]["stage"] == "after_train_step_000003"


def test_run_mem_probe_writes_failed_receipt_on_hard_cap_failure(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)

    def fake_run_mlx_train(probe_args, *, memory_probe):
        memory_probe.sample("start")
        memory_probe.sample("after_require_mlx_memory_limit_configuration_failed")
        raise mx1.MemoryLimitConfigurationError("soft MLX limit refused")

    monkeypatch.setattr(mx1, "run_mlx_train", fake_run_mlx_train)

    result = mx1.run_mem_probe(args)

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert result["status"] == "failed"
    assert receipt["status"] == "failed"
    assert receipt["metal_fire_clearance"] is False
    assert receipt["blocker"]["error_type"] == "MemoryLimitConfigurationError"
    assert receipt["blocker"]["last_sample_stage"] == "after_require_mlx_memory_limit_configuration_failed"


def test_run_mem_probe_budget_exceeded_writes_failed_receipt_and_raises(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)

    class FakeMx:
        @staticmethod
        def get_active_memory() -> int:
            return int(4 * mx1.GIB)

    def fake_run_mlx_train(probe_args, *, memory_probe):
        memory_probe.install_software_budget(
            {
                "software_cap_required": True,
                "software_budget_bytes": int(1 * mx1.GIB),
            }
        )
        memory_probe.sample_and_check("after_train_step_000001", mx=FakeMx())

    monkeypatch.setattr(mx1, "run_mlx_train", fake_run_mlx_train)

    with pytest.raises(mx1.MemoryBudgetExceeded):
        mx1.run_mem_probe(args)

    receipt = json.loads((args.run_dir / "mem_probe_receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["metal_fire_clearance"] is False
    assert receipt["blocker"]["error_type"] == "MemoryBudgetExceeded"
    assert receipt["blocker"]["software_budget"]["last_check"]["within_budget"] is False


def test_default_budget_uses_35_percent_and_probe_cap(monkeypatch) -> None:
    monkeypatch.setattr(mx1, "_system_available_bytes", lambda: int(100 * mx1.GIB))

    normal = mx1._derive_mem_budget_gb(None)
    probe = mx1._derive_mem_budget_gb(None, mem_probe=True)

    assert normal["budget_gb"] == 35.0
    assert normal["source"] == "default_35pct_of_available_memory_at_start"
    assert probe["budget_gb"] == 24.0
    assert probe["source"] == "mem_probe_min_24gb_default_35pct_of_available_memory_at_start"


def test_configure_mlx_memory_limits_installs_software_and_wired_caps_for_gpu(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(mx1, "_system_total_bytes", lambda: int(10 * mx1.GIB))

    class FakeMx:
        @staticmethod
        def set_memory_limit(value: int) -> None:
            calls.append(("memory", value))

        @staticmethod
        def set_cache_limit(value: int) -> None:
            calls.append(("cache", value))

        @staticmethod
        def set_wired_limit(value: int) -> None:
            calls.append(("wired", value))

    result = mx1._configure_mlx_memory_limits(
        FakeMx(),
        2.0,
        device="gpu",
        allow_soft_mem_limit=False,
    )

    assert result["enforcement"] == "software_stage_step_cap"
    assert result["software_cap_required"] is True
    assert result["software_cap_installed"] is True
    assert result["hard_limit_required"] is False
    assert result["hard_limit_satisfied"] is False
    assert calls[0] == ("memory", int(2.0 * mx1.GIB))
    assert calls[2] == ("wired", int(2.0 * mx1.GIB))
    assert result["calls"][0]["signature_form"] == "value_only_soft_guideline"


def test_configure_mlx_memory_limits_refuses_gpu_when_budget_cannot_be_derived(monkeypatch) -> None:
    monkeypatch.setattr(mx1, "_system_available_bytes", lambda: None)

    class FakeMx:
        @staticmethod
        def set_memory_limit(value: int) -> None:
            pass

        @staticmethod
        def set_cache_limit(value: int) -> None:
            pass

    with pytest.raises(mx1.MemoryLimitConfigurationError):
        mx1._configure_mlx_memory_limits(
            FakeMx(),
            None,
            device="gpu",
            allow_soft_mem_limit=False,
        )


def test_launch_ticket_requires_mem_probe_and_sequential_scheduling(tmp_path: Path) -> None:
    args = _args(tmp_path)

    ticket = mx1.launch_ticket(args, smoke=None, mlx_probe={"status": "blocked"})

    assert ticket["schema"] == "ddm_mx1_row1_launch_ticket.v4_software_cap_fire_guarded"
    assert ticket["mem_probe_receipt_required"] is True
    assert ticket["mem_probe_receipt_path"].endswith("mem_probe_receipt.json")
    assert ticket["fire_guard_required"] is True
    assert ticket["main_fire_sequence"][0]["step"] == "guard_precheck"
    assert ticket["main_fire_sequence"][1]["step"] == "probe"
    assert ticket["main_fire_sequence"][2]["step"] == "gate"
    assert ticket["main_fire_sequence"][3]["step"] == "fire"
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
        assert "--fire-guard-verdict" in ticket[key]
        assert "--launch-ticket-path" in ticket[key]
        assert "--fire-argv-key" in ticket[key]
        assert key in ticket[key]
        assert key in ticket["fire_guard_commands"]
        assert ticket["fire_guard_commands"][key][:2] == [".venv/bin/python", "tools/mx1_fire_guard.py"]
    assert ticket["mem_probe_command"][:4] == [
        ".venv/bin/python",
        "experiments/ddm_mx1_pr130_semantic_renderer.py",
        "--mode",
        "mem-probe",
    ]
    assert "--mem-probe-steps" in ticket["mem_probe_command"]
    assert "--launch-ticket-path" in ticket["mem_probe_command"]
    assert str(args.target_cache) in ticket["mem_probe_command"]
    assert ticket["safe_run_projection"]["schema"] == "ddm_mx1_row1_safe_run_projection.v1"
    assert ticket["safe_run_projection"]["projected_gib"] >= mx1.METAL_UNKNOWN_MARGIN_GIB
    assert ticket["fire_protocol"]["rr8_f1_refuse_condition"] == "pgrep rc>=2 AND ps rc!=0"
    assert ticket["memory_projection"]["enforcement"] == "software_stage_step_cap"


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


def test_mx1_gpu_train_refuses_without_fire_guard_verdict(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.setenv("TAC_GOVERNED_ADMISSION", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--out",
            str(tmp_path / "guard_refused.json"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        mx1.main()

    assert excinfo.value.code == 9


def test_mx1_gpu_train_refuses_failed_fire_guard_verdict(tmp_path: Path, monkeypatch) -> None:
    verdict = tmp_path / "fire_guard_verdict.json"
    verdict.write_text(
        json.dumps(
            {
                "schema": mx1.MX1_FIRE_GUARD_VERDICT_SCHEMA,
                "status": "failed",
                "reason_code": "mem_probe_receipt_missing",
            }
        )
    )
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.setenv("TAC_GOVERNED_ADMISSION", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--fire-guard-verdict",
            str(verdict),
            "--out",
            str(tmp_path / "guard_refused.json"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        mx1.main()

    assert excinfo.value.code == 9


def test_mx1_gpu_train_refuses_minimal_forged_passed_fire_guard_verdict(tmp_path: Path, monkeypatch) -> None:
    verdict = tmp_path / "fire_guard_verdict.json"
    ticket = tmp_path / "launch_ticket.json"
    receipt = tmp_path / "mem_probe_receipt.json"
    verdict.write_text(
        json.dumps(
            {
                "schema": mx1.MX1_FIRE_GUARD_VERDICT_SCHEMA,
                "status": "passed",
            }
        )
    )
    ticket.write_text("{}")
    receipt.write_text("{}")

    import tools.mx1_fire_guard as guard

    monkeypatch.setattr(
        guard,
        "evaluate_guard",
        lambda ticket_path, argv_key: {
            "schema": mx1.MX1_FIRE_GUARD_VERDICT_SCHEMA,
            "status": "passed",
            "reason_code": "fire_guard_passed",
            "ticket_path": str(ticket),
            "argv_key": "argv_n32_arm_cap",
            "receipt_path": str(receipt),
            "fire_config": {"fire_guard_verdict": str(verdict)},
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        mx1._assert_gpu_fire_guard(
            Namespace(
                fire_guard_verdict=verdict,
                launch_ticket_path=ticket,
                fire_argv_key="argv_n32_arm_cap",
            )
        )

    assert excinfo.value.code == 9


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
