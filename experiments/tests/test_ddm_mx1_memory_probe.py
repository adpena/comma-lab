# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
        microbatch_pairs=0,
        verdict_batch_size=32,
        mem_budget_gb=12.5,
        mem_probe_steps=3,
        allow_soft_mem_limit=False,
        fire_guard_verdict=None,
        launch_ticket_path=tmp_path / "launch_ticket.json",
        resume_from=None,
        out=tmp_path / "result.json",
    )


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _write_passed_mem_probe_receipt(
    path: Path,
    args: Namespace,
    *,
    pairs: int = 32,
    input_cache: Path | None = None,
    peak_rss_gib: float = 2.0,
    peak_mlx_reported_gib: float = 10.1,
) -> None:
    final_stage = f"after_train_step_{int(args.mem_probe_steps):06d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": mx1.MEM_PROBE_RECEIPT_SCHEMA,
        "status": "passed",
        "axis": "[load-phase memory telemetry; score_claim=false]",
        "score_claim": False,
        "host": mx1._host_fingerprint(),
        "mode": "mem-probe",
        "device_request": args.device,
        "pairs": pairs,
        "requested_training_steps": int(args.mem_probe_steps),
        "mem_budget_gb_arg": args.mem_budget_gb,
        "input_cache": str(input_cache or args.target_cache),
        "target_cache": str(args.target_cache),
        "init_checkpoint": str(args.init),
        "argv_config": {
            "device": args.device,
            "pairs": pairs,
            "lr": float(args.lr),
            "ce_fraction": float(args.ce_fraction),
            "softplus_fraction": float(args.softplus_fraction),
            "bits": int(args.bits),
            "microbatch_pairs": int(getattr(args, "microbatch_pairs", 0) or 0),
            "mem_budget_gb": args.mem_budget_gb,
            "allow_soft_mem_limit": bool(getattr(args, "allow_soft_mem_limit", False)),
            "input_cache": str(input_cache or args.target_cache),
            "target_cache": str(args.target_cache),
            "init": str(args.init),
        },
        "memory_limits": {
            "enforcement": "software_stage_step_cap",
            "software_cap_installed": True,
            "software_budget_bytes": int(24 * mx1.GIB),
        },
        "software_budget": {
            "enforcement": "software_stage_step_cap",
            "check_count": int(args.mem_probe_steps),
            "last_check": {"within_budget": True},
        },
        "samples": [
            {"stage": "after_require_mlx_and_memory_limits", "mlx_active_gib": 0.0},
            {
                "stage": final_stage,
                "mlx_active_gib": 1.0,
                "mlx_cache_gib": 0.25,
                "mlx_peak_gib": peak_mlx_reported_gib,
            },
        ],
        "peak": {
            "peak_rss_gib": peak_rss_gib,
            "peak_mlx_active_gib": 1.0,
            "peak_mlx_cache_gib": 0.25,
            "peak_mlx_reported_gib": peak_mlx_reported_gib,
            "sample_count": 2,
        },
        "clearance_checks": {
            "required_stage": final_stage,
            "has_required_stage_sample": True,
            "has_mlx_allocator_telemetry_at_required_stage": True,
            "software_budget_check_count": int(args.mem_probe_steps),
            "software_budget_within_limit": True,
        },
        "metal_fire_clearance": True,
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


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


def _json_bytes(payload: object) -> np.ndarray:
    return np.frombuffer(json.dumps(payload, sort_keys=True).encode("utf-8"), dtype=np.uint8)


def _write_mlx_layout_npz(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    config: dict[str, object],
    step: int,
    pair_ids: list[int],
) -> None:
    payload: dict[str, np.ndarray] = {
        "meta::config_json": _json_bytes(config),
        "meta::step": np.asarray([step], dtype=np.int64),
        "meta::history_json": _json_bytes(
            [{"step": step, "phase": "expected_flip", "loss": 0.25, "d_seg_batch": 0.125}]
        ),
        "meta::extra_json": _json_bytes(
            {
                "pair_ids": pair_ids,
                "axis": "[macOS-MLX research-signal]",
                "score_claim": False,
            }
        ),
    }
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = np.transpose(arr, (0, 2, 3, 1))
        payload[f"param::{name}"] = np.asarray(arr)
    np.savez(path, **payload)


def test_mlx_npz_checkpoint_maps_back_to_torch_state_dict(tmp_path: Path) -> None:
    lifted = mx1._load_lifted_semantic()
    config = {
        "width": 8,
        "blocks": 2,
        "frame_dim": 4,
        "num_pairs": 6,
        "num_tokens": 5,
        "phase_y": 1,
        "phase_x": 1,
        "temporal_radius": 0,
    }
    torch.manual_seed(17)
    model = lifted.SemanticTokenRenderer(**config)
    checkpoint_path = tmp_path / "tiny_mlx_checkpoint.npz"
    _write_mlx_layout_npz(
        checkpoint_path,
        model.state_dict(),
        config=config,
        step=7,
        pair_ids=[0, 2, 5],
    )

    checkpoint, meta = mx1._load_mlx_npz_checkpoint_for_torch(
        checkpoint_path,
        lifted=lifted,
    )

    assert meta["format"] == "mlx_npz"
    assert meta["step"] == 7
    assert meta["extra"]["pair_ids"] == [0, 2, 5]
    for name, expected in model.state_dict().items():
        assert torch.equal(checkpoint["state_dict"][name], expected), name


def test_mlx_npz_checkpoint_refuses_missing_or_extra_parameters(tmp_path: Path) -> None:
    lifted = mx1._load_lifted_semantic()
    config = {
        "width": 8,
        "blocks": 2,
        "frame_dim": 4,
        "num_pairs": 6,
        "num_tokens": 5,
        "phase_y": 1,
        "phase_x": 1,
        "temporal_radius": 0,
    }
    model = lifted.SemanticTokenRenderer(**config)
    checkpoint_path = tmp_path / "tiny_mlx_checkpoint.npz"
    _write_mlx_layout_npz(
        checkpoint_path,
        model.state_dict(),
        config=config,
        step=7,
        pair_ids=[0, 2, 5],
    )
    with np.load(checkpoint_path, allow_pickle=False) as payload:
        base_payload = {key: np.asarray(payload[key]) for key in payload.files}
    param_keys = sorted(key for key in base_payload if key.startswith("param::"))

    missing_path = tmp_path / "missing_param.npz"
    missing_payload = dict(base_payload)
    missing_payload.pop(param_keys[0])
    np.savez(missing_path, **missing_payload)
    with pytest.raises(ValueError, match="parameter set mismatch"):
        mx1._load_mlx_npz_checkpoint_for_torch(missing_path, lifted=lifted)

    extra_path = tmp_path / "extra_param.npz"
    extra_payload = dict(base_payload)
    extra_payload["param::__unexpected__"] = np.asarray([0.0], dtype=np.float32)
    np.savez(extra_path, **extra_payload)
    with pytest.raises(ValueError, match="parameter set mismatch"):
        mx1._load_mlx_npz_checkpoint_for_torch(extra_path, lifted=lifted)


def test_torch_verdict_history_comparison_refuses_missing_checkpoint_step() -> None:
    with pytest.raises(ValueError, match="no row for step 11"):
        mx1._history_row_at_step(
            [{"step": 10, "phase": "expected_flip", "d_seg_batch": 0.125}],
            11,
        )


def test_run_torch_verdict_receipt_schema_with_checkpoint_pair_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class TinyRenderer(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, tokens: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
            del pair_idx
            return torch.zeros((tokens.shape[0], 3, tokens.shape[1], tokens.shape[2]))

    def fake_render_for_seg(
        model: torch.nn.Module,
        tokens: torch.Tensor,
        idx: torch.Tensor,
        *,
        exact_path: bool,
    ) -> torch.Tensor:
        assert exact_path is True
        return model(tokens, idx)

    class TinySegNet(torch.nn.Module):
        def forward(self, frame: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(
                (frame.shape[0], 5, frame.shape[2], frame.shape[3]),
                dtype=frame.dtype,
            )
            logits[:, 0] = 1.0
            return logits

    fake_lifted = SimpleNamespace(
        SemanticTokenRenderer=TinyRenderer,
        render_for_seg=fake_render_for_seg,
    )
    monkeypatch.setattr(mx1, "_load_lifted_semantic", lambda: fake_lifted)
    monkeypatch.setattr(mx1, "_load_upstream_segnet", lambda device: TinySegNet())
    cache = tmp_path / "cache.pt"
    torch.save({"seg": torch.zeros((600, 2, 3), dtype=torch.int16)}, cache)
    checkpoint_path = tmp_path / "tiny_verdict.npz"
    _write_mlx_layout_npz(
        checkpoint_path,
        TinyRenderer().state_dict(),
        config={"width": 1, "blocks": 1, "frame_dim": 1, "num_pairs": 600, "num_tokens": 5},
        step=11,
        pair_ids=[4, 9, 20],
    )
    args = _args(tmp_path)
    args.init = checkpoint_path
    args.input_cache = cache
    args.target_cache = cache
    args.verdict_batch_size = 2

    receipt = mx1.run_torch_verdict(args)

    assert receipt["schema"] == "ddm_mx1_torch_verdict.v1"
    assert receipt["status"] == "passed"
    assert receipt["axis"] == "[macOS-CPU advisory torch upstream SegNet]"
    assert receipt["score_claim"] is False
    assert receipt["verdict_scope"] == "n32 arm-selection instrument"
    assert receipt["pair_ids"] == [4, 9, 20]
    assert receipt["aggregate_d_seg"] == 0.0
    assert receipt["segnet_batch_size"] == 2
    assert receipt["segnet_chunk_batch_sizes"] == [2, 1]
    assert receipt["comparison_row"]["checkpoint_step"] == 11
    assert receipt["comparison_row"]["mlx_in_training_d_seg_batch"] == 0.125
    assert receipt["comparison_row"]["fp1_flat_paint_floor_d_seg"] == mx1.FP1_FLAT_PAINT_FLOOR_D_SEG
    assert [row["pair_id"] for row in receipt["per_pair_d_seg"]] == [4, 9, 20]


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


def test_run_mem_probe_emits_flushed_load_phase_checkpoint_lines(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    args = _args(tmp_path)

    class FakeMx:
        @staticmethod
        def get_active_memory() -> int:
            return 256

    def fake_run_mlx_train(probe_args, *, memory_probe):
        memory_probe.install_software_budget(
            {
                "software_cap_required": True,
                "software_budget_bytes": int(2 * mx1.GIB),
            }
        )
        memory_probe.sample_and_check("before_test_allocator", mx=FakeMx())
        memory_probe.sample_and_check("after_train_step_000003", mx=FakeMx())
        return {
            "schema": "ddm_mx1_mlx_train.v1",
            "status": "passed",
            "steps": probe_args.steps,
            "seconds_per_step": 0.25,
            "memory_limits": {"enforcement": "software_stage_step_cap"},
            "microbatch_plan": {"mode": "serial_gradient_accumulation"},
            "software_budget": memory_probe.budget_summary(),
            "stage_checkpoint": str(probe_args.run_dir / "mlx_stage_step000003.npz"),
            "latest_checkpoint": str(probe_args.run_dir / "mlx.latest.npz"),
            "latest_checkpoint_sha256": "0" * 64,
            "load_memory_peak": memory_probe.peak(),
        }

    monkeypatch.setattr(mx1, "run_mlx_train", fake_run_mlx_train)

    result = mx1.run_mem_probe(args)
    captured = capsys.readouterr()

    assert result["status"] == "passed"
    lines = [line for line in captured.err.splitlines() if line.startswith("[mx1-load-phase] ")]
    assert len(lines) >= 2
    payloads = [json.loads(line.split("] ", 1)[1]) for line in lines]
    assert {payload["stage"] for payload in payloads} >= {
        "before_test_allocator",
        "after_train_step_000003",
    }
    assert payloads[0]["schema"] == "ddm_mx1_load_phase_checkpoint.v1"
    assert payloads[0]["mlx_active_gib"] == mx1._gib_or_none(256)


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


def test_gpu_train_defaults_to_four_pair_microbatches(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.device = "gpu"

    assert mx1._derive_train_microbatch_pairs(args, total_pairs=32) == 4

    args.device = "cpu"
    assert mx1._derive_train_microbatch_pairs(args, total_pairs=32) == 32

    args.device = "gpu"
    args.microbatch_pairs = 7
    assert mx1._derive_train_microbatch_pairs(args, total_pairs=32) == 7
    assert mx1._derive_train_microbatch_pairs(args, total_pairs=3) == 3


def test_mlx_token_chunks_cover_same_rows_as_full_selected_arrays() -> None:
    conditioning_np = np.arange(7 * 2 * 3, dtype=np.int32).reshape(7, 2, 3)
    target_np = (conditioning_np + 100).copy()
    pair_ids = [3, 9, 10, 17, 24, 31, 42]

    class FakeMx:
        @staticmethod
        def array(value):
            return np.asarray(value).copy()

    conditioning_chunks = []
    target_chunks = []
    idx_chunks = []
    for start, stop in mx1._iter_pair_chunks(len(pair_ids), 3):
        conditioning, target, idx = mx1._mlx_token_chunk(
            FakeMx(),
            conditioning_np,
            target_np,
            pair_ids,
            start,
            stop,
        )
        conditioning_chunks.append(conditioning)
        target_chunks.append(target)
        idx_chunks.append(idx)

    assert np.array_equal(np.concatenate(conditioning_chunks, axis=0), conditioning_np)
    assert np.array_equal(np.concatenate(target_chunks, axis=0), target_np)
    assert np.array_equal(np.concatenate(idx_chunks, axis=0), np.asarray(pair_ids, dtype=np.int32))


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
        "argv_n32_arm_cap_resume",
        "argv_n32_arm_veh_resume",
        "argv_n120_arm_cap_resume",
        "argv_n120_arm_veh_resume",
    ):
        assert key in ticket
        assert ticket[key][:2] == [".venv/bin/python", "tools/safe_run.py"]
        assert "--projected-gib" in ticket[key]
        assert _argv_value(ticket[key], "--projected-gib") == mx1.SAFE_RUN_RECEIPT_SENTINEL
        assert _argv_value(ticket[key], "--rss-mb") == mx1.SAFE_RUN_RECEIPT_SENTINEL
        assert "--status-receipt" in ticket[key]
        assert "--child-pidfile" in ticket[key]
        assert "--" in ticket[key]
        assert "--mem-budget-gb" in ticket[key]
        assert "12.5" in ticket[key]
        assert "--fire-guard-verdict" in ticket[key]
        assert "--launch-ticket-path" in ticket[key]
        assert "--fire-argv-key" in ticket[key]
        assert key in ticket[key]
        assert key in ticket["fire_guard_commands"]
        assert ticket["fire_guard_commands"][key][:2] == [".venv/bin/python", "tools/mx1_fire_guard.py"]
        assert ticket["safe_run_projections"][key]["receipt_path"] == ticket["mem_probe_receipt_paths"][key]
    assert ticket["mem_probe_command"][:4] == [
        ".venv/bin/python",
        "experiments/ddm_mx1_pr130_semantic_renderer.py",
        "--mode",
        "mem-probe",
    ]
    assert "--mem-probe-steps" in ticket["mem_probe_command"]
    assert "--launch-ticket-path" in ticket["mem_probe_command"]
    assert str(args.target_cache) in ticket["mem_probe_command"]
    assert ticket["safe_run_projection"]["schema"] == mx1.SAFE_RUN_RECEIPT_PROJECTION_SCHEMA
    assert ticket["safe_run_projection"]["status"] == "requires_fresh_mem_probe"
    assert ticket["safe_run_projection"]["reason_code"] == "mem_probe_receipt_missing"
    assert ticket["safe_run_projection"]["projected_gib"] == mx1.SAFE_RUN_RECEIPT_SENTINEL
    assert "mx1b_mem_probe_result" not in json.dumps(ticket["safe_run_projection"], sort_keys=True)
    assert ticket["safe_run_projection_policy"]["sentinel"] == mx1.SAFE_RUN_RECEIPT_SENTINEL
    assert "argv_n32_arm_cap_resume" in ticket["resume_protocol"]["resume_keys"]
    assert ticket["fire_protocol"]["rr8_f1_refuse_condition"] == "pgrep rc>=2 AND ps rc!=0"
    assert ticket["memory_projection"]["enforcement"] == "software_stage_step_cap"
    assert len(set(ticket["safe_run_status_receipt_paths"].values())) == 8
    assert len(set(ticket["detached_done_receipt_names"].values())) == 8
    assert ticket["main_fire_sequence"][3]["command"][:2] == [
        ".venv/bin/python",
        "tools/launch_detached_process.py",
    ]


def test_launch_ticket_derives_safe_run_projection_from_passed_receipt(tmp_path: Path) -> None:
    args = _args(tmp_path)
    receipt_path = (
        args.run_dir
        / "launch_arm_cap"
        / "n32_metal"
        / "mem_probe"
        / "mem_probe_receipt.json"
    )
    _write_passed_mem_probe_receipt(
        receipt_path,
        args,
        pairs=32,
        input_cache=args.target_cache,
        peak_rss_gib=2.0,
        peak_mlx_reported_gib=10.1,
    )

    ticket = mx1.launch_ticket(args, smoke=None, mlx_probe={"status": "blocked"})
    projection = ticket["safe_run_projections"]["argv_n32_arm_cap"]

    assert projection["status"] == "passed"
    assert projection["reason_code"] == "receipt_projection_derived"
    assert projection["receipt_path"] == str(receipt_path)
    assert projection["receipt_sha256"] == mx1._sha256_file(receipt_path)
    assert projection["measured_peak_gib"] == 10.1
    assert projection["projected_gib"] == 16
    assert projection["safe_run_rss_mb"] == mx1.SAFE_RUN_RSS_MB_FLOOR
    assert _argv_value(ticket["argv_n32_arm_cap"], "--projected-gib") == "16"
    assert _argv_value(ticket["argv_n32_arm_cap"], "--rss-mb") == str(mx1.SAFE_RUN_RSS_MB_FLOOR)
    assert ticket["safe_run_projections"]["argv_n32_arm_veh"]["status"] == "requires_fresh_mem_probe"


def test_launch_ticket_emits_explicit_resume_keys_and_resume_probe_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)

    ticket = mx1.launch_ticket(args, smoke=None, mlx_probe={"status": "blocked"})
    resume_key = "argv_n32_arm_cap_resume"
    resume_argv = ticket[resume_key]
    resume_inner = resume_argv[resume_argv.index("--") + 1 :]
    expected_run_dir = args.run_dir / "launch_arm_cap" / "n32_metal"
    expected_resume = expected_run_dir / "mlx.latest.npz"

    assert resume_key in ticket["resume_protocol"]["resume_keys"]
    assert "--resume-from" in resume_inner
    assert _argv_value(resume_inner, "--resume-from") == str(expected_resume)
    assert _argv_value(resume_inner, "--run-dir") == str(expected_run_dir)
    assert ticket["mem_probe_receipt_paths"][resume_key] == str(
        expected_run_dir / "mem_probe_resume" / "mem_probe_receipt.json"
    )
    resume_probe = ticket["mem_probe_commands"][resume_key]
    assert "--resume-from" in resume_probe
    assert _argv_value(resume_probe, "--resume-from") == str(expected_resume)
    assert _argv_value(resume_probe, "--run-dir") == str(expected_run_dir / "mem_probe_resume")
    assert ticket["fire_guard_commands"][resume_key][
        ticket["fire_guard_commands"][resume_key].index("--argv-key") + 1
    ] == resume_key


def test_launch_ticket_attempt_unique_receipts_prevent_stale_done_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    stamps = iter([
        "2026-08-07T14:35:58.000000Z",
        "2026-08-07T14:37:05.000000Z",
    ])
    monkeypatch.setattr(mx1, "_utc_now_iso", lambda: next(stamps))

    first = mx1.launch_ticket(args, smoke=None, mlx_probe={"status": "blocked"})
    second = mx1.launch_ticket(args, smoke=None, mlx_probe={"status": "blocked"})
    key = "argv_n32_arm_cap"

    assert first["ticket_attempt_id"] != second["ticket_attempt_id"]
    assert first["detached_done_receipt_names"][key] != second["detached_done_receipt_names"][key]
    assert first["safe_run_status_receipt_paths"][key] != second["safe_run_status_receipt_paths"][key]
    assert first["safe_run_child_pidfile_paths"][key] != second["safe_run_child_pidfile_paths"][key]
    assert first["detached_done_receipt_names"][key] not in set(
        second["detached_done_receipt_names"].values()
    )
    assert _argv_value(second[key], "--status-receipt") == second["safe_run_status_receipt_paths"][key]
    assert _argv_value(second[key], "--child-pidfile") == second["safe_run_child_pidfile_paths"][key]


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


def test_mx1_torch_verdict_skips_mlx_probe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "torch-verdict",
            "--out",
            str(tmp_path / "verdict.json"),
        ],
    )

    def fail_mlx_probe(*, device: str) -> dict[str, object]:
        raise AssertionError(f"torch-verdict must not probe MLX/Metal: {device}")

    monkeypatch.setattr(mx1, "mlx_device_probe", fail_mlx_probe)
    monkeypatch.setattr(mx1, "run_torch_verdict", lambda args: {"status": "passed"})
    written: dict[str, object] = {}
    monkeypatch.setattr(mx1, "write_json", lambda path, payload: written.update(payload))

    mx1.main()

    assert written["mode"] == "torch-verdict"
    assert written["status"] == "passed"
    assert written["mlx_probe"]["status"] == "not_run"


def test_mx1_torch_facets_skips_mlx_probe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "torch-facets",
            "--out",
            str(tmp_path / "facets.json"),
        ],
    )

    def fail_mlx_probe(*, device: str) -> dict[str, object]:
        raise AssertionError(f"torch-facets must not probe MLX/Metal: {device}")

    monkeypatch.setattr(mx1, "mlx_device_probe", fail_mlx_probe)
    monkeypatch.setattr(mx1, "run_torch_facets", lambda args: {"status": "passed"})
    written: dict[str, object] = {}
    monkeypatch.setattr(mx1, "write_json", lambda path, payload: written.update(payload))

    mx1.main()

    assert written["mode"] == "torch-facets"
    assert written["status"] == "passed"
    assert written["mlx_probe"]["status"] == "not_run"


def test_mx1_margin_histogram_uses_fixed_bins() -> None:
    hist = mx1._margin_histogram_empty()
    margins = torch.tensor([[[0.0, 0.049, 0.05, 0.249, 0.5, 2.0]]])
    mask = torch.tensor([[[True, True, True, True, True, False]]])

    mx1._margin_histogram_update(hist, margins, mask)
    finalized = mx1._margin_histogram_finalize(hist)

    assert finalized["total"] == 5
    assert [bucket["count"] for bucket in finalized["bins"]] == [2, 1, 1, 0, 1]


def test_mx1_boundary_band_mask_marks_four_neighbors() -> None:
    labels = torch.tensor(
        [
            [
                [0, 0, 1],
                [0, 0, 1],
                [2, 2, 2],
            ]
        ]
    )

    mask = mx1._boundary_band_mask(labels)

    assert mask.tolist() == [
        [
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ]
    ]


def test_mx1_class_accumulators_report_both_directions() -> None:
    target = torch.tensor([[[0, 0, 1, 1, 4]]])
    pred = torch.tensor([[[0, 1, 1, 2, 0]]])
    accum = mx1._new_class_accumulators()

    mx1._update_class_accumulators(accum, target, pred)
    finalized = mx1._finalize_class_accumulators(accum)
    by_name = {row["class_name"]: row for row in finalized["per_class_d_seg"]}

    assert by_name["Road"]["gt_sites"] == 2
    assert by_name["Road"]["gt_mispredicted"] == 1
    assert by_name["Road"]["pred_sites"] == 2
    assert by_name["Road"]["pred_false_positive"] == 1
    assert by_name["Lane"]["gt_sites"] == 2
    assert by_name["Lane"]["gt_mispredicted"] == 1
    assert by_name["Lane"]["pred_sites"] == 2
    assert by_name["Lane"]["pred_false_positive"] == 1
    assert finalized["class_order"][4] == {"class_id": 4, "class_name": "MyCar"}


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
