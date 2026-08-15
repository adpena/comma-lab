# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _driver():
    return _load_module("ddm_mx1_pr130_semantic_renderer_test", REPO / "experiments/ddm_mx1_pr130_semantic_renderer.py")


def _bench():
    return _load_module("wc1_wallclock_bench_test", REPO / "experiments/ddm_wc1_wallclock_bench.py")


def test_wc1_thread_pin_sets_env_without_real_torch(monkeypatch) -> None:
    driver = _driver()

    class FakeTorch:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def set_num_threads(self, value: int) -> None:
            self.calls.append(("set_num_threads", value))

        def set_num_interop_threads(self, value: int) -> None:
            self.calls.append(("set_num_interop_threads", value))

    fake = FakeTorch()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MLX_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    report = driver._apply_perf_thread_pin("one", torch_module=fake)

    assert report["applied"] is True
    assert fake.calls == [("set_num_threads", 1), ("set_num_interop_threads", 1)]
    assert report["env"]["OMP_NUM_THREADS"]["after"] == "1"
    assert report["env"]["MLX_NUM_THREADS"]["after"] == "1"


def test_wc1_microbatch_full_batch_variant_is_explicit() -> None:
    driver = _driver()

    auto_plan = driver._derive_train_microbatch_plan(
        argparse.Namespace(microbatch_pairs=0, microbatch_policy="auto", device="gpu"),
        total_pairs=32,
    )
    assert auto_plan["microbatch_pairs"] == 4
    assert auto_plan["source"] == "wc2_auto_empirical_wallclock_anchor"
    assert driver._derive_train_microbatch_pairs(
        argparse.Namespace(microbatch_pairs=0, microbatch_policy="auto", device="gpu"),
        total_pairs=32,
    ) == 4
    assert driver._derive_train_microbatch_pairs(
        argparse.Namespace(microbatch_pairs=32, microbatch_policy="auto", device="gpu"),
        total_pairs=32,
    ) == 32
    assert driver._derive_train_microbatch_pairs(
        argparse.Namespace(microbatch_pairs=0, microbatch_policy="full", device="gpu"),
        total_pairs=32,
    ) == 32


def test_wc1_train_compute_dtype_casts_parameter_tree() -> None:
    driver = _driver()

    class FakeArray:
        shape = (1,)

        def __init__(self, dtype: str = "fp32") -> None:
            self.dtype = dtype

        def astype(self, dtype: str) -> FakeArray:
            return FakeArray(dtype)

    params = {"weight": FakeArray(), "literal": 3}
    casted = driver._cast_mlx_parameter_tree(
        lambda tree: list(tree.items()),
        dict,
        params,
        "fp16",
    )

    assert casted["weight"].dtype == "fp16"
    assert casted["literal"] == 3
    assert driver._cast_mlx_parameter_tree(lambda tree: list(tree.items()), dict, params, None) is params


def test_wc1_compile_flag_wraps_loss_function() -> None:
    driver = _driver()

    class FakeMx:
        def __init__(self) -> None:
            self.called = False

        def compile(self, fn):
            self.called = True

            def wrapped(value):
                return fn(value) + 1

            return wrapped

    mx = FakeMx()

    def fn(value: int) -> int:
        return value * 2

    assert driver._maybe_compile_loss_function(mx, fn, enabled=False) is fn
    wrapped = driver._maybe_compile_loss_function(mx, fn, enabled=True)
    assert mx.called is True
    assert wrapped(3) == 7


def test_wc1_mem_probe_resume_horizon_advances_from_checkpoint(tmp_path: Path) -> None:
    driver = _driver()
    ckpt = tmp_path / "mlx.latest.npz"
    with ckpt.open("wb") as handle:
        np.savez(handle, **{"meta::step": np.asarray([4100], dtype=np.int64)})

    args = argparse.Namespace(
        mode="mem-probe",
        mem_probe_steps=2,
        resume_from=ckpt,
        eval_every=250,
        checkpoint_every=250,
    )
    probe_args = driver._mem_probe_args(args)

    assert probe_args.steps == 4102
    assert probe_args.mem_probe_resume_base_step == 4100
    assert probe_args.eval_every == 1
    assert probe_args.checkpoint_every == 2


def test_wc1_bench_plan_builds_guarded_variant_ticket(tmp_path: Path) -> None:
    bench = _bench()
    resume = tmp_path / "mlx.latest.npz"
    with resume.open("wb") as handle:
        np.savez(handle, **{"meta::step": np.asarray([3250], dtype=np.int64)})
    source_ticket = tmp_path / "source_ticket.json"
    run_dir = tmp_path / "source_run"
    source_ticket.write_text(
        json.dumps(
            {
                "argv_n32_arm_cap_resume": [
                    ".venv/bin/python",
                    "tools/safe_run.py",
                    "--rss-mb",
                    "45000",
                    "--",
                    ".venv/bin/python",
                    "experiments/ddm_mx1_pr130_semantic_renderer.py",
                    "--mode",
                    "mlx-train",
                    "--device",
                    "gpu",
                    "--pairs",
                    "32",
                    "--lr",
                    "2e-07",
                    "--ce-fraction",
                    "0.0",
                    "--softplus-fraction",
                    "-999.0",
                    "--bits",
                    "4",
                    "--seed",
                    "20260806",
                    "--input-cache",
                    str(tmp_path / "gt.pt"),
                    "--target-cache",
                    str(tmp_path / "gt.pt"),
                    "--init",
                    str(tmp_path / "init.pt"),
                    "--run-dir",
                    str(run_dir),
                    "--resume-from",
                    str(resume),
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        source_ticket=source_ticket,
        source_argv_key="argv_n32_arm_cap_resume",
        resume_from=None,
        output_root=tmp_path / "bench",
        ticket_out=tmp_path / "wc1_ticket.json",
        variants="baseline,threads,batched,compile,fp16-train",
        pairs=32,
        seed=20260806,
        bench_steps=5,
        mem_probe_steps=2,
        rss_mb=45_000,
        timeout_s=600.0,
        projected_gib=24.0,
        concurrent_verdict_batch_size=16,
        ane_parity_pairs=32,
        coreml_compute_units="CPU_AND_NE",
    )

    ticket, rows = bench.build_plan(args)

    assert ticket["schema"] == "ddm_wc1_wallclock_bench_ticket.v1"
    assert ticket["resume_step"] == 3250
    assert len(rows) == 5
    assert rows[1]["variant"] == "threads"
    assert rows[1]["env_overrides"]["OMP_NUM_THREADS"] == "1"
    assert "--microbatch-pairs" in rows[2]["train_command"]
    assert "--compile-train-loss" in rows[3]["train_command"]
    assert "--train-compute-dtype" in rows[4]["train_command"]
    assert ticket["mem_probe_receipt_paths"]["argv_wc1_fp16_train"].endswith("mem_probe_receipt.json")


def test_wc2_bench_plan_adds_ram_concurrent_and_ane_variants(tmp_path: Path) -> None:
    bench = _bench()
    resume = tmp_path / "mlx.latest.npz"
    with resume.open("wb") as handle:
        np.savez(handle, **{"meta::step": np.asarray([6000], dtype=np.int64)})
    source_ticket = tmp_path / "source_ticket.json"
    run_dir = tmp_path / "source_run"
    source_ticket.write_text(
        json.dumps(
            {
                "argv_n32_arm_cap": [
                    ".venv/bin/python",
                    "experiments/ddm_mx1_pr130_semantic_renderer.py",
                    "--mode",
                    "mlx-train",
                    "--device",
                    "gpu",
                    "--pairs",
                    "32",
                    "--lr",
                    "2e-07",
                    "--ce-fraction",
                    "0.0",
                    "--softplus-fraction",
                    "-999.0",
                    "--bits",
                    "4",
                    "--seed",
                    "20260806",
                    "--input-cache",
                    str(tmp_path / "veh.pt"),
                    "--target-cache",
                    str(tmp_path / "gt.pt"),
                    "--init",
                    str(tmp_path / "init.pt"),
                    "--run-dir",
                    str(run_dir),
                    "--resume-from",
                    str(resume),
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        source_ticket=source_ticket,
        source_argv_key="argv_n32_arm_cap",
        resume_from=None,
        output_root=tmp_path / "bench",
        ticket_out=tmp_path / "wc2_ticket.json",
        variants="ram-cache,derived-microbatch-4,concurrent-cpu-verdict,ane-verdict",
        pairs=32,
        seed=20260806,
        bench_steps=5,
        mem_probe_steps=2,
        rss_mb=45_000,
        timeout_s=600.0,
        projected_gib=24.0,
        concurrent_verdict_batch_size=8,
        ane_parity_pairs=32,
        coreml_compute_units="CPU_AND_NE",
    )

    ticket, rows = bench.build_plan(args)

    assert [row["variant"] for row in rows] == [
        "ram-cache",
        "derived-microbatch-4",
        "concurrent-cpu-verdict",
        "ane-verdict",
    ]
    assert "--cache-residency" in rows[0]["train_command"]
    assert "--microbatch-policy" in rows[1]["train_command"]
    assert rows[2]["cpu_verdict_command"][rows[2]["cpu_verdict_command"].index("--mode") + 1] == "torch-verdict"
    assert rows[3]["ane_parity_command"][rows[3]["ane_parity_command"].index("--mode") + 1] == "coreml-segnet-parity"
    assert "argv_wc1_ane_verdict_ane_parity" in ticket


def test_wc1_fire_guard_compares_throughput_flags(tmp_path: Path) -> None:
    from tools import mx1_fire_guard as guard

    input_cache = tmp_path / "gt.pt"
    init = tmp_path / "init.pt"
    input_cache.write_bytes(b"x")
    init.write_bytes(b"x")
    fire = guard._parsed_fire_config(
        [
            ".venv/bin/python",
            "tools/safe_run.py",
            "--rss-mb",
            "45000",
            "--",
            ".venv/bin/python",
            "experiments/ddm_mx1_pr130_semantic_renderer.py",
            "--mode",
            "mlx-train",
            "--device",
            "gpu",
            "--pairs",
            "32",
            "--lr",
            "2e-07",
            "--ce-fraction",
            "0.0",
            "--softplus-fraction",
            "-999.0",
            "--bits",
            "4",
            "--input-cache",
            str(input_cache),
            "--target-cache",
            str(input_cache),
            "--init",
            str(init),
            "--train-compute-dtype",
            "fp16",
            "--compile-train-loss",
            "--perf-thread-pin",
            "one",
            "--cache-residency",
            "ram-full",
        ]
    )
    receipt = guard._receipt_config(
        {
            "device_request": "gpu",
            "pairs": 32,
            "argv_config": {
                "device": "gpu",
                "pairs": 32,
                "lr": 2e-7,
                "ce_fraction": 0.0,
                "softplus_fraction": -999.0,
                "bits": 4,
                "microbatch_pairs": 0,
                "microbatch_policy": "auto",
                "cache_residency": "ram-full",
                "mem_budget_gb": None,
                "allow_soft_mem_limit": False,
                "input_cache": str(input_cache),
                "target_cache": str(input_cache),
                "init": str(init),
                "train_compute_dtype": "fp16",
                "compile_train_loss": True,
                "perf_thread_pin": "one",
            },
            "train_result_summary": {"microbatch_plan": {"microbatch_pairs": 4}},
        }
    )

    ok, reason, detail = guard._validate_config_match(fire, receipt)
    assert ok is True, detail
    receipt["train_compute_dtype"] = "fp32"
    ok, reason, detail = guard._validate_config_match(fire, receipt)
    assert ok is False
    assert reason == "receipt_config_mismatch"
    assert "train_compute_dtype" in detail["mismatches"]
    receipt["train_compute_dtype"] = "fp16"
    receipt["cache_residency"] = "selected"
    ok, reason, detail = guard._validate_config_match(fire, receipt)
    assert ok is False
    assert "cache_residency" in detail["mismatches"]
