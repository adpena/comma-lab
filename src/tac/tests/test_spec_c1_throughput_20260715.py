from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tac.witness_dsl.activation_ledger import duty_to_measure
from tac.witness_dsl.lever_registry import completeness, resolve_composable_lever
from tac.witness_dsl.spec_c1_throughput_20260715 import (
    ACTIVE_FACTORIES,
    BENCH_SCHEMA,
    PROGRAM_NAME,
    compile_c1_throughput_launch_config,
)
from tac.witness_dsl.typed_config import TypedLever, missing_perf_env_vars

TRAINER = Path("experiments/train_levelset_witness_realized_through_R_mlx.py")


def _flags(cfg) -> dict[str, str | None]:
    return dict(cfg.to_trainer_flags())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_admitted_receipt(path: Path, cfg, cache: Path, sidecar: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": BENCH_SCHEMA,
                "status": "MEASURED_PASS",
                "program_name": PROGRAM_NAME,
                "scientific_argv_sha256": cfg.dsl_program_manifest[
                    "scientific_argv_sha256"
                ],
                "typed_config_hash": cfg.typed.typed_config_hash(),
                "micro_batch_pairs": 1,
                "num_pairs": 24,
                "real_gt_cache": True,
                "gt_cache_sha256": _sha(cache),
                "sr_source_kind": "sidecar",
                "sr_cache_sha256": _sha(sidecar),
                "sr_consumer_exercised": True,
                "seconds_per_epoch": 12.5,
                "epoch_seconds_samples": [12.0, 13.0],
                "peak_rss_gib": 3.25,
                "rss_samples_gib": [3.1, 3.25],
                "bit_identity_spot_check": "PASS",
                "bit_identity_max_abs": 0,
                "bit_identity_compared_values": 1024,
                "identity_authority": "serial numpy-fp32 fixture",
                "hardware": "unit-test-host",
                "rss_method": "fixture",
                "runtime_custody": {"python": "unit-test"},
                "source_git_sha": "a" * 40,
                "measurement_argv": ["unit-test-benchmark"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_c1_compiles_every_admitted_strict_identity_control(tmp_path: Path) -> None:
    cache = tmp_path / "gt_n24.npz"
    sidecar = tmp_path / "gt_n24_sR.npz"
    cache.write_bytes(b"real-cache-fixture")
    sidecar.write_bytes(b"real-sr-fixture")
    receipt = tmp_path / "bench.json"

    blocked = compile_c1_throughput_launch_config(
        str(cache), num_pairs=24, epochs=3000, out_dir=str(tmp_path / "run"),
        bench_receipt_path=receipt,
    )
    flags = _flags(blocked)
    assert blocked.name == PROGRAM_NAME
    assert flags["--margin-saliency-reachability"] is None
    assert flags["--micro-batch-pairs"] == "1"
    assert flags["--safe-compile-regions"] == "none"
    assert flags["--fused-r-kernel"] is None
    assert flags["--cache-gt-skeleton"] is None
    assert flags["--training-torch-threads"] == "1"
    assert flags["--async-verdict"] is None
    assert flags["--verdict-batch"] == "32"
    assert flags["--verdict-pairs"] == "0"
    assert flags["--component-wallclock-telemetry"] is None
    assert flags["--component-wallclock-probe-every"] == "1"
    assert flags["--profile-timing"] is None
    assert blocked.required_perf_env == {
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD": "0",
        "TAC_MLX_CUSTOM_PERSISTENCE_POOL": "1",
    }
    assert blocked.dsl_program_manifest["held"] is True
    assert [row["id"] for row in blocked.dsl_program_manifest["launch_blockers"]] == [
        "C1_COMPOSED_BENCH_NOT_ADMITTED"
    ]

    _write_admitted_receipt(receipt, blocked, cache, sidecar)
    admitted = compile_c1_throughput_launch_config(
        str(cache), num_pairs=24, epochs=3000, out_dir=str(tmp_path / "run"),
        bench_receipt_path=receipt,
    )
    assert admitted.dsl_program_manifest["held"] is False
    assert admitted.dsl_program_manifest["launch_blockers"] == []
    assert admitted.dsl_program_manifest["benchmark_receipt"]["seconds_per_epoch"] == 12.5
    command = admitted.to_command()
    assert command.startswith(
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD=0 TAC_MLX_CUSTOM_PERSISTENCE_POOL=1"
    )
    assert missing_perf_env_vars(command, admitted.required_perf_env) == []


def test_benchmark_receipt_is_bound_to_input_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "gt_n24.npz"
    sidecar = tmp_path / "gt_n24_sR.npz"
    receipt = tmp_path / "bench.json"
    cache.write_bytes(b"cache-v1")
    sidecar.write_bytes(b"sr-v1")
    blocked = compile_c1_throughput_launch_config(
        str(cache), num_pairs=24, epochs=3000, out_dir=str(tmp_path / "run"),
        bench_receipt_path=receipt,
    )
    _write_admitted_receipt(receipt, blocked, cache, sidecar)
    sidecar.write_bytes(b"sr-mutated-after-measurement")

    refused = compile_c1_throughput_launch_config(
        str(cache), num_pairs=24, epochs=3000, out_dir=str(tmp_path / "run"),
        bench_receipt_path=receipt,
    )
    rows = refused.dsl_program_manifest["launch_blockers"]
    assert refused.dsl_program_manifest["held"] is True
    assert rows[0]["id"] == "C1_COMPOSED_BENCH_NOT_ADMITTED"
    assert "S_R-sidecar SHA mismatch" in rows[0]["detail"]


def test_every_c1_factory_is_name_composable_and_mapped() -> None:
    for name in ACTIVE_FACTORIES:
        lever = resolve_composable_lever(name)
        assert lever.name
    coverage = completeness(TRAINER)
    for flag in (
        "--safe-compile-regions",
        "--micro-batch-pairs",
        "--fused-r-kernel",
        "--cache-gt-skeleton",
        "--training-torch-threads",
        "--async-verdict",
        "--verdict-batch",
        "--verdict-pairs",
        "--component-wallclock-telemetry",
        "--component-wallclock-probe-every",
        "--profile-timing",
    ):
        assert flag in coverage.mapped
        assert flag not in coverage.unmapped
        assert flag not in coverage.stale


def test_activation_ledger_reports_duty_for_new_factories(tmp_path: Path) -> None:
    owed = set(duty_to_measure(path=tmp_path / "empty.jsonl"))
    assert set(ACTIVE_FACTORIES).issubset(owed)


def test_runtime_environment_is_typed_and_shell_safe() -> None:
    with pytest.raises(ValidationError):
        TypedLever(name="bad", runtime_environment={"NOT SAFE": "1"})
    with pytest.raises(ValidationError):
        TypedLever(name="bad", runtime_environment={"SAFE": "$(false)"})
