#!/usr/bin/env python3
"""Decisive PR130 sparse-embedding/row-local-Adam MPS parity probe.

Parent mode runs isolated CPU and MPS workers with CPU fallback disabled, then
compares their selected-row updates and optimizer state.  This tool is for a
real Metal host under the governed PyTorch pin; it is not a score measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

FP32_ATOL = 2e-6
FP32_RTOL = 2e-5
REQUIRED_TORCH_VERSION = "2.10.0"
WORKER_MARKER = "PQ1_SPARSE_MPS_WORKER="
STEP_ROW_IDS = ((5, 2, 5, 19, 2), (19, 7, 19, 5, 7))
N_ROWS = 600
ROW_WIDTH = 12
REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker(device_name: str) -> dict[str, Any]:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    sys.path.insert(0, str(REPO_ROOT / "src"))

    import torch

    from tac.pr130_lift.pose.source_loader import load_lifted_module

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this worker runtime")
    device = torch.device(device_name)
    lifted_train = load_lifted_module("train_pose_carrier_full")
    initial = torch.linspace(
        -0.375, 0.625, N_ROWS * ROW_WIDTH, dtype=torch.float32
    ).reshape(N_ROWS, ROW_WIDTH)
    embedding = torch.nn.Embedding(N_ROWS, ROW_WIDTH, sparse=True).to(device)
    with torch.no_grad():
        embedding.weight.copy_(initial.to(device))
    optimizer = lifted_train.RowLocalSparseAdam([embedding.weight], lr=0.03125)
    expected_union = sorted({row for step_rows in STEP_ROW_IDS for row in step_rows})
    per_step: list[dict[str, Any]] = []

    for step_index, raw_rows in enumerate(STEP_ROW_IDS, start=1):
        row_ids = torch.tensor(raw_rows, dtype=torch.long, device=device)
        before = embedding.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        selected = embedding(row_ids)
        loss = selected.square().sum()
        loss.backward()
        if embedding.weight.grad is None or not embedding.weight.grad.is_sparse:
            raise RuntimeError("Embedding(sparse=True) did not emit a sparse gradient")
        gradient = embedding.weight.grad.coalesce()
        observed_rows = gradient.indices()[0]
        expected_rows_cpu = torch.tensor(
            sorted(set(raw_rows)), dtype=torch.long
        )
        observed_rows_cpu = observed_rows.cpu()
        if not torch.equal(observed_rows_cpu, expected_rows_cpu):
            raise RuntimeError(
                f"step {step_index} selected rows mismatch: "
                f"{observed_rows_cpu.tolist()} != {expected_rows_cpu.tolist()}"
            )
        values_before_clip = gradient.values().detach().norm()
        lifted_train.clip_sparse_gradient(embedding.weight, 0.75)
        clipped = embedding.weight.grad.coalesce()
        values_after_clip = clipped.values().detach().norm()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()

        before_cpu = before.cpu()
        after_cpu = embedding.weight.detach().cpu()
        untouched = torch.ones(N_ROWS, dtype=torch.bool)
        untouched.index_fill_(0, expected_rows_cpu, False)
        untouched_bit_identical = torch.equal(
            after_cpu[untouched], before_cpu[untouched]
        )
        if not untouched_bit_identical:
            raise RuntimeError(f"step {step_index} changed an untouched row")
        per_step.append(
            {
                "step": step_index,
                "input_row_ids": list(raw_rows),
                "selected_rows": observed_rows_cpu.tolist(),
                "grad_is_sparse": True,
                "grad_is_coalesced": bool(clipped.is_coalesced()),
                "grad_norm_before_clip": float(values_before_clip.cpu()),
                "grad_norm_after_clip": float(values_after_clip.cpu()),
                "untouched_rows_bit_identical": untouched_bit_identical,
            }
        )

    state = optimizer.state[embedding.weight]
    union_cpu = torch.tensor(expected_union, dtype=torch.long)
    row_step_cpu = state["row_step"].cpu()
    weight_cpu = embedding.weight.detach().cpu()
    exp_avg_cpu = state["exp_avg"].cpu()
    exp_avg_sq_cpu = state["exp_avg_sq"].cpu()
    nonzero_clock_rows = torch.nonzero(row_step_cpu, as_tuple=False).flatten()
    result = {
        "schema": "ddm_pq1_sparse_mps_worker.v1",
        "device": device_name,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "fallback_env": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "embedding": {"rows": N_ROWS, "width": ROW_WIDTH, "sparse": True},
        "steps": per_step,
        "state_rows": expected_union,
        "nonzero_clock_rows": nonzero_clock_rows.cpu().tolist(),
        "row_step": row_step_cpu.index_select(0, union_cpu).tolist(),
        "weight": weight_cpu.index_select(0, union_cpu).tolist(),
        "exp_avg": exp_avg_cpu.index_select(0, union_cpu).tolist(),
        "exp_avg_sq": exp_avg_sq_cpu.index_select(0, union_cpu).tolist(),
    }
    return result


def _run_worker(device_name: str) -> tuple[dict[str, Any], str]:
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    pythonpath = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-device", device_name],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stderr = completed.stderr
    fallback_tokens = (
        "fallback",
        "fall back",
        "not currently supported on the mps backend",
    )
    fallback_lines = [
        line
        for line in stderr.splitlines()
        if any(token in line.casefold() for token in fallback_tokens)
    ]
    if fallback_lines:
        raise RuntimeError(
            f"{device_name} worker emitted CPU-fallback text: {fallback_lines}"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{device_name} worker failed rc={completed.returncode}: {stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(WORKER_MARKER)]
    if len(lines) != 1:
        raise RuntimeError(
            f"{device_name} worker emitted {len(lines)} result markers, expected one"
        )
    return json.loads(lines[0][len(WORKER_MARKER) :]), stderr


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [float(values)]


def _compare_float_payload(
    cpu: dict[str, Any],
    mps: dict[str, Any],
    key: str,
) -> dict[str, float]:
    cpu_values = _flatten(cpu[key])
    mps_values = _flatten(mps[key])
    if len(cpu_values) != len(mps_values):
        raise RuntimeError(f"{key} element-count mismatch")
    max_abs = 0.0
    max_rel = 0.0
    for cpu_value, mps_value in zip(cpu_values, mps_values, strict=True):
        max_abs = max(max_abs, abs(cpu_value - mps_value))
        denominator = max(abs(cpu_value), abs(mps_value), 1e-30)
        max_rel = max(max_rel, abs(cpu_value - mps_value) / denominator)
        if not math.isclose(
            cpu_value, mps_value, rel_tol=FP32_RTOL, abs_tol=FP32_ATOL
        ):
            raise RuntimeError(
                f"{key} parity failed: cpu={cpu_value}, mps={mps_value}, "
                f"atol={FP32_ATOL}, rtol={FP32_RTOL}"
            )
    return {"max_abs": max_abs, "max_rel": max_rel}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    if str(resolved).startswith(("/tmp/", "/private/tmp/")):
        raise ValueError("persisted probe receipts must not be written under /tmp")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_worker_contract(worker: dict[str, Any]) -> None:
    normalized_version = worker["torch_version"].split("+", 1)[0]
    if normalized_version != REQUIRED_TORCH_VERSION:
        raise RuntimeError(
            f"worker torch version {worker['torch_version']!r} is not the "
            f"governed {REQUIRED_TORCH_VERSION!r}"
        )
    if worker["fallback_env"] != "0":
        raise RuntimeError("worker did not disable MPS CPU fallback")


def _parent(out: Path) -> int:
    receipt: dict[str, Any] = {
        "schema": "ddm_pq1_sparse_mps_probe.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "authority": "macOS-MPS port verification only",
        "required_torch_version": REQUIRED_TORCH_VERSION,
        "fp32_tolerance": {"atol": FP32_ATOL, "rtol": FP32_RTOL},
        "fallback_policy": "PYTORCH_ENABLE_MPS_FALLBACK=0 and stderr scan",
        "command": [sys.executable, str(Path(__file__).resolve()), "--out", str(out)],
    }
    try:
        cpu, cpu_stderr = _run_worker("cpu")
        _assert_worker_contract(cpu)
        mps, mps_stderr = _run_worker("mps")
        _assert_worker_contract(mps)
        for worker in (cpu, mps):
            if worker["nonzero_clock_rows"] != worker["state_rows"]:
                raise RuntimeError("optimizer row clocks do not match selected-row union")
            for step_index, step in enumerate(worker["steps"]):
                expected = sorted(set(STEP_ROW_IDS[step_index]))
                if step["selected_rows"] != expected:
                    raise RuntimeError("worker selected-row set mismatch")
                if not step["untouched_rows_bit_identical"]:
                    raise RuntimeError("worker changed an untouched coefficient row")
                if not step["grad_is_sparse"] or not step["grad_is_coalesced"]:
                    raise RuntimeError("worker did not exercise coalesced sparse COO")
        if cpu["state_rows"] != mps["state_rows"]:
            raise RuntimeError("CPU/MPS state-row mismatch")
        if cpu["row_step"] != mps["row_step"]:
            raise RuntimeError("CPU/MPS row-local clocks differ")
        comparisons = {
            key: _compare_float_payload(cpu, mps, key)
            for key in ("weight", "exp_avg", "exp_avg_sq")
        }
        receipt.update(
            {
                "verdict": "PASS",
                "cpu": cpu,
                "mps": mps,
                "cpu_stderr": cpu_stderr,
                "mps_stderr": mps_stderr,
                "comparisons": comparisons,
                "zero_cpu_fallback_warnings": True,
            }
        )
        return_code = 0
    except Exception as error:  # the receipt must preserve a first-class failure
        receipt.update(
            {
                "verdict": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "zero_cpu_fallback_warnings": False,
            }
        )
        return_code = 1
    _atomic_write_json(out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="durable parent-mode JSON receipt")
    mode.add_argument(
        "--worker-device",
        choices=("cpu", "mps"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_device is not None:
        result = _worker(args.worker_device)
        print(WORKER_MARKER + json.dumps(result, sort_keys=True))
        return 0
    return _parent(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
