#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parity-gate and benchmark the custom compact sparse-adjoint Metal kernel.

The parent mode is staged and resumable:

1. run the same four NumPy-fp32 parity configurations in ten fresh processes;
2. require bit identity per configuration and one output hash across processes;
3. replay all 125 frozen-SegNet convolution shapes on Metal, using the sealed
   #486 family support fractions and the existing #212 dense input-adjoint
   kernel as the baseline;
4. measure the rank-2/K=2 state-stable-Jacobian composition on one real decoder
   shape; and
5. atomically seal a small receipt.

This is local compute research only.  It does not train, evaluate an archive,
mutate a live run, or claim a score.  The support masks in the wall replay are
deterministic evenly-spaced masks calibrated to the sealed family fractions;
they are not a learned predictor and are not presented as fidelity evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.local_acceleration.metal_sparse_adjoint import (  # noqa: E402
    Conv2DAdjointSpec,
    build_sparse_spatial_plan,
    compact_cotangent,
    dense_conv2d_input_adjoint_numpy_fp32,
    make_sparse_conv2d_input_adjoint_metal,
    sparse_conv2d_input_adjoint_numpy_fp32,
)

SCHEMA = "custom_sparse_adjoint_kernel_benchmark.v1"
LANE_ID = "custom_sparse_adjoint_kernel"
AXIS = "[macOS-MLX research-signal; NumPy-fp32 parity authority; non-promotable MEANS]"
DEFAULT_OUTPUT = REPO / "experiments/results/custom_sparse_adjoint_kernel_20260713"
DEFAULT_STATIC_RECEIPT = (
    REPO / ".omx/research/custom_sparse_adjoint_kernel_static_receipt_20260713.json"
)
PREDECESSOR_RECEIPT = (
    REPO
    / "experiments/results/p0_sparse_adjoint_costate_vjp_20260713"
    / "measurement_receipt.json"
)
PREDECESSOR_MEMO = REPO / ".omx/research/p0_sparse_adjoint_costate_vjp_20260713.md"
K2_SMOKE_RECEIPT = (
    REPO
    / "experiments/results/p0_costate_reuse_k2_smoke_v2_20260713"
    / "measurement_receipt.json"
)
KERNEL_MODULE = REPO / "src/tac/local_acceleration/metal_sparse_adjoint.py"

SEED = 486
CROSS_PROCESS_COUNT = 10
DERIVED_WHOLE_NETWORK_CEILING_X = 2.208577465069467
ORACLE_20PCT_RELATIVE_L2 = 0.026206284007981848
SOURCE_MARGIN_20PCT_RELATIVE_L2 = 0.5401736369574366


class BenchError(RuntimeError):
    """A source-custody, parity, or benchmark gate failed closed."""


def _utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BenchError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _git_status() -> list[str]:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()


def _contract() -> dict[str, Any]:
    sources = [Path(__file__).resolve(), KERNEL_MODULE, PREDECESSOR_RECEIPT, K2_SMOKE_RECEIPT]
    source_custody = {
        str(path.relative_to(REPO)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sources
    }
    payload = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "seed": SEED,
        "cross_process_count": CROSS_PROCESS_COUNT,
        "parity_rule": "bit-identical to NumPy-fp32 dense-on-support for every config",
        "wall_replay": "all frozen-SegNet Conv2d shapes; sealed #486 family support fractions",
        "whole_network_derived_ceiling_x": DERIVED_WHOLE_NETWORK_CEILING_X,
        "source_custody": source_custody,
        "training": False,
        "paid_dispatch": False,
        "live_run_mutation": False,
    }
    payload["contract_sha256"] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return payload


def _even_support(hw: tuple[int, int], fraction: float) -> np.ndarray:
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be finite and in (0,1]")
    size = hw[0] * hw[1]
    count = min(size, max(1, round(size * fraction)))
    indices = (np.arange(count, dtype=np.int64) * size) // count
    if np.unique(indices).size != count:
        raise BenchError("even-support construction produced duplicate indices")
    mask = np.zeros(size, dtype=np.bool_)
    mask[indices] = True
    return mask.reshape(hw)


def _parity_cases() -> tuple[tuple[str, Conv2DAdjointSpec, float, int], ...]:
    return (
        (
            "standard_3x3_rank1",
            Conv2DAdjointSpec(
                input_hw=(13, 15),
                output_hw=(13, 15),
                cin=7,
                cout=9,
                kernel_hw=(3, 3),
                padding_hw=(1, 1),
            ),
            0.2,
            1,
        ),
        (
            "grouped_3x3_stride2_rank2",
            Conv2DAdjointSpec(
                input_hw=(14, 16),
                output_hw=(7, 8),
                cin=8,
                cout=12,
                kernel_hw=(3, 3),
                stride_hw=(2, 2),
                padding_hw=(1, 1),
                groups=4,
            ),
            0.25,
            2,
        ),
        (
            "depthwise_5x5_stride2_rank3",
            Conv2DAdjointSpec(
                input_hw=(15, 17),
                output_hw=(8, 9),
                cin=6,
                cout=6,
                kernel_hw=(5, 5),
                stride_hw=(2, 2),
                padding_hw=(2, 2),
                groups=6,
            ),
            0.5,
            3,
        ),
        (
            "pointwise_full_support_rank8",
            Conv2DAdjointSpec(
                input_hw=(9, 11),
                output_hw=(9, 11),
                cin=11,
                cout=13,
                kernel_hw=(1, 1),
            ),
            1.0,
            8,
        ),
    )


def _run_parity_worker(output: Path) -> dict[str, Any]:
    import mlx.core as mx

    contract = _contract()
    rows = []
    combined = hashlib.sha256()
    for case_index, (name, spec, fraction, rank) in enumerate(_parity_cases()):
        rng = np.random.default_rng(SEED + case_index)
        mask = _even_support(spec.output_hw, fraction)
        plan = build_sparse_spatial_plan(mask, spec)
        weight = rng.standard_normal(spec.weight_shape).astype(np.float32)
        dense_cotangent = rng.standard_normal((rank, *spec.output_hw, spec.cout)).astype(np.float32)
        dense_cotangent *= mask[None, :, :, None]
        compact = compact_cotangent(dense_cotangent, plan)
        dense = dense_conv2d_input_adjoint_numpy_fp32(dense_cotangent, weight, spec)
        restricted = dense.reshape(rank, -1, spec.cin)[:, plan.input_indices, :]
        compact_numpy = sparse_conv2d_input_adjoint_numpy_fp32(compact, weight, plan)
        if not np.array_equal(restricted, compact_numpy):
            raise BenchError(f"{name}: compact NumPy implementation diverged from dense authority")
        executable = make_sparse_conv2d_input_adjoint_metal(compact, weight, plan)
        observed_mx = executable()
        mx.eval(observed_mx)
        observed = np.asarray(observed_mx)
        mismatch_count = int(np.count_nonzero(observed.view(np.uint32) != restricted.view(np.uint32)))
        max_abs = float(np.max(np.abs(observed.astype(np.float64) - restricted.astype(np.float64))))
        output_sha = _array_sha256(observed)
        combined.update(name.encode())
        combined.update(bytes.fromhex(output_sha))
        rows.append(
            {
                "config": name,
                "spec": spec.to_dict(),
                "rank": rank,
                "plan": plan.to_dict(),
                "numpy_dense_restricted_sha256": _array_sha256(restricted),
                "metal_output_sha256": output_sha,
                "bit_mismatch_count": mismatch_count,
                "max_abs_error": max_abs,
                "parity": "BIT_IDENTICAL" if mismatch_count == 0 else "FAIL",
            }
        )
    result = {
        "schema": "custom_sparse_adjoint_cross_process_parity.v1",
        "completed_at_utc": _utc(),
        "pid": os.getpid(),
        "contract_sha256": contract["contract_sha256"],
        "device": mx.metal.device_info(),
        "configs": rows,
        "combined_output_sha256": combined.hexdigest(),
        "all_bit_identical": all(row["bit_mismatch_count"] == 0 for row in rows),
    }
    _atomic_json(output, result)
    return result


def _run_static_verification(output: Path) -> dict[str, Any]:
    """Seal everything verifiable without treating an inaccessible GPU as evidence."""

    contract = _contract()
    parity_rows = []
    for case_index, (name, spec, fraction, rank) in enumerate(_parity_cases()):
        trial_hashes = []
        for trial in range(CROSS_PROCESS_COUNT):
            rng = np.random.default_rng(SEED + 100 * case_index + trial)
            mask = _even_support(spec.output_hw, fraction)
            plan = build_sparse_spatial_plan(mask, spec)
            weight = rng.standard_normal(spec.weight_shape).astype(np.float32)
            dense_cotangent = rng.standard_normal(
                (rank, *spec.output_hw, spec.cout)
            ).astype(np.float32)
            dense_cotangent *= mask[None, :, :, None]
            compact = compact_cotangent(dense_cotangent, plan)
            dense = dense_conv2d_input_adjoint_numpy_fp32(dense_cotangent, weight, spec)
            restricted = dense.reshape(rank, -1, spec.cin)[:, plan.input_indices, :]
            compact_numpy = sparse_conv2d_input_adjoint_numpy_fp32(compact, weight, plan)
            if not np.array_equal(restricted, compact_numpy):
                raise BenchError(f"{name}: NumPy compact implementation diverged")
            trial_hashes.append(_array_sha256(compact_numpy))
        parity_rows.append(
            {
                "config": name,
                "spec": spec.to_dict(),
                "rank": rank,
                "in_process_deterministic_trials": CROSS_PROCESS_COUNT,
                "all_numpy_compact_bit_identical_to_numpy_dense_on_support": True,
                "trial_output_sha256": trial_hashes,
                "authority_limit": (
                    "CPU implementation proof only; not the required N=10 cross-process "
                    "Metal-kernel parity gate"
                ),
            }
        )

    cases = _load_real_conv_cases()
    fractions = _family_support_fractions()
    family_counts: dict[str, int] = {}
    totals = {
        "dense_nominal_flops": 0,
        "sparse_nominal_flops": 0,
        "dense_exact_valid_tap_flops": 0,
        "sparse_exact_valid_tap_flops": 0,
    }
    for case in cases:
        family_counts[case.family] = family_counts.get(case.family, 0) + 1
        plan = build_sparse_spatial_plan(
            _even_support(case.spec.output_hw, fractions[case.family]), case.spec
        )
        totals["dense_nominal_flops"] += _nominal_flops(
            case.spec, case.spec.output_hw[0] * case.spec.output_hw[1]
        )
        totals["sparse_nominal_flops"] += _nominal_flops(
            case.spec, plan.output_site_count
        )
        totals["dense_exact_valid_tap_flops"] += 2 * plan.dense_fma_count
        totals["sparse_exact_valid_tap_flops"] += 2 * plan.sparse_fma_count
    totals["nominal_arithmetic_ceiling_x"] = (
        totals["dense_nominal_flops"] / totals["sparse_nominal_flops"]
    )
    totals["exact_valid_tap_arithmetic_ceiling_x"] = (
        totals["dense_exact_valid_tap_flops"]
        / totals["sparse_exact_valid_tap_flops"]
    )
    totals["absolute_nominal_replay_gap_from_flagship_ceiling_x"] = abs(
        totals["nominal_arithmetic_ceiling_x"] - DERIVED_WHOLE_NETWORK_CEILING_X
    )

    metal_probe: dict[str, Any]
    try:
        import mlx.core as mx

        default_device = str(mx.default_device())
        probe = mx.array([1.0], dtype=mx.float32)
        mx.eval(probe)
        metal_probe = {
            "available": True,
            "default_device": default_device,
            "device_info": mx.metal.device_info(),
        }
    except Exception as exc:
        metal_probe = {
            "available": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }

    status = (
        "READY_FOR_N10_METAL_PARITY_AND_WALL_BENCH"
        if metal_probe["available"]
        else "BLOCKED_NO_METAL_DEVICE_IN_EXECUTION_SANDBOX"
    )
    receipt = {
        "schema": "custom_sparse_adjoint_kernel_static_verification.v1",
        "completed_at_utc": _utc(),
        "lane_id": LANE_ID,
        "axis": AXIS,
        "status": status,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_delta": "NONE",
        "contract": contract,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_head": _git_head(),
            "argv": list(sys.argv),
        },
        "numpy_fp32_static_parity": {
            "verdict": "BIT_IDENTICAL_DENSE_ON_SUPPORT",
            "configs": parity_rows,
        },
        "metal_parity": {
            "verdict": "UNMEASURED_BLOCKED" if not metal_probe["available"] else "OWED",
            "required_gate": "N=10 cross-process, per-config bit identity",
            "probe": metal_probe,
        },
        "real_segnet_static_replay": {
            "conv_case_count": len(cases),
            "family_counts": family_counts,
            "family_support_fractions": fractions,
            "support_geometry": (
                "deterministic evenly-spaced masks at sealed #486 family aggregate fractions; "
                "not mask-fidelity evidence"
            ),
            "totals": totals,
            "flagship_derived_ceiling_x": DERIVED_WHOLE_NETWORK_CEILING_X,
        },
        "metal_wall_benchmark": {
            "verdict": "UNMEASURED_BLOCKED" if not metal_probe["available"] else "OWED",
            "wall_speedup_x": None,
            "achieved_to_ceiling_ratio": None,
            "gap_reason": (
                "execution substrate exposes no Metal device; memory, gather, occupancy, and "
                "launch-overhead attribution is forbidden until the benchmark runs"
            ),
        },
        "compositions": {
            "oracle_mask_predictor": {
                "posthoc_oracle_global_relative_l2": ORACLE_20PCT_RELATIVE_L2,
                "cheap_source_margin_global_relative_l2_at_same_support": (
                    SOURCE_MARGIN_20PCT_RELATIVE_L2
                ),
                "absolute_relative_l2_predictor_gap": (
                    SOURCE_MARGIN_20PCT_RELATIVE_L2 - ORACLE_20PCT_RELATIVE_L2
                ),
                "named_gap": "current_witness_oracle_mask_predictor_gap",
                "authority": "MEASURED predecessor n600; no predictor built here",
            },
            "rank2_k2": {
                "kernel_composition": "implemented rank-batched VJPs up to rank 8",
                "hardware_timing": "UNMEASURED_BLOCKED",
                "accuracy_authority": (
                    "separate guarded #487 K=2 provider; current n1 smoke NOT_ADMITTED"
                ),
            },
        },
        "verdict_scope": (
            "current execution substrate and default-off compact frozen-Conv2d input-adjoint "
            "source only; no Metal parity/wall, mask predictor, full nonlinear SegNet VJP, "
            "training, score, or pointer verdict"
        ),
        "req_R": (
            "run the resumable host command on real Metal; require N=10 per-config bit identity "
            "before accepting any wall result or wiring a DSL lever"
        ),
    }
    _atomic_json(output, receipt)
    return receipt


@dataclass(frozen=True)
class ConvCase:
    name: str
    family: str
    spec: Conv2DAdjointSpec
    weight: np.ndarray


def _module_family(name: str) -> str:
    if "segmentation_head" in name:
        return "segmentation_head"
    if "decoder" in name:
        return "decoder"
    if "encoder" in name:
        return "encoder"
    raise BenchError(f"unclassified frozen-SegNet convolution {name}")


def _load_real_conv_cases() -> list[ConvCase]:
    import torch

    round2 = _load_module("_custom_sparse_round2", REPO / "tools/probe_frozen_replay_convex_head.py")
    model = round2._load_cpu_segnet()
    rows: list[ConvCase] = []
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue

        def hook(layer: Any, inputs: Any, output: Any, *, layer_name: str = name) -> None:
            input_nchw = tuple(int(value) for value in inputs[0].shape)
            output_nchw = tuple(int(value) for value in output.shape)
            weight = np.ascontiguousarray(
                layer.weight.detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=np.float32
            )
            rows.append(
                ConvCase(
                    name=layer_name,
                    family=_module_family(layer_name),
                    spec=Conv2DAdjointSpec(
                        input_hw=(input_nchw[2], input_nchw[3]),
                        output_hw=(output_nchw[2], output_nchw[3]),
                        cin=input_nchw[1],
                        cout=output_nchw[1],
                        kernel_hw=tuple(int(value) for value in layer.kernel_size),
                        stride_hw=tuple(int(value) for value in layer.stride),
                        padding_hw=tuple(int(value) for value in layer.padding),
                        dilation_hw=tuple(int(value) for value in layer.dilation),
                        groups=int(layer.groups),
                    ),
                    weight=weight,
                )
            )

        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros((1, 3, 384, 512), dtype=torch.float32))
    for handle in handles:
        handle.remove()
    if len(rows) != 125:
        raise BenchError(f"expected 125 frozen-SegNet convolutions, found {len(rows)}")
    return rows


def _family_support_fractions() -> dict[str, float]:
    predecessor = json.loads(PREDECESSOR_RECEIPT.read_text())
    profile = predecessor["measurements"]["conv_backward_support_flop_bounds"][
        "top_output@0.047365976969"
    ]
    return {
        family: float(row["ideal_spatial_sparse_conv_backward_flops"])
        / float(row["dense_nominal_conv_backward_flops"])
        for family, row in profile.items()
        if family in {"encoder", "decoder", "segmentation_head"}
    }


@dataclass(frozen=True)
class DenseMetalInputAdjoint:
    kernel: Any
    inputs: tuple[Any, ...]
    output_shape: tuple[int, int, int, int]
    output_dtype: Any

    def __call__(self) -> Any:
        (output,) = self.kernel(
            inputs=list(self.inputs),
            output_shapes=[self.output_shape],
            output_dtypes=[self.output_dtype],
            grid=(math.prod(self.output_shape), 1, 1),
            threadgroup=(256, 1, 1),
        )
        return output


def _make_dense_metal_input_adjoint(
    dense_cotangent: np.ndarray,
    weight: np.ndarray,
    spec: Conv2DAdjointSpec,
) -> DenseMetalInputAdjoint:
    import mlx.core as mx

    from tac.local_acceleration import metal_grouped_conv_backward as grouped

    dense = np.ascontiguousarray(dense_cotangent, dtype=np.float32)
    if dense.ndim != 4 or tuple(dense.shape[1:]) != (*spec.output_hw, spec.cout):
        raise ValueError("dense cotangent shape mismatch")
    rank = int(dense.shape[0])
    x = mx.zeros((rank, *spec.input_hw, spec.cin), dtype=mx.float32)
    wt = mx.array(np.ascontiguousarray(weight, dtype=np.float32), dtype=mx.float32)
    cot = mx.array(dense, dtype=mx.float32)
    stride = mx.array(np.asarray(spec.stride_hw, dtype=np.int32))
    padding = mx.array(np.asarray(spec.padding_hw, dtype=np.int32))
    dilation = mx.array(np.asarray(spec.dilation_hw, dtype=np.int32))
    groups = mx.array(np.asarray([spec.groups], dtype=np.int32))
    grad_input_kernel, _ = grouped._kernels()
    return DenseMetalInputAdjoint(
        kernel=grad_input_kernel,
        inputs=(x, wt, cot, stride, padding, dilation, groups),
        output_shape=(rank, *spec.input_hw, spec.cin),
        output_dtype=mx.float32,
    )


def _eval_result(value: Any) -> None:
    import mlx.core as mx

    if isinstance(value, (tuple, list)):
        mx.eval(*value)
    else:
        mx.eval(value)


def _timing(fn: Callable[[], Any], *, warmups: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmups):
        _eval_result(fn())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        _eval_result(fn())
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "p10_ms": float(np.quantile(samples, 0.10)),
        "p90_ms": float(np.quantile(samples, 0.90)),
    }


def _nominal_flops(spec: Conv2DAdjointSpec, output_sites: int) -> int:
    return (
        2
        * output_sites
        * spec.cout
        * spec.input_channels_per_group
        * spec.kernel_hw[0]
        * spec.kernel_hw[1]
    )


def _run_wall_replay(*, warmups: int, repeats: int) -> dict[str, Any]:
    import mlx.core as mx

    cases = _load_real_conv_cases()
    fractions = _family_support_fractions()
    rows = []
    totals = {
        "dense_median_ms": 0.0,
        "sparse_median_ms": 0.0,
        "dense_nominal_flops": 0,
        "sparse_nominal_flops": 0,
        "dense_exact_flops": 0,
        "sparse_exact_flops": 0,
    }
    for case_index, case in enumerate(cases):
        fraction = fractions[case.family]
        support = _even_support(case.spec.output_hw, fraction)
        plan = build_sparse_spatial_plan(support, case.spec)
        rng = np.random.default_rng(SEED + 1000 + case_index)
        dense_cotangent = np.zeros((1, *case.spec.output_hw, case.spec.cout), dtype=np.float32)
        dense_flat = dense_cotangent.reshape(1, -1, case.spec.cout)
        dense_flat[:, plan.output_indices, :] = rng.standard_normal(
            (1, plan.output_site_count, case.spec.cout)
        ).astype(np.float32)
        compact = compact_cotangent(dense_cotangent, plan)
        sparse_executable = make_sparse_conv2d_input_adjoint_metal(compact, case.weight, plan)
        dense_executable = _make_dense_metal_input_adjoint(
            dense_cotangent, case.weight, case.spec
        )
        sparse_timing = _timing(sparse_executable, warmups=warmups, repeats=repeats)
        dense_timing = _timing(dense_executable, warmups=warmups, repeats=repeats)
        sparse_output = sparse_executable()
        dense_output = dense_executable()
        mx.eval(sparse_output, dense_output)
        sparse_np = np.asarray(sparse_output)
        dense_restricted = np.asarray(dense_output).reshape(1, -1, case.spec.cin)[
            :, plan.input_indices, :
        ]
        max_abs_vs_dense_metal = float(
            np.max(np.abs(sparse_np.astype(np.float64) - dense_restricted.astype(np.float64)))
        )
        dense_nominal = _nominal_flops(
            case.spec, case.spec.output_hw[0] * case.spec.output_hw[1]
        )
        sparse_nominal = _nominal_flops(case.spec, plan.output_site_count)
        row = {
            "name": case.name,
            "family": case.family,
            "spec": case.spec.to_dict(),
            "support_fraction_source": fraction,
            "plan": plan.to_dict(),
            "dense_nominal_flops": dense_nominal,
            "sparse_nominal_flops": sparse_nominal,
            "dense_timing": dense_timing,
            "sparse_timing": sparse_timing,
            "wall_speedup_x": dense_timing["median_ms"] / sparse_timing["median_ms"],
            "max_abs_vs_existing_dense_metal": max_abs_vs_dense_metal,
            "dense_metal_comparison_authority": "diagnostic only; NumPy-fp32 cross-process gate owns parity",
        }
        rows.append(row)
        totals["dense_median_ms"] += dense_timing["median_ms"]
        totals["sparse_median_ms"] += sparse_timing["median_ms"]
        totals["dense_nominal_flops"] += dense_nominal
        totals["sparse_nominal_flops"] += sparse_nominal
        totals["dense_exact_flops"] += 2 * plan.dense_fma_count
        totals["sparse_exact_flops"] += 2 * plan.sparse_fma_count
        del sparse_executable, dense_executable, sparse_output, dense_output
        mx.clear_cache()
    wall_speedup = totals["dense_median_ms"] / totals["sparse_median_ms"]
    nominal_ceiling = totals["dense_nominal_flops"] / totals["sparse_nominal_flops"]
    exact_ceiling = totals["dense_exact_flops"] / totals["sparse_exact_flops"]
    totals.update(
        {
            "conv_case_count": len(rows),
            "wall_speedup_x": wall_speedup,
            "nominal_arithmetic_ceiling_x": nominal_ceiling,
            "exact_valid_tap_arithmetic_ceiling_x": exact_ceiling,
            "flagship_derived_ceiling_x": DERIVED_WHOLE_NETWORK_CEILING_X,
            "achieved_to_flagship_ceiling_ratio": wall_speedup
            / DERIVED_WHOLE_NETWORK_CEILING_X,
            "gap_to_flagship_ceiling_x": DERIVED_WHOLE_NETWORK_CEILING_X - wall_speedup,
        }
    )
    return {
        "schema": "custom_sparse_adjoint_full_shape_replay.v1",
        "completed_at_utc": _utc(),
        "support_geometry": (
            "deterministic evenly-spaced synthetic sites at sealed #486 family-active fractions; "
            "real 125-layer shapes and frozen weights; not mask-fidelity evidence"
        ),
        "baseline": "existing #212 grouped-convolution Metal grad-input kernel",
        "warmups": warmups,
        "repeats": repeats,
        "family_support_fractions": fractions,
        "aggregate": totals,
        "cases": rows,
    }


def _run_rank2_k2(*, warmups: int, repeats: int) -> dict[str, Any]:
    cases = _load_real_conv_cases()
    case = next(row for row in cases if row.name == "decoder.blocks.2.conv2.0")
    plan = build_sparse_spatial_plan(np.ones(case.spec.output_hw, dtype=np.bool_), case.spec)
    rng = np.random.default_rng(SEED + 2000)
    dense_rank2 = rng.standard_normal((2, *case.spec.output_hw, case.spec.cout)).astype(np.float32)
    compact_rank2 = compact_cotangent(dense_rank2, plan)
    sparse_rank2 = make_sparse_conv2d_input_adjoint_metal(compact_rank2, case.weight, plan)
    dense_batched_rank2 = _make_dense_metal_input_adjoint(dense_rank2, case.weight, case.spec)
    dense_first = _make_dense_metal_input_adjoint(dense_rank2[0:1], case.weight, case.spec)
    dense_second = _make_dense_metal_input_adjoint(dense_rank2[1:2], case.weight, case.spec)

    def dense_sequential() -> tuple[Any, Any]:
        return dense_first(), dense_second()

    sparse_timing = _timing(sparse_rank2, warmups=warmups, repeats=repeats)
    dense_batch_timing = _timing(dense_batched_rank2, warmups=warmups, repeats=repeats)
    dense_seq_timing = _timing(dense_sequential, warmups=warmups, repeats=repeats)
    no_reuse_factor = (dense_seq_timing["median_ms"] / 2.0) / sparse_timing["median_ms"]
    k2_amortized_factor = dense_seq_timing["median_ms"] / sparse_timing["median_ms"]
    return {
        "schema": "custom_sparse_adjoint_rank2_k2_reuse.v1",
        "completed_at_utc": _utc(),
        "case": case.name,
        "spec": case.spec.to_dict(),
        "rank": 2,
        "reuse_window_k": 2,
        "support": "full; basis VJPs are not assumed spatially sparse",
        "sparse_rank2_shared_weight_timing": sparse_timing,
        "existing_dense_rank2_batched_timing": dense_batch_timing,
        "existing_dense_two_sequential_vjps_timing": dense_seq_timing,
        "no_reuse_speed_factor_vs_one_dense_vjp": no_reuse_factor,
        "k2_amortized_speed_factor_vs_two_sequential_dense_vjps": k2_amortized_factor,
        "economics_law": "basis wins over K steps iff T_basis / K < T_dense_per_step",
        "accuracy_boundary": (
            "timing only; state-stable Jacobian and coefficient fidelity remain owned by the guarded "
            "#487 K=2 provider, whose n1 smoke is NOT_ADMITTED and whose n600 proof is separate"
        ),
    }


def _aggregate_cross_process(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != CROSS_PROCESS_COUNT:
        raise BenchError(f"expected {CROSS_PROCESS_COUNT} cross-process rows")
    config_names = [name for name, *_ in _parity_cases()]
    configs = {}
    for name in config_names:
        selected = [
            next(config for config in row["configs"] if config["config"] == name) for row in rows
        ]
        hashes = sorted({config["metal_output_sha256"] for config in selected})
        mismatch_counts = [int(config["bit_mismatch_count"]) for config in selected]
        configs[name] = {
            "process_count": len(selected),
            "unique_output_hashes": hashes,
            "unique_output_hash_count": len(hashes),
            "bit_mismatch_counts": mismatch_counts,
            "all_bit_identical_to_numpy_fp32_dense_on_support": all(
                count == 0 for count in mismatch_counts
            ),
            "verdict": (
                "BIT_IDENTICAL"
                if len(hashes) == 1 and all(count == 0 for count in mismatch_counts)
                else "FAIL"
            ),
        }
    combined_hashes = sorted({row["combined_output_sha256"] for row in rows})
    green = len(combined_hashes) == 1 and all(
        config["verdict"] == "BIT_IDENTICAL" for config in configs.values()
    )
    return {
        "schema": "custom_sparse_adjoint_cross_process_aggregate.v1",
        "process_count": len(rows),
        "combined_unique_output_hashes": combined_hashes,
        "combined_unique_output_hash_count": len(combined_hashes),
        "configs": configs,
        "gate": "GREEN_BIT_IDENTICAL" if green else "FAIL",
        "green": green,
    }


def _runtime() -> dict[str, Any]:
    import mlx.core as mx

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "mlx": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "device": mx.metal.device_info(),
        "git_head": _git_head(),
        "git_status": _git_status(),
        "argv": list(sys.argv),
    }


def run(output_dir: Path, *, resume: bool, warmups: int, repeats: int) -> dict[str, Any]:
    import mlx.core as mx

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _contract()
    contract_path = output_dir / "run_contract.json"
    if contract_path.is_file():
        prior = json.loads(contract_path.read_text())
        if prior != contract:
            raise BenchError("run contract drift; use a new output directory")
        if not resume:
            raise BenchError("existing staged run found; pass --resume")
    else:
        _atomic_json(contract_path, contract)
    receipt_path = output_dir / "measurement_receipt.json"
    if receipt_path.is_file():
        if not resume:
            raise BenchError("completed receipt exists; pass --resume")
        return json.loads(receipt_path.read_text())

    lock_path = output_dir / ".single_writer.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise BenchError("another benchmark process owns the output directory") from exc
    try:
        probe = mx.array([1.0], dtype=mx.float32)
        mx.eval(probe)
        workers_dir = output_dir / "cross_process"
        workers_dir.mkdir(parents=True, exist_ok=True)
        workers = []
        for index in range(CROSS_PROCESS_COUNT):
            worker_path = workers_dir / f"worker_{index:02d}.json"
            usable = False
            if resume and worker_path.is_file():
                row = json.loads(worker_path.read_text())
                usable = row.get("contract_sha256") == contract["contract_sha256"]
            if not usable:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--parity-worker",
                        "--worker-output",
                        str(worker_path),
                    ],
                    cwd=REPO,
                    check=True,
                )
            workers.append(json.loads(worker_path.read_text()))
        parity = _aggregate_cross_process(workers)
        parity_stage = output_dir / "stage_parity_complete.json"
        _atomic_json(parity_stage, parity)
        if not parity["green"]:
            raise BenchError("cross-process NumPy-fp32 parity gate failed")

        wall_stage = output_dir / "stage_wall_replay_complete.json"
        if resume and wall_stage.is_file():
            wall = json.loads(wall_stage.read_text())
        else:
            wall = _run_wall_replay(warmups=warmups, repeats=repeats)
            _atomic_json(wall_stage, wall)

        reuse_stage = output_dir / "stage_rank2_k2_complete.json"
        if resume and reuse_stage.is_file():
            reuse = json.loads(reuse_stage.read_text())
        else:
            reuse = _run_rank2_k2(warmups=warmups, repeats=repeats)
            _atomic_json(reuse_stage, reuse)

        k2_smoke = json.loads(K2_SMOKE_RECEIPT.read_text())
        receipt = {
            "schema": SCHEMA,
            "completed_at_utc": _utc(),
            "lane_id": LANE_ID,
            "axis": AXIS,
            "status": "GREEN_PRIMITIVE_ONLY",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_delta": "NONE",
            "training_performed": False,
            "paid_dispatch": False,
            "live_run_mutated": False,
            "contract": contract,
            "runtime": _runtime(),
            "parity": parity,
            "wall_replay": wall,
            "compositions": {
                "oracle_mask_predictor": {
                    "posthoc_oracle_support_fraction": 0.20,
                    "posthoc_oracle_global_relative_l2": ORACLE_20PCT_RELATIVE_L2,
                    "cheap_source_margin_global_relative_l2_at_same_support": (
                        SOURCE_MARGIN_20PCT_RELATIVE_L2
                    ),
                    "absolute_relative_l2_predictor_gap": (
                        SOURCE_MARGIN_20PCT_RELATIVE_L2 - ORACLE_20PCT_RELATIVE_L2
                    ),
                    "named_gap": "current_witness_oracle_mask_predictor_gap",
                    "authority": "MEASURED predecessor n600 receipt; no predictor built here",
                },
                "rank2_k2_state_stable_jacobian_reuse": reuse,
                "k2_existing_smoke": {
                    "path": str(K2_SMOKE_RECEIPT.relative_to(REPO)),
                    "sha256": _sha256(K2_SMOKE_RECEIPT),
                    "n_pairs": k2_smoke["n_pairs"],
                    "admission": k2_smoke["admission_verdict"],
                    "costate_cosine_fp32": k2_smoke["measurement"]["costate_fidelity"][
                        "cosine_fp32"
                    ]["median"],
                    "renderer_gradient_cosine_fp32": k2_smoke["measurement"][
                        "renderer_gradient_fidelity"
                    ]["cosine_fp32"]["median"],
                    "authority": "MEASURED n1 diagnostic smoke; full n600 and in-loop timing remain owed",
                },
            },
            "verdict_scope": (
                "custom compact frozen-Conv2d input-adjoint kernel on this M5 Max/MLX fingerprint; "
                "synthetic support-geometry wall replay over all real SegNet convolution shapes. "
                "No verdict on mask prediction, global-SE approximation, nonlinear full-SegNet VJP, "
                "optimizer regret, live training, contest score, or another chip/runtime."
            ),
            "req_R": (
                "reopen live integration only after a cheap current-witness mask approaches the measured "
                "20% oracle error, global SE is explicitly handled, and n600 renderer-gradient/optimizer "
                "regret plus in-loop wall gates pass"
            ),
            "triality": {
                "equation": "custom_sparse_adjoint_achieved_vs_ceiling_v1",
                "dag_feed": ".omx/research/custom_sparse_adjoint_kernel_DAG_FEED_20260713.md",
                "dsl": "REFUSED_NOT_FIREABLE; primitive has no admitted accuracy provider",
            },
        }
        _atomic_json(receipt_path, receipt)
        return receipt
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--static-receipt",
        type=Path,
        default=DEFAULT_STATIC_RECEIPT,
        help="durable destination used by --static-verify",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--parity-worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument(
        "--static-verify",
        action="store_true",
        help="seal CPU/static proof and an honest Metal-availability blocker receipt",
    )
    args = parser.parse_args(argv)
    if args.parity_worker and args.static_verify:
        parser.error("--parity-worker and --static-verify are mutually exclusive")
    if args.static_verify:
        result = _run_static_verification(args.static_receipt.resolve())
    elif args.parity_worker:
        if args.worker_output is None:
            parser.error("--parity-worker requires --worker-output")
        result = _run_parity_worker(args.worker_output.resolve())
    else:
        if args.warmups < 1 or args.repeats < 3:
            parser.error("warmups must be >=1 and repeats must be >=3")
        result = run(
            args.output_dir,
            resume=args.resume,
            warmups=args.warmups,
            repeats=args.repeats,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
