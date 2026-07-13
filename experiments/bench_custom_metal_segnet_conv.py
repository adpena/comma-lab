#!/usr/bin/env python3
"""Benchmark custom frozen-SegNet Metal convs at the actual model shapes.

The output is a local research receipt, never a score.  If the evaluated Metal
probe fails, the harness writes an exact blocker receipt without timing rows and
returns exit code 2.  It never manufactures substitute CPU timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.local_acceleration import metal_segnet_conv as msc

AXIS = "[macOS-MLX research-signal]"
SCHEMA = "custom_metal_segnet_conv_benchmark_v1"
DEFAULT_RAW = REPO / "experiments/results/levelset_packet_20260708T221700Z/inflated/0.raw"
DEFAULT_OUT = REPO / "experiments/results/custom_metal_conv_20260713/receipt.json"
RAW_PAIR_SHAPE = (2, 874, 1164, 3)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_custody(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(REPO)),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        assert tmp is not None
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _run_text(argv: list[str]) -> str:
    proc = subprocess.run(
        argv,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _command_probe(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _git_custody() -> dict[str, Any]:
    status = _run_text(["git", "status", "--short"])
    return {
        "branch": _run_text(["git", "branch", "--show-current"]),
        "head": _run_text(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _load_frozen_segnet() -> tuple[Any, Path]:
    import torch
    from safetensors.torch import load_file

    upstream = REPO / "upstream"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from modules import SegNet, segnet_sd_path

    torch.set_num_threads(1)
    model = SegNet().eval()
    weights = Path(segnet_sd_path)
    model.load_state_dict(load_file(str(weights), device="cpu"))
    return model, weights


def _conv_kind(module: Any) -> str:
    kh, kw = map(int, module.kernel_size)
    if (kh, kw) == (1, 1) and int(module.groups) == 1:
        return "pointwise-1x1"
    if (
        kh == kw
        and kh in (3, 5)
        and int(module.groups) == int(module.in_channels)
        and int(module.out_channels) == int(module.in_channels)
    ):
        return "depthwise"
    return "mlx-native-remainder"


def _trace_conv_inventory(model: Any) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue

        def hook(layer: Any, inputs: tuple[Any, ...], output: Any, *, name: str = name) -> None:
            x = inputs[0]
            batch = int(x.shape[0])
            out_h, out_w = int(output.shape[-2]), int(output.shape[-1])
            kh, kw = map(int, layer.kernel_size)
            macs = (
                batch
                * out_h
                * out_w
                * int(layer.out_channels)
                * kh
                * kw
                * (int(layer.in_channels) // int(layer.groups))
            )
            rows.append(
                {
                    "name": name,
                    "scope": "encoder" if name.startswith("encoder.") else "decoder-head",
                    "kind": _conv_kind(layer),
                    "batch": batch,
                    "hin": int(x.shape[-2]),
                    "win": int(x.shape[-1]),
                    "cin": int(layer.in_channels),
                    "cout": int(layer.out_channels),
                    "kh": kh,
                    "kw": kw,
                    "stride": list(map(int, layer.stride)),
                    "padding": list(map(int, layer.padding)),
                    "dilation": list(map(int, layer.dilation)),
                    "groups": int(layer.groups),
                    "hout": out_h,
                    "wout": out_w,
                    "macs": int(macs),
                }
            )

        handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(torch.zeros((1, 3, 384, 512), dtype=torch.float32))
    for handle in handles:
        handle.remove()
    return rows


def _inventory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        total = sum(int(row["macs"]) for row in selected)
        by_kind: dict[str, dict[str, Any]] = {}
        for kind in ("pointwise-1x1", "depthwise", "mlx-native-remainder"):
            subset = [row for row in selected if row["kind"] == kind]
            macs = sum(int(row["macs"]) for row in subset)
            by_kind[kind] = {
                "calls": len(subset),
                "macs": macs,
                "mac_fraction": macs / total if total else 0.0,
            }
        return {"calls": len(selected), "macs": total, "by_kind": by_kind}

    return {
        "encoder": summarize([row for row in rows if row["scope"] == "encoder"]),
        "full_segnet": summarize(rows),
    }


def _shape_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["kind"],
        row["batch"],
        row["hin"],
        row["win"],
        row["cin"],
        row["cout"],
        row["kh"],
        row["stride"][0],
        row["padding"][0],
        row["hout"],
        row["wout"],
    )


def _representative_shapes(
    model: Any,
    inventory: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], Any, int]]:
    modules = dict(model.named_modules())
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        if row["kind"] in {"pointwise-1x1", "depthwise"}:
            grouped[_shape_key(row)].append(row)
    result = []
    for key in sorted(grouped, key=str):
        rows = grouped[key]
        representative = dict(rows[0])
        representative["occurrences"] = len(rows)
        representative["layer_names"] = [row["name"] for row in rows]
        result.append((representative, modules[rows[0]["name"]], len(rows)))
    return result


def _stable_shape_seed(seed: int, row: dict[str, Any]) -> int:
    digest = hashlib.sha256(
        (str(int(seed)) + "|" + repr(_shape_key(row))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _benchmark_call(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        value = fn()
        mx.eval(value)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "max_abs": float(np.max(delta)),
        "mean_abs": float(np.mean(delta)),
        "actual_sha256": _sha256_array(actual),
        "reference_sha256": _sha256_array(expected),
    }


def _pointwise_variant_runtime(
    weight_kn: np.ndarray,
    variant: str,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    import mlx.core as mx

    if variant == "fp16":
        rounded = weight_kn.astype(np.float16)
        return mx.array(rounded, dtype=mx.float16), {}, rounded
    if variant == "int8":
        packet = msc.quantize_pointwise_int8(weight_kn)
        return (
            mx.array(packet.values, dtype=mx.int8),
            {
                "scales": mx.array(packet.scales, dtype=mx.float32),
                "cout": packet.cout,
            },
            msc.dequantize_pointwise_int8(packet).astype(np.float16),
        )
    packet = msc.quantize_pointwise_int4(weight_kn)
    return (
        mx.array(packet.values, dtype=mx.uint8),
        {
            "scales": mx.array(packet.scales, dtype=mx.float32),
            "cout": packet.cout,
        },
        msc.dequantize_pointwise_int4(packet).astype(np.float16),
    )


def _benchmark_pointwise_shape(
    row: dict[str, Any],
    module: Any,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    rng = np.random.default_rng(_stable_shape_seed(seed, row))
    x_np = rng.standard_normal(
        (row["batch"], row["hin"], row["win"], row["cin"]),
        dtype=np.float32,
    )
    weight_oihw = np.asarray(module.weight.detach().cpu(), dtype=np.float32)
    weight_kn = msc._pointwise_weight_kn_from_oihw(weight_oihw)
    x_fp32 = mx.array(x_np, dtype=mx.float32)
    x_fp16 = mx.array(x_np.astype(np.float16), dtype=mx.float16)
    weight_mlx_fp32 = mx.array(weight_kn.T.reshape(row["cout"], 1, 1, row["cin"]))
    weight_mlx_fp16 = weight_mlx_fp32.astype(mx.float16)
    timings = {
        "mlx-native-fp32": _benchmark_call(
            lambda: mx.conv2d(x_fp32, weight_mlx_fp32),
            warmup=warmup,
            repeats=repeats,
        ),
        "mlx-native-fp16": _benchmark_call(
            lambda: mx.conv2d(x_fp16, weight_mlx_fp16),
            warmup=warmup,
            repeats=repeats,
        ),
    }
    variants: dict[str, Any] = {}
    for variant in ("fp16", "int8", "int4"):
        runtime_weight, kwargs, reference_weight = _pointwise_variant_runtime(
            weight_kn,
            variant,
        )

        def call(
            runtime_weight: Any = runtime_weight,
            kwargs: dict[str, Any] = kwargs,
            variant: str = variant,
        ) -> Any:
            return msc.pointwise_1x1_metal(
                x_fp16,
                runtime_weight,
                variant=variant,
                **kwargs,
            )

        timing = _benchmark_call(call, warmup=warmup, repeats=repeats)
        first = call()
        second = call()
        mx.eval(first, second)
        first_np = np.asarray(first)
        second_np = np.asarray(second)
        reference = msc.pointwise_1x1_numpy_fp32(
            x_np.astype(np.float16),
            reference_weight,
        )
        variants[variant] = {
            "timing": timing,
            "speedup_vs_mlx_native_fp32": (
                timings["mlx-native-fp32"]["median_ms"] / timing["median_ms"]
            ),
            "speedup_vs_mlx_native_fp16": (
                timings["mlx-native-fp16"]["median_ms"] / timing["median_ms"]
            ),
            "numpy_fp32_reference": _error_metrics(first_np, reference),
            "deterministic_repeat_equal": bool(np.array_equal(first_np, second_np)),
            "repeat_sha256": _sha256_array(second_np),
        }
    return {"shape": row, "native": timings, "custom": variants}


def _benchmark_depthwise_shape(
    row: dict[str, Any],
    module: Any,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    rng = np.random.default_rng(_stable_shape_seed(seed, row))
    x_np = rng.standard_normal(
        (row["batch"], row["hin"], row["win"], row["cin"]),
        dtype=np.float32,
    )
    weight_oihw = np.asarray(module.weight.detach().cpu(), dtype=np.float32)
    weight_mlx_np = np.ascontiguousarray(weight_oihw.transpose(0, 2, 3, 1))
    x_fp32 = mx.array(x_np, dtype=mx.float32)
    x_fp16 = mx.array(x_np.astype(np.float16), dtype=mx.float16)
    weight_fp32 = mx.array(weight_mlx_np, dtype=mx.float32)
    weight_fp16 = mx.array(weight_mlx_np.astype(np.float16), dtype=mx.float16)
    stride = int(row["stride"][0])
    padding = int(row["padding"][0])
    native_fp32 = _benchmark_call(
        lambda: mx.conv2d(
            x_fp32,
            weight_fp32,
            stride=(stride, stride),
            padding=(padding, padding),
            groups=int(row["groups"]),
        ),
        warmup=warmup,
        repeats=repeats,
    )
    native_fp16 = _benchmark_call(
        lambda: mx.conv2d(
            x_fp16,
            weight_fp16,
            stride=(stride, stride),
            padding=(padding, padding),
            groups=int(row["groups"]),
        ),
        warmup=warmup,
        repeats=repeats,
    )

    def call() -> Any:
        return msc.depthwise_conv2d_metal(
            x_fp16,
            weight_fp16,
            stride=stride,
            padding=padding,
        )

    custom = _benchmark_call(call, warmup=warmup, repeats=repeats)
    first = call()
    second = call()
    mx.eval(first, second)
    first_np = np.asarray(first)
    second_np = np.asarray(second)
    reference = msc.depthwise_conv2d_numpy_fp32(
        x_np.astype(np.float16),
        weight_mlx_np.astype(np.float16),
        stride=stride,
        padding=padding,
    )
    return {
        "shape": row,
        "native": {"mlx-native-fp32": native_fp32, "mlx-native-fp16": native_fp16},
        "custom": {
            "fp16": {
                "timing": custom,
                "speedup_vs_mlx_native_fp32": native_fp32["median_ms"]
                / custom["median_ms"],
                "speedup_vs_mlx_native_fp16": native_fp16["median_ms"]
                / custom["median_ms"],
                "numpy_fp32_reference": _error_metrics(first_np, reference),
                "deterministic_repeat_equal": bool(np.array_equal(first_np, second_np)),
                "repeat_sha256": _sha256_array(second_np),
            }
        },
    }


def _load_real_segnet_inputs(raw_path: Path, *, real_pairs: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    pair_bytes = int(np.prod(RAW_PAIR_SHAPE))
    available = raw_path.stat().st_size // pair_bytes
    if available < real_pairs:
        raise ValueError(f"raw artifact has {available} pairs, requested {real_pairs}")
    raw = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(available, *RAW_PAIR_SHAPE),
    )
    frame1 = np.asarray(raw[:real_pairs, 1], dtype=np.float32)
    nchw = torch.from_numpy(np.ascontiguousarray(frame1.transpose(0, 3, 1, 2)))
    resized = functional.interpolate(nchw, size=(384, 512), mode="bilinear")
    return np.ascontiguousarray(resized.numpy().transpose(0, 2, 3, 1))


def _time_full_forward(
    adapter: Any,
    inputs_nhwc: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], np.ndarray]:
    import mlx.core as mx

    all_samples: list[float] = []
    logits_rows = []
    for pair_index in range(int(inputs_nhwc.shape[0])):
        x = mx.array(inputs_nhwc[pair_index : pair_index + 1], dtype=mx.float32)

        def call(x: Any = x) -> Any:
            return adapter(x)

        pair_timing = _benchmark_call(call, warmup=warmup, repeats=repeats)
        all_samples.extend(pair_timing["samples_ms"])
        logits = call()
        mx.eval(logits)
        logits_rows.append(np.asarray(logits))
    timing = {
        "batch": 1,
        "real_pair_count": int(inputs_nhwc.shape[0]),
        "count": len(all_samples),
        "median_ms": statistics.median(all_samples),
        "mean_ms": statistics.fmean(all_samples),
        "min_ms": min(all_samples),
        "max_ms": max(all_samples),
        "samples_ms": all_samples,
    }
    return timing, np.concatenate(logits_rows, axis=0)


def _full_forward_fidelity(
    model: Any,
    inputs_nhwc: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from tac.local_acceleration import mlx_scorer_adapters as adapters

    native = adapters.torch_segnet_to_mlx(model)
    native_timing, native_logits = _time_full_forward(
        native,
        inputs_nhwc,
        warmup=warmup,
        repeats=repeats,
    )
    native_argmax = np.argmax(native_logits, axis=-1)
    arms: dict[str, Any] = {}
    for variant in ("fp16", "int8", "int4"):
        mx.clear_cache()
        custom = msc.build_custom_metal_segnet_adapter(
            model,
            pointwise_variant=variant,
            require_opt_in=False,
        )
        timing, logits = _time_full_forward(
            custom,
            inputs_nhwc,
            warmup=warmup,
            repeats=repeats,
        )
        delta = np.abs(logits.astype(np.float32) - native_logits.astype(np.float32))
        argmax = np.argmax(logits, axis=-1)
        flips = int(np.count_nonzero(argmax != native_argmax))
        arms[variant] = {
            "timing": timing,
            "direct_speedup_vs_native": native_timing["median_ms"] / timing["median_ms"],
            "logit_max_abs_delta": float(np.max(delta)),
            "logit_mean_abs_delta": float(np.mean(delta)),
            "argmax_flip_count": flips,
            "argmax_flip_rate": flips / int(native_argmax.size),
            "argmax_sha256": _sha256_array(argmax),
            "logits_sha256": _sha256_array(logits),
        }
    return {
        "real_pairs": int(inputs_nhwc.shape[0]),
        "total_argmax_pixels": int(native_argmax.size),
        "native": {
            "timing": native_timing,
            "argmax_sha256": _sha256_array(native_argmax),
            "logits_sha256": _sha256_array(native_logits),
        },
        "arms": arms,
    }


def _weighted_kernel_times_ms(
    rows: list[dict[str, Any]],
    *,
    variant: str,
) -> tuple[float, float]:
    """Return occurrence-weighted isolated native/custom kernel time.

    These totals are the only admissible inputs to a time-domain Amdahl
    substitution.  MAC fractions are retained separately as a structural
    model; they are not latency fractions.
    """

    native = 0.0
    custom = 0.0
    for row in rows:
        count = int(row["shape"]["occurrences"])
        native += count * float(row["native"]["mlx-native-fp32"]["median_ms"])
        custom += count * float(row["custom"][variant]["timing"]["median_ms"])
    return native, custom


def _compose_amdahl(
    inventory_summary: dict[str, Any],
    pointwise: list[dict[str, Any]],
    depthwise: list[dict[str, Any]],
    full_forward: dict[str, Any],
) -> dict[str, Any]:
    full_kinds = inventory_summary["full_segnet"]["by_kind"]
    point_fraction = float(full_kinds["pointwise-1x1"]["mac_fraction"])
    depth_fraction = float(full_kinds["depthwise"]["mac_fraction"])
    remainder = 1.0 - point_fraction - depth_fraction
    depth_native_ms, depth_custom_ms = _weighted_kernel_times_ms(
        depthwise,
        variant="fp16",
    )
    depth_speedup = depth_native_ms / depth_custom_ms
    full_native_ms = float(full_forward["native"]["timing"]["median_ms"])
    rows = {}
    for variant in ("fp16", "int8", "int4"):
        point_native_ms, point_custom_ms = _weighted_kernel_times_ms(
            pointwise,
            variant=variant,
        )
        point_speedup = point_native_ms / point_custom_ms
        mac_share_model = 1.0 / (
            remainder + point_fraction / point_speedup + depth_fraction / depth_speedup
        )
        eligible_native_ms = point_native_ms + depth_native_ms
        eligible_custom_ms = point_custom_ms + depth_custom_ms
        residual_native_ms = full_native_ms - eligible_native_ms
        if residual_native_ms >= 0.0:
            time_substitution = full_native_ms / (
                residual_native_ms + eligible_custom_ms
            )
            time_substitution_refusal = None
        else:
            time_substitution = None
            time_substitution_refusal = (
                "occurrence-weighted isolated native kernel time exceeds the "
                "direct full-forward median; refuse the substitution estimate"
            )
        direct = float(full_forward["arms"][variant]["direct_speedup_vs_native"])
        rows[variant] = {
            "measured_shape_weighted_pointwise_speedup": point_speedup,
            "measured_shape_weighted_depthwise_speedup": depth_speedup,
            "isolated_native_pointwise_ms": point_native_ms,
            "isolated_custom_pointwise_ms": point_custom_ms,
            "isolated_native_depthwise_ms": depth_native_ms,
            "isolated_custom_depthwise_ms": depth_custom_ms,
            "isolated_native_eligible_ms": eligible_native_ms,
            "isolated_custom_eligible_ms": eligible_custom_ms,
            "direct_native_full_forward_median_ms": full_native_ms,
            "derived_full_forward_amdahl_speedup": time_substitution,
            "derived_full_forward_amdahl_refusal": time_substitution_refusal,
            "derived_mac_share_model_speedup": mac_share_model,
            "measured_direct_full_forward_speedup": direct,
            "derived_training_wall_speedup_if_teacher_forward_fraction_0_78": (
                1.0 / (0.22 + 0.78 / direct)
            ),
        }
    return {
        "method": (
            "DERIVED time-domain Amdahl substitutes occurrence-weighted MEASURED "
            "real-shape isolated native/custom kernel medians into the directly "
            "MEASURED native full-forward wall; it refuses if isolated native time "
            "exceeds that wall. The MAC-share calculation is a structural model only. "
            "Measured direct full-forward timing takes precedence over both."
        ),
        "full_model_mac_fractions": {
            "pointwise": point_fraction,
            "depthwise": depth_fraction,
            "mlx_native_remainder": remainder,
        },
        "teacher_forward_fraction_of_training_wall": {
            "value": 0.78,
            "provenance": "operator-provided measured context; not remeasured by this harness",
        },
        "arms": rows,
    }


def _environment_custody() -> dict[str, Any]:
    import mlx.core as mx

    try:
        mlx_version = getattr(mx, "__version__", None)
    except Exception:
        mlx_version = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": mlx_version,
        "mlx_default_device_label": str(mx.default_device()),
        "offline_metal_compiler_probe": _command_probe(["xcrun", "metal", "--version"]),
    }


def _static_reference_validation(seed: int) -> dict[str, Any]:
    """Small deterministic NumPy-only receipt; never mislabeled as kernel parity."""

    rng = np.random.default_rng(int(seed) + 9001)
    x = rng.standard_normal((1, 3, 5, 11), dtype=np.float32).astype(np.float16)
    weight = rng.standard_normal((11, 13), dtype=np.float32).astype(np.float16)
    point_first = msc.pointwise_1x1_numpy_fp32(x, weight)
    point_second = msc.pointwise_1x1_numpy_fp32(x, weight)
    matmul = (x.astype(np.float32).reshape(-1, 11) @ weight.astype(np.float32)).reshape(
        point_first.shape
    )
    int8 = msc.quantize_pointwise_int8(weight.astype(np.float32))
    int4 = msc.quantize_pointwise_int4(weight.astype(np.float32))
    depth_weight = rng.standard_normal((11, 3, 3), dtype=np.float32).astype(np.float16)
    depth_first = msc.depthwise_conv2d_numpy_fp32(x, depth_weight, padding=1)
    depth_second = msc.depthwise_conv2d_numpy_fp32(x, depth_weight, padding=1)
    return {
        "scope": "NumPy-fp32 references and frozen-weight packet format only; no Metal execution",
        "pointwise": {
            "fixed_order_vs_numpy_matmul_max_abs": float(
                np.max(np.abs(point_first - matmul))
            ),
            "fixed_order_vs_numpy_matmul_mean_abs": float(
                np.mean(np.abs(point_first - matmul))
            ),
            "deterministic_repeat_equal": bool(np.array_equal(point_first, point_second)),
            "sha256": _sha256_array(point_first),
        },
        "int8_weight_packet": {
            "max_abs_weight_error": float(
                np.max(
                    np.abs(
                        msc.dequantize_pointwise_int8(int8)
                        - weight.astype(np.float32)
                    )
                )
            ),
            "deterministic_packet": bool(
                int8.values.tobytes()
                == msc.quantize_pointwise_int8(weight).values.tobytes()
            ),
            "payload_sha256": _sha256_array(int8.values),
        },
        "int4_weight_packet": {
            "max_abs_weight_error": float(
                np.max(
                    np.abs(
                        msc.dequantize_pointwise_int4(int4)
                        - weight.astype(np.float32)
                    )
                )
            ),
            "deterministic_packet": bool(
                int4.values.tobytes()
                == msc.quantize_pointwise_int4(weight).values.tobytes()
            ),
            "payload_sha256": _sha256_array(int4.values),
        },
        "depthwise": {
            "deterministic_repeat_equal": bool(np.array_equal(depth_first, depth_second)),
            "sha256": _sha256_array(depth_first),
        },
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = _utc_now()
    raw_path = Path(args.raw).resolve()
    out_path = Path(args.out).resolve()
    model, weights_path = _load_frozen_segnet()
    inventory = _trace_conv_inventory(model)
    inventory_summary = _inventory_summary(inventory)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preflight",
        "started_at_utc": started,
        "axis": AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "pointer_moved": False,
        "seed": int(args.seed),
        "kernel_signature": msc.custom_segnet_conv_signature(),
        "static_reference_validation": _static_reference_validation(args.seed),
        "inventory": inventory,
        "inventory_summary": inventory_summary,
        "custody": {
            "git": _git_custody(),
            "model_weights_path": str(weights_path.relative_to(REPO)),
            "model_weights_bytes": weights_path.stat().st_size,
            "model_weights_sha256": _sha256_file(weights_path),
            "raw_path": str(raw_path.relative_to(REPO)),
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": _sha256_file(raw_path),
            "implementation_files": [
                _file_custody(Path(msc.__file__)),
                _file_custody(
                    REPO
                    / "src/tac/local_acceleration/tests/test_metal_segnet_conv.py"
                ),
                _file_custody(Path(__file__)),
            ],
            "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        },
        "environment": _environment_custody(),
        "promotion_gate": msc.N600_FIDELITY_GATE,
        "verdict_scope": "local frozen-SegNet forward throughput formulation only",
    }
    if not msc.metal_segnet_conv_backend_available():
        receipt.update(
            {
                "status": "blocked-no-metal-device",
                "completed_at_utc": _utc_now(),
                "blocker": {
                    "classification": "local-substrate-access",
                    "exact_error": (
                        "MLX default device labels GPU, but an evaluated allocation raises: "
                        "[metal::load_device] No Metal device available"
                    ),
                    "measurements_written": False,
                    "next_action": (
                        "rerun the recorded command in a Metal-visible local execution context; "
                        "do not infer timings from CPU or device labels"
                    ),
                },
                "pointwise_results": [],
                "depthwise_results": [],
                "full_forward": None,
                "amdahl": None,
            }
        )
        _atomic_write_json(out_path, receipt)
        return receipt, 2

    pointwise_results = []
    depthwise_results = []
    try:
        for row, module, _ in _representative_shapes(model, inventory):
            if row["kind"] == "pointwise-1x1":
                pointwise_results.append(
                    _benchmark_pointwise_shape(
                        row,
                        module,
                        seed=args.seed,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                )
            else:
                depthwise_results.append(
                    _benchmark_depthwise_shape(
                        row,
                        module,
                        seed=args.seed,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                )
        inputs = _load_real_segnet_inputs(raw_path, real_pairs=args.real_pairs)
        full_forward = _full_forward_fidelity(
            model,
            inputs,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        amdahl = _compose_amdahl(
            inventory_summary,
            pointwise_results,
            depthwise_results,
            full_forward,
        )
    except Exception as exc:
        receipt.update(
            {
                "status": "failed-kernel-build-or-execution",
                "completed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pointwise_results": pointwise_results,
                "depthwise_results": depthwise_results,
                "full_forward": None,
                "amdahl": None,
            }
        )
        _atomic_write_json(out_path, receipt)
        return receipt, 2
    receipt.update(
        {
            "status": "measured",
            "completed_at_utc": _utc_now(),
            "pointwise_results": pointwise_results,
            "depthwise_results": depthwise_results,
            "full_forward": full_forward,
            "amdahl": amdahl,
        }
    )
    _atomic_write_json(out_path, receipt)
    return receipt, 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-pairs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.real_pairs <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("--real-pairs and --repeats must be positive; --warmup non-negative")
    try:
        receipt, code = run(args)
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "failed-closed",
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "completed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "verdict_scope": "current local kernel build/benchmark invocation only",
        }
        _atomic_write_json(Path(args.out).resolve(), failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "out": str(Path(args.out)),
                "pointwise_rows": len(receipt.get("pointwise_results") or []),
                "depthwise_rows": len(receipt.get("depthwise_results") or []),
            },
            indent=2,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
