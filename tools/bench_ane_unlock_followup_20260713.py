#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local CoreML/ANE + MLX follow-up benchmark for the frozen SegNet.

The outer ``--bootstrap-offline`` mode creates a success-cleaned scratch venv
from the local uv cache.  All durable evidence is small JSON under the owned
result directory; rebuildable model packages stay in scratch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations.ane_unlock_followup_20260713 import (  # noqa: E402
    account_weight_fit,
    admit_concurrency,
    batch_pairs_per_second,
    batch_seconds_per_pair,
)
from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    base_receipt,
    require_real_n600,
    sha256_file,
    stored_npy_memmap,
)

OUT_DIR = REPO / "experiments/results/ane_unlock_followup_20260713"
WEIGHTS = REPO / "upstream/models/segnet.safetensors"
GT_N600 = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
THROUGHPUT_RECEIPT = REPO / ".omx/research/throughput_fresh_eyes_measurements_20260713.json"
SCRATCH_WATERFALL = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
    Path("/private/tmp"),
)
BOOTSTRAP_PACKAGES = (
    "numpy",
    "torch==2.12.1",
    "torchvision==0.27.1",
    "segmentation-models-pytorch==0.5.0",
    "safetensors",
    "coremltools==9.0",
    "mlx==0.31.2",
)


def _storage_preflight() -> dict[str, Any]:
    tiers: list[dict[str, Any]] = []
    selected: Path | None = None
    for path in SCRATCH_WATERFALL:
        exists = path.exists()
        writable = exists and os.access(path, os.W_OK)
        free = shutil.disk_usage(path).free if exists else None
        tiers.append({"path": str(path), "exists": exists, "writable": writable, "free_bytes": free})
        if selected is None and writable and free is not None and free >= 8 * 2**30:
            selected = path
    if selected is None:
        raise RuntimeError("storage waterfall REFUSE: no writable tier with 8 GiB free")
    return {
        "tiers": tiers,
        "selected_scratch_root": str(selected),
        "minimum_free_bytes": 8 * 2**30,
        "cleanup": "context-managed scratch; automatically removed after child exits",
        "durable_evidence_never_in_scratch": True,
    }


def _bootstrap(args: argparse.Namespace) -> int:
    preflight = _storage_preflight()
    atomic_json(args.out_dir / "storage_preflight.json", base_receipt(**preflight))
    scratch_root = Path(str(preflight["selected_scratch_root"]))
    with tempfile.TemporaryDirectory(prefix="pact_ane_followup_env_", dir=scratch_root) as root:
        venv = Path(root) / ".venv"
        # The populated uv cache and project venv are read-only inputs under
        # this managed sandbox.  Make an isolated scratch interpreter and add
        # those exact installed package trees through PYTHONPATH; this avoids
        # cache locks, network access, copies, and persistent environment edits.
        archive_root = Path.home() / ".cache/uv/archive-v0"
        coreml_roots = sorted(path.parent for path in archive_root.glob("*/coremltools-9.0.dist-info"))
        cattrs_roots = sorted(path.parent for path in archive_root.glob("*/cattrs-*.dist-info"))
        if not coreml_roots or not cattrs_roots:
            raise RuntimeError("cached coremltools/cattrs package roots are unavailable")
        project_site = next((REPO / ".venv/lib").glob("python*/site-packages"))
        env = dict(os.environ)
        env["TMPDIR"] = "/private/tmp"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_site), str(coreml_roots[-1]), str(cattrs_roots[-1]), env.get("PYTHONPATH", "")]
        )
        commands = [[sys.executable, "-m", "venv", "--without-pip", str(venv)]]
        logs: list[dict[str, Any]] = []
        for command in commands:
            run = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=False)
            logs.append({
                "command": ["<SCRATCH>" if str(token).startswith(root) else token for token in command],
                "returncode": run.returncode,
                "stdout_tail": run.stdout[-8000:],
                "stderr_tail": run.stderr[-8000:],
            })
            if run.returncode != 0:
                atomic_json(args.out_dir / "bootstrap_receipt.json", base_receipt(status="BLOCKED", logs=logs))
                return run.returncode
        child_args = [str(venv / "bin/python"), str(Path(__file__).resolve()), "--mode", args.mode]
        child_args += ["--out-dir", str(args.out_dir), "--scratch-dir", str(Path(root) / "model_scratch")]
        child_args += ["--reps", str(args.reps), "--concurrency-steps", str(args.concurrency_steps)]
        child_args += ["--duration-s", str(args.duration_s)]
        if args.resume_sidecar:
            child_args.append("--resume-sidecar")
        if args.skip_lut:
            child_args.append("--skip-lut")
        run = subprocess.run(child_args, cwd=REPO, env=env, capture_output=True, text=True, check=False)
        logs.append({
            "command": ["<SCRATCH>" if str(token).startswith(root) else token for token in child_args],
            "returncode": run.returncode,
            "stdout_tail": run.stdout[-12000:],
            "stderr_tail": run.stderr[-12000:],
        })
        atomic_json(
            args.out_dir / "bootstrap_receipt.json",
            base_receipt(
                status="COMPLETE" if run.returncode == 0 else "BLOCKED",
                packages=list(BOOTSTRAP_PACKAGES),
                package_source="read-only project venv plus read-only uv archive roots",
                offline=True,
                logs=logs,
                scratch_deleted_on_exit=True,
            ),
        )
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        return run.returncode


def _summary(samples: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("empty timing sample")
    ordered = sorted(values)
    return {
        "count": len(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
        "samples_s": values,
        "p05_s": ordered[max(0, round(0.05 * (len(ordered) - 1)))],
        "p95_s": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
    }


def _package_bytes(path: Path) -> tuple[int, list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    total = 0
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        size = file.stat().st_size
        total += size
        files.append({"path": str(file.relative_to(path)), "bytes": size})
    return total, files


def _imports() -> dict[str, Any]:
    import coremltools as ct
    import mlx
    import mlx.core as mx
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as torch_f
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    return {
        "ct": ct,
        "mlx": mlx,
        "mx": mx,
        "np": np,
        "smp": smp,
        "torch": torch,
        "torch_f": torch_f,
        "load_file": load_file,
    }


def _load_model(api: dict[str, Any]) -> Any:
    smp, load_file = api["smp"], api["load_file"]
    model = smp.Unet("tu-efficientnet_b2", classes=5, activation=None, encoder_weights=None).eval()
    missing, unexpected = model.load_state_dict(load_file(str(WEIGHTS)), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    return model


def _frame_tensor(api: dict[str, Any], frames: Any, start: int, batch: int) -> Any:
    np, torch, torch_f = api["np"], api["torch"], api["torch_f"]
    array = np.asarray(frames[start : start + batch])
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    tensor = torch_f.interpolate(tensor, size=(384, 512), mode="bilinear", align_corners=False)
    return (tensor / 255.0).contiguous()


def _convert(api: dict[str, Any], model: Any, example: Any, precision: Any) -> Any:
    ct, torch = api["ct"], api["torch"]
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(shape=example.shape, name="x")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=precision,
        skip_model_load=True,
    )


def _compile_package(package: Path, scratch: Path) -> Path:
    compiled = scratch / f"{package.stem}.mlmodelc"
    if not compiled.exists():
        command = ["xcrun", "coremlcompiler", "compile", str(package), str(scratch)]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise RuntimeError(
                f"coremlcompiler failed for {package.name}: rc={run.returncode} stderr={run.stderr[-3000:]}"
            )
    return compiled


def _compile_runtime(api: dict[str, Any], package: Path, scratch: Path, *, function_name: str | None = None) -> Any:
    ct = api["ct"]
    compiled = _compile_package(package, scratch)
    return ct.models.CompiledMLModel(
        str(compiled),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        function_name=function_name,
    )


def _t4_transform(api: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    ct = api["ct"]
    preserved: set[str] = set()
    selected: set[str] = set()

    def selector(op: Any) -> bool:
        scopes: list[str] = []
        for source, values in getattr(op, "scopes", {}).items():
            if getattr(source, "name", "") == "TORCHSCRIPT_MODULE_NAME":
                scopes = [str(value) for value in values]
        keep = scopes[:1] == ["segmentation_head"]
        name = f"{getattr(op, 'op_type', '?')}:{getattr(op, 'name', '?')}:{'/'.join(scopes)}"
        (preserved if keep else selected).add(name)
        return not keep

    return ct.transform.FP16ComputePrecision(op_selector=selector), {
        "preserved_fp32_ops": preserved,
        "selected_fp16_ops": selected,
    }


def _weight_compress(api: dict[str, Any], base: Any, kind: str) -> Any:
    ct = api["ct"]
    if kind == "w8":
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=0
            )
        )
        return ct.optimize.coreml.linear_quantize_weights(base, config=config)
    if kind in {"lut4", "lut6"}:
        nbits = int(kind[-1])
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=nbits, weight_threshold=0
            )
        )
        return ct.optimize.coreml.palettize_weights(base, config=config)
    raise ValueError(kind)


def _predict(model: Any, value: Any) -> Any:
    import numpy as np

    runtime, name = model
    output = runtime.predict({"x": np.ascontiguousarray(value, dtype=np.float32)})
    return np.asarray(output[name], dtype=np.float32)


def _build_models(api: dict[str, Any], frames: Any, scratch: Path, skip_lut: bool) -> dict[str, Any]:
    ct = api["ct"]
    model = _load_model(api)
    packages: dict[str, Any] = {}
    models: dict[str, Any] = {"torch_fp32": model}
    source_payload = sum(parameter.numel() * parameter.element_size() for parameter in model.state_dict().values())
    fp16_payload = sum(parameter.numel() * 2 for parameter in model.state_dict().values())
    package_rows: dict[str, Any] = {}
    for batch in (1, 8, 32):
        example = _frame_tensor(api, frames, 0, batch)
        dense = _convert(api, model, example, ct.precision.FLOAT16)
        if batch == 1:
            models["dense_fp16_source_b1"] = dense
            transform, custody = _t4_transform(api)
            t4 = _convert(api, model, example, transform)
            models["t4_custody"] = custody
        w8 = _weight_compress(api, dense, "w8")
        models[f"w8_b{batch}"] = w8
        package = scratch / f"w8_b{batch}.mlpackage"
        w8.save(str(package))
        packages[f"w8_b{batch}"] = package
        total, files = _package_bytes(package)
        weight_files = [row for row in files if "weight" in row["path"].lower()]
        package_rows[f"w8_b{batch}"] = {
            "package_bytes": total,
            "weight_files": weight_files,
            "weight_file_bytes": sum(row["bytes"] for row in weight_files),
        }
    descriptor = ct.utils.MultiFunctionDescriptor()
    for batch in (1, 8, 32):
        descriptor.add_function(str(packages[f"w8_b{batch}"]), "main", f"fwd_b{batch}")
    descriptor.default_function_name = "fwd_b1"
    multi = scratch / "segnet_w8_batch_tiers.mlpackage"
    ct.utils.save_multifunction(descriptor, str(multi))
    total, files = _package_bytes(multi)
    package_rows["multifunction"] = {
        "package_bytes": total,
        "files": files,
        "function_names": ["fwd_b1", "fwd_b8", "fwd_b32"],
        "default_function_name": "fwd_b1",
    }
    models["multifunction_path"] = multi
    _compile_package(multi, scratch)
    package_rows["multifunction"]["coremlcompiler_status"] = "MEASURED_COMPILED"
    package_rows["multifunction"]["runtime_function_dispatch_status"] = (
        "BLOCKED_NOT_MEASURED__COMPILEDMLMODEL_FUNCTIONNAME_REJECTED_NON_MLPROGRAM_CONTAINER"
    )
    models["multifunction_models"] = {}
    for batch in (1, 8, 32):
        output_name = models[f"w8_b{batch}"].get_spec().description.output[0].name
        models["multifunction_models"][f"fwd_b{batch}"] = (
            _compile_runtime(api, packages[f"w8_b{batch}"], scratch),
            output_name,
        )
    dense_package = scratch / "dense_fp16_b1.mlpackage"
    models["dense_fp16_source_b1"].save(str(dense_package))
    models["dense_fp16_b1"] = (
        _compile_runtime(api, dense_package, scratch),
        models["dense_fp16_source_b1"].get_spec().description.output[0].name,
    )
    t4_package = scratch / "t4_b1.mlpackage"
    t4.save(str(t4_package))
    models["t4_b1"] = (
        _compile_runtime(api, t4_package, scratch),
        t4.get_spec().description.output[0].name,
    )
    if not skip_lut:
        dense = models["dense_fp16_source_b1"]
        for kind in ("lut6", "lut4"):
            try:
                models[f"{kind}_b1"] = _weight_compress(api, dense, kind)
                package = scratch / f"{kind}_b1.mlpackage"
                models[f"{kind}_b1"].save(str(package))
                source = models[f"{kind}_b1"]
                models[f"{kind}_runtime_b1"] = (
                    _compile_runtime(api, package, scratch),
                    source.get_spec().description.output[0].name,
                )
                ptotal, pfiles = _package_bytes(package)
                weight_files = [row for row in pfiles if "weight" in row["path"].lower()]
                package_rows[kind] = {
                    "status": "MEASURED_BUILT",
                    "package_bytes": ptotal,
                    "weight_file_bytes": sum(row["bytes"] for row in weight_files),
                }
            except Exception as exc:
                package_rows[kind] = {
                    "status": "BLOCKED_NOT_MEASURED",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:3000],
                }
    models["footprint"] = {
        "torch_state_dict_bytes_fp32_measured": source_payload,
        "same_tensor_count_fp16_bytes_derived": fp16_payload,
        "t4_op_selector_custody": {
            "preserved_fp32_ops": sorted(models["t4_custody"]["preserved_fp32_ops"]),
            "selected_fp16_ops": sorted(models["t4_custody"]["selected_fp16_ops"]),
        },
        "packages": package_rows,
    }
    return models


def _bench_batches(api: dict[str, Any], frames: Any, models: dict[str, Any], reps: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for batch in (1, 8, 32):
        function = f"fwd_b{batch}"
        model = models["multifunction_models"][function]
        value = _frame_tensor(api, frames, 0, batch).numpy()
        for _ in range(3):
            _predict(model, value)
        samples: list[float] = []
        for _ in range(reps):
            started = time.perf_counter()
            _predict(model, value)
            samples.append(time.perf_counter() - started)
        timing = _summary(samples)
        timing.update({
            "batch_size": batch,
            "seconds_per_pair": batch_seconds_per_pair(batch_seconds=timing["median_s"], batch_size=batch),
            "pairs_per_second": batch_pairs_per_second(batch_seconds=timing["median_s"], batch_size=batch),
        })
        rows[function] = timing
    return {
        "execution_surface": (
            "three individually compiled fixed-shape ML Programs that are also the exact source functions "
            "of the compiled multifunction package; multifunction functionName dispatch is blocked"
        ),
        "coreml_compute_units": "CPU_AND_NE requested; ANE placement NOT proved",
        "rows": rows,
    }


def _fidelity_n600(api: dict[str, Any], frames: Any, models: dict[str, Any]) -> dict[str, Any]:
    np, torch = api["np"], api["torch"]
    require_real_n600(int(frames.shape[0]))
    candidates = {
        "dense_fp16_b1": models["dense_fp16_b1"],
        "w8_b1": models["multifunction_models"]["fwd_b1"],
        "t4_b1": models["t4_b1"],
    }
    for key in ("lut6_b1", "lut4_b1"):
        runtime_key = key.replace("_b1", "_runtime_b1")
        if runtime_key in models:
            candidates[key] = models[runtime_key]
    aggregate = dict.fromkeys(candidates, 0)
    per_pair = {name: [] for name in candidates}
    total_pixels = 600 * 384 * 512
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(600):
            value = _frame_tensor(api, frames, index, 1)
            reference = models["torch_fp32"](value).detach().cpu().numpy().argmax(1)
            for name, candidate in candidates.items():
                labels = _predict(candidate, value.numpy()).argmax(1)
                count = int(np.count_nonzero(reference != labels))
                aggregate[name] += count
                per_pair[name].append(count / (384 * 512))
    rows: dict[str, Any] = {}
    for name in candidates:
        values = per_pair[name]
        worst = max(range(600), key=values.__getitem__)
        rows[name] = {
            "n_real_states": 600,
            "argmax_flips": aggregate[name],
            "total_pixels": total_pixels,
            "aggregate_flip_fraction": aggregate[name] / total_pixels,
            "worst_pair_index": worst,
            "worst_pair_flip_fraction": values[worst],
            "per_pair_flip_fraction": values,
        }
    control = rows["dense_fp16_b1"]
    for row in rows.values():
        row["advisory_no_worse_than_fp16_control"] = (
            row["aggregate_flip_fraction"] <= control["aggregate_flip_fraction"]
            and row["worst_pair_flip_fraction"] <= control["worst_pair_flip_fraction"]
        )
        row["training_gradient_eligible"] = False
        row["label_grade_eligible"] = False
    return {
        "reference": "Torch fp32, one thread, batch1, exact frozen safetensors; NumPy argmax accounting",
        "n_real_states": 600,
        "elapsed_s": time.perf_counter() - started,
        "candidates": rows,
    }


def _make_mlx_step(api: dict[str, Any]) -> tuple[Callable[[], None], dict[str, Any]]:
    mx = api["mx"]
    mx.random.seed(0)
    batch, height, width, channels = 1, 384, 512, 64
    x = mx.random.normal((batch, height, width, channels), dtype=mx.float32)
    weights = [mx.random.normal((64, 3, 3, 64), dtype=mx.float32) * 0.02 for _ in range(4)]

    def loss_fn(parameters: list[Any]) -> Any:
        value = x
        for weight in parameters:
            value = mx.maximum(mx.conv_general(value, weight, padding=1), 0.0)
        return mx.mean(value * value)

    value_and_grad = mx.value_and_grad(loss_fn)

    def step() -> None:
        nonlocal weights
        _, grads = value_and_grad(weights)
        weights = [weight - 1e-4 * grad for weight, grad in zip(weights, grads, strict=True)]
        mx.eval(weights)

    step()
    per_conv_flops = 2 * height * width * 64 * 64 * 3 * 3
    return step, {
        "geometry": [batch, height, width, channels],
        "layers": 4,
        "kernel": [3, 3],
        "dtype": "float32",
        "forward_conv_flops_derived": 4 * per_conv_flops,
        "workload": "four full-resolution conv-ReLU layers plus reverse-mode gradient and SGD update",
        "match_scope": (
            "representative mixed compute/bandwidth Metal load at witness spatial geometry; not the governed trainer"
        ),
    }


def _time_calls(function: Callable[[], None], count: int, barrier: threading.Barrier | None = None) -> list[float]:
    if barrier is not None:
        barrier.wait()
    samples: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return samples


def _powermetrics_start(path: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            "/usr/bin/powermetrics",
            "--samplers",
            "gpu_power,ane_power",
            "--sample-rate",
            "250",
            "--sample-count",
            "8",
            "--format",
            "plist",
            "--output-file",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _concurrency(api: dict[str, Any], frames: Any, models: dict[str, Any], count: int, scratch: Path) -> dict[str, Any]:
    teacher_model = models["dense_fp16_b1"]
    teacher_input = _frame_tensor(api, frames, 0, 1).numpy()

    def teacher() -> None:
        _predict(teacher_model, teacher_input)

    for _ in range(3):
        teacher()
    teacher_solo = _time_calls(teacher, count)
    try:
        mlx_step, load_manifest = _make_mlx_step(api)
        for _ in range(3):
            mlx_step()
    except Exception as exc:
        power_path = scratch / "powermetrics_access_probe.plist"
        try:
            power = subprocess.run(
                [
                    "/usr/bin/powermetrics",
                    "--samplers",
                    "gpu_power,ane_power",
                    "--sample-rate",
                    "250",
                    "--sample-count",
                    "1",
                    "--format",
                    "plist",
                    "--output-file",
                    str(power_path),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            power_receipt = {
                "status": "MEASURED" if power.returncode == 0 and power_path.exists() else "BLOCKED_NOT_MEASURED",
                "returncode": power.returncode,
                "stderr": power.stderr[-4000:],
            }
        except Exception as power_exc:
            power_receipt = {
                "status": "BLOCKED_NOT_MEASURED",
                "error_type": type(power_exc).__name__,
                "error": str(power_exc)[:4000],
            }
        return {
            "status": "BLOCKED_NOT_MEASURED",
            "protocol": "ABBA requested; stopped before concurrent arm because the representative MLX-GPU load could not execute",
            "coreml_compute_units": "CPU_AND_NE requested; placement NOT proved",
            "teacher_solo": _summary(teacher_solo),
            "teacher_concurrent": None,
            "mlx_solo": None,
            "mlx_concurrent": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "powermetrics": power_receipt,
            "verdict_scope": "representative MLX-GPU concurrency on this sandboxed local process only",
            "req_R": "an unsandboxed local process in which mlx.core.metal can acquire the Apple GPU, then the exact ABBA protocol with CPU_AND_NE placement telemetry",
            "admission": {
                "accepted": False,
                "reason": "timing bars not measured and ANE placement not proved",
            },
        }
    mlx_solo = _time_calls(mlx_step, count)
    power_path = scratch / "powermetrics.plist"
    power = _powermetrics_start(power_path)
    concurrent_teacher: list[float] = []
    concurrent_mlx: list[float] = []
    for _ in range(2):
        barrier = threading.Barrier(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(_time_calls, teacher, count, barrier)
            b = pool.submit(_time_calls, mlx_step, count, barrier)
            concurrent_teacher.extend(a.result())
            concurrent_mlx.extend(b.result())
    teacher_solo.extend(_time_calls(teacher, count))
    mlx_solo.extend(_time_calls(mlx_step, count))
    try:
        _, power_stderr = power.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        power.terminate()
        _, power_stderr = power.communicate(timeout=3)
    power_receipt: dict[str, Any] = {
        "returncode": power.returncode,
        "stderr": power_stderr[-4000:],
        "status": "MEASURED" if power.returncode == 0 and power_path.exists() else "BLOCKED_NOT_MEASURED",
    }
    if power_path.exists():
        raw = power_path.read_bytes()
        power_receipt["bytes"] = len(raw)
        power_receipt["sha256"] = sha256_file(power_path)
        power_receipt["plist_sample_count"] = raw.count(b"<plist")
    solo_teacher_summary = _summary(teacher_solo)
    solo_mlx_summary = _summary(mlx_solo)
    concurrent_teacher_summary = _summary(concurrent_teacher)
    concurrent_mlx_summary = _summary(concurrent_mlx)
    placement_proved = False
    admission = admit_concurrency(
        teacher_solo_s=solo_teacher_summary["median_s"],
        teacher_concurrent_s=concurrent_teacher_summary["median_s"],
        mlx_solo_s=solo_mlx_summary["median_s"],
        mlx_concurrent_s=concurrent_mlx_summary["median_s"],
        placement_proved=placement_proved,
    )
    return {
        "protocol": "ABBA: solo teacher + solo MLX; two concurrent rounds; repeated solos",
        "coreml_compute_units": "CPU_AND_NE requested; placement NOT proved",
        "representative_mlx_load": load_manifest,
        "teacher_solo": solo_teacher_summary,
        "teacher_concurrent": concurrent_teacher_summary,
        "mlx_solo": solo_mlx_summary,
        "mlx_concurrent": concurrent_mlx_summary,
        "admission": admission.to_dict(),
        "powermetrics": power_receipt,
    }


def _provenance(api: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "coremltools_version": api["ct"].__version__,
        "torch_version": api["torch"].__version__,
        "mlx_version": getattr(api["mlx"], "__version__", "unknown"),
        "numpy_version": api["np"].__version__,
        "torch_num_threads": api["torch"].get_num_threads(),
        "seed": 0,
        "weights_path": str(WEIGHTS.relative_to(REPO)),
        "weights_sha256": sha256_file(WEIGHTS),
        "gt_cache_path": str(GT_N600.relative_to(REPO)),
        "gt_cache_sha256": sha256_file(GT_N600),
        "throughput_receipt_path": str(THROUGHPUT_RECEIPT.relative_to(REPO)),
        "throughput_receipt_sha256": sha256_file(THROUGHPUT_RECEIPT),
    }


def _run_sidecar(args: argparse.Namespace, api: dict[str, Any], frames: Any) -> int:
    """Run the deterministic frozen-teacher load arm; never launches training."""

    scratch = args.scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)
    model = _load_model(api)
    example = _frame_tensor(api, frames, 0, 1)
    dense = _convert(api, model, example, api["ct"].precision.FLOAT16)
    package = scratch / "dense_fp16_sidecar_b1.mlpackage"
    dense.save(str(package))
    runtime = (
        _compile_runtime(api, package, scratch),
        dense.get_spec().description.output[0].name,
    )
    value = example.numpy()
    receipt_path = args.out_dir / "sidecar_receipt.json"
    prior_elapsed = 0.0
    prior_forwards = 0
    if args.resume_sidecar and receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        prior_elapsed = float(prior.get("elapsed_s", 0.0))
        prior_forwards = int(prior.get("forward_count", 0))
    remaining = max(0.0, float(args.duration_s) - prior_elapsed)
    started = time.monotonic()
    forwards = prior_forwards
    samples: list[float] = []
    checkpoint_deadline = started + 30.0
    while time.monotonic() - started < remaining:
        call_started = time.perf_counter()
        _predict(runtime, value)
        samples.append(time.perf_counter() - call_started)
        forwards += 1
        now = time.monotonic()
        if now >= checkpoint_deadline:
            atomic_json(
                receipt_path,
                base_receipt(
                    schema="ane_frozen_teacher_sidecar.v1",
                    status="in_progress",
                    elapsed_s=prior_elapsed + now - started,
                    forward_count=forwards,
                    latest_window=_summary(samples[-min(64, len(samples)) :]),
                    resumable=True,
                    checkpoint_interval_s=30,
                ),
            )
            checkpoint_deadline = now + 30.0
    elapsed = prior_elapsed + time.monotonic() - started
    atomic_json(
        receipt_path,
        base_receipt(
            schema="ane_frozen_teacher_sidecar.v1",
            status="complete",
            elapsed_s=elapsed,
            requested_duration_s=float(args.duration_s),
            forward_count=forwards,
            current_process_timing=_summary(samples) if samples else None,
            resumed_from_elapsed_s=prior_elapsed,
            resumable=True,
            checkpoint_interval_s=30,
            coreml_compute_units="CPU_AND_NE requested; ANE placement NOT proved",
        ),
    )
    return 0


def _inner(args: argparse.Namespace) -> int:
    api = _imports()
    frames = stored_npy_memmap(GT_N600, "gt_f1.npy")
    require_real_n600(int(frames.shape[0]))
    scratch = args.scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)
    if args.mode == "sidecar":
        return _run_sidecar(args, api, frames)
    receipt = base_receipt(
        schema="ane_unlock_followup_measurements.v1",
        status="in_progress",
        provenance=_provenance(api),
        n_real_states=600,
    )
    atomic_json(args.out_dir / "measurement_receipt.json", receipt)
    try:
        models = _build_models(api, frames, scratch, args.skip_lut)
        footprint = models["footprint"]
        w8_bytes = footprint["packages"]["multifunction"]["package_bytes"]
        footprint["multifunction_package_vs_32mib_cliff"] = account_weight_fit(
            w8_bytes, payload_evidence="MEASURED_WHOLE_MULTIFUNCTION_PACKAGE_BYTES_UPPER_BOUND"
        ).to_dict()
        receipt["footprint"] = footprint
        atomic_json(args.out_dir / "measurement_receipt.json", receipt)
        receipt["batch_tiers"] = _bench_batches(api, frames, models, args.reps)
        atomic_json(args.out_dir / "measurement_receipt.json", receipt)
        receipt["concurrency"] = _concurrency(api, frames, models, args.concurrency_steps, scratch)
        atomic_json(args.out_dir / "measurement_receipt.json", receipt)
        if args.mode in {"all", "fidelity"}:
            receipt["fidelity_n600"] = _fidelity_n600(api, frames, models)
        receipt["status"] = "complete"
        receipt["written_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(args.out_dir / "measurement_receipt.json", receipt)
        print(json.dumps({
            "status": receipt["status"],
            "batch_tiers": receipt.get("batch_tiers"),
            "concurrency": receipt.get("concurrency"),
            "fidelity_n600": receipt.get("fidelity_n600"),
        }, indent=2)[:30000])
        return 0
    except Exception as exc:
        receipt["status"] = "BLOCKED_NOT_MEASURED"
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)[:8000]
        receipt["verdict_scope"] = (
            "this host/SDK/sandbox and frozen model conversion formulation only; settled prior receipts remain valid"
        )
        receipt["req_R"] = "rerun in a clean local process with the same hashes after resolving the recorded blocker"
        atomic_json(args.out_dir / "measurement_receipt.json", receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-offline", action="store_true")
    parser.add_argument("--mode", choices=("all", "build", "fidelity", "sidecar"), default="all")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/private/tmp/pact_ane_followup_models"))
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--concurrency-steps", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=900.0)
    parser.add_argument("--resume-sidecar", action="store_true")
    parser.add_argument("--skip-lut", action="store_true")
    args = parser.parse_args()
    if args.reps < 3 or args.concurrency_steps < 3:
        parser.error("reps and concurrency-steps must be at least 3")
    if args.duration_s <= 0:
        parser.error("duration-s must be positive")
    args.out_dir = args.out_dir.resolve()
    if args.bootstrap_offline:
        return _bootstrap(args)
    return _inner(args)


if __name__ == "__main__":
    raise SystemExit(main())
