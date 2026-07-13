#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-state fp16/bf16 MLX scorer forward+backward and gradient-fidelity probe.

The probe is read-only on an existing level-set run.  It restores the preserved
EMA witness, renders fixed real pair states, and compares fp16/bf16 frozen-
scorer pixel cotangents against fp32.  It performs no optimizer update and
writes an atomic per-pair partial receipt so an interrupted n600 probe resumes.

All outputs are ``[macOS-MLX research-signal]``.  They cannot replace the
NumPy-fp32/CPU-torch verdict or move the frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments", REPO / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.local_acceleration.mlx_training_precision_probe import (  # noqa: E402
    PrecisionGoBars,
    aggregate_pair_gradient_metrics,
    cast_floating_mlx_arrays,
    evaluate_precision_gate,
    gradient_metrics,
)

DEFAULT_RUN_DIR = REPO / "experiments/results/levelset_v752_baseline_20260710T185913Z"
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CHECKPOINT = "levelset_witness_ema_mlx.npz"
DTYPE_NAMES = ("float32", "float16", "bfloat16")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _metal_preflight() -> dict[str, Any]:
    # MLX emits a Metal-cleanup exception at interpreter shutdown when a
    # sandbox has no GPU device.  Isolate that expected failure in a child so
    # the durable parent receipt remains clean and machine-readable.
    script = r'''
import json
import numpy as np
try:
    import mlx.core as mx
    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device
    symbols = {name: hasattr(mx, name) for name in ("float16", "bfloat16", "float32")}
    with temporary_mlx_device("gpu"):
        result = mx.array(np.asarray([1.0], dtype=np.float32)) + mx.array(
            np.asarray([2.0], dtype=np.float32)
        )
        mx.eval(result)
        value = float(np.asarray(result)[0])
    payload = {
        "available": value == 3.0,
        "default_device": str(mx.default_device()),
        "dtype_symbol_support": symbols,
        "error_type": None,
        "error": None,
    }
except Exception as exc:
    symbols = locals().get("symbols", {})
    payload = {
        "available": False,
        "default_device": None,
        "dtype_symbol_support": symbols,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
print(json.dumps(payload, sort_keys=True))
'''
    env = dict(os.environ)
    pythonpath = [str(REPO / "src"), str(REPO), str(REPO / "upstream")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    child = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    rows = [line for line in child.stdout.splitlines() if line.strip().startswith("{")]
    if not rows:
        return {
            "available": False,
            "default_device": None,
            "dtype_symbol_support": {},
            "error_type": "PreflightChildFailure",
            "error": child.stderr.strip()[-1000:] or f"returncode={child.returncode}",
        }
    payload = json.loads(rows[-1])
    payload["child_returncode"] = child.returncode
    return payload


def _host_receipt() -> dict[str, Any]:
    hardware: dict[str, Any] = {}
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"], text=True
        )
        row = json.loads(raw)["SPHardwareDataType"][0]
        # Deliberately select only non-identifying capacity fields; never copy
        # the serial number, platform UUID, or provisioning identifier.
        hardware = {
            "chip_type": row.get("chip_type"),
            "machine_model": row.get("machine_model"),
            "physical_memory": row.get("physical_memory"),
        }
    except (OSError, subprocess.SubprocessError, KeyError, IndexError, TypeError, ValueError):
        hardware = {"chip_type": "UNAVAILABLE", "machine_model": None, "physical_memory": None}

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "mlx_version": importlib.metadata.version("mlx"),
        "python": platform.python_version(),
        "hardware": hardware,
    }


def _base_receipt(run_dir: Path, checkpoint: Path, gt_cache: Path) -> dict[str, Any]:
    return {
        "schema": "cheapen_real95_mlx_precision_n600.v1",
        "written_at_utc": _utc(),
        "lane_id": "lane_cheapen_real95_tilehalo_fp16_20260713",
        "axis": "[macOS-MLX research-signal; NON-PROMOTABLE]",
        "authority": (
            "training-gradient probe only; NumPy-fp32 byte-close and CPU/CUDA exact eval remain untouched"
        ),
        "host": _host_receipt(),
        "provenance": {
            "git_sha": _git_sha(),
            "probe_source": "tools/probe_mlx_real_n600_precision.py",
            "probe_source_sha256": _sha256(Path(__file__).resolve()),
            "metrics_source": "src/tac/local_acceleration/mlx_training_precision_probe.py",
            "metrics_source_sha256": _sha256(
                REPO / "src/tac/local_acceleration/mlx_training_precision_probe.py"
            ),
            "scorer_adapter_source": "src/tac/local_acceleration/mlx_scorer_adapters.py",
            "scorer_adapter_source_sha256": _sha256(
                REPO / "src/tac/local_acceleration/mlx_scorer_adapters.py"
            ),
            "real_state_loader_source": "tools/witness_per_stage_annulus_attribution.py",
            "real_state_loader_source_sha256": _sha256(
                REPO / "tools/witness_per_stage_annulus_attribution.py"
            ),
            "run_dir": str(run_dir.relative_to(REPO)),
            "run_dir_mutated": False,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint),
            "gt_cache": str(gt_cache.relative_to(REPO)),
            "gt_cache_bytes": gt_cache.stat().st_size,
            "gt_cache_sha256": "DEFERRED_UNTIL_METAL_PREFLIGHT_PASSES",
        },
        "go_bars": PrecisionGoBars().to_dict(),
        "current_speed_stack_required": {
            "TAC_MLX_CUSTOM_GROUPED_BACKWARD": os.environ.get(
                "TAC_MLX_CUSTOM_GROUPED_BACKWARD", "1(default)"
            ),
            "precision_policy": "weights+activations low precision; loss reductions fp32",
            "micro_batch": "not used by this serial scorer precision probe",
        },
        "pointer_delta": "ZERO; means-only probe",
    }


def _iter_mlx_arrays(root: Any) -> list[Any]:
    arrays: list[Any] = []
    seen: set[int] = set()

    def walk(value: Any) -> None:
        module = type(value).__module__
        if module.startswith("mlx") and hasattr(value, "dtype") and hasattr(value, "shape"):
            arrays.append(value)
            return
        ident = id(value)
        if ident in seen:
            return
        if isinstance(value, (list, tuple)):
            seen.add(ident)
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            seen.add(ident)
            for item in value.values():
                walk(item)
        elif hasattr(value, "__dict__") and not callable(value):
            seen.add(ident)
            for item in vars(value).values():
                walk(item)

    walk(root)
    return arrays


def _adapter_class_histogram(root: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[int] = set()

    def walk(value: Any) -> None:
        ident = id(value)
        if ident in seen:
            return
        if isinstance(value, (str, bytes, int, float, bool, np.ndarray)):
            return
        module = type(value).__module__
        if module.startswith("mlx") and hasattr(value, "dtype"):
            return
        if module == "tac.local_acceleration.mlx_scorer_adapters":
            name = type(value).__name__
            counts[name] = counts.get(name, 0) + 1
        seen.add(ident)
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif hasattr(value, "__dict__") and not callable(value):
            for item in vars(value).values():
                walk(item)

    walk(root)
    return dict(sorted(counts.items()))


def _build_adapters(dtype_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    import mlx.core as mx

    from tac.local_acceleration.metal_grouped_conv_backward import (
        metal_grouped_conv2d_backend_available,
    )
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    adapters: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    with temporary_mlx_device("gpu"):
        backend = bool(metal_grouped_conv2d_backend_available())
        for name in dtype_names:
            dtype = getattr(mx, name)
            adapter = load_mlx_distortion_scorer_adapter_from_upstream(
                REPO / "upstream", device="cpu"
            )
            cast = cast_floating_mlx_arrays(adapter, dtype)
            arrays = _iter_mlx_arrays(adapter)
            mx.eval(*arrays)
            adapters[name] = adapter
            receipts[name] = {
                "cast": cast,
                "array_dtype_counts": {
                    key: sum(str(array.dtype).split(".")[-1] == key for array in arrays)
                    for key in DTYPE_NAMES
                },
                "adapter_class_histogram": _adapter_class_histogram(adapter),
                "custom_grouped_backward_backend_available": backend,
                "custom_grouped_adapter_count": _adapter_class_histogram(adapter).get(
                    "MLXCustomKernelStridedGroupedConvAdapter", 0
                ),
            }
    return adapters, receipts


def _load_real_state_context(checkpoint: Path, gt_cache: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    import train_witness_realized_through_R_mlx as T
    import witness_per_stage_annulus_attribution as W

    from tac.boundary_math.lever_b_levelset_generator import (
        int8_dequant_params,
        levelset_rgb_forward_numpy,
    )

    params, cfg = W.load_ckpt(checkpoint)
    scalars = W.cfg_scalars(cfg, params)
    if scalars["self_orient"]:
        raise ValueError(
            "probe currently requires the canonical checkpoint's self_orient=OFF; "
            "refuse to fabricate missing live direction-feature trajectory state"
        )
    coords, curvelet = W.build_render_context(scalars)
    deploy = int8_dequant_params(params)
    cache = np.load(gt_cache, allow_pickle=False)
    n_pairs = int(cache["n_pairs"])
    if n_pairs < 600:
        raise ValueError(f"GT cache is n{n_pairs}, not n600")
    # Each compressed member is materialized exactly once (the canonical OOM law).
    lstars = np.asarray(cache["lstars"][:600], dtype=np.int64)
    margins = np.asarray(cache["margins"][:600], dtype=np.float32)
    poses = np.asarray(cache["gt_poses"][:600], dtype=np.float32)

    forward_kwargs = {
        "n_hidden": scalars["n_hidden"],
        "hidden_dim": scalars["hidden_dim"],
        "n_classes": 5,
        "activation": scalars["activation"],
        "softmax_temp": scalars["softmax_temp"],
        "wire_w0": scalars["wire_w0"],
        "wire_s0": scalars["wire_s0"],
        "hosc_beta": scalars["hosc_beta"],
        "hosc_omega": scalars["hosc_omega"],
        "chroma": scalars["chroma"],
    }

    def post_r_pair(pair_index: int) -> np.ndarray:
        frames: list[np.ndarray] = []
        for code_index in (2 * pair_index, 2 * pair_index + 1):
            rgb, _phi = levelset_rgb_forward_numpy(
                deploy, curvelet, deploy["code"][code_index], **forward_kwargs
            )
            rgb = np.asarray(rgb, dtype=np.float32).reshape(
                scalars["render_h"], scalars["render_w"], 3
            )
            camera_u8 = T._torch_R_to_camera_uint8(rgb)
            tensor = torch.from_numpy(np.asarray(camera_u8)).permute(2, 0, 1)[None].float()
            scorer = F.interpolate(
                tensor, size=(384, 512), mode="bilinear", align_corners=False
            )
            frames.append(scorer[0].permute(1, 2, 0).numpy().astype(np.float32))
        return np.stack(frames, axis=0)

    return {
        "n_pairs": 600,
        "lstars": lstars,
        "margins": margins,
        "poses": poses,
        "post_r_pair": post_r_pair,
        "checkpoint_cfg": scalars,
        "gt_cache_n_pairs": n_pairs,
        "coords_shape": list(coords.shape),
        "curvelet_shape": list(curvelet.shape),
    }


def _make_scorer_ops(adapter: Any, dtype_name: str, state: dict[str, Any]):
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    dtype = getattr(mx, dtype_name)
    tau = float(state["checkpoint_cfg"]["softmax_temp"])

    def arrays(pair_index: int, pair_np: np.ndarray):
        pair = mx.array(pair_np, dtype=dtype)
        target = mx.array(state["lstars"][pair_index], dtype=mx.int32)
        pose_target = mx.array(state["poses"][pair_index], dtype=mx.float32)
        return pair, target, pose_target

    def loss(pair: Any, target: Any, pose_target: Any) -> Any:
        f1 = pair[1:2]
        logits = adapter.segnet(f1).astype(mx.float32)
        onehot = mx.eye(5, dtype=mx.float32)[target]
        target_logit = mx.sum(logits * onehot, axis=-1)
        seg = mx.mean(tau * mx.logsumexp(logits / tau, axis=-1) - target_logit)
        yuv = rgb_to_yuv6_mlx(pair[None])
        _, _, h2, w2, _ = yuv.shape
        pose_in = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (1, h2, w2, 12))
        pose = adapter.posenet(pose_in)["pose"][0, : pose_target.shape[-1]].astype(mx.float32)
        pose_mse = mx.mean(mx.square(pose - pose_target))
        return 100.0 * seg + mx.sqrt(10.0 * pose_mse + 1e-8)

    def outputs(pair: Any) -> tuple[Any, Any]:
        logits = adapter.segnet(pair[1:2]).astype(mx.float32)
        yuv = rgb_to_yuv6_mlx(pair[None])
        _, _, h2, w2, _ = yuv.shape
        pose_in = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (1, h2, w2, 12))
        pose = adapter.posenet(pose_in)["pose"][0, :6].astype(mx.float32)
        return logits, pose

    return arrays, loss, mx.value_and_grad(loss), outputs


def _time_mode(
    *,
    pair_indices: list[int],
    state: dict[str, Any],
    ops: tuple[Any, Any, Any, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    make_arrays, loss, value_and_grad, _outputs = ops
    fwd: list[float] = []
    fwd_bwd: list[float] = []
    dtype_receipt: dict[str, str] | None = None
    for pair_index in pair_indices:
        pair_np = state["post_r_pair"](pair_index)
        pair, target, pose_target = make_arrays(pair_index, pair_np)
        for _ in range(warmup):
            value, grad = value_and_grad(pair, target, pose_target)
            mx.eval(value, grad)
        for _ in range(repeats):
            start = time.perf_counter()
            value = loss(pair, target, pose_target)
            mx.eval(value)
            fwd.append(time.perf_counter() - start)
            start = time.perf_counter()
            value, grad = value_and_grad(pair, target, pose_target)
            mx.eval(value, grad)
            fwd_bwd.append(time.perf_counter() - start)
            if dtype_receipt is None:
                dtype_receipt = {
                    "input": str(pair.dtype),
                    "loss": str(value.dtype),
                    "pixel_cotangent": str(grad.dtype),
                }
    median_fwd = statistics.median(fwd)
    median_fb = statistics.median(fwd_bwd)
    return {
        "pair_indices": pair_indices,
        "warmup": warmup,
        "repeats": repeats,
        "median_forward_s": median_fwd,
        "median_forward_backward_s": median_fb,
        "derived_backward_s": max(0.0, median_fb - median_fwd),
        "dtype_receipt": dtype_receipt,
    }


def _quality_mode(
    *,
    candidate_name: str,
    state: dict[str, Any],
    fp32_ops: tuple[Any, Any, Any, Any],
    candidate_ops: tuple[Any, Any, Any, Any],
    quality_pairs: int,
    partial_path: Path,
) -> dict[str, Any]:
    import mlx.core as mx

    prior: dict[str, Any] = {}
    if partial_path.is_file():
        loaded = json.loads(partial_path.read_text())
        if loaded.get("candidate_dtype") == candidate_name:
            prior = {str(row["pair_index"]): row for row in loaded.get("rows", [])}
    rows = list(prior.values())
    complete = {int(row["pair_index"]) for row in rows}
    fp_arrays, _fp_loss, fp_vg, fp_outputs = fp32_ops
    ca_arrays, _ca_loss, ca_vg, ca_outputs = candidate_ops
    for pair_index in range(quality_pairs):
        if pair_index in complete:
            continue
        pair_np = state["post_r_pair"](pair_index)
        fp_pair, fp_target, fp_pose_tgt = fp_arrays(pair_index, pair_np)
        ca_pair, ca_target, ca_pose_tgt = ca_arrays(pair_index, pair_np)
        fp_value, fp_grad = fp_vg(fp_pair, fp_target, fp_pose_tgt)
        ca_value, ca_grad = ca_vg(ca_pair, ca_target, ca_pose_tgt)
        fp_logits, fp_pose = fp_outputs(fp_pair)
        ca_logits, ca_pose = ca_outputs(ca_pair)
        mx.eval(fp_value, fp_grad, ca_value, ca_grad, fp_logits, fp_pose, ca_logits, ca_pose)
        ref = np.asarray(fp_grad, dtype=np.float32)
        cand = np.asarray(ca_grad, dtype=np.float32)
        metrics = gradient_metrics(ref, cand)
        ref64, cand64 = ref.astype(np.float64), cand.astype(np.float64)
        fp_argmax = np.asarray(fp_logits).argmax(axis=-1)
        ca_argmax = np.asarray(ca_logits).argmax(axis=-1)
        row = {
            "pair_index": pair_index,
            **metrics,
            "gradient_dot_fp64": float(np.sum(ref64 * cand64, dtype=np.float64)),
            "reference_sq_fp64": float(np.sum(ref64 * ref64, dtype=np.float64)),
            "candidate_sq_fp64": float(np.sum(cand64 * cand64, dtype=np.float64)),
            "argmax_flip_count_vs_fp32": int(np.count_nonzero(fp_argmax != ca_argmax)),
            "argmax_pixel_count": int(fp_argmax.size),
            "pose_max_abs_vs_fp32": float(
                np.max(np.abs(np.asarray(ca_pose) - np.asarray(fp_pose)), initial=0.0)
            ),
            "fp32_loss": float(np.asarray(fp_value)),
            "candidate_loss": float(np.asarray(ca_value)),
        }
        rows.append(row)
        rows.sort(key=lambda item: int(item["pair_index"]))
        _atomic_json(
            partial_path,
            {
                "schema": "cheapen_real95_mlx_precision_quality_partial.v1",
                "candidate_dtype": candidate_name,
                "required_pairs": quality_pairs,
                "rows": rows,
                "last_completed_at_utc": _utc(),
            },
        )
    selected = rows[:quality_pairs]
    pair_metrics = aggregate_pair_gradient_metrics(selected)
    dot = sum(float(row["gradient_dot_fp64"]) for row in selected)
    ref_sq = sum(float(row["reference_sq_fp64"]) for row in selected)
    cand_sq = sum(float(row["candidate_sq_fp64"]) for row in selected)
    global_cosine = dot / math.sqrt(ref_sq * cand_sq) if ref_sq > 0 and cand_sq > 0 else 0.0
    return {
        "n_pairs": len(selected),
        "pair_gradient_metrics": pair_metrics,
        "global_gradient_cosine": global_cosine,
        "argmax_flip_count_vs_fp32": sum(
            int(row["argmax_flip_count_vs_fp32"]) for row in selected
        ),
        "argmax_pixel_count": sum(int(row["argmax_pixel_count"]) for row in selected),
        "pose_max_abs_vs_fp32": max(
            float(row["pose_max_abs_vs_fp32"]) for row in selected
        ),
        "partial_stage_checkpoint": str(partial_path.relative_to(REPO)),
    }


def run_probe(
    *,
    run_dir: Path,
    checkpoint_name: str,
    gt_cache: Path,
    out: Path,
    candidate_dtypes: list[str],
    timing_pairs: int,
    quality_pairs: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    checkpoint = run_dir / checkpoint_name
    gt_cache = gt_cache.resolve()
    for path in (run_dir, checkpoint, gt_cache):
        if not path.exists():
            raise FileNotFoundError(path)
    receipt = _base_receipt(run_dir, checkpoint, gt_cache)
    receipt["metal_preflight"] = _metal_preflight()
    if not receipt["metal_preflight"]["available"]:
        receipt.update(
            {
                "status": "BLOCKED_NOT_MEASURED",
                "dtype_results": {},
                "lever_b_verdict": {
                    "verdict": "NO_VERDICT_BLOCKED",
                    "cosine": None,
                    "speedup_x": None,
                    "verdict_scope": (
                        "this execution environment only; fp16/bf16 training-path family remains queued"
                    ),
                    "reformulation_queue": [
                        "execute this exact receipt tool from a Metal-enabled host process",
                        "retain fp32 reductions if low-precision loss reduction is unstable",
                    ],
                },
            }
        )
        return receipt

    receipt["provenance"]["gt_cache_sha256"] = _sha256(gt_cache)

    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

    dtype_names = ["float32", *candidate_dtypes]
    with temporary_mlx_device("gpu"):
        adapters, adapter_receipts = _build_adapters(dtype_names)
        state = _load_real_state_context(checkpoint, gt_cache)
        ops = {
            name: _make_scorer_ops(adapters[name], name, state) for name in dtype_names
        }
        timing_indices = np.linspace(0, 599, timing_pairs, dtype=np.int64).tolist()
        timings = {
            name: _time_mode(
                pair_indices=timing_indices,
                state=state,
                ops=ops[name],
                warmup=warmup,
                repeats=repeats,
            )
            for name in dtype_names
        }
        results: dict[str, Any] = {}
        for name in candidate_dtypes:
            partial = out.with_name(f"{out.stem}.{name}.quality_stage.json")
            try:
                quality = _quality_mode(
                    candidate_name=name,
                    state=state,
                    fp32_ops=ops["float32"],
                    candidate_ops=ops[name],
                    quality_pairs=quality_pairs,
                    partial_path=partial,
                )
                gate = evaluate_precision_gate(
                    fp32_seconds=timings["float32"]["median_forward_backward_s"],
                    candidate_seconds=timings[name]["median_forward_backward_s"],
                    global_cosine=quality["global_gradient_cosine"],
                    pair_cosine_min=quality["pair_gradient_metrics"]["cosine_min"],
                    quality_pairs=quality["n_pairs"],
                )
                results[name] = {
                    "status": "MEASURED",
                    "adapter_receipt": adapter_receipts[name],
                    "timing": timings[name],
                    "fp32_timing": timings["float32"],
                    "quality": quality,
                    "gate": gate,
                }
            except Exception as exc:
                results[name] = {
                    "status": "BLOCKED_OR_UNSUPPORTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "adapter_receipt": adapter_receipts.get(name),
                }
        receipt.update(
            {
                "status": "MEASURED",
                "state_receipt": {
                    key: value
                    for key, value in state.items()
                    if key not in {"lstars", "margins", "poses", "post_r_pair"}
                },
                "fp32_adapter_receipt": adapter_receipts["float32"],
                "dtype_results": results,
            }
        )
    measured = [value for value in receipt["dtype_results"].values() if value["status"] == "MEASURED"]
    go = [value for value in measured if value["gate"]["verdict"] == "GO"]
    receipt["lever_b_verdict"] = {
        "verdict": "GO" if go else "NO_GO",
        "winning_dtype": next(
            (name for name, value in receipt["dtype_results"].items() if value in go), None
        ),
        "verdict_scope": "measured dtype policies on this checkpoint and M5-class host",
        "reformulation_queue": (
            []
            if go
            else [
                "fp16 activations with selected fp32 hot layers/reductions",
                "bf16 activations with fp32 explicit SegNet head",
                "layerwise precision waterfill ranked by measured gradient damage",
            ]
        ),
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--candidate-dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--timing-pairs", type=int, default=8)
    parser.add_argument("--quality-pairs", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not (1 <= args.timing_pairs <= 600 and 1 <= args.quality_pairs <= 600):
        raise SystemExit("timing/quality pair counts must be in 1..600")
    out = args.out.resolve()
    if str(out).startswith("/tmp/") or str(out).startswith("/private/tmp/"):
        raise SystemExit("refusing /tmp durable evidence path")
    payload = run_probe(
        run_dir=args.run_dir,
        checkpoint_name=args.checkpoint,
        gt_cache=args.gt_cache,
        out=out,
        candidate_dtypes=args.candidate_dtypes,
        timing_pairs=args.timing_pairs,
        quality_pairs=args.quality_pairs,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    _atomic_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
