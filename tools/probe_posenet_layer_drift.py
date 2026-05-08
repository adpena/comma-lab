#!/usr/bin/env python3
"""Probe PoseNet layer-by-layer device drift.

This tool is a diagnostic tracer for the public scorer, not a score claimant.
It runs ``upstream/modules.py::PoseNet`` on two devices, records selected
activation tensors, and compares CPU/CUDA (or CPU/MPS for local research
smokes) layer by layer.

The critical design point is custody: scorer drift can happen before PoseNet
inside video decode/preprocess, inside PoseNet kernels, or in the final Hydra
head. This tool isolates PoseNet/preprocess drift on a shared input tensor; it
does not replace exact auth eval.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections import OrderedDict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
DEFAULT_MODULE_REGEX = (
    r"^(vision|vision\.stem|vision\.stages\.\d+|"
    r"vision\.stages\.\d+\.blocks\.\d+|vision\.head|"
    r"summarizer|summarizer\.\d+|hydra|hydra\..*)$"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def device_available(device: str) -> tuple[bool, str | None]:
    parsed = torch.device(device)
    if parsed.type == "cpu":
        return True, None
    if parsed.type == "cuda":
        return torch.cuda.is_available(), "torch.cuda.is_available() is false"
    if parsed.type == "mps":
        available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        return available, "torch.backends.mps.is_available() is false"
    return False, f"unsupported device type: {parsed.type}"


def flatten_tensors(value: Any, prefix: str = "out") -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, dict):
        rows: list[tuple[str, torch.Tensor]] = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            rows.extend(flatten_tensors(child, f"{prefix}.{key}"))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for idx, child in enumerate(value):
            rows.extend(flatten_tensors(child, f"{prefix}.{idx}"))
        return rows
    return []


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().to(device="cpu", dtype=torch.float64)
    if t.numel() == 0:
        return {
            "shape": list(t.shape),
            "numel": 0,
            "mean": None,
            "std": None,
            "rms": None,
            "min": None,
            "max": None,
        }
    return {
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "rms": float(torch.sqrt(torch.mean(t * t)).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> dict[str, Any]:
    aa = a.detach().to(device="cpu", dtype=torch.float64)
    bb = b.detach().to(device="cpu", dtype=torch.float64)
    if aa.shape != bb.shape:
        return {
            "shape_match": False,
            "shape_a": list(aa.shape),
            "shape_b": list(bb.shape),
        }
    diff = aa - bb
    abs_diff = diff.abs()
    denom = torch.maximum(aa.abs(), bb.abs()).clamp_min(eps)
    rel = abs_diff / denom
    if diff.numel() == 0:
        return {"shape_match": True, "numel": 0}
    return {
        "shape_match": True,
        "shape": list(aa.shape),
        "numel": int(diff.numel()),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rms_abs": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
        "rms_rel": float(torch.sqrt(torch.mean(rel * rel)).item()),
    }


def capture_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor out of the live graph before downstream in-place ops mutate it."""
    return tensor.detach().to("cpu").clone()


def select_named_modules(
    model: torch.nn.Module,
    *,
    module_regex: str,
    leaf_only: bool,
    max_modules: int | None,
) -> list[tuple[str, torch.nn.Module]]:
    pattern = re.compile(module_regex)
    selected: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if leaf_only and any(module.children()):
            continue
        if not pattern.search(name):
            continue
        selected.append((name, module))
        if max_modules is not None and len(selected) >= max_modules:
            break
    return selected


def module_inventory(model: torch.nn.Module) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    attention_like: list[dict[str, str]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        module_type = type(module).__name__
        type_counts[module_type] = type_counts.get(module_type, 0) + 1
        haystack = f"{name} {module_type}".lower()
        if any(token in haystack for token in ("attention", "softmax", "mha", "self_attn")):
            attention_like.append({"name": name, "type": module_type})
    return {
        "module_type_counts": dict(sorted(type_counts.items())),
        "attention_or_softmax_named_modules": attention_like,
    }


def _load_upstream_posenet(device: torch.device) -> torch.nn.Module:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from modules import PoseNet, posenet_sd_path  # type: ignore
    from safetensors.torch import load_file

    model = PoseNet().eval().to(device=device)
    state = load_file(posenet_sd_path, device=str(device))
    model.load_state_dict(state)
    return model


def _make_input(
    *,
    device: torch.device,
    input_kind: str,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if input_kind == "posenet":
        # PoseNet.forward consumes YUV6-packed two-frame input:
        # (B, 2 * 6, 384/2, 512/2).
        tensor = torch.randint(
            low=0,
            high=256,
            size=(batch_size, 12, 192, 256),
            dtype=torch.uint8,
            generator=generator,
        ).float()
        return tensor.to(device)
    if input_kind == "rgb_sequence":
        # PoseNet.preprocess_input consumes (B, T, C, H, W) RGB float frames.
        tensor = torch.randint(
            low=0,
            high=256,
            size=(batch_size, 2, 3, 874, 1164),
            dtype=torch.uint8,
            generator=generator,
        ).float()
        return tensor.to(device)
    if input_kind == "evaluator_rgb":
        # Evaluator loaders emit raw RGB uint8 batches as (B, T, H, W, C).
        tensor = torch.randint(
            low=0,
            high=256,
            size=(batch_size, 2, 874, 1164, 3),
            dtype=torch.uint8,
            generator=generator,
        ).float()
        return tensor.to(device)
    raise ValueError(f"unsupported input kind: {input_kind}")


def _load_input(path: Path, device: torch.device) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor")
    return tensor.to(device)


def evaluator_rgb_to_channel_first(x: torch.Tensor) -> torch.Tensor:
    """Convert evaluator-loader RGB batches from (B,T,H,W,C) to (B,T,C,H,W)."""
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError(
            "evaluator_rgb input must have shape (B,T,H,W,3); "
            f"got {tuple(x.shape)}"
        )
    return x.permute(0, 1, 4, 2, 3).contiguous()


def prepare_posenet_input_source(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    input_kind: str,
) -> torch.Tensor:
    """Normalize accepted input contracts to PoseNet's packed YUV6 tensor."""
    if input_kind == "posenet":
        return x
    if input_kind == "rgb_sequence":
        return model.preprocess_input(x)
    if input_kind == "evaluator_rgb":
        return model.preprocess_input(evaluator_rgb_to_channel_first(x))
    raise ValueError(f"unsupported input kind: {input_kind}")


def _capture_activations(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    input_kind: str,
    selected: Iterable[tuple[str, torch.nn.Module]],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, torch.Tensor]]:
    activations: OrderedDict[str, torch.Tensor] = OrderedDict()
    handles: list[Any] = []

    def _hook(name: str):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            for key, tensor in flatten_tensors(output):
                activations[f"{name}::{key}"] = capture_tensor(tensor)

        return capture

    for name, module in selected:
        handles.append(module.register_forward_hook(_hook(name)))

    try:
        with torch.inference_mode():
            posenet_input = prepare_posenet_input_source(model, x, input_kind=input_kind)
            activations["__input__::posenet_input"] = capture_tensor(posenet_input)
            normalized = (posenet_input - model._mean) / model._std
            activations["__input__::normalized_posenet_input"] = capture_tensor(normalized)
            output = model(posenet_input)
    finally:
        for handle in handles:
            handle.remove()

    output_tensors = {
        key: capture_tensor(tensor)
        for key, tensor in flatten_tensors(output, prefix="pose_output")
    }
    return activations, output_tensors


def _device_metadata() -> dict[str, Any]:
    cuda_meta: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }
    if torch.cuda.is_available():
        cuda_meta.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name_0": torch.cuda.get_device_name(0),
                "capability_0": list(torch.cuda.get_device_capability(0)),
            }
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda_meta,
        "mps_available": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
    }


def build_probe_report(args: argparse.Namespace) -> dict[str, Any]:
    available_a, reason_a = device_available(args.device_a)
    available_b, reason_b = device_available(args.device_b)
    report: dict[str, Any] = {
        "schema": "posenet_layer_precision_drift_probe.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "diagnostic",
        "diagnostic_kind": "posenet_layer_drift_probe",
        "repo": str(REPO),
        "upstream_modules": str(UPSTREAM / "modules.py"),
        "device_a": args.device_a,
        "device_b": args.device_b,
        "device_a_available": available_a,
        "device_b_available": available_b,
        "device_a_unavailable_reason": reason_a,
        "device_b_unavailable_reason": reason_b,
        "input_kind": args.input_kind,
        "input_tensor": str(args.input_tensor) if args.input_tensor else None,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "module_regex": args.module_regex,
        "leaf_only": args.leaf_only,
        "max_modules": args.max_modules,
        "rms_abs_threshold": args.rms_abs_threshold,
        "environment": _device_metadata(),
        "interpretation_guardrails": [
            "This probe isolates PoseNet/preprocess layer drift on a shared tensor; it does not score archives.",
            "A CPU/CUDA auth-eval gap can also arise from DALI-vs-AV video decode before PoseNet.",
            "Hook ordering is diagnostic: parent/child modules may both fire, so the first exceeding activation localizes the earliest captured surface, not a formal causal layer.",
            "Use exact paired auth eval JSONs for score claims; use this tool to localize mechanism only.",
        ],
    }
    if not (available_a and available_b):
        report["comparison_available"] = False
        if available_a:
            model = _load_upstream_posenet(torch.device(args.device_a))
            selected = select_named_modules(
                model,
                module_regex=args.module_regex,
                leaf_only=args.leaf_only,
                max_modules=args.max_modules,
            )
            report["selected_modules"] = [
                {"name": name, "type": type(module).__name__}
                for name, module in selected
            ]
            report["module_inventory"] = module_inventory(model)
        return report

    device_a = torch.device(args.device_a)
    device_b = torch.device(args.device_b)
    model_a = _load_upstream_posenet(device_a)
    model_b = _load_upstream_posenet(device_b)
    selected_a = select_named_modules(
        model_a,
        module_regex=args.module_regex,
        leaf_only=args.leaf_only,
        max_modules=args.max_modules,
    )
    selected_b = select_named_modules(
        model_b,
        module_regex=args.module_regex,
        leaf_only=args.leaf_only,
        max_modules=args.max_modules,
    )
    names_a = [name for name, _ in selected_a]
    names_b = [name for name, _ in selected_b]
    if names_a != names_b:
        raise RuntimeError("selected module names differ across devices")

    if args.input_tensor:
        x_a = _load_input(args.input_tensor, device_a)
        x_b = _load_input(args.input_tensor, device_b)
    else:
        x_a = _make_input(
            device=device_a,
            input_kind=args.input_kind,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        x_b = _make_input(
            device=device_b,
            input_kind=args.input_kind,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    act_a, out_a = _capture_activations(model_a, x_a, input_kind=args.input_kind, selected=selected_a)
    act_b, out_b = _capture_activations(model_b, x_b, input_kind=args.input_kind, selected=selected_b)
    rows: list[dict[str, Any]] = []
    first_exceeding = None
    for idx, key in enumerate(act_a.keys()):
        if key not in act_b:
            rows.append({"order": idx, "activation": key, "missing_on_device_b": True})
            continue
        comparison = compare_tensors(act_a[key], act_b[key])
        row = {
            "order": idx,
            "activation": key,
            "device_a_stats": tensor_stats(act_a[key]),
            "device_b_stats": tensor_stats(act_b[key]),
            "comparison": comparison,
        }
        if (
            first_exceeding is None
            and comparison.get("shape_match") is True
            and float(comparison.get("rms_abs", 0.0)) > args.rms_abs_threshold
        ):
            first_exceeding = row
        rows.append(row)

    output_rows = []
    for key, tensor_a in out_a.items():
        tensor_b = out_b.get(key)
        if tensor_b is None:
            output_rows.append({"output": key, "missing_on_device_b": True})
        else:
            output_rows.append(
                {
                    "output": key,
                    "device_a_stats": tensor_stats(tensor_a),
                    "device_b_stats": tensor_stats(tensor_b),
                    "comparison": compare_tensors(tensor_a, tensor_b),
                }
            )

    report.update(
        {
            "comparison_available": True,
            "selected_modules": [
                {"name": name, "type": type(module).__name__}
                for name, module in selected_a
            ],
            "module_inventory": module_inventory(model_a),
            "activation_rows": rows,
            "output_rows": output_rows,
            "first_captured_activation_exceeding_rms_abs_threshold": first_exceeding,
            "first_activation_exceeding_rms_abs_threshold": first_exceeding,
            "localization_caveat": (
                "Forward hooks can capture nested parent and child modules; "
                "the first threshold crossing is the first captured surface, "
                "not a proof of the first causal operation."
            ),
        }
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-a", default="cpu")
    parser.add_argument("--device-b", default="cuda")
    parser.add_argument(
        "--input-kind",
        choices=["posenet", "rgb_sequence", "evaluator_rgb"],
        default="posenet",
    )
    parser.add_argument("--input-tensor", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--module-regex", default=DEFAULT_MODULE_REGEX)
    parser.add_argument("--leaf-only", action="store_true")
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--rms-abs-threshold", type=float, default=1e-6)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_probe_report(args)
    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.get("comparison_available") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
