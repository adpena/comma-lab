#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded block profiler for the canonical frozen EfficientNet-B2 SegNet.

This tool profiles one explicitly selected real-video pair through the exact
``upstream/modules.py::SegNet`` last-frame path.  Its timings are deliberately
labelled as instrumented, local macOS-CPU advisory evidence: hooks and saved-
tensor accounting perturb wall time, and this receipt is neither a score nor a
Metal/CUDA throughput benchmark.

Example::

    .venv/bin/python tools/profile_segnet_blocks.py \
      --video upstream/videos/0.mkv --pair-index 0 \
      --threads 6 --warmups 1 --samples 1 --seed 0 \
      --out experiments/results/segnet_block_profile_20260712/profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO = DEFAULT_UPSTREAM / "videos" / "0.mkv"
TOOL_PATH = Path(__file__).resolve()
SCHEMA = "frozen_segnet_block_profile_v3"
AXIS = "[macOS-CPU advisory profiling; torch-fp32; no MPS/CUDA authority]"
OWNER_RULE = (
    "Each saved-tensor event is attributed to the innermost active selected "
    "top-level block, or 'unattributed' when it occurs in model glue. Logical "
    "bytes count every save. Each backing storage is counted once globally "
    "and assigned to the owner of its first observed save in deterministic "
    "forward execution order. These values are not peak RSS."
)
OVERHEAD_WARNING = (
    "Forward/backward hooks and saved_tensors_hooks perturb execution and memory. "
    "This is not an uninstrumented throughput benchmark. Use these timings only "
    "for within-receipt block attribution; use canonical uninstrumented grouped-"
    "backward measurements for Apple training throughput."
)
_TRANSIENT_PREFIXES = (
    Path("/tmp"),
    Path("/private/tmp"),
    Path("/var/tmp"),
    Path("/private/var/tmp"),
    Path("/private/var/folders"),
)
_DURABLE_ROOTS = (
    REPO_ROOT,
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: Any) -> str:
    import numpy as np

    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_durable_output(path: Path) -> Path:
    """Resolve and validate a durable repo/SSD output path.

    The production CLI exposes no bypass.  The atomic writer is kept separate
    so its mechanics can be tested in pytest's transient ``tmp_path``.
    """

    resolved = path.expanduser().resolve(strict=False)
    if any(_is_relative_to(resolved, prefix) for prefix in _TRANSIENT_PREFIXES):
        raise ValueError(f"refusing transient durable output path: {resolved}")
    if not any(_is_relative_to(resolved, root.resolve()) for root in _DURABLE_ROOTS):
        raise ValueError(
            "output must be under the repository or an approved Pact SSD root; "
            f"got {resolved}"
        )
    return resolved


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON using a same-directory file+directory fsync and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def select_profile_blocks(model: Any) -> list[tuple[str, Any]]:
    """Select only disjoint stem/encoder/decoder/head attribution units."""

    try:
        encoder_model = model.encoder.model
        encoder_blocks: Iterable[Any] = encoder_model.blocks
        decoder_blocks: Iterable[Any] = model.decoder.blocks
        stem = encoder_model.conv_stem
        stem_norm = encoder_model.bn1
        head = model.segmentation_head
    except AttributeError as exc:
        raise ValueError(
            "model must expose encoder.model.conv_stem, encoder.model.bn1, "
            "encoder.model.blocks, "
            "decoder.blocks, and segmentation_head"
        ) from exc

    selected = [("encoder.stem", stem)]
    selected.append(("encoder.bn1", stem_norm))
    selected.extend(
        (f"encoder.block{index}", block)
        for index, block in enumerate(encoder_blocks)
    )
    selected.extend(
        (f"decoder.block{index}", block)
        for index, block in enumerate(decoder_blocks)
    )
    selected.append(("segmentation_head", head))

    ids = [id(module) for _, module in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected profiler blocks must be unique modules")
    for index, (name, module) in enumerate(selected):
        descendants = {id(child) for child in module.modules() if child is not module}
        for other_index, (other_name, other) in enumerate(selected):
            if index != other_index and id(other) in descendants:
                raise ValueError(
                    f"selected blocks overlap: {name} contains {other_name}"
                )
    return selected


def _tensor_output_stats(output: Any) -> tuple[list[dict[str, Any]], int]:
    tensors: list[Any] = []

    def visit(value: Any) -> None:
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])

    visit(output)
    rows = [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "logical_bytes": int(tensor.numel() * tensor.element_size()),
        }
        for tensor in tensors
    ]
    return rows, sum(row["logical_bytes"] for row in rows)


def _storage_key(tensor: Any) -> tuple[str, int | None, int, int] | None:
    try:
        storage = tensor.untyped_storage()
        return (
            str(tensor.device.type),
            tensor.device.index,
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
    except (AttributeError, RuntimeError):
        return None


def _profile_one_sample(
    model: Any,
    model_input: Any,
    blocks: Sequence[tuple[str, Any]],
    *,
    loss_fn: Callable[[Any], Any],
) -> dict[str, Any]:
    import torch

    block_rows: dict[str, dict[str, Any]] = {
        name: {"name": name, "module_type": type(module).__name__}
        for name, module in blocks
    }
    forward_started: dict[str, int] = {}
    backward_started: dict[str, int] = {}
    owner_stack: list[str] = []
    saved_logical: defaultdict[str, int] = defaultdict(int)
    storage_first_owner: dict[tuple[str, int | None, int, int], str] = {}
    handles: list[Any] = []

    def make_forward_pre(name: str) -> Callable[..., None]:
        def hook(_module: Any, _args: Any) -> None:
            if owner_stack:
                raise RuntimeError(
                    "selected attribution blocks overlapped during execution: "
                    f"active={owner_stack[-1]} entering={name}"
                )
            owner_stack.append(name)
            forward_started[name] = time.perf_counter_ns()

        return hook

    def make_forward_post(name: str) -> Callable[..., None]:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            elapsed = (time.perf_counter_ns() - forward_started[name]) / 1_000_000.0
            outputs, output_bytes = _tensor_output_stats(output)
            block_rows[name]["forward_ms"] = elapsed
            block_rows[name]["outputs"] = outputs
            block_rows[name]["output_logical_bytes"] = output_bytes
            if not owner_stack or owner_stack.pop() != name:
                raise RuntimeError(f"profiler owner stack corrupted while leaving {name}")

        return hook

    def make_backward_pre(name: str) -> Callable[..., None]:
        def hook(_module: Any, _grad_output: Any) -> None:
            backward_started[name] = time.perf_counter_ns()

        return hook

    def make_backward_post(name: str) -> Callable[..., None]:
        def hook(_module: Any, _grad_input: Any, _grad_output: Any) -> None:
            elapsed = (time.perf_counter_ns() - backward_started[name]) / 1_000_000.0
            block_rows[name]["backward_ms"] = elapsed

        return hook

    for name, module in blocks:
        handles.extend(
            (
                module.register_forward_pre_hook(make_forward_pre(name)),
                module.register_forward_hook(make_forward_post(name)),
                module.register_full_backward_pre_hook(make_backward_pre(name)),
                module.register_full_backward_hook(make_backward_post(name)),
            )
        )

    def pack_saved(tensor: Any) -> Any:
        owner = owner_stack[-1] if owner_stack else "unattributed"
        saved_logical[owner] += int(tensor.numel() * tensor.element_size())
        key = _storage_key(tensor)
        if key is not None and key not in storage_first_owner:
            storage_first_owner[key] = owner
        return tensor

    def unpack_saved(tensor: Any) -> Any:
        return tensor

    sample_input = model_input.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    try:
        forward_start = time.perf_counter_ns()
        with torch.autograd.graph.saved_tensors_hooks(pack_saved, unpack_saved):
            output = model(sample_input)
        forward_ms = (time.perf_counter_ns() - forward_start) / 1_000_000.0
        loss = loss_fn(output)
        backward_start = time.perf_counter_ns()
        loss.backward()
        backward_ms = (time.perf_counter_ns() - backward_start) / 1_000_000.0
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        name
        for name in block_rows
        if "forward_ms" not in block_rows[name] or "backward_ms" not in block_rows[name]
    ]
    if missing:
        raise RuntimeError(f"selected blocks did not participate in both passes: {missing}")

    storage_bytes_by_owner: defaultdict[str, int] = defaultdict(int)
    for key, owner in storage_first_owner.items():
        storage_bytes_by_owner[owner] += key[-1]
    owners = sorted(set(saved_logical) | set(storage_bytes_by_owner))
    saved_rows = [
        {
            "owner": owner,
            "logical_saved_bytes": int(saved_logical[owner]),
            "first_owner_unique_storage_bytes": int(storage_bytes_by_owner[owner]),
        }
        for owner in owners
    ]
    for name, row in block_rows.items():
        row["total_ms"] = row["forward_ms"] + row["backward_ms"]
        row["logical_saved_bytes"] = int(saved_logical[name])
        row["first_owner_unique_storage_bytes"] = int(storage_bytes_by_owner[name])

    selected_forward_ms = sum(float(row["forward_ms"]) for row in block_rows.values())
    selected_backward_ms = sum(float(row["backward_ms"]) for row in block_rows.values())
    selected_total_ms = selected_forward_ms + selected_backward_ms
    end_to_end_total_ms = forward_ms + backward_ms
    unattributed_forward_ms = forward_ms - selected_forward_ms
    unattributed_backward_ms = backward_ms - selected_backward_ms
    unattributed_total_ms = end_to_end_total_ms - selected_total_ms

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_plus_backward_ms": forward_ms + backward_ms,
        "selected_block_forward_ms_sum": selected_forward_ms,
        "selected_block_backward_ms_sum": selected_backward_ms,
        "selected_block_total_ms_sum": selected_total_ms,
        "selected_timing_coverage": (
            selected_total_ms / end_to_end_total_ms if end_to_end_total_ms else 0.0
        ),
        "unattributed_forward_residual_ms": unattributed_forward_ms,
        "unattributed_backward_residual_ms": unattributed_backward_ms,
        "unattributed_total_residual_ms": unattributed_total_ms,
        "loss": float(loss.detach().cpu()),
        "input_gradient_sha256": _sha256_tensor(sample_input.grad),
        "blocks": list(block_rows.values()),
        "saved_tensors": {
            "owner_rule": OWNER_RULE,
            "owners": saved_rows,
            "logical_saved_bytes_total": int(sum(saved_logical.values())),
            "unique_storage_bytes_total": int(sum(key[-1] for key in storage_first_owner)),
        },
    }


def aggregate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Deterministically aggregate raw samples without discarding them."""

    if not samples:
        raise ValueError("at least one profile sample is required")
    block_names = [row["name"] for row in samples[0]["blocks"]]
    if any([row["name"] for row in sample["blocks"]] != block_names for sample in samples):
        raise ValueError("profile sample block order changed")

    total_medians = {
        key: statistics.median(float(sample[key]) for sample in samples)
        for key in ("forward_ms", "backward_ms", "forward_plus_backward_ms")
    }
    forward_share_samples = [
        float(sample["forward_ms"]) / float(sample["forward_plus_backward_ms"])
        for sample in samples
    ]
    backward_share_samples = [
        float(sample["backward_ms"]) / float(sample["forward_plus_backward_ms"])
        for sample in samples
    ]
    backward_only_removal_ceiling_samples = [
        float(sample["forward_plus_backward_ms"]) / float(sample["forward_ms"])
        for sample in samples
    ]
    aggregate_blocks: list[dict[str, Any]] = []
    for block_index, name in enumerate(block_names):
        rows = [sample["blocks"][block_index] for sample in samples]
        forward_values = [float(row["forward_ms"]) for row in rows]
        backward_values = [float(row["backward_ms"]) for row in rows]
        total_values = [float(row["total_ms"]) for row in rows]
        logical_values = [int(row["logical_saved_bytes"]) for row in rows]
        storage_values = [int(row["first_owner_unique_storage_bytes"]) for row in rows]
        paired_share_values = [
            float(row["total_ms"]) / float(sample["forward_plus_backward_ms"])
            for sample, row in zip(samples, rows, strict=True)
        ]
        aggregate_blocks.append(
            {
                "name": name,
                "module_type": rows[0]["module_type"],
                "outputs": rows[0]["outputs"],
                "output_logical_bytes": rows[0]["output_logical_bytes"],
                "forward_ms_samples": forward_values,
                "backward_ms_samples": backward_values,
                "total_ms_samples": total_values,
                "forward_ms_median": statistics.median(forward_values),
                "backward_ms_median": statistics.median(backward_values),
                "total_ms_median": statistics.median(total_values),
                "share_of_end_to_end_paired_samples": paired_share_values,
                "share_of_end_to_end_paired_median": statistics.median(
                    paired_share_values
                ),
                "logical_saved_bytes_samples": logical_values,
                "logical_saved_bytes_median": statistics.median(logical_values),
                "first_owner_unique_storage_bytes_samples": storage_values,
                "first_owner_unique_storage_bytes_median": statistics.median(
                    storage_values
                ),
            }
        )

    profiled_block_total = sum(row["total_ms_median"] for row in aggregate_blocks)
    end_to_end_total = total_medians["forward_plus_backward_ms"]
    for row in aggregate_blocks:
        row["share_of_profiled_block_median_total"] = (
            row["total_ms_median"] / profiled_block_total
            if profiled_block_total
            else 0.0
        )
        row["share_of_end_to_end_median"] = (
            row["total_ms_median"] / end_to_end_total if end_to_end_total else 0.0
        )

    logical_totals = [
        int(sample["saved_tensors"]["logical_saved_bytes_total"])
        for sample in samples
    ]
    storage_totals = [
        int(sample["saved_tensors"]["unique_storage_bytes_total"])
        for sample in samples
    ]
    selected_forward_totals = [
        float(sample["selected_block_forward_ms_sum"]) for sample in samples
    ]
    selected_backward_totals = [
        float(sample["selected_block_backward_ms_sum"]) for sample in samples
    ]
    selected_totals = [float(sample["selected_block_total_ms_sum"]) for sample in samples]
    timing_coverage = [float(sample["selected_timing_coverage"]) for sample in samples]
    unattributed_forward = [
        float(sample["unattributed_forward_residual_ms"]) for sample in samples
    ]
    unattributed_backward = [
        float(sample["unattributed_backward_residual_ms"]) for sample in samples
    ]
    unattributed_total = [
        float(sample["unattributed_total_residual_ms"]) for sample in samples
    ]

    def saved_owner_value(sample: dict[str, Any], key: str) -> int:
        for row in sample["saved_tensors"]["owners"]:
            if row["owner"] == "unattributed":
                return int(row[key])
        return 0

    unattributed_logical = [
        saved_owner_value(sample, "logical_saved_bytes") for sample in samples
    ]
    unattributed_storage = [
        saved_owner_value(sample, "first_owner_unique_storage_bytes")
        for sample in samples
    ]
    return {
        "sample_count": len(samples),
        "raw_samples": list(samples),
        "forward_ms_samples": [float(sample["forward_ms"]) for sample in samples],
        "backward_ms_samples": [float(sample["backward_ms"]) for sample in samples],
        "forward_plus_backward_ms_samples": [
            float(sample["forward_plus_backward_ms"]) for sample in samples
        ],
        "forward_ms_median": total_medians["forward_ms"],
        "backward_ms_median": total_medians["backward_ms"],
        "forward_plus_backward_ms_median": total_medians[
            "forward_plus_backward_ms"
        ],
        "forward_share_samples": forward_share_samples,
        "forward_share_median": statistics.median(forward_share_samples),
        "backward_share_samples": backward_share_samples,
        "backward_share_median": statistics.median(backward_share_samples),
        "backward_only_removal_ceiling_samples": backward_only_removal_ceiling_samples,
        "backward_only_removal_ceiling_median": statistics.median(
            backward_only_removal_ceiling_samples
        ),
        "profiled_block_ms_median_sum": profiled_block_total,
        "selected_block_forward_ms_sum_samples": selected_forward_totals,
        "selected_block_forward_ms_sum_median": statistics.median(
            selected_forward_totals
        ),
        "selected_block_backward_ms_sum_samples": selected_backward_totals,
        "selected_block_backward_ms_sum_median": statistics.median(
            selected_backward_totals
        ),
        "selected_block_total_ms_sum_samples": selected_totals,
        "selected_block_total_ms_sum_median": statistics.median(selected_totals),
        "selected_timing_coverage_samples": timing_coverage,
        "selected_timing_coverage_median": statistics.median(timing_coverage),
        "unattributed_time": {
            "definition": (
                "End-to-end hook-instrumented time minus the sum of disjoint selected "
                "block hook times in the same paired sample; this is a residual, not "
                "a separately timed module."
            ),
            "forward_residual_ms_samples": unattributed_forward,
            "forward_residual_ms_median": statistics.median(unattributed_forward),
            "backward_residual_ms_samples": unattributed_backward,
            "backward_residual_ms_median": statistics.median(unattributed_backward),
            "total_residual_ms_samples": unattributed_total,
            "total_residual_ms_median": statistics.median(unattributed_total),
        },
        "blocks": aggregate_blocks,
        "saved_tensors": {
            "owner_rule": OWNER_RULE,
            "logical_saved_bytes_samples": logical_totals,
            "logical_saved_bytes_median": statistics.median(logical_totals),
            "unique_storage_bytes_samples": storage_totals,
            "unique_storage_bytes_median": statistics.median(storage_totals),
            "unattributed_logical_saved_bytes_samples": unattributed_logical,
            "unattributed_logical_saved_bytes_median": statistics.median(
                unattributed_logical
            ),
            "unattributed_unique_storage_bytes_samples": unattributed_storage,
            "unattributed_unique_storage_bytes_median": statistics.median(
                unattributed_storage
            ),
            "not_peak_rss": True,
        },
    }


def profile_blocks(
    model: Any,
    model_input: Any,
    *,
    warmups: int,
    sample_count: int,
    loss_fn: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Profile a compatible model; public for tiny-network regression tests."""

    import torch

    if warmups < 0:
        raise ValueError("warmups must be >= 0")
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    blocks = select_profile_blocks(model)
    effective_loss = loss_fn or (lambda output: output.float().square().mean())
    for _ in range(warmups):
        warm_input = model_input.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        effective_loss(model(warm_input)).backward()
    samples = [
        _profile_one_sample(model, model_input, blocks, loss_fn=effective_loss)
        for _ in range(sample_count)
    ]
    result = aggregate_samples(samples)
    result["warmup_count"] = warmups
    result["loss_definition"] = "mean(square(float32_logits)) for input-gradient exercise"
    result["timing_overhead_warning"] = OVERHEAD_WARNING
    result["device"] = str(model_input.device)
    result["dtype"] = str(model_input.dtype)
    result["input_shape"] = list(model_input.shape)
    result["torch_grad_enabled"] = bool(torch.is_grad_enabled())
    return result


def configure_determinism(seed: int, threads: int, interop_threads: int) -> dict[str, Any]:
    if threads < 1 or interop_threads < 1:
        raise ValueError("thread counts must be >= 1")
    # Python reads PYTHONHASHSEED before interpreter startup. Record the actual
    # inherited value instead of pretending that a runtime env mutation can
    # retroactively reseed hash randomization. This profiler sorts all mappings,
    # so its accounting does not depend on hash iteration order.
    inherited_python_hash_seed = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "python_random_seed": seed,
        "python_hash_seed_at_process_start": inherited_python_hash_seed,
        "python_hash_seed_child_env": os.environ["PYTHONHASHSEED"],
        "hash_order_independence": "all profiler-owned mapping traversals are explicitly sorted",
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _ensure_import_paths(upstream_dir: Path) -> None:
    for path in (REPO_ROOT, REPO_ROOT / "src", upstream_dir):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def load_frozen_segnet(upstream_dir: Path) -> tuple[Any, Path]:
    import torch
    from safetensors.torch import load_file

    _ensure_import_paths(upstream_dir)
    from modules import SegNet

    weight_path = upstream_dir / "models" / "segnet.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"canonical SegNet weights not found: {weight_path}")
    model = SegNet().eval().to("cpu")
    model.load_state_dict(load_file(weight_path, device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze every SegNet parameter")
    if any(buffer.device.type != "cpu" for buffer in model.buffers()):
        raise RuntimeError("SegNet buffers must remain on CPU")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError(f"expected torch default fp32, got {torch.get_default_dtype()}")
    return model, weight_path


def decode_real_pair(
    video_path: Path, pair_index: int, *, upstream_dir: Path = DEFAULT_UPSTREAM
) -> tuple[Any, dict[str, Any]]:
    if pair_index < 0:
        raise ValueError("pair_index must be >= 0")
    _ensure_import_paths(upstream_dir)
    import av
    import torch
    from frame_utils import yuv420_to_rgb

    wanted = {2 * pair_index, 2 * pair_index + 1}
    decoded: list[Any] = []
    identities: list[dict[str, Any]] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    try:
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in wanted:
                if frame_index > max(wanted):
                    break
                continue
            rgb = yuv420_to_rgb(frame)
            if rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise RuntimeError(
                    f"canonical decoder returned unexpected frame: {rgb.shape}/{rgb.dtype}"
                )
            decoded.append(rgb)
            identities.append(
                {
                    "decoded_frame_index": frame_index,
                    "pts": frame.pts,
                    "dts": frame.dts,
                    "time_seconds": None if frame.time is None else float(frame.time),
                    "time_base": None
                    if frame.time_base is None
                    else str(frame.time_base),
                    "key_frame": bool(frame.key_frame),
                    "pict_type": str(frame.pict_type),
                    "pixel_format": None if frame.format is None else frame.format.name,
                    "decoded_rgb_shape": list(rgb.shape),
                    "decoded_rgb_dtype": str(rgb.dtype),
                    "decoded_rgb_min": int(rgb.min()),
                    "decoded_rgb_max": int(rgb.max()),
                    "decoded_rgb_sha256": _sha256_tensor(rgb),
                }
            )
            if len(decoded) == 2:
                break
    finally:
        container.close()
    if len(decoded) != 2:
        raise ValueError(f"video does not contain complete non-overlapping pair {pair_index}")
    pair_btchw = torch.stack(decoded).unsqueeze(0).permute(0, 1, 4, 2, 3).float()
    return pair_btchw, {"frames": identities, "pair_btchw_shape": list(pair_btchw.shape)}


def _git_custody() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        text=True,
    )
    tool_rel = str(TOOL_PATH.relative_to(REPO_ROOT))
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "dirty": bool(status),
        "status_entry_count": len(status.splitlines()),
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tool_status_rows": [line for line in status.splitlines() if tool_rel in line],
    }


def _source_file_custody(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(REPO_ROOT))
        if _is_relative_to(resolved, REPO_ROOT)
        else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _dependency_versions() -> dict[str, str]:
    distributions = {
        "av": "av",
        "numpy": "numpy",
        "safetensors": "safetensors",
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "timm": "timm",
        "torch": "torch",
    }
    return {
        key: importlib.metadata.version(distribution)
        for key, distribution in distributions.items()
    }


def _environment_custody() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "dependency_versions": _dependency_versions(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _invocation_custody(raw_argv: Sequence[str], args: argparse.Namespace) -> dict[str, Any]:
    executable_argv = [sys.executable, str(TOOL_PATH), *raw_argv]
    return {
        "cwd": str(Path.cwd()),
        "argv": executable_argv,
        "shell_command": shlex.join(executable_argv),
        "parsed_config": {
            "upstream_dir": str(args.upstream_dir),
            "video": str(args.video),
            "pair_index": args.pair_index,
            "seed": args.seed,
            "threads": args.threads,
            "interop_threads": args.interop_threads,
            "warmups": args.warmups,
            "samples": args.samples,
            "out": str(args.out),
        },
    }


def build_real_report(args: argparse.Namespace, raw_argv: Sequence[str]) -> dict[str, Any]:
    started = _utc_now()
    determinism = configure_determinism(args.seed, args.threads, args.interop_threads)
    upstream_dir = args.upstream_dir.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    model, weight_path = load_frozen_segnet(upstream_dir)
    pair_btchw, decode_custody = decode_real_pair(
        video_path, args.pair_index, upstream_dir=upstream_dir
    )
    model_input = model.preprocess_input(pair_btchw).detach().contiguous()
    if list(model_input.shape) != [1, 3, 384, 512]:
        raise RuntimeError(
            f"canonical SegNet preprocessing shape drift: {list(model_input.shape)}"
        )
    if model_input.dtype.__str__() != "torch.float32":
        raise RuntimeError(f"canonical model input must be fp32, got {model_input.dtype}")

    profiled = profile_blocks(
        model,
        model_input,
        warmups=args.warmups,
        sample_count=args.samples,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    frozen_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    return {
        "schema": SCHEMA,
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "training_loop_speedup_claim": False,
            "metal_cuda_ordering_claim": False,
            "purpose": "reproducible frozen-SegNet block attribution only",
            "verdict_scope": "single local pair/profile configuration",
        },
        "warnings": {
            "hook_overhead": OVERHEAD_WARNING,
            "saved_tensor_bytes": (
                "Logical saved bytes and first-owner unique-storage bytes are "
                "autograd accounting, not allocator peak or process RSS."
            ),
        },
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "invocation": _invocation_custody(raw_argv, args),
        "determinism": determinism,
        "environment": _environment_custody(),
        "git": _git_custody(),
        "tool": {
            "path": str(TOOL_PATH.relative_to(REPO_ROOT)),
            "bytes": TOOL_PATH.stat().st_size,
            "sha256": _sha256_file(TOOL_PATH),
        },
        "source_files": {
            "upstream_modules": _source_file_custody(upstream_dir / "modules.py"),
            "upstream_frame_utils": _source_file_custody(
                upstream_dir / "frame_utils.py"
            ),
            "uv_lock": _source_file_custody(REPO_ROOT / "uv.lock"),
            "pyproject": _source_file_custody(REPO_ROOT / "pyproject.toml"),
        },
        "model": {
            "class": "upstream/modules.py::SegNet",
            "architecture": (
                "smp.Unet('tu-efficientnet_b2', classes=5, activation=None, "
                "encoder_weights=None)"
            ),
            "device": "cpu",
            "dtype": "torch.float32",
            "eval_mode": not model.training,
            "parameter_count": parameter_count,
            "frozen_parameter_count": frozen_parameter_count,
            "all_parameters_frozen": parameter_count == frozen_parameter_count,
            "weights_path": str(weight_path.relative_to(REPO_ROOT)),
            "weights_bytes": weight_path.stat().st_size,
            "weights_sha256": _sha256_file(weight_path),
        },
        "input": {
            "video_path": str(video_path.relative_to(REPO_ROOT))
            if _is_relative_to(video_path, REPO_ROOT)
            else str(video_path),
            "video_bytes": video_path.stat().st_size,
            "video_sha256": _sha256_file(video_path),
            "pair_index": args.pair_index,
            "pair_policy": "non-overlapping decoded frames [2*p, 2*p+1]",
            "canonical_preprocess": (
                "upstream yuv420_to_rgb -> BHWC/BTCHW rearrange -> pair[:, -1] "
                "-> torch bilinear resize to (384,512)"
            ),
            "decoded": decode_custody,
            "model_input_shape": list(model_input.shape),
            "model_input_dtype": str(model_input.dtype),
            "model_input_min": float(model_input.min()),
            "model_input_max": float(model_input.max()),
            "model_input_sha256": _sha256_tensor(model_input),
        },
        "profile": profiled,
        "limitations": [
            "Local macOS-arm64 CPU only; no Metal or contest-CUDA ordering follows.",
            "One selected real pair does not establish training-loop or n600 behavior.",
            "The synthetic mean-square-logit scalar only exercises the exact input-gradient graph.",
            "Hook-instrumented timings are not an uninstrumented throughput benchmark.",
            "No d_seg, score, frontier, promotion, or family verdict is produced.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.pair_index < 0:
        parser.error("--pair-index must be >= 0")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.threads < 1 or args.interop_threads < 1:
        parser.error("thread counts must be >= 1")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    output_path = validate_durable_output(args.out)
    report = build_real_report(args, raw_argv)
    atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[segnet-block-profile] wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
