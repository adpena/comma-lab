#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure sparse/low-rank structure of frozen-SegNet input costates.

This is a local, read-only replay probe.  It regenerates the exact batch-size-1
costates from the immutable task-455 n600 replay, validates every regenerated
field against task-455's retained content hash, and writes only reduced
statistics.  Full-grid heldout costates are success-only scratch for the exact
120-state Gram spectrum; they are certified rebuildable and removed after the
receipt is sealed.  The probe does not train, mutate a source run, evaluate an
archive, or claim a contest score.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA = "p0_sparse_adjoint_costate_vjp.v1"
LANE_ID = "p0_sparse_adjoint"
AXIS = "[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]"
BOUNDARY_AREA = 0.04736597696940104
CONCENTRATION_AREAS = (0.001, 0.005, 0.01, 0.02, BOUNDARY_AREA, 0.05, 0.10, 0.20, 0.50)
MASK_AREAS = (0.01, 0.02, BOUNDARY_AREA, 0.10, 0.20, 0.50, 0.80)
MASS_TARGETS = (0.50, 0.80, 0.90, 0.95, 0.97, 0.99)
PROFILE_MODES = {
    "full",
    f"top_output@{BOUNDARY_AREA:.12f}",
    f"source_margin@{BOUNDARY_AREA:.12f}",
}
DEFAULT_OUTPUT = REPO / "experiments/results/p0_sparse_adjoint_costate_vjp_20260713"
STORAGE_PREFLIGHT = REPO / ".omx/research/p0_sparse_adjoint_storage_preflight_20260713.json"
ROUND2_RECEIPT = (
    REPO
    / "experiments/results/frozen_replay_convex_head_95kill_n600_20260713"
    / "measurement_receipt.json"
)
ROUND2_RECEIPT_SHA256 = "067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1"
TILE_HALO_RECEIPT = (
    REPO
    / "experiments/results/cheapen_real95_tilehalo_fp16_20260713"
    / "tile_halo_receipt.json"
)


class ProbeError(RuntimeError):
    """The measurement or custody contract failed closed."""


def _utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_module(name: str, relative: str) -> Any:
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot import {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True
    )
    return result.stdout.splitlines()


def _fraction_key(value: float) -> str:
    return f"{value:.12f}"


def _k_for_fraction(size: int, fraction: float) -> int:
    return min(size, max(1, round(size * fraction)))


def _rank_mask(values: Any, fraction: float, *, largest: bool) -> Any:
    import torch

    flat = values.reshape(-1)
    k = _k_for_fraction(flat.numel(), fraction)
    indices = torch.topk(flat if largest else -flat, k=k, sorted=False).indices
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    mask[indices] = True
    return mask.reshape(values.shape)


def _curve_from_nonnegative(values: np.ndarray, areas: tuple[float, ...]) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all() or np.any(flat < 0):
        raise ProbeError("concentration curve received invalid mass")
    total = float(flat.sum(dtype=np.float64))
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    captured: dict[str, float | None] = {}
    for area in areas:
        k = _k_for_fraction(flat.size, area)
        captured[_fraction_key(area)] = float(cumulative[k - 1] / total) if total else None
    area_for_mass: dict[str, float | None] = {}
    for target in MASS_TARGETS:
        if total:
            index = int(np.searchsorted(cumulative, target * total, side="left"))
            area_for_mass[_fraction_key(target)] = float((index + 1) / flat.size)
        else:
            area_for_mass[_fraction_key(target)] = None
    square_sum = float(np.square(flat, dtype=np.float64).sum(dtype=np.float64))
    participation = total * total / square_sum if square_sum else 0.0
    return {
        "total": total,
        "captured_fraction_at_top_area": captured,
        "area_fraction_for_captured_mass": area_for_mass,
        "exact_zero_fraction": float(np.count_nonzero(flat == 0.0) / flat.size),
        "participation_ratio_fraction": float(participation / flat.size),
    }


def _mask_capture(values: np.ndarray, ranking: np.ndarray, areas: tuple[float, ...]) -> dict[str, float | None]:
    mass = np.asarray(values, dtype=np.float64).reshape(-1)
    rank = np.asarray(ranking, dtype=np.float64).reshape(-1)
    if mass.shape != rank.shape or not np.isfinite(mass).all() or not np.isfinite(rank).all():
        raise ProbeError("mask-capture geometry or finiteness failed")
    total = float(mass.sum(dtype=np.float64))
    order = np.argsort(rank, kind="stable")
    result: dict[str, float | None] = {}
    for area in areas:
        k = _k_for_fraction(mass.size, area)
        result[_fraction_key(area)] = float(mass[order[:k]].sum(dtype=np.float64) / total) if total else None
    return result


def _vector_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    import torch

    ref = reference.detach().reshape(-1).to(dtype=torch.float64)
    cand = candidate.detach().reshape(-1).to(dtype=torch.float64)
    dot = float(torch.dot(ref, cand).item())
    ref2 = float(torch.dot(ref, ref).item())
    cand2 = float(torch.dot(cand, cand).item())
    delta = ref - cand
    delta2 = float(torch.dot(delta, delta).item())
    cosine = dot / math.sqrt(ref2 * cand2) if ref2 and cand2 else 0.0
    relative_l2 = math.sqrt(delta2 / ref2) if ref2 else 0.0
    return {
        "dot": dot,
        "reference_square_norm": ref2,
        "candidate_square_norm": cand2,
        "delta_square_norm": delta2,
        "cosine": cosine,
        "relative_l2_error": relative_l2,
        "candidate_to_reference_norm": math.sqrt(cand2 / ref2) if ref2 else 0.0,
    }


def _module_family(name: str) -> str:
    if "encoder" in name:
        return "encoder"
    if "decoder" in name:
        return "decoder"
    if "segmentation_head" in name:
        return "segmentation_head"
    return "other"


class BackwardSupportProfiler:
    """Count numerical cotangent support and nominal conv backward-data FLOPs."""

    def __init__(self, model: Any) -> None:
        import torch

        self.mode: str | None = None
        self.rows: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._forward_hook(name, module)))

    def reset(self) -> None:
        self.mode = None
        self.rows = defaultdict(dict)

    def _forward_hook(self, name: str, module: Any) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not hasattr(output, "register_hook") or output.ndim != 4 or not output.requires_grad:
                return
            n, c_out, height, width = map(int, output.shape)
            kernel_h, kernel_w = module.kernel_size
            dense_flops = float(
                2
                * n
                * height
                * width
                * c_out
                * (int(module.in_channels) // int(module.groups))
                * kernel_h
                * kernel_w
            )

            def gradient_hook(gradient: Any) -> None:
                mode = self.mode
                if mode not in PROFILE_MODES:
                    return
                detached = gradient.detach()
                spatial_active = float((detached != 0).any(dim=1).to(dtype=detached.dtype).mean().item())
                element_active = float((detached != 0).to(dtype=detached.dtype).mean().item())
                self.rows[mode][name] = {
                    "dense_flops": dense_flops,
                    "ideal_spatial_sparse_flops": dense_flops * spatial_active,
                    "ideal_element_sparse_flops": dense_flops * element_active,
                    "spatial_active_fraction": spatial_active,
                    "element_active_fraction": element_active,
                    "family": _module_family(name),
                }

            output.register_hook(gradient_hook)

        return hook

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for mode, rows in self.rows.items():
            families: dict[str, dict[str, float]] = defaultdict(
                lambda: {
                    "dense_flops": 0.0,
                    "ideal_spatial_sparse_flops": 0.0,
                    "ideal_element_sparse_flops": 0.0,
                    "conv_calls": 0.0,
                }
            )
            for row in rows.values():
                family = families[str(row["family"])]
                for key in ("dense_flops", "ideal_spatial_sparse_flops", "ideal_element_sparse_flops"):
                    family[key] += float(row[key])
                family["conv_calls"] += 1.0
            total = {
                key: sum(float(value[key]) for value in families.values())
                for key in ("dense_flops", "ideal_spatial_sparse_flops", "ideal_element_sparse_flops")
            }
            total["conv_calls"] = sum(float(value["conv_calls"]) for value in families.values())
            for bucket in [*families.values(), total]:
                dense = bucket["dense_flops"]
                bucket["ideal_spatial_sparse_speedup_upper_bound_x"] = (
                    dense / bucket["ideal_spatial_sparse_flops"]
                    if bucket["ideal_spatial_sparse_flops"]
                    else None
                )
                bucket["ideal_element_sparse_speedup_upper_bound_x"] = (
                    dense / bucket["ideal_element_sparse_flops"]
                    if bucket["ideal_element_sparse_flops"]
                    else None
                )
            summary[mode] = {"total": total, "families": dict(families)}
        return summary

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _output_gradient(logits: Any, labels: Any) -> Any:
    import torch

    with torch.no_grad():
        gradient = logits.detach().softmax(dim=1)
        gradient.scatter_add_(
            1,
            labels[:, None],
            -torch.ones_like(labels[:, None], dtype=gradient.dtype),
        )
        gradient /= labels.numel()
    return gradient


def _masked_vjp_curves(
    *, logits: Any, frame: Any, output_gradient: Any, margin: np.ndarray, exact_costate: Any, profiler: Any
) -> tuple[dict[str, Any], float]:
    import torch

    spatial_energy = torch.square(output_gradient).sum(dim=1)[0]
    margin_t = torch.as_tensor(np.asarray(margin), dtype=spatial_energy.dtype)
    jobs: list[tuple[str, float, Any]] = []
    for scheme in ("top_output", "source_margin"):
        for fraction in MASK_AREAS:
            mask = _rank_mask(
                spatial_energy if scheme == "top_output" else margin_t,
                fraction,
                largest=(scheme == "top_output"),
            )
            jobs.append((scheme, fraction, mask))
    results: dict[str, dict[str, Any]] = {"top_output": {}, "source_margin": {}}
    total_seconds = 0.0
    output_l1 = output_gradient.abs().sum(dim=1)[0]
    output_l2 = spatial_energy
    for index, (scheme, fraction, mask) in enumerate(jobs):
        mode = f"{scheme}@{fraction:.12f}"
        profiler.mode = mode if mode in PROFILE_MODES else None
        started = time.perf_counter()
        candidate = torch.autograd.grad(
            logits,
            frame,
            grad_outputs=output_gradient * mask[None, None],
            retain_graph=index + 1 < len(jobs),
        )[0].detach()
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        metrics = _vector_metrics(exact_costate, candidate)
        selected_l1 = float(output_l1[mask].sum(dtype=torch.float64).item())
        selected_l2 = float(output_l2[mask].sum(dtype=torch.float64).item())
        metrics.update(
            {
                "requested_area_fraction": fraction,
                "realized_area_fraction": float(mask.to(dtype=torch.float64).mean().item()),
                "output_l1_mass_retained": selected_l1
                / float(output_l1.sum(dtype=torch.float64).item()),
                "output_l2_energy_retained": selected_l2
                / float(output_l2.sum(dtype=torch.float64).item()),
                "vjp_seconds": elapsed,
            }
        )
        results[scheme][_fraction_key(fraction)] = metrics
    profiler.mode = None
    return results, total_seconds


def _measure_pair(
    *, assignment: Any, renderer: Any, segnet: Any, labels: Any, margins: Any, round2: Any, profiler: Any
) -> tuple[dict[str, Any], np.ndarray | None]:
    import torch
    import torch.nn.functional as functional

    pair = int(assignment.pair_index)
    label = np.array(labels[pair], dtype=np.int64, copy=True)
    margin = np.array(margins[pair], dtype=np.float32, copy=True)
    frame = round2._render_state_nchw(renderer, pair).detach().requires_grad_(True)
    profiler.reset()
    started = time.perf_counter()
    logits = segnet(frame)
    forward_seconds = time.perf_counter() - started
    label_t = torch.as_tensor(label[None], dtype=torch.long)
    loss = functional.cross_entropy(logits, label_t)
    output_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0].detach()
    keep_graph = assignment.split == "heldout"
    profiler.mode = "full" if keep_graph else None
    started = time.perf_counter()
    exact_costate = torch.autograd.grad(loss, frame, retain_graph=keep_graph)[0].detach()
    backward_seconds = time.perf_counter() - started
    profiler.mode = None
    costate_np = np.ascontiguousarray(exact_costate.cpu().numpy(), dtype=np.float32)
    exact_sha = round2.array_sha256(costate_np)
    source_dir = ROUND2_RECEIPT.parent
    if assignment.split == "heldout":
        source = json.loads((source_dir / f"heldout/pair_{pair:04d}.json").read_text())
        expected_sha = source["exact_costate_sha256"]
        target_kind = "full_grid_costate"
    else:
        source_npz = source_dir / f"train_cache/pair_{pair:04d}.npz"
        with np.load(source_npz, allow_pickle=False) as cache:
            expected_sha = str(cache["target_sha256"].item())
        sampled = round2.sampled_costate_rows(costate_np, stride=8)
        exact_sha = round2.array_sha256(sampled)
        target_kind = "stride8_costate_rows"
    if exact_sha != expected_sha:
        raise ProbeError(f"task-455 costate hash mismatch for pair {pair}: {exact_sha} != {expected_sha}")

    input_l1 = np.abs(costate_np[0]).sum(axis=0, dtype=np.float64)
    input_l2 = np.square(costate_np[0], dtype=np.float64).sum(axis=0, dtype=np.float64)
    output_np = output_gradient.cpu().numpy()[0]
    output_l1 = np.abs(output_np).sum(axis=0, dtype=np.float64)
    output_l2 = np.square(output_np, dtype=np.float64).sum(axis=0, dtype=np.float64)
    record: dict[str, Any] = {
        "schema": "p0_sparse_adjoint_pair.v1",
        "completed_at_utc": _utc(),
        "assignment": assignment.to_dict(),
        "axis": AXIS,
        "task455_hash_validation": {
            "status": "MEASURED_MATCH",
            "target_kind": target_kind,
            "expected_sha256": expected_sha,
            "regenerated_sha256": exact_sha,
        },
        "frame_sha256": round2.array_sha256(frame.detach().cpu().numpy()),
        "label_sha256": round2.array_sha256(label),
        "margin_sha256": round2.array_sha256(margin),
        "teacher": {
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "ce": float(loss.detach().item()),
            "dseg": float((logits.detach().argmax(1).cpu().numpy()[0] != label).mean()),
        },
        "output_logit_gradient": {
            "l1_mass": _curve_from_nonnegative(output_l1, CONCENTRATION_AREAS),
            "l2_energy": _curve_from_nonnegative(output_l2, CONCENTRATION_AREAS),
            "spatial_exact_zero_fraction": float(np.all(output_np == 0.0, axis=0).mean()),
            "element_exact_zero_fraction": float((output_np == 0.0).mean()),
            "source_margin_low_area_capture_l1": _mask_capture(
                output_l1, margin, CONCENTRATION_AREAS
            ),
            "source_margin_low_area_capture_l2": _mask_capture(
                output_l2, margin, CONCENTRATION_AREAS
            ),
        },
        "input_costate": {
            "l1_mass": _curve_from_nonnegative(input_l1, CONCENTRATION_AREAS),
            "l2_energy": _curve_from_nonnegative(input_l2, CONCENTRATION_AREAS),
            "spatial_exact_zero_fraction": float(np.all(costate_np[0] == 0.0, axis=0).mean()),
            "element_exact_zero_fraction": float((costate_np[0] == 0.0).mean()),
            "source_margin_low_area_capture_l1": _mask_capture(
                input_l1, margin, CONCENTRATION_AREAS
            ),
            "source_margin_low_area_capture_l2": _mask_capture(
                input_l2, margin, CONCENTRATION_AREAS
            ),
        },
    }
    scratch: np.ndarray | None = None
    if keep_graph:
        curves, masked_seconds = _masked_vjp_curves(
            logits=logits,
            frame=frame,
            output_gradient=output_gradient,
            margin=margin,
            exact_costate=exact_costate,
            profiler=profiler,
        )
        record["masked_vjp"] = curves
        record["teacher"]["masked_vjp_curve_seconds"] = masked_seconds
        record["conv_backward_support"] = profiler.summary()
        scratch = costate_np
    return record, scratch


def _quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _aggregate_capture(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    curves = []
    totals = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        curves.append(value["captured_fraction_at_top_area"])
        totals.append(float(value["total"]))
    result: dict[str, Any] = {}
    for area in CONCENTRATION_AREAS:
        key = _fraction_key(area)
        values = [float(curve[key]) for curve in curves]
        weighted = sum(value * total for value, total in zip(values, totals, strict=True)) / sum(totals)
        result[key] = {"global_mass_weighted": weighted, "per_state": _quantiles(values)}
    return result


def _aggregate_margin_capture(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for area in CONCENTRATION_AREAS:
        key = _fraction_key(area)
        values = []
        weights = []
        for record in records:
            value: Any = record
            for segment in path:
                value = value[segment]
            values.append(float(value[key]))
            mass_path: Any = record[path[0]]
            mass_key = "l1_mass" if path[-1].endswith("l1") else "l2_energy"
            weights.append(float(mass_path[mass_key]["total"]))
        result[key] = {
            "global_mass_weighted": sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights),
            "per_state": _quantiles(values),
        }
    return result


def _aggregate_masked(records: list[dict[str, Any]]) -> dict[str, Any]:
    heldout = [row for row in records if row["assignment"]["split"] == "heldout"]
    result: dict[str, Any] = {}
    for scheme in ("top_output", "source_margin"):
        result[scheme] = {}
        for area in MASK_AREAS:
            key = _fraction_key(area)
            rows = [row["masked_vjp"][scheme][key] for row in heldout]
            dot = sum(float(row["dot"]) for row in rows)
            ref2 = sum(float(row["reference_square_norm"]) for row in rows)
            cand2 = sum(float(row["candidate_square_norm"]) for row in rows)
            delta2 = sum(float(row["delta_square_norm"]) for row in rows)
            result[scheme][key] = {
                "state_count": len(rows),
                "global_cosine": dot / math.sqrt(ref2 * cand2),
                "global_relative_l2_error": math.sqrt(delta2 / ref2),
                "global_candidate_to_reference_norm": math.sqrt(cand2 / ref2),
                "per_state_cosine": _quantiles([float(row["cosine"]) for row in rows]),
                "per_state_relative_l2_error": _quantiles(
                    [float(row["relative_l2_error"]) for row in rows]
                ),
                "output_l1_mass_retained": _quantiles(
                    [float(row["output_l1_mass_retained"]) for row in rows]
                ),
                "output_l2_energy_retained": _quantiles(
                    [float(row["output_l2_energy_retained"]) for row in rows]
                ),
                "vjp_seconds": _quantiles([float(row["vjp_seconds"]) for row in rows]),
            }
    return result


def _aggregate_profiles(records: list[dict[str, Any]]) -> dict[str, Any]:
    heldout = [row for row in records if row["assignment"]["split"] == "heldout"]
    result: dict[str, Any] = {}
    for mode in sorted(PROFILE_MODES):
        modes = [row["conv_backward_support"][mode] for row in heldout]
        families = sorted({family for row in modes for family in row["families"]})
        buckets: dict[str, Any] = {}
        for family in ["total", *families]:
            rows = [row["total"] if family == "total" else row["families"][family] for row in modes]
            dense = sum(float(row["dense_flops"]) for row in rows)
            spatial = sum(float(row["ideal_spatial_sparse_flops"]) for row in rows)
            element = sum(float(row["ideal_element_sparse_flops"]) for row in rows)
            buckets[family] = {
                "state_count": len(rows),
                "dense_nominal_conv_backward_flops": dense,
                "ideal_spatial_sparse_conv_backward_flops": spatial,
                "ideal_element_sparse_conv_backward_flops": element,
                "ideal_spatial_sparse_speedup_upper_bound_x": dense / spatial if spatial else None,
                "ideal_element_sparse_speedup_upper_bound_x": dense / element if element else None,
                "dense_kernel_realized_speedup_x": 1.0,
            }
        result[mode] = buckets
    return result


def _spectrum_from_gram(gram: np.ndarray) -> dict[str, Any]:
    symmetric = (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)[::-1]
    tolerance = max(float(eigenvalues[0]), 1.0) * np.finfo(np.float64).eps * len(eigenvalues) * 32
    eigenvalues[np.abs(eigenvalues) <= tolerance] = 0.0
    if np.any(eigenvalues < -tolerance):
        raise ProbeError(f"Gram spectrum has material negative eigenvalue {eigenvalues.min()}")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    fractions = eigenvalues / total if total else eigenvalues
    cumulative = np.cumsum(fractions)

    def rank_for(target: float) -> int:
        return int(np.searchsorted(cumulative, target, side="left") + 1)

    positive = fractions[fractions > 0]
    entropy_rank = float(math.exp(-float(np.sum(positive * np.log(positive))))) if positive.size else 0.0
    rank_curve = {}
    for rank in (1, 2, 4, 8, 16, 32, 64, 96, 110, 119, 120):
        use = min(rank, len(eigenvalues))
        captured = float(cumulative[use - 1])
        rank_curve[str(use)] = {
            "energy_captured": captured,
            "relative_frobenius_reconstruction_error": math.sqrt(max(0.0, 1.0 - captured)),
        }
    return {
        "state_rank_ceiling": len(eigenvalues),
        "eigenvalues_descending": eigenvalues.tolist(),
        "singular_values_descending": np.sqrt(eigenvalues).tolist(),
        "rank_for_energy": {
            "50pct": rank_for(0.50),
            "80pct": rank_for(0.80),
            "90pct": rank_for(0.90),
            "95pct": rank_for(0.95),
            "97pct": rank_for(0.97),
            "99pct": rank_for(0.99),
        },
        "stable_rank": total / float(eigenvalues[0]) if eigenvalues[0] else 0.0,
        "entropy_effective_rank": entropy_rank,
        "numerical_rank_by_relative_eigenvalue": {
            f"{threshold:.0e}": int(np.count_nonzero(eigenvalues > eigenvalues[0] * threshold))
            for threshold in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
        },
        "rank_error_curve": rank_curve,
    }


def _compute_spectrum(scratch_paths: list[Path]) -> dict[str, Any]:
    started = time.perf_counter()
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in scratch_paths]
    if any(array.shape != (1, 3, 384, 512) or array.dtype != np.float32 for array in arrays):
        raise ProbeError("heldout scratch costate geometry drift")
    matrix = np.empty((len(arrays), 3 * 384 * 512), dtype=np.float32)
    for index, array in enumerate(arrays):
        matrix[index] = np.asarray(array).reshape(-1)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    gram32 = matrix @ matrix.T
    gram_seconds = time.perf_counter() - started
    gram = np.asarray(gram32, dtype=np.float64)
    mean_row = gram.mean(axis=1, keepdims=True)
    centered = gram - mean_row - mean_row.T + float(gram.mean())
    norms = np.sqrt(np.maximum(np.diag(gram), np.finfo(np.float64).tiny))
    normalized = gram / np.outer(norms, norms)
    return {
        "schema": "p0_sparse_adjoint_exact_heldout_spectrum.v1",
        "completed_at_utc": _utc(),
        "state_count": len(arrays),
        "ambient_dimension": int(matrix.shape[1]),
        "compared_elements": int(matrix.size),
        "matrix_dtype": "float32",
        "gram_accumulation_dtype": "float32",
        "eigendecomposition_dtype": "float64",
        "load_seconds": load_seconds,
        "gram_seconds": gram_seconds,
        "raw": _spectrum_from_gram(gram),
        "centered": _spectrum_from_gram(centered),
        "row_l2_normalized": _spectrum_from_gram(normalized),
    }


def _scratch_tree(scratch_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(scratch_dir.glob("pair_*.npy")):
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    canonical = "".join(
        f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n" for row in rows
    ).encode()
    return {
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": rows,
    }


def _certify_and_remove_scratch(output_dir: Path, spectrum_path: Path) -> dict[str, Any]:
    scratch_dir = output_dir / "scratch_costates"
    cleanup_path = output_dir / "cleanup_manifest.json"
    if cleanup_path.is_file():
        cleanup = json.loads(cleanup_path.read_text())
        if cleanup.get("status") == "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED":
            if scratch_dir.exists():
                raise ProbeError("scratch reappeared after certified removal")
            return cleanup
        if cleanup.get("status") == "CERTIFIED_REBUILDABLE_SCRATCH_PENDING_REMOVAL":
            if scratch_dir.exists():
                observed = _scratch_tree(scratch_dir)
                expected = cleanup["scratch_tree"]
                if {
                    key: observed[key] for key in ("file_count", "bytes", "tree_sha256")
                } != {key: expected[key] for key in ("file_count", "bytes", "tree_sha256")}:
                    raise ProbeError("scratch tree changed after cleanup certification")
                shutil.rmtree(scratch_dir)
            cleanup["status"] = "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED"
            cleanup["removed_at_utc"] = _utc()
            _atomic_json(cleanup_path, cleanup)
            return cleanup
        raise ProbeError("unknown cleanup manifest state")
    tree = _scratch_tree(scratch_dir)
    if tree["file_count"] != 120:
        raise ProbeError(f"expected 120 heldout scratch costates, got {tree['file_count']}")
    cleanup = {
        "schema": "p0_sparse_adjoint_cleanup.v1",
        "status": "CERTIFIED_REBUILDABLE_SCRATCH_PENDING_REMOVAL",
        "certified_at_utc": _utc(),
        "original_path": str(scratch_dir.relative_to(REPO)),
        "scratch_tree": tree,
        "rebuild_command": (
            ".venv/bin/python tools/probe_sparse_adjoint_costate_vjp.py "
            "--output-dir experiments/results/p0_sparse_adjoint_costate_vjp_20260713 --resume"
        ),
        "source_receipt": str(ROUND2_RECEIPT.relative_to(REPO)),
        "source_receipt_sha256": ROUND2_RECEIPT_SHA256,
        "derived_spectrum": str(spectrum_path.relative_to(REPO)),
        "derived_spectrum_sha256": _sha256(spectrum_path),
        "cold_store_destination": None,
        "reason": "success-only exact-spectrum scratch; deterministic batch-1 replay and task-455 hashes certify rebuildability",
        "false_authority": {"score_claim": False, "promotion_eligible": False},
    }
    _atomic_json(cleanup_path, cleanup)
    shutil.rmtree(scratch_dir)
    cleanup["status"] = "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED"
    cleanup["removed_at_utc"] = _utc()
    _atomic_json(cleanup_path, cleanup)
    return cleanup


def _validate_preflight(output_dir: Path) -> dict[str, Any]:
    if not STORAGE_PREFLIGHT.is_file():
        raise ProbeError("missing storage preflight")
    plan = json.loads(STORAGE_PREFLIGHT.read_text())
    if plan.get("blockers"):
        raise ProbeError(f"storage preflight blockers: {plan['blockers']}")
    if Path(plan.get("selected_workload_root", "")).resolve() != output_dir.resolve():
        raise ProbeError("storage preflight selected another workload root")
    return {
        "path": str(STORAGE_PREFLIGHT.relative_to(REPO)),
        "bytes": STORAGE_PREFLIGHT.stat().st_size,
        "sha256": _sha256(STORAGE_PREFLIGHT),
        "selected_tier": plan["selected_tier"],
        "requested_bytes": plan["requested_bytes"],
        "explicit_local_opt_in": plan["operator_storage_policy"]["local_disk_enabled"],
    }


def _source_custody(round2: Any) -> dict[str, Any]:
    if _sha256(ROUND2_RECEIPT) != ROUND2_RECEIPT_SHA256:
        raise ProbeError("task-455 receipt custody drift")
    inputs = round2._verify_input_custody()
    paths = (
        Path(__file__).resolve(),
        REPO / "tools/probe_frozen_replay_convex_head.py",
        REPO / "tools/probe_yopo_first_layer_costate.py",
        REPO / "upstream/modules.py",
        TILE_HALO_RECEIPT,
    )
    return {
        "task455_receipt": {
            "path": str(ROUND2_RECEIPT.relative_to(REPO)),
            "bytes": ROUND2_RECEIPT.stat().st_size,
            "sha256": ROUND2_RECEIPT_SHA256,
            "raw_costates_retained": False,
            "retained_authority": "per-state hashes, compact sufficient statistics, and heldout reductions",
        },
        "sealed_inputs": inputs,
        "sources": {
            str(path.relative_to(REPO)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths
        },
    }


def _aggregate_receipt(records: list[dict[str, Any]], spectrum: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_count": len(records),
        "heldout_full_grid_count": sum(row["assignment"]["split"] == "heldout" for row in records),
        "task455_hash_matches": sum(
            row["task455_hash_validation"]["status"] == "MEASURED_MATCH" for row in records
        ),
        "input_costate_top_area_l1_mass": _aggregate_capture(records, ("input_costate", "l1_mass")),
        "input_costate_top_area_l2_energy": _aggregate_capture(records, ("input_costate", "l2_energy")),
        "output_gradient_top_area_l1_mass": _aggregate_capture(
            records, ("output_logit_gradient", "l1_mass")
        ),
        "output_gradient_top_area_l2_energy": _aggregate_capture(
            records, ("output_logit_gradient", "l2_energy")
        ),
        "input_costate_source_margin_low_area_l1_mass": _aggregate_margin_capture(
            records, ("input_costate", "source_margin_low_area_capture_l1")
        ),
        "input_costate_source_margin_low_area_l2_energy": _aggregate_margin_capture(
            records, ("input_costate", "source_margin_low_area_capture_l2")
        ),
        "output_gradient_source_margin_low_area_l1_mass": _aggregate_margin_capture(
            records, ("output_logit_gradient", "source_margin_low_area_capture_l1")
        ),
        "output_gradient_source_margin_low_area_l2_energy": _aggregate_margin_capture(
            records, ("output_logit_gradient", "source_margin_low_area_capture_l2")
        ),
        "input_spatial_exact_zero_fraction": _quantiles(
            [float(row["input_costate"]["spatial_exact_zero_fraction"]) for row in records]
        ),
        "output_spatial_exact_zero_fraction": _quantiles(
            [float(row["output_logit_gradient"]["spatial_exact_zero_fraction"]) for row in records]
        ),
        "teacher_forward_seconds": _quantiles(
            [float(row["teacher"]["forward_seconds"]) for row in records]
        ),
        "teacher_backward_seconds": _quantiles(
            [float(row["teacher"]["backward_seconds"]) for row in records]
        ),
        "masked_vjp_error_curves": _aggregate_masked(records),
        "conv_backward_support_flop_bounds": _aggregate_profiles(records),
        "heldout_exact_spectrum": spectrum,
    }


def run(output_dir: Path, *, resume: bool, validate_only: bool) -> dict[str, Any]:
    import torch

    output_dir = output_dir.resolve()
    if output_dir != DEFAULT_OUTPUT.resolve():
        raise ProbeError("this sealed measurement accepts only the preflighted canonical output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = _validate_preflight(output_dir)
    round2 = _load_module("_p0_sparse_round2", "tools/probe_frozen_replay_convex_head.py")
    source = _source_custody(round2)
    if validate_only:
        return {"schema": SCHEMA, "status": "VALIDATED_ONLY", "storage": storage, "source": source}

    run_contract_path = output_dir / "run_contract.json"
    contract = {
        "schema": "p0_sparse_adjoint_run_contract.v1",
        "source_custody": source,
        "storage_preflight": storage,
        "n_pairs": 600,
        "heldout_full_grid_states": 120,
        "teacher_batch_size": 1,
        "seed": 455,
        "concentration_area_fractions": list(CONCENTRATION_AREAS),
        "masked_vjp_area_fractions": list(MASK_AREAS),
        "axis": AXIS,
        "source_runs_read_only": True,
        "training_enabled": False,
    }
    if run_contract_path.is_file():
        if not resume:
            raise ProbeError("an incomplete run exists; pass --resume")
        if json.loads(run_contract_path.read_text()) != contract:
            raise ProbeError("resume run-contract drift")
    else:
        if resume:
            raise ProbeError("--resume requested without a run contract")
        _atomic_json(run_contract_path, contract)

    receipt_path = output_dir / "measurement_receipt.json"
    if receipt_path.is_file():
        if not resume:
            raise ProbeError("completed output exists; pass --resume to validate/read it")
        return json.loads(receipt_path.read_text())
    lock_path = output_dir / ".single_writer.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ProbeError("another sparse-adjoint probe owns the output directory") from exc

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = round2.FrozenReplayConvexHeadPolicy(teacher_batch_size=1)
    assignments = round2.deterministic_replay_assignments(
        n_pairs=policy.n_pairs,
        checkpoint_names=tuple(row[0] for row in round2.CHECKPOINTS),
        holdout_period=policy.holdout_period,
        seed=policy.seed,
    )
    labels = round2._stored_npy_memmap(round2.GT_CACHE, "lstars.npy")
    margins = round2._stored_npy_memmap(round2.GT_CACHE, "margins.npy")
    yopo = round2._load_tool_module("_p0_sparse_yopo", "tools/probe_yopo_first_layer_costate.py")
    segnet = round2._load_cpu_segnet()
    profiler = BackwardSupportProfiler(segnet)
    records_dir = output_dir / "pairs"
    scratch_dir = output_dir / "scratch_costates"
    spectrum_path = output_dir / "stage_spectrum_complete.json"
    progress_path = output_dir / "progress.jsonl"
    started_at = _utc()
    _append_jsonl(
        progress_path,
        {
            "event": "probe_started",
            "timestamp_utc": started_at,
            "resume": resume,
            "pid": os.getpid(),
        },
    )
    try:
        completed_count = 0
        for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
            renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
            if model["n_pairs"] != 600 or code.shape[0] != 1200:
                raise ProbeError("source checkpoint is not n600")
            for assignment in assignments:
                if assignment.checkpoint_index != checkpoint_index:
                    continue
                pair = int(assignment.pair_index)
                record_path = records_dir / f"pair_{pair:04d}.json"
                scratch_path = scratch_dir / f"pair_{pair:04d}.npy"
                usable_record = record_path.is_file()
                if usable_record and assignment.split == "heldout" and not spectrum_path.is_file():
                    usable_record = scratch_path.is_file()
                    if usable_record:
                        prior = json.loads(record_path.read_text())
                        custody = prior.get("spectrum_scratch") or {}
                        if (
                            scratch_path.stat().st_size != custody.get("bytes")
                            or _sha256(scratch_path) != custody.get("sha256")
                        ):
                            raise ProbeError(f"heldout spectrum scratch drift at pair {pair}")
                if usable_record:
                    completed_count += 1
                    continue
                record, scratch = _measure_pair(
                    assignment=assignment,
                    renderer=renderer,
                    segnet=segnet,
                    labels=labels,
                    margins=margins,
                    round2=round2,
                    profiler=profiler,
                )
                if scratch is not None:
                    _atomic_npy(scratch_path, scratch)
                    record["spectrum_scratch"] = {
                        "path": str(scratch_path.relative_to(REPO)),
                        "bytes": scratch_path.stat().st_size,
                        "sha256": _sha256(scratch_path),
                        "rebuildable": True,
                    }
                _atomic_json(record_path, record)
                completed_count += 1
                if completed_count % 10 == 0:
                    event = {
                        "event": "pair_checkpoint",
                        "timestamp_utc": _utc(),
                        "completed_pairs": completed_count,
                        "last_pair": pair,
                    }
                    _append_jsonl(progress_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)

        profiler.close()
        records = [
            json.loads((records_dir / f"pair_{pair:04d}.json").read_text()) for pair in range(600)
        ]
        if any(row["task455_hash_validation"]["status"] != "MEASURED_MATCH" for row in records):
            raise ProbeError("not all 600 regenerated costates match task-455 custody")
        manifest = {
            "schema": "p0_sparse_adjoint_pair_stage.v1",
            "completed_at_utc": _utc(),
            "record_count": len(records),
            "heldout_count": sum(row["assignment"]["split"] == "heldout" for row in records),
            "records": {
                f"{pair:04d}": {
                    "path": str((records_dir / f"pair_{pair:04d}.json").relative_to(output_dir)),
                    "bytes": (records_dir / f"pair_{pair:04d}.json").stat().st_size,
                    "sha256": _sha256(records_dir / f"pair_{pair:04d}.json"),
                }
                for pair in range(600)
            },
        }
        _atomic_json(output_dir / "stage_pairs_complete.json", manifest)
        if spectrum_path.is_file():
            spectrum = json.loads(spectrum_path.read_text())
        else:
            heldout_paths = [
                scratch_dir / f"pair_{int(row['assignment']['pair_index']):04d}.npy"
                for row in records
                if row["assignment"]["split"] == "heldout"
            ]
            spectrum = _compute_spectrum(heldout_paths)
            _atomic_json(spectrum_path, spectrum)
        cleanup = _certify_and_remove_scratch(output_dir, spectrum_path)
        aggregate = _aggregate_receipt(records, spectrum)
        tile = json.loads(TILE_HALO_RECEIPT.read_text())
        receipt = {
            "schema": SCHEMA,
            "completed_at_utc": _utc(),
            "started_at_utc": started_at,
            "lane_id": LANE_ID,
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "pointer_delta": "NONE",
            "source_runs_read_only": True,
            "training_performed": False,
            "live_run_mutated": False,
            "paid_dispatch": False,
            "storage_preflight": storage,
            "source_custody": source,
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "torch_num_threads": torch.get_num_threads(),
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "git_head": _git_head(),
                "git_status": _git_status(),
                "argv": list(sys.argv),
            },
            "measurement_contract": {
                "n600_pair_states": 600,
                "full_grid_spectrum_states": 120,
                "full_grid_spectrum_elements": 70_778_880,
                "teacher_batch_size": 1,
                "seed": 455,
                "concentration_area_fractions": list(CONCENTRATION_AREAS),
                "masked_vjp_area_fractions": list(MASK_AREAS),
                "mask_schemes": ["top_output_gradient_l2", "lowest_cached_source_margin"],
                "source_margin_mask_is_not_equated_to_333_annulus": True,
            },
            "structural_exactness": {
                "exact_halo_pixels": tile["exact_tile_contract"]["local_halo_px"],
                "global_squeeze_excite_count": tile["architecture"]["squeeze_excite_blocks"],
                "exact_source_area_fraction": tile["exact_tile_contract"][
                    "exact_source_area_fraction"
                ],
                "exact_sparse_backward_speedup_x": 1.0,
                "reason": (
                    "all finite CE logit gradients must be retained for equality, while every EfficientNet "
                    "MBConv squeeze-excite VJP has a generically dense spatial mean term; dense kernels do "
                    "not skip numerical zeros"
                ),
                "verdict_scope": (
                    "exact output-mask or spatially sparse adjoint of the frozen EfficientNet-B2 U-Net "
                    "SegNet; approximate/stale-SE and local-surrogate formulations remain separate"
                ),
            },
            "measurements": aggregate,
            "cleanup": cleanup,
            "triality": {
                "equation": "sparse_adjoint_mask_error_and_se_support_closure_v1 (new module owed after receipt hash)",
                "dag_feed": ".omx/research/p0_sparse_adjoint_costate_vjp_DAG_FEED_20260713.md (owed)",
                "dsl": "NO LIVE LEVER; research-only policy disposition in memo",
            },
        }
        _atomic_json(receipt_path, receipt)
        _append_jsonl(
            progress_path,
            {
                "event": "probe_completed",
                "timestamp_utc": _utc(),
                "receipt_sha256": _sha256(receipt_path),
            },
        )
        return receipt
    finally:
        profiler.close()
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    receipt = run(args.output_dir, resume=args.resume, validate_only=args.validate_only)
    if args.validate_only:
        print(json.dumps(receipt, sort_keys=True, indent=2))
    else:
        boundary = _fraction_key(BOUNDARY_AREA)
        measurement = receipt["measurements"]
        print(
            json.dumps(
                {
                    "receipt": str((args.output_dir / "measurement_receipt.json").resolve()),
                    "input_l1_mass_in_top_4p7pct": measurement["input_costate_top_area_l1_mass"][
                        boundary
                    ]["global_mass_weighted"],
                    "masked_top_output_relative_l2": measurement["masked_vjp_error_curves"][
                        "top_output"
                    ][boundary]["global_relative_l2_error"],
                    "raw_r95": measurement["heldout_exact_spectrum"]["raw"]["rank_for_energy"][
                        "95pct"
                    ],
                    "receipt_sha256": _sha256(args.output_dir / "measurement_receipt.json"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
