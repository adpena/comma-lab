# SPDX-License-Identifier: MIT
"""Trace SegNet stage-0 squeeze-excite drift in PyTorch and MLX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    _mlx_array_to_numpy,
    mlx_sigmoid,
    mlx_silu,
    nchw_to_nhwc,
    nhwc_to_nchw,
    temporary_mlx_device,
    torch_segnet_to_mlx,
)
from tac.local_acceleration.mlx_scorer_response import (
    GPU_RESEARCH_SIGNAL_BLOCKER,
    _load_upstream_distortion_net,
    load_scorer_input_cache,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (
    _array_delta_summary,
    _jsonable,
)
from tac.local_acceleration.mlx_segnet_cross_input_probe import (
    _as_mlx_nhwc,
    _torch_stage0_block0_depthwise_values,
)

SCHEMA_VERSION = "mlx_segnet_stage0_se_trace.v1"


def build_mlx_segnet_stage0_se_trace_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
    cliff_threshold: float = 1.0e-5,
) -> dict[str, Any]:
    """Trace stage-0 block-0 SE internals on native and torch-forced inputs."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU SE traces are local "
            "research-signal implementation checks only; pass "
            "allow_gpu_research_signal=True after recording that rationale"
        )
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if float(cliff_threshold) < 0.0:
        raise ValueError(f"cliff_threshold must be >= 0, got {cliff_threshold}")

    cache = load_scorer_input_cache(cache_dir)
    total_pair_count = int(cache.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pair_count:
        raise ValueError(f"start_pair {start} is outside cache pair count {total_pair_count}")
    stop = min(total_pair_count, start + int(max_pairs))
    seg = np.asarray(cache.segnet_last_rgb[start:stop], dtype=np.float32)

    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    torch_values = _torch_stage0_block0_depthwise_values(dist.segnet, seg)
    torch_trace = _torch_stage0_se_trace(dist.segnet, torch_values)
    with temporary_mlx_device(device_type):
        mlx_segnet = torch_segnet_to_mlx(dist.segnet)
        native_trace = _mlx_stage0_se_native_trace(mlx_segnet, seg)
        forced_trace = _mlx_stage0_se_forced_trace(mlx_segnet, torch_values)

    rows = _se_trace_rows(
        torch_trace=torch_trace,
        native_trace=native_trace,
        forced_trace=forced_trace,
        cliff_threshold=float(cliff_threshold),
    )
    dominant_forced = max(
        rows,
        key=lambda row: _max_or_neg_inf(row["forced_torch_input_mlx_vs_torch"]),
    )
    output_row = next(row for row in rows if row["name"] == "se.output_multiply")
    output_forced = _as_float(
        output_row["forced_torch_input_mlx_vs_torch"].get("max_abs_delta")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_segnet_se_trace",
        "verdict": (
            "SE_FORCED_OUTPUT_EXCEEDS_CLIFF"
            if output_forced is None or output_forced > float(cliff_threshold)
            else "SE_FORCED_OUTPUT_WITHIN_CLIFF"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "device_type": device_type,
        "gpu_research_signal_allowed": bool(allow_gpu_research_signal),
        "cache_dir": str(cache_dir),
        "start_pair": start,
        "max_pairs": int(max_pairs),
        "pair_window": [start, stop],
        "n_samples": stop - start,
        "total_pair_count": total_pair_count,
        "cliff_threshold": float(cliff_threshold),
        "dominant_forced_row": {
            "name": dominant_forced["name"],
            "max_abs_delta": _as_float(
                dominant_forced["forced_torch_input_mlx_vs_torch"].get("max_abs_delta")
            ),
            "p99_abs_delta": _as_float(
                dominant_forced["forced_torch_input_mlx_vs_torch"].get("p99_abs_delta")
            ),
        },
        "rows": rows,
        "authority_status": (
            "MLX SegNet SE traces are diagnostic local implementation evidence only; "
            "exact auth eval remains required for score claims and promotion."
        ),
    }


def write_mlx_segnet_stage0_se_trace_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write an SE trace manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_stage0_se_trace(
    torch_segnet: Any,
    torch_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    import torch

    torch_segnet.eval()
    se = torch_segnet.encoder.model.blocks[0][0].se
    x = torch.from_numpy(
        np.array(torch_values["encoder.stage_0.block_0.bn1"], dtype=np.float32, copy=True)
    )
    with torch.inference_mode():
        pooled = x.mean((2, 3), keepdim=True)
        conv_reduce = se.conv_reduce(pooled)
        conv_reduce_raw = conv_reduce.clone()
        act1 = se.act1(conv_reduce)
        conv_expand = se.conv_expand(act1)
        gate = se.gate(conv_expand)
        output = x * gate
    return {
        "se.input": _torch_to_numpy(x),
        "se.pool": _torch_to_numpy(pooled),
        "se.conv_reduce": _torch_to_numpy(conv_reduce_raw),
        "se.act1_silu": _torch_to_numpy(act1),
        "se.conv_expand": _torch_to_numpy(conv_expand),
        "se.gate_sigmoid": _torch_to_numpy(gate),
        "se.output_multiply": _torch_to_numpy(output),
    }


def _mlx_stage0_se_native_trace(mlx_segnet: Any, x_nchw: np.ndarray) -> dict[str, np.ndarray]:
    import mlx.core as mx

    encoder_model = mlx_segnet.encoder.model
    block = encoder_model.stages[0].blocks[0]
    x = mx.array(nchw_to_nhwc(x_nchw))
    stem = encoder_model.stem(x)
    conv_dw = block.conv_dw(stem)
    bn1 = block.bn1(conv_dw)
    return _mlx_stage0_se_trace_from_input(block.se, bn1)


def _mlx_stage0_se_forced_trace(
    mlx_segnet: Any,
    torch_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    block = mlx_segnet.encoder.model.stages[0].blocks[0]
    return _mlx_stage0_se_trace_from_input(
        block.se,
        _as_mlx_nhwc(torch_values["encoder.stage_0.block_0.bn1"]),
    )


def _mlx_stage0_se_trace_from_input(mlx_se: Any, x_nhwc: Any) -> dict[str, np.ndarray]:
    import mlx.core as mx

    pooled = mx.mean(x_nhwc, axis=(1, 2), keepdims=True)
    conv_reduce = mlx_se.conv_reduce(pooled)
    act1 = mlx_silu(conv_reduce)
    conv_expand = mlx_se.conv_expand(act1)
    gate = mlx_sigmoid(conv_expand)
    output = x_nhwc * gate
    return {
        "se.input": _mlx_nchw(x_nhwc),
        "se.pool": _mlx_nchw(pooled),
        "se.conv_reduce": _mlx_nchw(conv_reduce),
        "se.act1_silu": _mlx_nchw(act1),
        "se.conv_expand": _mlx_nchw(conv_expand),
        "se.gate_sigmoid": _mlx_nchw(gate),
        "se.output_multiply": _mlx_nchw(output),
    }


def _se_trace_rows(
    *,
    torch_trace: dict[str, np.ndarray],
    native_trace: dict[str, np.ndarray],
    forced_trace: dict[str, np.ndarray],
    cliff_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(torch_trace):
        native_summary = _array_delta_summary(torch_trace[name], native_trace[name])
        forced_summary = _array_delta_summary(torch_trace[name], forced_trace[name])
        native_max = _as_float(native_summary.get("max_abs_delta"))
        forced_max = _as_float(forced_summary.get("max_abs_delta"))
        rows.append(
            {
                "index": index,
                "name": name,
                "native_mlx_vs_torch": native_summary,
                "forced_torch_input_mlx_vs_torch": forced_summary,
                "native_exceeds_cliff_threshold": (
                    native_max is None or native_max > cliff_threshold
                ),
                "forced_exceeds_cliff_threshold": (
                    forced_max is None or forced_max > cliff_threshold
                ),
            }
        )
    return rows


def _mlx_nchw(value_nhwc: Any) -> np.ndarray:
    return nhwc_to_nchw(_mlx_array_to_numpy(value_nhwc))


def _torch_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=True)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _max_or_neg_inf(summary: dict[str, Any]) -> float:
    value = _as_float(summary.get("max_abs_delta"))
    return float("-inf") if value is None else value


__all__ = [
    "SCHEMA_VERSION",
    "build_mlx_segnet_stage0_se_trace_manifest",
    "write_mlx_segnet_stage0_se_trace_manifest",
]
