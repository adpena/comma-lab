# SPDX-License-Identifier: MIT
"""Probe stage-0 SegNet squeeze-excite pooling variants in MLX."""

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
from tac.local_acceleration.mlx_segnet_se_trace import _torch_stage0_se_trace

SCHEMA_VERSION = "mlx_segnet_stage0_se_pool_variants.v1"
POOL_VARIANTS = (
    "mean_tuple",
    "mean_h_then_w",
    "mean_w_then_h",
    "sum_tuple_div",
)


def build_mlx_segnet_stage0_se_pool_variants_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    """Compare MLX SE pooling variants against PyTorch stage-0 SE output."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU SE pool variant probes are local "
            "research-signal implementation checks only; pass "
            "allow_gpu_research_signal=True after recording that rationale"
        )
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")

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
        mlx_se = torch_segnet_to_mlx(dist.segnet).encoder.model.stages[0].blocks[0].se
        rows = [
            _pool_variant_row(
                mlx_se=mlx_se,
                x_nchw=torch_values["encoder.stage_0.block_0.bn1"],
                torch_trace=torch_trace,
                variant=variant,
            )
            for variant in POOL_VARIANTS
        ]

    best_pool = min(rows, key=lambda row: _max_or_inf(row["pool_delta"]))
    best_output = min(rows, key=lambda row: _max_or_inf(row["output_delta"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_segnet_se_pool_variants",
        "verdict": f"BEST_SE_POOL_VARIANT:{best_output['variant']}",
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
        "best_pool_variant": {
            "variant": best_pool["variant"],
            "pool_max_abs_delta": best_pool["pool_delta"]["max_abs_delta"],
            "output_max_abs_delta": best_pool["output_delta"]["max_abs_delta"],
        },
        "best_output_variant": {
            "variant": best_output["variant"],
            "pool_max_abs_delta": best_output["pool_delta"]["max_abs_delta"],
            "output_max_abs_delta": best_output["output_delta"]["max_abs_delta"],
        },
        "rows": rows,
        "authority_status": (
            "MLX SegNet SE pool variant probes are diagnostic local implementation "
            "evidence only; exact auth eval remains required for score claims and "
            "promotion."
        ),
        "device_contract": {
            "gpu_research_signal_blocker": GPU_RESEARCH_SIGNAL_BLOCKER,
            "gpu_research_signal_required": device_type == "gpu",
            "forbidden_uses": [
                "score_claim",
                "promotion",
                "rank_or_kill",
                "replacement_for_cpu_or_cuda_auth_eval",
            ],
        },
    }


def write_mlx_segnet_stage0_se_pool_variants_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write an SE pool variant manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pool_variant_row(
    *,
    mlx_se: Any,
    x_nchw: np.ndarray,
    torch_trace: dict[str, np.ndarray],
    variant: str,
) -> dict[str, Any]:
    x_nhwc = _as_mlx_nhwc(x_nchw)
    pool = _pool_variant(x_nhwc, variant)
    conv_reduce = mlx_se.conv_reduce(pool)
    act1 = mlx_silu(conv_reduce)
    conv_expand = mlx_se.conv_expand(act1)
    gate = mlx_sigmoid(conv_expand)
    output = x_nhwc * gate
    pool_nchw = _mlx_nchw(pool)
    output_nchw = _mlx_nchw(output)
    return {
        "variant": variant,
        "pool_delta": _array_delta_summary(torch_trace["se.pool"], pool_nchw),
        "conv_reduce_delta": _array_delta_summary(
            torch_trace["se.conv_reduce"],
            _mlx_nchw(conv_reduce),
        ),
        "act1_silu_delta": _array_delta_summary(
            torch_trace["se.act1_silu"],
            _mlx_nchw(act1),
        ),
        "gate_sigmoid_delta": _array_delta_summary(
            torch_trace["se.gate_sigmoid"],
            _mlx_nchw(gate),
        ),
        "output_delta": _array_delta_summary(torch_trace["se.output_multiply"], output_nchw),
    }


def _pool_variant(x_nhwc: Any, variant: str) -> Any:
    import mlx.core as mx

    if variant == "mean_tuple":
        return mx.mean(x_nhwc, axis=(1, 2), keepdims=True)
    if variant == "mean_h_then_w":
        return mx.mean(mx.mean(x_nhwc, axis=1, keepdims=True), axis=2, keepdims=True)
    if variant == "mean_w_then_h":
        return mx.mean(mx.mean(x_nhwc, axis=2, keepdims=True), axis=1, keepdims=True)
    if variant == "sum_tuple_div":
        _, height, width, _ = x_nhwc.shape
        return mx.sum(x_nhwc, axis=(1, 2), keepdims=True) / np.float32(int(height) * int(width))
    raise ValueError(f"unknown SE pool variant {variant!r}")


def _mlx_nchw(value_nhwc: Any) -> np.ndarray:
    return nhwc_to_nchw(_mlx_array_to_numpy(value_nhwc))


def _max_or_inf(summary: dict[str, Any]) -> float:
    value = summary.get("max_abs_delta")
    if isinstance(value, bool) or value is None:
        return float("inf")
    return float(value)


__all__ = [
    "POOL_VARIANTS",
    "SCHEMA_VERSION",
    "build_mlx_segnet_stage0_se_pool_variants_manifest",
    "write_mlx_segnet_stage0_se_pool_variants_manifest",
]
