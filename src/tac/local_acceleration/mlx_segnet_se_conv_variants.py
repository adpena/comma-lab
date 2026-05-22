# SPDX-License-Identifier: MIT
"""Probe native-vs-explicit MLX 1x1 convs inside SegNet stage-0 SE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    _mlx_array_to_numpy,
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

SCHEMA_VERSION = "mlx_segnet_stage0_se_conv_variants.v1"


def build_mlx_segnet_stage0_se_conv_variants_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    """Compare native MLX Conv2d with explicit ordered 1x1 SE convs."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU SE conv variant probes are local "
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
    torch_se = dist.segnet.encoder.model.blocks[0][0].se
    with temporary_mlx_device(device_type):
        mlx_se = torch_segnet_to_mlx(dist.segnet).encoder.model.stages[0].blocks[0].se
        rows = [
            _conv_variant_row(
                op_name="conv_reduce",
                mlx_conv=mlx_se.conv_reduce,
                torch_conv=torch_se.conv_reduce,
                input_nchw=torch_trace["se.pool"],
                reference_nchw=torch_trace["se.conv_reduce"],
            ),
            _conv_variant_row(
                op_name="conv_expand",
                mlx_conv=mlx_se.conv_expand,
                torch_conv=torch_se.conv_expand,
                input_nchw=torch_trace["se.act1_silu"],
                reference_nchw=torch_trace["se.conv_expand"],
            ),
        ]

    worst_native = max(rows, key=lambda row: _max_or_neg_inf(row["native_mlx_conv2d_delta"]))
    worst_explicit = max(rows, key=lambda row: _max_or_neg_inf(row["explicit_ordered_1x1_delta"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_segnet_se_conv_variants",
        "verdict": _conv_variant_verdict(rows),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
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
        "worst_native_row": {
            "op": worst_native["op"],
            "max_abs_delta": worst_native["native_mlx_conv2d_delta"]["max_abs_delta"],
            "p99_abs_delta": worst_native["native_mlx_conv2d_delta"]["p99_abs_delta"],
        },
        "worst_explicit_row": {
            "op": worst_explicit["op"],
            "max_abs_delta": worst_explicit["explicit_ordered_1x1_delta"]["max_abs_delta"],
            "p99_abs_delta": worst_explicit["explicit_ordered_1x1_delta"]["p99_abs_delta"],
        },
        "rows": rows,
        "authority_status": (
            "MLX SegNet SE conv variant probes are diagnostic local implementation "
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


def write_mlx_segnet_stage0_se_conv_variants_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write an SE conv variant manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _conv_variant_row(
    *,
    op_name: str,
    mlx_conv: Any,
    torch_conv: Any,
    input_nchw: np.ndarray,
    reference_nchw: np.ndarray,
) -> dict[str, Any]:
    input_nhwc = _as_mlx_nhwc(input_nchw)
    native = nhwc_to_nchw(_mlx_array_to_numpy(mlx_conv(input_nhwc)))
    explicit = _explicit_ordered_1x1_conv_nchw(torch_conv, input_nhwc)
    return {
        "op": op_name,
        "native_mlx_conv2d_delta": _array_delta_summary(reference_nchw, native),
        "explicit_ordered_1x1_delta": _array_delta_summary(reference_nchw, explicit),
        "explicit_minus_native_max_abs_delta": (
            _max_or_inf(_array_delta_summary(reference_nchw, explicit))
            - _max_or_inf(_array_delta_summary(reference_nchw, native))
        ),
    }


def _explicit_ordered_1x1_conv_nchw(torch_conv: Any, input_nhwc: Any) -> np.ndarray:
    import mlx.core as mx

    weight = _torch_weight_1x1(torch_conv)
    bias = _torch_bias(torch_conv)
    out_channels, in_channels = weight.shape
    out = mx.zeros(
        (input_nhwc.shape[0], input_nhwc.shape[1], input_nhwc.shape[2], out_channels),
        dtype=input_nhwc.dtype,
    )
    for channel in range(in_channels):
        term_weight = mx.array(
            np.ascontiguousarray(weight[:, channel].reshape(1, 1, 1, out_channels))
        )
        out = out + input_nhwc[:, :, :, channel : channel + 1] * term_weight
    if bias is not None:
        out = out + mx.array(np.ascontiguousarray(bias.reshape(1, 1, 1, out_channels)))
    return nhwc_to_nchw(_mlx_array_to_numpy(out))


def _torch_weight_1x1(torch_conv: Any) -> np.ndarray:
    weight = torch_conv.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    if weight.ndim != 4 or weight.shape[2:] != (1, 1):
        raise NotImplementedError(f"expected 1x1 Conv2d weight, got {weight.shape}")
    if int(torch_conv.groups) != 1:
        raise NotImplementedError("explicit SE conv variant requires groups=1")
    return np.ascontiguousarray(weight[:, :, 0, 0])


def _torch_bias(torch_conv: Any) -> np.ndarray | None:
    if torch_conv.bias is None:
        return None
    return torch_conv.bias.detach().cpu().numpy().astype(np.float32, copy=True)


def _conv_variant_verdict(rows: list[dict[str, Any]]) -> str:
    improved = [
        row["op"]
        for row in rows
        if _max_or_inf(row["explicit_ordered_1x1_delta"])
        < _max_or_inf(row["native_mlx_conv2d_delta"])
    ]
    if improved:
        return "EXPLICIT_1X1_IMPROVES:" + ",".join(improved)
    return "EXPLICIT_1X1_DOES_NOT_IMPROVE"


def _max_or_inf(summary: dict[str, Any]) -> float:
    value = summary.get("max_abs_delta")
    if isinstance(value, bool) or value is None:
        return float("inf")
    return float(value)


def _max_or_neg_inf(summary: dict[str, Any]) -> float:
    value = summary.get("max_abs_delta")
    if isinstance(value, bool) or value is None:
        return float("-inf")
    return float(value)


__all__ = [
    "SCHEMA_VERSION",
    "build_mlx_segnet_stage0_se_conv_variants_manifest",
    "write_mlx_segnet_stage0_se_conv_variants_manifest",
]
