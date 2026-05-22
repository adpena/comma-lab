# SPDX-License-Identifier: MIT
"""Cross-input probe for localizing MLX SegNet stage-0 drift.

This diagnostic compares each stage-0 block-0 MLX op in two modes:

* native MLX input, where upstream MLX drift is allowed to propagate;
* torch-synchronized input, where the op receives the exact PyTorch tensor
  that the equivalent PyTorch op received.

The second mode separates operator-local drift from propagated input drift.
It is local implementation evidence only and must not be used as score
authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    _mlx_array_to_numpy,
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

SCHEMA_VERSION = "mlx_segnet_stage0_cross_input_probe.v1"


def build_mlx_segnet_stage0_cross_input_probe_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
    cliff_threshold: float = 1.0e-4,
) -> dict[str, Any]:
    """Build a stage-0 op-localization manifest for MLX SegNet drift."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU cross-input probes are local "
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
    with temporary_mlx_device(device_type):
        mlx_segnet = torch_segnet_to_mlx(dist.segnet)
        native_values = _mlx_stage0_block0_depthwise_native_values(mlx_segnet, seg)
        forced_values = _mlx_stage0_block0_depthwise_forced_values(mlx_segnet, torch_values)

    rows = _cross_input_rows(
        torch_values=torch_values,
        native_values=native_values,
        forced_values=forced_values,
        cliff_threshold=float(cliff_threshold),
    )
    native_cliff = _first_cliff(rows, key="native_mlx_vs_torch")
    forced_cliff = _first_cliff(rows, key="forced_torch_input_mlx_vs_torch")
    verdict = _cross_input_verdict(rows, cliff_threshold=float(cliff_threshold))

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_segnet_cross_input_probe",
        "verdict": verdict,
        "passed": verdict == "NO_STAGE0_BLOCK0_CROSS_INPUT_CLIFF",
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
        "cliff_threshold": float(cliff_threshold),
        "native_drift_cliff": native_cliff,
        "forced_input_drift_cliff": forced_cliff,
        "rows": rows,
        "authority_status": (
            "MLX SegNet cross-input probes are diagnostic local implementation "
            "evidence only; exact auth eval remains required for score claims "
            "and promotion."
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


def write_mlx_segnet_stage0_cross_input_probe_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a cross-input probe manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_stage0_block0_depthwise_values(torch_segnet: Any, x_nchw: np.ndarray) -> dict[str, np.ndarray]:
    import torch

    torch_segnet.eval()
    model = torch_segnet.encoder.model
    block = model.blocks[0][0]
    if type(block).__name__ != "DepthwiseSeparableConv":
        raise NotImplementedError(
            "stage_0.block_0 cross-input probe currently expects "
            f"DepthwiseSeparableConv, got {type(block).__name__}"
        )

    raw_input = np.array(x_nchw, dtype=np.float32, copy=True, order="C")
    x = torch.from_numpy(raw_input)
    with torch.inference_mode():
        stem = model.bn1(model.conv_stem(x))
        conv_dw = block.conv_dw(stem)
        bn1 = block.bn1(conv_dw)
        se = block.se(bn1)
        conv_pw = block.conv_pw(se)
        bn2 = block.bn2(conv_pw)
        values = {
            "input": raw_input,
            "encoder.stem": _torch_to_numpy(stem),
            "encoder.stage_0.block_0.conv_dw": _torch_to_numpy(conv_dw),
            "encoder.stage_0.block_0.bn1": _torch_to_numpy(bn1),
            "encoder.stage_0.block_0.se": _torch_to_numpy(se),
            "encoder.stage_0.block_0.conv_pw": _torch_to_numpy(conv_pw),
            "encoder.stage_0.block_0.bn2": _torch_to_numpy(bn2),
        }
        if bool(getattr(block, "has_skip", False)):
            values["encoder.stage_0.block_0.residual_add"] = _torch_to_numpy(bn2 + stem)
    return values


def _mlx_stage0_block0_depthwise_native_values(
    mlx_segnet: Any,
    x_nchw: np.ndarray,
) -> dict[str, np.ndarray]:
    import mlx.core as mx

    encoder_model = mlx_segnet.encoder.model
    block = encoder_model.stages[0].blocks[0]
    if type(block).__name__ != "MLXDepthwiseSeparableConvAdapter":
        raise NotImplementedError(
            "stage_0.block_0 cross-input probe currently expects "
            f"MLXDepthwiseSeparableConvAdapter, got {type(block).__name__}"
        )

    raw_input = np.array(x_nchw, dtype=np.float32, copy=True)
    x = mx.array(nchw_to_nhwc(raw_input))
    stem = encoder_model.stem(x)
    conv_dw = block.conv_dw(stem)
    bn1 = block.bn1(conv_dw)
    se = block.se(bn1)
    conv_pw = block.conv_pw(se)
    bn2 = block.bn2(conv_pw)
    values = {
        "input": raw_input,
        "encoder.stem": _mlx_nchw(stem),
        "encoder.stage_0.block_0.conv_dw": _mlx_nchw(conv_dw),
        "encoder.stage_0.block_0.bn1": _mlx_nchw(bn1),
        "encoder.stage_0.block_0.se": _mlx_nchw(se),
        "encoder.stage_0.block_0.conv_pw": _mlx_nchw(conv_pw),
        "encoder.stage_0.block_0.bn2": _mlx_nchw(bn2),
    }
    if bool(getattr(block, "has_skip", False)):
        values["encoder.stage_0.block_0.residual_add"] = _mlx_nchw(bn2 + stem)
    return values


def _mlx_stage0_block0_depthwise_forced_values(
    mlx_segnet: Any,
    torch_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    encoder_model = mlx_segnet.encoder.model
    block = encoder_model.stages[0].blocks[0]
    values = {
        "input": np.array(torch_values["input"], dtype=np.float32, copy=True),
        "encoder.stem": _mlx_nchw(encoder_model.stem(_as_mlx_nhwc(torch_values["input"]))),
        "encoder.stage_0.block_0.conv_dw": _mlx_nchw(
            block.conv_dw(_as_mlx_nhwc(torch_values["encoder.stem"]))
        ),
        "encoder.stage_0.block_0.bn1": _mlx_nchw(
            block.bn1(_as_mlx_nhwc(torch_values["encoder.stage_0.block_0.conv_dw"]))
        ),
        "encoder.stage_0.block_0.se": _mlx_nchw(
            block.se(_as_mlx_nhwc(torch_values["encoder.stage_0.block_0.bn1"]))
        ),
        "encoder.stage_0.block_0.conv_pw": _mlx_nchw(
            block.conv_pw(_as_mlx_nhwc(torch_values["encoder.stage_0.block_0.se"]))
        ),
        "encoder.stage_0.block_0.bn2": _mlx_nchw(
            block.bn2(_as_mlx_nhwc(torch_values["encoder.stage_0.block_0.conv_pw"]))
        ),
    }
    if bool(getattr(block, "has_skip", False)):
        values["encoder.stage_0.block_0.residual_add"] = _mlx_nchw(
            _as_mlx_nhwc(values["encoder.stage_0.block_0.bn2"])
            + _as_mlx_nhwc(torch_values["encoder.stem"])
        )
    return values


def _cross_input_rows(
    *,
    torch_values: dict[str, np.ndarray],
    native_values: dict[str, np.ndarray],
    forced_values: dict[str, np.ndarray],
    cliff_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(torch_values):
        torch_value = torch_values[name]
        native_summary = _array_delta_summary(torch_value, native_values[name])
        forced_summary = _array_delta_summary(torch_value, forced_values[name])
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
                "native_minus_forced_max_abs_delta": (
                    None if native_max is None or forced_max is None else native_max - forced_max
                ),
            }
        )
    return rows


def _cross_input_verdict(rows: list[dict[str, Any]], *, cliff_threshold: float) -> str:
    by_name = {row["name"]: row for row in rows}
    conv_pw = by_name.get("encoder.stage_0.block_0.conv_pw")
    bn2 = by_name.get("encoder.stage_0.block_0.bn2")
    if _forced_max(conv_pw) > cliff_threshold:
        return "CONV_PW_OPERATOR_DRIFT_CANDIDATE"
    if _forced_max(bn2) > cliff_threshold:
        return "BN2_OPERATOR_DRIFT_CANDIDATE"
    if _native_max(conv_pw) > cliff_threshold or _native_max(bn2) > cliff_threshold:
        return "UPSTREAM_INPUT_DRIFT_DOMINATES"
    for row in rows:
        if row["name"] in {"encoder.stem", "encoder.stage_0.block_0.conv_pw", "encoder.stage_0.block_0.bn2"}:
            continue
        if _forced_max(row) > cliff_threshold:
            return f"EARLY_OPERATOR_DRIFT_CANDIDATE:{row['name']}"
    for row in rows:
        if _native_max(row) > cliff_threshold:
            return f"EARLY_INPUT_DRIFT_DOMINATES:{row['name']}"
    return "NO_STAGE0_BLOCK0_CROSS_INPUT_CLIFF"


def _first_cliff(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any] | None:
    for row in rows:
        summary = row[key]
        max_abs = _as_float(summary.get("max_abs_delta"))
        if max_abs is None or row[f"{'native' if key.startswith('native') else 'forced'}_exceeds_cliff_threshold"]:
            return {
                "name": row["name"],
                "max_abs_delta": max_abs,
                "p99_abs_delta": _as_float(summary.get("p99_abs_delta")),
            }
    return None


def _forced_max(row: dict[str, Any] | None) -> float:
    if row is None:
        return float("inf")
    return _as_float(row["forced_torch_input_mlx_vs_torch"].get("max_abs_delta")) or 0.0


def _native_max(row: dict[str, Any] | None) -> float:
    if row is None:
        return float("inf")
    return _as_float(row["native_mlx_vs_torch"].get("max_abs_delta")) or 0.0


def _as_mlx_nhwc(value_nchw: np.ndarray) -> Any:
    import mlx.core as mx

    return mx.array(nchw_to_nhwc(np.asarray(value_nchw, dtype=np.float32)))


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


__all__ = [
    "SCHEMA_VERSION",
    "build_mlx_segnet_stage0_cross_input_probe_manifest",
    "write_mlx_segnet_stage0_cross_input_probe_manifest",
]
