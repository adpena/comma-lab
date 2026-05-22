# SPDX-License-Identifier: MIT
"""Prefix-reset probe for MLX SegNet stage-0 drift localization."""

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
    _jsonable,
    _segnet_deltas,
)
from tac.local_acceleration.mlx_segnet_cross_input_probe import (
    _as_mlx_nhwc,
    _torch_stage0_block0_depthwise_values,
)

SCHEMA_VERSION = "mlx_segnet_stage0_prefix_reset_probe.v1"

BOUNDARIES = (
    "input",
    "encoder.stem",
    "encoder.stage_0.block_0.conv_dw",
    "encoder.stage_0.block_0.bn1",
    "encoder.stage_0.block_0.se",
    "encoder.stage_0.block_0.conv_pw",
    "encoder.stage_0.block_0.bn2",
)


def build_mlx_segnet_stage0_prefix_reset_probe_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    """Run cumulative prefix resets through SegNet logits."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU prefix-reset probes are local "
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
    torch_logits = _torch_segnet_logits(dist.segnet, seg)
    with temporary_mlx_device(device_type):
        mlx_segnet = torch_segnet_to_mlx(dist.segnet)
        rows = [
            _prefix_reset_row(
                mlx_segnet=mlx_segnet,
                x_nchw=seg,
                torch_values=torch_values,
                torch_logits=torch_logits,
                boundary=boundary,
            )
            for boundary in BOUNDARIES
        ]

    baseline = rows[0]
    earliest_zero_argmax = next(
        (row for row in rows if row["segnet_argmax_diff_pixels"] == 0),
        None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_segnet_prefix_reset_probe",
        "verdict": _prefix_reset_verdict(
            baseline=baseline,
            earliest_zero_argmax=earliest_zero_argmax,
        ),
        "passed": baseline["segnet_argmax_diff_pixels"] == 0,
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
        "baseline_boundary": baseline["boundary"],
        "baseline_segnet_argmax_diff_pixels": baseline["segnet_argmax_diff_pixels"],
        "earliest_zero_argmax_boundary": (
            None if earliest_zero_argmax is None else earliest_zero_argmax["boundary"]
        ),
        "rows": rows,
        "authority_status": (
            "MLX SegNet prefix-reset probes are diagnostic local implementation "
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


def write_mlx_segnet_stage0_prefix_reset_probe_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a prefix-reset probe manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prefix_reset_row(
    *,
    mlx_segnet: Any,
    x_nchw: np.ndarray,
    torch_values: dict[str, np.ndarray],
    torch_logits: np.ndarray,
    boundary: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    mlx_logits = _mlx_logits_with_stage0_prefix_reset(
        mlx_segnet=mlx_segnet,
        x_nchw=x_nchw,
        torch_values=torch_values,
        boundary=boundary,
    )
    logit_delta, argmax_diff_pixels, pixel_count, mismatch_detail = _segnet_deltas(
        {"segnet": torch_logits},
        {"segnet": mlx_logits},
        blockers,
    )
    return {
        "boundary": boundary,
        "blockers": blockers,
        "segnet_logit_abs_max": logit_delta,
        "segnet_argmax_diff_pixels": argmax_diff_pixels,
        "segnet_argmax_pixel_count": pixel_count,
        "segnet_argmax_diff_fraction": (
            None
            if argmax_diff_pixels is None or pixel_count is None or pixel_count == 0
            else float(argmax_diff_pixels) / float(pixel_count)
        ),
        "segnet_argmax_mismatch_detail": mismatch_detail,
    }


def _mlx_logits_with_stage0_prefix_reset(
    *,
    mlx_segnet: Any,
    x_nchw: np.ndarray,
    torch_values: dict[str, np.ndarray],
    boundary: str,
) -> np.ndarray:
    import mlx.core as mx

    if boundary not in BOUNDARIES:
        raise ValueError(f"unsupported boundary {boundary!r}")
    reset_order = BOUNDARIES.index(boundary)
    encoder_model = mlx_segnet.encoder.model
    block0 = encoder_model.stages[0].blocks[0]

    x = mx.array(nchw_to_nhwc(x_nchw))
    if reset_order >= BOUNDARIES.index("encoder.stem"):
        stem = _as_mlx_nhwc(torch_values["encoder.stem"])
    else:
        stem = encoder_model.stem(x)
    out = stem

    stage_out_idx = set(encoder_model.stage_out_idx)
    features: list[Any] = []
    if 0 in stage_out_idx:
        features.append(stem)

    out = _maybe_reset_or_run(
        name="encoder.stage_0.block_0.conv_dw",
        out=out,
        reset_order=reset_order,
        torch_values=torch_values,
        fn=block0.conv_dw,
    )
    out = _maybe_reset_or_run(
        name="encoder.stage_0.block_0.bn1",
        out=out,
        reset_order=reset_order,
        torch_values=torch_values,
        fn=block0.bn1,
    )
    out = _maybe_reset_or_run(
        name="encoder.stage_0.block_0.se",
        out=out,
        reset_order=reset_order,
        torch_values=torch_values,
        fn=block0.se,
    )
    out = _maybe_reset_or_run(
        name="encoder.stage_0.block_0.conv_pw",
        out=out,
        reset_order=reset_order,
        torch_values=torch_values,
        fn=block0.conv_pw,
    )
    out = _maybe_reset_or_run(
        name="encoder.stage_0.block_0.bn2",
        out=out,
        reset_order=reset_order,
        torch_values=torch_values,
        fn=block0.bn2,
    )
    if bool(getattr(block0, "has_skip", False)):
        out = out + stem

    stage0 = encoder_model.stages[0]
    for block in stage0.blocks[1:]:
        out = block(out)
    if 1 in stage_out_idx:
        features.append(out)
    for stage_index, stage in enumerate(encoder_model.stages[1:], start=1):
        out = stage(out)
        if stage_index + 1 in stage_out_idx:
            features.append(out)
    if mlx_segnet.encoder.prepend_input:
        features = [x, *features]

    decoder_output = mlx_segnet.decoder(features)
    return nhwc_to_nchw(_mlx_array_to_numpy(mlx_segnet.segmentation_head(decoder_output)))


def _maybe_reset_or_run(
    *,
    name: str,
    out: Any,
    reset_order: int,
    torch_values: dict[str, np.ndarray],
    fn: Any,
) -> Any:
    if reset_order >= BOUNDARIES.index(name):
        return _as_mlx_nhwc(torch_values[name])
    return fn(out)


def _torch_segnet_logits(torch_segnet: Any, x_nchw: np.ndarray) -> np.ndarray:
    import torch

    torch_segnet.eval()
    x = torch.from_numpy(np.array(x_nchw, dtype=np.float32, copy=True, order="C"))
    with torch.inference_mode():
        return torch_segnet(x).detach().cpu().numpy().astype(np.float32, copy=True)


def _prefix_reset_verdict(
    *,
    baseline: dict[str, Any],
    earliest_zero_argmax: dict[str, Any] | None,
) -> str:
    if baseline["segnet_argmax_diff_pixels"] == 0:
        return "NATIVE_MLX_SEGNET_ARGMAX_ALREADY_MATCHES"
    if earliest_zero_argmax is None:
        return "STAGE0_PREFIX_RESETS_DO_NOT_FIX_ARGMAX"
    return f"PREFIX_RESET_FIXES_ARGMAX_AT:{earliest_zero_argmax['boundary']}"


__all__ = [
    "BOUNDARIES",
    "SCHEMA_VERSION",
    "build_mlx_segnet_stage0_prefix_reset_probe_manifest",
    "write_mlx_segnet_stage0_prefix_reset_probe_manifest",
]
