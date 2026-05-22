# SPDX-License-Identifier: MIT
"""Full-logit probe for repaired stage-0 SegNet squeeze-excite variants."""

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
    _jsonable,
    _segnet_deltas,
    run_torch_segnet_layer_trace_nchw,
)

SCHEMA_VERSION = "mlx_segnet_repaired_stage0_se_probe.v1"

VARIANTS = (
    ("native", "mean_tuple", "native"),
    ("cpu_pool_repair", "mean_w_then_h", "native"),
    ("gpu_conv_expand_repair", "mean_tuple", "explicit_ordered_1x1"),
    ("combined_repair", "mean_w_then_h", "explicit_ordered_1x1"),
)


def build_mlx_segnet_repaired_stage0_se_probe_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    """Compare native and repaired stage-0 SE variants through final logits."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU repaired-SE probes are local "
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
    torch_trace = run_torch_segnet_layer_trace_nchw(dist.segnet, seg)
    torch_logits = torch_trace["segmentation_head.logits"]
    torch_se = dist.segnet.encoder.model.blocks[0][0].se
    with temporary_mlx_device(device_type):
        mlx_segnet = torch_segnet_to_mlx(dist.segnet)
        rows = [
            _variant_row(
                label=label,
                pool_variant=pool_variant,
                conv_expand_variant=conv_expand_variant,
                mlx_segnet=mlx_segnet,
                torch_se=torch_se,
                x_nchw=seg,
                torch_logits=torch_logits,
            )
            for label, pool_variant, conv_expand_variant in VARIANTS
        ]

    best_argmax = min(
        rows,
        key=lambda row: (
            _none_to_inf(row["segnet_argmax_diff_pixels"]),
            _none_to_inf(row["segnet_logit_abs_max"]),
        ),
    )
    native = rows[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_segnet_repaired_se_probe",
        "verdict": _repair_verdict(native=native, best=best_argmax),
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
        "native_argmax_diff_pixels": native["segnet_argmax_diff_pixels"],
        "best_variant": {
            "label": best_argmax["label"],
            "pool_variant": best_argmax["pool_variant"],
            "conv_expand_variant": best_argmax["conv_expand_variant"],
            "segnet_argmax_diff_pixels": best_argmax["segnet_argmax_diff_pixels"],
            "segnet_logit_abs_max": best_argmax["segnet_logit_abs_max"],
        },
        "rows": rows,
        "authority_status": (
            "MLX SegNet repaired-SE probes are diagnostic local implementation "
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


def write_mlx_segnet_repaired_stage0_se_probe_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a repaired-SE probe manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variant_row(
    *,
    label: str,
    pool_variant: str,
    conv_expand_variant: str,
    mlx_segnet: Any,
    torch_se: Any,
    x_nchw: np.ndarray,
    torch_logits: np.ndarray,
) -> dict[str, Any]:
    logits = _run_mlx_segnet_with_stage0_se_variant(
        mlx_segnet=mlx_segnet,
        torch_se=torch_se,
        x_nchw=x_nchw,
        pool_variant=pool_variant,
        conv_expand_variant=conv_expand_variant,
    )
    blockers: list[str] = []
    logit_delta, argmax_diff_pixels, pixel_count, mismatch_detail = _segnet_deltas(
        {"segnet": torch_logits},
        {"segnet": logits},
        blockers,
    )
    return {
        "label": label,
        "pool_variant": pool_variant,
        "conv_expand_variant": conv_expand_variant,
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


def _run_mlx_segnet_with_stage0_se_variant(
    *,
    mlx_segnet: Any,
    torch_se: Any,
    x_nchw: np.ndarray,
    pool_variant: str,
    conv_expand_variant: str,
) -> np.ndarray:
    import mlx.core as mx

    encoder_model = mlx_segnet.encoder.model
    stage_out_idx = set(encoder_model.stage_out_idx)
    x = mx.array(nchw_to_nhwc(x_nchw))
    stem = encoder_model.stem(x)
    out = stem
    features: list[Any] = []
    if 0 in stage_out_idx:
        features.append(out)

    stage0 = encoder_model.stages[0]
    block0 = stage0.blocks[0]
    shortcut = out
    out = block0.conv_dw(out)
    out = block0.bn1(out)
    out = _stage0_se_variant(
        mlx_se=block0.se,
        torch_se=torch_se,
        x_nhwc=out,
        pool_variant=pool_variant,
        conv_expand_variant=conv_expand_variant,
    )
    out = block0.conv_pw(out)
    out = block0.bn2(out)
    if bool(getattr(block0, "has_skip", False)):
        out = out + shortcut
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


def _stage0_se_variant(
    *,
    mlx_se: Any,
    torch_se: Any,
    x_nhwc: Any,
    pool_variant: str,
    conv_expand_variant: str,
) -> Any:
    pool = _pool(x_nhwc, variant=pool_variant)
    reduced = mlx_silu(mlx_se.conv_reduce(pool))
    if conv_expand_variant == "native":
        expanded = mlx_se.conv_expand(reduced)
    elif conv_expand_variant == "explicit_ordered_1x1":
        expanded = _explicit_ordered_1x1_conv_nhwc(torch_se.conv_expand, reduced)
    else:
        raise ValueError(f"unknown conv_expand_variant {conv_expand_variant!r}")
    return x_nhwc * mlx_sigmoid(expanded)


def _pool(x_nhwc: Any, *, variant: str) -> Any:
    import mlx.core as mx

    if variant == "mean_tuple":
        return mx.mean(x_nhwc, axis=(1, 2), keepdims=True)
    if variant == "mean_w_then_h":
        return mx.mean(mx.mean(x_nhwc, axis=2, keepdims=True), axis=1, keepdims=True)
    raise ValueError(f"unknown pool_variant {variant!r}")


def _explicit_ordered_1x1_conv_nhwc(torch_conv: Any, input_nhwc: Any) -> Any:
    import mlx.core as mx

    weight = torch_conv.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    if weight.ndim != 4 or weight.shape[2:] != (1, 1):
        raise NotImplementedError(f"expected 1x1 Conv2d weight, got {weight.shape}")
    if int(torch_conv.groups) != 1:
        raise NotImplementedError("explicit stage-0 SE conv requires groups=1")
    weight = np.ascontiguousarray(weight[:, :, 0, 0])
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
    if torch_conv.bias is not None:
        bias = torch_conv.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        out = out + mx.array(np.ascontiguousarray(bias.reshape(1, 1, 1, out_channels)))
    return out


def _repair_verdict(*, native: dict[str, Any], best: dict[str, Any]) -> str:
    native_argmax = _none_to_inf(native["segnet_argmax_diff_pixels"])
    best_argmax = _none_to_inf(best["segnet_argmax_diff_pixels"])
    if best_argmax < native_argmax:
        return f"REPAIRED_SE_IMPROVES_ARGMAX:{best['label']}"
    native_logit = _none_to_inf(native["segnet_logit_abs_max"])
    best_logit = _none_to_inf(best["segnet_logit_abs_max"])
    if best_logit < native_logit:
        return f"REPAIRED_SE_IMPROVES_LOGITS:{best['label']}"
    return "REPAIRED_SE_NO_IMPROVEMENT"


def _none_to_inf(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return float("inf")
    return float(value)


__all__ = [
    "SCHEMA_VERSION",
    "VARIANTS",
    "build_mlx_segnet_repaired_stage0_se_probe_manifest",
    "write_mlx_segnet_repaired_stage0_se_probe_manifest",
]
