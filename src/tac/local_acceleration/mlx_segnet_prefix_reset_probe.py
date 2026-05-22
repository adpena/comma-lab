# SPDX-License-Identifier: MIT
"""Prefix-reset probe for MLX SegNet drift localization."""

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
    run_torch_segnet_layer_trace_nchw,
)
from tac.local_acceleration.mlx_segnet_cross_input_probe import (
    _as_mlx_nhwc,
)

SCHEMA_VERSION = "mlx_segnet_prefix_reset_probe.v2"

STAGE0_BLOCK0_BOUNDARIES = (
    "encoder.stage_0.block_0.conv_dw",
    "encoder.stage_0.block_0.bn1",
    "encoder.stage_0.block_0.se",
    "encoder.stage_0.block_0.conv_pw",
    "encoder.stage_0.block_0.bn2",
)

BOUNDARIES = (
    "input",
    "encoder.stem",
    *STAGE0_BLOCK0_BOUNDARIES,
    "encoder.stage_0",
    "encoder.stage_1",
    "encoder.stage_2",
    "encoder.stage_3",
    "encoder.stage_4",
    "encoder.stage_5",
    "encoder.stage_6",
    "encoder.all_features",
    "decoder.block_0",
    "decoder.block_1",
    "decoder.block_2",
    "decoder.block_3",
    "decoder.block_4",
    "decoder.output",
    "segmentation_head.logits",
)


def build_mlx_segnet_prefix_reset_probe_manifest(
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
    torch_values = run_torch_segnet_layer_trace_nchw(dist.segnet, seg)
    torch_logits = torch_values["segmentation_head.logits"]
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


def write_mlx_segnet_prefix_reset_probe_manifest(
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

    x = mx.array(nchw_to_nhwc(x_nchw))
    if _should_reset(reset_order, "encoder.stem"):
        stem = _as_mlx_nhwc(torch_values["encoder.stem"])
    else:
        stem = encoder_model.stem(x)
    out = stem

    stage_out_idx = set(encoder_model.stage_out_idx)
    features: list[Any] = []
    if 0 in stage_out_idx:
        features.append(stem)

    stage0 = encoder_model.stages[0]
    if _should_reset(reset_order, "encoder.stage_0"):
        out = _as_mlx_nhwc(torch_values["encoder.stage_0"])
    else:
        out = _stage0_block0_prefix_reset(
            block0=stage0.blocks[0],
            stem=stem,
            reset_order=reset_order,
            torch_values=torch_values,
        )
        for block in stage0.blocks[1:]:
            out = block(out)
    if 1 in stage_out_idx:
        features.append(out)
    for stage_index, stage in enumerate(encoder_model.stages[1:], start=1):
        stage_name = f"encoder.stage_{stage_index}"
        out = (
            _as_mlx_nhwc(torch_values[stage_name])
            if _should_reset(reset_order, stage_name)
            else stage(out)
        )
        if stage_index + 1 in stage_out_idx:
            features.append(out)
    if mlx_segnet.encoder.prepend_input:
        features = [x, *features]

    if _should_reset(reset_order, "encoder.all_features"):
        features = _torch_features_as_mlx(torch_values)

    decoder_output = _mlx_decoder_with_prefix_reset(
        mlx_segnet=mlx_segnet,
        features=features,
        torch_values=torch_values,
        reset_order=reset_order,
    )
    if _should_reset(reset_order, "decoder.output"):
        decoder_output = _as_mlx_nhwc(torch_values["decoder.output"])
    if _should_reset(reset_order, "segmentation_head.logits"):
        return np.asarray(torch_values["segmentation_head.logits"], dtype=np.float32)
    return nhwc_to_nchw(_mlx_array_to_numpy(mlx_segnet.segmentation_head(decoder_output)))


def _stage0_block0_prefix_reset(
    *,
    block0: Any,
    stem: Any,
    reset_order: int,
    torch_values: dict[str, np.ndarray],
) -> Any:
    out = stem
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
    return out


def _mlx_decoder_with_prefix_reset(
    *,
    mlx_segnet: Any,
    features: list[Any],
    torch_values: dict[str, np.ndarray],
    reset_order: int,
) -> Any:
    spatial_shapes = [(int(feature.shape[1]), int(feature.shape[2])) for feature in features]
    spatial_shapes = spatial_shapes[::-1]
    decoder_features = features[1:][::-1]
    out = decoder_features[0]
    skip_connections = decoder_features[1:]
    for index, decoder_block in enumerate(mlx_segnet.decoder.blocks):
        block_name = f"decoder.block_{index}"
        if _should_reset(reset_order, block_name):
            out = _as_mlx_nhwc(torch_values[block_name])
            continue
        target_height, target_width = spatial_shapes[index + 1]
        skip_connection = (
            skip_connections[index] if index < len(skip_connections) else None
        )
        out = decoder_block(out, target_height, target_width, skip_connection)
    return out


def _torch_features_as_mlx(torch_values: dict[str, np.ndarray]) -> list[Any]:
    feature_items = [
        (int(name.rsplit("_", 1)[1]), value)
        for name, value in torch_values.items()
        if name.startswith("encoder.feature_")
    ]
    return [
        _as_mlx_nhwc(value)
        for _, value in sorted(feature_items, key=lambda item: item[0])
    ]


def _maybe_reset_or_run(
    *,
    name: str,
    out: Any,
    reset_order: int,
    torch_values: dict[str, np.ndarray],
    fn: Any,
) -> Any:
    if _should_reset(reset_order, name):
        return _as_mlx_nhwc(torch_values[name])
    return fn(out)


def _should_reset(reset_order: int, name: str) -> bool:
    return reset_order >= BOUNDARIES.index(name)


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
    "build_mlx_segnet_prefix_reset_probe_manifest",
    "build_mlx_segnet_stage0_prefix_reset_probe_manifest",
    "write_mlx_segnet_prefix_reset_probe_manifest",
    "write_mlx_segnet_stage0_prefix_reset_probe_manifest",
]

build_mlx_segnet_stage0_prefix_reset_probe_manifest = (
    build_mlx_segnet_prefix_reset_probe_manifest
)
write_mlx_segnet_stage0_prefix_reset_probe_manifest = (
    write_mlx_segnet_prefix_reset_probe_manifest
)
