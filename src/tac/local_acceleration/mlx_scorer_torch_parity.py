# SPDX-License-Identifier: MIT
"""PyTorch-vs-MLX scorer parity audit on fixed scorer-input cache windows."""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_adapters import (
    _mlx_array_to_numpy,
    nchw_to_nhwc,
    nhwc_to_nchw,
    run_mlx_distortion_scorer_nchw,
    scorer_distortion_components_numpy,
    temporary_mlx_device,
    torch_distortion_net_to_mlx,
    torch_segnet_to_mlx,
)
from tac.local_acceleration.mlx_scorer_response import (
    GPU_RESEARCH_SIGNAL_BLOCKER,
    _load_upstream_distortion_net,
    load_scorer_input_cache,
)

SCHEMA_VERSION = "mlx_scorer_torch_parity.v1"
SWEEP_SCHEMA_VERSION = "mlx_scorer_torch_parity_sweep.v1"
SEGNET_TRACE_SCHEMA_VERSION = "mlx_segnet_layer_trace.v1"
TORCH_BATCH_INVARIANCE_SCHEMA_VERSION = "torch_segnet_batch_invariance.v1"
PASS_VERDICT = "PASS_MLX_TORCH_SCORER_PARITY"
FAIL_VERDICT = "FAIL_MLX_TORCH_SCORER_PARITY"
PASS_SWEEP_VERDICT = "PASS_MLX_TORCH_SCORER_PARITY_SWEEP"
FAIL_SWEEP_VERDICT = "FAIL_MLX_TORCH_SCORER_PARITY_SWEEP"
PASS_TORCH_BATCH_INVARIANCE_VERDICT = "PASS_TORCH_SEGNET_BATCH_INVARIANCE"
FAIL_TORCH_BATCH_INVARIANCE_VERDICT = "FAIL_TORCH_SEGNET_BATCH_INVARIANCE"


@dataclass(frozen=True)
class MLXTorchParityThresholds:
    """Tolerances for MLX scorer outputs against upstream PyTorch outputs."""

    max_posenet_output_abs_delta: float = 2.0e-3
    max_segnet_logit_abs_delta: float = 1.0e-2
    max_posenet_component_abs_delta: float = 2.0e-5
    max_segnet_argmax_diff_pixels: int = 0


def build_mlx_scorer_torch_parity_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    thresholds: MLXTorchParityThresholds | None = None,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    """Compare MLX scorer outputs against upstream PyTorch on one cache window."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU parity audits are local "
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
    pose = np.asarray(cache.posenet_yuv6_pair[start:stop], dtype=np.float32)
    seg = np.asarray(cache.segnet_last_rgb[start:stop], dtype=np.float32)

    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    torch_outputs = run_torch_distortion_scorer_nchw(dist, pose, seg)
    with temporary_mlx_device(device_type):
        mlx_outputs = run_mlx_distortion_scorer_nchw(
            torch_distortion_net_to_mlx(dist),
            pose,
            seg,
        )

    return build_mlx_scorer_torch_parity_manifest_from_outputs(
        torch_outputs=torch_outputs,
        mlx_outputs=mlx_outputs,
        thresholds=thresholds,
        run_id=run_id,
        device_type=device_type,
        cache_dir=str(cache_dir),
        start_pair=start,
        max_pairs=int(max_pairs),
        pair_window=[start, stop],
        n_samples=stop - start,
        total_pair_count=total_pair_count,
        cache_identity=_cache_identity(cache),
        gpu_research_signal_allowed=bool(allow_gpu_research_signal),
    )


def build_mlx_scorer_torch_parity_sweep_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int | None = None,
    window_pairs: int = 4,
    stride_pairs: int | None = None,
    max_windows: int | None = None,
    thresholds: MLXTorchParityThresholds | None = None,
    run_id: str | None = None,
    allow_gpu_research_signal: bool = False,
    progress_every: int = 0,
    progress_stream: Any | None = None,
) -> dict[str, Any]:
    """Compare PyTorch-vs-MLX scorer parity across a cache-window sweep."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU parity sweeps are local "
            "research-signal implementation checks only; pass "
            "allow_gpu_research_signal=True after recording that rationale"
        )
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if max_pairs is not None and int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1 when set, got {max_pairs}")
    if int(window_pairs) < 1:
        raise ValueError(f"window_pairs must be >= 1, got {window_pairs}")
    stride = int(window_pairs) if stride_pairs is None else int(stride_pairs)
    if stride < 1:
        raise ValueError(f"stride_pairs must be >= 1, got {stride_pairs}")
    if max_windows is not None and int(max_windows) < 1:
        raise ValueError(f"max_windows must be >= 1 when set, got {max_windows}")
    if int(progress_every) < 0:
        raise ValueError(f"progress_every must be >= 0, got {progress_every}")

    cache = load_scorer_input_cache(cache_dir)
    total_pair_count = int(cache.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pair_count:
        raise ValueError(f"start_pair {start} is outside cache pair count {total_pair_count}")
    sweep_stop = total_pair_count if max_pairs is None else min(total_pair_count, start + int(max_pairs))
    windows = _build_windows(
        start=start,
        stop=sweep_stop,
        window_pairs=int(window_pairs),
        stride_pairs=stride,
        max_windows=max_windows,
    )
    if not windows:
        raise ValueError("no parity windows selected")

    limits = thresholds or MLXTorchParityThresholds()
    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    rows: list[dict[str, Any]] = []
    started = time.time()
    stream = sys.stderr if progress_stream is None else progress_stream
    with temporary_mlx_device(device_type):
        adapter = torch_distortion_net_to_mlx(dist)
        for row_index, (window_start, window_stop) in enumerate(windows):
            pose = np.asarray(cache.posenet_yuv6_pair[window_start:window_stop], dtype=np.float32)
            seg = np.asarray(cache.segnet_last_rgb[window_start:window_stop], dtype=np.float32)
            torch_outputs = run_torch_distortion_scorer_nchw(dist, pose, seg)
            mlx_outputs = run_mlx_distortion_scorer_nchw(adapter, pose, seg)
            window_manifest = build_mlx_scorer_torch_parity_manifest_from_outputs(
                torch_outputs=torch_outputs,
                mlx_outputs=mlx_outputs,
                thresholds=limits,
                run_id=f"{run_id}:window{row_index}" if run_id else None,
                device_type=device_type,
                cache_dir=str(cache_dir),
                start_pair=window_start,
                max_pairs=window_stop - window_start,
                pair_window=[window_start, window_stop],
                n_samples=window_stop - window_start,
                total_pair_count=total_pair_count,
                cache_identity=_cache_identity(cache),
                gpu_research_signal_allowed=bool(allow_gpu_research_signal),
            )
            row = {
                "index": row_index,
                "passed": bool(window_manifest["passed"]),
                "verdict": window_manifest["verdict"],
                "blockers": list(window_manifest["blockers"]),
                "pair_window": [window_start, window_stop],
                "n_samples": window_stop - window_start,
                "deltas": window_manifest["deltas"],
            }
            rows.append(row)
            _clear_mlx_runtime_cache()
            if progress_every and (row_index + 1) % int(progress_every) == 0:
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "event": "mlx_torch_parity_sweep_progress",
                            "done_windows": row_index + 1,
                            "total_windows": len(windows),
                            "elapsed_seconds": elapsed,
                            "failed_windows": sum(1 for item in rows if not item["passed"]),
                            "last_pair_window": row["pair_window"],
                            "last_passed": row["passed"],
                            "windows_per_second": (row_index + 1) / elapsed
                            if elapsed > 0
                            else 0.0,
                        },
                        sort_keys=True,
                    ),
                    file=stream,
                    flush=True,
                )
            del pose, seg, torch_outputs, mlx_outputs, window_manifest
            gc.collect()
            _clear_mlx_runtime_cache()

    blockers = _sweep_blockers(rows)
    passed = not blockers
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": PASS_SWEEP_VERDICT if passed else FAIL_SWEEP_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "thresholds": asdict(limits),
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
        "cache_identity": _cache_identity(cache),
        "total_pair_count": total_pair_count,
        "start_pair": start,
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "window_pairs": int(window_pairs),
        "stride_pairs": stride,
        "max_windows": None if max_windows is None else int(max_windows),
        "window_count": len(rows),
        "covered_pair_window": [windows[0][0], windows[-1][1]],
        "summary": _summarize_rows(rows),
        "rows": rows,
        "authority_status": (
            "PyTorch-vs-MLX parity sweep is local implementation evidence only; "
            "paired CPU/CUDA auth eval remains required for score claims and promotion."
        ),
        "device_contract": {
            "gpu_research_signal_blocker": GPU_RESEARCH_SIGNAL_BLOCKER,
            "gpu_research_signal_required": device_type == "gpu",
            "forbidden_uses": [
                "score_claim",
                "promotion",
                "promotable",
                "rank_or_kill",
                "replacement_for_cpu_or_cuda_auth_eval",
            ],
        },
    }


def build_mlx_segnet_layer_trace_manifest(
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
    """Trace PyTorch-vs-MLX SegNet drift across encoder/decoder boundaries."""

    if device_type not in {"cpu", "gpu"}:
        raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
    if device_type == "gpu" and not allow_gpu_research_signal:
        raise ValueError(
            f"{GPU_RESEARCH_SIGNAL_BLOCKER}: MLX GPU layer traces are local "
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
    torch_trace = run_torch_segnet_layer_trace_nchw(dist.segnet, seg)
    with temporary_mlx_device(device_type):
        mlx_trace = run_mlx_segnet_layer_trace_nchw(
            torch_segnet_to_mlx(dist.segnet),
            seg,
        )

    rows = _layer_trace_rows(
        torch_trace=torch_trace,
        mlx_trace=mlx_trace,
        cliff_threshold=float(cliff_threshold),
    )
    logits_torch = torch_trace.get("segmentation_head.logits")
    logits_mlx = mlx_trace.get("segmentation_head.logits")
    if logits_torch is None or logits_mlx is None:
        argmax_detail: dict[str, Any] = {"missing_logits": True}
        argmax_diff_pixels = None
        argmax_diff_fraction = None
    else:
        _, argmax_diff_pixels, pixel_count, argmax_detail = _segnet_deltas(
            {"segnet": logits_torch},
            {"segnet": logits_mlx},
            [],
        )
        argmax_diff_fraction = (
            None
            if argmax_diff_pixels is None or pixel_count in (None, 0)
            else float(argmax_diff_pixels) / float(pixel_count)
        )

    drift_cliff = next((row for row in rows if row["exceeds_cliff_threshold"]), None)
    return {
        "schema_version": SEGNET_TRACE_SCHEMA_VERSION,
        "run_id": run_id,
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
        "cache_identity": _cache_identity(cache),
        "start_pair": start,
        "max_pairs": int(max_pairs),
        "pair_window": [start, stop],
        "n_samples": stop - start,
        "total_pair_count": total_pair_count,
        "cliff_threshold": float(cliff_threshold),
        "trace_count": len(rows),
        "drift_cliff": drift_cliff,
        "segnet_argmax_diff_pixels": argmax_diff_pixels,
        "segnet_argmax_diff_fraction": argmax_diff_fraction,
        "segnet_argmax_mismatch_detail": argmax_detail,
        "rows": rows,
        "authority_status": (
            "PyTorch-vs-MLX SegNet layer traces are diagnostic local "
            "implementation evidence only; exact auth eval remains required for "
            "score claims and promotion."
        ),
    }


def build_torch_segnet_batch_invariance_manifest(
    *,
    cache_dir: str | Path,
    repo_root: str | Path = ".",
    device_type: str = "cpu",
    start_pair: int = 0,
    max_pairs: int = 1,
    run_id: str | None = None,
    max_segnet_argmax_diff_pixels: int = 0,
    max_segnet_logit_abs_delta: float | None = None,
    use_deterministic_algorithms: bool = False,
    cudnn_benchmark: bool | None = None,
    cudnn_deterministic: bool | None = None,
    allow_tf32: bool | None = None,
    mkldnn_enabled: bool | None = None,
    torch_num_threads: int | None = None,
) -> dict[str, Any]:
    """Trace upstream PyTorch SegNet batch-vs-per-sample-loop behavior."""

    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"device_type must be one of cpu/cuda/mps, got {device_type!r}")
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if int(max_segnet_argmax_diff_pixels) < 0:
        raise ValueError(
            "max_segnet_argmax_diff_pixels must be >= 0, "
            f"got {max_segnet_argmax_diff_pixels}"
        )
    if max_segnet_logit_abs_delta is not None and float(max_segnet_logit_abs_delta) < 0.0:
        raise ValueError(
            "max_segnet_logit_abs_delta must be >= 0 when set, "
            f"got {max_segnet_logit_abs_delta}"
        )
    if torch_num_threads is not None and int(torch_num_threads) < 1:
        raise ValueError(f"torch_num_threads must be >= 1 when set, got {torch_num_threads}")

    cache = load_scorer_input_cache(cache_dir)
    total_pair_count = int(cache.pair_indices.shape[0])
    start = int(start_pair)
    if start >= total_pair_count:
        raise ValueError(f"start_pair {start} is outside cache pair count {total_pair_count}")
    stop = min(total_pair_count, start + int(max_pairs))
    seg = np.asarray(cache.segnet_last_rgb[start:stop], dtype=np.float32)

    dist = _load_upstream_distortion_net(Path(repo_root).resolve())
    with _torch_backend_options(
        use_deterministic_algorithms=bool(use_deterministic_algorithms),
        cudnn_benchmark=cudnn_benchmark,
        cudnn_deterministic=cudnn_deterministic,
        allow_tf32=allow_tf32,
        mkldnn_enabled=mkldnn_enabled,
        torch_num_threads=torch_num_threads,
    ):
        outputs = run_torch_segnet_batch_and_loop_nchw(
            dist.segnet,
            seg,
            device_type=device_type,
        )
        backend_metadata = _torch_backend_metadata(device_type)

    blockers: list[str] = []
    logit_delta, argmax_diff_pixels, pixel_count, mismatch_detail = _segnet_deltas(
        {"segnet": outputs["batch"]},
        {"segnet": outputs["loop"]},
        blockers,
    )
    mismatch_detail["comparison_labels"] = {
        "torch": "batch",
        "mlx": "per_sample_loop",
    }
    if (
        argmax_diff_pixels is not None
        and argmax_diff_pixels > int(max_segnet_argmax_diff_pixels)
    ):
        blockers.append(
            "segnet_batch_argmax_diff_pixels_exceeds_threshold:"
            f"{argmax_diff_pixels}>{int(max_segnet_argmax_diff_pixels)}"
        )
    if max_segnet_logit_abs_delta is not None and logit_delta is not None:
        limit = float(max_segnet_logit_abs_delta)
        if logit_delta > limit:
            blockers.append(
                "segnet_batch_logit_abs_delta_exceeds_threshold:"
                f"{logit_delta:.12g}>{limit:.12g}"
            )

    passed = not blockers
    return {
        "schema_version": TORCH_BATCH_INVARIANCE_SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": (
            PASS_TORCH_BATCH_INVARIANCE_VERDICT
            if passed
            else FAIL_TORCH_BATCH_INVARIANCE_VERDICT
        ),
        "passed": passed,
        "blockers": blockers,
        "thresholds": {
            "max_segnet_argmax_diff_pixels": int(max_segnet_argmax_diff_pixels),
            "max_segnet_logit_abs_delta": (
                None
                if max_segnet_logit_abs_delta is None
                else float(max_segnet_logit_abs_delta)
            ),
        },
        "evidence_grade": "[upstream-pytorch-backend-diagnostic]",
        "evidence_tag": "[upstream-pytorch-backend-diagnostic]",
        "score_axis": "[upstream-pytorch-backend-diagnostic]",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "comparison": "segnet_batch_vs_per_sample_loop",
        "device_type": device_type,
        "cache_dir": str(cache_dir),
        "cache_identity": _cache_identity(cache),
        "start_pair": start,
        "max_pairs": int(max_pairs),
        "pair_window": [start, stop],
        "n_samples": stop - start,
        "total_pair_count": total_pair_count,
        "requested_backend_options": {
            "use_deterministic_algorithms": bool(use_deterministic_algorithms),
            "cudnn_benchmark": cudnn_benchmark,
            "cudnn_deterministic": cudnn_deterministic,
            "allow_tf32": allow_tf32,
            "mkldnn_enabled": mkldnn_enabled,
            "torch_num_threads": None if torch_num_threads is None else int(torch_num_threads),
        },
        "torch_backend": backend_metadata,
        "deltas": {
            "segnet_logit_abs_max": logit_delta,
            "segnet_argmax_diff_pixels": argmax_diff_pixels,
            "segnet_argmax_pixel_count": pixel_count,
            "segnet_argmax_diff_fraction": (
                None
                if argmax_diff_pixels is None
                or pixel_count is None
                or pixel_count == 0
                else float(argmax_diff_pixels) / float(pixel_count)
            ),
            "segnet_argmax_mismatch_detail": mismatch_detail,
        },
        "allowed_use": (
            ["backend_batch_invariance_green_for_selected_cache_window"]
            if passed
            else [
                "diagnose_backend_batch_sensitive_argmax_pixels",
                "compare_against_cuda_auth_eval_hardware_before_mlx_exactness_claims",
            ]
        ),
        "authority_status": (
            "PyTorch backend batch-invariance traces are diagnostic evidence only; "
            "they do not replace contest CPU/CUDA auth eval or establish MLX score "
            "authority."
        ),
    }


def run_torch_segnet_layer_trace_nchw(torch_segnet: Any, x_nchw: np.ndarray) -> dict[str, np.ndarray]:
    """Run upstream PyTorch SegNet and capture structural boundary outputs."""

    import torch

    trace: dict[str, np.ndarray] = {"input": np.array(x_nchw, dtype=np.float32, copy=True)}
    x = torch.from_numpy(np.array(x_nchw, dtype=np.float32, copy=True, order="C"))
    with torch.inference_mode():
        features, stage_trace = _torch_segnet_encoder_features(torch_segnet, x)
        trace.update(stage_trace)
        for index, feature in enumerate(features):
            trace[f"encoder.feature_{index}"] = _torch_to_numpy(feature)
        decoder_output, decoder_trace = _torch_segnet_decoder_trace(torch_segnet, features)
        trace.update(decoder_trace)
        trace["decoder.output"] = _torch_to_numpy(decoder_output)
        trace["segmentation_head.logits"] = _torch_to_numpy(
            torch_segnet.segmentation_head(decoder_output)
        )
    return trace


def run_mlx_segnet_layer_trace_nchw(mlx_segnet: Any, x_nchw: np.ndarray) -> dict[str, np.ndarray]:
    """Run MLX SegNet and capture structural boundary outputs."""

    import mlx.core as mx

    trace: dict[str, np.ndarray] = {"input": np.array(x_nchw, dtype=np.float32, copy=True)}
    x = mx.array(nchw_to_nhwc(x_nchw))
    features, stage_trace = _mlx_segnet_encoder_features(mlx_segnet, x)
    trace.update(stage_trace)
    for index, feature in enumerate(features):
        trace[f"encoder.feature_{index}"] = nhwc_to_nchw(_mlx_array_to_numpy(feature))
    decoder_output, decoder_trace = _mlx_segnet_decoder_trace(mlx_segnet, features)
    trace.update(decoder_trace)
    trace["decoder.output"] = nhwc_to_nchw(_mlx_array_to_numpy(decoder_output))
    trace["segmentation_head.logits"] = nhwc_to_nchw(
        _mlx_array_to_numpy(mlx_segnet.segmentation_head(decoder_output))
    )
    return trace


def run_torch_segnet_batch_and_loop_nchw(
    torch_segnet: Any,
    x_nchw: np.ndarray,
    *,
    device_type: str = "cpu",
) -> dict[str, np.ndarray]:
    """Run upstream PyTorch SegNet as one batch and as a per-sample loop."""

    import torch

    device = _torch_device(device_type)
    x = torch.from_numpy(np.array(x_nchw, dtype=np.float32, copy=True, order="C")).to(device)
    model = torch_segnet.eval().to(device)
    with torch.inference_mode():
        batch = model(x).detach().cpu()
        loop = torch.cat(
            [model(x[index : index + 1]).detach().cpu() for index in range(int(x.shape[0]))],
            dim=0,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "batch": batch.numpy().astype(np.float32, copy=True),
        "loop": loop.numpy().astype(np.float32, copy=True),
    }


def run_torch_distortion_scorer_nchw(
    distortion_net: Any,
    posenet_yuv6_pair_nchw: np.ndarray,
    segnet_last_rgb_nchw: np.ndarray,
) -> dict[str, Any]:
    """Run upstream PyTorch PoseNet+SegNet on fixed scorer-input tensors."""

    import torch

    with torch.inference_mode():
        pose_tensor = torch.from_numpy(
            np.array(posenet_yuv6_pair_nchw, dtype=np.float32, copy=True, order="C")
        )
        seg_tensor = torch.from_numpy(
            np.array(segnet_last_rgb_nchw, dtype=np.float32, copy=True, order="C")
        )
        pose_outputs = distortion_net.posenet(pose_tensor)
        seg_outputs = distortion_net.segnet(seg_tensor)
    return {
        "posenet": {
            name: value.detach().cpu().numpy()
            for name, value in pose_outputs.items()
        },
        "segnet": seg_outputs.detach().cpu().numpy(),
    }


def write_mlx_scorer_torch_parity_sweep_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a parity sweep manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_scorer_torch_parity_manifest_from_outputs(
    *,
    torch_outputs: dict[str, Any],
    mlx_outputs: dict[str, Any],
    thresholds: MLXTorchParityThresholds | None = None,
    run_id: str | None = None,
    device_type: str | None = None,
    cache_dir: str | None = None,
    start_pair: int | None = None,
    max_pairs: int | None = None,
    pair_window: list[int] | None = None,
    n_samples: int | None = None,
    total_pair_count: int | None = None,
    cache_identity: dict[str, Any] | None = None,
    gpu_research_signal_allowed: bool = False,
) -> dict[str, Any]:
    """Build a parity manifest from already-computed PyTorch and MLX outputs."""

    limits = thresholds or MLXTorchParityThresholds()
    blockers: list[str] = []

    pose_deltas = _pose_output_deltas(torch_outputs, mlx_outputs)
    max_pose_delta = max(pose_deltas.values(), default=0.0)
    (
        seg_logit_delta,
        seg_argmax_diff_pixels,
        seg_argmax_pixel_count,
        seg_argmax_mismatch_detail,
    ) = _segnet_deltas(
        torch_outputs,
        mlx_outputs,
        blockers,
    )
    components = scorer_distortion_components_numpy(torch_outputs, mlx_outputs)
    pose_component_abs_max = float(np.max(np.abs(components["posenet"])))
    seg_component_diff_samples = int(np.sum(components["segnet"] > 0.0))

    if max_pose_delta > limits.max_posenet_output_abs_delta:
        blockers.append(
            "posenet_output_abs_delta_exceeds_threshold:"
            f"{max_pose_delta:.12g}>{limits.max_posenet_output_abs_delta:.12g}"
        )
    if (
        seg_logit_delta is not None
        and seg_logit_delta > limits.max_segnet_logit_abs_delta
    ):
        blockers.append(
            "segnet_logit_abs_delta_exceeds_threshold:"
            f"{seg_logit_delta:.12g}>{limits.max_segnet_logit_abs_delta:.12g}"
        )
    if pose_component_abs_max > limits.max_posenet_component_abs_delta:
        blockers.append(
            "posenet_component_abs_delta_exceeds_threshold:"
            f"{pose_component_abs_max:.12g}>{limits.max_posenet_component_abs_delta:.12g}"
        )
    if (
        seg_argmax_diff_pixels is not None
        and seg_argmax_diff_pixels > limits.max_segnet_argmax_diff_pixels
    ):
        blockers.append(
            "segnet_argmax_diff_pixels_exceeds_threshold:"
            f"{seg_argmax_diff_pixels}>{limits.max_segnet_argmax_diff_pixels}"
        )

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "thresholds": asdict(limits),
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
        "gpu_research_signal_allowed": bool(gpu_research_signal_allowed),
        "cache_dir": cache_dir,
        "cache_identity": cache_identity,
        "start_pair": start_pair,
        "max_pairs": max_pairs,
        "pair_window": pair_window,
        "n_samples": n_samples,
        "total_pair_count": total_pair_count,
        "deltas": {
            "posenet_outputs_max_abs": pose_deltas,
            "posenet_output_abs_max": max_pose_delta,
            "segnet_logit_abs_max": seg_logit_delta,
            "segnet_argmax_diff_pixels": seg_argmax_diff_pixels,
            "segnet_argmax_pixel_count": seg_argmax_pixel_count,
            "segnet_argmax_diff_fraction": (
                None
                if seg_argmax_diff_pixels is None
                or seg_argmax_pixel_count is None
                or seg_argmax_pixel_count == 0
                else float(seg_argmax_diff_pixels) / float(seg_argmax_pixel_count)
            ),
            "segnet_argmax_mismatch_detail": seg_argmax_mismatch_detail,
            "posenet_component_abs_max": pose_component_abs_max,
            "segnet_component_diff_samples": seg_component_diff_samples,
        },
        "allowed_use": (
            [
                "local_mlx_scorer_training_signal",
                "local_mlx_scorer_response_profiling",
                "candidate_generation_prior",
            ]
            if passed
            else [
                "do_not_use_mlx_scorer_signal_for_training_priority",
                "fall_back_to_upstream_pytorch_or_exact_eval_axis",
            ]
        ),
        "authority_status": (
            "PyTorch-vs-MLX parity is local implementation evidence only; paired "
            "CPU/CUDA auth eval remains required for score claims and promotion."
        ),
        "device_contract": {
            "gpu_research_signal_blocker": GPU_RESEARCH_SIGNAL_BLOCKER,
            "gpu_research_signal_required": device_type == "gpu",
            "forbidden_uses": [
                "score_claim",
                "promotion",
                "promotable",
                "rank_or_kill",
                "replacement_for_cpu_or_cuda_auth_eval",
            ],
        },
    }


def write_mlx_scorer_torch_parity_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a parity manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_mlx_segnet_layer_trace_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a SegNet layer trace manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_torch_segnet_batch_invariance_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a PyTorch backend batch-invariance manifest with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=True)


def _torch_segnet_encoder_features(torch_segnet: Any, x: Any) -> tuple[list[Any], dict[str, np.ndarray]]:
    model = torch_segnet.encoder.model
    stage_out_idx = set(getattr(model, "_stage_out_idx", {}))
    trace: dict[str, np.ndarray] = {}
    out = model.bn1(model.conv_stem(x))
    trace["encoder.stem"] = _torch_to_numpy(out)
    features: list[Any] = []
    if 0 in stage_out_idx:
        features.append(out)
    for index, stage in enumerate(model.blocks):
        for block_index, block in enumerate(stage):
            out = _torch_efficientnet_block_trace(
                block,
                out,
                prefix=f"encoder.stage_{index}.block_{block_index}",
                trace=trace,
            )
            trace[f"encoder.stage_{index}.block_{block_index}"] = _torch_to_numpy(out)
        trace[f"encoder.stage_{index}"] = _torch_to_numpy(out)
        if index + 1 in stage_out_idx:
            features.append(out)
    if not bool(getattr(torch_segnet.encoder, "_is_vgg_style", False)):
        features = [x, *features]
    return features, trace


def _mlx_segnet_encoder_features(mlx_segnet: Any, x: Any) -> tuple[list[Any], dict[str, np.ndarray]]:
    encoder_model = mlx_segnet.encoder.model
    trace: dict[str, np.ndarray] = {}
    out = encoder_model.stem(x)
    trace["encoder.stem"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
    features: list[Any] = []
    if 0 in encoder_model.stage_out_idx:
        features.append(out)
    for index, stage in enumerate(encoder_model.stages):
        for block_index, block in enumerate(stage.blocks):
            out = _mlx_efficientnet_block_trace(
                block,
                out,
                prefix=f"encoder.stage_{index}.block_{block_index}",
                trace=trace,
            )
            trace[f"encoder.stage_{index}.block_{block_index}"] = nhwc_to_nchw(
                _mlx_array_to_numpy(out)
            )
        trace[f"encoder.stage_{index}"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        if index + 1 in encoder_model.stage_out_idx:
            features.append(out)
    if mlx_segnet.encoder.prepend_input:
        features = [x, *features]
    return features, trace


def _torch_efficientnet_block_trace(
    block: Any,
    x: Any,
    *,
    prefix: str,
    trace: dict[str, np.ndarray],
) -> Any:
    class_name = type(block).__name__
    shortcut = x
    if class_name == "DepthwiseSeparableConv":
        out = block.conv_dw(x)
        trace[f"{prefix}.conv_dw"] = _torch_to_numpy(out)
        out = block.bn1(out)
        trace[f"{prefix}.bn1"] = _torch_to_numpy(out)
        out = block.se(out)
        trace[f"{prefix}.se"] = _torch_to_numpy(out)
        out = block.conv_pw(out)
        trace[f"{prefix}.conv_pw"] = _torch_to_numpy(out)
        out = block.bn2(out)
        trace[f"{prefix}.bn2"] = _torch_to_numpy(out)
    elif class_name == "InvertedResidual":
        out = block.conv_pw(x)
        trace[f"{prefix}.conv_pw"] = _torch_to_numpy(out)
        out = block.bn1(out)
        trace[f"{prefix}.bn1"] = _torch_to_numpy(out)
        out = block.conv_dw(out)
        trace[f"{prefix}.conv_dw"] = _torch_to_numpy(out)
        out = block.bn2(out)
        trace[f"{prefix}.bn2"] = _torch_to_numpy(out)
        out = block.se(out)
        trace[f"{prefix}.se"] = _torch_to_numpy(out)
        out = block.conv_pwl(out)
        trace[f"{prefix}.conv_pwl"] = _torch_to_numpy(out)
        out = block.bn3(out)
        trace[f"{prefix}.bn3"] = _torch_to_numpy(out)
    else:
        return block(x)
    if bool(getattr(block, "has_skip", False)):
        out = out + shortcut
        trace[f"{prefix}.residual_add"] = _torch_to_numpy(out)
    return out


def _mlx_efficientnet_block_trace(
    block: Any,
    x: Any,
    *,
    prefix: str,
    trace: dict[str, np.ndarray],
) -> Any:
    class_name = type(block).__name__
    shortcut = x
    if class_name == "MLXDepthwiseSeparableConvAdapter":
        out = block.conv_dw(x)
        trace[f"{prefix}.conv_dw"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.bn1(out)
        trace[f"{prefix}.bn1"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.se(out)
        trace[f"{prefix}.se"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.conv_pw(out)
        trace[f"{prefix}.conv_pw"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.bn2(out)
        trace[f"{prefix}.bn2"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
    elif class_name == "MLXInvertedResidualAdapter":
        out = block.conv_pw(x)
        trace[f"{prefix}.conv_pw"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.bn1(out)
        trace[f"{prefix}.bn1"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.conv_dw(out)
        trace[f"{prefix}.conv_dw"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.bn2(out)
        trace[f"{prefix}.bn2"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.se(out)
        trace[f"{prefix}.se"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.conv_pwl(out)
        trace[f"{prefix}.conv_pwl"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
        out = block.bn3(out)
        trace[f"{prefix}.bn3"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
    else:
        return block(x)
    if bool(getattr(block, "has_skip", False)):
        out = out + shortcut
        trace[f"{prefix}.residual_add"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
    return out


def _torch_segnet_decoder_trace(torch_segnet: Any, features: list[Any]) -> tuple[Any, dict[str, np.ndarray]]:
    spatial_shapes = [(int(feature.shape[2]), int(feature.shape[3])) for feature in features]
    spatial_shapes = spatial_shapes[::-1]
    decoder_features = features[1:][::-1]
    out = decoder_features[0]
    skip_connections = decoder_features[1:]
    trace: dict[str, np.ndarray] = {}
    for index, block in enumerate(torch_segnet.decoder.blocks):
        target_height, target_width = spatial_shapes[index + 1]
        skip_connection = (
            skip_connections[index] if index < len(skip_connections) else None
        )
        out = block(out, target_height, target_width, skip_connection)
        trace[f"decoder.block_{index}"] = _torch_to_numpy(out)
    return out, trace


def _mlx_segnet_decoder_trace(mlx_segnet: Any, features: list[Any]) -> tuple[Any, dict[str, np.ndarray]]:
    spatial_shapes = [(int(feature.shape[1]), int(feature.shape[2])) for feature in features]
    spatial_shapes = spatial_shapes[::-1]
    decoder_features = features[1:][::-1]
    out = decoder_features[0]
    skip_connections = decoder_features[1:]
    trace: dict[str, np.ndarray] = {}
    for index, decoder_block in enumerate(mlx_segnet.decoder.blocks):
        target_height, target_width = spatial_shapes[index + 1]
        skip_connection = (
            skip_connections[index] if index < len(skip_connections) else None
        )
        out = decoder_block(out, target_height, target_width, skip_connection)
        trace[f"decoder.block_{index}"] = nhwc_to_nchw(_mlx_array_to_numpy(out))
    return out, trace


def _layer_trace_rows(
    *,
    torch_trace: dict[str, np.ndarray],
    mlx_trace: dict[str, np.ndarray],
    cliff_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(torch_trace):
        torch_value = torch_trace[name]
        mlx_value = mlx_trace.get(name)
        if mlx_value is None:
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "present_in_torch": True,
                    "present_in_mlx": False,
                    "shape_match": False,
                    "exceeds_cliff_threshold": True,
                    "blockers": ["missing_mlx_trace"],
                }
            )
            continue
        summary = _array_delta_summary(torch_value, mlx_value)
        rows.append(
            {
                "index": index,
                "name": name,
                **summary,
                "exceeds_cliff_threshold": (
                    summary.get("max_abs_delta") is None
                    or float(summary["max_abs_delta"]) > cliff_threshold
                ),
            }
        )
    for name in mlx_trace:
        if name not in torch_trace:
            rows.append(
                {
                    "index": len(rows),
                    "name": name,
                    "present_in_torch": False,
                    "present_in_mlx": True,
                    "shape_match": False,
                    "exceeds_cliff_threshold": True,
                    "blockers": ["missing_torch_trace"],
                }
            )
    return rows


def _array_delta_summary(torch_value: np.ndarray, mlx_value: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(torch_value, dtype=np.float32)
    rhs = np.asarray(mlx_value, dtype=np.float32)
    if lhs.shape != rhs.shape:
        return {
            "present_in_torch": True,
            "present_in_mlx": True,
            "shape_match": False,
            "torch_shape": list(lhs.shape),
            "mlx_shape": list(rhs.shape),
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "rms_delta": None,
            "p95_abs_delta": None,
            "p99_abs_delta": None,
            "blockers": ["shape_mismatch"],
        }
    diff = np.abs(lhs - rhs).astype(np.float64, copy=False)
    return {
        "present_in_torch": True,
        "present_in_mlx": True,
        "shape_match": True,
        "torch_shape": list(lhs.shape),
        "mlx_shape": list(rhs.shape),
        "max_abs_delta": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_delta": float(np.mean(diff)) if diff.size else 0.0,
        "rms_delta": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
        "p95_abs_delta": float(np.quantile(diff, 0.95)) if diff.size else 0.0,
        "p99_abs_delta": float(np.quantile(diff, 0.99)) if diff.size else 0.0,
        "blockers": [],
    }


def _build_windows(
    *,
    start: int,
    stop: int,
    window_pairs: int,
    stride_pairs: int,
    max_windows: int | None,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor < stop:
        window_stop = min(stop, cursor + window_pairs)
        windows.append((cursor, window_stop))
        if max_windows is not None and len(windows) >= int(max_windows):
            break
        cursor += stride_pairs
    return windows


def _sweep_blockers(rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        if not row["passed"]:
            blockers.append(f"window_failed:index={row['index']}:pair_window={row['pair_window']}")
    return blockers


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "posenet_output_abs_max",
        "segnet_logit_abs_max",
        "posenet_component_abs_max",
        "segnet_argmax_diff_pixels",
        "segnet_argmax_diff_fraction",
        "segnet_component_diff_samples",
    )
    summary: dict[str, Any] = {
        "passed_windows": sum(1 for row in rows if row["passed"]),
        "failed_windows": sum(1 for row in rows if not row["passed"]),
    }
    for key in keys:
        values = [float(row["deltas"][key]) for row in rows if row["deltas"].get(key) is not None]
        summary[key] = _distribution(values)
    mismatch_details = [
        row["deltas"]["segnet_argmax_mismatch_detail"]
        for row in rows
        if row["deltas"]["segnet_argmax_mismatch_detail"].get("mismatch_pixels", 0) > 0
    ]
    summary["segnet_argmax_mismatch_pixels_total"] = sum(
        int(detail["mismatch_pixels"]) for detail in mismatch_details
    )
    summary["segnet_argmax_mismatch_min_top2_margin"] = _distribution(
        [float(detail["mismatch_min_top2_margin"]) for detail in mismatch_details]
    )
    summary["segnet_argmax_mismatch_logit_abs_delta_max"] = _distribution(
        [float(detail["mismatch_logit_abs_delta_max"]) for detail in mismatch_details]
    )
    return summary


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _clear_mlx_runtime_cache() -> None:
    try:
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()
    except (AttributeError, ImportError):
        return


def _pose_output_deltas(torch_outputs: dict[str, Any], mlx_outputs: dict[str, Any]) -> dict[str, float]:
    torch_pose = torch_outputs.get("posenet")
    mlx_pose = mlx_outputs.get("posenet")
    if not isinstance(torch_pose, dict) or not isinstance(mlx_pose, dict):
        raise ValueError("outputs must contain posenet dictionaries")
    keys = sorted(set(torch_pose) | set(mlx_pose))
    deltas: dict[str, float] = {}
    for key in keys:
        if key not in torch_pose or key not in mlx_pose:
            deltas[key] = float("inf")
            continue
        lhs = np.asarray(torch_pose.get(key), dtype=np.float32)
        rhs = np.asarray(mlx_pose.get(key), dtype=np.float32)
        deltas[key] = float("inf") if lhs.shape != rhs.shape else float(np.max(np.abs(lhs - rhs)))
    return deltas


def _segnet_deltas(
    torch_outputs: dict[str, Any],
    mlx_outputs: dict[str, Any],
    blockers: list[str],
) -> tuple[float | None, int | None, int | None, dict[str, Any]]:
    torch_seg = np.asarray(torch_outputs.get("segnet"), dtype=np.float32)
    mlx_seg = np.asarray(mlx_outputs.get("segnet"), dtype=np.float32)
    if torch_seg.shape != mlx_seg.shape:
        blockers.append(
            "segnet_shape_mismatch:"
            f"torch={list(torch_seg.shape)}:mlx={list(mlx_seg.shape)}"
        )
        return None, None, None, {"shape_mismatch": True}
    logit_delta = float(np.max(np.abs(torch_seg - mlx_seg)))
    argmax_mismatch = np.argmax(torch_seg, axis=1) != np.argmax(mlx_seg, axis=1)
    argmax_diff = int(np.sum(argmax_mismatch))
    pixel_count = int(argmax_mismatch.size)
    return logit_delta, argmax_diff, pixel_count, _segnet_mismatch_detail(
        torch_seg=torch_seg,
        mlx_seg=mlx_seg,
        argmax_mismatch=argmax_mismatch,
    )


def _segnet_mismatch_detail(
    *,
    torch_seg: np.ndarray,
    mlx_seg: np.ndarray,
    argmax_mismatch: np.ndarray,
) -> dict[str, Any]:
    torch_margin = _top2_margin(torch_seg)
    mlx_margin = _top2_margin(mlx_seg)
    if not np.any(argmax_mismatch):
        return {
            "mismatch_pixels": 0,
            "torch_top2_margin_min": float(np.min(torch_margin)),
            "mlx_top2_margin_min": float(np.min(mlx_margin)),
            "mismatch_torch_top2_margin_min": None,
            "mismatch_mlx_top2_margin_min": None,
            "mismatch_min_top2_margin": None,
            "mismatch_logit_abs_delta_max": None,
            "examples": [],
        }

    mismatch_logit_delta = np.max(
        np.abs(torch_seg - mlx_seg) * np.expand_dims(argmax_mismatch, axis=1),
        axis=1,
    )
    torch_argmax = np.argmax(torch_seg, axis=1)
    mlx_argmax = np.argmax(mlx_seg, axis=1)
    mismatch_torch_margin = torch_margin[argmax_mismatch]
    mismatch_mlx_margin = mlx_margin[argmax_mismatch]
    mismatch_min_margin = np.minimum(mismatch_torch_margin, mismatch_mlx_margin)
    coords = np.argwhere(argmax_mismatch)
    examples: list[dict[str, Any]] = []
    for sample_index, y, x in coords[:8]:
        examples.append(
            {
                "sample_index": int(sample_index),
                "y": int(y),
                "x": int(x),
                "torch_class": int(torch_argmax[sample_index, y, x]),
                "mlx_class": int(mlx_argmax[sample_index, y, x]),
                "torch_top2_margin": float(torch_margin[sample_index, y, x]),
                "mlx_top2_margin": float(mlx_margin[sample_index, y, x]),
                "logit_abs_delta_max": float(mismatch_logit_delta[sample_index, y, x]),
            }
        )
    return {
        "mismatch_pixels": int(coords.shape[0]),
        "torch_top2_margin_min": float(np.min(torch_margin)),
        "mlx_top2_margin_min": float(np.min(mlx_margin)),
        "mismatch_torch_top2_margin_min": float(np.min(mismatch_torch_margin)),
        "mismatch_mlx_top2_margin_min": float(np.min(mismatch_mlx_margin)),
        "mismatch_min_top2_margin": float(np.min(mismatch_min_margin)),
        "mismatch_logit_abs_delta_max": float(np.max(mismatch_logit_delta[argmax_mismatch])),
        "examples": examples,
    }


def _torch_backend_options(
    *,
    use_deterministic_algorithms: bool,
    cudnn_benchmark: bool | None,
    cudnn_deterministic: bool | None,
    allow_tf32: bool | None,
    mkldnn_enabled: bool | None,
    torch_num_threads: int | None,
) -> Any:
    class _TorchBackendOptions:
        def __enter__(self) -> None:
            import torch

            self.torch = torch
            self.previous = {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
                "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
                "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
                "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
                "mkldnn_enabled": (
                    getattr(torch.backends.mkldnn, "enabled", None)
                    if getattr(torch.backends, "mkldnn", None) is not None
                    else None
                ),
                "torch_num_threads": torch.get_num_threads(),
            }
            if use_deterministic_algorithms:
                torch.use_deterministic_algorithms(True)
            if cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            if cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
            if allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
                torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            if mkldnn_enabled is not None and getattr(torch.backends, "mkldnn", None) is not None:
                torch.backends.mkldnn.enabled = bool(mkldnn_enabled)
            if torch_num_threads is not None:
                torch.set_num_threads(int(torch_num_threads))

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            torch = self.torch
            previous = self.previous
            torch.use_deterministic_algorithms(bool(previous["deterministic_algorithms"]))
            if previous["cudnn_benchmark"] is not None:
                torch.backends.cudnn.benchmark = bool(previous["cudnn_benchmark"])
            if previous["cudnn_deterministic"] is not None:
                torch.backends.cudnn.deterministic = bool(previous["cudnn_deterministic"])
            if previous["cuda_matmul_allow_tf32"] is not None:
                torch.backends.cuda.matmul.allow_tf32 = bool(previous["cuda_matmul_allow_tf32"])
            if previous["cudnn_allow_tf32"] is not None:
                torch.backends.cudnn.allow_tf32 = bool(previous["cudnn_allow_tf32"])
            if previous["mkldnn_enabled"] is not None and getattr(torch.backends, "mkldnn", None) is not None:
                torch.backends.mkldnn.enabled = bool(previous["mkldnn_enabled"])
            torch.set_num_threads(int(previous["torch_num_threads"]))

    return _TorchBackendOptions()


def _torch_device(device_type: str) -> Any:
    import torch

    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device_type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    return torch.device("cpu")


def _torch_backend_metadata(device_type: str) -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_type": device_type,
        "num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    mkldnn = getattr(torch.backends, "mkldnn", None)
    if mkldnn is not None:
        metadata["mkldnn_enabled"] = bool(mkldnn.enabled)
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability(device))
    return metadata


def _top2_margin(logits_nchw: np.ndarray) -> np.ndarray:
    if logits_nchw.shape[1] < 2:
        return np.full(logits_nchw.shape[:1] + logits_nchw.shape[2:], np.inf, dtype=np.float32)
    top2 = np.partition(logits_nchw, kth=-2, axis=1)
    return np.asarray(top2[:, -1, ...] - top2[:, -2, ...], dtype=np.float32)


def _cache_identity(cache: Any) -> dict[str, Any]:
    manifest = cache.manifest
    return {
        "path": str(cache.root),
        "schema_version": manifest.get("schema_version"),
        "source": manifest.get("source"),
        "source_kind": manifest.get("source_kind"),
        "hash_domain": manifest.get("hash_domain"),
        "archive_sha256": _manifest_string(manifest, "archive_sha256"),
        "inflated_outputs_aggregate_sha256": _manifest_string(
            manifest,
            "inflated_outputs_aggregate_sha256",
        ),
        "raw_sha256": _manifest_string(manifest, "raw_sha256"),
        "source_video_sha256": _manifest_string(manifest, "source_video_sha256"),
        "array_sha256": manifest.get("array_sha256"),
        "pair_count": int(cache.pair_indices.shape[0]),
        "segnet_last_rgb_shape": list(cache.segnet_last_rgb.shape),
        "posenet_yuv6_pair_shape": list(cache.posenet_yuv6_pair.shape),
        "pair_indices_shape": list(cache.pair_indices.shape),
    }


def _manifest_string(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    return value if isinstance(value, str) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "FAIL_SWEEP_VERDICT",
    "FAIL_TORCH_BATCH_INVARIANCE_VERDICT",
    "FAIL_VERDICT",
    "PASS_SWEEP_VERDICT",
    "PASS_TORCH_BATCH_INVARIANCE_VERDICT",
    "PASS_VERDICT",
    "SCHEMA_VERSION",
    "SEGNET_TRACE_SCHEMA_VERSION",
    "SWEEP_SCHEMA_VERSION",
    "TORCH_BATCH_INVARIANCE_SCHEMA_VERSION",
    "MLXTorchParityThresholds",
    "build_mlx_scorer_torch_parity_manifest",
    "build_mlx_scorer_torch_parity_manifest_from_outputs",
    "build_mlx_scorer_torch_parity_sweep_manifest",
    "build_mlx_segnet_layer_trace_manifest",
    "build_torch_segnet_batch_invariance_manifest",
    "run_mlx_segnet_layer_trace_nchw",
    "run_torch_distortion_scorer_nchw",
    "run_torch_segnet_batch_and_loop_nchw",
    "run_torch_segnet_layer_trace_nchw",
    "write_mlx_scorer_torch_parity_manifest",
    "write_mlx_scorer_torch_parity_sweep_manifest",
    "write_mlx_segnet_layer_trace_manifest",
    "write_torch_segnet_batch_invariance_manifest",
]
