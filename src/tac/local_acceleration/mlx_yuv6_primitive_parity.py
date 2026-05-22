# SPDX-License-Identifier: MIT
"""MLX primitive parity for upstream ``rgb_to_yuv6`` preprocessing."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_yuv6_primitive_parity.v1"
PASS_VERDICT = "PASS_MLX_YUV6_PRIMITIVE_PARITY"
FAIL_VERDICT = "FAIL_MLX_YUV6_PRIMITIVE_PARITY"
DEFAULT_EPSILON = 1.0e-5

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


def json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def mlx_rgb_to_yuv6(rgb_chw: Any) -> Any:
    """Return MLX-native YUV6 for NCHW-like RGB tensors.

    The formula mirrors `upstream/frame_utils.py::rgb_to_yuv6`: BT.601 luma,
    clamped full-range U/V, 2x2 chroma averaging, and four luma polyphase
    channels. The channel axis is expected at ``-3``.
    """

    import mlx.core as mx

    rgb_input = mx.array(rgb_chw, dtype=mx.float32)
    h, w = int(rgb_input.shape[-2]), int(rgb_input.shape[-1])
    h2, w2 = h // 2, w // 2
    rgb = rgb_input[..., :, : 2 * h2, : 2 * w2]

    red = rgb[..., 0, :, :]
    green = rgb[..., 1, :, :]
    blue = rgb[..., 2, :, :]

    y = mx.clip(red * 0.299 + green * 0.587 + blue * 0.114, 0.0, 255.0)
    u = mx.clip((blue - y) / 1.772 + 128.0, 0.0, 255.0)
    v = mx.clip((red - y) / 1.402 + 128.0, 0.0, 255.0)

    u_sub = (
        u[..., 0::2, 0::2]
        + u[..., 1::2, 0::2]
        + u[..., 0::2, 1::2]
        + u[..., 1::2, 1::2]
    ) * 0.25
    v_sub = (
        v[..., 0::2, 0::2]
        + v[..., 1::2, 0::2]
        + v[..., 0::2, 1::2]
        + v[..., 1::2, 1::2]
    ) * 0.25

    y00 = y[..., 0::2, 0::2]
    y10 = y[..., 1::2, 0::2]
    y01 = y[..., 0::2, 1::2]
    y11 = y[..., 1::2, 1::2]
    out = mx.stack([y00, y10, y01, y11, u_sub, v_sub], axis=-3)
    mx.eval(out)
    return out


def build_mlx_yuv6_primitive_parity_manifest(
    *,
    rgb_chw: Any,
    repo_root: str | Path = ".",
    epsilon: float = DEFAULT_EPSILON,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare MLX-native YUV6 output against upstream PyTorch output."""

    tolerance = float(epsilon)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")
    rgb_np = np.asarray(rgb_chw, dtype=np.float32)
    if rgb_np.ndim < 3:
        raise ValueError(f"expected at least 3 dimensions with channel axis -3, got {rgb_np.shape}")
    if int(rgb_np.shape[-3]) != 3:
        raise ValueError(f"expected channel axis -3 to have size 3, got {rgb_np.shape}")

    upstream = _run_upstream_rgb_to_yuv6(rgb_np, repo_root=Path(repo_root))
    mlx_out = np.asarray(mlx_rgb_to_yuv6(rgb_np), dtype=np.float32)
    deltas = _delta_summary(upstream, mlx_out)
    passed = bool(deltas["max_abs_delta"] <= tolerance)
    blockers = []
    if not passed:
        blockers.append(
            f"mlx_yuv6_max_abs_delta_exceeds_epsilon:{deltas['max_abs_delta']}>{tolerance}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": "tac.local_acceleration.mlx_yuv6_primitive_parity",
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "allowed_use": "primitive_level_mlx_yuv6_parity_only",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "requires_exact_eval_before_promotion": True,
        "passed": passed,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "blockers": blockers,
        "epsilon": tolerance,
        "input_shape": list(rgb_np.shape),
        "output_shape": list(upstream.shape),
        "input_sha256": array_sha256(rgb_np),
        "upstream_output_sha256": array_sha256(upstream),
        "mlx_output_sha256": array_sha256(mlx_out),
        "deltas": deltas,
        "comparator": {
            "upstream_function": "upstream/frame_utils.py::rgb_to_yuv6",
            "mlx_function": "tac.local_acceleration.mlx_yuv6_primitive_parity.mlx_rgb_to_yuv6",
            "channel_axis": -3,
            "dtype": "float32",
        },
    }


def deterministic_rgb_fixture(
    *,
    seed: int = 0,
    batch: int = 3,
    height: int = 384,
    width: int = 512,
) -> np.ndarray:
    """Build a deterministic uint8-like float32 NCHW RGB fixture."""

    if batch < 1 or height < 2 or width < 2:
        raise ValueError("batch, height, and width must be positive; height/width >= 2")
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0,
        256,
        size=(int(batch), 3, int(height), int(width)),
        dtype=np.uint8,
    ).astype(np.float32)


def _run_upstream_rgb_to_yuv6(rgb_np: np.ndarray, *, repo_root: Path) -> np.ndarray:
    import torch

    upstream_path = str((repo_root / "upstream").resolve())
    inserted = False
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
        inserted = True
    try:
        frame_utils = importlib.import_module("frame_utils")
        with torch.no_grad():
            out = frame_utils.rgb_to_yuv6(torch.from_numpy(np.ascontiguousarray(rgb_np)))
        return out.detach().cpu().numpy().astype(np.float32)
    finally:
        if inserted:
            try:
                sys.path.remove(upstream_path)
            except ValueError:  # pragma: no cover
                pass


def _delta_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    diff = np.asarray(candidate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    abs_diff = np.abs(diff)
    rms = float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))
    ref_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    return {
        "max_abs_delta": float(abs_diff.max(initial=0.0)),
        "mean_abs_delta": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rms_delta": rms,
        "relative_rms_delta": 0.0 if ref_rms == 0.0 else float(rms / ref_rms),
    }


__all__ = [
    "DEFAULT_EPSILON",
    "FAIL_VERDICT",
    "FALSE_AUTHORITY",
    "PASS_VERDICT",
    "SCHEMA_VERSION",
    "array_sha256",
    "build_mlx_yuv6_primitive_parity_manifest",
    "deterministic_rgb_fixture",
    "json_text",
    "mlx_rgb_to_yuv6",
]
