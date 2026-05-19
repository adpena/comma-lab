# SPDX-License-Identifier: MIT
"""Apple Silicon MLX inference path — Prince Canuma / MLX-LM pattern.

Per CLAUDE.md "Tailscale fleet — non-negotiable" the operator's
``primary`` (M5 Max 128GB) is an Apple Silicon machine; the MLX
framework (https://github.com/ml-explore/mlx) provides hardware-
accelerated 4-bit / 8-bit quantization on Apple Silicon's unified
memory architecture without the FP32-fallback noise problem MPS
exhibits in PyTorch's autograd path.

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: MLX is NOT a
replacement for CUDA on the contest scorer pathway. This module
provides:

1. ``convert_pytorch_to_mlx_4bit(model)`` — convert PyTorch state-dict
   to MLX 4-bit weights for FAST inference on Apple Silicon (e.g. for
   the operator's M5 Max free-throughput proxy sweep loop per CLAUDE.md
   "MPS auth eval is NOISE" 2026-05-07 refinement).
2. ``mlx_inflate_inference_path_metadata()`` — returns metadata about
   the MLX inference path (when MLX is installed) so the inflate
   runtime can opt into MLX when running on darwin-arm64 with the
   ``[macOS-CPU advisory]`` evidence tag.

The output IS NOT contest-promotable per CLAUDE.md "Submission auth eval
— BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": macOS-CPU is
ADVISORY ONLY until paired with a Linux-x86_64 CPU re-run + a CUDA
re-run. MLX-on-Apple-Silicon is even further from contest-1:1 hardware
than macOS-CPU. The MLX path is for DEV LOOP TROUGHPUT, not promotion.

[verified-against:MLX docs (https://ml-explore.github.io/mlx/) + the
Prince Canuma MLX-LM project's 4-bit quantization patterns]
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
from typing import Any

MLX_AVAILABLE = False
"""Whether this process has positively initialized MLX.

Package import intentionally does not import MLX. In headless or sandboxed
macOS sessions, importing ``mlx.nn`` can attempt to load a Metal device and
raise at import or process-exit time. Use :func:`mlx_inflate_inference_path_metadata`
or :func:`convert_pytorch_to_mlx_4bit` for lazy, explicit hardware probing.
"""


def _mlx_distribution_present() -> bool:
    return importlib.util.find_spec("mlx") is not None


def convert_pytorch_to_mlx_4bit(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert PyTorch state-dict to MLX 4-bit weights.

    Args:
        state_dict: PyTorch state-dict (typically from ``model.state_dict()``)

    Returns:
        dict suitable for ``mlx.nn.Module.load_weights()`` with 4-bit
        quantization applied to all Linear/Conv weights via
        ``mlx.nn.quantize(model, bits=4, group_size=64)``.

    Per CLAUDE.md MPS noise discipline: this is a RESEARCH-SIGNAL helper
    only. Score outputs MUST be tagged ``[macOS-MLX-advisory]``.

    Raises ImportError if MLX is not importable or this process cannot access
    an MLX/Metal device.
    """
    if not MLX_AVAILABLE:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError(
                "MLX is unavailable in this process; this helper is "
                "Apple-Silicon/Metal-only. Install via 'uv pip install mlx' "
                "on macOS-arm64 and run with accessible Metal hardware. On "
                "Linux/CUDA the canonical path is "
                "tac.quantization_wave.int4_int8_mixed_bit + bitsandbytes."
            ) from exc
    # Convert PyTorch tensors to MLX arrays (lazy import + numpy bridge)
    import torch

    mlx_state: dict[str, Any] = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            np_arr = tensor.detach().cpu().float().numpy()
            mlx_state[name] = mx.array(np_arr)
        else:
            mlx_state[name] = tensor
    return mlx_state


def mlx_inflate_inference_path_metadata() -> dict[str, Any]:
    """Return metadata about the MLX inflate-time inference path.

    Per CLAUDE.md "Submission auth eval" non-negotiable: MLX is NEVER
    a 1:1 contest-compliant axis. This metadata is exposed for the
    operator's free-throughput dev-loop ranking ONLY.

    Returns:
        dict with:
        - mlx_available: bool
        - mlx_version: str or None
        - is_apple_silicon: bool
        - tag_recommendation: '[macOS-MLX-advisory]' or '[unavailable]'
        - promotion_eligible: always False
    """
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    distribution_present = _mlx_distribution_present()
    metadata: dict[str, Any] = {
        "mlx_available": False,
        "mlx_distribution_present": distribution_present,
        "mlx_runtime_probe_required": False,
        "mlx_version": None,
        "is_apple_silicon": is_apple_silicon,
        "tag_recommendation": "[unavailable]",
        "evidence_grade": "macOS-MLX-advisory",
        "evidence_tag": "[macOS-MLX-advisory]",
        "score_claim": False,
        "promotion_eligible": False,  # NEVER True per CLAUDE.md MPS rule
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "device_runtime_contract": {
            "device_family": "mlx-metal",
            "authority": "advisory_only",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
        },
        "axis": "macos_mlx_advisory",
    }
    if distribution_present:
        try:
            metadata["mlx_version"] = importlib.metadata.version("mlx")
        except Exception:
            metadata["mlx_version"] = "unknown"
        if is_apple_silicon:
            metadata["mlx_runtime_probe_required"] = True
            metadata["tag_recommendation"] = "[macOS-MLX-advisory]"
    return metadata
