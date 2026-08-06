"""Import-safe MLX wrapper for the PR130 12-D pose carrier.

This is a local training/parity surface, not a scorer authority.  It follows the
PR130 carrier shape while reusing the existing MLX resize and YUV helpers when
MLX is installed.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PoseCarrierConfig:
    basis_dim: int = 12
    carrier_height: int = 24
    carrier_width: int = 32
    carrier_amplitude: float = 64.0
    camera_hw: tuple[int, int] = (874, 1164)
    scorer_hw: tuple[int, int] = (384, 512)


def mlx_device_probe() -> dict[str, Any]:
    spec = importlib.util.find_spec("mlx")
    if spec is None:
        return {
            "available": False,
            "status": "BLOCKED",
            "reason": "ModuleNotFoundError: No module named 'mlx'",
        }
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
        probe = mx.array([1.0]) + 1.0
        mx.eval(probe)
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "available": False,
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "status": "AVAILABLE",
        "default_device": str(mx.default_device()),
    }


def require_mlx() -> Any:
    probe = mlx_device_probe()
    if not probe["available"]:
        raise RuntimeError(probe["reason"])
    import mlx.core as mx  # type: ignore[import-not-found]

    return mx


def _uint8_ste_mlx(x: Any) -> Any:
    mx = require_mlx()
    clipped = mx.clip(x, 0.0, 255.0)
    rounded = mx.round(clipped)
    return clipped + mx.stop_gradient(rounded - clipped)


def normalized_basis_mlx(raw_basis: Any, config: PoseCarrierConfig | None = None) -> Any:
    """Normalize and resize PR130 basis tensors to scorer resolution.

    Args:
        raw_basis: MLX array shaped ``(K, 3, H, W)``.
    """

    config = config or PoseCarrierConfig()
    mx = require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        resize_nhwc_align_corners_false,
    )

    basis = raw_basis - mx.mean(raw_basis, axis=(1, 2, 3), keepdims=True)
    rms = mx.sqrt(mx.mean(mx.square(basis), axis=(1, 2, 3), keepdims=True) + 1e-6)
    basis = basis / rms
    basis_nhwc = mx.transpose(basis, (0, 2, 3, 1))
    return resize_nhwc_align_corners_false(
        basis_nhwc,
        size=config.scorer_hw,
        mode="bilinear",
    )


def render_slave_camera_nhwc(
    master_camera_nhwc: Any,
    coeff: Any,
    raw_basis: Any,
    config: PoseCarrierConfig | None = None,
    *,
    carrier_base: str = "gray",
) -> Any:
    """Render the PR130 slave frame at camera RGB resolution."""

    config = config or PoseCarrierConfig()
    if carrier_base not in {"gray", "master"}:
        raise ValueError("carrier_base must be 'gray' or 'master'")
    mx = require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        bilinear_eval_roundtrip_downsample_nhwc,
        resize_nhwc_align_corners_false,
    )

    master_eval = bilinear_eval_roundtrip_downsample_nhwc(
        _uint8_ste_mlx(master_camera_nhwc),
        output_hw=config.scorer_hw,
    )
    if carrier_base == "gray":
        base = mx.zeros_like(master_eval) + 127.5
    else:
        base = master_eval

    basis = normalized_basis_mlx(raw_basis, config)
    coeff = mx.reshape(coeff, (*coeff.shape[:-1], config.basis_dim, 1, 1, 1))
    carrier = mx.sum(coeff * basis, axis=-4)
    carrier = carrier * (
        config.carrier_amplitude / float(config.basis_dim) ** 0.5
    )
    slave_eval = _uint8_ste_mlx(base + carrier)
    slave_camera = resize_nhwc_align_corners_false(
        slave_eval,
        size=config.camera_hw,
        mode="bicubic",
    )
    return _uint8_ste_mlx(slave_camera)


def render_pose_pair_yuv6_mlx(
    master_camera_nhwc: Any,
    coeff: Any,
    raw_basis: Any,
    config: PoseCarrierConfig | None = None,
    *,
    carrier_base: str = "gray",
) -> Any:
    """Return PoseNet-ready NHWC tensor with ``t*6 == 12`` channels."""

    config = config or PoseCarrierConfig()
    mx = require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        resize_nhwc_align_corners_false,
        rgb_to_yuv6_mlx,
    )

    slave_camera = render_slave_camera_nhwc(
        master_camera_nhwc,
        coeff,
        raw_basis,
        config,
        carrier_base=carrier_base,
    )
    master_camera = _uint8_ste_mlx(master_camera_nhwc)
    frames = mx.stack([slave_camera, master_camera], axis=-4)
    flat = mx.reshape(
        frames,
        (-1, int(frames.shape[-3]), int(frames.shape[-2]), int(frames.shape[-1])),
    )
    flat_eval = resize_nhwc_align_corners_false(
        flat,
        size=config.scorer_hw,
        mode="bilinear",
    )
    frames_eval = mx.reshape(
        flat_eval,
        (*frames.shape[:-3], config.scorer_hw[0], config.scorer_hw[1], 3),
    )
    yuv6 = rgb_to_yuv6_mlx(frames_eval)
    # Upstream PoseNet preprocesses ``b t c h w`` to ``b (t c) h w``; this is
    # the channels-last equivalent consumed by MLXPoseNetAdapter.
    leading = yuv6.shape[:-4]
    height = int(yuv6.shape[-3])
    width = int(yuv6.shape[-2])
    return mx.reshape(yuv6, (*leading, height, width, 12))


def pose_objective_mlx(
    posenet_adapter: Any,
    master_camera_nhwc: Any,
    coeff: Any,
    raw_basis: Any,
    target_pose: Any,
    config: PoseCarrierConfig | None = None,
    *,
    carrier_base: str = "gray",
) -> Any:
    """Compute first-six PoseNet MSE for a PR130 pose-carrier candidate."""

    mx = require_mlx()
    yuv6 = render_pose_pair_yuv6_mlx(
        master_camera_nhwc,
        coeff,
        raw_basis,
        config,
        carrier_base=carrier_base,
    )
    pred = posenet_adapter(yuv6)["pose"][..., :6]
    target = mx.reshape(target_pose, pred.shape)
    return mx.mean(mx.square(pred - target))
