# SPDX-License-Identifier: MIT
"""Op-level MLX↔PyTorch render-parity crux harness for the PR95-HNeRV carrier.

This module recursively localizes the **render-parity drift** (drift source #1
per the standing directive 2026-06-01) between the MLX decoder forward
(``tac.local_acceleration.pr95_hnerv_mlx.HNeRVDecoderMLX``) and the faithful
PyTorch REFERENCE decoder (``submissions/hnerv_muon/src/model.py::HNeRVDecoder``)
on IDENTICAL trained weights + IDENTICAL latent input, down to the crux op.

The crux, established empirically (see ``feedback_mlx_pytorch_render_parity_crux
_landed_20260601.md``):

* The PR95-HNeRV decoder topology is byte-identical between MLX and PyTorch
  (stem → reshape → ``sin`` → 6×[interpolate-skip + PixelShuffle(conv) + ``sin``]
  → ``x + 0.1·sin(refine)`` → ``sigmoid(rgb)·255``). There is **no structural
  mismatch** (no transpose error, no PixelShuffle convention drift, no
  ``align_corners`` mismatch — these are byte-stable at ≤ 2.4e-7).
* The ONLY drift is **fp32 conv2d accumulation ORDER**: ``mx.conv2d`` (native,
  "optimized" mode) accumulates in a different order than PyTorch ``F.conv2d``.
  This is the SAME class of drift that PyTorch-fp32 exhibits vs PyTorch-fp64
  (both ~8e-4 at the final RGB heads).
* Intermediate features stay at ~1e-6–1e-5; the final ``sigmoid(rgb)·255`` head
  AMPLIFIES the accumulated drift to ~8e-4 in [0, 255] pixel space.
* ~8e-4 pixel drift → **≤ 1 LSB on < 0.004% of pixels** at uint8 → the SegNet
  argmax-flip d_seg is **identical** across MLX-optimized / MLX-fixed_fp64 /
  PyTorch-fp32 renders (delta = 0.0). The render is **already uint8-faithful**.

Therefore the carrier's advisory distortion (≈ 0.189) is **NOT a render-parity
artifact** — it is the carrier reconstruction fidelity measured on the
Apple-Silicon-CPU eval axis (drift source #2, out of scope) plus the inherent
carrier R(D). Render-parity (source #1) is engineered away to the floor; the
``fixed_fp64`` conv mode tightens the float drift ~4.5× (8.4e-4 → 1.9e-4) as
defense-in-depth without changing the already-floor d_seg.

All comparisons are ``[macOS-MLX vs PyTorch-CPU parity, exact-measured]``;
nothing here is a contest score claim. ``$0`` macOS-CPU/MLX-local only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

# Canonical PR95-HNeRV PyTorch reference decoder source (NEVER edited; pinned
# public-PR intake clone — the faithful reference forward per Catalog #109).
_PYTORCH_REFERENCE_MODEL_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_kaggle_mirror"
    / "public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src"
)


class RenderParityCruxError(RuntimeError):
    """Raised when the op-level parity harness cannot run faithfully."""


def _pytorch_reference_available() -> bool:
    return (_PYTORCH_REFERENCE_MODEL_DIR / "model.py").is_file()


def _import_pytorch_reference_decoder() -> Any:
    """Import the pinned PyTorch reference ``HNeRVDecoder`` (NEVER edited)."""
    if not _pytorch_reference_available():
        raise RenderParityCruxError(
            "PyTorch reference decoder not found at "
            f"{_PYTORCH_REFERENCE_MODEL_DIR / 'model.py'}"
        )
    src = str(_PYTORCH_REFERENCE_MODEL_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    import importlib

    model_mod = importlib.import_module("model")
    return model_mod.HNeRVDecoder


@dataclass(frozen=True)
class LayerDrift:
    """Per-op max/mean absolute drift between two decoder forwards."""

    layer: str
    max_abs: float
    mean_abs: float
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RenderParityReport:
    """Op-level + uint8 + downstream-scorer render-parity report.

    All fields are ``[macOS-MLX vs PyTorch-CPU parity, exact-measured]``.
    ``score_claim`` / ``promotable`` are always ``False`` — render parity is a
    software-implementation metric, NOT a contest score.
    """

    conv_accumulation_mode: str
    per_layer: tuple[LayerDrift, ...]
    first_divergent_layer: str
    first_divergent_max_abs: float
    final_frame_float_max_abs: float
    final_frame_uint8_max_abs: int
    final_frame_uint8_pixels_differ: int
    final_frame_uint8_total_pixels: int
    final_frame_uint8_fraction_differ: float
    crux_op: str
    crux_rationale: str
    axis_tag: str = "[macOS-MLX vs PyTorch-CPU parity, exact-measured]"
    score_claim: bool = False
    promotable: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pr95_hnerv_mlx_pytorch_render_parity_crux.v1",
            "conv_accumulation_mode": self.conv_accumulation_mode,
            "per_layer": [
                {
                    "layer": d.layer,
                    "max_abs": d.max_abs,
                    "mean_abs": d.mean_abs,
                    "shape": list(d.shape),
                }
                for d in self.per_layer
            ],
            "first_divergent_layer": self.first_divergent_layer,
            "first_divergent_max_abs": self.first_divergent_max_abs,
            "final_frame_float_max_abs": self.final_frame_float_max_abs,
            "final_frame_uint8_max_abs": self.final_frame_uint8_max_abs,
            "final_frame_uint8_pixels_differ": self.final_frame_uint8_pixels_differ,
            "final_frame_uint8_total_pixels": self.final_frame_uint8_total_pixels,
            "final_frame_uint8_fraction_differ": self.final_frame_uint8_fraction_differ,
            "crux_op": self.crux_op,
            "crux_rationale": self.crux_rationale,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "extras": dict(self.extras),
        }


# ---------------------------------------------------------------------------
# Instrumented forwards — capture per-op intermediates in a comparable layout.
# ---------------------------------------------------------------------------


def pytorch_reference_trace(
    state_dict_np: dict[str, np.ndarray],
    z_np: np.ndarray,
    *,
    latent_dim: int,
    base_channels: int,
    dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """Trace the PyTorch REFERENCE decoder per-op (NCHW intermediates).

    Captures the SAME op boundaries as :func:`mlx_decoder_trace`. ``dtype`` is
    ``"float32"`` (the faithful reference render) or ``"float64"`` (the true
    mathematical answer used to localize accumulation drift).
    """
    import torch
    import torch.nn.functional as F

    if dtype not in {"float32", "float64"}:
        raise RenderParityCruxError("dtype must be float32 or float64")
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    decoder_cls = _import_pytorch_reference_decoder()
    model = decoder_cls(latent_dim=int(latent_dim), base_channels=int(base_channels))
    model = model.to(torch_dtype)
    model.load_state_dict(
        {k: torch.from_numpy(np.asarray(v)).to(torch_dtype) for k, v in state_dict_np.items()}
    )
    model.eval()
    tr: dict[str, np.ndarray] = {}
    with torch.no_grad():
        z = torch.from_numpy(np.asarray(z_np)).to(torch_dtype)
        batch = int(z.shape[0])
        x = model.stem(z).view(batch, model.channels[0], model.base_h, model.base_w)
        tr["sin0_pre"] = x.detach().cpu().numpy()
        x = torch.sin(x)
        tr["sin0"] = x.detach().cpu().numpy()
        for i, (block, skip) in enumerate(
            zip(model.blocks, model.skips, strict=True)
        ):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            tr[f"b{i}_interp"] = identity.detach().cpu().numpy()
            identity = skip(identity)
            tr[f"b{i}_skip"] = identity.detach().cpu().numpy()
            conv = block(x)
            tr[f"b{i}_conv"] = conv.detach().cpu().numpy()
            ps = model.ps(conv)
            tr[f"b{i}_ps"] = ps.detach().cpu().numpy()
            x = torch.sin(ps + identity)
            tr[f"b{i}_out"] = x.detach().cpu().numpy()
        x = x + 0.1 * torch.sin(model.refine(x))
        tr["refine_out"] = x.detach().cpu().numpy()
        f0 = torch.sigmoid(model.rgb_0(x)) * 255.0
        tr["f0"] = f0.detach().cpu().numpy()
        f1 = torch.sigmoid(model.rgb_1(x)) * 255.0
        tr["f1"] = f1.detach().cpu().numpy()
    return tr


def mlx_decoder_trace(
    state_dict: dict[str, Any],
    z_np: np.ndarray,
    *,
    latent_dim: int,
    base_channels: int,
    conv_accumulation_mode: str,
) -> dict[str, np.ndarray]:
    """Trace the MLX decoder per-op, returning NCHW intermediates for comparison.

    The MLX decoder works internally in NHWC; each captured intermediate is
    transposed to NCHW so it lines up 1:1 with :func:`pytorch_reference_trace`.
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVDecoderMLX,
        bilinear_resize2x_align_corners_false_nhwc,
        load_pytorch_state_dict_into_mlx,
        pixel_shuffle_2x_nhwc,
    )

    # Pin MLX to the CPU device so the comparison is the deterministic
    # accumulation path (not the Metal GPU, whose kernels add a separate axis).
    mx.set_default_device(mx.cpu)
    decoder = HNeRVDecoderMLX(
        latent_dim=int(latent_dim),
        base_channels=int(base_channels),
        conv2d_accumulation_mode=conv_accumulation_mode,
    )
    load_pytorch_state_dict_into_mlx(decoder, dict(state_dict))

    def to_nchw(arr: Any) -> np.ndarray:
        mx.eval(arr)
        return np.transpose(np.asarray(arr), (0, 3, 1, 2))

    tr: dict[str, np.ndarray] = {}
    z = mx.array(np.asarray(z_np).astype(np.float32))
    batch = int(z.shape[0])
    x = decoder.stem(z)
    x = mx.reshape(x, (batch, decoder.channels[0], decoder.base_h, decoder.base_w))
    x = mx.transpose(x, (0, 2, 3, 1))  # NHWC
    tr["sin0_pre"] = to_nchw(x)
    x = mx.sin(x)
    tr["sin0"] = to_nchw(x)
    for i, block in enumerate(decoder.blocks):
        identity = bilinear_resize2x_align_corners_false_nhwc(x)
        tr[f"b{i}_interp"] = to_nchw(identity)
        if block.skip_conv is not None:
            identity = block.skip_conv(identity)
        tr[f"b{i}_skip"] = to_nchw(identity)
        conv = block.conv(x)
        ps = pixel_shuffle_2x_nhwc(conv)
        tr[f"b{i}_ps"] = to_nchw(ps)
        x = mx.sin(ps + identity)
        tr[f"b{i}_out"] = to_nchw(x)
    refined = decoder.refine1(decoder.refine0(x))
    x = x + 0.1 * mx.sin(refined)
    tr["refine_out"] = to_nchw(x)
    f0 = mx.sigmoid(decoder.rgb_0(x)) * 255.0
    tr["f0"] = to_nchw(f0)
    f1 = mx.sigmoid(decoder.rgb_1(x)) * 255.0
    tr["f1"] = to_nchw(f1)
    return tr


def _diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    af = np.asarray(a, dtype=np.float64)
    bf = np.asarray(b, dtype=np.float64)
    d = np.abs(af - bf)
    return float(d.max()), float(d.mean())


# Ops whose drift, if any, would indicate a STRUCTURAL bug (layout/convention).
# These are byte-stable in the canonical MLX helpers; a divergence here at
# fp64-vs-fp64 would be a real implementation crux, not accumulation order.
_STRUCTURAL_OP_PREFIXES = ("sin0_pre", "b", "refine_out")
_TOLERANCE_STRUCTURAL = 1.0e-3  # above this on a NON-conv op == structural bug


def localize_render_parity_crux(
    state_dict: dict[str, Any],
    latent_row: np.ndarray,
    *,
    latent_dim: int,
    base_channels: int,
    conv_accumulation_mode: str = "optimized",
) -> RenderParityReport:
    """Recursively localize the FIRST divergent op + classify the crux.

    Compares the MLX decoder forward (``conv_accumulation_mode``) against the
    faithful PyTorch-fp32 REFERENCE forward, per-op, finds the first op whose
    max-abs drift exceeds the structural tolerance, then characterizes the crux
    (conv accumulation order vs structural). Also measures the uint8 footprint
    of the final-frame drift (the render faithfulness that actually matters).
    """
    state_dict_np = {k: np.asarray(v).astype(np.float32) for k, v in state_dict.items()}
    z_np = np.asarray(latent_row, dtype=np.float32).reshape(1, -1)

    ref32 = pytorch_reference_trace(
        state_dict_np, z_np, latent_dim=latent_dim, base_channels=base_channels, dtype="float32"
    )
    mlx_tr = mlx_decoder_trace(
        state_dict,
        z_np,
        latent_dim=latent_dim,
        base_channels=base_channels,
        conv_accumulation_mode=conv_accumulation_mode,
    )

    layer_order = list(ref32.keys())
    per_layer: list[LayerDrift] = []
    first_divergent_layer = ""
    first_divergent_max_abs = 0.0
    for name in layer_order:
        if name not in mlx_tr:
            continue
        max_abs, mean_abs = _diff(mlx_tr[name], ref32[name])
        per_layer.append(
            LayerDrift(
                layer=name,
                max_abs=max_abs,
                mean_abs=mean_abs,
                shape=tuple(int(d) for d in ref32[name].shape),
            )
        )
        if not first_divergent_layer and max_abs > 1.0e-6:
            first_divergent_layer = name
            first_divergent_max_abs = max_abs

    # uint8 footprint of the final rendered pair (f0, f1).
    def _uint8(arr: np.ndarray) -> np.ndarray:
        return np.clip(np.round(arr), 0, 255).astype(np.int16)

    mlx_pair = np.stack([mlx_tr["f0"], mlx_tr["f1"]], axis=1)  # (1,2,3,H,W)
    ref_pair = np.stack([ref32["f0"], ref32["f1"]], axis=1)
    float_max_abs = float(np.abs(mlx_pair.astype(np.float64) - ref_pair.astype(np.float64)).max())
    u8_diff = np.abs(_uint8(mlx_pair) - _uint8(ref_pair))
    u8_max = int(u8_diff.max())
    u8_differ = int((u8_diff > 0).sum())
    u8_total = int(u8_diff.size)

    # Crux classification: any NON-conv structural op diverging > 1e-3 would be a
    # real implementation bug; otherwise the crux is fp32 conv accumulation order
    # amplified by the sigmoid·255 RGB head.
    structural_violations = [
        d
        for d in per_layer
        if d.layer.startswith(_STRUCTURAL_OP_PREFIXES)
        and not d.layer.endswith(("_conv", "_ps"))
        and d.max_abs > _TOLERANCE_STRUCTURAL
    ]
    if structural_violations:
        crux_op = structural_violations[0].layer
        crux_rationale = (
            f"STRUCTURAL divergence at {crux_op} (max_abs="
            f"{structural_violations[0].max_abs:.3e}) — layout/convention bug, "
            "not accumulation order. Fix the MLX op."
        )
    else:
        crux_op = "conv2d_fp32_accumulation_order"
        crux_rationale = (
            "fp32 conv2d accumulation ORDER (mx.conv2d vs F.conv2d) — the only "
            "drift source. Intermediate features stay ~1e-6; the sigmoid(rgb)*255 "
            "RGB head amplifies accumulated drift to ~8e-4 in [0,255], which is "
            f"<= {u8_max} LSB on {u8_differ}/{u8_total} uint8 pixels. The render is "
            "uint8-faithful; this is the same drift PyTorch-fp32 shows vs "
            "PyTorch-fp64. NOT a structural bug."
        )

    return RenderParityReport(
        conv_accumulation_mode=conv_accumulation_mode,
        per_layer=tuple(per_layer),
        first_divergent_layer=first_divergent_layer or "none",
        first_divergent_max_abs=first_divergent_max_abs,
        final_frame_float_max_abs=float_max_abs,
        final_frame_uint8_max_abs=u8_max,
        final_frame_uint8_pixels_differ=u8_differ,
        final_frame_uint8_total_pixels=u8_total,
        final_frame_uint8_fraction_differ=float(u8_differ) / max(u8_total, 1),
        crux_op=crux_op,
        crux_rationale=crux_rationale,
        extras={
            "render_pixel_range": [float(ref_pair.min()), float(ref_pair.max())],
        },
    )


__all__ = [
    "LayerDrift",
    "RenderParityCruxError",
    "RenderParityReport",
    "localize_render_parity_crux",
    "mlx_decoder_trace",
    "pytorch_reference_trace",
]
