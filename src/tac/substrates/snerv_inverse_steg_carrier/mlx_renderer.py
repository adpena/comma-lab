# SPDX-License-Identifier: MIT
"""MLX-native SNeRV renderer for shared score-aware long training.

This module is the differentiable training-side counterpart of the receiver
SNAR1 grammar in :mod:`tac.substrates.snerv_inverse_steg_carrier.carrier`:

* per-pair/per-frame/per-channel LF planes are trainable latents;
* shared HF detail predictors are trainable decoder weights;
* reconstruction uses the same Haar synthesis algebra as the NumPy receiver;
* model-size controls such as ``fc_dim``, ``patch_radius``, MFU scales, and HFR
  gain change the trainable feature bank instead of living only in metadata.

It is deliberately MLX-first for local training velocity, but all archive output
still flows through the NumPy-portable SNAR1 packer. The renderer is not a score
authority surface by itself; it is a real train-time carrier used before
receiver-closed export.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    _DETAIL_KEYS,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
    SnervModelSizeConfig,
    _kernel_storage_shape,
    dwt2_multilevel,
    fit_hf_decoder_least_squares,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    official_output2_fusion_mlx,
    official_output2_fusion_shape,
)

if TYPE_CHECKING:
    from tac.substrates.snerv_inverse_steg_carrier.official_mfu import OfficialSnervMfu

try:  # pragma: no cover - exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SNERV_MLX_RENDERER_SCHEMA = "snerv_mlx_score_aware_haar_renderer.v1"
SNERV_MLX_OFFICIAL_MFU_HFR_TUB_RENDERER_SCHEMA = (
    "snerv_mlx_official_mfu_hfr_tub_score_renderer.v1"
)


class SnervMlxRendererError(ValueError):
    """Raised when the MLX SNeRV renderer contract is violated."""


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError(
            "MLX is not available on this host; SNeRV MLX score-aware training "
            "requires Apple Silicon with the mlx package installed. Original "
            f"import error: {_MLX_IMPORT_ERROR!r}"
        )


class SnervMlxHaarScoreRenderer(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Differentiable SNeRV Haar LF/HF renderer for the shared MLX harness."""

    def __init__(
        self,
        *,
        lf_init: np.ndarray,
        decoder: HfGenerationDecoder,
        output_hw: tuple[int, int],
        model_size: SnervModelSizeConfig | None = None,
        wavelet: str = "haar",
    ) -> None:
        _require_mlx()
        super().__init__()
        wavelet_key = str(wavelet).strip().lower()
        if wavelet_key not in {"haar", "db1"}:
            raise SnervMlxRendererError(
                "SnervMlxHaarScoreRenderer is the NumPy-portable Haar path; "
                f"got wavelet={wavelet!r}"
            )
        lf = np.asarray(lf_init, dtype=np.float32)
        if lf.ndim != 5 or lf.shape[1] != 2 or lf.shape[2] != 3:
            raise SnervMlxRendererError(
                "lf_init must be shaped (pairs, 2, 3, H_lf, W_lf); got "
                f"{tuple(lf.shape)}"
            )
        if not np.isfinite(lf).all():
            raise SnervMlxRendererError("lf_init contains non-finite values")
        self.schema = SNERV_MLX_RENDERER_SCHEMA
        self.levels = int(decoder.levels)
        if self.levels < 1:
            raise SnervMlxRendererError("levels must be >= 1")
        self.wavelet = "haar"
        self.num_pairs = int(lf.shape[0])
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.model_size = model_size or decoder.model_size
        if tuple(_kernel_storage_shape(self.model_size)) != tuple(
            np.asarray(next(iter(decoder.kernels[0].values()))).shape
        ):
            raise SnervMlxRendererError(
                "decoder kernel shape does not match model_size feature_count"
            )

        self.latents_lf_planes = mx.array(lf, dtype=mx.float32)  # type: ignore[union-attr]
        self.decoder_kernels: list[dict[str, Any]] = []
        for lvl in range(self.levels):
            level_row: dict[str, Any] = {}
            for subband in _DETAIL_KEYS:
                level_row[subband] = mx.array(  # type: ignore[union-attr]
                    np.asarray(decoder.kernels[lvl][subband], dtype=np.float32).reshape(-1),
                    dtype=mx.float32,  # type: ignore[union-attr]
                )
            self.decoder_kernels.append(level_row)

    @classmethod
    def from_numpy_pairs(
        cls,
        pairs_nchw255: np.ndarray,
        *,
        levels: int,
        wavelet: str = "haar",
        model_size: SnervModelSizeConfig | None = None,
    ) -> SnervMlxHaarScoreRenderer:
        """Initialize LF latents and decoder weights from real pair pixels."""

        model_size = model_size or SnervModelSizeConfig()
        pairs = np.asarray(pairs_nchw255, dtype=np.float32)
        if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[2] != 3:
            raise SnervMlxRendererError(
                "pairs_nchw255 must be shaped (pairs, 2, 3, H, W); got "
                f"{tuple(pairs.shape)}"
            )
        if not np.isfinite(pairs).all():
            raise SnervMlxRendererError("pairs_nchw255 contains non-finite values")
        if str(wavelet).strip().lower() not in {"haar", "db1"}:
            raise SnervMlxRendererError(
                "MLX score-aware SNeRV long training currently supports the "
                f"receiver-safe Haar path only; got {wavelet!r}"
            )
        n_pairs, _two, channels, h, w = (int(v) for v in pairs.shape)
        if h % (1 << int(levels)) or w % (1 << int(levels)):
            raise SnervMlxRendererError(
                f"H/W must be divisible by 2**levels for MLX Haar training; "
                f"got {(h, w)} levels={levels}"
            )
        pyramids = []
        lf_planes = np.empty(
            (n_pairs, 2, channels, h >> int(levels), w >> int(levels)),
            dtype=np.float32,
        )
        for pair_idx in range(n_pairs):
            for frame_idx in range(2):
                for channel_idx in range(channels):
                    pyr = dwt2_multilevel(
                        pairs[pair_idx, frame_idx, channel_idx],
                        levels=int(levels),
                        wavelet="haar",
                    )
                    pyramids.append(pyr)
                    lf_planes[pair_idx, frame_idx, channel_idx] = np.asarray(
                        pyr.lf,
                        dtype=np.float32,
                    )
        decoder = fit_hf_decoder_least_squares(
            pyramids,
            levels=int(levels),
            model_size=model_size,
            temporal_group_count=channels,
        )
        return cls(
            lf_init=lf_planes,
            decoder=decoder,
            output_hw=(h, w),
            model_size=model_size,
            wavelet="haar",
        )

    def reconstruct_pair(self, pair_indices: Any) -> tuple[Any, Any]:
        """Return ``(rgb_0, rgb_1)`` NCHW in ``[0, 1]`` for the shared harness."""

        _require_mlx()
        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim == 0:
            idx = idx.reshape((1,))
        lf = mx.take(self.latents_lf_planes, idx, axis=0)  # type: ignore[union-attr]
        flat = lf.reshape((-1, int(lf.shape[-2]), int(lf.shape[-1])))
        frame_offsets = mx.arange(2, dtype=mx.int32).reshape((1, 2, 1))  # type: ignore[union-attr]
        channel_offsets = mx.arange(3, dtype=mx.int32).reshape((1, 1, 3))  # type: ignore[union-attr]
        global_flat_indices = (
            idx.reshape((-1, 1, 1)) * 6 + frame_offsets * 3 + channel_offsets
        ).reshape((-1,))
        recon = self._reconstruct_flat_channels(
            flat,
            global_flat_indices=global_flat_indices,
        )
        b = int(idx.shape[0])
        h, w = self.output_hw
        pair = recon.reshape((b, 2, 3, h, w))
        pair01 = mx.clip(pair / 255.0, 0.0, 1.0)  # type: ignore[union-attr]
        return pair01[:, 0], pair01[:, 1]

    def __call__(self, pair_indices: Any) -> Any:
        """Return ``(B, 2, 3, H, W)`` in ``[0, 255]`` for export helpers."""

        rgb0, rgb1 = self.reconstruct_pair(pair_indices)
        return mx.stack([rgb0, rgb1], axis=1) * 255.0  # type: ignore[union-attr]

    def render_pairs_nchw255(
        self,
        *,
        pair_indices: Sequence[int] | None = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Render selected rows as a NumPy ``(pairs,2,3,H,W)`` uint surface."""

        _require_mlx()
        indices = (
            tuple(range(self.num_pairs))
            if pair_indices is None
            else tuple(int(value) for value in pair_indices)
        )
        if not indices:
            raise SnervMlxRendererError("pair_indices must not be empty")
        chunks: list[np.ndarray] = []
        for start in range(0, len(indices), max(1, int(batch_size))):
            chunk = indices[start : start + max(1, int(batch_size))]
            arr = self(mx.array(chunk, dtype=mx.int32))  # type: ignore[union-attr]
            mx.eval(arr)  # type: ignore[union-attr]
            chunks.append(np.asarray(arr, dtype=np.float32))
        out = np.concatenate(chunks, axis=0)
        return np.clip(out, 0.0, 255.0).astype(np.float32, copy=False)

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Return a NumPy-portable train-time checkpoint dictionary."""

        _require_mlx()
        out: dict[str, np.ndarray] = {
            "latents_lf_planes": np.asarray(
                self.latents_lf_planes,
                dtype=np.float32,
            ).copy()
        }
        for lvl, level_row in enumerate(self.decoder_kernels):
            for subband in _DETAIL_KEYS:
                out[f"decoder_kernels.{lvl}.{subband}"] = np.asarray(
                    level_row[subband],
                    dtype=np.float32,
                ).copy()
        return out

    def import_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore a state emitted by :meth:`export_state_dict` exactly."""

        _require_mlx()
        expected = set(self.export_state_dict())
        observed = set(state)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise SnervMlxRendererError(
                "SNeRV MLX renderer state key mismatch; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        lf = np.asarray(state["latents_lf_planes"], dtype=np.float32)
        if tuple(lf.shape) != tuple(self.latents_lf_planes.shape):
            raise SnervMlxRendererError(
                "latents_lf_planes shape mismatch; "
                f"state={tuple(lf.shape)} model={tuple(self.latents_lf_planes.shape)}"
            )
        self.latents_lf_planes = mx.array(lf, dtype=mx.float32)  # type: ignore[union-attr]
        for lvl, level_row in enumerate(self.decoder_kernels):
            for subband in _DETAIL_KEYS:
                key = f"decoder_kernels.{lvl}.{subband}"
                arr = np.asarray(state[key], dtype=np.float32)
                if tuple(arr.shape) != tuple(level_row[subband].shape):
                    raise SnervMlxRendererError(
                        f"{key} shape mismatch; "
                        f"state={tuple(arr.shape)} model={tuple(level_row[subband].shape)}"
                    )
                level_row[subband] = mx.array(arr, dtype=mx.float32)  # type: ignore[union-attr]

    def metadata(self) -> dict[str, Any]:
        """Return non-authority renderer metadata for train/export reports."""

        return {
            "schema": SNERV_MLX_RENDERER_SCHEMA,
            "levels": int(self.levels),
            "wavelet": self.wavelet,
            "num_pairs": int(self.num_pairs),
            "output_hw": [int(v) for v in self.output_hw],
            "lf_shape": [int(v) for v in self.latents_lf_planes.shape[-2:]],
            "model_size": self.model_size.as_jsonable(),
            "trainable_parameter_count": int(self.num_parameters()),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def num_parameters(self) -> int:
        _require_mlx()
        total = 0
        for _name, arr in tree_flatten(self.parameters()):  # type: ignore[operator]
            total += int(np.prod(tuple(arr.shape)))
        return total

    def _reconstruct_flat_channels(
        self,
        lf_flat: Any,
        *,
        global_flat_indices: Any | None = None,
    ) -> Any:
        approx = lf_flat
        for lvl in range(self.levels):
            temporal = None
            if int(self.model_size.temporal_context) > 0:
                if global_flat_indices is None:
                    raise SnervMlxRendererError(
                        "temporal_context>0 requires global LF flat indices"
                    )
                temporal = _temporal_context_features_mlx(
                    self.latents_lf_planes.reshape(
                        (-1, int(self.latents_lf_planes.shape[-2]), int(self.latents_lf_planes.shape[-1]))
                    ),
                    global_flat_indices=global_flat_indices,
                    target_hw=(int(approx.shape[1]), int(approx.shape[2])),
                    model_size=self.model_size,
                    group_count=3,
                )
            features = _decoder_features_mlx(
                approx,
                self.model_size,
                temporal_features=temporal,
            )
            kernels = self.decoder_kernels[lvl]
            lh = _linear_detail(features, kernels["LH"])
            hl = _linear_detail(features, kernels["HL"])
            hh = _linear_detail(features, kernels["HH"])
            if float(self.model_size.hfr_gain) > 0.0:
                lh, hl, hh = _hfr_restore_mlx(
                    approx,
                    (lh, hl, hh),
                    gain=float(self.model_size.hfr_gain),
                )
            approx = _haar_idwt2_level_mlx(approx, lh, hl, hh)
        return approx


def export_snerv_mlx_renderer_state_dict(
    model: SnervMlxHaarScoreRenderer,
    path: str | Path,
) -> None:
    """Write a portable ``.npz`` state snapshot for explicit bridge tests."""

    arrays = model.export_state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target.with_suffix(target.suffix + ".npz"), **arrays)


def _linear_detail(features: Any, kernel: Any) -> Any:
    return mx.sum(features * kernel.reshape((1, 1, 1, -1)), axis=-1)  # type: ignore[union-attr]


def _decoder_features_mlx(
    field: Any,
    model_size: SnervModelSizeConfig,
    *,
    temporal_features: Any | None = None,
) -> Any:
    if model_size.adapter == SNERV_SPECTRA_PRESERVING_ADAPTER:
        context = _mfu_features_mlx(
            field,
            feature_count=int(model_size.fc_dim),
            patch_radius=int(model_size.patch_radius),
            scales=tuple(int(v) for v in model_size.mfu_scales),
        )
    else:
        context = _patch_features_mlx(
            field,
            patch_radius=int(model_size.patch_radius),
            feature_count=int(model_size.fc_dim),
        )
    coord = _coordinate_embedding_features_mlx(
        (int(field.shape[1]), int(field.shape[2])),
        int(model_size.emb_size),
        batch=int(field.shape[0]),
    )
    if int(model_size.temporal_context) > 0:
        if temporal_features is None:
            raise SnervMlxRendererError(
                "temporal_context>0 requires receiver-visible temporal features"
            )
        expected = 2 * int(model_size.temporal_context)
        if int(temporal_features.shape[-1]) != expected:
            raise SnervMlxRendererError(
                "temporal feature count mismatch; "
                f"got {int(temporal_features.shape[-1])}, expected {expected}"
            )
    else:
        temporal_features = mx.zeros(  # type: ignore[union-attr]
            (int(field.shape[0]), int(field.shape[1]), int(field.shape[2]), 0),
            dtype=field.dtype,
        )
    return mx.concatenate([context, coord, temporal_features], axis=-1)  # type: ignore[union-attr]


def _temporal_context_features_mlx(
    lf_flat_all: Any,
    *,
    global_flat_indices: Any,
    target_hw: tuple[int, int],
    model_size: SnervModelSizeConfig,
    group_count: int,
) -> Any:
    radius = int(model_size.temporal_context)
    if radius <= 0:
        return mx.zeros(  # type: ignore[union-attr]
            (int(global_flat_indices.shape[0]), int(target_hw[0]), int(target_hw[1]), 0),
            dtype=lf_flat_all.dtype,
        )
    groups = global_flat_indices % int(group_count)
    sequence_indices = global_flat_indices // int(group_count)
    max_sequence_index = (int(lf_flat_all.shape[0]) - 1) // int(group_count)
    center = mx.take(lf_flat_all, global_flat_indices, axis=0)  # type: ignore[union-attr]
    bank = []
    inv_two_sqrt2 = 1.0 / (2.0 * np.sqrt(2.0))
    mode = str(model_size.temporal_mode)
    for offset in range(1, radius + 1):
        prev_seq = mx.clip(sequence_indices - int(offset), 0, max_sequence_index)  # type: ignore[union-attr]
        next_seq = mx.clip(sequence_indices + int(offset), 0, max_sequence_index)  # type: ignore[union-attr]
        prev = mx.take(lf_flat_all, prev_seq * int(group_count) + groups, axis=0)  # type: ignore[union-attr]
        nxt = mx.take(lf_flat_all, next_seq * int(group_count) + groups, axis=0)  # type: ignore[union-attr]
        if mode == "delta":
            bank.append(_resize_nn_bhw_mlx(center - prev, target_hw))
            bank.append(_resize_nn_bhw_mlx(nxt - center, target_hw))
        elif mode == "official_haar_dwt1d_lowpass":
            bank.append(_resize_nn_bhw_mlx((center + prev) * inv_two_sqrt2, target_hw))
            bank.append(_resize_nn_bhw_mlx((center + nxt) * inv_two_sqrt2, target_hw))
        else:
            raise SnervMlxRendererError(
                f"unsupported temporal_mode {model_size.temporal_mode!r}"
            )
    return mx.stack(bank, axis=-1)  # type: ignore[union-attr]


def _patch_features_mlx(field: Any, *, patch_radius: int, feature_count: int) -> Any:
    radius = int(patch_radius)
    wanted = int(feature_count)
    if wanted < 1:
        raise SnervMlxRendererError("feature_count must be >= 1")
    if radius < 0 or radius > 3:
        raise SnervMlxRendererError("patch_radius must be in [0, 3]")
    if radius == 0:
        base = field[:, :, :, None]
    else:
        padded = _reflect_pad_bhw(field, radius)
        h, w = int(field.shape[1]), int(field.shape[2])
        feats = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                row = radius + di
                col = radius + dj
                feats.append(padded[:, row : row + h, col : col + w])
        base = mx.stack(feats, axis=-1)  # type: ignore[union-attr]
    if wanted <= int(base.shape[-1]):
        return base[:, :, :, :wanted]
    extras = _extra_context_features_mlx(field, wanted - int(base.shape[-1]))
    return mx.concatenate([base, extras], axis=-1)  # type: ignore[union-attr]


def _mfu_features_mlx(
    field: Any,
    *,
    feature_count: int,
    patch_radius: int,
    scales: tuple[int, ...],
) -> Any:
    bank = []
    for scale in scales:
        pooled = _box_pool_upsample_mlx(field, int(scale))
        bank.append(pooled)
        if int(scale) > 1:
            bank.append(field - pooled)
    gy, gx = _central_gradients_mlx(field)
    bank.extend((gy, gx, mx.sqrt(gy * gy + gx * gx)))  # type: ignore[union-attr]
    local = _patch_features_mlx(
        field,
        patch_radius=int(patch_radius),
        feature_count=(2 * int(patch_radius) + 1) ** 2,
    )
    for idx in range(int(local.shape[-1])):
        bank.append(local[:, :, :, idx])
    return _select_feature_bank_mlx(bank, int(feature_count))


def _select_feature_bank_mlx(bank: list[Any], feature_count: int) -> Any:
    if not bank:
        raise SnervMlxRendererError("feature bank must be non-empty")
    normalized = []
    for feature in bank:
        mean = mx.mean(feature, axis=(1, 2), keepdims=True)  # type: ignore[union-attr]
        centered = feature - mean
        var = mx.mean(centered * centered, axis=(1, 2), keepdims=True)  # type: ignore[union-attr]
        normalized.append(centered / mx.sqrt(var + 1.0e-6))  # type: ignore[union-attr]
    out = [normalized[i % len(normalized)] for i in range(int(feature_count))]
    return mx.stack(out, axis=-1)  # type: ignore[union-attr]


def _extra_context_features_mlx(field: Any, count: int) -> Any:
    if int(count) <= 0:
        return mx.zeros(  # type: ignore[union-attr]
            (int(field.shape[0]), int(field.shape[1]), int(field.shape[2]), 0),
            dtype=field.dtype,
        )
    mean = mx.mean(field, axis=(1, 2), keepdims=True)  # type: ignore[union-attr]
    centered = field - mean
    var = mx.mean(centered * centered, axis=(1, 2), keepdims=True)  # type: ignore[union-attr]
    norm = centered / mx.sqrt(var + 1.0e-6)  # type: ignore[union-attr]
    bank = (
        norm * norm,
        mx.tanh(norm),  # type: ignore[union-attr]
        mx.sin(norm),  # type: ignore[union-attr]
        mx.cos(norm),  # type: ignore[union-attr]
        norm * mx.roll(norm, shift=1, axis=1),  # type: ignore[union-attr]
        norm * mx.roll(norm, shift=1, axis=2),  # type: ignore[union-attr]
    )
    out = [bank[i % len(bank)] for i in range(int(count))]
    return mx.stack(out, axis=-1)  # type: ignore[union-attr]


def _coordinate_embedding_features_mlx(
    target_hw: tuple[int, int],
    emb_size: int,
    *,
    batch: int,
) -> Any:
    emb = int(emb_size)
    h, w = int(target_hw[0]), int(target_hw[1])
    if emb <= 0:
        return mx.zeros((int(batch), h, w, 0), dtype=mx.float32)  # type: ignore[union-attr]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = (2.0 * yy / max(h - 1, 1)) - 1.0
    x = (2.0 * xx / max(w - 1, 1)) - 1.0
    bank_np = (
        y,
        x,
        y * x,
        y * y,
        x * x,
        np.sin(np.pi * y),
        np.cos(np.pi * y),
        np.sin(np.pi * x),
        np.cos(np.pi * x),
    )
    stacked = np.stack([bank_np[i % len(bank_np)] for i in range(emb)], axis=-1)
    arr = mx.array(stacked, dtype=mx.float32)  # type: ignore[union-attr]
    return mx.broadcast_to(arr[None, :, :, :], (int(batch), h, w, emb))  # type: ignore[union-attr]


def _reflect_pad_bhw(field: Any, radius: int) -> Any:
    r = int(radius)
    if r <= 0:
        return field
    h, w = int(field.shape[1]), int(field.shape[2])
    if h <= r or w <= r:
        raise SnervMlxRendererError(
            f"cannot reflect-pad shape {(h, w)} by radius={r}"
        )
    top = field[:, 1 : r + 1, :][:, ::-1, :]
    bottom = field[:, h - r - 1 : h - 1, :][:, ::-1, :]
    padded = mx.concatenate([top, field, bottom], axis=1)  # type: ignore[union-attr]
    left = padded[:, :, 1 : r + 1][:, :, ::-1]
    right = padded[:, :, w - r - 1 : w - 1][:, :, ::-1]
    return mx.concatenate([left, padded, right], axis=2)  # type: ignore[union-attr]


def _resize_nn_bhw_mlx(field: Any, target_hw: tuple[int, int]) -> Any:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = int(field.shape[1]), int(field.shape[2])
    if (h, w) == (th, tw):
        return field
    row_idx = mx.array((np.arange(th) * h // th).clip(0, h - 1), dtype=mx.int32)  # type: ignore[union-attr]
    col_idx = mx.array((np.arange(tw) * w // tw).clip(0, w - 1), dtype=mx.int32)  # type: ignore[union-attr]
    return mx.take(mx.take(field, row_idx, axis=1), col_idx, axis=2)  # type: ignore[union-attr]


def _box_pool_upsample_mlx(field: Any, scale: int) -> Any:
    s = int(scale)
    if s <= 1:
        return field
    b, h, w = int(field.shape[0]), int(field.shape[1]), int(field.shape[2])
    ph = ((h + s - 1) // s) * s
    pw = ((w + s - 1) // s) * s
    padded = mx.pad(  # type: ignore[union-attr]
        field,
        ((0, 0), (0, ph - h), (0, pw - w)),
        mode="edge",
    )
    pooled = mx.mean(  # type: ignore[union-attr]
        padded.reshape((b, ph // s, s, pw // s, s)),
        axis=(2, 4),
    )
    up = mx.repeat(mx.repeat(pooled, s, axis=1), s, axis=2)  # type: ignore[union-attr]
    return up[:, :h, :w]


def _central_gradients_mlx(field: Any) -> tuple[Any, Any]:
    gy = 0.5 * (
        mx.roll(field, shift=-1, axis=1) - mx.roll(field, shift=1, axis=1)  # type: ignore[union-attr]
    )
    gx = 0.5 * (
        mx.roll(field, shift=-1, axis=2) - mx.roll(field, shift=1, axis=2)  # type: ignore[union-attr]
    )
    return gy, gx


def _hfr_restore_mlx(
    context: Any,
    details: tuple[Any, Any, Any],
    *,
    gain: float,
) -> tuple[Any, Any, Any]:
    edge = context - _box_pool_upsample_mlx(context, 3)
    gy, gx = _central_gradients_mlx(edge)
    bases = (gy, gx, 0.5 * (gy + gx))
    out = []
    for detail, basis in zip(details, bases, strict=True):
        mean = mx.mean(basis, axis=(1, 2), keepdims=True)  # type: ignore[union-attr]
        centered = basis - mean
        scale = mx.sqrt(  # type: ignore[union-attr]
            mx.mean(centered * centered, axis=(1, 2), keepdims=True) + 1.0e-6  # type: ignore[union-attr]
        )
        out.append(detail + float(gain) * centered / scale)
    return tuple(out)  # type: ignore[return-value]


def _haar_idwt2_level_mlx(ll: Any, lh: Any, hl: Any, hh: Any) -> Any:
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    row0 = mx.stack([a, b], axis=-1).reshape(  # type: ignore[union-attr]
        (int(ll.shape[0]), int(ll.shape[1]), int(ll.shape[2]) * 2)
    )
    row1 = mx.stack([c, d], axis=-1).reshape(  # type: ignore[union-attr]
        (int(ll.shape[0]), int(ll.shape[1]), int(ll.shape[2]) * 2)
    )
    return mx.stack([row0, row1], axis=2).reshape(  # type: ignore[union-attr]
        (int(ll.shape[0]), int(ll.shape[1]) * 2, int(ll.shape[2]) * 2)
    )


class SnervMlxOfficialMfuHfrTubScoreRenderer(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Train-time MLX wrapper around receiver-bound official MFU/HFR/TUB tensors.

    The receiver payload stores official MFU weights, official HFR weights, and
    official MFU input tensors.  This module makes those payload atoms trainable
    in the shared MLX score-aware harness instead of fitting them only after a
    local surrogate renderer has produced frames.
    """

    schema = SNERV_MLX_OFFICIAL_MFU_HFR_TUB_RENDERER_SCHEMA

    def __init__(
        self,
        *,
        mfu: OfficialSnervMfu,
        hfr_heads: OfficialHfrHeads,
        low: np.ndarray,
        skip_mid: np.ndarray,
        skip_high: np.ndarray,
        output_hw: tuple[int, int],
        model_size: SnervModelSizeConfig | None = None,
        skip_high_mode: str | None = None,
        tub_current: np.ndarray | None = None,
        tub_previous: np.ndarray | None = None,
        tub_next_frame: np.ndarray | None = None,
        tub_temporal_encoder_concat: np.ndarray | None = None,
        tub_output2_raw: np.ndarray | None = None,
        tub_output2_fc_hw: tuple[int, int] | None = None,
    ) -> None:
        _require_mlx()
        super().__init__()
        low_np = np.asarray(low, dtype=np.float32)
        skip_mid_np = np.asarray(skip_mid, dtype=np.float32)
        skip_high_np = np.asarray(skip_high, dtype=np.float32)
        if low_np.ndim != 4 or skip_mid_np.ndim != 4 or skip_high_np.ndim != 4:
            raise SnervMlxRendererError(
                "official MFU/HFR renderer expects low/skip tensors in NCHW"
            )
        if int(skip_high_np.shape[0]) % 2:
            raise SnervMlxRendererError(
                "official MFU/HFR renderer stores pair-major frames; frame count "
                f"must be even, got {skip_high_np.shape[0]}"
            )
        if not (
            np.isfinite(low_np).all()
            and np.isfinite(skip_mid_np).all()
            and np.isfinite(skip_high_np).all()
        ):
            raise SnervMlxRendererError("official MFU/HFR renderer inputs contain non-finite values")
        self.mfu = mfu
        self.model_size = model_size or SnervModelSizeConfig(
            adapter="snerv_official_mfu_hfr_tub_numeric_primitives_v1"
        )
        requested_skip_high_mode = str(
            skip_high_mode or self.model_size.official_skip_high_mode
        ).strip().lower()
        if requested_skip_high_mode not in {
            "full",
            "shared_mean",
            "channel_mean",
            "scalar_mean",
        }:
            raise SnervMlxRendererError(
                "official skip_high_mode must be one of full, shared_mean, "
                "channel_mean, scalar_mean"
            )
        self.skip_high_mode = requested_skip_high_mode
        self.skip_high_full_shape = tuple(int(v) for v in skip_high_np.shape)
        self.num_pairs = int(skip_high_np.shape[0]) // 2
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.low = mx.array(low_np, dtype=mx.float32)  # type: ignore[union-attr]
        self.skip_mid = mx.array(skip_mid_np, dtype=mx.float32)  # type: ignore[union-attr]
        if self.skip_high_mode == "shared_mean":
            skip_high_np = np.mean(skip_high_np, axis=0, keepdims=True, dtype=np.float32)
        elif self.skip_high_mode == "channel_mean":
            skip_high_np = np.mean(
                skip_high_np,
                axis=(0, 2, 3),
                keepdims=True,
                dtype=np.float32,
            )
        elif self.skip_high_mode == "scalar_mean":
            skip_high_np = np.asarray(
                [[[[float(np.mean(skip_high_np, dtype=np.float32))]]]],
                dtype=np.float32,
            )
        self.skip_high = mx.array(skip_high_np, dtype=mx.float32)  # type: ignore[union-attr]
        self._hfr_head_names = ("lh", "hl", "hh")
        self._import_hfr_heads(hfr_heads)
        self._tub_current_np = _official_tub_frame_or_default(
            tub_current,
            skip_high_np,
            frame_index=min(1, int(skip_high_np.shape[0]) - 1),
        )
        self._tub_previous_np = _official_tub_frame_or_default(
            tub_previous,
            skip_high_np,
            frame_index=0,
        )
        self._tub_next_frame_np = _official_tub_frame_or_default(
            tub_next_frame,
            skip_high_np,
            frame_index=min(1, int(skip_high_np.shape[0]) - 1),
        )
        self.tub_temporal_encoder_concat = _official_tub_optional_payload_tensor_mlx(
            tub_temporal_encoder_concat,
            name="tub_temporal_encoder_concat",
        )
        self.tub_output2_raw = _official_tub_optional_payload_tensor_mlx(
            tub_output2_raw,
            name="tub_output2_raw",
        )
        self.tub_output2_fc_hw = _official_tub_output2_fc_hw_or_none(
            tub_output2_fc_hw,
            has_output2=(
                self.tub_temporal_encoder_concat is not None
                or self.tub_output2_raw is not None
            ),
        )
        self.tub_output2_fused_shape = _official_tub_output2_fused_shape_or_none(
            tub_temporal_encoder_concat,
            tub_output2_raw,
            fc_hw=self.tub_output2_fc_hw,
        )
        self.tub_output2_receiver_frame_shape = (
            int(self.skip_high_full_shape[0]),
            int(self.skip_high_full_shape[1]),
            int(self.output_hw[0]),
            int(self.output_hw[1]),
        )
        self.tub_output2_receiver_frame_bound = (
            self.tub_output2_fused_shape == self.tub_output2_receiver_frame_shape
        )

    def _import_hfr_heads(self, heads: OfficialHfrHeads) -> None:
        for name, head in zip(
            self._hfr_head_names,
            (heads.lh_head, heads.hl_head, heads.hh_head),
            strict=True,
        ):
            setattr(
                self,
                f"hfr_{name}_conv1_weight",
                mx.array(np.asarray(head.conv1.weight, dtype=np.float32), dtype=mx.float32),  # type: ignore[union-attr]
            )
            setattr(
                self,
                f"hfr_{name}_conv1_bias",
                mx.array(np.asarray(head.conv1.bias, dtype=np.float32), dtype=mx.float32),  # type: ignore[union-attr]
            )
            setattr(
                self,
                f"hfr_{name}_conv2_weight",
                mx.array(np.asarray(head.conv2.weight, dtype=np.float32), dtype=mx.float32),  # type: ignore[union-attr]
            )
            setattr(
                self,
                f"hfr_{name}_conv2_bias",
                mx.array(np.asarray(head.conv2.bias, dtype=np.float32), dtype=mx.float32),  # type: ignore[union-attr]
            )

    def reconstruct_pair(self, pair_indices: Any) -> tuple[Any, Any]:
        """Return ``(rgb_0, rgb_1)`` NCHW in ``[0, 1]`` for the shared harness."""

        _require_mlx()
        idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
        if idx.ndim == 0:
            idx = idx.reshape((1,))
        frame_offsets = mx.arange(2, dtype=mx.int32).reshape((1, 2))  # type: ignore[union-attr]
        frame_indices = (idx.reshape((-1, 1)) * 2 + frame_offsets).reshape((-1,))
        low = mx.take(self.low, frame_indices, axis=0)  # type: ignore[union-attr]
        skip_mid = mx.take(self.skip_mid, frame_indices, axis=0)  # type: ignore[union-attr]
        if self.skip_high_mode != "full":
            skip_high = mx.broadcast_to(  # type: ignore[union-attr]
                self.skip_high,
                (
                    int(frame_indices.shape[0]),
                    int(self.skip_high_full_shape[1]),
                    int(self.skip_high_full_shape[2]),
                    int(self.skip_high_full_shape[3]),
                ),
            )
        else:
            skip_high = mx.take(self.skip_high, frame_indices, axis=0)  # type: ignore[union-attr]
        mfu_out = self.mfu.forward_mlx(
            low,
            skip_mid,
            skip_high,
            accumulation_mode="optimized",
        )
        lh = self._hfr_head_forward("lh", mfu_out.pyr_out)
        hl = self._hfr_head_forward("hl", mfu_out.pyr_out)
        hh = self._hfr_head_forward("hh", mfu_out.pyr_out)
        recon = _haar_idwt2_level_mlx_nchw(mfu_out.pyr_out, lh, hl, hh)
        if self.tub_output2_receiver_frame_bound:
            recon = self._apply_tub_output2_residual(recon, frame_indices)
        b = int(idx.shape[0])
        h, w = self.output_hw
        pair = recon.reshape((b, 2, 3, h, w))
        pair01 = mx.clip(pair / 255.0, 0.0, 1.0)  # type: ignore[union-attr]
        return pair01[:, 0], pair01[:, 1]

    def __call__(self, pair_indices: Any) -> Any:
        rgb0, rgb1 = self.reconstruct_pair(pair_indices)
        return mx.stack([rgb0, rgb1], axis=1) * 255.0  # type: ignore[union-attr]

    def render_pairs_nchw255(
        self,
        *,
        pair_indices: Sequence[int] | None = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        _require_mlx()
        indices = (
            tuple(range(self.num_pairs))
            if pair_indices is None
            else tuple(int(value) for value in pair_indices)
        )
        if not indices:
            raise SnervMlxRendererError("pair_indices must not be empty")
        chunks: list[np.ndarray] = []
        for start in range(0, len(indices), max(1, int(batch_size))):
            chunk = indices[start : start + max(1, int(batch_size))]
            arr = self(mx.array(chunk, dtype=mx.int32))  # type: ignore[union-attr]
            mx.eval(arr)  # type: ignore[union-attr]
            chunks.append(np.asarray(arr, dtype=np.float32))
        return np.clip(np.concatenate(chunks, axis=0), 0.0, 255.0).astype(
            np.float32,
            copy=False,
        )

    def export_state_dict(self) -> dict[str, np.ndarray]:
        _require_mlx()
        out: dict[str, np.ndarray] = {
            "low": np.asarray(self.low, dtype=np.float32).copy(),
            "skip_mid": np.asarray(self.skip_mid, dtype=np.float32).copy(),
            "skip_high": np.asarray(self.skip_high, dtype=np.float32).copy(),
        }
        if self.tub_temporal_encoder_concat is not None:
            out["tub.temporal_encoder_concat"] = np.asarray(
                self.tub_temporal_encoder_concat,
                dtype=np.float32,
            ).copy()
        if self.tub_output2_raw is not None:
            out["tub.output2_raw"] = np.asarray(
                self.tub_output2_raw,
                dtype=np.float32,
            ).copy()
        for name in self._hfr_head_names:
            for layer in ("conv1", "conv2"):
                for field in ("weight", "bias"):
                    key = f"hfr.{name}.{layer}.{field}"
                    out[key] = np.asarray(
                        getattr(self, f"hfr_{name}_{layer}_{field}"),
                        dtype=np.float32,
                    ).copy()
        return out

    def import_state_dict(self, state: dict[str, np.ndarray]) -> None:
        _require_mlx()
        expected = set(self.export_state_dict())
        observed = set(state)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise SnervMlxRendererError(
                "official MFU/HFR renderer state key mismatch; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        for key, attr in (
            ("low", "low"),
            ("skip_mid", "skip_mid"),
            ("skip_high", "skip_high"),
        ):
            arr = np.asarray(state[key], dtype=np.float32)
            current = getattr(self, attr)
            if tuple(arr.shape) != tuple(current.shape):
                raise SnervMlxRendererError(
                    f"{key} shape mismatch; state={tuple(arr.shape)} "
                    f"model={tuple(current.shape)}"
                )
            setattr(self, attr, mx.array(arr, dtype=mx.float32))  # type: ignore[union-attr]
        for key, attr in (
            ("tub.temporal_encoder_concat", "tub_temporal_encoder_concat"),
            ("tub.output2_raw", "tub_output2_raw"),
        ):
            if key not in state:
                continue
            arr = np.asarray(state[key], dtype=np.float32)
            current = getattr(self, attr)
            if current is None:
                raise SnervMlxRendererError(f"{key} supplied but renderer was not initialized with it")
            if tuple(arr.shape) != tuple(current.shape):
                raise SnervMlxRendererError(
                    f"{key} shape mismatch; state={tuple(arr.shape)} "
                    f"model={tuple(current.shape)}"
                )
            setattr(self, attr, mx.array(arr, dtype=mx.float32))  # type: ignore[union-attr]
        for name in self._hfr_head_names:
            for layer in ("conv1", "conv2"):
                for field in ("weight", "bias"):
                    key = f"hfr.{name}.{layer}.{field}"
                    attr = f"hfr_{name}_{layer}_{field}"
                    arr = np.asarray(state[key], dtype=np.float32)
                    current = getattr(self, attr)
                    if tuple(arr.shape) != tuple(current.shape):
                        raise SnervMlxRendererError(
                            f"{key} shape mismatch; state={tuple(arr.shape)} "
                            f"model={tuple(current.shape)}"
                        )
                    setattr(self, attr, mx.array(arr, dtype=mx.float32))  # type: ignore[union-attr]

    def export_official_components(self) -> dict[str, Any]:
        """Return receiver encoder inputs from the current trained MLX state."""

        state = self.export_state_dict()
        skip_high = state["skip_high"].astype(np.float64)
        components = {
            "mfu": self.mfu,
            "hfr_heads": OfficialHfrHeads(
                lh_head=self._export_hfr_head("lh", state),
                hl_head=self._export_hfr_head("hl", state),
                hh_head=self._export_hfr_head("hh", state),
            ),
            "low": state["low"].astype(np.float64),
            "skip_mid": state["skip_mid"].astype(np.float64),
            "skip_high": skip_high,
            "skip_high_mode": self.skip_high_mode,
            "skip_high_full_shape": tuple(int(v) for v in self.skip_high_full_shape),
            "skip_high_export_storage_shape": tuple(int(v) for v in skip_high.shape),
            "skip_high_export_is_compact_train_state": self.skip_high_mode != "full",
            "tub_current": self._tub_current_np.astype(np.float64),
            "tub_previous": self._tub_previous_np.astype(np.float64),
            "tub_next_frame": self._tub_next_frame_np.astype(np.float64),
        }
        if "tub.temporal_encoder_concat" in state:
            components["tub_temporal_encoder_concat"] = state[
                "tub.temporal_encoder_concat"
            ].astype(np.float64)
        if "tub.output2_raw" in state:
            components["tub_output2_raw"] = state["tub.output2_raw"].astype(np.float64)
        if self.tub_output2_fc_hw is not None:
            components["fc_hw"] = tuple(int(v) for v in self.tub_output2_fc_hw)
        if self.tub_temporal_encoder_concat is not None:
            components["temporal_encoder_output_shape"] = tuple(
                int(v) for v in self.tub_temporal_encoder_concat.shape
            )
        if self.tub_output2_raw is not None:
            components["output2_decoder_output_shape"] = tuple(
                int(v) for v in self.tub_output2_raw.shape
            )
        return components

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "num_pairs": int(self.num_pairs),
            "output_hw": [int(v) for v in self.output_hw],
            "mfu_spec": {
                "low_channels": int(self.mfu.spec.low_channels),
                "mid_channels": int(self.mfu.spec.mid_channels),
                "high_channels": int(self.mfu.spec.high_channels),
                "mid_stride": int(self.mfu.spec.mid_stride),
                "high_stride": int(self.mfu.spec.high_stride),
                "num_blocks": int(self.mfu.spec.num_blocks),
            },
            "model_size": self.model_size.as_jsonable(),
            "official_skip_high_mode": self.skip_high_mode,
            "skip_high_full_shape": [int(v) for v in self.skip_high_full_shape],
            "trainable_parameter_count": int(self.num_parameters()),
            "trainable_payload_atoms": [
                "inputs.mfu.low",
                "inputs.mfu.skip_mid",
                "inputs.mfu.skip_high",
                "hfr.lh",
                "hfr.hl",
                "hfr.hh",
            ],
            "receiver_export_payload_atoms": [
                "inputs.mfu.low",
                "inputs.mfu.skip_mid",
                "inputs.mfu.skip_high",
                "hfr.lh",
                "hfr.hl",
                "hfr.hh",
                *(
                    ["tub.temporal_encoder_concat", "tub.output2_raw"]
                    if self.tub_temporal_encoder_concat is not None
                    and self.tub_output2_raw is not None
                    else []
                ),
            ],
            "official_tub_output2_payload_export_bound": bool(
                self.tub_temporal_encoder_concat is not None
                and self.tub_output2_raw is not None
            ),
            "official_tub_output2_fc_hw": (
                [int(v) for v in self.tub_output2_fc_hw]
                if self.tub_output2_fc_hw is not None
                else None
            ),
            "official_tub_output2_fused_shape": (
                [int(v) for v in self.tub_output2_fused_shape]
                if self.tub_output2_fused_shape is not None
                else None
            ),
            "official_tub_output2_receiver_frame_shape": [
                int(v) for v in self.tub_output2_receiver_frame_shape
            ],
            "official_tub_output2_receiver_frame_bound": bool(
                self.tub_output2_receiver_frame_bound
            ),
            "official_tub_output2_payload_loss_coupled": bool(
                self.tub_output2_receiver_frame_bound
            ),
            "receiver_export_payload_schema": "snerv_decoder_payload.official_mfu_hfr_tub.v1",
            "source_forward_replay_authority": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def _apply_tub_output2_residual(self, recon: Any, frame_indices: Any) -> Any:
        if (
            self.tub_temporal_encoder_concat is None
            or self.tub_output2_raw is None
            or self.tub_output2_fc_hw is None
        ):
            raise SnervMlxRendererError(
                "official TUB output2 residual requested without a complete "
                "temporal/raw/fc_hw payload"
            )
        _decoder_input, output2_fused = official_output2_fusion_mlx(
            self.tub_temporal_encoder_concat,
            self.tub_output2_raw,
            fc_hw=self.tub_output2_fc_hw,
        )
        residual = mx.take(output2_fused, frame_indices, axis=0)  # type: ignore[union-attr]
        if tuple(int(v) for v in residual.shape) != tuple(int(v) for v in recon.shape):
            raise SnervMlxRendererError(
                "official TUB output2 residual shape must match selected "
                f"receiver frames; got {tuple(int(v) for v in residual.shape)}, "
                f"expected {tuple(int(v) for v in recon.shape)}"
            )
        return mx.clip(recon + residual, 0.0, 255.0)  # type: ignore[union-attr]

    def num_parameters(self) -> int:
        _require_mlx()
        total = 0
        for _name, arr in tree_flatten(self.parameters()):  # type: ignore[operator]
            total += int(np.prod(tuple(arr.shape)))
        return total

    def _hfr_head_forward(self, name: str, pyr_out: Any) -> Any:
        conv1 = _trainable_conv2d_nchw_mlx(
            pyr_out,
            getattr(self, f"hfr_{name}_conv1_weight"),
            getattr(self, f"hfr_{name}_conv1_bias"),
            padding=0,
        )
        hidden = mx.where(conv1 >= 0.0, conv1, 0.1 * conv1)  # type: ignore[union-attr]
        return _trainable_conv2d_nchw_mlx(
            hidden,
            getattr(self, f"hfr_{name}_conv2_weight"),
            getattr(self, f"hfr_{name}_conv2_bias"),
            padding=1,
        )

    def _export_hfr_head(
        self,
        name: str,
        state: dict[str, np.ndarray],
    ) -> OfficialHfrConvBlock:
        return OfficialHfrConvBlock(
            conv1=OfficialConv2dNchw(
                state[f"hfr.{name}.conv1.weight"].astype(np.float64),
                state[f"hfr.{name}.conv1.bias"].astype(np.float64),
                padding=0,
            ),
            conv2=OfficialConv2dNchw(
                state[f"hfr.{name}.conv2.weight"].astype(np.float64),
                state[f"hfr.{name}.conv2.bias"].astype(np.float64),
                padding=1,
            ),
        )


def _official_tub_frame_or_default(
    value: np.ndarray | None,
    skip_high: np.ndarray,
    *,
    frame_index: int,
) -> np.ndarray:
    if value is not None:
        arr = np.asarray(value, dtype=np.float32)
    else:
        arr = np.asarray(skip_high[int(frame_index)], dtype=np.float32)
    if arr.ndim != 3:
        raise SnervMlxRendererError(f"official TUB frame must be CHW, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise SnervMlxRendererError("official TUB frame contains non-finite values")
    return arr.copy()


def _official_tub_optional_payload_tensor_mlx(
    value: np.ndarray | None,
    *,
    name: str,
) -> Any | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 4:
        raise SnervMlxRendererError(f"{name} must be NCHW, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise SnervMlxRendererError(f"{name} contains non-finite values")
    return mx.array(arr.copy(), dtype=mx.float32)  # type: ignore[union-attr]


def _official_tub_output2_fc_hw_or_none(
    value: tuple[int, int] | None,
    *,
    has_output2: bool,
) -> tuple[int, int] | None:
    if not has_output2:
        return None
    if value is None:
        raise SnervMlxRendererError(
            "official TUB output2 payload requires tub_output2_fc_hw"
        )
    fc_hw = tuple(int(v) for v in value)
    if len(fc_hw) != 2 or any(v <= 0 for v in fc_hw):
        raise SnervMlxRendererError(
            f"official TUB output2 fc_hw must be positive HW, got {value!r}"
        )
    return fc_hw


def _official_tub_output2_fused_shape_or_none(
    temporal_encoder_concat: np.ndarray | None,
    output2_raw: np.ndarray | None,
    *,
    fc_hw: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    if temporal_encoder_concat is None and output2_raw is None:
        return None
    if temporal_encoder_concat is None or output2_raw is None or fc_hw is None:
        raise SnervMlxRendererError(
            "official TUB output2 payload requires temporal, raw, and fc_hw "
            "shape controls"
        )
    temporal = np.asarray(temporal_encoder_concat, dtype=np.float32)
    raw = np.asarray(output2_raw, dtype=np.float32)
    shape = official_output2_fusion_shape(
        tuple(int(v) for v in temporal.shape),
        fc_hw=tuple(int(v) for v in fc_hw),
        decoder_output_shape=tuple(int(v) for v in raw.shape),
    )
    if shape.fused_output2_shape is None:
        raise SnervMlxRendererError("official TUB output2 fused shape is missing")
    return tuple(int(v) for v in shape.fused_output2_shape)


def _trainable_conv2d_nchw_mlx(x: Any, weight_oihw: Any, bias: Any, *, padding: int) -> Any:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))  # type: ignore[union-attr]
    weight_ohwi = mx.transpose(weight_oihw, (0, 2, 3, 1))  # type: ignore[union-attr]
    out = mx.conv2d(  # type: ignore[union-attr]
        x_nhwc,
        weight_ohwi,
        stride=(1, 1),
        padding=(int(padding), int(padding)),
    )
    out = out + mx.reshape(bias, (1, 1, 1, int(bias.shape[0])))  # type: ignore[union-attr]
    return mx.transpose(out, (0, 3, 1, 2))  # type: ignore[union-attr]


def _haar_idwt2_level_mlx_nchw(ll: Any, lh: Any, hl: Any, hh: Any) -> Any:
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    n, c_count, h, w = (int(ll.shape[0]), int(ll.shape[1]), int(ll.shape[2]), int(ll.shape[3]))
    row0 = mx.stack([a, b], axis=-1).reshape((n, c_count, h, w * 2))  # type: ignore[union-attr]
    row1 = mx.stack([c, d], axis=-1).reshape((n, c_count, h, w * 2))  # type: ignore[union-attr]
    return mx.stack([row0, row1], axis=-2).reshape((n, c_count, h * 2, w * 2))  # type: ignore[union-attr]


__all__ = [
    "SNERV_MLX_OFFICIAL_MFU_HFR_TUB_RENDERER_SCHEMA",
    "SNERV_MLX_RENDERER_SCHEMA",
    "SnervMlxHaarScoreRenderer",
    "SnervMlxOfficialMfuHfrTubScoreRenderer",
    "SnervMlxRendererError",
    "export_snerv_mlx_renderer_state_dict",
]
