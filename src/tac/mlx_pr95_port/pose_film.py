# SPDX-License-Identifier: MIT
"""Store-6-pose-scalars + FiLM-inject — Quantizr's pose carrier, in MLX (task #84).

The pose term is the limiting term of the capstone (#78). The #74 distillation
verdict found d_pose was the binding constraint (~100x the teacher) and the #80
pose-crux proved the pose-sensitive subspace is rank-1 BUT the linear pose-null is
tangent-only (no cheap linear escape; pixel-RMSE<3 is a real floor *if you must
recover pose from pixels*). The #81 full-stack audit found the escape: **Quantizr
(PR#55, 88K params, d_pose 0.00051) never recovers pose from pixels — he STORES
the 6-dim GT pose explicitly (``pose.npy.br``, ~kilobytes) and FiLM-injects it into
the frame head.** The net is HANDED the pose it must render, so it never has to
reproduce the teacher's frames to +-5/255.

This module is the 1:1 MLX realization of that mechanism, built on the verified
clean base (``tac.local_acceleration.pr95_hnerv_mlx``):

- the 6-dim GT pose per pair is STORED as a trainable+quantizable latent
  (``stored_pose``), brotli-compressed at export (~6 floats x n_pairs);
- a small ``pose_mlp`` lifts the 6-d pose -> a conditioning embedding;
- a ``FiLMHead`` produces per-channel ``(gamma, beta)`` from that embedding and
  modulates the shared decoder feature map ``h -> gamma * h + beta`` BEFORE the
  frame RGB head, so PoseNet reads the correct pose off the rendered frame.

Per #80 the FiLM is applied to BOTH frame heads by default (frame0 carries ~20x
the pose debt; frame1 also carries pose) — the per-slot ``film_slots`` knob lets
the capstone restrict it to frame0 only if the seg term needs frame1's head un-
modulated. The decoder feature trunk + RGB heads are the verified bit-exact PR95
kernels; this module adds ONLY the pose-conditioning path (Quantizr's delta over
a plain HNeRV decoder).

Authority: ``[macOS-MLX research-signal]`` (MLX decoder) / ``[local CPU-torch
advisory]`` (frozen torch scorer). Non-promotable per Catalog #192; a contest
score requires ``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tac.local_acceleration.pr95_hnerv_mlx import HNeRVDecoderMLX

try:  # pragma: no cover - import guard
    import mlx.core as mx
    import mlx.nn as nn
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


POSE_DIM = 6


def _require_mlx() -> None:
    if mx is None or nn is None:  # pragma: no cover
        raise RuntimeError("tac.mlx_pr95_port.pose_film requires mlx.core + mlx.nn.")


class _PoseFiLMHead(nn.Module if nn is not None else object):  # type: ignore[misc]
    """6-d pose -> conditioning embedding -> per-channel (gamma, beta).

    The conditioning path Quantizr uses: ``pose_mlp(pose6)`` then a FiLM projection
    to ``2 * channels`` (gamma, beta). The decoder feature is modulated
    ``h -> (1 + gamma) * h + beta`` (residual-gamma init so an untrained head is
    the identity — the carrier starts as the clean PR95 decoder and the pose path
    *adds* conditioning, which keeps the loop stable, the #74 lesson).
    """

    def __init__(self, channels: int, *, hidden: int = 32, seed: int = 0) -> None:
        _require_mlx()
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        # ``nn.Linear`` seeds its own weights from MLX's global RNG; the ``seed``
        # arg is part of the public surface (per-slot determinism) even though the
        # FiLM projection is re-zeroed to identity below.
        mx.random.seed(seed)
        # pose_mlp: 6 -> hidden -> hidden (small, depthwise-separable in spirit).
        self.fc1 = nn.Linear(POSE_DIM, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # FiLM projection: hidden -> 2*channels (gamma, beta).
        self.film = nn.Linear(hidden, 2 * self.channels)
        # Residual-gamma + zero-beta init: the FiLM starts as identity so the
        # carrier == the clean PR95 decoder until the pose path is trained
        # (stability; the untrained head must NOT corrupt the render).
        self.film.weight = mx.zeros_like(self.film.weight)
        self.film.bias = mx.zeros_like(self.film.bias)

    def __call__(self, pose6: Any) -> tuple[Any, Any]:
        """Return per-channel ``(gamma, beta)`` shaped ``(B, 1, 1, channels)``."""
        h = mx.tanh(self.fc1(pose6))
        h = mx.tanh(self.fc2(h))
        gb = self.film(h)  # (B, 2*channels)
        gamma = gb[:, : self.channels]
        beta = gb[:, self.channels :]
        gamma = mx.reshape(gamma, (gamma.shape[0], 1, 1, self.channels))
        beta = mx.reshape(beta, (beta.shape[0], 1, 1, self.channels))
        return gamma, beta


class PoseFiLMDecoderMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """PR95 HNeRV decoder + Quantizr pose-FiLM conditioning on the frame head(s).

    Wraps the verified bit-exact :class:`HNeRVDecoderMLX`. The shared feature trunk
    ``features_nhwc(z)`` is unchanged (the verified kernel); the pose-FiLM path
    modulates the trunk feature per-frame-slot BEFORE the (verified) RGB head, so
    each frame's PoseNet readout matches the STORED pose it is handed.

    The pose is NOT recovered from pixels — it is supplied as ``pose6`` and the net
    only has to render a frame whose PoseNet reads back that pose (the #81/#80
    escape from the pixel-RMSE<3 capacity wall).

    Args:
        latent_dim / base_channels: the PR95 decoder topology (the carrier size).
        film_slots: which frame slots get pose-FiLM. ``(0, 1)`` = both (default;
            #80: frame0 carries ~20x pose, frame1 also carries pose). ``(0,)`` =
            frame0 only; ``()`` = no FiLM (the severed control).
        film_hidden: the pose-MLP hidden width (Quantizr is small).
    """

    def __init__(
        self,
        *,
        latent_dim: int = 28,
        base_channels: int = 36,
        film_slots: Sequence[int] = (0, 1),
        film_hidden: int = 32,
        seed: int = 0,
        output_layout: str = "n2chw",
    ) -> None:
        _require_mlx()
        super().__init__()
        self.decoder = HNeRVDecoderMLX(
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_layout=output_layout,
        )
        self.output_layout = output_layout
        self.film_slots = tuple(int(s) for s in film_slots)
        for s in self.film_slots:
            if s not in (0, 1):
                raise ValueError("film_slots entries must be 0 or 1")
        final_ch = self.decoder.channels[-1]
        self.final_ch = int(final_ch)
        # One FiLM head per modulated slot (frame0 / frame1 carry different pose
        # debt per #80, so they get independent conditioning).
        if 0 in self.film_slots:
            self.film0 = _PoseFiLMHead(final_ch, hidden=film_hidden, seed=seed)
        if 1 in self.film_slots:
            self.film1 = _PoseFiLMHead(final_ch, hidden=film_hidden, seed=seed + 1)

    def _apply_film(self, feat: Any, gamma: Any, beta: Any) -> Any:
        """Residual-gamma FiLM: ``h -> (1 + gamma) * h + beta``."""
        return (1.0 + gamma) * feat + beta

    def decode_pair_nhwc(self, z: Any, pose6: Any) -> Any:
        """Render the pair (NHWC) conditioning the modulated frame head(s) on pose6.

        ``z``: ``(B, latent_dim)`` per-pair latent. ``pose6``: ``(B, 6)`` stored pose.
        """
        feat = self.decoder.features_nhwc(z)  # the verified shared trunk
        # frame0 head, optionally pose-FiLM-modulated.
        if 0 in self.film_slots:
            g0, b0 = self.film0(pose6)
            feat0 = self._apply_film(feat, g0, b0)
        else:
            feat0 = feat
        f0 = mx.sigmoid(self.decoder.rgb_0(feat0)) * 255.0
        # frame1 head, optionally pose-FiLM-modulated.
        if 1 in self.film_slots:
            g1, b1 = self.film1(pose6)
            feat1 = self._apply_film(feat, g1, b1)
        else:
            feat1 = feat
        f1 = mx.sigmoid(self.decoder.rgb_1(feat1)) * 255.0
        return mx.stack([f0, f1], axis=1)

    def __call__(self, z: Any, pose6: Any) -> Any:
        pair = self.decode_pair_nhwc(z, pose6)
        if self.output_layout == "n2hwc":
            return pair
        return mx.transpose(pair, (0, 1, 4, 2, 3))

    def architecture_manifest(self) -> dict[str, Any]:
        man = dict(self.decoder.architecture_manifest())
        man.update(
            {
                "pose_film_slots": list(self.film_slots),
                "pose_film_hidden": (
                    self.film0.hidden
                    if 0 in self.film_slots
                    else (self.film1.hidden if 1 in self.film_slots else 0)
                ),
                "pose_dim": POSE_DIM,
                "pose_carrier": "store_explicit_plus_film",
                "source_mechanism": "Quantizr PR#55 JointFrameGenerator FiLM-on-pose",
            }
        )
        return man


class StoredPoseBundleMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Pose-FiLM decoder + per-pair latents + STORED 6-d pose (the carrier).

    The contest carrier the capstone (#78) trains: the per-pair latent (the frame
    content) AND the per-pair 6-d pose (the explicit pose answer). At export the
    pose is brotli-quantized to ~kilobytes (``stored_pose_bytes``). The bundle's
    ``__call__(indices)`` renders the selected pairs conditioning each frame head
    on that pair's STORED pose.

    The stored pose is initialized from the GT pose targets (the answer is handed
    in, not learned from scratch) but remains trainable so the loop can fine-tune
    it within the brotli budget.
    """

    def __init__(
        self,
        *,
        latent_count: int,
        pose_targets: Any,
        latent_dim: int = 28,
        base_channels: int = 36,
        film_slots: Sequence[int] = (0, 1),
        film_hidden: int = 32,
        seed: int = 0,
        output_layout: str = "n2chw",
    ) -> None:
        _require_mlx()
        super().__init__()
        key = mx.random.key(seed)
        self.latents = mx.random.normal((latent_count, latent_dim), key=key) * 0.1
        pose_np = np.asarray(pose_targets, dtype=np.float32)
        if pose_np.shape != (latent_count, POSE_DIM):
            raise ValueError(
                f"pose_targets must be ({latent_count}, {POSE_DIM}); got {pose_np.shape}"
            )
        # STORE the 6-d GT pose explicitly (Quantizr's pose.npy.br). Trainable so
        # the loop can fine-tune within the brotli budget, init = the GT answer.
        self.stored_pose = mx.array(pose_np)
        self.pose_film_decoder = PoseFiLMDecoderMLX(
            latent_dim=latent_dim,
            base_channels=base_channels,
            film_slots=film_slots,
            film_hidden=film_hidden,
            seed=seed,
            output_layout=output_layout,
        )

    def __call__(self, indices: Any) -> Any:
        z = mx.take(self.latents, indices, axis=0)
        pose = mx.take(self.stored_pose, indices, axis=0)
        return self.pose_film_decoder(z, pose)

    def stored_pose_for(self, indices: Any) -> Any:
        return mx.take(self.stored_pose, indices, axis=0)


def stored_pose_bytes(pose_array: Any, *, quant_step: float = 1e-3) -> int:
    """Byte cost of the STORED pose after uniform-quant + brotli (~kilobytes).

    Quantizr stores ``pose.npy.br`` — the 6-d pose per pair, brotli-compressed. We
    quantize to ``quant_step`` then brotli the int payload (the honest export cost).
    """
    arr = np.asarray(pose_array, dtype=np.float32)
    q = np.round(arr / quant_step).astype(np.int32)
    payload = q.tobytes()
    try:
        import brotli  # type: ignore[import-not-found]

        return len(brotli.compress(payload, quality=11))
    except Exception:  # pragma: no cover - brotli is a runtime dep
        import zlib

        return len(zlib.compress(payload, 9))


__all__ = [
    "POSE_DIM",
    "PoseFiLMDecoderMLX",
    "StoredPoseBundleMLX",
    "stored_pose_bytes",
]
