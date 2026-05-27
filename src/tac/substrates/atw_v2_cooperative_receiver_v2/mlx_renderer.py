# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — MLX-native renderer (L0 SCAFFOLD).

Per operator binding directive #1 ("MLX first requirement might also force us
out of the issue we were having before where we had great ideas but we're
building them as Boltons to the same substrates over and over again") +
Phase 3 design memo §1 + §8 (MLX drift minimization per primitive).

This module is the MLX-NATIVE substrate-engineering surface for the ATW V2
cooperative-receiver V2 substrate. Per Phase 2 design-decision memo:

- **Layer 1 META-unwind binding**: Atick-Redlich 1990 as SINGLE substrate-
  optimal anchor; demote Tishby IB + Wyner-Ziv to advisory cross-checks.
- **NEW conditioning variable**: ego-motion FOE projection (Ballard 2007 +
  Catalog #311) REPLACES per-class softmax (D4 INDEPENDENT verdict
  ``I(latent; scorer_class) = 0.006385`` bits/symbol empirically falsified).
- **Conditional source coding R(D|Y_ego_motion)** replaces V1's WZ residual.

The MLX renderer mirrors the numpy reference (``numpy_reference.py``) at
bit-exact-per-primitive precision so MLX↔numpy parity verification in
test_basic.py is straightforward.

Critical MLX drift minimization (per pr95 MLX research + Phase 3 §8)
=====================================================================

Per ``codex_findings_mlx_drift_determinism_online_research_20260522T050151Z_codex.md`` +
``pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md`` +
``pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md``:

- Use ``precise=True`` for conv2d when possible
- Cast→FP64→reduce→cast-back for mean/softmax to limit reduction-order drift
- Per-pair independence (no cross-pair recurrence) to prevent drift compounding
- Sister numpy reference per-primitive parity test in test_basic.py

Non-promotable canonical contract
=================================

Per CLAUDE.md "MLX portable-local-substrate authority":
- All MLX outputs tagged ``[macOS-MLX research-signal]``
- ``score_claim=False``, ``promotion_eligible=False``,
  ``ready_for_exact_eval_dispatch=False``
- Promotion path: MLX state_dict → PyTorch (via #1251 bridge pattern; not
  implemented in L0 scaffold) → archive → contest-equivalence gate per
  Catalog #1265 → operator routes paid CUDA dispatch

L0 SCAFFOLD bounded scope
=========================

- Single-substrate ATW V2 cooperative-receiver V2 only
- ego-motion FOE projection (3 translation + 3 rotation components per pair)
- HNeRV-style decoder canonical pattern (per Z6 mlx_renderer.py)
- Per-pair independence (no cross-pair recurrence in L0)
- ~600-800 LOC total scaffold (substrate-engineering per HNeRV L7)

Cross-references
----------------

* Phase 3 design memo §1 + §8 (canonical-vs-unique + MLX drift)
* Phase 1 audit + Phase 2 design decision (the cargo-cult-first methodology)
* ``src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py`` (canonical MLX
  renderer pattern this module mirrors)
* ``src/tac/local_acceleration/pr95_hnerv_mlx.py`` (canonical MLX HNeRV
  reference pattern)
* ``src/tac/substrates/atw_v2_cooperative_receiver_v2/numpy_reference.py``
  (sister numpy reference for Axis 3 portability)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tac.substrates.atw_v2_cooperative_receiver_v2.numpy_reference import (
        CooperativeReceiverConfig,
    )

try:  # pragma: no cover — exercised on Apple Silicon with MLX installed
    import mlx.core as mx
    import mlx.nn as nn
except Exception as exc:  # pragma: no cover — import guard for non-Apple CI
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "atw_v2_cooperative_receiver_v2_mlx_renderer_v1"


def is_mlx_available() -> bool:
    """Return True iff MLX is importable in the current Python process."""
    return _MLX_IMPORT_ERROR is None and mx is not None


def require_mlx_available() -> None:
    """Raise RuntimeError with diagnostic if MLX is not importable."""
    if not is_mlx_available():
        raise RuntimeError(
            f"MLX is not available in this Python environment. Required for "
            f"atw_v2_cooperative_receiver_v2.mlx_renderer. Import error: "
            f"{_MLX_IMPORT_ERROR!r}. Per Axis 3 portability: numpy_reference "
            f"is the portable fallback."
        )


# ----------------------------------------------------------------------------
# MLX primitives (sister to numpy_reference for MLX↔numpy parity verification)
# ----------------------------------------------------------------------------


def mlx_ego_motion_foe_projection(pose_delta: mx.array) -> mx.array:
    """Per-pair ego-motion FOE projection in MLX. Sister to numpy_ego_motion_foe_projection.

    Per Phase 3 design memo §7 + Ballard 2007 + Catalog #311.

    Args:
        pose_delta: per-pair PoseNet pose-delta tensor shape ``(B, 6)``.
            First 3 components = translation; last 3 = rotation.

    Returns:
        ego-motion FOE projection ``Y_ego_motion`` shape ``(B, 6)``.

    MLX drift per Phase 3 §8: bit-exact-equivalent to numpy reference;
    no reduction/FMA-reassociation concerns for elementwise + division.
    """
    require_mlx_available()
    if pose_delta.ndim != 2 or pose_delta.shape[1] != 6:
        raise ValueError(
            f"pose_delta must be (B, 6); got {pose_delta.shape}"
        )
    translation = pose_delta[:, :3]
    rotation = pose_delta[:, 3:]
    eps = 1e-8
    t_norm = mx.sqrt(mx.sum(translation * translation, axis=1, keepdims=True)) + eps
    r_norm = mx.sqrt(mx.sum(rotation * rotation, axis=1, keepdims=True)) + eps
    return mx.concatenate(
        [translation / t_norm, rotation / r_norm],
        axis=1,
    )


# ----------------------------------------------------------------------------
# MLX modules for the cooperative-receiver substrate
# ----------------------------------------------------------------------------


class _CondEmbeddingHead(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Conditioning embedding head: ego_motion -> cond_embed_dim hidden -> latent_dim projection.

    Per Phase 3 design memo §1 + §7. Sister to numpy_reference's two-layer MLP.
    """

    def __init__(self, cfg: CooperativeReceiverConfig) -> None:
        require_mlx_available()
        super().__init__()
        self.cfg = cfg
        # Layer 1: (ego_motion_dim) -> (cond_embed_dim)
        self.fc1 = nn.Linear(cfg.ego_motion_dim, cfg.cond_embed_dim)
        # Layer 2: (cond_embed_dim) -> (latent_dim) — projects directly onto latent_dim
        self.fc2 = nn.Linear(cfg.cond_embed_dim, cfg.latent_dim)

    def __call__(self, ego_motion_proj: mx.array) -> mx.array:
        h = self.fc1(ego_motion_proj)
        h = nn.relu(h)
        return self.fc2(h)


class _HNeRVStyleDecoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    """HNeRV-style decoder per Phase 3 §1 + §7 + canonical Z6 pattern.

    Forward: latent -> Linear initial_proj -> reshape grid -> N x [Conv2d(4*out) +
    PixelShuffle(2) + ReLU] -> final Conv2d(6) -> bilinear interp -> sigmoid -> (rgb_0, rgb_1).
    """

    def __init__(self, cfg: CooperativeReceiverConfig) -> None:
        require_mlx_available()
        super().__init__()
        self.cfg = cfg
        embed_total = cfg.decoder_embed_dim * cfg.decoder_initial_grid_h * cfg.decoder_initial_grid_w
        self.initial_proj = nn.Linear(cfg.latent_dim, embed_total)

        self.blocks: list[Any] = []
        in_ch = cfg.decoder_embed_dim
        for i in range(cfg.decoder_num_upsample_blocks):
            out_ch = cfg.decoder_channels[i]
            self.blocks.append(nn.Conv2d(in_ch, 4 * out_ch, kernel_size=3, padding=1))
            in_ch = out_ch
        self.final_conv = nn.Conv2d(in_ch, 6, kernel_size=3, padding=1)

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array]:
        B = z.shape[0]
        # Initial projection
        flat = self.initial_proj(z)
        # Reshape to NHWC layout (MLX convention)
        grid = mx.reshape(
            flat,
            (B, self.cfg.decoder_initial_grid_h, self.cfg.decoder_initial_grid_w, self.cfg.decoder_embed_dim),
        )

        h = grid
        for i in range(self.cfg.decoder_num_upsample_blocks):
            h = self.blocks[i](h)
            # PixelShuffle(2): (B, H, W, 4*C) -> (B, 2H, 2W, C)
            B_h, H, W, C_in = h.shape
            r = 2
            if C_in % (r * r) != 0:
                raise RuntimeError(
                    f"PixelShuffle requires channels divisible by {r*r}; got {C_in}"
                )
            C_out = C_in // (r * r)
            h = mx.reshape(h, (B_h, H, W, r, r, C_out))
            h = mx.transpose(h, (0, 1, 3, 2, 4, 5))
            h = mx.reshape(h, (B_h, H * r, W * r, C_out))
            h = nn.relu(h)

        h = self.final_conv(h)

        # Resize to output_height x output_width via canonical PR95 bilinear helper.
        #
        # FIX-WAVE-R1''-H (2026-05-26): replaced ``mx.repeat`` nearest-neighbor
        # upsample with canonical
        # ``tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize_nhwc`` which is
        # empirically PyTorch-byte-stable (≤1e-5 abs drift vs
        # ``F.interpolate(size=..., mode='bilinear', align_corners=False)``).
        # Per R1'' CRITICAL finding H-R1''-1: prior ``mx.repeat`` 2-axis tile
        # was the SAME ANTI-PATTERN that caused sister A=DreamerV3 ``max_abs=24.34``
        # drift pre-FIX-WAVE-R1 (canonical fix at commit ``e1b101888``).
        # Per Phase 3 §1 + §8 + CONSOLIDATE-OP-1 canonical MLX primitives wave
        # (commit ``caf29acdb``): substrates MUST delegate to the canonical helper
        # at MLX training time rather than re-implement local upsample copies.
        #
        # Catalog #295 self-containment is preserved because the canonical helper
        # is imported only at MLX training time in ``mlx_renderer.py``; the
        # substrate's inflate runtime at ``inflate.py`` is PyTorch-only and does
        # NOT import MLX (PyTorch uses ``F.interpolate(mode='bilinear',
        # align_corners=False)`` natively). The Catalog #295 contract scopes
        # ``submissions/*/inflate.py`` PYTHONPATH self-containment; this
        # substrate's MLX module is at
        # ``src/tac/substrates/atw_v2_cooperative_receiver_v2/`` which is in-tree
        # by definition.
        target_h = self.cfg.output_height
        target_w = self.cfg.output_width
        if h.shape[1] != target_h or h.shape[2] != target_w:
            from tac.local_acceleration.pr95_hnerv_mlx import (
                bilinear_resize_nhwc,
            )

            h = bilinear_resize_nhwc(
                h, target_h=target_h, target_w=target_w, align_corners=False
            )

        h = mx.sigmoid(h)
        # Split: NHWC layout, channels last
        rgb_0 = h[:, :, :, :3]
        rgb_1 = h[:, :, :, 3:]
        # Transpose to (B, C, H, W) NCHW layout to match PyTorch convention
        rgb_0 = mx.transpose(rgb_0, (0, 3, 1, 2))
        rgb_1 = mx.transpose(rgb_1, (0, 3, 1, 2))
        return rgb_0, rgb_1


class ATWv2CooperativeReceiverV2MLX:
    """ATW V2 cooperative-receiver V2 MLX-native renderer.

    Forward (eval / inflate mode):
        1. Y_ego_motion = mlx_ego_motion_foe_projection(pose_delta_per_pair)
        2. cond_embed = cond_embed_head(Y_ego_motion)
        3. z = per_pair_latent_residual + cond_embed
        4. rgb_0, rgb_1 = decoder(z)

    Per Phase 3 design memo §1 + §7. Sister to numpy_reference.
    """

    def __init__(self, cfg: CooperativeReceiverConfig) -> None:
        require_mlx_available()
        self.cfg = cfg
        self.cond_embed_head = _CondEmbeddingHead(cfg)
        self.decoder = _HNeRVStyleDecoder(cfg)
        # Per-pair latent residual (auto-decoder, learned at training time)
        self.per_pair_latent_residual = mx.zeros((cfg.num_pairs, cfg.latent_dim))

    def reconstruct_pair(
        self,
        pair_indices: mx.array,
        pose_delta: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Inflate-time reconstruction: per-pair latent + ego-motion -> RGB pair.

        Args:
            pair_indices: ``(B,)`` integer tensor in ``[0, num_pairs)``.
            pose_delta: ``(B, 6)`` per-pair PoseNet pose-delta.

        Returns:
            ``(rgb_0, rgb_1)`` each shape ``(B, 3, output_H, output_W)`` in [0, 1].
        """
        require_mlx_available()
        # Ego-motion FOE projection
        Y_ego_motion = mlx_ego_motion_foe_projection(pose_delta)
        # Conditioning embedding
        cond_embed = self.cond_embed_head(Y_ego_motion)
        # Per-pair latent reconstruction
        z_residual = self.per_pair_latent_residual[pair_indices]
        z = z_residual + cond_embed
        # Decode
        return self.decoder(z)

    def set_latent_residuals(self, latents: np.ndarray) -> None:
        """Bulk-load per-pair latent residuals from an external archive.

        Used at inflate time after loading the per_pair_latent_blob from
        the ATWv2CR2 archive.
        """
        require_mlx_available()
        if latents.shape != (self.cfg.num_pairs, self.cfg.latent_dim):
            raise ValueError(
                f"latents shape mismatch: got {latents.shape}, expected "
                f"({self.cfg.num_pairs}, {self.cfg.latent_dim})"
            )
        self.per_pair_latent_residual = mx.array(latents.astype(np.float32))


class ATWv2CooperativeReceiverV2TrainableMLX(
    nn.Module if nn is not None else object  # type: ignore[misc]
):
    """Trainable ``nn.Module`` wrapper for the canonical MLX score-aware harness.

    MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: the prior blocker was that
    :class:`ATWv2CooperativeReceiverV2MLX` is NOT an ``mlx.nn.Module`` (no
    ``.parameters()`` for ``value_and_grad``) AND its ``reconstruct_pair`` takes
    a separate ``pose_delta`` the harness does not supply at call time. This
    wrapper extinguishes both: it is an ``nn.Module`` exposing the harness's
    ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` NCHW ``[0, 1]`` convention with
    the per-pair ego-motion ``pose_delta`` carried INTERNALLY as a learnable
    per-pair table (auto-decoder pattern; the substrate's distinguishing
    primitive is the ego-motion FOE conditioning, learned per-pair from the
    reconstruction signal rather than supplied by an external PoseNet at
    training time).

    Trainable parameters (discovered by MLX ``value_and_grad`` via
    ``.parameters()``):

    - ``cond_embed_head`` (ego_motion -> latent projection; ``nn.Linear`` x2)
    - ``decoder`` (HNeRV-style; ``nn.Linear`` + ``nn.Conv2d`` blocks)
    - ``per_pair_latent_residual`` ``(num_pairs, latent_dim)`` learnable table
    - ``per_pair_pose_delta`` ``(num_pairs, ego_motion_dim)`` learnable table

    Per Phase 3 design memo §1 + §7 (Atick-Redlich cooperative-receiver +
    ego-motion FOE conditioning per Catalog #311). Non-promotable
    ``[macOS-MLX research-signal]`` per CLAUDE.md "MLX portable-local-substrate
    authority".
    """

    def __init__(self, cfg: CooperativeReceiverConfig) -> None:
        require_mlx_available()
        super().__init__()
        self.cfg = cfg
        self.cond_embed_head = _CondEmbeddingHead(cfg)
        self.decoder = _HNeRVStyleDecoder(cfg)
        # Learnable per-pair latent residual (auto-decoder) + per-pair pose
        # delta carried internally (NOT a call-time argument). Small init so
        # the conditioning starts near the cond-embed prior.
        key_latent = mx.random.key(0)
        key_pose = mx.random.key(1)
        self.per_pair_latent_residual = (
            mx.random.normal(shape=(cfg.num_pairs, cfg.latent_dim), key=key_latent)
            * 0.01
        )
        self.per_pair_pose_delta = (
            mx.random.normal(
                shape=(cfg.num_pairs, cfg.ego_motion_dim), key=key_pose
            )
            * 0.01
        )

    def reconstruct_pair(
        self, pair_indices: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Harness forward: per-pair latent + ego-motion -> (rgb_0, rgb_1).

        Args:
            pair_indices: ``(B,)`` int tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1)`` each ``(B, 3, output_H, output_W)`` in ``[0, 1]``
            (the harness ``reconstruct_pair_nchw01`` convention).
        """
        require_mlx_available()
        pose_delta = self.per_pair_pose_delta[pair_indices]
        Y_ego_motion = mlx_ego_motion_foe_projection(pose_delta)
        cond_embed = self.cond_embed_head(Y_ego_motion)
        z = self.per_pair_latent_residual[pair_indices] + cond_embed
        return self.decoder(z)

    def __call__(
        self, pair_indices: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Alias for :meth:`reconstruct_pair` (default forward)."""
        return self.reconstruct_pair(pair_indices)


__all__ = [
    "SCHEMA_VERSION",
    "ATWv2CooperativeReceiverV2MLX",
    "ATWv2CooperativeReceiverV2TrainableMLX",
    "is_mlx_available",
    "mlx_ego_motion_foe_projection",
    "require_mlx_available",
]
