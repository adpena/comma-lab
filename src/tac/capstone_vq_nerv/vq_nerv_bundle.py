# SPDX-License-Identifier: MIT
"""The original VQ-NeRV bundle with explicit-pose-FiLM injection (Task #78).

The bundle is a thin, ORIGINAL composition over two verified MLX kernels:

* ``HNeRVDecoderMLX`` (the PR95 bit-exact decoder backbone, #81/#82).
* ``VectorQuantizerEMAMLX`` (the van den Oord VQ-EMA primitive, pact_nerv_vq).

The ORIGINAL synthesis (ours):

1. **Per-pair latent ``z_e``** (learnable, ``(num_pairs, latent_dim)``).
2. **VQ quantize** ``z_e -> z_q`` via the EMA codebook + straight-through
   estimator. The *index* is what the archive stores (bit-packed); the codebook
   is free in the native decode. The commitment loss is exposed for the
   score-aware Lagrangian.
3. **FiLM-pose injection**: the STORED 6-dim GT pose ``p`` (passed in per batch)
   is mapped by a tiny learned MLP to a ``(gamma, beta)`` pair over the decoder
   stem channels; the stem feature is modulated ``f -> gamma * f + beta`` BEFORE
   the upsample cascade. This injects the pose grammar into the render so the
   pose term is inherited from the stored scalars (Quantizr's store-pose
   approach), sidestepping the seg/pose small-conv antagonism.

The forward returns the same ``(B, 2, 3, 384, 512)`` N2CHW render the #82 bridge
consumes, so the working score-aware loop drives it unchanged. The bundle also
exposes ``last_commitment_loss`` and ``vq_indices(...)`` for the trainer + export.

NO MPS anywhere; pure MLX-GPU decode + torch-CPU scorer (the exact authority).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tac.local_acceleration.pr95_hnerv_mlx import HNeRVDecoderMLX
from tac.substrates.pact_nerv_vq.mlx_renderer import VectorQuantizerEMAMLX

try:  # pragma: no cover - import guard
    import mlx.core as mx
    import mlx.nn as nn
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None or nn is None:  # pragma: no cover
        raise RuntimeError("tac.capstone_vq_nerv.vq_nerv_bundle requires mlx.")


# The contest PoseNet output is 6-dim (the first 6 pose dims the evaluator scores).
POSE_DIM = 6


@dataclass
class CapstoneVqNervConfig:
    """Config for the original VQ-NeRV + FiLM-pose bundle.

    Defaults target the ~88K-class small basis that produced Quantizr's
    operating point (d_pose 0.00051 at 88K params). ``base_channels`` controls
    the decoder size; ``codebook_size`` controls the per-pair index bit cost
    (``ceil(log2(K))`` bits/pair).

    ``carrier`` selects the per-pair carrier geometry (Arm 2 of the pose A/B,
    the VQ-index-impoverishment fix per
    ``.omx/research/capstone_carrier_pivot_vq_index_impoverishment_*``):

    * ``"vq_index"`` (DEFAULT, the legacy behaviour, byte-identical to before this
      switch): the per-pair 28-d latent is VQ-quantized to an 8-bit codebook index
      (``ceil(log2(K))`` bits/pair). The codebook is free in the native decode; the
      INDEX is the budgeted carrier. **But 8 bits/pair cannot encode 600 distinct
      ego-motions** -> the per-pair content the FiLM/decoder sees is ~K buckets, so
      pose wanders (d_pose 0.06-0.34, never the ~1e-4 tube).
    * ``"stored_latent"`` (the frontier's / PR95's OWN carrier): the per-pair 28-d
      latent is stored DIRECTLY (no VQ, no codebook, no commitment loss) via
      temporal-delta + raw-LZMA (PR95 L24/L25). 28 floats/pair >> 8 bits, so the
      per-pair content is rich enough for pose, while temporal-delta keeps it
      rate-efficient (~10-15 KB for 600 pairs). The frontier reaches d_pose 2.9e-5
      with this carrier.
    """

    num_pairs: int = 600
    latent_dim: int = 28
    base_channels: int = 36
    # Per-pair carrier geometry. "vq_index" is the legacy default (byte-identical
    # to pre-switch); "stored_latent" stores the rich 28-d latent directly.
    carrier: Literal["vq_index", "stored_latent"] = "vq_index"
    codebook_size: int = 256  # 8 bits/pair index; 600 pairs -> 600 bytes packed.
    codebook_decay: float = 0.99
    commitment_weight: float = 0.25
    # FiLM-pose: the stored 6-dim pose is mapped to (gamma, beta) over the
    # stem channels by a tiny 2-layer MLP. film_hidden is the bottleneck width.
    film_enabled: bool = True
    film_hidden: int = 32
    pose_normalize: bool = True  # standardize stored pose before the FiLM MLP.
    # HiNeRV grid positional-encoding (opt-in; default-off = byte-identical to the
    # pre-switch decoder). The HiNeRV delta over HNeRV. When ON, a DETERMINISTIC
    # multi-frequency sinusoidal coordinate grid (computed from coords at decode --
    # ~0 stored bytes) is projected by a tiny learned linear (grid_pe_proj;
    # channels[0] x pe_dim, the only new stored params) and ADDED to the stem
    # feature BEFORE the sin. This injects the spatial inductive bias the pure
    # latent->Linear stem lacks (the grid-PE half of the ~72.3% BD-rate HiNeRV
    # lever; the bilinear-skip half is ALREADY structurally present in every
    # HNeRVDecoderMLX upsample block). pe_dim = 4 * grid_pe_num_freqs (sin/cos x
    # {x,y} x num_freqs). Identity at init (zero-init grid_pe_proj) so the
    # untrained ON-render == the OFF-render -- only training adds the grid grammar.
    hinerv_grid_pe: bool = False
    grid_pe_num_freqs: int = 4
    seed: int = 0


class _PoseFiLM(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Tiny MLP mapping a stored 6-dim pose to per-channel (gamma, beta).

    ``gamma`` is initialized near 1 and ``beta`` near 0 (identity FiLM at init)
    so the untrained bundle renders exactly the no-FiLM decoder — the FiLM only
    *adds* pose grammar as it trains. This keeps the #82 descent intact at init.
    """

    def __init__(self, *, pose_dim: int, channels: int, hidden: int) -> None:
        _require_mlx()
        super().__init__()
        self.channels = int(channels)
        self.fc1 = nn.Linear(int(pose_dim), int(hidden))
        # Output 2*channels: [gamma_pre, beta]. gamma = 1 + tanh(gamma_pre).
        self.fc2 = nn.Linear(int(hidden), 2 * int(channels))
        # Zero-init the second layer so FiLM is identity at init (gamma=1,beta=0).
        self.fc2.weight = mx.zeros_like(self.fc2.weight)  # type: ignore[union-attr]
        self.fc2.bias = mx.zeros_like(self.fc2.bias)  # type: ignore[union-attr]

    def __call__(self, pose: Any) -> tuple[Any, Any]:
        h = mx.sin(self.fc1(pose))  # type: ignore[union-attr]  # NeRF-style activation
        gb = self.fc2(h)
        gamma_pre = gb[:, : self.channels]
        beta = gb[:, self.channels :]
        gamma = 1.0 + mx.tanh(gamma_pre)  # type: ignore[union-attr]  # in (0, 2), =1 at init
        return gamma, beta


class _GridPE(nn.Module if nn is not None else object):  # type: ignore[misc]
    """HiNeRV grid positional-encoding: deterministic coord grid + learned proj.

    Holds the DETERMINISTIC ``(base_h*base_w, pe_dim)`` sinusoidal coordinate grid
    (built once from :func:`tac.capstone_vq_nerv.numpy_reference.grid_positional_encoding`
    so the MLX + numpy paths share the EXACT same grid op-for-op) as a NON-trainable
    constant, plus a tiny learned ``nn.Linear(pe_dim, channels[0])`` projection
    (``grid_pe_proj``). The projection is zero-init so the PE contribution is 0 at
    init (the untrained ON-render == the OFF-render); training adds the grid grammar.

    The grid itself stores ~0 archive bytes (regenerated from coords at inflate);
    only the projection ``weight``/``bias`` (``channels[0] x pe_dim``) are stored.
    """

    def __init__(self, *, base_h: int, base_w: int, num_freqs: int, channels: int) -> None:
        _require_mlx()
        super().__init__()
        from tac.capstone_vq_nerv.numpy_reference import grid_positional_encoding

        grid = grid_positional_encoding(base_h, base_w, num_freqs)  # (HW, pe_dim) numpy
        self.base_h = int(base_h)
        self.base_w = int(base_w)
        self.channels = int(channels)
        self.pe_dim = int(grid.shape[1])
        # Non-trainable deterministic grid (plain array, NOT nn.Parameter, so it is
        # excluded from trainable_parameters() and never enters mx.vjp).
        self._grid = mx.array(grid)  # type: ignore[union-attr]  (HW, pe_dim)
        self.proj = nn.Linear(self.pe_dim, self.channels)
        # Zero-init so grid-PE contributes 0 at init (ON == OFF before training).
        self.proj.weight = mx.zeros_like(self.proj.weight)  # type: ignore[union-attr]
        self.proj.bias = mx.zeros_like(self.proj.bias)  # type: ignore[union-attr]

    def __call__(self) -> Any:
        """Return the projected grid feature ``(base_h, base_w, channels)``."""
        proj = self.proj(self._grid)  # (HW, channels)
        return mx.reshape(proj, (self.base_h, self.base_w, self.channels))  # type: ignore[union-attr]


class CapstoneVqNervBundle(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Original VQ-NeRV decoder bundle with explicit-pose-FiLM injection.

    Forward signature: ``bundle(indices, pose=None) -> render (B,2,3,384,512)``.
    ``indices`` selects per-pair latents; ``pose`` is the stored 6-dim GT pose
    for the same pairs (the FiLM conditioning). When ``pose is None`` the FiLM
    is identity (the no-pose render).

    Trainer contract:
      - ``last_commitment_loss`` holds the most-recent VQ commitment loss.
      - ``vq_indices(indices)`` returns the int32 codebook index per pair.
      - ``ema_update_from_last()`` applies the VQ EMA codebook update from the
        most-recent forward's ``(z_e, indices)`` (call once per training step).
      - ``trainable_parameters()`` excludes the EMA codebook buffers (they are
        EMA-updated, not gradient-updated), matching the pact_nerv_vq contract.
    """

    def __init__(self, cfg: CapstoneVqNervConfig) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        # [SEEDS-PINNED FIX 2026-06-11] Seed the GLOBAL MLX RNG from cfg.seed BEFORE
        # constructing the nn layers (decoder convs / FiLM MLP / grid-PE proj draw
        # their init from the global RNG). Previously only the per-pair latent used
        # an explicit key (below) while the decoder/FiLM init was nondeterministic —
        # the campaign called ``mx.random.seed`` only inside train()/run_curriculum
        # (AFTER construction), so runs were not bit-reproducible and the
        # real-scorer test flaked on the unseeded init. Deterministic-by-construction
        # now (CLAUDE.md "Seeds pinned" non-negotiable); required for clean grid-PE
        # on/off + int8 A/B comparisons (same init both arms).
        mx.random.seed(int(cfg.seed))  # type: ignore[union-attr]
        self.carrier = str(cfg.carrier)
        if self.carrier not in {"vq_index", "stored_latent"}:
            raise ValueError(
                f"carrier must be 'vq_index' or 'stored_latent'; got {self.carrier!r}"
            )
        key = mx.random.key(int(cfg.seed))  # type: ignore[union-attr]
        # Per-pair learnable latent z_e.
        self.latents = (
            mx.random.normal((int(cfg.num_pairs), int(cfg.latent_dim)), key=key)  # type: ignore[union-attr]
            * 0.1
        )
        # VQ-EMA quantizer (codebook free in decode; index bit-packs). Only built
        # for the "vq_index" carrier — "stored_latent" stores the rich 28-d latent
        # directly (no codebook, no commitment loss), so the quantizer is None.
        if self.carrier == "vq_index":
            self.quantizer = VectorQuantizerEMAMLX(
                codebook_size=int(cfg.codebook_size),
                latent_dim=int(cfg.latent_dim),
                decay=float(cfg.codebook_decay),
            )
        else:
            self.quantizer = None
        # The decoder backbone (PR95 bit-exact, M-arch skip+refine ON).
        self.decoder = HNeRVDecoderMLX(
            latent_dim=int(cfg.latent_dim),
            base_channels=int(cfg.base_channels),
        )
        # PER-FRAME FiLM-pose injection over the FINAL feature channels
        # (channels[-1], the rgb-head input). CRUX (2026-06-10): PoseNet measures
        # ego-motion = the DIFFERENTIAL between frame0 and frame1, so a single
        # FiLM on the SHARED feature (the prior design) has ~zero Jacobian in the
        # rewarded pose direction (both frames modulated identically -> the pose
        # cannot steer the frame0<->frame1 difference -> d_pose bounces ~0.4 and
        # never reaches the tube). The fix (matching the #84 PoseFiLMDecoderMLX
        # that held d_pose 2.7e-4) is SEPARATE film0/film1 modulating the feature
        # DIFFERENTLY before each rgb head, giving direct control of the motion.
        self.stem_channels = int(self.decoder.channels[0])
        self.feat_channels = int(self.decoder.channels[-1])
        # HiNeRV grid-PE (opt-in; default-off byte-identical). Built ONLY when
        # enabled so the OFF path has no grid_pe_proj param and the forward is
        # byte-identical to the pre-switch decoder.
        self.hinerv_grid_pe = bool(cfg.hinerv_grid_pe)
        self.grid_pe_num_freqs = int(cfg.grid_pe_num_freqs)
        if self.hinerv_grid_pe:
            self.grid_pe_proj = _GridPE(
                base_h=int(self.decoder.base_h),
                base_w=int(self.decoder.base_w),
                num_freqs=self.grid_pe_num_freqs,
                channels=self.stem_channels,
            )
        self.film_enabled = bool(cfg.film_enabled)
        if self.film_enabled:
            self.pose_film0 = _PoseFiLM(
                pose_dim=POSE_DIM,
                channels=self.feat_channels,
                hidden=int(cfg.film_hidden),
            )
            self.pose_film1 = _PoseFiLM(
                pose_dim=POSE_DIM,
                channels=self.feat_channels,
                hidden=int(cfg.film_hidden),
            )
        # Pose standardization stats (set by the trainer from the stored pose).
        self._pose_mean = mx.zeros((POSE_DIM,))  # type: ignore[union-attr]
        self._pose_std = mx.ones((POSE_DIM,))  # type: ignore[union-attr]
        # Telemetry from the most-recent forward.
        self.last_commitment_loss: Any = mx.array(0.0)  # type: ignore[union-attr]
        self._last_z_e: Any = None
        self._last_indices: Any = None

    # ---- pose standardization (set once from the stored GT pose) -------------

    def set_pose_stats(self, pose_mean: Any, pose_std: Any) -> None:
        """Set the FiLM input standardization (mean/std over stored GT pose)."""
        _require_mlx()
        self._pose_mean = mx.array(np.asarray(pose_mean, dtype=np.float32))  # type: ignore[union-attr]
        std = np.asarray(pose_std, dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)  # guard zero-variance dims
        self._pose_std = mx.array(std)  # type: ignore[union-attr]

    def _norm_pose(self, pose: Any) -> Any:
        if not self.cfg.pose_normalize:
            return pose
        return (pose - self._pose_mean) / self._pose_std

    # ---- forward -------------------------------------------------------------

    def _quantize(self, indices: Any) -> Any:
        """Gather per-pair latents and produce the decoder input ``z``.

        ``vq_index`` carrier: VQ-quantize z_e -> straight-through z_q (the 8-bit
        index is the budgeted carrier; the commitment loss conditions the STE).

        ``stored_latent`` carrier: NO VQ — the gathered per-pair 28-d latent IS
        ``z`` directly (the rich carrier the frontier uses). The gradient flows
        straight into ``self.latents`` (no codebook, no STE), and the commitment
        loss is identically 0 (there is nothing to commit to). This is the
        VQ-index-impoverishment fix: 28 floats/pair >> 8 bits, so the per-pair
        content is rich enough for the FiLM/decoder to express distinct pose.
        """
        z_e = mx.take(self.latents, indices, axis=0)  # type: ignore[union-attr]  (B, latent_dim)
        if self.carrier == "stored_latent":
            # The stored latent IS the decoder input — no quantization. Commitment
            # loss is 0 (no codebook); the gradient updates self.latents directly.
            self.last_commitment_loss = mx.array(0.0)  # type: ignore[union-attr]
            self._last_z_e = z_e
            self._last_indices = None
            return z_e
        z_q_st, vq_idx, commit = self.quantizer(z_e)
        self.last_commitment_loss = commit
        self._last_z_e = z_e
        self._last_indices = vq_idx
        return z_q_st

    def _decode_with_film(self, z_q: Any, pose: Any | None) -> Any:
        """Decode z_q through the backbone, FiLM-modulating the stem feature."""
        dec = self.decoder
        batch = int(z_q.shape[0])
        x = dec.stem(z_q)
        x = mx.reshape(  # type: ignore[union-attr]
            x, (batch, dec.channels[0], dec.base_h, dec.base_w)
        )
        x = mx.transpose(x, (0, 2, 3, 1))  # type: ignore[union-attr]  -> NHWC
        # HiNeRV grid-PE: add the DETERMINISTIC coordinate grid (projected by the
        # tiny learned grid_pe_proj) to the stem feature BEFORE sin. Op-for-op with
        # numpy_reference._features_nhwc so the inflate reproduces it exactly.
        if self.hinerv_grid_pe:
            pe = self.grid_pe_proj()  # (base_h, base_w, channels[0])
            x = x + mx.reshape(pe, (1, dec.base_h, dec.base_w, dec.channels[0]))  # type: ignore[union-attr]
        x = mx.sin(x)  # type: ignore[union-attr]
        for block in dec.blocks:
            x = block(x)
        refined = dec.refine1(dec.refine0(x))
        feat = x + 0.1 * mx.sin(refined)  # type: ignore[union-attr]  (B,H,W,feat_channels)
        # PER-FRAME FiLM-pose injection: modulate the shared feature DIFFERENTLY
        # for each frame head so the pose can steer the frame0<->frame1 motion
        # (the rewarded PoseNet direction). Identity at init (zero-init fc2 ->
        # gamma=1,beta=0) so the untrained render == the no-FiLM render.
        if self.film_enabled and pose is not None:
            pn = self._norm_pose(pose)
            g0, b0 = self.pose_film0(pn)  # (B, feat_channels) each
            g1, b1 = self.pose_film1(pn)
            fc = self.feat_channels
            feat0 = mx.reshape(g0, (batch, 1, 1, fc)) * feat + mx.reshape(b0, (batch, 1, 1, fc))  # type: ignore[union-attr]
            feat1 = mx.reshape(g1, (batch, 1, 1, fc)) * feat + mx.reshape(b1, (batch, 1, 1, fc))  # type: ignore[union-attr]
        else:
            feat0 = feat
            feat1 = feat
        f0 = mx.sigmoid(dec.rgb_0(feat0)) * 255.0  # type: ignore[union-attr]
        f1 = mx.sigmoid(dec.rgb_1(feat1)) * 255.0  # type: ignore[union-attr]
        pair = mx.stack([f0, f1], axis=1)  # type: ignore[union-attr]  (B,2,H,W,C)
        return mx.transpose(pair, (0, 1, 4, 2, 3))  # type: ignore[union-attr]  -> N2CHW

    def __call__(self, indices: Any, pose: Any | None = None) -> Any:
        """Render the pair batch for ``indices`` with optional pose-FiLM."""
        _require_mlx()
        z_q = self._quantize(indices)
        return self._decode_with_film(z_q, pose)

    # ---- carrier accessors + EMA update --------------------------------------

    def vq_indices(self, indices: Any) -> Any:
        """Return the int32 codebook index per selected pair (for export).

        Only valid for the ``vq_index`` carrier; ``stored_latent`` has no codebook
        and stores the latent directly (use :meth:`all_latents`).
        """
        _require_mlx()
        if self.carrier != "vq_index":
            raise RuntimeError(
                "vq_indices() is only valid for the 'vq_index' carrier; the "
                f"'{self.carrier}' carrier stores the latent directly "
                "(use all_latents())."
            )
        z_e = mx.take(self.latents, indices, axis=0)  # type: ignore[union-attr]
        _, vq_idx, _ = self.quantizer(z_e)
        mx.eval(vq_idx)
        return vq_idx

    def all_vq_indices(self) -> np.ndarray:
        """Return the full ``(num_pairs,)`` codebook index array as numpy int32."""
        _require_mlx()
        idx = self.vq_indices(mx.arange(int(self.cfg.num_pairs)))  # type: ignore[union-attr]
        return np.asarray(idx, dtype=np.int32)

    def all_latents(self) -> np.ndarray:
        """Return the full ``(num_pairs, latent_dim)`` per-pair latent as numpy fp32.

        This is the ``stored_latent`` carrier: the rich 28-d per-pair latent the
        archive stores directly (temporal-delta + LZMA), NOT a quantized index.
        Valid for BOTH carriers (the underlying ``self.latents`` exists in both),
        but it is the EXPORT carrier only for ``stored_latent``.
        """
        _require_mlx()
        mx.eval(self.latents)  # type: ignore[union-attr]
        return np.asarray(self.latents, dtype=np.float32)

    def ema_update_from_last(self) -> None:
        """Apply the VQ EMA codebook update from the most-recent forward.

        No-op for the ``stored_latent`` carrier (no codebook to EMA-update).
        """
        if self.carrier != "vq_index":
            return
        if self._last_z_e is None or self._last_indices is None:
            return
        self.quantizer.ema_update(self._last_z_e, self._last_indices)

    def trainable_parameters(self) -> dict[str, Any]:
        """Trainable params EXCLUDING the EMA codebook buffers.

        The quantizer's ``_codebook`` / ``_ema_*`` are EMA-updated, not
        gradient-updated, so they must not enter ``mx.vjp``. MLX's
        ``VectorQuantizerEMAMLX`` stores them as plain arrays (not nn.Parameter),
        so ``trainable_parameters()`` already excludes them. We surface this
        method to make the contract explicit + testable.
        """
        return super().trainable_parameters()


__all__ = [
    "POSE_DIM",
    "CapstoneVqNervBundle",
    "CapstoneVqNervConfig",
]
