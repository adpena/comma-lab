# SPDX-License-Identifier: MIT
"""Cool-Chic non-conv basis CARRIER for the live-SegNet ``ScoreAwareTrainer``.

This is the REAL Cool-Chic carrier (lane ``lane_cool_chic_score_aware_basis_20260611``):
multiresolution learnable latent grids + a tiny synthesis net + an
autoregressive (ARM) rate estimator. It implements the ``reconstruct_pair(idx)``
interface the existing :class:`tac.score_aware_loop.trainer.ScoreAwareTrainer`
consumes, so the proven PR95-faithful live-SegNet loss / EMA / eval-roundtrip /
exact-``d_seg`` machinery is REUSED, not duplicated.

The thesis under test (operator 2026-06-11): conv-HNeRV needs ~162-256K params
to reach the ``d_seg`` 5.6e-4 basin; the param<->d_seg wall may be
BASIS-SPECIFIC, not fundamental. Cool-chic-video reaches competitive fidelity at
~hundreds of synthesis params [Leguay 2024, hal.science] because the spatial
structure lives in the (cheaply entropy-coded) latent GRIDS, not in dense conv
weights. The deliverable is the CHARGED BYTES at which the exact (torch-CPU)
``d_seg`` enters the basin — the single number that says basis-specific vs
fundamental.

Charged-byte accounting (honest, per CLAUDE.md "Forbidden score claims"):
  - **latent bytes** = sum over grids of the ARM-estimated entropy-coded bits
    (``-log2 p(z|ARM)``), in BYTES. This is the dominant payload and the only
    thing that scales with grid resolution. Measured at the SHARED grids (frame0
    state) plus the per-pair frame1 delta latents.
  - **synthesis + ARM weight bytes** = param_count * bytes_per_param (default
    fp16 = 2 B/param) — a small fixed cost, NOT video-derived.

Parity: the synthesis forward MUST match
:func:`tac.residual_basis.cool_chic_synthesis_numpy.synthesize_rgb_numpy` on REAL
(non-zero) weights+grids (the portability contract / fake-parity guard).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CoolChicGridSpec:
    """Multiresolution latent grid spec at a base ``(H, W)``.

    ``n_grids`` grids at resolutions ``(H>>l, W>>l)`` for ``l`` in
    ``[0, n_grids)``, each with ``channels_per_grid`` channels. The total latent
    count (which dominates charged bytes) is
    ``sum_l channels_per_grid * (H>>l) * (W>>l)``.
    """

    base_h: int
    base_w: int
    n_grids: int = 4
    channels_per_grid: int = 2

    def grid_shapes(self) -> list[tuple[int, int, int]]:
        shapes: list[tuple[int, int, int]] = []
        for level in range(self.n_grids):
            h = max(self.base_h >> level, 1)
            w = max(self.base_w >> level, 1)
            shapes.append((self.channels_per_grid, h, w))
        return shapes

    def total_latent_count(self) -> int:
        return sum(c * h * w for c, h, w in self.grid_shapes())

    @property
    def c_in(self) -> int:
        """Synthesis input channels = sum of per-grid channels (concatenated)."""
        return self.channels_per_grid * self.n_grids


class _ARM(nn.Module):
    """Tiny autoregressive rate model: predicts (mu, log_scale) per latent.

    Cool-Chic's ARM conditions each latent on its already-decoded causal
    neighbours via a masked conv. We use a 3x3 mask (top + left causal context)
    shared across grids and channels — a few dozen params. The rate is the
    Laplacian ``-log2 p`` summed over latents. This is differentiable so the
    score-aware loop can trade latent rate against d_seg.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        # 3x3 causal mask: rows above + left-of-center on the current row.
        mask = torch.zeros(1, 1, 3, 3)
        mask[0, 0, 0, :] = 1.0
        mask[0, 0, 1, 0] = 1.0
        self.register_buffer("mask", mask)
        self.ctx = nn.Conv2d(1, hidden, 3, padding=1, bias=True)
        self.head = nn.Conv2d(hidden, 2, 1, bias=True)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def rate_bits(self, z: torch.Tensor, *, quant_step: float = 1.0) -> torch.Tensor:
        """Return total entropy-coded bits for one grid ``z`` of shape (C,h,w).

        Each channel is treated as an independent (h, w) map; the masked conv
        gives the causal context. Rate = sum -log2(Laplacian(z|mu,scale)*step).
        """
        c, h, w = z.shape
        zc = z.reshape(c, 1, h, w)
        masked_w = self.ctx.weight * self.mask
        ctx = F.conv2d(zc, masked_w, self.ctx.bias, padding=1)
        ctx = F.gelu(ctx)
        params = self.head(ctx)  # (c, 2, h, w)
        mu = params[:, 0]
        log_scale = params[:, 1].clamp(-8.0, 8.0)
        scale = torch.exp(log_scale).clamp_min(1e-6)
        abs_dev = (z - mu).abs()
        density = (1.0 / (2.0 * scale)) * torch.exp(-abs_dev / scale)
        prob = (density * quant_step).clamp(1e-12, 1.0)
        return (-torch.log2(prob)).sum()


class _PoseFiLM(nn.Module):
    """Torch port of the capstone stored-pose FiLM (Quantizr store-pose lesson).

    Adapted (NOT cargo-culted) from
    :class:`tac.capstone_vq_nerv.vq_nerv_bundle._PoseFiLM` (the MLX bundle that
    already holds ``d_pose <= 3e-4`` via stored-pose FiLM). Mechanism identical;
    backend differs (torch-CPU here per the GPU-busy constraint):

      pose (B, 6) -> sin(fc1) -> fc2 -> [gamma_pre, beta] (each (B, channels)).
      gamma = 1 + tanh(gamma_pre)  (in (0, 2), =1 at init), beta (=0 at init).

    The second layer is zero-init so FiLM is IDENTITY at init (gamma=1, beta=0):
    the untrained carrier renders EXACTLY the no-FiLM Cool-Chic carrier, so the
    proven #82 d_seg descent is intact and FiLM only *adds* pose grammar as it
    trains. Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE" — this is an
    OPERATIONAL mechanism (it modulates the rendered frame1 from a STORED pose).
    """

    def __init__(self, *, pose_dim: int, channels: int, hidden: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.fc1 = nn.Linear(int(pose_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), 2 * int(channels))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.sin(self.fc1(pose))  # NeRF-style activation (matches capstone)
        gb = self.fc2(h)
        gamma_pre = gb[:, : self.channels]
        beta = gb[:, self.channels :]
        gamma = 1.0 + torch.tanh(gamma_pre)  # (0, 2), =1 at init
        return gamma, beta

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CoolChicPairCarrier(nn.Module):
    """Cool-Chic carrier rendering ``(B, 3, H, W)`` pairs for the score loop.

    Frame0 is rendered from SHARED multiresolution grids (a global "state" the
    pose carrier already handles for d_pose per the orphan inventory). Frame1 is
    rendered from frame0's features plus a small PER-PAIR frame1-delta latent
    grid — the part the score-aware loop tunes to lower d_seg on the LAST frame
    (the SegNet term uses ``x[:, -1, ...]`` = frame1).

    Pose-fold (operator 2026-06-11, the Quantizr STORE-pose lesson): when
    ``pose_film_enabled`` the carrier holds a STORED 6-dim pose per pair (a
    non-trainable buffer set from the GT PoseNet pose via :meth:`set_stored_pose`)
    and a tiny :class:`_PoseFiLM` that maps it to per-channel ``(gamma, beta)``
    modulating the synthesis HIDDEN features of frame1 BEFORE the second conv.
    This injects the pose grammar from STORED scalars (~1 KB charged) rather than
    asking the coarse latent grid to *reconstruct* PoseNet ego-motion — the
    failure mode B1-CLOSE measured (d_pose 0.06-2.49). The FiLM is identity at
    init so the proven d_seg descent is unchanged when it begins training.

    Args:
        n_pairs: number of pairs (per-pair frame1-delta latent rows).
        spec: multiresolution grid spec.
        synth_hidden: synthesis hidden channels (param count driver; default 12).
        out_hw: rendered ``(H, W)`` (the trainer resizes to scorer res).
        bytes_per_param: charged bytes/param for weights (fp16 default = 2).
        pose_film_enabled: if True, fold the stored-pose FiLM into frame1.
        pose_dim: stored pose dimensionality (contest PoseNet = 6).
        film_hidden: FiLM MLP bottleneck width.

    The module's parameter names use ``latent`` for the grids so the trainer's
    optimizer gives them the 10x latent LR (per ``ScoreAwareTrainer``).
    """

    def __init__(
        self,
        n_pairs: int,
        spec: CoolChicGridSpec,
        *,
        synth_hidden: int = 12,
        out_hw: tuple[int, int] = (96, 128),
        bytes_per_param: float = 2.0,
        quant_step: float = 1.0,
        pose_film_enabled: bool = False,
        pose_dim: int = 6,
        film_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.spec = spec
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.bytes_per_param = float(bytes_per_param)
        self.quant_step = float(quant_step)
        self.pose_film_enabled = bool(pose_film_enabled)
        self.pose_dim = int(pose_dim)

        shapes = spec.grid_shapes()
        # Shared frame0 grids (the global state); per CLAUDE.md "EMA"/"latent"
        # naming so the trainer applies the latent LR multiplier.
        self.latent_grids = nn.ParameterList(
            [nn.Parameter(torch.randn(c, h, w) * 0.3) for (c, h, w) in shapes]
        )
        # Per-pair frame1 delta latents at the COARSEST grid only (cheap; the
        # frame1 motion delta is low-frequency). Shape (n_pairs, C, h, w).
        c0, h0, w0 = shapes[-1]
        self.frame1_delta = nn.Parameter(
            torch.randn(self.n_pairs, c0, h0, w0) * 0.1
        )
        self._coarse_shape = (c0, h0, w0)

        c_in = spec.c_in
        self.synth_hidden = int(synth_hidden)
        # Tiny synthesis: 1x1 conv (c_in->hidden) -> GELU -> 1x1 conv (hidden->3).
        self.w1 = nn.Parameter(torch.randn(self.synth_hidden, c_in) * (1.0 / math.sqrt(c_in)))
        self.b1 = nn.Parameter(torch.zeros(self.synth_hidden))
        self.w2 = nn.Parameter(torch.randn(3, self.synth_hidden) * (1.0 / math.sqrt(self.synth_hidden)))
        self.b2 = nn.Parameter(torch.zeros(3))
        # A small head turning the coarse frame1-delta into the same c_in feature
        # space (so frame1 = synth(grids + upsampled delta contribution)).
        self.delta_proj = nn.Parameter(torch.randn(c_in, c0) * (1.0 / math.sqrt(c0)))

        self.arm = _ARM(hidden=8)

        # Stored-pose FiLM: a non-trainable per-pair pose buffer (charged ~1 KB)
        # + the tiny identity-at-init FiLM MLP over the synthesis HIDDEN channels.
        # The buffer is always allocated so the trainer / byte-close paths are
        # uniform; it is only CONSUMED when pose_film_enabled.
        self.register_buffer(
            "stored_pose", torch.zeros(self.n_pairs, self.pose_dim), persistent=True
        )
        if self.pose_film_enabled:
            self.pose_film = _PoseFiLM(
                pose_dim=self.pose_dim, channels=self.synth_hidden, hidden=int(film_hidden)
            )
        else:
            self.pose_film = None

    def set_stored_pose(self, pose: torch.Tensor) -> None:
        """Set the STORED per-pair 6-dim pose buffer (from the GT PoseNet pose).

        This is the Quantizr STORE-pose payload: the contest pose is *stored*
        (and later range-coded into the archive ~1 KB), not reconstructed from
        pixels. ``pose`` is ``(n_pairs, pose_dim)``.
        """
        with torch.no_grad():
            p = pose.detach().to(self.stored_pose.dtype)
            if p.shape != self.stored_pose.shape:
                raise ValueError(
                    f"stored_pose expects {tuple(self.stored_pose.shape)}, got {tuple(p.shape)}"
                )
            self.stored_pose.copy_(p)

    # -- synthesis (matches numpy reference) ---------------------------------

    def _upsample_concat(self, extra_coarse: torch.Tensor | None) -> torch.Tensor:
        """Bilinear-upsample all grids to out_hw and concat -> (c_in, H, W).

        ``extra_coarse`` (optional, shape (c_in, H, W)) is added to the
        concatenated feature (the frame1 delta contribution).
        """
        h, w = self.out_hw
        ups = []
        for g in self.latent_grids:
            gg = g.unsqueeze(0)  # (1,C,gh,gw)
            u = F.interpolate(gg, size=(h, w), mode="bilinear", align_corners=False)
            ups.append(u.squeeze(0))
        feat = torch.cat(ups, dim=0)  # (c_in, H, W)
        if extra_coarse is not None:
            feat = feat + extra_coarse
        return feat

    def _synth(
        self,
        feat: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """1x1 conv -> [FiLM] -> GELU -> 1x1 conv -> sigmoid; (3, H, W) in [0,1].

        When ``film`` is ``(gamma, beta)`` (each ``(hidden,)``), the hidden
        feature is modulated ``h -> gamma * h + beta`` BEFORE the GELU — the
        canonical FiLM injection point (matches the capstone bundle). gamma=1 /
        beta=0 reproduces the no-FiLM render exactly.
        """
        c_in, h, w = feat.shape
        flat = feat.reshape(c_in, h * w)
        hidden = self.w1 @ flat + self.b1[:, None]
        if film is not None:
            gamma, beta = film
            hidden = gamma[:, None] * hidden + beta[:, None]
        hidden = F.gelu(hidden)
        out = self.w2 @ hidden + self.b2[:, None]
        return torch.sigmoid(out).reshape(3, h, w)

    def reconstruct_pair(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rgb_0, rgb_1)`` each ``(B, 3, H, W)`` in ``[0, 255]``."""
        h, w = self.out_hw
        feat0 = self._upsample_concat(None)  # (c_in, H, W) shared frame0
        rgb0_single = self._synth(feat0) * 255.0  # (3, H, W)
        b = int(idx.shape[0])
        rgb0 = rgb0_single.unsqueeze(0).expand(b, 3, h, w)

        # frame1: add the per-pair coarse delta (projected into c_in, upsampled).
        deltas = self.frame1_delta[idx]  # (B, C, h0, w0)
        c0, h0, w0 = self._coarse_shape
        # project coarse channels -> c_in via delta_proj, then upsample.
        dflat = deltas.reshape(b, c0, h0 * w0)
        proj = torch.einsum("ec,bcn->ben", self.delta_proj, dflat)  # (B, c_in, h0*w0)
        proj = proj.reshape(b, self.spec.c_in, h0, w0)
        proj_up = F.interpolate(proj, size=(h, w), mode="bilinear", align_corners=False)
        # Stored-pose FiLM (per-pair gamma/beta over synthesis hidden channels).
        film_gamma = film_beta = None
        if self.pose_film_enabled and self.pose_film is not None:
            pose_b = self.stored_pose[idx]  # (B, pose_dim) STORED scalars
            film_gamma, film_beta = self.pose_film(pose_b)  # each (B, hidden)
        rgb1_list = []
        for bi in range(b):
            feat1 = feat0 + proj_up[bi]
            film_bi = (
                (film_gamma[bi], film_beta[bi]) if film_gamma is not None else None
            )
            rgb1_list.append(self._synth(feat1, film_bi) * 255.0)
        rgb1 = torch.stack(rgb1_list, dim=0)
        return rgb0, rgb1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        rgb0, rgb1 = self.reconstruct_pair(idx)
        return torch.stack([rgb0, rgb1], dim=1)

    # -- charged-byte accounting --------------------------------------------

    def latent_rate_bytes(self) -> float:
        """ARM-estimated entropy-coded bytes for ALL latents (the dominant payload).

        Shared grids are counted ONCE (global state); the per-pair frame1 deltas
        are summed over pairs. This is the differentiable rate term — but the
        returned float is detached (a measurement, not a loss).
        """
        with torch.no_grad():
            bits = torch.zeros((), dtype=torch.float64)
            for g in self.latent_grids:
                bits = bits + self.arm.rate_bits(g, quant_step=self.quant_step).double()
            c0, h0, w0 = self._coarse_shape
            for p in range(self.n_pairs):
                bits = bits + self.arm.rate_bits(
                    self.frame1_delta[p], quant_step=self.quant_step
                ).double()
            return float(bits.item()) / 8.0

    def latent_rate_bits_tensor(self) -> torch.Tensor:
        """Differentiable total latent bits (for an optional rate penalty)."""
        bits = self.latent_grids[0].sum() * 0.0
        for g in self.latent_grids:
            bits = bits + self.arm.rate_bits(g, quant_step=self.quant_step)
        for p in range(self.n_pairs):
            bits = bits + self.arm.rate_bits(
                self.frame1_delta[p], quant_step=self.quant_step
            )
        return bits

    def weight_param_count(self) -> int:
        """Synthesis + ARM + delta_proj (+ FiLM) param count (fixed payload)."""
        n = (
            self.w1.numel() + self.b1.numel() + self.w2.numel() + self.b2.numel()
            + self.delta_proj.numel() + self.arm.param_count()
        )
        if self.pose_film is not None:
            n += self.pose_film.param_count()
        return int(n)

    def weight_bytes(self) -> float:
        return self.weight_param_count() * self.bytes_per_param

    def stored_pose_bytes(self) -> float:
        """Charged bytes for the STORED per-pair pose (the Quantizr payload).

        Estimate at ``bytes_per_param`` (fp16=2) per scalar; the REAL byte cost
        is the range-coded count measured at byte-close time. With 6 dims x
        ``n_pairs`` this is the ~1 KB the pose-fold pays for d_pose collapse.
        """
        if not self.pose_film_enabled:
            return 0.0
        return float(self.n_pairs * self.pose_dim) * self.bytes_per_param

    def charged_bytes(self) -> dict[str, float]:
        """Honest charged-byte breakdown for the param-at-basin curve.

        ``total = latent_bytes + weight_bytes + stored_pose_bytes``. The latent
        bytes are the only term that scales with grid resolution (the conv-HNeRV
        comparison axis); ``stored_pose_bytes`` is the pose-fold payload.
        """
        latent = self.latent_rate_bytes()
        weight = self.weight_bytes()
        pose = self.stored_pose_bytes()
        return {
            "latent_bytes": latent,
            "weight_bytes": weight,
            "stored_pose_bytes": pose,
            "total_bytes": latent + weight + pose,
            "latent_count": float(self.spec.total_latent_count()),
            "weight_param_count": float(self.weight_param_count()),
        }
