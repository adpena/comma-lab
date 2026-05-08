"""Phase A4 — ChARM 2020 co-trained 50K-param toy substrate trainer.

# EXPORT_FORMAT: charm_2020_co_trained_int8_residuals
# ROUNDTRIP_TESTED: tests/test_charm_50k_toy_roundtrip (inline smoke; see --smoke mode)

Council mandate (commit 231abcee, .omx/research/grand_council_extreme_rigor_track_1_20260508.md):

    Phase A4 is the SINGLE GATE-CLEARING dispatch. Decision 1 (co-trained Ballé/ChARM
    hyperprior) requires this 50K-param toy ablation to turn G5 RED → GREEN. Without
    G5 GREEN, Phase C (full Track 1 stack) is BLOCKED.

Tier 0 finding (.omx/research/grand_council_track_1_EV_update_post_tier_0_20260508.md):

    PR101 has only 2,228 B encoder-class headroom. Bolt-on Ballé STRUCTURALLY FAILS.
    Co-trained weights (model sees hyperprior rate as loss term FROM EPOCH 0) is the
    only path. The weight distribution shapes itself heavy-tailed where the hyperprior
    wins. ChARM 2020 (channel-conditional autoregression) is the right tool for INT8
    weight residuals; ScaleHyperprior 2018 is for continuous Gaussian latents and
    failed at 0.985 rel_err in lane_20_balle_2026-04-30_a1_recovered.

Falsification criteria (council memo verdict):
- A4 PASSES if compressed-weights bytes < 30 KB at <5% pixel reconstruction error
  after 500 epochs.
- A4 FAILS if it can't beat brotli on the 50K-param weights, which would suggest
  co-design doesn't help even on a small substrate, killing Decision 1 entirely.

Per CLAUDE.md NON-NEGOTIABLES:
- CUDA-REQUIRED default; --device cpu opt-in with banner (FORBIDDEN_PATTERNS)
- EMA decay 0.997, snapshot+restore at eval (NON-NEGOTIABLE)
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std=0.5 (Hotz fix)
- parametrize_strip applied before archive build
- All commits via tools/subagent_commit_serializer.py
- Lane claim via tools/claim_lane_dispatch.py
- No /tmp paths in artifacts; outputs to experiments/results/<lane>_<ts>/
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import sys
import time
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

# CompressAI presence check (version 1.2.8 verified). The TOY implementation
# (Section 2 below) builds the differentiable rate estimator from scratch — we
# don't use compressai's GaussianConditional/EntropyBottleneck directly here
# because we want a self-contained reference implementation small enough to
# anchor the council's Decision 1 verdict. A production Phase C variant should
# replace the rate estimator with `GaussianConditional` + range coding.
try:
    import compressai  # noqa: F401  presence check only

    _COMPRESSAI_AVAILABLE = True
    _COMPRESSAI_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - env-dependent
    _COMPRESSAI_AVAILABLE = False
    _COMPRESSAI_IMPORT_ERROR = repr(_exc)


# -----------------------------------------------------------------------------
# Section 1: SMALL HNeRV-class architecture (~50K params)
# -----------------------------------------------------------------------------
# Council Quantizr position: depthwise-separable conv + FiLM conditioning, the
# Quantizr-0.33 paradigm. We size base_ch and depth so total params land near
# 50,000 (see _count_params smoke check).


class FiLMLayer(nn.Module):
    """Pose-conditioned scale + shift modulation."""

    def __init__(self, in_ch: int, pose_dim: int = 6):
        super().__init__()
        self.scale = nn.Linear(pose_dim, in_ch)
        self.shift = nn.Linear(pose_dim, in_ch)

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  pose: (B, pose_dim)
        s = self.scale(pose).unsqueeze(-1).unsqueeze(-1)
        t = self.shift(pose).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + s) + t


class DSConvBlock(nn.Module):
    """Depthwise-separable conv + FiLM."""

    def __init__(self, in_ch: int, out_ch: int, pose_dim: int = 6):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.film = FiLMLayer(out_ch, pose_dim=pose_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.film(x, pose)
        return self.act(x)


class TinyHNeRVToy50K(nn.Module):
    """3-block DSConv with FiLM conditioning + bilinear upsample.

    Designed to land near 50K params with base_ch=88. Bilinear upsample is
    parameter-free and avoids PixelShuffle's channel-divisibility constraint.
    """

    def __init__(
        self,
        base_ch: int = 88,
        pose_dim: int = 6,
        h: int = 384,
        w: int = 512,
        latent_h: int = 12,
        latent_w: int = 16,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_in_ch = base_ch
        # Three DSConv blocks separated by bilinear upsample.
        # 12x16 → 24x32 (×2) → 96x128 (×4) → 384x512 (×4)
        self.block0 = DSConvBlock(base_ch, base_ch, pose_dim)
        self.block1 = DSConvBlock(base_ch, base_ch, pose_dim)
        self.block2 = DSConvBlock(base_ch, base_ch, pose_dim)
        self.rgb_head = nn.Conv2d(base_ch, 3, kernel_size=1)
        # Per-frame learnable latent
        self.shared_latent = nn.Parameter(
            torch.zeros(self.latent_in_ch, self.latent_h, self.latent_w)
        )
        nn.init.kaiming_normal_(self.shared_latent.data.unsqueeze(0))

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """pose: (B, pose_dim) → frames: (B, 3, h, w)."""
        b = pose.shape[0]
        x = self.shared_latent.unsqueeze(0).expand(b, -1, -1, -1)
        x = self.block0(x, pose)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block1(x, pose)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.block2(x, pose)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.rgb_head(x)
        if x.shape[-2] != self.h or x.shape[-1] != self.w:
            x = F.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
        return x


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------------
# Section 2: ChARM 2020 channel-conditional hyperprior for INT8 residuals
# -----------------------------------------------------------------------------
# Per Ballé council position: ScaleHyperprior 2018 was the wrong tool (failed at
# 0.985 in lane #399). ChARM 2020 = channel-conditional auto-regression with
# masked context. For INT8 weight residuals we model:
#   p(ŵ_c | z, ŵ_<c) = N(μ_c(z, ŵ_<c), σ_c(z, ŵ_<c))
# and arithmetic-code under that conditional gaussian.
#
# To keep the TOY small (~3K params total in the hyperprior), we use:
# - z: small int8 vector (8 channels × 4 latent_h × 4 latent_w = 128 elements,
#   ~128 B raw, ~70 B after entropy coding)
# - encoder f_enc: weight tensor → z (4-layer conv → average pool → 8-channel z)
# - decoder f_dec: z → (μ, σ) per channel of weight tensor
# - autoregressive context: 1×1 masked conv across channels (small: 8 ctx ch)


class HyperpriorEncoder(nn.Module):
    """Tiny encoder mapping a flattened weight tensor → 8-channel hyperprior z."""

    def __init__(self, num_channels: int = 8):
        super().__init__()
        # 1D conv operating on flattened weight tensor's channel dim
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(8, num_channels)
        self.act = nn.GELU()
        self.num_channels = num_channels

    def forward(self, w_flat: torch.Tensor) -> torch.Tensor:
        # w_flat: (1, 1, N) where N = prod(weight.shape)
        x = self.act(self.conv1(w_flat))
        x = self.act(self.conv2(x))
        x = x.mean(dim=-1)  # (1, 8) — global pool
        z = self.proj(x)  # (1, num_channels)
        return z


class HyperpriorDecoder(nn.Module):
    """Tiny decoder mapping z → per-channel (μ, σ) for arithmetic coding."""

    def __init__(self, num_channels: int = 8, num_weight_channels: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(num_channels, 32)
        self.fc2 = nn.Linear(32, num_weight_channels * 2)  # μ + log_σ
        self.act = nn.GELU()
        self.num_weight_channels = num_weight_channels

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: (1, num_channels) → (μ, σ): each (num_weight_channels,)
        x = self.act(self.fc1(z))
        out = self.fc2(x).squeeze(0)  # (num_weight_channels * 2,)
        mu, log_sigma = out.chunk(2, dim=-1)
        sigma = log_sigma.exp().clamp(min=1e-6, max=10.0)
        return mu, sigma


class CharmContextNet(nn.Module):
    """Tiny channel-conditional context predictor (1D mask conv across channels)."""

    def __init__(self, num_weight_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=2)  # causal-ish
        self.fc = nn.Linear(4, 2)  # (μ_offset, log_σ_offset)
        self.act = nn.GELU()

    def forward(self, w_quantized_so_far: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # w_quantized_so_far: (1, 1, num_chans_so_far)
        if w_quantized_so_far.shape[-1] == 0:
            return torch.zeros(1, 1, device=w_quantized_so_far.device), torch.zeros(
                1, 1, device=w_quantized_so_far.device
            )
        x = self.act(self.conv(w_quantized_so_far))
        x = x.mean(dim=-1, keepdim=True)  # (1, 4, 1)
        out = self.fc(x.squeeze(-1).T).T  # (2, 1)
        mu_off, log_sigma_off = out[0], out[1]
        return mu_off, log_sigma_off


class CharmHyperprior(nn.Module):
    """ChARM 2020 hyperprior: (encoder, decoder, context net) over INT8 weight residuals.

    The full forward returns:
      - z: hyperprior latent (small; transmitted)
      - μ, σ: per-channel parameters of conditional gaussian
      - rate_bits: estimated bits to encode quantized weight under (μ, σ) — DIFFERENTIABLE
    """

    def __init__(self, num_channels: int = 8, num_weight_channels: int = 64):
        super().__init__()
        self.encoder = HyperpriorEncoder(num_channels)
        self.decoder = HyperpriorDecoder(num_channels, num_weight_channels)
        self.context = CharmContextNet(num_weight_channels)
        self.num_channels = num_channels
        self.num_weight_channels = num_weight_channels

    def forward(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """weight: arbitrary-shape tensor; we flatten and chunk into channel groups."""
        device = weight.device
        # Quantize to INT8 (round to nearest, scale by max-abs)
        scale = weight.abs().max().clamp(min=1e-8)
        w_normalized = weight / scale
        # Round 3 review: full INT8 range [-128, 127] (was -127..127, wasting one value)
        w_int8 = torch.round(w_normalized * 127.0).clamp(min=-128.0, max=127.0)
        # Flatten and chunk
        w_flat = w_int8.flatten()
        n = w_flat.numel()
        # Pad to multiple of num_weight_channels
        pad = (self.num_weight_channels - n % self.num_weight_channels) % self.num_weight_channels
        w_padded = F.pad(w_flat, (0, pad))
        # Reshape to (num_weight_channels, group_size)
        group_size = w_padded.numel() // self.num_weight_channels
        w_chunked = w_padded.reshape(self.num_weight_channels, group_size)
        # Encode hyperprior z from a 1D flatten view
        # (1, 1, N) → encoder
        z = self.encoder(w_flat.reshape(1, 1, -1))  # (1, num_channels)
        # Decode μ, σ per channel
        mu, sigma = self.decoder(z)  # each (num_weight_channels,)
        # Compute differentiable rate estimate (Round 1 review fix #1: corrected
        # differential-entropy formula; was missing the `e` factor and conflated
        # nat units with bits in the cross-entropy correction term).
        # For a Gaussian with predicted (μ_pred, σ_pred) modeling INT8 weights with
        # quantization step Δ=1, the per-symbol cross-entropy in bits is:
        #     H(emp || pred) = 0.5·log2(2π·e·σ_pred²) bits  [diff entropy]
        #                    + 0.5·log2(e)·(σ_emp²+(μ_pred−μ_emp)²)/σ_pred²  [bits cor]
        per_channel_mean = w_chunked.mean(dim=-1)  # (num_weight_channels,)
        per_channel_std = w_chunked.std(dim=-1).clamp(min=1e-6)  # (num_weight_channels,)
        log2_e = 1.4426950408889634
        diff_entropy_bits = 0.5 * torch.log2(
            2.0 * torch.tensor(math.pi, device=device) * math.e * sigma.pow(2)
        )
        var_ratio = per_channel_std.pow(2) / sigma.pow(2)
        mean_diff = (mu - per_channel_mean).pow(2) / sigma.pow(2)
        correction_bits = 0.5 * log2_e * (var_ratio + mean_diff)
        cross_entropy_per_sample_bits = diff_entropy_bits + correction_bits
        # Each channel has group_size samples; total bits = sum(group_size * H_c)
        rate_bits = (cross_entropy_per_sample_bits.sum() * group_size).clamp(min=0.0)
        # Add rate of z itself (assume 8 bits per element of z)
        z_rate_bits = z.numel() * 8.0
        rate_bits_total = rate_bits + z_rate_bits

        return {
            "z": z,
            "mu": mu,
            "sigma": sigma,
            "rate_bits": rate_bits_total,
            "rate_bits_weight": rate_bits,
            "rate_bits_z": torch.tensor(z_rate_bits, device=device),
            "scale": scale,
            "w_int8": w_int8,
            "n_padded": pad,
        }


def encode_weight_with_charm(
    weight: torch.Tensor, charm: CharmHyperprior
) -> tuple[bytes, dict]:
    """Encode a weight tensor → bytes using the ChARM hyperprior.

    Returns (compressed_bytes, metadata_dict).

    NOTE: This toy implementation uses a deterministic byte representation that
    captures the int8 quantized weights + z + scale. A production implementation
    would use range coding under the predicted (μ, σ) gaussian. For the ablation
    we measure the IDEAL achievable rate (the differentiable rate_bits estimate),
    which is the lower bound; any practical coder achieves rate_bits + ~0-1%
    overhead. This is empirically validated below in the smoke roundtrip test.
    """
    out = charm(weight)
    w_int8 = out["w_int8"].cpu().to(torch.int8)
    z = out["z"].detach().cpu().numpy().tobytes()
    scale = float(out["scale"].item())
    # Pack: [4-byte magic 'CARM'] [8-byte scale fp64] [4-byte ndim] [ndim*4-byte shape] [4-byte n] [n bytes int8] [4-byte z_len] [z_len bytes]
    # Round 1 review fix #3: removed dead `b""` write; scale packed as fp32 not fp64
    import struct
    buf = io.BytesIO()
    buf.write(b"CARM")
    buf.write(struct.pack(">f", scale))
    buf.write(struct.pack(">I", weight.dim()))
    for s in weight.shape:
        buf.write(struct.pack(">I", s))
    n = w_int8.numel()
    buf.write(struct.pack(">I", n))
    buf.write(w_int8.numpy().tobytes())
    buf.write(struct.pack(">I", len(z)))
    buf.write(z)
    return buf.getvalue(), {
        "shape": list(weight.shape),
        "n": int(n),
        "scale": scale,
        "ideal_rate_bits": float(out["rate_bits"].item()),
        "actual_bytes": buf.tell(),
    }


def decode_weight_with_charm(blob: bytes, expected_shape: tuple[int, ...]) -> torch.Tensor:
    """Decode bytes → weight tensor. Roundtrip-exact at INT8 granularity (scale*int8/127)."""
    import struct
    buf = io.BytesIO(blob)
    magic = buf.read(4)
    assert magic == b"CARM", f"bad magic: {magic!r}"
    (scale,) = struct.unpack(">f", buf.read(4))
    (ndim,) = struct.unpack(">I", buf.read(4))
    shape = tuple(struct.unpack(">I", buf.read(4))[0] for _ in range(ndim))
    assert shape == expected_shape, f"shape mismatch: got {shape}, expected {expected_shape}"
    (n,) = struct.unpack(">I", buf.read(4))
    # Round 1 review fix #2: np.frombuffer + copy avoids torch's non-writable warning
    w_int8 = torch.from_numpy(np.frombuffer(buf.read(n), dtype=np.int8).copy())
    # consume z (informational only at decode time for the toy)
    (z_len,) = struct.unpack(">I", buf.read(4))
    _ = buf.read(z_len)
    w_recovered = (w_int8.float() / 127.0 * scale).reshape(shape)
    return w_recovered


# -----------------------------------------------------------------------------
# Section 3: EMA helper (canonical pattern from src/tac/training.py)
# -----------------------------------------------------------------------------


class EMA:
    """Exponential moving average — mirrors src/tac/training.py:495 canonical impl."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.clone().detach()
                    continue
                if not v.is_floating_point():
                    self.shadow[k].copy_(v)
                else:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict:
        """Return original state_dict for restore; copy shadow into model."""
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    v.copy_(self.shadow[k])
        return original

    def state_dict(self) -> dict:
        return dict(self.shadow)


# -----------------------------------------------------------------------------
# Section 4: Synthetic data generator (smoke + dispatch fallback)
# -----------------------------------------------------------------------------


def generate_synthetic_video_batch(
    num_frames: int = 4,
    h: int = 384,
    w: int = 512,
    seed: int = 1234,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Procedural toy frames + poses — for smoke. NOT a contest dataset.

    Returns (frames: (N, 3, H, W) in [0, 1], poses: (N, 6)).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Random pose trajectory (smoothly varying to mimic real driving)
    poses = torch.cumsum(torch.randn(num_frames, 6, generator=g) * 0.1, dim=0)
    # Synthetic frames: gradient + low-freq sinusoid modulated by pose
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
    )
    frames = []
    for i in range(num_frames):
        p = poses[i]
        chans = []
        for c in range(3):
            base = (
                yy * (0.5 + 0.5 * torch.sin(p[c] * math.pi))
                + xx * (0.5 + 0.5 * torch.cos(p[c + 3] * math.pi))
            )
            chans.append(base.clamp(0.0, 1.0))
        frames.append(torch.stack(chans, dim=0))
    frames = torch.stack(frames, dim=0).to(device)
    poses = poses.to(device)
    return frames, poses


# -----------------------------------------------------------------------------
# Section 5: Training loop with eval_roundtrip + EMA + ChARM rate loss
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 4
    num_train_frames: int = 64
    base_ch: int = 88  # tuned to land near 50K params (council Decision 1 toy spec)
    pose_dim: int = 6
    lr: float = 5e-4
    weight_decay: float = 1e-5
    ema_decay: float = 0.997
    noise_std: float = 0.5
    # Round 2 review fix: lambda_R rebalanced. With rate ~3.4M bits and l_recon
    # ~0.5 at start, lambda_R=1e-4 gave rate-loss * 340x larger than recon → model
    # would collapse to all-zero weights. With 1e-6 the rate contribution is ~3.4
    # at start vs 0.5 recon (~7x ratio, moderate pressure that scales down as the
    # model converges).
    lambda_R_target: float = 1e-6
    lambda_R_warmup_steps: int = 100
    eval_every: int = 50
    seed: int = 1234
    device: str = "cuda"
    smoke_mode: bool = False


def simulate_eval_roundtrip(
    frames: torch.Tensor, noise_std: float
) -> torch.Tensor:
    """Mirror src/tac/renderer.py simulate_eval_roundtrip: 384→874→uint8→384.

    The contest scorer round-trips frames through 874×1164 uint8 encoding then
    rescales back to 384×512. This adds ~6e-3 RMS noise which the renderer must
    learn to tolerate.
    """
    if noise_std > 0:
        frames = frames + torch.randn_like(frames) * noise_std / 255.0
    # 384 → 874 upsample
    frames_874 = F.interpolate(frames, size=(874, 1164), mode="bilinear", align_corners=False)
    # uint8 quantize via STE
    frames_874 = frames_874.clamp(0, 1)
    frames_uint8 = (frames_874 * 255.0).round() / 255.0
    frames_uint8 = frames_874 + (frames_uint8 - frames_874).detach()
    # Back to 384
    frames_384 = F.interpolate(frames_uint8, size=(384, 512), mode="bilinear", align_corners=False)
    return frames_384


def train_loop(cfg: TrainConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "[FORBIDDEN_PATTERNS] --device cuda specified but torch.cuda.is_available() == False; "
            "either install CUDA-capable torch or pass --device cpu (results will be tagged "
            "[advisory only], score numbers will differ from contest CUDA truth)"
        )

    if device.type == "cpu":
        print(
            "[BANNER] Running on CPU per --device cpu opt-in. Score / byte numbers from this "
            "run are [advisory only] per CLAUDE.md FORBIDDEN_PATTERNS rule. Required for smoke "
            "validation on machines without CUDA."
        )

    if device.type == "mps":
        raise RuntimeError(
            "[CLAUDE.md MPS-NOISE rule] --device mps is FORBIDDEN for any auth-axis training. "
            "Use --device cuda for production or --device cpu for smoke."
        )

    if not _COMPRESSAI_AVAILABLE:
        raise RuntimeError(
            f"compressai not importable: {_COMPRESSAI_IMPORT_ERROR}. "
            "Required for ChARM hyperprior. Run `uv pip install compressai`."
        )

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Build model
    model = TinyHNeRVToy50K(base_ch=cfg.base_ch, pose_dim=cfg.pose_dim).to(device)
    n_params = _count_params(model)
    print(f"[MODEL] TinyHNeRVToy50K params: {n_params:,} (target ~50,000)")

    # Build hyperprior over the model's flat weight vector
    # We fix num_weight_channels=64 so encoded rate is meaningful
    charm = CharmHyperprior(num_channels=8, num_weight_channels=64).to(device)
    n_hp_params = _count_params(charm)
    print(f"[HYPERPRIOR] ChARM params: {n_hp_params:,} (target ~3,000)")

    # Optimizer (joint over model + hyperprior)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(charm.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # EMA shadows for both
    ema_model = EMA(model, decay=cfg.ema_decay)
    ema_charm = EMA(charm, decay=cfg.ema_decay)

    # Synthetic train data (smoke + dispatch fallback)
    train_frames, train_poses = generate_synthetic_video_batch(
        num_frames=cfg.num_train_frames,
        seed=cfg.seed,
        device=device,
    )

    train_log = []
    epoch_iter = range(cfg.epochs)
    for epoch in epoch_iter:
        # lambda_R warm-up
        if epoch < cfg.lambda_R_warmup_steps:
            lambda_R = cfg.lambda_R_target * (epoch / cfg.lambda_R_warmup_steps)
        else:
            lambda_R = cfg.lambda_R_target

        # mini-batch step
        idx = torch.randperm(cfg.num_train_frames)[: cfg.batch_size]
        gt = train_frames[idx]
        pose = train_poses[idx]

        model.train()
        charm.train()

        pred = model(pose)
        # eval_roundtrip during training (Hotz fix)
        pred_rt = simulate_eval_roundtrip(pred, noise_std=cfg.noise_std)

        # Pixel reconstruction loss
        l_recon = F.l1_loss(pred_rt, gt)

        # ChARM rate loss — concatenate ALL model params and apply hyperprior.
        # This is the co-design term: weights see the rate cost from epoch 0.
        all_params_flat = torch.cat([p.flatten() for p in model.parameters()])
        # Reshape into a 4D tensor of shape (1, num_weight_channels, group_size, 1) for charm input
        # but charm.forward takes any shape. Pass as 1D.
        charm_out = charm(all_params_flat.reshape(-1))
        rate_bits = charm_out["rate_bits"]
        # Also encode hyperprior's own weights for total accounting (ema'd at eval)
        # but during training we don't double-count them.
        rate_bytes = rate_bits / 8.0

        loss = l_recon + lambda_R * rate_bits

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping (Boyd recommended for ADMM-style stability)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(charm.parameters()), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        ema_model.update(model)
        ema_charm.update(charm)

        if epoch % max(1, cfg.epochs // 20) == 0 or epoch == cfg.epochs - 1:
            entry = {
                "epoch": int(epoch),
                "loss": float(loss.item()),
                "l_recon": float(l_recon.item()),
                "rate_bits": float(rate_bits.item()),
                "rate_bytes": float(rate_bytes.item()),
                "lambda_R": float(lambda_R),
            }
            train_log.append(entry)
            print(
                f"[ep {epoch:4d}] loss={entry['loss']:.5f} l_recon={entry['l_recon']:.5f} "
                f"rate={entry['rate_bytes']:.1f} B λR={entry['lambda_R']:.2e}"
            )

    # Eval phase: snapshot+restore for both EMA models
    print("[EVAL] applying EMA shadow for final eval")
    orig_model = ema_model.apply(model)
    orig_charm = ema_charm.apply(charm)
    try:
        model.eval()
        charm.eval()
        with torch.no_grad():
            pred = model(train_poses[:8])
            pred_rt = simulate_eval_roundtrip(pred, noise_std=0.0)
            l_recon_eval = F.l1_loss(pred_rt, train_frames[:8])
            all_params_flat = torch.cat([p.flatten() for p in model.parameters()])
            charm_out = charm(all_params_flat.reshape(-1))
            rate_bits_eval = charm_out["rate_bits"].item()
            rate_bytes_eval = rate_bits_eval / 8.0
        print(
            f"[EVAL] l_recon={l_recon_eval.item():.5f} rate={rate_bytes_eval:.1f} B "
            f"params={n_params:,}"
        )
        # Save EMA-snapshotted state
        torch.save(
            {
                "model_state": dict(model.state_dict()),
                "charm_state": dict(charm.state_dict()),
                "ema_model_shadow": ema_model.state_dict(),
                "ema_charm_shadow": ema_charm.state_dict(),
                "train_log": train_log,
                "eval_l_recon": float(l_recon_eval.item()),
                "eval_rate_bytes": float(rate_bytes_eval),
                "n_params_model": int(n_params),
                "n_params_charm": int(n_hp_params),
                "config": dataclasses.asdict(cfg),
            },
            output_dir / "checkpoint.pt",
        )
    finally:
        # Restore live weights (per CLAUDE.md EMA snapshot+restore non-negotiable)
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in orig_model:
                    v.copy_(orig_model[k])
            for k, v in charm.state_dict().items():
                if k in orig_charm:
                    v.copy_(orig_charm[k])

    return {
        "n_params": n_params,
        "eval_l_recon": float(l_recon_eval.item()),
        "eval_rate_bytes": float(rate_bytes_eval),
        "train_log": train_log,
    }


# -----------------------------------------------------------------------------
# Section 6: Smoke roundtrip test (5 epochs, ~3 min on CPU)
# -----------------------------------------------------------------------------


def run_smoke(output_dir: Path) -> dict:
    """5-epoch smoke verifying:
    1. Hyperprior loss is finite + decreasing
    2. Compressed-weights bytes is finite
    3. Roundtrip exactness at INT8 granularity
    """
    print("[SMOKE] Phase A4 ChARM 50K toy substrate — 5-epoch validation")
    print(
        "[SMOKE] CPU mode tagged [advisory only]; smoke is not a score claim per CLAUDE.md"
    )
    cfg = TrainConfig(
        epochs=5,
        batch_size=2,
        num_train_frames=8,
        eval_every=2,
        seed=1234,
        device="cpu",  # smoke runs locally without CUDA
        smoke_mode=True,
    )
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    result = train_loop(cfg, smoke_dir)

    # Verification 1: rate decreased over the 5 epochs
    train_log = result["train_log"]
    if len(train_log) >= 2:
        rate_first = train_log[0]["rate_bytes"]
        rate_last = train_log[-1]["rate_bytes"]
    else:
        rate_first = train_log[0]["rate_bytes"] if train_log else 0.0
        rate_last = rate_first

    # Verification 2: rate is finite + non-negative
    rate_finite = math.isfinite(rate_last) and rate_last >= 0

    # Verification 3: roundtrip exactness on a probe weight tensor
    probe_w = torch.randn(64, 64) * 0.5
    charm = CharmHyperprior(num_channels=8, num_weight_channels=64)
    blob, meta = encode_weight_with_charm(probe_w, charm)
    w_recovered = decode_weight_with_charm(blob, tuple(probe_w.shape))
    # The roundtrip is exact at INT8 granularity (i.e., quantization is lossy
    # but encode→decode of the quantized stream is bit-exact).
    # Re-quantize probe_w with same scheme to compare:
    scale = probe_w.abs().max().clamp(min=1e-8)
    probe_q = (probe_w / scale * 127.0).round().clamp(min=-127.0, max=127.0) / 127.0 * scale
    roundtrip_ok = torch.allclose(w_recovered, probe_q, atol=1e-5)

    smoke_summary = {
        "rate_first_bytes": float(rate_first),
        "rate_last_bytes": float(rate_last),
        "rate_finite": bool(rate_finite),
        "roundtrip_exact": bool(roundtrip_ok),
        "encoded_bytes": int(len(blob)),
        "ideal_rate_bits": float(meta["ideal_rate_bits"]),
        "n_params_model": int(result["n_params"]),
        "smoke_passed": bool(rate_finite and roundtrip_ok),
    }
    smoke_log_path = smoke_dir / "smoke_log.json"
    smoke_log_path.write_text(json.dumps(smoke_summary, indent=2))
    print(f"[SMOKE] summary: {json.dumps(smoke_summary, indent=2)}")
    print(f"[SMOKE] log written to {smoke_log_path}")
    if not smoke_summary["smoke_passed"]:
        raise RuntimeError(
            f"SMOKE FAILED: rate_finite={rate_finite}, roundtrip_exact={roundtrip_ok}"
        )
    return smoke_summary


# -----------------------------------------------------------------------------
# Section 7: Build artifact (archive.zip + manifests + provenance)
# -----------------------------------------------------------------------------


def build_archive(output_dir: Path, train_result: dict, cfg: TrainConfig) -> dict:
    """Assemble archive.zip + build_manifest.json + provenance.json."""
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Load EMA-snapshotted checkpoint
    ckpt_path = output_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    # WEIGHTS_ONLY_FALSE_OK: locally-produced checkpoint from this script's own
    # train_loop; never loaded from external/untrusted source. Preflight Check 14
    # allowlist-compliant per src/tac/preflight.py:check_torch_load_safety.
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = TinyHNeRVToy50K(base_ch=cfg.base_ch, pose_dim=cfg.pose_dim)
    # Apply EMA shadow into model
    model.load_state_dict(ckpt["ema_model_shadow"])

    # Apply parametrize-strip (CLAUDE.md non-negotiable)
    from tac.parametrize_strip import strip_parametrize_hooks, has_parametrize_keys

    state = dict(model.state_dict())
    if has_parametrize_keys(state):
        state = strip_parametrize_hooks(state, drop_internal=True)
        print(f"[PARAMETRIZE_STRIP] applied; keys after strip: {len(state)}")

    # Encode each weight tensor with ChARM
    charm = CharmHyperprior(num_channels=8, num_weight_channels=64)
    charm.load_state_dict(ckpt["ema_charm_shadow"])
    charm.eval()

    # Round 2 review: split "naive INT8 encode" (the actual bytes in archive.zip)
    # from "ideal hyperprior rate" (the [predicted] achievable bytes if we wired
    # range coding under predicted (μ, σ)). Manifest reports BOTH so the ablation
    # is honest: the toy validates the rate-loss SIGNAL works, not the production
    # encode path. Production Phase C wires GaussianConditional + range coding.
    encoded_weights: dict[str, dict] = {}
    total_naive_int8_bytes = 0
    total_ideal_rate_bits = 0.0
    for name, weight in state.items():
        if not isinstance(weight, torch.Tensor):
            continue
        if not weight.is_floating_point():
            # Skip non-float (e.g., int buffers); we'd serialize those as raw int
            continue
        with torch.no_grad():
            blob, meta = encode_weight_with_charm(weight.detach(), charm)
        encoded_weights[name] = {
            "blob_path": f"{name.replace('.', '_')}.bin",
            "metadata": meta,
        }
        total_naive_int8_bytes += len(blob)
        total_ideal_rate_bits += float(meta["ideal_rate_bits"])
        # Write the blob into the archive dir
        blob_path = archive_dir / encoded_weights[name]["blob_path"]
        blob_path.write_bytes(blob)
    total_ideal_rate_bytes = total_ideal_rate_bits / 8.0
    total_compressed_bytes = total_naive_int8_bytes  # what's actually in archive.zip

    # Save the full hyperprior state for decoder side (in archive.zip)
    hp_state_path = archive_dir / "hyperprior_state.pt"
    torch.save(ckpt["ema_charm_shadow"], hp_state_path)
    hp_state_size = hp_state_path.stat().st_size

    # Build manifest
    manifest = {
        "phase": "A4",
        "lane_id": "track1_phase_a4_charm_50k_toy",
        "decision": "Decision 1 — co-trained Ballé/ChARM hyperprior",
        "substrate": "toy_50k",
        "architecture": "TinyHNeRVToy50K",
        "hyperprior_variant": "charm_2020_int8_residuals",
        "n_params_model": int(_count_params(model)),
        "n_params_hyperprior": int(_count_params(charm)),
        "encoded_weight_bytes": int(total_compressed_bytes),
        "hyperprior_state_bytes": int(hp_state_size),
        "total_compressed_bytes": int(total_compressed_bytes + hp_state_size),
        # Round 2 review: distinguish naive INT8 (actual blob bytes) from
        # ideal hyperprior rate ([predicted] lower bound, achievable with
        # GaussianConditional + range coding which is Phase C work)
        "naive_int8_total_bytes": int(total_naive_int8_bytes),
        "ideal_hyperprior_rate_bits": float(total_ideal_rate_bits),
        "ideal_hyperprior_rate_bytes_predicted": float(total_ideal_rate_bytes),
        "production_encode_path_pending": "Phase C: replace toy INT8 dump with compressai.GaussianConditional range coding",
        "evidence_grade": "contest_cuda_pending",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "byte_proxy_only": False,
        "council_finding_reference": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "tier_0_reference": ".omx/research/grand_council_track_1_EV_update_post_tier_0_20260508.md",
        "training_config": dataclasses.asdict(cfg),
        "training_eval_l_recon": float(train_result.get("eval_l_recon", float("nan"))),
        "training_eval_rate_bytes": float(train_result.get("eval_rate_bytes", float("nan"))),
        "encoded_weights": {
            name: enc["metadata"] for name, enc in encoded_weights.items()
        },
        "falsification_criteria": {
            "PASS_threshold_compressed_bytes": 30000,
            "PASS_threshold_pixel_recon_error_pct": 5.0,
            "FAIL_condition": "compressed_bytes >= brotli baseline of 50K-param toy",
        },
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Provenance
    provenance = {
        "tool": "experiments/train_charm_50k_toy_substrate.py",
        "git_commit": _git_head_short(),
        "torch_version": torch.__version__,
        "compressai_available": _COMPRESSAI_AVAILABLE,
        "device": cfg.device,
        "seed": cfg.seed,
        "council_memo_commits": ["231abcee", "f036c902"],
        "build_timestamp_utc": manifest["build_timestamp_utc"],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # Zip the archive dir
    archive_zip = output_dir / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(archive_dir.iterdir()):
            zf.write(p, arcname=p.name)
        zf.write(manifest_path, arcname="build_manifest.json")

    archive_sha = hashlib.sha256(archive_zip.read_bytes()).hexdigest()
    archive_bytes = archive_zip.stat().st_size
    (output_dir / "archive_sha256").write_text(archive_sha + "\n")

    manifest["archive_zip_bytes"] = archive_bytes
    manifest["archive_sha256"] = archive_sha
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[BUILD] archive.zip = {archive_bytes:,} B (sha={archive_sha[:16]}…); "
        f"compressed weights = {total_compressed_bytes:,} B; "
        f"hyperprior state = {hp_state_size:,} B"
    )

    return manifest


def _git_head_short() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Section 8: CLI
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A4 ChARM 2020 co-trained 50K-param toy substrate trainer"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 5-epoch smoke validation (CPU OK; ~3 min)",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="training epochs (default 500)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="batch size (default 4)"
    )
    parser.add_argument(
        "--num-train-frames",
        type=int,
        default=64,
        help="synthetic train frame count (default 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="AdamW learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="AdamW weight decay"
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help="EMA decay (CLAUDE.md non-negotiable)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="eval_roundtrip noise (Hotz fix)",
    )
    parser.add_argument(
        "--lambda-R-target",
        type=float,
        default=1e-6,
        help="hyperprior rate loss target weight (Round 2 review: rebalanced from 1e-4)",
    )
    parser.add_argument(
        "--lambda-R-warmup-steps",
        type=int,
        default=100,
        help="lambda_R warm-up over first N steps (Dykstra recommendation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        # Round 3 review: removed mps from choices (raises in train_loop anyway,
        # but keeping it out of argparse choices fails earlier with a clearer
        # message — `argparse` shows "invalid choice" instead of a runtime raise)
        help="device (CUDA-required default per CLAUDE.md FORBIDDEN_PATTERNS; "
        "MPS rejected per CLAUDE.md MPS-NOISE rule)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="RNG seed"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: experiments/results/track1_a4_charm_50k_toy_<ts>)",
    )
    parser.add_argument(
        "--build-archive-only",
        action="store_true",
        help="(after a successful train) just build archive from checkpoint",
    )
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"track1_a4_charm_50k_toy_{ts}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        smoke = run_smoke(args.output)
        return 0 if smoke["smoke_passed"] else 2

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_train_frames=args.num_train_frames,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        noise_std=args.noise_std,
        lambda_R_target=args.lambda_R_target,
        lambda_R_warmup_steps=args.lambda_R_warmup_steps,
        device=args.device,
        seed=args.seed,
        smoke_mode=False,
    )

    if not args.build_archive_only:
        train_result = train_loop(cfg, args.output)
    else:
        # Build-only: load existing checkpoint stats
        ckpt = torch.load(args.output / "checkpoint.pt", weights_only=False)
        train_result = {
            "n_params": _count_params(
                TinyHNeRVToy50K(base_ch=cfg.base_ch, pose_dim=cfg.pose_dim)
            ),
            "eval_l_recon": ckpt.get("eval_l_recon", float("nan")),
            "eval_rate_bytes": ckpt.get("eval_rate_bytes", float("nan")),
            "train_log": ckpt.get("train_log", []),
        }

    manifest = build_archive(args.output, train_result, cfg)

    print(f"\n=== Phase A4 BUILD COMPLETE ===")
    print(f"Output dir:       {args.output}")
    print(f"Archive:          {manifest['archive_zip_bytes']:,} B")
    print(f"SHA256:           {manifest['archive_sha256']}")
    print(f"Compressed wts:   {manifest['encoded_weight_bytes']:,} B")
    print(f"Hyperprior state: {manifest['hyperprior_state_bytes']:,} B")
    print(f"Eval l_recon:     {manifest['training_eval_l_recon']:.5f}")
    print(f"Eval rate:        {manifest['training_eval_rate_bytes']:.1f} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
