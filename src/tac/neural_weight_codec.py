"""Lane J-NWC: Neural Weight Compression for renderer.bin.

Reference: arXiv 2510.11234 — "Neural Weight Compression for Language Models"
(late 2025). Trains a neural codec on weight tensors themselves. SOTA 4-6
bits/weight with smooth tradeoff. Paper claims gains extend across diverse
architectures including vision encoders — fits our 80-100K param renderer
regime.

Mechanism summary:
  * Pretrain a tiny VQ-VAE-style weight codec on a corpus of small-renderer
    checkpoints (we have hundreds saved under experiments/results/).
  * The codec maps each Conv2d kernel-shaped tensor (or a flattened slice
    thereof) → discrete codebook indices → reconstructed tensor.
  * At deploy time, encode the final renderer.bin to NWC1 binary format
    (4 bits/weight target → ~−126KB rate gain on a 88K-param renderer).

Architecture (this module):
  * BLOCK_SIZE flat-element windows (default 16).
  * Encoder MLP:  block (Bs) → hidden (64) → hidden (32) → latent (16).
  * Decoder MLP:  latent (16) → hidden (32) → hidden (64) → block (Bs).
  * VQ codebook of K=64 codes × 16 dims. Look-up is nearest-codebook by L2.
  * Vector-quantization commitment loss (Van den Oord 2017 STE).
  * Per-block float16 scale (computed from |w|.amax) — codec operates on
    unit-normalized blocks so a single codebook generalizes across layer
    magnitudes.

Storage cost of one block:
    1 × float16 scale (2 bytes) + 1 × uint8 code index (1 byte) = 3 bytes
    per BLOCK_SIZE (16) elements ≈ 1.5 bits/weight nominal.

The 4-bits/weight target is achieved when scales are amortized over larger
blocks; we keep BLOCK_SIZE=16 for the default test config because rate gain
scales with block size and small renderers have small layers. Operators can
override BLOCK_SIZE at codec construction time.

NOT a replacement for SCv1 / OMG1 — Lane J-NWC is a separate compositional
lane (per CLAUDE.md "no premature convergence" rule).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.neural_weight_corpus import build_corpus_from_checkpoints


__all__ = [
    "WeightCodec",
    "WeightCodecConfig",
    "train_codec",
    "tensor_to_blocks",
    "blocks_to_tensor",
    "build_corpus_from_checkpoints",
]


# ── Config / constants ────────────────────────────────────────────────────


@dataclass
class WeightCodecConfig:
    """Static config for a NWC weight codec.

    Defaults are sized for the comma video compression challenge renderer:
        block_size=16    → 16 weight elements per code → ~1.5 bits/weight nominal
        codebook_size=64 → uint8 code index fits in 1 byte
        latent_dim=16    → codec inner dimension
        hidden=64        → encoder/decoder MLP hidden width

    Total codec param count: ~16K (encoder + decoder + codebook). The codec
    itself is trained once on a corpus and SHIPS WITHIN the archive only when
    operating in the "ship the codec" mode (not the default — see
    Lane J-NWC compress-time docs).
    """

    block_size: int = 16
    codebook_size: int = 64
    latent_dim: int = 16
    hidden: int = 64


# ── Codec module ──────────────────────────────────────────────────────────


class WeightCodec(nn.Module):
    """Tiny VQ-VAE-style codec for weight tensor blocks.

    Forward path (training):
        x_block       (B, block_size)   — unit-normalized weight block
        ↓ encoder
        z_e           (B, latent_dim)
        ↓ VQ lookup (nearest codebook entry)
        z_q           (B, latent_dim)
        ↓ decoder
        x_recon       (B, block_size)

    The VQ lookup is non-differentiable; the standard VQ-VAE STE
    ``z_q = z_e + (z_q - z_e).detach()`` lets gradients flow back to the
    encoder. Codebook is updated via a commitment loss
    (β * ||z_e - sg(z_q)||² + ||sg(z_e) - z_q||²).
    """

    def __init__(self, config: WeightCodecConfig | None = None):
        super().__init__()
        self.config = config or WeightCodecConfig()
        cfg = self.config

        if cfg.block_size <= 0 or cfg.codebook_size <= 0 or cfg.latent_dim <= 0:
            raise ValueError(
                f"All codec dims must be positive: {cfg}"
            )
        if cfg.codebook_size > 256:
            raise ValueError(
                f"codebook_size > 256 cannot fit in uint8 code; got {cfg.codebook_size}"
            )

        # 3-layer encoder MLP (per spec)
        self.encoder = nn.Sequential(
            nn.Linear(cfg.block_size, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden // 2, cfg.latent_dim),
        )

        # 3-layer decoder MLP (per spec)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden // 2, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.block_size),
        )

        # VQ codebook (initialized from N(0, 0.1) so distances are non-degenerate)
        self.codebook = nn.Parameter(
            torch.randn(cfg.codebook_size, cfg.latent_dim) * 0.1
        )

        self.commitment_beta = 0.25  # Van den Oord 2017 default

    # ── Encode / decode (latent space) ────────────────────────────────

    def _quantize_latent(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Snap encoder output to nearest codebook entry.

        Returns:
            z_q:        (B, latent_dim) quantized latent (non-STE)
            indices:    (B,) long tensor of codebook indices
            distances:  (B, K) squared L2 distances (used for analytics + loss)
        """
        # squared L2 distance from each block to each codebook entry
        # (B, K) = sum(z² , dim=1, keep) - 2 z @ C.T + sum(C², dim=1)
        z_sq = (z_e ** 2).sum(dim=-1, keepdim=True)            # (B, 1)
        c_sq = (self.codebook ** 2).sum(dim=-1).unsqueeze(0)    # (1, K)
        cross = z_e @ self.codebook.T                           # (B, K)
        distances = z_sq + c_sq - 2.0 * cross
        indices = distances.argmin(dim=-1)                     # (B,)
        z_q = self.codebook.index_select(0, indices)           # (B, latent_dim)
        return z_q, indices, distances

    def forward(self, x_block: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode → quantize → decode a normalized weight block.

        Returns dict with:
            recon:        (B, block_size) reconstruction
            z_e:          (B, latent_dim) encoder output (pre-VQ)
            z_q:          (B, latent_dim) quantized latent (post-STE)
            indices:      (B,) codebook indices
            commit_loss:  scalar VQ commitment + codebook loss
        """
        z_e = self.encoder(x_block)
        z_q, indices, _ = self._quantize_latent(z_e)
        # VQ-VAE STE: copy gradient from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()
        recon = self.decoder(z_q_st)
        # Commitment + codebook losses
        commit_loss = (
            F.mse_loss(z_e, z_q.detach())
            + self.commitment_beta * F.mse_loss(z_q, z_e.detach())
        )
        return {
            "recon": recon,
            "z_e": z_e,
            "z_q": z_q_st,
            "indices": indices,
            "commit_loss": commit_loss,
        }

    # ── Tensor-level encode / decode with bytes serialization ──────────

    def encode(self, tensor: torch.Tensor) -> bytes:
        """Quantize + serialize a single tensor.

        Returns bytes laid out as:
            [4 B] little-endian uint32 ndim
            [4*ndim B] little-endian uint32 shape entries
            [4 B] little-endian uint32 num_blocks
            [num_blocks * 2 B] float16 per-block scales
            [num_blocks * 1 B] uint8 codebook indices
            [tail B] float16 leftover-tail elements (numel %% block_size)

        The leftover tail is stored verbatim (rare; happens only when a
        layer's flat numel is not a multiple of block_size).
        """
        if not torch.is_floating_point(tensor):
            raise TypeError(f"WeightCodec.encode expects floating tensor, got {tensor.dtype}")
        device = next(self.parameters()).device
        t = tensor.detach().to(device).float()
        flat = t.reshape(-1)
        Bs = self.config.block_size
        N = flat.numel()
        n_blocks = N // Bs
        tail_n = N - n_blocks * Bs

        scales: torch.Tensor
        codes: torch.Tensor
        if n_blocks == 0:
            scales = torch.zeros(0, dtype=torch.float32, device=device)
            codes = torch.zeros(0, dtype=torch.long, device=device)
        else:
            blocks = flat[: n_blocks * Bs].reshape(n_blocks, Bs)
            scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
            blocks_norm = blocks / scales.unsqueeze(1)
            with torch.no_grad():
                z_e = self.encoder(blocks_norm)
                _, codes, _ = self._quantize_latent(z_e)

        tail = flat[n_blocks * Bs :] if tail_n > 0 else torch.zeros(0, device=device)

        # serialize
        buf = bytearray()
        shape = list(tensor.shape)
        buf.extend(struct.pack("<I", len(shape)))
        for s in shape:
            buf.extend(struct.pack("<I", int(s)))
        buf.extend(struct.pack("<I", int(n_blocks)))
        buf.extend(scales.cpu().to(torch.float16).numpy().tobytes())
        buf.extend(codes.cpu().to(torch.uint8).numpy().tobytes())
        # tail floats stored as float16
        buf.extend(tail.cpu().to(torch.float16).numpy().tobytes())
        return bytes(buf)

    def decode(self, blob: bytes) -> torch.Tensor:
        """Deserialize + dequantize a tensor previously produced by `encode`.

        Returns a tensor on CPU with the original shape and float32 dtype.
        """
        device = next(self.parameters()).device
        Bs = self.config.block_size
        offset = 0

        ndim = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        if ndim == 0 or ndim > 8:
            raise ValueError(f"NWC.decode: implausible ndim={ndim}")
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", blob, offset)[0])
            offset += 4
        n_blocks = struct.unpack_from("<I", blob, offset)[0]
        offset += 4

        scales_nbytes = n_blocks * 2  # float16
        scales_buf = blob[offset : offset + scales_nbytes]
        offset += scales_nbytes
        codes_nbytes = n_blocks * 1  # uint8
        codes_buf = blob[offset : offset + codes_nbytes]
        offset += codes_nbytes

        # Reconstruct expected numel from shape; tail = numel - n_blocks*Bs
        numel = 1
        for s in shape:
            numel *= int(s)
        tail_n = numel - n_blocks * Bs
        if tail_n < 0:
            raise ValueError(
                f"NWC.decode: negative tail (n_blocks={n_blocks}*Bs={Bs} > numel={numel})"
            )
        tail_nbytes = tail_n * 2  # float16
        tail_buf = blob[offset : offset + tail_nbytes]
        offset += tail_nbytes

        if n_blocks > 0:
            import numpy as np
            scales = torch.from_numpy(
                np.frombuffer(scales_buf, dtype=np.float16).copy()
            ).to(device=device, dtype=torch.float32)
            codes = torch.from_numpy(
                np.frombuffer(codes_buf, dtype=np.uint8).copy()
            ).to(device=device, dtype=torch.long)
            with torch.no_grad():
                z_q = self.codebook.index_select(0, codes)            # (n_blocks, latent_dim)
                recon_norm = self.decoder(z_q)                        # (n_blocks, block_size)
                recon_blocks = recon_norm * scales.unsqueeze(1)
            flat_recon = recon_blocks.reshape(-1)
        else:
            flat_recon = torch.zeros(0, device=device, dtype=torch.float32)

        if tail_n > 0:
            import numpy as np
            tail = torch.from_numpy(
                np.frombuffer(tail_buf, dtype=np.float16).copy()
            ).to(device=device, dtype=torch.float32)
        else:
            tail = torch.zeros(0, device=device, dtype=torch.float32)

        full = torch.cat([flat_recon, tail], dim=0)
        if full.numel() != numel:
            raise ValueError(
                f"NWC.decode size mismatch: got {full.numel()}, expected {numel}"
            )
        return full.reshape(*shape).cpu()


# ── Block / corpus utilities ──────────────────────────────────────────────


def tensor_to_blocks(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flat tensor view into (n_blocks, block_size) + per-block scales.

    Tail elements (numel %% block_size) are dropped — caller handles them
    separately. Useful for assembling training corpora.
    """
    flat = tensor.detach().reshape(-1).float()
    n_blocks = flat.numel() // block_size
    if n_blocks == 0:
        return (
            torch.zeros(0, block_size, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )
    blocks = flat[: n_blocks * block_size].reshape(n_blocks, block_size)
    scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
    return blocks / scales.unsqueeze(1), scales


def blocks_to_tensor(
    blocks_norm: torch.Tensor, scales: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """Inverse of tensor_to_blocks (no tail handling)."""
    blocks = blocks_norm * scales.unsqueeze(1)
    flat = blocks.reshape(-1)
    return flat.reshape(*shape)


# ── Training loop ─────────────────────────────────────────────────────────


def train_codec(
    corpus: torch.Tensor,
    codec: WeightCodec | None = None,
    *,
    num_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    log_interval: int = 100,
    seed: int = 1234,
) -> tuple[WeightCodec, list[float]]:
    """Train a `WeightCodec` on a corpus of unit-normalized weight blocks.

    Args:
        corpus:     (N, block_size) float tensor of normalized blocks
        codec:      optional pre-existing codec (otherwise constructed from
                    the corpus block_size with default config).
        num_steps:  number of optimizer steps
        batch_size: blocks per step
        lr:         AdamW learning rate
        device:     compute device
        log_interval: steps between log lines (set ≥ num_steps to silence).

    Returns: (trained codec, list of step-wise total-loss values).
    """
    if corpus.dim() != 2:
        raise ValueError(
            f"train_codec corpus must be 2-D (n_blocks, block_size), "
            f"got shape {tuple(corpus.shape)}"
        )
    block_size = corpus.shape[1]

    if codec is None:
        codec = WeightCodec(WeightCodecConfig(block_size=block_size))
    elif codec.config.block_size != block_size:
        raise ValueError(
            f"codec.config.block_size ({codec.config.block_size}) does not "
            f"match corpus block_size ({block_size})"
        )

    device = torch.device(device)
    codec = codec.to(device)
    corpus = corpus.to(device)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    opt = torch.optim.AdamW(codec.parameters(), lr=lr)

    losses: list[float] = []
    n = corpus.shape[0]
    for step in range(num_steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        x = corpus[idx]
        out = codec(x)
        recon_loss = F.mse_loss(out["recon"], x)
        loss = recon_loss + out["commit_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % log_interval == 0 or step == num_steps - 1:
            print(
                f"[nwc-train] step={step:5d} "
                f"recon={recon_loss.detach().item():.6f} "
                f"commit={out['commit_loss'].detach().item():.6f} "
                f"total={loss.item():.6f}"
            )
    return codec, losses
