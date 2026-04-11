"""VQ-VAE codec: learned discrete latent representation for video frames.

Instead of: GT -> SegNet -> 5-class masks -> AV1 -> renderer -> RGB
We do: GT -> VQ Encoder -> discrete codes -> entropy coding -> VQ Decoder -> RGB

The VQ-VAE learns what information to keep (task-relevant for PoseNet/SegNet)
and what to discard, rather than relying on SegNet's fixed 5-class segmentation.

Key advantage: the learned codebook captures texture gradients, lighting,
small objects, and road marking detail that 5-class masks cannot represent.

Architecture:
    Encoder: 384x512 -> 96x128 -> 24x32  (strided convolutions)
    VQ Layer: 24x32 spatial positions each select from K=512 codebook vectors
    Decoder: 24x32 -> 96x128 -> 384x512  (transposed convolutions)

Bitrate budget:
    Raw: 24*32*9bits = 864 bytes/frame, 1200 frames = 1.04MB
    With temporal delta coding: ~100-200KB
    With arithmetic coding on deltas: ~50-100KB

Archive budget:
    Codebook (K=512, D=64): ~32KB
    Decoder (~200KB FP4)
    Codes (~100KB compressed)
    Total: ~332KB

Modules:
    - VectorQuantizer: codebook lookup with straight-through estimator
    - VQEncoder: RGB -> continuous latent -> quantized codes
    - VQDecoder: quantized codes -> RGB reconstruction
    - VQVAE: full encoder-decoder with VQ bottleneck
    - VQVAEPairGenerator: produces scorer-compatible frame pairs
    - TemporalDeltaCoder: compresses code indices via temporal differencing
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vector Quantization Layer ──────────────────────────────────────────


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator.

    Each spatial position in the latent map selects the nearest codebook
    vector. Gradients flow through via the straight-through trick:
    forward uses quantized vectors, backward uses continuous encoder output.

    Uses EMA codebook updates (more stable than gradient-based) when
    ema_update=True during training. Falls back to loss-based updates
    when EMA is disabled.

    Args:
        num_embeddings: codebook size K (default 512)
        embedding_dim: dimension D of each codebook vector (default 64)
        commitment_cost: weight for commitment loss (encoder -> codebook)
        ema_decay: decay for EMA codebook updates (0 = disabled)
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA tracking for codebook updates
        if ema_decay > 0:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("_ema_w", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent to nearest codebook entries.

        Args:
            z: (B, D, H, W) continuous latent from encoder

        Returns:
            z_q_st: (B, D, H, W) quantized latent (straight-through)
            vq_loss: scalar VQ loss (codebook + commitment)
            indices: (B, H, W) integer codebook indices
        """
        B, D, H, W = z.shape

        # Reshape: (B, D, H, W) -> (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # Compute distances: ||z - e||^2 = ||z||^2 - 2*z*e^T + ||e||^2
        # This is more numerically stable than torch.cdist for large codebooks
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )

        # Find nearest codebook entry
        indices = d.argmin(dim=1)  # (B*H*W,)

        # Look up quantized vectors
        z_q = self.embedding(indices)  # (B*H*W, D)

        # EMA codebook update (training only)
        if self.training and self.ema_decay > 0:
            with torch.no_grad():
                # One-hot encodings for cluster assignment
                encodings = F.one_hot(indices, self.K).float()  # (B*H*W, K)
                # Update cluster sizes
                self._ema_cluster_size.mul_(self.ema_decay).add_(encodings.sum(0), alpha=1 - self.ema_decay)
                # Update cluster centroids
                dw = encodings.t() @ z_flat  # (K, D)
                self._ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                # Laplace smoothing to avoid empty clusters
                n = self._ema_cluster_size.sum()
                cluster_size = (self._ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
                self.embedding.weight.data.copy_(self._ema_w / cluster_size.unsqueeze(1))

        # Reshape back: (B*H*W, D) -> (B, H, W, D) -> (B, D, H, W)
        z_q = z_q.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Losses
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: forward uses z_q, backward uses z
        z_q_st = z + (z_q - z).detach()

        # Reshape indices for output
        indices = indices.reshape(B, H, W)

        return z_q_st, vq_loss, indices

    def indices_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors from indices (for decode-only path).

        Args:
            indices: (B, H, W) integer indices in [0, K)

        Returns:
            (B, D, H, W) codebook vectors
        """
        B, H, W = indices.shape
        z_q = self.embedding(indices.reshape(-1))  # (B*H*W, D)
        return z_q.reshape(B, H, W, self.D).permute(0, 3, 1, 2)

    def codebook_usage(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in a batch (diagnostic)."""
        unique = indices.unique().numel()
        return unique / self.K


# ── VQ Encoder ─────────────────────────────────────────────────────────


class VQEncoder(nn.Module):
    """Encode RGB frames to continuous latent representations.

    Architecture:
        384x512 x 3ch -> 192x256 x base_ch  (stride-2 conv)
        192x256 -> 96x128 x base_ch*2       (stride-2 conv)
        96x128 -> 48x64 x base_ch*4         (stride-2 conv)
        48x64 -> 24x32 x embedding_dim      (stride-2 conv)

    The encoder outputs a (B, D, 24, 32) continuous latent map that
    the VQ layer discretizes. 4 downsampling stages give 16x reduction.

    The encoder is NOT needed at inflate time -- only for encoding.

    Args:
        in_channels: input channels (3 for RGB)
        embedding_dim: output channel dimension (must match VQ codebook dim)
        base_ch: base channel width, doubled at each stage
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 64,
        base_ch: int = 64,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # 384x512 -> 192x256
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True),
            # 192x256 -> 96x128
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),
            # 96x128 -> 48x64
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),
            # 48x64 -> 24x32
            nn.Conv2d(base_ch * 4, embedding_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, embedding_dim),
            # No final activation -- VQ layer handles the rest
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames to continuous latent.

        Args:
            x: (B, 3, H, W) float tensor in [0, 255]

        Returns:
            (B, D, H//16, W//16) continuous latent
        """
        return self.encoder(x / 255.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── VQ Decoder ─────────────────────────────────────────────────────────


class VQDecoder(nn.Module):
    """Decode quantized latent codes back to RGB frames.

    Architecture (mirror of encoder):
        24x32 x D -> 48x64 x base_ch*4       (transposed conv)
        48x64 -> 96x128 x base_ch*2          (transposed conv)
        96x128 -> 192x256 x base_ch          (transposed conv)
        192x256 -> 384x512 x 3               (transposed conv)

    Uses soft sigmoid output for always-flowing gradients (same as MaskRenderer).
    The decoder is the component that ships in the archive.

    Args:
        embedding_dim: input channel dimension (must match VQ codebook dim)
        out_channels: output channels (3 for RGB)
        base_ch: base channel width
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        out_channels: int = 3,
        base_ch: int = 64,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            # 24x32 -> 48x64
            nn.ConvTranspose2d(embedding_dim, base_ch * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),
            # 48x64 -> 96x128
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),
            # 96x128 -> 192x256
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True),
            # 192x256 -> 384x512
            nn.ConvTranspose2d(base_ch, out_channels, 4, stride=2, padding=1),
        )
        # Zero-init final layer for stable training start
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to RGB frames.

        Args:
            z_q: (B, D, 24, 32) quantized latent vectors

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        logits = self.decoder(z_q)
        # Soft sigmoid: gradients always flow, init at mid-gray
        return 255.0 * torch.sigmoid(logits / 50.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Full VQ-VAE ────────────────────────────────────────────────────────


class VQVAE(nn.Module):
    """Complete VQ-VAE: encoder + vector quantization + decoder.

    Training modes:
        Phase 1 (pretrain): reconstruction loss only (L1 + perceptual)
        Phase 2 (scorer):   add scorer loss for task-specific quality
        Phase 3 (commit):   increase VQ commitment to encourage codebook usage

    The encoder runs at ENCODE time (once per video).
    The decoder runs at INFLATE time (1200 frames).
    Only decoder + codebook ship in the archive.

    Args:
        num_embeddings: codebook size K
        embedding_dim: codebook vector dimension D
        base_ch: base channel width for encoder/decoder
        commitment_cost: VQ commitment loss weight
        ema_decay: EMA decay for codebook updates
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        base_ch: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = VQEncoder(
            in_channels=3,
            embedding_dim=embedding_dim,
            base_ch=base_ch,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        self.decoder = VQDecoder(
            embedding_dim=embedding_dim,
            out_channels=3,
            base_ch=base_ch,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode-quantize-decode pass.

        Args:
            x: (B, 3, H, W) float tensor in [0, 255]

        Returns:
            x_recon: (B, 3, H, W) reconstructed frames in [0, 255]
            vq_loss: scalar VQ loss
            indices: (B, 24, 32) codebook indices
        """
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames to discrete codebook indices.

        This is the ENCODE-time path. Run once per video.

        Args:
            x: (B, 3, H, W) float tensor in [0, 255]

        Returns:
            (B, 24, 32) integer codebook indices
        """
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices to RGB frames.

        This is the INFLATE-time path. Runs on every frame.
        Only requires decoder + codebook (no encoder).

        Args:
            indices: (B, 24, 32) integer codebook indices

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        z_q = self.quantizer.indices_to_embeddings(indices)
        return self.decoder(z_q)

    def param_count(self) -> dict[str, int]:
        """Parameter counts per component."""
        return {
            "encoder": self.encoder.param_count(),
            "decoder": self.decoder.param_count(),
            "codebook": self.quantizer.K * self.quantizer.D,
            "total": sum(p.numel() for p in self.parameters()),
        }

    def archive_param_count(self) -> int:
        """Parameters that ship in the archive (decoder + codebook only)."""
        return self.decoder.param_count() + self.quantizer.K * self.quantizer.D


# ── VQ-VAE Pair Generator ─────────────────────────────────────────────


class VQVAEPairGenerator(nn.Module):
    """Generate PoseNet-compatible frame pairs from VQ-VAE reconstruction.

    Integrates with the existing scorer training infrastructure by
    producing (B, 2, H, W, 3) HWC pair format from forward().

    VQ loss and indices are available via the last_vq_loss and last_indices
    attributes after each forward() call, keeping the return interface
    consistent with other PairGenerator variants (single tensor output).

    Note: last_vq_loss and last_indices are overwritten on each forward()
    call. In DataParallel or gradient accumulation, only the last
    mini-batch's values are retained.

    For training: GT frames -> encoder -> VQ -> decoder -> pairs -> scorer
    For inference: stored indices -> VQ -> decoder -> pairs

    Args:
        vqvae: VQVAE instance
    """

    def __init__(self, vqvae: VQVAE):
        super().__init__()
        self.vqvae = vqvae
        # Exposed after each forward() call for training code to consume.
        # WARNING: mutable instance state — not thread-safe. If used from
        # multiple threads, read these values immediately after forward()
        # or switch to returning a dict from forward().
        self.last_vq_loss: torch.Tensor | None = None
        self.last_indices: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a scored frame pair through VQ-VAE reconstruction.

        After calling forward(), access self.last_vq_loss and
        self.last_indices for the VQ loss and codebook indices.

        Args:
            frame_t: (B, 3, H, W) float [0, 255] -- frame at time t
            frame_t1: (B, 3, H, W) float [0, 255] -- frame at time t+1

        Returns:
            (B, 2, H, W, 3) float HWC pair for scorer
        """
        recon_t, vq_loss_t, idx_t = self.vqvae(frame_t)
        recon_t1, vq_loss_t1, idx_t1 = self.vqvae(frame_t1)

        self.last_vq_loss = vq_loss_t + vq_loss_t1
        self.last_indices = (idx_t, idx_t1)

        # CHW -> HWC: (B, 3, H, W) -> (B, H, W, 3)
        f_t_hwc = recon_t.permute(0, 2, 3, 1)
        f_t1_hwc = recon_t1.permute(0, 2, 3, 1)
        pair = torch.stack([f_t_hwc, f_t1_hwc], dim=1)

        return pair

    def forward_from_indices(
        self,
        idx_t: torch.Tensor,
        idx_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate pair from pre-computed indices (inflate-time path).

        Args:
            idx_t: (B, 24, 32) codebook indices for frame t
            idx_t1: (B, 24, 32) codebook indices for frame t+1

        Returns:
            (B, 2, H, W, 3) float HWC pair
        """
        recon_t = self.vqvae.decode_from_indices(idx_t)
        recon_t1 = self.vqvae.decode_from_indices(idx_t1)

        f_t_hwc = recon_t.permute(0, 2, 3, 1)
        f_t1_hwc = recon_t1.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)


# ── Temporal Delta Coder ───────────────────────────────────────────────


class TemporalDeltaCoder:
    """Compress VQ code indices using temporal delta coding.

    Consecutive video frames have highly correlated VQ codes -- most
    spatial positions keep the same codebook index. Delta coding stores
    only the changes, which compresses extremely well.

    Encoding:
        1. First frame stored as-is (24*32 = 768 indices)
        2. Subsequent frames: store delta (changed positions only)
        3. Delta format: (position, new_index) pairs
        4. Run-length encode the delta stream

    With K=512 (9-bit indices) and typical ~5-10% change rate:
        Raw: 1200 * 768 * 9bits = 1.04MB
        Delta: ~50-100KB after RLE
    """

    @staticmethod
    def encode_sequence(indices_seq: list[torch.Tensor]) -> bytes:
        """Encode a sequence of index frames to compressed bytes.

        Args:
            indices_seq: list of (H, W) long tensors (one per frame)

        Returns:
            compressed bytes
        """
        if not indices_seq:
            return b""

        buf = io.BytesIO()
        H, W = indices_seq[0].shape

        # Header: H, W, num_frames, max_index (for bit width)
        num_frames = len(indices_seq)
        buf.write(struct.pack("<HHI", H, W, num_frames))

        # First frame: store all indices as uint16
        first = indices_seq[0].cpu().numpy().astype("uint16")
        buf.write(first.tobytes())

        # Subsequent frames: delta encoding
        prev = indices_seq[0].cpu()
        for frame_idx in range(1, num_frames):
            curr = indices_seq[frame_idx].cpu()
            # Find changed positions
            changed_mask = curr != prev
            changed_positions = changed_mask.nonzero(as_tuple=False)  # (N, 2)
            num_changed = changed_positions.shape[0]

            # Write number of changes
            buf.write(struct.pack("<I", num_changed))

            if num_changed > 0:
                # Batch-encode (row, col, new_value) triples via numpy
                rows = changed_positions[:, 0].numpy().astype(np.uint16)
                cols = changed_positions[:, 1].numpy().astype(np.uint16)
                vals = curr[changed_positions[:, 0], changed_positions[:, 1]].numpy().astype(np.uint16)
                # Interleave into struct-compatible layout: (r, c, val_lo, val_hi) per entry
                packed = np.empty(num_changed, dtype=[("r", "<u2"), ("c", "<u2"), ("v", "<u2")])
                packed["r"] = rows
                packed["c"] = cols
                packed["v"] = vals
                buf.write(packed.tobytes())

            prev = curr

        return buf.getvalue()

    @staticmethod
    def decode_sequence(data: bytes, device: torch.device | str = "cpu") -> list[torch.Tensor]:
        """Decode compressed bytes back to index frame sequence.

        Args:
            data: bytes from encode_sequence
            device: target device for output tensors

        Returns:
            list of (H, W) long tensors
        """
        if not data:
            return []

        buf = io.BytesIO(data)

        # Header
        H, W, num_frames = struct.unpack("<HHI", buf.read(8))

        # First frame
        first_data = buf.read(H * W * 2)
        first = torch.from_numpy(np.frombuffer(first_data, dtype=np.uint16).reshape(H, W).copy()).long().to(device)

        frames = [first]
        prev = first.clone()

        # Subsequent frames: apply deltas
        for _ in range(1, num_frames):
            num_changed = struct.unpack("<I", buf.read(4))[0]
            curr = prev.clone()

            if num_changed > 0:
                # Batch-decode all changed positions at once via numpy
                chunk = buf.read(num_changed * 6)
                delta = np.frombuffer(chunk, dtype=[("r", "<u2"), ("c", "<u2"), ("v", "<u2")])
                rows = torch.from_numpy(delta["r"].astype(np.int64))
                cols = torch.from_numpy(delta["c"].astype(np.int64))
                vals = torch.from_numpy(delta["v"].astype(np.int64))
                curr[rows, cols] = vals

            frames.append(curr)
            prev = curr.clone()

        return frames

    @staticmethod
    def compression_stats(indices_seq: list[torch.Tensor]) -> dict[str, float]:
        """Compute compression statistics for a sequence.

        Returns dict with raw_bytes, compressed_bytes, ratio, avg_change_rate.
        """
        if not indices_seq:
            return {"raw_bytes": 0, "compressed_bytes": 0, "ratio": 0, "avg_change_rate": 0}

        H, W = indices_seq[0].shape
        raw_bytes = len(indices_seq) * H * W * 2  # uint16

        compressed = TemporalDeltaCoder.encode_sequence(indices_seq)
        compressed_bytes = len(compressed)

        # Compute average change rate
        total_changes = 0
        total_positions = 0
        prev = indices_seq[0]
        for i in range(1, len(indices_seq)):
            curr = indices_seq[i]
            total_changes += (curr != prev).sum().item()
            total_positions += H * W
            prev = curr

        avg_change_rate = total_changes / max(total_positions, 1)

        return {
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": raw_bytes / max(compressed_bytes, 1),
            "avg_change_rate": avg_change_rate,
        }


# ── Loss Functions ─────────────────────────────────────────────────────


def vqvae_reconstruction_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """L1 reconstruction loss in [0, 255] range.

    Args:
        x_recon: (B, 3, H, W) reconstructed frames
        x_target: (B, 3, H, W) ground truth frames

    Returns:
        scalar L1 loss
    """
    return F.l1_loss(x_recon, x_target)


def vqvae_perceptual_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """Simple perceptual loss using feature differences at multiple scales.

    Uses average pooling at 2x and 4x downscale as a lightweight
    perceptual proxy (no VGG needed, MPS-compatible).

    Args:
        x_recon: (B, 3, H, W) reconstructed frames
        x_target: (B, 3, H, W) ground truth frames

    Returns:
        scalar multi-scale L1 loss
    """
    loss = F.l1_loss(x_recon, x_target)

    # 2x downscale
    recon_2x = F.avg_pool2d(x_recon, 2)
    target_2x = F.avg_pool2d(x_target, 2)
    loss = loss + F.l1_loss(recon_2x, target_2x)

    # 4x downscale
    recon_4x = F.avg_pool2d(x_recon, 4)
    target_4x = F.avg_pool2d(x_target, 4)
    loss = loss + F.l1_loss(recon_4x, target_4x)

    return loss / 3.0


def vqvae_combined_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    vq_loss: torch.Tensor,
    vq_weight: float = 1.0,
    perceptual_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined VQ-VAE training loss.

    Args:
        x_recon: (B, 3, H, W) reconstructed frames
        x_target: (B, 3, H, W) ground truth frames
        vq_loss: scalar VQ loss from quantizer
        vq_weight: weight for VQ loss term
        perceptual_weight: weight for perceptual loss term

    Returns:
        total_loss: scalar combined loss
        components: dict of individual loss values for logging
    """
    recon_loss = vqvae_reconstruction_loss(x_recon, x_target)
    percep_loss = vqvae_perceptual_loss(x_recon, x_target)

    total = recon_loss + perceptual_weight * percep_loss + vq_weight * vq_loss

    components = {
        "recon_l1": recon_loss.item(),
        "perceptual": percep_loss.item(),
        "vq_loss": vq_loss.item(),
        "total": total.item(),
    }
    return total, components


# ── Serialization ──────────────────────────────────────────────────────


def save_decoder_and_codebook(
    vqvae: VQVAE,
    output_dir: str | Path,
    prefix: str = "vqvae",
) -> dict[str, int]:
    """Save only the decoder and codebook (inflate-time components).

    The encoder is NOT saved -- it's only needed at encode time.

    Args:
        vqvae: trained VQVAE model
        output_dir: directory to save to
        prefix: filename prefix

    Returns:
        dict with file sizes in bytes
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save decoder state dict
    decoder_path = out / f"{prefix}_decoder.pt"
    torch.save(vqvae.decoder.state_dict(), decoder_path)

    # Save codebook
    codebook_path = out / f"{prefix}_codebook.pt"
    torch.save(vqvae.quantizer.embedding.weight.data, codebook_path)

    sizes = {
        "decoder_bytes": decoder_path.stat().st_size,
        "codebook_bytes": codebook_path.stat().st_size,
    }
    sizes["total_bytes"] = sizes["decoder_bytes"] + sizes["codebook_bytes"]
    return sizes


def load_decoder_and_codebook(
    input_dir: str | Path,
    prefix: str = "vqvae",
    num_embeddings: int = 512,
    embedding_dim: int = 64,
    base_ch: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[VQDecoder, VectorQuantizer]:
    """Load decoder and codebook for inflate-time use.

    Args:
        input_dir: directory containing saved files
        prefix: filename prefix
        num_embeddings: codebook size K
        embedding_dim: codebook vector dimension D
        base_ch: decoder base channel width
        device: target device

    Returns:
        (decoder, quantizer) tuple ready for inference
    """
    inp = Path(input_dir)
    device = torch.device(device) if isinstance(device, str) else device

    # Load decoder
    decoder = VQDecoder(embedding_dim=embedding_dim, base_ch=base_ch)
    decoder_state = torch.load(inp / f"{prefix}_decoder.pt", map_location=device, weights_only=True)
    decoder.load_state_dict(decoder_state)
    decoder.to(device).eval()

    # Load codebook into quantizer
    quantizer = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        ema_decay=0,  # No EMA needed for inference
    )
    codebook_weights = torch.load(inp / f"{prefix}_codebook.pt", map_location=device, weights_only=True)
    quantizer.embedding.weight.data.copy_(codebook_weights)
    quantizer.to(device).eval()

    return decoder, quantizer


# ── Factory ────────────────────────────────────────────────────────────


def build_vqvae(
    num_embeddings: int = 512,
    embedding_dim: int = 64,
    base_ch: int = 64,
    commitment_cost: float = 0.25,
    ema_decay: float = 0.99,
) -> VQVAE:
    """Build a VQ-VAE model with specified hyperparameters.

    Default configuration:
        K=512 codebook entries, D=64 embedding dim, base_ch=64
        Encoder: ~600K params (not shipped)
        Decoder: ~600K params (shipped, ~200KB FP4)
        Codebook: 512*64 = 32K params (~32KB)

    Args:
        num_embeddings: codebook size K
        embedding_dim: codebook vector dimension D
        base_ch: base channel width for encoder/decoder
        commitment_cost: VQ commitment loss weight
        ema_decay: EMA decay for codebook updates

    Returns:
        VQVAE instance
    """
    return VQVAE(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        base_ch=base_ch,
        commitment_cost=commitment_cost,
        ema_decay=ema_decay,
    )


def build_vqvae_pair_generator(
    num_embeddings: int = 512,
    embedding_dim: int = 64,
    base_ch: int = 64,
    commitment_cost: float = 0.25,
    ema_decay: float = 0.99,
) -> VQVAEPairGenerator:
    """Build VQ-VAE with pair generator for scorer training.

    Returns a module that produces (B, 2, H, W, 3) HWC pairs
    compatible with the existing training infrastructure.

    Args:
        num_embeddings: codebook size K
        embedding_dim: codebook vector dimension D
        base_ch: base channel width
        commitment_cost: VQ commitment loss weight
        ema_decay: EMA decay for codebook updates

    Returns:
        VQVAEPairGenerator instance
    """
    vqvae = build_vqvae(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        base_ch=base_ch,
        commitment_cost=commitment_cost,
        ema_decay=ema_decay,
    )
    return VQVAEPairGenerator(vqvae)
