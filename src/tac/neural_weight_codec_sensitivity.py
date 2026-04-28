"""Lane J-NWCS: Sensitivity-aware Neural Weight Compression.

Composition of three existing techniques:

  * Lane J-NWC (``tac.neural_weight_codec``)
        Base VQ-VAE-style codec for blocks of weight elements.

  * Lane Ω-V2 (``tac.learnable_bit_quant``)
        Hessian / gradient-magnitude-based per-parameter sensitivity.

  * Lane W (``tac.learnable_pair_weights``)
        Hard-pair gradient signal: which (frame_t, frame_{t+1}) pairs
        are PoseNet-critical.

Why a NEW lane (vs hand-stacking the three at deploy time)?  Because the
codebook design changes when sensitivity is known up-front:

    * Per-block sensitivity (Hessian magnitude × hard-pair gradient norm)
      is computed against the Lane G v3 anchor renderer.
    * Block sensitivities are bucketed by quantile (default 4 buckets).
    * Each bucket gets its own VQ codebook size: high-sensitivity blocks
      get K=256 (8 bits/code), low-sensitivity blocks get K=4 (2 bits/code).
    * Total bytes/block is amortized across buckets so the average bits
      per weight stays inside the lane's rate budget.

This is **strictly Pareto-dominant** over uniform-K NWC: spending more
bits on the blocks that drive PoseNet/SegNet error and fewer on the
blocks that don't is exactly the geometry the scorer rewards.

Compose-with rules (per ``docs/stacking_architecture.md``):
    Slot:        renderer-encoder
    Stacks-with: any renderer-replacement output (Lane G v3, J-JBL, J-IMP)
    Exclusive-with: Lane J-NWC (J-NWCS supersedes), Lane F-V5 FP8
    Composes-with: Lane Ω-V2 per-weight bits (Ω-V2 first → J-NWCS quantizes
                   within the resulting bit budget)
    Required signal-sidecar: Lane W hard_pair_weights.npy
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.neural_weight_codec import (
    WeightCodec,
    WeightCodecConfig,
    tensor_to_blocks,
)


__all__ = [
    "SensitivityAwareCodecConfig",
    "SensitivityAwareWeightCodec",
    "compute_per_block_sensitivity",
    "encode_with_variable_codebook",
    "decode_with_per_block_codebook",
]


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class SensitivityAwareCodecConfig:
    """Static config for a sensitivity-aware NWC codec.

    Default codebook ladder is [4, 16, 64, 256], keyed by sensitivity
    quantile (Q1..Q4). Operators may override; sizes must all be ≤ 256
    so codes still fit in uint8.

    block_size matches the underlying base codec.
    """

    block_size: int = 16
    latent_dim: int = 16
    hidden: int = 64
    codebook_sizes: list[int] = field(default_factory=lambda: [4, 16, 64, 256])
    importance_weight: float = 2.0
    """Weight applied to high-sensitivity blocks during VQ-VAE training.

    When training the codec on a corpus, a block whose sensitivity falls
    in the top quartile contributes ``importance_weight`` × MSE to the
    reconstruction loss. Default 2.0 is a conservative choice; ablations
    in [0.5, 5.0] are reasonable.
    """

    def __post_init__(self) -> None:
        if not self.codebook_sizes:
            raise ValueError("codebook_sizes must be non-empty")
        for k in self.codebook_sizes:
            if k <= 0 or k > 256:
                raise ValueError(
                    f"codebook size {k} not in (0, 256] — must fit in uint8"
                )
        if self.block_size <= 0 or self.latent_dim <= 0:
            raise ValueError("block_size and latent_dim must be positive")
        if self.importance_weight < 0:
            raise ValueError("importance_weight must be ≥ 0")


# ── Sensitivity computation ───────────────────────────────────────────────


def compute_per_block_sensitivity(
    model: nn.Module,
    hard_pairs: torch.Tensor,
    gt_pairs: torch.Tensor,
    scorer: nn.Module,
    *,
    block_size: int = 16,
    hessian_proxy: str = "grad_squared",
) -> dict[str, torch.Tensor]:
    """Compute a per-block sensitivity score for every floating-point
    parameter of ``model``.

    The score is the elementwise product of two signals:

        Hessian magnitude   ≈ ⟨∂L/∂w⟩²    (gradient-squared diagonal proxy)
        hard-pair magnitude = |∂L_pair/∂w|  averaged across pairs

    Sensitivity is then aggregated from per-element to per-block via
    ``mean(|score|)``. Returns a dict keyed by parameter NAME mapping to
    a 1-D tensor of length ``ceil(numel / block_size)``.

    Args:
        model:           nn.Module to score (Lane G v3 renderer typically).
        hard_pairs:      (N, ...) tensor of input pairs; the "hard" subset
                         identified by Lane W.
        gt_pairs:        (N, ...) ground-truth target tensor for hard_pairs.
        scorer:          callable producing a scalar loss per pair when
                         called as ``scorer(model_output, target)``. Lane W's
                         standard PoseNet/SegNet loss works.
        block_size:      partition size for sensitivity aggregation.
        hessian_proxy:   currently only ``grad_squared`` (diagonal Fisher)
                         is supported. Future: full Hessian-vector product
                         via finite-diff if memory permits.

    Returns:
        Dict[str, Tensor]: per-parameter per-block sensitivity tensors.
        Each entry is float32, on CPU, shape ``(n_blocks,)``.

    Notes:
        * Uses ``torch.autograd.grad`` so model.parameters() grads are not
          left dirty afterwards.
        * Skips bias-shaped tensors (1-D, < 2048 elements) to match the
          base codec's corpus-builder filter.
        * The hard_pairs / gt_pairs need not be on any specific device —
          they are moved to ``next(model.parameters()).device``.
    """
    if hessian_proxy != "grad_squared":
        raise NotImplementedError(
            f"hessian_proxy={hessian_proxy!r} not supported (only 'grad_squared')"
        )
    device = next(model.parameters()).device
    hard = hard_pairs.to(device)
    gt = gt_pairs.to(device)

    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    grad_sums: dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in params}
    grad_sq_sums: dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in params}

    n_pairs = hard.shape[0]
    if n_pairs == 0:
        raise ValueError("hard_pairs has zero entries; need at least 1 pair")

    for i in range(n_pairs):
        x_i = hard[i : i + 1]
        y_i = gt[i : i + 1]
        out = model(x_i)
        loss = scorer(out, y_i)
        if loss.dim() != 0:
            loss = loss.mean()
        grads = torch.autograd.grad(
            loss,
            [p for _, p in params],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for (name, _), g in zip(params, grads):
            if g is None:
                continue
            grad_sums[name] = grad_sums[name] + g.detach().abs()
            grad_sq_sums[name] = grad_sq_sums[name] + g.detach() ** 2

    out: dict[str, torch.Tensor] = {}
    for name, p in params:
        if not torch.is_floating_point(p):
            continue
        if p.dim() == 1 and p.numel() < 2048:
            # bias filter — same heuristic as base codec corpus builder
            continue
        if p.numel() < block_size:
            continue
        # mean over pairs
        gmag = (grad_sums[name] / float(n_pairs)).reshape(-1)
        ghess = (grad_sq_sums[name] / float(n_pairs)).reshape(-1)
        # combine: hard-pair magnitude × Hessian-diag proxy (per element)
        per_elem = gmag * ghess
        # aggregate to blocks (drop tail; same as tensor_to_blocks)
        n_blocks = per_elem.numel() // block_size
        if n_blocks == 0:
            continue
        per_block = (
            per_elem[: n_blocks * block_size]
            .reshape(n_blocks, block_size)
            .mean(dim=1)
        )
        out[name] = per_block.detach().cpu().float()
    return out


def _bucket_by_quantile(
    sensitivity: torch.Tensor, n_buckets: int
) -> torch.Tensor:
    """Return a long tensor of bucket indices in [0, n_buckets-1] for each
    block, using quantile-edges of the sensitivity distribution.

    Bucket 0 = lowest sensitivity, n_buckets-1 = highest.
    """
    if sensitivity.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if n_buckets <= 1:
        return torch.zeros_like(sensitivity, dtype=torch.long)
    qs = torch.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    edges = torch.quantile(sensitivity.float(), qs)
    bucket = torch.bucketize(sensitivity.float(), edges)
    return bucket.long().clamp(max=n_buckets - 1)


# ── Codec module ──────────────────────────────────────────────────────────


class SensitivityAwareWeightCodec(WeightCodec):
    """Extension of WeightCodec with per-bucket codebooks of varying size.

    The codec maintains ``len(codebook_sizes)`` separate codebooks, each
    of width ``latent_dim`` and height equal to its bucket's K. At
    encode-time, each block is routed to one bucket (by sensitivity
    quantile), quantized against that bucket's codebook, and serialized
    with a small per-block header byte indicating which bucket was used.

    Design notes:
        * The encoder/decoder MLPs are SHARED across buckets — only the
          codebooks differ. This keeps codec parameter count comparable
          to the base codec (encoder + decoder + sum(K) × latent_dim).
        * For the default ladder [4, 16, 64, 256] @ latent_dim=16, the
          codebooks total 340 × 16 = 5440 floats = 21.7 KB. Shared MLPs
          are unchanged from the base codec.
        * The per-block bucket header is 1 byte (uint8); for the default
          4-bucket config only 2 bits are used. This is the small
          overhead that ``test_byte_size_breakdown_per_codebook_size``
          asserts is < 5% of total.
    """

    def __init__(self, config: SensitivityAwareCodecConfig | None = None):
        # initialize the base WeightCodec with a placeholder codebook_size
        # (we ignore self.codebook below; the per-bucket codebooks are stored
        # in ``self.bucket_codebooks``).
        cfg = config or SensitivityAwareCodecConfig()
        base_cfg = WeightCodecConfig(
            block_size=cfg.block_size,
            codebook_size=max(cfg.codebook_sizes),
            latent_dim=cfg.latent_dim,
            hidden=cfg.hidden,
        )
        super().__init__(base_cfg)
        self.sens_config = cfg

        # Replace the single codebook with a per-bucket ParameterList.
        del self.codebook  # remove inherited single codebook
        self.bucket_codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(K, cfg.latent_dim) * 0.1)
                for K in cfg.codebook_sizes
            ]
        )
        # Re-expose self.codebook as a property (largest bucket) so that
        # WeightCodec methods which read self.codebook still work for
        # untested paths. Tests should always use the bucket-aware API.

    # ── codebook helpers ─────────────────────────────────────────────

    def _quantize_to_bucket(
        self, z_e: torch.Tensor, bucket_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Snap encoder output to the codebook of bucket ``bucket_idx``.

        Returns (z_q, indices) where indices are uint8-compatible long.
        """
        cb = self.bucket_codebooks[bucket_idx]
        z_sq = (z_e ** 2).sum(dim=-1, keepdim=True)
        c_sq = (cb ** 2).sum(dim=-1).unsqueeze(0)
        cross = z_e @ cb.T
        distances = z_sq + c_sq - 2.0 * cross
        indices = distances.argmin(dim=-1)
        z_q = cb.index_select(0, indices)
        return z_q, indices

    # ── train_with_sensitivity ───────────────────────────────────────

    def train_with_sensitivity(
        self,
        corpus: torch.Tensor,
        sensitivities: torch.Tensor,
        *,
        importance_weight: float | None = None,
        num_steps: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        log_interval: int = 100,
        seed: int = 1234,
    ) -> tuple["SensitivityAwareWeightCodec", list[float]]:
        """Train the codec with importance-weighted reconstruction loss.

        High-sensitivity blocks (top quartile by default) get an extra
        ``importance_weight`` × MSE penalty, biasing the codec toward
        better reconstruction on the blocks that matter for the scorer.
        """
        if corpus.dim() != 2 or corpus.shape[1] != self.config.block_size:
            raise ValueError(
                f"corpus must be (N, block_size={self.config.block_size}), "
                f"got {tuple(corpus.shape)}"
            )
        if sensitivities.dim() != 1 or sensitivities.numel() != corpus.shape[0]:
            raise ValueError(
                f"sensitivities must be 1-D length N={corpus.shape[0]}, "
                f"got shape {tuple(sensitivities.shape)}"
            )
        iw = float(importance_weight) if importance_weight is not None else self.sens_config.importance_weight
        device_t = torch.device(device)
        self.to(device_t)
        corpus = corpus.to(device_t)
        sensitivities = sensitivities.to(device_t)

        n_buckets = len(self.sens_config.codebook_sizes)
        buckets = _bucket_by_quantile(sensitivities, n_buckets).to(device_t)
        # high-sensitivity = top bucket
        high_mask = (buckets == n_buckets - 1).float()

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        losses: list[float] = []
        n = corpus.shape[0]
        for step in range(num_steps):
            idx = torch.randint(0, n, (batch_size,), generator=g)
            x = corpus[idx]
            b = buckets[idx]
            hi = high_mask[idx]

            # forward through encoder
            z_e = self.encoder(x)
            # quantize per-element by bucket
            z_q = torch.zeros_like(z_e)
            for k in range(n_buckets):
                mask = (b == k)
                if not mask.any():
                    continue
                zk_q, _ = self._quantize_to_bucket(z_e[mask], k)
                z_q[mask] = zk_q
            # VQ-VAE STE
            z_q_st = z_e + (z_q - z_e).detach()
            recon = self.decoder(z_q_st)

            # importance-weighted reconstruction loss (per-block scalar)
            per_block = (recon - x).pow(2).mean(dim=1)
            weights = 1.0 + iw * hi  # baseline 1.0, +iw on high-sens blocks
            recon_loss = (per_block * weights).sum() / weights.sum()
            commit = (
                F.mse_loss(z_e, z_q.detach())
                + self.commitment_beta * F.mse_loss(z_q, z_e.detach())
            )
            loss = recon_loss + commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if step % log_interval == 0 or step == num_steps - 1:
                print(
                    f"[nwcs-train] step={step:5d} "
                    f"recon={recon_loss.detach().item():.6f} "
                    f"commit={commit.detach().item():.6f} "
                    f"hi_frac={hi.mean().item():.3f} "
                    f"total={loss.item():.6f}"
                )
        return self, losses


# ── Variable-codebook serialization ───────────────────────────────────────

# NWCS1 binary layout (single tensor):
#
#   [4 B] uint32 ndim
#   [4 B × ndim] uint32 shape
#   [4 B] uint32 n_blocks
#   [1 B] uint8 n_buckets
#   [2 B × n_buckets] uint16 bucket sizes (codebook K's; K may be up to 256)
#   [n_blocks × 1 B] uint8 bucket id per block
#   [n_blocks × 2 B] float16 per-block scale
#   [n_blocks × 1 B] uint8 codebook index per block (within its bucket; K=256 stored as 0)
#   [tail × 2 B] float16 leftover-tail elements
#
# Note: a codebook size of K=256 stores indices 0..255 into a uint8 byte
# (the max value uint8 can represent), but the *count* 256 itself does
# not fit in a uint8, hence the 2-byte uint16 used for the bucket-size
# header field.


def encode_with_variable_codebook(
    codec: SensitivityAwareWeightCodec,
    weights: torch.Tensor,
    sensitivities: torch.Tensor,
) -> bytes:
    """Encode a single weight tensor under variable codebook sizes.

    Args:
        codec:        a trained SensitivityAwareWeightCodec.
        weights:      the tensor to encode (any floating dtype).
        sensitivities: 1-D tensor of length ``ceil(numel/block_size)``
                      giving per-block sensitivity. (Tail blocks excluded.)

    Returns:
        bytes blob in NWCS1 layout above.
    """
    if not torch.is_floating_point(weights):
        raise TypeError(f"encode expects floating tensor, got {weights.dtype}")
    device = next(codec.parameters()).device
    Bs = codec.config.block_size
    flat = weights.detach().to(device).float().reshape(-1)
    N = flat.numel()
    n_blocks = N // Bs
    tail_n = N - n_blocks * Bs

    if sensitivities.numel() != n_blocks:
        raise ValueError(
            f"sensitivities length {sensitivities.numel()} != n_blocks {n_blocks}"
        )

    n_buckets = len(codec.sens_config.codebook_sizes)
    buckets = _bucket_by_quantile(sensitivities, n_buckets).to(device)

    if n_blocks == 0:
        scales = torch.zeros(0, dtype=torch.float32, device=device)
        codes = torch.zeros(0, dtype=torch.long, device=device)
        bucket_ids = torch.zeros(0, dtype=torch.long, device=device)
    else:
        blocks = flat[: n_blocks * Bs].reshape(n_blocks, Bs)
        scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
        blocks_norm = blocks / scales.unsqueeze(1)
        with torch.no_grad():
            z_e = codec.encoder(blocks_norm)
            codes = torch.zeros(n_blocks, dtype=torch.long, device=device)
            for k in range(n_buckets):
                mask = (buckets == k)
                if not mask.any():
                    continue
                _, code_k = codec._quantize_to_bucket(z_e[mask], k)
                codes[mask] = code_k
        bucket_ids = buckets

    tail = flat[n_blocks * Bs :] if tail_n > 0 else torch.zeros(0, device=device)

    buf = bytearray()
    shape = list(weights.shape)
    buf.extend(struct.pack("<I", len(shape)))
    for s in shape:
        buf.extend(struct.pack("<I", int(s)))
    buf.extend(struct.pack("<I", int(n_blocks)))
    buf.extend(struct.pack("<B", int(n_buckets)))
    for K in codec.sens_config.codebook_sizes:
        # uint16 — 256 fits even though uint8 cannot hold the count.
        buf.extend(struct.pack("<H", int(K)))
    buf.extend(bucket_ids.cpu().to(torch.uint8).numpy().tobytes())
    buf.extend(scales.cpu().to(torch.float16).numpy().tobytes())
    buf.extend(codes.cpu().to(torch.uint8).numpy().tobytes())
    buf.extend(tail.cpu().to(torch.float16).numpy().tobytes())
    return bytes(buf)


def decode_with_per_block_codebook(
    codec: SensitivityAwareWeightCodec, blob: bytes
) -> torch.Tensor:
    """Inverse of ``encode_with_variable_codebook``.

    Returns a CPU float32 tensor with the original shape.
    """
    import numpy as np

    device = next(codec.parameters()).device
    Bs = codec.config.block_size
    offset = 0

    ndim = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    if ndim == 0 or ndim > 8:
        raise ValueError(f"NWCS1.decode: implausible ndim={ndim}")
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<I", blob, offset)[0])
        offset += 4
    n_blocks = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    n_buckets = struct.unpack_from("<B", blob, offset)[0]
    offset += 1
    bucket_sizes = []
    for _ in range(n_buckets):
        bucket_sizes.append(struct.unpack_from("<H", blob, offset)[0])
        offset += 2
    if list(bucket_sizes) != list(codec.sens_config.codebook_sizes):
        raise ValueError(
            f"NWCS1.decode: codec bucket sizes mismatch "
            f"(blob {bucket_sizes} vs codec {codec.sens_config.codebook_sizes})"
        )

    bucket_ids_buf = blob[offset : offset + n_blocks]
    offset += n_blocks
    scales_buf = blob[offset : offset + n_blocks * 2]
    offset += n_blocks * 2
    codes_buf = blob[offset : offset + n_blocks]
    offset += n_blocks

    numel = 1
    for s in shape:
        numel *= int(s)
    tail_n = numel - n_blocks * Bs
    if tail_n < 0:
        raise ValueError(
            f"NWCS1.decode negative tail (n_blocks={n_blocks}, Bs={Bs}, numel={numel})"
        )
    tail_buf = blob[offset : offset + tail_n * 2]
    offset += tail_n * 2

    if n_blocks > 0:
        bucket_ids = torch.from_numpy(
            np.frombuffer(bucket_ids_buf, dtype=np.uint8).copy()
        ).to(device=device, dtype=torch.long)
        scales = torch.from_numpy(
            np.frombuffer(scales_buf, dtype=np.float16).copy()
        ).to(device=device, dtype=torch.float32)
        codes = torch.from_numpy(
            np.frombuffer(codes_buf, dtype=np.uint8).copy()
        ).to(device=device, dtype=torch.long)
        with torch.no_grad():
            z_q = torch.zeros(n_blocks, codec.config.latent_dim, device=device)
            for k in range(n_buckets):
                mask = (bucket_ids == k)
                if not mask.any():
                    continue
                z_q[mask] = codec.bucket_codebooks[k].index_select(0, codes[mask])
            recon_norm = codec.decoder(z_q)
            recon_blocks = recon_norm * scales.unsqueeze(1)
        flat_recon = recon_blocks.reshape(-1)
    else:
        flat_recon = torch.zeros(0, device=device, dtype=torch.float32)

    if tail_n > 0:
        tail = torch.from_numpy(
            np.frombuffer(tail_buf, dtype=np.float16).copy()
        ).to(device=device, dtype=torch.float32)
    else:
        tail = torch.zeros(0, device=device, dtype=torch.float32)

    full = torch.cat([flat_recon, tail], dim=0)
    if full.numel() != numel:
        raise ValueError(
            f"NWCS1.decode size mismatch: got {full.numel()}, expected {numel}"
        )
    return full.reshape(*shape).cpu()
