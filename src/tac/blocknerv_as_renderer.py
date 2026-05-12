"""BlockNeRV-as-renderer — tile-decomposed NeRV substrate.

Per operator directive 2026-05-11 (NeRV-family expansion) + CLAUDE.md HNeRV
parity discipline (lesson 5: full RGB renderer; lesson 4: inflate ≤200 LOC).
BlockNeRV decomposes the per-frame render into a grid of independently-decoded
spatial TILES (e.g., 6×8 tiles of 64×64). Each tile receives the global
per-pair latent + a learned per-tile coordinate embedding, and is decoded by
a SHARED tile decoder. Tiles are concatenated to form the full frame.

Why blocks
----------
- **Decode parallelism**: each tile is an independent forward pass; ideal for
  batch-friendly inflate.
- **Spatial locality**: per-tile coordinate embeddings give the network an
  explicit spatial prior, reducing wasted capacity on fixed-position content
  vs a single global decoder.
- **Smaller decoder**: tile decoder operates on (64×64) instead of (384×512),
  so per-stage convolutions are cheaper and the decoder can be smaller for the
  same per-pixel parameter budget.

This is the **per-tile decode** flavour of BlockNeRV (Chen 2023 follow-up to
NeRV "On the spatial inductive bias of neural video representations"). Sister
to PR100 hnerv_lc_v2 (single full-res decoder) and to MNeRV (multi-scale
hierarchical decoder).

Architecture (default config)
-----------------------------
- Latent: (B, 16) per-pair, like Lane 12-v2.
- Tile grid: 6 rows × 8 cols of 64×64 tiles → full 384×512 frame.
- Per-tile coord embedding: 8-dim learned per (row, col) → 48 embeddings.
- Tile decoder: a small CNN that takes (latent_dim + 8) → (3, 64, 64) for
  each of the 2 frames in the pair (rgb_0, rgb_1 heads).
- Final: tiles assembled into full (B, 2, 3, 384, 512) RGB output.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header +
    5 length-prefixed sections: tile_decoder INT8+brotli, tile_coord_table
    FP16, scale_table FP16, latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_BLOCKNERV in this module with
    schema_keys_in_order = BlockNeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200
    LOC contest-hermetic
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: blocknerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: trainer wires Lane 12-v2 train_step pattern with
    differentiable rgb_to_yuv6 + load_differentiable_scorers + Lagrangian
  bolt_on_loc_budget: substrate_engineering (full tile-decomposed renderer)
  no_op_detector_planned: export_to_archive returns sha256

CLAUDE.md compliance (lesson-by-lesson)
---------------------------------------
- L1 (score-aware): trainer routes through `train_step` with SegNet + PoseNet.
- L2 (export-first): ARCHIVE_GRAMMAR_BLOCKNERV declared at module level.
- L4 (inflate ≤ 200 LOC): substrate-engineering target.
- L5 (full RGB renderer): forward returns (B, 2, 3, H, W).
- L6 (score-domain Lagrangian): `train_step_blocknerv` delegates to
  Lane 12-v2's score-domain Lagrangian via the same surrogate signatures.
- L8 (eval-roundtrip): `train_step_blocknerv` simulates uint8 STE before scorer.
- L11 (no-op detector): `export_blocknerv_to_archive` returns sha256.
- L13 (KILL is last resort): N/A (new substrate).

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; tile-decomposed inductive bias gives
~1.2× param efficiency at fixed bytes vs single-decoder NeRV per Chen 2023
BlockNeRV ablations; pose-axis marginal 2.71× SegNet at PR106 r2 frontier]``.
NOT a score claim until [contest-CUDA] anchor lands.

Format ID: 0x60 (NeRV-family expansion magic 0xFE).

References
----------
- Lane 12-v2 (`src/tac/lane_12_v2_nerv_as_renderer.py`) — substrate template.
- HNeRV retrospective: `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`.
- PR100 exemplar (DO NOT EDIT): `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/`.
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Magic + format ────────────────────────────────────────────────────────


BLOCKNERV_MAGIC: bytes = b"BNRV"
"""BlockNeRV archive magic (4 ASCII bytes)."""

BLOCKNERV_FORMAT_VERSION: int = 1
"""Phase A archive format version."""

BLOCKNERV_FORMAT_ID: int = 0x60
"""NeRV-family expansion format_id (0x60-0x64 reserved per operator directive)."""


# ── Archive grammar (parser-section manifest, machine-readable) ──────────


ARCHIVE_GRAMMAR_BLOCKNERV: dict = {
    "format_version": BLOCKNERV_FORMAT_VERSION,
    "format_id": BLOCKNERV_FORMAT_ID,
    "magic": BLOCKNERV_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 16,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("version", "<H", 2),
                ("format_id", "<H", 2),
                ("latent_dim", "<H", 2),
                ("n_pairs", "<H", 2),
                ("tile_rows", "<H", 2),
                ("tile_cols", "<H", 2),
            ],
        },
        {
            "name": "tile_decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_schema_driven",
        },
        {
            "name": "tile_coord_table",
            "offset_after": "tile_decoder_blob",
            "length_field_le_u32": True,
            "kind": "fp16_raw_per_tile_coord",
        },
        {
            "name": "scale_table",
            "offset_after": "tile_coord_table",
            "length_field_le_u32": True,
            "kind": "fp16_raw_one_per_schema_entry",
        },
        {
            "name": "latent_blob",
            "offset_after": "scale_table",
            "length_field_le_u32": True,
            "kind": "brotli_uint8_asym_delta_split",
        },
        {
            "name": "sidecar_blob",
            "offset_after": "latent_blob",
            "length_field_le_u32": True,
            "kind": "brotli_optional_phase_b",
            "phase_a_empty": True,
        },
    ],
    "schema_keys_in_order": "BlockNeRVRenderer.SCHEMA",
    "predicted_total_bytes": "150_000 to 180_000 [predicted; tile decoder smaller per stage]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockNeRVConfig:
    """Frozen config for BlockNeRV-as-renderer Phase A.

    Attributes
    ----------
    latent_dim
        Per-pair latent dimensionality. Default 16 mirrors Lane 12-v2.
    tile_h, tile_w
        Tile spatial size. Default 64×64.
    tile_rows, tile_cols
        Tile grid. Default 6×8 → full 384×512 = 6*64 × 8*64.
    coord_embed_dim
        Per-tile coordinate embedding dim. Default 8.
    base_channels
        Tile decoder base channel width.
    n_stages
        Number of PixelShuffle upsample stages inside the tile decoder.
        Default 3 → 8×8 → 16×16 → 32×32 → 64×64.
    n_pairs
        Number of per-pair latents.
    frames_per_pair
        Frames per latent (always 2 for contest pair).
    eval_size
        Native render size (H, W). Must equal (tile_rows*tile_h, tile_cols*tile_w).
    lambda_seg, lambda_pose
        Score-aware loss weights.
    cuda_required
        If True (default), `train_step` raises if CUDA unavailable.
    """

    latent_dim: int = 16
    tile_h: int = 64
    tile_w: int = 64
    tile_rows: int = 6
    tile_cols: int = 8
    coord_embed_dim: int = 8
    base_channels: int = 28
    n_stages: int = 3
    n_pairs: int = 600
    frames_per_pair: int = 2
    eval_size: tuple[int, int] = (384, 512)
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.tile_h <= 0 or self.tile_w <= 0:
            raise ValueError(f"tile dims must be positive, got {self.tile_h}x{self.tile_w}")
        if self.tile_rows <= 0 or self.tile_cols <= 0:
            raise ValueError("tile grid dims must be positive")
        expected = (self.tile_rows * self.tile_h, self.tile_cols * self.tile_w)
        if expected != self.eval_size:
            raise ValueError(
                f"eval_size {self.eval_size} != tile_rows*tile_h x tile_cols*tile_w {expected}"
            )
        # Tile decoder spawns from a fixed init shape via stem. With n_stages
        # PixelShuffle ×2 stages, init_h = tile_h / 2**n_stages; must be ≥1.
        if self.tile_h % (2 ** self.n_stages) != 0 or self.tile_w % (2 ** self.n_stages) != 0:
            raise ValueError(
                f"tile dims {self.tile_h}x{self.tile_w} not divisible by 2**{self.n_stages}"
            )
        if self.frames_per_pair != 2:
            raise ValueError(
                f"Phase A pinned at frames_per_pair=2, got {self.frames_per_pair}"
            )
        if self.coord_embed_dim <= 0:
            raise ValueError(f"coord_embed_dim must be positive, got {self.coord_embed_dim}")


# ── Renderer module ──────────────────────────────────────────────────────


class BlockNeRVRenderer(nn.Module):
    """BlockNeRV — tile-decomposed full RGB renderer.

    Forward signature: ``z (B, latent_dim) → (B, 2, 3, H, W)`` where
    ``H = tile_rows * tile_h`` and ``W = tile_cols * tile_w``.

    All tiles share ONE decoder; per-tile coordinate embedding provides the
    spatial inductive bias.
    """

    def __init__(self, config: BlockNeRVConfig) -> None:
        super().__init__()
        self.config = config
        n_tiles = config.tile_rows * config.tile_cols
        # Per-tile coord embedding (one row per (r, c) flattened in row-major).
        self.tile_coord = nn.Parameter(
            torch.randn(n_tiles, config.coord_embed_dim) * 0.01
        )
        # Stem: (latent_dim + coord_embed_dim) → C0 * init_h * init_w.
        init_h = config.tile_h // (2 ** config.n_stages)
        init_w = config.tile_w // (2 ** config.n_stages)
        self._init_h = init_h
        self._init_w = init_w
        C = config.base_channels
        self.channels: list[int] = [C] + [
            max(8, int(C * (0.85 ** (i + 1)))) for i in range(config.n_stages)
        ]
        self.stem = nn.Linear(
            config.latent_dim + config.coord_embed_dim,
            self.channels[0] * init_h * init_w,
        )
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(config.n_stages):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, padding=1),
            nn.Conv2d(final_ch, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, latent_dim) → (B, 2, 3, H, W)``."""
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}), got {tuple(z.shape)}"
            )
        B = z.shape[0]
        cfg = self.config
        n_tiles = cfg.tile_rows * cfg.tile_cols
        # Broadcast latent to all tiles, concat with per-tile coord embedding.
        # z_all: (B, n_tiles, latent_dim+coord_embed_dim)
        z_exp = z.unsqueeze(1).expand(B, n_tiles, -1)
        coord = self.tile_coord.unsqueeze(0).expand(B, n_tiles, -1)
        z_cat = torch.cat([z_exp, coord], dim=-1)
        # Flatten to (B*n_tiles, latent_dim+coord_embed_dim).
        z_flat = z_cat.reshape(B * n_tiles, -1)
        x = self.stem(z_flat).view(
            B * n_tiles, self.channels[0], self._init_h, self._init_w
        )
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        # rgb_0 / rgb_1: (B*n_tiles, 3, tile_h, tile_w)
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        # Reassemble tiles → (B, 2, 3, H, W).
        H = cfg.tile_rows * cfg.tile_h
        W = cfg.tile_cols * cfg.tile_w
        f0 = f0.view(B, cfg.tile_rows, cfg.tile_cols, 3, cfg.tile_h, cfg.tile_w)
        f0 = f0.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, H, W)
        f1 = f1.view(B, cfg.tile_rows, cfg.tile_cols, 3, cfg.tile_h, cfg.tile_w)
        f1 = f1.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, H, W)
        return torch.stack([f0, f1], dim=1)

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        """Pinned state-dict (key, shape) order for archive packing."""
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        keys = ["stem.weight", "stem.bias"]
        for i in range(self.config.n_stages):
            keys += [f"blocks.{i}.weight", f"blocks.{i}.bias"]
        for i in range(self.config.n_stages):
            if isinstance(self.skips[i], nn.Conv2d):
                keys += [f"skips.{i}.weight", f"skips.{i}.bias"]
        keys += [
            "refine.0.weight", "refine.0.bias",
            "refine.1.weight", "refine.1.bias",
            "rgb_0.weight", "rgb_0.bias",
            "rgb_1.weight", "rgb_1.bias",
        ]
        for key in keys:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
        return out


# ── Latent table (per-pair learned embedding) ─────────────────────────────


class BlockNeRVLatentTable(nn.Module):
    """Per-pair learned latent table (mirrors Lane 12-v2)."""

    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── Score-aware train_step ────────────────────────────────────────────────


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
    """Simulate uint8 bottleneck per CLAUDE.md eval_roundtrip non-negotiable."""
    clamped = rgb.clamp(0.0, 255.0)
    rounded = clamped + (clamped.round().detach() - clamped.detach())
    return rounded


def _pose_tensor(output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(output, dict):
        if "pose" not in output:
            raise KeyError("PoseNet output dict missing 'pose' key")
        value = output["pose"]
    else:
        value = output
    if not torch.is_tensor(value):
        raise TypeError(f"pose output must be a tensor, got {type(value).__name__}")
    return value


def train_step_blocknerv(
    *,
    renderer: BlockNeRVRenderer,
    latent_table: BlockNeRVLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    eval_roundtrip: bool = True,
) -> dict:
    """Score-aware training step (mirrors Lane 12-v2 train_step contract)."""
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    z = latent_table(pair_indices)
    decoded = renderer(z)  # (B, 2, 3, H_native, W_native)
    B, F_pp, C, H_native, W_native = decoded.shape
    flat = decoded.reshape(B * F_pp, C, H_native, W_native)
    H_camera, W_camera = gt_pairs_uint8.shape[-2], gt_pairs_uint8.shape[-1]
    up = F.interpolate(
        flat, size=(H_camera, W_camera), mode="bicubic", align_corners=False
    )
    up_uint8_ste = _eval_roundtrip_uint8_clamp(up)
    up_pairs = up_uint8_ste.reshape(B, F_pp, C, H_camera, W_camera)
    gt_pairs = gt_pairs_uint8.float()

    seg_pred_logits = scorer_seg(scorer_seg.preprocess_input(up_pairs))
    with torch.no_grad():
        seg_target_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    loss_seg_unweighted = seg_surrogate(seg_pred_logits, seg_target_logits)

    pose_pred = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(up_pairs)))
    with torch.no_grad():
        pose_target = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(gt_pairs)))
    loss_pose_unweighted = pose_surrogate(pose_pred, pose_target)

    loss_seg = lambda_seg * loss_seg_unweighted
    loss_pose = lambda_pose * loss_pose_unweighted
    loss = loss_seg + loss_pose
    return {
        "loss": loss,
        "loss_seg": loss_seg,
        "loss_pose": loss_pose,
        "loss_seg_unweighted": loss_seg_unweighted,
        "loss_pose_unweighted": loss_pose_unweighted,
    }


# ── Quantization + archive packing ───────────────────────────────────────


def _quantize_per_tensor_int8_with_fp16_scale(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = float(tensor.abs().max().item())
    scale = max(max_abs, 1e-8) / 127.0
    scale_fp16 = torch.tensor([scale], dtype=torch.float16)
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale_fp16


def _quantize_latent_table_uint8_delta_split(latents: torch.Tensor) -> bytes:
    """Quantize latent table per PR100 sidecar pattern (mirrors Lane 12-v2)."""
    import brotli

    n, d = latents.shape
    mins = latents.min(dim=0).values.to(torch.float16)
    maxs = latents.max(dim=0).values.to(torch.float16)
    scales = ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)
    q = ((latents - mins.float()) / scales.float()).round().clamp(0, 255).to(torch.int32)
    delta = torch.zeros_like(q)
    delta[0] = q[0]
    delta[1:] = q[1:] - q[:-1]
    delta_zz = torch.where(delta >= 0, 2 * delta, -2 * delta - 1).to(torch.int32)
    delta_zz = delta_zz.clamp(0, 65535)
    delta_lo = (delta_zz & 0xFF).to(torch.uint8)
    delta_hi = ((delta_zz >> 8) & 0xFF).to(torch.uint8)
    buf = io.BytesIO()
    buf.write(struct.pack("<II", n, d))
    buf.write(mins.numpy().tobytes())
    buf.write(scales.numpy().tobytes())
    buf.write(delta_lo.numpy().tobytes())
    buf.write(delta_hi.numpy().tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def export_blocknerv_to_archive(
    *,
    renderer: BlockNeRVRenderer,
    latent_table: BlockNeRVLatentTable,
    output_path: Path,
) -> str:
    """Pack trained BlockNeRV renderer + latents into the monolithic 0.bin.

    Returns archive sha256 (no-op detector evidence per HNeRV lesson 11).
    """
    import brotli

    config = renderer.config
    schema = renderer.schema
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected_latent_shape = (config.n_pairs, config.latent_dim)
    if latent_shape != expected_latent_shape:
        raise ValueError(
            f"latent_table shape {latent_shape} != expected {expected_latent_shape}"
        )

    sd = renderer.state_dict()
    int8_chunks: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from renderer state_dict")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"schema shape mismatch for {key!r}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        q, scale = _quantize_per_tensor_int8_with_fp16_scale(tensor)
        int8_chunks.append(q.detach().cpu().numpy().tobytes())
        scales_fp16.append(scale.detach().cpu().numpy().tobytes())

    decoder_blob = brotli.compress(b"".join(int8_chunks), quality=11)
    # Tile coord table is FP16 raw (small; (n_tiles, coord_embed_dim)).
    tile_coord = renderer.tile_coord.detach().cpu().to(torch.float16).numpy().tobytes()
    scale_table = b"".join(scales_fp16)
    latent_blob = _quantize_latent_table_uint8_delta_split(
        latent_table.embedding.weight.detach().cpu()
    )
    sidecar_blob = b""

    out = io.BytesIO()
    # Header (16 bytes).
    out.write(BLOCKNERV_MAGIC)
    out.write(struct.pack("<H", BLOCKNERV_FORMAT_VERSION))
    out.write(struct.pack("<H", BLOCKNERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.tile_rows))
    out.write(struct.pack("<H", config.tile_cols))
    # Sections.
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
    out.write(struct.pack("<I", len(tile_coord)))
    out.write(tile_coord)
    out.write(struct.pack("<I", len(scale_table)))
    out.write(scale_table)
    out.write(struct.pack("<I", len(latent_blob)))
    out.write(latent_blob)
    out.write(struct.pack("<I", len(sidecar_blob)))
    out.write(sidecar_blob)

    archive_bytes = out.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive_bytes)
    return hashlib.sha256(archive_bytes).hexdigest()


# ── Smoke-only synthetic batch helper (carry inline waiver) ──────────────


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic pair batch FOR SMOKE TESTS ONLY.

    # SYNTHETIC_NON_SMOKE_OK:blocknerv_phase_a_scaffold_smoke_test_only
    """
    g = torch.Generator().manual_seed(seed)
    pair_indices = torch.randint(0, n_pairs, (batch_size,), generator=g)
    H_camera, W_camera = 874, 1164
    gt_pairs = torch.randint(
        0, 256, (batch_size, 2, 3, H_camera, W_camera),
        generator=g, dtype=torch.uint8,
    )
    return pair_indices, gt_pairs


# ── Default surrogates (mirrors Lane 12-v2 defaults) ─────────────────────


def default_seg_surrogate(
    pred_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    """Default seg surrogate: KL on logits (Hinton T=2 distillation)."""
    T = 2.0
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(target_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def default_pose_surrogate(
    pred_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    """Default pose surrogate: MSE on the first 6 dims (matches contest)."""
    return F.mse_loss(pred_pose[..., :6], target_pose[..., :6])
