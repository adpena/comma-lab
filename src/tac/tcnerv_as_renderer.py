"""TCNeRV-as-renderer — temporal-convolutional NeRV substrate.

Per operator directive 2026-05-11 (NeRV-family expansion alternative direction)
+ CLAUDE.md HNeRV parity discipline (lesson 5: full RGB renderer; lesson 4:
inflate ≤200 LOC). TCNeRV replaces the per-pair (independent) latent decode
with a **temporal 1D conv stack** over the entire latent sequence, then a 2D
spatial decoder per pair. The temporal conv exploits inter-frame correlation
in the latent table — adjacent pairs in a smooth video should have similar
latents and can share parameters via temporal convolution.

Why temporal conv on the latent
-------------------------------
- **Smooth-trajectory prior**: most contest videos have smooth camera motion
  → adjacent latents should look similar. A small 1D temporal conv (e.g.
  kernel size 3) provides this smoothness as an architectural prior.
- **Compositional with HNeRV cluster**: TCNeRV's temporal conv operates on
  ANY per-pair latent stream, so it composes orthogonally with single-stage
  NeRV / BlockNeRV / FFNeRV / DSNeRV / HiNeRV decoders. Phase A pins it to
  Lane 12-v2-style spatial decoder for parity.
- **Compress-time decode**: at inference time, the temporal conv runs ONCE
  over the full 600-pair latent sequence; per-pair forward is unchanged.

Architecture (default config)
-----------------------------
- Per-pair latent table: (n_pairs, latent_dim) as in Lane 12-v2.
- Temporal conv (1D): processes the entire latent sequence (n_pairs,
  latent_dim) → (n_pairs, latent_dim) before spatial decode. Default 2 layers
  of 1D conv with kernel 3 + residual.
- 2D spatial decoder: same PixelShuffle stack as Lane 12-v2.
- Output: (B, 2, 3, 384, 512) RGB.

Inference contract
------------------
At inflate time, `inflate.py` reconstructs ALL latents from the latent_blob,
runs the temporal conv ONCE, then iterates per-pair forward through the 2D
decoder.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header +
    5 length-prefixed sections: spatial_decoder INT8+brotli, temporal_conv
    INT8+brotli, scale_table FP16, latent_blob uint8 delta-zigzag+brotli,
    sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_TCNERV in this module with
    schema_keys_in_order = TCNeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: tcnerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: trainer wires Lane 12-v2 train_step pattern with the
    temporal conv applied to a window of (B, latent_dim) before forward
  bolt_on_loc_budget: substrate_engineering (full TC-spatial decoder)
  no_op_detector_planned: export_to_archive returns sha256

CLAUDE.md compliance — same as Lane 12-v2.

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; temporal-conv prior on latent stream
exploits smooth-trajectory contest videos; compositional with HNeRV cluster;
pose-axis marginal 2.71× SegNet at PR106 r2 frontier]``. NOT a score claim
until [contest-CUDA] anchor lands.

Format ID: 0x64 (NeRV-family expansion magic 0xFE).
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


TCNERV_MAGIC: bytes = b"TCNV"
TCNERV_FORMAT_VERSION: int = 1
TCNERV_FORMAT_ID: int = 0x64


# ── Archive grammar ──────────────────────────────────────────────────────


ARCHIVE_GRAMMAR_TCNERV: dict = {
    "format_version": TCNERV_FORMAT_VERSION,
    "format_id": TCNERV_FORMAT_ID,
    "magic": TCNERV_MAGIC.decode("ascii"),
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
                ("temporal_kernel", "<H", 2),
                ("temporal_n_layers", "<H", 2),
            ],
        },
        {
            "name": "spatial_decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_schema_driven",
        },
        {
            "name": "temporal_conv_blob",
            "offset_after": "spatial_decoder_blob",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_temporal",
        },
        {
            "name": "scale_table",
            "offset_after": "temporal_conv_blob",
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
    "schema_keys_in_order": "TCNeRVRenderer.SCHEMA",
    "predicted_total_bytes": "155_000 to 180_000 [predicted; +small temporal conv params]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TCNeRVConfig:
    """Frozen config for TCNeRV-as-renderer Phase A.

    Attributes
    ----------
    latent_dim, base_channels, base_h, base_w, eval_size, n_pairs, n_stages,
    frames_per_pair
        Standard NeRV decoder knobs.
    temporal_kernel
        1D temporal conv kernel size. Default 3 (causal-symmetric ±1 neighbour).
    temporal_n_layers
        Number of 1D conv layers in the temporal stack. Default 2.
    temporal_residual
        If True (default), residual-add the temporal conv output to the input
        (so the temporal stack is a corrective refinement of the raw latents).
    lambda_seg, lambda_pose
        Score-aware loss weights.
    cuda_required
        If True (default), train_step raises if CUDA unavailable.
    """

    latent_dim: int = 16
    base_channels: int = 36
    base_h: int = 6
    base_w: int = 8
    eval_size: tuple[int, int] = (384, 512)
    n_pairs: int = 600
    n_stages: int = 6
    frames_per_pair: int = 2
    temporal_kernel: int = 3
    temporal_n_layers: int = 2
    temporal_residual: bool = True
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.n_stages != 6:
            raise ValueError("Phase A pinned at n_stages=6")
        if self.frames_per_pair != 2:
            raise ValueError("Phase A pinned at frames_per_pair=2")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError(
                f"temporal_kernel must be odd positive, got {self.temporal_kernel}"
            )
        if self.temporal_n_layers <= 0:
            raise ValueError(f"temporal_n_layers must be positive, got {self.temporal_n_layers}")


# ── Temporal conv stack ──────────────────────────────────────────────────


class TemporalConvStack(nn.Module):
    """1D temporal conv stack operating on (n_pairs, latent_dim) sequences.

    Treats latent_dim as channels and n_pairs as time. Padding is symmetric
    so the output sequence length matches input.
    """

    def __init__(self, latent_dim: int, kernel: int = 3, n_layers: int = 2,
                 residual: bool = True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel = kernel
        self.n_layers = n_layers
        self.residual = residual
        layers = []
        pad = kernel // 2
        for _ in range(n_layers):
            layers.append(nn.Conv1d(latent_dim, latent_dim, kernel, padding=pad))
        self.layers = nn.ModuleList(layers)

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """``(n_pairs, latent_dim) → (n_pairs, latent_dim)``."""
        if latent_seq.dim() != 2 or latent_seq.shape[1] != self.latent_dim:
            raise ValueError(
                f"TemporalConvStack expected (n_pairs, {self.latent_dim}), got {tuple(latent_seq.shape)}"
            )
        # (n_pairs, latent_dim) → (1, latent_dim, n_pairs) for Conv1d.
        x = latent_seq.t().unsqueeze(0)
        identity = x
        for layer in self.layers:
            x = torch.tanh(layer(x))
        if self.residual:
            x = x + identity
        # (1, latent_dim, n_pairs) → (n_pairs, latent_dim).
        return x.squeeze(0).t().contiguous()


# ── Renderer ──────────────────────────────────────────────────────────────


class TCNeRVRenderer(nn.Module):
    """TCNeRV — temporal-conv on the latent table + 2D spatial decoder.

    Inference path:
        1. Reconstruct latent table from archive (n_pairs, latent_dim).
        2. Apply ``temporal_conv`` ONCE to the entire sequence.
        3. For each pair, forward through the spatial decoder.

    The spatial decoder mirrors Lane 12-v2's PixelShuffle stack so any
    cross-substrate diagnostics (param counts, sensitivity, etc.) compose.
    """

    def __init__(self, config: TCNeRVConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_conv = TemporalConvStack(
            latent_dim=config.latent_dim,
            kernel=config.temporal_kernel,
            n_layers=config.temporal_n_layers,
            residual=config.temporal_residual,
        )
        C = config.base_channels
        self.channels: list[int] = [
            C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)
        ]
        self.stem = nn.Linear(
            config.latent_dim, self.channels[0] * config.base_h * config.base_w
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
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def spatial_forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, latent_dim) → (B, 2, 3, H, W)`` after temporal conv applied."""
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"spatial_forward expected (B, {self.config.latent_dim}), got {tuple(z.shape)}"
            )
        B = z.shape[0]
        x = self.stem(z).view(
            B, self.channels[0], self.config.base_h, self.config.base_w
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
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)

    def forward(self, latent_seq: torch.Tensor,
                pair_indices: torch.Tensor | None = None) -> torch.Tensor:
        """``latent_seq (n_pairs, latent_dim), pair_indices? (B,) → (B, 2, 3, H, W)``.

        If ``pair_indices`` is None, decode ALL pairs in order. If provided, the
        temporal conv is still applied to the WHOLE sequence (its output is
        global / not per-pair) but only the selected pairs are spatially decoded.
        """
        z_temporal = self.temporal_conv(latent_seq)
        if pair_indices is None:
            return self.spatial_forward(z_temporal)
        return self.spatial_forward(z_temporal.index_select(0, pair_indices))

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        keys: list[str] = []
        for i in range(self.config.temporal_n_layers):
            keys += [f"temporal_conv.layers.{i}.weight", f"temporal_conv.layers.{i}.bias"]
        keys += ["stem.weight", "stem.bias"]
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


# ── Latent table ─────────────────────────────────────────────────────────


class TCNeRVLatentTable(nn.Module):
    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def full_sequence(self) -> torch.Tensor:
        """Return the full latent table (n_pairs, latent_dim)."""
        return self.embedding.weight

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── train_step ───────────────────────────────────────────────────────────


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
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


def train_step_tcnerv(
    *,
    renderer: TCNeRVRenderer,
    latent_table: TCNeRVLatentTable,
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
    """Score-aware training step.

    The temporal conv is applied to the FULL latent table on every step (so
    gradients flow back through both the temporal conv and the per-pair
    embeddings, including non-batched pairs — this is a feature, since the
    temporal smoothness should be learned over all pairs).
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    full_seq = latent_table.full_sequence()
    decoded = renderer(full_seq, pair_indices)  # (B, 2, 3, H, W)
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


def export_tcnerv_to_archive(
    *,
    renderer: TCNeRVRenderer,
    latent_table: TCNeRVLatentTable,
    output_path: Path,
) -> str:
    """Pack TCNeRV weights into 0.bin. Splits the spatial decoder and
    temporal conv into their own brotli-compressed sections so an inflate
    parser can selectively load each."""
    import brotli

    config = renderer.config
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected = (config.n_pairs, config.latent_dim)
    if latent_shape != expected:
        raise ValueError(f"latent_table shape {latent_shape} != expected {expected}")

    sd = renderer.state_dict()
    spatial_int8: list[bytes] = []
    temporal_int8: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in renderer.schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from state_dict")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"schema shape mismatch for {key!r}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        q, scale = _quantize_per_tensor_int8_with_fp16_scale(tensor)
        if key.startswith("temporal_conv."):
            temporal_int8.append(q.detach().cpu().numpy().tobytes())
        else:
            spatial_int8.append(q.detach().cpu().numpy().tobytes())
        scales_fp16.append(scale.detach().cpu().numpy().tobytes())

    spatial_decoder_blob = brotli.compress(b"".join(spatial_int8), quality=11)
    temporal_conv_blob = brotli.compress(b"".join(temporal_int8), quality=11) if temporal_int8 else b""
    scale_table = b"".join(scales_fp16)
    latent_blob = _quantize_latent_table_uint8_delta_split(
        latent_table.embedding.weight.detach().cpu()
    )
    sidecar_blob = b""

    out = io.BytesIO()
    out.write(TCNERV_MAGIC)
    out.write(struct.pack("<H", TCNERV_FORMAT_VERSION))
    out.write(struct.pack("<H", TCNERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.temporal_kernel))
    out.write(struct.pack("<H", config.temporal_n_layers))
    out.write(struct.pack("<I", len(spatial_decoder_blob)))
    out.write(spatial_decoder_blob)
    out.write(struct.pack("<I", len(temporal_conv_blob)))
    out.write(temporal_conv_blob)
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


# ── Smoke + default surrogates ──────────────────────────────────────────


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """# SYNTHETIC_NON_SMOKE_OK:tcnerv_phase_a_scaffold_smoke_test_only"""
    g = torch.Generator().manual_seed(seed)
    pair_indices = torch.randint(0, n_pairs, (batch_size,), generator=g)
    gt_pairs = torch.randint(
        0, 256, (batch_size, 2, 3, 874, 1164), generator=g, dtype=torch.uint8,
    )
    return pair_indices, gt_pairs


def default_seg_surrogate(
    pred_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    T = 2.0
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(target_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def default_pose_surrogate(
    pred_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(pred_pose[..., :6], target_pose[..., :6])
