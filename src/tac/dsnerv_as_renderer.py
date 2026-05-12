"""DSNeRV-as-renderer — diffusion-supervised NeRV substrate.

Per operator directive 2026-05-11 (NeRV-family expansion) + CLAUDE.md HNeRV
parity discipline (lesson 5: full RGB renderer; lesson 4: inflate ≤200 LOC).
DSNeRV adds a **denoising training objective** on top of the NeRV substrate:
during training the latent + noisy intermediate is iteratively refined toward
the target frame following a learned noise schedule. At inference time the
network performs a single forward pass (the noise schedule is a TRAINING
regularizer; inflate stays cheap).

Why diffusion-supervised
------------------------
- **Implicit ensemble training**: the network sees the same target many
  times under different noise levels per epoch — gives a much stronger
  training signal at fixed wall clock vs single-pass training.
- **Better generalization**: noise schedules act as data augmentation in
  representation space, reducing overfit to the training video.
- **Inference unchanged**: the network is the same NeRV decoder; the noise
  schedule is consumed only by the trainer's `train_step`. `inflate.py`
  calls one forward pass per latent (no iterative denoising at inference).

Architecture (default config)
-----------------------------
- Latent: (B, 16) per-pair as in Lane 12-v2.
- Noise schedule: cosine schedule with T=10 training steps. Each train step
  samples a random `t ∈ [1, T]` and corrupts the latent with noise level
  `σ_t`; the network predicts the clean RGB target.
- NeRV decoder: same PixelShuffle stack as Lane 12-v2 (so single-forward
  inference parity is preserved).
- Output: (B, 2, 3, 384, 512) RGB.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (12-byte fixed header +
    4 length-prefixed sections: decoder_blob INT8+brotli, scale_table FP16,
    latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_DSNERV in this module with
    schema_keys_in_order = DSNeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: dsnerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: trainer wires Lane 12-v2 train_step pattern + noise schedule
  bolt_on_loc_budget: substrate_engineering (full denoising-trained renderer)
  no_op_detector_planned: export_to_archive returns sha256

CLAUDE.md compliance — same as Lane 12-v2 (eval_roundtrip / EMA / no MPS / etc).

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; diffusion-supervised training acts as
implicit ensemble per Karras 2022 EDM noise schedule; pose-axis marginal
2.71× SegNet at PR106 r2 frontier]``. NOT a score claim until [contest-CUDA]
anchor lands.

Format ID: 0x62 (NeRV-family expansion magic 0xFE).
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Magic + format ────────────────────────────────────────────────────────


DSNERV_MAGIC: bytes = b"DSNV"
DSNERV_FORMAT_VERSION: int = 1
DSNERV_FORMAT_ID: int = 0x62


# ── Archive grammar ──────────────────────────────────────────────────────


ARCHIVE_GRAMMAR_DSNERV: dict = {
    "format_version": DSNERV_FORMAT_VERSION,
    "format_id": DSNERV_FORMAT_ID,
    "magic": DSNERV_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 12,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("version", "<H", 2),
                ("format_id", "<H", 2),
                ("latent_dim", "<H", 2),
                ("n_pairs", "<H", 2),
            ],
        },
        {
            "name": "decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_schema_driven",
        },
        {
            "name": "scale_table",
            "offset_after": "decoder_blob",
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
    "schema_keys_in_order": "DSNeRVRenderer.SCHEMA",
    "predicted_total_bytes": "150_000 to 175_000 [predicted; same arch as Lane 12-v2]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DSNeRVConfig:
    """Frozen config for DSNeRV-as-renderer Phase A.

    Attributes
    ----------
    latent_dim, base_channels, base_h, base_w, eval_size, n_pairs, n_stages,
    frames_per_pair
        Same architectural knobs as Lane 12-v2.
    n_diffusion_steps
        Length of the noise schedule (T). Default 10.
    noise_schedule
        One of "cosine" (default; Nichol & Dhariwal 2021) or "linear"
        (Ho et al. 2020 DDPM).
    sigma_max
        Maximum noise stddev (in latent-space units). Default 1.0.
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
    n_diffusion_steps: int = 10
    noise_schedule: str = "cosine"
    sigma_max: float = 1.0
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_stages != 6:
            raise ValueError(f"Phase A pinned at n_stages=6, got {self.n_stages}")
        if self.frames_per_pair != 2:
            raise ValueError(f"Phase A pinned at frames_per_pair=2, got {self.frames_per_pair}")
        if self.n_diffusion_steps <= 0:
            raise ValueError(f"n_diffusion_steps must be positive, got {self.n_diffusion_steps}")
        if self.noise_schedule not in {"cosine", "linear"}:
            raise ValueError(f"noise_schedule must be 'cosine' or 'linear', got {self.noise_schedule}")
        if self.sigma_max <= 0:
            raise ValueError(f"sigma_max must be positive, got {self.sigma_max}")


# ── Noise schedule ───────────────────────────────────────────────────────


class NoiseSchedule:
    """Per-step noise stddev table for diffusion-supervised training.

    Linear: σ_t = σ_max * t / T.
    Cosine: σ_t = σ_max * (1 - cos(π*t/(2T))).
    """

    def __init__(self, n_steps: int, sigma_max: float, schedule: str = "cosine") -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if sigma_max <= 0:
            raise ValueError("sigma_max must be positive")
        if schedule == "linear":
            sigmas = sigma_max * torch.arange(1, n_steps + 1).float() / n_steps
        elif schedule == "cosine":
            t = torch.arange(1, n_steps + 1).float()
            sigmas = sigma_max * (1.0 - torch.cos(math.pi * t / (2.0 * n_steps)))
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        self.sigmas = sigmas
        self.n_steps = n_steps
        self.schedule = schedule

    def sigma_at(self, step: int) -> float:
        if step < 1 or step > self.n_steps:
            raise ValueError(f"step out of range [1, {self.n_steps}], got {step}")
        return float(self.sigmas[step - 1])

    def sample_step(self, generator: torch.Generator | None = None) -> int:
        """Uniform sample of t in [1, n_steps]."""
        return int(torch.randint(1, self.n_steps + 1, (1,), generator=generator).item())


# ── Renderer (same arch as Lane 12-v2; the noise lives in the trainer) ───


class DSNeRVRenderer(nn.Module):
    """DSNeRV — same NeRV decoder arch; diffusion supervision is in train_step.

    Forward signature: ``z (B, latent_dim) → (B, 2, 3, H, W)``. At inference
    time, callers pass clean latents; during training, the noise is added to
    the latent BEFORE forward and the network learns to denoise toward the
    target frame.
    """

    def __init__(self, config: DSNeRVConfig) -> None:
        super().__init__()
        self.config = config
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}), got {tuple(z.shape)}"
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

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
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


# ── Latent table ─────────────────────────────────────────────────────────


class DSNeRVLatentTable(nn.Module):
    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── train_step (diffusion-supervised) ────────────────────────────────────


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


def train_step_dsnerv(
    *,
    renderer: DSNeRVRenderer,
    latent_table: DSNeRVLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    noise_schedule: NoiseSchedule,
    diffusion_step: int | None = None,
    eval_roundtrip: bool = True,
    generator: torch.Generator | None = None,
) -> dict:
    """Score-aware training step with diffusion supervision.

    Adds noise to the latent before forward; the network learns to denoise
    toward the target. At inference time, callers pass clean latents and the
    noise schedule is unused (network has learned to be robust to noise).
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    z_clean = latent_table(pair_indices)
    # Sample diffusion step + add noise.
    if diffusion_step is None:
        t = noise_schedule.sample_step(generator=generator)
    else:
        t = diffusion_step
    sigma = noise_schedule.sigma_at(t)
    noise = torch.randn_like(z_clean) * sigma
    z_noisy = z_clean + noise
    decoded = renderer(z_noisy)

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
        "diffusion_step": t,
        "diffusion_sigma": sigma,
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


def export_dsnerv_to_archive(
    *,
    renderer: DSNeRVRenderer,
    latent_table: DSNeRVLatentTable,
    output_path: Path,
) -> str:
    import brotli

    config = renderer.config
    schema = renderer.schema
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected = (config.n_pairs, config.latent_dim)
    if latent_shape != expected:
        raise ValueError(f"latent_table shape {latent_shape} != expected {expected}")

    sd = renderer.state_dict()
    int8_chunks: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from state_dict")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"schema shape mismatch for {key!r}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        q, scale = _quantize_per_tensor_int8_with_fp16_scale(tensor)
        int8_chunks.append(q.detach().cpu().numpy().tobytes())
        scales_fp16.append(scale.detach().cpu().numpy().tobytes())

    decoder_blob = brotli.compress(b"".join(int8_chunks), quality=11)
    scale_table = b"".join(scales_fp16)
    latent_blob = _quantize_latent_table_uint8_delta_split(
        latent_table.embedding.weight.detach().cpu()
    )
    sidecar_blob = b""

    out = io.BytesIO()
    out.write(DSNERV_MAGIC)
    out.write(struct.pack("<H", DSNERV_FORMAT_VERSION))
    out.write(struct.pack("<H", DSNERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
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
    """# SYNTHETIC_NON_SMOKE_OK:dsnerv_phase_a_scaffold_smoke_test_only"""
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
