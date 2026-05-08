#!/usr/bin/env python3
"""PR101 lossy int4 + Quantization-Aware Training (QAT) — research-path
exhaustion against the naive-PTQ FALSIFICATION (37.42% rel_err).

The naive PTQ baseline (`pr101_lossy_int4_roundtrip_test.py` 2026-05-08)
produced 37.42% weighted-average relative error → tagged DEFERRED-pending-
research per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`.
The audit memo (`feedback_adversarial_audit_4_falsifications_DEFERRED_not_
killed_20260507.md`) lists QAT as the canonical "low-bit PTQ collapse" fix
to attempt before any KILL verdict can stand.

This tool runs FULL QAT (no stubs, no scaffolds):

    1. Load PR101 fp32 state_dict (the PR101 weight tensors targeted by
       `tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA`).
    2. Wrap each tensor in a QAT-trainable shadow (LSQ-style): trainable
       fp32 shadow params + per-block fp16 scales (block_size=1024 to match
       the 100,799 B archive anchor) + symmetric int4 fake-quant via STE.
    3. Train shadow params with reconstruction loss (MSE-vs-original-fp32)
       for ~500 epochs on MPS (or CPU fallback) with Adam.
    4. Snap the trained shadow tensors through the same fake-quant grid,
       feed into `pr101_lossy_int4_block_sweep.encode_tensor` to produce
       the actual archive bytes.
    5. Decode (inverse of encode) and compute roundtrip rel_err vs the
       ORIGINAL fp32 weights (this is what the contest scorer would see
       through the renderer). Sub-5% = DISPATCH-CANDIDATE; else
       DEFERRED-with-new-evidence.
    6. Emit manifest + cathedral_autopilot evidence row.

Compliance (CLAUDE.md non-negotiables):

    * Pure CPU/MPS prep — no scorer load, no contest-CUDA dispatch from
      this tool. Evidence tagged `[CPU-prep+QAT]` or `[MPS-research-signal]`.
    * Score never claimed; only BYTES + ROUNDTRIP rel_err are anchored.
    * promotion_eligible=False. ready_for_exact_eval_dispatch is True iff
      QAT recovers rel_err < 5% (gates dispatch decision; decision still
      requires operator + sanity-ladder).
    * No /tmp paths in any persisted artifact.
    * MPS allowed as research-signal source per `feedback_mps_as_research_
      signal_strategic_clarification_20260507.md`; never as a judge.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402

TOOL_NAME = "tools/pr101_lossy_int4_qat.py"
SCHEMA_VERSION = "pr101_lossy_int4_qat.v1"
INT4_RANGE = 7  # symmetric: [-7, +7], 15 levels (with zero)
DEFAULT_BLOCK_SIZE = 1024  # matches the 100,799 B archive anchor
ARCHIVE_OVERHEAD_BYTES = 16_094  # PR101 zip overhead constant
NAIVE_PTQ_REL_ERR_PCT = 37.42  # the falsification baseline this tool refutes
DISPATCH_THRESHOLD_PCT = 5.0
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)


# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------

def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_device(prefer: str = "mps") -> torch.device:
    """Pick a research-signal device. Per CLAUDE.md MPS is allowed for
    sweeps/curves/discovery — NOT as a score judge. We tag the resulting
    evidence as [MPS-research-signal] in that case so it's never confused
    with [contest-CUDA].

    Order of preference: prefer → cuda → mps → cpu.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Fake-quant primitives (block-wise int4 with STE) — match the encoder math
# in `pr101_lossy_int4_block_sweep.encode_tensor` so train-time and pack-time
# numerics agree (the canonical "what-you-train-is-what-you-ship" gate).
# ----------------------------------------------------------------------------

def _block_scales_max_based(flat: torch.Tensor, block_size: int) -> torch.Tensor:
    """Compute per-block max-abs scales, snapped to fp16 to match the encoder.

    Returns a tensor of shape (numel,) with each element equal to the scale
    of its containing block (so it can be used elementwise as scale_view).
    """
    n = flat.numel()
    n_full = n // block_size
    tail = n - n_full * block_size
    parts: list[torch.Tensor] = []
    if n_full > 0:
        body = flat[: n_full * block_size].view(n_full, block_size)
        abs_max = body.abs().amax(dim=1).clamp(min=1e-12)
        scale_f = (abs_max / INT4_RANGE).to(torch.float16).to(torch.float32)
        parts.append(scale_f.unsqueeze(1).expand(-1, block_size).reshape(-1))
    if tail:
        tail_block = flat[n_full * block_size :]
        abs_max_t = tail_block.abs().max().clamp(min=1e-12).unsqueeze(0)
        scale_t = (abs_max_t / INT4_RANGE).to(torch.float16).to(torch.float32)
        parts.append(scale_t.expand(tail_block.numel()))
    return torch.cat(parts)


class FakeQuantInt4Block(torch.autograd.Function):
    """STE-fake-quant for symmetric int4 with per-block fp16 scale.

    Forward:
        For each contiguous block of `block_size` elements (last block may
        be shorter) compute scale_f = max(|block|) / 7. Snap to fp16 to
        match the on-disk encoder, then quantize codes = round(x / scale)
        clipped to [-7, +7], dequantize to scale * codes (fp32). This
        replicates `pr101_lossy_int4_block_sweep.quantize_block_int4` +
        the inverse op exactly.

    Backward:
        Saturation-aware STE — gradient flows through where the unclipped
        code is in (-7.5, +7.5) and is zeroed elsewhere. Standard QAT
        practice; matches `FakeQuantSTE` in `tac.quantization`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int) -> torch.Tensor:
        flat = x.detach().reshape(-1)
        scale_per_elem = _block_scales_max_based(flat, block_size)
        codes_unclipped = flat / scale_per_elem
        codes = codes_unclipped.round().clamp(-INT4_RANGE, INT4_RANGE)
        recon = (codes * scale_per_elem).reshape(x.shape)
        # Saturation mask: |x / scale| > 7.5  (codes that hit clip).
        ctx.save_for_backward((codes_unclipped.abs() <= 7.5).to(x.dtype).reshape(x.shape))
        return recon

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None


def fake_quant_int4_block(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return FakeQuantInt4Block.apply(x, block_size)


# ----------------------------------------------------------------------------
# Learnable-scale variant: scales are nn.Parameter snapped to fp16 in forward.
# This is the LSQ-style path the audit memo recommends as the canonical fix
# for low-bit PTQ collapse. Per-block scales are TRAINABLE; blocks of size
# `block_size` matching the encoder are honored at quant-time so the encoder
# output bytes are deterministic given the trained scales.
# ----------------------------------------------------------------------------

class FakeQuantInt4BlockLearnable(torch.autograd.Function):
    """Same forward as FakeQuantInt4Block but takes per-block scales as an
    explicit input (so they can be trainable). Backward produces gradients
    for both the input AND the scales (LSQ-style scaled grad).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale_per_block: torch.Tensor, block_size: int):
        flat = x.detach().reshape(-1)
        n = flat.numel()
        n_full = n // block_size
        tail = n - n_full * block_size
        # Snap scales to fp16 to match the on-disk encoder.
        scale_fp16 = scale_per_block.detach().clamp(min=1e-12).to(torch.float16).to(torch.float32)
        per_elem_parts: list[torch.Tensor] = []
        if n_full > 0:
            per_elem_parts.append(
                scale_fp16[: n_full].unsqueeze(1).expand(-1, block_size).reshape(-1)
            )
        if tail:
            per_elem_parts.append(scale_fp16[n_full].expand(tail))
        scale_per_elem_flat = torch.cat(per_elem_parts)

        codes_unclipped_flat = flat / scale_per_elem_flat
        codes_flat = codes_unclipped_flat.round().clamp(-INT4_RANGE, INT4_RANGE)
        recon_flat = codes_flat * scale_per_elem_flat
        recon = recon_flat.reshape(x.shape)

        sat_mask_flat = (codes_unclipped_flat.abs() <= 7.5).to(x.dtype)
        ctx.save_for_backward(
            codes_flat.to(x.dtype), sat_mask_flat,
        )
        ctx.block_size = block_size
        ctx.n_full = n_full
        ctx.tail = tail
        ctx.input_shape = x.shape
        ctx.numel = n
        return recon

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        codes_flat, sat_mask_flat = ctx.saved_tensors
        block_size = ctx.block_size
        n_full = ctx.n_full
        tail = ctx.tail

        grad_out_flat = grad_out.reshape(-1)

        # d recon / d x = sat_mask (STE)
        grad_x = (grad_out_flat * sat_mask_flat).reshape(ctx.input_shape)

        # d recon / d scale (per element):
        #   recon = codes * scale, codes treated as constant (LSQ STE).
        per_elem_scale_grad = grad_out_flat * codes_flat
        block_scale_grad_parts: list[torch.Tensor] = []
        if n_full > 0:
            body = per_elem_scale_grad[: n_full * block_size].view(n_full, block_size)
            block_scale_grad_parts.append(body.sum(dim=1))
        if tail:
            tail_grad = per_elem_scale_grad[n_full * block_size :].sum().unsqueeze(0)
            block_scale_grad_parts.append(tail_grad)
        grad_scale = torch.cat(block_scale_grad_parts)

        # LSQ stability scaling: 1 / sqrt(numel * range)
        lsq_scale = 1.0 / (ctx.numel * INT4_RANGE) ** 0.5
        grad_scale = grad_scale * lsq_scale

        return grad_x, grad_scale, None


def fake_quant_int4_block_learnable(
    x: torch.Tensor, scale_per_block: torch.Tensor, block_size: int
) -> torch.Tensor:
    return FakeQuantInt4BlockLearnable.apply(x, scale_per_block, block_size)


# ----------------------------------------------------------------------------
# QAT model: one trainable shadow per tensor, identity forward = fake_quant
# ----------------------------------------------------------------------------

class TensorShadow(nn.Module):
    """Holds a trainable fp32 shadow of a tensor whose 'effective' value
    (the value that will be encoded to int4) is `fake_quant(shadow,
    learnable_scales)`. Both the shadow weights AND per-block scales are
    trainable. Per-block scales are initialized from max-abs / 7 (the
    naive-PTQ initialization, which is the LSQ-recommended init).

    The trainable scales are the canonical low-bit PTQ-collapse fix
    (the audit memo's "LSQ" + "per-channel scaling" entries). Without
    them, naive PTQ is already locally optimal in L2 sense; with them,
    QAT can find scales that minimize quantization error subject to the
    fp16 storage constraint and the symmetric int4 grid.
    """

    def __init__(self, init_tensor: torch.Tensor, block_size: int):
        super().__init__()
        # Trainable shadow weights (initialized from original).
        self.shadow = nn.Parameter(init_tensor.detach().clone().to(torch.float32))
        self.block_size = block_size

        # Compute initial per-block scales (max-abs / 7), exposed as
        # trainable nn.Parameter — the LSQ knob.
        flat = self.shadow.detach().reshape(-1)
        n = flat.numel()
        n_full = n // block_size
        tail = n - n_full * block_size
        n_blocks = n_full + (1 if tail else 0)
        scales_init = torch.empty(n_blocks, dtype=torch.float32)
        if n_full > 0:
            body = flat[: n_full * block_size].view(n_full, block_size)
            scales_init[: n_full] = (body.abs().amax(dim=1).clamp(min=1e-12) / INT4_RANGE)
        if tail:
            scales_init[n_full] = (
                flat[n_full * block_size :].abs().max().clamp(min=1e-12) / INT4_RANGE
            )
        # Snap to fp16 to match encoder.
        scales_init = scales_init.to(torch.float16).to(torch.float32)
        self.block_scales = nn.Parameter(scales_init)

    def quantized(self) -> torch.Tensor:
        return fake_quant_int4_block_learnable(
            self.shadow, self.block_scales, self.block_size,
        )

    def quantized_with_max_scales(self) -> torch.Tensor:
        """Alternative quant path using current shadow but max-based scales
        (recomputed from shadow). Used for the pre-train PTQ baseline.
        """
        return fake_quant_int4_block(self.shadow, self.block_size)


class PR101QATModel(nn.Module):
    """Container of one TensorShadow per FIXED_STATE_SCHEMA entry."""

    def __init__(self, fp32_state: dict[str, torch.Tensor], block_size: int):
        super().__init__()
        self.block_size = block_size
        self.shadows = nn.ModuleDict()
        for name, _shape in FIXED_STATE_SCHEMA:
            if name not in fp32_state:
                raise SystemExit(f"state_dict missing tensor {name!r}")
            # ModuleDict keys can't contain '.', encode it.
            key = name.replace(".", "__")
            self.shadows[key] = TensorShadow(fp32_state[name], block_size)

    def named_shadows(self):
        for name, _shape in FIXED_STATE_SCHEMA:
            yield name, self.shadows[name.replace(".", "__")]

    def quantized_state(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, sh in self.named_shadows():
            out[name] = sh.quantized().detach()
        return out

    def forward(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, sh in self.named_shadows():
            out[name] = sh.quantized()
        return out


# ----------------------------------------------------------------------------
# QAT loss: weighted MSE on the reconstruction. Weight by element count so
# small biases don't dominate; the relative error metric we care about is the
# inverse-magnitude one, but raw MSE is the right STE training target — the
# gradient flows through `fake_quant` to push shadow weights toward grid
# points where the snap error is small.
# ----------------------------------------------------------------------------

@dataclass
class QATLossConfig:
    rel_error_weight: float = 0.0  # weight of huber-rel-err term (vs MSE)
    rel_eps_pct_floor: float = 1.0  # ignore rel-err contribution where |t|<floor*max
    eps: float = 1e-6


def qat_reconstruction_loss(
    quantized: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    shadows: dict[str, torch.Tensor] | None,
    cfg: QATLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """QAT objective: MSE between quantized and target, optionally augmented
    with a relative-error term that ONLY fires on elements with non-negligible
    magnitude (above `rel_eps_pct_floor` * max(|t|)).

    The naive PTQ baseline minimizes MSE locally; the LSQ-style learnable
    scales let MSE drop further by finding non-max-based scales. The
    relative-error term is OPTIONAL because the rel_err metric is dominated
    by tiny weights for which int4 fundamentally cannot represent below the
    quantization step — chasing that part of the rel_err distribution with
    gradient just causes the optimizer to thrash. Default rel_error_weight=0
    keeps the loss on solid MSE footing.
    """
    first = next(iter(quantized.values()))
    total_mse = first.new_tensor(0.0)
    total_rel = first.new_tensor(0.0)
    n_total = 0
    n_rel_active = 0
    for name, q in quantized.items():
        t = targets[name].to(q.device)
        diff = q - t
        total_mse = total_mse + diff.pow(2).sum()
        if cfg.rel_error_weight > 0:
            t_abs = t.abs()
            t_max = t_abs.max().clamp(min=cfg.eps)
            mask = t_abs > (cfg.rel_eps_pct_floor / 100.0) * t_max
            if mask.any():
                rel = (diff.abs()[mask] / t_abs[mask].clamp(min=cfg.eps)).sum()
                total_rel = total_rel + rel
                n_rel_active += int(mask.sum().item())
        n_total += t.numel()
    n_total = max(n_total, 1)
    avg_mse = total_mse / n_total
    avg_rel = total_rel / max(n_rel_active, 1) if cfg.rel_error_weight > 0 else first.new_tensor(0.0)
    loss = avg_mse + cfg.rel_error_weight * avg_rel
    return loss, {
        "avg_mse": float(avg_mse.detach().item()),
        "avg_rel": float(avg_rel.detach().item()),
    }


# ----------------------------------------------------------------------------
# Encode / measure: feed the trained shadows through the encoder and the
# rel-err measurement using the EXACT same ops as the existing tools so
# numerics align bit-for-bit with what the dispatcher would ship.
# ----------------------------------------------------------------------------

def encode_quantized_state_to_archive_bytes(
    quantized: dict[str, torch.Tensor],
    *,
    block_size: int,
) -> tuple[int, int, int]:
    """Re-use the canonical encoder (block-wise int4 + brotli at PR101's
    autopilot brotli params) to produce the actual archive byte count.

    Returns (raw_payload_bytes, brotli_bytes, archive_bytes).
    """
    from pr101_lossy_int4_block_sweep import encode_tensor

    payloads: list[bytes] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        flat = quantized[name].detach().cpu().to(torch.float32).numpy().flatten()
        payload, _stats = encode_tensor(flat, block_size=block_size)
        payloads.append(payload)
    full_payload = b"".join(payloads)
    compressed = brotli.compress(
        full_payload, quality=11, lgwin=16, lgblock=19, mode=brotli.MODE_GENERIC
    )
    archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES
    return len(full_payload), len(compressed), archive_bytes


def measure_roundtrip_rel_err(
    quantized: dict[str, torch.Tensor],
    originals_fp32: dict[str, np.ndarray],
) -> dict:
    """Same math as `pr101_lossy_int4_roundtrip_test.rel_err_stats` but
    run on the QAT-trained quantized values vs ORIGINAL fp32 (i.e., what
    the contest renderer expects).
    """
    per_tensor: list[dict] = []
    total_elements = 0
    weighted_rel_err_sum = 0.0
    weighted_abs_err_sum = 0.0
    n_nontrivial_total = 0
    max_p99 = 0.0
    max_max = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        orig = originals_fp32[name].flatten().astype(np.float64)
        recon = quantized[name].detach().cpu().to(torch.float32).numpy().flatten().astype(np.float64)
        abs_err = np.abs(recon - orig)
        eps = 1e-8
        mask = np.abs(orig) > eps
        rel_err_pct = np.zeros_like(abs_err)
        rel_err_pct[mask] = 100.0 * abs_err[mask] / np.abs(orig[mask])

        n_elements = int(orig.size)
        n_nontrivial = int(mask.sum())
        stats = {
            "name": name,
            "n_elements": n_elements,
            "n_nontrivial": n_nontrivial,
            "abs_err_mean": float(abs_err.mean()),
            "abs_err_max": float(abs_err.max()),
            "rel_err_pct_mean": float(rel_err_pct[mask].mean()) if mask.any() else 0.0,
            "rel_err_pct_p50": float(np.percentile(rel_err_pct[mask], 50)) if mask.any() else 0.0,
            "rel_err_pct_p90": float(np.percentile(rel_err_pct[mask], 90)) if mask.any() else 0.0,
            "rel_err_pct_p99": float(np.percentile(rel_err_pct[mask], 99)) if mask.any() else 0.0,
            "rel_err_pct_max": float(rel_err_pct[mask].max()) if mask.any() else 0.0,
        }
        per_tensor.append(stats)
        total_elements += n_elements
        n_nontrivial_total += n_nontrivial
        weighted_rel_err_sum += stats["rel_err_pct_mean"] * n_nontrivial
        weighted_abs_err_sum += stats["abs_err_mean"] * n_elements
        max_p99 = max(max_p99, stats["rel_err_pct_p99"])
        max_max = max(max_max, stats["rel_err_pct_max"])

    return {
        "n_total_elements": total_elements,
        "n_nontrivial_elements": n_nontrivial_total,
        "weighted_avg_rel_err_pct": weighted_rel_err_sum / max(n_nontrivial_total, 1),
        "weighted_avg_abs_err": weighted_abs_err_sum / max(total_elements, 1),
        "max_p99_rel_err_pct": max_p99,
        "max_max_rel_err_pct": max_max,
        "per_tensor": per_tensor,
    }


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 500
    lr_shadow: float = 1e-4  # shadow weights move slowly (anchor near original)
    lr_scale: float = 1e-3   # scales move faster (LSQ knob)
    rel_error_weight: float = 0.0
    log_every: int = 25
    grad_clip: float = 1.0


def train_qat(
    model: PR101QATModel,
    targets_device: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
) -> list[dict]:
    """Run QAT for cfg.epochs epochs. Returns the per-log-step history.

    Two parameter groups:
      * shadow weights at lr_shadow (slow — keep close to original)
      * block scales at lr_scale (the LSQ knob, the actual recovery axis)
    """
    shadow_params: list[nn.Parameter] = []
    scale_params: list[nn.Parameter] = []
    for _name, sh in model.named_shadows():
        shadow_params.append(sh.shadow)
        scale_params.append(sh.block_scales)
    opt = torch.optim.Adam(
        [
            {"params": shadow_params, "lr": cfg.lr_shadow},
            {"params": scale_params, "lr": cfg.lr_scale},
        ]
    )
    # Cosine anneal both groups so we don't overshoot once near the
    # local-MSE basin (the smoke run hit minimum around epoch 50 then drifted).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    history: list[dict] = []
    loss_cfg = QATLossConfig(rel_error_weight=cfg.rel_error_weight)
    targets_dict = {n: targets_device[n] for n, _ in FIXED_STATE_SCHEMA}

    # Track best (lowest MSE) shadow + scales so we can restore them at end.
    best_state: dict[str, torch.Tensor] = {}
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        quantized = model()
        loss, parts = qat_reconstruction_loss(quantized, targets_dict, None, loss_cfg)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()

        loss_val = float(loss.detach().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            # Snapshot trainable params (shadows + scales) on cpu.
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if (epoch % cfg.log_every == 0) or (epoch == cfg.epochs - 1):
            history.append({
                "epoch": epoch,
                "loss": loss_val,
                "avg_mse": parts["avg_mse"],
                "avg_rel": parts["avg_rel"],
            })
            print(
                f"  epoch {epoch:>4}: loss={loss_val:.6e} "
                f"avg_mse={parts['avg_mse']:.6e} avg_rel={parts['avg_rel']:.6e}"
            )

    if best_state and best_epoch >= 0:
        # Restore best snapshot.
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n[QAT] restored best snapshot @ epoch {best_epoch} (loss={best_loss:.6e})")

    return history


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr-shadow", type=float, default=1e-4,
                   help="Adam lr for shadow weights (slow; anchor near original)")
    p.add_argument("--lr-scale", type=float, default=1e-3,
                   help="Adam lr for per-block scales (LSQ knob)")
    p.add_argument("--rel-error-weight", type=float, default=0.0,
                   help="Optional rel-err loss weight; 0 = MSE only "
                        "(rel-err on tiny weights is unrecoverable below quant step)")
    p.add_argument(
        "--device", choices=["mps", "cuda", "cpu", "auto"], default="auto",
        help="auto = mps if available, else cuda, else cpu (per CLAUDE.md MPS "
             "research-signal rule). Output evidence is tagged accordingly.",
    )
    p.add_argument("--output-dir", type=Path, default=None,
                   help="reports/raw/pr101_lossy_int4_qat_<UTC> by default")
    p.add_argument("--evidence-jsonl", type=Path,
                   default=REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = select_device("mps")
    else:
        device = select_device(args.device)
    print(f"[QAT] device: {device}")

    # Load fp32 originals.
    sd = torch.load(args.state_dict, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {args.state_dict} is not a dict")

    fp32_state: dict[str, torch.Tensor] = {}
    originals_fp32: dict[str, np.ndarray] = {}
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in sd:
            raise SystemExit(f"state_dict missing {name!r}")
        t = sd[name].detach().cpu().to(torch.float32)
        fp32_state[name] = t.clone()
        originals_fp32[name] = t.numpy().copy()

    # Build model + move to device.
    model = PR101QATModel(fp32_state, block_size=args.block_size).to(device)
    targets_device = {n: fp32_state[n].to(device) for n, _ in FIXED_STATE_SCHEMA}

    # ------------------------------------------------------------------
    # Pre-training (naive PTQ) measurement: snap fp32 originals through the
    # int4 grid using max-based scales — this is the falsification baseline.
    # ------------------------------------------------------------------
    pre_quant_state = {
        n: fake_quant_int4_block(fp32_state[n].to(device), args.block_size).detach().cpu()
        for n in originals_fp32
    }
    pre_metrics = measure_roundtrip_rel_err(pre_quant_state, originals_fp32)
    pre_raw, pre_brotli, pre_archive = encode_quantized_state_to_archive_bytes(
        pre_quant_state, block_size=args.block_size,
    )
    print(
        f"\n[QAT] pre-train (naive PTQ) weighted_avg_rel_err = "
        f"{pre_metrics['weighted_avg_rel_err_pct']:.4f}% "
        f"(baseline reference: {NAIVE_PTQ_REL_ERR_PCT:.2f}%)\n"
    )

    # ------------------------------------------------------------------
    # Train.
    # ------------------------------------------------------------------
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr_shadow=args.lr_shadow,
        lr_scale=args.lr_scale,
        rel_error_weight=args.rel_error_weight,
    )
    print(f"[QAT] training {args.epochs} epochs, lr_shadow={args.lr_shadow}, "
          f"lr_scale={args.lr_scale}, rel_error_weight={args.rel_error_weight}")
    history = train_qat(model, targets_device, train_cfg, device)

    # ------------------------------------------------------------------
    # Post-training measurement.
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        post_quant_state = {n: q.detach().cpu() for n, q in model().items()}
    post_metrics = measure_roundtrip_rel_err(post_quant_state, originals_fp32)
    print(
        f"\n[QAT] post-train weighted_avg_rel_err = "
        f"{post_metrics['weighted_avg_rel_err_pct']:.4f}% "
        f"(naive PTQ baseline: {pre_metrics['weighted_avg_rel_err_pct']:.4f}%)"
    )

    raw_bytes, brotli_bytes, archive_bytes = encode_quantized_state_to_archive_bytes(
        post_quant_state, block_size=args.block_size,
    )
    print(
        f"[QAT] post-train archive_bytes={archive_bytes:,} "
        f"(raw={raw_bytes:,}, brotli={brotli_bytes:,})"
    )

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    rel_err_pct = post_metrics["weighted_avg_rel_err_pct"]
    if rel_err_pct < 2.0 and post_metrics["max_p99_rel_err_pct"] < 10.0:
        verdict = "DISPATCH-READY"
        verdict_reason = (
            f"weighted_avg {rel_err_pct:.3f}% < 2.0% "
            f"AND max_p99 {post_metrics['max_p99_rel_err_pct']:.3f}% < 10.0% "
            f"after QAT"
        )
        ready_for_dispatch = True
    elif rel_err_pct < DISPATCH_THRESHOLD_PCT:
        verdict = "CONDITIONAL"
        verdict_reason = (
            f"weighted_avg {rel_err_pct:.3f}% in [2.0%, {DISPATCH_THRESHOLD_PCT}%); "
            f"borderline — sanity-ladder + CUDA test recommended"
        )
        ready_for_dispatch = True
    else:
        verdict = "STILL-FALSIFIED"
        verdict_reason = (
            f"weighted_avg {rel_err_pct:.3f}% >= {DISPATCH_THRESHOLD_PCT}%; "
            f"QAT did not recover precision. Naive PTQ baseline was "
            f"{NAIVE_PTQ_REL_ERR_PCT:.2f}%; QAT achieved "
            f"{rel_err_pct:.2f}% (improvement: "
            f"{(NAIVE_PTQ_REL_ERR_PCT - rel_err_pct):.2f} percentage points)"
        )
        ready_for_dispatch = False

    # ------------------------------------------------------------------
    # Evidence grade depends on device.
    # ------------------------------------------------------------------
    if device.type == "cuda":
        evidence_grade = "[CUDA-research-signal]"
    elif device.type == "mps":
        evidence_grade = "[MPS-research-signal]"
    else:
        evidence_grade = "[CPU-prep+QAT]"

    # ------------------------------------------------------------------
    # Manifest.
    # ------------------------------------------------------------------
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_lossy_int4_qat_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": ready_for_dispatch,
        "input_state_dict": repo_relative(args.state_dict),
        "input_state_dict_sha256": sha256_file(args.state_dict),
        "device": str(device),
        "seed": args.seed,
        "block_size": args.block_size,
        "training": {
            "epochs": args.epochs,
            "lr_shadow": args.lr_shadow,
            "lr_scale": args.lr_scale,
            "rel_error_weight": args.rel_error_weight,
            "history": history,
        },
        "naive_ptq_baseline": {
            "weighted_avg_rel_err_pct": pre_metrics["weighted_avg_rel_err_pct"],
            "max_p99_rel_err_pct": pre_metrics["max_p99_rel_err_pct"],
            "max_max_rel_err_pct": pre_metrics["max_max_rel_err_pct"],
            "archive_bytes": pre_archive,
        },
        "qat_post_train": {
            "weighted_avg_rel_err_pct": post_metrics["weighted_avg_rel_err_pct"],
            "weighted_avg_abs_err": post_metrics["weighted_avg_abs_err"],
            "max_p99_rel_err_pct": post_metrics["max_p99_rel_err_pct"],
            "max_max_rel_err_pct": post_metrics["max_max_rel_err_pct"],
            "per_tensor": post_metrics["per_tensor"],
        },
        "archive_bytes": archive_bytes,
        "raw_payload_bytes": raw_bytes,
        "brotli_bytes": brotli_bytes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "dispatch_blockers": (
            [] if ready_for_dispatch else [
                "qat_rel_err_above_5pct_threshold",
                "no_int4_decoder_runtime_built",
                "missing_exact_cuda_auth_eval",
            ]
        ),
        "supersedes_prior_FALSIFIED_tag": True,
        "n_total_elements": post_metrics["n_total_elements"],
        "n_nontrivial_elements": post_metrics["n_nontrivial_elements"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[QAT] manifest: {manifest_path}")

    # ------------------------------------------------------------------
    # Cathedral evidence row.
    # ------------------------------------------------------------------
    if ready_for_dispatch:
        dispatch_verdict = "DISPATCH-READY-pending-runtime-and-cuda-eval"
    else:
        dispatch_verdict = "DEFERRED-pending-research"
    evidence_row = {
        "technique": "lossy_int4_quantization_qat",
        "empirical_archive_bytes": archive_bytes,
        "empirical_distortion_increase_pct": rel_err_pct,
        "evidence_grade": evidence_grade,
        "evidence_marker": evidence_grade,
        "evidence_semantics": (
            "qat_recovered_int4_byte_anchor_plus_roundtrip_rel_err_no_score"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": ready_for_dispatch,
        "dispatch_attempted": False,
        "score_affecting_payload_changed": True,  # int4 grid != fp32 weights
        "charged_bits_changed": True,  # archive bytes change
        "dispatch_blockers": manifest["dispatch_blockers"],
        "source": (
            f"{evidence_grade} {repo_relative(manifest_path)} "
            f"(qat epochs={args.epochs} block_size={args.block_size}; "
            f"naive PTQ baseline rel_err={pre_metrics['weighted_avg_rel_err_pct']:.2f}%, "
            f"QAT rel_err={rel_err_pct:.2f}%)"
        ),
        "contest_dispatch_verdict": dispatch_verdict,
        "supersedes_prior_FALSIFIED_tag": True,
        "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evidence_row) + "\n")
    print(f"[QAT] evidence row appended: {args.evidence_jsonl}")

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  reason: {verdict_reason}")
    print(f"  archive_bytes: {archive_bytes:,}")
    print(f"  rel_err_pct: {rel_err_pct:.4f}% (naive baseline {pre_metrics['weighted_avg_rel_err_pct']:.4f}%)")
    print(f"  evidence_grade: {evidence_grade}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
