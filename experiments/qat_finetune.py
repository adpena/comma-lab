#!/usr/bin/env python3
"""Quantization-Aware Training (QAT) fine-tuning for FP4 deployment.

Implements progressive quantization following Bit-by-Bit (ICLR 2026)
and Compute-Optimal QAT (Apple, Sep 2025) findings:

  Phase A: INT8 warm-up  — 50 epochs, lets weights migrate toward
           quantization-friendly regions at coarser granularity first.
  Phase B: FP4 fine-tune — 250 epochs with our FP4 codebook, cosine
           decay from 0.1x base LR to zero.

Research-backed design choices:
  - Progressive bit-reduction beats direct 4-bit QAT (Bit-by-Bit, ICLR 2026)
  - QAT fraction 30% of total training for <200K-param models (Apple, Sep 2025)
  - Per-block FP4 with block_size=32 (our codebook matches BOF4 structure)
  - STE gradient estimator (PEGE offers 1-2% but adds complexity)
  - ALL layers quantized to FP4 (Conv2d, ConvTranspose2d, Embedding, Linear)
    to match export_asymmetric_checkpoint_fp4 exactly — train matches deploy.
  - eval_roundtrip=True + noise_std=0.5 in loss (mandatory per our findings)
  - Scorer loss matches training: hinge SegNet + PoseNet MSE

Usage (after overnight float training completes):
    PYTHONPATH=src:upstream:$PWD python experiments/qat_finetune.py \\
        --checkpoint experiments/results/overnight_small_renderer/distill_best.pt \\
        --upstream upstream/ \\
        --device cuda \\
        --output-dir experiments/results/qat_fp4

References:
    [1] Compute-Optimal QAT, Apple, Sep 2025, arXiv:2509.22935
    [2] Bit-by-Bit Progressive QAT, ICLR 2026, arXiv:2604.07888
    [3] BOF4: Block-Wise Optimal Float, May 2025, arXiv:2505.06653
    [4] NeuroQuant: Variable-Rate Neural Video, ICLR 2025, arXiv:2502.11729
"""
from __future__ import annotations

# DX-fix 2026-04-25: line-buffer stdout/stderr so progress logs flush
# immediately when piped to log files (Python buffers ~8KB by default,
# making long-running scripts appear silent for hours per the optimize_poses
# incident on the A100 today).
import sys as _dx_sys
try:
    _dx_sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _dx_sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass


import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

_UPSTREAM_CANDIDATES = [
    Path(os.environ.get("TAC_UPSTREAM_DIR", "")),
    Path(os.environ.get("UPSTREAM_ROOT", "")),
    _ROOT / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _UPSTREAM_CANDIDATES:
    if _p and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        sys.path.insert(0, str(_p))
        break


@dataclass
class QATConfig:
    """All hyperparameters for QAT fine-tuning."""

    # Architecture (must match the float checkpoint — use ArchConfig defaults)
    embed_dim: int = 6
    base_ch: int = 36
    mid_ch: int = 60
    motion_hidden: int = 32
    depth: int = 1
    pose_dim: int = 6
    max_flow_px: float = 20.0
    max_residual: float = 20.0
    use_dsconv: bool = False
    # R38 fix: distill_distill.py overrides ArchConfig padding_mode to
    # "replicate" (boundary-artifact avoidance per Yousfi). QAT default
    # was "zeros" creating silent arch divergence with checkpoints trained
    # by train_distill.py. Match the canonical training default.
    padding_mode: str = "replicate"
    use_dilation: bool = False
    use_zoom_flow: bool = False

    # Paths
    checkpoint_path: str = ""
    upstream_dir: str = "upstream/"
    mixed_precision_json: str = ""  # scorer sensitivity JSON for mixed-precision QAT
    # Lane F-V4: per-layer FP4 sensitivity profile (from
    # experiments/profile_fp4_layer_sensitivity.py).
    mixed_precision_from_sensitivity: str = ""
    mixed_precision_target_rate: float = 0.70  # fraction of params at bulk_bits (FP4)
    mixed_precision_bulk_bits: int = 4
    mixed_precision_critical_bits: int = 16
    # Lane SG (Codex F2 2026-04-28): which scorer-prior protection list
    # forces layers to ``mixed_precision_critical_bits`` regardless of
    # what the FP4-sensitivity profile said. Two valid values:
    # "posenet_prior" (legacy, FiLM/motion/head protected) and
    # "segnet_prior" (Lane SG, decoder/class-affine protected). Threaded
    # from --protected-pattern-set into the bit allocator below so the
    # operator's stated hypothesis is actually applied (the prior wiring
    # parsed the flag but never threaded it, making Lane SG runs byte-
    # identical to legacy QAT).
    protected_pattern_set: str = "posenet_prior"
    output_dir: str = "experiments/results/qat_fp4"

    # QAT schedule — research-backed (Apple 2025, Bit-by-Bit 2026)
    int8_warmup_epochs: int = 50       # Phase A: INT8 warm-up
    fp4_epochs: int = 250              # Phase B: FP4 fine-tune
    base_lr: float = 3e-5             # 0.1x of phase2 LR in float training
    lr_min_ratio: float = 0.0         # cosine decay to zero (Apple 2025)
    warmup_steps: int = 20            # re-warmup at QAT start (Apple 2025)
    batch_size: int = 4               # scorer-limited
    grad_clip: float = 1.0

    # FP4 quantization
    fp4_block_size: int = 32          # weights per scale group

    # Scorer loss (must match float training)
    seg_weight: float = 100.0
    pose_weight: float = 10.0
    segnet_loss_mode: str = "hinge"
    hinge_margin: float = 0.5
    eval_roundtrip: bool = True
    noise_std: float = 0.5

    # Monitoring
    eval_every: int = 25
    log_every: int = 5
    checkpoint_every: int = 50
    device: str = "cuda"
    seed: int = 42

    # Council D 2026-04-29 PM: EMA on QAT (Quantizr's #1 + Selfcomp's #2
    # full pipelines run EMA through anchor → finetune → joint → QAT →
    # final). Mandatory per CLAUDE.md "EMA — NON-NEGOTIABLE". The EMA
    # shadow set INCLUDES LSQ step-size scales / FP4 fake-quant scales
    # (van den Oord + Ballé council endorsement: codec scales benefit
    # MORE than weights from EMA because their gradient is naturally
    # noisier through the round operator's surrogate gradient).
    ema_decay: float = 0.997


def create_model(cfg: QATConfig, device: torch.device) -> nn.Module:
    """Create renderer matching the float checkpoint.

    Routes through build_renderer — the SAME function train_renderer uses —
    so PairGenerator vs AsymmetricPairGenerator dispatch is identical on
    both sides. PairGenerator (use_zoom_flow=False) wraps MaskRenderer +
    MotionPredictor with motion.head emitting 2 channels of flow; the
    AsymmetricPairGenerator path (use_zoom_flow=True) emits more channels
    for zoom + flow + residual gating. Hardcoding either constructor here
    was the LANE-D QAT crash root cause (mirrors pipeline.step_export
    third-layer fix from c5214993).
    """
    from tac.renderer import build_renderer

    model = build_renderer(
        num_classes=5,
        embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch,
        mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden,
        depth=cfg.depth,
        pose_dim=cfg.pose_dim,
        use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode,
        use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
        # Bug 3 fix (2026-04-27 audit): forward QATConfig fields previously
        # advertised but silently dropped. The float checkpoint built by
        # train_distill.py uses cfg.max_flow_px / cfg.max_residual; not
        # forwarding them here means a non-default value in train_distill
        # would silently produce a QAT model with the wrong scale.
        max_flow_px=cfg.max_flow_px,
        max_residual=cfg.max_residual,
    )
    return model.to(device)


def load_float_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load the float checkpoint from distillation training or ASYM .bin export.

    Returns the model that should be used for QAT (may differ from the input
    `model` for ASYM/FP4A paths — those formats know their own arch and we
    must use the LOADED model's arch, not the one constructed from CLI args).

    Council R-arch-drift fix (2026-04-27): ASYM/FP4A binaries embed arch in
    their header. The previous code constructed `model` from CLI defaults
    (e.g. motion.head=[4]) and then tried to load a state_dict with a
    different arch (e.g. motion.head=[6]) → catastrophic shape-mismatch.
    Now we trust the binary's arch and return the loaded model directly.
    The .pt path keeps the construct-then-load behavior since .pt has no
    arch header.
    """
    from pathlib import Path
    raw = Path(path).read_bytes()

    # Detect format by magic bytes
    if raw[:4] == b"ASYM":
        from tac.renderer_export import load_asymmetric_checkpoint
        loaded = load_asymmetric_checkpoint(raw, device=str(device))
        print(f"  Loaded ASYM checkpoint from {path} (using loaded arch, not CLI defaults)")
        return loaded
    elif raw[:4] == b"FP4A":
        from tac.renderer_export import load_asymmetric_checkpoint_fp4
        loaded = load_asymmetric_checkpoint_fp4(raw, device=str(device))
        print(f"  Loaded FP4A checkpoint from {path} (using loaded arch, not CLI defaults)")
        return loaded
    else:
        # PyTorch .pt format (from training)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # Fourth-layer arch-drift fix (mirrors pipeline.step_export from
        # c5214993): train_renderer can save with torch.nn.utils.parametrize
        # hooks attached for self-compression / fake-quantization. Use the
        # canonical strip helper (factored 2026-04-28 from this very block,
        # then hardened for root-level keys + multi-original weight_norm +
        # nested chains via Round 11 codex review).
        from tac.parametrize_strip import strip_parametrize_hooks, has_parametrize_keys
        if has_parametrize_keys(state):
            n_before = len(state)
            state = strip_parametrize_hooks(state)
            print(f"  Stripped parametrize hooks: {n_before} → {len(state)} keys")

        # strict=False so we surface missing/unexpected lists; raise with a
        # helpful error if either is non-empty (equivalent to strict=True
        # but with the actual mismatch printed instead of a wall of keys).
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"QAT checkpoint shape mismatch — refuse to fine-tune wrong arch.\n"
                f"  missing={list(missing)[:5]}\n"
                f"  unexpected={list(unexpected)[:5]}\n"
                f"  Path: {path}\n"
                f"  cfg: use_zoom_flow={getattr(model, '_use_zoom_flow', '?')}, "
                f"base_ch / mid_ch / motion_hidden / use_dsconv must match training.\n"
                f"  Hint: PairGenerator (use_zoom_flow=False) has motion.head.bias [2]; "
                f"AsymmetricPairGenerator (use_zoom_flow=True) has [6]."
            )
        print(f"  Loaded .pt checkpoint from {path}")
    return model


def apply_int8_fake_quant(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Wrap Conv2d weights with INT8 FakeQuant STE via nn.utils.parametrize.

    Returns list of (module, param_name) for later removal.
    """
    from tac.quantization import FakeQuantSTE

    class Int8Parametrize(nn.Module):
        def forward(self, weight: torch.Tensor) -> torch.Tensor:
            return FakeQuantSTE.apply(weight.contiguous())

    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.utils.parametrize.register_parametrization(
                module, "weight", Int8Parametrize()
            )
            wrapped.append((module, "weight"))
    print(f"  INT8 parametrized: {len(wrapped)} layers")
    return wrapped


def remove_parametrizations(wrapped: list[tuple[nn.Module, str]]) -> None:
    """Remove all registered parametrizations."""
    for module, param_name in wrapped:
        if nn.utils.parametrize.is_parametrized(module, param_name):
            nn.utils.parametrize.remove_parametrizations(module, param_name)


def apply_fp4_fake_quant(
    model: nn.Module,
    block_size: int = 32,
) -> list[tuple[nn.Module, str]]:
    """Wrap ALL quantizable layers with FP4 FakeQuant STE (Phase B).

    Wraps Conv2d, ConvTranspose2d, Embedding, and Linear (including FiLM).
    All layers are quantized because export_asymmetric_checkpoint_fp4 quantizes
    ALL layers — training must match deployment exactly.

    Args:
        model: the renderer to wrap
        block_size: FP4 block size (32 = default, good for small models)

    Returns:
        list of (module, param_name) for cleanup
    """
    from tac.fp4_quantize import FP4Parametrize, DEFAULT_CODEBOOK

    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding, nn.Linear)):
            if hasattr(module, "weight") and module.weight.ndim >= 2:
                nn.utils.parametrize.register_parametrization(
                    module,
                    "weight",
                    FP4Parametrize(DEFAULT_CODEBOOK.clone(), block_size),
                )
                wrapped.append((module, "weight"))

    print(f"  FP4 parametrized: {len(wrapped)} layers (all Conv/Linear/Embedding)")
    return wrapped


def bit_allocation_from_sensitivity(
    sensitivity_path: str | Path,
    model: nn.Module,
    target_rate: float = 0.70,
    bulk_bits: int = 4,
    critical_bits: int = 16,
    protected_pattern_set: str = "posenet_prior",
) -> tuple[dict[str, int], dict[str, float]]:
    """Build a per-layer bit allocation from a Lane F-V4 sensitivity profile.

    Lane F-V4 design: sort layers by FP4 sensitivity (delta distortion when
    only that layer is FP4-quantized) ASCENDING. The bottom ``target_rate``
    fraction by PARAM COUNT (default 70%) gets ``bulk_bits`` (default FP4=4).
    The remaining ``1 - target_rate`` (top 30% by sensitivity) gets
    ``critical_bits`` (default FP16=16, i.e. NO quantization).

    The Lagrangian-style ``--mixed-precision-target-rate`` knob lets the
    council sweep the rate-distortion tradeoff: 1.0 → uniform FP4 (matches
    Lane F-V3), 0.0 → no FP4 (matches FP32 baseline), intermediate values
    let mixed-precision land in between.

    Args:
        sensitivity_path: path to ``layer_sensitivity.pt`` from
            ``profile_fp4_layer_sensitivity.py``.
        model: the renderer (used to filter sensitivity entries to layers
            that actually exist in this model — guards against arch-drift
            between profile time and QAT time).
        target_rate: fraction of params (NOT layers) that should be FP4.
            Default 0.70 = 70% of weight bytes are FP4-quantized, the
            remaining 30% (the most sensitive layers) stay FP16.
        bulk_bits: bit depth for the FP4-tolerable layers (default 4).
        critical_bits: bit depth for the FP4-intolerant layers (default 16,
            i.e. unchanged FP16 weights).

    Returns:
        (allocation, sensitivity_used) where allocation maps qualified
        param names ("foo.bar.weight") to int bit depth, and
        sensitivity_used maps the same names to the float delta from
        the profile.

    Council justification (Yousfi + Fridrich + Hotz, 2026-04-28):
      Uniform FP4 on dilated-h64 ASYM destroyed PoseNet because a small
      number of layers carry the YUV6 statistics that FastViT-T12
      attention is sensitive to. The empirical profile identifies them;
      the bit allocator preserves them.
    """
    sensitivity = torch.load(str(sensitivity_path), map_location="cpu", weights_only=False)
    if "delta" not in sensitivity:
        raise ValueError(
            f"sensitivity file {sensitivity_path} missing 'delta' key. "
            f"Re-run experiments/profile_fp4_layer_sensitivity.py."
        )
    delta: dict[str, float] = sensitivity["delta"]
    profile_param_count: dict[str, int] = sensitivity.get("param_count", {})

    # Map "foo.bar" (module name in profile) to "foo.bar.weight" (param name in QAT)
    model_params = dict(model.named_parameters())
    layer_records: list[tuple[str, float, int]] = []  # (param_name, delta, n_params)
    for module_name, d in delta.items():
        param_name = module_name + ".weight"
        if param_name not in model_params:
            # Layer disappeared from the model (arch drift or rename) — skip
            continue
        n = profile_param_count.get(module_name, model_params[param_name].numel())
        layer_records.append((param_name, float(d), int(n)))

    if not layer_records:
        raise ValueError(
            f"No sensitivity entries match this model's parameters. "
            f"Profile checkpoint may have been built from a different "
            f"architecture. Sensitivity has {len(delta)} entries; model "
            f"has {len(model_params)} parameters."
        )

    # F2 fix (Codex 2026-04-28): resolve the scorer-prior protection list
    # BEFORE allocation so we can FORCE protected layers to critical_bits
    # regardless of what sensitivity profile claimed. This is what the
    # operator's --protected-pattern-set flag is supposed to do — without
    # this guard, segnet_prior would have been byte-identical to the
    # posenet_prior default whenever the sensitivity profile happened to
    # rank PoseNet-prior layers as low-sensitivity.
    from tac.self_compress import get_protected_patterns
    protection_list = get_protected_patterns(protected_pattern_set)
    def _is_protected_layer(param_name: str) -> bool:
        # param_name is "module.foo.bar.weight"; strip ".weight" suffix
        # and match by full-name OR suffix against each protection pattern.
        module_name = param_name[:-len(".weight")] if param_name.endswith(".weight") else param_name
        for pat in protection_list:
            if module_name == pat or module_name.endswith("." + pat):
                return True
        return False

    # Sort ASCENDING by sensitivity (FP4-tolerable first)
    layer_records.sort(key=lambda r: r[1])

    total_params = sum(r[2] for r in layer_records)
    fp4_budget = int(round(total_params * target_rate))

    allocation: dict[str, int] = {}
    sensitivity_used: dict[str, float] = {}
    forced_protected: list[str] = []  # F2: surface for diagnostics + tests
    cumulative = 0
    for param_name, d, n in layer_records:
        sensitivity_used[param_name] = d
        # F2: protected layers ALWAYS get critical_bits (typically FP16),
        # bypassing the sensitivity-based budget. They still consume budget
        # accounting downstream so subsequent layers can fill the remaining
        # FP4 quota — this matches the operator's mental model: "protect
        # these layers, then FP4-quantize the rest up to the budget."
        if _is_protected_layer(param_name):
            allocation[param_name] = critical_bits
            forced_protected.append(param_name)
            continue
        if cumulative + n <= fp4_budget:
            allocation[param_name] = bulk_bits
            cumulative += n
        else:
            allocation[param_name] = critical_bits

    actual_rate = cumulative / max(total_params, 1)
    print(f"  Mixed-precision allocation built from {sensitivity_path}:")
    print(f"    protected_pattern_set: {protected_pattern_set!r} "
          f"({len(protection_list)} patterns; {len(forced_protected)} layers "
          f"forced to FP{critical_bits})")
    print(f"    total layers: {len(layer_records)}, total params: {total_params:,}")
    print(f"    target_rate: {target_rate:.2%}, actual: {actual_rate:.2%} "
          f"({cumulative:,} FP{bulk_bits} / {total_params - cumulative:,} FP{critical_bits})")
    print(f"    layers FP{bulk_bits}: "
          f"{sum(1 for v in allocation.values() if v == bulk_bits)}, "
          f"layers FP{critical_bits}: "
          f"{sum(1 for v in allocation.values() if v == critical_bits)}")
    if forced_protected:
        print(f"    forced-protected ({protected_pattern_set}): {forced_protected[:10]}"
              f"{'…' if len(forced_protected) > 10 else ''}")
    return allocation, sensitivity_used


def apply_mixed_precision_quant(
    model: nn.Module,
    bit_allocation: dict[str, int],
    block_size: int = 32,
) -> list[tuple[nn.Module, str]]:
    """Wrap layers with VARIABLE bit-depth FakeQuant (scorer-optimal allocation).

    Uses Yousfi's scorer-Jacobian sensitivity to allocate bits per layer.
    Layers with negative scorer sensitivity get 2 bits (quantization noise
    acts as beneficial steganalytic texture). High-sensitivity layers get 8 bits.

    Args:
        model: the renderer to wrap
        bit_allocation: dict mapping parameter name → bit depth (from sensitivity sweep)
        block_size: quantization block size

    Returns:
        list of (module, param_name) for cleanup
    """
    from tac.fp4_quantize import FP4Parametrize, DEFAULT_CODEBOOK

    class VariableBitParametrize(nn.Module):
        """Parametrize that quantizes to a specific bit-depth via STE."""
        def __init__(self, bits: int, codebook: torch.Tensor, blk_size: int):
            super().__init__()
            self.bits = bits
            self.register_buffer("codebook", codebook)
            self.blk_size = blk_size

        def forward(self, weight: torch.Tensor) -> torch.Tensor:
            w = weight.contiguous()
            if self.bits >= 16:
                return w
            if self.bits == 4:
                # FP4_HARDWARE_DISCLOSED: NVFP4 hardware needs Blackwell
                # (CC >= 10.0). RTX 4090 is CC 8.9 → fake_quant_fp4 SIMULATES
                # FP4 in FP32. The renderer's inflate-time inference still
                # runs at FP32 — only the archive bytes are 4-bit packed.
                # See Lane F-V5 (hardware FP8 via torchao) for the proper
                # rescue path on Ada/Lovelace+ hardware. Reference:
                # project_cosmos_deep_dive_addendum_20260428.
                from tac.fp4_quantize import fake_quant_fp4
                return fake_quant_fp4(w, self.codebook, self.blk_size)
            # For other bit-depths: uniform symmetric quantization with STE
            n_levels = 2 ** self.bits
            if w.ndim >= 2:
                flat = w.detach().reshape(w.shape[0], -1)
                scale = flat.abs().amax(dim=1) / (n_levels / 2 - 1)
                scale = scale.clamp(min=1e-10)
                scale_view = scale.reshape(-1, *([1] * (w.ndim - 1)))
                q = (w / scale_view).round().clamp(-(n_levels // 2), n_levels // 2 - 1)
                return (q * scale_view - w).detach() + w  # STE
            else:
                scale = w.detach().abs().max() / (n_levels / 2 - 1)
                if scale.item() < 1e-10:
                    return w
                q = (w / scale).round().clamp(-(n_levels // 2), n_levels // 2 - 1)
                return (q * scale - w).detach() + w  # STE

    wrapped = []
    bits_summary = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding, nn.Linear)):
            if hasattr(module, "weight") and module.weight.ndim >= 2:
                # Find the matching allocation key (exact match, not substring)
                param_name = name + ".weight"
                if param_name not in dict(model.named_parameters()):
                    param_name = None
                bits = bit_allocation.get(param_name, 4)  # default to FP4
                bits_summary[param_name or name] = bits

                nn.utils.parametrize.register_parametrization(
                    module, "weight",
                    VariableBitParametrize(bits, DEFAULT_CODEBOOK.clone(), block_size),
                )
                wrapped.append((module, "weight"))

    # Summary
    total_bits = sum(
        dict(model.named_parameters())[n].numel() * b
        for n, b in bits_summary.items()
        if n in dict(model.named_parameters())
    )
    n_2bit = sum(1 for b in bits_summary.values() if b == 2)
    n_4bit = sum(1 for b in bits_summary.values() if b == 4)
    n_8bit = sum(1 for b in bits_summary.values() if b == 8)
    print(f"  Mixed-precision: {len(wrapped)} layers "
          f"(2b={n_2bit}, 4b={n_4bit}, 8b={n_8bit}, "
          f"total={total_bits / 8 / 1024:.1f}KB)")
    return wrapped


def compute_scorer_loss(
    model: nn.Module,
    masks_even: torch.Tensor,
    masks_odd: torch.Tensor,
    gt_frames_even: torch.Tensor,
    gt_frames_odd: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: nn.Module,
    segnet: nn.Module,
    cfg: QATConfig,
    device: torch.device,
    ego_flow: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the contest-faithful scorer loss.

    Matches train_distill.py Phase 2 loss exactly:
      loss = seg_weight * hinge_loss + pose_weight * pose_mse

    With eval_roundtrip + noise_std for contest fidelity.
    """
    from tac.renderer import simulate_eval_roundtrip
    from tac.camera import CAMERA_H, CAMERA_W

    B = masks_even.shape[0]

    # Forward pass through renderer
    pair_kwargs = {}
    if poses is not None and hasattr(model, "pose_dim") and model.pose_dim > 0:
        pair_kwargs["pose"] = poses
    if ego_flow is not None:
        pair_kwargs["ego_flow"] = ego_flow
    # Handle QAT wrapper
    renderer = model.base if hasattr(model, "base") else model
    pairs = renderer(masks_even, masks_odd, **pair_kwargs)  # (B, 2, H, W, 3)

    pred_even = pairs[:, 0]  # (B, H, W, 3)
    pred_odd = pairs[:, 1]

    # Flatten to (2B, H, W, 3) for scorer
    pred_all = torch.cat([pred_even, pred_odd], dim=0)
    gt_all = torch.cat([gt_frames_even, gt_frames_odd], dim=0)

    # eval_roundtrip: simulate contest resize chain
    # Noise only on PRED (not GT) — matches train_distill.py (I1 fix)
    if cfg.eval_roundtrip:
        pred_chw = pred_all.permute(0, 3, 1, 2)
        pred_chw = simulate_eval_roundtrip(
            pred_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=cfg.noise_std,
        )
        pred_for_loss = pred_chw.permute(0, 2, 3, 1)

        with torch.no_grad():
            gt_chw = gt_all.permute(0, 3, 1, 2)
            gt_chw = simulate_eval_roundtrip(
                gt_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0,
            )
            gt_for_loss = gt_chw.permute(0, 2, 3, 1)
    else:
        pred_for_loss = pred_all
        gt_for_loss = gt_all

    # SegNet loss (hinge) — use pre-extracted integer masks as targets (C2 fix)
    # Running SegNet on GT frames is wasteful and can produce different targets
    # than the pre-extracted masks due to preprocessing differences.
    # .contiguous() required for MPS: eval_roundtrip + permute can produce
    # non-contiguous tensors that fail upstream's x.view(-1, 1) in AllNorm
    pred_seg_chw = pred_for_loss.permute(0, 3, 1, 2).contiguous()
    pred_seg_in = segnet.preprocess_input(pred_seg_chw.unsqueeze(1).contiguous())
    pred_seg_logits = segnet(pred_seg_in)

    # Use the pre-extracted masks directly as GT argmax targets
    gt_seg_argmax = torch.cat([masks_even, masks_odd], dim=0)  # (2B, H, W)
    # Resize to match logit spatial dims
    logit_h, logit_w = pred_seg_logits.shape[2], pred_seg_logits.shape[3]
    if gt_seg_argmax.shape[1] != logit_h or gt_seg_argmax.shape[2] != logit_w:
        gt_seg_argmax = F.interpolate(
            gt_seg_argmax.float().unsqueeze(1), size=(logit_h, logit_w), mode="nearest",
        ).squeeze(1).long()

    if cfg.segnet_loss_mode == "hinge":
        correct = pred_seg_logits.gather(1, gt_seg_argmax.unsqueeze(1)).squeeze(1)
        mask_inf = torch.zeros_like(pred_seg_logits)
        mask_inf.scatter_(1, gt_seg_argmax.unsqueeze(1), float("-inf"))
        runner_up = (pred_seg_logits + mask_inf).max(dim=1).values
        seg_loss = F.relu(cfg.hinge_margin - (correct - runner_up)).mean()
    else:
        seg_loss = F.cross_entropy(pred_seg_logits, gt_seg_argmax)

    # PoseNet loss (MSE on 6D pose) — GT path with no_grad (C3 fix)
    # .contiguous() for MPS: upstream AllNorm uses x.view(-1, 1) which fails
    # on non-contiguous tensors from permute operations
    pred_pose_chw = pred_for_loss.permute(0, 3, 1, 2).contiguous()
    pred_pose_pairs = torch.stack([pred_pose_chw[:B], pred_pose_chw[B:]], dim=1).contiguous()
    pred_pose_in = posenet.preprocess_input(pred_pose_pairs)
    pred_pose_out = posenet(pred_pose_in)["pose"][..., :6]

    with torch.no_grad():
        gt_pose_chw = gt_for_loss.permute(0, 3, 1, 2).contiguous()
        gt_pose_pairs = torch.stack([gt_pose_chw[:B], gt_pose_chw[B:]], dim=1).contiguous()
        gt_pose_in = posenet.preprocess_input(gt_pose_pairs)
        gt_pose_out = posenet(gt_pose_in)["pose"][..., :6]

    pose_loss = (pred_pose_out - gt_pose_out).pow(2).mean()

    # Combined loss
    total = cfg.seg_weight * seg_loss + cfg.pose_weight * pose_loss

    metrics = {
        "seg_loss": seg_loss.item(),
        "pose_loss": pose_loss.item(),
        "total_loss": total.item(),
    }
    return total, metrics


def cosine_lr(step: int, total_steps: int, base_lr: float, min_ratio: float, warmup: int) -> float:
    """Cosine LR with linear warmup (Apple 2025 QAT schedule)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate_fp4_quality(
    model: nn.Module,
    masks: torch.Tensor,
    gt_frames: list,
    poses: torch.Tensor | None,
    device: torch.device,
    n_pairs: int = 20,
    distortion_net: nn.Module | None = None,
    zoom_warp: nn.Module | None = None,
) -> dict[str, float]:
    """Quick quality check: generate frames and compute distortion via upstream scorer.

    Args:
        distortion_net: pre-loaded DistortionNet to reuse (avoids reloading from
            disk on every eval call). If None, loads from upstream models.
    """
    owns_dn = distortion_net is None
    if owns_dn:
        upstream_root = UPSTREAM_ROOT or Path("upstream")
        if not (upstream_root / "modules.py").exists():
            raise RuntimeError(
                f"Cannot find upstream at {upstream_root}. "
                f"Set TAC_UPSTREAM_DIR or pass --upstream."
            )
        upstream_str = str(upstream_root)
        if upstream_str not in sys.path:
            sys.path.insert(0, upstream_str)
        from modules import DistortionNet

        distortion_net = DistortionNet().eval().to(device)
        distortion_net.load_state_dicts(
            upstream_root / "models" / "posenet.safetensors",
            upstream_root / "models" / "segnet.safetensors",
            device,
    )

    renderer = model.base if hasattr(model, "base") else model
    renderer.eval()

    pd_list, sd_list = [], []
    with torch.inference_mode():
        for i in range(min(n_pairs, len(gt_frames) // 2)):
            m_t = masks[2 * i : 2 * i + 1].to(device=device, dtype=torch.int64)
            m_t1 = masks[2 * i + 1 : 2 * i + 2].to(device=device, dtype=torch.int64)
            p = poses[i : i + 1].to(device) if poses is not None else None
            kwargs = {"pose": p} if p is not None else {}
            if zoom_warp is not None:
                kwargs["ego_flow"] = zoom_warp(torch.tensor([i], device=device), m_t.shape[1], m_t.shape[2])
            pair = renderer(m_t, m_t1, **kwargs)
            chw = pair[0].permute(0, 3, 1, 2).float()
            # eval_roundtrip: upscale to camera res + uint8 quantize
            # (DistortionNet internally resizes to scorer res — matches upstream)
            cam = F.interpolate(chw, size=(874, 1164), mode="bilinear", align_corners=False)
            cam = cam.round().clamp(0, 255).to(torch.uint8).float()

            gt_p = torch.stack([
                torch.from_numpy(gt_frames[2 * i]).float(),
                torch.from_numpy(gt_frames[2 * i + 1]).float(),
            ]).unsqueeze(0).to(device)
            comp_p = cam.permute(0, 2, 3, 1).unsqueeze(0).contiguous()
            pd, sd = distortion_net.compute_distortion(gt_p, comp_p)  # upstream convention: (gt, compressed)
            pd_list.append(pd.item())
            sd_list.append(sd.item())

    renderer.train()
    if owns_dn:
        del distortion_net
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_pose = sum(pd_list) / max(len(pd_list), 1)
    avg_seg = sum(sd_list) / max(len(sd_list), 1)
    distortion = 100 * avg_seg + math.sqrt(10 * avg_pose)

    return {"pose_d": avg_pose, "seg_d": avg_seg, "distortion": distortion}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QAT fine-tuning: float checkpoint → FP4-robust model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to float checkpoint (.pt)")
    parser.add_argument("--upstream", default="upstream/")
    parser.add_argument("--output-dir", default="experiments/results/qat_fp4")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])

    # Architecture (must match float training — use ArchConfig defaults)
    parser.add_argument("--base-ch", type=int, default=36)
    parser.add_argument("--mid-ch", type=int, default=60)
    parser.add_argument("--pose-dim", type=int, default=6)
    parser.add_argument("--use-dsconv", action="store_true")
    parser.add_argument("--padding-mode", type=str, default="replicate",
                        choices=["zeros", "reflect", "replicate", "circular"])
    parser.add_argument("--use-dilation", action="store_true")
    parser.add_argument("--motion-hidden", type=int, default=32,
                        help="MotionPredictor hidden channels (must match training)")
    parser.add_argument("--depth", type=int, default=1,
                        help="Renderer depth (must match training, 1 or 2)")
    parser.add_argument("--embed-dim", type=int, default=6,
                        help="Embedding dim (must match training)")
    parser.add_argument("--use-zoom-flow", action="store_true",
                        help="Enable RadialZoomWarp (4ch MotionPredictor)")
    # Lane SG (2026-04-28 council EUREKA #5): re-scope which layers stay FP32
    # in QAT based on which scorer we are protecting. SegNet contributes
    # 100×seg vs sqrt(10·pose) — SegNet is 2-5× more impactful per unit
    # distortion at our operating point. Lane SG protects SegNet-relevant
    # decoder layers instead of PoseNet-relevant FiLM/motion layers.
    parser.add_argument("--protected-pattern-set", type=str,
                        default="posenet_prior",
                        choices=["posenet_prior", "segnet_prior"],
                        help="Lane SG: which scorer-prior protection list "
                             "to use during QAT. 'posenet_prior' = legacy "
                             "default; 'segnet_prior' = Lane SG.")

    # QAT schedule
    parser.add_argument("--int8-warmup-epochs", type=int, default=50)
    parser.add_argument("--fp4-epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--poses", type=str, default=None,
                        help="Path to optimized_poses.pt to thread to QAT loop. "
                             "REQUIRED for renderers with pose_dim>0. Auto-discovery "
                             "of upstream/gt_poses.pt is intentionally NOT done — "
                             "explicit > implicit per CLAUDE.md FORBIDDEN PATTERNS. "
                             "See findings.md 'Lane F regression' (2026-04-27): "
                             "the silent fallback to zero poses caused a +58% PoseNet "
                             "regression on Lane F because the renderer was QAT-trained "
                             "against zero poses then deployed against real poses (per "
                             "memory project_baseline_poses_load_bearing).")

    # Control
    parser.add_argument("--skip-int8-warmup", action="store_true",
                        help="Skip INT8 warm-up phase (direct FP4)")
    parser.add_argument("--mixed-precision-json", type=Path, default=None,
                        help="Path to scorer sensitivity JSON for mixed-precision QAT. "
                             "Use experiments/scorer_sensitivity_sweep.py to generate.")
    # Lane F-V4 mixed-precision FP4 from per-layer sensitivity profile.
    parser.add_argument("--mixed-precision-from-sensitivity", type=Path, default=None,
                        help="Path to layer_sensitivity.pt from "
                             "experiments/profile_fp4_layer_sensitivity.py. "
                             "Lane F-V4 mixed-precision: bottom "
                             "--mixed-precision-target-rate fraction of params "
                             "(by sensitivity ASC) → FP4, rest → FP16.")
    parser.add_argument("--mixed-precision-target-rate", type=float, default=0.70,
                        help="Lagrangian-style knob for mixed-precision sweep. "
                             "Fraction of TOTAL PARAMS that should be FP4-quantized "
                             "(rest stay FP16). 0.70 = bulk 0.70 FP4 + critical 0.30 "
                             "FP16. 1.0 = uniform FP4 (matches Lane F-V3). 0.0 = no "
                             "FP4 (matches FP32 baseline).")
    parser.add_argument("--mixed-precision-bulk-bits", type=int, default=4,
                        help="Bit depth for FP4-tolerable bulk layers (default 4).")
    parser.add_argument("--mixed-precision-critical-bits", type=int, default=16,
                        help="Bit depth for FP4-intolerant critical layers "
                             "(default 16 = unchanged FP16).")

    # Council D 2026-04-29 PM: EMA on QAT (Quantizr full pipeline). Per
    # CLAUDE.md non-negotiable, every training path must EMA the model.
    parser.add_argument("--ema-decay", type=float, default=0.997,
                        help="EMA decay for the QAT model (Quantizr 0.997). "
                             "EMA is mandatory per CLAUDE.md non-negotiable; "
                             "the EMA shadow is what gets saved as the best "
                             "checkpoint and exported as FP4 binary.")

    args = parser.parse_args()

    cfg = QATConfig(
        checkpoint_path=args.checkpoint,
        upstream_dir=args.upstream,
        output_dir=args.output_dir,
        device=args.device,
        base_ch=args.base_ch,
        mid_ch=args.mid_ch,
        pose_dim=args.pose_dim,
        use_dsconv=args.use_dsconv,
        motion_hidden=args.motion_hidden,
        depth=args.depth,
        embed_dim=args.embed_dim,
        padding_mode=args.padding_mode,
        use_dilation=args.use_dilation,
        use_zoom_flow=args.use_zoom_flow,
        int8_warmup_epochs=0 if args.skip_int8_warmup else args.int8_warmup_epochs,
        fp4_epochs=args.fp4_epochs,
        base_lr=args.lr,
        batch_size=args.batch_size,
        mixed_precision_json=str(args.mixed_precision_json) if args.mixed_precision_json else "",
        mixed_precision_from_sensitivity=str(args.mixed_precision_from_sensitivity)
            if args.mixed_precision_from_sensitivity else "",
        mixed_precision_target_rate=float(args.mixed_precision_target_rate),
        mixed_precision_bulk_bits=int(args.mixed_precision_bulk_bits),
        mixed_precision_critical_bits=int(args.mixed_precision_critical_bits),
        # F2 fix (2026-04-28): thread --protected-pattern-set into QATConfig
        # so the bit allocator below can FORCE protected layers to critical_bits.
        # Without this thread, the flag is parsed + logged but produces no
        # behavior change — Lane SG runs were byte-identical to legacy QAT.
        protected_pattern_set=str(args.protected_pattern_set),
        # Council D 2026-04-29 PM: thread EMA decay through QATConfig.
        ema_decay=float(args.ema_decay),
    )

    # Validate mixed-precision mutual exclusion + range.
    if args.mixed_precision_from_sensitivity and args.mixed_precision_json:
        raise SystemExit(
            "FATAL: --mixed-precision-from-sensitivity and --mixed-precision-json "
            "are mutually exclusive. Pick ONE source for the bit allocation."
        )
    if not (0.0 <= cfg.mixed_precision_target_rate <= 1.0):
        raise SystemExit(
            f"FATAL: --mixed-precision-target-rate must be in [0, 1], "
            f"got {cfg.mixed_precision_target_rate}"
        )
    if cfg.mixed_precision_from_sensitivity and not Path(cfg.mixed_precision_from_sensitivity).exists():
        raise SystemExit(
            f"FATAL: --mixed-precision-from-sensitivity path does not exist: "
            f"{cfg.mixed_precision_from_sensitivity}\n"
            f"       Run experiments/profile_fp4_layer_sensitivity.py first."
        )

    # Preflight
    from tac.preflight import preflight_check
    preflight_check(renderer_path=cfg.checkpoint_path)

    # ── Bug 1 fix (Lane F regression, 2026-04-27): fail FAST if pose_dim>0
    # but no --poses. We do this BEFORE data loading + scorer init so the
    # operator (and the regression test) can falsify the silent-WARN bug
    # within seconds, not minutes. The full path-load + shape validation
    # still runs below after data loading, but any missing-arg error is
    # raised here at the cheapest possible point.
    if cfg.pose_dim > 0 and not args.poses:
        raise SystemExit(
            "FATAL: renderer has pose_dim={} but --poses was not provided.\n"
            "       Pass `--poses <path-to-optimized_poses.pt>` explicitly.\n"
            "       Auto-discovery of gt_poses.pt is intentionally NOT done\n"
            "       (CLAUDE.md FORBIDDEN PATTERNS + findings.md Lane F\n"
            "       regression 2026-04-27). The renderer + poses are a JOINT\n"
            "       artifact; training against zero poses while deploying\n"
            "       against real poses degrades PoseNet by 23x (memory:\n"
            "       project_baseline_poses_load_bearing).".format(cfg.pose_dim)
        )
    if cfg.pose_dim > 0 and args.poses and not Path(args.poses).exists():
        raise SystemExit(
            f"FATAL: --poses path does not exist: {args.poses}\n"
            "       Verify the path. Common locations:\n"
            "       - submissions/baseline_dilated_h64_0_90/optimized_poses.pt\n"
            "       - experiments/results/lane_a_landed/optimized_poses.pt"
        )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qat_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 70)
    print("QAT FINE-TUNING: Float → FP4-Robust")
    print("=" * 70)
    print(f"  Float checkpoint: {cfg.checkpoint_path}")
    print(f"  Schedule: {cfg.int8_warmup_epochs} INT8 warm-up + {cfg.fp4_epochs} FP4")
    print(f"  LR: {cfg.base_lr} (cosine → 0)")
    print(f"  Device: {cfg.device}")
    print()

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data...")

    # GT video
    from tac.eval.auth_eval import AuthEvaluator
    evaluator = AuthEvaluator(
        upstream_dir=Path(cfg.upstream_dir) if UPSTREAM_ROOT is None else UPSTREAM_ROOT,
        device=cfg.device,
    )
    evaluator.load_scorers()
    gt_frames = evaluator.decode_gt_video("0.mkv")
    masks = evaluator.extract_masks(gt_frames, batch_size=4)
    print(f"  GT: {len(gt_frames)} frames, masks: {masks.shape}")

    # Scorers (differentiable)
    from tac.scorer import load_differentiable_scorers
    upstream_dir = str(UPSTREAM_ROOT) if UPSTREAM_ROOT else cfg.upstream_dir
    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)

    # GT poses — explicit only (per CLAUDE.md FORBIDDEN PATTERNS + findings.md
    # "Lane F regression" 2026-04-27). The fail-fast missing-arg/path checks
    # ran already above; here we load + shape-validate. The previous
    # auto-discovery was REMOVED — see Bug 1 commit for context.
    poses = None
    if cfg.pose_dim > 0:
        # Both args.poses and Path(args.poses).exists() were validated above;
        # an invariant failure here is a programming bug, not a user error.
        poses_path = Path(args.poses)
        assert poses_path.exists(), f"invariant violated: {poses_path}"
        poses = torch.load(str(poses_path), map_location="cpu", weights_only=True)
        if isinstance(poses, dict):
            poses = poses.get("poses", poses.get("gt_poses"))
        if poses is None:
            raise SystemExit(
                f"FATAL: --poses dict has neither 'poses' nor 'gt_poses' key: {poses_path}"
            )
        poses = poses.float()
        if poses.ndim != 2 or poses.shape[1] != 6:
            raise SystemExit(
                f"FATAL: --poses wrong shape {tuple(poses.shape)} (expected (N, 6)): {poses_path}"
            )
        print(f"  Poses: {poses.shape} (loaded from --poses {poses_path})")
    elif args.poses:
        # pose_dim=0 renderer cannot use poses — accept --poses but warn.
        print(f"  [INFO] --poses {args.poses} provided but renderer has pose_dim=0; ignoring.")

    # ── Create model and load float weights ───────────────────────────
    print("Creating model...")
    model = create_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    # Auto-detect architecture from ASYM header if possible
    raw = Path(cfg.checkpoint_path).read_bytes()
    if raw[:4] == b"ASYM":
        import struct
        header_len = struct.unpack("<I", raw[4:8])[0]
        header = json.loads(raw[8:8+header_len])
        detected_cfg = {
            "base_ch": header.get("base_ch", cfg.base_ch),
            "mid_ch": header.get("mid_ch", cfg.mid_ch),
            "pose_dim": header.get("pose_dim", cfg.pose_dim),
            "use_dsconv": header.get("use_dsconv", cfg.use_dsconv),
            "depth": header.get("depth", cfg.depth),
            "embed_dim": header.get("embed_dim", cfg.embed_dim),
            "padding_mode": header.get("padding_mode", cfg.padding_mode),
            "use_dilation": header.get("use_dilation", cfg.use_dilation),
            "use_zoom_flow": header.get("use_zoom_flow", cfg.use_zoom_flow),
        }
        if any(detected_cfg[k] != getattr(cfg, k) for k in detected_cfg):
            print(f"  Auto-detected architecture from ASYM header:")
            for k, v in detected_cfg.items():
                old = getattr(cfg, k)
                if v != old:
                    print(f"    {k}: {old} → {v}")
                    setattr(cfg, k, v)
            # Recreate model with correct architecture
            model = create_model(cfg, device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Recreated model: {n_params:,} parameters")

    # ASYM/FP4A binaries embed their arch in the header — load_float_checkpoint
    # may return a model with DIFFERENT arch than the one we built from CLI
    # defaults. Trust the loaded one (council R-arch-drift fix 2026-04-27).
    model = load_float_checkpoint(model, cfg.checkpoint_path, device)
    n_params_loaded = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model: {n_params_loaded:,} parameters (motion.head shape "
          f"{tuple(model.motion.head.weight.shape)})")

    # ── Prepare training data at scorer resolution ─────────────────────
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    n_pairs = masks.shape[0] // 2
    gt_frames_tensor = torch.stack([
        F.interpolate(
            torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0),
            size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        for f in gt_frames
    ])  # (N, 384, 512, 3)

    # ── Zoom warp for use_zoom_flow models ───────────────────────────
    # MUST be before baseline eval — otherwise UnboundLocalError (Round 15 C1)
    zoom_warp = None
    if cfg.use_zoom_flow:
        from tac.radial_zoom import RadialZoomWarp
        zoom_warp = RadialZoomWarp(n_pairs=n_pairs).to(device)
        # Load zoom scalars from checkpoint
        ckpt_path = Path(cfg.checkpoint_path)
        if ckpt_path.suffix == ".pt":
            ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "zoom_warp_state_dict" in ckpt_data:
                zoom_warp.load_state_dict(ckpt_data["zoom_warp_state_dict"])
                print(f"  Loaded zoom scalars from checkpoint")
        else:
            zoom_path = ckpt_path.parent / "zoom_scalars.bin"
            if zoom_path.exists():
                from tac.radial_zoom import load_zoom_scalars
                zoom_warp = load_zoom_scalars(zoom_path, device=str(device))
                print(f"  Loaded zoom scalars from {zoom_path.name}")
        print(f"  RadialZoomWarp: {n_pairs} pairs")

    # ── Baseline quality (float, no quantization) ─────────────────────
    print("\nBaseline quality (float32):")
    baseline = evaluate_fp4_quality(model, masks, gt_frames, poses, device, zoom_warp=zoom_warp)
    print(f"  pose_d={baseline['pose_d']:.5f} seg_d={baseline['seg_d']:.5f} "
          f"distortion={baseline['distortion']:.3f}")

    # ── EMA (Council D 2026-04-29 PM) ──────────────────────────────────
    # Mandatory per CLAUDE.md "EMA — NON-NEGOTIABLE". Quantizr (#1, 0.33)
    # and Selfcomp (#2, 0.38) both EMA through QAT. The EMA shadow set
    # union covers model.parameters AND any LSQ step-size / FP4 fake-quant
    # scales the parametrize layer registers (van den Oord + Ballé
    # endorsement). The Codex-2 hardening at training.py L356-358
    # catches any late-bound parameter added after EMA construction.
    from tac.training import EMA
    ema = EMA(model, decay=cfg.ema_decay)
    print(f"  EMA enabled (decay={cfg.ema_decay})")

    best_distortion = float("inf")
    best_state = None
    history = []

    # ══════════════════════════════════════════════════════════════════
    # PHASE A: INT8 Warm-Up (progressive quantization)
    # Research: Bit-by-Bit (ICLR 2026) — coarse quant first, then fine
    # ══════════════════════════════════════════════════════════════════
    if cfg.int8_warmup_epochs > 0:
        print(f"\n{'='*70}")
        print(f"PHASE A: INT8 Warm-Up ({cfg.int8_warmup_epochs} epochs)")
        print(f"{'='*70}")

        int8_wrapped = apply_int8_fake_quant(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, weight_decay=1e-4)
        total_steps = cfg.int8_warmup_epochs

        model.train()
        for epoch in range(cfg.int8_warmup_epochs):
            lr = cosine_lr(epoch, total_steps, cfg.base_lr, cfg.lr_min_ratio, cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Sample random batch of pairs
            idx = torch.randperm(n_pairs)[:cfg.batch_size]
            m_even = masks[idx * 2].to(device, dtype=torch.int64)
            m_odd = masks[idx * 2 + 1].to(device, dtype=torch.int64)
            gt_even = gt_frames_tensor[idx * 2].to(device)
            gt_odd = gt_frames_tensor[idx * 2 + 1].to(device)
            p = poses[idx].to(device) if poses is not None else None
            batch_ego_flow = None
            if zoom_warp is not None:
                batch_ego_flow = zoom_warp(idx.to(device), m_even.shape[1], m_even.shape[2])

            optimizer.zero_grad()
            loss, metrics = compute_scorer_loss(
                model, m_even, m_odd, gt_even, gt_odd, p,
                posenet, segnet, cfg, device, ego_flow=batch_ego_flow,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            # Council D 2026-04-29: EMA update AFTER optim.step(). The
            # shadow set covers FP4-fake-quant scales as well as weights
            # (LSQ scales benefit from EMA per van den Oord / Ballé).
            ema.update(model)

            if epoch % cfg.log_every == 0:
                print(f"  [INT8] ep {epoch:>4d}/{total_steps} | "
                      f"loss={metrics['total_loss']:.4f} "
                      f"(seg={metrics['seg_loss']:.4f} pose={metrics['pose_loss']:.4f}) "
                      f"lr={lr:.2e}")

            if epoch > 0 and epoch % cfg.eval_every == 0:
                # Council D 2026-04-29: eval THROUGH EMA shadow with
                # snapshot+restore (canonical pattern). Without this the
                # eval prints noisy single-step state instead of the
                # smoothed shadow that ships in the FP4 binary.
                _live_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
                try:
                    ema.apply(model)
                    q = evaluate_fp4_quality(model, masks, gt_frames, poses, device, n_pairs=10, zoom_warp=zoom_warp)
                finally:
                    model.load_state_dict(_live_snapshot)
                print(f"  [INT8] eval: distortion={q['distortion']:.3f} (EMA shadow)")
                history.append({"phase": "int8", "epoch": epoch, **q})

        # Remove INT8 parametrizations before Phase B
        remove_parametrizations(int8_wrapped)
        print("  INT8 parametrizations removed")

        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PHASE B: FP4 Fine-Tune (target quantization)
    # Research: Apple 2025 — cosine decay to zero, no cooldown phase
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE B: FP4 Fine-Tune ({cfg.fp4_epochs} epochs)")
    print(f"{'='*70}")

    mp_json_path = Path(cfg.mixed_precision_json) if cfg.mixed_precision_json else None
    mp_sens_path = Path(cfg.mixed_precision_from_sensitivity) if cfg.mixed_precision_from_sensitivity else None
    bit_allocation_v4: dict[str, int] | None = None
    if mp_sens_path and mp_sens_path.exists():
        # Lane F-V4: mixed-precision allocation derived from per-layer
        # FP4 sensitivity profile (experiments/profile_fp4_layer_sensitivity.py).
        print(f"  Lane F-V4 mixed-precision from {mp_sens_path}")
        print(f"    target_rate={cfg.mixed_precision_target_rate} "
              f"bulk={cfg.mixed_precision_bulk_bits}b "
              f"critical={cfg.mixed_precision_critical_bits}b")
        bit_allocation_v4, sensitivity_used = bit_allocation_from_sensitivity(
            mp_sens_path,
            model,
            target_rate=cfg.mixed_precision_target_rate,
            bulk_bits=cfg.mixed_precision_bulk_bits,
            critical_bits=cfg.mixed_precision_critical_bits,
            # F2 (Codex 2026-04-28): the operator's --protected-pattern-set
            # selection lands here so segnet_prior actually FORCES SegNet
            # decoder layers to FP16, even when the sensitivity profile
            # ranked them as FP4-tolerable.
            protected_pattern_set=cfg.protected_pattern_set,
        )
        # Persist the realised allocation alongside the QAT output for the
        # mixed-precision MXLZ export at end of training.
        (out_dir / "mixed_precision_allocation.json").write_text(json.dumps({
            "allocation": bit_allocation_v4,
            "sensitivity_used": sensitivity_used,
            "target_rate": cfg.mixed_precision_target_rate,
            "bulk_bits": cfg.mixed_precision_bulk_bits,
            "critical_bits": cfg.mixed_precision_critical_bits,
            "protected_pattern_set": cfg.protected_pattern_set,
            "source_profile": str(mp_sens_path),
        }, indent=2))
        fp4_wrapped = apply_mixed_precision_quant(
            model, bit_allocation_v4, block_size=cfg.fp4_block_size,
        )
    elif mp_json_path and mp_json_path.exists():
        mp_data = json.loads(mp_json_path.read_text())
        bit_allocation = mp_data.get("allocation", {})
        print(f"  Using scorer-optimal mixed-precision from {mp_json_path}")
        fp4_wrapped = apply_mixed_precision_quant(
            model, bit_allocation, block_size=cfg.fp4_block_size,
        )
    else:
        fp4_wrapped = apply_fp4_fake_quant(
            model, block_size=cfg.fp4_block_size,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, weight_decay=1e-4)
    total_steps = cfg.fp4_epochs

    model.train()
    for epoch in range(cfg.fp4_epochs):
        lr = cosine_lr(epoch, total_steps, cfg.base_lr, cfg.lr_min_ratio, cfg.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample random batch
        idx = torch.randperm(n_pairs)[:cfg.batch_size]
        m_even = masks[idx * 2].to(device, dtype=torch.int64)
        m_odd = masks[idx * 2 + 1].to(device, dtype=torch.int64)
        gt_even = gt_frames_tensor[idx * 2].to(device)
        gt_odd = gt_frames_tensor[idx * 2 + 1].to(device)
        p = poses[idx].to(device) if poses is not None else None
        batch_ego_flow = None
        if zoom_warp is not None:
            batch_ego_flow = zoom_warp(idx.to(device), m_even.shape[1], m_even.shape[2])

        optimizer.zero_grad()
        loss, metrics = compute_scorer_loss(
            model, m_even, m_odd, gt_even, gt_odd, p,
            posenet, segnet, cfg, device, ego_flow=batch_ego_flow,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        # Council D 2026-04-29: EMA update AFTER optim.step(). Phase B
        # FP4 fake-quant; the EMA shadow tracks both weights and the
        # FP4 codebook scales the parametrize layer registers.
        ema.update(model)

        if epoch % cfg.log_every == 0:
            print(f"  [FP4] ep {epoch:>4d}/{total_steps} | "
                  f"loss={metrics['total_loss']:.4f} "
                  f"(seg={metrics['seg_loss']:.4f} pose={metrics['pose_loss']:.4f}) "
                  f"lr={lr:.2e}")

        if epoch > 0 and epoch % cfg.eval_every == 0:
            # Council D 2026-04-29 PM: evaluate THROUGH the EMA shadow.
            # Snapshot live → apply ema → eval → restore live (canonical
            # train_distill.py + train_renderer_fridrich.py pattern).
            # Without this, "best" is the noisy single-step state, not
            # the smoothed shadow that ships in the FP4 binary.
            _live_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
            try:
                ema.apply(model)
                q = evaluate_fp4_quality(model, masks, gt_frames, poses, device, n_pairs=15, zoom_warp=zoom_warp)
            finally:
                model.load_state_dict(_live_snapshot)
            print(f"  [FP4] eval: distortion={q['distortion']:.3f} "
                  f"(pose={q['pose_d']:.5f} seg={q['seg_d']:.5f}) (EMA shadow)")
            history.append({"phase": "fp4", "epoch": epoch, **q})

            if q["distortion"] < best_distortion:
                best_distortion = q["distortion"]
                # Capture from the EMA shadow (NOT the live model). Strip
                # parametrize keys to plain weight keys so load_state_dict
                # works AFTER parametrizations are removed (C1 fix). The
                # ema.state_dict() returns clones; we then strip via the
                # same C1 path as before.
                raw_state = ema.state_dict()
                clean_state = {}
                for k, v in raw_state.items():
                    if ".parametrizations.weight.original" in k:
                        clean_state[k.replace(".parametrizations.weight.original", ".weight")] = v.cpu().clone()
                    elif ".parametrizations." in k:
                        continue  # skip codebook buffers from parametrize
                    else:
                        clean_state[k] = v.cpu().clone()
                best_state = clean_state
                print(f"  [FP4] ★ NEW BEST: distortion={best_distortion:.3f} "
                      f"(captured from EMA shadow)")

        if epoch > 0 and epoch % cfg.checkpoint_every == 0:
            ckpt_path = out_dir / f"qat_epoch_{epoch}.pt"
            torch.save({
                # Council D 2026-04-29: ship EMA shadow as the primary
                # state_dict (CLAUDE.md non-negotiable: inference bytes
                # come from EMA shadow, never from live final-epoch).
                # ``model_state_dict_live`` preserved for resume.
                "model_state_dict": ema.state_dict(),
                "model_state_dict_live": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "ema_decay": cfg.ema_decay,
                "epoch": epoch,
                "phase": "fp4",
                "distortion": metrics["total_loss"],
            }, ckpt_path)

    # ── Remove FP4 parametrizations and restore best weights ──────────
    remove_parametrizations(fp4_wrapped)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (distortion={best_distortion:.3f})")
    else:
        print("\nNo improvement found during QAT — using final weights")

    # ── Final quality check ───────────────────────────────────────────
    print("\nFinal quality (QAT-trained, float inference):")
    final_float = evaluate_fp4_quality(model, masks, gt_frames, poses, device, zoom_warp=zoom_warp)
    print(f"  pose_d={final_float['pose_d']:.5f} seg_d={final_float['seg_d']:.5f} "
          f"distortion={final_float['distortion']:.3f}")

    # ── Export binary ──────────────────────────────────────────────────
    mp_json_path = Path(cfg.mixed_precision_json) if cfg.mixed_precision_json else None
    mp_sens_path = Path(cfg.mixed_precision_from_sensitivity) if cfg.mixed_precision_from_sensitivity else None
    if mp_sens_path and mp_sens_path.exists() and bit_allocation_v4 is not None:
        # Lane F-V4 mixed-precision MXLZ export (matches training allocation).
        # WARNING: MXLZ format is NOT yet supported by inflate_renderer.py
        # for contest submission. Use FP4A binary for the archive.
        print("\nExporting Lane F-V4 mixed-precision MXLZ binary...")
        print("  WARNING: MXLZ not yet supported by inflate_renderer.py for contest submission")
        from tac.mixed_precision_export import export_int4_lzma2
        fp4_path = out_dir / "renderer_mixed.bin"
        export_int4_lzma2(model, fp4_path, bit_allocation=bit_allocation_v4)
        # ALSO export FP4A for contest-compliant path
        fp4a_path = out_dir / "renderer_fp4.bin"
        from tac.renderer_export import export_asymmetric_checkpoint_fp4
        export_asymmetric_checkpoint_fp4(model, fp4a_path)
        print(f"  Also exported FP4A for contest: {fp4a_path.stat().st_size:,} bytes")
    elif mp_json_path and mp_json_path.exists():
        # Mixed-precision MXLZ export (matches training allocation)
        # WARNING: MXLZ format is NOT yet supported by inflate_renderer.py
        # (standalone contest submission path). Use FP4A for contest submissions.
        # MXLZ is for research/paper evaluation only until inflate support is added.
        print("\nExporting mixed-precision MXLZ binary...")
        print("  WARNING: MXLZ not yet supported by inflate_renderer.py for contest submission")
        from tac.mixed_precision_export import export_int4_lzma2
        mp_data = json.loads(mp_json_path.read_text())
        bit_allocation = mp_data.get("allocation", {})
        fp4_path = out_dir / "renderer_mixed.bin"
        export_int4_lzma2(model, fp4_path, bit_allocation=bit_allocation)
        # ALSO export FP4A for contest-compliant path
        fp4a_path = out_dir / "renderer_fp4.bin"
        from tac.renderer_export import export_asymmetric_checkpoint_fp4
        export_asymmetric_checkpoint_fp4(model, fp4a_path)
        print(f"  Also exported FP4A for contest: {fp4a_path.stat().st_size:,} bytes")
    else:
        # Uniform FP4 export (default, contest-compliant)
        print("\nExporting FP4 binary...")
        from tac.renderer_export import export_asymmetric_checkpoint_fp4
        fp4_path = out_dir / "renderer_fp4.bin"
        export_asymmetric_checkpoint_fp4(model, fp4_path)
    fp4_size = fp4_path.stat().st_size
    print(f"  Export: {fp4_size:,} bytes ({fp4_size/1024:.1f} KB)")

    # Save float checkpoint too (for pose TTO)
    float_path = out_dir / "qat_best_float.pt"
    torch.save({"model_state_dict": model.state_dict()}, float_path)

    # ── Quality after actual FP4 round-trip ───────────────────────────
    print("\nQuality after FP4 round-trip (actual deployment):")
    from tac.renderer_export import load_asymmetric_checkpoint_fp4
    model_fp4 = load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device=str(device))
    model_fp4.eval()

    # Quick distortion check
    fp4_quality = evaluate_fp4_quality(model_fp4, masks, gt_frames, poses, device, zoom_warp=zoom_warp)
    print(f"  pose_d={fp4_quality['pose_d']:.5f} seg_d={fp4_quality['seg_d']:.5f} "
          f"distortion={fp4_quality['distortion']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"QAT SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline (float):    distortion={baseline['distortion']:.3f}")
    print(f"  QAT-trained (float): distortion={final_float['distortion']:.3f}")
    print(f"  FP4 round-trip:      distortion={fp4_quality['distortion']:.3f}")
    print(f"  FP4 degradation:     {fp4_quality['distortion'] - baseline['distortion']:+.3f}")
    print(f"  FP4 binary size:     {fp4_size:,} bytes ({fp4_size/1024:.1f} KB)")
    rate_estimate = 25 * (fp4_size + 280000 + 8000) / 37545489
    print(f"  Estimated archive:   {fp4_size + 280000 + 8000:,} bytes")
    print(f"  Estimated rate:      {rate_estimate:.4f}")
    print(f"  Projected score:     {fp4_quality['distortion'] + rate_estimate:.3f}")
    print(f"  Output: {out_dir}")

    # Save results
    results = {
        "baseline": baseline,
        "qat_float": final_float,
        "fp4_roundtrip": fp4_quality,
        "fp4_degradation": fp4_quality["distortion"] - baseline["distortion"],
        "fp4_size_bytes": fp4_size,
        "n_params": n_params,
        "config": asdict(cfg),
        "history": history,
    }
    (out_dir / "qat_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_dir / 'qat_results.json'}")


if __name__ == "__main__":
    main()
