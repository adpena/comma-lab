#!/usr/bin/env python
"""v62 slave codec — parameterized SingleNeRV slave with three training stages.

Provides the slave half of the v62 master/slave codec via three subcommands:

  init      Phase 1 (SegNet warmup) + Phase 2 (FP main pose) + Phase 3 (QAT + SWA)
            from scratch. Channel widths and latent size are CLI-configurable for
            fast size/pose ablation. Saves ``slave_v62_shrink_{tag}_{fp_end,best,swa}.pt``.

  ft        Continuation fine-tune of an existing slave checkpoint at low LR with
            QAT kept on. Optional ``--freeze-conv`` trains only codes + per-pair-bias.
            Bakes the master to inflate-time FP16 weights so the slave trains against
            the exact pixel values inflate will produce. Saves
            ``slave_v62_shrink_{tag}_{best,swa}.pt``.

  dali-ft   Same continuation FT recipe, but GT references (pose, master tokens,
            slave tokens) are computed from DALI-decoded pairs (the same path the
            official evaluator uses) and the master is loaded with SCN ON without
            an FP16 cast (matches ship export). Saves
            ``slave_v62_shrink_{tag}_{best,swa}.pt`` (default tag=``dali``).
"""
import sys
import time
import math
import argparse
from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent))

from data import preload_rgb_pairs_av, SEGNET_IN_W, SEGNET_IN_H, CAMERA_W, CAMERA_H
from frame_utils import DaliVideoDataset, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from master import TokenRendererV62

# === slave-side architecture (LSQ INT4 QAT layers, NeRVGen, posenet preprocess) ===
# === Moved here from data.py (slave-only after split from data.py) ===

FEAT_H, FEAT_W = 6, 8
NUM_CLASSES = 5


# =============================================================================
# LSQ+ 4-bit symmetric quantization (learnable step per output channel)
# =============================================================================
# Reference: Esser et al., "Learned Step Size Quantization", ICLR 2020.
# For weights we use symmetric 4-bit: levels in [-8, 7].
# Ship: int8 (values clipped to [-8,7]) packed as nibbles + per-channel step fp16.
N_NEG = 8
N_POS = 7


def lsq_quantize(w: torch.Tensor, step: torch.Tensor, out_axis: int = 0):
    """w: tensor with out_axis as the channel dim.
    step: (out_ch,) tensor — per output-channel step size (positive).
    Returns dequantized tensor with STE-through gradient to w AND step.
    """
    # Broadcast step to w shape
    shape = [1] * w.ndim
    shape[out_axis] = -1
    s = step.view(*shape).abs().clamp_min(1e-10)

    # Grad scale factor (from Esser et al.) — helps balance grad between w and s.
    # g_s = 1 / sqrt(N_weights * N_POS)
    n_w_per_ch = w.numel() / step.numel()
    g_s = 1.0 / math.sqrt(max(1.0, n_w_per_ch * N_POS))
    s_scaled = (s - s.detach()) * g_s + s.detach()

    w_scaled = w / s_scaled
    w_clip = w_scaled.clamp(-N_NEG, N_POS)
    w_round = torch.round(w_clip)
    # STE: pass gradient of w_scaled but value of w_round on backward for the round op
    w_q = w_clip + (w_round - w_clip).detach()
    return w_q * s_scaled


def build_hadamard(d: int) -> torch.Tensor:
    """Build d×d normalized Hadamard matrix (d must be power of 2). H @ H.T = I.
    v47: rotate codes so per-row magnitudes are uniform → better int8 quantization."""
    assert (d & (d - 1)) == 0 and d > 0, f"d must be power of 2, got {d}"
    h = torch.tensor([[1.0]])
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (d ** 0.5)


def fake_quant_codes_int8(codes: torch.Tensor) -> torch.Tensor:
    """Per-row int8 fake-quant for embedding codes (v39 code-QAT).
    Mirrors export_model() codes save logic exactly so training-eval == saved-model.
    Closes train-eval / real-eval gap caused by post-training code quantization.
    codes shape: (N, d_lat). Returns dequantized codes with STE through gradient.
    """
    row_max = codes.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    row_scale = row_max / 127.0
    q = (codes / row_scale).round().clamp(-127, 127)
    dq = q * row_scale
    return codes + (dq - codes).detach()




# =============================================================================
# Quantization-aware layers with LSQ+ 4-bit
# =============================================================================
class QLayerBase(nn.Module):
    """Mixin for layers with LSQ+ step parameter."""

    def init_lsq(self, out_channels: int, quantize_weight: bool = True):
        self.quantize_weight = quantize_weight
        self.qat_enabled = False
        if quantize_weight:
            self.step = nn.Parameter(torch.ones(out_channels))
            self._step_initialized = False

    def maybe_init_step(self, w: torch.Tensor):
        # Initialize step per Esser et al.: s = 2 * |w|.mean() / sqrt(N_POS), per output channel.
        if self.quantize_weight and not self._step_initialized:
            with torch.no_grad():
                w_flat = w.reshape(w.shape[0], -1)
                mean_abs = w_flat.abs().mean(dim=1).clamp_min(1e-6)
                init = 2.0 * mean_abs / math.sqrt(N_POS)
                self.step.data.copy_(init)
            self._step_initialized = True

    def set_qat(self, enabled: bool):
        self.qat_enabled = enabled


class QConv2d(nn.Conv2d, QLayerBase):
    def __init__(self, *args, quantize_weight: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_lsq(self.out_channels, quantize_weight)

    def forward(self, x):
        if self.qat_enabled and self.quantize_weight:
            self.maybe_init_step(self.weight)
            w = lsq_quantize(self.weight, self.step, out_axis=0)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QLinear(nn.Linear, QLayerBase):
    def __init__(self, *args, quantize_weight: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_lsq(self.out_features, quantize_weight)

    def forward(self, x):
        if self.qat_enabled and self.quantize_weight:
            self.maybe_init_step(self.weight)
            w = lsq_quantize(self.weight, self.step, out_axis=0)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


# =============================================================================
# NeRV-style blocks with GeLU
# =============================================================================
def make_act(num_ch: int) -> nn.Module:
    """GeLU activation (parameter-free, smooth non-monotonic)."""
    return nn.GELU()


class NeRVBlock(nn.Module):
    """Depthwise + Pointwise expand + PixelShuffle(s) + GeLU."""

    def __init__(self, c_in: int, c_out: int, s: int = 2, qw: bool = True):
        super().__init__()
        self.dw = QConv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False, quantize_weight=qw)
        self.pw = QConv2d(c_in, c_out * s * s, kernel_size=1, bias=True, quantize_weight=qw)
        self.ps = nn.PixelShuffle(s)
        self.act = make_act(c_out)

    def forward(self, x):
        return self.act(self.ps(self.pw(self.dw(x))))


class NeRVGen(nn.Module):
    """Index → frame pair generator (v3: v1-style joint 6-channel head, cam-res output).

    Joint head collapses the former master_head / slave_head into a single 1×1 conv
    that emits 6 channels, split 3+3 into (slave, master). This gives both outputs
    the full decoder capacity (v1 design) rather than bottlenecking through separate
    small refine paths (v2 design, which regressed pose).
    """

    CHANNELS = [96, 64, 48, 32, 24, 16, 16]  # v3 (champion: real-eval 0.79). v4 wider tried, hurt rate w/o helping seg.

    def __init__(self, num_pairs: int, d_lat: int = 32, use_film: bool = False, film_dim: int = 32, use_per_pair_bias: bool = False, use_hadamard_codes: bool = False):
        super().__init__()
        self.num_pairs = num_pairs
        self.d_lat = d_lat
        self.use_film = use_film
        self.use_per_pair_bias = use_per_pair_bias
        self.use_hadamard_codes = use_hadamard_codes
        self.codes = nn.Embedding(num_pairs, d_lat)
        nn.init.normal_(self.codes.weight, std=0.05)
        self.qat_codes = False  # v39: enabled by set_qat() to fake-quant codes to int8

        # v47: Hadamard rotation buffer (non-persistent — derivable from d_lat)
        if use_hadamard_codes:
            self.register_buffer("hadamard_H", build_hadamard(d_lat), persistent=False)

        # v42: Per-pair learnable bias on head output (3600 params, ~7 KB fp16).
        # Each pair gets its own per-channel offset; frees decoder from memorizing per-pair color shifts.
        if use_per_pair_bias:
            self.per_pair_bias = nn.Embedding(num_pairs, 6)
            nn.init.zeros_(self.per_pair_bias.weight)
        else:
            self.per_pair_bias = None

        base = self.CHANNELS[0]
        self.stem = QLinear(d_lat, base * FEAT_H * FEAT_W, bias=True, quantize_weight=True)
        self.stem_act = make_act(base)

        blocks = []
        for i in range(6):
            blocks.append(NeRVBlock(self.CHANNELS[i], self.CHANNELS[i + 1], s=2))
        self.blocks = nn.ModuleList(blocks)

        trunk_ch = self.CHANNELS[-1]
        if use_film:
            # FiLM-dual-heads: pose info routed via FiLM only into slave (so master stays seg-pure)
            self.master_head = QConv2d(trunk_ch, 3, kernel_size=1, bias=True, quantize_weight=False)
            self.pose_mlp = nn.Sequential(
                QLinear(d_lat, film_dim, bias=True, quantize_weight=True),
                nn.GELU(),
                QLinear(film_dim, film_dim, bias=True, quantize_weight=True),
            )
            self.slave_film_proj = QLinear(film_dim, trunk_ch * 2, bias=True, quantize_weight=True)  # (gamma, beta)
            self.slave_head = QConv2d(trunk_ch, 3, kernel_size=1, bias=True, quantize_weight=False)
        else:
            # Joint head: single 1×1 conv → 6 channels (3 slave + 3 master), quant-free final layer
            self.head = QConv2d(trunk_ch, 6, kernel_size=1, bias=True, quantize_weight=False)

    def set_qat(self, enabled: bool):
        # v39 code-QAT was tested, hurt pose stability (real-eval 0.73 vs v3.6 0.71). Disabled.
        # self.qat_codes = enabled  # uncomment to re-enable code-QAT
        for m in self.modules():
            if isinstance(m, (QConv2d, QLinear)) and m is not self.codes:
                m.set_qat(enabled)

    def _embed(self, pair_idx: torch.Tensor) -> torch.Tensor:
        """Embedding lookup with optional v39 code-QAT and v47 Hadamard rotation."""
        if self.qat_codes:
            code = fake_quant_codes_int8(self.codes.weight)[pair_idx]
        else:
            code = self.codes(pair_idx)
        if self.use_hadamard_codes:
            code = code @ self.hadamard_H
        return code

    def trunk_forward(self, pair_idx: torch.Tensor) -> torch.Tensor:
        code = self._embed(pair_idx)
        x = self.stem(code).view(-1, self.CHANNELS[0], FEAT_H, FEAT_W)
        x = self.stem_act(x)
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, trunk_ch, 384, 512)

    def forward(self, pair_idx: torch.Tensor):
        feat = self.trunk_forward(pair_idx)
        if self.use_film:
            # FiLM-dual-heads: master uses raw feat (seg-pure); slave gets pose-conditioned feat.
            code = self._embed(pair_idx)
            pose_emb = self.pose_mlp(code)
            film = self.slave_film_proj(pose_emb)
            trunk_ch = feat.shape[1]
            gamma = film[:, :trunk_ch].view(-1, trunk_ch, 1, 1)
            beta = film[:, trunk_ch:].view(-1, trunk_ch, 1, 1)
            slave_feat = feat * (1.0 + gamma) + beta
            master_lo = torch.sigmoid(self.master_head(feat)) * 255.0
            slave_lo = torch.sigmoid(self.slave_head(slave_feat)) * 255.0
        else:
            head_out = self.head(feat)
            if self.per_pair_bias is not None:
                head_out = head_out + self.per_pair_bias(pair_idx).view(-1, 6, 1, 1)
            raw = torch.sigmoid(head_out) * 255.0  # (B, 6, 384, 512)
            slave_lo = raw[:, 0:3]
            master_lo = raw[:, 3:6]
        # Camera-res output: bilinear to (874, 1164) inside the generator.
        master = F.interpolate(master_lo, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        slave = F.interpolate(slave_lo, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        return slave, master



# =============================================================================
# Differentiable YUV6 + PoseNet preprocess
# =============================================================================
def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., : 2 * h2, : 2 * w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    y00, y10 = y[:, 0::2, 0::2], y[:, 1::2, 0::2]
    y01, y11 = y[:, 0::2, 1::2], y[:, 1::2, 1::2]
    u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)


def posenet_preprocess_grad(pair: torch.Tensor) -> torch.Tensor:
    """Differentiable PoseNet preprocess. pair: (B, 2, 3, H, W)."""
    b, t = pair.shape[0], pair.shape[1]
    x = einops.rearrange(pair, "b t c h w -> (b t) c h w", b=b, t=t, c=3)
    x = F.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode="bilinear")
    return einops.rearrange(diff_rgb_to_yuv6(x), "(b t) c h w -> b (t c) h w", b=b, t=t, c=6)






DEV = "cuda"
N = 600
BS = 16
LR_WARMUP = 1e-3
LR_MAIN = 1e-3
LR_QAT = 1e-4
TEMP_W_START, TEMP_W_END = 1.0, 0.2
SEG_STABILITY_W = 0.0
SAVE_DIR = Path(__file__).resolve().parent.parent / "training_workspace"
MASTER_PT = SAVE_DIR / "v62_full_best.pt"


class NeRVBlock(nn.Module):
    """Quantized depthwise + pointwise NeRV upsampling block."""

    def __init__(self, c_in, c_out, s=2):
        super().__init__()
        self.dw = QConv2d(c_in, c_in, kernel_size=3, padding=1,
                          groups=c_in, bias=False, quantize_weight=True)
        self.pw = QConv2d(c_in, c_out * s * s, kernel_size=1,
                          bias=True, quantize_weight=True)
        self.ps = nn.PixelShuffle(s)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.ps(self.pw(self.dw(x))))


class ShrinkSingleNeRV(nn.Module):
    """Parameterized SingleNeRV slave with INT4 conv weights and INT8 codes.

    Channel widths and latent size are CLI-configurable to sweep small/tiny configs.
    Includes a per-pair RGB bias (3 floats per frame) that materially improves pose
    when trained end-to-end.
    """

    def __init__(self, num_pairs=N, d_lat=16, channels=(64, 48, 32, 24, 16, 12, 12)):
        super().__init__()
        assert len(channels) == 7, "channels must have 7 entries (1 stem + 6 blocks)"
        self.codes = nn.Embedding(num_pairs, d_lat)
        nn.init.normal_(self.codes.weight, std=0.5)
        self.stem = QLinear(d_lat, channels[0] * FEAT_H * FEAT_W,
                            bias=True, quantize_weight=True)
        self.stem_act = nn.GELU()
        self.blocks = nn.ModuleList([
            NeRVBlock(channels[i], channels[i + 1], s=2) for i in range(6)
        ])
        self.head = QConv2d(channels[-1], 3, kernel_size=1, bias=True, quantize_weight=False)
        self.per_pair_bias = nn.Embedding(num_pairs, 3)
        nn.init.zeros_(self.per_pair_bias.weight)
        self.qat_codes = False
        self.channels = channels

    def set_qat(self, on):
        self.qat_codes = on
        for m in self.modules():
            if isinstance(m, (QConv2d, QLinear)):
                m.set_qat(on)

    def forward(self, idx):
        z = self.codes(idx)
        if self.qat_codes:
            z = fake_quant_codes_int8(z)
        x = self.stem(z).view(-1, self.channels[0], FEAT_H, FEAT_W)
        x = self.stem_act(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x) + self.per_pair_bias(idx).view(-1, 3, 1, 1)
        raw = torch.sigmoid(out) * 255.0
        return F.interpolate(raw, size=(CAMERA_H, CAMERA_W),
                             mode="bilinear", align_corners=False)


def cmd_init(args):
    """Train slave from scratch: SegNet warmup -> FP main pose -> QAT + SWA."""
    channels = tuple(int(c) for c in args.channels.split(","))
    if args.smoke:
        WARMUP_EPS, MAIN_EPS, QAT_EPS = 50, 150, 300
    else:
        WARMUP_EPS, MAIN_EPS, QAT_EPS = 200, 400, 800
    if args.skip_warmup:
        WARMUP_EPS = 0

    print(f"[v62ss] tag={args.tag}  channels={channels}  d_lat={args.d_lat}  "
          f"W={WARMUP_EPS} M={MAIN_EPS} Q={QAT_EPS}", flush=True)

    print(f"[v62ss] loading v62 master from {MASTER_PT}", flush=True)
    master_data = torch.load(MASTER_PT, map_location=DEV, weights_only=False)
    print(f"[v62ss] master best_seg={master_data.get('best_seg')}", flush=True)
    master = TokenRendererV62().to(DEV).eval()
    master.load_state_dict(master_data["state_dict"], strict=True)
    master.set_scn(True)
    for p in master.parameters():
        p.requires_grad = False

    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    posenet = PoseNet().eval().to(DEV)
    posenet.load_state_dict(load_file(posenet_sd_path, device=DEV))
    for p in segnet.parameters():
        p.requires_grad = False
    for p in posenet.parameters():
        p.requires_grad = False

    print("[v62ss] preload RGB pairs...", flush=True)
    rgb_pairs = preload_rgb_pairs_av(ROOT / "videos", ROOT / "public_test_video_names.txt")

    print("[v62ss] gt_slave_tokens, gt_master_tokens, gt_pose...", flush=True)
    gt_slave_tokens = torch.empty(N, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long, device=DEV)
    gt_master_tokens = torch.empty(N, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long, device=DEV)
    gt_pose = torch.empty(N, 6, dtype=torch.float32, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            x = rgb_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)
            x_s = x[:, 0:1]
            x_m = x[:, 1:2]
            gt_slave_tokens[s:e] = segnet(segnet.preprocess_input(x_s)).argmax(1)
            gt_master_tokens[s:e] = segnet(segnet.preprocess_input(x_m)).argmax(1)
            gt_pose[s:e] = posenet(posenet.preprocess_input(x))["pose"][..., :6].float()
    del rgb_pairs

    print("[v62ss] precompute master frames (frozen)...", flush=True)
    master_cam = torch.empty(N, 3, CAMERA_H, CAMERA_W, dtype=torch.float16, device="cpu")
    with torch.inference_mode():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = torch.arange(s, e, device=DEV)
            mst = master(gt_master_tokens[s:e], idx).clamp(0, 255).round()
            master_cam[s:e] = mst.half().cpu()

    @torch.no_grad()
    def eval_pose(slave_gen):
        pose_sum = 0.0
        pose_n = 0
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = torch.arange(s, e, device=DEV)
            slv = slave_gen(idx).clamp(0, 255).round()
            mst = master_cam[s:e].to(DEV).float()
            pair = torch.stack([slv, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_sum += F.mse_loss(fp, gt_pose[s:e], reduction="sum").item() / 6.0
            pose_n += pair.shape[0]
        return pose_sum / max(pose_n, 1)

    gen = ShrinkSingleNeRV(num_pairs=N, d_lat=args.d_lat, channels=channels).to(DEV)
    gen.set_qat(True)
    n_params = sum(p.numel() for p in gen.parameters())
    n_quantizable = sum(p.numel() for n, p in gen.named_parameters()
                        if 'codes' not in n and 'per_pair_bias' not in n)
    est_int4_kb = n_quantizable / 2 / 1024
    est_codes_kb = N * args.d_lat / 1024  # INT8 codes
    est_master_total_kb = est_int4_kb + est_codes_kb
    print(f"[v62ss] slave params: {n_params:_}  decoder INT4 ≈ {est_int4_kb:.1f} KB  "
          f"codes INT8 ≈ {est_codes_kb:.1f} KB  total ≈ {est_master_total_kb:.1f} KB", flush=True)

    init_pose = eval_pose(gen)
    print(f"[v62ss] init pose: {init_pose:.6f}", flush=True)

    t0 = time.time()
    if WARMUP_EPS > 0:
        print(f"[v62ss] === Phase 1: SegNet warmup ({WARMUP_EPS} ep) ===", flush=True)
        opt = torch.optim.Adam(gen.parameters(), lr=LR_WARMUP, betas=(0.9, 0.99))
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=WARMUP_EPS)
        for ep in range(WARMUP_EPS):
            gen.train()
            progress = ep / max(WARMUP_EPS - 1, 1)
            # Anneal CE temperature from 1.0 to 0.2 to sharpen targets late in warmup
            temp = TEMP_W_START * (TEMP_W_END / TEMP_W_START) ** progress
            perm = torch.randperm(N, device=DEV)
            for s in range(0, N, BS):
                e = min(s + BS, N)
                idx = perm[s:e]
                slv = gen(idx)
                slv_q = slv + (slv.round() - slv).detach()
                x = slv_q.unsqueeze(1)
                logits = segnet(segnet.preprocess_input(x))
                loss = F.cross_entropy(logits / temp, gt_slave_tokens[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                opt.step()
            sched1.step()
            if (ep + 1) % 25 == 0 or ep == 0:
                gen.eval()
                pose = eval_pose(gen)
                gen.train()
                print(f"[v62ss.warm] ep {ep+1:>3d}/{WARMUP_EPS}  pose={pose:.6f}  dt={time.time()-t0:.0f}s",
                      flush=True)

    print(f"[v62ss] === Phase 2: FP main pose ({MAIN_EPS} ep) ===", flush=True)
    opt = torch.optim.Adam(gen.parameters(), lr=LR_MAIN, betas=(0.9, 0.99))
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAIN_EPS)
    for ep in range(MAIN_EPS):
        gen.train()
        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = perm[s:e]
            slv = gen(idx)
            slv_q = slv + (slv.round() - slv).detach()
            with torch.no_grad():
                mst = master_cam[idx.cpu()].to(DEV).float()
            pair = torch.stack([slv_q, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_loss = F.mse_loss(fp, gt_pose[idx])
            x = slv_q.unsqueeze(1)
            seg_logits = segnet(segnet.preprocess_input(x))
            ce = F.cross_entropy(seg_logits, gt_slave_tokens[idx])
            # 30x pose weight balances pose mse against the seg CE stabilizer
            loss = 30.0 * pose_loss + SEG_STABILITY_W * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
        sched2.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            gen.eval()
            pose = eval_pose(gen)
            gen.train()
            print(f"[v62ss.main] ep {ep+1:>3d}/{MAIN_EPS}  pose={pose:.6f}  dt={time.time()-t0:.0f}s",
                  flush=True)

    gen.eval()
    fp_pose = eval_pose(gen)
    print(f"\n[v62ss] === END FP: pose={fp_pose:.6f} ===\n", flush=True)
    torch.save({"state_dict": gen.state_dict(), "fp_pose": fp_pose,
                "channels": channels, "d_lat": args.d_lat,
                "kind": f"v62_slave_shrink_{args.tag}_fp"},
               SAVE_DIR / f"slave_v62_shrink_{args.tag}_fp_end.pt")

    print(f"[v62ss] === Phase 3: QAT continuation ({QAT_EPS} ep) ===", flush=True)
    opt = torch.optim.Adam(gen.parameters(), lr=LR_QAT, betas=(0.9, 0.99))
    sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=QAT_EPS, eta_min=LR_QAT * 0.05)
    best_pose = fp_pose
    best_state = None
    swa_buf = None
    swa_count = 0
    # Start SWA in the last quarter (or last 50 ep, whichever is larger)
    SWA_FROM = max(0, QAT_EPS - max(50, QAT_EPS // 4))

    for ep in range(QAT_EPS):
        gen.train()
        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = perm[s:e]
            slv = gen(idx)
            slv_q = slv + (slv.round() - slv).detach()
            with torch.no_grad():
                mst = master_cam[idx.cpu()].to(DEV).float()
            pair = torch.stack([slv_q, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_loss = F.mse_loss(fp, gt_pose[idx])
            x = slv_q.unsqueeze(1)
            seg_logits = segnet(segnet.preprocess_input(x))
            ce = F.cross_entropy(seg_logits, gt_slave_tokens[idx])
            loss = 30.0 * pose_loss + SEG_STABILITY_W * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
        sched3.step()
        if ep >= SWA_FROM:
            if swa_buf is None:
                swa_buf = {k: v.detach().clone().float()
                           for k, v in gen.state_dict().items() if v.dtype.is_floating_point}
                swa_count = 1
            else:
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point and k in swa_buf:
                        swa_buf[k].mul_(swa_count / (swa_count + 1)).add_(v.detach().float() / (swa_count + 1))
                swa_count += 1
        if (ep + 1) % 25 == 0 or ep == 0:
            gen.eval()
            pose = eval_pose(gen)
            gen.train()
            if pose < best_pose:
                best_pose = pose
                best_state = {k: v.detach().clone().cpu() for k, v in gen.state_dict().items()}
            print(f"[v62ss.qat] ep {ep+1:>3d}/{QAT_EPS}  pose={pose:.6f}  best={best_pose:.6f}  dt={time.time()-t0:.0f}s",
                  flush=True)

    gen.eval()
    final_pose = eval_pose(gen)
    print(f"\n[v62ss] FINAL pose: {final_pose:.6f}", flush=True)
    print(f"[v62ss] best:       {best_pose:.6f}", flush=True)

    save_state = best_state if best_state is not None else {k: v.detach().cpu() for k, v in gen.state_dict().items()}
    torch.save({"state_dict": save_state, "best_pose": best_pose, "final_pose": final_pose,
                "channels": channels, "d_lat": args.d_lat,
                "est_master_total_kb": est_master_total_kb,
                "kind": f"v62_slave_shrink_{args.tag}_best"},
               SAVE_DIR / f"slave_v62_shrink_{args.tag}_best.pt")

    swa_pose = None
    if swa_buf is not None and swa_count > 1:
        backup = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        swa_state = dict(backup)
        for k, v in swa_buf.items():
            swa_state[k] = v.to(backup[k].device, backup[k].dtype)
        gen.load_state_dict(swa_state)
        gen.eval()
        swa_pose = eval_pose(gen)
        print(f"[v62ss] SWA-{swa_count} pose: {swa_pose:.6f}", flush=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in swa_state.items()},
                    "swa_pose": swa_pose, "channels": channels, "d_lat": args.d_lat,
                    "kind": f"v62_slave_shrink_{args.tag}_swa"},
                   SAVE_DIR / f"slave_v62_shrink_{args.tag}_swa.pt")

    # Score projection assumes v62 master full numbers + neural compress v3
    seg = 0.000695
    pose_for_score = swa_pose if swa_pose is not None and swa_pose < best_pose else best_pose
    master_kb = 4 + 5 + 85 + 9   # decoder + film + tokens + compressor model (v2-mild target)
    total_kb = master_kb + est_master_total_kb
    rate = total_kb * 1024 / 37545489
    score = 100 * seg + (10 * pose_for_score) ** 0.5 + 25 * rate
    print(f"\n[v62ss] === SCORE PROJECTION ({args.tag}) ===", flush=True)
    print(f"[v62ss] slave: {est_master_total_kb:.1f} KB   total submission: {total_kb:.1f} KB", flush=True)
    print(f"[v62ss] seg={seg:.6f}  pose={pose_for_score:.6f}  rate={rate:.5f}", flush=True)
    print(f"[v62ss] score = {100*seg:.3f} + {(10*pose_for_score)**0.5:.3f} + {25*rate:.3f} = {score:.3f}",
          flush=True)
    print(f"[v62ss] DONE — total wall {time.time()-t0:.0f}s", flush=True)


def cmd_ft(args):
    """Continuation FT of an existing slave with master baked to inflate-time FP16.

    The bake step casts master weights to FP16 and back to FP32 so the slave
    trains against the EXACT pixel values inflate produces; without it the
    train/eval pose gap is ~2x.
    """
    print(f"[v62ssft] tag={args.tag} init={Path(args.init).name} "
          f"epochs={args.epochs} lr={args.lr} freeze_conv={args.freeze_conv}", flush=True)

    master_data = torch.load(MASTER_PT, map_location=DEV, weights_only=False)
    master = TokenRendererV62().to(DEV).eval()
    master.load_state_dict(master_data["state_dict"], strict=True)
    master.set_scn(True)
    # Bake master to inflate-time FP16 weights to re-pair slave with deployed master.
    # Without this, slave is trained against FP32-SCN master but eval uses FP16-baked
    # master, producing a 2x train/eval pose gap.
    from master import _scn_quantize as _scn_q, SCNConv2d, SCNLinear
    with torch.no_grad():
        for mod in master.modules():
            if isinstance(mod, (SCNConv2d, SCNLinear)):
                w_dq = _scn_q(mod.weight, mod.b, mod.e, 0)
                mod.weight.data = w_dq.to(torch.float16).to(torch.float32)
                mod._scn_on = False
        for p in master.parameters():
            if torch.is_floating_point(p):
                p.data = p.data.to(torch.float16).to(torch.float32)
    print(f"[v62ssft] master baked to inflate-time FP16 weights (eliminates pose train-eval gap)",
          flush=True)
    for p in master.parameters():
        p.requires_grad = False
    print(f"[v62ssft] master best_seg={master_data.get('best_seg')}", flush=True)

    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    posenet = PoseNet().eval().to(DEV)
    posenet.load_state_dict(load_file(posenet_sd_path, device=DEV))
    for p in segnet.parameters():
        p.requires_grad = False
    for p in posenet.parameters():
        p.requires_grad = False

    print("[v62ssft] preload RGB pairs...", flush=True)
    rgb_pairs = preload_rgb_pairs_av(ROOT / "videos", ROOT / "public_test_video_names.txt")

    print("[v62ssft] gt tokens + pose...", flush=True)
    gt_slave_tokens = torch.empty(N, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long, device=DEV)
    gt_master_tokens = torch.empty(N, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long, device=DEV)
    gt_pose = torch.empty(N, 6, dtype=torch.float32, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            x = rgb_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)
            x_s = x[:, 0:1]
            x_m = x[:, 1:2]
            gt_slave_tokens[s:e] = segnet(segnet.preprocess_input(x_s)).argmax(1)
            gt_master_tokens[s:e] = segnet(segnet.preprocess_input(x_m)).argmax(1)
            gt_pose[s:e] = posenet(posenet.preprocess_input(x))["pose"][..., :6].float()
    del rgb_pairs

    print("[v62ssft] precompute master frames (frozen)...", flush=True)
    master_cam = torch.empty(N, 3, CAMERA_H, CAMERA_W, dtype=torch.float16, device="cpu")
    with torch.inference_mode():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = torch.arange(s, e, device=DEV)
            mst = master(gt_master_tokens[s:e], idx).clamp(0, 255).round()
            master_cam[s:e] = mst.half().cpu()

    @torch.no_grad()
    def eval_pose(slave_gen):
        pose_sum = 0.0
        pose_n = 0
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = torch.arange(s, e, device=DEV)
            slv = slave_gen(idx).clamp(0, 255).round()
            mst = master_cam[s:e].to(DEV).float()
            pair = torch.stack([slv, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_sum += F.mse_loss(fp, gt_pose[s:e], reduction="sum").item() / 6.0
            pose_n += pair.shape[0]
        return pose_sum / max(pose_n, 1)

    blob = torch.load(args.init, map_location=DEV, weights_only=False)
    channels = tuple(blob.get("channels", (24, 16, 12, 8, 8, 6, 6)))
    d_lat = int(blob.get("d_lat", 6))
    print(f"[v62ssft] slave config: channels={channels}  d_lat={d_lat}", flush=True)

    gen = ShrinkSingleNeRV(num_pairs=N, d_lat=d_lat, channels=channels).to(DEV)
    gen.set_qat(True)
    miss, unexp = gen.load_state_dict(blob["state_dict"], strict=False)
    # Prevent maybe_init_step() from overwriting the loaded LSQ step.
    # _step_initialized is a Python attr, not a state_dict buffer, so it must be
    # set manually after load_state_dict.
    for m in gen.modules():
        if isinstance(m, (QConv2d, QLinear)) and getattr(m, "quantize_weight", False):
            m._step_initialized = True
    print(f"[v62ssft] loaded init: miss={len(miss)} unexp={len(unexp)} "
          f"prev_pose={blob.get('swa_pose', blob.get('best_pose', '?'))}", flush=True)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"[v62ssft] slave params: {n_params:_}", flush=True)

    init_pose = eval_pose(gen)
    print(f"[v62ssft] init pose: {init_pose:.6f}", flush=True)

    if args.freeze_conv:
        for n, p in gen.named_parameters():
            if "codes" in n or "per_pair_bias" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        print(f"[v62ssft] frozen QAT conv weights; trainable params: {n_trainable:_}", flush=True)

    # Param groups: codes and per-pair-bias get higher LR than conv weights
    code_params, bias_params, other_params = [], [], []
    for n, p in gen.named_parameters():
        if not p.requires_grad:
            continue
        if "codes" in n:
            code_params.append(p)
        elif "per_pair_bias" in n:
            bias_params.append(p)
        else:
            other_params.append(p)
    pg = []
    if other_params:
        pg.append({"params": other_params, "lr": args.lr})
    if code_params:
        pg.append({"params": code_params, "lr": args.lr_codes})
    if bias_params:
        pg.append({"params": bias_params, "lr": args.lr_bias})
    opt = torch.optim.Adam(pg, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_pose = init_pose
    best_state = {k: v.detach().clone().cpu() for k, v in gen.state_dict().items()}
    best_ep = -1
    swa_buf = None
    swa_count = 0
    t0 = time.time()
    for ep in range(args.epochs):
        gen.train()
        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, BS):
            e = min(s + BS, N)
            idx = perm[s:e]
            slv = gen(idx)
            slv_q = slv + (slv.round() - slv).detach()
            with torch.no_grad():
                mst = master_cam[idx.cpu()].to(DEV).float()
            pair = torch.stack([slv_q, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_loss = F.mse_loss(fp, gt_pose[idx])
            x = slv_q.unsqueeze(1)
            seg_logits = segnet(segnet.preprocess_input(x))
            ce = F.cross_entropy(seg_logits, gt_slave_tokens[idx])
            loss = args.pose_w * pose_loss + args.seg_stability * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep >= args.swa_from:
            if swa_buf is None:
                swa_buf = {k: v.detach().clone().float()
                           for k, v in gen.state_dict().items() if v.dtype.is_floating_point}
                swa_count = 1
            else:
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point and k in swa_buf:
                        swa_buf[k].mul_(swa_count / (swa_count + 1)).add_(v.detach().float() / (swa_count + 1))
                swa_count += 1

        if (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1:
            gen.eval()
            pose = eval_pose(gen)
            gen.train()
            improved = pose < best_pose
            if improved:
                best_pose = pose
                best_ep = ep
                best_state = {k: v.detach().clone().cpu() for k, v in gen.state_dict().items()}
                torch.save({"state_dict": best_state, "best_pose": best_pose,
                            "channels": channels, "d_lat": d_lat,
                            "kind": f"v62_slave_shrink_{args.tag}_best"},
                           SAVE_DIR / f"slave_v62_shrink_{args.tag}_best.pt")
            tag_s = " *NEW BEST*" if improved else ""
            print(f"[v62ssft] ep {ep+1:>4d}/{args.epochs}  pose={pose:.6f}  best={best_pose:.6f}@ep{best_ep+1}  "
                  f"init={init_pose:.6f}  dt={time.time()-t0:.0f}s{tag_s}", flush=True)

    swa_pose = None
    if swa_buf is not None and swa_count > 1:
        backup = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        swa_state = dict(backup)
        for k, v in swa_buf.items():
            swa_state[k] = v.to(backup[k].device, backup[k].dtype)
        gen.load_state_dict(swa_state)
        gen.eval()
        swa_pose = eval_pose(gen)
        print(f"[v62ssft] SWA-{swa_count} pose: {swa_pose:.6f}", flush=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in swa_state.items()},
                    "swa_pose": swa_pose, "channels": channels, "d_lat": d_lat,
                    "kind": f"v62_slave_shrink_{args.tag}_swa"},
                   SAVE_DIR / f"slave_v62_shrink_{args.tag}_swa.pt")

    print(f"\n[v62ssft] DONE  init={init_pose:.6f}  best={best_pose:.6f}@ep{best_ep+1}  "
          f"swa={swa_pose if swa_pose is None else f'{swa_pose:.6f}'}  "
          f"wall={time.time()-t0:.0f}s", flush=True)


def _gt_dali_load(bs):
    """Load all 600 GT pairs via DALI/NVDEC (matches official evaluator decode path)."""
    print("[sd-ft] loading GT via DALI/NVDEC...", flush=True)
    with open(ROOT / "public_test_video_names.txt") as f:
        names = [line.strip() for line in f.readlines()]
    ds = DaliVideoDataset(names, data_dir=ROOT / "videos",
                          batch_size=bs, device=torch.device("cuda"))
    ds.prepare_data()
    parts = []
    for _, _, batch in ds:
        parts.append(batch.detach().cpu())
    all_frames = torch.cat(parts, dim=0)
    assert all_frames.shape[0] >= N, f"DALI got {all_frames.shape[0]}, need {N}"
    return all_frames[:N].contiguous()


def cmd_dali_ft(args):
    """Continuation FT with DALI-decoded GT references and SCN-on master (no FP16 cast).

    Two fixes vs cmd_ft:
      1. GT references (gt_pose, gt_*_tokens) come from DALI-decoded pairs — the
         same path the official evaluator uses. PyAV->DALI explained 100% of the
         train/eval pose gap in the gap probe.
      2. Master is loaded with SCN ON without an FP16 cast. Ship export keeps
         non-SCN params (GroupNorm/biases/embeddings) as FP32; the prior FP16
         cast was a real divergence from ship.
    """
    print(f"[sd-ft] init_slave={Path(args.init).name}  init_master={Path(args.master).name}  "
          f"epochs={args.epochs} bs={args.bs} lr={args.lr} freeze_conv={args.freeze_conv}", flush=True)

    # Master: SCN ON, no FP16 cast (matches ship export bit-for-bit).
    master_data = torch.load(args.master, map_location=DEV, weights_only=False)
    master = TokenRendererV62().to(DEV).eval()
    master.load_state_dict(master_data["state_dict"], strict=True)
    master.set_scn(True)
    for p in master.parameters():
        p.requires_grad = False
    print(f"[sd-ft] master loaded SCN ON (no FP16 cast). best_seg={master_data.get('best_seg')} "
          f"gt_decoder={master_data.get('gt_decoder', 'unknown')}", flush=True)

    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    posenet = PoseNet().eval().to(DEV)
    posenet.load_state_dict(load_file(posenet_sd_path, device=DEV))
    for p in segnet.parameters():
        p.requires_grad = False
    for p in posenet.parameters():
        p.requires_grad = False

    gt_pairs = _gt_dali_load(bs=args.bs)
    H_in, W_in = segnet_model_input_size[1], segnet_model_input_size[0]
    print(f"[sd-ft] computing gt_tokens, gt_pose at {H_in}x{W_in} from DALI GT...", flush=True)
    gt_slave_tokens = torch.empty(N, H_in, W_in, dtype=torch.long, device=DEV)
    gt_master_tokens = torch.empty(N, H_in, W_in, dtype=torch.long, device=DEV)
    gt_pose = torch.empty(N, 6, dtype=torch.float32, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            x = gt_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)
            x_s = x[:, 0:1]
            x_m = x[:, 1:2]
            gt_slave_tokens[s:e] = segnet(segnet.preprocess_input(x_s)).argmax(1)
            gt_master_tokens[s:e] = segnet(segnet.preprocess_input(x_m)).argmax(1)
            gt_pose[s:e] = posenet(posenet.preprocess_input(x))["pose"][..., :6].float()
    del gt_pairs
    torch.cuda.empty_cache()

    print("[sd-ft] precompute master_cam (frozen, SCN on)...", flush=True)
    master_cam = torch.empty(N, 3, CAMERA_H, CAMERA_W, dtype=torch.float16, device="cpu")
    with torch.inference_mode():
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = torch.arange(s, e, device=DEV)
            mst = master(gt_master_tokens[s:e], idx).clamp(0, 255).round()
            # Integers in [0, 255] are exact in FP16
            master_cam[s:e] = mst.half().cpu()

    @torch.no_grad()
    def eval_pose(slave_gen):
        slave_gen.eval()
        pose_sum = 0.0
        pose_n = 0
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = torch.arange(s, e, device=DEV)
            slv = slave_gen(idx).clamp(0, 255).round()
            mst = master_cam[s:e].to(DEV).float()
            pair = torch.stack([slv, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_sum += F.mse_loss(fp, gt_pose[s:e], reduction="sum").item() / 6.0
            pose_n += pair.shape[0]
        return pose_sum / max(pose_n, 1)

    blob = torch.load(args.init, map_location=DEV, weights_only=False)
    channels = tuple(blob.get("channels", (24, 16, 12, 8, 8, 6, 6)))
    d_lat = int(blob.get("d_lat", 6))
    print(f"[sd-ft] slave: channels={channels}  d_lat={d_lat}", flush=True)

    gen = ShrinkSingleNeRV(num_pairs=N, d_lat=d_lat, channels=channels).to(DEV)
    gen.set_qat(True)
    miss, unexp = gen.load_state_dict(blob["state_dict"], strict=False)
    # Prevent maybe_init_step() from overwriting loaded LSQ step
    for m in gen.modules():
        if isinstance(m, (QConv2d, QLinear)) and getattr(m, "quantize_weight", False):
            m._step_initialized = True
    print(f"[sd-ft] slave init: miss={len(miss)} unexp={len(unexp)} "
          f"prev_pose={blob.get('swa_pose', blob.get('best_pose', '?'))}", flush=True)

    if args.freeze_conv:
        for n, p in gen.named_parameters():
            p.requires_grad = ("codes" in n or "per_pair_bias" in n)
        n_trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        print(f"[sd-ft] frozen QAT conv weights; trainable params: {n_trainable:_}", flush=True)

    init_pose = eval_pose(gen)
    print(f"[sd-ft] init pose (DALI refs): {init_pose:.6f}", flush=True)

    code_params, bias_params, other_params = [], [], []
    for n, p in gen.named_parameters():
        if not p.requires_grad:
            continue
        if "codes" in n:
            code_params.append(p)
        elif "per_pair_bias" in n:
            bias_params.append(p)
        else:
            other_params.append(p)
    pg = []
    if other_params:
        pg.append({"params": other_params, "lr": args.lr})
    if code_params:
        pg.append({"params": code_params, "lr": args.lr_codes})
    if bias_params:
        pg.append({"params": bias_params, "lr": args.lr_bias})
    opt = torch.optim.Adam(pg, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_pose = init_pose
    best_ep = -1
    swa_buf = None
    swa_count = 0
    t0 = time.time()
    for ep in range(args.epochs):
        gen.train()
        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = perm[s:e]
            slv = gen(idx)
            slv_q = slv + (slv.round() - slv).detach()
            with torch.no_grad():
                mst = master_cam[idx.cpu()].to(DEV).float()
            pair = torch.stack([slv_q, mst], dim=1)
            ppi = posenet_preprocess_grad(pair)
            fp = posenet(ppi)["pose"][..., :6].float()
            pose_loss = F.mse_loss(fp, gt_pose[idx])
            x = slv_q.unsqueeze(1)
            seg_logits = segnet(segnet.preprocess_input(x))
            ce = F.cross_entropy(seg_logits, gt_slave_tokens[idx])
            loss = args.pose_w * pose_loss + args.seg_stability * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep >= args.swa_from:
            if swa_buf is None:
                swa_buf = {k: v.detach().clone().float() for k, v in gen.state_dict().items()
                           if v.dtype.is_floating_point}
                swa_count = 1
            else:
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point and k in swa_buf:
                        swa_buf[k].mul_(swa_count / (swa_count + 1)).add_(v.detach().float() / (swa_count + 1))
                swa_count += 1

        if (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1:
            pose = eval_pose(gen)
            gen.train()
            improved = pose < best_pose
            if improved:
                best_pose = pose
                best_ep = ep
                torch.save({"state_dict": {k: v.detach().clone().cpu() for k, v in gen.state_dict().items()},
                            "best_pose": best_pose, "channels": channels, "d_lat": d_lat,
                            "kind": f"v62_slave_shrink_{args.tag}_best", "gt_decoder": "dali"},
                           SAVE_DIR / f"slave_v62_shrink_{args.tag}_best.pt")
            tag_s = " *NEW BEST*" if improved else ""
            print(f"[sd-ft] ep {ep+1:>4d}/{args.epochs}  pose={pose:.6f}  "
                  f"best={best_pose:.6f}@ep{best_ep+1}  init={init_pose:.6f}  "
                  f"dt={time.time()-t0:.0f}s{tag_s}", flush=True)

    swa_pose = None
    if swa_buf is not None and swa_count > 1:
        backup = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        swa_state = dict(backup)
        for k, v in swa_buf.items():
            swa_state[k] = v.to(backup[k].device, backup[k].dtype)
        gen.load_state_dict(swa_state)
        gen.eval()
        swa_pose = eval_pose(gen)
        print(f"[sd-ft] SWA-{swa_count} pose: {swa_pose:.6f}", flush=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in swa_state.items()},
                    "swa_pose": swa_pose, "channels": channels, "d_lat": d_lat,
                    "kind": f"v62_slave_shrink_{args.tag}_swa", "gt_decoder": "dali"},
                   SAVE_DIR / f"slave_v62_shrink_{args.tag}_swa.pt")

    print(f"\n[sd-ft] DONE init={init_pose:.6f}  best={best_pose:.6f}@ep{best_ep+1}  "
          f"swa={swa_pose if swa_pose is None else f'{swa_pose:.6f}'}  "
          f"wall={time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="v62 slave codec training (init / ft / dali-ft)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Phase 1+2+3 from-scratch slave training")
    p_init.add_argument("--tag", type=str, required=True, help="output filename tag")
    p_init.add_argument("--channels", type=str, default="64,48,32,24,16,12,12",
                        help="7 comma-separated channel sizes")
    p_init.add_argument("--d-lat", type=int, default=16)
    p_init.add_argument("--smoke", action="store_true",
                        help="short schedule (50/150/300 = 500 ep, ~25 min) instead of full (200/400/800 = 1400 ep)")
    p_init.add_argument("--skip-warmup", action="store_true",
                        help="skip Phase 1 SegNet warmup entirely")
    p_init.set_defaults(func=cmd_init)

    p_ft = sub.add_parser("ft", help="continuation fine-tune (PyAV GT, master baked to FP16)")
    p_ft.add_argument("--init", type=str,
                      default=str(SAVE_DIR / "slave_v62_shrink_ultra_swa.pt"))
    p_ft.add_argument("--tag", type=str, default="ultra_ft")
    p_ft.add_argument("--epochs", type=int, default=200)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--lr-codes", type=float, default=2e-3,
                      help="Higher LR for codes (per-frame embeddings)")
    p_ft.add_argument("--lr-bias", type=float, default=2e-3,
                      help="Higher LR for per_pair_bias")
    p_ft.add_argument("--swa-from", type=int, default=100)
    p_ft.add_argument("--eval-every", type=int, default=10)
    p_ft.add_argument("--seg-stability", type=float, default=SEG_STABILITY_W)
    p_ft.add_argument("--pose-w", type=float, default=30.0)
    p_ft.add_argument("--freeze-conv", action="store_true",
                      help="Freeze QAT conv/linear weights; train only codes+bias")
    p_ft.set_defaults(func=cmd_ft)

    p_dali = sub.add_parser("dali-ft", help="continuation fine-tune (DALI GT, SCN-on master)")
    p_dali.add_argument("--init", type=str,
                        default=str(SAVE_DIR / "slave_v62_shrink_ultra_ft_best.pt"))
    p_dali.add_argument("--master", type=str,
                        default=str(SAVE_DIR / "v62_full_dali_best.pt"))
    p_dali.add_argument("--tag", type=str, default="dali")
    p_dali.add_argument("--epochs", type=int, default=100)
    p_dali.add_argument("--bs", type=int, default=8)
    p_dali.add_argument("--lr", type=float, default=1e-4)
    p_dali.add_argument("--lr-codes", type=float, default=5e-4)
    p_dali.add_argument("--lr-bias", type=float, default=5e-4)
    p_dali.add_argument("--swa-from", type=int, default=50)
    p_dali.add_argument("--eval-every", type=int, default=10)
    p_dali.add_argument("--seg-stability", type=float, default=0.0)
    p_dali.add_argument("--pose-w", type=float, default=1.0)
    p_dali.add_argument("--freeze-conv", action="store_true",
                        help="Freeze QAT conv/linear weights, train only codes+bias")
    p_dali.set_defaults(func=cmd_dali_ft)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
