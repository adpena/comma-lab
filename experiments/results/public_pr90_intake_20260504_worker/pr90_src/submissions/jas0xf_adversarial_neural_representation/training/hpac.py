#!/usr/bin/env python
"""HPAC: hierarchical parallel autoregressive token compressor.

Single-file consolidation of the HPAC entropy model, arithmetic codec, and
training loop.

  * Architecture: patch+group autoregressive over 32x32 patches with stride
    delta=2 scan. Each frame decoded in 94 sequential group-steps (vs
    196,608 for raster AR). SCN-quantized layers learn per-channel bit
    budgets jointly with the model.
  * Codec: group-by-group arithmetic coding via constriction.
  * Training: residual-token objective. Compress
        res[i] = (tok[i] - tok[i-1]) mod 5      (i > 0)
        res[0] = tok[0]                          (no prev frame)
    The residual alphabet is still 5 classes but heavily skewed toward 0
    (most pixels unchanged frame-to-frame on driving video). The model
    still sees raw prev_tokens for spatial-temporal context via conv_past,
    but predicts the RESIDUAL at the current frame.
    Decoder reconstructs:  tok[i] = (res[i] + tok[i-1]) mod 5.

Usage:
    python hpac.py train --save hpac.pt
"""
import sys, time, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import constriction
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import preload_rgb_pairs_av, SEGNET_IN_W, SEGNET_IN_H
from modules import SegNet, segnet_sd_path


DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 600
NUM_CLASSES = 5
SAVE_DIR = Path(__file__).resolve().parent.parent / "training_workspace"


B_INIT = 4.0
E_INIT = -3.0
B_MIN = 0.5
B_MAX = 8.0


def _scn_quantize(w, b, e):
    b_clip = b.clamp(B_MIN, B_MAX)
    shape = [1] * w.ndim
    shape[0] = -1
    bv = b_clip.view(shape)
    ev = e.view(shape)
    scale = torch.pow(2.0, ev)
    max_q = torch.pow(2.0, bv - 1) - 1
    min_q = -torch.pow(2.0, bv - 1)
    q = torch.clamp(w / scale, min_q, max_q)
    q_round = q + (q.round() - q).detach()
    return q_round * scale


class SCNConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in // groups, k, k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.b = nn.Parameter(torch.full((c_out,), B_INIT))
        self.e = nn.Parameter(torch.full((c_out,), E_INIT))
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._scn_on = False
        self._w_per_ch = (c_in // groups) * k * k

    def forward(self, x):
        w = _scn_quantize(self.weight, self.b, self.e) if self._scn_on else self.weight
        return F.conv2d(x, w, self.bias, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def total_bits(self):
        return (self.b.clamp(B_MIN, B_MAX) * self._w_per_ch).sum()


class SCNLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.b = nn.Parameter(torch.full((out_f,), B_INIT))
        self.e = nn.Parameter(torch.full((out_f,), E_INIT))
        self._scn_on = False
        self._w_per_ch = in_f

    def forward(self, x):
        w = _scn_quantize(self.weight, self.b, self.e) if self._scn_on else self.weight
        return F.linear(x, w, self.bias)

    def total_bits(self):
        return (self.b.clamp(B_MIN, B_MAX) * self._w_per_ch).sum()


def patch_group_mask(k, dilation, delta, type_):
    """Return [k, k] bool mask. Center is at index (k-1)//2.

    For kernel offset (dr_idx - center)*dilation, (dc_idx - center)*dilation,
    the input position relative to output is offset (dr*dil, dc*dil).
    Group difference is dc*dil + delta*(dr*dil) = dil * (dc + delta*dr).

    Type-A: mask = 1 iff dc + delta*dr < 0
    Type-B: mask = 1 iff dc + delta*dr <= 0
    """
    mask = torch.zeros(k, k, dtype=torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + delta * dr
            if type_ == 'A':
                if val < 0:
                    mask[dr_idx, dc_idx] = 1.0
            elif type_ == 'B':
                if val <= 0:
                    mask[dr_idx, dc_idx] = 1.0
            else:
                raise ValueError(type_)
    return mask


class SCNMaskedConv2dPG(nn.Module):
    """SCN-quantized masked conv with patch+group causal mask.

    Conv is applied to PATCH-FLATTENED tensor [B*Npatches, C, P, P], so
    cross-patch context is naturally blocked by per-patch batching.
    """
    def __init__(self, c_in, c_out, k, padding=0, dilation=1, groups=1,
                 type_='B', delta=2, bias=True):
        super().__init__()
        assert type_ in ('A', 'B')
        self.weight = nn.Parameter(torch.empty(c_out, c_in // groups, k, k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.b = nn.Parameter(torch.full((c_out,), B_INIT))
        self.e = nn.Parameter(torch.full((c_out,), E_INIT))
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.type_ = type_
        self.delta = delta
        m = patch_group_mask(k, dilation, delta, type_)
        self.register_buffer("mask", m.view(1, 1, k, k), persistent=False)
        self._effective_w_per_ch = float(m.sum().item()) * (c_in // groups)
        self._scn_on = False

    def forward(self, x):
        w = _scn_quantize(self.weight, self.b, self.e) if self._scn_on else self.weight
        w = w * self.mask
        return F.conv2d(x, w, self.bias, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def total_bits(self):
        return (self.b.clamp(B_MIN, B_MAX) * self._effective_w_per_ch).sum()

    def effective_weight_count(self):
        return int(self._effective_w_per_ch * self.weight.shape[0])


class ChannelNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # Norm over channel dim, per spatial position
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)


class CausalSPM(nn.Module):
    """Cross-patch context derived from PREV-FRAME features ONLY.

    Causal by construction: prev_tokens are fully observed at decode time, so
    pool->conv->broadcast on prev features cannot leak current-frame state.
    Adds ~5K raw weights (DW3x3 + PW1x1) -> ~6 KB INT8 archive cost.

    Input:  h_past   [B, ch, H, W]  full-resolution prev-frame features
    Output: spm_full [B, ch, H, W]  per-patch summary, broadcast to pixel grid
    """
    def __init__(self, ch, P=32):
        super().__init__()
        self.P = P
        self.norm = ChannelNorm2d(ch)
        # 3x3 depthwise across the patch grid (NRp x NCp = 12 x 16)
        self.dw = SCNConv2d(ch, ch, k=3, padding=1, groups=ch)
        self.pw = SCNConv2d(ch, ch, k=1)

    def forward(self, h_past):
        B, C, H, W = h_past.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x_p = h_past.view(B, C, NRp, P, NCp, P).mean(dim=(3, 5))  # [B,C,NRp,NCp]
        x_p = self.norm(x_p)
        x_p = self.dw(x_p)
        x_p = F.gelu(x_p)
        x_p = self.pw(x_p)
        x_full = x_p.unsqueeze(3).unsqueeze(5).expand(B, C, NRp, P, NCp, P).contiguous()
        return x_full.view(B, C, NRp * P, NCp * P)


class HPACMini(nn.Module):
    def __init__(self, num_pairs=600, num_classes=5, P=32, delta=2,
                 d_film=32, ch=64, use_spm=False, b_init=None):
        """b_init: optional override for SCN bit-budget initialization.
        If set (e.g. 7.0), all SCN layers get their b initialized higher.
        Higher b -> finer quant grid during training -> lower bpp at no INT8 storage cost.
        Default None keeps the module-level B_INIT (4.0)."""
        super().__init__()
        self.num_classes = num_classes
        self.P = P
        self.delta = delta
        self.ch = ch
        self.use_spm = use_spm
        # Per-frame memory (FiLM)
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        nn.init.normal_(self.frame_embed.weight, std=0.02)
        self.film_gen = SCNLinear(d_film, ch * 2)  # scale + shift
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)
        # Spatial branch (patch+group masked)
        self.conv_a  = SCNMaskedConv2dPG(num_classes + 2, ch, k=7, padding=3,
                                          type_='A', delta=delta)
        self.gn_a    = ChannelNorm2d(ch)
        self.conv_b1 = SCNMaskedConv2dPG(ch, ch, k=5, padding=4, dilation=2,
                                          groups=ch, type_='B', delta=delta)
        self.gn_b1   = ChannelNorm2d(ch)
        self.conv_b2 = SCNMaskedConv2dPG(ch, ch, k=3, padding=4, dilation=4,
                                          groups=ch, type_='B', delta=delta)
        self.gn_b2   = ChannelNorm2d(ch)
        # Temporal branch (NO mask, full prev frame)
        self.conv_past = SCNConv2d(num_classes, ch, k=3, padding=1)
        # Optional cross-patch SPM (uses prev-frame features only - causal)
        self.spm = CausalSPM(ch, P=P) if use_spm else None
        # Output head
        self.head = SCNConv2d(ch, num_classes, k=1, padding=0)
        # Coord cache (per-patch coords)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_P = -1

        if b_init is not None:
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, (SCNConv2d, SCNLinear, SCNMaskedConv2dPG)):
                        m.b.fill_(float(b_init))

    def set_scn(self, on):
        for m in self.modules():
            if isinstance(m, (SCNConv2d, SCNLinear, SCNMaskedConv2dPG)):
                m._scn_on = on

    def _patch_coord_grid(self, B, device):
        """Return per-patch coord grid [B, 2, P, P] (within-patch normalized coords)."""
        if self._cached_P != self.P or self._coord_cache.numel() == 0:
            P = self.P
            ys = torch.linspace(-1.0, 1.0, P, device=device).view(1, 1, P, 1).expand(1, 1, P, P)
            xs = torch.linspace(-1.0, 1.0, P, device=device).view(1, 1, 1, P).expand(1, 1, P, P)
            grid = torch.cat([ys, xs], dim=1)  # [1, 2, P, P]
            self._coord_cache = grid
            self._cached_P = self.P
        return self._coord_cache.expand(B, -1, -1, -1)

    def _to_patches(self, x):
        """[B, C, H, W] -> [B*NRp*NCp, C, P, P]."""
        B, C, H, W = x.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x = x.view(B, C, NRp, P, NCp, P).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B * NRp * NCp, C, P, P)

    def _from_patches(self, x_p, B, NRp, NCp):
        """[B*NRp*NCp, C, P, P] -> [B, C, H, W]."""
        P = self.P
        C = x_p.shape[1]
        x_p = x_p.view(B, NRp, NCp, C, P, P).permute(0, 3, 1, 4, 2, 5).contiguous()
        return x_p.view(B, C, NRp * P, NCp * P)

    def forward(self, tokens, idx, prev_tokens):
        """tokens: [B, H, W] long, idx: [B] long, prev_tokens: [B, H, W] long."""
        B, H, W = tokens.shape
        P = self.P
        NRp, NCp = H // P, W // P
        Np = NRp * NCp

        # Spatial input: one_hot + per-patch coord grid
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x_p = self._to_patches(x)
        coord_p = self._patch_coord_grid(B * Np, x.device)
        x_in_p = torch.cat([x_p, coord_p], dim=1)  # [B*Np, 7, P, P]

        # Spatial first layer (Type-A)
        h_p = self.gn_a(self.conv_a(x_in_p))

        # FiLM modulation (broadcast per patch)
        emb = self.frame_embed(idx)  # [B, d_film]
        film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)  # [B, ch]
        scale_p = scale.view(B, 1, self.ch, 1, 1).expand(B, Np, self.ch, 1, 1).reshape(B * Np, self.ch, 1, 1)
        shift_p = shift.view(B, 1, self.ch, 1, 1).expand(B, Np, self.ch, 1, 1).reshape(B * Np, self.ch, 1, 1)
        h_p = h_p * (1.0 + scale_p) + shift_p
        h_p = F.gelu(h_p)

        # Temporal branch: full-frame conv on prev tokens, then patch-ify
        x_prev = F.one_hot(prev_tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        h_past_full = self.conv_past(x_prev)  # [B, ch, H, W]
        h_past_p = self._to_patches(h_past_full)  # [B*Np, ch, P, P]
        h_p = h_p + h_past_p

        # Optional cross-patch SPM (uses prev-frame features only - causal)
        if self.spm is not None:
            h_p = h_p + self._to_patches(self.spm(h_past_full))

        # Continue spatial branch (Type-B)
        h_p = F.gelu(self.gn_b1(self.conv_b1(h_p)))
        h_p = F.gelu(self.gn_b2(self.conv_b2(h_p)))
        logits_p = self.head(h_p)  # [B*Np, NUM_CLASSES, P, P]

        logits = self._from_patches(logits_p, B, NRp, NCp)  # [B, NUM_CLASSES, H, W]
        return logits

    def scn_total_bits(self):
        total = 0
        for m in self.modules():
            if isinstance(m, (SCNConv2d, SCNLinear, SCNMaskedConv2dPG)):
                total = total + m.total_bits()
        return total

    def effective_weight_count(self):
        n = 0
        for m in self.modules():
            if isinstance(m, (SCNConv2d, SCNLinear)):
                n += m.weight.numel()
            elif isinstance(m, SCNMaskedConv2dPG):
                n += m.effective_weight_count()
        return n


@torch.no_grad()
def causality_check(gen, H=64, W=64, n_classes=5, P=32, delta=2):
    """Verify: changing input at group s should NOT affect logits at groups < s.

    Per-patch check: for a single patch (P x P), perturb input at (r0, c0)
    [group s0 = c0 + delta * r0] and verify that logits at any (r, c) with
    group s = c + delta * r < s0 are UNCHANGED.

    Returns True if causal, False otherwise.
    """
    dev = next(gen.parameters()).device
    gen.eval()
    B = 1
    H = P
    W = P
    tokens = torch.zeros(B, H, W, dtype=torch.long, device=dev)
    prev = torch.zeros_like(tokens)
    idx = torch.zeros(B, dtype=torch.long, device=dev)
    base = gen(tokens, idx, prev).clone()
    r0, c0 = P // 2, P // 2
    s0 = c0 + delta * r0
    tokens2 = tokens.clone()
    tokens2[0, r0, c0] = (n_classes - 1)  # max class
    out = gen(tokens2, idx, prev)
    diff = (out - base).abs()  # [B, NC, H, W]
    diff_per_pos = diff.amax(dim=1)[0]  # [H, W]
    changed = diff_per_pos > 1e-5
    rs = torch.arange(P, device=dev).view(P, 1).expand(P, P)
    cs = torch.arange(P, device=dev).view(1, P).expand(P, P)
    s_grid = cs + delta * rs
    leaky_positions = changed & (s_grid < s0)
    n_leaks = int(leaky_positions.sum().item())
    if n_leaks == 0:
        print(f"[hpac causality] OK (perturbed s={s0} at ({r0},{c0}); all changes at s>=s0)")
        return True
    else:
        idxs = torch.nonzero(leaky_positions, as_tuple=False)[:5]
        print(f"[hpac causality] LEAK: {n_leaks} positions changed at s<{s0}: {idxs.tolist()}")
        return False


def _patch_group_grid(P, delta, device):
    """[P, P] tensor of group indices s = c + delta * r."""
    rs = torch.arange(P, device=device).view(P, 1).expand(P, P)
    cs = torch.arange(P, device=device).view(1, P).expand(P, P)
    return cs + delta * rs


def _full_mask_for_group(s_grid, s, NRp, NCp):
    """[H, W] bool mask: True at every position whose intra-patch group == s."""
    P = s_grid.shape[0]
    mask_p = (s_grid == s)  # [P, P]
    full = mask_p.unsqueeze(0).unsqueeze(0).expand(NRp, NCp, P, P)
    return full.permute(0, 2, 1, 3).reshape(NRp * P, NCp * P)


@torch.no_grad()
def encode_frame(gen, gt_tokens, idx, prev_tokens, encoder, P=32, delta=2, prob_eps=1e-7):
    """Encode one frame group-by-group. Updates encoder in-place.

    gt_tokens: [1, H, W] long
    idx: [1] long
    prev_tokens: [1, H, W] long
    encoder: constriction RangeEncoder
    """
    dev = gt_tokens.device
    H, W = gt_tokens.shape[-2:]
    NRp, NCp = H // P, W // P
    s_grid = _patch_group_grid(P, delta, dev)
    n_groups = int((1 + delta) * P - delta)

    current = torch.zeros_like(gt_tokens)
    n_total = 0
    for s in range(n_groups):
        full_mask = _full_mask_for_group(s_grid, s, NRp, NCp)  # [H, W]
        n_pos = int(full_mask.sum().item())
        if n_pos == 0:
            continue
        logits = gen(current, idx, prev_tokens)  # [1, NC, H, W]
        probs = F.softmax(logits.float(), dim=1)
        probs_at_s = probs[0][:, full_mask].permute(1, 0).contiguous()  # [n_pos, NC]
        gt_at_s = gt_tokens[0][full_mask].cpu().numpy().astype(np.int32)  # [n_pos]
        probs_np = probs_at_s.cpu().numpy().astype(np.float64)
        probs_np = np.clip(probs_np, prob_eps, 1.0)
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        for i in range(n_pos):
            cat = constriction.stream.model.Categorical(probabilities=probs_np[i], perfect=False)
            encoder.encode(int(gt_at_s[i]), cat)
        # Update state with GT values
        current[0, full_mask] = gt_tokens[0, full_mask]
        n_total += n_pos
    return n_total


@torch.no_grad()
def decode_frame(gen, decoder, idx, prev_tokens, H, W, P=32, delta=2, prob_eps=1e-7):
    """Decode one frame group-by-group from `decoder`. Returns [1, H, W] long.
    Mirror of encode_frame state machine.
    """
    dev = prev_tokens.device
    NRp, NCp = H // P, W // P
    s_grid = _patch_group_grid(P, delta, dev)
    n_groups = int((1 + delta) * P - delta)

    current = torch.zeros((1, H, W), dtype=torch.long, device=dev)
    for s in range(n_groups):
        full_mask = _full_mask_for_group(s_grid, s, NRp, NCp)  # [H, W]
        n_pos = int(full_mask.sum().item())
        if n_pos == 0:
            continue
        logits = gen(current, idx, prev_tokens)
        probs = F.softmax(logits.float(), dim=1)
        probs_at_s = probs[0][:, full_mask].permute(1, 0).contiguous()  # [n_pos, NC]
        probs_np = probs_at_s.cpu().numpy().astype(np.float64)
        probs_np = np.clip(probs_np, prob_eps, 1.0)
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        decoded = np.empty(n_pos, dtype=np.int64)
        for i in range(n_pos):
            cat = constriction.stream.model.Categorical(probabilities=probs_np[i], perfect=False)
            decoded[i] = decoder.decode(cat)
        current[0, full_mask] = torch.from_numpy(decoded).to(dev)
    return current


def compute_residuals(gt_tokens):
    """gt_tokens: [N, H, W] long. Returns gt_residuals same shape.
    res[0] = tok[0]; res[i] = (tok[i] - tok[i-1]) % NUM_CLASSES
    """
    res = torch.empty_like(gt_tokens)
    res[0] = gt_tokens[0]
    res[1:] = (gt_tokens[1:] - gt_tokens[:-1]) % NUM_CLASSES
    return res


def cmd_train(args):
    if args.smoke:
        args.epochs = 3
        args.scn_from = 999
        print("[hpacR] SMOKE mode", flush=True)

    print(f"[hpacR] N={N} BS={args.bs} epochs={args.epochs} ch={args.ch} P={args.P} delta={args.delta}", flush=True)
    gen = HPACMini(num_pairs=N, num_classes=NUM_CLASSES, P=args.P, delta=args.delta, ch=args.ch).to(DEV)
    n_w = gen.effective_weight_count()
    n_total = sum(p.numel() for p in gen.parameters())
    print(f"[hpacR] eff_SCN={n_w:_} total={n_total:_}", flush=True)

    ok = causality_check(gen, P=args.P, delta=args.delta)
    if not ok:
        print("[hpacR] ABORT: causality leak"); return

    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    for p in segnet.parameters():
        p.requires_grad = False

    print("[hpacR] preload + gt_tokens...", flush=True)
    rgb_pairs = preload_rgb_pairs_av(ROOT / "videos", ROOT / "public_test_video_names.txt")
    gt_tokens = torch.empty(N, SEGNET_IN_H, SEGNET_IN_W, dtype=torch.long, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            x = rgb_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)
            gt_tokens[s:e] = segnet(segnet.preprocess_input(x[:, 1:2])).argmax(1)
    del rgb_pairs

    gt_residuals = compute_residuals(gt_tokens)
    counts = torch.bincount(gt_residuals.flatten(), minlength=NUM_CLASSES)
    fracs = counts.float() / counts.sum().float()
    print(f"[hpacR] residual class fractions: " +
          " ".join(f"{i}={fracs[i].item():.4f}" for i in range(NUM_CLASSES)), flush=True)

    # prev_tokens stays as RAW tokens (not residuals) for the conv_past temporal branch
    prev_tokens_all = torch.zeros_like(gt_tokens)
    prev_tokens_all[1:] = gt_tokens[:-1]

    H, W = gt_tokens.shape[1:]
    n_pixels = N * H * W

    @torch.no_grad()
    def eval_bpp():
        gen.eval()
        total_nats = 0.0
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = torch.arange(s, e, device=DEV)
            res = gt_residuals[s:e]   # input/target is residual
            prev = prev_tokens_all[s:e]  # raw prev tokens for conv_past context
            logits = gen(res, idx, prev)
            ce = F.cross_entropy(logits, res, reduction='sum')
            total_nats += float(ce)
        bits = total_nats / 0.6931471805599453
        return bits / n_pixels

    init_bpp = eval_bpp()
    print(f"[hpacR] init bpp={init_bpp:.4f}  est_KB={init_bpp*n_pixels/8/1024:.1f}", flush=True)

    frame_emb_params = [gen.frame_embed.weight]
    other_params = [p for n, p in gen.named_parameters() if n != "frame_embed.weight"]
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr},
         {"params": frame_emb_params, "lr": args.lr_emb}],
        eps=1e-8,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    if args.lam_init > 0 and args.lam_final > 0:
        log_step = (math.log(args.lam_final) - math.log(args.lam_init)) / max(args.epochs - args.scn_from, 1)
    else:
        log_step = 0.0

    best_bpp = init_bpp
    t0 = time.time()
    for ep in range(args.epochs):
        gen.train()
        if ep == args.scn_from:
            print(f"[hpacR] enabling SCN at ep {ep}", flush=True)
            gen.set_scn(True)
        if ep < args.scn_from:
            lam_bits = 0.0
        else:
            lam_bits = args.lam_init * math.exp(log_step * (ep - args.scn_from))
        perm = torch.randperm(N, device=DEV)
        ep_ce = 0.0; ep_count = 0
        for s in range(0, N, args.bs):
            e = min(s + args.bs, N)
            idx = perm[s:e]
            res = gt_residuals[idx]
            prev = prev_tokens_all[idx]
            logits = gen(res, idx, prev)
            ce = F.cross_entropy(logits, res)
            loss = ce
            if lam_bits > 0:
                bits_loss = lam_bits * gen.scn_total_bits() / max(n_w, 1)
                loss = loss + bits_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            ep_ce += float(ce) * res.shape[0]
            ep_count += res.shape[0]
        sched.step()

        if (ep + 1) % 10 == 0 or ep == 0 or ep == args.scn_from:
            bpp = eval_bpp()
            if bpp < best_bpp:
                best_bpp = bpp
            train_bpp = (ep_ce / max(ep_count, 1)) / 0.6931471805599453
            kb = bpp * n_pixels / 8 / 1024
            sb = float(gen.scn_total_bits()) / 8
            print(f"[hpacR] ep {ep+1:>4d}/{args.epochs}  "
                  f"train_bpp={train_bpp:.4f}  eval_bpp={bpp:.4f}  best={best_bpp:.4f}  "
                  f"est_KB={kb:.1f}  scn_KB={sb/1024:.2f}  "
                  f"lam={lam_bits:.1e}  dt={time.time()-t0:.0f}s",
                  flush=True)

    gen.eval()
    final_bpp = eval_bpp()
    final_kb = final_bpp * n_pixels / 8 / 1024
    scn_bytes = float(gen.scn_total_bits()) / 8
    print(f"\n[hpacR] FINAL bpp={final_bpp:.4f} best={best_bpp:.4f}  est_KB={final_kb:.1f}  scn_KB={scn_bytes/1024:.2f}",
          flush=True)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in gen.state_dict().items()},
        "best_bpp": best_bpp, "final_bpp": final_bpp,
        "P": args.P, "delta": args.delta, "ch": args.ch,
        "kind": "hpac_residual",
    }, save_path)
    print(f"[hpacR] saved {save_path.name}  wall={time.time()-t0:.0f}s", flush=True)


def _build_train_parser(sub):
    ap = sub.add_parser("train", help="Train HPAC on residual tokens.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-emb", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--scn-from", type=int, default=50)
    ap.add_argument("--lam-init", type=float, default=1e-5)
    ap.add_argument("--lam-final", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default=str(SAVE_DIR / "hpac_residual.pt"))
    ap.add_argument("--smoke", action="store_true")
    ap.set_defaults(func=cmd_train)
    return ap


def main():
    parser = argparse.ArgumentParser(description="HPAC: entropy model + arithmetic codec + training")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _build_train_parser(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
