"""Self-contained H3 generator. Inlined from autoresearch/{train,prepare}.py
so the submission has no external dependencies beyond comma's own modules.

Exposes: Generator, GeneratorPoseLR, apply_fp4_to_model,
         MODEL_H, MODEL_W, OUT_H, OUT_W, COND_DIM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants (from autoresearch/prepare.py + train.py) ─────────────────
MODEL_H, MODEL_W = 384, 512        # SegNet model input size (H, W)
OUT_H, OUT_W = 874, 1164           # camera_size flipped (H, W)
EMB_DIM       = 6
COND_DIM      = 64
HEAD_HIDDEN   = 52
C1            = 56
C2            = 64
DM            = 1


# ── FP4 quantization helpers ────────────────────────────────────────────
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def fp4_round_trip(x, block_size=32):
    orig = x.shape
    flat = x.reshape(-1)
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    ma = blocks.abs().amax(1, keepdim=True)
    sc = torch.where(ma > 0, ma / 6.0, torch.ones_like(ma))
    norm = blocks / sc
    signs = norm < 0
    lvl = _FP4_LEVELS.to(x.device, x.dtype).view(1, 1, -1)
    mag = (norm.abs().unsqueeze(-1) - lvl).abs().argmin(-1)
    q = torch.where(signs, -lvl[0, 0, mag], lvl[0, 0, mag])
    return (q * sc).view(-1)[:x.numel()].view(orig)


def apply_fp4_to_model(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and getattr(m, 'quantize_weight', True):
                m.weight.data = fp4_round_trip(m.weight.data)
            elif isinstance(m, nn.Embedding) and getattr(m, 'quantize_weight', True):
                m.weight.data = fp4_round_trip(m.weight.data)


# ── Quantizable layer wrappers ──────────────────────────────────────────
class QConv2d(nn.Conv2d):
    def __init__(self, *a, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.quantize_weight = quantize_weight
        self.qat = False
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QEmb(nn.Embedding):
    def __init__(self, *a, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.quantize_weight = quantize_weight
        self.qat = False
    def forward(self, x):
        return F.embedding(x, self.weight, self.padding_idx)


class QLinear(nn.Module):
    """Linear via internal 1x1 QConv2d so weights get FP4 byte treatment."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.conv = QConv2d(in_features, out_features, 1, bias=bias)
    @property
    def weight(self): return self.conv.weight
    @property
    def bias(self):   return self.conv.bias
    def forward(self, x):
        orig = x.shape
        x = x.reshape(-1, orig[-1], 1, 1)
        x = self.conv(x)
        return x.view(*orig[:-1], -1)


class LowRankLinear(nn.Module):
    """rank-r factorization B(A(x)). SVD-warm-started in the trained checkpoint."""
    def __init__(self, in_f, out_f, rank, bias=True):
        super().__init__()
        self.a = nn.Linear(in_f, rank, bias=False)
        self.b = nn.Linear(rank, out_f, bias=bias)
    def forward(self, x):
        return self.b(self.a(x))


# ── Building blocks ─────────────────────────────────────────────────────
class DSConv(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, act=True):
        super().__init__()
        mid = ic * DM
        self.dw = QConv2d(ic, mid, k, stride=s, padding=k//2, groups=ic, bias=False)
        self.pw = QConv2d(mid, oc, 1, bias=True)
        self.norm = nn.GroupNorm(min(2, oc), oc)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class Res(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = DSConv(ch, ch)
        mid = ch * DM
        self.dw2 = QConv2d(ch, mid, 3, padding=1, groups=ch, bias=False)
        self.pw2 = QConv2d(mid, ch, 1, bias=True)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.norm(self.pw2(self.dw2(self.c1(x)))))


class FiLMRes(nn.Module):
    def __init__(self, ch, cd):
        super().__init__()
        self.c1 = DSConv(ch, ch)
        mid = ch * DM
        self.dw2 = QConv2d(ch, mid, 3, padding=1, groups=ch, bias=False)
        self.pw2 = QConv2d(mid, ch, 1, bias=True)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.film = QLinear(cd, ch * 2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        r = self.norm(self.pw2(self.dw2(self.c1(x))))
        g, b = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        return self.act(x + r * (1 + g) + b)


def _coords(B, H, W, dev):
    ys = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx*2-1, yy*2-1], 0).unsqueeze(0).expand(B, -1, -1, -1)


# ── Heads + trunk + Generator ───────────────────────────────────────────
class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = QEmb(5, EMB_DIM, quantize_weight=False)
        self.stem = DSConv(EMB_DIM + 2, C1)
        self.s1 = Res(C1)
        self.down = DSConv(C1, C2, s=2)
        self.d1 = Res(C2)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DSConv(C2, C1),
        )
        self.fuse = DSConv(C1 * 2, C1)
        self.f1 = Res(C1)
    def forward(self, mask, co):
        e = F.interpolate(self.emb(mask.long()).permute(0,3,1,2), co.shape[-2:], mode="bilinear", align_corners=False)
        s = self.s1(self.stem(torch.cat([e, co], 1)))
        z = self.up(self.d1(self.down(s)))
        return self.f1(self.fuse(torch.cat([z, s], 1)))


class Head2(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = Res(C1)
        self.pre = DSConv(C1, HEAD_HIDDEN)
        self.out = QConv2d(HEAD_HIDDEN, 3, 1, quantize_weight=False)
    def forward(self, f):
        return torch.sigmoid(self.out(self.pre(self.r1(f)))) * 255.0


class Head1(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = FiLMRes(C1, COND_DIM)
        self.r2 = FiLMRes(C1, COND_DIM)
        self.out = QConv2d(C1, 3, 1, quantize_weight=False)
    def forward(self, f, c):
        return torch.sigmoid(self.out(self.r2(self.r1(f, c), c))) * 255.0


class Generator(nn.Module):
    """H3 generator with dense (FP16) pose_mlp. Used in the autoresearch loop."""
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.pose_mlp = nn.Sequential(
            nn.Linear(6, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM),
        )
        self.h1 = Head1()
        self.h2 = Head2()
    def forward(self, mask, pose):
        co = _coords(mask.shape[0], MODEL_H, MODEL_W, mask.device)
        feat = self.trunk(mask, co)
        cond = self.pose_mlp(pose)
        return self.h1(feat, cond), self.h2(feat)


class GeneratorPoseLR(Generator):
    """H3 + LowRank pose_mlp at rank=16 (SVD warm-started). The shipping model."""
    def __init__(self, rank=16):
        super().__init__()
        self.pose_mlp = nn.Sequential(
            nn.Linear(6, COND_DIM), nn.SiLU(),
            LowRankLinear(COND_DIM, COND_DIM, rank, bias=True), nn.SiLU(),
            LowRankLinear(COND_DIM, COND_DIM, rank, bias=True),
        )
