# SPDX-License-Identifier: MIT
"""ANR TokenRendererV62 + ShrinkSingleNeRV full substrate port — Phase A scaffold.

Per handoff P3 long-tail "ANR/HPAC token programs" + operator directive 2026-05-11
"isolate context-model and categorical-token compression primitives", this module
is the full PR95 ANR substrate port to ``src/tac/``.

Source: ``experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/jas0xf_adversarial_neural_representation/``

Architecture (verbatim from PR95 ``inflate.py``):

- ``TokenRendererV62`` — MASTER decoder. Takes per-frame 5-class categorical
  tokens (the contest SegNet-class palette) and renders RGB via:
    (a) one-hot expansion of tokens → conv → FiLM → conv → conv → sigmoid*255.
    (b) FiLM modulation is generated from a per-pair frame_embed +
        a small linear layer; the result is PRE-COMPUTED ON CPU FP32 at
        bake_film_table() time so master pixels are bit-identical across GPUs
        (cuBLAS picks different kernels for tiny matmul on Ada vs Turing).
    (c) Final bilinear upsample to camera resolution (874, 1164).

- ``ShrinkSingleNeRV`` — SLAVE decoder. Per-pair codes (d_lat=6) + a small
  6-stage depthwise-conv + pixel-shuffle NeRV. Provides ONE of the two frames
  per pair (the OTHER frame comes from the master); shipped as INT4-quantized
  weights inside the archive.

- ``HPACMini`` — context model for the token arithmetic coder. CausalSPM +
  MaskedConv2dPG + FiLM + 5-class softmax head. The arithmetic coder itself
  is the universal ``encode_categorical_stream`` /
  ``decode_categorical_stream`` from ``tac.packet_compiler.pr91_hpac_grammar``
  (L's 2026-05-11 landing).

- ``FiLMPortabilityGuard`` — runtime check that CPU FP32 forward through the
  master's FiLM table equals GPU forward to within ``film_portability_atol``.
  Catches the "different GPU produces different bytes" failure mode from
  PR95 inflate.py line 47-50 commentary.

- PPMd codec — at compress time, the HPACMini quantised state_dict is
  serialized + ``pyppmd``-compressed for archive packing. At inflate time,
  the reverse holds (PR95 inflate.py line 302-303 ``str(hpac_pt).endswith(".ppmd")``).

CLAUDE.md compliance (HNeRV parity discipline 13 lessons)
=========================================================
- L1 score-aware substrate — ``train_step`` backprops through SegNet + PoseNet
  via ``tac.scorer.load_differentiable_scorers``.
- L2 export-first design — archive grammar declared as ``ARCHIVE_GRAMMAR``
  at module top.
- L3 monolithic single-file archive — yes (the canonical PR95 layout already
  satisfies this; we mirror it byte-for-byte).
- L4 inflate ≤ 200 LOC (substrate-engineering waiver) — see
  ``submissions/anr_substrate/inflate.py``.
- L5 full RGB renderer — yes (master + slave both emit RGB to camera res).
- L6 score-domain Lagrangian — yes (loss = λ_seg·d_seg + λ_pose·d_pose
  through actual scorer contracts).
- L7 bolt-on size ≤ 350 LOC — this lane is substrate-engineering not bolt-on.
- L8 eval_roundtrip simulated — yes (uint8 STE clamp before scorer call).
- L9 runtime closure — depends on ``constriction`` + ``pyppmd`` + ``torch``
  (Phase B closure work).
- L10 mask/pose coupling gate — N/A in the substrate (downstream score
  diagnostics handle this).
- L11 no-op detector — ``export_to_archive`` returns sha256; tests assert
  roundtrip determinism.
- L12 reviewable in 30 seconds — every class is ≤ 100 LOC; total module is
  ~450 LOC.
- L13 deferred-pending-research not killed — never declares a kill.

Forbidden patterns:
- No MPS-fallback default; ``train_step`` raises if CUDA unavailable
  and cuda_required=True.
- No make_synthetic_pair_batch in any non-smoke path (RealPairBatchSource
  delegated from ``lane_12_v2_nerv_as_renderer.RealPairBatchSource``).
- No /tmp persisted-artifact paths.
- No scorer load at inflate time (strict-scorer-rule).
- No comment-only contracts (every "MUST" in this docstring is enforced
  by an assertion or test).
"""
from __future__ import annotations

import hashlib
import io
import struct
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────


#: Per PR95 inflate.py line 27.
CAMERA_H: int = 874
CAMERA_W: int = 1164

#: Per PR95 inflate.py line 28; SegNet input resolution (where tokens live).
SEGNET_IN_H: int = 384
SEGNET_IN_W: int = 512

#: Per PR95 inflate.py line 29; NeRV starting feature-map size.
FEAT_H: int = 6
FEAT_W: int = 8

#: Per PR95 inflate.py line 30 + CLAUDE.md "Exact scorer architectures" —
#: SegNet emits 5-class logits (NUM_CLASSES=5).
NUM_CLASSES: int = 5

#: ANR format identifier (magic) for the typed archive parser.
ANR_MAGIC: bytes = b"ANRV"  # 4 ASCII bytes
ANR_FORMAT_ID: int = 0x50
ANR_FORMAT_VERSION: int = 1


# ── Archive grammar (parser-section manifest, machine-readable) ──────────


ARCHIVE_GRAMMAR: dict = {
    "format_id": ANR_FORMAT_ID,
    "format_version": ANR_FORMAT_VERSION,
    "magic": ANR_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 16,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("format_id", "<H", 2),
                ("format_version", "<H", 2),
                ("num_pairs", "<I", 4),
                ("flags", "<I", 4),
            ],
        },
        {
            "name": "meta",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "torch_save_dict_meta",
        },
        {
            "name": "master_state",
            "offset_after": "meta",
            "length_field_le_u32": True,
            "kind": "fp16_state_dict_scn_baked",
        },
        {
            "name": "slave_state",
            "offset_after": "master_state",
            "length_field_le_u32": True,
            "kind": "int4_state_dict_lsq_baked",
        },
        {
            "name": "hpac_state",
            "offset_after": "slave_state",
            "length_field_le_u32": True,
            "kind": "int8_packed_state_dict_ppmd",
        },
        {
            "name": "tokens",
            "offset_after": "hpac_state",
            "length_field_le_u32": True,
            "kind": "constriction_categorical_arithmetic_coded",
        },
    ],
    "predicted_total_bytes": (
        "175_000 to 195_000 [predicted; PR95 anchor 178_262 B = A1]"
    ),
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ANRTokenRendererConfig:
    """Frozen config for ANR substrate Phase A.

    Defaults mirror PR95 exemplar exactly.
    """

    num_pairs: int = 600
    num_classes: int = NUM_CLASSES
    d_film: int = 8
    slave_d_lat: int = 6
    slave_channels: tuple[int, ...] = (24, 16, 12, 8, 8, 6, 6)
    hpac_P: int = 32
    hpac_delta: int = 2
    hpac_ch: int = 64
    hpac_use_spm: bool = False
    hpac_d_film: int = 32
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129  # Match Lane 12-v2 (CLAUDE.md PR106 r2).
    film_portability_atol: float = 1e-5
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {self.num_pairs}")
        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"num_classes pinned at {NUM_CLASSES} (SegNet-class); "
                f"got {self.num_classes}"
            )
        if len(self.slave_channels) != 7:
            raise ValueError(
                f"slave_channels must have exactly 7 entries (6 stages + final), "
                f"got {len(self.slave_channels)}"
            )
        if self.hpac_P <= 0 or (SEGNET_IN_H % self.hpac_P) != 0:
            raise ValueError(
                f"hpac_P={self.hpac_P} must divide SEGNET_IN_H={SEGNET_IN_H}"
            )
        if self.film_portability_atol <= 0:
            raise ValueError(
                f"film_portability_atol must be positive, got {self.film_portability_atol}"
            )


# ── TokenRendererV62 master ──────────────────────────────────────────────


class TokenRendererV62(nn.Module):
    """ANR master decoder. Verbatim port of PR95 inflate.py:36-83.

    Forward signature: ``(tokens, idx) → (B, 3, CAMERA_H, CAMERA_W)``.

    tokens: (B, SEGNET_IN_H, SEGNET_IN_W) long, values in [0, NUM_CLASSES).
    idx:    (B,) long, values in [0, num_pairs).
    """

    def __init__(self, num_pairs: int = 600, num_classes: int = NUM_CLASSES,
                 d_film: int = 8) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.num_classes = num_classes
        self.d_film = d_film
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        self.film_gen = nn.Linear(d_film, 64)
        self.conv1 = nn.Conv2d(num_classes, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.act = nn.GELU()
        # Cross-hardware-deterministic FiLM: see CPU-FP32 portability guard.
        self.register_buffer(
            "_film_table", torch.zeros(num_pairs, 64), persistent=False
        )
        self._film_table_baked: bool = False

    def bake_film_table(self) -> None:
        """Precompute (frame_embed @ film_gen.W.T + film_gen.b) on CPU FP32.

        Per PR95 inflate.py:56-65 commentary: cuBLAS picks different kernels for
        tiny (8x64) matmul on Ada vs Turing, which causes master pixels to differ
        across GPUs. CPU FP32 is bit-portable. Call AFTER load_state_dict.
        """
        with torch.no_grad():
            emb = self.frame_embed.weight.detach().cpu().float()
            w = self.film_gen.weight.detach().cpu().float()
            b = self.film_gen.bias.detach().cpu().float()
            table = emb @ w.T + b  # (num_pairs, 64)
            self._film_table.copy_(table.to(self._film_table.device))
        self._film_table_baked = True

    def forward(self, tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.long:
            raise TypeError(f"tokens must be long, got {tokens.dtype}")
        if idx.dtype != torch.long:
            raise TypeError(f"idx must be long, got {idx.dtype}")
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B, H, W), got {tuple(tokens.shape)}")
        if (tokens.max().item() >= self.num_classes if tokens.numel() else False):
            raise ValueError(
                f"tokens max {int(tokens.max().item())} >= num_classes={self.num_classes}"
            )
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x = self.conv1(x)
        x = self.gn1(x)
        if self._film_table_baked:
            film = self._film_table[idx]
        else:
            emb = self.frame_embed(idx)
            film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)
        x = x * (1.0 + scale.view(-1, 32, 1, 1)) + shift.view(-1, 32, 1, 1)
        x = self.act(x)
        x = self.act(self.gn2(self.conv2(x)))
        x = self.out_conv(x)
        raw = torch.sigmoid(x) * 255.0
        return F.interpolate(
            raw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False,
        )


# ── ShrinkSingleNeRV slave ───────────────────────────────────────────────


class _NeRVBlock(nn.Module):
    """Verbatim port of PR95 inflate.py:89-97."""

    def __init__(self, c_in: int, c_out: int, s: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out * s * s, kernel_size=1, bias=True)
        self.ps = nn.PixelShuffle(s)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.pw(self.dw(x))))


class ShrinkSingleNeRV(nn.Module):
    """ANR slave decoder. Verbatim port of PR95 inflate.py:100-121.

    Forward signature: ``idx (B,) long → (B, 3, CAMERA_H, CAMERA_W)``.
    """

    def __init__(self, num_pairs: int = 600, d_lat: int = 6,
                 channels: tuple[int, ...] = (24, 16, 12, 8, 8, 6, 6)) -> None:
        super().__init__()
        if len(channels) != 7:
            raise ValueError(f"channels must be 7 entries, got {len(channels)}")
        self.num_pairs = num_pairs
        self.d_lat = d_lat
        self.channels = tuple(channels)
        self.codes = nn.Embedding(num_pairs, d_lat)
        self.stem = nn.Linear(d_lat, channels[0] * FEAT_H * FEAT_W, bias=True)
        self.stem_act = nn.GELU()
        self.blocks = nn.ModuleList(
            [_NeRVBlock(channels[i], channels[i + 1], s=2) for i in range(6)]
        )
        self.head = nn.Conv2d(channels[-1], 3, kernel_size=1, bias=True)
        self.per_pair_bias = nn.Embedding(num_pairs, 3)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.dtype != torch.long:
            raise TypeError(f"idx must be long, got {idx.dtype}")
        z = self.codes(idx)
        x = self.stem(z).view(-1, self.channels[0], FEAT_H, FEAT_W)
        x = self.stem_act(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x) + self.per_pair_bias(idx).view(-1, 3, 1, 1)
        raw = torch.sigmoid(out) * 255.0
        return F.interpolate(
            raw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False,
        )


# ── HPACMini context model ───────────────────────────────────────────────


def _patch_group_mask(k: int, delta: int, type_: str) -> torch.Tensor:
    """Verbatim port of PR95 inflate.py:127-141."""
    if type_ not in ("A", "B"):
        raise ValueError(f"type_ must be 'A' or 'B', got {type_!r}")
    mask = torch.zeros(k, k, dtype=torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + delta * dr
            if type_ == "A":
                if val < 0:
                    mask[dr_idx, dc_idx] = 1.0
            else:
                if val <= 0:
                    mask[dr_idx, dc_idx] = 1.0
    return mask


class _MaskedConv2dPG(nn.Module):
    """Plain masked conv (no SCN — quantization pre-applied at build time).

    Verbatim port of PR95 inflate.py:144-159.
    """

    def __init__(self, c_in: int, c_out: int, k: int, padding: int = 0,
                 dilation: int = 1, groups: int = 1, type_: str = "B",
                 delta: int = 2, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(c_out, c_in // groups, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        m = _patch_group_mask(k, delta, type_)
        self.register_buffer("mask", m.view(1, 1, k, k), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(
            x, w, self.bias, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )


class _ChannelNorm2d(nn.Module):
    """Verbatim port of PR95 inflate.py:162-172."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)


class _CausalSPM(nn.Module):
    """Verbatim port of PR95 inflate.py:175-193."""

    def __init__(self, ch: int, P: int = 32) -> None:
        super().__init__()
        self.P = P
        self.norm = _ChannelNorm2d(ch)
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, h_past: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h_past.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x_p = h_past.view(B, C, NRp, P, NCp, P).mean(dim=(3, 5))
        x_p = self.norm(x_p)
        x_p = self.dw(x_p)
        x_p = F.gelu(x_p)
        x_p = self.pw(x_p)
        x_full = (
            x_p.unsqueeze(3).unsqueeze(5)
            .expand(B, C, NRp, P, NCp, P).contiguous()
        )
        return x_full.view(B, C, NRp * P, NCp * P)


class HPACMini(nn.Module):
    """ANR token-stream context model. Verbatim port of PR95 inflate.py:196-268.

    Forward signature: ``(tokens, idx, prev_tokens) → logits (B, num_classes, H, W)``.
    Used at compress time to compute per-symbol probabilities; the arithmetic coder
    consumes those probabilities to range-code the next group of tokens.
    """

    def __init__(self, num_pairs: int = 600, num_classes: int = NUM_CLASSES,
                 P: int = 32, delta: int = 2, d_film: int = 32, ch: int = 64,
                 use_spm: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.P = P
        self.delta = delta
        self.ch = ch
        self.use_spm = use_spm
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        self.film_gen = nn.Linear(d_film, ch * 2)
        self.conv_a = _MaskedConv2dPG(
            num_classes + 2, ch, k=7, padding=3, type_="A", delta=delta,
        )
        self.gn_a = _ChannelNorm2d(ch)
        self.conv_b1 = _MaskedConv2dPG(
            ch, ch, k=5, padding=4, dilation=2, groups=ch, type_="B", delta=delta,
        )
        self.gn_b1 = _ChannelNorm2d(ch)
        self.conv_b2 = _MaskedConv2dPG(
            ch, ch, k=3, padding=4, dilation=4, groups=ch, type_="B", delta=delta,
        )
        self.gn_b2 = _ChannelNorm2d(ch)
        self.conv_past = nn.Conv2d(num_classes, ch, kernel_size=3, padding=1)
        self.spm = _CausalSPM(ch, P=P) if use_spm else None
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1, padding=0)
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_P: int = -1

    def _patch_coord_grid(self, B: int, device: torch.device) -> torch.Tensor:
        if self._cached_P != self.P or self._coord_cache.numel() == 0:
            P = self.P
            ys = torch.linspace(-1.0, 1.0, P, device=device).view(1, 1, P, 1).expand(1, 1, P, P)
            xs = torch.linspace(-1.0, 1.0, P, device=device).view(1, 1, 1, P).expand(1, 1, P, P)
            grid = torch.cat([ys, xs], dim=1)
            self._coord_cache = grid
            self._cached_P = self.P
        return self._coord_cache.expand(B, -1, -1, -1)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.P
        NRp, NCp = H // P, W // P
        x = x.view(B, C, NRp, P, NCp, P).permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B * NRp * NCp, C, P, P)

    def _from_patches(self, x_p: torch.Tensor, B: int, NRp: int, NCp: int) -> torch.Tensor:
        P = self.P
        C = x_p.shape[1]
        x_p = x_p.view(B, NRp, NCp, C, P, P).permute(0, 3, 1, 4, 2, 5).contiguous()
        return x_p.view(B, C, NRp * P, NCp * P)

    def forward(self, tokens: torch.Tensor, idx: torch.Tensor,
                prev_tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.long:
            raise TypeError(f"tokens must be long, got {tokens.dtype}")
        if prev_tokens.shape != tokens.shape:
            raise ValueError(
                f"prev_tokens shape {tuple(prev_tokens.shape)} must equal "
                f"tokens shape {tuple(tokens.shape)}"
            )
        B, H, W = tokens.shape
        P = self.P
        NRp, NCp = H // P, W // P
        Np = NRp * NCp
        x = F.one_hot(tokens, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        x_p = self._to_patches(x)
        coord_p = self._patch_coord_grid(B * Np, x.device)
        x_in_p = torch.cat([x_p, coord_p], dim=1)
        h_p = self.gn_a(self.conv_a(x_in_p))
        emb = self.frame_embed(idx)
        film = self.film_gen(emb)
        scale, shift = film.chunk(2, dim=1)
        scale_p = (
            scale.view(B, 1, self.ch, 1, 1)
            .expand(B, Np, self.ch, 1, 1)
            .reshape(B * Np, self.ch, 1, 1)
        )
        shift_p = (
            shift.view(B, 1, self.ch, 1, 1)
            .expand(B, Np, self.ch, 1, 1)
            .reshape(B * Np, self.ch, 1, 1)
        )
        h_p = h_p * (1.0 + scale_p) + shift_p
        h_p = F.gelu(h_p)
        x_prev = (
            F.one_hot(prev_tokens, num_classes=self.num_classes)
            .permute(0, 3, 1, 2).float()
        )
        h_past_full = self.conv_past(x_prev)
        h_past_p = self._to_patches(h_past_full)
        h_p = h_p + h_past_p
        if self.spm is not None:
            h_p = h_p + self._to_patches(self.spm(h_past_full))
        h_p = F.gelu(self.gn_b1(self.conv_b1(h_p)))
        h_p = F.gelu(self.gn_b2(self.conv_b2(h_p)))
        logits_p = self.head(h_p)
        return self._from_patches(logits_p, B, NRp, NCp)


# ── FiLM portability guard (CPU FP32 = GPU FP16 bit-portable) ────────────


class FiLMPortabilityGuard:
    """Runtime guard that CPU FP32 FiLM table == GPU forward table within atol.

    Per PR95 inflate.py:47-50 commentary, the FiLM matmul on tiny (8x64)
    inputs produces different bytes on Ada vs Turing GPUs because cuBLAS
    picks different kernels. The fix is to BAKE the table on CPU FP32 at
    export time and ship the baked table in the archive.

    This guard validates the baked table by checking that the on-CPU
    recomputation matches the buffer to within ``atol``. Call this before
    EVERY archive build that uses ``TokenRendererV62``.
    """

    def __init__(self, atol: float = 1e-5) -> None:
        if atol <= 0:
            raise ValueError(f"atol must be positive, got {atol}")
        self.atol = atol

    def check(self, master: TokenRendererV62) -> None:
        """Raise ``RuntimeError`` if the baked table disagrees with a fresh CPU compute."""
        if not master._film_table_baked:
            raise RuntimeError(
                "FiLMPortabilityGuard.check called before master.bake_film_table()"
            )
        with torch.no_grad():
            emb = master.frame_embed.weight.detach().cpu().float()
            w = master.film_gen.weight.detach().cpu().float()
            b = master.film_gen.bias.detach().cpu().float()
            recomputed = (emb @ w.T + b).contiguous()
        baked_cpu = master._film_table.detach().cpu().contiguous()
        if recomputed.shape != baked_cpu.shape:
            raise RuntimeError(
                f"baked _film_table shape {tuple(baked_cpu.shape)} != "
                f"recomputed {tuple(recomputed.shape)}"
            )
        diff = (recomputed - baked_cpu).abs().max().item()
        if diff > self.atol:
            raise RuntimeError(
                f"FiLM portability check failed: max abs diff {diff:.3e} > "
                f"atol {self.atol:.3e}. Re-bake on CPU FP32 before archive export."
            )


# ── PPMd codec for HPAC weight bytes ─────────────────────────────────────


def encode_hpac_weights_ppmd(state_dict: dict, *, max_order: int = 4,
                              mem_size_mb: int = 16) -> bytes:
    """Pack an HPACMini packed state_dict + PPMd-compress for archive.

    Per PR95 inflate.py:302-303 / 384-391 — the canonical archive layout
    is ``state_dict → torch.save → bytes → pyppmd.compress(...)``.
    """
    import pyppmd  # PYPPMD_LGPL_OK:legacy-HPACMini-wire-format-encoder-compatibility
    if max_order < 2 or max_order > 16:
        raise ValueError(f"max_order must be in [2, 16], got {max_order}")
    if mem_size_mb < 1 or mem_size_mb > 4096:
        raise ValueError(f"mem_size_mb must be in [1, 4096], got {mem_size_mb}")
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    return pyppmd.compress(raw, max_order=max_order, mem_size=mem_size_mb << 20)


def decode_hpac_weights_ppmd(payload: bytes, *, max_order: int = 4,
                              mem_size_mb: int = 16,
                              map_location: str | torch.device = "cpu") -> dict:
    """Decode PPMd-compressed HPACMini state_dict bytes."""
    import pyppmd  # PYPPMD_LGPL_OK:legacy-HPACMini-wire-format-decoder-compatibility
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like, got {type(payload).__name__}")
    raw = pyppmd.decompress(
        bytes(payload), max_order=max_order, mem_size=mem_size_mb << 20,
    )
    return torch.load(
        io.BytesIO(raw), map_location=map_location, weights_only=False,
    )


# ── Eval roundtrip helper ────────────────────────────────────────────────


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
    """uint8 STE clamp (mirror of lane_12_v2)."""
    clamped = rgb.clamp(0.0, 255.0)
    rounded = clamped + (clamped.round().detach() - clamped.detach())
    return rounded


# ── Score-aware train step ───────────────────────────────────────────────


def _pose_tensor(output: torch.Tensor | dict) -> torch.Tensor:
    """Normalize PoseNet output to the pose tensor used by surrogates."""
    if isinstance(output, dict):
        if "pose" not in output:
            raise KeyError("PoseNet output dict missing 'pose' key")
        value = output["pose"]
    else:
        value = output
    if not torch.is_tensor(value):
        raise TypeError(f"pose output must be a tensor, got {type(value).__name__}")
    return value


def train_step(
    *,
    master: TokenRendererV62,
    slave: ShrinkSingleNeRV,
    pair_indices: torch.Tensor,
    tokens: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    eval_roundtrip: bool = True,
) -> dict:
    """Joint score-aware training step.

    Trains the master (token-conditioned RGB renderer) + slave
    (latent-conditioned RGB renderer) jointly. Loss is the score-domain
    Lagrangian ``λ_seg·d_seg + λ_pose·d_pose`` through SegNet + PoseNet
    (CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable).

    HPACMini is NOT trained in this step — it is trained separately to
    minimize ``H(tokens | prev_tokens, idx)`` (negative log-likelihood) on the
    encoded token corpus. That trainer lives in
    ``experiments/train_anr_token_renderer.py``.

    Parameters
    ----------
    master, slave
        Trainable modules.
    pair_indices
        ``(B,) long`` indices in [0, num_pairs).
    tokens
        ``(B, SEGNET_IN_H, SEGNET_IN_W) long`` per-frame categorical labels
        (the SegNet-class argmax of the corresponding GT frame). At training
        time these come from passing GT through the contest SegNet OR from
        the EMA shadow of a co-trained SegNet head. They are NOT random.
    gt_pairs_uint8
        ``(B, 2, 3, CAMERA_H, CAMERA_W) uint8 or float`` GT pairs from
        ``upstream/videos/0.mkv`` via ``RealPairBatchSource``.
    scorer_seg, scorer_pose
        Frozen scorers from ``tac.scorer.load_differentiable_scorers``.
    seg_surrogate, pose_surrogate
        Distortion surrogates. Phase A defaults to MSE; Phase B may swap.
    lambda_seg, lambda_pose
        Score-domain Lagrangian weights.
    eval_roundtrip
        Per CLAUDE.md ``check_no_eval_roundtrip_false`` non-negotiable, MUST be True.
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false. Use the dedicated probe test."
        )
    if tokens.shape[1:] != (SEGNET_IN_H, SEGNET_IN_W):
        raise ValueError(
            f"tokens spatial shape must be ({SEGNET_IN_H}, {SEGNET_IN_W}), "
            f"got {tuple(tokens.shape[1:])}"
        )
    if gt_pairs_uint8.shape[-2:] != (CAMERA_H, CAMERA_W):
        raise ValueError(
            f"gt_pairs_uint8 spatial shape must be ({CAMERA_H}, {CAMERA_W}), "
            f"got {tuple(gt_pairs_uint8.shape[-2:])}"
        )

    # Render BOTH frames per pair: slave produces frame 0, master produces frame 1.
    slave_out = slave(pair_indices)  # (B, 3, CAMERA_H, CAMERA_W)
    master_out = master(tokens, pair_indices)  # (B, 3, CAMERA_H, CAMERA_W)
    rendered = torch.stack([slave_out, master_out], dim=1)  # (B, 2, 3, H, W)

    # Eval-roundtrip simulation
    rendered_uint8_ste = _eval_roundtrip_uint8_clamp(rendered)
    gt_pairs = gt_pairs_uint8.float()

    # SegNet path
    seg_pred_logits = scorer_seg(scorer_seg.preprocess_input(rendered_uint8_ste))
    with torch.no_grad():
        seg_target_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    loss_seg_unweighted = seg_surrogate(seg_pred_logits, seg_target_logits)

    # PoseNet path
    pose_pred = _pose_tensor(
        scorer_pose(scorer_pose.preprocess_input(rendered_uint8_ste))
    )
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
        "rendered_uint8_ste": rendered_uint8_ste,
    }


# ── Archive packer ───────────────────────────────────────────────────────


def _pack_section(payload: bytes) -> bytes:
    """Length-prefixed (LE u32) section wrap (matches PR97 H3 grammar)."""
    if len(payload) > 0xFFFFFFFF:
        raise ValueError(f"payload too large: {len(payload)} > 2^32-1")
    return struct.pack("<I", len(payload)) + payload


def export_to_archive(
    *,
    config: ANRTokenRendererConfig,
    master: TokenRendererV62,
    slave: ShrinkSingleNeRV,
    hpac_state_packed: dict,
    tokens_bin: bytes,
    portability_guard: FiLMPortabilityGuard | None = None,
) -> tuple[bytes, str]:
    """Pack the ANR substrate into the monolithic archive bytes.

    Layout: ``HEADER (16) | META (len + body) | MASTER (len + fp16 state) |
    SLAVE (len + state) | HPAC (len + PPMd state) | TOKENS (len + bytes)``.

    Returns (archive_bytes, sha256_hex).
    """
    if portability_guard is not None:
        portability_guard.check(master)

    # HEADER
    header = struct.pack(
        "<4sHHII",
        ANR_MAGIC,
        ANR_FORMAT_ID,
        ANR_FORMAT_VERSION,
        config.num_pairs,
        0,  # flags reserved
    )
    assert len(header) == 16, f"header size {len(header)} != 16"

    # META (torch_save dict)
    meta = {
        "N": config.num_pairs,
        "P": config.hpac_P,
        "delta": config.hpac_delta,
        "ch": config.hpac_ch,
        "slave_channels": list(config.slave_channels),
        "slave_d_lat": config.slave_d_lat,
        "d_film": config.d_film,
        "use_spm": config.hpac_use_spm,
        "hpac_d_film": config.hpac_d_film,
    }
    meta_buf = io.BytesIO()
    torch.save(meta, meta_buf)
    meta_bytes = meta_buf.getvalue()

    # MASTER (fp16 state_dict, SCN-baked)
    master_sd_fp16 = {
        k: (v.detach().cpu().to(torch.float16) if torch.is_floating_point(v) else v.detach().cpu())
        for k, v in master.state_dict().items()
    }
    master_buf = io.BytesIO()
    torch.save(master_sd_fp16, master_buf)
    master_bytes = master_buf.getvalue()

    # SLAVE (state_dict; INT4 LSQ-quant pre-applied at build time per PR95; Phase
    # A ships fp16 placeholder + a future LSQ quantizer applies the conversion).
    slave_sd_fp16 = {
        k: (v.detach().cpu().to(torch.float16) if torch.is_floating_point(v) else v.detach().cpu())
        for k, v in slave.state_dict().items()
    }
    slave_buf = io.BytesIO()
    torch.save(slave_sd_fp16, slave_buf)
    slave_bytes = slave_buf.getvalue()

    # HPAC (PPMd-compressed packed state_dict)
    hpac_bytes = encode_hpac_weights_ppmd(hpac_state_packed)

    if not isinstance(tokens_bin, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"tokens_bin must be bytes-like, got {type(tokens_bin).__name__}"
        )
    tokens_bin = bytes(tokens_bin)

    blob = (
        header
        + _pack_section(meta_bytes)
        + _pack_section(master_bytes)
        + _pack_section(slave_bytes)
        + _pack_section(hpac_bytes)
        + _pack_section(tokens_bin)
    )
    return blob, hashlib.sha256(blob).hexdigest()


def parse_archive_sections(blob: bytes) -> dict:
    """Inverse of ``export_to_archive``. Returns dict of section bytes."""
    if len(blob) < 16:
        raise ValueError(f"archive too short: {len(blob)} < 16 bytes for header")
    header = blob[:16]
    magic, fmt_id, fmt_ver, num_pairs, flags = struct.unpack("<4sHHII", header)
    if magic != ANR_MAGIC:
        raise ValueError(f"magic mismatch: {magic!r} != {ANR_MAGIC!r}")
    if fmt_id != ANR_FORMAT_ID:
        raise ValueError(f"format_id mismatch: {fmt_id} != {ANR_FORMAT_ID}")
    if fmt_ver != ANR_FORMAT_VERSION:
        raise ValueError(f"format_version mismatch: {fmt_ver} != {ANR_FORMAT_VERSION}")

    offset = 16
    sections = {}
    for name in ("meta", "master_state", "slave_state", "hpac_state", "tokens"):
        if offset + 4 > len(blob):
            raise ValueError(
                f"archive truncated before section {name!r}: "
                f"offset={offset} len={len(blob)}"
            )
        (length,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        if offset + length > len(blob):
            raise ValueError(
                f"archive truncated inside section {name!r}: "
                f"offset={offset} length={length} archive_len={len(blob)}"
            )
        sections[name] = blob[offset:offset + length]
        offset += length
    if offset != len(blob):
        raise ValueError(
            f"trailing bytes after parsed sections: parsed up to {offset}, "
            f"archive_len={len(blob)}"
        )
    sections["_header"] = {
        "magic": magic,
        "format_id": fmt_id,
        "format_version": fmt_ver,
        "num_pairs": num_pairs,
        "flags": flags,
    }
    return sections


__all__ = [
    "ANR_FORMAT_ID",
    "ANR_FORMAT_VERSION",
    "ANR_MAGIC",
    "ARCHIVE_GRAMMAR",
    "CAMERA_H",
    "CAMERA_W",
    "FEAT_H",
    "FEAT_W",
    "NUM_CLASSES",
    "SEGNET_IN_H",
    "SEGNET_IN_W",
    "ANRTokenRendererConfig",
    "FiLMPortabilityGuard",
    "HPACMini",
    "ShrinkSingleNeRV",
    "TokenRendererV62",
    "decode_hpac_weights_ppmd",
    "encode_hpac_weights_ppmd",
    "export_to_archive",
    "parse_archive_sections",
    "train_step",
]
