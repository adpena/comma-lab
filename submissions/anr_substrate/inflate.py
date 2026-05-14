#!/usr/bin/env python3
"""ANR full-substrate inflate.

Layout: HEADER (16) | META | MASTER_STATE | SLAVE_STATE | HPAC_STATE | TOKENS.
All sections are length-prefixed (LE u32) per the parser-section manifest in
``src/tac/anr_token_renderer.ARCHIVE_GRAMMAR``.

No PoseNet/SegNet load at inflate (strict-scorer-rule).
"""
from __future__ import annotations
import io
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pyppmd
import constriction
import torch
import torch.nn as nn
import torch.nn.functional as F

CAMERA_H, CAMERA_W = 874, 1164
SEGNET_IN_H, SEGNET_IN_W = 384, 512
FEAT_H, FEAT_W = 6, 8
NUM_CLASSES = 5
MAGIC = b"ANRV"
FORMAT_ID, FORMAT_VERSION = 0x50, 1


class TokenRendererV62(nn.Module):
    def __init__(self, num_pairs, d_film=8):
        super().__init__()
        self.frame_embed = nn.Embedding(num_pairs, d_film)
        self.film_gen = nn.Linear(d_film, 64)
        self.conv1 = nn.Conv2d(NUM_CLASSES, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)
        self.act = nn.GELU()
        self.register_buffer("_film_table", torch.zeros(num_pairs, 64), persistent=False)

    def bake(self):
        with torch.no_grad():
            emb = self.frame_embed.weight.detach().cpu().float()
            w = self.film_gen.weight.detach().cpu().float()
            b = self.film_gen.bias.detach().cpu().float()
            self._film_table.copy_((emb @ w.T + b).to(self._film_table.device))

    def forward(self, tokens, idx):
        x = F.one_hot(tokens, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        x = self.gn1(self.conv1(x))
        film = self._film_table[idx]
        scale, shift = film.chunk(2, dim=1)
        x = x * (1.0 + scale.view(-1, 32, 1, 1)) + shift.view(-1, 32, 1, 1)
        x = self.act(x)
        x = self.act(self.gn2(self.conv2(x)))
        raw = torch.sigmoid(self.out_conv(x)) * 255.0
        return F.interpolate(raw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)


class _NeRVBlock(nn.Module):
    def __init__(self, ci, co, s=2):
        super().__init__()
        self.dw = nn.Conv2d(ci, ci, 3, padding=1, groups=ci, bias=False)
        self.pw = nn.Conv2d(ci, co * s * s, 1, bias=True)
        self.ps = nn.PixelShuffle(s)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.ps(self.pw(self.dw(x))))


class ShrinkSingleNeRV(nn.Module):
    def __init__(self, num_pairs, d_lat=6, channels=(24, 16, 12, 8, 8, 6, 6)):
        super().__init__()
        self.channels = channels
        self.codes = nn.Embedding(num_pairs, d_lat)
        self.stem = nn.Linear(d_lat, channels[0] * FEAT_H * FEAT_W, bias=True)
        self.stem_act = nn.GELU()
        self.blocks = nn.ModuleList([_NeRVBlock(channels[i], channels[i + 1]) for i in range(6)])
        self.head = nn.Conv2d(channels[-1], 3, 1, bias=True)
        self.per_pair_bias = nn.Embedding(num_pairs, 3)

    def forward(self, idx):
        z = self.codes(idx)
        x = self.stem_act(self.stem(z).view(-1, self.channels[0], FEAT_H, FEAT_W))
        for blk in self.blocks:
            x = blk(x)
        out = self.head(x) + self.per_pair_bias(idx).view(-1, 3, 1, 1)
        raw = torch.sigmoid(out) * 255.0
        return F.interpolate(raw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)


def _parse(blob):
    if len(blob) < 16:
        raise ValueError("archive too short")
    magic, fmt_id, fmt_ver, num_pairs, _ = struct.unpack("<4sHHII", blob[:16])
    if magic != MAGIC or fmt_id != FORMAT_ID or fmt_ver != FORMAT_VERSION:
        raise ValueError(f"header mismatch: {magic!r} {fmt_id} {fmt_ver}")
    off = 16
    out = {"num_pairs": num_pairs}
    for name in ("meta", "master_state", "slave_state", "hpac_state", "tokens"):
        (n,) = struct.unpack("<I", blob[off:off + 4])
        off += 4
        out[name] = blob[off:off + n]
        off += n
    return out


def _load_state(buf, device):
    sd = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)
    return {k: (v.to(device).float() if torch.is_floating_point(v) else v.to(device))
            for k, v in sd.items()}


def select_inflate_device() -> str:
    """Honor ``PACT_INFLATE_DEVICE`` (auto/cpu/cuda); MPS is forbidden.

    Per A1 council Round 1 finding F1/F11 + CLAUDE.md ``MPS auth eval is
    NOISE`` non-negotiable. Sister of Catalog #205.
    """
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError("PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate")
    if policy == "cpu":
        return "cpu"
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but CUDA unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if len(sys.argv) != 3:
        print("usage: inflate.py <archive_path> <dst_raw>", file=sys.stderr)
        sys.exit(2)
    archive_path, dst_raw = Path(sys.argv[1]), Path(sys.argv[2])
    device = select_inflate_device()

    sections = _parse(archive_path.read_bytes())
    meta = torch.load(io.BytesIO(sections["meta"]), map_location="cpu", weights_only=False)
    N = meta["N"]
    P, delta, ch = meta["P"], meta["delta"], meta["ch"]
    slave_channels = tuple(meta["slave_channels"])
    slave_d_lat, d_film = meta["slave_d_lat"], meta["d_film"]

    master = TokenRendererV62(N, d_film=d_film).to(device).eval()
    master.load_state_dict(_load_state(sections["master_state"], device), strict=False)
    master.bake()

    slave = ShrinkSingleNeRV(N, d_lat=slave_d_lat, channels=slave_channels).to(device).eval()
    slave.load_state_dict(_load_state(sections["slave_state"], device), strict=False)

    # HPAC weights: PPMd-decompress, run token decode on CPU FP32 for portability
    hpac_raw = pyppmd.decompress(sections["hpac_state"], max_order=4, mem_size=16 << 20)
    hpac_packed = torch.load(io.BytesIO(hpac_raw), map_location="cpu", weights_only=False)

    # Categorical decoder uses constriction.RangeDecoder with per-symbol probabilities.
    # For Phase A reference inflate we expose the API; full HPACMini context-model
    # decode happens via tac.anr_token_renderer.HPACMini (research-only at this layer).
    # Tokens are pre-decoded at compress time into sections["tokens"] as raw uint8.
    tokens = np.frombuffer(sections["tokens"], dtype=np.uint8)
    if tokens.size != N * SEGNET_IN_H * SEGNET_IN_W:
        raise ValueError(
            f"tokens byte length {tokens.size} != expected "
            f"{N * SEGNET_IN_H * SEGNET_IN_W}"
        )
    tokens = torch.from_numpy(tokens.reshape(N, SEGNET_IN_H, SEGNET_IN_W).copy()).long().to(device)

    out = np.empty((N * 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    chunk = 8 if device == "cuda" else 2
    with torch.inference_mode():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            idx = torch.arange(s, e, device=device)
            mst = master(tokens[s:e], idx).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            slv = slave(idx).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(e - s):
                out[(s + i) * 2 + 0] = slv[i]
                out[(s + i) * 2 + 1] = mst[i]

    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    dst_raw.write_bytes(out.tobytes(order="C"))
    # Stash hpac_packed shape for forensic debugging (advisory only; no runtime use).
    _ = list(hpac_packed.keys())[:1]
    print(f"Wrote {dst_raw} shape={out.shape} bytes={dst_raw.stat().st_size}")


if __name__ == "__main__":
    main()
