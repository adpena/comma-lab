#!/usr/bin/env python3
"""Categorical full-substrate inflate.

Layout: HEADER (16) | META | RENDERER_STATE | PALETTE | TOKENS.
Per src/tac/categorical_substrate.ARCHIVE_GRAMMAR.

No PoseNet/SegNet load at inflate (strict-scorer-rule).
"""
from __future__ import annotations
import io
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAMERA_H, CAMERA_W = 874, 1164
SEGNET_IN_H, SEGNET_IN_W = 384, 512
NUM_CLASSES = 5
MAGIC = b"CATG"
FORMAT_ID, FORMAT_VERSION = 0x51, 1


class CategoricalRenderer(nn.Module):
    def __init__(self, num_pairs, palette_dim=8, shading_channels=16):
        super().__init__()
        self.palette_dim = palette_dim
        C = NUM_CLASSES
        SC = shading_channels
        self.frame_embed = nn.Embedding(num_pairs, palette_dim)
        self.palette = nn.Parameter(torch.zeros(C, palette_dim))
        in_ch = palette_dim + C + 2
        self.conv1 = nn.Conv2d(in_ch, SC, 3, padding=1)
        self.conv2 = nn.Conv2d(SC, SC, 3, padding=1)
        self.rgb_0 = nn.Conv2d(SC, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(SC, 3, 3, padding=1)
        self.register_buffer("_coord", torch.zeros(0), persistent=False)

    def _coord_grid(self, H, W, device):
        if self._coord.numel() == 0 or self._coord.shape[-2:] != (H, W):
            ys = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
            xs = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
            self._coord = torch.cat([ys, xs], dim=1)
        return self._coord

    def forward(self, tokens, idx):
        B, H, W = tokens.shape
        C = NUM_CLASSES
        P = self.palette_dim
        one_hot = F.one_hot(tokens, num_classes=C).permute(0, 3, 1, 2).float()
        palette_at_pixel = self.palette[tokens].permute(0, 3, 1, 2)
        emb = self.frame_embed(idx).view(B, P, 1, 1)
        modulated = palette_at_pixel * (1.0 + emb)
        coord = self._coord_grid(H, W, tokens.device).expand(B, -1, -1, -1)
        feats = torch.cat([modulated, one_hot, coord], dim=1)
        x = F.gelu(self.conv1(feats))
        x = F.gelu(self.conv2(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        rendered = torch.stack([f0, f1], dim=1)  # (B, 2, 3, H, W)
        Bn, Fp, Cn, Hn, Wn = rendered.shape
        flat = rendered.reshape(Bn * Fp, Cn, Hn, Wn)
        up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        return up.reshape(Bn, Fp, Cn, CAMERA_H, CAMERA_W)


def _parse(blob):
    if len(blob) < 16:
        raise ValueError("archive too short")
    magic, fmt_id, fmt_ver, num_pairs, _ = struct.unpack("<4sHHII", blob[:16])
    if magic != MAGIC or fmt_id != FORMAT_ID or fmt_ver != FORMAT_VERSION:
        raise ValueError(f"header mismatch: {magic!r} {fmt_id} {fmt_ver}")
    off = 16
    out = {"num_pairs": num_pairs}
    for name in ("meta", "renderer_state", "palette", "tokens"):
        (n,) = struct.unpack("<I", blob[off:off + 4])
        off += 4
        out[name] = blob[off:off + n]
        off += n
    return out


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
    palette_dim = meta["palette_dim"]
    shading_channels = meta["shading_channels"]

    renderer = CategoricalRenderer(N, palette_dim=palette_dim,
                                    shading_channels=shading_channels).to(device).eval()
    sd = torch.load(io.BytesIO(sections["renderer_state"]), map_location="cpu",
                    weights_only=False)
    sd = {k: (v.to(device).float() if torch.is_floating_point(v) else v.to(device))
          for k, v in sd.items()}
    renderer.load_state_dict(sd, strict=False)

    # Palette ships separately as a single fp16 tensor.
    palette_fp16 = torch.load(io.BytesIO(sections["palette"]), map_location="cpu",
                               weights_only=False)
    renderer.palette.data = palette_fp16.to(device).float()

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
            r = renderer(tokens[s:e], idx)  # (chunk, 2, 3, H, W)
            r = r.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            for i in range(e - s):
                out[(s + i) * 2 + 0] = r[i, 0].transpose(1, 2, 0)
                out[(s + i) * 2 + 1] = r[i, 1].transpose(1, 2, 0)

    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    dst_raw.write_bytes(out.tobytes(order="C"))
    print(f"Wrote {dst_raw} shape={out.shape} bytes={dst_raw.stat().st_size}")


if __name__ == "__main__":
    main()
