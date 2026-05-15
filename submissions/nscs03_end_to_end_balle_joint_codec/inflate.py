#!/usr/bin/env python
"""NSCS03 (end-to-end Ballé joint codec) self-contained contest inflate.

Per HNeRV parity L4 + L9: the contest runtime tree MUST be self-contained.
Inflate MUST NOT import from ``tac.*`` (no PACT repo dependency at inflate
time). This file therefore inlines the minimal slice of architecture +
archive parsing needed to decode pixels from the NS03 0.bin bytes.

Wire format: see `tac.substrates.nscs03_end_to_end_balle_joint_codec.archive`
for the canonical declaration. Only the DECODER side (g_s + h_s + entropy
state_dicts + main+hyper latents + meta) is required for pixel emission;
the encoder + h_a state_dicts ride along for archive-completeness +
structural-consumption proof per Catalog #105/#139/#220.

Per Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device`)
the local `select_inflate_device` helper mirrors the canonical
`tac.substrates._shared.inflate_runtime.select_inflate_device` byte-for-byte
contract: honors ``PACT_INFLATE_DEVICE`` env var, refuses ``mps``, falls
back to CPU when CUDA absent.
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import torch
import torch.nn as nn
import torch.nn.functional as F

CAMERA_H, CAMERA_W = 874, 1164

NS03_MAGIC = b"NS03"
NS03_HEADER_FMT = "<4sBHHHHHHHIIIIIIII"
NS03_HEADER_SIZE = struct.calcsize(NS03_HEADER_FMT)


def select_inflate_device() -> str:
    """Mirror of `tac.substrates._shared.inflate_runtime.select_inflate_device`.

    Catalog #205 contract: must NOT silently fork on cuda.is_available; must
    honor ``PACT_INFLATE_DEVICE`` env var (auto/cpu/cuda); MPS forbidden.
    """
    value = (os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but torch.cuda is not available")
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda")


class _GDN(nn.Module):
    """Inlined GDN/IGDN per Ballé 2018; mirror of substrate architecture._GDN."""

    def __init__(self, channels: int, *, inverse: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.inverse = bool(inverse)
        self.eps = float(eps)
        self.raw_beta = nn.Parameter(torch.full((channels,), float(math.log(math.expm1(1.0)))))
        self.raw_gamma = nn.Parameter(torch.eye(channels) * float(math.log(math.expm1(0.1))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = F.softplus(self.raw_beta).to(x.dtype)
        gamma = F.softplus(self.raw_gamma).to(x.dtype)
        x_sq = x * x
        norm = F.conv2d(x_sq, gamma.view(self.channels, self.channels, 1, 1))
        norm = norm + beta.view(1, -1, 1, 1)
        norm = norm.clamp(min=self.eps).sqrt()
        return x * norm if self.inverse else x / norm


class _SynthesisTransform(nn.Module):
    """g_s: y_hat -> RGB-stack output (mirror of substrate)."""

    def __init__(self, channels: tuple[int, ...], out_channels: int, *, gdn_eps: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = channels[0]
        for c in channels[1:]:
            layers.append(nn.ConvTranspose2d(prev, c, 5, stride=2, padding=2, output_padding=1))
            layers.append(_GDN(c, inverse=True, eps=gdn_eps))
            prev = c
        layers.append(nn.ConvTranspose2d(prev, out_channels, 5, stride=2, padding=2, output_padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.net(y_hat)


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = torch.load(io.BytesIO(raw), weights_only=False)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _dequantize(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def parse_archive(blob: bytes):
    """Minimal NS03 parser. Returns dict of all sections."""
    if len(blob) < NS03_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    fields = struct.unpack(NS03_HEADER_FMT, blob[:NS03_HEADER_SIZE])
    (magic, version, num_pairs, main_c, main_h, main_w,
     hyper_c, hyper_h, hyper_w,
     ga_len, gs_len, ha_len, hs_len, eb_len,
     main_lat_len, hyper_lat_len, meta_len) = fields
    if magic != NS03_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported schema version: {version}")
    p = NS03_HEADER_SIZE
    ga_blob = blob[p:p + ga_len]; p += ga_len
    gs_blob = blob[p:p + gs_len]; p += gs_len
    ha_blob = blob[p:p + ha_len]; p += ha_len
    hs_blob = blob[p:p + hs_len]; p += hs_len
    eb_blob = blob[p:p + eb_len]; p += eb_len
    main_blob = blob[p:p + main_lat_len]; p += main_lat_len
    hyper_blob = blob[p:p + hyper_lat_len]; p += hyper_lat_len
    meta = json.loads(blob[p:p + meta_len].decode("utf-8"))
    import numpy as np
    q_main = torch.from_numpy(np.frombuffer(main_blob, dtype=np.int16).copy()).view(num_pairs, main_c, main_h, main_w)
    q_hyper = torch.from_numpy(np.frombuffer(hyper_blob, dtype=np.int16).copy()).view(num_pairs, hyper_c, hyper_h, hyper_w)
    main = _dequantize(q_main, float(meta["_main_quant_scale"]), float(meta["_main_quant_zero_point"]))
    hyper = _dequantize(q_hyper, float(meta["_hyper_quant_scale"]), float(meta["_hyper_quant_zero_point"]))
    return {
        "decoder_sd": _deserialize_state_dict(gs_blob),
        "main_latents": main,
        "hyper_latents": hyper,
        "meta": meta,
        "num_pairs": num_pairs,
    }


def inflate_video(archive_bytes: bytes, dst_raw: Path, *, device: str) -> int:
    parsed = parse_archive(archive_bytes)
    cfg_meta = parsed["meta"]["config"]
    g_s_channels = tuple(int(c) for c in cfg_meta["g_s_channels"])
    out_channels = int(cfg_meta["out_channels"])
    gdn_eps = float(cfg_meta.get("gdn_eps", 1e-6))
    output_h = int(cfg_meta["output_height"])
    output_w = int(cfg_meta["output_width"])
    # Build only the decoder (other state_dicts ride along for completeness;
    # we discard them at inflate-time per the L4 budget).
    g_s = _SynthesisTransform(g_s_channels, out_channels, gdn_eps=gdn_eps).to(device).eval()
    g_s_sd = {f"net.{k}" if not k.startswith("net.") else k: v for k, v in parsed["decoder_sd"].items()}
    # The state_dict keys are already "net.0.weight" etc. since the trainer
    # called .state_dict() on the _SynthesisTransform module which has self.net.
    incompat = g_s.load_state_dict(parsed["decoder_sd"], strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected keys in decoder_sd: {incompat.unexpected_keys}")

    main_latents = parsed["main_latents"].to(device=device, dtype=next(g_s.parameters()).dtype)
    num_pairs = int(main_latents.shape[0])
    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), dst_raw.open("wb") as fh:
        for i in range(num_pairs):
            y_hat = main_latents[i:i + 1]
            recon = g_s(y_hat)  # (1, 6, H_dec, W_dec)
            if recon.shape[-2:] != (output_h, output_w):
                recon = F.interpolate(recon, size=(output_h, output_w), mode="bilinear", align_corners=False)
            recon = torch.sigmoid(recon)
            # Split into 2 frames + interpolate to camera resolution
            rgb_0 = recon[:, 0:3]
            rgb_1 = recon[:, 3:6]
            for rgb in (rgb_0, rgb_1):
                if tuple(rgb.shape[-2:]) != (CAMERA_H, CAMERA_W):
                    rgb = F.interpolate(rgb * 255.0, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
                else:
                    rgb = rgb * 255.0
                u8 = rgb.clamp(0.0, 255.0).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
                fh.write(u8.tobytes(order="C"))
                frames_written += 1
    return frames_written


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list = Path(sys.argv[3]).read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    device = select_inflate_device()
    for fname in file_list:
        if not fname.strip():
            continue
        base = fname.rsplit(".", 1)[0]
        dst = output_dir / f"{base}.raw"
        inflate_video(archive_bytes, dst, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
