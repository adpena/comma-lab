#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self-contained receiver for torch-vehicle HNeRV D2/FILM archives."""
from __future__ import annotations

import io
import json
import struct
import sys

import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAMERA_H, CAMERA_W = 874, 1164
POSE_MAGIC = b"PFLM"


class HNeRVDecoder(nn.Module):
    def __init__(self, latent_dim=28, base_channels=36, eval_size=(384, 512)):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        c = int(base_channels)
        self.channels = [c, c, c, int(c * 0.75), int(c * 0.58), int(c * 0.5), int(c * 0.5)]
        self.stem = nn.Linear(int(latent_dim), self.channels[0] * self.base_h * self.base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z):
        b = z.shape[0]
        x = self.stem(z).view(b, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=False):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


class _PoseFiLM(nn.Module):
    def __init__(self, *, pose_dim: int, channels: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(pose_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), 2 * int(channels))

    def forward(self, pose):
        h = torch.sin(self.fc1(pose))
        gb = self.fc2(h)
        gamma = 1.0 + torch.tanh(gb[:, : self.fc2.out_features // 2])
        beta = gb[:, self.fc2.out_features // 2 :]
        return gamma, beta


class PoseFiLMHNeRVWrapper(nn.Module):
    def __init__(self, decoder, *, n_pairs: int, pose_dim: int = 6, film_hidden: int = 8):
        super().__init__()
        self.decoder = decoder
        self.pose_film = _PoseFiLM(
            pose_dim=pose_dim, channels=int(decoder.channels[0]), hidden=film_hidden
        )
        self.register_buffer("stored_pose", torch.zeros(int(n_pairs), int(pose_dim)))

    def set_stored_pose(self, pose):
        with torch.no_grad():
            self.stored_pose.copy_(pose.to(self.stored_pose.device, self.stored_pose.dtype))

    def forward(self, z, idx):
        d = self.decoder
        b = z.shape[0]
        x = d.stem(z).view(b, d.channels[0], d.base_h, d.base_w)
        gamma, beta = self.pose_film(self.stored_pose[idx])
        x = gamma[:, :, None, None] * x + beta[:, :, None, None]
        x = torch.sin(x)
        for block, skip in zip(d.blocks, d.skips, strict=False):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = d.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(d.refine(x))
        f0 = torch.sigmoid(d.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(d.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def zigzag_decode_u8(arr_u8):
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def decode_decoder(data):
    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    n = struct.unpack("<I", buf.read(4))[0]
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl).decode("utf-8")
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8)
        q = zigzag_decode_u8(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


def decode_decoder_variable(data):
    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    flag = struct.unpack("<B", buf.read(1))[0]
    n = struct.unpack("<I", buf.read(4))[0]
    sd = {}
    for _ in range(n):
        nl_name = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl_name).decode("utf-8")
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        if flag == 1:
            _ = struct.unpack("<B", buf.read(1))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8)
        q = zigzag_decode_u8(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


def decode_latents(raw):
    buf = io.BytesIO(raw)
    n, d = struct.unpack("<II", buf.read(8))
    mins = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    return torch.from_numpy(q.astype(np.uint8).astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def decode_pose_section(section_bytes):
    buf = io.BytesIO(section_bytes)
    if buf.read(4) != POSE_MAGIC:
        raise ValueError("bad pose section magic")
    n, dpose = struct.unpack("<II", buf.read(8))
    blob_len = struct.unpack("<I", buf.read(4))[0]
    raw = brotli.decompress(buf.read(blob_len))
    rb = io.BytesIO(raw)
    mins = np.frombuffer(rb.read(dpose * 2), dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(rb.read(dpose * 2), dtype=np.float16).astype(np.float32)
    total = n * dpose
    lo = np.frombuffer(rb.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(rb.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, dpose)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    return torch.from_numpy(q.astype(np.uint8).astype(np.float32) * scales[None, :] + mins[None, :])


def parse_archive(archive_bytes):
    buf = io.BytesIO(archive_bytes)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta = json.loads(brotli.decompress(buf.read(meta_len)))
    dec_len = struct.unpack("<I", buf.read(4))[0]
    decoder_blob = buf.read(dec_len)
    lat_len = struct.unpack("<I", buf.read(4))[0]
    latents = decode_latents(brotli.decompress(buf.read(lat_len)))
    trailing = buf.read()
    var_meta = meta.get("variable_level_waterfill") or {}
    if bool(var_meta.get("decoder_blob_is_variable_format")):
        decoder_sd = decode_decoder_variable(decoder_blob)
    else:
        decoder_sd = decode_decoder(decoder_blob)
    pose = None
    if len(trailing) >= 4 and trailing[:4] == POSE_MAGIC:
        pose = decode_pose_section(trailing)
    return decoder_sd, latents, meta, pose


@torch.inference_mode()
def inflate(src_bin: str, dst_raw: str) -> int:
    with open(src_bin, "rb") as fh:
        decoder_sd, latents, meta, pose = parse_archive(fh.read())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    film_sd = {
        k[len("pose_film.") :]: v for k, v in decoder_sd.items() if k.startswith("pose_film.")
    }
    dec_sd = {k: v for k, v in decoder_sd.items() if not k.startswith("pose_film.")}
    decoder.load_state_dict({k: v.to(device) for k, v in dec_sd.items()})
    decoder.eval()
    model = decoder
    use_film = bool(film_sd) and pose is not None
    if use_film:
        model = PoseFiLMHNeRVWrapper(
            decoder, n_pairs=int(meta["n_pairs"]), film_hidden=8
        ).to(device)
        model.pose_film.load_state_dict({k: v.to(device) for k, v in film_sd.items()})
        model.set_stored_pose(pose.to(device))
        model.eval()
    latents = latents.to(device)
    n_pairs = int(meta["n_pairs"])
    eval_h, eval_w = meta["eval_size"]
    n = 0
    with open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            z = latents[i:j]
            if use_film:
                idx = torch.arange(i, j, device=device)
                decoded = model(z, idx)
            else:
                decoded = model(z)
            flat = decoded.reshape((j - i) * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8)
            fout.write(frames.cpu().numpy().tobytes())
            n += (j - i) * 2
    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
