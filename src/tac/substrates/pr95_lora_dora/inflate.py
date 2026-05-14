# SPDX-License-Identifier: MIT
"""Inflate runtime for PR95 + LoRA/DoRA TRAILER archives.

Substrate-engineering opt-out per Catalog #124; LOC budget ≤200.

Contest contract (per Catalog #146): the deployed `inflate.sh` invokes this
file as `python inflate.py <archive_dir> <output_dir> <file_list>`, iterates
file_list, decodes each archive to raw uint8 RGB at camera resolution
(874, 1164), and writes contiguous `(N_frames * H * W * 3)` bytes per video.

No PoseNet / SegNet / scorer imports per CLAUDE.md strict-scorer-rule.

Behaviorally equivalent to PR95's `inflate.py` (72 LOC) for archives without
a LoRA trailer; for archives WITH a trailer, the adapters are applied to the
loaded base weights BEFORE the forward pass.
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Camera resolution (per upstream evaluate.py contract)
CAMERA_H, CAMERA_W = 874, 1164


def _load_pr95_base(base_bytes: bytes):
    """Parse PR95's monolithic 0.bin grammar into (decoder_sd, latents, meta).

    Mirror of PR95 `codec.py::parse_archive`. Vendored here so inflate.py is
    self-contained (no intake-clone import).
    """
    import json

    import brotli
    import numpy as np

    buf = io.BytesIO(base_bytes)
    (meta_len,) = struct.unpack("<I", buf.read(4))
    meta = json.loads(brotli.decompress(buf.read(meta_len)))
    (dec_len,) = struct.unpack("<I", buf.read(4))
    decoder_blob = buf.read(dec_len)
    (lat_len,) = struct.unpack("<I", buf.read(4))
    latents_brotli = buf.read(lat_len)

    # Decoder blob: brotli-compressed [n_tensors][per-tensor records]
    dec_raw = brotli.decompress(decoder_blob)
    dbuf = io.BytesIO(dec_raw)
    (n,) = struct.unpack("<I", dbuf.read(4))
    decoder_sd: dict[str, torch.Tensor] = {}
    for _ in range(n):
        (nl,) = struct.unpack("<I", dbuf.read(4))
        name = dbuf.read(nl).decode("utf-8")
        (nd,) = struct.unpack("<I", dbuf.read(4))
        shape = tuple(struct.unpack("<I", dbuf.read(4))[0] for _ in range(nd))
        (scale,) = struct.unpack("<f", dbuf.read(4))
        (size,) = struct.unpack("<I", dbuf.read(4))
        zz = np.frombuffer(dbuf.read(size), dtype=np.uint8)
        # zigzag decode
        zz32 = zz.astype(np.int32)
        q = np.where(zz32 % 2 == 0, zz32 // 2, -(zz32 // 2) - 1).astype(np.int8)
        decoder_sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale

    # Latents
    lat_raw = brotli.decompress(latents_brotli)
    lbuf = io.BytesIO(lat_raw)
    n_pairs, d = struct.unpack("<II", lbuf.read(8))
    mins = torch.from_numpy(np.frombuffer(lbuf.read(d * 2), dtype=np.float16).copy()).float()
    scales_l = torch.from_numpy(np.frombuffer(lbuf.read(d * 2), dtype=np.float16).copy()).float()
    total = n_pairs * d
    lo = np.frombuffer(lbuf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(lbuf.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n_pairs, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2,
                     -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n_pairs):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    latents = torch.from_numpy(q.astype(np.float32)) * scales_l.unsqueeze(0) + mins.unsqueeze(0)

    return decoder_sd, latents, meta


def _apply_adapters_to_state_dict(decoder_sd, adapter_records):
    """Fold each LoRA/DoRA adapter into the corresponding base weight tensor.

    For LoRA: W_eff = W_frozen + (alpha/r) * B @ A (reshaped back to original).
    For DoRA: W_eff = m * (W_frozen + (alpha/r) * B @ A) / ||...||_col.
    """
    for rec in adapter_records:
        name = rec["name"] + ".weight"
        if name not in decoder_sd:
            continue
        W = decoder_sd[name].float()
        original_shape = tuple(W.shape)
        if W.dim() == 4:
            flat = W.reshape(W.shape[0], -1)
        elif W.dim() == 2:
            flat = W
        else:
            continue

        alpha = float(rec["alpha"])
        rank = int(rec["rank"])
        scale = alpha / max(rank, 1)
        delta = scale * (rec["B"].float() @ rec["A"].float())

        if rec["kind"] == "lora":
            eff = flat + delta
        else:
            V = flat + delta
            V_norm = torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
            mag = rec["magnitude"].float().unsqueeze(1)
            eff = mag * (V / V_norm)

        decoder_sd[name] = eff.reshape(original_shape)
    return decoder_sd


def inflate(src_bin: str, dst_raw: str) -> int:
    """Read one archive, write raw uint8 RGB frames at camera resolution."""
    # Imports inside the function to keep cold-start minimal
    from .archive import parse_lora_archive
    from .pr95_base import HNeRVDecoder

    with open(src_bin, "rb") as f:
        archive_bytes = f.read()

    pr95_base_bytes, adapter_records = parse_lora_archive(archive_bytes)
    decoder_sd, latents, meta = _load_pr95_base(pr95_base_bytes)

    if adapter_records:
        decoder_sd = _apply_adapters_to_state_dict(decoder_sd, adapter_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]

    n_frames_written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                               mode="bicubic", align_corners=False)
            frames = (up.clamp(0, 255).permute(0, 2, 3, 1)
                        .round().to(torch.uint8).cpu().numpy())
            fout.write(frames.tobytes())
            n_frames_written += B * 2

    return n_frames_written


def _main(argv: list[str]) -> int:
    """CLI entry point. Contract: <archive_dir> <output_dir> <file_list>.

    file_list is a newline-delimited file of video basenames (no extension).
    For each basename, we decode <archive_dir>/<base>.bin -> <output_dir>/<base>.raw.
    """
    if len(argv) != 3:
        print("Usage: python inflate.py <archive_dir> <output_dir> <file_list>",
              file=sys.stderr)
        return 1
    archive_dir = Path(argv[0])
    output_dir = Path(argv[1])
    file_list_path = Path(argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(file_list_path) as fl:
        for line in fl:
            base = line.strip()
            if not base:
                continue
            src = archive_dir / f"{base}.bin"
            dst = output_dir / f"{base}.raw"
            n = inflate(str(src), str(dst))
            print(f"{base}: wrote {n} frames -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
