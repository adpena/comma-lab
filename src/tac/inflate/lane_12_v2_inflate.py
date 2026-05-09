"""Lane 12-v2 NeRV-as-renderer inflate (≤ 100 LOC, ≤ 2 deps).

Reads ``0.bin`` per ARCHIVE_GRAMMAR in the design memo §2 and reconstructs
the renderer. Output: uint8 RGB frames at camera resolution (874, 1164).

Deps: torch + brotli (matches PR100 hnerv_lc_v2 exactly).
"""
from __future__ import annotations
import io, struct, sys
from pathlib import Path
import brotli  # noqa: F401 (declared dep)
import torch
import torch.nn.functional as F

from tac.lane_12_v2_nerv_as_renderer import (
    LANE_12_V2_MAGIC, Lane12V2NeRVConfig, Lane12V2NeRVRenderer, Lane12V2LatentTable,
)

CAMERA_H, CAMERA_W = 874, 1164


def _parse_archive(buf: bytes) -> tuple[Lane12V2NeRVConfig, dict, torch.Tensor]:
    if buf[:4] != LANE_12_V2_MAGIC:
        raise ValueError(f"bad magic {buf[:4]!r}")
    version, latent_dim, n_pairs, base_channels = struct.unpack("<HHHH", buf[4:12])
    if version != 1:
        raise ValueError(f"unsupported version {version}")
    config = Lane12V2NeRVConfig(
        latent_dim=latent_dim, base_channels=base_channels, n_pairs=n_pairs,
        cuda_required=False,  # inflate runs on whatever the contest provides
    )
    o = 12
    dec_len = struct.unpack("<I", buf[o:o + 4])[0]; o += 4
    dec_blob = buf[o:o + dec_len]; o += dec_len
    sca_len = struct.unpack("<I", buf[o:o + 4])[0]; o += 4
    sca_blob = buf[o:o + sca_len]; o += sca_len
    lat_len = struct.unpack("<I", buf[o:o + 4])[0]; o += 4
    lat_blob = buf[o:o + lat_len]; o += lat_len
    side_len = struct.unpack("<I", buf[o:o + 4])[0]; o += 4
    _sidecar_blob = buf[o:o + side_len]; o += side_len  # Phase A: ignored
    if o != len(buf):
        raise ValueError(f"trailing bytes: {o} vs {len(buf)}")
    # Build a temp renderer to read the schema (deterministic from config).
    template = Lane12V2NeRVRenderer(config)
    schema = template.schema
    raw_codes = brotli.decompress(dec_blob)
    scales = torch.frombuffer(bytearray(sca_blob), dtype=torch.float16).float()
    sd: dict = {}
    o2 = 0
    for i, (key, shape) in enumerate(schema):
        n_el = 1
        for d in shape:
            n_el *= d
        chunk = torch.frombuffer(bytearray(raw_codes[o2:o2 + n_el]), dtype=torch.int8)
        sd[key] = (chunk.float() * float(scales[i])).reshape(shape)
        o2 += n_el
    if o2 != len(raw_codes):
        raise ValueError(f"decoder leftover: {o2} vs {len(raw_codes)}")
    latents = _decode_latents(lat_blob, n_pairs, latent_dim)
    return config, sd, latents


def _decode_latents(blob: bytes, n: int, d: int) -> torch.Tensor:
    raw = brotli.decompress(blob)
    buf = io.BytesIO(raw)
    n_chk, d_chk = struct.unpack("<II", buf.read(8))
    if (n_chk, d_chk) != (n, d):
        raise ValueError(f"latent shape mismatch: {(n_chk, d_chk)} vs {(n, d)}")
    mins = torch.frombuffer(bytearray(buf.read(d * 2)), dtype=torch.float16).float()
    scales = torch.frombuffer(bytearray(buf.read(d * 2)), dtype=torch.float16).float()
    total = n * d
    lo = torch.frombuffer(bytearray(buf.read(total)), dtype=torch.uint8).to(torch.int32)
    hi = torch.frombuffer(bytearray(buf.read(total)), dtype=torch.uint8).to(torch.int32)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = torch.where(delta_zz % 2 == 0, delta_zz // 2, -(delta_zz // 2) - 1).to(torch.int32)
    q = torch.zeros_like(delta)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    return q.float() * scales[None, :] + mins[None, :]


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    config, sd, latents = _parse_archive(archive_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = Lane12V2NeRVRenderer(config).to(device)
    renderer.load_state_dict(sd)
    renderer.eval()
    latents = latents.to(device)
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, config.n_pairs, 16):
            j = min(i + 16, config.n_pairs)
            decoded = renderer(latents[i:j])
            B = j - i
            flat = decoded.reshape(B * 2, 3, *config.eval_size)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(frames.tobytes())
            n += B * 2
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m tac.inflate.lane_12_v2_inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
