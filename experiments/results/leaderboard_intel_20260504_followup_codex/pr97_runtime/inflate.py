#!/usr/bin/env python
"""H3 + range-coded mask inflate.

Reads payload "p" from data_dir, decodes mask via range_mask_codec, loads H3
model (LowRankLinear pose_mlp), generates frames pair-by-pair, writes 0.raw.
"""
import io, os, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

import brotli, einops, numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Self-contained H3 architecture (no autoresearch/ dependency).
from model import GeneratorPoseLR, apply_fp4_to_model, MODEL_H, MODEL_W, OUT_H, OUT_W  # noqa: F401


# ── Range mask codec ──
def compile_codec(src):
    bin_path = src.with_suffix(".exe") if os.name == "nt" else src.with_suffix(".bin")
    if bin_path.exists() and bin_path.stat().st_mtime > src.stat().st_mtime:
        return bin_path
    for compiler in ("c++", "g++", "clang++"):
        if shutil.which(compiler):
            subprocess.run([compiler, "-O3", "-std=c++17", str(src), "-o", str(bin_path)], check=True)
            return bin_path
    raise RuntimeError("no c++ compiler found (need c++/g++/clang++)")


MASK_BAND_W_SPLITS = (3, 8, 8, 3)  # must match build_archive.py
MASK_H_PART = 128
MASK_TILE_TRANSFORMS = (
    "id", "revHW", "revT_revH",
    "id", "id", "id", "revT", "revT", "revH", "revT", "id",
    "id", "id", "id", "revW", "revT_revH", "revT_revHW", "id", "id",
    "id", "id", "id",
)

def invert_mask_tile_transform(chunk, tag):
    """Inverse of build_archive.py's static scan transform. All current ops are self-inverse."""
    ops = set(tag.split("_"))
    out = chunk
    if "revW" in ops or "revHW" in ops:
        out = out[:, :, ::-1]
    if "revH" in ops or "revHW" in ops:
        out = out[:, ::-1, :]
    if "revT" in ops:
        out = out[::-1, :, :]
    return np.ascontiguousarray(out)

def decode_mask_range(mask_bytes, codec_bin):
    """Decode masks from tiled multi-stream format.

    Wire format: [u32 n_chunks][u32 size_0][bytes_0][u32 size_1][bytes_1]...
    4 horizontal bands × per-band W splits (sum = 22 chunks). Reverse:
    tile back, transpose (600, 512, 384) -> (600, 384, 512).
    """
    import struct
    n_chunks_expected = sum(MASK_BAND_W_SPLITS)
    o = 0
    n_chunks = struct.unpack("<I", mask_bytes[o:o+4])[0]; o += 4
    if n_chunks != n_chunks_expected:
        raise RuntimeError(f"mask chunk count {n_chunks} != expected {n_chunks_expected}")
    streams = []
    for _ in range(n_chunks):
        sz = struct.unpack("<I", mask_bytes[o:o+4])[0]; o += 4
        streams.append(mask_bytes[o:o+sz]); o += sz
    if o != len(mask_bytes):
        raise RuntimeError(f"mask payload trailing bytes: {o} vs {len(mask_bytes)}")

    mt = np.empty((600, 512, 384), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx = 0
        for i, n_w in enumerate(MASK_BAND_W_SPLITS):
            w_part = 384 // n_w
            for j in range(n_w):
                packed = td / f"c_{i}_{j}.bin"
                raw = td / f"c_{i}_{j}.raw"
                packed.write_bytes(streams[idx])
                subprocess.run([str(codec_bin), "decode", str(packed), str(raw)], check=True)
                chunk = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(600, MASK_H_PART, w_part)
                chunk = invert_mask_tile_transform(chunk, MASK_TILE_TRANSFORMS[idx])
                mt[:, i*MASK_H_PART:(i+1)*MASK_H_PART, j*w_part:(j+1)*w_part] = chunk
                idx += 1
    if idx != len(MASK_TILE_TRANSFORMS):
        raise RuntimeError(f"mask tile transform count mismatch: {idx} vs {len(MASK_TILE_TRANSFORMS)}")
    arr = np.ascontiguousarray(mt.transpose(0, 2, 1))  # (600, 384, 512)
    return torch.from_numpy(arr).contiguous()


# ── Payload split ──
def split_payload(blob):
    """Returns (mask, pose, model, sidecar_or_b'') — sidecar slot optional for fwd compat."""
    o = 0
    def read():
        nonlocal o
        n = struct.unpack("<I", blob[o:o+4])[0]; o += 4
        b = blob[o:o+n]; o += n
        return b
    mask = read(); pose = read(); model = read()
    sidecar = read() if o < len(blob) else b""
    if o != len(blob):
        raise RuntimeError(f"payload trailing bytes: {o} vs {len(blob)}")
    return mask, pose, model, sidecar


def main():
    if len(sys.argv) < 4:
        print("usage: inflate.py <data_dir> <out_dir> <file_list>")
        sys.exit(2)
    data_dir, out_dir, file_list = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [l.strip() for l in file_list.read_text().splitlines() if l.strip()]

    payload = (data_dir / "p").read_bytes()
    print(f"[inflate] payload {len(payload)} bytes")
    mask_b, pose_b, model_b, sidecar_b = split_payload(payload)
    print(f"[inflate] mask {len(mask_b)} | pose {len(pose_b)} | model {len(model_b)} | sidecar {len(sidecar_b)}")

    # Decode mask
    codec_src = HERE / "range_mask_codec.cpp"
    codec_bin = compile_codec(codec_src)
    print(f"[inflate] decoding mask via {codec_bin.name}...")
    masks = decode_mask_range(mask_b, codec_bin)
    print(f"[inflate] masks {tuple(masks.shape)} {masks.dtype}")

    # Decode pose (per-dim N-bit packed quant + brotli)
    pose_raw = brotli.decompress(pose_b)
    o = 0
    n_pair, n_dim = struct.unpack("<II", pose_raw[o:o+8]); o += 8
    bits_per_dim = list(pose_raw[o:o+n_dim]); o += n_dim
    los_scales = []
    for d in range(n_dim):
        lo, scale = struct.unpack("<ff", pose_raw[o:o+8]); o += 8
        los_scales.append((lo, scale))
    bs = pose_raw[o:]
    # Unpack the bit stream, then slice per-dim runs of n_pair*bits[d] bits
    total_bits = sum(bits_per_dim) * n_pair
    needed_bytes = (total_bits + 7) // 8
    bs = bs[:needed_bytes]
    bits_arr = np.unpackbits(np.frombuffer(bs, dtype=np.uint8))[:total_bits]
    poses_np = np.empty((n_pair, n_dim), dtype=np.float32)
    bit_offset = 0
    for d in range(n_dim):
        bits = bits_per_dim[d]
        block = bits_arr[bit_offset : bit_offset + n_pair * bits].reshape(n_pair, bits)
        weights = (1 << np.arange(bits-1, -1, -1)).astype(np.uint32)
        vals = (block.astype(np.uint32) * weights).sum(axis=1)
        lo, scale = los_scales[d]
        poses_np[:, d] = lo + vals.astype(np.float32) * scale
        bit_offset += n_pair * bits
    poses = torch.from_numpy(poses_np).float()
    print(f"[inflate] poses {tuple(poses.shape)} bits_per_dim={bits_per_dim}")

    # Sidecar: mask edits + pose deltas + per-pair f1 warp (applied below)
    sidecar = None
    if sidecar_b:
        from sidecar import decode_sidecar_blob, apply_mask_edits, apply_pose_deltas
        sidecar = decode_sidecar_blob(sidecar_b)
        print(f"[inflate] sidecar: x2={len(sidecar['x2'])} cmaes={len(sidecar['cmaes'])} "
              f"pattern={len(sidecar['pattern'])} pose={len(sidecar['pose'])} warp={len(sidecar['warp'])}")
        # Mutate masks in-place (in CPU long form, before .to(device))
        masks_long = masks.long()
        apply_mask_edits(masks_long, sidecar)
        masks = masks_long
        apply_pose_deltas(poses, sidecar)

    # Load model — flat-FP4 decode using the schema baked into schema_h3.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inflate] device={device}")
    sys.path.insert(0, str(HERE))
    from flat_fp4 import decode as flat_fp4_decode
    from schema_h3 import SCHEMA
    raw_model = brotli.decompress(model_b)
    print(f"[inflate] flat-fp4 decoding {len(raw_model)} model bytes...")
    sd = flat_fp4_decode(raw_model, SCHEMA)
    gen = GeneratorPoseLR().to(device)
    missing, unexpected = gen.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)
    apply_fp4_to_model(gen)
    gen.eval()

    # Generate frames
    n_pairs = masks.shape[0]
    pairs_per_file = 600
    out_h, out_w = 874, 1164
    bs = 4

    cursor = 0
    for fname in files:
        end = cursor + pairs_per_file
        f_masks = masks[cursor:end].to(device)
        f_poses = poses[cursor:end].to(device)
        cursor = end

        out_arr = np.empty((pairs_per_file, 2, out_h, out_w, 3), dtype=np.uint8)
        with torch.inference_mode():
            for s in tqdm(range(0, pairs_per_file, bs), desc=f"gen {fname}"):
                b_mask = f_masks[s:s+bs].long()
                b_pose = f_poses[s:s+bs]
                p1, p2 = gen(b_mask, b_pose)
                f1u = F.interpolate(p1, (out_h, out_w), mode="bilinear", align_corners=False)
                f2u = F.interpolate(p2, (out_h, out_w), mode="bilinear", align_corners=False)
                f1 = f1u.clamp(0, 255).round().to(torch.uint8)
                f2 = f2u.clamp(0, 255).round().to(torch.uint8)
                # (B, C, H, W) -> (B, H, W, C)
                f1 = einops.rearrange(f1, "b c h w -> b h w c").cpu().numpy()
                f2 = einops.rearrange(f2, "b c h w -> b h w c").cpu().numpy()
                out_arr[s:s+f1.shape[0], 0] = f1
                out_arr[s:s+f2.shape[0], 1] = f2
        # Apply per-pair f1 warps from sidecar (post-upsample, in OUT_H × OUT_W space)
        if sidecar and sidecar["warp"]:
            from sidecar import apply_warps_to_f1_inplace
            # Translate per-file warp indices to global pair indices used in sidecar
            offset = cursor - pairs_per_file
            file_sidecar = {"warp": {pi - offset: qxqy for pi, qxqy in sidecar["warp"].items()
                                     if offset <= pi < offset + pairs_per_file}}
            if file_sidecar["warp"]:
                apply_warps_to_f1_inplace(out_arr, file_sidecar, device=device)
                print(f"[inflate] applied {len(file_sidecar['warp'])} f1 warps")
        out_path = out_dir / f"{Path(fname).stem}.raw"
        out_arr.tofile(out_path)
        print(f"[inflate] wrote {out_path} {out_arr.nbytes} bytes")


if __name__ == "__main__":
    main()
