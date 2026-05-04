#!/usr/bin/env python
"""Pack trained checkpoints + encoded tokens into archive.zip.

Steps:
  1. DALI-decode the input videos and run SegNet to produce 600 token maps.
  2. Bake the master state-dict (SCN -> FP16-exact + non-SCN floats kept FP32),
     gzip-compress -> master.pt.gz.
  3. Bake the slave state-dict (LSQ-INT4 weights -> FP16, INT8 codes),
     gzip-compress -> slave.pt.gz.
  4. Reuse the PPMd-compressed HPAC weights as-is.
  5. Encode the token maps via HPAC range-coder -> tokens.bin.
  6. Write meta.pt and zip everything together.
"""
from __future__ import annotations
import sys, io, gzip, shutil, time, zipfile, argparse
from pathlib import Path
import numpy as np
import pyppmd
import torch
import torch.nn.functional as F
import constriction
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from frame_utils import DaliVideoDataset, segnet_model_input_size
from modules import SegNet, segnet_sd_path
from hpac import HPACMini, encode_frame

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 600
SAVE_DIR = HERE.parent / "training_workspace"
SUB_DIR = HERE.parent
ARCH_DIR = SUB_DIR / "archive_build"

HPAC_CFG = dict(P=32, delta=2, ch=64, d_film=8, use_spm=True)
PPMD_MAX_ORDER = 4
PPMD_MEM_SIZE = 16 << 20
B_MIN, B_MAX = 0.5, 8.0


def compute_gt_tokens(bs: int = 8) -> np.ndarray:
    """Decode 600 video pairs via DALI and run SegNet on the second frame
    of each pair. Output is uint8 tokens of shape (600, H, W)."""
    print(f"[archive] DALI-decoding GT for {N} pairs...", flush=True)
    with open(ROOT / "public_test_video_names.txt") as f:
        names = [line.strip() for line in f if line.strip()]
    ds = DaliVideoDataset(names, data_dir=ROOT / "videos",
                          batch_size=bs, device=torch.device(DEV))
    ds.prepare_data()
    parts = []
    for _, _, batch in ds:
        parts.append(batch.detach().cpu())
    rgb_pairs = torch.cat(parts, dim=0)[:N].contiguous()

    print("[archive] running SegNet to produce gt_tokens...", flush=True)
    H_in, W_in = segnet_model_input_size[1], segnet_model_input_size[0]
    segnet = SegNet().eval().to(DEV)
    segnet.load_state_dict(load_file(segnet_sd_path, device=DEV))
    gt = torch.empty(N, H_in, W_in, dtype=torch.uint8, device=DEV)
    with torch.inference_mode():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            x = rgb_pairs[s:e].to(DEV).float().permute(0, 1, 4, 2, 3)
            tok = segnet(segnet.preprocess_input(x[:, 1:2])).argmax(1).to(torch.uint8)
            gt[s:e] = tok
    return gt.cpu().numpy()


def export_master(master_pt: Path, out_pt: Path):
    """Bake SCN-quantized master to FP16 + keep other floats FP32, gzip-save.

    SCN weights satisfy w == q * 2^e exactly, so casting to FP16 is lossless.
    Other params (biases, GroupNorm, frame_embed) are NOT exact in FP16, so
    we keep them FP32 to preserve bit-identity with inflate.py.
    """
    print(f"[archive] export master <- {master_pt.name}", flush=True)
    data = torch.load(master_pt, map_location="cpu", weights_only=False)
    sd = data["state_dict"]
    out = {k: v for k, v in sd.items() if not k.endswith("_step_initialized")}
    bases = sorted({k[:-2] for k in out if k.endswith(".b") and (k[:-2] + ".e") in out})
    for base in bases:
        b = out[base + ".b"].float().clamp(B_MIN, B_MAX)
        e = out[base + ".e"].float()
        w_key = base + ".weight"
        if w_key not in out:
            continue
        w = out[w_key].float()
        shape = [1] * w.ndim; shape[0] = -1
        scale = torch.pow(2.0, e.view(*shape))
        max_q = torch.pow(2.0, b.view(*shape) - 1) - 1
        min_q = -torch.pow(2.0, b.view(*shape) - 1)
        q = torch.clamp(w / scale, min_q, max_q).round()
        out[w_key] = (q * scale).to(torch.float16)
        del out[base + ".b"], out[base + ".e"]
    for k, v in list(out.items()):
        if torch.is_floating_point(v):
            out[k] = v.detach() if v.dtype == torch.float16 else v.detach().to(torch.float32)
        else:
            out[k] = v.detach()
    buf = io.BytesIO()
    torch.save(out, buf)
    with gzip.open(out_pt, "wb", compresslevel=9) as f:
        f.write(buf.getvalue())
    sz = out_pt.stat().st_size
    print(f"[archive] master.pt.gz = {sz:_} bytes ({sz / 1024:.1f} KB)", flush=True)
    return sz, data.get("best_seg")


def export_slave(slave_pt: Path, out_pt: Path):
    """Bake LSQ INT4 slave weights to FP16, quantize per-pair codes to INT8."""
    print(f"[archive] export slave <- {slave_pt.name}", flush=True)
    data = torch.load(slave_pt, map_location="cpu", weights_only=False)
    sd = data["state_dict"]
    channels = list(data.get("channels", (24, 16, 12, 8, 8, 6, 6)))
    d_lat = data.get("d_lat", 6)
    out = {k: v for k, v in sd.items()
           if not (k.endswith("_step_initialized") or k.endswith(".step"))}
    for name in list(out.keys()):
        if not name.endswith(".weight"):
            continue
        step_key = name.replace(".weight", ".step")
        if step_key not in sd:
            continue
        w = out[name].float()
        step = sd[step_key].float().abs().clamp_min(1e-10)
        s_shape = [1] * w.ndim; s_shape[0] = -1
        step = step.view(*s_shape)
        q = (w / step).clamp(-8, 7).round()
        out[name] = (q * step).to(torch.float16)
    if "codes.weight" in out:
        codes = out["codes.weight"].float()
        abs_max = codes.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        step_codes = abs_max / 127.0
        q = (codes / step_codes).round().clamp(-127, 127)
        out["codes.weight"] = (q * step_codes).to(torch.float16)
    for k, v in list(out.items()):
        if torch.is_floating_point(v):
            out[k] = v.to(torch.float16)
    buf = io.BytesIO()
    torch.save(out, buf)
    with gzip.open(out_pt, "wb", compresslevel=9) as f:
        f.write(buf.getvalue())
    sz = out_pt.stat().st_size
    print(f"[archive] slave.pt.gz  = {sz:_} bytes ({sz / 1024:.1f} KB)  "
          f"channels={channels} d_lat={d_lat}", flush=True)
    return sz, data.get("best_pose"), channels, d_lat


def reconstruct_hpac_state_dict(packed_sd):
    """Rehydrate INT8-packed HPAC state-dict to FP32 (mirror of inflate-time logic)."""
    out = {}
    bases = sorted({k[:-len(".weight_q")] for k in packed_sd if k.endswith(".weight_q")})
    for base in bases:
        q = packed_sd[base + ".weight_q"].float()
        scale = packed_sd[base + ".weight_scale"]
        shape = [1] * q.ndim; shape[0] = -1
        out[base + ".weight"] = (q * scale.view(*shape)).to(torch.float32)
    skip = {base + suffix for base in bases for suffix in (".weight_q", ".weight_scale")}
    for k, v in packed_sd.items():
        if k not in skip:
            out[k] = v
    return out


def load_hpac_from_ppmd(ppmd_path: Path) -> HPACMini:
    """Decompress hpac.pt.ppmd and load weights into a fresh HPACMini."""
    print(f"[archive] loading HPAC from {ppmd_path.name} (PPMd max_order={PPMD_MAX_ORDER})...",
          flush=True)
    decoded = pyppmd.decompress(ppmd_path.read_bytes(),
                                max_order=PPMD_MAX_ORDER, mem_size=PPMD_MEM_SIZE)
    packed_sd = torch.load(io.BytesIO(decoded), map_location=DEV, weights_only=False)
    sd = reconstruct_hpac_state_dict(packed_sd)
    gen = HPACMini(num_pairs=N, num_classes=5, **HPAC_CFG).to(DEV).eval()
    gen.load_state_dict(sd, strict=False)
    gen.set_scn(False)
    return gen


def write_tokens(gen: HPACMini, tokens_np: np.ndarray, out_bin: Path):
    """Arithmetic-code 600 token frames against gen, write to tokens.bin."""
    P = HPAC_CFG["P"]; delta = HPAC_CFG["delta"]
    print(f"[archive] HPAC encoding {tokens_np.shape[0]} frames "
          f"(P={P} δ={delta} use_spm={HPAC_CFG['use_spm']})...", flush=True)
    tokens_t = torch.from_numpy(tokens_np).long().to(DEV)
    prev_all = torch.zeros_like(tokens_t)
    prev_all[1:] = tokens_t[:-1]
    encoder = constriction.stream.queue.RangeEncoder()
    n_total = 0
    t0 = time.time()
    with torch.inference_mode():
        for f in range(N):
            n_total += encode_frame(gen, tokens_t[f:f + 1],
                                    torch.tensor([f], dtype=torch.long, device=DEV),
                                    prev_all[f:f + 1], encoder, P=P, delta=delta)
            if (f + 1) % 100 == 0 or f == 0:
                print(f"[archive]   encoded {f+1}/{N}  dt={time.time() - t0:.0f}s",
                      flush=True)
    blob = encoder.get_compressed().tobytes()
    out_bin.write_bytes(blob)
    bpp = len(blob) * 8 / max(n_total, 1)
    print(f"[archive] tokens.bin   = {len(blob):_} bytes ({len(blob) / 1024:.1f} KB)  "
          f"bpp={bpp:.4f}", flush=True)
    return len(blob), bpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-pt", type=Path, default=SAVE_DIR / "v62_full_dali_swa.pt")
    ap.add_argument("--slave-pt",  type=Path, default=SAVE_DIR / "slave_v62_shrink_dali_best.pt")
    ap.add_argument("--hpac-ppmd", type=Path, default=ARCH_DIR / "hpac.pt.ppmd")
    ap.add_argument("--sub-dir",   type=Path, default=SUB_DIR)
    args = ap.parse_args()

    sub_dir: Path = args.sub_dir
    arch_dir = sub_dir / "archive_build"
    sub_dir.mkdir(parents=True, exist_ok=True)
    arch_dir.mkdir(parents=True, exist_ok=True)

    gt = compute_gt_tokens()
    print(f"[archive] gt_tokens range [{gt.min()}, {gt.max()}]", flush=True)

    master_sz, best_seg = export_master(args.master_pt, arch_dir / "master.pt.gz")
    slave_sz, best_pose, channels, d_lat = export_slave(args.slave_pt, arch_dir / "slave.pt.gz")

    hpac_dst = arch_dir / "hpac.pt.ppmd"
    if hpac_dst.resolve() != args.hpac_ppmd.resolve():
        shutil.copyfile(args.hpac_ppmd, hpac_dst)
    hpac_sz = hpac_dst.stat().st_size
    print(f"[archive] hpac.pt.ppmd = {hpac_sz:_} bytes ({hpac_sz / 1024:.1f} KB)",
          flush=True)

    gen = load_hpac_from_ppmd(hpac_dst)
    tokens_sz, bpp = write_tokens(gen, gt, arch_dir / "tokens.bin")

    meta = {
        "N": N, "mode": "hpac",
        "P": HPAC_CFG["P"], "delta": HPAC_CFG["delta"], "ch": HPAC_CFG["ch"],
        "hpac_d_film": HPAC_CFG["d_film"], "use_spm": HPAC_CFG["use_spm"],
        "slave_channels": channels, "slave_d_lat": d_lat,
        "d_film": 8,
        "best_seg_train": best_seg, "best_pose_train": best_pose,
        "tokens_bpp": bpp, "gt_decoder": "dali",
        "ppmd_max_order": PPMD_MAX_ORDER,
    }
    torch.save(meta, arch_dir / "meta.pt")
    meta_sz = (arch_dir / "meta.pt").stat().st_size

    archive_zip = sub_dir / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
        for f in ("master.pt.gz", "slave.pt.gz", "hpac.pt.ppmd", "tokens.bin", "meta.pt"):
            zf.write(arch_dir / f, arcname=f)
    zip_sz = archive_zip.stat().st_size

    print("\n" + "=" * 60, flush=True)
    print(f"[archive] components:", flush=True)
    print(f"  master.pt.gz : {master_sz:_} bytes ({master_sz / 1024:.1f} KB)", flush=True)
    print(f"  slave.pt.gz  : {slave_sz:_} bytes  ({slave_sz / 1024:.1f} KB)", flush=True)
    print(f"  hpac.pt.ppmd : {hpac_sz:_} bytes  ({hpac_sz / 1024:.1f} KB)", flush=True)
    print(f"  tokens.bin   : {tokens_sz:_} bytes ({tokens_sz / 1024:.1f} KB)", flush=True)
    print(f"  meta.pt      : {meta_sz:_} bytes", flush=True)
    print(f"[archive] archive.zip = {zip_sz:_} bytes ({zip_sz / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
