#!/usr/bin/env python
"""Reverse-engineer + capacity-RD prune the CONVERGED PR95 bc36 HNeRV archive.

This is a REAL executor (NO FAKE): it inflates the actual PR95 archive, renders the
real HNeRVDecoder, applies the exact contest uint8 inflate roundtrip, and evaluates
the exact frozen SegNet/PoseNet DistortionNet (the same modules upstream/evaluate.py
uses) on the real GT decoded via frame_utils.yuv420_to_rgb (NEVER PyAV rgb24).

Authority caveats honored:
  * GT decode = upstream frame_utils.yuv420_to_rgb (the AVVideoDataset CPU path) so
    this is bit-exact to the contest --device cpu evaluator's GT.
  * d_seg = (argmax(seg_gt) != argmax(seg_recon)).mean() per upstream/modules.py.
  * d_pose = MSE on first out//2 pose dims per upstream/modules.py.
  * Score components via tac.contest_score.compute_contest_score.
  * Device defaults to CPU (the witness/inflate decode is CPU-native; MPS is NEVER
    a score authority). All numbers tagged [contest-CPU advisory] until a 600-pair
    upstream/evaluate.py --device cpu (promotion) / --device cuda (submission) row.

Subcommands:
  sanity   -- inflate PR95, render, eval; confirm d_seg ~ 5.6e-4 (reverse-eng works).
  prune    -- structured L2 channel prune to a target base_channels; prune-only eval.
  curve    -- run sanity + a list of prune rungs, write the capacity-RD curve JSON.

Borrowed-substrate accounting: PR95 weights are a COMPETITOR's published method.
This is a MEASUREMENT (optimal capacity) + a DEFENSIVE bank, NOT an innovative
submission. See the memo for borrowed_substrate_accounting.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"
HNERV_SRC = (
    REPO
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src"
)
PR95_ARCHIVE = REPO / "experiments/results/public_pr95_intake_20260504_codex/archive.zip"

# upstream must be importable for frame_utils / modules (bare imports inside them)
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(HNERV_SRC))

CAMERA_W, CAMERA_H = 1164, 874  # frame_utils.camera_size = (W, H)
SEQ_LEN = 2


# --------------------------------------------------------------------------- #
# Archive inflate (reuse the PR95 codec parser verbatim)                       #
# --------------------------------------------------------------------------- #
def _read_inner_bin(archive_path: Path) -> bytes:
    """PR95 archive.zip wraps a single ``0.bin`` blob; return its raw bytes."""
    import zipfile

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as z:
            names = z.namelist()
            inner = next((n for n in names if n.endswith(".bin")), names[0])
            return z.read(inner)
    return archive_path.read_bytes()


def inflate_pr95(archive_path: Path):
    """Return (decoder_sd dict[name->float32 tensor], latents (n_pairs,d), meta)."""
    from codec import parse_archive  # PR95 src/codec.py

    archive_bytes = _read_inner_bin(archive_path)
    decoder_sd, latents, meta = parse_archive(archive_bytes)
    return decoder_sd, latents, meta


def build_decoder(meta, base_channels=None):
    from model import HNeRVDecoder

    bc = base_channels if base_channels is not None else meta["base_channels"]
    dec = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=bc,
        eval_size=tuple(meta["eval_size"]),
    )
    return dec


# --------------------------------------------------------------------------- #
# GT cache (decode 0.mkv once via the exact contest CPU path)                  #
# --------------------------------------------------------------------------- #
def decode_gt_pairs(n_pairs: int, cache: Path) -> torch.Tensor:
    """(n_pairs, 2, H, W, 3) uint8 GT, decoded via frame_utils.yuv420_to_rgb.

    Cached to disk (mmap-friendly .npy) so repeated rungs reuse the same GT.
    """
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape[0] >= n_pairs:
            return torch.from_numpy(np.ascontiguousarray(arr[:n_pairs]))
    import av
    from frame_utils import yuv420_to_rgb

    video = UPSTREAM / "videos" / "0.mkv"
    container = av.open(str(video))
    stream = container.streams.video[0]
    frames = []
    need = n_pairs * SEQ_LEN
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8
        if len(frames) >= need:
            break
    container.close()
    frames = torch.stack(frames[:need])  # (need, H, W, 3)
    pairs = frames.view(n_pairs, SEQ_LEN, CAMERA_H, CAMERA_W, 3).contiguous()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, pairs.numpy())
    return pairs


# --------------------------------------------------------------------------- #
# Render the (pruned) decoder + exact inflate uint8 roundtrip                  #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def render_recon_pairs(decoder, latents, meta, device, batch=16) -> torch.Tensor:
    """Render -> bicubic upsample to camera res -> uint8, exactly per inflate.py.

    Returns (n_pairs, 2, H, W, 3) uint8 on CPU.
    """
    eval_h, eval_w = meta["eval_size"]
    n_pairs = latents.shape[0]
    decoder = decoder.to(device).eval()
    latents = latents.to(device)
    out = torch.empty((n_pairs, SEQ_LEN, CAMERA_H, CAMERA_W, 3), dtype=torch.uint8)
    for i in range(0, n_pairs, batch):
        j = min(i + batch, n_pairs)
        decoded = decoder(latents[i:j])  # (B,2,3,eval_h,eval_w)
        B = j - i
        flat = decoded.reshape(B * SEQ_LEN, 3, eval_h, eval_w)
        up = F.interpolate(
            flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
        )
        frames = (
            up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu()
        )  # (B*2,H,W,3)
        out[i:j] = frames.view(B, SEQ_LEN, CAMERA_H, CAMERA_W, 3)
    return out


# --------------------------------------------------------------------------- #
# Exact in-process eval via the contest DistortionNet                         #
# --------------------------------------------------------------------------- #
class ExactEvaluator:
    def __init__(self, device):
        from modules import DistortionNet, segnet_sd_path, posenet_sd_path

        self.device = device
        self.net = DistortionNet().eval().to(device)
        self.net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    @torch.inference_mode()
    def eval_pairs(self, gt_pairs, recon_pairs, batch=8):
        """Return (mean_d_pose, mean_d_seg) over all pairs. uint8 inputs (B,2,H,W,3)."""
        n = gt_pairs.shape[0]
        pose_sum = 0.0
        seg_sum = 0.0
        for i in range(0, n, batch):
            j = min(i + batch, n)
            g = gt_pairs[i:j].to(self.device)
            r = recon_pairs[i:j].to(self.device)
            pose_d, seg_d = self.net.compute_distortion(g, r)
            pose_sum += float(pose_d.sum())
            seg_sum += float(seg_d.sum())
        return pose_sum / n, seg_sum / n


# --------------------------------------------------------------------------- #
# Structured L2 channel prune of the HNeRV decoder                            #
# --------------------------------------------------------------------------- #
def _taper(C):
    return [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]


def prune_decoder_state(src_sd, meta, target_bc):
    """Structured channel prune of the converged bc36 state_dict to target_bc.

    Keeps the highest-L2-norm channels at each upsample stage (channel importance
    on the conv producing each stage's PixelShuffle input). Returns a state_dict
    compatible with HNeRVDecoder(base_channels=target_bc). PRUNE-ONLY (no retrain).

    Mechanism: the decoder channel widths are derived from base_channels via the
    HNeRV taper. We select, per stage, which output channels of the *previous*
    stage's conv to keep (by L2 over the conv's output filters), and correspondingly
    slice every tensor that touches those channel axes. The stem and rgb heads are
    handled by their channel dependencies.
    """
    src_C = meta["base_channels"]
    src_ch = _taper(src_C)
    dst_ch = _taper(target_bc)
    latent_dim = meta["latent_dim"]
    base_h, base_w = 6, 8

    # ----- choose kept indices per *channel width* level -------------------- #
    # The taper has 7 entries but only distinct widths matter per axis. We pick
    # kept channels per stage's "in_ch" (= src_ch[i]) and "out_ch" (= src_ch[i+1]).
    # For each source width w_src -> w_dst we rank channels by an importance score.
    # Importance for a given width is computed from the conv whose OUTPUT has that
    # width (so the kept set is consistent everywhere that width appears).

    # Build importance per width-level index 0..6.
    # blocks[i]: Conv2d(in=src_ch[i], out=src_ch[i+1]*4)  weight (out4, in, 3,3)
    #   the PixelShuffle(2) maps out4 -> out_ch=src_ch[i+1] by grouping 4 spatial.
    #   channel c of the post-PS tensor uses output filters [4c, 4c+1, 4c+2, 4c+3].
    # So importance of post-stage channel c at level (i+1) = L2 over those 4 filters.
    # Level 0 (stem output) importance = L2 over stem weight rows grouped per channel.

    importance = [None] * 7

    # stem: Linear(latent_dim -> src_ch[0]*base_h*base_w). weight (out, latent_dim).
    stem_w = src_sd["stem.weight"]  # (src_ch[0]*H*W, latent_dim)
    stem_w = stem_w.view(src_ch[0], base_h * base_w, latent_dim)
    importance[0] = stem_w.flatten(1).norm(dim=1)  # (src_ch[0],)

    for i in range(6):
        bw = src_sd[f"blocks.{i}.weight"]  # (out4, in, 3,3)
        out4 = bw.shape[0]
        out_ch = out4 // 4
        # group filters by post-PS channel: filter f belongs to channel f//4? No:
        # PixelShuffle(2) on (B, out_ch*4, h, w) -> (B, out_ch, 2h, 2w). The 4
        # sub-pixels of channel c are at planes [c*4 : c*4+4] in the *default*
        # torch PixelShuffle layout (C r^2 -> C with r^2 contiguous per channel).
        bw_g = bw.view(out_ch, 4, *bw.shape[1:])  # (out_ch,4,in,3,3)
        importance[i + 1] = bw_g.flatten(1).norm(dim=1)  # (out_ch,)

    # kept indices per level
    kept = []
    for lvl in range(7):
        w_src = src_ch[lvl]
        w_dst = dst_ch[lvl]
        if w_dst >= w_src:
            kept.append(torch.arange(w_src))
        else:
            idx = torch.topk(importance[lvl], w_dst).indices.sort().values
            kept.append(idx)

    new_sd = {}

    # stem: out rows are (channel, h*w). Keep rows for kept[0] channels.
    stem_w_full = src_sd["stem.weight"].view(src_ch[0], base_h * base_w, latent_dim)
    stem_b_full = src_sd["stem.bias"].view(src_ch[0], base_h * base_w)
    new_sd["stem.weight"] = stem_w_full[kept[0]].reshape(-1, latent_dim).contiguous()
    new_sd["stem.bias"] = stem_b_full[kept[0]].reshape(-1).contiguous()

    for i in range(6):
        in_lvl, out_lvl = i, i + 1
        ki, ko = kept[in_lvl], kept[out_lvl]
        bw = src_sd[f"blocks.{i}.weight"]  # (out4, in,3,3)
        bb = src_sd[f"blocks.{i}.bias"]  # (out4,)
        out_ch = bw.shape[0] // 4
        bw_g = bw.view(out_ch, 4, bw.shape[1], 3, 3)  # (out_ch,4,in,3,3)
        bb_g = bb.view(out_ch, 4)
        # keep out channels ko (and their 4 subpixels), in channels ki
        bw_g = bw_g[ko][:, :, ki, :, :]  # (out_dst,4,in_dst,3,3)
        bb_g = bb_g[ko]  # (out_dst,4)
        new_sd[f"blocks.{i}.weight"] = bw_g.reshape(
            len(ko) * 4, len(ki), 3, 3
        ).contiguous()
        new_sd[f"blocks.{i}.bias"] = bb_g.reshape(-1).contiguous()

        # skip: Identity if in==out else Conv2d(in,out,1)
        sk_w_key = f"skips.{i}.weight"
        if sk_w_key in src_sd:
            sk_w = src_sd[sk_w_key]  # (out, in,1,1)
            sk_b = src_sd[f"skips.{i}.bias"]
            new_sd[sk_w_key] = sk_w[ko][:, ki, :, :].contiguous()
            new_sd[f"skips.{i}.bias"] = sk_b[ko].contiguous()
        # else Identity -> requires in_ch==out_ch in BOTH src and dst. The HNeRV
        # taper has equal widths only at levels 0->1 and 1->2 (C->C). dst keeps
        # that property since dst taper is the same formula. If src had Identity
        # but dst widths differ from each other, that's impossible by construction.

    # refine: Sequential[ Conv2d(final, final//2, 3,d2), Conv2d(final//2, final,3) ]
    final_src = src_ch[-1]
    kf = kept[-1]  # kept final channels
    r0w = src_sd["refine.0.weight"]  # (final//2, final,3,3)
    r0b = src_sd["refine.0.bias"]  # (final//2,)
    r1w = src_sd["refine.1.weight"]  # (final, final//2,3,3)
    r1b = src_sd["refine.1.bias"]  # (final,)
    half_src = final_src // 2
    half_dst = len(kf) // 2
    # importance of the hidden 'half' channels: L2 over r0 output filters
    half_imp = r0w.flatten(1).norm(dim=1)  # (half_src,)
    kh = torch.topk(half_imp, half_dst).indices.sort().values if half_dst < half_src else torch.arange(half_src)
    new_sd["refine.0.weight"] = r0w[kh][:, kf, :, :].contiguous()
    new_sd["refine.0.bias"] = r0b[kh].contiguous()
    new_sd["refine.1.weight"] = r1w[kf][:, kh, :, :].contiguous()
    new_sd["refine.1.bias"] = r1b[kf].contiguous()

    # rgb heads: Conv2d(final, 3, 3)
    for head in ("rgb_0", "rgb_1"):
        hw = src_sd[f"{head}.weight"]  # (3, final,3,3)
        hb = src_sd[f"{head}.bias"]  # (3,)
        new_sd[f"{head}.weight"] = hw[:, kf, :, :].contiguous()
        new_sd[f"{head}.bias"] = hb.contiguous()

    return new_sd, kept


# --------------------------------------------------------------------------- #
# Byte cost via the exact PR95 codec (INT8 + brotli)                          #
# --------------------------------------------------------------------------- #
def archive_bytes_for(decoder_sd, latents, meta):
    """Build a full PR95-grammar archive blob and return its byte length.

    This is the REAL rate cost (INT8 per-tensor + brotli q11), the same codec the
    converged PR95 archive uses. Latents are unchanged (capacity prune only touches
    the decoder), so the latent blob is identical across rungs.
    """
    from codec import build_archive

    blob = build_archive(decoder_sd, latents, meta)
    return len(blob), blob


def count_params(sd):
    return int(sum(v.numel() for v in sd.values()))


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run_sanity(n_pairs, device, gt_cache, out_json):
    from tac.contest_score import compute_contest_score, seg_term, pose_term, rate_term

    t0 = time.time()
    decoder_sd, latents, meta = inflate_pr95(PR95_ARCHIVE)
    dec = build_decoder(meta)
    dec.load_state_dict(decoder_sd)
    np_total = count_params(dec.state_dict())

    gt = decode_gt_pairs(n_pairs, gt_cache)
    n_pairs = gt.shape[0]
    latents = latents[:n_pairs]

    recon = render_recon_pairs(dec, latents, meta, device)
    ev = ExactEvaluator(device)
    d_pose, d_seg = ev.eval_pairs(gt, recon)

    arch_bytes = PR95_ARCHIVE.stat().st_size
    sc = compute_contest_score(d_seg, d_pose, arch_bytes)
    result = {
        "stage": "sanity_gate_pr95_bc36",
        "evidence_grade": "[contest-CPU advisory]" if str(device) == "cpu" else f"[{device} advisory]",
        "device": str(device),
        "n_pairs": int(n_pairs),
        "base_channels": int(meta["base_channels"]),
        "decoder_params": np_total,
        "archive_bytes": int(arch_bytes),
        "d_seg": float(d_seg),
        "d_pose": float(d_pose),
        "score": float(sc),
        "score_components": {
            "seg": float(seg_term(d_seg)),
            "pose": float(pose_term(d_pose)),
            "rate": float(rate_term(arch_bytes)),
        },
        "target_d_seg_5p6e-4": 5.6e-4,
        "reproduced": bool(d_seg < 2.0e-3),  # within ~3.5x of target on a partial-pair sanity
        "wall_s": round(time.time() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def run_curve(n_pairs, device, rungs, gt_cache, out_json):
    from tac.contest_score import compute_contest_score, seg_term, pose_term, rate_term

    t0 = time.time()
    decoder_sd, latents, meta = inflate_pr95(PR95_ARCHIVE)
    full_dec = build_decoder(meta)
    full_dec.load_state_dict(decoder_sd)

    gt = decode_gt_pairs(n_pairs, gt_cache)
    n_pairs = gt.shape[0]
    latents = latents[:n_pairs]
    ev = ExactEvaluator(device)

    curve = []

    # rung 0: full bc36 (sanity anchor)
    recon = render_recon_pairs(full_dec, latents, meta, device)
    d_pose, d_seg = ev.eval_pairs(gt, recon)
    arch_bytes, _ = archive_bytes_for(decoder_sd, latents, meta)
    sc = compute_contest_score(d_seg, d_pose, arch_bytes)
    curve.append({
        "base_channels": int(meta["base_channels"]),
        "kind": "full_converged",
        "decoder_params": count_params(decoder_sd),
        "archive_bytes": int(arch_bytes),
        "d_seg": float(d_seg),
        "d_pose": float(d_pose),
        "score": float(sc),
    })
    print(f"[bc{meta['base_channels']:>2} full] d_seg={d_seg:.6e} d_pose={d_pose:.6e} bytes={arch_bytes} S={curve[-1]['score']:.4f}")

    for bc in rungs:
        pruned_sd, _ = prune_decoder_state(decoder_sd, meta, bc)
        pdec = build_decoder(meta, base_channels=bc)
        # verify shapes load
        pdec.load_state_dict(pruned_sd)
        recon = render_recon_pairs(pdec, latents, meta, device)
        d_pose, d_seg = ev.eval_pairs(gt, recon)
        arch_bytes, _ = archive_bytes_for(pruned_sd, latents, meta)
        sc = compute_contest_score(d_seg, d_pose, arch_bytes)
        row = {
            "base_channels": int(bc),
            "kind": "prune_only",
            "decoder_params": count_params(pruned_sd),
            "archive_bytes": int(arch_bytes),
            "d_seg": float(d_seg),
            "d_pose": float(d_pose),
            "score": float(sc),
        }
        curve.append(row)
        print(f"[bc{bc:>2} prune] d_seg={d_seg:.6e} d_pose={d_pose:.6e} bytes={arch_bytes} S={row['score']:.4f}")

    result = {
        "stage": "capacity_rd_prune_curve",
        "evidence_grade": "[contest-CPU advisory]" if device == "cpu" else f"[{device} advisory]",
        "device": str(device),
        "n_pairs": int(n_pairs),
        "sub019_d_seg_threshold": 9.2e-4,
        "sub015_d_seg_threshold": 3.2e-4,
        "curve": curve,
        "wall_s": round(time.time() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result, indent=2))
    return result


def kd_frame_mse(student_frames, teacher_frames):
    """Frame-MSE KD objective (mirror of tac.torch_vehicle.kd_warm_start.kd_frame_mse).

    Per-pixel L2 between student and teacher pair frames, both (B,2,3,H,W) in [0,255].
    Teacher is detached (rendered under no_grad by the caller)."""
    return F.mse_loss(student_frames, teacher_frames.detach())


def run_kd(
    n_pairs,
    device,
    target_bc,
    epochs,
    batch_size,
    lr,
    train_latents,
    gt_cache,
    out_json,
    train_device=None,
):
    """KD-finetune a pruned rung from the FROZEN converged bc36 teacher (warm-start).

    NO FAKE: teacher is the real converged bc36 HNeRVDecoder, eval()+requires_grad_(False);
    student is the pruned bc decoder (warm-started by structured prune, NOT random init);
    the KD step ACTUALLY lowers student-vs-teacher frame-MSE (first_loss vs last_loss prove
    the distillation ran). After KD the rung is re-evaluated on the EXACT SegNet/PoseNet.

    Device split (the canonical MPS-gradient / CPU-authority contract): KD gradient runs on
    ``train_device`` (MPS is 104x faster, a VALID training-gradient device) but the exact
    SegNet/PoseNet eval + final render run on ``device`` (CPU authority). MPS is NEVER a
    score authority.
    """
    from tac.contest_score import compute_contest_score, seg_term, pose_term, rate_term

    t0 = time.time()
    td = torch.device(train_device) if train_device else device
    decoder_sd, latents_full, meta = inflate_pr95(PR95_ARCHIVE)
    gt = decode_gt_pairs(n_pairs, gt_cache)
    n_pairs = gt.shape[0]
    latents = latents_full[:n_pairs].to(td)

    # frozen teacher = converged bc36 (on the TRAIN device)
    teacher = build_decoder(meta).to(td).eval()
    teacher.load_state_dict(decoder_sd)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # warm-started student = structured prune of the teacher (on the TRAIN device)
    pruned_sd, _ = prune_decoder_state(decoder_sd, meta, target_bc)
    student = build_decoder(meta, base_channels=target_bc).to(td)
    student.load_state_dict(pruned_sd)
    student.train()

    lat_param = latents.clone().detach().requires_grad_(bool(train_latents))
    params = [{"params": [p for p in student.parameters() if p.requires_grad], "lr": lr}]
    if train_latents:
        params.append({"params": [lat_param], "lr": lr * 10.0})
    opt = torch.optim.AdamW(params, weight_decay=0.0)

    g = torch.Generator(device="cpu").manual_seed(0)
    first_loss = last_loss = float("nan")
    for ep in range(int(epochs)):
        perm = torch.randperm(n_pairs, generator=g).to(td)
        ep_loss, nb = 0.0, 0
        for bs in range(0, n_pairs, int(batch_size)):
            idx = perm[bs : bs + int(batch_size)]
            with torch.no_grad():
                teacher_frames = teacher(lat_param[idx])
            student_frames = student(lat_param[idx])
            loss = kd_frame_mse(student_frames, teacher_frames)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        mean = ep_loss / max(nb, 1)
        if ep == 0:
            first_loss = mean
        last_loss = mean
        print(f"[kd bc{target_bc} ep{ep} train={td}] frame_mse={mean:.4f}", flush=True)

    # exact eval after KD on the AUTHORITY device (CPU); render on CPU too.
    student.eval()
    student_cpu = build_decoder(meta, base_channels=target_bc).to(device)
    student_cpu.load_state_dict({k: v.detach().cpu() for k, v in student.state_dict().items()})
    student_cpu.eval()
    lat_cpu = lat_param.detach().to(device)
    recon = render_recon_pairs(student_cpu, lat_cpu, meta, device)
    ev = ExactEvaluator(device)
    d_pose, d_seg = ev.eval_pairs(gt, recon)
    kd_sd = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    arch_bytes, _ = archive_bytes_for(kd_sd, lat_cpu, meta)
    sc = compute_contest_score(d_seg, d_pose, arch_bytes)
    result = {
        "stage": "kd_finetune_pruned_rung",
        "evidence_grade": "[contest-CPU advisory]" if str(device) == "cpu" else f"[{device} advisory]",
        "device": str(device),
        "n_pairs": int(n_pairs),
        "base_channels": int(target_bc),
        "kind": "prune_then_kd",
        "train_device": str(td),
        "kd_epochs": int(epochs),
        "kd_first_frame_mse": float(first_loss),
        "kd_last_frame_mse": float(last_loss),
        "kd_distillation_ran": bool(last_loss < first_loss),
        "decoder_params": count_params(kd_sd),
        "archive_bytes": int(arch_bytes),
        "d_seg": float(d_seg),
        "d_pose": float(d_pose),
        "score": float(sc),
        "score_components": {
            "seg": float(seg_term(d_seg)),
            "pose": float(pose_term(d_pose)),
            "rate": float(rate_term(arch_bytes)),
        },
        "sub019_d_seg_threshold": 9.2e-4,
        "sub015_d_seg_threshold": 3.2e-4,
        "wall_s": round(time.time() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sanity", "curve", "kd"])
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rungs", default="32,28,24,20,16")
    ap.add_argument("--target-bc", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-train-latents", action="store_true")
    ap.add_argument(
        "--train-device",
        default=None,
        help="KD-gradient device (e.g. mps); exact eval stays on --device (CPU authority).",
    )
    ap.add_argument("--gt-cache", default=str(REPO / ".omx/tmp/pr95_gt_pairs_600.npy"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    gt_cache = Path(args.gt_cache)
    if args.cmd == "sanity":
        out = Path(args.out or REPO / "experiments/results/reveng_pr95_prune_20260623/sanity.json")
        run_sanity(args.n_pairs, device, gt_cache, out)
    elif args.cmd == "kd":
        out = Path(
            args.out
            or REPO / f"experiments/results/reveng_pr95_prune_20260623/kd_bc{args.target_bc}.json"
        )
        run_kd(
            args.n_pairs,
            device,
            args.target_bc,
            args.epochs,
            args.batch_size,
            args.lr,
            not args.no_train_latents,
            gt_cache,
            out,
            train_device=args.train_device,
        )
    else:
        rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
        out = Path(args.out or REPO / "experiments/results/reveng_pr95_prune_20260623/curve.json")
        run_curve(args.n_pairs, device, rungs, gt_cache, out)


if __name__ == "__main__":
    main()
