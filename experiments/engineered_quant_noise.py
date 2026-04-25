#!/usr/bin/env python3
"""Engineered Quantization Noise: flip SegNet argmax via targeted perturbation.

Eureka 6: SegNet distortion = argmax disagreement. Small pixel perturbations
(+-1..2) can flip WRONG predictions to CORRECT, reducing distortion. Finds such
perturbations at compress time; stores as gradient_corrections.bin for inflate.

Usage:
    PYTHONPATH=src:upstream python experiments/engineered_quant_noise.py \
        --checkpoint path/to/renderer_best.pt [--device cuda|mps] [--smoke]
"""
from __future__ import annotations
import argparse, json, os, struct, sys, time, zlib
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

# Bootstrap project root onto sys.path so `from experiments.X` imports work
# regardless of cwd (R41 fix: subprocess invocation from pipeline.py only adds
# experiments/ to sys.path, not project root, so 'from experiments.X' would
# silently fail at runtime).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

for _p in [Path(os.environ.get("TAC_UPSTREAM_DIR", "")),
           Path(os.environ.get("UPSTREAM_ROOT", "")),
           Path(__file__).resolve().parent.parent / "upstream"]:
    if _p.name and (_p / "modules.py").exists():
        if str(_p) not in sys.path: sys.path.insert(0, str(_p))
        UPSTREAM_ROOT = _p; break
else:
    UPSTREAM_ROOT = None

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "engineered_quant"


def segnet_forward(frames_hwc, segnet, device, bs=32):
    """(N,H,W,3) float -> (masks:(N,sH,sW) long, logits:(N,5,sH,sW))."""
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    masks_l, logits_l = [], []
    for i in range(0, frames_hwc.shape[0], bs):
        b = frames_hwc[i:i+bs].to(device).permute(0,3,1,2).contiguous()
        if b.shape[2] != SEGNET_INPUT_H or b.shape[3] != SEGNET_INPUT_W:
            b = F.interpolate(b, (SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
        with torch.no_grad():
            lg = segnet(segnet.preprocess_input(b.unsqueeze(1)))
        masks_l.append(lg.argmax(1).cpu()); logits_l.append(lg.cpu())
    return torch.cat(masks_l).long(), torch.cat(logits_l)


def find_perturbations(frames_hwc, gt_masks, our_masks, segnet, device, max_delta=2, bs=32):
    """Gradient-guided search for small perturbations that flip wrong predictions."""
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    N, H, W, C = frames_hwc.shape
    disagree = (our_masks != gt_masks)
    n_dis = disagree.sum().item()
    if n_dis == 0:
        return np.array([], dtype=np.uint32), np.zeros((0,3), dtype=np.int8)
    print(f"  Disagreeing: {n_dis:,}/{N*H*W:,} ({n_dis/(N*H*W)*100:.2f}%)")
    all_idx, all_dlt = [], []
    fp = 0  # frames processed counter
    for i in range(0, N, bs):
        end = min(i+bs, N)
        bh = frames_hwc[i:end].clone().to(device).requires_grad_(True)
        bg, bp, bd = gt_masks[i:end].to(device), our_masks[i:end].to(device), disagree[i:end].to(device)
        if not bd.any():
            fp += end-i; continue
        bc = bh.permute(0,3,1,2).contiguous()
        if bc.shape[2] != SEGNET_INPUT_H or bc.shape[3] != SEGNET_INPUT_W:
            bc = F.interpolate(bc, (SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
        lg = segnet(segnet.preprocess_input(bc.unsqueeze(1)))
        margin = lg.gather(1, bg.unsqueeze(1)).squeeze(1) - lg.gather(1, bp.unsqueeze(1)).squeeze(1)
        (margin * bd.float()).sum().backward()
        grad = bh.grad.detach()
        for j in range(end-i):
            fd = bd[j]
            if not fd.any(): continue
            fg = grad[j].cpu()
            direction = fg / fg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            delta = (direction * max_delta).round().clamp(-max_delta, max_delta).to(torch.int8)
            coords = fd.nonzero(as_tuple=False)
            r, c = coords[:,0], coords[:,1]
            pd = delta[r, c]
            flat = (fp+j) * H * W + r.long() * W + c.long()
            nz = pd.abs().sum(-1) > 0
            if nz.any():
                all_idx.append(flat[nz].numpy().astype(np.uint32))
                all_dlt.append(pd[nz].numpy().astype(np.int8))
        fp += end-i
    if not all_idx:
        return np.array([], dtype=np.uint32), np.zeros((0,3), dtype=np.int8)
    return np.concatenate(all_idx), np.concatenate(all_dlt)


def pack_corrections(indices, deltas, shape, max_delta, qbits=8):
    """Pack into gradient_corrections.bin format (inflate_renderer compatible).

    Dequant formula: val / 127.0 * scale. We set scale=max_delta and encode
    delta as round(delta / max_delta * 127) so dequant recovers the delta.
    """
    n_total = shape[0] * shape[1] * shape[2]
    divisor = 127.0 if qbits == 8 else 7.0
    enc = (deltas.astype(np.float32) / max_delta * divisor).round().clip(-divisor, divisor).astype(np.int8)
    header = json.dumps({"scale": float(max_delta), "shape": list(shape[:4]),
                         "top_k_pct": len(indices)/n_total*100, "quantize_bits": qbits,
                         "n_kept": len(indices), "n_total": n_total}).encode()
    data = struct.pack("<I", len(header)) + header + indices.tobytes() + enc.tobytes()
    return zlib.compress(data, level=9)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--n-frames", type=int, default=1200); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-delta", type=int, default=2); p.add_argument("--output-dir", default=None)
    p.add_argument("--video", default=None); p.add_argument("--smoke", action="store_true")
    p.add_argument("--gt-poses-path", default=None); p.add_argument("--quantize-bits", type=int, default=8)
    a = p.parse_args()
    if a.smoke: a.n_frames = 20
    a.n_frames -= a.n_frames % 2
    n_pairs = a.n_frames // 2
    device = torch.device(a.device)
    if UPSTREAM_ROOT is None: sys.exit("ERROR: Cannot find upstream. Set TAC_UPSTREAM_DIR.")
    out = Path(a.output_dir or str(RESULTS_DIR / f"eq_{time.strftime('%Y%m%dT%H%M%S')}"))
    out.mkdir(parents=True, exist_ok=True)
    video = a.video or str(UPSTREAM_ROOT / "videos" / "0.mkv")
    print(f"[config] device={device}, frames={a.n_frames}, delta={a.max_delta}")
    t0 = time.monotonic()

    # Load scorer + renderer
    from tac.scorer import load_differentiable_scorers, extract_gt_masks, extract_gt_pose_targets
    from tac.data import load_gt_video
    from experiments.precompute_gradient_corrections import load_renderer
    posenet, segnet = load_differentiable_scorers(UPSTREAM_ROOT, device=str(device))
    renderer = load_renderer(a.checkpoint, device)
    gt_frames = load_gt_video(video, n_frames=a.n_frames)
    a.n_frames = len(gt_frames); n_pairs = a.n_frames // 2

    # GT masks + render
    gt_masks = extract_gt_masks(gt_frames, segnet, device)
    H, W = gt_masks.shape[1], gt_masks.shape[2]
    poses = None
    if getattr(renderer, "pose_dim", 0) > 0:
        pt = extract_gt_pose_targets(gt_frames, posenet, device)
        poses = (torch.load(a.gt_poses_path, map_location="cpu", weights_only=True).float()[:n_pairs]
                 if a.gt_poses_path and Path(a.gt_poses_path).exists() else pt[:n_pairs].clone())
    rendered = []
    bp = max(1, min(a.batch_size // 2, n_pairs))
    for bi in range(0, n_pairs, bp):
        be = min(bi+bp, n_pairs); fs = 2*bi; fe = 2*be
        mt, mt1 = gt_masks[fs:fe:2].to(device), gt_masks[fs+1:fe+1:2].to(device)
        kw = {"pose": poses[bi:be].to(device)} if poses is not None else {}
        with torch.no_grad():
            pr = renderer(mt, mt1, **kw); f0, f1 = pr[:,0], pr[:,1]
            rendered.append(torch.stack([f0,f1],1).reshape(-1,*f0.shape[1:]).cpu())
    rend = torch.cat(rendered).float()
    print(f"  Rendered {rend.shape[0]} frames at {rend.shape[1]}x{rend.shape[2]}")

    # SegNet on rendered, find perturbations
    our_masks, our_logits = segnet_forward(rend, segnet, device, a.batch_size)
    ntot = our_masks.numel(); ndis = (our_masks != gt_masks).sum().item()
    print(f"  Baseline disagreement: {ndis:,}/{ntot:,} ({ndis/ntot*100:.2f}%)")

    indices, deltas = find_perturbations(rend, gt_masks, our_masks, segnet, device, a.max_delta, a.batch_size)
    print(f"  Perturbations found: {len(indices):,}")
    if len(indices) == 0: print("  Nothing to fix."); return

    packed = pack_corrections(indices, deltas, tuple(rend.shape), a.max_delta, a.quantize_bits)
    (out / "gradient_corrections.bin").write_bytes(packed)
    print(f"  Packed: {len(packed):,} bytes ({len(packed)/1024:.1f} KB), rate={len(packed)/37_545_489:.6f}")

    # Validate
    corr = rend.clone().reshape(-1, 3)
    corr[indices.astype(np.int64)] += torch.from_numpy(deltas.astype(np.float32))
    corr.clamp_(0, 255)
    new_masks, _ = segnet_forward(corr.reshape(rend.shape), segnet, device, a.batch_size)
    ndis2 = (new_masks != gt_masks).sum().item(); fixed = ndis - ndis2
    print(f"  After: {ndis2:,} disagree ({ndis2/ntot*100:.2f}%), fixed {fixed:,} ({fixed/max(ndis,1)*100:.1f}%)")
    print(f"  SegNet distortion: {ndis/ntot*100:.4f}% -> {ndis2/ntot*100:.4f}%")

    (out / "summary.json").write_text(json.dumps({
        "config": vars(a), "baseline_disagree": ndis, "after_disagree": ndis2,
        "fixed": fixed, "total_pixels": ntot, "n_perturbations": len(indices),
        "packed_bytes": len(packed), "rate_cost": len(packed)/37_545_489,
        "time_s": time.monotonic()-t0}, indent=2, default=str))
    print(f"  Time: {time.monotonic()-t0:.1f}s | Results: {out}")


if __name__ == "__main__":
    main()
