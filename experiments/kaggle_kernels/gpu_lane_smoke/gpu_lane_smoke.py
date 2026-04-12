"""GPU Lane Smoke Test — Constrained Optimization on Kaggle GPU.

Fixes from v1:
- Install P100-compatible PyTorch (sm_60 support)
- Use DistortionNet.load_state_dicts() not PoseNet.load()
- Handles both T4 and P100

EXPERIMENT METADATA
-------------------
type: gpu_lane_smoke_test
platform: kaggle_p100
frames: 8 (NOT 1200 — P100 16GB VRAM memory constraint)
steps: 100 (NOT 1000 — Kaggle kernel time constraint)
purpose: Viability test — does constrained generation work on real GPU hardware?
         Results are DIRECTIONAL ONLY. This is NOT a score.
         A proxy < 5.0 means the approach is viable and worth scaling up.
"""
import subprocess, sys, os, time

# Install P100-compatible PyTorch first (sm_60 = compute capability 6.0)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch==2.6.0", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "safetensors", "timm", "einops", "segmentation-models-pytorch", "av"])

if not os.path.exists("/kaggle/working/upstream"):
    subprocess.check_call(["git", "clone", "--depth", "1",
        "https://github.com/commaai/comma_video_compression_challenge.git",
        "/kaggle/working/upstream"])
    subprocess.check_call(["git", "lfs", "pull"], cwd="/kaggle/working/upstream")

sys.path.insert(0, "/kaggle/working/upstream")

import torch
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA cap: {torch.cuda.get_device_capability(0)}")

# Load scorer via DistortionNet (the correct API)
from modules import DistortionNet
distortion_net = DistortionNet().eval().to(device)
distortion_net.load_state_dicts(
    "/kaggle/working/upstream/models/posenet.safetensors",
    "/kaggle/working/upstream/models/segnet.safetensors",
    device=device,
)
posenet = distortion_net.posenet
segnet = distortion_net.segnet
print("Scorer loaded")

# Load 20 frames for smoke test
import av
container = av.open("/kaggle/working/upstream/videos/0.mkv")
stream = container.streams.video[0]
frames = []
for packet in container.demux(stream):
    for frame in packet.decode():
        frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).float())
        if len(frames) >= 8: break
    if len(frames) >= 8: break
container.close()

frames_t = torch.stack(frames).to(device)
N = frames_t.shape[0]
print(f"Loaded {N} frames: {frames_t.shape}")

# Extract GT masks + poses
with torch.no_grad():
    masks = []
    for i in range(N):
        f_btchw = frames_t[i].permute(2,0,1).unsqueeze(0).unsqueeze(0).contiguous()
        seg_in = segnet.preprocess_input(f_btchw)
        masks.append(segnet(seg_in).argmax(dim=1).squeeze(0))
    masks = torch.stack(masks)

    gt_poses = []
    for i in range(N-1):
        pair = torch.stack([frames_t[i], frames_t[i+1]]).unsqueeze(0)
        pair_chw = pair.permute(0,1,4,2,3).contiguous()
        pose_in = posenet.preprocess_input(pair_chw)
        gt_poses.append(posenet(pose_in)["pose"][..., :6].squeeze(0))
    gt_poses = torch.stack(gt_poses)

print(f"Masks: {masks.shape}, GT poses: {gt_poses.shape}")

# Initialize from class colors + noise
H, W = frames_t.shape[1], frames_t.shape[2]
class_colors = torch.tensor([[128,128,128],[170,170,170],[100,80,60],[120,140,160],[180,200,230]],
    dtype=torch.float32, device=device)
masks_full = F.interpolate(masks.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1).long()
gen = class_colors[masks_full] + torch.randn(N,H,W,3,device=device) * 10.0
gen = gen.clamp(0,255).requires_grad_(True)

torch.cuda.empty_cache()
optimizer = torch.optim.Adam([gen], lr=1.0)
t0 = time.time()

for step in range(100):
    optimizer.zero_grad()
    seg_loss = torch.tensor(0.0, device=device)
    for i in range(0, N, 2):
        b = gen[i:i+4].permute(0,3,1,2).unsqueeze(1).contiguous()
        logits = segnet(segnet.preprocess_input(b))
        H_o, W_o = logits.shape[2], logits.shape[3]
        tgt = F.interpolate(masks[i:i+4].float().unsqueeze(1), size=(H_o,W_o), mode="nearest").squeeze(1).long()
        seg_loss = seg_loss + F.cross_entropy(logits, tgt)

    pose_loss = torch.tensor(0.0, device=device)
    for i in range(min(N-1, 4)):
        pair = torch.stack([gen[i], gen[i+1]]).unsqueeze(0).permute(0,1,4,2,3).contiguous()
        pred = posenet(posenet.preprocess_input(pair))["pose"][..., :6]
        pose_loss = pose_loss + (pred - gt_poses[i:i+1]).pow(2).mean()

    tv = (gen[:,1:,:,:]-gen[:,:-1,:,:]).abs().mean() + (gen[:,:,1:,:]-gen[:,:,:-1,:]).abs().mean()
    total = 100.0 * seg_loss + (10.0 * pose_loss + 1e-8).sqrt() + 0.1 * tv
    total.backward()
    optimizer.step()
    with torch.no_grad(): gen.data.clamp_(0, 255)

    if (step+1) % 10 == 0:
        print(f"  step {step+1:3d}: total={total.item():.3f} seg={seg_loss.item():.4f} pose={pose_loss.item():.4f} tv={tv.item():.0f} [{time.time()-t0:.0f}s]")

# Evaluate
print("\n=== EVALUATION ===")
with torch.no_grad():
    seg_d, pose_d, n_p = 0.0, 0.0, 0
    for i in range(N):
        g = gen[i].permute(2,0,1).unsqueeze(0).unsqueeze(0).contiguous()
        o = frames_t[i].permute(2,0,1).unsqueeze(0).unsqueeze(0).contiguous()
        gs = F.softmax(segnet(segnet.preprocess_input(g)), dim=1)
        os_ = F.softmax(segnet(segnet.preprocess_input(o)), dim=1)
        seg_d += 1.0 - (gs * os_).sum(dim=1).mean().item()
    seg_d /= N
    for i in range(N-1):
        gp = torch.stack([gen[i],gen[i+1]]).unsqueeze(0).permute(0,1,4,2,3).contiguous()
        op = torch.stack([frames_t[i],frames_t[i+1]]).unsqueeze(0).permute(0,1,4,2,3).contiguous()
        gpose = posenet(posenet.preprocess_input(gp))["pose"][..., :6]
        opose = posenet(posenet.preprocess_input(op))["pose"][..., :6]
        pose_d += (gpose - opose).pow(2).mean().item()
        n_p += 1
    pose_d /= max(n_p, 1)

proxy = 100*seg_d + (10*pose_d)**0.5
print(f"  SegNet:  {seg_d:.6f}")
print(f"  PoseNet: {pose_d:.6f}")
print(f"  Proxy (no rate): {proxy:.4f}")
print(f"  Time: {time.time()-t0:.0f}s")
print(f"\n  {'GPU LANE VIABLE!' if proxy < 5.0 else 'Needs work'}")
