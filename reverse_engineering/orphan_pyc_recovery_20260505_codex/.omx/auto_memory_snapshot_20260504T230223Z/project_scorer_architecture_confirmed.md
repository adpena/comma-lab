---
name: Scorer Architecture Confirmed
description: SegNet = EfficientNet-B4 U-Net (smp), PoseNet uses YUV6 internally. DDELab confirms no forensic-grade detection needed.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## SegNet (confirmed from comma10k-baseline + upstream modules.py)
- **Architecture**: `smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet', classes=6, activation=None)`
- **Preprocessing**: `x[:, -1, ...]` (last frame), bilinear resize to (384, 512). Raw RGB — no YUV, no normalization in preprocess_input.
- **Classes**: 6 values {0=padding, 41=road, 76=lane, 90=undrivable, 124=movable, 161=ego}
- **Training**: CrossEntropyLoss, Adam lr=1e-4, with blur/noise augmentation but NOT compression artifacts
- **Key**: Class boundaries are most vulnerable. Compression artifacts near road/lane boundaries cause most disagreement.

## PoseNet (confirmed from upstream modules.py)
- **Preprocessing**: rearrange (B,T,C,H,W)→(BT,C,H,W), bilinear resize to (384,512), then **rgb_to_yuv6** → (B, T*6, 192, 256)
- **Output**: dict with 'pose' key, shape (..., 6+) — first 6 are [tx, ty, tz, rx, ry, rz]
- **Operates in YUV420 space internally**

## DDELab/deepsteganalysis (Fridrich + Yousfi's lab)
- SRNet, OneHotConv, EfficientNet with `nostride` surgery for forensic detection
- **Key finding**: comma scorers are DRIVING models, not forensic detectors. No nostride surgery. Focus on semantic/geometric fidelity, not pixel-level statistical anomalies.
- Our generated frames don't need to be forensically perfect — geometric accuracy matters more.

## Openpilot PRs (#35240, #35816)
- Vision model 35MB, Policy model 16MB — split architecture
- Policy outputs velocity, curvature, acceleration — PoseNet-adjacent behavior
- Vision model unchanged across PRs — same feature extraction backbone
