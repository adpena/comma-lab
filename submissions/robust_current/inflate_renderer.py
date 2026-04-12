#!/usr/bin/env python
"""Inflate path using a trained DP-SIMS neural renderer.

The renderer generates RGB frames purely from SegNet masks extracted from
the ground-truth video.  No compressed video is stored — only the renderer
weights (~200KB) are in the archive.  This is the ultimate rate-quality
tradeoff: fixed rate regardless of content complexity.

Pipeline:
    GT video  ->  SegNet (upstream)  ->  masks (384x512)
    masks     ->  DPSIMSRenderer     ->  frames (384x512)
    frames    ->  bilinear upscale   ->  raw RGB (1164x874)

Architecture classes (SPADE, SPADEResBlock, DPSIMSRenderer) are inlined
for standalone operation on scorer machines without the tac package.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import av


# ============================================================
# Constants
# ============================================================
OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES  # 3,656,649,600


# ============================================================
# Canonical YUV->RGB (BT.601 limited range, matches frame_utils.py)
# Copied from inflate_postfilter.py — must stay identical.
# ============================================================
def yuv420_to_rgb(frame) -> torch.Tensor:
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


# ============================================================
# Inline DPSIMSRenderer (forward-only, no training code)
# Self-contained fallback for scorer machines without tac.
# ============================================================
try:
    from tac.dp_sims_renderer import SPADE, SPADEResBlock, CrossAttentionNoiseInjector, DPSIMSRenderer
    _TAC_RENDERER_AVAILABLE = True
except ImportError:
    _TAC_RENDERER_AVAILABLE = False

    class SPADE(nn.Module):
        """Spatially-Adaptive Normalization (Park et al., CVPR 2019)."""

        def __init__(self, norm_channels: int, mask_channels: int = 5, hidden: int = 64):
            super().__init__()
            self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
            self.mask_channels = mask_channels
            self.shared = nn.Sequential(
                nn.Conv2d(mask_channels, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.gamma_conv = nn.Conv2d(hidden, norm_channels, 3, padding=1)
            self.beta_conv = nn.Conv2d(hidden, norm_channels, 3, padding=1)
            nn.init.zeros_(self.gamma_conv.weight)
            nn.init.zeros_(self.gamma_conv.bias)
            nn.init.zeros_(self.beta_conv.weight)
            nn.init.zeros_(self.beta_conv.bias)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            normalized = self.norm(x)
            _, _, fH, fW = x.shape
            mask_onehot = self._encode_mask(mask, fH, fW, x.device)
            shared = self.shared(mask_onehot)
            gamma = self.gamma_conv(shared)
            beta = self.beta_conv(shared)
            return normalized * (1.0 + gamma) + beta

        def _encode_mask(self, mask: torch.Tensor, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
            B = mask.shape[0]
            if mask.shape[1] != target_h or mask.shape[2] != target_w:
                mask_resized = (
                    F.interpolate(mask.unsqueeze(1).float(), size=(target_h, target_w), mode="nearest")
                    .squeeze(1).long()
                )
            else:
                mask_resized = mask
            onehot = torch.zeros(B, self.mask_channels, target_h, target_w, device=device, dtype=torch.float32)
            onehot.scatter_(1, mask_resized.unsqueeze(1), 1.0)
            return onehot

    class SPADEResBlock(nn.Module):
        """Residual block with SPADE normalization."""

        def __init__(self, in_channels: int, out_channels: int, mask_channels: int = 5, spade_hidden: int = 64):
            super().__init__()
            self.learned_skip = in_channels != out_channels
            self.spade1 = SPADE(in_channels, mask_channels, hidden=spade_hidden)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.spade2 = SPADE(out_channels, mask_channels, hidden=spade_hidden)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.act = nn.ReLU(inplace=True)
            if self.learned_skip:
                self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            nn.init.zeros_(self.conv2.weight)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.spade1(x, mask)
            h = self.act(h)
            h = self.conv1(h)
            h = self.spade2(h, mask)
            h = self.act(h)
            h = self.conv2(h)
            if self.learned_skip:
                x = self.skip_conv(x)
            return x + h

    class CrossAttentionNoiseInjector(nn.Module):
        """Cross-attention noise injection for texture diversity."""

        def __init__(self, channels: int, mask_channels: int = 5, noise_dim: int = 16):
            super().__init__()
            self.channels = channels
            self.mask_channels = mask_channels
            self.noise_dim = noise_dim
            self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
            self.noise_proj = nn.Conv2d(noise_dim + mask_channels, channels, 1, bias=False)
            self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
            self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
            self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
            self.gate = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        def forward(self, x: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
            import math
            B, C, H, W = x.shape
            if noise is None:
                noise = torch.randn(B, self.noise_dim, H, W, device=x.device, dtype=x.dtype)
            if mask.shape[1] != H or mask.shape[2] != W:
                mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).long()
            else:
                mask_resized = mask
            mask_onehot = torch.zeros(B, self.mask_channels, H, W, device=x.device, dtype=x.dtype)
            mask_onehot.scatter_(1, mask_resized.unsqueeze(1), 1.0)
            noise_mask = torch.cat([noise, mask_onehot], dim=1)
            noise_features = self.noise_proj(noise_mask)
            q = self.to_q(x)
            k = self.to_k(noise_features)
            v = self.to_v(noise_features)
            scale = math.sqrt(C)
            attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / scale)
            attended = attn * v
            out = self.out_proj(attended)
            return x + self.gate * out

    class DPSIMSRenderer(nn.Module):
        """SPADE-based progressive generator for mask-to-RGB synthesis."""

        def __init__(
            self,
            num_classes: int = 5,
            channels: tuple[int, ...] = (256, 128, 64, 32),
            init_h: int = 24,
            init_w: int = 32,
            spade_hidden: int = 64,
            noise_dim: int = 16,
            use_noise: bool = True,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.init_h = init_h
            self.init_w = init_w
            self.use_noise = use_noise
            self.num_stages = len(channels)
            self.const = nn.Parameter(torch.randn(1, channels[0], init_h, init_w) * 0.02)
            self.spade_blocks = nn.ModuleList()
            self.noise_injectors = nn.ModuleList()
            in_ch = channels[0]
            for i, out_ch in enumerate(channels):
                sh = max(32, min(spade_hidden, out_ch))
                self.spade_blocks.append(SPADEResBlock(in_ch, out_ch, num_classes, spade_hidden=sh))
                if use_noise:
                    self.noise_injectors.append(CrossAttentionNoiseInjector(out_ch, num_classes, noise_dim))
                in_ch = out_ch
            self.final_upsample = nn.ConvTranspose2d(channels[-1], channels[-1], 4, stride=2, padding=1, bias=False)
            self.head = nn.Conv2d(channels[-1], 3, 3, padding=1, bias=True)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, masks: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
            B = masks.shape[0]
            x = self.const.expand(B, -1, -1, -1)
            for i, block in enumerate(self.spade_blocks):
                x = block(x, masks)
                if self.use_noise and i < len(self.noise_injectors):
                    x = self.noise_injectors[i](x, masks)
                if i < self.num_stages - 1:
                    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            _, _, cur_h, cur_w = x.shape
            target_h, target_w = masks.shape[1], masks.shape[2]
            if cur_h != target_h or cur_w != target_w:
                x = self.final_upsample(x)
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            rgb = 255.0 * torch.sigmoid(self.head(x) / 50.0)
            return rgb


# ============================================================
# Upstream discovery
# ============================================================
def _find_upstream_root(archive_dir: str) -> Path:
    """Locate the upstream directory containing modules.py and models/.

    Search order:
        1. archive_dir/../../  (scorer environment: archive/ is 2 levels deep)
        2. <script_dir>/../../upstream/  (local dev layout)
        3. UPSTREAM_ROOT env var
    """
    candidates = []

    # 1. Scorer environment layout
    candidates.append(Path(archive_dir).resolve().parent.parent)

    # 2. Local dev layout
    candidates.append(Path(__file__).resolve().parent.parent.parent / "upstream")

    # 3. Environment variable
    env_root = os.environ.get("UPSTREAM_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    # Also check COMMA_CHALLENGE_ROOT (used by inflate_postfilter.py)
    env_root2 = os.environ.get("COMMA_CHALLENGE_ROOT")
    if env_root2:
        candidates.append(Path(env_root2))

    for candidate in candidates:
        if not candidate.exists():
            continue
        modules_py = candidate / "modules.py"
        models_dir = candidate / "models"
        if modules_py.exists() and models_dir.exists():
            return candidate

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find upstream root (need modules.py + models/ dir).\n"
        f"Tried:\n  {tried}\n"
        f"Set UPSTREAM_ROOT or COMMA_CHALLENGE_ROOT env var."
    )


# ============================================================
# SegNet loading
# ============================================================
def _load_segnet(upstream_root: Path, device: str) -> nn.Module:
    """Load frozen SegNet from upstream for mask extraction."""
    t0 = time.monotonic()

    # Import SegNet from upstream modules.py
    upstream_str = str(upstream_root)
    sys.path.insert(0, upstream_str)
    try:
        from modules import SegNet
    finally:
        # Remove the exact entry we inserted at position 0
        try:
            sys.path.pop(sys.path.index(upstream_str))
        except ValueError:
            pass  # already removed

    segnet = SegNet()
    segnet_path = upstream_root / "models" / "segnet.safetensors"
    if not segnet_path.exists():
        raise FileNotFoundError(f"SegNet weights not found: {segnet_path}")

    from safetensors.torch import load_file
    sd = load_file(str(segnet_path), device=device)
    segnet.load_state_dict(sd)
    segnet.to(device).eval()

    # Freeze all parameters
    for p in segnet.parameters():
        p.requires_grad = False

    elapsed = time.monotonic() - t0
    print(f"  SegNet loaded from {segnet_path} ({elapsed:.1f}s)", file=sys.stderr)
    return segnet


# ============================================================
# GT video decoding
# ============================================================
def _decode_gt_video(mkv_path: str) -> list[np.ndarray]:
    """Decode ground-truth video via PyAV.

    Returns list of (H, W, 3) uint8ndarrays in RGB order.
    Uses yuv420_to_rgb for BT.601 limited-range decode matching the scorer.
    """
    t0 = time.monotonic()
    container = av.open(mkv_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8 tensor
        frames.append(rgb.numpy())
    container.close()
    elapsed = time.monotonic() - t0
    print(f"  Decoded {len(frames)} GT frames from {mkv_path} ({elapsed:.1f}s)", file=sys.stderr)
    return frames


# ============================================================
# Mask extraction
# ============================================================
def _extract_masks(
    frames: list[np.ndarray],
    segnet: nn.Module,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Extract SegNet masks from GT frames.

    Args:
        frames: list of (H, W, 3) uint8 ndarrays
        segnet: frozen SegNet module
        device: torch device string
        batch_size: inference batch size

    Returns:
        (N, 384, 512) long tensor of class indices in [0, 4]
    """
    t0 = time.monotonic()
    N = len(frames)
    masks_list = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            # Stack frames -> (B, H, W, 3) uint8 -> (B, 3, H, W) float
            batch_np = np.stack(frames[i:end], axis=0)  # (B, H, W, 3)
            batch_t = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            # SegNet expects (B, 1, 3, H, W) for preprocess_input
            inp = batch_t.unsqueeze(1)  # (B, 1, 3, H, W)
            seg_in = segnet.preprocess_input(inp)  # (B, 3, 384, 512)
            logits = segnet(seg_in)  # (B, 5, 384, 512)
            mask = logits.argmax(dim=1)  # (B, 384, 512)
            masks_list.append(mask.cpu())

            if (i + batch_size) % (batch_size * 10) == 0 or end == N:
                print(f"    Masks: {end}/{N} frames", file=sys.stderr, flush=True)

    masks = torch.cat(masks_list, dim=0)  # (N, 384, 512)
    elapsed = time.monotonic() - t0
    print(f"  Extracted {masks.shape[0]} masks ({elapsed:.1f}s)", file=sys.stderr)
    return masks


# ============================================================
# Renderer loading
# ============================================================
def _load_renderer(renderer_path: str, device: str) -> nn.Module:
    """Load DPSIMSRenderer from a .bin checkpoint.

    Supports two checkpoint formats:
        1. Standalone renderer state_dict (keys start with spade_blocks, head, etc.)
        2. PairGenerator checkpoint with model_state_dict containing renderer.* prefixed keys

    Config metadata (channels, init_h, init_w, etc.) is read from the
    checkpoint's 'config' key if present, otherwise defaults are used.
    """
    t0 = time.monotonic()
    # weights_only=False because config dict contains Python tuples/dicts
    # that the restricted unpickler rejects. This is a trusted checkpoint
    # from our own archive.zip.
    ckpt = torch.load(renderer_path, map_location=device, weights_only=False)

    # Extract config for architecture reconstruction
    config = ckpt.get("config", {})
    num_classes = config.get("num_classes", 5)
    channels = config.get("channels", (256, 128, 64, 32))
    if isinstance(channels, list):
        channels = tuple(channels)
    init_h = config.get("init_h", 24)
    init_w = config.get("init_w", 32)
    spade_hidden = config.get("spade_hidden", 64)
    noise_dim = config.get("noise_dim", 16)
    use_noise = config.get("use_noise", True)

    print(f"  Renderer config: classes={num_classes}, channels={channels}, "
          f"init={init_h}x{init_w}, spade_hidden={spade_hidden}, "
          f"noise={use_noise}", file=sys.stderr)

    renderer = DPSIMSRenderer(
        num_classes=num_classes,
        channels=channels,
        init_h=init_h,
        init_w=init_w,
        spade_hidden=spade_hidden,
        noise_dim=noise_dim,
        use_noise=use_noise,
    )

    # Determine which state_dict to use
    if "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        # Assume the checkpoint IS the state_dict
        raw_sd = ckpt

    # Check if keys are prefixed with "renderer." (from DPSIMSPairGenerator)
    renderer_prefix = "renderer."
    has_prefix = any(k.startswith(renderer_prefix) for k in raw_sd.keys())

    if has_prefix:
        # Extract only renderer.* keys, strip prefix
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith(renderer_prefix):
                sd[k[len(renderer_prefix):]] = v
        print(f"  Extracted {len(sd)} renderer keys from PairGenerator checkpoint", file=sys.stderr)
    else:
        sd = raw_sd

    # Load weights
    renderer.load_state_dict(sd, strict=True)
    renderer.to(device).eval()

    # Freeze all parameters
    for p in renderer.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in renderer.parameters())
    elapsed = time.monotonic() - t0
    print(f"  Renderer loaded: {n_params:,} params ({elapsed:.1f}s)", file=sys.stderr)
    return renderer


# ============================================================
# Frame generation + write
# ============================================================
def _generate_and_write(
    masks: torch.Tensor,
    renderer: nn.Module,
    output_path: str,
    device: str,
    batch_size: int,
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> int:
    """Generate frames from masks via renderer, upscale, and write raw RGB.

    Args:
        masks: (N, 384, 512) long tensor
        renderer: DPSIMSRenderer
        output_path: path to output .raw file
        device: torch device string
        batch_size: inference batch size
        out_h: output frame height
        out_w: output frame width

    Returns:
        Number of frames written
    """
    t0 = time.monotonic()
    N = masks.shape[0]
    n_written = 0

    with open(output_path, 'wb') as f:
        with torch.inference_mode():
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                batch_masks = masks[i:end].to(device)

                # Generate frames at SegNet resolution (384x512)
                # Note: noise kwarg is dead code in DPSIMSRenderer.forward —
                # noise injectors are called without it. Use manual seed for
                # reproducibility if use_noise=True (training default is False).
                frames = renderer(batch_masks)  # (B, 3, 384, 512)

                # Upscale to output resolution
                frames_up = F.interpolate(
                    frames, size=(out_h, out_w),
                    mode="bilinear", align_corners=False,
                )  # (B, 3, out_h, out_w)

                # Quantize and write as HWC uint8
                frames_uint8 = frames_up.round().clamp(0, 255).to(torch.uint8)
                frames_hwc = frames_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                f.write(frames_hwc.tobytes())
                n_written += B

                if end % (batch_size * 10) == 0 or end == N:
                    print(f"    Generated: {end}/{N} frames", file=sys.stderr, flush=True)

    elapsed = time.monotonic() - t0
    raw_size = os.path.getsize(output_path)
    print(f"  Generated {n_written} frames -> {output_path} "
          f"({raw_size:,} bytes, {elapsed:.1f}s)", file=sys.stderr)
    return n_written


# ============================================================
# Main inflate function
# ============================================================
def inflate_renderer(
    archive_dir: str,
    inflated_dir: str,
    video_names_file: str,
    renderer_filename: str = "renderer.bin",
    out_w: int = OUT_W,
    out_h: int = OUT_H,
) -> None:
    """Full inflate pipeline: GT video -> masks -> renderer -> raw RGB.

    Args:
        archive_dir: directory containing renderer.bin
        inflated_dir: output directory for .raw files
        video_names_file: text file listing video names (one per line)
        renderer_filename: renderer checkpoint filename within archive_dir
        out_w: output frame width
        out_h: output frame height
    """
    t_total_start = time.monotonic()

    # ---- Device detection ----
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 16
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})", file=sys.stderr)
    else:
        device = "cpu"
        batch_size = 4
        print(f"Device: CPU ({os.cpu_count()} cores)", file=sys.stderr)

    # ---- Upstream discovery ----
    print("Stage 1: Discovering upstream environment ...", file=sys.stderr)
    upstream_root = _find_upstream_root(archive_dir)
    print(f"  Upstream root: {upstream_root}", file=sys.stderr)

    # ---- Load SegNet ----
    print("Stage 2: Loading SegNet ...", file=sys.stderr)
    segnet = _load_segnet(upstream_root, device)

    # ---- Load renderer ----
    print("Stage 3: Loading renderer ...", file=sys.stderr)
    renderer_path = Path(archive_dir) / renderer_filename
    if not renderer_path.exists():
        raise FileNotFoundError(
            f"Renderer not found: {renderer_path}\n"
            f"Expected {renderer_filename} inside archive directory."
        )
    renderer = _load_renderer(str(renderer_path), device)

    # ---- Process each video ----
    output_path = Path(inflated_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_names = Path(video_names_file).read_text().splitlines()
    video_names = [v.strip() for v in video_names if v.strip()]

    for idx, rel in enumerate(video_names):
        t_video_start = time.monotonic()
        stem = rel.rsplit(".", 1)[0]
        raw_out = output_path / f"{stem}.raw"
        raw_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Video {idx+1}/{len(video_names)}: {rel}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Find GT video: look in the upstream/scorer data directory
        # The scorer provides GT videos alongside the archive for evaluation
        gt_candidates = [
            Path(archive_dir).parent / rel,  # scorer layout: data/<video>.mkv
            Path(archive_dir).parent.parent / "data" / rel,
            upstream_root / "data" / rel,
        ]
        # Also check COMMA_DATA_DIR env var
        data_dir = os.environ.get("COMMA_DATA_DIR")
        if data_dir:
            gt_candidates.insert(0, Path(data_dir) / rel)

        gt_path = None
        for candidate in gt_candidates:
            if candidate.exists():
                gt_path = candidate
                break

        if gt_path is None:
            tried = "\n  ".join(str(c) for c in gt_candidates)
            raise FileNotFoundError(
                f"GT video not found for {rel}.\nTried:\n  {tried}\n"
                f"Set COMMA_DATA_DIR env var to the directory containing GT videos."
            )

        print(f"  GT video: {gt_path}", file=sys.stderr)

        # Stage 4: Decode GT video
        print("Stage 4: Decoding GT video ...", file=sys.stderr)
        gt_frames = _decode_gt_video(str(gt_path))

        if len(gt_frames) != NUM_FRAMES:
            print(f"  WARNING: expected {NUM_FRAMES} frames, got {len(gt_frames)}", file=sys.stderr)

        # Stage 5: Extract masks
        print("Stage 5: Extracting SegNet masks ...", file=sys.stderr)
        masks = _extract_masks(gt_frames, segnet, device, batch_size)
        del gt_frames  # free memory

        # Verify mask resolution
        assert masks.shape[1] == SEG_H and masks.shape[2] == SEG_W, \
            f"Mask resolution mismatch: {masks.shape} vs expected ({SEG_H}, {SEG_W})"

        # Stage 6: Generate and write
        print("Stage 6: Generating frames via renderer ...", file=sys.stderr)
        n_written = _generate_and_write(masks, renderer, str(raw_out), device, batch_size, out_h, out_w)
        del masks  # free memory

        # Verify output
        actual_size = os.path.getsize(str(raw_out))
        expected_size = out_w * out_h * 3 * n_written
        if actual_size != expected_size:
            print(f"  ERROR: output size {actual_size:,} != expected {expected_size:,}",
                  file=sys.stderr)

        t_video_elapsed = time.monotonic() - t_video_start
        print(f"  Video complete: {n_written} frames in {t_video_elapsed:.1f}s "
              f"({n_written / max(t_video_elapsed, 0.01):.1f} fps)",
              file=sys.stderr)

    t_total = time.monotonic() - t_total_start
    print(f"\nTotal inflate time: {t_total:.1f}s", file=sys.stderr)


# ============================================================
# Click CLI (matches inflate_postfilter.py pattern)
# ============================================================
def _cli():
    """Click CLI entry point for inflate_renderer."""
    try:
        import click
    except ImportError:
        # Fallback to plain argparse if click not available
        import argparse
        parser = argparse.ArgumentParser(description="Inflate via neural renderer")
        parser.add_argument("archive_dir", help="Directory containing renderer.bin")
        parser.add_argument("inflated_dir", help="Output directory for .raw files")
        parser.add_argument("video_names_file", help="Text file listing video names")
        parser.add_argument("--renderer-filename", default="renderer.bin",
                            help="Renderer checkpoint filename")
        parser.add_argument("--target-w", type=int, default=OUT_W)
        parser.add_argument("--target-h", type=int, default=OUT_H)
        args = parser.parse_args()
        inflate_renderer(
            args.archive_dir, args.inflated_dir, args.video_names_file,
            renderer_filename=args.renderer_filename,
            out_w=args.target_w, out_h=args.target_h,
        )
        return

    @click.command()
    @click.argument("archive_dir", type=click.Path(exists=True))
    @click.argument("inflated_dir", type=click.Path())
    @click.argument("video_names_file", type=click.Path(exists=True))
    @click.option("--renderer-filename", default="renderer.bin", envvar="RENDERER_FILENAME",
                  help="Renderer checkpoint filename within archive_dir.")
    @click.option("--target-w", type=int, envvar="SOURCE_W",
                  default=OUT_W, help="Output frame width.")
    @click.option("--target-h", type=int, envvar="SOURCE_H",
                  default=OUT_H, help="Output frame height.")
    def inflate(archive_dir, inflated_dir, video_names_file,
                renderer_filename, target_w, target_h):
        """Inflate compressed archive using a trained neural renderer.

        \b
        Positional arguments (compatible with inflate.sh dispatch):
          ARCHIVE_DIR       Directory containing renderer.bin
          INFLATED_DIR      Output directory for .raw files
          VIDEO_NAMES_FILE  Text file listing video names (one per line)

        \b
        The renderer generates RGB frames from SegNet masks extracted from
        the ground-truth video. No compressed video is needed in the archive.

        \b
        Device is auto-detected (CUDA if available, else CPU).
        Batch size: GPU=16, CPU=4.
        """
        inflate_renderer(
            archive_dir, inflated_dir, video_names_file,
            renderer_filename=renderer_filename,
            out_w=target_w, out_h=target_h,
        )

    inflate()


if __name__ == "__main__":
    _cli()
