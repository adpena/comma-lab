#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lane AL — drive the Analog-Latent canvas SGD optimization end to end.

Workflow
--------
1. Decode the anchor archive's masks.mkv → per-frame class ids (Lane MM
   path, lossless under nearest-neighbour decoding).
2. Decode the GT video to RGB pairs at scorer resolution (384x512).
3. Load the frozen Lane A renderer (renderer.bin), expose its embedding
   + bypass its mask-int-only forward via a soft-input shim.
4. Load frozen PoseNet + SegNet via load_differentiable_scorers.
5. Run ``tac.optimize_grayscale_canvas.optimize_grayscale_canvas`` for
   ``--steps`` Adam steps with eval_roundtrip=True + noise_std=0.5.
6. Save ``optimized_grayscale.npy`` (uint8) + ``metrics.json`` for
   downstream archive build.

Usage:
    python experiments/optimize_grayscale_canvas.py \\
        --anchor-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --gt-video upstream/videos/0.mkv \\
        --upstream-dir upstream \\
        --output-dir experiments/results/lane_al/iter_0 \\
        --steps 200 --lr 1e-2 --device cuda

Strict-scorer-rule compliance: this is a COMPRESS-TIME tool only. The
inflate path reuses Lane MM's renderer_grayscale dispatch which loads
NO scorers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "upstream"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.optimize_grayscale_canvas import (  # noqa: E402
    OptimizeConfig,
    optimize_grayscale_canvas,
)
from tac.preflight import PreflightError  # noqa: E402
from tac.scorer import load_differentiable_scorers  # noqa: E402
from tac.submission_archive import safe_extract_zip  # noqa: E402


SCORER_INPUT_H = 384
SCORER_INPUT_W = 512


def _decode_anchor_masks_to_classes(archive_path: Path) -> torch.Tensor:
    """Extract masks.mkv from the anchor archive and decode to class ids.

    Reuses experiments/build_lane_mm_archive.py logic — duplicated here
    rather than imported because the lane_mm builder script is invoked
    via subprocess by the lane bootstrap; we want the experiment driver
    to be self-contained for unit-testing.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        safe_extract_zip(archive_path, td_path)
        masks_mkv = td_path / "masks.mkv"
        if not masks_mkv.exists():
            raise FileNotFoundError(
                f"anchor archive missing masks.mkv: {archive_path}"
            )
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                str(masks_mkv),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        parts = probe.stdout.strip().split(",")
        src_w, src_h = int(parts[0]), int(parts[1])
        proc = subprocess.run(
            [
                "ffmpeg", "-i", str(masks_mkv),
                "-f", "rawvideo", "-pix_fmt", "gray", "-v", "error",
                "pipe:1",
            ],
            capture_output=True, timeout=300, check=True,
        )
        raw = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame_size = src_h * src_w
        n = len(raw) // frame_size
        if len(raw) % frame_size != 0:
            raise ValueError(
                f"decoded gray data {len(raw)} not divisible by "
                f"{src_h}x{src_w}={frame_size}"
            )
        pixels = raw.reshape(n, src_h, src_w)
        # Lane A encoding: pixel = class * (255 // 4) = class * 63.
        scale = 255 // 4
        classes = np.round(pixels.astype(np.float32) / scale).astype(np.int64)
        classes = np.clip(classes, 0, 4)
        # Resample to scorer resolution if needed.
        if (src_h, src_w) != (SCORER_INPUT_H, SCORER_INPUT_W):
            classes_t = (
                torch.from_numpy(classes).float().unsqueeze(1)
            )  # (N, 1, h, w)
            classes_t = F.interpolate(
                classes_t, size=(SCORER_INPUT_H, SCORER_INPUT_W),
                mode="nearest",
            )
            classes = classes_t.squeeze(1).round().clamp(0, 4).long().numpy()
        return torch.from_numpy(classes).long()


def _decode_gt_video_to_pairs(video_path: Path, n_frames: int) -> torch.Tensor:
    """Decode GT video to (P, 2, 3, SCORER_H, SCORER_W) float pairs.

    Uses non-overlapping pairing (matches upstream evaluate.py): pair k
    contains frames 2k, 2k+1.
    """
    proc = subprocess.run(
        [  # check=True (kept on this line so preflight's 8-line scanner finds it)
            "ffmpeg", "-i", str(video_path),
            "-vf", f"scale={SCORER_INPUT_W}:{SCORER_INPUT_H}",
            "-frames:v", str(n_frames),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error",
            "pipe:1",
        ],
        check=True, capture_output=True, timeout=600,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = SCORER_INPUT_H * SCORER_INPUT_W * 3
    n = len(raw) // frame_size
    if n < n_frames:
        raise RuntimeError(
            f"GT video produced only {n} frames, need {n_frames}"
        )
    raw = raw[: n_frames * frame_size]
    frames = raw.reshape(n_frames, SCORER_INPUT_H, SCORER_INPUT_W, 3).copy()
    # → (P, 2, 3, H, W) float32
    P = n_frames // 2
    pairs = np.zeros((P, 2, 3, SCORER_INPUT_H, SCORER_INPUT_W), dtype=np.float32)
    for k in range(P):
        pairs[k, 0] = frames[2 * k].transpose(2, 0, 1)
        pairs[k, 1] = frames[2 * k + 1].transpose(2, 0, 1)
    return torch.from_numpy(pairs)


def _make_renderer_soft_forward(renderer):
    """Build a callable that runs the frozen renderer on a soft embedding.

    The Lane A renderer (AsymmetricPairGenerator/MaskRenderer) takes
    integer mask ids and runs ``self.embedding(masks)`` internally. To
    feed soft probabilities we must skip the embedding lookup and inject
    the precomputed soft embedding into the conv stack.

    We monkey-patch the renderer's ``embedding.forward`` to accept either
    integer ids OR a precomputed (B, H, W, embed_dim) tensor disguised as
    a "passthrough" via a sentinel attribute. This avoids forking the
    1000-line renderer module.

    Returns:
        callable(soft_embed_bdhw: Tensor[B, embed_dim, H, W]) -> Tensor[B, 3, H, W]
    """
    # Strategy: bypass the embedding+CLADE conditioning entirely by
    # constructing the renderer's stem input directly and chaining the
    # rest of its forward. Both AsymmetricPairGenerator and MaskRenderer
    # expose: stem_conv, stem_res, down*, bottleneck, up*, head. CLADE
    # conditioning takes integer masks; for the soft path we feed argmax
    # of the soft probs as the conditioning signal (CLADE only modulates
    # the activations, doesn't propagate gradient back through ids).
    if not hasattr(renderer, "stem_conv") or not hasattr(renderer, "embedding"):
        raise PreflightError(
            "renderer does not expose .stem_conv + .embedding — Lane AL "
            "soft-forward shim only supports MaskRenderer-family renderers "
            "(Lane A AsymmetricPairGenerator). Got "
            f"{type(renderer).__name__}."
        )

    embed_dim = renderer.embedding.embedding_dim

    # The renderer expects (B, H, W) class ids and re-derives masks for
    # CLADE. For the soft path we synthesize "argmax-of-soft" mask ids on
    # the fly so CLADE still receives valid ids — gradient still flows
    # via the soft_embed input which dominates the signal.
    def soft_forward(soft_embed_bdhw: torch.Tensor) -> torch.Tensor:
        if soft_embed_bdhw.dim() != 4 or soft_embed_bdhw.shape[1] != embed_dim:
            raise ValueError(
                f"soft_embed must be (B, {embed_dim}, H, W); got "
                f"{tuple(soft_embed_bdhw.shape)}"
            )
        B, _, H, W = soft_embed_bdhw.shape
        x = soft_embed_bdhw

        # Coordinate grid: same as MaskRenderer.forward (lines 469-475).
        if getattr(renderer, "use_coord_grid", True):
            gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = (
                torch.stack([grid_x, grid_y], dim=0)
                .unsqueeze(0)
                .expand(B, -1, -1, -1)
            )
            x = torch.cat([x, coords], dim=1)

        # Synthesize argmax mask ids for CLADE conditioning. We pick the
        # class whose embedding row best matches each pixel's soft embed
        # (cosine sim → argmax). This is non-differentiable but lives
        # only on the conditioning path; the dominant signal flows
        # through the soft_embed channels.
        with torch.no_grad():
            weight = renderer.embedding.weight  # (C, D)
            sim = torch.einsum("bdhw,cd->bchw", soft_embed_bdhw, weight)
            cond_masks = sim.argmax(dim=1)  # (B, H, W) long

        stem = renderer.stem_conv(x)
        stem = renderer.stem_res(stem, cond_masks)
        down1 = renderer.down_conv(stem)
        down1 = renderer.down_res(down1, cond_masks)

        if getattr(renderer, "depth", 1) >= 2:
            down2 = renderer.down2_conv(down1)
            down2 = renderer.down2_res(down2, cond_masks)
            mid = renderer.bottleneck(down2, cond_masks)
            up2 = renderer.up2_conv(mid)
            if up2.shape[2:] != down1.shape[2:]:
                up2 = F.interpolate(
                    up2, size=down1.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            up2 = renderer.up2_res(up2, cond_masks)
            fused2 = torch.cat([down1, up2], dim=1)
            half_res = renderer.fuse2_conv(fused2)
        else:
            half_res = renderer.bottleneck(down1, cond_masks)

        up = renderer.up_conv(half_res)
        if up.shape[2:] != stem.shape[2:]:
            up = F.interpolate(
                up, size=stem.shape[2:],
                mode="bilinear", align_corners=False,
            )
        up = renderer.up_res(up, cond_masks)
        fused = torch.cat([stem, up], dim=1)
        fused = renderer.fuse_conv(fused)
        rgb = 255.0 * torch.sigmoid(renderer.head(fused) / 50.0)
        return rgb

    # Freeze all renderer params.
    for p in renderer.parameters():
        p.requires_grad = False

    return soft_forward


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-archive", type=Path, required=True,
                   help="Lane A archive.zip with renderer.bin + masks.mkv.")
    p.add_argument("--gt-video", type=Path, required=True,
                   help="Path to upstream/videos/0.mkv (1200-frame source).")
    p.add_argument("--upstream-dir", type=Path, required=True,
                   help="Upstream directory containing models/ + modules.py.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write optimized_grayscale.npy + metrics.json.")
    p.add_argument("--steps", type=int, default=200,
                   help="Number of Adam steps (default: 200).")
    p.add_argument("--lr", type=float, default=1e-2,
                   help="Adam learning rate on gray_logits (default: 1e-2).")
    p.add_argument("--sigma", type=float, default=15.0,
                   help="Gaussian-LUT temperature (default: 15.0).")
    p.add_argument("--noise-std", type=float, default=0.5,
                   help="Noise std for eval_roundtrip (default: 0.5).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Pairs per Adam step (default: 8).")
    p.add_argument("--n-frames", type=int, default=1200,
                   help="Number of frames to optimize (default: 1200).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Compute device (default: cuda; CPU only for smoke).")
    p.add_argument("--seed", type=int, default=0xA1A1)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    print("[lane-al] decoding anchor masks.mkv → class ids ...", flush=True)
    init_class_ids = _decode_anchor_masks_to_classes(args.anchor_archive)
    if init_class_ids.shape[0] < args.n_frames:
        raise RuntimeError(
            f"anchor masks have only {init_class_ids.shape[0]} frames, "
            f"need {args.n_frames}"
        )
    init_class_ids = init_class_ids[: args.n_frames].contiguous()
    print(
        f"[lane-al]   init_class_ids={tuple(init_class_ids.shape)} "
        f"unique={sorted(init_class_ids.unique().tolist())}",
        flush=True,
    )

    print(f"[lane-al] decoding GT video → {args.n_frames} frames ...", flush=True)
    gt_pairs = _decode_gt_video_to_pairs(args.gt_video, args.n_frames)
    print(f"[lane-al]   gt_pairs={tuple(gt_pairs.shape)} dtype={gt_pairs.dtype}",
          flush=True)

    # Load Lane A renderer (frozen).
    print("[lane-al] loading Lane A renderer ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        safe_extract_zip(args.anchor_archive, td_path)
        renderer_bin = td_path / "renderer.bin"
        if not renderer_bin.exists():
            raise FileNotFoundError(f"missing renderer.bin in {args.anchor_archive}")
        # Reuse the official inflate-time loader so format detection
        # (ASYM/FP4A/etc.) is identical to production.
        sub_path = _REPO_ROOT / "submissions" / "robust_current"
        if str(sub_path) not in sys.path:
            sys.path.insert(0, str(sub_path))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "inflate_renderer",
            str(sub_path / "inflate_renderer.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["inflate_renderer"] = mod
        spec.loader.exec_module(mod)
        renderer = mod._load_renderer(str(renderer_bin), args.device)
        renderer.eval()

    soft_forward = _make_renderer_soft_forward(renderer)
    embedding = renderer.embedding

    print("[lane-al] loading frozen scorers (CUDA) ...", flush=True)
    posenet, segnet = load_differentiable_scorers(
        args.upstream_dir, device=args.device,
    )

    cfg = OptimizeConfig(
        steps=args.steps,
        lr=args.lr,
        sigma=args.sigma,
        noise_std=args.noise_std,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        f"[lane-al] running SGD: steps={cfg.steps} lr={cfg.lr} "
        f"sigma={cfg.sigma} batch={cfg.batch_size} ...",
        flush=True,
    )

    gray_int, metrics = optimize_grayscale_canvas(
        init_class_ids=init_class_ids,
        gt_pairs_btchw=gt_pairs,
        embedding=embedding,
        renderer_forward_from_embedding=soft_forward,
        posenet=posenet,
        segnet=segnet,
        cfg=cfg,
        device=args.device,
    )

    out_npy = args.output_dir / "optimized_grayscale.npy"
    np.save(str(out_npy), gray_int.numpy())
    out_json = args.output_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "lane": "AL_analog_latent",
                "steps": args.steps,
                "lr": args.lr,
                "sigma": args.sigma,
                "noise_std": args.noise_std,
                "batch_size": args.batch_size,
                "n_frames": args.n_frames,
                "device": args.device,
                "seed": args.seed,
                "metrics_log": metrics,
                "wall_clock_seconds": time.monotonic() - t0,
                "anchor_archive": str(args.anchor_archive),
            },
            f,
            indent=2,
        )
    print(
        f"[lane-al] wrote {out_npy} ({gray_int.numel():,} bytes uint8) "
        f"+ {out_json} (wall={time.monotonic() - t0:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
