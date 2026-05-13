#!/usr/bin/env python3
"""Lane F-V4 Phase 1 — per-layer FP4 sensitivity profiler.

For each Conv2d / Linear weight in a renderer, quantize ONLY that one layer
to FP4 (everything else stays FP32) and measure the resulting score
delta against the FP32 baseline. The output is a per-layer "sensitivity"
score that Phase 2 (the mixed-precision selector inside qat_finetune.py)
consumes to decide which layers stay FP16/FP32 (top 30% by sensitivity)
and which can be FP4-quantized (bottom 70%).

This is the V4 anti-regression for Lane F:
  V1 → 2.73 (silent zero-pose bug)
  V2 → 1.79 (uniform FP4 → 20x PoseNet penalty on FastViT YUV6 path)
  V3 → 1.85 (INT8 warmup helped PoseNet -17%, hurt SegNet +38%)
  V4 → predicted [1.20, 1.50] (mixed-precision FP4: keep the
        PoseNet-critical layers in FP16, FP4 the rest)

The architectural insight: dilated-h64 ASYM has a small set of layers
whose YUV statistics are load-bearing for FastViT-T12 attention. Uniform
FP4 destroys those statistics. Identifying them empirically (rather than
guessing from name patterns) is the only way to know.

Output schema (`layer_sensitivity.pt`):
    {
      "delta": {layer_name: float total_distortion_delta_vs_baseline},
      "delta_pose": {layer_name: float pose_distortion_delta},
      "delta_seg": {layer_name: float seg_distortion_delta},
      "param_count": {layer_name: int weight tensor numel},
      "baseline": {"pose_d": float, "seg_d": float, "distortion": float},
      "metadata": {
        "checkpoint": str,
        "n_pairs_used": int,
        "n_eligible_layers": int,
        "git_hash": str,
        "torch_version": str,
        "device": str,
        "elapsed_s": float,
        "predicted_band": [float, float],
      }
    }

CLAUDE.md compliance:
  * --device cuda required by default. CPU opt-in only with explicit
    "advisory only" tag — rejected here because per-layer FP4 quantization
    on the PoseNet-FastViT path needs CUDA-native float math (MPS-CUDA
    drift = 23x per CLAUDE.md non-negotiable).
  * No MPS fallback ternary anywhere.
  * Strict-scorer-rule compliant: scorers loaded ONLY for sensitivity
    measurement, NEVER bundled into any archive.
  * eval_roundtrip is forced True in measurement (matches the scorer's
    upscale → uint8 → downscale chain that Lane A renderer was trained
    against; without this the per-layer deltas would not predict the
    archive-time deltas).

Usage:
    PYTHONPATH=src:upstream python experiments/profile_fp4_layer_sensitivity.py \\
        --checkpoint experiments/results/lane_a_landed/iter_0/renderer.bin \\
        --video upstream/videos/0.mkv \\
        --masks-mkv experiments/results/lane_a_landed/extracted/masks.mkv \\
        --poses experiments/results/lane_a_landed/optimized_poses.pt \\
        --output experiments/results/lane_f_v4/layer_sensitivity.pt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))


# ── Eligible-layer selection ────────────────────────────────────────────────


def select_eligible_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Return dict of qualified_name → Conv2d/Linear module with weight.ndim>=2.

    We include ALL eligible layers (no protection list) because the entire
    point of Phase 1 is to discover empirically which layers ARE protected.
    Lane S's hand-coded protection patterns are a strong prior, but the
    profile measures every layer so a surprising result (e.g. fuse_conv
    being FP4-tolerable while a "bulk" stem_conv is critical) won't hide.

    Excludes:
      - ConvTranspose2d (different STE behavior; Lane S already protects)
      - Embedding (small, contributes negligibly to size)
      - layers with weight.ndim<2 (1D biases, not eligible for FP4 block-quant)
    """
    out: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ConvTranspose2d):
            continue
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            if hasattr(mod, "weight") and mod.weight.ndim >= 2:
                out[name] = mod
    return out


# ── Per-layer FP4 quantization (single layer only, others untouched) ────────


def quantize_one_layer_fp4(module: nn.Module, block_size: int = 32) -> torch.Tensor:
    """Replace ``module.weight`` (in place) with the FP4 round-tripped tensor.

    Returns the original FP32 weight so the caller can restore it after
    the measurement. No autograd / parametrize machinery — we want a pure
    forward-pass measurement of the FP4 round-trip, NOT QAT signal.
    """
    from tac.fp4_quantize import DEFAULT_CODEBOOK, fake_quant_fp4

    original = module.weight.detach().clone()
    with torch.no_grad():
        codebook = DEFAULT_CODEBOOK.to(module.weight.device)
        # FP4_HARDWARE_DISCLOSED: this profiler measures the simulated-FP4
        # round-trip noise (FakeQuantFP4 in FP32). NVFP4 hardware needs
        # Blackwell (CC >= 10.0); 4090 is CC 8.9 → FP4 inference is always
        # simulated. The profile predicts the LANDED archive's noise but
        # NOT actual hardware-FP4 behavior. See Lane F-V5 (hardware FP8
        # rescue path) in project_cosmos_deep_dive_addendum_20260428.
        # fake_quant_fp4 with stochastic=False + robust_scale=False matches
        # the export-time deterministic round-trip (matches what
        # export_asymmetric_checkpoint_fp4 produces). This is what we want
        # because the profile must predict the LANDED archive's behavior.
        quantized = fake_quant_fp4(
            module.weight.data,
            codebook=codebook,
            block_size=block_size,
            stochastic=False,
            robust_scale=False,
        )
        module.weight.data.copy_(quantized)
    return original


def restore_layer(module: nn.Module, original: torch.Tensor) -> None:
    with torch.no_grad():
        module.weight.data.copy_(original)


def _gray_mask_to_class_ids(gray: torch.Tensor) -> torch.Tensor:
    """Convert encoded grayscale mask pixels to class IDs in ``[0, 4]``.

    Contest mask videos encode classes as ``class_id * (255 // 4)`` grayscale
    values. Feeding raw luma values such as 63/126/189/252 into renderer
    embeddings corrupts every downstream sensitivity conclusion.
    """
    scale = 255 // 4
    classes = torch.round(gray.to(torch.float32) / float(scale)).to(torch.long)
    return classes.clamp_(0, 4)


# ── Distortion measurement (identical to qat_finetune.evaluate_fp4_quality) ─


def measure_distortion(
    model: nn.Module,
    masks: torch.Tensor,
    gt_frames: list,
    poses: torch.Tensor | None,
    distortion_net: nn.Module,
    device: torch.device,
    n_pairs: int,
    zoom_warp: nn.Module | None = None,
) -> dict[str, float]:
    """Run forward pass on ``n_pairs`` pairs and return mean distortion.

    Uses the upstream DistortionNet directly (NO proxy — the proxy MSE
    is 100-350x off per CLAUDE.md proxy_auth_math_useless rule). This
    is still NOT contest-CUDA inflate (it skips archive rate + the eval
    pipeline), but it's the authoritative distortion-measurement loop
    that the contest scorer actually uses.

    eval_roundtrip is implicit (we upscale to camera res + uint8) so the
    per-layer deltas predict the archive-time deltas faithfully.
    """
    model.eval()
    pd_list: list[float] = []
    sd_list: list[float] = []
    with torch.inference_mode():
        for i in range(n_pairs):
            m_t = masks[2 * i : 2 * i + 1].to(device=device, dtype=torch.int64)
            m_t1 = masks[2 * i + 1 : 2 * i + 2].to(device=device, dtype=torch.int64)
            kwargs = {}
            if poses is not None:
                kwargs["pose"] = poses[i : i + 1].to(device)
            if zoom_warp is not None:
                pair_idx = torch.tensor([i], device=device)
                kwargs["ego_flow"] = zoom_warp(pair_idx, m_t.shape[1], m_t.shape[2])
            pair = model(m_t, m_t1, **kwargs)  # (B=1, 2, H, W, 3)
            chw = pair[0].permute(0, 3, 1, 2).float()  # (2, 3, H, W)
            cam = F.interpolate(
                chw, size=(874, 1164), mode="bilinear", align_corners=False,
            )
            cam = cam.round().clamp(0, 255).to(torch.uint8).float()

            gt_p = torch.stack([
                torch.from_numpy(gt_frames[2 * i]).float(),
                torch.from_numpy(gt_frames[2 * i + 1]).float(),
            ]).unsqueeze(0).to(device)
            comp_p = cam.permute(0, 2, 3, 1).unsqueeze(0).contiguous()
            pd, sd = distortion_net.compute_distortion(gt_p, comp_p)
            pd_list.append(pd.item())
            sd_list.append(sd.item())
    avg_pose = sum(pd_list) / max(len(pd_list), 1)
    avg_seg = sum(sd_list) / max(len(sd_list), 1)
    distortion = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose)
    return {"pose_d": avg_pose, "seg_d": avg_seg, "distortion": distortion}


# ── Main profile ────────────────────────────────────────────────────────────


def profile_fp4_layer_sensitivity(
    *,
    checkpoint: str,
    video_mkv: str,
    masks_mkv: str,
    poses_path: str | None,
    upstream_dir: str,
    output: str,
    device: str,
    n_pairs: int,
    block_size: int,
    predicted_band: tuple[float, float] = (1.20, 1.50),
    allow_diagnostic_cpu: bool = False,
) -> dict:
    t_start = time.monotonic()

    # ── Strict device check (CLAUDE.md non-negotiable) ──
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but CUDA not available. "
            "Lane F-V4 profiler runs on CUDA only (FP4 round-trip + "
            "PoseNet-FastViT YUV6 has 23x MPS-CUDA drift per CLAUDE.md). "
            "Use --device cpu --allow-diagnostic-cpu only for non-promotable "
            "debug output."
        )
    if device == "cpu" and not allow_diagnostic_cpu:
        raise SystemExit(
            "FATAL: --device cpu is diagnostic-only and non-promotable. Pass "
            "--allow-diagnostic-cpu to make that status explicit, or run on "
            "--device cuda for sensitivity evidence."
        )
    if device == "mps":
        raise SystemExit(
            "FATAL: --device mps forbidden — CLAUDE.md non-negotiable: "
            "PoseNet on MPS drifts 23x; per-layer sensitivity would be noise."
        )

    print(f"[profile_fp4] device={device}")
    print(f"[profile_fp4] checkpoint={checkpoint}")
    torch_device = torch.device(device)

    # ── Load renderer ──
    from tac.renderer_export import load_any_renderer_checkpoint

    model = load_any_renderer_checkpoint(checkpoint, device=device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    has_film = bool(getattr(model, "pose_dim", 0))
    has_zoom = bool(getattr(model, "use_zoom_flow", False))
    print(f"[profile_fp4]   {type(model).__name__} {n_params:,} params "
          f"film={has_film} zoom={has_zoom}")

    # ── Load poses if FiLM ──
    poses = None
    if has_film:
        if poses_path is None:
            raise SystemExit(
                "FATAL: model has FiLM (pose_dim={}) but no --poses provided. "
                "Per CLAUDE.md FORBIDDEN PATTERNS: silent zero-pose fallback "
                "caused +58% PoseNet regression in Lane F-V1 (memory: "
                "project_baseline_poses_load_bearing).".format(model.pose_dim)
            )
        from tac.submission_archive import load_optimized_poses
        poses = load_optimized_poses(poses_path, pose_dim=int(model.pose_dim)).to(device)
        print(f"[profile_fp4]   poses: {tuple(poses.shape)}")

    # ── Load GT video + masks + scorers ──
    print(f"[profile_fp4] loading GT video {video_mkv}")
    import av
    container = av.open(video_mkv)
    stream = container.streams.video[0]
    gt_frames: list = []
    for frame in container.decode(stream):
        gt_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    print(f"[profile_fp4]   {len(gt_frames)} frames")

    print(f"[profile_fp4] loading masks {masks_mkv}")
    container = av.open(masks_mkv)
    stream = container.streams.video[0]
    masks_list: list = []
    for frame in container.decode(stream):
        masks_list.append(_gray_mask_to_class_ids(torch.from_numpy(frame.to_ndarray(format="gray"))))
    container.close()
    masks = torch.stack(masks_list, dim=0).to(torch.long)
    if int(masks.min()) < 0 or int(masks.max()) > 4:
        raise SystemExit(
            f"FATAL: decoded masks contain class range "
            f"[{int(masks.min())}, {int(masks.max())}], expected [0, 4]"
        )
    print(f"[profile_fp4]   masks shape={tuple(masks.shape)}")

    # ── Load DistortionNet (upstream authoritative scorer) ──
    upstream_root = Path(upstream_dir)
    if not (upstream_root / "modules.py").exists():
        raise SystemExit(f"FATAL: upstream/modules.py not at {upstream_root}")
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from modules import DistortionNet  # noqa: E402

    distortion_net = DistortionNet().eval().to(torch_device)
    distortion_net.load_state_dicts(
        upstream_root / "models" / "posenet.safetensors",
        upstream_root / "models" / "segnet.safetensors",
        torch_device,
    )

    # ── ZoomWarp (if model needs it) ──
    zoom_warp = None
    if has_zoom:
        from tac.radial_zoom import RadialZoomWarp
        n_pairs_total = masks.shape[0] // 2
        zoom_warp = RadialZoomWarp(n_pairs=n_pairs_total).to(torch_device)
        # Try to load zoom scalars from sibling zoom_scalars.bin if present
        ckpt_path = Path(checkpoint)
        zoom_path = ckpt_path.parent / "zoom_scalars.bin"
        if zoom_path.exists():
            from tac.radial_zoom import load_zoom_scalars
            scalars, foe = load_zoom_scalars(zoom_path, n_pairs=n_pairs_total)
            with torch.no_grad():
                zoom_warp.zoom_scalars.copy_(scalars.to(torch_device))
                if foe is not None and zoom_warp.learn_foe:
                    zoom_warp.foe.copy_(foe.to(torch_device))
            print(f"[profile_fp4]   zoom scalars loaded from {zoom_path.name}")

    # ── Eligible layers ──
    eligible = select_eligible_layers(model)
    print(f"[profile_fp4] eligible layers: {len(eligible)} "
          f"(total params in those layers: "
          f"{sum(m.weight.numel() for m in eligible.values()):,})")

    # ── Baseline (FP32) measurement ──
    print(f"[profile_fp4] measuring FP32 baseline on {n_pairs} pairs ...")
    baseline = measure_distortion(
        model, masks, gt_frames, poses, distortion_net,
        torch_device, n_pairs, zoom_warp=zoom_warp,
    )
    print(f"[profile_fp4]   baseline pose_d={baseline['pose_d']:.6f} "
          f"seg_d={baseline['seg_d']:.6f} distortion={baseline['distortion']:.4f}")

    # ── Per-layer FP4 sweep ──
    delta: dict[str, float] = {}
    delta_pose: dict[str, float] = {}
    delta_seg: dict[str, float] = {}
    param_count: dict[str, int] = {}

    for i, (name, module) in enumerate(eligible.items()):
        param_count[name] = int(module.weight.numel())
        original = quantize_one_layer_fp4(module, block_size=block_size)
        try:
            q = measure_distortion(
                model, masks, gt_frames, poses, distortion_net,
                torch_device, n_pairs, zoom_warp=zoom_warp,
            )
            d_total = q["distortion"] - baseline["distortion"]
            d_pose = q["pose_d"] - baseline["pose_d"]
            d_seg = q["seg_d"] - baseline["seg_d"]
        finally:
            restore_layer(module, original)

        delta[name] = float(d_total)
        delta_pose[name] = float(d_pose)
        delta_seg[name] = float(d_seg)
        print(f"[profile_fp4]   [{i+1:>3d}/{len(eligible)}] {name:<40s} "
              f"({param_count[name]:>6d} params) Δd={d_total:+.4f} "
              f"(pose {d_pose:+.5f}, seg {d_seg:+.5f})", flush=True)

    # ── Summary stats ──
    sorted_by_delta = sorted(delta.items(), key=lambda kv: kv[1])
    print("[profile_fp4] LEAST sensitive (FP4-tolerable, bottom 5):")
    for n, d in sorted_by_delta[:5]:
        print(f"    {n:<40s} Δd={d:+.4f} ({param_count[n]:>6d} params)")
    print("[profile_fp4] MOST sensitive (keep FP16, top 5):")
    for n, d in sorted_by_delta[-5:][::-1]:
        print(f"    {n:<40s} Δd={d:+.4f} ({param_count[n]:>6d} params)")

    # ── Provenance ──
    git_hash = "no-git"
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    metadata = {
        "checkpoint": checkpoint,
        "video_mkv": video_mkv,
        "masks_mkv": masks_mkv,
        "poses": poses_path,
        "n_pairs_used": int(n_pairs),
        "n_eligible_layers": len(eligible),
        "fp4_block_size": int(block_size),
        "git_hash": git_hash,
        "torch_version": torch.__version__,
        "device": device,
        "promotion_eligible": device == "cuda",
        "evidence_grade": "empirical_cuda_proxy" if device == "cuda" else "diagnostic_cpu",
        "mask_decode": "gray_luma_to_class_id_round_clamp_v1",
        "elapsed_s": time.monotonic() - t_start,
        "predicted_band": list(predicted_band),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta": delta,
            "delta_pose": delta_pose,
            "delta_seg": delta_seg,
            "param_count": param_count,
            "baseline": baseline,
            "metadata": metadata,
        },
        out_path,
    )
    sidecar = out_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "delta": delta,
        "delta_pose": delta_pose,
        "delta_seg": delta_seg,
        "param_count": param_count,
        "baseline": baseline,
        "metadata": metadata,
    }, indent=2))
    print(f"[profile_fp4] WROTE {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"[profile_fp4] WROTE {sidecar}")
    return {"delta": delta, "baseline": baseline, "metadata": metadata}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lane F-V4 Phase 1: per-layer FP4 sensitivity profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Renderer .bin / .pt (default: Lane A's renderer.bin)")
    parser.add_argument("--video", required=True,
                        help="GT video .mkv (1200 frames)")
    parser.add_argument("--masks-mkv", required=True,
                        help="Masks .mkv (matches frame count of GT video)")
    parser.add_argument("--poses", default=None,
                        help="optimized_poses.pt or .bin (REQUIRED if model has FiLM)")
    parser.add_argument("--upstream", default="upstream",
                        help="Upstream dir (default: upstream)")
    parser.add_argument("--output", required=True,
                        help="Output .pt path for layer_sensitivity tensor")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default cuda; MPS forbidden)")
    parser.add_argument(
        "--allow-diagnostic-cpu",
        action="store_true",
        help="Allow non-promotable CPU profiling for local debugging only.",
    )
    parser.add_argument("--n-pairs", type=int, default=30,
                        help="Number of pairs per per-layer measurement "
                             "(default 30 — balances signal/noise vs runtime; "
                             "30 pairs * ~80 layers * ~2 sec = ~80 min total)")
    parser.add_argument("--block-size", type=int, default=32,
                        help="FP4 block size (must match training/export)")
    parser.add_argument("--predicted-band", nargs=2, type=float,
                        default=(1.20, 1.50), metavar=("LOW", "HIGH"),
                        help="Council predicted band recorded in provenance")
    args = parser.parse_args(argv)

    profile_fp4_layer_sensitivity(
        checkpoint=args.checkpoint,
        video_mkv=args.video,
        masks_mkv=args.masks_mkv,
        poses_path=args.poses,
        upstream_dir=args.upstream,
        output=args.output,
        device=args.device,
        n_pairs=args.n_pairs,
        block_size=args.block_size,
        predicted_band=tuple(args.predicted_band),
        allow_diagnostic_cpu=args.allow_diagnostic_cpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
