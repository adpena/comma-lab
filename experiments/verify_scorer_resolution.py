#!/usr/bin/env python
"""
Verify Eureka 3: Scorer Resolution Exploit

Hypothesis: Both PoseNet and SegNet internally downscale to 384x512 before
any neural computation. If we store frames at 384x512 and bilinear-upscale
to 874x1164 at inflate time, the scorer's own downscale recovers 384x512.
The round-trip (bilinear-up then bilinear-down) should be nearly lossless.

PRELIMINARY FINDING (mathematical analysis, no models needed):
  The bilinear round-trip is NOT the identity. Scale factors 874/384=2.276
  and 1164/512=2.273 are not integers, causing grid misalignment.
  Mean pixel difference is ~19/255 (~7.4% of dynamic range), SNR ~10 dB.
  The "zero error" claim is BUSTED at the preprocessing level.

  However, this script still measures the actual SCORER OUTPUT difference,
  which could be smaller if the neural networks are insensitive to the
  high-frequency content lost in the round-trip.

This script measures the exact numerical difference between:
  A) scorer(native_874x1164)
  B) scorer(downscale_to_384x512 -> upscale_to_874x1164)

Reports both preprocessing-level and output-level differences to determine
whether the round-trip error propagates through the scorer networks.

Usage:
  PYTHONPATH=src:upstream python experiments/verify_scorer_resolution.py \
      --upstream-dir upstream --device cuda --max-pairs 50

  # MPS fallback (no DALI):
  PYTHONPATH=src:upstream python experiments/verify_scorer_resolution.py \
      --upstream-dir upstream --device mps --max-pairs 50

  # CPU fallback:
  PYTHONPATH=src:upstream python experiments/verify_scorer_resolution.py \
      --upstream-dir upstream --device cpu --max-pairs 50
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def load_scorer_models(upstream_dir: Path, device: torch.device):
    """Load PoseNet and SegNet from upstream model weights."""
    sys.path.insert(0, str(upstream_dir))
    from modules import PoseNet, SegNet
    from safetensors.torch import load_file

    posenet_path = upstream_dir / "models" / "posenet.safetensors"
    segnet_path = upstream_dir / "models" / "segnet.safetensors"

    assert posenet_path.exists(), f"PoseNet weights not found: {posenet_path}"
    assert segnet_path.exists(), f"SegNet weights not found: {segnet_path}"

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_path), device=str(device)))

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_path), device=str(device)))

    return posenet, segnet


def make_synthetic_pairs(n_pairs: int, device: torch.device, seed: int = 42):
    """Generate synthetic frame pairs at native 874x1164 resolution.

    Uses random uint8 tensors to stress-test the round-trip. Real video
    frames would show even smaller differences due to smoother content.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    # Shape: (n_pairs, seq_len=2, C=3, H=874, W=1164)
    frames = torch.randint(
        0, 256, (n_pairs, 2, 3, 874, 1164),
        generator=rng, dtype=torch.uint8
    )
    return frames.float().to(device)


def load_real_frames(upstream_dir: Path, device: torch.device, max_pairs: int):
    """Load real video frames from the upstream test set if available."""
    video_dir = upstream_dir / "videos"
    names_file = upstream_dir / "public_test_video_names.txt"

    if not video_dir.exists() or not names_file.exists():
        return None

    try:
        import av
    except ImportError:
        print("  [info] PyAV not available, skipping real video frames")
        return None

    video_names = names_file.read_text().splitlines()
    frames_collected = []

    for vname in video_names:
        vpath = video_dir / vname
        if not vpath.exists():
            continue

        fmt = "hevc" if str(vpath).endswith(".hevc") else None
        try:
            container = av.open(str(vpath), format=fmt)
        except Exception:
            continue

        stream = container.streams.video[0]
        seq_buf = []

        for frame in container.decode(stream):
            # Convert to RGB tensor
            arr = frame.to_ndarray(format="rgb24")
            t = torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)
            seq_buf.append(t)

            if len(seq_buf) == 2:
                pair = torch.stack(seq_buf)  # (2, 3, H, W)
                frames_collected.append(pair)
                seq_buf = []

                if len(frames_collected) >= max_pairs:
                    break

        container.close()
        if len(frames_collected) >= max_pairs:
            break

    if not frames_collected:
        return None

    # Stack: (N, 2, 3, H, W)
    batch = torch.stack(frames_collected[: max_pairs]).to(device)
    return batch


def round_trip(frames: torch.Tensor, target_h: int = 384, target_w: int = 512) -> torch.Tensor:
    """Downscale to target resolution, then upscale back to original.

    frames: (N, T, C, H, W) float tensor
    Returns: same shape, after down-then-up round trip.
    """
    N, T, C, H, W = frames.shape
    flat = frames.reshape(N * T, C, H, W)

    # Down to scorer internal resolution
    down = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
    # Back up to native
    up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)

    return up.reshape(N, T, C, H, W)


@torch.inference_mode()
def run_posenet_comparison(posenet, native: torch.Tensor, roundtripped: torch.Tensor):
    """Compare PoseNet outputs on native vs round-tripped frames.

    native, roundtripped: (N, 2, 3, H, W) float tensors

    Returns dict with per-pair MSE and aggregate stats.
    """
    # PoseNet.preprocess_input expects (B, T, C, H, W)
    native_pp = posenet.preprocess_input(native)
    rt_pp = posenet.preprocess_input(roundtripped)

    # Check preprocessed inputs are similar
    pp_diff = (native_pp - rt_pp).abs()
    pp_max = pp_diff.max().item()
    pp_mean = pp_diff.mean().item()

    # Forward pass
    native_out = posenet(native_pp)
    rt_out = posenet(rt_pp)

    # Per-pair MSE on pose output (first 6 dims = translation+rotation)
    pose_native = native_out["pose"]  # (N, 12)
    pose_rt = rt_out["pose"]  # (N, 12)

    per_pair_mse = (pose_native - pose_rt).pow(2).mean(dim=1)  # (N,)

    # Also compute the distortion metric the scorer uses (first 6 dims)
    distortion_diff = posenet.compute_distortion(native_out, rt_out)  # (N,)

    return {
        "preprocess_max_diff": pp_max,
        "preprocess_mean_diff": pp_mean,
        "per_pair_mse": per_pair_mse.cpu().numpy(),
        "per_pair_distortion_diff": distortion_diff.cpu().numpy(),
        "pose_native": pose_native.cpu().numpy(),
        "pose_rt": pose_rt.cpu().numpy(),
        "max_output_diff": (pose_native - pose_rt).abs().max().item(),
        "mean_output_diff": (pose_native - pose_rt).abs().mean().item(),
    }


@torch.inference_mode()
def run_segnet_comparison(segnet, native: torch.Tensor, roundtripped: torch.Tensor):
    """Compare SegNet outputs on native vs round-tripped frames.

    native, roundtripped: (N, 2, 3, H, W) float tensors
    SegNet uses only the last frame (index -1).

    Returns dict with per-frame argmax disagreement and logit differences.
    """
    # SegNet.preprocess_input expects (B, T, C, H, W) - takes x[:, -1, ...]
    native_pp = segnet.preprocess_input(native)
    rt_pp = segnet.preprocess_input(roundtripped)

    # Preprocessed input diff
    pp_diff = (native_pp - rt_pp).abs()
    pp_max = pp_diff.max().item()
    pp_mean = pp_diff.mean().item()

    # Forward pass
    native_logits = segnet(native_pp)  # (N, 5, 384, 512)
    rt_logits = segnet(rt_pp)  # (N, 5, 384, 512)

    # Argmax disagreement
    native_argmax = native_logits.argmax(dim=1)  # (N, 384, 512)
    rt_argmax = rt_logits.argmax(dim=1)  # (N, 384, 512)

    per_frame_disagree = (native_argmax != rt_argmax).float().mean(dim=(1, 2))  # (N,)

    # Logit MSE
    per_frame_logit_mse = (native_logits - rt_logits).pow(2).mean(dim=(1, 2, 3))  # (N,)

    # Also compute using scorer's own distortion metric
    distortion_diff = segnet.compute_distortion(native_logits, rt_logits)  # (N,)

    return {
        "preprocess_max_diff": pp_max,
        "preprocess_mean_diff": pp_mean,
        "per_frame_argmax_disagreement": per_frame_disagree.cpu().numpy(),
        "per_frame_logit_mse": per_frame_logit_mse.cpu().numpy(),
        "per_frame_distortion_diff": distortion_diff.cpu().numpy(),
        "max_logit_diff": (native_logits - rt_logits).abs().max().item(),
        "mean_logit_diff": (native_logits - rt_logits).abs().mean().item(),
    }


def report(posenet_results: dict, segnet_results: dict, label: str):
    """Print a summary report."""
    print(f"\n{'='*70}")
    print(f"  RESOLUTION ROUND-TRIP VERIFICATION: {label}")
    print(f"{'='*70}")

    n = len(posenet_results["per_pair_mse"])

    print(f"\n  Tested on {n} frame pairs")

    # Preprocessing comparison
    print("\n  --- Preprocessing (after scorer's internal downscale) ---")
    print(f"  PoseNet preprocess max |diff|:  {posenet_results['preprocess_max_diff']:.10f}")
    print(f"  PoseNet preprocess mean |diff|: {posenet_results['preprocess_mean_diff']:.10f}")
    print(f"  SegNet  preprocess max |diff|:  {segnet_results['preprocess_max_diff']:.10f}")
    print(f"  SegNet  preprocess mean |diff|: {segnet_results['preprocess_mean_diff']:.10f}")

    # PoseNet
    print("\n  --- PoseNet Output ---")
    mse = posenet_results["per_pair_mse"]
    print(f"  Per-pair MSE:    min={mse.min():.2e}  max={mse.max():.2e}  mean={mse.mean():.2e}")
    print(f"  Max output |diff|:  {posenet_results['max_output_diff']:.2e}")
    print(f"  Mean output |diff|: {posenet_results['mean_output_diff']:.2e}")

    dist = posenet_results["per_pair_distortion_diff"]
    print(f"  Scorer distortion metric diff:  min={dist.min():.2e}  max={dist.max():.2e}  mean={dist.mean():.2e}")

    # SegNet
    print("\n  --- SegNet Output ---")
    disagree = segnet_results["per_frame_argmax_disagreement"]
    print(f"  Argmax disagreement:  min={disagree.min():.6f}  max={disagree.max():.6f}  mean={disagree.mean():.6f}")
    print(f"  Max logit |diff|:  {segnet_results['max_logit_diff']:.2e}")
    print(f"  Mean logit |diff|: {segnet_results['mean_logit_diff']:.2e}")

    logit_mse = segnet_results["per_frame_logit_mse"]
    print(f"  Logit MSE:  min={logit_mse.min():.2e}  max={logit_mse.max():.2e}  mean={logit_mse.mean():.2e}")

    dist_s = segnet_results["per_frame_distortion_diff"]
    print(f"  Scorer distortion metric diff:  min={dist_s.min():.2e}  max={dist_s.max():.2e}  mean={dist_s.mean():.2e}")

    # Verdict
    print("\n  --- VERDICT ---")

    # Thresholds: PoseNet distortion contributes sqrt(10*d) to score
    # SegNet distortion contributes 100*d to score
    # "Negligible" means < 0.001 contribution to final score

    posenet_ok = dist.max() < 1e-4  # sqrt(10 * 1e-4) = 0.03 score contribution
    segnet_ok = disagree.max() < 1e-4  # 100 * 1e-4 = 0.01 score contribution

    if posenet_ok and segnet_ok:
        # Check if truly zero
        if posenet_results["preprocess_max_diff"] < 1e-6 and segnet_results["preprocess_max_diff"] < 1e-6:
            print("  EXACT MATCH: bilinear round-trip is identity after scorer preprocessing")
            print("  The scorer literally cannot distinguish 384x512 upscaled frames from native.")
            print(f"  Rate reduction: 874*1164 / (384*512) = {(874*1164)/(384*512):.1f}x storage")
            print("  (But we already use a neural model, so storage is the model, not raw frames.)")
            print("  KEY INSIGHT: upscale at inflate time introduces ZERO scorer error.")
        else:
            print("  NEAR-EXACT: differences are within floating-point tolerance")
            print(f"  Max score impact: < {100*disagree.max() + np.sqrt(10*dist.max()):.4f}")
            print("  Upscale at inflate time introduces negligible scorer error.")
    else:
        print("  FAILED: round-trip introduces measurable scorer differences")
        if not posenet_ok:
            max_pose_score = np.sqrt(10 * dist.max())
            print(f"    PoseNet score impact: up to {max_pose_score:.4f}")
        if not segnet_ok:
            max_seg_score = 100 * disagree.max()
            print(f"    SegNet score impact: up to {max_seg_score:.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify scorer resolution exploit: bilinear round-trip fidelity"
    )
    parser.add_argument(
        "--upstream-dir", type=Path, default=Path("upstream"),
        help="Path to upstream repo (contains models/, modules.py, frame_utils.py)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda, mps, cpu (default: auto-detect)"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=50,
        help="Maximum number of frame pairs to test"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for scorer inference"
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip synthetic random frame test (only run on real video)"
    )
    parser.add_argument(
        "--skip-real", action="store_true",
        help="Skip real video frame test (only run on synthetic)"
    )
    parser.add_argument(
        "--math-only", action="store_true",
        help="Only run mathematical analysis (no model loading needed)"
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Upstream: {args.upstream_dir.resolve()}")

    if args.math_only:
        run_math_analysis(device)
        return

    # Load models
    print("Loading scorer models...")
    posenet, segnet = load_scorer_models(args.upstream_dir.resolve(), device)

    def run_verification(frames: torch.Tensor, label: str):
        """Run full verification on a batch of frames."""
        N = frames.shape[0]
        rt_frames = round_trip(frames)

        # Process in batches
        all_posenet = []
        all_segnet = []

        for i in range(0, N, args.batch_size):
            b_native = frames[i : i + args.batch_size]
            b_rt = rt_frames[i : i + args.batch_size]

            p = run_posenet_comparison(posenet, b_native, b_rt)
            s = run_segnet_comparison(segnet, b_native, b_rt)

            all_posenet.append(p)
            all_segnet.append(s)

        # Merge results
        def merge(results_list: list[dict]) -> dict:
            merged = {}
            for key in results_list[0]:
                vals = [r[key] for r in results_list]
                if isinstance(vals[0], np.ndarray):
                    merged[key] = np.concatenate(vals)
                elif isinstance(vals[0], float):
                    if "max" in key:
                        merged[key] = max(vals)
                    elif "mean" in key:
                        merged[key] = np.mean(vals)
                    else:
                        merged[key] = max(vals)
                else:
                    merged[key] = max(vals)
            return merged

        report(merge(all_posenet), merge(all_segnet), label)

    # Test 1: Synthetic random frames (worst case - high frequency content)
    if not args.skip_synthetic:
        print("\n--- Generating synthetic random frames (worst case for bilinear) ---")
        synthetic = make_synthetic_pairs(min(args.max_pairs, 16), device)
        run_verification(synthetic, "SYNTHETIC RANDOM (worst case)")
        del synthetic
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Test 2: Real video frames (actual use case)
    if not args.skip_real:
        print("\n--- Loading real video frames ---")
        real = load_real_frames(args.upstream_dir.resolve(), device, args.max_pairs)
        if real is not None:
            print(f"  Loaded {real.shape[0]} real frame pairs")
            run_verification(real, "REAL VIDEO FRAMES")
            del real
        else:
            print("  [skip] No real video frames available (need upstream/videos/)")

    # Mathematical analysis (always runs, no models needed)
    run_math_analysis(device)


def run_math_analysis(device: torch.device):
    """Pure mathematical analysis of the bilinear round-trip. No models needed."""
    print(f"\n{'='*70}")
    print("  MATHEMATICAL ANALYSIS (no models needed)")
    print(f"{'='*70}")
    print()
    print("  The scorer preprocessing chain is:")
    print("    PoseNet: rearrange -> F.interpolate(size=(384,512), bilinear) -> rgb_to_yuv6")
    print("    SegNet:  x[:,-1]   -> F.interpolate(size=(384,512), bilinear)")
    print()
    print("  Our inflate pipeline: render at 384x512 -> F.interpolate to 874x1164 -> raw")
    print("  Scorer then: raw 874x1164 -> F.interpolate to 384x512")
    print()
    print("  Scale factors: 874/384 = {:.6f}, 1164/512 = {:.6f}".format(874/384, 1164/512))
    print("  These are NOT integer ratios -> grid misalignment -> round-trip error")
    print()

    # Direct test with random data
    print("  --- Direct preprocessing round-trip test ---")
    torch.manual_seed(42)

    # Test with uint8-range values (realistic)
    x = torch.randint(0, 256, (4, 3, 384, 512), dtype=torch.float32, device=device)
    up = F.interpolate(x, size=(874, 1164), mode="bilinear", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)

    diff = (down - x).abs()
    print("  384x512 -> 874x1164 -> 384x512 (uint8-range random):")
    print(f"    max |diff|  = {diff.max().item():.4f}")
    print(f"    mean |diff| = {diff.mean().item():.4f}")
    print(f"    as % of 255 = {diff.mean().item()/255*100:.1f}%")
    print()

    # Distribution of errors
    for thresh in [0.5, 1.0, 5.0, 10.0, 25.0]:
        frac = (diff > thresh).float().mean().item()
        print(f"    Pixels with |diff| > {thresh:5.1f}: {frac*100:.1f}%")
    print()

    # YUV amplification
    max_yuv_amp = max(0.299 + 0.587 + 0.114, 1/1.772 + 1, 1/1.402 + 1)
    print(f"  After rgb_to_yuv6 (max amplification {max_yuv_amp:.3f}x):")
    print(f"    Worst case YUV diff: {diff.max().item() * max_yuv_amp:.2f}")
    print()

    # Test with SMOOTH data (more realistic for natural images)
    print("  --- Smooth data test (gaussian-blurred random, more like real video) ---")
    import torch.nn.functional as Fk
    kernel_size = 15
    sigma = 3.0
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    x_raw = torch.randint(0, 256, (4, 3, 384, 512), dtype=torch.float32, device=device)
    x_smooth = Fk.conv2d(x_raw, kernel_2d, padding=kernel_size // 2, groups=3)

    up_s = F.interpolate(x_smooth, size=(874, 1164), mode="bilinear", align_corners=False)
    down_s = F.interpolate(up_s, size=(384, 512), mode="bilinear", align_corners=False)

    diff_s = (down_s - x_smooth).abs()
    print("  384x512 -> 874x1164 -> 384x512 (smooth/blurred):")
    print(f"    max |diff|  = {diff_s.max().item():.4f}")
    print(f"    mean |diff| = {diff_s.mean().item():.4f}")
    print(f"    as % of 255 = {diff_s.mean().item()/255*100:.1f}%")
    print()

    # VERDICT
    print("  --- MATHEMATICAL VERDICT ---")
    print()
    if diff.mean().item() > 1.0:
        print("  HYPOTHESIS BUSTED at preprocessing level.")
        print(f"  The bilinear round-trip introduces mean {diff.mean().item():.1f} pixel")
        print(f"  difference ({diff.mean().item()/255*100:.1f}% of dynamic range).")
        print("  This is NOT negligible -- the scorer sees substantially different inputs.")
        print()
        print("  The 'zero scorer error' claim from Eureka 3 is FALSE.")
        print("  The round-trip acts as a low-pass filter, not the identity.")
        print()
        print("  HOWEVER: the scorer OUTPUT difference (requires model inference)")
        print("  could still be small if the neural networks are insensitive to")
        print("  the high-frequency content destroyed by the round-trip.")
        print("  Run with --upstream-dir pointing to models to measure this.")
    else:
        print(f"  Preprocessing differences are small ({diff.mean().item():.4f} mean).")
        print("  Model inference test needed to confirm scorer output equivalence.")
    print()


if __name__ == "__main__":
    main()
