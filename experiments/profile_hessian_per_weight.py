"""Lane Ω Phase 1 — per-weight Fisher importance profiler driven by hard-pair score.

For a renderer (default: Lane A's renderer.bin) and a set of pose / mask
artifacts, compute per-element Fisher importance:

    I(w_ij) = (1 / K) * Σ_{p ∈ hard pairs} pair_weight_p * (∂L_p / ∂w_ij)²

where L_p is the contest score on pair p (100*seg_p + sqrt(10*pose_p)) using
upstream PoseNet + SegNet (NOT the proxy). The result is a per-element
importance tensor that Phase 2 (water-fill bit allocator) consumes.

The "hardest" pairs come from either:
  * --pair-weights /path/to/pair_weights.pt — Lane W's per-pair importance
    (preferred when available; biases the Hessian toward the contest's
    heavy tail per memory feedback_posenet_tracking).
  * --all-pairs — uniform weight = 1/600 (fallback when Lane W's profile
    isn't built yet).

Eligible layers match the SC_PROTECTED_NAME_PATTERNS exclusion list from
src/tac/self_compress.py: protected layers (renderer.head, motion.head, FiLM
linears, fuse_conv) are SKIPPED — they will stay FP32 in the Ωv1 export.
This matches Lane S's strategy and prevents Lane Ω from re-introducing the
same scorer-sensitive bottleneck Lane F-V2 hit (memory:
project_lane_f_fp4_qat_regression_20260427).

Output:
  hessian_per_weight.pt  — torch.save dict:
    {
      "importance":  {layer_name → torch.Tensor (same shape as state_dict[name])},
      "metadata": {
        "checkpoint":     str,
        "poses":          str | None,
        "masks_mkv":      str,
        "video_mkv":      str,
        "pair_weights":   str | None,
        "top_k":          int,
        "n_pairs_seen":   int,
        "n_eligible_layers": int,
        "n_eligible_weights": int,
        "imp_min":        float,
        "imp_max":        float,
        "imp_median":     float,
        "imp_p95":        float,
        "imp_p99":        float,
        "git_hash":       str,
        "torch_version":  str,
        "device":         str,
        "elapsed_s":      float,
      }
    }

CLAUDE.md compliance:
  * --device cuda required by default (CPU opt-in only with --device cpu
    + an explicit "advisory only" tag — rejected here, profiling Fisher
    on MPS would give 23x-drift importance estimates).
  * No MPS fallback ternary.
  * Strict-scorer-rule compliant: scorers are loaded ONLY for profiling,
    NOT included in the Ωv1 archive.
  * Provenance keys mirror remote_lane_s_self_compress.sh's stage 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))


def _load_renderer(checkpoint: str, device: str) -> torch.nn.Module:
    from tac.renderer_export import load_any_renderer_checkpoint

    model = load_any_renderer_checkpoint(checkpoint, device=device)
    model.eval()
    return model


def _load_poses(poses_path: str | None, pose_dim_default: int = 6) -> torch.Tensor | None:
    if poses_path is None:
        return None
    from tac.submission_archive import load_optimized_poses

    return load_optimized_poses(
        poses_path,
        pose_dim=pose_dim_default,
        expected_n_pairs=600,
    )


def _load_pair_weights(path: str | None, n_pairs: int) -> torch.Tensor:
    """Load (or default to uniform) per-pair weights.

    Returns: tensor of shape (n_pairs,), float32, sums to 1.0.
    """
    if path is None:
        return torch.full((n_pairs,), 1.0 / n_pairs, dtype=torch.float32)
    # Loader-format-safety: pair_weights.pt is always a pickle (Lane W's
    # tensor save), never a renderer .bin. Verify magic bytes are NOT
    # FP4A/ASYM/DPSM/I4LZ before pickle-loading. (preflight rule
    # check_loader_format_safety / DEN-V2 bug class.)
    with open(path, "rb") as _f:
        _magic = _f.read(4)
    if _magic in (b"FP4A", b"ASYM", b"DPSM", b"I4LZ", b"CCh1", b"C3R1", b"SCv1"):
        raise ValueError(
            f"pair_weights file {path} has renderer magic {_magic!r} — "
            f"expected pytorch pickle. Wrong file?"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        # Lane W convention: {"weights": tensor, "metadata": {...}}
        if "weights" in data:
            w = data["weights"]
        elif "pair_weights" in data:
            w = data["pair_weights"]
        else:
            raise ValueError(
                f"pair_weights file {path} dict has no 'weights' / "
                f"'pair_weights' key; got keys: {list(data.keys())[:8]}"
            )
    else:
        w = data
    w = torch.as_tensor(w, dtype=torch.float32).reshape(-1)
    if w.numel() != n_pairs:
        raise ValueError(
            f"pair_weights has {w.numel()} entries, expected {n_pairs}"
        )
    if (w < 0).any():
        raise ValueError("pair_weights contain negative values")
    s = float(w.sum().item())
    if s <= 0:
        raise ValueError("pair_weights sum to ≤ 0 — degenerate")
    return w / s


def _decode_gt_video(video_mkv: str) -> torch.Tensor:
    """Decode 1200-frame GT video → uint8 tensor (1200, 3, H, W) on CPU."""
    import av

    print(f"[profile] decoding GT video {video_mkv} ...")
    t0 = time.monotonic()
    container = av.open(video_mkv)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3)
        frames.append(torch.from_numpy(rgb))
    container.close()
    out = torch.stack(frames, dim=0).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
    print(f"[profile]   decoded {out.shape[0]} frames in {time.monotonic() - t0:.1f}s")
    return out


def _gray_mask_to_class_ids(gray: torch.Tensor) -> torch.Tensor:
    """Convert encoded grayscale mask pixels to class IDs in ``[0, 4]``.

    Contest mask videos encode class labels as grayscale values using
    ``class_id * (255 // 4)``. Reading those bytes directly as class IDs sends
    values like 63/126/189/252 into the renderer embedding and crashes CUDA.
    """
    scale = 255 // 4
    classes = torch.round(gray.to(torch.float32) / float(scale)).to(torch.long)
    return classes.clamp_(0, 4)


def _load_masks_video(masks_mkv: str) -> torch.Tensor:
    """Decode masks .mkv (luma only) → int64 mask tensor (N, H, W)."""
    import av

    print(f"[profile] decoding masks {masks_mkv} ...")
    container = av.open(masks_mkv)
    stream = container.streams.video[0]
    masks = []
    for frame in container.decode(stream):
        # Masks are encoded as monochrome. The luma plane carries the class.
        gray = frame.to_ndarray(format="gray")
        masks.append(_gray_mask_to_class_ids(torch.from_numpy(gray)))
    container.close()
    out = torch.stack(masks, dim=0).to(torch.long)
    if int(out.min()) < 0 or int(out.max()) > 4:
        raise RuntimeError(
            f"decoded masks contain class range [{int(out.min())}, {int(out.max())}], "
            "expected [0, 4]"
        )
    print(f"[profile]   decoded {out.shape[0]} masks shape={tuple(out.shape)}")
    return out


def _select_eligible_params_with_exclusions(
    model: torch.nn.Module,
    *,
    include_protected_conv2d: bool = False,
) -> tuple[dict[str, torch.nn.Parameter], dict[str, str]]:
    """Return dict of layer_name → weight Parameter for every eligible Conv2d
    / Linear weight, excluding the SC_PROTECTED_NAME_PATTERNS list.

    Matches Lane S's protection list so Lane Ω only quantizes the same
    "bulk" weights Lane S would.
    """
    from tac.self_compress import _is_protected_name

    out: dict[str, torch.nn.Parameter] = {}
    excluded: dict[str, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            # ConvTranspose2d is a subclass mismatch — explicitly skip it.
            if isinstance(mod, torch.nn.ConvTranspose2d):
                excluded[f"{name}.weight"] = "convtranspose2d_not_profiled"
                continue
            if _is_protected_name(name) and not include_protected_conv2d:
                excluded[f"{name}.weight"] = "protected_conv2d"
                continue
            out[f"{name}.weight"] = mod.weight
        elif isinstance(mod, torch.nn.Linear):
            if _is_protected_name(name):
                excluded[f"{name}.weight"] = "protected_linear"
                continue
            out[f"{name}.weight"] = mod.weight
    return out, excluded


def _select_eligible_params(
    model: torch.nn.Module,
    *,
    include_protected_conv2d: bool = False,
) -> dict[str, torch.nn.Parameter]:
    eligible, _excluded = _select_eligible_params_with_exclusions(
        model,
        include_protected_conv2d=include_protected_conv2d,
    )
    return eligible


def profile_hessian(
    *,
    checkpoint: str,
    video_mkv: str,
    masks_mkv: str,
    poses_path: str | None,
    pair_weights_path: str | None,
    top_k: int,
    output: str,
    device: str,
    upstream_dir: str,
    pair_batch: int = 4,
    score_clip: float = 1e6,
    include_protected_conv2d: bool = False,
) -> dict:
    """End-to-end profiler. See module docstring for output format."""
    t_start = time.monotonic()

    # ── strict device check (CLAUDE.md MPS-CUDA drift = 23x) ──
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but CUDA not available. "
            "Lane Ω profiler runs on CUDA only (MPS Fisher importance has "
            "23x-drift in the PoseNet path per CLAUDE.md). "
            "Pass --device cpu explicitly if you accept advisory-only output."
        )
    if device == "mps":
        raise SystemExit(
            "FATAL: --device mps forbidden. CLAUDE.md non-negotiable: "
            "MPS Fisher importance is unreliable for the contest scorer."
        )

    print(f"[profile] device={device}")
    print(f"[profile] loading renderer: {checkpoint}")
    model = _load_renderer(checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    has_film = hasattr(model, "pose_dim") and getattr(model, "pose_dim", 0) > 0
    has_zoom = bool(getattr(model, "use_zoom_flow", False))
    print(f"[profile]   {type(model).__name__} {n_params:,} params "
          f"film={has_film} zoom={has_zoom}")

    # ── select eligible weights (drops protected layers) ──
    eligible, excluded = _select_eligible_params_with_exclusions(
        model,
        include_protected_conv2d=include_protected_conv2d,
    )
    protected_note = (
        "protected Conv2d included for sensitivity; protected Linear layers stay excluded"
        if include_protected_conv2d
        else "protected layers stay FP32"
    )
    print(f"[profile] eligible weights: {len(eligible)} layers, "
          f"{sum(p.numel() for p in eligible.values()):,} weights "
          f"({protected_note})")
    if include_protected_conv2d:
        print("[profile]   OWV3 mode: protected Conv2d weights included for sensitivity only")

    # ── load poses if model has FiLM ──
    poses = None
    pose_dim_default = int(getattr(model, "pose_dim", 6) or 6)
    if has_film:
        if poses_path is None:
            raise SystemExit(
                "FATAL: model has FiLM (pose_dim={}) but no --poses provided. "
                "Lane Ω profiler would compute meaningless gradients with "
                "zero poses (PoseNet collapses 32x — see "
                "feedback_film_eval_no_poses_critical).".format(model.pose_dim)
            )
        poses = _load_poses(poses_path, pose_dim_default).to(device)
        print(f"[profile]   poses loaded: {tuple(poses.shape)}")

    # ── decode GT video + masks (CPU) ──
    gt_frames_cpu = _decode_gt_video(video_mkv)  # (N, 3, H, W) uint8
    masks_cpu = _load_masks_video(masks_mkv)  # (N, H, W) int64
    n_frames = gt_frames_cpu.shape[0]
    n_pairs = n_frames // 2
    if masks_cpu.shape[0] != n_frames:
        raise SystemExit(
            f"FATAL: gt_frames has {n_frames} but masks has "
            f"{masks_cpu.shape[0]} — must match"
        )
    print(f"[profile] {n_frames} frames → {n_pairs} pairs")

    # ── pair weights (Lane W profile or uniform fallback) ──
    pair_w = _load_pair_weights(pair_weights_path, n_pairs).to(device)
    if pair_weights_path is None:
        print(f"[profile]   pair_weights=uniform (1/{n_pairs})")
    else:
        # Pick top_k hardest by weight; renormalize over only those
        topk_idx = torch.topk(pair_w, k=min(top_k, n_pairs)).indices
        mask = torch.zeros_like(pair_w)
        mask[topk_idx] = pair_w[topk_idx]
        pair_w = mask / mask.sum().clamp(min=1e-30)
        print(f"[profile]   pair_weights: top-{top_k} hardest "
              f"(min={pair_w[pair_w > 0].min().item():.6f} max={pair_w.max().item():.6f})")

    # ── load scorers (differentiable variant for backprop) ──
    print(f"[profile] loading differentiable scorers from {upstream_dir} ...")
    from tac.scorer import load_differentiable_scorers

    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)
    posenet.eval()
    segnet.eval()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)

    # ── ego_flow / zoom_warp setup if needed ──
    zoom_warp = None
    if has_zoom:
        from tac.radial_zoom import RadialZoomWarp

        print("[profile]   instantiating RadialZoomWarp")
        zoom_warp = RadialZoomWarp().to(device)

    # ── prepare per-weight grad accumulators ──
    importance: dict[str, torch.Tensor] = {
        name: torch.zeros_like(p, device="cpu", dtype=torch.float64)
        for name, p in eligible.items()
    }

    # Make eligible weights require_grad even though model is in eval mode.
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in eligible.items():
        p.requires_grad_(True)

    # ── iterate over pairs that have nonzero weight ──
    nonzero_idx = (pair_w > 0).nonzero(as_tuple=True)[0].tolist()
    n_seen = 0
    print(f"[profile] iterating {len(nonzero_idx)} pairs with nonzero weight "
          f"(batch={pair_batch})")

    OUT_H_SCORER = 384
    OUT_W_SCORER = 512

    for batch_start in range(0, len(nonzero_idx), pair_batch):
        batch = nonzero_idx[batch_start : batch_start + pair_batch]
        if not batch:
            break

        # Gather paired (mask_t, mask_t+1, gt_t, gt_t+1) for each p ∈ batch
        m_t_list = []
        m_t1_list = []
        gt_t_list = []
        gt_t1_list = []
        pose_list = [] if has_film else None
        weight_list = []
        for p in batch:
            j = p * 2
            m_t_list.append(masks_cpu[j])
            m_t1_list.append(masks_cpu[j + 1])
            gt_t_list.append(gt_frames_cpu[j])
            gt_t1_list.append(gt_frames_cpu[j + 1])
            if has_film:
                pose_list.append(poses[p])
            weight_list.append(pair_w[p])

        masks_t = torch.stack(m_t_list).to(device, dtype=torch.long)
        masks_t1 = torch.stack(m_t1_list).to(device, dtype=torch.long)
        # GT frames in (B, 3, H, W) uint8 → float
        gt_t = torch.stack(gt_t_list).to(device).float()  # (B, 3, H, W)
        gt_t1 = torch.stack(gt_t1_list).to(device).float()

        kwargs = {}
        if has_film:
            kwargs["pose"] = torch.stack(pose_list).to(device)
        if has_zoom and zoom_warp is not None:
            pair_indices = torch.tensor(batch, device=device)
            ego_flow = zoom_warp(pair_indices, masks_t.shape[1], masks_t.shape[2])
            kwargs["ego_flow"] = ego_flow

        # Forward: model emits (B, 2, H, W, 3) RGB float in [0, 255]
        pairs_pred = model(masks_t, masks_t1, **kwargs)
        # → (B, 2, 3, H, W) for downstream scoring
        pairs_pred = pairs_pred.permute(0, 1, 4, 2, 3)
        # Resize each frame to scorer input
        B, T, C, H, W = pairs_pred.shape
        pred_flat = pairs_pred.reshape(B * T, C, H, W)
        pred_resized = F.interpolate(
            pred_flat, size=(OUT_H_SCORER, OUT_W_SCORER),
            mode="bilinear", align_corners=False,
        ).reshape(B, T, C, OUT_H_SCORER, OUT_W_SCORER)
        # GT is at native resolution (e.g. 874x1164); resize to scorer input
        gt_t_r = F.interpolate(
            gt_t, size=(OUT_H_SCORER, OUT_W_SCORER),
            mode="bilinear", align_corners=False,
        )
        gt_t1_r = F.interpolate(
            gt_t1, size=(OUT_H_SCORER, OUT_W_SCORER),
            mode="bilinear", align_corners=False,
        )
        gt_pair = torch.stack([gt_t_r, gt_t1_r], dim=1)  # (B, 2, 3, H, W)

        # PoseNet expects (B, T, 3, H, W) per scorer.preprocess_input contract
        with torch.enable_grad():
            posenet_pred = posenet.preprocess_input(pred_resized)
            posenet_gt = posenet.preprocess_input(gt_pair)
            pose_out_pred = posenet(posenet_pred)
            pose_out_gt = posenet(posenet_gt)
            # PoseNet distortion = MSE of first half of pose head
            pose_dim_h = pose_out_pred["pose"].shape[-1] // 2
            pose_dist_per_pair = (
                pose_out_pred["pose"][..., :pose_dim_h]
                - pose_out_gt["pose"][..., :pose_dim_h]
            ).pow(2).mean(dim=tuple(range(1, pose_out_pred["pose"].ndim)))
            # SegNet: argmax disagreement is non-differentiable, but we can use
            # softmax-CE between pred and GT logits as the differentiable proxy.
            # This matches what TTO + training use (loss not score).
            segnet_pred = segnet.preprocess_input(pred_resized)
            segnet_gt = segnet.preprocess_input(gt_pair)
            seg_logits_pred = segnet(segnet_pred)
            with torch.no_grad():
                seg_target = segnet(segnet_gt).argmax(dim=1)
            seg_dist_per_pair = F.cross_entropy(
                seg_logits_pred, seg_target, reduction="none",
            ).mean(dim=(1, 2))

            # Per-pair contribution to score: 100 * seg + sqrt(10 * pose).
            # We avoid sqrt at zero (gradient is inf) by adding a tiny eps.
            score_per_pair = (
                100.0 * seg_dist_per_pair
                + torch.sqrt(10.0 * pose_dist_per_pair + 1e-12)
            ).clamp(max=score_clip)

            # Weight by pair importance and sum.
            batch_w = torch.stack(weight_list).to(device)
            loss = (score_per_pair * batch_w).sum()

            # Backward — accumulate Fisher importance (∂L/∂w)² per element.
            for p in eligible.values():
                if p.grad is not None:
                    p.grad = None
            loss.backward()
            for name, p in eligible.items():
                if p.grad is None:
                    continue
                # Accumulate squared-grad on CPU as float64 (avoid CUDA OOM
                # for the accumulator since we keep ALL importance tensors).
                importance[name] += (
                    p.grad.detach().pow(2).to(torch.float64).cpu()
                )
        n_seen += len(batch)
        if (batch_start // pair_batch) % 10 == 0:
            print(f"[profile]   processed {n_seen}/{len(nonzero_idx)} pairs", flush=True)

    # ── normalize: Fisher importance is averaged over hard-pair count ──
    # We weight-summed; if pair_w sums to 1 then `loss` already averages.
    # For safety we report the sum of pair_w that contributed.
    contributed_w = float(pair_w[(pair_w > 0)].sum().item())
    if contributed_w > 0:
        for name in importance:
            importance[name] = importance[name] / contributed_w

    # ── stats ──
    all_imp = torch.cat([t.reshape(-1) for t in importance.values()])
    stats = {
        "imp_min": float(all_imp.min().item()),
        "imp_max": float(all_imp.max().item()),
        "imp_mean": float(all_imp.mean().item()),
        "imp_median": float(all_imp.median().item()),
        "imp_p95": float(all_imp.quantile(0.95).item()),
        "imp_p99": float(all_imp.quantile(0.99).item()),
    }
    print(f"[profile] importance stats: {stats}")

    # ── git hash for provenance ──
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
        "poses": poses_path,
        "masks_mkv": masks_mkv,
        "video_mkv": video_mkv,
        "pair_weights": pair_weights_path,
        "top_k": top_k,
        "n_pairs_seen": n_seen,
        "n_eligible_layers": len(eligible),
        "n_eligible_weights": int(sum(t.numel() for t in importance.values())),
        "eligible_layer_names": sorted(eligible),
        "excluded_layer_reasons": dict(sorted(excluded.items())),
        "include_protected_conv2d": bool(include_protected_conv2d),
        "git_hash": git_hash,
        "torch_version": torch.__version__,
        "device": device,
        "elapsed_s": time.monotonic() - t_start,
        **stats,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"importance": {n: t.contiguous() for n, t in importance.items()},
         "metadata": metadata},
        out_path,
    )
    sidecar = out_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(metadata, indent=2))
    print(f"[profile] WROTE {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"[profile] WROTE {sidecar} (metadata sidecar)")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lane Ω Phase 1: per-weight Fisher importance profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Renderer .bin or .pt (default: Lane A's renderer.bin)")
    parser.add_argument("--video", required=True,
                        help="GT video .mkv (1200 frames)")
    parser.add_argument("--masks-mkv", required=True,
                        help="Masks .mkv (matches frame count of GT video)")
    parser.add_argument("--poses", default=None,
                        help="Poses .pt or .bin (REQUIRED if model has FiLM)")
    parser.add_argument("--upstream", default="upstream",
                        help="Upstream dir (default: upstream)")
    parser.add_argument("--output", required=True,
                        help="Output .pt path for hessian_per_weight tensor")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of hardest pairs to weight (default 30)")
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument(
        "--pair-weights", default=None,
        help="Path to Lane W's pair_weights.pt (preferred)",
    )
    weight_group.add_argument(
        "--all-pairs", action="store_true",
        help="Use uniform 1/600 weighting over ALL pairs (Lane W fallback)",
    )
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device (default cuda; MPS forbidden)")
    parser.add_argument("--pair-batch", type=int, default=4,
                        help="Pairs per backward step (default 4)")
    parser.add_argument(
        "--include-protected-conv2d",
        action="store_true",
        help=(
            "Include SC-protected Conv2d weights in Fisher profiling. Required "
            "for OWV3 sensitivity-map conversion because every Conv2d action "
            "must have measured CUDA sensitivity. Protected Linear layers stay "
            "excluded."
        ),
    )
    args = parser.parse_args(argv)

    pair_weights_path = None if args.all_pairs else args.pair_weights

    profile_hessian(
        checkpoint=args.checkpoint,
        video_mkv=args.video,
        masks_mkv=args.masks_mkv,
        poses_path=args.poses,
        pair_weights_path=pair_weights_path,
        top_k=args.top_k,
        output=args.output,
        device=args.device,
        upstream_dir=args.upstream,
        pair_batch=args.pair_batch,
        include_protected_conv2d=args.include_protected_conv2d,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
