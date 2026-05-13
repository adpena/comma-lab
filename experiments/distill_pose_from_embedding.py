#!/usr/bin/env python3
"""Lane M-V3 (Path A): distill a pose-from-embedding MLP at compress time.

The MLP learns to predict the renderer's 6-DOF FiLM pose conditioning from
(PoseNet 12-dim head output, mask-pair features). At inflate the embedding
input is zeroed (PoseNet not loaded — strict-scorer-rule), so the MLP is
trained with EMBEDDING DROPOUT to learn the mask-only path equally well.

Pipeline
--------
1. Load Lane A renderer + Lane A optimized poses (target labels) +
   GT video + masks (for both PoseNet GT extraction AND mask features).
2. Extract PoseNet GT embeddings: PoseNet(GT_pair) → (P, 12) tensor. This
   is the "embedding" input to the MLP at compress time.
3. Train the MLP to map (embedding, mask_features) → Lane A's optimized
   poses, with embedding-dropout p=0.5 so the MLP MUST work without the
   embedding input at inflate.
4. Save the MLP (FP16 ~1-2 KB) + the canonical sentinel file.

Usage
-----
    python experiments/distill_pose_from_embedding.py \\
        --renderer        experiments/results/lane_a_landed/iter_0/renderer.bin \\
        --target-poses    experiments/results/lane_a_landed/optimized_poses.pt \\
        --masks           experiments/results/lane_a_landed/extracted/masks.mkv \\
        --gt-video        upstream/videos/0.mkv \\
        --upstream        upstream \\
        --output-dir      experiments/results/lane_m_v3 \\
        --device          cuda

Outputs
-------
* ``<output-dir>/pose_from_embedding_v1.pt``  - MLP weights (FP16)
* ``<output-dir>/pose_from_embedding_v1``     - 0-byte sentinel file
* ``<output-dir>/distill.log``                - training log
* ``<output-dir>/distill_provenance.json``    - SHA256s + metrics

Strict-scorer-rule
------------------
PoseNet IS loaded HERE (compress time only). The output MLP and sentinel
go into the archive; PoseNet does NOT. At inflate, the MLP runs on mask
features alone (zero embedding input) — verified by the dropout-trained
embedding pathway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

from tac.pose_from_embedding import (  # noqa: E402
    POSE_FROM_EMBEDDING_SENTINEL,
    POSE_FROM_EMBEDDING_WEIGHTS_FILENAME,
    PoseFromEmbeddingMLP,
    save_mlp,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _decode_masks(masks_mkv: Path) -> torch.Tensor:
    """Decode masks.mkv → ``(N, H, W)`` long-tensor of class indices."""
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            f"PyAV is required to decode {masks_mkv}: {e!r}. "
            "Install with `uv pip install av` and retry."
        )
    container = av.open(str(masks_mkv))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="gray")
        frames.append(torch.from_numpy(arr))
    container.close()
    if not frames:
        raise SystemExit(f"FATAL: {masks_mkv} decoded zero frames")
    masks_uint8 = torch.stack(frames)
    scale_factor = 255 // 4
    return (masks_uint8.float() / scale_factor).round().long().clamp(0, 4)


def _decode_video(video_path: Path) -> torch.Tensor:
    """Decode GT video → ``(N, H, W, 3)`` uint8 tensor (HWC, RGB)."""
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            f"PyAV is required to decode {video_path}: {e!r}."
        )
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    container.close()
    if not frames:
        raise SystemExit(f"FATAL: {video_path} decoded zero frames")
    return torch.stack(frames)  # (N, H, W, 3)


def _extract_posenet_embeddings(
    gt_video: torch.Tensor,
    posenet: nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract PoseNet 12-dim head output for non-overlapping pairs.

    Returns ``(P, 12)`` float tensor where P = N // 2.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    n_frames = gt_video.shape[0]
    n_pairs = n_frames // 2
    embeddings = []
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_pairs = []
        for k in range(start, end):
            f0 = gt_video[2 * k].float()
            f1 = gt_video[2 * k + 1].float()
            pair = torch.stack([f0, f1], dim=0)
            batch_pairs.append(pair)
        pairs = torch.stack(batch_pairs).to(device)
        # (B, T, H, W, C) → (B, T, C, H, W)
        pairs_chw = pairs.permute(0, 1, 4, 2, 3).contiguous()
        b, t, c, h, w = pairs_chw.shape
        if h != SEGNET_INPUT_H or w != SEGNET_INPUT_W:
            pairs_flat = pairs_chw.reshape(b * t, c, h, w)
            pairs_flat = F.interpolate(
                pairs_flat,
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            pairs_chw = pairs_flat.reshape(b, t, c, SEGNET_INPUT_H, SEGNET_INPUT_W)
        posenet_in = posenet.preprocess_input(pairs_chw)
        with torch.no_grad():
            out = posenet(posenet_in)
        # The full 12-dim head output (NOT just the first 6)
        embeddings.append(out["pose"].cpu())
    return torch.cat(embeddings, dim=0).float()  # (P, 12)


def _train(
    mlp: PoseFromEmbeddingMLP,
    embeddings: torch.Tensor,        # (P, 12)
    masks: torch.Tensor,             # (N, H, W) long
    target_poses: torch.Tensor,      # (P, 6) float
    *,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 3e-3,
    embedding_dropout_p: float = 0.5,
    log_interval: int = 20,
    log_lines: list[str],
) -> dict:
    """Train the MLP with embedding-dropout supervision."""
    mlp = mlp.to(device).train()
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    n_pairs = target_poses.shape[0]
    if embeddings.shape[0] != n_pairs:
        raise SystemExit(
            f"FATAL: embeddings have {embeddings.shape[0]} pairs but "
            f"target_poses have {n_pairs} (mismatch from different runs)"
        )
    if masks.shape[0] != 2 * n_pairs:
        raise SystemExit(
            f"FATAL: masks decode to {masks.shape[0]} frames but expected "
            f"2 * {n_pairs} = {2 * n_pairs}"
        )

    # Pre-compute mask features for all pairs ONCE (saves an order of
    # magnitude in training time vs recomputing per epoch). The feature
    # extractor itself is part of the MLP so we compute features
    # forward-pass-by-forward-pass during training.
    # Pair the masks now so each training step reads (P, 2, H, W).
    paired = torch.stack([masks[0::2], masks[1::2]], dim=1)  # (P, 2, H, W)
    paired = paired.to(device, non_blocking=True)
    embeddings = embeddings.to(device, non_blocking=True)
    target_poses = target_poses.to(device, non_blocking=True)

    metrics = {
        "epochs_run": 0, "best_loss": float("inf"),
        "best_loss_no_emb": float("inf"),
        "loss_history": [], "loss_history_no_emb": [],
    }
    rng = torch.Generator(device="cpu").manual_seed(42)

    for epoch in range(epochs):
        # Shuffle pair indices on CPU rng (deterministic)
        perm = torch.randperm(n_pairs, generator=rng)
        epoch_losses_with = []
        epoch_losses_without = []
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            idx = perm[start:end].to(device)
            batch_paired = paired[idx]                # (B, 2, H, W)
            batch_emb = embeddings[idx]               # (B, 12)
            batch_target = target_poses[idx]          # (B, 6)

            # One-hot the masks here (mirrors inflate-side path)
            batch_onehot = F.one_hot(
                batch_paired.clamp(0, mlp.n_classes - 1),
                num_classes=mlp.n_classes,
            ).permute(0, 1, 4, 2, 3).contiguous()
            batch_onehot = batch_onehot.reshape(
                batch_paired.shape[0], 2 * mlp.n_classes, *batch_paired.shape[-2:],
            ).float()
            mask_features = mlp.feature_extractor(batch_onehot)

            # Embedding dropout: drop the WHOLE embedding for some samples
            # so the MLP learns a mask-only path. At inflate the embedding
            # is ALWAYS zero, so this is the regime that matters.
            keep_mask = (torch.rand(batch_emb.shape[0], 1, device=device) >= embedding_dropout_p).float()
            emb_in = batch_emb * keep_mask

            pred = mlp.forward(emb_in, mask_features)
            loss = F.mse_loss(pred, batch_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses_with.append(loss.detach().item())

            # Eval the no-embedding path on the SAME batch (no grad)
            mlp.eval()
            with torch.no_grad():
                pred_no_emb = mlp.forward(torch.zeros_like(batch_emb), mask_features)
                loss_no_emb = F.mse_loss(pred_no_emb, batch_target)
            mlp.train()
            epoch_losses_without.append(float(loss_no_emb.item()))

        epoch_loss = float(sum(epoch_losses_with) / max(1, len(epoch_losses_with)))
        epoch_loss_no_emb = float(sum(epoch_losses_without) / max(1, len(epoch_losses_without)))
        metrics["loss_history"].append(epoch_loss)
        metrics["loss_history_no_emb"].append(epoch_loss_no_emb)
        metrics["epochs_run"] = epoch + 1
        metrics["best_loss"] = min(metrics["best_loss"], epoch_loss)
        metrics["best_loss_no_emb"] = min(metrics["best_loss_no_emb"], epoch_loss_no_emb)
        if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
            line = (
                f"[distill] epoch {epoch + 1}/{epochs}  "
                f"loss(with-emb)={epoch_loss:.5f}  "
                f"loss(no-emb)={epoch_loss_no_emb:.5f}  "
                f"best={metrics['best_loss']:.5f}/{metrics['best_loss_no_emb']:.5f}"
            )
            print(line, file=sys.stderr)
            log_lines.append(line)
    mlp.eval()
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--renderer", type=Path, required=True,
                   help="Lane A renderer.bin (anchor)")
    p.add_argument("--target-poses", type=Path, required=True,
                   help="Lane A optimized_poses.pt — supervision targets")
    p.add_argument("--masks", type=Path, required=True,
                   help="Lane A masks.mkv (used at compress + inflate)")
    p.add_argument("--gt-video", type=Path, required=True,
                   help="Ground-truth video for PoseNet embedding extraction")
    p.add_argument("--upstream", type=Path, required=True,
                   help="Path to upstream repo root (for PoseNet weights)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory (will be created)")
    p.add_argument("--device", type=str, required=True,
                   help="cuda (required for PoseNet extraction; CPU forbidden)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--embedding-dropout-p", type=float, default=0.5,
                   help="Probability of zeroing the embedding input "
                        "during training (default 0.5 = balanced)")
    p.add_argument("--log-interval", type=int, default=20)
    args = p.parse_args()

    if args.device not in ("cuda",):
        raise SystemExit(
            f"FATAL: --device must be cuda (got {args.device!r}). "
            "MPS is forbidden (23x PoseNet drift); CPU is forbidden (PoseNet "
            "extraction must match contest scorer arithmetic)."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but torch.cuda.is_available() "
            "is False. CPU fallback would invalidate the PoseNet "
            "embeddings (BT.601 ULP drift, 23x dim-0 noise on MPS — "
            "memory feedback_mps_cuda_drift_critical). Run on a CUDA "
            "host and retry."
        )

    # Validate inputs exist
    for f in (args.renderer, args.target_poses, args.masks, args.gt_video):
        if not f.exists():
            raise SystemExit(f"FATAL: missing {f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def _log(msg: str) -> None:
        line = f"[distill] {msg}"
        print(line, file=sys.stderr)
        log_lines.append(line)

    device = torch.device(args.device)
    _log(f"device = {device}")

    # Load PoseNet (compress-time only)
    from tac.scorer import load_default_scorers
    posenet, _segnet = load_default_scorers(str(args.upstream), device=str(device))
    posenet.eval()
    _log(f"loaded PoseNet from {args.upstream}/models/posenet.safetensors")

    # Decode inputs
    _log(f"decoding masks: {args.masks}")
    masks = _decode_masks(args.masks)
    _log(f"  masks shape: {tuple(masks.shape)} dtype={masks.dtype}")

    _log(f"decoding GT video: {args.gt_video}")
    gt_video = _decode_video(args.gt_video)
    _log(f"  gt_video shape: {tuple(gt_video.shape)}")

    if masks.shape[0] != gt_video.shape[0]:
        raise SystemExit(
            f"FATAL: masks have {masks.shape[0]} frames but GT video has "
            f"{gt_video.shape[0]} frames — pair count mismatch."
        )

    _log(f"loading target poses: {args.target_poses}")
    target_poses = torch.load(str(args.target_poses), map_location="cpu", weights_only=True)
    if target_poses.dim() != 2 or target_poses.shape[1] not in (6,):
        raise SystemExit(
            f"FATAL: target_poses shape {tuple(target_poses.shape)} is not (P, 6). "
            f"Expected Lane A's optimized_poses.pt format."
        )
    _log(f"  target_poses shape: {tuple(target_poses.shape)}")

    if 2 * target_poses.shape[0] != masks.shape[0]:
        raise SystemExit(
            f"FATAL: target_poses pairs ({target_poses.shape[0]}) * 2 != "
            f"masks frames ({masks.shape[0]})"
        )

    _log("extracting PoseNet embeddings on GT pairs (compress time only)...")
    t0 = time.monotonic()
    embeddings = _extract_posenet_embeddings(gt_video, posenet, device)
    dt = time.monotonic() - t0
    _log(f"  extracted {tuple(embeddings.shape)} in {dt:.1f}s; "
         f"dim0 mean={embeddings[:, 0].mean().item():.3f} "
         f"std={embeddings[:, 0].std().item():.3f}")

    # Build + train the MLP
    mlp = PoseFromEmbeddingMLP()
    n_params = sum(p.numel() for p in mlp.parameters())
    _log(f"MLP n_params = {n_params}")

    metrics = _train(
        mlp, embeddings, masks, target_poses,
        device=device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        embedding_dropout_p=args.embedding_dropout_p,
        log_interval=args.log_interval,
        log_lines=log_lines,
    )

    # Save the MLP + sentinel
    weights_path = args.output_dir / POSE_FROM_EMBEDDING_WEIGHTS_FILENAME
    sentinel_path = args.output_dir / POSE_FROM_EMBEDDING_SENTINEL
    bytes_written = save_mlp(mlp, weights_path, fp16=True)
    sentinel_path.write_bytes(b"")
    _log(f"saved MLP weights: {weights_path} ({bytes_written} bytes, FP16)")
    _log(f"wrote sentinel:    {sentinel_path} (0 bytes)")

    # Sanity-evaluate the inflate-side path: zero embedding + masks → poses
    _log("sanity check: predicting poses on first 4 pairs with zero embedding...")
    mlp.eval()
    with torch.no_grad():
        sample_masks = masks[:8].to(device)
        pred = mlp.predict_poses_from_masks(
            sample_masks,
            embedding=torch.zeros(4, mlp.embedding_dim, device=device),
        )
    _log(f"  pred shape: {tuple(pred.shape)} (expected (4, 6))")
    _log(f"  pred dim0:  {pred[:, 0].cpu().tolist()}")
    _log(f"  target dim0:{target_poses[:4, 0].cpu().tolist()}")
    no_emb_rmse = float(((pred.cpu() - target_poses[:4]) ** 2).mean().sqrt().item())
    _log(f"  no-embedding RMSE on first 4 pairs: {no_emb_rmse:.4f}")

    # Provenance
    provenance = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "lane_m_v3_pose_from_embedding_path_a",
        "device": str(device),
        "torch_version": torch.__version__,
        "renderer_sha256": _sha256(args.renderer),
        "target_poses_sha256": _sha256(args.target_poses),
        "masks_sha256": _sha256(args.masks),
        "gt_video_sha256": _sha256(args.gt_video),
        "weights_path": str(weights_path),
        "weights_bytes": int(bytes_written),
        "weights_sha256": _sha256(weights_path),
        "sentinel_path": str(sentinel_path),
        "n_params": int(n_params),
        "n_pairs": int(target_poses.shape[0]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "embedding_dropout_p": float(args.embedding_dropout_p),
        "metrics": metrics,
        "sanity_check": {
            "no_embedding_rmse_first4": no_emb_rmse,
        },
        "predicted_band": [1.10, 1.18],
    }
    prov_path = args.output_dir / "distill_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    _log(f"wrote provenance: {prov_path}")

    log_path = args.output_dir / "distill.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    _log(f"wrote log:        {log_path}")

    print(f"\n[distill] DONE — MLP weights at {weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
