# SPDX-License-Identifier: MIT
"""Capstone campaign runner (Task #78/#65) — the missing local actuator.

Trains the ORIGINAL VQ-NeRV + FiLM-pose capstone basis against the LIVE frozen
contest scorer (the #82 1:1-MLX-port bridge), at a chosen byte budget
(``base_channels`` + ``decoder_dtype``), byte-closes the int8 archive, and
recomputes the EXACT advisory score from the live-render d_seg/d_pose + the
real archive.zip size (``evaluate.py`` law). $0, local, MLX-renderer +
torch-CPU-scorer (TRUSTED per CLAUDE.md "local CPU + MLX GPU good"; MPS NEVER).

This is the thin CLI per CLAUDE.md "thin CLIs delegate to tac modules": every
real piece lives in ``tac`` — ``score_aware_loop.targets`` (frozen DistortionNet
+ GT targets), ``mlx_pr95_port.score_bridge`` (the torch<->mlx vjp bridge),
``capstone_vq_nerv`` (bundle + trainer + int8 export).

Authority: the score this prints is ``[macOS-CPU advisory]`` (the torch scorer
on local CPU is trusted but it is NOT a contest-axis row). It RANKS + gates; it
does NOT move the canonical frontier pointer. A sub-0.15 advisory here is the
GATE to a paired contest-CPU+CUDA exact eval (the only pointer-moving step).

Usage (smoke):
    .venv/bin/python experiments/run_capstone_campaign.py \
        --max-pairs 2 --base-channels 16 --epochs 3 --decoder-dtype int8 \
        --out-dir experiments/results/capstone_smoke

Usage (decisive budget run, local detached daemon):
    nohup .venv/bin/python experiments/run_capstone_campaign.py \
        --max-pairs 600 --base-channels 16 --epochs 300 --decoder-dtype int8 \
        --muon-lr 3e-2 --grad-clip 50 --grad-clip-muon 50 \
        --out-dir experiments/results/capstone_full_b16_int8 \
        > .omx/tmp/capstone_full_b16_int8.log 2>&1 &
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

RATE_DENOM = 37_545_489  # evaluate.py:64


def _minimal_zip(payload: bytes, member: str = "x") -> bytes:
    """Wrap ``payload`` in a STORED single-member ZIP (the #79 100 B-floor
    container the frontier uses). The result is exactly what evaluate.py counts
    (``archive.zip``)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(member, payload)
    return buf.getvalue()


def _load_or_build_targets(max_pairs: int, cache_dir: Path, device: str):
    """Cache (seg_targets_hard, pose_targets) — the slow GT precompute. Reusable
    across base_channels / epochs sweeps."""
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"gt_targets_n{max_pairs}.pt"
    net = load_frozen_distortion_net(device=device)
    if cache.exists():
        blob = torch.load(cache, map_location=device, weights_only=False)
        seg_t, pose_t, n = blob["seg"], blob["pose"], int(blob["n"])
        print(f"[targets] loaded cache {cache} n={n}", flush=True)
        return net, seg_t, pose_t, n
    t0 = time.time()
    seg_t, pose_t, n = build_gt_targets(net, max_pairs=max_pairs, device=device)
    torch.save({"seg": seg_t, "pose": pose_t, "n": n}, cache)
    print(f"[targets] built+cached n={n} in {time.time()-t0:.1f}s -> {cache}", flush=True)
    return net, seg_t, pose_t, n


def _export_int8_archive(bundle, pose_store: np.ndarray, decoder_dtype: str):
    """Extract the trained carrier (decoder weights + codebook + REAL trained vq
    indices + stored pose) and byte-close the int8 archive. Returns
    (archive_zip_bytes, account, payload_bytes)."""
    from mlx.utils import tree_flatten

    from tac.capstone_vq_nerv.export import build_capstone_archive_bytes

    flat = tree_flatten(bundle.trainable_parameters())
    decoder_weights = {}
    for k, v in flat:
        if k.startswith("latents"):  # the VQ index is the carrier; latents not stored
            continue
        decoder_weights[k] = np.asarray(v, dtype=np.float32)
    codebook = np.asarray(bundle.quantizer._codebook, dtype=np.float32)
    vq_indices = np.asarray(bundle.all_vq_indices(), dtype=np.int32)  # REAL trained carrier
    payload, account = build_capstone_archive_bytes(
        decoder_weights=decoder_weights,
        codebook=codebook,
        vq_indices=vq_indices,
        pose_scalars=np.asarray(pose_store, dtype=np.float32),
        codebook_size=int(codebook.shape[0]),
        decoder_dtype=decoder_dtype,
    )
    archive_zip = _minimal_zip(payload)
    return archive_zip, account, payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=2)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--decoder-dtype", choices=("fp16", "int8"), default="int8")
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument("--muon-lr", type=float, default=3e-2)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--grad-clip-muon", type=float, default=50.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")  # torch scorer device; NEVER mps
    ap.add_argument("--out-dir", default="experiments/results/capstone_smoke")
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    args = ap.parse_args()

    if args.device == "mps":
        raise SystemExit("MPS is NEVER an authority (CLAUDE.md). Use --device cpu.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    net, seg_t, pose_t, n = _load_or_build_targets(
        args.max_pairs, Path(args.targets_cache), args.device
    )

    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle, CapstoneVqNervConfig
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n, base_channels=args.base_channels,
            codebook_size=args.codebook_size, seed=args.seed,
        )
    )
    bridge = TorchScorerBridge(
        net, seg_t, pose_t,
        seg_loss_form="ce_seg_loss",
        seg_weight=args.seg_weight, pose_weight=args.pose_weight,
        eval_roundtrip=True,
    )
    pose_store = pose_t.float().cpu().numpy()
    cfg = CapstoneTrainConfig(
        epochs=args.epochs, seg_weight=args.seg_weight, pose_weight=args.pose_weight,
        muon_lr=args.muon_lr, grad_clip=args.grad_clip, grad_clip_muon=args.grad_clip_muon,
        eval_every=args.eval_every, seed=args.seed,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)

    d_seg_init = trainer.exact_d_seg()
    d_pose_init = trainer.mean_d_pose()
    print(f"[init] n={n} base_ch={args.base_channels} d_seg={d_seg_init:.5f} "
          f"d_pose={d_pose_init:.5f}", flush=True)

    train_out = trainer.train()
    d_seg = trainer.exact_d_seg()
    d_pose = trainer.mean_d_pose()

    archive_zip, account, payload = _export_int8_archive(bundle, pose_store, args.decoder_dtype)
    archive_bytes = len(archive_zip)
    (out / "archive.zip").write_bytes(archive_zip)

    rate_term = 25.0 * archive_bytes / RATE_DENOM
    seg_term = 100.0 * d_seg
    pose_term = float(np.sqrt(10.0 * d_pose))
    score = seg_term + pose_term + rate_term

    summary = {
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "n_pairs": n,
        "base_channels": args.base_channels,
        "decoder_dtype": args.decoder_dtype,
        "epochs": args.epochs,
        "muon_lr": args.muon_lr,
        "grad_clip": args.grad_clip,
        "d_seg_init": d_seg_init, "d_pose_init": d_pose_init,
        "d_seg_final": d_seg, "d_pose_final": d_pose,
        "archive_bytes": archive_bytes,
        "payload_bytes": len(payload),
        "decoder_bytes": account.decoder_bytes,
        "codebook_bytes": account.codebook_bytes,
        "score_seg_contribution": seg_term,
        "score_pose_contribution": pose_term,
        "score_rate_contribution": rate_term,
        "advisory_score": score,
        "sub_0_15": score < 0.15,
        "sub_0_19": score < 0.19,
        "wall_s": time.time() - t_start,
        "traj": train_out.get("traj") if isinstance(train_out, dict) else None,
    }
    (out / "capstone_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nADVISORY S = {score:.5f}  (seg {seg_term:.4f} + pose {pose_term:.4f} "
          f"+ rate {rate_term:.4f})  [macOS-CPU advisory, NOT a pointer move]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
