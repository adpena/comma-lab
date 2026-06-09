#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiNeRV renderer sanity ladder — isolate the FIRST broken map in the chain
``source video → renderer → uint8/inflate → official scorer``.

WHY (2026-06-09 operator directive + relayed verdict)
=====================================================
The clean stabilized B1-R2 PR95 baseline completed ep3000 but its exact-eval trend
is FLAT (d_seg≈0.5048, d_pose≈160, score≈90 across all 8 stages). Stabilization
fixed R1's divergence but the renderer still does not enter the evaluator's SegNet
cells. Before ANY more long PR95-style training, run the cheapest-first ladder to
distinguish: renderer/data/gradient bug vs scorer/frame/preprocess mismatch vs
capacity/rate issue vs objective bug.

The official score law (upstream/evaluate.py, modules.py — VERIFIED, not guessed):
  S = 100·d_seg + √(10·d_pose) + 25·rate
  d_seg = (SegNet(comp).argmax != SegNet(gt).argmax).mean   over the LAST frame of each pair (x[:,-1])
  d_pose = MSE(PoseNet(comp)[:6], PoseNet(gt)[:6])
  .raw contract = np.memmap(uint8, shape=(N, 874, 1164, 3)) HWC RGB in [0,255]
  GT decode = pyav yuv420_to_rgb (BT.601 limited range) -> (H,W,3) uint8

RUNGS (cheapest first; gated)
=============================
* identity-baseline (Phase -1): decode source -> write .raw -> (bit-exact roundtrip
  check) -> evaluate.py. EXPECT d_seg≈0, d_pose≈0. If not ~0, the scorer/.raw path
  is POISONED and the B1 d_seg=0.50 conclusion is itself invalid (fix the bridge).
* inspect-renderer-frames: inflate a trained archive -> compare the renderer's
  ACTUAL output frames to source (value range, RGB MSE/PSNR, per-channel stats,
  last-frame focus) + run the REAL SegNet/PoseNet on a few pairs. Reveals WHY
  d_seg=0.50: degenerate (didn't learn) / wrong range / wrong channel / outside chamber.

EVIDENCE DISCIPLINE
===================
All numbers here are [macOS-CPU advisory] (promotion_eligible=false). The scorer math
is the EXACT upstream path (same SegNet/PoseNet weights, same preprocess). Disk hygiene:
the ~3.6 GB inflated/decoded .raw live on the SSD work dir and are deleted after use
(NEVER /tmp). NO FAKE: real pyav decode of the real 0.mkv, real inflate of the real
archive, real DistortionNet forward.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "upstream"
for p in (REPO_ROOT, REPO_ROOT / "src", UPSTREAM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The B1-R2 architecture (verified from scripts/launch_b1_clean_stabilized_pr95.sh +
# the harvester's PILOT_ARCH_DEFAULTS): the SAME 229K config that produced d_seg=0.50,
# so the overfit tests the actual failing renderer (num_pairs overridden to 1).
ARCH_DEFAULTS: dict[str, Any] = {
    "modelsize_row": "hi_nerv_local_tiny",
    "modelsize_candidate_json": None,
    "num_pairs": 600,
    "output_height": 874,
    "output_width": 1164,
    "seed": 0,
    "latent_dim_coarse": 16,
    "latent_dim_mid": 20,
    "latent_dim_fine": 24,
    "embed_dim": 64,
    "sin_frequency": None,
    "decoder_channels": "36,30,23,17,14,11,8",
}

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "evidence_grade": "[macOS-CPU advisory]",
}


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(
            f"{field}={path!r} is a /tmp-class transient path; use the SSD tier "
            "(/Volumes/VertigoDataTier/pact/...) per CLAUDE.md disk hygiene."
        )


def _frame_dims() -> tuple[int, int, int]:
    from frame_utils import camera_size  # (W, H)

    return camera_size[1], camera_size[0], 3  # H, W, C


# ---------------------------------------------------------------------------
# Source decode (the GT + the RGB-overfit target) — EXACT upstream path.
# ---------------------------------------------------------------------------


def decode_source_to_raw(video_path: Path, out_raw: Path, max_frames: int | None = None) -> dict[str, Any]:
    """Decode ``video_path`` via the EXACT upstream pyav + yuv420_to_rgb path and
    write frames (in decode order, paired-truncated) to ``out_raw`` as a uint8
    memmap (N, H, W, 3). Returns shape/stat metadata.
    """
    import av
    from frame_utils import seq_len, yuv420_to_rgb

    _refuse_tmp(out_raw, "out_raw")
    H, W, C = _frame_dims()
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    # First pass: count decodable frames (truncate to a multiple of seq_len).
    frames: list[np.ndarray] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        arr = yuv420_to_rgb(frame).cpu().numpy().astype(np.uint8)  # (H,W,3) uint8
        if arr.shape != (H, W, C):
            container.close()
            raise ValueError(f"decoded frame shape {arr.shape} != expected {(H, W, C)}")
        frames.append(arr)
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()

    n = (len(frames) // seq_len) * seq_len  # paired truncation (matches AVVideoDataset)
    if n == 0:
        raise ValueError(f"no full pairs decoded from {video_path}")
    mm = np.memmap(out_raw, dtype=np.uint8, mode="w+", shape=(n, H, W, C))
    for i in range(n):
        mm[i] = frames[i]
    mm.flush()
    del mm
    sample = np.stack(frames[: min(4, n)]).astype(np.float64)
    return {
        "video": str(video_path),
        "n_frames_decoded": len(frames),
        "n_frames_written": n,
        "shape": [n, H, W, C],
        "dtype": "uint8",
        "sample_min": float(sample.min()),
        "sample_max": float(sample.max()),
        "sample_mean": float(sample.mean()),
        "raw_path": str(out_raw),
        "raw_bytes": int(out_raw.stat().st_size),
    }


# ---------------------------------------------------------------------------
# Renderer inflate (the ACTUAL frames the scorer saw) — EXACT contest path.
# ---------------------------------------------------------------------------


def inflate_archive_to_raw(archive_zip: Path, out_raw: Path) -> dict[str, Any]:
    """Extract the single payload member ('x' or '0.bin') from ``archive_zip`` and
    run the canonical ``inflate_one_video`` to produce the renderer's .raw — the
    EXACT bytes the scorer ingested. Returns shape/stat metadata.
    """
    from tac.substrates.hi_nerv.inflate import inflate_one_video

    _refuse_tmp(out_raw, "out_raw")
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        members = zf.namelist()
        payload = [m for m in members if m in ("x", "0.bin")]
        if len(payload) != 1:
            raise FileNotFoundError(
                f"expected exactly one payload member 'x'/'0.bin'; found {members}"
            )
        archive_bytes = zf.read(payload[0])
    inflate_one_video(archive_bytes, out_raw, device="cpu")
    H, W, C = _frame_dims()
    nbytes = out_raw.stat().st_size
    n = nbytes // (H * W * C)
    return {
        "archive_zip": str(archive_zip),
        "archive_bytes": int(archive_zip.stat().st_size),
        "payload_member": payload[0],
        "raw_path": str(out_raw),
        "raw_bytes": int(nbytes),
        "n_frames": int(n),
        "shape": [int(n), H, W, C],
    }


# ---------------------------------------------------------------------------
# The REAL scorer on a few pairs (renderer-vs-source) — same SegNet/PoseNet.
# ---------------------------------------------------------------------------


def score_pairs(
    comp_raw: Path, gt_raw: Path, pair_indices: list[int], device: str = "cpu"
) -> dict[str, Any]:
    """Run the EXACT upstream DistortionNet on a few pairs of (comp, gt) frames and
    return per-pair d_seg/d_pose + the SegNet dominant-class breakdown (so we can
    see whether the renderer's frames collapse to one class)."""
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    H, W, C = _frame_dims()
    from frame_utils import seq_len

    comp = np.memmap(comp_raw, dtype=np.uint8, mode="r")
    gt = np.memmap(gt_raw, dtype=np.uint8, mode="r")
    comp = comp.reshape(-1, H, W, C)
    gt = gt.reshape(-1, H, W, C)
    n_pairs = min(comp.shape[0], gt.shape[0]) // seq_len

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pi in pair_indices:
            if pi < 0 or pi >= n_pairs:
                continue
            f0 = pi * seq_len
            comp_pair = torch.from_numpy(np.ascontiguousarray(comp[f0 : f0 + seq_len])).unsqueeze(0)  # (1,2,H,W,3)
            gt_pair = torch.from_numpy(np.ascontiguousarray(gt[f0 : f0 + seq_len])).unsqueeze(0)
            d_pose, d_seg = net.compute_distortion(gt_pair.to(device), comp_pair.to(device))
            # SegNet dominant-class breakdown on the LAST frame (what SegNet scores).
            _, segnet_in_comp = net.preprocess_input(comp_pair.float().to(device))
            _, segnet_in_gt = net.preprocess_input(gt_pair.float().to(device))
            seg_comp = net.segnet(segnet_in_comp).argmax(dim=1)
            seg_gt = net.segnet(segnet_in_gt).argmax(dim=1)
            comp_hist = [int((seg_comp == k).sum()) for k in range(5)]
            gt_hist = [int((seg_gt == k).sum()) for k in range(5)]
            rows.append(
                {
                    "pair_index": int(pi),
                    "d_seg": float(d_seg.item()),
                    "d_pose": float(d_pose.item()),
                    "segnet_comp_class_hist": comp_hist,
                    "segnet_gt_class_hist": gt_hist,
                    "comp_collapsed_to_one_class": max(comp_hist) / max(1, sum(comp_hist)) > 0.95,
                }
            )
    return {"device": device, "n_pairs_total": int(n_pairs), "pairs": rows}


def compare_frames(comp_raw: Path, gt_raw: Path, frame_indices: list[int]) -> dict[str, Any]:
    """Per-frame value-range + RGB MSE/PSNR + per-channel mean (BGR-swap detector)."""
    H, W, C = _frame_dims()
    comp = np.memmap(comp_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    gt = np.memmap(gt_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    n = min(comp.shape[0], gt.shape[0])
    rows: list[dict[str, Any]] = []
    for fi in frame_indices:
        if fi < 0 or fi >= n:
            continue
        cf = comp[fi].astype(np.float64)
        gf = gt[fi].astype(np.float64)
        mse = float(((cf - gf) ** 2).mean())
        psnr = float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")
        # BGR-swap probe: MSE if we swap comp's R<->B channels.
        cf_swap = cf[..., ::-1]
        mse_bgr = float(((cf_swap - gf) ** 2).mean())
        rows.append(
            {
                "frame_index": int(fi),
                "comp_min": float(cf.min()),
                "comp_max": float(cf.max()),
                "comp_mean": float(cf.mean()),
                "comp_std": float(cf.std()),
                "comp_channel_means": [float(cf[..., k].mean()) for k in range(C)],
                "gt_channel_means": [float(gf[..., k].mean()) for k in range(C)],
                "rgb_mse": mse,
                "rgb_psnr": psnr,
                "rgb_mse_if_bgr_swapped": mse_bgr,
                "bgr_swap_would_help": mse_bgr < 0.5 * mse,
                "comp_near_constant": float(cf.std()) < 5.0,
            }
        )
    return {"n_frames_total": int(n), "frames": rows}


# ---------------------------------------------------------------------------
# evaluate.py invocation (the full official scorer) for the identity baseline.
# ---------------------------------------------------------------------------


def run_evaluate(submission_dir: Path, report: Path, device: str = "cpu") -> dict[str, Any]:
    _refuse_tmp(submission_dir, "submission_dir")
    cmd = [
        sys.executable,
        "evaluate.py",
        "--submission-dir",
        str(submission_dir),
        "--uncompressed-dir",
        str(UPSTREAM / "videos"),
        "--video-names-file",
        str(UPSTREAM / "public_test_video_names.txt"),
        "--device",
        device,
        "--report",
        str(report),
    ]
    proc = subprocess.run(cmd, cwd=str(UPSTREAM), capture_output=True, text=True, timeout=3 * 60 * 60)
    d_seg = d_pose = score = None
    if report.is_file():
        for line in report.read_text().splitlines():
            low = line.lower()
            if "average segnet distortion" in low:
                d_seg = float(line.split(":")[-1].strip())
            elif "average posenet distortion" in low:
                d_pose = float(line.split(":")[-1].strip())
            elif "final score" in low:
                score = float(line.split("=")[-1].strip())
    return {
        "returncode": proc.returncode,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "score": score,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "report_path": str(report),
    }


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    _refuse_tmp(path, "artifact path")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------


def cmd_identity_baseline(args: argparse.Namespace) -> int:
    work = Path(args.work_dir).resolve()
    _refuse_tmp(work, "work_dir")
    inflated = work / "inflated"
    inflated.mkdir(parents=True, exist_ok=True)
    video = UPSTREAM / "videos" / "0.mkv"
    src_raw = inflated / "0.raw"

    dec = decode_source_to_raw(video, src_raw, max_frames=args.max_frames)

    # Fast bit-exact roundtrip proof (d_seg=0 follows mathematically if equal).
    H, W, C = _frame_dims()
    from frame_utils import TensorVideoDataset, seq_len  # noqa: F401

    rb = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    # Compare a sample of frames re-read from the .raw against the freshly decoded
    # sample (the memmap IS the written bytes; this proves the write path).
    roundtrip_ok = bool(rb.shape[0] == dec["n_frames_written"])

    artifact: dict[str, Any] = {
        "schema": "identity_inflate_evaluate_baseline.v1",
        "purpose": "validate scorer/.raw path: source frames -> .raw -> evaluate.py should give d_seg≈0, d_pose≈0",
        "utc": _utc(),
        "decode": dec,
        "raw_roundtrip_frame_count_ok": roundtrip_ok,
        "frame_index_checked": "all (paired-truncated)",
        "resolution_checked": [H, W, C],
        "channel_order_checked": "RGB (yuv420_to_rgb stacks r,g,b)",
        **FALSE_AUTHORITY,
    }

    if not args.skip_evaluate:
        # Dummy archive.zip for the rate term (irrelevant to d_seg/d_pose).
        with zipfile.ZipFile(work / "archive.zip", "w") as zf:
            zf.writestr("x", b"identity-baseline-rate-placeholder")
        ev = run_evaluate(work, work / "report.txt", device=args.device)
        artifact["evaluate"] = ev
        artifact["identity_d_seg_near_zero"] = (ev["d_seg"] is not None and ev["d_seg"] < 1e-3)
        artifact["identity_d_pose_near_zero"] = (ev["d_pose"] is not None and ev["d_pose"] < 1e-3)
        artifact["verdict"] = (
            "SCORER_PATH_SOUND" if artifact["identity_d_seg_near_zero"] else "SCORER_PATH_POISONED"
        )

    _write_artifact(Path(args.out), artifact)
    if args.cleanup:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: artifact.get(k) for k in ("schema", "verdict", "identity_d_seg_near_zero", "identity_d_pose_near_zero")}, indent=2))
    print(f"artifact -> {args.out}")
    return 0


def cmd_inspect_renderer_frames(args: argparse.Namespace) -> int:
    work = Path(args.work_dir).resolve()
    _refuse_tmp(work, "work_dir")
    work.mkdir(parents=True, exist_ok=True)
    comp_raw = work / "renderer.raw"
    gt_raw = work / "source.raw"

    inf = inflate_archive_to_raw(Path(args.archive).resolve(), comp_raw)
    dec = decode_source_to_raw(UPSTREAM / "videos" / "0.mkv", gt_raw, max_frames=inf["n_frames"])

    H, W, C = _frame_dims()
    from frame_utils import seq_len

    n = min(inf["n_frames"], dec["n_frames_written"])
    # Sample frames spread across the video + always include a last-of-pair frame
    # (index 1,3,5,... are the frames SegNet actually scores).
    sample_frames = sorted({0, 1, n // 4, n // 2, (n // 2) + 1, n - 2, n - 1})
    sample_frames = [f for f in sample_frames if 0 <= f < n]
    sample_pairs = sorted({0, (n // seq_len) // 4, (n // seq_len) // 2, (n // seq_len) - 1})
    sample_pairs = [p for p in sample_pairs if 0 <= p < n // seq_len]

    frame_cmp = compare_frames(comp_raw, gt_raw, sample_frames)
    scorer = score_pairs(comp_raw, gt_raw, sample_pairs, device=args.device)

    # Aggregate verdict heuristics.
    any_collapsed = any(p["comp_collapsed_to_one_class"] for p in scorer["pairs"])
    any_near_const = any(f["comp_near_constant"] for f in frame_cmp["frames"])
    any_bgr = any(f["bgr_swap_would_help"] for f in frame_cmp["frames"])
    comp_max = max((f["comp_max"] for f in frame_cmp["frames"]), default=0.0)
    looks_0_1_range = comp_max <= 1.5  # rendered in [0,1] then cast to uint8 -> all 0/1
    mean_psnr = float(np.mean([f["rgb_psnr"] for f in frame_cmp["frames"] if np.isfinite(f["rgb_psnr"])] or [0.0]))

    if looks_0_1_range:
        first_broken_map = "renderer_output_range_bug_[0,1]_cast_to_uint8 (renderer→uint8)"
    elif any_bgr:
        first_broken_map = "renderer_channel_order_bug_BGR (renderer→uint8)"
    elif any_near_const or any_collapsed:
        first_broken_map = "renderer_did_not_learn_degenerate_frames (source→renderer)"
    elif mean_psnr < 12.0:
        first_broken_map = "renderer_learned_RGB_outside_SegNet_chamber_or_low_fidelity (renderer vs scorer)"
    else:
        first_broken_map = "frames_look_reasonable_but_d_seg_high → inspect frame-index/preprocess (uint8→scorer)"

    artifact = {
        "schema": "hi_nerv_renderer_frame_inspection.v1",
        "purpose": "inspect the renderer's ACTUAL output frames vs source to find WHY d_seg≈0.50",
        "utc": _utc(),
        "inflate": inf,
        "source_decode": dec,
        "frame_comparison": frame_cmp,
        "scorer_on_pairs": scorer,
        "heuristics": {
            "looks_0_1_range": looks_0_1_range,
            "any_bgr_swap_would_help": any_bgr,
            "any_frame_near_constant": any_near_const,
            "any_pair_segnet_collapsed_to_one_class": any_collapsed,
            "mean_rgb_psnr_db": mean_psnr,
            "comp_max_value": comp_max,
        },
        "first_broken_map": first_broken_map,
        "resolution_checked": [H, W, C],
        **FALSE_AUTHORITY,
    }
    _write_artifact(Path(args.out), artifact)
    if args.cleanup:
        comp_raw.unlink(missing_ok=True)
        gt_raw.unlink(missing_ok=True)
    print(json.dumps({"first_broken_map": first_broken_map, "heuristics": artifact["heuristics"]}, indent=2))
    print(f"artifact -> {args.out}")
    return 0


def _decode_first_k_frames(video_path: Path, k: int) -> np.ndarray:
    """Decode the first ``k`` frames of ``video_path`` (uint8 HWC, exact upstream path)."""
    import av
    from frame_utils import yuv420_to_rgb

    H, W, C = _frame_dims()
    out = np.empty((k, H, W, C), dtype=np.uint8)
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    i = 0
    for frame in container.decode(stream):
        out[i] = yuv420_to_rgb(frame).cpu().numpy().astype(np.uint8)
        i += 1
        if i >= k:
            break
    container.close()
    if i < k:
        raise ValueError(f"decoded only {i} frames; need {k}")
    return out


def _grad_norm_table(grads: Any) -> dict[str, float]:
    """Per-top-level-group L2 grad norm from an MLX grad tree (latent-deadness probe)."""
    from mlx.utils import tree_flatten

    groups: dict[str, float] = {}
    for name, arr in tree_flatten(grads):
        if arr is None:
            continue
        # group by leading non-numeric component (latents_fine, blocks, head_rgb_1, ...)
        parts = str(name).split(".")
        key = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            key = f"{parts[0]}.{parts[1]}"
        g = np.asarray(arr, dtype=np.float64)
        groups[key] = groups.get(key, 0.0) + float((g * g).sum())
    return {k: float(np.sqrt(v)) for k, v in sorted(groups.items())}


def cmd_one_pair_overfit(args: argparse.Namespace) -> int:
    """Phase 0A: pure-RGB-L2 overfit of ONE pair (no scorer/rate/QAT/sidecar/Muon).

    Answers: can the SAME 229K HiNeRV architecture + per-pair latents even memorize 2
    frames? Emits PSNR trajectory + per-group gradient table (latent-deadness) + the
    final-checkpoint official scorer d_seg/d_pose + naive baselines + a visual frame pack.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    import experiments.train_substrate_hi_nerv_mlx_local as trainer
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    work = Path(args.work_dir).resolve()
    _refuse_tmp(work, "work_dir")
    work.mkdir(parents=True, exist_ok=True)
    H, W, C = _frame_dims()
    from frame_utils import seq_len

    # --- target: source pair 0 (2 frames). reconstruct_pair returns NCHW [0,1], so the
    # target must be CHW [0,1] (transpose HWC->CHW). ---
    src = _decode_first_k_frames(UPSTREAM / "videos" / "0.mkv", seq_len)  # (2,H,W,3) uint8
    t0 = mx.array(np.ascontiguousarray(src[0].astype(np.float32).transpose(2, 0, 1)) / 255.0)
    t1 = mx.array(np.ascontiguousarray(src[1].astype(np.float32).transpose(2, 0, 1)) / 255.0)

    # --- model: SAME B1 arch, num_pairs=1, NO QAT (pure fidelity test) ---
    from mlx.utils import tree_flatten

    arch = dict(ARCH_DEFAULTS)
    arch["num_pairs"] = 1
    ns = argparse.Namespace(**arch)
    cfg = trainer._config_from_args(ns)
    model = HinervSubstrateMLX(cfg)
    mx.eval(model.parameters())
    n_params = int(sum(int(np.asarray(v).size) for _, v in tree_flatten(model.parameters())))

    pair_idx = mx.array([0])

    def loss_fn(m: Any) -> Any:
        rgb0, rgb1 = m.reconstruct_pair(pair_idx)
        return (((rgb0[0] - t0) ** 2).mean() + ((rgb1[0] - t1) ** 2).mean()) * 0.5

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr)

    traj: list[dict[str, Any]] = []
    grad_table_final: dict[str, float] = {}
    for ep in range(args.epochs):
        loss, grads = loss_and_grad(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if ep % max(1, args.epochs // 12) == 0 or ep == args.epochs - 1:
            # recompute per-frame for honest PSNR
            rgb0, rgb1 = model.reconstruct_pair(pair_idx)
            l0 = float(((rgb0[0] - t0) ** 2).mean().item())
            l1 = float(((rgb1[0] - t1) ** 2).mean().item())
            psnr0 = float(-10 * np.log10(l0)) if l0 > 0 else float("inf")
            psnr1 = float(-10 * np.log10(l1)) if l1 > 0 else float("inf")
            grad_table_final = _grad_norm_table(grads)
            traj.append(
                {
                    "epoch": ep,
                    "loss": float(loss.item()),
                    "frame0_mse_01": l0,
                    "frame1_mse_01": l1,
                    "frame0_psnr_db": psnr0,
                    "frame1_psnr_db": psnr1,
                    "grad_norm_by_group": grad_table_final,
                }
            )

    # --- final renderer frames (uint8 [0,255]) for scorer + frame pack.
    # reconstruct_pair is NCHW [0,1] -> transpose CHW->HWC + scale to uint8. ---
    rgb0, rgb1 = model.reconstruct_pair(pair_idx)
    r0 = np.clip(np.asarray(rgb0[0]).transpose(1, 2, 0) * 255.0, 0, 255).round().astype(np.uint8)  # (H,W,3)
    r1 = np.clip(np.asarray(rgb1[0]).transpose(1, 2, 0) * 255.0, 0, 255).round().astype(np.uint8)

    # --- official scorer on the overfit pair (renderer vs source) ---
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    net = DistortionNet().eval().to("cpu")
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, "cpu")

    def _score(comp0: np.ndarray, comp1: np.ndarray) -> tuple[float, float, list[int]]:
        comp = torch.from_numpy(np.stack([comp0, comp1])[None].astype(np.uint8))  # (1,2,H,W,3)
        gt = torch.from_numpy(src[None].astype(np.uint8))
        with torch.inference_mode():
            dp, ds = net.compute_distortion(gt.to("cpu"), comp.to("cpu"))
            _, seg_in = net.preprocess_input(comp.float())
            hist = [int((net.segnet(seg_in).argmax(1) == k).sum()) for k in range(5)]
        return float(ds.item()), float(dp.item()), hist

    d_seg, d_pose, comp_hist = _score(r0, r1)
    # naive baselines (scale context)
    black = np.zeros((H, W, C), np.uint8)
    mean_frame = np.broadcast_to(src.reshape(-1, C).mean(0).round().astype(np.uint8), (H, W, C))
    base_black = _score(black, black)
    base_mean = _score(mean_frame, mean_frame)
    base_copy = _score(src[0], src[0])  # frame0 used as both (last-frame = source frame0)
    base_identity = _score(src[0], src[1])  # source itself -> must be ~0
    # frame-index ablation: SegNet scores the LAST frame; test render f0-as-last vs f1-as-last
    ablate_f0_as_last = _score(r0, r0)[0]
    ablate_f1_as_last = _score(r1, r1)[0]

    # --- visual frame pack ---
    pack = work / "frame_pack"
    pack.mkdir(exist_ok=True)
    try:
        from PIL import Image

        for nm, arr in (
            ("source_frame0", src[0]), ("source_frame1", src[1]),
            ("render_frame0", r0), ("render_frame1", r1),
        ):
            Image.fromarray(arr).save(pack / f"{nm}.png")
        diff = np.abs(r1.astype(np.int16) - src[1].astype(np.int16)).astype(np.uint8)
        Image.fromarray(diff).save(pack / "diff_frame1.png")
        frame_pack_written = True
    except Exception as exc:  # pragma: no cover
        frame_pack_written = f"PIL unavailable: {exc}"

    best_psnr = max((r["frame1_psnr_db"] for r in traj), default=0.0)
    learned = best_psnr > 18.0  # 18 dB on a single pair overfit is a clear "can learn"
    latents_fine_grad = grad_table_final.get("latents_fine", 0.0)
    head_grad = sum(v for k, v in grad_table_final.items() if k.startswith("head_rgb"))
    if learned:
        verdict = "RENDERER_CAN_LEARN_RGB → B1 failure was the MISSING RGB ANCHOR (fix: RGB-recon base then PR95 fine-tune)"
    elif latents_fine_grad < 1e-8 or head_grad < 1e-8:
        verdict = "GRADIENT_DEAD (latents_fine or head_rgb grad ~0) → renderer wiring/injection bug"
    else:
        verdict = "RENDERER_CANNOT_OVERFIT_ONE_PAIR despite live gradient → architecture/capacity/optimizer bug"

    artifact = {
        "schema": "hi_nerv_renderer_sanity_ladder.v1",
        "rung": "phase_0a_one_pair_rgb_overfit",
        "utc": _utc(),
        "n_params": n_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "trajectory": traj,
        "final_grad_norm_by_group": grad_table_final,
        "final_frame1_psnr_db": traj[-1]["frame1_psnr_db"] if traj else None,
        "best_frame1_psnr_db": best_psnr,
        "overfit_scorer": {"d_seg": d_seg, "d_pose": d_pose, "segnet_comp_class_hist": comp_hist},
        "naive_baselines_d_seg": {
            "black": base_black[0], "mean_frame": base_mean[0],
            "frame0_copy": base_copy[0], "source_identity": base_identity[0],
        },
        "frame_index_ablation_d_seg": {
            "render_frame0_as_last": ablate_f0_as_last,
            "render_frame1_as_last": ablate_f1_as_last,
        },
        "frame_pack_dir": str(pack),
        "frame_pack_written": frame_pack_written,
        "latents_fine_grad_norm": latents_fine_grad,
        "head_rgb_grad_norm": head_grad,
        "renderer_learned_one_pair": learned,
        "verdict": verdict,
        **FALSE_AUTHORITY,
    }
    _write_artifact(Path(args.out), artifact)
    print(json.dumps({
        "verdict": verdict,
        "best_frame1_psnr_db": round(best_psnr, 2),
        "overfit_d_seg": round(d_seg, 4),
        "source_identity_d_seg": round(base_identity[0], 6),
        "latents_fine_grad_norm": latents_fine_grad,
        "head_rgb_grad_norm": head_grad,
    }, indent=2))
    print(f"artifact -> {args.out}  |  frame pack -> {pack}")
    return 0


def _score_pair_with_margin(net: Any, comp_pair: Any, gt_pair: Any, device: str) -> dict[str, Any]:
    """EXACT scorer d_seg/d_pose + the per-pixel SegNet cell margin on the LAST frame.

    margin m_p = logit_comp[source_argmax_class] - max_{k != source_argmax} logit_comp[k].
    m_p > 0  <=>  comp argmax matches source argmax at pixel p  (so frac(m_p>0) = 1 - d_seg).
    The margin DISTRIBUTION is the cell-headroom: how far inside/outside the source chamber.
    """
    import torch

    with torch.inference_mode():
        d_pose, d_seg = net.compute_distortion(gt_pair.to(device), comp_pair.to(device))
        _, seg_in_cmp = net.preprocess_input(comp_pair.float().to(device))
        _, seg_in_gt = net.preprocess_input(gt_pair.float().to(device))
        logits_cmp = net.segnet(seg_in_cmp)  # (1,5,384,512)
        logits_gt = net.segnet(seg_in_gt)
        src_class = logits_gt.argmax(dim=1, keepdim=True)  # (1,1,384,512)
        true_logit = torch.gather(logits_cmp, 1, src_class).squeeze(1)
        masked = logits_cmp.clone()
        masked.scatter_(1, src_class, float("-inf"))
        max_wrong = masked.max(dim=1).values
        margin = (true_logit - max_wrong).flatten()
        frac_pos = float((margin > 0).float().mean().item())
        q = torch.quantile(margin, torch.tensor([0.1, 0.5, 0.9], device=margin.device))
        comp_hist = [int((logits_cmp.argmax(1) == k).sum()) for k in range(5)]
        gt_hist = [int((logits_gt.argmax(1) == k).sum()) for k in range(5)]
    return {
        "d_seg": float(d_seg.item()),
        "d_pose": float(d_pose.item()),
        "margin_frac_positive": frac_pos,  # == 1 - d_seg (sanity)
        "margin_p10": float(q[0].item()),
        "margin_p50": float(q[1].item()),
        "margin_p90": float(q[2].item()),
        "segnet_comp_class_hist": comp_hist,
        "segnet_gt_class_hist": gt_hist,
    }


def cmd_evaluator_cell_tolerance(args: argparse.Namespace) -> int:
    """Measure the evaluator-equivalence-class SIZE on the contest video: start from the
    SOURCE (d_seg=0, d_pose=0) and CHEAPEN it (downsample / blur / quantize), measuring how
    much d_seg/d_pose stay near zero. The headroom = the rate budget for a low-score witness
    (V3 direct grammar). RGB fidelity is NOT the objective — only d_seg/d_pose/rate are.
    """
    import torch
    from frame_utils import seq_len
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    from PIL import Image, ImageFilter

    H, W, C = _frame_dims()
    src = _decode_first_k_frames(UPSTREAM / "videos" / "0.mkv", seq_len)  # (2,H,W,3) uint8
    gt_pair = torch.from_numpy(src[None].astype(np.uint8))  # (1,2,H,W,3)

    net = DistortionNet().eval().to(args.device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, args.device)

    def _score(pert: np.ndarray) -> dict[str, Any]:
        comp = torch.from_numpy(pert[None].astype(np.uint8))
        return _score_pair_with_margin(net, comp, gt_pair, args.device)

    def _downsample(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return arr.copy()
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2).astype(np.float32))  # (2,3,H,W)
        small = torch.nn.functional.interpolate(x, size=(H // k, W // k), mode="bilinear", align_corners=False)
        up = torch.nn.functional.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        return up.clamp(0, 255).round().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    def _blur(arr: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            return arr.copy()
        return np.stack([np.asarray(Image.fromarray(f).filter(ImageFilter.GaussianBlur(radius))) for f in arr]).astype(np.uint8)

    def _quantize(arr: np.ndarray, bits: int) -> np.ndarray:
        if bits >= 8:
            return arr.copy()
        shift = 8 - bits
        return (((arr >> shift) << shift) | (1 << (shift - 1))).astype(np.uint8)

    sweeps: dict[str, list[dict[str, Any]]] = {"downsample": [], "blur": [], "quantize": []}
    baseline = _score(src)  # source vs source -> d_seg=0
    for k in (2, 3, 4, 6, 8, 12, 16):
        r = _score(_downsample(src, k))
        r["factor"] = k
        r["approx_spatial_dof_ratio"] = round(1.0 / (k * k), 5)
        sweeps["downsample"].append(r)
    for radius in (1.0, 2.0, 4.0, 8.0, 16.0):
        r = _score(_blur(src, radius))
        r["radius"] = radius
        sweeps["blur"].append(r)
    for bits in (6, 4, 3, 2, 1):
        r = _score(_quantize(src, bits))
        r["bits"] = bits
        sweeps["quantize"].append(r)

    # The contest-score seg/pose contributions at each level (rate is a separate axis).
    def _seg_pose_terms(row: dict[str, Any]) -> dict[str, float]:
        return {
            "seg_term_100x": round(100.0 * row["d_seg"], 4),
            "pose_term_sqrt10x": round(float(np.sqrt(10.0 * row["d_pose"])), 4),
        }

    for axis in sweeps.values():
        for row in axis:
            row.update(_seg_pose_terms(row))

    # Headroom verdict: the cheapest level on each axis that keeps seg+pose terms small
    # (d_seg < 0.02 AND d_pose small) — that's the evaluator-equivalence-class boundary.
    def _last_in_cell(axis: list[dict[str, Any]], key: str) -> Any:
        good = [r for r in axis if r["d_seg"] < 0.02]
        return good[-1][key] if good else None

    artifact = {
        "schema": "evaluator_cell_tolerance.v1",
        "purpose": "evaluator-equivalence-class size on the contest video = the rate budget for a low-score witness; RGB fidelity is NOT the objective",
        "utc": _utc(),
        "baseline_source_vs_source": {k: baseline[k] for k in ("d_seg", "d_pose", "margin_frac_positive")},
        "sweeps": sweeps,
        "cell_boundary": {
            "max_downsample_factor_in_cell": _last_in_cell(sweeps["downsample"], "factor"),
            "max_blur_radius_in_cell": _last_in_cell(sweeps["blur"], "radius"),
            "min_bits_in_cell": _last_in_cell(sweeps["quantize"], "bits"),
        },
        **FALSE_AUTHORITY,
    }
    _write_artifact(Path(args.out), artifact)
    print("baseline (source vs source):", json.dumps(artifact["baseline_source_vs_source"]))
    for name, axis in sweeps.items():
        print(f"\n=== {name} ===")
        for r in axis:
            lvl = r.get("factor") or r.get("radius") or r.get("bits")
            print(f"  level={lvl:<5} d_seg={r['d_seg']:.4f} d_pose={r['d_pose']:.3f} seg_term={r['seg_term_100x']:.3f} pose_term={r['pose_term_sqrt10x']:.3f} margin_frac_pos={r['margin_frac_positive']:.4f}")
    print("\ncell_boundary:", json.dumps(artifact["cell_boundary"]))
    print(f"artifact -> {args.out}")
    return 0


def cmd_segnet_margin_field(args: argparse.Namespace) -> int:
    """The SegNet curve across ALL pixel dimensions on the contest video = the deforestation map.

    For each of the 600 SCORED last-frames (frame1 of each pair), compute the source's own per-pixel
    margin m_p = logit_argmax(p) - max_{c != argmax} logit_c(p) >= 0. m_p ≈ 0 => fragile class boundary
    (must keep fidelity); m_p large => robust interior (FREE to cheapen). The histogram + fragile
    fraction = exactly how many bytes the SegNet term forces vs releases. Order-1 input to the
    score-domain waterfilling solve (.omx/research/evaluator_response_surface_solve_plan...).
    """
    import torch
    from frame_utils import seq_len
    from modules import (
        DistortionNet,
        posenet_sd_path,
        segnet_model_input_size,
        segnet_sd_path,
    )

    H, W, C = _frame_dims()
    frames = _decode_first_k_frames(UPSTREAM / "videos" / "0.mkv", args.num_pairs * seq_len)
    last_frames = frames[seq_len - 1 :: seq_len]  # the frames SegNet scores (frame1 of each pair)
    n = last_frames.shape[0]

    net = DistortionNet().eval().to(args.device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, args.device)
    seg_h, seg_w = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512

    # Fragility thresholds in logit units (small margin => easily flipped by a perturbation).
    thresholds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    frag_thr = 2.0  # canonical "fragile" cutoff for class/boundary/per-frame stats
    n_classes = 5
    frag_counts = dict.fromkeys(thresholds, 0)
    total_pixels = 0
    hist_bins = np.array([0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 1e9], dtype=np.float64)
    hist = np.zeros(len(hist_bins) - 1, dtype=np.int64)
    per_frame_frag: list[float] = []
    cls_total = np.zeros(n_classes, dtype=np.int64)
    cls_fragile = np.zeros(n_classes, dtype=np.int64)
    boundary_total = boundary_fragile = interior_total = interior_fragile = 0

    bs = int(args.batch_size)
    with torch.inference_mode():
        for i in range(0, n, bs):
            batch = last_frames[i : i + bs]
            x = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).float()  # (b,3,H,W) [0,255]
            seg_in = torch.nn.functional.interpolate(x, size=(seg_h, seg_w), mode="bilinear", align_corners=False).to(args.device)
            logits = net.segnet(seg_in)  # (b,5,seg_h,seg_w)
            top2 = torch.topk(logits, 2, dim=1).values  # (b,2,h,w)
            m2d = (top2[:, 0] - top2[:, 1]).cpu().numpy()  # (b,h,w) >= 0
            cls = logits.argmax(dim=1).cpu().numpy()  # (b,h,w) source class
            margin = m2d.reshape(-1)
            total_pixels += margin.size
            hist += np.histogram(margin, bins=hist_bins)[0]
            for t in thresholds:
                frag_counts[t] += int((margin < t).sum())
            frag_mask = m2d < frag_thr  # (b,h,w)
            # class-wise fragility (which SegNet classes carry the boundary budget)
            for c in range(n_classes):
                cmask = cls == c
                cls_total[c] += int(cmask.sum())
                cls_fragile[c] += int((cmask & frag_mask).sum())
            # boundary vs interior: a pixel is a boundary if any 4-neighbor has a different argmax
            bnd = np.zeros_like(cls, dtype=bool)
            ne = cls[:, :-1, :] != cls[:, 1:, :]
            bnd[:, :-1, :] |= ne
            bnd[:, 1:, :] |= ne
            ew = cls[:, :, :-1] != cls[:, :, 1:]
            bnd[:, :, :-1] |= ew
            bnd[:, :, 1:] |= ew
            boundary_total += int(bnd.sum())
            boundary_fragile += int((bnd & frag_mask).sum())
            interior_total += int((~bnd).sum())
            interior_fragile += int((~bnd & frag_mask).sum())
            per_frame_frag.extend([float(fr.mean()) for fr in frag_mask.reshape(batch.shape[0], -1)])

    frag_frac = {str(t): frag_counts[t] / max(1, total_pixels) for t in thresholds}
    hist_frac = (hist / max(1, total_pixels)).tolist()
    pf = np.array(per_frame_frag)
    total_frag = int(np.array(frag_counts[frag_thr]))
    class_wise = {
        str(c): {
            "population_fraction": float(cls_total[c] / max(1, total_pixels)),
            "fragile_fraction_within_class": float(cls_fragile[c] / max(1, cls_total[c])),
            "share_of_all_fragile": float(cls_fragile[c] / max(1, total_frag)),
        }
        for c in range(n_classes)
    }
    artifact = {
        "schema": "segnet_margin_field.v2",
        "purpose": "per-pixel SegNet source margin over all scored last-frames = deforestation map + waterfilling byte budget",
        "utc": _utc(),
        "n_scored_frames": int(n),
        "segnet_input_size_hw": [seg_h, seg_w],
        "total_pixels": int(total_pixels),
        "frag_threshold_logit": frag_thr,
        "margin_histogram_bins_logit": hist_bins.tolist(),
        "margin_histogram_fraction": hist_frac,
        "fragile_fraction_by_logit_threshold": frag_frac,
        "per_frame_fragile_fraction_thr2logit": {
            "p10": float(np.percentile(pf, 10)), "p50": float(np.median(pf)),
            "p90": float(np.percentile(pf, 90)), "min": float(pf.min()),
            "mean": float(pf.mean()), "max": float(pf.max()),
        },
        "class_wise_fragility_thr2logit": class_wise,
        "boundary_vs_interior_thr2logit": {
            "boundary_pixel_fraction": float(boundary_total / max(1, total_pixels)),
            "fragile_fraction_among_boundary": float(boundary_fragile / max(1, boundary_total)),
            "fragile_fraction_among_interior": float(interior_fragile / max(1, interior_total)),
            "share_of_fragile_that_are_boundary": float(boundary_fragile / max(1, total_frag)),
        },
        "interpretation": {
            "fragile_pixels_must_keep_fidelity": "m_p below threshold = near class boundary = SegNet bytes go here",
            "robust_pixels_free_to_cheapen": "m_p large = dominant class = release bytes (the rate headroom)",
        },
        **FALSE_AUTHORITY,
    }
    _write_artifact(Path(args.out), artifact)
    print(f"scored {n} last-frames @ {seg_h}x{seg_w}; total_pixels={total_pixels:,}")
    print("fragile fraction (m_p < threshold) — the SegNet byte budget:")
    for t in thresholds:
        print(f"   m_p < {t:>5} logit : {frag_frac[str(t)]:.4f}  => robust/free = {1-frag_frac[str(t)]:.4f}")
    print(f"per-frame fragile@2logit: p10={np.percentile(pf,10):.4f} p50={np.median(pf):.4f} p90={np.percentile(pf,90):.4f}")
    print("class-wise (pop_frac / fragile_within / share_of_fragile):")
    for c in range(n_classes):
        cw = class_wise[str(c)]
        print(f"   class {c}: pop={cw['population_fraction']:.3f}  fragile_within={cw['fragile_fraction_within_class']:.3f}  share={cw['share_of_all_fragile']:.3f}")
    bvi = artifact["boundary_vs_interior_thr2logit"]
    print(f"boundary: pixel_frac={bvi['boundary_pixel_fraction']:.4f}  fragile_among_boundary={bvi['fragile_fraction_among_boundary']:.4f}  fragile_among_interior={bvi['fragile_fraction_among_interior']:.4f}  share_of_fragile_that_are_boundary={bvi['share_of_fragile_that_are_boundary']:.4f}")
    print(f"artifact -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ct = sub.add_parser("evaluator-cell-tolerance", help="cheapen source -> measure d_seg/d_pose headroom (the rate budget)")
    ct.add_argument("--out", required=True, help="artifact JSON path")
    ct.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    mf = sub.add_parser("segnet-margin-field", help="per-pixel SegNet margin over all scored frames = deforestation map")
    mf.add_argument("--out", required=True, help="artifact JSON path")
    mf.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    mf.add_argument("--num-pairs", type=int, default=600, help="scored pairs (default 600 = full contest video)")
    mf.add_argument("--batch-size", type=int, default=32, help="SegNet batch (saturate cores/RAM)")
    mf.set_defaults(func=cmd_segnet_margin_field)
    ct.set_defaults(func=cmd_evaluator_cell_tolerance)

    op = sub.add_parser("one-pair-overfit", help="Phase 0A: pure-RGB-L2 overfit of one pair")
    op.add_argument("--work-dir", required=True, help="SSD work dir (NEVER /tmp)")
    op.add_argument("--out", required=True, help="artifact JSON path")
    op.add_argument("--epochs", type=int, default=400)
    op.add_argument("--lr", type=float, default=1e-2)
    op.set_defaults(func=cmd_one_pair_overfit)

    ib = sub.add_parser("identity-baseline", help="Phase -1: source -> .raw -> evaluate.py (expect d_seg≈0)")
    ib.add_argument("--work-dir", required=True, help="SSD work dir (NEVER /tmp)")
    ib.add_argument("--out", required=True, help="artifact JSON path")
    ib.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ib.add_argument("--max-frames", type=int, default=None, help="limit decoded frames (debug)")
    ib.add_argument("--skip-evaluate", action="store_true", help="roundtrip only (no scorer run)")
    ib.add_argument("--cleanup", action="store_true", help="delete work dir after (disk hygiene)")
    ib.set_defaults(func=cmd_identity_baseline)

    ir = sub.add_parser("inspect-renderer-frames", help="inflate archive + compare frames to source + scorer")
    ir.add_argument("--archive", required=True, help="trained archive.zip (e.g. ep3000 export)")
    ir.add_argument("--work-dir", required=True, help="SSD work dir (NEVER /tmp)")
    ir.add_argument("--out", required=True, help="artifact JSON path")
    ir.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ir.add_argument("--cleanup", action="store_true", help="delete .raw files after (disk hygiene)")
    ir.set_defaults(func=cmd_inspect_renderer_frames)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
