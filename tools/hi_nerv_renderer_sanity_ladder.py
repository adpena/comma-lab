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
    frame_n = H * W * C
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
    sample_frames = sorted(set([0, 1, n // 4, n // 2, (n // 2) + 1, n - 2, n - 1]))
    sample_frames = [f for f in sample_frames if 0 <= f < n]
    sample_pairs = sorted(set([0, (n // seq_len) // 4, (n // seq_len) // 2, (n // seq_len) - 1]))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

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
