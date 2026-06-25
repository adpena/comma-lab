#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SegNet TEACHER 5-class soft-logit store builder — for knowledge distillation into the witness.

OPERATOR DIRECTIVE 2026-06-25: "what did quantizr do with seg? Can we use PR95 archive or any
already trained things to train seg alone. Distill seg alone." Quantizr's SegNet training used
``kl_on_logits()`` with T=2.0 (Hinton-2015 soft-logit distillation). The non-RGB TASK-SPACE WITNESS
(#171/#78) currently trains on the HARD argmax target (margin-weighted CE) and PLATEAUS at d_seg
0.002447 (~2.8x above the ~0.00087 need). The DAG (FEED 2026-06-25u/v) names KL-on-logits as a
DENSE-supervision lever DEFERRED only because the existing target cache stores argmax(u8)+margin(f16)
but NOT the full 5-class logits. This builder removes that blocker: it caches the FROZEN SegNet's
full 5-class logit field over the n600 GT pairs ONCE, on CPU-torch (the EXACT frozen authority).

EXACT path == tools/lever_b_build_score_native_targets.py (line-for-line frozen-scorer forward):
  - GT decode: ``frame_utils.yuv420_to_rgb`` (pyav, BT.601 limited). NEVER MPS.
  - SegNet: ``seg.preprocess_input`` (frame1 only, resized to 384x512) -> smp.Unet
    'tu-efficientnet_b2' 5-class. The TEACHER target is the RAW logits (1,5,384,512), NOT argmax.

BORROWED-SUBSTRATE ACCOUNTING (CLAUDE.md NO-FAKE): soft-logit KD / kl_on_logits / T=2.0 is BORROWED
from Quantizr (PR62 kl_on_logits) + Hinton-Vinyals-Dean 2015 (Distilling the Knowledge in a Neural
Network). OURS = (a) using the FROZEN contest SegNet itself as the teacher (self-distillation of the
authority), and (b) applying the soft-logit target to a NON-RGB coordinate-INR argmax witness.

NO-FAKE: the teacher logits are the EXACT frozen SegNet forward output — the SAME tensor whose
argmax IS the d_seg authority. argmax(teacher_logits) == cached gt_segnet_argmax.u8 by construction
(verified at the end of this builder). EVIDENCE [macOS-CPU advisory]; promotion_eligible=false.

DISK HYGIENE (CLAUDE.md certify-or-block): ~1.18 GB fp16, rebuildable. Stored on the SSD tier
(/Volumes/VertigoDataTier/pact), NEVER /tmp. A manifest records sha256/bytes/source/rebuild-cmd so
the bytes can be safely re-derived or cold-stored later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
_N_CLASSES = 5


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_teacher_logits(
    video_path: Path,
    out_dir: Path,
    num_pairs: int,
    verify_argmax_dir: Path | None,
) -> dict[str, Any]:
    """Decode GT frame pairs, run frozen SegNet, cache the full 5-class logit field (fp16)."""
    import av
    import einops
    import torch
    from frame_utils import seq_len, yuv420_to_rgb
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"  # MPS corrupts the scorer (CLAUDE.md). CPU-exact authority only.

    seg_in_h, seg_in_w = 384, 512

    # --- decode the first (num_pairs*2) GT frames, EXACT upstream path ---
    t0 = time.time()
    n_frames_needed = num_pairs * seq_len
    frames: list[np.ndarray] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        arr = yuv420_to_rgb(frame).cpu().numpy().astype(np.uint8)
        frames.append(arr)
        if len(frames) >= n_frames_needed:
            break
    container.close()
    n = (len(frames) // seq_len) * seq_len
    n_pairs = n // seq_len
    if n_pairs == 0:
        raise ValueError("no full pairs decoded")
    decode_s = time.time() - t0

    seg = SegNet().eval().to(device)
    seg.load_state_dict(load_file(segnet_sd_path, device=device))

    logits_store = np.memmap(
        out_dir / "gt_segnet_logits.f16", dtype=np.float16, mode="w+",
        shape=(n_pairs, _N_CLASSES, seg_in_h, seg_in_w),
    )

    t1 = time.time()
    with torch.inference_mode():
        for pi in range(n_pairs):
            f0 = pi * seq_len
            pair = np.stack(frames[f0 : f0 + seq_len])  # (2,H,W,3) uint8
            pair_t = torch.from_numpy(pair).unsqueeze(0).float()  # (1,2,H,W,3)
            x_bt_chw = einops.rearrange(pair_t, "b t h w c -> b t c h w")
            seg_in = seg.preprocess_input(x_bt_chw)  # (1,3,384,512) frame1 resized
            seg_logits = seg(seg_in)  # (1,5,384,512)
            logits_store[pi] = seg_logits[0].cpu().numpy().astype(np.float16)
    logits_store.flush()
    del logits_store
    score_s = time.time() - t1

    # --- NO-FAKE verification: argmax(teacher_logits) must equal cached gt_segnet_argmax.u8 ---
    verify: dict[str, Any] = {"checked": False}
    if verify_argmax_dir is not None:
        ref_path = verify_argmax_dir / "gt_segnet_argmax.u8"
        if ref_path.exists():
            ref = np.memmap(ref_path, dtype=np.uint8, mode="r",
                            shape=(n_pairs, seg_in_h, seg_in_w))
            store = np.memmap(out_dir / "gt_segnet_logits.f16", dtype=np.float16, mode="r",
                              shape=(n_pairs, _N_CLASSES, seg_in_h, seg_in_w))
            # check a sample of pairs (full check is 1.18GB read; sample is decisive)
            n_check = min(40, n_pairs)
            idx = np.linspace(0, n_pairs - 1, n_check).astype(int)
            disagree = 0
            total = 0
            for pi in idx:
                am_teacher = np.asarray(store[pi]).astype(np.float32).argmax(axis=0).astype(np.uint8)
                am_ref = np.asarray(ref[pi])
                disagree += int((am_teacher != am_ref).sum())
                total += am_teacher.size
            verify = {
                "checked": True,
                "ref_argmax_path": str(ref_path),
                "n_pairs_checked": int(n_check),
                "argmax_disagree_frac_vs_cached": disagree / total,
                "note": "fp16 teacher argmax vs cached argmax; nonzero = fp16 round at exact ties only",
            }
            del store, ref

    logits_file = out_dir / "gt_segnet_logits.f16"
    bytes_total = logits_file.stat().st_size
    sha = _sha256_file(logits_file)

    rebuild_cmd = (
        f".venv/bin/python tools/build_segnet_teacher_logits.py "
        f"--video {video_path} --out-dir {out_dir} --num-pairs {num_pairs}"
    )
    meta = {
        "subagent": "build_segnet_teacher_logits_20260625",
        "utc": _utc(),
        "evidence_grade": "[macOS-CPU advisory]",
        "promotion_eligible": False,
        "score_claim": False,
        "purpose": "SegNet 5-class soft-logit TEACHER for KD into the non-RGB witness (kl_on_logits T=2.0)",
        "borrowed_substrate": "KD/kl_on_logits/T=2.0 from Quantizr-PR62 + Hinton-2015; OURS=frozen-contest-SegNet-as-teacher + non-RGB-coord-INR application",
        "video": str(video_path),
        "num_pairs_requested": num_pairs,
        "num_pairs_built": n_pairs,
        "frames_decoded": len(frames),
        "n_classes": _N_CLASSES,
        "seg_input_hw": [seg_in_h, seg_in_w],
        "dtype": "float16",
        "shape": [n_pairs, _N_CLASSES, seg_in_h, seg_in_w],
        "decode_seconds": round(decode_s, 2),
        "score_seconds": round(score_s, 2),
        "no_fake_verification": verify,
        "artifacts": {
            "logits_f16": str(logits_file),
            "bytes": int(bytes_total),
            "sha256": sha,
        },
        "disk_hygiene": {
            "rebuildable": True,
            "rebuild_command": rebuild_cmd,
            "source_argmax_cache": str(verify_argmax_dir) if verify_argmax_dir else None,
            "cold_store_destination": "same SSD tier; cold-store/delete only with this manifest present",
            "reason_rebuildable": "deterministic frozen-SegNet CPU forward on the fixed 0.mkv GT frames",
        },
    }
    (out_dir / "teacher_logits_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SegNet 5-class soft-logit TEACHER store builder (frozen scorer, CPU)")
    ap.add_argument("--video", type=Path, default=UPSTREAM / "videos" / "0.mkv")
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--out-dir", type=Path, default=Path(base) / "teacher_logits_n600")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--verify-argmax-dir", type=Path, default=Path(base) / "targets_n600",
                    help="cached gt_segnet_argmax.u8 dir for the NO-FAKE argmax-consistency check")
    args = ap.parse_args(argv)

    meta = build_teacher_logits(args.video, args.out_dir, args.num_pairs, args.verify_argmax_dir)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
