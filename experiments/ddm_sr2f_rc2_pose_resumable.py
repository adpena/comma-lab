"""ddm_sr2f — resumable chunked driver for the sr1 FO-1 A1 pose leg on the rc2 body.

WHY THIS EXISTS.  `experiments/ddm_sr1_manufactured_seg_recovery.py --stage a1pose` saves its
`poses` array only after the full 600-pair loop finishes (loop-end-only saving), which CLAUDE.md
forbids for exactly this reason: an interrupted run loses ALL work.  On this machine the stage is
killed by an external SIGTERM at pair 140 / ~341 s, DETERMINISTICALLY (attempts 3 and 4, 340 s and
341 s), with the governor exonerated (`safe_run` peak RSS 1,417 MiB of 16,000, `kill_action: null`,
elapsed 357 s of a 2700 s timeout).  This driver does not diagnose that killer; it makes the
measurement survive it by checkpointing every chunk and resuming from disk.

INSTRUMENT IDENTITY.  Every numerical primitive is IMPORTED from the sealed tool, never re-typed:
`_a1_axis_operators`, `_a1_custody_control`, `_a1_delta_camera`, `_PoseInstrument`, `open_raw`,
`A1_ALPHAS`.  The per-pair computation is byte-for-byte the same object `stage_a1pose` computes:
`cam'(a) = round(clamp(cam + a * delta_cam))`, pose6 = frozen CPU PoseNet first 6 dims, batch = 1
pair, `torch.set_num_threads(8)`.  The alpha = 0 bit-identity assertion is preserved.

The ONLY deliberate difference from the sealed stage is the checkpointing, plus an optional seeded
RANDOM pair subset (never a prefix -- [[m88]]/[[m96]]: a prefix of a skewed population is a
different population, and pose prefixes measure 2.54-4.21x HARDER).  Default is the full n600.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEALED = REPO / "experiments" / "ddm_sr1_manufactured_seg_recovery.py"
PIN_THREADS = 8


def _load_sealed():
    spec = importlib.util.spec_from_file_location("ddm_sr1_sealed", SEALED)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load sealed tool at {SEALED}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ddm_sr1_sealed"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="checkpoint + receipt prefix")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--threads", type=int, default=PIN_THREADS)
    ap.add_argument("--chunk", type=int, default=40, help="pairs per checkpoint")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = run until done")
    ap.add_argument("--subset-n", type=int, default=0,
                    help="0 = all frames; else a SEEDED RANDOM subset of this size (never a prefix)")
    ap.add_argument("--subset-seed", type=int, default=20260821)
    ap.add_argument("--tikhonov", type=float, default=1e-6)
    args = ap.parse_args(argv)

    if args.threads != PIN_THREADS:
        raise SystemExit(f"instrument pin violation: requires --threads {PIN_THREADS}")

    m = _load_sealed()
    alphas = list(m.A1_ALPHAS)

    if args.subset_n:
        rng = np.random.default_rng(args.subset_seed)
        pair_ids = np.sort(rng.choice(args.frames, size=args.subset_n, replace=False))
    else:
        pair_ids = np.arange(args.frames)
    n = len(pair_ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    poses_path = args.out.with_suffix(".poses.npy")
    state_path = args.out.with_suffix(".state.json")

    if poses_path.exists() and state_path.exists():
        poses = np.load(poses_path)
        state = json.loads(state_path.read_text())
        done = int(state["done"])
        if poses.shape != (n, len(alphas), 6) or state.get("pair_ids_sha") != _ids_sha(pair_ids):
            raise SystemExit("checkpoint does not match this configuration; refusing to mix runs")
        print(f"RESUME: {done}/{n} pairs already done", flush=True)
    else:
        poses = np.zeros((n, len(alphas), 6), dtype=np.float64)
        done = 0
        state = {"done": 0, "pair_ids_sha": _ids_sha(pair_ids)}

    if done >= n:
        print("already complete", flush=True)
        _write_receipt(args, m, poses, pair_ids, alphas, state)
        return 0

    ops = m._a1_axis_operators(args.tikhonov)
    custody = m._a1_custody_control(ops, args.work)
    raw = m.open_raw(args.raw)
    inst = m._PoseInstrument(args.threads)

    t0 = time.time()
    chunks = 0
    while done < n:
        stop = min(done + args.chunk, n)
        for k in range(done, stop):
            t = int(pair_ids[k])
            cam_u8 = np.asarray(raw[2 * t + 1])
            frame0 = np.asarray(raw[2 * t])
            cam_f = cam_u8.astype(np.float64)
            _, delta_cam = m._a1_delta_camera(cam_f, ops)
            for i, a in enumerate(alphas):
                cam_prime = np.clip(cam_f + a * delta_cam, 0.0, 255.0).round().astype(np.uint8)
                if a == 0.0 and not np.array_equal(cam_prime, cam_u8):
                    raise SystemExit(f"pair {t}: alpha=0 is not bit-identical to the shipped frame")
                poses[k, i] = inst.pose6(frame0, cam_prime)
        done = stop
        tmp = poses_path.with_suffix(".tmp.npy")
        np.save(tmp, poses)
        tmp.replace(poses_path)
        state["done"] = done
        state_tmp = state_path.with_suffix(".tmp.json")
        state_tmp.write_text(json.dumps(state))
        state_tmp.replace(state_path)
        el = time.time() - t0
        print(f"  chunk saved: {done}/{n} pairs  {el:.0f}s", flush=True)
        chunks += 1
        if args.max_chunks and chunks >= args.max_chunks:
            print("max-chunks reached; rerun to continue", flush=True)
            return 2

    _write_receipt(args, m, poses, pair_ids, alphas, state, custody=custody)
    return 0


def _ids_sha(ids: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(ids.astype(np.int64)).tobytes()).hexdigest()


def _write_receipt(args, m, poses, pair_ids, alphas, state, custody=None) -> None:
    base = poses[:, 0, :]
    rows = []
    for i, a in enumerate(alphas):
        drift = poses[:, i, :] - base
        rows.append(
            {
                "alpha": float(a),
                "pose_drift_rms_vs_alpha0": float(np.sqrt(np.mean(drift**2))),
                "pose_drift_max_abs": float(np.abs(drift).max()),
            }
        )
    rec = {
        "schema": "ddm_sr2f_rc2_pose_resumable.v1",
        "arm": "ddm_sr2f",
        "axis": "[macOS-CPU advisory] -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "n_pairs": len(pair_ids),
        "subset_n": int(args.subset_n),
        "subset_seed": int(args.subset_seed),
        "pair_selection": "ALL n600" if not args.subset_n else "SEEDED RANDOM (never a prefix)",
        "raw": str(args.raw),
        "instrument": {
            "scorer": "frozen CPU torch PoseNet, upstream/models/posenet.safetensors",
            "batch_pairs": 1,
            "torch_threads": int(args.threads),
            "preprocess": "upstream PoseNet.preprocess_input verbatim, first 6 pose dims",
            "primitives_imported_from": str(SEALED),
        },
        "custody": custody,
        "ladder": rows,
    }
    out = args.out.with_suffix(".receipt.json")
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rows, indent=2), flush=True)
    print(f"receipt -> {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
