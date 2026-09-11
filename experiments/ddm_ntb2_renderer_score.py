"""Measure d_seg AND d_pose at n600 for ntb2's renderer precision packets.

Continues `experiments/ddm_ntb2_renderer_seed.py`, which retained 14 single-layer
3-bit starting packets and their exact archive prices but ran no scorer. This
producer supplies the scorer leg the charter assigns to this arm.

The instrument is the arm's own, not a proxy: `jg1.render_frame1` is the receiver's
own forward model at batch 1, `jg1.argmax_from_camera_frames` is the frozen CPU
SegNet through the evaluator's preprocess, and `up2.render_frame0` +
`up2.pose_from_frames` are the shipped frame-0 carrier path and the frozen CPU
PoseNet. Every treatment is measured against a CONTROL run of this same producer on
move 44's own archive, so only the DELTA in each leg is ever quoted: the absolute
local level is not the contest-CUDA level and is never presented as one.

Axis: [macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]. No score claim,
no promotion, no archive is built here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1
import ddm_up2_shipping_pose_solve as up2
import numpy as np

from experiments import ddm_ntb2_control as control

AXIS = "[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]"
STORE = control.ROOT.parent / "renderer_score"
SEEDS = control.ROOT.parent / "renderer_seeds"
TOKEN_FIELD = control.ROOT / "trace_input/field.u8"
RESERVE_BYTES = 40 << 30
BYTE_TO_SCORE = 25.0 / 37_545_489.0
BASE_BYTES = 180_406
STAGE_EVERY = 20


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def retain(path: Path, payload: bytes) -> dict:
    """Persist a payload inside this producer's store, refusing to fill the tier."""
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise ValueError(f"write outside renderer_score store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise RuntimeError("STORAGE_BLOCK: keep all payloads; free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return fact(path)


def archive_for(treatment: str) -> Path:
    """Resolve a treatment name to the archive whose renderer it measures."""
    if treatment == "control":
        return control.ROOT / "source_runtime/archive.zip"
    candidate = SEEDS / treatment / "archive.twin0.zip"
    if not candidate.is_file():
        raise SystemExit(f"no retained packet for treatment {treatment!r}: {candidate}")
    return candidate


def load_instrument(archive: Path, tree: Path, threads: int):
    """Everything the two legs need, bound to the archive under test.

    ``tree`` is the receiver tree root (it holds ``runtime/`` and ``cpr1/``). The two
    helpers take DIFFERENT arguments off it: ``jg1`` wants the ``runtime`` package
    directory because it inserts that path's PARENT, while ``up2`` wants the tree root
    because it inserts that path itself.
    """
    import torch

    runtime_dir = tree / "runtime"
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    semantic = jg1.load_semantic_renderer(archive_path=archive, runtime_dir=runtime_dir)
    net = jg1.load_segnet()
    posenet = up2.load_posenet("cpu")
    # The carrier lives in a section this lever never touches, so frame 0 is the
    # SAME object in every treatment; `verify_archive=False` because up2 pins the
    # OLD pointer body, and the identity that binds here is control.BASE_SHA,
    # recorded in the receipt by the caller.
    state = up2.load_carrier_state(tree, verify_archive=False)
    tokens = jg1.load_tokens(TOKEN_FIELD)
    gt_seg = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    gt_pose, pose_lineage = up2.load_gt_poses(jg1.DEFAULT_GT_DALI)
    if pose_lineage != up2.LINEAGE_DALI:
        raise SystemExit(f"pose GT lineage is {pose_lineage}, not DALI")
    return semantic, net, posenet, state, tokens, gt_seg, gt_pose


def measure(args) -> int:
    import torch

    started = time.time()
    out = STORE / args.treatment
    out.mkdir(parents=True, exist_ok=True)
    tree = control.ROOT / "source_runtime"
    base_sha = sha256_file(tree / "archive.zip")
    if base_sha != control.BASE_SHA:
        raise SystemExit(f"base runtime archive moved: {base_sha}")
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != control.BASE_SHA:
        raise SystemExit("POINTER_MOVED")

    archive = archive_for(args.treatment)
    binding = {
        "schema": "ddm_ntb2_renderer_score.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "treatment": args.treatment,
        "archive_under_test": fact(archive),
        "base_archive_sha256": base_sha,
        "receiver_tree": str(tree),
        "token_field": fact(TOKEN_FIELD),
        "gt_seg_table": fact(jg1.DEFAULT_GT_DALI),
        "producer": fact(Path(__file__)),
        "threads": args.threads,
        "chunk": args.chunk,
        "retention": (
            "per-pair d_seg/d_pose rows, the full SegNet argmax plane, the full pose "
            "vectors, and a per-chunk sha256 of the rendered frame 1 are all kept; the "
            "1.83 GB rendered frame-1 raw is kept only with --retain-frames because the "
            "tier has under 11 GiB usable above its 40 GiB reserve, and it is exactly "
            "rebuildable from the retained archive + token field by this producer"
        ),
    }
    retain(out / "INPUTS.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())

    semantic, net, posenet, state, tokens, gt_seg, gt_pose = load_instrument(
        archive, tree, args.threads
    )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    argmax_plane = np.zeros((jg1.N_PAIRS, jg1.EVAL_H, jg1.EVAL_W), dtype=np.uint8)
    poses = np.zeros((jg1.N_PAIRS, up2.POSE_DIMS), dtype=np.float64)
    chunk_shas: list[dict] = []
    done = 0
    latest = ckpt_dir / "LATEST.json"
    if args.resume and latest.is_file():
        state_json = json.loads(latest.read_text())
        done = int(state_json["pairs_done"])
        rows = state_json["rows"]
        chunk_shas = state_json["chunk_shas"]
        with np.load(ckpt_dir / "LATEST.npz") as blob:
            argmax_plane = blob["argmax"]
            poses = blob["poses"]
        print(json.dumps({"resumed_at_pair": done}), flush=True)

    frames_path = out / "frame1.u8"
    frames_memmap = None
    if args.retain_frames:
        expected = jg1.N_PAIRS * jg1.CAMERA_H * jg1.CAMERA_W * 3
        if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + expected:
            raise RuntimeError("STORAGE_BLOCK: not enough room to retain frame 1")
        frames_memmap = np.memmap(
            frames_path,
            dtype=np.uint8,
            mode="r+" if (args.resume and frames_path.is_file()) else "w+",
            shape=(jg1.N_PAIRS, jg1.CAMERA_H, jg1.CAMERA_W, 3),
        )

    while done < jg1.N_PAIRS:
        chunk = np.arange(done, min(done + args.chunk, jg1.N_PAIRS), dtype=np.int64)
        frames = jg1.render_frame1(semantic, tokens[chunk], chunk)
        chunk_shas.append(
            {"first_pair": int(chunk[0]), "pairs": len(chunk), "sha256": sha256_bytes(frames.tobytes())}
        )
        if frames_memmap is not None:
            frames_memmap[chunk[0] : chunk[-1] + 1] = frames
        argmax = jg1.argmax_from_camera_frames(net, frames)
        argmax_plane[chunk] = argmax
        seg_per_pair = jg1.d_seg_per_pair(argmax, gt_seg[chunk])
        with torch.inference_mode():
            frame0 = up2.render_frame0(state.coefficients[chunk], state, chunk, differentiable=False)
            frame1 = up2.frames_to_bchw(frames)
            pose = up2.pose_from_frames(posenet, frame0, frame1)
        pose_np = pose.to(torch.float64).numpy()
        poses[chunk] = pose_np
        pose_per_pair = ((pose_np - gt_pose[chunk]) ** 2).mean(axis=1)
        for offset, pair in enumerate(chunk):
            rows.append(
                {
                    "pair": int(pair),
                    "d_seg": float(seg_per_pair[offset]),
                    "d_pose": float(pose_per_pair[offset]),
                }
            )
        done = int(chunk[-1]) + 1
        # `done % STAGE_EVERY == 0` would silently never fire for a chunk that does not
        # divide STAGE_EVERY, leaving the whole run un-resumable until its last pair.
        if done % STAGE_EVERY < args.chunk or done == jg1.N_PAIRS:
            stage = {
                "schema": "ddm_ntb2_renderer_score.stage.v1",
                "treatment": args.treatment,
                "pairs_done": done,
                "elapsed_seconds": time.time() - started,
                "running_d_seg": float(np.mean([r["d_seg"] for r in rows])),
                "running_d_pose": float(np.mean([r["d_pose"] for r in rows])),
                "rows": rows,
                "chunk_shas": chunk_shas,
            }
            np.savez(ckpt_dir / "LATEST.npz.tmp.npz", argmax=argmax_plane, poses=poses)
            (ckpt_dir / "LATEST.npz.tmp.npz").replace(ckpt_dir / "LATEST.npz")
            (ckpt_dir / "LATEST.json.tmp").write_text(json.dumps(stage, indent=1, sort_keys=True))
            (ckpt_dir / "LATEST.json.tmp").replace(latest)
            print(
                json.dumps(
                    {
                        "treatment": args.treatment,
                        "done": done,
                        "d_seg": stage["running_d_seg"],
                        "d_pose": stage["running_d_pose"],
                        "seconds": round(stage["elapsed_seconds"], 1),
                    }
                ),
                flush=True,
            )

    frame1_raw = None
    if frames_memmap is not None:
        frames_memmap.flush()
        del frames_memmap
        # Recorded on the RESULT, not on the already-written INPUTS: the raw only
        # exists once the last chunk has been flushed.
        frame1_raw = fact(frames_path)

    d_seg = float(np.mean([r["d_seg"] for r in rows]))
    d_pose = float(np.mean([r["d_pose"] for r in rows]))
    archive_bytes = archive.stat().st_size
    result = {
        "schema": "ddm_ntb2_renderer_score.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "treatment": args.treatment,
        "archive_under_test": fact(archive),
        "frame1_raw": frame1_raw,
        "archive_bytes": archive_bytes,
        "delta_bytes_vs_move44": archive_bytes - BASE_BYTES,
        "n_pairs": jg1.N_PAIRS,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "pose_leg": math.sqrt(10.0 * d_pose),
        "seg_leg": 100.0 * d_seg,
        "rate_leg": BYTE_TO_SCORE * archive_bytes,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "chunk_shas": chunk_shas,
        "note": (
            "ABSOLUTE levels are this instrument's, not the contest-CUDA row's; only "
            "deltas against this producer's own control are interpretable."
        ),
    }
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    np.savez_compressed(out / "argmax_plane.npz.tmp.npz", argmax=argmax_plane, poses=poses)
    (out / "argmax_plane.npz.tmp.npz").replace(out / "argmax_plane.npz")
    summary = {k: v for k, v in result.items() if k not in ("rows", "chunk_shas")}
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True, help="control, or a layer name under renderer_seeds/")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--retain-frames", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--resume-from", type=Path, default=None, help="launcher compatibility; must be the store")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume_from is not None and args.resume_from.resolve() != STORE.resolve():
        raise SystemExit(f"wrong resume root: {args.resume_from}")
    STORE.mkdir(parents=True, exist_ok=True)
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
