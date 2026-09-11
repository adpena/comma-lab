"""ddm_ren1 step-0 probe: the shipped renderer's RENDER FLOOR and its pose sensitivity.

Two questions this producer answers at n600 through the frozen CPU scorer, before
any training byte is spent (md4: *"the free step-0 probe predicts unreachability at
9 in 10 -- use it before burning"*):

1. **The render floor at a CORRECT partition.**  The shipped vehicle renders frame
   ``2p+1`` from a 5-class token plane.  Feed the renderer the GT SegNet argmax
   instead of the shipped field and the partition is perfect by construction; every
   argmax error that survives is manufactured by the render-plus-scorer path.  That
   residual is the floor a weight refit would have to beat, and obx2's decomposition
   is exactly this measurement translated onto a vehicle that SHIPS its partition.

2. **The pose exposure of a render change.**  A refit moves frame 1.  Perturbing the
   shipped frame 1 by a zero-mean field of RMS 0.5 LSB -- half a uint8 step, the
   amplitude at which a refit first changes the shipped bytes -- prices what that
   costs on Pose BEFORE the carrier re-solve.  Two spectra at MATCHED camera-plane
   RMS test obx2's 9.40x smooth-vs-noise law on this object rather than importing it.

Instrument, not a proxy: ``jg1.load_semantic_renderer`` reads the weights out of the
archive under test, ``jg1.render_frame1`` is the receiver's own batch-1 forward model,
``jg1.argmax_from_camera_frames`` is the frozen CPU SegNet through the evaluator's
preprocess, and ``up2.render_frame0`` + ``up2.pose_from_frames`` are the shipped
frame-0 carrier path and the frozen CPU PoseNet.  MPS is never touched here.

Axis: ``[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]``.  Absolute levels
are this instrument's, not the contest-CUDA row's; only DELTAS against the ``control``
treatment measured in the SAME pass are interpretable.  No score claim, no promotion,
no archive is built.
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

AXIS = "[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]"

#: Move 48 -- the pointer this arm is bound to.  Copy-only; never written.
POINTER_TREE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime"
)
POINTER_ARCHIVE_SHA256 = (
    "d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c"
)
POINTER_ARCHIVE_BYTES = 179_111
#: The shipped token field (sha ``a92e7d90...``), unchanged since move 43.
TOKEN_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
TOKEN_FIELD_SHA256 = (
    "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
)

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ren1/step0_probe")
RESERVE_BYTES = 40 << 30
STAGE_EVERY = 20

#: Perturbation amplitude, in uint8 least-significant bits, as camera-plane RMS.
PERTURB_RMS_LSB = 0.5
#: Low-frequency grid the SMOOTH field is drawn on before bicubic upsampling to the
#: camera plane.  24x32 is the carrier basis's own resolution, so "smooth" here means
#: the same band the shipped vehicle already spends its pose bytes in.
SMOOTH_GRID_H, SMOOTH_GRID_W = 24, 32

TREATMENTS = ("control", "gt_partition", "smooth_p05", "noise_p05")


class Ren1ProbeError(RuntimeError):
    """Refusal raised by this producer.  Never downgraded to a warning."""


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
        raise Ren1ProbeError(f"write outside the ren1 probe store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise Ren1ProbeError("STORAGE_BLOCK: keep all payloads; free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return fact(path)


def _bind_pointer() -> dict:
    """Fail closed unless the archive and the token field are the bytes claimed."""
    archive = POINTER_TREE / "archive.zip"
    observed = sha256_file(archive)
    if observed != POINTER_ARCHIVE_SHA256:
        raise Ren1ProbeError(f"pointer archive drifted: {observed}")
    if archive.stat().st_size != POINTER_ARCHIVE_BYTES:
        raise Ren1ProbeError("pointer archive size drifted")
    field = sha256_file(TOKEN_FIELD)
    if field != TOKEN_FIELD_SHA256:
        raise Ren1ProbeError(f"token field drifted: {field}")
    return {"archive": fact(archive), "token_field": fact(TOKEN_FIELD)}


def perturbation_fields(seed: int):
    """The two RMS-matched camera-plane fields, built once and reused for every pair.

    Both are zero-mean and are scaled so their camera-plane RMS is EXACTLY
    ``PERTURB_RMS_LSB``; the only thing that differs between them is the spectrum.
    That is what makes the smooth/noise ratio a measurement of the spectrum rather
    than of the magnitude.  A single shared field (not one per pair) keeps the two
    arms comparable pair-by-pair and keeps the probe deterministic.
    """
    import torch
    from torch.nn import functional

    generator = torch.Generator().manual_seed(seed)
    low = torch.randn(
        (1, 3, SMOOTH_GRID_H, SMOOTH_GRID_W), generator=generator, dtype=torch.float32
    )
    smooth = functional.interpolate(
        low, size=(jg1.CAMERA_H, jg1.CAMERA_W), mode="bicubic", align_corners=False
    )[0]
    smooth = smooth - smooth.mean()
    smooth = smooth * (PERTURB_RMS_LSB / smooth.square().mean().sqrt().clamp_min(1e-12))

    noise = torch.randn(
        (3, jg1.CAMERA_H, jg1.CAMERA_W), generator=generator, dtype=torch.float32
    )
    noise = noise - noise.mean()
    noise = noise * (PERTURB_RMS_LSB / noise.square().mean().sqrt().clamp_min(1e-12))
    # (H, W, 3) to match the uint8 camera frames the receiver hands the scorer.
    return smooth.permute(1, 2, 0).contiguous(), noise.permute(1, 2, 0).contiguous()


def apply_perturbation(frames_bhwc: np.ndarray, field) -> np.ndarray:
    """Add a float field to uint8 camera frames and re-quantise, as a render would.

    The receiver's own last step is ``clamp(0,255).round()`` into uint8
    (``cpr1/inflate.py:322-325``).  A refit that moved the pre-round float by this
    much would land exactly here, so the perturbed arm is a real alternative decode
    and not a float fiction the receiver could never emit.
    """
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(frames_bhwc)).float()
    moved = (tensor + field[None]).clamp(0.0, 255.0).round()
    return moved.to(torch.uint8).numpy()


def changed_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """Share of camera samples the treatment actually moved.

    A perturbation of half an LSB that rounded back to the SAME uint8 everywhere
    would be a no-op wearing a treatment's name, and its null Pose reading would be
    a fact about the rounding rather than about the spectrum.  This number is
    recorded per chunk so the reader can refuse the arm rather than trust it.
    """
    return float(np.count_nonzero(before != after)) / float(before.size)


def load_instrument(threads: int):
    """Everything both legs need, bound to the move-48 archive under test."""
    import torch

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    archive = POINTER_TREE / "archive.zip"
    semantic = jg1.load_semantic_renderer(
        archive_path=archive, runtime_dir=POINTER_TREE / "runtime"
    )
    net = jg1.load_segnet()
    posenet = up2.load_posenet("cpu")
    # ``verify_archive=False`` because up2 pins an OLDER pointer body by constant;
    # the identity that binds this run is POINTER_ARCHIVE_SHA256, checked above and
    # recorded in INPUTS.json.
    state = up2.load_carrier_state(POINTER_TREE, verify_archive=False)
    tokens = jg1.load_tokens(TOKEN_FIELD)
    gt_seg = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    gt_pose, pose_lineage = up2.load_gt_poses(jg1.DEFAULT_GT_DALI)
    if pose_lineage != up2.LINEAGE_DALI:
        raise Ren1ProbeError(f"pose GT lineage is {pose_lineage}, not DALI")
    # The GT argmax is used as a TOKEN PLANE by the gt_partition arm, so it has to
    # live inside the renderer's embedding domain; a stray label would index past
    # ``nn.Embedding(NUM_CLASSES, width)`` instead of failing.
    if int(gt_seg.max()) >= jg1.NUM_CLASSES:
        raise Ren1ProbeError(
            f"GT label {int(gt_seg.max())} is outside the embedding domain "
            f"(<{jg1.NUM_CLASSES}); it cannot be rendered as a token plane"
        )
    return semantic, net, posenet, state, tokens, gt_seg, gt_pose


def measure(args) -> int:
    import torch

    started = time.time()
    treatments = tuple(args.treatments)
    unknown = set(treatments) - set(TREATMENTS)
    if unknown:
        raise Ren1ProbeError(f"unknown treatments: {sorted(unknown)}")
    if "control" not in treatments:
        raise Ren1ProbeError("every pass must carry its own control; deltas need it")

    out = STORE / args.label
    out.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema": "ddm_ren1_step0_probe.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "label": args.label,
        "treatments": list(treatments),
        "pointer": _bind_pointer(),
        "pointer_move": 48,
        "gt_seg_table": fact(jg1.DEFAULT_GT_DALI),
        "producer": fact(Path(__file__)),
        "jg1": fact(Path(jg1.__file__)),
        "up2": fact(Path(up2.__file__)),
        "threads": args.threads,
        "chunk": args.chunk,
        "seed": args.seed,
        "perturb_rms_lsb": PERTURB_RMS_LSB,
        "retention": (
            "per-pair d_seg/d_pose rows for every treatment, the full SegNet argmax "
            "plane and full pose vectors per treatment, and a per-chunk sha256 of each "
            "treatment's camera frames; the ~1.8 GB frame rasters are NOT retained "
            "because they are exactly rebuildable from the pinned archive + token field "
            "by this producer and the tier holds a 40 GiB reserve"
        ),
    }
    retain(out / "INPUTS.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())

    semantic, net, posenet, state, tokens, gt_seg, gt_pose = load_instrument(args.threads)
    smooth_field, noise_field = perturbation_fields(args.seed)
    retain(
        out / "PERTURBATION.json",
        (
            json.dumps(
                {
                    "seed": args.seed,
                    "rms_lsb_target": PERTURB_RMS_LSB,
                    "smooth_rms": float(smooth_field.square().mean().sqrt()),
                    "noise_rms": float(noise_field.square().mean().sqrt()),
                    "smooth_grid": [SMOOTH_GRID_H, SMOOTH_GRID_W],
                    "smooth_sha256": sha256_bytes(smooth_field.numpy().tobytes()),
                    "noise_sha256": sha256_bytes(noise_field.numpy().tobytes()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    rows: dict[str, list[dict]] = {name: [] for name in treatments}
    argmax_planes = {
        name: np.zeros((jg1.N_PAIRS, jg1.EVAL_H, jg1.EVAL_W), dtype=np.uint8)
        for name in treatments
    }
    poses = {
        name: np.zeros((jg1.N_PAIRS, up2.POSE_DIMS), dtype=np.float64)
        for name in treatments
    }
    chunk_shas: list[dict] = []
    done = 0
    latest = ckpt_dir / "LATEST.json"
    if args.resume and latest.is_file():
        stage = json.loads(latest.read_text())
        if stage["treatments"] != list(treatments):
            raise Ren1ProbeError("checkpoint treatments differ from this invocation")
        done = int(stage["pairs_done"])
        rows = {name: stage["rows"][name] for name in treatments}
        chunk_shas = stage["chunk_shas"]
        with np.load(ckpt_dir / "LATEST.npz") as blob:
            for name in treatments:
                argmax_planes[name] = blob[f"argmax_{name}"]
                poses[name] = blob[f"poses_{name}"]
        print(json.dumps({"resumed_at_pair": done}), flush=True)

    while done < jg1.N_PAIRS:
        chunk = np.arange(done, min(done + args.chunk, jg1.N_PAIRS), dtype=np.int64)
        base = jg1.render_frame1(semantic, tokens[chunk], chunk)
        frames_by_treatment: dict[str, np.ndarray] = {}
        for name in treatments:
            if name == "control":
                frames_by_treatment[name] = base
            elif name == "gt_partition":
                frames_by_treatment[name] = jg1.render_frame1(
                    semantic, gt_seg[chunk], chunk
                )
            elif name == "smooth_p05":
                frames_by_treatment[name] = apply_perturbation(base, smooth_field)
            elif name == "noise_p05":
                frames_by_treatment[name] = apply_perturbation(base, noise_field)
            else:  # pragma: no cover - guarded above
                raise Ren1ProbeError(f"unhandled treatment {name}")

        with torch.inference_mode():
            frame0 = up2.render_frame0(
                state.coefficients[chunk], state, chunk, differentiable=False
            )
        record = {"first_pair": int(chunk[0]), "pairs": len(chunk)}
        for name in treatments:
            frames = frames_by_treatment[name]
            record[f"sha256_{name}"] = sha256_bytes(frames.tobytes())
            if name != "control":
                record[f"changed_fraction_{name}"] = changed_fraction(base, frames)
            argmax = jg1.argmax_from_camera_frames(net, frames)
            argmax_planes[name][chunk] = argmax
            seg_per_pair = jg1.d_seg_per_pair(argmax, gt_seg[chunk])
            with torch.inference_mode():
                pose = up2.pose_from_frames(posenet, frame0, up2.frames_to_bchw(frames))
            pose_np = pose.to(torch.float64).numpy()
            poses[name][chunk] = pose_np
            pose_per_pair = ((pose_np - gt_pose[chunk]) ** 2).mean(axis=1)
            for offset, pair in enumerate(chunk):
                rows[name].append(
                    {
                        "pair": int(pair),
                        "d_seg": float(seg_per_pair[offset]),
                        "d_pose": float(pose_per_pair[offset]),
                    }
                )
        chunk_shas.append(record)
        done = int(chunk[-1]) + 1

        # ``done % STAGE_EVERY == 0`` would silently never fire for a chunk size that
        # does not divide STAGE_EVERY, leaving the run un-resumable until its last pair.
        if done % STAGE_EVERY < args.chunk or done == jg1.N_PAIRS:
            stage = {
                "schema": "ddm_ren1_step0_probe.stage.v1",
                "label": args.label,
                "treatments": list(treatments),
                "pairs_done": done,
                "elapsed_seconds": time.time() - started,
                "running": {
                    name: {
                        "d_seg": float(np.mean([r["d_seg"] for r in rows[name]])),
                        "d_pose": float(np.mean([r["d_pose"] for r in rows[name]])),
                    }
                    for name in treatments
                },
                "rows": rows,
                "chunk_shas": chunk_shas,
            }
            payload = {}
            for name in treatments:
                payload[f"argmax_{name}"] = argmax_planes[name]
                payload[f"poses_{name}"] = poses[name]
            np.savez(ckpt_dir / "LATEST.npz.tmp.npz", **payload)
            (ckpt_dir / "LATEST.npz.tmp.npz").replace(ckpt_dir / "LATEST.npz")
            (ckpt_dir / "LATEST.json.tmp").write_text(json.dumps(stage, indent=1, sort_keys=True))
            (ckpt_dir / "LATEST.json.tmp").replace(latest)
            print(
                json.dumps(
                    {
                        "label": args.label,
                        "done": done,
                        "running": stage["running"],
                        "seconds": round(stage["elapsed_seconds"], 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {}
    for name in treatments:
        d_seg = float(np.mean([r["d_seg"] for r in rows[name]]))
        d_pose = float(np.mean([r["d_pose"] for r in rows[name]]))
        summary[name] = {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "seg_leg": 100.0 * d_seg,
            "pose_leg": math.sqrt(10.0 * d_pose),
        }
    control = summary["control"]
    deltas = {
        name: {
            "d_seg_ratio_vs_control": summary[name]["d_seg"] / control["d_seg"],
            "d_pose_ratio_vs_control": summary[name]["d_pose"] / control["d_pose"],
            "delta_seg_leg": summary[name]["seg_leg"] - control["seg_leg"],
            "delta_pose_leg": summary[name]["pose_leg"] - control["pose_leg"],
        }
        for name in treatments
        if name != "control"
    }
    result = {
        "schema": "ddm_ren1_step0_probe.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "label": args.label,
        "n_pairs": jg1.N_PAIRS,
        "treatments": list(treatments),
        "summary": summary,
        "deltas_vs_control": deltas,
        "elapsed_seconds": time.time() - started,
        "binding": binding,
        "rows": rows,
        "chunk_shas": chunk_shas,
        "note": (
            "ABSOLUTE levels are this instrument's, not the contest-CUDA row's; only "
            "deltas against this pass's own control are interpretable."
        ),
    }
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    payload = {}
    for name in treatments:
        payload[f"argmax_{name}"] = argmax_planes[name]
        payload[f"poses_{name}"] = poses[name]
    np.savez_compressed(out / "planes.npz.tmp.npz", **payload)
    (out / "planes.npz.tmp.npz").replace(out / "planes.npz")
    printed = {k: v for k, v in result.items() if k not in ("rows", "chunk_shas", "binding")}
    print(json.dumps(printed, indent=1, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="subdirectory under the probe store")
    parser.add_argument(
        "--treatments",
        nargs="+",
        required=True,
        help=f"any of {TREATMENTS}; 'control' is mandatory in every pass",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="launcher compatibility; must be the store"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume_from is not None and args.resume_from.resolve() != STORE.resolve():
        raise SystemExit(f"wrong resume root: {args.resume_from}")
    STORE.mkdir(parents=True, exist_ok=True)
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
