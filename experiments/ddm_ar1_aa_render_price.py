#!/usr/bin/env python3
"""ddm_ar1 -- the $0 price of the footprint (AA) render on the BORN field.

WHAT THIS MEASURES
------------------
The qbt1 born trainer renders POINT-sampled at exactly ``EVAL_H, EVAL_W = 384, 512``
(``ddm_qbt1_qbflow_trainer.QBFLOWTorch.forward`` -> ``rgb = sigmoid(_linear(render_state, ...))``),
then bicubic-upsamples to the camera grid and rounds to uint8.  The AA-SDF observation render
(``tac.boundary_math.aa_sdf_observation_render``) replaces that point sample with the pixel
FOOTPRINT INTEGRAL: render at ``(ss*H, ss*W)`` and box-downsample by ``ss``.

The footprint integral is a DECODE-TIME operation on an UNCHANGED archive, so it is a ~0-rate
lever.  This instrument prices it on the trained born field by re-scoring the SAME sealed
checkpoint weights through the SAME roundtrip and the SAME frozen CPU-torch scorers, changing
ONLY the render sampling.

WHY THE FINE LATTICE IS THE MODULE'S OWN LATTICE (verified at source)
--------------------------------------------------------------------
``aa_sdf_observation_render.build_supersampled_coords(h, w, ss)`` is defined as
``build_render_coords(h*ss, w*ss)`` = ``np.linspace(-1, 1, h*ss)`` x ``np.linspace(-1, 1, w*ss)``
(endpoint-inclusive).  ``ddm_qbt1_qbflow_trainer._base_features(height=ss*H, width=ss*W)`` builds
``torch.linspace(-1.0, 1.0, height)`` / ``torch.linspace(-1.0, 1.0, width)`` with
``meshgrid(..., indexing="ij")``.  These are the SAME lattice, so calling the trainer's own
``forward`` at ``(ss*H, ss*W)`` and box-downsampling by ``ss`` reproduces the module's
supersample->box contract.

TWO HONEST BOUNDS ON THAT CLAIM, both MEASURED, neither load-bearing for the delta:

1. The two lattices are equal to within 0.5 ULP of float32, NOT bit-for-bit: ``torch.linspace``
   and ``np.linspace`` round the last bit differently on up to ~49% of entries.  Measured maximum
   absolute gap 5.960464e-08 at every grid used here (n = 384, 512, 768, 1024, 1152, 1536), i.e.
   1.52e-05 of one base-grid pixel pitch (2/511).  ``render_rgb_pair`` builds BOTH the ss=1 and
   the ss>1 grids through the trainer's own ``forward``, so the comparison never straddles the
   two implementations and this gap cancels out of the delta entirely.
2. The endpoint-inclusive fine grid is NOT the exact sub-cell-centre quadrature, and the gap is
   large enough to matter.  MEASURED: the ss-block means of ``linspace(-1, 1, n*ss)`` drift from
   ``linspace(-1, 1, n)`` by up to 0.2497 coarse pixels at ss=2 (0.3328 at ss=3, 0.3743 at ss=4),
   inward at both frame edges and exactly 0 at the centre -- the module's AA image is a slightly
   CONTRACTED copy of the field.  So a delta measured on this lattice mixes FOOTPRINT AVERAGING
   with a sub-pixel REGISTRATION shift.  ``--lattice footprint_centred`` supplies the corrected
   grid (see :func:`footprint_centred_span`) so the two components can be separated.  The published
   law ``aa_sdf_observation_footprint_render_dseg_v1`` was measured on the module lattice, so
   ``--lattice module_endpoint`` (the default) is the operator the law names.

``ss == 1`` is the trainer's current point-sampled path bit-for-bit -- the box downsample is
skipped outright, so the ss=1 tensor is the object ``forward`` returned.

AUTHORITY
---------
* GT authority is DALI (``gt_cache_dali.pt``); ``gt_n600.npz`` is PyAV lineage and is reported
  ONLY as the burn's own continuity frame.  Both reads go through ``assert_gt_lineage``.
* Scorers are the frozen upstream CPU-torch SegNet/PoseNet.  CPU only -- MPS is never an
  authority.  The burn's own milestone numbers are ``[macOS-MPS n32 stratified advisory]``, so a
  CPU-vs-MPS gap in the calibration gate is an AXIS difference, not an arithmetic failure; the
  calibration decomposition below separates the two.
* This instrument makes NO score claim: it is ``[macOS-CPU advisory]`` and non-promotable.

CALIBRATION DECOMPOSITION (three legs, so a gate miss names its own cause)
-------------------------------------------------------------------------
1. ARITHMETIC -- recompute d_seg/d_pose from the burn's OWN retained per-pair arrays.  Must
   reproduce the recorded milestone exactly; this proves the formula matches.
2. SCORER AXIS -- re-score the burn's OWN retained ``camera_pair_u8`` bytes on CPU scorers.  The
   input bytes are identical, so any gap is pure CPU-vs-MPS scorer drift.
3. RENDER AXIS -- render ss=1 on CPU and score on CPU.  The gap against leg 2 is the CPU-vs-MPS
   drift of the render itself.

ALWAYS KEEP THE PAYLOAD: every rendered argmax is persisted; camera uint8 is persisted for the
first ``--retain-camera-pairs`` pairs of every mode; every per-pair row is appended to a JSONL as
it is produced, so a crash loses nothing and the run resumes from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qbt1_qbflow_trainer as qbt
from tac.boundary_math.aa_sdf_observation_render import box_downsample_np
from tac.gt_lineage import DALI_NVDEC, PYAV_YUV420_TO_RGB, assert_gt_lineage

INSTRUMENT = "ddm_ar1_aa_render_price"
SCHEMA = "ddm_ar1_aa_render_price.v1"
AXIS = "[macOS-CPU advisory; frozen scorer; not contest authority]"

DALI_GT = Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
PYAV_GT = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# Canonical comma10k class order (CLAUDE.md, MEASURED 2026-06-27; never luma-sorted).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

SEEDED_N32_SEED = 20260903


class AR1Error(RuntimeError):
    """Fail-closed error for the ar1 instrument."""


# ---------------------------------------------------------------------------
# custody helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance is best-effort, never load-bearing
        return "unknown"


def peak_rss_gib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(usage) * scale / float(1 << 30)


# ---------------------------------------------------------------------------
# model + ground truth
# ---------------------------------------------------------------------------
def load_ema_model(checkpoint: Path) -> tuple[qbt.QBFLOWTorch, dict[str, Any]]:
    """Load the checkpoint's EMA shadow into the trainer's own twin.

    Mirrors ``ddm_qbt1_qbflow_trainer.load_r1_ema_model`` exactly (build the immutable QBF1 twin
    via ``load_initial_model``, then ``load_state_dict`` the EMA shadow).  The burn's evaluation
    path enters ``qbt.ema_scope``, which applies the same shadow, so these are the weights the
    burn's own milestone scored.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    shadow = payload.get("ema", {}).get("shadow")
    if not shadow:
        raise AR1Error(f"checkpoint {checkpoint} carries no EMA shadow")
    model = qbt.load_initial_model(torch.device("cpu"))
    if set(shadow) != set(model.state_dict()):
        raise AR1Error("checkpoint EMA tensor set differs from the immutable QBF1 twin")
    model.load_state_dict(
        {name: value.detach().clone() for name, value in shadow.items()}, strict=True
    )
    model.eval()
    meta = {
        "schema": payload.get("schema"),
        "stage": payload.get("stage"),
        "completed_steps": payload.get("completed_steps", payload.get("step")),
        "config_identity_sha256": payload.get("config_identity_sha256"),
        "ema_decay": payload.get("ema", {}).get("decay"),
        "ema_num_updates": payload.get("ema", {}).get("num_updates"),
    }
    return model, meta


def load_ground_truth() -> dict[str, Any]:
    """Load both GT frames with per-read lineage asserts.

    DALI is the authority.  PyAV is loaded ONLY so the burn's own continuity frame can be
    reported alongside; the two are never mixed inside one number.
    """
    dali_entry = assert_gt_lineage(DALI_GT, required=DALI_NVDEC, instrument=INSTRUMENT)
    pyav_entry = assert_gt_lineage(PYAV_GT, required=PYAV_YUV420_TO_RGB, instrument=INSTRUMENT)
    dali = torch.load(DALI_GT, map_location="cpu", weights_only=False)
    dali_seg = np.asarray(dali["seg"].numpy(), dtype=np.uint8)
    dali_pose = np.asarray(dali["pose"].numpy(), dtype=np.float32)
    with np.load(PYAV_GT, allow_pickle=False) as payload:
        pyav_seg = np.asarray(payload["lstars"], dtype=np.uint8)
        pyav_pose = np.asarray(payload["gt_poses"], dtype=np.float32)
    for name, seg in (("dali", dali_seg), ("pyav", pyav_seg)):
        if seg.shape != (qbt.N, qbt.EVAL_H, qbt.EVAL_W):
            raise AR1Error(f"{name} seg GT geometry differs: {seg.shape}")
    for name, pose in (("dali", dali_pose), ("pyav", pyav_pose)):
        if pose.shape != (qbt.N, 6):
            raise AR1Error(f"{name} pose GT geometry differs: {pose.shape}")
    off_sites = int((dali_seg != pyav_seg).sum())
    return {
        "dali_seg": dali_seg,
        "dali_pose": dali_pose,
        "pyav_seg": pyav_seg,
        "pyav_pose": pyav_pose,
        "lineage": {
            "authority": {"lineage": dali_entry.lineage, "sha256": dali_entry.sha256,
                          "file": file_fact(DALI_GT)},
            "burn_continuity": {"lineage": pyav_entry.lineage, "sha256": pyav_entry.sha256,
                                "file": file_fact(PYAV_GT)},
            "argmax_sites_dali_vs_pyav": off_sites,
        },
    }


# ---------------------------------------------------------------------------
# the render under test
# ---------------------------------------------------------------------------
LATTICE_MODULE = "module_endpoint"
LATTICE_CENTRED = "footprint_centred"


def footprint_centred_span(n: int, ss: int) -> tuple[float, float]:
    """``(lo, hi)`` for the fine grid whose ``ss``-blocks are CENTRED on ``linspace(-1, 1, n)``.

    MEASURED DEFECT this repairs: ``build_supersampled_coords`` uses ``linspace(-1, 1, n*ss)``,
    which shares the coarse grid's endpoints.  Its ``ss``-block means are therefore NOT the coarse
    sample points -- they drift inward, by 0.2497 coarse pixels at ss=2, 0.3328 at ss=3 and 0.3743
    at ss=4 at the frame edges (exactly 0 at the centre), so the AA image is a slightly CONTRACTED
    copy of the field.  Part of any AA-vs-point delta measured on that lattice is registration, not
    footprint averaging.

    The repair: with coarse pitch ``p = 2/(n-1)``, the sub-cell centres of coarse sample ``i`` are
    ``-1 + i*p + p*(k + 0.5 - ss/2)/ss`` for ``k in [0, ss)``.  Flattened that is a uniform grid of
    ``n*ss`` points with pitch ``p/ss`` running from ``-1 - p/2 + p/(2*ss)`` to ``1 + p/2 -
    p/(2*ss)``, whose ``ss``-block means are the coarse samples EXACTLY.
    """
    if n < 2:
        raise AR1Error(f"footprint span needs at least 2 coarse samples; got {n}")
    ss = int(ss)
    if ss < 1:
        raise AR1Error(f"supersample factor must be >= 1; got {ss}")
    pitch = 2.0 / (n - 1)
    inset = pitch / 2.0 - pitch / (2.0 * ss)
    return -1.0 - inset, 1.0 + inset


def _centred_linspace_shim(height: int, width: int, ss: int):
    """A ``torch.linspace`` stand-in that re-spans ONLY the two render-grid axis calls.

    ``qbt._base_features`` is the trainer's own feature builder and is used unmodified; only the
    two ``torch.linspace(-1.0, 1.0, n)`` calls that lay down the sample grid are re-spanned, so the
    field is evaluated at footprint-centred coordinates and every downstream feature follows.  Any
    other ``linspace`` call passes straight through.
    """
    real_linspace = torch.linspace
    targets = {height * ss: footprint_centred_span(height, ss),
               width * ss: footprint_centred_span(width, ss)}

    def shim(start, end, steps, **kwargs):
        try:
            matched = float(start) == -1.0 and float(end) == 1.0 and int(steps) in targets
        except (TypeError, ValueError):
            # Anything whose endpoints are not plain scalars is not the render grid; pass it on
            # rather than risking a crash mid-run.
            matched = False
        if matched:
            lo, hi = targets[int(steps)]
            return real_linspace(lo, hi, int(steps), **kwargs)
        return real_linspace(start, end, steps, **kwargs)

    return shim


def render_rgb_pair(
    model: qbt.QBFLOWTorch, pair_id: int, ss: int, lattice: str = LATTICE_MODULE
) -> torch.Tensor:
    """Render one pair at supersample factor ``ss`` and footprint-integrate back to the base grid.

    Returns ``[1, 2, 3, EVAL_H, EVAL_W]`` float32 in [0, 1] -- the exact shape/scale the trainer's
    ``roundtrip_to_camera_uint8_ste`` consumes.  ``ss == 1`` is the trainer's current path
    unchanged (the box downsample is skipped entirely).

    ``lattice`` selects the fine sample grid: ``module_endpoint`` reproduces
    ``aa_sdf_observation_render.build_supersampled_coords`` (the operator the published law was
    measured with); ``footprint_centred`` uses the registration-corrected grid above.  Comparing
    the two separates the AA delta's blur component from its registration component.
    """
    if ss < 1:
        raise AR1Error(f"supersample factor must be >= 1; got {ss}")
    if lattice not in (LATTICE_MODULE, LATTICE_CENTRED):
        raise AR1Error(f"unknown lattice {lattice!r}")
    ids = torch.tensor([int(pair_id)], dtype=torch.long)
    height, width = qbt.EVAL_H * ss, qbt.EVAL_W * ss
    with torch.no_grad():
        if lattice == LATTICE_CENTRED and ss > 1:
            saved = qbt.torch.linspace
            qbt.torch.linspace = _centred_linspace_shim(qbt.EVAL_H, qbt.EVAL_W, ss)
            try:
                outputs = model(ids, height=height, width=width)
            finally:
                qbt.torch.linspace = saved
        else:
            outputs = model(ids, height=height, width=width)
        rgb = outputs["rgb_pair_01"]
    if ss == 1:
        return rgb
    # box_downsample_np is the numpy AUTHORITY for the footprint integral (NHWC ss x ss block
    # mean).  Route through it rather than re-implementing the block mean in torch.
    fine = rgb.permute(0, 1, 3, 4, 2).reshape(2, qbt.EVAL_H * ss, qbt.EVAL_W * ss, 3)
    coarse = box_downsample_np(fine.contiguous().numpy(), ss)
    if coarse.shape != (2, qbt.EVAL_H, qbt.EVAL_W, 3):
        raise AR1Error(f"box downsample produced {coarse.shape}")
    return (
        torch.from_numpy(np.ascontiguousarray(coarse))
        .reshape(1, 2, qbt.EVAL_H, qbt.EVAL_W, 3)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )


def score_camera(
    camera: torch.Tensor, posenet: torch.nn.Module, segnet: torch.nn.Module
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-scorer forward -> (argmax uint8 [H, W], pose6 float32 [6])."""
    with torch.no_grad():
        pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
    argmax = logits.argmax(dim=1)[0].to(torch.uint8).numpy()
    return argmax, pose6[0].to(torch.float32).numpy()


def d_seg_d_pose(
    argmax: np.ndarray, pose6: np.ndarray, seg_gt: np.ndarray, pose_gt: np.ndarray
) -> tuple[float, float]:
    """The burn's own per-pair arithmetic (``_retain_eval_outputs``), reproduced exactly."""
    d_seg = float(np.mean(argmax != seg_gt))
    d_pose = float(np.mean(np.square(pose6.astype(np.float64) - pose_gt.astype(np.float64))))
    return d_seg, d_pose


def bhw_split(
    base_argmax: np.ndarray, aa_argmax: np.ndarray, seg_gt: np.ndarray
) -> list[dict[str, Any]]:
    """Per-target-class fixed / broken / net site counts between the base and AA renders.

    ``fixed``  -- wrong under the base render, correct under AA (AA healed it).
    ``broken`` -- correct under the base render, wrong under AA (AA broke it).
    ``net``    -- ``fixed - broken``; positive means AA is a net win on that class.
    Sites are attributed by the TARGET class, so the rows partition the frame exactly.
    """
    base_ok = base_argmax == seg_gt
    aa_ok = aa_argmax == seg_gt
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = seg_gt == class_id
        total = int(mask.sum())
        fixed = int((mask & ~base_ok & aa_ok).sum())
        broken = int((mask & base_ok & ~aa_ok).sum())
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "target_sites": total,
                "base_wrong": int((mask & ~base_ok).sum()),
                "aa_wrong": int((mask & ~aa_ok).sum()),
                "fixed": fixed,
                "broken": broken,
                "net": fixed - broken,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# pair selections
# ---------------------------------------------------------------------------
def seeded_random_32() -> tuple[int, ...]:
    """The charter's pre-registered n32: ``rng.choice(600, 32, replace=False)`` at seed 20260903."""
    rng = np.random.default_rng(SEEDED_N32_SEED)
    return tuple(int(v) for v in np.sort(rng.choice(qbt.N, 32, replace=False)))


def resolve_pairs(spec: str) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(qbt.N))
    if spec == "selection":
        return tuple(int(v) for v in qbt.SELECTION_IDS)
    if spec == "seeded32":
        return seeded_random_32()
    return tuple(sorted({int(v) for v in spec.split(",") if v.strip()}))


# ---------------------------------------------------------------------------
# main measurement loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    torch.set_num_threads(int(args.threads))
    torch.manual_seed(0)
    out = Path(args.out)
    (out / "argmax").mkdir(parents=True, exist_ok=True)
    (out / "camera").mkdir(parents=True, exist_ok=True)
    rows_path = out / "per_pair_rows.jsonl"

    checkpoint = Path(args.checkpoint)
    model, ckpt_meta = load_ema_model(checkpoint)
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    gt = load_ground_truth()

    pair_ids = resolve_pairs(args.pairs)
    ss_values = tuple(int(v) for v in args.ss.split(","))
    lattice = str(args.lattice)
    suffix = "" if lattice == LATTICE_MODULE else "_centred"

    done: set[tuple[int, int, str]] = set()
    if args.resume and rows_path.exists():
        with open(rows_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Rows written before the lattice split are module-endpoint by construction.
                done.add((int(row["pair_id"]), int(row["ss"]), row.get("lattice", LATTICE_MODULE)))
        print(f"[ar1] resuming; {len(done)} (pair, ss, lattice) rows already on disk", flush=True)

    # ss=1 is the point-sampled baseline every AA delta is measured against, and it is
    # lattice-free, so a later lattice pass may reuse the baseline already on disk rather than
    # re-rendering it.  It must exist SOMEWHERE, though, or the pass produces nothing comparable.
    if 1 not in ss_values and not any(pair_ss == 1 for _pair, pair_ss, _lat in done):
        raise AR1Error(
            "ss=1 (the trainer's point-sampled baseline) is neither in --ss nor already on disk"
        )

    started = time.time()
    written = 0
    with open(rows_path, "a", encoding="utf-8") as sink:
        for pair_id in pair_ids:
            for ss in ss_values:
                if (pair_id, ss, lattice) in done:
                    continue
                t_render0 = time.time()
                rgb = render_rgb_pair(model, pair_id, ss, lattice)
                t_render = time.time() - t_render0
                t_score0 = time.time()
                camera = qbt.roundtrip_to_camera_uint8_ste(rgb)
                argmax, pose6 = score_camera(camera, posenet, segnet)
                t_score = time.time() - t_score0

                np.save(out / "argmax" / f"pair_{pair_id:04d}_ss{ss}{suffix}.npy", argmax)
                camera_fact = None
                if pair_ids.index(pair_id) < int(args.retain_camera_pairs):
                    camera_u8 = camera.round().clamp(0, 255)[0].to(torch.uint8).numpy()
                    camera_path = out / "camera" / f"pair_{pair_id:04d}_ss{ss}{suffix}.npy"
                    np.save(camera_path, camera_u8)
                    camera_fact = file_fact(camera_path)

                d_seg_a, d_pose_a = d_seg_d_pose(
                    argmax, pose6, gt["dali_seg"][pair_id], gt["dali_pose"][pair_id]
                )
                d_seg_c, d_pose_c = d_seg_d_pose(
                    argmax, pose6, gt["pyav_seg"][pair_id], gt["pyav_pose"][pair_id]
                )
                row = {
                    "schema": SCHEMA,
                    "axis": AXIS,
                    "score_claim": False,
                    "pair_id": int(pair_id),
                    "ss": int(ss),
                    "lattice": lattice,
                    "render_grid": [qbt.EVAL_H * ss, qbt.EVAL_W * ss],
                    "base_grid": [qbt.EVAL_H, qbt.EVAL_W],
                    "d_seg_dali_authority": d_seg_a,
                    "d_pose_dali_authority": d_pose_a,
                    "d_seg_pyav_burn_continuity": d_seg_c,
                    "d_pose_pyav_burn_continuity": d_pose_c,
                    "wall_render_s": t_render,
                    "wall_score_s": t_score,
                    "wall_total_s": t_render + t_score,
                    "argmax_sha256": hashlib.sha256(argmax.tobytes()).hexdigest(),
                    "pose6": [float(v) for v in pose6],
                    "camera_retained": camera_fact,
                    "payload_retained": True,
                }
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                written += 1
                if written % 25 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[ar1] {written} rows in {elapsed:.0f}s "
                        f"(pair {pair_id}, ss {ss}, peak_rss {peak_rss_gib():.1f} GiB)",
                        flush=True,
                    )

    manifest = {
        "schema": SCHEMA + ".manifest",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "instrument": INSTRUMENT,
        "git_head": git_head(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "threads": int(args.threads),
        "checkpoint": file_fact(checkpoint),
        "checkpoint_meta": ckpt_meta,
        "trainer_source": file_fact(REPO / "experiments/ddm_qbt1_qbflow_trainer.py"),
        "aa_module_source": file_fact(REPO / "src/tac/boundary_math/aa_sdf_observation_render.py"),
        "gt_lineage": gt["lineage"],
        "pairs_spec": args.pairs,
        "lattice": lattice,
        "pair_ids": list(pair_ids),
        "ss_values": list(ss_values),
        "seeded32": list(seeded_random_32()),
        "selection_ids": list(qbt.SELECTION_IDS),
        "rows_written_this_process": written,
        "elapsed_seconds_this_process": time.time() - started,
        "peak_rss_gib": peak_rss_gib(),
        "rows_path": str(rows_path.resolve()),
        "all_rendered_argmax_retained": True,
    }
    (out / f"MANIFEST{'' if lattice == LATTICE_MODULE else '_centred'}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[ar1] wrote {written} rows; manifest at {out / 'MANIFEST.json'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# calibration gate -- three legs, so a miss names its own cause
# ---------------------------------------------------------------------------
def calibrate(args: argparse.Namespace) -> int:
    """Reproduce the burn's own recorded milestone read, decomposed into three legs.

    The burn evaluated on MPS; this instrument runs on CPU (MPS is never an authority).  A single
    pass/fail on the aggregate would therefore conflate arithmetic, scorer axis and render axis.
    The three legs separate them:

    * ``leg1_arithmetic``   -- recompute from the burn's OWN retained arrays.  Zero gap expected.
    * ``leg2_scorer_axis``  -- CPU scorers on the burn's OWN retained camera uint8 bytes.
    * ``leg3_render_axis``  -- CPU render at ss=1, then CPU scorers.  This is the ss=1 baseline the
      AA measurement compares against, so leg 3 is the number the gate is really about.
    """
    torch.set_num_threads(int(args.threads))
    milestone_dir = Path(args.milestone)
    milestone = json.loads((milestone_dir / "MILESTONE.json").read_text())
    recorded = {
        "d_seg_hat": float(milestone["d_seg_hat"]),
        "d_pose_hat": float(milestone["d_pose_hat"]),
        "S_hat": float(milestone["S_hat"]),
        "archive_bytes_exact": int(milestone["archive_bytes_exact"]),
        "axis": milestone["axis"],
        "step": int(milestone["step"]),
    }

    model, ckpt_meta = load_ema_model(Path(args.checkpoint))
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    gt = load_ground_truth()

    legs: dict[str, dict[int, tuple[float, float]]] = {"leg1": {}, "leg2": {}, "leg3": {}}
    for pair_id in qbt.SELECTION_IDS:
        with np.load(milestone_dir / "realized" / f"pair_{pair_id:04d}.npz", allow_pickle=False) as p:
            burn_argmax = np.asarray(p["segnet_argmax_u8"], dtype=np.uint8)
            burn_pose = np.asarray(p["posenet_pose6_f32"], dtype=np.float32)
            burn_target_argmax = np.asarray(p["target_argmax_u8"], dtype=np.uint8)
            burn_target_pose = np.asarray(p["target_pose6_f32"], dtype=np.float32)
            burn_camera = np.asarray(p["camera_pair_u8"], dtype=np.uint8)
        # Leg 1: pure arithmetic on the burn's own arrays and its own (PyAV) target.
        legs["leg1"][pair_id] = d_seg_d_pose(
            burn_argmax, burn_pose, burn_target_argmax, burn_target_pose
        )
        # Leg 2: CPU scorers on the burn's own camera bytes; same target frame as leg 1.
        camera = torch.from_numpy(burn_camera).to(torch.float32).unsqueeze(0)
        argmax2, pose2 = score_camera(camera, posenet, segnet)
        legs["leg2"][pair_id] = d_seg_d_pose(
            argmax2, pose2, burn_target_argmax, burn_target_pose
        )
        # Leg 3: CPU render at ss=1 -> CPU roundtrip -> CPU scorers; same target frame.
        rgb = render_rgb_pair(model, pair_id, 1)
        argmax3, pose3 = score_camera(
            qbt.roundtrip_to_camera_uint8_ste(rgb), posenet, segnet
        )
        legs["leg3"][pair_id] = d_seg_d_pose(
            argmax3, pose3, burn_target_argmax, burn_target_pose
        )

    report: dict[str, Any] = {
        "schema": SCHEMA + ".calibration",
        "axis": AXIS,
        "score_claim": False,
        "recorded_milestone": recorded,
        "milestone_source": str(milestone_dir.resolve()),
        "checkpoint": file_fact(Path(args.checkpoint)),
        "checkpoint_meta": ckpt_meta,
        "gate_tolerance_relative": float(args.tolerance),
        "gt_frame_used": "PyAV (the burn's own target arrays, retained in its milestone npz)",
        "gt_lineage": gt["lineage"],
        "legs": {},
    }
    for leg_name, label in (
        ("leg1", "arithmetic on the burn's own retained arrays"),
        ("leg2", "CPU scorers on the burn's own retained camera bytes"),
        ("leg3", "CPU render at ss=1 -> CPU roundtrip -> CPU scorers"),
    ):
        seg = {pid: v[0] for pid, v in legs[leg_name].items()}
        pose = {pid: v[1] for pid, v in legs[leg_name].items()}
        d_seg_ht, d_pose_ht = _ht_mean(seg), _ht_mean(pose)
        if d_seg_ht is None or d_pose_ht is None:
            raise AR1Error(
                f"{leg_name} did not cover the burn's full selection; the HT estimator is "
                "undefined and the gate would be vacuous"
            )
        entry = {
            "label": label,
            "d_seg_hat": d_seg_ht,
            "d_pose_hat": d_pose_ht,
            "S_hat": _s_hat(d_seg_ht, d_pose_ht, recorded["archive_bytes_exact"]),
            "d_seg_relative_gap": (d_seg_ht - recorded["d_seg_hat"]) / recorded["d_seg_hat"],
            "d_pose_relative_gap": (d_pose_ht - recorded["d_pose_hat"]) / recorded["d_pose_hat"],
            "per_pair": {str(pid): {"d_seg": seg[pid], "d_pose": pose[pid]} for pid in seg},
        }
        entry["within_tolerance"] = bool(
            abs(entry["d_seg_relative_gap"]) <= float(args.tolerance)
            and abs(entry["d_pose_relative_gap"]) <= float(args.tolerance)
        )
        report["legs"][leg_name] = entry

    report["gate"] = {
        "leg1_arithmetic_reproduces": report["legs"]["leg1"]["within_tolerance"],
        "leg3_cpu_render_reproduces": report["legs"]["leg3"]["within_tolerance"],
        "verdict": (
            "PASS"
            if report["legs"]["leg1"]["within_tolerance"]
            and report["legs"]["leg3"]["within_tolerance"]
            else "EXPLAINED_AXIS_GAP"
            if report["legs"]["leg1"]["within_tolerance"]
            else "FAIL_ARITHMETIC"
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "CALIBRATION.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for leg_name, entry in report["legs"].items():
        print(
            f"[ar1] {leg_name}: d_seg_hat={entry['d_seg_hat']:.9f} "
            f"({entry['d_seg_relative_gap']:+.3%})  "
            f"d_pose_hat={entry['d_pose_hat']:.9e} ({entry['d_pose_relative_gap']:+.3%})  "
            f"-- {entry['label']}"
        )
    print(f"[ar1] recorded: d_seg_hat={recorded['d_seg_hat']:.9f} "
          f"d_pose_hat={recorded['d_pose_hat']:.9e} axis={recorded['axis']}")
    print(f"[ar1] gate verdict: {report['gate']['verdict']}; written to {path}")
    return 0


# ---------------------------------------------------------------------------
# aggregation (re-runnable from the retained per-pair rows + argmax payloads)
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if len(values) else float("nan")


def _ht_mean(lookup: dict[int, float]) -> float | None:
    """The burn's own Horvitz-Thompson estimator over ``qbt.SELECTION_IDS``."""
    if set(lookup) != set(qbt.SELECTION_IDS):
        return None
    total = sum(
        weight * lookup[pair_id]
        for pair_id, weight in zip(qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS, strict=True)
    )
    return float(total / qbt.N)


def _s_hat(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return (
        100.0 * d_seg
        + float(np.sqrt(10.0 * d_pose))
        + 25.0 * float(archive_bytes) / float(qbt.RATE_DENOMINATOR)
    )


def _subset_block(
    rows_by_ss: dict[int, dict[int, dict[str, Any]]],
    ids: Sequence[int],
    archive_bytes: int,
    gt_key: str,
) -> dict[str, Any]:
    """Aggregate one pair subset on one GT frame, for every measured ``ss``."""
    ids = [int(v) for v in ids]
    block: dict[str, Any] = {"n": len(ids), "pair_ids": ids}
    per_ss: dict[str, Any] = {}
    for ss, rows in sorted(rows_by_ss.items()):
        if not all(pid in rows for pid in ids):
            continue
        seg = {pid: float(rows[pid][f"d_seg_{gt_key}"]) for pid in ids}
        pose = {pid: float(rows[pid][f"d_pose_{gt_key}"]) for pid in ids}
        entry = {
            "d_seg_mean": _mean(list(seg.values())),
            "d_pose_mean": _mean(list(pose.values())),
            "wall_total_s_mean": _mean([float(rows[pid]["wall_total_s"]) for pid in ids]),
            "wall_render_s_mean": _mean([float(rows[pid]["wall_render_s"]) for pid in ids]),
        }
        entry["S_hat_mean_estimator"] = _s_hat(
            entry["d_seg_mean"], entry["d_pose_mean"], archive_bytes
        )
        ht_seg, ht_pose = _ht_mean(seg), _ht_mean(pose)
        if ht_seg is not None and ht_pose is not None:
            entry["d_seg_ht"] = ht_seg
            entry["d_pose_ht"] = ht_pose
            entry["S_hat_ht"] = _s_hat(ht_seg, ht_pose, archive_bytes)
        per_ss[str(ss)] = entry
    block["per_ss"] = per_ss
    base = per_ss.get("1")
    if base is not None:
        deltas = {}
        for ss_key, entry in per_ss.items():
            if ss_key == "1":
                continue
            d_seg_ratio = (
                base["d_seg_mean"] / entry["d_seg_mean"] if entry["d_seg_mean"] > 0 else float("inf")
            )
            delta_s = entry["S_hat_mean_estimator"] - base["S_hat_mean_estimator"]
            deltas[ss_key] = {
                "d_seg_ratio_base_over_aa": d_seg_ratio,
                "d_seg_delta": entry["d_seg_mean"] - base["d_seg_mean"],
                "d_pose_delta": entry["d_pose_mean"] - base["d_pose_mean"],
                "delta_S_mean_estimator": delta_s,
                "delta_S_equivalent_bytes": delta_s
                / (25.0 / float(qbt.RATE_DENOMINATOR)),
                "wall_cost_multiple": (
                    entry["wall_total_s_mean"] / base["wall_total_s_mean"]
                    if base["wall_total_s_mean"] > 0
                    else float("nan")
                ),
            }
            if "S_hat_ht" in base and "S_hat_ht" in entry:
                deltas[ss_key]["delta_S_ht"] = entry["S_hat_ht"] - base["S_hat_ht"]
                deltas[ss_key]["d_seg_ratio_ht"] = (
                    base["d_seg_ht"] / entry["d_seg_ht"] if entry["d_seg_ht"] > 0 else float("inf")
                )
            # A mean can be dragged by one pair, so record the per-pair sign census too.
            base_rows = rows_by_ss[1]
            aa_rows = rows_by_ss[int(ss_key)]
            per_pair = [
                (
                    float(base_rows[pid][f"d_seg_{gt_key}"]),
                    float(aa_rows[pid][f"d_seg_{gt_key}"]),
                )
                for pid in ids
            ]
            deltas[ss_key]["pairs_aa_better"] = sum(1 for b, a in per_pair if a < b)
            deltas[ss_key]["pairs_aa_worse"] = sum(1 for b, a in per_pair if a > b)
            deltas[ss_key]["pairs_unchanged"] = sum(1 for b, a in per_pair if a == b)
            deltas[ss_key]["d_seg_ratio_median_over_pairs"] = float(
                np.median([b / a if a > 0 else np.inf for b, a in per_pair])
            )
        block["deltas_vs_ss1"] = deltas
    return block


def aggregate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows_path = out / "per_pair_rows.jsonl"
    lattice = str(args.lattice)
    rows_by_ss: dict[int, dict[int, dict[str, Any]]] = {}
    with open(rows_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # ss=1 is lattice-free (no supersample, no downsample), so it is the shared baseline
            # for every lattice.  Rows predating the lattice split are module-endpoint.
            row_lattice = row.get("lattice", LATTICE_MODULE)
            if int(row["ss"]) != 1 and row_lattice != lattice:
                continue
            rows_by_ss.setdefault(int(row["ss"]), {})[int(row["pair_id"])] = row

    if 1 not in rows_by_ss or len(rows_by_ss) < 2:
        raise AR1Error(
            "aggregate needs the ss=1 baseline and at least one AA mode; "
            f"found ss values {sorted(rows_by_ss)}"
        )
    gt = load_ground_truth()
    archive_bytes = int(args.archive_bytes)
    # Only pairs measured at EVERY ss enter an aggregate, so no subset mixes modes.
    common = sorted(set.intersection(*(set(v) for v in rows_by_ss.values())))
    if not common:
        raise AR1Error("no pair was measured at every ss; nothing is comparable")

    # The QBR1 burn optimises ONLY its 32 selection pairs (config ``pair_ids`` == SELECTION_IDS),
    # so the born field's per-pair latents are trained on those 32 and on no others.  The
    # remaining 568 pairs carry inherited latents that no d_seg/d_pose gradient ever touched
    # (they move only by AdamW's decoupled weight decay).  They are a DIFFERENT object and are
    # never averaged into the born-field number.
    selection = set(qbt.SELECTION_IDS)
    subsets = {
        "burn_selection_n32_trained": [p for p in qbt.SELECTION_IDS if p in common],
        "untrained_n568": [p for p in common if p not in selection],
        "charter_seeded_n32": [p for p in seeded_random_32() if p in common],
        "all_measured_mixed_population": common,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA + ".aggregate",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_bytes_used_for_S": archive_bytes,
        "rate_at_archive_bytes": 25.0 * archive_bytes / float(qbt.RATE_DENOMINATOR),
        "exchange_rate_S_per_byte_reference": 6.658589531221714e-7,
        "gt_lineage": gt["lineage"],
        "lattice": lattice,
        "ss_measured": sorted(rows_by_ss),
        "population_scope": {
            "burn_trained_pair_ids": list(qbt.SELECTION_IDS),
            "burn_trained_n": len(qbt.SELECTION_IDS),
            "untrained_n_measured": len([p for p in common if p not in set(qbt.SELECTION_IDS)]),
            "charter_seeded_n32_trained_overlap": sorted(
                set(seeded_random_32()) & set(qbt.SELECTION_IDS)
            ),
            "note": (
                "the burn's config pair_ids equals SELECTION_IDS; only those 32 pairs received a "
                "d_seg/d_pose gradient. Non-selection latents moved only by AdamW decoupled "
                "weight decay. The born-field claim is the burn_selection_n32_trained block."
            ),
        },
        "subsets": {},
    }
    for gt_key, gt_label in (
        ("dali_authority", "DALI (authority)"),
        ("pyav_burn_continuity", "PyAV (burn continuity only)"),
    ):
        report["subsets"][gt_key] = {
            "gt_label": gt_label,
            **{
                name: _subset_block(rows_by_ss, ids, archive_bytes, gt_key)
                for name, ids in subsets.items()
                if ids
            },
        }

    # per-class B/H/W between ss=1 and every ss>1, summed over the measured pairs.
    argmax_suffix = "" if lattice == LATTICE_MODULE else "_centred"
    bhw: dict[str, Any] = {}
    for ss in sorted(rows_by_ss):
        if ss == 1:
            continue
        totals = [
            {"class_id": c, "class_name": n, "target_sites": 0, "base_wrong": 0,
             "aa_wrong": 0, "fixed": 0, "broken": 0, "net": 0}
            for c, n in enumerate(CLASS_NAMES)
        ]
        for pair_id in common:
            base = np.load(out / "argmax" / f"pair_{pair_id:04d}_ss1.npy")
            aa = np.load(out / "argmax" / f"pair_{pair_id:04d}_ss{ss}{argmax_suffix}.npy")
            for row in bhw_split(base, aa, gt["dali_seg"][pair_id]):
                target = totals[row["class_id"]]
                for key in ("target_sites", "base_wrong", "aa_wrong", "fixed", "broken", "net"):
                    target[key] += row[key]
        bhw[str(ss)] = {
            "gt": "DALI (authority)",
            "pairs": len(common),
            "per_class": totals,
            "total_fixed": sum(r["fixed"] for r in totals),
            "total_broken": sum(r["broken"] for r in totals),
            "total_net": sum(r["net"] for r in totals),
        }
    report["bhw_vs_ss1"] = bhw

    path = out / f"AGGREGATE{'' if lattice == LATTICE_MODULE else '_centred'}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["subsets"]["dali_authority"], indent=2, sort_keys=True))
    print(f"[ar1] aggregate written to {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    measure = sub.add_parser("measure", help="render + score every (pair, ss) cell")
    measure.add_argument("--checkpoint", required=True)
    measure.add_argument("--out", required=True)
    measure.add_argument("--pairs", default="all", help="all | selection | seeded32 | csv ids")
    measure.add_argument("--ss", default="1,2", help="comma-separated supersample factors")
    measure.add_argument("--threads", type=int, default=4)
    measure.add_argument("--retain-camera-pairs", type=int, default=4)
    measure.add_argument(
        "--lattice", default=LATTICE_MODULE, choices=[LATTICE_MODULE, LATTICE_CENTRED]
    )
    measure.add_argument("--resume", action="store_true")

    cal = sub.add_parser("calibrate", help="three-leg reproduction of the burn's own read")
    cal.add_argument("--checkpoint", required=True)
    cal.add_argument("--milestone", required=True, help="a FINISHED milestone step directory")
    cal.add_argument("--out", required=True)
    cal.add_argument("--threads", type=int, default=4)
    cal.add_argument("--tolerance", type=float, default=0.02)

    agg = sub.add_parser("aggregate", help="aggregate the retained per-pair rows")
    agg.add_argument("--out", required=True)
    agg.add_argument("--archive-bytes", type=int, required=True)
    agg.add_argument(
        "--lattice", default=LATTICE_MODULE, choices=[LATTICE_MODULE, LATTICE_CENTRED]
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "measure":
        return run(args)
    if args.mode == "calibrate":
        return calibrate(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
