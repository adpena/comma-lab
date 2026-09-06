#!/usr/bin/env python3
"""ddm_sj1 -- the JOINT half: pose damage, carrier re-solve, Lagrange admission.

WHY THIS EXISTS
---------------
Token edits are a seg actuator AND a pose actuator.  SegNet reads only frame ``2p+1``
(``upstream/modules.py:108``), but PoseNet reads BOTH frames
(``upstream/modules.py:72-80``), and frame ``2p`` is a photometric probe whose carrier
coefficients were solved against the ORIGINAL frame ``2p+1``.  Editing tokens therefore
leaves every edited pair shipping a carrier that is stale BY CONSTRUCTION.  ``ddm_jg4``
measured that stale carrier costing pose x387; ``ddm_jg5`` measured the re-solve
recovering it to ~1.07x at ~0 bytes.  The two actuators COMPOSE, and this module is
where the composition is measured rather than assumed.

WHAT IS REUSED VERBATIM
-----------------------
* ``ddm_jg5.refine_pair`` -- br1's damped Gauss-Newton on the shipped 12-dim basis and
  the shipped signed-int12 lattice, alternating with the +-2 polish, stopping on the
  pair's OWN measured decay against a DERIVED materiality floor.  Re-deriving a solver
  here would only risk a weaker one (br1's binding lesson was that the wall was up2's
  +-2 single-coordinate SEARCH, not the basis).
* ``ddm_up2.measure_pose`` / ``load_carrier_state`` / ``load_gt_poses`` -- the receiver's
  own frame-0 render and the DALI-lineage GT table.
* ``ddm_jg5``'s multiplier sweep STRUCTURE -- seg credit and rate cost are additive per
  pair, pose damage is not, which is exactly why pose is the multiplier's subject.

AUTHORITY
---------
Frozen CPU-torch PoseNet, GT decoded by DALI -- the lineage ``upstream/evaluate.py`` uses
on the contest-CUDA axis.  ``score_claim=false``, ``promotable=false``: nothing here is a
row until MAIN fires T4 on a sealed archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_jg1_seg_solve as jg1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_sj1_multipass_token_predistortion as sj1
import ddm_up2_shipping_pose_solve as up2

N_PAIRS = sj1.N_PAIRS
CAMERA_H, CAMERA_W = jg1.CAMERA_H, jg1.CAMERA_W
SCORE_RATE_DENOMINATOR = 37_545_489


class Sj1JointError(RuntimeError):
    """A precondition of the joint half failed.  Fail closed."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _set_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, threads))


def load_edit_planes(path: Path) -> dict[int, np.ndarray]:
    """Load an edited-field npz as ``{pair: (384, 512) uint8}``, refusing a non-field."""
    planes: dict[int, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as blob:
        for key in blob.files:
            plane = np.asarray(blob[key], dtype=np.uint8)
            if plane.shape != (sj1.EVAL_H, sj1.EVAL_W):
                raise Sj1JointError(f"edit plane {key} has shape {plane.shape}")
            if plane.max() >= sj1.NUM_CLASSES:
                raise Sj1JointError(f"edit plane {key} carries a token outside the domain")
            planes[int(key)] = plane
    return planes


class OverlayRaw:
    """The base decode with the CANDIDATE's odd frames spliced in.

    ``ddm_jg5`` solved against the candidate's own full decode.  Here the same object is
    assembled without a 3.66 GB copy: the candidate differs from the base ONLY on odd
    frames of edited pairs (SegNet's frame; token edits cannot reach frame ``2p``, which
    the carrier renders), so those frames are rendered once, persisted to a memmap, and
    substituted on read.  Everything else is the base decode's own bytes.

    Access is restricted to the shapes the pose instruments actually use --
    ``raw[int_array]`` -- and anything else refuses rather than silently returning base
    frames where candidate frames were meant.
    """

    def __init__(self, base, overlay: np.ndarray, index: dict[int, int]) -> None:
        self.base = base
        self.overlay = overlay
        self.index = index
        self.shape = base.shape
        self.dtype = base.dtype

    def __getitem__(self, key):
        arr = np.asarray(key)
        if arr.ndim != 1:
            raise Sj1JointError(
                f"OverlayRaw only serves 1-D frame-index arrays, got ndim={arr.ndim}; "
                "a different access shape would silently mix base and candidate frames"
            )
        out = np.asarray(self.base[arr]).copy()
        for row, frame in enumerate(arr.tolist()):
            if frame % 2 == 1:
                pair = frame // 2
                slot = self.index.get(pair)
                if slot is not None:
                    out[row] = self.overlay[slot]
        return out


def open_overlay(base_raw_path: Path, overlay_dir: Path) -> OverlayRaw:
    manifest = json.loads((overlay_dir / "OVERLAY.json").read_text())
    pairs = [int(p) for p in manifest["pairs"]]
    overlay = np.memmap(
        overlay_dir / "odd_frames.u8",
        dtype=np.uint8,
        mode="r",
        shape=(len(pairs), CAMERA_H, CAMERA_W, 3),
    )
    base = np.memmap(
        base_raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3),
    )
    return OverlayRaw(base, overlay, {pair: i for i, pair in enumerate(pairs)})


# --------------------------------------------------------------------------------------
# render-edits
# --------------------------------------------------------------------------------------


def cmd_render_edits(args) -> int:
    """Render frame 2p+1 for every edited pair, at the receiver's own batch 1."""
    _set_threads(args.threads)
    planes = load_edit_planes(args.field)
    body = sj1.load_body(with_raw=False, verify_shas=not args.no_verify_shas)
    pairs = sorted(planes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "odd_frames.u8"
    overlay = np.memmap(
        path, dtype=np.uint8, mode="w+", shape=(len(pairs), CAMERA_H, CAMERA_W, 3)
    )
    started = time.time()
    for slot, pair in enumerate(pairs):
        overlay[slot] = jg1.render_frame1(
            body.semantic, planes[pair][None], np.array([pair])
        )[0]
        if args.progress and (slot + 1) % 25 == 0:
            print(
                f"  rendered {slot + 1}/{len(pairs)} "
                f"({(time.time() - started) / (slot + 1):.2f} s/pair)",
                flush=True,
            )
    overlay.flush()
    del overlay
    manifest = {
        "schema": "ddm_sj1_overlay.v1",
        "pairs": pairs,
        "field_npz": str(args.field),
        "field_npz_sha256": _sha256_file(args.field),
        "odd_frames_path": str(path),
        "odd_frames_sha256": _sha256_file(path),
        "odd_frames_bytes": path.stat().st_size,
        "semantic_batch": 1,
        "elapsed_seconds": time.time() - started,
        "receipts": body.receipts,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }
    (args.out_dir / "OVERLAY.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("odd_frames_bytes", "elapsed_seconds")}))
    return 0


# --------------------------------------------------------------------------------------
# pose measurement
# --------------------------------------------------------------------------------------


def load_pose_instrument(overlay_dir: Path | None):
    """br1's instrument on the cl2 body, reading whichever decode is asked for."""
    sj1.assert_carrier_is_pointer(sj1.POINTER_TREE)
    state = up2.load_carrier_state(sj1.POINTER_TREE, verify_archive=False)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise Sj1JointError(
            f"GT pose lineage is {lineage}, not {up2.LINEAGE_DALI}: that would solve the "
            "contest-CPU objective, which is a different object (evaluate.py:31-42)"
        )
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    if overlay_dir is None:
        raw = np.memmap(
            sj1.BODY_RAW,
            dtype=np.uint8,
            mode="r",
            shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3),
        )
    else:
        raw = open_overlay(sj1.BODY_RAW, overlay_dir)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


def cmd_pose(args) -> int:
    """Per-pair d_pose over ALL 600 pairs on one decode with one set of codes."""
    _set_threads(args.threads)
    inst = load_pose_instrument(args.overlay)
    codes = (
        np.load(args.codes).astype(np.int32)
        if args.codes
        else np.asarray(inst.state.codes, dtype=np.int32)
    )
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Sj1JointError(f"codes have shape {codes.shape}")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    indices = np.arange(N_PAIRS, dtype=np.int64)
    started = time.time()
    per_pair, _poses = up2.measure_pose(
        inst.posenet,
        inst.state,
        coefficients,
        inst.raw,
        inst.targets,
        indices,
        batch_size=args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, per_pair)
    report = {
        "schema": "ddm_sj1_pose_leg.v1",
        "tag": args.tag,
        "decode": "candidate_overlay" if args.overlay else "base",
        "overlay_dir": str(args.overlay) if args.overlay else None,
        "codes_source": str(args.codes) if args.codes else "shipped_carrier",
        "pairs": N_PAIRS,
        "d_pose_mean": float(per_pair.mean()),
        "pose_leg": jg5.pose_leg(float(per_pair.mean())),
        "per_pair_path": str(args.out),
        "per_pair_sha256": _sha256_file(args.out),
        "elapsed_seconds": time.time() - started,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    (args.out.parent / f"POSE_{args.tag}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps({k: report[k] for k in ("tag", "d_pose_mean", "pose_leg", "elapsed_seconds")}))
    return 0


def cmd_refine(args) -> int:
    """Re-solve the carrier on the CANDIDATE's renders, one strided shard of pairs."""
    _set_threads(args.threads)
    inst = load_pose_instrument(args.overlay)
    targets = sorted(load_edit_planes(args.field))
    shard = [p for i, p in enumerate(targets) if i % args.shard_count == args.shard_index]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / f"refine_rows_{args.shard_index}.jsonl"
    done: set[int] = set()
    if rows_path.is_file() and args.resume:
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(int(json.loads(line)["pair"]))
    base_pose = np.load(args.base_pose)
    if base_pose.shape != (N_PAIRS,):
        raise Sj1JointError(f"base pose vector has shape {base_pose.shape}")
    dd_threshold = jg5.materiality_dd_threshold(float(base_pose.mean()))
    started = time.time()
    for count, pair in enumerate(shard):
        if pair in done:
            continue
        row = jg5.refine_pair(
            inst,
            pair,
            np.asarray(inst.state.codes, dtype=np.int32)[pair],
            dd_threshold=dd_threshold,
            outer_rounds=args.outer_rounds,
            max_gn_iterations=args.max_gn_iterations,
        )
        with rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if args.progress:
            print(
                f"  shard {args.shard_index}: {count + 1}/{len(shard)} pair={pair} "
                f"{row['start_d_pose']:.3e} -> {row['final_d_pose']:.3e} "
                f"({(time.time() - started) / (count + 1):.1f} s/pair)",
                flush=True,
            )
    receipt = {
        "schema": "ddm_sj1_refine_shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": shard,
        "rows_path": str(rows_path),
        "dd_threshold": dd_threshold,
        "elapsed_seconds": time.time() - started,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    (args.out_dir / f"REFINE_SHARD_{args.shard_index}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)
    )
    print(json.dumps({k: receipt[k] for k in ("shard_index", "elapsed_seconds")}))
    return 0


def cmd_codes(args) -> int:
    """Merge refine shards into a full (600, 12) code table, base codes elsewhere."""
    sj1.assert_carrier_is_pointer(sj1.POINTER_TREE)
    state = up2.load_carrier_state(sj1.POINTER_TREE, verify_archive=False)
    codes = np.asarray(state.codes, dtype=np.int32).copy()
    solved = np.zeros(N_PAIRS, dtype=bool)
    resolved = {}
    for path in args.rows:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                pair = int(row["pair"])
                codes[pair] = np.asarray(row["codes"], dtype=np.int32)
                solved[pair] = True
                resolved[pair] = float(row["final_d_pose"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, codes)
    np.save(args.out.with_name(args.out.stem + "_solved_mask.npy"), solved)
    report = {
        "schema": "ddm_sj1_codes.v1",
        "pairs_resolved": int(solved.sum()),
        "codes_path": str(args.out),
        "codes_sha256": _sha256_file(args.out),
        "coordinates_changed_vs_shipped": int(
            (codes != np.asarray(state.codes, dtype=np.int32)).sum()
        ),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }
    (args.out.parent / "CODES.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------------------
# admission -- jg5's multiplier sweep, on this body's constants
# --------------------------------------------------------------------------------------


def composed_score(d_seg: float, d_pose: float, archive_bytes: float) -> float:
    return (
        100.0 * d_seg
        + math.sqrt(10.0 * d_pose)
        + 25.0 * archive_bytes / SCORE_RATE_DENOMINATOR
    )


def cmd_admit(args) -> int:
    """Lagrange sweep: which edited pairs pay once their pose damage is charged?

    Seg credit and rate cost are ADDITIVE per pair; the pose leg is a square root of a
    MEAN and therefore is not, which is exactly why pose is the multiplier's subject
    rather than another additive term (``ddm_jg5.run_waterfill``).  The rate leg here is
    MODELLED from the per-pair token-change count and is re-measured exactly by
    re-encoding the admitted field; ``score_modelled`` is optimistic on rate until then.
    """
    rows = [json.loads(line) for line in Path(args.pass_rows).read_text().splitlines() if line.strip()]
    seg_repaired = np.zeros(N_PAIRS, dtype=np.float64)
    tokens_changed = np.zeros(N_PAIRS, dtype=np.float64)
    for row in rows:
        seg_repaired[int(row["pair"])] = float(row["flips_repaired"])
        tokens_changed[int(row["pair"])] = float(row["tokens_changed"])
    edited = tokens_changed > 0

    # PER-PAIR RATE LEG.  Preferred source is the two per-frame bit ledgers the REAL
    # re-encodes wrote: their difference is the marginal cost of each pair's edits under
    # the shipped model, measured rather than apportioned.  The uniform
    # bytes-per-changed-token fallback exists only for a field that has not been encoded
    # yet, and it is exactly the average-for-marginal substitution that
    # ``token_rate_model_direction_dependence_v1`` warns about -- fs3 measured that
    # substitution wrong by 2.24x -- so the ledgers win whenever they exist.
    if args.bits_control and args.bits_candidate:
        bits_c = np.load(args.bits_control).astype(np.float64)
        bits_e = np.load(args.bits_candidate).astype(np.float64)
        if bits_c.shape != (N_PAIRS,) or bits_e.shape != (N_PAIRS,):
            raise Sj1JointError(
                f"bit ledgers have shapes {bits_c.shape} / {bits_e.shape}"
            )
        per_pair_bytes = (bits_e - bits_c) / 8.0
        rate_source = {
            "kind": "measured_per_frame_bit_ledgers",
            "bits_control": str(args.bits_control),
            "bits_candidate": str(args.bits_candidate),
            "ledger_total_bytes": float(per_pair_bytes.sum()),
        }
    else:
        per_pair_bytes = tokens_changed * args.modelled_bytes_per_changed_token
        rate_source = {
            "kind": "uniform_bytes_per_changed_token",
            "bytes_per_changed_token": args.modelled_bytes_per_changed_token,
            "ledger_total_bytes": float(per_pair_bytes.sum()),
        }

    base_pose = np.load(args.base_pose)
    stale_pose = np.load(args.stale_pose)
    resolved_pose = np.load(args.resolved_pose)
    for name, vec in (
        ("base", base_pose),
        ("stale", stale_pose),
        ("resolved", resolved_pose),
    ):
        if vec.shape != (N_PAIRS,):
            raise Sj1JointError(f"{name} pose vector has shape {vec.shape}")

    # Seg credit in score units, on the T4 axis, carried by the SAME-instrument ratio.
    ratio_t4 = sj1.POINTER_D_SEG_T4 / args.instrument_base_d_seg
    seg_credit = 100.0 * seg_repaired * ratio_t4 / (N_PAIRS * sj1.EVAL_H * sj1.EVAL_W)
    rate_cost = 25.0 * per_pair_bytes / SCORE_RATE_DENOMINATOR
    pose_damage = resolved_pose - base_pose

    def score_subset(keep: np.ndarray) -> dict[str, float]:
        repaired = float(seg_repaired[keep].sum())
        d_seg_instr = args.instrument_base_d_seg - repaired / (
            N_PAIRS * sj1.EVAL_H * sj1.EVAL_W
        )
        d_seg_t4 = d_seg_instr * ratio_t4
        pose = np.where(keep, resolved_pose, base_pose)
        d_pose = float(pose.mean())
        archive_bytes = args.base_archive_bytes + float(per_pair_bytes[keep].sum())
        return {
            "pairs_kept": int(keep.sum()),
            "flips_repaired": repaired,
            "d_seg_instrument": d_seg_instr,
            "d_seg_t4": d_seg_t4,
            "d_pose": d_pose,
            "archive_bytes_modelled": archive_bytes,
            "score_modelled": composed_score(d_seg_t4, d_pose, archive_bytes),
        }

    reference = score_subset(np.zeros(N_PAIRS, dtype=bool))
    full = score_subset(edited.copy())
    lambdas = np.concatenate([np.zeros(1), np.logspace(-3.0, 6.0, num=1801)])
    best: dict[str, Any] | None = None
    best_keep: np.ndarray | None = None
    trace: list[dict[str, float]] = []
    seen: set[bytes] = set()
    for lam in lambdas:
        keep = edited & ((seg_credit - rate_cost) > lam * pose_damage)
        key = np.packbits(keep).tobytes()
        if key in seen:
            continue
        seen.add(key)
        scored = score_subset(keep)
        scored["lambda"] = float(lam)
        trace.append(scored)
        if best is None or scored["score_modelled"] < best["score_modelled"]:
            best = dict(scored)
            best_keep = keep.copy()
    if best is None or best_keep is None:
        raise Sj1JointError("multiplier sweep produced no subset")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "keep_mask.npy", best_keep)
    kept = sorted(int(p) for p in np.where(best_keep)[0])
    (args.out_dir / "kept_pairs.json").write_text(json.dumps(kept))
    source = load_edit_planes(args.field)
    missing = [p for p in kept if p not in source]
    if missing:
        raise Sj1JointError(
            f"{len(missing)} admitted pairs are absent from {args.field} "
            f"(first: {missing[:5]}); the admission and the edit field disagree"
        )
    # THE SUBSET MUST BE WRITTEN AGAINST THE CARRY FIELD, NOT AGAINST NOTHING.
    # An edit npz is spliced onto the ORIGINAL base field, so a pair merely ABSENT from it
    # reverts all the way to the original -- which on a SUCCESSOR round silently undoes the
    # edits the live pointer already banked.  Measured on this arm's pass-3 admission: 230
    # non-admitted pairs would have reverted 2,337 pass-2a tokens, and nothing downstream
    # could see it, because the result is a perfectly valid field.  Same genus as the
    # carrier silent-revert: the cure is to write what the pair should HOLD, never to rely
    # on absence meaning "unchanged".
    carry: dict[int, np.ndarray] = {}
    if args.carry_field:
        carry = load_edit_planes(args.carry_field)
    planes = {int(pair): source[pair] for pair in kept}
    for pair, plane in carry.items():
        planes.setdefault(pair, plane)
    subset = args.out_dir / "field_admitted.npz"
    np.savez_compressed(subset, **{str(pair): plane for pair, plane in sorted(planes.items())})

    # Fail closed: every carried pair must ship EXACTLY what the live row ships, and every
    # admitted pair must ship its new plane.
    written = load_edit_planes(subset)
    for pair, plane in carry.items():
        if pair in kept:
            continue
        if not np.array_equal(written.get(pair), plane):
            raise Sj1JointError(
                f"pair {pair} is not admitted but the subset field does not carry the "
                "live row's plane for it; that would revert banked edits"
            )
    for pair in kept:
        if not np.array_equal(written.get(pair), source[pair]):
            raise Sj1JointError(f"admitted pair {pair} is not written with its new plane")
    summary = {
        "schema": "ddm_sj1_admission.v1",
        "edited_pairs": int(edited.sum()),
        "reference_drop_everything": reference,
        "full_edit_set": full,
        "best": best,
        "best_kept_pairs": len(kept),
        "pointer_score_t4": sj1.POINTER_SCORE_T4,
        "pointer_archive_sha256": sj1.POINTER_ARCHIVE_SHA256,
        "net_vs_pointer_modelled": best["score_modelled"] - sj1.POINTER_SCORE_T4,
        "admitted_field_npz": str(subset),
        "admitted_field_sha256": _sha256_file(subset),
        "rate_leg_source": rate_source,
        "rate_leg_is_modelled_not_measured": rate_source["kind"] != "measured_per_frame_bit_ledgers",
        "instrument_base_d_seg": args.instrument_base_d_seg,
        "t4_ratio_applied": ratio_t4,
        "axis": (
            "seg from the jg1 DALI instrument carried onto T4 by the same-instrument "
            "ratio; pose [macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]; rate "
            "MODELLED and MUST be re-measured by re-encoding"
        ),
        "score_claim": False,
        "promotable": False,
        "trace": trace,
    }
    (args.out_dir / "ADMISSION.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))
    return 0


def patch_inflate_pins(runtime_root: Path, archive_sha256: str, archive_bytes: int) -> dict[str, Any]:
    """Re-pin the receiver's two archive constants, finding them by PATTERN.

    ``ddm_cl2``'s patcher matches the literal it expects to replace and formats the size
    with ``_`` separators; the pointer tree carries a different sha and writes ``174786``
    plain, so that helper refuses here.  Binding a patcher to one generation's literals is
    the same expiring-constant shape this arm already met in the pointer table, so this
    one reads whatever the tree currently pins, checks there is exactly one of each, and
    verifies what it wrote.
    """
    import re

    path = Path(runtime_root) / "inflate.py"
    text = path.read_text(encoding="utf-8")
    sha_pat = re.compile(r'^ARCHIVE_SHA256 = "([0-9a-f]{64})"$', re.MULTILINE)
    bytes_pat = re.compile(r"^ARCHIVE_BYTES = ([0-9_]+)$", re.MULTILINE)
    shas, sizes = sha_pat.findall(text), bytes_pat.findall(text)
    if len(shas) != 1 or len(sizes) != 1:
        raise Sj1JointError(
            f"{path} pins are absent or ambiguous: {len(shas)} sha, {len(sizes)} size"
        )
    if shas[0] != sj1.POINTER_ARCHIVE_SHA256:
        raise Sj1JointError(
            f"{path} currently pins {shas[0]}, not the live pointer "
            f"{sj1.POINTER_ARCHIVE_SHA256}; this is not the tree it claims to be"
        )
    text = sha_pat.sub(f'ARCHIVE_SHA256 = "{archive_sha256}"', text, count=1)
    text = bytes_pat.sub(f"ARCHIVE_BYTES = {archive_bytes}", text, count=1)
    path.write_text(text, encoding="utf-8")

    check = path.read_text(encoding="utf-8")
    if sha_pat.findall(check) != [archive_sha256] or bytes_pat.findall(check) != [
        str(archive_bytes)
    ]:
        raise Sj1JointError("inflate.py pin patch did not land exactly once")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "pinned_archive_sha256": archive_sha256,
        "pinned_archive_bytes": archive_bytes,
        "previous_pinned_sha256": shas[0],
    }


# --------------------------------------------------------------------------------------
# stage-tail -- put the re-encoded token stream into the LIVE POINTER's member
# --------------------------------------------------------------------------------------


def cmd_stage_tail(args) -> int:
    """Splice the edited field's token stream into the pointer's archive, and stage a tree.

    The stream is produced by cl2's encoder (whose hpac coding jg2 mirrors) but must ship
    inside the POINTER's member, which carries rc1's recoded MODEL sections and pc1's
    re-solved carrier.  The byte-identical tail across those bodies is what makes that
    legal, so it is CHECKED here rather than assumed: the pointer's own tail must equal
    the tail of the tree the encoder mirrored, or the two are not siblings and the splice
    is meaningless.
    """
    import ddm_jg2_tail_reencode as jg2

    pointer = Path(args.pointer_runtime)
    mirrored = Path(args.encoder_runtime)
    sj1.assert_carrier_is_pointer(pointer)

    pointer_member = jg2.read_archive_member(pointer / "archive.zip")
    mirrored_member = jg2.read_archive_member(mirrored / "archive.zip")
    p_sections = jg2.split_member(pointer_member)
    m_sections = jg2.split_member(mirrored_member)
    if p_sections["tail"] != m_sections["tail"]:
        raise Sj1JointError(
            "pointer tail differs from the tail the encoder mirrored "
            f"({len(p_sections['tail'])} B vs {len(m_sections['tail'])} B); the two "
            "bodies do not hold the same token stream, so this splice would be a delta "
            "against a baseline that does not exist"
        )

    stream = Path(args.stream).read_bytes()
    residual = p_sections["tail"][: jg2.RESIDUAL_COMPACT_BYTES]
    p_sections["tail"] = residual + stream

    args.out_dir.mkdir(parents=True, exist_ok=True)
    staged = args.out_dir / "staged_runtime"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(pointer, staged)
    for cache in staged.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    archive = staged / "archive.zip"
    jg2.pack_archive(jg2.join_member(p_sections), archive)
    facts = {
        "sha256": _sha256_file(archive),
        "bytes": archive.stat().st_size,
    }

    # The receiver refuses an archive whose pinned sha/size do not match its own bytes,
    # so the two pins move with the archive or the tree is dead on the contest path.
    pins = patch_inflate_pins(staged, facts["sha256"], facts["bytes"])

    # Only the archive and its two pins may differ from the pointer tree.  Anything else
    # would mean this candidate is not the pointer body plus a token stream.
    differing = sorted(
        str(f.relative_to(staged))
        for f in staged.rglob("*")
        if f.is_file()
        and (pointer / f.relative_to(staged)).is_file()
        and f.read_bytes() != (pointer / f.relative_to(staged)).read_bytes()
    )
    if differing != ["archive.zip", "inflate.py"]:
        raise Sj1JointError(
            f"staged tree differs from the pointer in {differing}, expected exactly "
            "['archive.zip', 'inflate.py']"
        )

    report = {
        "schema": "ddm_sj1_stage_tail.v1",
        "pointer_runtime": str(pointer),
        "pointer_archive_sha256": sj1.POINTER_ARCHIVE_SHA256,
        "pointer_archive_bytes": sj1.POINTER_ARCHIVE_BYTES,
        "encoder_runtime": str(mirrored),
        "tail_sibling_check": "PASS (pointer tail == mirrored tail)",
        "stream": {"path": str(args.stream), "bytes": len(stream),
                   "sha256": hashlib.sha256(stream).hexdigest()},
        "staged_runtime": str(staged),
        "staged_archive": facts,
        "delta_bytes_vs_pointer": facts["bytes"] - sj1.POINTER_ARCHIVE_BYTES,
        "inflate_pins": pins,
        "sections": {k: len(v) for k, v in p_sections.items()},
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
    }
    (args.out_dir / "STAGE_TAIL.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------------------
# close -- splice the re-solved carrier into the token-edited body and price it EXACTLY
# --------------------------------------------------------------------------------------


def cmd_close(args) -> int:
    """Build the composed candidate and price it on measured legs only.

    The pricing arithmetic is this arm's OWN, not ``jg5.run_close``'s, for one measured
    reason: jg5 carries the seg leg onto T4 with the ratio of the ANCESTOR body's legs
    (0.00030309 / 0.00030307), and this body's ratio is 0.00020139 / 0.00020132277.  The
    two differ by 2.8e-4 relative, which is 5.6e-6 in score units -- a quarter of the
    2e-5 admission bar, so using the wrong one would be a real mispricing, not a rounding
    detail ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]).  Every
    other mechanism -- ``splice.parse_shipped_body``, ``splice.build_archive`` with its
    parse-back verify, ``jg5.section_identity`` -- is jg5's and is reused verbatim.
    """
    import ddm_up3_carrier_splice as splice

    runtime = Path(args.body_runtime)
    base_bytes = (runtime / "archive.zip").read_bytes()
    observed = hashlib.sha256(base_bytes).hexdigest()
    if args.expect_body_sha256 and observed != args.expect_body_sha256:
        raise Sj1JointError(f"body sha256 {observed} != expected {args.expect_body_sha256}")
    body = splice.parse_shipped_body(runtime, verify_sha=False)
    base_codes = np.asarray(body.codes, dtype=np.int32)

    # CONTROL: splicing the body's OWN codes back in must reproduce its bytes, or a byte
    # delta cannot be attributed to the re-solve rather than to the rebuild.
    identity = splice.build_archive(
        body, base_codes, runtime_dir=runtime, container_search=True
    )
    identity_ok = identity["archive_sha256"] == observed
    if not identity_ok and not args.allow_identity_drift:
        raise Sj1JointError(
            "CONTROL FAILED: rebuilding the body from its own codes gives "
            f"{identity['archive_sha256']} ({identity['archive_size']} B), not "
            f"{observed} ({len(base_bytes)} B)"
        )

    codes = np.load(args.codes).astype(np.int32)
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Sj1JointError(f"codes have shape {codes.shape}")
    admitted = set(json.loads(Path(args.admitted_pairs).read_text()))
    candidate_codes = base_codes.copy()
    for pair in sorted(admitted):
        candidate_codes[pair] = codes[pair]

    built = splice.build_archive(
        body, candidate_codes, runtime_dir=runtime, container_search=True, verify=True
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "candidate_archive.zip"
    archive_path.write_bytes(built["archive_bytes"])
    np.save(args.out_dir / "candidate_codes.npy", candidate_codes)

    proof = jg5.section_identity(built["archive_bytes"], base_bytes, runtime=runtime)
    if not proof["frame1_sections_all_identical"]:
        raise Sj1JointError(
            f"a frame-1 section moved ({proof}); refusing to launder a seg leg through "
            "a changed odd-frame section"
        )

    d_seg_instrument = args.d_seg_instrument
    ratio_t4 = sj1.POINTER_D_SEG_T4 / args.instrument_base_d_seg
    d_seg_t4 = d_seg_instrument * ratio_t4
    d_pose = args.d_pose
    archive_bytes = int(built["archive_size"])
    score = composed_score(d_seg_t4, d_pose, archive_bytes)
    report = {
        "schema": "ddm_sj1_close.v1",
        "body_runtime": str(runtime),
        "body_archive_sha256": observed,
        "body_archive_bytes": len(base_bytes),
        "identity_control_passed": identity_ok,
        "identity_control_sha256": identity["archive_sha256"],
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": built["archive_sha256"],
            "bytes": archive_bytes,
        },
        "carrier_pairs_spliced": len(admitted),
        "carrier_coordinates_changed": int((candidate_codes != base_codes).sum()),
        "frame1_section_identity": proof,
        "d_seg_instrument": d_seg_instrument,
        "instrument_base_d_seg": args.instrument_base_d_seg,
        "t4_ratio_applied": ratio_t4,
        "d_seg_t4_projected": d_seg_t4,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "score_projected": score,
        "pointer_score_t4": sj1.POINTER_SCORE_T4,
        "pointer_archive_sha256": sj1.POINTER_ARCHIVE_SHA256,
        "net_dS_vs_pointer": score - sj1.POINTER_SCORE_T4,
        "lineage_scores": {
            "fs2_base": sj1.BASE_SCORE_T4,
            "cl2_projected": sj1.CL2_PROJECTED_SCORE,
        },
        "admit_bar": -2e-05,
        "clears_admit_bar_vs_pointer": bool(score - sj1.POINTER_SCORE_T4 <= -2e-05),
        "axis": (
            "seg from the jg1 DALI instrument carried onto T4 by the SAME-instrument "
            "ratio; pose [macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]; bytes "
            "EXACT"
        ),
        "score_claim": False,
        "promotable": False,
    }
    (args.out_dir / "CLOSE.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_parseback(args) -> int:
    """Decode the candidate through the RECEIVER TREE'S OWN inflate path, on CPU.

    This is ``ddm_cl2.stage_parseback``'s contract on this arm's tree: build the RC64
    backend and the native f26 corrector exactly as ``inflate.sh`` does, run the tree's
    own ``_verify_input`` pin check, then ``runtime.f26_inflate.inflate_archive`` with
    ``device_name="cpu"``.  It proves the candidate is decodable by the bytes that ship
    with it AND produces the ``0.raw`` the seg leg must finally be measured on -- a
    distortion claim read off the encoder's own field rather than the receiver's render
    would be measuring the wrong object.
    """
    import importlib
    import importlib.util
    import os
    import subprocess
    import zipfile

    runtime = Path(args.runtime)
    archive = runtime / "archive.zip"
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        (data_dir / "p").write_bytes(zf.read("p"))
    build = root / "build"
    build.mkdir(exist_ok=True)
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        raise Sj1JointError("a C compiler is required for the RC64 backend (inflate.sh contract)")
    rc64_so = build / "rc64_backend.so"
    subprocess.run(
        [cc, "-O3", "-std=c11", "-shared", "-fPIC",
         str(runtime / "runtime/entropy/rc64_backend.c"), "-o", str(rc64_so)],
        check=True,
    )
    os.environ["CPR1_RC64_LIBRARY"] = str(rc64_so)
    corrector_so = build / "f26_corrector_native.so"
    subprocess.run(
        [cc, "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
         str(runtime / "runtime/f26_corrector_native.c"), "-lm", "-o", str(corrector_so)],
        check=True,
    )
    os.environ["F26_CORRECTOR_NATIVE_LIBRARY"] = str(corrector_so)
    os.environ.setdefault("F26_TOKEN_DECODER", "python")

    spec = importlib.util.spec_from_file_location("sj1_receiver_inflate", runtime / "inflate.py")
    if spec is None or spec.loader is None:
        raise Sj1JointError("cannot import the candidate tree's inflate.py")
    sys.path.insert(0, str(runtime))
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    receiver._verify_input(data_dir, archive)
    f26 = importlib.import_module("runtime.f26_inflate")
    if Path(f26.__file__).resolve() != (runtime / "runtime" / "f26_inflate.py").resolve():
        raise Sj1JointError("parse-back imported a runtime outside the candidate tree")

    destination = root / "0.raw"
    started = time.perf_counter()
    report = f26.inflate_archive(
        archive, destination, renderer_dir=runtime / "cpr1", device_name="cpu",
        num_threads=args.threads, checkpoint_dir=root / ".f26_decode_checkpoints",
    )
    elapsed = time.perf_counter() - started

    decoded_path = root / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.u8"
    field_identity = None
    if args.expect_field and decoded_path.is_file():
        expect = load_edit_planes(args.expect_field)
        base = jg1.load_tokens(sj1.BODY_TOKENS)
        want = np.array(base, dtype=np.uint8)
        for pair, plane in expect.items():
            want[pair] = plane
        got = np.fromfile(decoded_path, dtype=np.uint8).reshape(want.shape)
        field_identity = bool(np.array_equal(got, want))

    result = {
        "schema": "ddm_sj1_parseback.v1",
        "runtime": str(runtime),
        "archive": {"path": str(archive), "sha256": _sha256_file(archive),
                    "bytes": archive.stat().st_size},
        "pin_check": "PASS (candidate tree's own inflate.py _verify_input)",
        "rendered_raw": {"path": str(destination), "sha256": _sha256_file(destination),
                         "bytes": destination.stat().st_size},
        "decoded_field_matches_admitted": field_identity,
        "wall_clock_seconds": elapsed,
        "device": "cpu",
        "libraries": {"CPR1_RC64_LIBRARY": str(rc64_so),
                      "F26_CORRECTOR_NATIVE_LIBRARY": str(corrector_so)},
        "inflate_report": report,
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
    }
    (root / "PARSEBACK_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "inflate_report"},
                     indent=2, sort_keys=True, default=str))
    if field_identity is False:
        raise Sj1JointError("decoded token field does not match the admitted field")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render-edits", help="render the candidate's odd frames")
    render.add_argument("--field", type=Path, required=True)
    render.add_argument("--out-dir", type=Path, required=True)
    render.add_argument("--threads", type=int, default=4)
    render.add_argument("--progress", action="store_true")
    render.add_argument("--no-verify-shas", action="store_true")
    render.set_defaults(func=cmd_render_edits)

    pose = sub.add_parser("pose", help="per-pair d_pose over all 600 pairs")
    pose.add_argument("--overlay", type=Path, default=None)
    pose.add_argument("--codes", type=Path, default=None)
    pose.add_argument("--tag", required=True)
    pose.add_argument("--out", type=Path, required=True)
    pose.add_argument("--batch-size", type=int, default=8)
    pose.add_argument("--threads", type=int, default=4)
    pose.set_defaults(func=cmd_pose)

    refine = sub.add_parser("refine", help="carrier re-solve on the candidate's renders")
    refine.add_argument("--overlay", type=Path, required=True)
    refine.add_argument("--field", type=Path, required=True)
    refine.add_argument("--base-pose", type=Path, required=True)
    refine.add_argument("--out-dir", type=Path, required=True)
    refine.add_argument("--shard-index", type=int, default=0)
    refine.add_argument("--shard-count", type=int, default=1)
    refine.add_argument("--outer-rounds", type=int, default=40)
    refine.add_argument("--max-gn-iterations", type=int, default=400)
    refine.add_argument("--threads", type=int, default=2)
    refine.add_argument("--resume", action="store_true")
    refine.add_argument("--progress", action="store_true")
    refine.set_defaults(func=cmd_refine)

    codes = sub.add_parser("codes", help="merge refine shards into a (600,12) table")
    codes.add_argument("--rows", nargs="+", required=True)
    codes.add_argument("--out", type=Path, required=True)
    codes.set_defaults(func=cmd_codes)

    admit = sub.add_parser("admit", help="Lagrange sweep over pose damage")
    admit.add_argument("--pass-rows", type=Path, required=True)
    admit.add_argument("--field", type=Path, required=True)
    admit.add_argument(
        "--carry-field",
        type=Path,
        default=None,
        help="the LIVE ROW's field; pairs the sweep drops revert to THIS, not to the "
        "original base. Omitting it on a successor round silently undoes banked edits.",
    )
    admit.add_argument("--base-pose", type=Path, required=True)
    admit.add_argument("--stale-pose", type=Path, required=True)
    admit.add_argument("--resolved-pose", type=Path, required=True)
    admit.add_argument("--out-dir", type=Path, required=True)
    admit.add_argument(
        "--base-archive-bytes",
        type=float,
        default=float(sj1.POINTER_ARCHIVE_BYTES),
        help="archive the edits are priced ON TOP OF; defaults to the LIVE pointer (rc1)",
    )
    admit.add_argument(
        "--instrument-base-d-seg",
        type=float,
        required=True,
        help="this instrument's OWN base d_seg, so the T4 carry is same-instrument",
    )
    admit.add_argument(
        "--bits-control",
        type=Path,
        default=None,
        help="per-frame bit ledger of the UNEDITED field from the real control encode",
    )
    admit.add_argument(
        "--bits-candidate",
        type=Path,
        default=None,
        help="per-frame bit ledger of the EDITED field from the real encode; with "
        "--bits-control this makes the per-pair rate leg MEASURED, not apportioned",
    )
    admit.add_argument(
        "--modelled-bytes-per-changed-token",
        type=float,
        default=0.0,
        help="fallback only, used when the bit ledgers are absent",
    )
    admit.set_defaults(func=cmd_admit)

    stage = sub.add_parser(
        "stage-tail", help="splice the re-encoded token stream into the pointer's member"
    )
    stage.add_argument("--pointer-runtime", type=Path, default=sj1.POINTER_TREE)
    stage.add_argument("--encoder-runtime", type=Path, required=True)
    stage.add_argument("--stream", type=Path, required=True)
    stage.add_argument("--out-dir", type=Path, required=True)
    stage.set_defaults(func=cmd_stage_tail)

    pb = sub.add_parser("parseback", help="decode the candidate through its own receiver")
    pb.add_argument("--runtime", type=Path, required=True)
    pb.add_argument("--out-dir", type=Path, required=True)
    pb.add_argument("--expect-field", type=Path, default=None)
    pb.add_argument("--threads", type=int, default=4)
    pb.set_defaults(func=cmd_parseback)

    close = sub.add_parser("close", help="splice the re-solved carrier and price exactly")
    close.add_argument("--body-runtime", type=Path, required=True)
    close.add_argument("--expect-body-sha256", default=None)
    close.add_argument("--codes", type=Path, required=True)
    close.add_argument("--admitted-pairs", type=Path, required=True)
    close.add_argument("--out-dir", type=Path, required=True)
    close.add_argument("--allow-identity-drift", action="store_true")
    close.add_argument("--d-seg-instrument", type=float, required=True)
    close.add_argument("--instrument-base-d-seg", type=float, required=True)
    close.add_argument("--d-pose", type=float, required=True)
    close.set_defaults(func=cmd_close)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
