#!/usr/bin/env python3
"""ddm_rp1 pose leg -- base / stale / resolved on the MOVE-37 pointer's own configuration.

WHY THIS EXISTS RATHER THAN CALLING sj1's CHAIN
-----------------------------------------------
``ddm_sj1_joint_admission`` binds its pose instrument to sj1's own ``POINTER_LINEAGE``
through ``assert_carrier_is_pointer``, and that lineage ends at move 35.  The pointer is now
move 37 (cmp2, sha ``670d38d0…``).  sj1's guard would REFUSE -- correctly -- and its files are
read-only to this arm, so the instrument is rebuilt here against the live tree from the same
primitives (``up2`` for the carrier state and the pose measurement, ``br1`` for the low
basis, ``jg5`` for ``refine_pair``).

THE ONE THING THAT IS NOT BORROWED: THE BASE
--------------------------------------------
The pose base is MEASURED here, on the pointer's own configuration, through this arm's own
overlay -- never lifted from a sister arm's receipt.  A base measured without the overlay
reads ~500x off (the 2026-09-09 incident that put the pose-base law in memory), and a base
lifted from another arm's tree silently prices this arm's damage against a different body.
So ``base`` renders the LIVE field's own odd frames and measures them exactly as ``stale``
and ``resolved`` are measured; base and candidate then differ by this arm's token edits and
by nothing else.

CARRIER COMPOSITION, VERIFIED NOT ASSUMED
------------------------------------------
Moves 36 and 37 changed the hpac/tail coder and the semantic section's coding.  The CARRIER
section has been byte-identical at 18,586 B since sj1's pass 4, so the coefficients this
module re-solves from are the ones the pointer ships -- and that identity is CHECKED at
load, not remembered.

``[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1  # noqa: E402
import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_jg2_tail_reencode as jg2  # noqa: E402
import ddm_jg5_pose_resolve_on_edited_renders as jg5  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402
import ddm_sj1_joint_admission as sj1ja  # noqa: E402  (READ-ONLY: overlay helpers)
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

N_PAIRS = jg1.N_PAIRS
CAMERA_H, CAMERA_W = jg1.CAMERA_H, jg1.CAMERA_W

#: The live pointer (move 37, cmp2).  Re-read from the pointer file on every entry point.
POINTER_TREE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime"
)
POINTER_ARCHIVE_SHA256 = (
    "670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc"
)
#: The carrier section, byte-identical from sj1 pass 4 through cmp1 and cmp2.
CARRIER_BYTES = 18_586
#: The receiver decode this arm overlays onto (cl2's own parse-back, as sj1 uses).
BODY_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder"
    "/parseback/lambda_1p0/0.raw"
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_pointer_and_carrier() -> dict[str, Any]:
    """Refuse unless the live pointer is the tree this module solves against."""
    live = rp1.verify_pointer(expect_sha=POINTER_ARCHIVE_SHA256)
    if not live["matches_expected"]:
        raise rp1.Rp1Error(
            f"pointer moved: file says {live['archive_sha256']}, this module is pinned to "
            f"{POINTER_ARCHIVE_SHA256}; re-base before solving a carrier"
        )
    sections = jg2.split_member(jg2.read_archive_member(POINTER_TREE / "archive.zip"))
    if len(sections["carrier"]) != CARRIER_BYTES:
        raise rp1.Rp1Error(
            f"pointer carrier is {len(sections['carrier'])} B, expected {CARRIER_BYTES}; "
            "the coefficients this arm re-solves from are not the ones it ships"
        )
    return {
        "pointer": live,
        "carrier_bytes": len(sections["carrier"]),
        "tail_bytes": len(sections["tail"]),
        "semantic_bytes": len(sections["semantic"]),
    }


def load_instrument(overlay_dir: Path | None):
    """br1's pose instrument on the LIVE pointer's carrier, reading the asked-for decode."""
    state = up2.load_carrier_state(POINTER_TREE, verify_archive=False)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise rp1.Rp1Error(f"GT pose lineage is {lineage}, not {up2.LINEAGE_DALI}")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    if overlay_dir is None:
        raw = np.memmap(
            BODY_RAW, dtype=np.uint8, mode="r",
            shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3),
        )
    else:
        raw = sj1ja.open_overlay(BODY_RAW, overlay_dir)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


def cmd_render(args: argparse.Namespace) -> int:
    """Render frame 2p+1 for every plane in a 600-plane field, at the receiver's batch 1."""
    rp1.set_threads(args.threads)
    planes = sj1ja.load_edit_planes(Path(args.field))
    semantic = jg1.load_semantic_renderer(
        archive_path=rp1.ENCODER_TREE / "archive.zip",
        runtime_dir=rp1.ENCODER_TREE / "runtime",
    )
    pairs = sorted(planes)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "odd_frames.u8"
    overlay = np.memmap(
        path, dtype=np.uint8, mode="w+", shape=(len(pairs), CAMERA_H, CAMERA_W, 3)
    )
    started = time.time()
    for slot, pair in enumerate(pairs):
        overlay[slot] = jg1.render_frame1(
            semantic, planes[pair][None], np.array([pair])
        )[0]
        if (slot + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "rendered": slot + 1,
                        "of": len(pairs),
                        "s_per_pair": (time.time() - started) / (slot + 1),
                    }
                ),
                flush=True,
            )
    overlay.flush()
    del overlay
    manifest = {
        "schema": "ddm_sj1_overlay.v1",
        "pairs": pairs,
        "field_npz": str(args.field),
        "field_npz_sha256": _sha256_file(Path(args.field)),
        "odd_frames_path": str(path),
        "odd_frames_sha256": _sha256_file(path),
        "odd_frames_bytes": path.stat().st_size,
        "semantic_batch": 1,
        "elapsed_seconds": time.time() - started,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }
    (out / "OVERLAY.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"pairs": len(pairs), "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0


def cmd_pose(args: argparse.Namespace) -> int:
    """Per-pair d_pose over all 600 pairs on one decode with one set of codes."""
    rp1.set_threads(args.threads)
    receipts = assert_pointer_and_carrier()
    inst = load_instrument(Path(args.overlay) if args.overlay else None)
    codes = (
        np.load(args.codes).astype(np.int32)
        if args.codes
        else np.asarray(inst.state.codes, dtype=np.int32)
    )
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise rp1.Rp1Error(f"codes have shape {codes.shape}")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    started = time.time()
    per_pair, _ = up2.measure_pose(
        inst.posenet,
        inst.state,
        coefficients,
        inst.raw,
        inst.targets,
        np.arange(N_PAIRS, dtype=np.int64),
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, per_pair)
    report = {
        "schema": "ddm_rp1_pose_leg.v1",
        "tag": args.tag,
        "decode": "candidate_overlay" if args.overlay else "shipped_decode",
        "overlay_dir": str(args.overlay) if args.overlay else None,
        "codes_source": str(args.codes) if args.codes else "pointer_carrier",
        "d_pose_mean": float(per_pair.mean()),
        "pose_leg": jg5.pose_leg(float(per_pair.mean())),
        "per_pair_path": str(out),
        "per_pair_sha256": _sha256_file(out),
        "elapsed_seconds": time.time() - started,
        "receipts": receipts,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    (out.parent / f"POSE_{args.tag}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps({k: report[k] for k in ("tag", "d_pose_mean", "pose_leg", "elapsed_seconds")}))
    return 0


def cmd_refine(args: argparse.Namespace) -> int:
    """Re-solve the carrier on the candidate's renders, for the CHANGED pairs only."""
    rp1.set_threads(args.threads)
    assert_pointer_and_carrier()
    inst = load_instrument(Path(args.overlay))
    changed = [int(p) for p in json.loads(Path(args.changed_pairs).read_text())]
    shard = [p for i, p in enumerate(sorted(changed)) if i % args.shard_count == args.shard_index]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"refine_rows_{args.shard_index}.jsonl"
    done: set[int] = set()
    if rows_path.is_file() and args.resume:
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))
    base_pose = np.load(args.base_pose)
    if base_pose.shape != (N_PAIRS,):
        raise rp1.Rp1Error(f"base pose vector has shape {base_pose.shape}")
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
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "done": count + 1,
                    "of": len(shard),
                    "pair": pair,
                    "start_d_pose": row["start_d_pose"],
                    "final_d_pose": row["final_d_pose"],
                    "s_per_pair": (time.time() - started) / (count + 1),
                }
            ),
            flush=True,
        )
    (out / f"REFINE_SHARD_{args.shard_index}.json").write_text(
        json.dumps(
            {
                "schema": "ddm_rp1_refine_shard.v1",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "pairs": shard,
                "rows_path": str(rows_path),
                "dd_threshold": dd_threshold,
                "elapsed_seconds": time.time() - started,
                "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_codes(args: argparse.Namespace) -> int:
    """Merge refine shards into a (600, 12) code table; untouched pairs keep pointer codes."""
    receipts = assert_pointer_and_carrier()
    state = up2.load_carrier_state(POINTER_TREE, verify_archive=False)
    codes = np.asarray(state.codes, dtype=np.int32).copy()
    merged = 0
    kept = 0
    for path in args.rows:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("final_d_pose") is None:
                continue
            # Only ADOPT a re-solve that actually improved the pair; a solve that came
            # back worse is a solve, not an improvement, and adopting it would price this
            # arm's pose leg against the solver's noise.
            if float(row["final_d_pose"]) <= float(row["start_d_pose"]):
                codes[int(row["pair"])] = np.asarray(row["codes"], dtype=np.int32)
                merged += 1
            else:
                kept += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, codes)
    print(
        json.dumps(
            {
                "codes_path": str(out),
                "pairs_resolved": merged,
                "pairs_kept_at_pointer_codes": kept,
                "receipts": receipts,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render a 600-plane field's odd frames")
    render.add_argument("--field", required=True)
    render.add_argument("--out-dir", required=True)
    render.add_argument("--threads", type=int, default=3)
    render.set_defaults(func=cmd_render)

    pose = sub.add_parser("pose", help="per-pair d_pose over all 600 pairs")
    pose.add_argument("--overlay", default=None)
    pose.add_argument("--codes", default=None)
    pose.add_argument("--tag", required=True)
    pose.add_argument("--out", required=True)
    pose.add_argument("--batch-size", type=int, default=8)
    pose.add_argument("--threads", type=int, default=4)
    pose.set_defaults(func=cmd_pose)

    refine = sub.add_parser("refine", help="carrier re-solve on the changed pairs")
    refine.add_argument("--overlay", required=True)
    refine.add_argument("--changed-pairs", required=True)
    refine.add_argument("--base-pose", required=True)
    refine.add_argument("--out-dir", required=True)
    refine.add_argument("--shard-index", type=int, default=0)
    refine.add_argument("--shard-count", type=int, default=1)
    refine.add_argument("--outer-rounds", type=int, default=40)
    refine.add_argument("--max-gn-iterations", type=int, default=400)
    refine.add_argument("--threads", type=int, default=2)
    refine.add_argument("--resume", action="store_true")
    refine.set_defaults(func=cmd_refine)

    codes = sub.add_parser("codes", help="merge refine shards into a (600,12) table")
    codes.add_argument("--rows", nargs="+", required=True)
    codes.add_argument("--out", required=True)
    codes.set_defaults(func=cmd_codes)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
