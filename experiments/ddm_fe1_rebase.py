#!/usr/bin/env python3
"""ddm_fe1 RE-BASE: carry this arm's admitted frame-embedding edit onto a new pointer.

One command.  Given a new pointer tree it re-verifies every admitted pair against the
NEW body and re-prices the whole candidate, dropping anything the move no longer buys.

Why a re-base is not free here, and what exactly has to be redone:

* If the new pointer changed the SEMANTIC section, the renderer weights changed and this
  arm's entire n600 search is void -- the script REFUSES rather than pretend otherwise.
* If it changed the TOKEN TAIL (sj1's passes do), the rendered frames of the affected
  pairs changed, so each admitted pair's realized flip gain must be RE-MEASURED on the
  new field; a pair whose gain vanished is dropped.
* If it changed the CARRIER (pc2's move did), the coefficient parametrisation changed, so
  every re-solve must START from the new tree's coefficients and basis scales.
* The rate leg must be re-priced by a REAL encode on the NEW archive, with the container
  search re-derived there -- the shipped brotli shape does not automatically survive a
  move (measured: (ck2,11,16) and (ck2,11,24) can tie on LENGTH and differ in BYTES).

Stages 1-5 run here.  Stage 6 (stage / public smoke / parse-back / seal) is printed as
the exact commands to run, because each is an existing surface and none should be
re-implemented inside a re-base script.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_fe1_admit_and_build as ab
import ddm_fe1_frame_embedding_search as fe1
import ddm_fe1_pose_price as price
import ddm_jg1_seg_solve as jg1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_up3_carrier_splice as up3

CELL_COUNT = ab.CELL_COUNT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sections(archive: Path, tree: Path) -> dict[str, Any]:
    """Split an archive into its RX1 sections using THAT tree's own reader."""
    reader, _cr, _ar1, _cp = up3._import_runtime(tree)
    outer = zipfile.ZipFile(io.BytesIO(archive.read_bytes())).read("p")
    header = reader.RX1_MODEL_HEADER.unpack_from(outer)
    size = reader.RX1_MODEL_HEADER.size
    hpac, semantic, carrier = header[5], header[6], header[7]
    return {
        "reserved": header[4],
        "hpac": outer[size : size + hpac],
        "semantic": outer[size + hpac : size + hpac + semantic],
        "carrier": outer[size + hpac + semantic : size + hpac + semantic + carrier],
        "tail": outer[size + hpac + semantic + carrier :],
    }


def load_field(path: Path) -> np.ndarray:
    """The new pointer's token field, from an npz of per-pair planes or a raw u8 dump."""
    if path.suffix == ".npz":
        with np.load(path) as blob:
            planes = []
            for pair in range(fe1.N_PAIRS):
                key = str(pair)
                if key not in blob:
                    raise fe1.Fe1Error(f"token field {path} has no plane for pair {pair}")
                planes.append(blob[key])
        field = np.stack(planes)
    else:
        field = np.fromfile(path, dtype=np.uint8).reshape(
            fe1.N_PAIRS, fe1.EVAL_H, fe1.EVAL_W
        )
    if field.shape != (fe1.N_PAIRS, fe1.EVAL_H, fe1.EVAL_W) or field.dtype != np.uint8:
        raise fe1.Fe1Error(f"token field {path} has shape {field.shape}/{field.dtype}")
    return np.ascontiguousarray(field)


def base_argmax_for(pairs: Sequence[int], args) -> dict[int, np.ndarray]:
    """The NEW base's realized argmax for the pairs we care about.

    Read from the new pointer's own parse-back -- either its retained argmax table or its
    ``0.raw``.  A re-base that compared against the OLD base's flips would credit this arm
    for cells the new pointer already repaired, which is the whole reason this exists.
    """
    if args.base_argmax:
        table = np.load(args.base_argmax, mmap_mode="r")
        if table.shape != (fe1.N_PAIRS, fe1.EVAL_H, fe1.EVAL_W):
            raise fe1.Fe1Error(f"base argmax has shape {table.shape}")
        return {int(p): np.asarray(table[int(p)]) for p in pairs}
    if not args.base_raw:
        raise fe1.Fe1Error(
            "the new base's realized argmax is REQUIRED: pass --base-argmax (an "
            "n600 argmax table) or --base-raw (the new pointer's own 0.raw). Comparing "
            "against the old base would credit this arm for the new pointer's repairs."
        )
    raw = fe1._open_raw(Path(args.base_raw))
    net = jg1.load_segnet()
    out: dict[int, np.ndarray] = {}
    for pair in pairs:
        frame = np.asarray(raw[2 * int(pair) + 1])[None]
        out[int(pair)] = jg1.argmax_from_camera_frames(net, frame)[0]
    return out


def cmd_rebase(args) -> int:
    fe1._set_threads(args.threads)
    tree = Path(args.pointer_tree).resolve()
    archive = tree / "archive.zip"
    if not archive.is_file():
        raise fe1.Fe1Error(f"no archive at {archive}")
    new_bytes = archive.stat().st_size
    new_sha = _sha256(archive)

    # ---- stage 1: what moved? -------------------------------------------------
    old = split_sections(fe1.LIVE_ARCHIVE, fe1.LIVE_TREE)
    new = split_sections(archive, tree)
    moved = {k: (len(old[k]), len(new[k]), old[k] == new[k]) for k in ("hpac", "semantic", "carrier", "tail")}
    for name, (was, now, same) in moved.items():
        print(f"  {name:>9}: {was} -> {now}  identical={same}")
    if not moved["semantic"][2]:
        raise fe1.Fe1Error(
            "the new pointer changed the SEMANTIC section: the renderer weights this "
            "arm searched against are gone, so the n600 search is VOID and this is a "
            "re-run, not a re-base. Refusing to carry an edit onto a different renderer."
        )
    tail_moved = not moved["tail"][2]
    carrier_moved = not moved["carrier"][2]

    # ---- stage 2: the admitted set, and the new field --------------------------
    admitted = json.loads(Path(args.admitted).read_text())
    best = admitted.get("best", admitted)
    pairs = [int(p) for p in best["pairs_included"]]
    rows = {
        int(r["pair"]): r
        for r in (
            json.loads(line)
            for line in Path(args.pose_rows).read_text().splitlines()
            if line.strip()
        )
        if int(r["pair"]) in set(pairs)
    }
    if set(rows) != set(pairs):
        raise fe1.Fe1Error("pose rows do not cover every admitted pair")

    body = fe1.load_body(with_raw=False, verify_shas=False)
    if tail_moved:
        if not args.token_field:
            raise fe1.Fe1Error(
                "the new pointer changed the TOKEN TAIL, so the rendered frames of the "
                "affected pairs changed: --token-field is REQUIRED (the new pointer's "
                "own decoded field, npz of per-pair planes or a u8 dump)"
            )
        body.tokens = load_field(Path(args.token_field))
        print(f"  token field replaced from {args.token_field}")

    # ---- stage 3: re-verify each admitted pair's realized gain ------------------
    base_argmax = base_argmax_for(pairs, args)
    survivors: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for pair in pairs:
        row = rows[pair]
        gt = body.gt[pair]
        base_flips = int((base_argmax[pair] != gt).sum())
        fe1.set_pair_codes(body, pair, row["final_row"])
        moved_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        fe1.restore_pair_codes(body, pair)
        cells = base_flips - moved_flips
        entry = dict(row)
        entry.update(
            {
                "base_flips_new": base_flips,
                "moved_flips_new": moved_flips,
                "cells_new": cells,
                "cells_old": int(row["cells"]),
            }
        )
        print(
            f"  pair {pair:>3}: base {base_flips:>3} -> {moved_flips:>3} "
            f"({cells:+d} cells; was {int(row['cells']):+d})"
            + ("" if cells > 0 else "   DROPPED"),
            flush=True,
        )
        (survivors if cells > 0 else dropped).append(entry)
    if not survivors:
        print("no admitted pair survives the re-base; nothing to seal")
        return 3

    # ---- stage 4: re-solve the carrier from the NEW tree's coefficients ---------
    raw = fe1._open_raw(Path(args.base_raw)) if args.base_raw else fe1._open_raw(fe1.LIVE_RAW)
    frames = {}
    for entry in survivors:
        pair = int(entry["pair"])
        fe1.set_pair_codes(body, pair, entry["final_row"])
        frames[pair] = fe1.render_pair(body, pair)[0]
        fe1.restore_pair_codes(body, pair)
    overlay = price._MemoryOverlayRaw(raw, frames)
    base_inst = _instrument(raw, tree)
    moved_inst = _instrument(overlay, tree)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    threshold = jg5.materiality_dd_threshold(args.base_mean_d_pose)
    for entry in survivors:
        pair = int(entry["pair"])
        entry["d_pose_base"] = float(
            br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0]
        )
        control = jg5.refine_pair(
            base_inst, pair, live_codes[pair], dd_threshold=threshold
        )
        entry["d_pose_base_resolved"] = float(control["final_d_pose"])
        entry["d_pose_stale"] = float(
            br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0]
        )
        refined = jg5.refine_pair(
            moved_inst, pair, live_codes[pair], dd_threshold=threshold
        )
        entry["d_pose_resolved"] = float(refined["final_d_pose"])
        entry["resolved_codes"] = [int(c) for c in refined["codes"]]
        print(
            f"  pair {pair:>3}: d_pose {entry['d_pose_base']:.4e} (control "
            f"{entry['d_pose_base_resolved']:.4e}) -> stale {entry['d_pose_stale']:.4e} "
            f"-> resolved {entry['d_pose_resolved']:.4e}",
            flush=True,
        )

    # ---- stage 5: re-price by REAL encode on the NEW archive --------------------
    section = fe1.load_semantic_section(archive_path=archive, runtime_dir=tree / "runtime")
    shipped = up3.parse_shipped_body(tree, verify_sha=False)
    new_carrier = np.asarray(shipped.codes, dtype=np.int32)
    control_build = ab.build_candidate_archive(
        section, section.codes.astype(np.int64), new_carrier, tree_dir=tree, verify=True
    )
    if control_build["archive_sha256"] != new_sha:
        raise fe1.Fe1Error(
            "the null-build identity control FAILED on the new pointer: rebuilding its "
            f"own codes gives {control_build['archive_sha256']}, not {new_sha}. The "
            "container shape must be re-derived on THIS archive before anything is priced."
        )
    print(f"  null build on the new pointer: {control_build['archive_size']} B, byte-identical")

    base_pose = float(args.base_mean_d_pose)
    leg0 = math.sqrt(10.0 * base_pose)
    for entry in survivors:
        delta = entry["d_pose_resolved"] - entry["d_pose_base"]
        entry["dS_seg"] = -entry["cells_new"] * 100.0 / CELL_COUNT
        entry["dS_pose"] = math.sqrt(10.0 * (base_pose + delta / fe1.N_PAIRS)) - leg0
        entry["value"] = entry["dS_seg"] + entry["dS_pose"]
    ranked = sorted(survivors, key=lambda e: e["value"])

    results = []
    best_cut = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for count in range(1, len(ranked) + 1):
        subset = ranked[:count]
        codes = section.codes.astype(np.int64).copy()
        carrier = new_carrier.copy()
        for entry in subset:
            codes[int(entry["pair"])] = np.asarray(entry["final_row"], dtype=np.int64)
            carrier[int(entry["pair"])] = np.asarray(entry["resolved_codes"], dtype=np.int32)
        built = ab.build_candidate_archive(
            section, codes, carrier, tree_dir=tree, verify=True
        )
        cells = sum(int(e["cells_new"]) for e in subset)
        pose_mean = base_pose + sum(
            e["d_pose_resolved"] - e["d_pose_base"] for e in subset
        ) / fe1.N_PAIRS
        d_seg_new = (args.base_cells - cells) / CELL_COUNT
        dS = (
            100.0 * (d_seg_new - args.base_cells / CELL_COUNT)
            + (math.sqrt(10.0 * pose_mean) - leg0)
            + (built["archive_size"] - new_bytes) * fe1.RATE_PER_BYTE
        )
        entry = {
            "pairs": count,
            "cells": cells,
            "archive_bytes": built["archive_size"],
            "archive_sha256": built["archive_sha256"],
            "d_archive_bytes": built["archive_size"] - new_bytes,
            "semantic_container": built["semantic_container"],
            "d_seg_local": d_seg_new,
            "d_pose_mean": pose_mean,
            "dS_total": dS,
            "admits": bool(dS < ab.ADMIT_BAR),
            "pairs_included": sorted(int(e["pair"]) for e in subset),
        }
        results.append(entry)
        print(
            f"  cut {count:>3}: {built['archive_size']} B ({entry['d_archive_bytes']:+d}) "
            f"-> dS {dS:+.6e}{'  ADMITS' if entry['admits'] else ''}",
            flush=True,
        )
        if best_cut is None or dS < best_cut["dS_total"]:
            best_cut = entry
            (out_dir / "archive.zip").write_bytes(built["archive_bytes"])
            np.save(out_dir / "frame_embed_codes.npy", codes.astype(np.int8))
            np.save(out_dir / "carrier_codes.npy", carrier)

    report = {
        "schema": "ddm_fe1_rebase.v1",
        "axis": (
            "d_seg [macOS-CPU advisory, jg1/sj1 instrument, DALI GT]; d_pose "
            "[cpu_torch fp32]; bytes EXACT through real archive builds on the NEW tree"
        ),
        "score_claim": False,
        "new_pointer": {"tree": str(tree), "bytes": new_bytes, "sha256": new_sha},
        "sections_moved": {k: {"was": v[0], "now": v[1], "identical": v[2]} for k, v in moved.items()},
        "token_field": str(args.token_field) if args.token_field else None,
        "base_cells": args.base_cells,
        "base_mean_d_pose": base_pose,
        "survivors": survivors,
        "dropped": dropped,
        "cuts": results,
        "best": best_cut,
        "null_build_identity": True,
        "carrier_re_solved_from": "the new tree's own coefficients and basis scales",
        "carrier_moved": carrier_moved,
        "tail_moved": tail_moved,
    }
    (out_dir / "REBASE.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"best": best_cut and {k: best_cut[k] for k in ("pairs", "cells", "archive_bytes", "dS_total", "admits")}}, indent=1))
    print(
        "\nSTAGE 6 -- run these, in order:\n"
        f"  .venv/bin/python experiments/ddm_fe1_admit_and_build.py stage "
        f"--archive {out_dir}/archive.zip --out-dir {out_dir}/candidate_runtime\n"
        f"  .venv/bin/python experiments/ddm_sj1_joint_admission.py public-smoke "
        f"--candidate-runtime {out_dir}/candidate_runtime --frontier-runtime {tree} "
        f"--out {out_dir}/PUBLIC_SMOKE.json\n"
        f"  .venv/bin/python experiments/ddm_sj1_joint_admission.py parseback "
        f"--runtime {out_dir}/candidate_runtime --out-dir <APDataStore>/parseback --threads 4\n"
        f"  .venv/bin/python experiments/ddm_fe1_frame_embedding_search.py step0 "
        f"--raw <APDataStore>/parseback/0.raw --threads 3 --out {out_dir}/STEP0.json\n"
        f"  .venv/bin/python tools/make_candidate_seal.py --candidate-id "
        f"ddm_fe1_frame_embedding_predistortion --runtime-dir {out_dir}/candidate_runtime "
        f"--axis contest_cuda --public-entrypoint-smoke {out_dir}/PUBLIC_SMOKE.json "
        f"--admit-bar-net-ds -2e-05 --out {out_dir}/SEAL.json ...\n"
    )
    return 0


def _instrument(raw, tree: Path):
    """A pose instrument bound to a GIVEN tree's carrier state, not to fe1's live one."""
    import ddm_up2_shipping_pose_solve as up2

    state = up2.load_carrier_state(tree, verify_archive=False)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise fe1.Fe1Error(f"GT pose lineage is {lineage}")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer-tree", required=True, help="the NEW pointer's runtime tree root")
    parser.add_argument("--token-field", default="", help="the new pointer's decoded token field (npz or u8); REQUIRED if the tail moved")
    parser.add_argument("--base-argmax", default="", help="the new base's n600 argmax table")
    parser.add_argument("--base-raw", default="", help="the new base's own 0.raw, if no argmax table")
    parser.add_argument("--base-cells", type=int, required=True, help="the new base's flipped-cell count")
    parser.add_argument("--base-mean-d-pose", type=float, required=True, help="the new base's n600 mean d_pose")
    parser.add_argument("--admitted", default=str(fe1.WORK / "candidate/FINE_SWEEP.json"))
    parser.add_argument("--pose-rows", default=str(fe1.WORK / "admission/pose/pose_rows.jsonl"))
    parser.add_argument("--out-dir", default=str(fe1.WORK / "rebase"))
    parser.add_argument("--threads", type=int, default=4)
    parser.set_defaults(func=cmd_rebase)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
