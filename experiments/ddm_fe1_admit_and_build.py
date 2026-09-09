#!/usr/bin/env python3
"""ddm_fe1 admission and archive build for per-pair frame-embedding moves.

The search leaves a per-pair ledger of realized code moves.  This turns that ledger into
an archive, with the same discipline sj1 used on the token tail:

* every candidate pair is re-rendered, its pose measured STALE, its carrier re-solved
  from the LIVE coefficients, and its pose measured RESOLVED -- per pair, retained;
* the admitted subset is chosen by a sweep and **priced by a REAL archive build**, never
  by summing a per-pair bit ledger (sj1's pass-3 seal measured that sum under-charging
  by 19.6 B, and here the semantic section's cost is not even additive: it is a fixed
  container-break fee, so a ledger sum would be badly wrong in both directions);
* the container shape (brotli quality, window, CK2 interleave) is SEARCHED, because none
  of it is transmitted -- the receiver reads a self-describing brotli stream and one
  ``reserved`` bit.

The archive is assembled by handing the carrier to ``up3.build_archive`` -- which verifies
its own bytes parse back to the requested codes -- and then splicing the new semantic
section into the outer it produced, refusing unless the finished bytes parse back to BOTH
the requested frame_embed codes and the requested carrier codes with a byte-identical
token tail.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_fe1_frame_embedding_search as fe1
import ddm_fe1_pose_price as price
import ddm_jg1_seg_solve as jg1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_up2_shipping_pose_solve as up2
import ddm_up3_carrier_splice as up3

N_PAIRS = fe1.N_PAIRS
CELL_COUNT = price.CELL_COUNT
ADMIT_BAR = -2e-5


def load_candidates(search_dir: Path) -> list[dict[str, Any]]:
    """Every pair whose realized search found a flip-reducing move."""
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted(search_dir.glob("search_rows_*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                pair = int(row["pair"])
                if pair in seen:
                    raise fe1.Fe1Error(f"pair {pair} appears in two shards")
                seen.add(pair)
                if row["final_delta"] < 0:
                    rows.append(row)
    return sorted(rows, key=lambda r: int(r["pair"]))


def cmd_merge(args) -> int:
    search_dir = Path(args.search_dir)
    rows = load_candidates(search_dir)
    covered = 0
    for path in sorted(search_dir.glob("search_rows_*.jsonl")):
        covered += sum(1 for line in path.open() if line.strip())
    total_cells = sum(int(r["final_delta"]) for r in rows)
    total_codes = sum(int(r["changed_codes"]) for r in rows)
    result = {
        "schema": "ddm_fe1_merge.v1",
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "pairs_searched": covered,
        "pairs_offering": len(rows),
        "offer_rate": covered and len(rows) / covered,
        "cells_repaired": -total_cells,
        "codes_changed": total_codes,
        "candidates": [
            {
                "pair": int(r["pair"]),
                "base_flips": int(r["base_flips"]),
                "final_flips": int(r["final_flips"]),
                "cells": -int(r["final_delta"]),
                "changed_codes": int(r["changed_codes"]),
                "base_row": r["base_row"],
                "final_row": r["final_row"],
            }
            for r in rows
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(
        json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=1)
    )
    return 0


def build_candidate_archive(
    section,
    codes: np.ndarray,
    carrier_codes: np.ndarray,
    *,
    tree_dir: Path = fe1.LIVE_TREE,
    container_search: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    """Archive bytes carrying ``codes`` (frame_embed) and ``carrier_codes`` (carrier).

    ``tree_dir`` is the TREE ROOT.  Two conventions meet here and mixing them is a real
    failure mode: ``up3`` wants the root (it reads ``<root>/archive.zip`` and imports the
    ``runtime`` package from it) while ``jg1``/``fe1`` want the ``runtime/`` package dir.
    Both are derived from the root here so no caller has to remember which is which.
    """
    runtime_dir = Path(tree_dir) / "runtime"
    import brotli

    body = up3.parse_shipped_body(tree_dir, verify_sha=False)
    built = up3.build_archive(
        body,
        carrier_codes,
        runtime_dir=tree_dir,
        container_search=container_search,
        verify=verify,
    )
    with zipfile.ZipFile(io.BytesIO(built["archive_bytes"])) as archive:
        outer = archive.read("p")

    ra, _cr, _ar1, _cp = up3._import_runtime(tree_dir)
    header = ra.RX1_MODEL_HEADER.unpack_from(outer)
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = header
    offset = ra.RX1_MODEL_HEADER.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes + semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    section_tail = outer[offset:]

    stream = section.stream_with_codes(codes)
    interleaved = up3._ck2_interleave_planes(stream)
    shapes: dict[tuple[str, int, int], bytes] = {}
    for quality in price.CONTAINER_QUALITIES:
        for lgwin in price.CONTAINER_LGWINS:
            shapes[("ck2", quality, lgwin)] = brotli.compress(
                interleaved, quality=quality, lgwin=lgwin
            )
            shapes[("plain", quality, lgwin)] = brotli.compress(
                stream, quality=quality, lgwin=lgwin
            )
    # Ties go to the SHIPPED shape.  Two shapes can produce the same NUMBER of bytes and
    # different bytes (brotli records its window size), so a length-only tie-break would
    # ship a stream that differs from the pointer's for no reason and break the null-build
    # identity control that makes every later byte number checkable.
    chosen = min(
        shapes,
        key=lambda key: (len(shapes[key]), key != price.SHIPPED_SHAPE, key),
    )
    semantic_stream = shapes[chosen]
    reserved = (
        reserved | ra.CK2_RESERVED_SEMANTIC_PLANE2
        if chosen[0] == "ck2"
        else reserved & ~ra.CK2_RESERVED_SEMANTIC_PLANE2
    )
    new_outer = b"".join(
        (
            ra.RX1_MODEL_HEADER.pack(
                magic,
                version,
                codec,
                table_mode,
                reserved,
                hpac_bytes,
                len(semantic_stream),
                len(carrier_stream),
            ),
            hpac_stream,
            semantic_stream,
            carrier_stream,
            section_tail,
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        entry = zipfile.ZipInfo("p", date_time=tuple(body.zip_info["date_time"]))
        entry.compress_type = body.zip_info["compress_type"]
        entry.external_attr = body.zip_info["external_attr"]
        entry.create_system = body.zip_info["create_system"]
        archive.writestr(entry, new_outer)
    archive_bytes = buffer.getvalue()

    if verify:
        # Parse the FINISHED bytes back through the receiver's own reader.  A tree-local
        # scratch file is used because ``read_residual_archive`` takes a path; it is
        # rewritten every call and never becomes evidence.
        scratch = fe1.WORK / "candidate/.verify_archive.zip"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_bytes(archive_bytes)
        recovered_section = fe1.load_semantic_section(
            archive_path=scratch, runtime_dir=runtime_dir
        )
        if not np.array_equal(
            recovered_section.codes.astype(np.int64), np.asarray(codes, dtype=np.int64)
        ):
            raise fe1.Fe1Error(
                "the written archive does not parse back to the requested frame_embed "
                "codes; refusing to return unverified bytes"
            )
        recovered_carrier, _info = up3.parse_back_codes(
            archive_bytes, runtime_dir=tree_dir
        )
        if not np.array_equal(
            np.asarray(recovered_carrier, dtype=np.int64),
            np.asarray(carrier_codes, dtype=np.int64),
        ):
            raise fe1.Fe1Error(
                "the written archive does not parse back to the requested carrier codes"
            )
        # The reference for this invariant is the TREE BEING BUILT ON, not the module's
        # live pointer.  Binding it to fe1.LIVE_ARCHIVE was a real bug: a re-base onto a
        # pointer whose tail moved (sj1's passes move it) would refuse a candidate whose
        # tail is in fact byte-identical to its OWN base.  It failed closed, which is why
        # it was findable; the fix is to compare against the right object.
        if bytes(ra.read_residual_archive(scratch).token_stream) != bytes(
            ra.read_residual_archive(Path(tree_dir) / "archive.zip").token_stream
        ):
            raise fe1.Fe1Error(
                "the token stream is not byte-identical to the tree this candidate is "
                "built on; this arm changes the semantic and carrier sections only"
            )

    return {
        "archive_bytes": archive_bytes,
        "archive_size": len(archive_bytes),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "semantic_stream_bytes": len(semantic_stream),
        "semantic_container": list(chosen),
        "carrier_stream_bytes": len(carrier_stream),
        "carrier_container": built["container"],
        "rx1_reserved": f"{reserved:#x}",
    }



# ----------------------------------------------------------------------------------
# admit -- render, pose, re-solve, then a subset sweep priced by REAL archive builds
# ----------------------------------------------------------------------------------


def cmd_pose(args) -> int:
    """Render every candidate pair's moved frame, then price its pose leg.

    One JSONL row per pair, appended as the pair finishes.  The rendered frames are KEPT
    (a memmap beside the rows), because they are the payload the admission and every
    re-price after it consume; measuring their d_pose and discarding them would force a
    full re-render on the next question asked of this ledger.
    """
    fe1._set_threads(args.threads)
    candidates = json.loads(Path(args.candidates).read_text())["candidates"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "pose_rows.jsonl"
    done: set[int] = set()
    if rows_path.is_file() and args.resume:
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(int(json.loads(line)["pair"]))

    body = fe1.load_body(with_raw=False, verify_shas=not args.no_sha)
    base_raw = fe1._open_raw(fe1.LIVE_RAW)
    pairs = [int(c["pair"]) for c in candidates]
    frames_path = out_dir / "odd_frames.u8"
    shape = (len(pairs), jg1.CAMERA_H, jg1.CAMERA_W, 3)
    mode = "r+" if frames_path.is_file() else "w+"
    frames = np.memmap(frames_path, dtype=np.uint8, mode=mode, shape=shape)
    slot = {pair: index for index, pair in enumerate(pairs)}
    for candidate in candidates:
        pair = int(candidate["pair"])
        if pair in done:
            continue
        fe1.set_pair_codes(body, pair, candidate["final_row"])
        frames[slot[pair]] = fe1.render_pair(body, pair)[0]
        fe1.restore_pair_codes(body, pair)
    frames.flush()
    (out_dir / "OVERLAY.json").write_text(
        json.dumps({"pairs": pairs, "shape": list(shape)}, indent=1)
    )

    overlay = price._MemoryOverlayRaw(
        base_raw, {pair: np.asarray(frames[slot[pair]]) for pair in pairs}
    )
    base_inst = price.build_pose_instrument(base_raw)
    moved_inst = price.build_pose_instrument(overlay)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(args.base_mean_d_pose)
    started = time.time()
    completed = 0
    for candidate in candidates:
        pair = int(candidate["pair"])
        if pair in done:
            continue
        d_pose_base = float(
            br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0]
        )
        # CONTROL: re-solving the pair on its UNMOVED render tells us how much of any
        # apparent pose gain belongs to the move and how much was simply left on the
        # table by the live carrier.  Without it a lucky re-solve reads as a move's win.
        control = jg5.refine_pair(
            base_inst,
            pair,
            live_codes[pair],
            dd_threshold=dd_threshold,
            outer_rounds=args.outer_rounds,
            max_gn_iterations=args.max_gn_iterations,
        )
        d_pose_stale = float(
            br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0]
        )
        refined = jg5.refine_pair(
            moved_inst,
            pair,
            live_codes[pair],
            dd_threshold=dd_threshold,
            outer_rounds=args.outer_rounds,
            max_gn_iterations=args.max_gn_iterations,
        )
        row = {
            "pair": pair,
            "cells": int(candidate["cells"]),
            "changed_codes": int(candidate["changed_codes"]),
            "final_row": candidate["final_row"],
            "d_pose_base": d_pose_base,
            "d_pose_base_resolved": float(control["final_d_pose"]),
            "control_codes": [int(c) for c in control["codes"]],
            "d_pose_stale": d_pose_stale,
            "d_pose_resolved": float(refined["final_d_pose"]),
            "resolved_codes": [int(c) for c in refined["codes"]],
            "stale_x": d_pose_stale / d_pose_base if d_pose_base > 0 else math.inf,
            "recovery_x": (
                d_pose_stale / refined["final_d_pose"]
                if refined["final_d_pose"] > 0
                else math.inf
            ),
            "stop_reason": refined["stop_reason"],
        }
        with rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        completed += 1
        if args.progress:
            print(
                f"  {completed}/{len(candidates) - len(done)} pair {pair}: base "
                f"{d_pose_base:.4e} (control re-solve {control['final_d_pose']:.4e}) -> "
                f"stale {d_pose_stale:.4e} -> resolved {refined['final_d_pose']:.4e} "
                f"({(time.time() - started) / completed:.1f} s/pair)",
                flush=True,
            )
    receipt = {
        "schema": "ddm_fe1_pose_rows.v1",
        "axis": "[cpu_torch fp32 PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "pairs": pairs,
        "rows_path": str(rows_path),
        "frames_path": str(frames_path),
        "dd_threshold": dd_threshold,
        "elapsed_s": round(time.time() - started, 1),
    }
    (out_dir / "POSE_ROWS.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("pairs", "elapsed_s")}, default=len))
    return 0


def cmd_admit(args) -> int:
    """Sweep subsets of the priced pairs and pick the one the REAL archive likes best."""
    fe1._set_threads(args.threads)
    rows = [
        json.loads(line)
        for line in Path(args.pose_rows).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise fe1.Fe1Error("no priced pairs")
    section = fe1.load_semantic_section()
    body = up3.parse_shipped_body(fe1.LIVE_TREE, verify_sha=False)
    live_carrier = np.asarray(body.codes, dtype=np.int32)
    base_pose = float(args.base_mean_d_pose)
    base_leg = math.sqrt(10.0 * base_pose)

    # Rank by what each pair is worth on its own: seg gain minus the pose it costs after
    # the re-solve, both in score units.  Bytes are NOT apportioned here -- they are not
    # additive on this section (the container-break law), so they are priced per subset
    # by a real build below.
    for row in rows:
        pose_delta = row["d_pose_resolved"] - row["d_pose_base"]
        row["dS_seg"] = -row["cells"] * 100.0 / CELL_COUNT
        row["dS_pose"] = (
            math.sqrt(10.0 * (base_pose + pose_delta / N_PAIRS)) - base_leg
        )
        row["value"] = row["dS_seg"] + row["dS_pose"]
    ranked = sorted(rows, key=lambda r: r["value"])

    cuts = sorted(
        {
            count
            for count in (
                [len(ranked)]
                + [int(len(ranked) * f) for f in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
            )
            if count > 0
        }
    )
    results = []
    best = None
    for count in cuts:
        subset = ranked[:count]
        codes = section.codes.astype(np.int64).copy()
        carrier = live_carrier.copy()
        for row in subset:
            codes[int(row["pair"])] = np.asarray(row["final_row"], dtype=np.int64)
            carrier[int(row["pair"])] = np.asarray(row["resolved_codes"], dtype=np.int32)
        built = build_candidate_archive(
            section, codes, carrier, verify=not args.no_verify
        )
        cells = sum(int(r["cells"]) for r in subset)
        pose_mean = base_pose + sum(
            r["d_pose_resolved"] - r["d_pose_base"] for r in subset
        ) / N_PAIRS
        d_seg_new = (fe1.LIVE_D_SEG_CELLS - cells) / CELL_COUNT
        dS = (
            100.0 * (d_seg_new - fe1.LIVE_D_SEG_LOCAL)
            + (math.sqrt(10.0 * pose_mean) - base_leg)
            + (built["archive_size"] - fe1.LIVE_ARCHIVE_BYTES) * fe1.RATE_PER_BYTE
        )
        entry = {
            "pairs": count,
            "cells": cells,
            "archive_bytes": built["archive_size"],
            "archive_sha256": built["archive_sha256"],
            "d_archive_bytes": built["archive_size"] - fe1.LIVE_ARCHIVE_BYTES,
            "semantic_stream_bytes": built["semantic_stream_bytes"],
            "semantic_container": built["semantic_container"],
            "carrier_stream_bytes": built["carrier_stream_bytes"],
            "d_seg_local": d_seg_new,
            "d_pose_mean": pose_mean,
            "dS_seg": 100.0 * (d_seg_new - fe1.LIVE_D_SEG_LOCAL),
            "dS_pose": math.sqrt(10.0 * pose_mean) - base_leg,
            "dS_rate": (built["archive_size"] - fe1.LIVE_ARCHIVE_BYTES)
            * fe1.RATE_PER_BYTE,
            "dS_total": dS,
            "admits": bool(dS < ADMIT_BAR),
            "pairs_included": [int(r["pair"]) for r in subset],
        }
        results.append(entry)
        print(
            f"  cut {count:>4} pairs / {cells:>4} cells: archive "
            f"{built['archive_size']} B ({entry['d_archive_bytes']:+d}) -> dS "
            f"{dS:+.6e} {'ADMITS' if entry['admits'] else 'below bar'}",
            flush=True,
        )
        if best is None or dS < best["dS_total"]:
            best = entry
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / "archive.zip").write_bytes(built["archive_bytes"])
            np.save(Path(args.out_dir) / "frame_embed_codes.npy", codes.astype(np.int8))
            np.save(Path(args.out_dir) / "carrier_codes.npy", carrier)
    result = {
        "schema": "ddm_fe1_admission.v1",
        "axis": (
            "d_seg [macOS-CPU advisory, jg1/sj1 instrument, DALI GT]; d_pose "
            "[cpu_torch fp32]; bytes EXACT through a real archive build"
        ),
        "score_claim": False,
        "live_archive_bytes": fe1.LIVE_ARCHIVE_BYTES,
        "live_score_t4": fe1.LIVE_SCORE_T4,
        "admit_bar": ADMIT_BAR,
        "pricing": "REAL archive build per cut; no ledger sum anywhere",
        "cuts": results,
        "best": best,
        "projected_score_t4": fe1.LIVE_SCORE_T4 + (best["dS_total"] if best else 0.0),
    }
    out = Path(args.out_dir) / "ADMISSION.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"best": best and {k: best[k] for k in ("pairs", "cells", "archive_bytes", "dS_total", "admits")}, "projected_score_t4": result["projected_score_t4"]}, indent=1))
    return 0



def cmd_stage(args) -> int:
    """Copy the live tree, drop the candidate archive in, and re-pin the receiver.

    The receiver pins its own archive's sha and size, so a candidate tree that keeps the
    pointer's pins would refuse its own bytes.  The pin patcher is sj1's, imported rather
    than re-written: it finds the two constants by PATTERN and verifies what it wrote,
    which is the shape that survives a generation change of the literals.
    """
    import shutil

    import ddm_sj1_joint_admission as sj1_joint

    archive = Path(args.archive)
    if not archive.is_file():
        raise fe1.Fe1Error(f"no candidate archive at {archive}")
    archive_bytes = archive.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    shutil.copytree(fe1.LIVE_TREE, out_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    (out_dir / "archive.zip").write_bytes(archive_bytes)
    pins = sj1_joint.patch_inflate_pins(out_dir, archive_sha, len(archive_bytes))

    # Census: everything except archive.zip and inflate.py must be byte-identical to the
    # live tree, so the candidate differs only where it means to.
    changed = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(out_dir)
        live = fe1.LIVE_TREE / rel
        if not live.is_file():
            changed.append({"path": str(rel), "why": "absent from the live tree"})
            continue
        if path.read_bytes() != live.read_bytes():
            changed.append({"path": str(rel), "why": "bytes differ"})
    expected = {"archive.zip", "inflate.py"}
    unexpected = [c for c in changed if c["path"] not in expected]
    if unexpected:
        raise fe1.Fe1Error(f"staged tree differs outside {sorted(expected)}: {unexpected}")

    receipt = {
        "schema": "ddm_fe1_stage.v1",
        "axis": "[scorer-free EXACT byte measurement]",
        "score_claim": False,
        "live_tree": str(fe1.LIVE_TREE),
        "staged_tree": str(out_dir),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "d_archive_bytes": len(archive_bytes) - fe1.LIVE_ARCHIVE_BYTES,
        "inflate_pins": pins,
        "files_differing_from_live_tree": [c["path"] for c in changed],
    }
    (out_dir.parent / "STAGE.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    return 0



def cmd_base_pose(args) -> int:
    """Measure the LIVE row's own per-pair d_pose over all 600 pairs, on this instrument.

    The admission's pose arithmetic divides by a population mean.  Taking that mean from
    another arm's seal would make this arm's pose leg depend on a number it never
    measured, so it is measured here -- live decode, live carrier codes, the same
    ``up2.measure_pose`` the candidate legs use -- and the gap to the inherited value is
    reported rather than assumed away.
    """
    fe1._set_threads(args.threads)
    raw = fe1._open_raw(fe1.LIVE_RAW)
    inst = price.build_pose_instrument(raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "base_pose_per_pair.npy", per_pair)
    measured = float(per_pair.mean())
    result = {
        "schema": "ddm_fe1_base_pose.v1",
        "axis": "[cpu_torch fp32 PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "archive_sha256": fe1.LIVE_ARCHIVE_SHA256,
        "archive_bytes": fe1.LIVE_ARCHIVE_BYTES,
        "decode": str(fe1.LIVE_RAW),
        "d_pose_mean_measured": measured,
        "d_pose_mean_inherited": fe1.LIVE_D_POSE,
        "relative_gap": measured / fe1.LIVE_D_POSE - 1.0,
        "pose_leg_measured": math.sqrt(10.0 * measured),
        "pose_leg_inherited": math.sqrt(10.0 * fe1.LIVE_D_POSE),
        "leg_gap_in_score_units": math.sqrt(10.0 * measured)
        - math.sqrt(10.0 * fe1.LIVE_D_POSE),
        "per_pair_path": str(out_dir / "base_pose_per_pair.npy"),
        "elapsed_s": round(time.time() - started, 1),
    }
    (out_dir / "BASE_POSE.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="collect the search's flip-reducing pairs")
    merge.add_argument("--search-dir", default=str(fe1.WORK / "search"))
    merge.add_argument("--out", default=str(fe1.WORK / "admission/CANDIDATES.json"))
    merge.set_defaults(func=cmd_merge)

    pose = sub.add_parser("pose", help="render + price the pose leg of every candidate")
    pose.add_argument("--candidates", default=str(fe1.WORK / "admission/CANDIDATES.json"))
    pose.add_argument("--out-dir", default=str(fe1.WORK / "admission/pose"))
    pose.add_argument("--threads", type=int, default=3)
    pose.add_argument("--no-sha", action="store_true")
    pose.add_argument("--resume", action="store_true", default=True)
    pose.add_argument("--progress", action="store_true", default=True)
    pose.add_argument("--outer-rounds", type=int, default=40)
    pose.add_argument("--max-gn-iterations", type=int, default=400)
    pose.add_argument("--base-mean-d-pose", type=float, default=fe1.LIVE_D_POSE)
    pose.set_defaults(func=cmd_pose)

    admit = sub.add_parser("admit", help="sweep subsets, priced by real archive builds")
    admit.add_argument("--pose-rows", default=str(fe1.WORK / "admission/pose/pose_rows.jsonl"))
    admit.add_argument("--out-dir", default=str(fe1.WORK / "candidate"))
    admit.add_argument("--threads", type=int, default=3)
    admit.add_argument("--no-verify", action="store_true")
    admit.add_argument("--base-mean-d-pose", type=float, default=fe1.LIVE_D_POSE)
    admit.set_defaults(func=cmd_admit)

    stage = sub.add_parser("stage", help="stage the candidate runtime tree")
    stage.add_argument("--archive", default=str(fe1.WORK / "candidate/archive.zip"))
    stage.add_argument("--out-dir", default=str(fe1.WORK / "candidate/candidate_runtime"))
    stage.add_argument("--clean", action="store_true", default=True)
    stage.set_defaults(func=cmd_stage)

    bp = sub.add_parser("base-pose", help="measure the live row's own n600 pose leg")
    bp.add_argument("--out-dir", default=str(fe1.WORK / "admission"))
    bp.add_argument("--threads", type=int, default=3)
    bp.add_argument("--batch-size", type=int, default=8)
    bp.set_defaults(func=cmd_base_pose)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
