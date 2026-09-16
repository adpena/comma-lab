#!/usr/bin/env python3
"""ddm_pd6 -- the PRICE-FIRST generator on the move-52 token field.

WHY THIS EXISTS
---------------
Every pre-distortion pass on this object (sj1 passes 1-8, pd1-pd5) generated its proposals by
POSE SALIENCY and priced them afterwards.  pd5 measured the consequence and closed the
multi-token formulation on it: the context discount compounds, the credit does not.  pd5 Sect. 9
scopes its closure to EXCLUDE the reverse generation order, and pd4 named it as the one price
lever nobody has run.  This module is that order, and only that order:

    1. ``capture``     read the SHIPPED coder's OWN probability rows off one control encode of
                       move 52's own field.  Every one of the 600 x 384 x 512 positions is coded
                       (``LaneMixer.end_frame`` refuses a frame unless ``self.seen.all()``), and
                       every row is a full K=5 simplex, so ONE encode yields the first-order
                       price of EVERY single-token change in the plane:
                       ``delta_bits(p, t) = -log2 p_model(t) + log2 p_model(truth)``.
    2. ``candidates``  the CHEAP HALF: per pair, the changes that first-order price cheapest,
                       masked to the argmax interior, plus 2-token combinations built from the
                       cheap list itself.  This is a RANKING, never a charge
                       ([[first_order_token_price_is_a_ranking_never_a_charge_tail_flag_mass_untouchable_20260909]]).
    3. ``sheets``      pd4's sheet construction over THOSE candidates -- one proposal per pair
                       per sheet, so one real 600-frame encode CHARGES a whole sheet.
    4. ``realize``     credit AFTERWARDS: render, frozen-argmax seg census, jg5 carrier
                       re-solve, per-pair resolved pose credit -- on the candidates the real
                       coder has already charged cheaply.

Everything downstream is pd4's and pd5's, imported and unchanged: the move-52 binding
(``pd4.bind_move52``), the economics, ``pd4.cmd_price_merge`` for the per-proposal charge,
``pd4.cmd_carry``/``cmd_assemble``, ``pd5.cmd_setprice``/``carry-from-set``/``setabsorb`` for the
SET price, ``ddm_sj1_rlc1_price`` for the encode and ``ddm_sj1_joint_admission`` for the three
legs.  Two functions are rebound PROCESS-LOCALLY for the duration of one call and restored in
``finally``:

* ``runtime.rlc1_mixer.LaneMixer.coding`` / ``.end_frame`` during ``capture`` -- the wrappers
  RECORD the probability rows the shipped mixer already computed and then call the true
  originals; the coder is not re-implemented, re-weighted or re-ordered, and the encode's own
  control identity (byte-identical to move 52's archive) is the proof that nothing changed.
* ``pd4._encode_bits`` during ``price-merge`` -- so pd4's merge reads ``pd6sheet…`` inside pd6's
  own store rather than a ``pd4sheet…`` name that would be a provenance lie here (pd5 made the
  same swap for the same reason).

No scorer weights ship anywhere, no Modal call, no authorization, no fire, no packet, no pointer
write.  Only ``upstream/evaluate.py`` on shipped bytes is a score, and MAIN fires.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True  # every tree this arm reads is another arm's custody

import ddm_pd4_pose_directed_pass4 as pd4  # the path above must be set first

pd3 = pd4.pd3
N_PAIRS = pd4.N_PAIRS
EVAL_H, EVAL_W = pd4.EVAL_H, pd4.EVAL_W
CELL_COUNT = pd4.CELL_COUNT
FLOOR_PAIRS = pd4.FLOOR_PAIRS
BASE_BAND_ABS = pd4.BASE_BAND_ABS
S_PER_BYTE = pd4.S_PER_BYTE

PD6_STORE = Path("/Volumes/APDataStore/pact/ddm_pd6")

row_edits = pd4.row_edits
proposal_identity = pd4.proposal_key
seg_s_per_cell = pd4.seg_s_per_cell
pose_s_per_pair_unit = pd4.pose_s_per_pair_unit
sha256_file = pd4.sha256_file


class Pd6Error(RuntimeError):
    """A pd6 input or invariant is not what the measured object says it is."""


# ----------------------------------------------------------------------------------
# stage: bind -- re-point every stale module at MOVE 52 and re-verify the inherited decode
# ----------------------------------------------------------------------------------


def cmd_bind(args) -> int:
    receipts = pd4.bind_move52(verify_raw=bool(args.verify_raw), raw=args.raw)
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    live = pointer["our_local_frontier_contest_cuda"]
    if live["archive_sha256"] != pd4.MOVE52_ARCHIVE_SHA256:
        raise Pd6Error(
            f"POINTER MOVED: the live row is {live['archive_sha256']}, not move 52's "
            f"{pd4.MOVE52_ARCHIVE_SHA256}; re-derive the base before pricing anything"
        )
    payload = {
        "schema": "ddm_pd6_bind.v1",
        "axis": "[macOS-CPU advisory / binding receipt, no score]",
        "score_claim": False,
        "pd4_receipts": receipts,
        "live_pointer": {
            "archive_sha256": live["archive_sha256"],
            "score": live.get("score"),
            "lane_id": live.get("lane_id"),
            "re_derived_from": str(REPO / ".omx/state/canonical_frontier_pointer.json"),
        },
        "decode_provenance": (
            "INHERITED: this arm did not re-decode move 52's archive.  pd3 and pd4 each parsed "
            "the SAME shipped bytes back cold on their own copies of the runtime tree and "
            "produced a bit-identical 0.raw (pd4 MEASURED max abs per-pair pose difference "
            "0.000e+00 against pd3's vector over all 600 pairs); pd5 re-hashed it.  This arm "
            "re-verifies the same raw by sha256 and stands on it -- a fourth decode would add "
            "3.66 GB to a store with 14 GiB free and could only reproduce the same sha."
        ),
    }
    if args.verify_raw:
        payload["raw_reverified"] = {
            "path": str(args.raw), "sha256": sha256_file(Path(args.raw)),
            "bytes": Path(args.raw).stat().st_size,
            "expect_sha256": pd4.PD3_RAW_SHA256, "expect_bytes": pd4.PD3_RAW_BYTES,
        }
        got = payload["raw_reverified"]
        if got["sha256"] != pd4.PD3_RAW_SHA256 or got["bytes"] != pd4.PD3_RAW_BYTES:
            raise Pd6Error(f"inherited raw does not reproduce move 52's decode: {got}")
    if args.control_archive is not None:
        sha = sha256_file(Path(args.control_archive))
        payload["control_archive"] = {
            "path": str(args.control_archive), "sha256": sha,
            "bytes": Path(args.control_archive).stat().st_size,
        }
        if sha != pd4.MOVE52_ARCHIVE_SHA256:
            raise Pd6Error(f"control archive sha {sha} != {pd4.MOVE52_ARCHIVE_SHA256}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: capture -- the shipped coder's OWN probability rows, off one real encode
# ----------------------------------------------------------------------------------


def _interior(argmax_plane: np.ndarray, radius: int) -> np.ndarray:
    import ddm_pd1_pose_directed as pd1

    return pd1.interior_mask(np.asarray(argmax_plane), radius)


def cmd_capture(args) -> int:
    """Run ONE real RLC1 encode and record the model's own price for every alternative symbol.

    The wrappers below are pure observers: each calls the shipped ``LaneMixer`` method and
    returns its value unchanged.  The encode's own control-identity gate (a byte-identical
    re-pack of move 52's archive) is what proves the observation changed nothing.
    """
    import ddm_sj1_rlc1_price as rlc1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    argmax = np.load(args.argmax, mmap_mode="r")
    if argmax.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise Pd6Error(f"argmax is {argmax.shape}, expected {(N_PAIRS, EVAL_H, EVAL_W)}")
    keep = int(args.keep_per_frame)
    radius = int(args.interior_radius)

    # the shipped runtime package only becomes importable once jg2 has put the pricer's own
    # sha-verified runtime COPY on sys.path; rlc1.encode does that itself, so do it first here
    # and let encode's own (idempotent) call find the module already in sys.modules -- the
    # patch below must be installed on the SAME class object encode will use.
    inputs = rlc1.guard()
    rlc1.jg2.load_runtime(Path(inputs["runtime_copy"]))
    from runtime.rlc1_mixer import LaneMixer

    original_coding = LaneMixer.coding
    original_end = LaneMixer.end_frame
    state: dict[str, Any] = {"frame": None, "logp": None, "seen": None, "frames_done": []}

    def coding(self, rows, positions, plane, previous):
        result = original_coding(self, rows, positions, plane, previous)
        frame = int(self.frame)
        if state["frame"] != frame:
            state["frame"] = frame
            state["logp"] = np.zeros((EVAL_H * EVAL_W, int(np.shape(result)[1])),
                                     dtype=np.float32)
            state["seen"] = np.zeros(EVAL_H * EVAL_W, dtype=bool)
        idx = np.asarray(positions, dtype=np.int64)
        probs = np.clip(np.asarray(result, dtype=np.float64), 1e-12, 1.0)
        state["logp"][idx] = (-np.log2(probs)).astype(np.float32)
        state["seen"][idx] = True
        return result

    def end_frame(self, plane, previous):
        frame = int(self.frame)
        path = frames_dir / f"frame_{frame:04d}.npz"
        if state["logp"] is not None and state["frame"] == frame and not path.exists():
            if not bool(state["seen"].all()):
                raise Pd6Error(f"frame {frame} did not observe every coded position")
            truth = np.asarray(plane, dtype=np.int64).reshape(-1)
            logp = state["logp"]
            own = logp[np.arange(logp.shape[0]), truth]
            delta = logp - own[:, None]          # bits ADDED by switching to each symbol
            delta[np.arange(logp.shape[0]), truth] = np.inf
            if frame in FLOOR_PAIRS:
                mask = np.zeros(EVAL_H * EVAL_W, dtype=bool)
            else:
                mask = _interior(argmax[frame], radius).reshape(-1)
            delta[~mask] = np.inf
            flat = delta.reshape(-1)
            finite = int(np.isfinite(flat).sum())
            take = min(keep, finite)
            if take > 0:
                order = np.argpartition(flat, take - 1)[:take]
                order = order[np.argsort(flat[order], kind="stable")]
                pos = (order // delta.shape[1]).astype(np.int32)
                sym = (order % delta.shape[1]).astype(np.uint8)
                bits = flat[order].astype(np.float32)
            else:
                pos = np.zeros(0, dtype=np.int32)
                sym = np.zeros(0, dtype=np.uint8)
                bits = np.zeros(0, dtype=np.float32)
            finite_bits = flat[np.isfinite(flat)]
            summary = {
                "frame": frame,
                "candidates_finite": finite,
                "interior_positions": int(mask.sum()),
                "own_bits_sum": float(own.sum()),
                "delta_quantiles_bits": [
                    float(np.quantile(finite_bits, q)) for q in (0.0, 0.01, 0.05, 0.25, 0.5)
                ] if finite else [],
            }
            temporary = path.with_suffix(".npz.pending")
            # np.savez APPENDS .npz to a PATH that lacks it, so the atomic rename must be
            # driven through an open handle or the temp file lands under another name.
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, position=pos, symbol=sym, delta_bits=bits,
                                    truth=truth[pos].astype(np.uint8),
                                    summary=np.frombuffer(
                                        json.dumps(summary).encode(), dtype=np.uint8))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            state["frames_done"].append(frame)
        state["logp"] = None
        state["seen"] = None
        state["frame"] = None
        return original_end(self, plane, previous)

    LaneMixer.coding = coding
    LaneMixer.end_frame = end_frame
    started = time.time()
    try:
        result = rlc1.encode(args.field, args.tag)
    finally:
        LaneMixer.coding = original_coding
        LaneMixer.end_frame = original_end

    receipt = {
        "schema": "ddm_pd6_capture.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT model rows through the shipped coder]",
        "score_claim": False,
        "field": args.field, "tag": args.tag,
        "frames_captured_this_process": sorted(state["frames_done"]),
        "frames_on_disk": len(list(frames_dir.glob("frame_*.npz"))),
        "keep_per_frame": keep, "interior_radius": radius,
        "floor_pairs_excluded": sorted(FLOOR_PAIRS),
        "encode": {
            "archives": result["archives"],
            "archive_matches_live_pointer": result["archive_matches_live_pointer"],
            "stream_bytes": result["stream_bytes"],
            "decoded_field_sha256": result["decoded_field_sha256"],
            "ideal_bytes": result["ideal_bytes"],
        },
        "method": (
            "delta_bits(position, symbol) = -log2 p_model(symbol) + log2 p_model(truth), read "
            "off the probability rows the SHIPPED LaneMixer produced during a real 600-frame "
            "encode.  FIRST-ORDER: it is the model's own charge for the symbol at that "
            "position under the base field's state, and it does NOT carry the group-causal "
            "geometry update or the adaptive response the change itself causes.  It is the "
            "GENERATOR; the sheet encode is the CHARGE."
        ),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "CAPTURE.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "frames_captured_this_process"}, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: candidates -- the CHEAP HALF, per pair, single- and two-token
# ----------------------------------------------------------------------------------


def _load_frame(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def cmd_candidates(args) -> int:
    """Rank each pair's plane by the model's own first-order price and emit the cheap head.

    Two families, both generated by PRICE and nothing else:

    * ``single``  the pair's k cheapest single-token changes.
    * ``pair``    two cheap changes realized JOINTLY.  Preference is given to a partner
      inside ``--combo-radius`` of the head, because the coder's context model discounts a
      neighbour (pd5 MEASURED the marginal second token at 5.6 bits against 15.4 isolated);
      when no cheap neighbour exists the next cheapest cell anywhere in the plane is used, so
      the family is never empty for price reasons.
    """
    frames_dir = Path(args.frames)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    control = np.load(args.control_npz)
    planes = {int(k): control[k] for k in control.files}
    if len(planes) != N_PAIRS:
        raise Pd6Error(f"control field has {len(planes)} planes, expected {N_PAIRS}")

    per_pair: dict[int, list[dict[str, Any]]] = {}
    census: list[dict[str, Any]] = []
    for pair in range(N_PAIRS):
        path = frames_dir / f"frame_{pair:04d}.npz"
        if not path.exists():
            raise Pd6Error(f"capture is incomplete: {path} is missing")
        data = _load_frame(path)
        summary = json.loads(bytes(data["summary"]).decode())
        pos = data["position"].astype(np.int64)
        sym = data["symbol"].astype(np.int64)
        bits = data["delta_bits"].astype(np.float64)
        truth = data["truth"].astype(np.int64)
        plane = planes[pair]
        rows: list[dict[str, Any]] = []
        used_cells: set[tuple[int, int]] = set()
        for i in range(len(pos)):
            r, c = int(pos[i] // EVAL_W), int(pos[i] % EVAL_W)
            if (r, c) in used_cells:
                continue        # one change per cell: the cheapest symbol at that cell
            if int(plane[r, c]) != int(truth[i]):
                raise Pd6Error(
                    f"pair {pair} cell ({r},{c}) holds {int(plane[r, c])} in the control field "
                    f"but the capture recorded {int(truth[i])}"
                )
            used_cells.add((r, c))
            rows.append({"row": r, "col": c, "old": int(truth[i]), "new": int(sym[i]),
                         "first_order_bits": float(bits[i])})
        # SPATIAL SEPARATION between the ranks a pair contributes.  The cheapest cells are
        # already spread (frame 0: the cheapest 1024 span 201 distinct rows), but nothing
        # guarantees ranks 0..3 are not four adjacent cells, which would make the four sheets
        # four copies of one experiment.  The constraint uses only geometry -- no pose, no
        # saliency, no credit -- so the generator stays the price and nothing else.
        separation = int(args.min_separation)
        singles: list[dict[str, Any]] = []
        for row in rows:
            if len(singles) >= int(args.per_pair_single):
                break
            if any(max(abs(row["row"] - k["row"]), abs(row["col"] - k["col"])) < separation
                   for k in singles):
                continue
            singles.append(row)
        entries: list[dict[str, Any]] = []
        for rank, row in enumerate(singles):
            entries.append({
                "pair": pair, "family": "price_single", "tokens_changed": 1,
                "edits": [[row["row"], row["col"], row["old"], row["new"]]],
                "first_order_bits": row["first_order_bits"],
                "first_order_rank": rank,
            })
        # two-token: the head plus its cheapest partner, NEIGHBOURS FIRST because the coder's
        # context model discounts a neighbour (pd5: 5.6 marginal bits against 15.4 isolated)
        combos = 0
        if rows and int(args.per_pair_combo) > 0:
            head = rows[0]
            radius = int(args.combo_radius)

            def near(row: dict[str, Any], head=head, radius=radius) -> bool:
                return (abs(row["row"] - head["row"]) <= radius
                        and abs(row["col"] - head["col"]) <= radius)

            partners = ([r for r in rows[1:] if near(r)]
                        + [r for r in rows[1:] if not near(r)])
            for row in partners[: int(args.per_pair_combo)]:
                entries.append({
                    "pair": pair, "family": "price_pair", "tokens_changed": 2,
                    "edits": [[head["row"], head["col"], head["old"], head["new"]],
                              [row["row"], row["col"], row["old"], row["new"]]],
                    "first_order_bits": head["first_order_bits"] + row["first_order_bits"],
                    "first_order_rank": len(singles) + combos,
                    "partner_is_neighbour": near(row),
                })
                combos += 1
        # DELIBERATE PRICE CONTROLS.  Every pair here has more candidates than there are
        # sheets, so nothing would be held fixed and pd4's noise floor (F3) would have no
        # object to measure.  Every Nth pair therefore contributes ONLY its rank-0 candidate
        # and is repeated across every sheet, so the spread of its measured delta bits IS the
        # spill a crowd of neighbours causes on a price.
        held_fixed = int(args.hold_fixed_every) > 0 and pair % int(args.hold_fixed_every) == 0
        if held_fixed:
            entries = entries[:1]
        if entries:
            per_pair[pair] = entries
        census.append({
            "pair": pair, "interior_positions": summary["interior_positions"],
            "own_bits_sum": summary["own_bits_sum"],
            "candidates_finite": summary["candidates_finite"],
            "cells_available": len(rows),
            "singles": len(singles), "combos": combos,
            "cheapest_bits": rows[0]["first_order_bits"] if rows else None,
            "delta_quantiles_bits": summary["delta_quantiles_bits"],
        })

    # CONTROL: the captured rows must be the rows the coder actually charged.  Per frame the
    # sum of -log2 p over the TRUE symbols is exactly the encode's own per_frame_bits, so the
    # two must agree to float32 rounding.  If they do not, the wrappers observed a different
    # object from the one that priced the archive.
    control: dict[str, Any] = {}
    if args.encode_json:
        charged = np.asarray(json.loads(Path(args.encode_json).read_text())["per_frame_bits"],
                             dtype=np.float64)
        observed = np.array([c["own_bits_sum"] for c in census], dtype=np.float64)
        rel = np.abs(observed - charged) / np.maximum(np.abs(charged), 1.0)
        control = {
            "encode_json": str(args.encode_json),
            "frames": int(charged.size),
            "max_abs_bits": float(np.abs(observed - charged).max()),
            "max_relative": float(rel.max()),
            "meaning": ("sum over the frame's coded positions of -log2 p(TRUE symbol), captured "
                        "by the wrapper, against the same sum the pricer accumulated"),
        }
        if control["max_relative"] > 1e-5:
            raise Pd6Error(f"captured rows are not the charged rows: {control}")

    cheapest = np.array([c["cheapest_bits"] for c in census
                         if c["cheapest_bits"] is not None], dtype=np.float64)
    report = {
        "schema": "ddm_pd6_candidates.v1",
        "axis": "[macOS-CPU advisory / scorer-free, first-order model price]",
        "score_claim": False,
        "generator": "PRICE FIRST: the model's own first-order bits, ranked ascending per pair",
        "captured_rows_are_the_charged_rows": control,
        "estimator_scope": (
            "a RANKING, never a charge; every candidate emitted here is charged by a real "
            "600-frame sheet encode before any credit is measured"
        ),
        "pairs_with_candidates": len(per_pair),
        "pairs_without": [c["pair"] for c in census if c["cells_available"] == 0],
        "proposals": int(sum(len(v) for v in per_pair.values())),
        "singles": int(sum(1 for v in per_pair.values() for e in v
                           if e["tokens_changed"] == 1)),
        "combos": int(sum(1 for v in per_pair.values() for e in v
                          if e["tokens_changed"] == 2)),
        "combo_neighbour_fraction": float(np.mean([
            1.0 if e.get("partner_is_neighbour") else 0.0
            for v in per_pair.values() for e in v if e["tokens_changed"] == 2
        ])) if any(e["tokens_changed"] == 2 for v in per_pair.values() for e in v) else None,
        "cheapest_first_order_bits": {
            "pairs": int(cheapest.size),
            "min": float(cheapest.min()) if cheapest.size else None,
            "median": float(np.median(cheapest)) if cheapest.size else None,
            "mean": float(cheapest.mean()) if cheapest.size else None,
            "max": float(cheapest.max()) if cheapest.size else None,
        },
        "per_pair_single": int(args.per_pair_single),
        "min_separation_chebyshev": int(args.min_separation),
        "hold_fixed_every": int(args.hold_fixed_every),
        "held_fixed_price_control_pairs": int(sum(1 for v in per_pair.values() if len(v) == 1)),
        "negative_price_pairs": int(sum(1 for c in census
                                        if c["cheapest_bits"] is not None
                                        and c["cheapest_bits"] < 0.0)),
        "per_pair_combo": int(args.per_pair_combo),
        "combo_radius": int(args.combo_radius),
        "census": census,
    }
    (out_dir / "CANDIDATES.json").write_text(json.dumps(
        {"report": {k: v for k, v in report.items() if k != "census"},
         "per_pair": {str(k): v for k, v in sorted(per_pair.items())}},
        indent=1, sort_keys=True))
    (out_dir / "CANDIDATE_CENSUS.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "census"},
                     indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: sheets -- pd4's sheet rule over PRICE-FIRST candidates (no credit exists yet)
# ----------------------------------------------------------------------------------


def cmd_sheets(args) -> int:
    """Sheet k carries each pair's rank-k candidate, or its last when it has fewer.

    This is pd4's rule verbatim (identical edit density across sheets, so a pair held fixed
    measures the price's own noise floor).  The ONE difference is the ranking that feeds it:
    pd4 ranked by modelled benefit, which required a realized credit; this arm ranks by the
    model's own first-order bits, which requires nothing but the coder.
    """
    payload = json.loads(Path(args.candidates).read_text())
    per_pair = {int(k): v for k, v in payload["per_pair"].items()}
    source = np.load(args.control_npz)
    planes = {int(k): source[k] for k in source.files}
    if len(planes) != N_PAIRS:
        raise Pd6Error(f"control field has {len(planes)} planes, expected {N_PAIRS}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_count = min(int(args.sheets), max(len(v) for v in per_pair.values()))
    manifest: list[dict[str, Any]] = []
    for k in range(sheet_count):
        field = {str(p): planes[p].copy() for p in range(N_PAIRS)}
        placed = 0
        for pair, entries in sorted(per_pair.items()):
            rank = min(k, len(entries) - 1)
            entry = entries[rank]
            plane = field[str(pair)]
            for r, c, old, new in row_edits(entry):
                if int(plane[r, c]) != old:
                    raise Pd6Error(
                        f"pair {pair} cell ({r},{c}) holds {int(plane[r, c])} in the control "
                        f"field but the candidate says {old}"
                    )
                plane[r, c] = new
            placed += 1
            manifest.append({
                "sheet": k, "pair": pair, "rank": rank, "held_fixed": rank != k,
                "edits": row_edits(entry),
                "tokens_changed": int(entry["tokens_changed"]),
                "family": entry["family"],
                "first_order_bits": float(entry["first_order_bits"]),
                "first_order_rank": int(entry["first_order_rank"]),
                "row_source": "ddm_pd6_price_first",
            })
        path = out_dir / f"{args.name_prefix}_{k:02d}.npz"
        np.savez_compressed(path, **field)
        written = np.load(path)
        for pair in range(N_PAIRS):
            if not np.array_equal(written[str(pair)], field[str(pair)]):
                raise Pd6Error(f"sheet {k} does not round-trip plane {pair}")
        print(json.dumps({"sheet": k, "path": str(path), "pairs_edited": placed,
                          "bytes": path.stat().st_size}), flush=True)

    report = {
        "schema": "ddm_pd6_price_sheets.v1",
        "axis": "[macOS-CPU advisory / scorer-free field construction]",
        "score_claim": False,
        "rule": ("sheet k carries each pair's rank-k PRICE-FIRST candidate, or its last when "
                 "it has fewer; edit density is identical across sheets and a pair with one "
                 "candidate is a held-fixed price control"),
        "ranking": "the model's own first-order bits, ascending -- no credit was consulted",
        "control_npz": str(args.control_npz),
        "sheets": sheet_count,
        "pairs_priced": len(per_pair),
        "proposals_priced": len({(e["pair"], tuple(tuple(x) for x in e["edits"]))
                                 for e in manifest}),
        "held_fixed_control_pairs": sorted(p for p, v in per_pair.items()
                                           if len(v) < sheet_count),
        "manifest": manifest,
    }
    (out_dir / "SHEETS.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"},
                     indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: pick -- the realization list, chosen by FIRST-ORDER rank inside each family
# ----------------------------------------------------------------------------------


def cmd_pick(args) -> int:
    """Emit the proposals to realize FIRST, before the sheet encodes have finished charging.

    Scheduling only: the sheets charge every candidate this arm emitted, and admission reads
    the REAL charge.  Realizing the first-order-cheapest member of each family while the
    encodes run costs nothing that the charge would not also have selected -- the two rails
    rank the same object -- and it buys two hours of wall clock.  Any cheap proposal the
    charge later reveals and this list missed is realized in a top-up wave.
    """
    payload = json.loads(Path(args.candidates).read_text())
    want = {f: int(n) for f, n in (item.split("=") for item in args.per_family.split(","))}
    out: list[dict[str, Any]] = []
    for pair, entries in sorted(((int(k), v) for k, v in payload["per_pair"].items())):
        taken: dict[str, int] = {}
        for entry in sorted(entries, key=lambda e: float(e["first_order_bits"])):
            family = entry["family"]
            if taken.get(family, 0) >= want.get(family, 0):
                continue
            taken[family] = taken.get(family, 0) + 1
            row = dict(entry)
            row["pair"] = pair
            out.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"proposals": len(out), "pairs": len({r["pair"] for r in out}),
                      "per_family": want, "out": str(args.out),
                      "by_family": {f: sum(1 for r in out if r["family"] == f)
                                    for f in sorted({r["family"] for r in out})}},
                     indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: price-merge -- pd4's merge, reading pd6's own sheet encodes
# ----------------------------------------------------------------------------------


def cmd_price_merge(args) -> int:
    original = pd4._encode_bits

    def rebound(rlc1_root, field: str, tag: str):
        name = field.replace("pd4sheet", "pd6sheet") if field.startswith("pd4sheet") else field
        return original(rlc1_root, name, tag)

    pd4._encode_bits = rebound
    try:
        return pd4.cmd_price_merge(args)
    finally:
        pd4._encode_bits = original


# ----------------------------------------------------------------------------------
# stage: cheap-half -- the charged cheap set, and the sensitivity at 5 / 6 / 8 bits
# ----------------------------------------------------------------------------------


def cmd_cheap_half(args) -> int:
    """Split the CHARGED proposals at the cheap-half threshold and report the sensitivity."""
    rows = [json.loads(line) for line in Path(args.priced).read_text().splitlines()
            if line.strip()]
    for row in rows:
        row["real_bits_per_token"] = (float(row["real_delta_bits"])
                                      / float(int(row["tokens_changed"])))
    thresholds = [float(t) for t in args.thresholds.split(",")]
    sensitivity = []
    for t in thresholds:
        cheap = [r for r in rows if r["real_bits_per_token"] <= t]
        pairs = {int(r["pair"]) for r in cheap}
        sensitivity.append({
            "threshold_bits_per_token": t,
            "proposals": len(cheap),
            "pairs": len(pairs),
            "tokens": int(sum(int(r["tokens_changed"]) for r in cheap)),
            "fraction_of_charged": len(cheap) / len(rows) if rows else 0.0,
            "median_bits_per_token": float(np.median(
                [r["real_bits_per_token"] for r in cheap])) if cheap else None,
        })
    chosen = float(args.threshold)
    cheap = [r for r in rows if r["real_bits_per_token"] <= chosen]
    # realize the cheapest --per-pair per pair, cheapest first
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in sorted(cheap, key=lambda r: r["real_bits_per_token"]):
        by_pair.setdefault(int(row["pair"]), []).append(row)
    selected = [r for pair, rows_in in sorted(by_pair.items())
                for r in rows_in[: int(args.per_pair)]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in
                           sorted(selected, key=lambda r: (int(r["pair"]),
                                                           r["real_bits_per_token"]))))
    bits = np.array([r["real_bits_per_token"] for r in rows], dtype=np.float64)
    report = {
        "schema": "ddm_pd6_cheap_half.v1",
        "axis": "[macOS-CPU advisory / EXACT bits through the shipped coder]",
        "score_claim": False,
        "charged_proposals": len(rows),
        "bits_per_token_over_all_charged": {
            "min": float(bits.min()), "p05": float(np.quantile(bits, 0.05)),
            "p25": float(np.quantile(bits, 0.25)), "median": float(np.median(bits)),
            "p75": float(np.quantile(bits, 0.75)), "max": float(bits.max()),
            "mean": float(bits.mean()),
        },
        "histogram_bits_per_token": {
            "edges": [-math.inf, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, math.inf],
            "counts": [int(c) for c in np.histogram(
                bits, bins=[-1e9, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 1e9])[0]],
        },
        "threshold_bits_per_token": chosen,
        "sensitivity": sensitivity,
        "selected_for_realization": len(selected),
        "selected_pairs": len({int(r["pair"]) for r in selected}),
        "per_pair_cap": int(args.per_pair),
        "out": str(out),
    }
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: realize -- CREDIT AFTERWARDS, on the candidates the coder already charged
# ----------------------------------------------------------------------------------


def cmd_realize(args) -> int:
    """Render, frozen-argmax seg census and jg5 carrier re-solve, per cheap proposal.

    The call sequence is pd4's ``cluster-search`` inner loop verbatim -- one render, one
    frozen argmax, one carrier re-solve, credited against the PAIR's own base and never
    summed from parts.  The only difference is where the proposal came from.
    """
    pd4.bind_move52(verify_raw=False, raw=args.raw)
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pp1_pose_actuation as pp1

    pp1.set_threads(args.threads)
    started = time.time()
    rows_in = [json.loads(line) for line in Path(args.proposals).read_text().splitlines()
               if line.strip()]
    shard = int(args.shard_index)
    count = int(args.shard_count)
    pairs_all = sorted({int(r["pair"]) for r in rows_in})
    mine = {p for i, p in enumerate(pairs_all) if i % count == shard}
    todo: dict[int, list[dict[str, Any]]] = {}
    for row in rows_in:
        if int(row["pair"]) in mine:
            todo.setdefault(int(row["pair"]), []).append(row)

    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    body = pp1.load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = pp1.build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_mean)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"realized_{shard}.jsonl"
    screen_path = out_dir / f"screen_{shard}.jsonl"
    done: set[tuple] = set()
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((int(row["pair"]),
                          tuple(tuple(int(x) for x in e) for e in row["edits"])))
    if args.resume and screen_path.exists():
        for line in screen_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("refused_by_seg_screen"):
                    done.add((int(row["pair"]),
                              tuple(tuple(int(x) for x in e) for e in row["edits"])))
    handle = rows_path.open("a")
    screen_handle = screen_path.open("a")
    counters = {"pairs": 0, "screened": 0, "refined": 0, "seg_refused": 0, "skipped_done": 0}

    for pair in sorted(todo):
        # a pair whose every proposal is already on disk costs a render, an argmax and a
        # batch-1 pose evaluation before the loop can discover that; skip it outright so a
        # resume or a top-up wave does not re-pay the per-pair base cost 600 times
        if all((pair, tuple(tuple(int(x) for x in e) for e in row["edits"])) in done
               for row in todo[pair]):
            counters["skipped_done"] += len(todo[pair])
            continue
        fe1.restore_pair_codes(body, pair)
        base_plane = body.tokens[pair].copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        gap = abs(d_pose_base - float(base[pair]))
        if gap > args.base_band_abs:
            raise Pd6Error(
                f"pair {pair} batch-1 base {d_pose_base:.12e} is {gap:.3e} from the n600 base "
                f"{float(base[pair]):.12e}, outside the MEASURED band {args.base_band_abs:.3e}"
            )
        counters["pairs"] += 1
        for row in todo[pair]:
            edits = [tuple(int(x) for x in e) for e in row["edits"]]
            key = (pair, tuple(edits))
            if key in done:
                counters["skipped_done"] += 1
                continue
            for r, c, old, new in edits:
                if int(base_plane[r, c]) != old:
                    raise Pd6Error(
                        f"pair {pair} cell ({r},{c}) holds {int(base_plane[r, c])} but the "
                        f"proposal says {old}"
                    )
                body.tokens[pair][r, c] = new
            frame = fe1.render_pair(body, pair)
            moved_flips = fe1.flips_pair(
                jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair)
            counters["screened"] += 1
            d_cells = moved_flips - base_flips
            entry = dict(row)
            entry.update({"base_flips": base_flips, "flips": moved_flips,
                          "d_cells": int(d_cells), "edits": [list(e) for e in edits],
                          "interior": True})
            if d_cells > int(args.max_cells):
                for r, c, old, _new in edits:
                    body.tokens[pair][r, c] = old
                counters["seg_refused"] += 1
                entry["refused_by_seg_screen"] = True
                screen_handle.write(json.dumps(entry) + "\n")
                screen_handle.flush()
                continue
            overlay = pp1.MemoryOverlayRaw(raw, {2 * pair + 1: frame[0]})
            moved_inst = br1.Instrument(
                base_inst.state, overlay, base_inst.targets, base_inst.posenet,
                base_inst.blow, base_inst.gram, base_inst.bmat,
            )
            stale = float(br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0])
            refined = jg5.refine_pair(
                moved_inst, pair, live_codes[pair], dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
            )
            for r, c, old, _new in edits:
                body.tokens[pair][r, c] = old
            resolved = float(refined["final_d_pose"])
            counters["refined"] += 1
            credit = resolved - d_pose_base
            entry.update({
                "d_pose_base": d_pose_base,
                "d_pose_stale": stale,
                "d_pose_resolved": resolved,
                "resolved_over_base": resolved / d_pose_base if d_pose_base > 0 else math.nan,
                "carrier_codes": [int(c) for c in refined["codes"]],
                "base_gap_abs_vs_n600": gap,
                "credit_d_pose": credit,
                "dS_seg": int(d_cells) * seg_cell,
                "dS_pose_resolved": credit * pose_unit,
                "row_source": "ddm_pd6_realize",
            })
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            screen_handle.write(json.dumps({k: entry[k] for k in
                                            ("pair", "edits", "d_cells")}
                                           | {"refused_by_seg_screen": False}) + "\n")
            screen_handle.flush()
            price = row.get("real_bits_per_token")
            if price is None:
                price = float(row["first_order_bits"]) / float(int(row["tokens_changed"]))
            print(f"pair {pair} {entry['family']} cells {d_cells:+d} bits/token "
                  f"{float(price):.3f} resolved {resolved:.4e} "
                  f"credit {credit:+.3e}", flush=True)
        np.testing.assert_array_equal(body.tokens[pair], base_plane)
    handle.close()
    screen_handle.close()
    (out_dir / f"REALIZE_{shard}.json").write_text(json.dumps({
        "schema": "ddm_pd6_realize.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "proposal_origin": "PRICE FIRST: the coder charged these before any credit was known",
        "shard_index": shard, "shard_count": count,
        "pairs": sorted(todo), "counters": counters,
        "max_cells": int(args.max_cells), "refine_rounds": int(args.outer_rounds),
        "rows_path": str(rows_path), "screen_path": str(screen_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path), "counters": counters}))
    return 0


# ----------------------------------------------------------------------------------
# stage: credit-report + enrich -- the credit histogram, and the rows the admission reads
# ----------------------------------------------------------------------------------


def _realized_rows(spec: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in spec:
        path = Path(item)
        files = sorted(path.glob("realized_*.jsonl")) if path.is_dir() else [path]
        for file in files:
            if file.name.startswith("._") or not file.exists():
                continue
            for line in file.read_text().splitlines():
                if line.strip():
                    out.append(json.loads(line))
    return out


def cmd_credit(args) -> int:
    """The credit histogram on the CHEAP set: how many pairs the price-first pool can pay.

    Reads the ENRICHED rows (the charge joined to the credit), because every economic column
    here is charge-against-credit and a row without its real charge has no economics.
    """
    rows = _realized_rows(args.realized)
    missing = [r for r in rows if "real_delta_bits" not in r]
    if missing:
        raise Pd6Error(
            f"{len(missing)} of {len(rows)} rows carry no real charge; run `enrich` first and "
            "point --realized at its output"
        )
    base = np.load(args.base_pose)
    pose_unit = pose_s_per_pair_unit(float(base.mean()))
    seg_cell = seg_s_per_cell()
    for row in rows:
        row["benefit_S"] = (-(float(row["credit_d_pose"]) * pose_unit)
                            - float(row["d_cells"]) * seg_cell)
        row["dS_rate_real"] = (float(row["real_delta_bits"]) / 8.0) * S_PER_BYTE
        row["dS_modelled_real_price"] = -row["benefit_S"] + row["dS_rate_real"]
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair"]), []).append(row)
    best = {p: min(v, key=lambda r: r["dS_modelled_real_price"]) for p, v in by_pair.items()}
    positive = [p for p, v in by_pair.items()
                if any(float(r["credit_d_pose"]) < 0.0 for r in v)]
    paying = [p for p, r in best.items() if r["dS_modelled_real_price"] < 0.0]
    credits = np.array([float(r["credit_d_pose"]) for r in rows], dtype=np.float64)
    cells = np.array([int(r["d_cells"]) for r in rows], dtype=np.int64)
    bits = np.array([float(r["real_bits_per_token"]) for r in rows], dtype=np.float64)
    report = {
        "schema": "ddm_pd6_credit.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]",
        "score_claim": False,
        "pose_solver": ("ddm_jg5_pose_resolve_on_edited_renders.refine_pair at PoseNet batch 1 "
                        "on the moved render"),
        "realized_rows": len(rows),
        "pairs_realized": len(by_pair),
        "pairs_with_any_positive_resolved_pose_credit": len(positive),
        "pairs_whose_best_row_pays_at_its_own_real_price": len(paying),
        "credit_d_pose": {
            "improves": int((credits < 0).sum()),
            "worsens": int((credits > 0).sum()),
            "median": float(np.median(credits)), "mean": float(credits.mean()),
            "best": float(credits.min()), "worst": float(credits.max()),
        },
        "seg_cells": {
            "repaid": int((cells < 0).sum()), "neutral": int((cells == 0).sum()),
            "cost_one": int((cells == 1).sum()), "cost_two": int((cells == 2).sum()),
            "median": float(np.median(cells)),
        },
        "bits_per_token_realized": {
            "median": float(np.median(bits)), "mean": float(bits.mean()),
            "min": float(bits.min()), "max": float(bits.max()),
        },
        "by_family": {
            fam: {
                "rows": int(sum(1 for r in rows if r["family"] == fam)),
                "improves": int(sum(1 for r in rows if r["family"] == fam
                                    and float(r["credit_d_pose"]) < 0.0)),
                "median_bits_per_token": float(np.median(
                    [r["real_bits_per_token"] for r in rows if r["family"] == fam])),
                "median_credit": float(np.median(
                    [float(r["credit_d_pose"]) for r in rows if r["family"] == fam])),
            } for fam in sorted({r["family"] for r in rows})
        },
        "modelled_net_dS_if_every_paying_pair_is_taken": float(
            sum(best[p]["dS_modelled_real_price"] for p in paying)),
        "pose_unit_S_per_pair_d_pose": pose_unit,
        "seg_cell_S": seg_cell,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def cmd_enrich(args) -> int:
    """Attach the realized credit to the CHARGED rows, so pd4/pd5's admission can read them."""
    priced = [json.loads(line) for line in Path(args.priced).read_text().splitlines()
              if line.strip()]
    realized = {(int(r["pair"]), tuple(tuple(int(x) for x in e) for e in r["edits"])): r
                for r in _realized_rows(args.realized)}
    out: list[dict[str, Any]] = []
    for row in priced:
        key = (int(row["pair"]), tuple(tuple(int(x) for x in e) for e in row["edits"]))
        got = realized.get(key)
        if got is None:
            continue
        merged = dict(row)
        for field in ("d_cells", "base_flips", "flips", "d_pose_base", "d_pose_stale",
                      "d_pose_resolved", "resolved_over_base", "carrier_codes",
                      "base_gap_abs_vs_n600", "credit_d_pose", "dS_seg", "dS_pose_resolved"):
            merged[field] = got[field]
        merged["real_bits_per_token"] = (float(merged["real_delta_bits"])
                                         / float(int(merged["tokens_changed"])))
        merged["row_source"] = "ddm_pd6_price_first"
        out.append(merged)
    out.sort(key=lambda r: (int(r["pair"]), float(r["real_delta_bits"])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"rows": len(out), "pairs": len({int(r['pair']) for r in out}),
                      "priced_in": len(priced), "realized_in": len(realized),
                      "out": str(args.out)}, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    bind = sub.add_parser("bind", help="re-point every stale module at move 52")
    bind.add_argument("--raw", required=True)
    bind.add_argument("--verify-raw", action="store_true")
    bind.add_argument("--control-archive", default=None)
    bind.add_argument("--out", required=True)
    bind.set_defaults(func=cmd_bind)

    cap = sub.add_parser("capture", help="one real encode; record the model's own price rows")
    cap.add_argument("--field", default="control")
    cap.add_argument("--tag", default="capture")
    cap.add_argument("--argmax", required=True)
    cap.add_argument("--interior-radius", type=int, default=3)
    cap.add_argument("--keep-per-frame", type=int, default=1024)
    cap.add_argument("--out-dir", required=True)
    cap.set_defaults(func=cmd_capture)

    cand = sub.add_parser("candidates", help="the CHEAP HALF, per pair, single and 2-token")
    cand.add_argument("--frames", required=True)
    cand.add_argument("--control-npz", required=True)
    cand.add_argument("--per-pair-single", type=int, default=4)
    cand.add_argument("--per-pair-combo", type=int, default=2)
    cand.add_argument("--combo-radius", type=int, default=1)
    cand.add_argument("--min-separation", type=int, default=4)
    cand.add_argument("--hold-fixed-every", type=int, default=10)
    cand.add_argument("--encode-json", default=None,
                      help="the capture encode's ENCODE.json, for the charged-rows control")
    cand.add_argument("--out-dir", required=True)
    cand.set_defaults(func=cmd_candidates)

    sh = sub.add_parser("sheets", help="pd4's sheet rule over the price-first candidates")
    sh.add_argument("--candidates", required=True)
    sh.add_argument("--control-npz", required=True)
    sh.add_argument("--sheets", type=int, default=6)
    sh.add_argument("--name-prefix", default="sheet")
    sh.add_argument("--out-dir", required=True)
    sh.set_defaults(func=cmd_sheets)

    pk = sub.add_parser("pick", help="the realization list, by first-order rank per family")
    pk.add_argument("--candidates", required=True)
    pk.add_argument("--per-family", default="price_single=1,price_pair=1")
    pk.add_argument("--out", required=True)
    pk.set_defaults(func=cmd_pick)

    pm = sub.add_parser("price-merge", help="pd4's merge, reading pd6's own sheet encodes")
    pm.add_argument("--sheets", required=True)
    pm.add_argument("--rlc1-root", required=True)
    pm.add_argument("--control-tag", default="primary")
    pm.add_argument("--sheet-tag", default="primary")
    pm.add_argument("--cross-check-field", default=None)
    pm.add_argument("--cross-check-sheets", default=None)
    pm.add_argument("--raw", required=True)
    pm.add_argument("--out-dir", required=True)
    pm.set_defaults(func=cmd_price_merge)

    ch = sub.add_parser("cheap-half", help="split the CHARGED rows at the cheap threshold")
    ch.add_argument("--priced", required=True)
    ch.add_argument("--threshold", type=float, default=6.0)
    ch.add_argument("--thresholds", default="5,6,8")
    ch.add_argument("--per-pair", type=int, default=2)
    ch.add_argument("--out", required=True)
    ch.add_argument("--report", required=True)
    ch.set_defaults(func=cmd_cheap_half)

    rz = sub.add_parser("realize", help="CREDIT AFTERWARDS on the cheap charged proposals")
    rz.add_argument("--proposals", required=True)
    rz.add_argument("--base-pose", required=True)
    rz.add_argument("--base-band-abs", type=float, default=BASE_BAND_ABS)
    rz.add_argument("--raw", required=True)
    rz.add_argument("--max-cells", type=int, default=2)
    rz.add_argument("--outer-rounds", type=int, default=40)
    rz.add_argument("--max-gn-iterations", type=int, default=400)
    rz.add_argument("--threads", type=int, default=2)
    rz.add_argument("--shard-index", type=int, default=0)
    rz.add_argument("--shard-count", type=int, default=1)
    rz.add_argument("--resume", action="store_true")
    rz.add_argument("--out-dir", required=True)
    rz.set_defaults(func=cmd_realize)

    cr = sub.add_parser("credit", help="the credit histogram on the cheap set")
    cr.add_argument("--realized", nargs="+", required=True)
    cr.add_argument("--base-pose", required=True)
    cr.add_argument("--out", required=True)
    cr.set_defaults(func=cmd_credit)

    en = sub.add_parser("enrich", help="charged rows + realized credit -> admission rows")
    en.add_argument("--priced", required=True)
    en.add_argument("--realized", nargs="+", required=True)
    en.add_argument("--out", required=True)
    en.set_defaults(func=cmd_enrich)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
