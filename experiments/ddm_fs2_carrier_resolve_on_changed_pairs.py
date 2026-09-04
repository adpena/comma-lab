#!/usr/bin/env python3
"""ddm_fs2 -- re-solve the pose carrier on the pairs whose frame-0 SELECTOR moved.

WHY THIS ARM EXISTS
-------------------
``ddm_fs1`` re-selected the frame-0 selector on 21 of 600 pairs and bought the
twenty-fourth pointer move (``S 0.14786319521362173`` @ 180,022 B,
``[contest-CUDA T4 n600]``).  It optimised the selector against a FIXED carrier,
and said so (``fs1`` Sec 8): *"re-solving each changed pair's carrier against its
new frame 0 can only help, and neither arm did it."*

The staleness is real and mechanical.  ``ddm_up2.render_frame0`` is *carrier
render THEN selector op*, so a pair whose selector op changed is being scored on
a frame the carrier codes were never fitted for.  Those 21 code rows are the
only rows in the body that are demonstrably off their own operating point.

WHAT IS REUSED, VERBATIM -- no mechanism reduction
--------------------------------------------------
* the solver is ``ddm_jg5.refine_pair`` -- br1's damped Gauss-Newton on the
  shipped 12-dim basis and the shipped signed-int12 lattice, alternated with the
  +-2 polish under jg5's DERIVED materiality stopping rule.  ``ddm_pr1`` Sec 8.1
  established this is the OPTIMAL FORM here (``ddm_up2``'s +-2-only radius is a
  truncation the shipped residual exceeds on 100.0% of pairs);
* the instrument is ``ddm_pr1.build_instrument`` -- the SAME assembly that
  reproduced the contest-CUDA pose leg to 0.068%.  The only thing this module
  changes is the ``archive_sha256`` it is willing to gate on, because the body
  under test is fs1's candidate B and not the afr1 body pr1 hardcodes;
* the byte price is ``ddm_up2.price_full_resolve_bytes``, which refuses to
  return a delta unless its own control reproduces the SHIPPED Rice payload.

THE ONE THING THIS MODULE ADDS
------------------------------
An explicit PAIR LIST.  ``pr1``'s ``solve`` takes a COUNT and a seed, which is
right for an unbiased population estimate and wrong here: the 21 pairs are not a
sample, they are the exact set whose frame 0 moved, and they are DERIVED by
diffing the two bodies' own selector vectors rather than typed.

AUTHORITY
---------
Frozen CPU-torch PoseNet on DALI-lineage GT.  ``[macOS-CPU advisory]``,
``score_claim=false``, ``promotable=false``.  Only ``upstream/evaluate.py`` on
contest hardware, on the exact shipped bytes, is a score.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

#: The body this arm re-solves: fs1's candidate B, the twenty-fourth pointer move.
#: Named and sha-gated, never assumed -- a delta against a superseded body
#: double-counts a banked gain
#: ([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]).
FS1_ARCHIVE_SHA256 = (
    "50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf"
)
FS1_ARCHIVE_BYTES = 180_022

#: The afr1 body fs1 spliced FROM: the control leg of every "is this staleness or
#: is this slack?" question this arm asks.
BASE_ARCHIVE_SHA256 = (
    "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
)
BASE_ARCHIVE_BYTES = 180_002

#: fs1's contest-CUDA T4 n600 receipt legs
#: (``.omx/research/ddm_fs1_pointer_move_24_20260904.md``, recomputed from
#: components per #877 -- never the rounded ``Final score`` display).
FS1_D_SEG_T4 = 0.00020139
FS1_D_POSE_T4 = 6.17e-06
FS1_SCORE_T4 = 0.14786319521362173

#: fs1's own n600 batch-8 advisory row for candidate B -- the operating point the
#: materiality floor is DERIVED at.  Evaluating the floor at a higher (staler)
#: mean would raise it by sqrt(stale/target) and stop the solver early.
FS1_D_POSE_ADVISORY_N600_BATCH8 = 6.169860284911831e-06

N_PAIRS = 600
CARRIER_DIM = 12
BYTE_TO_SCORE = 25.0 / 37_545_489.0


class Fs2Error(RuntimeError):
    """A ddm_fs2 precondition failed.  Fail closed, never approximate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def rate_leg(archive_bytes: int) -> float:
    return 25.0 * archive_bytes / 37_545_489.0


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + pose_leg(d_pose) + rate_leg(archive_bytes)


# --------------------------------------------------------------------------
# The pair set -- DERIVED from the two bodies, never typed.
# --------------------------------------------------------------------------


def selector_choices_of(runtime: Path, expect_sha256: str | None) -> np.ndarray:
    """The per-pair selector vector the receiver would read out of a runtime."""
    import ddm_up2_shipping_pose_solve as up2

    runtime = Path(runtime)
    observed = sha256_file(runtime / "archive.zip")
    if expect_sha256 and observed != expect_sha256:
        raise Fs2Error(
            f"runtime archive sha256 {observed} != expected {expect_sha256}; "
            "refusing to read a selector out of an unidentified body"
        )
    state = up2.load_carrier_state(runtime, verify_archive=False)
    return np.asarray(state.selector_choices, dtype=np.int64)


def changed_selector_pairs(base_runtime: Path, candidate_runtime: Path, *,
                           base_sha256: str | None = BASE_ARCHIVE_SHA256,
                           candidate_sha256: str | None = FS1_ARCHIVE_SHA256,
                           ) -> tuple[np.ndarray, dict[str, Any]]:
    """Pairs whose frame-0 op differs between two bodies, with a receipt.

    This is the whole scope of the arm and it is a DIFF, not a selection rule:
    a pair is in iff the receiver would apply a different pixel op to its frame
    0 than it did before.  Pairs that merely stayed active at the same mode are
    NOT stale and are excluded, or the "re-solve on stale rows" claim would be
    measuring rows that are not stale.
    """
    base = selector_choices_of(base_runtime, base_sha256)
    candidate = selector_choices_of(candidate_runtime, candidate_sha256)
    if base.shape != (N_PAIRS,) or candidate.shape != (N_PAIRS,):
        raise Fs2Error(
            f"selector vectors are {base.shape} / {candidate.shape}, expected ({N_PAIRS},)"
        )
    pairs = np.flatnonzero(base != candidate).astype(np.int64)
    receipt = {
        "base_runtime": str(base_runtime),
        "candidate_runtime": str(candidate_runtime),
        "base_active": np.flatnonzero(base).tolist(),
        "candidate_active": np.flatnonzero(candidate).tolist(),
        "changed_pairs": pairs.tolist(),
        "changed_count": int(pairs.size),
        "transitions": [
            {"pair": int(p), "from_mode": int(base[p]), "to_mode": int(candidate[p])}
            for p in pairs
        ],
        "base_choices_sha256": sha256_array(base.astype(np.uint8)),
        "candidate_choices_sha256": sha256_array(candidate.astype(np.uint8)),
    }
    return pairs, receipt


def parse_pairs_argument(text: str | None, derived: np.ndarray) -> np.ndarray:
    """Explicit override, else the derived diff.  An override must be a SUBSET.

    A pair outside the diff is not stale, so solving it would be measuring the
    shipped carrier's own slack under a re-solve label -- exactly the
    slack-vs-repair confusion ``ddm_pr1`` Sec 8.1 exists to separate.  Use the
    ``control`` mode for that question instead.
    """
    if text is None:
        return derived
    wanted = np.array(sorted({int(t) for t in text.replace(",", " ").split()}),
                      dtype=np.int64)
    extra = sorted(set(wanted.tolist()) - set(derived.tolist()))
    if extra:
        raise Fs2Error(
            f"--pairs {extra} are not in the selector diff; they are not stale. "
            "Measure the shipped carrier's slack with the control leg, not here."
        )
    return wanted


# --------------------------------------------------------------------------
# mode=solve -- jg5.refine_pair on an explicit pair list, resumable by pair.
# --------------------------------------------------------------------------


def load_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if rows_path.is_file():
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[int(row["pair"])] = row
    return done


def run_solve(args) -> int:
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pr1_pose_resolve_on_renderer_change as pr1

    candidate_runtime = Path(args.runtime)
    derived, diff_receipt = changed_selector_pairs(
        Path(args.base_runtime), candidate_runtime,
        base_sha256=args.base_archive_sha256 or None,
        candidate_sha256=args.expect_archive_sha256 or None,
    )
    pairs = parse_pairs_argument(args.pairs, derived)
    if pairs.size == 0:
        raise Fs2Error("the selector diff is empty; there is nothing stale to re-solve")

    instrument, meta = pr1.build_instrument(
        runtime=candidate_runtime, gt_cache=Path(args.gt_cache), axis=args.axis,
        renderer_source=Path(args.renderer), tokens_path=Path(args.tokens),
        archive_sha256=args.expect_archive_sha256 or None,
    )
    dd_threshold = jg5.materiality_dd_threshold(args.materiality_operating_point)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    done = load_done(rows_path)
    started = time.time()
    print(json.dumps({
        "label": args.label,
        "body": meta["archive_sha256"],
        "pairs": pairs.tolist(),
        "materiality_operating_point_d_pose": args.materiality_operating_point,
        "dd_threshold": dd_threshold,
        "already_done": sorted(int(p) for p in done if int(p) in set(pairs.tolist())),
    }, indent=2), flush=True)

    with rows_path.open("a", encoding="utf-8") as stream:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            start_codes = instrument.state.codes[int(pair)]
            row = jg5.refine_pair(
                instrument, int(pair), start_codes, dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds,
                max_gn_iterations=args.max_gn_iterations,
            )
            row["solver"] = "jg5"
            row["body_archive_sha256"] = meta["archive_sha256"]
            row["selector_mode"] = int(
                np.asarray(instrument.state.selector_choices)[int(pair)]
            )
            done[int(pair)] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            elapsed = time.time() - started
            remaining = sum(1 for p in pairs if int(p) not in done)
            print(
                f"[{len(pairs) - remaining}/{len(pairs)}] pair={int(pair)} "
                f"mode={row['selector_mode']} "
                f"start={row['start_d_pose']:.6e} final={row['final_d_pose']:.6e} "
                f"recov={row['start_d_pose'] / max(row['final_d_pose'], 1e-30):.4g}x "
                f"dcoord={row['changed_coordinates']} stop={row['stop_reason']} "
                f"elapsed={elapsed / 60:.1f}m "
                f"eta={elapsed / max(1, position + 1) * remaining / 60:.1f}m",
                flush=True,
            )

    ordered = [done[int(p)] for p in pairs if int(p) in done]
    total_start = float(sum(r["start_d_pose"] for r in ordered))
    total_final = float(sum(r["final_d_pose"] for r in ordered))
    summary = {
        "schema": "tac.ddm_fs2.solve.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "solver": "jg5",
        "solver_reference_form": (
            "ddm_jg5.refine_pair (br1 damped Gauss-Newton on the shipped 12-dim "
            "basis and int12 lattice, +-2 polish, jg5 derived materiality stop)"
        ),
        "instrument": meta,
        "selector_diff": diff_receipt,
        "pairs_solved": [int(p) for p in pairs],
        "materiality_operating_point_d_pose": args.materiality_operating_point,
        "dd_threshold": dd_threshold,
        "pairs": len(ordered),
        "pairs_with_any_code_change": int(
            sum(1 for r in ordered if r["changed_coordinates"] > 0)
        ),
        "pairs_improved": int(
            sum(1 for r in ordered if r["final_d_pose"] < r["start_d_pose"])
        ),
        "total_start_d_pose": total_start,
        "total_final_d_pose": total_final,
        "total_gain_d_pose": total_start - total_final,
        # The score consequence of this SUBSET, carried as the n600 MEAN it will
        # enter S as.  Reporting the subset's own mean would be a different
        # number than the one the score sees ([[m88]] on the aggregation axis).
        "n600_mean_gain_d_pose": (total_start - total_final) / N_PAIRS,
        "stop_reasons": {
            reason: int(sum(1 for r in ordered if r.get("stop_reason") == reason))
            for reason in sorted({r.get("stop_reason", "unknown") for r in ordered})
        },
        "total_changed_coordinates": int(
            sum(r["changed_coordinates"] for r in ordered)
        ),
        "total_evaluations": int(sum(r["evaluations"] for r in ordered)),
        "rows_path": str(rows_path),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "label", "pairs", "pairs_with_any_code_change", "pairs_improved",
        "total_gain_d_pose", "n600_mean_gain_d_pose", "total_changed_coordinates",
        "stop_reasons", "elapsed_seconds")}, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
# mode=codes -- merge solved rows into a 600x12 table and PRICE it.
# --------------------------------------------------------------------------


def run_codes(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    runtime = Path(args.runtime)
    observed = sha256_file(runtime / "archive.zip")
    if args.expect_archive_sha256 and observed != args.expect_archive_sha256:
        raise Fs2Error(
            f"runtime archive sha256 {observed} != {args.expect_archive_sha256}; "
            "the unsolved rows would be another body's carrier codes"
        )
    state = up2.load_carrier_state(runtime, verify_archive=False)
    shipped = state.codes.astype(np.int32)
    codes = shipped.copy()

    merged: dict[int, list[int]] = {}
    row_gain: dict[int, dict[str, float]] = {}
    for path in args.rows:
        for pair, row in load_done(Path(path)).items():
            merged[int(pair)] = row["codes"]
            row_gain[int(pair)] = {
                "start_d_pose": float(row["start_d_pose"]),
                "final_d_pose": float(row["final_d_pose"]),
                "changed_coordinates": int(row["changed_coordinates"]),
                "stop_reason": row.get("stop_reason", "unknown"),
            }
    if not merged:
        raise Fs2Error("no solved rows found; nothing to merge")

    # Adopt only rows that IMPROVED.  A solver row that ended where it started
    # writes the shipped codes back and costs nothing; a row that ended WORSE
    # (the solver refuses those, but the merge must not depend on that) would
    # buy negative pose for positive bytes.
    adopted: list[int] = []
    for pair, row_codes in sorted(merged.items()):
        gain = row_gain[pair]
        if args.adopt_all or gain["final_d_pose"] < gain["start_d_pose"]:
            codes[pair] = np.asarray(row_codes, dtype=np.int32)
            adopted.append(pair)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, codes)

    price, base_codes = up2.price_full_resolve_bytes(runtime, codes)
    if not np.array_equal(base_codes, shipped):
        raise Fs2Error(
            "the pricer's base codes differ from the runtime's own carrier codes; "
            "the byte delta would be against a different body"
        )
    delta_bytes = int(price["delta_bytes"])
    total_gain = float(sum(
        row_gain[p]["start_d_pose"] - row_gain[p]["final_d_pose"] for p in adopted
    ))
    n600_mean_gain = total_gain / N_PAIRS
    base_mean = float(args.base_d_pose)
    candidate_mean = base_mean - n600_mean_gain
    record = {
        "schema": "tac.ddm_fs2.codes.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "path": str(out),
        "rows_merged_from": [str(p) for p in args.rows],
        "runtime_archive_sha256": observed,
        "pairs_in_rows": sorted(merged),
        "pairs_adopted": adopted,
        "adopt_all": bool(args.adopt_all),
        "per_pair": {str(p): row_gain[p] for p in sorted(merged)},
        "carrier_rice_price": price,
        "delta_bytes": delta_bytes,
        "delta_score_rate": delta_bytes * BYTE_TO_SCORE,
        "shipped_codes_sha256": sha256_array(shipped),
        "codes_sha256": sha256_array(codes),
        "changed_pairs": int((codes != shipped).any(axis=1).sum()),
        "changed_coordinates": int((codes != shipped).sum()),
        # PROJECTED, from the per-pair solver rows; the honest number is the
        # n600 batch-8 re-measure through the built archive, not this.
        "projection": {
            "base_d_pose_n600": base_mean,
            "base_d_pose_source": args.base_d_pose_source,
            "projected_d_pose_n600": candidate_mean,
            "total_gain_d_pose_over_adopted_pairs": total_gain,
            "n600_mean_gain_d_pose": n600_mean_gain,
            "delta_score_pose": pose_leg(candidate_mean) - pose_leg(base_mean),
            "delta_score_rate": delta_bytes * BYTE_TO_SCORE,
            "delta_score_seg": 0.0,
            "net_delta_score": (
                pose_leg(candidate_mean) - pose_leg(base_mean)
                + delta_bytes * BYTE_TO_SCORE
            ),
            "note": (
                "PROJECTED from per-pair solver rows at batch 1 against an n600 "
                "batch-8 mean; a cross-shape composition. Only the re-measure is "
                "the arm's own advisory number."
            ),
        },
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in (
        "pairs_adopted", "delta_bytes", "delta_score_rate", "changed_pairs",
        "changed_coordinates", "projection")}, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
# mode=build -- splice the re-solved codes into archive.zip, the receiver's way.
# --------------------------------------------------------------------------

#: The F26 semantic-joint body's OWN container shape.  ``ddm_fs1`` Sec 3.3
#: MEASURED it: q=9/lgwin=16 and q=9/lgwin=24 both give the shipped size, q=10
#: and q=11 give 2 bytes MORE.  ``ddm_up3``'s module defaults (q=11/lgwin=24)
#: are a DIFFERENT generation's shipped shape and would break the byte-identity
#: control here -- which is exactly why they are named rather than inherited.
BROTLI_QUALITY = 9
BROTLI_LGWIN = 16

#: Encoder-only alternatives, searched for reporting.  The shipped shape is
#: index 0 and wins every tie, so the identity control stays exact.
CONTAINER_OPTIONS: tuple[tuple[int, int], ...] = (
    (BROTLI_QUALITY, BROTLI_LGWIN),
    (9, 24),
    (10, 16),
    (10, 24),
    (11, 16),
    (11, 24),
)


def _shipped_container(body) -> tuple[tuple[bool, int, int], ...]:
    return ((bool(body.ck2_carrier), BROTLI_QUALITY, BROTLI_LGWIN),)


def run_build(args) -> int:
    import ddm_fs1_frame0_selector_reselection as fs1
    import ddm_up3_carrier_splice as up3

    runtime = Path(args.runtime)
    archive_path = runtime / "archive.zip"
    observed = sha256_file(archive_path)
    if args.expect_archive_sha256 and observed != args.expect_archive_sha256:
        raise Fs2Error(
            f"runtime archive sha256 {observed} != {args.expect_archive_sha256}; "
            "refusing to splice into an unidentified body"
        )
    shipped_bytes = archive_path.read_bytes()
    # up3's own sha gate names ITS generation's pointer body; the gate that
    # matters here is the one above, against the body this arm was handed.
    body = up3.parse_shipped_body(runtime, verify_sha=False)

    # CONTROL 1 -- container identity.  Rebuilding the shipped body with the
    # SHIPPED codes must reproduce archive.zip bit for bit, or the byte delta
    # this module reports would be a mixture of the codes and the container and
    # no delta would be attributable.
    identity_built = up3.build_archive(
        body, body.codes, runtime_dir=runtime,
        container_options=_shipped_container(body),
    )
    identity = {
        "control": "identity_shipped_codes_reproduce_this_body",
        "expected_sha256": observed,
        "observed_sha256": identity_built["archive_sha256"],
        "byte_identical": identity_built["archive_sha256"] == observed,
        "expected_bytes": len(shipped_bytes),
        "observed_bytes": identity_built["archive_size"],
        "packed_metadata_identical": identity_built["packed_metadata_identical"],
        "rice_payload_identical": identity_built["rice_payload_identical"],
        "rice_bits_shipped": identity_built["rice_bits"],
        "rice_ks_shipped": identity_built["rice_ks"],
        "container": identity_built["container"],
    }
    if not identity["byte_identical"]:
        raise Fs2Error(
            "the container identity control FAILED: re-encoding the shipped codes "
            "does not reproduce the shipped bytes, so no byte delta this module "
            "reports would be the carrier's. Refusing to build."
        )

    codes = np.load(Path(args.codes)).astype(np.int32)
    shipped_codes = np.asarray(body.codes, dtype=np.int32)
    if codes.shape != shipped_codes.shape:
        raise Fs2Error(f"codes {codes.shape} != shipped {shipped_codes.shape}")

    # OPTIONAL second variable: the frame-0 selector.  The alternation
    # (``ddm_fs1`` Sec 9.3) can move a selector label after the carrier is
    # re-solved, and a label change on an ALREADY-ACTIVE pair leaves both the
    # active count and the position set untouched, so the blob length -- and
    # therefore the archive -- is unchanged.  It is a byte-free pose move, but
    # only if it is proved rather than assumed, so the length is checked here.
    selector_change: dict[str, Any] | None = None
    choices: np.ndarray | None = None
    if args.selector_choices:
        from tac.semantic_pipeline.frame0_selector_codec import (
            STORED_PREFIX,
            encode_selector,
            stored_tail,
        )

        choices = np.load(Path(args.selector_choices)).astype(np.uint8)
        if choices.shape != (N_PAIRS,):
            raise Fs2Error(f"choices must be ({N_PAIRS},), got {choices.shape}")
        blob = encode_selector(choices)
        tail = stored_tail(blob)
        shipped_blob = STORED_PREFIX + bytes(body.body_tail)
        selector_change = {
            "shipped_blob_bytes": len(shipped_blob),
            "shipped_blob_sha256": hashlib.sha256(shipped_blob).hexdigest(),
            "candidate_blob_bytes": len(blob),
            "candidate_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "blob_delta_bytes": len(blob) - len(shipped_blob),
            "active_pairs": int(np.count_nonzero(choices)),
        }
        body = dataclasses.replace(body, body_tail=tail)

    if np.array_equal(codes, shipped_codes) and selector_change is None:
        raise Fs2Error(
            "the requested codes equal the shipped ones and no selector change was "
            "given; there is nothing to build"
        )

    built = up3.build_archive(
        body, codes, runtime_dir=runtime,
        container_options=_shipped_container(body),
    )
    candidate = built["archive_bytes"]

    # CONTROL 2 -- the no-op detector, through the SHIPPED receiver's own parse.
    # This is fs1's detector read in the OPPOSITE sense: fs1 changed the selector
    # and required the CAP1 carrier to be identical; this arm changes the carrier
    # and requires the selector -- and every other section -- to be identical.
    parsed = fs1.parse_back_parts(candidate, runtime)
    base_parsed = fs1.parse_back_parts(shipped_bytes, runtime)
    parsed_choices = parsed.pop("selector_choices")
    base_choices = base_parsed.pop("selector_choices")
    selector_moved = not np.array_equal(parsed_choices, base_choices)
    if selector_moved and selector_change is None:
        raise Fs2Error(
            "the written archive parses back to a DIFFERENT selector but no "
            "selector change was requested; the splice is not one-variable"
        )
    if selector_change is not None:
        assert choices is not None  # set together, one branch above
        if not np.array_equal(parsed_choices, choices):
            raise Fs2Error(
                "the written archive parses back to a selector that is not the "
                "requested one; refusing to return unverified bytes"
            )
        selector_change["changed_pairs"] = np.flatnonzero(
            parsed_choices != base_choices
        ).tolist()
    section_identity = {
        key: bool(parsed[key] == base_parsed[key]) for key in parsed
    }
    allow_move = ("carrier_cap1",) + (
        ("selector_blob",) if selector_change is not None else ()
    )
    must_be_identical = [
        k for k in section_identity
        if not k.startswith(allow_move) and not section_identity[k]
    ]
    if must_be_identical:
        raise Fs2Error(
            f"sections other than the CAP1 carrier changed: {must_be_identical}; "
            "the splice is not one-variable and refuses"
        )
    if (
        section_identity.get("carrier_cap1_sha256", True)
        and not np.array_equal(codes, shipped_codes)
    ):
        raise Fs2Error(
            "the CAP1 carrier section is byte-identical to the base; the re-solved "
            "codes did not reach the archive and the build would be a no-op"
        )

    container_search = []
    if args.container_search:
        for quality, lgwin in CONTAINER_OPTIONS:
            for ck2 in (bool(body.ck2_carrier), not bool(body.ck2_carrier)):
                alt = up3.build_archive(
                    body, codes, runtime_dir=runtime,
                    container_options=((ck2, quality, lgwin),), verify=False,
                )
                container_search.append({
                    "ck2_carrier_plane2": ck2,
                    "brotli_quality": quality,
                    "brotli_lgwin": lgwin,
                    "archive_bytes": alt["archive_size"],
                    "delta_vs_shipped_shape": alt["archive_size"] - len(candidate),
                })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_out = out_dir / "archive.zip"
    tmp = archive_out.with_suffix(".zip.partial")
    tmp.write_bytes(candidate)
    tmp.replace(archive_out)
    np.save(out_dir / "codes.npy", codes)
    np.save(out_dir / "shipped_codes.npy", shipped_codes)

    changed_pairs = np.flatnonzero((codes != shipped_codes).any(axis=1)).tolist()
    report = {
        "schema": "tac.ddm_fs2.build.v1",
        "axis": "[bytes -- exact, device-free]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "source_body": {
            "runtime_dir": str(runtime),
            "archive_sha256": observed,
            "archive_bytes": len(shipped_bytes),
            "rx1_reserved": f"{body.rx1_header[4]:#06x}",
            "ck2_carrier_plane2": bool(body.ck2_carrier),
            "shipped_codes_sha256": sha256_array(shipped_codes),
        },
        "container_identity_control": identity,
        "candidate": {
            "path": str(archive_out),
            "sha256": hashlib.sha256(candidate).hexdigest(),
            "bytes": len(candidate),
            "delta_bytes": len(candidate) - len(shipped_bytes),
            "codes_sha256": sha256_array(codes),
            "changed_pairs": changed_pairs,
            "changed_pair_count": len(changed_pairs),
            "changed_coordinates": int((codes != shipped_codes).sum()),
            "rice_bits": built["rice_bits"],
            "rice_bits_delta": built["rice_bits"] - identity_built["rice_bits"],
            "rice_ks": built["rice_ks"],
            "rice_payload_bytes": built["rice_payload_bytes"],
            "container": built["container"],
        },
        "selector_change": selector_change,
        "no_op_detector": {
            "sections": section_identity,
            "selector_identical": not selector_moved,
            "carrier_cap1_differs": not section_identity.get(
                "carrier_cap1_sha256", True
            ),
            "base": base_parsed,
            "candidate": parsed,
        },
        "parse_back_codes_exact": True,
        "container_search": container_search,
        "delta_score_rate": (len(candidate) - len(shipped_bytes)) * BYTE_TO_SCORE,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate": report["candidate"],
        "container_identity_control_byte_identical": identity["byte_identical"],
        "delta_score_rate": report["delta_score_rate"],
    }, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
# mode=compose -- the closing arithmetic, one instrument, one variable.
# --------------------------------------------------------------------------

AXIS_ADVISORY = "[macOS-CPU advisory, frozen CPU-torch PoseNet]"
UNCOMPRESSED_SOURCE_BYTES = 37_545_489


def run_compose(args) -> int:
    import ddm_fs1_frame0_selector_reselection as fs1

    from tac.report_8dp_bounds import (
        derive_pose_score_bound,
        derive_seg_score_bound,
    )

    base = json.loads(Path(args.base_measure).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate_measure).read_text(encoding="utf-8"))
    build = json.loads(Path(args.build_report).read_text(encoding="utf-8"))
    if build.get("schema") != "tac.ddm_fs2.build.v1":
        raise Fs2Error(f"{args.build_report} is not a ddm_fs2 build report")

    # ONE INSTRUMENT.  jg5 Sec 4b measured d_pose moving with the batch shape, so
    # a before/after pair taken at different shapes, GT lineages or renderers is a
    # cross-instrument comparison wearing a delta's clothes.
    for key in ("gt_cache", "renderer"):
        if base["instrument"][key] != cand["instrument"][key]:
            raise Fs2Error(
                f"base and candidate measures disagree on instrument.{key}; a delta "
                "across two instruments is not a delta"
            )
    if base["batch_size"] != cand["batch_size"]:
        raise Fs2Error("base and candidate were measured at different batch shapes")
    if base["pairs"] != cand["pairs"]:
        raise Fs2Error("base and candidate were measured over different pair sets")

    # ONE VARIABLE, the exact inverse of fs1's gate: this arm moves the CARRIER
    # CODES and (optionally, in the alternation) a selector LABEL.  A run where
    # the codes did not move would be reporting a null difference as a result.
    if base["codes_sha256"] == cand["codes_sha256"]:
        raise Fs2Error(
            "base and candidate carry the SAME carrier codes; this would report a "
            "null difference as a result"
        )
    selector_change = build.get("selector_change")
    selector_moved = base["selector_choices_sha256"] != cand["selector_choices_sha256"]
    if selector_moved and selector_change is None:
        raise Fs2Error(
            "the candidate carries a DIFFERENT selector but the build declared no "
            "selector change; the delta would not be attributable"
        )
    if selector_change is not None and not selector_moved:
        raise Fs2Error(
            "the build declared a selector change but the two measures carry the "
            "same selector; the built archive is not the one measured"
        )

    if base["measured_archive_sha256"] != build["source_body"]["archive_sha256"]:
        raise Fs2Error("the base measure was not taken on the body the build spliced")
    if cand["measured_archive_sha256"] != build["candidate"]["sha256"]:
        raise Fs2Error("the candidate measure was not taken on the built archive")

    base_pp = np.load(base["payload"]["per_pair_d_pose"]["path"])
    cand_pp = np.load(cand["payload"]["per_pair_d_pose"]["path"])
    base_d_pose = float(base_pp.mean())
    cand_d_pose = float(cand_pp.mean())
    base_bytes = int(build["source_body"]["archive_bytes"])
    cand_bytes = int(build["candidate"]["bytes"])
    delta_bytes = cand_bytes - base_bytes

    ds_pose = pose_leg(cand_d_pose) - pose_leg(base_d_pose)
    ds_rate = 25.0 * delta_bytes / UNCOMPRESSED_SOURCE_BYTES
    net = ds_pose + ds_rate

    # The pairs the BUILD says it touched -- carrier rows and, if present, the
    # alternation's selector label.  Everything else is the unchanged control.
    changed = sorted({int(p) for p in build["candidate"]["changed_pairs"]} | {
        int(p) for p in (selector_change or {}).get("changed_pairs", [])
    })
    if not changed:
        raise Fs2Error("the build report records no changed pair; nothing to compose")
    unchanged_mask = np.ones(N_PAIRS, dtype=bool)
    unchanged_mask[np.asarray(changed, dtype=np.int64)] = False
    if not unchanged_mask.any():
        raise Fs2Error("every pair changed; the unchanged-pair control would be vacuous")
    unchanged_max_abs = float(
        np.abs(base_pp[unchanged_mask] - cand_pp[unchanged_mask]).max()
    )
    # A pair the build did NOT touch must not move at all.  This is the no-op
    # detector read at the SCORE level rather than the byte level.
    moved_but_untouched = np.flatnonzero(
        (base_pp != cand_pp) & unchanged_mask
    ).tolist()
    if moved_but_untouched:
        raise Fs2Error(
            f"pairs {moved_but_untouched[:8]} moved but the build did not touch them; "
            "the splice is not one-variable"
        )

    per_pair_rows = [
        {
            "pair": int(p),
            "base_d_pose": float(base_pp[p]),
            "candidate_d_pose": float(cand_pp[p]),
            "gain": float(base_pp[p] - cand_pp[p]),
            "ratio": (
                float(base_pp[p] / cand_pp[p]) if cand_pp[p] > 0 else float("inf")
            ),
        }
        for p in changed
    ]

    reproduction = None
    if args.projected_net_dS is not None:
        projected = float(args.projected_net_dS)
        reproduction = {
            "projection_net_dS": projected,
            "measured_net_dS": net,
            "relative_difference": (net - projected) / abs(projected),
            "within_exchange_noise_floor_6pct": bool(
                abs(net - projected) / abs(projected) <= 0.06
            ),
        }

    admissibility = fs1._bootstrap_admissibility(
        base_pp, cand_pp, delta_bytes=delta_bytes, exact_net=net
    )
    # The T4 prints d_seg and d_pose to 8dp, and the bounds ADD across the two
    # rows being differenced.  A landed net dS inside this bound is UNRESOLVED,
    # not a win.
    pose_bound = (
        derive_pose_score_bound(base_d_pose) + derive_pose_score_bound(cand_d_pose)
    )
    # ``ddm_fs1`` quoted the FULL row bound, which adds the seg leg's own 8dp
    # rounding to each row.  On a d_seg-INVARIANT edit the two rows print the
    # same d_seg, so that term is the same number twice and cancels exactly in
    # the difference -- but only if the T4 does print the same d_seg, which is
    # the seal's own first falsifier.  Both are reported: the pose-only bound is
    # the one that applies if the falsifier holds, the conservative one is what
    # applies if it does not.
    seg_bound = 2.0 * derive_seg_score_bound()
    two_row_bound = pose_bound

    report = {
        "schema": "tac.ddm_fs2.compose.v1",
        "axis": AXIS_ADVISORY,
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "instrument": base["instrument"],
        "batch_size": base["batch_size"],
        "base": {
            "archive_sha256": build["source_body"]["archive_sha256"],
            "archive_bytes": base_bytes,
            "d_pose": base_d_pose,
            "pose_leg": pose_leg(base_d_pose),
            "measure": str(Path(args.base_measure).resolve()),
        },
        "candidate": {
            "archive_sha256": build["candidate"]["sha256"],
            "archive_bytes": cand_bytes,
            "d_pose": cand_d_pose,
            "pose_leg": pose_leg(cand_d_pose),
            "measure": str(Path(args.candidate_measure).resolve()),
        },
        "delta": {
            "d_pose": cand_d_pose - base_d_pose,
            "delta_S_pose": ds_pose,
            "delta_bytes": delta_bytes,
            "delta_S_rate": ds_rate,
            "net_delta_S": net,
            "delta_S_seg": 0.0,
            "delta_S_seg_justification": (
                "STRUCTURAL, verified at source: this splice rewrites the CAP1 "
                "carrier (frame 0) and, in the alternation, one frame-0 selector "
                "label. Both write output[2*frame_ids] (f26_inflate.py:133), i.e. "
                "frame 2p, while SegNet scores x[:, -1, ...] "
                "(upstream/modules.py:100), i.e. frame 2p+1. The build's no-op "
                "detector proves the semantic section, token stream, HPAC model, "
                "residual payload and table codes are byte-identical through the "
                "receiver's own parse, so the odd frames are bit-identical and "
                "d_seg cannot move."
            ),
        },
        "projected_score": {
            "base_score_contest_cuda_t4": FS1_SCORE_T4,
            "base_score_source": (
                ".omx/research/ddm_fs1_pointer_move_24_20260904.md -- the T4 receipt, "
                "recomputed from components (#877)"
            ),
            "projected_score": FS1_SCORE_T4 + net,
            "composition_note": (
                "the LEVEL is the contest-CUDA T4 receipt; the DELTA is this arm's "
                "advisory same-instrument difference. Not a score."
            ),
        },
        "unchanged_pairs": {
            "count": int(unchanged_mask.sum()),
            "max_abs_delta_d_pose": unchanged_max_abs,
            "bit_identical": unchanged_max_abs == 0.0,
        },
        "changed_pairs": per_pair_rows,
        "report_resolution": {
            "two_row_8dp_delta_bound_pose_only": pose_bound,
            "two_row_8dp_delta_bound_conservative": pose_bound + seg_bound,
            "seg_leg_bound_cancels_if_d_seg_prints_identically": seg_bound,
            "net_over_bound": abs(net) / two_row_bound if two_row_bound else float("inf"),
            "net_over_conservative_bound": abs(net) / (pose_bound + seg_bound),
            "resolvable_by_the_t4_report": abs(net) > pose_bound + seg_bound,
            "source": (
                "tac.report_8dp_bounds.derive_pose_score_bound + "
                "derive_seg_score_bound, never typed"
            ),
        },
        "admissibility": admissibility,
        "reproduction": reproduction,
        "build_report": str(Path(args.build_report).resolve()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "base": report["base"], "candidate": report["candidate"],
        "delta": report["delta"], "projected_score": report["projected_score"],
        "unchanged_pairs": report["unchanged_pairs"],
        "report_resolution": report["report_resolution"],
        "admissibility": report["admissibility"],
        "reproduction": report["reproduction"],
    }, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser(
        "solve", help="jg5 Gauss-Newton re-solve on the selector-changed pairs"
    )
    solve.add_argument("--runtime", required=True,
                       help="the body to re-solve (fs1 candidate B)")
    solve.add_argument("--base-runtime", required=True,
                       help="the body it was spliced FROM (afr1), for the diff")
    solve.add_argument("--expect-archive-sha256", default=FS1_ARCHIVE_SHA256)
    solve.add_argument("--base-archive-sha256", default=BASE_ARCHIVE_SHA256)
    solve.add_argument("--gt-cache", required=True)
    solve.add_argument("--renderer", required=True)
    solve.add_argument("--tokens", required=True)
    solve.add_argument("--axis", default="contest_cuda")
    solve.add_argument("--pairs", default=None,
                       help="explicit subset of the selector diff; default is all of it")
    solve.add_argument("--outer-rounds", type=int, default=40)
    solve.add_argument("--max-gn-iterations", type=int, default=400)
    solve.add_argument(
        "--materiality-operating-point", type=float,
        default=FS1_D_POSE_ADVISORY_N600_BATCH8,
        help="mean d_pose at which the DERIVED materiality floor is evaluated",
    )
    solve.add_argument("--threads", type=int, default=4)
    solve.add_argument("--label", default="fs2_solve")
    solve.add_argument("--out", required=True)
    solve.set_defaults(func=run_solve)

    codes = sub.add_parser("codes", help="merge solved rows into a 600x12 table + price")
    codes.add_argument("--runtime", required=True)
    codes.add_argument("--expect-archive-sha256", default=FS1_ARCHIVE_SHA256)
    codes.add_argument("--rows", nargs="+", required=True)
    codes.add_argument("--adopt-all", action="store_true",
                       help="adopt every solved row, including non-improving ones")
    codes.add_argument("--base-d-pose", type=float,
                       default=FS1_D_POSE_ADVISORY_N600_BATCH8)
    codes.add_argument("--base-d-pose-source",
                       default="ddm_fs1 retained/measure_candB_byte_optimal_101_n600.json")
    codes.add_argument("--threads", type=int, default=4)
    codes.add_argument("--out", required=True)
    codes.set_defaults(func=run_codes)

    build = sub.add_parser("build", help="splice the re-solved codes into archive.zip")
    build.add_argument("--runtime", required=True)
    build.add_argument("--expect-archive-sha256", default=FS1_ARCHIVE_SHA256)
    build.add_argument("--codes", required=True)
    build.add_argument(
        "--selector-choices", default=None,
        help=(
            "optional (600,) selector vector to splice ALONGSIDE the codes -- the "
            "alternation step. Both changes are proved through the receiver's own "
            "parse-back; neither is assumed."
        ),
    )
    build.add_argument("--out-dir", required=True)
    build.add_argument(
        "--container-search", action="store_true",
        help=(
            "measure the encoder-only alternatives for the RECORD. The candidate "
            "is always written at this body's own shipped shape, so a smaller "
            "alternative is REPORTED, never silently selected -- a container "
            "credit and a carrier credit in one number would not be separable."
        ),
    )
    build.add_argument("--threads", type=int, default=4)
    build.add_argument("--label", default="fs2_build")
    build.add_argument("--out", required=True)
    build.set_defaults(func=run_build)

    compose = sub.add_parser("compose", help="the closing arithmetic, one instrument")
    compose.add_argument("--base-measure", required=True)
    compose.add_argument("--candidate-measure", required=True)
    compose.add_argument("--build-report", required=True)
    compose.add_argument("--projected-net-dS", type=float, default=None)
    compose.add_argument("--threads", type=int, default=4)
    compose.add_argument("--label", default="fs2_compose")
    compose.add_argument("--out", required=True)
    compose.set_defaults(func=run_compose)

    return parser


def main(argv: list[str] | None = None) -> int:
    import os

    import torch

    args = build_parser().parse_args(argv)
    # Thread control is set HERE, before any torch work, exactly as ``ddm_pr1``
    # does it.  Left to the default, two concurrent solve cells would each claim
    # every core and the wall-clock of both would be worse than serial.
    threads = max(1, int(getattr(args, "threads", 4)))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
