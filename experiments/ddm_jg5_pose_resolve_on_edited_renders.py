"""ddm_jg5 -- re-solve the pose carrier against the composed candidate's OWN renders.

WHY THIS ARM EXISTS
-------------------
``ddm_jg4`` spliced ``ddm_jg3``'s seg token edits (573 pairs) into the ``ddm_br1``
body.  Those edits change the SEMANTIC stream, which renders frame 1 -- the odd
frame.  PoseNet reads BOTH frames (``upstream/modules.py:72-80``), so every edited
pair's pose residual moved, while the carrier coefficients riding in that archive
are still ``ddm_br1``'s, solved against the OLD frame 1.  The composed candidate
therefore ships a carrier that is stale by construction.

This module re-solves the carrier against the candidate's own decode.  The solver
is ``ddm_br1``'s damped Gauss-Newton on the SHIPPED 12-dim basis and the shipped
signed-int12 lattice, reused verbatim (``ddm_br1_pose_basis_reorientation.gn_solve_pair``)
-- the OPTIMAL FORM of this family is br1's, and br1's binding lesson was that the
wall was ``ddm_up2``'s +-2 single-coordinate SEARCH, not the basis.  Nothing about
that lesson changes when frame 1 moves, so re-deriving a solver here would only
risk a weaker one.

THE ONE THING THAT CHANGES: THE RENDERS
---------------------------------------
``br1.load_instrument`` reads odd frames from the POINTER body's decode and notes
that carrier edits never touch odd frames, so that decode stays valid.  Here the
premise is inverted -- token edits DO touch odd frames -- so the raw decode is an
explicit argument, and it is the candidate's own
(``ddm_jg4/advisory_complete_r2/work/inflated/0.raw``).  Solving on the base decode
would optimise against frames the candidate does not ship.

WHY d_seg CANNOT MOVE
---------------------
``ddm_up3_carrier_splice.build_archive`` rebuilds the archive by copying
``hpac_stream``, ``semantic_stream`` and ``section_tail`` VERBATIM and re-encoding
only the carrier stream (``ddm_up3_carrier_splice.py:577-594``).  SegNet reads only
the odd frame (``upstream/modules.py:108``) and the odd frame is produced by the
semantic stream and the token tail, neither of which this module writes.  ``mode=close``
proves it per candidate by byte-diffing those three sections against the jg4 body
rather than asserting it.

AUTHORITY
---------
Frozen CPU-torch PoseNet, GT decoded by DALI -- the lineage ``upstream/evaluate.py``
uses on the contest-CUDA axis the pointer row was measured on.  The pose lineage gap
is ADDITIVE, not a transferable factor
([[gt_lineage_pose_gap_is_additive_not_multiplicative_20260819]]), so this module
scores directly on the DALI table and never rescales an advisory number.
``score_claim=false``, ``promotable=false``.

EVERY COMPARISON IS SAME-INSTRUMENT.  The candidate's seg leg was measured on the
jg1 DALI instrument (0.00017460) against that instrument's OWN base leg (0.00030307);
the T4 base leg is 0.00030309.  The projection below carries the ratio explicitly and
never compares an advisory number against a T4 number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_up2_shipping_pose_solve as up2
import ddm_up3_carrier_splice as splice

K = up2.CARRIER_DIM
N_PAIRS = up2.N_PAIRS_TOTAL
BYTE_TO_SCORE = br1.BYTE_TO_SCORE

# --------------------------------------------------------------------------
# The composed candidate this arm re-solves. Every value is a receipt, not a guess.
# --------------------------------------------------------------------------

#: The jg4 composed candidate: br1's body carrying jg3's 573-pair seg token edits.
CANDIDATE_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_jg4/candidate_runtime_complete")
CANDIDATE_ARCHIVE_SHA256 = (
    "4b0dc2724117aa3076a6c271e56f11476120cdce255e42cf6bcfc31b79c253e4"
)
CANDIDATE_ARCHIVE_BYTES = 181_636
#: The candidate's OWN decode -- the renders the carrier must now be tuned against.
CANDIDATE_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_jg4/advisory_complete_r2/work/inflated/0.raw"
)

#: The live pointer row (contest-CUDA, Tesla T4, n600) the composed row must beat.
POINTER_ARCHIVE_SHA256 = (
    "44e9e6507d60bf8b6429ce066983aa814b23f2f929869aa5a10a8b8dacda5c7d"
)
POINTER_ARCHIVE_BYTES = 176_429
POINTER_SCORE = 0.15615242950573233
POINTER_D_SEG_T4 = 0.00030309
POINTER_D_POSE_DALI = 6.99315662169577e-06

#: Seg legs, both on the jg1 DALI per-pair instrument
#: (``ddm_jg4/retained/perpair_dseg_decomposition.json``).
JG1_BASE_D_SEG = 0.00030307345920138883
JG1_CAND_D_SEG = 0.0001746029324001736

#: Admission bar for a net score move, in score units (negative is better).
ADMIT_BAR = br1.ADMIT_BAR


class Jg5Error(RuntimeError):
    """A ddm_jg5 precondition failed. Fail closed, never approximate."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def pose_report_bound(d_pose: float) -> float:
    """Half-ULP of the 8dp d_pose report, in score units, quoted PER ROW.

    Bounds ADD for a delta and are UNEQUAL per row because the pose leg's
    sensitivity grows as d_pose falls
    ([[concavity_helps_when_you_pay_the_axis_upward_20260818]]).
    """
    if d_pose <= 0.0:
        return pose_leg(up2.REPORT_HALF_ULP)
    return 5.0 / math.sqrt(10.0 * d_pose) * up2.REPORT_HALF_ULP


def candidate_d_seg_t4_projected() -> float:
    """The candidate's seg leg carried onto the T4 axis by the SAME-instrument ratio.

    The jg1 instrument's own base leg and the T4 base leg differ by 1.000055x; the
    ratio is applied explicitly so the composed projection never silently mixes an
    advisory seg number with a T4 one.
    """
    return POINTER_D_SEG_T4 * (JG1_CAND_D_SEG / JG1_BASE_D_SEG)


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + pose_leg(d_pose) + 25.0 * archive_bytes / 37_545_489.0


# --------------------------------------------------------------------------
# The instrument -- br1's, with the candidate's renders.
# --------------------------------------------------------------------------


def load_candidate_instrument(
    *,
    runtime: Path = CANDIDATE_RUNTIME,
    expect_archive_sha256: str = CANDIDATE_ARCHIVE_SHA256,
    raw_path: Path = CANDIDATE_RAW,
    expect_raw_sha256: str | None = None,
):
    """br1's instrument, pinned to the candidate body AND the candidate's own decode.

    Both the archive and the raw decode are identified by sha256 before anything is
    measured: a delta against an unidentified body or against someone else's frames
    is unanchored
    ([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]).
    """
    runtime = Path(runtime)
    observed_archive = _sha256_file(runtime / "archive.zip")
    if observed_archive != expect_archive_sha256:
        raise Jg5Error(
            f"runtime archive sha256 {observed_archive} != expected "
            f"{expect_archive_sha256}; refusing to solve against an unidentified body"
        )

    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise Jg5Error(f"candidate raw decode does not exist: {raw_path}")
    if expect_raw_sha256:
        observed_raw = _sha256_file(raw_path)
        if observed_raw != expect_raw_sha256:
            raise Jg5Error(
                f"candidate raw sha256 {observed_raw} != expected {expect_raw_sha256}"
            )
    expected_size = 2 * N_PAIRS * up2.CAMERA_H * up2.CAMERA_W * 3
    actual_size = raw_path.stat().st_size
    if actual_size != expected_size:
        raise Jg5Error(
            f"candidate raw is {actual_size} B, expected {expected_size} B "
            f"({2 * N_PAIRS} frames of {up2.CAMERA_H}x{up2.CAMERA_W}x3)"
        )
    raw = np.memmap(
        raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS, up2.CAMERA_H, up2.CAMERA_W, 3),
    )

    state = up2.load_carrier_state(runtime, verify_archive=False)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise Jg5Error(
            f"GT lineage is {lineage}, not {up2.LINEAGE_DALI}: this would solve the "
            "contest-CPU objective, which is a different object (evaluate.py:31-42)"
        )
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


def shard_pairs(shard_index: int, shard_count: int) -> np.ndarray:
    """STRIDED shard of the full 600, never a contiguous block.

    A contiguous block of this video is a different population and the bias is worst
    on the pose axis ([[m88]] / [[m96]]), so a shard that dies leaves an unbiased
    partial rather than a scene-block one.
    """
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise Jg5Error(f"bad shard {shard_index}/{shard_count}")
    return np.arange(shard_index, N_PAIRS, shard_count, dtype=np.int64)


# --------------------------------------------------------------------------
# mode=control -- the forward model must reproduce the candidate's shipped frame 0.
# --------------------------------------------------------------------------


def run_control(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst = load_candidate_instrument(expect_raw_sha256=args.expect_raw_sha256)

    pairs = [0, 1, 137, 299, 431, 599]
    forward = up2.validate_forward_model(inst.state, inst.raw, pairs)

    br1_codes_path = Path(args.br1_codes)
    codes_match_br1 = None
    if br1_codes_path.is_file():
        br1_codes = np.load(br1_codes_path).astype(np.int32)
        codes_match_br1 = bool(
            np.array_equal(np.asarray(inst.state.codes, dtype=np.int32), br1_codes)
        )

    start = time.time()
    d_pose_smoke, _poses = up2.measure_pose(
        inst.posenet,
        inst.state,
        inst.state.coefficients,
        inst.raw,
        inst.targets,
        np.asarray(pairs, dtype=np.int64),
        batch_size=2,
    )
    report = {
        "schema": "ddm_jg5_control.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
        "candidate_archive_sha256": CANDIDATE_ARCHIVE_SHA256,
        "candidate_raw": str(CANDIDATE_RAW),
        "forward_model_byte_exact_on_candidate_frame0": forward,
        "candidate_codes_equal_br1_solved_codes": codes_match_br1,
        "smoke_pairs": pairs,
        "smoke_d_pose_per_pair": d_pose_smoke.tolist(),
        "smoke_seconds": time.time() - start,
    }
    (out_dir / "CONTROL.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not forward["byte_exact"]:
        raise Jg5Error(
            "CONTROL FAILED: render_frame0 does not reproduce the candidate's shipped "
            f"frame-0 bytes (max |delta| {forward['max_abs_delta']}); every later "
            "number would be measuring a different renderer than the one that ships"
        )
    return 0


# --------------------------------------------------------------------------
# mode=refine -- run each pair to DIMINISHING RETURNS, not to a fixed budget.
# --------------------------------------------------------------------------
#
# ``br1.gn_solve_pair`` stops on two caps that are budgets, not convergence tests:
#
#   1. ``iterations=6`` is a hard count. Measured on this arm's n600 solve, 16 pairs
#      accepted an improving step on ALL SIX iterations -- every one of them still
#      descending >2% on its last step. Those pairs were stopped by the counter.
#   2. The pass structure is GN-then-polish ONCE. It never re-enters GN after the
#      +-2 polish has moved the point, so a pair that polish carries into a region
#      where the Gauss-Newton direction has more to give never gets to use it.
#
# This is the same genus as ``ddm_up2``'s +-2 search radius, which ``ddm_br1`` itself
# was built to escape ([[caps_genus_trajectory_stopping_20260805]]): a limit that
# reads as "converged" in the receipt but is really "converged WITHIN TOLERANCE".
#
# The cure changes SCOPE, never MECHANISM. Same damped Gauss-Newton, same shipped
# 12-dim basis, same int12 lattice, same STEP_LADDER, same +-2 polish, same
# realized-evaluation acceptance -- the solver still proposes and the receiver still
# disposes. Only the STOPPING changes: alternate GN and polish until a whole round
# buys less than ``--min-rel-improvement``, with large budgets as a backstop rather
# than as the operative limit. Every row records WHY it stopped, so a future reader
# can tell a floor from a ceiling without re-deriving it.


REFINE_STOP_REASONS = (
    "lattice_floor",  # PHYSICAL: no ladder fraction moves the point (< half a code unit)
    "no_improving_step",  # PHYSICAL: a realized step was evaluated and rejected
    "converged_below_materiality_floor",  # DERIVED: projected gain < this pair's band share
    "gn_iteration_budget",  # backstop; a non-empty count here means the rule never bound
    "outer_round_budget",  # backstop; same
)

#: The MEASURED T4 admission/noise band, in score units. It is the same number as
#: ``br1.ADMIT_BAR``; it is named separately here because it is being used as an
#: INSTRUMENT NOISE FLOOR, not as an admission threshold, and a reader must be able to
#: see which role each use is playing.
T4_NOISE_BAND_S = abs(ADMIT_BAR)


def pose_sensitivity_per_pair(mean_d_pose: float) -> float:
    """Exact d(S)/d(d_pose_i): how one pair's pose moves the contest score.

    S carries pose as ``sqrt(10 * mean_i d_i)``, so
    ``dS/dd_i = (10/N) / (2 sqrt(10 m)) = 10 / (2 N sqrt(10 m))``.
    Nothing here is fitted; it is the derivative of the scoring function.
    """
    if mean_d_pose <= 0.0:
        raise Jg5Error("sensitivity is undefined at mean d_pose <= 0")
    return 10.0 / (2.0 * N_PAIRS * math.sqrt(10.0 * mean_d_pose))


def materiality_dd_threshold(mean_d_pose: float) -> float:
    """Per-pair d_pose gain whose projected score gain equals this pair's noise share.

    The floor is the measured T4 band EQUAL-ALLOCATED across the population: if every
    pair left exactly this much on the table, the total left would be one full band.
    Dividing it by the exact sensitivity converts a score threshold into the d_pose
    threshold the solver can actually test. Every constant is a receipt: the band is
    measured, N is the population, the sensitivity is a derivative.
    """
    return (T4_NOISE_BAND_S / N_PAIRS) / pose_sensitivity_per_pair(mean_d_pose)


def projected_remaining_gain(step_gains: list[float]) -> tuple[float, float]:
    """Geometric extrapolation of what is left, from the pair's OWN decay.

    Returns ``(remaining, ratio)``. With ``r = g_k / g_{k-1} < 1`` the tail of a
    geometric sequence sums to ``g_k * r / (1 - r)``. ``r >= 1`` means the pair is not
    decaying geometrically and no extrapolation is admissible, so the caller must keep
    iterating rather than guess -- reported as ``inf``.
    """
    if len(step_gains) < 2 or step_gains[-2] <= 0.0:
        return math.inf, math.nan
    ratio = step_gains[-1] / step_gains[-2]
    if ratio >= 1.0:
        return math.inf, ratio
    return step_gains[-1] * ratio / (1.0 - ratio), ratio


def refine_pair(
    inst,
    pair: int,
    start_codes: np.ndarray,
    *,
    dd_threshold: float,
    outer_rounds: int = 40,
    max_gn_iterations: int = 400,
) -> dict[str, Any]:
    """Alternate GN and +-2 polish until the PROJECTED remaining gain is immaterial.

    ``dd_threshold`` is the derived per-pair d_pose floor from
    ``materiality_dd_threshold``. There is no relative tolerance and no hand-set
    iteration count anywhere in the stopping decision: the solver stops when the
    pair's own measured decay projects less remaining gain than its share of the
    measured instrument band, or when the receiver refuses every proposed step.
    """
    import torch

    index = np.array([pair], dtype=np.int64)
    frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
    target_row = inst.targets[pair]
    tb = torch.from_numpy(target_row[None]).float()
    scales = inst.state.coefficient_scales.double().numpy()

    current = start_codes.astype(np.int32).copy()
    best = float(br1.evaluate_codes(inst, pair, current[None])[0])
    start_value = best
    history = [best]
    evaluations = 1
    demanded_max = 0.0
    gn_iterations = 0
    polish_steps = 0
    stop_reason = "outer_round_budget"
    rounds = 0

    round_gains: list[float] = []
    last_ratio = math.nan
    last_remaining = math.inf
    for _round in range(outer_rounds):
        rounds += 1
        round_entry = best
        inner_stop = "gn_iteration_budget"
        step_gains: list[float] = []
        for _ in range(max_gn_iterations):
            coeff = up2.codes_to_coefficients(
                current[None], inst.state.coefficient_scales
            )
            jac, res, _ = up2.jacobian_and_residual(
                inst.posenet, inst.state, coeff, frame1, tb, index
            )
            step = br1.min_image_norm_step(
                jac[0].double(), res[0].double(), inst.gram
            ).numpy()
            demanded_max = max(demanded_max, float(np.abs(step / scales).max()))
            block = [
                trial
                for fraction in br1.STEP_LADDER
                if not np.array_equal(
                    (trial := br1.realize(current + fraction * step / scales)), current
                )
            ]
            if not block:
                inner_stop = "lattice_floor"
                break
            block_arr = np.stack(block)
            values = br1.evaluate_codes(inst, pair, block_arr)
            evaluations += len(block_arr)
            winner = int(values.argmin())
            if values[winner] >= best:
                inner_stop = "no_improving_step"
                break
            step_gains.append(best - float(values[winner]))
            best = float(values[winner])
            current = block_arr[winner].copy()
            history.append(best)
            gn_iterations += 1
            remaining, ratio = projected_remaining_gain(step_gains)
            last_ratio, last_remaining = ratio, remaining
            if remaining <= dd_threshold:
                inner_stop = "converged_below_materiality_floor"
                break

        while True:
            block, _labels = up2.candidate_codes_for_pair(current, (-2, -1, 1, 2))
            if len(block) == 0:
                break
            values = br1.evaluate_codes(inst, pair, block)
            evaluations += len(block)
            winner = int(values.argmin())
            if values[winner] >= best:
                break
            best = float(values[winner])
            current = block[winner].copy()
            history.append(best)
            polish_steps += 1

        round_gains.append(round_entry - best)
        if round_gains[-1] <= 0.0:
            # A whole round bought nothing at all: both the GN direction and the +-2
            # neighbourhood refused. That is a physical stop, not a tolerance.
            stop_reason = inner_stop if inner_stop != "gn_iteration_budget" else (
                "no_improving_step"
            )
            break
        remaining, ratio = projected_remaining_gain(round_gains)
        last_ratio, last_remaining = ratio, remaining
        if remaining <= dd_threshold:
            stop_reason = "converged_below_materiality_floor"
            break

    return {
        "pair": int(pair),
        "start_d_pose": start_value,
        "final_d_pose": best,
        "gn_d_pose": best,
        "ratio": best / start_value if start_value > 0 else 1.0,
        "gn_ratio": best / start_value if start_value > 0 else 1.0,
        "evaluations": evaluations,
        "demanded_code_units_max": demanded_max,
        "codes": current.astype(np.int32).tolist(),
        "changed_coordinates": int((current != start_codes).sum()),
        "max_abs_code": int(np.abs(current).max()),
        "history": history,
        "stop_reason": stop_reason,
        "outer_rounds_used": rounds,
        "gn_iterations_total": gn_iterations,
        "polish_steps_total": polish_steps,
        "dd_threshold": dd_threshold,
        "last_decay_ratio": None if math.isnan(last_ratio) else last_ratio,
        "projected_remaining_dd": (
            None if math.isinf(last_remaining) else last_remaining
        ),
        "non_geometric_at_stop": bool(
            math.isinf(last_remaining) and stop_reason.endswith("budget")
        ),
        "refined": True,
    }


def run_refine(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    prior = merge_rows([Path(p) for p in args.rows])
    if not prior:
        raise Jg5Error(f"no prior rows to refine at {args.rows}")

    # The stopping rule is DERIVED from the score arithmetic at the waterfill operating
    # point, not chosen. Recompute it if that operating point moves materially.
    dd_threshold = materiality_dd_threshold(args.operating_point_d_pose)
    print(
        json.dumps(
            {
                "operating_point_mean_d_pose": args.operating_point_d_pose,
                "dS_per_pair_d_pose": pose_sensitivity_per_pair(
                    args.operating_point_d_pose
                ),
                "delta_floor_S_per_pair": T4_NOISE_BAND_S / N_PAIRS,
                "dd_threshold": dd_threshold,
            }
        ),
        flush=True,
    )

    inst = load_candidate_instrument(expect_raw_sha256=args.expect_raw_sha256)
    targets = (
        np.load(args.targets).astype(np.int64)
        if args.targets
        else np.array(sorted(prior), dtype=np.int64)
    )
    targets = targets[args.shard_index :: args.shard_count]

    done = br1.load_done(rows_path)
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        for position, pair in enumerate(targets):
            pair = int(pair)
            if pair in done:
                continue
            # WARM START from the banked solve: refinement must never throw away work
            # it is trying to extend.
            start_codes = np.asarray(prior[pair]["codes"], dtype=np.int32)
            row = refine_pair(
                inst,
                pair,
                start_codes,
                dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds,
                max_gn_iterations=args.max_gn_iterations,
            )
            row["prior_d_pose"] = prior[pair]["final_d_pose"]
            row["start_d_pose"] = prior[pair]["start_d_pose"]
            row["ratio"] = (
                row["final_d_pose"] / row["start_d_pose"]
                if row["start_d_pose"] > 0
                else 1.0
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[pair] = row
            print(
                f"[refine {args.shard_index}/{args.shard_count}]"
                f"[{position + 1}/{len(targets)}] pair {pair}: "
                f"{row['prior_d_pose']:.6e} -> {row['final_d_pose']:.6e} "
                f"({row['stop_reason']}, rounds {row['outer_rounds_used']}, "
                f"gn {row['gn_iterations_total']}, polish {row['polish_steps_total']}) "
                f"[{time.time() - started:.0f}s]",
                flush=True,
            )

    solved = [done[int(p)] for p in targets if int(p) in done]
    improved = [r for r in solved if r["final_d_pose"] < r.get("prior_d_pose", np.inf)]
    summary = {
        "schema": "ddm_jg5_refine_summary.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": len(solved),
        "pairs_improved_by_refinement": len(improved),
        "operating_point_mean_d_pose": args.operating_point_d_pose,
        "dd_threshold": dd_threshold,
        "delta_floor_S_per_pair": T4_NOISE_BAND_S / N_PAIRS,
        "stop_reasons": {
            reason: sum(1 for r in solved if r["stop_reason"] == reason)
            for reason in REFINE_STOP_REASONS
        },
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=baseline -- what the composed candidate's stale carrier actually scores.
# --------------------------------------------------------------------------


def run_baseline(args) -> int:
    """Exact n600 d_pose of the composed candidate AS IT STANDS, on both decodes.

    This is the number the re-solve has to start from, and it is measured over the
    WHOLE population rather than a sample: pose is the axis where a contiguous or
    small sample misleads worst ([[m96]]).  Both decodes are scored on the SAME
    instrument and the SAME DALI GT, so the base-vs-candidate difference isolates
    the token edit's effect on frame 1 and nothing else.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst = load_candidate_instrument(expect_raw_sha256=args.expect_raw_sha256)
    pairs = np.arange(N_PAIRS, dtype=np.int64)

    # MEASURED 2026-08-19: this forward is deterministic at a FIXED batch shape
    # (batch-1 repeats are bit-identical) but its VALUE moves with the batch shape --
    # 6e-5 relative on large per-pair values, up to 7.7e-3 on the smallest
    # ([[batch_shape_is_part_of_the_forward_instrument_20260806]]). The solver's own
    # rows come from mixed ladder/polish chunk widths, so any per-pair comparison
    # against this baseline must be re-measured HERE, at this one declared shape.
    coefficients = inst.state.coefficients
    codes_source = "archive"
    if args.codes:
        codes = np.load(args.codes).astype(np.int32)
        if codes.shape != (N_PAIRS, K):
            raise Jg5Error(f"codes have shape {codes.shape}, expected {(N_PAIRS, K)}")
        coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
        codes_source = str(args.codes)

    started = time.time()
    cand_per_pair, cand_poses = up2.measure_pose(
        inst.posenet,
        inst.state,
        coefficients,
        inst.raw,
        inst.targets,
        pairs,
        batch_size=args.batch_size,
    )
    # ALWAYS KEEP THE PAYLOAD: the per-pair vector and the pose vectors are
    # materialised here, so they are written before any scalar is taken from them.
    tag = args.tag
    np.save(out_dir / f"d_pose_per_pair_{tag}.npy", cand_per_pair)
    np.save(out_dir / f"poses_{tag}.npy", cand_poses)

    base_report: dict[str, Any] | None = None
    if args.base_raw:
        base_raw = np.memmap(
            Path(args.base_raw),
            dtype=np.uint8,
            mode="r",
            shape=(2 * N_PAIRS, up2.CAMERA_H, up2.CAMERA_W, 3),
        )
        base_per_pair, base_poses = up2.measure_pose(
            inst.posenet,
            inst.state,
            inst.state.coefficients,
            base_raw,
            inst.targets,
            pairs,
            batch_size=args.batch_size,
        )
        np.save(out_dir / "d_pose_per_pair_base_odd_frames.npy", base_per_pair)
        np.save(out_dir / "poses_base_odd_frames.npy", base_poses)
        base_report = {
            "raw": str(args.base_raw),
            "note": (
                "SAME carrier codes, SAME GT, only the ODD frames come from the base "
                "decode -- so the difference is the token edit's effect on frame 1"
            ),
            "d_pose_mean": float(base_per_pair.mean()),
            "d_pose_median": float(np.median(base_per_pair)),
            "d_pose_max": float(base_per_pair.max()),
            "pairs_worse_on_candidate": int((cand_per_pair > base_per_pair).sum()),
            "pairs_better_on_candidate": int((cand_per_pair < base_per_pair).sum()),
        }

    d_pose = float(cand_per_pair.mean())
    d_seg_t4 = candidate_d_seg_t4_projected()
    report = {
        "schema": "ddm_jg5_baseline.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
        "candidate_archive_sha256": CANDIDATE_ARCHIVE_SHA256,
        "candidate_archive_bytes": CANDIDATE_ARCHIVE_BYTES,
        "candidate_raw": str(CANDIDATE_RAW),
        "codes_source": codes_source,
        "batch_size": args.batch_size,
        "pairs": int(len(pairs)),
        "d_pose_mean": d_pose,
        "d_pose_median": float(np.median(cand_per_pair)),
        "d_pose_p95": float(np.percentile(cand_per_pair, 95)),
        "d_pose_max": float(cand_per_pair.max()),
        "pointer_d_pose_dali": POINTER_D_POSE_DALI,
        "ratio_vs_pointer": d_pose / POINTER_D_POSE_DALI,
        "base_odd_frames_control": base_report,
        "composed_projection": {
            "d_seg_t4_projected": d_seg_t4,
            "seg_leg": 100.0 * d_seg_t4,
            "pose_leg": pose_leg(d_pose),
            "rate_leg": 25.0 * CANDIDATE_ARCHIVE_BYTES / 37_545_489.0,
            "score": composed_score(d_seg_t4, d_pose, CANDIDATE_ARCHIVE_BYTES),
            "pointer_score_t4": POINTER_SCORE,
        },
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "BASELINE.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=gn -- br1's damped Gauss-Newton, on the candidate's renders.
# --------------------------------------------------------------------------


def run_gn(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    inst = load_candidate_instrument(expect_raw_sha256=args.expect_raw_sha256)
    start_all = np.asarray(inst.state.codes, dtype=np.int32)
    if start_all.shape != (N_PAIRS, K):
        raise Jg5Error(f"candidate codes have shape {start_all.shape}")

    pairs = shard_pairs(args.shard_index, args.shard_count)
    if args.limit:
        pairs = pairs[: args.limit]

    done = br1.load_done(rows_path)
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            row = br1.gn_solve_pair(
                inst,
                int(pair),
                start_all[int(pair)],
                iterations=args.iterations,
                polish=not args.no_polish,
            )
            row["start_from"] = "jg4_candidate_shipped_codes"
            row["renders"] = "jg4_candidate_own_decode"
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[int(pair)] = row
            print(
                f"[shard {args.shard_index}/{args.shard_count}]"
                f"[{position + 1}/{len(pairs)}] pair {pair}: "
                f"{row['start_d_pose']:.6e} -> {row['final_d_pose']:.6e} "
                f"(gn {row['gn_ratio']:.4f}, final {row['ratio']:.4f}) "
                f"demanded {row['demanded_code_units_max']:.0f} "
                f"[{time.time() - started:.0f}s]",
                flush=True,
            )

    rows = [done[int(p)] for p in pairs if int(p) in done]
    before = float(np.mean([r["start_d_pose"] for r in rows])) if rows else 0.0
    after = float(np.mean([r["final_d_pose"] for r in rows])) if rows else 0.0
    summary = {
        "schema": "ddm_jg5_gn_summary.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": len(rows),
        "d_pose_start_mean_shard": before,
        "d_pose_final_mean_shard": after,
        "ratio_shard": after / before if before else 1.0,
        "pairs_improved": int(
            sum(1 for r in rows if r["final_d_pose"] < r["start_d_pose"])
        ),
        "pairs_worsened": int(
            sum(1 for r in rows if r["final_d_pose"] > r["start_d_pose"])
        ),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=waterfill -- edits and carrier solved as ONE problem, not composed candidates.
# --------------------------------------------------------------------------
#
# ``ddm_jg3`` solved seg ALONE.  The direction it moved along is seg-descending but
# NOT pose-null, and the measured price of that omission is the whole finding of this
# arm.  The cure is not to price two finished candidates against each other but to
# solve the admission JOINTLY
# ([[edits_and_drops_are_one_waterfill_solve_jointly_20260819]]).
#
# Each pair has exactly two admissible states, and BOTH legs of both states are
# measured, never modelled:
#
#   DROP  -- frame 1 reverts to the base decode and the carrier reverts to br1's
#            codes, so the pair's pose is its base value and it costs no tokens.
#   KEEP  -- frame 1 is the edited render and the carrier is this arm's re-solve.
#
# The pose leg is sqrt-CONCAVE, so per-pair pose costs do NOT add in score units
# ([[concavity_helps_when_you_pay_the_axis_upward_20260818]]).  A fixed-ratio greedy
# is therefore wrong at the margin; the admission is swept over a Lagrange multiplier
# on pose damage and every candidate subset is scored EXACTLY through the real
# formula, with the greedy order kept as a control.


def _per_pair_legs(args) -> dict[str, np.ndarray]:
    """Load the three measured per-pair legs. Every one is a receipt on disk."""
    decomposition = json.loads(Path(args.seg_decomposition).read_text())
    seg = np.zeros(N_PAIRS, dtype=np.float64)
    for pair, delta in decomposition["per_pair_delta_nonzero"].items():
        seg[int(pair)] = float(delta)

    base_pose = np.load(args.base_pose)
    if base_pose.shape != (N_PAIRS,):
        raise Jg5Error(f"base pose vector has shape {base_pose.shape}")

    bits_candidate = np.load(args.bits_candidate).astype(np.float64)
    bits_control = np.load(args.bits_control).astype(np.float64)
    delta_bytes = (bits_candidate - bits_control) / 8.0

    # An edited pair with NO solved carrier row must fall back to the STALE-carrier
    # value it actually has, never to its base value. Falling back to base would give
    # an unsolved pair zero pose damage and let the sweep claim its seg credit for
    # free -- anti-conservative, and invisible in the output. The fallback is the
    # honest worst case and the count is refused rather than reported.
    stale_pose = np.load(args.candidate_pose)
    if stale_pose.shape != (N_PAIRS,):
        raise Jg5Error(f"candidate pose vector has shape {stale_pose.shape}")

    rows = merge_rows([Path(p) for p in args.rows])
    resolved = stale_pose.copy()
    solved_mask = np.zeros(N_PAIRS, dtype=bool)
    for pair, row in rows.items():
        resolved[pair] = float(row["final_d_pose"])
        solved_mask[pair] = True

    # The solver's own rows come from mixed ladder/polish chunk widths, and this
    # forward's value moves with the batch shape. Comparing them against a base
    # vector measured at ONE shape is a cross-instrument comparison. When a matched
    # re-measurement exists it REPLACES the rows' values, so KEEP/DROP is decided on
    # one instrument ([[batch_shape_is_part_of_the_forward_instrument_20260806]]).
    if args.resolved_pose:
        matched = np.load(args.resolved_pose)
        if matched.shape != (N_PAIRS,):
            raise Jg5Error(f"resolved pose vector has shape {matched.shape}")
        resolved = matched
        solved_mask[:] = True
    return {
        "seg": seg,
        "base_pose": base_pose,
        "stale_pose": stale_pose,
        "resolved_pose": resolved,
        "delta_bytes": delta_bytes,
        "solved": solved_mask,
    }


def _score_subset(
    keep: np.ndarray, legs: dict[str, np.ndarray]
) -> dict[str, float]:
    """Composed S for an admission mask. Seg and pose EXACT; rate MODELLED and LOW.

    The rate leg counts only the TOKEN bits the kept edits add.  It does NOT include
    the carrier's own re-encode cost: larger code deltas make the Rice payload grow,
    and that is only knowable by building the archive.  So ``score_modelled`` is an
    OPTIMISTIC bound on the rate side and the winner must be re-measured by
    ``mode=close`` before any of it is quoted as a row.
    """
    d_seg_jg1 = JG1_BASE_D_SEG + float(legs["seg"][keep].sum()) / N_PAIRS
    d_seg_t4 = d_seg_jg1 * (POINTER_D_SEG_T4 / JG1_BASE_D_SEG)
    pose = np.where(keep, legs["resolved_pose"], legs["base_pose"])
    d_pose = float(pose.mean())
    archive_bytes = POINTER_ARCHIVE_BYTES + float(legs["delta_bytes"][keep].sum())
    return {
        "pairs_kept": int(keep.sum()),
        "d_seg_jg1": d_seg_jg1,
        "d_seg_t4": d_seg_t4,
        "d_pose": d_pose,
        "archive_bytes_modelled": archive_bytes,
        "seg_leg": 100.0 * d_seg_t4,
        "pose_leg": pose_leg(d_pose),
        "rate_leg": 25.0 * archive_bytes / 37_545_489.0,
        "score_modelled": composed_score(d_seg_t4, d_pose, archive_bytes),
    }


def run_waterfill(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    legs = _per_pair_legs(args)
    edited = legs["seg"] != 0.0
    unsolved_edited = int((edited & ~legs["solved"]).sum())
    if unsolved_edited and not args.allow_unsolved:
        raise Jg5Error(
            f"{unsolved_edited} edited pairs have no solved carrier row; their pose "
            "would be priced at the stale-carrier value, which understates what the "
            "re-solve can buy -- finish the shards or pass --allow-unsolved to sweep "
            "on the conservative fallback"
        )

    # Score in score units per pair, for the multiplier sweep. The seg and rate legs
    # ARE additive; the pose leg is not, which is exactly why it is the multiplier's
    # subject rather than another additive term.
    ratio_t4 = POINTER_D_SEG_T4 / JG1_BASE_D_SEG
    seg_credit = -100.0 * legs["seg"] * ratio_t4 / N_PAIRS  # positive = a gain
    rate_cost = 25.0 * legs["delta_bytes"] / 37_545_489.0
    pose_damage = legs["resolved_pose"] - legs["base_pose"]

    none_kept = np.zeros(N_PAIRS, dtype=bool)
    all_kept = edited.copy()
    reference = _score_subset(none_kept, legs)
    full = _score_subset(all_kept, legs)

    lambdas = np.concatenate(
        [np.zeros(1), np.logspace(-3.0, 6.0, num=1801, dtype=np.float64)]
    )
    best: dict[str, Any] | None = None
    trace: list[dict[str, float]] = []
    seen: set[bytes] = set()
    for lam in lambdas:
        keep = edited & ((seg_credit - rate_cost) > lam * pose_damage)
        key = np.packbits(keep).tobytes()
        if key in seen:
            continue
        seen.add(key)
        scored = _score_subset(keep, legs)
        scored["lambda"] = float(lam)
        trace.append(scored)
        if best is None or scored["score_modelled"] < best["score_modelled"]:
            best = dict(scored)
            best["keep"] = keep.copy()

    if best is None:
        raise Jg5Error("multiplier sweep produced no subset")
    keep_best = best.pop("keep")
    # ALWAYS KEEP THE PAYLOAD: the admission mask and the pair list are the artifact
    # the re-encode consumes, so they go to disk before any scalar is reported.
    np.save(out_dir / "keep_mask.npy", keep_best)
    kept_pairs = sorted(int(p) for p in np.where(keep_best)[0])
    (out_dir / "kept_pairs.json").write_text(json.dumps(kept_pairs))

    # The subset edit field is written HERE, by the same code that made the decision,
    # so the re-encode consumes exactly the admission that was priced. Hand-filtering
    # it later is how a decision and its artifact drift apart.
    subset_path: str | None = None
    if args.edits_npz:
        source = np.load(args.edits_npz)
        missing = [p for p in kept_pairs if str(p) not in source.files]
        if missing:
            raise Jg5Error(
                f"{len(missing)} admitted pairs are absent from {args.edits_npz} "
                f"(first: {missing[:5]}); the admission and the edit field disagree"
            )
        subset_path = str(out_dir / "seg_edits_subset.npz")
        np.savez_compressed(
            subset_path, **{str(p): source[str(p)] for p in kept_pairs}
        )

    summary = {
        "schema": "ddm_jg5_waterfill.v1",
        "axis": (
            "seg leg from the jg1 DALI per-pair instrument projected onto T4; pose leg "
            "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]; rate leg "
            "MODELLED from per-frame token bits and MUST be re-measured by building"
        ),
        "score_claim": False,
        "promotable": False,
        "rate_leg_is_modelled_not_measured": True,
        "rate_leg_excludes_the_carrier_reencode_cost": True,
        "score_modelled_is_optimistic_on_rate": True,
        "edited_pairs": int(edited.sum()),
        "edited_pairs_without_a_solved_carrier_row": unsolved_edited,
        "reference_drop_everything": reference,
        "full_edit_set": full,
        "best": best,
        "best_kept_pairs": len(kept_pairs),
        "pointer_score_t4": POINTER_SCORE,
        "net_vs_pointer": best["score_modelled"] - POINTER_SCORE,
        "beats_pointer": bool(best["score_modelled"] < POINTER_SCORE),
        "under_0_15": bool(best["score_modelled"] < 0.15),
        "keep_mask": str(out_dir / "keep_mask.npy"),
        "subset_edits_npz": subset_path,
        "trace": trace,
    }
    (out_dir / "WATERFILL.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=close -- merge the shards, splice, prove d_seg invariance, price the row.
# --------------------------------------------------------------------------


def merge_rows(row_paths: list[Path]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for path in row_paths:
        for pair, row in br1.load_done(path).items():
            if pair in merged and merged[pair]["final_d_pose"] <= row["final_d_pose"]:
                continue
            merged[pair] = row
    return merged


def section_identity(
    candidate_bytes: bytes,
    base_bytes: bytes,
    *,
    runtime: Path = CANDIDATE_RUNTIME,
) -> dict[str, Any]:
    """Byte-diff every frame-1-producing section of two archives.

    SegNet reads only the odd frame (``upstream/modules.py:108``).  The odd frame
    comes from the hpac stream, the semantic stream and the token tail; if all three
    are byte-identical the seg leg provably cannot move.  This DERIVES the sections
    from each archive's own RX1 header rather than trusting a pinned offset.
    """
    import io
    import zipfile

    ra, _cr, _ar1, _cp = splice._import_runtime(runtime)

    def sections(data: bytes) -> dict[str, bytes]:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.namelist() != ["p"]:
                raise Jg5Error("archive must contain exactly member p")
            outer = archive.read("p")
        header = ra.RX1_MODEL_HEADER.unpack_from(outer)
        _reserved, hpac_bytes, semantic_bytes, carrier_bytes = header[4:]
        offset = ra.RX1_MODEL_HEADER.size
        hpac = outer[offset : offset + hpac_bytes]
        offset += hpac_bytes
        semantic = outer[offset : offset + semantic_bytes]
        offset += semantic_bytes
        carrier = outer[offset : offset + carrier_bytes]
        offset += carrier_bytes
        return {
            "hpac": hpac,
            "semantic": semantic,
            "carrier": carrier,
            "tail": outer[offset:],
        }

    new = sections(candidate_bytes)
    old = sections(base_bytes)
    result = {
        name: {
            "identical": new[name] == old[name],
            "bytes_candidate": len(new[name]),
            "bytes_base": len(old[name]),
        }
        for name in ("hpac", "semantic", "carrier", "tail")
    }
    result["frame1_sections_all_identical"] = bool(
        result["hpac"]["identical"]
        and result["semantic"]["identical"]
        and result["tail"]["identical"]
    )
    return result


def run_close(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = out_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_rows([Path(p) for p in args.rows])
    if not rows:
        raise Jg5Error(f"no solved rows at {args.rows}")
    if len(rows) != N_PAIRS and not args.allow_partial:
        raise Jg5Error(
            f"only {len(rows)}/{N_PAIRS} pairs solved; a population d_pose from a "
            "partial set is a different object -- pass --allow-partial to price anyway"
        )

    # The BODY is an argument. After a waterfill drops edits the token stream is
    # re-encoded, so the body this arm prices against is no longer the jg4 candidate;
    # pricing a subset carrier against the full-edit body would be a delta without its
    # baseline.
    runtime = Path(args.body_runtime)
    base_bytes = (runtime / "archive.zip").read_bytes()
    observed = hashlib.sha256(base_bytes).hexdigest()
    if args.expect_body_sha256 and observed != args.expect_body_sha256:
        raise Jg5Error(
            f"body sha256 {observed} != expected {args.expect_body_sha256}"
        )
    body_bytes = len(base_bytes)
    body = splice.parse_shipped_body(runtime, verify_sha=False)
    base_codes = np.asarray(body.codes, dtype=np.int32)

    # CONTROL: splicing the body's OWN codes back in must reproduce its bytes.
    # Without it a byte delta cannot be attributed to the re-solve rather than to the
    # rebuild, and the whole pricing is unanchored.
    identity = splice.build_archive(
        body, base_codes, runtime_dir=runtime, container_search=True
    )
    identity_ok = identity["archive_sha256"] == observed
    if not identity_ok and not args.allow_identity_drift:
        raise Jg5Error(
            "CONTROL FAILED: rebuilding the body from its own codes gives "
            f"{identity['archive_sha256']} ({identity['archive_size']} B), not "
            f"{observed} ({body_bytes} B)"
        )

    # The pose population is over ALL pairs. Pairs whose edit was dropped keep the
    # body's own (br1) codes AND revert to the base renders, so their pose is the base
    # value -- taking their damaged start value would price a frame they do not ship.
    eligible = set(rows)
    if args.carrier_pairs:
        allowed = set(json.loads(Path(args.carrier_pairs).read_text()))
        eligible &= allowed
    base_pose_vector = np.load(args.base_pose) if args.base_pose else None

    start_pose = np.zeros(N_PAIRS, dtype=np.float64)
    for pair in range(N_PAIRS):
        row = rows.get(pair)
        if pair in eligible and row is not None:
            start_pose[pair] = row["start_d_pose"]
        elif base_pose_vector is not None:
            start_pose[pair] = base_pose_vector[pair]
        elif row is not None:
            start_pose[pair] = row["start_d_pose"]
        else:
            raise Jg5Error(
                f"pair {pair} has neither a solved row nor a base pose value; pass "
                "--base-pose so dropped pairs are priced on the frames they ship"
            )
    d_pose_start = float(start_pose.mean())

    improving = [
        r
        for p, r in rows.items()
        if p in eligible and r["final_d_pose"] < r["start_d_pose"]
    ]
    improving.sort(key=lambda r: r["start_d_pose"] - r["final_d_pose"], reverse=True)
    if not improving:
        raise Jg5Error("no improving pairs to close")

    d_seg_t4 = (
        args.d_seg_jg1 * (POINTER_D_SEG_T4 / JG1_BASE_D_SEG)
        if args.d_seg_jg1
        else candidate_d_seg_t4_projected()
    )
    base_score = composed_score(d_seg_t4, d_pose_start, body_bytes)

    levels = sorted(
        {max(1, int(round(len(improving) * f))) for f in (0.1, 0.25, 0.5, 0.75, 1.0)}
    )
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for count in levels:
        chosen = improving[:count]
        candidate = base_codes.copy()
        for row in chosen:
            candidate[int(row["pair"])] = np.asarray(row["codes"], dtype=np.int32)
        built = splice.build_archive(
            body,
            candidate,
            runtime_dir=runtime,
            container_search=True,
            verify=True,
        )
        # ALWAYS KEEP THE PAYLOAD: the bytes exist in memory, so they are written
        # before anything is measured from them, and EVERY level is retained, not
        # only the winner.
        archive_bytes = int(built["archive_size"])
        level_path = archive_dir / f"archive_level_{count:04d}.zip"
        level_path.write_bytes(built["archive_bytes"])
        np.save(archive_dir / f"codes_level_{count:04d}.npy", candidate)

        gain = sum(r["start_d_pose"] - r["final_d_pose"] for r in chosen)
        # EXACT, not an approximation: pair pose residuals are independent, so the
        # population mean is the mean of per-pair values under the chosen code set.
        # The divisor is the POPULATION, never the solved count -- d_pose_start is
        # already a mean over all N_PAIRS.
        d_pose_new = d_pose_start - gain / N_PAIRS
        proof = section_identity(built["archive_bytes"], base_bytes, runtime=runtime)
        if not proof["frame1_sections_all_identical"]:
            raise Jg5Error(
                f"level {count}: a frame-1 section moved ({proof}); refusing to "
                "launder a seg leg through a changed odd-frame section"
            )
        score = composed_score(d_seg_t4, d_pose_new, archive_bytes)
        row_out = {
            "pairs_admitted": count,
            "archive_bytes": archive_bytes,
            "delta_bytes_vs_body": archive_bytes - body_bytes,
            "archive_sha256": built["archive_sha256"],
            "archive_path": str(level_path),
            "d_pose_final": d_pose_new,
            "delta_score_pose": pose_leg(d_pose_new) - pose_leg(d_pose_start),
            "delta_score_rate": (archive_bytes - body_bytes) * BYTE_TO_SCORE,
            "composed_score": score,
            "net_delta_score_vs_body": score - base_score,
            "pose_report_bound": pose_report_bound(d_pose_new),
            "frame1_sections": proof,
            "container": built["container"],
        }
        results.append(row_out)
        print(
            f"[level {count}/{len(improving)}] bytes {archive_bytes} "
            f"(dB {archive_bytes - body_bytes:+d})  "
            f"d_pose {d_pose_new:.6e}  S {score:.8f}",
            flush=True,
        )
        if best is None or score < best["composed_score"]:
            best = dict(row_out)
            best["codes"] = candidate

    if best is None:
        raise Jg5Error("no level produced a result")
    codes_path = out_dir / "jg5_candidate_codes.npy"
    np.save(codes_path, best.pop("codes"))

    summary = {
        "schema": "ddm_jg5_close.v1",
        "axis": (
            "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT] for the "
            "pose leg; seg leg projected from the jg1 DALI instrument onto the T4 axis; "
            "rate leg EXACT (archive bytes)"
        ),
        "score_claim": False,
        "promotable": False,
        "control_identity_rebuild_is_byte_identical": identity_ok,
        "control_identity_observed": {
            "sha256": identity["archive_sha256"],
            "bytes": identity["archive_size"],
        },
        "pointer": {
            "archive_sha256": POINTER_ARCHIVE_SHA256,
            "archive_bytes": POINTER_ARCHIVE_BYTES,
            "score_t4": POINTER_SCORE,
            "d_seg_t4": POINTER_D_SEG_T4,
            "d_pose_dali": POINTER_D_POSE_DALI,
        },
        "body_before_resolve": {
            "runtime": str(runtime),
            "archive_sha256": observed,
            "archive_bytes": body_bytes,
            "d_seg_jg1_instrument": args.d_seg_jg1 or JG1_CAND_D_SEG,
            "d_seg_t4_projected": d_seg_t4,
            "d_pose_dali_stale_carrier": d_pose_start,
            "composed_score": base_score,
        },
        "rows_solved": len(rows),
        "carrier_pairs_eligible": len(eligible),
        "pairs_improving": len(improving),
        "admit_bar": ADMIT_BAR,
        "levels": results,
        "best": best,
        "beats_pointer": bool(best["composed_score"] < POINTER_SCORE),
        "under_0_15": bool(best["composed_score"] < 0.15),
        "candidate_codes": str(codes_path),
    }
    (out_dir / "CLOSE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "levels"}, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=retain -- the P0 custody receipt for everything this arm materialised.
# --------------------------------------------------------------------------
#
# ALWAYS KEEP THE PAYLOAD is a precondition for RUNNING, not a reporting step, so
# every mode above writes its bytes before it measures them.  This mode only walks
# what is already on disk and records sha256 + size so the next consumer can prove
# byte-identity.  Third arm in a row to need this (br1, jg4, jg5): a canonical
# ``tools/build_retention_manifest.py`` is OWED per the least-hand-typing law.


def run_retain(args) -> int:
    root = Path(args.root)
    if not root.is_dir():
        raise Jg5Error(f"custody root does not exist: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        relative = path.relative_to(root)
        if any(part in {"logs"} for part in relative.parts) and not args.include_logs:
            continue
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema": "ddm_jg5_retention_manifest.v1",
        "arm": "ddm_jg5",
        "date": args.date,
        "axis": (
            "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT] pose; "
            "seg leg projected from the jg1 DALI instrument; rate leg EXACT"
        ),
        "score_claim": False,
        "promotable": False,
        "headline": args.headline,
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    out = Path(args.out)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    return 0


def run_codes(args) -> int:
    """Materialise a (600,12) code lattice from solved rows, for matched measurement.

    The solver's rows carry per-pair codes; scoring them at ONE batch shape requires
    the whole lattice in one array. Pairs with no row keep the body's shipped codes.
    """
    rows = merge_rows([Path(p) for p in args.rows])
    if not rows:
        raise Jg5Error(f"no rows at {args.rows}")
    inst_codes = np.asarray(
        up2.load_carrier_state(CANDIDATE_RUNTIME, verify_archive=False).codes,
        dtype=np.int32,
    )
    codes = inst_codes.copy()
    for pair, row in rows.items():
        codes[pair] = np.asarray(row["codes"], dtype=np.int32)
    np.save(args.out, codes)
    print(
        json.dumps(
            {
                "out": args.out,
                "rows": len(rows),
                "pairs_differing_from_body": int((codes != inst_codes).any(axis=1).sum()),
                "coordinates_changed": int((codes != inst_codes).sum()),
            },
            indent=2,
        )
    )
    return 0


def run_select(args) -> int:
    """Per-pair, keep whichever code lattice measures better AT THE SAME BATCH SHAPE.

    Two lattices (the budget-stopped solve and the materiality-stopped refine) are each
    scored by ``mode=baseline`` at one declared shape. Choosing between them on the
    solver's own rows would compare a batch-10 number against a batch-32 one and could
    pick the worse lattice on instrument noise
    ([[batch_shape_is_part_of_the_forward_instrument_20260806]]). This chooses on the
    matched vectors only, and writes the chosen pose vector beside the chosen codes so
    the waterfill consumes exactly what was compared.
    """
    codes_a = np.load(args.codes_a).astype(np.int32)
    codes_b = np.load(args.codes_b).astype(np.int32)
    pose_a = np.load(args.pose_a)
    pose_b = np.load(args.pose_b)
    for name, arr, shape in (
        ("codes_a", codes_a, (N_PAIRS, K)),
        ("codes_b", codes_b, (N_PAIRS, K)),
        ("pose_a", pose_a, (N_PAIRS,)),
        ("pose_b", pose_b, (N_PAIRS,)),
    ):
        if arr.shape != shape:
            raise Jg5Error(f"{name} has shape {arr.shape}, expected {shape}")

    take_b = pose_b < pose_a
    chosen_codes = np.where(take_b[:, None], codes_b, codes_a)
    chosen_pose = np.where(take_b, pose_b, pose_a)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "chosen_codes.npy", chosen_codes)
    np.save(out_dir / "chosen_d_pose_per_pair.npy", chosen_pose)
    report = {
        "schema": "ddm_jg5_select.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
        "matched_batch_shape": args.batch_size,
        "codes_a": args.codes_a,
        "codes_b": args.codes_b,
        "pairs_taking_b": int(take_b.sum()),
        "mean_d_pose_a": float(pose_a.mean()),
        "mean_d_pose_b": float(pose_b.mean()),
        "mean_d_pose_chosen": float(chosen_pose.mean()),
        "chosen_codes": str(out_dir / "chosen_codes.npy"),
        "chosen_pose": str(out_dir / "chosen_d_pose_per_pair.npy"),
    }
    (out_dir / "SELECT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    select = sub.add_parser("select", help="per-pair best of two lattices, one shape")
    select.add_argument("--codes-a", required=True)
    select.add_argument("--pose-a", required=True)
    select.add_argument("--codes-b", required=True)
    select.add_argument("--pose-b", required=True)
    select.add_argument("--batch-size", type=int, default=8)
    select.add_argument("--out", required=True)
    select.set_defaults(func=run_select)

    codes = sub.add_parser("codes", help="merge rows into a (600,12) code lattice")
    codes.add_argument("--rows", nargs="+", required=True)
    codes.add_argument("--out", required=True)
    codes.set_defaults(func=run_codes)

    control = sub.add_parser("control", help="forward-model + codes-identity controls")
    control.add_argument("--out", required=True)
    control.add_argument("--expect-raw-sha256", default=None)
    control.add_argument(
        "--br1-codes",
        default="/Volumes/APDataStore/pact/ddm_br1/retained/byte_close_n600/"
        "br1_candidate_codes.npy",
    )
    control.set_defaults(func=run_control)

    refine = sub.add_parser("refine", help="re-solve to diminishing returns, not a budget")
    refine.add_argument("--rows", nargs="+", required=True)
    refine.add_argument("--out", required=True)
    refine.add_argument("--targets", default=None, help="npy of pair indices")
    refine.add_argument("--shard-index", type=int, default=0)
    refine.add_argument("--shard-count", type=int, default=1)
    refine.add_argument("--outer-rounds", type=int, default=40)
    refine.add_argument("--max-gn-iterations", type=int, default=400)
    refine.add_argument(
        "--operating-point-d-pose",
        type=float,
        required=True,
        help="mean d_pose of the current waterfill kept set; sets the derived floor",
    )
    refine.add_argument("--expect-raw-sha256", default=None)
    refine.set_defaults(func=run_refine)

    baseline = sub.add_parser("baseline", help="exact n600 d_pose of the stale carrier")
    baseline.add_argument("--out", required=True)
    baseline.add_argument("--expect-raw-sha256", default=None)
    baseline.add_argument("--batch-size", type=int, default=8)
    baseline.add_argument("--tag", default="candidate")
    baseline.add_argument(
        "--codes",
        default=None,
        help="npy (600,12) int32 to score instead of the archive's own codes",
    )
    baseline.add_argument(
        "--base-raw",
        default=str(up2.DEFAULT_RAW),
        help="base decode, scored with the SAME carrier so only frame 1 differs",
    )
    baseline.set_defaults(func=run_baseline)

    gn = sub.add_parser("gn", help="damped Gauss-Newton re-solve on candidate renders")
    gn.add_argument("--out", required=True)
    gn.add_argument("--shard-index", type=int, default=0)
    gn.add_argument("--shard-count", type=int, default=1)
    gn.add_argument("--limit", type=int, default=0, help="cap pairs (smoke only)")
    gn.add_argument("--iterations", type=int, default=6)
    gn.add_argument("--no-polish", action="store_true")
    gn.add_argument("--expect-raw-sha256", default=None)
    gn.set_defaults(func=run_gn)

    water = sub.add_parser("waterfill", help="joint edit+carrier admission sweep")
    water.add_argument("--rows", nargs="+", required=True)
    water.add_argument("--out", required=True)
    water.add_argument(
        "--seg-decomposition",
        default="/Volumes/APDataStore/pact/ddm_jg4/retained/"
        "perpair_dseg_decomposition.json",
    )
    water.add_argument(
        "--base-pose",
        default="/Volumes/APDataStore/pact/ddm_jg5/work/baseline/"
        "d_pose_per_pair_base_odd_frames.npy",
    )
    water.add_argument(
        "--candidate-pose",
        default="/Volumes/APDataStore/pact/ddm_jg5/work/baseline/"
        "d_pose_per_pair_candidate.npy",
        help="stale-carrier per-pair pose; the conservative fallback for unsolved pairs",
    )
    water.add_argument("--allow-unsolved", action="store_true")
    water.add_argument(
        "--resolved-pose",
        default=None,
        help="matched-batch-shape per-pair pose for the chosen codes; replaces the rows",
    )
    water.add_argument(
        "--edits-npz",
        default="/Volumes/APDataStore/pact/ddm_jg3/retained/seg_edits_n600_complete.npz",
        help="full edit field; the admitted subset is written beside the mask",
    )
    water.add_argument(
        "--bits-candidate",
        default="/Volumes/APDataStore/pact/ddm_jg4/retained/"
        "bits_per_frame_complete_n600.npy",
    )
    water.add_argument(
        "--bits-control",
        default="/Volumes/APDataStore/pact/ddm_jg4/retained/"
        "bits_per_frame_control_600.npy",
    )
    water.set_defaults(func=run_waterfill)

    close = sub.add_parser("close", help="merge shards, splice, prove, price")
    close.add_argument("--rows", nargs="+", required=True)
    close.add_argument("--out", required=True)
    close.add_argument("--allow-partial", action="store_true")
    close.add_argument("--allow-identity-drift", action="store_true")
    close.add_argument(
        "--body-runtime",
        default=str(CANDIDATE_RUNTIME),
        help="the body to splice into; a subset re-encode moves it off the jg4 candidate",
    )
    close.add_argument("--expect-body-sha256", default=CANDIDATE_ARCHIVE_SHA256)
    close.add_argument(
        "--carrier-pairs",
        default=None,
        help="json list of pairs whose re-solved codes may be used (waterfill keep set)",
    )
    close.add_argument(
        "--base-pose",
        default="/Volumes/APDataStore/pact/ddm_jg5/work/baseline/"
        "d_pose_per_pair_base_odd_frames.npy",
        help="per-pair pose for pairs whose edit was dropped (they ship base frames)",
    )
    close.add_argument(
        "--d-seg-jg1",
        type=float,
        default=None,
        help="the body's seg leg on the jg1 instrument; defaults to the full-edit value",
    )
    close.set_defaults(func=run_close)

    retain = sub.add_parser("retain", help="sha256 + size receipt for every payload")
    retain.add_argument("--root", required=True)
    retain.add_argument("--out", required=True)
    retain.add_argument("--date", default="2026-08-19")
    retain.add_argument("--headline", default="")
    retain.add_argument("--include-logs", action="store_true")
    retain.set_defaults(func=run_retain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
