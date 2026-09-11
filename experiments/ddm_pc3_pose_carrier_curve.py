"""ddm_pc3 -- price the pose carrier's rate/distortion CURVE on the move-44 field.

WHY THIS ARM EXISTS
-------------------
At pointer move 44 the pose term is ``sqrt(10 * 4.59e-06) = 0.006775`` and its
MARGINAL value is ``5 / sqrt(10 * d_pose) = 738`` score units per unit d_pose --
7.4x the seg term's constant 100.  Every carrier pass so far RE-SOLVED the shipped
lattice; nobody priced what MORE carrier capacity buys.  This arm prices that curve.

THE TWO WALLS, DERIVED BEFORE ANY MEASUREMENT
---------------------------------------------
1. **The absolute-gain wall.**  The pose term is 0.006775 S in total.  Driving
   ``d_pose`` to exactly ZERO gains at most 0.006775 S, which at the exchange
   ``25/37_545_489 = 6.658589531221714e-07`` S/B is worth at most **10,174.9 B**.
   Any capacity increase that costs more than that can NEVER pay, whatever it buys.
   The shipped basis is 12 atoms x 3 x 24 x 32 at 5 bits; one more dozen atoms is
   ~12.3 kB, so a rank doubling is dead by arithmetic before a single encode.
2. **The span wall.**  Refining the coefficient lattice -- a smaller step, more
   coefficient bits, a finer quantiser, in any combination -- cannot beat the
   CONTINUOUS optimum inside the shipped 12-dimensional span.  That continuous
   optimum is measurable with the real renderer and the real frozen scorer, at zero
   archive cost, and it BOUNDS the entire family in one measurement.  ``mode=ceiling``
   measures it.  ``mode=base`` measures the lattice point actually shipped.  The
   difference is every byte the family could ever buy.

WHAT IS A RECEIVER CHANGE AND WHAT IS NOT
------------------------------------------
``cpr1/carrier_codec.py`` hardcodes ``BASIS_BITS = 5`` and ``COEFFICIENT_BITS = 12``,
and ``up2.CARRIER_DIM``/``inflate.CARRIER_H``/``CARRIER_W`` fix the rank and the atom
shape in receiver CODE.  So rank, basis precision and coefficient WIDTH are all
receiver changes (first-measurement contract).  The coefficient SCALES are twelve
float32 words INSIDE the archive, so refining the lattice STEP is a pure data change
with the receiver untouched -- and ``max|code| = 154`` of the +-2047 field leaves
13.3x of headroom, so several halvings fit the shipped container.  That is the only
capacity axis this arm can seal by itself, and it is exactly the axis the ceiling
bounds.

WHY THE SOLVER HAD TO CHANGE FOR THE FINE RUNGS (pc2 ITEM 3)
-------------------------------------------------------------
``jg5.refine_pair``'s polish is +-2 LATTICE units, so at step /2^m it spans 2^-m of
the coefficient distance it spanned on the shipped lattice: the search shrinks exactly
as fast as the lattice refines.  pc2 measured that failure at x1/8 (2,500x worse on
pair 0) and named the cure -- "a polish radius expressed in COEFFICIENT units rather
than lattice units" -- as the only route that could reopen its FORMULATION-scope
refusal.  ``polish_offsets`` here is that cure: the neighbourhood is a fixed
COEFFICIENT radius realised on whichever lattice is live, so the rungs are compared at
constant search reach.  The GN leg was already scale-invariant (``step / scales``).

AUTHORITY
---------
Frozen CPU-torch PoseNet, GT poses on the DALI lineage (the lineage
``upstream/evaluate.py`` uses on the contest-CUDA axis the pointer row was measured
on), n600 always.  ``[macOS-CPU advisory]``; ``score_claim=false``,
``promotable=false``.  Every comparison here is same-instrument: the base leg is
measured on the POINTER's own configuration on THIS instrument first (the pose-base
law), and no number is ever compared against a T4 print.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

# --------------------------------------------------------------------------------------
# The body this arm measures. Every value is a receipt, not a memory.
# --------------------------------------------------------------------------------------

#: Pointer move 44 (lane ddm_rlc5_counted_rider_rebase_move43_first_measurement_20260910).
MOVE44_TREE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime"
)
MOVE44_ARCHIVE_SHA256 = (
    "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
)
MOVE44_ARCHIVE_BYTES = 180_406
#: The row's own COLD public decode -- the frames the shipped body actually produces.
#: PoseNet reads frame 1 from here; frame 0 is re-rendered from the carrier under test.
MOVE44_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw"
)
MOVE44_RAW_SHA256 = (
    "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc"
)

#: The exact T4 row. Printed for context only -- never compared against a local number.
MOVE44_D_SEG_T4 = 0.00010345
MOVE44_D_POSE_T4 = 4.59e-06
MOVE44_SCORE_T4 = 0.1372449041713402

N_PAIRS = 600
CAMERA_H, CAMERA_W = 874, 1164
BYTE_TO_SCORE = 25.0 / 37_545_489.0
#: A net score move must clear this to be worth a dispatch (negative is better).
ADMIT_BAR = -2e-5


class Pc3Error(RuntimeError):
    """A ddm_pc3 precondition failed. Fail closed, never approximate."""


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * float(d_pose))


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return (
        100.0 * float(d_seg)
        + pose_leg(d_pose)
        + 25.0 * int(archive_bytes) / 37_545_489.0
    )


def set_threads(threads: int) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    import torch

    torch.set_num_threads(threads)


# --------------------------------------------------------------------------------------
# The instrument
# --------------------------------------------------------------------------------------


def build_instrument(*, verify_raw: bool = False):
    """``br1.Instrument`` pinned to move 44's archive and move 44's own cold decode."""
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_up2_shipping_pose_solve as up2

    observed = _sha256_file(MOVE44_TREE / "archive.zip")
    if observed != MOVE44_ARCHIVE_SHA256:
        raise Pc3Error(
            f"carrier runtime {MOVE44_TREE} has archive sha {observed}, not move 44's "
            f"{MOVE44_ARCHIVE_SHA256}; a carrier measured against a different body is a "
            "different object, and every code would still be a valid int12"
        )
    if verify_raw:
        raw_sha = _sha256_file(MOVE44_RAW)
        if raw_sha != MOVE44_RAW_SHA256:
            raise Pc3Error(
                f"decode {MOVE44_RAW} has sha {raw_sha}, not {MOVE44_RAW_SHA256}"
            )
    state = up2.load_carrier_state(MOVE44_TREE, verify_archive=False)
    if state.has_compensation:
        raise Pc3Error("this body carries a compensation overlay; unmeasured path")
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise Pc3Error(
            f"GT pose lineage is {lineage}, not {up2.LINEAGE_DALI}: that is the "
            "contest-CPU objective, a different object (evaluate.py:31-42)"
        )
    gate = up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    raw = np.memmap(
        MOVE44_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3),
    )
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    instrument = br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)
    meta = {
        "runtime_dir": str(MOVE44_TREE),
        "archive_sha256": observed,
        "archive_bytes": MOVE44_ARCHIVE_BYTES,
        "raw": str(MOVE44_RAW),
        "raw_sha256_verified": bool(verify_raw),
        "gt_cache": str(up2.DEFAULT_DALI_GT),
        "gt_lineage_gate": gate,
        "shipped_codes_sha256": _sha256_array(np.asarray(state.codes, dtype=np.int32)),
        "shipped_max_abs_code": int(np.abs(np.asarray(state.codes)).max()),
        "coefficient_scales": np.asarray(
            state.coefficient_scales, dtype=np.float64
        ).reshape(-1).tolist(),
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    return instrument, meta


def shard_of(pairs: np.ndarray, shard_index: int, shard_count: int) -> np.ndarray:
    """STRIDED shard, never a contiguous block: a dead shard leaves an UNBIASED partial.

    The pose axis is the one where a contiguous prefix measures 2.54-4.21x harder than
    the population ([[m96]]), and this field's pose mass is concentrated in the 57-91
    scene block, so a block shard would be the worst possible partial.
    """
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise Pc3Error(f"bad shard {shard_index}/{shard_count}")
    return pairs[shard_index::shard_count]


# --------------------------------------------------------------------------------------
# Realized evaluation at arbitrary REAL coefficients (the ceiling's objective)
# --------------------------------------------------------------------------------------


def evaluate_coefficients(inst, pair: int, coefficient_block: np.ndarray) -> np.ndarray:
    """Realized d_pose for a block of REAL coefficient rows, exact receiver path.

    .. note::
       ``ddm_up2_shipping_pose_solve`` is imported at call scope throughout this
       module, matching ``br1``/``jg5``: the receiver modules it loads are read out
       of a runtime tree on ``sys.path`` and must not be bound at import time.

    ``br1.evaluate_codes`` is the same function restricted to integer codes: it calls
    ``up2.codes_to_coefficients``, which casts to int32.  The continuous ceiling needs
    the un-projected objective, so the coefficients are built here instead.  Everything
    downstream -- bicubic to 384x512, ``round`` to uint8, bicubic to camera, ``round``
    again, the selector override, PoseNet -- is the shipped path untouched, so the
    ceiling is still bounded by the PIXEL lattice.  That is the honest floor: it is what
    infinite coefficient precision would actually buy, not what a linear model predicts.
    """
    import ddm_up2_shipping_pose_solve as up2
    import torch

    block = np.asarray(coefficient_block, dtype=np.float64)
    if block.ndim != 2:
        raise Pc3Error(f"coefficient block must be 2-D, got ndim={block.ndim}")
    if len(block) == 0:
        return np.zeros(0, dtype=np.float64)
    index = np.array([pair], dtype=np.int64)
    frame1 = inst_frame1(inst, index)
    target_row = inst.targets[pair]
    out = np.zeros(len(block), dtype=np.float64)
    batch = 32
    for start in range(0, len(block), batch):
        chunk = block[start : start + batch]
        coefficients = torch.from_numpy(np.ascontiguousarray(chunk)).float()
        indices = np.full(len(chunk), pair, dtype=np.int64)
        frames1 = frame1.expand(len(chunk), -1, -1, -1).contiguous()
        with torch.inference_mode():
            frame0 = up2.render_frame0(
                coefficients, inst.state, indices, differentiable=False
            )
            pose = up2.pose_from_frames(inst.posenet, frame0, frames1)
        out[start : start + len(chunk)] = (
            (pose.to(torch.float64).numpy() - target_row[None]) ** 2
        ).mean(axis=1)
    return out


def inst_frame1(inst, index: np.ndarray):
    import ddm_up2_shipping_pose_solve as up2

    return up2.frames_to_bchw(inst.raw[2 * index + 1])


def jacobian_at(inst, pair: int, coefficients_row: np.ndarray):
    """Per-pair Jacobian d(pose)/d(coeff) and residual at REAL coefficients."""
    import ddm_up2_shipping_pose_solve as up2
    import torch

    index = np.array([pair], dtype=np.int64)
    frame1 = inst_frame1(inst, index)
    tb = torch.from_numpy(inst.targets[pair][None]).float()
    coeff = torch.from_numpy(
        np.ascontiguousarray(np.asarray(coefficients_row, dtype=np.float64)[None])
    ).float()
    jac, res, _pose = up2.jacobian_and_residual(
        inst.posenet, inst.state, coeff, frame1, tb, index
    )
    return jac, res


#: Geometric ladder of step fractions, extended well below ``br1.STEP_LADDER``'s 0.015
#: floor.  The continuous objective is piecewise-constant under the two ``round`` calls
#: in the render, so the search must be allowed to walk down to the pixel lattice before
#: it may report "no improving step" -- otherwise the ceiling is a SEARCH artefact and
#: would understate the gain the family could buy.
CONTINUOUS_LADDER = (
    1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03, 0.015,
    0.008, 0.004, 0.002, 0.001, 5e-4, 2e-4, 1e-4,
)

#: Polish radii in COEFFICIENT units, expressed as multiples of the SHIPPED lattice
#: step.  This is pc2 ITEM 3's cure: a neighbourhood that does not shrink when the
#: lattice does, so every rung is searched at the same physical reach.
POLISH_RADII_IN_SHIPPED_STEPS = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)


def continuous_refine_pair(
    inst,
    pair: int,
    start_coefficients: np.ndarray,
    *,
    dd_threshold: float,
    outer_rounds: int = 24,
    max_gn_iterations: int = 60,
    shipped_step: np.ndarray | None = None,
) -> dict[str, Any]:
    """Damped Gauss-Newton in CONTINUOUS coefficient space, real objective only.

    Same shape as ``jg5.refine_pair`` -- alternate a GN leg with a neighbourhood polish
    until a whole round buys nothing or the projected remaining gain falls under the
    pair's derived materiality floor -- with exactly one thing removed: the projection
    onto the int12 lattice.  The value returned is therefore the floor of EVERY lattice
    refinement inside this span: a finer step, more coefficient bits, a better
    quantiser, in any combination, all live above it.
    """
    import ddm_br1_pose_basis_reorientation as br1

    scales = np.asarray(inst.state.coefficient_scales, dtype=np.float64).reshape(-1)
    if shipped_step is None:
        shipped_step = scales
    current = np.asarray(start_coefficients, dtype=np.float64).copy()
    best = float(evaluate_coefficients(inst, pair, current[None])[0])
    start_value = best
    history = [best]
    evaluations = 1
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
            jac, res = jacobian_at(inst, pair, current)
            step = br1.min_image_norm_step(
                jac[0].double(), res[0].double(), inst.gram
            ).numpy()
            block = np.stack(
                [current + fraction * step for fraction in CONTINUOUS_LADDER]
            )
            values = evaluate_coefficients(inst, pair, block)
            evaluations += len(block)
            winner = int(values.argmin())
            if values[winner] >= best:
                inner_stop = "no_improving_step"
                break
            step_gains.append(best - float(values[winner]))
            best = float(values[winner])
            current = block[winner].copy()
            history.append(best)
            gn_iterations += 1
            remaining, ratio = projected_remaining_gain(step_gains)
            last_ratio, last_remaining = ratio, remaining
            if remaining <= dd_threshold:
                inner_stop = "converged_below_materiality_floor"
                break

        while True:
            block = polish_block(current, shipped_step)
            if len(block) == 0:
                break
            values = evaluate_coefficients(inst, pair, block)
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
            stop_reason = (
                inner_stop
                if inner_stop != "gn_iteration_budget"
                else "no_improving_step"
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
        "ratio": best / start_value if start_value > 0 else 1.0,
        "evaluations": evaluations,
        "coefficients": current.tolist(),
        "codes_if_projected": np.rint(current / scales).astype(np.int64).tolist(),
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
    }


def polish_block(current: np.ndarray, shipped_step: np.ndarray) -> np.ndarray:
    """Single-coordinate neighbourhood at a ladder of COEFFICIENT radii.

    The radii are multiples of the SHIPPED lattice step, so the neighbourhood spans the
    same physical coefficient distance no matter which lattice the caller is on.  That
    is the whole of pc2 ITEM 3: ``jg5.refine_pair``'s +-2 LATTICE polish shrinks with the
    lattice, which is why its x1/8 rung landed 2,500x worse on pair 0 and why that
    failure was a SEARCH failure rather than a representation one.
    """
    rows = []
    for radius in POLISH_RADII_IN_SHIPPED_STEPS:
        offset = radius * np.asarray(shipped_step, dtype=np.float64)
        for dim in range(len(current)):
            for sign in (-1.0, 1.0):
                trial = current.copy()
                trial[dim] += sign * offset[dim]
                rows.append(trial)
    if not rows:
        return np.zeros((0, len(current)), dtype=np.float64)
    return np.stack(rows)


def projected_remaining_gain(gains: list[float]) -> tuple[float, float]:
    """Geometric extrapolation of the remaining gain from the measured decay.

    Identical in form to ``jg5.projected_remaining_gain``: the stop is the pair's own
    measured decay against a DERIVED floor, never a hand-set iteration count.
    """
    if len(gains) < 2:
        return math.inf, math.nan
    previous, latest = float(gains[-2]), float(gains[-1])
    if previous <= 0.0 or latest <= 0.0:
        return 0.0, 0.0
    ratio = latest / previous
    if ratio >= 1.0:
        return math.inf, ratio
    return latest * ratio / (1.0 - ratio), ratio


def materiality_dd_threshold(operating_point_d_pose: float) -> float:
    """Per-pair d_pose floor below which a gain cannot reach the admission bar.

    DERIVED, not chosen: one pair's share of the mean is ``1/600``, the score
    sensitivity at the operating point is ``5 / sqrt(10 * d_pose)``, and the smallest
    score move this arm will act on is ``|ADMIT_BAR|``.  Stopping below this floor can
    never change a verdict.
    """
    sensitivity = 5.0 / math.sqrt(10.0 * float(operating_point_d_pose))
    return abs(ADMIT_BAR) / (sensitivity * N_PAIRS)


# --------------------------------------------------------------------------------------
# mode=reach -- the UNCONSTRAINED-field floor: what ANY number of atoms could ever buy
# --------------------------------------------------------------------------------------
#
# The shipped frame 0 is not a reconstruction of anything: it is
# ``round(clamp(127.5 + 64 * bicubic(F)))``, where ``F`` is a 3 x 24 x 32 field that the
# carrier constrains to the 12-atom span.  So the carrier's reach is a 12-dimensional
# slice of a 2,304-dimensional field space, and the floor of EVERY possible rank -- 13
# atoms, 24, 2,304, a different basis, a per-pair basis -- is the unconstrained field
# optimum.  Measuring it decides the one question the rank axis turns on: is the residual
# on the pairs that carry the pose mass a RANK limit (extra atoms could buy it) or a
# RENDER limit (nothing in this carrier family can)?
#
# A rank change is a RECEIVER change (``up2.CARRIER_DIM`` is receiver code), so this
# number can never be sealed by this arm.  It is measured because a bound that closes a
# receiver-change family is worth more than a rung that cannot be sealed anyway.


def dct_completion_atoms(basis_raw: np.ndarray, extra: int) -> np.ndarray:
    """``extra`` deterministic 3x24x32 atoms spanning the complement of the shipped 12.

    The completion is the separable 2D DCT-II product basis at 24x32, one channel at a
    time, taken in increasing (u + v) frequency order and Gram-Schmidt orthogonalised
    against the twelve SHIPPED raw atoms and against each other in the 2,304-dimensional
    field inner product.  Two properties matter and neither is an assumption:

    * it is GENERIC -- a closed-form transform of nothing but the atom geometry, so the
      reach number it produces is a property of the renderer, not of a fit to this video;
    * it is a strict ENLARGEMENT -- the shipped twelve are unchanged and the new
      directions are exactly the ones they do not already reach, so rank R's floor is a
      genuine floor for rank R and not a re-parametrisation of rank 12.

    pc1 closed REPLACING the basis with a generated DCT one (39,748x past break-even at
    d_pose >= 0.9986).  This does not replace it; it appends to it, and only to measure a
    bound on what any larger rank could buy.
    """
    shipped = np.asarray(basis_raw, dtype=np.float64).reshape(len(basis_raw), -1)
    channels, height, width = basis_raw.shape[1:]
    # The twelve shipped atoms are NOT mutually orthogonal, so projecting against them
    # one at a time does not remove their span.  An SVD gives an orthonormal basis of
    # exactly that span, which is the object the completion must be orthogonal to.
    _unitary, singular, right = np.linalg.svd(shipped, full_matrices=False)
    keep = singular > singular.max() * 1e-10
    orthonormal = [row.copy() for row in right[keep]]
    grid_u = np.arange(height)[:, None]
    grid_v = np.arange(width)[:, None]
    cos_h = np.cos(np.pi * (np.arange(height)[None, :] + 0.5) * grid_u / height)
    cos_w = np.cos(np.pi * (np.arange(width)[None, :] + 0.5) * grid_v / width)
    order = sorted(
        ((u + v, u, v) for u in range(height) for v in range(width)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    produced: list[np.ndarray] = []
    for _total, u, v in order:
        if len(produced) >= extra:
            break
        plane = np.outer(cos_h[u], cos_w[v])
        for channel in range(channels):
            if len(produced) >= extra:
                break
            atom = np.zeros((channels, height, width), dtype=np.float64)
            atom[channel] = plane
            vector = atom.reshape(-1)
            reference = float(np.linalg.norm(vector))
            # Two passes of modified Gram-Schmidt: one pass leaves a residual of the
            # order of the cancellation, and these candidates are heavily overlapped
            # with the shipped span.
            for _pass in range(2):
                for existing in orthonormal:
                    vector = vector - float(vector @ existing) * existing
            norm = float(np.linalg.norm(vector))
            # RELATIVE threshold: a direction the shipped span already covers has
            # nothing to add, and normalising its numerical dust would manufacture a
            # fake new dimension out of round-off.
            if not np.isfinite(norm) or norm <= 1e-7 * max(reference, 1.0):
                continue
            vector = vector / norm
            orthonormal.append(vector)
            produced.append(vector.reshape(channels, height, width))
    if len(produced) < extra:
        raise Pc3Error(
            f"DCT completion produced {len(produced)} of {extra} requested atoms"
        )
    stacked = np.stack(produced)
    if not np.isfinite(stacked).all():
        raise Pc3Error("DCT completion produced a non-finite atom")
    return stacked


def build_extended_basis(inst, extra: int):
    """``normalized_basis`` over the shipped raw atoms PLUS ``extra`` completion atoms.

    The first twelve rows go through the identical receiver formula on the identical
    inputs, so they are the shipped ``basis_norm`` rows.  ``reach_control`` proves that
    on this body instead of asserting it.
    """
    import torch

    runtime = MOVE44_TREE
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / "cpr1"))
    try:
        import inflate as renderer_module  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    raw = np.asarray(inst.state.basis_raw.detach().cpu().numpy(), dtype=np.float64)
    if extra:
        addition = dct_completion_atoms(raw, extra)
        # Each completion atom is unit-norm in field space while the shipped atoms are
        # not; ``normalized_basis`` divides every row by its own RMS afterwards, so the
        # scale here only sets the coefficient UNITS of the new dimensions, never the
        # span.  Matching the shipped atoms' mean magnitude keeps the GN step ladder
        # meaningful across old and new dimensions.
        addition = addition * float(np.sqrt((raw**2).sum(axis=(1, 2, 3)).mean()))
        raw = np.concatenate([raw, addition], axis=0)
    stacked = torch.from_numpy(np.ascontiguousarray(raw)).to(
        inst.state.basis_raw.dtype
    )
    return renderer_module.normalized_basis(stacked)


def evaluate_extended(
    inst, pair: int, coefficient_block: np.ndarray, basis_extended
) -> np.ndarray:
    """Realized d_pose over an EXTENDED basis, through the receiver's own renderer."""
    import ddm_up2_shipping_pose_solve as up2
    import torch

    block = np.asarray(coefficient_block, dtype=np.float64)
    if len(block) == 0:
        return np.zeros(0, dtype=np.float64)
    index = np.array([pair], dtype=np.int64)
    frame1 = inst_frame1(inst, index)
    target_row = inst.targets[pair]
    out = np.zeros(len(block), dtype=np.float64)
    batch = 16
    for start in range(0, len(block), batch):
        chunk = block[start : start + batch]
        coefficients = torch.from_numpy(np.ascontiguousarray(chunk)).float()
        indices = np.full(len(chunk), pair, dtype=np.int64)
        frames1 = frame1.expand(len(chunk), -1, -1, -1).contiguous()
        with torch.inference_mode():
            frames = up2.render_frame0_float(
                coefficients, basis_extended, differentiable=False
            )
            frame0 = up2.apply_selector_float(
                frames, inst.state.selector_modes, inst.state.selector_choices[indices]
            )
            pose = up2.pose_from_frames(inst.posenet, frame0, frames1)
        out[start : start + len(chunk)] = (
            (pose.to(torch.float64).numpy() - target_row[None]) ** 2
        ).mean(axis=1)
    return out


def reach_control(inst, pair: int, coefficients_row: np.ndarray, basis_extended):
    """Prove the extended renderer reproduces the shipped render BIT-FOR-BIT at zero."""
    import ddm_up2_shipping_pose_solve as up2
    import torch

    width = int(basis_extended.shape[0])
    padded = np.zeros((1, width), dtype=np.float64)
    padded[0, : up2.CARRIER_DIM] = np.asarray(coefficients_row, dtype=np.float64)
    index = np.array([pair], dtype=np.int64)
    with torch.inference_mode():
        frames = up2.render_frame0_float(
            torch.from_numpy(padded).float(), basis_extended, differentiable=False
        )
        by_extended = up2.apply_selector_float(
            frames, inst.state.selector_modes, inst.state.selector_choices[index]
        )
        by_shipped = up2.render_frame0(
            torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(coefficients_row, dtype=np.float64)[None]
                )
            ).float(),
            inst.state,
            index,
            differentiable=False,
        )
    return {
        "pair": int(pair),
        "rank": width,
        "bit_identical": bool(torch.equal(by_extended, by_shipped)),
        "max_abs_pixel_difference": float((by_extended - by_shipped).abs().max()),
    }


def extended_jacobian_at(inst, pair: int, coefficients_row: np.ndarray, basis_extended):
    """Jacobian d(pose)/d(coeff) over the extended basis, at REAL coefficients."""
    import ddm_up2_shipping_pose_solve as up2
    import torch

    index = np.array([pair], dtype=np.int64)
    frame1 = inst_frame1(inst, index)
    target = torch.from_numpy(inst.targets[pair][None]).float()
    coefficients = (
        torch.from_numpy(
            np.ascontiguousarray(np.asarray(coefficients_row, dtype=np.float64)[None])
        )
        .float()
        .requires_grad_(True)
    )
    frames = up2.render_frame0_float(
        coefficients, basis_extended, differentiable=True
    )
    frame0 = up2.apply_selector_float(
        frames, inst.state.selector_modes, inst.state.selector_choices[index]
    )
    pose = up2.pose_from_frames(inst.posenet, frame0, frame1)
    rows = []
    for component in range(6):
        grad = torch.autograd.grad(
            pose[0, component], coefficients, retain_graph=component < 5
        )[0]
        rows.append(grad.reshape(-1))
    jacobian = torch.stack(rows, dim=0).detach().double()
    residual = (pose.detach().double() - target.double())[0]
    return jacobian, residual


def reach_refine_pair(
    inst,
    pair: int,
    start_coefficients: np.ndarray,
    basis_extended,
    *,
    dd_threshold: float,
    outer_rounds: int = 16,
) -> dict[str, Any]:
    """Minimum-norm Gauss-Newton over the EXTENDED basis, realized objective only.

    The metric is the identity in coefficient space here, not ``br1``'s span Gram: the
    completion directions are orthonormal in field space by construction while the
    shipped twelve are not, so there is no single image-norm metric that is the right one
    for both.  This measures a FLOOR, and a floor does not depend on which descent metric
    reaches it -- only on whether the descent converges, which ``stop_reason`` reports.
    """
    import torch

    current = np.asarray(start_coefficients, dtype=np.float64).copy()
    best = float(evaluate_extended(inst, pair, current[None], basis_extended)[0])
    start_value = best
    evaluations = 1
    iterations = 0
    stop_reason = "outer_round_budget"
    history = [best]
    for _round in range(outer_rounds):
        jacobian, residual = extended_jacobian_at(
            inst, pair, current, basis_extended
        )
        middle = jacobian @ jacobian.T
        middle = middle + 1e-12 * torch.eye(6, dtype=middle.dtype)
        step = (jacobian.T @ torch.linalg.solve(middle, -residual)).numpy()
        block = np.stack([current + fraction * step for fraction in CONTINUOUS_LADDER])
        values = evaluate_extended(inst, pair, block, basis_extended)
        evaluations += len(block)
        winner = int(values.argmin())
        if values[winner] >= best:
            stop_reason = "no_improving_step"
            break
        best = float(values[winner])
        current = block[winner].copy()
        history.append(best)
        iterations += 1
        if best <= dd_threshold:
            stop_reason = "converged_below_materiality_floor"
            break
    return {
        "pair": int(pair),
        "rank": len(current),
        "start_d_pose": start_value,
        "final_d_pose": best,
        "ratio": best / start_value if start_value > 0 else 1.0,
        "evaluations": evaluations,
        "iterations": iterations,
        "history": history,
        "stop_reason": stop_reason,
    }


def cmd_reach(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    set_threads(args.threads)
    inst, meta = build_instrument(verify_raw=False)
    scales = np.asarray(inst.state.coefficient_scales, dtype=np.float64).reshape(-1)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    shipped_coefficients = codes.astype(np.float64) * scales[None]
    dd_threshold = materiality_dd_threshold(args.operating_point_d_pose)

    basis_extended = build_extended_basis(inst, args.extra_atoms)
    rank = int(basis_extended.shape[0])

    pairs = np.asarray(json.loads(Path(args.pairs_file).read_text()), dtype=np.int64)
    pairs = shard_of(pairs, args.shard_index, args.shard_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    controls = [
        reach_control(
            inst, int(pair), shipped_coefficients[int(pair)], basis_extended
        )
        for pair in pairs[: args.control_pairs]
    ]

    rows_path = args.out_dir / f"reach_rows_r{rank}_{args.shard_index}.jsonl"
    done = load_done(rows_path) if args.resume else {}
    started = time.time()
    completed = 0
    with rows_path.open("a") as stream:
        for position, pair in enumerate(pairs.tolist()):
            if pair in done:
                continue
            start = np.zeros(rank, dtype=np.float64)
            start[: up2.CARRIER_DIM] = shipped_coefficients[pair]
            row = reach_refine_pair(
                inst,
                pair,
                start,
                basis_extended,
                dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds,
            )
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            completed += 1
            if args.progress:
                elapsed = time.time() - started
                print(
                    f"  reach r{rank} shard {args.shard_index}: "
                    f"{position + 1}/{len(pairs)} pair={pair} "
                    f"{row['start_d_pose']:.3e} -> {row['final_d_pose']:.3e} "
                    f"({elapsed / max(completed, 1):.1f} s/pair)",
                    flush=True,
                )
    summary = {
        "schema": "ddm_pc3_reach_shard.v2",
        "rank": rank,
        "extra_atoms": int(args.extra_atoms),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": pairs.tolist(),
        "completed_this_run": completed,
        "rows_path": str(rows_path),
        "renderer_controls": controls,
        "dd_threshold": dd_threshold,
        "elapsed_seconds": time.time() - started,
        "instrument": meta,
        "what_this_bounds": (
            "what MORE atoms could buy on this body. A rank change is a receiver "
            "change (up2.CARRIER_DIM is receiver code) and cannot be sealed by this "
            "arm; the number decides whether it is worth a first-measurement contract, "
            "against a basis payload of ~1,023 B per atom measured on move 44."
        ),
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    (args.out_dir / f"REACH_SHARD_r{rank}_{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps({k: summary[k] for k in ("rank", "completed_this_run")}))
    return 0




# --------------------------------------------------------------------------------------
# mode=base -- the pose base on the POINTER's own configuration, on THIS instrument
# --------------------------------------------------------------------------------------


def cmd_base(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    set_threads(args.threads)
    started = time.time()
    inst, meta = build_instrument(verify_raw=args.verify_raw)
    codes = (
        np.load(args.codes).astype(np.int32)
        if args.codes
        else np.asarray(inst.state.codes, dtype=np.int32)
    )
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Pc3Error(f"codes have shape {codes.shape}, expected (600, 12)")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    indices = np.arange(N_PAIRS, dtype=np.int64)
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
    payload = {
        "schema": "ddm_pc3_pose_base.v1",
        "tag": args.tag,
        "pairs": int(N_PAIRS),
        "d_pose_mean": float(per_pair.mean()),
        "pose_leg": pose_leg(float(per_pair.mean())),
        "per_pair_path": str(args.out),
        "per_pair_sha256": _sha256_array(per_pair),
        "codes_source": str(args.codes) if args.codes else "move44_shipped_archive",
        "codes_sha256": _sha256_array(codes),
        "elapsed_seconds": time.time() - started,
        "instrument": meta,
        "t4_context_not_a_comparison": {
            "d_pose_t4": MOVE44_D_POSE_T4,
            "note": (
                "printed for context only; the local print and the T4 print are "
                "different instruments and this arm compares local to local"
            ),
        },
        "score_claim": False,
        "promotable": False,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: payload[k] for k in ("tag", "d_pose_mean", "pose_leg")}))
    return 0


# --------------------------------------------------------------------------------------
# mode=ceiling -- the continuous optimum inside the shipped span
# --------------------------------------------------------------------------------------


def load_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not rows_path.is_file():
        return done
    for line in rows_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[int(row["pair"])] = row
    return done


def cmd_ceiling(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    set_threads(args.threads)
    inst, meta = build_instrument(verify_raw=False)
    dd_threshold = materiality_dd_threshold(args.operating_point_d_pose)
    scales = np.asarray(inst.state.coefficient_scales, dtype=np.float64).reshape(-1)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    start = codes.astype(np.float64) * scales[None]

    pairs = np.arange(N_PAIRS, dtype=np.int64)
    if args.pairs_file:
        pairs = np.asarray(
            json.loads(Path(args.pairs_file).read_text()), dtype=np.int64
        )
    pairs = shard_of(pairs, args.shard_index, args.shard_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / f"ceiling_rows_{args.shard_index}.jsonl"
    done = load_done(rows_path) if args.resume else {}
    started = time.time()
    completed = 0
    with rows_path.open("a") as stream:
        for position, pair in enumerate(pairs.tolist()):
            if pair in done:
                continue
            row = continuous_refine_pair(
                inst,
                pair,
                start[pair],
                dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds,
                shipped_step=scales,
            )
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            completed += 1
            if args.progress:
                elapsed = time.time() - started
                print(
                    f"  shard {args.shard_index}: {position + 1}/{len(pairs)} "
                    f"pair={pair} {row['start_d_pose']:.3e} -> "
                    f"{row['final_d_pose']:.3e} ({elapsed / max(completed, 1):.1f} s/pair)",
                    flush=True,
                )
    summary = {
        "schema": "ddm_pc3_ceiling_shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": pairs.tolist(),
        "completed_this_run": completed,
        "rows_path": str(rows_path),
        "dd_threshold": dd_threshold,
        "operating_point_d_pose": args.operating_point_d_pose,
        "carrier_dim": int(up2.CARRIER_DIM),
        "elapsed_seconds": time.time() - started,
        "instrument": meta,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    (args.out_dir / f"CEILING_SHARD_{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps({k: summary[k] for k in ("shard_index", "completed_this_run")}))
    return 0


# --------------------------------------------------------------------------------------
# mode=project -- turn the ceiling into the REALIZED curve, one evaluation per rung
# --------------------------------------------------------------------------------------
#
# The ceiling is the lattice-free floor.  A rung that actually ships has to land on a
# lattice, so the realized number is what the continuous optimum becomes when it is
# ROUNDED onto that rung's lattice and re-scored through the real renderer.  That is one
# evaluation per pair per rung -- minutes, not hours -- and it brackets every rung from
# both sides: the ceiling above it, this projection below it.  (Below, because the
# projection of the continuous optimum need not be the BEST point of that lattice; only a
# full re-solve on the rung can close the bracket, and it is only worth paying for when
# the bracket still straddles break-even.)
#
# The ``shipped`` rung is not filler.  The shipped codes are a ``refine_pair`` fixed point
# under the +-1/+-2 SINGLE-coordinate polish (pc2 ITEM 1: 40 rounds moved 17 of 7,200
# coordinates), but the continuous optimum's rounding is a MULTI-coordinate move, so it
# can land somewhere the polish could not reach.  If it scores better, that is a pose gain
# at ZERO bytes and zero receiver change.


def lattice_menu(dimensions: int, halvings: tuple[int, ...]) -> list[tuple[str, np.ndarray]]:
    """The rungs: global refinements, then one refined dimension at a time.

    Per-DIMENSION rungs exist because the coefficient scales are twelve independent
    float32 words in the archive and CAP1 carries an independent Rice ``k`` per
    dimension, so a halving can be bought one dimension at a time.  That is a 12x finer
    granularity on the COST axis than a global halving, and on move 44 it is the only
    granularity whose cheapest step is small enough to be interesting.
    """
    menu: list[tuple[str, np.ndarray]] = [
        ("shipped", np.ones(dimensions, dtype=np.float64))
    ]
    for halving in halvings:
        menu.append(
            (f"global_div{2 ** halving}", np.full(dimensions, 0.5**halving))
        )
    for dim in range(dimensions):
        factors = np.ones(dimensions, dtype=np.float64)
        factors[dim] = 0.5
        menu.append((f"dim{dim}_div2", factors))
    return menu


def cmd_project(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    set_threads(args.threads)
    inst, meta = build_instrument(verify_raw=False)
    scales = np.asarray(inst.state.coefficient_scales, dtype=np.float64).reshape(-1)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    shipped_coefficients = codes.astype(np.float64) * scales[None]

    rows: dict[int, dict[str, Any]] = {}
    for path in args.rows:
        rows.update(load_done(Path(path)))
    pairs = sorted(rows)
    if not pairs:
        raise Pc3Error(f"no ceiling rows at {args.rows}")

    menu = lattice_menu(up2.CARRIER_DIM, tuple(args.halvings))
    labels = [label for label, _factors in menu]
    realized = np.zeros((N_PAIRS, len(menu)), dtype=np.float64)
    control = np.zeros(N_PAIRS, dtype=np.float64)
    max_abs_code = np.zeros((N_PAIRS, len(menu)), dtype=np.int64)
    started = time.time()

    for position, pair in enumerate(pairs):
        continuous = np.asarray(rows[pair]["coefficients"], dtype=np.float64)
        block = [shipped_coefficients[pair]]
        for _label, factors in menu:
            step = scales * factors
            integers = np.rint(continuous / step)
            max_abs_code[pair, len(block) - 1] = int(np.abs(integers).max())
            block.append(integers * step)
        values = evaluate_coefficients(inst, pair, np.stack(block))
        control[pair] = values[0]
        realized[pair] = values[1:]
        if args.progress and (position + 1) % 25 == 0:
            elapsed = time.time() - started
            print(
                f"  projected {position + 1}/{len(pairs)} "
                f"({elapsed / (position + 1):.2f} s/pair)",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "projected_per_pair.npy", realized)
    np.save(args.out_dir / "projection_control_per_pair.npy", control)
    base = np.load(args.base)
    summary = {
        "schema": "ddm_pc3_projection.v1",
        "pairs_measured": len(pairs),
        "labels": labels,
        "base_d_pose_mean": float(base.mean()),
        "control_d_pose_mean_on_measured_pairs": float(control[pairs].mean()),
        "base_d_pose_mean_on_measured_pairs": float(base[pairs].mean()),
        "rungs": [
            {
                "rung": label,
                "d_pose_mean_measured_pairs": float(realized[pairs, index].mean()),
                "gain_vs_control": float(
                    control[pairs].mean() - realized[pairs, index].mean()
                ),
                "max_abs_code": int(max_abs_code[pairs, index].max()),
                "int12_headroom_ok": bool(max_abs_code[pairs, index].max() <= 2047),
            }
            for index, label in enumerate(labels)
        ],
        "note": (
            "gain_vs_control differences INSIDE one batch shape; the control is the "
            "shipped coefficients evaluated in the same batch as the rungs, so the "
            "batch-shape offset cancels"
        ),
        "elapsed_seconds": time.time() - started,
        "instrument": meta,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
    }
    (args.out_dir / "PROJECTION.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps({k: summary[k] for k in ("pairs_measured", "rungs")}, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# mode=halfstep -- what a per-dimension lattice halving ACTUALLY buys, from the shipped point
# --------------------------------------------------------------------------------------
#
# ``mode=project`` rounds the CONTINUOUS optimum onto each rung's lattice, and on this body
# that estimator is useless in a way worth writing down: every projected rung, including
# one on a lattice 16x finer than the shipped one, scored an ORDER OF MAGNITUDE worse than
# the shipped codes.  The continuous optimum is not robust.  Two ``round`` calls sit inside
# the render, so the objective is rough at the scale of a lattice step, and the continuous
# solution sits on a knife edge that no lattice lands on.
#
# So the rung must be realised the way it would actually be built: as a REFINEMENT OF THE
# SHIPPED POINT, not of the continuous one.  A dimension-j halving lets that coordinate
# take half-integer values; the new points it reaches near the shipped fixed point are
# +-0.5 and +-1.5 steps (+-1, +-2 are the integer neighbours the shipped point is already a
# ``refine_pair`` fixed point against).  Evaluating those four is the rung's realised
# first-order gain, per dimension, per pair, on the real renderer and the real scorer.


HALF_STEP_OFFSETS = (-1.5, -0.5, 0.5, 1.5)


def cmd_halfstep(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    set_threads(args.threads)
    inst, meta = build_instrument(verify_raw=False)
    scales = np.asarray(inst.state.coefficient_scales, dtype=np.float64).reshape(-1)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    shipped = codes.astype(np.float64) * scales[None]
    dimensions = int(up2.CARRIER_DIM)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / f"halfstep_rows_{args.shard_index}.jsonl"
    pairs = shard_of(np.arange(N_PAIRS, dtype=np.int64), args.shard_index, args.shard_count)
    done = load_done(rows_path) if args.resume else {}
    started = time.time()
    completed = 0
    with rows_path.open("a") as stream:
        for position, pair in enumerate(pairs.tolist()):
            if pair in done:
                continue
            block = [shipped[pair]]
            labels = []
            for dim in range(dimensions):
                for offset in HALF_STEP_OFFSETS:
                    trial = shipped[pair].copy()
                    trial[dim] += offset * scales[dim]
                    block.append(trial)
                    labels.append((dim, offset))
            values = evaluate_coefficients(inst, pair, np.stack(block))
            control = float(values[0])
            best = {}
            for (dim, offset), value in zip(labels, values[1:], strict=True):
                if dim not in best or value < best[dim][0]:
                    best[dim] = (float(value), offset)
            row = {
                "pair": int(pair),
                "control_d_pose": control,
                "per_dimension_best": {
                    str(dim): {"d_pose": best[dim][0], "offset_in_steps": best[dim][1],
                               "gain": control - best[dim][0]}
                    for dim in range(dimensions)
                },
                "best_single_dimension_gain": max(
                    control - best[dim][0] for dim in range(dimensions)
                ),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            completed += 1
            if args.progress and completed % 10 == 0:
                elapsed = time.time() - started
                print(
                    f"  halfstep shard {args.shard_index}: {position + 1}/{len(pairs)} "
                    f"({elapsed / completed:.2f} s/pair)",
                    flush=True,
                )
    summary = {
        "schema": "ddm_pc3_halfstep_shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": pairs.tolist(),
        "completed_this_run": completed,
        "rows_path": str(rows_path),
        "offsets_in_lattice_steps": list(HALF_STEP_OFFSETS),
        "what_this_measures": (
            "the realised first-order gain of a per-dimension lattice halving, starting "
            "from the SHIPPED point rather than from the continuous optimum"
        ),
        "elapsed_seconds": time.time() - started,
        "instrument": meta,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
    }
    (args.out_dir / f"HALFSTEP_SHARD_{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps({k: summary[k] for k in ("shard_index", "completed_this_run")}))
    return 0


# --------------------------------------------------------------------------------------
# mode=report -- the curve
# --------------------------------------------------------------------------------------


def cmd_report(args) -> int:
    base = np.load(args.base)
    if base.shape != (N_PAIRS,):
        raise Pc3Error(f"base per-pair has shape {base.shape}")
    rows: dict[int, dict[str, Any]] = {}
    for path in args.rows:
        rows.update(load_done(Path(path)))
    missing = sorted(set(range(N_PAIRS)) - set(rows))
    #: The headline is the measured GAIN applied to the base, never the solver's absolute
    #: final.  The solver evaluates one pair replicated across a batch of 32 while the
    #: base leg evaluates 8 DIFFERENT pairs per batch, and batched float reduction is not
    #: associative, so the solver's ``start_d_pose`` and the base differ by a few times
    #: 1e-10 per pair (measured: max |delta| 3.4e-09, sum over 26 pairs -1.06e-08, versus
    #: a per-pair gain of 1.15e-07 -- 0.35% of the signal).  Differencing inside ONE
    #: instrument shape cancels that offset exactly; taking the absolute final would
    #: import it.  ``ceiling_absolute_form`` below carries the other convention so the
    #: two can be compared rather than assumed equal.
    gain = np.zeros(N_PAIRS, dtype=np.float64)
    absolute = base.copy()
    for pair, row in rows.items():
        gain[pair] = float(row["start_d_pose"]) - float(row["final_d_pose"])
        absolute[pair] = float(row["final_d_pose"])
    ceiling = base - gain

    base_mean = float(base.mean())
    ceiling_mean = float(ceiling.mean())
    gain_d_pose = base_mean - ceiling_mean
    gain_score = pose_leg(base_mean) - pose_leg(ceiling_mean)
    payable_bytes = gain_score / BYTE_TO_SCORE
    whole_term_bytes = pose_leg(base_mean) / BYTE_TO_SCORE

    stops: dict[str, int] = {}
    for row in rows.values():
        stops[row["stop_reason"]] = stops.get(row["stop_reason"], 0) + 1

    order = np.argsort(base)[::-1]
    top = [
        {
            "pair": int(p),
            "base": float(base[p]),
            "ceiling": float(ceiling[p]),
            "ratio": float(ceiling[p] / base[p]) if base[p] > 0 else 1.0,
        }
        for p in order[:20].tolist()
    ]

    #: The curve's own rungs, priced against the shipped archive. ``delta_bytes`` is what
    #: the rung would ADD; ``d_pose_needed`` is the d_pose it must reach just to break
    #: even, DERIVED from the byte cost and the exact score arithmetic, never assumed.
    rungs = []
    for label, delta_bytes in args.rung:
        needed_leg = pose_leg(base_mean) - float(delta_bytes) * BYTE_TO_SCORE
        needed_d_pose = (needed_leg**2) / 10.0 if needed_leg > 0 else float("-inf")
        rungs.append(
            {
                "rung": label,
                "delta_bytes": int(delta_bytes),
                "archive_bytes": MOVE44_ARCHIVE_BYTES + int(delta_bytes),
                "break_even_d_pose": needed_d_pose,
                "break_even_d_pose_reduction_fraction": (
                    (base_mean - needed_d_pose) / base_mean if base_mean > 0 else 0.0
                ),
                "ceiling_reaches_break_even": bool(ceiling_mean <= needed_d_pose),
                "best_possible_net_score_at_this_rung": (
                    composed_score(
                        MOVE44_D_SEG_T4,
                        ceiling_mean,
                        MOVE44_ARCHIVE_BYTES + int(delta_bytes),
                    )
                    - composed_score(
                        MOVE44_D_SEG_T4, base_mean, MOVE44_ARCHIVE_BYTES
                    )
                ),
            }
        )

    payload = {
        "schema": "ddm_pc3_ceiling_report.v1",
        "pairs_measured": len(rows),
        "rungs_priced_against_the_ceiling": rungs,
        "pointer_row_for_the_seg_and_byte_legs": {
            "d_seg_t4": MOVE44_D_SEG_T4,
            "archive_bytes": MOVE44_ARCHIVE_BYTES,
            "score_t4": MOVE44_SCORE_T4,
            "note": (
                "the seg leg and the byte leg are the pointer's; only the pose leg "
                "moves in this arm, and it is measured locally on the pointer's own "
                "configuration, so every delta here is local-to-local"
            ),
        },
        "pairs_missing": missing,
        "base_d_pose_mean": base_mean,
        "ceiling_d_pose_mean": ceiling_mean,
        "base_pose_leg": pose_leg(base_mean),
        "ceiling_pose_leg": pose_leg(ceiling_mean),
        "gain_d_pose": gain_d_pose,
        "gain_score": gain_score,
        "gain_fraction_of_base": gain_d_pose / base_mean if base_mean > 0 else 0.0,
        "payable_bytes_for_the_whole_lattice_family": payable_bytes,
        "payable_bytes_if_d_pose_were_zero": whole_term_bytes,
        "byte_to_score": BYTE_TO_SCORE,
        "admit_bar": ADMIT_BAR,
        "stop_reasons": stops,
        "ceiling_absolute_form": {
            "d_pose_mean": float(absolute.mean()),
            "delta_vs_gain_form": float(absolute.mean() - ceiling_mean),
            "why_it_differs": (
                "batched float reduction is not associative and the two legs use "
                "different batch shapes; the gain form cancels the offset, this one "
                "carries it, and the size of the difference is the size of the offset"
            ),
        },
        "worst_20_by_base": top,
        "pairs_improved": int((ceiling < base - 0.0).sum()),
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]",
        "score_claim": False,
        "promotable": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    np.save(args.out.with_suffix(".perpair.npy"), ceiling)
    print(json.dumps(payload, indent=2, sort_keys=True)[:4000])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    base = sub.add_parser("base", help="pose base on move 44's own configuration")
    base.add_argument("--tag", default="base")
    base.add_argument("--codes", type=Path, default=None)
    base.add_argument("--out", type=Path, required=True)
    base.add_argument("--json-out", type=Path, required=True)
    base.add_argument("--threads", type=int, default=6)
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--verify-raw", action="store_true")
    base.set_defaults(func=cmd_base)

    ceiling = sub.add_parser(
        "ceiling", help="continuous optimum inside the shipped 12-dim span"
    )
    ceiling.add_argument("--out-dir", type=Path, required=True)
    ceiling.add_argument("--shard-index", type=int, default=0)
    ceiling.add_argument("--shard-count", type=int, default=1)
    ceiling.add_argument("--threads", type=int, default=2)
    ceiling.add_argument("--outer-rounds", type=int, default=24)
    ceiling.add_argument("--operating-point-d-pose", type=float, required=True)
    ceiling.add_argument("--pairs-file", type=Path, default=None)
    ceiling.add_argument("--resume", action="store_true")
    ceiling.add_argument("--progress", action="store_true")
    ceiling.set_defaults(func=cmd_ceiling)

    reach = sub.add_parser(
        "reach", help="unconstrained 3x24x32 field floor: the bound on EVERY rank"
    )
    reach.add_argument("--out-dir", type=Path, required=True)
    reach.add_argument("--pairs-file", type=Path, required=True)
    reach.add_argument("--shard-index", type=int, default=0)
    reach.add_argument("--shard-count", type=int, default=1)
    reach.add_argument("--threads", type=int, default=2)
    reach.add_argument("--outer-rounds", type=int, default=16)
    reach.add_argument("--extra-atoms", type=int, required=True)
    reach.add_argument("--control-pairs", type=int, default=2)
    reach.add_argument("--operating-point-d-pose", type=float, required=True)
    reach.add_argument("--resume", action="store_true")
    reach.add_argument("--progress", action="store_true")
    reach.set_defaults(func=cmd_reach)

    project = sub.add_parser(
        "project", help="realized d_pose of the ceiling rounded onto each rung's lattice"
    )
    project.add_argument("--rows", nargs="+", required=True)
    project.add_argument("--base", type=Path, required=True)
    project.add_argument("--out-dir", type=Path, required=True)
    project.add_argument("--halvings", type=int, nargs="+", default=[1, 2, 3, 4])
    project.add_argument("--threads", type=int, default=6)
    project.add_argument("--progress", action="store_true")
    project.set_defaults(func=cmd_project)

    half = sub.add_parser(
        "halfstep", help="realised per-dimension halving gain, from the SHIPPED point"
    )
    half.add_argument("--out-dir", type=Path, required=True)
    half.add_argument("--shard-index", type=int, default=0)
    half.add_argument("--shard-count", type=int, default=1)
    half.add_argument("--threads", type=int, default=2)
    half.add_argument("--resume", action="store_true")
    half.add_argument("--progress", action="store_true")
    half.set_defaults(func=cmd_halfstep)

    report = sub.add_parser("report", help="assemble the ceiling into the curve")
    report.add_argument("--base", type=Path, required=True)
    report.add_argument("--rows", nargs="+", required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument(
        "--rung",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "DELTA_BYTES"),
        help="a capacity rung to price against the ceiling, e.g. --rung step_div2 900",
    )
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
