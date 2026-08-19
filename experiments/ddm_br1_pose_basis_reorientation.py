"""ddm_br1 -- pose carrier basis re-orientation, and what actually binds instead.

WHAT THIS ARM WAS ASKED TO DO, AND WHY THAT MOVE IS NULL
-------------------------------------------------------
The charter asked for a *re-orientation of the existing 12 basis dimensions*: a
rotation costs ~0 storage (same coefficient count), so if it bought pose it would
be nearly free.  It cannot buy pose, and the reason is geometric rather than
empirical.

The shipped carrier renders frame 0 as ``einsum(coeff, basis) / sqrt(12)`` from a
basis of 12 fields living at 24x32x3 (``cpr1/inflate.py:335-352``).  Those 12
fields span a 12-dimensional subspace ``S`` of the 2304-dimensional space of
24x32x3 band-limited fields.  Re-mixing the 12 vectors by any invertible 12x12
matrix produces a different *basis* for the very same subspace ``S``.

The quantity that decides whether the carrier can reach its own pose residual is
the smallest IMAGE perturbation inside ``S`` that cancels the residual:

    minimise ||dX||  subject to  J dX = -r,  dX in S

That is a property of ``S`` alone -- no basis appears in the statement -- so it is
invariant under every re-mixing.  ``mode=geometry`` measures the invariance rather
than asserting it, and finds it holds to ~1e-9 (machine precision).

A REQUIRED INSTRUMENT CORRECTION
--------------------------------
The prior probe (``ddm_up2``'s ``basis_conditioning_probe.py``) computed both the
12-dof and 2304-dof steps with ``torch.linalg.lstsq``.  Both systems are
UNDERDETERMINED (6 equations, 12 or 2304 unknowns), and ``lstsq`` returns *a*
solution, not the minimum-norm one.  Minimum COEFFICIENT-norm is basis-dependent;
minimum IMAGE-norm is the basis-free quantity.  Measured here, the lstsq method
reports a penalty ~11% larger than the true minimum-image-norm penalty.  This
does not overturn up2's direction -- the span penalty is real and large -- but the
number it produced is not the span property it was read as.

WHAT ACTUALLY BINDS (the premise this arm had to re-derive)
----------------------------------------------------------
``ddm_up2`` reported its n600 solve as CONVERGED.  Read at the source, that
convergence is scoped: ``solve_pair_realized`` proposes SINGLE-coordinate moves of
``offsets=(-2,-1,1,2)`` and stops when no such neighbour improves
(``ddm_up2_shipping_pose_solve.py:694-782``).  It is a coordinate-wise +-2 local
optimum, not a lattice optimum.  Measured here, the step the residual actually
demands is 57 to 14,079 int12 code units, multi-coordinate, against a per-pair
range headroom of roughly 700-1500.  For a substantial share of pairs the demanded
step FITS inside the lattice and the +-2 search simply never travelled that far.

So "the remaining wall is the 12-dim basis" was not established: the search was
never excluded as the binding constraint.  ``mode=gn`` excludes it, by taking the
damped Gauss-Newton step directly and realising it on the lattice.  Only what
survives that control can be attributed to the basis.

AUTHORITY
---------
Frozen CPU-torch PoseNet, GT decoded by DALI -- the lineage ``upstream/evaluate.py``
uses on the contest-CUDA axis that the pointer row was measured on
(``evaluate.py:31-42``, assert-enforced at ``frame_utils.py:113/:188``).  The pose
lineage gap is ADDITIVE, not a transferable factor
([[gt_lineage_pose_gap_is_additive_not_multiplicative_20260819]]), so this module
scores directly on the DALI table and never rescales an advisory number.
``score_claim=false``, ``promotable=false``: only ``upstream/evaluate.py`` on
contest hardware is a score.

d_seg is invariant BY CONSTRUCTION: SegNet reads only the odd frame
(``upstream/modules.py:108``) and this module writes only even (carrier) frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import ddm_up2_shipping_pose_solve as up2

EVAL_H, EVAL_W = 384, 512
K = up2.CARRIER_DIM
AMP = up2.CARRIER_AMPLITUDE
CODE_MIN, CODE_MAX = up2.COEFF_CODE_MIN, up2.COEFF_CODE_MAX

#: up2's banked n600 result on the same instrument -- the incumbent to beat.
UP2_D_POSE_BEFORE = 7.76948388629175e-06
UP2_D_POSE_AFTER = 7.649246787072966e-06
UP2_SOLVED_CODES = Path("/Volumes/APDataStore/pact/ddm_up2/retained/solved_codes.npy")

#: THE LIVE POINTER BODY. ``ddm_up3`` landed the thirteenth move on 2026-08-19, so
#: the body ``ddm_up2`` solved against (to1, 50e56145...) is SUPERSEDED. Every br1
#: delta is measured against this body; measuring against to1 would re-count up2's
#: already-banked gain.
LIVE_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_up3/candidate_runtime")
LIVE_ARCHIVE_SHA256 = (
    "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"
)
LIVE_POINTER_SCORE = 0.15652626435208142
#: The live body's population d_pose on the DALI instrument = up2's solved value.
LIVE_D_POSE = UP2_D_POSE_AFTER

#: Admission bar for a net score move, in score units (negative is better).
ADMIT_BAR = -3.5e-06


class Br1Error(RuntimeError):
    """A ddm_br1 precondition failed. Fail closed, never approximate."""


# ---------------------------------------------------------------------------
# Geometry: the span, expressed on its own 24x32 lattice.
# ---------------------------------------------------------------------------


def low_basis(state):
    """The 12 shipped basis vectors at their native 24x32 resolution.

    ``normalized_basis`` interpolates 24x32 -> 384x512, subtracts the per-dim mean
    and divides by the per-dim RMS.  Interpolation is linear and carries a constant
    field to a constant field, so the identical vector at 24x32 is
    ``(raw - mean) / rms``; ``mode=geometry`` checks that interpolating it back
    reproduces ``basis_norm``, rather than assuming the commutation.
    """
    from torch.nn import functional

    raw = state.basis_raw.double()
    up = functional.interpolate(
        raw, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    )
    # The receiver computes the RMS on the CENTRED basis (normalized_basis in
    # cpr1/inflate.py), so this must too or the reconstruction control drifts.
    mean = up.mean(dim=(1, 2, 3), keepdim=True)
    rms = (up - mean).square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    return (raw - mean) / rms


def span_gram(blow):
    """Gram matrix of the span generators, in field (image) inner product."""
    bmat = blow.reshape(K, -1) / math.sqrt(K)
    return bmat @ bmat.T, bmat


def min_image_norm_step(jac, residual, gram):
    """Smallest IMAGE-norm coefficient step that cancels the residual to 1st order.

    Solves   min dc^T G dc   s.t.  J dc = -r
    whose stationary point is  dc = G^-1 J^T (J G^-1 J^T)^-1 (-r).
    G is the span's Gram matrix, so ``dc^T G dc == ||dX||^2`` -- minimising it
    minimises the IMAGE perturbation, the basis-free quantity, not the
    basis-dependent coefficient norm that ``lstsq`` happens to return.
    """
    import torch

    ginv = torch.linalg.inv(gram)
    middle = jac @ ginv @ jac.T
    return ginv @ jac.T @ torch.linalg.solve(middle, -residual)


def realize(codes_float):
    """Project a float code vector onto the shipped signed-int12 lattice."""
    return np.clip(np.rint(codes_float), CODE_MIN, CODE_MAX).astype(np.int32)


# ---------------------------------------------------------------------------
# Sampling -- seeded random, never a prefix.
# ---------------------------------------------------------------------------


def sample_pairs(count: int, seed: int) -> np.ndarray:
    """Seeded random pairs, or the full field.

    A contiguous prefix of this video is a DIFFERENT population and the bias is
    worst on exactly this axis: pose prefixes measure 2.54-4.21x HARDER than the
    population (``ddm_na2`` / ``ddm_bp2``).  A prefix-based pose number is the
    canonical false-negative shape, so every sub-n600 run samples.
    """
    if count >= up2.N_PAIRS_TOTAL:
        return np.arange(up2.N_PAIRS_TOTAL, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(up2.N_PAIRS_TOTAL, size=count, replace=False)
    ).astype(np.int64)


@dataclass
class Instrument:
    """Everything a measurement needs, loaded once."""

    state: Any
    raw: Any
    targets: np.ndarray
    posenet: Any
    blow: Any
    gram: Any
    bmat: Any


def load_instrument(
    *,
    runtime: Path | None = None,
    expect_archive_sha256: str | None = None,
    verify_sha: bool = False,
):
    """Load the carrier from a named runtime, pinned to an expected archive sha.

    The runtime is an ARGUMENT because the pointer body moves: ``ddm_up3`` landed
    the thirteenth move and the live body is no longer the one ``ddm_up2`` solved
    against.  Measuring a delta against a superseded body double-counts a gain that
    is already banked, so the body is named and its sha256 is checked, never
    assumed ([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]).

    Only ODD (frame-1, semantic) frames are read from the raw decode, and carrier
    edits do not touch odd frames, so a decode of a body that differs only in its
    carrier coefficients remains valid for frame 1.
    """
    runtime = Path(runtime) if runtime is not None else Path(up2.DEFAULT_RUNTIME)
    if expect_archive_sha256:
        observed = up2._sha256_file(runtime / "archive.zip")
        if observed != expect_archive_sha256:
            raise Br1Error(
                f"runtime archive sha256 {observed} != expected {expect_archive_sha256}; "
                "refusing to measure a delta against an unidentified body"
            )
    state = up2.load_carrier_state(
        runtime, verify_archive=expect_archive_sha256 is None
    )
    raw = up2.open_raw(up2.DEFAULT_RAW, verify_sha=verify_sha)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise Br1Error(
            f"GT lineage is {lineage}, not {up2.LINEAGE_DALI}: this would solve the "
            "contest-CPU objective, which is a different object (evaluate.py:31-42)"
        )
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = low_basis(state)
    gram, bmat = span_gram(blow)
    return Instrument(state, raw, targets, posenet, blow, gram, bmat)


def evaluate_codes(inst: Instrument, pair: int, code_block: np.ndarray) -> np.ndarray:
    """Realized d_pose for a block of candidate code rows, exact receiver path."""
    import torch

    if len(code_block) == 0:
        return np.zeros(0, dtype=np.float64)
    index = np.array([pair], dtype=np.int64)
    frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
    target_row = inst.targets[pair]
    out = np.zeros(len(code_block), dtype=np.float64)
    batch = 32
    for start in range(0, len(code_block), batch):
        chunk = code_block[start : start + batch]
        coefficients = up2.codes_to_coefficients(chunk, inst.state.coefficient_scales)
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


# ---------------------------------------------------------------------------
# mode=gn -- damped Gauss-Newton on the lattice, then the +-2 polish.
# ---------------------------------------------------------------------------

#: Geometric ladder of step fractions.  The GN step is a FIRST-ORDER prediction and
#: the demanded steps are large, so the full step is usually not the best one; the
#: ladder is searched by REALIZED evaluation, never by the linear model.
STEP_LADDER = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03, 0.015)


def gn_solve_pair(
    inst: Instrument,
    pair: int,
    start_codes: np.ndarray,
    *,
    iterations: int = 6,
    polish: bool = True,
) -> dict[str, Any]:
    """Damped Gauss-Newton descent for one pair, realized on the int12 lattice.

    Each iteration re-linearises at the CURRENT lattice point, forms the minimum
    IMAGE-norm step that would cancel the residual, then picks the step fraction by
    REALIZED evaluation.  Accepting only realized improvements means the linear
    model can be wrong without corrupting the result -- it proposes, the receiver
    disposes.
    """
    import torch

    index = np.array([pair], dtype=np.int64)
    frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
    target_row = inst.targets[pair]
    tb = torch.from_numpy(target_row[None]).float()
    scales = inst.state.coefficient_scales.double().numpy()

    current = start_codes.astype(np.int32).copy()
    best = float(evaluate_codes(inst, pair, current[None])[0])
    start_value = best
    history = [best]
    evaluations = 1
    demanded_max = 0.0

    for _ in range(iterations):
        coeff = up2.codes_to_coefficients(current[None], inst.state.coefficient_scales)
        jac, res, _ = up2.jacobian_and_residual(
            inst.posenet, inst.state, coeff, frame1, tb, index
        )
        step = min_image_norm_step(
            jac[0].double(), res[0].double(), inst.gram
        ).numpy()
        demanded_max = max(demanded_max, float(np.abs(step / scales).max()))
        block = []
        for fraction in STEP_LADDER:
            trial = realize(current + fraction * step / scales)
            if not np.array_equal(trial, current):
                block.append(trial)
        if not block:
            break
        block_arr = np.stack(block)
        values = evaluate_codes(inst, pair, block_arr)
        evaluations += len(block_arr)
        winner = int(values.argmin())
        if values[winner] >= best:
            break
        best = float(values[winner])
        current = block_arr[winner].copy()
        history.append(best)

    gn_value = best
    if polish:
        while True:
            block, _labels = up2.candidate_codes_for_pair(current, (-2, -1, 1, 2))
            if len(block) == 0:
                break
            values = evaluate_codes(inst, pair, block)
            evaluations += len(block)
            winner = int(values.argmin())
            if values[winner] >= best:
                break
            best = float(values[winner])
            current = block[winner].copy()
            history.append(best)

    return {
        "pair": int(pair),
        "start_d_pose": start_value,
        "gn_d_pose": gn_value,
        "final_d_pose": best,
        "ratio": best / start_value if start_value > 0 else 1.0,
        "gn_ratio": gn_value / start_value if start_value > 0 else 1.0,
        "evaluations": evaluations,
        "demanded_code_units_max": demanded_max,
        "codes": current.astype(np.int32).tolist(),
        "changed_coordinates": int((current != start_codes).sum()),
        "max_abs_code": int(np.abs(current).max()),
        "history": history,
    }


def load_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    """Resume support: every completed pair is already on disk."""
    done: dict[int, dict[str, Any]] = {}
    if not rows_path.is_file():
        return done
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pair" in row:
                done[int(row["pair"])] = row
    return done


def run_gn(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    inst = load_instrument(
        runtime=Path(args.runtime), expect_archive_sha256=args.expect_archive_sha256
    )
    pairs = sample_pairs(args.pairs, args.seed)

    if args.start_from == "up2":
        if not UP2_SOLVED_CODES.is_file():
            raise Br1Error(f"up2 solved codes missing: {UP2_SOLVED_CODES}")
        start_all = np.load(UP2_SOLVED_CODES).astype(np.int32)
    else:
        start_all = inst.state.codes.astype(np.int32)
    if start_all.shape != (up2.N_PAIRS_TOTAL, K):
        raise Br1Error(f"start codes have shape {start_all.shape}")

    done = load_done(rows_path)
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            row = gn_solve_pair(
                inst,
                int(pair),
                start_all[int(pair)],
                iterations=args.iterations,
                polish=not args.no_polish,
            )
            row["start_from"] = args.start_from
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[int(pair)] = row
            print(
                f"[{position + 1}/{len(pairs)}] pair {pair}: "
                f"{row['start_d_pose']:.6e} -> {row['final_d_pose']:.6e} "
                f"(gn {row['gn_ratio']:.4f}, final {row['ratio']:.4f}) "
                f"demanded {row['demanded_code_units_max']:.0f} "
                f"[{time.time() - started:.0f}s]",
                flush=True,
            )

    rows = [done[int(p)] for p in pairs if int(p) in done]
    before = float(np.mean([r["start_d_pose"] for r in rows]))
    after = float(np.mean([r["final_d_pose"] for r in rows]))
    gn_only = float(np.mean([r["gn_d_pose"] for r in rows]))
    summary = {
        "schema": "ddm_br1_gn_summary.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "gt_lineage": up2.LINEAGE_DALI,
        "start_from": args.start_from,
        "pairs": len(rows),
        "seed": args.seed,
        "sampled_not_prefix": args.pairs < up2.N_PAIRS_TOTAL,
        "d_pose_start_mean": before,
        "d_pose_after_gn_mean": gn_only,
        "d_pose_final_mean": after,
        "ratio_gn": gn_only / before if before else 1.0,
        "ratio_final": after / before if before else 1.0,
        "pairs_improved": int(sum(1 for r in rows if r["final_d_pose"] < r["start_d_pose"])),
        "pairs_worsened": int(sum(1 for r in rows if r["final_d_pose"] > r["start_d_pose"])),
        "changed_coordinates_total": int(sum(r["changed_coordinates"] for r in rows)),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# mode=geometry -- the invariance control and the true span penalty.
# ---------------------------------------------------------------------------


def run_geometry(args) -> int:
    import torch
    from torch.nn import functional

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst = load_instrument(
        runtime=Path(args.runtime), expect_archive_sha256=args.expect_archive_sha256
    )

    recon = functional.interpolate(
        inst.blow.float(), size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    )
    commute_err = float((recon - inst.state.basis_norm).abs().max())

    pairs = sample_pairs(args.pairs, args.seed)
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    rmat = torch.randn(K, K, generator=generator, dtype=torch.float64)

    # The re-mixed basis, renormalised exactly as the receiver would.
    blow_r = torch.einsum("jk,kchw->jchw", rmat, inst.blow)
    upr = functional.interpolate(
        blow_r.float(), size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
    ).double()
    mean_r = upr.mean(dim=(1, 2, 3), keepdim=True)
    rms_r = (upr - mean_r).square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    blow_r = (blow_r - mean_r) / rms_r
    qmat, _ = torch.linalg.qr(inst.bmat.T)
    qmat_r, _ = torch.linalg.qr((blow_r.reshape(K, -1) / math.sqrt(K)).T)

    batch = 8
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        frame1 = up2.frames_to_bchw(inst.raw[2 * chunk + 1])
        tb = torch.from_numpy(inst.targets[chunk]).float()
        coeff = inst.state.coefficients[chunk].clone()

        x0 = (
            torch.einsum("bk,kchw->bchw", coeff.double(), inst.blow) / math.sqrt(K)
        ).float().clone().detach().requires_grad_(True)
        carrier = functional.interpolate(
            x0, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
        )
        low = up2._round_ste((127.5 + AMP * carrier).clamp(0.0, 255.0))
        slave = functional.interpolate(
            low, size=(up2.CAMERA_H, up2.CAMERA_W), mode="bicubic", align_corners=False
        )
        frame0 = up2._round_ste(slave.clamp(0.0, 255.0))
        frame0 = up2.apply_selector_float(
            frame0, inst.state.selector_modes, inst.state.selector_choices[chunk]
        )
        pose = up2.pose_from_frames(inst.posenet, frame0, frame1)
        grads = []
        for component in range(up2.POSE_DIMS):
            grad = torch.autograd.grad(
                pose[:, component].sum(),
                x0,
                retain_graph=component < up2.POSE_DIMS - 1,
            )[0]
            grads.append(grad.reshape(len(chunk), -1))
        j2304 = torch.stack(grads, dim=1).double()
        residual = (pose.detach().double() - tb.double())

        for offset, pair in enumerate(chunk):
            res = residual[offset]
            jq = j2304[offset] @ qmat
            step_span = qmat @ (torch.linalg.pinv(jq) @ -res)
            step_full = torch.linalg.pinv(j2304[offset]) @ -res
            jq_r = j2304[offset] @ qmat_r
            step_span_r = qmat_r @ (torch.linalg.pinv(jq_r) @ -res)
            rot_delta = float(
                (step_span_r - step_span).norm() / step_span.norm().clamp_min(1e-30)
            )
            lstsq_step = torch.linalg.lstsq(
                (j2304[offset] @ inst.bmat.T), -res.unsqueeze(-1)
            ).solution.squeeze(-1)
            lstsq_img = (lstsq_step @ inst.bmat).norm()
            rows.append(
                {
                    "pair": int(pair),
                    "residual_norm": float(res.norm()),
                    "min_image_norm_span": float(step_span.norm()),
                    "min_image_norm_full": float(step_full.norm()),
                    "lstsq_image_norm_span": float(lstsq_img),
                    "penalty_true": float(step_span.norm() / step_full.norm()),
                    "penalty_lstsq_method": float(lstsq_img / step_full.norm()),
                    "rotation_relative_change": rot_delta,
                }
            )
        print(f"  geometry {min(start + batch, len(pairs))}/{len(pairs)}", flush=True)

    pen_true = np.array([r["penalty_true"] for r in rows])
    pen_lstsq = np.array([r["penalty_lstsq_method"] for r in rows])
    rot = np.array([r["rotation_relative_change"] for r in rows])
    summary = {
        "schema": "ddm_br1_geometry.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "pairs": len(rows),
        "seed": args.seed,
        "sampled_not_prefix": args.pairs < up2.N_PAIRS_TOTAL,
        "control_interp_lowbasis_reproduces_basis_norm_max_abs": commute_err,
        "rotation_invariance_relative_change_max": float(rot.max()),
        "rotation_invariance_relative_change_median": float(np.median(rot)),
        "span_penalty_true_median": float(np.median(pen_true)),
        "span_penalty_lstsq_method_median": float(np.median(pen_lstsq)),
        "lstsq_overstatement_median": float(np.median(pen_lstsq / pen_true)),
        "rows": rows,
    }
    (out_dir / "GEOMETRY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# mode=ceiling -- how much pose is available to ANY band-limited frame 0?
# ---------------------------------------------------------------------------


def ceiling_pair(
    inst: Instrument,
    pair: int,
    *,
    iterations: int = 40,
) -> dict[str, Any]:
    """Best realized d_pose for an UNCONSTRAINED 24x32x3 frame-0 field.

    This prices the ceiling before anything is attributed to the 12-dim basis.
    The field is free (2304 dof, no basis, no int12 lattice) but still rendered
    through the exact receiver path -- both rounds, the clamp, the selector -- and
    scored by the frozen PoseNet on DALI GT.  Only the ROUNDS use a straight-through
    gradient; every reported value is a realized non-STE measurement.

    If this ceiling sits at the shipped d_pose, no basis and no search can pay, and
    the pose-carrier family is closed for a reason that has nothing to do with the
    basis.  If it sits far below, the 12-dim span is genuinely leaving pose on the
    table and re-choosing it is worth its bytes.
    """
    import torch
    from torch.nn import functional

    index = np.array([pair], dtype=np.int64)
    frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
    target = torch.from_numpy(inst.targets[pair][None]).double()

    def realized(field):
        """Exact receiver path, no STE, no grad -- the reported number."""
        with torch.inference_mode():
            carrier = functional.interpolate(
                field, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
            )
            low = torch.round((127.5 + AMP * carrier).clamp(0.0, 255.0))
            slave = functional.interpolate(
                low, size=(up2.CAMERA_H, up2.CAMERA_W), mode="bicubic",
                align_corners=False,
            )
            frame0 = torch.round(slave.clamp(0.0, 255.0))
            frame0 = up2.apply_selector_float(
                frame0, inst.state.selector_modes, inst.state.selector_choices[index]
            )
            pose = up2.pose_from_frames(inst.posenet, frame0, frame1)
        return float(((pose.double() - target) ** 2).mean())

    x = (
        torch.einsum(
            "bk,kchw->bchw",
            inst.state.coefficients[index].double(),
            inst.blow,
        )
        / math.sqrt(K)
    ).float()
    best_field = x.clone()
    best = realized(best_field)
    start_value = best

    for _ in range(iterations):
        leaf = best_field.clone().detach().requires_grad_(True)
        carrier = functional.interpolate(
            leaf, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
        )
        low = up2._round_ste((127.5 + AMP * carrier).clamp(0.0, 255.0))
        slave = functional.interpolate(
            low, size=(up2.CAMERA_H, up2.CAMERA_W), mode="bicubic", align_corners=False
        )
        frame0 = up2._round_ste(slave.clamp(0.0, 255.0))
        frame0 = up2.apply_selector_float(
            frame0, inst.state.selector_modes, inst.state.selector_choices[index]
        )
        pose = up2.pose_from_frames(inst.posenet, frame0, frame1)
        rows = []
        for component in range(up2.POSE_DIMS):
            grad = torch.autograd.grad(
                pose[:, component].sum(),
                leaf,
                retain_graph=component < up2.POSE_DIMS - 1,
            )[0]
            rows.append(grad.reshape(-1))
        jac = torch.stack(rows).double()
        res = (pose.detach().double() - target).reshape(-1)
        step = (torch.linalg.pinv(jac) @ -res).reshape(best_field.shape)

        improved = False
        for fraction in STEP_LADDER:
            trial = (best_field.double() + fraction * step).float()
            value = realized(trial)
            if value < best:
                best = value
                best_field = trial
                improved = True
                break
        if not improved:
            break

    return {
        "pair": int(pair),
        "start_d_pose": start_value,
        "ceiling_d_pose": best,
        "ratio": best / start_value if start_value > 0 else 1.0,
    }


def run_ceiling(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    inst = load_instrument(
        runtime=Path(args.runtime), expect_archive_sha256=args.expect_archive_sha256
    )
    pairs = sample_pairs(args.pairs, args.seed)
    done = load_done(rows_path)
    started = time.time()

    with rows_path.open("a", encoding="utf-8") as handle:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            row = ceiling_pair(inst, int(pair), iterations=args.iterations)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[int(pair)] = row
            print(
                f"[{position + 1}/{len(pairs)}] pair {pair}: "
                f"{row['start_d_pose']:.6e} -> {row['ceiling_d_pose']:.6e} "
                f"(ratio {row['ratio']:.4f}) [{time.time() - started:.0f}s]",
                flush=True,
            )

    rows = [done[int(p)] for p in pairs if int(p) in done]
    before = float(np.mean([r["start_d_pose"] for r in rows]))
    after = float(np.mean([r["ceiling_d_pose"] for r in rows]))
    summary = {
        "schema": "ddm_br1_ceiling.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "what_this_bounds": (
            "best realized d_pose for an UNCONSTRAINED 24x32x3 frame-0 field: no "
            "12-dim basis, no int12 lattice. Any basis or search result must lie "
            "at or above this."
        ),
        "pairs": len(rows),
        "seed": args.seed,
        "sampled_not_prefix": args.pairs < up2.N_PAIRS_TOTAL,
        "d_pose_start_mean": before,
        "d_pose_ceiling_mean": after,
        "ratio": after / before if before else 1.0,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# mode=price -- what the solved coefficients cost, and which pairs earn it.
# ---------------------------------------------------------------------------

#: One archive byte in score units (upstream/evaluate.py:63 rate term).
BYTE_TO_SCORE = 25.0 / 37_545_489.0


@dataclass
class Pricer:
    """The shipped carrier's Rice model, loaded once and reused."""

    module: Any
    carrier_repack: Any
    predictor: Any
    model: Any
    base_codes: np.ndarray
    ks_shipped: np.ndarray
    bits_shipped: int


def load_pricer(*, archive: Path, runtime: Path) -> Pricer:
    """Load the shipped Rice model and prove it reproduces the shipped payload.

    The control is not optional: a pricer that cannot re-encode the bytes already
    in the container cannot be trusted to price a candidate against them.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import ddm_t1h_carrier_byte_pricer as pricer

    (carrier_repack, _cap1, predictor, _blob, _sel, info, model, base_codes, _comp) = (
        pricer.load_shipped(archive, runtime)
    )
    ks_ship, bits_ship = pricer.rice_bits(base_codes, model, carrier_repack, predictor)
    if bits_ship != int(info["rice_payload_bits"]):
        raise Br1Error(
            "CONTROL FAILED: the pricer does not reproduce the shipped Rice bit "
            f"count ({bits_ship} != {info['rice_payload_bits']}); refusing to price"
        )
    return Pricer(
        pricer, carrier_repack, predictor, model, base_codes, ks_ship, int(bits_ship)
    )


def price_codes(candidate: np.ndarray, pricer: Pricer) -> dict[str, Any]:
    """Rice-payload cost of a full (600,12) candidate lattice, against shipped.

    The coefficients are stored as zigzag deltas ALONG THE PAIR AXIS and Rice
    coded per dimension (``cpr1/inflate.py:239-247``), so changing one pair
    perturbs two deltas and the per-dimension Rice parameter k is global.  Larger
    steps therefore are NOT free the way ``ddm_up2``'s +-2 steps were: this prices
    them instead of assuming.
    """
    base_codes = pricer.base_codes
    ks_ship, bits_ship = pricer.ks_shipped, pricer.bits_shipped
    ks_cand, bits_cand = pricer.module.rice_bits(
        candidate, pricer.model, pricer.carrier_repack, pricer.predictor
    )
    bytes_ship = (bits_ship + 7) // 8
    bytes_cand = (bits_cand + 7) // 8
    return {
        "control_reproduces_shipped_payload": True,
        "rice_bits_shipped": int(bits_ship),
        "rice_bits_candidate": int(bits_cand),
        "rice_payload_bytes_shipped": int(bytes_ship),
        "rice_payload_bytes_candidate": int(bytes_cand),
        "delta_bytes": int(bytes_cand - bytes_ship),
        "rice_ks_shipped": ks_ship.astype(int).tolist(),
        "rice_ks_candidate": ks_cand.astype(int).tolist(),
        "rice_ks_changed": bool(not np.array_equal(ks_ship, ks_cand)),
        "changed_pairs": int((candidate != base_codes).any(axis=1).sum()),
        "changed_coordinates": int((candidate != base_codes).sum()),
    }


def run_price(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_done(Path(args.rows))
    if not rows:
        raise Br1Error(f"no solved rows at {args.rows}")

    runtime = Path(args.runtime)
    archive = runtime / "archive.zip"
    observed = up2._sha256_file(archive)
    if args.expect_archive_sha256 and observed != args.expect_archive_sha256:
        raise Br1Error(
            f"pricing archive sha256 {observed} != expected {args.expect_archive_sha256}"
        )
    pricer = load_pricer(archive=archive, runtime=runtime)
    inst_codes = pricer.base_codes.astype(np.int32)

    # Rate-aware selection: a pair is admitted only while its pose gain outweighs
    # the bytes its larger deltas cost.  Greedy by gain-per-byte, re-priced at each
    # step because the Rice parameter k is global and can move.
    ranked = sorted(
        rows.values(), key=lambda r: r["start_d_pose"] - r["final_d_pose"], reverse=True
    )
    n_total = up2.N_PAIRS_TOTAL
    d_pose_ship = args.pointer_d_pose

    selected: list[dict[str, Any]] = []
    candidate = inst_codes.copy()
    best_net = 0.0
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []

    for row in ranked:
        trial = candidate.copy()
        trial[int(row["pair"])] = np.asarray(row["codes"], dtype=np.int32)
        priced = price_codes(trial, pricer)
        gain_sum = sum(
            r["start_d_pose"] - r["final_d_pose"] for r in [*selected, row]
        )
        d_pose_new = d_pose_ship - gain_sum / n_total
        delta_pose = math.sqrt(10.0 * d_pose_new) - math.sqrt(10.0 * d_pose_ship)
        delta_rate = priced["delta_bytes"] * BYTE_TO_SCORE
        net = delta_pose + delta_rate
        trace.append(
            {
                "pair": int(row["pair"]),
                "cumulative_pairs": len(selected) + 1,
                "delta_bytes": priced["delta_bytes"],
                "delta_score_pose": delta_pose,
                "delta_score_rate": delta_rate,
                "net_delta_score": net,
            }
        )
        selected.append(row)
        candidate = trial
        if net < best_net:
            best_net = net
            best_state = {
                "pairs": [int(r["pair"]) for r in selected],
                "priced": priced,
                "d_pose_final": d_pose_new,
                "delta_score_pose": delta_pose,
                "delta_score_rate": delta_rate,
                "net_delta_score": net,
            }

    summary = {
        "schema": "ddm_br1_price.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "pointer_d_pose": d_pose_ship,
        "pointer_archive_bytes": up2.POINTER_ARCHIVE_BYTES,
        "rows_priced": len(rows),
        "admit_bar": ADMIT_BAR,
        "best": best_state,
        "clears_bar": bool(best_state is not None and best_net < ADMIT_BAR),
        "trace": trace,
    }
    (out_dir / "PRICE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ddm_br1 pose carrier basis study")
    sub = parser.add_subparsers(dest="mode", required=True)

    geo = sub.add_parser("geometry", help="invariance control + true span penalty")
    geo.add_argument("--pairs", type=int, default=32)
    geo.add_argument("--seed", type=int, default=20260819)
    geo.add_argument("--out", required=True)
    geo.add_argument("--runtime", default=str(LIVE_RUNTIME))
    geo.add_argument("--expect-archive-sha256", default=LIVE_ARCHIVE_SHA256)
    geo.set_defaults(func=run_geometry)

    gn = sub.add_parser("gn", help="damped Gauss-Newton lattice solve")
    gn.add_argument("--pairs", type=int, default=24)
    gn.add_argument("--seed", type=int, default=20260819)
    gn.add_argument("--iterations", type=int, default=6)
    gn.add_argument("--start-from", choices=("shipped", "up2"), default="shipped")
    gn.add_argument("--no-polish", action="store_true")
    gn.add_argument("--out", required=True)
    gn.add_argument("--runtime", default=str(LIVE_RUNTIME))
    gn.add_argument("--expect-archive-sha256", default=LIVE_ARCHIVE_SHA256)
    gn.set_defaults(func=run_gn)

    ceil = sub.add_parser(
        "ceiling", help="best realized d_pose for an unconstrained band-limited frame 0"
    )
    ceil.add_argument("--pairs", type=int, default=12)
    ceil.add_argument("--seed", type=int, default=20260819)
    ceil.add_argument("--iterations", type=int, default=40)
    ceil.add_argument("--out", required=True)
    ceil.add_argument("--runtime", default=str(LIVE_RUNTIME))
    ceil.add_argument("--expect-archive-sha256", default=LIVE_ARCHIVE_SHA256)
    ceil.set_defaults(func=run_ceiling)

    price = sub.add_parser("price", help="byte-price solved rows, rate-aware selection")
    price.add_argument("--rows", required=True, help="rows.jsonl from mode=gn")
    price.add_argument("--pointer-d-pose", type=float, default=LIVE_D_POSE)
    price.add_argument("--out", required=True)
    price.add_argument("--runtime", default=str(LIVE_RUNTIME))
    price.add_argument("--expect-archive-sha256", default=LIVE_ARCHIVE_SHA256)
    price.set_defaults(func=run_price)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
