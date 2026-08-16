#!/usr/bin/env python3
"""JC1 consumers: the common pose-null dimension K(tolerance), and the pose-metric re-fit.

Consumes the retained JC1 producer payload (per-pair J_i, pose6_gen, pose6_gt) plus the
carrier decoded from the pinned frontier archive through the PROVEN ra2b chain. No
scorer forward, no render, no dispatch -- everything below is arithmetic on measured
arrays, which is why one producer run answers two open items.

=========================  PRE-REGISTRATION  =========================
Written BEFORE this analysis was run. The producer payload existed; these numbers did
not. Git history is the proof.

THE BAR IS DERIVED FROM THE SCORE FORMULA, NOT BORROWED. upstream/evaluate.py:92

    S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/D,      D = 37,545,489

d_seg is invariant to frame_0 carrier edits (MEASURED identically to 8 s.f. on three
independent treatments: alpha=0, alpha=1, rank-4). So a carrier edit returning dB bytes
and multiplying d_pose by `ratio` moves S by

    dS  =  POSE*(sqrt(ratio) - 1)  -  R,     R = 25*dB/D,  POSE = sqrt(10*d_pose_base)

and is admissible iff  ratio <= (1 + R/POSE)^2.  Nothing there is fitted: both inputs
are known before this script runs. The damage constant K = 350,427 and its
recalibration K_eff = 37,953 are deliberately NOT used -- this is a direct first-order
computation from a measured Jacobian, which is the whole point of building it.

FALSIFIER, both directions, fixed in advance:

  (b) K -- the common pose-null dimension, K = 12 - rank(J_stack).
      * K >= 1 at a tolerance defensible as "numerically zero" ==> a strictly pose-free
        AND d_seg-free direction exists; dropping it returns ~1,854 B = -0.001234 S =
        12.9% of the remaining gap, per dimension. Live candidate, owes a byte-closed row.
      * K = 0 at every defensible tolerance ==> no EXACTLY free direction; the exactness
        framing closes and only the approximate one survives, priced by (a).
      Generic expectation is K = 0. That is a PRIOR and is not spent as a measurement.

  (a) The pose-metric re-fit vs the Euclidean re-fit, at MATCHED keep-set size, both
      selected EXHAUSTIVELY over all C(12,r) keep-sets -- the same search
      ddm_ra2a_carrier_fidelity_pose_ladder.py:143-160 runs, with the same Gram, so the
      only difference is the objective.
      * CLEARS if the pose-optimal predicted ratio <= (1 + R/POSE)^2 on the advisory
        axis at any r <= 11. The rate rung re-opens and owes a byte-closed row.
      * REFUSED if it misses at every rung. The carrier family is then closed in the
        pose metric too, at first order, and the carrier stops consuming slots.
      * The CONTRAST is reported either way -- it is the first DIRECT measurement of what
        the metric choice is worth on this object. ra2c could only bound it indirectly
        (9.23x, which was a statement about the damage law's calibration, not about
        subspace choice).

  (c) LINEARITY CONTROL. The rank-4 SVD truncation is the one treatment whose true
      d_pose is MEASURED (0.35402399, ra2c). Predicting it from J tests the
      linearization's validity radius at 25.14% Frobenius error. A large miss does not
      invalidate small-perturbation predictions; it BOUNDS where they apply, and it is
      reported either way.
======================================================================

Axis: [macOS-CPU advisory]. score_claim=false, promotable=false. No paid compute.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RA2B_SOURCE = REPO / "experiments/ddm_ra2b_carrier_chain_control.py"
RA2C_SOURCE = REPO / "experiments/ddm_ra2c_alpha_ladder.py"

DEFAULT_PAYLOAD = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")
DEFAULT_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/archive.zip"
)

UNCOMPRESSED_BYTES = 37_545_489
BASE_D_POSE = 0.00014747
BASE_D_SEG = 0.00042714
FRONTIER_S = 0.15959729295498598
TARGET_S = 0.15

#: T4 (shipping axis) pose term from the frontier row; the advisory instrument is 4.63x larger.
POSE_TERM_T4 = 0.0082945765
POSE_TERM_ADVISORY = math.sqrt(10.0 * BASE_D_POSE)

CARRIER_DIM = 12
POSE_ROWS = 6

#: ra2c 8.1's measured byte-return ladder: 1,854 B at r=11, 3,707 at r=10, 14,828 at r=4
#: -- linear at 1853.5 B per dropped coefficient dimension.
BYTES_PER_DROPPED_DIM = 1853.5

#: ra2c's MEASURED rank-4 row, the single point where true d_pose is known off-base.
RA2C_RANK4_MEASURED_D_POSE = 0.35402399

#: Ridges emitted as EXACT-eval candidates. Not a chosen value -- the point is that the
#: first-order screen cannot pick one honestly (step is flat across the sweep while the
#: predicted ratio moves 4000x), so several are measured and the MEASUREMENT arbitrates.
EXACT_EVAL_RIDGES = (1e2, 1e1, 1e0, 1e-1)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_module(name: str, source: Path):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bar_for_bytes(returned_bytes: float, pose_term: float) -> float:
    """ratio <= (1 + R/POSE)^2, derived from upstream/evaluate.py:92. Not fitted."""
    return (1.0 + (25.0 * returned_bytes / UNCOMPRESSED_BYTES) / pose_term) ** 2


def predicted_d_pose(jac: np.ndarray, resid: np.ndarray, delta: np.ndarray) -> float:
    """First order: d_pose_i ~= ||r_i + J_i . delta_i||^2 / 6, averaged over pairs."""
    moved = resid + np.einsum("nij,nj->ni", jac, delta)
    return float((moved ** 2).sum(axis=1).mean() / POSE_ROWS)


def pose_refit_delta(jac: np.ndarray, coeff: np.ndarray, resid: np.ndarray,
                     keep_idx: np.ndarray, drop_idx: np.ndarray,
                     ridge: float = 0.0) -> np.ndarray:
    """Pose-metric re-fit: minimize ||r_i + J_i.delta_i||^2 + ridge*||delta_kept||^2.

    Dropped coordinates are FORCED to -c_i (their column is not stored at all); the kept
    coordinates are free.

    ==> THE DEGENERACY THIS FUNCTION MUST NOT HIDE. d_pose reads only 6 scalars per pair.
    With r >= 6 kept coordinates, J_S (6 x r) generically has FULL ROW RANK, so the
    unregularized system is UNDER-DETERMINED and its residual can be driven to EXACTLY
    zero -- the first-order model will happily report d_pose ~ 0, BELOW base, for every
    keep-set of size >= 6. That is the linear model steering the scorer onto its own
    ground truth by extrapolating far outside its validity radius; it is an artifact of
    dimension counting, not a compression result. The caller MUST read the returned step
    size (``step_rms_relative``) alongside any prediction, and a nonzero ``ridge`` is the
    trust region that makes the ladder discriminate at all.
    """
    n = jac.shape[0]
    delta = np.zeros((n, CARRIER_DIM))
    forced = -coeff[:, drop_idx] if drop_idx.size else np.zeros((n, 0))
    if drop_idx.size:
        delta[:, drop_idx] = forced
    if keep_idx.size:
        b = resid + (np.einsum("nij,nj->ni", jac[:, :, drop_idx], forced)
                     if drop_idx.size else 0.0)
        js = jac[:, :, keep_idx]                                    # (n, 6, r)
        if ridge > 0.0:
            # Tikhonov / Levenberg trust region: (J^T J + ridge I) x = -J^T b.
            gram_k = np.einsum("nij,nik->njk", js, js)
            gram_k += ridge * np.eye(keep_idx.size)[None]
            rhs = -np.einsum("nij,ni->nj", js, b)
            delta[:, keep_idx] = np.linalg.solve(gram_k, rhs)
        else:
            delta[:, keep_idx] = -np.einsum("nrj,nj->nr", np.linalg.pinv(js), b)
    return delta


def step_rms_relative(delta: np.ndarray, coeff: np.ndarray) -> float:
    """RMS ||delta_i|| / RMS ||c_i||. A step >= 1 rewrites the carrier, not refines it."""
    return float(np.sqrt((delta ** 2).sum(axis=1).mean())
                 / np.sqrt((coeff ** 2).sum(axis=1).mean()))


def euclid_refit_delta(coeff: np.ndarray, gram: np.ndarray,
                       keep_idx: np.ndarray, drop_idx: np.ndarray) -> np.ndarray:
    """The INCUMBENT Euclidean re-fit, reproduced from ra2a:143-160 exactly.

    Generalized least squares in the receiver's own normalized-basis Gram: the refit
    minimizes the rendered-FIELD error, with no reference to what the scorer reads.
    """
    n = coeff.shape[0]
    approx = np.zeros((n, CARRIER_DIM))
    if keep_idx.size:
        grr = gram[np.ix_(keep_idx, keep_idx)]
        grc = gram[np.ix_(keep_idx, np.arange(CARRIER_DIM))]
        approx[:, keep_idx] = np.linalg.lstsq(grr, grc @ coeff.T, rcond=None)[0].T
    return approx - coeff


def euclid_field_mse(coeff: np.ndarray, gram: np.ndarray, delta: np.ndarray) -> float:
    return float(np.einsum("bi,ij,bj->b", delta, gram, delta).mean())


def null_dimension_curve(jac: np.ndarray, label: str, rationale: str) -> dict:
    """K(tolerance) = 12 - rank(J_stack). One SVD; the spectrum IS the curve.

    v is in null(G_i) for every i  <=>  J_i v = 0 for every i  <=>  J_stack v = 0,
    where J_stack is the (600*6 x 12) vertical stack. That identity is why the
    600-fold intersection costs one SVD of a 3600x12 matrix.
    """
    stack = jac.reshape(-1, jac.shape[-1])
    sv = np.linalg.svd(stack, compute_uv=False)
    rel = sv / float(sv[0])
    numpy_tol = max(stack.shape) * np.finfo(np.float64).eps
    tolerances = (1e-16, numpy_tol, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5,
                  1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1)
    return {
        "coordinate": label,
        "coordinate_rationale": rationale,
        "stack_shape": list(stack.shape),
        "singular_values": [float(v) for v in sv],
        "singular_values_relative_to_sigma1": [float(v) for v in rel],
        "condition_number": float(sv[0] / sv[-1]),
        "numpy_default_rel_tolerance": float(numpy_tol),
        "K_at_numpy_default_tolerance": int(np.count_nonzero(rel <= numpy_tol)),
        "K_curve": [
            {"rel_tolerance": float(t), "K": int(np.count_nonzero(rel <= t))}
            for t in tolerances
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--candidate-ranks", type=str, default="11,10,8",
                        help="emit candidate coefficient matrices at these keep-ranks for "
                             "EXACT n600 re-evaluation through the real chain")
    args = parser.parse_args()
    payload = Path(args.payload)

    receipt = json.loads((payload / "JC1_PRODUCER.json").read_text())
    if receipt.get("partial") and not args.allow_partial:
        raise SystemExit("producer payload is PARTIAL -- refusing to draw a verdict from it")
    for name, meta in receipt["payload"].items():
        actual = sha256_file(payload / name)
        if actual != meta["sha256"]:
            raise SystemExit(f"payload sha mismatch for {name}: {actual} != {meta['sha256']}")

    jac = np.load(payload / "jacobian_pose6_x_coeff12.float64.npy")
    pose_gen = np.load(payload / "pose6_generated.float64.npy")
    pose_gt = np.load(payload / "pose6_groundtruth.float64.npy")
    have_gt = np.load(payload / "have_gt.bool.npy")
    control = np.load(payload / "control_byte_identical.bool.npy")
    pair_ids = np.load(payload / "pair_ids.int32.npy")
    if not control.all():
        raise SystemExit("producer byte-identity control did not pass on every pair -- refusing")
    if not have_gt.all():
        raise SystemExit("producer lacks GT pose on some pairs -- the residual r_i is required")
    n = int(jac.shape[0])

    # ---- decode the carrier through the SAME proven chain ----------------------
    ra2b = load_module("ra2b_chain", RA2B_SOURCE)
    chain = ra2b.load_chain()
    renderer = chain[0]
    basis, coeff_t, _sel, _prov = ra2b.decode_carrier(Path(args.archive), chain)
    coeff = np.asarray(coeff_t, dtype=np.float64)
    flat = renderer.normalized_basis(basis.double()).reshape(basis.shape[0], -1)
    gram = (flat @ flat.T).numpy() / flat.shape[1]      # ra2a:222-223, verbatim
    if not np.isfinite(gram).all() or not np.isfinite(coeff).all():
        raise SystemExit("carrier decode produced non-finite values")

    # ---- POSITIVE CONTROL: rebuild base d_pose from the residuals --------------
    resid = pose_gen - pose_gt
    rebuilt = float((resid ** 2).sum(axis=1).mean() / POSE_ROWS)
    positive_control = {
        "rebuilt_base_d_pose": rebuilt,
        "retained_report_d_pose": BASE_D_POSE,
        "relative_error": abs(rebuilt - BASE_D_POSE) / BASE_D_POSE,
        "note": (
            "The retained report prints d_pose to 8 decimals, so agreement is bounded "
            "below by that rounding (~3.4e-5 relative). Agreement at that level confirms "
            "this producer reproduces upstream evaluate.py's pose pipeline end to end, "
            "including the batch-1 vs batch-16 instrument question."
        ),
    }

    # ---- CONSUMER (b): K(tolerance), two NAMED coordinates ---------------------
    coeff_rms = np.sqrt((coeff ** 2).mean(axis=0))
    curves = [
        null_dimension_curve(
            jac, "raw_stored_coefficient",
            "the coordinate the archive actually stores; units = pose output per stored unit",
        ),
        null_dimension_curve(
            jac * coeff_rms[None, None, :], "coefficient_rms_whitened",
            "each column scaled by RMS_i(c_ik); units = pose output per ONE TYPICAL USE of "
            "that dimension, which is the size of the move 'drop this column' actually makes",
        ),
    ]

    # ---- CONSUMER (a): exhaustive keep-set, both metrics ----------------------
    zero_delta = -coeff  # alpha = 0: the whole carrier deleted
    rungs = []
    candidate_ranks = {int(v) for v in args.candidate_ranks.split(",") if v.strip()}
    candidates: dict[str, np.ndarray] = {}
    for r in range(11, 0, -1):
        dropped = CARRIER_DIM - r
        returned = dropped * BYTES_PER_DROPPED_DIM
        best = {}
        for subset in itertools.combinations(range(CARRIER_DIM), r):
            keep_idx = np.asarray(subset)
            drop_idx = np.asarray([k for k in range(CARRIER_DIM) if k not in subset])
            d_pose = pose_refit_delta(jac, coeff, resid, keep_idx, drop_idx)
            d_eucl = euclid_refit_delta(coeff, gram, keep_idx, drop_idx)
            p_pose = predicted_d_pose(jac, resid, d_pose)
            p_eucl = predicted_d_pose(jac, resid, d_eucl)
            f_eucl = euclid_field_mse(coeff, gram, d_eucl)
            if not best or p_pose < best["pose_best_d_pose"]:
                best["pose_best_d_pose"] = p_pose
                best["pose_best_keep"] = list(subset)
                best["pose_best_step"] = step_rms_relative(d_pose, coeff)
                best["pose_best_delta"] = d_pose
            if not best.get("euclid_selected") or f_eucl < best["euclid_field_mse"]:
                best["euclid_selected"] = list(subset)
                best["euclid_field_mse"] = f_eucl
                best["euclid_selected_d_pose"] = p_eucl
                best["euclid_step"] = step_rms_relative(d_eucl, coeff)
                best["euclid_delta"] = d_eucl
        underdetermined = r >= POSE_ROWS
        entry = {
            "keep_rank": r,
            "dropped_dims": int(dropped),
            "returned_bytes": float(returned),
            "bar_advisory": bar_for_bytes(returned, POSE_TERM_ADVISORY),
            "bar_t4_ratio_transfer": bar_for_bytes(returned, POSE_TERM_T4),
            "first_order_underdetermined": bool(underdetermined),
            "underdetermined_note": (
                "r >= 6 = the number of scored pose dims, so the unregularized first-order "
                "residual is drivable to ~0 for EVERY keep-set of this size. The predicted "
                "ratio below is therefore NOT a compression result and MUST NOT be read as "
                "clearing any bar; read the trust-region row and the step size instead."
                if underdetermined else
                "r < 6: the first-order system is over-determined, so the predicted ratio "
                "discriminates between keep-sets on its own."
            ),
            "pose_metric_optimal_unregularized": {
                "keep_set": best["pose_best_keep"],
                "d_pose_predicted": best["pose_best_d_pose"],
                "ratio_vs_base": best["pose_best_d_pose"] / BASE_D_POSE,
                "step_rms_relative": best["pose_best_step"],
            },
            "euclidean_incumbent": {
                "keep_set": best["euclid_selected"],
                "field_mse": best["euclid_field_mse"],
                "d_pose_predicted": best["euclid_selected_d_pose"],
                "ratio_vs_base": best["euclid_selected_d_pose"] / BASE_D_POSE,
                "step_rms_relative": best["euclid_step"],
            },
        }
        # Trust-region sweep on the pose-optimal keep-set: the ladder that DOES
        # discriminate, because it prices the step the linear model is allowed to take.
        keep_idx = np.asarray(best["pose_best_keep"])
        drop_idx = np.asarray([k for k in range(CARRIER_DIM) if k not in keep_idx])
        tr = []
        for ridge in (1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2):
            d = pose_refit_delta(jac, coeff, resid, keep_idx, drop_idx, ridge=ridge)
            tr.append({
                "ridge": float(ridge),
                "d_pose_predicted": predicted_d_pose(jac, resid, d),
                "ratio_vs_base": predicted_d_pose(jac, resid, d) / BASE_D_POSE,
                "step_rms_relative": step_rms_relative(d, coeff),
            })
        entry["pose_trust_region_sweep"] = tr
        if r in candidate_ranks:
            for ridge in (1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2):
                if ridge in EXACT_EVAL_RIDGES:
                    d = pose_refit_delta(jac, coeff, resid, keep_idx, drop_idx, ridge=ridge)
                    candidates[f"pose_refit_r{r}_ridge{ridge:g}"] = coeff + d
        rungs.append(entry)
        if r in candidate_ranks:
            candidates[f"euclid_refit_r{r}"] = coeff + best["euclid_delta"]

    # ---- CONTROL (c): the linearization against the ONE measured off-base point
    ra2c = load_module("ra2c_ladder", RA2C_SOURCE)
    coeff_r4, r4_info = ra2c.truncate_carrier_rank(basis, coeff_t, 4)
    delta_r4 = np.asarray(coeff_r4, dtype=np.float64) - coeff
    linearity = {
        "treatment": "ra2c rank-4 SVD truncation of the rendered field (Euclidean-optimal)",
        "frobenius_rel_err": float(r4_info["rel_frobenius_err"]),
        "step_rms_relative": step_rms_relative(delta_r4, coeff),
        "d_pose_measured": RA2C_RANK4_MEASURED_D_POSE,
        "d_pose_first_order_predicted": predicted_d_pose(jac, resid, delta_r4),
        "d_pose_alpha0_first_order_predicted": predicted_d_pose(jac, resid, zero_delta),
        "note": (
            "A first-order prediction at 25% Frobenius error is FAR outside any expected "
            "validity radius. This row measures HOW far, so the small-perturbation rungs "
            "above carry an honest error bar instead of an assumed one."
        ),
    }

    result = {
        "schema": "ddm_jc1_pose_null_and_refit.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "verdict_scope_note": "first order on the STE-relaxed chain; a NECESSARY screen, not sufficient",
        "n_pairs": n,
        "pair_id_span": [int(pair_ids.min()), int(pair_ids.max())],
        "gradient_status": receipt["gradient_status"],
        "producer_receipt_sha256": sha256_file(payload / "JC1_PRODUCER.json"),
        "base_d_pose": BASE_D_POSE,
        "base_d_seg": BASE_D_SEG,
        "frontier_S": FRONTIER_S,
        "remaining_gap_to_target": FRONTIER_S - TARGET_S,
        "positive_control": positive_control,
        "null_dimension": curves,
        "refit_ladder": rungs,
        "linearity_control": linearity,
        "column_pose_sensitivity_raw": [
            float(v) for v in np.linalg.norm(jac.reshape(-1, CARRIER_DIM), axis=0)
        ],
        "coeff_rms_per_dim": [float(v) for v in coeff_rms],
    }

    # ALWAYS KEEP THE PAYLOAD: the decoded carrier + Gram this analysis consumed.
    extra = {"coeff.float64.npy": coeff, "basis_gram.float64.npy": gram,
             "coeff_rms.float64.npy": coeff_rms}
    for name, arr in candidates.items():
        extra[f"candidate_{name}.float64.npy"] = arr
    stored = {}
    for name, arr in extra.items():
        target = payload / name
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
        os.replace(tmp, target)
        stored[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size,
                        "shape": list(np.asarray(arr).shape)}
    result["consumed_payload"] = stored

    text = json.dumps(result, indent=2, sort_keys=True)
    (payload / "JC1_CONSUMERS.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
