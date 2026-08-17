"""ra2crr: the PRICED successor to the binary common-pose-null test, from banked artifacts only.

jc1 MEASURED ``K = dim(intersect_i null(J_i)) = 0`` at every relative tolerance down to 3%
(``JC1_CONSUMERS.json``), which closes the *exactness* framing: no carrier direction is
exactly pose-free.  ``K`` is a rank test on the stacked Jacobian, so it answers "is there a
direction with ZERO pose response" and nothing else.  Two quantities that decide the SCORE are
invisible to it:

  1. **the per-pair coefficient weighting.**  Dropping direction ``v`` perturbs pair ``i`` by
     ``delta_i = -(z_i . v) v`` -- the damage is weighted by how much each pair actually USES
     ``v``.  A direction with a large pose response that no pair uses is cheap; ``sigma_min``
     of the stacked Jacobian cannot see this.  jc1's RMS-whitened coordinate is a diagonal
     approximation of the weighting, not the weighting.
  2. **the residual cross-term.**  ``d_pose`` is measured against a ground truth the base render
     already misses, so the damage of a perturbation ``p_i`` is ``2<r_i, p_i> + |p_i|^2`` with
     ``r_i`` the BASE residual.  The cross-term is first order and can be NEGATIVE -- dropping a
     direction can move pose TOWARD the ground truth.  ra3 measured that ignoring this term
     misprices by 2.64x precisely near break-even.

This tool minimises the exact first-order damage over the unit sphere in carrier coordinates,
prices the minimiser against the MEASURED byte credit, and calibrates the linear model against
every retained ra3 candidate so the bound is quantified rather than asserted.

Scorer-free by construction: consumes banked ``pose6``/Jacobian arrays and the authority GT
cache.  No render, no scorer forward, no dispatch.

MODEL SCOPE, stated up front.  ``J`` is jc1's STE-relaxed Jacobian
(``gradient_status = MEASURED_ON_STE_RELAXED_CHAIN``); the exactly-quantized chain has a zero
Jacobian almost everywhere and carries no local information, so the STE relaxation is the only
local model available.  Every number here is therefore a MODEL quantity.  Stage ``calibration``
measures the model's signed error against 12 realised candidates spanning four decades of
perturbation size, which is what converts the minimiser into a usable one-sided bound.

Axis: [authority-tracking GT, MEASURED 1.00081x vs contest-CUDA by pi2].  Not a score claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JC1_RETAINED = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")
RA3_RETAINED = Path("/Volumes/APDataStore/pact/ddm_ra3/retained")
RA3_ROOT = Path("/Volumes/APDataStore/pact/ddm_ra3")
OUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_ra2crr")

#: MEASURED by pi2: tracks the contest authority at 1.00081x.  READ-ONLY (VertigoDataTier full).
GT_CACHE_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")

FRAMES, POSE_DIMS, CARRIER_DIM = 600, 6, 12
S_PER_BYTE = 25.0 / 37_545_489.0

#: ra3's authority-GT base, re-derived here as a control (refuses on mismatch).
AUTHORITY_BASE_D_POSE = 6.88559506e-06
#: hv1 ep0634 frontier, [contest-CUDA T4, n600].
FRONTIER_S = 0.15959729295498598
TARGET_S = 0.15

#: MEASURED byte credit for dropping ONE carrier dimension, three conventions.
#: 913/918 = ra3 RA3_RATE_CHECK_r11.json, real CPR1 blobs through the shipped carrier_codec.
#: 1658.08 = ra3 "most favourable" (basis pro-rated, coefficient half measured).
#: 1846.75 = ra2 assumed uniform (22,161 B pool / 12 dims); kept for comparability.
CREDIT_BYTES = {"measured_container": 913.0,
                "most_favourable": 1658.0833333333335,
                "ra2_assumed_uniform": 1846.75}

#: sha256 pins from JC1_PRODUCER.json -- custody is asserted, not assumed.
JC1_SHA_PINS = {
    "jacobian_pose6_x_coeff12.float64.npy":
        "fc5743c970c13f4f5f41cea2609e1196cc2dcb0286678991720f87efdc1675c5",
    "coeff.float64.npy": None,          # not pinned by the producer receipt; shape-checked only
    "pose6_generated.float64.npy":
        "2dba9590b1811f0e5b9085712de6a83de854445a7c096916d54525c8ebeaa537",
}


#: numpy 1.26.4 on Apple Accelerate raises a SPURIOUS ``divide by zero encountered in matmul``
#: for finite inputs with a finite result.  POSITIVE CONTROL: it fires on pure-random finite
#: ``(600,12) @ (12,)`` input whose product is finite, so it is a BLAS flag artefact and not a
#: statement about our data (the banked arrays are checked all-finite by
#: :func:`stage_controls`).  The flag is suppressed only around the matmul; the invariant that
#: actually matters -- a finite objective value -- is asserted explicitly at each call site.
_BLAS_FLAG_SUPPRESSION = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


class PricedNullRefusal(RuntimeError):
    """Fail-closed refusal: missing custody, wrong shape, or a broken control."""


# ---------------------------------------------------------------- payload custody (P0)

def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def retain_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Persist an array with sha256 + byte count.  ALWAYS KEEP THE PAYLOAD (P0)."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    _atomic_bytes(path, payload)
    return {"path": str(path), "bytes": len(payload), "dtype": str(array.dtype),
            "shape": list(array.shape), "sha256": hashlib.sha256(payload).hexdigest()}


def _load_pinned(directory: Path, name: str, shape: tuple[int, ...]) -> np.ndarray:
    path = directory / name
    if not path.exists():
        raise PricedNullRefusal(f"banked artifact missing: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    pin = JC1_SHA_PINS.get(name)
    if pin is not None and digest != pin:
        raise PricedNullRefusal(f"custody break on {name}: {digest} != pinned {pin}")
    array = np.load(io.BytesIO(payload), allow_pickle=False).astype(np.float64)
    if array.shape != shape:
        raise PricedNullRefusal(f"{name} shape {array.shape} != {shape}")
    return array


# ---------------------------------------------------------------- score arithmetic

def d_pose(generated: np.ndarray, ground_truth: np.ndarray) -> float:
    """upstream's pose distortion: mean over the 6 dims, then mean over pairs."""
    return float(((generated - ground_truth) ** 2).mean(axis=1).mean())


def dS_pose(candidate: float, base: float) -> float:
    """Score delta of the pose term between two d_pose values on the SAME axis."""
    return float(np.sqrt(10.0 * candidate) - np.sqrt(10.0 * base))


def break_even_delta_d_pose(bytes_returned: float, base: float) -> float:
    """Largest absolute d_pose increase a byte return still pays for."""
    credit = bytes_returned * S_PER_BYTE
    return float(((np.sqrt(10.0 * base) + credit) ** 2) / 10.0 - base)


# ---------------------------------------------------------------- the model

def damage_no_refit(v: np.ndarray, coeff: np.ndarray, jac: np.ndarray,
                    residual: np.ndarray) -> float:
    """Exact first-order ``delta d_pose`` of DROPPING unit direction ``v``, no re-fit.

    ``delta_i = -(z_i . v) v``  =>  ``p_i = J_i delta_i = -a_i w_i``  with ``a_i = z_i . v``,
    ``w_i = J_i v``.  Then

        delta d_pose = (1/(6N)) sum_i [ 2<r_i, p_i> + |p_i|^2 ]
                     = (1/(6N)) sum_i [ -2 a_i <r_i, w_i> + a_i^2 |w_i|^2 ].

    The cross-term is retained with its sign: it is first order and may be negative.
    """
    if not np.all(np.isfinite(v)):
        return float("inf")
    with np.errstate(**_BLAS_FLAG_SUPPRESSION):
        a = coeff @ v                                 # (N,)
    w = np.einsum("nij,j->ni", jac, v)                # (N,6)
    cross = np.einsum("ni,ni->n", residual, w)        # (N,)
    per_pair = -2.0 * a * cross + (a ** 2) * (w ** 2).sum(axis=1)
    value = float(per_pair.sum() / (POSE_DIMS * coeff.shape[0]))
    if not np.isfinite(value):                        # the invariant the flag cannot certify
        return float("inf")
    return value


def damage_grad(v: np.ndarray, coeff: np.ndarray, jac: np.ndarray,
                residual: np.ndarray) -> np.ndarray:
    """Analytic gradient of :func:`damage_no_refit` w.r.t. an UNNORMALISED ``v``."""
    if not np.all(np.isfinite(v)):
        return np.zeros(CARRIER_DIM)
    with np.errstate(**_BLAS_FLAG_SUPPRESSION):
        a = coeff @ v
    w = np.einsum("nij,j->ni", jac, v)
    cross = np.einsum("ni,ni->n", residual, w)
    s = (w ** 2).sum(axis=1)
    jt_r = np.einsum("nij,ni->nj", jac, residual)     # (N,12)  J_i^T r_i
    g_v = np.einsum("nij,ni->nj", jac, w)             # (N,12)  G_i v
    grad = (-2.0 * cross[:, None] * coeff
            - 2.0 * a[:, None] * jt_r
            + 2.0 * a[:, None] * s[:, None] * coeff
            + 2.0 * (a ** 2)[:, None] * g_v)
    return grad.sum(axis=0) / (POSE_DIMS * coeff.shape[0])


def _riemannian_descent(v0, coeff, jac, residual, iterations=4000, tol=1e-16):
    """Minimise on the unit sphere directly.

    The objective is scale-invariant, so an unconstrained parametrisation ``v = u/|u|`` has an
    exactly flat radial direction; L-BFGS then drifts the norm to 0 or inf and produces
    non-finite iterates.  Renormalising every step removes the degeneracy instead of penalising
    it.  Backtracking guarantees monotone descent, so the returned point is a genuine local
    minimum of the sphere-restricted objective.
    """
    v = v0 / np.linalg.norm(v0)
    value = damage_no_refit(v, coeff, jac, residual)
    step = 1.0
    for _ in range(iterations):
        raw = damage_grad(v, coeff, jac, residual)
        grad = raw - (raw @ v) * v                    # project onto the tangent space
        norm = np.linalg.norm(grad)
        if norm < 1e-18:
            break
        improved = False
        for _ in range(60):
            trial = v - step * grad
            trial_norm = np.linalg.norm(trial)
            if trial_norm > 1e-300 and np.isfinite(trial_norm):
                trial = trial / trial_norm
                candidate = damage_no_refit(trial, coeff, jac, residual)
                if np.isfinite(candidate) and candidate < value - tol * abs(value):
                    v, value, improved = trial, candidate, True
                    step = min(step * 2.0, 1e12)      # cap: unbounded growth overflows
                    break
            step *= 0.5
        if not improved:
            break
    return v, value


# ---------------------------------------------------------------- stages

def stage_controls(coeff, jac, generated, gt_dali, gt_pyav) -> dict[str, Any]:
    """Re-derive the base on the authority axis and check the model's own preconditions."""
    for name, array in (("coeff", coeff), ("jacobian", jac), ("generated", generated),
                        ("gt_dali", gt_dali), ("gt_pyav", gt_pyav)):
        if not np.all(np.isfinite(array)):
            raise PricedNullRefusal(f"banked array {name} carries non-finite entries")
    base_dali = d_pose(generated, gt_dali)
    base_pyav = d_pose(generated, gt_pyav)
    closure = abs(base_dali - AUTHORITY_BASE_D_POSE) / AUTHORITY_BASE_D_POSE
    if closure > 1e-6:
        raise PricedNullRefusal(
            f"authority base d_pose {base_dali!r} does not reproduce ra3's "
            f"{AUTHORITY_BASE_D_POSE!r} (relative {closure:.3e})")
    grams = np.einsum("nij,nik->njk", jac, jac)
    per_pair_rank = [int(np.linalg.matrix_rank(g, tol=1e-10 * max(1.0, np.abs(g).max())))
                     for g in grams]
    return {
        "base_d_pose_authority_gt": base_dali,
        "base_d_pose_pyav_advisory": base_pyav,
        "authority_base_reproduces_ra3_relative": closure,
        "per_pair_gram_rank_min": int(min(per_pair_rank)),
        "per_pair_gram_rank_max": int(max(per_pair_rank)),
        "per_pair_gram_rank_is_6_fraction": float(np.mean(np.array(per_pair_rank) == 6)),
        "coeff_rms": float(np.sqrt((coeff ** 2).mean())),
    }


def stage_calibration(coeff, jac, generated, gt_dali, base_dali) -> dict[str, Any]:
    """Signed model error against every realised ra3 candidate -- four decades of step size.

    This is what makes the minimiser a BOUND rather than an extrapolation: it measures whether
    the linear model understates or overstates realised damage, and by how much.
    """
    rescore = json.loads((RA3_ROOT / "RA3_AUTHORITY_GT_RESCORE_r11.json").read_text())
    measured = {row["label"]: row["d_pose_dali_authority"] for row in rescore["rows"]}
    residual = generated - gt_dali
    rows = []
    candidates = [(p.name[len("ra3_r11_"):-len(".float64.npy")], p)
                  for p in sorted(RA3_RETAINED.glob("ra3_r11_*.float64.npy"))]
    for label, path in candidates:
        if label.startswith("pose6") or label.endswith(("_q12", "_authority_gt")):
            continue
        candidate = np.load(path, allow_pickle=False).astype(np.float64)
        if candidate.shape != (FRAMES, CARRIER_DIM):
            continue
        if label not in measured:                     # only candidates ra3 actually realised
            continue
        delta = candidate - coeff
        p = np.einsum("nij,nj->ni", jac, delta)
        predicted = float((2.0 * (residual * p).sum(axis=1) + (p ** 2).sum(axis=1)).sum()
                          / (POSE_DIMS * FRAMES))
        realised = measured[label] - base_dali
        rows.append({
            "label": label,
            "step_rms_over_coeff_rms": float(np.sqrt((delta ** 2).mean())
                                             / np.sqrt((coeff ** 2).mean())),
            "predicted_delta_d_pose": predicted,
            "realised_delta_d_pose": realised,
            "realised_over_predicted": (float(realised / predicted)
                                        if predicted not in (0.0,) else None),
        })
    rows.sort(key=lambda r: r["step_rms_over_coeff_rms"])
    ratios = [r["realised_over_predicted"] for r in rows
              if r["realised_over_predicted"] is not None]
    understates = [r for r in ratios if r > 1.0]
    return {
        "rows": rows,
        "n_candidates": len(rows),
        "model_understates_damage_count": len(understates),
        "model_understates_damage_fraction": (len(understates) / len(ratios)) if ratios else None,
        "realised_over_predicted_min": min(ratios) if ratios else None,
        "realised_over_predicted_max": max(ratios) if ratios else None,
    }


def stage_refit_degeneracy(jac, rng) -> dict[str, Any]:
    """Is a MODEL bound on the RE-FIT family possible at all?

    Dropping ``v`` and re-fitting the surviving 11 coefficients per pair means solving
    ``r_i + J_i P w = a_i J_i v`` for ``w`` in ``v``-perp.  ``J_i P`` is 6x12 restricted to an
    11-dim subspace; if it has rank 6 the system is exactly solvable and the model claims ZERO
    residual for every pair -- i.e. the model-optimal re-fit is vacuous and can bound nothing.
    """
    probes = []
    for _ in range(8):
        v = rng.normal(size=CARRIER_DIM)
        v /= np.linalg.norm(v)
        projector = np.eye(CARRIER_DIM) - np.outer(v, v)
        ranks = np.array([np.linalg.matrix_rank(j @ projector, tol=1e-9) for j in jac])
        probes.append(float(np.mean(ranks == POSE_DIMS)))
    return {
        "random_directions_probed": len(probes),
        "fraction_pairs_with_full_rank_6_min": float(min(probes)),
        "fraction_pairs_with_full_rank_6_max": float(max(probes)),
        "model_optimal_refit_residual_is_zero": bool(min(probes) == 1.0),
        "consequence": (
            "6 pose constraints against 11 free coefficients: the first-order re-fit is "
            "exactly solvable for every pair and every direction, so the model-optimal "
            "re-fit damage is identically zero and NO model-based bound on the re-fit "
            "family exists. Any re-fit verdict must be REALISED. This is the structural "
            "reason jc1's designer failed and ra3 needed realised acceptance."),
    }


def stage_minimise(coeff, jac, generated, gt_dali, restarts, seed) -> dict[str, Any]:
    """Minimise the exact first-order drop damage over the unit sphere."""
    residual = generated - gt_dali
    rng = np.random.default_rng(seed)

    def evaluate(v):
        return damage_no_refit(v / np.linalg.norm(v), coeff, jac, residual)

    structured: dict[str, np.ndarray] = {}
    for k in range(CARRIER_DIM):
        e = np.zeros(CARRIER_DIM)
        e[k] = 1.0
        structured[f"coordinate_axis_{k}"] = e
    stack = jac.reshape(FRAMES * POSE_DIMS, CARRIER_DIM)
    _, _, vt = np.linalg.svd(stack, full_matrices=False)
    for k in range(CARRIER_DIM):
        structured[f"stacked_jacobian_right_singular_{k}"] = vt[k]
    gram = np.einsum("nij,nik->jk", jac, jac) / FRAMES
    eigval, eigvec = np.linalg.eigh(gram)
    for k in range(CARRIER_DIM):
        structured[f"mean_gram_eigvec_{k}"] = eigvec[:, k]

    structured_rows = sorted(
        ({"direction": name, "delta_d_pose": evaluate(vec)}
         for name, vec in structured.items()),
        key=lambda r: r["delta_d_pose"])

    # Seed from every structured direction as well as random restarts, so the reported optimum
    # can never be worse than the best structured direction.
    starts = list(structured.values())
    starts += [rng.normal(size=CARRIER_DIM) for _ in range(restarts)]
    best_value, best_vec, local_minima = np.inf, None, []
    for start in starts:
        v, value = _riemannian_descent(start, coeff, jac, residual)
        local_minima.append(value)
        if value < best_value:
            best_value, best_vec = value, v
    distinct = float(np.std(local_minima)) / max(abs(float(np.mean(local_minima))), 1e-300)

    # CONTROL: the analytic gradient against a central finite difference at the optimum.
    step, fd = 1e-6, np.zeros(CARRIER_DIM)
    for k in range(CARRIER_DIM):
        e = np.zeros(CARRIER_DIM)
        e[k] = step
        fd[k] = (damage_no_refit(best_vec + e, coeff, jac, residual)
                 - damage_no_refit(best_vec - e, coeff, jac, residual)) / (2 * step)
    analytic = damage_grad(best_vec, coeff, jac, residual)
    scale = max(np.abs(fd).max(), 1e-30)
    gradient_check = float(np.abs(analytic - fd).max() / scale)
    if gradient_check > 1e-4:
        raise PricedNullRefusal(
            f"analytic gradient disagrees with finite difference: {gradient_check:.3e}")

    if best_value > structured_rows[0]["delta_d_pose"] + 1e-18:
        raise PricedNullRefusal(
            f"optimiser returned {best_value!r}, worse than the best structured direction "
            f"{structured_rows[0]['delta_d_pose']!r} -- descent is broken")

    return {
        "restarts": restarts,
        "starts_including_structured": len(starts),
        "structured_directions": structured_rows,
        "best_structured_delta_d_pose": structured_rows[0]["delta_d_pose"],
        "best_structured_direction": structured_rows[0]["direction"],
        "global_min_delta_d_pose": best_value,
        "global_min_direction": [float(x) for x in best_vec],
        "local_minima_relative_spread": distinct,
        "local_minima_count_within_1pct_of_best": int(
            sum(1 for m in local_minima if m <= best_value + 0.01 * abs(best_value))),
        "gradient_finite_difference_relative_error": gradient_check,
        "_best_vec": best_vec,
    }


def stage_price(min_delta: float, base_dali: float) -> dict[str, Any]:
    """Price the cheapest droppable direction against the MEASURED credit bracket."""
    gap = FRONTIER_S - TARGET_S
    rows = []
    for name, credit_bytes in CREDIT_BYTES.items():
        credit_s = credit_bytes * S_PER_BYTE
        tolerance = break_even_delta_d_pose(credit_bytes, base_dali)
        cost_s = dS_pose(base_dali + max(min_delta, 0.0), base_dali) if min_delta > 0 else 0.0
        rows.append({
            "credit_convention": name,
            "credit_bytes": credit_bytes,
            "rate_credit_S": credit_s,
            "break_even_delta_d_pose": tolerance,
            "achieved_min_delta_d_pose": min_delta,
            "miss_factor": (min_delta / tolerance) if tolerance > 0 else None,
            "pose_cost_S_at_min": cost_s,
            "net_S_at_min": cost_s - credit_s,
            "affordable": bool(min_delta <= tolerance),
            "ceiling_if_direction_were_FREE_S": credit_s,
            "ceiling_fraction_of_gap": credit_s / gap,
        })
    return {"gap_to_target_S": gap, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jc1-retained", type=Path, default=JC1_RETAINED)
    parser.add_argument("--gt-cache-dali", type=Path, default=GT_CACHE_DALI)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path,
                        default=OUT_ROOT / "RA2CRR_PRICED_POSE_NULL.json")
    args = parser.parse_args()

    import torch

    if not args.gt_cache_dali.exists():
        raise PricedNullRefusal(f"authority GT cache missing: {args.gt_cache_dali}")
    gt_payload = args.gt_cache_dali.read_bytes()
    gt_dali = torch.load(io.BytesIO(gt_payload), map_location="cpu")["pose"].double().numpy()
    if gt_dali.shape != (FRAMES, POSE_DIMS):
        raise PricedNullRefusal(f"authority GT shape {gt_dali.shape}")

    jac = _load_pinned(args.jc1_retained, "jacobian_pose6_x_coeff12.float64.npy",
                       (FRAMES, POSE_DIMS, CARRIER_DIM))
    coeff = _load_pinned(args.jc1_retained, "coeff.float64.npy", (FRAMES, CARRIER_DIM))
    generated = _load_pinned(args.jc1_retained, "pose6_generated.float64.npy",
                             (FRAMES, POSE_DIMS))
    gt_pyav = _load_pinned(args.jc1_retained, "pose6_groundtruth.float64.npy",
                           (FRAMES, POSE_DIMS))

    rng = np.random.default_rng(args.seed)
    controls = stage_controls(coeff, jac, generated, gt_dali, gt_pyav)
    base_dali = controls["base_d_pose_authority_gt"]
    calibration = stage_calibration(coeff, jac, generated, gt_dali, base_dali)
    degeneracy = stage_refit_degeneracy(jac, rng)
    minimised = stage_minimise(coeff, jac, generated, gt_dali, args.restarts, args.seed)
    best_vec = minimised.pop("_best_vec")
    pricing = stage_price(minimised["global_min_delta_d_pose"], base_dali)

    residual = generated - gt_dali
    with np.errstate(**_BLAS_FLAG_SUPPRESSION):
        a = coeff @ best_vec
    w = np.einsum("nij,j->ni", jac, best_vec)
    per_pair = (-2.0 * a * np.einsum("ni,ni->n", residual, w)
                + (a ** 2) * (w ** 2).sum(axis=1)) / POSE_DIMS

    retained = {
        "min_direction": retain_array(
            OUT_ROOT / "retained/ra2crr_min_damage_direction.float64.npy", best_vec),
        "per_pair_delta_d_pose_at_min": retain_array(
            OUT_ROOT / "retained/ra2crr_per_pair_delta_at_min.float64.npy", per_pair),
        "authority_gt_pose6": retain_array(
            OUT_ROOT / "retained/ra2crr_authority_gt_pose6.float64.npy", gt_dali),
        "base_residual_authority": retain_array(
            OUT_ROOT / "retained/ra2crr_base_residual_authority.float64.npy", residual),
    }

    report = {
        "schema": "ddm_ra2crr_priced_pose_null_direction.v1",
        "arm": "ddm_ra2_carrier_rank_refit",
        "axis": "[authority-tracking GT, MEASURED 1.00081x vs contest-CUDA]",
        "score_claim": False,
        "promotable": False,
        "measurement_status": "MODEL_QUANTITY_ON_STE_RELAXED_JACOBIAN_NO_NEW_FORWARDS",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "gt_cache_sha256": hashlib.sha256(gt_payload).hexdigest(),
        "frontier_S": FRONTIER_S,
        "controls": controls,
        "calibration": calibration,
        "refit_degeneracy": degeneracy,
        "minimisation": minimised,
        "pricing": pricing,
        "retained_payload": retained,
    }
    _atomic_bytes(args.output, json.dumps(report, indent=2, sort_keys=True).encode())
    print(json.dumps({"output": str(args.output),
                      "base_d_pose_authority": base_dali,
                      "global_min_delta_d_pose": minimised["global_min_delta_d_pose"],
                      "best_structured": minimised["best_structured_direction"],
                      "calibration_ratio_range": [calibration["realised_over_predicted_min"],
                                                  calibration["realised_over_predicted_max"]],
                      "refit_model_vacuous": degeneracy["model_optimal_refit_residual_is_zero"],
                      "affordable_any_convention": any(r["affordable"]
                                                       for r in pricing["rows"])},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
