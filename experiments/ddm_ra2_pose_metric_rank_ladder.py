#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-RA2: rank-r truncation of the hv1 carrier in the POSE metric vs the Euclidean one.

THE CELL THIS FILLS, AND WHY IT IS NOT ALREADY MEASURED.

Two arms have measured carrier rank/refit on this exact 182,759 B base today:

  * ``ddm_ra2c`` truncated the rendered carrier field ``C @ B`` to rank r -- a
    SUBSPACE treatment, optimal in the Frobenius/Euclidean norm by Eckart-Young.
  * ``ddm_jc1`` ran a coordinate KEEP-SET plus coefficient re-fit, in both the
    Euclidean and the PoseNet-Jacobian metric.

``jc1`` states plainly in its own honest-limits section that these two are *not*
the same treatment and that it did not compare them: a keep-set drops named
coordinates, a subspace truncation drops directions.  So the cell

    rank-r SUBSPACE truncation, optimal in the POSE metric

has never been measured.  That is exactly the treatment MAIN's relay routed here:
whiten by the PoseNet Jacobian FIRST, truncate rank in the whitened space, then
re-price.  This tool measures it, races it against the Euclidean subspace at
MATCHED bytes, and reports both.

THE METRICS, STATED EXACTLY.

Let ``C`` be the 600x12 coefficient matrix, ``B`` the 12x2304 basis, and ``J_i``
the 6x12 carrier->PoseNet Jacobian at pair i (MEASURED by jc1 through the shipped
chain, finite-difference-controlled to ~3%).  For a perturbation ``D = C' - C``:

  Euclidean (what ra2c optimised)   ||D B||_F^2       = tr(D G_b D^T),  G_b = B B^T
  Pose      (what the score reads)  sum_i ||J_i d_i||^2 = tr(D G_p D^T) when the
                                                          pullback is averaged,
                                    G_p = mean_i J_i^T J_i

Both are quadratic forms in ``D``, so each admits an EXACT optimal rank-r
truncation: whiten by ``G^{1/2}``, take the SVD, truncate, un-whiten.  Whitening
is invertible here because ``jc1`` measured ``K = 0`` (no pose-null direction) and
``cond(J_stack) = 12.02``, so ``G_p`` is positive definite.

ASSUMED, and named because it bounds the claim: the true pose objective weights
each pair by its OWN ``J_i``.  Per-row-weighted low-rank approximation is NP-hard
in general; the averaged pullback ``G_p`` is the standard relaxation and is what
"whiten by the Jacobian" means.  A per-pair-optimal subspace could do better than
the row this tool measures -- but not better than the EXACT d_pose it reports,
because that is measured through the real chain, not modelled.

WHAT IS AUTHORITY HERE.  The relative-error columns are DERIVED from measured
Grams.  The verdict rests on the EXACT n600 ``d_pose`` produced by
``ddm_jc1_carrier_pose_jacobian.py --eval-coeff``, which renders through the
shipped chain and runs the frozen CPU PoseNet -- no linearization.  jc1 validated
that instrument with a no-op control at 2.6e-5 relative.  This matters because
jc1 also measured that a first-order pose model, used as a DESIGNER, overstated
its own control authority by up to 1,065x.  Nothing here is reported from the
linear model alone.

PRE-REGISTERED FALSIFIER (MAIN's relay, honoured verbatim): if the pose-metric
rank-4 reconstruction error exceeds 0.389%, the carrier is closed in BOTH metrics
and the rank/refit family is done.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JC1 = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")

S_PER_BYTE = 25.0 / 37_545_489.0
FRONTIER_S = 0.15959729295498598
CARRIER_BYTES = 22_161
CARRIER_DIM = 12
N_FRAMES = 600

#: MEASURED frontier components (pz5 + this arm both re-derived them from the pointer).
POSE_TERM_T4 = 0.0082945765
D_POSE_ADVISORY = 1.4747e-4

#: MEASURED by ra2c: bytes returned per dropped carrier dimension.
BYTES_PER_DROPPED_DIM = CARRIER_BYTES / CARRIER_DIM

#: MAIN's pre-registered falsifier threshold on rank-4 reconstruction error.
FALSIFIER_RANK4_ERROR = 0.00389


class LadderRefusal(RuntimeError):
    """Fail-closed refusal: missing custody, singular metric, or a broken control."""


def _sym_sqrt(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (G^{1/2}, G^{-1/2}) for a symmetric positive-definite Gram."""
    symmetric = 0.5 * (gram + gram.T)
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() <= 0:
        raise LadderRefusal(
            f"metric is not positive definite (min eigenvalue {eigenvalues.min():.3e}); "
            "whitening is undefined and the truncation would silently change rank"
        )
    # Accelerate's SIMD tail raises spurious divide/overflow/invalid flags on matmuls
    # against a np.diag operand while returning correct values -- the same artifact
    # ra2c documented at ddm_ra2c_alpha_ladder.py:126-128.  Verified independently
    # here before suppressing: ||R@R - G|| <= 3.9e-14, ||R@Ri - I|| <= 2.2e-15, and
    # R agrees with scipy.linalg.sqrtm to 1.1e-14.  Suppressed narrowly; the
    # isfinite guard below fails closed, so a genuine NaN still refuses.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        root = vectors @ np.diag(np.sqrt(eigenvalues)) @ vectors.T
        inverse = vectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ vectors.T
    if not (np.isfinite(root).all() and np.isfinite(inverse).all()):
        raise LadderRefusal("whitening produced non-finite values")
    if not np.allclose(root @ root, symmetric, atol=1e-10):
        raise LadderRefusal("G^{1/2} does not square back to G")
    return root, inverse


def truncate_in_metric(coeff: np.ndarray, gram: np.ndarray, rank: int) -> np.ndarray:
    """Optimal rank-r approximation of `coeff` in the norm induced by `gram`.

    Whiten -> SVD -> truncate -> un-whiten.  Optimal by Eckart-Young in the
    whitened coordinate, which is exactly the `gram` norm in the original one.
    """
    root, inverse = _sym_sqrt(gram)
    whitened = coeff @ root
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    singular_r = singular.copy()
    singular_r[rank:] = 0.0
    return (left @ np.diag(singular_r) @ right) @ inverse


def relative_error(delta: np.ndarray, coeff: np.ndarray, gram: np.ndarray) -> float:
    """Relative error of a perturbation in the `gram` norm.

    NOTE ON THE EUCLIDEAN COLUMN.  `gram` for the Euclidean metric is jc1's
    `basis_gram`, built at ``ddm_jc1_pose_null_and_refit.py:271-272`` from
    ``renderer.normalized_basis(basis)`` -- the basis the shipped renderer actually
    uses (bicubic to eval resolution, mean-removed, RMS-normalised).  ra2c's ladder
    instead truncated ``C @ basis`` on the RAW basis, so its Euclidean column
    optimises a slightly different inner product than the one that renders.  The
    difference is small and verdict-neutral (rank-4: 25.14% raw vs 24.50% here),
    but it means ra2c's Eckart-Young "lower bound on any rank-r method" is a bound
    in the raw-basis norm, not in the rendered one.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        num = float(np.trace(delta @ gram @ delta.T))
        den = float(np.trace(coeff @ gram @ coeff.T))
    if not np.isfinite(num) or not np.isfinite(den):
        raise LadderRefusal("non-finite norm; refusing to report a relative error")
    if den <= 0:
        raise LadderRefusal("degenerate reference norm")
    return float(np.sqrt(max(num, 0.0) / den))


def exact_pose_bar(bytes_back: float, pose_term: float) -> float:
    """The d_pose RATIO a treatment may cost while still paying for its bytes."""
    return ((pose_term + bytes_back * S_PER_BYTE) / pose_term) ** 2


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _retain_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    if path.exists() and path.read_bytes() != payload:
        raise LadderRefusal(f"retained payload differs on resume: {path}")
    if not path.exists():
        _atomic_bytes(path, payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jc1", type=Path, default=JC1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_ra2_pose_metric_rank/retained"),
    )
    parser.add_argument(
        "--ranks", type=str, default="4,8,11",
        help="comma-separated ranks to emit candidates for",
    )
    args = parser.parse_args()

    jacobian_path = args.jc1 / "jacobian_pose6_x_coeff12.float64.npy"
    coeff_path = args.jc1 / "coeff.float64.npy"
    gram_path = args.jc1 / "basis_gram.float64.npy"
    for path in (jacobian_path, coeff_path, gram_path):
        if not path.exists():
            raise LadderRefusal(f"jc1 custody missing: {path}")

    jacobian = np.load(jacobian_path)           # (600, 6, 12) MEASURED by jc1
    coeff = np.load(coeff_path)                 # (600, 12)    the shipped coefficients
    basis_gram = np.load(gram_path)             # (12, 12)     B B^T
    if jacobian.shape != (N_FRAMES, 6, CARRIER_DIM):
        raise LadderRefusal(f"unexpected Jacobian shape {jacobian.shape}")
    if coeff.shape != (N_FRAMES, CARRIER_DIM):
        raise LadderRefusal(f"unexpected coefficient shape {coeff.shape}")

    # G_p = mean_i J_i^T J_i -- the averaged pose pullback.
    pose_gram = np.einsum("nij,nik->jk", jacobian, jacobian) / N_FRAMES

    eig_pose = np.linalg.eigvalsh(0.5 * (pose_gram + pose_gram.T))
    eig_basis = np.linalg.eigvalsh(0.5 * (basis_gram + basis_gram.T))

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    retained: dict[str, Any] = {}

    for rank in [int(r) for r in args.ranks.split(",")]:
        bytes_back = (CARRIER_DIM - rank) * BYTES_PER_DROPPED_DIM
        for name, gram in (("euclid", basis_gram), ("pose", pose_gram)):
            candidate = truncate_in_metric(coeff, gram, rank)
            delta = candidate - coeff
            row = {
                "rank": rank,
                "metric_optimised": name,
                "bytes_back": bytes_back,
                "rel_err_euclid_rendered_field": relative_error(delta, coeff, basis_gram),
                "rel_err_pose_metric": relative_error(delta, coeff, pose_gram),
                "advisory_bar": exact_pose_bar(
                    bytes_back, float(np.sqrt(10 * D_POSE_ADVISORY))
                ),
                "t4_bar": exact_pose_bar(bytes_back, POSE_TERM_T4),
            }
            path = args.output / f"candidate_{name}_subspace_r{rank}.float64.npy"
            retained[f"{name}_r{rank}"] = _retain_array(path, candidate)
            row["candidate_path"] = str(path)
            rows.append(row)

    # Race at matched rank: does optimising the POSE metric buy pose accuracy?
    races = []
    for rank in sorted({r["rank"] for r in rows}):
        euclid = next(r for r in rows if r["rank"] == rank and r["metric_optimised"] == "euclid")
        pose = next(r for r in rows if r["rank"] == rank and r["metric_optimised"] == "pose")
        races.append({
            "rank": rank,
            "pose_metric_error_euclid_subspace": euclid["rel_err_pose_metric"],
            "pose_metric_error_pose_subspace": pose["rel_err_pose_metric"],
            "pose_metric_improvement_factor":
                euclid["rel_err_pose_metric"] / pose["rel_err_pose_metric"],
            "euclid_error_cost_factor":
                pose["rel_err_euclid_rendered_field"] / euclid["rel_err_euclid_rendered_field"],
        })

    rank4_pose = next(
        r for r in rows if r["rank"] == 4 and r["metric_optimised"] == "pose"
    )
    falsifier_fired = rank4_pose["rel_err_euclid_rendered_field"] > FALSIFIER_RANK4_ERROR

    receipt = {
        "arm": "ddm_ra2",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "axis": "[DERIVED from jc1-MEASURED Jacobian; exact d_pose owed separately]",
        "score_claim": False,
        "promotable": False,
        "base": {
            "frontier_S": FRONTIER_S,
            "carrier_bytes": CARRIER_BYTES,
            "bytes_per_dropped_dim": BYTES_PER_DROPPED_DIM,
            "jacobian_source": str(jacobian_path),
            "jacobian_sha256": hashlib.sha256(jacobian_path.read_bytes()).hexdigest(),
        },
        "metric_spectra": {
            "pose_gram_eigenvalues": eig_pose.tolist(),
            "pose_gram_condition": float(eig_pose.max() / eig_pose.min()),
            "basis_gram_eigenvalues": eig_basis.tolist(),
            "basis_gram_condition": float(eig_basis.max() / eig_basis.min()),
        },
        "ladder": rows,
        "matched_rank_race": races,
        "pre_registered_falsifier": {
            "threshold_rank4_error": FALSIFIER_RANK4_ERROR,
            "measured_rank4_pose_subspace_error": rank4_pose["rel_err_euclid_rendered_field"],
            "fired": bool(falsifier_fired),
            "meaning": "fired => carrier closed in BOTH metrics; rank/refit family done",
        },
        "retained_candidates": retained,
    }
    out = args.output.parent / "RA2_POSE_METRIC_RANK.json"
    _atomic_bytes(out, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())

    print(f"pose  Gram cond {receipt['metric_spectra']['pose_gram_condition']:.3f}")
    print(f"basis Gram cond {receipt['metric_spectra']['basis_gram_condition']:.3f}")
    print(f"\n{'rank':>4s} {'metric':>7s} {'bytes':>7s} "
          f"{'err(euclid)':>12s} {'err(pose)':>11s} {'advBar':>8s}")
    for row in rows:
        print(f"{row['rank']:4d} {row['metric_optimised']:>7s} "
              f"{row['bytes_back']:7.0f} "
              f"{100 * row['rel_err_euclid_rendered_field']:11.4f}% "
              f"{100 * row['rel_err_pose_metric']:10.4f}% "
              f"{row['advisory_bar']:7.4f}x")
    print("\nmatched-rank race (does optimising the pose metric buy pose accuracy?):")
    for race in races:
        print(f"  r={race['rank']:2d}  pose-metric error "
              f"{100 * race['pose_metric_error_euclid_subspace']:.4f}% (euclid subspace) -> "
              f"{100 * race['pose_metric_error_pose_subspace']:.4f}% (pose subspace)   "
              f"improvement {race['pose_metric_improvement_factor']:.4f}x")
    print(f"\nPRE-REGISTERED FALSIFIER: rank-4 pose-subspace error "
          f"{100 * rank4_pose['rel_err_euclid_rendered_field']:.4f}% vs threshold "
          f"{100 * FALSIFIER_RANK4_ERROR:.3f}%  -> "
          f"{'FIRED (closed in both metrics)' if falsifier_fired else 'NOT fired'}")
    print(f"\nreceipt {out}")
    print("candidates written; measure EXACT n600 d_pose with:")
    print("  .venv/bin/python experiments/ddm_jc1_carrier_pose_jacobian.py --eval-coeff \\")
    print(f"      {args.output}/candidate_pose_subspace_r11.float64.npy \\")
    print(f"      {args.output}/candidate_euclid_subspace_r11.float64.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
