#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-RA3: the one untried cell -- subspace projection PLUS a trust-regioned per-pair
re-fit against the REALIZED pose objective, accepted by measurement and never by model.

THE CELL, AND WHY IT IS THE ONLY ONE LEFT.

Three arms priced the hv1 pose carrier (22,161 B, 12.1% of the archive) on 2026-08-16 and
between them measured a 2x2 of design choices:

                          Euclidean metric      Pose metric
    keep-set   (jc1)          235.3x              238.9x
    subspace   (ra2)           28.616x             15.211x

against a bar of 1.0651x (advisory) / 1.3185x (T4 ratio-transfer).  The treatment axis is
worth 15.7x; the metric axis is worth 1.88x but ONLY inside the subspace row.  Reading the
diagonal made the family look dead at 221x when the true miss is 11.5x.

Both surviving cells optimise the WRONG functional, in opposite directions:

  * ``jc1``  minimised the realised residual ``|| r_i + J_i d_i ||^2`` -- the right
    objective -- but let an unbounded linear model DESIGN the step.  Measured error of the
    model on its own designs: up to 1,065x.  The candidates it built were catastrophic.
  * ``ra2``  minimised the PERTURBATION ``|| J_i d_i ||^2`` and ignored the base residual
    ``r_i`` entirely.  Safe, but it throws away the one thing that can help: the carrier's
    existing pose error is a target the re-fit could partly cancel.

This tool measures the combination: the pose-optimal rank-r SUBSPACE (ra2's treatment) as
the trust-region CENTRE, plus a per-pair correction that minimises the realised objective
under an explicit trust region, with every step ACCEPTED ONLY ON REALISED MEASUREMENT.

THE MATH, STATED EXACTLY (and this is why the centring matters).

Let ``G_p = mean_i J_i^T J_i`` be the averaged pose pullback, ``R = G_p^{1/2}``, and
``y_i = R c_i`` the whitened coefficients.  SVD ``Y = U S V^T``; ``V_r`` (12 x r) spans the
pose-optimal rank-r row space.  Split ``y_i = V_r z_i^proj + y_i^perp``.  A candidate stores
``z_i in R^r`` and renders ``c_i' = R^{-1} V_r z_i``.  Writing ``u = z_i - z_i^proj`` for the
correction ON TOP of ra2's projection, and ``Jh_i = J_i R^{-1}`` for the whitened Jacobian:

    d_i        = c_i' - c_i          = R^{-1} ( V_r u - y_i^perp )
    J_i d_i    = Jh_i ( V_r u - y_i^perp )
    || d_i ||_{G_p}^2                = || u ||^2 + || y_i^perp ||^2      (V_r orthonormal,
                                                                          V_r u _|_ y_i^perp)

so the trust-regioned problem separates cleanly into an r-dimensional ridge regression:

    minimise_u  || b_i + A_i u ||^2  +  lambda_i || u ||^2
    A_i = Jh_i V_r  (6 x r)      b_i = r_i - Jh_i y_i^perp  (6,)
    ==> ( A_i^T A_i + lambda_i I ) u = -A_i^T b_i

``lambda_i -> infinity`` gives ``u = 0``, i.e. EXACTLY ra2's projection.  ``lambda_i -> 0``
gives the minimum-norm first-order-residual-zeroing step, i.e. EXACTLY jc1's failure mode.
The path between them is the Levenberg trust region, and it is anchored at the incumbent --
which is the structural reason realised acceptance can never return a worse row than ra2's.

``lambda_i = mu * mean(eig(A_i^T A_i))`` makes the grid dimensionless and comparable across
pairs whose Jacobians differ in scale by an order of magnitude.

MODEL-PROPOSED, REALISED-ACCEPTED -- the campaign law this arm exists to honour.
Every ``u_i(mu)`` is a PROPOSAL.  For each pair and each ``mu`` the tool renders frame_0
through the shipped chain with hard quantization and runs the frozen CPU PoseNet, then keeps
the ``mu`` whose MEASURED per-pair squared residual is smallest, with ``mu = infinity`` (the
projection) always in the candidate set.  Shrink-on-ascent is therefore exhaustive rather
than greedy: a pair whose every proposal ascends simply keeps the incumbent step.  No number
in the verdict comes from the linear model.

INSTRUMENT.  ``ddm_jc1_carrier_pose_jacobian.py`` is imported wholesale -- same render, same
sparse frame-0 selector, same upstream PoseNet preprocess, same frozen weights, same retained
ground truth, same 600 pairs.  That is ra2's instrument, unchanged, because the verdict is a
RATIO against ra2's 15.211x and a cross-instrument ratio is not a ratio.  The advisory pose
axis is ~18-21x optimistic against contest-CUDA (rn1, 2026-08-16); every number here is
therefore ADVISORY and within-axis only, and the incumbent is re-measured in this same run.

Axis: [macOS-CPU advisory]. score_claim=false, promotable=false. No Modal, no paid GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JC1_SOURCE = REPO / "experiments/ddm_jc1_carrier_pose_jacobian.py"
JC1_PAYLOAD = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")
RA2_PAYLOAD = Path("/Volumes/APDataStore/pact/ddm_ra2_pose_metric_rank/retained")
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_ra3/retained")

DEFAULT_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/archive.zip"
)
DEFAULT_BASE_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/0.raw"
)
DEFAULT_UPSTREAM = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")

N_FRAMES = 600
CARRIER_DIM = 12
POSE_ROWS = 6
CAMERA_H, CAMERA_W = 874, 1164

S_PER_BYTE = 25.0 / 37_545_489.0
CARRIER_BYTES = 22_161
BYTES_PER_DROPPED_DIM = CARRIER_BYTES / CARRIER_DIM

#: MEASURED base rows on this instrument (jc1 producer, re-derived by ra2).
BASE_D_POSE = 0.00014747
BASE_D_SEG = 0.00042714
#: MEASURED frontier pose term on the contest-CUDA axis (for the ratio-transfer column).
POSE_TERM_T4 = 0.0082945765

#: MEASURED by ra2 on this exact instrument -- the incumbent this arm must beat.
RA2_INCUMBENT_R11_D_POSE = 0.0022432311489210995

#: Shipped coefficient codes are 12-bit (ddm_ra2_cpr1_entropy_headroom.py:455).
STORAGE_CODE_BITS = 12

AXIS = "[macOS-CPU advisory]"


class RefitRefusal(RuntimeError):
    """Fail-closed refusal: broken custody, singular metric, or a failed control."""


# --------------------------------------------------------------------------- #
# custody helpers
# --------------------------------------------------------------------------- #
def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def retain_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Persist a materialised payload with its sha256 and byte count.

    ALWAYS KEEP THE PAYLOAD (P0): a candidate coefficient matrix that existed in memory
    is written to disk before any scalar derived from it is reported.
    """
    payload = _npy_bytes(array)
    _atomic_bytes(path, payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_jc1_module():
    """Import the jc1 producer wholesale -- same chain, same controls, no re-implementation."""
    spec = importlib.util.spec_from_file_location("ddm_jc1_producer", JC1_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_jc1_producer"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# design phase -- cheap, DERIVED, every step is a PROPOSAL
# --------------------------------------------------------------------------- #
def sym_sqrt(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (G^{1/2}, G^{-1/2}) for a symmetric positive-definite Gram.

    Reproduces ``ddm_ra2_pose_metric_rank_ladder._sym_sqrt`` including its narrowly
    scoped errstate suppression (Accelerate raises spurious SIMD-tail flags on matmuls
    against a np.diag operand while returning correct values); the isfinite and
    square-back guards below still fail closed on a genuine NaN.
    """
    symmetric = 0.5 * (gram + gram.T)
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() <= 0:
        raise RefitRefusal(
            f"pose metric is not positive definite (min eig {eigenvalues.min():.3e}); "
            "whitening is undefined"
        )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        root = vectors @ np.diag(np.sqrt(eigenvalues)) @ vectors.T
        inverse = vectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ vectors.T
    if not (np.isfinite(root).all() and np.isfinite(inverse).all()):
        raise RefitRefusal("whitening produced non-finite values")
    if not np.allclose(root @ root, symmetric, atol=1e-10):
        raise RefitRefusal("G^{1/2} does not square back to G")
    return root, inverse


class SubspaceGeometry:
    """The rank-r pose-optimal subspace and everything the trust region needs on it."""

    def __init__(self, coeff: np.ndarray, jacobian: np.ndarray, rank: int) -> None:
        self.rank = rank
        self.coeff = coeff
        pose_gram = np.einsum("nij,nik->jk", jacobian, jacobian) / coeff.shape[0]
        self.root, self.inverse = sym_sqrt(pose_gram)
        # Same narrowly scoped Accelerate suppression as ``coeff_from_z``; the guard
        # below plus main's centre control are what actually verify these values.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            whitened = coeff @ self.root                       # y_i^T rows
            _, _, right = np.linalg.svd(whitened, full_matrices=False)
            self.v_r = right[:rank].T                          # (12, r) orthonormal cols
            self.z_proj = whitened @ self.v_r                  # (n, r)
            self.y_perp = whitened - self.z_proj @ self.v_r.T  # (n, 12)
            self.jac_white = np.einsum("nij,jk->nik", jacobian, self.inverse)  # (n,6,12)
            self.a_mat = np.einsum("nij,jr->nir", self.jac_white, self.v_r)    # (n,6,r)
        for name, block in (("z_proj", self.z_proj), ("y_perp", self.y_perp),
                            ("a_mat", self.a_mat)):
            if not np.isfinite(block).all():
                raise RefitRefusal(f"whitened geometry produced non-finite {name}")
        orthogonality = float(np.abs(self.v_r.T @ self.v_r - np.eye(rank)).max())
        if orthogonality > 1e-10:
            raise RefitRefusal(f"subspace basis is not orthonormal ({orthogonality:.3e})")
        self.orthogonality_residual = orthogonality

    def projection_coeff(self) -> np.ndarray:
        """ra2's rank-r pose-metric projection -- the trust-region centre."""
        return self.coeff_from_z(self.z_proj)

    def coeff_from_z(self, z: np.ndarray) -> np.ndarray:
        """Un-whiten a stored r-vector back to shipped carrier coordinates.

        The errstate suppression is the same Accelerate SIMD-tail artifact ra2c and ra2
        documented: matmuls against the eigen-reconstructed inverse raise spurious
        divide/overflow/invalid flags while returning correct values.  VERIFIED here
        rather than assumed -- ``main``'s centre control compares this exact code path
        against ra2's independently retained projection and refuses above 1e-12
        (MEASURED agreement: 6.3e-15).  The isfinite guard still fails closed on a
        genuine NaN.
        """
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = (z @ self.v_r.T) @ self.inverse
        if not np.isfinite(out).all():
            raise RefitRefusal("un-whitening produced non-finite carrier coefficients")
        return out

    def solve_correction(self, residual: np.ndarray, mu: float) -> np.ndarray:
        """PROPOSE the trust-regioned correction u_i(mu). Never authority on its own."""
        b = residual - np.einsum("nij,nj->ni", self.jac_white, self.y_perp)   # (n, 6)
        gram = np.einsum("nir,nis->nrs", self.a_mat, self.a_mat)             # (n, r, r)
        scale = np.trace(gram, axis1=1, axis2=2) / self.rank                 # (n,)
        damped = gram + (mu * scale)[:, None, None] * np.eye(self.rank)[None]
        rhs = -np.einsum("nir,ni->nr", self.a_mat, b)
        return np.linalg.solve(damped, rhs)

    def step_norms(self, z: np.ndarray) -> np.ndarray:
        """||d_i||_{G_p} for a stored z -- the pose-metric step length, per pair."""
        correction = z - self.z_proj
        return np.sqrt((correction ** 2).sum(axis=1) + (self.y_perp ** 2).sum(axis=1))


def quantize_z(z: np.ndarray, code_bits: int = STORAGE_CODE_BITS) -> tuple[np.ndarray, np.ndarray]:
    """Round the stored r-vector onto a per-dimension grid at the SHIPPED code width.

    The shipped format stores 12-bit codes times a per-dimension float32 scale
    (``coeff = codes * coeff_scales``, ddm_ra2b:151).  A real rank-r carrier would store
    ``r`` codes of the same width, so matching the width is the like-for-like quantization
    tax.  Reported as a PROXY for the storage format, which does not exist yet -- it prices
    the precision, not the container.
    """
    levels = float(2 ** (code_bits - 1) - 1)
    scale = np.abs(z).max(axis=0) / levels
    scale[scale <= 0] = 1.0
    return np.round(z / scale[None]) * scale[None], scale


def exact_pose_bar(bytes_back: float, pose_term: float) -> float:
    """The d_pose RATIO a treatment may cost while still paying for its bytes."""
    return ((pose_term + bytes_back * S_PER_BYTE) / pose_term) ** 2


# --------------------------------------------------------------------------- #
# realised phase -- EXACT forward, the only authority in this file
# --------------------------------------------------------------------------- #
class RealisedEvaluator:
    """Render + frozen-PoseNet forward for many candidate rows at one pair.

    Wraps jc1's chain verbatim.  Batching the mu-sweep per pair amortises the frame_1
    load and the selector lookup; the PoseNet forward, which dominates, is unchanged.
    """

    def __init__(self, jc1, args) -> None:
        import torch
        import torch.nn.functional as F

        from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

        self.torch = torch
        self.F = F
        self.yuv6 = differentiable_rgb_to_yuv6

        archive = Path(args.archive)
        archive_sha = jc1.sha256_file(archive)
        if archive_sha != jc1.ARCHIVE_SHA or archive.stat().st_size != jc1.ARCHIVE_BYTES:
            raise RefitRefusal(
                "CUSTODY REFUSED: archive sha/bytes differ from the pinned frontier\n"
                f"  got  {archive_sha} / {archive.stat().st_size}"
            )
        base_raw = Path(args.base_raw)
        expected = N_FRAMES * 2 * CAMERA_H * CAMERA_W * 3
        if base_raw.stat().st_size != expected:
            raise RefitRefusal(f"base render is {base_raw.stat().st_size} B, expected {expected}")

        ra2b = jc1.load_ra2b()
        chain = ra2b.load_chain()
        renderer = chain[0]
        basis, _coeff, selector_blob, _prov = ra2b.decode_carrier(archive, chain)
        self.normalized_basis = renderer.normalized_basis(basis).detach()

        import runtime.frame0_selector as selector_module

        self.selector_module = selector_module
        self.modes = self.sel_choices = None
        if selector_blob is not None:
            self.modes, self.sel_choices = chain[5](selector_blob)

        self.posenet = jc1.build_posenet(torch, Path(args.upstream))
        self.raw = np.memmap(
            base_raw, dtype=np.uint8, mode="r",
            shape=(2 * N_FRAMES, CAMERA_H, CAMERA_W, 3),
        )
        self.jc1 = jc1
        self.forwards = 0

    def pose6_many(self, pair_id: int, rows: np.ndarray) -> np.ndarray:
        """EXACT pose6 for each candidate coefficient row at this pair. (m, 6)."""
        torch = self.torch
        frame1 = torch.from_numpy(np.asarray(self.raw[2 * pair_id + 1]).copy())
        frame1 = frame1.permute(2, 0, 1)[None].float()
        out = np.zeros((rows.shape[0], POSE_ROWS), dtype=np.float64)
        with torch.no_grad():
            for slot in range(rows.shape[0]):
                row = torch.from_numpy(rows[slot : slot + 1].astype(np.float64)).float()
                frame0 = self.jc1.render_frame0_differentiable(
                    torch, self.F, self.normalized_basis, row, ste=False
                )
                if self.modes is not None:
                    frame0 = self.jc1.apply_pixel_mode_differentiable(
                        torch, frame0, self.modes[int(self.sel_choices[pair_id])],
                        self.selector_module,
                    )
                frame0 = frame0.to(torch.uint8).float()
                out[slot] = self.jc1.posenet_pose6(
                    torch, self.F, self.posenet, self.yuv6, frame0, frame1
                ).numpy()[0]
                self.forwards += 1
        return out


def d_pose_from_pose6(pose: np.ndarray, pose_gt: np.ndarray) -> float:
    """upstream's pose distortion, recomputed from components (never a rounded field)."""
    return float(((pose - pose_gt) ** 2).sum(axis=1).mean() / POSE_ROWS)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jc1-payload", type=Path, default=JC1_PAYLOAD)
    parser.add_argument("--ra2-payload", type=Path, default=RA2_PAYLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--base-raw", type=Path, default=DEFAULT_BASE_RAW)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument(
        "--mu", type=str, default="1e2,3e1,1e1,3e0,1e0,3e-1,1e-1,1e-2",
        help="dimensionless Levenberg grid; mu=infinity (the projection) is always included",
    )
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--design-only", action="store_true",
                        help="emit proposals and DERIVED columns, run no forward")
    args = parser.parse_args()

    payload = Path(args.jc1_payload)
    jacobian = np.load(payload / "jacobian_pose6_x_coeff12.float64.npy")
    coeff = np.load(payload / "coeff.float64.npy")
    pose_gen = np.load(payload / "pose6_generated.float64.npy")
    pose_gt = np.load(payload / "pose6_groundtruth.float64.npy")
    if jacobian.shape != (N_FRAMES, POSE_ROWS, CARRIER_DIM):
        raise RefitRefusal(f"unexpected Jacobian shape {jacobian.shape}")
    if coeff.shape != (N_FRAMES, CARRIER_DIM):
        raise RefitRefusal(f"unexpected coefficient shape {coeff.shape}")
    residual = pose_gen - pose_gt

    base_rebuilt = d_pose_from_pose6(pose_gen, pose_gt)
    if abs(base_rebuilt - BASE_D_POSE) / BASE_D_POSE > 1e-3:
        raise RefitRefusal(
            f"base d_pose rebuilt from retained residuals is {base_rebuilt:.10f}, "
            f"expected {BASE_D_POSE}"
        )

    geometry = SubspaceGeometry(coeff, jacobian, args.rank)

    # CONTROL: the projection this tool builds must be byte-identical to ra2's retained
    # candidate, or the trust region is not centred where the comparison assumes.
    projection = geometry.projection_coeff()
    ra2_candidate_path = Path(args.ra2_payload) / f"candidate_pose_subspace_r{args.rank}.float64.npy"
    centre_control: dict[str, Any] = {"available": ra2_candidate_path.exists()}
    if centre_control["available"]:
        ra2_candidate = np.load(ra2_candidate_path)
        max_abs = float(np.abs(projection - ra2_candidate).max())
        centre_control.update({
            "ra2_candidate": str(ra2_candidate_path),
            "max_abs_difference": max_abs,
            "byte_identical_npy": _npy_bytes(projection) == ra2_candidate_path.read_bytes(),
            "passed": max_abs <= 1e-12,
        })
        if not centre_control["passed"]:
            raise RefitRefusal(
                f"trust-region centre differs from ra2's retained projection by {max_abs:.3e}; "
                "the ratio against 15.211x would not be within-treatment"
            )

    mus = [float(v) for v in args.mu.split(",")]
    bytes_back = (CARRIER_DIM - args.rank) * BYTES_PER_DROPPED_DIM
    advisory_bar = exact_pose_bar(bytes_back, float(np.sqrt(10 * BASE_D_POSE)))
    t4_bar = exact_pose_bar(bytes_back, POSE_TERM_T4)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- design phase: PROPOSALS only ---------------------------------------
    z_stack = [geometry.z_proj]                       # mu = infinity, the incumbent
    labels = ["projection"]
    proposals: list[dict[str, Any]] = []
    for mu in mus:
        z = geometry.z_proj + geometry.solve_correction(residual, mu)
        z_stack.append(z)
        labels.append(f"mu{mu:g}")
        step = geometry.step_norms(z)
        incumbent_step = geometry.step_norms(geometry.z_proj)
        proposals.append({
            "mu": mu,
            "correction_over_incumbent_step_median":
                float(np.median(step / np.maximum(incumbent_step, 1e-300))),
            "step_rms_relative_to_coeff":
                float(np.sqrt(((geometry.coeff_from_z(z) - coeff) ** 2).sum(axis=1).mean())
                      / np.sqrt((coeff ** 2).sum(axis=1).mean())),
            "first_order_predicted_d_pose": float(
                (((residual + np.einsum(
                    "nij,nj->ni", jacobian, geometry.coeff_from_z(z) - coeff)) ** 2)
                 .sum(axis=1).mean()) / POSE_ROWS
            ),
        })

    candidate_rows = np.stack([geometry.coeff_from_z(z) for z in z_stack], axis=1)  # (n, m, 12)
    retained: dict[str, Any] = {}
    for slot, label in enumerate(labels):
        retained[label] = retain_array(
            out / f"ra3_r{args.rank}_{label}.float64.npy", candidate_rows[:, slot, :]
        )

    print(f"rank {args.rank}  bytes_back {bytes_back:.0f}  "
          f"advisory bar {advisory_bar:.4f}x  T4 bar {t4_bar:.4f}x")
    print(f"{'mu':>8s} {'step/coeff':>11s} {'step/incumb':>12s} {'1st-order d_pose':>17s}")
    for row in proposals:
        print(f"{row['mu']:8.3g} {row['step_rms_relative_to_coeff']:11.4f} "
              f"{row['correction_over_incumbent_step_median']:12.4f} "
              f"{row['first_order_predicted_d_pose']:17.6e}")

    if args.design_only:
        _atomic_bytes(
            out.parent / f"RA3_DESIGN_r{args.rank}.json",
            (json.dumps({
                "schema": "ddm_ra3_trust_region_design.v1", "axis": AXIS,
                "score_claim": False, "promotable": False,
                "measurement_status": "DERIVED_PROPOSALS_ONLY_NO_FORWARD",
                "rank": args.rank, "bytes_back": bytes_back,
                "advisory_bar": advisory_bar, "t4_bar": t4_bar,
                "centre_control": centre_control, "proposals": proposals,
                "retained": retained,
            }, indent=2, sort_keys=True) + "\n").encode(),
        )
        return 0

    # ---- realised phase: EXACT forward, per pair, per mu --------------------
    jc1 = load_jc1_module()
    evaluator = RealisedEvaluator(jc1, args)
    n_cand = candidate_rows.shape[1]
    pose_all = np.zeros((N_FRAMES, n_cand, POSE_ROWS), dtype=np.float64)
    started = time.time()
    for pair_id in range(N_FRAMES):
        pose_all[pair_id] = evaluator.pose6_many(pair_id, candidate_rows[pair_id])
        if (pair_id + 1) % args.report_every == 0:
            elapsed = time.time() - started
            print(f"  realised {pair_id + 1}/{N_FRAMES}  {elapsed:.0f}s  "
                  f"eta {elapsed / (pair_id + 1) * (N_FRAMES - pair_id - 1):.0f}s", flush=True)
    retained["pose6_all_candidates"] = retain_array(
        out / f"ra3_r{args.rank}_pose6_by_mu.float64.npy", pose_all
    )

    per_pair_sq = ((pose_all - pose_gt[:, None, :]) ** 2).sum(axis=2)     # (n, m)
    per_candidate = [
        {"label": labels[slot],
         "d_pose_measured": float(per_pair_sq[:, slot].mean() / POSE_ROWS),
         "ratio_vs_base": float(per_pair_sq[:, slot].mean() / POSE_ROWS / BASE_D_POSE)}
        for slot in range(n_cand)
    ]

    # REALISED ACCEPTANCE. The incumbent (slot 0) is always in the set, so a pair whose
    # every proposal ascends keeps its incumbent step -- shrink-on-ascent, exhaustively.
    accepted_slot = per_pair_sq.argmin(axis=1)
    accepted_z = np.stack([z_stack[accepted_slot[i]][i] for i in range(N_FRAMES)])
    accepted_coeff = geometry.coeff_from_z(accepted_z)
    accepted_pose = pose_all[np.arange(N_FRAMES), accepted_slot]
    accepted_d_pose = d_pose_from_pose6(accepted_pose, pose_gt)
    retained["accepted"] = retain_array(
        out / f"ra3_r{args.rank}_accepted.float64.npy", accepted_coeff
    )
    retained["accepted_slot"] = retain_array(
        out / f"ra3_r{args.rank}_accepted_slot.int32.npy", accepted_slot.astype(np.int32)
    )

    # ---- quantization tax, measured on BOTH the incumbent and the accepted row ----
    quant_rows = []
    for label, z in (("projection", geometry.z_proj), ("accepted", accepted_z)):
        z_q, scale = quantize_z(z)
        cand_q = geometry.coeff_from_z(z_q)
        retained[f"{label}_quantized"] = retain_array(
            out / f"ra3_r{args.rank}_{label}_q{STORAGE_CODE_BITS}.float64.npy", cand_q
        )
        pose_q = np.zeros((N_FRAMES, POSE_ROWS), dtype=np.float64)
        for pair_id in range(N_FRAMES):
            pose_q[pair_id] = evaluator.pose6_many(pair_id, cand_q[pair_id : pair_id + 1])[0]
        d_q = d_pose_from_pose6(pose_q, pose_gt)
        quant_rows.append({
            "label": label, "code_bits": STORAGE_CODE_BITS,
            "grid_scale_per_dim": scale.tolist(),
            "d_pose_measured": d_q, "ratio_vs_base": d_q / BASE_D_POSE,
        })
        retained[f"pose6_{label}_quantized"] = retain_array(
            out / f"ra3_r{args.rank}_pose6_{label}_q{STORAGE_CODE_BITS}.float64.npy", pose_q
        )
        print(f"  quantized {label}: d_pose {d_q:.8f}  ratio {d_q / BASE_D_POSE:.4f}x", flush=True)

    incumbent_measured = next(
        r for r in per_candidate if r["label"] == "projection"
    )["d_pose_measured"]
    receipt = {
        "arm": "ddm_ra3",
        "schema": "ddm_ra3_trust_region_refit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "measurement_status": "EXACT_FORWARD_NO_LINEARIZATION_REALISED_ACCEPTANCE",
        "instrument": {
            "source": str(JC1_SOURCE),
            "note": "ra2's instrument unchanged; the incumbent is re-measured in this run "
                    "so every ratio is within-axis",
            "posenet_forwards": evaluator.forwards,
        },
        "rank": args.rank,
        "bytes_back": bytes_back,
        "advisory_bar": advisory_bar,
        "t4_bar_ratio_transfer": t4_bar,
        "base_d_pose": BASE_D_POSE,
        "base_d_pose_rebuilt": base_rebuilt,
        "base_d_seg": BASE_D_SEG,
        "d_seg_note": "invariant to frame_0 carrier edits; MEASURED identically on 4 treatments",
        "centre_control": centre_control,
        "ra2_incumbent_of_record": RA2_INCUMBENT_R11_D_POSE,
        "incumbent_reproduction_relative_error":
            abs(incumbent_measured - RA2_INCUMBENT_R11_D_POSE) / RA2_INCUMBENT_R11_D_POSE,
        "proposals_derived": proposals,
        "per_candidate_measured": per_candidate,
        "accepted": {
            "d_pose_measured": accepted_d_pose,
            "ratio_vs_base": accepted_d_pose / BASE_D_POSE,
            "improvement_over_incumbent": incumbent_measured / accepted_d_pose,
            "advisory_miss": accepted_d_pose / BASE_D_POSE / advisory_bar,
            "t4_miss_ratio_transfer": accepted_d_pose / BASE_D_POSE / t4_bar,
            "pairs_improved": int((accepted_slot != 0).sum()),
            "slot_histogram": {labels[s]: int((accepted_slot == s).sum())
                               for s in range(n_cand)},
        },
        "quantization_tax": quant_rows,
        "retained": retained,
    }
    _atomic_bytes(
        out.parent / f"RA3_TRUST_REGION_REFIT_r{args.rank}.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )

    print("\n--- MEASURED, exact n600, within-axis ---")
    for row in per_candidate:
        print(f"  {row['label']:>12s}  d_pose {row['d_pose_measured']:.8f}  "
              f"{row['ratio_vs_base']:9.4f}x")
    print(f"  {'ACCEPTED':>12s}  d_pose {accepted_d_pose:.8f}  "
          f"{accepted_d_pose / BASE_D_POSE:9.4f}x   "
          f"improvement over incumbent {incumbent_measured / accepted_d_pose:.4f}x")
    print(f"  advisory bar {advisory_bar:.4f}x -> miss "
          f"{accepted_d_pose / BASE_D_POSE / advisory_bar:.2f}x")
    print(f"  T4 bar (ratio-transfer) {t4_bar:.4f}x -> miss "
          f"{accepted_d_pose / BASE_D_POSE / t4_bar:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
