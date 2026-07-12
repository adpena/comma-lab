# SPDX-License-Identifier: MIT
"""Streaming closed-form ELM/INR solve for the level-set witness SDF head.

The coordinate trunk, pair codes, FiLM maps, texture head, and palette are frozen.  For hidden
features ``h`` this module fits only the affine SDF readout ``phi = h @ W.T + b``.  It extends
the already-settled in-memory global helper
``lever_b_levelset_generator.fit_out_sdf_to_structured_target`` rather than re-deriving that
solve: the new work is bounded streaming, atomic resume custody, local POU heads, and the
explicit fold back to the current global-head decoder.  It is the portable least-squares seed
for the exact through-R #341 terminal Gauss-Newton finisher; it is not an evaluator or a score
surface.  At zero ridge the 1x1 path is contract-tested against the settled helper.  At positive
ridge this implementation follows the new spec's unregularized-bias normal equation, while the
legacy helper regularizes its augmented intercept.

The local ELM formulation uses rectangular partition-of-unity (POU) weights.  Because the
current decoder owns one global affine SDF head, local predictions are explicitly folded back
to that deployable head with a second streaming ridge solve.  The fold residual remains part of
the receipt: a local POU fit is not claimed to survive the decoder when that residual is large.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Return SHA-256 over the exact file bytes without materializing the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    """Stable JSON encoding used for resumability custody hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def smoothed_ce_logit_targets(
    labels: np.ndarray,
    *,
    n_classes: int,
    smoothing: float,
    temperature: float,
) -> np.ndarray:
    r"""Map categorical labels to finite, centered CE-logit targets.

    For label-smoothed categorical probabilities ``q``, the returned target is

    ``temperature * (log(q) - mean(log(q), class_axis))``.

    Strictly positive smoothing is required.  The exact one-hot/argmax objective has infinite
    logits and no derivative, so silently clipping a zero-smoothing request would hide a change
    in the mathematical target.
    """

    y = np.asarray(labels)
    k = int(n_classes)
    eps = float(smoothing)
    temp = float(temperature)
    if k < 2:
        raise ValueError(f"n_classes must be >=2, got {k}")
    upper = float(k - 1) / float(k)
    if not (0.0 < eps < upper):
        raise ValueError(
            f"smoothing must be strictly between 0 and {(k - 1)}/{k} so the labeled class remains "
            f"the unique target argmax, got {eps}"
        )
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"temperature must be finite and >0, got {temp}")
    if y.size and (not np.issubdtype(y.dtype, np.integer) or int(y.min()) < 0 or int(y.max()) >= k):
        raise ValueError(f"labels must be integer class ids in [0,{k}), got dtype={y.dtype}")

    flat = y.reshape(-1).astype(np.int64, copy=False)
    off = eps / float(k - 1)
    q = np.full((flat.size, k), off, dtype=np.float64)
    if flat.size:
        q[np.arange(flat.size), flat] = 1.0 - eps
    log_q = np.log(q)
    centered = temp * (log_q - log_q.mean(axis=1, keepdims=True))
    return centered.reshape((*y.shape, k))


def _tent_axis(values: np.ndarray, n_cells: int) -> np.ndarray:
    """Cardinal linear B-spline/tent weights on ``[0,1]``."""

    n = int(n_cells)
    if n < 1:
        raise ValueError(f"grid dimensions must be >=1, got {n}")
    if n == 1:
        return np.ones((values.size, 1), dtype=np.float64)
    centers = np.linspace(0.0, 1.0, n, dtype=np.float64)
    spacing = 1.0 / float(n - 1)
    return np.maximum(1.0 - np.abs(values[:, None] - centers[None, :]) / spacing, 0.0)


def rectangular_pou_weights(
    coords_xy: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    coord_min: tuple[float, float] = (-1.0, -1.0),
    coord_max: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Return nonnegative rectangular POU weights in row-major ``(row, col)`` order.

    The witness coordinate convention is ``(x,y) in [-1,1]^2``.  Explicit ``coord_min`` and
    ``coord_max`` keep the helper reusable without guessing another coordinate convention.
    Products of 1-D tent weights are normalized defensively; every returned row sums to one.
    """

    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_xy must have shape (N,2), got {coords.shape}")
    rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
    if rows < 1 or cols < 1:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")
    lo = np.asarray(coord_min, dtype=np.float64)
    hi = np.asarray(coord_max, dtype=np.float64)
    if lo.shape != (2,) or hi.shape != (2,) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("coord_min/coord_max must be finite length-2 tuples")
    if np.any(hi <= lo):
        raise ValueError(f"coord_max must exceed coord_min, got {coord_min}, {coord_max}")
    tol = 32.0 * np.finfo(np.float32).eps
    if coords.size and (np.any(coords < lo - tol) or np.any(coords > hi + tol)):
        raise ValueError("coordinates lie outside the declared normalized coordinate box")
    unit = np.clip((coords - lo) / (hi - lo), 0.0, 1.0)
    wx = _tent_axis(unit[:, 0], cols)
    wy = _tent_axis(unit[:, 1], rows)
    weights = (wy[:, :, None] * wx[:, None, :]).reshape(coords.shape[0], rows * cols)
    row_sum = weights.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise AssertionError("rectangular tent construction produced an uncovered coordinate")
    weights /= row_sum
    return weights


@dataclass(frozen=True)
class RidgeSolveDiagnostics:
    """Rank/conditioning receipt for one regularized normal equation."""

    rank: int
    dimension: int
    condition_number: float
    sample_count: int
    weight_sum: float
    ridge: float


@dataclass(frozen=True)
class PartitionedAffineHeadSolution:
    """Complete target/direct/POU/fold solution for the current single-head receiver.

    ``direct_global_beta`` is deliberately the unregularized target-SSE comparator.  The
    configured ridge applies to local POU heads and (when needed) their global fold.  This
    distinction keeps the receiver claim honest: no globally affine projection can have lower
    target SSE than the direct comparator, even if local regularization is useful upstream.
    """

    local_beta: np.ndarray
    direct_global_beta: np.ndarray
    folded_global_beta: np.ndarray
    direct_global_target_rmse: float
    pou_local_target_rmse: float
    folded_global_target_rmse: float
    fold_vs_local_rmse: float
    direct_global_diagnostics: RidgeSolveDiagnostics
    local_diagnostics: tuple[RidgeSolveDiagnostics, ...]
    fold_diagnostics: RidgeSolveDiagnostics | None
    fold_second_solve_applied: bool


class StreamingRidgeNormalEquations:
    """Streaming ``X^T W X`` / ``X^T W Y`` accumulator with an unregularized bias."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        *,
        ridge: float,
        pinv_rcond: float = 1e-12,
    ) -> None:
        d = int(n_features)
        k = int(n_outputs)
        lam = float(ridge)
        rc = float(pinv_rcond)
        if d < 1 or k < 1:
            raise ValueError(f"n_features/n_outputs must be positive, got {d}/{k}")
        if not np.isfinite(lam) or lam < 0.0:
            raise ValueError(f"ridge must be finite and >=0, got {lam}")
        if not np.isfinite(rc) or rc <= 0.0:
            raise ValueError(f"pinv_rcond must be finite and >0, got {rc}")
        self.n_features = d
        self.n_outputs = k
        self.ridge = lam
        self.pinv_rcond = rc
        self.gram = np.zeros((d + 1, d + 1), dtype=np.float64)
        self.rhs = np.zeros((d + 1, k), dtype=np.float64)
        self.target_square_sum = 0.0
        self.weight_sum = 0.0
        self.sample_count = 0

    def update(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Accumulate one bounded batch; ``features`` never persists inside the object."""

        h = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.n_features:
            raise ValueError(f"features must have shape (N,{self.n_features}), got {h.shape}")
        if y.ndim != 2 or y.shape != (h.shape[0], self.n_outputs):
            raise ValueError(f"targets must have shape ({h.shape[0]},{self.n_outputs}), got {y.shape}")
        if np.any(~np.isfinite(h)) or np.any(~np.isfinite(y)):
            raise ValueError("features and targets must be finite")
        if sample_weight is None:
            w = np.ones(h.shape[0], dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.shape != (h.shape[0],):
                raise ValueError(f"sample_weight must have shape ({h.shape[0]},), got {w.shape}")
            if np.any(~np.isfinite(w)) or np.any(w < 0.0):
                raise ValueError("sample_weight must be finite and nonnegative")
        if h.shape[0] == 0:
            return
        x = np.empty((h.shape[0], self.n_features + 1), dtype=np.float64)
        x[:, :-1] = h
        x[:, -1] = 1.0
        wx = w[:, None] * x
        # macOS Accelerate can emit spurious overflow/divide warnings for finite fp64 matmuls;
        # the explicit postcondition below still refuses a genuinely non-finite accumulation.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.gram += x.T @ wx
            self.rhs += x.T @ (w[:, None] * y)
        if np.any(~np.isfinite(self.gram)) or np.any(~np.isfinite(self.rhs)):
            raise FloatingPointError("normal-equation accumulation produced non-finite values")
        self.target_square_sum += float(np.sum(w[:, None] * y * y, dtype=np.float64))
        self.weight_sum += float(w.sum(dtype=np.float64))
        self.sample_count += int(h.shape[0])

    def regularized_matrix(self) -> np.ndarray:
        matrix = self.gram.copy()
        if self.ridge:
            diag = np.full(self.n_features + 1, self.ridge, dtype=np.float64)
            diag[-1] = 0.0  # bias is deliberately unregularized
            matrix.flat[:: matrix.shape[0] + 1] += diag
        return matrix

    def solve(self) -> tuple[np.ndarray, RidgeSolveDiagnostics]:
        """Return ``beta`` with shape ``(D+1,K)`` using a deterministic pseudoinverse."""

        if self.weight_sum <= 0.0:
            raise ValueError("cannot solve an empty/zero-weight normal equation")
        matrix = self.regularized_matrix()
        # As above, Accelerate may warn inside a finite SVD/matmul.  The explicit finite
        # postcondition is the authority; warnings are not allowed to contaminate receipts.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            beta = np.linalg.pinv(matrix, rcond=self.pinv_rcond, hermitian=True) @ self.rhs
        if np.any(~np.isfinite(beta)):
            raise FloatingPointError("ridge pseudoinverse produced non-finite coefficients")
        rank = int(np.linalg.matrix_rank(matrix, tol=self.pinv_rcond * max(np.linalg.norm(matrix, 2), 1.0)))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            condition = float(np.linalg.cond(matrix))
        return beta, RidgeSolveDiagnostics(
            rank=rank,
            dimension=int(matrix.shape[0]),
            condition_number=condition,
            sample_count=self.sample_count,
            weight_sum=self.weight_sum,
            ridge=self.ridge,
        )

    def residual_rmse(self, beta: np.ndarray) -> float:
        """Derive weighted training RMSE from sufficient statistics without a third stream."""

        b = np.asarray(beta, dtype=np.float64)
        if b.shape != (self.n_features + 1, self.n_outputs):
            raise ValueError(f"beta has shape {b.shape}, expected {(self.n_features + 1, self.n_outputs)}")
        if self.weight_sum <= 0.0:
            raise ValueError("cannot derive residual for an empty normal equation")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sse = self.target_square_sum - 2.0 * float(np.sum(b * self.rhs)) + float(
                np.sum(b * (self.gram @ b))
            )
        scale = max(self.target_square_sum, 1.0)
        if sse < 0.0 and abs(sse) <= 1e-10 * scale:
            sse = 0.0
        if sse < 0.0:
            raise FloatingPointError(f"derived negative residual SSE {sse}")
        return float(np.sqrt(sse / (self.weight_sum * self.n_outputs)))

    def state_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}__gram": self.gram,
            f"{prefix}__rhs": self.rhs,
            f"{prefix}__target_square_sum": np.asarray(self.target_square_sum, np.float64),
            f"{prefix}__weight_sum": np.asarray(self.weight_sum, np.float64),
            f"{prefix}__sample_count": np.asarray(self.sample_count, np.int64),
            f"{prefix}__n_features": np.asarray(self.n_features, np.int64),
            f"{prefix}__n_outputs": np.asarray(self.n_outputs, np.int64),
            f"{prefix}__ridge": np.asarray(self.ridge, np.float64),
            f"{prefix}__pinv_rcond": np.asarray(self.pinv_rcond, np.float64),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray], prefix: str) -> StreamingRidgeNormalEquations:
        obj = cls(
            int(np.asarray(state[f"{prefix}__n_features"])),
            int(np.asarray(state[f"{prefix}__n_outputs"])),
            ridge=float(np.asarray(state[f"{prefix}__ridge"])),
            pinv_rcond=float(np.asarray(state[f"{prefix}__pinv_rcond"])),
        )
        obj.gram[...] = np.asarray(state[f"{prefix}__gram"], dtype=np.float64)
        obj.rhs[...] = np.asarray(state[f"{prefix}__rhs"], dtype=np.float64)
        obj.target_square_sum = float(np.asarray(state[f"{prefix}__target_square_sum"]))
        obj.weight_sum = float(np.asarray(state[f"{prefix}__weight_sum"]))
        obj.sample_count = int(np.asarray(state[f"{prefix}__sample_count"]))
        return obj


class StreamingPartitionedRidge:
    """One streaming affine-head normal equation per rectangular POU subdomain."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        *,
        grid_shape: tuple[int, int],
        ridge: float,
        pinv_rcond: float = 1e-12,
    ) -> None:
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        if self.grid_shape[0] < 1 or self.grid_shape[1] < 1:
            raise ValueError(f"grid_shape must be positive, got {grid_shape}")
        self.parts = [
            StreamingRidgeNormalEquations(
                self.n_features,
                self.n_outputs,
                ridge=ridge,
                pinv_rcond=pinv_rcond,
            )
            for _ in range(self.grid_shape[0] * self.grid_shape[1])
        ]

    def update(self, features: np.ndarray, targets: np.ndarray, coords_xy: np.ndarray) -> None:
        weights = rectangular_pou_weights(coords_xy, grid_shape=self.grid_shape)
        for index, part in enumerate(self.parts):
            part.update(features, targets, weights[:, index])

    def solve(self) -> tuple[np.ndarray, tuple[RidgeSolveDiagnostics, ...]]:
        solved = [part.solve() for part in self.parts]
        return np.stack([item[0] for item in solved]), tuple(item[1] for item in solved)

    def state_dict(self, prefix: str = "local") -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {
            f"{prefix}__grid_shape": np.asarray(self.grid_shape, np.int64),
            f"{prefix}__n_features": np.asarray(self.n_features, np.int64),
            f"{prefix}__n_outputs": np.asarray(self.n_outputs, np.int64),
        }
        for index, part in enumerate(self.parts):
            state.update(part.state_dict(f"{prefix}__part_{index}"))
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray], prefix: str = "local") -> StreamingPartitionedRidge:
        grid = tuple(int(v) for v in np.asarray(state[f"{prefix}__grid_shape"]).tolist())
        first = StreamingRidgeNormalEquations.from_state_dict(state, f"{prefix}__part_0")
        obj = cls(
            int(np.asarray(state[f"{prefix}__n_features"])),
            int(np.asarray(state[f"{prefix}__n_outputs"])),
            grid_shape=(grid[0], grid[1]),
            ridge=first.ridge,
            pinv_rcond=first.pinv_rcond,
        )
        obj.parts[0] = first
        for index in range(1, len(obj.parts)):
            obj.parts[index] = StreamingRidgeNormalEquations.from_state_dict(
                state, f"{prefix}__part_{index}"
            )
        return obj


def closed_form_affine_head_seed(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    pinv_rcond: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, RidgeSolveDiagnostics]:
    """One-shot convenience wrapper returning MLX-oriented ``(weight, bias, diagnostics)``."""

    h = np.asarray(features)
    y = np.asarray(targets)
    if h.ndim != 2 or y.ndim != 2 or h.shape[0] != y.shape[0]:
        raise ValueError(f"features/targets must be aligned rank-2 arrays, got {h.shape}/{y.shape}")
    normal = StreamingRidgeNormalEquations(
        h.shape[1],
        y.shape[1],
        ridge=ridge,
        pinv_rcond=pinv_rcond,
    )
    normal.update(h, y)
    beta, diagnostics = normal.solve()
    return beta[:-1].T.astype(np.float32), beta[-1].astype(np.float32), diagnostics


def partitioned_affine_predict(
    features: np.ndarray,
    coords_xy: np.ndarray,
    local_beta: np.ndarray,
    *,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Evaluate the local affine heads and blend them with the rectangular POU."""

    h = np.asarray(features, dtype=np.float64)
    beta = np.asarray(local_beta, dtype=np.float64)
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if h.ndim != 2:
        raise ValueError(f"features must have shape (N,D), got {h.shape}")
    expected = (rows * cols, h.shape[1] + 1)
    if beta.ndim != 3 or beta.shape[:2] != expected:
        raise ValueError(
            f"features/local_beta incompatible: features={h.shape}, beta={beta.shape}, "
            f"expected beta prefix={expected}"
        )
    x = np.concatenate([h, np.ones((h.shape[0], 1), dtype=np.float64)], axis=1)
    weights = rectangular_pou_weights(coords_xy, grid_shape=(rows, cols))
    local = np.einsum("nd,sdk->nsk", x, beta, optimize=True)
    return np.einsum("ns,nsk->nk", weights, local, optimize=True)


def solve_partitioned_affine_head_with_fold(
    features: np.ndarray,
    targets: np.ndarray,
    coords_xy: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    ridge: float,
    pinv_rcond: float = 1e-12,
) -> PartitionedAffineHeadSolution:
    """Solve the full deterministic POU pipeline and its direct-global comparator.

    The current witness decoder contains exactly one affine ``out_sdf`` receiver.  Therefore:

    * the direct unregularized global solve is the target-SSE optimum in that receiver family;
    * local POU heads are evaluated against the original target before projection;
    * for grids larger than 1x1, the local field is folded through a second ridge solve and
      evaluated both against the local field and the original target;
    * for a 1x1 grid, the already-deployable local head is returned directly.  It is never put
      through a second ridge solve (which would apply ridge shrinkage twice).
    """

    h = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coords = np.asarray(coords_xy, dtype=np.float64)
    if h.ndim != 2 or y.ndim != 2 or h.shape[0] != y.shape[0]:
        raise ValueError(f"features/targets must be aligned rank-2 arrays, got {h.shape}/{y.shape}")
    if coords.shape != (h.shape[0], 2):
        raise ValueError(f"coords_xy must have shape ({h.shape[0]},2), got {coords.shape}")

    local = StreamingPartitionedRidge(
        h.shape[1],
        y.shape[1],
        grid_shape=grid_shape,
        ridge=ridge,
        pinv_rcond=pinv_rcond,
    )
    # Ridge zero is not an omitted constant: this is the defining law of the target-SSE
    # comparator.  It is separate from the configured local/fold regularization.
    direct_target = StreamingRidgeNormalEquations(
        h.shape[1],
        y.shape[1],
        ridge=0.0,
        pinv_rcond=pinv_rcond,
    )
    local.update(h, y, coords)
    direct_target.update(h, y)
    local_beta, local_diagnostics = local.solve()
    direct_beta, direct_diagnostics = direct_target.solve()
    local_prediction = partitioned_affine_predict(h, coords, local_beta, grid_shape=grid_shape)
    pou_local_rmse = float(np.sqrt(np.mean((local_prediction - y) ** 2, dtype=np.float64)))

    if tuple(int(value) for value in grid_shape) == (1, 1):
        folded_beta = np.array(local_beta[0], copy=True)
        fold_rmse = 0.0
        fold_diagnostics = None
        second_solve = False
    else:
        fold = StreamingRidgeNormalEquations(
            h.shape[1],
            y.shape[1],
            ridge=ridge,
            pinv_rcond=pinv_rcond,
        )
        fold.update(h, local_prediction)
        folded_beta, fold_diagnostics = fold.solve()
        fold_rmse = fold.residual_rmse(folded_beta)
        second_solve = True

    direct_rmse = direct_target.residual_rmse(direct_beta)
    folded_target_rmse = direct_target.residual_rmse(folded_beta)
    # The unregularized direct solve is the minimum target SSE over this exact receiver family.
    # A larger tolerance is allowed only for normal-equation/pseudoinverse roundoff.
    tolerance = max(1e-10, 1e-8 * max(direct_rmse, folded_target_rmse, 1.0))
    if folded_target_rmse + tolerance < direct_rmse:
        raise FloatingPointError(
            "POU fold appears to beat the direct-global target-SSE optimum; normal equations "
            "are numerically inconsistent"
        )
    if not np.isfinite(fold_rmse):
        # The explicit prediction is also used above; this guard makes a derived-statistic
        # failure visible rather than allowing a non-finite receipt.
        raise FloatingPointError("fold-vs-local RMSE is non-finite")
    return PartitionedAffineHeadSolution(
        local_beta=np.asarray(local_beta, np.float64),
        direct_global_beta=np.asarray(direct_beta, np.float64),
        folded_global_beta=np.asarray(folded_beta, np.float64),
        direct_global_target_rmse=float(direct_rmse),
        pou_local_target_rmse=float(pou_local_rmse),
        folded_global_target_rmse=float(folded_target_rmse),
        fold_vs_local_rmse=float(fold_rmse),
        direct_global_diagnostics=direct_diagnostics,
        local_diagnostics=local_diagnostics,
        fold_diagnostics=fold_diagnostics,
        fold_second_solve_applied=second_solve,
    )


def extract_levelset_hidden_numpy(
    params: Mapping[str, np.ndarray],
    feats: np.ndarray,
    code_row: np.ndarray,
    *,
    n_hidden: int,
    hidden_dim: int,
    activation: str,
    wire_w0: float,
    wire_s0: float,
    hosc_beta: float,
    hosc_omega: float,
) -> np.ndarray:
    """Mirror the canonical NumPy witness through its last frozen hidden activation.

    Optional ``film_pl.*`` and ``concat_pl.*`` branches are key-detected exactly like
    :func:`tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy`.
    """

    from tac.boundary_math.lever_b_levelset_generator import _act

    p = {key: np.asarray(value, np.float64) for key, value in params.items()}
    x = np.asarray(feats, np.float64)
    code = np.asarray(code_row, np.float64)
    hidden = int(hidden_dim)
    depth = int(n_hidden)
    if x.ndim != 2:
        raise ValueError(f"feats must have shape (N,D), got {x.shape}")
    akw = {"w0": wire_w0, "s0": wire_s0, "beta": hosc_beta, "omega": hosc_omega}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _act(x @ p["in_proj.weight"].T + p["in_proj.bias"], activation, **akw)
        film = (code @ p["film.weight"].T + p["film.bias"]).reshape(depth, 2, hidden)
        has_film_pl = any(key.startswith("film_pl.") for key in p)
        has_concat = any(key.startswith("concat_pl.") for key in p)
        for layer in range(depth):
            scale = 1.0 + film[layer, 0]
            shift = film[layer, 1]
            if has_film_pl:
                pair_film = (
                    code @ p[f"film_pl.{layer}.weight"].T + p[f"film_pl.{layer}.bias"]
                ).reshape(2, hidden)
                scale = scale + pair_film[0]
                shift = shift + pair_film[1]
            pre = (h @ p[f"hidden.{layer}.weight"].T + p[f"hidden.{layer}.bias"]) * scale + shift
            if has_concat:
                pre = pre + code @ p[f"concat_pl.{layer}.weight"].T + p[f"concat_pl.{layer}.bias"]
            h = _act(pre, activation, **akw)
    return np.asarray(h, np.float32)


def atomic_save_npz(path: str | Path, arrays: Mapping[str, np.ndarray], *, compressed: bool = True) -> Path:
    """Write an NPZ with file+containing-directory fsync and atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # The rename itself must survive a crash/power loss.  This is the same post-replace
        # directory-fsync pattern used by tools/profile_segnet_blocks.py.
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_seed_checkpoint_atomic(
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Path:
    """Preserve every source array/metadata scalar while replacing only ``out_sdf``."""

    source = Path(source_checkpoint)
    with np.load(source, allow_pickle=True) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if "out_sdf.weight" not in arrays or "out_sdf.bias" not in arrays:
        raise KeyError("source checkpoint lacks out_sdf.weight/out_sdf.bias")
    new_weight = np.asarray(weight, np.float32)
    new_bias = np.asarray(bias, np.float32)
    if new_weight.shape != arrays["out_sdf.weight"].shape:
        raise ValueError(
            f"weight shape {new_weight.shape} does not match checkpoint {arrays['out_sdf.weight'].shape}"
        )
    if new_bias.shape != arrays["out_sdf.bias"].shape:
        raise ValueError(f"bias shape {new_bias.shape} does not match checkpoint {arrays['out_sdf.bias'].shape}")
    arrays["out_sdf.weight"] = new_weight
    arrays["out_sdf.bias"] = new_bias
    return atomic_save_npz(output_checkpoint, arrays, compressed=False)


def verify_seed_checkpoint_preservation(
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
) -> dict[str, Any]:
    """Fail closed unless every non-SDF array is exactly preserved in the seeded checkpoint."""

    with np.load(source_checkpoint, allow_pickle=True) as source, np.load(
        output_checkpoint, allow_pickle=True
    ) as output:
        source_keys = set(source.files)
        output_keys = set(output.files)
        if source_keys != output_keys:
            raise AssertionError(
                f"seed checkpoint key set changed: missing={sorted(source_keys - output_keys)}, "
                f"added={sorted(output_keys - source_keys)}"
            )
        changed = sorted(key for key in source_keys if not np.array_equal(source[key], output[key]))
    allowed = {"out_sdf.weight", "out_sdf.bias"}
    unauthorized = sorted(set(changed) - allowed)
    if unauthorized:
        raise AssertionError(f"seed checkpoint mutated non-SDF arrays: {unauthorized}")
    return {
        "source_key_count": len(source_keys),
        "non_head_key_count": len(source_keys - allowed),
        "changed_keys": changed,
        "all_non_head_arrays_exact": True,
    }


__all__ = [
    "PartitionedAffineHeadSolution",
    "RidgeSolveDiagnostics",
    "StreamingPartitionedRidge",
    "StreamingRidgeNormalEquations",
    "atomic_save_npz",
    "canonical_json",
    "closed_form_affine_head_seed",
    "extract_levelset_hidden_numpy",
    "partitioned_affine_predict",
    "rectangular_pou_weights",
    "sha256_file",
    "smoothed_ce_logit_targets",
    "solve_partitioned_affine_head_with_fold",
    "verify_seed_checkpoint_preservation",
    "write_seed_checkpoint_atomic",
]
