"""Curvature-spectrum SENSE state via HVP-Lanczos (task #312 Phase B; the D-3/4/5 first
measurement).

The council's 2nd-order costate: the top-k eigenvalues of the through-R seg-loss Hessian are
the local curvature the optimizer is descending. λ_max is SHARPNESS (large = a knife-edge
minimum, small = a flat basin); the anisotropy λ_1/λ_k is how ELONGATED the local bowl is
(large = ill-conditioned, slow along the flat directions); the trace estimate is the total
curvature budget. These are the 2nd-order complement to the 1st-order costate λ = ∂S/∂x.

This module is the REUSABLE, correctness-critical core: a Lanczos iteration and a Hutchinson
trace estimator that operate on a GENERIC symmetric matrix-vector product ``matvec(v) -> Hv``
(numpy). It is proven on known matrices; the MLX HVP adapter (nested grad through R) lives in
the standalone tool + trainer hook that call in. No score claims; [macOS advisory].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXIS_TAG = "[macOS advisory] NON-PROMOTABLE"


@dataclass
class Spectrum:
    top_k: list[float]          # Ritz values (largest algebraic first)
    lambda_max: float
    lambda_min_topk: float      # smallest of the top-k (the k-th eigenvalue)
    trace_estimate: float
    sharpness_ratio: float      # λ_max / |mean(top_k)| — how much λ_max dominates the band
    anisotropy: float           # λ_1 / λ_k over the top-k (elongation of the local bowl)
    negative_curvature: bool    # any Ritz value < 0 => a saddle direction is present
    n_iter: int
    dim: int
    n_matvecs: int

    def to_row(self, *, stage: str, ep: int, k_pairs: int, source: str = "") -> dict:
        return {
            "stage": "curvature_spectrum", "seg_stage": str(stage), "ep": int(ep),
            "k_pairs": int(k_pairs), "dim": int(self.dim),
            "top_k_eigs": [round(float(v), 6) for v in self.top_k],
            "lambda_max": round(float(self.lambda_max), 6),
            "lambda_min_topk": round(float(self.lambda_min_topk), 6),
            "trace_estimate": round(float(self.trace_estimate), 4),
            "sharpness_ratio": round(float(self.sharpness_ratio), 4),
            "anisotropy": round(float(self.anisotropy), 4),
            "negative_curvature": bool(self.negative_curvature),
            "n_iter": int(self.n_iter), "n_matvecs": int(self.n_matvecs),
            "source": str(source), "axis": AXIS_TAG, "score_neutral": True,
        }


def lanczos_tridiag(matvec, dim: int, n_iter: int, *, seed: int = 0,
                    reorth: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Lanczos iteration on a symmetric operator given by ``matvec(v)->Hv``. Returns
    ``(alpha, beta, steps)`` — the tridiagonal diagonal (len m) and off-diagonal (len m-1) and
    the number of steps actually taken (may be < n_iter on early breakdown). Full
    reorthogonalization (``reorth=True``) keeps the small-m Ritz values accurate."""
    rng = np.random.default_rng(seed)
    m = int(min(n_iter, dim))
    v = rng.standard_normal(dim).astype(np.float64)
    v /= np.linalg.norm(v) + 1e-30
    alphas: list[float] = []
    betas: list[float] = []
    V = [v]
    n_mv = 0
    w_prev = np.zeros(dim, dtype=np.float64)
    beta_prev = 0.0
    for _j in range(m):
        w = np.asarray(matvec(V[-1]), dtype=np.float64).ravel()
        n_mv += 1
        alpha = float(np.dot(w, V[-1]))
        w = w - alpha * V[-1] - beta_prev * w_prev
        if reorth:  # full re-orthogonalization against the built basis (numerical stability)
            for u in V:
                w = w - float(np.dot(w, u)) * u
        beta = float(np.linalg.norm(w))
        alphas.append(alpha)
        w_prev = V[-1]
        beta_prev = beta
        if beta < 1e-10:  # invariant subspace found -> stop
            break
        v = w / beta
        betas.append(beta)
        V.append(v)
    return np.array(alphas), np.array(betas), len(alphas)


def ritz_values(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Eigenvalues of the tridiagonal (alpha diag, beta off-diag) — the Ritz values, sorted
    ascending. These approximate the extreme eigenvalues of the full operator."""
    if alpha.size == 0:
        return np.zeros(0)
    T = np.diag(alpha)
    if beta.size:
        b = beta[: alpha.size - 1]
        T = T + np.diag(b, 1) + np.diag(b, -1)
    w = np.linalg.eigvalsh(T)
    return np.sort(w)


def hutchinson_trace(matvec, dim: int, *, n_probes: int = 8, seed: int = 0) -> tuple[float, int]:
    """Hutchinson stochastic trace estimate ``tr(H) ≈ mean_i z_i^T H z_i`` with Rademacher
    probes ``z_i ∈ {±1}``. Returns ``(trace_estimate, n_matvecs)``."""
    rng = np.random.default_rng(seed + 777)
    acc = 0.0
    n_mv = 0
    for _ in range(max(1, n_probes)):
        z = rng.integers(0, 2, size=dim).astype(np.float64) * 2.0 - 1.0
        hz = np.asarray(matvec(z), dtype=np.float64).ravel()
        n_mv += 1
        acc += float(np.dot(z, hz))
    return acc / max(1, n_probes), n_mv


def compute_spectrum(matvec, dim: int, *, k: int = 8, n_iter: int | None = None,
                     seed: int = 0, trace_probes: int = 8) -> Spectrum:
    """Top-k curvature spectrum of a symmetric operator via Lanczos + Hutchinson trace.

    ``k`` top eigenvalues; ``n_iter`` Lanczos steps (default ``max(2k, k+4)`` for Ritz-value
    accuracy). Returns a :class:`Spectrum` with λ_max, anisotropy λ_1/λ_k, sharpness ratio,
    trace estimate, and a negative-curvature (saddle) flag."""
    k = int(min(k, dim))
    m = int(n_iter) if n_iter else int(min(dim, max(2 * k, k + 4)))
    alpha, beta, steps = lanczos_tridiag(matvec, dim, m, seed=seed)
    rv = ritz_values(alpha, beta)  # ascending
    n_mv = steps
    top = rv[::-1][:k]  # largest algebraic first
    top = [float(x) for x in top]
    lam_max = top[0] if top else 0.0
    lam_k = top[-1] if top else 0.0
    tr, tr_mv = hutchinson_trace(matvec, dim, n_probes=trace_probes, seed=seed)
    n_mv += tr_mv
    mean_top = float(np.mean(top)) if top else 0.0
    sharp = lam_max / (abs(mean_top) + 1e-30)
    aniso = lam_max / (lam_k if abs(lam_k) > 1e-30 else np.sign(lam_k) * 1e-30 or 1e-30)
    return Spectrum(top_k=top, lambda_max=lam_max, lambda_min_topk=lam_k, trace_estimate=tr,
                    sharpness_ratio=float(sharp), anisotropy=float(aniso),
                    negative_curvature=bool(rv.size and float(rv[0]) < -1e-8),
                    n_iter=steps, dim=int(dim), n_matvecs=int(n_mv))


# ─────────────────────────── MLX HVP adapter (thin) ─────────────────────────
def make_mlx_hvp(loss_of_vector, x0: np.ndarray):
    """Build a numpy ``matvec(v)->Hv`` for the Hessian of ``loss_of_vector`` (an MLX scalar fn
    of a FLAT mx param vector) at ``x0``, via nested ``mx.grad`` (Hessian-vector product =
    grad of the directional derivative of the gradient — no dense Hessian materialized).

    ``loss_of_vector(mx_vec) -> mx scalar``. The caller flattens the witness params into one
    vector and provides the unflatten inside ``loss_of_vector`` (see the standalone tool)."""
    import mlx.core as mx

    x0m = mx.array(np.asarray(x0, dtype=np.float32))
    grad_fn = mx.grad(loss_of_vector)

    def matvec(v: np.ndarray) -> np.ndarray:
        vm = mx.array(np.asarray(v, dtype=np.float32))

        def gdot(x):
            return mx.sum(grad_fn(x) * vm)  # directional derivative of the gradient

        hv = mx.grad(gdot)(x0m)
        mx.eval(hv)
        return np.asarray(hv, dtype=np.float64)

    return matvec


def mlx_model_hvp(scalar_loss_of_model, model) -> tuple:
    """Build ``(matvec, dim)`` for the Hessian of ``scalar_loss_of_model(model)`` w.r.t. ALL
    of an MLX ``nn.Module``'s parameters, flattened to one vector (deterministic tree order).

    The Hessian is materialized matrix-free (nested ``mx.grad`` on the flat vector). The model's
    parameters are SNAPSHOTTED and RESTORED, so the measurement leaves the module unchanged
    (score-neutral). Used by the standalone tool AND the in-trainer curvature hook so the
    Hessian is always taken on the EXACT live forward — no re-implemented render."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    flat = tree_flatten(model.parameters())      # [(key, mx.array)] deterministic order
    keys = [k for k, _ in flat]
    arrs = [a for _, a in flat]
    shapes = [tuple(a.shape) for a in arrs]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    dim = int(sum(sizes))
    orig = tree_unflatten(list(zip(keys, [mx.array(np.asarray(a)) for a in arrs])))
    x0 = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrs]) if arrs \
        else np.zeros(0, dtype=np.float32)

    def _unflatten_vec(xvec):
        parts = []
        off = 0
        for sz, sh in zip(sizes, shapes):
            parts.append(mx.reshape(xvec[off:off + sz], sh) if sh else mx.reshape(xvec[off:off + sz], ()))
            off += sz
        return tree_unflatten(list(zip(keys, parts)))

    def loss_of_vector(xvec):
        model.update(_unflatten_vec(xvec))   # params become graph fns of xvec -> Hessian flows
        return scalar_loss_of_model(model)

    matvec = make_mlx_hvp(loss_of_vector, x0)

    def guarded_matvec(v: np.ndarray) -> np.ndarray:
        out = matvec(v)
        model.update(orig)   # restore the module after each probe (score-neutral)
        return out

    return guarded_matvec, dim
