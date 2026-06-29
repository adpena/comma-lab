# SPDX-License-Identifier: MIT
"""MD-Decoupling (Magnitude-Direction Decoupling) optimizer for MLX.

Reference: Hägele, Hernández-Cano, Kosson, Jaggi, *"Improving Neural Network
Training by Decoupling the Magnitude and Direction of Weight Vectors"*
(arXiv:2606.25971, EPFL MLO, Jun 2026) + the companion blog
``haeggee.github.io/posts/magnitude-direction-decoupling``.

WHY THIS EXISTS (the re-audit root-cause fix, see
``.omx/research/reaudit_refounding_and_md_decoupling_20260626.md``): the witness
"capacity walls" (R2: 0 of 11 proven fundamental) were OPTIMIZER pathologies —
early-step collapse, no-LR-transfer-across-width (muon_lr-150x), and warmup
dependence. MD-Decoupling addresses ALL three principledly: the relative update
is regulated from step one (anti-collapse, no warmup), and the optimal LR
transfers across model width (no per-bc-size retune). It is the NEXT-ITERATION
optimizer ABLATION for the witness trainer — a candidate MEASURED win, never
assumed (NO-FAKE).

THE METHOD (exact, from the paper/blog; quoted formulas in the module docstrings):
  Each 2D weight matrix W (out, in) is reparametrized as
      W = diag(gamma_row) @ What @ diag(gamma_col)
  where ``What`` is held on a FIXED-NORM Frobenius sphere (||What||_F constant,
  initialized to ||W_init||_F) and ``gamma_row in R^out`` / ``gamma_col in R^in``
  are learnable per-row / per-column scalar magnitude GAINS (init 1.0, so the
  reparam is the identity at step 0). The direction ``What`` and the gains are
  updated at SEPARATE learning rates (paper: eta_W ~ 3e-3, eta_gamma ~ 1e-3 ->
  ratio ~1/3, exposed here as ``gain_lr_scale``). After the direction step the
  ``What`` is PROJECTED back onto the sphere (``What <- What * ||What_init|| /
  ||What||``). Weight decay is dropped (the weight is already on the sphere) and
  warmup is unnecessary (the relative update is regulated from step one).

  Gradients (chain rule through W = gr_i * What_ij * gc_j, with G = dL/dW):
      dL/dWhat_ij = G_ij * gr_i * gc_j                    (== gr (x) gc  .* G)
      dL/dgr_i    = sum_j G_ij * What_ij * gc_j           (reduce over cols)
      dL/dgc_j    = sum_i G_ij * gr_i  * What_ij          (reduce over rows)

  The DIRECTION base optimizer is pluggable: ``"adam"`` (Adam on What) or
  ``"muon"`` (momentum + Newton-Schulz orthogonalization, matching MLX's
  ``mlx.optimizers.Muon`` convention incl. the ``max(1, rows/cols)**0.5`` LR
  scale) — the paper states MD composes with BOTH. The GAINS always use Adam
  (1D vectors; Muon does not apply to non-matrices).

DESIGN (UNIQUE-AND-COMPLETE-PER-METHOD, CLAUDE.md): the factorization lives
ENTIRELY in optimizer STATE. The model's parameter tree is UNCHANGED — every
``model.parameters()`` leaf is still the materialized ``W`` — so the witness
trainer's EMA, int8+brotli byte-close, quantize, and CPU-torch verdict paths
compose with MD WITHOUT modification (non-invasive). Only ``*.weight`` leaves
with ``ndim == 2`` are decoupled by default; biases, the per-frame ``code``
latent (video-derived payload — NOT a weight matrix), and the fixed Fourier
``_B`` table fall through to a plain (decoupled-free) Adam step at ``eta_W``.

This optimizer exposes the surface the trainer reads from an ``mlx.optimizers``
optimizer: ``update(model, grads)``, ``learning_rate`` (schedule-aware), and
``state`` (a tree of mx arrays for ``mx.eval``). It is a drop-in for the
trainer's ``AdamW`` at the optimizer-construction site.

Evidence grade: this is a TRAINING-GRADIENT actuator (no score authority). It
makes NO score claim; the d_seg/d_pose VERDICT remains the frozen CPU-torch
authority per CLAUDE.md.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "MDDecoupledOptimizer",
    "default_md_eligible",
    "newton_schulz5",
    "stiefel_project_columns",
    "stiefel_residual",
]


def default_md_eligible(name: str, param: Any) -> bool:
    """Default decoupling predicate: a 2D ``*.weight`` leaf.

    Excludes biases (1D), the per-frame ``code`` embedding (video-derived
    payload, not a weight matrix), and the fixed Fourier ``_B`` table. Matches
    the paper's scope (weight MATRICES of linear/conv layers).
    """
    try:
        ndim = int(param.ndim)
    except Exception:  # pragma: no cover - defensive
        return False
    return name.endswith(".weight") and ndim == 2


def newton_schulz5(X: Any, steps: int = 5):
    """Zeropower-via-Newton-Schulz5 orthogonalization (Keller Jordan / Muon).

    Byte-faithful to ``mlx.optimizers.Muon._zeropower_via_newtonschulz5``: the
    quintic coefficients (3.4445, -4.7750, 2.0315), the pre-normalization, and
    the transpose-for-wide-matrices convention. Operates on a 2D mx array.
    """
    import mlx.core as mx

    assert X.ndim == 2, f"Newton-Schulz expects 2D, got {X.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose_needed = X.shape[-2] > X.shape[-1]
    if transpose_needed:
        X = X.T
    X = X / (mx.linalg.norm(X, keepdims=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = mx.addmm(b * A, A, A, beta=1.0, alpha=c)
        X = mx.addmm(a * X, B, X, beta=1.0, alpha=1.0)
    if transpose_needed:
        X = X.T
    return X


def stiefel_project_columns(W: Any, iters: int = 8):
    """Project a tall matrix ``W`` (out, in) onto the Stiefel manifold of
    ORTHONORMAL COLUMNS (``WᵀW = I_in``) via the polar factor ``W (WᵀW)^{-1/2}``.

    This is the DM1 byte-free structural cure (design memo
    ``per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629.md`` §0):
    with the FiLM conditioning ``W`` (here ``film.weight``, shape (2·H·L, mod))
    on the Stiefel manifold, ``W`` is an ISOMETRY on its row space, so the
    nonzero spectrum of ``E[M Mᵀ] = W·cov(code)·Wᵀ`` equals the spectrum of
    ``cov(code)`` and therefore ``PR(M) = PR(cov(code))`` — exactly in the
    orthonormal-columns limit, and in practice to the projection's fp32 residual
    (``‖WᵀW−I‖_F ≈ 1e-2`` after ~8 cubic steps) — so the multiplicative-resonance
    rank-collapse cannot concentrate through ``W``. (review R1-L1: the identity is
    ``WᵀW=I``-exact; the SHIPPED projection holds it to the ~1e-2 residual, not bit-exactly.)

    Method (NOT the Muon ``newton_schulz5``): the cubic Newton–Schulz polar
    iteration ``X ← X (3/2 I − 1/2 XᵀX)`` from a Frobenius-normalized start.
    This converges QUADRATICALLY to the orthogonal polar factor (column norms →
    1); the Muon quintic ``newton_schulz5`` is tuned for *direction*
    orthogonalization of a gradient/momentum and EMPIRICALLY does NOT
    orthonormalize the columns here (it leaves ``‖WᵀW−I‖_F ≈ 1.3``), so reusing
    it would be a FAKE Stiefel projection. The cubic iteration reaches the fp32
    floor ``‖WᵀW−I‖_F ≈ 1e-2`` in ~8 steps, GPU-native (no CPU/eigh sync).

    Magnitude is intentionally DISCARDED (the projection re-normalizes every
    column to unit norm): this is why a per-step Stiefel projection NEUTRALIZES
    the GLOBAL-MAGNITUDE COMPONENT of decoupled weight-decay on ``W`` — AdamW's
    ``W ← W(1−lr·wd)`` scales all columns by the same scalar, which
    column-renormalization removes — so the design's "WD=0 on W" intent is
    satisfied WITHOUT touching the optimizer construction. (review R1-L2: this
    neutralizes the uniform-shrink component; any non-uniform / per-column WD
    interaction with the Adam preconditioner is NOT claimed to be removed.)

    Parameters
    ----------
    W : mx.array
        2-D weight, shape (out, in) with ``out >= in`` (tall; orthonormal columns).
    iters : int
        Cubic Newton–Schulz steps (default 8; reaches the fp32 residual floor).

    Returns
    -------
    mx.array
        ``W`` projected to orthonormal columns (same shape, same dtype).
    """
    import mlx.core as mx

    if int(getattr(W, "ndim", 0)) != 2:
        raise ValueError(f"stiefel_project_columns expects 2D (out,in); got {getattr(W, 'shape', None)}")
    # (review R1-L3/R3) ORTHONORMAL COLUMNS require a TALL matrix (out >= in): a WIDE matrix has more
    # columns than rows, so WᵀW (in x in) is rank-deficient and CANNOT equal I_in -- the cubic
    # iteration would silently converge to a non-orthonormal fixed point (a FAKE Stiefel projection).
    # Fail closed rather than return a matrix whose stiefel_residual never reaches the floor.
    if W.shape[-2] < W.shape[-1]:
        raise ValueError(
            f"stiefel_project_columns requires a TALL matrix (out >= in) for orthonormal COLUMNS; got "
            f"shape {tuple(W.shape)} (out={W.shape[-2]} < in={W.shape[-1]}). WᵀW is rank-deficient for "
            "a wide matrix so WᵀW=I_in is unreachable; transpose for orthonormal ROWS instead.")
    in_dim = W.shape[-1]
    eye = mx.eye(in_dim, dtype=W.dtype)
    # Frobenius normalization guarantees the start spectral norm < 1 < sqrt(3) so the
    # cubic iteration is in its quadratic-convergence basin for every column scale.
    X = W / (mx.sqrt(mx.sum(W * W)) + 1e-12)
    for _ in range(int(iters)):
        X = X @ (1.5 * eye - 0.5 * (X.T @ X))
    return X


def stiefel_residual(W: Any) -> float:
    """``‖WᵀW − I‖_F`` — the orthonormal-columns residual (observability/telemetry
    + the post-projection assertion target). 0 == perfect Stiefel; the fp32 floor
    of :func:`stiefel_project_columns` is ~1e-2."""
    import mlx.core as mx

    G = W.T @ W
    eye = mx.eye(W.shape[-1], dtype=W.dtype)
    return float(mx.sqrt(mx.sum((G - eye) * (G - eye))))


def _frob(X: Any):
    import mlx.core as mx

    return mx.sqrt(mx.sum(X * X))


class MDDecoupledOptimizer:
    """Magnitude-Direction Decoupling optimizer (MLX).

    Args:
        learning_rate: direction LR ``eta_W`` — a float OR an mlx schedule
            callable ``step -> lr`` (the witness trainer passes its warmup+cosine
            schedule; MD itself does not need warmup, but the schedule is honored
            so the flag composes with the existing path).
        gain_lr_scale: ``eta_gamma = gain_lr_scale * eta_W``. Paper default ~1/3
            (eta_W 3e-3 / eta_gamma 1e-3).
        base: direction base optimizer, ``"adam"`` or ``"muon"``.
        betas, eps: Adam hyperparameters (used for gains always, and for the
            direction when ``base == "adam"``).
        momentum, nesterov, ns_steps: Muon hyperparameters (direction when
            ``base == "muon"``).
        md_eligible: predicate ``(name, param) -> bool`` selecting decoupled
            leaves. Default: 2D ``*.weight``.
    """

    def __init__(
        self,
        learning_rate: float | Callable[[Any], Any],
        gain_lr_scale: float = 1.0 / 3.0,
        base: str = "adam",
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        md_eligible: Callable[[str, Any], bool] | None = None,
    ) -> None:
        import mlx.core as mx

        base = str(base).lower()
        if base not in ("adam", "muon"):
            raise ValueError(f"base must be 'adam' or 'muon', got {base!r}")
        self.base = base
        self.gain_lr_scale = float(gain_lr_scale)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.md_eligible = md_eligible or default_md_eligible

        self._mx = mx
        self._schedule = learning_rate if callable(learning_rate) else None
        self._global_step = mx.array(0)
        if self._schedule is not None:
            self._lr = mx.array(self._schedule(self._global_step))
        else:
            self._lr = mx.array(float(learning_rate))
        # Per-parameter state: {name: {field: mx.array}}.
        self._pstate: dict[str, dict[str, Any]] = {}

    # ---- mlx.optimizers-compatible read surface -----------------------------
    @property
    def learning_rate(self):
        """Current direction LR ``eta_W`` (mx scalar; ``float(...)``-able)."""
        return self._lr

    @property
    def state(self) -> dict[str, Any]:
        """Tree of mx arrays for ``mx.eval`` (per CLAUDE.md max-observability)."""
        return {
            "step": self._global_step,
            "learning_rate": self._lr,
            "params": self._pstate,
        }

    # ---- internal Adam (one variable) ---------------------------------------
    def _adam_update(self, st: dict[str, Any], prefix: str, var: Any, grad: Any, lr: Any):
        mx = self._mx
        m_key, v_key, t_key = f"{prefix}_m", f"{prefix}_v", f"{prefix}_t"
        if m_key not in st:
            st[m_key] = mx.zeros_like(var)
            st[v_key] = mx.zeros_like(var)
            st[t_key] = mx.array(0)
        t = st[t_key] + 1
        st[t_key] = t
        m = self.b1 * st[m_key] + (1.0 - self.b1) * grad
        v = self.b2 * st[v_key] + (1.0 - self.b2) * (grad * grad)
        st[m_key] = m
        st[v_key] = v
        tf = t.astype(mx.float32)
        bc1 = 1.0 - mx.power(mx.array(self.b1), tf)
        bc2 = 1.0 - mx.power(mx.array(self.b2), tf)
        mhat = m / bc1
        vhat = v / bc2
        return var - lr * mhat / (mx.sqrt(vhat) + self.eps)

    # ---- internal Muon direction step (on What) -----------------------------
    def _muon_update(self, st: dict[str, Any], var: Any, grad: Any, lr: Any):
        mx = self._mx
        if "mom" not in st:
            st["mom"] = mx.zeros_like(var)
        v = self.momentum * st["mom"] + (1.0 - self.momentum) * grad
        st["mom"] = v
        if self.nesterov:
            update = grad * (1.0 - self.momentum) + v * self.momentum
        else:
            update = v
        update = newton_schulz5(update, steps=self.ns_steps)
        scale = max(1.0, update.shape[-2] / update.shape[-1]) ** 0.5
        return var - lr * scale * update

    # ---- the MD step for one 2D weight --------------------------------------
    def _md_step(self, name: str, W: Any, G: Any, eta_w: Any, eta_g: Any):
        mx = self._mx
        st = self._pstate.setdefault(name, {})
        if "What" not in st:
            # Init: What = W, gains = 1 -> reconstructed W == W (identity at step 0).
            st["What"] = mx.array(W)
            out, inn = W.shape
            st["gr"] = mx.ones((out,), dtype=W.dtype)
            st["gc"] = mx.ones((inn,), dtype=W.dtype)
            st["sphere"] = _frob(W)  # fixed Frobenius radius (||W_init||_F)
        What = st["What"]
        gr = st["gr"]
        gc = st["gc"]
        gr_c = gr[:, None]
        gc_r = gc[None, :]
        # Chain-rule gradients (see module docstring).
        g_what = G * gr_c * gc_r
        g_gr = mx.sum(G * What * gc_r, axis=1)
        g_gc = mx.sum(G * What * gr_c, axis=0)
        # Direction step + sphere projection.
        if self.base == "muon":
            What = self._muon_update(st, What, g_what, eta_w)
        else:
            What = self._adam_update(st, "w", What, g_what, eta_w)
        norm = _frob(What)
        What = What * (st["sphere"] / (norm + 1e-12))
        st["What"] = What
        # Gain steps (separate LR, always Adam).
        gr = self._adam_update(st, "gr", gr, g_gr, eta_g)
        gc = self._adam_update(st, "gc", gc, g_gc, eta_g)
        st["gr"] = gr
        st["gc"] = gc
        # Materialize the effective weight (model tree stays = W).
        return gr[:, None] * What * gc[None, :]

    # ---- plain (decoupled-free) Adam for non-MD leaves ----------------------
    def _plain_adam(self, name: str, p: Any, g: Any, eta_w: Any):
        st = self._pstate.setdefault(name, {})
        return self._adam_update(st, "p", p, g, eta_w)

    # ---- the trainer-facing update ------------------------------------------
    def update(self, model: Any, gradients: dict) -> None:
        """Apply ``gradients`` to ``model`` in place (mirrors ``Optimizer.update``).

        ``model.parameters()`` leaves remain the materialized ``W`` so downstream
        EMA / byte-close / verdict are untouched.
        """
        mx = self._mx
        from mlx.utils import tree_flatten, tree_unflatten

        # Advance the schedule (direction LR).
        self._global_step = self._global_step + 1
        if self._schedule is not None:
            self._lr = mx.array(self._schedule(self._global_step))
        eta_w = self._lr
        eta_g = eta_w * self.gain_lr_scale

        grads = dict(tree_flatten(gradients))
        params = dict(tree_flatten(model.parameters()))
        new_items: list[tuple[str, Any]] = []
        for name, p in params.items():
            g = grads.get(name)
            if g is None:
                new_items.append((name, p))
                continue
            if self.md_eligible(name, p):
                new_items.append((name, self._md_step(name, p, g, eta_w, eta_g)))
            else:
                new_items.append((name, self._plain_adam(name, p, g, eta_w)))
        model.update(tree_unflatten(new_items))
