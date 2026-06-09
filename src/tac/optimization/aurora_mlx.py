# SPDX-License-Identifier: MIT
"""Aurora leverage-uniform polar kernel (Tilde Research, 2026) — MLX port.

Source-faithful MLX port of the *kernel* of Tilde Research's Aurora optimizer
(https://github.com/tilde-research/aurora-release, MIT, commit verified
2026-06-09). Aurora is a "leverage-aware" optimizer that fixes a structural
"neuron death" failure mode of Muon on **tall / rectangular** weight matrices.

What Aurora fixes (verified primary source)
-------------------------------------------

Muon's update is ``polar(G)`` (Newton-Schulz orthogonalization → nearest
semi-orthogonal matrix). For a **tall** ``m x n`` matrix (``m > n``) it is
*mathematically impossible* for ``polar(G)`` to be column-orthonormal
(``UᵀU = I_n``) **and** have uniform row norms simultaneously. Muon enforces
only the first, so the left-singular row norms ("leverage scores") become
non-uniform: low-leverage rows get a self-reinforcing small share of the update
mass. Under an activation ``φ`` with ``φ(0)=0 ∧ φ'(0)≈0`` (SwiGLU, ReLU², GELU,
SiLU — per Tilde's blog https://blog.tilderesearch.com/blog/aurora) this becomes
a vanishing-gradient feedback loop and neurons die permanently ("more than one
in four neurons effectively dead by step 500" at 340M scale — `[external-claim]`).

Aurora's fix: alternating projection onto the **intersection** of the Stiefel
manifold (``UᵀU = I_n``) and the row-oblique manifold (``‖U_i‖² = n/m``), via a
damped diagonal-preconditioning iteration (``pp_iterations`` default 2,
``pp_beta`` default 0.5). For **square** matrices it provably **reduces to the
standard Muon polar update** (no leverage freedom to exploit). For **wide**
matrices it transposes to tall; Tilde notes row-normalization is "unnecessary
or perhaps even harmful for square/wide cases" because the orthogonality
constraint already forces uniform leverage on the short dimension.

Relationship to the canonical MLX Muon kernel
---------------------------------------------

This module is the Aurora analogue of
``tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx`` (the
canonical PR95/Keller-Jordan Muon NS kernel). The two are *interchangeable at
the kernel call site* inside ``apply_pr95_mlx_optimizer_step`` — Aurora's
``aurora_leverage_uniform_polar_mlx`` is a drop-in replacement for the
``zeropower_via_newtonschulz5_mlx`` call, with the surrounding momentum /
weight-decay / aspect-ratio scaffolding unchanged. See the wire-in SPEC in
``.omx/research/tilde_research_optimizers_survey_and_aurora_build_20260609.md``.

NOTE on the difference from Keller-Jordan's 5-step NS coefficients: Aurora's
reference ``polar.py`` uses the **12-step simple-quintic** polar
``p(σ) = 2σ - 1.5σ³ + 0.5σ⁵`` (matching the modded-nanoGPT track-3 baseline at
"not optimizing for wallclock speed"), NOT Keller-Jordan's tuned 5-step
``(3.4445, -4.7750, 2.0315)``. This module reproduces Aurora's 12-step quintic
exactly so that any optimizer built on it matches Aurora's leaderboard behavior.

Regime fit for the contest HNeRV decoder (honest)
-------------------------------------------------

The contest ~229K HNeRV decoder's Muon-eligible partition (per
``partition_pr95_mlx_parameter_names``: ndim≥2 weights, excluding stem / rgb
heads / latents) is — empirically — **11 WIDE matrices, 0 TALL, 0 SQUARE**. The
one genuinely tall weight (``stem.weight``, aspect 61.7) is in the **AdamW**
partition (input-adjacent, per Keller-Jordan canon). And HNeRV uses the **sin**
activation (``sin'(0)=1 ≠ 0``), which **breaks Aurora's neuron-death
precondition**. So there are TWO independent reasons predicting that Aurora's
benefit over vanilla Muon at HNeRV stage-8 is ≈ 0 (possibly mildly negative on
wide matrices). This module exists to make that hypothesis **cheaply
falsifiable** via a ``$0`` local MLX A/B arbitrated by the exact evaluator-action
waterfiller — not to assert a win. See the build memo for the full analysis.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------

This kernel produces NO contest score. MLX numbers are
``[macOS-MLX research-signal]`` only; ``score_claim=false``,
``promotable=false``, ``promotion_eligible=false``,
``ready_for_exact_eval_dispatch=false``. Any score/promote/rank/kill decision
requires the canonical paired CUDA + Linux-x86_64 CPU auth-eval path on the
exact byte-closed archive per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA" + "MPS auth eval is NOISE".

Cross-references
----------------

- Reference impl: https://github.com/tilde-research/aurora-release (MIT)
- Tilde blog: https://blog.tilderesearch.com/blog/aurora
- Canonical Muon kernel (the interchange partner):
  ``tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx``
- Canonical torch Muon: ``tac.optimization.muon``
- Existing harness ``aurora_like`` Wave-N+11 optimizer object:
  ``tac.substrates._shared.mlx_score_aware.adapter._build_aurora_like_mlx_optimizer``
  (this module factors out the leverage-uniform polar that adapter duplicates;
  the wire-in SPEC routes both through this single tested kernel).
- Build/survey memo:
  ``.omx/research/tilde_research_optimizers_survey_and_aurora_build_20260609.md``
- Lane: ``lane_tilde_opt_aurora_20260609`` (research_only).
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "AURORA_DEFAULT_PP_BETA",
    "AURORA_DEFAULT_PP_ITERATIONS",
    "AURORA_SIMPLE_QUINTIC_COEFFS",
    "AURORA_SIMPLE_QUINTIC_STEPS",
    "AURORA_SOURCE_LICENSE",
    "AURORA_SOURCE_REPO",
    "AuroraMlxError",
    "aurora_leverage_uniform_polar_mlx",
    "aurora_simple_quintic_polar_mlx",
    "aurora_update_mlx",
    "classify_matrix_shape",
]

# Verified primary-source provenance (clone inspected 2026-06-09).
AURORA_SOURCE_REPO = "https://github.com/tilde-research/aurora-release"
AURORA_SOURCE_LICENSE = "MIT"

# Aurora reference ``polar.py``: 12-step simple-quintic p(σ)=2σ-1.5σ³+0.5σ⁵.
AURORA_SIMPLE_QUINTIC_COEFFS: tuple[float, float, float] = (2.0, -1.5, 0.5)
AURORA_SIMPLE_QUINTIC_STEPS: int = 12

# Aurora reference defaults (``aurora.py``).
AURORA_DEFAULT_PP_ITERATIONS: int = 2
AURORA_DEFAULT_PP_BETA: float = 0.5


class AuroraMlxError(ValueError):
    """Raised on malformed Aurora kernel arguments."""


def _require_mlx() -> Any:
    """Return the ``mlx.core`` module or raise a clear error if unavailable."""

    try:  # pragma: no cover - exercised only on Apple Silicon with MLX.
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
        raise AuroraMlxError(
            "aurora_mlx requires the 'mlx' package (Apple Silicon). "
            f"import failed: {exc!r}"
        ) from exc
    return mx


def classify_matrix_shape(rows: int, cols: int) -> str:
    """Classify a 2-D shape as ``'tall'`` / ``'wide'`` / ``'square'``.

    Aurora's leverage-uniform refinement only changes the update for **tall**
    matrices (``rows > cols``). ``'square'`` reduces exactly to the Muon polar
    update; ``'wide'`` is transposed to tall internally (and Tilde notes
    row-normalization may be unnecessary/harmful for wide).
    """

    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise AuroraMlxError(f"shape must be positive, got ({rows}, {cols})")
    if rows > cols:
        return "tall"
    if rows < cols:
        return "wide"
    return "square"


def aurora_simple_quintic_polar_mlx(
    matrix: Any,
    *,
    steps: int = AURORA_SIMPLE_QUINTIC_STEPS,
    eps: float = 1.0e-7,
    cast_float32_to_bfloat16: bool = True,
) -> Any:
    """Polar factor via Aurora's 12-step simple-quintic Newton-Schulz.

    Byte-for-byte faithful to Aurora's ``polar.py``: it maps every non-zero
    singular value of ``matrix`` to 1 using ``p(σ) = 2σ - 1.5σ³ + 0.5σ⁵`` with
    ``σ=1`` super-attracting. For a tall input it transposes to wide first (so
    the Gram ``X Xᵀ`` is the smaller ``n x n``), mirroring the reference.

    Differs from ``zeropower_via_newtonschulz5_mlx`` (Keller-Jordan 5-step,
    tuned coefficients) — this is the *Aurora* polar so Aurora-built optimizers
    reproduce Aurora's behavior.

    Args:
        matrix: 2-D MLX array of shape ``(m, n)``.
        steps: quintic iterations (default 12, the Aurora reference value).
        eps: numerical-stability constant for the initial spectral-norm scale.
        cast_float32_to_bfloat16: match the reference bf16 compute path when the
            input is float32. Set ``False`` to keep the input dtype (used for
            tighter float32 parity checks).

    Returns:
        Polar factor of ``matrix``, same shape; dtype matches the input.
    """

    mx = _require_mlx()
    if len(matrix.shape) != 2:
        raise AuroraMlxError(
            f"aurora polar expects a 2-D matrix, got shape {tuple(matrix.shape)}"
        )
    if int(steps) < 1:
        raise AuroraMlxError(f"steps must be >= 1, got {steps}")
    if float(eps) <= 0.0:
        raise AuroraMlxError(f"eps must be positive, got {eps}")

    original_dtype = matrix.dtype
    x = (
        matrix.astype(mx.bfloat16)
        if (cast_float32_to_bfloat16 and original_dtype == mx.float32)
        else matrix
    )
    transposed = int(x.shape[-2]) > int(x.shape[-1])
    if transposed:
        x = x.T
    x = x / (mx.linalg.norm(x, keepdims=True) + eps)
    a, b, c = AURORA_SIMPLE_QUINTIC_COEFFS
    for _ in range(int(steps)):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.astype(original_dtype)


def aurora_leverage_uniform_polar_mlx(
    update: Any,
    *,
    pp_iterations: int = AURORA_DEFAULT_PP_ITERATIONS,
    pp_beta: float = AURORA_DEFAULT_PP_BETA,
    eps: float = 1.0e-7,
    apply_aspect_scale: bool = True,
    polar_steps: int = AURORA_SIMPLE_QUINTIC_STEPS,
    polar_cast_float32_to_bfloat16: bool = False,
) -> Any:
    """Aurora's leverage-uniform polar projection of a (possibly N-D) update.

    This is the **drop-in replacement** for the
    ``zeropower_via_newtonschulz5_mlx`` call inside
    ``apply_pr95_mlx_optimizer_step``. Given the (already momentum-mixed) update
    tensor it returns the orthogonalized step:

    - ``ndim == 4`` conv weight ``(out, kh, kw, in)`` or ``(out, in, kh, kw)``:
      flattened to 2-D ``(out, prod(rest))`` for the projection, then reshaped
      back (matching the canonical Muon dispatch convention).
    - ``ndim == 2``: projected directly.
    - **square** ``(m == n)``: returns ``polar(update)`` — exactly the Muon step
      (Aurora provably reduces to Muon here).
    - **tall** ``(m > n)``: alternating projection onto Stiefel ∩ row-oblique,
      damped diagonal preconditioner ``D_k = D_{k-1}^β · diag(r_k)^{1-β}``.
    - **wide** ``(m < n)``: transposed to tall, projected, transposed back.

    The optional ``apply_aspect_scale`` multiplies by Muon's
    ``max(1, m/n)**0.5`` so the step magnitude matches the canonical kernel's
    output convention (the caller may instead apply the scale itself; set
    ``False`` to get the bare projection).

    Reference: ``aurora.py`` lines 40-63 (the polar/preconditioning block).
    The reference computes the inner preconditioning loop in float32 — this
    mirror does the same (``polar_cast_float32_to_bfloat16=False`` by default).

    Args:
        update: MLX array, ndim 2-4.
        pp_iterations: damped projection iterations (>=1; default 2).
        pp_beta: damping exponent (>0; default 0.5).
        eps: row-norm clamp constant.
        apply_aspect_scale: apply Muon's aspect-ratio scaling on exit.
        polar_steps: quintic iterations per polar (default 12).
        polar_cast_float32_to_bfloat16: bf16 inside the per-iteration polar.

    Returns:
        Orthogonalized update of the same shape and dtype as ``update``.
    """

    mx = _require_mlx()
    if int(pp_iterations) < 1:
        raise AuroraMlxError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if float(pp_beta) <= 0.0:
        raise AuroraMlxError(f"pp_beta must be positive, got {pp_beta}")
    if float(eps) <= 0.0:
        raise AuroraMlxError(f"eps must be positive, got {eps}")

    original_shape = tuple(int(dim) for dim in update.shape)
    ndim = len(original_shape)
    if ndim < 2:
        raise AuroraMlxError(
            f"aurora leverage-uniform polar expects ndim>=2, got {original_shape}"
        )
    if ndim > 2:
        rows = original_shape[0]
        cols = math.prod(original_shape[1:])
        work = mx.reshape(update, (rows, cols))
    else:
        rows, cols = original_shape
        work = update

    def _polar(mat: Any) -> Any:
        return aurora_simple_quintic_polar_mlx(
            mat,
            steps=polar_steps,
            eps=eps,
            cast_float32_to_bfloat16=polar_cast_float32_to_bfloat16,
        )

    if rows == cols:
        # Square: standard polar (no leverage freedom to exploit) == Muon.
        projected = _polar(work)
    else:
        # Wide -> transpose to tall, apply, transpose back: polar(G·D) trick.
        transposed = rows < cols
        tall = work.T if transposed else work
        tall_rows = int(tall.shape[0])
        tall_cols = int(tall.shape[1])
        tall32 = tall.astype(mx.float32)
        target_row_sq = float(tall_cols) / float(tall_rows)
        row_norm = mx.sqrt(mx.sum(tall32 * tall32, axis=-1, keepdims=True))
        d = 1.0 / mx.maximum(row_norm, mx.array(eps, dtype=mx.float32))
        projected = tall32
        for index in range(int(pp_iterations)):
            projected = _polar(d * tall32).astype(mx.float32)
            if index < int(pp_iterations) - 1:
                row_sq = mx.sum(projected * projected, axis=-1, keepdims=True)
                row_sq = mx.maximum(
                    row_sq, mx.array(eps * eps, dtype=mx.float32)
                )
                d = d * ((target_row_sq / row_sq) ** float(pp_beta))
        if transposed:
            projected = projected.T
        projected = projected.astype(work.dtype)

    if apply_aspect_scale:
        projected = projected * max(1.0, math.sqrt(float(rows) / float(cols)))

    if not bool(mx.all(mx.isfinite(projected)).item()):
        raise AuroraMlxError(
            "aurora leverage-uniform polar produced a non-finite update for a "
            f"parameter of shape {original_shape}; check for NaN/Inf gradients "
            "or an ill-conditioned weight matrix."
        )

    if ndim > 2:
        return mx.reshape(projected, original_shape)
    return projected


def aurora_update_mlx(
    weight: Any,
    gradient: Any,
    momentum: Any,
    *,
    eta: float = 0.05,
    weight_decay: float = 0.025,
    mu: float = 0.95,
    nesterov: bool = True,
    pp_iterations: int = AURORA_DEFAULT_PP_ITERATIONS,
    pp_beta: float = AURORA_DEFAULT_PP_BETA,
    eps: float = 1.0e-7,
    polar_steps: int = AURORA_SIMPLE_QUINTIC_STEPS,
    polar_cast_float32_to_bfloat16: bool = False,
) -> tuple[Any, Any]:
    """Full Aurora reference update for a single 2-D weight (functional, MLX).

    Source-faithful functional mirror of Aurora's ``aurora(W, G, momentum, ...)``
    (``aurora.py``). Because MLX arrays are immutable, this returns
    ``(new_weight, new_momentum)`` instead of mutating in place.

    Step sequence (matching the reference):
      1. SGD momentum: ``momentum = lerp(momentum, G, 1-mu)``.
      2. ``update = lerp(G, momentum, mu)`` if Nesterov else ``momentum``.
      3. Aurora leverage-uniform polar (square→Muon; tall/wide→balanced).
      4. Muon aspect-ratio scaling ``max(1, m/n)**0.5``.
      5. Decoupled weight decay then apply:
         ``W = W·(1 - eta·wd) - eta·update``.

    This is primarily for **numerical parity testing** against the cloned
    PyTorch reference and as a self-contained reference; the production wire-in
    uses ``aurora_leverage_uniform_polar_mlx`` inside the canonical
    ``apply_pr95_mlx_optimizer_step`` scaffolding (which already owns momentum /
    weight-decay / AdamW for the non-Muon partition).

    Args:
        weight: 2-D MLX weight array.
        gradient: gradient, same shape as ``weight``.
        momentum: momentum buffer, same shape as ``weight``.
        eta: learning rate (>0).
        weight_decay: decoupled weight decay (>=0).
        mu: momentum coefficient in (0, 1).
        nesterov: use Nesterov momentum.
        pp_iterations / pp_beta / eps: leverage-uniform polar controls.
        polar_steps / polar_cast_float32_to_bfloat16: inner polar controls.

    Returns:
        ``(new_weight, new_momentum)``.
    """

    _require_mlx()
    if len(weight.shape) != 2:
        raise AuroraMlxError(
            f"aurora_update_mlx expects 2-D weight, got {tuple(weight.shape)}"
        )
    if tuple(gradient.shape) != tuple(weight.shape):
        raise AuroraMlxError(
            f"gradient shape {tuple(gradient.shape)} must match weight "
            f"{tuple(weight.shape)}"
        )
    if tuple(momentum.shape) != tuple(weight.shape):
        raise AuroraMlxError(
            f"momentum shape {tuple(momentum.shape)} must match weight "
            f"{tuple(weight.shape)}"
        )
    if not (0.0 < float(mu) < 1.0):
        raise AuroraMlxError(f"mu must be in (0, 1), got {mu}")
    if float(eta) <= 0.0:
        raise AuroraMlxError(f"eta must be positive, got {eta}")
    if float(weight_decay) < 0.0:
        raise AuroraMlxError(f"weight_decay must be >= 0, got {weight_decay}")

    mu_f = float(mu)
    # SGD-momentum (Nesterov by default). lerp(a, b, t) = a + t*(b - a).
    new_momentum = momentum + (1.0 - mu_f) * (gradient - momentum)
    update = (
        gradient + mu_f * (new_momentum - gradient) if nesterov else new_momentum
    )

    # Leverage-uniform polar already applies the Muon aspect-ratio scaling.
    update = aurora_leverage_uniform_polar_mlx(
        update,
        pp_iterations=pp_iterations,
        pp_beta=pp_beta,
        eps=eps,
        apply_aspect_scale=True,
        polar_steps=polar_steps,
        polar_cast_float32_to_bfloat16=polar_cast_float32_to_bfloat16,
    )

    # Decoupled weight decay then apply.
    new_weight = weight * (1.0 - float(eta) * float(weight_decay))
    new_weight = new_weight - float(eta) * update
    return new_weight, new_momentum
