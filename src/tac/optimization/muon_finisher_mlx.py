# SPDX-License-Identifier: MIT
"""MLX Muon FINISHER stage glue (PR95 stage-8 pattern) for the level-set witness trainer.

Reuse-first
-----------

The orthogonalized-momentum optimizer itself is **reused** from MLX's built-in
``mlx.optimizers.Muon`` (MomentUm Orthogonalized by Newton-Schulz; Keller Jordan
2024). The repo's own ``tac.optimization.md_decoupling.newton_schulz5`` is
documented as byte-faithful to ``mlx.optimizers.Muon._zeropower_via_newtonschulz5``
(same quintic coefficients ``(3.4445, -4.7750, 2.0315)``, pre-normalization, and
transpose-for-wide convention), so the built-in is the canonical MLX Muon. The
torch sister is ``tac.optimization.muon.MuonOptimizer`` /
``partition_params_for_muon``; this module is its MLX analog at the *param-routing*
layer.

This module is ONLY the trainer-specific glue that PR95 stage-8 needs but the
bare optimizer does not provide:

1. ``muon_finisher_param_filter`` — the PR95 partition: route the 2-D hidden
   weight matrices to Muon while keeping biases / 1-D gains / the per-pair
   ``code`` latent / the final ``out_sdf`` (K-class SDF head) + ``out_tex`` (RGB
   head) under AdamW. The MLX Muon docstring itself states Muon is sub-optimal
   for "the embedding layer, the final fully connected layer, or any 0D/1D
   parameters" — those "should be optimized by a different method (e.g. AdamW)".
   We honor that via ``mlx.optimizers.MultiOptimizer``.

2. ``build_muon_finisher_optimizer`` — assembles
   ``MultiOptimizer([Muon(2D weights), AdamW(rest)], [filter])`` (the last
   optimizer is the fallback, per the MultiOptimizer contract).

3. ``muon_active_for_epoch`` — the pure per-epoch switch predicate (default-off
   => the optimizer stays AdamW throughout => bit-identical to the pre-Muon path).

4. ``count_muon_adamw_split`` — observability helper (how many params land in
   each group) for the switch log row.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------

This module produces no contest score. Any score claim requires the canonical
``upstream/evaluate.py`` dispatch on the exact archive bytes (contest-CPU AND
contest-CUDA per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"); MLX is a
training-gradient / advisory substrate only.

Cross-references
----------------

- MLX built-in: ``mlx.optimizers.Muon`` + ``mlx.optimizers.MultiOptimizer``
- Byte-faithful NS reference: ``tac.optimization.md_decoupling.newton_schulz5``
- Torch sister: ``tac.optimization.muon`` (``MuonOptimizer`` /
  ``partition_params_for_muon``)
- PR95 stage-8: CLAUDE.md L14/L15 (8-stage curriculum; Muon final stage only,
  177K of 229K params under Muon = the 2-D weight matrices)
- Consumer: ``experiments/train_levelset_witness_realized_through_R_mlx.py``
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MUON_EXCLUDE_TOKENS",
    "build_muon_finisher_optimizer",
    "count_muon_adamw_split",
    "muon_active_for_epoch",
    "muon_finisher_param_filter",
]

# Param-name tokens routed to AdamW even when 2-D ``*.weight`` (PR95 / MLX Muon
# docstring: final FC + embeddings + stem are Muon-sub-optimal). ``out_sdf`` is the
# K-class SDF head; ``out_tex`` is the RGB texture head; ``stem`` / ``rgb`` are
# future-proofing against renamed heads (mirrors the torch
# ``partition_params_for_muon`` exclusion set).
MUON_EXCLUDE_TOKENS: tuple[str, ...] = ("out_sdf", "out_tex", "stem", "rgb")


def muon_finisher_param_filter(name: str, weight: Any) -> bool:
    """MultiOptimizer predicate: ``True`` iff ``(name, weight)`` is Muon-eligible.

    Muon-eligible = a 2-D ``*.weight`` matrix that is NOT a final head / stem /
    embedding (per :data:`MUON_EXCLUDE_TOKENS`). For the level-set witness this
    selects ``in_proj.weight``, ``film.weight``, ``hidden.<i>.weight`` and routes
    everything else (all biases/1-D gains, the per-pair ``code`` latent, the
    ``out_sdf`` + ``out_tex`` heads) to the AdamW fallback.

    Parameters
    ----------
    name : str
        Dotted MLX tree path of the leaf (e.g. ``"hidden.0.weight"``, ``"code"``).
    weight : Any
        The leaf array; only ``.ndim`` is read (MLX or numpy array).

    Returns
    -------
    bool
        ``True`` => route to Muon; ``False`` => route to the AdamW fallback.
    """
    ndim = int(getattr(weight, "ndim", 0))
    if ndim != 2:
        return False
    low = str(name).lower()
    if not low.endswith(".weight"):
        return False
    return not any(tok in low for tok in MUON_EXCLUDE_TOKENS)


def muon_active_for_epoch(epoch: int, muon_start_epoch: int | None) -> bool:
    """Pure switch predicate: is the Muon finisher active at ``epoch``?

    ``muon_start_epoch is None`` (the default) => ALWAYS ``False`` => the trainer
    never switches off AdamW => bit-identical to the pre-Muon path. Otherwise the
    Muon stage is active for every ``epoch >= muon_start_epoch`` (the ``>=`` makes
    a RESUME into an epoch past the boundary correctly re-enter the Muon stage).

    Parameters
    ----------
    epoch : int
        1-based current epoch.
    muon_start_epoch : int | None
        The Muon-finisher start epoch, or ``None`` for "AdamW throughout".

    Returns
    -------
    bool
    """
    if muon_start_epoch is None:
        return False
    return int(epoch) >= int(muon_start_epoch)


def count_muon_adamw_split(params: Any) -> tuple[int, int]:
    """Count ``(n_muon_leaves, n_adamw_leaves)`` under the finisher partition.

    Observability helper for the switch log row (CLAUDE.md max-observability:
    decomposable per signal). Operates on an MLX parameter tree (dict of dicts /
    lists of mx arrays); falls back gracefully if ``tree_flatten`` is unavailable.

    Parameters
    ----------
    params : Any
        MLX parameter tree (e.g. ``model.trainable_parameters()``).

    Returns
    -------
    tuple[int, int]
        ``(muon_leaf_count, adamw_leaf_count)``.
    """
    from mlx.utils import tree_flatten

    n_muon = 0
    n_adamw = 0
    for name, leaf in tree_flatten(params):
        if muon_finisher_param_filter(name, leaf):
            n_muon += 1
        else:
            n_adamw += 1
    return n_muon, n_adamw


def _adamw_bias_correction_for(adamw_beta2: float) -> bool:
    """#222/#224 Wave D: mirror the trainer's ``_adam_bias_correction_for`` gate for the finisher's
    AdamW rest-group. Bias correction is REQUIRED off the 0.999 default (e.g. the all-levers small-n
    beta2 0.9999999): without it the second moment ``v ~ (1-beta2)*mean(g^2)`` is not divided by
    ``(1-beta2^t)``, so ``sqrt(v)`` is ``~sqrt(1-beta2)`` too small early => ~100x step-1 LR blowup =>
    AdamW random-walk / divergence. At exactly 0.999 (== the MLX AdamW default) it stays False so the
    finisher path is BYTE-IDENTICAL to the pre-Wave-D construction (betas + bias_correction default)."""
    return abs(float(adamw_beta2) - 0.999) > 1e-9


def build_muon_finisher_optimizer(
    *,
    muon_lr: float,
    muon_adamw_lr: float,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 1e-4,
    muon_ns_steps: int = 5,
    adamw_weight_decay: float = 1e-4,
    adamw_beta2: float = 0.999,
    nesterov: bool = True,
    muon_lr_final_frac: float = 1.0,
    muon_anneal_steps: int = 0,
) -> Any:
    """Build the PR95 stage-8 finisher optimizer (MLX ``MultiOptimizer``).

    Routes the 2-D hidden weight matrices to ``mlx.optimizers.Muon`` (Newton-Schulz
    orthogonalized momentum) and every other leaf (biases / 1-D / ``code`` latent /
    final heads) to ``mlx.optimizers.AdamW``, exactly the PR95 "Muon final stage
    only; vectors/biases/stem/rgb head under AdamW" partition.

    The returned object is an ``mlx.optimizers.Optimizer`` subclass exposing the
    same ``.init`` / ``.update`` / ``.state`` surface as the trainer's existing
    ``AdamW`` optimizer => it is a drop-in replacement at the switch epoch.

    Parameters
    ----------
    muon_lr : float
        Learning rate for the Muon (2-D weight) group. Muon normalizes its update
        to ~unit spectral norm, so this is a step size in spectral-norm units;
        tune to the lever's own optimum per the OPTIMAL-FORM discipline.
    muon_adamw_lr : float
        Learning rate for the AdamW fallback group (biases / ``code`` / heads).
    muon_momentum : float
        Muon momentum (default 0.95).
    muon_weight_decay : float
        Decoupled weight decay for the Muon group (default 1e-4).
    muon_ns_steps : int
        Newton-Schulz iteration count (default 5; Keller Jordan's tuned value).
    adamw_weight_decay : float
        Weight decay for the AdamW fallback group (default 1e-4).
    adamw_beta2 : float
        AdamW second-moment decay for the fallback group (beta1 fixed 0.9). Default 0.999 == the MLX
        AdamW default => byte-identical to the pre-Wave-D finisher. The all-levers path threads the
        derived small-n optimum (0.9999999) so the Muon-stage rest-group is CONSISTENT with the main
        AdamW; bias correction is auto-gated ON off the 0.999 default (see ``_adamw_bias_correction_for``).
    nesterov : bool
        Nesterov momentum for Muon (default ``True``; recommended).
    muon_lr_final_frac : float
        (GAP 1, default-off) COSINE-DECAY the Muon-GROUP LR from ``muon_lr`` down to
        ``muon_lr * muon_lr_final_frac`` across ``muon_anneal_steps`` optimizer
        updates. Muon's Newton-Schulz fixes update MAGNITUDE, so a flat Muon LR
        cannot self-reduce the step near the minimum (river-valley Muon 2606.21514)
        => plateau/overstep; the decay lets the finisher settle. Only the Muon group
        decays (the AdamW fallback self-adapts its effective step via its second
        moment). ``1.0`` (the default) OR ``muon_anneal_steps <= 0`` => the Muon
        group keeps the SCALAR ``muon_lr`` => BYTE-IDENTICAL to the pre-GAP-1 build.
        Must be in ``(0, 1]``.
    muon_anneal_steps : int
        (GAP 1) Number of optimizer ``.update`` calls over which the cosine decay
        runs (the schedule denominator). The caller computes it as
        ``(epochs - muon_start_epoch + 1) * opt_updates_per_epoch`` so the schedule
        is DETERMINISTIC in ``muon_start_epoch`` (a RESUME into the finisher rebuilds
        the SAME schedule and the restored ``opt.step`` reproduces the bit-faithful
        LR). ``0`` (default) => no decay (scalar LR). Inert unless
        ``muon_lr_final_frac < 1.0``.

    Returns
    -------
    mlx.optimizers.MultiOptimizer
    """
    import mlx.optimizers as optim

    if muon_lr <= 0.0:
        raise ValueError(f"muon_lr must be positive, got {muon_lr}")
    if muon_adamw_lr <= 0.0:
        raise ValueError(f"muon_adamw_lr must be positive, got {muon_adamw_lr}")
    if int(muon_ns_steps) < 1:
        raise ValueError(f"muon_ns_steps must be >= 1, got {muon_ns_steps}")
    if not 0.0 <= float(muon_momentum) < 1.0:
        raise ValueError(f"muon_momentum must be in [0, 1), got {muon_momentum}")
    if not 0.0 < float(muon_lr_final_frac) <= 1.0:
        raise ValueError(f"muon_lr_final_frac must be in (0, 1], got {muon_lr_final_frac}")
    if int(muon_anneal_steps) < 0:
        raise ValueError(f"muon_anneal_steps must be >= 0, got {muon_anneal_steps}")

    # GAP 1 (default-off): scalar LR (byte-identical) unless a real decay is requested.
    # final_frac >= 1.0 OR anneal_steps <= 0 => keep the scalar muon_lr EXACTLY as before.
    if float(muon_lr_final_frac) < 1.0 and int(muon_anneal_steps) > 0:
        muon_learning_rate: Any = optim.cosine_decay(
            float(muon_lr), int(muon_anneal_steps),
            end=float(muon_lr) * float(muon_lr_final_frac),
        )
    else:
        muon_learning_rate = float(muon_lr)

    muon = optim.Muon(
        learning_rate=muon_learning_rate,
        momentum=float(muon_momentum),
        weight_decay=float(muon_weight_decay),
        nesterov=bool(nesterov),
        ns_steps=int(muon_ns_steps),
    )
    # #224 Wave D (R4 non-blocking #2): thread the derived small-n beta2 + the same bias-correction
    # gate as the trainer's main AdamW so the ep726-1000 rest-group uses beta2 CONSISTENTLY (not
    # reverting to 0.999). Default 0.999 => betas [0.9, 0.999] + bias_correction False == the MLX
    # AdamW default => byte-identical to the pre-Wave-D finisher construction.
    adamw = optim.AdamW(
        learning_rate=float(muon_adamw_lr),
        weight_decay=float(adamw_weight_decay),
        betas=[0.9, float(adamw_beta2)],
        bias_correction=_adamw_bias_correction_for(adamw_beta2),
    )
    # filters has len == len(optimizers) - 1; the last optimizer (adamw) is the
    # fallback for everything the muon filter rejects.
    return optim.MultiOptimizer([muon, adamw], [muon_finisher_param_filter])
