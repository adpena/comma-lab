# SPDX-License-Identifier: MIT
"""Model soup averaging — Wortsman et al 2022 canonical pattern.

Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi,
Carmon, Kornblith, Schmidt — "Model soups: averaging weights of multiple fine-
tuned models improves accuracy without increasing inference time" (ICML 2022).

The empirical claim: averaging weights of N independently-trained models
(SAME initialization, DIFFERENT hyperparameters) often beats the best single
model. Two variants land:

* :class:`UniformModelSoup` — simple weighted average over all N checkpoints;
  weights default to uniform 1/N.
* :class:`GreedyModelSoup` — sort N checkpoints by validation score; add to
  soup in greedy-best-first order; KEEP only if it improves the held-out
  metric. Per Wortsman §3.2.

Both variants REQUIRE that all checkpoints share the same architecture (same
state_dict keys + tensor shapes). Per CLAUDE.md "Apples-to-apples evidence
discipline": model soups across DIFFERENT initializations (e.g. random seed
0 vs seed 42 + 5K-step pretraining divergence) violate the linear-mode-
connectivity assumption and produce unpredictable results. We refuse with
:class:`ModelSoupError` on shape mismatch; we DO NOT refuse on init divergence
because we can't detect it from state_dicts alone (operator responsibility).

Kaggle ensembling tie-in
────────────────────────
The Kaggle competition pattern "average N-fold cross-validation checkpoints"
IS a uniform model soup. Substrates can apply this to checkpoints harvested
across hyperparameter sweeps without re-training.

A1 BOLT-ON-on-A1 wave tie-in
────────────────────────────
Per T4 SYMPOSIUM Priority 1 Decision 2C: a BOLT-ON-on-A1 lane that produces
3 sub-A1 candidates (Ballé hyperprior / PR101 entropy stack / VQ-codebook) can
greedy-soup their respective bolt-on heads (with A1 backbone frozen) and may
produce a sub-best-individual candidate.

`[literature-extrapolation]` claims:
- Wortsman 2022 reports ~1-2 point improvement on ImageNet top-1 when greedy-
  souping 71 fine-tuned ViT checkpoints. Whether this generalizes to 100k-
  param contest substrates at scorer-axis floor 0.193 is empirically unknown
  (one BOLT-ON lane post-T4-SYMPOSIUM would be the first anchor).

`[derived]` claims:
- Linear-mode-connectivity assumption: if loss surface is convex along the
  segment connecting two checkpoints, the soup interpolates linearly between
  them. This is a strong assumption; soup variants do NOT assert it.

Cargo-cult audit per assumption
───────────────────────────────
* "Uniform averaging always helps" — CARGO-CULTED for arbitrary checkpoint
  pools; HARD-EARNED only for fine-tuned-from-same-init pools per Wortsman.
* "Greedy soup with held-out validation is robust to selection bias" —
  CARGO-CULTED unless held-out is the actual contest scorer (which is
  non-differentiable + slow); for proxy validation, selection bias is real.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* State-dict averaging arithmetic → ADOPT canonical (tensor add + scalar
  multiply; PyTorch native).
* Held-out validation metric → UNIQUE per substrate (caller supplies the
  metric callable).
* Linear-mode-connectivity check → DEFERRED (no canonical implementation;
  flagged as `[would-need-empirical]` in design memo).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


class ModelSoupError(RuntimeError):
    """Raised when model-soup invariants are violated."""


@dataclass(frozen=True)
class ModelSoupResult:
    """Result of a model-soup operation.

    Args:
        soup_state_dict: The averaged state_dict (CPU; ready for ``model.load_
            state_dict``).
        num_checkpoints_in_soup: How many checkpoints were averaged (after any
            greedy filtering).
        checkpoint_keys_kept: Tuple of operator-readable checkpoint keys that
            were kept in the soup (for greedy soup: the subset that improved
            held-out; for uniform soup: all input keys).
        held_out_metric_before: Held-out metric on the BEST individual
            checkpoint (greedy soup) or ``None`` (uniform soup, no comparison).
        held_out_metric_after: Held-out metric on the soup (or ``None`` if no
            held-out evaluator provided).
    """

    soup_state_dict: dict[str, torch.Tensor]
    num_checkpoints_in_soup: int
    checkpoint_keys_kept: tuple[str, ...]
    held_out_metric_before: float | None
    held_out_metric_after: float | None


def _validate_checkpoint_shapes(
    checkpoints: dict[str, dict[str, torch.Tensor]],
) -> None:
    if not checkpoints:
        raise ModelSoupError("model soup requires >= 1 checkpoint")
    reference_key = next(iter(checkpoints))
    reference = checkpoints[reference_key]
    for k, ckpt in checkpoints.items():
        if set(ckpt.keys()) != set(reference.keys()):
            raise ModelSoupError(
                f"checkpoint {k!r} state_dict keys differ from reference "
                f"{reference_key!r}; refusing soup to prevent silent corruption"
            )
        for tk, tv in ckpt.items():
            if tv.shape != reference[tk].shape:
                raise ModelSoupError(
                    f"checkpoint {k!r} tensor {tk!r} shape {tuple(tv.shape)} "
                    f"!= reference {reference_key!r} shape "
                    f"{tuple(reference[tk].shape)}"
                )


class UniformModelSoup:
    """Uniform-weighted model soup over N checkpoints."""

    def __call__(
        self,
        checkpoints: dict[str, dict[str, torch.Tensor]],
        *,
        weights: dict[str, float] | None = None,
    ) -> ModelSoupResult:
        """Average ``checkpoints`` into one soup state_dict.

        Args:
            checkpoints: Dict of ``{checkpoint_key: state_dict}``.
            weights: Optional dict of ``{checkpoint_key: weight}``. Must
                cover all keys in ``checkpoints``; weights are renormalized
                to sum to 1.0. Default uniform 1/N.

        Returns:
            :class:`ModelSoupResult` with ``num_checkpoints_in_soup == N``
            and ``checkpoint_keys_kept == tuple(checkpoints.keys())``.

        Raises:
            :class:`ModelSoupError` on shape mismatch or weight validation.
        """
        _validate_checkpoint_shapes(checkpoints)
        keys = sorted(checkpoints.keys())
        if weights is None:
            normalized_weights = {k: 1.0 / len(keys) for k in keys}
        else:
            if set(weights.keys()) != set(keys):
                raise ModelSoupError(
                    f"weights keys {sorted(weights.keys())} != checkpoint keys "
                    f"{keys}"
                )
            for k, w in weights.items():
                if w < 0:
                    raise ModelSoupError(
                        f"weight for {k!r} is {w} < 0; weights must be non-negative"
                    )
            total = sum(weights.values())
            if total <= 0:
                raise ModelSoupError(
                    f"sum of weights is {total}; must be > 0"
                )
            normalized_weights = {k: w / total for k, w in weights.items()}

        # Initialize soup with zero tensors of correct shape + dtype.
        reference = checkpoints[keys[0]]
        soup: dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in reference.items()
        }
        for ckpt_key in keys:
            w = normalized_weights[ckpt_key]
            ckpt = checkpoints[ckpt_key]
            for tk, tv in ckpt.items():
                if not tv.is_floating_point():
                    # Non-float buffers: copy from first checkpoint (averaging
                    # ints is undefined per same pattern as tac.training.EMA).
                    if ckpt_key == keys[0]:
                        soup[tk] = tv.detach().clone()
                else:
                    soup[tk].add_(tv.detach().to(torch.float32), alpha=w)

        # Restore original dtypes (cast soup back to reference dtype per key).
        soup_final: dict[str, torch.Tensor] = {}
        for k, v in reference.items():
            if v.is_floating_point():
                soup_final[k] = soup[k].to(v.dtype)
            else:
                soup_final[k] = soup[k]  # already copied as int buffer

        return ModelSoupResult(
            soup_state_dict=soup_final,
            num_checkpoints_in_soup=len(keys),
            checkpoint_keys_kept=tuple(keys),
            held_out_metric_before=None,
            held_out_metric_after=None,
        )


class GreedyModelSoup:
    """Greedy model soup (Wortsman §3.2)."""

    def __call__(
        self,
        checkpoints: dict[str, dict[str, torch.Tensor]],
        *,
        held_out_metric_fn: Callable[[dict[str, torch.Tensor]], float],
        minimize: bool = True,
    ) -> ModelSoupResult:
        """Greedy-build a soup by adding checkpoints best-first.

        Algorithm (Wortsman §3.2):
        1. Score each checkpoint individually via ``held_out_metric_fn``.
        2. Sort by score (best first per ``minimize``).
        3. Initialize soup with the best checkpoint.
        4. For each remaining checkpoint in score order:
           a. Compute candidate soup = uniform-average(soup + checkpoint).
           b. Score candidate soup via ``held_out_metric_fn``.
           c. KEEP if candidate is BETTER than current soup; else REJECT.
        5. Return final soup.

        Args:
            checkpoints: Dict of ``{checkpoint_key: state_dict}``.
            held_out_metric_fn: Callable that takes a state_dict and returns
                a scalar metric. The metric is interpreted per ``minimize``.
            minimize: If True (default), lower metric is better (contest-axis
                score semantics). If False, higher is better.

        Returns:
            :class:`ModelSoupResult` with ``num_checkpoints_in_soup`` >= 1
            and ``checkpoint_keys_kept`` = subset that survived greedy
            filtering.

        Raises:
            :class:`ModelSoupError` on shape mismatch.
        """
        _validate_checkpoint_shapes(checkpoints)
        if not callable(held_out_metric_fn):
            raise ModelSoupError("held_out_metric_fn must be callable")

        # Score each checkpoint individually.
        individual_scores: dict[str, float] = {
            k: float(held_out_metric_fn(v)) for k, v in checkpoints.items()
        }
        sort_reverse = not minimize
        sorted_keys = sorted(
            individual_scores.keys(),
            key=lambda k: individual_scores[k],
            reverse=sort_reverse,
        )

        best_key = sorted_keys[0]
        best_score = individual_scores[best_key]
        soup: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in checkpoints[best_key].items()
        }
        kept_keys: list[str] = [best_key]
        current_soup_score = best_score
        uniform = UniformModelSoup()

        for ckpt_key in sorted_keys[1:]:
            # Build candidate soup from kept + this checkpoint (uniform).
            candidate_pool = {k: checkpoints[k] for k in kept_keys + [ckpt_key]}
            candidate_result = uniform(candidate_pool)
            candidate_score = float(
                held_out_metric_fn(candidate_result.soup_state_dict)
            )
            improvement = (
                candidate_score < current_soup_score
                if minimize
                else candidate_score > current_soup_score
            )
            if improvement:
                soup = candidate_result.soup_state_dict
                kept_keys.append(ckpt_key)
                current_soup_score = candidate_score

        return ModelSoupResult(
            soup_state_dict=soup,
            num_checkpoints_in_soup=len(kept_keys),
            checkpoint_keys_kept=tuple(kept_keys),
            held_out_metric_before=best_score,
            held_out_metric_after=current_soup_score,
        )
