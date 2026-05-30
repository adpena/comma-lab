# SPDX-License-Identifier: MIT
"""Z8 Phase 2 M8 — canonical Yousfi-grounded score-aware per-level loss.

This module implements the
:class:`tac.substrates.z8_hierarchical_predictive_coding.binding_contract.ScoreAwareLevelLoss`
Protocol from ``binding_contract.py:419-472``. The canonical Yousfi-grounded
loss form (per the Protocol docstring verbatim) is:

    loss_at_level_i = sum_pixel(
        scorer_sensitivity_at_pixel_at_level_i
        * reconstruction_error_at_pixel
    )

NOT generic L2. The bit budget gets spent where the scorer is actually
sensitive — exactly Yousfi's "find the detector's blind spots and embed
there" methodology (CLAUDE.md "Fridrich inverse steganalysis"). Sister of
Slot GGG's per-pixel-roll SegNet-null finding as the first empirical anchor
in the sensitivity map (commit ``32a70c051``).

Canonical helper chain (M7 → M8)
--------------------------------

The ``scorer_sensitivity_map`` argument is produced by the canonical M7
helper at
``tac.substrates.z8_hierarchical_predictive_coding.scorer_sensitivity_map.Z8ScorerSensitivityMap.get_for_level(...)``
landed at commits ``8a95c9cc5`` (Phase A) + ``300702cdf`` (Phase C + D). The
M7 dispatcher supports four canonical paths:

  * **UNIFORM** (Path A): trivial all-ones at the level's resolution —
    reduces this loss to standard per-level L2 (the Protocol invariant from
    ``binding_contract.py:467-470``).
  * **EMPIRICAL_FROM_MASTER_GRADIENT** (Path B2): consumes
    ``tac.master_gradient_comparison.multi_granularity.extract_M_pixel`` +
    ``broadcast_sensitivity_map_to_channels`` + optional
    ``decompose_M_contest_per_level`` for the canonical Mallat dyadic
    projection — the OPERATIONAL Yousfi-grounded path.
  * **EMPIRICAL_SLOT_GGG** (Path B): DEFERRED-pending-research per
    ``empirical_sensitivity_map_from_slot_ggg`` reactivation criteria.
  * **FINITE_DIFFERENCE_UNIWARD** (Path C): DEFERRED-pending-paid-GPU per
    ``yousfi_uniward_finite_difference_sensitivity_map`` criteria.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": Path B
+ Path C deferrals are IMPLEMENTATION-LEVEL pending the data-side and
compute-side prereqs, NOT paradigm-level kills. The Yousfi UNIWARD-analog
paradigm is intact.

Protocol invariants (canonical witness tests in
``tests/test_score_aware_level_loss.py``)
-----------------------------------------------------------------------------

The Protocol declaration at ``binding_contract.py:451-471`` mandates:

1. **Shape contract**: ``reconstruction``, ``target`` are ``(B, C, H, W)``
   at the level's resolution; ``scorer_sensitivity_map`` is ``(B, C, H, W)``
   or broadcast-compatible.

2. **Non-negative sensitivity weights**: ``scorer_sensitivity_map`` values
   must be >= 0 (higher = scorer more sensitive at this pixel/channel).
   Negative weights produce a non-loss (gradient pointing AWAY from the
   target) and are rejected at validate-time.

3. **Uniform-sensitivity-reduces-to-L2 invariant**: when
   ``scorer_sensitivity_map == 1`` everywhere, ``per_level_loss`` reduces to
   standard L2/L1 reconstruction loss. Verified by canonical witness test
   ``test_uniform_sensitivity_reduces_to_l2``.

Framework-agnostic via element-wise duck-typed numpy/torch operations
---------------------------------------------------------------------

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th directive
+ ``tac.framework_agnostic``: this loss operates on any framework whose
tensors support ``-``, ``*``, ``**``, ``abs`` and a ``mean()``/``sum()``
reduction (numpy, torch, mlx, tinygrad all qualify). The reference
implementation uses framework-agnostic element-wise operations without
explicit framework branching — the same code path serves numpy / torch /
mlx by duck-typing.

For torch trainers with gradient flow, the canonical wire-in is identical
to numpy: pass ``torch.Tensor`` arguments and the framework's
``__mul__`` / ``__sub__`` / ``__pow__`` / ``.mean()`` Protocol-conformant
methods produce a torch scalar with autograd attached.

Per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD: this lives in the Z8
substrate package because it is Z8-specific (per-level / per-hierarchy /
Yousfi-grounded). Sister ``tac.composition.*_inverse_steganalysis_*``
packages operate at the per-archive-bolt-on surface, not the per-level
integrated surface Z8 needs.

Per Catalog #287 evidence-tag discipline: every numerical claim is paired
with adjacent source/citation evidence in test assertions; no docstring
overstatement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tac.substrates.z8_hierarchical_predictive_coding.binding_contract import (
    LevelDimensionContract,
    ScoreAwareLevelLoss,
)


__all__ = [
    "ScoreAwareLevelLossImpl",
    "build_score_aware_level_loss_for_level",
    "InvalidSensitivityMapError",
]


_LEGAL_NORMS: tuple[str, ...] = ("l2", "l1")
_LEGAL_REDUCTIONS: tuple[str, ...] = ("mean", "sum")


class InvalidSensitivityMapError(ValueError):
    """Raised when a sensitivity map violates the M8 Protocol invariants.

    Per the Protocol docstring at ``binding_contract.py:462-464``:

        "values are non-negative weights (higher = scorer more sensitive
         at this pixel/channel)"

    Negative values are rejected because they would produce a
    *non-loss* (gradient pointing AWAY from the target), violating the
    M8 Protocol contract.
    """


def _validate_norm(norm: str) -> str:
    if norm not in _LEGAL_NORMS:
        raise ValueError(
            f"norm must be one of {_LEGAL_NORMS}; got {norm!r}. "
            f"'l2' = squared-error (canonical Yousfi formulation); "
            f"'l1' = absolute-error (sister formulation)."
        )
    return norm


def _validate_reduction(reduction: str) -> str:
    if reduction not in _LEGAL_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {_LEGAL_REDUCTIONS}; got "
            f"{reduction!r}. 'mean' (default) reduces over batch + "
            f"channels + spatial dimensions (the Yousfi canonical "
            f"convention); 'sum' returns the raw weighted sum."
        )
    return reduction


def _validate_sensitivity_map_non_negative(scorer_sensitivity_map: Any) -> None:
    """Reject sensitivity maps with negative entries per M8 Protocol invariant.

    The Protocol docstring at ``binding_contract.py:462-464`` mandates
    non-negative weights. Negative weights would produce a non-loss
    (gradient pointing AWAY from the target).

    This check is framework-agnostic: relies on the tensor's element-wise
    comparison + boolean reduction protocol (numpy ``(x < 0).any()`` /
    torch ``(x < 0).any().item()`` / mlx equivalent). Catches user-facing
    accidents before the loss returns a non-loss value.

    Raises:
        InvalidSensitivityMapError: any entry in ``scorer_sensitivity_map``
            is < 0.
    """
    # Framework-agnostic non-negativity check: every backend's tensor
    # supports element-wise comparison; the result supports either
    # ``.any()`` (numpy / mlx / torch) or ``bool(...)`` of a scalar
    # boolean. We try ``.any()`` first (the canonical reduction); if
    # that returns a tensor scalar we coerce to bool via ``bool(...)``.
    try:
        has_neg_tensor = scorer_sensitivity_map < 0
        has_neg_any = has_neg_tensor.any()
    except Exception as exc:  # pragma: no cover — defensive
        raise InvalidSensitivityMapError(
            f"scorer_sensitivity_map must support element-wise comparison "
            f"and .any() reduction; got {type(scorer_sensitivity_map).__name__}: "
            f"{exc}"
        ) from exc

    # has_neg_any may be a 0-d tensor (torch/mlx) or a numpy bool; coerce
    # to bool deterministically.
    try:
        has_neg = bool(has_neg_any)
    except Exception as exc:  # pragma: no cover — defensive
        raise InvalidSensitivityMapError(
            f"scorer_sensitivity_map.any() result must be coercible to "
            f"bool; got {type(has_neg_any).__name__}: {exc}"
        ) from exc

    if has_neg:
        raise InvalidSensitivityMapError(
            "scorer_sensitivity_map must be element-wise non-negative "
            "per the M8 Protocol invariant (binding_contract.py:462-464). "
            "Higher values mean the scorer is more sensitive at that "
            "pixel/channel; negative values produce a non-loss (gradient "
            "pointing AWAY from the target) and are forbidden. Use "
            "Path A (UNIFORM) for the all-ones baseline."
        )


def _shape_of(tensor: Any) -> tuple[int, ...]:
    """Framework-agnostic shape extraction; mirrors ``tensor_protocol.shape_of``.

    Inlined here so this module does NOT depend on
    ``tac.framework_agnostic.tensor_protocol`` (which itself is unused by
    sister Z8 modules and would broaden the dependency surface). Both
    numpy.ndarray and torch.Tensor expose ``.shape`` as a tuple-coercible
    attribute.
    """
    try:
        return tuple(int(d) for d in tensor.shape)
    except Exception as exc:  # pragma: no cover — defensive
        raise TypeError(
            f"Cannot extract shape from {type(tensor).__name__}: {exc}"
        ) from exc


@dataclass(frozen=True)
class ScoreAwareLevelLossImpl:
    """Canonical implementation of the M8 ``ScoreAwareLevelLoss`` Protocol.

    Single-responsibility frozen dataclass that satisfies the Protocol from
    ``binding_contract.py:419-472`` via the canonical Yousfi-grounded
    weighted reconstruction loss:

        loss = REDUCE(scorer_sensitivity_map * |reconstruction - target|^p)

    where ``p == 2`` for ``norm='l2'`` (canonical) and ``p == 1`` for
    ``norm='l1'``. ``REDUCE`` is ``mean`` (default) or ``sum``.

    The dataclass holds NO trainable parameters (the Yousfi grounding lives
    in the sensitivity map, which is M7's responsibility). It is frozen so
    M8 instances are hashable + safe to share across hierarchy levels.

    Args:
        norm: 'l2' (default; canonical Yousfi formulation per Fridrich
            UNIWARD inverse steganalysis) or 'l1' (sister formulation
            useful when reconstruction error is heavy-tailed).
        reduction: 'mean' (default) reduces over all dimensions (B + C +
            H + W) per the canonical Yousfi convention; 'sum' returns the
            raw weighted sum.
        validate_non_negative_sensitivity: when True (default), validate
            the sensitivity map is element-wise non-negative per the M8
            Protocol invariant. Disable for hot-path trainers that have
            already validated upstream (the M7 helpers all produce
            non-negative maps by construction).

    Protocol satisfaction:
        ``isinstance(ScoreAwareLevelLossImpl(), ScoreAwareLevelLoss)`` is
        True (Protocol is ``@runtime_checkable``). Verified by canonical
        witness test ``test_satisfies_score_aware_level_loss_protocol``.

    Per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD: substrate-engineering
    implementation; lives in Z8 package because it is Z8-specific (per-
    level / per-hierarchy / Yousfi-grounded). Sister
    ``tac.composition.*_inverse_steganalysis_*`` packages operate at the
    per-archive-bolt-on surface, not the per-level integrated surface.
    """

    norm: Literal["l2", "l1"] = "l2"
    reduction: Literal["mean", "sum"] = "mean"
    validate_non_negative_sensitivity: bool = True

    def __post_init__(self) -> None:
        # Validate enum-string fields per Catalog #287 explicit-input discipline.
        _validate_norm(self.norm)
        _validate_reduction(self.reduction)

    def per_level_loss(
        self,
        reconstruction: Any,
        target: Any,
        scorer_sensitivity_map: Any,
    ) -> Any:
        """Compute per-level loss weighted by empirical scorer sensitivity.

        Satisfies the Protocol contract from ``binding_contract.py:451-471``:

        Shape contract:
            reconstruction: ``(B, C, H, W)`` at this level's resolution
            target: ``(B, C, H, W)`` at this level's resolution
            scorer_sensitivity_map: ``(B, C, H, W)`` or broadcast-compatible;
                non-negative weights
            return: scalar tensor (loss)

        Invariants verified by canonical witness tests:
          1. Uniform sensitivity (== 1 everywhere) reduces to standard L2/L1.
          2. Non-uniform sensitivity reweights the per-pixel contribution.
          3. Negative sensitivity is rejected (raises
             InvalidSensitivityMapError) when
             ``validate_non_negative_sensitivity=True`` (default).

        Args:
            reconstruction: predicted (B, C, H, W) tensor.
            target: ground-truth (B, C, H, W) tensor.
            scorer_sensitivity_map: per-pixel non-negative weights, shape
                (B, C, H, W) or broadcast-compatible (e.g. (1, 1, H, W) for
                channel-uniform per-spatial-location maps per Fridrich
                UNIWARD canonical convention).

        Returns:
            Scalar tensor with the same framework as the input tensors
            (numpy ``np.float32`` scalar for numpy input; torch ``Tensor``
            with autograd for torch input).

        Raises:
            InvalidSensitivityMapError: ``scorer_sensitivity_map`` contains
                negative entries (when
                ``self.validate_non_negative_sensitivity=True``).
            ValueError: ``reconstruction.shape != target.shape``.
        """
        # Shape contract: reconstruction and target MUST match
        # element-wise. The sensitivity map is broadcast-compatible per
        # the Protocol docstring (e.g. (1, 1, H, W) is valid for
        # channel-uniform per-spatial-location maps per Fridrich
        # UNIWARD canonical convention).
        recon_shape = _shape_of(reconstruction)
        target_shape = _shape_of(target)
        if recon_shape != target_shape:
            raise ValueError(
                f"reconstruction and target must have identical shapes; "
                f"got reconstruction={recon_shape}, target={target_shape}. "
                f"Per the M8 Protocol contract at binding_contract.py:"
                f"459-461 both MUST be (B, C, H, W) at this level's "
                f"resolution."
            )

        if self.validate_non_negative_sensitivity:
            _validate_sensitivity_map_non_negative(scorer_sensitivity_map)

        # Canonical Yousfi-grounded weighted reconstruction loss.
        # Framework-agnostic via duck-typed element-wise operations:
        # every backend's tensor supports ``-``, ``*``, ``**``, ``abs()``
        # and ``.mean()``/``.sum()`` reductions.
        error = reconstruction - target
        if self.norm == "l2":
            # Squared error per Fridrich UNIWARD canonical formulation.
            # error ** 2 is element-wise squaring; works for numpy /
            # torch / mlx alike.
            squared_or_abs_error = error * error
        else:
            # norm == "l1": absolute error. ``abs()`` is the framework-
            # native call; numpy and torch both support ``abs(tensor)``
            # via Python's __abs__ protocol.
            squared_or_abs_error = abs(error)

        weighted_error = scorer_sensitivity_map * squared_or_abs_error

        # Reduction. Both numpy and torch tensors expose ``.mean()`` and
        # ``.sum()`` as zero-argument methods returning a scalar tensor.
        if self.reduction == "mean":
            return weighted_error.mean()
        # reduction == "sum"
        return weighted_error.sum()


def build_score_aware_level_loss_for_level(
    level: LevelDimensionContract,
    *,
    norm: Literal["l2", "l1"] = "l2",
    reduction: Literal["mean", "sum"] = "mean",
    validate_non_negative_sensitivity: bool = True,
) -> ScoreAwareLevelLossImpl:
    """Single-call canonical builder for M8 trainer callsites.

    Convenience constructor producing a :class:`ScoreAwareLevelLossImpl`
    bound to the given ``level``'s shape contract (no per-level
    customization required at this stage — every level uses the same
    canonical Yousfi-grounded loss form; the level-specific signal lives
    in the per-level sensitivity map produced by M7's
    ``Z8ScorerSensitivityMap.get_for_level(level)``).

    Sister of ``build_z8_scorer_sensitivity_map_for_level`` (M7) and
    ``build_z8_mallat_dwt_adapter_for_level`` (M5) and
    ``build_z8_mamba2_adapter_for_level`` (M4) per the canonical
    per-level builder convention.

    Args:
        level: per-level contract — currently unused by the loss itself
            (every level uses the same canonical form) but pinned in the
            signature for API parity with sister M4/M5/M7 builders. The
            Z8 trainer holds one ``ScoreAwareLevelLossImpl`` per level
            so the level binding is explicit at construction time.
        norm: forwarded to ``ScoreAwareLevelLossImpl(norm=...)``.
        reduction: forwarded to ``ScoreAwareLevelLossImpl(reduction=...)``.
        validate_non_negative_sensitivity: forwarded.

    Returns:
        :class:`ScoreAwareLevelLossImpl` instance bound to the level.

    Raises:
        TypeError: ``level`` is not a ``LevelDimensionContract``.
        ValueError: ``norm`` or ``reduction`` is not a legal value.
    """
    if not isinstance(level, LevelDimensionContract):
        raise TypeError(
            f"level must be LevelDimensionContract, got {type(level).__name__}"
        )
    # ``level`` is currently unused by ScoreAwareLevelLossImpl (every
    # level uses the same canonical Yousfi-grounded form; level-specific
    # signal lives in the sensitivity map from M7). The argument is
    # pinned for API parity with sister M4/M5/M7 builders + future
    # per-level customization (e.g. per-level norm choices).
    _ = level
    # Verify Protocol satisfaction at construction time (early-fail beats
    # late-fail per CLAUDE.md "Bugs must be permanently fixed AND
    # self-protected against"). The Protocol is @runtime_checkable so the
    # isinstance check verifies structural conformance.
    impl = ScoreAwareLevelLossImpl(
        norm=norm,
        reduction=reduction,
        validate_non_negative_sensitivity=validate_non_negative_sensitivity,
    )
    assert isinstance(impl, ScoreAwareLevelLoss), (
        "ScoreAwareLevelLossImpl must satisfy ScoreAwareLevelLoss Protocol "
        "from binding_contract.py:419-472 (Protocol is @runtime_checkable)"
    )
    return impl
