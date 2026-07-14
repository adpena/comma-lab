# SPDX-License-Identifier: MIT
"""Deterministic lowest-class tie snapping for approximate scorer logits.

The frozen SegNet authority is ``torch.argmax``: an exact tie is resolved to
the lowest class index.  A numerically approximate backend can turn that tie
into a tiny strict ordering.  This module exposes a deliberately narrow
decision-head formulation: among classes within a preregistered epsilon of the
candidate maximum, choose the lowest index.

This is not an authority proof by itself.  Epsilon must be selected on a
calibration split and validated on disjoint held-out rows, and any deployment
still inherits the receipt gates of the logits producer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DYADIC_TIE_EPSILONS: tuple[float, ...] = (
    0.0,
    *(float(2.0**exponent) for exponent in range(-24, -9)),
)


def epsilon_arm_name(epsilon: float) -> str:
    value = float(epsilon)
    if value == 0.0:
        return "epsilon_zero"
    for exponent in range(-24, -9):
        if value == float(2.0**exponent):
            return f"epsilon_2m{abs(exponent)}"
    raise ValueError("epsilon is not in the preregistered dyadic ladder")


def tie_snap_argmax_numpy(
    logits: Any,
    *,
    epsilon: float,
    class_axis: int = -1,
) -> np.ndarray:
    """Choose the lowest class within ``epsilon`` of the candidate maximum."""

    source = np.asarray(logits, dtype=np.float32)
    if source.ndim == 0:
        raise ValueError("tie-snap logits must have a class dimension")
    if not np.all(np.isfinite(source)):
        raise ValueError("tie-snap logits contain non-finite values")
    value = float(epsilon)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tie-snap epsilon must be finite and non-negative")
    axis = int(class_axis)
    if axis < 0:
        axis += source.ndim
    if axis < 0 or axis >= source.ndim:
        raise ValueError("tie-snap class axis is out of range")
    moved = np.moveaxis(source, axis, -1)
    classes = int(moved.shape[-1])
    if classes <= 0:
        raise ValueError("tie-snap class dimension is empty")
    maximum = np.max(moved, axis=-1, keepdims=True)
    within = moved >= (maximum - np.float32(value))
    indices = np.arange(classes, dtype=np.int64)
    return np.min(np.where(within, indices, classes), axis=-1).astype(
        np.int64,
        copy=False,
    )


def tie_snap_argmax_mlx(
    logits: Any,
    *,
    epsilon: float,
) -> Any:
    """MLX last-axis twin of :func:`tie_snap_argmax_numpy`."""

    import mlx.core as mx

    value = float(epsilon)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tie-snap epsilon must be finite and non-negative")
    if len(logits.shape) == 0 or int(logits.shape[-1]) <= 0:
        raise ValueError("tie-snap logits must have a non-empty class dimension")
    source = logits.astype(mx.float32)
    maximum = mx.max(source, axis=-1, keepdims=True)
    within = source >= (maximum - value)
    classes = int(source.shape[-1])
    indices = mx.arange(classes, dtype=mx.int32)
    return mx.min(mx.where(within, indices, classes), axis=-1)


def class_pair_tie_snap_argmax_numpy(
    logits: Any,
    *,
    epsilon: float,
    winner_class: int,
    runner_class: int,
    class_axis: int = -1,
) -> np.ndarray:
    """Snap one preregistered ordered winner/runner pair to its runner.

    The ordinary lowest-class epsilon rule is intentionally broader: it can
    alter any near tie.  This formulation changes the decision only when the
    candidate's deterministic top two classes are exactly
    ``(winner_class, runner_class)`` and their gap is no larger than
    ``epsilon``.  It therefore remains label- and frame-independent at
    runtime while allowing a calibration-derived class-pair correction to be
    validated on a disjoint corpus split.
    """

    source = np.asarray(logits, dtype=np.float32)
    if source.ndim == 0:
        raise ValueError("class-pair tie-snap logits must have a class dimension")
    if not np.all(np.isfinite(source)):
        raise ValueError("class-pair tie-snap logits contain non-finite values")
    value = float(epsilon)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("class-pair tie-snap epsilon must be finite and non-negative")
    axis = int(class_axis)
    if axis < 0:
        axis += source.ndim
    if axis < 0 or axis >= source.ndim:
        raise ValueError("class-pair tie-snap class axis is out of range")
    moved = np.moveaxis(source, axis, -1)
    classes = int(moved.shape[-1])
    winner_value = int(winner_class)
    runner_value = int(runner_class)
    if classes < 2:
        raise ValueError("class-pair tie-snap requires at least two classes")
    if not 0 <= winner_value < classes or not 0 <= runner_value < classes:
        raise ValueError("class-pair tie-snap selected class is out of range")
    if winner_value == runner_value:
        raise ValueError("class-pair tie-snap winner and runner must differ")

    winner = np.argmax(moved, axis=-1).astype(np.int64, copy=False)
    indices = np.arange(classes, dtype=np.int64)
    without_winner = np.where(
        indices == winner[..., np.newaxis],
        np.float32(-np.inf),
        moved,
    )
    runner = np.argmax(without_winner, axis=-1).astype(np.int64, copy=False)
    winner_logit = np.take_along_axis(moved, winner[..., np.newaxis], axis=-1)[..., 0]
    runner_logit = np.take_along_axis(moved, runner[..., np.newaxis], axis=-1)[..., 0]
    snap = (
        (winner == winner_value)
        & (runner == runner_value)
        & ((winner_logit - runner_logit) <= np.float32(value))
    )
    return np.where(snap, runner_value, winner).astype(np.int64, copy=False)


def class_pair_tie_snap_argmax_mlx(
    logits: Any,
    *,
    epsilon: float,
    winner_class: int,
    runner_class: int,
) -> Any:
    """MLX last-axis twin of :func:`class_pair_tie_snap_argmax_numpy`."""

    import mlx.core as mx

    value = float(epsilon)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("class-pair tie-snap epsilon must be finite and non-negative")
    if len(logits.shape) == 0:
        raise ValueError("class-pair tie-snap logits must have a class dimension")
    classes = int(logits.shape[-1])
    winner_value = int(winner_class)
    runner_value = int(runner_class)
    if classes < 2:
        raise ValueError("class-pair tie-snap requires at least two classes")
    if not 0 <= winner_value < classes or not 0 <= runner_value < classes:
        raise ValueError("class-pair tie-snap selected class is out of range")
    if winner_value == runner_value:
        raise ValueError("class-pair tie-snap winner and runner must differ")

    source = logits.astype(mx.float32)
    indices = mx.arange(classes, dtype=mx.int32)
    maximum = mx.max(source, axis=-1, keepdims=True)
    winner = mx.min(mx.where(source == maximum, indices, classes), axis=-1)
    without_winner = mx.where(
        indices == mx.expand_dims(winner, axis=-1),
        -float("inf"),
        source,
    )
    runner_maximum = mx.max(without_winner, axis=-1, keepdims=True)
    runner = mx.min(
        mx.where(without_winner == runner_maximum, indices, classes),
        axis=-1,
    )
    gap = mx.squeeze(maximum - runner_maximum, axis=-1)
    snap = (winner == winner_value) & (runner == runner_value) & (gap <= value)
    return mx.where(snap, runner_value, winner)


__all__ = [
    "DYADIC_TIE_EPSILONS",
    "class_pair_tie_snap_argmax_mlx",
    "class_pair_tie_snap_argmax_numpy",
    "epsilon_arm_name",
    "tie_snap_argmax_mlx",
    "tie_snap_argmax_numpy",
]
