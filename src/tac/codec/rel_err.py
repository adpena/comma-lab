# SPDX-License-Identifier: MIT
"""Canonical relative-error helper for codec curve dataflow.

Background
----------

Per the 2026-05-08 council deliberation captured in
``.omx/research/rel_err_inconsistency_audit_20260508_claude.md`` (codex
adversarial review #1), the codebase historically computed ``rel_err`` in at
least three different forms:

* **L2 ratio** — ``‖q-q_orig‖₂ / max(‖q_orig‖₂, ε)``
  (``tac.codec.per_tensor_codecs.encode_sparsity_alpha``).
* **L1 ratio (sum/sum)** — ``Σ|q-q_orig| / max(Σ|q_orig|, ε)``
  (``tac.codec.per_tensor_codecs.encode_lossy_K_coarsen`` and most
  ``tools/pr101_*`` / ``tools/pr106_*``).
* **Per-element L1 percentage** — ``mean(100 · |q-q_orig|/|q_orig|)``
  (``tools/pr101_lossy_int4_*``).

These different definitions all flowed into the same allocator key
``rel_err`` and were squared by the Lagrangian cost
``cost = bytes + λ · w · rel_err²``. Because ``L1²`` is not the squared
distortion of an inner-product space, the allocator was effectively
solving an ill-posed dual when its inputs were L1 ratios.

The 10/10 council verdict (Shannon-Dykstra-Yousfi-Fridrich-Contrarian-
Quantizr-Hotz-MacKay-Ballé-Selfcomp): canonical form is **RMS** =
``sqrt(mean((q-q_orig)² / max(orig_norm², ε)))``. The Lagrangian penalty
``λ · RMS²`` corresponds to the standard rate-MSE-distortion dual, matches
the contest score's PoseNet term ``sqrt(10·d_pose)``, and composes
cleanly across stacked codecs.

L1 and max forms remain available via explicit ``mode="l1"`` / ``mode="max"``
opt-ins; callers that select a non-RMS form are responsible for declaring
that choice (the curves they emit gain a ``rel_err_form`` tag that the
allocator inspects).

Score-relevance
---------------

This module does NOT change any encoder's numerical output today — the
canonical helper is *available* but not yet *enforced* on existing call
sites. Existing primitives in :mod:`tac.codec.per_tensor_codecs` keep
their current numerical behavior to preserve every empirical anchor that
shipped before this refactor (including the 0.0386 / 0.0415 PR101 Path-B
anchors). The fix is structural: a single typed entry point for new code
+ a runtime assertion in the allocator + a preflight check that catches
silent re-implementation.

Public API
----------

``RelErrForm`` enum, ``compute_rel_err``, ``aggregate_rel_err``,
``REL_ERR_FORM_KEY`` (the tag key used in curve rows).
"""
from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np


REL_ERR_FORM_KEY: Final[str] = "rel_err_form"
"""Canonical curve-row tag key declaring the form of ``rel_err``.

Curves emitted by canonical helpers carry this key alongside ``rel_err``
so the Lagrangian allocator can assert form-uniformity at the entry of
its bisection. Allowed values are the lowercase ``RelErrForm`` names.
"""

EPS_NUMERICAL: Final[float] = 1e-12
"""Numerical floor for division stability (matches existing module conventions)."""


class RelErrForm(enum.Enum):
    """Canonical relative-error forms.

    Members:
        RMS: ``sqrt(mean((q-q_orig)²) / max(mean(q_orig²), ε))``. The default
            and the only form whose square is the squared-distortion dual of
            the Lagrangian penalty ``λ · w · rel_err²``.
        L1_RATIO: ``sum|q-q_orig| / max(sum|q_orig|, ε)``. Legacy form used by
            the PR101 lossy_coarsening / PR106 UNIWARD-packet path. Preserved
            so historical anchors stay valid; allocator behavior with L1²
            is mathematically loose and should be documented at the call
            site.
        L2_RATIO: ``‖q-q_orig‖₂ / max(‖q_orig‖₂, ε)``. Used by
            ``encode_sparsity_alpha``. Equivalent to RMS up to the
            ``sqrt(N)`` normalization both sides cancel.
        MAX_RATIO: ``max|q-q_orig| / max(max|q_orig|, ε)``. Worst-case
            element bound; available for reporting only.
    """

    RMS = "rms"
    L1_RATIO = "l1_ratio"
    L2_RATIO = "l2_ratio"
    MAX_RATIO = "max_ratio"


def compute_rel_err(
    q: np.ndarray,
    q_orig: np.ndarray,
    *,
    mode: RelErrForm | str = RelErrForm.RMS,
) -> float:
    """Compute relative error between reconstruction ``q`` and original ``q_orig``.

    Args:
        q: reconstruction (numpy array, any numeric dtype, any shape).
        q_orig: original reference of same shape as ``q``.
        mode: one of :class:`RelErrForm` (or its string name). Defaults to
            ``RMS``, the canonical form.

    Returns:
        Non-negative float relative error per the requested form.

    Raises:
        ValueError: if shapes mismatch or mode is unknown.
    """
    if isinstance(mode, str):
        try:
            mode = RelErrForm(mode)
        except ValueError as e:
            raise ValueError(
                f"unknown rel_err mode {mode!r}; allowed: "
                f"{[m.value for m in RelErrForm]}"
            ) from e

    arr_q = np.asarray(q, dtype=np.float64)
    arr_o = np.asarray(q_orig, dtype=np.float64)
    if arr_q.shape != arr_o.shape:
        raise ValueError(
            f"shape mismatch q={arr_q.shape} q_orig={arr_o.shape}"
        )

    diff = arr_q - arr_o

    if mode is RelErrForm.RMS:
        # sqrt(mean(diff^2)) / sqrt(mean(orig^2 + eps))
        # Equivalent (and numerically more stable) to
        # ‖diff‖₂ / sqrt(N) divided by ‖orig‖₂ / sqrt(N) when N matches,
        # i.e. ‖diff‖₂ / ‖orig‖₂. We compute via mean to make the
        # eps-floor explicit on the energy scale.
        mean_sq = float(np.mean(arr_o.astype(np.float64) ** 2))
        denom = math.sqrt(max(mean_sq, EPS_NUMERICAL))
        num = math.sqrt(float(np.mean(diff.astype(np.float64) ** 2)))
        return num / denom if denom > 0.0 else 0.0

    if mode is RelErrForm.L2_RATIO:
        denom = float(np.linalg.norm(arr_o)) + EPS_NUMERICAL
        return float(np.linalg.norm(diff)) / denom

    if mode is RelErrForm.L1_RATIO:
        abs_orig = float(np.abs(arr_o).sum())
        if abs_orig <= EPS_NUMERICAL:
            return 0.0
        return float(np.abs(diff).sum()) / abs_orig

    if mode is RelErrForm.MAX_RATIO:
        max_orig = float(np.abs(arr_o).max()) if arr_o.size else 0.0
        if max_orig <= EPS_NUMERICAL:
            return 0.0
        return float(np.abs(diff).max()) / max_orig

    raise ValueError(f"unhandled mode {mode!r}")  # pragma: no cover


def aggregate_rel_err(
    values: Sequence[float] | Iterable[float],
    *,
    mode: RelErrForm | str = RelErrForm.RMS,
) -> float:
    """Aggregate per-tensor relative errors into a single scalar.

    The aggregation form should match the per-tensor form: RMS aggregates
    via root-mean-square, L1 aggregates via mean (NOT sum, to keep the
    output a ratio rather than a magnitude), L2 aggregates via root-sum-
    square, and max aggregates via max.

    Args:
        values: per-tensor non-negative relative errors.
        mode: aggregation form (defaults to RMS).

    Returns:
        Aggregated scalar.
    """
    if isinstance(mode, str):
        mode = RelErrForm(mode)
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if (arr < 0).any():
        # rel_err is non-negative by construction; surface upstream bugs
        raise ValueError("aggregate_rel_err received a negative value")

    if mode is RelErrForm.RMS:
        return float(np.sqrt(np.mean(arr ** 2)))
    if mode is RelErrForm.L1_RATIO:
        return float(np.mean(arr))
    if mode is RelErrForm.L2_RATIO:
        return float(np.sqrt(np.sum(arr ** 2)))
    if mode is RelErrForm.MAX_RATIO:
        return float(np.max(arr))
    raise ValueError(f"unhandled mode {mode!r}")  # pragma: no cover


def assert_uniform_rel_err_form(
    curves: Sequence[Sequence[dict]],
    *,
    strict: bool = False,
) -> RelErrForm | None:
    """Assert all curve rows declare the same ``rel_err_form`` (or none).

    The Lagrangian allocator squares ``rel_err`` and bisects on its
    aggregate; that math is only sound when every curve row reports the
    same form. This helper enables the allocator to fail loud on mixed
    inputs.

    Args:
        curves: per-tensor curves (list-of-lists of dict rows).
        strict: when True, missing form tags raise ``ValueError``; when
            False (default), missing tags are tolerated for backward
            compatibility with curves built before this refactor.

    Returns:
        The discovered form, or ``None`` if no row carries a tag.

    Raises:
        ValueError: if multiple distinct forms are present, or if
            ``strict`` and any row lacks the tag.
    """
    seen: set[str] = set()
    missing = 0
    total = 0
    for tensor_rows in curves:
        for row in tensor_rows:
            total += 1
            form = row.get(REL_ERR_FORM_KEY)
            if form is None:
                missing += 1
                continue
            seen.add(str(form))

    if len(seen) > 1:
        raise ValueError(
            f"mixed rel_err_form across curves: {sorted(seen)}; "
            "the Lagrangian allocator requires uniform form. Tag rows "
            f"with '{REL_ERR_FORM_KEY}' from "
            "tac.codec.rel_err.RelErrForm and recompute."
        )
    if strict and missing > 0:
        raise ValueError(
            f"{missing}/{total} curve rows lack '{REL_ERR_FORM_KEY}'; "
            "strict=True requires every row to declare its form."
        )
    if seen:
        return RelErrForm(next(iter(seen)))
    return None


__all__ = [
    "EPS_NUMERICAL",
    "REL_ERR_FORM_KEY",
    "RelErrForm",
    "aggregate_rel_err",
    "assert_uniform_rel_err_form",
    "compute_rel_err",
]
