"""Lane Joint-ADMM proximal wrapper for water_filling_codec_v2 (Lane Ω-W-V2).

Wraps :mod:`tac.water_filling_codec_v2` as a :class:`StreamProximalCodec` so the
coordinator in :mod:`tac.joint_admm_coordinator` can include the renderer-weights
stream in joint optimisation.

Per memory `project_codec_stacking_composition_canonical_orders_20260429.md`:
- Renderer/water-fill is the LARGEST score-arithmetic stream after masks
- Stack order: block-FP → water-fill → arithmetic ← TERMINAL (V2)
- Water-fill has 200-450 bp on Selfcomp/block-FP renderers; the ADMM
  coordinator equilibrates this against pose-delta (smaller stream) and
  any other wrapped streams to find the joint KKT waterline

Per memory `project_phases_2_3_4_design_implementation_math_provenance_20260429`
§"Lane 10 V2 wrap" (this file): converts the synthetic 2-stream KKT
validation (Round 5 CONCERN-1) into a real 4-stream codec validation
when combined with PoseDeltaProximalCodec + 2 synthetic mask streams.

Design notes
------------
* The Lane Ω-W-V2 codec exposes a single discrete knob: ``total_bits``
  (int, signed-integer bit budget). The water-fill internally allocates
  to per-channel qint widths.
* The proximal step takes the coordinator's ``target_bytes`` and converts
  to a ``total_bits`` budget. Bytes are 8 bits, BUT the water-fill produces
  per-channel qint widths whose actual encoded size depends on the
  arithmetic-terminal compression ratio — which is data-dependent and not
  perfectly linear in total_bits.
* We sample a CACHED frontier at construction time (one-shot encode at
  N evenly-spaced total_bits values). The proximal step picks the largest
  frontier sample whose encoded_bytes <= target_bytes.
* The marginal dScore/dByte is finite-differenced from the cached frontier.

CLAUDE.md compliance
--------------------
* COMPRESS-time only.
* Strict-scorer-rule: score_at_bytes is a CACHED surface; no live scorer load
  inside ``proximal_step`` (Check H STRICT + scorer-rule non-negotiable).
* No silent defaults: every public arg has explicit None or required-keyword
  contract (Check 81 STRICT).
* Tagged claims: any score-saving estimate is [prediction] until V2 measures
  on a real Lane G v3-class archive.
* No GPU dependency; pure CPU encode + dispatch through arithmetic_qint_codec.

Round 5 follow-up gating (battleplan §3.1 #3)
---------------------------------------------
This wrapper is the empirical-confirm path for Lane Joint-ADMM Round 5
CONCERN-1 (KKT residual 0.02 only on convex synthetic). Combined with
PoseDeltaProximalCodec + 2 synthetic mask streams, the run_admm 4-stream
test in test_joint_admm_proximal_water_filling_v2.py validates that the
coordinator handles discrete-staircase R(D) functions (not just smooth
quadratics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from tac.joint_admm_coordinator import (
    ProximalStepResult,
    StreamProximalCodec,
)
from tac.water_filling_codec import WaterFillError
from tac.water_filling_codec_v2 import (
    GateRegression,
    encode_omega_w_v2,
)


@dataclass(frozen=True)
class WaterFillingV2FrontierSample:
    """One sampled R(D) point on the renderer-weights stream.

    Fields
    ------
    total_bits : int
        Signed-integer bit budget passed to ``encode_omega_w_v2``.
    bytes_used : int
        Encoded byte count (after arithmetic-terminal) at this total_bits.
    score_cost : float
        Score-cost contribution at this operating point. Provided by the
        caller's CACHED ``score_at_bytes`` callback (typically a one-shot
        offline frontier sweep against pre-recorded scorer outputs;
        NEVER a live SegNet/PoseNet call inside this module).
    """

    total_bits: int
    bytes_used: int
    score_cost: float


class WaterFillingV2ProximalCodec:
    """StreamProximalCodec wrapper for ``tac.water_filling_codec_v2``.

    Construction
    ------------
    weights : torch.Tensor
        Float (O, I, kH, kW) conv weight, eligible for block-FP encoding.
    hessian : torch.Tensor
        Per-output-channel Hessian (O,) tensor.
    frontier : list[WaterFillingV2FrontierSample]
        CACHED R(D) samples (one per total_bits choice). Coordinator uses
        these to estimate score-cost + marginal dScore/dByte at the
        coordinator's requested target_bytes. Sorted by bytes_used asc at
        construction.
    name : str, default "water_fill_v2"
        Stream identifier in coordinator logs.
    variance : torch.Tensor | None
        Per-output-channel variance (O,). If None, derived from weights at
        encode time (deterministic — not a silent override).
    """

    def __init__(
        self,
        weights: torch.Tensor,
        hessian: torch.Tensor,
        frontier: Sequence[WaterFillingV2FrontierSample],
        name: str = "water_fill_v2",
        variance: torch.Tensor | None = None,
    ) -> None:
        if weights.ndim != 4:
            raise ValueError(
                f"weights must be 4-D (O, I, kH, kW); got {tuple(weights.shape)}"
            )
        if hessian.ndim != 1 or int(hessian.shape[0]) != int(weights.shape[0]):
            raise ValueError(
                f"hessian must be 1-D shape (O={int(weights.shape[0])},); "
                f"got {tuple(hessian.shape)}"
            )
        if not frontier:
            raise ValueError(
                "frontier must contain >=1 WaterFillingV2FrontierSample; pass "
                "at least one sample from a one-shot encode_omega_w_v2 sweep."
            )
        self._weights = weights
        self._hessian = hessian
        self._variance = variance
        # Sort by bytes ascending so finite-difference marginals are well-defined.
        self._frontier = sorted(frontier, key=lambda s: s.bytes_used)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def frontier(self) -> tuple[WaterFillingV2FrontierSample, ...]:
        """Read-only view of the cached R(D) frontier (sorted by bytes asc)."""
        return tuple(self._frontier)

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        """Return the frontier sample closest to ``target_bytes`` from below.

        Discrete codec: bytes_used is one of a small set; the "proximal step"
        picks the largest bytes_used <= target_bytes (or the smallest sample
        if target undershoots all). Marginal is finite-differenced from the
        adjacent sample.

        ``dual`` parameter: at large-dual regimes the coordinator wants tighter
        byte budgets. We don't currently bias selection by dual (the frontier
        is sparse + discrete; bias would just snap to the same sample). V3
        adds dual-aware tie-breaking when adjacent samples have identical
        bytes_used but different score_cost.
        """
        # Pick selected sample = largest bytes_used <= target_bytes.
        # If target undershoots ALL frontier samples, pick the smallest.
        selected = self._frontier[0]
        for s in self._frontier:
            if s.bytes_used <= target_bytes:
                selected = s
            else:
                break
        # Marginal: finite-difference vs the next-larger sample (or zero if
        # at the high end). Sign convention: positive = "more bytes => lower
        # score" (typical R(D) monotone region).
        idx = self._frontier.index(selected)
        if idx + 1 < len(self._frontier):
            nxt = self._frontier[idx + 1]
            db = float(nxt.bytes_used - selected.bytes_used)
            dc = float(selected.score_cost - nxt.score_cost)
            marginal = (dc / db) if db > 0 else 0.0
        else:
            marginal = 0.0
        return ProximalStepResult(
            encoded_bytes=int(selected.bytes_used),
            score_delta=float(selected.score_cost),
            marginal=float(marginal),
            state=selected,
        )


def build_water_filling_v2_frontier(
    weights: torch.Tensor,
    hessian: torch.Tensor,
    total_bits_grid: Sequence[int],
    score_at_bytes: Callable[[int], float],
    variance: torch.Tensor | None = None,
) -> list[WaterFillingV2FrontierSample]:
    """Construct the cached frontier for a weights tensor by sweeping total_bits.

    For each ``total_bits`` in ``total_bits_grid``:
      1. Call ``encode_omega_w_v2(...)``
      2. Measure encoded byte count
      3. Query ``score_at_bytes(bytes)`` for the score-cost

    Samples that raise ``GateRegression`` (V2 arithmetic terminal not
    amortising at that budget) OR ``WaterFillError`` (budget too small
    to fit Q=1 floor across all channels) are SKIPPED — the coordinator
    can still operate with a sparser frontier, and skipping is honest
    about what operating points are actually reachable.

    Args:
        weights: (O, I, kH, kW) tensor. Required.
        hessian: (O,) per-channel Hessian. Required.
        total_bits_grid: iterable of int budgets to sweep. Caller chooses
            the sampling density. Required.
        score_at_bytes: cached function bytes -> score_cost. Expected to be
            evaluated against pre-recorded scorer outputs (e.g. an offline
            per-byte-budget sweep). NOT a live SegNet/PoseNet call.
            Required.
        variance: (O,) optional; derived from weights if None (deterministic).

    Returns:
        list[WaterFillingV2FrontierSample] — may be SHORTER than
        ``total_bits_grid`` if some samples gate-regress.

    Raises:
        ValueError: if ``total_bits_grid`` is empty or yields zero accepted
            samples (would leave an empty frontier).
    """
    if weights is None or hessian is None:
        raise ValueError(
            "build_water_filling_v2_frontier: weights and hessian are required "
            "(no silent defaults — Check 81 STRICT)."
        )
    grid = list(total_bits_grid)
    if not grid:
        raise ValueError(
            "build_water_filling_v2_frontier: total_bits_grid is empty; "
            "pass an explicit list of budgets (e.g. [800, 1200, 1800, 2400])."
        )
    if score_at_bytes is None:
        raise ValueError(
            "build_water_filling_v2_frontier: score_at_bytes callback is "
            "required (no silent default)."
        )
    samples: list[WaterFillingV2FrontierSample] = []
    for tb in grid:
        try:
            blob = encode_omega_w_v2(
                weights,
                hessian,
                total_bits=int(tb),
                variance=variance,
            )
        except (GateRegression, WaterFillError):
            # Honest skip: this budget can't beat V1 raw OR can't fit Q=1
            # floor; leave it out of the frontier.
            continue
        bytes_used = len(blob)
        samples.append(
            WaterFillingV2FrontierSample(
                total_bits=int(tb),
                bytes_used=int(bytes_used),
                score_cost=float(score_at_bytes(bytes_used)),
            )
        )
    if not samples:
        raise ValueError(
            "build_water_filling_v2_frontier: every total_bits in grid raised "
            "GateRegression. Either widen the grid or use V1 directly."
        )
    return samples


# ── Compile-time Protocol conformance check ───────────────────────────────
# Guards against accidental Protocol drift at import time, matching the
# pattern in joint_admm_proximal_pose_delta.py. We construct a minimal
# 1-channel weight + frontier sample so the conformance asserts before any
# downstream code runs.

_dummy_weights = torch.zeros(2, 2, 1, 1, dtype=torch.float32)
_dummy_hessian = torch.ones(2, dtype=torch.float32)
_dummy_frontier = [
    WaterFillingV2FrontierSample(total_bits=64, bytes_used=200, score_cost=0.01)
]
_assert_protocol: StreamProximalCodec = WaterFillingV2ProximalCodec(
    weights=_dummy_weights,
    hessian=_dummy_hessian,
    frontier=_dummy_frontier,
)


__all__ = [
    "WaterFillingV2FrontierSample",
    "WaterFillingV2ProximalCodec",
    "build_water_filling_v2_frontier",
]
