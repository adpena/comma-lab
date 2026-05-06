"""Lane Joint-ADMM proximal wrapper for pose_delta_codec (Lane PD).

Wraps :mod:`tac.pose_delta_codec` as a :class:`StreamProximalCodec` so the
coordinator in :mod:`tac.joint_admm_coordinator` can include the pose-delta
stream in joint optimisation.

Per memory `project_codec_stacking_composition_canonical_orders_20260429.md`
§"Score-arithmetic priority": pose savings cap at ~5KB total (-0.003 score even
at infinite compression) — pose is the SMALLEST stream and the natural starting
point for the proximal-codec wrapping. Wrapping bigger streams (renderer.bin,
masks) is V2 scope.

Design notes
------------
* The Lane PD codec exposes a discrete bit-width parameter ``delta_bits``
  (currently only 8 is implemented). The proximal step is therefore essentially
  a ``no-op`` until V2 adds 4-bit / 6-bit variants — but we expose the
  interface NOW so the coordinator can be exercised end-to-end on real codecs.
* Score-cost surface is CACHED at construction time from a one-shot frontier
  measurement (the user supplies a ``score_at_bytes`` callback). The
  proximal step does NOT re-measure; this preserves the strict-scorer-rule.
* The marginal dScore/dByte is estimated by a finite-difference over the cached
  frontier samples.

CLAUDE.md compliance
--------------------
* COMPRESS-time only.
* Strict-scorer-rule: score_at_bytes is a CACHED surface; no live scorer load.
* No silent defaults.
* Tagged claims: any score-saving estimate is [prediction] until V2 measures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tac.joint_admm_coordinator import (
    ProximalStepResult,
    StreamProximalCodec,
)
from tac.pose_delta_codec import encode_pose_deltas


@dataclass
class PoseDeltaFrontierSample:
    """One sampled R(D) point on the pose stream.

    Fields
    ------
    delta_bits : int
        Quantisation width (currently only 8 implemented; declared here so
        V2 4-bit / 6-bit additions slot in without API change).
    bytes_used : int
        Encoded byte count at this delta_bits.
    score_cost : float
        Score-cost contribution at this operating point.
    """

    delta_bits: int
    bytes_used: int
    score_cost: float


class PoseDeltaProximalCodec:
    """StreamProximalCodec wrapper for ``tac.pose_delta_codec``.

    Construction
    ------------
    poses : torch.Tensor
        (N, pose_dim) tensor encoded by Lane PD at compress time.
    frontier : list[PoseDeltaFrontierSample]
        CACHED R(D) samples (one per delta_bits choice). Coordinator uses
        these to estimate score-cost + marginal dScore/dByte at the
        coordinator's requested target_bytes. Sorted by bytes_used asc.
    name : str, default "pose_delta"
        Stream identifier in coordinator logs.
    """

    def __init__(
        self,
        poses: torch.Tensor,
        frontier: list[PoseDeltaFrontierSample],
        name: str = "pose_delta",
    ) -> None:
        if poses.ndim != 2:
            raise ValueError(
                f"poses must be 2-D (N, pose_dim); got {tuple(poses.shape)}"
            )
        if not frontier:
            raise ValueError(
                "frontier must contain >=1 PoseDeltaFrontierSample; pass at "
                "least one (delta_bits=8, bytes_used=..., score_cost=...) "
                "from a one-shot encode_pose_deltas measurement."
            )
        self._poses = poses
        # Sort by bytes ascending so finite-difference marginals are well-defined.
        self._frontier = sorted(frontier, key=lambda s: s.bytes_used)
        self._name = name
        # PARADIGM-γ audit fix #4 (2026-05-06): cache the 8-bit
        # ``encode_pose_deltas`` result. The proximal_step sanity check below
        # called encode_pose_deltas on EVERY ADMM iteration (typically
        # 50-200 iters), each costing ~100 µs for 600 poses × 6 dims. Since
        # ``self._poses`` is immutable for the lifetime of this codec, the
        # encode is a pure function of constructor arguments — cache once.
        self._cached_encode_8bit: dict | None = None

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        """Return the frontier sample closest to ``target_bytes`` from below.

        Discrete codec: bytes_used is one of a small set; the "proximal step"
        picks the largest bytes_used <= target_bytes (or the smallest sample
        if target undershoots all). Marginal is finite-differenced from the
        adjacent sample.

        ``dual`` is currently unused — Lane PD has only one knob (delta_bits)
        and a single discrete level, so the dual cannot bias selection. V2
        wrapping with multiple bit-widths will use the dual to break ties
        between adjacent samples on the bytes/score frontier.
        """
        # Pick selected sample = largest bytes_used <= target_bytes.
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
        # Verify the encoder agrees with the cached bytes_used (cheap sanity
        # check that the cache is still valid for THIS poses tensor).
        if selected.delta_bits == 8:
            # We only validate the canonical 8-bit point; other bits are V2.
            try:
                # PARADIGM-γ audit #4: cache encode result; invariant on poses.
                if self._cached_encode_8bit is None:
                    self._cached_encode_8bit = encode_pose_deltas(
                        self._poses, delta_bits=8
                    )
                encoded = self._cached_encode_8bit
                # Rough byte estimate: anchor (12) + delta_scale (12) +
                # deltas_q (numel int8) + small dict overhead (~50).
                approx_bytes = (
                    2 * encoded["anchor"].numel()
                    + 2 * encoded["delta_scale"].numel()
                    + encoded["deltas_q"].numel()
                    + 50
                )
                # Frontier sample MUST be within 30% of the encoder's estimate
                # — if not, the cache is stale and the caller should rebuild.
                if (
                    abs(approx_bytes - selected.bytes_used)
                    > 0.3 * selected.bytes_used
                ):
                    raise RuntimeError(
                        f"PoseDeltaProximalCodec: cached frontier sample "
                        f"bytes_used={selected.bytes_used} disagrees with "
                        f"encoder estimate {approx_bytes} by >30%; rebuild "
                        f"the frontier with a fresh encode_pose_deltas call."
                    )
            except NotImplementedError:
                # delta_bits != 8 not yet implemented; skip the sanity check.
                pass
        return ProximalStepResult(
            encoded_bytes=int(selected.bytes_used),
            score_delta=float(selected.score_cost),
            marginal=float(marginal),
            state=selected,
        )


def build_pose_delta_frontier(
    poses: torch.Tensor,
    score_at_bytes: Callable[[int], float],
) -> list[PoseDeltaFrontierSample]:
    """Construct the cached frontier for a poses tensor.

    Args:
        poses: (N, pose_dim) tensor.
        score_at_bytes: cached function bytes -> score_cost. Expected to be
            evaluated against pre-recorded scorer outputs (e.g. an offline
            per-byte-budget sweep). NOT a live SegNet/PoseNet call.

    Returns:
        list with one sample (delta_bits=8). V2 will add more bit-widths.
    """
    encoded = encode_pose_deltas(poses, delta_bits=8)
    bytes_used = (
        2 * encoded["anchor"].numel()
        + 2 * encoded["delta_scale"].numel()
        + encoded["deltas_q"].numel()
        + 50
    )
    return [
        PoseDeltaFrontierSample(
            delta_bits=8,
            bytes_used=int(bytes_used),
            score_cost=float(score_at_bytes(bytes_used)),
        )
    ]


# Compile-time check that the wrapper actually conforms to the Protocol.
# (Catches accidental drift at import time, not just at run_admm time.)
_assert_protocol: StreamProximalCodec = PoseDeltaProximalCodec(
    poses=torch.zeros(2, 6),
    frontier=[PoseDeltaFrontierSample(delta_bits=8, bytes_used=100, score_cost=0.001)],
)


__all__ = [
    "PoseDeltaFrontierSample",
    "PoseDeltaProximalCodec",
    "build_pose_delta_frontier",
]
