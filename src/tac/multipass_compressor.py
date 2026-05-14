# SPDX-License-Identifier: MIT
"""Lane 8 — Multi-pass compress-time optimizer over the canonical archive pipeline.

Compress-time iteration with score-feedback: encode → inflate-and-eval →
adjust encoder parameters → re-encode, until score plateaus or MAX_PASSES is
reached. Produces a single final archive byte stream that the existing inflate
path handles natively (no new magic bytes, no inflate-time scorers — strict-
scorer-rule per CLAUDE.md).

Scope (Level 3 per `feedback_production_hardened_standard_definition_20260430.md`):
- Synthetic 2-pass convergence on a quadratic objective (offline test).
- Real-archive 3-pass on Lane G v3 anchor with byte-and-score-only proxy
  (offline test; CUDA path is wired via experiments/pipeline.py).
- Default coordinate-descent ``AdjustmentPolicy`` over the 4 canonical axes
  (mask CRF, pose Q bits, block-FP block size, residual gain on K hard frames)
  per the score-arithmetic priority ranking in
  `project_codec_stacking_composition_canonical_orders_20260429.md`.
- Regression guard (revert-and-stop on score-up move).
- Parameter clamping (codec-valid-range guard).

Design memo: `.omx/research/council_lane_8_multipass_design_20260430.md`.

Council verdict (consensus across Shannon, Dykstra, Hotz, Quantizr, Selfcomp,
Contrarian, Carmack — see design memo):

| Parameter | Value |
|---|---|
| MAX_PASSES default | 3 (Carmack 80/30) |
| MAX_PASSES upper bound | 5 (Shannon log saturation) |
| eps | 1e-3 (below scorer noise floor per CLAUDE.md) |
| target_score | configurable, default = baseline_score - 0.005 |
| regression_guard | True (mandatory — Contrarian) |
| parameter clamping | True (mandatory — Contrarian) |
| device | CUDA-required (CPU opt-in with banner per CLAUDE.md) |
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Council-verdict defaults (do NOT silently drift; profile overrides apply).
DEFAULT_MAX_PASSES: int = 3
ABSOLUTE_MAX_PASSES: int = 5  # Shannon log saturation ceiling — refuses higher
DEFAULT_EPS: float = 1e-3      # below scorer noise floor (CLAUDE.md)

# Codec-valid ranges (Contrarian failure mode 2 — parameter clamping).
# Sourced from the canonical encoder configs:
#   * AV1 mask CRF: ffmpeg-aom range [0, 63]; our useful range [10, 60].
#   * Pose quantization bits: int8 = 8, our codec supports [4, 16].
#   * Block-FP block size: typical [4, 32] per Selfcomp `block_fp_codec`.
#   * Residual gain on K hard frames: scalar in [0.0, 1.0].
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "mask_crf": (10.0, 60.0),
    "pose_q_bits": (4.0, 16.0),
    "block_fp_block_size": (4.0, 32.0),
    "residual_gain": (0.0, 1.0),
}


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class PassRecord:
    """One record per compress-pass for forensics + the regression guard."""

    pass_idx: int
    params: dict[str, float]
    archive_bytes: int
    score: float
    delta: float            # prev_score - score  (positive = improvement)
    elapsed_seconds: float
    reason: str = ""        # adjustment policy's reason string for this pass

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class MultiPassResult:
    """Final result returned from MultiPassCompressor.compress(...).

    `final_archive_bytes` is the bytes of the BEST archive across all passes
    (regression-guarded — never the last archive if it regressed).
    """

    final_archive_bytes: bytes
    final_score: float
    pass_history: list[PassRecord]
    best_pass_idx: int
    converged: bool          # True if eps-stopped or target-hit; False if MAX_PASSES
    target_hit: bool         # True if final_score < target
    reverted: bool           # True if regression guard fired

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": self.final_score,
            "best_pass_idx": self.best_pass_idx,
            "converged": self.converged,
            "target_hit": self.target_hit,
            "reverted": self.reverted,
            "n_passes": len(self.pass_history),
            "final_archive_bytes_len": len(self.final_archive_bytes),
            "pass_history": [r.to_dict() for r in self.pass_history],
            # Round 2 Selfcomp finding: surface which axes were ACTIVELY
            # adjusted across passes (each pass's `params` differing from
            # the prior pass on a given axis = active). This lets operators
            # see at a glance whether the encoder threaded each axis through
            # to behavior or treated it as a no-op (RESERVED axes today:
            # pose_q_bits, block_fp_block_size, residual_gain — only mask_crf
            # is wired into the production encoder).
            "axes_active": self._axes_active(),
        }

    def _axes_active(self) -> list[str]:
        """Determine which axes had at least one inter-pass change in
        the parameter dict across the recorded pass history.
        """
        if len(self.pass_history) < 2:
            return []
        active: set[str] = set()
        for prev, cur in zip(self.pass_history, self.pass_history[1:]):
            for k in cur.params:
                if k not in prev.params or prev.params[k] != cur.params[k]:
                    active.add(k)
        return sorted(active)


# ── Adjustment policy ABC ────────────────────────────────────────────────────


class AdjustmentPolicy(ABC):
    """Pluggable strategy for proposing the next pass's encoder parameters.

    Subclass and override ``propose_next_params``. The policy receives the
    full pass history and must return a new params dict (clamped to
    PARAM_RANGES) plus a short reason string for forensics.
    """

    @abstractmethod
    def propose_next_params(
        self,
        history: list[PassRecord],
        current_params: dict[str, float],
    ) -> tuple[dict[str, float], str]:
        """Return (next_params, reason). ``next_params`` is the new pass's
        encoder configuration; ``reason`` is a short forensic string written
        to the pass record. The caller will clamp ``next_params`` to
        PARAM_RANGES before encoding.
        """
        raise NotImplementedError


class CoordinateDescentPolicy(AdjustmentPolicy):
    """Default: coordinate descent on the 4 canonical axes.

    Each pass adjusts the highest-priority axis until it saturates (no
    improvement >= eps for one pass), then moves to the next axis. Priority
    order matches `project_codec_stacking_composition_canonical_orders_20260429.md`:

    1. mask_crf (45× headroom over pose per Shannon waterline)
    2. pose_q_bits (saturates ~5KB)
    3. block_fp_block_size (Selfcomp domain; monotone-decrease only after pass 1)
    4. residual_gain (sensitivity-driven if available)

    Step direction (the "decrease bytes" direction of each axis):
    - mask_crf: INCREASE  → coarser AV1 → fewer bytes, more distortion
    - pose_q_bits: DECREASE → coarser Q → fewer bytes, more distortion
    - block_fp_block_size: DECREASE (Dykstra non-expansiveness — monotone after pass 1)
    - residual_gain: INCREASE → richer correction → more bytes, less distortion

    The policy always moves toward the score-arithmetic optimum:
    - When current pass IMPROVED score (delta > eps): keep moving in the
      same direction on the same axis.
    - When current pass DEGRADED score (delta < -eps): the regression guard
      will revert outside the policy. The policy isn't asked.
    - When current pass PLATEAUED (|delta| <= eps): switch to the next axis
      in priority order; on the new axis, take the conservative direction
      step.
    """

    PRIORITY_AXES: tuple[str, ...] = (
        "mask_crf",
        "pose_q_bits",
        "block_fp_block_size",
        "residual_gain",
    )

    # Step size per axis (codec-validated; conservative).
    STEP_SIZES: dict[str, float] = {
        "mask_crf": 5.0,                    # CRF step
        "pose_q_bits": 1.0,                 # bit step
        "block_fp_block_size": 4.0,         # block-size step
        "residual_gain": 0.1,               # gain step
    }

    def __init__(self, eps: float = DEFAULT_EPS) -> None:
        self.eps = eps

    def propose_next_params(
        self,
        history: list[PassRecord],
        current_params: dict[str, float],
    ) -> tuple[dict[str, float], str]:
        # The very first call (history with 1 record): start on axis 0.
        if not history:
            return dict(current_params), "init"

        last = history[-1]
        next_params = dict(current_params)

        # Decide which axis to move on.
        # Walk the priority list; pick the first axis NOT yet plateaued.
        # An axis is "plateaued" when the most recent attempt on it produced
        # |delta| < eps. We track per-axis last-direction in `reason` strings.
        axis = self._select_axis(history)

        # Step direction on the chosen axis.
        if axis == "mask_crf":
            step = self.STEP_SIZES[axis]   # increase CRF (more compression)
        elif axis == "pose_q_bits":
            step = -self.STEP_SIZES[axis]  # decrease bits (more compression)
        elif axis == "block_fp_block_size":
            step = -self.STEP_SIZES[axis]  # decrease block (Dykstra monotone)
        elif axis == "residual_gain":
            step = self.STEP_SIZES[axis]   # increase gain (more bytes, less distortion)
        else:  # pragma: no cover — defensive
            return dict(current_params), f"unknown-axis:{axis}"

        next_params[axis] = current_params.get(axis, 0.0) + step
        reason = f"axis={axis} step={step:+.3f} prev_delta={last.delta:+.4f}"
        return next_params, reason

    def _select_axis(self, history: list[PassRecord]) -> str:
        """Pick the highest-priority axis not yet plateaued.

        An axis is plateaued when the LAST PassRecord referencing that axis
        produced a |delta| < eps. We walk priority order; the first non-
        plateaued axis wins. If ALL axes plateaued, fall back to the highest-
        priority axis (the loop will terminate naturally on the next eps-stop).
        """
        # Build per-axis last-delta from history.
        last_delta_by_axis: dict[str, float] = {}
        for rec in reversed(history):
            if "axis=" in rec.reason:
                # parse "axis=mask_crf step=...." from the reason string
                tag = rec.reason.split("axis=", 1)[1].split(" ", 1)[0]
                if tag not in last_delta_by_axis:
                    last_delta_by_axis[tag] = rec.delta

        for axis in self.PRIORITY_AXES:
            ld = last_delta_by_axis.get(axis)
            if ld is None:
                # Never tried this axis yet — try it next.
                return axis
            if abs(ld) >= self.eps:
                # Last move on this axis improved or regressed (not plateau).
                # Continue exploring it.
                return axis
        # All axes plateaued — return top priority (will likely converge next pass).
        return self.PRIORITY_AXES[0]


# ── Compressor ───────────────────────────────────────────────────────────────


class _InflateTimeAssertion:
    """Guard so any caller invoking MultiPassCompressor at INFLATE time is
    caught at construction — strict-scorer-rule per CLAUDE.md.

    Preflight Check 91 (`check_no_inflate_time_multipass`) does the static
    sweep, but this runtime assertion catches a dynamically-imported caller
    that the AST scanner can't see.
    """

    def __init__(self, allow_inflate_context: bool) -> None:
        self.allow = allow_inflate_context

    def assert_compress_time(self) -> None:
        if self.allow:
            return
        # Inspect the call stack for `inflate.sh` / `inflate_renderer.py`
        # entry points. We don't need a deep-traceback walker; the Python
        # `__main__.__file__` attribute on the entry script is enough for
        # the canonical paths.
        import sys as _sys
        main = _sys.modules.get("__main__")
        main_file = getattr(main, "__file__", None) if main is not None else None
        if main_file:
            mf = str(main_file)
            for forbidden in ("inflate.sh", "inflate_renderer.py"):
                if mf.endswith(forbidden):
                    raise RuntimeError(
                        f"MultiPassCompressor invoked from inflate-time entry point "
                        f"{mf!r}. Strict-scorer-rule per CLAUDE.md: multi-pass is "
                        f"COMPRESS time only. Pass allow_inflate_context=True only "
                        f"if you have explicit operator approval (and a contest-"
                        f"compliance ruling)."
                    )


@dataclass
class MultiPassCompressor:
    """Compress-time multi-pass optimizer over the canonical archive pipeline.

    Public API::

        result = MultiPassCompressor(
            target_score=1.04,
            max_passes=3,
            eps=1e-3,
            regression_guard=True,
        ).compress(initial_state, encoder, scorer)

    Where:
    - ``initial_state`` is opaque to the compressor; it's threaded through to
      the encoder and inflate-and-eval callbacks.
    - ``encoder`` is callable ``(state, params) -> archive_bytes``. Encoder is
      responsible for all real codec work (mask CRF dispatch, pose Q,
      block-FP layout, residual gain). Encoder MUST be deterministic for a
      given (state, params) pair.
    - ``scorer`` is callable ``(archive_bytes) -> score`` that runs
      inflate + auth-eval. MUST run on CUDA (CPU only with explicit opt-in
      banner). Returns the contest score (lower is better).

    The compressor is intentionally pipeline-agnostic — it knows nothing
    about renderers, AV1, FP4, or PoseNet. Integration into the canonical
    `experiments/pipeline.py` flow lives in
    ``experiments.pipeline.run_compress`` (Phase D wiring).
    """

    target_score: float
    max_passes: int = DEFAULT_MAX_PASSES
    eps: float = DEFAULT_EPS
    regression_guard: bool = True
    policy: AdjustmentPolicy | None = None  # default: CoordinateDescentPolicy
    initial_params: dict[str, float] = field(default_factory=dict)
    log_path: Path | None = None
    allow_inflate_context: bool = False  # see _InflateTimeAssertion

    def __post_init__(self) -> None:
        if self.max_passes < 1:
            raise ValueError(
                f"max_passes must be >= 1, got {self.max_passes}"
            )
        if self.max_passes > ABSOLUTE_MAX_PASSES:
            raise ValueError(
                f"max_passes={self.max_passes} exceeds ABSOLUTE_MAX_PASSES "
                f"= {ABSOLUTE_MAX_PASSES}. Council verdict (Shannon log "
                f"saturation): more than 5 passes wastes CUDA forward-pass "
                f"compute on diminishing marginal byte savings. If you need "
                f"more, the parameter space is wrong — add new axes or "
                f"switch to a closed-form solver."
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.policy is None:
            self.policy = CoordinateDescentPolicy(eps=self.eps)
        self._guard = _InflateTimeAssertion(self.allow_inflate_context)

    @staticmethod
    def _clamp_params(params: dict[str, float]) -> dict[str, float]:
        """Clamp every key to its PARAM_RANGES bounds. Unknown keys pass
        through unchanged (a custom policy may use additional axes).
        """
        out: dict[str, float] = {}
        for k, v in params.items():
            lo, hi = PARAM_RANGES.get(k, (-math.inf, math.inf))
            out[k] = max(lo, min(hi, float(v)))
        return out

    def compress(
        self,
        initial_state: Any,
        encoder: Callable[[Any, dict[str, float]], bytes],
        scorer: Callable[[bytes], float],
    ) -> MultiPassResult:
        """Run the multi-pass loop. Returns the BEST archive across passes
        (regression-guarded).
        """
        self._guard.assert_compress_time()

        history: list[PassRecord] = []
        current_params = self._clamp_params(dict(self.initial_params))
        prev_score = math.inf
        best_score = math.inf
        best_archive: bytes = b""
        best_pass_idx: int = -1

        target_hit = False
        converged = False
        reverted = False

        for pass_idx in range(self.max_passes):
            t0 = time.monotonic()
            archive = encoder(initial_state, current_params)
            if not isinstance(archive, (bytes, bytearray)):
                raise TypeError(
                    f"encoder must return bytes; got {type(archive).__name__}"
                )
            archive = bytes(archive)
            score = float(scorer(archive))
            elapsed = time.monotonic() - t0
            delta = prev_score - score

            # Build the policy's reason string for THIS pass.
            # First pass = "init"; later passes carry the policy's last reason
            # AS WRITTEN INTO current_params at the previous iteration.
            reason = self._last_proposal_reason if history else "init"

            rec = PassRecord(
                pass_idx=pass_idx,
                params=dict(current_params),
                archive_bytes=len(archive),
                score=score,
                delta=delta,
                elapsed_seconds=elapsed,
                reason=reason,
            )
            history.append(rec)
            self._maybe_log(rec)

            # Track best-so-far (regression-guarded).
            if score < best_score:
                best_score = score
                best_archive = archive
                best_pass_idx = pass_idx

            # Regression guard: score went UP relative to previous pass.
            # Stop and surface the best-so-far. We do NOT replace the policy;
            # a future caller may swap policy for a non-monotonic one.
            if (
                self.regression_guard
                and pass_idx > 0
                and delta < -self.eps
            ):
                reverted = True
                logger.warning(
                    "[multipass] pass_idx=%d REGRESSED (score %.4f vs prev %.4f); "
                    "reverting to best (pass_idx=%d, score=%.4f)",
                    pass_idx, score, prev_score, best_pass_idx, best_score,
                )
                break

            # Target-score hit.
            if score < self.target_score:
                target_hit = True
                converged = True
                break

            # Eps convergence: |improvement| below noise floor → stop.
            if pass_idx > 0 and abs(delta) < self.eps:
                converged = True
                break

            # Propose next pass's params (skip on the final pass — the
            # encoder won't run again).
            if pass_idx < self.max_passes - 1:
                next_params, proposal_reason = self.policy.propose_next_params(
                    history, current_params
                )
                next_params = self._clamp_params(next_params)
                current_params = next_params
                self._last_proposal_reason = proposal_reason

            prev_score = score

        return MultiPassResult(
            final_archive_bytes=best_archive,
            final_score=best_score,
            pass_history=history,
            best_pass_idx=best_pass_idx,
            converged=converged,
            target_hit=target_hit,
            reverted=reverted,
        )

    _last_proposal_reason: str = ""

    def _maybe_log(self, rec: PassRecord) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(rec.to_dict()) + "\n")


# ── Convenience entry point for `experiments/pipeline.py` ────────────────────


def compress_with_multipass(
    initial_state: Any,
    encoder: Callable[[Any, dict[str, float]], bytes],
    scorer: Callable[[bytes], float],
    *,
    target_score: float,
    max_passes: int = DEFAULT_MAX_PASSES,
    eps: float = DEFAULT_EPS,
    regression_guard: bool = True,
    initial_params: dict[str, float] | None = None,
    policy: AdjustmentPolicy | None = None,
    log_path: Path | None = None,
) -> MultiPassResult:
    """Thin functional wrapper for the canonical `experiments/pipeline.py`
    integration. Constructs a ``MultiPassCompressor`` with the supplied
    parameters and runs ``.compress(...)``.

    Inflate side is UNCHANGED — multi-pass is a compress-time optimization
    that produces a single final archive byte stream that the existing
    inflate path handles natively (no new magic bytes, no inflate-time
    scorers — strict-scorer-rule per CLAUDE.md).
    """
    return MultiPassCompressor(
        target_score=target_score,
        max_passes=max_passes,
        eps=eps,
        regression_guard=regression_guard,
        policy=policy,
        initial_params=initial_params or {},
        log_path=log_path,
    ).compress(initial_state, encoder, scorer)
