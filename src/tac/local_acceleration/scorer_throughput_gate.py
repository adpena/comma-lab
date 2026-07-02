# SPDX-License-Identifier: MIT
"""#224 Wave D — SegNet fwd+bwd THROUGHPUT gate for the witness launcher.

The launch gate must verify the ~17x custom-grouped-backward fast path is ACTUALLY
active on THIS machine, not merely that ``TAC_MLX_CUSTOM_GROUPED_BACKWARD`` is set
in the env. A flag can be set while the custom Metal kernel silently falls back
(missing kernel, adapter drift, cold cache) => the run grinds ~17x slower. The
canonical detector is a MEASURED one-shot SegNet fwd+bwd micro-bench, not a grep.

Measured separation (compute design pass, `[macOS-MLX advisory]`, B=8, 384x512):
  * custom-backward ON  ~ 396 ms   (the fast path)
  * custom-backward OFF ~ 6713 ms  (the reference accumulator)
=> a 700 ms absolute threshold cleanly separates the two (well below 6713, well
above the ~396 fast path + headroom for a busy machine).

Authority: ``[macOS-MLX advisory]``. TIMING ONLY — the micro-bench uses a synthetic
random input of the CORRECT shape ``(B, SEG_H, SEG_W, 3)``: wall-clock is FLOP-bound,
not input-content-bound, so synthetic input gives the same timing as the real cache
(this is NOT a score/verdict — no `score_claim`; the "synthetic-fixture" forbidden
class applies to score/verdict claims, not to a pure throughput measurement). The
gate NEVER produces a contest score and NEVER blocks a launch on unavailability
(only on a measured SLOW result).

Design: the heavy measurement (`measure_segnet_fwd_bwd_ms`) is isolated so the
verdict logic (`evaluate_throughput`) is unit-testable via an injected ``measure_fn``
with NO GPU / NO scorer weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

# Measured anchors (compute design pass 2026-07-02; [macOS-MLX advisory]). Overridable.
ON_REF_MS: float = 396.0      # custom-grouped-backward ON (fast path) SegNet fwd+bwd @ B=8
OFF_REF_MS: float = 6713.0    # custom-grouped-backward OFF (reference accumulator)
ABS_THRESHOLD_MS: float = 700.0   # sub-part 1: FAIL if the median fwd+bwd exceeds this
CEILING_MULT: float = 1.5         # sub-part 2: first-10 median must be <= mult * baseline
SEG_H: int = 384
SEG_W: int = 512


@dataclass(frozen=True)
class ThroughputVerdict:
    """Result of the launch-time throughput gate.

    ``status``:
      * ``"fast"``        — measured, within both the absolute + relative gates (OK to launch).
      * ``"slow"``        — measured, exceeds a gate => the custom fast path is NOT active (REFUSE).
      * ``"unavailable"`` — could not measure (MLX/scorer/GPU absent) => WARN, do not block.
    """

    status: str
    segnet_fwd_bwd_ms: Optional[float]
    abs_threshold_ms: float
    baseline_ms: float
    ceiling_mult: float
    within_abs: Optional[bool]
    within_ceiling: Optional[bool]
    reason: str = ""

    @property
    def ok(self) -> bool:
        """True when the launch may proceed (fast OR unmeasurable — never block on unavailability)."""
        return self.status in ("fast", "unavailable")

    @property
    def is_slow(self) -> bool:
        return self.status == "slow"


def step_time_within_ceiling(observed_ms: float, baseline_ms: float,
                             ceiling_mult: float = CEILING_MULT) -> bool:
    """Sub-part 2 (pure): the observed first-N median step-time must be <= mult * baseline.

    Catches a relative regression (a per-step overhead the absolute gate would still pass,
    e.g. a slow-but-under-700ms drift) — orthogonal to the absolute sub-part-1 gate."""
    if baseline_ms <= 0.0:
        raise ValueError(f"baseline_ms must be positive, got {baseline_ms}")
    return float(observed_ms) <= float(ceiling_mult) * float(baseline_ms)


def measure_segnet_fwd_bwd_ms(*, batch: int = 8, warmup: int = 2, iters: int = 10,
                              custom_backward: bool = True, upstream_dir: str = "upstream",
                              seg_h: int = SEG_H, seg_w: int = SEG_W, seed: int = 0) -> float:
    """MEASURE the median SegNet fwd+bwd wall-clock (ms) at B=``batch`` on MLX-GPU.

    Sets ``TAC_MLX_CUSTOM_GROUPED_BACKWARD`` per ``custom_backward`` (the fast path the
    launch uses is ``True``). Synthetic random input of shape ``(batch, seg_h, seg_w, 3)``
    (NHWC) — the SAME shape the trainer feeds ``adapter.segnet`` (``_render_R`` output is
    ``(1, SEG_H, SEG_W, 3)``). Raises on any unavailability (caller catches).
    """
    import os
    import sys
    from pathlib import Path

    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1" if custom_backward else "0"
    # defensive path setup: load_frozen_distortion_net does ``from modules import ...`` (the pinned
    # upstream snapshot), so ensure the repo's src + upstream are importable regardless of the caller's
    # PYTHONPATH (the launcher only puts src/tools on the path). REPO = <...>/src/tac/local_acceleration
    # -> parents[3]. If upstream is absent (e.g. a worktree without the snapshot) the import raises and
    # the caller (evaluate_throughput) records "unavailable" (WARN, never blocks).
    _repo = Path(__file__).resolve().parents[3]
    for _p in (str(_repo / "src"), str(_repo / upstream_dir), str(_repo / "upstream")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import numpy as np
    import mlx.core as mx

    from tac.local_acceleration.mlx_scorer_adapters import (
        temporary_mlx_device,
        torch_distortion_net_to_mlx,
    )
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    dist = load_frozen_distortion_net(upstream_dir=upstream_dir, device="cpu")
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch, seg_h, seg_w, 3)).astype(np.float32)
    with temporary_mlx_device("gpu"):
        adapter = torch_distortion_net_to_mlx(dist)
        x = mx.array(x_np)

        def seg_loss(z):
            return (adapter.segnet(z).astype(mx.float32) ** 2).mean()

        grad_fn = mx.value_and_grad(seg_loss)
        for _ in range(max(0, warmup)):
            mx.eval(grad_fn(x))
        ts = []
        for _ in range(max(1, iters)):
            mx.synchronize()
            t0 = time.perf_counter()
            out = grad_fn(x)
            mx.eval(out)
            mx.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return float(ts[len(ts) // 2])


def evaluate_throughput(*, abs_threshold_ms: float = ABS_THRESHOLD_MS,
                        baseline_ms: float = ON_REF_MS, ceiling_mult: float = CEILING_MULT,
                        batch: int = 8, iters: int = 10,
                        measure_fn: Optional[Callable[..., float]] = None) -> ThroughputVerdict:
    """Run the SegNet fwd+bwd micro-bench (custom fast path) and apply BOTH gates.

    * sub-part 1 (absolute): median <= ``abs_threshold_ms`` (700) — catches the ~6713ms
      custom-kernel-OFF footgun.
    * sub-part 2 (relative): median <= ``ceiling_mult`` * ``baseline_ms`` (1.5 * 396) —
      catches a slower-but-sub-700 drift.

    ``measure_fn`` (default :func:`measure_segnet_fwd_bwd_ms`) is injectable for GPU-free
    unit tests. On ANY measurement error the verdict is ``"unavailable"`` (never blocks)."""
    mfn = measure_fn or measure_segnet_fwd_bwd_ms
    try:
        ms = float(mfn(batch=batch, iters=iters, custom_backward=True))
    except Exception as exc:  # noqa: BLE001 — unavailability must NOT crash the launch
        return ThroughputVerdict(
            status="unavailable", segnet_fwd_bwd_ms=None, abs_threshold_ms=abs_threshold_ms,
            baseline_ms=baseline_ms, ceiling_mult=ceiling_mult, within_abs=None,
            within_ceiling=None, reason=f"{type(exc).__name__}: {exc}")
    within_abs = ms <= abs_threshold_ms
    within_ceiling = step_time_within_ceiling(ms, baseline_ms, ceiling_mult)
    status = "fast" if (within_abs and within_ceiling) else "slow"
    reason = "" if status == "fast" else (
        f"segnet fwd+bwd {ms:.1f}ms "
        f"{'exceeds abs threshold %.0fms' % abs_threshold_ms if not within_abs else ''}"
        f"{' and ' if (not within_abs and not within_ceiling) else ''}"
        f"{'exceeds %.1fx baseline %.0fms' % (ceiling_mult, baseline_ms) if not within_ceiling else ''}"
        " => custom-grouped-backward fast path NOT active (set TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 and "
        "verify the Metal kernel builds)")
    return ThroughputVerdict(
        status=status, segnet_fwd_bwd_ms=ms, abs_threshold_ms=abs_threshold_ms,
        baseline_ms=baseline_ms, ceiling_mult=ceiling_mult, within_abs=within_abs,
        within_ceiling=within_ceiling, reason=reason)


def assert_compile_step_bit_identical(loss_and_grad_fn, args: tuple, **kw) -> bool:
    """Sub-part 3: if a ``--compile-step`` path is wired, the compiled step MUST match the
    uncompiled step (same math) + be deterministic. Thin wrapper over the canonical
    :func:`tac.local_acceleration.mlx_compile_step.assert_compile_bit_identical` returning a
    bool verdict (it RAISES on failure). Called by the launch gate ONLY when the emitted
    flags include a ``--compile*`` flag (else there is nothing to assert)."""
    from tac.local_acceleration.mlx_compile_step import assert_compile_bit_identical

    assert_compile_bit_identical(loss_and_grad_fn, args, **kw)
    return True
