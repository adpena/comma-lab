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

# ── wall-clock PROJECTION anchor (L45: verify THROUGHPUT + wall-clock, not just flags) ──
# The throughput micro-bench proves the ~17x fast path is ON; it does NOT tell the operator how
# many DAYS a 3000-epoch run will take. This MEASURED anchor lets the launch gate PROJECT total
# wall-clock (advisory; REFUSE only against an explicit budget). Value-provenance ladder
# (req-T): class-(3) MEASURED-ANCHOR, config-conditional.
#
# v7.3 round-2 DELTA (SEAL A2, seal_v73_r2_deepmath MAJOR + bugs REVISE-4): re-anchored to the
# STARTUP-AMORTIZED 3000-ep cadence, NOT the early-epoch incl-startup rate. The prior 3.62 was the
# ep~46-115 incl-startup cadence (two figures floated: ep46 → 167.85/46 = 3.649 ≈ 3.65; ledger H2 →
# 3.62), which OVER-counts startup for a 3000-ep run: incl-startup cadence(n) = r_ss + S/n falls as
# n grows, so using an early-n rate as the whole-run rate double-counts the one-time startup S. The
# deepmath offered 3.12 (from the memo's r_ss=3.1 LOWER BOUND); this session RE-DERIVED r_ss + S
# DIRECTLY from run-1's log (levelset_n600_crucible_v6_run1_20260708T095730Z verdict timestamps,
# launch 09:57:30Z):
#   * steady-state r_ss re-fit on the WIDER ep25→ep125 window (SEAL v7.4 round-3 DM-MINOR-1): the
#     single ep75→ep100 window gave 3.371, but the per-interval slopes BOUNCE 3.25–3.73 (fleet
#     thermal/jitter), so a one-window pick is noisy. The full ep25→ep125 average (MEASURED from
#     run-1's log: (483.13 − 137.77)/100) = 3.4537 min/ep is the more representative steady slope —
#     the 3.371 window happened to be a slow-adjacent-fast trough that under-counted by ~0.08.
#   * ramp-inclusive S (two-point fit at ep100 with the wider r_ss) = wall(ep100) − r_ss·100 =
#     396.62 − 345.37 = 51.25 min (the higher steady slope leaves LESS residual for one-time startup);
#   * amortized(3000) = r_ss + S/3000 = 3.4537 + 51.25/3000 = 3.4537 + 0.0171 = 3.47 min/ep.
# So the honest 3000-ep amortized cadence is 3.47 (was 3.39 on the narrow ep75→100 window; both far
# below the startup-over-counted 3.62), and anchoring here makes the refuse a TRUE ~15% gate (refuse @
# 3.47·1.15 = 3.99 min/ep = 15% above the honest cadence). DM-MINOR-1 direction: STRICTLY MORE
# CONSERVATIVE (the wider window is ~0.08 min/ep slower => the refuse ceiling tightens and the budget
# projection grows ~0.2 d — never lets a too-slow run through). REASONED DEVIATION from the synthesis's
# suggested 3.12: run-1's OWN measured steady slope contradicts the r_ss=3.1 lower bound the 3.12
# rested on; the value-provenance ladder forbids anchoring on a lower bound. rc=8's at-admission REAL
# SegNet bench remains the FINAL arbiter; this anchor sets the derived REFUSE ceiling. Re-fit if the
# run-1 startup/steady split changes or a lever batches the scorer forward (micro-batch>1 => fewer,
# larger forwards => a DIFFERENT min/ep; the anchor is B=1-accum-forward-conditional).
RUN1_MEASURED_MIN_PER_EP: float = 3.47  # crucible_v6 run-1 startup-AMORTIZED 3000-ep cadence, ep25->125 wider window (MEASURED; SEAL v7.4 r3 DM-MINOR-1; n600 B=1 accum)
RUN1_ANCHOR_SEGNET_MS: float = ON_REF_MS  # the run-1 machine's SegNet fwd+bwd micro-bench reference

# ── wall-clock BUDGET slack factor (req-T tagged) ──
# A DERIVED wall-clock budget = anchor min/ep x epochs x this slack (project_wall_clock_days x slack).
# The budget is a REFUSE CEILING (default-on gate), not a target: it fires when THIS machine's measured
# SegNet bench projects a run SLOWER than the run-1 anchor by more than the slack. Provenance
# (value-provenance ladder, req-T): HARDCODED-WITH-WAIVER-class magnitude with a MEASURED anchor for the
# ceiling. As of the v7.3 round-2 re-anchor the min/ep anchor is the STARTUP-AMORTIZED 3000-ep cadence
# (3.47; startup S is now amortized IN the anchor, not double-counted), so this slack is a slim PURE
# thermal/per-ep-jitter headroom and the gate is a TRUE >15% refuse (>15% slower than the honest
# amortized cadence => REFUSE) rather than the ~23% gate the un-amortized 3.62 anchor gave. RE-DERIVE if
# the run-1 startup/steady split changes or a lever batches the forward (micro-batch>1 => a different
# min/ep => a different implied ceiling).
WALL_CLOCK_SLACK_FACTOR: float = 1.15


def derive_wall_clock_budget_days(epochs: int,
                                  min_per_ep: float = RUN1_MEASURED_MIN_PER_EP,
                                  slack: float = WALL_CLOCK_SLACK_FACTOR) -> float:
    """DERIVE a config's wall-clock BUDGET (days) from the measured anchor x epochs x slack — the
    single SoT for both the typed-config REQUIRED field and the launcher's default-on fallback.
    ``budget = project_wall_clock_days(min_per_ep, epochs) * slack``. Pure. NEVER hand-pick a budget;
    compute it here so the ceiling tracks the anchor. Raises on non-positive inputs."""
    if epochs <= 0:
        raise ValueError(f"epochs must be positive to derive a wall-clock budget, got {epochs}")
    if min_per_ep <= 0.0 or slack <= 0.0:
        raise ValueError(f"min_per_ep/slack must be positive, got {min_per_ep}/{slack}")
    return project_wall_clock_days(min_per_ep, int(epochs)) * float(slack)


def implied_segnet_ms_ceiling(budget_days: float, epochs: int,
                              ref_min_per_ep: float = RUN1_MEASURED_MIN_PER_EP,
                              ref_segnet_ms: float = RUN1_ANCHOR_SEGNET_MS) -> float:
    """The MEASURED-bench ceiling implied by a declared budget (fix #3 framing): a machine whose
    SegNet fwd+bwd bench exceeds this projects a run OVER budget — REFUSE even when the ~17x env is
    present + the 700ms absolute throughput gate passes (catches a NON-env perf regression: kernel
    not loading, wrong device, thermal throttle). Inverts :func:`project_min_per_ep`:
    ``ceiling_ms = ref_segnet_ms * (implied_min_per_ep / ref_min_per_ep)`` where
    ``implied_min_per_ep = budget_days * 1440 / epochs``. Pure. Note: budget already carries the slack,
    so the ceiling does too. Mathematically equivalent to ``project_launch_wall_clock(...).over_budget``;
    exposed as an explicit ceiling for the coupling assertion + a legible refuse message."""
    if epochs <= 0 or budget_days <= 0.0:
        raise ValueError(f"epochs/budget_days must be positive, got {epochs}/{budget_days}")
    if ref_min_per_ep <= 0.0 or ref_segnet_ms <= 0.0:
        raise ValueError(f"ref_min_per_ep/ref_segnet_ms must be positive, got {ref_min_per_ep}/{ref_segnet_ms}")
    implied_min_per_ep = float(budget_days) * (60.0 * 24.0) / float(epochs)
    return float(ref_segnet_ms) * (implied_min_per_ep / float(ref_min_per_ep))


def project_min_per_ep(machine_segnet_ms: float,
                       ref_min_per_ep: float = RUN1_MEASURED_MIN_PER_EP,
                       ref_segnet_ms: float = RUN1_ANCHOR_SEGNET_MS) -> float:
    """PROJECT this machine's min/ep by SCALING the measured run-1 anchor by how much slower THIS
    machine's SegNet fwd+bwd micro-bench is vs the anchor's reference. The witness step is
    scorer-forward-BOUND (per-pair SegNet+PoseNet fwd+bwd dominates), so min/ep scales ~linearly
    with the SegNet micro-bench. HONEST: it is a projection off a same-class anchor, NOT a fresh
    per-ep measurement (labelled as such by the caller). Pure."""
    if ref_segnet_ms <= 0.0:
        raise ValueError(f"ref_segnet_ms must be positive, got {ref_segnet_ms}")
    return float(ref_min_per_ep) * (float(machine_segnet_ms) / float(ref_segnet_ms))


def project_wall_clock_days(min_per_ep: float, epochs: int) -> float:
    """Total projected wall-clock in DAYS for ``epochs`` at ``min_per_ep`` minutes/epoch. Pure."""
    if min_per_ep < 0.0 or epochs < 0:
        raise ValueError(f"min_per_ep/epochs must be non-negative, got {min_per_ep}/{epochs}")
    return float(min_per_ep) * float(epochs) / (60.0 * 24.0)


@dataclass(frozen=True)
class WallClockProjection:
    """Advisory wall-clock projection for a launch (L45). ``over_budget`` is None when no budget
    was supplied (pure advisory), else the boolean the gate REFUSES on."""

    min_per_ep: float
    epochs: int
    total_days: float
    budget_days: Optional[float]
    over_budget: Optional[bool]

    @property
    def detail(self) -> str:
        base = (f"projected {self.total_days:.2f} days "
                f"({self.min_per_ep:.2f} min/ep x {self.epochs} ep)")
        if self.budget_days is None:
            return base + " [advisory — no --wall-clock-budget-days]"
        verdict = "OVER" if self.over_budget else "within"
        return base + f" {verdict} budget {self.budget_days:.2f} days"


def project_launch_wall_clock(machine_segnet_ms: Optional[float], epochs: int,
                              budget_days: Optional[float] = None,
                              ref_min_per_ep: float = RUN1_MEASURED_MIN_PER_EP,
                              ref_segnet_ms: float = RUN1_ANCHOR_SEGNET_MS,
                              ) -> Optional[WallClockProjection]:
    """Compose the projection for the launch gate. Returns None when no SegNet measurement is
    available (the gate stays silent — never blocks on unavailability). ``over_budget`` is set
    only when ``budget_days`` is supplied. Pure."""
    if machine_segnet_ms is None:
        return None
    mpe = project_min_per_ep(machine_segnet_ms, ref_min_per_ep, ref_segnet_ms)
    days = project_wall_clock_days(mpe, epochs)
    over = None if budget_days is None else bool(days > float(budget_days))
    return WallClockProjection(min_per_ep=mpe, epochs=int(epochs), total_days=days,
                               budget_days=budget_days, over_budget=over)


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
