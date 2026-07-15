# SPDX-License-Identifier: MIT
"""AutoClip percentile gradient-clip law — the #B-4 grad-clip cure state machine.

MEASURED TELEMETRY this responds to (audit ``.omx/research/v9_missing_signal_constants_
audit_20260715.md`` §A-1, C0 run ``levelset_n600_witness_20260715T095030Z``):
``--grad-clip 0.5`` is SATURATED — ``grad_clip_activation`` rows ep1-39 show global
``frac_clipped=1.0`` at EVERY accum step with ``norm_mean≈5.9-6.2`` (max 17.5).

HONEST MECHANISM STATE (fresh-eyes F5 correction, 2026-07-15): the original
"effective step ≈ lr/12, LR cosine dethroned" reading of that telemetry is REFUTED
on the C0 (per-param normalize) lineage — per-param normalize runs on the
already-clipped tree and divides out ANY uniform norm scaling, so the saturation
was INERT there (telemetry != mechanism;
``perparam_normalize_masks_all_norm_clipping_c0_confound_20260715``). The lr/12
magnitude mechanism CAN bind only on normalize-none arms, where the clip scale
actually reaches the applied update (formulation-scoped; the n24 magnitude-law
A/B arms are exactly this lineage test — armC normalize-none+fixed vs armB
normalize-none+autoclip attribution owed). The trainer REFUSES the
autoclip x per-param-normalize composition outright (armed-but-inert lever =
orphaned signal).

THE LAW (AutoClip, Seetharaman et al., arXiv:2007.14469): at each optimizer step,
observe the gradient norm into a history, then clip at the p-th percentile of the
observed history — the threshold ADAPTS to the realized norm distribution, so
clipping suppresses only spikes (the guard's real job) instead of silently
re-parametrizing every step into normalized flow. ZClip (arXiv:2504.02507) is the
z-score spike sibling; the principled successor is the #500 categorical-Fisher
trust region which subsumes clipping entirely (ticket 1 in
``tac.witness_dsl.adaptivization_tickets_20260715``).

Semantics (paper-faithful, deterministic):
  * ``observe(gnorm)`` FIRST (finite norms only — a nonfinite norm is already a
    spike-guard skip and must never poison the history), THEN ``threshold()``
    includes the current step's norm, exactly as in the paper's Algorithm 1.
  * During warmup (fewer than ``warmup_steps`` observations) the threshold is the
    FIXED fallback clip (the incumbent ``--grad-clip`` value) — the percentile of
    a near-empty history is noise, and the fallback preserves the sealed guard.
  * The history is a fixed-size ring buffer (``window`` most-recent finite norms)
    so the threshold tracks the CURRENT stage's norm scale across curriculum
    stages (the constant 0.5 was never re-validated at l7/Muon/finisher).
  * ``np.percentile`` with the default linear interpolation — deterministic,
    bit-reproducible from the same history on any host (numpy-fp64 spine).

RESUME (P0): the ring + cursor + count serialize via ``state_arrays`` /
``restore_from_cfg`` under the ``__acl_`` prefix (canonical resume-registry
protocol; additive + legacy-compatible: a sidecar with no ``__acl_*`` keys
restores to a fresh state, and fixed-mode runs persist NOTHING => byte-identical
sidecars when the lever is off).

Equations leg: ``tac.canonical_equations.autoclip_percentile_grad_clip_20260715``
(``autoclip_percentile_threshold_v1``). DSL leg:
``tac.witness_dsl.curriculum_dsl.AdaptiveGradClip`` (``--grad-clip-mode autoclip``).

means != ends: this module is a MEANS. It makes NO score claim; every efficacy
number is owed to a bounded n24 A/B (gradient-quality + no-flicker admission bar
per ``feedback_wallclock_to_target_joint_objective_drift_ok_if_gradient_good_
20260715``) and, for any promotion claim, to a byte-closed exact n600 row.
Pointer UNMOVED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

AUTOCLIP_RESUME_PREFIX = "__acl_"
GRAD_CLIP_MODES = ("fixed", "autoclip")


def _finite_positive(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return out


@dataclass
class AutoClipPercentileState:
    """One AutoClip percentile threshold over a ring buffer of gradient norms."""

    percentile: float = 10.0
    window: int = 1000
    warmup_steps: int = 10
    fallback_clip: float = 0.5
    _ring: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _count: int = 0   # total finite observations ever (monotone)
    _pos: int = 0     # next write cursor in the ring

    def __post_init__(self) -> None:
        if not (0.0 < float(self.percentile) < 100.0):
            raise ValueError(f"percentile must be in (0,100), got {self.percentile!r}")
        if int(self.window) < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}")
        if int(self.warmup_steps) < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {self.warmup_steps!r}")
        _finite_positive(self.fallback_clip, name="fallback_clip")
        if self._ring is None:
            self._ring = np.zeros(int(self.window), dtype=np.float64)
        else:
            self._ring = np.asarray(self._ring, dtype=np.float64)
            if self._ring.shape != (int(self.window),):
                raise ValueError(
                    f"ring shape {self._ring.shape} != (window={int(self.window)},)")

    @property
    def filled(self) -> int:
        return min(int(self._count), int(self.window))

    def observe(self, gnorm: float) -> bool:
        """Record one gradient norm. Nonfinite/negative norms are SKIPPED (they are
        spike-guard territory; poisoning the percentile with inf would disarm the
        clip forever). Returns True iff the norm was recorded."""
        g = float(gnorm)
        if not np.isfinite(g) or g < 0.0:
            return False
        self._ring[self._pos] = g
        self._pos = (self._pos + 1) % int(self.window)
        self._count += 1
        return True

    def threshold(self) -> float:
        """The clip threshold for the CURRENT step (call AFTER observe()).

        Warmup (< warmup_steps observations) => the fixed fallback clip.
        Else => the p-th percentile of the filled ring (deterministic linear
        interpolation)."""
        if self._count < int(self.warmup_steps):
            return float(self.fallback_clip)
        return float(np.percentile(self._ring[: self.filled], float(self.percentile)))

    # ---- canonical resume-registry protocol (additive, legacy-compatible) ----
    def state_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        if self._count == 0:
            return {}  # fresh state persists nothing (byte-identical sidecar)
        return {
            prefix + "ring": np.asarray(self._ring, np.float64),
            prefix + "count": np.asarray(int(self._count), np.int64),
            prefix + "pos": np.asarray(int(self._pos), np.int64),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        key = prefix + "ring"
        if key not in cfg:
            return False  # legacy sidecar: fresh state
        ring = np.asarray(cfg[key], np.float64)
        if ring.shape != (int(self.window),):
            raise ValueError(
                f"autoclip resume ring shape {ring.shape} != configured window "
                f"({int(self.window)},) — resuming with a CHANGED --grad-clip-window is a "
                "config divergence; match the window or start fresh")
        self._ring = ring
        self._count = int(np.asarray(cfg[prefix + "count"]).item())
        self._pos = int(np.asarray(cfg[prefix + "pos"]).item())
        return True


@dataclass
class AutoClipController:
    """Trainer-facing AutoClip controller: one GLOBAL state + optional lazily-built
    PER-GROUP states (the ``--per-group-grad-clip`` C4-confound sibling: each
    top-level parameter group gets its OWN percentile threshold so a volatile
    regularizer gradient cannot throttle the seg/pose groups via a shared scale).

    Also accumulates per-epoch telemetry (applied thresholds + clip events) for a
    score-neutral ``grad_clip_autoclip`` row, reset by :meth:`row`."""

    percentile: float = 10.0
    window: int = 1000
    warmup_steps: int = 10
    fallback_clip: float = 0.5
    global_state: AutoClipPercentileState = field(default=None, repr=False)  # type: ignore[assignment]
    group_states: dict[str, AutoClipPercentileState] = field(default_factory=dict)
    _ep_thresholds: list[float] = field(default_factory=list, repr=False)
    _ep_clipped: int = 0
    _ep_steps: int = 0

    def __post_init__(self) -> None:
        if self.global_state is None:
            self.global_state = self._new_state()

    def _new_state(self) -> AutoClipPercentileState:
        return AutoClipPercentileState(
            percentile=float(self.percentile), window=int(self.window),
            warmup_steps=int(self.warmup_steps), fallback_clip=float(self.fallback_clip))

    def group(self, name: str) -> AutoClipPercentileState:
        st = self.group_states.get(str(name))
        if st is None:
            st = self.group_states[str(name)] = self._new_state()
        return st

    def note_step(self, threshold: float, clipped: bool) -> None:
        self._ep_thresholds.append(float(threshold))
        self._ep_steps += 1
        if clipped:
            self._ep_clipped += 1

    def row(self, *, epoch: int) -> dict[str, Any]:
        """One per-epoch telemetry row; RESETS the epoch accumulators."""
        th = np.asarray(self._ep_thresholds, np.float64)
        out = {
            "stage": "grad_clip_autoclip",
            "epoch": int(epoch),
            "ep": int(epoch),
            "mode": "autoclip",
            "percentile": float(self.percentile),
            "window": int(self.window),
            "warmup_steps": int(self.warmup_steps),
            "fallback_clip": float(self.fallback_clip),
            "steps": int(self._ep_steps),
            "frac_clipped": (float(self._ep_clipped / self._ep_steps)
                             if self._ep_steps else 0.0),
            "clip_t_min": (float(th.min()) if th.size else None),
            "clip_t_mean": (float(th.mean()) if th.size else None),
            "clip_t_max": (float(th.max()) if th.size else None),
            "history_filled": int(self.global_state.filled),
            "warmed_up": bool(self.global_state._count >= int(self.warmup_steps)),
            "score_neutral": True,
        }
        self._ep_thresholds = []
        self._ep_clipped = 0
        self._ep_steps = 0
        return out

    # ---- canonical resume-registry protocol (aggregates global + groups) ----
    def state_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        out = self.global_state.state_arrays(prefix + "g_")
        for name in sorted(self.group_states):
            out.update(self.group_states[name].state_arrays(prefix + f"grp_{name}_"))
        if out:
            out[prefix + "groups"] = np.asarray(
                ",".join(sorted(self.group_states)) or "")
        return out

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        restored = self.global_state.restore_from_cfg(prefix + "g_", cfg)
        gkey = prefix + "groups"
        if gkey in cfg:
            raw = cfg[gkey]
            names = str(raw.item() if hasattr(raw, "item") else raw)
            for name in filter(None, names.split(",")):
                restored = self.group(name).restore_from_cfg(
                    prefix + f"grp_{name}_", cfg) or restored
        return restored


__all__ = [
    "AUTOCLIP_RESUME_PREFIX",
    "GRAD_CLIP_MODES",
    "AutoClipController",
    "AutoClipPercentileState",
]
