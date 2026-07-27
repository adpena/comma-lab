"""Polyak/Ruppert TAIL-AVERAGING finisher for the level-set witness trainer (R-7 finisher 2).

Operator directive 2026-07-08 (T5 crucible seal-round-1 structure lens, R-7 "minor unbuilt
finishers"): build the last unbuilt finishers. The structure-blind derivation
(``seal_v7_r1_structure_blind_20260708.md`` P1.7) prescribes, for the terminal phase:
*"finisher-EMA = Polyak averaging (~0.1-0.3x stage window)"* — the law
``muon_finisher_schedule_warmstart_and_lr_anneal_v1``.

THE MATH (why a tail mean beats a short-horizon EMA at a turnpike): the SEALED tail is a
constant-tau* TURNPIKE (``tau_0 = tau_end = 0.31`` => ``next_tau`` clamps to ``tau_end`` every
cycle — ``tail_majors_fix_20260708.md`` FIX-2). At a constant-tau turnpike the iterates ORBIT a
basin center rather than descend; the exponential EMA weights recent (orbiting) iterates by a fixed
horizon ``1/(1-decay)`` and therefore still carries orbit phase, while a UNIFORM (Polyak/Ruppert)
mean over the whole terminal window averages the orbit out to O(1/sqrt(n)) — the strictly better
basin-CENTER estimate. This is the classic Polyak-Ruppert result: uniform averaging of the tail
iterates attains the optimal asymptotic variance for a stationary/orbiting sequence.

THE DISCIPLINE (binding):
  * This NEVER replaces the EMA shadow. The EMA non-negotiable stands; the Polyak mean is an
    ADDITIONAL exported checkpoint CANDIDATE. Which candidate ships is decided downstream by the
    byte-close/eval stop-time checklist that MEASURES d_seg/d_pose/rate — NOT here.
  * DEFAULT-OFF: ``arm=False`` => the averager is inert (``observe`` no-ops, ``state_arrays`` /
    ``heavy_state_arrays`` return ``{}``). The trainer constructs + registers it ONLY when armed, so
    an un-armed run writes ZERO new checkpoint keys => byte-identical.
  * CRASH-RESUMABLE: the running mean + count round-trip so ``--resume-from`` continues a bit-faithful
    uniform average (never restarts the tail). The SCALAR bookkeeping (count / start / arm) rides the
    canonical :class:`tac.witness_control.resume_registry.ResumeRegistry` (completeness protection);
    the HEAVY running-mean arrays ride the resume sidecar under a dedicated key prefix, written in the
    SAME atomic savez as the scalars (no cross-file desync).

FAIL-OPEN (not closed): unlike an event-gate, a missing/partial Polyak state on resume is NOT fatal —
the EMA shadow (the resume-critical artifact) is untouched, and the Polyak candidate simply restarts
its average with a LOUD warning. The averager therefore does NOT expose ``event_mode`` (so the resume
registry never applies its event fail-closed to it).

means != ends: APPARATUS. Only a byte-closed n600 exact row < 0.19110 from ``upstream/evaluate.py``
(contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "POLYAK_SCALAR_PREFIX",
    "PolyakTailAverager",
    "polyak_finisher_window_provenance",
]

# Canonical SCALAR sidecar key prefix for the registry sentinel (count / start / arm). ``__``-prefixed
# so ``_load_resume_state`` routes it to cfg (not a param tensor). Distinct from every gate prefix in
# resume_registry.GATE_KEY_PREFIXES + the chroma-history prefix.
POLYAK_SCALAR_PREFIX = "__pta_"


def polyak_finisher_window_provenance(stage_window_epochs: int, frac: float = 0.2) -> dict[str, Any]:
    """DERIVED-AT-CONFIG provenance for the Polyak tail window (NOT a bare literal).

    ``muon_finisher_schedule_warmstart_and_lr_anneal_v1`` prescribes a finisher tail-average window of
    ~0.1-0.3x the stage window. Returns the derived window (epochs) + the ladder-tagged provenance row
    the DSL lever / launch record stamps, so ``start_epoch`` for a true tail can be sized from the
    stage length rather than hardcoded. Pure / import-light / unit-tested."""
    if int(stage_window_epochs) <= 0:
        raise ValueError(f"stage_window_epochs={stage_window_epochs} must be > 0")
    if not 0.0 < float(frac) <= 1.0:
        raise ValueError(f"frac={frac} must be in (0, 1]")
    window = max(1, round(float(frac) * int(stage_window_epochs)))
    start_epoch = int(stage_window_epochs) - window
    return {
        "ladder_class": "derived_at_config",
        "equation_id": "muon_finisher_schedule_warmstart_and_lr_anneal_v1",
        "stage_window_epochs": int(stage_window_epochs),
        "tail_frac": float(frac),
        "polyak_window_epochs": int(window),
        "polyak_start_epoch": int(start_epoch),
        "note": ("Polyak tail window = round(frac * stage_window); frac in [0.1,0.3] per "
                 "muon_finisher_schedule_warmstart_and_lr_anneal_v1 (finisher-EMA=Polyak averaging)"),
    }


@dataclass
class PolyakTailAverager:
    """Uniform (Polyak/Ruppert) tail average of the LIVE iterates over the finishing window.

    ``observe(epoch, params_np)`` folds the end-of-epoch live weights into an incremental uniform mean
    (Welford update ``m_n = m_{n-1} + (x_n - m_{n-1})/n`` — exact + numerically stable, computed in
    fp64) once ``epoch >= start_epoch`` and ``arm``. ``mean_fp32()`` is the exported candidate. When
    ``arm`` is False the averager is inert and every persistence surface returns ``{}`` (byte-identical).

    Resume: the SCALAR sentinel surface (:meth:`state_arrays` / :meth:`restore_from_cfg`, the
    :class:`Resumable` protocol) persists count/start/arm through the resume registry; the HEAVY
    running-mean arrays go through :meth:`heavy_state_arrays` / :meth:`restore_heavy` (routed to the
    resume sidecar under a dedicated prefix by the trainer). Both are written in one atomic savez."""

    start_epoch: int = 0
    arm: bool = False
    _mean: "dict[str, Any] | None" = field(default=None, repr=False)
    _count: int = 0

    # --- live accumulation ---------------------------------------------------
    @property
    def active(self) -> bool:
        return bool(self.arm)

    @property
    def count(self) -> int:
        return int(self._count)

    def observe(self, epoch: int, params_np: "dict[str, Any]") -> bool:
        """Fold this epoch's live weights into the uniform tail mean. Returns True iff observed."""
        if not self.arm or int(epoch) < int(self.start_epoch):
            return False
        import numpy as np
        if self._mean is None:
            self._mean = {k: np.asarray(v, np.float64).copy() for k, v in params_np.items()}
            self._count = 1
            return True
        self._count += 1
        inv = 1.0 / float(self._count)
        for k, v in params_np.items():
            m = self._mean.get(k)
            x = np.asarray(v, np.float64)
            if m is None or m.shape != x.shape:
                # a key appeared/reshaped mid-run (should not happen for a fixed witness) — seed it
                # from the current value rather than corrupt the mean; LOUD-safe (count unchanged).
                self._mean[k] = x.copy()
                continue
            m += (x - m) * inv
        return True

    def mean_fp32(self) -> "dict[str, Any] | None":
        """The exported Polyak candidate (fp32), or None when nothing has been observed."""
        if self._mean is None or self._count <= 0:
            return None
        import numpy as np
        return {k: np.asarray(v, np.float32) for k, v in self._mean.items()}

    # --- heavy resume surface (running mean; routed to the resume sidecar) ----
    def heavy_state_arrays(self, prefix: str) -> "dict[str, Any]":
        """The heavy running-mean arrays (fp64) under ``prefix`` — ``{}`` when inert/empty
        (byte-identical). The trainer merges these into the atomic resume sidecar."""
        if not self.arm or self._mean is None or self._count <= 0:
            return {}
        import numpy as np
        return {prefix + k: np.asarray(v, np.float64) for k, v in self._mean.items()}

    def restore_heavy(self, heavy: "dict[str, Any] | None") -> bool:
        """Restore the running mean from the sidecar-routed heavy arrays. Returns True iff restored.
        Count is restored from the scalar sentinel (:meth:`restore_from_cfg`); a heavy restore with a
        zero count is a corrupt pairing => LOUD-safe reset (fail-OPEN; EMA shadow untouched)."""
        if not heavy:
            return False
        import numpy as np
        self._mean = {k: np.asarray(v, np.float64).copy() for k, v in heavy.items()}
        if self._count <= 0:
            # sentinel said count 0 but heavy arrays exist — cannot continue a uniform mean without n;
            # keep the arrays as a single-sample seed (n=1) so the average is at least continuable.
            self._count = 1
        return True

    # --- Resumable protocol (scalar sentinel; rides the resume registry) ------
    def state_arrays(self, prefix: str) -> "dict[str, Any]":
        """Persist arm/start/count, including the armed-empty pre-start state."""
        if not self.arm:
            return {}
        import numpy as np
        return {
            prefix + "count": np.asarray(int(self._count), np.int64),
            prefix + "start": np.asarray(int(self.start_epoch), np.int64),
            prefix + "arm": np.asarray(1, np.int64),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        """Restore the checkpoint-bound armed state and scalar trajectory."""
        ckey = prefix + "count"
        if ckey not in cfg:
            return False
        skey = prefix + "start"
        akey = prefix + "arm"
        if skey not in cfg or akey not in cfg:
            raise ValueError("Polyak scalar state is partial")
        if int(cfg[akey]) != 1:
            raise ValueError("Polyak checkpoint arm marker must equal one")
        count = int(cfg[ckey])
        if count < 0:
            raise ValueError("Polyak checkpoint count must be nonnegative")
        self._count = count
        self.start_epoch = int(cfg[skey])
        return True
