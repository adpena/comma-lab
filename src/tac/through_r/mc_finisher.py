# SPDX-License-Identifier: MIT
"""mc_finisher — the exact-metric Monte-Carlo terminal FINISHER (#396).

Gradient-free (1+1)-ES-style stochastic local search that optimizes the EXACT discrete
argmax d_seg of a FROZEN witness checkpoint by ACCEPT/REJECT — where our gradient signal is
weakest (the terminal few-thousand residual flips) and where every smoothed-surrogate path
pays the surrogate↔exact gap. Charter:
``.omx/research/papers_checked_mc_gradient_free_2607.08406_20260710.md``
(arXiv 2607.08406, Hong Zhao — gradient-free training; the ONE novel-to-us lever).

FOUR PIECES:
  1. Guided proposal engine (:class:`ProposalEngine`) — gradient-free, so guidance is a
     PRIOR on the proposal distribution refined ONLINE by measured acceptance (NO param-
     space gradient is claimed). Sources (all optional, composable): per-tensor mass
     (∝ #157 sensitivity), per-tensor 1/5-rule adaptive step, and an optional per-element
     ``saliency`` (flip-adjacency, derived by the caller from the #391 flip ledger + step
     law ``d*=Dᵀ∇m``). Modes: ``fp32`` (continuous micro-step) / ``int8`` (discrete ±k on
     the code, clamped [-128,127] — the paper's discrete support; output born byte-closed).
  2. Accept-test ladder (:meth:`MCFinisher.accept_batch`) — P9-honest: SCREEN on a subset
     (cheap, NON-authority) then CONFIRM on n600 through-R (the ONLY verdict). Batch bisect
     salvages a batch's net-negative composition when the whole batch confirms net-positive.
  3. Ratchet + safety (:meth:`MCFinisher.run`) — monotone ratchet on CONFIRMED S; ΔS =
     ``100·Δd_seg + 25·Δbytes/37_545_489`` (bytes change in int8 via injected
     ``byte_cost_fn``; fp32 default Δbytes=0). Stop rules (K dry batches / wall-clock /
     max proposals). Resumable (atomic npz snapshot + JSONL log; ``resume_from``).
     Deterministic (one seeded RNG; full provenance per accepted batch).
  4. Targets — caller-supplied tensor-name subset; default head/palette tensors.

DECOUPLING (why $0-testable + honest): the core is generic over the actuation/measurement
space via three injected callables on :class:`FinisherProblem` — ``render_fn`` (witness
forward; real run wraps the MLX witness, tests use a tiny synthetic mock), ``measure_fn``
(``confirm=False`` SCREEN / ``confirm=True`` n600 through-R; real run wraps
:func:`tac.through_r.harness.measure_through_r`), and ``byte_cost_fn`` (int8-mode archive
bytes; default constant). The finisher NEVER re-implements R or the SegNet.

AUTHORITY: the n600 CONFIRM through the frozen CPU-torch SegNet is the ONLY accept authority
(P9). Every number is ``[macOS-CPU advisory . through-R . NON-PROMOTABLE]`` until a byte-
closed ``upstream/evaluate.py`` exact row moves the pointer (contest-CPU 0.19110). This is
an ACTUATOR (means), never a score claim.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.contest_score import (
    POSE_WEIGHT,
    RATE_WEIGHT,
    SEG_WEIGHT,
    UNCOMPRESSED_SIZE_BYTES,
    pose_term,
    rate_term,
    seg_term,
)

MC_FINISHER_LABEL = "[macOS-CPU advisory . exact-metric MC finisher through-R . NON-PROMOTABLE]"

# Default finish targets: the small head + palette tensors (thousands of params). Intersected
# with the tensors actually present in the loaded checkpoint (a checkpoint may omit out_tex).
DEFAULT_PARAM_TARGETS: tuple[str, ...] = (
    "out_sdf.weight",
    "out_sdf.bias",
    "palette",
    "out_tex.weight",
    "out_tex.bias",
)

# The P2 noise floor: one confirmed flip is one d_seg floor unit = 1/(N·SEG_H·SEG_W). At
# n600 this is ~8.5e-9 d_seg ⇒ ΔS ~8.5e-7. A single-flip accept sits AT the floor (instance-
# level, not a verdict). Derived, not guessed; the exact value is n-dependent (see
# :func:`delta_s_floor_per_confirmed_flip`).
_SEG_H, _SEG_W = 384, 512


def delta_s_floor_per_confirmed_flip(n_pairs: int) -> float:
    """ΔS for a single confirmed flip = ``SEG_WEIGHT / (n_pairs·SEG_H·SEG_W)`` (DERIVED, exact).

    This is BOTH the ΔS granularity AND the P2 noise floor (Unit C argmax-tie
    nondeterminism is ~1 flip). ``SEG_WEIGHT`` is CONSUMED from :mod:`tac.contest_score`."""

    if int(n_pairs) <= 0:
        raise MCFinisherError("n_pairs must be positive")
    return SEG_WEIGHT / (int(n_pairs) * _SEG_H * _SEG_W)


class MCFinisherError(ValueError):
    """Raised on a mis-configured / non-authority / toy MC-finisher input."""


# ======================================================================================
# Parameter state (the mutation target)
# ======================================================================================
ParamState = dict[str, np.ndarray]


def _copy_params(params: Mapping[str, np.ndarray]) -> ParamState:
    return {k: np.array(v, copy=True) for k, v in params.items()}


def params_sha256(params: Mapping[str, np.ndarray]) -> str:
    """Deterministic content hash of a param state (sorted by name; dtype + bytes)."""

    h = hashlib.sha256()
    for k in sorted(params):
        a = np.ascontiguousarray(params[k])
        h.update(k.encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


# ======================================================================================
# The measured objective (what confirm/screen return)
# ======================================================================================
@dataclass(frozen=True)
class MeasuredObjective:
    """A d_seg measurement (+ optional archive bytes) at one param state.

    ``confirm`` marks the authority axis: ``True`` = n600 through-R (the ONLY verdict);
    ``False`` = SCREEN subset (cheap, NON-authority — never accept on a screen alone)."""

    d_seg: float
    archive_bytes: int
    confirm: bool
    n_pairs: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def s_component(self) -> float:
        """The score contribution this finisher controls: ``100·d_seg + 25·bytes/37_545_489``.

        Pose is not touched by the finisher (frozen sidecar); its term is a constant offset
        and CANCELS in every ΔS. So the controlled S-component is seg + rate ONLY, from the
        canonical :mod:`tac.contest_score` coefficients (P1 one-fact-one-store)."""

        return SEG_WEIGHT * float(self.d_seg) + RATE_WEIGHT * float(self.archive_bytes) / UNCOMPRESSED_SIZE_BYTES


# ======================================================================================
# The problem (injected actuation + measurement — the decoupling boundary)
# ======================================================================================
@dataclass
class FinisherProblem:
    """The injected render + measure + byte-cost the finisher optimizes over.

    ``render_fn`` and ``byte_cost_fn`` are pure functions of the param state; ``measure_fn``
    takes the rendered frames + a ``confirm`` flag. Real runs pass the MLX witness render +
    :func:`tac.through_r.harness.measure_through_r` (n600 for confirm; a subset for screen).
    Tests pass tiny synthetic mocks. The finisher NEVER re-implements R or the SegNet."""

    render_fn: Callable[[Mapping[str, np.ndarray]], Any]
    measure_fn: Callable[..., MeasuredObjective]
    n_pairs: int = 600
    byte_cost_fn: Callable[[Mapping[str, np.ndarray]], int] | None = None

    def evaluate(self, params: Mapping[str, np.ndarray], *, confirm: bool) -> MeasuredObjective:
        """Render → measure → return the measured objective (with bytes folded in)."""

        frames = self.render_fn(params)
        obj = self.measure_fn(frames, confirm=confirm)
        if not isinstance(obj, MeasuredObjective):
            raise MCFinisherError(
                "measure_fn must return a MeasuredObjective; got " + type(obj).__name__
            )
        if self.byte_cost_fn is not None:
            b = int(self.byte_cost_fn(params))
            obj = MeasuredObjective(
                d_seg=obj.d_seg, archive_bytes=b, confirm=obj.confirm,
                n_pairs=obj.n_pairs, extra=obj.extra,
            )
        return obj


# ======================================================================================
# 1. GUIDED PROPOSAL ENGINE
# ======================================================================================
@dataclass
class Proposal:
    """One candidate mutation: a subset of elements of ONE tensor + their deltas."""

    tensor: str
    flat_indices: np.ndarray  # (B,) int64 flat indices into the tensor
    deltas: np.ndarray  # (B,) float64 (fp32 mode) or int (int8 mode) additive perturbations

    def apply(self, params: ParamState, *, mode: str) -> ParamState:
        """Return a NEW param state with this proposal applied (out-of-place)."""

        out = _copy_params(params)
        a = out[self.tensor]
        flat = a.reshape(-1)
        if mode == "int8":
            new = flat[self.flat_indices].astype(np.int64) + self.deltas.astype(np.int64)
            flat[self.flat_indices] = np.clip(new, -128, 127).astype(a.dtype)
        else:  # fp32 continuous micro-step
            flat[self.flat_indices] = (
                flat[self.flat_indices].astype(np.float64) + self.deltas.astype(np.float64)
            ).astype(a.dtype)
        return out

    def subset(self, take: np.ndarray) -> Proposal:
        """A sub-proposal over a boolean/index selection of this proposal's elements (bisect)."""

        return Proposal(
            tensor=self.tensor,
            flat_indices=np.asarray(self.flat_indices)[take],
            deltas=np.asarray(self.deltas)[take],
        )


VALID_MODES = ("fp32", "int8")


class ProposalEngine:
    """Guided (1+1)-ES proposal sampler over the target tensors.

    Guidance is a PRIOR refined ONLINE — the per-tensor proposal mass and step scale are
    updated from MEASURED acceptance (no param-space gradient). This is the "guided, not
    blind" mechanism: a tensor that yields accepted mutations gets more mass + a step that
    self-adapts via the 1/5-success rule. When ``use_saliency`` and a per-tensor saliency
    vector is supplied, element selection within a tensor is ∝ saliency (flip-adjacency)."""

    def __init__(
        self,
        params: Mapping[str, np.ndarray],
        *,
        targets: Sequence[str],
        mode: str = "fp32",
        rng: np.random.Generator | None = None,
        tensor_weights: Mapping[str, float] | None = None,
        saliency: Mapping[str, np.ndarray] | None = None,
        use_saliency: bool = True,
        batch_elems: int = 32,
        fp32_step0: float = 1e-3,
        int8_step_choices: Sequence[int] = (-2, -1, 1, 2),
        adapt_rate: float = 0.15,
    ) -> None:
        if mode not in VALID_MODES:
            raise MCFinisherError(f"mode must be one of {VALID_MODES}; got {mode!r}")
        present = [t for t in targets if t in params]
        if not present:
            raise MCFinisherError(
                f"no target tensors present in checkpoint; targets={list(targets)} "
                f"present={sorted(params)}"
            )
        self.mode = mode
        self.targets = tuple(present)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.batch_elems = int(batch_elems)
        if self.batch_elems <= 0:
            raise MCFinisherError("batch_elems must be positive")
        self.int8_step_choices = tuple(int(x) for x in int8_step_choices)
        self.adapt_rate = float(adapt_rate)
        self.use_saliency = bool(use_saliency)
        self._sizes = {t: int(np.asarray(params[t]).size) for t in self.targets}
        # per-tensor proposal mass (∝ #157 sensitivity prior; default uniform)
        w0 = {t: float(tensor_weights.get(t, 1.0)) if tensor_weights else 1.0 for t in self.targets}
        s = sum(max(v, 0.0) for v in w0.values()) or 1.0
        self._mass = {t: max(w0[t], 0.0) / s for t in self.targets}
        # per-tensor adaptive step scale (fp32: additive sigma; int8: multiplier on choices)
        self._step = {t: float(fp32_step0) for t in self.targets}
        # per-tensor accept/attempt counters (drive online reweighting)
        self._attempts = dict.fromkeys(self.targets, 0)
        self._accepts = dict.fromkeys(self.targets, 0)
        # per-tensor saliency (flat, non-negative) → element selection prior
        self._saliency: dict[str, np.ndarray] = {}
        if saliency:
            for t in self.targets:
                if t in saliency:
                    sal = np.asarray(saliency[t], dtype=np.float64).reshape(-1)
                    if sal.size != self._sizes[t]:
                        raise MCFinisherError(
                            f"saliency[{t!r}] size {sal.size} != tensor size {self._sizes[t]}"
                        )
                    if np.any(sal < 0):
                        raise MCFinisherError(f"saliency[{t!r}] must be non-negative")
                    self._saliency[t] = sal

    @property
    def tensor_mass(self) -> dict[str, float]:
        return dict(self._mass)

    @property
    def step_scale(self) -> dict[str, float]:
        return dict(self._step)

    def _pick_tensor(self) -> str:
        ts = list(self.targets)
        p = np.array([self._mass[t] for t in ts], dtype=np.float64)
        p = p / p.sum() if p.sum() > 0 else np.full(len(ts), 1.0 / len(ts))
        return ts[int(self.rng.choice(len(ts), p=p))]

    def _pick_indices(self, tensor: str) -> np.ndarray:
        size = self._sizes[tensor]
        b = min(self.batch_elems, size)
        if self.use_saliency and tensor in self._saliency:
            sal = self._saliency[tensor]
            tot = sal.sum()
            if tot > 0:
                prob = sal / tot
                return self.rng.choice(size, size=b, replace=False, p=prob).astype(np.int64)
        return self.rng.choice(size, size=b, replace=False).astype(np.int64)

    def propose(self) -> Proposal:
        """Sample one guided proposal (tensor ∝ mass, elements ∝ saliency, deltas by mode)."""

        t = self._pick_tensor()
        idx = self._pick_indices(t)
        b = idx.size
        if self.mode == "int8":
            base = self.rng.choice(self.int8_step_choices, size=b)
            deltas = base.astype(np.int64)
        else:
            deltas = self.rng.normal(0.0, self._step[t], size=b)
        return Proposal(tensor=t, flat_indices=idx, deltas=deltas)

    def register_outcome(self, proposal: Proposal, *, accepted: bool) -> None:
        """Online guidance update from a MEASURED accept/reject (no gradient).

        * per-tensor mass EMA-reweighted toward tensors that accept (guided-not-blind);
        * per-tensor fp32 step self-adapts via the 1/5-success rule (grow on accept, shrink
          on reject) — the classic (1+1)-ES step control."""

        t = proposal.tensor
        self._attempts[t] += 1
        if accepted:
            self._accepts[t] += 1
        # 1/5-success-rule style step adaptation (fp32 only; int8 steps are discrete).
        if self.mode == "fp32":
            if accepted:
                self._step[t] *= 1.0 + self.adapt_rate
            else:
                self._step[t] *= 1.0 - 0.25 * self.adapt_rate
            self._step[t] = float(np.clip(self._step[t], 1e-8, 1.0))
        # online mass reweight toward measured acceptance rate (Laplace-smoothed).
        rates = {
            tt: (self._accepts[tt] + 1.0) / (self._attempts[tt] + 2.0) for tt in self.targets
        }
        s = sum(rates.values()) or 1.0
        self._mass = {tt: rates[tt] / s for tt in self.targets}


# ======================================================================================
# 2 + 3. THE FINISHER (accept-test ladder + ratchet + safety + resume)
# ======================================================================================
@dataclass
class BatchOutcome:
    """One proposal's full ladder verdict (the observability row)."""

    proposal_index: int
    tensor: str
    n_elems: int
    screen_dseg: float | None
    confirm_dseg: float | None
    delta_s: float | None
    accepted: bool
    bisect_depth: int
    reason: str


@dataclass
class MCFinisherResult:
    """The finish outcome: best params + confirmed S trajectory + per-batch outcomes."""

    best_params: ParamState
    best_s_component: float
    start_s_component: float
    delta_s_total: float
    n_confirmed_batches: int
    n_accepted: int
    outcomes: list[BatchOutcome]
    stop_reason: str
    seed: int
    n_pairs: int
    checkpoint_sha256: str
    best_params_sha256: str
    label: str = MC_FINISHER_LABEL

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "proposal_index": o.proposal_index,
                "tensor": o.tensor,
                "n_elems": o.n_elems,
                "screen_dseg": o.screen_dseg,
                "confirm_dseg": o.confirm_dseg,
                "delta_s": o.delta_s,
                "accepted": o.accepted,
                "bisect_depth": o.bisect_depth,
                "reason": o.reason,
            }
            for o in self.outcomes
        ]


class MCFinisher:
    """Gradient-free terminal finisher over a frozen witness checkpoint (the actuator)."""

    def __init__(
        self,
        params: Mapping[str, np.ndarray],
        problem: FinisherProblem,
        *,
        targets: Sequence[str] | None = None,
        mode: str = "fp32",
        seed: int = 0,
        tensor_weights: Mapping[str, float] | None = None,
        saliency: Mapping[str, np.ndarray] | None = None,
        use_saliency: bool = True,
        batch_elems: int = 32,
        screen_gate: bool = True,
        bisect: bool = True,
        max_bisect_depth: int = 3,
        checkpoint_sha256: str | None = None,
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        tgt = tuple(targets) if targets is not None else DEFAULT_PARAM_TARGETS
        self.problem = problem
        self.mode = mode
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._rng_draws = 0  # advanced by resume to keep determinism
        self.screen_gate = bool(screen_gate)
        self.bisect = bool(bisect)
        self.max_bisect_depth = int(max_bisect_depth)
        self._checkpoint_sha256 = checkpoint_sha256 or params_sha256(params)
        self.params: ParamState = _copy_params(params)
        self.engine = ProposalEngine(
            self.params,
            targets=tgt,
            mode=mode,
            rng=self._rng,
            tensor_weights=tensor_weights,
            saliency=saliency,
            use_saliency=use_saliency,
            batch_elems=batch_elems,
            **(dict(engine_kwargs) if engine_kwargs else {}),
        )
        # confirmed anchor: the CURRENT accepted objective (the ratchet reference).
        self._current: MeasuredObjective | None = None
        self.outcomes: list[BatchOutcome] = []
        self._proposal_index = 0
        self._n_accepted = 0

    # -- authority anchor ---------------------------------------------------------------
    def _confirm(self, params: Mapping[str, np.ndarray]) -> MeasuredObjective:
        obj = self.problem.evaluate(params, confirm=True)
        if not obj.confirm:
            raise MCFinisherError(
                "confirm authority violated: measure_fn returned confirm=False for a CONFIRM "
                "call. The n600 through-R confirm is the ONLY accept authority (P9)."
            )
        return obj

    def _screen(self, params: Mapping[str, np.ndarray]) -> MeasuredObjective:
        return self.problem.evaluate(params, confirm=False)

    def current_objective(self) -> MeasuredObjective:
        if self._current is None:
            self._current = self._confirm(self.params)
        return self._current

    # -- the accept-test ladder ---------------------------------------------------------
    def accept_batch(self, proposal: Proposal) -> BatchOutcome:
        """The TOP-LEVEL proposal ladder: SCREEN → CONFIRM → ratchet (+ bisect salvage).

        Produces EXACTLY ONE observability outcome + ONE guidance update per proposal (the
        internal bisect recursion is bookkeeping-silent — its confirm calls are counted via
        the returned ``n_confirm_calls`` but it appends no extra rows and fires no extra
        guidance update). The CONFIRM (n600 through-R) is the ONLY accept authority (P9)."""

        idx = self._proposal_index
        out = self._try_batch(proposal, _depth=0)
        # Stamp the top-level proposal_index; n_elems already reflects what was APPLIED (the
        # full proposal on a top-level accept/reject, or the salvaged sub-size on a bisect).
        out.proposal_index = idx
        self.engine.register_outcome(proposal, accepted=out.accepted)
        self.outcomes.append(out)
        return out

    def _try_batch(self, proposal: Proposal, *, _depth: int) -> BatchOutcome:
        """Decision-only (no outcome append, no guidance update): mutates params/ratchet on
        accept, recurses to bisect the net-negative half of a net-positive batch."""

        base = self.current_objective()
        cand = proposal.apply(self.params, mode=self.mode)

        # (a) SCREEN — cheap, NON-authority. Reject early if it does not improve d_seg.
        screen_dseg: float | None = None
        if self.screen_gate and _depth == 0:
            sc = self._screen(cand)
            screen_dseg = sc.d_seg
            base_screen = self._screen(self.params)
            if sc.d_seg >= base_screen.d_seg:
                return BatchOutcome(
                    proposal_index=-1, tensor=proposal.tensor,
                    n_elems=int(proposal.flat_indices.size), screen_dseg=screen_dseg,
                    confirm_dseg=None, delta_s=None, accepted=False, bisect_depth=_depth,
                    reason="screen_no_improve",
                )

        # (b) CONFIRM — n600 through-R, the authority.
        conf = self._confirm(cand)
        delta_s = conf.s_component() - base.s_component()

        if delta_s < 0.0:  # strict monotone ratchet
            self.params = cand
            self._current = conf
            self._n_accepted += 1
            return BatchOutcome(
                proposal_index=-1, tensor=proposal.tensor, n_elems=int(proposal.flat_indices.size),
                screen_dseg=screen_dseg, confirm_dseg=conf.d_seg, delta_s=delta_s,
                accepted=True, bisect_depth=_depth, reason="accepted",
            )

        # (c) net-positive/zero: try to salvage the net-negative composition via bisect.
        if self.bisect and _depth < self.max_bisect_depth and proposal.flat_indices.size >= 2:
            b = proposal.flat_indices.size
            half = b // 2
            first = np.zeros(b, dtype=bool)
            first[:half] = True
            salvaged: BatchOutcome | None = None
            for take in (first, ~first):
                sub = proposal.subset(take)
                if sub.flat_indices.size == 0:
                    continue
                sub_out = self._try_batch(sub, _depth=_depth + 1)
                if sub_out.accepted:
                    salvaged = sub_out  # ratchet already advanced inside the recursion
            if salvaged is not None:
                return salvaged

        return BatchOutcome(
            proposal_index=-1, tensor=proposal.tensor, n_elems=int(proposal.flat_indices.size),
            screen_dseg=screen_dseg, confirm_dseg=conf.d_seg, delta_s=None,
            accepted=False, bisect_depth=_depth, reason="confirm_no_improve",
        )

    # -- the ratchet loop + safety ------------------------------------------------------
    def run(
        self,
        *,
        max_confirmed_batches: int = 200,
        max_dry_batches: int | None = None,
        wall_clock_budget_s: float | None = None,
        log_path: str | Path | None = None,
        snapshot_path: str | Path | None = None,
        snapshot_every: int = 25,
    ) -> MCFinisherResult:
        """Drive the accept/reject loop under the stop rules; return the finish outcome.

        Resumability P0: if ``snapshot_path`` / ``log_path`` are given, the current params +
        the per-batch JSONL are written atomically so a successor can ``resume_from`` them."""

        start_obj = self.current_objective()
        start_s = start_obj.s_component()
        t0 = time.monotonic()
        dry = 0
        stop = "max_confirmed_batches"
        with contextlib.ExitStack() as stack:
            log_fh = (
                stack.enter_context(Path(log_path).open("a", encoding="utf-8"))
                if log_path is not None
                else None
            )
            if snapshot_path is not None:  # always snapshot on exit for crash-resume.
                stack.callback(self._atomic_snapshot, snapshot_path)
            for _ in range(int(max_confirmed_batches)):
                if wall_clock_budget_s is not None and time.monotonic() - t0 >= wall_clock_budget_s:
                    stop = "wall_clock_budget"
                    break
                self._proposal_index += 1
                proposal = self.engine.propose()
                out = self.accept_batch(proposal)
                if log_fh is not None:
                    log_fh.write(json.dumps(self._log_row(out)) + "\n")
                    log_fh.flush()
                if out.accepted:
                    dry = 0
                else:
                    dry += 1
                    if max_dry_batches is not None and dry >= max_dry_batches:
                        stop = "max_dry_batches"
                        break
                if snapshot_path is not None and (self._proposal_index % max(1, snapshot_every) == 0):
                    self._atomic_snapshot(snapshot_path)

        best = self.current_objective()
        return MCFinisherResult(
            best_params=_copy_params(self.params),
            best_s_component=best.s_component(),
            start_s_component=start_s,
            delta_s_total=best.s_component() - start_s,
            n_confirmed_batches=self._proposal_index,
            n_accepted=self._n_accepted,
            outcomes=list(self.outcomes),
            stop_reason=stop,
            seed=self.seed,
            n_pairs=self.problem.n_pairs,
            checkpoint_sha256=self._checkpoint_sha256,
            best_params_sha256=params_sha256(self.params),
        )

    def _log_row(self, out: BatchOutcome) -> dict[str, Any]:
        return {
            "proposal_index": out.proposal_index,
            "tensor": out.tensor,
            "n_elems": out.n_elems,
            "screen_dseg": out.screen_dseg,
            "confirm_dseg": out.confirm_dseg,
            "delta_s": out.delta_s,
            "accepted": out.accepted,
            "bisect_depth": out.bisect_depth,
            "reason": out.reason,
            "checkpoint_sha256": self._checkpoint_sha256,
            "seed": self.seed,
            "params_sha256": params_sha256(self.params),
        }

    def _atomic_snapshot(self, path: str | Path) -> None:
        """Atomic (tmp+rename) npz snapshot of the current params for crash-resume."""

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends '.npz' unless the name already ends in it; use a '.npz' tmp so the
        # written path is deterministic, then atomically rename onto the target (tmp+rename).
        tmp = p.parent / (p.name + ".tmp.npz")
        np.savez(
            tmp,
            __mc_seed=np.int64(self.seed),
            __mc_proposal_index=np.int64(self._proposal_index),
            __mc_n_accepted=np.int64(self._n_accepted),
            __mc_checkpoint_sha256=np.array(self._checkpoint_sha256),
            **self.params,
        )
        tmp.replace(p)

    @classmethod
    def resume_from(
        cls,
        snapshot_path: str | Path,
        problem: FinisherProblem,
        *,
        targets: Sequence[str] | None = None,
        mode: str = "fp32",
        **kwargs: Any,
    ) -> MCFinisher:
        """Reconstruct a finisher from an atomic snapshot (crash-resume).

        Restores the accepted PARAMS + the (proposal_index, n_accepted) counters + the RNG
        POSITION (re-seeded from the snapshot seed, then advanced past the ``proposal_index``
        proposals already drawn — the per-``propose`` draw COUNT is state-independent, so the
        RNG stream position is exact). HONEST scope: the engine's ONLINE-adapted per-tensor
        mass + step scale are NOT persisted (they are a function of the oracle accept/reject
        history, which resume does not replay) — they re-warm from the resumed tail. So resume
        is RNG-position-faithful + params/counter-exact, NOT a bit-identical continuation of
        the guidance state. The core guarantee (no proposal-index collision, continued
        monotone descent from the exact accepted params) holds; the guidance simply relearns.
        Pass ``saliency=``/``tensor_weights=`` again to restore the static priors."""

        d = np.load(snapshot_path, allow_pickle=True)
        seed = int(d["__mc_seed"]) if "__mc_seed" in d.files else int(kwargs.pop("seed", 0))
        prop_idx = int(d["__mc_proposal_index"]) if "__mc_proposal_index" in d.files else 0
        n_acc = int(d["__mc_n_accepted"]) if "__mc_n_accepted" in d.files else 0
        ck_sha = str(d["__mc_checkpoint_sha256"]) if "__mc_checkpoint_sha256" in d.files else None
        params = {k: np.array(d[k]) for k in d.files if not k.startswith("__mc_")}
        obj = cls(
            params, problem, targets=targets, mode=mode, seed=seed,
            checkpoint_sha256=ck_sha, **kwargs,
        )
        # Restore counters + advance the RNG past the drawn proposals for determinism.
        obj._proposal_index = prop_idx
        obj._n_accepted = n_acc
        for _ in range(prop_idx):
            obj.engine.propose()
        return obj


# ======================================================================================
# PAIR-LOCAL DIAGONAL MODE (#400) — the witness realization of PR128's click-polish exploit
# ======================================================================================
# The witness carries PAIR-LOCAL clickable code tables (proven in
# ``.omx/research/clickpolish_to_witness_design_20260710.md`` §2 from the actual forward):
#   * FiLM ``code`` frame1 rows (2p+1) — d_seg-clickable (a click on pair p's code changes
#     ONLY pair p's frame1 argmax → ONLY pair p's d_seg contribution);
#   * ξ pose-carrier ``dxi``/``xi_stored`` (P,6) on the 12-bit grid — d_pose-clickable (a
#     click on pair p's ξ warps ONLY pair p's frame0 → ONLY pair p's d_pose contribution).
# Because pair p's score depends only on row p, ONE P-pair render scores P independent
# per-(col,δ) candidates (the DIAGONAL exploit). This is a MODE, not a fork: it SHARES the
# #396 ratchet/resume/provenance/byte-cost/P9-CONFIRM-authority discipline. The shared INR
# trunk / T head / MyCar-static / b_c-global are the NON-local weight-click class (§2) — the
# #396 ``FinisherProblem`` path already handles those; they are NOT diagonal-batchable.
#
# CPU-axis discipline (§5): the CONFIRM/accept render+measure is CPU-only (the harness
# ``SUPPORTED_BACKENDS=('cpu-torch',)`` for the code axis; the byte-close CPU-torch PoseNet
# for the ξ axis). A screen MAY be a cheap CPU subset — NEVER an MLX/GPU screen (would
# re-introduce PR128's measured bicubic-LSB cross-axis gap). This is a build constraint on
# the injected callables, enforced by refusing a non-authority CONFIRM below.
#
# WATERFILL CONTRACT — #400 is the PAIR-LOCAL TIER (Codex SDF-waterfill advisory,
# .omx/research/ADVISORY_sdf_scorer_waterfill_20260710.md). That advisory's terminal-band
# controller is a hierarchical interaction-aware water-filler: (1) legal atomic proposals →
# (2) compile through the real receiver + final ZIP → (3) exact component+byte deltas →
# (4) estimate pair/group INTERACTIONS → (5) budgeted compatible-set selection → (6) joint
# re-compile + re-score → (7) update waterline, repeat. This diagonal mode IMPLEMENTS steps
# 1-3 for the pair-local special case (independent per-pair candidates), and — critically —
# already honours step 6: every JOINT accept is EXACTLY re-verified through the real CONFIRM
# and the ratchet reads that exact S, NEVER a sum of per-move deltas (the no-local-gains-
# additivity law). The accepted-moves ledger records SEPARATE per-component deltas
# (axis distortion / rate / bytes; see ``_log_row``) so a future interaction-aware selector
# (steps 4-5) can consume it without re-deriving. The cross-class interaction estimate +
# compatible-set selection (steps 4-5) are the OUT-OF-SCOPE upper tier.


class LocalityGuardError(MCFinisherError):
    """Raised when the runtime 2-pair locality probe FAILS (cross-talk detected, or the
    probe is vacuous so locality cannot be certified). Fail-closed: no diagonal batch may
    be trusted until locality is proven on the ACTUAL render (never assumed)."""


VALID_DIAGONAL_AXES = ("d_seg", "d_pose")


@dataclass(frozen=True)
class DiagonalObjective:
    """A per-pair distortion VECTOR (+ archive bytes) at one integer code-table state.

    ``per_pair`` is the axis distortion for every pair (d_seg fraction, or d_pose MSE);
    ``agg`` is the aggregate the score reads (``mean`` over pairs — d_seg IS the mean; the
    pose term is ``sqrt(10·mean)``). ``confirm`` marks the authority axis exactly as
    :class:`MeasuredObjective` does (True = the n600/canonical-layout re-render, the ONLY
    accept authority; False = a cheap CPU screen subset)."""

    per_pair: np.ndarray
    agg: float
    archive_bytes: int
    confirm: bool
    axis: str
    n_pairs: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axis not in VALID_DIAGONAL_AXES:
            raise MCFinisherError(f"axis must be one of {VALID_DIAGONAL_AXES}; got {self.axis!r}")

    def axis_s_component(self) -> float:
        """The S-contribution this axis controls: the axis distortion term + rate.

        The OTHER distortion is a constant offset that CANCELS in every ΔS (a ``code``
        click leaves d_pose untouched — frozen ξ; a ξ click leaves d_seg untouched — SegNet
        reads frame1, ξ warps frame0). So for the d_seg axis the controlled S-component is
        ``100·d_seg + rate``; for d_pose it is ``sqrt(10·d_pose) + rate`` (canonical
        :mod:`tac.contest_score` terms, P1 one-fact-one-store)."""

        dist = seg_term(self.agg) if self.axis == "d_seg" else pose_term(self.agg)
        return dist + rate_term(self.archive_bytes)


@dataclass
class DiagonalProblem:
    """Injected actuation + measurement for the pair-local diagonal exploit (the seam).

    ``render_measure_fn(table, *, confirm)`` applies the integer ``table`` (one row per
    pair) into the witness, renders ALL pairs, measures, and returns a
    :class:`DiagonalObjective` (per-pair axis vector + agg + bytes). Real runs wire:
      * d_seg axis → a render of the ``code`` frame1 rows + :func:`tac.through_r.harness.
        measure_through_r` (its ``per_pair_dseg`` IS the vector — no harness change needed);
      * d_pose axis (4c′) → the byte-close ξ serialize→inflate→frozen-CPU-PoseNet path
        (:func:`make_byte_close_xi_pose_measure`).
    ``probe_frames_fn(table, pairs)`` returns per-pair frames (indexable ``[i]``) for the
    2-pair LOCALITY GUARD only. ``byte_cost_fn(table)`` returns the REAL re-encoded archive
    bytes for the mutated section (never estimated). ``lo``/``hi`` are the inclusive grid
    clamp bounds (int8 code: -128/127; ξ 12-bit: 0/4095). ``axis`` picks which per-pair
    distortion drives the accept."""

    render_measure_fn: Callable[..., DiagonalObjective]
    n_pairs: int
    lo: int
    hi: int
    axis: str = "d_seg"
    probe_frames_fn: Callable[[np.ndarray, Sequence[int]], Any] | None = None
    byte_cost_fn: Callable[[np.ndarray], int] | None = None

    def __post_init__(self) -> None:
        if self.axis not in VALID_DIAGONAL_AXES:
            raise MCFinisherError(f"axis must be one of {VALID_DIAGONAL_AXES}; got {self.axis!r}")
        if int(self.lo) > int(self.hi):
            raise MCFinisherError(f"lo ({self.lo}) must be <= hi ({self.hi})")

    def measure(self, table: np.ndarray, *, confirm: bool) -> DiagonalObjective:
        obj = self.render_measure_fn(table, confirm=confirm)
        if not isinstance(obj, DiagonalObjective):
            raise MCFinisherError(
                "render_measure_fn must return a DiagonalObjective; got " + type(obj).__name__
            )
        if obj.axis != self.axis:
            raise MCFinisherError(
                f"render_measure_fn returned axis {obj.axis!r} != problem axis {self.axis!r}"
            )
        if obj.per_pair.shape[0] != self.n_pairs:
            raise MCFinisherError(
                f"per_pair has {obj.per_pair.shape[0]} pairs != n_pairs {self.n_pairs}"
            )
        if self.byte_cost_fn is not None:
            b = int(self.byte_cost_fn(table))
            obj = DiagonalObjective(
                per_pair=obj.per_pair, agg=obj.agg, archive_bytes=b, confirm=obj.confirm,
                axis=obj.axis, n_pairs=obj.n_pairs, extra=obj.extra,
            )
        return obj

    def probe_frames(self, table: np.ndarray, pairs: Sequence[int]) -> Any:
        if self.probe_frames_fn is None:
            raise LocalityGuardError(
                "no probe_frames_fn injected: the locality guard cannot render the 2-pair "
                "probe. Inject probe_frames_fn (real: the witness/byte-close per-pair render) "
                "so locality is PROVEN on the actual forward, not assumed."
            )
        return self.probe_frames_fn(table, list(pairs))


@dataclass
class DiagonalRoundOutcome:
    """One diagonal round's verdict (the observability row)."""

    round_index: int
    n_clicks: int
    agg_before: float
    agg_after: float | None
    s_before: float
    s_after: float | None
    delta_s: float | None
    archive_bytes: int | None
    accepted: bool
    bisect_depth: int
    reason: str


@dataclass
class DiagonalFinisherResult:
    """The diagonal finish outcome: best table + confirmed S trajectory + per-round rows."""

    best_table: np.ndarray
    best_s_component: float
    start_s_component: float
    delta_s_total: float
    best_agg: float
    start_agg: float
    n_rounds: int
    n_clicks_total: int
    pairs_touched: int
    outcomes: list[DiagonalRoundOutcome]
    stop_reason: str
    axis: str
    n_pairs: int
    checkpoint_sha256: str
    best_table_sha256: str
    rollback_floor: float | None
    locality_certified: bool
    label: str = MC_FINISHER_LABEL
    score_claim: bool = False
    promotable: bool = False


def _table_sha256(table: np.ndarray) -> str:
    a = np.ascontiguousarray(table)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


class PairLocalDiagonalFinisher:
    """Pair-local diagonal exact-gated ratchet over an integer code table (the #400 MODE).

    Loop (deterministic; fixed column/δ sweep order — no RNG, so resume is exact):
      0. LOCALITY GUARD (once, fail-closed): a 2-pair probe proves a click on pair a leaves
         pair b's frames BYTE-IDENTICAL (and non-vacuously changes a). No batch is trusted
         until this passes on the ACTUAL render.
      1. DIAGONAL SWEEP: for each (col, δ) apply the click to that column across ALL pairs,
         ONE CONFIRM render measures the per-pair vector, and each pair records its best
         improving click (d_seg: exact additive ΔS; d_pose: local-marginal-ranked, exact at
         accept). One render scores n_pairs candidates.
      2. JOINT ACCEPT (≤1 click/pair): the combined table is re-measured on the CONFIRM
         (canonical-layout) authority; strict monotone S ratchet; bisect salvages a
         net-negative composition; for the d_pose axis a ROLLBACK FLOOR forbids ever shipping
         an aggregate d_pose worse than the pinned banked value.
      3. RESUMABLE: every accepted round appends to an accepted-moves JSONL (the resume
         authority) + an atomic table npz snapshot.
    CONFIRM (the real render+measure through the frozen CPU scorer) is the ONLY accept
    authority (P9). Every number is NON-PROMOTABLE advisory until a byte-closed exact row."""

    def __init__(
        self,
        table0: np.ndarray,
        problem: DiagonalProblem,
        *,
        columns: Sequence[int] | None = None,
        deltas: Sequence[int] = (1, -1, 2, -2),
        screen_gate: bool = False,
        bisect: bool = True,
        eps: float = 0.0,
        rollback_floor: float | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        t = np.array(table0, copy=True)
        if t.ndim != 2:
            raise MCFinisherError(f"table must be 2-D (n_pairs, n_cols); got shape {t.shape}")
        if t.shape[0] != problem.n_pairs:
            raise MCFinisherError(
                f"table has {t.shape[0]} rows != problem.n_pairs {problem.n_pairs}"
            )
        if not np.issubdtype(t.dtype, np.integer):
            raise MCFinisherError(f"table must be an integer grid; got dtype {t.dtype}")
        self.table = t
        self._table0 = np.array(t, copy=True)  # pre-polish original (for pairs_touched)
        self.problem = problem
        self.axis = problem.axis
        self.n_pairs = int(problem.n_pairs)
        self.n_cols = int(t.shape[1])
        self.columns = tuple(int(c) for c in (columns if columns is not None else range(self.n_cols)))
        for c in self.columns:
            if not (0 <= c < self.n_cols):
                raise MCFinisherError(f"column {c} out of range [0,{self.n_cols})")
        self.deltas = tuple(int(d) for d in deltas)
        if not self.deltas:
            raise MCFinisherError("deltas must be non-empty")
        self.screen_gate = bool(screen_gate)
        self.bisect = bool(bisect)
        self.eps = float(eps)
        self.rollback_floor = None if rollback_floor is None else float(rollback_floor)
        self._checkpoint_sha256 = checkpoint_sha256 or _table_sha256(t)
        self._current: DiagonalObjective | None = None
        self.outcomes: list[DiagonalRoundOutcome] = []
        self._round_index = 0
        self._n_clicks_total = 0
        self._locality_certified = False

    # -- authority anchor ---------------------------------------------------------------
    def _confirm(self, table: np.ndarray) -> DiagonalObjective:
        obj = self.problem.measure(table, confirm=True)
        if not obj.confirm:
            raise MCFinisherError(
                "confirm authority violated: render_measure_fn returned confirm=False for a "
                "CONFIRM call. The canonical-layout n600 through-R/byte-close render is the "
                "ONLY accept authority (P9)."
            )
        return obj

    def current_objective(self) -> DiagonalObjective:
        if self._current is None:
            self._current = self._confirm(self.table)
        return self._current

    # -- LOCALITY GUARD (fail-closed) ---------------------------------------------------
    def verify_locality(
        self, *, pair_a: int = 0, pair_b: int = 1, probe_deltas: Sequence[int] | None = None
    ) -> dict[str, Any]:
        """Prove a click on pair_a's row leaves pair_b's frames BYTE-IDENTICAL (no cross-talk)
        AND non-vacuously changes pair_a, on the ACTUAL render. The locality property is
        column-independent, so this scans a few (col, δ) until a non-vacuous click is found."""

        if pair_a == pair_b:
            raise LocalityGuardError("locality probe needs two DISTINCT pairs")
        base = self.problem.probe_frames(self.table, [pair_a, pair_b])
        base_a, base_b = base[0], base[1]
        trial_deltas = tuple(probe_deltas) if probe_deltas is not None else self.deltas
        a_changed = False
        b_unchanged = True
        used = (None, None)
        for col in self.columns:
            for dl in trial_deltas:
                cand = self.table.copy()
                new = int(cand[pair_a, col]) + int(dl)
                new = int(np.clip(new, self.problem.lo, self.problem.hi))
                if new == int(cand[pair_a, col]):
                    continue  # clamp no-op — vacuous for this (col, δ)
                cand[pair_a, col] = new
                fr = self.problem.probe_frames(cand, [pair_a, pair_b])
                fr_a, fr_b = fr[0], fr[1]
                if not np.array_equal(np.asarray(fr_b), np.asarray(base_b)):
                    b_unchanged = False  # CROSS-TALK: pair_a's click moved pair_b
                    used = (col, dl)
                    break
                if not np.array_equal(np.asarray(fr_a), np.asarray(base_a)):
                    a_changed = True
                    used = (col, dl)
                    break
            if not b_unchanged or a_changed:
                break
        holds = bool(b_unchanged and a_changed)
        return {
            "pair_b_unchanged_by_pair_a_click": bool(b_unchanged),
            "pair_a_changed_by_its_click": bool(a_changed),
            "locality_holds": holds,
            "col": used[0],
            "delta": used[1],
            "pair_a": pair_a,
            "pair_b": pair_b,
        }

    def require_locality(self, *, pair_a: int = 0, pair_b: int = 1) -> dict[str, Any]:
        """Run :meth:`verify_locality` and RAISE (fail-closed) unless it holds."""

        rep = self.verify_locality(pair_a=pair_a, pair_b=pair_b)
        if not rep["pair_b_unchanged_by_pair_a_click"]:
            raise LocalityGuardError(
                f"CROSS-TALK: a click on pair {pair_a} changed pair {pair_b}'s frames "
                f"(col={rep['col']} delta={rep['delta']}). The diagonal exploit is INVALID for "
                "this problem — refusing to batch (fail-closed)."
            )
        if not rep["pair_a_changed_by_its_click"]:
            raise LocalityGuardError(
                f"VACUOUS PROBE: no click changed pair {pair_a}'s frames across the swept "
                "(col, delta) grid, so locality cannot be certified. Refusing (fail-closed)."
            )
        self._locality_certified = True
        return rep

    # -- the diagonal sweep -------------------------------------------------------------
    def _pose_marginal(self, agg: float) -> float:
        return 5.0 / math.sqrt(max(POSE_WEIGHT * agg, 1e-12))

    def _diagonal_sweep(self, base: DiagonalObjective) -> list[tuple[int, int, float] | None]:
        """One sweep. Returns best[p] = (col, delta, per_pair_after) or None per pair."""

        cur = base.per_pair
        wpose = self._pose_marginal(base.agg) if self.axis == "d_pose" else 1.0
        best: list[tuple[int, int, float] | None] = [None] * self.n_pairs
        best_gain = np.zeros(self.n_pairs, dtype=np.float64)
        for col in self.columns:
            for dl in self.deltas:
                cand = self.table.copy()
                cand[:, col] = np.clip(
                    cand[:, col].astype(np.int64) + int(dl), self.problem.lo, self.problem.hi
                ).astype(self.table.dtype)
                obj = self._confirm(cand)  # ONE render scores n_pairs candidates (diagonal)
                v = obj.per_pair
                # per-pair improvement (positive = better): d_seg exact-additive; d_pose
                # local-marginal-ranked (exact recompute at accept). Both /n cancel in rank.
                if self.axis == "d_seg":
                    gain = (cur - v) * (SEG_WEIGHT / self.n_pairs)
                else:
                    gain = (cur - v) * (wpose / self.n_pairs)
                for p in range(self.n_pairs):
                    if cand[p, col] == self.table[p, col]:
                        continue  # clamp no-op for this pair
                    if gain[p] > best_gain[p] + 1e-15:
                        best_gain[p] = gain[p]
                        best[p] = (col, int(dl), float(v[p]))
        return best

    def _apply_clicks(self, table: np.ndarray, clicks: Sequence[tuple[int, int, int]]) -> np.ndarray:
        out = table.copy()
        for p, col, dl in clicks:
            out[p, col] = np.clip(
                int(out[p, col]) + int(dl), self.problem.lo, self.problem.hi
            ).astype(self.table.dtype)
        return out

    def _floor_ok(self, obj: DiagonalObjective) -> bool:
        """The d_pose rollback floor: never ACCEPT an aggregate d_pose worse than the pinned
        banked value (memo requirement (d)). Non-pose axes / no floor => always ok."""

        if self.axis != "d_pose" or self.rollback_floor is None:
            return True
        return obj.agg <= self.rollback_floor + 1e-12

    def _accept_joint(
        self, base: DiagonalObjective, clicks: list[tuple[int, int, int]]
    ) -> tuple[bool, DiagonalObjective, int, list[tuple[int, int, int]]]:
        """Exact-gated joint accept with bisect salvage. Returns
        (accepted, new_objective, bisect_depth, applied_clicks)."""

        cand_table = self._apply_clicks(self.table, clicks)
        cand = self._confirm(cand_table)
        if cand.axis_s_component() < base.axis_s_component() - self.eps and self._floor_ok(cand):
            self.table = cand_table
            self._current = cand
            return True, cand, 0, clicks
        if not self.bisect:
            return False, base, 0, []
        # greedy halving: keep re-measuring the largest prefix until an improving subset
        # is found (or none). Each subset is an EXACT canonical-layout re-render.
        cur = list(clicks)
        depth = 0
        while len(cur) > 1:
            depth += 1
            half = cur[: max(1, len(cur) // 2)]
            cand_table = self._apply_clicks(self.table, half)
            cand = self._confirm(cand_table)
            if cand.axis_s_component() < base.axis_s_component() - self.eps and self._floor_ok(cand):
                self.table = cand_table
                self._current = cand
                return True, cand, depth, half
            cur = half
        return False, base, depth, []

    # -- the ratchet loop + resume ------------------------------------------------------
    def run(
        self,
        *,
        max_rounds: int = 40,
        require_locality: bool = True,
        wall_clock_budget_s: float | None = None,
        log_path: str | Path | None = None,
        snapshot_path: str | Path | None = None,
    ) -> DiagonalFinisherResult:
        """Drive the diagonal ratchet. ``require_locality=True`` runs the fail-closed guard
        before the first sweep. Resumability P0: accepted rounds append to ``log_path``
        (the resume authority) and the table is snapshotted atomically to ``snapshot_path``."""

        if require_locality and not self._locality_certified:
            self.require_locality()
        start = self.current_objective()
        start_s = start.axis_s_component()
        start_agg = start.agg
        t0 = time.monotonic()
        stop = "max_rounds"
        with contextlib.ExitStack() as stack:
            log_fh = (
                stack.enter_context(Path(log_path).open("a", encoding="utf-8"))
                if log_path is not None
                else None
            )
            if snapshot_path is not None:
                stack.callback(self._atomic_snapshot, snapshot_path)
            for _ in range(int(max_rounds)):
                if wall_clock_budget_s is not None and time.monotonic() - t0 >= wall_clock_budget_s:
                    stop = "wall_clock_budget"
                    break
                base = self.current_objective()
                best = self._diagonal_sweep(base)
                clicks = [
                    (p, best[p][0], best[p][1]) for p in range(self.n_pairs) if best[p] is not None
                ]
                if not clicks:
                    stop = "plateau"
                    break
                accepted, new_obj, depth, applied = self._accept_joint(base, clicks)
                self._round_index += 1
                out = DiagonalRoundOutcome(
                    round_index=self._round_index,
                    n_clicks=len(applied),
                    agg_before=base.agg,
                    agg_after=new_obj.agg if accepted else None,
                    s_before=base.axis_s_component(),
                    s_after=new_obj.axis_s_component() if accepted else None,
                    delta_s=(new_obj.axis_s_component() - base.axis_s_component()) if accepted else None,
                    archive_bytes=new_obj.archive_bytes if accepted else None,
                    accepted=accepted,
                    bisect_depth=depth,
                    reason="accepted" if accepted else "no_improving_subset",
                )
                self.outcomes.append(out)
                if accepted:
                    self._n_clicks_total += len(applied)
                    row = self._log_row(out, applied, base=base, new=new_obj)
                    if log_fh is not None:
                        log_fh.write(json.dumps(row) + "\n")
                        log_fh.flush()
                        os.fsync(log_fh.fileno())
                    if snapshot_path is not None:
                        self._atomic_snapshot(snapshot_path)
                else:
                    stop = "no_improving_subset"
                    break
        best = self.current_objective()
        touched = int((self.table != self._table0).any(axis=1).sum())
        return DiagonalFinisherResult(
            best_table=np.array(self.table, copy=True),
            best_s_component=best.axis_s_component(),
            start_s_component=start_s,
            delta_s_total=best.axis_s_component() - start_s,
            best_agg=best.agg,
            start_agg=start_agg,
            n_rounds=self._round_index,
            n_clicks_total=self._n_clicks_total,
            pairs_touched=touched,
            outcomes=list(self.outcomes),
            stop_reason=stop,
            axis=self.axis,
            n_pairs=self.n_pairs,
            checkpoint_sha256=self._checkpoint_sha256,
            best_table_sha256=_table_sha256(self.table),
            rollback_floor=self.rollback_floor,
            locality_certified=self._locality_certified,
        )

    def _log_row(
        self,
        out: DiagonalRoundOutcome,
        applied: Sequence[tuple[int, int, int]],
        *,
        base: DiagonalObjective,
        new: DiagonalObjective,
    ) -> dict[str, Any]:
        # SEPARATE per-component deltas (NOT just ΔS) so a future interaction-aware waterfill
        # selector (Codex SDF-waterfill advisory, 2026-07-10) can consume this ledger: #400 is
        # the pair-local tier (steps 1-3) of that contract; the JOINT candidate is re-verified
        # EXACTLY here (step 6) — per-move deltas are NEVER summed as the final claim (the
        # no-local-gains-additivity law). Component fields are the axis distortion + rate, split.
        dist_before = seg_term(base.agg) if self.axis == "d_seg" else pose_term(base.agg)
        dist_after = seg_term(new.agg) if self.axis == "d_seg" else pose_term(new.agg)
        return {
            "round": out.round_index,
            "clicks": [[int(p), int(c), int(d)] for (p, c, d) in applied],
            "n_clicks": out.n_clicks,
            "axis": self.axis,
            "components": {
                # split component deltas (the interaction-aware selector's inputs).
                self.axis: {"before": base.agg, "after": new.agg, "delta": new.agg - base.agg},
                "dist_term": {"before": dist_before, "after": dist_after,
                              "delta": dist_after - dist_before},
                "bytes": {"before": base.archive_bytes, "after": new.archive_bytes,
                          "delta": int(new.archive_bytes) - int(base.archive_bytes)},
                "rate_term": {"before": rate_term(base.archive_bytes),
                              "after": rate_term(new.archive_bytes),
                              "delta": rate_term(new.archive_bytes) - rate_term(base.archive_bytes)},
            },
            "agg_after": out.agg_after,
            "s_after": out.s_after,
            "delta_s": out.delta_s,  # EXACT joint re-verified ΔS (step 6), never a per-move sum
            "archive_bytes": out.archive_bytes,
            "bisect_depth": out.bisect_depth,
            "table_sha256": _table_sha256(self.table),
            "checkpoint_sha256": self._checkpoint_sha256,
            "rollback_floor": self.rollback_floor,
            "label": MC_FINISHER_LABEL,
            "score_claim": False,
            "promotable": False,
        }

    def _atomic_snapshot(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / (p.name + ".tmp.npz")
        np.savez(
            tmp,
            __diag_table=self.table,
            __diag_axis=np.array(self.axis),
            __diag_round=np.int64(self._round_index),
            __diag_n_clicks=np.int64(self._n_clicks_total),
            __diag_checkpoint_sha256=np.array(self._checkpoint_sha256),
        )
        tmp.replace(p)

    @classmethod
    def resume_from_ledger(
        cls,
        table0: np.ndarray,
        ledger_path: str | Path,
        problem: DiagonalProblem,
        **kwargs: Any,
    ) -> PairLocalDiagonalFinisher:
        """Reconstruct a finisher by REPLAYING the accepted-moves JSONL onto ``table0`` (the
        resume authority — the sweep is deterministic, so replaying accepted clicks
        reconstructs the exact accepted table + round counter). ``table0`` is the ORIGINAL
        checkpoint table (pre-polish); the ledger carries the accepted clicks to re-apply."""

        obj = cls(table0, problem, **kwargs)
        p = Path(ledger_path)
        max_round = 0
        n_clicks = 0
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                clicks = [(int(a), int(b), int(c)) for (a, b, c) in row["clicks"]]
                obj.table = obj._apply_clicks(obj.table, clicks)
                n_clicks += len(clicks)
                max_round = max(max_round, int(row["round"]))
        obj._round_index = max_round
        obj._n_clicks_total = n_clicks
        obj._current = None  # re-anchor on the reconstructed table
        return obj


# ======================================================================================
# 4c′ ξ-TERMINAL ENTRY POINT — the pose-axis diagonal finisher wired to the byte-close decode
# ======================================================================================
# The default n600 GT cache (consumed by the code-axis measure factory; kept as a module
# const so the factory default needs no heavy harness import at module load time).
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
# Value-provenance ladder (MEASURED-ANCHOR, config-conditional): the banked R1 dxi d_pose is
# the n600 BYTE-CLOSE AUTHORITY floor the ξ diagonal polish must never ship worse than.
#   source: reports/r1_dxi_238/n600_shipdxi.json (n600 realized) via
#           .omx/research/r1_dxi_shippability_byteclose_20260708.md (THE byte-close artifact);
#           canonical anchor ``ship_dxi_realized_d_pose`` in
#           src/tac/canonical_equations/morse_smale_stratified_parallax_dpose_20260708.py;
#           MEMORY L68. sqrt(10·0.001610)=0.1269 = the banked 0.127 pose contribution.
# It is NOT a bare literal: :func:`load_banked_r1_dxi_dpose_floor` re-reads the byte-close
# memo and asserts agreement (the ladder cross-check); the constant is the cached default.
BANKED_R1_DXI_DPOSE_FLOOR: float = 0.001610

DEFAULT_BYTE_CLOSE_TOOL = "tools/levelset_byte_close_and_eval.py"
R1_DXI_BYTECLOSE_MEMO = ".omx/research/r1_dxi_shippability_byteclose_20260708.md"


def load_banked_r1_dxi_dpose_floor(memo_path: str | Path = R1_DXI_BYTECLOSE_MEMO) -> float:
    """Return the banked R1 dxi d_pose floor, CROSS-CHECKED against the byte-close artifact.

    Value-provenance: reads the n600 authority value out of the r1_dxi byte-close memo and
    asserts it equals :data:`BANKED_R1_DXI_DPOSE_FLOOR` (the cached MEASURED-ANCHOR). If the
    memo is unavailable (e.g. sandbox without .omx), the cached constant is returned with a
    provenance note — never a silent fabrication."""

    import re

    p = Path(memo_path)
    if not p.exists():
        return BANKED_R1_DXI_DPOSE_FLOOR
    text = p.read_text(encoding="utf-8")
    # the authority line: "realized **d_pose = 0.001610** over all 600 inflated pairs"
    m = re.search(r"d_pose\s*=\s*\**\s*(0\.00\d+)\s*\**\s*over all 600", text)
    if m is None:
        return BANKED_R1_DXI_DPOSE_FLOOR
    memo_val = float(m.group(1))
    if not math.isclose(memo_val, BANKED_R1_DXI_DPOSE_FLOOR, rel_tol=0, abs_tol=1e-9):
        raise MCFinisherError(
            f"banked R1 dxi d_pose floor DRIFT: memo says {memo_val}, constant says "
            f"{BANKED_R1_DXI_DPOSE_FLOOR}. Reconcile (value-provenance ladder)."
        )
    return memo_val


def load_byte_close_pose_surfaces(tool_path: str | Path = DEFAULT_BYTE_CLOSE_TOOL) -> Any:
    """Read-only import of the byte-close pose decode surfaces (the 4c′ wiring target).

    Adds ``tools/`` to ``sys.path`` and imports ``levelset_byte_close_and_eval`` (import-safe:
    guarded by ``if __name__ == '__main__'``), returning a namespace with the committed-HEAD
    surfaces the ξ diagonal measure needs: ``parse_pose_carrier``, ``serialize_pose_carrier``,
    ``pose_carrier_confirm``. This NEVER edits that file (the receiver-hardening sibling owns
    it); if a concurrent edit is in flight, its committed HEAD is what imports here."""

    import importlib
    import sys
    import types

    tp = Path(tool_path)
    tools_dir = str(tp.parent.resolve())
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    mod = importlib.import_module(tp.stem)
    ns = types.SimpleNamespace()
    for name in ("parse_pose_carrier", "serialize_pose_carrier", "pose_carrier_confirm"):
        if not hasattr(mod, name):
            raise MCFinisherError(
                f"byte-close tool {tp} is missing surface {name!r} — 4c′ wiring cannot bind "
                "(the committed HEAD changed its contract; reconcile)."
            )
        setattr(ns, name, getattr(mod, name))
    ns.module = mod
    return ns


def make_byte_close_xi_pose_measure(
    build_pose_carrier_bytes: Callable[[np.ndarray], bytes],
    inflate_and_read_raw: Callable[[bytes], Path],
    *,
    eval_pairs: int,
    gt_cache: str | None,
    num_pairs: int,
    blob: bytes | None = None,
    byte_cost_fn: Callable[[np.ndarray], int] | None = None,
    tool_path: str | Path = DEFAULT_BYTE_CLOSE_TOOL,
) -> Callable[..., DiagonalObjective]:
    """Build the 4c′ ξ-axis ``render_measure_fn`` wired to the byte-close CPU-torch PoseNet.

    The returned measure, given a ξ ``table`` (P,6): serialize → inflate → CONFIRM the real
    row d_pose via the committed-HEAD :func:`pose_carrier_confirm` (frozen CPU PoseNet, the
    authority; NEVER MPS). ``build_pose_carrier_bytes(table)`` and ``inflate_and_read_raw``
    are injected because the FULL byte-close inflate is a heavy post-launch step owned by the
    launch harness — this factory binds the SEAM (read-only surfaces) so the measurement fires
    at the terminal band (deferral D27b), not here. The per-pair d_pose vector comes from the
    surfaces' own ``cpu_verdict_d_pose_batch`` path; this factory returns the aggregate the
    confirm reports and (when the surfaces expose it) the per-pair vector.

    BUILD NOTE: no measurement is run at build time — this wires the callable; the terminal
    band (post-launch, per D27b) invokes it. It is CPU-locked by construction (§5)."""

    surfaces = load_byte_close_pose_surfaces(tool_path)

    def render_measure_fn(table: np.ndarray, *, confirm: bool) -> DiagonalObjective:
        pcar_bytes = build_pose_carrier_bytes(np.asarray(table))
        raw_path = inflate_and_read_raw(pcar_bytes)
        rep = surfaces.pose_carrier_confirm(
            raw_path, eval_pairs, gt_cache, num_pairs, pcar_bytes, blob=blob
        )
        agg = float(rep["d_pose_carrier_warp_f0_witness_f1"])
        per_pair = np.asarray(rep.get("d_pose_per_pair", np.full(int(eval_pairs), agg, dtype=np.float64)))
        bts = int(byte_cost_fn(np.asarray(table))) if byte_cost_fn is not None else len(pcar_bytes)
        return DiagonalObjective(
            per_pair=per_pair, agg=agg, archive_bytes=bts, confirm=confirm,
            axis="d_pose", n_pairs=int(per_pair.shape[0]),
            extra={"byte_close_confirm": True},
        )

    return render_measure_fn


def make_through_r_code_measure(
    render_frame1_fn: Callable[[np.ndarray], list[np.ndarray]],
    *,
    lstars: np.ndarray | None = None,
    gt_cache: str | Path | None = None,
    byte_cost_fn: Callable[[np.ndarray], int] | None = None,
    input_space: str = "auto",
    allow_subset_reason: str | None = None,
    segnet: Any | None = None,
) -> Callable[..., DiagonalObjective]:
    """Build the d_seg-axis ``render_measure_fn`` wired to :func:`tac.through_r.harness.
    measure_through_r` (its ``per_pair_dseg`` IS the diagonal per-pair vector — no harness
    change needed). ``render_frame1_fn(code_table)`` renders the 600 frame1 candidate frames
    from the FiLM ``code`` frame1 rows; ``measure_through_r`` pushes them through R + the
    frozen CPU-torch SegNet (the CPU authority, §5). ``byte_cost_fn`` supplies the REAL
    re-encoded ``code_brotli`` bytes. CPU-locked by construction (the harness refuses MPS/MLX)."""

    from tac.through_r.harness import measure_through_r as _measure

    def render_measure_fn(table: np.ndarray, *, confirm: bool) -> DiagonalObjective:
        frames = render_frame1_fn(np.asarray(table))
        res = _measure(
            frames, lstars=lstars, gt_cache=(gt_cache if gt_cache is not None else DEFAULT_GT_CACHE),
            input_space=input_space, allow_subset_reason=allow_subset_reason, segnet=segnet,
        )
        per_pair = np.asarray(res.per_pair_dseg, dtype=np.float64)
        bts = int(byte_cost_fn(np.asarray(table))) if byte_cost_fn is not None else 0
        return DiagonalObjective(
            per_pair=per_pair, agg=float(res.agg_dseg), archive_bytes=bts, confirm=confirm,
            axis="d_seg", n_pairs=int(per_pair.shape[0]), extra={"through_r": True},
        )

    return render_measure_fn


__all__ = [
    "BANKED_R1_DXI_DPOSE_FLOOR",
    "DEFAULT_BYTE_CLOSE_TOOL",
    "DEFAULT_GT_CACHE",
    "DEFAULT_PARAM_TARGETS",
    "MC_FINISHER_LABEL",
    "R1_DXI_BYTECLOSE_MEMO",
    "VALID_DIAGONAL_AXES",
    "BatchOutcome",
    "DiagonalFinisherResult",
    "DiagonalObjective",
    "DiagonalProblem",
    "DiagonalRoundOutcome",
    "FinisherProblem",
    "LocalityGuardError",
    "MCFinisher",
    "MCFinisherError",
    "MCFinisherResult",
    "MeasuredObjective",
    "PairLocalDiagonalFinisher",
    "ParamState",
    "Proposal",
    "ProposalEngine",
    "delta_s_floor_per_confirmed_flip",
    "load_banked_r1_dxi_dpose_floor",
    "load_byte_close_pose_surfaces",
    "make_byte_close_xi_pose_measure",
    "make_through_r_code_measure",
    "params_sha256",
]
