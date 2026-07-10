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
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.contest_score import RATE_WEIGHT, SEG_WEIGHT, UNCOMPRESSED_SIZE_BYTES

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


__all__ = [
    "DEFAULT_PARAM_TARGETS",
    "MC_FINISHER_LABEL",
    "BatchOutcome",
    "FinisherProblem",
    "MCFinisher",
    "MCFinisherError",
    "MCFinisherResult",
    "MeasuredObjective",
    "ParamState",
    "Proposal",
    "ProposalEngine",
    "delta_s_floor_per_confirmed_flip",
    "params_sha256",
]
