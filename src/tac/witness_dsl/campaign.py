"""Witness A/B campaign engine — the θ* actuator built on the curriculum DSL.

Three pieces that turn the declarative DSL (Layer-0 rails) into the full
here->theta* campaign engine (task #189, operator-approved 2026-06-28):

  1. CYCLIC RECURSION (``Cycle`` + ``expand_cycles``): a cyclic curriculum
     (Muon-prime -> l7-refine -> Muon -> ... , the A12-A15 riff) expands to a
     CHAIN of warm-started ``WitnessProgram`` runs — each resumes from the prior
     cycle's checkpoint. No trainer change needed: recursion = a chain of runs.

  2. HARVEST (``ArmResult`` + ``harvest_arm``): read a run's verdict log, compute
     best/final d_seg + the Δ vs baseline. Pure-read (no GPU); closes the
     measure loop the per-arm protocol needs.

  3. COMPOSE (``select_winners`` + ``compose_theta_star``): bind the measured-
     positive levers into ONE program — theta*. Composition is principled
     because each lever is a term/relaxation of the SAME contest energy S.

CONTAINMENT: ``plan_campaign`` / ``compile`` only EMIT commands (dry-run default).
Actually spawning a GPU arm is the operator's call — this module never auto-fires.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from tac.witness_dsl.curriculum_dsl import Lever, Muon, WitnessProgram


# ---------------------------------------------------------------------------
# 1. CYCLIC RECURSION
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cycle:
    """One cycle of a cyclic curriculum: ``window`` epochs with the levers that
    ``lever_fn(start_epoch)`` returns (a factory so epoch-dependent levers like
    Muon get the correct ``--muon-start-epoch`` for THIS cycle)."""

    name: str
    window: int
    lever_fn: Callable[[int], tuple[Lever, ...]] = lambda _ep: ()


def expand_cycles(
    base: WitnessProgram,
    cycles: list[Cycle],
    *,
    start_resume_from: str,
    start_epoch: int,
    out_dir_prefix: str,
) -> list[WitnessProgram]:
    """Expand a cyclic curriculum into a chain of warm-started programs.

    Program i resumes from program (i-1)'s ``levelset_resume_state.npz`` and runs
    for ``cycles[i].window`` more epochs. This is the cyclic-stage recursion
    (Muon-priming / multiple Muon/l7/CE) as a sequence of real runs.
    """
    programs: list[WitnessProgram] = []
    resume = start_resume_from
    epoch = start_epoch
    for i, cyc in enumerate(cycles):
        out_dir = f"{out_dir_prefix}_cyc{i}_{cyc.name}"
        levers = tuple(cyc.lever_fn(epoch))
        prog = base.with_lever(*levers, resume_from=resume, out_dir=out_dir)
        # window is authoritative for this cycle (overrides any lever epochs_delta)
        prog = replace(prog, epochs=epoch + cyc.window)
        programs.append(prog)
        resume = f"{out_dir}/levelset_resume_state.npz"
        epoch += cyc.window
    return programs


# ---------------------------------------------------------------------------
# 2. HARVEST (the measure loop)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmResult:
    label: str
    seg_form: str | None
    n_verdicts: int
    d_seg_best: float | None
    d_seg_best_epoch: int | None
    d_seg_final: float | None
    d_pose_final: float | None
    blob_bytes_final: int | None
    delta_vs_baseline: float | None  # d_seg_best - baseline; NEGATIVE == improvement
    log_path: str

    @property
    def improved(self) -> bool:
        return self.delta_vs_baseline is not None and self.delta_vs_baseline < 0

    def summary(self) -> str:
        d = self.delta_vs_baseline
        dtxt = "n/a" if d is None else f"{d:+.6f} ({'IMPROVED' if d < 0 else 'worse'})"
        bb = self.d_seg_best
        btxt = "n/a" if bb is None else f"{bb:.6f}@ep{self.d_seg_best_epoch}"
        return (f"[{self.label}] verdicts={self.n_verdicts} best={btxt} "
                f"final={self.d_seg_final} Δvs_base={dtxt}")


def harvest_arm(
    log_path: str | Path,
    *,
    label: str | None = None,
    seg_form: str | None = None,
    baseline_d_seg: float | None = None,
) -> ArmResult:
    """Parse a witness verdict log → ArmResult. Pure-read. ``seg_form`` filters to
    a stage (e.g. 'l7_softplus'); ``baseline_d_seg`` enables Δ computation."""
    path = Path(log_path)
    rows: list[tuple[int, float, float, int | None]] = []
    if path.exists():
        for ln in path.read_text(errors="ignore").splitlines():
            ln = ln.strip().rstrip(",")
            if '"stage": "verdict"' not in ln and '"stage":"verdict"' not in ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("stage") != "verdict" or "d_seg" not in d or "epoch" not in d:
                continue
            if seg_form is not None and d.get("seg_form") != seg_form:
                continue
            rows.append((int(d["epoch"]), float(d["d_seg"]),
                         float(d.get("d_pose", float("nan"))),
                         d.get("blob_bytes")))
    if not rows:
        return ArmResult(label or path.stem, seg_form, 0, None, None, None, None,
                         None, None, str(path))
    best_ep, best = min(((e, s) for e, s, _, _ in rows), key=lambda t: t[1])
    last = rows[-1]
    delta = (best - baseline_d_seg) if baseline_d_seg is not None else None
    return ArmResult(
        label=label or path.stem, seg_form=seg_form, n_verdicts=len(rows),
        d_seg_best=best, d_seg_best_epoch=best_ep,
        d_seg_final=last[1], d_pose_final=last[2], blob_bytes_final=last[3],
        delta_vs_baseline=delta, log_path=str(path),
    )


# ---------------------------------------------------------------------------
# 3. COMPOSE (bind the winners -> theta*)
# ---------------------------------------------------------------------------
def select_winners(
    results: dict[str, ArmResult],
    levers_by_label: dict[str, Lever],
    *,
    threshold: float = -1e-5,
) -> list[Lever]:
    """Pick the levers whose arm improved d_seg by more than ``|threshold|``
    (threshold is negative; Δ must be below it). Band-noise levers are dropped."""
    winners: list[Lever] = []
    for label, res in results.items():
        if res.delta_vs_baseline is not None and res.delta_vs_baseline <= threshold:
            lv = levers_by_label.get(label)
            if lv is not None:
                winners.append(lv)
    return winners


def compose_theta_star(
    base: WitnessProgram,
    winners: list[Lever],
    *,
    resume_from: str,
    out_dir: str,
    epochs: int | None = None,
) -> WitnessProgram:
    """Bind the measured-winner levers into ONE warm-started program = theta*."""
    prog = base.with_lever(*winners, resume_from=resume_from, out_dir=out_dir)
    if epochs is not None:
        prog = replace(prog, epochs=epochs)
    return prog


# ---------------------------------------------------------------------------
# Campaign planning (dry-run; CONTAINMENT — emits, never fires)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    label: str
    program: WitnessProgram
    log: str


def plan_campaign(arms: list[Arm]) -> list[dict]:
    """Validate + compile each arm to its launch command WITHOUT running anything.

    Returns one row per arm: {label, valid, violations, daemon_argv}. Any
    invalid arm (invented flag / behavior violation) is surfaced, not launched.
    """
    plan: list[dict] = []
    for arm in arms:
        violations = arm.program.validate()
        plan.append({
            "label": arm.label,
            "valid": not violations,
            "violations": violations,
            "daemon_argv": arm.program.compile_daemon_argv(label=arm.label, log=arm.log),
        })
    return plan


# ---------------------------------------------------------------------------
# 4. ADAPTIVE STACKING — the REACTIVE curriculum (operator riff 2026-06-29)
# ---------------------------------------------------------------------------
# "Build and stack a curriculum in response to realtime results using our stage/
#  epoch/best/other checkpoints and resumables." We KNOW the opening (S0 seed -> S1
#  CE -> S2 tau_softplus, ``curriculum_dsl.openpilot_seeded_opening``); the NEXT stage
#  is STACKED from the prior stage's MEASURED d_seg trajectory off its checkpoints.
#
# DETERMINISTIC-REPRODUCIBLE BY CONSTRUCTION: ``decide_next_stage`` is a PURE function
# of (the trajectory rows read from disk, the policy thresholds) — no RNG, no wall-clock,
# no env. Same (seed, measured-trajectory) -> same stacked curriculum; the StageDecision
# is returned (serializable) so every decision is RECORDED. CONTAINMENT: ``plan_adaptive_
# step`` only EMITS the next program's launch command (dry-run); it never auto-fires a GPU
# arm. Substrate = the per-stage + EMA-shadow + --resume-from checkpoints (the
# never-launch-non-resumable / per-stage-checkpoint non-negotiable).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StagePolicy:
    """Deterministic thresholds for the reactive stacking decision.

    The decision is read off the trailing-``window`` least-squares slope of the stage's
    realized-through-R d_seg verdicts (per-verdict slope; verdicts land every --eval-every).
    """

    window: int = 4                  # trailing verdicts used to estimate the slope
    plateau_abs_slope: float = 1e-6  # |slope/verdict| below this == PLATEAU -> ADVANCE
    descend_slope: float = -1e-5     # slope below this (steep neg) == still DESCENDING -> EXTEND
    rise_tol: float = 1e-5           # final - best above this == stage RAISED d_seg -> ROLLBACK
    extend_window: int = 300         # epochs added when EXTENDing a still-descending stage
    advance_window: int = 300        # epochs for the ADVANCEd (next) stage


def _trailing_slope(rows: tuple[tuple[int, float], ...], window: int) -> float | None:
    """Least-squares slope of d_seg vs verdict-index over the trailing ``window`` rows.

    Pure (no numpy/MLX) so it never touches a GPU. Returns None if <2 usable rows.
    NEGATIVE slope == d_seg descending (good); ~0 == plateau; positive == rising.
    """
    tail = list(rows)[-max(2, window):]
    n = len(tail)
    if n < 2:
        return None
    xs = list(range(n))
    ys = [r[1] for r in tail]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0.0:
        return None
    return cov / var


@dataclass(frozen=True)
class StageDecision:
    """The recorded (deterministic) decision for what to stack next."""

    action: str            # "ADVANCE" | "EXTEND" | "ROLLBACK_BRANCH"
    reason: str
    slope: float | None
    best_d_seg: float | None
    best_epoch: int | None
    final_d_seg: float | None
    n_verdicts: int
    resume_from: str       # the chosen ckpt: final-state for ADVANCE/EXTEND, BEST for ROLLBACK

    def to_record(self) -> dict:
        return {
            "action": self.action, "reason": self.reason, "slope": self.slope,
            "best_d_seg": self.best_d_seg, "best_epoch": self.best_epoch,
            "final_d_seg": self.final_d_seg, "n_verdicts": self.n_verdicts,
            "resume_from": self.resume_from,
        }


def stage_trajectory(
    log_path: str | Path, *, seg_form: str | None = None,
) -> tuple[tuple[int, float], ...]:
    """Read the (epoch, d_seg) verdict rows from a run log. Pure-read (no GPU).

    ``seg_form`` filters to a single stage (e.g. 'tau_softplus'). Rows are returned in
    file order (== epoch order) so the trailing window is the most-recent verdicts.
    """
    path = Path(log_path)
    rows: list[tuple[int, float]] = []
    if path.exists():
        for ln in path.read_text(errors="ignore").splitlines():
            ln = ln.strip().rstrip(",")
            if '"stage": "verdict"' not in ln and '"stage":"verdict"' not in ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("stage") != "verdict" or "d_seg" not in d or "epoch" not in d:
                continue
            if seg_form is not None and d.get("seg_form") != seg_form:
                continue
            rows.append((int(d["epoch"]), float(d["d_seg"])))
    return tuple(rows)


def decide_next_stage(
    trajectory: tuple[tuple[int, float], ...],
    *,
    policy: StagePolicy,
    final_ckpt: str,
    best_ckpt: str,
) -> StageDecision:
    """PURE deterministic policy: given a stage's measured d_seg trajectory, decide whether
    to ADVANCE (plateau -> stack the next stage + reheat), EXTEND (still descending ->
    longer-per-stage pays), or ROLLBACK_BRANCH (the stage RAISED d_seg -> roll to BEST + skip).

    Same trajectory + policy -> same decision (no RNG/clock/env)."""
    rows = tuple(trajectory)
    if not rows:
        return StageDecision("EXTEND", "no verdicts yet -> extend (conservative)",
                             None, None, None, None, 0, final_ckpt)
    best_ep, best = min(rows, key=lambda t: t[1])
    final_ep, final = rows[-1]
    slope = _trailing_slope(rows, policy.window)
    # 1. the stage RAISED d_seg vs its own best -> roll back to BEST + branch past it.
    if final - best > policy.rise_tol:
        return StageDecision(
            "ROLLBACK_BRANCH",
            f"final {final:.6f} rose >{policy.rise_tol:.0e} above best {best:.6f}@ep{best_ep} "
            "-> rollback to best + branch (skip the d_seg-raising stage)",
            slope, best, best_ep, final, len(rows), best_ckpt)
    # 2. still DESCENDING (steep negative slope) -> EXTEND (longer-per-stage pays, long900).
    if slope is not None and slope <= policy.descend_slope:
        return StageDecision(
            "EXTEND", f"slope {slope:.2e}/verdict <= {policy.descend_slope:.0e} still descending -> extend",
            slope, best, best_ep, final, len(rows), final_ckpt)
    # 3. PLATEAU (|slope| below the flat threshold) -> ADVANCE to the next stage (+ reheat).
    if slope is not None and abs(slope) < policy.plateau_abs_slope:
        return StageDecision(
            "ADVANCE", f"|slope| {abs(slope):.2e}/verdict < {policy.plateau_abs_slope:.0e} plateau -> advance",
            slope, best, best_ep, final, len(rows), final_ckpt)
    # 4. ambiguous gentle descent (between descend and plateau) -> EXTEND (conservative default).
    return StageDecision(
        "EXTEND", f"slope {slope} in ambiguous band -> extend (conservative; longer-per-stage pays)",
        slope, best, best_ep, final, len(rows), final_ckpt)


# --- program transforms (the curriculum ORDER is the caller's; WHEN to advance is the policy's) ---
def extend_stage(
    prev: WitnessProgram, *, resume_from: str, out_dir: str, window: int,
) -> WitnessProgram:
    """EXTEND the current (still-descending) stage: run ``window`` more epochs of the SAME
    stage. The l7 boundary is pushed OUT to the new end (kept a no-op tail) so the extension
    is more tau (not an accidental advance). Warm-starts from the stage's final state."""
    new_epochs = prev.epochs + window
    lev = Lever("extend_stage", overrides={"--l7-start-epoch": new_epochs}, epochs_delta=window,
                notes="extend the current stage (push l7 boundary out; longer-per-stage)")
    return prev.with_lever(lev, resume_from=resume_from, out_dir=out_dir)


def advance_to_l7(
    prev: WitnessProgram, *, resume_from: str, out_dir: str, window: int,
) -> WitnessProgram:
    """ADVANCE to the l7_softplus hard-pixel-refine stage, warm-started from the (plateaued)
    tau state. l7 fires immediately at the resume epoch; reheat (inherited from the opening's
    --stage-transition-* base) fires at the tau->l7 boundary BY CONSTRUCTION."""
    at = prev.epochs
    lev = Lever("advance_l7", overrides={"--l7-start-epoch": at}, epochs_delta=window,
                notes="engage l7_softplus (5x margin<thresh refine) at the resume epoch")
    return prev.with_lever(lev, resume_from=resume_from, out_dir=out_dir)


def advance_to_muon(
    prev: WitnessProgram, *, resume_from: str, out_dir: str, window: int = 100,
    muon_lr: float | None = 2e-3,
) -> WitnessProgram:
    """ADVANCE to the Muon finisher (the spectral-conditioning drop, FEED-fk), warm-started
    from the l7 state. ``muon_lr`` defaults to 2e-3 (FEED-fi's CONSERVATIVE measured band
    1e-3..2e-3 for the flat refinement finisher — NOT the unwired-recall 0.03). None ->
    trainer auto-derives 0.1*lr. Freezes tau + softmax-temp at the l7-end value (clean A/B,
    FEED-fm FIX-2). The Muon lever sets --stage-transition-reset-moments (already on)."""
    at = prev.epochs
    prog = prev.with_lever(Muon(start_epoch=at, window=window), resume_from=resume_from,
                           out_dir=out_dir)
    if muon_lr is not None:
        prog = replace(prog, base={**prog.base, "--muon-lr": muon_lr})
    return prog


def stack_next_program(
    prev: WitnessProgram,
    decision: StageDecision,
    *,
    advance_to: str,
    out_dir: str,
    policy: StagePolicy,
    muon_lr: float | None = 2e-3,
) -> WitnessProgram:
    """Apply the deterministic ``decision`` to ``prev`` -> the next warm-started program.

    ``advance_to`` is the curriculum's KNOWN-next stage ("l7" | "muon"); the POLICY only
    decides extend-vs-advance-vs-rollback. ROLLBACK resumes from the BEST ckpt then advances
    (skipping the d_seg-raising stage). EXTEND resumes from the final state and re-runs the
    same stage longer."""
    resume = decision.resume_from
    if decision.action == "EXTEND":
        return extend_stage(prev, resume_from=resume, out_dir=out_dir, window=policy.extend_window)
    # ADVANCE or ROLLBACK_BRANCH -> engage the next stage (ROLLBACK resumes from BEST).
    if advance_to == "muon":
        return advance_to_muon(prev, resume_from=resume, out_dir=out_dir,
                               window=policy.advance_window, muon_lr=muon_lr)
    if advance_to == "l7":
        return advance_to_l7(prev, resume_from=resume, out_dir=out_dir, window=policy.advance_window)
    raise ValueError(f"advance_to must be 'l7' or 'muon', got {advance_to!r}")


def _resolve_best_ckpt(out_dir: str | Path, best_epoch: int | None) -> str:
    """Resolve the BEST-stage rollback checkpoint from the run dir (deterministic given the
    dir contents): the PRESERVED stage ckpt with the largest epoch <= best_epoch, else the
    nearest preserved ckpt, else the live resume_state."""
    d = Path(out_dir)
    cands: list[tuple[int, str]] = []
    for p in d.glob("levelset_resume_*_ep*.npz"):
        m = re.search(r"_ep(\d+)\.npz$", p.name)
        if m:
            cands.append((int(m.group(1)), str(p)))
    if cands and best_epoch is not None:
        le = [c for c in cands if c[0] <= best_epoch]
        if le:
            return max(le, key=lambda c: c[0])[1]
        return min(cands, key=lambda c: abs(c[0] - best_epoch))[1]
    if cands:
        return max(cands, key=lambda c: c[0])[1]
    return str(d / "levelset_resume_state.npz")


def plan_adaptive_step(
    prev: WitnessProgram,
    prev_out_dir: str | Path,
    log_path: str | Path,
    *,
    advance_to: str,
    out_dir: str,
    next_log: str,
    policy: StagePolicy = StagePolicy(),
    seg_form: str | None = None,
    muon_lr: float | None = 2e-3,
) -> dict:
    """The closed-loop reactive step (EMIT-ONLY, CONTAINMENT): harvest the prior stage's
    measured trajectory from ``log_path`` -> ``decide_next_stage`` (pure) -> build + validate
    the next warm-started program -> return its launch command + the recorded decision.

    Disk reads (the verdict log + the ckpt glob) are deterministic given the on-disk state;
    NOTHING is launched. The returned dict is the operator-routable, recordable step."""
    traj = stage_trajectory(log_path, seg_form=seg_form)
    final_ckpt = str(Path(prev_out_dir) / "levelset_resume_state.npz")
    best_ep = min(traj, key=lambda t: t[1])[0] if traj else None
    best_ckpt = _resolve_best_ckpt(prev_out_dir, best_ep)
    decision = decide_next_stage(traj, policy=policy, final_ckpt=final_ckpt, best_ckpt=best_ckpt)
    nxt = stack_next_program(prev, decision, advance_to=advance_to, out_dir=out_dir,
                             policy=policy, muon_lr=muon_lr)
    violations = nxt.validate()
    return {
        "decision": decision.to_record(),
        "advance_to": advance_to,
        "valid": not violations,
        "violations": violations,
        "next_epochs": nxt.epochs,
        "next_resume_from": nxt.resume_from,
        "daemon_argv": nxt.compile_daemon_argv(label=Path(out_dir).name, log=next_log),
    }
