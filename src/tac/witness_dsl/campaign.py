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
# Flags that change the INPUT-feature COUNT or a WEIGHT SHAPE -> a warm-start resume would
# crash on a shape mismatch -> the pass that changes one MUST be a FRESH arm (resume cleared).
# (Curvelet coarse->fine via --max-bank-freq / --freq-across / --freq-along changes frequency
# VALUES at FIXED feature count => warm-start SAFE; adding bands / capacity changes the count.)
_SHAPE_CHANGING_FLAGS = frozenset({
    "--hidden-dim", "--mod-dim", "--n-hidden",
    "--bank-n-scales", "--bank-n-orient0", "--n-dir-freqs", "--bank-n-iso",
})


@dataclass(frozen=True)
class Cycle:
    """One cycle of a cyclic curriculum: ``window`` epochs with the levers that
    ``lever_fn(start_epoch)`` returns (a factory so epoch-dependent levers like
    Muon get the correct ``--muon-start-epoch`` for THIS cycle).

    MULTI-PASS-with-VARYING-CONFIG (operator enrichment 2026-06-29): ``config`` carries
    per-pass flag overrides so the SAME stage can be re-run with a DIFFERENT config across
    passes (e.g. a 2nd tau pass at tau=0.2; multiple Muon passes at different lr). ``fresh``
    forces a from-scratch pass (no warm-start) — REQUIRED when a pass changes a feature-count /
    weight-shape flag (else the resume crashes on a shape mismatch)."""

    name: str
    window: int
    lever_fn: Callable[[int], tuple[Lever, ...]] = lambda _ep: ()
    config: dict = field(default_factory=dict)
    fresh: bool = False


def expand_cycles(
    base: WitnessProgram,
    cycles: list[Cycle],
    *,
    start_resume_from: str,
    start_epoch: int,
    out_dir_prefix: str,
) -> list[WitnessProgram]:
    """Expand a cyclic curriculum into a chain of warm-started programs (the multi-pass SEARCH).

    Program i resumes from program (i-1)'s ``levelset_resume_state.npz`` and runs for
    ``cycles[i].window`` more epochs. This is the cyclic-stage recursion (Muon-priming /
    multiple Muon/l7/CE / re-run-with-new-config) as a sequence of real runs.

    Per cycle: ``config`` overrides are applied as an explicit config-lever (multi-pass varying
    config); a shape-changing config flag (or ``fresh=True``) makes the pass FROM-SCRATCH;
    l7 is auto-PARKED at the new epoch end UNLESS the cycle explicitly engages it (so a
    same-stage re-run does not accidentally trip the l7 boundary).
    """
    programs: list[WitnessProgram] = []
    resume = start_resume_from
    epoch = start_epoch
    for i, cyc in enumerate(cycles):
        out_dir = f"{out_dir_prefix}_cyc{i}_{cyc.name}"
        levers = tuple(cyc.lever_fn(epoch))
        new_epochs = epoch + cyc.window
        cfg = dict(cyc.config)
        # auto-park l7 (keep it a no-op tail) unless this cycle explicitly sets/engages it
        engages_l7 = ("--l7-start-epoch" in cfg) or any(
            "--l7-start-epoch" in lv.overrides for lv in levers)
        if not engages_l7:
            cfg["--l7-start-epoch"] = new_epochs
        # a shape-changing config flag forces a FRESH arm (else resume shape-mismatch crash)
        fresh = cyc.fresh or any(f in _SHAPE_CHANGING_FLAGS for f in cfg)
        res = None if fresh else resume
        all_levers = levers + ((Lever(f"{cyc.name}_config", overrides=cfg),) if cfg else ())
        prog = base.with_lever(*all_levers, resume_from=res, out_dir=out_dir)
        # window is authoritative for this cycle (overrides any lever epochs_delta)
        prog = replace(prog, epochs=new_epochs)
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
    """Deterministic thresholds for the reactive stacking decision (= task #188 trajectory-
    dynamics early-stop instrument: ADVANCE on PLATEAU, ROLLBACK on REGRESSION, all resumable).

    The decision is read off the trailing-``window`` least-squares slope of the stage's
    realized-through-R d_seg verdicts (per-verdict slope; verdicts land every --eval-every).
    """

    window: int = 4                  # trailing verdicts used to estimate the slope
    plateau_abs_slope: float = 1e-6  # |slope/verdict| below this == PLATEAU -> ADVANCE (early-stop)
    descend_slope: float = -1e-5     # slope below this (steep neg) == still DESCENDING -> EXTEND
    rise_tol: float = 1e-5           # final - best above this == stage REGRESSED -> ROLLBACK to BEST
    extend_window: int = 300         # epochs added when EXTENDing a still-descending stage
    advance_window: int = 300        # epochs for the ADVANCEd (next) stage
    rerun_window: int = 300          # epochs for a RERUN_NEW_CONFIG pass (sharper same-stage config)
    # MULTI-PASS (operator enrichment 2026-06-29): if a stage PLATEAUS but its best d_seg is still
    # ABOVE ``rerun_floor``, RE-RUN the SAME stage with a sharper config (e.g. lower tau) before
    # advancing — the stage has not yet extracted its full value. None == never RERUN (3-action
    # back-compat). ``rerun_floor_by_primer`` makes the floor depend on WHAT PRIMED the stage
    # (priming as a first-class policy input): a dict {primer_stage_name -> floor}.
    rerun_floor: float | None = None
    rerun_floor_by_primer: dict | None = None


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

    action: str            # "ADVANCE" | "EXTEND" | "RERUN_NEW_CONFIG" | "ROLLBACK_BRANCH"
    reason: str
    slope: float | None
    best_d_seg: float | None
    best_epoch: int | None
    final_d_seg: float | None
    n_verdicts: int
    resume_from: str       # the chosen ckpt: final-state for ADVANCE/EXTEND/RERUN, BEST for ROLLBACK
    primed_by: str = ""    # PRIMING (first-class): what stage/config primed THIS stage (recorded)

    def to_record(self) -> dict:
        return {
            "action": self.action, "reason": self.reason, "slope": self.slope,
            "best_d_seg": self.best_d_seg, "best_epoch": self.best_epoch,
            "final_d_seg": self.final_d_seg, "n_verdicts": self.n_verdicts,
            "resume_from": self.resume_from, "primed_by": self.primed_by,
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
    priming: "PrimingContext | None" = None,
) -> StageDecision:
    """PURE deterministic policy (= the task #188 early-stop instrument): given a stage's
    measured d_seg trajectory, decide whether to:
      * ROLLBACK_BRANCH — the stage REGRESSED (final rose above best) -> roll to BEST + skip;
      * EXTEND — still DESCENDING (steep negative slope) -> longer-per-stage pays (long900);
      * RERUN_NEW_CONFIG — PLATEAUED but best still ABOVE ``rerun_floor`` -> re-run the SAME
        stage with a sharper config (multi-pass; the stage hasn't extracted its full value);
      * ADVANCE — PLATEAUED at/below the floor -> stack the NEXT stage (+ reheat).

    ``priming`` (the resume ckpt's primer) is a first-class input: it can select a
    primer-conditioned ``rerun_floor`` and is recorded in ``primed_by``. Same
    (trajectory, policy, priming) -> same decision (no RNG/clock/env) -> a crash-resumed
    early-stop reproduces bit-faithfully (decisions recorded)."""
    rows = tuple(trajectory)
    primed_by = priming.primer_stage if priming is not None else ""
    if not rows:
        return StageDecision("EXTEND", "no verdicts yet -> extend (conservative)",
                             None, None, None, None, 0, final_ckpt, primed_by)
    best_ep, best = min(rows, key=lambda t: t[1])
    final_ep, final = rows[-1]
    slope = _trailing_slope(rows, policy.window)
    # 1. the stage REGRESSED vs its own best -> roll back to BEST + branch past it (early-stop).
    if final - best > policy.rise_tol:
        return StageDecision(
            "ROLLBACK_BRANCH",
            f"final {final:.6f} rose >{policy.rise_tol:.0e} above best {best:.6f}@ep{best_ep} "
            "-> rollback to best + branch (skip the d_seg-regressing stage)",
            slope, best, best_ep, final, len(rows), best_ckpt, primed_by)
    # 2. still DESCENDING (steep negative slope) -> EXTEND (longer-per-stage pays, long900).
    if slope is not None and slope <= policy.descend_slope:
        return StageDecision(
            "EXTEND", f"slope {slope:.2e}/verdict <= {policy.descend_slope:.0e} still descending -> extend",
            slope, best, best_ep, final, len(rows), final_ckpt, primed_by)
    # 3. PLATEAU (early-stop trigger). Resolve the (optionally primer-conditioned) rerun floor.
    if slope is not None and abs(slope) < policy.plateau_abs_slope:
        floor = policy.rerun_floor
        if priming is not None and policy.rerun_floor_by_primer:
            floor = policy.rerun_floor_by_primer.get(priming.primer_stage, policy.rerun_floor)
        # 3a. plateau but best still ABOVE the floor -> RE-RUN this stage with a sharper config.
        if floor is not None and best > floor:
            return StageDecision(
                "RERUN_NEW_CONFIG",
                f"plateau (|slope| {abs(slope):.2e}) but best {best:.6f} > rerun_floor {floor:.6f} "
                "-> re-run SAME stage with a sharper config (multi-pass)",
                slope, best, best_ep, final, len(rows), final_ckpt, primed_by)
        # 3b. plateau at/below the floor (stage exhausted) -> ADVANCE to the next stage (+ reheat).
        return StageDecision(
            "ADVANCE", f"|slope| {abs(slope):.2e}/verdict < {policy.plateau_abs_slope:.0e} plateau -> advance",
            slope, best, best_ep, final, len(rows), final_ckpt, primed_by)
    # 4. ambiguous gentle descent (between descend and plateau) -> EXTEND (conservative default).
    return StageDecision(
        "EXTEND", f"slope {slope} in ambiguous band -> extend (conservative; longer-per-stage pays)",
        slope, best, best_ep, final, len(rows), final_ckpt, primed_by)


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


def rerun_stage_new_config(
    prev: WitnessProgram, *, resume_from: str, out_dir: str, window: int,
    config_overrides: dict,
) -> WitnessProgram:
    """RE-RUN the SAME stage with a SHARPER config (multi-pass varying config), warm-started
    from the plateaued state — e.g. a 2nd tau pass at ``{"--tau-softplus-tau": 0.2}``, or another
    Muon pass at a different ``--muon-lr``. l7 stays PARKED at the new end (still the same stage)
    UNLESS the override engages it. A shape-changing override would crash a warm-start, so this
    transform refuses it (use a fresh ``scale_progression`` pass for capacity/band-count changes)."""
    bad = [f for f in config_overrides if f in _SHAPE_CHANGING_FLAGS]
    if bad:
        raise ValueError(
            f"rerun_stage_new_config got shape-changing flag(s) {bad} -> a warm-start resume would "
            "crash on a shape mismatch; use scale_progression with fresh=True for capacity/band changes")
    new_epochs = prev.epochs + window
    ov = dict(config_overrides)
    if "--l7-start-epoch" not in ov:
        ov["--l7-start-epoch"] = new_epochs  # keep l7 parked (same-stage re-run, not an advance)
    lev = Lever("rerun_new_config", overrides=ov, epochs_delta=window,
                notes="re-run the same stage with a sharper config (multi-pass)")
    return prev.with_lever(lev, resume_from=resume_from, out_dir=out_dir)


def stack_next_program(
    prev: WitnessProgram,
    decision: StageDecision,
    *,
    advance_to: str,
    out_dir: str,
    policy: StagePolicy,
    muon_lr: float | None = 2e-3,
    rerun_config: dict | None = None,
) -> WitnessProgram:
    """Apply the deterministic ``decision`` to ``prev`` -> the next warm-started program.

    ``advance_to`` is the curriculum's KNOWN-next stage ("l7" | "muon"); the POLICY only
    decides extend / advance / rerun / rollback. ROLLBACK resumes from the BEST ckpt then
    advances (skipping the regressing stage). EXTEND / RERUN_NEW_CONFIG resume from the final
    state and re-run the same stage (longer / with a sharper ``rerun_config`` respectively)."""
    resume = decision.resume_from
    if decision.action == "EXTEND":
        return extend_stage(prev, resume_from=resume, out_dir=out_dir, window=policy.extend_window)
    if decision.action == "RERUN_NEW_CONFIG":
        if not rerun_config:
            raise ValueError("RERUN_NEW_CONFIG requires a rerun_config (the sharper same-stage config)")
        return rerun_stage_new_config(prev, resume_from=resume, out_dir=out_dir,
                                      window=policy.rerun_window, config_overrides=rerun_config)
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
    priming: "PrimingContext | None" = None,
    rerun_config: dict | None = None,
) -> dict:
    """The closed-loop reactive step (EMIT-ONLY, CONTAINMENT): harvest the prior stage's
    measured trajectory from ``log_path`` -> ``decide_next_stage`` (pure, priming-aware) ->
    build + validate the next warm-started program -> return its launch command + the recorded
    decision. ``rerun_config`` is the sharper same-stage config used iff the decision is
    RERUN_NEW_CONFIG (multi-pass).

    Disk reads (the verdict log + the ckpt glob) are deterministic given the on-disk state;
    NOTHING is launched. The returned dict is the operator-routable, recordable step."""
    traj = stage_trajectory(log_path, seg_form=seg_form)
    final_ckpt = str(Path(prev_out_dir) / "levelset_resume_state.npz")
    best_ep = min(traj, key=lambda t: t[1])[0] if traj else None
    best_ckpt = _resolve_best_ckpt(prev_out_dir, best_ep)
    decision = decide_next_stage(traj, policy=policy, final_ckpt=final_ckpt, best_ckpt=best_ckpt,
                                 priming=priming)
    nxt = stack_next_program(prev, decision, advance_to=advance_to, out_dir=out_dir,
                             policy=policy, muon_lr=muon_lr, rerun_config=rerun_config)
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


# ---------------------------------------------------------------------------
# 5. SYNERGIES — bind lever COMBINATIONS that COMPOUND (operator enrichment 2026-06-29)
# ---------------------------------------------------------------------------
# theta* compose (select_winners -> compose_theta_star) is the synergy binder. A lever is
# a TERM of the contest action E0; terms INTERACT, so a combo can beat the sum of its parts
# (superadditive) or interfere (subadditive). ``measure_synergy`` records the measured
# pairwise/triple lift; ``select_synergistic_combos`` keeps the superadditive ones to bind.
# d_seg deltas are NEGATIVE-is-better, so a synergy BELOW the (negative) threshold = compounding.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynergyResult:
    combo: tuple[str, ...]            # the lever labels in the combination
    individual_deltas: tuple[float, ...]  # each lever's solo delta_vs_baseline (neg == better)
    expected_additive: float | None  # sum of the solo deltas (the independence null)
    combined_delta: float | None     # the MEASURED combo delta_vs_baseline
    synergy: float | None            # combined - expected_additive (NEG == superadditive/compound)

    @property
    def superadditive(self) -> bool:
        return self.synergy is not None and self.synergy < 0.0

    def summary(self) -> str:
        s = "n/a" if self.synergy is None else (
            f"{self.synergy:+.6f} ({'COMPOUND' if self.synergy < 0 else 'interfere'})")
        return f"[{'+'.join(self.combo)}] expected={self.expected_additive} measured={self.combined_delta} synergy={s}"


def measure_synergy(
    individual: dict[str, ArmResult],
    combo_result: ArmResult,
    combo_labels: tuple[str, ...],
) -> SynergyResult:
    """Pairwise/triple synergy = measured(combo) - SUM(solo deltas). Pure-read (no GPU).

    NEGATIVE synergy == the combo improved d_seg MORE than the independence null == compounding
    (the E0 terms reinforce). POSITIVE == interference. None when a delta is missing."""
    deltas = tuple((individual[l].delta_vs_baseline if l in individual else None)
                   for l in combo_labels)
    if any(d is None for d in deltas) or combo_result.delta_vs_baseline is None:
        return SynergyResult(combo_labels, tuple(d for d in deltas if d is not None),  # type: ignore[arg-type]
                             None, combo_result.delta_vs_baseline, None)
    expected = sum(deltas)  # type: ignore[arg-type]
    combined = combo_result.delta_vs_baseline
    return SynergyResult(combo_labels, deltas, expected, combined, combined - expected)  # type: ignore[arg-type]


def synergy_map(
    individual: dict[str, ArmResult],
    combos: dict[tuple[str, ...], ArmResult],
) -> dict[tuple[str, ...], SynergyResult]:
    """Build the pairwise/triple synergy map: {combo_labels -> SynergyResult}. Deterministic."""
    return {labels: measure_synergy(individual, res, labels) for labels, res in combos.items()}


def select_synergistic_combos(
    smap: dict[tuple[str, ...], SynergyResult],
    levers_by_label: dict[str, Lever],
    *,
    threshold: float = -1e-6,
) -> list[tuple[Lever, ...]]:
    """Keep the COMPOUNDING combos (synergy <= threshold, threshold negative) -> the lever tuples
    to bind via ``compose_theta_star``. Combos with a missing lever are dropped."""
    out: list[tuple[Lever, ...]] = []
    for labels, res in smap.items():
        if res.synergy is not None and res.synergy <= threshold:
            levs = tuple(levers_by_label[l] for l in labels if l in levers_by_label)
            if len(levs) == len(labels):
                out.append(levs)
    return out


# ---------------------------------------------------------------------------
# 6. PRIMING — the warm-start chain IS priming (operator enrichment 2026-06-29)
# ---------------------------------------------------------------------------
# Each stage PRIMES the next stage's loss landscape (seed primes all; CE primes tau; soft
# primes sharp). Priming is a FIRST-CLASS policy input: a stage's expected value (and its
# rerun_floor) depends on WHAT primed it (the resume ckpt's predecessor stage/config). The
# PrimingContext is recorded (StageDecision.primed_by) and queryable (priming_chain).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrimingContext:
    primer_stage: str                       # the stage/config that produced the resume ckpt
    resume_ckpt: str                        # the ckpt that primes THIS stage
    primer_final_d_seg: float | None = None  # the primer stage's final measured d_seg
    primer_config: dict = field(default_factory=dict)  # the primer's config (e.g. {"--tau-softplus-tau":0.3})


def priming_chain(programs: list[WitnessProgram]) -> tuple[dict, ...]:
    """Extract the priming chain (each program's out_dir <- the ckpt that primed it). Pure.
    The chain makes 'seed -> CE -> tau -> l7 -> Muon' EXPLICIT and queryable for max signal."""
    return tuple({"out_dir": p.out_dir, "primed_by_ckpt": p.resume_from,
                  "from_scratch": p.resume_from is None} for p in programs)


# ---------------------------------------------------------------------------
# 7. SCALING = the CURVELET multi-scale COARSE->FINE band climb (operator enrichment 2026-06-29)
# ---------------------------------------------------------------------------
# The SCALE dimension of the search is a coarse->fine MULTI-SCALE CURVELET climb (the #1 MEASURED
# d_seg lever: all-class DIRECTIONAL/anisotropic curvelet basis on the boundary tangent = -48%
# d_seg, ~0 byte, DAG FEED 2026-06-25t; Candes-Donoho 2004 curvelet edge-basis O(N^-2(logN)^3)
# for cartoon (C2-curve + cross-edge step) functions, FEED 2026-06-25j; research:
# .omx/research/levelset_curvelet_witness_feasibility_20260627.md). The openpilot seed gives the
# COARSE/low-freq lane structure free (NTK); the curriculum CLIMBS to finer curvelet bands oriented
# to the boundary annulus. Substrate: tac.torch_vehicle.boundary_routing.{local_boundary_tangent,
# directional_positional_encoding} (the oriented-PE helper; 0 archive bytes => rule-118 FREE).
#
# WARM-START SAFETY (critical): --max-bank-freq / --freq-across / --freq-along change frequency
# VALUES at a FIXED feature count => the first-layer input dim is unchanged => warm-start SAFE
# (the coarse->fine band climb). --bank-n-scales / --n-dir-freqs / --bank-n-orient0 / --bank-n-iso
# (more bands) and --hidden-dim / --mod-dim (capacity) change the COUNT/shape => a warm-start
# resume would CRASH => those passes MUST be FRESH (auto-forced via _SHAPE_CHANGING_FLAGS).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalePass:
    """One coarse->fine pass: ``window`` epochs with ``overrides`` (e.g. a finer curvelet band
    via --max-bank-freq). ``fresh`` (from-scratch) is auto-forced when an override changes a
    feature-count/weight-shape flag (else a warm-start resume crashes on a shape mismatch)."""

    name: str
    window: int
    overrides: dict = field(default_factory=dict)
    fresh: bool = False

    @property
    def is_fresh(self) -> bool:
        return self.fresh or any(f in _SHAPE_CHANGING_FLAGS for f in self.overrides)


def curvelet_scale_passes(
    max_bank_freqs: tuple[float, ...],
    *,
    window: int,
    freq_across: float | None = None,
    freq_along: float | None = None,
) -> list[ScalePass]:
    """Build the coarse->fine CURVELET band schedule: one warm-start-safe pass per finest-band
    frequency in ``max_bank_freqs`` (ascending == coarse->fine), each raising --max-bank-freq
    (value-only => same feature count => warm-startable). Optionally pin the anisotropy ratio
    (--freq-across HIGH across the edge / --freq-along LOW along it = the -48% directional lever).
    REQUIRES --self-orient ON in the base program (the directional/curvelet basis enable; there is
    NO --use-dir flag -- recorded trainer-gap). All passes are WARM (no shape change)."""
    passes: list[ScalePass] = []
    for i, f in enumerate(max_bank_freqs):
        ov: dict = {"--max-bank-freq": f}
        if freq_across is not None:
            ov["--freq-across"] = freq_across
        if freq_along is not None:
            ov["--freq-along"] = freq_along
        passes.append(ScalePass(f"curvelet_band_{f:g}", window=window, overrides=ov))
    return passes


def scale_progression(
    base: WitnessProgram,
    passes: list[ScalePass],
    *,
    start_resume_from: str,
    start_epoch: int,
    out_dir_prefix: str,
) -> list[WitnessProgram]:
    """Expand a scaling progression into a warm-started chain (the SCALE axis of the search).

    Pass i resumes from pass (i-1)'s ckpt and runs ``passes[i].window`` more epochs at the new
    scale. A FRESH pass (capacity / band-count change) is from-scratch (resume cleared) — and the
    NEXT pass then resumes from THIS fresh pass's ckpt. l7 is auto-PARKED at the new end (each
    pass re-runs the current stage at a finer scale, not an advance). This realizes the curvelet
    coarse->fine climb as real, deterministic, emit-only programs."""
    programs: list[WitnessProgram] = []
    resume = start_resume_from
    epoch = start_epoch
    for sp in passes:
        out_dir = f"{out_dir_prefix}_{sp.name}"
        new_epochs = epoch + sp.window
        ov = dict(sp.overrides)
        if "--l7-start-epoch" not in ov:
            ov["--l7-start-epoch"] = new_epochs  # keep l7 parked (same-stage finer-scale re-run)
        res = None if sp.is_fresh else resume
        prog = base.with_lever(Lever(sp.name, overrides=ov), resume_from=res, out_dir=out_dir)
        prog = replace(prog, epochs=new_epochs)
        programs.append(prog)
        resume = f"{out_dir}/levelset_resume_state.npz"
        epoch += sp.window
    return programs
