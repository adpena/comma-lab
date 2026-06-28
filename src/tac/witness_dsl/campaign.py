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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from tac.witness_dsl.curriculum_dsl import Lever, WitnessProgram


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
