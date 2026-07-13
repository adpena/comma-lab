# GO PACKET — in-loop teacher-component timer (resolves the 82%-backward P0 pivot)

**Status:** OPERATOR-GO REQUIRED (bounded governed run; CONTAINMENT — heavy/config launch is operator-gated).
**What it resolves:** the per-epoch accounting (`per_epoch_detailed_accounting_20260713.md`) found the
diagnostic-harness split says the BACKWARD/costate-VJP is ~82% of teacher cost (537 ms fwd / 3009 ms fwd+bwd
per pair) — but that harness is ~12× heavier than the in-loop path, so the RATIO is UNVERIFIED in-loop. This
run measures the REAL in-loop `_teacher_fwd` vs `_teacher_bwd` (+ `_r_fwd` render, `_wit_bwd` witness) split.
The verdict decides whether the entire forward-attack sub-campaign (#455 surrogate-forward, #456
cheaper-forward, #482 ANE-forward — all Amdahl-capped ~1.22× IF backward dominates) was mis-aimed and the P0
target is the backward (feeding p0_sparse_adjoint #P0a / p0_checkpoint_backward #P0b / p0_costate_reuse #P0c
/ the invprob-operator-fold BCR cheap-adjoint). **Single highest-leverage P0 measurement.**

## The instrument (BUILT — no new code needed)
`tac.witness_control.telemetry_producers` (committed) + the da_db D-A `_measure_component_decomposition`
producer wired at `experiments/train_levelset_witness_realized_through_R_mlx.py:9343` (edit UNCOMMITTED —
the deferred hot-file wiring from #480/#408). It emits an 8-field monotonic timer row per profiled epoch:
`_r_fwd` (render-through-R), `_teacher_fwd` (frozen-SegNet forward), `_teacher_bwd` (costate VJP), `_wit_fwd`,
`_wit_bwd`, + loss/verdict. In-loop, real harness, real 1-thread CPU-torch teacher.

## Prerequisite (one of)
- **(A) commit the D-A wiring first** — the deferred #408 reconciliation lands the trainer + resume_registry
  + curriculum_dsl(VerdictLiveGap) edits via a dedicated serializer commit (NOT absorbed into a harvest), then
  run on a clean tree. PREFERRED (clean provenance).
- **(B) run on the current working tree** — the D-A edits are already present uncommitted; a bounded profile
  run would use them as-is. Faster but the run's tree carries other siblings' uncommitted edits (impure
  provenance). Acceptable for a throwaway timing measurement, NOT for a score claim (this is wall-clock only).

## The run (bounded, memory-safe, minutes)
- Config: **n24** (gt_n24), **epochs 4**, muon@4, verdict-pairs 2, the v4/V9 base argv, component-timer ON.
  n24 → the accum loop is per-pair linear, so per-component ms/pair extrapolates ×25 to n600 (legitimate for
  wall-clock accounting, per #306's cross-calibration method).
- Launch path: the governed launcher (`tools/launch_witness_run.py`) with the memory preflight
  (`tools/witness_memory_preflight.py`) — n24 is far under any RAM ceiling (no OOM risk; the #205 OOM was the
  n600 verdict-batch spike, absent at n24 + verdict-pairs 2).
- Governor: standard admission gate (n24 is light; admits trivially).
- Expected cost: ~4 × 6 s/ep pure-step + startup ≈ **1-2 minutes wall-clock**, <10 GiB RSS. $0 (local).

## What to read off
The emitted timer rows → the in-loop `_teacher_fwd : _teacher_bwd : _r_fwd : _wit_bwd` ratio. Decision:
- **backward ≥ ~60% in-loop** → CONFIRM the pivot: 95%-kill target = the costate VJP. The forward arms are
  Amdahl-capped; prioritize p0_sparse_adjoint (BCR low-rank cheap-adjoint) + p0_checkpoint + p0_reuse + #484.
- **forward ≥ ~60% in-loop** → the diagnostic ratio did NOT transfer; the forward arms were correctly aimed;
  keep #455/#456/#482 + ANE-verdict-advisory.
- **mixed (~40-60)** → both matter; #484 (whole-teacher-over-boundary) is the robust hedge regardless.

## Operator action
Reply GO (+ pick prerequisite A or B). Main will: [A] land the D-A reconciliation commit, then launch the
bounded n24 profile via the governed launcher; or [B] launch directly on the worktree. Either way: harvest the
timer rows → append the resolved split to `per_epoch_detailed_accounting_20260713.md` + fire/re-scope the P0
arms on the verdict. NO score claim (wall-clock only); pointer untouched.
