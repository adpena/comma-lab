# Build specification — timer curriculum completion

Date: 2026-07-13
Owner/lane: `timer_curriculum_complete`
Authority: local build, static validation, tests, and governed-launcher dry-run only. Heavy Metal boot/run remains operator-GO. This timer is throughput apparatus/MEANS; only a byte-closed exact evaluator row can move a contest pointer.

## Reproduced defect

- `compile_throughput_component_timer_ticket()` currently refuses before launch because the partial `--muon-start-epoch=4` repair collides with live LADDER windows 340/260.
- The v7.5.2 parent at `epochs=4` reports zero `WitnessProgram.validate()` violations while emitting `--curriculum`, Muon/pose starts 726, lane/chroma/screw starts 450–500, Polyak start 5, and live LADDER windows. This is the config-build-time feasibility hole.
- The trainer master `--curriculum` flag gates the discrete CE/tau/l7 curriculum, but it does not independently disable Muon, pose finish, LADDER, Polyak, birth forces, or other composed levers. The timer must therefore strip those stage/levers by construction, not merely toggle the master.

## Required implementation

### 1. Timer becomes a one-stage CE-only program

Edit `src/tac/witness_dsl/spec_throughput_component_timer_20260713.py`.

- Start from the already compiled v7.5.2 parent so the exact witness geometry/perf environment remains the measured component surface.
- Rebuild the typed program as a timer-specific program with:
  - `base["--curriculum"] = False`, compiling to the real `--no-curriculum` BooleanOptionalAction;
  - curriculum event/nucleus/reanchor booleans explicitly false or removed so no event controller is presented as active;
  - `stages=()` (or an equivalent CE entry-only stage set) so Muon and pose-finish start epochs are not emitted;
  - inherited score/curriculum levers removed; compose only the timer treatment lever;
  - `--w-pose=0.0` and every already identified non-CE loss/regularizer weight at zero;
  - no per-stage Muon/l7 cap overrides;
  - LADDER, Polyak, area/birth controllers, and other non-CE stage actuators absent or explicitly off by a real BooleanOptionalAction;
  - tau frozen at 1 and `--seg-form-unify-tau` retained, so the active segmentation objective is exactly `L_tau(tau=1)=CE`.
- Add fail-closed assertions over the emitted flag map proving curriculum is off, no stage-start epoch is emitted, no parent lever survives, and the expected zero-weight objective contract holds.
- Preserve async-versus-solo as the only matched difference.

### 2. Config-build-time schedule feasibility class guard

Edit `src/tac/witness_dsl/curriculum_dsl.py` and minimally wire the launcher if necessary.

- Add a pure helper that evaluates effective schedule feasibility from emitted `(flag, value)` pairs or a `WitnessProgram` flag dict using the trainer’s real argparse definitions/defaults.
- Law: for an enabled curriculum, every effective/emitted `--*-start-epoch` stage/cap (excluding `--warm-start-epoch`, which is resume metadata) must satisfy `start_epoch <= epochs`. Event governance does not waive the cap’s obligation to exist within the configured run budget. If curriculum is explicitly disabled, the curriculum-family budget law is vacuous; independent active finisher/lever guards remain the trainer’s own responsibility and the timer strips them.
- The helper must surface all offending flags and values in one clear violation message containing config epochs and the remediation: disable curriculum for a true single-stage program or use a feasible schedule.
- Invoke it from `WitnessProgram.validate()` so typed configs fail before `_rebind_typed()`/launcher construction.
- Invoke the same helper immediately after `derive_named_config()` in `tools/launch_witness_run.py`, before launch.sh or run-directory writes, so legacy/non-migrated named configs such as `proven_base` cannot bypass it. This check is strict even for `--dry-run`; it is a compiler/config construction invariant, not an advisory launch gate.
- Do not edit the trainer for this landing.

### 3. Tests

- Extend `src/tac/tests/test_spec_throughput_component_timer_20260713.py`:
  - both variants compile;
  - `--no-curriculum` is emitted and `--curriculum` is not;
  - no `--*-start-epoch` tokens remain;
  - no inherited parent lever survives;
  - all non-CE weights including pose are zero;
  - async/solo differ only by the BooleanOptional async flag;
  - schedule-provenance remains clean.
- Add a focused new test module for the class guard:
  - curriculum enabled, epochs 4, stage start 726 -> violation;
  - multiple out-of-budget starts are all reported;
  - stage start equal to epochs -> pass;
  - curriculum explicitly off -> pass;
  - default sealed named configs in the requested families pass at their sealed budgets;
  - short-epoch `proven_base` and at least one typed v7/V9 config refuse before launch construction;
  - both timer configs pass the generic helper.
- Keep tests pure: no Metal, daemon, training, evaluator, or paid dispatch.

### 4. Triality equation

Add `src/tac/canonical_equations/curriculum_epoch_budget_feasibility_20260713.py` plus tests.

- Equation identifier: `curriculum_epoch_budget_feasibility_v1`.
- Define the derived feasibility margin `m_sched = E - max(S_active)` for an enabled curriculum, feasible iff `m_sched >= 0`; disabled curriculum returns an explicit vacuous/pass state rather than pretending a schedule was measured.
- Include verdict scope: config/boot-runnability only; it says nothing about training quality, score, archive bytes, or promotion.
- Include req-R: re-run compile audit whenever trainer stage-start argparse, defaults, or named config schedule ownership changes.
- Keep this as a pure canonical equation entity; do not mutate the dirty shared equation registry in this code-writing slice.

## Non-goals / containment

- No trainer edits.
- No `witness_control/*` edits.
- No heavy Metal boot/run, daemon, evaluator, archive, or live-run mutation.
- Preserve sibling-owned dirty files and do not stage or revert them.
- Do not write the final research memo or DAG FEED in the implementation slice; the parent reviewer will generate them from measured dry-run/audit evidence.

## Acceptance before handoff to reviewer

- Targeted pytest and Ruff pass.
- A pure compile of both timer variants succeeds.
- Return exact changed file list and any caveat; do not commit.
