# Task #537 implementation spec — resumability seal path

## Objective

Close the three owed resumability gaps in
`experiments/train_levelset_witness_realized_through_R_mlx.py` without changing score,
pointer, provider, or launch authority:

1. Every periodic intra-stage checkpoint writes a distinct stage-and-epoch-encoded EMA
   deploy checkpoint and a matching full resume-state checkpoint, atomically, while the
   rolling aliases remain for compatibility. Retain only the newest configurable `M`
   periodic pairs per stage; never delete stage-boundary/final checkpoints.
2. A normal crash continuation fails closed unless live weights, EMA shadow, optimizer
   moments, RNG state, event-state custody, and stage/epoch position are present. The
   explicit `--warm-start-weights-only` re-treatment remains the only path allowed to drop
   optimizer/state continuity. Legacy full-state checkpoints may pass only when each
   required semantic leg is directly evidenced by their keys; emit a loud compatibility
   row, never silently assume state.
3. Resume-round schedules/events are re-anchored for an intentional warm-start re-treatment
   to the restored start epoch and current beta2/steps-per-epoch geometry. A normal
   bit-faithful continuation restores the event ledger and must not move already-fired
   anchors. Reuse the existing `adam_v_variance_warmup_length_v1` / #518 geometry rather
   than inventing a constant. Emit one machine-readable re-anchor row containing old/new
   anchors, resume epoch, derived window, and treatment kind.
4. Add a warn-only static preflight check that scans the live trainer's checkpoint writes
   and reports any save path which can overwrite a periodic full-state checkpoint without a
   distinct stage+epoch preservation path. Wire it into `preflight_all(..., strict=False)`
   and test clean and violating source fixtures.

## Constraints

- Minimal additive diff; trainer is hot.
- No writes to `experiments/results/levelset_n600_witness_20260717T113932Z/`.
- No paid/remote dispatch, no exact score, no pointer mutation.
- Use `tmp + os.replace` for every checkpoint write. Keep EMA shadow as the deploy artifact.
- Bounded retention deletes only periodic files produced by this new naming grammar; never
  rolling aliases, BEST, stage-boundary, final, or unknown files.
- Preserve old CLI behavior where safe. Add a typed integer CLI knob such as
  `--ckpt-retain-per-stage` with a sane positive default and validation.
- Keep prose free of provider-filter trigger vocabulary called out in the authority brief.
- Do not commit; the parent session reviews and commits.

## Files owned

- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `experiments/tests/test_levelset_checkpoint_resume.py`
- `experiments/tests/test_levelset_crash_resume_smoke.py` only if needed for the n24 receipt
- `src/tac/preflight.py`
- one focused new preflight test under `src/tac/tests/`

Do not touch upstream, launcher/provenance hot surfaces, the frontier pointer, or unrelated
DSL/spec files.

## Acceptance criteria

- Pure tests prove distinct periodic names, paired EMA/resume preservation, per-stage bounded
  retention, stage-boundary immunity, required-state refusal per missing leg, legacy semantic
  validation, exact continuation leaves anchors unchanged, and warm re-treatment re-anchors
  against the derived current geometry.
- The existing resume registry static coverage remains green.
- The warn-only preflight reports a deliberately overwriting fixture and reports zero live
  violations.
- The real n24 crash-resume command can use the existing real cache at
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n24.npz`, a tiny real
  epoch budget, `--mlx-device cpu`, and an isolated worktree-local relative out-dir. It must
  actually terminate after a periodic checkpoint, resume, and compare continuous/resumed
  telemetry plus live/EMA tensor hashes and preserved checkpoint loadability.
- A copy of the sacred run's real full sidecar passes the semantic guard (or records an exact,
  honest missing-leg refusal if the source predates that leg); a copy with one required leg
  removed refuses. Record source/copy hashes without modifying the source.

## Required verification commands

```text
python3 -m py_compile experiments/train_levelset_witness_realized_through_R_mlx.py src/tac/preflight.py
.venv/bin/python -m pytest experiments/tests/test_levelset_checkpoint_resume.py -q
.venv/bin/python -m pytest src/tac/tests/test_resume_registry.py -q
.venv/bin/python -m pytest src/tac/tests/test_levelset_checkpoint_save_path_preflight.py -q
```

Do not claim the mandatory real receipt from unit fixtures. The parent session will run and
review the n24 crash/resume and sacred-layout copy proofs after implementation.
