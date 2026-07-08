# Resume registry #358: fold the four non-gate controllers under the static gate (2026-07-08)

STORES CONSULTED (proactive recall before building):
- CLAUDE.md non-negotiables: "resumability + per-stage checkpoints MANDATORY", "deterministic
  reproducibility" (resume == continuous, seeded, bit-faithful), "Bugs must be permanently fixed AND
  self-protected against" (fix + static gate), "Confound self-protection", "Subagent commits MUST use
  serializer" (+ `--patch-file` intent-manifest for the hot trainer).
- `.omx/research/resume_registry_canonical_20260708.md` (the predecessor architecture — the gate
  registry + `Resumable` protocol + manifest; its named RESIDUAL RISK was exactly the four non-gate
  controllers "remain on their own hand-written pairs ... NOT yet routed through the registry, so the
  static class-gate does not cover them"). This task closes that residual.
- `src/tac/witness_control/resume_registry.py` (ResumeRegistry, TrailingSeriesResumable,
  build_gate_resume_registry, manifest logic).
- `src/tac/witness_control/polyak_finisher.py` + `.omx/research/r7_finishers_20260708.md` (the polyak
  precedent: it ALREADY rides the registry — scalar sentinel via `state_arrays`/`restore_from_cfg`
  under `__pta_`; heavy running-mean inline under `polyakM__`, one atomic savez).
- the trainer's `_build_resume_state_arrays` / `_load_resume_state` / `_cl_state_arrays` /
  `_cl_sidecar_snapshot` / `_rng_state_arrays` / `tau_advance_state_arrays` / inline `__evt_*` block.
- docs/operating_manual_craft_handoff.md (do LESS but REAL; point-fix != class-fix; attack your own
  conclusion; label MEASURED/DERIVED).

## The residual this closes

The registry (commit `2b7332f4b`) covered the `EventBackstopGate` class + the polyak scalar. FOUR
stateful controllers still rode hand-written `(state_arrays, restore_from_cfg)` pairs OUTSIDE the
registry, so a NEW controller of their shape could still be forgotten at the WRITE surface (the exact
recurrence class the registry exists to kill):

| controller | prefix | write (before) | restore (before) |
|---|---|---|---|
| rng streams | `__rng_` | `_rng_state_arrays` @ separate call site | `_restore_rng_state` inline |
| closed-loop | `__cl_` | `_cl_state_arrays` inline in `_build_resume_state_arrays` | `_cl_restore_from_cfg` inline |
| tau-advance | `__ta_` | `tau_advance_state_arrays` inline | `tau_advance_restore_from_cfg` inline |
| evt-curriculum | `__evt_` | inline block in `_build_resume_state_arrays` | inline (+ cap-fallback) |

## What landed

**Commit 1 `51ae8ea8d`** (`resume_registry.py` + tests, self-contained apparatus):
- `FunctionResumable(write, restore)` — a generic, NON-EVENT `Resumable` adapter delegating to the
  canonical free functions (thin; byte-identical keys by construction).
- Manifest stamp rule REFINED: stamp `__resume_registry_manifest` iff an **event-active** controller
  wrote (was: iff ANY controller wrote). This is what makes folding the always-on non-event rng (and
  opt-in cl/tau/evt) byte-identical — a cap-only run adds NO manifest to a config that had none. When a
  manifest IS stamped (an event gate wrote), it still lists ALL who wrote → non-event vanish is a LOUD
  warning, never the event fail-closed.
- 5 tests (protocol, non-event flag, roundtrip byte-identity, no-manifest-for-non-event, manifest lists
  non-event when event gate wrote, vanished-non-event warns-not-raises).

**Commit 2 `7834cda31`** (trainer via `--patch-file` HEAD-seeded + tests):
- Extracted `_evt_state_arrays` / `_evt_restore_from_cfg` (verbatim from the former inline block — same
  keys) so evt has a named producer the static gate can see.
- SINGLE-SOURCED the WRITE: removed the evt/cl/tau blocks + params from `_build_resume_state_arrays` and
  the separate rng update; registered all four as `FunctionResumable`s into `_resume_registry` (one
  block, `# NON-GATE RESUME FOLD (#358)`). `registry.state_arrays()` is now the sole writer.
- WIDENED static gate `test_every_nongate_state_arrays_producer_is_registered`: asserts the four
  canonical registrations (name+prefix) AND that every `*_state_arrays` producer in the trainer (minus
  the model-tensor builder) is wired into the fold region → a new non-gate producer cannot ship
  unpersisted.
- +byte-identity test (`test_nongate_registry_byte_identical_to_legacy_direct_calls`: registry-emitted
  keys+arrays == legacy direct free-function calls, NO manifest) + evt keys/roundtrip test.
- Updated 2 stale source-string assertions in `experiments/test_closed_loop_control.py` that pinned the
  removed call-site string (faithfully re-pointed to the new `_cl_write`/adapter contract + a
  single-caller count guard).

## Per-controller before → after

| controller | WRITE before | WRITE after | RESTORE (unchanged) |
|---|---|---|---|
| rng | `_rng_state_arrays(hardness_rng)` @ separate line | `FunctionResumable(write=lambda: _rng_state_arrays(hardness_rng))` in registry | inline `_restore_rng_state` @ its ordered site |
| closed-loop | `_cl_state_arrays(_cl_sidecar_snapshot(), ...)` inline | `_cl_write` adapter (same snapshot, `_cl_on`-gated) | inline `_cl_restore_from_cfg` + pending-reconcile |
| tau-advance | `tau_advance_state_arrays(_tau_ctrl)` inline | `FunctionResumable(write=lambda: _ta_write(_tau_ctrl))` | inline `tau_advance_restore_from_cfg` |
| evt-curriculum | inline `__evt_*` dict | `_evt_state_arrays(_evt_state if _evt_on else None)` adapter | inline (with-keys branch + cap-fallback) |

**Write/restore split (deliberate, documented):** the four are constructed AFTER the single early
`_resume_registry.restore` (hardness_rng@L6495, _tau_ctrl@L6766, _evt_state@L6707, _cl_state@L5580 are
below the L6274 restore), so the registry drives their WRITE + completeness/manifest while their RESTORE
stays at its correctly-ordered inline site (evt cap-fallback, cl pending-reconcile, rng ordering, all
telemetry-coupled). The adapters' `restore_from_cfg` is the protocol-conformant, unit-tested delegate;
the trainer's inline path is authoritative. Moving restore into the registry would require moving the
four's construction before the finisher-decision restore — invasive + risky to the live run — for no
correctness gain (write-completeness + static coverage is the recurrence-killer, and it is WRITE-side).

## Key-name-preservation proof method

`test_nongate_registry_byte_identical_to_legacy_direct_calls` imports the trainer module and asserts
`set(registry.state_arrays()) == set({**_evt_state_arrays(evt), **_cl_state_arrays(cl,v), **_rng_state_arrays(rng)})`
with per-key array-value equality AND `__resume_registry_manifest NOT in` the emitted dict (cap-only
gates + non-event controllers). Because each adapter delegates verbatim to the same free function, keys
are identical by construction; the test is the guard against future drift.

## Legacy-compat proof (the LIVE run — MEASURED, read-only, run dir untouched)

The live run `levelset_n600_crucible_v6_run1_20260708T095730Z` (pid 63069) is `--curriculum-event-triggered`,
cap-only gates, polyak NOT armed. Its ACTUAL sidecar carries `__rng_`(6) + `__evt_`(5) + NO manifest.
- Parsed the real sidecar (read-only) and ran `build_gate_resume_registry([cap-only gates]).restore(cfg)`
  on my landed code: `legacy=True, manifest_present=False, warnings=0, gates={muon:False, lane_band:False,
  seg_chroma_boundary:False}` — NO raise, identical cap-fallback. The inline rng/evt restores handle the
  `__rng_`/`__evt_` keys unchanged.
- Its FUTURE sidecar under my code: rng writes `__rng_`, evt writes `__evt_` (evt_on), cl/tau/gates → {}
  (off/clock/cap-only), no event-active wrote → NO manifest → byte-identical key set. If it crashes it
  resumes bit-faithfully on this code.

Pre-registry event-muon sidecars still restore (`test_legacy_pre_registry_muon_sidecar_still_restores`,
unchanged). Polyak was ALREADY fully folded (scalar registry + heavy inline by design) — no change
needed; the only interaction is that a polyak-armed, no-event-gate run no longer stamps a solo-polyak
manifest, which is functionally inert (polyak is fail-OPEN: missing scalar → restart, same as a
non-fatal vanish warning) and affects NO live run.

## Tests / gates

- `src/tac/tests/test_resume_registry.py`: 24 pass (16 pre-existing + 5 commit-1 + 3 commit-2).
- `experiments/test_closed_loop_control.py`: 22 pass, 2 FAIL — **pre-existing, NOT mine** (both assert
  `src.index("v = realized_verdict()")`, a symbol a sibling refactor removed from the trainer; verified
  FAILING on HEAD with my change stashed). Out of scope.
- tau_advance (20) + fixall_wave_a (31) + warm_start (16) suites: all pass.
- ruff `--select F` clean on all touched files. Trainer imports cleanly; AST OK.
- review gate: 2 clean passes on all `.py`; serializer commits (commit 1 normal, commit 2 `--patch-file`
  HEAD-seeded); post-commit stat verified NO foreign sibling absorption on either commit.

## Sibling collisions

None. Siblings landed `1e2978251` (micro-batch bit-identity probe), `8cf226e18`/`3f8ce1ea7` (D1 GPU
verdict probe / crucible ledger) around my work — none touched the trainer resume machinery; the
`--patch-file` HEAD-seeded commit carried exactly my 3 files.

## Pointer

**0.19110 UNMOVED.** This is APPARATUS / means (resumability write-completeness + static coverage), not
a score-mover — only a byte-closed n600 `upstream/evaluate.py` row < 0.19110 (contest-CPU/CUDA, NEVER
MPS) moves the pointer.
