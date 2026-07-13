# D-A/D-B telemetry producers + task #408 Q1-Q7 — build receipt — 2026-07-13

`research_only=false` · `score_claim=false` · `promotion_eligible=false` · `pointer_moved=false`  
Lane: `lane_da_db_producers_20260713` · checkpoint: `da_db_producers` · budget: `$0 local`

## Outcome

- **D-A rows emitting: YES (code-complete).** The trainer now emits
  `witness_component_wallclock.v1` once per epoch through fcntl-locked JSONL, with exactly:
  `teacher_forward_s`, `teacher_backward_s`, `witness_forward_s`, `witness_backward_s`,
  `realized_R_s`, `verdict_s`, `checkpoint_io_s`, and `epoch_total_s`.
- **D-B hook wired: YES (code-complete).** The trainer observes screw engagement at ep450 and phase
  engagement at ep726, at each boundary +/- a configurable window (default 2), with deterministic
  n600 K=4 strata `(75,225,375,525)`. Actual transition events supersede a same-epoch nominal event,
  preventing duplicate expensive observations.
- **Q1-Q7 landed: 7/7.** All pure telemetry defaults ON. The only score-affecting route,
  `--verdict-live-gap-every`, is held by the typed `VerdictLiveGap` DSL lever and defaults OFF at zero.
- **Launch-ticket result:** both canonical variants clear
  `D_A_EXACT_COMPONENT_TIMERS_MISSING` and `D_B_EXACT_ENGAGEMENT_HOOK_MISSING`. Both remain governed
  `REFUSE_RC11_NO_SPAWN` on the independent B2 and SSD blockers. This landing does not claim launch GO.

These are producer-existence results, not measured wall-split or engaged-gradient results. A real
resumed n600 run must emit the rows before either can become MEASURED evidence.

## D-A implementation

`ComponentWallclock` uses `time.perf_counter_ns()` and per-epoch aggregation. The training update path
is not retimed by splitting or replacing its forward/backward calls: an update-free one-pair observer
computes the five decomposition terms against the same frozen teacher/witness/R surfaces, under an RNG
guard. Real verdict duration is recorded in both sync and async paths. Real stage/best/final checkpoint
I/O is added to the matching epoch, and the final epoch row is deferred until the mandatory final
checkpoint is written.

Every row distinguishes `real`, `structural-zero`, `missing`, and `error` observations. A later real
measurement removes an earlier not-invoked marker. `epoch_total_s` is the monotonic epoch envelope;
the component fields are diagnostic sums and are not asserted disjoint when verdict work is async.
The row is also mirrored as a no-score causal-manifest boundary event by the existing manifest composer.

## D-B implementation and verdict scope

The hook reuses the exact math and output vocabulary of
`tools/probe_sps_gradient_role_conflict.py`: trunk gradients are flattened, zero-norm cosine is null,
coactivity uses `1e-12 * max(abs(g), 1)`, and the preregistered conflict predicate is global cosine
`<= -0.05` plus negative-product scalar share `>= 0.10`. A parity test compares the new NumPy reducer
directly with the standalone Torch implementation.

The primary SPS discriminator is seg-vs-temporal; pose-vs-temporal and fully armed
seg+pose-vs-temporal rows are retained in the same payload. The hook temporarily activates the named
temporal gate, restores it, preserves optimizer/EMA/RNG state, filters only the declared witness trunk
prefixes, and latches a resume event only after successful emission. The axis is explicitly
`[macOS-MLX gradient observer] NON-PROMOTABLE`; verdict scope is engaged-regime mechanism diagnostic,
not score or family authority.

## Task #408 Q1-Q7 closure

| Q | Emission | State |
|---|---|---|
| Q1 | one per-epoch global/per-group `grad_clip_activation` row | landed, default ON |
| Q2 | chroma + stacked levers in term domination; sustained `term_inert` alarm | landed, default ON |
| Q3 | exact live-vs-EMA verdict gap at `--verdict-live-gap-every K` | landed, DSL default OFF |
| Q4 | explicit `tail_cycle_endpoint` | landed, default ON |
| Q5 | powerlaw-meat and annulus-plateau would-fire rows | landed, default ON |
| Q6 | resume-safe `ladder_birth_complete` | landed, default ON |
| Q7 | uniform `{stage:"lever_engage", lever, status, epoch, via}` companions | landed, default ON |

The Q3 pool row is `built-never-fired`; no activation or measurement event is fabricated. The existing
SPS reformulation row remains `reformulation-queue`, now pointing at the real engaged n600 boundary gate.
Both are appended with canonical `record_candidate` using hyphenated status values.

## Resume and replay safety

- `da_db_telemetry` is a direct controller in the canonical resume registry.
- Its state is additive and versioned: epoch, emitted boundary keys, ladder-completion keys, and uniform
  engagement keys. Missing legacy state loads deterministic empty defaults.
- Existing run-log readers ignore unknown additive stages. Existing thread-standard, Muon, and causal
  manifest regions remain in place and are static-regression tested.
- Telemetry is default ON but score-neutral; the only extra scorer inference is Q3 and remains OFF unless
  the DSL compiles a positive cadence.
- All checkpoint files remain distinct/atomic under the trainer's existing P0 preservation path. This
  change does not weaken or replace any stage checkpoint.

## Recompiled tickets and governed preflight

| Variant | DSL/parser | D-A | D-B | Memory snapshot | Terminal gate |
|---|---:|---:|---:|---|---|
| full | 231/231 flags; 21 active levers | CLEARED | CLEARED | REFUSE, 134.9 GiB > 89.9 GiB | `REFUSE_RC11_NO_SPAWN`; B2 + SSD |
| trimmed-compliant | 219/219 flags; 20 active levers | CLEARED | CLEARED | ADMIT, 87.6 GiB <= 90.1 GiB | `REFUSE_RC11_NO_SPAWN`; B2 + SSD |

Composer receipts report typed validation PASS and exactly two launch blockers for each variant.
The governed launcher completed the preceding zero-dollar gates and returned rc=11 before spawn. The
sandbox's MLX atexit warning occurred after the gate result; it started no trainer or GPU process.

## Verification

- **Focused acceptance:** `175 passed` across the producer, resume-registry, typed-DSL, and both-ticket
  specification suites (`51` of these are the producer + ticket tests themselves).
- **Broader witness-control regression (earlier same landing):** `392 passed, 1 skipped`; one unrelated
  environment failure was `No Metal device available` in the MLX AdamW finite-step smoke.
- **Syntax/parser:** `py_compile` PASS for trainer, producer, resume registry, and DSL; parser flags are
  AST/static-regression checked. The trainer's pre-existing argparse help formatter has an unrelated
  unescaped-percent defect, so `--help` is not used as parse authority here.
- **Lint:** full Ruff PASS for the new producer/test; `E9,F` PASS for every edited Python file, including
  the legacy trainer.
- **Ticket receipts:** JSON parse PASS; compiler typed validation PASS for full and trimmed variants.

No score, archive, contest-CPU/CUDA result, paid dispatch, or frontier-pointer delta was produced.

## Coordination and collision control

The trainer was checked before edit for a live launcher/trainer file descriptor; none was visible.
No `fresh_ab_respawn` registry/checkpoint row was present at that check. The trainer edits are confined
to an additive named producer region and do not rewrite the sibling-owned scorer-surrogate, Metal-conv,
or #336 modules. The shared worktree remains intentionally uncommitted, and no sibling file is staged.

## STORES CONSULTED

`CLAUDE.md` · `AGENTS.md` · `docs/operating_manual_craft_handoff.md` ·
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` ·
`.omx/research/SPEC_v8_perclass_decomposition_20260708.md` ·
`.omx/research/telemetry_enhancement_audit_v7x_v8_20260710.md` (#404) ·
`.omx/research/launch_prego_worklist_20260713.md` · both canonical ticket `GO_BRIEF.md` and
`preflight_summary.json` files · `.omx/state/lane_registry.json` ·
`.omx/state/subagent_progress.jsonl` · `.omx/state/curriculum_candidate_pool.jsonl` ·
`.omx/state/canonical_task_status.jsonl` · the current trainer, resume registry, typed DSL, causal
manifest composer, and standalone SPS probe. No canonical task-status row for #408 was found, so no
task-status transition was invented.

## Triality and pointer-delta honesty

- **DSL:** `TelemetryCadence` for score-neutral default-ON telemetry; `VerdictLiveGap` for Q3's
  score-affecting default-OFF cadence.
- **DAG:** `.omx/research/da_db_producers_DAG_FEED_20260713.md`.
- **Equations:** monotonic component-time accumulation and the preregistered SPS gradient-cosine rule;
  no new empirical equation or anchor is registered before real rows exist.
- **Pointer delta:** exactly zero. The canonical frontier is neither read as a result nor updated.

## Remaining literal blockers

1. `MEMORY_WATERFILL_B2_UNMEASURED_N600`
2. `SSD_WORKLOAD_ROOT_MISSING`

The full variant additionally fails the current memory-governor snapshot. The trimmed variant's memory
snapshot is admitted, but rc=11 still correctly refuses launch on the two literal blockers above.
