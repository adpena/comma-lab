# p0_328 + p0_408 — merge-window prep (2026-07-17)

Branch `claude/p0_328_408_merge_window_prep_20260717` (isolated worktree; **merges at the
post-v9c2 boundary, AFTER the #518 branch — NOT by this agent**). Pointer contest-CPU
**0.19108 UNMOVED** — everything here is MEANS (boundary apparatus for the c2 lineage), no
score claim. The live v9c2 run is untouched (all edits stay on this branch; main's frozen
trainer keeps crash-resuming).

## STORES CONSULTED (recall-before-design)

- `tools/graph_memory_recall.py "clip_profile phase 2 consumers"` → FEED-06z (`tac.clip_profile`
  BUILT, #322/#323) + FEED-clipprofile2 (#328 Phase-2 plan: Phase-1 rewired 1 safe advisory
  consumer, inventoried ~12 byte-coupled/score-critical consumers to Phase-2b, **2 gold
  discrepancy findings** — `_V_HORIZON` 174 swept-optimal #327 vs profile median 175; `_CAM_H`
  1.2 lane-IPM vs profile 1.22 — "do NOT silently switch").
- `tools/graph_memory_recall.py "Q1-Q7 telemetry resume boundary"` → FEED-telemetry-p0 (#404
  audit, Q1–Q7 queued designs) + FEED-da-db-producers (#480 producers built) + FEED-515build +
  FEED-ratetelemetry (rate non-monotonic → rolling-avg soft-signal, NEVER a kill) + FEED-p0-recovery.
- P0 ledger rows `p0_328_clip_profile_rewire` + `p0_408_telemetry_resume_boundary`
  (`tools/operator_p0_digest.py --all`). Harness #408 Q-batch enumeration (DAG L14334).
- Source read at cited lines: `tac.clip_profile` (fields), `tools/measure_pose_warp_dseg.py`
  (Phase-1 canonical rewire pattern), `tools/levelset_byte_close_and_eval.py` (byte-close
  constants), `src/tac/boundary_math/lane_sdf_component.py` (scorer intrinsics), the trainer
  (verdict_live_gap mechanism, lever_engage emitters, resume registry, `_we_rate_term_mlx`),
  `src/tac/witness_control/{rate_rolling_telemetry,telemetry_producers,resume_registry}.py`,
  `src/tac/witness_dsl/constants_telemetry_build_wave_20260715.py` (queued RateRollingTelemetry).

## SCOPE A — p0_328 Phase-2 clip_profile consumer rewire (commit `0bfd135a41`)

**Finding on entry:** Phase-1 already rewired the safe advisory consumer (`measure_pose_warp`);
Phase-2b (byte-coupled trainer byte-path + score byte-close) was queued. This unit lands it under
the measured-no-regression contract: route the AGREEING per-clip constants through the canonical
MEASURED `tac.clip_profile` SoT (literal fallback when cache absent, byte-identical on 0.mkv);
LEAVE the two discrepancy findings hardcoded.

- **`tools/levelset_byte_close_and_eval.py` (score byte-close path): DONE.** `CAMERA_H/W` (874/1164)
  ← `native_h/w`; `RATE_DENOM` (37_545_489) ← `video_bytes`; `_XI_FX/_XI_CX/_XI_CY` (910/582/437)
  ← `fx/cx/cy_native`; `_XI_D` (1.22) ← `device_height_m`. All AGREE bit-exactly (verified against
  the real cache). module-level try/for_video/except-literal (matches Phase-1).
- **`src/tac/boundary_math/lane_sdf_component.py` (trainer byte path, imported at
  `train_levelset…:5169`): DONE.** scorer intrinsics `_FX/_FY/_CX` (400.3/399.5/256) ← profile
  `fx/fy/cx_scorer`, byte-identical. `_CAM_H` (1.2) + `_V_HORIZON` (174) **DELIBERATELY LEFT
  HARDCODED** — they DISAGREE with the profile (1.22 / 175); switching them changes IPM geometry →
  byte-close output (the two FEED-clipprofile2 routed-to-reconciliation findings). Asserted
  not-silently-switched in tests.
- **launcher `tools/launch_witness_run.py`: N/A** — grep found ZERO hardcoded per-clip constants
  (it composes argv, doesn't hardcode camera/rate/horizon).
- **`src/tac/raft_pose.py:84` (`fx=910.0` API default): NOT rewired (deliberate)** — it is a general
  library public-API DEFAULT with existing tests, not a pipeline per-clip constant; coupling a
  library default to a 0.mkv-specific profile lookup is out of the measured-no-regression envelope
  (FEED-clipprofile2 flagged it "changing risks byte/API change"). Left for a dedicated unit.
- Phase-3 (the 60 `measure_*.py` fold) explicitly OUT OF SCOPE per the task.
- **No-regression proof:** all rewired constants == the historical literals AND == the profile
  fields (cache present); byte-close/trainer output byte-identical on 0.mkv. Tests
  `src/tac/tests/test_clip_profile_rewire_byte_close.py` (7): 3 literal-bit-identity + 3
  profile-tracking (skip in the isolated worktree — no 37MB video/cache; verified pass against the
  real cache) + 1 discrepancy-not-switched. `_XI_FX`-on-plain-import is a PRE-EXISTING partial-import
  artifact (main identical; the assignment only runs as `__main__`, where it reads the verified
  `_CP_XI_FX==910`).

## SCOPE B — p0_408 / #404 Q-batch telemetry at the resume boundary (commit `828e772235`)

**Finding on entry (reconcile, don't duplicate):** Q1–Q7 emission sites are ALREADY WIRED on main
HEAD — ClipActivationAggregator (Q1, `:11557`), term_inert_rows (Q2, `:12857`), tail_cycle (Q4,
`:12520`), would_fire (Q5, `:10769`), ladder_birth_complete (Q6, `:11628`), lever_engage_row (Q7,
~20 sites). `--verdict-live-gap-every` (Q3) EXISTS + is DSL-held (`VerdictLiveGap()` →
`verdict_live_gap`, single emitter — verified; no duplication). The genuine remaining gaps:

- **Rate rolling-avg soft-signal (FEED-ratetelemetry): DONE.** Producer was BUILT but the DSL lever
  `RateRollingTelemetry()` was FAIL-CLOSED (`TrainerWireInQueued`) pending the trainer flag. Landed
  the trainer `--rate-rolling-telemetry` (BooleanOptionalAction, default TRUE per off-is-orphan) +
  a synchronous verdict-cadence emission block (after the dm1 telemetry block) that maintains a
  `weight_entropy_bits` proxy series (`_we_rate_term_mlx(model, σ)` — pure read of the COUNTED
  weights, no grad/mutation → score-neutral) and emits `rate_rolling_row(ep, series, baseline=…)`
  (WITHIN→DRIFTING_UP→SUSTAINED_GROWTH; `informs_only=True`, NEVER kills). Resume-safe via an
  ADDITIVE `_resume_registry.register("rate_rolling_telemetry", "__raterolling_", …)` (persists the
  proxy tail + t0 baseline; legacy sidecars restore False → empty series → honest re-anchor).
  `RateRollingTelemetry()` now auto-unlocks; TRAINER_WIREIN_QUEUE[-1] status → landed; its test
  updated (queued→landed).
- **lever_engage uniform schema: DONE.** `lever_engage_row` gained an additive `extra=` param
  (canonical stage/lever/status/epoch/via authoritative — reserved-key collision RAISES; extras
  additive). The one hand-rolled `{"stage":"lever_engage",…additive_margin diagnostics}` literal
  (trainer `:6246`) now routes through `lever_engage_row(status="armed", extra={…})`, folding the
  prior 2-row (diagnostic + redundant plain armed) emission into ONE canonical armed row carrying
  the diagnostics. Every lever_engage row now shares the stable base schema.
- **Q1/Q2/Q4/Q5/Q6 + verdict_live_gap: VERIFIED present + correct (no code needed).**
- **byte-identity of training numerics:** REQUIRED and preserved BY CONSTRUCTION — every surface is
  read-only (weights/config) + prints JSON; no loss term, gradient, optimizer/EMA/model mutation.
  The rate-proxy is a LIVE-weight drift proxy (not the shipped EMA-shadow archive rate) — fine for an
  informs-only soft-signal; noted. Tests
  `src/tac/witness_control/tests/test_p0_408_rate_rolling_and_lever_engage_wirein.py` (10) +
  updated `test_constants_telemetry_build_wave_20260715.py`.

## Own round-1 adversarial review (findings + dispositions)

- **F1 (Scope A):** `lane_sdf_component` now calls `for_video` at MODULE IMPORT → one sha256 of the
  37 MB video per process (cache-hit is fast; cache-miss raises quickly → literal fallback). Matches
  the Phase-1 precedent; module import is per-process-cached. → ACCEPTED, documented.
- **F2 (Scope A):** `_XI_FX`-absent-on-plain-import → confirmed PRE-EXISTING (unedited main
  identical; partial import when `experiments/` off path). NOT introduced. → NO FIX.
- **F3 (Scope B):** rate proxy uses LIVE model weights, not the EMA shadow byte-close scores. Fine
  for a monotone drift soft-signal; not a shipped-rate claim. → ACCEPTED, noted in the emission
  comment.
- **F4 (Scope B):** the emission adds one cheap MLX weight-histogram eval per verdict in the main
  thread (guarded by try/except → never blocks). Score-neutral → default-ON justified. → ACCEPTED.
- **F5 (Scope B):** lever_engage fold reduces 2 rows → 1 (additive fields preserved). Field-based
  readers unaffected; only a row-COUNT reader would differ. → ACCEPTED (deliberate consolidation).
- **F6 (Scope B):** `_rate_rolling_restore` zip truncates on a corrupt sidecar length mismatch
  (advisory telemetry, non-blocking). → ACCEPTED.
- **F7:** verified the full trainer argparse builds with `--rate-rolling-telemetry` (no flag clash)
  via `real_trainer_flags(None)`.
- **F8:** the `__raterolling_` keys grow the RESUME sidecar (not the scored archive) when the flag is
  on; additive + legacy-compatible (flag-off → empty write → byte-identical sidecar). Archive bytes
  unchanged. → ACCEPTED (correct resume-registry behavior).

## MERGE INSTRUCTION (for the merging agent — NOT this agent)

1. **Order:** this branch merges into `main` **AFTER** `claude/p0_518_resume_warmup_geometry_20260717`,
   at the post-v9c2 boundary (once the live run has landed / been governed-stopped).
2. **Conflict prediction (MEASURED via `git merge-tree`):** `git merge-tree --write-tree --name-only
   claude/p0_328_408… claude/p0_518…` returns **rc=0, ZERO conflicting files** — the only shared file
   (the trainer `experiments/train_levelset_witness_realized_through_R_mlx.py`) 3-way merges CLEAN.
   #518's base (`97c1d7c4`) is behind main HEAD (`02f81666`, this branch's base); #518 touches the
   trainer at the resume block (~9848–10180), update site (~12720–13010), verdict (~9290/9445), and
   the `main` argparse block, plus `curriculum_dsl.py`/`evaluators.py`/`launch_witness_run.py`/new
   files — **NONE of which this branch edits** (my trainer hunks: lever_engage `:6246`, resume
   registry `:8375`, epoch-loop telemetry `:13307`, argparse flag `:14806`; my other files:
   byte-close, lane_sdf, telemetry_producers, constants_telemetry + tests).
3. **Proximity zones to eyeball on merge** (git auto-resolved them, but drift-if-main-advances):
   (a) the `main` argparse block — both add `add_argument` flags (union both sets: keep #518's
   fork/margin/pose/ema flags AND `--rate-rolling-telemetry`); (b) the epoch loop — #518 at the
   update/verdict site, mine at the dm1/seg-focal telemetry anchor (distinct sub-regions).
4. **Post-merge sanity:** `pytest src/tac/witness_control/tests/test_p0_408_… src/tac/tests/
   test_clip_profile_rewire_byte_close.py src/tac/witness_dsl/tests/
   test_constants_telemetry_build_wave_20260715.py`; `real_trainer_flags(None)` contains both flag
   sets; `RateRollingTelemetry()` + #518's factories all compile.

## 6-hook wire-in (Subagent coherence-by-default)

Sensitivity-map / Pareto / bit-allocator / cathedral-autopilot: **N/A** — read-only score-neutral
observability + a measured-no-regression plumbing rewire; adds no score-affecting bytes/levers.
Continual-learning: the rate-rolling drift signal feeds the costate SENSE layer (informs-only).
Probe-disambiguator: **N/A** (no competing interpretations). DSL leg: `RateRollingTelemetry()`
auto-unlocked (flag now real). Equations leg: `# NO_EQUATION_NEEDED` (read-only telemetry; no new
measured law). DAG leg: this memo.
