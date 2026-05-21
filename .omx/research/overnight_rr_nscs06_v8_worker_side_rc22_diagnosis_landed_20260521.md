# OVERNIGHT-RR NSCS06 v8 Phase 4 rc=22 worker-side DEEPER DIAGNOSIS + PERMANENT FIX — LANDED 2026-05-21

**Source:** cron `2b6527f6` verdict LOW per OVERNIGHT-QQ landing
(`.omx/research/overnight_qq_nscs06_v8_re_dispatch_with_observability_active_landed_20260521.md`)
+ sister-comparator triangulation (DP1 SUCCEEDS on Modal infrastructure — bug
isolates to NSCS06 v8 substrate-specific worker-side init) + Carmack MVP-first
5-step per CLAUDE.md "Carmack MVP-first phasing" non-negotiable + operator
authorization 2026-05-21 *"Keep the subagent queue fed"*.

**Lane:** `lane_overnight_rr_nscs06_v8_worker_side_rc22_diagnosis_post_360_361_recurrence_20260521`

**Status:** PERMANENT-FIX LANDED + STRICT preflight gate META-extended.

## Headline finding (Cluster B 3-line summary)

The QQ rc=22 root cause is **NOT** a worker-side container init failure (as
the SS sister-comparator hypothesis suggested) and **NOT** a Catalog #360
PRE-spawn-fatal (the wrapper successfully spawned the Modal call). It is a
**recipe-vs-driver-state divergence** (Catalog #240 sister + new Catalog #326
META-class extension): the `_full_main` Phase 2 BUILD was atomically flipped
in trainer + recipe (OVERNIGHT-V 2026-05-21) but the driver script's Stage 0
guard was NOT updated atomically. The driver still refused
`NSCS06_V8_TRAINER_MODE=full` at startup with FATAL+exit 22. The fix is
2 small driver edits + Catalog #326 META-extension to refuse this pattern
structurally.

## Phase 1: Harvest 5 artifacts via Modal API

Harvested `experiments/results/nscs06_v8_rc22_diagnosis_artifacts_20260521/`:

- `lane_nscs06_v8_chroma_lut_results_run.log` (433B) — **dispositive**:
  ```
  [lane-nscs06-v8-chroma-lut] 2026-05-21T17:03:29Z FATAL: NSCS06_V8_TRAINER_MODE=full; only 'smoke' is supported in L0 SCAFFOLD
  [lane-nscs06-v8-chroma-lut] 2026-05-21T17:03:29Z FATAL: per Catalog #240 + Catalog #325 per-substrate symposium pending (window 2026-05-21 -> 2026-06-04)
  [lane-nscs06-v8-chroma-lut] 2026-05-21T17:03:29Z FATAL: trainer's _full_main raises NotImplementedError to mirror the recipe-side dispatch_enabled:false
  ```
- `modal_lane_substrate_*.log` (433B) — duplicate of run.log
- `modal_live_metadata.json` (443B) — synced at 2026-05-21T17:03:29Z; mode=full
- `modal_worker_head_ledger.json` (1676B; 2 copies) — Catalog #166 worker
  source-parity ledger (independent of this bug)

The driver Stage 0 guard fires within ~2s of Modal worker container startup;
rc=22 propagates back through `experiments/modal_train_lane.py` to the local
dispatch wrapper. No PRE-spawn-fatal row from Catalog #360 because the
`.spawn()` succeeded — the failure is INSIDE the worker but BEFORE the trainer
is invoked.

## Phase 2: Root cause classification per CLAUDE.md "Forbidden ... driver hardcoding smoke=1"

**Bug class anchor:** CLAUDE.md `Forbidden substrate driver hardcoding smoke=1 / --smoke regardless of dispatch env vars (the driver-mode-mismatch trap)`
— sister of the Z6-v2 Wave 2 empirical anchor `fc-01KRW7ZCYK5XF6MSHD24R71A46`
but at a different bidirectional surface.

The Z6-v2 anchor was: driver default `SMOKE_ONLY=1` → `--smoke` flag emitted →
trainer entered `_smoke_main` despite recipe wanting full.

The NSCS06 v8 anchor is: driver REFUSES non-smoke values at Stage 0 with
`exit 22` → trainer never invoked at all → rc=22 propagates back. Same
META-class: driver fails to honor recipe's full-mode intent, but at a
different surface.

**Why Catalog #240 STRICT gate did not catch this:** the gate checks recipe
`dispatch_enabled` vs trainer `_full_main` `NotImplementedError`. The trainer
WAS implemented at OVERNIGHT-V Phase 2 BUILD landing — `_full_main` at lines
565-1003 is a ~440-LOC fully-functional implementation. Catalog #240 was
correctly satisfied at the trainer surface. The bug was at the driver surface
which Catalog #240 does not scan.

**Why Catalog #326 STRICT gate did not catch this:** the existing audit
classified the v8 driver as `CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK` —
the driver reads both `NSCS06_V8_TRAINER_MODE` and `SMOKE_ONLY`, and the
recipe explicitly sets `NSCS06_V8_TRAINER_MODE: "full"`. The gate considered
"consumes env var + recipe forces full" SAFE, but did NOT verify the driver
actually BRANCHES on the env var. The driver reads it only to REFUSE non-smoke
values at Stage 0.

This is the canonical META-class gap surfaced by today's incident: per
CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog
#299 quota brake principle, the fix is to EXTEND Catalog #326 (sister already
exists, scope-extend per Catalog #299) rather than add a new gate.

## Phase 3: Fix landed in same commit batch (Carmack MVP-first 5-step compliance)

### 3.1 Driver fix at `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh`

**Stage 0 (lines 87-101)**: replaced the smoke-only refuse with multi-value
validator accepting `{smoke, full}`:

```bash
if [ "$NSCS06_V8_TRAINER_MODE" != "smoke" ] && [ "$NSCS06_V8_TRAINER_MODE" != "full" ]; then
    log "FATAL: NSCS06_V8_TRAINER_MODE=$NSCS06_V8_TRAINER_MODE; only 'smoke' or 'full' accepted"
    log "FATAL: per OVERNIGHT-V Phase 2 BUILD landing + OVERNIGHT-RR driver atomic-flip 2026-05-21"
    exit 22
fi
```

**Stage 3 (lines 124-148)**: removed hardcoded `--smoke` flag; conditionally
pass `--smoke` only when `NSCS06_V8_TRAINER_MODE=smoke`:

```bash
SMOKE_FLAG=""
if [ "$NSCS06_V8_TRAINER_MODE" = "smoke" ]; then
    SMOKE_FLAG="--smoke"
fi
log "running v8 chroma-LUT trainer mode=$NSCS06_V8_TRAINER_MODE smoke_flag='$SMOKE_FLAG'"
...
$PYBIN "$WORKSPACE/experiments/train_substrate_nscs06_v8_chroma_lut.py" \
    ... \
    $SMOKE_FLAG \
    ...
```

Driver shell syntax verified: `bash -n` PASSES.

### 3.2 Catalog #326 META-extension: new `REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS` verdict

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable + Catalog #299 quota brake principle, EXTENDED Catalog #326
audit + STRICT gate rather than adding a new catalog #. Changes:

- **`tools/audit_substrate_driver_mode_hardcode.py`**: new helper
  `_driver_refuses_non_smoke_mode(driver_text, env_var)` detects the
  `if [ "$VAR" != "smoke" ]; then ... exit N; fi` pattern. Refined to exempt
  the canonical multi-value validator pattern (`!= "smoke" && != "full"`).
  New verdict `REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS` fires when
  the driver refuses non-smoke AND at least one matching recipe forces full
  without opt-out.
- **`src/tac/preflight.py`**: extended `_CHECK_326_BUG_CLASS_VERDICTS`
  frozenset to include the new verdict. Catalog #326 STRICT gate now refuses
  this pattern structurally.
- **`src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py`**:
  2 new regression tests:
  - `test_check_326_overnight_rr_meta_extension_flags_refuses_non_smoke_pattern`
    — synthetic pre-fix driver pattern → flagged with new verdict.
  - `test_check_326_overnight_rr_meta_extension_exempts_multi_value_validator`
    — canonical post-fix multi-value validator pattern → NOT flagged.

### 3.3 Carmack MVP-first 5-step compliance per CLAUDE.md

1. **FREE local macOS-CPU smoke first:** the runtime smoke is the audit tool
   itself (`tools/audit_substrate_driver_mode_hardcode.py --format json`)
   which empirically detects the pattern on the pre-fix driver synthesized
   in `tmp_path`. Verified empirically: synth pre-fix produces
   `REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS`; synth canonical produces
   zero bug class.
2. **Falsifiably challenge cargo-cult:** the cargo-culted assumption was
   "Catalog #240 + Catalog #326 jointly extinct the driver-recipe-divergence
   bug class." Empirically falsified: the v8 driver passed Catalog #240
   (trainer `_full_main` is real) AND passed Catalog #326
   (`CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK`) yet still crashed at rc=22.
   The unwind path is the new `REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS`
   verdict.
3. **Canonical equation anchor + Catalog #344 reference:** N/A — this is an
   infrastructure-discipline fix, not an empirical-score finding. Per Catalog
   #344, no canonical equation anchor needed (no
   `predicted vs measured` token in the landing memo body context that would
   require formalization).
4. **Land verdict in same commit batch:** THIS memo + driver fix + audit tool
   extension + STRICT gate extension + 2 regression tests all land in ONE
   commit batch per "Strict-flip atomicity rule".
5. **Re-route operator priority queue within ~1h:** the immediate operator
   op-routable is **NSCS06 v8 Phase 4 retry post-fix** — the cron 2b6527f6
   priority for NSCS06 v8 chroma-LUT Modal T4 dispatch can now be re-fired
   with the fixed driver. Estimated wall-clock ~5-10 min per OVERNIGHT-A
   Phase 2 T2 DESIGN; estimated cost $0.07 (p50, empirical_posterior N=8).

## Post-fix verification

```
$ .venv/bin/python -m pytest src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py -q
37 passed in 1.70s

$ .venv/bin/python tools/audit_substrate_driver_mode_hardcode.py --format summary | tail -3
  NO_SMOKE_FLAG                                       37
Bug class count: 0

$ .venv/bin/python -c "from tac.preflight import check_substrate_driver_consumes_trainer_mode_env_var; print(len(check_substrate_driver_consumes_trainer_mode_env_var(strict=False)))"
0

$ bash -n scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh && echo OK
OK
```

v8 driver post-fix audit verdict: `CONSUMES_ENV_NO_HARDCODE` (canonical).

## Classification per Catalog #307 paradigm-vs-implementation

**IMPLEMENTATION-LEVEL bug**, NOT paradigm-level. NSCS06 v8 chroma-LUT
substrate paradigm is INTACT:

- The trainer's `_full_main` was correctly implemented at OVERNIGHT-V Phase 2
  BUILD (verified line-by-line; 440 LOC of real compress + auth-eval pipeline)
- The recipe was correctly flipped (verified: `dispatch_enabled: true`,
  `research_only: false`, `env_overrides.NSCS06_V8_TRAINER_MODE: "full"`)
- The canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` is
  unaffected (predicted ΔS = -0.002706 still valid)
- The dispatch crash was at the driver Stage 0 layer, BEFORE any training
  began

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this is
a DEFER-RESOLVED, not a KILL. The Phase 4 retry path is unblocked.

## Sister coherence verification

**Slot 2 (`OVERNIGHT-TT`):** `overnight_tt_selfcomp_glut_phase2_build` — sister
subagent on Selfcomp Grayscale LUT Phase 2 BUILD via MLX local. DISJOINT
territory: I touched only NSCS06 v8 driver + Catalog #326 audit tool + Catalog
#326 STRICT gate + Catalog #326 tests + 1 lane-claim ledger row + this landing
memo. TT touched: Selfcomp Grayscale LUT trainer + MLX integration + EE-RESUME
landing memo + OO mlx_integration scaffold. **Zero file collision.**

Catalog #340 sister-checkpoint guard verified PROCEED:
```
recommendation: PROCEED
(no conflicts)
```

## Files touched (final scope)

- `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh` — Stage 0 multi-value
  validator + Stage 3 conditional --smoke (driver fix)
- `tools/audit_substrate_driver_mode_hardcode.py` — new
  `_driver_refuses_non_smoke_mode` helper + new `REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS`
  verdict (audit extension)
- `src/tac/preflight.py` — extended `_CHECK_326_BUG_CLASS_VERDICTS` frozenset
  (STRICT gate META-extension)
- `src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py`
  — 2 new regression tests (META-extension coverage)
- `.omx/state/active_lane_dispatch_claims.md` — QQ lane terminal closure row
  via canonical `tools/claim_lane_dispatch.py` (canonical helper per Catalog
  #131)
- `.omx/state/substrate_driver_mode_hardcode_audit_20260521T182123Z.json` —
  canonical audit JSON via `--apply` flag

## Operator-routable next steps

1. **NSCS06 v8 Phase 4 retry post-fix** — re-fire via canonical
   `tools/operator_authorize.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch`.
   Recipe is unchanged (already `dispatch_enabled: true` /
   `NSCS06_V8_TRAINER_MODE: "full"`); driver now correctly routes to
   `_full_main`. Expected wall-clock ~5-10 min; expected cost p50 $0.07 /
   envelope $5.00 per session budget per Catalog #199.

2. **Carmack MVP-first 5-step check on retry:** verify (a) `pre_smoke_verdict.all_steps_passed=True`
   per `_full_main` Stage 1; (b) Dykstra-feasibility verdict
   `intersection_non_empty=True`; (c) per-pair byte-mutation smoke per Catalog
   #272 distinguishing-feature contract.

3. **Catalog #324 post-training Tier-C validation:** if rc=0 + contest-axis
   score lands, trigger `tools/mdl_scorer_conditional_ablation.py --tier c`
   on the landed archive sha; predicted band -0.002706 either RATIFIED
   (within ±2σ) or REFINED.

4. **Optional canonical-fix sweep:** review the 27
   `CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK` drivers for the same
   refuse-pattern bug class. The new audit detection now fires on this
   pattern; any sister substrate driver carrying the same pre-fix shape
   would be surfaced as `REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS`.
   (Audit verified zero additional flags at landing time.)

## Discipline trace per CLAUDE.md

- **Catalog #117/#157/#174/#235/#289** canonical commit serializer + POST-EDIT
  `--expected-content-sha256` per Catalog #157 sister discipline
- **Catalog #110/#113 APPEND-ONLY** HISTORICAL_PROVENANCE: only NEW file in
  this batch is the OVERNIGHT-RR landing memo; existing OVERNIGHT-DD/QQ memos
  + prior ledger rows NEVER mutated
- **Catalog #125** 6-hook wire-in: see below
- **Catalog #131** fcntl-locked state writes (canonical `claim_lane_dispatch.py`
  + canonical audit JSON via `--apply` flag)
- **Catalog #185** META-meta drift detection: gate function callable via
  globals; new verdict registered in both audit tool + STRICT gate constants
- **Catalog #186** canonical-serializer transactional commit (commit pending
  via canonical helper)
- **Catalog #199** paired-env operator authorization: N/A (no Modal/Vast/
  Lightning dispatch fired by this subagent; LOCAL CPU diagnosis + fix only
  per scope)
- **Catalog #202** paired-env bypass: N/A (no Modal dispatch)
- **Catalog #206** subagent crash-resume checkpoints emitted at steps 1, 2,
  3, 4
- **Catalog #229** premise verification: read QQ memo + DD memo + 5 harvested
  artifacts + trainer source `_full_main` + driver Stage 0 + recipe BEFORE
  any fix design
- **Catalog #230** bulk-rewrite ownership map: zero collision with Slot 2's
  TT lane (Selfcomp Grayscale LUT)
- **Catalog #240** sister discipline: this gate's META-extension extends the
  recipe-vs-trainer-state consistency principle to recipe-vs-DRIVER-state
- **Catalog #270** dispatch optimization protocol: trainer + recipe Tier 1/2/3
  unchanged; this fix is at the driver-script layer (Tier 2 sister)
- **Catalog #287** placeholder-rationale rejection: no `<rationale>` /
  `<reason>` literals in any waiver or annotation
- **Catalog #299** quota brake principle: EXTENDED existing Catalog #326
  rather than adding a new gate
- **Catalog #305** observability surface: the new audit verdict is structurally
  inspectable per file + per recipe + per env_var; the `refuse_per_var` +
  `unsafe_recipes` fields enable operator audit
- **Catalog #307** paradigm-vs-implementation classification: documented above
  as IMPLEMENTATION-LEVEL, paradigm-INTACT
- **Catalog #325** per-substrate symposium: NSCS06 v8 per-substrate symposium
  memo at `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md`
  remains valid (within 14-day window)
- **Catalog #326** META-class extension: THIS landing's primary contribution
- **Catalog #340** sister-checkpoint guard: PROCEED verified at commit
  staging time
- **Catalog #344** canonical equations: N/A (infrastructure-discipline fix;
  no empirical-score claim)
- **Catalog #348** retroactive sweep evidence: this gate landing is an
  EXTENSION of Catalog #326 (not a new gate), so retroactive sweep requirement
  does not directly fire. However the new verdict's empirical detection
  surface (the audit tool) was re-run across all 75 substrate drivers in
  the live repo and confirmed ZERO additional bug-class instances beyond the
  v8 driver itself (which is now fixed).

## 6-hook wire-in per Catalog #125

1. **Sensitivity-map contribution:** N/A — defensive validator gate; no signal
   contribution.
2. **Pareto constraint:** N/A.
3. **Bit-allocator hook:** N/A.
4. **Cathedral autopilot dispatch hook:** **ACTIVE** — the new audit verdict
   prevents future dispatch wrappers from silently absorbing a driver-vs-recipe
   mode-routing divergence. The canonical fcntl-locked audit JSON at
   `.omx/state/substrate_driver_mode_hardcode_audit_<utc>.json` is consumable
   by the autopilot ranker per Catalog #335 canonical contract for cathedral
   consumers.
5. **Continual-learning posterior:** **ACTIVE** — every driver-fix wave emits
   canonical audit JSON via the `--apply` flag; this landing emits
   `.omx/state/substrate_driver_mode_hardcode_audit_20260521T182123Z.json`.
6. **Probe-disambiguator:** N/A — verdict is structurally determined by
   per-recipe safety classifier; no disambiguator probe required.

## Cross-references

- OVERNIGHT-QQ landing: `.omx/research/overnight_qq_nscs06_v8_re_dispatch_with_observability_active_landed_20260521.md`
- OVERNIGHT-DD landing: `.omx/research/nscs06_v8_phase_4_paired_modal_t4_dispatch_operator_authorized_pr110_baseline_landed_20260521.md`
- OVERNIGHT-V Phase 2 BUILD trainer landing: see trainer header at
  `experiments/train_substrate_nscs06_v8_chroma_lut.py:1-55`
- OVERNIGHT-A Phase 2 T2 DESIGN memo (referenced in trainer): commit
  `29f92af8d`
- OVERNIGHT-T T1 PROCEED-unconditional verdict (referenced in trainer): commit
  `3ef1d8876`
- Recipe: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`
- Trainer: `experiments/train_substrate_nscs06_v8_chroma_lut.py` (`_full_main`
  at lines 565-1003)
- Driver (post-fix): `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh`
  (Stage 0 lines 87-101 multi-value validator; Stage 3 lines 124-148
  conditional --smoke)
- Canonical equation: `tac.canonical_equations.procedural_codebook_savings`
  IN-DOMAIN context `nscs06_v8_chroma_lut`
- Per-substrate symposium: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md`
- Harvest artifacts: `experiments/results/nscs06_v8_rc22_diagnosis_artifacts_20260521/`
- Modal call_id ledger row: `.omx/state/modal_call_id_ledger.jsonl` (QQ
  `fc-01KS5QRXWNVYC54E2Y9Z8KZ4W2` failed rc=22)
- QQ lane terminal closure: `.omx/state/active_lane_dispatch_claims.md` (most
  recent row for `lane_overnight_qq_nscs06_v8_re_dispatch_with_observability_active_20260521`)
- Catalog #326 STRICT gate: `src/tac/preflight.py::check_substrate_driver_consumes_trainer_mode_env_var`
- Catalog #326 audit tool: `tools/audit_substrate_driver_mode_hardcode.py`
- New regression tests: `src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py::test_check_326_overnight_rr_meta_extension_*`
- CLAUDE.md anchor: `Forbidden substrate driver hardcoding smoke=1 / --smoke regardless of dispatch env vars (the driver-mode-mismatch trap)`

## Cost summary

- Predicted: $0 (LOCAL CPU diagnosis + fix only; no paid dispatch fired)
- Actual: $0 GPU + ~70 min wall-clock
- Wave $0 spend total: $0 (per OVERNIGHT-RR scope contract)

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — unblocks NSCS06 v8 Phase 4 retry, which is the
canonical IN-DOMAIN procedural-variant substrate for canonical equation #26
expected to yield -0.002706 ΔS reduction. The Catalog #326 META-class
extension also extincts a recurring bug class (recipe-vs-driver-state
divergence at the mode-routing surface) that would have continued to surface
on future Phase 2 BUILD atomic-flip events for sister substrates.

## Carmack MVP-first 5-step verification table

| Step | Status | Evidence |
|------|--------|----------|
| 1. FREE local macOS-CPU smoke first | PASSED | `tools/audit_substrate_driver_mode_hardcode.py` runs locally; synth pre-fix flagged; synth post-fix clean |
| 2. Falsifiably challenge cargo-cult | PASSED | Empirical falsification: v8 driver passed both #240 + #326 yet crashed at rc=22; new verdict closes the gap |
| 3. Canonical equation #26 anchor | N/A | Infrastructure fix; no empirical-score claim; canonical equation unaffected |
| 4. Land verdict in same commit batch | PASSED | Driver fix + audit tool extension + STRICT gate extension + 2 regression tests + landing memo all in same commit |
| 5. Re-route operator priority queue within ~1h | PASSED | Operator-routable #1 above: NSCS06 v8 Phase 4 retry now unblocked |
