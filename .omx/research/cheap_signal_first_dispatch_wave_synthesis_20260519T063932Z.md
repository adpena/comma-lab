# CHEAP-SIGNAL-FIRST-DISPATCH-WAVE — Synthesis 2026-05-19

## Authority

Per Cable C synthesis 2026-05-19 (commit `ef077819e`) "cheap-signal-first DAG" + Cable C6 synthesis 2026-05-19 (commit `4c056724c`) "cheap-signal-first sequencing" + operator-frontier-override 2026-05-19 verbatim "All operator fates and decisions approved".

Total Tier 1 envelope authorized: $3.20. Total Tier 1 envelope SPENT: $0 (operational failures on both Modal dispatches; $0 actually charged because apps stopped without function-call execution).

## Per-dispatch verdict table

| # | Item | Envelope | Spent | Verdict | Mechanism |
|---|---|---|---|---|---|
| 1 | Z6 Wave 2 4c smoke ($3) | $3 | $0 | DEFER (operational) | Modal app `ap-uvrwMMiX2if6fmzp2EeCdk` initialized + objects created BUT no function call executed; app stopped with 0 tasks; no call_id in canonical ledger; no `modal_metadata.json` written. Per Catalog #313 probe outcome registered as `cheap_signal_first_wave_z6_v2_4c_operational_failure_20260519T063202Z` verdict DEFER. |
| 2 | STC v2 dispatch ($0.20) | $0.20 | $0 | DEFER (operational; 3rd consecutive failure) | Modal app `ap-rlIMf5jMhPaF1FbwNVLpZq` initialized + objects created BUT no function call executed; app stopped with 0 tasks. Per Catalog #313 probe outcome registered as `cheap_signal_first_wave_stc_v2_operational_failure_20260519T063156Z` verdict DEFER. 3rd consecutive operational failure (sister `fc-01KRSB76` rc=25 driver crash + sister `fc-01KRSVKF` rc=25 driver crash). |
| 3 | C6.3 PR106 #05+#06 REFORMULATED design memo ($0) | $0 | $0 | PROCEED ($0 design memo landed) | `.omx/research/pr106_05_06_reformulated_design_memo_20260519T063640Z.md` landed (550+ lines, canonical-vs-unique decision per layer + 9-dim checklist evidence + cargo-cult audit + predicted-band Dykstra-feasibility + observability surface). |
| 4 | C6.4 lane_pr101_compressai_balle_full REDIRECT to NSCS03 ($0) | $0 | $0 | PROCEED ($0 redirect memo landed) | `.omx/research/lane_pr101_compressai_balle_full_redirect_to_nscs03_20260519T063640Z.md` landed. Path A NSCS03 ALREADY LANDS (commit 2026-05-15 per NSCS03 landing memo); no additional spend needed. Path B + C are Tier 2 $10 sister REDIRECT pair (out of scope for cheap-signal-first wave). |

## Total spend vs envelope

- **Authorized envelope**: $3.20 (Z6 4c $3 + STC v2 $0.20 + C6.3 $0 + C6.4 $0)
- **Actually spent**: $0 (both Modal dispatches stopped without function-call execution)
- **Tier 1 cheap-signal-first $0 work complete**: 2 of 4 items via $0 design memos
- **Tier 1 paid dispatch items**: 2 of 4 in DEFER (operational) state per Catalog #313

## Critical operational finding: STC v2 + Z6 4c silent-no-spawn pattern

Both dispatches exhibited the SAME silent-no-spawn failure pattern:
1. Modal app initialized (`✓ Initialized. View run at https://modal.com/apps/...`)
2. Mounts created (`✓ Created objects` + `Created mount` lines)
3. Functions created (`Created function run_lane_training_{cpu,t4,a10g,a100,h100}`)
4. Wrapper printed dispatch lane line (`=== modal_train_lane: scripts/remote_lane_substrate_... ===`)
5. Wrapper exited cleanly (exit code 0)
6. **NO call_id printed; NO modal_metadata.json written; NO function execution observed**
7. Modal app state: `stopped` with 0 tasks; 0 versions in history

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE": this is NOT a normal `.spawn()` detach pattern — a normal detach would have produced a call_id in the canonical ledger via `tac.deploy.modal.call_id_ledger.register_dispatched_call_id`. The wrapper appears to have exited BEFORE the spawn registration.

**Hypothesis 1** (most likely): `experiments/modal_train_lane.py` has a fast-exit code path that prints the dispatch banner THEN exits without spawning (possibly under specific paired-env conditions or under the `OPERATOR_AUTHORIZE_SKIP_LOCAL_PRE_DEPLOY_CHECK=1` bypass branch).

**Hypothesis 2**: Modal queue capacity saturation at this hour caused the spawn() call to silently fail with the app immediately stopping (would surface as `stopped` state with 0 tasks).

**Hypothesis 3**: The `tools/run_modal_smoke_before_full.py` wrapper consumed only the smoke phase (per `--smoke-only` flag) but the smoke spawn was queued behind a capacity wait that the wrapper's timeout cut.

## Z6 verdict's impact on Cable B substrate cascade routing

Per Cable C synthesis "Z6 Wave 2 4c $3 cheap signal gates 3 of 5 Cable C substrates (Z7-LSTM, Z8, TT5L V2)":

Since Z6 4c dispatch DID NOT execute, the gating signal for Z7/Z8/TT5L V2 cascade routing is UNRESOLVED. Per CLAUDE.md "Forbidden premature KILL": this is NOT a method falsification; it is an operational dispatch failure that DEFERS the cascade decision, not kills it.

**Cable B cascade routing recommendation**: PAUSE Z7-LSTM/Z8/TT5L V2 dispatches until Z6 4c operational re-investigation lands a successful smoke. Sister Z6 Wave 2 Candidate 1 capacity-unwind branch (per Z6 Phase 3 council Revision #6) is a viable alternative cheap-signal-first path if Z6 4c's `_full_main` proves operationally unrecoverable.

## STC verdict's impact on resurrection candidate priority queue

3rd consecutive STC v2 operational failure (fc-01KRSB76 + fc-01KRSVKF + 2026-05-19 silent-no-spawn) indicates STRUCTURAL operational issue with the STC v2 dispatch surface itself, not the STC paradigm. Per Cable C6.2 design + CLAUDE.md "Forbidden premature KILL":

**Resurrection candidate priority queue update**:
1. **STC-as-sidecar over Selfcomp** (Cable C6.2 Wave 1b, $1-2 envelope) → PROMOTE to next Tier 1 spot (sister formulation that avoids STC v2 standalone dispatch surface)
2. **Filler-Pevny STC kernel as L5/Cable B substrate wrapping** (per resurrection audit Tier 1 #2) → ALTERNATIVE Tier 2 path
3. **STC v2 standalone re-investigation** → DEFERRED to operational-fix wave (NOT Tier 1)

## Operator-routable next-step recommendations

1. **Operational re-investigation** (Tier 1, $0 editor): Investigate `experiments/modal_train_lane.py` for silent-no-spawn code path under paired-env bypass conditions. Add explicit `[modal_train_lane] dispatch_completed call_id=<id>` log line before normal exit so silent failures surface immediately. Sister gate candidate: extend Catalog #245 modal_call_id_ledger registration to fire BEFORE the wrapper exits (currently fires inside spawn() handler which may not execute on early-exit).

2. **Z6 alternate cheap signal** (Tier 1, $3 envelope): Re-fire Z6 Wave 2 Candidate 1 capacity-unwind branch as alternate cheap signal for Cable B Z7/Z8/TT5L V2 cascade gating. Per Z6 Phase 3 council Revision #6, Candidate 1 tests increased FiLM capacity (sister hypothesis to Candidate 4c's side-info-channel branch); same $3 envelope; same cascade-gating value.

3. **STC pivot to sidecar formulation** (Tier 1, $1-2 envelope): Drop STC v2 standalone dispatch surface; promote STC-as-sidecar over Selfcomp per Cable C6.2 Wave 1b. Sister formulation avoids the operational failure surface.

4. **C6.3 Wave 2 dispatch ratification** (Tier 2, $10 envelope per Cable C6 synthesis): Operator-frontier-override on Wave 2 paired CPU+CUDA smoke of reformulated UNIWARD-bit-allocation + grayscale-LUT-latent-bias variants. Per Cable C6.3 DRAFT predicted band [0.180, 0.200] contest-CUDA. CONTINGENT on Wave 1 $0 design memo PROCEED (LANDED THIS WAVE).

5. **C6.4 NSCS03 Phase 2 council λ_R sweep + σ-floor calibration** (Tier 2-3, future): Per NSCS03 landing memo recipe stays `research_only=true` until Phase 2 council ratification. Path B + C $10 sister REDIRECT pair (ATW V2 + NSCS06 v7 smokes) CONTINGENT on Phase 2 council outcome.

## Lane / probe outcome / continual-learning posterior wire-in

- **Lane registered**: `lane_cheap_signal_first_dispatch_wave_20260519` L1 (impl_complete via design memos + memory_entry; real_archive_empirical DEFERRED per operational failures)
- **Probe outcomes** registered to `.omx/state/probe_outcomes.jsonl` (2 rows, both verdict=DEFER per Catalog #313)
- **Modal call_id ledger**: NO new rows (per silent-no-spawn pattern; no call_ids surfaced)
- **Council deliberation posterior** (Catalog #300): synthesis verdict NOT registered as a T2+ council anchor (this is a T1 single-author synthesis; per Catalog #300 "T1 working groups SHOULD emit an anchor when their finding crosses an elevation trigger" — silent-no-spawn pattern qualifies as elevation trigger per Catalog #302 sister-subagent coherence; T2 elevation deferred to operator review)

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map** = N/A (this wave produced design memos + probe outcomes; no sensitivity contribution)
2. **Pareto constraint** = ACTIVE (4 items inform Cable B + C cascade Pareto ranking)
3. **Bit-allocator hook** = N/A (no bit-allocator-relevant signal)
4. **Cathedral autopilot dispatch hook** = ACTIVE (probe outcomes inform autopilot ranker via probe_outcomes_ledger consumer)
5. **Continual-learning posterior update** = ACTIVE (probe outcomes registered to canonical ledger)
6. **Probe-disambiguator** = ACTIVE (DEFER verdict explicitly enumerates 3 alternative paths per recommendation #1, #2, #3 above)

## Cross-references

- Cable C synthesis 2026-05-19 commit `ef077819e`: `.omx/research/cable_c_substrate_symposium_draft_batch_synthesis_20260519T053356Z.md`
- Cable C6 synthesis 2026-05-19 commit `4c056724c`: `.omx/research/cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z.md`
- C6.3 reformulated design memo (THIS WAVE): `.omx/research/pr106_05_06_reformulated_design_memo_20260519T063640Z.md`
- C6.4 NSCS03 redirect memo (THIS WAVE): `.omx/research/lane_pr101_compressai_balle_full_redirect_to_nscs03_20260519T063640Z.md`
- STC v2 probe outcome: `.omx/state/probe_outcomes.jsonl` row `probe_id=cheap_signal_first_wave_stc_v2_operational_failure_20260519T063156Z`
- Z6 4c probe outcome: `.omx/state/probe_outcomes.jsonl` row `probe_id=cheap_signal_first_wave_z6_v2_4c_operational_failure_20260519T063202Z`
- Lane registry: `lane_cheap_signal_first_dispatch_wave_20260519` L1
- CLAUDE.md "Forbidden premature KILL" + "Modal `.spawn()` HARVEST OR LOSE" + "Race-mode rigor inversion" non-negotiables
- Catalog #245 (Modal call_id ledger); Catalog #313 (probe-outcomes ledger); Catalog #300 (council deliberation v2 frontmatter); Catalog #325 (per-substrate symposium evidence)
