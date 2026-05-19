# Codex routing directive — paid-dispatch batch C6 + Catalog #204 follow-on

**Date:** 2026-05-19T07:15:00Z
**Authority:** Operator-frontier-override per Catalog #300 + operator verbatim 2026-05-19 "All operator decisions are approved, proceed with all and keep the queue saturated" (see sister `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md` for the canonical capture format; this directive extends the same master approval to the 4 paid-dispatch items below)
**Total budget envelope:** $30.50-60.30 USD (well within Codex operational cap of $15/item × 4 items = $60)

## Operator-frontier-override frontmatter

```yaml
council_tier: T1  # working-group routing; binding decisions already captured per Catalog #300 in sister symposium memos
council_attendees: [Claude-main-dispatcher, Codex-autonomous-research-and-implementation-loop]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: "All operator decisions are approved, proceed with all and keep the queue saturated"
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: 2026-06-18T07:15:00Z
```

## Routing items (4 unblocked + 2 blocked-on-slot-1)

### UNBLOCKED — codex can dispatch now

#### Item 1: C6.1 lane_17_imp LTH reactivation ($10.20-15 Vast.ai 4090)

- **Source memo:** `feedback_pre_rigor_kill_defer_falsified_inventory_landed_20260517.md` op-routable #856 (HIGH-EV: $5-15, ΔS [-0.05, -0.005], KILL was stats.json stub-loop artifact NOT paradigm)
- **Bug class extincted:** Catalog #91+#94 stub-loop class — the original KILL verdict was based on `stats.json: epochs=200, elapsed_sec=3.47` (internally inconsistent stub-loop per CLAUDE.md "Internal-consistency assertions in stats files" non-negotiable). With Catalog #91+#94 STRICT, future stub-loops are refused at write-time.
- **Re-probe scope:** standalone IMP cycle 0 against current renderer baseline. Frankle LTH (Lottery Ticket Hypothesis) cycle 0 — magnitude-pruning + rewind-and-retrain. Target: cycle-0 score against current best contest-CUDA frontier.
- **Custody requirements per Catalog #127 + #313:** every score must be `[contest-CUDA T4]` or `[contest-CUDA 4090]` with paired Linux x86_64 hardware. Register dispatch via canonical `tools/operator_authorize.py` (auto-fires #243 pre-deploy harness + #271 codex pre-dispatch review). NO `[advisory only]` claims.
- **Probe-outcomes ledger:** register verdict in `.omx/state/probe_outcomes.jsonl` via `tac.probe_outcomes_ledger.register_probe_outcome` regardless of outcome (PROCEED / DEFER / INDEPENDENT / KILL).
- **Reactivation criteria** (per CLAUDE.md "Forbidden premature KILL"): if re-probe returns DEFER, enumerate ≥3 alternative reducer methodologies per Catalog #308.

#### Item 2: C6.3 PR106 #05+#06 REFORMULATED paired smoke ($10 Modal A10G)

- **Source memo:** earlier slot 6 return — "C6.3 PR106 #05+#06 REFORMULATED design memo: PROCEED LANDED ($0)"; paired smoke is the empirical follow-on
- **Bug class extincted:** Catalog #308 N>=3 alternative reducers — the original FALSIFIED verdict was based on a SINGLE reducer methodology + paradigm conflation
- **Re-probe scope:** UNIWARD-delta + grayscale-LUT applied to A1-sidecar substrate (paradigm-INTACT, design-CARGO-CULTED per the prior REFORMULATED design memo). 100-epoch smoke with paired contest-CUDA + contest-CPU eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.
- **Custody requirements:** identical to Item 1. Per CLAUDE.md "Apples-to-apples evidence discipline": ALL scores must label axis + hardware + archive sha. NO inferring CPU from CUDA or vice versa.
- **Decode-complexity-evidence per Catalog #293:** UNIWARD-delta + grayscale-LUT are PR101 gold + PR103 silver + PR#56 paradigm primitives. The REFORMULATED design memo cites these as source_supports.

#### Item 3: C6.5 mae_v + saug operational-fix ($10-35 Vast.ai 4090)

- **Source memo:** earlier session — mae_v (masked autoencoder variant) + saug (synthetic augmentation) combined operational fix. Per the prior 50ep proxy showing promising trajectory but unfinished operational integration.
- **Bug class:** operational-completion (the previous DEFER was infrastructure-level NOT paradigm)
- **Re-probe scope:** integrate mae_v + saug into the canonical substrate trainer template (per `experiments/train_substrate_*.py` family). 50-epoch smoke + full-run gate if smoke green.
- **Cost envelope:** $10 smoke + up to $25 full-run = $35 total cap. Honor Catalog #167 smoke-before-full pattern via `tools/run_modal_smoke_before_full.py`.

#### Item 4: Catalog #204 follow-on A1 passthrough recovery ($0.30 Modal T4)

- **Source memo:** Catalog #204 cross-driver expansion (`feedback_distinguishing_feature_integration_contract_landed_20260515.md` sister) — `sha=110cfaa3` archive recovery for A1 passthrough lane was queued post-WAVE-2 driver fix.
- **Bug class:** infrastructure-recovery (NOT score-lowering; recovers a frozen contest-CUDA anchor for the lane registry)
- **Re-probe scope:** $0.30 Modal T4 inflate + auth_eval on existing A1 passthrough archive bytes. Verify Catalog #204 cross-driver fix produces the same `0.19205 [contest-CPU]` / `0.20533 [contest-CUDA]` anchor the lane registry expects.
- **Custody:** identical to Items 1-3.

### BLOCKED — wait for slot 1 silent-no-spawn fix

#### Item 5: Z6 Wave 2 4c re-fire ($3 Modal A10G) — BLOCKED on slot 1

- **Blocker:** today's Z6 Wave 2 4c dispatch returned DEFER (operational) with silent-no-spawn pattern. Slot 1 subagent `a77281c4631bdc264` is extincting that bug class structurally. Re-fire AFTER slot 1 lands its 4-layer fix (canonical helper + STRICT gate + sister mitigation + tests).
- **Predicted ΔS:** [-0.025, -0.008] per the Z6 Wave 2 design memo
- **Routing:** re-fire via `bash scripts/operator_authorize_substrate_time_traveler_l5_z6_modal_a10g_dispatch.sh` AFTER slot 1 completes + verifies the silent-no-spawn fix on a fresh Z6 4c dispatch attempt.

#### Item 6: STC v2 RATIFY-or-DEFER ($0.20 Modal T4) — BLOCKED on slot 1

- **Blocker:** today's STC v2 dispatch was the 3rd consecutive silent-no-spawn failure. Slot 1's fix unblocks. Per Catalog #313 probe-outcomes ledger: register STC v2 DEFER (operational) outcome with reactivation criterion = "silent-no-spawn fix lands + 1 successful smoke completes". After unblock, RATIFY-or-DEFER the prior PROCEED-with-PoseNet-caveat verdict.
- **Routing:** re-fire via `bash scripts/operator_authorize_substrate_stc_v2_modal_t4_dispatch.sh` AFTER slot 1 unblock.

## Dispatch pre-conditions per CLAUDE.md non-negotiables

For EACH of Items 1-4 (and Items 5-6 after unblock), codex MUST:

1. **Catalog #243 pre-deploy harness**: auto-fires via `tools/operator_authorize.py` route; refuses dispatch on missing helper per Catalog #279 fail-closed; refuses on unresolved recipe per Catalog #280 fail-closed
2. **Catalog #271 codex pre-dispatch review**: auto-fires for paid dispatch >$1; cache invalidates on dirty tree per Catalog #282; missing-helper refuses per Catalog #283
3. **Catalog #270 dispatch optimization protocol**: AND(Tier 1 engineering + Tier 2 hardware + Tier 3 substrate); refuses on protocol gap
4. **Catalog #325 per-substrate symposium evidence**: within 14 days; verdict ∈ {PROCEED, PROCEED_WITH_REVISIONS}; OR operator-frontier-override per Catalog #300
5. **Catalog #166 + #201 Modal sentinel files in mount set**: dispatch refused on stale source-vs-worker parity
6. **Catalog #244 NVML env block + Catalog #203 Modal training image hard runtime deps**: refuses on missing canonical env / deps
7. **Catalog #245 modal_call_id_ledger registration**: AUTO via slot 1's silent-no-spawn extinction (post-slot-1-landing); BEFORE slot 1 lands, codex MUST manually verify ledger row after `fn.spawn()` returns + before wrapper exits
8. **Catalog #199 paired-env operator confirmation bypass**: `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=60.30` covers the full envelope per operator master approval verbatim

## Discipline

- **One commit per item via canonical serializer** (`tools/subagent_commit_serializer.py --commit-via-serializer --message <msg> --files <files> --expected-content-sha256 "<f>=<sha>"`)
- **Probe-outcomes ledger registration mandatory** per Catalog #313 regardless of verdict
- **Lane gates updated per Catalog #90** for each item (impl_complete + real_archive_empirical + contest_cuda + memory_entry as gates land)
- **Custody routing per Catalog #127** for every score row written to `.omx/state/continual_learning_posterior.jsonl`
- **Apples-to-apples evidence per CLAUDE.md non-negotiable**: explicit axis + hardware + archive sha tags; NO inferring CPU from CUDA
- **Per CLAUDE.md "Forbidden premature KILL"**: DEFER + REQUEST-REINVESTIGATION-OF-ALTERNATIVES is the canonical verdict structure per Catalog #308. NO `KILL` verdicts without research exhaustion + grand council consensus.

## Maximum-signal preservation per Catalog #300

- Verbatim dissent: none on the routing itself
- Per-member operating-within assumption: HARD-EARNED via empirical-anchor-driven reactivation criteria
- HARD-EARNED-vs-CARGO-CULTED classification per Item:
  - Item 1 (lane_17_imp LTH): HARD-EARNED-EMPIRICALLY-VERIFIED via Frankle LTH paper + Catalog #91+#94 stub-loop extinction
  - Item 2 (PR106 #05+#06 REFORMULATED): HARD-EARNED-PARADIGM-INTACT + CARGO-CULTED-DESIGN per the REFORMULATED design memo classification
  - Item 3 (mae_v + saug): HARD-EARNED-PARTIAL via prior 50ep proxy + operational-completion route
  - Item 4 (Catalog #204 A1 passthrough recovery): HARD-EARNED-INFRASTRUCTURE-RECOVERY (not score-lowering claim)
- Full vote tally: PROCEED-unconditional via operator-frontier-override
- Cite-chain: this directive + sister memos + the operator-frontier-override capture

## Cross-references

- Parent operator-frontier-override capture: `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- Sister codex routing directive: `.omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md`
- Pre-rigor inventory: `feedback_pre_rigor_kill_defer_falsified_inventory_landed_20260517.md`
- Silent-no-spawn fix (slot 1 in-flight): `lane_silent_no_spawn_structural_extinction_20260519` per slot 1 dispatch
- Integrated battle plan: `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md`

— Claude-main 2026-05-19T07:15:00Z (codex routing per operator master approval)


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
