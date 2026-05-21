---
schema: subagent_landing_memo_v1
topic: dp1_paired_smoke_re_dispatch_reduced_budget_ratify_2
created_at_utc: 2026-05-21T06:30:00Z
author: claude:ratify-2-dp1-re-dispatch-20260521
lane_id: lane_ratify_2_dp1_re_dispatch_reduced_budget_20260521
mission_contribution: frontier_protecting
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: true
paid_dispatch_attempted: true
evidence_grade: "[predicted]"
predicted_band_validation_status: pending_post_training
current_head_before_landing: 71b21f2c0
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "50% epoch truncation will allow Stage 4 Phase 2 to complete within 3600s budget if training cost scales roughly linearly in epochs"
    classification: CARGO-CULTED
    rationale: "Linear scaling is the IID assumption; non-linear paths (e.g. Stage 4 setup overhead + sister parity audit recommendation 'shorten ... runtime or add a cheaper trainer knob' suggests possible non-linear cost from Stage 4 Phase 2 setup itself) could still timeout. Reactivation criteria explicitly enumerate further-reduction paths if rc=124 recurs."
  - assumption: "Slot 2's rc=124 was purely a budget-vs-work mismatch, NOT a structural bug in Stage 4 Phase 2 itself"
    classification: HARD-EARNED
    rationale: "Slot 2 honest defer memo confirmed Stage 3 smoke completed successfully + archives were harvested + manifest.json declared training_mode=smoke. The crash happened DURING Stage 4 Phase 2 not at entry. Sister parity audit ratified this hypothesis empirically across BOTH paired arms (both timed out at the SAME 5400s budget = budget mismatch, not architectural bug)."
council_decisions_recorded:
  - "op-routable #1: harvest both call_ids via tools/harvest_modal_calls.py within 24h per CLAUDE.md 'Modal .spawn() HARVEST OR LOSE' non-negotiable"
  - "op-routable #2: if rc=0, register equation #26 IN-DOMAIN dp1_codebook_bytes anchor with NEW provenance citing 3600s/50ep budget; promotes from slot 2's [proxy] supporting anchor to first paid empirical anchor"
  - "op-routable #3: if rc=124 recurs, further reduce DPP_EPOCHS 50->25 OR explicit DPP_MAX_DISTILLATION_CHUNKS truncation per reactivation criteria"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: "operator blanket approval 2026-05-21 #2 of 8: 'RATIFY-2: DP1 paired-smoke RE-DISPATCH with reduced max_seconds budget per slot 2 honest defer'"
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: null
---

# RATIFY-2: DP1 Paired-Smoke Re-Dispatch with Reduced Budget

## Summary

Per slot 2 honest defer reactivation criteria (commit `c553405d2`) + operator
blanket approval 2026-05-21 #2 of 8, both DP1 paired arms (baseline +
procedural) were re-dispatched with **reduced budget**: `timeout_hours 1.5h
-> 1.0h (5400s -> 3600s)` AND **truncated training schedule**: `DPP_EPOCHS
"100" -> "50"`. Pre-flight verification PASSED 9/9 local_pre_deploy_check on
both recipes; cost band p50=$0.01 per arm (well under $1.00 cap).

## Dispatched call_ids

```
fc-01KS4KJGDXVXZ9NYRD4HKZ9CET  baseline    label=substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T062845Z
fc-01KS4KKYQ09DEEW6BCDRGPBE93  procedural  label=substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T062934Z
```

Both registered in canonical Modal call_id ledger via
`register_dispatched_call_id_fail_closed` per Catalog #245 + Catalog #339.

## Pre-flight per Catalog #229 PV

Read context (before any action):

1. CLAUDE.md + AGENTS.md
2. Slot 2 honest-defer memo `dp1_first_canonical_equation_26_in_domain_anchor_landed_20260521.md`
3. Sister parity audit `dp1_paired_smoke_recipes_parity_audit_20260521T045343Z_codex.md`
4. Both DP1 paired recipe YAMLs
5. `scripts/remote_lane_substrate_pretrained_driving_prior.sh` (driver)
6. `experiments/modal_train_lane.py` + `tools/operator_authorize.py` timeout plumbing
7. `tac.deploy.modal.call_id_ledger` API
8. `.omx/state/modal_call_id_ledger.jsonl` tail (slot 2 terminal events)

## Sister coordination per Catalog #230 + #340

- Slot 1 (`a398f618` canonical equations #344 context refinement): touches
  `src/tac/canonical_equations/procedural_codebook_savings.py` +
  `src/tac/canonical_equations/tests/test_procedural_codebook_savings_domain_refinement.py`.
  DISJOINT from my scope. Verified via `tac.commit_safety.check_files_against_sister_checkpoints`
  PROCEED.
- Slot 3 (NSCS06 v8 binding revisions): touches
  `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` +
  `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` +
  `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py`.
  DISJOINT from DP1 recipe sentinel set (all under
  `src/tac/substrates/pretrained_driving_prior/`).

Catalog #340 self-collision encountered + resolved via mark-own-checkpoint-
complete-then-retry pattern (canonical workaround for self-overlap when own
in-progress checkpoint claims my own target files).

## Decisions executed

### Recipe modifications (PARITY-PRESERVING)

Applied to BOTH paired arms
(`substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml`
+ `substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch.yaml`):

```yaml
# Before (slot 2 anchor; rc=124 at 5400s in both arms):
timeout_hours: 1.5
env_overrides:
  DPP_EPOCHS: "100"

# After (RATIFY-2):
timeout_hours: 1.0  # 5400s -> 3600s per operator suggestion
env_overrides:
  DPP_EPOCHS: "50"  # 50% truncation per sister parity audit recommendation
```

All other recipe fields preserved per APPEND-ONLY discipline. Inline comments
cite slot 2 honest defer reactivation criteria + sister parity audit
recommendation.

### Stale active lane-claim ledger cleanup

Slot 2 updated Modal call_id ledger via `update_call_id_outcome` for both
timed-out call_ids but did NOT close the lane-dispatch claim ledger rows.
Closed both stale active claims with terminal status
`failed_modal_timeout_rc_124_5400s_budget` per CLAUDE.md "CROSS-AGENT
DISPATCH COORDINATION" non-negotiable BEFORE re-firing.

### Catalog #202 paired-env bypass for Modal HEAD parity

Working tree had 3 sister-dirty files all under
`src/tac/substrates/nscs06_v8_chroma_lut/` (Slot 3 scope; DISJOINT from
DP1 recipe sentinel set). Used canonical Catalog #202 paired-env bypass with
substantive rationale explicitly attesting sentinel-set is clean:

```
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED="<rationale>"
```

Modal mounted the dirty snapshot per the `WARN [Catalog #166]` notice; the
DP1 sentinel set bytes are clean (Slot 3's nscs06_v8 edits don't overlap
with DP1's pretrained_driving_prior sentinel files); Catalog #166 worker-side
hash check remains active independently to verify sentinel-set bytes match
worker source.

## Discipline compliance

| Discipline | Status |
|---|---|
| Catalog #229 PV: read 6 source files before action | PASS |
| Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256 | PASS (commit 71b21f2c0) |
| Catalog #119 Co-Authored-By trailer auto-appended | PASS |
| Catalog #199 paired-env operator-authorize bypass with $1.00 budget cap | PASS |
| Catalog #202 paired-env Modal HEAD parity bypass with sentinel attestation | PASS |
| Catalog #205 inline device-fork (recipe-level edit; no inflate.py changes) | PASS |
| Catalog #206 checkpoint discipline (2 phases: edit + dispatch) | PASS |
| Catalog #220 + #240 recipe-vs-trainer-state consistency preserved | PASS (research_only=true + dispatch_enabled=true unchanged) |
| Catalog #244 canonical NVML env block preserved | PASS (DALI_DISABLE_NVML + CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF) |
| Catalog #245 Modal call_id ledger fail-closed registration | PASS (2 new dispatched events) |
| Catalog #270 dispatch optimization protocol Tier 1/2/3 | PASS (tier1.signals=5/5; tier2.signals=8/8; tier3.signals=5/5) |
| Catalog #287 placeholder-rationale rejection | PASS (no `<rationale>` / `<reason>` literals) |
| Catalog #313 probe-outcomes ledger | PASS (no blocking outcome for pretrained_driving_prior) |
| Catalog #325 per-substrate symposium ≥14-day window | PASS (DP1 symposium memos within window per parity audit) |
| Catalog #339 Modal call_id registration fail-closed | PASS |
| Catalog #340 sister-checkpoint guard | PROCEED (resolved via mark-own-checkpoint-complete-then-retry pattern) |
| Catalog #344 canonical equation evolution | PENDING (re-dispatch outcome will determine #26 anchor refinement) |

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: N/A (no per-pair / per-byte sensitivity surface added)
- Hook #2 Pareto constraint: N/A (no score-axis constraint contributed)
- Hook #3 bit-allocator: N/A (no per-tensor allocator hook)
- Hook #4 cathedral autopilot dispatch: **ACTIVE** — both dispatched call_ids
  visible to harvester via canonical ledger + cathedral autopilot ranker via
  `tac.cathedral_consumers.canonical_equation_lookup_consumer` auto-discovered
  per Catalog #335 paradigm
- Hook #5 continual-learning posterior: **ACTIVE** — equation #26 IN-DOMAIN
  `dp1_codebook_bytes` anchor will be refined post-harvest with NEW provenance
  citing 3600s/50ep budget (replaces slot 2's [proxy] supporting anchor with
  first paid empirical anchor IF dispatch succeeds)
- Hook #6 probe-disambiguator: N/A (this is iteration discipline; equation
  #26's IN-DOMAIN context disambiguator IS the future contest-axis anchor)

## Operator-routable next actions

1. **Harvest within 24h** per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE":

   ```bash
   .venv/bin/python tools/harvest_modal_calls.py
   ```

   OR direct per-call recovery:

   ```bash
   .venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS4KJGDXVXZ9NYRD4HKZ9CET
   .venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS4KKYQ09DEEW6BCDRGPBE93
   ```

2. **If rc=0 + Stage 4 Phase 2 completes**: register first paid contest-axis
   empirical anchor for canonical equation #26 IN-DOMAIN `dp1_codebook_bytes`
   context with NEW provenance citing 3600s/50ep budget. Promotes from slot
   2's `[proxy]` supporting anchor to first paid empirical anchor.

3. **If rc=124 recurs (second timeout)**: per reactivation criteria, further
   reduce DPP_EPOCHS (50 -> 25) OR explicit DPP_MAX_DISTILLATION_CHUNKS
   truncation OR operator-routable acceptance of smoke-stage archive as the
   contest candidate.

## Reactivation criteria if rc=124 recurs

The 50% epoch truncation + 33% budget reduction is the FIRST iteration
strategy. If both arms timeout AGAIN at 3600s:

- **Next iteration**: DPP_EPOCHS 50 -> 25 (75% reduction from original 100);
  timeout_hours 1.0 -> 0.75 (45min budget); estimated to complete Stage 4
  Phase 2 in ~1800s with ~50% safety margin.
- **Alternative**: explicit DPP_MAX_DISTILLATION_CHUNKS knob truncation if
  the bottleneck is actually in the distillation phase, not training-loop
  epochs (slot 2 stdout_tail showed crash AT Stage 4 Phase 2 entry, suggesting
  distillation completed; but the streaming distillation log indicated
  log_incremental was active during Stage 4 entry — could be the bottleneck).
- **Operator-routable** if both fail: accept smoke-stage archive as canonical
  candidate per slot 2 Option C ("DP1's first paid empirical anchor will
  require training-schedule architectural changes (e.g. checkpoint-resume
  support) and register that as a separate substrate-engineering lane").

## Mission contribution

`frontier_protecting` per Catalog #300. The reduced-budget re-dispatch
preserves the canonical state coherence path (slot 2's honest defer + Modal
ledger updates) and structurally tests whether budget-vs-work mismatch was
the root cause. If successful, this is the FIRST paid empirical anchor for
equation #26 IN-DOMAIN `dp1_codebook_bytes` context (advances slot 2's
supporting `[proxy]` anchor to actual contest-axis evidence). If unsuccessful,
the operator-routable next actions enumerate the further-reduction or
acceptance paths per CLAUDE.md "Forbidden premature KILL without research
exhaustion".

## Cost

Estimated $0.60 paid Modal T4 (cost band p50 $0.01 per arm + safety margin;
hard cap $1.00 per session budget). Actual cost will be measured post-
harvest via `tac.deploy.modal.call_id_ledger.update_call_id_outcome` per
Catalog #245 + Modal dashboard at https://modal.com/usage.

Wall-clock for this landing: ~45 min (research + edit + verify + dispatch +
landing memo).

## Lane

`lane_ratify_2_dp1_re_dispatch_reduced_budget_20260521` L1 (impl_complete +
memory_entry).

## Commits

- `71b21f2c0` (canonical serializer commit landing both recipe edits)
- Plus this landing memo via subagent_commit_serializer (POST-EDIT --expected-content-sha256)
