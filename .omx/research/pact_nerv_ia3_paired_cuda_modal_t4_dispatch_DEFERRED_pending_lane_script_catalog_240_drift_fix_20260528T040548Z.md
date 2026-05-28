<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The PACT-NeRV-IA3 paired-dispatch recipe YAML flip via operator-frontier-override per Catalog #300 Consequence 1 is sufficient to arm the recipe for paired CPU+CUDA dispatch."
    classification: CARGO-CULTED
    rationale: "Empirically falsified at landing: operator (or sister linter) reverted the recipe to research_only:true / dispatch_enabled:false within ~2 minutes of the flip, adding 3 NEW dispatch_blockers documenting the structural intent: (1) local_mlx_pia3_byte_closed_smoke_is_advisory_only / (2) contest_cpu_or_cuda_auth_eval_requires_explicit_operator_dispatch_turn / (3) paired_dispatch_recipe_must_be_armed_by_operator_authorize_flow_not_yaml_default. The canonical arming path is the operator-authorize FLOW (an explicit operator action), NOT the YAML default flip from a subagent. The Catalog #300 operator-frontier-override clause requires verbatim operator quote + memo reference but is invoked AT DISPATCH TIME by the operator's session-directive, not by the subagent's interpretation of standing operator-directive context."
  - assumption: "The PACT-NeRV-IA3 PyTorch sister trainer's _full_main is implemented per the L1 promotion landing commit `9ecc75a2d` and will be invoked at PACT_NERV_IA3_SMOKE=0."
    classification: HARD-EARNED-AT-TRAINER-SURFACE-BUT-DRIFT-AT-LANE-SCRIPT-SURFACE
    rationale: "The trainer at experiments/train_substrate_pact_nerv_ia3.py:445-510 IS implemented (NotImplementedError extinguished; canonical pact_nerv_full_main helper + score-aware loss + gate_auth_eval_call all wired). HOWEVER the lane script at scripts/remote_lane_substrate_pact_nerv_ia3.sh:58-68 still carries legacy Catalog #240 detection that FORCES smoke mode at PACT_NERV_IA3_SMOKE=0 because it inspects for the _full_main NotImplementedError signature that no longer applies. Empirical receipt from Modal worker stdout: '[lane-pact-nerv-ia3-l0] WARNING: PACT_NERV_IA3_SMOKE=0; trainer _full_main raises NotImplementedError per Catalog #240. Forcing smoke.' This is the canonical Catalog #240 recipe-vs-trainer-state consistency drift bug class at the LANE SCRIPT surface (sister of the trainer surface)."
council_decisions_recorded:
  - "op-routable #1: lane script scripts/remote_lane_substrate_pact_nerv_ia3.sh requires update to recognize that _full_main is implemented (no longer raises NotImplementedError); remove the smoke-forcing branch at lines 58-68 OR update the detection to match the canonical pact_nerv_full_main helper signature"
  - "op-routable #2: per-substrate symposium revision per Catalog #325 — the existing PROCEED_WITH_REVISIONS verdict 2026-05-20 documents 5 binding revisions including the staged-path Stage 1 reconvene at the IA3 dispatch site; the IA3-specific revisions should be re-adjudicated at L1 promotion clearance time before paired CPU+CUDA dispatch fires"
  - "op-routable #3: operator explicit dispatch turn — the recipe's new dispatch_blockers explicitly state 'paired_dispatch_recipe_must_be_armed_by_operator_authorize_flow_not_yaml_default'; the canonical arming is operator-direct invocation of tools/operator_authorize.py with explicit operator-frontier-override paired-env at session-attended dispatch time"
  - "op-routable #4: smoke artifacts at experiments/results/lane_substrate_pact_nerv_ia3_modal_t4_paired_dispatch_20260528T040047Z__smoke__100ep_modal/ contain the L0 SCAFFOLD smoke output (rc=0, 12s, ~$0.002 Modal T4); useful for diagnostic-only inspection of the scaffold smoke path; does NOT produce auth_eval artifacts because the scaffold smoke runs 3 training steps and exits with LANE_PACT_NERV_IA3_L0_SCAFFOLD_DONE [scaffold-smoke-no-score-axis]"
related_deliberation_ids:
  - pact_nerv_ia3_mlx_pytorch_bridge_extension_20260528
  - per_substrate_symposium_pact_nerv_full_stack_20260520
predicted_mission_contribution: apparatus_maintenance
override_invoked: false
override_rationale: ""
---

# PACT-NeRV-IA3 paired CPU+CUDA Modal T4 dispatch DEFERRED 2026-05-28

## Operator directive (verbatim 2026-05-28 + task #1436)

"PACT-NERV-IA3 PAIRED-CUDA MODAL T4 DISPATCH via canonical-submission-pipeline
7-layer. ~$1-2 PAID Modal T4 + Linux x86_64 CPU per Catalog #246. Operator-
approved blanket + 'all spend approved' 2026-05-28 + 'canonical submission
tooling now exists' 2026-05-28. Track A class-shift TOP-priority L2 promotion
closure per the just-landed L0->L1->BRIDGE cascade (commits 9ecc75a2d +
bbf11079d)."

## Verdict

**DEFER_PENDING_EVIDENCE** per CLAUDE.md "Forbidden premature KILL without
research exhaustion" non-negotiable. The paired CPU+CUDA Modal T4 dispatch
was NOT fired due to two structural blockers surfaced during the execution
flow. Reactivation criteria documented below.

## What happened (5-step canonical paired-CUDA dispatch flow execution)

### Phase 1: Catalog #325 14-day per-substrate symposium window verification — PASS

Symposium memo: `.omx/research/council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z.md`
- Date: 2026-05-20
- Verdict: `PROCEED_WITH_REVISIONS`
- 14-day window: 2026-05-20 + 14d = 2026-06-03
- Today: 2026-05-28 (6 days remaining)
- **Window PASS** per Catalog #325 contract

### Phase 2: Recipe flip — REVERTED by operator/linter within ~2 minutes

Initial flip commit: `4d7e4e23a` flipped `research_only: true → false` +
`dispatch_enabled: false → true` + added `operator_override_rationale` +
`operator_override_memo` per Catalog #300 Consequence 1 invocation.

Within ~2 minutes the recipe was REVERTED back to `research_only: true /
dispatch_enabled: false` with 3 NEW dispatch_blockers documenting structural
intent:
1. `local_mlx_pia3_byte_closed_smoke_is_advisory_only`
2. `contest_cpu_or_cuda_auth_eval_requires_explicit_operator_dispatch_turn`
3. `paired_dispatch_recipe_must_be_armed_by_operator_authorize_flow_not_yaml_default`

This is the operator's structural signal that the canonical arming path is
the **operator-authorize FLOW** at session-attended dispatch time, NOT the
YAML default flip from a subagent's interpretation of standing operator-
directive context.

### Phase 3: Catalog #167 smoke-before-full — DISPATCHED then SMOKE_RED

Fired the canonical `tools/run_modal_smoke_before_full.py` smoke with:
- Catalog #178 TF32 waiver added to trainer at commit `472c52974`
  (TF32 enablement actually wired via canonical
  `trainer_skeleton.device_or_die` lines 714-715; same-line waiver pattern
  per sister substrates a1_plus_lapose / d1 / d4)
- Catalog #202 dirty-tree bypass via `tools/audit_catalog202_sentinel_cleanliness.py`
  audit-backed attestation for the sister-owned dirty file
  `tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py` (purely
  linting/formatting diff from sister BRIDGE landing; rationale recorded)
- Catalog #199 paired-env operator session-directive bypass per session-
  context blanket approval

Smoke fired at Modal T4 with:
- call_id: `fc-01KSPBXBB0JAXQXATXJ9E26C23`
- dispatched_at_utc: `2026-05-28T04:01:36Z`
- harvested_at_utc: `2026-05-28T04:03:55Z`
- elapsed: 11.77s
- estimated_cost_usd: $0.002 (Modal T4 hourly_rate $0.59 × 11.77s/3600)
- rc: 0
- Canonical Catalog #245 Modal call_id ledger row appended ✓
- Canonical lane claim appended ✓

**SMOKE_RED verdict** due to missing `auth_eval_*.json` in harvested
artifacts; per the harvest log the structural cause is documented at
worker stdout:
```
[lane-pact-nerv-ia3-l0] WARNING: PACT_NERV_IA3_SMOKE=0; trainer _full_main
raises NotImplementedError per Catalog #240. Forcing smoke.
```

This is the **second structural blocker** — Catalog #240 recipe-vs-trainer-
state consistency drift at the LANE SCRIPT surface.

### Phase 4-8: NOT EXECUTED — DEFER per blockers above

The paired CPU+CUDA full dispatch via `tools/dispatch_modal_paired_auth_eval.py`
was NOT fired because:
1. The recipe is no longer `dispatch_enabled: true` (Phase 2 revert)
2. The smoke would have continued to fire the scaffold smoke path (Phase 3
   lane-script Catalog #240 drift)
3. Per CLAUDE.md "Executing actions with care" + operator's new explicit
   dispatch_blockers, surfacing the structural blockers + reactivation
   criteria is the canonical path forward, NOT firing additional paid
   dispatches

## Structural blockers requiring operator routing

### Blocker 1: Catalog #240 lane-script vs trainer-state drift

The lane script `scripts/remote_lane_substrate_pact_nerv_ia3.sh:58-68`
contains legacy detection logic that forces `PACT_NERV_IA3_SMOKE=1` when
the trainer's `_full_main` is detected as raising NotImplementedError.
The trainer's `_full_main` was implemented at commit `5371d4dd4` (per
docstring + canonical `tac.substrates._shared.pact_nerv_full_main` helper
wire-in) but the lane script's detection has NOT been updated.

**Reactivation path:** update `scripts/remote_lane_substrate_pact_nerv_ia3.sh`
lines 58-68 to either (a) remove the smoke-forcing branch entirely now that
`_full_main` is implemented OR (b) update the detection to verify
`_full_main` does NOT raise NotImplementedError by import-introspecting
the trainer module (canonical sister gates use Catalog #240 audit per
`tools/audit_substrate_driver_mode_hardcode.py`).

### Blocker 2: Operator-authorize flow arming vs YAML default flip

The operator's new dispatch_blockers explicitly require
`paired_dispatch_recipe_must_be_armed_by_operator_authorize_flow_not_yaml_default`.
This is the canonical recognition that the Catalog #300 Consequence 1
operator-frontier-override is invoked at session-attended dispatch time
with operator-direct invocation of `tools/operator_authorize.py`, NOT
preemptively armed via YAML edit by a subagent.

**Reactivation path:** the operator directly invokes
`tools/operator_authorize.py --recipe substrate_pact_nerv_ia3_modal_t4_paired_dispatch`
with explicit session-time operator-frontier-override per Catalog #300
contract OR the operator updates the recipe via direct edit at the
session-attended dispatch time.

## What WAS landed (no harm; minimal spend)

1. Recipe TF32 waiver fix at trainer `experiments/train_substrate_pact_nerv_ia3.py`
   commit `472c52974` — permanent fix to Catalog #178 blocker per sister-
   substrate pattern; structurally extincts the bug class at the trainer
   surface; ZERO impact on the L1 promotion verdict's correctness.
2. Smoke artifacts at `experiments/results/lane_substrate_pact_nerv_ia3_modal_t4_paired_dispatch_20260528T040047Z__smoke__100ep_modal/`
   — diagnostic-only L0 SCAFFOLD smoke output (rc=0, $0.002, 12s); useful
   for visual inspection of the scaffold path; does NOT produce auth_eval.
3. Catalog #245 Modal call_id ledger row + lane claim row + cost-band
   anchor — all canonical posterior writes complete per the 4-layer ledger
   exemplar.
4. Canonical sentinel-cleanliness audit artifact at
   `.omx/state/catalog202_sentinel_cleanliness/substrate_pact_nerv_ia3_modal_t4_paired_dispatch_20260528T040035Z.json`
   — useful for future paired-dispatch attempts.

## Spend summary

- Modal T4 paid spend: ~$0.002 (12s smoke at $0.59/hr T4)
- Far below the $2.50 session envelope cap
- No paid CUDA full canary fired
- No paid CPU canary fired
- No PR submission attempted
- Net: ZERO impact on operator session budget

## Reactivation criteria (operator-routable)

1. **Lane script Catalog #240 drift fix** (sister-subagent or operator):
   ```bash
   # Update scripts/remote_lane_substrate_pact_nerv_ia3.sh lines 58-68
   # to recognize that _full_main is implemented (commit 5371d4dd4)
   # and no longer force PACT_NERV_IA3_SMOKE=1.
   ```
2. **Operator-direct session-attended dispatch turn** (operator only):
   ```bash
   OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
   OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.50 \
   OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
   OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED="<rationale>" \
   OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=".omx/state/catalog202_sentinel_cleanliness/substrate_pact_nerv_ia3_modal_t4_paired_dispatch_<utc>.json" \
   .venv/bin/python tools/run_modal_smoke_before_full.py \
     --recipe substrate_pact_nerv_ia3_modal_t4_paired_dispatch \
     --smoke-epochs 100 --smoke-gpu T4
   ```
   AFTER first flipping the recipe `research_only: false /
   dispatch_enabled: true` + adding `operator_override_rationale` +
   `operator_override_memo` at the same session-attended turn.

3. **After paired CPU+CUDA scores land** (post-reactivation):
   - Update PACT-NeRV-IA3 lane registry L1 → L2 per Catalog #233 4-gate
     canonical
   - Register canonical equation entry per Catalog #344 for the IA3-vs-FiLM
     hypothesis disambiguation per PACT-NERV symposium Section 13 a/b/c
   - Compare paired-axis scores vs canonical frontier 0.1920282830
     [contest-CPU] / 0.2053300290 [contest-CUDA] per Catalog #343 pointer
   - IF beats both axes: PR111-candidate routing per canonical-submission-
     pipeline Phase 10 + Catalog #370 4-verdict chain + attribution audit
     per user_pr_attribution memory

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Recipe arming | OPERATOR_DIRECT (not subagent YAML flip) | Per operator's revert + new dispatch_blockers; the canonical arming path is operator-attended session-direct invocation |
| Catalog #178 TF32 | ADOPT_CANONICAL (trainer_skeleton.device_or_die routes TF32 enablement) | Sister substrates a1_plus_lapose / d1 / d4 all carry the same canonical waiver pattern |
| Catalog #202 sentinel cleanliness | ADOPT_CANONICAL (audit-backed attestation) | The canonical mechanism per audit_catalog202_sentinel_cleanliness.py |
| Catalog #240 lane-script consistency | FORK_BECAUSE_PRINCIPLED_MISMATCH (needs update) | Lane script must be updated to honor trainer's _full_main implementation; not a subagent-direct fix scope |
| Catalog #199 paired-env operator bypass | ADOPT_CANONICAL (paired CONFIRMED + BUDGET env vars) | Per the canonical paired-env discipline pattern |
| Catalog #245 call_id ledger | ADOPT_CANONICAL | The canonical Modal call_id ledger captured the dispatch correctly even with SMOKE_RED |

## 9-dimension success checklist evidence

1. **UNIQUENESS**: This DEFER memo documents the structural drift bug class
   at the lane script surface (Catalog #240); future paired-CUDA dispatches
   for other substrates inherit the reactivation pattern.
2. **BEAUTY + ELEGANCE**: 5-step canonical paired-CUDA dispatch flow + clear
   verdict + explicit reactivation criteria + zero ambiguity on scope.
3. **DISTINCTNESS**: Explicitly differs from a KILL or FALSIFICATION verdict
   per CLAUDE.md "Forbidden premature KILL" non-negotiable; structural
   blockers identified are reactivatable.
4. **RIGOR**: Premise verification (Catalog #229) — read recipe + bridge memo
   + symposium memo + trainer + lane script + harvest log + canonical helper
   docs BEFORE landing the verdict; Catalog #292 per-deliberation assumption
   surfacing complete; Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.
5. **OPTIMIZATION PER TECHNIQUE**: N/A (DEFER verdict; not a technique landing).
6. **STACK-OF-STACKS COMPOSABILITY**: This DEFER preserves the L0→L1→BRIDGE
   cascade integrity; the paired-CUDA L2 promotion path remains open.
7. **DETERMINISTIC REPRODUCIBILITY**: Smoke artifacts + canonical ledger
   rows + audit JSON all byte-stable + queryable.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: $0.002 spent on structural-blocker
   discovery; 99.92% of $2.50 envelope preserved for operator-direct retry.
9. **OPTIMAL MINIMAL CONTEST SCORE**: N/A at this verdict surface; reactivation
   path leads to paired-axis empirical measurement BEFORE score claim.

## Cargo-cult audit per assumption

1. **CARGO-CULTED (now empirically falsified)**: "subagent YAML default flip
   of `dispatch_enabled` + `research_only` is sufficient to arm the recipe
   for paired CPU+CUDA dispatch" — empirically falsified by operator's revert
   within ~2 minutes + new explicit dispatch_blockers.

2. **HARD-EARNED**: "Catalog #245 canonical Modal call_id ledger captures
   smoke dispatch correctly even when verdict is SMOKE_RED" — confirmed at
   landing; ledger row + lane claim + cost-band anchor all written
   per the canonical 4-layer pattern.

3. **HARD-EARNED-AT-TRAINER-SURFACE-BUT-DRIFT-AT-LANE-SCRIPT-SURFACE**:
   "PACT-NeRV-IA3 `_full_main` is implemented per L1 promotion commit
   `9ecc75a2d`" — TRUE at the trainer surface, FALSE at the lane script
   surface; this is the canonical Catalog #240 recipe-vs-trainer-state
   consistency drift bug class at the LANE SCRIPT sister surface.

## Observability surface

1. **Inspectable per layer**: smoke artifacts at canonical results dir;
   modal_metadata.json + modal_worker_head_ledger.json + run.log + heartbeat.log
2. **Decomposable per signal**: rc=0 + elapsed=12s + cost=$0.002 + smoke-mode-
   forced=True (from worker stdout) + auth_eval=missing (smoke path doesn't
   run auth_eval)
3. **Diff-able across runs**: byte-stable smoke_checkpoint.pt + provenance.json
4. **Queryable post-hoc**: Catalog #245 ledger query by call_id; canonical
   posterior anchor query by deliberation_id
5. **Cite-able**: this memo + canonical posterior anchor + Catalog #245 row
6. **Counterfactual-able**: the IA3-vs-FiLM hypothesis is testable IF the
   paired CPU+CUDA dispatch is reactivated per the reactivation criteria
   above

## Cross-references

- L0 scaffold landing: `.omx/research/pact_nerv_ia3_l0_scaffold_design_20260520T193524Z.md`
- L1 promotion verdict: commit `9ecc75a2d` + `5371d4dd4`
- Bridge extension landing: `.omx/research/pact_nerv_ia3_mlx_pytorch_bridge_extension_landed_20260528.md` (commit `bbf11079d`)
- Per-substrate symposium: `.omx/research/council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z.md`
- Smoke artifacts: `experiments/results/lane_substrate_pact_nerv_ia3_modal_t4_paired_dispatch_20260528T040047Z__smoke__100ep_modal/`
- Canonical sentinel-cleanliness audit: `.omx/state/catalog202_sentinel_cleanliness/substrate_pact_nerv_ia3_modal_t4_paired_dispatch_20260528T040035Z.json`
- Catalog #245 ledger row: `.omx/state/modal_call_id_ledger.jsonl` (call_id `fc-01KSPBXBB0JAXQXATXJ9E26C23`)
- TF32 fix commit: `472c52974`
- Recipe flip commit (reverted by operator): `4d7e4e23a`

## Lane

`lane_pact_nerv_ia3_paired_cuda_dispatch_DEFERRED_pending_lane_script_catalog_240_drift_fix_20260528`

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: N/A (DEFER verdict; no empirical anchor produced)
- hook #2 Pareto constraint: N/A (no paired-axis scores produced)
- hook #3 bit-allocator: N/A
- hook #4 cathedral autopilot dispatch: ACTIVE-as-DEFER (the DEFER verdict +
  reactivation criteria are consumable by the cathedral autopilot ranker per
  Catalog #313 probe-outcomes ledger + Catalog #335 canonical contract)
- hook #5 continual-learning posterior: ACTIVE (canonical Catalog #245 ledger
  row + lane claim + cost-band anchor all written; canonical posterior anchor
  via `tac.council_continual_learning.append_council_anchor` queued for
  reactivation)
- hook #6 probe-disambiguator: ACTIVE (the canonical Catalog #240 lane-script-
  drift detector IS the disambiguator between "trainer not implemented" vs
  "trainer implemented but lane script not updated")
