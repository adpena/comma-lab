# B1 E.7 + E.8 Modal Dispatch — Catalog #313 BLOCKED + Operator-Routable Surface

**Lane**: `lane_b1_e7_e8_modal_dispatch_catalog_313_blocked_op_routable_20260519`
**Subagent**: `claude_slot_cc_b1_e7_e8_modal_dispatch_20260519`
**Date**: 2026-05-20T01:30Z (UTC)
**GPU spend**: $0 (zero dispatch fired; Catalog #313 BLOCKED at predecessor-probe stage)
**Wall-clock**: ~30 min (Phase 1 PV + Phase 2 #313 check + Phase 5 landing memo)
**Per CLAUDE.md**: "Forbidden premature KILL without research exhaustion" + "Cross-agent dispatch coordination" + Catalog #313 (probe-outcomes ledger predecessor check)

## TL;DR

**Sub-state verdict per task brief Phase 1**: **(e) operator-routable harvest of existing in-flight dispatch** —
specifically, **operator-routable surfacing of Catalog #313 BLOCKING DEFER state**, with
all 4 prior E.7+E.8 dispatches harvested 2026-05-19T06:04Z + the canonical
infrastructure remediation plan already captured in the probe outcomes ledger.

**No new paid dispatch fired**. Per Catalog #313 STRICT preflight gate + the
operator-authorize `_check_predecessor_probe_outcome` runtime wire-in, every
dispatch attempt at this slot would be REFUSED with `rc=1` per the same path
codex sister `019de465` was refused at 2026-05-19T21:13:13Z
(memo: `.omx/research/codex_findings_catalog204_item4_dispatch_refused_by_probe_outcome_20260519T211313Z_codex.md`).

The honest answer to the council T3 rank-2 op-routable: **the empirical answers
to VQ K-sweep + SGLD DEFER cannot land via this slot's dispatch path until the
2 infrastructure-tier blockers are resolved by sister subagent / operator-routed
work.**

## Phase 1 PV verdict (Catalog #229 premise verification)

The task brief asked which sub-state applies. PV evidence:

| Substrate | Recipe path | Trainer path | Driver | Prior dispatch history |
|---|---|---|---|---|
| E.7 VQ K-sweep | `substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml` | `experiments/train_substrate_vq_vae.py` | `scripts/remote_lane_substrate_vq_vae.sh` | 2 dispatches 2026-05-19; both FAILED rc=1 |
| E.8 SGLD convergence | `substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml` | `experiments/train_substrate_stack_of_stacks.py` | `scripts/remote_lane_substrate_stack_of_stacks.sh` | 3 dispatches 2026-05-19; all FAILED rc=1 |

**Sub-state**: (e) operator-routable harvest (predecessors already harvested) +
Catalog #313 BLOCKED surface.

## Phase 2 Catalog #313 predecessor probe outcome verdicts

### E.7 vq_vae substrate (canonical authority `tools/check_predecessor_probe_outcome.py --substrate vq_vae --json`):

```json
{
  "probe_id": "harvest_e7_vq_k_sweep_1_t4_oom_20260519",
  "blocker_status": "blocking",
  "verdict": "DEFER",
  "metric_name": "cuda_oom_at_segnet_batchnorm_fwd_bytes_needed",
  "metric_value": 27262976,
  "threshold": 15665380000,
  "expires_at_utc": "2026-06-02T06:10:00Z",
  "next_action": "Re-dispatch on A100-80GB OR enable autocast_fp16+reduce batch_size 16->4 OR gradient_checkpointing per Catalog #218",
  "evidence_path": null
}
```

**Recipe drift confirmed**: recipe at HEAD still declares `gpu: "${MODAL_GPU:-T4}"` + `min_vram_gb: 14` + `min_smoke_gpu: "T4"`. Per the Catalog #313 `next_action`, the recipe MUST be updated to A100-80GB OR the trainer MUST add the engineering primitives (autocast_fp16 + batch_size reduction + gradient_checkpointing) per CLAUDE.md Catalog #270 dispatch optimization protocol + Catalog #172 (autocast) + Catalog #218 (mini-batch reconstruct).

The operator-frontier-override `e7_e8_symposium_operator_frontier_override_20260519T051028Z.md` was written BEFORE the empirical T4 OOM was discovered (override 2026-05-19T05:10Z; OOM dispatch 05:42Z; harvest verdict 06:04Z). The override does not retroactively cover infrastructure failures discovered AFTER its issuance.

A second probe row `vq_vae_k_sweep_dispatch_a10g_retry_ratified_20260519` (PROCEED, advisory) was written by sister subagent `cable_b1_e7_e8_dispatch` between attempts authorizing A10G retry, but the recipe was never updated to consume the A10G ratification, AND the substrate-level latest blocking row remains DEFER.

### E.8 stack_of_stacks substrate (canonical authority `tools/check_predecessor_probe_outcome.py --substrate stack_of_stacks --json`):

```json
{
  "probe_id": "harvest_e8_sgld_1_instant_crash_20260519",
  "blocker_status": "blocking",
  "verdict": "DEFER",
  "metric_name": "elapsed_seconds_before_crash",
  "metric_value": 2.11,
  "expires_at_utc": "2026-06-02T06:10:00Z",
  "next_action": "Re-dispatch with --capture-output to surface root cause; reproduce on Modal CPU container locally"
}
```

Plus 2 sister DEFER rows:
- `sgld_convergence_dispatch_trainer_only_single_arm_passthrough_not_real_sgld_DEFER_20260519` (BLOCKING; 30-day window) — empirical finding that the recipe's entrypoint enters single-arm A1 passthrough NOT the SGLD polish loop; `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` is INERT in passthrough mode
- `harvest_e8_sgld_2_auth_eval_temp_workdir_20260519` (BLOCKING; 14-day window) — Catalog #204 `/tmp/pact` work_dir refusal

Sister `catalog_204_driver_fix_stack_of_stacks_landed_20260519` (PARTIAL, advisory) addressed the Catalog #204 portion, but the SGLD trainer-scope blocker (`sgld_convergence_dispatch_trainer_only_single_arm_passthrough_not_real_sgld_DEFER_20260519`) was NOT superseded. Sister codex `019de465` 2026-05-19T21:13Z attempted dispatch via `operator_authorize.py` and was correctly refused per Catalog #313.

## Phase 3+4 result: NO dispatch fired; NO harvest of new artifacts needed

Per task brief constraints:

> "DO NOT fire dispatch without Catalog #313 predecessor check."
> "IF verdict in `{INDEPENDENT, KILL, DEFER}` blocking: STOP and surface as op-routable."

Both substrates have BLOCKING DEFER verdicts. Both DEFERs were authored by sister claude/codex subagents within the last 24h. The codex sister has already attempted the same path with explicit operator-frontier-override env vars set and was correctly refused per Catalog #313.

**Per CLAUDE.md "Cross-agent dispatch coordination" + Catalog #199 paired-env discipline + the codex sister's explicit advice in their landing memo** (*"Do not set `OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1` from this Codex loop"* — same advice transitively applies to this slot), **I refuse to bypass Catalog #313 via paired-env override**.

The operator-frontier-override memo issued earlier 2026-05-19 covered the symposium-ratification surface (Catalog #325), not the predecessor-probe-outcome surface (Catalog #313). Per Catalog #313 design, infrastructure-discovered DEFERs require remediation BEFORE the gate accepts re-dispatch — they cannot be transitively waived by an earlier symposium override.

Prior 4 dispatched call_ids are already harvested as of 2026-05-19T06:04:00Z by `codex:harvest_modal_calls`:

| call_id | substrate | rc | elapsed | outcome | terminal-claim status |
|---|---|---|---|---|---|
| `fc-01KRZC53Y0D28B6BYEQ1MRG347` | vq_vae (T4) | 1 | 141s | RC_1 SegNet OOM | `failed_modal_training_rc_1` |
| `fc-01KRZCX15GAF5Z5E3E568Q60FF` | vq_vae (A10G) | unknown | unknown | not in harvest log | active or stale |
| `fc-01KRZCHVY6C1TSFNNS6KN13G70` | stack_of_stacks | 1 | 2.1s | RC_1 instant crash | `failed_modal_training_rc_1` |
| `fc-01KRZCSQ7FPVMSAXZQDSZJCTN4` | stack_of_stacks | 1 | 19.5s | RC_1 passthrough+temp_workdir | `failed_modal_training_rc_1` |

The A10G VQ retry call_id `fc-01KRZCX15GAF5Z5E3E568Q60FF` is missing from the harvest set — needs follow-up via `tools/harvest_modal_calls.py --recover-from-tmp` per Catalog #339 sister recovery path.

## Total paid GPU spend vs $4.20 cap

**$0** this slot (no new dispatch fired).

**Prior 2026-05-19 dispatch spend per harvest cost-band anchors**:
- VQ T4 OOM: 141s × $0.59/h = $0.023
- VQ A10G retry: unknown elapsed (not harvested)
- SGLD instant crash: 2.1s × $0.59/h = $0.0003
- SGLD passthrough: 19.5s × $0.59/h = $0.003

Total prior empirical spend: **~$0.026** + unknown A10G retry. None landed an empirical answer to the council T3 rank-2 question.

## Auth-eval results per axis per substrate

**ZERO** — no archive built per substrate that reached `contest_auth_eval.py` successfully. The 4 prior dispatch attempts ALL failed before producing a scored archive; the SGLD A1 passthrough archive `110cfaa3f2ebbd02` was built but auth_eval refused via Catalog #204.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #127 custody routing: NO score claims are made by this landing memo. The council T3 rank-2 op-routable empirical answer remains **DEFERRED-pending-infrastructure-remediation**.

## Highest-EV op-routable surfaced

**OP-ROUTABLE TO OPERATOR**: 4-tier remediation plan (in priority order; landing required BEFORE B1 E.7+E.8 dispatch can re-fire per Catalog #313):

### OP-1: E.7 recipe upgrade T4 → A10G or A100-80GB

**Edit target**: `.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`

**Change**: `gpu: "${MODAL_GPU:-T4}"` → `gpu: "${MODAL_GPU:-A10G}"` (or `A100`); `min_vram_gb: 14` → `22` (A10G) or `40` (A100); `min_smoke_gpu: "T4"` → `"A10G"`.

**Rationale**: per `harvest_e7_vq_k_sweep_1_t4_oom_20260519::next_action` "Re-dispatch on A100-80GB OR enable autocast_fp16+reduce batch_size 16->4 OR gradient_checkpointing per Catalog #218". Recipe filename `_modal_t4_dispatch.yaml` would be misleading post-edit; consider renaming to `_modal_a10g_dispatch.yaml` per Catalog #249 (forbidden misleading-directory-name pattern). Per-K dispatch cost ~$0.45 on A10G vs ~$0.30 on T4; 8-K sweep envelope **~$3.60** (still within original $4.20 cap).

**After landing**: register fresh `register_probe_outcome` PROCEED row referencing the recipe edit; the substrate-level latest blocking row will be superseded; Catalog #313 will accept the dispatch.

**Sister-subagent**: this edit is SAFE for a future B1 sister slot to perform (operator approves single-line recipe edit per Catalog #110/#113 APPEND-ONLY discipline — the YAML is LIVE_RECIPE per Catalog #113).

### OP-2: E.8 SGLD trainer scope fix

**Edit target**: `experiments/train_substrate_stack_of_stacks.py` (NOT the recipe — the recipe is correct; the trainer doesn't have a SGLD-only entrypoint)

**Change**: expose `--sgld-only-polish-mode` flag OR `--enable-sgld-convergence-diagnostic` flag that bypasses the single-arm A1 passthrough and directly runs the Welling-Teh SGLD polish loop with `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` consumed.

**Rationale**: per `sgld_convergence_dispatch_trainer_only_single_arm_passthrough_not_real_sgld_DEFER_20260519::next_action` "(1) re-scope SGLD convergence-diagnostic to a dedicated trainer that actually runs Welling-Teh SGLD polish epochs (not a single-arm A1 passthrough) AND emits convergence curves". This is a 200-400 LOC trainer-side fix per Catalog #220 (substrate L1+ scaffold operational mechanism declaration).

**Sister-subagent**: this is NOT a single-line edit; requires a dedicated subagent slot per CLAUDE.md "Subagent coherence-by-default". Estimated 1-2 hours scaffold + tests.

### OP-3: Re-fire harvest on missing A10G call_id

**Action**: `tools/harvest_modal_calls.py --call-id fc-01KRZCX15GAF5Z5E3E568Q60FF` (or `--recover-from-tmp` per Catalog #339)

**Rationale**: the A10G retry dispatch at 05:56:22Z is not in the harvest set; either it's stale (>24h cache) or harvest hasn't run for it yet. Confirming its rc/elapsed would resolve whether A10G already solved E.7 OOM (in which case OP-1 is the recipe re-pin, NOT a fresh dispatch).

### OP-4: Refresh predecessor expiration after OP-1+OP-2 land

**Action**: write fresh `register_probe_outcome` PROCEED rows for both substrates (the existing rows expire 2026-06-02 → 2026-06-18 anyway, but APPEND-ONLY latest-row-wins discipline per Catalog #110/#113 means a PROCEED row dated 2026-05-20+ supersedes the DEFER row).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A — no archive built, no per-axis weights
2. **Pareto constraint**: ACTIVE in the negative direction — this slot's verdict adds a Pareto-feasibility refusal anchor (Catalog #313 BLOCKING) to the cathedral autopilot's candidate ranker via `tac.probe_outcomes_ledger`
3. **Bit-allocator hook**: N/A — no archive built
4. **Cathedral autopilot dispatch hook**: ACTIVE — the canonical predecessor-probe-outcome surface this slot consulted via Catalog #313 IS the autopilot dispatch gate; this landing memo + the probe outcomes ledger are the operator-facing canonical answer
5. **Continual-learning posterior update**: ACTIVE — landing this memo + commit body cross-references the cathedral autopilot evidence stream via the canonical `.omx/state/probe_outcomes.jsonl` (no new write from this slot per Catalog #131 bare-write discipline; relying on the existing ledger as the source of truth)
6. **Probe-disambiguator**: ACTIVE — the predecessor-probe-outcome verdict IS the canonical disambiguator between "fire dispatch" vs "remediate infrastructure first"; this slot honors the disambiguator's BLOCKED verdict

## Canonical-vs-unique decision per layer

(Per Catalog #290 + UNIQUE-AND-COMPLETE-PER-METHOD operating mode)

| Layer | Canonical adopted | Unique-fork | Rationale |
|---|---|---|---|
| Predecessor probe check | `tools/check_predecessor_probe_outcome.py` (Catalog #313) | N/A | Canonical authority for this surface; no reason to fork |
| Dispatch entry point | `tools/operator_authorize.py` (Catalog #176) | N/A | Canonical operator-authorize wraps Catalog #152/#243/#271/#313/#167/#244/#339/#245 |
| Modal call_id ledger | `tac.deploy.modal.call_id_ledger` (Catalog #245) | N/A | Canonical 4-layer pattern |
| Sister-checkpoint guard | `tac.commit_safety.check_files_against_sister_checkpoints` (Catalog #340) | N/A | Canonical wrapper inside the serializer |
| Probe outcomes ledger | `tac.probe_outcomes_ledger` (Catalog #313) | N/A | Canonical helper this slot consulted; did not write new rows because no NEW probe outcome arose from this slot (existing rows already capture the DEFER state) |
| Landing memo | `.omx/research/<topic>_landed_<YYYYMMDD>.md` per Catalog #229/#290/#294/#303/#305 | N/A | Canonical OSS-hermetic location |

**No unique-forks needed** — this slot is a defensive/observational slot per Phase 1+2 verdict; the canonical apparatus did its job, the slot honors the verdict.

## Cargo-cult audit per assumption

(Per Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| "Council T3 rank-2 op-routable + operator-frontier-override = dispatch can fire now" | **CARGO-CULTED** | The override was issued BEFORE empirical infrastructure failures (T4 OOM + single-arm passthrough) were discovered; the override does not retroactively cover predecessor-probe DEFERs |
| "Catalog #313 BLOCKED can be bypassed via paired-env override" | **CARGO-CULTED-AND-REFUSED** | Per CLAUDE.md "Cross-agent dispatch coordination" + codex sister's explicit advice 2026-05-19T21:13Z; bypass would be duplicate-spend AND would re-introduce the same infrastructure failures |
| "Recipe edits are within my scope" | **HARD-EARNED** | Task brief explicitly states: "DO NOT touch active sister files (cathedral_autopilot / master_gradient* / sensitivity_map* / cathedral_consumers/master_gradient_*)" and operator-authorize_recipes/*.yaml is "READ only unless recipe doesn't exist + you need to CREATE". The recipes exist + are stale infrastructure-side; operator-routed remediation is the correct path |
| "Empirical answers to council T3 rank-2 question can be produced this session" | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | The session evidence (4 prior failed dispatches + codex refusal + recipe staleness) empirically falsifies the assumption that 1 more dispatch attempt produces the answer; infrastructure remediation must come first |

## Observability surface

(Per Catalog #305)

**Inspectable per layer**:
- Predecessor probe outcome: `tools/check_predecessor_probe_outcome.py --substrate vq_vae --json` AND `--substrate stack_of_stacks --json`
- Prior dispatch history: `grep -E "(vq_vae_k_sweep|sgld_convergence)" .omx/state/modal_call_id_ledger.jsonl`
- Active dispatch claims: `grep -iE "(vq_vae|sgld|cable_b1)" .omx/state/active_lane_dispatch_claims.md`
- Recipe state at HEAD: `git log --oneline -- .omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`

**Decomposable per signal**: 2 substrates × 4-5 distinct DEFER probe outcomes = 8-10 atomic decision points the operator can review independently

**Diff-able across runs**: `git log` on probe_outcomes.jsonl + modal_call_id_ledger.jsonl shows full evolution since 2026-05-19T03:31Z (original DEFER) through 2026-05-19T21:13Z (codex refusal) through 2026-05-20T01:30Z (this landing)

**Queryable post-hoc**:
```bash
.venv/bin/python -c "from tac.probe_outcomes_ledger import query_blocking_outcomes; \
  outcomes = query_blocking_outcomes(); \
  print('\n'.join(f\"{o['substrate']}: {o['probe_id']} -> {o['verdict']}\" for o in outcomes if o['substrate'] in ('vq_vae', 'stack_of_stacks')))"
```

**Cite-able**: this memo cites every probe outcome row by `probe_id`, every prior dispatch by `call_id`, every prior agent attempt by their landing memo path

**Counterfactual-able**: after OP-1 lands (recipe edit to A10G), the operator can issue a single fresh `register_probe_outcome` PROCEED row; Catalog #313 will then accept dispatch; the next slot can re-fire WITHOUT touching trainer code (OP-2 is independent and gates only E.8)

## Sister coordination per Catalog #230 ownership map

- Sister Z (`a3d213fbd1f65427f`, cathedral autopilot): DISJOINT scope — touches `tools/cathedral_autopilot_autonomous_loop.py` + `src/tac/cathedral_consumers/`; this slot touches only `.omx/research/<new memo>.md` + checkpoint JSONL via canonical helpers
- Sister AA (`abc004f1e7a427c12`, Cable D wire-in): DISJOINT scope — touches `src/tac/master_gradient*.py` + `src/tac/sensitivity_map*.py`; no overlap
- This slot: touches ONLY new landing memo (NEW file, no edit) + checkpoint JSONL via canonical helper

## Catalog #229 premise verification log

- PV-0: Council memo `.omx/research/council_t3_tier_45_backlog_prioritization_20260519.md` (commit `79bd5695d`) confirms B1 E.7+E.8 = rank #2 op-routable + dispatch envelope $3.30-4.20
- PV-1: Operator-frontier-override `e7_e8_symposium_operator_frontier_override_20260519T051028Z.md` confirms symposium ratification (Catalog #325 surface)
- PV-2: E.7 recipe exists at canonical path, declares `dispatch_enabled: true`, declares `operator_override_*` per Catalog #300 mission-alignment
- PV-3: E.8 recipe exists at canonical path, same declarations
- PV-4: **`tools/check_predecessor_probe_outcome.py --substrate vq_vae --json` returns BLOCKING DEFER** `harvest_e7_vq_k_sweep_1_t4_oom_20260519`
- PV-5: **`tools/check_predecessor_probe_outcome.py --substrate stack_of_stacks --json` returns BLOCKING DEFER** `harvest_e8_sgld_1_instant_crash_20260519`
- PV-6: Recipe-path queries return null (no recipe-keyed blocker, but substrate-keyed BLOCKING dominates — this is the canonical Catalog #313 stop-condition)
- PV-7: Prior dispatch ledger shows 4 attempts 2026-05-19T05:42-05:56Z, all rc=1, all harvested 06:04Z; recipe is STILL at T4 + min_vram_gb=14
- PV-8: Codex sister `019de465` memo confirms `operator_authorize.py --recipe ... --yes` exits rc=1 per Catalog #313 with explicit "Do not bypass from Codex loop" advice
- PV-9: Sister checkpoint guard reveals no active overlapping subagents on my declared files_touched

## Cross-references

- Council T3 op-routable: `.omx/research/council_t3_tier_45_backlog_prioritization_20260519.md` (commit `79bd5695d`)
- Operator-frontier-override (symposium-tier; does NOT cover predecessor-probe DEFERs): `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- E.7+E.8 PREP combined memo: `.omx/research/e7_e8_prep_synthesis_20260519T043602Z.md`
- E.7 dispatch verdict (prior subagent landing): `.omx/research/e7_vq_k_sweep_dispatch_verdict_20260519T060000Z.md`
- E.8 dispatch verdict (prior subagent landing): `.omx/research/e8_sgld_convergence_dispatch_verdict_20260519T060000Z.md`
- Codex 2026-05-19T21:13Z refusal memo: `.omx/research/codex_findings_catalog204_item4_dispatch_refused_by_probe_outcome_20260519T211313Z_codex.md`
- Probe outcomes ledger: `.omx/state/probe_outcomes.jsonl`
- Modal call_id ledger: `.omx/state/modal_call_id_ledger.jsonl`
- Active dispatch claims: `.omx/state/active_lane_dispatch_claims.md`
- CLAUDE.md non-negotiables: "Forbidden premature KILL without research exhaustion" + "Cross-agent dispatch coordination" + Catalog #313 + Catalog #199 + Catalog #340 + Catalog #270 + Catalog #218 + Catalog #220 + Catalog #110/#113

<!-- # FORMALIZATION_PENDING:landing_memo_documents_catalog_313_blocked_state_no_new_canonical_equation_emitted_this_slot_per_catalog_344_canonical_equations_registry_landing_implications_only_per_forbidden_premature_kill_discipline -->
