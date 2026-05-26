# SPDX-License-Identifier: MIT
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: paired with .omx/research/comprehensive_roadmap_20260526.md; commands derived from T3 symposium top-3 EV ranking (Section 4.2 priorities 1-3) + sister T3 symposium commit `3ae0b700a`. -->
<!-- # FORMALIZATION_PENDING:operator_command_sheet_canonical_paired_env_discipline_per_catalog_199_no_new_canonical_equation_needed -->

# COMPREHENSIVE ROADMAP — OPERATOR COMMAND SHEET 2026-05-26

**Paired with:** `.omx/research/comprehensive_roadmap_20260526.md` (synthesis memo)
**Discipline:** Per Catalog #199 paired-env bypass discipline + CLAUDE.md "Executing actions with care" non-negotiable.
**NOT operator-executed:** these are ready-to-paste canonical commands. Operator decides whether/when to fire each.

---

## TOP-3 highest-EV next-session spawn commands

### COMMAND 1 — D=Z6 L2 LONGTRAIN extension to 3000ep contest resolution

**Predicted ΔS bracket:** [-0.015, -0.005] vs PR110 baseline (0.192028 [contest-CPU])
**Cost envelope:** $0 + ~$0.50 paid CUDA bridge calibration
**Wall-clock:** 12-24h M5 Max MLX
**Horizon class:** FRONTIER-PURSUIT (per Catalog #309)
**Dependencies (all LANDED today):** Sister #1265 gate parameterized `fc44aa670` + L2 helper `f5e4789ef` + D=Z6 L1 promotion `8833b9db5` + first canonical L2 run `ab4df5d4e`

**Operator-routable subagent dispatch (TaskCreate template):**

```
TaskCreate: SLOT-Z6-L2-3000EP-EXTENSION
prompt: |
  Extend D=Z6 L2 LONG-TRAINING from 300ep proof-of-pattern to 3000ep at contest resolution (384x512 / 600 pairs).

  MANDATORY PRE-FLIGHT:
  1. Read CLAUDE.md + AGENTS.md. Honor all NON-NEGOTIABLE.
  2. Read .omx/state/lane_registry.json + .omx/state/subagent_progress.jsonl + sister T3 symposium memo `slot_gc_t3_strategy_comprehensive_symposium_landed_20260526.md` (commit 3ae0b700a; verdict PROCEED_WITH_REVISIONS).
  3. Read .omx/research/path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526T123709Z.md for proof-of-pattern reference.
  4. `tools/subagent_checkpoint.py read --subagent-id slot-z6-l2-3000ep-extension-20260526`.
  5. Verify L2 helper at `src/tac/training/long_training_canonical/` + sister #1265 gate parameterized via `Z6PCWM1` grammar.

  YOUR SCOPE:
  - Extend the D=Z6 L2 long-training run from 300ep to 3000ep at contest resolution (384x512 / 600 pairs).
  - Use the canonical `tac.training.long_training_canonical` helper (LANDED commit `f5e4789ef`).
  - Run MLX-local on M5 Max (no paid CUDA per MLX-first doctrine 4107bbf8d).
  - Emit canonical posterior anchor per Catalog #128 fcntl-locked.
  - Emit per-100ep telemetry to `.omx/state/long_training_canonical/` per L2 helper conventions.
  - On convergence: emit landing memo at `.omx/research/path_3_d_z6_l2_3000ep_landed_<utc>.md` documenting per-100ep loss curves + final archive sha256 + bridge-calibration paired CUDA plan (~$0.50 envelope).
  - If sub-PR110 candidate emerges (score < 0.192028 [contest-CPU] paired Linux x86_64): rebaseline canonical_frontier_pointer + register canonical equation per T3 symposium REVISION 2.

  DISCIPLINE:
  - Per Catalog #229 PV.
  - Per Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256.
  - Per Catalog #206 checkpoint every 10 tool uses.
  - Per CLAUDE.md "Forbidden premature KILL without research exhaustion": if convergence stalls before 3000ep, declare DEFERRED-pending-research-with-XYZ-applied, NOT KILL.

  REPORT under 350 words: ep-converged + per-100ep loss curve summary + final archive sha256 + sub-PR110 candidate verdict + operator-routable bridge-calibration paired CUDA dispatch plan.
sister_context: pr110-opt-frame0-bundle + hinton-mlx-local-pivot + comprehensive-roadmap (all in-flight per checkpoint scan at landing time).
```

---

### COMMAND 2 — E=BoostNeRV-PR110 EMPIRICAL L1 build

**Predicted ΔS bracket:** [-0.010, +0.0045] vs PR110 baseline (direct stacking against frozen PR110 base)
**Cost envelope:** $0 macOS-local
**Wall-clock:** 1-2h M5 Max MLX
**Horizon class:** FRONTIER-PURSUIT
**Dependencies (all LANDED today):** L1-PROMOTION-CASCADE structural shell `d8203efda` (BoostNeRV adapter shell + L2 entry-point) + L2 helper `f5e4789ef` + PR110 frozen base byte-stable artifact (operator-side; sha 6bae0201fb08)

**Operator-routable subagent dispatch (TaskCreate template):**

```
TaskCreate: SLOT-E-BOOSTNERV-PR110-L1-EMPIRICAL
prompt: |
  Land empirical L1 build for E=BoostNeRV-PR110 substrate (currently STRUCTURAL-only per commit d8203efda).

  MANDATORY PRE-FLIGHT:
  1. Read CLAUDE.md + AGENTS.md + Catalog #220 (L1+ scaffold operational mechanism) + Catalog #233 (L1->L2 promotion 4-gate).
  2. Read sister T3 symposium memo `slot_gc_t3_strategy_comprehensive_symposium_landed_20260526.md` Section 4.2 priority 2.
  3. Read `.omx/research/path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_aggregate_landed_20260526T134600Z.md` for structural shell context.
  4. `tools/subagent_checkpoint.py read --subagent-id slot-e-boostnerv-pr110-l1-empirical-20260526`.

  YOUR SCOPE:
  - Wire the E=BoostNeRV-PR110 STRUCTURAL adapter shell to an mlx.nn.Module renderer wrap (the missing L1 EMPIRICAL surface per T3 symposium Section 6.2 bottlenecks).
  - Train iteratively against PR110 frozen base (sha 6bae0201fb08) per BoostNeRV residual-stacking paradigm.
  - Emit canonical posterior anchor per Catalog #128.
  - Run MLX-local on M5 Max (no paid CUDA per MLX-first doctrine).
  - On empirical L1 convergence: emit landing memo `.omx/research/path_3_e_boostnerv_pr110_l1_empirical_landed_<utc>.md` documenting ΔS bracket measured + final residual archive sha256.
  - If sub-PR110 candidate emerges via stack: rebaseline frontier pointer per T3 symposium REVISION 2.

  DISCIPLINE:
  - Per Catalog #229 PV.
  - Per Catalog #220 + #272 substrate L1+ operational mechanism + distinguishing-feature contract.
  - Per Catalog #117/#157/#174 canonical serializer.
  - Per Catalog #287 placeholder-rationale rejection.
  - Per Catalog #126 lane pre-registration: lane `lane_e_boostnerv_pr110_l1_empirical_20260526`.

  REPORT under 350 words: residual archive sha256 + empirical ΔS bracket measured + sub-PR110 verdict + L2 long-training plan if L1 converges.
sister_context: D=Z6 L2 3000ep extension (Command 1) running in parallel; ensure disjoint file scope.
```

---

### COMMAND 3 — G=NIRVANA EMPIRICAL L1 build

**Predicted ΔS bracket:** [-0.005, +0.005] vs PR110 baseline (3-axis fully evidenced; sister-canonical numpy portability)
**Cost envelope:** $0 macOS-local
**Wall-clock:** 1-2h M5 Max MLX
**Horizon class:** FRONTIER-PURSUIT
**Dependencies (all LANDED today):** L1-PROMOTION-CASCADE structural shell `d8203efda` (NIRVANA adapter shell + L2 entry-point) + L2 helper `f5e4789ef` + numpy reference complete per FIX-WAVE-R3 `path_3_fix_wave_r3_g_op1_nirvana_test_expand_landed_20260526.md`

**Operator-routable subagent dispatch (TaskCreate template):**

```
TaskCreate: SLOT-G-NIRVANA-L1-EMPIRICAL
prompt: |
  Land empirical L1 build for G=NIRVANA cascading NeRV substrate (currently STRUCTURAL-only per commit d8203efda).

  MANDATORY PRE-FLIGHT:
  1. Read CLAUDE.md + AGENTS.md + Catalog #220 + #233.
  2. Read sister T3 symposium memo `slot_gc_t3_strategy_comprehensive_symposium_landed_20260526.md` Section 4.2 priority 3.
  3. Read `.omx/research/path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_aggregate_landed_20260526T134600Z.md`.
  4. Read `.omx/research/path_3_fix_wave_r3_g_op1_nirvana_test_expand_landed_20260526.md` for 3-axis evidence + numpy ref.
  5. `tools/subagent_checkpoint.py read --subagent-id slot-g-nirvana-l1-empirical-20260526`.

  YOUR SCOPE:
  - Wire the G=NIRVANA STRUCTURAL adapter shell to mlx.nn.Module cascading-NeRV renderer wrap.
  - Train via canonical L2 helper per cascading-residual paradigm (NIRVANA = hierarchical-residual sister of E=BoostNeRV iterative-boosting).
  - Use numpy reference (3-axis evidence) as bit-exact correctness gate.
  - Emit canonical posterior anchor per Catalog #128.
  - Run MLX-local on M5 Max.
  - On empirical L1 convergence: emit landing memo `.omx/research/path_3_g_nirvana_l1_empirical_landed_<utc>.md`.
  - Cross-reference sister numpy portability pattern for downstream MLX cascade subagents.

  DISCIPLINE:
  - Per Catalog #229 PV.
  - Per Catalog #220 + #272.
  - Per Catalog #117/#157/#174 canonical serializer.
  - Per Catalog #126 lane pre-registration: lane `lane_g_nirvana_l1_empirical_20260526`.

  REPORT under 350 words: cascading-NeRV archive sha256 + empirical ΔS bracket + numpy bit-exact verdict + L2 long-training plan if L1 converges.
sister_context: D=Z6 L2 3000ep (Command 1) + E=BoostNeRV (Command 2) running in parallel; ensure disjoint file scope (G=NIRVANA touches different substrate dir).
```

---

## Critical operator decisions (from T3 symposium PROCEED_WITH_REVISIONS verdict)

### D1 — Z6/Z7/Z8 predictive-coding class 14-day go/no-go

**Context:** D=Z6 L2 first canonical run LANDED today (commit `ab4df5d4e`; 300ep / 66% loss reduction / 23× #1265 gate margin). Z7+Z8 are L0 SCAFFOLD with cargo-cult-first design memos.

**Options:** A) COMMIT-TO-3000EP (above Command 1) — RECOMMENDED. B) DEFER-WITH-REACTIVATION-CRITERIA. C) PARALLEL D=Z6 + Z7/Z8 design memo iteration.

**Cost of delay:** HIGH (each day loses L2 helper reference-template value).

### D2 — Cascade doctrine HOLD-Tier-3-L0-spawns elevation + Catalog #298 retirement sweep

**Context:** ~50 stale L0 SCAFFOLD lanes from 2026-04 / 2026-05-early. Per Schmidhuber dissent: 80% of L1+ lanes never completed L2 long-training.

**Options:** A) ELEVATE to CLAUDE.md non-negotiable + retire ~50 L0 lanes. B) PRESERVE doctrine-only. C) PARTIAL — keep doctrine + retire ~50 L0 lanes. RECOMMENDED: C.

**Operator-runnable retirement sweep command (per Catalog #298 + CLAUDE.md "Substrate retirement discipline"):**

```bash
# Dry run first:
.venv/bin/python tools/audit_stale_l1_substrates.py --report-out /tmp/stale_l0_audit_$(date +%Y%m%dT%H%M%S)Z.json

# Apply (after operator review of dry-run):
.venv/bin/python tools/lane_maturity.py mark --lane <lane_id> --gate impl_complete --evidence "STALE_L0_ARCHIVED_per_catalog_298_2026-05-26"
# Loop for each stale lane in the JSON report, OR:
.venv/bin/python tools/audit_stale_l1_substrates.py --apply --operator-approved "adpena:$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

### D5 — PR111 cadence post-sub-PR110-emergence

**Context:** PR #110 LIVE on commaai/comma_video_compression_challenge per operator 2026-05-21. Yousfi 2026-05-11 PR #108 closure: "we are going to reward folks publishing their code even if not in top 3."

**Options:** A) PR111 as soon as sub-PR110 candidate emerges. RECOMMENDED. B) WAIT for top-3 contender. C) PUBLISH WAVE 2-3 PRs.

---

## Sister coordination + dispatch hygiene

**Before spawning ANY of the above commands, operator should:**

1. Confirm no scope-overlap with currently in-flight sister subagents:
   ```bash
   tail -50 .omx/state/subagent_progress.jsonl | .venv/bin/python -c "
   import json, sys
   from datetime import datetime, timezone
   rows = []
   for line in sys.stdin:
       try: rows.append(json.loads(line.strip()))
       except: pass
   now = datetime.now(timezone.utc)
   inflight = {}
   for r in rows:
       sid = r.get('subagent_id', '')
       if sid: inflight[sid] = r
   for sid, r in inflight.items():
       if r.get('status') == 'in_progress':
           ts = datetime.fromisoformat(r.get('written_at_utc', '').replace('Z', '+00:00'))
           age_min = (now - ts).total_seconds() / 60
           if age_min < 360:
               print(f'  IN-FLIGHT: {sid[:50]} {age_min:6.1f}min')
   "
   ```

2. Verify canonical frontier pointer is current:
   ```bash
   .venv/bin/python tools/refresh_canonical_frontier.py --strict
   ```

3. Cap concurrent subagents at ≤5 per Catalog #302 (current 3 in-flight + 3 commands above = 6; exceeds cap; recommend stage Commands 1+2 first, then Commands 3 after).

---

## Discipline + paired-env bypass

Per Catalog #199: every command above runs through `tools/operator_authorize.py` for paid dispatch (Command 1 paired CUDA bridge + Command 9 PR111 submission). Bypass via paired env vars required for non-interactive subprocess invocation:

```bash
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=10.0  # operator-set envelope cap
```

NEVER set CONFIRMED without BUDGET (raises SystemExit 11). NEVER bypass without budget envelope per Catalog #199 paired-env discipline.

---

## Per-bucket task closure summary (per Section B of paired synthesis memo)

**Recommended CLOSE/RETIRE (8 stale tasks):** #1124 (end-of-day cascade reconciliation; superseded by T3 inventory) + #1186-#1188 + #1191 (HFV builders; superseded by MLX-first per operator #1196 NOT-APPROVED) + #1304 (FIX-WAVE-R1''-K-MLX-PRIMITIVES; LANDED `2d59283d4`) + #1306 (COMPREHENSIVE-BUG-AUDIT; LANDED `086890143`) + #1262 (HINTON-MLX-NUMPY-EXPORT-PARITY-PROOF; SUPERSEDED-BY-current-#1330-subagent).

**Recommended DEFER (4 tasks per Catalog #298):** #1115 + #1119 + 3 non-active PR110-OPT vectors (per cascade doctrine HOLD).

**Recommended HOLD per cascade doctrine (4 tasks):** #1280-#1283 Path 3 L/M/N/O Tier 3 L0 SCAFFOLD spawns (reactivation criterion: ≥3 L2 substrates converge).

**Recommended SPAWN (3 tasks; THIS document's TOP-3):** #1316 (E=BoostNeRV L1) + #1317 (G=NIRVANA L1) + new SLOT-Z6-L2-3000EP-EXTENSION.

**Operator-decision (3 tasks):** #1131 + D1 + D2 + D5 above.

**Continue in-flight (3 sister subagents at this landing time):** pr110-opt-frame0-bundle + hinton-mlx-local-pivot + comprehensive-roadmap (THIS subagent completing now).
