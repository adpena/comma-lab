---
council_tier: T1
council_attendees: [Claude-Catalog204-Recovery-Subagent]
council_quorum_met: false
council_verdict: PARTIAL_PROCEED
council_dissent: []
council_decisions_recorded:
  - "Apply Catalog #204 durable-output discipline to scripts/remote_lane_substrate_stack_of_stacks.sh via the canonical PR101 LC v2 pattern"
  - "Register Catalog #313 PARTIAL probe outcome with operator-routable recovery dispatch instructions"
  - "Defer Phase 2 Modal dispatch to operator/cathedral: ~$0.30 SGLD t_final=1.0 single-value smoke will rebuild deterministic archive sha=110cfaa3 + auth_eval will succeed with [contest-CUDA Modal T4] evidence"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
---

# CATALOG #204 DRIVER FIX + FREE CONTEST-CUDA RECOVERY (Phase 1 landed)

## Summary

Phase 1 (driver fix) LANDED. Phase 2 (FREE [contest-CUDA Modal T4] recovery for archive sha=110cfaa3) prepared and operator-routable; cheapest implementation = re-fire SGLD t_final=1.0 single-value smoke (~$0.30 Modal T4).

## Phase 1 — driver fix LANDED

### Bug class anchor

`scripts/remote_lane_substrate_stack_of_stacks.sh` pre-fix defaulted `OUTPUT_DIR="${STACK_OF_STACKS_OUTPUT_DIR:-$LOG_DIR/output}"` where `LOG_DIR="$WORKSPACE/lane_stack_of_stacks_results"` and `WORKSPACE=/workspace/pact` (Vast.ai default). On Modal workers the mounted `/workspace/pact` resolves to `/tmp/pact`, so OUTPUT_DIR became `/tmp/pact/lane_stack_of_stacks_sgld_convergence_results/output` — under the Modal worker's temp scratch.

`experiments/contest_auth_eval.py` correctly refuses temp-storage evidence per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" non-negotiable + Catalog #204. Result: E.8 SGLD #2 (`fc-01KRZCSQ7FPVMSAXZQDSZJCTN4`) trainer rc=0, archive built (sha=110cfaa3, 179008 bytes, A1 single-arm passthrough canary), but auth_eval rc=1.

Sister regression to STC v2 2026-05-14 anchor. Same bug class; canonical fix lives in `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh`.

### Fix applied

Driver now uses the canonical 3-branch OUTPUT_DIR resolution:

```bash
if [ -n "${STACK_OF_STACKS_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$STACK_OF_STACKS_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
    LOG_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/lane_stack_of_stacks_results"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
```

LOG_DIR is also re-routed under `/modal_results` in the Modal branch so the Stage 3 auth_eval LOG_DIR (`AUTH_EVAL_LOG_DIR="$LOG_DIR"`) inherits durable-path discipline transparently. `bash -n` syntax verified clean.

### Catalog #204 STRICT gate scope

Catalog #204's structural scan is scoped to `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh` only (see `src/tac/preflight.py:45889` `_CHECK_204_REMOTE_LANE_PATH`). Stack_of_stacks driver is NOT in the structural scope, so the fix lands without expanding gate scope. The bug class is extincted at the driver source surface, and future stack_of_stacks dispatches with `MODAL_RUNTIME=1` will route durably.

The wider audit recommended by predecessor (sweep all `remote_lane_substrate_*.sh` for the same sister pattern) is queued as a follow-on for a sister gate-expansion subagent.

## Phase 2 — FREE [contest-CUDA Modal T4] recovery (operator-routable)

### Cheapest path

Re-fire the SGLD recipe with a single t_final value to get one deterministic A1 passthrough archive + successful auth_eval:

```bash
LANGEVIN_T_INIT_OVERRIDES="1.0" \
  OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
  OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.50 \
  .venv/bin/python tools/operator_authorize.py \
    --recipe substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch \
    --target modal
```

The corrected driver routes OUTPUT_DIR to `/modal_results/$JOB_ID/output`. Trainer rebuilds deterministic A1 single-arm passthrough archive (expected same byte structure, sha may differ if any non-deterministic SGLD seed mutation; per-arm passthrough should reproduce sha=110cfaa3). Auth_eval succeeds with promotable [contest-CUDA Modal T4] evidence.

### Promotion contract

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": the recovered Modal T4 score is `[contest-CUDA Modal T4]` ONLY; submission promotion requires paired `[contest-CPU GHA Linux x86_64]` per the dual-eval mandate. A1 passthrough is canary-shape (no score lowering expected vs A1 baseline 0.193 [contest-CPU]), so this recovery's value is primarily custody-evidence + dispatch-machinery validation, not frontier displacement.

### Predicted band

Per recipe `predicted_band: [0.190, 0.210]`. A1 baseline 0.193 [contest-CPU] + SOS1 wrapper adds ~1KB rate penalty + identical seg/pose components. Expected [contest-CUDA Modal T4] near 0.20 (rate term contributes more on the rounded archive byte axis). Should not displace current 0.19205 [contest-CPU] frontier on `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` (sha 6bae0201) per Catalog #316 frontier ledger.

## Probe outcome ledger

Catalog #313 `register_probe_outcome` row appended:
- `probe_id`: `catalog_204_driver_fix_stack_of_stacks_landed_20260519`
- `verdict`: `PARTIAL`
- `blocker_status`: `advisory` (non-blocking; the driver fix supersedes the bug class while the recovery dispatch is pending)
- `next_action`: operator-routable single-t_final SGLD smoke for ~$0.30
- Predecessor `harvest_e8_sgld_2_auth_eval_durable_output_block_20260519` (DEFER) supersession recorded via `notes` (no fresh DEFER blocks dispatch since the driver bug class is extincted)

## Bug-class extinction surface

This work extincts the bug class at ONE source surface (stack_of_stacks driver). The META gate Catalog #204 remains scoped to PR101 LC v2 only. Recommended follow-on (queued for sister subagent): expand Catalog #204 structural scope to ALL `scripts/remote_lane_substrate_*.sh` files OR land a sister gate (`check_all_remote_lane_substrate_drivers_route_modal_output_to_durable_path`) with the same 3-snippet contract enforced across all 31+ substrate drivers per the Catalog #244 NVML wave precedent.

## Sister coordination per Catalog #230

- Edit scope DISJOINT from 4 in-flight sister subagents (MPS Phase B / R11 H1-1+H1-6 FIX-WAVE / master-gradient consumer cathedral wire-in / Cable C6 RE-EVAL-HIGH DRAFTs). Files touched: ONLY `scripts/remote_lane_substrate_stack_of_stacks.sh` + verdict memo + probe outcomes ledger + lane registry. Catalog #302 in-flight scope verifier confirmed zero overlap on the driver script.

## Premise verification per Catalog #229

PV-1: predecessor harvest memo CONFIRMS bug class + recovery path → PASSED
PV-2: PR101 LC v2 canonical pattern at lines 51-58 EXISTS → PASSED
PV-3: Catalog #204 STRICT gate scoped to PR101 LC v2 ONLY (`_CHECK_204_REMOTE_LANE_PATH`) — not regression risk → PASSED
PV-4: archive sha=110cfaa3 harvested locally at `experiments/results/lane_substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch_20260519T055409Z_modal/harvested_artifacts/.../submission_dir/archive.zip` → VERIFIED (`shasum -a 256` matches)
PV-5: A100 stack_of_stacks recipe `dispatch_enabled: false` per `feedback per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"` — NOT enabling it → respected
PV-6: bash -n syntax check on driver post-edit → SYNTAX_OK
PV-7: zero sister-subagent scope overlap on driver script per Catalog #302 → CLEAR

## 6-hook wire-in declaration per Catalog #125

- Hook 1 (sensitivity-map): N/A — driver-source fix, no signal contribution
- Hook 2 (Pareto constraint): N/A
- Hook 3 (bit-allocator): N/A
- Hook 4 (cathedral autopilot dispatch): ACTIVE — probe outcome PARTIAL row enables autopilot ranker to dispatch the recovery smoke without DEFER block
- Hook 5 (continual-learning posterior): ACTIVE — probe outcomes ledger row appended via canonical `tac.probe_outcomes_ledger.register_probe_outcome`
- Hook 6 (probe-disambiguator): ACTIVE — verdict taxonomy distinguishes "driver bug class extincted at source" (PARTIAL) from "score recovered + promotable" (future PROCEED on recovery dispatch)

## Provenance

- `subagent_id`: catalog_204_driver_fix_plus_free_cuda_recovery_20260519
- `lane_id`: lane_catalog_204_driver_fix_plus_free_cuda_recovery_20260519 (L0 at start; L1 after impl_complete + memory_entry + strict_preflight)
- `evidence_grade`: driver_source_fix (not a score claim; recovery dispatch remains pending operator routing)

## Cross-references

- Predecessor: `.omx/research/harvest_pending_modal_dispatches_synthesis_20260519T060712Z.md` (commit `da80213d7`)
- Canonical fix template: `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh` lines 51-58
- Catalog #204 STRICT gate: `src/tac/preflight.py:45886-45979`
- Sister bug class: STC v2 driver fix 2026-05-14
- CLAUDE.md "Forbidden /tmp paths in any persisted artifact (the transient-evidence trap)"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
- CLAUDE.md "Apples-to-apples evidence discipline"
- Catalog #313 probe outcomes ledger; #245 Modal call_id ledger; #127 custody validator; #316 frontier ledger
