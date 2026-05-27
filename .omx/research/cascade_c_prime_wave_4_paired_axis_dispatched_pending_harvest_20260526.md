# Cascade C' WAVE-4: Modal T4 dispatch fired; pending harvest

**Date**: 2026-05-26 (UTC 2026-05-27T01:25:47Z)
**Lane**: `lane_cascade_c_prime_option_a_build_scaffold_20260526`
**Subagent**: `cascade-c-prime-frame-1-segnet-waterfill-substrate-WAVE-4-paired-cuda-cpu-re-dispatch-post-fix-wave-3-strict-gates-and-sister-codex-inflate-py-fix-4th-attempt-20260526`
**Verdict**: DISPATCHED-PENDING-HARVEST (FunctionCall in-flight on Modal T4 at session-budget exhaustion)
**Mission contribution** per Catalog #300: `frontier_breaking` (Cascade C' Atick-Redlich asymmetric scorer channel L0 SCAFFOLD validation cycle; predicted -0.058820 score delta vs frontier per `[macOS-MLX research-signal; paired-CUDA-pending]`)

## Canonical ledger anchors

- Modal call_id: **`fc-01KSKGKACS7X28HM3RKDJ7MRF8`** (registered to `.omx/state/modal_call_id_ledger.jsonl` per Catalog #245 + #339 fail-closed)
- Modal app: `ap-I1gG6UXKnuyhGI3ZQOUyBq` ([https://modal.com/apps/adpena/main/ap-I1gG6UXKnuyhGI3ZQOUyBq](https://modal.com/apps/adpena/main/ap-I1gG6UXKnuyhGI3ZQOUyBq))
- Label: `substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch_20260527T012512Z`
- Recipe: `substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch` (`dispatch_enabled: true`, `research_only: false`)
- GPU: T4; cost band p10/p50/p90 = $0.00/$0.02/$0.19 (N=11 empirical_posterior)
- HEAD at dispatch: `b6922f84e4f3` (sentinel files all HEAD-clean)
- Local artifact dir: `experiments/results/lane_substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch_20260527T012512Z_modal/`

## Pre-flight verification (all PASS)

1. **HEAD verification**: 3 required commits present — `b026dab3e` (FIX-WAVE 3 STRICT gates) + `5bcb53070` (sister codex inflate.py fix) + `d0c4517ea` (sister codex repair-waterfill integration)
2. **3 NEW STRICT Catalog gates verified in `src/tac/preflight.py`**: `_check_365_*` (lines 59285+) + `_check_366_*` (59528+) + `_check_367_*` (59850+)
3. **Local pre-deploy harness `tools/local_pre_deploy_check.py --strict`: 9/9 PASS** — `py_compile` + `trainer_importable` + `full_main_implemented` + `archive_grammar` + `auth_eval_reachability` + `canonical_inflate_device` + `deterministic_zip` + `recipe_status_consistent_with_trainer_state` + `dispatch_optimization_protocol` (Tier 1/2/3 = 5/5 + 8/8 + 5/5)
4. **Canonical dispatch optimization protocol**: PASS per Catalog #270

## Dispatch flow

- **Attempt 1** (rc=2): Catalog #166 `--require-clean-head` REFUSED because `.omx/state/active_lane_dispatch_claims.md` + `.omx/state/modal_call_id_ledger.jsonl` were dirty (subagent-state-ephemeral bucket). All 6 sentinel files (Catalog #166 contract) HEAD-clean — verified via `git diff --name-only HEAD -- <sentinel-set>` returning empty.
- **Attempt 2** (SUCCESS): Catalog #202 paired-env bypass (`OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1`) per CLAUDE.md Catalog #199 sister. Modal worker-side hash check (Catalog #166) still runs; bypass only relaxes whole-tree refusal.
- **Operator-authorize bypass**: paired-env per Catalog #199 (`OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00`)
- **codex pre-dispatch review**: `verdict=advisory` (cost-gate skipped at $0.02 < $1.00 threshold)
- **D9 routing**: class=`smoke` canonical=modal/T4 (pass-through)
- **Catalog #245 register_dispatched_call_id_fail_closed**: SUCCESS (canonical ledger row landed)

## Auth-eval axis configuration observation (operator-routable)

`modal_metadata.json` carries `auth_eval_device: "cpu"` + `auth_eval_advisory_only: true` + `inline_auth_eval_contract_required: true`. **This is NOT a paired-CUDA scoring dispatch by the canonical Catalog #246 definition** — it is the cost-band-smoke variant routing CPU-side advisory auth-eval. For a true paired-axis (CUDA + CPU) scoring anchor producing a `[contest-CUDA]` + `[contest-CPU]` Provenance pair, the canonical path is `tools/dispatch_modal_paired_auth_eval.py --skip-axis-if-promotable-anchor-exists` per Catalog #246, NOT direct `operator_authorize` of a smoke recipe.

## Poll status at session-budget exhaustion

- Poll cycle 1: 600s — `still_running`
- Poll cycle 2: ~400s — backgrounded; ledger query at exit shows `1 dispatched event, 0 terminal events`
- Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE": FunctionCall is durably registered in canonical ledger (~24h TTL); artifacts in result-cache.

## Operator-routable next steps

1. **Harvest** (when FunctionCall completes; auto-fires per Catalog #343 dispatcher hook):
   ```
   .venv/bin/python tools/harvest_modal_calls.py
   .venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KSKGKACS7X28HM3RKDJ7MRF8
   ```
2. **If empirical CPU-advisory anchor PASS** + score < 0.20 (sub-PR101-GOLD band per canonical frontier pointer's `our_local_frontier_contest_cpu.score = 0.19202828295713675`): route paired-axis dispatch via `tools/dispatch_modal_paired_auth_eval.py` for the contest-CUDA anchor needed for canonical equation #344 PROMOTION + Catalog #324 `predicted_band_validation_status: validated_post_training`.
3. **If empirical anchor FAIL** (smoke crash / OOM / inflate error): per Catalog #307 IMPLEMENTATION-LEVEL falsification + Catalog #325 per-substrate symposium re-deliberation; PARADIGM INTACT (Atick-Redlich asymmetric channel theory unchanged).
4. **If WAVE-5 needed** (5th attempt for cycle closure): sister subagent per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first".

## Discipline declaration

- Catalog #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` for commit
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #206 checkpoint discipline (5 in-progress checkpoints + 1 complete on session end)
- Catalog #229 premise verification (verified HEAD commits + 3 STRICT gates + sister codex fixes)
- Catalog #230 sister-disjoint (zero overlap with FIX-WAVE preflight.py work)
- Catalog #287 placeholder rejection (no `<rationale>` / `<reason>` literals)
- Catalog #340 sister-checkpoint guard PROCEED (no in-flight sister collision)
- Catalog #343 NO hardcoded frontier-score literals (cite canonical pointer)
- Catalog #344 PROMOTION DEFERRED (paired-axis not yet measured)
- Catalog #167 + #199 + #245 + #339 + #166 + #202 honored

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: N/A (defensive dispatch action)
- Hook #2 Pareto constraint: N/A
- Hook #3 bit-allocator: N/A
- Hook #4 cathedral autopilot dispatch: **ACTIVE** (Modal call_id registered to canonical ledger; downstream autopilot ranker will consume harvested artifact post-completion)
- Hook #5 continual-learning posterior: **ACTIVE PENDING** (post-harvest auth_eval result will append `ContestResult` row to `.omx/state/continual_learning_posterior.jsonl` per `tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome` hook)
- Hook #6 probe-disambiguator: N/A (single substrate dispatch; not a probe-disambiguator branch)

## Cycle closure verdict

**DEFERRED-PENDING-HARVEST**. Dispatch is durably registered; harvest is operator-routable. Validation cycle for Cascade C' WAVE-4 cannot CLOSE until FunctionCall completes + harvester appends terminal `harvested`/`failed`/`stale` event to canonical ledger. PARADIGM-LEVEL Atick-Redlich asymmetric scorer channel synthesis unchanged.
