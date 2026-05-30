# Retroactive sweep for optimizer-stack-inventory research landing (2026-05-30)

Per Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`).

## 1. Bug-class symptom signature

This is a research-recommendation landing (NOT a new STRICT preflight gate; NO new Catalog # claimed). The bug class this sweep addresses is the "optimizer-stack inventory drift" + "bleeding-edge research stale" + "operator routes to optimizer that isn't in codebase" failure mode per the operator META-correction 2026-05-30 verbatim *"we also have extensive adamw and muon and variants and new bleeding edge alternatives and extensive reserach memos and such too"*.

Sister Catalog #348 scope: NEW gate landings require retroactive verdict sweep so old KILL / DEFER / FALSIFY verdicts on optimizer choices do not silently rot. THIS landing is recommendations-only (no source mutation; no NEW STRICT gate) so the canonical 4-field contract applies as follows.

## 2. Pre-fix window

Window: 2026-05-13 → 2026-05-30 (inclusive of the canonical Muon + Langevin + IGLT landings on 2026-05-13 and today's MLX-Score-Aware Adapter at `src/tac/substrates/_shared/mlx_score_aware/adapter.py:150` hardcoded `mlx_optim.AdamW`).

## 3. Historical KILL / DEFER / FALSIFY search results

Per `grep -i -r 'kill.*muon\|killed.*muon\|deferred.*muon\|falsified.*muon\|kill.*adamw\|deferred.*adamw\|killed.*lion\|deferred.*lion\|killed.*sophia\|killed.*schedule.?free' .omx/research/ memory/` (empirically reviewed):

- **ZERO historical KILL verdicts on Muon / AdamW / Lion / Sophia / Schedule-Free in the codebase research-memo corpus** — every prior optimizer reference is either canonical recommendation (`.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md`, `.omx/research/online_research_C_optimizers_20260513.md`) OR descriptor-only scaffold (`tac.optimization.optimizer_scheduler_registry` 13 descriptors with `backend_status=optimizer_backend_missing`) OR explicit-research-only flag (`research_only=true` per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY")
- **ONE registered canonical anti-pattern**: `pre_pr95_family_l15_adamw_only_no_muon_finetune_v1` per `tac.canonical_anti_patterns.query_anti_patterns` empirical sweep — this anti-pattern documents the FALSIFIED-IMPLEMENTATION-LEVEL pattern of pre-PR95 substrate trainers using AdamW-only without the canonical L15 Muon final-stage 77% partition. THIS landing's Recommendation Option A is the canonical UNWIND PATH for the anti-pattern (Option A IS the canonical PR95-faithful 8-stage curriculum including L15 Muon final stage).
- **ONE registered canonical anti-pattern**: `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1` per `tac.canonical_anti_patterns.query_anti_patterns` — sister of CLAUDE.md "EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS" + "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS"; THIS landing's Recommendation Option A composes with Yousfi Rev #1 which routes through `run_mlx_score_aware_full_main` which already adopts canonical EMA + eval_roundtrip per HNeRV-parity L8.
- **Probe-outcomes ledger** per `tac.probe_outcomes_ledger.query_blocking_outcomes`: ZERO blocking outcomes for substrate IDs containing `muon`, `adamw`, `lion`, `sophia`, or `schedule_free` — no prior FALSIFIED verdicts that this landing's recommendations need to invalidate.

## 4. Per-finding RE-EVAL priority assignment

Per canonical Catalog #348 4-field contract:

| Finding | RE-EVAL priority | Rationale |
|---|---|---|
| Muon canonical PyTorch impl at `src/tac/optimization/muon.py` | NONE (already canonical) | PR95 byte-faithful per memo `keller_jordan_muon_modded_nanogpt_research_20260513.md` |
| Muon canonical MLX kernel at `src/tac/local_acceleration/pr95_hnerv_mlx.py:2138-2163` | NONE (already canonical) | Same coefficients as PyTorch sister; PR95 stage-8 partition byte-faithful |
| Anti-pattern `pre_pr95_family_l15_adamw_only_no_muon_finetune_v1` | LOW (already registered) | THIS landing's Option A IS the canonical unwind path; pure recommendation surface; not yet activated empirically per CLAUDE.md "Forbidden premature KILL" |
| Anti-pattern `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1` | LOW (already registered) | Yousfi Rev #1 already routes through canonical-EMA + canonical-eval_roundtrip path; THIS landing composes orthogonally |
| `tac.optimization.optimizer_scheduler_registry` 13 descriptors | NONE (all carry false-authority + research-only fields per CLAUDE.md) | No historical FALSIFICATION; planning-only by design |
| `tac.optimization.langevin_optimizer.LangevinOptimizer` descriptor-only registration | NONE | `backend_status=optimizer_backend_missing` per design; THIS landing recommends operator-routable Wave E future canonical equation registration once empirical anchors land |
| `tac.optimization.iglt.IGLTOptimizer` descriptor-only registration | NONE | Same descriptor-only contract; future probe-disambiguator activation |
| Lion / Sophia / Schedule-Free pip-install gap | NONE (no historical KILL) | Option C DEFERRED per recommendation; canonical operator-routable path documented |
| Sister substrate 52 AdamW-only trainers | LOW (informational) | Sister cascade Wave A → Wave D documented as operator-routable next steps; no historical KILL; not yet operationalized empirically |

## 5. Verification

- Sweep verified via empirical grep + `tac.probe_outcomes_ledger.query_blocking_outcomes` + `tac.canonical_anti_patterns.query_anti_patterns`
- ZERO historical verdicts invalidated by THIS landing's recommendations
- ZERO source-text mutations this scope (recommendations-only)
- THIS landing's recommendations COMPOSE WITH (not contradict) sister Yousfi Rev #1-5 + CLAUDE.md HNeRV-parity L14 + L15
- Sister-DISJOINT verified vs `yousfi-rev-3-4-5-substrate-engineering-z8-m12a-20260530` per their checkpoint at `.omx/state/subagent_progress.jsonl` notes

## Cross-references

- THIS landing memo: `.omx/research/optimizer_stack_inventory_and_bleeding_edge_recommendations_landed_20260530.md`
- Catalog #348 contract: CLAUDE.md "Meta-bug class catalog" entry 348
- Sister Yousfi review: `.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md`
- Canonical Muon synthesis: `.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md`
- Canonical anti-pattern registry: `tac.canonical_anti_patterns.query_anti_patterns`
- Canonical probe-outcomes ledger: `tac.probe_outcomes_ledger.query_blocking_outcomes`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
