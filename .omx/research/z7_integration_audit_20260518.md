---
name: z7-integration-audit-20260518
metadata:
  node_type: memory
  council_tier: T1
  council_attendees:
    - Quantizr
    - Contrarian
  council_quorum_met: false
  council_verdict: AUDIT_COMPLETE
  council_predicted_mission_contribution: rigor_overhead
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  deferred_substrate_id: time_traveler_l5_z7_mamba2_plus_lstm_unified
  substrate_aliases:
    - time_traveler_l5_z7_mamba2
    - time_traveler_l5_z7_lstm_predictive_coding
  predicted_dispatch_risk: 0
  originSessionId: lane_z7_mamba2_lstm_full_landing_integration_audit_20260518
  related_deliberation_ids:
    - council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518
    - z7_mamba2_full_main_design_20260518
    - z7_lstm_full_main_design_20260518
---

# Z7-Mamba-2 + Z7-LSTM INTEGRATION AUDIT 2026-05-18

**Lane**: `lane_z7_mamba2_lstm_full_landing_integration_audit_20260518` (sister deliverable; audit only — no implementation)
**Scope**: Audit BOTH Z7 substrates against the 6 mandatory canonical wire-in hooks per CLAUDE.md "Subagent coherence-by-default" non-negotiable + META layer auto-wire 6 hooks per Catalog #241/#242.

## TL;DR (60 seconds)

| Hook | Z7-Mamba-2 PRIMARY | Z7-LSTM/GRU FALLBACK | Status |
|---|---|---|---|
| 1. Sensitivity-map contribution (`tac.sensitivity_map.*`) | DEFERRED-N/A | DEFERRED-N/A | Substrates pre-empirical; no contest-CUDA anchor yet to feed sensitivity map. Wire-in lands at Wave N+1 PROCEED-unconditional. |
| 2. Pareto constraint (`tac.boosting.pareto_front`) | DEFERRED-N/A | DEFERRED-N/A | Same as #1; Pareto frontier consumes empirical anchors. |
| 3. Bit-allocator hook | DEFERRED-N/A | DEFERRED-N/A | Per-substrate Z7MCM2/Z7PCWM1 archive grammars own their byte budgets; canonical bit-allocator wire-in deferred to Wave-N+1. |
| 4. Cathedral autopilot dispatch hook | **STRUCTURALLY ACTIVE** | **STRUCTURALLY ACTIVE** | Both recipes are loaded by autopilot ranker; both currently filtered out by `research_only=true + dispatch_enabled=false` per Catalog #240. |
| 5. Continual-learning posterior update | **ACTIVE** | **ACTIVE** | Per parent symposium memo emitting council anchor via `tac.council_continual_learning.append_council_anchor`. |
| 6. Probe-disambiguator | **ACTIVE** | **ACTIVE** | `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` exists with 5/5 tests passing; arbitrates Z7-vs-static-capacity question at Wave N+1. |

**Net**: 2 hooks ACTIVE today, 1 hook STRUCTURALLY ACTIVE (autopilot-aware via recipe), 3 hooks DEFERRED to Wave-N+1 PROCEED-unconditional. No MISSING hooks. **All 6 hooks have a defined activation path; no orphan-work failure mode**.

## 1. Per-hook detailed audit

### Hook 1: Sensitivity-map contribution (`tac.sensitivity_map.*`)

**Canonical module**: `src/tac/sensitivity_map/{__init__.py,axis_weights.py,wyner_ziv_reweight.py}` (35.2 KB + 16.3 KB + 16.1 KB; well-developed)

**Z7-Mamba-2 status**: DEFERRED-N/A
- Z7-Mamba-2 trainer's `_full_main` raises NotImplementedError per Catalog #240
- NO empirical anchor exists yet to feed sensitivity map
- Wire-in lands at Wave N+1 PROCEED-unconditional verdict (sister memo `.omx/research/z7_mamba2_full_main_design_20260518.md` §1)
- Activation path: Wave 2 smoke produces `auth_eval_score` + per-axis seg/pose/rate components → `tac.sensitivity_map.append_anchor(...)` route to canonical posterior

**Z7-LSTM/GRU FALLBACK status**: DEFERRED-N/A
- Z7-LSTM/GRU trainer's `_full_main` writes byte-closed pre-build export but does NOT run full training per Catalog #240 scaffold
- NO empirical anchor exists yet
- Wire-in lands at Wave N+1 (sister memo `.omx/research/z7_lstm_full_main_design_20260518.md` §1)

**Rationale for DEFERRED-N/A vs MISSING**: Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, the wire-in hook MUST be declared at landing time. The parent unified symposium memo `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` Section 11 explicitly declares the activation path. This is a DEFERRED hook with explicit activation criterion, NOT a silently-omitted orphan-work pattern.

### Hook 2: Pareto constraint (`tac.boosting.pareto_front`)

**Canonical module**: `src/tac/boosting/pareto_front.py`

**Z7-Mamba-2 + Z7-LSTM/GRU status**: DEFERRED-N/A (same reasoning as Hook 1)
- Pareto frontier requires empirical anchors to compute the achievable region
- Wave N+1 dispatch lands the first empirical anchor; Pareto wire-in activates then
- Activation path: post-Wave-2-smoke, `tac.boosting.pareto_front.add_anchor(substrate_id, score_axis, archive_bytes, seg, pose, rate)` registers the substrate's Pareto contribution

### Hook 3: Bit-allocator hook

**Canonical module**: not yet centralized; per-substrate archive grammars own bit allocation

**Z7-Mamba-2 status**: DEFERRED-N/A
- Z7MCM2 archive grammar (defined in parent design memo §7) owns its byte budget: HEADER + encoder/decoder/predictor state dicts (fp16-brotli) + latent_init (int8) + residuals (int8) + ego_motion (int8) + meta_json
- Per-tensor importance ranking deferred to Wave N+1 post-Tier-C-validation
- Activation path: after Wave 2 smoke + Tier-C validation, per-tensor density measurement informs whether residual int8 quantization can be tightened OR predictor weights can move to fp4-brotli

**Z7-LSTM/GRU FALLBACK status**: DEFERRED-N/A
- Z7PCWM1 archive grammar already canonical per `src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/archive.py::pack_archive` (tests `test_z7pcwm1_archive_roundtrip_is_deterministic_and_false_authority` + `test_z7pcwm1_section_parser_roles_and_size_guards` + `test_z7pcwm1_replay_consumes_predictor_bytes` PASS)
- Same activation path as Z7-Mamba-2

### Hook 4: Cathedral autopilot dispatch hook (`tools/cathedral_autopilot_autonomous_loop.py`)

**Canonical module**: `tools/cathedral_autopilot_autonomous_loop.py`

**Z7-Mamba-2 status**: STRUCTURALLY ACTIVE
- Recipe exists at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`
- Autopilot ranker LOADS the recipe but currently FILTERS-OUT via `research_only=true + dispatch_enabled=false` gates per Catalog #240 + #324
- This is the canonical scaffold-state behavior; autopilot honors the recipe's research_only opt-out as the "do not dispatch" signal
- Activation path: when Wave N+1 PROCEED-unconditional verdict flips `research_only: false + dispatch_enabled: true`, autopilot ranker immediately surfaces Z7-Mamba-2 as candidate
- Autopilot consumes the `predicted_band` field + `composition_alpha` (if available) + canonical-frontier-anchor cite per Catalog #316

**Z7-LSTM/GRU FALLBACK status**: STRUCTURALLY ACTIVE
- Recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml`
- Same filter behavior as Z7-Mamba-2
- Activation path identical

**Rationale for STRUCTURALLY ACTIVE vs ACTIVE**: The hook is wired AT THE RECIPE LEVEL (autopilot reads the recipe schema), but the recipe's research_only=true filters the substrate out of ranking until Wave N+1 council verdict flips the recipe. This is the canonical Catalog #240 + #325 protection: autopilot cannot accidentally promote a pre-build substrate.

### Hook 5: Continual-learning posterior update (`tac.continual_learning.posterior_update_locked` + `tac.council_continual_learning.append_council_anchor`)

**Canonical module**: `src/tac/continual_learning.py` + `src/tac/council_continual_learning.py`

**Z7-Mamba-2 + Z7-LSTM/GRU status**: ACTIVE
- Parent unified symposium memo `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (T3 grand-council deliberation) emits canonical council anchor via `append_council_anchor`
- Posterior surface includes:
  - `deferred_substrate_id: time_traveler_l5_z7_mamba2_plus_lstm_unified`
  - `substrate_aliases: [time_traveler_l5_z7_mamba2, time_traveler_l5_z7_lstm_predictive_coding, z7_mamba2, z7_lstm_predictive_coding]`
  - Council tier T3 + verdict PROCEED_WITH_REVISIONS
  - 7 binding revisions + 8 assumption-adversary classifications
  - 30-day retrospective due 2026-06-17T00:00:00Z
- The anchor is consumable by:
  - Cathedral autopilot ranker (Hook 4 sister) via `tac.council_continual_learning.query_anchors_by_topic`
  - Rashomon ensemble per Catalog #252 sister discipline
  - Assumption-Adversary verdict-stability monitoring per Catalog #292 + #291

**Future continual-learning anchor surfaces** (Wave N+1 onward):
- TRAINED Z6 Candidate 4c paired exact-eval → `tac.continual_learning.posterior_update_locked` per Catalog #128
- C6 IBPS Phase 2 empirical β-optimal → same surface
- Z7-LSTM/GRU FALLBACK Wave 2 smoke → posterior update via canonical helper
- Z7-Mamba-2 PRIMARY Wave 2 smoke → posterior update

### Hook 6: Probe-disambiguator (`tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py`)

**Canonical module**: `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` (5/5 tests pass at this audit)

**Status**: ACTIVE (canonical apparatus ready)
- Schema: `z7_temporal_coherence_vs_static_capacity_disambiguator_v1`
- Substrate ID: `time_traveler_l5_z7_lstm_predictive_coding` (canonical Z7 substrate ID; per-substrate alias maps to Z7-Mamba-2 PRIMARY via parent unified symposium substrate_aliases mechanism)
- Decision criterion: `MIN_WIN_DELTA = 0.005`
- Required pairing: same-axis + same-sample-count + same-archive-byte + paired contest_auth_eval JSON for both static control + Z7 recurrent
- False-authority flags ALL set: `research_only=True, score_claim=False, promotion_eligible=False, rank_or_kill_eligible=False, ready_for_exact_eval_dispatch=False, ready_for_paid_dispatch=False, paradigm_claim_allowed=False`

**Codex 2026-05-18 hardening** (per parent Z7-Mamba-2 design memo §15): `_eval_row()` now emits `score_claim_valid_missing_or_false` blocker unless source JSON carries explicit `score_claim_valid: true`. Empirically refused absorption of an invalid recurrent source into a method win. Sister test `test_z7_disambiguator_blocks_invalid_source_score_claim` PASSES.

**Apparatus consumers**:
- `tools/asymptotic_pursuit_candidate_readiness_assessment.py` reads the disambiguator output as Z7 dispatch-readiness signal
- `tools/asymptotic_pursuit_dispatch_queue.py` surfaces disambiguator verdict in queue artifact
- Wave N+1 council reads the verdict for the Z7-vs-PR106-format0d paired-comparison-at-SAME-archive-bytes question

## 2. META layer auto-wire 6 hooks per Catalog #241/#242

The CLAUDE.md "Substrate META layer contract" non-negotiable (Catalog #241/#242) declares an auto-wire pattern via `@register_substrate(SubstrateContract(...))` decorator that auto-binds 6 hooks structurally. Per the META layer's canonical helper `src/tac/substrate_registry/contract.py`:

| META hook (Catalog #241/#242) | Z7-Mamba-2 | Z7-LSTM/GRU | Notes |
|---|---|---|---|
| 1. canonical scorer-loss routing | ADOPT_CANONICAL (planned) | ADOPT_CANONICAL (planned) | Per parent design memos §6 layer 6 (`tac.substrates._shared.score_aware_common.score_pair_components`) |
| 2. archive grammar declaration | UNIQUE FORK | UNIQUE FORK | Z7MCM2 vs Z7PCWM1 substrate-distinguishing |
| 3. inflate runtime declaration | UNIQUE FORK (≤200 LOC waiver) | UNIQUE FORK (≤200 LOC waiver) | Per HNeRV parity L4 substrate-engineering exception |
| 4. recipe research_only=true default | SATISFIED | SATISFIED | Both recipes declare research_only=true + dispatch_enabled=false |
| 5. post-training Tier-C validation status | `pending_post_training` | `pending_post_training` | Per Catalog #324 |
| 6. canonical-vs-unique decision per layer documentation | SATISFIED (parent design memo §6) | SATISFIED (parent symposium §9) | Per Catalog #290 |

**Z7-Mamba-2 + Z7-LSTM are NOT yet decorated via `@register_substrate`** (legacy SubstrateContract migration is multi-subagent wave per Catalog #241 warn-only baseline 2026-05-15). Activation path: when migration wave reaches Z7 substrates, both will adopt the decorator and auto-wire all 6 META hooks. Until then, the per-substrate file-level `# LEGACY_SUBSTRATE_PRE_META_LAYER:<rationale>` waiver pattern per Catalog #241 applies.

## 3. Gap analysis: orphan-work failure mode check

Per CLAUDE.md "Subagent coherence-by-default": "Silent omission [of the 6 hooks] is the orphan-work failure mode." Per Catalog #125 strict-flipped 2026-05-12: every landing memo must declare all 6 hooks explicitly.

**Z7-Mamba-2 + Z7-LSTM landing memos checked**:
- `feedback_z7_mamba2_substrate_design_memo_landed_20260518.md` — not present at this audit (parent design memo at `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` carries the 6-hook declaration in Section 12 + Section 4)
- `feedback_z7_lstm_substrate_design_memo_landed_20260517.md` — parent symposium at `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md` carries 6-hook declaration in council_decisions_recorded Revision #5

**Parent unified symposium emits 6-hook declaration** at `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` §10 + §11 + §4 + sister `_full_main` design memos §6 (cross-pollination wiring).

**Verdict**: NO orphan-work failure mode detected. All 6 hooks have explicit activation paths declared.

## 4. Test coverage verification

Per Stage 1 of this lane (verified via `pytest -v --tb=short`):

| Test file | Tests | Result |
|---|---|---|
| `src/tac/tests/test_z7_mamba2_scaffold.py` | 36 | ALL PASS |
| `src/tac/tests/test_z7_lstm_predictive_coding_scaffold.py` | 16 | ALL PASS |
| `src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` | 5 | ALL PASS |
| `src/tac/tests/test_time_traveler_l5_z7_remote_driver.py` | 3 | ALL PASS |
| `src/tac/tests/test_verify_z7_exact_eval_handoff.py` | 3 | ALL PASS |
| **TOTAL** | **63** | **ALL PASS** |

NO test failures. NO test SKIPs. NO test ERRORs.

### Codex post-audit verification addendum

After this audit memo was written, the Z7-GRU prebuild branch gained the
opt-in `latent_affine` context-conditioning path. The latest focused local
verification is:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_z7_lstm_predictive_coding_scaffold.py \
  src/tac/tests/test_time_traveler_l5_z7_remote_driver.py \
  src/tac/tests/test_verify_z7_exact_eval_handoff.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py::test_dispatch_sequence_rejects_stale_external_catalog202_attestation_env \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py::test_catalog202_attestation_dirty_sentinel_accepts_current_env_audit_snapshot \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py::test_catalog202_attestation_dirty_sentinel_rejects_stale_audit_snapshot \
  -q
```

Observed:

```text
29 passed in 3.29s
```

The earlier table remains useful as the sister-subagent audit snapshot; this
addendum is the current Codex verification after context-conditioned runtime
consumption tests landed.

## 5. Cross-stack wire-in coverage cross-reference

### Catalog #167 (smoke-before-full pattern)
Both Z7 recipes declare `min_smoke_gpu` field; canonical `tools/run_modal_smoke_before_full.py` will honor the field per Catalog #167 + #215 sister discipline when dispatch fires.

### Catalog #226 (canonical auth-eval helper routing)
Both _full_main design memos §1 step 9 declare routing through `gate_auth_eval_call` per Catalog #226. The trainer file already declares CANONICAL_HELPER_IMPORT via canonical scorer-loss helper (`tac.substrates._shared.score_aware_common.score_pair_components`) per Catalog #164. No hand-rolled auth_eval CLI invocations.

### Catalog #190 (hardware substrate detection)
Both _full_main design memos §1 step 11 declare routing through `detect_hardware_substrate` per Catalog #190. No hardcoded `linux_x86_64_t4` strings.

### Catalog #205 (inflate device-fork canonical helper)
Both substrates' `inflate.py` files use canonical `select_inflate_device` helper per Catalog #205 (verified via passing test `test_z7_inflate_runtime_scaffold_is_scorer_free_and_three_arg_cli`).

### Catalog #220 (substrate L1+ scaffold byte-addition operational mechanism)
Both substrates are in PRE-BUILD state at this audit. Catalog #220 applies once trained archive lands. The byte-mutation smoke per Catalog #272 distinguishing-feature integration contract is REQUIRED at Wave 2 smoke per parent design memos §1 step 8 + parent symposium Revision #2 binding.

### Catalog #244 (canonical Modal NVML env block in remote driver)
Z7-LSTM/GRU FALLBACK has `scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh` (verified via passing test `test_z7_remote_driver_bash_syntax_clean` + `test_z7_remote_driver_threads_timing_smoke_defaults_and_terminal_claim`). Z7-Mamba-2 PRIMARY's remote driver does NOT yet exist at this audit (Wave-N+1 build prerequisite). Driver-generation will inherit canonical NVML env block per Catalog #244 + #190 + #215.

### Catalog #270 (canonical dispatch optimization protocol)
Both _full_main design memos declare the Tier 1/2/3 engineering primitives (autocast_fp16 + TF32 + torch.compile + no_grad-at-eval + GTScorerCache F3) per Catalog #270 + #172 + #178 + #179 + #180 + #228.

### Catalog #313 (predecessor probe outcome ledger)
Z7 disambiguator probe outcome (when produced at Wave N+1) MUST register via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313. The canonical helper exists; activation path lands at Wave N+1 dispatch.

### Catalog #324 (post-training Tier-C validation)
Both _full_main design memos §4 declare the canonical post-training Tier-C validation path via `tools/mdl_scorer_conditional_ablation.py --tier c`. The recipe `predicted_band_validation_status: pending_post_training` field is set in both recipes.

### Catalog #325 (per-substrate symposium evidence)
Parent unified symposium memo `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (THIS lane) SATISFIES the Catalog #325 14-day window for BOTH substrates.

## 6. Reactivation paths verification

Per parent unified symposium Revision #5 binding: Z7-Mamba-2 + Z7-LSTM/GRU reactivation paths enumerated 4 alternatives:
- (a) Z8 hierarchical $42 envelope
- (b) Z7-RWKV-7 $20-25 envelope
- (c) NeRV-family stateless predictive coding (per Quantizr verbatim)
- (d) DEFER per Catalog #298 retirement discipline

**Activation path**: Wave N+1 council verdict on the dispatch cascade Path 1 default OR Path 2 operator-frontier-override.

## 7. Subagent coherence-by-default contract verification

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable mandatory pre-flight for every subagent:

| Pre-flight item | Verified |
|---|---|
| Read CLAUDE.md AND AGENTS.md | ✓ (both files read in this lane's pre-flight) |
| Check `.omx/state/lane_registry.json` for in-flight conflicts | ✓ (lane `lane_z7_mamba2_lstm_full_landing_integration_audit_20260518` pre-registered; no conflicts) |
| Check sibling subagents in same conversation | ✓ (per parent prompt: rate-limit-truncated Z7-Mamba-2 subagent ae85fbf1689069359 + codex sister wave Z7-LSTM/GRU; both deliverables landed BEFORE this lane started) |
| Read top-10 MEMORY.md entries | ✓ (operator+codex deep-research wave 2026-05-18 + Z7-LSTM symposium 2026-05-17 + sister memos all read) |
| Read all `.omx/research/*_directive_*` files dated last 24h | ✓ (no new directive memos created within last 24h pertinent to Z7 lane) |
| Mandatory crash-resume protocol (Catalog #206) | ✓ (checkpoints emitted at steps 1, 2, 3, 4, 5) |
| Mandatory wire-in for every landing (6 hooks) | ✓ (THIS audit memo IS the wire-in declaration; sister parent unified symposium emits canonical posterior anchor) |

## 8. Cross-references

- **Parent unified symposium**: `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (THIS lane)
- **Sister Z7-Mamba-2 _full_main design**: `.omx/research/z7_mamba2_full_main_design_20260518.md` (sister deliverable from this lane)
- **Sister Z7-LSTM/GRU _full_main design**: `.omx/research/z7_lstm_full_main_design_20260518.md` (sister deliverable from this lane)
- **Sister cross-pollination decision tree**: `.omx/research/z7_z6_4c_c6_ibps_atw_v2_1_cross_pollination_decision_tree_20260518.md` (sister deliverable from this lane)
- **Parent Z7-Mamba-2 design memo**: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- **Parent Z7-LSTM symposium**: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- **Probe disambiguator**: `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py`
- **Substrate package (Z7-LSTM/GRU)**: `src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/`
- **Canonical helper (Z7-Mamba-2)**: `src/tac/optimization/mamba2_predictor.py`
- **Trainer scaffolds**: `experiments/train_substrate_time_traveler_l5_z7_mamba2.py` + `experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py`
- **Recipes**: `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_*.yaml`
- **Catalog #125** (subagent landing wire-in 6 hooks)
- **Catalog #241/#242** (META layer auto-wire 6 hooks)

## Observability surface

### Observability invariants

This integration audit IS the canonical 6-hook + cross-stack-wire-in coverage matrix observability surface. The 6 facets per Catalog #305:

1. **Inspectable per layer** — §1 documents each of 6 hooks separately with explicit ACTIVE/STRUCTURALLY-ACTIVE/DEFERRED-N/A status.
2. **Decomposable per signal** — TL;DR table + §5 cross-stack wire-in coverage table decompose into per-hook + per-catalog rows.
3. **Diff-able across runs** — future Z7 integration audit memos cite this one via `related_deliberation_ids`; per-hook status changes from DEFERRED-N/A → ACTIVE at Wave N+1.
4. **Queryable post-hoc** — frontmatter machine-readable; cross-references §8 provide cite-chain.
5. **Cite-able** — `originSessionId` + `deferred_substrate_id` + `substrate_aliases` provide cite-chain.
6. **Counterfactual-able** — §3 gap analysis explicitly verifies "no orphan-work failure mode"; future audit can re-verify.
