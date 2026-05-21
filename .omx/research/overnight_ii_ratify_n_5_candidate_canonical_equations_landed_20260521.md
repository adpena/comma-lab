# OVERNIGHT-II RATIFY-N: 5 candidate canonical equations from FF T4 symposium — LANDED 2026-05-21

**Lane**: `lane_overnight_ii_ratify_n_5_candidate_canonical_equations_from_ff_t4_symposium_20260521`
**Source authorization**: operator 2026-05-21 verbatim *"Approved continue with all"*
**Source candidates**: `.omx/research/t4_grand_council_symposium_full_stack_synthesis_theoretical_floor_cascade_landed_20260521.md` §4.3 lines 520-530 (FF T4 symposium commit `7719d4c81`)
**Protocol**: Catalog #344 canonical equations registry evolution (RATIFY-N) + CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE"
**Discipline**: Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (prior 29 equations preserved unchanged) + Catalog #323 canonical Provenance umbrella + Catalog #287 evidence-tag

## Registry transition: 29 → 34 entries (delta +5)

5 NEW `EVENT_REGISTERED` rows appended to `.omx/state/canonical_equations_registry.jsonl` via canonical `tac.canonical_equations.register_canonical_equation` helper (fcntl-locked JSONL append-only per Catalog #131/#138 sister discipline; canonical Provenance per Catalog #323 enforced at `CanonicalEquation.__post_init__`).

## Per-equation summary

### 1. `triple_substrate_composition_alpha_v1`

- **Summary**: `alpha_ABC = w_p * mean(rho_AB, rho_BC, rho_AC) + w_j * J(top_K(A), top_K(B), top_K(C))`
- **Generalizes**: pairwise predictors #22 (`cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1`) + #23 (sister Pearson predictor) to 3-way substrate compositions
- **IN-DOMAIN**: `triple_substrate_composition_alpha_prediction` / `stack_of_stacks_3_way_pareto_polytope` / `additive_super_additive_class_extension_to_triples`
- **EXCLUDED**: single-substrate or pairwise-only / non-additive 4+ way compositions
- **Producers**: `tac.optimization.substrate_composition_matrix` / `tools/audit_substrate_composition_matrix.py`
- **Consumers**: `tools/cathedral_autopilot_autonomous_loop.py` / `tools/predict_triple_substrate_composition_alpha_smoke.py`
- **Empirical anchor status**: design-only; awaits first triple-substrate smoke for first calibration
- **Operator-routable**: cathedral_autopilot consumer auto-discovery per Catalog #335 + sister triple-substrate cascade landing

### 2. `scorer_conditional_joint_rate_distortion_floor_v1`

- **Summary**: `S_floor = 25 * R_joint(D_seg=0, D_pose=0) / 37545489` — Shannon-canonical lower bound for ANY contest archive
- **Producer**: `tac.symposium_impls.tao_boyd_blahut_arimoto_theoretical_floor_v2` + `tac.findings_lagrangian.compute_findings_lagrangian`
- **Consumers**: `tools/cathedral_autopilot_autonomous_loop.py` / `tools/predict_theoretical_floor_smoke.py`
- **IN-DOMAIN**: `scorer_conditional_joint_rate_distortion_theoretical_floor` / `lower_bound_for_any_contest_archive_at_zero_distortion` / `shannon_canonical_floor_pareto_polytope_lower_boundary`
- **EXCLUDED**: non-zero distortion target (use `R_joint(D_seg, D_pose)` directly) / marginal seg-only or pose-only (use `categorical_blahut_arimoto_rate_distortion_v1` #16)
- **Sister equations**: #16 categorical Blahut-Arimoto / #24 cross-codec super-additive orthogonality
- **Empirical anchor status**: design-only; awaits Blahut-Arimoto joint rate-distortion numerical solver landing
- **Operator-routable**: Phase 2 numerical solver implementation enables first empirical anchor

### 3. `hnerv_class_substrate_geometry_saturation_v1`

- **Summary**: All HNeRV-derived substrates with backbone sensitivity overlap > 0.9 saturate at the same Pareto frontier point per medal-cluster geometry
- **Generalizes**: #25 (`hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1`) from specific medal cluster (PR101/PR102/PR103) to ALL HNeRV-derived substrates above the overlap threshold
- **Producer**: `tac.optimization.substrate_composition_matrix`
- **Consumers**: `tools/cathedral_autopilot_autonomous_loop.py` / `tools/audit_substrate_class_saturation.py`
- **IN-DOMAIN**: `hnerv_class_substrate_geometry_saturation_prediction` / `medal_cluster_pareto_frontier_geometry` / `pr101_pr102_pr103_medal_cluster_class`
- **EXCLUDED**: non-HNeRV class (overlap below 0.9) / categorical substrate class / predictive-coding class
- **Sister equations**: #22 cross-substrate top-K Jaccard / #25 medal cluster saturation
- **Overlap threshold canonical**: 0.9 per OVERNIGHT-FF T4 §4.3 empirical extrapolation
- **Empirical anchor status**: draft; awaits first 3 HNeRV-class substrate smoke anchors for calibration

### 4. `foveation_sidecar_bolt_on_rate_hurdle_v1`

- **Summary**: Sidecar of N bytes pays `+25*N/37545489` rate cost; must produce component savings exceeding that hurdle to improve total score (OVERNIGHT-K empirical floor)
- **Producer**: `tac.cathedral_consumers.foveation_rate_hurdle_consumer` + `tools/predict_sidecar_rate_hurdle_smoke.py`
- **Consumers**: `tools/cathedral_autopilot_autonomous_loop.py` / `tools/audit_sidecar_rate_hurdle.py`
- **IN-DOMAIN**: `foveation_sidecar_bolt_on_rate_hurdle_prediction` / `telescopic_foveation_sidecar_class` / `la_pose_sidecar_class` / `raft_ego_motion_sidecar_class` / `any_additive_sidecar_with_pre_existing_archive_base` / `overnight_k_rate_hurdle_audit_class`
- **EXCLUDED**: replacement paradigm (use #26 procedural codebook or #28 static packet custody) / Wyner-Ziv layer class (use #29 HFV2 sparse-pair sidecar) / joint rate-distortion (use #31 new joint floor)
- **Sister equations**: #7 score-marginal Lagrange multipliers / #28 static packet custody byte delta / #29 HFV2 sparse-pair sidecar replacement
- **Empirical anchor status**: draft; awaits OVERNIGHT-K audit artifact extraction for first 3 anchors
- **Operator-routable reference**: OVERNIGHT-K rate-hurdle audit per FF T4 symposium §4.3 lines 798 + 1114

### 5. `cathedral_autopilot_tier_b_score_contribution_bound_v1`

- **Summary**: `|delta_S_consumer_tier_b| <= 0.05 * |delta_S_candidate_baseline|` per Catalog #357 sister bound; prevents single-consumer dominance at ranker aggregation
- **Producer**: `tac.cathedral.consumer_contract` + `tac.cathedral_consumers`
- **Consumers**: `tools/cathedral_autopilot_autonomous_loop.py` / `tools/audit_cathedral_consumer_tier_b_contributions.py`
- **IN-DOMAIN**: `cathedral_autopilot_tier_b_per_consumer_score_contribution_bound` / `single_consumer_dominance_prevention_at_ranker_aggregation` / `tier_b_score_contributing_consumer_class_per_catalog_357`
- **EXCLUDED**: Tier A observability-only consumers (zero score contribution by construction) / Tier B aggregate across multiple consumers (no per-consumer bound) / Phase 2 Meta-Lagrangian dual-variable consumers
- **Sister equations**: #7 score-marginal Lagrange multipliers
- **Beta bound canonical**: 0.05 = 5% per Catalog #357 dual-tier architecture observability ceiling + Meta-Lagrangian Phase 1 bounded adjustment factor band [0.95, 1.05]
- **Cross-reference**: Catalog #357 dual-tier architecture
- **Empirical anchor status**: draft; awaits first Tier B consumer landing per Catalog #357 Dim 6 Step 6.5

## Sister coherence verification

**FF T4 symposium cross-reference**: lines 520-530 enumerate exactly these 5 candidate equations as "queued for operator RATIFY-N" per Catalog #344 DRAFT-only scope; this RATIFY-N landing converts the DRAFT-only queue to live registry entries via the canonical helper.

**Sister-DISJOINT scope verification**:
- Slot 2 (DP1 harvest cron `977634d6`) — DISJOINT (touches Modal harvest infrastructure, not canonical equations registry)
- 3 IN-FLIGHT Modal paid dispatches (QQ NSCS06 v8 LOW + JJ DP1 baseline + JJ DP1 procedural) — DISJOINT (touches Modal call_id ledger + harvest artifacts, not canonical equations registry)
- All sisters landed (EE-RESUME `80eca11a1` + JJ `0b496a651` + QQ `cd036aa61` + OO `6e77d37ec` + HH `32329c41b` + GG `83ed831e3` + FF `7719d4c81`) — none touched canonical equations registry

**Touched files in this landing**:
- `.omx/state/canonical_equations_registry.jsonl` (5 NEW EVENT_REGISTERED appended; APPEND-ONLY per Catalog #110/#113)
- `.omx/research/overnight_ii_ratify_n_5_candidate_canonical_equations_landed_20260521.md` (THIS landing memo)

**NOT touched** (sister-territory dirty files left for sister subagents):
- `.omx/state/modal_call_id_ledger.jsonl` (Modal harvest sister)
- `experiments/results/_modal_harvest_summary.json` (cron sister)
- `reports/cathedral_autopilot_evidence.jsonl` (autopilot sister)
- `tools/build_hfv1_sparse_sidecar_candidate.py` (HFV builder sister)

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution**: ACTIVE — equations #2 (joint R-D floor) + #4 (rate-hurdle) feed sensitivity-map consumers via canonical equation registry queries (`query_equations_by_consumer(...)`)
- **Hook #2 Pareto constraint**: ACTIVE — equation #2 (joint R-D floor) IS the canonical Pareto polytope lower-boundary constraint per CLAUDE.md "Meta-Lagrangian/Pareto solver"
- **Hook #3 bit-allocator hook**: ACTIVE — equation #4 (rate-hurdle) directly bounds sidecar bit budgets per the `25 * N / 37545489` canonical formula
- **Hook #4 cathedral autopilot dispatch hook**: ACTIVE — all 5 equations declare `tools/cathedral_autopilot_autonomous_loop.py` in `canonical_consumers`; auto-discovery per Catalog #335 will surface them on next loop tick
- **Hook #5 continual-learning posterior update**: ACTIVE — each equation declares `next_recalibration_trigger = "when_3+_new_empirical_anchors_in_domain"` so the first 3 empirical anchors per equation trigger automatic posterior refit via `auto_recalibrate_from_continual_learning_posterior`
- **Hook #6 probe-disambiguator**: N/A at RATIFY-N landing (probe-disambiguator surface fires when 2+ defensible interpretations exist for a single empirical anchor; no anchors yet for these 5 equations — future op-routable when first anchors land)

## Operator-routable downstream

1. **Auto-discovery**: cathedral_autopilot consumers per Catalog #335 will surface these 5 new equations on next loop tick via `query_equations_by_consumer("tools/cathedral_autopilot_autonomous_loop.py")` — no further wire-in needed
2. **First empirical anchors**:
   - #1 (`triple_substrate_composition_alpha_v1`): land first when a triple-substrate smoke executes
   - #2 (`scorer_conditional_joint_rate_distortion_floor_v1`): land first when Blahut-Arimoto numerical solver is implemented
   - #3 (`hnerv_class_substrate_geometry_saturation_v1`): land first when 3+ HNeRV-class substrate smokes complete (T4 § extrapolation reaches calibration threshold)
   - #4 (`foveation_sidecar_bolt_on_rate_hurdle_v1`): land first when OVERNIGHT-K rate-hurdle audit JSON artifact is parsed (extract 3 sidecar smoke anchors)
   - #5 (`cathedral_autopilot_tier_b_score_contribution_bound_v1`): land first when first Tier B consumer lands per Catalog #357 Dim 6 Step 6.5
3. **Sister consumer landings**: per `canonical_consumers` lists for each equation, sister subagents can implement the consumer modules (`tools/predict_triple_substrate_composition_alpha_smoke.py`, `tools/predict_theoretical_floor_smoke.py`, `tools/audit_substrate_class_saturation.py`, `tools/audit_sidecar_rate_hurdle.py`, `tools/audit_cathedral_consumer_tier_b_contributions.py`) without further canonical-equation work
4. **Canonical equation deprecation/refinement**: per Catalog #344 protocol, future operator-routable RATIFY events can append `EVENT_DOMAIN_REFINED` (refine IN/EXCLUDED contexts) or `EVENT_DEPRECATED` rows; the canonical registry is APPEND-ONLY so the original DRAFT-only event from this landing is preserved as the foundational provenance

## Verification

```bash
.venv/bin/python -c "
from tac.canonical_equations import query_equations, get_equation_by_id
eqs = query_equations()
print(f'Total: {len(eqs)}')  # 34
for eq_id in [
    'triple_substrate_composition_alpha_v1',
    'scorer_conditional_joint_rate_distortion_floor_v1',
    'hnerv_class_substrate_geometry_saturation_v1',
    'foveation_sidecar_bolt_on_rate_hurdle_v1',
    'cathedral_autopilot_tier_b_score_contribution_bound_v1',
]:
    eq = get_equation_by_id(eq_id)
    assert eq is not None, f'MISSING: {eq_id}'
    print(f'OK {eq_id}')
"
```

Expected output: `Total: 34` + `OK <each 5 equation_id>`

## Discipline checklist

- [x] Catalog #229 premise verification: read FF T4 symposium §4.3 BEFORE drafting equations
- [x] Catalog #287 evidence-tag: each equation Provenance carries `[predicted]` axis (PREDICTED grade per `build_provenance_for_predicted`)
- [x] Catalog #323 canonical Provenance umbrella: each equation + landing memo carries canonical Provenance
- [x] Catalog #344 RATIFY-N protocol: each equation declared via canonical helper (NOT bypass)
- [x] Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: prior 29 equations preserved unchanged; 5 NEW EVENT_REGISTERED rows append-only
- [x] Catalog #131/#138 fcntl-locked JSONL + strict-load discipline: enforced via canonical helper
- [x] Catalog #335 cathedral consumer auto-discovery: all 5 equations declare `tools/cathedral_autopilot_autonomous_loop.py` consumer
- [x] Catalog #125 6-hook wire-in: declared above; hooks 1-5 ACTIVE, hook 6 N/A (no anchors yet)
- [x] Catalog #206 subagent checkpoint discipline: 4 checkpoints emitted (steps 1-4)
- [x] Catalog #117/#157/#174 canonical commit serializer + POST-EDIT --expected-content-sha256
- [x] Catalog #230 sister-subagent ownership map: 4 dirty sister-files NOT touched
- [x] Catalog #340 sister-checkpoint guard: PROCEED (no overlap with sister checkpoint files)

## Cost + wall-clock

- $0 paid GPU (LOCAL CPU registration only; APPEND-ONLY fcntl-locked JSONL writes)
- ~30-40 min wall-clock total (most spent on equation-domain authoring + memo)

## Lane registration

- `lane_id`: `lane_overnight_ii_ratify_n_5_candidate_canonical_equations_from_ff_t4_symposium_20260521`
- Level: L1 (impl_complete + memory_entry)
