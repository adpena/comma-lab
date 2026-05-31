# Retroactive sweep — Wave N+31 DIAMOND-HUNT historical DEFER/orphan reactivation audit

**Generated**: 2026-05-31T00:30Z UTC per Catalog #348 retroactive-sweep-for-new-gate discipline
**Lane**: `lane_wave_n31_diamond_hunt_historical_defer_orphan_reactivation_audit_20260530` L1 (research_only=true)
**Parent audit memo**: `.omx/research/wave_n31_diamond_hunt_historical_defer_orphan_reactivation_audit_20260530.md`

## 4-field contract per Catalog #348

### Field 1 — Bug-class symptom signature

The DIAMOND-HUNT audit is a READ-ONLY audit, NOT a NEW STRICT preflight gate landing. The companion bug class it surfaces is:

**Bug class**: "deferred-item-with-paradigm-INTACT-and-reactivation-criterion-MET-by-wave-window-sister-landings sits invisible to the next dispatcher because the canonical posterior is not auto-queried for paradigm-intact + sister-satisfied reactivation paths".

This is structurally complementary to the canonical feeder-audit bug class (which honestly extincts FAKE-pickup of unmet reactivation criteria). DIAMOND-HUNT identifies the HIGHER-LEVERAGE complement: cases where the reactivation criterion IS met via TODAY's sister landings but no dispatcher consumes the signal.

### Field 2 — Pre-fix window

The pre-fix window is the entire period before the canonical DIAMOND-HUNT audit was registered as a recurring discipline (i.e. before 2026-05-31). Empirical receipts from the audit:

- 5 Cable C6 RE-EVAL-HIGH DRAFTs at `.omx/research/council_t3_*_re_eval_high_symposium_DRAFT_20260519T060557Z.md` LANDED 2026-05-19 but never CONVOCATED → 12 days invisible to dispatcher
- C6.2 `lane_stc_clean_source` reactivation criterion: requires canonical Filler-STC implementation → LANDED TODAY at `396488202` (Fridrich-school extension); MET but no dispatcher had this in its work queue
- TOP-1 `pr110_opt7_via_yousfi_t1` 4-canonical-helper trainer wire-in: LANDED TODAY at `86e3f4c38`; reactivation criterion MET but recipe `dispatch_enabled:false` still requires operator flip
- TOP-8 `c6_e4_mdl_ibps` reactivation path #1: requires DreamerV3 RSSM math-fidelity audit LANDED yesterday (`feedback_wave_3_dreamerv3_rssm_math_audit_landed_20260529`); MET but no dispatcher consumed the path

### Field 3 — Historical-KILL/DEFER/FALSIFY search results

For each TOP-10 DIAMOND_HIGH candidate, the audit confirms the historical KILL/DEFER/FALSIFY verdict + the reactivation criterion + today's sister-landing-induced criterion-MET status:

| Candidate | Historical verdict | Date | Reactivation criterion | Today's sister landing | MET? |
|-----------|-------------------|------|------------------------|------------------------|------|
| TOP-1 pr110_opt7_via_yousfi_t1 | DEFER 2026-05-30 | 2026-05-30 | Trainer wires 4 canonical helpers | `86e3f4c38` trainer wire-in LANDED | **YES** |
| TOP-2 slot_yy_hill_canonical | DEFER 2026-05-29 | 2026-05-29 | paired CUDA + paired CPU per Catalog #246 + canonical HILL helper | `396488202` Fridrich-school landing includes HILL adjacent canonical helpers | **PARTIAL** (helper landed; paired-CUDA pending) |
| TOP-3 slot_ccc_hugo_canonical | DEFER 2026-05-29 | 2026-05-29 | paired CUDA per Catalog #246 + canonical orthogonality probe vs UNIWARD | `396488202` Fridrich-school landing includes canonical HUGO + STC + UNIWARD patterns | **PARTIAL** (helper landed; paired-CUDA pending) |
| TOP-4 slot_ff_pr110_opt_7 | DEFER 2026-05-29 | 2026-05-29 | paired-CUDA + paired-CPU per Catalog #246 1:1 contest-compliant | Sister `1230b3b9c` PR110-OPT-7 L1 promotion + `86e3f4c38` wire-in | **YES** |
| TOP-5 lane_stc_clean_source | FALSIFIED 2026-04-29 → UNDETERMINED | 2026-04-29 | MPS-PROXY taint extinction + ≥3 alternative reducers + canonical STC implementation | All 3 satisfied: Catalog #1+#127+#192 LANDED + Catalog #308 LANDED + `396488202` canonical Filler-STC LANDED | **YES** |
| TOP-6 lane_mae_v + lane_saug | DEFER-pending-operational 2026-04-28 | 2026-04-28 | Vast.ai DNS bug fixed + per-substrate symposium | Operational fix complete + Cable C6 DRAFT exists | **PARTIAL** (DRAFT pending convocation) |
| TOP-7 lane_pr106_05_06_REFORMULATED | FALSIFIED-as-non-applicable 2026-05-04 | 2026-05-04 | per-substrate symposium + reformulation for HNeRV latent | Cable C6 DRAFT + sister NSCS06 v8 chroma_lut Wave 9 LANDED today | **PARTIAL** (DRAFT pending; reformulation specific to HNeRV latent stream NOT yet landed) |
| TOP-8 c6_e4_mdl_ibps (DreamerV3 path B2) | DEFER 2026-05-19 | 2026-05-19 | DreamerV3 RSSM categorical posterior smoke (path 1 of 4) | `feedback_wave_3_dreamerv3_rssm_math_audit_landed_20260529.md` LANDED | **YES** (math-fidelity audit; smoke pending) |
| TOP-9 lane_17_imp | KILL→WITHDRAWN 2026-04-30 | 2026-04-30 | per-substrate symposium + proper train_distill fine-tune | Cable C6 DRAFT exists; train_distill infrastructure exists | **PARTIAL** (DRAFT pending convocation; paid dispatch needed) |
| TOP-10 slot_tt_pr110_opt_5 | DEFER 2026-05-29 | 2026-05-29 | paired-CUDA + per-substrate empirical verification | Sister to TOP-1/TOP-4 PR110-OPT-7 family | **PARTIAL** |

### Field 4 — Per-finding RE-EVAL-priority assignment

Per Catalog #348 retroactive sweep contract:

| Candidate | RE-EVAL-priority | Operator-routable action |
|-----------|-----------------|--------------------------|
| TOP-1 pr110_opt7_via_yousfi_t1 | **RE-EVAL-HIGH-NOW** | Flip recipe `dispatch_enabled:true` + smoke-before-full per Catalog #167 + paired CUDA+CPU per Catalog #246 ($0.30 envelope) |
| TOP-2 slot_yy_hill_canonical | **RE-EVAL-HIGH-NOW** | Per-substrate symposium per Catalog #325 (cheap MLX-LOCAL prior smoke) + paired CUDA per Catalog #246 ($0.06 envelope) |
| TOP-3 slot_ccc_hugo_canonical | **RE-EVAL-HIGH-NOW** | Per-substrate symposium per Catalog #325 + canonical Cauchy-Schwarz orthogonality probe + paired CUDA ($0.06 envelope) |
| TOP-4 slot_ff_pr110_opt_7 | **RE-EVAL-HIGH-NOW** | Sister of TOP-1; can compose paired smoke window ($0.30 envelope) |
| TOP-5 lane_stc_clean_source | **RE-EVAL-HIGH-NOW** | Convocate Cable C6 T3 DRAFT to binding T3 council deliberation; use canonical Filler-STC at `src/tac/composition/fridrich_school_inverse_steganalysis_patterns/canonical_syndrome_trellis_coding_filler.py`; per-substrate symposium + STC-as-DELTA/SIDECAR over Selfcomp baseline path ($0.20 envelope) |
| TOP-6 lane_mae_v + lane_saug | **RE-EVAL-MEDIUM-NEXT-CAP-WINDOW** | Convocate Cable C6 T3 DRAFT + MLX-LOCAL prior smoke + per-substrate symposium per Catalog #325 + paid dispatch $10-25 envelope |
| TOP-7 lane_pr106_05_06_REFORMULATED | **RE-EVAL-MEDIUM-NEXT-CAP-WINDOW** | Convocate Cable C6 T3 DRAFT + per-substrate symposium + UNIWARD-delta + grayscale-LUT reformulation for HNeRV latent stream + latent codebook ($5-15 envelope) |
| TOP-8 c6_e4_mdl_ibps (DreamerV3 path B2) | **RE-EVAL-MEDIUM-NEXT-CAP-WINDOW** | DreamerV3 RSSM categorical posterior smoke per reactivation path #1 ($5-15 envelope) |
| TOP-9 lane_17_imp | **RE-EVAL-MEDIUM-NEXT-CAP-WINDOW** | Convocate Cable C6 T3 DRAFT + per-substrate symposium + L40S train_distill fine-tune ($5-15 envelope) |
| TOP-10 slot_tt_pr110_opt_5 | **RE-EVAL-MEDIUM-NEXT-CAP-WINDOW** | Per-substrate symposium + paired CUDA per Catalog #246 ($0.06-0.30 envelope) |

## Historical findings invalidated by THIS audit

**ZERO** historical findings invalidated. DIAMOND-HUNT is COMPLEMENTARY to the canonical feeder audit (which honestly extincts FAKE-pickup); DIAMOND-HUNT adds HIGHER-LEVERAGE signal without contradicting any prior verdict. All canonical posterior anchors remain valid.

## Cross-references

- Parent audit memo: `.omx/research/wave_n31_diamond_hunt_historical_defer_orphan_reactivation_audit_20260530.md`
- Sister feeder audit: `deferred_items_feeder_audit_post_recovery_wave_20260530.md`
- Pre-rigor inventory: `.omx/state/pre_rigor_symposium_inventory_20260518T030340Z.json`
- Resurrection audit foundation: `resurrection_audit_20260516.md`
- Cable C6 DRAFTs synthesis: `cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z.md`
- Catalog #348 retroactive sweep canonical pattern: `tools/retroactive_sweep_for_*.md` + sister gates

Generated 2026-05-31T00:30Z UTC.
