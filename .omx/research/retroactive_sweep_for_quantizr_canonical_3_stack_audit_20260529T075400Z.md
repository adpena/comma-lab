# Retroactive Sweep — Slot EE Quantizr Canonical 3-Stack Audit per Catalog #348

**Date**: 2026-05-29T07:54:00Z
**Lane**: `lane_slot_ee_quantizr_catalog_325_audit_20260529`
**Trigger**: Slot EE landing of canonical equation `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1` + canonical anti-pattern `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1`

## Catalog #348 4-field contract

### 1. Bug-class symptom signature

Substrate trainer at `experiments/train_substrate_*.py` scaffolds training on contest SegNet/PoseNet scorers WITHOUT all 3 Quantizr-canonical primitives:

- EMA decay 0.997 weight averaging (per CLAUDE.md "EMA - NON-NEGOTIABLE")
- KL distill T=2.0 SegNet temperature distillation (per CLAUDE.md "Quantizr intelligence")
- eval_roundtrip=True in training inner loop (per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE")

**Symptom**: substrate trainer landing memo claims canonical Quantizr-paradigm alignment but training run produces score >0.20 due to:

- Single-epoch noise dominating final checkpoint without EMA shadow stabilization (Council D anchor)
- SegNet distortion drift without KL T=2.0 SegNet distill warmup (Quantizr 0.33 anchor)
- Proxy-auth gap 2-11x on PoseNet without eval_roundtrip baked into inner loop (PR #95 Finding A anchor)

### 2. Pre-fix window

Bug class active across the entire substrate-trainer corpus prior to Slot CC T3 grand-council strategic-reprioritization Quantizr binding revision #5 (2026-05-29T07:30Z) + Slot EE audit landing (2026-05-29T07:54Z):

- Audit covers all 100 substrate trainers at `experiments/train_substrate_*.py` as of 2026-05-29T07:54Z
- Bug class spans substrate landings 2026-04-26 (earliest substrate trainer) through 2026-05-29 (latest substrate trainer)
- 99 of 100 substrate trainers carry the bug class at varying severity (1 primitive missing through 3 primitives missing)

### 3. Historical KILL/DEFER/FALSIFY search results

Per Catalog #307 paradigm-vs-implementation classification + CLAUDE.md "Forbidden premature KILL without research exhaustion":

**No KILL verdicts retroactively re-tagged**: the bug class is IMPLEMENTATION-LEVEL (missing primitive wiring) NOT PARADIGM-LEVEL (substrate-paradigm-falsification). Per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable: KILL conversion requires (a) every plausible alternative config attempted empirically, (b) exact custody/recomputation/failure classification, (c) grand council CONSENSUS, (d) reactivation criteria documented. None of these apply at the IMPLEMENTATION-LEVEL canonical-primitive-missing scope.

**No DEFER verdicts retroactively re-tagged**: substrate trainers with missing primitives remain DEFERRED-pending-Catalog-325-symposium-canonical-vs-fork-decision per Catalog #290 not retroactive falsification.

**No FALSIFY verdicts retroactively re-tagged**: the canonical equation `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1` registers with NO empirical anchors (DEFERRED per "iterate not force"); no retroactive falsification is possible because no empirical-vs-predicted contradiction has been measured.

**Sister gate canonical equations preserved**:

- `ema_decay_substrate_stage_aware_v1` (sister at per-stage EMA decay surface): 0 anchors at registration; canonical equation predicts decay = 1 - 1/(target_window_fraction * total_steps) recovers Quantizr 0.997 at 1666 steps. No retroactive falsification.
- `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` (sister at per-T2-temperature surface): 17 anchors at registration; preserved as canonical PR-95-parity prior. No retroactive falsification.
- `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` (sister at QAT composition surface): 3 anchors; canonical equation predicts P2 distillation CATALYZES P4 QAT savings via 3-position triangular pattern. No retroactive falsification.

### 4. Per-finding RE-EVAL-priority assignment

| Finding affected by canonical 3-stack audit | RE-EVAL priority | Reactivation criteria |
|---|---|---|
| 99/100 substrate trainers lacking full 3-stack | LOW priority bulk re-eval (per-Catalog-325-symposium) | Slot CC Class A wavelet + Class D Wyner-Ziv Catalog #325 symposia first; iterate per "iterate not force" |
| `train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py` (EMA only; missing KL + eval_roundtrip) | HIGH priority | CLAUDE.md non-negotiable eval_roundtrip MUST be honored; backfill or research_only opt-out required; Catalog #5 violation if dispatched |
| `train_substrate_pr101_with_dp1_prior_regularizer.py` (none of 3-stack) | HIGHEST priority | Same; bare PR101 sister substrate; backfill or research_only opt-out required immediately |
| `train_substrate_pact_nerv_distilled_scorer.py` (full 3-stack canonical) | NONE | Reference exemplar; preserve canonical-3-stack-canonical-exemplar status |
| Class A wavelet 3 substrates (EMA + ER; missing KL T=2.0) | MEDIUM (gates Slot CC scope-lock) | Per Catalog #325 symposium ADOPT vs FORK per Catalog #290 |
| Class D Wyner-Ziv 2 substrates (EMA + ER; missing KL T=2.0) | MEDIUM (gates Slot CC scope-lock) | Per Catalog #325 symposium ADOPT vs FORK per Catalog #290 |

**Operator-routable bulk re-eval**: per "iterate not force" + Slot CC STRATEGIC RESET #1 cap=1 apparatus_maintenance-per-turn for ONE WEEK, the 99/100 substrate-trainer 3-stack gap is NOT a bulk-fix priority. The gap surfaces via this audit memo + canonical anti-pattern; per-substrate fixes are routed through per-substrate Catalog #325 6-step symposium content per Slot CC STRATEGIC RESET #2.

## Cross-references

- Canonical equation `quantizr_canonical_ema_kl_eval_roundtrip_stack_savings_v1` (this audit's canonical-equation-candidate; landed)
- Canonical anti-pattern `substrate_trainer_missing_quantizr_canonical_3_stack_ema_kl_eval_roundtrip_v1` (this audit's canonical-anti-pattern-candidate; landed)
- Council anchor `slot_ee_quantizr_catalog_325_audit_ema_kl_eval_roundtrip_20260529` (canonical posterior anchor; landed)
- Design memo `.omx/research/quantizr_canonical_3_stack_audit_design_20260529.md`
- Landing memo `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_ee_quantizr_catalog_325_audit_ema_kl_eval_roundtrip_per_slot_cc_dissent_landed_20260529.md`
- CLAUDE.md amendment proposal `.omx/research/claude_md_quantizr_canonical_3_stack_amendment_proposal_20260529.md`
- Sister canonical equations `ema_decay_substrate_stage_aware_v1` + `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` + `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1`

## Lane

`lane_slot_ee_quantizr_catalog_325_audit_20260529` L1
