# SPDX-License-Identifier: MIT
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- # FORMALIZATION_PENDING:operator_facing_command_sheet_companion_to_t3_council_pr110_stacking_pivot_ordering_landed_20260526_no_new_canonical_equation_needed -->

# T3 Council PR110 stacking pivot ordering — OPERATOR-FACING COMMAND SHEET

**Companion to**: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
**Operator decisions queued**: D-WINNER-1, D-WINNER-2, D-DP1-REACTIVATION, D-ATW-V2-REMOVAL, D-MATRIX-EXTENSION (see source memo §10)
**Per CLAUDE.md "Executing actions with care"**: ALL commands are operator-routable; NO paid dispatch fires from this subagent's session.
**Per Catalog #199 paired-env discipline**: every paid Modal dispatch via `tools/operator_authorize.py` requires `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=<float>`.

---

## WINNER #1: NSCS06 v8 chroma_lut first paid 4-arm paired auth_eval

### Step 0 (PRE-DISPATCH; ~30 min $0): cls_stream wire-in landing per Yousfi REVISION 2

```bash
# Sister subagent spawn (operator decision; NOT invoked by THIS subagent)
# Scope: wire cls_stream consumption at L0 inflate per path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md L1 promotion blocker
# Expected output: ~30 LOC inflate.py extension + 5-10 tests; $0 free-CPU smoke validation
# Reference: .omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md
```

### Step 1 (PRE-DISPATCH; ~30 min $0): alternative-reducer enumeration per Catalog #308 + Carmack REVISION 1

Sister subagent enumerates 3-4 alternative reducer arms for NSCS06 v8 PRE-dispatch (so reactivation criteria are READY if first paid anchor regresses):

1. **per-temporal-window LUT** (different chroma medians per video segment)
2. **per-spatial-region LUT** (different chroma medians per image region)
3. **hybrid-with-residual-overlay** (per-(level,class) median + per-pair residual byte stream)
4. **cls_stream-conditioned LUT** (chroma LUT modulated by segnet class-stream at inflate time)

### Step 2 (PAID DISPATCH; ~$0.50-1.00 Modal): canonical 4-arm paired auth_eval per Catalog #246

```bash
# Operator runs after Step 0 + Step 1 land:
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00

# Step 2a: baseline_cpu (Modal Linux x86_64 CPU; ~$0.10-0.25)
.venv/bin/python tools/operator_authorize.py \
    --recipe .omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_baseline_modal_cpu_dispatch.yaml \
    --pair-group-id nscs06_v8_first_paired_4arm_20260526 \
    --expected-arm baseline_cpu

# Step 2b: baseline_cuda (Modal Linux x86_64 T4; ~$0.10-0.25)
.venv/bin/python tools/operator_authorize.py \
    --recipe .omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_baseline_modal_t4_dispatch.yaml \
    --pair-group-id nscs06_v8_first_paired_4arm_20260526 \
    --expected-arm baseline_cuda

# Step 2c: procedural_cpu (Modal Linux x86_64 CPU; ~$0.10-0.25)
.venv/bin/python tools/operator_authorize.py \
    --recipe .omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_procedural_modal_cpu_dispatch.yaml \
    --pair-group-id nscs06_v8_first_paired_4arm_20260526 \
    --expected-arm procedural_cpu

# Step 2d: procedural_cuda (Modal Linux x86_64 T4; ~$0.10-0.25)
.venv/bin/python tools/operator_authorize.py \
    --recipe .omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_procedural_modal_t4_dispatch.yaml \
    --pair-group-id nscs06_v8_first_paired_4arm_20260526 \
    --expected-arm procedural_cuda

# Step 3: HARVEST (per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable; canonical helper)
.venv/bin/python tools/parallel_harvest_actuator.py \
    --pair-group-id nscs06_v8_first_paired_4arm_20260526 \
    --max-wait-seconds 7200
```

### Step 4 (POST-DISPATCH): canonical equation #26 anchor registration per Catalog #344

```bash
# Operator runs after Step 3 harvest completes (rc=0 across all 4 arms):
.venv/bin/python -c "
from tac.canonical_equations import update_equation_with_empirical_anchor
# Register IN-DOMAIN nscs06_v8_chroma_lut anchor per OVERNIGHT-YY DP1 sister pattern
update_equation_with_empirical_anchor(
    equation_id='procedural_codebook_from_seed_compression_savings_v1',
    anchor_id='nscs06_v8_chroma_lut_4arm_paired_auth_eval_first_paid_contest_axis_anchor_20260526',
    in_domain_context='nscs06_v8_chroma_lut',
    predicted_delta_s=-0.0027063,
    empirical_delta_s_axis_cuda=<TBD from Step 3>,
    empirical_delta_s_axis_cpu=<TBD from Step 3>,
    # ...
)
"
```

### Step 5 (POST-DISPATCH): Catalog #313 probe-outcome registration

```bash
# Register probe-outcome per OVERNIGHT-YY DP1 sister pattern
.venv/bin/python -c "
from tac.probe_outcomes_ledger import register_probe_outcome
register_probe_outcome(
    probe_id='nscs06_v8_chroma_lut_4arm_paired_auth_eval_first_paid_contest_axis_anchor_20260526',
    substrate='nscs06_v8_chroma_lut',
    probe_kind='paired_cpu_cuda_auth_eval_4arm_full_axis_canonical_equation_26_nscs06_v8_chroma_lut_replacement_savings_prediction_test',
    verdict=<'PROCEED' if empirical converges; 'KILL/DEFER' if regression >+0.5>,
    blocker_status=<'blocking' if regression; 'advisory' if converges>,
    metric_value=<empirical_delta_s>,
)
"
```

---

## WINNER #2: grayscale_lut procedural variant first paid 4-arm paired auth_eval

**Contingent on WINNER #1 result (per Carmack REVISION 1 SERIAL recommendation; Hassabis REVISION 3 paradigm-class-class caveat)**:
- IF WINNER #1 converges (empirical_delta_s within ±0.005 of predicted -0.002706): authorize WINNER #2 with same 4-arm paired discipline; expected predicted ΔS -0.000149.
- IF WINNER #1 regresses (full-axis +0.5 or more): SPAWN sister T3 supplemental symposium per Catalog #310 TRIGGER A BEFORE WINNER #2 dispatch; explicit Round 2 self-reflection on chroma-LUT-paradigm-class CARGO-CULTED assumption.

```bash
# Same command structure as WINNER #1 Step 2; recipes:
# substrate_grayscale_lut_baseline_modal_{cpu,t4}_dispatch.yaml
# substrate_grayscale_lut_procedural_modal_{cpu,t4}_dispatch.yaml
# --pair-group-id grayscale_lut_first_paired_4arm_20260526_after_winner_1
```

---

## WINNER #3: VQ-VAE indices_blob first paid 4-arm paired auth_eval (sister equation #359)

**Contingent on WINNER #1 + WINNER #2 results (Hassabis REVISION 3 paradigm-class-interleaving)**:
- DIFFERENT paradigm class (RESIDUAL-HYBRID vs REPLACEMENT IN-DOMAIN); failure mode structurally distinct.
- First paid empirical anchor for sister equation `procedural_predictor_plus_residual_correction_savings_v1`; per Catalog #344 the anchor registration extends the sister equation's domain-of-validity empirical surface.

```bash
# Recipes:
# substrate_vq_vae_baseline_modal_{cpu,t4}_dispatch.yaml
# substrate_vq_vae_indices_blob_procedural_modal_{cpu,t4}_dispatch.yaml
# --pair-group-id vq_vae_indices_blob_first_paired_4arm_20260526_after_winners_1_2
```

---

## WINNER #4: ATW V2 cdf_table_blob REMOVAL paradigm Phase 3 BUILD

**Operator-routable as sister-substrate-class candidate (NOT direct fec6 stacking)**:
- $0 free-CPU local smoke first per Carmack MVP-first phasing
- Grammar-layer change: update `parse_atw2_archive_bytes()` + `pack_archive()` + `inflate.py` to skip the cdf_table_blob section
- Predicted ΔS = -0.001705 (sister-substrate class; NOT direct fec6 stacking; operator-routable to ATW V2 second-wave submission)

```bash
# Sister subagent spawn (operator decision; NOT invoked by THIS subagent)
# Scope: ATW V2 REMOVAL paradigm Phase 3 BUILD per d588d6aec OVERNIGHT-I + atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md §4
# Expected output: grammar-layer change + parser-safe-extension precedent verification + sister-canonical 2,560-byte savings
```

---

## WINNER #5: DP1 — DEFERRED-pending-reactivation (NOT killed)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + OVERNIGHT-YY HIGH verdict registration:

**Reactivation criteria per Catalog #308 alternative-reducer enumeration** (sister subagent spawn within 7 days per D-DP1-REACTIVATION recommendation B):

1. Joint retraining of dp1 decoder WITH procedural seed substitution DURING training (not post-training byte replacement)
2. Score-opacity audit of dp1 codebook bytes via byte-mutation smoke per Catalog #139 / #272 distinguishing-feature contract
3. At least 3 alternative codebook substitution strategies attempted per Catalog #308
4. Promotable axis validation via PROCEED-unconditional council per Catalog #315 after alternative reducer works

**Catalog #313 probe-outcome PROCEED-blocking** with 30-day expiry **2026-06-20**. After expiry, sister T3 deliberation re-evaluates whether to extend, archive, or reactivate.

---

## Sister supplemental T3 symposium spawn conditions (Catalog #310 recursive self-reflection)

**TRIGGER A**: WINNER #1 NSCS06 v8 paired anchor regresses full-axis >+0.5 (similar pattern as DP1's +86.08).
**TRIGGER B**: L1-PROMOTION-CASCADE sister produces sub-PR110 candidate (rebaseline pivot ordering).
**TRIGGER C**: Operator decides to defer NSCS06 v8 in favor of grayscale_lut or VQ-VAE as #1.

For each trigger, spawn sister subagent prompt template:

```text
You are a T3 grand council supplemental symposium subagent. Predecessor:
.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md
(Round 1 verdict PROCEED_WITH_REVISIONS; TRIGGER <A/B/C> fired).

Per Catalog #310 council recursive self-reflection protocol Round 2:
explicitly re-evaluate the assumption surfaced as material-unverified at
Round 1 landing. Spawn 4-tier council per Catalog #346 canonical roster
+ Catalog #292 per-deliberation assumption surfacing + Catalog #300 v2
frontmatter. Verdict either ratifies Round 1 ordering OR pivots to new
ordering with binding empirical-anchor citation.

Memory: .omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526_round_2_supplemental_landed_<UTC>.md
```

---

## Cost envelope summary

| WINNER | Paid Modal | Free $0 work | Wall-clock |
|---|---:|---|---|
| #1 NSCS06 v8 chroma_lut | $0.50-1.00 | ~60 min (cls_stream + alt-reducer enum + canonical posterior + Catalog #313 + #344 registration) | ~2-4 hours total |
| #2 grayscale_lut | $0.50-1.00 (conditional) | ~30 min | ~1-2 hours (after #1) |
| #3 VQ-VAE indices_blob | $0.50-1.00 (conditional) | ~30 min | ~1-2 hours (after #2) |
| #4 ATW V2 REMOVAL paradigm | $0.50-1.50 (Phase 3 BUILD) | ~60 min (grammar-layer change) | ~2-4 hours (sister-substrate class) |
| #5 DP1 reactivation | $0.50-2.00 (alt-reducer arms) | ~120 min (joint-retraining + score-opacity audit) | ~4-8 hours (7-day-window operator-routable) |
| **TOTAL (full cascade)** | **~$2.50-6.50** | **~5 hours** | **~10-20 hours** wall-clock |

Per CLAUDE.md "Long-burn score-lowering campaign default" — explicit budget envelope; cost-band acceptable per the 5-substrate canonical hypothesis predicted aggregate ΔS = -0.013.

— end of operator command sheet —
