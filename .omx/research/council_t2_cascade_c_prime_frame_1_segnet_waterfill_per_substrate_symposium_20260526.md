---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Atick
  - Redlich
  - Tishby
  - Wyner
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: |
      synthesis prediction -0.058820 is 10-30× literature overestimation common pre-empirical.
      paired-CUDA validation MUST run before any promotion. abstain on score magnitude;
      PROCEED on substrate scaffold + paired-CUDA validation gate.
  - member: Atick
    verbatim: |
      asymmetric channel theory applies BUT the synthesis frame-1 menu (8 modes) is arbitrary;
      production trainer SHOULD probe K_frame_1 ∈ {4, 8, 12, 16} to verify Pareto-optimal menu size.
council_assumption_adversary_verdict:
  - assumption: "Atick-Redlich asymmetric channel applies to SegNet's x[:,-1,...] slice"
    classification: HARD-EARNED
    rationale: |
      derived from upstream/modules.py SegNet preprocess_input + verified empirically via
      Cascade C' synthesis 25.2% frame-1 routing matches sister #1324 PoseNet-null 22.3% within 3pp
  - assumption: "Frame-1 modes extract ≥5× per-pair PoseNet savings vs frame-0"
    classification: CARGO-CULTED
    rationale: |
      synthesis prior; literature 10-30× overestimation common; requires paired-CUDA validation
      per Catalog #324 post-training Tier-C re-measurement on landed paired smoke archive sha
  - assumption: "Per-pair Lagrangian dual converges in single argmin pass"
    classification: HARD-EARNED
    rationale: |
      tac.findings_lagrangian Phase 1-3 wire-in (Catalog #355) verified via pytest +
      Cascade C' synthesis convergence_status=single_pass_argmin_no_iteration_required
  - assumption: "1-bit-per-pair sidecar compresses to ≤80 bytes via brotli for 600 pairs"
    classification: HARD-EARNED
    rationale: |
      Cascade C' synthesis empirical: 79 bytes (Option B). Sister synthesis script
      replicates deterministically at any seed; archive scaffold test_byte_mutation_smoke
      PASS confirms structural invariants.
  - assumption: "K=24 codebook expansion adds ~30-50 bytes overhead (Option A)"
    classification: CARGO-CULTED
    rationale: |
      synthesis estimate; not validated against actual PR110 Huffman codebook
      expansion. Operator-routable: validate via build_pr101_frame_exploit_selector_packet_markov.py
      with K=24 + diff against PR110 K=16 frontier sha 6bae0201.
council_decisions_recorded:
  - "op-routable #1: 7th-order subagent builds MLX-first trainer (sister to NSCS06 v8 mlx_iteration.py 1089 LOC)"
  - "op-routable #2: 7th-order subagent builds inflate.sh 3-arg signature wrapper (sister to NSCS06 v8 driver template)"
  - "op-routable #3: paired-CUDA Modal T4 smoke + paired-CPU per CLAUDE.md 'Submission auth eval' (PAID; operator-decision-required)"
  - "op-routable #4: post-training Tier-C density re-measurement per Catalog #324 on landed paired smoke archive sha"
  - "op-routable #5: canonical equation #344 anchor registration via tools/register_atick_redlich_asymmetric_scorer_channel_canonical_equation_20260526.py (sister pattern commits 7ab5f58ae + 04f34ea40)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: 2026-06-25T00:00:00Z
deferred_substrate_id: cascade_c_prime_frame_1_segnet_waterfill
related_deliberation_ids:
  - cascade_c_prime_parent_synthesis_landing_20260526
  - cascade_c_prime_paired_cuda_validation_deferred_pending_substrate_scaffold_20260526
  - sister_1324_pose_null_classification_20260526
---

# T2 per-substrate symposium — cascade_c_prime_frame_1_segnet_waterfill (Atick-Redlich asymmetric scorer channel)

- **substrate_id**: `cascade_c_prime_frame_1_segnet_waterfill`
- **lane_id**: `lane_cascade_c_prime_option_a_build_scaffold_20260526`
- **date_utc**: 2026-05-26T20:56:00Z
- **tier**: T2 (per Catalog #300 — substrate dispatch eligibility per Catalog #325)
- **window_start_utc**: 2026-05-26T00:00:00Z
- **window_end_utc**: 2026-06-09T00:00:00Z (14-day per Catalog #325)
- **verdict**: PROCEED_WITH_REVISIONS

## Topic

Should the cascade_c_prime_frame_1_segnet_waterfill substrate scaffold be promoted from L0 SCAFFOLD to L1 INTEGRATION (pre-paired-CUDA validation)? The substrate implements Atick-Redlich asymmetric scorer channel theory (1990 cooperative-receiver) via per-pair Lagrangian dual routing decision over a joint frame-0 + frame-1 mode menu. Cascade C' synthesis predicts -0.058820 score delta at PR106 frontier; even at 10× literature overestimate, -0.006 remains PR111-PLAUSIBLE.

## Canonical 6-step contract per Catalog #325

### 1. Cargo-cult audit per Catalog #303

See `## Cargo-cult audit per assumption` table in the substrate's `__init__.py`. Summary:

- 3 HARD-EARNED assumptions (Atick-Redlich theory + Lagrangian dual convergence + brotli sidecar empirical)
- 2 CARGO-CULTED assumptions (synthesis -0.058820 score magnitude + K=24 codebook overhead)

### 2. 9-dimension success checklist evidence per Catalog #294

See `## 9-dimension success checklist evidence` table in `__init__.py`. All 9 dimensions documented; Dim 9 OPTIMAL-MINIMAL-CONTEST-SCORE is the PROVISIONAL-PENDING-VERIFICATION dimension per Catalog #363 recursive self-reflection protocol.

### 3. Observability surface declaration per Catalog #305

See `## Observability surface` table in `__init__.py`. All 6 facets declared (inspectable per layer / decomposable per signal / diff-able / queryable / cite-able / counterfactual-able).

### 4. Sextet pact deliberation + grand council attendees

- **Inner sextet** (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary): PROCEED with revisions per dissent block
- **Inner 4-co-lead structure** (Shannon + Dykstra + Rudin + Daubechies) per CLAUDE.md "Council conduct amendment 2026-05-19": all 4 present
- **Grand council topical attendees**: Atick + Redlich (Atick-Redlich cooperative-receiver canonical 1990 paper) + Tishby (Tishby-Zaslavsky 2015 IB framework Atick-Redlich-Tishby sister) + Wyner (Wyner-Ziv 1976 source coding with side information — routing-decision sidecar IS side info)
- **Roster validation per Catalog #346**: `tac.canonical_council_roster.validate_council_dispatch_roster` complete=True (12 attendees; sextet + 4-co-lead + Atick-Redlich-Tishby-Wyner topical specialists)

### 5. Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"

5 reactivation paths per recipe `reactivation_criteria` block:

1. Per-substrate symposium PROCEED verdict landed (THIS memo) within 14-day window
2. MLX-first trainer lands at `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py`
3. inflate.sh lands at `scripts/inflate_cascade_c_prime_frame_1_segnet_waterfill.sh` (3-arg signature per Catalog #146)
4. Canonical equation #344 anchor lands at `.omx/state/canonical_equations_registry.jsonl` after paired-CUDA validation
5. Tier-C density post-training validation per Catalog #324 confirms within-class or across-class classification

### 6. Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status: pending_post_training` declared in recipe + substrate contract. Reactivation criterion: post-training Tier-C re-measurement on landed paired smoke archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c`.

## Verdict justification

**PROCEED_WITH_REVISIONS** (12-of-12 quorum; 0 REFUSE; 2 dissent with substantive revisions):

- The Atick-Redlich asymmetric channel paradigm is HARD-EARNED (Atick-Redlich 1990 cooperative-receiver theorem + empirical sister #1324 cross-validation within 3pp)
- The synthesis -0.058820 score magnitude is CARGO-CULTED (10-30× literature overestimation common pre-empirical) — Contrarian + Atick dissent BOTH flagged this; verdict requires paired-CUDA validation BEFORE promotion
- Substrate scaffold (substrate contract + architecture + archive + inflate + tests) all land structurally valid; 13/13 pytest tests PASS including Catalog #139 byte-mutation smoke
- MLX-first trainer + inflate.sh + paired-CUDA Modal dispatch deferred to 7th-order operator-routable iteration per credit-cap awareness + CLAUDE.md "Executing actions with care" (PAID dispatch operator-decision-required)

## Revisions binding for PROCEED-unconditional

1. **MLX-first trainer landing** at `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py` (~700-1000 LOC sister to NSCS06 v8 `mlx_iteration.py`)
2. **inflate.sh landing** at `scripts/inflate_cascade_c_prime_frame_1_segnet_waterfill.sh` 3-arg signature wrapper (~50 LOC sister to NSCS06 v8 driver template)
3. **Paired-CUDA Modal T4 smoke** per Catalog #167 smoke-before-full pattern + Catalog #246 paired-dispatch helper — operator-decision-required (PAID; ~$0.30-0.50)
4. **Paired-CPU Linux x86_64 anchor** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
5. **Canonical equation #344 anchor registration** at `.omx/state/canonical_equations_registry.jsonl` for `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` after paired-CUDA empirical lands; per Catalog #344 sister registration pattern

After all 5 revisions land + paired-CUDA validation produces empirical anchor: re-convene this symposium for PROCEED-unconditional verdict.

## Cross-references

- Cascade C' parent synthesis landing memo (commit `2d5337f27`; subagent `aa563bbb31adadfd6`)
- Cascade C' Modal validation DEFERRED-pending-substrate-scaffold verdict (commit `aa1a9cf32`; subagent `a1d16a40f4a722e26`)
- Sister #1324 Cascade C P19 PoseNet-null bucket classification (cross-validation reference)
- Pre-execution gate report: `.omx/research/cascade_c_prime_option_a_build_scaffold_pre_execution_gate_report_20260526.md`
- Substrate package: `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/`
- Operator-authorize recipe: `.omx/operator_authorize_recipes/substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch.yaml`
- Canonical equation #344 anchor proposal: `.omx/research/cascade_c_prime_canonical_equation_344_anchor_proposal_20260526.md`
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable + Catalog #325 STRICT preflight gate
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the per-pair Lagrangian dual primitive)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable (13 inviolable lessons documented in substrate `__init__.py`)
