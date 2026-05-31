# Retroactive sweep — Wave N+11 QUAD HALT anchor ratification

Per Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP discipline applied
to this Wave N+11 QUAD HALT anchor landing 2026-05-30.

## Bug-class symptom signature

A canonical equation (`triple_substrate_composition_orthogonal_pose_axis_savings_v1`)
whose Wave N+6 TRIPLE composition was empirically FALSIFIED (92.48 paired CUDA+CPU)
and whose Wave N+11 QUAD extension was HALTED at the phantom-provenance pre-check
(Catalog #321/#322, $0 spend) could leave the QUAD HALT verdict unrecorded in the
canonical posterior. A downstream consumer (cathedral autopilot ranker,
`canonical_equation_lookup_consumer`) reading the equation without the QUAD HALT
anchor would not see that the QUAD composition was already evaluated + deferred,
risking a re-dispatch of the same phantom-provenance composition.

## Pre-fix window

2026-05-30 (Wave N+11 HALT memo landed at commit `ee15561e9`) → 2026-05-30 (this
anchor landing). During this window the Wave N+11 QUAD HALT was documented in the
memo layer but NOT recorded as a canonical equation anchor.

HONEST correction: the Wave N+6 TRIPLE empirical FALSIFICATION was NOT in this
window — it was already ratified 2026-05-28 (verified present, not re-landed).

## Historical KILL/DEFER/FALSIFY search results (HONEST)

Searched the canonical equation registry + research memos for sister composition
equations whose verdicts may be tainted by the Wave N+11 QUAD HALT:

1. **`triple_substrate_composition_orthogonal_pose_axis_savings_v1`** — the subject
   of THIS landing. Wave N+6 falsification already ratified; Wave N+11 QUAD HALT now
   recorded as anchor 4. RE-EVAL-priority: RESOLVED-BY-THIS-LANDING.

2. **`triple_substrate_composition_alpha_v1`** — sister composition equation.
   Carries the same Wave N+6 TRIPLE substrate set under the alpha-model framing.
   The Wave N+6 92.48 empirical falsification implicitly falsifies the
   super-additive alpha assumption FOR THE COMPOUND C RENDERER. RE-EVAL-priority:
   MEDIUM — operator-routable sister wave should append a cross-reference anchor
   noting the Compound C alpha over-estimate (out of cap=1-per-turn scope for this
   lane). The paradigm (super-additive alpha CAN hold for orthogonal axes) is
   INTACT per Catalog #307; only the Compound C instantiation is falsified.

3. **`cross_codec_super_additive_orthogonality_predictor_v1`** /
   **`cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1`** — broader
   orthogonality/composition predictors. The Wave N+11 QUAD HALT does NOT taint
   these: the HALT was a phantom-provenance refusal (no measurement), not an
   empirical refutation that orthogonal axes compound. RE-EVAL-priority: NONE.

4. **`z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`** — the
   Z7-Mamba-2 equation (the 4th QUAD substrate). Its Wave N+11 stabilizer anchor
   used a MOCK scorer teacher (pose=0); the Z7-Mamba-2 landing memo op-routable #1
   already flags the need for a REAL-scorer-teacher re-fire BEFORE downstream
   composition use. The Wave N+11 QUAD HALT is consistent with this. RE-EVAL-priority:
   LOW — the Z7-Mamba-2 equation should record (in a sister wave) that its
   stabilizer anchor is mock-teacher and BLOCKED for composition until paired-CUDA.

## RE-EVAL-priority assignment

| equation | RE-EVAL-priority | action |
|---|---|---|
| triple_substrate_composition_orthogonal_pose_axis_savings_v1 | RESOLVED | this landing (Wave N+11 HALT anchor) |
| triple_substrate_composition_alpha_v1 | MEDIUM | operator-routable sister wave cross-reference anchor (out of cap=1 scope) |
| cross_codec_super_additive_orthogonality_predictor_v1 | NONE | no taint (HALT was non-measurement refusal) |
| cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1 | NONE | no taint |
| z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1 | LOW | sister wave: record mock-teacher stabilizer is composition-BLOCKED |

## No NEW STRICT gate

This landing does NOT introduce a NEW `check_*` STRICT preflight gate (it appends
an empirical anchor via the existing canonical `update_equation_with_empirical_anchor`
helper). Per Catalog #348 the sweep is still recorded for the verdict-taint audit
trail. The 3 relevant pre-existing gates verified at 0 violations post-landing:
- Catalog #344 (canonical equation memo-reference): 0
- Catalog #371 (orphan auto-trigger stub): 0
- Catalog #185 (live-count-zero drift): 0

## Cross-references

- `.omx/research/canonical_equation_falsification_ratification_wave_n6_triple_wave_n11_quad_recovery_landed_20260530.md` (landing memo)
- Catalog #307 + #344 + #348 + #371 + #321/#322/#323
