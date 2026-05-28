---
council_tier: T2
council_attendees:
  - Shannon-LEAD
  - Dykstra-CO-LEAD
  - Rudin-CO-LEAD
  - Daubechies-CO-LEAD
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Atick-Redlich
  - Rao-Ballard
  - Tishby-memorial
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
canonical_equation_reference: ego_motion_conditioned_predictive_coding_paradigm_savings_v1
horizon_class: asymptotic_pursuit
predicted_band_validation_status: pending_post_training
council_assumption_adversary_verdict:
  - assumption: Single-layer FiLM predictor sufficient for ego-motion-conditioned video
    classification: CARGO-CULTED
    rationale: Rao-Ballard capacity critique 2026-05-27; single-layer cannot capture global+local motion hierarchy
  - assumption: 2-level Rao-Ballard hierarchy is OPTIMAL (vs 1-level Z6 OR 3-level Z8)
    classification: CARGO-CULTED-pending-ablation
    rationale: Sequenced trajectory selection per Z6/Z7/Z8 scoping memo; 2-level chosen as engineering-risk middle ground; ablation needed to validate
  - assumption: MLX-first numpy-portable inflate bridge produces byte-stable archives
    classification: HARD-EARNED
    rationale: Per CLAUDE.md 8th MLX-first standing directive + Catalog #213 sister DP1 canonical Comma2k19 pattern + sister Z5 L1 scaffold byte-stability proof
  - assumption: Cooperative-receiver gradient binding per Atick-Redlich applies to ego-motion-conditioned next-frame prediction
    classification: HARD-EARNED-at-theorem-level CARGO-CULTED-at-Pact-application-level
    rationale: Atick-Redlich 1990 is canonical at neuroscience level; the specific application to scorer-as-shared-prior architecture is pending empirical anchor
  - assumption: L0 scaffold sufficient to land canonical contract before L1 trainer implementation
    classification: HARD-EARNED
    rationale: Per Catalog #241/#242 META layer contract + sister Z5/D1 L0 scaffold pattern; the contract IS the structural deliverable at L0
council_dissent:
  - member: Contrarian
    verbatim: "Z6 original substrate package does NOT EXIST yet in the repo; Z6-v2 designation jumps the version sequence. The honest version label would be Z6-1 OR z6_a per APPEND-ONLY HISTORICAL_PROVENANCE Catalog #110/#113. The v2 naming inherits from the reformulated design memo's filename which itself predates the never-built Z6-v1. Recommendation: future operators should be aware that 'Z6-v2' is the FIRST cargo-cult-unwind iteration on the Z6 design space, not a second iteration on an existing Z6 implementation."
council_decisions_recorded:
  - "op-routable #1: L0 scaffold lands the canonical SubstrateContract per Catalog #241/#242; no trainer/architecture/inflate code at L0 (sister Z5/D1 pattern)"
  - "op-routable #2: L1 promotion requires implementing architecture.py (2-level Rao-Ballard FiLM-ego-motion predictor) + trainer.py (MLX-first with cooperative-receiver loss) + inflate.py (numpy-portable <=200 LOC) + archive.py (Z6V2CU1 grammar) per Catalog #233 4-gate canonical"
  - "op-routable #3: predicted ΔS band derivation deferred to L1 design refinement; Z6-v2 inherits sister Z6 [0.13, 0.16] band as starting prior (Dykstra-feasibility) per the Z6/Z7/Z8 scoping memo Section 18; subject to refinement after Rao-Ballard 2-level capacity is empirically measured"
  - "op-routable #4: Z6-v2 routing per N1 path-5 STRUCTURAL CEILING REINFORCED 2026-05-28 + Catalog #311 ego-motion-conditioned predictive coding paradigm; the surrogate-head pose-binding ceiling at the canonical Z6 surface motivates Z6-v2's cargo-cult-unwind redesign"
  - "op-routable #5: substrate routes through cathedral autopilot via canonical Catalog #335 auto-discovery + Catalog #355 continual learning posterior per the META-layer contract auto-wire pattern"
---

# Z6-v2 cargo-cult-unwind L0 SCAFFOLD design memo

**Date**: 2026-05-27 (canonical UTC anchor: `20260527T053000Z`)
**Substrate ID**: `z6_v2_cargo_cult_unwind`
**Lane ID**: `lane_z6_v2_cargo_cult_unwind_l0_scaffold_20260527`
**Substrate package**: `src/tac/substrates/z6_v2_cargo_cult_unwind/`
**Predecessor design memos** (PV per Catalog #229):
- `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md` (1148 lines; Z6/Z7/Z8 trajectory enumeration)
- `.omx/research/time_traveler_l5_z6_v2_reformulated_design_20260517.md` (1200+ lines; Z6-v2 4-candidate redesign enumeration)
- `.omx/research/n1_path_5_8_seed_bootstrap_k5_landed_20260528T014859Z.md` (28 KB; STRUCTURAL CEILING REINFORCED routing to Catalog #311 ego-motion-conditioned predictive coding)

## 1. Executive summary

L0 SCAFFOLD landing for Z6-v2, the Rao-Ballard capacity-critique-driven
cargo-cult-unwind of the Z6 ego-motion-conditioned predictive-coding
substrate. Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand
council symposium" non-negotiable + Catalog #325 6-step contract.

**Key clarification surfaced by Contrarian**: the Z6 original substrate
package does NOT exist in this repo. The "Z6-v2" designation inherits from
the reformulated design memo (`time_traveler_l5_z6_v2_reformulated_design_20260517.md`)
which itself was authored before any Z6 trainer/architecture/inflate code
existed. Z6-v2 IS therefore the FIRST cargo-cult-unwind iteration on the
Z6 design space, not a second iteration on a built Z6.

The L0 scaffold lands ONLY the canonical `SubstrateContract` per Catalog
#241/#242 META layer contract registration — no architecture / trainer /
inflate / archive code. This is the canonical sister-Z5/D1 L0 pattern
(register the contract; defer implementation to L1 promotion).

**Substrate-distinguishing primitives** (per Catalog #272 contract;
deferred to L1 implementation):

1. **2-level Rao-Ballard hierarchical FiLM-ego-motion predictor** — engineering
   middle ground between Z6 original's single-layer (insufficient per
   Rao-Ballard capacity critique) and Z8's full 3-level (over-engineering
   for L1).
2. **Atick-Redlich cooperative-receiver gradient binding** per Catalog #311.
3. **MLX-first training on M5 Max** per CLAUDE.md 8th MLX-first standing
   directive 2026-05-26.
4. **numpy-portable inflate runtime** (<=200 LOC + <=2 deps per HNeRV
   parity L4 substrate-engineering waiver).

## 2. Operating-within assumption-statement (per Catalog #292)

The shared assumption this design is operating within: *"The cargo-cult-
unwind of the Z6 ego-motion-conditioned predictive-coding substrate via
2-level Rao-Ballard hierarchical binding + Atick-Redlich cooperative-
receiver primitive will close the surrogate-head pose-binding STRUCTURAL
CEILING surfaced by N1 path-5 8-seed bootstrap 2026-05-28 at the canonical
class-shift level per Catalog #311."*

This assumption is **HARD-EARNED at the paradigm level** (per Rao-Ballard
1999 + Atick-Redlich 1990 + Catalog #311 canonical routing) and
**CARGO-CULTED at the Pact application level** (no Pact empirical anchor
on the 2-level Rao-Ballard hierarchy specifically; the sister Z5/Z6
single-layer variants have not produced empirical receipts either). The
L1 trainer implementation is the canonical refutation surface.

## 3. Cargo-cult audit per assumption (per Catalog #303)

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | Single-layer FiLM predictor sufficient for ego-motion video | **CARGO-CULTED** | Z6-v2 unwinds via 2-level Rao-Ballard hierarchy |
| 2 | 2-level chosen as OPTIMAL middle ground | **CARGO-CULTED-pending-ablation** | L1 trainer ablates 1 vs 2 vs 3 level capacity |
| 3 | MLX state_dict -> npz -> ZIP-member -> numpy bridge byte-stable | **HARD-EARNED** | Catalog #213 + sister Z5 L1 scaffold byte-stability proof |
| 4 | Cooperative-receiver applies to ego-motion next-frame prediction | **HARD-EARNED-theorem CARGO-CULTED-application** | L1 empirical anchor |
| 5 | L0 contract-only scaffold sufficient as L0 deliverable | **HARD-EARNED** | Sister Z5/D1 L0 pattern |
| 6 | 50ep smoke cost band ~$0.30 sufficient for L0 calibration | **CARGO-CULTED** | Refined post-empirical at L1 |
| 7 | T4 floor capacity sufficient for 2-level Rao-Ballard predictor | **CARGO-CULTED-pending-measurement** | min_vram_gb refined at L1 |

The 4 CARGO-CULTED rows above are operator-routable for L1 promotion
research; each has explicit refutation path declared per CLAUDE.md
"Forbidden premature KILL without research exhaustion" + Catalog #308
alternative-probe-methodology enumeration.

## 4. 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS** (class-shift not within-class): YES per Catalog #311.
   Z6-v2 is scorer-relationship class-shift via ego-motion-conditioned
   predictive coding paradigm. Distinct from architecture class-shift,
   decode-time-contract class-shift, training-time-paradigm class-shift,
   wire-grammar class-shift axes.

2. **BEAUTY + ELEGANCE** (30-sec-reviewable): YES at L0 design level. The
   canonical contract registration in `__init__.py` is the entire L0
   scaffold; ~200 lines reviewable in 30 seconds. L1 architecture +
   trainer + inflate will be substrate-engineering scope per HNeRV L7.

3. **DISTINCTNESS** (explicitly different from sisters): YES. Z6-v2 differs
   from Z5 (cooperative-receiver loss only) by adding the FiLM-ego-motion
   predictor architectural primitive. Differs from sister Z6 design (which
   was single-layer FiLM) by adding 2-level Rao-Ballard hierarchy. Differs
   from Z8 (3-level full hierarchy + DreamerV3) by staying at 2 levels.

4. **RIGOR** (premise verification + adversarial review + assumption
   classification + empirical anchor): YES. PV per Catalog #229 read 8+
   predecessor memos including the Z6/Z7/Z8 scoping memo (1148 lines) and
   the Z6-v2 reformulated design memo. Assumption classification per
   Section 3 above. Empirical anchor: N1 path-5 8-seed bootstrap 2026-05-28
   STRUCTURAL CEILING REINFORCED verdict at the canonical Z6 surface.

5. **OPTIMIZATION PER TECHNIQUE** (canonical-vs-unique decision per layer
   per Catalog #290): substrate adopts canonical inflate device selection
   (Catalog #205) + canonical scorer-preprocess routing (Catalog #164) +
   canonical auth-eval helper (Catalog #226) + canonical 3-arg archive
   grammar (Catalog #146); forks unique substrate-distinguishing primitives
   (2-level Rao-Ballard FiLM predictor + Atick-Redlich cooperative-receiver
   gradient binding). L1 trainer will materialize the unique forks.

6. **STACK-OF-STACKS-COMPOSABILITY** (orthogonal axes + additive ΔS): per
   sister Z6/Z7/Z8 scoping memo Section 13 composition matrix, Z-variants
   compose orthogonally with NSCS06 v8 Path B (wavelet residual) +
   A-STACK (architectural primitives) + Rudin floor (interpretable
   decoder) per the Path 2 LATTICE directive 2026-05-15.

7. **DETERMINISTIC REPRODUCIBILITY** (byte-stable + seed-pinned): YES per
   Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger
   + canonical seed pinning + deterministic ZIP archive helper per sister
   NSCS06 v8 Path B + DP1 patterns.

8. **EXTREME OPTIMIZATION + PERFORMANCE**: per CLAUDE.md
   "Production-hardened dispatch optimization protocol" + Catalog #270
   umbrella gate (declared at L1 trainer); Tier 1 engineering primitives
   (autocast / TF32 / torch.compile / no_grad / GTScorerCache F3 / canonical
   scorer-loss helper routing); Tier 2 hardware correctness
   (min_vram_gb / min_smoke_gpu / video_input_strategy / pyav_decode_strategy
   / target_modes / canonical NVML env block); Tier 3 substrate correctness
   (canonical auth-eval / canonical inflate device / recipe-vs-trainer-state
   consistency / no phantom device-named output directories).

9. **OPTIMAL MINIMAL CONTEST SCORE**: ASYMPTOTIC-PURSUIT target band
   [0.13, 0.16] inherited from sister Z6 prediction per Z6/Z7/Z8 scoping
   memo Section 18 Dykstra-feasibility analysis. Subject to refinement
   after 2-level Rao-Ballard capacity is empirically measured at L1.

## 5. Observability surface (per Catalog #305)

Declared inline in `__init__.py` `OBSERVABILITY_SURFACE` constant across
all 6 facets per CLAUDE.md "Max observability — non-negotiable":

- **inspectable_per_layer**: 2-level Rao-Ballard hierarchy with per-layer
  prediction-error norms exposed via `architecture.layerwise_inspector`
- **decomposable_per_signal**: per-pair total loss decomposed into 4 terms
  (micro-residual cooperative-receiver + meso-residual rate + FoE ego-
  motion consistency + Rao-Ballard 2-level binding)
- **diff_able_across_runs**: byte-stable archive under deterministic seed
- **queryable_post_hoc**: fcntl-locked JSONL observability stream
- **cite_able**: every artifact carries canonical Provenance per Catalog #323
- **counterfactual_able**: byte-mutation smoke discipline per Catalog
  #105/#139/#220/#272

## 6. Predicted ΔS band + Dykstra-feasibility (per Catalog #296)

**Inherited band** per sister Z6 prediction per Z6/Z7/Z8 scoping memo Section
18: **CPU [contest-CPU Linux x86_64 GHA] [0.13, 0.16]** `[prediction;
Dykstra-feasibility-validated; HIGH VARIANCE pending L1 empirical anchor;
pending_post_training per Catalog #324]`.

**Dykstra-feasibility convex-intersection projection** (per the parent
scoping memo Section 18; Z6-v2 inherits the Z6 polytope projection at this
L0 stage; refined at L1 trainer empirical anchor):

```
Z6_v2_achievable_polytope = projection(rate_budget) ∩
                            projection(2_level_predictive_coding_rate_R_pc) ∩
                            projection(2_level_predictor_residual_bound) ∩
                            projection(SegNet_distortion_bound) ∩
                            projection(PoseNet_distortion_bound) ∩
                            projection(cooperative_receiver_atick_redlich_bound)
```

The 2-level Rao-Ballard hierarchy is predicted to capture more residual
entropy reduction than the single-layer baseline at the cost of ~30 KB
additional archive bytes (level-1 weights), netting to roughly the same
[0.13, 0.16] band per Dykstra's subadditivity penalty. L1 trainer empirical
anchor refines this.

**First-principles citations** per Catalog #296: Rao-Ballard 1999
predictive coding hierarchy + Atick-Redlich 1990 cooperative-receiver
theorem + Tishby-Zaslavsky 2015 IB framework + Wyner-Ziv 1976
side-information theorem (Wyner-Ziv applied to frame-1-against-frame-0
deferred to Z7 sister substrate).

## 7. Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical `tac.substrates._shared.trainer_skeleton` | TF32 + Catalog #178/#190 hardware substrate detection |
| Scorer loss helper | ADOPT canonical `tac.substrates._shared.score_aware_common.score_pair_components` | Catalog #164 HARD-EARNED per PR95 differentiability lesson |
| eval_roundtrip | ADOPT canonical `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" HARD-EARNED |
| YUV6 patch | ADOPT canonical `patch_upstream_yuv6_globally()` | Catalog #187 HARD-EARNED per PR95 |
| EMA decay | ADOPT canonical 0.997 weights | CLAUDE.md "EMA — NON-NEGOTIABLE" HARD-EARNED |
| Archive grammar (Z6V2CU1) | UNIQUE FORK | Substrate-distinguishing per Catalog #272 |
| 2-level Rao-Ballard FiLM-ego-motion predictor | UNIQUE | Substrate-distinguishing primitive #1 |
| Atick-Redlich cooperative-receiver gradient binding | ADOPT canonical sister primitive from `tac.codec.cooperative_receiver.atick_redlich` (when landed) | Shared with ATW + Z4 |
| MLX state_dict bridge | UNIQUE | Per 8th MLX-first standing directive |
| numpy-portable inflate (<=200 LOC + <=2 deps) | UNIQUE | Per HNeRV parity L4 substrate-engineering waiver |
| Inflate runtime `select_inflate_device` | ADOPT canonical | Catalog #205 HARD-EARNED |
| Auth eval helper | ADOPT canonical `gate_auth_eval_call` | Catalog #226 HARD-EARNED |
| Modal call_id ledger | ADOPT canonical | Catalog #245 HARD-EARNED |
| Lane registry pre-registration | ADOPT canonical | Catalog #126 + Catalog #90 HARD-EARNED |
| Subagent commit serializer | ADOPT canonical | Catalog #117/#157/#174 HARD-EARNED |

## 8. Reactivation criteria per CLAUDE.md "Forbidden premature KILL"

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": Z6-v2
L0 scaffold is `research_only=true` + `dispatch_enabled=false`. If the L1
empirical anchor falsifies the predicted [0.13, 0.16] band, the verdict is
DEFERRED-pending-research with the following 4 reactivation paths per
priority:

1. **Ablate 1 vs 2 vs 3 level Rao-Ballard hierarchy capacity** at the same
   trainer + same archive grammar. Determines whether 2-level is the
   optimal middle ground vs Z6-original-1-level vs Z8-full-3-level.
2. **Probe per-region latent vs per-pair latent reducer** per Catalog
   #308 alternative-probe-methodology. The N1 path-5 result was specific
   to per-pair-dominant SegNet argmax reducer; per-region disambiguation
   could change the verdict.
3. **Refine FoE prior conditioning** per Ballard embodied-vision lens —
   the focus-of-expansion prior may need explicit camera intrinsics
   conditioning per Gibson 1950 ego-motion-matched foveation.
4. **Comma2k19 dashcam pretraining** per Catalog #213 canonical helper
   to warm up the predictor's temporal-pattern capture before contest-
   video fine-tuning (sister DP1 pattern). Cost: ~$5 Modal A100 ~2 hours.

## 9. Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status: pending_post_training` per Catalog
#324. The Tier-C density measurement on the post-training archive sha
will land at L1 promotion; the L0 scaffold's predicted band is
inherited-from-sister-Z6 + Dykstra-projection lower bound per Section 6
above, NOT a phantom-random-init Tier-C density per the C6 IBPS bug class
anchor (`feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`).

Reactivation criterion: post-training Tier-C density measurement via
`tools/mdl_scorer_conditional_ablation.py --tier c` on the L1-landed
archive sha.

## 10. Operator-routable next steps

1. **L1 trainer implementation** — `architecture.py` (2-level Rao-Ballard
   FiLM-ego-motion predictor) + `trainer.py` (MLX-first with cooperative-
   receiver loss; canonical scorer-preprocess routing per Catalog #164;
   canonical EMA per CLAUDE.md non-negotiable) + `inflate.py` (numpy-
   portable <=200 LOC per HNeRV L4) + `archive.py` (Z6V2CU1 grammar
   per Section 6).
2. **L0 → L1 promotion gate** per Catalog #233 canonical 4-gate: (1)
   smoke green; (2) Tier C MDL density measured; (3) 100ep auth-eval
   anchor with byte-deterministic archive sha256; (4) custody validated
   per Catalog #127.
3. **Per-substrate symposium re-deliberation** at L1 per Catalog #325
   14-day window: re-convene sextet + grand-council attendees with the
   L1 empirical anchor in hand to re-classify CARGO-CULTED-pending-
   ablation assumptions per Section 3.

## 11. Cross-references

- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
  symposium" non-negotiable
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  non-negotiable
- CLAUDE.md "INDIVIDUALLY-FRACTAL" standing directive 2026-05-27
- CLAUDE.md 8th MLX-first standing directive 2026-05-26
- Catalog #241/#242 (META layer contract validation)
- Catalog #303 (cargo-cult audit section)
- Catalog #294 (9-dimension success checklist evidence section)
- Catalog #305 (observability surface section)
- Catalog #296 (predicted-band Dykstra-feasibility check)
- Catalog #309 (horizon_class declaration)
- Catalog #325 (per-substrate symposium 14-day window)
- Catalog #324 (post-training Tier-C validation discipline)
- Catalog #311 (ego-motion-conditioned predictive coding paradigm)
- Catalog #233 (L1→L2 promotion canonical 4-gate)
- Sister design memo `time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md`
- Sister design memo `time_traveler_l5_z6_v2_reformulated_design_20260517.md`
- N1 routing memo `n1_path_5_8_seed_bootstrap_k5_landed_20260528T014859Z.md`

## 12. Mission contribution per Catalog #300

`frontier_breaking_enabler` — Z6-v2 unblocks downstream L1 promotion +
paid-CUDA dispatch eligibility per Catalog #325 14-day window. The
canonical contract registration is the structural foundation that
enables sister consumers (cathedral autopilot per Catalog #335 auto-
discovery + continual-learning posterior per Catalog #355 wire-in) to
incorporate the Z6-v2 substrate without manual wiring.
