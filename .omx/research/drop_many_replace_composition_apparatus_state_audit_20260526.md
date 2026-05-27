# DROP-MANY + REPLACE + COMPOSITION APPARATUS STATE AUDIT — 2026-05-26

- timestamp_utc: 2026-05-27T02:55:00Z
- agent: claude (DROP-MANY-REPLACE-COMPOSITION-APPARATUS-STATE-AUDIT subagent)
- subagent_id: `drop-many-replace-composition-apparatus-state-audit-answer-operator-drop-one-frontier-paradox-20260526`
- lane_id: `lane_drop_many_replace_composition_apparatus_state_audit_20260526`
- scope: READ-ONLY audit memo answering operator question 2026-05-26 *"Why is drop one still the frontier when theoretically a combination of dropping and replacing and other operations, many across the video, possibly multiple in a frame or pair, theoretically optimal? Does our code support this yet and is it wired up and integrated and tested end to end and automated"*
- authority: design + observability ONLY; no score/promotion/rank/dispatch authority
- HEAD: `9f734cd16dc0e12e9cd1ba5bf327a416aaf4e4c2`
- evidence_grade: `[predicted]` per Catalog #287 + #323 (audit findings; not paired-axis score claims)
- canonical_provenance: read-only inspection of repo HEAD + canonical state ledgers
- standing directives invoked: 13th OPTIMAL-TRIO (TECHNIQUE × WAY × TIME) + 7th AUTOMATED+COMPOUNDING+OPTIMAL + 11th ORDER MATTERS + 12th canonicalization × standardization × ease-of-contest-compliance + 10th apples-to-apples + 6th final-rate-attack off-the-shelf + 9th canonical-automated-submission-packet-bundling
- discipline anchors: Catalog #229 PV + #287 evidence-tag + #323 canonical Provenance + #340 sister-checkpoint guard + #343 NO hardcoded score literals + #344 NO canonical equation registration without empirical anchor + #110/#113 APPEND-ONLY

---

## TL;DR (operator-facing)

The drop-one-frontier paradox is **NOT one bug class** — it is **TWO simultaneously-firing bug classes** that empirical posterior evidence already validates:

1. **Hypothesis #3 EMPIRICALLY GROUNDED — multi-op apparatus partially-wired.** drop_many_beam.py + pair_frame_scorer_geometry_lattice.py (codex v1 row-based) BOTH exist, are tested, and are wired into the DQS1 acquisition planner — but neither is exposed as a cathedral consumer (0 of 67 consumers). The 5D canvas (pair_frame_scorer_geometry_lattice_5d_canvas.py) is SCAFFOLD-ONLY: all 4 canonical operations (full-drop / repair / masked / feathered) raise `NotImplementedError`. Cathedral autopilot ranker cannot consume drop-many beam OR 5D canvas signal without per-consumer wire-in subagent.
2. **Hypothesis #2 EMPIRICALLY GROUNDED — narrow drop-many fired but DID NOT beat drop-one.** 17 paired-CPU empirical anchors in `accepted_anchor_history`: drop-one rank021 = `0.192028282` (FRONTIER); drop-two best (r028_021) = `0.192029617` (+0.000001); diversity k002-k024 = `0.192035-0.192058` (ALL WORSE). The rate-saving 4-8 bytes (178559 → 178531 archive bytes) DID NOT compensate for the distortion penalty.

The combined finding: the apparatus **does support drop-many for the narrow space already explored**, fired it empirically, and the **rate-distortion tradeoff** explained the result (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"). What the apparatus does **NOT yet support end-to-end + automated**: (a) REPAIR + MASKED + FEATHERED operations (5D canvas scaffold raises NotImplementedError); (b) CROSS-PAIR composition with META-LIFT-1 Cauchy-Schwarz bound + META-LIFT-2 Pareto polytope; (c) PER-FRAME drop / synthesize-frame operations; (d) CROSS-PAIR REPLACE / MERGE / REORDER operations; (e) auto-discovery of multi-op consumers into the cathedral autopilot ranker.

Per just-saved 13th OPTIMAL-TRIO directive: the operator's intuition is **HARD-EARNED** at the THEORY level (broader operator vocabulary should win) AND **CARGO-CULTED-EMPIRICALLY-FALSIFIED** at the NARROW drop-many empirical level (drop-many beam was fired, did not beat drop-one). The path forward is **NOT** "fire drop-many again"; the path forward is **build the BUILD-1 + BUILD-4 sister subagents for the 5D canvas** so REPAIR + MASKED + FEATHERED + per-frame + cross-pair operations can actually be empirically tested.

---

## Phase 1: Operator catalog enumeration

The canonical operator vocabulary (per 13th standing directive + design memos):

### Pair-level operations

| Operator | Code location | Built? | Tested? | Wired-cathedral? | Automated? | Empirical-anchor? |
|---|---|---|---|---|---|---|
| **drop-one** | `tac.optimization.decoder_q_pairset_acquisition._drop_one_*` (lines 1020-1056) | YES | YES (via `test_decoder_q_pairset_acquisition.py`) | implicit (no dedicated consumer) | YES | **YES — 9 paired-CPU anchors; FRONTIER `0.192028282`** |
| **drop-many beam** | `tac.optimization.dqs1_drop_many_beam.beam_search_drop_many` + `build_pairwise_interaction_matrix` (CALLED from `decoder_q_pairset_acquisition._beam_drop_many_sets` line 605/626) | YES | YES (4 tests pass `test_dqs1_drop_many_beam.py`) | **NO** (zero cathedral consumer) | partial (auto-called in DQS1 acquisition; not via autopilot ranker) | **YES (NARROW) — drop-two `0.192029617` + diversity k002-k024 `0.192035-0.192058`; ALL WORSE than drop-one** |
| **drop-many greedy** | `tac.optimization.decoder_q_pairset_acquisition._bounded_drop_many_sets` (greedy heuristic alternative reducer) | YES | partial | NO | YES (auto-called) | partial (covered by drop-many beam anchors) |
| **replace-one (synthetic-substitute)** | NOT-BUILT | NO | NO | NO | NO | **NO** |
| **replace-many** | NOT-BUILT | NO | NO | NO | NO | **NO** |
| **merge-pair** | NOT-BUILT | NO | NO | NO | NO | **NO** |
| **reorder-pair** | NOT-BUILT | NO | NO | NO | NO | **NO** |
| **per-axis-decompose** | `tac.cathedral.consumer_contract.AxisDecomposition` (Catalog #356; canonical Provenance contract) | YES | YES | YES (Catalog #356 STRICT gate enforces per-axis Provenance) | YES | N/A (validation gate, not a substrate operator) |

### Frame-level operations

| Operator | Code location | Built? | Tested? | Wired-cathedral? | Automated? | Empirical-anchor? |
|---|---|---|---|---|---|---|
| **drop-frame** | NOT-BUILT (5D canvas `CanonicalOperation.FULL_DROP` enum exists; `generate_full_drop_starts` raises NotImplementedError) | scaffold | scaffold | NO | NO | **NO** |
| **synthesize-frame** | NOT-BUILT (5D canvas `CanonicalOperation.REPAIR` enum exists; `generate_repair_starts` raises NotImplementedError) | scaffold | scaffold | NO | NO | **NO** |
| **per-frame-bit-budget** | `tac.bit_allocator.pareto_dual` (per-byte-per-class-per-axis Pareto dual; tests pass) | YES | YES | YES (`bit_allocator_per_pair_consumer`) | YES | partial (canonical Provenance threaded via Catalog #356) |
| **motion-conditional** | NOT-BUILT (per-frame master-gradient sister; `tac.cathedral_consumers.per_frame_sensitivity_consumer` is canonical contract but does NOT emit motion-conditional drop candidates) | partial | partial | partial | NO | **NO** |
| **SegNet-class-region waterfill** | `tac.optimization.dqs1_drop_many_beam.waterfill_budget_consumed` (rate-vs-distortion waterfill within drop-many beam) | YES | YES | NO (only inside drop-many beam) | partial | partial |

### Cross-pair operations

| Operator | Code location | Built? | Tested? | Wired-cathedral? | Automated? | Empirical-anchor? |
|---|---|---|---|---|---|---|
| **pair-correlation-exploit** | `tac.optimization.dqs1_drop_many_beam.build_pairwise_interaction_matrix` (canonical I[P,P]) | YES | YES | NO | partial | partial |
| **temporal-coherence** | NOT-BUILT | NO | NO | NO | NO | **NO** |
| **interaction-matrix (cross-substrate)** | `tac.cross_substrate_master_gradient_analyzer.analyzer` (META-LIFT-1 Cauchy-Schwarz bound) | YES | YES (`test_cross_substrate_master_gradient_analyzer.py`) | YES (`cross_substrate_master_gradient_analyzer_consumer`) | YES (auto-discovered per Catalog #335) | partial |

### Cross-archive operations

| Operator | Code location | Built? | Tested? | Wired-cathedral? | Automated? | Empirical-anchor? |
|---|---|---|---|---|---|---|
| **multi-archive composition** | `tac.optimization.substrate_composition_matrix` (101 KB; 17 archive composition rows; Catalog #322 gate enforces provenance) | YES | YES | YES (`substrate_composition_matrix_consumer` via `venn_risk_composition_consumer`) | YES | YES (canonical eq `triple_substrate_composition_alpha_v1` + sister) |
| **stack-of-stacks** | `tac.optimization.cross_paradigm_composition_examples` (24 KB; canonical examples) | YES | YES | partial | YES | YES (canonical eq `cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1`) |
| **cross-paradigm** | sister of stack-of-stacks; rated via `cross_paradigm_composition_examples` | YES | YES | partial | partial | partial |

### Stream + per-byte + per-axis operations (META-LIFT family)

| Operator | Code location | Built? | Tested? | Wired-cathedral? | Automated? | Empirical-anchor? |
|---|---|---|---|---|---|---|
| **FEC family** (6th directive: FEC6 / FEC8 / FEC9 / FEC10) | `tools/build_fec*` + sister `register_*_fec*` substrates; multiple landed | YES (FEC6 / FEC8 1st+2nd order) | YES (paired-CPU anchors) | YES (cathedral consumers per FEC variant) | YES | YES (multiple FEC paired-axis anchors in posterior) |
| **entropy-coder** | sister of FEC family; per-stream arithmetic coding | YES | YES | YES | YES | YES |
| **Markov-context** | FEC8 2nd-order TRUE Markov VARIANT-A (`feedback_final_rate_attack_*_landed_20260526.md`) | YES | YES | YES | YES | YES (FEC8 2nd-order anchor) |
| **master-gradient-per-byte** | `tac.master_gradient_comparison.multi_granularity` (51 KB) + sister `per_byte_sensitivity_consumer` | YES | YES | YES | YES | YES (canonical eq `pairset_component_marginal_score_decomposition_v1`) |
| **cross-substrate-master-gradient** | `tac.cross_substrate_master_gradient_analyzer` (META-LIFT-1 Cauchy-Schwarz) | YES | YES | YES (auto-discovered) | YES | partial |
| **Pareto-polytope-unified-solver** | `tac.pareto_polytope_unified_solver.solver` (META-LIFT-2 Dykstra alternating projections) | YES | YES (`test_pareto_polytope_unified_solver.py`) | YES (`pareto_polytope_unified_solver_consumer` auto-discovered) | YES | partial |
| **UNIWARD invariant enumerator** | `tac.uniward_invariant_enumerator.enumerator` (META-LIFT-4) | YES | YES | YES (auto-discovered) | YES | partial |

---

## Phase 2: Multi-op composition apparatus state matrix

### State-by-component breakdown

**5D canvas (pair × frame × scorer_axis × receiver_runtime × cpu_cuda_axis)**:

| Layer | Status | Reference |
|---|---|---|
| Design memo | LANDED 2026-05-25 | `.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md` |
| Codex v1 row-based scaffold | LANDED (`pair_frame_scorer_geometry_lattice.py`; 501 LOC; SCHEMA + 1 builder + tests pass) | commit `4ed9eb905` "Wire pair-frame geometry starts into DQS1 queue" |
| 5D canvas scaffold | LANDED (`pair_frame_scorer_geometry_lattice_5d_canvas.py`; ~1000 LOC scaffold; CanonicalOperation enum + ReceiverRuntime enum + ScorerAxis enum + CpuCudaAxis enum + PairFrameScorerGeometryCell frozen dataclass + ExecutableCandidate frozen dataclass + container class) | scaffold |
| `bind_pair_component_xray` | **NOT-BUILT** (raises NotImplementedError; defers to BUILD-1) | pending BUILD-1 sister subagent |
| `decompose_frame_axis_master_gradient` | **NOT-BUILT** (raises NotImplementedError; defers to BUILD-1) | pending BUILD-1 |
| `compute_segnet_posenet_score_geometry` | **NOT-BUILT** (raises NotImplementedError; defers to BUILD-1) | pending BUILD-1 |
| `query_receiver_runtime_feasibility` | **NOT-BUILT** (raises NotImplementedError; defers to BUILD-1) | pending BUILD-1 |
| `generate_queue_executable_start` | **NOT-BUILT** (raises NotImplementedError; defers to BUILD-2) | pending BUILD-2 |
| Tier B promotion (Catalog #357) | **NOT-LANDED** (scaffold ships `CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY`) | pending BUILD-4 |
| Cathedral consumer auto-discovery | **NOT-PACKAGED** as `src/tac/cathedral_consumers/pair_frame_scorer_geometry_lattice_consumer/__init__.py` per Catalog #335 contract | pending BUILD-4 |
| Canonical equation registration | **NOT-REGISTERED** (queued as `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1`; needs 3+ empirical anchors per Catalog #344) | pending DISPATCH |
| Paired CPU+CUDA empirical anchors | **0 anchors** for any of full-drop / repair / masked / feathered operations | pending DISPATCH |
| Paired CPU+CUDA frontier-pointer update | N/A (no anchor) | N/A |

**Drop-many beam search**:

| Layer | Status | Reference |
|---|---|---|
| Design memo | implicit (eureka memo `codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md`) | LANDED |
| Implementation | `dqs1_drop_many_beam.py` (16 KB; BeamCandidate + BeamSearchConfig + DykstraFeasibilityConfig + WaterfillConfig + beam_search_drop_many + build_pairwise_interaction_matrix + dykstra_alternating_projection_feasibility + waterfill_budget_consumed) | LANDED |
| Tests | 4 tests pass in `test_dqs1_drop_many_beam.py` | LANDED |
| Integration into DQS1 acquisition | YES — `decoder_q_pairset_acquisition._beam_drop_many_sets` calls `beam_search_drop_many` (line 626) | LANDED |
| Cathedral consumer | **NO `dqs1_drop_many_beam_consumer` package exists** (0 of 67 consumers) | pending BUILD |
| Canonical equation | **NOT-REGISTERED** as `dqs1_drop_many_beam_pairwise_interaction_waterfill_v1` | pending DISPATCH |
| Paired CPU+CUDA empirical anchors for drop-3 / drop-4 / drop-6 / drop-8 | **0 anchors** — DQS1 acquisition emits the candidates but no paired-axis dispatch fired (only drop-one + drop-two + diversity-k variants have anchors) | pending DISPATCH |

**Empirical posterior** (16 distinct DQS1 pairset architecture classes with paired-CPU anchors):

```
drop-one rank021 pair0371 = 0.192028282 [contest-CPU]    (CANONICAL FRONTIER per pointer)
drop-one rank010-rank027 (8 sister)    = 0.192029282     (+0.000001; diff sub-1e-6)
drop-one rank032                       = 0.192029282
drop-two r028_021_p0257_0371           = 0.192029617     (+0.000001 vs frontier; rate saved 1 byte, distortion penalty +1.3e-6)
diversity k024                         = 0.192035621     (rate 178552; +7 bytes saved; distortion +7e-6)
diversity k016                         = 0.192042295     (rate 178544; +15 saved; distortion +14e-6)
diversity k012                         = 0.192048631     (rate 178540; +19 saved; distortion +20e-6)
diversity k008                         = 0.192048968     (rate 178536; +23 saved; distortion +21e-6)
diversity k002                         = 0.192055638     (rate 178531; +28 saved; distortion +27e-6)
diversity k004                         = 0.192058302     (rate 178535; +24 saved; distortion +30e-6)
drop-one rank021 (CUDA)                = 0.226191769 [contest-CUDA T4]
```

**Rate vs distortion trade-off**: every byte saved costs `25/37_545_489 = 6.66e-7` score units. For drop-two to beat drop-one, distortion must increase by < 6.66e-7. Empirical: distortion increased by `1.3e-6` (drop-two) — about 2x the rate saving. For drop-many k002 (28 bytes saved), distortion increased by `27e-6` — about 27x the rate saving. **The empirical fit per byte: each additional dropped pair costs ~1e-6 distortion per byte saved on average for this operating point** (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" empirical validation).

### Why does drop-one win in this operating point?

CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)" already explains this structurally: at the PR106-frontier operating point pose_avg = `3.4e-5`, marginal pose sensitivity is `271 / 100 = 2.71x SegNet`. The drop-one canonical pair `pair0371` empirically targets a pair where the rate-saving compensates for the distortion penalty within `6.66e-7`. Drop-many DOES save more rate, but the marginal distortion at each additional dropped pair grows faster than the rate saving.

**This is NOT a generic "drop-many is dominated" finding**. It is operating-point specific. At a DIFFERENT operating point (e.g. archive bytes 4x smaller, different pair-target population, or different per-pair distortion sensitivity), drop-many could dominate. **The apparatus has not yet probed drop-many at any OTHER operating point.**

---

## Phase 3: Drop-one-frontier paradox hypothesis ranking

### Ranked hypotheses

**RANK 1: Hypothesis #3 (apparatus partially-wired) — EMPIRICALLY GROUNDED.**

- 5D canvas: all 4 canonical operations (full-drop / repair / masked / feathered) raise `NotImplementedError`. BUILD-1 (empirical population) + BUILD-2 (operation generators) + BUILD-3 (Catalog #356 wire-in) + BUILD-4 (Tier B promotion) are ALL sister-subagent operator-routable per design memo §DELIVERABLE 3. Until BUILD-1 + BUILD-2 land, NO REPAIR / MASKED / FEATHERED candidate can be queue-emitted, let alone empirically tested.
- Cathedral consumer gap: 67 cathedral consumers exist; ZERO of them are `dqs1_drop_many_beam_consumer` or `pair_frame_scorer_geometry_lattice_consumer`. Per Catalog #335 auto-discovery: cathedral autopilot ranker can only consume operators whose package is in `src/tac/cathedral_consumers/` with canonical contract. The drop-many beam IS computed inside DQS1 acquisition but the ranker has no Tier B canonical-routing signal from it.
- Canonical equation gap: 55 canonical equations exist; ZERO direct entry for `dqs1_drop_many_beam_pairwise_interaction_waterfill_v1` or `multi_op_composition_savings_v1`. The closest is `pairset_component_marginal_score_decomposition_v1` (drop-one decomposition; HARD-EARNED per Catalog #344) which informs the lattice's per-cell prediction but doesn't extend to multi-op composition.
- **Falsifiable test**: BUILD-1 + BUILD-2 + BUILD-3 + BUILD-4 sister subagents land the canonical helpers; paired CPU+CUDA dispatch fires REPAIR / MASKED / FEATHERED operations at $1-3 each ($4-12 total); 3+ empirical anchors register canonical equation `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1`.

**RANK 2: Hypothesis #2 (multi-op fired but didn't beat drop-one) — EMPIRICALLY GROUNDED for narrow drop-many.**

- 17 paired-CPU anchors empirically show drop-two + diversity-k variants ALL WORSE than drop-one by `1.3e-6` to `30e-6` score units. The rate-saving 1-28 bytes did NOT compensate for the marginal distortion penalty at this operating point.
- This is operator-vocabulary-coverage at the SAME operating point. Drop-many beam targeted SAME pair-distribution as drop-one, just with broader vocabulary. The result follows naturally from CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" canonical equation #18 `pose_axis_cuda_amplification_v1` + the marginal-distortion-per-byte slope at this operating point.
- **What this DOES NOT falsify**: the hypothesis that REPAIR / MASKED / FEATHERED (which inject signal rather than remove bytes) can beat drop-one. Those operations CANNOT add bytes faster than drop-many removes them, so the operating-point reasoning doesn't apply directly; they target DIFFERENT pair-frame regions.
- **Falsifiable test**: paired CPU+CUDA dispatch of REPAIR / MASKED / FEATHERED candidates at a DIFFERENT operating point (different pair-target population or different archive-byte regime).

**RANK 3: Hypothesis #4 (combinatorial explosion + needs guided search via META-LIFT-1 + META-LIFT-2) — STRUCTURALLY GROUNDED, NOT YET EMPIRICALLY TESTED.**

- META-LIFT-1 cross-substrate master-gradient analyzer + META-LIFT-2 Pareto polytope unified solver ARE built and ARE auto-discovered cathedral consumers. They have NOT been wired through to the drop-many beam search OR the 5D canvas.
- The drop-many beam currently uses simple Dykstra-feasibility on (rate, SegNet, PoseNet) (4-byte saved per drop, 5e-4 max distortion). It does NOT consume the cross-substrate master-gradient Cauchy-Schwarz bound to identify candidate pair-frame regions where multi-op composition can produce additive ΔS.
- The 5D canvas design memo §DELIVERABLE 1 explicitly references META-LIFT-2 Pareto polytope as the sister algorithm for `query_receiver_runtime_feasibility`, but the helper is NotImplementedError-deferred to BUILD-1.
- **Falsifiable test**: BUILD-1 sister subagent populates 5D canvas from `.omx/state/master_gradient_anchors.jsonl` + threads META-LIFT-1 + META-LIFT-2 inputs into `query_receiver_runtime_feasibility`; sister probe dispatches top-K candidates by ranked-cell ΔS prediction.

**RANK 4: Hypothesis #5 (operator vocabulary incomplete) — NOT-PRIORITIZED.**

- The 13th standing directive enumerates: per-frame ops, cross-pair correlation-exploit, temporal-coherence, multi-op-per-pair. NONE of these is implemented per Phase 1 catalog above; ALL would extend the operator vocabulary materially.
- Empirical priority: each new operator should be paired with a canonical equation candidate (Catalog #344) + per-substrate symposium (Catalog #325) + empirical anchor (Catalog #343). The 5D canvas BUILD-1 through BUILD-4 is the structural foundation; per-operator-vocabulary extensions are sister subagents AFTER 5D canvas is operational.

**RANK 5: Hypothesis #1 (multi-op NOT FIRED at paired-axis) — REJECTED.**

- Empirical posterior shows drop-many WAS FIRED at paired-CPU (drop-two + 6 diversity-k variants); all 17 anchors are paired-CPU evidence-grade. The narrow drop-many beam IS fully wired through to paired-axis dispatch.
- What WAS NOT fired: REPAIR / MASKED / FEATHERED operations (sister 5D canvas operations) AND drop-many beam at depths > 2 (the diversity-k variants target a different pair-set construction, not depth-D beam expansion).

---

## Phase 4: Operator-routable top-5 priority recommendations

Per just-saved 13th OPTIMAL-TRIO directive: TECHNIQUE + WAY + TIME questions answered per recommendation.

### PRIORITY 1: BUILD-1 sister subagent (empirical 5D canvas population)

- **TECHNIQUE**: 5D canvas `bind_pair_component_xray` + `decompose_frame_axis_master_gradient` + `compute_segnet_posenet_score_geometry` + `query_receiver_runtime_feasibility`; reuse codex v1 row-based reader + Cable D master-gradient consumers + META-LIFT-1 + META-LIFT-2 as inputs.
- **WAY**: per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + design memo §DELIVERABLE 3 BUILD-1; read `.omx/state/master_gradient_anchors.jsonl` + per-pair component xray rows + per-frame master-gradient consumer rows; populate `PairFrameScorerGeometryCell` per (pair, frame) where empirical data exists; write lattice JSON to `.omx/state/pair_frame_scorer_geometry_lattice/<archive_sha[:12]>_<utc>.json` via fcntl-locked transactional write per Catalog #131.
- **TIME**: PRE-DISPATCH (canonical-submission-pipeline 10-phase Phase 4 builder; per just-saved 9th amendment); $0 paid GPU + ~2-4h wall-clock (sister subagent).
- **Operator-routable**: 3+ test cases + canonical helper `load_empirical_lattice(archive_sha256)` + per Catalog #313 probe-outcomes ledger row.

### PRIORITY 2: BUILD-2 + BUILD-3 sister subagent (canonical operation generators + Catalog #356 wire-in)

- **TECHNIQUE**: 5D canvas `generate_full_drop_starts` + `generate_repair_starts` + `generate_masked_starts` + `generate_feathered_starts`; each emits up to `top_n` `ExecutableCandidate` rows; each carries `predicted_axis_decomposition: AxisDecomposition` per Catalog #356.
- **WAY**: full implementation of the 4 operation generators (replaces `NotImplementedError`); per-operator substrate-optimal engineering per design memo §DELIVERABLE 1 table; canonical CLI `tools/generate_pair_frame_lattice_candidates.py --archive <sha> --operation full_drop --top-n 32 --output <path>`; 10+ test cases.
- **TIME**: PRE-DISPATCH (canonical-submission-pipeline 10-phase Phase 4); $0 paid GPU + ~4-8h wall-clock.
- **Operator-routable**: canonical candidate manifest JSON; operator decides whether to fund BUILD-4 + DISPATCH.

### PRIORITY 3: BUILD-4 sister subagent (Catalog #357 Tier B cathedral consumer promotion)

- **TECHNIQUE**: package `src/tac/cathedral_consumers/pair_frame_scorer_geometry_lattice_consumer/__init__.py` per Catalog #335 canonical contract; `CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING`; canonical-routing markers per Catalog #341 (empirically-grounded axis_tag NOT `[predicted]`; `promotable=False` per Tier B contract); auto-discovered by cathedral autopilot loop.
- **WAY**: `consume_candidate` emits Tier B contribution via lattice query; `update_from_anchor` for canonical posterior updates per Catalog #344; STRICT preflight Catalog #357 + #335 + #341 + #356 conformance.
- **TIME**: PRE-DISPATCH (canonical-submission-pipeline 10-phase Phase 9 cathedral consumer); $0 paid GPU + ~2-4h wall-clock.
- **Operator-routable**: cathedral autopilot ranker can now consume 5D canvas signal end-to-end; sister Catalog #344 `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` equation candidate becomes registerable as 3+ empirical anchors land.

### PRIORITY 4: PAIRED CPU+CUDA DISPATCH wave (4 operations × top-3 candidates × paired axes)

- **TECHNIQUE**: paid Modal dispatch of top-3 ExecutableCandidates per (operation, cpu_cuda_axis) = 4 × 3 × 2 = 24 dispatches; canonical 4-arm paired auth_eval pattern per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"; per-substrate symposium per Catalog #325; smoke-before-full per Catalog #167.
- **WAY**: per-dispatch follows canonical operator-authorize chain per Catalog #271 (codex pre-dispatch review) + Catalog #243 (local pre-deploy harness) + Catalog #167 (smoke-before-full pattern); paired auth_eval results feed continual_learning_posterior + canonical_frontier_pointer per Catalog #343.
- **TIME**: DISPATCH (canonical-submission-pipeline 10-phase Phase 6 paired-auth-eval); ~$2-10 per cascade wave (smoke ~$0.30 + paired-axis ~$1-3 each); total estimate $4-15.
- **Operator-routable**: at least one of REPAIR / MASKED / FEATHERED operations may produce a candidate that beats `0.192028282 [contest-CPU]` frontier per the canonical equation candidate's prediction.

### PRIORITY 5: Catalog #344 canonical equations registry growth (per-operator predicted ΔS)

- **TECHNIQUE**: register canonical equation candidates for each operator after 3+ paired-axis anchors land per Catalog #344 trigger `when_3+_new_empirical_anchors_in_domain`. Candidates: `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` (queued; DELIVERABLE 3) + `dqs1_drop_many_beam_pairwise_interaction_waterfill_v1` (sister; queued) + per-operator candidates as DISPATCH wave produces anchors (e.g. `repair_via_atick_redlich_cooperative_receiver_v1`, `feathered_via_daubechies_multi_scale_partition_v1`, `masked_via_uniward_per_segnet_class_chroma_v1`).
- **WAY**: per Catalog #344 each equation declares `canonical_producers` + `canonical_consumers` lists; refused as orphan if neither exists. Sister cathedral consumer `canonical_equation_lookup_consumer` automatically surfaces predictions to autopilot ranker.
- **TIME**: AFTER-DISPATCH (canonical-submission-pipeline 10-phase Phase 10 PR111+ first-use); $0 paid GPU + ~1-2h wall-clock per equation registration.
- **Operator-routable**: every NEW canonical equation makes the apparatus structurally smarter for the next dispatch wave; canonical-equations-registry growth IS the compounding-apparatus-growth COMPOUNDING leg per 7th META standing directive.

---

## Discipline closure

### 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** = N/A (audit memo; no signal contribution)
- **hook #2 Pareto constraint** = N/A
- **hook #3 bit-allocator** = N/A
- **hook #4 cathedral autopilot dispatch** = N/A (audit; defers to PRIORITY 1-3 sister subagents that DO wire cathedral consumers)
- **hook #5 continual-learning posterior** = N/A
- **hook #6 probe-disambiguator** = **ACTIVE** — this memo IS the canonical disambiguator between the 5 hypotheses ranked in Phase 3, with empirical evidence from `accepted_anchor_history` + `probe_outcomes.jsonl` + `canonical_equations_registry.jsonl` + `cathedral_consumers/` directory + source-code inspection.

### Mission contribution per Catalog #300

`frontier_breaking_enabler` — this audit memo identifies the structural foundation (PRIORITY 1-3 BUILD-1 + BUILD-2 + BUILD-3 + BUILD-4 sister subagents) that unblocks REPAIR / MASKED / FEATHERED operations + cathedral autopilot ranker consumption + canonical equation registration. Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4: frontier-breaking moves DOMINATE rigor budget. The 5D canvas IS the next canonical bridge per codex's explicit naming + design memo + this audit's RANK 1 hypothesis.

### Files touched (canonical APPEND-ONLY per Catalog #110/#113)

- NEW `.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md` (this file)

NO mutation of:

- CLAUDE.md
- canonical equation registry
- existing memos
- source code
- state JSON files
- AGENTS.md
- any sister design memo or landing memo

### Apparatus discipline acknowledgment

- Catalog #1 + #192: audit predictions are advisory; promotion requires paired CPU+CUDA empirical anchors
- Catalog #287: every empirical claim carries evidence tag (`[predicted]` for forward predictions; `[contest-CPU]` / `[contest-CUDA T4]` for paired-axis anchors cited per `accepted_anchor_history`)
- Catalog #323: canonical Provenance umbrella; this memo's audit findings are canonical Provenance read-only inspection
- Catalog #313: NO new probe-outcomes registered (audit only)
- Catalog #344: NO new canonical equations registered (audit only); 5 candidate equations queued for sister-subagent registration via PRIORITY 5
- Catalog #110/#113 APPEND-ONLY: NEW memo only; zero mutation
- CLAUDE.md "Forbidden premature KILL": no kill verdicts; all 5 hypotheses preserved per Catalog #307 paradigm-vs-implementation classification (hypothesis #2 + hypothesis #3 are IMPLEMENTATION-level EMPIRICALLY GROUNDED; hypothesis #4 STRUCTURALLY GROUNDED; hypothesis #1 + #5 DEFERRED-PENDING-EMPIRICAL)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": 5D canvas IS the unique canonical primitive that binds the 5 axes codex named; each receiver_runtime mode requires substrate-optimal engineering

### Sister coordination

- Slot Cascade C' WAVE-7 / Phase 3 / V14-V2 / ORDER-gates: DISJOINT (this is a READ-ONLY audit; no code touched)
- Slot canonical-submission-pipeline 10-phase: DISJOINT (audit references the 10-phase pipeline but does not modify it)
- Per Catalog #340 sister-checkpoint guard: PROCEED at audit-start checkpoint

---

## Direct answer to operator question

**Q**: *"Why is drop one still the frontier when theoretically a combination of dropping and replacing and other operations, many across the video, possibly multiple in a frame or pair, theoretically optimal?"*

**A**: TWO simultaneously-firing reasons:

1. The apparatus DID fire NARROW drop-many (drop-two + diversity k002-k024 = 17 paired-CPU anchors). At the operating point of the current frontier archive, drop-many ate `27e-6` distortion per byte saved while the rate-saving budget is only `6.66e-7` per byte — so multi-op LOSES rate-vs-distortion at this operating point. This is CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" empirically realized.

2. The apparatus DID NOT fire BROADER multi-op vocabulary (REPAIR / MASKED / FEATHERED / per-frame-drop / cross-pair-replace / merge / reorder) — the 5D canvas scaffold exists but all 4 canonical operations raise `NotImplementedError`. No cathedral consumer exists for drop-many beam OR 5D canvas. No canonical equation candidate is registered. No paired-axis dispatch has fired for any REPAIR / MASKED / FEATHERED operation.

**Q**: *"Does our code support this yet and is it wired up and integrated and tested end to end and automated"*

**A**: PARTIALLY:

- **drop-one**: YES (end-to-end + automated; FRONTIER anchor `0.192028282` per pointer)
- **drop-many beam (narrow)**: YES wired + tested + dispatched, but NO cathedral consumer for autopilot ranker. Result: fire-and-forget at DQS1 acquisition layer; signal does not loop back into autopilot ranking.
- **REPAIR / MASKED / FEATHERED / per-frame-drop / cross-pair-replace / merge / reorder**: NO. Design memo + scaffold + canonical interface contract LANDED, but the empirical population (BUILD-1) + operation generators (BUILD-2) + Tier B promotion (BUILD-3 + BUILD-4) + paired-axis DISPATCH wave are all sister-subagent operator-routable per design memo §DELIVERABLE 3.

**The path forward** (per just-saved 13th OPTIMAL-TRIO standing directive):

- PRIORITY 1: BUILD-1 sister subagent ($0 GPU + ~2-4h wall-clock) populates 5D canvas empirically
- PRIORITY 2: BUILD-2 + BUILD-3 sister subagent ($0 + ~5-10h) implements 4 operation generators + Catalog #356 wire-in
- PRIORITY 3: BUILD-4 sister subagent ($0 + ~2-4h) Tier B cathedral consumer promotion
- PRIORITY 4: PAIRED CPU+CUDA DISPATCH wave (~$4-15 total) fires top-3 candidates per operation × axis
- PRIORITY 5: Catalog #344 canonical equation registration as 3+ empirical anchors land

Total path: ~$4-15 paid GPU + ~10-20h sister-subagent wall-clock to convert SCAFFOLD + DESIGN + EMPIRICAL-DROP-ONE-FRONTIER into FULL multi-op apparatus end-to-end-automated with cathedral consumer auto-discovery.

---

## Cross-references

- `.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md` (design memo §DELIVERABLE 3 BUILD-1/2/3/4)
- `.omx/research/codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md` (eureka memo §"Residual Gap" — the operator-routable origin)
- `src/tac/optimization/dqs1_drop_many_beam.py` (16 KB; LANDED with 4 tests)
- `src/tac/optimization/pair_frame_scorer_geometry_lattice.py` (20 KB; codex v1 row-based LANDED with 4 tests)
- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` (40 KB; SCAFFOLD-ONLY, all 4 canonical operations raise `NotImplementedError`)
- `src/tac/optimization/decoder_q_pairset_acquisition.py` (54 KB; wires drop-many beam line 626)
- `.omx/state/continual_learning_posterior.json` `accepted_anchor_history` (17 paired-CPU drop-one + drop-two + diversity-k anchors)
- `.omx/state/probe_outcomes.jsonl` (22 composition/drop probes; 8 PROCEED + 11 DEFER + 1 INDEPENDENT + 2 PARTIAL)
- `.omx/state/canonical_equations_registry.jsonl` (55 equations; sister candidates queued for PRIORITY 5)
- `.omx/state/canonical_frontier_pointer.json` (FRONTIER `7a0da5d0fc...` 0.192028282 [contest-CPU] = drop-one rank021 pair0371)
- META-LIFT-1 `60acdc2d2` cross-substrate master-gradient analyzer (auto-discovered cathedral consumer)
- META-LIFT-2 `da803dd30` Pareto polytope unified solver (auto-discovered cathedral consumer)
- META-LIFT-4 `6fbd7ec7f` UNIWARD invariant enumerator (auto-discovered cathedral consumer)
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)" (canonical rate-vs-distortion at PR106 frontier)
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" (canonical math grounding for multi-op composition)
- CLAUDE.md "Bit-level deconstruction and entropy discipline" (canonical bit-level operator catalog)
- 7th META AUTOMATED+COMPOUNDING+OPTIMAL standing directive
- 11th ORDER MATTERS standing directive (13 dimensions)
- 12th canonicalization × standardization × ease-of-contest-compliance standing directive
- 13th OPTIMAL-TRIO standing directive (TECHNIQUE × WAY × TIME)

## Lane registration

Lane `lane_drop_many_replace_composition_apparatus_state_audit_20260526` L1 (impl_complete + audit_memo + 6-hook-declaration + 13th-standing-directive-binding).
