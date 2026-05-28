<!-- SPDX-License-Identifier: MIT -->
<!-- Council deliberation memo per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300 v2 frontmatter. -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, MacKay, Atick_Memorial, Rao, Ballard, Tishby_Memorial, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Cross-family PARITY at 3.4038 vs V3+Hinton 3.3963 is empirically convincing within 0.22% but the per-axis decomposition is EMPTY in per_epoch_metrics so we cannot decompose seg vs pose vs archive-bytes contributions; refusing to call this a definitive cooperative-receiver paradigm test until per-axis decomposition is populated."
  - member: Assumption-Adversary
    verbatim: "The cooperative-receiver paradigm in Z6-v2 architecture does include FoE ego-motion conditioning + 2-level Rao-Ballard FiLM but the in-training loss is dominated by reconstruction + Hinton-distilled scorer surrogate — the cooperative-receiver gradient binding at Catalog #311 happens DOWNSTREAM of the scorer loss. The HARD-EARNED conclusion: in-training convergence floor 3.40 reflects the scorer-bound gradient landscape SHARED with PACT-NeRV cascade; the cooperative-receiver paradigm's distinguishing signal MUST be sought elsewhere (archive-encode time per V2+V4+VQ verdict OR per-pair difficulty atlas OR class-shift inference)."
council_assumption_adversary_verdict:
  - assumption: "Hinton-distilled scorer-bound gradient is the dominant convergence driver across substrate families"
    classification: HARD-EARNED
    rationale: "5th empirical instance after V2/V3/V4/VQ (commits ab650cc78 / 1860ea2ac / 84a4893e4): scorer-bound gradient drives in-training convergence floor at 3.40 +/- 5% regardless of substrate-distinguishing primitive (PACT-NeRV IA3 vs SELECTOR variants vs Z6-v2 cooperative-receiver). Cross-family pattern stable across 5 architectures."
  - assumption: "Z6-v2 cooperative-receiver paradigm produces empirically DIFFERENT in-training convergence than PACT-NeRV per-method"
    classification: IMPLEMENTATION-LEVEL-FALSIFIED
    rationale: "Empirical receipt: Z6-v2 + Hinton + 600-pair = 3.4038 vs V3 + Hinton + 600-pair = 3.3963 (gap 0.0075 = 0.22%). PARITY confirmed within parity band [3.06, 3.74]. Per Catalog #307: this is IMPLEMENTATION-LEVEL FALSIFICATION of cross-family-differentiation-in-training NOT PARADIGM-LEVEL refutation of cooperative-receiver. The paradigm is INTACT; the differentiation surface is downstream of in-training (archive-encode-time + per-pair-difficulty + sub-frontier-inference)."
council_decisions_recorded:
  - "op-routable #1: Cross-family hypothesis CROSS-FAMILY PARITY confirmed; canonical equation #1 anchor 15 -> 16 (cross-family scope expansion to cooperative-receiver paradigm); canonical equation #344 NEW entry FORMALIZATION_PENDING per parity-floor-3.40-canonical-paradigm-independent emergent pattern (deferred until 6th + 7th cross-family anchor confirm)"
  - "op-routable #2: DEFERRED-PENDING-RESEARCH per CLAUDE.md 'Forbidden premature KILL' — Z6-v2 cooperative-receiver paradigm's distinguishing signal sought at archive-encode-time (V2+V4+VQ verdict consistency) + per-pair-difficulty-atlas + class-shift-inference surfaces; in-training-cross-family-differentiation IMPLEMENTATION-LEVEL falsified but paradigm INTACT"
  - "op-routable #3: Z6-v2 archive sha 5cdcdcca02ea5d25481a84e6d97c089775b5676926889a40d53c366fbeef20be (612,704 bytes) DEFERRED paired CUDA reactivation criterion per Catalog #246 — MLX-research-signal non-promotable; paired contest-CUDA + contest-CPU L2 dispatch is operator-routable IFF cooperative-receiver paradigm produces sub-frontier (<0.18) signal at the actual contest scorer (a follow-up subagent should test the archive on contest_auth_eval)"
  - "op-routable #4: per-axis-decomposition gap surfaced — RendererBundle.run_mlx_score_aware_full_main did NOT populate per_axis_decomposition in per_epoch_metrics; Contrarian VETO without seg/pose/archive_bytes attribution; sister-subagent should extend mlx_score_aware harness to populate the per-axis surface so cross-family + cooperative-receiver paradigm tests can decompose convergence attribution"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: z6_v2_cargo_cult_unwind_mlx_local
deferred_substrate_retrospective_due_utc: "2026-06-27T09:57:00Z"
related_deliberation_ids:
  - v2_v4_vq_hinton_distill_600pair_long_mlx_landed_20260528
  - z6_v2_cargo_cult_unwind_l1_long_run_mlx_landed_20260528
  - z6_v2_cargo_cult_unwind_design_20260527T053000Z
---

# Z6-v2 + Hinton-Distilled Scorer Surrogate + 600-pair LONG MLX-LOCAL LANDED 2026-05-28

**Cross-family hypothesis test**: cooperative-receiver paradigm (Rao-Ballard hierarchical predictive coding + ego-motion FoE conditioning + Atick-Redlich gradient binding per Catalog #311) under Hinton-distilled scorer-bound gradient.

**Operator mandate**: Z6-v2 + Hinton + 600-pair LONG MLX cross-family empirical test per V2+V4+VQ 600-pair parity landing operator-routable TOP-1 (cross-family substrate extension). $0 MLX-local non-promotable per Catalog #192/#127/#323 + 8th MLX-first standing directive REINFORCED 2026-05-28.

## Empirical results

| Metric | Value |
|---|---|
| Substrate | `z6_v2_cargo_cult_unwind_mlx_local` |
| Lane | `lane_z6_v2_cargo_cult_unwind_l1_long_run_mlx_local_20260528` |
| Epochs | 2000 |
| Pairs | 600 |
| Wall clock | 247.71 s |
| Learning rate | 1e-3 |
| Distillation weight | 0.5 |
| Pose distillation weight | 1.0 |
| Initial loss (epoch 0) | 107.5330 |
| Final loss (epoch 1999) | **3.4038** |
| Loss reduction | **31.59×** |
| log-log slope (epoch 100→1999) | -0.3466 |
| First epoch loss < parity band UPPER (3.74) | epoch 175 |
| First epoch loss < parity band LOWER (3.06) | NEVER (saturated at floor) |
| EMA drift L2 (final) | 1.7777 |
| Archive sha256 | `5cdcdcca02ea5d25481a84e6d97c089775b5676926889a40d53c366fbeef20be` |
| Archive bytes | 612,704 |
| Promotable | False per Catalog #192/#317/#341 |
| Axis tag | `[macOS-MLX research-signal]` |

## Cross-family verdict per Catalog #307

**VERDICT: CROSS-FAMILY PARITY** (paradigm-level INTACT; implementation-level FALSIFIED for cross-family-differentiation-in-training hypothesis).

**Empirical receipts** (cumulative cross-family Hinton-distilled parity floor):

| Substrate | Family | Hinton + 600-pair final loss | Gap vs Z6-v2 |
|---|---|---|---|
| Z6-v2 (THIS RUN) | cooperative-receiver | **3.4038** | — |
| V3 + Hinton + 600-pair | PACT-NeRV SELECTOR | 3.3963 | +0.0075 (+0.22%) |
| V2/V4/VQ + Hinton + 600-pair PARITY BAND | PACT-NeRV SELECTOR | ~3.40 (band 3.06-3.74) | within band |
| PACT-NeRV-IA3 + Hinton (sister IA3 cascade) | PACT-NeRV IA3 | ~3.40 band | within band |

**5th empirical instance** of cross-family parity: scorer-bound gradient under Hinton-distilled scorer surrogate drives in-training convergence to a SHARED floor (~3.40 +/- 5%) regardless of substrate-distinguishing primitive (IA3 γ-only ego-pose modulation vs SELECTOR variants vs cooperative-receiver Rao-Ballard hierarchical FiLM-ego-motion + FoE conditioning).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation classification:
- **Paradigm INTACT**: cooperative-receiver mathematical structure (Atick-Redlich 1990 + Rao-Ballard 1999 + Wyner-Ziv 1976) is NOT refuted. The paradigm's distinguishing signal is sought downstream of in-training (archive-encode-time + per-pair difficulty atlas + sub-frontier inference).
- **Implementation-level FALSIFIED**: the cross-family-differentiation-in-training hypothesis (that cooperative-receiver paradigm under Hinton-distilled scorer-bound gradient would produce empirically DIFFERENT convergence signature than PACT-NeRV parity floor) is empirically falsified at 0.22% deviation.

## Cargo-cult unwind per Catalog #303

The original cargo-cult-unwind design memo (`.omx/research/z6_v2_cargo_cult_unwind_design_20260527T053000Z.md`) targeted FOUR cargo-cults from Z6-v1: (1) `np.roll` global translation → FoE ego-motion conditioning HARD-EARNED; (2) single-level FiLM → 2-level Rao-Ballard hierarchical HARD-EARNED; (3) scorer-naive loss → Atick-Redlich cooperative-receiver gradient binding HARD-EARNED via Catalog #311; (4) pixel-MSE objective → Hinton-distilled KL T=2.0 scorer surrogate THIS LANDING.

Per the L1 LONG RUN landing memo (`z6_v2_cargo_cult_unwind_l1_long_run_mlx_landed_20260528`) the pure-pixel-MSE 32-pair baseline achieved 20.45× / 0.0163 final loss — STRUCTURAL FLOOR with NO scorer binding. THIS landing wires scorer binding via Hinton-distilled per Catalog #164 + replicates cross-family parity at 600-pair.

## 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: cooperative-receiver paradigm (Rao-Ballard + Atick-Redlich + Wyner-Ziv triple) is the substrate-distinguishing primitive per Catalog #272.
2. **BEAUTY + ELEGANCE**: 113-line trainer wire-in adds Hinton-distilled binding via canonical 4-helper pattern (`build_mlx_segnet_pair_teacher` + `build_mlx_posenet_pair_teacher` + `build_learnable_student_head` + `build_learnable_pose_student_head`); single commit reviewable in 30 seconds.
3. **DISTINCTNESS**: Z6-v2 architecture explicitly different from PACT-NeRV cascade (2-level FiLM + FoE conditioning + cooperative-receiver gradient binding); cross-family parity verdict reflects in-training scorer-bound gradient SHARED-floor not architectural identity.
4. **RIGOR**: empirical convergence curve across 2000 epochs + log-log slope -0.3466 + parity band crossing at epoch 175 + final 3.4038 + cumulative 5-instance cross-family evidence.
5. **OPTIMIZATION PER TECHNIQUE**: ADOPT_CANONICAL training loop + EMA + score-aware loss harness; FORK Z6-v2-specific 2-level Rao-Ballard FiLM-ego-motion predictor + FoE conditioning + cooperative-receiver gradient binding per Catalog #290.
6. **STACK-OF-STACKS COMPOSABILITY**: cooperative-receiver paradigm orthogonal to Hinton-distilled scorer surrogate via canonical bundle field threading.
7. **DETERMINISTIC REPRODUCIBILITY**: seed=0 + canonical fcntl-locked posterior + canonical helper invocations + archive sha256 stable.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 247s wall-clock for 2000ep + 600 pairs + dual-teacher Hinton + 2-level hierarchical FiLM on M5 Max @ $0 GPU.
9. **OPTIMAL MINIMAL CONTEST SCORE**: in-training proxy 3.40 ≈ parity floor; sub-frontier contest score TBD pending operator-routable DEFERRED paired CUDA reactivation (NOT promotable per Catalog #192).

## Observability surface per Catalog #305

- **Inspectable per layer**: telemetry JSONL at `.omx/research/z6_v2_hinton_distill_600pair_long_mlx_20260528T095121Z/telemetry.jsonl` (613.2KB)
- **Decomposable per signal**: `per_epoch_metrics` populated 2000 rows (loss + ema_drift_l2 + lr + wall_clock); per_axis_decomposition GAP surfaced as op-routable #4
- **Diff-able across runs**: archive sha256 byte-stable; cross-family comparison v2/v3/v4/vq/z6-v2 enabled via canonical posterior anchors
- **Queryable post-hoc**: training_artifact.json (744KB) + telemetry.jsonl + EMA shadow checkpoint
- **Cite-able**: canonical Provenance `{kind: predicted_from_model, evidence_grade: predicted, axis_tag: [macOS-MLX research-signal]}` per Catalog #323
- **Counterfactual-able**: archive bytes available for byte-mutation smoke per Catalog #139

## Predicted ΔS band per Catalog #296

Per Dykstra-feasibility convex-intersection: cooperative-receiver paradigm shares scorer-bound gradient feasibility surface with PACT-NeRV cascade in-training (5-instance empirical evidence). At MLX-local proxy convergence floor 3.40, the predicted MLX-research-signal contest score band derived from Shannon R(D) bound + PACT-NeRV cascade calibration is [0.18, 0.21] **[predicted; not contest-CUDA / contest-CPU; reactivation criterion paired auth-eval at operator routable]** per Catalog #324 post-training Tier-C validation discipline.

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Training loop | ADOPT_CANONICAL | `mlx_score_aware.run_mlx_score_aware_full_main` is per-substrate-agnostic and operates correctly for cooperative-receiver paradigm |
| EMA | ADOPT_CANONICAL | EMA discipline universal per CLAUDE.md "EMA - NON-NEGOTIABLE" |
| Score-aware loss harness | ADOPT_CANONICAL | `mlx_score_aware` Hinton wire-in identical to PACT-NeRV cascade |
| Hinton-distilled KL T=2.0 | ADOPT_CANONICAL | Hinton-Vinyals-Dean 2014 standard |
| 2-level Rao-Ballard FiLM | FORK_BECAUSE_PRINCIPLED_MISMATCH | Z6-v2 UNIQUE primitive per Catalog #272 |
| FoE ego-motion conditioning | FORK_BECAUSE_PRINCIPLED_MISMATCH | Z6-v2 UNIQUE primitive per Catalog #272 |
| Atick-Redlich gradient binding | FORK_BECAUSE_PRINCIPLED_MISMATCH | Cooperative-receiver paradigm per Catalog #311 |
| Provenance | ADOPT_CANONICAL | `tac.provenance.builders.build_provenance_for_predicted` |
| Posterior anchor | ADOPT_CANONICAL | `tac.council_continual_learning.append_council_anchor` |

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| Hinton-distilled scorer-bound gradient drives in-training convergence floor | HARD-EARNED | 5th empirical cross-family instance |
| Cooperative-receiver paradigm produces cross-family-differentiation-in-training | CARGO-CULTED | Empirically falsified at 0.22% deviation; differentiation surface is downstream of in-training |
| Z6-v2's 2-level FiLM + FoE conditioning is the substrate-distinguishing primitive | HARD-EARNED | Catalog #272 contract; Z6-v2 architecture-distinct from PACT-NeRV cascade |
| In-training convergence floor reflects contest score floor | CARGO-CULTED | MLX-research-signal is reconstruction-proxy NOT contest-bound; reactivation criterion paired CUDA |

## Operator-routable

1. **TOP-1**: Canonical equation #1 anchor 15 → 16 (cross-family scope expansion); see canonical_equations registry update below.
2. **DEFERRED-PENDING-RESEARCH** per CLAUDE.md "Forbidden premature KILL": Z6-v2 cooperative-receiver paradigm distinguishing signal sought at archive-encode-time + per-pair-difficulty-atlas + sub-frontier-inference surfaces.
3. **DEFERRED paired CUDA reactivation** per Catalog #246: archive sha `5cdcdcca02ea5d25481a84e6d97c089775b5676926889a40d53c366fbeef20be` (612,704 bytes) is non-promotable MLX-research-signal; paired contest-CUDA + contest-CPU L2 dispatch is operator-routable IFF sub-frontier (<0.18) anchor required.
4. **per_axis_decomposition gap** surfaced as op-routable #4: `RendererBundle.run_mlx_score_aware_full_main` did NOT populate per_axis_decomposition in per_epoch_metrics; sister-subagent should extend `mlx_score_aware` harness to populate the per-axis surface so cross-family + cooperative-receiver paradigm tests can decompose convergence attribution (Contrarian VETO without seg/pose/archive_bytes per-axis attribution).

## Mission contribution per Catalog #300

`frontier_protecting`: extincts cross-family-differentiation-in-training hypothesis empirically; preserves cooperative-receiver paradigm intact for downstream tests; canonical equation #1 anchor 16 compounds the cross-family scorer-bound gradient parity floor as solver prior for future substrate dispatches.

## Sister cross-references

- L0 SCAFFOLD: `.omx/research/z6_v2_cargo_cult_unwind_design_20260527T053000Z.md` (commit `afa5ba837`)
- L1 LONG RUN baseline (pure pixel-MSE 32-pair): `.omx/research/z6_v2_cargo_cult_unwind_l1_long_run_mlx_landed_20260528.md` (commit `16c0e75bd`)
- V2+V4+VQ 600-pair parity landing: `.omx/research/v2_v4_vq_hinton_distill_600pair_long_mlx_landed_20260528.md` (commit `84a4893e4`)
- V3 + Hinton + 600-pair sister: commit `ab650cc78`
- V2/V4/VQ cascade Hinton wire-in: commit `1860ea2ac`
- IA3 + Hinton canonical pattern: commit `b551bfd34`
- Z6-v2 trainer wire-in: commit `1a855a9a6` (THIS landing)
- Canonical equation #1 registry: `tac.canonical_equations.canonical_equations_registry::hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` (anchor 15 → 16)
- CLAUDE.md non-negotiables binding this landing: "MLX portable-local-substrate authority" / "Forbidden premature KILL without research exhaustion" / "EMA - NON-NEGOTIABLE" / "Submission auth eval - BOTH CPU AND CUDA" / "Subagent coherence-by-default" / "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
- Catalogs cited: #1 / #110 / #113 / #127 / #131 / #164 / #176 / #185 / #192 / #206 / #229 / #246 / #265 / #270 / #272 / #287 / #292 / #294 / #296 / #300 / #303 / #305 / #307 / #311 / #313 / #317 / #323 / #324 / #325 / #335 / #340 / #341 / #344 / #346 / #348 / #361 / #371

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map contribution**: per-epoch loss trajectory + EMA drift L2 in telemetry.jsonl; per-axis decomposition GAP surfaced as op-routable #4
- **hook #2 Pareto constraint**: cross-family parity floor 3.40 +/- 5% added to substrate-composition matrix per `tac.optimization.substrate_composition_matrix`
- **hook #3 bit-allocator hook**: archive bytes 612,704 + sha 5cdcdcca02ea5d25481a84e6d97c089775b5676926889a40d53c366fbeef20be for per-byte sensitivity backfill per `tac.master_gradient_consumers`
- **hook #4 cathedral autopilot dispatch hook**: canonical posterior anchor + cross-family parity-band augmentation to autopilot ranker priors
- **hook #5 continual-learning posterior update**: append_council_anchor + canonical equation #1 anchor 15 → 16 + Catalog #371 auto-recalibration trigger
- **hook #6 probe-disambiguator**: cross-family parity verdict IS the canonical disambiguator between (a) cross-family-differentiation-in-training hypothesis (FALSIFIED) and (b) cooperative-receiver paradigm INTACT downstream (DEFERRED to archive-encode-time / per-pair-difficulty-atlas / sub-frontier-inference)

## Discipline

Catalog #229 PV (read full state of Z6-v2 trainer + canonical Hinton patterns + V2+V4+VQ verdict pre-edit) + Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256 + Catalog #206 (4 checkpoints) + Catalog #110/#113 APPEND-ONLY (NEW landing memo; trainer wire-in adds new code; zero mutation of historical artifacts) + Catalog #230 sister-subagent ownership map (no overlap; this is the only Z6-v2 work) + Catalog #287 placeholder-rationale rejection (all rationales ≥4 chars) + Catalog #340 sister-checkpoint guard (own-checkpoint mark-complete-retry pattern) + CLAUDE.md "Subagent coherence-by-default" + "MLX portable-local-substrate authority". $0 GPU verified throughout (MLX-local M5 Max only; non-promotable per Catalog #192/#317/#341).

## Lane

`lane_z6_v2_cargo_cult_unwind_hinton_distill_600pair_long_mlx_20260528` L1 (impl_complete + memory_entry + canonical_posterior_anchor + canonical_equation_refinement).
