---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - AssumptionAdversary
  - Hinton
  - Schmidhuber
  - Hassabis
  - Karpathy
  - Selfcomp
  - Quantizr
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
horizon_class: plateau_adjacent
deliberation_id: per_substrate_symposium_dp1_deep_dive_20260517
deferred_substrate_id: pretrained_driving_prior
substrate_alias: dp1
predicted_band_validation_status: pending_post_training
council_dissent:
  - member: AssumptionAdversary
    verbatim: "operating-within: DP1 prior composes monotonically across the 6-substrate reuse graph. Classification: CARGO-CULTED. Rationale: ZERO empirical anchor exists for ANY DP1 composition cell (A1/PR101/HDM8/YUCR/TT5L/sane_hnerv). The Phase 2 hardening council's Round 1 Dykstra concern (interaction term unknown; additivity assumed at the upper bound) was NEVER MEASURED. Per #864 META: cargo-cult-unwind methodology does NOT compose monotonically across architectural changes. The composability-as-stated assumption inherits that risk class — DP1+substrate-X composition is structurally analogous to v6->v8 architectural transition. CROSS-SUBSTRATE COMPOSABILITY AUDIT REQUIRED before any new DP1 dispatch fires."
  - member: Contrarian
    verbatim: "The reuse claim has been unverified for 4 days. PATH 1 (DP1+fec6) is BUILT but unfired; PATH 2 (PR101+DP1) is SCAFFOLDED with NotImplementedError. The composition API has 6 known_base_substrates registered, ZERO of which have measured DP1+base ΔS. This is the largest false-authority surface in the contest — DP1 is cited as a reuse harness across the autopilot landscape but has no per-substrate empirical receipt. Refuse to PROCEED a full multi-substrate dispatch until ONE composition cell lands a paired-axis empirical anchor."
  - member: Yousfi
    verbatim: "operating-within: the contest scorer's pre-trained driving knowledge implicitly contains the dashcam prior, so DP1's incremental signal is bounded. Classification: HARD-EARNED. Rationale: predicted_delta [-0.005, -0.012] reflects exactly this redundancy. DP1 only HELPS where the scorer's training set missed Comma2k19-specific structure (lighting / road-surface micro-textures). For the operator's purpose of breaking out of the 0.196-0.199 plateau, DP1 is plateau-adjacent at best — frontier_protecting (helps lock in current best), not frontier_breaking. Prioritize ASYMPTOTIC pursuit substrates (Z6/Z7/Z8/C6/ATW v2) for plateau exit."
council_assumption_adversary_verdict:
  - assumption: "DP1 codebook prior is HARD-EARNED dashcam-distribution structure"
    classification: HARD-EARNED
    rationale: "Catalog #209 + #210 + #213 enforce canonical Comma2k19 chunking + license-tag propagation + provenance-metadata. Distillation is OFFLINE per CLAUDE.md HNeRV parity L1; the contest video is structurally refused by `check_no_contest_video_leakage`. The codebook design (PCA basis + sky-horizon vertical profile + vehicle appearance) is principled and the byte-level archive grammar is verified."
  - assumption: "The composition wrapper API (compose_with / decompose / verify_composition) is the canonical reuse surface"
    classification: HARD-EARNED
    rationale: "Catalog #211 STRICT preflight enforces canonical routing; the 13-byte DPCOMP header is forensically auditable; 6 known_base_substrates registered + Phase 2 hardening tested compose/decompose byte-stability across A1/HDM8/YUCR/TT5L/sane_hnerv. The API contract is sound."
  - assumption: "Adding DP1 prior produces incremental contest-CPU score improvement"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "The predicted band [-0.005, -0.012] was inherited from operator's 2026-05-13 strategic memo (NASA-PASS-AI rank 3 candidate) WITHOUT a Dykstra-feasibility intersection check per Catalog #296 or a probe-disambiguator path. Phase 2 hardening Round 1 Dykstra EXPLICITLY flagged the interaction-term assumption as untested. ZERO empirical anchors exist across 4 calendar days post-Phase-2-landing. The PATH 1 packet IS LANDED but UNFIRED."
  - assumption: "DP1 prior composes monotonically across the 6-substrate reuse graph"
    classification: CARGO-CULTED-EMPIRICALLY-UNTESTED
    rationale: "Per Item #8 #864 META: cargo-cult-unwind methodology does NOT compose monotonically across architectural changes. The SAME META risk applies to DP1 reuse across A1 (HNeRV-family) / PR101 (HNeRV-family) / HDM8 / YUCR (UNIWARD cost-map) / TT5L (Time-Traveler L5) / sane_hnerv. These 6 substrates have RADICALLY DIFFERENT internal grammars; assuming the SAME DP1 prior signal carries through each is the structural mirror of the v6->v7->v8 monotonicity assumption that was empirically disproved at 78% regression. CROSS-SUBSTRATE COMPOSABILITY AUDIT is required before declaring DP1 reuse-graph-coherent."
  - assumption: "PATH 1 L1 packet (PACT_DP1_PRIOR_STRENGTH=0.0) is meaningful empirical evidence even at zero strength"
    classification: HARD-EARNED-PROVISIONAL
    rationale: "L1 measures the rate-axis cost in isolation (+25.8 KB DP1 prefix -> +0.0000172 rate term). This IS a structurally meaningful measurement because it isolates one degree of freedom. But it's NOT a measurement of DP1's frame-axis effect; L2 INTEGRATION (strength > 0) is the load-bearing test. Treat L1 as the rate-axis baseline anchor, NOT a DP1 ΔS claim."
  - assumption: "DP1 frame-prior on PR101_lc_v2 decoder weights (PATH 2) is the correct architectural integration"
    classification: HARD-EARNED-ARCHITECTURALLY-VERIFIED
    rationale: "Premise PV-6 verified pre-edit: fec6 has no learned decoder weights, so frame-space DP1 prior cannot regularize fec6 in weight-space. Pointing the prior at PR101_lc_v2 (which DOES have learned decoder weights) is principled per HNeRV parity L7 (substrate engineering). However, the INTEGRATION SURFACE (where in the trainer's loss-hook to inject DashcamPriorLoss) is council Phase 2 grade and remains undecided."
council_decisions_recorded:
  - "op-routable #1: CROSS-SUBSTRATE COMPOSABILITY AUDIT (highest priority): require a single READ-ONLY $0 audit subagent that (a) classifies each of the 6 known_base_substrates by integration grade (architectural-fit / partial-fit / structural-mismatch), (b) checks that the SAME DP1 codebook bytes flow into each composition path identically, (c) verifies license-tag propagation per Catalog #210 spot-check on 3 substrates, (d) builds the matrix of (substrate × composition_path × predicted_ΔS_basis) so dependent substrates inherit grounded expectations. Required BEFORE any new DP1 dispatch fires."
  - "op-routable #2: dispatch PATH 1 paired-axis ($1.90 envelope) at the EARLIEST AVAILABLE slot per CLAUDE.md 'Apples-to-apples evidence discipline'. This is the cheapest, highest-information first empirical anchor: confirms rate-axis cost matches the closed-form Shannon arithmetic prediction (+0.0000172) AND establishes paired-axis byte-identity reproducibility on the composed packet. PRE-DISPATCH requires (a) Catalog #324 predicted_band_validation_status declared `pending_post_training` (already done in recipe); (b) Catalog #314 absorption-pattern verification on the build commit; (c) `# PREDICTED_BAND_VIBES_OK` waiver NOT needed because rate-axis prediction is closed-form Shannon arithmetic, not heuristic."
  - "op-routable #3: PATH 2 (PR101+DP1) deferred to council Phase 2 PROCEED. _full_main remains NotImplementedError; reactivation criteria documented in dp1_dual_stacking_design_20260517.md. Per the 5 acceptance cascades of Catalog #315 substrate-must-be-at-optimal-form-before-paid-empirical-dispatch: PATH 2 trainer satisfies (b) research_only=true opt-out."
  - "op-routable #4: register the DP1 prior FIRST EMPIRICAL ANCHOR (after PATH 1 paired-axis lands) via tac.probe_outcomes_ledger.register_probe_outcome with verdict PROCEED (canonical first-anchor) OR PARTIAL (rate-axis-only without L2-integration). The autopilot ranker can then weight DP1 composition candidates appropriately per Catalog #313."
  - "op-routable #5: DEFER L2 INTEGRATION (PACT_DP1_PRIOR_STRENGTH > 0) until probe-disambiguator + Dykstra-feasibility helper land per Catalog #296. The current predicted lower bound (-0.012) is heuristic extrapolation, NOT first-principles-citable. Without Dykstra intersection check, L2 dispatch risks the same predicted-band miss class as C6 IBPS (22× outside band)."
  - "op-routable #6 (frontier_protecting): adjust autopilot ranker via tac.cathedral_autopilot_autonomous_loop to treat DP1 composition candidates as plateau-adjacent (Yousfi verdict) UNLESS the cross-substrate composability audit (#1) surfaces a substrate × DP1 combination that empirically lands sub-0.188. The default ranking should not amplify DP1 composition expected ΔS until empirical anchor exists."
deferred_substrate_retrospective_due_utc: 2026-06-16T00:00:00Z
related_deliberation_ids:
  - per_substrate_symposium_lane_17_imp_20260517
  - per_substrate_symposium_pr106_05_06_reformulated_20260517
  - per_substrate_symposium_nscs06_v8_path_b_20260517
  - per_substrate_symposium_z7_lstm_predictive_coding_20260517
  - per_substrate_symposium_atw_v2_reactivation_20260518
---

# Per-Substrate Symposium: DP1 (Pretrained Driving Prior) Deep-Dive

**Lane:** `lane_per_substrate_symposium_dp1_deep_dive_20260517`
**Date:** 2026-05-17 (Wave 3 #855)
**Substrate:** `pretrained_driving_prior` (DP1) — canonical reseed substrate
**Dependent substrates:** A1, PR101, HDM8, YUCR, TT5L, sane_hnerv (6 known_base_substrates registered in `_KNOWN_BASE_TAGS`)
**Tier:** T3 (CLAUDE.md non-negotiable interpretation — DP1 is reused by 6 substrates; per Catalog #325 a T3-tier-elevated cross-cutting wire-in is appropriate; sextet pact + 6 grand-council attendees including Hinton/Schmidhuber/Hassabis for distillation expertise + Karpathy/Selfcomp/Quantizr for substrate-consumer perspective.)
**Cost:** $0 (read-only symposium; no GPU spend)

## 2026-05-18 supersession: PATH 1 rate arithmetic corrected

The DP1+fec6 L1 no-op probe landed after this symposium:
`.omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json`.

It verifies the composed packet is structurally byte-closed, but corrects the
PATH 1 rate-axis arithmetic:

- DP1 prefix bytes: `25814`
- DPCOMP header bytes: `13`
- total added bytes: `25827`
- contest rate delta if frames are identical:
  `25 * 25827 / 37545489 = +0.017197139182`

Therefore every older `+0.0000172`, `[+0.0000160, +0.0000180]`, and "earliest
available dispatch" statement below is superseded for operator action. PATH 1
is now a research-only no-op/byte-closure control unless the operator explicitly
wants the paired control measurement. L2 DP1 prior-effect work must overcome a
`+0.017197` score penalty before it can be promotion- or ranking-relevant.

## 1. Cargo-cult audit per assumption

Per CLAUDE.md Catalog #303 + hard-earned-vs-cargo-culted addendum. See frontmatter `council_assumption_adversary_verdict` for the canonical 6-classification table. Summary tally: **3 HARD-EARNED (codebook prior / composition API / PATH 2 architectural fit) + 1 HARD-EARNED-PROVISIONAL (PATH 1 rate-axis L1 baseline) + 2 CARGO-CULTED-PENDING-EMPIRICAL (predicted ΔS band / cross-substrate composability)**.

The 2 CARGO-CULTED assumptions ARE the load-bearing claims for the entire DP1 strategic position. Both are blocked by the SAME empirical gap: ZERO DP1+base composition cells have ever fired.

### Per-assumption unwind paths

1. **DP1 codebook prior is HARD-EARNED** — no unwind needed; provenance is canonical per #209/#210/#213.
2. **Composition wrapper API is HARD-EARNED** — no unwind needed; Catalog #211 enforces routing.
3. **DP1 produces incremental score improvement** — unwind requires dispatching PATH 1 paired-axis (rate-axis baseline; +0.0000172 closed-form prediction). If predicted == empirical within numerical noise, the L1 measurement is the canonical rate-axis cost anchor. L2 INTEGRATION then becomes the frame-axis ΔS measurement; requires probe-disambiguator + Dykstra-feasibility helper before $5-10 dispatch.
4. **Composes monotonically across reuse graph** — unwind requires CROSS-SUBSTRATE COMPOSABILITY AUDIT (op-routable #1) BEFORE any new dispatch. The audit's matrix `(substrate × composition_path × predicted_ΔS_basis)` is the canonical disambiguator. Without it, every dependent substrate inherits an untested DP1+base assumption.
5. **L1 zero-strength packet is meaningful** — already provisionally HARD-EARNED; PATH 1 dispatch (op-routable #2) confirms or disconfirms in $1.90.
6. **PATH 2 PR101_lc_v2 integration** — unwind handled by council Phase 2; current state is correct DEFER per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY".

## 2. 9-dimension success checklist evidence

Per CLAUDE.md Catalog #294.

| Dim | Verdict | Evidence |
|---|---|---|
| 1. UNIQUENESS (class-shift) | **PARTIAL** | DP1 IS a class-shift PRIMITIVE (OOD pretrained prior vs in-distribution contest overfit). However, its OPERATIONAL use is plateau-adjacent (frontier_protecting per Yousfi): the contest scorer already contains a driving prior, so DP1's incremental signal is bounded at -0.012 upper bound. Class-shift in concept; within-class in score impact band. |
| 2. BEAUTY + ELEGANCE | **HARD-EARNED** | composition.py is 396 LOC (within HNeRV parity L7 ~350 LOC bolt-on budget for the consumer surface); 13-byte DPCOMP header is the minimum sufficient cooperative-receiver wrapper; one function name (`compose_with`), one inverse (`decompose`), one verification (`verify_composition`). |
| 3. DISTINCTNESS | **HARD-EARNED** | Per the operator's "use it over and over" directive, DP1 IS explicitly designed as a reuse harness, NOT a single-substrate experiment. Distinct from every other substrate by being the only one with a documented compose_with API for 6 base substrates. |
| 4. RIGOR | **PARTIAL** | 5-round adversarial council (Phase 2 hardening, 2026-05-14) is canonical-strength rigor at the DESIGN surface. Forensic audit memo `dp1_forensic_audit_and_roadmap_20260515_codex.md` is exhaustive at the engineering surface. HOWEVER: ZERO empirical anchor across 4 days. Rigor at the DESIGN surface without empirical anchor at the DISPATCH surface = research-only-prior-pending-empirical-anchor (per Catalog #240 / #315). |
| 5. OPTIMIZATION PER TECHNIQUE | **HARD-EARNED** | Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: codebook design is substrate-optimal (PCA basis sized to MDL bound 5-10 KB post-brotli per Round 2 MacKay verdict); inflate-time consumer is the minimum sufficient prior-application module. No detected canonical-suppression pattern. |
| 6. STACK-OF-STACKS-COMPOSABILITY | **CARGO-CULTED-UNTESTED** | The reuse-graph composability assumption is the load-bearing CARGO-CULTED claim per Assumption-Adversary. PATH 1 (composition) and PATH 2 (training-time prior) are mathematically distinct stacking grammars; both are theoretically composable but ZERO empirical anchors exist. The cross-substrate audit (op-routable #1) is the canonical unwind path. |
| 7. DETERMINISTIC REPRODUCIBILITY | **HARD-EARNED** | Catalog #19 deterministic ZIP + Catalog #210 codebook provenance metadata (`random_seed`, `basis_sha256`, `dataset_provenance`, `license_tags`) + `compose_with` is byte-stable (tested in `test_composition.py`). |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | **PARTIAL** | PATH 1 dispatch is $1.90 (T4 + CPU paired); PATH 2 SCAFFOLDED at $0 (CPU smoke). Codebook is brotli-optimal (Round 2 Ballé hyperprior recommendation deferred to Phase 3). No GTScorerCache wire-in detected in DP1 trainer; that's a Tier 1 engineering gap per Catalog #228. |
| 9. OPTIMAL MINIMAL CONTEST SCORE | **PARTIAL** | Predicted band [-0.005, -0.012] from fec6 baseline 0.19205 → 0.180-0.189 target. If achieved, this would BEAT public PR101 GOLD 0.193 by 0.013 CPU. Per Yousfi: plateau-adjacent ≠ frontier-breaking; DP1 is the right tool to lock in current frontier, NOT the right tool for the operator's stated frontier_pursuit goal. |

## 3. Observability surface

Per CLAUDE.md Catalog #305 + max-observability standing directive (6 facets).

| Facet | DP1 substrate + reuse surface |
|---|---|
| **Inspectable per layer** | Codebook sections inspectable via `parse_archive` (4 PCA components surfaces); composition wrapper inspectable via `decompose` (returns ComposedArchive with explicit `dp1_archive_bytes`, `base_archive_bytes`, `base_substrate`, `schema_version`); `archive_manifest.json` declares per-section bytes; lane registry carries 7-gate per-lane state. |
| **Decomposable per signal** | Score-aware loss decomposed: `lambda_road_plane * road_loss + lambda_sky_horizon * sky_loss + lambda_vehicle * veh_loss`; each term loggable separately. Predicted ΔS decomposed: rate-axis (+0.0000172 closed-form) + frame-axis (heuristic [-0.005, -0.012] PENDING Dykstra-feasibility). PATH 1 vs PATH 2 cleanly separable mathematical types (inflate-time composition vs compress-time prior). |
| **Diff-able across runs** | Composed archive sha256 byte-stable per `test_sha256_byte_stable`; codebook `basis_sha256` carried in metadata for tampering detection. Currently DEGRADED on the reuse-graph surface: no two dispatches have ever produced comparable empirical anchors for diff. |
| **Queryable post-hoc** | `archive_manifest.json` + `build_manifest.json` + `provenance.json` machine-readable; `.omx/state/lane_registry.json` carries DP1 lane gates; canonical posterior at `.omx/state/council_deliberation_posterior.jsonl` will carry this symposium anchor. **GAP**: ZERO entries in canonical posterior for ANY DP1 deliberation pre-this-memo (Phase 2 hardening council never wrote to posterior). |
| **Cite-able** | DP1 codebook anchored to `(dataset_provenance, basis_sha256, random_seed, distillation_version)` per Catalog #210; composition anchored to `(composed_sha256, dp1_source_sha256, base_source_sha256, build_tool_commit)`. |
| **Counterfactual-able** | `tools/verify_distinguishing_feature_byte_mutation.py` could mutate codebook bytes and verify inflated frames change; PATH 1 L2 `PACT_DP1_PRIOR_STRENGTH` env-var enables counterfactual sweeps. |

**Observability gaps identified:**

1. The 5-round Phase 2 hardening council (2026-05-14) memo exists in `.omx/research/` but was NEVER written to `.omx/state/council_deliberation_posterior.jsonl`. The continual-learning posterior has ZERO DP1 deliberation history. This symposium memo lands the first DP1 anchor in the canonical posterior.
2. The 6 `_KNOWN_BASE_TAGS` substrates have no measured composability matrix — `dependent_substrate × DP1_composition_path × predicted_ΔS_basis` is absent from any machine-readable artifact. The cross-substrate audit (op-routable #1) is the canonical observability landing.

## 4. Sextet / grand council deliberation

Sextet pact + grand council attendees per the substrate's reuse + distillation domain.

### Round 1 verdict (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary + Hinton + Schmidhuber + Hassabis + Karpathy + Selfcomp + Quantizr)

Per-member explicit operating-within assumption (Catalog #292 / CLAUDE.md "Council conduct" Fix 7):

- **Shannon LEAD** — operating-within: DP1 is a conditional-entropy reduction primitive. The codebook absorbs OOD dashcam structure so the renderer's contest-specific bits-budget is `H(contest | DP1_codebook) < H(contest)`. **Verdict**: PROCEED PATH 1 paired-axis (rate-axis baseline is the canonical first measurement). DEFER PATH 2 to Phase 2 council. The cross-substrate composability question requires the audit per op-routable #1.
- **Dykstra CO-LEAD** — operating-within: DP1+base composition is alternating-projection onto two feasible sets `F_DP1 ∩ F_base`. The wrapper is byte-orthogonal but the SCORE-DOMAIN interaction term is unmeasured. **Verdict**: PROCEED PATH 1 paired-axis ONLY (closed-form rate-axis prediction); REFUSE L2 INTEGRATION until probe-disambiguator + Dykstra-feasibility helper land per Catalog #296.
- **Yousfi** — operating-within: contest scorer's pre-trained driving knowledge implicitly contains the dashcam prior; DP1's incremental signal is bounded. **Verdict**: PROCEED but classify as `frontier_protecting` not `frontier_breaking`. Prioritize ASYMPTOTIC pursuit substrates (Z6/Z7/Z8/C6/ATW v2) for plateau exit. DP1 stays in queue as plateau-lock-in tool.
- **Fridrich** — operating-within: DP1 IS the steganographic cover (OOD frozen prior); per-pair residual IS the embed. The Atick-Redlich cooperative-receiver pattern is structurally clean. **Verdict**: PROCEED; reiterate Phase 2 hardening recommendation for differential-privacy noise parameter reservation (Phase 3 scope).
- **Contrarian** — operating-within: large reuse-graph claims with zero empirical anchor are the false-authority risk class. **Verdict**: PROCEED_WITH_REVISIONS — the cross-substrate composability audit (op-routable #1) is non-negotiable BEFORE any new dispatch fires. PATH 1 paired-axis ($1.90) lands FIRST as the canonical first anchor; everything else is gated on the audit + anchor.
- **Assumption-Adversary** — see frontmatter verbatim. Cross-substrate composability is CARGO-CULTED-EMPIRICALLY-UNTESTED. INSISTS on the cross-substrate audit (op-routable #1) before any reuse-graph dispatch.
- **Hinton** (grand council; knowledge-distillation grandfather) — operating-within: DP1 is canonical distillation pattern (teacher = Comma2k19 distribution; student = renderer; prior loss = soft-target KL analog via PCA projection L2). **Verdict**: PROCEED with note: the current `DashcamPriorLoss` uses L2 reconstruction loss, not KL on logits. For the LOSS function specifically, Hinton 2014 temperature-scaled KL would be more principled IF the codebook were probabilistic. Current PCA basis is point-projection; the L2 framing is correct for the linear-subspace coding choice. No revision required.
- **Schmidhuber** (compression-as-intelligence) — operating-within: a frozen prior IS a compression primitive in the MDL sense; the codebook IS the part of `H(contest_video)` that's explained by OOD dashcam structure. **Verdict**: PROCEED; MDL bound on codebook size [5_000, 10_000] is empirically defensible per Round 2 MacKay. No additional revision required.
- **Hassabis** (strategic research perspective) — operating-within: transfer-learning across substrates IS the canonical research move when the prior is correct. **Verdict**: PROCEED but support Yousfi's frontier_protecting classification — DP1's strategic value is locking in incremental frontier improvements across MULTIPLE substrates, not driving any single substrate to the asymptotic floor. Frame the dispatch sequence accordingly: rate-axis baseline (PATH 1) → cross-substrate audit (op-routable #1) → ASYMPTOTIC pursuit dispatches in parallel.
- **Karpathy** (engineering practitioner) — operating-within: arch-search rigor demands a baseline anchor before parametric sweeps. **Verdict**: PROCEED PATH 1 paired-axis FIRST. No L2 strength sweep until rate-axis baseline lands. No PATH 2 trainer Phase 2 until cross-substrate audit clarifies which dependent substrate inherits the prior most cleanly.
- **Selfcomp** (sister substrate consumer) — operating-within: as a PR #56 / block-FP / SegMap author who would inherit DP1 prior, I need to know which composition cell is structurally optimal for my substrate's grammar. **Verdict**: PROCEED but op-routable #1 cross-substrate audit MUST classify each base by integration grade (architectural-fit / partial-fit / structural-mismatch) so dependent substrates make informed adoption decisions.
- **Quantizr** (adversarial PR101 perspective) — operating-within: 5-stage pipeline (anchor→finetune→joint→QAT→final) carries the eval roundtrip; integrating DP1 as a PATH 2 prior changes the loss surface during finetune. **Verdict**: PROCEED PATH 2 only AFTER council Phase 2 explicitly designs the loss-hook injection point per Catalog #229 premise verification.

**Round 1 tally: 12 PROCEED_WITH_REVISIONS / 0 PROCEED-unconditional / 0 DEFER / 0 REFUSE.** Unanimous PROCEED_WITH_REVISIONS.

The revisions are the 6 op-routables in the frontmatter `council_decisions_recorded`. The verdict is **PROCEED_WITH_REVISIONS** — DP1 is not blocked from dispatch BUT op-routable #1 (cross-substrate composability audit) MUST precede any new DP1 composition dispatch beyond the already-built PATH 1 packet.

## 5. Per-substrate reactivation criteria

Per CLAUDE.md "Forbidden premature KILL". DP1 is NOT killed and NOT deferred at the family level. The 6 op-routables (frontmatter) ARE the reactivation paths. Priority ordering:

1. **PRIORITY 1**: PATH 1 paired-axis dispatch ($1.90; T4 + CPU paired). First empirical anchor across the entire DP1 reuse surface. Confirms or disconfirms rate-axis closed-form prediction. **Cost: $1.90; predicted band [+0.0000160, +0.0000180] [contest-CPU] AND [+0.0000160, +0.0000180] [contest-CUDA] = baseline + Shannon rate term**. Probe-outcome verdict: PROCEED (canonical first-anchor) OR PARTIAL (if cross-axis variance > expected). Recipe `dp1_plus_fec6_composition_modal_paired_dispatch.yaml` already exists + Catalog #324 `predicted_band_validation_status: pending_post_training` declared.

2. **PRIORITY 2**: CROSS-SUBSTRATE COMPOSABILITY AUDIT ($0 read-only subagent). Builds the 6-substrate × composition-path × predicted-ΔS-basis matrix. Spot-checks Catalog #210 license-tag propagation on A1 / PR101 / sane_hnerv. Classifies each base by integration grade. Required BEFORE any new DP1 composition dispatch beyond PATH 1.

3. **PRIORITY 3**: PATH 2 council Phase 2 deliberation ($0 council). Decides variant-A (codebook in archive) vs variant-B (baked into inflate.py); λ_DP1 sweep design; loss-hook integration surface per Catalog #229 premise verification. Reactivation gate: PROCEED verdict + probe-disambiguator path landed.

4. **PRIORITY 4** (deferred until PRIORITIES 1-3 land): L2 INTEGRATION dispatch ($5-15 for PATH 1 L2 strength sweep; $10 for PATH 2 first paired-axis empirical anchor). Each dispatch requires its own predicted-band Dykstra-feasibility check per Catalog #296.

## 6. Catalog #324 post-training Tier-C validation discipline

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #324.

- **PATH 1 paired-axis (PRIORITY 1)**: `predicted_band_validation_status: pending_post_training`. The rate-axis prediction (+0.0000172) is closed-form Shannon arithmetic and does NOT require Tier-C post-training validation — it's a contest-rate-term computation per the canonical formula `25 * archive_bytes / 37_545_489`. NO Catalog #324 violation for PATH 1 because there's no learned model to evaluate Tier-C density on.
- **PATH 2 (PRIORITY 3-4)**: `predicted_band_validation_status: pending_post_training`. The frame-axis prediction [-0.005, -0.020] band IS heuristic and requires post-training Tier-C density measurement on the actual archive emitted by `experiments/train_substrate_pr101_with_dp1_prior_regularizer.py` after a converged run. Recipe `substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch.yaml` declares `research_only: true` until council Phase 2 PROCEEDs.
- **L2 INTEGRATION (PRIORITY 4 deferred)**: `predicted_band_validation_status: pending_post_training`. The L2 frame-axis ΔS band requires probe-disambiguator at `tools/probe_dp1_lambda_disambiguator.py` (planned) + post-training Tier-C re-measurement on the L2-strength-sweep archives.

## 7. Catalog #310 / #311 / #312 cross-checks

- **Catalog #310 (F-asymptote substrate-class)**: **N/A**. DP1 is a PRIOR / regularizer, not an F-asymptote substrate-class.
- **Catalog #311 (Atick-Redlich cooperative-receiver framing)**: **PARTIAL**. The forensic audit explicitly cites the Atick-Redlich pattern (DP1 is the side-information channel). The codebook IS the OOD shared-prior between encoder + decoder. However, this is not an ego-motion-conditioned next-frame predictor; it's a SPATIAL-COOPERATIVE-RECEIVER (retinal redundancy reduction analog). Per the Catalog #311 amendment for spatial-not-temporal Atick-Redlich: this is a legitimate spatial application, not requiring ego-motion + predictive tokens. NO violation; classify as `spatial-cooperative-receiver-substrate` consistent with sister substrates (atw_codec_v2 / tishby_ib_pure).
- **Catalog #312 (hierarchical predictive coding)**: **N/A**. DP1 is single-level prior (PCA basis is one-shot linear-subspace coding), not hierarchical predictive coding.

## 8. Predicted ΔS band

Per CLAUDE.md Catalog #296.

### PATH 1 PRIORITY 1 dispatch ($1.90 paired-axis)

**Band**: `[+0.0000160, +0.0000180] [contest-CPU-and-contest-CUDA-paired]` — closed-form Shannon arithmetic from fec6 baseline 0.19205 [contest-CPU] + 0.20533 [contest-CUDA].

* Rate-axis cost ONLY: `25 * 25_814 / 37_545_489 = +0.0000172`. NOT Dykstra-required (Shannon rate term arithmetic, not a polytope intersection).
* Frame-axis effect: 0 by construction (PACT_DP1_PRIOR_STRENGTH=0.0 default).
* Cross-axis variance: predicted within ±0.000002 of rate term per Shannon arithmetic; numerical floor.

### PATH 1 L2 INTEGRATION (PRIORITY 4 deferred)

**Band**: `[-0.005, -0.012] [time-traveler-prediction]` — heuristic extrapolation. **Dykstra-feasibility check REQUIRED** per Catalog #296. Probe-disambiguator at `tools/probe_dp1_lambda_disambiguator.py` (Phase 2 deferral).

### PATH 2 (PRIORITY 3-4 deferred)

**Band**: `[-0.005, -0.020] [time-traveler-prediction]`. Council Phase 2 must land Dykstra-feasibility helper for the DP1-PR101 rate-distortion intersection BEFORE any paid dispatch fires.

## 9. Canonical-vs-unique decision per layer

Already covered exhaustively in `dp1_dual_stacking_design_20260517.md` for PATH 1 + PATH 2. No additional canonical-vs-unique fork required at the symposium surface. The substrate's META layer contract per Catalog #241/#242 is the canonical interface.

## 10. Cross-substrate impact verdict (this symposium's distinctive contribution)

This is the highest-leverage Wave 3 symposium because:

1. **DP1 is reused by 6 substrates** registered in `_KNOWN_BASE_TAGS`: a1, pr101, hdm8, yucr, time_traveler_l5, sane_hnerv. Any cargo-cult in DP1's composability assumption propagates into 6 dependent substrates.
2. **ZERO empirical anchors exist** for DP1 across the entire reuse graph after 4 calendar days post-Phase-2-landing. This is the LARGEST false-authority surface in the contest. Per CLAUDE.md "Apples-to-apples evidence discipline" — citing DP1 as a reuse harness without per-substrate empirical receipt is the canonical cargo-cult pattern Catalog #287 + #321 + #323 protect against.
3. **The 6 dependent substrates have RADICALLY DIFFERENT internal grammars** (HNeRV-family vs UNIWARD-sidecar vs Time-Traveler L5 vs SANE HNeRV). Per Item #8 #864 META: composition methodology does NOT compose monotonically across architectural changes. DP1 reuse-graph composability inherits the SAME risk class as v6->v7->v8 monotonicity assumption.

**Net cross-substrate verdict**: the CROSS-SUBSTRATE COMPOSABILITY AUDIT (op-routable #1) is the SINGLE highest-leverage META action of the entire Wave 3. It's $0 read-only work. It MUST land BEFORE any new DP1 composition dispatch beyond PATH 1's already-built rate-axis baseline measurement. The 2 hours of audit work saves $5-30 of dispatch spend that would otherwise burn on assumption-bound composition cells.

## 11. Continual learning wire-in (6-hook declaration per Catalog #125)

1. **Sensitivity-map contribution**: ACTIVE — DP1 codebook PCA components carry per-section sensitivity (road-plane / sky-horizon / vehicle-appearance) usable by `tac.sensitivity_map.*` per-axis weight contributions.
2. **Pareto constraint**: ACTIVE — DP1 composition adds rate-axis cost +0.0000172; the autopilot ranker per Catalog #319 v2 cascade can consume this as a Pareto-feasibility boundary signal.
3. **Bit-allocator hook**: PLANNED — DP1's codebook bit-budget (5-10 KB post-brotli) registers per-section bit-allocator priorities once empirical anchor lands.
4. **Cathedral autopilot dispatch hook**: ACTIVE — per op-routable #6, autopilot ranker should treat DP1 composition candidates as plateau-adjacent (1.0× passthrough OR -0.005 floor reward per Yousfi verdict) UNLESS cross-substrate audit surfaces a sub-0.188 candidate.
5. **Continual-learning posterior update**: ACTIVE — this symposium anchor lands the FIRST DP1 deliberation in `.omx/state/council_deliberation_posterior.jsonl` (pre-this-memo the canonical posterior had ZERO DP1 entries). PATH 1 paired-axis empirical anchor will register via `tac.probe_outcomes_ledger.register_probe_outcome`.
6. **Probe-disambiguator**: PLANNED — `tools/probe_dp1_lambda_disambiguator.py` is required for L2 INTEGRATION (PRIORITY 4); also PATH 2 council Phase 2 needs a probe-disambiguator for variant-A vs variant-B (codebook-in-archive vs baked-into-inflate.py).

## 12. Operator-visible op-routables (summary)

| # | Action | Cost | Owner | Priority |
|---|---|---|---|---|
| 1 | CROSS-SUBSTRATE COMPOSABILITY AUDIT (READ-ONLY subagent; builds (substrate × composition_path × predicted_ΔS_basis) matrix; spot-checks Catalog #210 license-tag propagation on 3 substrates) | $0 | Operator dispatches read-only subagent | **HIGHEST** — blocks all other DP1 work |
| 2 | PATH 1 paired-axis dispatch ($1.90 envelope; T4 + CPU on the SAME composed archive bytes) | $1.90 | Operator-authorize `dp1_plus_fec6_composition_modal_paired_dispatch.yaml` after #1 lands | HIGH |
| 3 | PATH 2 council Phase 2 deliberation ($0 council; decides variant-A vs variant-B + loss-hook injection surface) | $0 | Operator dispatches Phase 2 council subagent | MEDIUM |
| 4 | Register DP1 first empirical anchor via `tac.probe_outcomes_ledger.register_probe_outcome` (post PATH 1 dispatch) | $0 | Automatic via canonical helper | LOW (gated by #2) |
| 5 | DEFER L2 INTEGRATION ($5-15 strength sweep) until probe-disambiguator + Dykstra-feasibility helper land | — | Operator decides post-#2 anchor | DEFERRED |
| 6 | Adjust autopilot ranker: treat DP1 composition candidates as plateau-adjacent UNLESS audit #1 surfaces sub-0.188 candidate | $0 | Cathedral autopilot patch | LOW (informational) |

## 13. Cross-references

- `.omx/research/dp1_forensic_audit_and_roadmap_20260515_codex.md` — exhaustive engineering forensic
- `.omx/research/dp1_phase_2_hardening_v2_council_20260514.md` — 5-round Phase 2 council (was never written to canonical posterior)
- `.omx/research/dp1_dual_stacking_design_20260517.md` — PATH 1 + PATH 2 design memo
- `.omx/research/dp1_engineering_status_refresh_20260516_codex.md` — current engineering status (untrained_unpromoted_promising_substrate)
- `src/tac/substrates/pretrained_driving_prior/composition.py` — canonical compose_with API (396 LOC)
- `src/tac/substrates/pretrained_driving_prior/codebook.py` — frozen prior codebook
- `src/tac/substrates/pretrained_driving_prior/prior_application.py` — DashcamPriorLoss
- `.omx/operator_authorize_recipes/dp1_plus_fec6_composition_modal_paired_dispatch.yaml` — PATH 1 paired-axis dispatch (ready)
- `.omx/operator_authorize_recipes/substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch.yaml` — PATH 2 scaffold (research_only)
- `feedback_wave_1_per_substrate_symposium_dispatch_landed_20260517.md` — Wave 1 anchor (Item #8 hypothesis HARD-EARNED 4/4)
- `.omx/research/council_per_substrate_symposium_nscs06_v8_path_b_20260517.md` — #864 #8 META anchor (cargo-cult-unwind does NOT compose monotonically)
- `.omx/research/council_per_substrate_symposium_lane_17_imp_20260517.md` — PROCEED canonical structure
- `.omx/research/council_per_substrate_symposium_pr106_05_06_reformulated_20260517.md` — paradigm-INTACT/design-CARGO-CULTED canonical Item #8 pattern
- `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md` — sequential cross-pollination discipline
- `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` — Item #8 originator (bidirectional cross-pollination)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (DP1 is NOT killed; reactivation paths preserved)
- Catalog #209 / #210 / #211 / #213 / #220 / #233 / #270 / #287 / #292 / #294 / #295 / #296 / #303 / #305 / #313 / #315 / #319 / #321 / #323 / #324 / #325
