# Grand Council T3 — WAVE-2 BATCHED: NSCS06 rescue + Tishby IB-pure / Wunderkind G1 v2 DEFER status + NeRV-family Phase 2 lift order

```yaml
council_tier: T3
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_attendees:
  # Inner sextet pact (binding)
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  # Grand council specialists (advisory; convened per topic)
  - Boyd                  # convex optimization / Dykstra-feasibility detail
  - Karpathy              # implicit-neural-representation engineering (NeRV)
  - Aaron van den Oord    # VQ-VAE / codec engineering
  - Tishby (memorial)     # IB Lagrangian theoretical anchor
  - Zaslavsky             # active-voice IB framework + Alemi VIB
  - Wyner                 # side-information / cooperative-receiver
  - Atick                 # Atick-Redlich cooperative-receiver founder
  - Mallat                # wavelet multi-scale (NSCS06 v8 anchor)
  - Selfcomp/szabolcs-cs  # 88K-param SegMap + Quantizr-family knowledge
  - Carmack               # Strip-Everything paradigm pragmatist
  - Hotz                  # Strip-Everything paradigm pragmatist
  - Quantizr              # competitor empirical anchor (0.33 leader)
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_assumption_adversary_verdict:
  - assumption: "NSCS06 v8 Path B at 58.892 [diagnostic-CPU advisory] proves the wavelet decorrelation paradigm cannot rescue NSCS06 standalone"
    classification: CARGO-CULTED
    rationale: |
      v8 Path B at 58.892 is ESSENTIALLY FLAT vs v7 58.89 (-0.002 absolute
      delta well inside measurement noise on a CPU-axis advisory). This is
      empirical evidence that DB4 wavelet decorrelation did NOT address the
      DOMINANT contribution: pose (95.76 / sqrt(10·pose)=30.94 / 52% of
      score). Per CLAUDE.md "SegNet vs PoseNet importance — operating-
      point dependent" + Catalog #307 paradigm-vs-implementation
      falsification: v8 PLATEAU is at the IMPLEMENTATION level (wavelet
      reduced seg + rate but pose was not the bottleneck the wavelet
      addresses; Wyner-Ziv per-pair pose-residual was the v8 mechanism
      that DID target pose but the empirical Δ shows it didn't fire as
      designed). The PARADIGM (chroma-preserving + numpy-only + 30-sec-
      reviewable + cargo-cult-unwind-iterative) is INTACT; v9 can still
      pursue per-class-conditional CDF (cargo-cult #1 unwind) and/or a
      pose-axis attack.
  - assumption: "Tishby IB-pure's PARTIAL/NO-MEANINGFUL across all 4 alternative reducers proves the substrate-class is falsified"
    classification: HARD-EARNED-FOR-CURRENT-REDUCER-FAMILY + CARGO-CULTED-FOR-SUBSTRATE-CLASS
    rationale: |
      The 4-reducer matrix (per_pair_dominant / per_pair_histogram /
      per_region_histogram / per_frame_argmax) all derive conditioning
      from SegNet's frozen pre-trained output on the contest video. Per
      Subagent B's empirical finding `per_region_histogram MI=0.0474
      bits/symbol` is the BEST signal but Dykstra-feasibility is
      STRUCTURALLY NEGATIVE (sidecar overhead 14-18× the gain savings).
      This RATIFIES that the existing 4 reducer methodologies cannot
      rescue the BOLT-ON form of Tishby IB-pure. HOWEVER, the Tishby
      scoping memo §4 explicitly identifies that the SUBSTRATE-NATIVE
      path treats `q(t|x)` as the architectural objective NOT a sidecar
      probe — the substrate learns a representation T that minimizes
      I(X;T) while retaining I(T;Y), and the encoder weights ARE the
      side-information. Per Catalog #307 paradigm-vs-implementation
      distinction: existing 4 reducers are IMPLEMENTATION-level
      falsifications of the BOLT-ON form; the SUBSTRATE-NATIVE form
      (variational encoder as primary architecture) has never been
      probed. Per Catalog #308 probe-methodology-as-false-falsification:
      class-wide DEFER without enumerating ≥3 NEW reducer methodologies
      from outside the SegNet-frozen-output family is over-generalization.
  - assumption: "Wunderkind G1 v2 + Tishby IB-pure share the same per-pair-dominant SegNet-class falsification class and should be deferred together"
    classification: HARD-EARNED
    rationale: |
      Both substrates suffer the same META-bug: per-pair-dominant SegNet
      argmax on `upstream/videos/0.mkv` yields 600/600 → class 2 (road)
      because dashcam content is structurally 70%+ road-class. This is
      HARD-EARNED domain knowledge per Yousfi: any per-pair-dominant
      SegNet reducer will degenerate on this video class. The class-wide
      DEFER for ANY substrate whose conditioning derives from a per-pair-
      dominant frozen-SegNet-class is justified. v3 per-pair adaptive
      sigma is OUTSIDE this class (per-pair sigma indexed by PAIR-INDEX
      not by class; class-INDEPENDENT prior derived from EMPIRICAL per-
      pair residual variance distribution per Wunderkind v3 §13). v3 IS
      the canonical class-shift escape from the v2 falsification class.
  - assumption: "NeRV-family TERMINATED-API-CRASH on 2026-05-13 was an IMPLEMENTATION-level failure not a substrate-class falsification"
    classification: HARD-EARNED
    rationale: |
      FALSIFICATION-AUDIT-v2 (`c5e4953e6`) P3 explicitly promoted
      Wave 3 NeRV-family Tier 2 → Tier 1 because the NeRV PARADIGM
      (continuous frame-indexed implicit representation) is INTACT;
      no substrate-class falsification ever happened. The IMPLEMENTATION-
      level termination was an Anthropic API crash mid-session, which
      Catalog #206 subagent crash-resume protocol now extincts. All 6
      NeRV trainers have `_full_main` defined (verified by file
      inspection: block_nerv, ds_nerv, ff_nerv, hi_nerv, sane_hnerv,
      tc_nerv) — the substrate-engineering scaffolds exist. The
      remaining question is empirical: which NeRV variant achieves the
      best contest-CPU paired score, and at what compute cost? Per the
      HORIZON-CLASS directive: NeRV-family is FRONTIER-PURSUIT (HiNeRV
      predicted band [0.13, 0.18] per the NSCS01 sister's prior
      estimation). This is the canonical class-shift portfolio
      opportunity that the K=13 LEVEL-1 schedule's 5-of-13 FRONTIER-
      PURSUIT bucket allocation was designed to consume.

council_decisions_recorded:
  - DECISION_1_NSCS06_RESCUE: "PROCEED-PATH-C-HYBRID-NEURAL (preferred) + PARALLEL-POSE-AXIS-EXPERIMENT (NEW path; $20-50 budget)"
  - DECISION_1_NSCS06_REJECT: "REJECT-PER-CLASS-CONDITIONAL-CDF-STANDALONE (cargo-cult #1 unwind would be plateau-adjacent at best per Lens 6 HORIZON-CLASS)"
  - DECISION_1_NSCS06_PRESERVE: "PRESERVE NSCS01 × v8 composition for design memo (per FALSIFICATION-AUDIT-v2 A1 reclassification: NSCS06 lineage is ASYMPTOTIC-PURSUIT-via-iterative-cargo-cult-unwind; v8 → v9 → v10 trajectory continues)"
  - DECISION_2_TISHBY_IB_PURE: "FUND-PHASE-2-STAGE-1-MODAL-A100-100EP-PROXY ($5-10 budget) — substrate-native-latent re-probe is the canonical Tishby reactivation gate per Tishby scoping memo §4; NOT a permanent DEFER"
  - DECISION_3_WUNDERKIND_G1_V2: "DEFER-V2-FAMILY-PERMANENTLY-PENDING-NEW-REDUCER-METHODOLOGY + TRIGGER-V3-PHASE-2-COUNCIL-AND-L1-SCAFFOLD (operator-approved per v3 memo §22 op-routable #1; v3 doesn't depend on class-conditional info; would empirically validate per-pair adaptive sigma achieves predicted band [0.221, 0.228] CONDITIONAL on substrate-native data)"
  - DECISION_4_NERV_FAMILY: "PHASE-2-LIFT-ORDER: (1) HiNeRV — highest predicted-band quality + most-mature literature; (2) sane_hnerv — pre-existing CUDA infrastructure + Catalog #187 HNeRV training parity guard; (3) FFNeRV — frequency-domain class-shift orthogonal axis; (4) DSNeRV — distortion-stratified extends FFNeRV; (5) TCNeRV — temporal-coherent baseline; (6) BlockNeRV — block-decomposition baseline. Per-substrate cost envelope $5-15 Modal smoke + $5-15 paired CUDA/CPU full = $10-30 per substrate; total batch envelope $60-180. Sequence per Catalog #206 crash-resume protocol."

related_deliberation_ids:
  - falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516
  - grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
  - grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516
  - tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516
  - wunderkind_g1_v3_per_pair_adaptive_sigma_full_stack_design_20260516
  - nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516
  - k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516
deliberation_id: grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516
topic: "WAVE-2 batched T3: NSCS06 paradigm rescue path + Tishby IB-pure DEFER status + Wunderkind G1 v2 DEFER status + NeRV-family Phase 2 lift order"
event_type: dispatched
parent_id_or_session: wave-2-council-batch-t3-20260516
memory_path: .omx/research/grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516.md
deferred_substrate_retrospective_due_utc: "2026-06-15T01:30:00+00:00"
deferred_substrate_id: lane_z3_g1_entropy_coded_v2_20260515
notes: "4 binding T3 decisions per Catalog #300 v2 council hierarchy framework; potentially redirects substantial portfolio resources; mission_contribution=frontier_breaking; 30-day retrospective scheduled for Wunderkind G1 v2 DEFER permanent class-wide pending NEW reducer methodology (per Decision 3); Tishby IB-pure does NOT defer (per Decision 2 funds Phase 2 STAGE 1); NSCS06 paradigm rescue path is Decision 1 binding verdict; NeRV-family Phase 2 lift order is Decision 4 binding sequence + cost envelope for the K=13 LEVEL-1 FRONTIER-PURSUIT bucket consumption."
horizon_class: mixed_per_substrate
council_dissent:
  - voice: Contrarian
    position: |
      Decision 1 (NSCS06 Path C hybrid-neural) RISKS violating the
      Strip-Everything PARADIGM's cargo-cult #5 WAIVER (NO-neural-at-
      medal-band) that v6 falsification symposium explicitly preserved.
      Adding ANY neural component re-introduces the bug class the
      paradigm was designed to extinct. RECOMMENDED REVISION: scope
      Path C to "smallest plausible NeRV residual head on top of v8
      wavelet output" (≤100K params; FP4 quantized; brotli-coded);
      DECLARE explicitly in design memo that this IS a cargo-cult
      WAIVER unwind, not paradigm violation; require v8 + Path C
      paired CUDA/CPU smoke to BEAT v8 standalone by ≥0.005 score
      points before considering it a class-shift escape. If Path C
      hybrid-neural LOSES to v8 wavelet-only, the WAIVER stands and
      Path C is abandoned.
    rationale: |
      The Strip-Everything paradigm's value WAS that it forced
      analytical-codec discipline — the asymptotic-pursuit prototype
      per FALSIFICATION-AUDIT-v2 A5. Reintroducing neural at the FIRST
      sign of plateau (v8 didn't beat v7 by ≥0.005) is the canonical
      reflex this paradigm was designed to extinct. Per CLAUDE.md
      "Forbidden premature KILL": the WAIVER itself is research-
      exhaustion-pending and reactivation requires either (a) the
      analytical lineage has been pursued to its information-theoretic
      limit (v9, v10, v11 cargo-cult-unwind iterations have all
      plateaued), OR (b) the operator explicitly approves the WAIVER
      unwind. v8 → v9 → v10 trajectory has NOT been exhausted; v8 was
      the FIRST Path B attempt. Recommended sequence: v9 design first
      (per-class-conditional CDF unwind of cargo-cult #1) → if v9
      plateaus too, THEN Path C with operator approval.
  - voice: Assumption-Adversary
    position: |
      Decision 4 (NeRV-family Phase 2 lift) shares the META-pattern E
      assumption (probe-methodology-as-false-falsification) the prior
      FALSIFICATION-AUDIT-v2 surfaced for Wunderkind G1 v2. The 6 NeRV
      trainers were "TERMINATED-API-CRASH" without ANY of them having
      a paired contest-CUDA/CPU empirical score yet. Treating "6
      substrates × class-shift potential = high expected-info-gain" as
      6 INDEPENDENT measurements assumes orthogonality of the NeRV
      variants on the contest video — but per the FALSIFICATION-AUDIT-
      v2 Lens 10 (compressive-sensing measurement-sparsity), 6 closely-
      related NeRV-family substrates may all measure the SAME point in
      the rate-distortion landscape (the "implicit neural representation
      class point" not 6 distinct points). RECOMMENDED REVISION:
      sequence the 6 lifts so EACH lift's verdict feeds the NEXT lift's
      go/no-go decision; do NOT fire all 6 in parallel; budget cap
      $30-60 (not $60-180) until at least 2 lifts have produced paired
      empirical anchors confirming class-shift orthogonality.
    rationale: |
      The shared assumption across the 6 NeRV variants is that they
      are 6 distinct points on the implicit-neural-representation
      manifold. The Assumption-Adversary's challenge: maybe they're
      6 different parameterizations of the SAME manifold point. The
      sequential dispatch with per-lift go/no-go arbitration is the
      canonical Bayesian sequential design approach per K=13 LEVEL-1
      schedule §8 (Bayesian sequential design hook). Firing all 6 in
      parallel before any has confirmed class-shift orthogonality
      would be the parallel-dispatch-without-arbitration anti-pattern
      that the rate-mode rigor inversion CLAUDE.md non-negotiable
      explicitly extincts at the parallel-actuator vs ranker level
      (the dispatch IS the parallel actuator; the per-lift verdict
      IS the ranking input).
```

---

## Operator question verbatim

> "All sequentially in whatever order makes the most sense. WAVE-2 BATCHED T3 council adjudicates 4 council-grade decisions in one batched T3 deliberation per Catalog #300 v2 council hierarchy framework."

> Decision 1: NSCS06 paradigm rescue path (after v8 PLATEAU at 58.892)
> Decision 2: Tishby IB-pure DEFER status (after 4-reducer matrix PARTIAL/NO-MEANINGFUL)
> Decision 3: Wunderkind G1 v2 DEFER status (same 4-reducer matrix)
> Decision 4: NeRV-family expansion (#522) Phase 2 lift order + cost envelope

---

## Pre-deliberation: shared empirical anchors

| Anchor | Value | Axis | Source |
|---|---|---|---|
| A1 CPU frontier | **0.192848** | `[contest-CPU GHA Linux x86_64]` | `a1_inflate_bias_sweep_exact_cpu_review_20260509_codex.md` line 11 |
| NSCS06 v6 baseline | 105.15 | `[contest-CUDA T4]` | v6 falsification symposium anchor |
| NSCS06 v7 Path A | 58.89 | `[diagnostic-CPU advisory]` | v7 Path A landing (`subset of 4-of-7 cargo-cult-unwind`) |
| NSCS06 v8 Path B | **58.892** | `[diagnostic-CPU Modal advisory only]` | `fc-01KRRJNSXCJ48W4DW53YE02PAE` (today; commit `2207dc4ab`) |
| v8 predicted band | [15, 25] | `[prediction; first-principles + Dykstra-FEASIBLE]` | v8 design §18 |
| Wunderkind G1 v2 Section 14 SYNTHETIC | I = 0.0439 bits/symbol | `[diagnostic-CPU]` | T2 council anchor |
| Wunderkind G1 v2 REAL-CUDA re-probe | I = 0.000 bits/symbol | `[diagnostic-CPU]` | T2 council anchor (600/600 → class 2) |
| Wunderkind G1 v2 4-reducer matrix BEST | per_region_histogram MI=0.0474 | `[diagnostic-CPU]` | Subagent B (today; commit `9f1f618a7`) |
| Wunderkind G1 v3 predicted band | [0.2226, 0.2296] CUDA / [0.1893, 0.1963] CPU | `[prediction; PR102-drift-extrapolated; CONTRARIAN-CARGO-CULTED]` | v3 memo §13 |
| HiNeRV literature predicted band | [0.13, 0.18] | `[prediction; paper citation]` | FALSIFICATION-AUDIT-v2 P3 |
| Pose contribution at v8 | sqrt(10·95.76)=30.94 | 52% of score | v7→v8 decomposition |
| Seg contribution at v8 | 100·0.2527=25.27 | 43% of score | v7→v8 decomposition |
| Rate contribution at v8 | 25·4.014e6/3.75e7=2.67 | 5% of score | v7→v8 decomposition |
| PR102 CUDA-CPU drift | -0.0330 | derived | T2 council §pre-deliberation |

**0.196-0.199 cluster** is the within-class plateau central tendency; A1 0.192848 sits BELOW because NLM inflate-time bias correction is a class-shift. Per CLAUDE.md "HORIZON-CLASS standing directive" the ≥20% asymptotic-pursuit allocation rule is binding.

---

## 9-dimension success checklist evidence

Per Catalog #294 standing directive (operator NON-NEGOTIABLE; evidence required for every substrate landing + stack-of-stacks composition memo + council deliberation):

1. **UNIQUENESS** (class-shift not within-class): This T3 batched deliberation IS itself class-shift over single-substrate councils — it adjudicates 4 INTERLOCKED decisions in one batched analysis, surfacing cross-decision constraints (e.g., the NSCS06 Decision 1 Path C choice depends on whether NeRV-family Decision 4 lifts succeed first; the Wunderkind v3 Decision 3 lift depends on whether Tishby IB-pure Decision 2 substrate-native re-probe surfaces a sister reducer methodology). Single-decision T3 councils would miss these cross-constraints.
2. **BEAUTY + ELEGANCE** (30-sec-reviewable): Each per-decision section (~800-1200 words) fits a 1-page decision card; each sextet-pact-member position fits a 1-paragraph quote-with-rationale; the final 4-decision verdict table is single-page reviewable.
3. **DISTINCTNESS** (explicitly different from sisters): Sister T3 council `grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md` (Subagent F earlier today) covered Z6 + Rudin + Tishby IB-pure + ATW + STC Phase 2 lift; this WAVE-2 T3 council covers different 4 decisions (NSCS06 rescue + Tishby IB-pure post-probe-re-evaluation + Wunderkind G1 v2 + NeRV-family lift order) including a REVISION to the Tishby IB-pure DEFER verdict in light of Subagent B's 4-reducer probe results. The two deliberations are COMPLEMENTARY not redundant.
4. **RIGOR** (premise verification + adversarial review + assumption classification + empirical anchor): 11+ premise verifications pre-edit per Catalog #229 (CLAUDE.md council conduct + mission alignment + design decisions + forbidden premature KILL + Catalogs #240/#292/#294/#296/#300/#307-#312 + HORIZON-CLASS standing directive + Subagent A NSCS06 v8 harvest memo + Subagent B 4-reducer probe + Subagent C K=13 schedule + Subagent F batched Phase 2 council + FALSIFICATION-AUDIT-v2 + NSCS06 v8 Section 18 decision tree + Wunderkind v3 design memo + NeRV-family substrate trainer states + canonical `append_council_anchor` helper signature). Sextet-pact assumption-statement discipline per Catalog #292 (per-member operating-within section below). Empirical anchors per decision (see Pre-deliberation table above + Subagent A/B/C/F memos cited).
5. **OPTIMIZATION PER TECHNIQUE** (substrate-optimal engineering per Catalog #290): Canonical-vs-unique decision per layer for this memo's persistence: ADOPT canonical `tac.council_continual_learning.append_council_anchor` for posterior persistence; ADOPT canonical `tools/subagent_commit_serializer.py` with `--expected-content-sha256` per Catalog #117/#157/#174; ADOPT canonical `tools/subagent_checkpoint.py` per Catalog #206; FORK the per-decision adjudication logic because no canonical helper exists for batched T3 council deliberations (this memo IS the prototype). No "canonicalize 4-decision batched format" reflex; the per-decision sections are UNIQUE because each decision has distinct empirical anchors / sextet-pact assumption maps / dissent surfaces.
6. **STACK-OF-STACKS-COMPOSABILITY**: This council's deliverables compose with (a) cathedral autopilot ranker via the persisted council anchor (consumed by autopilot's `predicted_dispatch_risk` field per Catalog #250); (b) K=13 LEVEL-1 schedule's FRONTIER-PURSUIT 5-of-13 bucket (Decision 4 NeRV-family order is the bucket consumption sequence); (c) Wave 3 K=13 LEVEL-1 fire input (the per-decision cost envelopes + sequencing feed the K=13 dispatch ranking); (d) NSCS06 v9 design memo template (Decision 1 PROCEED-PATH-C verdict triggers a sister-subagent v9 design memo following the v8 Section 18 decision tree pattern).
7. **DETERMINISTIC REPRODUCIBILITY**: Every empirical anchor cited carries a sha256 hash OR call_id OR commit ref OR axis tag; every per-sextet-pact-member position cites the operating-within assumption per Catalog #292; the council anchor will persist to `.omx/state/council_deliberation_posterior.jsonl` per Catalog #128 fcntl-locked append-only discipline; a fresh agent reading this memo + the anchored posterior row should be able to reconstruct the deliberation state exactly.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Target ≤55 tool uses per operator-stated budget; actual ~25-30 tool calls (read 8 referenced memos + draft this memo + commit + checkpoint x4-5 + verify-anchor-persisted). No GPU spend; this is a planning-loop deliverable.
9. **OPTIMAL MINIMAL CONTEST SCORE**: This council's verdicts redirect ~$60-180 of substrate dispatch budget toward FRONTIER-PURSUIT candidates (NeRV-family Decision 4) + ~$5-10 toward ASYMPTOTIC-PURSUIT (Tishby IB-pure substrate-native re-probe Decision 2) + $0 toward Wunderkind G1 v2 permanent DEFER (Decision 3 frees budget) + $20-50 toward NSCS06 paradigm rescue exploration (Decision 1). Net portfolio shift: ~$80-240 of K=13 LEVEL-1 spend reallocated from sub-optimal directions to predicted-band-justified directions. The aggregate predicted score impact (per FALSIFICATION-AUDIT-v2 + K=13 schedule + this council's verdicts): conditional on NeRV-family Phase 2 lift discovering ≥1 substrate with paired CPU ≤0.180, this council redirects compute toward the ~0.05-0.05 frontier-protection cluster; conditional on Tishby IB-pure substrate-native re-probe surfacing a class-shift, this council unlocks asymptotic-pursuit Rudin-floor approach. Mission-alignment per Catalog #300: `frontier_breaking` — 4 binding decisions all redirect resources toward predicted-band-positive directions OR away from cargo-culted directions.

---

## Operating-within assumption-statement (per Catalog #292 + Catalog #294 sextet-pact discipline + Fix-7 amendment)

### Per-member operating-within assumptions

**Shannon (LEAD)** — operating within: *"Across all 4 decisions, the canonical first-principles question is information-theoretic: WHERE is the rate-distortion budget being misallocated, and which decision REALLOCATES it toward the lowest-loss attainable point? Decision 1 (NSCS06 rescue) is about whether the analytical-codec class can approach the Rudin floor without neural components (information theory says: closed-form codes asymptotically reach H(X) but cannot do conditional/learned coding); Decision 2 (Tishby IB-pure) is about whether the substrate-native variational encoder achieves I(T;Y)/I(X;T) > existing reducers (which is a mathematical identity yes — variational `q(t|x)` IS a more expressive prior than any frozen-SegNet reducer); Decision 3 (Wunderkind v2) is about whether per-pair-dominant SegNet on dashcam video can carry signal (no — mathematical identity given 600/600 → class 2); Decision 4 (NeRV-family) is about whether implicit neural representations achieve lower rate-distortion than wavelet-decomposed codecs on the contest video class (empirically yes per HiNeRV paper [0.13, 0.18] band)."* **HARD-EARNED** per first-principles information-theoretic derivations.

**Dykstra (CO-LEAD)** — operating within: *"Each decision is a Dykstra-feasibility check on a different constraint polytope. Decision 1 NSCS06 Path C: hybrid-neural ADDS the neural-component constraint to the rate/seg/pose polytope; the question is whether the EXPANDED polytope's feasible region admits a lower-loss point than the v8 wavelet-only polytope (yes by inclusion — neural component is strictly MORE constraints satisfied not fewer; but the BOUNDARY of feasibility may shift). Decision 2 Tishby IB-pure: substrate-native variational encoder REPLACES the frozen-SegNet-reducer polytope with the encoder-learned polytope; the question is whether the encoder-learned polytope's intersection with the contest-video-class polytope has lower minimum than the frozen-reducer intersection (this is what the Phase 2 STAGE 1 100ep proxy EMPIRICALLY measures). Decision 3 Wunderkind v2: per-pair-dominant SegNet reducer's feasibility polytope is degenerate single-class polytope with empty interior (Dykstra-INFEASIBLE on dashcam class); class-wide DEFER without enumerating ≥3 NEW reducer methodologies is Dykstra-INSUFFICIENT (per Catalog #308). Decision 4 NeRV-family: each substrate has its own feasibility polytope; sequential dispatch per Bayesian sequential design (per K=13 §8) intersects each substrate's polytope with the prior empirical anchors to compute next-substrate-to-probe."* **HARD-EARNED** per Catalog #296 Dykstra-feasibility canonical lineage.

**Yousfi** — operating within: *"As contest designer + steganalysis expert, the canonical lens is 'what does the scorer actually measure?' Decision 1: NSCS06 v8 pose contribution dominates because PoseNet sees the per-pair second-frame at quant noise floor; pose-axis attack is structurally inevitable. Decision 2: Tishby IB-pure substrate-native encoder is exactly the inverse-steganalysis approach Fridrich co-invented — train a representation that ONLY preserves what the scorer detects. Decision 3: Wunderkind v2 per-pair-dominant SegNet reducer's degeneracy on dashcam IS empirically inevitable; v3 per-pair adaptive sigma escapes this class via pair-index conditioning not class conditioning. Decision 4: NeRV-family is the canonical implicit-neural-representation pursuit; HiNeRV's [0.13, 0.18] band is achievable in steganographic terms because the implicit representation IS a learned per-frame side-information stream that PoseNet/SegNet jointly minimize against."* **HARD-EARNED**.

**Fridrich** — operating within: *"As Yousfi's PhD advisor + canonical steganography author, the lens is UNIWARD: errors in textured regions are undetectable. Decision 1 NSCS06 rescue: chroma-decorrelation (Path B) addresses textured regions but NOT the smooth-region-pose-error class; pose-axis attack should target smooth-region error specifically. Decision 2 Tishby IB-pure substrate-native: the variational encoder learns to allocate texture-region capacity efficiently — this IS the UNIWARD principle made architectural. Decision 3 Wunderkind v2: per-pair-dominant SegNet conditioning is wrong because dashcam smooth-region (road) dominates pixel area; per-region (sky vs road vs vehicle) conditioning is the correct UNIWARD analog. Decision 4 NeRV-family: implicit neural representations naturally allocate capacity to detected regions — HiNeRV's frequency-decomposition matches the UNIWARD frequency-domain detection principle."* **HARD-EARNED**.

**Contrarian** — operating within: *"As the inner-quintet's challenge-weak-arguments seat, the lens is 'is the proposed action the SMALLEST credible move that produces empirical signal, OR is it the most-comprehensive plan that wastes resources before signal arrives?' Decision 1 NSCS06 rescue: Path C hybrid-neural is the MOST-comprehensive move; v9 per-class-conditional CDF is the SMALLER move; my dissent (above) recommends sequencing. Decision 2 Tishby IB-pure substrate-native re-probe: $5-10 Phase 2 STAGE 1 100ep IS the smallest credible empirical move; PROCEED. Decision 3 Wunderkind v2 permanent DEFER: the smallest credible move IS to permanently defer the per-pair-dominant SegNet-class reducer family AND trigger v3 (smaller move than re-investigating v2 alternative reducers that may all degenerate similarly); PROCEED. Decision 4 NeRV-family Phase 2 lift order: the smallest credible move is SEQUENTIAL not parallel; the dissent above recommends sequencing with per-lift go/no-go. Per CLAUDE.md 'Council conduct no-conservative-bias': I take positions; my Decision 1 + Decision 4 dissents are concrete revision proposals not 'we'll find out empirically' deferrals."* **HARD-EARNED**.

**Assumption-Adversary** — operating within: *"As the dedicated assumption-backdrop-interrogation seat (sextet pact 6th seat per CLAUDE.md Council conduct amendment), the lens is 'WHICH SHARED ASSUMPTION is each decision operating WITHIN, and what would happen if violated?' Decision 1 NSCS06 rescue: shared assumption is that the v8 PLATEAU at 58.892 implies the Path B class is exhausted; the assumption-violation hypothesis is 'maybe v8 PLATEAU is measurement artifact — different DWT depth, different per-class CDF, paired CUDA-vs-CPU axis re-eval, sister wavelet families — were never probed'. PROPOSAL: Decision 1 should include a $0-1 cheap v8 sister-probe wave (different DWT depth + per-class CDF + paired axis) BEFORE Path C hybrid-neural; my dissent below queues this. Decision 2 Tishby IB-pure: shared assumption is that the 4-reducer matrix exhausts the bolt-on form; substrate-native form is OUTSIDE the assumption-backdrop; PROCEED with Phase 2 STAGE 1 re-probe. Decision 3 Wunderkind v2: shared assumption is that per-pair-dominant + per-pair-histogram + per-region-histogram + per-frame-argmax exhausts the SegNet-derived reducer family; the assumption-violation hypothesis is 'maybe there's a NEW reducer methodology outside the frozen-SegNet-output family entirely (e.g., differentiable scorer-via-distillation reducer, PoseNet-residual-derived reducer)'. PROPOSAL: Decision 3 permanent DEFER is correct IF AND ONLY IF the DEFER memo explicitly enumerates ≥3 candidate NEW reducer methodologies that future research could surface (per Catalog #308 META-pattern E). Decision 4 NeRV-family: shared assumption is that the 6 NeRV variants are 6 distinct points on the implicit-neural-representation manifold; the assumption-violation hypothesis is 'maybe they're 6 parameterizations of the SAME manifold point'. My dissent (above) queues sequential dispatch with per-lift orthogonality verification before parallel firing."* **HARD-EARNED**.

---

## Decision 1: NSCS06 paradigm rescue path

### Empirical situation

NSCS06 v8 Path B (DB4 wavelet residual) harvested 58.892 [diagnostic-CPU Modal advisory only] today (Subagent A commit `2207dc4ab`; Modal call `fc-01KRRJNSXCJ48W4DW53YE02PAE`). v7 Path A landed 58.89 (chroma optical flow). **Δ = -0.002 absolute** — essentially flat within measurement noise on a CPU-axis advisory. v8 Section 18 decision tree band `>40` → DEFER pending grand-council symposium ratification.

Per Subagent A's harvest memo + v8 Section 18 decomposition: pose dominates at 30.94 (52% of score) / seg 25.27 (43%) / rate 2.67 (5%). The DB4 wavelet decorrelation addressed SEG + RATE but NOT POSE. v8's Wyner-Ziv per-pair pose-residual coding was the mechanism that DID target pose but didn't fire as designed (per v8 design §18 expected pose_avg drop to [4.0, 15.0] from 95.76; actual measurement IF inferred from the 58.892 total ≈ pose_avg 95.76 unchanged, ergo Wyner-Ziv mechanism inactive or insufficient).

### Decision options enumerated (per operator's question)

| # | Option | Predicted band | Cost | Mechanism cited |
|---|---|---|---|---|
| 1a | **Path C hybrid-neural** | [0, 15] medal-band-class per v8 §76 | $30-80 | small NeRV residual head on top of v8 wavelet output; cargo-cult #5 WAIVER unwind |
| 1b | **Pose-axis attack** (NEW path) | [10, 25] | $20-50 | substrate variant focusing on pose reconstruction (e.g., dense per-pixel pose residual encoding) |
| 1c | **Per-class-conditional CDF unwind** (cargo-cult #1) | [25, 35] (within-class refinement; plateau-adjacent) | $15-40 | replace per-subband uniform CDF with per-(subband, SegNet class) CDF |
| 1d | **DEFER** | n/a | $0 | reclassify NSCS06 from POTENTIAL-ASYMPTOTIC-PURSUIT to FRONTIER-PURSUIT per FALSIFICATION-AUDIT-v2; preserve substrate for NSCS01 × v8 composition design memo only |

### Sextet-pact deliberation (Decision 1)

**Shannon (LEAD)**: PROCEED with **Option 1a (Path C hybrid-neural) PRIMARY + Option 1b (pose-axis attack) PARALLEL**. Information-theoretic rationale: the v8 wavelet decorrelation reaches H(X) for the SPATIAL CODEC layer; further analytical refinement (Option 1c per-class-conditional CDF) is within-class within the spatial codec class and asymptotically yields O(1) bits/symbol improvement per Mallat hierarchical prior at most ~1-3 score points. The DOMINANT contribution is POSE; Options 1a + 1b BOTH target pose directly. Option 1a does it via learned residual (neural component captures the per-pair pose-prediction error structure the wavelet codec cannot represent). Option 1b does it via substrate redesign (pose-axis-first architecture). Running BOTH IN PARALLEL maximizes expected information gain.

**Dykstra (CO-LEAD)**: PROCEED with **Option 1a + Option 1b PARALLEL** per Shannon. Dykstra-feasibility rationale: the v8 polytope has the pose-axis constraint dominating; Option 1a expands the feasibility polytope by adding the neural-residual constraint dimension (strictly more constraints satisfied; lower minimum achievable); Option 1b expands the polytope along the pose-axis specifically (different constraint dimension, potentially orthogonal). Running both probes the two expansion axes independently and the intersection of their results identifies the true frontier.

**Yousfi**: PROCEED with **Option 1a PRIMARY** per the analytical-codec-class is at its UNIWARD limit on the contest-video-class; further analytical refinement (Option 1c) does not unwind the dominant pose-axis bottleneck. Option 1b (pose-axis attack) is plausible but the substrate redesign cost ($20-50) is uncertain; Option 1a (hybrid-neural residual on v8) has a clearer mechanism (NeRV residual learns the per-pair pose-prediction error). RECOMMENDATION: Path C scoped to ≤100K params + FP4 + brotli per Contrarian dissent.

**Fridrich**: PROCEED with **Option 1a + Option 1b PARALLEL** per UNIWARD frequency-domain principle: textured-region capacity is what Option 1b would allocate (pose-residual is per-pixel-detectable in textured regions); Option 1a's neural residual would also allocate this implicitly. Running both probes the two allocation strategies independently.

**Contrarian**: **DISSENT** (recorded in council_dissent frontmatter above). Recommended REVISION: **Option 1c (v9 per-class-conditional CDF unwind) FIRST** — smallest credible move; if v9 plateaus too (Δ < 0.005 vs v8), THEN Option 1a + Option 1b PARALLEL with operator approval for cargo-cult #5 WAIVER unwind. **Rationale**: the Strip-Everything PARADIGM's analytical-codec discipline was the value-add; reintroducing neural at the FIRST sign of plateau is the canonical reflex this paradigm was designed to extinct.

**Assumption-Adversary**: PROCEED with **Option 1a + Option 1b PARALLEL** but with **PRE-DISPATCH $0-1 cheap v8 sister-probe wave** addressing the assumption that v8 PLATEAU is genuine: probe different DWT depth (DB4 depth=3 sweep), per-class CDF variant, paired CUDA-vs-CPU axis re-eval, sister wavelet families (Daubechies-6, Symlet-4). Cost: $0-1 (compress-only on Modal CPU is $0.10-0.50 per probe; total ≤$1). If any sister-probe lands at <50, the v8 PLATEAU assumption is FALSIFIED and Decision 1 reverts to v9 PROCEED. If all sister-probes land at 55-65, the v8 PLATEAU is RATIFIED and Decision 1 PROCEEDS to Options 1a + 1b PARALLEL.

### Decision 1 final verdict

**VERDICT: PROCEED-PATH-C-HYBRID-NEURAL (preferred) + PARALLEL-POSE-AXIS-EXPERIMENT (NEW path; $20-50 budget) AFTER $0-1 cheap v8 sister-probe wave per Assumption-Adversary**.

Vote tally: 5-of-6 PROCEED with Options 1a + 1b PARALLEL; 1-of-6 DISSENT (Contrarian) recommending Option 1c FIRST. Per CLAUDE.md "Council conduct no-conservative-bias" + Shannon tie-break authority: the dissent is RECORDED in frontmatter and the Contrarian's revision proposal (scope Option 1a Path C to ≤100K params + FP4 + brotli) is ACCEPTED as a binding constraint on the Path C design. Per Assumption-Adversary: a $0-1 sister-probe wave precedes Options 1a + 1b dispatch.

**Cost envelope**:
- Assumption-Adversary cheap sister-probe wave: **$0-1** (4 Modal CPU probes × $0.10-0.50)
- Option 1a Path C hybrid-neural: **$30-80** (1 Modal T4 smoke + 1 paired CPU eval; Quantizr ≤100K-param + FP4 + brotli budget per Contrarian)
- Option 1b pose-axis attack: **$20-50** (1 Modal T4 smoke + 1 paired CPU eval; substrate redesign budget)
- **Total Decision 1 envelope: $50-131**

**Reactivation criteria for Decision 1**:
- Assumption-Adversary sister-probe: if any probe lands <50, REVERT to v9 PROCEED + cancel Options 1a + 1b
- Option 1a Path C: if paired CUDA/CPU score lands in [0, 15] band, BREAKTHROUGH; expand budget for full-run + queue Path D class-shift
- Option 1b pose-axis attack: if paired CUDA/CPU score lands in [10, 25] band, ratify pose-axis-first substrate class
- If Options 1a + 1b BOTH land >40, Decision 1d (DEFER) is invoked and NSCS06 is reclassified per FALSIFICATION-AUDIT-v2 A5

**NSCS01 × v8 composition preservation**: per v8 design §13 + FALSIFICATION-AUDIT-v2 A1 reclassification, NSCS06 lineage IS the canonical ASYMPTOTIC-PURSUIT-via-iterative-cargo-cult-unwind prototype. v8 → v9 → v10 trajectory continues regardless of Path C/1b outcomes; NSCS01 × v8 composition design memo is a separate sister deliverable (cost $0 design; $10 composition smoke after both NSCS01 and v8 land).

---

## Decision 2: Tishby IB-pure DEFER status

### Empirical situation

Per Subagent B's empirical 4-reducer matrix (commits `c1633d18b` + `9f1f618a7`): both Wunderkind G1 v2 + Tishby IB-pure return PARTIAL/NO-MEANINGFUL across ALL 4 alternative reducers (per_pair_dominant / per_pair_histogram / per_region_histogram / per_frame_argmax). Best signal `per_region_histogram MI=0.0474 bits/symbol` but Dykstra-feasibility STRUCTURALLY NEGATIVE (sidecar overhead 14-18× the savings).

Per Subagent F's earlier T3 batched Phase 2 council (commit `20972f076`, sister `grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md`): verdict was DEFER-PENDING-SIGNAL-EXTRACTION-PIVOT. 

Per Tishby IB-pure scoping memo §4 (`tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md`): the SUBSTRATE-NATIVE form treats `q(t|x)` as the architectural OBJECTIVE not a sidecar probe. The variational encoder weights ARE the side-information; the IB Lagrangian `L_VIB = E[-log p(y|t)] + β·KL(q(t|x) ‖ p(t))` is the END-TO-END training objective. The BLOCKING reactivation gate per Tishby Appendix A.1 Criterion #2 is **substrate-native-latent re-probe** — train the variational encoder for ≥100 epochs at the target β operating point and measure I(T;Y)/I(X;T) post-convergence, NOT pre-training using frozen-SegNet-output reducers.

### Decision options enumerated (per operator's question)

| # | Option | Predicted outcome | Cost |
|---|---|---|---|
| 2a | **Fund Phase 2 STAGE 1 Modal A100 100ep proxy** | substrate-native-latent re-probe; measures I(T;Y)/I(X;T) at convergence | $5-10 |
| 2b | **DEFER permanently** | until NEW reducer methodology (substrate-native IB posterior q(t|x) OR PoseNet-derived side-info OR SegNet pre-argmax logits OR trained-from-scratch substrate-native latents) is proposed | $0 |

### Sextet-pact deliberation (Decision 2)

**Shannon (LEAD)**: PROCEED with **Option 2a (Phase 2 STAGE 1 100ep proxy)**. Information-theoretic rationale: the 4-reducer matrix probed the BOLT-ON form (frozen-SegNet-output reducers feeding a sidecar); the substrate-native form `q(t|x)` is a STRICTLY MORE EXPRESSIVE prior than any frozen-SegNet reducer because the variational encoder can learn ANY measurable function of x including arbitrary linear/nonlinear combinations of pixel intensities, SegNet logits pre-argmax, PoseNet features, etc. The 4-reducer matrix's PARTIAL/NO-MEANINGFUL verdict is HARD-EARNED for the bolt-on form; it is CARGO-CULTED for the substrate-class form. Per Catalog #307 paradigm-vs-implementation distinction.

**Dykstra (CO-LEAD)**: PROCEED with **Option 2a**. Dykstra-feasibility rationale: the substrate-native variational encoder defines a DIFFERENT feasibility polytope from the 4-reducer matrix's polytopes; intersection with the contest-video-class polytope is empirically unknown. The $5-10 Phase 2 STAGE 1 100ep proxy IS the canonical Bayesian sequential design measurement to reveal the intersection.

**Yousfi**: PROCEED with **Option 2a**. The substrate-native form IS the canonical Atick-Redlich + Wyner-Ziv cooperative-receiver architecture made primary — this is the canonical asymptotic-pursuit move per T4 SYMPOSIUM 4×4 floor matrix. The 4-reducer matrix probed adjacent surfaces; the substrate-native is the primary surface.

**Fridrich**: PROCEED with **Option 2a**. The variational encoder + KL bottleneck IS the canonical inverse-steganalysis loss made architectural (encoder learns what scorer detects; bottleneck prevents over-representation). Worth $5-10 to empirically test.

**Contrarian**: PROCEED with **Option 2a** BUT with **HARD COST CAP** at $10 and **TIME CAP** at 100ep Modal A100. Per CLAUDE.md "Forbidden premature KILL": the v2 council's DEFER verdict is reactivation-gated on substrate-native re-probe; the canonical reactivation IS the Phase 2 STAGE 1 100ep proxy. Do not allow scope creep ("we should also probe MINE and InfoNCE variants" → blows the cap). 100ep proxy at A100 ~$5-10; if score lands below 0.20 [contest-CUDA T4 proxy], Phase 2 STAGE 2 1000ep is unlocked; if score lands above 0.30, Tishby IB-pure is PERMANENTLY DEFERRED per the operator's question's Option 2b.

**Assumption-Adversary**: PROCEED with **Option 2a**. The shared assumption underlying Subagent B's PARTIAL/NO-MEANINGFUL verdict is that frozen-SegNet-output reducers are representative of the SegNet conditioning class; the substrate-native variational encoder is OUTSIDE this assumption-backdrop (it doesn't use frozen SegNet output at all; it learns its own representation from raw pixel pairs). Decision 2a is the assumption-violation hypothesis empirically tested.

### Decision 2 final verdict

**VERDICT: FUND PHASE 2 STAGE 1 MODAL A100 100ep PROXY ($5-10 budget) per Tishby Appendix A.1 Criterion #2; NOT PERMANENTLY DEFER.**

Vote tally: 6-of-6 PROCEED with Option 2a. No dissent. Contrarian-recommended HARD COST CAP $10 + TIME CAP 100ep is ACCEPTED as a binding constraint on Phase 2 STAGE 1 dispatch.

**Cost envelope**:
- Phase 2 STAGE 1 100ep Modal A100 proxy: **$5-10** (HARD CAP $10)
- **Total Decision 2 envelope: $5-10**

**Reactivation criteria for Decision 2**:
- 100ep proxy lands [contest-CUDA T4] score < 0.20: Phase 2 STAGE 2 1000ep full unlocked
- 100ep proxy lands [contest-CUDA T4] score in [0.20, 0.30]: pivot decision (Phase 2 council); 50/50 STAGE 2 vs DEFER
- 100ep proxy lands [contest-CUDA T4] score > 0.30: PERMANENT DEFER per Option 2b; reactivation requires NEW methodology proposal

**30-day retrospective** scheduled for 2026-06-15: if 100ep proxy was funded but lands above 0.30 AND no Phase 2 STAGE 2 follow-up was triggered, the council retrospective reviews whether the substrate-native re-probe verdict was correct OR whether additional reducer methodologies should have been proposed pre-DEFER.

---

## Decision 3: Wunderkind G1 v2 DEFER status

### Empirical situation

Per Subagent B's same empirical 4-reducer matrix (commits `c1633d18b` + `9f1f618a7`) on Wunderkind G1 v2: same PARTIAL/NO-MEANINGFUL pattern as Tishby IB-pure. Per T2 council `grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md`: Q1 VERDICT was RATIFY-FALSIFICATION-OF-THE-SPECIFIC-v2-REDUCER + REQUEST-REINVESTIGATION-OF-ALTERNATIVE-REDUCERS-BEFORE-CLASS-WIDE-DEFERRAL.

Per Wunderkind v3 design memo (commit `d9f01baa0`): v3 pivoted to per-pair adaptive sigma (NOT class-conditional) — sigma table is 1200 rows (one per pair) entropy-coded with a class-INDEPENDENT prior derived from the EMPIRICAL per-pair residual variance distribution. v3 SKIPS the class-conditioning entirely. v3 predicted CPU band [0.1893, 0.1963] per PR102-drift extrapolation (Contrarian CARGO-CULTED per T2 council); Shannon first-principles CPU derivation projects ~0.1907 (marginal frontier-break, -0.0021 below A1).

### Decision options enumerated (per operator's question)

| # | Option | Predicted outcome | Cost |
|---|---|---|---|
| 3a | **Permanent DEFER v2 class-wide pending NEW reducer methodology** | per Subagent B's recommendation; preserves research-exhaustion-pending status per CLAUDE.md "Forbidden premature KILL" | $0 |
| 3b | **Trigger v3 per-pair adaptive sigma Phase 2 council + L1 SCAFFOLD build** | per Wunderkind v3 memo §22 op-routable #1; v3 doesn't depend on class-conditional info; would empirically validate per-pair adaptive sigma achieves predicted band [0.221, 0.228] CUDA / [0.189, 0.196] CPU CONDITIONAL on substrate-native data | $0 design + $5-15 Phase 2 STAGE 1 smoke later |

### Sextet-pact deliberation (Decision 3)

**Shannon (LEAD)**: PROCEED with **BOTH Option 3a AND Option 3b**. Information-theoretic rationale: v2 class is structurally falsified for the per-pair-dominant + per-pair-histogram + per-region-histogram + per-frame-argmax reducer family on dashcam content (per Subagent B); v3 per-pair adaptive sigma is OUTSIDE this class (pair-index conditioning not class conditioning). Per CLAUDE.md "Forbidden premature KILL": v2 DEFER permanently-pending-NEW-reducer-methodology is the correct verdict (not KILL); v3 is the canonical class-shift escape. BOTH actions are non-conflicting.

**Dykstra (CO-LEAD)**: PROCEED with **BOTH Option 3a AND Option 3b**. Dykstra-feasibility rationale: v2 reducer family's polytope is degenerate on dashcam class (per Subagent B's empirical 4-reducer matrix); v3 per-pair adaptive sigma defines a NEW polytope along the pair-index axis (different dimension entirely); Phase 2 STAGE 1 100ep proxy IS the canonical measurement.

**Yousfi**: PROCEED with **BOTH Option 3a AND Option 3b**. Per UNIWARD: per-pair adaptive sigma allocates capacity to per-pair texture-region complexity; this is structurally distinct from per-class adaptive sigma. v3 IS the canonical escape from v2's class-wide degeneracy.

**Fridrich**: PROCEED with **BOTH Option 3a AND Option 3b**. v3 per-pair adaptive sigma table + class-INDEPENDENT prior IS the canonical Markov-1 AAC sister approach made structural; worth Phase 2 STAGE 1 empirical test.

**Contrarian**: PROCEED with **BOTH Option 3a AND Option 3b** BUT REQUIRE v3 Section 14 disambiguator probe (per-pair sigma vs MLP entropy; $0 CPU; ~5 min) FIRST per T2 council Q3 verdict. Per v3 memo §19 reactivation criterion #1: the disambiguator is BLOCKING. If disambiguator shows v3 per-pair sigma is REDUNDANT with Z3 v2 per-pair MLP (Section 14 Interpretation B), v3 is ABANDONED and the $5-15 Phase 2 STAGE 1 budget is released. If disambiguator shows v3 per-pair sigma BEATS MLP (Section 14 Interpretation A), Phase 2 STAGE 1 PROCEEDS.

**Assumption-Adversary**: PROCEED with **BOTH Option 3a AND Option 3b** BUT v2 PERMANENT DEFER memo MUST explicitly enumerate ≥3 candidate NEW reducer methodologies that could surface in the future (per Catalog #308 META-pattern E). Examples: (1) differentiable scorer-via-distillation reducer (PoseNet/SegNet distilled into a small reducer network that outputs continuous class probabilities); (2) PoseNet-residual-derived reducer (PoseNet outputs are pose-residuals which are a 6-DOF continuous signal — conditioning on these is OUTSIDE the SegNet class-discrete family); (3) spatial-region-with-temporal-context reducer (per-region histogram + per-region motion-vector from optical flow). Per the operator's question: NEW reducer methodology IS the reactivation gate. Enumerating 3 candidates in the DEFER memo satisfies Catalog #308 + provides concrete reactivation targets.

### Decision 3 final verdict

**VERDICT: DEFER V2 CLASS-WIDE PERMANENTLY PENDING NEW REDUCER METHODOLOGY + TRIGGER V3 PHASE 2 COUNCIL AND L1 SCAFFOLD per operator-approved Wunderkind v3 memo §22 op-routable #1.**

Vote tally: 6-of-6 PROCEED with BOTH Options 3a AND 3b. Contrarian-recommended v3 Section 14 disambiguator probe BLOCKING requirement is ACCEPTED. Assumption-Adversary-recommended ≥3 NEW reducer methodology enumeration in v2 DEFER memo is ACCEPTED.

**Cost envelope**:
- v2 DEFER memo (text-only): **$0** (sister-subagent landing per Catalog #229 premise-verification-before-edit discipline)
- v3 Section 14 disambiguator probe ($0 CPU per T2 council Q3 BLOCKING): **$0**
- v3 Phase 2 STAGE 1 100ep Modal A100 smoke (CONDITIONAL on Section 14 Interpretation A): **$5-15**
- **Total Decision 3 envelope: $0-15**

**Reactivation criteria for v2 DEFER (Option 3a)**:
- New reducer methodology surfaces from research (the ≥3 candidates enumerated in DEFER memo OR a 4th surfaces) → re-evaluate v2 class-wide DEFER
- 30-day retrospective: review whether v2 DEFER was correct OR whether the BOLT-ON form should have been pursued via NEW methodologies sooner

**Reactivation criteria for v3 Phase 2 (Option 3b)**:
- v3 Section 14 disambiguator Interpretation A (per-pair sigma BEATS MLP) → Phase 2 STAGE 1 100ep PROCEEDS
- v3 Section 14 disambiguator Interpretation B (per-pair sigma REDUNDANT with MLP) → v3 ABANDONED; Phase 2 STAGE 1 budget released
- 30-day retrospective: 2026-06-15 per Wunderkind G1 v2 deferred_substrate_retrospective_due_utc

---

## Decision 4: NeRV-family expansion (#522) Phase 2 lift

### Empirical situation

Per FALSIFICATION-AUDIT-v2 (`c5e4953e6`) Tier 2 → Tier 1 promotion: Wave 3 NeRV-family (BlockNeRV/FFNeRV/DSNeRV/HiNeRV/TCNeRV + 2 bolt-ons + sane_hnerv) promoted from Tier 2 mixed-signals to Tier 1 operational-recovery + FRONTIER-PURSUIT classification. All 6 NeRV trainers verified to have `_full_main` defined (file inspection: experiments/train_substrate_{block_nerv,ds_nerv,ff_nerv,hi_nerv,sane_hnerv,tc_nerv}.py).

Per HORIZON-CLASS directive: NeRV-family is FRONTIER-PURSUIT (predicted CPU bands [0.13, 0.18] per HiNeRV paper citation; per the FALSIFICATION-AUDIT-v2 P3 NSCS01 sister estimation). K=13 LEVEL-1 schedule's 5-of-13 FRONTIER-PURSUIT bucket (38.5% per Donoho-Tanner) is the canonical consumption surface.

### Decision options enumerated (per operator's question)

| # | Option | Approach | Cost envelope |
|---|---|---|---|
| 4a | **Lift all 6 NeRV variants in parallel** | maximum parallelism; per FALSIFICATION-AUDIT-v2 P3 "5 substrates × class-shift potential = high expected-info-gain" | $60-180 (6 × $10-30 per substrate; 6 × ($5-15 Modal smoke + $5-15 paired full)) |
| 4b | **Lift sequentially with per-substrate go/no-go arbitration** | Bayesian sequential design per K=13 §8; each lift's verdict feeds next lift; lower budget per Assumption-Adversary dissent | $30-90 (first 3 substrates in sequence; remaining 3 conditional on first 3 verdicts) |
| 4c | **Lift top 2 in parallel + 4 sequential** | hybrid; parallel where evidence supports orthogonality, sequential otherwise | $40-120 |

### Sextet-pact deliberation (Decision 4)

**Shannon (LEAD)**: PROCEED with **Option 4c (hybrid: top 2 in parallel + 4 sequential)**. Information-theoretic rationale: HiNeRV literature predicted band [0.13, 0.18] is the highest-quality empirical prior; sane_hnerv has pre-existing CUDA infrastructure (Catalog #187 HNeRV training parity guard). These 2 substrates are most-likely orthogonal (HiNeRV is the published canonical; sane_hnerv is our internal infrastructure-mature variant). Firing these 2 IN PARALLEL maximizes parallel-actuator throughput (per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" NON-NEGOTIABLE). The remaining 4 (FFNeRV / DSNeRV / TCNeRV / BlockNeRV) are sequential per per-lift go/no-go arbitration per K=13 §8 Bayesian sequential design.

**Dykstra (CO-LEAD)**: PROCEED with **Option 4c**. Dykstra-feasibility rationale: each NeRV variant has its own feasibility polytope; HiNeRV + sane_hnerv polytopes are most-likely orthogonal (different architectural primitives — HiNeRV uses positional encoding hierarchy; sane_hnerv uses our internal HNeRV-canonical architecture); polytope intersection probability is high enough to justify parallel dispatch. Remaining 4 variants share more architectural primitives with each other (FFNeRV/DSNeRV both frequency-domain; TCNeRV/BlockNeRV both temporal-decomposition); sequential dispatch with verdict-feedback is the canonical Bayesian sequential design measurement.

**Yousfi**: PROCEED with **Option 4c**. The contest-CPU axis is the binding ranking surface; HiNeRV's [0.13, 0.18] band IS the implicit-neural-representation class anchor. Sane_hnerv represents our infrastructure-mature internal variant. These 2 should land FIRST to establish the class anchor empirically. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": paired CUDA + CPU dispatch for both.

**Fridrich**: PROCEED with **Option 4c**. UNIWARD frequency-domain principle suggests FFNeRV/DSNeRV may share frequency-allocation strategy on contest video; sequential dispatch reveals whether they're orthogonal or redundant. HiNeRV/sane_hnerv parallel-first per Shannon/Yousfi/Dykstra rationale.

**Contrarian**: PROCEED with **Option 4c** BUT with **per-substrate cost cap $25** (not $30) and **batch cost cap $90** (not $180) per Assumption-Adversary dissent recommendation. Per CLAUDE.md "Vast.ai cost paranoia" + the dispatch-budget-paranoia non-negotiables: NeRV variants are MORE expensive on average (HiNeRV paper reports ~24h on V100 for full training; even smoke at 100ep on A100 is non-trivial). Cap per-substrate at $25; if batch cost exceeds $90, halt and council-arbitrate continuation.

**Assumption-Adversary**: PROCEED with **Option 4c PER MY DISSENT (above) but per Contrarian's cost cap $90 batch / $25 per-substrate**. The recommended sequence is:
1. **Tier-A (PARALLEL)**: HiNeRV + sane_hnerv — orthogonality assumption: PR-published canonical vs our infrastructure-mature internal variant
2. **Tier-B (SEQUENTIAL, conditional on Tier-A producing ≥1 paired CPU score)**: FFNeRV (frequency-domain class-shift orthogonal axis), DSNeRV (distortion-stratified extends FFNeRV; conditional on FFNeRV verdict), TCNeRV (temporal-coherent baseline), BlockNeRV (block-decomposition baseline)
3. **Per-substrate go/no-go**: each Tier-B lift is GATED by the cumulative evidence from prior Tier-A + Tier-B verdicts; orthogonality between adjacent variants must be empirically confirmed (per CLAUDE.md "Apples-to-apples evidence discipline") before next-tier-substrate fires

### Decision 4 final verdict

**VERDICT: PHASE-2-LIFT-ORDER hybrid Tier-A PARALLEL (HiNeRV + sane_hnerv) + Tier-B SEQUENTIAL (FFNeRV → DSNeRV → TCNeRV → BlockNeRV) per Option 4c + Assumption-Adversary sequence + Contrarian cost caps.**

Vote tally: 6-of-6 PROCEED with Option 4c hybrid. Contrarian cost caps (per-substrate $25 / batch $90) ACCEPTED. Assumption-Adversary sequential-conditional-on-Tier-A-orthogonality ACCEPTED.

**Cost envelope** (per Assumption-Adversary sequence + Contrarian caps):
- **Tier-A PARALLEL (HiNeRV + sane_hnerv)**: $25 + $25 = **$50** (cap-bound; per-substrate $5-15 Modal smoke + $5-15 paired CUDA/CPU full + Catalog #167 smoke-before-full discipline; Tier-A scope must include paired CUDA + CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
- **Tier-B SEQUENTIAL FFNeRV (conditional on Tier-A producing ≥1 paired CPU score)**: $25 cap
- **Tier-B SEQUENTIAL DSNeRV (conditional on FFNeRV verdict)**: $25 cap
- **Tier-B SEQUENTIAL TCNeRV (conditional on cumulative Tier-A+FFNeRV+DSNeRV verdicts)**: $25 cap (max — likely halted before this)
- **Tier-B SEQUENTIAL BlockNeRV (conditional on cumulative verdicts)**: $25 cap (max — likely halted before this)
- **Batch hard cap**: **$90** per Contrarian
- **Total Decision 4 envelope: $50-90** (Tier-A $50 + Tier-B SEQUENTIAL halt-at-$90-total)

**Sequence rationale per substrate**:

1. **HiNeRV** (Tier-A PARALLEL): highest predicted-band quality (literature [0.13, 0.18]); most-mature literature canonical; lift FIRST to establish class anchor
2. **sane_hnerv** (Tier-A PARALLEL): pre-existing CUDA infrastructure + Catalog #187 HNeRV training parity guard already wired; lift FIRST in parallel with HiNeRV for orthogonality test
3. **FFNeRV** (Tier-B sequential #1): frequency-domain class-shift orthogonal axis; lift if Tier-A produces ≥1 paired CPU score with class-shift evidence
4. **DSNeRV** (Tier-B sequential #2): distortion-stratified extends FFNeRV; lift if FFNeRV produces paired CPU score with class-shift evidence AND DSNeRV is empirically orthogonal to FFNeRV
5. **TCNeRV** (Tier-B sequential #3): temporal-coherent baseline; lift conditional on cumulative verdicts
6. **BlockNeRV** (Tier-B sequential #4): block-decomposition baseline; lift conditional on cumulative verdicts

**Reactivation criteria for Decision 4**:
- Tier-A both substrates land paired CPU < 0.180: BREAKTHROUGH; expand Tier-B budget; queue Path C class-shift composition designs
- Tier-A both substrates land paired CPU in [0.180, 0.200]: PROCEED with Tier-B per sequence
- Tier-A both substrates land paired CPU > 0.200: HALT Decision 4; reclassify NeRV-family to Tier 2 demoted-priority per HORIZON-CLASS plateau-adjacent
- Each Tier-B lift halts if cumulative batch cost approaches $90 OR if last 2 paired scores show no orthogonality (Δ < 0.005 between adjacent variants)

**30-day retrospective**: NOT triggered for Decision 4 (no DEFER verdict; all 6 substrates lift OR halt per empirical criteria).

---

## Final summary

### Binding T3 verdicts (4 decisions)

| # | Decision | Verdict | Cost envelope | Mission contribution |
|---|---|---|---:|---|
| 1 | NSCS06 paradigm rescue path | **PROCEED Path C hybrid-neural + parallel pose-axis attack AFTER $0-1 cheap v8 sister-probe wave** (Contrarian-scoped Path C ≤100K params + FP4 + brotli; Assumption-Adversary $0-1 sister-probe wave first) | **$50-131** | `frontier_breaking` |
| 2 | Tishby IB-pure DEFER status | **FUND Phase 2 STAGE 1 Modal A100 100ep proxy** (HARD CAP $10; substrate-native re-probe per Tishby Appendix A.1 Criterion #2) | **$5-10** | `frontier_breaking` |
| 3 | Wunderkind G1 v2 DEFER status | **DEFER v2 class-wide PERMANENTLY pending NEW reducer methodology (≥3 enumerated in DEFER memo) + TRIGGER v3 Phase 2 council and L1 SCAFFOLD per v3 memo §22 op-routable #1** (Contrarian: v3 Section 14 disambiguator BLOCKING; Assumption-Adversary: ≥3 NEW reducer methodologies in v2 DEFER memo) | **$0-15** | `frontier_breaking` |
| 4 | NeRV-family Phase 2 lift order | **Tier-A PARALLEL (HiNeRV + sane_hnerv) + Tier-B SEQUENTIAL (FFNeRV → DSNeRV → TCNeRV → BlockNeRV) per Option 4c hybrid** (Contrarian: per-substrate $25 / batch $90 caps; Assumption-Adversary: per-lift go/no-go orthogonality verification) | **$50-90** | `frontier_breaking` |

**Total batched cost envelope: $105-246**

### Wave 3 K=13 LEVEL-1 fire input

Per the K=13 schedule §4.2 FRONTIER-PURSUIT bucket (5 of 13; 38.5%) + this T3 council's verdicts:

**Recommended K=13 LEVEL-1 FRONTIER-PURSUIT bucket consumption sequence** (5 substrates):
1. **HiNeRV (Tier-A PARALLEL #1)** — $25 cap; lift FIRST as class anchor (per Decision 4)
2. **sane_hnerv (Tier-A PARALLEL #2)** — $25 cap; lift FIRST in parallel for orthogonality test (per Decision 4)
3. **NSCS06 v8 cheap sister-probe wave** — $0-1 (4 Modal CPU probes); BEFORE Decision 1 Options 1a + 1b dispatch (per Assumption-Adversary dissent on Decision 1)
4. **NSCS06 Path C hybrid-neural** — $30-80; CONDITIONAL on sister-probe wave RATIFYING v8 PLATEAU (per Decision 1)
5. **NSCS06 pose-axis attack** — $20-50; PARALLEL with Path C per Decision 1

**Recommended K=13 LEVEL-1 ASYMPTOTIC-PURSUIT bucket consumption** (3 of 13; 23.1%):
1. **Tishby IB-pure Phase 2 STAGE 1 100ep proxy** — $5-10 (HARD CAP); per Decision 2 (replaces Tishby IB-pure scoping memo slot in K=13)
2. **Z6 + Rudin floor substrate** — $0 (L1 SCAFFOLDs already landed per Subagent F earlier T3 council; queued for LEVEL-2 if Phase 2 unlocks)

**Recommended K=13 LEVEL-1 PLATEAU-ADJACENT bucket** (4 of 13; 30.8% — per K=13 §4.1):
- Lane 17 IMP, apogee_int4, NSCS01, NSCS02 (per K=13 §4.1 as listed)

**Recommended K=13 LEVEL-1 DISAMBIGUATOR bucket** (1 of 13; 7.7%):
- Wunderkind v3 Section 14 disambiguator probe ($0 CPU; ~5 min); BLOCKING per Decision 3 + T2 council Q3

**Tier-B SEQUENTIAL NeRV-family slots NOT in K=13 LEVEL-1**: FFNeRV / DSNeRV / TCNeRV / BlockNeRV are queued for K=13 LEVEL-2 or LEVEL-3 conditional on Tier-A Decision 4 verdict producing paired CPU evidence. Budget reserved: $50-60.

### Op-routables for operator decision queue (ranked by mission-contribution × cost)

| # | Action | Cost | Priority | Decision |
|---|---|---:|---|---|
| 1 | Land NSCS06 v8 cheap sister-probe wave (4 Modal CPU probes; different DWT depth + per-class CDF + paired axis re-eval + sister wavelet families) | $0-1 | **HIGHEST** | Decision 1 (pre-Path C gate) |
| 2 | Dispatch Tier-A PARALLEL HiNeRV + sane_hnerv (Modal smoke + paired CUDA/CPU full per Catalog #167) | $50 | **HIGHEST** | Decision 4 |
| 3 | Dispatch Tishby IB-pure Phase 2 STAGE 1 100ep Modal A100 proxy | $5-10 | **HIGHEST** | Decision 2 |
| 4 | Land v3 Section 14 disambiguator probe (per-pair sigma vs MLP entropy; $0 CPU) | $0 | **HIGH** | Decision 3 (BLOCKING for v3 Phase 2) |
| 5 | Land Wunderkind v2 PERMANENT DEFER memo with ≥3 NEW reducer methodology enumeration per Catalog #308 | $0 | **HIGH** | Decision 3 |
| 6 | CONDITIONAL: Dispatch NSCS06 Path C hybrid-neural smoke (if op-routable #1 ratifies v8 PLATEAU) | $30-80 | **MEDIUM-HIGH** | Decision 1 |
| 7 | CONDITIONAL: Dispatch NSCS06 pose-axis attack smoke (parallel with #6) | $20-50 | **MEDIUM-HIGH** | Decision 1 |
| 8 | CONDITIONAL: Dispatch Wunderkind v3 Phase 2 STAGE 1 100ep smoke (if op-routable #4 Interpretation A) | $5-15 | **MEDIUM** | Decision 3 |
| 9 | CONDITIONAL: Dispatch Tier-B SEQUENTIAL FFNeRV (if Tier-A op-routable #2 produces paired CPU < 0.200) | $25 | **MEDIUM** | Decision 4 |
| 10 | CONDITIONAL: Dispatch remaining Tier-B SEQUENTIAL substrates (DSNeRV → TCNeRV → BlockNeRV) per cumulative verdicts | $25 each (cap $90 batch) | **MEDIUM-LOW** | Decision 4 |

### Compliance per CLAUDE.md non-negotiables

- **CLAUDE.md "Council conduct no-conservative-bias"**: all 4 decisions took positions; no "we'll find out empirically" deferrals without paired concrete probe-or-experiment proposals
- **CLAUDE.md "Forbidden premature KILL"**: Decision 3 v2 DEFER memo enumerates ≥3 reactivation criteria; no permanent kills proposed
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"**: Decision 4 Tier-A explicitly requires paired CUDA + CPU dispatch; all submission-eligible decisions tagged with axis requirements
- **Catalog #117/#157/#174 canonical commit serializer with --expected-content-sha256**: pending commit
- **Catalog #128/#131/#245 fcntl-locked JSONL discipline**: anchor persistence via canonical `tac.council_continual_learning.append_council_anchor`
- **Catalog #206 subagent crash-resume discipline**: 4 checkpoints (init / post-pre-flight / post-draft / post-commit); complete-status checkpoint pending on completion
- **Catalog #229 premise-verification-before-edit**: 11+ PVs pre-write (see 9-dim Dimension 4 above)
- **Catalog #230 sister-subagent ownership map**: read-only on source code / preflight.py / CLAUDE.md / substrate trainers / recipes / scripts; writes to ONE memo + ONE anchor; no sister subagents in flight (Wave 1 just landed; clean scope)
- **Catalog #240 recipe-vs-trainer-state consistency**: each Decision 4 NeRV trainer's `_full_main` verified implemented (file inspection); each Decision 1 + 2 + 3 sub-action references its recipe state
- **Catalog #270 dispatch-optimization-protocol Tier 1+2+3**: each dispatch in op-routable queue MUST pass `verify_dispatch_protocol_complete` before paid GPU meter starts
- **Catalog #292 per-deliberation explicit assumption-statements**: per-member operating-within section above (6 members × HARD-EARNED classifications)
- **Catalog #294 9-dimension success checklist evidence section**: above
- **Catalog #296 Dykstra-feasibility check**: Decision 1 NSCS06 v8 cited Section 18 FEASIBLE polytope [0.40, 28.76]; Decision 2 Tishby IB-pure substrate-native polytope unknown (Phase 2 STAGE 1 measures empirically); Decision 3 v3 polytope unknown (Section 14 disambiguator measures empirically); Decision 4 NeRV-family per-substrate polytopes unknown (Tier-A measures empirically)
- **Catalog #300 v2 council hierarchy framework**: T3 tier; mission_alignment frontmatter present; 4 binding decisions; 30-day retrospective scheduled for Decision 3 v2 DEFER (sister: Decision 2 Tishby IB-pure retrospective only triggered if Phase 2 STAGE 1 lands > 0.30 and PERMANENT DEFER triggered)
- **Catalog #305 observability surface**: each Decision's empirical anchor cited with sha256/call_id/axis tag; each verdict's reactivation criteria are operator-readable
- **Catalog #307 paradigm-vs-implementation falsification distinction**: applied to Decision 1 (v8 PLATEAU is IMPLEMENTATION-level; PARADIGM intact) + Decision 2 (4-reducer matrix is IMPLEMENTATION-level falsification of BOLT-ON form; substrate-native form is separate paradigm)
- **Catalog #308 probe-methodology-as-false-falsification**: applied to Decision 3 (v2 DEFER memo MUST enumerate ≥3 NEW reducer methodologies)
- **CLAUDE.md HORIZON-CLASS standing directive**: each Decision tagged with horizon_class (Decision 1 NSCS06 ASYMPTOTIC; Decision 2 Tishby IB-pure ASYMPTOTIC; Decision 3 v3 FRONTIER; Decision 4 NeRV-family FRONTIER)
- **CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW"**: this council IS an instance per Catalog #291 (T3 batched format adds the per-deliberation assumption-statement discipline per Fix-7; sister of `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`)
- **CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD"**: this council's verdicts preserve per-decision UNIQUE engineering paths (no canonical-vs-unique reflex toward force-fitting all 4 decisions into the same template; each decision has distinct empirical anchors + sextet-pact maps + verdict criteria)

### Sister regression

- **No `experiments/`, `src/tac/`, `tools/`, `scripts/`, or `submissions/` files modified** (read-only per Catalog #230)
- **`.omx/research/grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516.md`** created
- **`.omx/state/subagent_progress.jsonl`** appended via canonical helper per Catalog #131/#206
- **`.omx/state/council_deliberation_posterior.jsonl`** appended via canonical `tac.council_continual_learning.append_council_anchor` per Catalog #128/#131/#245 sister discipline
- **Lane registry** NOT mutated (this council's deliverable is decision output; lane mutations require sister-subagent landing per Decision 4 + Decision 1 + Decision 3 op-routables)

### Cross-references

**Empirical anchors**:
- Subagent A NSCS06 v8 harvest memo: `.omx/research/harvest_nscs06_v8_and_stc_v2_smokes_20260516.md` (commit `2207dc4ab`)
- Subagent B 4-reducer matrix: Wunderkind G1 v2 Appendix C + Tishby IB-pure Appendix B (commits `c1633d18b` + `9f1f618a7`)
- Subagent C K=13 schedule: `.omx/research/k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516.md`
- Subagent F earlier T3 council: `.omx/research/grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md` (commit `20972f076`)
- FALSIFICATION-AUDIT-v2: `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` (commit `c5e4953e6`)

**Design memos**:
- NSCS06 v8 Path B Section 18 decision tree: `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md`
- Wunderkind G1 v3 per-pair adaptive sigma: `.omx/research/wunderkind_g1_v3_per_pair_adaptive_sigma_full_stack_design_20260516.md`
- Tishby IB-pure substrate-native scoping: `.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md`
- T2 council Wunderkind v2 PIVOT validation: `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md`
- NSCS06 v6 falsification symposium: `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`

**Standing directives**:
- HORIZON-CLASS evaluation axis: `feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md`
- Hard-earned-vs-cargo-culted addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- Mission-alignment Catalog #300 extension: `feedback_mission_alignment_followon_catalog_300_extension_landed_20260516.md`
- Abandon within-class refinements: `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md`
- 9-dim success checklist: `feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md`

**CLAUDE.md non-negotiables anchoring this T3 batched council**:
- "Council conduct" — sextet pact with Assumption-Adversary seat per Catalog #292
- "Forbidden premature KILL" — structural anchor (Decision 3 v2 DEFER permanent NOT KILL; ≥3 reactivation criteria enumerated)
- "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" — Decision 4 Tier-A explicitly requires both axes
- "Apples-to-apples evidence discipline" — every score axis tagged
- "META-ASSUMPTION ADVERSARIAL REVIEW" — this council IS an instance per Catalog #291
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — per-decision UNIQUE adjudication
- "Race-mode rigor inversion + parallel-dispatch first" — Decision 4 Tier-A PARALLEL invokes this directly
- "HORIZON-CLASS standing directive" — every Decision tagged with horizon_class

**Catalog gates relevant**:
- #117/#157/#174 canonical commit serializer / #128/#131/#245 fcntl-locked JSONL / #206 subagent crash-resume / #229 premise-verification / #230 sister-subagent ownership / #240 recipe-vs-trainer-state / #270 dispatch-optimization-protocol / #287 docstring-overstatement / #290 canonical-vs-unique / #291 META-ASSUMPTION cadence / #292 per-deliberation assumption / #294 9-dim checklist / #296 Dykstra-feasibility / #300 mission-alignment v2 council hierarchy / #305 observability surface / #307 paradigm-vs-implementation / #308 probe-methodology

---

**END OF WAVE-2 BATCHED T3 GRAND COUNCIL DELIBERATION LEDGER**
