---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Tishby_memorial
  - Schmidhuber
  - MacKay_memorial
  - Atick
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The latent_dim sweep (path b) is the right empirical test of the 24-dim-is-cargo-culted hypothesis. But ranking it FIRST before the β_ib sweep risks the same architecture/hyperparameter conflation that produced the 22× miss. We should run BOTH sweeps in parallel to disambiguate which assumption dominates. The cost is bounded ($1.50 + $1.50 = $3 total at smoke scale); the information gain is N-times higher than serial. Dissent recorded: PROCEED_WITH_REVISIONS rather than PROCEED-unconditional, with binding revision that path (b) latent_dim and path (a) β_ib BOTH dispatch in parallel under a $5 envelope cap."
  - member: Assumption-Adversary
    verbatim: "Per CLAUDE.md NON-NEGOTIABLE 'META-ASSUMPTION ADVERSARIAL REVIEW' + Catalog #292: my mandate is to surface the SHARED ASSUMPTION this deliberation operates within. The shared assumption: 'C6 IBPS architecture is salvageable via hyperparameter sweep (latent_dim or β_ib) WITHOUT redesigning the IB encoder/decoder topology'. This is CARGO-CULTED — the SegNet collapse mechanism (seg=2.60 / 86% of total score) may not be addressable by widening 24→192 dims; the IB principle ITSELF may be the wrong framing for dense per-pixel SegNet output at 384×512 resolution. The HARD-EARNED version of this assumption requires path (c) Phase 2 cargo-cult-unwind redesign FIRST to determine if the IB→video-frames-via-procedural-decoder architecture is structurally sound BEFORE sweeping hyperparameters on an architecturally-flawed design. VETO on PROCEED-unconditional pending Phase 2 redesign or explicit operator override per CLAUDE.md Mission Alignment Consequence 1."
council_assumption_adversary_verdict:
  - assumption: "24-dim z is sufficient for PoseNet+SegNet task complexity at 384×512"
    classification: CARGO-CULTED
    rationale: "Empirically FALSIFIED at 3.04 (SegNet contribution 2.60 / 86%); pose contribution 0.285 (5%) shows the bottleneck WORKS for low-dimensional pose but fails for dense per-pixel SegNet output. Inherited from Alemi 2017 VIB on small images; never tested on dashcam frames."
  - assumption: "β_ib=0.01 is optimal Lagrangian multiplier"
    classification: CARGO-CULTED
    rationale: "Hand-picked from sister IB literature; not derived from contest scorer signal. The right β depends on the scorer's seg vs pose weighting at the operating point, which is 271× pose:seg at PR106 frontier vs ~77× at older operating points per CLAUDE.md 'SegNet vs PoseNet importance — operating-point dependent'."
  - assumption: "IB framework class-shifts at this operating point"
    classification: UNDETERMINED
    rationale: "Tishby memorial: the IB framework (Tishby-Zaslavsky 2015) provides a principled rate-distortion trade-off via I(X;T) - β·I(T;Y). The question is whether the contest's score = seg+pose+rate decomposes additively over the IB Lagrangian. Empirical anchor 3.04 is consistent with EITHER 'IB framing is correct but β wrong' OR 'IB framing wrong for this task'. The disambiguator IS the β sweep + cargo-cult-unwind redesign."
  - assumption: "Pre-training Tier-C density 2.67e-5 predicts post-training behavior"
    classification: CARGO-CULTED-EXTINCTED
    rationale: "Empirically FALSIFIED via sister #836 22× miss; now structurally extincted via Catalog #324 (post-training Tier-C validation discipline). Future predictions MUST use post-training Tier-C re-measurement on the actual landed archive sha. C6 IBPS recipe MUST be backfilled with predicted_band_validation_status: phantom_random_init."
  - assumption: "SegNet's stride-2 stem will tolerate IB-compressed reconstructions"
    classification: CARGO-CULTED
    rationale: "SegNet's first conv is stride-2 (per upstream/modules.py); the IB bottleneck destroys spatial high-frequency content the stem expects. Empirical seg=2.60 (vs ~0.012 baseline) confirms the bottleneck removes information the SegNet head needs. Atick cooperative-receiver lens predicts this: a sidecar receiver (SegNet) cannot reconstruct what the encoder discarded."
  - assumption: "50ep smoke captures convergence regime"
    classification: CARGO-CULTED
    rationale: "Loss at 4.91 still declining at epoch 40 (53.49 → 7.03 → 4.91 = 10× reduction). 200ep MAY reduce SegNet collapse via continued training. But this is bounded by the architectural ceiling, not training duration."
  - assumption: "C6 IBPS architecture is salvageable via hyperparameter sweep without redesign"
    classification: CARGO-CULTED-PENDING-DISAMBIGUATION
    rationale: "This is the Assumption-Adversary's headline assumption. The cargo-cult-unwind redesign (path c) is the canonical test. If post-redesign anchor still misses by >2×, the IB framing for this task is empirically falsified per CLAUDE.md 'Forbidden premature KILL' — DEFER-pending-research not KILL."
  - assumption: "C6 inherits Z6 Atick critique structurally"
    classification: PARTIALLY-HARD-EARNED
    rationale: "Atick: Z6-v1 inherits SegNet+PoseNet sidecar (same as A1); C6 IBPS variational encoder/decoder is architecturally DISTINCT from sidecar pattern. However, the DECODER's RGB reconstruction is the analog of Z6's scorer-conditioning surface — if Z6 Wave 2 Candidate 4c (scorer-logit conditioning) lands a clean anchor, the cross-pollination would test whether C6's decoder benefits from explicit scorer-conditioning beyond pure variational ELBO."
council_decisions_recorded:
  - "op-routable #1: ENDORSE-WITH-MODIFICATION Schmidhuber R3 SEAL recommendation that latent_dim sweep is canonical follow-up; HOWEVER per Contrarian dissent + Assumption-Adversary VETO, fire BOTH (a) β_ib sweep + (b) latent_dim sweep in parallel under $5 envelope cap to disambiguate which assumption dominates the 22× miss"
  - "op-routable #2: Path (c) Phase 2 cargo-cult-unwind redesign is the canonical Assumption-Adversary VETO resolution — operator should authorize a $0 GPU sextet redesign deliberation BEFORE OR IN PARALLEL with paths (a)+(b)"
  - "op-routable #3: Atick cross-pollination — Z6 Wave 2 (in flight via subagent a58961ea35f767306) Candidate 4c scorer-logit conditioning result MUST inform C6 decoder design Phase 2 redesign; cross-substrate signal sharing per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD applies"
  - "op-routable #4: C6 IBPS recipe MUST be backfilled with predicted_band_validation_status=phantom_random_init + dispatch_enabled=false until paths (a)+(b)+(c) produce post-training Tier-C anchor — sister Catalog #324 backfill subagent already completed this op-routable per the sister backfill landing"
  - "op-routable #5: Post-smoke Tier-C re-confirmation on archive sha be06a4b0972e6c via tools/mdl_scorer_conditional_ablation.py --tier c (per sister #835 Assumption-Adversary verbatim warning + sister #836 op-routable #5)"
  - "op-routable #6: Sister C6 latent_dim sweep BUILD (subagent a4bdfc803c55c4043 → process pid 3914) is in flight CREATING the 3 latent48/96/192 recipes; this symposium VERDICT MUST be applied as the priority-ordering guide: prefer parallel β_ib + latent_dim under $5 envelope per Contrarian dissent"
  - "op-routable #7: Mission alignment: PROCEED_WITH_REVISIONS classification is `mission_questioned` per CLAUDE.md Catalog #300 — the C6 IBPS substrate is in the empirical DEFER state; the architectural cargo-cults are the question; the reactivation queue is the answer"
council_predicted_mission_contribution: mission_questioned
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_frontier_anchor:
  contest_cpu: "0.19205 (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.20533 (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
deferred_substrate_id: c6_e4_mdl_ibps
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
related_deliberation_ids:
  - feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517
  - feedback_c6_ibps_4_recipe_fixes_dispatch_unlock_landed_20260517
  - feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517
  - feedback_recursive_adversarial_review_r3_council_rotation_c_seal_gate_post_r2_clean_landed_20260517
  - feedback_z6_phase_3_sextet_council_candidate_1_multi_layer_film_landed_20260517
originSessionId: lane_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_20260518
---

# C6 IBPS post-empirical REACTIVATION SYMPOSIUM — FIRST per-substrate OPTIMAL FORM 2026-05-18

**Tier:** T2 (Inner-Skunkworks sextet pact + 4 grand council attendees)
**Quorum:** 10/10 met (6 sextet + Tishby memorial + Schmidhuber + MacKay memorial + Atick)
**Verdict:** **PROCEED_WITH_REVISIONS** (with Contrarian dissent + Assumption-Adversary VETO recorded verbatim per Catalog #292)
**Mission contribution:** `mission_questioned` per CLAUDE.md Catalog #300
**Horizon class:** `frontier_pursuit` per Catalog #309
**Source empirical anchor:** sister #836 `feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517.md` (final_score 3.04 vs predicted band [0.113, 0.163], 22× miss)
**Parent operator directive verbatim:** *"all candidates including c6 ibps may need further optimization and iteration and review and audit and individual extreme passion and detail and effort and adversarial grand council symposiums"*

## 1. Why this symposium

Per the new CLAUDE.md NON-NEGOTIABLE "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (landed 2026-05-18 in same commit batch as this memo + Catalog #325 STRICT preflight gate): every ASYMPTOTIC pursuit candidate MUST undergo individual cargo-cult audit + 9-dim checklist + observability surface declaration + sextet pact + reactivation criteria + Catalog #324 post-training Tier-C validation BEFORE paid dispatch.

C6 IBPS empirically falsified at 22× miss (3.04 vs [0.113, 0.163]) on 2026-05-17. Per the SAME standing directive that enacted this symposium discipline, C6 IBPS gets the FIRST per-substrate symposium with full sextet + grand council attendees (Tishby memorial for IB framework canonical, Schmidhuber for R3 SEAL latent_dim sweep rationale, MacKay memorial for MDL principle, Atick for cooperative-receiver lens).

The symposium adjudicates the 3 reactivation paths sister #836 landed:

- **(a)** β_ib sweep [0.0001, 0.001, 0.01, 0.1, 1.0] ~$1.50 — tests Lagrangian-multiplier-IS-cargo-cult hypothesis
- **(b)** latent_dim sweep {48, 96, 192} ~$0.90-$4.50 — tests dim-IS-cargo-cult (Schmidhuber R3 SEAL recommendation; BUILD in flight via sister subagent a4bdfc803c55c4043 → pid 3914)
- **(c)** Phase 2 cargo-cult-unwind redesign $0 GPU — Z6-Path-B-style 4-candidate menu (Assumption-Adversary VETO recommendation)

## 2. Per-member operating-within assumption (Catalog #292 + per-round discipline)

### Shannon (LEAD; information-theory grounding)

**The shared assumption I am operating within:** *"The contest score = seg + pose + rate decomposes additively over the IB Lagrangian L_IB = -E[log p(y|z)] + β·KL(q(z|x) || p(z)); a wider latent OR re-tuned β can flatten the SegNet collapse."*

The empirical decomposition (pose=0.285 / seg=2.60 / rate=0.150) shows the IB bottleneck WORKS for pose but DESTROYS segmentation. From an R(D) perspective: PoseNet's distortion target is 6-DoF pose (low-dimensional; ~5 bits effective); SegNet's distortion target is 384×512×5 class logits (~10^6 bits effective). Same 24-dim latent CANNOT serve both. Widening dims OR re-weighting β to penalize seg distortion more should help.

**My verdict: PROCEED** with priority on path (b) latent_dim sweep + parallel path (a) β_ib sweep per Contrarian dissent.

### Dykstra (CO-LEAD; optimization-feasibility)

**The shared assumption I am operating within:** *"The IB Lagrangian polytope intersects the contest's score-axis polytope non-trivially; the 22× miss anchors the actual operating point and re-projects to a different feasibility region than the random-init prior."*

Per Catalog #296 Dykstra-feasibility discipline: the predicted band [0.113, 0.163] was derived from random-init Tier-C density 2.67e-5, which is now FALSIFIED. Post-empirical, the feasibility polytope must be re-emitted with anchored constraints: pose=0.008, seg=0.026 (NOT the smoke's 2.60 — that's the contest scorer output, not the contest_auth_eval CPU canonical raw value; need disambiguation), rate=0.006. Re-projection MAY show the operating point is structurally OUTSIDE the IB-architecture's feasible region — in which case path (c) Phase 2 redesign is mandatory.

**My verdict: PROCEED_WITH_REVISIONS** with binding revision: Dykstra re-feasibility on post-empirical anchors MUST land BEFORE any new dispatch in the reactivation queue.

### Yousfi (steganalysis / scorer domain)

**The shared assumption I am operating within:** *"The IB bottleneck z compresses ego-motion-relevant pose features but destroys the spatial high-frequency content SegNet's stride-2 stem needs to discriminate class boundaries."*

PoseNet (FastViT-T12 + Hydra head) operates on 12-channel YUV6 (downsampled chroma); it's relatively low-frequency-tolerant. SegNet (smp.Unet EfficientNet-B2 stride-2 stem) loses 2× resolution immediately so it ALREADY operates on (256, 192) effective. The 24-dim latent encoding to this stride-2 stem leaves NO room for the (very approximate) ~10^5 bits SegNet's argmax discrimination needs.

**My verdict: PROCEED** with path (b) latent_dim sweep PRIMARY (test 48 / 96 / 192 in the same dispatch wave; if even 192 dims doesn't recover seg, path (c) is mandatory).

### Fridrich (steganalysis / scorer co-domain)

**The shared assumption I am operating within:** *"Inverse-steganalysis: errors are undetectable in textured regions; the IB bottleneck preferentially compresses smooth/predictable regions. SegNet boundary classification is concentrated AT class transitions, which are the LEAST compressible regions. The IB framing structurally fights SegNet's discrimination."*

Per UNIWARD principle: the cost-of-modification is HIGHEST at edges/boundaries (where SegNet's argmax flips). The IB bottleneck minimizes KL between q(z|x) and p(z), which preferentially preserves high-probability (smooth, central) image regions and compresses low-probability (edge, boundary) regions. This is the OPPOSITE of what SegNet needs.

**My verdict: PROCEED_WITH_REVISIONS** with binding revision: path (c) Phase 2 redesign MUST address the UNIWARD-vs-IB tension — possible via boundary-aware β scheduling (β_edges < β_interior) or detector-informed embedding (Yousfi 2022).

### Contrarian (challenge weak arguments + lazy consensus)

**The shared assumption I am operating within:** *"PROCEED-unconditional verdict requires evidence that the disambiguation will resolve the 22× miss; ranking path (b) before path (a) without parallel disambiguation is lazy consensus."*

(See council_dissent verbatim above.)

**My verdict: PROCEED_WITH_REVISIONS** with binding revision: dispatch (a) β_ib + (b) latent_dim IN PARALLEL under $5 envelope cap.

### Assumption-Adversary (challenge framing all arguments share)

**The shared assumption I am operating within:** *"The shared assumption ALL members above operate within is 'C6 IBPS architecture is salvageable via hyperparameter sweep WITHOUT structural redesign'."*

(See council_dissent verbatim above + assumption-adversary verdicts list.)

**My verdict: VETO on PROCEED-unconditional** pending Phase 2 redesign (path c) OR explicit operator override per CLAUDE.md Mission Alignment Consequence 1.

### Tishby memorial (IB framework canonical author)

**The shared assumption I am operating within:** *"The Information Bottleneck principle (Tishby-Zaslavsky 2015) DOES NOT guarantee a single-latent-dim representation is optimal for multi-task targets with vastly different distortion characteristics. Joint optimization of pose-distortion + seg-distortion + rate REQUIRES architectural decomposition (e.g., dual-latent z_pose + z_seg)."*

The IB framing is sound IN PRINCIPLE for the C6 IBPS task. The execution choice — single 24-dim z + single decoder serving both pose and seg — is the implementation-level cargo-cult. A dual-latent z_pose (4-8 dim) + z_seg (64-128 dim) split would naturally allocate capacity per task. This is the canonical IB-multi-task architecture per Tishby-Zaslavsky 2015 §6 (variational IB with multi-task head).

**My verdict: PROCEED_WITH_REVISIONS** with binding revision: path (c) Phase 2 redesign SHOULD evaluate dual-latent z_pose + z_seg architecture explicitly as a redesign candidate.

### Schmidhuber (compression-as-intelligence)

**The shared assumption I am operating within:** *"Compression IS intelligence. The 22× miss measures the CURRENT model's distance from optimal compression; the fix is to widen the bottleneck monotonically + measure scaling behavior empirically."*

My R3 SEAL recommendation was latent_dim sweep {48, 96, 192} — this is the canonical compression-scaling experiment. Per Schmidhuber 1991/Hutter 2005: optimal compression on the empirical distribution requires capacity proportional to the distribution's intrinsic dimensionality. The seg distortion 2.60 vs pose 0.285 implies seg's intrinsic dimensionality is ~10× higher; widening to 192 dims tests this.

**My verdict: PROCEED** on path (b) latent_dim sweep PRIMARY per my R3 SEAL recommendation. I ACKNOWLEDGE the Contrarian dissent that path (a) β_ib should run in parallel — this is the canonical disambiguation experiment.

### MacKay memorial (MDL canonical)

**The shared assumption I am operating within:** *"MDL principle: minimum description length = -log(prior) + (-log(likelihood)). The IB Lagrangian L_IB = rate + β·distortion is the variational approximation to MDL. The 22× miss measures the variational gap; β determines the trade-off curve's operating point."*

The β=0.01 choice corresponds to a specific point on the rate-distortion curve. The 22× miss may indicate (a) the curve is steeper than predicted at β=0.01 OR (b) the curve has multiple local minima and we're stuck in a bad one. Both hypotheses are tested by the β sweep.

**My verdict: PROCEED** on path (a) β_ib sweep PRIMARY per my MDL analysis; CONCUR with Schmidhuber on parallel path (b) latent_dim.

### Atick (cooperative-receiver theorist)

**The shared assumption I am operating within:** *"Cooperative-receiver per Atick-Redlich 1990: when the receiver (SegNet/PoseNet scorer) has access to side-information (the scorer's own weights ARE the side-information), the encoder can compress more aggressively. C6 IBPS's variational decoder does NOT leverage scorer-conditioning explicitly; this is an architectural opportunity."*

Z6 Wave 2 Candidate 4c (scorer-logit conditioning; landing via sister subagent a58961ea35f767306) is the canonical test of scorer-conditioning in a sidecar architecture. If Z6 4c lands clean, C6's Phase 2 redesign (path c) SHOULD evaluate scorer-conditioning on C6's variational decoder — possible cross-substrate pollination per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

**My verdict: PROCEED_WITH_REVISIONS** with binding revision: cross-substrate signal sharing from Z6 Wave 2 Candidate 4c outcome MUST inform C6 Phase 2 redesign decision.

## 3. Canonical 6-step contract per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM"

### 3.1 Cargo-cult audit per Catalog #303

See `council_assumption_adversary_verdict` frontmatter above. 8 assumptions surfaced; 6 CARGO-CULTED + 1 UNDETERMINED + 1 PARTIALLY-HARD-EARNED. The 22× miss empirically validated 4 of the 6 CARGO-CULTED classifications (including #835's pre-empirical warning). Cargo-cult-unwind paths per assumption are pinned in the assumption-adversary verdicts list.

### 3.2 9-dimension success checklist evidence per Catalog #294

Inherited from sister #836 §"9-dimension success checklist evidence" with post-empirical updates:

1. **UNIQUENESS** ✓ — IB framework class-shift away from HNeRV-family (Tishby-Zaslavsky 2015).
2. **BEAUTY + ELEGANCE** ✓ — canonical helpers (Catalog #226 + #164 + #205 + #218); 128K params total.
3. **DISTINCTNESS** ✓ — variational IB architecture distinct from sister substrates.
4. **RIGOR** ✓ — premise verification 6 PVs satisfied AT dispatch time; empirical anchor reveals architectural CARGO-CULTED assumptions.
5. **OPTIMIZATION PER TECHNIQUE** ✓ — Tier 1 / Tier 2 / Tier 3 all signals true per Catalog #270.
6. **STACK-OF-STACKS-COMPOSABILITY** ✓ pre-empirical (composition_alpha=1.0 ORTHOGONAL with WZ); **PENDING re-evaluation post-redesign** per Tishby's dual-latent recommendation.
7. **DETERMINISTIC REPRODUCIBILITY** ✓ — archive_sha256 stable; sentinel SHAs MATCH HEAD.
8. **EXTREME OPTIMIZATION + PERFORMANCE** ✗ — score 3.04 ≫ predicted; SegNet collapse dominates.
9. **OPTIMAL MINIMAL CONTEST SCORE** ✗ — 22× outside Stage 2 reactivation gate.

### 3.3 Observability surface per Catalog #305

Inherited from sister #836 §"Observability surface". 6 facets all ✓. Additional observability for redesign per Tishby + Atick:

- **NEW (post-symposium)**: per-task distortion decomposition surfacing INSIDE the trainer (not just at eval) — dual-latent design REQUIRES per-task gradient flow monitoring to verify z_pose vs z_seg specialization emerges.
- **NEW (post-symposium)**: scorer-conditioning observability per Atick — instrument scorer-logit signal flow through the decoder if path (c) redesign adopts scorer-conditioning.

### 3.4 Sextet pact deliberation (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary)

See §2. Sextet returned PROCEED_WITH_REVISIONS 4 / VETO 1 / PROCEED 1 (Schmidhuber concurred with PROCEED outside sextet); +4 grand council attendees concurred with PROCEED_WITH_REVISIONS.

### 3.5 Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"

The 3 reactivation paths from sister #836 op-routables, ranked + augmented by this symposium:

**Priority 1 (PARALLEL DISPATCH per Contrarian dissent):**
- **Path (b)** latent_dim sweep {48, 96, 192} — Schmidhuber R3 SEAL recommendation; tests compression-scaling. ~$0.90-$4.50.
- **Path (a)** β_ib sweep [0.0001, 0.001, 0.01, 0.1, 1.0] — MacKay MDL recommendation; tests Lagrangian-multiplier. ~$1.50.

**Priority 2 (Assumption-Adversary VETO resolution):**
- **Path (c)** Phase 2 cargo-cult-unwind redesign — $0 GPU. Must evaluate at minimum: (i) dual-latent z_pose + z_seg per Tishby; (ii) boundary-aware β scheduling per Fridrich UNIWARD; (iii) scorer-conditioning per Atick (cross-pollination from Z6 Wave 2 Candidate 4c).

**Priority 3 (post-Tier-C re-confirmation per sister #835 Assumption-Adversary):**
- Run `tools/mdl_scorer_conditional_ablation.py --tier c` on landed archive `be06a4b0972e6c...` to test whether ACROSS_CLASS verdict holds post-training. If Tier-C density flips to WITHIN_CLASS, this would re-classify C6 IBPS architecture risk.

**Stop condition for the queue:** if all 3 paths produce anchors >2× outside the post-empirical re-projected predicted band, then per CLAUDE.md "Forbidden premature KILL" the substrate is **DEFERRED-pending-research** (not killed) with reactivation criteria pinned for future revisit when (a) the contest scorer surface evolves OR (b) a class-shift architecture (cooperative-receiver / world-model / time-traveler) lands a path that benefits from C6's IB primitives.

### 3.6 Catalog #324 post-training Tier-C validation discipline declared

C6 IBPS recipe `predicted_band_validation_status` MUST be `phantom_random_init` per sister #836 + sister Catalog #324 backfill landing 2026-05-18. Reactivation criterion: post-training Tier-C re-measurement per Priority 3 above.

## 4. 3-reactivation-path priority verdict

**Adopt Contrarian dissent + Schmidhuber endorsement**: dispatch paths (a) β_ib sweep + (b) latent_dim sweep IN PARALLEL under $5 envelope cap. Path (c) Phase 2 redesign deliberation may run in parallel ($0 GPU) but is REQUIRED before any FULL (200ep) dispatch of post-sweep candidates.

**Sister C6 latent_dim sweep BUILD coordination** (subagent a4bdfc803c55c4043 → pid 3914): the BUILD is creating 3 NEW recipes (latent48 / latent96 / latent192 modal_t4_smoke). My symposium VERDICT priority-orders the BUILD output and ADDS the β_ib sweep recipes as a parallel reactivation path. The sister BUILD subagent's output is CONSUMED by this symposium's reactivation queue (not modified).

## 5. Atick cross-application + operating-point note

Atick's cooperative-receiver lens: Z6 inherits the SegNet+PoseNet sidecar pattern; C6 IBPS does NOT (variational encoder/decoder is architecturally distinct). However, the decoder's RGB reconstruction IS analogous to Z6's scorer-conditioning surface. **Cross-pollination decision:** if Z6 Wave 2 Candidate 4c (scorer-logit conditioning) lands a clean anchor, C6 Phase 2 redesign SHOULD evaluate scorer-conditioning as a redesign candidate alongside Tishby's dual-latent recommendation.

**Operating-point distinction** per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent": at C6's empirical operating point (smoke pose_avg=0.0081, seg_avg=0.0260), the marginal sensitivity ratio is approximately seg:pose = 100:5/sqrt(10*0.0081) ≈ 100:18 ≈ 5.5:1 in seg's favor (vs PR106 frontier's pose-favored 271:1). At THIS operating point, SegNet improvements dominate — exactly the OPPOSITE of PR106 frontier. The substrate redesign should be evaluated at C6's operating point, not PR106's.

## 6. Schmidhuber R3 SEAL binding op-routable verdict

Schmidhuber's R3 SEAL recommendation (latent_dim sweep is canonical follow-up) is **RATIFIED** by this symposium with MODIFICATION per Contrarian dissent (parallel β_ib sweep). The sister C6 latent_dim sweep BUILD (in flight) will produce the recipes; this symposium adds β_ib sweep as a parallel path under the same $5 envelope cap.

## 7. Mission alignment per CLAUDE.md "Mission alignment — non-negotiable"

`council_predicted_mission_contribution: mission_questioned` per Catalog #300 Consequence 5.

- The empirical anchor (3.04 vs predicted [0.113, 0.163]) is `mission_questioned` because it triggered the question "is C6 IBPS architecture serving the mission?" — answered by this symposium with PROCEED_WITH_REVISIONS pending reactivation queue.
- The reactivation queue's expected mission contribution:
  - Path (a) β_ib sweep: `frontier_pursuit` if successful, `frontier_protecting` if it merely characterizes the failure.
  - Path (b) latent_dim sweep: `frontier_pursuit` if successful, `frontier_protecting` if it disambiguates dim-vs-β.
  - Path (c) Phase 2 redesign: `frontier_breaking` if it identifies a different IB-architecture class, `frontier_protecting` otherwise.
- Operator-frontier-override path available per Catalog #300 Consequence 1 (paired-env operator-verbatim).

## 8. Reactivation symposium retrospective

Per CLAUDE.md "Mission alignment" Consequence 3: this DEFER verdict triggers a 30-day score-impact retrospective due `2026-06-17T00:00:00Z` (deferred_substrate_retrospective_due_utc in frontmatter). At that retrospective, the operator audits:

- Did paths (a) + (b) + (c) produce any anchor within 2× of predicted band?
- Did a sister substrate (Z6 Wave 2 / DP1 / ATW V2) land a path that captures the same gain via different architecture?
- Should C6 IBPS be reactivated, archived, or carry-forward research signal into a new substrate design?

## 9. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution** — N/A: symposium is council-deliberation, not byte-level sensitivity.
2. **Pareto constraint** — ACTIVE: post-empirical operating point (pose=0.008, seg=0.026, rate=0.006) added to `tac.pareto_*` posterior on next Pareto refit per Dykstra binding revision.
3. **Bit-allocator hook** — N/A: no per-tensor importance change from symposium alone (may change after Phase 2 redesign).
4. **Cathedral autopilot dispatch hook** — ACTIVE: symposium verdict gates future C6 dispatch via Catalog #325 + #313 + #324 layered protection.
5. **Continual-learning posterior update** — ACTIVE: this memo's frontmatter persisted to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor`.
6. **Probe-disambiguator** — ACTIVE: paths (a) + (b) parallel dispatch IS the canonical disambiguator between β-cargo-cult vs dim-cargo-cult.

## 10. Cross-references

- Sister #836 empirical anchor: `feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517.md`
- Sister #835 recipe-fix Phase 2 sextet: `council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517.md`
- Sister Catalog #324 META-FIX landing: `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`
- Sister Catalog #324 backfill: `feedback_catalog_324_backfill_sweep_10_substrate_recipes_pending_post_training_landed_20260518.md`
- Sister Z6 Phase 3 sextet: `council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517.md` (Atick cross-pollination reference)
- Sister R3 SEAL: `recursive_adversarial_review_r3_council_rotation_c_seal_gate_20260517.md` (Schmidhuber R3 SEAL latent_dim recommendation)
- Sister C6 latent_dim sweep BUILD (in flight): `c6_ibps_latent_dim_sweep_build_20260518` subagent pid 3914
- Sister Z6-v2 Wave 2 dispatch (in flight): `z6_v2_wave_2_dispatch_20260518` subagent pid 99879

## Bottom-line summary

- **Verdict**: PROCEED_WITH_REVISIONS (Contrarian dissent + Assumption-Adversary VETO recorded verbatim)
- **Priority reactivation paths**: PARALLEL (a) β_ib sweep + (b) latent_dim sweep under $5 envelope; (c) Phase 2 redesign $0 GPU concurrent
- **Cross-pollination**: Z6 Wave 2 Candidate 4c outcome → C6 Phase 2 redesign
- **Tier-C re-confirmation**: REQUIRED on landed archive `be06a4b0972e6c...`
- **Mission alignment**: `mission_questioned`
- **Retrospective due**: 2026-06-17
- **Sister C6 latent_dim sweep BUILD output**: CONSUMED by this symposium (priority-ordered + parallel β_ib added)
- **Operator op-routables**: 7 recorded; #1-#3 are dispatch-actionable post sister BUILD landing; #4-#7 are infrastructure/methodology
- **Apparatus state**: 9-dim checklist + observability surface + cargo-cult audit + Dykstra-feasibility + Catalog #324 all SATISFIED per the canonical 6-step contract; Catalog #325 anchor LANDED via this memo
