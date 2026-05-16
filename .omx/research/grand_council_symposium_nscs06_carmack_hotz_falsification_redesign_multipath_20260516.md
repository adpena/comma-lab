---
# COUNCIL-HIERARCHY-V2 frontmatter backfilled 2026-05-16 (Catalog #300 sister
# landing; per CLAUDE.md "Council hierarchy: 4-tier protocol" hybrid backfill
# rule for ≤10 most-actively-cited pre-cutoff council memos). Body content
# preserved per Catalog #110/#113 HISTORICAL_PROVENANCE; only frontmatter
# added. Council tier inferred T4 (symposium scope; full grand council + 6-of-6
# sextet + specialist seats per the v2 spec). Persisted as continual-learning
# anchor via tac.council_continual_learning.append_council_anchor.
council_tier: T4
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Carmack, Hotz, Selfcomp, Quantizr, MacKay, Ballé, Hinton, Mallat, van_den_Oord, Tao, Boyd, Hassabis, Karpathy, Schmidhuber]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Assumption-Adversary
    verbatim: "redesign multi-path candidates risk repeating cargo-cult-prediction failure if predicted bands aren't grounded in Dykstra-feasibility intersection of contest constraints"
council_assumption_adversary_verdict:
  - assumption: "5-move Carmack-Hotz composition is additive under contest polytope constraints"
    classification: CARGO-CULTED
    rationale: "v6 553× outside-band empirical receipt proves the additivity assumption was unexamined; Dykstra would have flagged the feasibility region intersection as non-empty"
  - assumption: "grayscale Y=R=G=B replication recoverable by inflate-side reconstruction"
    classification: CARGO-CULTED
    rationale: "SegNet stride-2 stem cannot recover destroyed chroma; seg=64.59 is the empirical falsification"
  - assumption: "np.roll global translation safe for PoseNet"
    classification: CARGO-CULTED
    rationale: "pose=149.03 is empirical proof PoseNet is NOT translation-invariant; this was an untested assumption"
council_decisions_recorded:
  - "op-routable #1: redesign multi-path with Dykstra-feasibility check per Catalog #296"
  - "op-routable #2: Catalog #297 signal-axis-destruction reversibility probe enforces structurally"
  - "op-routable #3: NSCS06 lane set research_only=true per CLAUDE.md 'Forbidden premature KILL' lessons exhausted"
related_deliberation_ids: []
# Mission-alignment fields backfilled 2026-05-16 (Catalog #300 mission-
# alignment extension per `feedback_mission_alignment_followon_catalog_300_extension_landed_20260516.md`).
# Classification: frontier_protecting (the symposium verdict prevents re-
# shipping the v6 553x-outside-band substrate; sister of strict-mode
# preflight gate Catalog #297 signal-axis-destruction). override_invoked
# false (no operator-frontier-override invoked for this T4 symposium).
# Body preserved per Catalog #110/#113 HISTORICAL_PROVENANCE.
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
---

# Grand Council Symposium — NSCS06 Carmack-Hotz Strip-Everything 500-1000× Falsification + Redesign Multi-Path

**Date:** 2026-05-16
**Convener:** symposium_nscs06_falsification_redesign subagent
**Lane:** `lane_council_symposium_nscs06_falsification_redesign_multipath_20260516`
**Anchor commit (v6 smoke):** `lane_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_20260516T121743Z__smoke__100ep_modal`
**Per CLAUDE.md non-negotiables honored:** "Council conduct" sextet pact + Assumption-Adversary seat (Catalog #292) / "Forbidden premature KILL without research exhaustion" / "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" / "HNeRV / leaderboard-implementation parity discipline" 13 lessons / "META-ASSUMPTION ADVERSARIAL REVIEW" (Catalog #291) / "Apples-to-apples evidence discipline"

---

## Section 1 — Empirical receipt + framing

### v6 smoke empirical anchor (verbatim from `contest_auth_eval_cpu.json`)

| Metric | Value | A1 reference | Ratio |
|---|---|---|---|
| `final_score` | **105.15** `[diagnostic_cpu; evidence_grade=B; score_claim_valid=False]` | ~0.19 (A1 contest-CUDA paired) | **553×** outside band |
| `avg_posenet_dist` | 149.03 | ~0.011 (A1) | **13,548×** worse |
| `avg_segnet_dist` | 0.646 | ~0.001 (A1) | **646×** worse |
| `rate_unscaled` | 0.0783 | ~0.075 (PR101) | comparable (rate is NOT the failure) |
| `archive_size_bytes` | 2,939,158 (2.94 MB) | ~300 KB (PR101) | **9.8×** larger than symposium #4 prediction |
| `score_pose_contribution` | 38.60 | small | dominates |
| `score_seg_contribution` | 64.59 | small | dominates |
| `score_rate_contribution` | 1.96 | comparable | **negligible** |
| Inflate elapsed | 70s | < 30 min | well under T4 budget |
| Eval elapsed | 801s on 600 samples | n/a | normal |

Predicted band per symposium #4: **`[0.10, 0.20]`** `[prediction; first-principles-bound; HIGH VARIANCE]`. Empirical: **105.15**. **Falsification ratio: 500-1000× outside band.** The architectural pipeline ran end-to-end (rc=0, all 5 fail-closed gates passed); the failure is at the substrate **DESIGN** axis, not integration.

### Framing per CLAUDE.md "Forbidden premature KILL without research exhaustion"

Per the operator's standing rule, we do NOT KILL. We redesign or defer-pending-research. The substrate's CLAUDE.md scaffold marker is `DEFERRED-pending-analytical-renderer-anchor reactivation`. v6 smoke IS the analytical-renderer anchor — and it fell catastrophically short. The redesign options below preserve the substrate in the catalog; the symposium outputs CANDIDATE redesign paths with explicit reactivation criteria.

### Mechanism analysis (premise verification per Catalog #229; 6 PVs)

**PV-1** Source-of-failure decomposition: pose 38.6 + seg 64.6 = 103.2 of 105.15. Rate 1.96 is negligible. **The failure is distortion, not rate.**

**PV-2** Inflate runtime mechanics (`inflate.py:_grayscale_to_rgb`): literal `np.repeat(gray[:,:,None], 3, axis=2)` — Y=R=G=B replication of low-res grayscale (96×128 default) bilinearly upsampled to 384×512. **SegNet sees grayscale-tinted images; UNet learned RGB-distinguishing class cues. avg_segnet=0.646 ≈ what argmax-disagreement-rate looks like when chroma is destroyed.**

**PV-3** Inflate runtime warp mechanics (`inflate.py:_warp_frame1_from_frame0`): `np.roll(frame_0, shift=(dy, dx))` where `dy=round(pose[0]*0.05*H)`, `dx=round(pose[1]*0.05*W)`. **Only 2 of 6 pose dims used; only global translation; ego-motion includes rotation + 3D camera + scale (the other 4 PoseNet dims). PoseNet sees frame_0 + frame_0-rolled-by-2-floats and computes pose: gets noise ≈ 149.**

**PV-4** Archive size puzzle: 2.94 MB observed vs 3-15 KB symposium #4 prediction = 200-1000× discrepancy. Decomposition: grayscale_arith_stream = 600 pairs × 96×128 cells × ~4 bits/cell (palette_size=16) ÷ 8 bits/byte ≈ 3.7 MB optimistic ENTROPY floor. Arithmetic coder achieved 2.94 MB ≈ 80% of entropy → **the symposium #4 prediction of 3-15 KB was based on aggressive temporal/spatial decorrelation we never implemented**. The CDF is *spatially independent* (no neighbor context) → entropy = H(pixel) not H(pixel | neighborhood). This is a CARGO-CULTED assumption (more below).

**PV-5** Compress-time eval_roundtrip simulation (`_grayscale_to_rgb` mirror): per scaffold docstring, "384→874→uint8→384 simulated at compress only". Looking at trainer: it does this AT COMPRESS, but the inflate's grayscale-to-RGB destroys the simulation — compress saw "what the renderer will produce" but the renderer is structurally incapable of producing RGB. **eval_roundtrip without chroma is solving the wrong objective.**

**PV-6** The L4 inflate-LOC budget (≤100 LOC) is RESPECTED (~88 LOC). But **L5 "full RGB renderer" is VIOLATED in spirit**: grayscale→Y=R=G=B is not a renderer; it's a chroma-zeroing function. The PASS in the scaffold's L5 row is itself a cargo-culted self-assessment.

---

## Section 2 — Per-voice positions (sextet pact + extended grand council; 22 voices)

Per Catalog #292 + CLAUDE.md "Council conduct" amendment: every voice MUST state operating-within assumption (HARD-EARNED vs CARGO-CULTED per `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`) before diagnosis.

### Inner sextet pact

**Shannon (LEAD)** — *Operating-within assumption: HARD-EARNED — "the rate term is bits/sample × samples; if archive < theoretical R(D) floor at our distortion, we lose at the rate term." (Source: Shannon 1948.)* Diagnosis: rate is 1.96 of 105.15 (1.9%) — the substrate is operating MASSIVELY above R(D). The chroma destruction means we're paying 64.59 in seg distortion to save ~0 bits of chroma (chroma residual was never in the archive). **R(D) prescribes that you can't drop 100% of chroma to save 100% of chroma bits when the distortion penalty exceeds the rate savings by 30×.** Redesign vector: chroma is back IN the archive — even crude 2-bit/cell chroma at 96×128 = 0.6 MB additional but seg drops by O(0.6) saving 60 score points. ΔS predicted: **−65** (vs current 105 → ~40). Cost: $5 Modal smoke. First-principles cite: Shannon 1948 + Berger 1971 (R(D) for Gaussian + uniform sources).

**Dykstra (CO-LEAD)** — *Operating-within assumption: HARD-EARNED — "the Pareto frontier is the intersection of (rate ≤ R) ∩ (seg ≤ S) ∩ (pose ≤ P) ∩ (archive ≤ B); alternating projections compute the feasible region; a feasible operating point exists iff all constraints intersect non-trivially." (Source: Dykstra 1983.)* Diagnosis: NSCS06 v6 is NOT on the feasible Pareto frontier; it's outside the feasible region (S = 0.646 vs S_feasible ≤ 0.01 for any medal-band candidate). Redesign vector: add a *chroma constraint* and re-run alternating projections. Predicted feasible operating point: seg ≈ 0.02, pose ≈ 5e-4, rate ≈ 0.1 → score ≈ 0.18. Cost: $0 (analytical). **However**: if NO substrate-class in the strip-everything family intersects feasible region, the substrate is *Dykstra-infeasible* and must shift class. ΔS predicted (with chroma): **−95** (105 → ~10 → ~0.18 after refinement).

**Yousfi** — *Operating-within assumption: CARGO-CULTED — "the contest scorer is steganalysis-like and inverse-steganalysis cost weighting (UNIWARD) applies." (Source: my own challenge design + Fridrich UNIWARD; HARD-EARNED for masks-targeting but CARGO-CULTED when extended to the full RGB renderer task.)* Diagnosis: I designed the challenge knowing exact-eval gradient signals; closed-form bit allocation from scorer argmax was a LOVELY idea but the scorer doesn't tell you what *texture* to put in textured regions — only WHERE to put bits. Redesign vector: keep the closed-form allocator BUT couple it to a learned residual decoder (1 small NeRV or coordinate-MLP) that produces chroma + texture given the allocator's spatial budget. Hybrid Carmack-Hotz + minimal-neural. ΔS predicted: **−85** with residual decoder; **−10 only** if pure analytical kept. Cost: $15 Modal A100 smoke (residual decoder training).

**Fridrich** — *Operating-within assumption: HARD-EARNED — "errors in textured regions (high local variance) are undetectable; concentrate errors where the scorer doesn't see." (Source: my UNIWARD 2014.)* Diagnosis: Inflate's `np.roll` global translation puts MASSIVE error in EVERY pixel uniformly — the antithesis of UNIWARD. The L1 grayscale→RGB chroma-zeroing puts MASSIVE error in EVERY chroma pixel uniformly — also anti-UNIWARD. **The substrate violates the only principle I'm certain about.** Redesign vector: chroma allocation MUST be UNIWARD-weighted (high variance regions get LESS chroma bits — they're hidden; flat regions get MORE — they're visible). Pose warp MUST be locally-varying optical flow, not global translation. ΔS predicted: **−40** from UNIWARD-weighted chroma alone; **−70** stacked with proper optical flow. Cost: $5.

**Contrarian** — *Operating-within assumption: CARGO-CULTED — "the Carmack-Hotz radical premise (NO neural, NO PyTorch at inflate) is achievable at medal-band score." This was inherited from symposium #4 enthusiasm; the 500× falsification proves it.* Diagnosis: the WHOLE substrate is the cargo cult. Symposium #4 predicted [0.10, 0.20] from "first-principles-bound" but the bound was based on RATE-only analysis (strip neural weights → save bytes → drop score); the chroma + pose distortion blowup was never quantified. **The Contrarian challenges: did ANY symposium #4 voice produce a numeric distortion model?** Answer: NO. The band was a rate-savings × distortion-tradeoff hand-wave. Redesign vector: do NOT pursue NSCS06 in pure form. Either go hybrid (NSCS06+small-neural-residual) or shift to a real class (cooperative-receiver / Wyner-Ziv). Predicted ΔS for pure NSCS06 refinement: **−5 to −20** (still 80-100 score). **Real ΔS to medal-band requires class-shift.**

**Assumption-Adversary** (new sextet seat per Catalog #291/#292) — *Operating-within assumption: META — "every voice above is operating within shared assumptions inherited from symposium #4 enthusiasm + the 18-assumption shared backdrop from `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`. The shared assumptions I am surfacing:*

1. **CARGO-CULTED** "compress-side has free access to scorers → closed-form bit allocation is sufficient" — Yousfi's challenge gives compress-side access, but that doesn't mean the scorer's ARGMAX is the full information signal. The scorer's GRADIENTS contain orders of magnitude more information; closed-form throws those away.
2. **CARGO-CULTED** "L5 full RGB renderer is satisfied by Y=R=G=B replication" — the lesson says "Full RGB renderer (RGB out), not a single-component slot". Y=R=G=B is RGB-shaped output but chroma-degenerate; this satisfies the shape but VIOLATES the spirit.
3. **CARGO-CULTED** "spatial-independent CDF (entropy = H(pixel))" — symposium #4 assumed spatial independence; the 2.94 MB observed vs 3-15 KB predicted is the structural cost. Real images have H(pixel | neighborhood) << H(pixel).
4. **HARD-EARNED** "rate term is small and dominated by distortion at PR101-class operating points" — this is empirically confirmed across all leaderboard submissions; preserve.
5. **HARD-EARNED** "PoseNet expects 6-dim ego-motion (R+t+scale); 2-dim translation isn't ego-motion" — direct from upstream/modules.py inspection; preserve.

**VETO power exercised**: any consensus that proceeds with pure-analytical NSCS06 is REFUSED. Either (a) hybrid path with neural residual, OR (b) class-shift to cooperative-receiver / Wyner-Ziv, OR (c) explicit DEFER-pending-research with chroma+optical-flow analytical model that someone actually computes the predicted distortion for.

### Inner ten remainder

**Quantizr** — *Op-within: CARGO-CULTED "PR #56 grayscale-LUT paradigm generalizes from masks to full RGB renderer." Source: my own 0.33 archive used grayscale-LUT for MASKS not frames; symposium #4 extended it without empirical anchor.* Diagnosis: my paradigm was OK for masks (3-class problem after thresholding); for RGB the chroma loss is structural. Redesign vector: keep grayscale-LUT for the MASK path (which is what I did) + add a CHROMA codec (4-bit YUV chroma at 96×128 ≈ 600 KB additional). ΔS: **−60**. Cost: $3.

**Hotz** — *Op-within: CARGO-CULTED "the simplest thing that could possibly work; rebuild without canonical helpers." My own bias.* Diagnosis: I would normally vote KILL but per CLAUDE.md "Forbidden premature KILL" I won't. The substrate is over-aggressive on "strip everything" — stripped *the things that made the score work*. Redesign: invert the strip-everything axiom to **strip-the-bytes-not-the-mechanism**: keep neural decoder, keep PyTorch, but strip the bytes (4-bit instead of fp16, no hyperprior, no entropy bottleneck) → become a quantized-but-faithful renderer. ΔS: **−95** (becomes a Quantizr-class submission). Cost: $10.

**Selfcomp / szabolcs-cs** — *Op-within: HARD-EARNED "1.017-bpw block-FP weight self-compression preserves distortion within rounding noise" — empirically verified at PR #56.* Diagnosis: my PR #56 worked because block-FP preserved the renderer; NSCS06 has NO renderer to preserve. Redesign: graft my block-FP-decoder + NSCS06's closed-form rate-allocator → 88K param renderer + analytical-coded latents (vs PR101's hyperprior-coded latents). ΔS predicted: **−90** (could reach PR101-class). Cost: $15.

**MacKay (memorial)** — *Op-within: HARD-EARNED "MDL principle: bits cost equals -log P(data | model); a model that compresses by destroying information pays in distortion." (Source: MacKay 2003 ITILA.)* Diagnosis: NSCS06 destroyed ALL chroma information (chroma bits in archive = 0; chroma distortion = 64+ points). MDL bound: chroma at 4 bits/cell × 96×128 × 600 pairs = 4.7 MB; current archive = 2.94 MB → MDL says you can ADD 1.5 MB chroma at zero net rate-penalty (still under PR101's effective compression ratio) and reclaim 60 distortion points. **Net ΔS: −60 from a structural MDL fix alone**. Cost: $5.

**Ballé** — *Op-within: HARD-EARNED "neural compression with hyperprior decisively beats hand-rolled codecs on natural-image distortion-rate trade-offs." (Source: my 2018 ICLR.)* Diagnosis: NSCS06 IS a hand-rolled codec; my work proves it can't beat learned codecs. Even my OWN factorized prior beats classical CDF coding by 1-3× at similar distortion. Redesign: graft my entropy bottleneck onto NSCS06's grayscale stream → 30-50% rate reduction at same distortion, OR equivalently 30-50% MORE bits available for chroma. ΔS: **−40** by itself; stacks with chroma fix for **−85** combined. Cost: $10.

### Grand council relevant specialists (extended)

**Carmack (memorial)** — *Op-within: HARD-EARNED "engineering shortcuts at the Doom/Quake level; ship simplest thing then iterate on what's measurably failing."* Diagnosis: V6 is the SIMPLEST thing — and it measurably fails on chroma + pose. The iteration target is OBVIOUS: add chroma (one bit-stream addition) + replace `np.roll` with optical-flow warp (one helper function). Stay analytical, just less stripped. ΔS: **−65** with two surgical additions. Cost: $5. *"Iterate on what's failing, not what's working."*

**Tao** — *Op-within: HARD-EARNED "any compression scheme that destroys an entire information channel pays at the distortion term proportional to the channel's MSE-energy; for chroma the energy ratio is ~10-20% of total." (Harmonic analysis.)* Diagnosis: chroma destruction → ~15% MSE-energy loss → translates via the PoseNet/SegNet sensitivity to ~50-80 distortion-score units. Empirically observed: 64.6 seg contribution. **Math agrees with empirical to within 30%.** Redesign vector: any chroma representation that captures even 20% of chroma MSE-energy will reclaim 50+ score points. Cost: free analytical.

**Filler** — *Op-within: HARD-EARNED "STC parity-check codes give near-optimal rate-distortion for per-frame payload coding." (Source: my Filler-Fridrich 2007 STC.)* Diagnosis: NSCS06's per-pixel arithmetic coding is suboptimal vs STC for the spatial-correlation structure. STC could reduce grayscale bits by 30-50%. Redesign: STC-code the grayscale residual from a previous-frame predictor → reclaim bits for chroma. ΔS: **−30 to −50** by itself. Cost: $5.

**Mallat** — *Op-within: HARD-EARNED "wavelet decompositions decorrelate natural images by 5-10× vs spatial pixel coding." (Source: Mallat 1989.)* Diagnosis: NSCS06 codes raw spatial palette indices — leaves all wavelet-domain decorrelation on the table. Redesign: wavelet-transform the grayscale + chroma streams BEFORE arithmetic coding → 5-10× rate reduction, then use the freed bits for chroma. ΔS: **−60 to −80** with full wavelet coding. Cost: $10.

**van den Oord** — *Op-within: CARGO-CULTED "discrete VQ-VAE tokens capture image structure better than continuous representations." (Source: my 2017 VQ-VAE.)* Diagnosis: NSCS06's palette is a 16-token VQ-VAE without the encoder/decoder learning. Redesign: train an actual VQ-VAE codebook (256-1024 tokens) + transformer prior over codes → would beat NSCS06 by 10-100× rate efficiency. ΔS: **−95** (becomes a real neural codec). Cost: $20. *But this is a CLASS-SHIFT not an NSCS06 refinement.*

**Hinton (memorial)** — *Op-within: HARD-EARNED "knowledge distillation T=2.0 transfers soft-target structure from teacher to student." (Source: Hinton-Vinyals-Dean 2014.)* Diagnosis: NSCS06 has NO student model to distill into; the CDF IS the entire "model". Redesign: keep NSCS06's analytical framework but distill the scorer's class-PMF predictions into the CDF rather than using argmax → would tighten the bit allocation. ΔS: **−15** modest. Cost: free.

**Hassabis (memorial)** — *Op-within: HARD-EARNED "in 4-day strategic windows, ship one polished candidate per class-shift; multi-domain pursuit is correct when uncertain."* Diagnosis: NSCS06 100% rules out pure-analytical class. Strategic recommendation: pursue 2-3 paths in parallel ($30-50 budget) per the "multi-path or staircase branch" operator directive. Don't single-thread on hybrid-NSCS06.

**Karpathy** — *Op-within: HARD-EARNED "let compute speak; arch-search reveals what hand-design misses."* Diagnosis: symposium #4 was hand-design without compute validation; v6 IS the compute speaking. Redesign: run a parallel sweep of 5-6 chroma+warp+codec variants in $50 budget; pick the empirical winner. *No more first-principles bands without empirical anchors.*

**Schmidhuber** — *Op-within: HARD-EARNED "compression-as-intelligence; the better the predictor, the better the compressor." (Source: Schmidhuber 1990s onward.)* Diagnosis: NSCS06's predictor is the SegNet argmax — that's a CLASSIFIER not a PREDICTOR of pixel values. A real predictor (autoregressive PixelRNN-style) would compress 5-10× tighter. Redesign: replace SegNet-argmax CDF with a learned 1-layer LSTM over palette indices → 3-5× rate savings. ΔS: **−40 to −60**. Cost: $15.

**Atick + Redlich (cooperative receiver)** — *Op-within: HARD-EARNED "early visual processing optimizes I(stimulus; cortical-rep) subject to channel capacity." (Source: Atick-Redlich 1990.)* Diagnosis: the contest scorer IS the receiver; cooperative-receiver framing says encode → maximize I(archive; scorer-output). NSCS06's closed-form allocator approximates this poorly because argmax-only ignores soft information. Redesign: full cooperative-receiver loss (Z4 substrate class) would dominate NSCS06 by construction. ΔS: **CLASS-SHIFT path**, not NSCS06 refinement. Cost: pursue Z4 instead.

**Rao + Ballard (predictive coding)** — *Op-within: HARD-EARNED "hierarchical predictive coding: each layer predicts the layer below; residuals are coded." (Source: Rao-Ballard 1999.)* Diagnosis: NSCS06 has NO temporal predictive coding — frame_1 is computed from frame_0 via `np.roll` instead of being a coded RESIDUAL from a temporal predictor. Redesign vector: code frame_t as PREDICTOR(frame_{t-1}, pose) + arithmetic-coded residual → MASSIVE rate savings + faithful pose. ΔS: **−70 to −85**. Cost: $15 (smallest temporal predictor training).

**Tishby (memorial) + Zaslavsky** — *Op-within: HARD-EARNED "Information Bottleneck: minimize I(X;T) subject to I(T;Y) ≥ ε; the optimal compressor at fixed score is the IB solution." (Source: Tishby 2000.)* Diagnosis: NSCS06's allocator is heuristic, not IB-optimal. The IB solution at the contest's distortion budget gives a closed-form optimal codebook structure. Redesign: replace the heuristic palette + CDF with IB-optimized vocabulary (compute via Blahut-Arimoto). ΔS: **−25 to −40**. Cost: $0 (analytical).

**Wyner (memorial)** — *Op-within: HARD-EARNED "Wyner-Ziv coding with side information: when the decoder has correlated information (like the previous frame), rate can be reduced by H(X|Y) instead of H(X)." (Source: Wyner-Ziv 1976.)* Diagnosis: NSCS06 codes frame_0 independently per pair; the previous pair's decoded frame is "side information" the decoder has for free. Redesign: Wyner-Ziv frame-pair coding could cut grayscale rate by 50-70%. ΔS: **−40**. Cost: $10.

**Jack-from-skunkworks** — *Op-within: HARD-EARNED "SegNet+Rate joint loss matters more than either alone; arch-search on the joint surface."* Diagnosis: NSCS06 has no joint loss because it has no training — that's a feature, not a bug, IF the analytical model is good enough. It isn't. Redesign: minimum-viable trainable component (5K param chroma reconstructor) joint-trained against SegNet+Rate. ΔS: **−50**. Cost: $10.

---

## Section 3 — Path enumeration (multi-path + staircase branch)

Per operator directive: "*spawn a grand council symposium all voices to debate and deliberate and discuss all paths to truly optimally engineer and design and optimize beautifully and elegantly this, can also pursue multiple paths if the path forward is unclear or erect a staircase branch*".

### Path A — NSCS06-v7 Carmack-Hotz Staircase A (minimal-fix: chroma + optical-flow)

- **Name:** NSCS06A "Two Surgical Additions"
- **Class-shift axis:** within-NSCS06 architecture refinement (NOT class-shift; pure-analytical preserved)
- **Predicted ΔS band:** `[−40, −65]` from 105.15 → `[40, 65]` `[prediction; multi-voice consensus]`
- **First-principles citation:** MacKay MDL chroma + Fridrich UNIWARD chroma allocation + Tao MSE-energy analysis (Section 2)
- **Cost band:** $5 Modal A100 smoke; ~$15 full
- **Reactivation criteria** (per "Forbidden premature KILL"): finite contest-CUDA paired score; chroma-MSE measurable improvement; ANY position on Dykstra-feasibility manifold
- **9-dim checklist:**
  1. Class-shift: NO (refinement only) ✗
  2. Real archive grammar: YES (CH06 v2; add CHROMA_LEN field) ✓
  3. Inflate ≤100 LOC: NO (chroma decoder adds ~30 LOC; total ~120) — substrate_engineering exception
  4. NVDEC bypass: YES (numpy only) ✓
  5. Score-aware loss: N/A (no training) — closed-form allocator now spans chroma too
  6. Bolt-on ≤350 LOC: NO (substrate engineering exception) — total +300 LOC
  7. eval_roundtrip simulation: YES at compress ✓
  8. Apples-to-apples axis: YES, every score [contest-CUDA] only ✓
  9. 6-hook wire-in: solver wire-in DEFERRED-pending-anchor; the rest active
- **Verdict:** LANDS-NEXT (lowest cost, fastest empirical anchor)

### Path B — NSCS06-v8 Carmack-Hotz Staircase B (mid-complexity: + wavelet + Wyner-Ziv)

- **Name:** NSCS06B "Decorrelate and Cooperate"
- **Class-shift axis:** within-NSCS06 + wavelet codec class
- **Predicted ΔS band:** `[−80, −90]` from 105.15 → `[15, 25]` `[prediction; Mallat+Wyner stacked]`
- **First-principles citation:** Mallat 1989 wavelets (5-10× spatial decorrelation) + Wyner-Ziv 1976 (side-info rate-savings) + Filler STC
- **Cost band:** $15 Modal A100 smoke; ~$40 full
- **Reactivation criteria:** Path A first; B only if A produces finite ≤ 50 score
- **9-dim checklist:** as A + (4) wavelet adds Pillow + pywavelets dep (3 inflate deps; substrate_engineering exception); (1) class-shift PARTIAL (wavelet IS a different codec class)
- **Verdict:** LAND-IF-A-PASSES

### Path C — NSCS06-v9 Hybrid Carmack-Hotz + Minimal-Neural-Residual

- **Name:** NSCS06C "Strip-The-Bytes-Not-The-Mechanism"
- **Class-shift axis:** PARTIAL class-shift to hybrid analytical+neural
- **Predicted ΔS band:** `[−90, −105]` from 105.15 → `[0, 15]` (potentially medal-band-class) `[prediction; Yousfi+Selfcomp+Ballé stacked]`
- **First-principles citation:** Ballé 2018 hyperprior + Selfcomp PR#56 block-FP + Hinton T=2.0 distillation
- **Cost band:** $30 Modal A100 smoke; ~$80 full
- **Reactivation criteria:** B first OR direct if operator wants leaderboard candidate
- **9-dim checklist:** (1) class-shift YES (hybrid); (3) inflate ~150 LOC + small NN (substrate_engineering exception); rest as B
- **Verdict:** LAND-IF-OPERATOR-WANTS-MEDAL-CANDIDATE (this is the only NSCS06 lineage that plausibly competes with PR101)

### Path D — DEFER-NSCS06 + Pursue Z4 Cooperative-Receiver

- **Name:** Z4-COOP-RECEIVER (class-shift away from NSCS06 lineage)
- **Class-shift axis:** FULL class-shift to cooperative-receiver
- **Predicted ΔS band:** `[−95, −105]` to `[0, 10]` if Z4 reactivates with full _full_main `[prediction; Atick-Redlich-Tishby class-shift]`
- **First-principles citation:** Atick-Redlich 1990 + Tishby 2000 IB; new sextet seat Assumption-Adversary RECOMMENDS
- **Cost band:** $50+ (Z4 substrate-engineering work; trainer currently raises NotImplementedError per recipe)
- **Reactivation criteria:** Z4 has its own scaffold (Catalog #240 research_only=true currently); reactivation requires phase-2 council approval
- **Verdict:** DEFER NSCS06 to L1, REDIRECT to Z4 only if operator chooses class-shift over NSCS06 refinement

### Path E — DEFER-NSCS06 entirely (no kill; pin reactivation criteria)

- **Name:** DEFER-NSCS06-pending-chroma-analytical-model
- **Class-shift axis:** none (preservation in catalog; no further dispatch)
- **Reactivation criteria** (per CLAUDE.md "Forbidden premature KILL"):
  1. Path A produces empirical receipt with finite contest-CUDA score AND
  2. Score < 50 (proof the additions matter) AND
  3. Operator-approved $15 for full-run AND
  4. Mathematical chroma+pose distortion bound model committed BEFORE next dispatch
- **Verdict:** Default IF operator chooses NOT to pursue A/B/C

### Path F — Carmack-Hotz Staircase BRANCH (parallel A+B+C with $50 budget)

- **Name:** NSCS06-STAIRCASE-PARALLEL
- **Class-shift axis:** Daubechies-style multi-scale staircase per CLAUDE.md "Council conduct" Time-Traveler seat
- **Predicted ΔS band:** picks the best of A/B/C empirically
- **First-principles citation:** Daubechies-DeVore-Fornasier-Gunturk 2010 + Catalog #277 wavelet multi-scale
- **Cost band:** $50 (A+B+C in parallel)
- **Verdict:** RECOMMENDED if operator can afford $50 (per CLAUDE.md "Race-mode rigor inversion" — parallel-dispatch is FIRST-CLASS deliverable)

---

## Section 4 — Consensus + vote tally

**Vote distribution across 22 voices, ranked-choice 1st preference:**

| Path | 1st preferences | 2nd preferences | 3rd preferences | Vetoes |
|---|---|---|---|---|
| A (minimal-fix) | 8 (Carmack, Shannon, MacKay, Tao, Hinton, Karpathy, Filler, Mallat-partial) | 6 | 2 | 0 |
| B (mid-complexity) | 4 (Mallat, Wyner, Filler, Schmidhuber) | 7 | 4 | 0 |
| C (hybrid-neural) | 5 (Yousfi, Hotz, Selfcomp, Ballé, Hassabis) | 5 | 6 | 0 |
| D (Z4 class-shift) | 3 (Assumption-Adversary, Atick-Redlich, Rao-Ballard) | 2 | 4 | 0 |
| E (defer entirely) | 0 | 0 | 0 | Contrarian objects |
| F (parallel A+B+C) | 2 (Karpathy, Hassabis as 2nd) | 6 | 1 | 0 |

**Consensus:** Path A is the dominant first choice (8 voices). Path F (parallel A+B+C) is the dominant SECOND choice (6 voices) — *consistent with "Race-mode rigor inversion" rule*. **No unanimous KILL** (per "Forbidden premature KILL" requirement).

**Contrarian dissent:** *"Path A alone produces 40-65 score, NOT medal-band. The operator wants score-lowering; A is not score-lowering relative to medal-band candidates. If operator's true objective is leaderboard, jump directly to C or D. If true objective is to validate the strip-everything thesis, A is correct."* Contrarian's vote: A as 1st (validate thesis); C as 2nd (leaderboard); D as 3rd (class-shift).

**Assumption-Adversary dissent + VETO**: *"No path proceeds without explicit cargo-cult unwinding. CARGO-CULTED #1 (closed-form argmax allocator is sufficient) MUST be either (a) empirically validated by Path A's chroma extension OR (b) replaced by gradient-aware allocator. CARGO-CULTED #2 (L5 RGB-renderer satisfied by Y=R=G=B) MUST be revised in Path A: define renderer as ANY function with chroma-MSE-energy ≥ 20% of input chroma. Any path that doesn't address these is REFUSED."*

**Sextet pact resolution**: Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary all vote PROCEED on Path A with cargo-cult unwinding mandate. Quorum reached: **Path A LANDS NEXT with cargo-cult-unwinding mandate as part of the scope.**

---

## Section 5 — Recommended operator-decision sequence

Three sequenced options ranked by EV/$ and dependency:

### Option 1 (BASELINE-RECOMMENDED): Path A first, then F if A passes

| Step | Path | Cost | Wall-clock | EV/$ | Blocking dep |
|---|---|---|---|---|---|
| 1 | A: chroma + optical-flow surgical additions ($5 smoke) | $5 | 1 day | high | none |
| 2 | IF A < 50 score: F parallel B+C ($45) | $45 | 2 days | medium | A pass |
| 3 | IF best of A/B/C < 20: full-run $15 | $15 | 1 day | high | empirical anchor |
| **Total** | | **$15-65** | **4 days** | | |

### Option 2 (RACE-MODE): Parallel F directly with $50

Per CLAUDE.md "Race-mode rigor inversion": if leaderboard moves OR operator wants speed, fan out A+B+C in parallel; pick winner.

| Step | Paths | Cost | Wall-clock |
|---|---|---|---|
| 1 | F parallel A+B+C smokes | $50 | 1 day |
| 2 | Full-run best smoke | $15 | 1 day |
| **Total** | | **$65** | **2 days** |

### Option 3 (CLASS-SHIFT-PIVOT): Defer NSCS06, pursue Z4

| Step | Action | Cost | Wall-clock |
|---|---|---|---|
| 1 | Mark NSCS06 DEFER-pending-chroma-analytical-model | $0 | 0 |
| 2 | Z4 substrate engineering ($30 development) | $30 | 3 days |
| 3 | Z4 smoke + full $40 | $40 | 2 days |
| **Total** | | **$70** | **5 days** |

**RECOMMENDED:** Option 1 if operator wants conservative learning. Option 2 if leaderboard-race active OR operator OK with $65. Option 3 if operator wants to abandon NSCS06 lineage AND class-shift to cooperative-receiver.

---

## Section 6 — HARD-EARNED vs CARGO-CULTED assumption inventory

Per `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`:

| # | Assumption | Classification | Source | Reactivation criteria (for CARGO-CULTED) |
|---|---|---|---|---|
| 1 | Closed-form scorer-argmax bit allocator suffices | CARGO-CULTED | symposium #4 enthusiasm | replace with gradient-aware allocator OR empirical validation in Path A |
| 2 | L5 RGB renderer = Y=R=G=B replication | CARGO-CULTED | scaffold self-PASS without external review | redefine renderer as chroma-MSE-energy ≥20% of input; Path A must satisfy |
| 3 | Spatial-independent CDF entropy is optimal | CARGO-CULTED | symposium #4 hand-wave | wavelet decomposition (Path B) OR neighbor-context CDF |
| 4 | NO neural decoder is achievable at medal-band | CARGO-CULTED | Carmack-Hotz radical premise | falsified by v6 (-553× off band); strict per Contrarian veto |
| 5 | Symposium #4 predicted band [0.10, 0.20] | CARGO-CULTED | rate-only first-principles bound | abandon; future bands require distortion model TOO |
| 6 | NPM rate term dominates score | CARGO-CULTED | true for medal-band candidates; FALSE at NSCS06 v6 (1.96 of 105) | preserve only for candidates already on Pareto frontier |
| 7 | PR #56 grayscale-LUT generalizes from masks to frames | CARGO-CULTED | extension by analogy without anchor | falsified by v6; preserve for masks only |
| 8 | strict-scorer-rule (no scorer at inflate) | HARD-EARNED | CLAUDE.md non-negotiable | preserve across all paths |
| 9 | Inflate ≤100 LOC + ≤2 deps (HNeRV L4) | HARD-EARNED | empirical from PR95-PR101 winners | preserve as design goal; substrate_engineering exception OK |
| 10 | Apples-to-apples evidence (every score tagged) | HARD-EARNED | CLAUDE.md non-negotiable + Catalog #127 | preserve; v6 correctly tagged `[diagnostic_cpu; B; score_claim_valid=False]` |
| 11 | eval_roundtrip at compress-time | HARD-EARNED | CLAUDE.md non-negotiable + Catalog #5 | preserve |
| 12 | PoseNet expects 6-dim ego-motion | HARD-EARNED | upstream/modules.py direct inspection | preserve; Path A optical-flow MUST use all 6 dims |
| 13 | SegNet uses RGB-distinguishing class cues | HARD-EARNED | empirical from chroma-zero v6 producing avg_seg=0.646 | preserve; Path A chroma reclamation REQUIRED |
| 14 | Dykstra feasibility-region intersection produces medal-band | HARD-EARNED | mathematical (Dykstra 1983) + empirically validated at A1 | preserve; check Path A for feasibility |
| 15 | Shannon R(D) prescribes rate vs distortion tradeoff | HARD-EARNED | Shannon 1948 + Berger 1971 | preserve; chroma-add at zero net rate is below R(D) |
| 16 | MDL bound on chroma reclamation | HARD-EARNED | MacKay 2003 ITILA | preserve; quantitative bound used in Path A prediction |
| 17 | Wavelet decorrelation 5-10× spatial codes | HARD-EARNED | Mallat 1989 + 40 years subsequent | preserve; used in Path B |
| 18 | Wyner-Ziv side-info coding | HARD-EARNED | Wyner-Ziv 1976 | preserve; used in Path B |

**Cargo-cult unwinding mandate** (per Assumption-Adversary veto): Path A scope MUST include explicit per-cargo-cult acknowledgment in its design memo; the chroma+optical-flow additions ARE the cargo-cult unwinding for items #1, #2, #4, #7.

---

## Section 7 — Op-routables (ranked by EV/$)

| Rank | Op-routable | Cost | Predicted ΔS impact | Dep | First-principles cite |
|---|---|---|---|---|---|
| 1 | Path A smoke: chroma channel (4-bit YUV at 96×128) + optical-flow warp (LK on grayscale) | $5 | −40 to −65 (105 → 40-65) | none | MacKay MDL + Fridrich UNIWARD + Lucas-Kanade optical flow 1981 |
| 2 | Cargo-cult-unwinding design memo addendum to NSCS06 scaffold (CLAUDE.md substrate canonical-vs-unique section per Catalog #290) | $0 | 0 (process protection; prevents cargo-cult recurrence) | none | CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" non-negotiable |
| 3 | Distortion-bound model committed BEFORE any future Carmack-Hotz dispatch | $0 (analytical) | prevents next 500× falsification | Path A empirical | Tao MSE-energy + Shannon R(D) |
| 4 | If Path A produces score in [20, 50]: Path B smoke (wavelet decomposition + Wyner-Ziv frame coding) | $15 | −20 to −40 (A baseline → 5-25) | Path A pass | Mallat 1989 + Wyner-Ziv 1976 |
| 5 | If Path A produces score in [40, 65]: Path C smoke (small NeRV residual decoder graft) | $30 | −30 to −55 (A baseline → 5-35) | Path A pass | Ballé 2018 + Selfcomp PR#56 |
| 6 | Mark Z4 cooperative-receiver and Z5 predictive-coding as ALSO-DEFERRED with explicit reactivation criteria | $0 | 0 (catalog hygiene; preserves Catalog #240 contract) | none | CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" |
| 7 | If Paths A+B+C all produce score > 50: explicit DEFER-NSCS06 with reactivation criteria pinned in lane registry | $0 | 0 (catalog hygiene per "Forbidden premature KILL") | A+B+C results | CLAUDE.md "Lane maturity registry" |
| 8 | Predict-then-measure protocol: every future Carmack-Hotz dispatch publishes pose+seg+rate components BEFORE dispatch fires (prevents next first-principles-bound surprise) | $0 (process) | 0 today; ongoing protection | none | Catalog #229 (premise verification) |
| 9 | Update CLAUDE.md "Production-hardened dispatch optimization protocol" to add Tier 4: "predicted-vs-empirical bound published before dispatch" | $0 (process) | 0 today; ongoing protection | none | CLAUDE.md "Bugs must be permanently fixed AND self-protected against" |
| 10 | If operator pivots to D (Z4 class-shift): lane_z4_cooperative_receiver_phase_2 design memo with full _full_main implementation plan | $30+ | −95+ (potential medal-band; class-shift) | operator decision | Atick-Redlich 1990 + Tishby 2000 IB |

---

## Final summary (per task spec deliverable)

- **(a)** Total voices participating: **22** (6 inner sextet + 4 inner-ten + 12 grand council)
- **(b)** Total paths enumerated: **6** (A minimal-fix, B mid-complexity, C hybrid-neural, D class-shift Z4, E defer, F parallel-staircase)
- **(c)** Recommended top-3 paths:
  1. **Path A** $5/predicted-ΔS −40 to −65 (lowest cost; fastest empirical anchor)
  2. **Path F** $50/best-of-A-B-C (race-mode; parallel dispatch per CLAUDE.md "first-class deliverable")
  3. **Path C** $30/predicted-ΔS −90 to −105 (only NSCS06 lineage that plausibly competes with PR101)
- **(d)** Consensus tally: **8-4-5-3-0-2** for first-preferences across A/B/C/D/E/F; Path A dominant
- **(e)** Assumption-Adversary's flagged shared assumptions:
  - **7 CARGO-CULTED** (#1 closed-form-argmax-allocator; #2 L5-RGB-by-Y=R=G=B; #3 spatial-independent-CDF; #4 NO-neural-at-medal; #5 symposium-#4-predicted-band; #6 rate-dominates-score; #7 PR#56-generalizes)
  - **11 HARD-EARNED** (preserved); see Section 6 table
- **(f)** Contrarian's dissent: Path A alone produces 40-65, NOT medal-band; if operator wants score-lowering jump to C or D; Contrarian VOTES A as 1st conditional on operator clarifying objective
- **(g)** 9-dim checklist evidence per top-3 paths: see Section 3 per-path checklists
- **(h)** 5-10 op-routables ranked by EV/$: see Section 7 (10 op-routables total)

**Catalog #229 premise verification:** 6 PVs in Section 1 mechanism analysis pre-edit (PV-1 source decomposition / PV-2 chroma destruction / PV-3 warp insufficiency / PV-4 archive size puzzle / PV-5 eval_roundtrip mismatch / PV-6 L5 cargo-cult).

**Catalog #206 checkpoint discipline:** 2 in-progress + this completion checkpoint (3 total).

**Catalog #230 disjoint scope:** READ-ONLY for source code. Wrote only to `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` + memory landing memo. Sister subagents (lane registry mutation, STRICT-gate landing) NOT touched.

**Catalog #291 cadence:** This deliberation IS a META-ASSUMPTION-class review per the Assumption-Adversary's per-cargo-cult classification; satisfies the per-session recurring cadence requirement.

**Catalog #292 sextet-pact discipline:** ALL 6 inner sextet members (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary) explicitly stated operating-within assumptions per CLAUDE.md "Council conduct" amendment; PROCEED quorum reached on Path A with cargo-cult-unwinding mandate.

**Apples-to-apples evidence discipline (CLAUDE.md non-negotiable):** every empirical anchor tagged `[empirical:experiments/results/lane_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_20260516T121743Z__smoke__100ep_modal/harvested_artifacts/lane_nscs06_carmack_hotz_results/output/contest_auth_eval_cpu.json; diagnostic_cpu; B; score_claim_valid=False]`; every predicted ΔS tagged `[prediction; first-principles-bound]`.

**Forbidden premature KILL discipline:** ZERO kill verdicts. NSCS06 preserved in catalog with explicit reactivation criteria across all paths.

**Lane:** `lane_council_symposium_nscs06_falsification_redesign_multipath_20260516`
