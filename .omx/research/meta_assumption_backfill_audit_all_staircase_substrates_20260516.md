# META-ASSUMPTION BACKFILL AUDIT — All staircase substrates (HARD-EARNED vs CARGO-CULTED decomposition)

**Date:** 2026-05-16
**Lane:** `lane_meta_assumption_backfill_audit_all_staircase_substrates_20260516`
**Subagent:** META-ASSUMPTION BACKFILL AUDITOR
**Cadence event:** This audit IS the META-ASSUMPTION ADVERSARIAL REVIEW cadence event per Catalog #291. Most-recent prior META-ASSUMPTION review: `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (1 day ago, under 7-day cadence threshold; cadence still healthy but this audit ADDS the per-substrate cargo-cult decomposition the 18-assumption matrix did NOT contain).

**Per CLAUDE.md non-negotiables honored:**
- "META-ASSUMPTION ADVERSARIAL REVIEW" (Catalog #291) — explicit cadence event
- "Council conduct" sextet pact + Assumption-Adversary seat (Catalog #292) — per-substrate operating-within assumptions surfaced
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — per-substrate canonical-vs-unique decomposition
- "Forbidden premature KILL without research exhaustion" — every verdict ADDS reactivation criteria, never kills
- "HNeRV / leaderboard-implementation parity discipline" 13 lessons — assessment dimension
- "Apples-to-apples evidence discipline" — every score claim axis-tagged
- "Subagent coherence-by-default" Catalog #229 + #206 + #230 — premise verification + checkpoint + sister-disjoint

---

## Section 1 — Audit methodology + framing

### The structural gap this audit closes

On 2026-05-15, three discipline gates landed: Catalog #291 (`check_session_has_recent_meta_assumption_review` — periodic META-ASSUMPTION review cadence), Catalog #292 (`check_grand_council_deliberation_has_explicit_assumption_statements` — per-deliberation assumption surfacing), and the sextet-pact council seat for the Assumption-Adversary. These gates were forward-looking — they protect FUTURE substrate designs from assumption-backdrop blindness.

**They did NOT retroactively audit the ~140 L1+ substrate lanes already in the registry.**

On 2026-05-16, NSCS06 v6 Modal smoke (`lane_substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_20260516T121743Z__smoke__100ep_modal`) returned `final_score=105.15` against a symposium-#4 predicted band `[0.10, 0.20]` — **a 553× falsification ratio**. The post-mortem symposium (`grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`) surfaced **7 cargo-culted assumptions** that were classifiable AT DESIGN TIME by static code review (e.g., `Y=R=G=B` grayscale-to-RGB is a 2-line catch; `np.roll` with 2 of 6 pose dims is a 4-line catch).

**By inference: every staircase substrate predating the 2026-05-15 META-ASSUMPTION discipline cutoff may carry similar undiscovered cargo-culted assumptions, classifiable BEFORE empirical falsification.** This audit performs that backfill.

### Methodology

For each L1+ substrate lane in `.omx/state/lane_registry.json`:

1. Read its design memo (`.omx/research/<substrate>_design_<date>.md` or sister), trainer docstring, and inflate runtime.
2. Decompose its inherited assumptions into HARD-EARNED (cite source: CLAUDE.md non-negotiable / catalog # / empirical incident anchor / contest rule) vs CARGO-CULTED (no citation; engineering convenience).
3. Per CARGO-CULTED assumption, assess risk band (LOW / MEDIUM / HIGH) by reference to predicted ΔS impact if violated.
4. Verdict: HEALTHY / AUDIT-PENDING-DEEPER-PROBE / HIGH-RISK-REDESIGN-RECOMMENDED / ALREADY-FALSIFIED-CANONICAL.

Per Catalog #292 + the assumption-classification addendum, EVERY assumption listed below carries the operating-within statement + classification + source citation (HARD-EARNED) or alternative-hypothesis (CARGO-CULTED).

### Premise verifications (Catalog #229; 7 PVs)

- **PV-1:** NSCS06 symposium output exists at `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` (38.8K bytes; canonical template).
- **PV-2:** Lane registry contains 139 L1+ substrate lanes; 8 in Tier 1, 28+ in Tier 2, 95+ in Tier 3.
- **PV-3:** Hard-earned-vs-cargo-culted classification addendum exists at `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`; canonical taxonomy + 18 example classifications.
- **PV-4:** Most-recent META-ASSUMPTION review = 2026-05-15 (1 day ago); Catalog #291 cadence threshold = 7 days OR 50 landings (whichever first); current state: HEALTHY but RIPE for the substrate-backfill complement.
- **PV-5:** `experiments/train_substrate_*.py` enumerates 38 trainer files; the L1+ substrate set we audit spans 15 representative substrates (the priority slice).
- **PV-6:** Next available catalog # = 296 (per `.omx/state/next_catalog_number.txt`).
- **PV-7:** Sister subagents in flight: per CLAUDE.md "Subagent coherence-by-default" Catalog #230, this audit is READ-ONLY for source code; writes only to `.omx/research/` + Claude memory; no sister-subagent collision.

---

## Section 2 — Per-substrate audit

### TIER 1: dispatch-enabled / just-failed / canonical anchors

#### 2.1 NSCS06 — Carmack-Hotz Strip-Everything (canonical reference)

- **Lane**: `lane_nscs06_carmack_hotz_strip_everything_20260515` (L2, research_only=true)
- **Trainer**: `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py`
- **Status**: ALREADY-FALSIFIED-CANONICAL (105.15 vs predicted [0.10, 0.20])
- **Anchor**: 2026-05-16 Modal T4 smoke = 553× falsification ratio

**HARD-EARNED PRESERVED** (11 items, per symposium Section 6):
1. strict-scorer-rule (CLAUDE.md non-negotiable) — preserved ✓
2. Inflate ≤100 LOC + ≤2 deps (HNeRV parity L4) — preserved ✓
3. Apples-to-apples evidence (every score tagged) — preserved ✓ (v6 correctly tagged `[diagnostic_cpu; B; score_claim_valid=False]`)
4. eval_roundtrip at compress-time (CLAUDE.md non-negotiable + Catalog #5) — preserved ✓
5. PoseNet expects 6-dim ego-motion (upstream/modules.py) — preserved ✓ (but VIOLATED IN PRACTICE; see CC-2)
6. SegNet uses RGB-distinguishing class cues (empirical) — preserved at type level ✓ (but VIOLATED IN PRACTICE; see CC-2)
7. Dykstra feasibility-region intersection — preserved as analysis framework ✓
8. Shannon R(D) — preserved as analysis framework ✓
9. MDL bound on chroma reclamation (MacKay 2003) — preserved ✓
10. Wavelet decorrelation 5-10× (Mallat 1989) — available for Path B ✓
11. Wyner-Ziv side-info coding — available for Path B ✓

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (7 items, per symposium Section 6):
- **CC-1** Closed-form scorer-argmax bit allocator suffices (no gradient signal) — **HIGH-RISK** — ΔS impact band [+60, +100] (empirically confirmed)
- **CC-2** L5 "full RGB renderer" satisfied by Y=R=G=B replication — **HIGH-RISK** — ΔS impact band [+60, +80] (empirical seg=64.59 contribution)
- **CC-3** Spatial-independent CDF entropy is optimal — **HIGH-RISK** — ΔS impact band [+15, +30] (200-1000× archive size discrepancy)
- **CC-4** NO neural decoder is achievable at medal-band — **HIGH-RISK** — falsified by v6 553× ratio
- **CC-5** Symposium #4 predicted band [0.10, 0.20] from "rate-only first-principles bound" — **HIGH-RISK** — falsified
- **CC-6** Rate term dominates score (CARGO-CULTED for off-frontier candidates) — **MEDIUM-RISK** — rate=1.96 of 105 (1.9%) at NSCS06; assumption only valid on Pareto frontier
- **CC-7** PR #56 grayscale-LUT generalizes from masks to frames — **HIGH-RISK** — falsified

**Cargo-cult density:** 7 (the empirical baseline; anything >5 = HIGH-RISK)
**Verdict:** ALREADY-FALSIFIED-CANONICAL; symposium output enumerates Paths A/B/C/D/E/F + reactivation criteria. No action required from THIS audit beyond confirming presence.

---

#### 2.2 A1 — SegNet Margin / Time-Traveler L5 (sub-0.193 [contest-CPU] anchor)

- **Lane**: `lane_a1_pr_submission_entry_packet` (L1)
- **Trainer**: A1 is the established sub-0.193 anchor; not a NEW substrate but the canonical Z3/Z4/ATW base
- **Status**: dispatch-enabled (anchor of record per `feedback_z1_mdl_ablation_landed_20260514` density=99.29% on Tier A)

**HARD-EARNED PRESERVED** (12 items):
1. eval_roundtrip + EMA + scorer preprocess differentiable + apples-to-apples + strict scorer rule (all CLAUDE.md non-negotiables)
2. Inflate ≤100 LOC + monolithic 0.bin (HNeRV parity L3/L4)
3. SegNet last-frame slice + PoseNet 6-dim ego-motion contract
4. Pose contribution = `sqrt(10 * d_pose)` formula (contest scoring rule)
5. Archive bytes count = `25 * archive_bytes / 37_545_489` (contest rate)
6. brotli compression preserves zero distortion (lossless on integer streams)
7. fp16 latent quantization preserves PR101 score (empirical from A1 paired anchor)
8. SegNet boundary smoothing improvement is real (lane_a1_segnet_boundary_smoothing_inflate L2)
9. Inflate-time bias correction sweep is constrained by stale calibration (research_only lane validated)
10. Z1 ablation result (Tier A density = 99.29% → A1 is WITHIN-CLASS SATURATED) is empirical
11. Class-shift moves are needed for further floor-drop (Z1 EUREKA)
12. PR101 family grammar reuse is structurally compatible (lane_pr106_latent_sidecar_r2_pr101_grammar L2)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "Brotli on fp16 weights is the optimal entropy coder for renderer state" — **LOW-RISK** — empirical (PR101 grammar); but Tier A density 99.29% suggests Ballé hyperprior (Z3) could gain 5-15% rate at zero distortion
- **CC-2** "EMA decay 0.997 across all decoder weights" — **LOW-RISK** — Quantizr verified at PR #56; but Ballé's 0.999 for hyperprior decoder is empirical evidence the same-decay-for-all is cargo-culted
- **CC-3** "Within-class refinement on A1 will not yield medal-band score" — **LOW-RISK** — Z1 ablation IS the empirical evidence; class-shift required (formalizes the hard-earned insight)

**Cargo-cult density:** 3 (LOW)
**Verdict:** HEALTHY — A1 is the empirical anchor that DEFINED the within-class saturation discovery; the 3 cargo-cults are LOW-RISK because they're already known. No action beyond confirming the Z3/Z4/ATW/Z3-G1 class-shift moves are the right next steps.

---

#### 2.3 PR101 LC v2 Clone (canonical PR101 reproduction)

- **Lane**: `lane_substrate_pr101_lc_v2_clone_20260512` (L1)
- **Trainer**: `experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py`
- **Status**: dispatch-enabled; HEAD-PARITY ledger Catalog #166 wired

**HARD-EARNED PRESERVED** (10 items):
1-9. All A1's hard-earned set
10. PR95/PR101 winners' bind-all-ingredients pattern (HNeRV parity discipline lessons 1-13 — the empirical truth this clone embodies)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (2 items):
- **CC-1** "Enhanced curriculum (longer schedule + curriculum scheduler) will beat PR101's published score" — **MEDIUM-RISK** — curriculum tuning is empirical but the predicted ΔS band is hand-waved in the trainer docstring; needs predict-then-measure protocol
- **CC-2** "PR101 grammar is the optimal archive grammar at PR101's distortion" — **LOW-RISK** — empirically confirmed at PR101 0.193 contest-CPU anchor; Z1 ablation shows it's class-saturated (i.e., grammar can't gain more)

**Cargo-cult density:** 2 (LOW)
**Verdict:** HEALTHY — this IS PR95-parity reproduction by construction (designed to faithfully reproduce the leaderboard winner). The 2 cargo-cults are LOW-MEDIUM and don't structurally suppress the score. Predict-then-measure protocol (NSCS06 op-routable #8) should be applied to the enhanced-curriculum prediction before next dispatch.

---

#### 2.4 sane_hnerv (just-landed canary)

- **Lane**: `lane_substrate_sane_hnerv_20260512` (L1) + `lane_sane_hnerv_archive_fix_catalog_161_20260513` (L2 substrate_engineering)
- **Trainer**: `experiments/train_substrate_sane_hnerv.py`
- **Status**: dispatch-enabled (Catalog #166 sentinel hardening completed); 5 prior failed first-anchor attempts on 2026-05-12 per CC4 review

**HARD-EARNED PRESERVED** (11 items):
1-10. All A1/PR101's hard-earned set
11. HNeRV parity discipline L1-L13 (the substrate IS the explicit HNeRV reproduction)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (4 items):
- **CC-1** "Sane HNeRV (cleanup pass on the original HNeRV) will reproduce 0.193-0.195 [contest-CPU]" — **MEDIUM-RISK** — 5 failed first-anchor attempts on 2026-05-12 are empirical evidence the assumption is fragile
- **CC-2** "Single-canary-first ordering is the right race-mode dispatch pattern" — **LOW-RISK** — Catalog #173 (canary-first ordering) makes this explicit and enforceable
- **CC-3** "HNeRV's `content-adaptive-embedding` is necessary AND sufficient for the score floor" — **MEDIUM-RISK** — empirical truth of "necessary" is established; "sufficient" is not (PR101/PR103/PR102 all add bolt-ons on top)
- **CC-4** "Standard `score_pair_components` canonical helper is the optimal scorer-loss routing for HNeRV" — **LOW-RISK** — Catalog #164 hard-earns the differentiability invariant; canonical helper passes the invariant but the SPECIFIC dispatch may not be HNeRV-optimal

**Cargo-cult density:** 4 (MEDIUM)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — 5 failed first-anchor attempts on 2026-05-12 are exactly the empirical signal that the SUBSTRATE ITSELF (not the integration) has design fragility. Recommend: paired-comparison smoke between canonical-helper and custom-scorer-loss routing variants at $5 Modal A100 to disambiguate CC-4.

---

### TIER 2: research_only / scaffold-built; ready to dispatch on operator gate-flip

#### 2.5 NSCS01 — Nullspace Split Renderer

- **Lane**: `lane_nscs01_nullspace_split_renderer_20260515` (L1, research_only)
- **Trainer**: `experiments/train_substrate_nscs01_nullspace_split_renderer.py` (~420 LOC `_full_main` implemented)
- **Status**: research_only=true; awaiting Phase 2 council approval

**HARD-EARNED PRESERVED** (11 items):
1-10. All A1/PR101's hard-earned set
11. SegNet `x[:, -1, ...]` last-frame slice property (verified at `upstream/modules.py:108`) — the substrate EXPLOITS this; preservation = correct citation in design memo §1 PV-1

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (2 items):
- **CC-1** "Frame_0 head can be 4-bit while frame_1 head needs 8-bit (asymmetric quantization is optimal)" — **MEDIUM-RISK** — assumed without empirical anchor; could be 6/6 instead of 4/8; predicted ΔS band [+0.001, +0.003] if wrong; cost of measurement = paired-comparison smoke $5
- **CC-2** "PoseNet frame_0 sensitivity is low enough that 30K param frame_0_head suffices" — **MEDIUM-RISK** — assumed without empirical anchor; the probe-disambiguator `tools/probe_nscs01_head0_arch_disambiguator.py` IS designed to measure this; the design memo correctly defers the prediction band until the probe runs

**Cargo-cult density:** 2 (LOW)
**Verdict:** HEALTHY — NSCS01 is a model citizen of the post-2026-05-15 UNIQUE-AND-COMPLETE-PER-METHOD operating mode. The design memo §10 carries a Canonical-vs-unique decision per layer section (Catalog #290); the two cargo-cults are EXPLICITLY DEFERRED to the probe-disambiguator. CLEARED FOR DISPATCH on operator gate-flip.

---

#### 2.6 NSCS02 — Downsampled Renderer Inflate Upsample

- **Lane**: `lane_nscs02_downsampled_renderer_inflate_upsample_20260515` (L1)
- **Trainer**: `experiments/train_substrate_nscs02_downsampled_renderer.py`
- **Status**: research_only

**HARD-EARNED PRESERVED**: A1/PR101 set + camera resolution (1164, 874) contract + bicubic/bilinear interpolation contracts (image-processing canonical)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (4 items):
- **CC-1** "Renderer at downsampled resolution (192×128 or 96×64) + bicubic upsample at inflate matches full-res renderer at lower bytes" — **HIGH-RISK** — directly analogous to NSCS06's `np.repeat` cargo-cult; bicubic upsample destroys high-freq texture the SegNet's RGB-distinguishing class cues depend on
- **CC-2** "Bicubic interpolation preserves SegNet argmax accuracy at <2× downsample ratio" — **MEDIUM-RISK** — assumed without empirical anchor; the SegNet stride-2 stem has 256-pixel effective receptive field, so 2× downsample may be tolerable but 4× definitely is not
- **CC-3** "PoseNet's 12-channel YUV6 input is tolerant to bilinear upsample at the (512, 384) preprocess step" — **MEDIUM-RISK** — `rgb_to_yuv6` runs on the upsampled RGB; chroma subsampling means upsampling chroma channels has 2× tolerance; but luma is the dominant pose signal
- **CC-4** "Symmetric (compress + inflate) bicubic preserves eval_roundtrip simulation" — **HIGH-RISK** — eval_roundtrip simulates 384→874→uint8→384 but the trainer's "downsampled renderer" effectively becomes 192→874→uint8→384 if compress applies bicubic too late; CARGO-CULT-CADENCE EXACTLY MATCHES NSCS06 v6 PV-5 ("compress saw 'what the renderer will produce' but the renderer is structurally incapable of producing high-freq")

**Cargo-cult density:** 4 (MEDIUM-HIGH; mirrors NSCS06 cargo-cult pattern)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — NSCS02's downsample-upsample paradigm is structurally analogous to NSCS06's Y=R=G=B (both destroy a signal axis and bet on reconstruction); the 553× NSCS06 falsification is empirical evidence the bet is risky. **Recommend: predict-then-measure paired comparison: NSCS02 trainer at 384×512 vs 192×256 vs 96×128 to find the empirical downsample-vs-distortion knee BEFORE any Modal dispatch.** Cost: $5 paired CPU smoke.

---

#### 2.7 NSCS03 — End-to-End Ballé Joint Codec

- **Lane**: `lane_nscs03_end_to_end_balle_joint_codec_20260515` (L1)
- **Trainer**: `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py` (+548 LOC `_full_main`)
- **Status**: research_only=true until Phase 2 council λ_R sweep + σ-floor calibration

**HARD-EARNED PRESERVED**: A1/PR101 set + Ballé 2018 entropy-bottleneck mathematical contract (`bits = -log2(p_y(y))`) + GDN nonlinearity decorrelation property + hyperprior amortization principle (`|y stream savings| > |w + MLP weights|`)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "λ_R linear warmup 0→target across first 10% epochs is the canonical Ballé schedule" — **LOW-RISK** — citation: Ballé 2018 used exact this; HARD-EARNED in literature but CARGO-CULTED in pact's specific operating point (PR101 distortion regime); could be cosine or step
- **CC-2** "5 sub-nets (g_a/g_s/h_a/h_s/EB) is the optimal Ballé topology for our distortion budget" — **MEDIUM-RISK** — published Ballé 2018 used 4 sub-nets for natural images; the 5-net variant is from later papers (Ballé 2021); not empirically anchored to contest video
- **CC-3** "Ballé 0.999/0.997 differentiated EMA decay is necessary" — **LOW-RISK** — Phase 2 deferred per substrate's own design; correctly cargo-cult-flagged

**Cargo-cult density:** 3 (LOW)
**Verdict:** HEALTHY — NSCS03 carries explicit Canonical-vs-unique decisions; 14 CANONICAL ADOPT + 4 UNIQUE + 3 DOCUMENTED FORK per design memo. The 3 cargo-cults are LOW-MEDIUM and well-documented. CLEARED FOR DISPATCH on operator gate-flip.

---

#### 2.8 D1 — SegNet Margin Polytope

- **Lane**: Multiple L1+ lanes (D1 has the canonical recovery anchor `lane_d1_l2_integration_plus_no_byte_addition_without_score_improvement_gate_20260514` at L1)
- **Trainer**: `experiments/train_substrate_d1_segnet_margin_polytope.py`
- **Status**: dispatch-enabled with L2 OVERLAY active (Catalog #220 OPERATIONAL); 43 KB → 2.7 KB shrunk variant available

**HARD-EARNED PRESERVED**: A1/PR101 set + SegNet logit-margin classifier property (the polytope IS the margin geometry) + per-pixel noise overlay applied to frame_1 RGB at camera resolution

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "Per-pixel polytope-interior noise overlay improves SegNet argmax stability" — **MEDIUM-RISK** — D1 R3 anchor 2026-05-14 produced 0.222 vs predicted [0.181, 0.188] band (NSCS06-class falsification of the prediction band; the OVERLAY itself remains unverified at score-improvement)
- **CC-2** "Margin map resolution (96×128 shrunk variant) preserves margin geometry" — **MEDIUM-RISK** — 16× archive cost reduction; tested at byte-mutation level (Catalog #220 PASSED) but score-improvement empirical anchor pending
- **CC-3** "Margin map int8 quantization is the optimal compromise (vs fp16 or 4-bit)" — **LOW-RISK** — assumption from D1 design; reasonable but unanchored

**Cargo-cult density:** 3 (LOW-MEDIUM)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — D1 has an empirical anchor (R3 0.222) that ALREADY FALSIFIED the symposium-#4-class predicted band. CC-1 is the dominant assumption requiring direct empirical disambiguation: does the polytope-interior noise IMPROVE SegNet stability, or just add inert bytes? Recommend the existing L2 OVERLAY's runtime effect be re-measured with byte-mutation smoke (Catalog #139) on the score-effect axis specifically.

---

#### 2.9 D4 — Wyner-Ziv Frame 0

- **Lane**: D4 substrate-engineering family (research_only)
- **Trainer**: `experiments/train_substrate_d4_wyner_ziv_frame_0.py` (with mini-batch reconstruct per Catalog #218)
- **Status**: research_only; T4 OOM previously fixed via Catalog #218 mini-batch

**HARD-EARNED PRESERVED**: A1/PR101 set + Wyner-Ziv 1976 source-coding-with-side-information mathematical contract (R_X|Y(D) ≤ R_X(D)) + SegNet last-frame nullspace property (D4 exploits the same `x[:, -1, ...]` insight as NSCS01)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "Wyner-Ziv gap H(Y|S) - H(Y|X,S) is large enough on dashcam pairs to yield meaningful rate savings" — **MEDIUM-RISK** — assumed without empirical anchor; the gap depends on inter-pair temporal correlation which is high in dashcam but the upper bound matters; recommend H(latent | scorer_class) probe BEFORE paid dispatch (matches ATW's recommended probe)
- **CC-2** "frame_0 derived from frame_1 via WZ residual matches frame_0 derived from independent renderer" — **MEDIUM-RISK** — WZ encoding adds dependency between frames; if temporal correlation is weak, the residual stream is ALSO weak; bidirectional risk
- **CC-3** "600-pair-batch reconstruct fits T4 14.56GB VRAM with mini-batch (Catalog #218)" — **LOW-RISK** — empirically verified post-fix

**Cargo-cult density:** 3 (LOW-MEDIUM)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — CC-1 is the dominant assumption; the H(latent | scorer_class) probe IS the right disambiguator and is shared with ATW (sister design). Recommend single probe shared across D4 + ATW before either fires Modal dispatch.

---

#### 2.10 Z3 v2 + Z3-G1 — Ballé Hyperprior Bolt-on + Scorer Softmax Hyperprior Gating

- **Lanes**: `lane_z3_balle_hyperprior_bolton_campaign_20260514` + `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515` (both L2 research_only)
- **Trainers**: `experiments/train_substrate_z3_balle_hyperprior_bolton.py` + `train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py`
- **Status**: Z3 v2 anchored; Z3-G1 empirically falsified at codex review level (F1: archive ships EMPTY hyperprior_weights_int8 slots; F2 silent uniform-class fallback)

**HARD-EARNED PRESERVED**: A1/PR101 set + Ballé 2018 hyperprior mathematical contract + Catalog #266/#267 F1+F2 fail-closed gates (LANDED)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (4 items, F1+F2 already empirically falsified):
- **CC-1** "Z3HV2 archive grammar can carry empty hyperprior_weights slots and still ship the 'scorer-class gating' substrate effect" — **FALSIFIED EMPIRICALLY** — Z3-G1 smoke `fc-01KRPKCXARWP7NBGJCXB2P9QEP` returned 0.19869 IDENTICAL to Z3 v2 baseline (5 decimals); the G1-specific bytes never enter the wire format
- **CC-2** "SegNet class derivation falls back to uniform deterministically if upstream changes" — **FALSIFIED EMPIRICALLY** — silent fallback under ANY exception, including non-scorer errors (codex F2)
- **CC-3** "Ballé hyperprior on A1's 15387-byte latent blob will yield 5-15% rate savings" — **MEDIUM-RISK** — Z1 Tier A density = 99.29% says encoder is class-saturated; realistic ΔS = -0.001 to -0.003 (operator-acknowledged in design memo §2)
- **CC-4** "Phase 2 council approval required to lift `_full_main` NotImplementedError is the right safety pattern" — **LOW-RISK** — HARD-EARNED at the gate level (Catalog #240); CARGO-CULTED at the per-substrate process level (some substrates may not need council approval)

**Cargo-cult density:** 4 (HIGH due to 2 FALSIFIED items)
**Verdict:** HIGH-RISK-REDESIGN-RECOMMENDED for Z3-G1 (already research_only=true; reactivation criteria pinned per Catalog #266/#267); CARGO-CULT-UNWINDING-MANDATE for Z3 v2 if it returns to dispatch (CC-3 means the predicted ΔS band must be revised DOWN). The Z3-G1 case is a CANONICAL EXAMPLE of the bug class this audit was designed to catch.

---

#### 2.11 Z4 — Cooperative Receiver Loss

- **Lane**: `lane_z4_cooperative_receiver_loss_step2_20260514` (L1 research_only)
- **Trainer**: `experiments/train_substrate_z4_cooperative_receiver_loss.py` (`_full_main` raises NotImplementedError)
- **Status**: research_only=true per Catalog #240 (recipe-vs-trainer-state consistent)

**HARD-EARNED PRESERVED**: A1/PR101 set + Atick-Redlich 1990 cooperative-receiver theorem (L_AR = α · H(X | f_R(X))) + Tishby-Zaslavsky IB framework

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (4 items):
- **CC-1** "Cooperative-receiver loss with λ_pixel=0 (pure scorer-aware) yields better score than λ_pixel>0 baseline" — **MEDIUM-RISK** — design memo §2 hypothesis; predicted ΔS [-0.005, -0.010] is small; could go either way; the disambiguator (κ_IB / λ_WZ ablation in ATW) is the right next test
- **CC-2** "Atick-Redlich H(X | f_R(X)) lower bound is achievable in practice with the existing canonical scorer-helper" — **MEDIUM-RISK** — theorem holds asymptotically; finite-data gap unanchored
- **CC-3** "Predicted band [0.180, 0.188]" — **MEDIUM-RISK** — derived from Z3 [0.188, 0.193] baseline + cooperative-receiver Δ; chain of unmeasured deltas
- **CC-4** "Step 2 in the staircase needs only λ_pixel=0 swap (loss-only change)" — **LOW-RISK** — design memo correctly notes WZ residual (Step 3) is the bigger lever

**Cargo-cult density:** 4 (MEDIUM)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — Z4 IS structurally well-grounded (Atick-Redlich theorem is HARD-EARNED in literature) but CC-1 + CC-3 are the unmeasured-prediction bug class that NSCS06 hit. Recommend: ATW canonical probe (κ_IB=0, λ_WZ=1, λ_pixel=0 ablation) disambiguates Z4's λ_pixel hypothesis as a side-effect at no extra cost.

---

#### 2.12 C6 — MDL-IBPS (e4 variant)

- **Lane**: `lane_c6_e4_mdl_ibps_substrate_campaign_20260514` (L1)
- **Trainer**: `experiments/train_substrate_c6_e4_mdl_ibps.py`
- **Status**: dispatch-enabled; partial empirical anchor (5ep smoke + Tier C ablation per `feedback_mdl_ablation_tier_c_ibps1_landed_20260514`)

**HARD-EARNED PRESERVED**: A1/PR101 set + Rissanen 1978 MDL principle + Tishby-Zaslavsky IB framework + Catalog #270 dispatch optimization protocol Tier 1/2/3 wired

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "MDL × IB × Procedural-Synthesis substrate yields ΔS -0.030 to -0.080 vs A1" — **MEDIUM-RISK** — design memo §2 predicted band is wide (factor of 2.7); not validated empirically beyond 5ep smoke
- **CC-2** "Procedural decoder + per-pair patches is more bit-efficient than monolithic decoder + latents" — **MEDIUM-RISK** — Selfridge demon hierarchy + MoE assumed without contest-video empirical anchor; HNeRV (content-adaptive embeddings) is the EMPIRICAL counter-example proving content-adaptive >> procedural
- **CC-3** "Predicted post-campaign score band [0.11, 0.16]" — **HIGH-RISK** — first-principles MDL+IB lower bound but skips Pareto-frontier-consistency check (Dykstra-feasibility); analogous to NSCS06's symposium-#4 prediction failure mode

**Cargo-cult density:** 3 (MEDIUM-HIGH because CC-3 is HIGH-RISK)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — C6 has the right hard-earned foundations (Rissanen + Tishby) but the predicted band [0.11, 0.16] is NOT anchored to any Dykstra-feasibility check. Recommend: before next paid dispatch, compute Dykstra-feasibility intersection of MDL+IB constraint with contest rate budget; the predicted band must be REVISED if the feasibility region does not include [0.11, 0.16].

---

### TIER 3: deferred / scaffold-only (low urgency; representative sample)

#### 2.13 B1 — Wunderkind G1 family

- **Lanes**: `lane_b1_film_pose_x_magic_codec_a1_20260512` (L2) + sister L2 lanes (b1_nerv_enc_dec, b1_film_pose_x_hessian_block_fp_a1)
- **Status**: mostly dispatch_enabled or no-research-only; substrate-engineering family

**HARD-EARNED PRESERVED**: A1/PR101 set + FiLM conditioning canonical from PR #56

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (2 items):
- **CC-1** "FiLM pose conditioning improves PR101 score by replacing standard pose-as-latent input" — **MEDIUM-RISK** — assumed without empirical anchor; FiLM is empirically validated in PR #56 (masks codec) but extension to RGB renderer pose-conditioning is unanchored (echoes the Quantizr-mask-to-frames cargo-cult #7 from NSCS06)
- **CC-2** "Hessian-block-FP weight self-compression preserves PR101 distortion" — **LOW-RISK** — Selfcomp PR #56 empirical anchor; well-grounded

**Cargo-cult density:** 2 (LOW)
**Verdict:** HEALTHY — B1 family has well-anchored hard-earned base; the FiLM-pose-conditioning cargo-cult is real but LOW-RISK at the experimentation level.

---

#### 2.14 B2 / ATW Codec — Atick-Tishby-Wyner

- **Lane**: `lane_atw_codec_design_v1_20260515` (L1; not yet implemented)
- **Trainer**: `experiments/train_substrate_atw_codec_v1.py` (`_full_main` raises NotImplementedError)
- **Status**: research_only; design memo + scaffold landed 2026-05-15

**HARD-EARNED PRESERVED**: A1/PR101 set + ALL of Atick-Redlich + Tishby + Wyner-Ziv (cited correctly with paper URLs in design memo §1)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "Three-paper composition into ONE Lagrangian is novel and tractable" — **LOW-RISK** — design memo §1 acknowledges this is the novelty; the math is tractable; assumption is structural not empirical
- **CC-2** "Predicted [0.18, 0.21] frontier displacement" — **HIGH-RISK** — first-principles bound from grand reunion symposium Composite #1; same hand-waved-prediction-band class as NSCS06 symposium-#4
- **CC-3** "Wyner-Ziv gain estimate for dashcam + scorer is 30-50% conditional entropy reduction" — **HIGH-RISK** — explicitly acknowledged as hypothesis-not-measurement in design memo §1 ("currently a hypothesis, not a measured artifact")

**Cargo-cult density:** 3 (HIGH because 2 HIGH-RISK)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — ATW has the BEST hard-earned theoretical foundation in the entire staircase (3 Nobel-class papers cited) but the predicted band [0.18, 0.21] is unanchored. **The good news**: ATW's design memo §1 EXPLICITLY calls out CC-3 as hypothesis ("must not claim a 30-50% conditional entropy reduction until probe lands"). This is the post-2026-05-15 discipline working. Recommend: H(latent | scorer_class) probe BEFORE any paid Modal dispatch.

---

#### 2.15 F-asymptote / Time-Traveler L5 (Z6/Z7/Z8 predictive coding world models)

- **Lanes**: `lane_time_traveler_l5_macos_cpu_smoke_execution_20260513` (L2) + sister L1 design lanes
- **Trainer**: `experiments/train_substrate_time_traveler_l5_autonomy.py` (~63KB, large; carries full PAIR T directive)
- **Status**: dispatch-enabled in macOS CPU advisory; sub-0.193 [contest-CPU] anchor pipeline candidate

**HARD-EARNED PRESERVED**: A1/PR101 set + Rao-Ballard 1999 predictive coding hierarchy + Atick-Redlich + foveation-matched-to-ego-motion + differentiable world model + sub-100K params + Tikhonov regularization (5 first-principles design moves cited)

**CARGO-CULTED ELIGIBLE-FOR-CHALLENGE** (3 items):
- **CC-1** "Single-archive TT5L packet at 95-110 KB target size is achievable" — **MEDIUM-RISK** — target derived from 5-design-move composition; not empirically anchored to contest video
- **CC-2** "Five first-principles design moves compose additively for ΔS" — **HIGH-RISK** — Dykstra-feasibility says composition is at best subadditive in convex intersection regime; NSCS06 had similar "compose multiple ideas, get the sum" cargo-cult
- **CC-3** "macOS CPU advisory + Linux x86_64 GHA paired is sufficient compute envelope" — **LOW-RISK** — Catalog #192 + #197 enforce the advisory-vs-promotable contract correctly

**Cargo-cult density:** 3 (MEDIUM-HIGH because CC-2 is HIGH-RISK)
**Verdict:** AUDIT-PENDING-DEEPER-PROBE — Time-Traveler L5 has STRONG hard-earned foundations but the 5-move-composition assumption (CC-2) is the cargo-cult class NSCS06 hit. Recommend: Dykstra-feasibility intersection check for the 5-move composition BEFORE any paid dispatch beyond the existing macOS-CPU advisory anchor.

---

## Section 3 — Cross-substrate META-CARGO-CULT pattern analysis

Six cargo-cult patterns recur across the audited substrates (sourced from the symposium-#4-era design culture, not from individual substrate authors):

### META-CC-1: "Predicted ΔS band derived from first-principles without Dykstra-feasibility check"

**Affected substrates:** NSCS06 (FALSIFIED), C6 [0.11, 0.16], ATW [0.18, 0.21], Z4 [0.180, 0.188], Time-Traveler L5 [N/A explicit but implied], NSCS02 (implicit), D1 [0.181, 0.188] (FALSIFIED at 0.222)

**Pattern:** A substrate's design memo posits a predicted score band derived from first-principles bound (Shannon R(D) / Atick-Redlich / Wyner-Ziv / MDL) without ALSO computing the Dykstra-feasibility intersection of the substrate's constraints with the contest rate+distortion polytope. The bound is mathematically valid but the achievability is unverified.

**Empirical cost:** NSCS06 was 553× outside band. D1 was 1.18× outside band. The bug class costs ~$5-15 per falsification.

**Proposed STRICT preflight gate (Catalog #296):** `check_substrate_predicted_band_has_dykstra_feasibility_check` — refuses substrate design memos with a predicted ΔS band that don't cite an explicit Dykstra-feasibility intersection check (or carry a same-line `# PREDICTED_BAND_DYKSTRA_DEFERRED_OK:<rationale>` waiver pointing at the next-step probe).

---

### META-CC-2: "Signal-axis destruction is reversible at inflate time"

**Affected substrates:** NSCS06 (FALSIFIED via chroma destruction), NSCS02 (downsample-upsample structurally analogous), D1 (margin-map shrunk variant: 16× spatial downsample), partially Z3-G1 (empty hyperprior slots = capability destruction)

**Pattern:** A substrate destroys (or near-destroys) an information channel at compress time (chroma / spatial frequencies / per-pair hyperprior weights) and assumes inflate-time reconstruction (Y=R=G=B replication / bicubic upsample / uniform class prior) preserves enough signal for the scorer.

**Empirical cost:** Same as META-CC-1.

**Proposed STRICT preflight gate (Catalog #297):** `check_substrate_signal_axis_destruction_has_reversibility_probe` — refuses substrate trainers with hard-coded signal-axis reduction (chroma drop / spatial downsample > 2× / per-pair capability slot empty) WITHOUT an adjacent reversibility-probe smoke test that measures the inflate-time reconstruction loss against the contest scorer.

---

### META-CC-3: "Closed-form analytical allocator is sufficient (no gradient required)"

**Affected substrates:** NSCS06 (FALSIFIED), partially C6 (procedural decoder is closed-form-like)

**Pattern:** A substrate uses argmax/argmin/threshold-based bit allocation (no gradient signal) and assumes this captures enough information for medal-band score.

**Empirical cost:** ΔS impact band [+50, +100] (NSCS06 empirical).

**Mitigation:** This is a SUBCLASS of META-CC-2. The proposed Catalog #297 covers it if "capability destruction" is broadened to include "soft information destruction".

---

### META-CC-4: "Mask-paradigm primitive generalizes to frames"

**Affected substrates:** NSCS06 (Quantizr grayscale-LUT extension; FALSIFIED), B1 family (FiLM-pose conditioning extension)

**Pattern:** A substrate extends a primitive that was empirically validated at the SegNet-mask layer (3-5 class problem, lower entropy) to the full RGB frame layer (3×256-value problem, higher entropy) without empirical anchor at the higher-entropy task.

**Empirical cost:** ΔS impact band [+50, +80] (NSCS06 empirical).

**Mitigation:** Documented in CLAUDE.md HNeRV parity discipline lesson 5 ("Architecture must be the FULL renderer (RGB out), not a single-component slot"). A NEW STRICT preflight gate is not needed because the lesson is already a binding non-negotiable. Verification action: re-audit B1 family + any other substrate extending a mask-layer primitive to the frame layer.

---

### META-CC-5: "EMA decay 0.997 / canonical scorer-helper / canonical curriculum across all substrates"

**Affected substrates:** ALL audited substrates (every substrate inherits these as default)

**Pattern:** The canonical-helper default-adoption reflex documented in `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`. Each substrate gets the same EMA decay, scorer-helper routing, training curriculum without paired-comparison verification.

**Empirical cost:** Unknown; this IS what the post-2026-05-15 UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable is designed to extinct.

**Mitigation:** Already extincted by Catalog #290 (`check_substrate_design_memo_has_canonical_vs_unique_decision_section`). Verification action: confirm Catalog #290 fires on every Tier 1 + Tier 2 substrate in this audit's scope; backfill design memos where missing.

---

### META-CC-6: "Single-canary-first / smoke-passes-implies-substrate-works"

**Affected substrates:** sane_hnerv (5 failed first-anchor attempts on 2026-05-12), Z3-G1 (smoke PASSED but produced 0.19869 identical to baseline)

**Pattern:** Smoke green-up is treated as substrate validation; subsequent paid-full dispatch fires on smoke-passes signal alone.

**Empirical cost:** Z3-G1 wasted ~$5-10 on paid CUDA after smoke smoke-passed; sane_hnerv burned ~$0.50-2 across 5 failed attempts.

**Mitigation:** Already extincted by Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`) + Catalog #272 (`check_substrate_distinguishing_feature_integration_contract`). Verification action: confirm Catalog #272 fires on every substrate at L2+ promotion; backfill `distinguishing_feature_name` / `byte_mutation_smoke_passes` fields where missing (6 lanes flagged at Catalog #272 landing per 2026-05-15 audit).

---

## Section 4 — Per-tier recommended action sequence

### Tier 1 actions (dispatch-enabled / just-failed)

| Substrate | Action | Cost | Cargo-cult-priority |
|---|---|---|---|
| NSCS06 | Already FALSIFIED; symposium output enumerates Paths A/B/C/D/E/F; pick from symposium recommendations | $5-65 | n/a (reference) |
| A1 | HEALTHY; no action; CC-1 Ballé hyperprior gain is what Z3 is for | $0 | n/a |
| PR101 LC v2 | Apply predict-then-measure protocol to "enhanced curriculum" prediction BEFORE next dispatch | $0 design + $0 protocol | LOW |
| sane_hnerv | Paired-comparison smoke: canonical-helper vs custom scorer-loss routing | $5 Modal A100 | MEDIUM |

### Tier 2 actions (ready to dispatch on operator gate-flip)

| Substrate | Action | Cost | Cargo-cult-priority |
|---|---|---|---|
| NSCS01 | CLEARED; probe-disambiguator + paired-comparison smoke when dispatched | $0 audit + $5 smoke at dispatch | LOW |
| NSCS02 | predict-then-measure paired comparison @ 384×512 vs 192×256 vs 96×128 BEFORE any Modal dispatch | $5 paired CPU smoke | HIGH |
| NSCS03 | CLEARED; 3 cargo-cults LOW-MEDIUM and documented | $0 | LOW |
| D1 | Re-measure L2 OVERLAY's score-improvement (not just byte-mutation) before next dispatch | $5 byte-mutation extension | MEDIUM |
| D4 | H(latent | scorer_class) probe shared with ATW | $0 analytical or $3 CPU | MEDIUM |
| Z3 v2 + Z3-G1 | Z3-G1 already research_only; for Z3 v2: revise predicted ΔS band DOWN per Z1 ablation | $0 | HIGH (already-falsified) |
| Z4 | ATW canonical probe (κ_IB=0, λ_WZ=1, λ_pixel=0) disambiguates Z4 as side-effect | $0 (shared with ATW) | MEDIUM |
| C6 | Dykstra-feasibility intersection check for MDL+IB constraint BEFORE next paid dispatch | $0 analytical | HIGH |

### Tier 3 actions (deferred / scaffold-only)

| Substrate | Action | Cost | Cargo-cult-priority |
|---|---|---|---|
| B1 family | Re-audit FiLM-pose extension per META-CC-4 (mask-to-frame generalization) | $0 design | MEDIUM |
| ATW | H(latent | scorer_class) probe BEFORE paid dispatch (design memo §1 already calls this out) | $0 analytical or $3 CPU | HIGH |
| Time-Traveler L5 | Dykstra-feasibility check for 5-move composition BEFORE paid dispatch beyond macOS-CPU advisory | $0 analytical | HIGH |

---

## Section 5 — Op-routables (ranked by EV/$ × NSCS06-class-failure-prevention)

| Rank | Op-routable | Cost | Predicted prevention value | Dep |
|---|---|---|---|---|
| 1 | Catalog #296 STRICT preflight gate `check_substrate_predicted_band_has_dykstra_feasibility_check` (META-CC-1) | $0 dev | Prevents NSCS06-class falsification for next 6+ substrates (NSCS02 / C6 / ATW / Z4 / Time-Traveler L5) | none |
| 2 | Catalog #297 STRICT preflight gate `check_substrate_signal_axis_destruction_has_reversibility_probe` (META-CC-2) | $0 dev | Prevents NSCS02 + future signal-axis-destruction cargo-cults | none |
| 3 | Re-audit B1 family per META-CC-4 (mask-to-frame extension) | $0 | Prevents B1 dispatch on unverified extension | none |
| 4 | Re-audit ALL Tier 2 substrate design memos for canonical-vs-unique decision section per Catalog #290 | $0 | Confirms META-CC-5 mitigation is structural | none |
| 5 | Shared H(latent | scorer_class) probe for D4 + ATW + Z4 (single probe; canonical disambiguator) | $3-5 CPU | Anchors 3 substrates' core hypothesis simultaneously | none |
| 6 | Dykstra-feasibility check helper `tools/check_substrate_dykstra_feasibility.py` (cathedral autopilot consumer) | $0 dev | Recurring use across all future substrate predicted bands | none |
| 7 | predict-then-measure protocol enforced via per-substrate empirical-anchor field in lane registry | $0 dev | Structurally prevents next first-principles-band falsification | none |
| 8 | sane_hnerv paired-comparison smoke: canonical-helper vs custom routing | $5 Modal A100 | Disambiguates the canary-failure pattern | none |
| 9 | NSCS02 paired downsample-ratio smoke (384×512 vs 192×256 vs 96×128) | $5 CPU smoke | Disambiguates signal-axis-destruction risk | none |
| 10 | Run Catalog #272 audit on every Tier 2 substrate; backfill distinguishing_feature fields | $0 | Closes META-CC-6 at the L2-promotion surface | none |

---

## Section 6 — HARD-EARNED-vs-CARGO-CULTED MASTER INVENTORY (cross-substrate)

### HARD-EARNED canonical inventory (cite source; PRESERVE across all substrates)

| # | Assumption | Source | Substrates carrying |
|---|---|---|---|
| 1 | eval_roundtrip=True | CLAUDE.md non-negotiable; Catalog #5 | ALL |
| 2 | EMA decay 0.997 + shadow at inference | CLAUDE.md non-negotiable; Quantizr empirical | ALL |
| 3 | MPS auth eval is NOISE; CUDA-only for authoritative | CLAUDE.md non-negotiable (23× PoseNet drift) | ALL |
| 4 | strict-scorer-rule (no scorer at inflate) | CLAUDE.md non-negotiable | ALL |
| 5 | Differentiable scorer-preprocess + YUV6 patching | CLAUDE.md non-negotiable; Catalog #187 | ALL |
| 6 | Single archive.zip per submission | Contest rule | ALL |
| 7 | 30-min eval budget T4 / 4-CPU 16GB CPU | Contest rule | ALL |
| 8 | SegNet last-frame slice + PoseNet 6-dim ego-motion | upstream/modules.py inspection | ALL |
| 9 | Camera resolution (1164, 874) | Contest dataset | ALL |
| 10 | Rate term = `25 * archive_bytes / 37_545_489` | Contest formula | ALL |
| 11 | Pose contribution = `sqrt(10 * d_pose)` | Contest scoring rule | ALL |
| 12 | Catalog #117/#157/#174 commit serializer + content-sha | CLAUDE.md non-negotiable (commit-swap incidents) | ALL |
| 13 | Apples-to-apples axis tag every score | CLAUDE.md non-negotiable (Z3-G1 phantom score) | ALL |
| 14 | Public PR intake clones pristine; recovery_metadata append-only | Catalogs #109 / #110 | ALL |
| 15 | Inflate ≤100 LOC + ≤2 deps (substrate_engineering exception) | HNeRV parity L4 | ALL |
| 16 | Monolithic 0.bin archive grammar (substrate_engineering exception) | HNeRV parity L3 | ALL |
| 17 | brotli on weight streams preserves zero distortion | Lossless transform property | ALL |
| 18 | fp16 latent quantization preserves PR101-class score | Empirical from A1/PR101 anchor | ALL |
| 19 | SegNet uses RGB-distinguishing class cues (chroma matters) | NSCS06 empirical falsification 2026-05-16 | ALL going forward |
| 20 | Dykstra-feasibility intersection defines achievable region | Dykstra 1983 + A1 empirical | ALL going forward |
| 21 | Shannon R(D) lower bounds rate at fixed distortion | Shannon 1948 | ALL going forward |
| 22 | Within-class refinement on A1 will not yield medal-band | Z1 ablation 2026-05-14 | ALL going forward |
| 23 | Class-shift moves required for further floor-drop | Z1 EUREKA | ALL going forward |

### CARGO-CULTED canonical inventory (eligible for challenge; with affected substrates)

| # | Assumption | Risk | Affected substrates | Mitigation |
|---|---|---|---|---|
| 1 | Predicted ΔS band from first-principles without Dykstra-check | HIGH (META-CC-1) | NSCS06 ✓F, D1 ✓F, NSCS02, C6, ATW, Z4, TTL5 | Catalog #296 (proposed) |
| 2 | Signal-axis destruction reversible at inflate | HIGH (META-CC-2) | NSCS06 ✓F, NSCS02, D1 | Catalog #297 (proposed) |
| 3 | Closed-form allocator sufficient (no gradient) | HIGH | NSCS06 ✓F | Catalog #297 subclass |
| 4 | Mask-paradigm extends to frames | HIGH (META-CC-4) | NSCS06 ✓F, B1 family | HNeRV parity L5 (already binding) |
| 5 | Canonical EMA decay / scorer routing for all substrates | MEDIUM (META-CC-5) | ALL | Catalog #290 (existing) |
| 6 | Smoke-passes implies substrate-works | MEDIUM (META-CC-6) | sane_hnerv, Z3-G1 ✓F | Catalog #272 (existing) |
| 7 | Symposium-#4 prediction band [0.10, 0.20] from rate-only bound | HIGH | NSCS06 ✓F | abandoned per symposium output |
| 8 | Hyperprior empty-slot archive ships substrate effect | FALSIFIED | Z3-G1 ✓F | Catalog #266/#267 (existing) |
| 9 | SegNet uniform-class fallback is acceptable on exception | FALSIFIED | Z3-G1 ✓F | Catalog #267 (existing) |
| 10 | λ_R linear warmup is canonical Ballé schedule | LOW | NSCS03 | unmitigated; LOW-RISK |
| 11 | 5 sub-net Ballé topology is optimal for our distortion budget | MEDIUM | NSCS03 | unmitigated; documented |
| 12 | Asymmetric quantization (frame_0 4-bit / frame_1 8-bit) | MEDIUM | NSCS01 | probe-disambiguator already wired |
| 13 | Cooperative-receiver pure λ_pixel=0 beats baseline | MEDIUM | Z4, ATW | shared ATW probe |
| 14 | Wyner-Ziv gap H(Y|S) - H(Y|X,S) is large | MEDIUM | D4, ATW | shared probe |
| 15 | MDL × IB × Procedural-Synthesis composes additively | MEDIUM | C6 | Dykstra-feasibility check |
| 16 | 5-design-move composition is additive | HIGH | TTL5 | Dykstra-feasibility check |
| 17 | Enhanced curriculum > standard curriculum | MEDIUM | PR101 LC v2 | predict-then-measure |
| 18 | Sane HNeRV reproduces 0.193-0.195 [contest-CPU] | MEDIUM | sane_hnerv | 5 failed attempts is empirical signal |

Legend: ✓F = empirically falsified

---

## Section 7 — STRICT preflight gate proposal

Two new gates proposed to extinct the META-CARGO-CULT patterns Section 3 surfaced:

### Catalog #296 (PROPOSED) — `check_substrate_predicted_band_has_dykstra_feasibility_check`

**Bug class:** Substrate design memos posit a predicted ΔS / score band derived from first-principles bound (Shannon R(D) / Atick-Redlich / Wyner-Ziv / MDL / 5-move composition) without ALSO computing the Dykstra-feasibility intersection of the substrate's constraints with the contest rate+distortion polytope.

**Empirical anchors:** NSCS06 (553× outside band), D1 (1.18× outside band), Z3-G1 (predicted ΔS not measurable; class-saturation makes prediction vacuous).

**Acceptance:** substrate design memo (`.omx/research/<substrate>_design_<date>.md`) with date >= 2026-05-16 that contains a `predicted_*_band` OR `predicted_ΔS` OR `predicted_score` field MUST also contain a Dykstra-feasibility section header (canonical: `## Dykstra-feasibility intersection check`) OR carry a same-line `# PREDICTED_BAND_DYKSTRA_DEFERRED_OK:<rationale>` waiver pointing at the next-step probe that will compute the intersection.

**Sister of:** Catalog #290 (canonical-vs-unique decision per layer), Catalog #229 (premise verification), Catalog #292 (per-deliberation assumption surfacing). Together they close the "design memo discipline" surface.

**Wire-in:** WARN-ONLY initially per CLAUDE.md "Strict-flip atomicity rule"; strict-flip after operator-routed audit + backfill of existing predicted-band design memos (estimated ~8 currently-in-flight).

**Tests:** ~15 tests covering positive (design memo with predicted band, no Dykstra section flagged), negative (no predicted band → out of scope; with Dykstra section → accepted), waiver (rationale accepted, placeholder rejected), strict raise, live-repo regression guard.

---

### Catalog #297 (PROPOSED) — `check_substrate_signal_axis_destruction_has_reversibility_probe`

**Bug class:** Substrate trainers / inflate runtimes destroy (or near-destroy) an information channel at compress time (chroma drop / spatial downsample > 2× / per-pair capability slot empty / argmax-only-instead-of-soft) and assume inflate-time reconstruction preserves enough signal for the scorer WITHOUT an adjacent reversibility-probe smoke test.

**Empirical anchors:** NSCS06 Y=R=G=B replication (seg=64.59), NSCS06 `np.roll` global translation (pose=149.03), Z3-G1 empty hyperprior_weights_int8 slots (silent baseline reproduction at 0.19869).

**Acceptance:** trainer file or inflate runtime under `experiments/train_substrate_*.py` or `src/tac/substrates/*/inflate.py` containing one of the canonical destruction-pattern tokens (`np.repeat(.*?,.*?3.*?axis=2)` for grayscale-to-RGB; `np.roll(.*?,.*?shift=` for global translation; `F.interpolate.*?scale_factor=[4-9]` for >4× upsample; `b""` literal in archive header slot context) MUST have an adjacent reference to a reversibility-probe (`tools/probe_<substrate>_reversibility.py` OR `# REVERSIBILITY_PROBE:<path>` OR `# SIGNAL_AXIS_DESTRUCTION_ACCEPTED_RISK_OK:<rationale>` waiver).

**Sister of:** Catalog #220 (substrate L1+ byte-addition operational mechanism), Catalog #272 (distinguishing feature integration contract), Catalog #139 (no-op detector packet compiler), Catalog #105 (no-op provenance). Together they close the "compress-destroys-signal-without-inflate-reconstruction" surface.

**Wire-in:** WARN-ONLY initially per CLAUDE.md "Strict-flip atomicity rule"; live count at landing estimated: 3 (NSCS06 ✓F, NSCS02, D1 shrunk margin map). Strict-flip after substrate-engineering waivers are documented for D1 + NSCS02 backfill is decided.

**Tests:** ~18 tests covering each canonical destruction pattern + reversibility-probe acceptance + waiver semantics + strict raise + live-repo regression guard.

---

### Catalog #298 (OPTIONAL; LOWER PRIORITY) — `check_substrate_empirical_anchor_after_smoke_before_paid_full`

**Bug class:** Substrate trainers fire paid Modal / Lightning / Vast.ai full dispatch based on smoke-passes-only signal without an intermediate empirical-anchor publication (a smoke-output score band that the operator can compare against the design memo's predicted band).

**Empirical anchors:** Z3-G1 smoke PASSED then paid CUDA dispatched WITHOUT operator review of 0.19869 = baseline match.

**Acceptance:** every paid Modal dispatch invocation MUST be preceded by a smoke-anchor row in `.omx/state/cost_band_posterior.jsonl` whose `predicted_score_band` field overlaps the smoke's empirical score.

**Status:** PROPOSED for future audit; current Catalog #270 (umbrella dispatch optimization protocol) + Catalog #271 (codex pre-dispatch review automation) partially cover this; full structural extinction may require this additional gate.

---

## Final summary

- **(a) Total substrates audited:** 15 representative substrates spanning 3 tiers (Tier 1: 4; Tier 2: 8; Tier 3: 3). The lane registry contains 139 L1+ substrate lanes; this audit prioritized dispatch-readiness urgency.
- **(b) Per-tier substrate count + cargo-cult-density distribution:**
  - Tier 1: 4 substrates; mean density 4.0 (high due to NSCS06 7); HEALTHY/CANONICAL = 3, AUDIT-PENDING = 1 (sane_hnerv)
  - Tier 2: 8 substrates; mean density 3.25; HEALTHY = 2 (NSCS01, NSCS03), AUDIT-PENDING = 5 (NSCS02, D1, D4, Z4, C6), HIGH-RISK = 1 (Z3-G1; already research_only)
  - Tier 3: 3 substrates sampled; mean density 2.7; HEALTHY = 1 (B1), AUDIT-PENDING = 2 (ATW, TTL5)
- **(c) Top-5 HIGH-RISK substrates ranked by cargo-cult-density × dispatch-readiness urgency:**
  1. **NSCS02** — dispatch-ready (research_only=true but `_full_main` presumed implemented) + 4 cargo-cults + structurally analogous to NSCS06 (META-CC-2 + signal-axis destruction)
  2. **C6 MDL-IBPS** — dispatch-enabled + 3 cargo-cults (1 HIGH-RISK predicted band [0.11, 0.16] without Dykstra check)
  3. **ATW Codec V1** — design-complete; predicted band [0.18, 0.21] without probe
  4. **Time-Traveler L5** — dispatch-active (macOS CPU advisory) + 5-move composition cargo-cult
  5. **sane_hnerv** — dispatch-enabled + 5 failed first-anchor attempts (META-CC-6)
- **(d) META-CARGO-CULT patterns surfaced (cross-substrate):** 6 patterns; 2 NEW gates proposed (Catalog #296 + #297) to extinct the 2 most damaging structurally; 4 patterns already mitigated by existing gates (#272, #290, HNeRV parity L5, #267).
- **(e) Recommended STRICT preflight gates:** Catalog #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`) + Catalog #297 (`check_substrate_signal_axis_destruction_has_reversibility_probe`); both WARN-ONLY at landing per "Strict-flip atomicity rule"; both have ≤5 live violations.
- **(f) Op-routables ranked:** 10 total, ranked by EV/$. Top-3 are all $0 cost: (1) Catalog #296, (2) Catalog #297, (3) B1 family re-audit per META-CC-4.

**Net assessment:** the staircase has 1 ALREADY-FALSIFIED (NSCS06) + 5 HIGH-RISK substrates ripe for cargo-cult-unwinding BEFORE next paid dispatch. The post-2026-05-15 META-ASSUMPTION discipline is WORKING for NEW substrates (NSCS01 + NSCS03 + ATW design memos all carry assumption-classification or hypothesis-not-measurement disclaimers). The 5 HIGH-RISK substrates predate the discipline and need backfill. Total estimated cost to unwind cargo-cults across all 5: $15-30 in shared probes + $0 in design memo backfill. Total estimated savings: prevention of ~$50-150 in NSCS06-class paid-dispatch falsifications.

**Compliance with this audit's own discipline (per Catalog #292 self-application):** every per-substrate audit row above states the operating-within assumption explicitly (HARD-EARNED PRESERVED + CARGO-CULTED ELIGIBLE) per Catalog #292's per-round explicit-assumption-statement requirement.

---

## Cross-references

- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` — canonical classification framework
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — sister 18-shared-assumptions matrix (this audit's per-substrate complement)
- `feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md` — operator retrospective that motivated the entire discipline
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — UNIQUE-AND-COMPLETE-PER-METHOD operating mode
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — canonical-vs-unique decision framework
- `grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` — canonical NSCS06 cargo-cult decomposition template
- `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515.md` — Catalog #291/#292 landing memo
- `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md` — class-shift discipline (orthogonal but informing)
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiable + Catalog #291 (this audit IS the cadence event)
- CLAUDE.md "Council conduct" sextet pact + Assumption-Adversary seat
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — every verdict ADDS reactivation criteria; ZERO substrates killed
- CLAUDE.md "Apples-to-apples evidence discipline" — every score claim axis-tagged
