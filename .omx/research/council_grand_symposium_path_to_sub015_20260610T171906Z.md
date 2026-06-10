---
council_tier: T4
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, PR95Author, Quantizr, Hotz, Selfcomp, MacKay, Balle, Carmack, Boyd, Tao, Mallat, Atick, Redlich, Tishby, Wyner, Hinton, Schmidhuber, Karpathy, Hassabis, Filler, vdOord, JackFromSkunkworks, Rao, Ballard, Zaslavsky, TimeTravelerProtege, TimeTraveler, Rudin_Grand, Daubechies_Grand, HaoChen_NeRV, Shrivastava_INR, Gwilliam_RNeRV, Kang_SNeRV, Bull_HiNeRV]
council_quorum_met: true
council_roster_validation: "complete=True (T4: all 4 co-leads present Shannon/Dykstra/Rudin/Daubechies; full inner council 14; all topic-relevant grand-council specialists summoned; tac.canonical_council_roster.validate_council_dispatch_roster -> complete)"
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
council_topic: "The pattern-behind-patterns of the closed spec; IS sub-0.15 reachable; the directional verdict + ranked exact-score-movers"
council_dissent:
  - member: Contrarian
    verbatim: "Three of the top movers (PTNC, lever-C joint carrier, #63 conditioning) are NEEDS-CAMPAIGN with $0 smokes gating them; none is a proven exact-row mover. The ONLY proven exact-axis win in the entire landscape remains R1+R2 (-0.00092, banked-class), and it is a borrowed-substrate absorb-recode that FAILS the Innovation Gate. The symposium must not let an elegant pattern (the polytope-quotient invariant) substitute for the fact that we have shipped nothing original below 0.191 and every score-native frame1 path has walled twice. Rank the $0 SMOKES first; they are the only thing that converts a beautiful invariant into a row."
  - member: Yousfi
    verbatim: "d_seg is the per-pixel argmax-flip RATE of a frozen EfficientNet-B2 — it IS inverse steganalysis on the partition. The whole field clusters at 0.19 because everyone optimizes a smooth recon surrogate of a non-smooth detector functional. The #63 decisive test (argmax-CE vs KL-T2 vs margin-hinge) is the single most important UNRUN experiment we have: it tells us whether the wall is FUNDAMENTAL (loss-invariant frozen-SegNet-argmax conditioning) or a WRONG-LOSS artifact. Run it before any carrier campaign — it re-ranks everything below it."
  - member: Hotz
    verbatim: "The frontier is a 177KB neural decoder that is LOSSLESS-EXHAUSTED. We keep building third and fourth frame1 carriers that wall on the same RGB->frozen-SegNet ill-conditioning. Stop. Either (a) find the cheap door the floor report named (a smaller amortizer) by attacking the EXISTING decoder's weights with a score-aware requant bank, or (b) prove the #63 conditioning wall is loss-not-fundamental and THEN build one carrier with the winning loss. No fifth carrier."
  - member: Daubechies
    verbatim: "The partition is 0.687% boundary, 35 regions/frame, geometry-not-labels (3KB labels vs 250KB geometry). That is a multi-scale object: coarse regions are certain (large margin), fine boundary cracks are fragile (margin<0.5). The right carrier is hierarchical-coarse-gates-fine on the RAG, and the MWCC margin cost-map IS the wavelet-thresholding of the boundary. But the floor report MEASURED partition-direct loses to amortization on RATE (0.169 vs 0.118) — so MWCC is a RESIDUAL-repair coder on an amortized base, never a standalone carrier. I will not endorse a standalone contour carrier."
  - member: Tishby
    verbatim: "evaluate.py is an information bottleneck: it squeezes the 874x1164x3 frame through a low-dim relevant variable (the 5-class partition + 6 pose scalars). The minimal sufficient statistic of {frame | scored output} is exactly what S_floor measures. The reason every path walls the same way is that the RGB->relevant-variable map is the encoder of an IB, and a smooth generator cannot represent the sharp decision boundary the bottleneck induces. This is not a contest quirk; it is the IB compression-vs-relevance tradeoff made literal."
  - member: PR95Author
    verbatim: "The 0.19 cluster I sit on did NOT use argmax-CE on a cheap renderer. It used KL-T=2.0 SegNet-logit distillation through a FULL-resolution RGB renderer with 8-stage curriculum + Muon. The lesson the lab keeps half-learning: the full renderer is not optional for d_seg, and the loss must be soft-logit not hard-argmax. #63 tests exactly the loss half on a CHEAP renderer; if it FAILS, the renderer-capacity half is the binding constraint and the only score-native door is a full-fidelity per-pair conv carrier competing on TOTAL bytes (lever C / AFSR-1 fresh-init), not a cheap frame1."
  - member: Carmack
    verbatim: "The runtime-less Zig rasterizer is the right end-state ONLY after the representation settles. Building it now is premature — we do not yet know if the carrier is contour+luma-opcode (grammar interpreter) or a conv decoder. Defer (d) until (c) settles. But keep tac-boundary-decode parity-gated so the bake-off is one commit away."
council_assumption_adversary_verdict:
  - assumption: "Storing the SegNet argmax partition DIRECTLY (the closed-spec carrier premise) is the rate floor / 'far below a full RGB renderer'."
    classification: CARGO-CULTED
    rationale: "MEASURED-FALSE (information_theoretic_floor_report_v1, F1/F3): the optimal temporal-context coder of the 600 partitions = 253,413 B = rate 0.169, ABOVE the amortized decoder's 0.118 by 0.052. The closed-spec memo's elegant 'store the scored object directly' framing inherited the assumption that the partition's low PER-REGION entropy implies low TOTAL entropy; it does not (21,304 regions over 600 temporally-varying frames). The right data structure (RAG/contour) for the SOLVE is HARD-EARNED; the standalone-storage RATE-WIN is CARGO-CULTED and falsified."
    empirical_verification_status: VERIFIED_VIA_SOURCE
  - assumption: "The #62 d_seg wall (cheap RGB renderer stalls at flat-frame floor) is FUNDAMENTAL to the small-renderer family."
    classification: UNVERIFIED-FRAMING
    rationale: "#62 used argmax-CE ONLY. The boundary-solver (#55) flips argmax in closed form via the margin-polytope gradient; PR95 drove d_seg down via KL-T=2.0 soft-logit distillation. The #63 decisive test (argmax-CE vs KL-T2 vs margin-hinge, matched arch/bytes) is DESIGNED but UNRUN. Asserting 'fundamental' before running #63 is operating within an unverified framing. This GATES the directional verdict (Catalog #363): the 'defer score-native frame1' verdict is PROVISIONAL until #63 resolves whether the wall is loss-invariant or wrong-loss."
    empirical_verification_status: ASSUMED_AWAITING_VERIFICATION
  - assumption: "Pose is NOT a floor driver / pose is cheap (~1.5KB output entropy)."
    classification: HARD-EARNED
    rationale: "MEASURED (floor report F2, P6 RESOLVED): 600x6 temporal-delta pose-output entropy = 1,557 B at frontier operating point. BUT (Assumption-Adversary nuance) this is the pose-OUTPUT floor; the pose CARRIER blocker (#57) is that frame1 must carry BOTH d_seg AND pose-relevant luma, and the score-native palette/argmax frame1 is pose-BLIND (d_pose 12.14 alone). The 1.5KB is achievable only if a carrier exists that lands the partition AND holds the luma tube — which is the open lever C. Output-entropy cheapness != carrier feasibility."
    empirical_verification_status: VERIFIED_VIA_SOURCE
  - assumption: "S_floor = 0.118 is the binding wall and sub-0.118 needs a smaller amortizer (Kolmogorov-open)."
    classification: HARD-EARNED
    rationale: "DERIVED+MEASURED (floor report F4/F5/F6): S_floor=25*177169/D=0.11797 (frontier bytes at zero distortion); sub-0.15 is a DISTORTION threshold at constant bytes (already 0.118<0.15 IF d_seg->0 AND d_pose->0); sub-0.118 requires a provably smaller cell-lander, no nontrivial lower bound exists (K uncomputable). The re-rank (distortion-before-rate to reach 0.118, THEN rate for sub-0.118) is sound."
    empirical_verification_status: VERIFIED_VIA_SOURCE
  - assumption: "The waterfilling allocator on the EXISTING frontier base has actionable seg-correction input."
    classification: CARGO-CULTED
    rationale: "MEASURED (deferral ledger #54, E3): on the frontier base every seg-correction component is UNDER-WATER (1.525 B/flip position-only floor > 1.27 B/flip break-even; 95% single-pixel scattered flips, GT-snap net -536). The allocator input is EMPTY on the frontier. It becomes non-empty ONLY on a base with a contiguous/repairable residual (the lever-B generator base, 74% contiguous). The allocator is a SOLVE on the RIGHT base, not the frontier."
    empirical_verification_status: VERIFIED_VIA_SOURCE
council_decisions_recorded:
  - "op-routable #1 (TOP, $0): RUN the #63 d_seg-loss conditioning decisive test (argmax-CE vs KL-T2 vs margin-hinge, matched conv_pair_decoder). It is the single highest-information UNRUN experiment; it gates the entire directional verdict per Catalog #363."
  - "op-routable #2 (TOP, $0 smoke -> campaign): PTNC Jacobian-projected pose carrier $0 MLX smoke (the live prize, #57 reactivation #1) — only if #63 reopens the cheap-renderer family OR as the pose half of a full-fidelity carrier."
  - "op-routable #3 (READY-NOW, ~$0.3, BANK-class): R1+R2 lossless recode materializer -0.00092 exact (the only proven exact-axis win; flag DEFENSIVE BANK, fails Innovation Gate)."
  - "op-routable #4 (DESIGN-then-build, $0): score-aware decoder-WEIGHT requant bank (the #69-class lurking lever) — the door the floor report named (smaller amortizer) attacked on the EXISTING 177KB decoder weights, not a new frame1 carrier."
  - "op-routable #5 ($0): MWCC margin-weighted contour residual-repair coder smoke on the lever-B contiguous-residual base (NOT standalone; clears the >=5% / <1.27 B/flip gate the STC DEFER named)."
  - "Catalog #307 audit CLEAN: every DEFER (lever B/C frame1 carriers, AFSR-1, lever G, STC-clean-source) is IMPLEMENTATION-scoped with pinned reactivation criteria; NO premature paradigm kill. The score-native frame1 carrier is DEFERRED-pending-#63, reactivatable."
related_deliberation_ids: []
---

# T4 GRAND COUNCIL SYMPOSIUM — the pattern-behind-patterns + the path to sub-0.15

**UTC 2026-06-10T171906Z · Task #70 · facilitator subagent `council_symposium_sub015_20260609`.**
Authority of every number cited: the source verdict's own grade (`[contest-CPU]` for the frontier pointer
`0.19109982419209975`, sha `b46897267ded…`, 177,169 B; `[macOS-CPU advisory]` / `[local CPU-torch advisory]`
for the floor + carrier measurements — all recomputed-from-components on the EXACT frozen scorers, GT via
`frame_utils.yuv420_to_rgb`, NO MPS). `promotable=false`, `score_claim=false`, `$0` spend, no dispatch fired.
This memo is an ADVISORY directional verdict that ROUTES to exact-row actions; it is not itself a score claim.

---

## 1. THE PATTERN-BEHIND-PATTERNS — the deep invariant (one paragraph)

**Every path walls the same way because the contest is a single information bottleneck whose relevant
variable is a NON-SMOOTH, LOW-DIMENSIONAL QUOTIENT of the high-dimensional frame, and we keep optimizing a
smooth high-dimensional surrogate of it.** Concretely: `evaluate.py ∘ modules.py` maps each 874×1164×3 frame
to exactly two scored objects — the SegNet 5-class **argmax PARTITION** of frame1 (a *combinatorial set
functional*: `d_seg` = per-pixel argmax-flip RATE, piecewise-constant in pixels, gradient ZERO almost
everywhere with deltas only at the argmax boundary) and the PoseNet **6-of-12 output** of both frames (a
*smooth low-dim regression*, globally pooled before the √). The optimal archive is the shortest program
landing in the **argmax-equivalence CELL** `{F : argmax S(F) = L*}` — a polytope of linear inequalities
(§4 of the closed-spec memo) — intersected with the pose tube. The ONE deep invariant that explains the
0.19 field-cluster, our 0.0075 generator floor (lever B), the #57/#61 pose antagonism, the #62 d_seg-stall
at the flat-frame floor, the #64 lossless exhaustion, and the rate-dominated S_floor=0.118 JOINTLY is this:
**the scored functional factors through a low-dim non-smooth quotient (the partition + 6 pose scalars), and
(a) recon-MSE / smooth generators optimize the WRONG (smooth, high-dim) surrogate that over-smooths the 1-D
boundary that is the entire d_seg signal; (b) the only carrier MEASURED to land the cell cheaply is an
AMORTIZED full-RGB decoder (177 KB), because the partition's TOTAL geometry entropy (253 KB direct) exceeds
what amortization buys (162 KB), AND a cheap frame1 cannot simultaneously be a sharp partition (d_seg) and a
luma tube (d_pose) — frame1's DUAL constraint.** The walls are not independent failures; they are five faces
of one invariant: *we have been solving a combinatorial set-functional + a coupled dual-fidelity frame
problem with smooth high-dimensional generative tools.* The leverage point the invariant reveals: **the
gradient that moves d_seg lives in the SegNet LOGIT/margin space, not the RGB-recon space** — and whether a
*cheap* renderer's representable manifold intersects that margin-polytope under a *better-conditioned loss*
(soft-KL / margin-hinge, not argmax-CE) is the single UNRESOLVED hinge (#63, DESIGNED-not-run). If it does,
the cheap-carrier family reopens and sub-0.15 is reachable by distortion-closure at constant bytes; if it
does NOT, the wall is the renderer-capacity half and the only door is a full-fidelity smaller amortizer.

## 2. THE DIRECTIONAL VERDICT (PROVISIONAL per Catalog #363, gated on #63)

**IS sub-0.15 reachable?** — **YES, information-theoretically, by distortion-closure at constant bytes**
(DERIVED, floor report F5): the frontier's 177,169 B already scores **0.11797 < 0.15** at d_seg=d_pose=0.
The 0.191→0.118 gap is ENTIRELY the recoverable 0.073 seg+pose distortion residual, NOT bytes. **The rate
attack is only required BELOW 0.118.** Whether sub-0.15 is reachable by a *realizable program* hinges on
the #63 verdict: closing the seg residual on the existing 177 KB decoder (zero-byte corrections / score-aware
requant / cheap-carrier-with-right-loss) is the proven-arithmetic path to 0.118; sub-0.118 needs a provably
smaller amortizer (Kolmogorov-open, no proven wall).

**The near-term arc (the program's directional commitment):**
1. **RESOLVE the hinge first ($0).** Run #63. It re-ranks everything below it and tells us which half of the
   invariant (loss-conditioning vs renderer-capacity) is binding. This is NON-NEGOTIABLE before any carrier
   campaign — running a campaign on the wrong half is the lab's recurring waste.
2. **DEFER the score-native CHEAP-frame1 carrier** (per #307, reactivatable on a #63-PASS) — it has walled
   across coordinate-INR (#57/#61) AND conv (#62) families; do NOT iterate a third cheap frame1.
3. **PURSUE the distortion-closure path to 0.118** on the EXISTING frontier bytes: the #69-class score-aware
   decoder-WEIGHT requant bank (attack the 162 KB decoder weights with the measured margin-polytope as the
   quantization importance, not a new carrier) + PTNC for the pose half + MWCC residual-repair only on a
   repairable base. This is the **frontier_breaking** spine.
4. **BANK the proven lossless win** (R1+R2, −0.00092, DEFENSIVE only — fails the Innovation Gate; report the
   2 operator dispositions, do not pause).
5. **The runtime-less Zig / grammar-interpreter end-state (lever A/(d))** waits until the carrier
   representation settles (Carmack's deferral).

**Mission contribution: frontier_breaking.** The leverage point (move d_seg in margin/logit space, attack
the existing amortizer's weights for the smaller-program door) is a genuine class-shift direction, NOT
apparatus maintenance — but it is GATED on the $0 #63 disambiguator, so the verdict is PROCEED_WITH_REVISIONS.

## 3. TOP-3 OP-ROUTABLE EXACT-SCORE-MOVERS (ranked; full list in frontmatter `council_decisions_recorded`)

| # | action | predicted ΔS | gate | axis | original-vs-bank |
|---|---|---|---|---|---|
| **1** | **RUN #63 d_seg-loss conditioning decisive test** (argmax-CE vs KL-T2 vs margin-hinge, matched conv_pair_decoder; the DESIGN is pre-registered) | **gating, not direct** — a PASS reopens the cheap-carrier family worth the full 0.073 distortion headroom (→0.118); a FAIL redirects to full-fidelity/requant. Information value is maximal. | **$0** local CPU-torch smoke, NO dispatch | seg (conditioning) | **ORIGINAL** (the publishable finding either branch) |
| **2** | **PTNC ($0 MLX smoke → campaign)** — Jacobian-projected pose carrier realizing the ~1.5 KB pose floor; the #57 live-prize reactivation | conditional: closes the pose 0.017 term (9% of headroom) at ~1.5 KB; composes with a seg carrier toward sub-0.15 | **$0 MLX smoke** first; campaign only on smoke-pass + #63 favorable | pose | **ORIGINAL** (measured-Jacobian IDSE; no competitor has the frozen oracle's Jacobian) |
| **3** | **R1+R2 lossless recode materializer (READY-NOW)** | **−0.00092 exact** (177,169→~177,114 B; the only proven exact-axis win) | **~$0.3** single paired CPU+CUDA replay | rate | **BANK** (PR#112 absorb-recode; fails Innovation Gate — defensive hold, NOT the submission) |

(#4 score-aware decoder-weight requant bank and #5 MWCC residual-repair smoke are the next two $0/design movers;
both attack the EXISTING frontier / a repairable base rather than a new cheap frame1 carrier.)

## 4. RECURSIVE SELF-REFLECTION (Catalog #363) — verification-status gating

Round-1 surfaced 5 load-bearing assumptions (frontmatter `council_assumption_adversary_verdict`). Round-2
re-classification: **two are VERIFIED_VIA_SOURCE** (S_floor=0.118 rate-dominance; pose-output 1.5 KB),
**two are CARGO-CULTED-and-MEASURED-FALSE** (partition-direct as rate floor; frontier-base allocator has
input), **one is ASSUMED_AWAITING_VERIFICATION** (the #62 d_seg wall is "fundamental"). Per #363, a verdict
depending on an unverified assumption is downgraded to PROVISIONAL: **the directional verdict's "defer
score-native frame1" leg is PROVISIONAL-PENDING-#63** — the reactivation criterion is exactly the #63 result.
Round-3 resolution: the gate is a $0 experiment already pre-registered (op-routable #1), so the resolution
path is RUN #63, not ESCALATE. No verdict is asserted as final on an unverified assumption. SEAL after #63
lands (the 3-clean-pass equivalent: #63 result + PTNC smoke + requant smoke each close one open assumption).

## 5. WIRE-IN (Catalog #125)

1. **sensitivity-map — ACTIVE.** The deep invariant (d_seg-gradient lives in margin/logit space; frontier-base
   seg-correction is under-water; pose marginal concentrated in frame1 luma) is the top-level prior re-ranking
   the offensive levers: distortion-before-rate to 0.118, margin/logit-space loss over RGB-recon.
2. **Pareto constraint — ACTIVE.** F1 (partition-direct 253 KB optimal-context floor) + F3 (amortization beats
   direct storage by 0.052) + the #57/#61/#62 frame1 dual-fidelity wall are hard Pareto walls; the feasible
   move is NOT a cheaper frame1, it is right-loss + existing-amortizer-weight-attack.
3. **bit-allocator — ACTIVE.** Re-rank: next bytes/effort to distortion-closure (right-loss carrier / requant /
   zero-byte corrections) until 0.118; rate attack (smaller amortizer) only below 0.118. The closed-form λ\*=
   6.66e-7 score/byte water level + the 1.27 B/flip seg admission test are the allocator's analytic inputs.
4. **cathedral autopilot dispatch — N/A.** Deliberation surface; no archive bytes emitted. The #63/PTNC smokes
   are $0 local; only op-routable #3 (R1+R2) would dispatch (~$0.3), gated separately.
5. **continual-learning posterior — ACTIVE.** Anchor emitted via `tac.council_continual_learning.append_council_anchor`
   (the deep-invariant + the #63 gate + the distortion-before-rate re-rank reseed the planner).
6. **probe-disambiguator — ACTIVE.** The ONE remaining disambiguator is #63 (is the d_seg wall loss-invariant
   or wrong-loss?). Its DESIGN is pre-registered (`dseg_loss_conditioning_decisive_test_DESIGN_…`); running it
   is op-routable #1. The secondary disambiguator (does any amortizer beat 177 KB at near-zero distortion —
   the requant bank / fresh-init smaller-arch) is the door below 0.118.

## 6. CROSS-REFERENCES

`closed_spec_boundary_math_system_of_equations_20260610.md` (the closed-spec reframe — the polytope/quotient
invariant) · `information_theoretic_floor_report_v1_20260610T102335Z.md` (S_floor=0.118, F1–F6; partition-direct
loses) · `score_native_pose_carrier_20260610T125000Z.md` (#57 — frame1 dual-fidelity blocker) ·
`lever_c_viability_smoke_20260610T144739Z.md` (#62 — conv d_seg stalls at flat-frame floor, argmax-CE) ·
`dseg_loss_conditioning_decisive_test_DESIGN_20260610T145917Z.md` (#63 — the DESIGNED-not-run decisive
disambiguator, op-routable #1) · `lossless_stack_pointer_move_20260610T165749Z.md` (#64 — frontier lossless
exhausted, no-op) · `deferral_recovery_ledger_20260610T130200Z.md` (the get-back-on-track table, #307 audit
CLEAN) · `sota_plus_original_inventions_20260610T125100Z.md` (#59 — PTNC / MWCC / λ\*-allocator inventions) ·
`GOAL_standing_v3_20260610.md` (the levers) · `upstream/{evaluate.py,modules.py,videos/0.mkv}` (frozen authority).
