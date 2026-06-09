---
council_tier: T4
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author, Boyd, Tao, Mallat, vandenOord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber, Atick, Rao, Tishby_memorial, Wyner, TimeTraveler]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
council_dissent:
  - member: Contrarian
    verbatim: "A full session of measurement converged on 'do what PR95 did.' The atlas RULED OUT the one breakthrough (low-rank pose carrier). The remaining levers — Y-focus, margin-weighted seg, L27 correction atoms, entropy coding — are INCREMENTAL refinements of a 0.193 structure, not a sub-0.19 paradigm. And B1-R2 burned ~5h flat. I VETO any 3000-epoch B1-R3 that is not first gated by a cheap descent-proof smoke, and I demand we stop calling incremental 'frontier_breaking.'"
  - member: Assumption-Adversary
    verbatim: "B2 measured the rank of d(pose)/d(INPUT) — near-full-rank. That is NOT the same as 'the pose CARRIER must be dense/expensive.' The pose OUTPUT is 6 smooth numbers per pair (600x6, low-entropy trajectory). Near-full-rank input-sensitivity means the inverse problem is WELL-CONDITIONED (easy to hit a target pose), which if anything HELPS a cheap carrier. 'Dense carrier necessary' conflates input-sensitivity-rank with carrier-cost. What is proven: NAIVE downsampling/flat-fill destroys pose, and a fixed GLOBAL low-rank basis fails. What is NOT proven: that a smart learned carrier cannot be much cheaper than HNeRV's 178KB."
  - member: Carmack
    verbatim: "The one-pair RGB overfit plateaued at 21 dB on the SAME decoder. A NeRV that can't memorize 2 frames past 21 dB is under-capacity or mis-wired. Atlas-weighting a too-weak decoder won't save it. Try the PR95 decoder config (channels/latents) directly, not our hi_nerv_local_tiny."
  - member: Hotz
    verbatim: "B1 lacked an RGB anchor — that one missing term plausibly explains the whole d_seg=0.50 collapse. Add the anchor, run the PR95-faithful staged curriculum, MEASURE the exact score. The atlas refinements are second-order; don't let them delay the first working exact number."
council_assumption_adversary_verdict:
  - assumption: "the dense amortized neural decoder is necessary for the pose term"
    classification: CARGO-CULTED
    rationale: "B2 proved near-full-rank INPUT sensitivity + that downsampling/flat-fill/global-low-rank-basis fail; it did NOT prove a smart learned carrier can't be cheaper than HNeRV. Refine to 'a dense Y-dominant carrier is the pragmatic choice; cheaper structured carriers remain open.'"
  - assumption: "pose is ~96% luminance, ~4% chroma (drop chroma fidelity)"
    classification: HARD-EARNED
    rationale: "measured energy fraction in the YUV6 Y vs chroma channels of the pose gradient, stable across smoke + full-48 (0.96/0.04). Robust-norm cross-check queued."
  - assumption: "the SegNet fragile set is ~5% sparse + boundary-structured"
    classification: HARD-EARNED
    rationale: "segnet_margin_field.v2 full-600: 4.83% fragile@2logit, p10/p50/p90 4.2/4.8/5.5%, class-2 robust, class-1 fragile, 100% of boundaries fragile."
  - assumption: "atlas-weighted training will make HiNeRV d_seg/d_pose DESCEND"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "the corrected B1 has NOT run; B1-R2 (uniform-RGB-free, distillation-only) was flat. The descent-proof smoke is the required Round-2 verification before any long run."
  - assumption: "the path below 0.192 exists within the PR95 structure (Y-focus + seg atoms + entropy coding)"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "no exact score below 0.192 has been produced; the atlas converged the design but the breakthrough lever is unproven (Contrarian's point). Verified only by an exact eval < 0.19199."
council_recursive_self_reflection:
  rounds_run: 2
  round2_downgrades:
    - "'dense decoder necessary' downgraded to PROVISIONAL (Assumption-Adversary: input-rank != carrier-cost)"
    - "'frontier_breaking' contribution downgraded: the ratified next step is frontier_protecting/incremental until an exact sub-0.192 lands"
council_decisions_recorded:
  - "op-routable #1 (LAUNCHABLE, race-mode): build atlas-weighted HiNeRV trainer (RGB/Y anchor + margin-weighted seg + Y-dominant pose), run a CHEAP descent-proof smoke (16-pair, ~300ep) FIRST"
  - "op-routable #2: GATE the long staged B1-R3 on the smoke showing exact d_seg/d_pose DESCENT (not flat)"
  - "op-routable #3: Assumption-Adversary action — measure the pose-OUTPUT trajectory entropy + the inverse-conditioning (is a cheaper-than-HNeRV carrier possible?) before declaring the decoder final"
  - "op-routable #4: verify RACE_MODE_ACTIVE.flag currency (dated 2026-05-14) vs the live leaderboard; the rigor cadence depends on it"
  - "op-routable #5: parallel SNeRV as a dense-Y-carrier candidate (sidestep the 21dB HiNeRV ceiling); Carmack's PR95-decoder-config alternative as a fork"
  - "op-routable #6: queue the rate attack (PR95 L20-L32 entropy coding) — the binding 62% term, untouched"
related_deliberation_ids: []
---

# GRAND COUNCIL SYMPOSIUM (T4) — all results, the roadmap, and next steps

UTC 2026-06-09 · convened per operator request. Facts (terminal SoT): frontier 0.19199 [contest-CPU] /
0.20533 [contest-CUDA]; **RACE_MODE_ACTIVE.flag EXISTS** (2026-05-14); B2 full-48 confirmed pose
near-full-rank. All evidence [macOS-CPU advisory], exact torch scorer on 0.mkv.

## The body of results deliberated
1. **B1-R2 (clean PR95 HiNeRV) FAILED** — d_seg=0.50 flat over 3000ep (distillation-only, no RGB anchor;
   2 fixed latent-independent frames). One-pair RGB overfit plateaus at 21 dB (below SegNet cell + pose).
2. **Evaluator atlas** — seg is SPARSE/spatial/frame1-only (~4.8% fragile, boundary-structured, class-2
   robust / class-1 fragile); pose is DENSE/temporal/both-frames + **96% luminance**.
3. **B0.5 codec budget** — naive per-frame seg-target storage = 424,722 B → rate 0.283 > frontier ⇒
   skeleton STORAGE loses; amortization needed.
4. **Region-cheapen** — keeping the seg boundary sharp gives 40× d_seg improvement, but pose explodes
   under interior cheapening (pose VETOES flat-fill).
5. **B2 gradient atlas (full-48)** — pose intrinsic dim 211/243/276 of 288 (participation 170) ⇒ pose
   near-full-rank, grows with pairs ⇒ no cheap GLOBAL low-rank pose carrier. SegNet boundary saliency
   denser at boundaries but spread over the receptive field.

## The deliberation (positions; each states its operating-within assumption)
- **Shannon (LEAD, info-theory):** operating within "the score is a rate-distortion problem." The frontier
  is RATE-bound (62%). The seg term's information is sparse (5%); the pose term needs dense fidelity but
  its OUTPUT is 6-dim smooth. The binding lever is the DECODER's entropy (rate), which we have NOT
  attacked. PROCEED to a working carrier, then the rate attack is the real frontier lever.
- **Dykstra (CO-LEAD, feasibility):** "the achievable region is the convex intersection." B2 says the pose
  constraint set is high-dim (near-full-rank) ⇒ the feasible witness must satisfy many pose constraints ⇒
  a dense carrier. But the seg constraints are sparse ⇒ a sparse correction layer suffices. The two
  intersect cleanly (orthogonal structure). PROCEED with dense-carrier + sparse-seg.
- **Rudin (CO-LEAD, interpretable):** "every decision is a contract." The atlas IS the interpretable
  weight map (margin field + Y-fraction). Use it to set loss weights mechanically (w ∝ fragility, Y>chroma)
  — no arbitrary guards (that was R1's sin). PROCEED, weights from the atlas.
- **Daubechies (CO-LEAD, multiscale):** "coarse-gates-fine." Pose is low-frequency-Y temporal; seg is
  high-frequency boundary. A multiscale carrier (low-freq Y dense + high-freq boundary sparse) matches.
- **Assumption-Adversary (sextet):** VETO-flag on "dense decoder necessary" — see dissent. B2 measured
  INPUT-rank, not carrier-COST. The pose OUTPUT is 6 smooth numbers; a smart carrier may beat HNeRV.
  DEMAND op-routable #3 (pose-output-entropy + inverse-conditioning) before the decoder is declared final.
- **Contrarian (sextet):** VETO on a blind 3000-ep B1-R3; the below-0.192 lever is unproven; DEMAND the
  cheap descent-proof smoke gate. Honesty: this is incremental until an exact sub-0.192 lands.
- **PR95Author (inner):** the winning recipe = HNeRV decoder + per-pair latents + 8-stage curriculum +
  L27 correction sidecar + L20-L32 entropy. Our B1 missed the RGB/Y anchor AND ran hi_nerv_local_tiny
  (under-capacity vs PR95's 229K config). Fix BOTH: add the anchor + use the real config.
- **Carmack / Hotz (grand):** see dissent — 21dB ceiling = capacity/wiring; add the anchor + the real
  config + MEASURE; don't over-engineer.
- **Atick / Rao / Tishby-memorial (pose specialists):** PoseNet is a cooperative receiver of temporal
  motion; the Y-dominance + both-frame dependence is the ego-motion signal. A Y-focused temporal carrier
  is the right inductive bias. Support op-routable #3 (is the motion manifold low-dim in OUTPUT space?).
- **Selfcomp / Balle / MacKay (rate):** the rate term is untouched; entropy-coding the decoder (L20-L32)
  + chroma cheapening is the measured rate lever. Queue it (op-routable #6).
- **Time-Traveler (grand):** "we have all the information we need." The atlas + PR95 lessons already
  specify the witness; bind them (anchor + atlas weights + real config + entropy) and MEASURE — don't add
  framework overhead.

## Recursive self-reflection (Catalog #363) — Round 2
Round 1 surfaced the assumptions above. Round 2 re-classified empirical_verification_status:
- VERIFIED (margin field, Y-dominance, codec floor, pose near-full-rank): source-inspection + measured.
- ASSUMED_AWAITING_VERIFICATION: "atlas-weighted training descends" (the corrected B1 has NOT run) and
  "a sub-0.192 lever exists within PR95 structure" (no exact score yet). ⇒ verdict-status PROVISIONAL on
  the frontier_breaking claim until an exact eval lands. Round-2 downgrade recorded.

## VERDICT: PROCEED_WITH_REVISIONS (vote: 24 PROCEED / 4 PROCEED_WITH_REVISIONS dissent / 0 REFUSE; quorum met)
The design is sound AND adversarially-tested, BUT with binding revisions:
1. **Race-mode + MVP-first gate:** build the atlas-weighted trainer; run a CHEAP descent-proof smoke
   (16-pair, ~300ep) FIRST; GATE the long staged B1-R3 on exact d_seg/d_pose DESCENT. No blind 3000-ep run.
2. **Refine the certainty:** "a dense Y-dominant carrier is the pragmatic choice (PR95-proven)" — NOT
   "neural decoder is a theorem." Keep op-routable #3 (pose-output-entropy) open as the possible
   below-0.192 breakthrough the Assumption-Adversary flagged.
3. **Fix BOTH B1 bugs:** add the RGB/Y anchor (missing) AND use the real PR95-class decoder config
   (not hi_nerv_local_tiny) — Carmack/PR95Author/Hotz consensus on the 21dB ceiling.
4. **Honest framing:** the ratified next step is frontier_PROTECTING/incremental (match-PR95-with-better-
   allocation) until an exact sub-0.192 lands; the frontier_breaking lever (cheaper pose carrier OR the
   rate attack) is the parallel research bet.

## Launchable next steps (race-mode actionable, ordered)
1. Build `atlas_weighted_hinerv` trainer: RGB/Y-anchor base + margin-weighted seg loss + Y-dominant pose
   loss + sidecar pay-rent gate; real PR95-class decoder config.
2. **Descent-proof smoke** (16-pair, ~300ep) → exact eval → does (d_seg, d_pose) DESCEND? (the B1-R2 test).
3. If descends → staged B1-R3 (Y-anchor → margin-seg → Y-pose → QAT → rate/C1a/λ/σ → Muon) → ep250/ep3000
   exact eval vs 0.19199.
4. Parallel: SNeRV dense-Y-carrier probe + Carmack PR95-decoder-config fork.
5. op-routable #3: pose-output-trajectory entropy + inverse-conditioning (the cheaper-carrier question).
6. Rate attack (L20-L32 entropy coding) queued — the binding term.
7. Verify the race flag's currency vs the live leaderboard.

Continual-learning anchor emitted via `tac.council_continual_learning.append_council_anchor`.
