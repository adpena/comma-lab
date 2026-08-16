# ns1 — negative-signal audit (post-na7 window) + the missing patterns toward sub-0.15

Date: 2026-08-16 · Owner: MAIN · Operator steers: "Audit and optimize against all negative signal" +
"What other patterns are we missing that would lead us to frontier score lowering beyond 0.15?"
Window: the post-na7 corpus (na7 sealed the arc through 08-14; this audit covers 08-14 → 08-16).
Frontier: hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]; gap −0.0095973.

STORES CONSULTED: MP2_ADVISORY_ADJUDICATION.json + the three mp2 contest_auth_eval rows + retained
semantic_state tensors (base + 3 candidates, THIS audit's new computation) + wd2 refusal verdict +
wd3 n120 family disposition + #1058 close memo + rfo2/mz2/pz4a verdicts + qs2/qs4/qs5 verdicts +
pk4 FORMULATION verdict + #850 (pose GN truncation) + #996 (coder axis vs memoryless) + #1054
(CPU pose 21×) + eu4 route + na7 breakthrough map + memories [[m48 NEG↔CURE]] +
[[same_defect_negatives...]] + [[m66 gap-to-demonstrated-floor]] + [[m94 a-negative-measures-the-
instrument]] + [[sparse_plus_learned_prior...]].

## A. NEW MEASUREMENT ($0, from retained payloads): pose brittleness is ANISOTROPIC, ~94×

Per-candidate semantic-weight perturbation norms vs the base (38 tensors, global L2 75.16):

| candidate | ΔB | ‖ΔW‖ (rel) | tensors touched | Δd_pose | Δd_pose per unit rel |
|---|---|---|---|---|---|
| keep75∖keep87 | −25 | 0.2615 (0.35%) | 3 (FiLM blocks_1) | +4.08e-4 | 0.117 |
| keep87 | −130 | 0.2102 (0.28%) | 3 (FiLM blocks_1) | +5.36e-4 | 0.192 |
| q3/q4 mixed | −823 | 21.41 (**28.5%**) | 4 (frame_embed dom.) | +5.84e-4 | **0.00205** |

1. **The pose-critical subspace is LOCATED: `blocks_1_film_weight` rows.** A 0.28% perturbation
   there does the damage of a 28.5% perturbation in frame_embed — ~94× sensitivity spread.
2. **My #1058-close mechanism statement is REFINED (self-correction):** "dose-response in bytes"
   stands, but damage is NOT norm-proportional — it is direction-specific. The family closure
   SURVIVES the refinement quantitatively: the admission budget allows Δd_pose ≈ 5.1e-9·ΔB, and
   even the LOWEST-sensitivity measured direction overshoots it ~100× (rel ≤0.2% allowed vs 28.5%
   used for −823 B). Sensitivity-aware post-hoc allocation cannot rescue the family by ~2 orders.
   verdict_scope: family, refinement recorded at source.
3. **Constructive product:** the FiLM-row protection list (exclude blocks_1 FiLM from ANY future
   coarsening; add train-time perturbation robustness THERE first) + the quantitative screen
   (Δd_pose_budget = 5.1e-9·ΔB at this operating point) — the $0 pre-proof gate for any future
   weight-edit proposal, replacing bought advisory rows.

## B. Post-na7 negative corpus — scope re-grades + NEG↔CURE extraction

| negative | scope (checked) | the cure/lever it implies |
|---|---|---|
| mp2 ×3 REFUSED (pose 3.8–5.0×) | FAMILY upheld + §A refinement | Train-time weight robustness = the rate unlock (see P1); FiLM protection list; $0 screen |
| wd2 refused (8.2× seg bar, decelerating) | instance (prune+refit @60ep) | Distillation-aware TEACHER training (teacher never trained to be distillable) |
| wd3 fresh-init family negative (n120) | family @65ep, ladder recorded | Optimizer-state pose-carry law (warm 3×) → judge students at seg asymptote; warm-lineage rung stays live |
| QAT-leg trajectory-stop (ep46 dominated) | instance | Marginal S/Metal-hour on THIS vehicle+regime ≈ 0 → all Metal hours route to regime-change work |
| carrier exact race TIE_INCUMBENT | measured closed (lossless) | Confirms coder axis shut at q11; lossy leg now dead by §A screen |
| pz4a coarsening +2,232 B | instance | Same weight-brittleness genus — folds into P1 |
| pk4 linear frame-0 overlays GATE_FAIL ×3 | FORMULATION (linear-fitted) | The EXACT-solve sibling WON (qs5 compensation, pose BELOW base) → P3 |
| #850 pose GN truncated (2-3 relins, still descending 13–23%/iter) | under-convergence, never cured | Uncapped pose solve = named unfinished measurement → P3 |
| #996 coder axis "closed" | closed vs MEMORYLESS bounds only | Learned/context-prior race (hp1 AR prior) not confirmed on hv1 sections → P5 verify-then-race |
| ops: governor stuck-throttle (#1073) · liveness false-positive (#1064) · wc1 r1 wrong-object flag | named, cures owned | #1073/#1064 remain the two open apparatus debts; wc1 lesson already encoded |

**Instrument checks (m94):** all refusals rode the deterministic same-instrument advisory path
(repeat-identical); the pose magnitudes are instrument-conditioned (CPU pose 21× vs CUDA, #1054)
but the refusals are robust under any transfer model (additive or multiplicative both leave the
pose leg ≥2,000× the bar). No verdict rests on instrument noise.

**Winning-sibling pattern (m48):** every pointer move in this lineage came from (a) decode-identical
recodes, (b) exact solves with in-compile compensation, (c) long joint training. Every refusal is a
POST-HOC lossy edit. The boundary is now measured, located (§A), and priced.

## C. THE ANSWER — patterns we are missing toward sub-0.15 (ranked)

**The meta-finding first:** within this vehicle AND this training regime, all three axes now sit at
MEASURED floors — rate (e960 fit converged, coder at memoryless bounds, post-hoc edits pose-fatal),
seg (QAT-leg asymptote), pose (every post-hoc family formulation-closed). The negatives collectively
prove the remaining −0.0096 is NOT reachable by edits on this frozen object. The missing patterns
are REGIME-level:

**P1 — Train-for-editability (the inverse of every §B refusal).** Pose brittleness is a TRAINED
property: the e960 burn ran QAT on TOKENS (which is why token quantization ships) but never on the
semantic WEIGHTS — so q3/q4, row-prune, rank-cut, and width-distill all die post-hoc. One burn-2
regime adds: weight-perturbation robustness on the located FiLM-critical rows + mixed-q3/q4 QAT +
row-dropout + spectral (rank) penalty + distillation-aware objective. Every one of §B's refused
byte pools (−823, −2,051, rank pool, width multi-KB) then becomes harvestable at ~zero pose tax —
this is the only named supplier class for the −15,157 B rung. Machinery exists (resume-proven
lineage, watchers, selector). Est. reach: multi-KB → the rate half of the gap.

**P2 — Token-drop × PROVEN Schur compensation (the one unraced rate rung, and it is frame-level).**
rfo2's last surviving rung. Token edits decode to FRAME changes — exactly the class where qs5
PROVED in-compile frame-0 compensation carries ~zero pose tax (the winning sibling of the entire
pose-damage genus). Compose: #869-style adaptive token-drop map × B/H benefit-exact model × 0.785
flips/B breakeven × qs5 compensation. Unmeasured at ANY drop level on this vehicle.

**P3 — Exact realized-acceptance pose-aimed solve (zeroing pose = 86% of the gap by itself).**
Pose contribution 0.0082946; pose→0 buys −0.0083 and can afford ~12.4 KB of edits at the rate
exchange. What is DEAD is the linear-fitted overlay (pk4, FORMULATION). What is PROVEN is the
exact-solve mechanism (qs5 moved d_pose BELOW base as a side effect). What is UNFINISHED is #850:
every pose GN in the corpus stopped at an iteration cap while still descending 13–23%/iter. The
un-run measurement: uncapped, per-pair, realized-acceptance pose solve on the hv1 base with seg-hold
— priced with the qs 4 B/pair coder. This is also js8's territory — route it AS the js8 successor's
first concrete row rather than a rival lane.

**P4 — The joint line umbrella (already owned, now with sharper inputs).** js8 implicit-joint
conditioning · #982 trained receiver on OUR labels · #984 composed campaign. This audit hands them:
the FiLM protection list, the §A sensitivity screen, the P1 robustness spec, and the compute law
(zero marginal S/hour on the old regime → the Metal slot belongs to these).

**P5 — Verify-then-race (cheap, possibly already dead):** hp1's learned AR prior was fired (#976)
— confirm whether it ever raced on the hv1/HPAC sections; #996 closed the coder axis only vs
MEMORYLESS bounds, so a learned-context gain is not excluded by any receipt I hold. One grep + at
most one $0 race. Also noted, not ranked: the CPU-axis pose 21× degradation is a property someone's
vehicle could be robust to (leaderboard ranks CPU) — parked as a design input for P1/P4 vehicles.

**Explicitly NOT missing (measured dead, do not re-open without new preconditions):** post-hoc
weight edits at any allocation (§A screen) · lossless recoding (mz2/carrier tie) · fresh-init
distillation @65ep (wd3 ladder governs) · linear frame-0 pose overlays (pk4) · singleton int12
moves (F26 converged) · more epochs on the current regime (QAT-leg stop + e960 fit).

## D. Fire-orders

1. P1 burn-2 regime charter (train-for-editability) — compose at the next charter boundary; the
   §A protection list + screen are its OPTIMAL FORM inputs. Owner MAIN (charter), arm at spawn.
2. P2 token-drop×compensation — $0 arithmetic pass first (drop-map × B/H × breakeven on retained
   token fields), scorer row only if projection clears the §A-style screen. Owner MAIN.
3. P3 uncapped pose solve — fold into the js8 successor charter as its first row (avoid a rival
   lane); #850's cap-lift is the named prerequisite.
4. P5 — RESOLVED IN THIS AUDIT: hp1 raced the learned AR prior on the tq1c IX2 token stream and
   was BYTE-NEGATIVE (+114,870 B; the counted ≤10K→456KB model family cannot realize the measured
   context headroom economically — receipt ddm_hp1_20260806). Scope: instance (IX2 stream, that
   model family). On the hv1/HPAC sections the cheapest instance of the same rung is ALREADY
   QUEUED: the rx2 ep60 RCF1 table harvest (a 100 B learned table, mid-run probes measured it
   NEUTRAL). hp1's economics + #996's section measurements + the RCF1 neutrality make this a
   low-probability rung — stays queued LOW, no new fire.
5. Apparatus debts unchanged: #1073 (daemon OFF until landed) + #1064.
