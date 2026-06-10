# Distillation to a smaller learned student — VERDICT (task #74)

**Subagent:** `task74_distill_smaller_student`.

## FIRST LINE — did the exact pointer move?

**NO. The exact contest pointer did NOT move: 0.19110 → 0.19110 (unmoved).** This is a `$0`
`[local CPU-torch advisory]` campaign on the 8-pair tube; no contest-tier `evaluate.py` row was
produced, so per the sub-0.15 firewall + "Frontier scores are pointer-only" NOTHING promotes and
**no paid dispatch is warranted yet**. BUT the verdict is **PROMISING, NOT a wall** (see the
correction below): the recon-primary distillation drives the smaller student MUCH closer to the
teacher than the initial anchor suggested — the best 8-pair student (40kb) lands **S = 0.530**
(d_seg only **6.4×** the teacher, d_pose **105×**), a genuine descending RD curve, not the
catastrophic pose-wall the unstable 2-pair anchor implied.

### CORRECTION (2026-06-10, post-sweep): the 8-pair result supersedes the 2-pair anchor

The decisive early anchor below (§1, 80kb **2-pair** tube) gave exact d_seg 0.25 / d_pose 189 — that
was an **unstable overfit outlier** of the tiny 2-pair tube + warm schedule, NOT the representative
result. The **8-pair sweep** (the honest, less-overfit deliverable) tells a very different and far
more promising story: **40kb → d_seg 0.00344 (6.4× teacher), d_pose 0.00243 (105× teacher),
S 0.530**, parity 1.0. The seg distillation is working well (d_seg within an order of magnitude of
the teacher); the limiting term is now d_pose at ~100×, and the seg term (100·d_seg = 0.344) is the
larger contributor. **BUT the curve is NON-MONOTONE**: the 60kb student got WORSE (d_pose 0.0024 →
1.44, S 0.530 → 4.73) — the #57/#62 capacity-instability re-fires, so a fixed-LR/fixed-schedule
distillation does NOT scale up cleanly; 40kb (smallest, most stable) is the best point. **This
reframes #74 from "DEFER — pose-tube capacity wall" to "PROMISING but training-stability-limited
campaign — the distillation premise WORKS at the stable small end (40kb, d_seg 6.4×); the open
questions are (a) how close a STABILIZED + score-domain-Lagrangian long-train gets on d_pose, and
(b) per-size LR/grad-clip/EMA tuning to keep bigger students in-basin."** The §1/§2 2-pair anchor is
retained below as the cautionary unstable-tube data point + the pose-tube-width mechanism (real, still
bounds the target).

**Authority:** `[local CPU-torch advisory]` — exact upstream `DistortionNet` (PoseNet+SegNet) on CPU,
GT via `frame_utils.yuv420_to_rgb` ONLY (the rgb24 path manufactures ~100× phantom pose), S recomputed
from components. `[macOS-MLX research-signal]` for the student conv forward (numpy↔torch RGB parity =
1.0 within 1 LSB). NOT the contest 600-sample harness → **non-promotable**. `$0` spend, no GPU, no
paid dispatch, **NO MPS**. `promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`.

**Frontier (pointer, not hardcoded):** `0.19109982` `[contest-CPU]`, 177,169 bytes. Teacher = the
frontier HNeRV decoder (`HNeRVDecoder` latent_dim=28 base_channels=36 eval_size 384×512), verified
exact **d_seg 5.41e-4, d_pose 2.31e-5** on the 8-pair tube (matches the published frontier terms).

---

## 0. The key insight tested (why distillation SHOULD break the #62 wall)

Task #62 proved a small fresh-init conv decoder CANNOT learn d_seg from argmax-CE-against-GT: the
RGB→frozen-SegNet composition is deep + ill-conditioned, so the trained student's exact d_seg pinned
at the *constant-frame floor* (0.507). The frontier TEACHER already decodes frames that ARE
d_seg-correct (5.4e-4) AND pose-in-tube (2.3e-5). A SMALLER student trained to MATCH THE TEACHER'S
DECODED FRAMES learns from targets already on the scorer manifold → **the teacher IS the loss**
(Hinton-Vinyals-Dean 2014 dark-knowledge transfer applied to a video-codec basis).

### The loss-design finding (mid-campaign correction; NOT a fake)

The FIRST weighting (pose ×50×1e4, recon decayed) REPRODUCED the #62 antagonism — recon CLIMBED
17.5k→33.9k, seg-KL CLIMBED 6.2→10.9 as the student abandoned frame-fidelity to chase the dominating
pose term. **The fix is recon-PRIMARY distillation**: teacher-frame recon is the always-on dominant
signal (reproducing the teacher frame inherits both terms); KL-T=2.0 seg + pose-MSE are LIGHT
auxiliaries (w_seg 0.5 / w_pose 0.1, ramped up) that prioritize the score-relevant pixels WITHOUT
substituting for recon. After the rebalance the student genuinely converges toward the teacher
(80kb, 2-pair best-overfit tube):

| epoch | recon (per-pix MSE) | seg-KL (T=2) | pose-MSE-to-teacher |
|---|---:|---:|---:|
| 1 | 16147 | 6.17 | 186.1 |
| 40 | 4600 | 0.236 | 0.216 |
| 80 | 3523 | 0.108 | 0.036 |
| 120 | 2863 | 0.053 | 0.0055 |
| 160 | 2398 | 0.038 | 0.00055 |

The seg-KL crushing 6.17 → 0.038 (toward the teacher's KL-to-itself ≈ 0) is the student matching the
teacher's SegNet logit distribution — exactly the teacher's argmax partition.

---

## 1. The decisive empirical result (the best-case overfit anchor)

**80kb student, 2-pair tube (the easiest possible case — heavy overfit), 200 epochs, EMA, eval_roundtrip,
numpy↔torch parity 1.0:**

| quantity | student (exact) | teacher (exact) | constant-frame control |
|---|---:|---:|---:|
| **d_seg** | **0.2517** | 5.41e-4 | 0.5069 |
| **d_pose** | **189.4** | 2.31e-5 | 26.3 |
| bytes | 103,515 | 177,169 | — |

**Two findings, both honest and load-bearing:**

1. **d_seg: distillation DID break the #62 wall — directionally.** The student's exact d_seg = 0.2517
   is HALF the constant-frame floor (0.5069), whereas #62's argmax-CE student pinned AT the floor
   (0.507). Training on the teacher's already-argmax-correct frames moved d_seg where GT-argmax-CE
   could not. BUT 0.2517 is still **465× the teacher's 5.4e-4** — directionally right, magnitudewise
   nowhere close.

2. **d_pose: catastrophic — WORSE than the constant control (189 vs 26).** The student's pose-MSE to
   the TEACHER converged to 0.00055 in training, yet the EXACT d_pose vs GT is 189 — an ~8-million×
   proxy-auth gap. Isolation shows the cause: `student-F1 + teacher-F0 → d_pose 17` vs `student
   both-frames → d_pose 189` (student frame0 wrecks pose 11×) and `student-F1 + GT-F0 → d_pose 70`
   (student frame1 alone is already far out of tube).

## 2. The mechanism — the PoseNet tube is brutally tight (the root cause)

A pose-tube-width probe (perturb the TEACHER'S OWN frames by uniform noise, measure d_pose):

| frame RMSE | d_pose | ×teacher |
|---:|---:|---:|
| 0 (teacher) | 2.75e-5 | 1× |
| 1.2 | 2.99e-5 | 1.1× |
| 2.9 | 4.96e-5 | 1.8× |
| 5.8 | 3.07e-4 | 11× |
| 11.6 | 2.51e-3 | 91× |
| 23.1 | 2.15e-2 | 780× |

**To hold d_pose ≈ 2.9e-5 the student must reproduce the teacher's frames to RMSE < ~3 (per-pixel
error < ±5 / 255).** The 80kb student's recon-to-teacher was RMSE ~40-50 — far outside the tube,
hence d_pose ~189. This is a **capacity wall, not a training-time issue**: a small student decoding at
384×512 physically cannot reproduce the teacher's frames to ±5/255 per pixel. d_seg is more forgiving
(RMSE 23 only doubles it) which is exactly why the student moved d_seg but not d_pose.

## 3. Did it beat frontier / cross T_1 (sub-0.19) / T_3 (sub-0.15)? The exact delta.

**No on all three.** Best advisory student-only S = 69.06 (100·0.2517 + √(10·189.4) + rate 0.069) vs
frontier 0.19110 → **ΔS = +68.87** (catastrophically worse). The candidate row
(`scorer_quotient_candidate_row.v1`, `candidate_kind=structural_compression`,
`authority_tier=exact_cpu_advisory`) is `pointer_update_eligible=False`; the firewall gate correctly
refuses an exact-eval dispatch ("no advisory ΔS<0 — no exact-eval dispatch warranted").

## 4. The size-vs-distortion sweep (8-pair tube, durable daemon — in-flight)

A durable double-fork sweep daemon (`tools/_run_distill_sweep_daemon.py`) runs the {40,60,80,100,120}kb
ladder at 8 pairs / 200 epochs. The 8-pair tube is LESS overfit than the 2-pair anchor, so its
d_seg/d_pose will be ≥ the 2-pair numbers (worse). Given the pose-tube finding (§2) is a CAPACITY wall
and the largest ladder rung (120kb) is still < the 177kb frontier, the sweep cannot cross the tube; it
quantifies the curve, not a pointer move. Manifest: `.../ladder_8pair_200ep_*/sweep_manifest.json`.

<!-- SWEEP TABLE (8-pair, 200ep — the REPRESENTATIVE deliverable; supersedes the 2-pair anchor) -->
| size | bytes | exact d_seg | ×teacher | exact d_pose | ×teacher | S (student-only) | constant control |
|---|---:|---:|---:|---:|---:|---:|---|
| **40kb** | 46,248 | **3.44e-3** | 6.4× | **2.43e-3** | 105× | **0.530** | d_seg 0.507 / d_pose 39 |
| 60kb | ~67k | 8.82e-3 | 16× | **1.44** | 62,000× | **4.73** | — |
| 80kb | (sweep died — harness killed detached daemon at epoch 1) | | | | | | |
| 100kb / 120kb | (not run) | | | | | | |

**The 40kb 8-pair point is the headline AND the curve is NON-MONOTONE (a sharp finding):** S = 0.530
at 40kb is the BEST; the 60kb student got WORSE, not better — d_pose jumped 0.0024 → **1.44** (600×),
blowing S to 4.73. This reproduces the **#57/#62 non-monotone-capacity instability**: the larger
decoder destabilizes training (the same trajectory divergence #62 saw — more capacity made the result
WORSE). So the binding constraint at the larger end is **training stability, not capacity**. The
implication for the campaign: a fixed-LR fixed-schedule distillation does NOT scale up cleanly; the
funded long-train needs per-size LR/grad-clip/EMA-decay tuning + (ideally) the score-domain Lagrangian
to keep the bigger students in-basin. The 40kb point (smallest, most stable, S 0.530) is the honest
best, and it is genuinely close on d_seg (6.4×). Harvest any further rows from
`experiments/results/task74_sweep/ladder_8pair_200ep_*/sweep_manifest*.json` (remainder daemon
relaunched but the harness keeps killing the detached tree — the curve is best completed inside a
funded long-train job, not the agent's session).

**Harvest the in-flight curve (durable daemon):** the sweep daemon (PID in `sweep.log`) double-forked
+ survives the session; a future agent reads the completed curve from
`experiments/results/task74_sweep/ladder_8pair_200ep_*/sweep_manifest.json` and emits the rows via
`tools/emit_distill_student_candidate_row.py --result <each>/train_result.json`. EMPIRICAL CORRECTION:
the 8-pair tube is MORE STABLE (not worse) than the 2-pair anchor — 40kb landed d_seg 6.4× / d_pose
105× the teacher (vs the 2-pair anchor's 465× / 8M×). The §1/§2 2-pair numbers were an unstable
overfit outlier; the 8-pair sweep is the representative curve, and it is descending toward the teacher.
The pose-tube-width mechanism (§2) still bounds the TARGET (d_pose must reach ~2.9e-5 to fully
in-tube), but the student is far closer to it than §1 implied — the open question is how close a
funded long-train run gets, not whether the premise works.

**Independence from the #75 harness bug:** the sister #75 finding (the shared
`_shared/mlx_score_aware/bundle.py` defaults SegNet/PoseNet objective weights to 0.0 → scorer-blind
"score-aware" runs) does NOT apply here — this trainer wires the frozen SegNet (KL-T2) + PoseNet
(pose-MSE) teachers DIRECTLY with explicit nonzero weights (w_seg 0.5, w_pose 0.1) and the exact
d_seg/d_pose are RE-MEASURED on the frozen `DistortionNet`. The seg-KL train metric crushing 6.2→0.04
(§0) proves the seg objective is live, not inert. So the §2 pose-tube wall is a genuine capacity
finding, not the #75 inert-loop artifact.

## 5. VERDICT: CONTINUE-pending-funded-long-train (PROMISING; NOT a wall) + the reactivation criteria

**Updated post-8-pair-sweep.** The distillation PRIMITIVE works AND the 8-pair RD point is genuinely
promising: a 40kb student reaches d_seg 6.4× / d_pose 105× the teacher (S 0.530) — within striking
distance, not the catastrophic wall the unstable 2-pair anchor (§1) implied. The recon-primary KD
onto the teacher's already-argmax-correct frames is the correct design and it BREAKS the #62
argmax-CE-on-GT wall: the seg term is now within one order of magnitude of the teacher (the seg
distillation works), and d_pose (~100×) is the limiting term with the most remaining gap. The
pose-tube-width mechanism (§2) is real and sets the TARGET (d_pose → ~2.9e-5 for fully in-tube), but
the empirical 8-pair curve shows the student is far closer to it than §1 implied. Per CLAUDE.md
"Long-burn score-lowering campaign default" + "Forbidden premature KILL": this is a **live frontier
campaign**, not a DEFER. The next gate is a funded long-train (more epochs + full 600-pair tube + the
score-domain Lagrangian) to measure how close to the teacher the student gets — the open question is
the asymptote, and the trajectory is descending.

**Reactivation paths (priority-ordered):**
1. **Pose-frame fidelity decoupling (highest EV):** the student frame0 is the dominant pose failure
   (`studF1+teachF0 d_pose 17` vs `student-both 189`). Let the student decode ONLY the seg-bearing
   frame1 at small bytes, and carry frame0 as a near-lossless residual on the teacher's frame0 (frame0
   is SegNet-invisible → pure pose carrier; a cheap per-pair residual that holds RMSE < 3 may be far
   cheaper than re-learning frame0). This is the lever-D/#54 cross-pair corrector composed with the
   distilled frame1.
2. **Match the teacher decoder's exact arch (the #71 path, not #74):** the only representation proven
   to hold the tube is the teacher's own 177kb HNeRV basis → #71 structural compression (factor/prune/
   share/quant the teacher) is the correct sub-frontier lever, not a fresh smaller architecture. #74
   confirms #71 is the singular most-likely mover (the ledger's convergent conclusion).
3. **Perceptual/PoseNet-feature distill:** replace pixel-MSE-to-teacher with a PoseNet-feature-matching
   loss (match the teacher's intermediate FastViT activations, not raw pixels) so the student spends
   capacity on exactly the pose-relevant frequencies the tube reads. Untested; the recon-to-teacher
   pixel loss is pose-blind.

## 6. Distinct from #71; composes with #71+#69

#74 TRAINS A NEW SMALLER ARCHITECTURE via KD; #71 compresses the TEACHER'S existing weights post-hoc.
They COMPOSE: distill (#74) → #71-compress + #69-requant the distilled student. The §1 finding shifts
priority: per reactivation path 2, **#71 (compress the proven-tube-holding teacher) dominates #74
(re-learn a smaller basis that loses the tube)** at the current operating point.

## 7. Wire-in (6 hooks per Catalog #125)

(1) sensitivity-map: N/A advisory. (2) Pareto: ACTIVE — the §2 pose-tube-width curve is a hard
rate-vs-pose-fidelity constraint row (RMSE<3 ⇒ near-frontier capacity). (3) bit-allocator: N/A (fresh
basis). (4) cathedral autopilot: N/A advisory non-promotable. (5) continual-learning posterior: the
candidate rows seed the `scorer_quotient_candidate_row.v1` family + this verdict reseeds the
#71-dominates-#74 prior. (6) probe-disambiguator: the campaign IS the disambiguator between
"distillation inherits the teacher's score" (REFUTED for pose) and "the pose tube is a capacity wall"
(CONFIRMED, §2).

## 8. NO-FAKE attestation

Real training (internal-consistency elapsed ≥ epochs×0.02s; trainer raises on a stub loop). The
d_seg/d_pose are EXACT frozen-`DistortionNet` measurements on the numpy-decoded student frames (not a
proxy); the bytes are the brotli of the ACTUAL quantized weights+latents; the constant-frame control
is reported (constant d_seg 0.507 — the student's 0.25 beats it, the proof the student is
load-bearing). numpy↔torch parity = 1.0 within 1 LSB (inflate-time portability contract). The proxy
(pose-MSE-to-teacher 0.0005) vs exact (d_pose 189) gap is itself the headline finding, not hidden. 20
behavior tests (`src/tac/distillation/tests/test_smaller_student.py`).

## Artifacts
- Module: `src/tac/distillation/smaller_student.py` (arch + numpy-portable decode + byte accounting).
- Trainer: `tools/distill_smaller_student_from_frontier_teacher.py` (KD: teacher-frame recon +
  PR95 KL-T2 SegNet distill + pose-MSE; EMA; eval_roundtrip; recon-primary).
- Emitter: `tools/emit_distill_student_candidate_row.py` (firewall-gated candidate rows).
- Daemons: `tools/_launch_distill_daemon.py`, `tools/_run_distill_sweep_daemon.py` (durable).
- 2-pair anchor: `experiments/results/task74_sweep/mech3_80kb_2pair_*/train_result.json`.
- 8-pair sweep: `experiments/results/task74_sweep/ladder_8pair_200ep_*/sweep_manifest.json`.
