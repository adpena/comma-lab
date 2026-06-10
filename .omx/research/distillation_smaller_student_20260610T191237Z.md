# Distillation to a smaller learned student — VERDICT (task #74)

**Subagent:** `task74_distill_smaller_student`.

## FIRST LINE — did the exact pointer move?

**NO. The exact contest pointer did NOT move: 0.19110 → 0.19110 (unmoved).** This is a `$0`
`[local CPU-torch advisory]` campaign on the small (2/8-pair) tube; no contest-tier `evaluate.py`
row was produced, and the best advisory ΔS is **+68.87** (hugely WORSE, not better), so per the
sub-0.15 firewall + "Frontier scores are pointer-only" NOTHING promotes and **no paid dispatch is
warranted**. The campaign instead produces a sharp, mechanistically-explained verdict on WHY a
smaller learned student does not (yet) cross the wall.

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

<!-- SWEEP TABLE (filled as the daemon lands rows) -->
| size | bytes | exact d_seg | exact d_pose | S (student-only) | holds teacher? |
|---|---:|---:|---:|---:|:--:|
| (8-pair sweep landing — harvest from sweep_manifest.json) | | | | | NO (predicted) |

**Harvest the in-flight curve (durable daemon):** the sweep daemon (PID in `sweep.log`) double-forked
+ survives the session; a future agent reads the completed curve from
`experiments/results/task74_sweep/ladder_8pair_200ep_*/sweep_manifest.json` and emits the rows via
`tools/emit_distill_student_candidate_row.py --result <each>/train_result.json`. The 8-pair tube is
LESS overfit than the 2-pair anchor so each row's d_seg/d_pose will be ≥ (worse than) §1; the curve
quantifies the rate-vs-distortion tradeoff but, per §2, cannot cross the pose tube at any ladder rung
(all < the 177kb frontier).

**Independence from the #75 harness bug:** the sister #75 finding (the shared
`_shared/mlx_score_aware/bundle.py` defaults SegNet/PoseNet objective weights to 0.0 → scorer-blind
"score-aware" runs) does NOT apply here — this trainer wires the frozen SegNet (KL-T2) + PoseNet
(pose-MSE) teachers DIRECTLY with explicit nonzero weights (w_seg 0.5, w_pose 0.1) and the exact
d_seg/d_pose are RE-MEASURED on the frozen `DistortionNet`. The seg-KL train metric crushing 6.2→0.04
(§0) proves the seg objective is live, not inert. So the §2 pose-tube wall is a genuine capacity
finding, not the #75 inert-loop artifact.

## 5. VERDICT: DEFER-pending-research (NOT a kill) + the reactivation criteria

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 IMPLEMENTATION-LEVEL: the distillation
PRIMITIVE works (the student genuinely converges to the teacher's frames + SegNet distribution; the
recon-primary loss design is correct; numpy-portable + parity 1.0). The PARADIGM finding is **the
pose-tube width, not the distillation, is the binding constraint** — convergent with #62 (seg wall),
#73 (generic-basis feasibility needs ≥625KB/pair), and the 4-no-move meta-finding (the 177kb learned
HNeRV basis IS the cheap-feasible representation). A smaller learned student inherits d_seg-DIRECTION
from the teacher but cannot reproduce the teacher's frames to the ±5/255 fidelity PoseNet's tube
demands at sub-frontier byte budgets.

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
