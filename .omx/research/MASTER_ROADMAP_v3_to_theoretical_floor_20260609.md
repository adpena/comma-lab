# MASTER ROADMAP v3 — frontier score lowering to the FINAL ULTIMATE THEORETICAL FLOOR

UTC 2026-06-09 · claude · extends `MASTER_ROADMAP_v2_post_swarm_deforestation_20260609.md`
(v1/v2 PRESERVED, append-only). Frontier (pointer, never hardcoded): **0.19199 [contest-CPU]** /
**0.20533 [contest-CUDA]**. Score law (frozen, the only authority):

    S = 100·d_seg + √(10·d_pose) + 25·B/N        N = 37_545_489 (source bytes)

d_seg = SegNet last-frame 5-class argmax DISAGREEMENT RATE ∈ [0,1].
d_pose = PoseNet 6-of-12-dim MSE (≥0). B = compressed archive.zip bytes.

---

## 0. THE DECISIVE FINDING that reframes everything (B1-R2 clean baseline)

The clean, stabilized (grad-clip 1.0 + warmup + cosine + wd) PR95-family 229K HiNeRV backend-only
baseline `b1_229k_clean_20260609T085348Z` RAN TO COMPLETION (ep3000, TRAIN_EXIT rc=0). Its complete
exact-eval trend (12 backend-only [macOS-CPU advisory] points):

| epoch | score | d_seg | d_pose | bytes | stage |
|------:|------:|------:|-------:|------:|------|
| 250  | 90.12 | 0.5048 | 155.75 | 256072 | 1 CE |
| 500  | 91.28 | 0.5048 | 165.01 | 255192 | 2 τ-softplus |
| 750  | 90.39 | 0.5048 | 157.90 | 254364 | 2 |
| 1000 | 90.39 | 0.5048 | 157.90 | 254364 | 3 smooth |
| 1250 | 90.39 | 0.5048 | 157.90 | 254364 | 3 |
| 1500 | 91.49 | 0.5041 | 167.33 | 254181 | 5 hard-pixel/C1a |
| 2000 | 91.49 | 0.5041 | 167.33 | 254181 | 6 λ-sweep |
| 2500 | 91.56 | 0.5045 | 167.58 | 254138 | 7 σ-sweep |
| 2750 | 104.88 | 0.6386 | 166.91 | 254318 | 8 Muon (transient WORSE) |
| 3000 | 90.36 | 0.5048 | 157.70 | 254012 | 8 Muon |

**VERDICT: the clean baseline is FLAT — d_seg stuck at ~0.50, d_pose stuck at ~160, score ~90, for
the ENTIRE 3000-epoch 8-stage curriculum.** The renderer never learns evaluator-equivalent frames.
This is NOT "early" (the full curriculum incl. QAT + Muon completed). Stabilization fixed R1's
DIVERGENCE (no NaN, SEG proxy-loss stable) but NOT the underlying NON-LEARNING.

### Strict scrutiny on the negative (is it real, or a measurement bug?)
REAL training failure, not a bridge bug: bytes vary across checkpoints (export reads real distinct
weights); d_seg varies slightly 0.5041–0.6386 (inflate responds to weights); EMA-best export is a
smoothed real checkpoint. d_seg≈0.50 is the signature of DEGENERATE frames (renderer outputs near-
constant content; SegNet maps it to one dominant class → ~50% of source pixels disagree). It is
OUR HiNeRV, NOT the PR95 paradigm (PR95 = 0.193 with this recipe). Decision recorded:
`INSPECT_BINDING_CONSTRAINT` (binding=seg), auto_kill=False (Forbidden premature KILL).

### Top root-cause hypotheses (Phase 0 diagnosis settles which)
1. **No RGB-reconstruction anchor (MOST LIKELY).** B1 trained on scorer-DISTILLATION only
   (`--distillation-weight 1.0` SegNet-KL + `--pose-distillation-weight 1.0`). HiNeRV is fundamentally
   a video-FITTING net: it must first MEMORIZE the video via RGB reconstruction, THEN scorer-aware
   fine-tune shapes it toward the cells. Distillation-only through a frozen argmax-hinge + YUV6 head
   gives weak/noisy gradients → collapse to degenerate frames.
2. **Inflate-resolution mismatch (historical 48×64 catastrophe class).** If inflate emits frames at
   the wrong resolution, SegNet sees garbage → d_seg≈0.50 regardless of training.
3. **Differentiable-YUV6 / eval-roundtrip gradient path severed** (upstream rgb_to_yuv6 is @no_grad).
4. **Architecture mis-sized / coordinate-latent wiring wrong** (229K param count right, structure wrong).

**Phase-0 probe ($0 local MLX, MVP-first):** train HiNeRV with a DIRECT RGB-reconstruction loss
(L2 vs source `upstream/videos/0.mkv` frames) for ~300 ep; measure d_seg. If d_seg drops sharply →
hypothesis 1 confirmed (distillation-only was the bug; add RGB anchor as the curriculum base). If
d_seg stays ~0.50 → architecture/inflate bug (hypotheses 2–4); inspect inflate output resolution +
a single-frame overfit test.

---

## 1. THE FINAL ULTIMATE THEORETICAL FLOOR (math, not vibes)

The floor = min S over all legal archives. The three terms decompose the problem:

### 1a. The frontier is RATE-BOUND (the dominant lever)
At 0.19199 with B≈178,493: rate = 25·178493/N = **0.1189 (62% of the score)**. seg+pose ≈ 0.073.
**To go below the frontier, the dominant lever is BYTES**, while keeping d_seg/d_pose contributions small.

### 1b. d_seg and d_pose can both → ~0 (the slack the evaluator hands us)
- **d_seg slack:** SegNet scores only the ARGMAX (not logits) of the LAST frame of each pair. Per-pixel
  precision needed = the top-2 logit MARGIN (boundary pixels protected; segment interiors free). And
  ~600/1200 frames are SEG-FREE (only the 2nd-of-pair is read). So d_seg→0 costs only the boundary +
  last-frame information. This is per-pixel WATERFILLING (UNIWARD-style: perturb where SegNet is blind).
- **d_pose slack:** PoseNet scores only 6 of 12 dims; ego-motion is a SMOOTH low-dim trajectory (a car
  driving). The √(10·d_pose) term is CONCAVE — cheap to push d_pose small, expensive to push it to
  EXACTLY 0. The optimum accepts a tiny d_pose. Half the pose head is null space (free).
⇒ With enough (but few) bytes, **d_seg≈0 and d_pose≈ε**, so S ≈ rate.

### 1c. The floor is therefore ≈ 25·B_min/N + ε
B_min = the entropy of the EVALUATOR'S VIEW of video 0.mkv (NOT the RGB video):
- **600 seg-argmax maps** (last-frames): spatially smooth (5 classes) + temporally coherent. Entropy-
  coded (temporal-delta + spatial context model), ~50–150 B/frame → **30–90 KB**. THE BULK.
- **600 poses × 6 dims:** smooth trajectory, delta+entropy-coded → **~1–3 KB**.
- **minimal RGB carrier** to make SegNet/PoseNet emit the target cells (boundary-preserving + pose-
  preserving RGB; the cheapest member of each evaluator-equivalence class) → small.

| representation | B floor | rate term | S floor (d_seg≈0,d_pose≈ε) |
|---|---|---|---|
| current frontier (neural decoder) | 178 KB | 0.119 | 0.192 |
| neural-decoder, aggressive entropy coding (V1/V2 + L20–L32) | ~100 KB | 0.067 | **~0.07–0.10** |
| DIRECT GRAMMAR / evaluator-inverse (V3) | ~40–80 KB | 0.027–0.053 | **~0.03–0.06** |
| absolute Kolmogorov (video-specific, all slack exploited) | ~25–40 KB | 0.017–0.027 | **~0.02–0.03** |

**THE FINAL ULTIMATE THEORETICAL FLOOR ≈ 0.02–0.05** (direct grammar, all evaluator slack exploited),
vs the current 0.192 frontier — **~4–10× headroom, ALL on the rate term**, accessible ONLY via the
evaluator-inverse representation. No public entry (incl. gold 0.193) has reached it; neural decoders
plateau ~0.19 because the decoder weights ARE the rate floor. Going below REQUIRES dropping the
neural decoder for the direct grammar (the "deforestation to the skeleton").

(Tagged THEORETICAL ESTIMATE from the score law + read-surface, not a measured score. The waterfiller
replaces each estimate with exact ΔS as candidates land.)

---

## 2. THE ROADMAP — 5 phases from here to the floor

Through-line: the **evaluator-action waterfiller** (`tac.optimization.harvest_evidence` +
`evaluator_action_waterfill`). EVERY candidate (any vehicle) → archive → exact d_seg/d_pose/bytes →
`CandidateActionEvaluation` → admit iff ΔS<0 (pays rent). One currency, all phases.

### PHASE 0 — B1 root-cause diagnosis (NOW; $0 local MLX; gates everything)
The neural-decoder path is blocked until the renderer LEARNS. Run the RGB-reconstruction probe
(§0). Deliverable: a HiNeRV config whose exact d_seg DROPS over training (the first descending trend).
Fastest path to "is it the objective or the architecture?" Without this, V1 is dead weight.

### PHASE 1 — a WORKING neural-decoder base (V1 HiNeRV or V2 SNeRV; whichever descends first)
Apply the Phase-0 fix → train to a real descending exact-eval trend → dual CPU(Linux x86_64)+CUDA(T4)
exact eval. This is the "base" the waterfiller builds on. Target: a real S in the 0.2–0.5 range that
DESCENDS (proof the renderer fits the evaluator), then drives toward ~0.19 with the curriculum.
V2 SNeRV runs in parallel (C2 source-forward binding + C3 LF/HF byte-pressure → C4 exact eval); the
output_2 DROP is already proven (rent-optimal eliding).

### PHASE 2 — the RATE ATTACK (the dominant lever; PR95 L20–L32 entropy lessons)
Once d_seg≈small, d_pose≈small, S≈rate. Drive B down with the canonical byte-allocator primitives
(each is a waterfiller atom, admitted iff ΔS<0): monolithic 4-section grammar (L20), per-tensor
byte-maps (L21), CONV4 storage perms (L22), split brotli streams (L23), raw-LZMA latents (L24),
temporal-delta uint8 latents (L25), canonical-Huffman ranked sidecar (L26), per-pair single-dim
correction sidecar (L27, −0.001..−0.003 alone), decode-side channel postprocess (L28, 0 bytes),
fp16 per-tensor scales (L29), range/arithmetic coding (L30), colex-rank no-op (L31), brotli q11 (L32).
Target: B ~100–150 KB → S ~0.15–0.17.

### PHASE 3 — the EVALUATOR-INVERSE DIRECT GRAMMAR (V3; the path BELOW the neural-decoder floor)
The frontier-breaking move. Replace/augment the neural decoder with the DIRECT skeleton grammar
(operator "deforestation"): entropy-coded seg-argmax maps + pose trajectory + minimal RGB carrier,
emitted by inflate.sh as real RGB frames ENGINEERED to hit the evaluator cells (compliance: must be
real RGB the scorer runs on). Read-surface atoms already scoped (`scorer_read_surface_atoms.py`):
per-pixel argmax-margin tolerance map, even-frame-seg-free classifier, pose-null projection. Each
grammar section is a waterfiller atom. Target: B ~40–80 KB → S ~0.05–0.08.

### PHASE 4 — WATERFILLER COMPOSITION → THE FLOOR
Compose V1/V2/V3 atoms via commutator-aware greedy waterfilling (`candidate_queue` + `LoweringRace`):
admit each atom iff exact ΔS<0, base-bound to prevent drift; noncommutative atoms ordered by ΔS/byte.
Exploit ALL evaluator slack: argmax-margin interior-free pixels, pose-null half-head, even-frame
seg-free 600 frames, concave-pose-term ε. Target: S → the floor **~0.02–0.05**.

### PHASE 5 — AUTHORITATIVE DUAL-AXIS EVAL + SUBMISSION
Every floor candidate: dual CPU(Linux x86_64, the leaderboard axis) + CUDA(T4) exact eval on the EXACT
archive bytes (NEVER MPS/macOS-CPU for authority). Compliance gate (`pre_submission_compliance_check
--contest-final`). Submit to the contest iff it beats the public frontier on BOTH axes. Submission
escrow: keep the best dual-axis archive ready; ship before deadline risk, update on better replays.

---

## 3. Vehicle → phase map (parallelism)
- **V1 HiNeRV:** Phase 0 (diagnose) → Phase 1 (working base) → Phases 2/4 (rate + compose). BLOCKED on Phase 0.
- **V2 SNeRV:** Phase 1 (parallel base, C2/C3/C4) → Phases 2/4. Independent of V1's blocker.
- **V3 evaluator-inverse:** Phase 3 (the floor path) + the read-surface atoms feed Phase 4 NOW (primitives, not the live loop, per Hotz SEAL: live loop waits for a working Phase-1 base).
- **Waterfiller:** the through-line for Phases 1–5 (already real: harvest_evidence + trajectory mode).

## 4. Immediate next steps (ordered)
1. **Phase 0 B1 diagnosis** — RGB-reconstruction probe ($0 MLX). Settles the V1 blocker.
2. **V2 SNeRV C2/C3** — parallel base (not blocked on V1).
3. **Mark B1-R2 trend** — durable verdict (flat, INSPECT, diagnose; NOT kill).
4. **Fix launch-manifest fidelity** — emit canonical + scaled boundaries + totals (manifest bug).
5. Hold optimizer novelty (Aurora/Lion/comp-Muon) until a Phase-1 base exists; comp-Muon is V3/attention research.

## 5. Cross-cutting discipline (unchanged)
Rent law (admit iff ΔS<0); anti-drift base-binding; dual CPU+CUDA authority (never MPS/macOS-CPU);
tie-corrector on every authority decision; serializer + --expected-content-sha256 + 2-clean .py
review-gate; SSD disk hygiene, no /tmp evidence; NO FAKE; Forbidden premature KILL; submit only if
exact dual-axis S beats the public frontier.
