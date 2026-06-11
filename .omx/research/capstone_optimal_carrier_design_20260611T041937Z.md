# OPTIMAL per-pair CARRIER — math/geometry/info-theory design verdict (2026-06-11)

**Subagent:** `optimal_carrier_design`. **Operator grant:** full authority to design the optimal carrier(s)
then engineer the full stack; this memo is the research + design + crux foundation feeding the
orchestrator's synthesis. **Authority of every number:** `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`
— derived from MEASURED artifacts (the floor probe's 600-frame scorer pass, lever-B's 600-pair smoke,
the #57 exact-CPU pose-carrier RD sweep). NON-PROMOTABLE per the GOAL authority ladder. `$0` spend, NO
cloud, NO paid GPU, NO MPS, NO /tmp. `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`. Frontier read from pointer: **0.19109982 [contest-CPU], 177,169 B**.
**DID NOT TOUCH** the running d_seg daemon (`experiments/results/capstone_curriculum_b20_n48/`) nor the
sister builder's stored-latent files (`src/tac/capstone_vq_nerv/{vq_nerv_bundle,export,numpy_reference,inflate}.py`).

NO FAKE: every dimension/bit claim below is a MEASURED number from a cited artifact or a closed-form
derivation shown inline; every "loses/wins" is a measured/derived description-length or distortion cost,
not assertion. The one new measurement (the per-pair intrinsic-dimension split) is the committed $0 probe
`.omx/research/probes/carrier_intrinsic_dim_probe_20260610.py` (sha `ba7fc17f…`).

---

## 0. THE HEADLINE (the design verdict in one paragraph)

The per-pair carrier optimization is **NOT a choice between decomposition and rendering — it is a
capacity-relocation problem on a full-RGB amortized decoder.** Five audits + the floor measurement + this
probe converge on a single geometric fact: **frame1 must be a high-fidelity RGB frame** (PoseNet reads its
luma texture; a label-map/palette frame1 collapses pose to d_pose 12.14, MEASURED #57), AND **the cheapest
way to land 600 RGB frame-pairs in the scored cell is to AMORTIZE them into a decoder** (partition-direct
storage costs 0.169 rate, MEASURED-LOSES to the amortized 0.118, floor F3). Therefore the winning carrier is
**C1′ — a score-aware-retrained SMALLER full-RGB HNeRV-class decoder + a stored-float per-pair latent +
a DECOUPLED scalar pose-store**, where the pose-store exploits the probe's decisive finding that **pose has
intrinsic dimension 1.00** (dim-0 carries 99.80% of the trajectory variance; ~21 bits/pair). This is
lever C of `GOAL_standing_v3`, now with the SPECIFIC carrier geometry derived. Predicted byte budget
**~45–75 KB → rate 0.030–0.050** (vs frontier 0.107 decoder rate), a **2.1–3.6× rate win** that, combined
with the daemon's distortion closure, is the credible **sub-0.118** path (sub-0.15 is reachable by
distortion-closure at frontier bytes alone, DERIVED F5).

---

## 1. THE OPTIMIZATION (formalized) + THE MEASURED PER-TERM INFORMATION BUDGET

### 1.1 The objective

    S = 100·d_seg + √(10·d_pose) + 25·B/D       D = 37,545,489   (frozen, evaluate.py:92)

    archive B  =  B_decoder (shared, amortized)  +  B_carrier (per-pair)  +  B_pose (per-pair)  +  B_framing

The carrier-design problem: **minimize B subject to (d_seg, d_pose) holding the cell**, over the family of
{decoder architecture, per-pair carrier representation, pose representation}. The scored objects (frozen,
from `modules.py` read in full):
- **d_seg** = per-pixel argmax-flip RATE of frame1's SegNet 5-class argmax partition (384×512), vs GT.
  A SET functional on the argmax — invariant to the 80.67% scorer-null pixel energy (lever-G margin field:
  98.64% of px carry >0.5 logits free, 95.23% carry >2 logits free; MEASURED, lever-B §2).
- **d_pose** = MSE on PoseNet's first 6 of 12 output dims, on YUV6 of BOTH frames. GLOBAL pool, concave √.
- **rate** = linear in archive bytes; 25/D = 6.6586e-7 ΔS/byte.

### 1.2 The MEASURED per-pair information budget (the carrier-design number — NEW, this probe)

The decisive question for every carrier: **how many bits/pair does each scored term actually need?**
Computed from MEASURED artifacts (`carrier_intrinsic_dim_probe_20260610.py`):

| scored term | per-pair budget (MEASURED) | source | structure |
|---|---:|---|---|
| **d_pose** (tube 2.96e-5) | **20.8 bits/pair** (1,557 B / 600) | floor probe temporal-delta entropy F2 | **intrinsic dim = 1.00** (dim-0 = 99.80% of variance) — a SCALAR high-precision signal |
| **d_seg** per-pair refinement | **225 bits/pair** (16,888 B / 600, lever-B mod) | lever-B 600-pair smoke §4 | higher-dim, low-precision; refines a SHARED amortized base |
| **TOTAL score-DOF per pair** | **~246 bits/pair** (30.7 B/pair) | sum | — |
| frontier 28-d joint latent (MEASURED) | **201 bits/pair** (15,070 B / 600) | smaller-basis memo §1 | joint — exploits seg↔pose cross-structure |

**Three decisive readings:**
1. **Pose is a scalar, not a vector.** Participation ratio of the 6-dim pose variance spectrum = **1.004**;
   dim-0 (std 1.256, range −0.13..35.05) carries 99.80%, dims 1–5 (std 0.007–0.036, near-constant
   ego-motion) carry 0.20% combined. Pose needs **PRECISION (≈21 bits on one scalar), not dimensions**.
2. **The 8-bit VQ index is structurally too small for pose** (the capstone's measured wall): 8 bits < 21
   bits, AND 256 buckets cannot encode 600 distinct dim-0 ego-motions over a 35-unit range → the MEASURED
   d_pose 0.06–0.34 wandering (Quantizr-pose audit). This is the root cause, quantified.
3. **The frontier's 201-bit joint latent is BELOW the naive 246-bit sum** → the joint latent exploits
   seg↔pose cross-correlation through the shared decoder (the per-pair seg refinement and the pose scalar
   are not independent given the shared base). This is why a *jointly-trained* decoder beats a factorized
   one on rate — and why C3 (fully factorized) is rate-suboptimal even before the pose-blindness blocker.

### 1.3 The byte geography (where the bytes actually are — MEASURED)

Frontier 177,169 B decomposed (smaller-basis memo, MEASURED member-x):

| section | bytes | rate (25·B/D) | share | carrier role |
|---|---:|---:|---:|---|
| **decoder weights** | **161,104** | **0.10727** | **90.9%** | the amortizer (THE rate lever) |
| latent (28-d × 600) | 15,070 | 0.01003 | 8.5% | per-pair carrier (seg+pose joint) |
| sidecar/selector/tails/zip | 995 | 0.00066 | 0.6% | framing |

**THE DECODER IS THE CARRIER COST.** 91% of the archive is decoder weights. The per-pair latent is already
cheap (8.5%). **Any carrier optimization that does not shrink the 161 KB decoder cannot move rate
materially.** This re-frames the whole task: "design the optimal per-pair carrier" ⟹ "design the optimal
SHARED AMORTIZER (decoder) such that a tiny per-pair carrier suffices." The frozen 161 KB decoder is
lossless-exhausted (98.6% of iid Shannon, MEASURED) — the only rate lever is a SMALLER amortizer (lever C),
which is a TRAINING problem, not a coding problem (counting bound G2: you cannot regenerate 161 KB of
near-iid weights from a small seed).

---

## 2. RANKED CARRIER CANDIDATES — the crux of each (what makes it win or lose)

Ranked by `P(reaches sub-0.15) × P(reaches sub-0.118) / (build cost × risk)`, grounded in the measured
geometry. **The binding structural constraint that orders the list: frame1 must be high-fidelity RGB
(MEASURED #57 — palette frame1 → d_pose 12.14), and amortization beats direct storage on rate (MEASURED
F3 — 0.169 > 0.118).** Together these KILL the pure-decomposition carriers (C2/C3) on the SAME video and
elevate the smaller-RGB-decoder carriers (C1/C5).

### RANK 1 — C1′: score-aware-retrained SMALLER full-RGB decoder + stored-float latent + decoupled pose-store (WINNER)

This is lever C with the carrier geometry derived. It is C1 (the frontier's own carrier) at a SMALLER,
freshly-trained capacity, with the per-pair latent split per the measured intrinsic dims.

- **Decoder:** fresh-init (NOT memorized-point continuation — that DEGRADES, KILLED) HNeRV-class conv decoder
  trained against `α·B + β·d_seg + γ·√d_pose` with the PR95 non-negotiables (cosine LR + EMA + eval_roundtrip
  + diff-YUV6). Target parameter count **0.3–0.5× the frontier** (the §3 free-decoder-conditional
  intrinsic-dimension band: **amortized seg core ~20–55 KB**, DERIVED, smaller-basis §3).
- **Per-pair carrier:** the **stored 28-d float latent, temporal-delta + raw-LZMA coded** (PR95 L24/L25),
  ~12–15 KB / 600 pairs — content-rich (28 floats ≫ 8 bits), the frontier's PROVEN pose-capable carrier
  (reaches d_pose 2.9e-5). This is exactly what the SISTER BUILDER is testing.
- **Decoupled pose-store (the probe's lever):** because pose intrinsic dim = 1.00, OPTIONALLY pull the pose
  scalar OUT of the latent into a **separate 1,557 B temporal-delta pose-store** (FROZEN GT pose, like
  Quantizr), freeing the latent to carry seg-only refinement. Crux below decides joint-vs-split.

**CRUX (joint latent vs split pose-store):** the probe shows the frontier's joint latent (201 bits/pair) is
BELOW the naive sum (246 bits/pair) — the joint exploits cross-structure, so it is rate-efficient. BUT the
joint latent must be RICH ENOUGH for the 21-bit pose scalar (the capstone's 8-bit failure). **The decision:**
keep the **joint stored-float latent** (frontier-proven, rate-efficient) as the primary; the split pose-store
is the FALLBACK if the smaller decoder's joint latent can't hold the tube (it gives a guaranteed-tube pose
at +1,557 B, decoupling pose risk from the decoder-shrink risk). **Why C1′ wins:** it is the ONLY carrier
that (a) renders full-RGB frame1 (passes the #57 pose constraint), (b) amortizes (passes the F3 rate
constraint), and (c) shrinks the 161 KB decoder (the only material rate lever). Predicted: decoder 25–55 KB
+ latent 12–15 KB + pose-store 0–1.6 KB + framing 1 KB = **~40–72 KB, rate 0.027–0.048**.

**Predicted S decomposition (advisory, the campaign falsifies):**
- IF the smaller decoder holds the cell (d_seg→~5e-4, d_pose→tube) at 40–72 KB:
  S ≈ 100·(5.6e-4) + √(10·2.9e-5) + 25·(56,000)/D = 0.056 + 0.017 + 0.037 = **~0.110** (sub-0.118).
- IF distortion does not fully close at the smaller capacity (the honest risk, #71 joint-entanglement):
  the knee is where d_seg/d_pose re-enter the tube — the campaign measures it.

### RANK 2 — C1 at frontier capacity + DISTORTION closure (the sub-0.15 BANK, lower risk)

Keep the frontier's 161 KB decoder; do NOT shrink it; instead **close the distortion residual** (d_seg
0.008→5e-4 via the running daemon's curriculum; hold d_pose tube). This is the DERIVED sub-0.15 path:
frontier bytes already score **0.11797 at d_seg=d_pose=0** (F5), so any distortion-closure below the
current 0.073 residual lowers S. **CRUX:** sub-0.15 is a DISTORTION threshold at constant bytes (F5), NOT a
rate problem — the daemon's job. **Why RANK 2 not 1:** it does NOT win the innovation gate (it is the
frontier's own carrier, distortion-tuned) and does NOT reach sub-0.118 (rate unmoved). It is the DEFENSIVE
BANK + the gating measurement that tells us whether the small basis is d_seg-walled (Conclusion-1 fix). It
RUNS NOW (the daemon).

### RANK 3 — C5: pose-DECOUPLED carrier (split latent + scalar pose-store) — a C1′ VARIANT, promoted to its own rank for the crux

Pull pose entirely out of the latent: **[seg-only smaller latent] ⊕ [1,557 B scalar pose-store] ⊕ [smaller
RGB decoder]**. The probe's pose-intrinsic-dim-1.00 finding makes this attractive: pose is a scalar the
decoder need not learn to synthesize — store it, FiLM-condition the moving frame on it (Quantizr's proven
`JointFrameGenerator` mechanism, d_pose 0.00051). **CRUX (the decisive open question):** does a stored scalar
pose + FiLM-on-moving-frame reach the tube, OR does PoseNet need the pose to be IMPLICIT in the rendered
frame1 luma motion (not just FiLM-injected)? Quantizr's MEASURED d_pose 0.00051 says the FiLM-in-conv-block
mechanism WORKS — but only with a full stored 384×512 mask trunk (kilobytes/pair). **Why RANK 3:** it is the
cleanest pose-risk decoupling, but it carries the Quantizr mask-trunk byte cost (the audit flagged "a stored
384×512 mask per pair is kilobytes — re-derive the byte budget"). If the mask-trunk is needed, C5's rate
loses to C1′'s joint latent. C5 is the FALLBACK if C1′'s joint latent walls on pose at the smaller capacity.

### RANK 4 (DEAD on this video) — C2/C3: pure score-native decomposition (label-map frame1)

`[seg-argmax generator] ⊕ [pose-store] ⊕ [palette/label frame1]`. Lever B proved the seg generator is REAL
and CHEAP (d_seg 0.00826, 63,802 B, 2.54× smaller than the frontier seg-share). **But MEASURED-DEAD on
pose:** the palette/label frame1 has NO luma texture → PoseNet d_pose 12.14 (frame1 alone, #57 §3) → S 11.65
(#57 §4). **CRUX (why it loses):** frame1 carries a DUAL constraint — it must (a) land the SegNet argmax AND
(b) be a high-fidelity RGB for PoseNet — and the piecewise-constant palette satisfies (a) but is pose-blind
on (b). This is HNeRV-parity lesson 5 (full renderer, not single-component slot), MEASURED. **The seg
generator is NOT wasted:** it becomes a lever-G/H distortion-closure tool ON the RGB decoder (repair the
contiguous seg residual — #56 confirmed the residual is repairable), not a standalone frame1 carrier.

### RANK 5 (NEGATIVE, closed) — C4 pure pose-INR / frame0-warp; fixed-basis coding

- **C4 coordinate-INR pose carrier:** MEASURED RD ceiling **~0.0036 d_pose, NON-monotone in capacity** (#57
  §2 — 28× more bytes buys NO pose gain; the coordinate-MLP family cannot amortize 2-frame luma motion to
  the tube). The frame0-warp idea (store flow frame1→frame0) is downstream-blocked by needing a
  high-fidelity frame1 first. **A conv per-pair-latent decoder (HNeRV-class = C1′) is the structurally
  expressive carrier**, not a coordinate-INR.
- **Fixed-basis / free-PRNG-codebook coding of the frozen decoder:** MEASURED-EMPTY (smaller-basis G1: the
  decoder is near-iid, total correlation ≈ 0, so NO orthonormal basis compacts it; order-1 LOSES 14 KB,
  delta LOSES 43 KB). The free-inflate exploit pays ONLY fused into C1′'s forward map as a fixed-codebook VQ
  carrier on the RETRAINED (smaller) weights, never as a post-hoc transform.

---

## 3. THE ONE CARRIER TO ENGINEER (the byte budget + predicted decomposition)

**C1′ — score-aware-retrained SMALLER full-RGB HNeRV decoder + stored-float 28-d latent + (fallback) decoupled
scalar pose-store.** This is lever C / Conclusion-1+2 fused. It is the ONLY candidate that simultaneously
satisfies the three MEASURED constraints (full-RGB frame1 for pose; amortization for rate; smaller decoder for
the rate win) and wins the innovation gate (a fresh-init score-aware NAS carrier no leaderboard entry has).

| section | byte budget | rate | basis |
|---|---:|---:|---|
| smaller RGB decoder (fresh-init, score-aware) | 25,000–55,000 | 0.017–0.037 | §3 conditional-floor band (DERIVED) |
| stored-float 28-d latent (temporal-delta + LZMA) | 12,000–15,000 | 0.008–0.010 | frontier-proven, MEASURED |
| decoupled scalar pose-store (FALLBACK, if joint walls) | 0 or 1,557 | 0–0.001 | floor F2 MEASURED |
| sidecar/selector/framing | ~1,000 | 0.0007 | frontier MEASURED |
| **archive total** | **~40,000–72,000** | **0.027–0.048** | **2.1–3.6× rate win vs 0.107 decoder** |

**Predicted d_seg / d_pose:** the carrier's distortion is a TRAINING outcome (the campaign measures it). The
honest prediction band: d_seg ∈ [5.6e-4 (frontier-parity) , 0.008 (lever-B-smoke level)] depending on whether
the smaller decoder + daemon curriculum closes the residual; d_pose ∈ [2.9e-5 (tube, if joint latent holds) ,
3.6e-3 (the coordinate-INR ceiling, the risk if the conv decoder under-fits pose)]. **The decoupled pose-store
caps the pose risk at the tube** (+1,557 B), which is why it is the fallback: it converts "pose might wall" into
"+0.001 rate, pose guaranteed."

**Why C1′ beats the frontier's 0.118 rate:** the frontier pays 161 KB to be score-aligned on a MEMORIZED
point with capacity slack (#71: the frozen weights have no separable score-irrelevant subset — but that bounds
the FROZEN point, not the retrained floor). A fresh-init net trained to hold the cell concentrates capacity on
score-relevant directions; the §3 DERIVED band says the intrinsic score-aligned amortizer is **~25–55 KB**,
2.9–6.4× below the frontier decoder. The over-pay is the memorized point's slack — sheddable ONLY by
retraining (proven #71), which is exactly C1′.

---

## 4. THE FULL-STACK ENGINEERING PLAN

The carrier-INDEPENDENT recipe fixes (Conclusion-1, the 5 highest-EV fixes) are the GATE — they make the
advisory a trustworthy `inflate.sh→evaluate.py` predictor and unwall d_seg. They are landed/in-flight (the
daemon already has cosine-LR + EMA + curriculum per the synthesis memo). The C1′-specific build:

**Phase 0 — gating measurement (RUNNING NOW, do not disturb):** the daemon's 200ep constant-vs-cosine A/B on
the VQ-NeRV resolves "is the small basis d_seg-walled" (Conclusion-1). Harvest its trajectory.jsonl on exit.

**Phase 1 — carrier swap (the sister builder's lane):** replace the 8-bit VQ index with the stored-float 28-d
latent (temporal-delta + LZMA), in `src/tac/capstone_vq_nerv/`. This is the corrected pivot (the impoverishment
note's RANK-1). DO NOT duplicate — the sister owns `vq_nerv_bundle.py`/`export.py`/`numpy_reference.py`/
`inflate.py`. My design feeds their build; I do not edit those files.

**Phase 2 — decoder shrink (the lever-C campaign, the NEW build):** fresh-init train an HNeRV-class conv
decoder at 0.3–0.5× frontier params against `α·B + β·d_seg + γ·√d_pose`. Recipe (PR95 non-negotiables, all
landed):
- **Loss:** score-domain Lagrangian (NOT rel_err²). β·d_seg via the curriculum's 4-stage seg-loss + lever-G
  margin-weighted hinge (boundary pixels only — 95% interior is free); γ·√d_pose via differentiable
  rgb_to_yuv6 + eval_roundtrip (the diff-YUV6 patch + fail-closed assertion, Conclusion-3 C1).
- **Optimizer:** Muon final-stage (PR95 L15) + AdamW; **cosine LR per-stage-restart** (B1, the dominant
  d_seg-floor fix); **weight-EMA 0.997, export the SHADOW** (A1, non-negotiable).
- **Per-pair carrier:** stored-float 28-d latent; OPTIONALLY decoupled scalar pose-store + FiLM-on-moving-
  frame (Quantizr's `JointFrameGenerator` mechanism) as the pose-risk fallback.
- **MLX-first → numpy reference → torch parity** (the canonical_kernels Backend contract); detached nohup
  daemon + durable harvest waiter (NEVER session-bound).
- **Score the RELOADED int8 archive + BICUBIC inflate** (A2/A3 — close the advisory↔eval decouple E1) + the
  `inflate.sh→evaluate.py` smoke on a tiny real archive BEFORE any paid dispatch.

**Phase 3 — distortion closure on the rendered frames (levers G/H, stack AFTER the decoder):** the lever-B seg
generator's contiguous-residual repair (#56 confirmed repairable) becomes a ZERO-byte decode-time correction
(lever G, PR95-L28 precedent) OR a ≤few-KB postfilter (lever H) ON the RGB decoder's frame1 — closing the d_seg
residual without re-rendering.

**Phase 4 — byte-closure + ONE paired exact eval:** assemble the monolithic archive (frontier's 4-section
grammar), prove lossless parse-back parity, recompute advisory S from components; if advisory S beats the
frontier OR hits sub-0.15, launch ONE paired CPU+CUDA exact eval (~$0.6, within budget) — the authority row.

**Pre-registered KILL/DEFER:** if the smaller net's d_seg/d_pose cannot re-enter the tube at ANY capacity below
the frontier (the conditional-floor band proven unreachable by this architecture class), record the
architectural-ceiling band + reactivate via the decoupled pose-store (caps pose) + a larger seg-only decoder.

---

## 5. THE $0 PROBE I RAN (the crux it settled)

**Probe:** `carrier_intrinsic_dim_probe_20260610.py` (committed, sha `ba7fc17f…`). **Crux settled:** the
per-pair intrinsic-dimension split — how many bits/pair each scored term needs, and whether the frontier's
28-d latent is mis-allocated. **Result (MEASURED-derived):**
- **Pose intrinsic dim = 1.004** (participation ratio of the 6-dim variance spectrum); dim-0 = 99.80% of
  variance; pose needs ~21 bits/pair of PRECISION on one scalar, NOT 28 dims. → explains the 8-bit VQ pose
  wall (8 < 21 bits) quantitatively; justifies the decoupled scalar pose-store.
- **Seg per-pair refinement = 225 bits/pair** (over a shared amortized base) — the dominant per-pair carrier.
- **Frontier joint latent = 201 bits/pair < 246-bit naive sum** → the joint exploits seg↔pose cross-structure
  → a jointly-trained decoder is rate-efficient; C3 (fully factorized) is rate-suboptimal AND pose-blind.

**What is unprovable without a run (named):** C1′'s decoder-shrink-vs-distortion knee — whether 25–55 KB of
score-aware-trained decoder holds d_seg=5.6e-4 + d_pose=2.9e-5 — is a TRAINING outcome the §3 DERIVED band
predicts but cannot prove (Kolmogorov-uncomputable; #71 says the FROZEN point is entangled, which does not
bound the RETRAINED floor). The named run: **the lever-C fresh-init MLX retraining campaign (Phase 2)**, the
only >$1 lever, which the §3 band is the falsifiable prediction for.

---

## 6. WIRE-IN (Catalog #125) + SCOREBOARD

1. **sensitivity-map — ACTIVE:** the per-pair intrinsic-dim split (pose dim=1.00 scalar @21 bits; seg-mod
   @225 bits; decoder=91% of bytes) is the new top-level carrier-allocation prior. The aiming surface for
   rate is the SHARED DECODER (lever C), not the already-cheap per-pair latent.
2. **Pareto — ACTIVE:** adds the hard wall that pure-decomposition carriers (C2/C3) are DOMINATED on the same
   video (palette frame1 → d_pose 12.14 MEASURED; partition-direct 0.169 rate MEASURED-LOSES); the
   Pareto-feasible carrier is a smaller full-RGB amortizer.
3. **bit-allocator — ACTIVE:** the literal allocator is {smaller decoder 25–55 KB} + {stored latent 12–15 KB}
   + {scalar pose-store 0–1.6 KB}; pose gets bits-as-precision (1 scalar), seg gets bits-as-refinement.
4. **cathedral-autopilot — gate-conditional:** the C1′ campaign → (conditional advisory-beats-frontier) →
   ONE paired exact eval is the dispatch surface (Phase 4).
5. **continual-learning — ACTIVE:** reseeds the V3 judge: (a) pose is a scalar (dim=1.00), not a vector —
   the 8-bit VQ wall is a precision deficit not a dimension deficit; (b) frame1's dual constraint forbids
   the label-map carrier (lesson 5, MEASURED); (c) the decoder is 91% of bytes — carrier optimization IS
   decoder-shrink; (d) the joint latent is rate-efficient (exploits cross-structure).
6. **probe-disambiguator — RESOLVED:** "is the per-pair latent mis-allocated?" → pose is over-dimensioned /
   under-precisioned in a VQ index; "joint vs split pose?" → joint is rate-efficient, split is the pose-risk
   fallback; "which carrier?" → C1′ (smaller RGB decoder + stored latent + fallback pose-store).

**UPPER (vs T_1 sub-0.19):** unchanged (design memo, no archive). Frontier holds 0.19110.
**LOWER (the floor):** the C1′ predicted band rate 0.027–0.048 sits inside the §3 free-decoder-conditional
intrinsic-dimension floor (0.016–0.043), CONSISTENT — C1′ is the named carrier that targets that DERIVED
floor. The sub-0.118 door is C1′'s decoder-shrink; the sub-0.15 door is RANK-2 distortion-closure (the daemon).

## 7. CROSS-REFERENCES
`GOAL_standing_v3_20260610.md` (lever C = C1′; lever B = the seg generator demoted to a distortion tool) ·
`information_theoretic_floor_report_v1_20260610T102335Z.md` (F2 pose 1,557 B; F3 amortization beats direct;
F5 sub-0.15 = distortion) · `smaller_learned_basis_deep_math_20260610T191009Z.md` (§3 conditional floor
25–65 KB; G1 fixed-basis empty; G2 counting bound) · `score_native_pose_carrier_20260610T125000Z.md` (#57 —
frame1 dual constraint; coordinate-INR pose ceiling 0.0036 NON-monotone) · `score_native_first_candidate_…md`
(#56 — palette frame1 pose-collapse; residual repairable) · `lever_b_score_native_argmax_smoke_verdict_…md`
(seg generatable, 63,802 B; margin field interior budget) · `capstone_pr95_fullstack_definitive_audit_…md`
(Conclusion 1 recipe / Conclusion 2 carrier pivot) · `capstone_carrier_pivot_vq_index_impoverishment_…md`
(stored-latent corrected pivot) · `src/tac/optimization/evaluator_response_atlas.py` (the per-pair seg-margin
+ pose-Jacobian + joint-cone index) · `src/tac/null_space_exploiter/core.py` (80.67%-invisible byte subspace)
· `src/tac/sensitivity_map/axis_weights.py` (2.71× pose-marginal flip) · `carrier_intrinsic_dim_probe_…py`
(the $0 probe, this memo's new measurement) · `upstream/{modules.py,evaluate.py,frame_utils.py,README.md}`
(frozen authority, read in full).
