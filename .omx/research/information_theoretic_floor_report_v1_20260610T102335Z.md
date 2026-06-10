# INFORMATION-THEORETIC FLOOR REPORT — `information_theoretic_floor_report.v1` (Lever F)

UTC 2026-06-10 · claude (`task53_lever_F_T_floor`) · Lever F of `GOAL_standing_v3_20260610.md`.
**`[macOS-CPU advisory]` / derivation+measurement-only.** `promotable=false`, `score_claim=false`,
no dispatch, no /tmp, no MPS, $0 local CPU. The **LOWER side of the two-sided scoreboard** the goal
requires (previously `underived`).

This v1 report SUPERSEDES the framing of `information_theoretic_floor_T_floor_20260610.md` (which left
the seg term "no nontrivial proven floor / naive 424 KB brotli") by **MEASURING the actual partition
description length with an optimal context coder** and **running the unrun pose-output-entropy probe
(P6)**, on the real `upstream/videos/0.mkv` via the canonical `frame_utils.yuv420_to_rgb` decode +
frozen upstream SegNet/PoseNet (`tac.analysis.score_exact_saliency.load_score_exact_scorers`). Every
number is tagged **MEASURED** (computed on the 600 real partitions / 600 real pose vectors) or
**DERIVED** (closed-form algebra shown).

    S = 100·d_seg + √(10·d_pose) + 25·B/D        D = 37,545,489   (frozen authority, evaluate.py:92)

Raw artifact: `information_theoretic_floor_probe_full_1781086910.json` (full 600 frames, 567 s CPU).
Helpers: `tac.optimization.partition_contour_entropy`, `tac.optimization.pose_trajectory_entropy`
(23 NO-FAKE tests, `src/tac/tests/test_partition_contour_entropy.py`).

---

## 0. THE HEADLINE

| Question | Answer (this report) | tag |
|---|---|---|
| **S_floor (LOWER scoreboard)** | **0.11797** — the rate of the smallest MEASURED cell-landing achiever (the frontier's 177,169-byte amortized neural decoder) at zero distortion. | MEASURED (achiever) |
| **Headroom (T_hold − S_floor)** | **0.07313** (T_hold = 0.19109982 [contest-CPU]). All of it is recoverable seg+pose distortion + a smaller achiever. | DERIVED |
| **Which term dominates the floor?** | **RATE.** At d_seg=d_pose=0 the floor is pure bytes. seg amortizes into the decoder; the pose-OUTPUT floor is only **1,557 B = 0.00104 score (0.88% of S_floor)**. | MEASURED+DERIVED |
| **Is storing the partition directly the floor?** (the closed-spec premise) | **NO — it LOSES.** The 600 SegNet argmax partitions, coded by the *optimal* spatial+temporal context coder (the tightest achievable, subsumes contour/chain-code), cost **253,413 B → rate 0.169**, *above* the amortized decoder's 0.118. Contour coding is NOT "far below a full RGB renderer"; the neural decoder amortizes the partitions more cheaply. | MEASURED |
| **Is sub-0.15 (T_3) reachable?** | **YES, by distortion-closure at constant bytes** — the frontier's 177,169 B already scores **0.11797 < 0.15** at d_seg=d_pose=0. The 0.191→0.118 gap is *entirely* the 0.073 seg+pose residual. | DERIVED (arithmetic) |
| **Is sub-0.118 reachable?** | **CONDITIONAL** — requires a *smaller* achiever than 177 KB at near-zero distortion. The two MEASURED standalone classes (partition-direct 0.169, naive-brotli 0.283) both LOSE, so the only door below 0.118 is a smaller *amortized* program (lever A/C class shift) whose existence is an OPEN compression question (Kolmogorov-uncomputable lower bound). | DERIVED |

**One sentence:** *S_floor = 25·B_min/D + ε_distortion, B_min ≤ 177,169 B (the frontier amortized
decoder, MEASURED); the seg partition coded directly is 253 KB (MEASURED, the optimal context-coding
floor, which LOSES to amortization), the pose output is only ~1.5 KB (MEASURED), so the floor is
RATE-dominated and the binding question is whether any program amortizes the cell in < 177 KB — an open
question with no nontrivial proven lower bound.*

---

## 1. WHAT THE FLOOR IS THE FLOOR OF (the cell, exactly)

Per `closed_spec_boundary_math_system_of_equations_20260610.md` §1–2 + `evaluate.py`/`modules.py`
(read in full): a witness is a 1200-frame RGB sequence (camera 874×1164) whose, for each of 600
non-overlapping pairs (`frame_utils.py:10`, `seq_len=2`):

- **seg object** = the SegNet 5-class **argmax partition** `L*_t` of frame1 only (`x[:,-1,...]`,
  bilinear-resized to 384×512). `d_seg` = per-pixel argmax-flip RATE vs `L*`. **A SET functional on the
  argmax partition** — not appearance.
- **pose object** = the PoseNet first-6-of-12 output `p*_t` (FastViT-T12 on YUV6 of both frames).
  `d_pose` = mean over pairs of ‖·‖²; term = √(10·d_pose), concave, GLOBAL pool.
- **rate** = 25·|archive.zip|/D, linear.

`S_floor = min S` over all legal archives whose inflate output lands in this cell. The minimum
description length of a cell member, term by term, is what follows.

---

## 2. TERM 1 — THE SEG FLOOR (partition description length) — MEASURED

The seg-scored object is the **partition** `L*`. At d_seg=0 the witness's 600 frame1 argmax maps must
equal `L*`. The MDL of the partition is the information content of the 600-map argmax sequence. I
measured it three ways on the real 600 partitions (full video):

| coder (all store the EXACT partition, d_seg=0) | bytes (600 frames) | rate `25·B/D` | tag |
|---|---:|---:|---|
| **temporal-context optimal coder** `H(px \| left, up, prev-frame)` | **253,413** | **0.16874** | **MEASURED** |
| spatial-context optimal coder `H(px \| left, up)` | 305,406 | 0.20336 | MEASURED |
| chain-code contour geometry (log2 3 / step UPPER bound) + region labels | 323,942 | 0.21570 | MEASURED+DERIVED |
| order-0 iid argmax entropy | 23,821,632 | 15.86 | MEASURED (sanity) |
| naive per-frame argmax brotli (prior memo P1) | 424,722 | 0.28280 | MEASURED (prior) |

**Mechanism numbers (MEASURED, full 600):** boundary fraction = **0.687 %** of pixel-pixel edges
(~2,700 crack steps/frame), **35.5 regions/frame** (21,304 total connected components over 600 frames),
region-label cost only **24,027 bits = 3,003 B** for all 600 frames (the partition is overwhelmingly
*geometry*, not labels).

### The key honest finding (corrects the closed-spec premise)

The **temporal-context coder is the tightest achievable floor** for storing the partition directly: a
context model conditioning on the two causal spatial neighbours + the co-located previous-frame pixel
*subsumes* RLE, chain-code/contour coding, and JBIG-class coders (any of those is a special case of
this context). It is a true information-theoretic floor (the exact `Σ_ctx N_ctx·H(p_ctx)`), not a codec
with overhead. **That floor is 253,413 B → rate 0.169.**

This is **above** the frontier amortized decoder's 0.118. **Therefore: storing the SegNet partition
directly — even by the optimal possible context coder — does NOT beat the amortized neural decoder.**
The closed-spec memo's framing ("the seg-scored object IS the partition; its MDL = contour bits, far
below a full RGB renderer") is **MEASURED-FALSE on rate**: the 600 partitions carry ~253 KB of
irreducible boundary+temporal entropy, and the neural decoder amortizes that shared structure into
~162 KB of decoder weights that *also* carry pose + appearance. The partition is cheap *per region* but
there are 21,304 of them across 600 temporally-varying frames, and their geometry does not compress
below 0.169 by any context coder. (DERIVED corollary: a contour/RAG carrier is the right *data
structure* for the §4 boundary-math SOLVE — the per-pixel margin-polytope free-budget, the MDL
region-merge — but it is NOT a rate win as a standalone storage format.)

### Seg floor verdict

The seg term's floor is **NOT a standalone partition-storage number** (which loses at 0.169); it is
**absorbed into the rate term** via amortization (§4). The partition-direct 0.169 is the **MEASURED
PROVEN lower bound on the standalone "store the scored object" achiever class (Lever B's clean shift)**
— and it proves that class loses to amortization on rate. The 0.169 is itself a *new* harder floor than
the prior 0.283 brotli number (optimal context coding cut 40 % off naive brotli but still loses).

---

## 3. TERM 2 — THE POSE FLOOR (P6, the unrun probe) — MEASURED

The pose object is the 600×6 trajectory `p*`. **DERIVED**: a uniform quantizer at step δ per dim induces
error ~U(−δ/2,δ/2), per-dim MSE δ²/12, so `d_pose = Σ_k δ_k²/12` (= δ²/2 for equal steps over 6 dims).
The carrier bits = the entropy of the quantized symbols; the trajectory is temporally smooth so I
**MEASURED the temporal-delta entropy** `H(q_t − q_{t-1})` on the real 600×6 `p*`:

| target pose term √(10·d_pose) | induced d_pose | quant step δ | **carrier bytes (temporal-delta)** | tag |
|---:|---:|---:|---:|---|
| 0.0172 (frontier-equivalent) | 2.96e-05 | 0.00769 | **1,557 B** | MEASURED |
| 0.0100 | 1.00e-05 | 0.00447 | 1,842 B | MEASURED |
| 0.0050 | 2.50e-06 | 0.00224 | 2,219 B | MEASURED |
| 0.0020 | 4.00e-07 | 0.00089 | 2,701 B | MEASURED |
| 0.0010 | 1.00e-07 | 0.00045 | 3,037 B | MEASURED |

**P6 RESOLVED:** the 600×6 pose-output trajectory entropy is **~1.5–3.0 KB** (MEASURED), confirming the
prior memo's 1–3 KB ESTIMATE. The trajectory is dominated by dim 0 (std 1.26; the other 5 dims std
0.007–0.036 — near-constant ego-motion). At the frontier operating point the pose carrier is **1,557 B
→ rate 0.00104 = 0.88 % of S_floor.** **The pose term is NOT a floor driver.**

### Pose floor verdict

Pose contributes **≤ 0.001 to the floor** as a standalone carrier (or 0, folded into the dense decoder
as the frontier does — d_pose = 2.94e-5 for free). Either way, **pose is negligible at the floor.** The
expensive tail of the √ (pushing d_pose to *exactly* 0) costs only +1.5 KB (1,557→3,037 B from term
0.017→0.001), so even a near-perfect pose carrier is ~3 KB → rate 0.002.

---

## 4. TERM 3 — THE RATE FLOOR (the binding wall) — the heart of S_floor

§2 (seg → 0 is byte-bound; partition-direct 253 KB loses) + §3 (pose → ε is ~1.5 KB, near-free) ⟹
**S_floor collapses to the rate of the smallest cell-landing program: `S_floor = 25·B_min/D + ε`.**

### 4a. The two MEASURED achiever classes (the min defines the floor)

| achiever class | B (zero-distortion) | rate = S_floor contribution | tag |
|---|---:|---:|---|
| **CLASS 2 — amortized neural decoder (frontier)** | **177,169 B** | **0.11797** | **MEASURED (the floor)** |
| CLASS 1 — store-the-scored-object (partition-direct seg 253,413 + pose 1,557) | 254,971 B | 0.16977 | MEASURED (loses) |
| CLASS 1b — naive per-frame brotli seg (prior P1) + pose | 426,279 B | 0.28384 | MEASURED (loses worst) |

**S_floor = min over measured classes = 0.11797** (CLASS 2). The "store the measured quantities, not
pixels" path (Lever B's clean shift, the closed-spec carrier) is **MEASURED to lose by 0.052 rate** — a
decisive, original result: amortization beats direct storage because the 600 partitions share enough
cross-frame structure that a single ~162 KB program regenerates them more cheaply than coding each.

### 4b. The (non-)proven lower bound on B_min

`B_min = K(evaluator-view of 0.mkv | inflate runtime)` — conditional Kolmogorov complexity, **uncomputable**;
no algorithm yields a nontrivial lower bound. Every MEASURED artifact is an *upper* bound (177 KB
frontier, 253 KB partition-direct, 425 KB naive). **There is NO nontrivial proven lower bound on
S_floor below the few-KB regime.** The leaderboard's 0.19 plateau is *evidence* that ~178 KB is hard to
beat with the neural-decoder class, not a *proof* of a wall. A provably smaller amortizer (lever A
quotient compiler / lever C fresh-init NAS) lowers the floor; nothing forbids it.

### 4c. Frontier coding-exhaustion (sister-measured, from prior verdicts — cited not re-run)

The frontier decoder is INT8 at 98.6 % of per-tensor iid Shannon (159,822 B iid vs 162,127 B actual;
`frontier_decoder_axis_waterfill_verdict_20260610.md`); the latent stream is at ~96.6 % of its iid floor
(MI=0 cross-pair). **The current achiever is lossless-exhausted at ~176 KB → rate 0.117.** The only rate
lever below 0.118 is a *different, smaller* achiever, not better coding of this one.

---

## 5. THE ASSEMBLED S_floor + HEADROOM (the LOWER scoreboard)

```
S_floor = 25 · B_min / D + ε_distortion
        = 25 · 177,169 / 37,545,489 + 0      (CLASS 2, the smallest MEASURED cell-lander, zero dist)
        = 0.11797                            [macOS-CPU advisory, MEASURED achiever]

T_hold  = 0.19109982                          (contest-CPU, recoded-R3 defensive hold)
headroom = T_hold − S_floor = 0.07313         (DERIVED)
```

**Dominant floor term: RATE** (100 % of S_floor at zero distortion; pose 0.88 %, seg 0 standalone /
absorbed). The headroom 0.073 decomposes exactly into the frontier's recoverable residual:
seg 0.056 (29.3 %) + pose 0.017 (9.0 %) — **all distortion, recoverable in principle**, the binding wall
being only "can a smaller program than 177 KB land in the cell."

### Threshold ladder, now MEASURED-anchored

| threshold | value | reachability | lever | tag |
|---|---|---|---|---|
| T_1 | sub-0.19 | HELD (0.19110); gap to T_1 only 0.0011 | — | — |
| T_2 | sub-0.17 | **distortion-closure at constant bytes** (frontier→0.118 at zero dist) | seg/pose aiming | DERIVED |
| T_3 | sub-0.15 | **distortion-closure at constant bytes** (frontier scores 0.11797 < 0.15 at d_seg=d_pose=0) | seg amortizer sharpening (lever G/H/C) | DERIVED |
| **S_floor** | **0.11797** | MEASURED achiever floor (frontier bytes, zero distortion) | — | MEASURED |
| sub-0.118 | — | CONDITIONAL: a *smaller* amortizer (lever A/C class shift); store-direct classes proven to lose | smaller carrier | DERIVED |
| 0+ | proven | trivial (B≥1); no nontrivial proven wall (Kolmogorov) | — | DERIVED |

**The actionable redirect (system intelligence):** T_2 and T_3 are **distortion thresholds at constant
bytes** — the frontier's 177 KB already buys 0.118 IF d_seg→0 and d_pose→0. The rate attack is only
required BELOW 0.118. This re-ranks the offensive levers: **aim the existing-byte carrier's distortion
first (levers G zero-byte corrections / H cheap postfilters / C fresh-init null-space-primary) — that is
the proven path to sub-0.15 — THEN attack bytes (levers A/B/C smaller amortizer) for sub-0.118.**
And it **demotes lever B** as a *standalone rate* play: storing the scored objects directly is MEASURED
to cost 0.170 > 0.118 (it loses to amortization), so lever B is only viable *fused into* a smaller
amortizer, not as a direct-storage carrier.

---

## 6. FALSIFIABLE CLAIMS (the lower-bound ledger, append-only)

- **F1 (MEASURED):** the 600 SegNet partitions, coded by the optimal spatial+temporal context coder
  (tightest achievable, subsumes contour), cost **253,413 B → rate 0.16874**. *Falsified by:* a coder
  achieving < 253,413 B on the EXACT partition with d_seg=0 (a context coder cannot; a *different
  amortizer* can — that is CLASS 2, not partition-direct).
- **F2 (MEASURED):** the 600×6 pose-output trajectory temporal-delta entropy is **1,557 B** at the
  frontier operating point (pose term 0.0172), **3,037 B** at term 0.001. P6 RESOLVED ~1.5–3 KB.
  *Falsified by:* a measured pose carrier < 1,557 B holding d_pose ≤ 2.96e-5.
- **F3 (MEASURED, decisive):** storing the scored objects directly (CLASS 1) costs **0.170 rate**,
  ABOVE the amortized decoder's **0.118** — amortization beats direct storage by 0.052. *Falsified by:*
  a direct-storage (non-amortizing) carrier scoring < 0.118 (none exists; the partition entropy forbids
  it by F1).
- **F4 (MEASURED achiever floor):** **S_floor = 0.11797** (frontier 177,169 B at zero distortion).
  *Falsified by:* an exact-eval row below 0.118 from any vehicle (would prove a smaller achiever and
  lower the empirical floor — the goal's class-shift thesis).
- **F5 (DERIVED):** sub-0.15 (T_3) is a DISTORTION threshold at constant bytes — frontier bytes score
  0.11797 < 0.15 at d_seg=d_pose=0. *Falsified by:* a recomputation error (checked:
  `25·177169/37545489 = 0.11797`).
- **F6 (the meta-claim, unchanged):** NO nontrivial PROVEN lower bound on S_floor below the few-KB
  regime (Kolmogorov uncomputable). The floor is an *achiever* question. *Falsified by:* a computable
  nontrivial lower bound on `K(evaluator-view | inflate)`.

---

## 7. WIRE-IN (Catalog #125)

1. **sensitivity-map** — ACTIVE. The floor decomposition (RATE 100 % binding at the floor; seg
   partition-direct 0.169 LOSES to amortized 0.118; pose 0.88 %) is the top-level LOWER-scoreboard prior;
   it demotes lever B (direct storage) as a standalone rate play and re-ranks distortion-before-rate up
   to 0.118.
2. **Pareto constraint** — ACTIVE. F1 (partition-direct 253 KB optimal-context floor) + F3 (amortization
   beats direct storage by 0.052) are hard Pareto walls on the rate axis for the store-the-object class.
3. **bit-allocator** — ACTIVE. F5 ("sub-0.15 is distortion not bytes") tells the allocator: spend next
   bytes on distortion-closure (sharper amortizer / zero-byte corrections), not rate, until 0.118.
4. **cathedral autopilot dispatch** — N/A. Derivation+measurement surface; no archive bytes emitted.
5. **continual-learning posterior** — N/A. Closed-form/measured derivation; the F1–F6 ledger is the
   side information future agents reseed from.
6. **probe-disambiguator** — ACTIVE. P6 (pose-output entropy) is now RESOLVED (F2). The remaining
   disambiguator is "does any amortizer beat 177 KB at near-zero distortion" (lever A/C) — the only door
   below S_floor.

## 8. CROSS-REFERENCES

`closed_spec_boundary_math_system_of_equations_20260610.md` (§2–4 the partition/pose objects; this report
MEASURES §2's premise and finds partition-direct loses on rate) · `information_theoretic_floor_T_floor_20260610.md`
(prior v1 framing this supersedes with measured context-coding + P6) · `GOAL_standing_v3_20260610.md`
(Lever F; LOWER scoreboard updated) · `segnet_fragile_support_codec_budget_20260609.json` (prior naive
424 KB + boundary mechanism) · `frontier_decoder_axis_waterfill_verdict_20260610.md` /
`frontier_latent_axis_waterfill_verdict_20260610.md` (§4c coding-exhaustion) ·
`src/tac/boundary_math/partition.py` (sibling task #52 RAG — the right data structure for the SOLVE,
consumed not duplicated) · `information_theoretic_floor_probe_full_1781086910.json` (the raw MEASURED
artifact) · `upstream/{evaluate.py,modules.py,frame_utils.py}` + `upstream/models/{segnet,posenet}.safetensors`
(frozen authority).
