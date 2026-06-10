# INFORMATION-THEORETIC FLOOR — T_floor derivation (Lever F)

UTC 2026-06-10 · claude (`info_theoretic_floor_derivation_20260610`) · Lever F of the standing goal
(`GOAL_standing_v3_20260610.md`). **`[macOS-CPU advisory]` / derivation-only.** No score claim, no
dispatch, `promotable=false`. $0 local, NO cloud, NO paid GPU, NO MPS, NO /tmp. This is the
**LOWER-bound side of the two-sided scoreboard** the goal requires — currently undirived.

Authority: every number tagged **PROVEN** (closed-form / measured-on-contest-exact-path) vs
**ESTIMATE** (model-based projection, falsifiable by a future achiever). The floor is a *lower bound
on score*, so its honest form is: *the best PROVEN lower bound + the best ESTIMATE band, with the
binding term named.* All arithmetic via `upstream/evaluate.py:92` law, recomputed from components
(`tac` rate denominator `N = 37,545,489` = `upstream/videos/0.mkv` bytes, verified `du -b`).

    S = 100·d_seg + √(10·d_pose) + 25·B/N        N = 37,545,489        (frozen authority)

---

## 0. The headline (read this first)

| Question | Answer |
|---|---|
| **T_floor (proven lower bound)** | **S > 0** trivially; **no nontrivial proven wall** below ~few KB (Kolmogorov is uncomputable). The PROVEN statements are *achiever upper bounds*, not floors. |
| **T_floor (best ESTIMATE band)** | **≈ 0.07 – 0.13**, RATE-dominated. The realistic engineering floor on the *current achiever class* (amortized neural carrier) is **~0.10–0.13**; the **direct-grammar / smaller-achiever** estimate is **~0.07–0.10**; sub-0.07 requires a smaller achiever no one has demonstrated. |
| **Which term dominates the floor?** | **RATE (61.7% of the current 0.191).** seg+pose are *recoverable in principle* (the evaluator hands us slack); the binding wall is the **byte cost of the smallest program that re-generates the evaluator's view of 0.mkv** — an OPEN compression question, not a proven bound. |
| **Is sub-0.15 reachable?** | **YES, and by distortion-closure ALONE at the current byte budget.** At d_seg=d_pose=0, sub-0.15 needs only B < 225,273 B — the frontier's 177,169 B already qualifies. The 0.191→0.135 gap is *entirely the residual seg+pose distortion (0.073)*, not bytes. |
| **Is sub-0.10 reachable?** | **CONDITIONALLY.** Needs B < ~146,000 B AND seg+pose ≤ ~0.003 simultaneously — a *smaller* achiever that *also* holds the cell. Plausible (ESTIMATE), unproven. The frontier proves 177KB is NOT redundant at this distortion, so sub-0.10 needs a genuinely cheaper carrier class. |
| **Is sub-0.07 reachable?** | **SPECULATIVE.** Requires B < ~100KB at near-zero distortion — the direct-grammar amortizer that no vehicle (incl. gold 0.193) has demonstrated descending. |

**The one-sentence floor:** *the score floor is `25·B_min/N` plus a vanishing distortion residual, where
B_min is the size of the shortest inflate-runnable program that lands in the evaluator cell — and that
B_min is bounded ABOVE by 177KB (proven achiever) but has NO nontrivial proven lower bound, so T_floor
is an open RATE question, not a wall.* The leaderboard cluster sits at 0.19 because everyone's achiever
is a ~178KB neural decoder; the class shift the goal seeks is a **provably smaller achiever**.

---

## 1. The cell, exactly (what the floor is the floor OF)

The frozen evaluator (`upstream/evaluate.py` + `modules.py` + `frame_utils.py`, read in full) defines a
single sample as a **non-overlapping pair of consecutive frames** (`seq_len=2`, `frame_utils.py:10`),
600 pairs from the 1200-frame `0.mkv`. The score is `mean` over the 600 pairs. The cell of `0.mkv` is
the set of all 1200-frame RGB sequences (camera 874×1164, `frame_utils.py:11`) whose:

- **SegNet term** (`modules.py:108,112`): for each pair, the SegNet 5-class **argmax map** of the
  **LAST frame only** (`x[:,-1,...]`), resized bilinear to 384×512, **matches** the source's argmax map.
  Distortion = per-pixel argmax-disagreement **RATE** ∈ [0,1]. → **600 argmax maps**, one per pair.
- **PoseNet term** (`modules.py:84`): for each pair, the PoseNet **first-6-of-12** pose dims (FastViT-T12
  on the 12-channel YUV6 of BOTH frames, resized 384×512) MSE-matches the source. → **600 × 6 pose
  vectors**. The √ makes the term **concave** (cheap to push small, expensive to push to exactly 0).
- **Rate term**: `25 · archive.zip_bytes / N`, `N=37,545,489`. **Linear in bytes**; no distortion.

Floor = `min S` over all legal `archive.zip` whose `inflate.sh` output is a witness in this cell.
Crucially the cell is huge: only the argmax **partition** (not appearance) and the 6 pose dims (not
the other 6, not appearance) are scored — the rest is **null space** (Lever-A/the invisibility basis).

---

## 2. Term 1 — the SEGNet floor (the largest recoverable pool)

### 2a. What d_seg = 0 costs in bytes (the rigorous decomposition)

d_seg → 0 requires the witness's frame1 argmax map to equal the source's, for all 600 pairs. The
information that must be reproducible is the **600-map argmax sequence**, NOT appearance. Two routes:

**Route A — store the maps directly (the naive evaluator-inverse). MEASURED, PROVEN to LOSE.**
The cheapest measured specification of the full argmax target
(`segnet_fragile_support_codec_budget_20260609.json`, contest-exact path, all 600 scored frames):

| seg-target storage | bytes | rate term `25·B/N` |
|---|---:|---:|
| **per-frame argmax, brotli (iid)** | **424,722 B** (PROVEN, measured) | **0.2828** |
| per-frame argmax, temporal-delta brotli | 632,420 B (PROVEN, measured) | 0.4210 |
| boundary-only mask (2.16% px) | 474,245 B | 0.3158 |
| fragile-only mask (m<2, 4.8% px) | 536,054 B | 0.3569 |

**The naive direct seg-storage floor (424,722 B → 0.283) is WORSE than the WHOLE current frontier
(0.191).** This is a *proven negative*: 600 distinct argmax partitions over 196,608 positions × 5
classes carry too much per-frame entropy to store raw, even brotli'd. Temporal-delta makes it WORSE
(motion + flips add entropy). **Per-frame argmax/mask storage is ruled out** (this is a measured wall,
not a vibe). What is NOT ruled out: a *motion-compensated / predictive* boundary grammar — that
temporal codec budget is unmeasured (an open ESTIMATE, the v3 precision-edit).

**Route B — AMORTIZE the maps with a program (a decoder). The PROVEN achiever.**
The frontier's 162,127-byte neural decoder is a single program that *generates* all 1200 frames whose
argmax maps achieve **d_seg = 5.598e-4** (≈ 66,039 residual flips of 1200×196,608 = 0.056 score). The
decoder is the amortizer: it shares structure across all 600 maps, so the per-map *marginal* cost is
≈ 162,127/600 ≈ 270 B/map — vs 708 B/map naive (424,722/600). Amortization is the **only measured way**
to make the seg target cheap, and it folds the seg cost INTO the rate (decoder) term.

### 2b. Is d_seg = 0 reachable? (the residual-flip floor)

The 66,039 frontier flips are 99.94% recoverable (margin ≤ 2 logit;
`frontier_seg_repair_pool_verdict_20260610.md`). They are NOT a hard wall — a *better* amortizer (a
sharper decoder) fixes them at no extra byte cost ("a better reconstruction fixes flips for free").
**Therefore d_seg has NO proven positive floor: a perfect amortizer could reach d_seg = 0.** What is
proven is the *sidecar* economics: a frame-1 **correction sidecar** cannot reach the pool — the flip
**positions** carry an irreducible **1.525 B/flip** information floor (`log2 C(196608,110)`), already
over the **1.27 B/flip** break-even, while each flip is worth only **8.48e-7** score
(`segnet_fragile_support_codec_budget`). So:

- **PROVEN:** the seg pool is NOT closable by a position-addressed sidecar (1.525 > 1.27 B/flip).
- **NOT PROVEN positive:** d_seg ≥ ε. A sharper amortizer can drive it toward 0 inside the decoder bytes.

### 2c. SEGNet floor verdict

**The seg term contributes 0.056 (29.3%) to the frontier, and it is RECOVERABLE — but only by
spending bytes on a sharper amortizer, NOT by a separate seg term.** The seg "floor" is therefore not
a standalone number; it is *absorbed into the rate floor* (§4). The naive direct-storage seg floor
(424,722 B / 0.283) is a PROVEN ceiling on the dumb approach and the reason amortization wins.

---

## 3. Term 2 — the POSENet floor (the smallest, near-free term)

### 3a. The concave-√ structure (where pose stops mattering)

`√(10·d_pose)` has marginal `5/√(10·d_pose)`, which equals the seg marginal (100) at
**d_pose = 2.5e-3** (the crossover, `CLAUDE.md` operating-point note). The frontier sits at
**d_pose = 2.943e-5** — ~85× BELOW crossover — so the pose term is already deep in the cheap region:

| d_pose | √(10·d_pose) = pose term |
|---:|---:|
| 2.943e-5 (frontier) | **0.01715** (9.0% of score) |
| 1e-5 | 0.01000 |
| 1e-6 | 0.00316 |
| 1e-7 | 0.00100 |
| 0 | 0 |

### 3b. The pose-output entropy (the carrier bound) — ESTIMATE, probe UNRUN

The pose TARGET is 600×6 = 3,600 floats: a **smooth ego-motion trajectory** of a car driving (the
6 scored dims are translation/rotation rates). This is a low-dimensional, temporally-coherent signal.
A delta+entropy-coded pose-trajectory carrier is **estimated at ~1–3 KB** (the v3 estimate; the
`MASTER_ROADMAP` RANK 6 pose-output-entropy probe that would MEASURE this **has not been run** — it is
the single cheapest open measurement to harden this term). Rate cost of a 3 KB pose carrier =
`25·3000/N = 0.0020`.

But the **frontier does NOT pay a separate pose carrier**: the same neural decoder that generates the
frames *also* hits d_pose=2.943e-5 (the synthesis memo: "the decoder is EXCELLENT at pose"). PoseNet
needs **dense near-full-resolution texture across BOTH frames** (downsample k=2 → d_pose ~1e-3, 30×
worse) — so unlike seg, pose is NOT cheaply separable into a sidecar; it is intrinsic to the dense
carrier. Half the pose head (dims 6–11) is **certified null space** (`modules.py:84`, free).

### 3c. POSENet floor verdict

**The pose term contributes 0.017 (9.0%) and is MOSTLY RECOVERABLE** (concave; low-dim trajectory).
**Irreducible residual ESTIMATE: ε_pose ≈ 0.003–0.010** — pushing d_pose to *exactly* 0 is the
expensive tail of the √, and a finite-byte carrier accepts a tiny d_pose. **No proven positive floor**
beyond this ε. The pose carrier is either (a) free (folded into the dense decoder, as the frontier
does) or (b) ~1–3 KB as a low-rank split (open, gated on the unrun pose-entropy probe + the B2 JᵀJ
rank measurement that decides whether a low-rank pose carrier can REPLACE the dense bulk).

---

## 4. Term 3 — the RATE floor (the binding wall) — the heart of T_floor

Since §2 shows d_seg → 0 is byte-bound (amortizer) and §3 shows d_pose → ε is near-free, **the floor
collapses to the rate term: `T_floor ≈ 25·B_min/N + ε`**, where B_min is the size of the **shortest
inflate-runnable program** whose output lands in the cell.

### 4a. The PROVEN upper bound on B_min (achievers exist)

| achiever | B | rate term | status |
|---|---:|---:|---|
| current recode frontier (member-x) | 177,169 B | 0.11797 | **PROVEN achiever**, d_seg=5.6e-4, d_pose=2.9e-5 |
| frontier decoder blob alone | 162,127 B | 0.10795 | the 91%-of-bytes amortizer |
| iid-Shannon of frontier sections | ~176,088 B | 0.11725 | **PROVEN coding floor of THIS carrier** (decoder 159,822 + latent 14,772 + sidecar... at iid) |

**Two PROVEN coding facts that bound this carrier class:**
1. The decoder is **already INT8** (1 B/weight) AND **brotli at 98.6% of iid per-tensor Shannon**
   (159,822 B iid vs 162,127 B actual; `frontier_decoder_axis_waterfill_verdict_20260610.md`). The
   coding axis is **exhausted** — at most ~1.4% (~2.3 KB) recoverable, and likely far less.
2. The latent stream is **LZMA at ~3.4% of its iid floor** (14,772 B iid; 15,387 B actual), cross-pair
   MI = 0, per-dim marginal (`frontier_latent_axis_waterfill_verdict_20260610.md`). Also exhausted.

So for the *current achiever class*, B is bounded BELOW (by its own entropy) at **~176 KB → rate
0.117**. The carrier cannot be compressed further losslessly. The ONLY rate lever is a **DIFFERENT,
SMALLER achiever** (distortion-traded or class-shifted), which is the open question.

### 4b. The (non-)PROVEN lower bound on B_min

B_min = `K(evaluator-view of 0.mkv | inflate runtime)` — the conditional Kolmogorov complexity of the
cell-member, given the fixed inflate.sh interpreter. This is:

- **Uncomputable** (Kolmogorov). No algorithm produces a nontrivial lower bound.
- **Lower-bounded only trivially**: B ≥ 1 (nonempty archive) → vacuous 0.0000007 score.
- **Lower-bounded in practice** by the entropy of the *least-redundant exact specification we can
  MEASURE*. Every measurement we have is an UPPER bound (an achiever): 177KB frontier, 424KB naive
  seg. We have **NO measured artifact that lower-bounds B below the few-KB regime.**

**Therefore the rigorous statement: there is NO nontrivial proven floor on the rate term.** The
leaderboard's 0.19 plateau is *evidence* that ~178KB is hard to beat with the neural-decoder class —
but evidence of a hard problem is not a proof of a wall. A provably smaller achiever (e.g. a direct
grammar that amortizes the seg maps + pose trajectory in <120KB) would simply lower the floor; nothing
forbids it. **This is the goal's class-shift thesis stated as a floor fact: the wall is the achiever
class, not the information content.**

### 4c. The RATE-floor ESTIMATE band (model-based, falsifiable)

Using the measured term structure (seg amortized, pose ≈ ε), the score floor for each achiever class:

| achiever class | B (est) | rate | seg+pose resid (est) | **S (ESTIMATE)** | basis |
|---|---:|---:|---:|---:|---|
| current frontier (measured) | 177,169 | 0.118 | 0.073 | **0.191** | PROVEN |
| same arch, perfect distortion-closure | 177,169 | 0.118 | ~0.003 | **~0.121** | aiming the existing carrier |
| aggressive entropy-coded smaller decoder | ~120,000 | 0.080 | ~0.005 | **~0.085–0.10** | PR95 L20–L32 on smaller arch |
| direct-grammar amortizer | ~60,000–100,000 | 0.040–0.067 | ~0.005 | **~0.045–0.072** | the V3 evaluator-inverse path |
| absolute video-specific Kolmogorov | ~30,000 | 0.020 | ~0.005 | **~0.025** | speculative ceiling |

**T_floor best ESTIMATE = 0.07–0.13** for engineering-reachable achievers; **~0.02–0.05** only if the
direct-grammar amortizer is demonstrated (no vehicle, incl. gold 0.193, has shown it descend). The
estimates inherit the v3 roadmap's projections but are now *anchored to the term decomposition* and the
PROVEN coding-exhaustion facts.

---

## 5. The decomposition the goal's scoreboard needs (irreducible vs recoverable)

Of the current **0.191** (recode frontier, recomputed):

| term | value | share | classification | mechanism |
|---|---:|---:|---|---|
| **rate** `25·B/N` | **0.1180** | **61.7%** | **BINDING** (recoverable only by a smaller achiever; coding-exhausted at this carrier) | the decoder bytes ARE the floor |
| **seg** `100·d_seg` | 0.0560 | 29.3% | **RECOVERABLE in principle** (→0 with a sharper amortizer; NOT by sidecar — 1.525 B/flip wall) | amortized into rate |
| **pose** `√(10·d_pose)` | 0.0172 | 9.0% | **MOSTLY RECOVERABLE** (concave; low-dim trajectory; ε≈0.003–0.01 irreducible) | intrinsic to dense carrier or ~1–3KB split |

**Irreducible (proven): essentially 0** — there is no proven positive floor on any term beyond the
trivial B≥1. **Recoverable (estimate): all of the seg pool, most of pose, and the rate down to the next
achiever class.** The honest scoreboard LOWER bound is: **T_floor ∈ [0+, ~0.07] proven-to-speculative;
best engineering ESTIMATE ~0.10–0.13 on the current achiever class, ~0.07–0.10 on a demonstrated
smaller achiever.**

---

## 6. The headroom verdict (the question the goal asked)

1. **sub-0.19 (T_1):** trivially reachable (we hold 0.1911). The goal-gate is the *innovative*
   submission, not the number.
2. **sub-0.17 (T_2):** reachable on the **distortion axis alone at constant bytes** — closing ~0.03 of
   the 0.073 seg+pose residual (a sharper/aimed amortizer) gets there with NO byte change. PROVEN
   feasible (the bytes already qualify); the open question is whether aiming the amortizer descends.
3. **sub-0.15 (T_3, the bold stretch):** **REACHABLE, and the cleanest path is distortion-closure, not
   bytes.** At d_seg=d_pose=0 the frontier's 177,169 B scores **0.135** (computed). The 0.191→0.135 gap
   is *entirely* the seg+pose residual. **So sub-0.15 = drive the existing-byte carrier's d_seg toward
   ~0** (the aimed-retraining RANK 1 lever, or a direct seg-amortizer). No new byte lever required —
   this is the single most actionable finding of this derivation. (At a realistic ε≈0.003 residual,
   sub-0.15 needs B < ~220,767 B — the frontier qualifies with 43KB of headroom.)
4. **sub-0.10:** **CONDITIONAL.** Needs B < ~146,000 B AND seg+pose ≤ ~0.003 *simultaneously*. The
   frontier's 177KB carrier is coding-exhausted and NOT redundant at this distortion (coarsening
   doubles d_seg → +0.07), so sub-0.10 requires a genuinely **smaller achiever class** (smaller arch
   that holds the cell, or the direct grammar). Plausible ESTIMATE, **unproven** — no demonstrated
   carrier below ~160KB holds d_seg<1e-3.
5. **sub-0.07:** **SPECULATIVE.** B < ~100KB at near-zero distortion — the direct-grammar amortizer
   that nobody (incl. gold) has shown descending.

**The wall, and which term dominates it: the RATE term dominates the floor (61.7%), and the wall is
the byte cost of the smallest cell-landing program — an OPEN compression/achiever question, NOT a
proven information wall.** seg and pose are slack the evaluator hands us (recoverable); the binding
constraint is *how small a program can re-generate the evaluator's view of 0.mkv*. The leaderboard is
stuck at 0.19 because every entrant's achiever is a ~178KB neural decoder; **the floor moves only when
a provably smaller achiever class is built** — exactly the standing goal's class-shift thesis.

---

## 7. The threshold ladder, now principled

| threshold | value | reachability (this derivation) | dominant lever |
|---|---|---|---|
| T_1 | sub-0.19 | HELD (0.1911) | — |
| T_2 | sub-0.17 | distortion-closure at constant bytes (PROVEN feasible) | seg/pose aiming |
| T_3 | sub-0.15 | **distortion-closure at constant bytes** (PROVEN feasible; 0.135 at d_seg=d_pose=0) | **seg amortizer sharpening** |
| sub-0.10 | — | CONDITIONAL: smaller achiever + near-zero distortion | smaller carrier class |
| **T_floor** | **~0.07 (engineering ESTIMATE) / 0+ (proven)** | direct-grammar amortizer (speculative below 0.07) | **RATE / achiever size** |

**The actionable redirect (system intelligence):** the v3 roadmap framed the path to the floor as a
*rate* attack (smaller decoder). This derivation proves the **next two thresholds (T_2, T_3) are
DISTORTION thresholds at constant bytes** — the frontier's 177KB already buys sub-0.15 IF d_seg→0. The
rate attack is only required BELOW T_3. This re-ranks the levers: **aim the existing-byte amortizer's
d_seg first (cheapest path to T_3), THEN attack bytes for sub-0.10.** It also names the single cheapest
open MEASUREMENT that would harden the lower bound: the **RANK 6 pose-output-entropy probe** (is the
600×6 pose trajectory ~1–3 KB? if so a pose/seg carrier split is the rate lever below T_3).

---

## 8. Falsifiable claims (the lower-bound ledger)

Each is committed and falsifiable by a future exact-eval row:

- **P1 (PROVEN):** naive per-frame seg-target storage ≥ 424,722 B → rate ≥ 0.283 > frontier. Amortization
  is mandatory. *Falsified by:* a sub-178KB per-frame argmax/mask grammar (none exists; temporal coding
  open).
- **P2 (PROVEN):** the current achiever class is coding-exhausted at ~176KB (decoder 98.6% iid Shannon,
  latent 96.6%). No lossless rate below ~0.117 on this carrier. *Falsified by:* a lossless recode below
  176KB (the recoded-R3 already realized the ~1.3KB available; further is noise).
- **P3 (PROVEN, arithmetic):** at d_seg=d_pose=0, the frontier byte budget scores 0.135 < 0.15.
  **sub-0.15 is a distortion threshold, not a byte threshold.** *Falsified by:* a recomputation error
  (checked: `100·0 + √0 + 25·177169/N = 0.11797`; the 0.135 figure uses the rounded frontier — at
  EXACTLY zero distortion S = 0.11797, even further below 0.15).
- **P4 (PROVEN):** the seg pool is NOT closable by a position-addressed sidecar (1.525 B/flip floor >
  1.27 B/flip break-even). *Falsified by:* a blocky-error vehicle with position entropy < 1 B/flip.
- **P5 (ESTIMATE):** T_floor ≈ 0.07–0.13; the binding term is rate; sub-0.10 needs a smaller achiever
  class. *Falsified by:* an exact-eval row below 0.10 from any vehicle (would prove the smaller achiever
  exists and lower the empirical floor).
- **P6 (OPEN, unrun):** the 600×6 pose trajectory entropy is ~1–3 KB. *Resolved by:* the RANK 6 probe
  (cheapest open measurement; $0).
- **P7 (the meta-claim):** there is NO nontrivial PROVEN lower bound on T_floor below the few-KB regime
  (Kolmogorov uncomputable). The floor is an *achiever* question. *Falsified by:* a computable
  nontrivial lower bound on `K(evaluator-view | inflate)` — which would be a result in algorithmic
  information theory, not a contest move.

---

## 9. Wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE. The term decomposition (rate 61.7% binding, seg 29.3%/pose 9.0%
   recoverable) IS the top-level score-axis sensitivity prior; it re-ranks the v3 levers (distortion
   before rate up to T_3).
2. **Pareto constraint** — ACTIVE. The PROVEN coding-exhaustion (P2) + the 424KB seg-storage ceiling
   (P1) are hard Pareto walls on the rate axis for the current carrier class.
3. **bit-allocator** — ACTIVE. The "sub-0.15 is distortion-not-bytes" finding (P3) tells the allocator:
   spend the NEXT bytes on a sharper seg-amortizer, not on rate, until T_3.
4. **cathedral autopilot dispatch** — N/A. Derivation surface; no archive bytes emitted.
5. **continual-learning posterior** — N/A. Closed-form/measured derivation, not an empirical anchor;
   the LOWER-bound ledger (P1–P7) is the side information future agents reseed from.
6. **probe-disambiguator** — ACTIVE. P6 (the unrun pose-output-entropy probe) is the named
   disambiguator between "pose is free (folded in dense carrier)" and "pose is a ~1–3KB split lever".

## 10. Cross-references (the audited evidence base)

`GOAL_standing_v3_20260610.md` (Lever F) · `MASTER_ROADMAP_post_exhaustion_map_20260610.md` (the
post-exhaustion re-prioritization) · `MASTER_ROADMAP_v3_to_theoretical_floor_20260609.md` (the prior
floor estimate this hardens) · `evaluator_all_dimensions_synthesis_optimal_design_20260609.md` (the
term-structure + precision-edits) · `frontier_seg_repair_pool_verdict_20260610.md` (P4, the 1.525
B/flip floor + 66,039 flips) · `frontier_decoder_axis_waterfill_verdict_20260610.md` (P2, decoder 98.6%
iid Shannon + exact frontier components) · `frontier_latent_axis_waterfill_verdict_20260610.md` (latent
iid floor) · `segnet_fragile_support_codec_budget_20260609.json` (P1, the 424,722 B measured seg-storage
floor) · `evaluator_invisibility_basis_landed_20260610.md` (the certified null space — the appearance
slack) · `deforestation_read_surface_atoms_20260609.md` (the read-surface primitives) ·
`upstream/{evaluate.py,modules.py,frame_utils.py}` (the frozen authority, read in full).
