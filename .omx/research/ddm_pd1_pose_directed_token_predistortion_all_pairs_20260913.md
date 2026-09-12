# ddm_pd1 — POSE-DIRECTED token pre-distortion, the re-solve credit as the objective

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913`. Base: move 49,
S 0.13632299781031237 @ 179,153 B, archive sha
`73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container.
**No score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN
fires.** No Modal, no fire, no packet.

Pose solver, stated once: every resolved pose number here comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` (= pass 8's refine = cr1's; the same
object at these budgets), driven at PoseNet batch 1 on the moved render; every n600 base
number comes from `up2.measure_pose` at batch 8 on move 49's own cold parse-back `0.raw`.

---

## 1. The headline

**PLACEHOLDER — filled from the receipts at the end of the run.**

---

## 2. The base, reproduced rather than inherited

| quantity | measured | control | agreement |
|---|---:|---|---|
| n600 mean d_pose on move 49's own decode | **4.543568679770593e−06** | pp1's independently measured vector | **bit-identical**, max abs difference **0.0** |
| " | " | pass 8's F7 gate ratio | **0.998586523026504**, reproduced to all digits |
| " | " | the T4 print 4.55e−06 | ratio 0.998587 |
| median / max | 3.34503739671486e−07 / 2.1283759625946122e−04 | — | max/median **636×** |
| top-12 share | 52.49 % | pp1 | same 12 pairs |
| carrier codes | (600, 12) int32, sha `78cf4fe54b2bb6a9…` | read from move 49's own archive | — |

Receipts: `base/POSE_BASE_MOVE49.json`, `base/pose_base_move49.npy` (sha
`05bd30248a167a6b9dab00841dceb343f232265240c1590912875f2e625c11f1`), `base/codes_move49.npy`.

**F_base_reproduction did not fire.** Three independent measurements of the same object
agree bit for bit.

### The base band is ABSOLUTE, and this arm treats a population spanning 3,000×

pp1 §10's lesson is the denominator, not the epsilon. Re-measured here on **40 pairs drawn
across the whole d_pose range** (the 20 stratified smoke pairs plus 20 seeded random
non-floor pairs):

| quantity | measured |
|---|---:|
| batch-1 repeat, all 40 pairs | **bit-exact** |
| max **absolute** batch-1 vs batch-8 gap | **2.2643e−09** |
| max **relative** gap | 9.9181e−03 (on a pair at ~1e−07 — the same absolute noise over a tiny denominator) |
| gate used | **2.2643e−08** = 10× the observed absolute maximum |

A relative gate would have been ~400× too loose on the smallest pairs and ~4× too tight on
the largest. Receipt: `base/BASE_TOLERANCE.json`.

---

## 3. The economics, DERIVED before the search, and they EXPIRE

At this operating point (`base mean 4.5435687e−06`):

| term | value |
|---|---:|
| S per archive byte | 6.658589531221714e−07 |
| S per flipped SegNet cell (T4-carried) | 8.482724499051702e−07 |
| S per unit of **one pair's** d_pose | **1.2362895701574137** |

**A hard bound, not a heuristic.** A single-token edit changes one pair's pose, and
`|credit| ≤ base` because the resolved d_pose cannot go below zero. So a pair whose own
base d_pose is below `rate_S_per_token / 1.2363` can **never** admit with one token, no
matter what the edit does:

| price | bits/token | required per-pair credit | non-floor pairs able to pay |
|---|---:|---:|---:|
| pass 8's full field | 8.93 | 6.0121e−07 | **220** |
| pass 7's admitted subset | 5.0149 | 3.3762e−07 | **286** |
| an optimistic 3.0 | 3.00 | 2.0197e−07 | **345** |

The search population is the **345** pairs above the optimistic bound, excluding pp1's 12
floor pairs. The 243 pairs below it were not searched because no realizable credit there can
pay for its own token. Receipts: `PREREGISTRATION.json`, `search_population.json`.

---

## 4. The smoke — and it falsified the prior this arm was most afraid of

20 stratified pairs, pure pose-saliency ranking (pp1's `search_b` verbatim, K = 3 cells ×
{−1,+1}), every proposal fully realized.

**MEASURED: the median best per-pair resolved credit is −13.0 % of the pair's own d_pose,
best −86.1 %.** pp1 measured 0.25 %–0.45 % on the 12 hard pairs and found a FLOOR there.
**That floor does not generalize.** The hard pairs are hard because their residual lies
outside the carrier's twelve-dimensional reach; ordinary pairs are not, and on them a single
token edit moves the resolved pose by tens of per cent.

The binding leg is therefore **seg, not pose**: 51 of 53 smoke proposals cost ≥ 1 cell, and
one cell (8.48e−07 S) already exceeds the pose credit most pairs can give.

Stratified extrapolation of the smoke (MODELLED on rate at 8.93 bits/token):
**−1.62e−05 S = 0.81× the −2e−05 bar → stop-rule branch B (MARGINAL)**, which is the branch
the pre-registration said to expect. Receipts: `SMOKE_YIELD.json`, `SMOKE_VERDICT.json`.

---

## 5. K, and why the full search screens before it refines

**PLACEHOLDER — K declaration and timing.**

---

## 6. The search

**PLACEHOLDER.**

---

## 7. The admission

**PLACEHOLDER.**

---

## 8. Two defects this arm caught with its own controls

**PLACEHOLDER.**

---

## 9. What this does NOT claim

**PLACEHOLDER.**

---

## 10. Custody

**PLACEHOLDER.**

<!-- # FORMALIZATION_PENDING: the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, and only on an exact row. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm — MAIN fires):
**S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]** (move 49).
