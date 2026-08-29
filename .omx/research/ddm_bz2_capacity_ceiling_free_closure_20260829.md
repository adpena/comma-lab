# bz2 born-small capacity ceiling — CLOSED WITHOUT THE SCORER RUN (MAIN, 2026-08-29)

**Axis:** `[macOS-CPU frozen-scorer advisory]` on the bo2 inputs · `score_claim=false` ·
`promotable=false`. No archive bytes moved; the frontier is untouched by this memo.

**What this closes:** task #1325 (`ddm_bz2` born-small capacity ceiling), whose arm landed
`PARTIAL-PROVED-AND-QUEUED` with `capacity_ceiling: null` because it never got the scorer lane.
MAIN adjudicated it from the arm's own retained receipts. **The scorer terminal in
`FIRE_ORDER.json` is NOT fired** — §4 states exactly why, and what it would still buy.

## 1. WHAT THE ARM MEASURED — the capacity probe already ran, at the native layer

The charter's §2 probe (supervised fit of the born-small representation directly to the
DALI-lineage GT partition, held out) **completed**. Its output is a native categorical
agreement, not a `d_seg` — the arm labelled it correctly throughout
(`[native categorical representation diagnostic; NOT d_seg]`).

| quantity | value | source |
|---|---:|---|
| ancestor (bo2's HG1 field) native GT mismatch | 0.011301566229926216 | `NATIVE_ANCESTOR_COMPARISON.json` |
| **GT-supervised direct fit** native GT mismatch | **0.011229510837131076** | same |
| absolute improvement | 7.206e-05 | derived |
| **relative improvement** | **0.6376%** | derived |
| sites changed vs ancestor | 74,650 / 117,964,800 = **0.06328%** | same |

**Fitting the representation DIRECTLY to the exact target — supervised, at optimal form —
improves its agreement with that target by 0.64%.** The ancestor was already at the schema's
capacity. That is the capacity ceiling, measured, on the axis the probe could reach.

### Generalization: the fit is WORSE than not fitting at all, out-of-sample

| split | train | holdout | ratio |
|---|---:|---:|---:|
| **pair** holdout (480/120) | 0.011260743 | **0.016231452** | **1.4414× WORSE** |
| spatial holdout (600/600 px-split) | 0.011523446 | 0.011626068 | 1.0089× |

The pair holdout (0.016231) is **1.4362× worse than the un-fitted ancestor** (0.011302).
Mechanism: the representation generalizes **spatially** (within-frame structure is shared) and
**not across pairs** (per-pair content is not learnable from other pairs by this schema).

This is the **pk3/pk4 lesson at a third independent site** — in-sample positive, held-out
negative-or-worse (#1041: 23/23 in-sample = 0/23 LOO). Three arms, three vehicles, one law.

## 2. THE POSE LEG — no transfer needed, and it is the decisive one

bo2's authority row on the ancestor field (`BO2_REDERIVATION.json`, matched-delta convention):

| term | value | share of damage |
|---|---:|---:|
| seg contribution `100·d_seg` (d_seg 0.012949210) | 1.294921 | 24.88% |
| **pose contribution `√(10·d_pose)`** (d_pose 1.52821589) | **3.909240** | **75.12%** |
| candidate distortion | 5.204161 | — |
| base distortion | 0.073082 | — |
| delta / budget 0.024543 | **209.06669×** | published 209.06668593299145 ✓ |

**The counterfactual that closes it.** Set `d_seg = 0` — grant the partition PERFECT capacity,
better than any fit could ever be:

```
distortion at perfect seg = √(10 × 1.52821589) = 3.909240195
delta vs base             = 3.836158287
REFUSAL MULTIPLE          = 156.30491×      ← still dead
```

To *tie* at perfect seg, `d_pose` would have to fall to **9.530581e-04** — a **1,603.5×**
reduction. The capacity probe fit the **partition**; it did nothing for pose by construction.

## 3. VERDICT — born-small is walled on BOTH terms, for different reasons

`verdict_scope: FAMILY` for born-small (the object, all rungs), **n=1** for the cross-object law.

- **SEG — CAPACITY-WALLED.** Optimal-form supervised GT fit buys 0.6376% in-sample and is
  1.4362× WORSE than the un-fitted ancestor out-of-sample. There is no optimization headroom to
  recover; the achieved distortion *was* the ceiling.
- **POSE — STRUCTURALLY REFUSED.** 75.12% of the damage is pose, and even at *perfect* seg the
  object is refused **156.3×**. No partition-side work reaches it.

The bz2 §3 three-way fork therefore resolves to branch 1 **for born-small**:
*"both ceilings ≈ their achieved distortion → capacity-walled."* bo2's 209× refusal
(#1262) **stands, and is now explained**: it was never a budget artifact.

**Cross-object law: still n=1.** `ddm_qbz1` owns the qbt2b leg. When it lands, the law becomes
n=2 and gets stated at that scope — not before (#821: count distinct facts).

## 4. WHY THE SCORER TERMINAL IS NOT FIRED

`FIRE_ORDER.json`'s trigger is fully MET (fcd3 released the sole scorer lane; MAIN owns it; hashes
revalidate). I am declining it anyway, on the arithmetic:

1. **It cannot change the verdict.** The pose bound (156.3× at perfect seg) is exact and
   independent of any seg measurement. The seg bound rests on a 0.64% in-sample improvement that
   would need to become a 209× realized improvement — an amplification of ~32,600× in the
   improvement ratio — to invert.
2. **Cost vs. what it buys:** ~840 s + 3.66 GB of inflated raws (bo2's measured predecessor) with
   **AP at 17 GiB free and Vertigo at 8.3 GiB**. What it would buy is real but not verdict-bearing:
   a **second point** calibrating the native→realized map, currently n=1 (bo2's 1.1458× native→d_seg
   amplification).
3. **The host runtime is gone.** bo2's render host (`ddm_tv1_tolerance_curve/runtimes/dx2_shipped`)
   was certify-and-MOVE reclaimed by sr3 into `SR3_ORIGINAL_TREE.tar.zst` (cert `RECLAIMED_VERIFIED`,
   per-file SHA manifest intact — the discipline worked). Firing requires a selective extract first.

**QUEUED-WITH-A-FIRE-ORDER (owner MAIN, superseding the arm's):** run the terminal at the next
storage-clear boundary, purpose relabelled **native→realized CALIBRATION (n=1 → n=2)**, not
capacity adjudication. Trigger: AP free ≥ 25 GiB. Recipe: extract `runtimes/dx2_shipped` from the
sr3 tarball, publish bz2's `archive_parseback_tokens.u8` (sha `968ffca2…`) as the token-cache
payload under the host's binding key, then `fire_local_advisory` per bo2's `ADVISORY_LAUNCH.json`.

**FALSIFIER for §2's one transfer** (bz2's fitted field's `d_pose` is inferred ≈ ancestor's, since
the fit moved 0.063% of sites and targeted the partition): measure `d_pose` on the fitted field. If
it lands **below 9.530581e-04**, this closure is refuted and born-small reopens. Nothing else in
this memo depends on that transfer.

## 5. NEXT_IF_RESUMED

- **CLOSED** — #1325 born-small capacity ceiling: seg capacity-walled (0.64% in-sample, negative
  held-out), pose structurally refused (156.3× at perfect seg). FAMILY scope for the object.
- **LIVE** — `ddm_qbz1` (#1324) owns the qbt2b leg; the n=2 cross-object law is stated only after
  it lands. Its charter carries the same self-imposed "fcd3 owns the scorer lane" precondition that
  blocked bz2 — **it is stale, the lane is free, and MAIN owns any terminal its fire order names.**
- **QUEUED-W-FIRE-ORDER** — the calibration terminal above (owner MAIN, storage-gated).

## LIVE-HYPOTHESES

- The pair-vs-spatial holdout split (1.4414× vs 1.0089×) says these representations learn
  *within-frame* structure and not *cross-pair* content. That is a statement about the whole
  born-object family, and it predicts the same signature in qbz1 — a free cross-check when it lands.

## DEAD-ENDS

- Firing the scorer terminal to decide capacity: the verdict is already determined by the pose leg,
  which no partition-side measurement can move.
- Reading the 0.6376% native improvement as "small headroom worth chasing": out-of-sample it is
  *negative*, so the headroom is not merely small, it is the wrong sign.

Own-vehicle frontier UNMOVED this turn: **S 0.14803010583079396 @ 180,083 B [contest-CUDA T4 n600]**
(lb1, archive sha `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9`).
