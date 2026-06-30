# Witness per-stage / per-pixel / per-class d_seg attribution  [macOS-numpy advisory . NON-PROMOTABLE]

- generated: `2026-06-30T16:50:37Z`
- verdict pairs: **96** of 200 (strided `range(0,200,2)[:96]`, identical across all ckpts)
- render: deploy-faithful fp32 ONE-CODEPATH, int8-dequantized EMA shadow, self-orient fixed-point `so_iters=4` (byte-close/inflate authority)
- d_seg = realized through-R frozen CPU-torch SegNet argmax disagreement vs GT `lstars` (NOT a proxy). Pointer 0.19110 UNMOVED; this is advisory.
- class order (CLAUDE.md): 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar

## Stage chain + realized d_seg

| stage | epoch | softmax_temp | mean fp-iters | realized d_seg | n_wrong | annulus% of wrong |
|---|--:|--:|--:|--:|--:|--:|
| CE | 299 | 0.5287 | 4.00 | **0.005443** | 102,727 | 98.5% |
| Tau | 599 | 0.0500 | 4.00 | **0.004563** | 86,131 | 98.0% |
| l7 | 725 | 0.1361 | 4.00 | **0.004287** | 80,918 | 97.9% |
| MuonStart | 726 | 1.0000 | 4.00 | **0.004311** | 81,371 | 97.8% |
| MuonBest | 900 | 0.2157 | 4.00 | **0.004117** | 77,706 | 98.1% |
| MuonLatest | 925 | 0.2157 | 4.00 | **0.003997** | 75,432 | 97.7% |

## Per-class disagreement per stage  (fraction of each GT class's pixels mislabeled)

| stage | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| CE | 0.50% | 47.23% | 0.10% | 6.70% | 0.11% |
| Tau | 0.59% | 32.22% | 0.10% | 5.10% | 0.09% |
| l7 | 0.58% | 30.12% | 0.09% | 4.79% | 0.08% |
| MuonStart | 0.62% | 29.54% | 0.08% | 4.86% | 0.07% |
| MuonBest | 0.65% | 26.25% | 0.10% | 3.68% | 0.06% |
| MuonLatest | 0.67% | 24.27% | 0.09% | 3.72% | 0.07% |

### Flip MASS share per stage  (of ALL wrong pixels, fraction whose GT class is c)

| stage | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| CE | 21.5% | 50.2% | 8.7% | 14.4% | 5.1% |
| Tau | 30.1% | 40.8% | 11.0% | 13.1% | 5.0% |
| l7 | 31.5% | 40.6% | 10.0% | 13.1% | 4.8% |
| MuonStart | 33.5% | 39.6% | 9.3% | 13.2% | 4.4% |
| MuonBest | 37.0% | 36.9% | 11.6% | 10.5% | 4.0% |
| MuonLatest | 38.8% | 35.1% | 11.0% | 10.9% | 4.2% |

## Stage transitions  (per-pixel, summed over the verdict pairs)

| transition | net d_seg Delta | corrected | regressed | persist-wrong | PRIMED | STUCK | corrected annulus% | persist-wrong annulus% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CE->Tau | -0.000879 | 47,795 | 31,199 | 54,932 | 38,442 | 16,490 | 97.5% | 99.4% |
| Tau->l7 | -0.000276 | 31,535 | 26,322 | 54,596 | 25,877 | 28,719 | 96.4% | 98.9% |
| l7->MuonStart | +0.000024 | 6,331 | 6,784 | 74,587 | 15,553 | 59,034 | 97.7% | 98.0% |
| MuonStart->MuonBest | -0.000194 | 39,831 | 36,166 | 41,540 | 19,930 | 21,610 | 96.1% | 99.4% |
| MuonBest->MuonLatest | -0.000120 | 28,018 | 25,744 | 49,688 | 20,652 | 29,036 | 96.9% | 98.7% |

### Per-transition CORRECTED by class

| transition | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| CE->Tau | 11,732 | 21,827 | 4,642 | 6,865 | 2,729 |
| Tau->l7 | 11,535 | 9,825 | 4,533 | 3,846 | 1,796 |
| l7->MuonStart | 1,809 | 1,615 | 1,352 | 961 | 594 |
| MuonStart->MuonBest | 15,388 | 12,733 | 3,960 | 5,689 | 2,061 |
| MuonBest->MuonLatest | 12,001 | 7,694 | 4,306 | 2,703 | 1,314 |

### Per-transition PRIMED by class  (persist-wrong with realized SegNet margin moving toward GT)

| transition | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| CE->Tau | 5,564 | 23,485 | 2,335 | 5,486 | 1,572 |
| Tau->l7 | 6,183 | 12,284 | 2,522 | 3,775 | 1,113 |
| l7->MuonStart | 2,980 | 6,386 | 2,461 | 2,514 | 1,212 |
| MuonStart->MuonBest | 5,328 | 9,555 | 1,574 | 2,726 | 747 |
| MuonBest->MuonLatest | 6,503 | 9,302 | 1,990 | 2,207 | 650 |

### Per-transition STUCK by class  (persist-wrong, margin flat/worse -> store/deterministic candidates)

| transition | Road | Lane | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| CE->Tau | 4,833 | 6,221 | 2,006 | 2,468 | 962 |
| Tau->l7 | 8,193 | 13,042 | 2,406 | 3,671 | 1,407 |
| l7->MuonStart | 20,681 | 24,860 | 4,317 | 7,117 | 2,059 |
| MuonStart->MuonBest | 6,541 | 9,943 | 2,016 | 2,342 | 768 |
| MuonBest->MuonLatest | 10,279 | 11,642 | 2,746 | 3,237 | 1,132 |

## Error-map overlays

- `experiments/results/witness_per_stage_attribution/errmap_CE.png`
- `experiments/results/witness_per_stage_attribution/errmap_Tau.png`
- `experiments/results/witness_per_stage_attribution/errmap_l7.png`
- `experiments/results/witness_per_stage_attribution/errmap_MuonStart.png`
- `experiments/results/witness_per_stage_attribution/errmap_MuonBest.png`
- `experiments/results/witness_per_stage_attribution/errmap_MuonLatest.png`

---

# The 4 operator questions — answered from the measured diffs

> Authority caveat (NO-FAKE): every number is the REALIZED through-R frozen CPU-torch SegNet argmax
> vs GT `lstars`, deploy-faithful (int8-dequantized EMA shadow, self-orient fixed-point so_iters=4 =
> the byte-close/inflate authority). It is `[macOS-numpy advisory . NON-PROMOTABLE]` — the realized
> verdict surface, NOT a contest score. Pointer 0.19110 UNMOVED. The deploy d_seg tracks the
> trainer's live trajectory numbers closely (CE 0.005443 vs live ~0.00544; l7 0.004287 vs ~0.004227;
> Tau 0.004563 vs ~0.00431) — within the documented deploy-fixed-point-vs-trajectory gap. NOTE the
> live Muon arm KEPT TRAINING during this analysis: `MuonBest`=ep900, `MuonLatest`=ep925 (more
> current than the ep825/ep850 in the prompt) — these ARE the best-d_seg EMA shadow at read time.

## (1) WHERE the d_seg improvements come from
**Almost entirely from Lane (class 1), and almost entirely on the boundary annulus.** Lane disagreement
falls **47.2% → 24.3%** of all lane pixels across the run (lane error mass 50.2% → 35.1%); it is the one
class that is far from solved at CE and is what every learning stage eats. Movable (cars) is the #2 win:
**6.70% → 3.72%** (Muon did most of it). Road/Undrivable/MyCar are already near-solved at CE (0.50% /
0.10% / 0.11%) — they are the static-core successes (MyCar hood, Undrivable sky, Road bulk land at the
floor immediately). **97.7–98.5% of every stage's wrong pixels sit inside the GT inter-class boundary
band (radius 2)** → the entire d_seg game is codim-1, exactly as predicted. Corrections are also ~96–97%
on-annulus.

## (2) What is STILL INTRACTABLE (the residual no stage corrects)
Two residuals, opposite in character:
- **The thin-Lane orbit (the hard long-tail).** 24.3% of lane pixels still wrong at MuonLatest (26,475 px).
  It is being *converted* (PRIMED — see Q3) but slowly; this is the ~8-dim lane-orbit CLAUDE.md names.
- **The Road boundary residual — and it is GROWING, not stuck-flat.** Road disagreement RISES monotonically
  **0.50% → 0.67%** (22,129 → 29,248 px). Because Road is huge (4.39M px), its error MASS climbs from 21.5%
  to **38.8%** and by MuonLatest **Road has overtaken Lane (35.1%) as the single largest error bucket**.
  The witness is trading road-boundary accuracy for lane gains as it sharpens. This emerging Road-dominated
  residual is consistent with — and converging toward — the CLAUDE.md "~50% Road" flip-mass
  characterization of the underlying hard set (the CLAUDE.md figure describes the stickier floor; this run
  is still burning down the easy Lane bulk and trending into it).

## (3) What Muon is improving NOW (MuonStart ep726 → MuonBest ep900 → MuonLatest ep925)
**Muon is grinding the Lane residual down** (29.5% → 26.2% → 24.3%) and **fixed Movable** early
(4.86% → 3.68%). The two real Muon transitions descend d_seg −0.000194 then −0.000120; each corrects
~8–13k Lane px and PRIMES ~9–10k more (lane persist-wrong mean margin-toward-GT still positive). Muon is
doing exactly the finisher's job: converting the primed boundary residual. The one thing Muon does NOT fix
is Road (0.62% → 0.67%, still drifting up). NOTE: `l7→MuonStart` (+0.000024, 59k STUCK) is a
**temperature-reset artifact**, not learning — the Muon arm resumes at softmax_temp 1.0 vs l7's 0.136, so
that row is the de-sharpen shuffle; read the real Muon signal from MuonStart→Best→Latest only.

## (4) Coherence verdict — does each stage match its expected role / prime the next?
**Largely COHERENT, with two honest caveats.**
- **CE = floor-set ✓.** Establishes the partition at 0.00544; all static-core classes land at the floor
  immediately (Road 0.50%, Undriv 0.10%, MyCar 0.11%); Lane is left as the open residual (47%). Exactly the
  expected "set the cartoon, leave the boundary."
- **Tau (softplus-sharpen) = the single biggest drop ✓ (−0.00088).** Sharpening (temp 0.529→0.050) corrects
  21.8k Lane px AND **PRIMES 38.4k residual px (mean margin-toward-GT +0.366, the strongest priming of any
  transition; 79% of the persist-wrong lane is PRIMED)** — it sets up the residual for l7/Muon. Caveat: it
  REGRESSES 15.5k Road px (sharpening over-commits at road boundaries) → the start of the Road drift.
- **l7 (conditioning) = modest refinement ✓ (−0.00028).** Real but small; continued Lane + Road + Undriv
  corrections, priming slowing (margin +0.044). Coherent as a capacity/FiLM refinement, not a breakthrough.
- **Muon (finisher) = ✓.** After the temp-reset artifact, it steadily converts the primed Lane residual and
  fixes Movable — the expected grind-down.

**Is the residual PRIMED or STUCK?** SPLIT, and that split is the actionable signal:
- **Lane = PRIMED / learnable** — margins keep moving toward GT every transition (CE→Tau lane +0.51;
  Muon transitions +0.029–0.067), and Muon keeps converting it. Keep-training is working on Lane.
- **Road = STUCK / antagonized** — Road persist-wrong margins are ~0 or negative in the Muon transitions
  (MuonBest→Latest Road +0.004; l7→MuonStart Road −0.020), and Road argmax errors grow. Training is NOT
  priming Road; it is the residual being pushed the wrong way by lane-sharpening.

---

# Surgical-repair toolbox (per-class, ranked by Δd_seg-per-byte, grounded in the PRIMED/STUCK split)

All targets are ~98% on-annulus. Mechanism chosen per the measured priming behavior (PRIMED→LEARN;
STUCK→deterministic/store), cheapest Δd_seg-per-byte first:

1. **[0 byte, HIGHEST leverage] Road-boundary protection — the GROWING STUCK residual (29,248 px,
   38.8% of error mass).** Road is regressing BECAUSE of lane-sharpening and is NOT primed → more training
   will not fix it. Two rule-118-FREE moves: (a) a **curriculum re-treatment that protects Road during the
   tau/Muon sharpen** (per the "different stages need different treatment" discipline — the sharpen stage is
   inheriting no road-boundary guard); (b) a **DETERMINISTIC road-horizon geometric prior** (the
   `road_horizon_component` / static-core trapezoid is generic same-rig geometry, FREE in inflate). Road
   boundary is the largest single bucket and is currently going the wrong way at $0 archive cost to fix.
2. **[0 byte, learned-in-weights] Keep Muon training the PRIMED Lane orbit.** Lane (26,475 px, 24.3%) is
   still descending with positive margin movement each transition — no byte cost, just epochs. This is the
   confirmed-working path; do not store what is still converting.
3. **[~0 byte deterministic] LEVER-4 directional / UNIWARD on the STUCK thin-lane sub-residual.** The lane
   pixels that are STUCK (not moving, ~10–12k/transition) are thin oriented structures — the directional
   (self-orient) / UNIWARD oriented-texture lever is the topology-matched, near-0-byte mechanism before any
   storage.
4. **[counted, LAST RESORT] Yousfi store-the-flip sidecar (~1.27 B/flip)** for the final irreducible STUCK
   core only (thin-lane + hard Movable cars). At ~26k stuck lane px this is ~12.7KB if applied broadly →
   too expensive for ~5e-4 d_seg; reserve for the last few-thousand-px irreducible core AFTER 1–3, and only
   admit it on a measured Δd_seg-per-byte that survives through-R.

Undrivable (0.09%) and MyCar (0.067%) are SOLVED (static-core) — no action.

**Means≠ends reminder:** this is the realized-through-R advisory verdict surface; it sharpens WHERE to spend
the next byte/epoch, but the pointer (0.19110) only moves when a byte-closed packet beats it under
`upstream/evaluate.py`. The #1 recommendation (FREE Road-boundary protection) is the highest-leverage,
lowest-cost next move it points to.

