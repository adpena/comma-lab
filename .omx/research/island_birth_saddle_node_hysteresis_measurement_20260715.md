# Island-birth: saddle-node + hysteresis test ($0, frozen artifacts) + DERIVED per-class birth-weight lever

**Date:** 2026-07-15 · **Axis:** `[macOS-CPU/MLX advisory] NON-PROMOTABLE` — pointer 0.19108 UNMOVED (means).
**Task:** test whether witness island-birth is a genuine saddle-node bifurcation with hysteresis and, if so,
compute λ_c so the curriculum's birth-λ is *set from the bifurcation, not guessed*.
**Motivation:** memory `curriculum_is_continuation_instabilities_are_bifurcations_20260714` (curriculum =
continuation; island-birth = saddle-node in the class-occupancy order parameter) + papers-batch item-5
(Fork Dynamics — fit the movable-occupancy normal form, check fold-coincidence, continue in λ). Value is
GATED on deriving the reduced-order model, not the tool.

## Verdict first (honest)

- **Saddle-node: NOT CONFIRMED from $0 frozen data — and NOT refuted.** The occupancy order parameter that
  IS measurable at $0 (continuation across the curriculum stages of an existing run) is **SMOOTH and
  monotone/plateaued — no discontinuous jump, no turning point → no fold visible.**
- **Hysteresis is UNMEASURABLE at $0 by construction.** Hysteresis is a property of the training *flow*
  (a slow-fast dynamical system whose slow state has memory). A static λ-sweep on a **frozen** checkpoint is
  algebraically a deterministic function f(λ) → ramp-up ≡ ramp-down, zero loop, *regardless of the true
  dynamical bifurcation*. Reporting such a static sweep as a "hysteresis test" would be a surrogate-as-authority
  FAKE (NO-FAKE class 8). Stated, not run.
- **Normal-form fit: data too coarse / wrong axis.** 6 curriculum-stage points, monotone-smooth, no
  multivalued turning point ⇒ the algebraic fold `x²≈(λ−λ_c)` has nothing to fit; λ_c not extractable from
  frozen artifacts.
- **The real $0 deliverable is a DERIVED reduced-order lever:** the birth-balance dimensionless group makes
  the per-class birth-weight threshold scale with the class's isoperimetric homogenization drain `(P/A)_c`.
  MEASURED GT geometry ⇒ **Lane needs ≈8.9× Movable's birth-weight** (geometry-grounded) to nucleate on the
  same schedule. This replaces per-class hand-tuning with a computed ratio *today*; the ABSOLUTE λ_c (the
  drain rate δ) still needs the one thing forbidden here — a live W_birth up/down ramp.

## Method (all read-only; no training; no paid launch)

- **Order parameter (occupancy):** for rare classes Lane(1) and Movable(3): (a) predicted area fraction;
  (b) per-pair presence fraction; (c) **islands/pair = connected-component count ≥4 px** (the true "island"
  observable). Computed via `scipy.ndimage.label` on the frozen `argmax` maps vs `gt_n96` `lstars`.
- **Continuation axis:** `experiments/results/witness_per_stage_attribution/maps_{CE,Tau,l7,MuonStart,MuonBest,MuonLatest}.npz`
  — the SAME 96 pairs rendered-through-R at successive curriculum stages (epoch 299→925). The curriculum IS
  the continuation path (per the source memory), so occupancy(stage) is the measurable continuation curve.
- **Cross-config branches:** the same order parameter on `mod32cap` (n16) and `perclass_baseline_n600` (n600)
  frozen maps, to look for a distinct "absent-island" branch (the putative other side of a bistability).
- **GT geometry:** per-class GT area + perimeter/area (isoperimetric drain) on n96.
- **Scale labels:** curriculum-continuation = **n96** (advisory); cross-config includes **n600**; GT = n96.
  Per the allergic-to-non-n600 rule this feeds a design/curriculum decision, not a score claim.

## MEASURED — occupancy continuation across curriculum stages (n96, one run)

CLASSES: 0=Road 1=Lane 2=Undriv 3=Movable 4=MyCar. GT: Lane area 0.589% / **21.3 islands/pair**;
Movable area 1.557% / **3.1 islands/pair**; both classes **100% pair-presence** at every stage.

| stage | epoch | Lane area% | Lane islands/pair | Movable area% | Movable islands/pair |
|---|---:|---:|---:|---:|---:|
| CE | 299 | 0.349 | 15.9 | 1.127 | 2.9 |
| Tau | 599 | 0.457 | 16.1 | 1.151 | 3.0 |
| l7 | 725 | 0.470 | 16.2 | 1.150 | 3.0 |
| MuonStart | 726 | 0.476 | 16.4 | 1.150 | 3.0 |
| MuonBest | 900 | 0.504 | 15.4 | 1.170 | 3.0 |
| MuonLatest | 925 | 0.522 | 15.6 | 1.166 | 3.0 |

**Reading:** at CE (ep299) both rare classes are ALREADY present at ~75% (Lane islands 15.9/21.3) and ~94%
(Movable 2.9/3.1) of GT. Over the entire curriculum the island order parameter is **flat within noise**
(Lane ~16, Movable ~3.0) and the area rises **smoothly and monotonically** (Lane 0.349→0.522%). **No
discontinuous birth event; no fold.** This is the dash-homogenization deficit (memory L65 "dashes
unrecoverable below crossover at ANY capacity"), a *plateaued sub-GT gap*, not a nucleation transition.

## MEASURED — cross-config branches (no "zero-island" branch in frozen artifacts)

| config | n | Lane islands/pair | Lane area% | Movable islands/pair | Movable area% |
|---|---:|---:|---:|---:|---:|
| per_stage MuonBest | 96 | 15.4 | 0.504 | 3.0 | 1.170 |
| mod32cap BEST | 16 | 15.6 | 0.507 | 2.9 | 1.092 |
| mod32cap END | 16 | 15.1 | 0.505 | 2.9 | 1.105 |
| perclass_baseline | 600 | 15.9 | 1.383 | 2.6 | 1.932 |
| **GT** | 96 | **21.3** | **0.589** | **3.1** | **1.557** |

All frozen configs cluster at one sub-GT plateau (Lane ≈15–16 islands/pair, Movable ≈2.6–3.0). **The
memory's "mod32cap ZERO lane/movable islands" (L2/L3) is NOT reproduced at the connected-component level in
these frozen maps** — the classes/islands are present-but-deficient, not absent. INFERRED: the "unborn"
framing referred to the *correctly-flipped / GT-matched* dash mass (the missing ~25% of Lane island count),
or to an earlier config not in these artifacts — not to a class that is globally absent. Either way, no
absent↔present branch pair exists in frozen data to form a hysteresis loop.

## The one measured bistability hint (n=2, NOT a loop) — #300

The only measured signature bearing on a *saddle-node threshold* is the #300 paint-seed result already in
`islands_composed_ceiling_arithmetic_20260707.md`: an **un-gated seed** (large initial occupancy
perturbation) **plateaued d_seg ~0.027 and was ABSORBED back** vs control 0.0078 at matched ep50
("seed-absorption starvation"). A finite seed decaying back toward absence is exactly the **down-branch /
basin-memory** behavior of a subcritical threshold (separatrix): the "present" state is not reachable from
generic init and an insufficient/un-held seed relaxes back. This is **consistent with a saddle-node/subcritical
nucleation threshold** but it is **2 operating points, not a continuous hysteresis loop** — it cannot locate
λ_c and it cannot distinguish saddle-node from a steep transcritical.

## DERIVED — reduced-order birth-balance model + the actionable lever

Let `x = A_c` (predicted occupancy of rare class c). Two competing region-competition forces during the
level-set/Chan-Vese flow:
- **birth/data drive** ∝ `W_birth` (the birth-weight the curriculum turns on) times the class's "room+signal";
- **homogenization/MCF drain** ∝ `δ` times the class's boundary — the mean-curvature erasure that removes
  thin structure. For a class of area A and perimeter P, the curvature drain per unit area scales with the
  **isoperimetric ratio (P/A)_c**: thin/dashed classes (high P/A) are drained hardest.

Birth-balance (nucleation when drive ≥ drain at small x) ⇒ dimensionless control
`λ_c ≡ W_birth / (δ · (P/A)_c)`, with **nucleation for λ_c ≳ 1**, i.e. per-class threshold
**`W_birth,c* ≈ δ · (P/A)_c`**.

MEASURED GT geometry (n96):

| class | A_GT% | perimeter/area (P/A) | GT islands/pair |
|---|---:|---:|---:|
| Lane | 0.589 | **0.760** | 30.0 (all comps) |
| Movable | 1.557 | 0.086 | 3.1 |

⇒ **`(P/A)_Lane / (P/A)_Movable = 8.9`.** The geometry-grounded per-class birth-weight ratio is
**Lane ≈ 8.9× Movable** (a cruder `1/A_GT` proxy gives 2.6×; the isoperimetric factor is the physically
correct drain and is much steeper — consistent with Lane being the measured HARD class and Movable nearly
saturated at CE). This is a **DERIVED, $0, ready-to-wire curriculum lever**: set per-class birth-weight
`W_birth,c ∝ (P/A)_c` so all rare classes cross their nucleation threshold on the same schedule instead of
Lane starving while Movable over-drives. Sisters: L83 per-class-λ homotopy, #433 per-class-λ, #300/#315/#323.

**Honest limit:** the SCALING (per-class ratio) is derived from measured GT geometry; the **ABSOLUTE**
threshold needs δ (the homogenization rate), a training-flow quantity NOT in frozen data.

## Verdict + next step

- **Saddle-node YES/NO: NOT CONFIRMED (frozen-data continuation is smooth; hysteresis unmeasurable at $0).**
  Weak-but-real bistability hint from #300 (n=2 seed-absorption) is consistent with a subcritical threshold
  but cannot locate λ_c.
- **Concrete curriculum lever available NOW (DERIVED, $0):** per-class birth-weight `∝ (P/A)_c` →
  **Lane ≈ 8.9× Movable**. Wire as a per-class scaling on the birth/growth loss (register as a DSL Lever;
  duty-to-measure), not a hand-tuned pair of numbers.
- **The decisive test is NOT $0 (need X):** resume-from-disk on an existing EMA-BEST
  (e.g. `levelset_v752_baseline_20260710T185913Z/levelset_witness_ema_BEST.npz`) and **quasi-statically ramp
  a single per-class W_birth UP then DOWN** over a small epoch budget, logging Lane/Movable **islands/pair**
  (the order parameter measured here) per epoch on the n96 verdict. A bistable window + loop width ⇒ measured
  λ_c ± hysteresis margin, and δ falls out (δ ≈ W_birth*/(P/A)_c at the up-branch fold). Cheapest faithful:
  one short resumed run; operator-GO (heavy/paid). Until it lands, ship the `∝(P/A)_c` ratio as the interim
  birth-weight lever.

## Repro

```python
import numpy as np; from scipy.ndimage import label, binary_erosion
g=np.load("experiments/results/mlx_fleet_gt_cache/gt_n96.npz",allow_pickle=True); gt=g["lstars"].astype("int8")
z=np.load("experiments/results/witness_per_stage_attribution/maps_MuonBest.npz"); a=z["argmax"].astype("int8")
def isl(x,c,mp=4):  # islands/pair
    n=0
    for p in range(x.shape[0]):
        m=x[p]==c
        if m.any():
            L,k=label(m); n+=sum((L==i).sum()>=mp for i in range(1,k+1))
    return n/x.shape[0]
# per-stage maps_{CE,Tau,l7,MuonStart,MuonBest,MuonLatest} give the continuation curve.
```

## Assumptions (labeled)

1. Curriculum-stage sequence ≈ a continuation path in the effective control (per source memory). n96 subset.
2. The frozen static-sweep = FAKE-hysteresis argument (kills the $0 static path) is a DERIVED structural fact
   (f(λ) has no memory), not a run.
3. Birth-balance dimensionless group + `(P/A)_c` drain scaling = DERIVED reduced-order model; the per-class
   RATIO uses MEASURED GT geometry; the ABSOLUTE λ_c/δ is UNMEASURED (needs the flow sweep).
4. Every number `[macOS advisory]`; the pointer (0.19108) moves ONLY through a byte-closed exact row.
