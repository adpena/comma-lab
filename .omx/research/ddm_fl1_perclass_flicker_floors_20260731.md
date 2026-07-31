# ddm_fl1 (#813) — per-class GT-flicker floors (gc13 R3, sandwich lower-bound leg)

**Date:** 2026-07-31 · **Actor:** ddm_fl1 · **Task:** #813 · **Charter:** `.omx/research/ddm_gc13_optimal_control_shape_20260731.md` §5 + §9 op-routable **R3**.
**Authority:** `[macOS-CPU advisory]`, derived-from-cached-GT, `research_only=true`, `score_claim=false`.
**Pointer 0.1910828242 [contest-CPU] UNMOVED** — this is MEANS ($0, scorer-FREE: zero SegNet/PoseNet forwards; cached GT argmax only; the live burn-4 scorer slot untouched).

**Headline (answer first):** the per-class GT-flicker floor decomposes the registered aggregate
0.005318 (reproduced to +4.4e-7) as **Lane 0.2316 S · Road 0.1889 S · MyCar 0.0434 S · Undrivable
0.0394 S · Movable 0.0285 S** (charge-by-`lstars[t]`, /598 interior pairs). **Falsifier:** the literal
gc13 trigger fires for **all 5 classes** (each smooth-label floor is 5.0–16.8× its corner-C
allocation), but the correct verdict is **NO re-waterfill** — this floor is FORMULATION-scoped
(smooth-label/static-describe only) and is *pierced* by phase-faithful renderers (PR130 d_seg
2.966e-4, ep641 0.004264, FEED-ma 0.00086 all sit BELOW it). The binding sandwich lower-bound leg is
ru1's GT-jitter-typed reachable (~6e-4 ≈ corner-C), **not** the 5.3e-3 smooth-label floor. The
per-class floor's live value is a **ranking of phase-faithfulness debt** (Lane binds hardest) plus a
**residual-above-floor** finding that names Undrivable + Movable — Undrivable being exactly the
eroding-unguarded class of gc13 §2.0/B1.

---

## §0 Prior-law prediction lines (recall FIRST — anti-re-anchor)

State the established laws, then measure and diff. Nothing below is a discovery of a new floor; it is
a *decomposition* of an already-registered one, plus the honest scope the laws already predict.

1. **PHASE-flicker floor law** (`gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`,
   `flicker_transform_geometry_term_design_20260710.md` §2.4): the GT single-frame **spike rate** —
   a scored-frame GT-argmax pixel that differs from BOTH stride-2 neighbors — is **0.005318** of the
   frame (1046 px/pair, n600), 97.7% repairable. It equals the *smooth-label* witness d_seg floor (a
   temporal-majority oracle scores exactly the spike rate). Part D already gives the class shares:
   **Lane 43.6% / Road 35.5% / MyCar 8.2% / Undriv 7.4% / Movable 5.4%**. My job is to reproduce this
   from cached `lstars` and convert to S-units per class → these numbers are a PREDICTION my
   measurement must land on (it does, to <0.1%).
2. **Flicker floor is NOT hard** (operator 2026-07-15,
   `feedback_flicker_floor_not_hard_fire_phase_stack_stop_deferring_20260715.md`): 0.005318 binds
   ONLY witnesses temporally SMOOTHER than GT; it is FORMULATION-scoped, NEVER a paradigm floor.
   Existence proofs below it already in-ledger: FEED-ma appearance-phase 0.00086; PR95/bc36 ancestor
   ~6e-4; our ep641 witness 0.004264. Citing 0.005318 as a HARD d_seg floor is a FORBIDDEN cargo-cult
   claim. → **PREDICTION: the falsifier's literal "floor > corner-C ⇒ infeasible" must NOT be read as
   re-waterfill-with-hard-floors; it re-confirms this scope law per class.**
3. **SegNet class order** `[Road, Lane, Undrivable, Movable, MyCar]` (canonical comma10k order,
   NEVER luma-derived; verified live: `lstars` range [0,4]). Confirmed by the Part D share match.
4. **xp1 per-class residual** (ep641 r1c, `ddm_xp1_20260731/xp1_verdict.json`):
   `[Road 0.18845, Lane 0.12589, Undriv 0.05574, Mov 0.03792, MyCar 0.0184]` S; total d_seg
   0.004264052. → **PREDICTION: total residual (0.004264) < total floor (0.005318): the witness
   already pierces the smooth-label floor in aggregate — so some per-class residuals sit BELOW their
   floor.**
5. **ru1 endpoint typing** (`ddm_ru1_20260729/atlas_flat.npz`, merge cf0e2f5b8b): 458,738 realized
   flips, 94% GT-boundary-jitter, per-flip `gt_flicker` flag → the per-class join.
6. **Staleness-is-a-named-confound** (gc13 §0 law 6): xp1 is r1c ep641; ru1 is ep399 tb1; my floor is
   GT-only (vehicle-free). Cross-endpoint use = LABELED structure transfer, not a number transfer.

**Diff — what is genuinely NEW here (not in the Part D row):** (i) the per-class floor in **S-units**
(Part D only had spike *shares*); (ii) the exact **denominator convention** (registered 0.005318 =
/598 interior pairs, reproduced to +4.4e-7); (iii) the **residual-above-floor** join naming
Undrivable+Movable; (iv) the ru1 `gt_flicker`-fraction-of-residual per class; (v) the **scoped
falsifier** verdict (literal-trigger-all-5 vs no-re-waterfill).

## §1 Custody + method (charter step 1)

| item | value |
|---|---|
| GT cache | `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` |
| SHA-256 (streamed, this run) | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` |
| bytes | 5,078,017,610 |
| lineage verdict | **CLEAN** — matches ch1 S8 (`ddm_ch1_recursive_confound_pass_20260728.md`: producer log 2026-06-26 → ch1 SHA → fd1r `run_identity`; `lstars` int64 0-4, no NaN; rp1 C0 GT→lstars 0 flips) |
| `lstars` | shape (600, 384, 512) int64, values [0,4] — the frozen CPU-torch SegNet argmax of each pair's scored (last) frame |
| driver + outputs (my custody) | `/Volumes/VertigoDataTier/pact/ddm_fl1_20260731/` (`ddm_fl1_verdict.json`, `ddm_fl1_join_falsifier.json`, `ddm_fl1_perclass_flicker.py`, manifest) |

**Definition (exact, matches the registered eq):** a scored-frame pixel is a **spike** iff its GT
argmax `lstars[t]` differs from BOTH stride-2 neighbours `lstars[t-1]` and `lstars[t+1]`. Consecutive
non-overlapping pairs' scored frames ARE stride-2 in the source (frames 2t-1, 2t+1, 2t+3), so the
spike is computable from `lstars` alone over the 598 **interior** pairs (both neighbours present) —
exactly the gc13 §5 "$0 producer: gt_n600 cached argmax inter-frame per-class disagreement."

**Convention declaration (charter step 2 — not ambiguous once traced):** the spike is charged to the
**scored-frame GT label `lstars[t]`** (the spiking class). This is the *same* convention xp1's
`_per_class_flip_counts` uses (charge a flip by GT class): a smooth-label / temporal-majority witness
outputs the repaired (neighbour-majority) label at the spike, so it is WRONG there and the flip is
charged to GT = `lstars[t]`. Cross-check: L67 "44% of CE-residual spikes = LANE" is the same object
from the target side, and the measured Lane share 43.55% lands on it. (Secondary convention — charge
by the neighbour-majority/repaired label — is recorded in `ddm_fl1_verdict.json`; it only redistributes
the 2.3% non-repairable mass and does not change any verdict.)

**Conversion (charter step 3):** per-class floor `d_seg_c = spike_px_c / (N·H·W)`, `S_c = 100·d_seg_c`,
with `N=598` interior pairs, `H·W = 384·512 = 196,608`. Registered 0.005318 is reproduced with **N=598**
(the interior-pair mean); the alternative N=600 (treating the 2 endpoint pairs as contributing 0 flicker,
a conservative d_seg-comparable variant) shifts every row down 0.33% and changes **no** verdict.

## §2 The per-class flicker floor table (MEASURED)

Sum-check anchor: per-class `d_seg` (/598) **sums to 0.0053184**, matching registered **0.005318**
to **+4.4e-7** (rounding of the registered value). PASS. Validation vs Part D shares: exact to <0.1%.

| class | spike px | share | **floor d_seg** | **floor S (/598)** | floor S (/600) | Part D ref share |
|---|---:|---:|---:|---:|---:|---:|
| Road | 222,135 | 35.52% | 0.0018894 | **0.18894** | 0.18831 | 35.5% |
| Lane | 272,322 | 43.55% | 0.0023162 | **0.23162** | 0.23085 | 43.6% |
| Undrivable | 46,312 | 7.41% | 0.0003939 | **0.03939** | 0.03926 | 7.4% |
| Movable | 33,469 | 5.35% | 0.0002847 | **0.02847** | 0.02837 | 5.4% |
| MyCar | 51,059 | 8.17% | 0.0004343 | **0.04343** | 0.04328 | 8.2% |
| **TOTAL** | **625,297** | 100% | **0.0053184** | **0.53184** | 0.53007 | 100% |

Aggregate cross-checks: px/pair (/598) = 1045.65 ≈ Part D 1046; repairable fraction (neighbours agree)
= 0.9773 ≈ Part D 0.977.

## §3 The join — floor vs ep641 residual vs corner-C vs ru1 typing (charter steps 4–5)

`residual-above-floor = max(0, xp1_residual_S − floor_S)`. corner-C alloc = gc13 §4 proportional-to-current
CONTROL split of seg 0.06 (**DERIVED**, NOT waterfilled — the waterfilled split is blocked on R1's
endpoint per-class descent rates). ru1 `gt_flicker%` = fraction of that class's ep399 residual flips
typed single-frame-flicker (cross-base structure label per §0.6).

| class | floor S (/598) | xp1 resid S (ep641) | **residual−floor** | corner-C S | **floor/corner-C** | ru1 flicker% of resid |
|---|---:|---:|---:|---:|---:|---:|
| Road | 0.18894 | 0.18845 | ≈0 (−0.0005, at floor) | 0.02652 | **7.13×** | 37.0% |
| Lane | 0.23162 | 0.12589 | 0 (−0.106, **below floor**) | 0.01771 | **13.08×** | 66.8% |
| Undrivable | 0.03939 | 0.05574 | **+0.01635 (above)** | 0.00784 | **5.02×** | 40.2% |
| Movable | 0.02847 | 0.03792 | **+0.00945 (above)** | 0.00534 | **5.33×** | 38.5% |
| MyCar | 0.04343 | 0.01840 | 0 (−0.025, **below floor**) | 0.00259 | **16.77×** | 54.0% |
| **TOTAL** | **0.53184** | 0.42640 | +0.0258 (net −0.104) | 0.0600 | **8.86×** | 49.5% (agg) |

**Reading of the join (DERIVED):**
- **Two classes sit ABOVE their smooth-label floor: Undrivable (+0.01635 S) and Movable (+0.00945 S).**
  This is *honestly-reachable headroom* — removable by continuation WITHOUT beating the smooth-label
  floor (the witness is simply worse than a temporal-majority oracle there = stable-structure erasure).
  Undrivable is **exactly the eroding-and-unguarded class** gc13 §2.0/B1 named (+0.00204 S/window): its
  erosion is entirely in the reachable regime → a guard on it protects genuine headroom, not a
  flicker-limited quantity. Movable is above-floor but descending (−0.01778/window) → continuation is
  already draining it. **This is the load-bearing per-class result for cg1's first guarded set.**
- **Three classes sit AT/BELOW their smooth-label floor: Lane (−0.106), MyCar (−0.025), Road (≈0).**
  The witness already *pierces* these classes' smooth-label floors via per-pair conditioning (Lane
  hardest: residual 0.126 is 46% below its 0.232 floor). ru1 corroborates: Lane's residual is 66.8%
  flicker-typed, MyCar 54.0% — flicker-dominated residuals in the piercing regime. Their remaining
  residual is NOT smooth-label-flicker-limited; it is the erasure/thin-structure tail (ru1: Lane→Road
  38.5% deepest bulk, tail 65% Lane) that needs positional carriers, not smooth-label descent.

**Falsifier (charter step 5, pre-registered by gc13 R3):**
- **LITERAL trigger:** every per-class smooth-label floor exceeds its corner-C allocation — Road 7.1×,
  Lane 13.1×, Undrivable 5.0×, Movable 5.3×, MyCar 16.8×. By the literal rule ("any class floor above
  its corner-C allocation ⇒ RE-WATERFILL-REQUIRED"), the trigger fires for **all 5 classes**.
- **SCOPED verdict (the correct consumer action): NO re-waterfill.** The trigger is EXPECTED and
  RE-CONFIRMS the §0.2 law per class: the smooth-label flicker floor is FORMULATION-scoped and is NOT
  a hard constraint on the corner-C witness class, which is a **per-pair phase-faithful renderer** that
  provably pierces it — PR130 existence (d_seg 2.966e-4 = S 0.02966, **18× below** the aggregate floor
  and below corner-C 6e-4), FEED-ma appearance-phase 0.00086, and our own ep641 witness 0.004264 all sit
  below the 0.005318 smooth-label floor. "Re-waterfill with smooth-label floors as HARD constraints" is
  the FORBIDDEN cargo-cult reading (operator 2026-07-15). **The binding sandwich lower-bound leg is
  ru1's GT-jitter-typed reachable (~6e-4 ≈ corner-C), not the smooth-label 5.3e-3.**
- **What the per-class floor IS binding for:** it RANKS the per-class phase-faithfulness debt a
  corner-C witness must pay — **Lane #1 (0.2316 S, 13.1×), Road #2 (0.1889 S, 7.1×)** — the design input
  for the appearance-phase / lane-corridor carriers (ru1 named carriers; T1 phase-advection lever).
  **Binding class (for RE-WATERFILL): NONE. Binding class (for phase-faithfulness debt): Lane.**

## §4 Verdicts (typed)

| item | verdict |
|---|---|
| per-class flicker floor (S-units) | **MEASURED** (cached `lstars`, $0); sum-check to registered 0.005318 PASS (+4.4e-7); Part D shares reproduced <0.1% |
| convention = charge-by-`lstars[t]` | **DERIVED** (matches xp1 `_per_class_flip_counts`; L67 44%-Lane cross-check) |
| denominator = 598 interior pairs | **DERIVED** (uniquely reproduces registered 0.005318) |
| residual-above-floor: {Undrivable, Movable} | **DERIVED w/ receipt** (xp1 ep641 − floor); Undrivable = gc13 B1 eroding-unguarded class; INSTANCE scope (one endpoint) |
| falsifier literal-trigger (all 5) | **MEASURED**; **SCOPED verdict = NO re-waterfill** (formulation-scoped floor, pierced; binding leg = ru1 ~6e-4). FORMULATION-level negative on the "hard floor" reading |
| binding phase-faithfulness debt | Lane (#1, 13.1× corner-C) — DERIVED ranking, design input not a wall |

## STORES CONSULTED (multi-pass)

**Pass 1 (custody + producers):** gt_n600.npz (ch1 S8 lineage) · `ddm_gc13_optimal_control_shape_20260731.md`
(§4 corner-C, §5 sandwich, §9 R3) · `flicker_transform_geometry_term_design_20260710.md` §2.4 Part D +
§5 registered eq · `ddm_ch1_recursive_confound_pass_20260728.md` (S8 gt_n600 CLEAN) ·
`ddm_xp1_20260731/xp1_verdict.json` (ep641 per-class S) · `ddm_ru1_20260729/atlas_flat.npz` +
`ddm_ru1_recursive_upstream_endpoint_typing_20260729.md` (gt_flicker typing).
**Pass 2 (laws):** `feedback_flicker_floor_not_hard_fire_phase_stack_stop_deferring_20260715.md`
(FORMULATION-scope law) · MEMORY established-findings (flicker=GT-floor; SegNet class order) ·
gc13 §0 laws (staleness confound, generic-basis, objective-is-min-S-over-set).

**Pointer honesty:** 0.1910828242 [contest-CPU] UNMOVED. Everything here is MEANS: a decomposed floor +
a scoped falsifier, feeding cg1's guard ledger (R2/R3 consumers). No archive, no scorer forward, no ΔS.
