# Phase-stack efficacy probe — the SPEC_v10 §14.5(b) decisive $0 gate (2026-07-18)

**Pointer 0.19108 UNMOVED (means/measurement).** Every number below is
`[macOS-CPU advisory] NON-PROMOTABLE`, `score_claim=false`, measured on a LABELED
stratified sample (n24 stride-25 pairs {0,25,…,575} of the 600 scored pairs — the organ-B
sample convention). Nothing here is a score; the pointer moves only through
`upstream/evaluate.py`.

## The question (SPEC_v10 §14.2 / §14.5(b))

Does firing the phase stack (#424 phase-advection / #360 Force-3 tie-locus conditioning +
the #425 phase-carrier store leg) at organ-B's amplitude-OPEN `Road->Lane` strata REDUCE
d_seg through the real byte-closed decode (R → frozen CPU-torch SegNet)? §14.2's
curriculum-order law ASSUMED this; it was unmeasured.

## Apparatus (all REAL inputs — quoted; no surrogates)

- **Substrate:** banked v9c2 best EMA `experiments/results/banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz`
  (ep725, sha256 `b0a431e9…13cef`, n600 verdict d_seg 0.003458 per `levelset_best.json`).
  Live run dir untouched (read-only bank copy).
- **Decode/score chain (the authority decode, advisory axis):**
  `tac.witness_control.factorized_features.snapshot_witness_margins` → canonical
  `decode_levelset_torch` (render → bicubic↑874×1164 → round/clamp → **uint8 camera
  frames** = the real R realization) → frozen CPU-torch SegNet via the REAL
  `segnet.preprocess_input` (bitwise parity-asserted differentiable twin,
  `realization_regime._assert_preprocess_parity`) → argmax vs bit-exact GT `lstars`
  (`gt_n600.npz`, the frozen-scorer authority cache). Same chain and conventions as
  organ-B; fp32-EMA render (no int8-dequant leg — same convention as the organ snapshot;
  the shipped-verdict int8 leg is a labeled difference, not expected to flip signs).
- **"Fire the phase stack" post-hoc:** the phase stack's target is the GT sub-pixel tie
  coordinate (`phase_primitives.gt_tie_targets_numpy` — op-for-op the #424/Force-3
  target). On a frozen checkpoint the realizable firing is: at each `Road->Lane` flip
  pixel, the min-norm camera-space displacement crossing the exact pairwise margin
  (`delta* = −s·m·g/‖g‖²`, full-chain VJP — the necessity-solver/organ-B convention),
  applied to the camera frame, **rounded to uint8**, re-scored. Scales s ∈ {1.05, 1.5,
  2.0}. This is the ORACLE CEILING of post-hoc phase conditioning (consumes GT; never
  shippable; legal as a $0 gate).
- **Tool:** `tools/probe_phase_stack_efficacy_road_lane.py` (resumable per-frame
  checkpoints; RAM refuse-floor honored; peak probe RSS 3.1 GiB, avail never <60 GiB;
  live c2 run verified alive before/after; no governor path needed — no heavy/paid
  action). Artifact:
  `experiments/results/phase_stack_efficacy_probe_20260718/road_lane_n24.json`; ledger
  `.omx/state/phase_stack_efficacy_probe.jsonl` (2 rows — see the duplicate-writer note).

## MEASURED results (n24 sample; 1,500 of the 5,166 `Road->Lane` flip pixels treated, seed 0)

Baseline sample d_seg **0.003004** (14,170 flips; `Road->Lane` = 5,166 = 36.5% of flip mass).

**Framing validation first (both §14.2 premises CONFIRMED):**
- **Phase-addressable:** 1,451/1,500 = **96.7%** of treated `Road->Lane` flips lie ON the
  GT phase straddle band (`gt_tie_targets_numpy`, band 1.0, partner-dilated) — the
  stratum's residual IS boundary sub-pixel phase error, as the flicker framing claims.
- **Amplitude-OPEN (organ-B validated):** only 397/1,500 (26.5%) are sub-LSB-predicted
  (a_max < 0.5; median a_max 1.22), and under the composed treatment the targeted pixels
  DO flip to GT: fix rate 70.9% (s1.05) → 86.1% (s2.0); gross stratum recovery at s1.05 =
  (1,064 targeted + 3,493 untargeted-also-fixed)/5,166 ≈ **88%**. The flip mass is
  amplitude-reachable — the stratum is NOT realization-locked.

**The gate number — Δd_seg through the real chain (composed treatment):**

| scale | targeted fixed | untargeted also fixed | collateral NEW flips | d_seg sample | Δd_seg |
|---|---|---|---|---|---|
| baseline | — | — | — | 0.003004 | — |
| 1.05 | 1,064/1,500 (70.9%) | 3,493 | **+12,575** | 0.004703 | **+0.001699 (+56.6%)** |
| 1.5 | 1,255/1,500 (83.7%) | 4,321 | +20,736 | 0.006216 | +0.003213 (+107.0%) |
| 2.0 | 1,292/1,500 (86.1%) | 4,809 | +28,931 | 0.007842 | +0.004838 (+161.1%) |

Collateral is 2/3 **`Lane->Road` overshoot** (s1.05: 8,250 of 12,575; plus
`Undrivable->Road` 1,476): the composed correction does not stop at the GT boundary — it
paints the lane WIDER than GT. SegNet integrates over its ERF, so superposed per-pixel
phase moves act as a REGIONAL boundary shift (the treated 29% of the stratum dragged 88%
of it — and dragged the boundary past GT everywhere nearby).

**Isolation pass (150 single-pixel treatments, s=1.5 — the mechanism disambiguation):**
own-pixel fix rate **6%** (9/150); 14% of deltas round away ENTIRELY under uint8;
collateral mean 5.6/median 2 vs others-fixed mean 5.1 → net **+0.47 flips per treated
pixel** (103/150 net-non-harmful, median net 0). Alone, the min-norm move is
inert-to-slightly-harmful; the composed fix rates come from superposition.

## Verdict — **post-hoc DEAD, train-side/joint-solve OPEN** (verdict_scope: FORMULATION)

Firing the phase stack at `Road->Lane` **post-hoc does NOT reduce d_seg — it roughly
doubles it** at every scale arm, even with full GT oracle aiming (the lever's ceiling).
The wall is NOT the organ's sub-LSB rounding (26.5% sub-LSB; 80% of sub-LSB-predicted
pixels flipped under composition) — the wall is **cross-pixel collateral coupling**:
per-pixel amplitude moves through a region-integrating scorer cannot set adjacent phases
independently (isolated → rounds away/inert; composed → regional overshoot). This
sharpens organ-B's reading: *amplitude-OPEN ≠ harvestable-by-amplitude-moves*; the
binding constraint at `Road->Lane` is coupling, not amplitude.

Scope honesty (what this negative does and does not close):
- **CLOSED (this formulation):** decode-time post-hoc application of stored phase as
  per-pixel amplitude corrections — i.e. the naive "apply #425's decoded corrections to
  the frames at inflate" reading. Both composition modes measured net-negative.
- **NOT closed:** (a) **train-side joint descent** (#424/#360 losses price collateral on
  every pixel by construction — exactly the mode §14.2's curriculum stage specifies);
  (b) a **constrained joint SOLVE** (Dykstra projection onto {targets flipped ∧ rest
  unchanged} — #341/#342 terminal solve; structurally different from summed unconstrained
  crossings); (c) **coherent object-level phase moves** (warping a whole dash's texture by
  its GT phase δ(s) — the #425 curve-domain carrier's natural through-R application; the
  §13.11 label-space −34% XOR recovery corresponds to this move family, still owed
  through-R).

## v10 design implication (§14.2 / §14.5)

1. The §14.2 flicker-conditioning stage is admissible ONLY as a TRAINING stage (joint,
   collateral-priced) or as a constrained terminal SOLVE — never as a decode-time
   post-hoc frame edit. §14.2's stage placement (a training stage) survives; the probe
   rules out the cheap post-hoc shortcut and hence any v10 rate plan that counts on
   "store phase, apply at inflate."
2. The headroom is REAL: `Road->Lane` is 36.5% of remaining flip mass, 96.7%
   phase-addressable, 88% amplitude-reachable — worth ~0.0011 of the bank's 0.003458
   d_seg if fully harvested collateral-free. What training must WIN is the collateral
   trade, not the amplitude.
3. Echo of the pose lesson (L68): sidecar-SHAPED bytes, joint-descent VALUES. The #425
   carrier's bytes only pay when the render is SHAPED (by training or a constrained
   solve) to consume them — post-hoc storage without joint shaping is dead here exactly
   as it was for pose (different wall: coupling, not photometrics).
4. The curriculum-ORDER half of §14.2 (island-birth BEFORE phase) is untouched by this
   probe — we measured application MODE, not stage ordering.

## Duplicate-writer note (process, honest)

The harness SIGURG-killed the launcher bash at frame 11; the python survived (the
recorded phantom-death class — rtk-proxied `ps|grep` false-negative), so the detached
`--resume` relaunch briefly ran alongside it. Both writers are seeded-deterministic and
computed IDENTICAL treatments; the final artifact is the complete 24-frame doc + summary
+ isolation from the resumed writer; the ledger carries both rows (first without
isolation). No measurement ambiguity — verified by the exact recomposition identity
(mean per-frame baseline ≡ sample baseline) and identical per-frame lines across writers.

## Triality + stores consulted

- **DAG leg:** FEED row appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations leg:** `# FORMALIZATION_PENDING: single-checkpoint advisory-subset
  measurement (n24 stride, one vehicle) — register the collateral-coupling law
  (amplitude-open ≠ amplitude-harvestable) when the train-side or joint-solve A/B lands
  at n600` (FEED-519 precedent).
- **DSL leg:** n/a (measurement; no lever landed; the probe is a tool, not a trainer flag).
- **Stores consulted:** SPEC_v10 §13.11/§13.12/§14 · organ-B modules
  (`factorized_features`, `realization_regime` — reused, not re-derived) ·
  `phase_primitives` (#424 target definition reused) · canonical class order + n600-scale
  discipline (sample ≥ organ convention, labeled) · L68 pose photometric-wall memory ·
  phantom-death memory (hit it live, honored the psutil rule on the second check).

Artifacts: `tools/probe_phase_stack_efficacy_road_lane.py` ·
`src/tac/tests/test_phase_stack_efficacy_probe.py` ·
`experiments/results/phase_stack_efficacy_probe_20260718/road_lane_n24.{json,log}` ·
`.omx/state/phase_stack_efficacy_probe.jsonl`.
