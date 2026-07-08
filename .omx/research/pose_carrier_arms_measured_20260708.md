# POSE-CARRIER ARMS — MEASURED through the real byte-close/decode (#248 pose launch-fork)

**Date:** 2026-07-08 · **Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` · **$0, CPU-torch, read-only**
(run-1 pid 63069 + run dir UNTOUCHED; checkpoint SNAPSHOT-copied before use; NO launch, NO training,
NO paid eval). **Pointer contest-CPU 0.19110 UNMOVED — MEANS.** verdict_scope tags inline.
n = **8** bounded subset (memory envelope: run-1 live + 2 sibling research agents) → **direction-only**;
n600 owed before any promotable pose number. Sister of `pose_pb_filmreadback_diagnosis_20260708.md`
(H-target verdict) — this memo MEASURES the numbers that memo's "EXACT NEXT ACTION" #1/#2 called for.

## APPARATUS VALIDITY (precondition MET)
Positive control: `d_pose([gt_f0, gt_f1] vs gt_poses)` = **1.2e-12 ≈ 0** (max 2.4e-12). The frozen
CPU-torch PoseNet reproduces the cached `gt_poses` on the real GT pair → the instrument is trusted; every
d_pose below is a real through-decode number. Checkpoint = run-1 latest EMA (`levelset_witness_ema_mlx.npz`,
ep~ latest, n_pairs=600, params=117527, self_orient). GT = `mlx_fleet_gt_cache/gt_n24.npz` (pairs 0–7).

## THE TWO MEASUREMENTS (+ a controlled third that flips the premise)

All three feed the **frozen CPU-torch PoseNet the actual byte-closed / inflated frames**. frame1 is
ALWAYS the witness INR render (run-1 EMA); only frame0 differs. gt_poses = PoseNet on the real GT pair.

| arm | frame0 source | d_pose (n8 mean) | pose §rate (n8) | note |
|---|---|---:|---:|---|
| **store_nothing (generated)** | warp(**witness** f0 render, calib H) | **1.995** | **1,019 B** | *consistent* cartoon pair; ≈ run-1 telemetry 1.79 |
| **real f0 + witness f1** (meas 2) | **real gt_f0** (exact, unwarped) | **10.42** | (kf payload) | *inconsistent* photo+cartoon → WORSE |
| **warp_real_luma** (meas 1) | warp(**real** keyframe, calib H) | **37.4** | 34.4 MB / 24 kf | *inconsistent* + s_t=0.16 miscal → worst |

Same 8 pairs, same witness f1 across all rows → a controlled frame0 sweep. The 5×–19× spread is PURELY
the frame0 source.

### MEASUREMENT 1 — real_keyframe source ceiling: **REFUTED as stated; the ceiling is NOT source-independent-at-any-pair, it is CONSISTENCY-bound.**
- Byte-close `warp_real_luma` (real keyframe, per-pair ego H, s_t=0.16 default): **d_pose = 37.4**, NOT
  ~2.5. `store_nothing` (generated witness-render source): **d_pose = 1.995**. So the **generated source
  is ~19× BETTER than the real-keyframe source** on the actual decode — the opposite of "just use real
  luma." verdict_scope: **formulation** (byte-close deterministic carrier; s_t un-self-fit).
- WHY: PoseNet regresses ego-motion from the **flow between the two frames**, and is indifferent to
  photorealism. Two frames from the SAME distribution (warp(witness render) + witness render) form a
  *consistent* pair with a clean homography flow → d_pose ~2.0. Mixing a **real photo f0** with the
  **cartoon (task-space) witness f1** is an *inconsistent* pair (photo→cartoon "flow" is meaningless) →
  d_pose blows up to 10–37. The diagnosis's Anchor A (~2.5) was a *consistent* real pair
  `[real_f0, warp(real_f0)]`; my clean n8 reproduction of it gave 22.7 @ best s_t (NOT 2.5) — a
  subset/pitch-calibration mismatch (default pitch 0.02, n8 high-motion pairs), so I treat the exact
  ~2.5 value as **config-sensitive / not-load-bearing**; the ROBUST claim is the controlled 1.995-vs-10.42
  contrast above.
- **Warp-MODEL cap CONFIRMED (consistent pairs):** store_nothing deterministic = **1.995**, run-1
  trained xi_stored+dxi = **1.79** — the rank-6 per-pair twist residual shaves only ~11% off the
  deterministic homography floor. The homography cannot reproduce true 2-frame optical flow (off-plane
  content), so a consistent generated pair floors at ~1.8–2.0 regardless of the trained residual. This
  is the wall.

### MEASUREMENT 2 — store real f0 directly + f1-witness isolation: **store-f0 DOUBLY DOMINATED.**
- `d_pose([REAL gt_f0, witness_f1])` = **10.42** (median 10.12; per-pair 9.1–13.1, tight). Even a
  **perfect, exact real f0** leaves d_pose at 10.42 — because it is paired with the cartoon witness f1
  (pair inconsistency), and it is **WORSE than the store_nothing generated baseline (1.995)**. Storing a
  real f0 does NOT fix pose; it BREAKS the pair and makes it worse.
- **Keyframe byte cost (MEASURED through the real codec):** native-lossless per-pair keyframe =
  **1,434,015 B/keyframe** (34,416,357 B for 24 keyframes, brotli q11). n600 → **860.4 MB** →
  rate_term = 25·860.4e6/37,545,489 = **≈ 573**. `store_nothing` for the same is **1,019 B** total
  (rate_term ≈ 0.06). Even at `--pc-keyframe-downscale 2` (~4×) the rate_term is ≈ **143** — still
  catastrophic.
- **Break-even (re-derived with MEASURED, not borrowed, d_pose):**
  - ΔS_pose = √(10·1.79) − √(10·10.42) = 4.231 − 10.208 = **−5.98** (pose gets WORSE by ~6).
  - Δrate = **+573** (n600 native) / +143 (downscale 2).
  - Net ΔS ≈ **+579** — store-f0 loses on BOTH axes by orders of magnitude. REFUTED.

## SELECTED FIX: **warp-MODEL upgrade (dense/per-pixel flow). NOT store-f0. f1-witness "cartoon" is NOT the wall.**
- store-f0-paid: **REJECTED** (d_pose worse 1.995→10.42 AND rate +573). Doubly dominated.
- f1-witness-is-the-wall: **REJECTED as framed** — the cartoon witness f1 works FINE when paired
  consistently with a warped witness f0 (d_pose 2.0). f1 photorealism is not what PoseNet needs;
  pair-consistent flow is.
- **warp-model-upgrade: SELECTED.** The binding wall is the rank-6 homography flow model, which caps a
  consistent generated pair at d_pose ~1.8–2.0 (√(10·1.8)=4.24 pose contribution ≫ the 0.19 target).
  To break sub-2, the generated pair must reproduce **true optical flow** (dense/per-pixel warp or a
  learned flow field), NOT a global ground-plane homography — exactly the "rank-6 twist is provably
  insufficient" limit of the diagnosis Leg 3, now with the source question settled: keep the generated
  (witness-render) source, upgrade the flow. This is v8-coupling, council-grade (changes the render's
  frame0 synthesis), not a cheap knob.

## DEPLOY-PATH NOTE (secondary, corrected)
The checkpoint carries trained `pose_carrier.xi_stored (600,6)` + `pose_carrier.dxi (600,6)`; the
byte-close REBUILDS a deterministic GT-calibration and does NOT load them. For run-1's actual
`store_nothing/generated` config this is NOT catastrophic — the deterministic deploy (1.995) ≈ the
trained telemetry (1.79), so the trained residual is a ~11% refinement, not load-bearing. (My initial
worry that the deploy was "broken" was an artifact of reading the inconsistent-pair arms; corrected.)

## POSE 3.4e-5 STAYS ANCESTOR-BORROWED
Never reproduced on the witness. The ancestor read two photometrically-reconstructed real frames
`[recon_f0≈gt_f0, recon_f1≈gt_f1]` → a consistent PHOTOMETRIC pair with true flow. The task-space
witness renders no photometric f1; its best consistent-pair pose is ~1.8–2.0 (homography-flow-capped).
Ours-vs-borrowed: 1.79/1.995 = OURS (measured on the witness); 3.4e-5 = BORROWED (ancestor, does not
transfer).

## TRIALITY / CANDIDATE EQUATION
Investigation only — no DSL change, no equation registered. Candidate (council-flagged, NOT registered):
`witness_pose_pair_consistency_v1` — d_pose is governed by PoseNet's two-frame FLOW consistency;
a *consistent* generated cartoon pair floors at the homography model cap (~1.8–2.0 measured n8), a
photo+cartoon *inconsistent* pair (store-f0) is strictly worse (10.4, n8). Owed before registration:
n600 confirmation + a dense-flow floor measurement.

## FINAL STATE
$0 CPU-torch, n8 direction-only; pid 63069 UNTOUCHED; NO launch/train/paid. **Pointer 0.19110 UNMOVED —
MEANS.** Evidence packets: `experiments/results/levelset_packet_20260708T222911Z` (warp_real_luma) +
`…T223705Z` (store_nothing), each `inflated/0.raw` = 8 pairs. Scratch scripts + logs under the session
scratchpad.
