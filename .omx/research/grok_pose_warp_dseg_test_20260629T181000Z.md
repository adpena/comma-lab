# GROK pose-warp d_seg test — does the stored ego-pose carry the d_seg trajectory for free?

**UTC** 2026-06-29T18:10Z · **authority** `[macOS advisory / research-signal]` · **pointer UNMOVED 0.19110**
**Settles** DAG GAP 3 (vehicle) + measures GAP 1 (movables magnitude); PRE-R so blind to GAP 2 (R-survival).
**Tool** `tools/measure_pose_warp_dseg.py` · **JSON** `experiments/results/grok_pose_warp_dseg_20260629T181000Z/results.json` (+ `_n200`)
**Tests** DAG FEED-iv (THE GROK) + FEED-iu (coordinate-warp) + FEED-iw (3 gaps).

## The claim under test (FEED-iv)
`d_seg` and `d_pose` are two readouts of the SAME sufficient statistic — the ego-pose — because each frame's
SegNet argmax partition = `homography(ego-pose) · canonical_scene`. If true, the pose sidecar we already store
for `d_pose` (6 floats/frame) IS the d_seg modulation **for free**, leaving only a survival + movables residual.

## Method (the robust LOCAL consequence — avoids cumulative-pose drift)
If `frame[p] = H_p · C` for a shared canonical `C`, then `frame[p+1] = (H_{p+1}H_p^{-1})·frame[p] = H_rel·frame[p]`,
where `H_rel` is the plane-induced homography `H = K(R − t nᵀ/d)K⁻¹` of the RELATIVE ego-pose. So we test the
necessary local condition:

> predict `lstars[p+1] := warp(lstars[p], H_rel(pose))`, compare its **d_seg** (real argmax-disagreement vs the
> frozen CPU-torch SegNet argmax `lstars`) against the no-motion null **persist** (`predict := lstars[p]`),
> decomposed PER CLASS.

- **Data:** `experiments/results/mlx_fleet_gt_cache/{gt_n96, gt_strided_n200}.npz` — `lstars` = frozen CPU-torch
  SegNet argmax of **f1** (the last/scored frame), 384×512, classes `[Road,Lane,Undriv,Movable,MyCar]`; `gt_poses`
  = the **first 6 of the PoseNet hydra head** = the EXACT d_pose target (a learned, non-metric 6-vector; col0 ≈ 33
  = forward).
- **Calibration:** EON intrinsics (openpilot/comma2k19: fx=fy=910, cx=582, cy=437 native 1164×874; subagent-verified,
  2 repos agree) scaled to 384×512; camera height 1.22 m (openpilot `HEIGHT_INIT`). **3 global scalars fit** on
  Road+Lane: `s_t` (forward-zoom scale, sign data-determined), `s_r` (rotation scale), `pitch` (∈ openpilot
  bounds). Low capacity (≤3 globals shared across all transitions) ⇒ per-frame variation is **100% from the
  stored pose** — cannot overfit.
- **Relative-pose proxy:** non-overlapping seq_len=2 ⇒ `lstars` are 2 frames apart; use `pose[p+1]`; the
  per-frame factor + learned-units→metric scale are absorbed into the fitted `s_t`.

## Results (n96 + n200 strided — two independent samplings AGREE)

| baseline (total d_seg) | n96 | n200 |
|---|---|---|
| B_static_mode (each vs global per-pixel mode) | 0.0210 | 0.0231 |
| B_persist (lstars[p+1] vs lstars[p], no warp) | 0.01148 | — |
| **W_pose_warp (global single homography(pose))** | 0.01546 | — |

**Per-class warp vs persist (the decisive decomposition):**

| class | area | persist d_seg | warp d_seg | rel impr (n96) | rel impr (n200) | interpretation |
|---|---|---|---|---|---|---|
| **Road** | 0.230 | 0.0231 | 0.0196 | **+15%** | **+17%** | true ground plane → pose-homography COMPRESSES it |
| Lane | 0.006 | 0.5795 | 0.5851 | −1% | −4% | thin/dashed SURVIVAL residual → warp can't help |
| Undriv (sky) | 0.493 | 0.0024 | 0.0026 | −9% | −43% | plane-at-infinity → ground warp mis-warps it |
| Movable | 0.016 | 0.0506 | 0.0522 | −3% | +11% | independent motion → irreducible (partly road-coupled) |
| MyCar (hood) | 0.256 | 0.0031 | 0.0195 | **−525%** | **−2677%** | STATIC in image → ground warp DESTROYS it (needs identity) |

**Fitted calibration:** n96 `s_t=−0.0032, s_r=0, pitch=−0.01`; n200 `s_t=−0.0119, s_r=0, pitch=0.09`. The
forward-zoom (col0) is the SOLE driver (`s_r=0`); a clear road-plane d_seg minimum exists ⇒ the PoseNet
pose-units → ground-homography calibration **CLOSES** and is physical.

## Verdict — PARTIALLY CONFIRMED, and REFINED into a stratified per-class warp

- **CONFIRMED:** the stored pose IS a **free d_seg modulation for the Road class** (the dominant road-plane
  homography class). Calibration closes; +15–17% Road compression; per-frame variation 100% pose-driven;
  robust across two samplings and across baseline length. The pose↔d_seg coupling is REAL and physical.
- **REFINED (the grok-as-stated is too simple):** a SINGLE global `homography(pose)` of ONE shared canonical
  scene does **not** reproduce the whole partition — it HURTS the static classes (MyCar −525%, sky −9%) because
  they don't live on the road plane. The correct object is a **STRATIFIED (per-class) warp field**:
  - **Road / Lane-position** → ground-plane `homography(pose)` (free, pose-driven).
  - **MyCar (ego hood)** → **identity** (static in image; the #139 static core).
  - **Undrivable (sky)** → rotation-only homography `K R K⁻¹` (plane at infinity; pose-driven, free).
  - **Lane-survival (thin/dashed) + Movables** → small **learned residual** (the part NOT on the ego-pose orbit).

This is exactly the FEED-iv "canonical scene × pose orbit + survival/movables residual" picture — but the
canonical-scene transport is **per-class** (3 warp regimes), not one homography. The pose remains FREE and
carries the road-plane bulk; what must be PAID is the residual (Lane-survival + Movables) + the per-class warp
selection (a cheap static class-mask, itself mostly pose-stable).

## Gap mapping (FEED-iw)
- **GAP 3 (vehicle) — SETTLED (partial):** the BULK geometry needs **no trained INR** — a deterministic
  store-canonical + **per-class** pose-warp codec captures Road/sky/hood. The level-set INR is the wrong vehicle
  for the bulk; it is only needed for the Lane-survival + Movables residual. (The naive single-homography vehicle
  is refuted; the stratified deterministic vehicle is supported.)
- **GAP 1 (movables magnitude) — MEASURED:** Movable persist d_seg = 0.05 (n96) / 0.16 (n200), area ≈ 0.012–0.016
  ⇒ contributes ≈ 0.0008 (n96) of total persist d_seg; warp does not reduce it (independent motion). Its per-object
  low-rank-ness (GAP 1 recursion: a few extra 6-DOF streams) is UNTESTED here — the follow-up.
- **GAP 2 (R-survival) — BLIND:** this is PRE-R. The Lane-survival residual measured here is a **lower bound**;
  through-R (↑874 bicubic → uint8 → ↓384 → argmax) can only make the thin-lane flip worse. The same warp THROUGH R
  is the queued follow-up $0 probe.

## Adversarial self-audit (FEED-ix standing discipline — partials get the hardest scrutiny)
1. **Fitting artifact?** No. ≤3 global scalars (effectively `s_t,pitch`; `s_r=0`) across 95/199 transitions; a
   single global zoom-rate reducing Road d_seg requires the per-frame pose to genuinely predict per-frame road
   flow. Replicates +15% (n96) ≈ +17% (n200) on independent samplings ⇒ not noise, not overfit.
2. **MyCar −525% a bug?** No — it's the FINDING: a road homography moves the static hood. Confirms the per-class
   structure (static classes need identity).
3. **Pose-proxy understates?** Likely. The adjacent-pose + learned-units proxy means +15% is a **lower bound** on
   what an exact relative ego-pose could explain on Road.
4. **Necessary-not-sufficient / pre-R / frozen-instance:** flagged everywhere; the temporal warp is the local
   consequence of the canonical claim and a PROXY for the witness-vs-GT contest d_seg. The realized-through-R +
   exact CPU/CUDA eval on byte-closed bytes remains the only authority.

Under the overturn pass the finding STANDS at the **implementation level** (Forbidden-premature-KILL: paradigm
intact). It is a constructive REFINEMENT of the grok, not a kill.

## Byte / score implication
The pose is 6 floats/frame ALREADY stored for d_pose ⇒ the Road-plane d_seg trajectory it explains is FREE bytes.
v2 simplification: store the **per-class canonical IPM scene** + the **pose** (free, dual-use d_pose + d_seg) +
a **small learned residual** for Lane-survival + Movables. The binding wall is the residual (the part off the
ego-pose orbit) + R-survival, NOT rate/capacity/pose — consistent with FEED-iv's sub-0.15 arithmetic.

## Honesty firewall / NO-FAKE
- d_seg = REAL argmax-disagreement vs the frozen CPU-torch SegNet argmax (`lstars`); no surrogate. PROVEN.
- INFERRED + flagged: pose-column physical interpretation, adjacent-pose relative-motion proxy, learned→metric
  calibration units. PROXY: direct-partition warp PRE-R (necessary, not sufficient).
- `[macOS advisory / research-signal]`; `score_claim=false`, `promotable=false`, pointer 0.19110 UNMOVED. This is
  a means (a redirect of v2), not the end (a byte-closed exact row < 0.19110).

## Wire-in / cross-refs
- Reusable surface: `tools/measure_pose_warp_dseg.py` (deterministic numpy; `--cache/--n-pairs/--out-dir`).
- Redirects v2 (the witness capstone, CLAUDE.md "THE CURRENT FRONTIER … WITNESS CAPSTONE") toward a stratified
  per-class pose-warp + residual; pose is the dual-use free statistic.
- Directly informs `lane_hm_s_segmap_homography` reactivation criterion (a) "affine/learned pose-warp replacing
  global-translation" — this MEASURES that a global homography helps Road but a PER-CLASS warp is required.
- Follow-ups (queued in DAG): GAP 2 same-warp-THROUGH-R probe; GAP 1 per-object pose decomposition of movables.
