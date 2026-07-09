# APERTURE PROBE (A0T / A2T) — the aperture hypothesis is FALSIFIED, MEASURED (2026-07-08)

**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` · **$0, CPU-torch, read-only** (run-1 pid 63069 + run
dir UNTOUCHED; EMA snapshot-copied; NO launch/train/paid/GPU/MLX). **Pointer contest-CPU 0.19110 UNMOVED
— MEANS.** Checkpoint = crucible **run-1** EMA (`levelset_witness_ema_mlx.npz`, ep200, n_pairs=600,
params=117527, self_orient, w_pose=1.0). Positive control reproduced (`d_pose([gt_f0,gt_f1])` = 2.1e-12
≈ 0 → instrument trusted). Harness (scratchpad `pose_aperture_probe.py`) REUSES `pose_mladder.py`
(70649531f) EXACTLY: same `renders_n24.npz` witness renders, same through-R frozen CPU-torch PoseNet
authority, same `GroundHomography` warp. n=24 (n600 OWED before any promotable pose number).

STORES CONSULTED: `pose_legible_witness_aperture_design_20260708` (494f1a35c — the aperture diagnosis +
§4 A0T/A2T probe spec) · `posenet_scorer_side_deepdive_20260708` (0d3dcd5fb — the DERIVED texture spec:
isotropic band-limited, period 24–32 px @874, peak 8–16 uint8, non-AA-bilinear lower bound ~9 px, chroma
+ DC-match toggles, |t| stratification) · `pose_mladder_depthwarp_measured_20260708` (A0 1.685 / A2 1.486
/ A2+ 1.223) · `pose_taskspace_native_morse_smale_depth_warp_design` · CLAUDE.md L80 class order.

---

## HEADLINE — APERTURE HYPOTHESIS **FALSIFIED** (verdict_scope: FORMULATION / this store-nothing carrier + run-1 ckpt)

**Painting ξ-consistent, observable, isotropic band-limited texture into the A0 consistent pair does NOT
reduce d_pose — it dramatically INCREASES it, monotonically with amplitude, with NO selective low-|t|
rescue.** Best-case textured point (P=32 px, peak amp=4) = **d_pose 15.1**, ~9× WORSE than the flat A0
floor **1.685**; the rest of the derived-spec grid sits 27–118. The flat cartoon's floor is NOT a
correct-but-unobservable flow being starved by the aperture — it is a WEAK residual signal. Adding
observable texture forces PoseNet to read the carrier's ground-homography flow **W**, and **W is not the
true ego-motion** (diagnostic: warp(frame) vs the same frame reads d_pose 166–186). So observability
amplifies the carrier's flow ERROR rather than curing a deficit. **The wall is FLOW-MODEL CORRECTNESS,
not flow observability.** This reconciles exactly with the M-ladder (A2/A2+ solving DOF still floor ~1.2)
and confirms the design memo's own hedge (§4: "if A0T ≈ A0 the diagnosis is wrong → pose-as-budget-item").

Semantic-prior dominance (the coordinator's alternative: PoseNet ignores pixels, keys on scene priors) is
**also NOT the indicated mechanism** — PoseNet responds STRONGLY to the injected texture flow (d_pose
swings 1.7 → 118), so it is emphatically NOT ignoring the pixels. The refined mechanism is
**WRONG-FLOW-OBSERVABILITY**: a cheap store-nothing carrier can only paint the ground-homography flow, and
making that (wrong) flow legible is worse than leaving it flat.

## THE MEASURED A0T GRID (isotropic band-limited LUMA texture, DC-matched, advected by A0's H(ξ); n24 median)

| period px@874 | peak amp | d_pose med | low-\|t\| / high-\|t\| | vs A0 1.685 | d_seg flip med / max |
|---:|---:|---:|---|---|---:|
| — (A0 flat) | 0 | **1.685** | 1.799 / 1.568 | baseline | 0 |
| 32 | **4** | **15.14** | 17.55 / 13.82 | **9.0× WORSE** | 0.037 / 0.044 |
| 16 | 8 | 27.01 | 27.54 / 25.07 | 16× worse | 0.073 / 0.084 |
| 48 | 8 | 50.10 | 54.88 / 48.88 | 30× worse | 0.095 / 0.113 |
| 24 | 8 | 63.71 | 62.85 / 65.59 | 38× worse | 0.073 / 0.085 |
| 32 | 8 | 78.59 | 78.48 / 78.72 | 47× worse | 0.084 / 0.100 |
| 48 | 16 | 93.91 | 92.25 / 94.91 | 56× worse | 0.122 / 0.148 |
| 24 | 16 | 96.68 | 92.06 / 101.2 | 57× worse | 0.104 / 0.122 |
| 32 | 16 | 110.5 | 107.9 / 116.1 | 66× worse | 0.111 / 0.130 |
| 32 | 32 | 117.5 | 114.5 / 123.2 | 70× worse | 0.129 / 0.146 |

Toggles at (P=32, amp=16): **chroma ON** 92.2 (vs luma 110.5 — slightly less bad, still catastrophic);
**DC-match OFF** 110.4 (≈ ON 110.5 — no effect). Monotone in amplitude at every period (P=32: amp
4→8→16→32 gives 15→79→110→118). **NO selective low-|t| rescue** — the derived aperture-fix signature
(low-|t| tail dropping while high-|t| holds) is ABSENT; low-|t| tracks or exceeds high-|t| everywhere.
Every painting amplitude also flips 3.7–13% of SegNet argmax on f1 (texture is not sub-margin even at
amp 4) — but that d_seg guard is moot given d_pose already refutes the hypothesis.

## THE DIAGNOSTIC THAT PINS THE MECHANISM (why texture hurts)

- **warp(f0_render, ξ) vs f0_render = d_pose 166.8** ; **warp(f1_render, ξ) vs f1_render = 186.1** (n24).
  The ground-homography warp of ANY frame, paired against that same frame, reads as a HUGE ego-motion far
  from GT. So the carrier's flow W (calibrated s_t=0.16 for the A0 render-mismatch scheme) is NOT the true
  ego-motion flow. A0's low 1.685 comes from the FLAT residual of `warp(f0r)` vs the DIFFERENT render
  `f1s` reading weakly-near-GT — not from W depicting the right flow.
- Textured A0T advects the texture by W (`warp(f0r+tex)` → `warp(f0r)+warp(tex)`), so the texture carries
  the full W displacement while the untextured scene carried only the small residual. Making W legible →
  PoseNet reads W → d_pose blows up. Larger amplitude / coarser (lower-freq) texture ⇒ stronger legible-W
  ⇒ more damage (matches the monotone amp trend and P=48/16 > P=16/8 pattern).

## A2T — does the 6-DOF ξ_eff SOLVE revive on a textured pair? (best-shot P=32/amp4, n8)

**NO — the solve does NOT revive; texture is strictly harmful even to the SOLVE.** 6-DOF ξ_eff damped-GN
on the textured pair (best-shot P=32/amp4), init = A0 ξ: **d_pose median 12.65** (mean 14.09), from
init-median 14.52 — descends only −13% and floors ~12.65, which is **~8.5× WORSE than A2's flat-pair
1.486**. Per-pair finals 8.8 / 10.8 / 15.7 / 14.7 / 10.1 / 14.0 / 27.4 / 11.3. The reachable warp family
still only spans ground-homography flow; with texture painted on it, that wrong flow is now high-contrast
and the solver's minimum sits far above the flat-pair A2 minimum. (Charter gated A2T on "A0T shows a real
drop" — it did NOT; A2T was run anyway to close the question, and it confirms the falsification.)

## SYNTHESIS + HONEST NEXT STEP (verdict_scope ladder)

- **Aperture hypothesis: FALSIFIED at the FORMULATION level** for the store-nothing homography carrier on
  the crucible run-1 checkpoint. NOT a family/paradigm kill: the aperture DIAGNOSIS (flat interiors →
  unobservable flow) is simply not the binding constraint; the binding constraint is that the observable
  flow a cheap carrier can paint is the WRONG (plane-homography) flow.
- **NOT refuted (named reformulations that remain open):** (a) a dedicated JOINT pose-descent training RUN
  where the RENDER co-adapts (R1-class — the only measured path to low pose; a RUN not a post-hoc carrier);
  (b) **Option-A depth-warp** (per-clip stored depth + stored ξ, SfMLearner K·T(ξ)·D·K⁻¹) from the pose
  ROOT-CAUSE memo (560b16634) — a carrier whose flow is CORRECT (planar-homo γ(p)≡0 is the real defect,
  not observability); (c) pose-as-budget-item + #238 (the design's own honest fallback).
- **Implication for §5 of the aperture design:** do NOT build a pose-legibility texture term on this
  carrier — the probe GREEN gate is RED. Texture legibility only pays once the underlying flow model is
  correct (Option-A depth-warp); on a flow-wrong carrier it is strictly harmful.
- Owed before any promotable pose number: n600 + exact-eval. These rungs are `[macOS-CPU advisory]`.

## TRIALITY / EQUATION

Canonical equation `morse_smale_stratified_parallax_dpose_v1` gains the MEASURED A0T advisory anchor
(`aperture_probe_A0T_falsified_crucible_run1_20260708`): observable ξ-consistent texture on the cheap
carrier raises d_pose (best 15.1 ≫ A0 1.685), aperture-fix FALSIFIED, wrong-flow-observability mechanism;
n600 + exact-eval OWED. No DSL change (investigation/probe, no new trainer lever fired).

## FINAL STATE

$0 CPU-torch; pid 63069 UNTOUCHED; NO launch/train/paid/GPU/MLX. **Pointer 0.19110 UNMOVED — MEANS.**
Scratch: `pose_aperture_probe.py` + `aperture_a0t_v2.json` + `aperture_a2t_p32a4.json` +
`aperture_diag{,2}.py` under the session scratchpad.
