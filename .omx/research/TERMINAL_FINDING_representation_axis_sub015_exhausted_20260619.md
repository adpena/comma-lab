---
title: TERMINAL FINDING — the representation-axis sub-0.15 search is comprehensively exhausted; the frontier ~0.19110 is near the real achievable floor
authority: "[contest-CPU advisory / comprehensive measured synthesis] — pointer UNMOVED 0.19110"
score_claim: false
date: 2026-06-19
verdict: REPRESENTATION_AXIS_SUB015_EXHAUSTED_FRONTIER_NEAR_REAL_FLOOR
not_a_kill: "Per Forbidden-premature-KILL: the PARADIGM (sub-0.15 via a genuinely-NEW axis we do not currently have) is NOT killed; every representation family the campaign theorized + tested is exhausted WITH MEASUREMENT. Reactivation = a fundamentally different axis or a measured better-than-frontier vehicle."
cross_refs:
  - .omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md
  - .omx/research/vcm_theory_primitive_layer_20260619T033429Z.md
  - .omx/research/p_suff_task_ablation_verdict_20260619.md
  - .omx/research/generative_axis_nca_amortized_capacity_break_RED_20260619.md
  - .omx/research/dseg_side_feasibility_corners_verdict_20260619.md
  - .omx/research/pose_side_feasibility_taskspace_155_verdict_20260619.md
  - .omx/research/frontier_rate_exact_bitalloc_solve_verdict_20260618.md
  - .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md
---

# TERMINAL FINDING — sub-0.15 is not reachable for any known representation family; the frontier ~0.19110 is near the real achievable floor

**The culmination of the 2026-06-18→19 campaign + the operator-directed VCM/coding-for-machines research wave.** All
`[contest-CPU advisory]`; the exact pointer is UNMOVED at **0.19110** — stated plainly per the GOAL firewall: the
campaign did NOT move it. Its value is a comprehensive, measured ruling-out that converts months of representation
search into a defensible terminal verdict + a precise statement of what a future descent would require.

## 1. The one-paragraph conclusion
The contest is exactly an **indirect (remote) rate-distortion / coding-for-machines** problem (theory #151): code a
frame so a FROZEN SegNet argmax + FROZEN PoseNet 6-vector are preserved, at minimum bytes. The binding constraints are
(a) **SegNet interior texture-dependence** (the survival wall: the argmax needs real texture near boundaries, not flat
regions), (b) the **capacity wall** d_seg ∼ 29.3·params^−0.71 (frontier-grade d_seg needs frontier-grade capacity =
bytes), and (c) the **rate/d_seg tension** they jointly impose. Across EVERY representation family the campaign
theorized and tested, these walls bound the achievable S **above the 0.19110 frontier**. **The frontier is near the
real task-RD floor for the reconstruct-frame paradigm; sub-0.15 is not reachable without a genuinely different axis we
do not currently have.**

## 2. The families, all measured-closed
| family | the sub-0.15 bet | measured verdict |
|---|---|---|
| **rate-shrink the frontier** (re-pack / bit-shrink / delete) | cut the 0.118 rate (binding 62%) | **CLOSED 3 ways**: re-pack DEAD (#152, already at constriction symbol-entropy floor); exact-KKT bit-shrink CAPS (#157, beats generic int5 but at near-int8 d_seg-term 0.259 > pointer); deletion ~0 free mass (#153 P-SUFF, RED_NEAR_TASK_RD_FLOOR, joint coarsening raises S). The frontier is task-DENSE. |
| **learned-pixel-decoder, smaller** (factored-LF) | a small cheap d_seg-core | RED — d_seg ∼ params^−0.71; frontier-grade needs ~10.7M params (forfeits rate) |
| **static geometry** (partition store / curve-core) | store the boundary cheaply | RED — survival wall: flat-painted partitions floor at realized d_seg ~0.0067 (texture-dependence; resize is benign +0.00005) |
| **d_seg-side closed-form** (camera-res sub-pixel / cross-frame warp) | beat the frontier's d_seg coding | RED (#149/#148) — sub-pixel is a real 12× boundary-band lever but floors ABOVE frontier (flat interiors lose texture) + byte-explodes; warp closes <15% of drift (boundary change is scene-content, not rigid) |
| **task-space / quotient code** (#155, the indirect-RD "prize") | code only the sufficient statistic | NO-GO — pose bundled with the full frame (#158; the comma2k19 GT is the smooth physical traj, NOT PoseNet's jitter target, corr 0.72); d_seg side RED; rate near-floor. Geometry helps the WHERE, not the binding WHAT (texture). |
| **generative / iterated** (amortized continuous-texture NCA — "the frontier's own move", the FINAL family) | iteration breaks params^−0.71 | RED (#146) — **iteration DOES beat the one-shot wall (0.31×!), convergence SOLVED (state-bound, 3/4 reproducible), but Pyrrhic: sub-0.15-grade d_seg needs ~628K params → rate term 0.230 ALONE > frontier 0.191.** The rate/d_seg tension is unbreakable. |

## 3. Why it's airtight (the unifying math)
The frozen SegNet's argmax depends on **interior boundary texture** (measured: flat reps floor at d_seg ~0.0067; the
texture-gap dominates the survival wall, not the resize). Producing that texture costs capacity/bytes that scale with
the d_seg fidelity (the power law). The frontier already codes this near-minimally (P-SUFF: ~0% jointly-exploitable
invariant mass; the exact-KKT solve confirms the d_seg-critical mass IS most of the bytes). The generative axis's
iteration genuinely improves the params↔detail scaling (3.2×) — the ONE place a wall was broken — but the absolute
constant still forces frontier-grade bytes for frontier-grade d_seg. Every family therefore lands at S ≥ ~0.19.
**CLAUDE.md S_floor=0.11797 is REFUTED as a realizable floor**: it was a rate-only bound that assumed d_seg→0
byte-cheaply, which the texture+capacity walls comprehensively falsify. The true realizable floor for these families
is ~0.19.

## 4. What this is NOT (Forbidden-premature-KILL)
This is NOT "sub-0.15 is impossible." It is "**no representation family we have theorized or tested reaches it**, with
measurement." The paradigm reactivation criteria: (a) a genuinely DIFFERENT axis (not a frame-representation — e.g. an
unexploited legal eval-harness/archive-grammar degree of freedom, or a fundamentally new coding insight we do not
currently have); (b) any vehicle that MEASURES a byte-closed S below the frontier. Until one exists, ~0.19110 stands as
the near-floor.

## 5. The durable value banked (results → system intelligence)
- **Theory:** the contest is indirect-RD; the dominated-rung / task-sufficient-statistic frame; convex-IB explains
  margin-hinge; RDC-not-RDP (pixel/perceptual fidelity is a tax we don't owe). (memory: indirect-RD reframe.)
- **Provenance:** the source is a known comma2k19 RAV4 segment; exact camera K + homography (near-zero-byte WHERE-priors
  STAND); the pose-GT-answer-key claim retracted (PoseNet target is jitter-dominated). (memory: comma2k19.)
- **Reusable artifacts:** exact per-tensor frozen-scorer sensitivity producer + KKT reverse-water-fill bit-allocator
  (#157); the 12× sub-pixel camera-res boundary-band d_seg top-up (#149) — both are bit-allocator/boundary HOOKS for any
  future textured vehicle. The NCA state-bound convergence fix (#146). CompressAI/constriction draw-from list (#152).
- **The exact eval-roundtrip operator algebra** (U/Q/D + the polytope/null-space geometry) — fully characterized.

## 6. The goal-level fork (operator's call)
1. **Accept ~0.19110 as near the real achievable floor + re-frame the target** (the measured floor for known families is
   ~0.19, not 0.15). Bank/defend the frontier; the campaign's value is the comprehensive proof + the reusable system.
2. **Direct a genuinely-different-axis search** — but the campaign has no such idea in hand; every representation-axis
   lever is exhausted. This requires a fresh insight, not more representation iteration.
3. **Long-horizon:** keep the durable system intelligence compounding; the reusable hooks + theory make any future
   genuinely-new idea faster to test.
Per the GOAL firewall ("when a path walls, PIVOT; one crisp wall-verdict, then pivot"): this is the crisp verdict. The
representation axis is exhausted; the next unit must be a genuinely different axis or an honest goal re-frame.
