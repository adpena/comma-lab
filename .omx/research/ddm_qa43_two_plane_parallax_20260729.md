---
schema: ddm_qa43_two_plane_parallax.v1
date_utc: 2026-07-29
arm: MAIN (operator openpilot pointer, fired inline; QA43 stage-1a)
lane_id: "lane_ddm_qa43_two_plane_parallax_20260729"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory; per-pair realized through the real receiver + frozen PoseNet; composed byte-close + n600 evaluate gate OWED]"
operator_verbatim: "Openpilot has polytope and other related to yaw and rotation on multiple axes and near vs far and parallax"
tool: "experiments/ddm_qa43_two_plane_parallax_probe.py (commits 3f570b5fc2 v1, 04e59e5933 v2 multi-start)"
data: "SSD ddm_qa43_20260729/{two_plane_probe.partial.jsonl (v1 n8), two_plane_probe_v2.partial.jsonl (v2, INTERIM n35 of 112, resumable), probe*_run.log}"
status: "INTERIM n35/112 — sweep resumable + running; §5 final totals appended at completion"
---

# ddm_qa43 stage-1a — TWO-PLANE (near/far parallax) warp: the pose-tail generator fix

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** All rows advisory.

## §1 The derived mechanism (operator pointer → measured chain)

The shipped receiver reconstructs frame_0 = warp(frame_1, H) with ONE plane-induced homography
H = K(R − t·nᵀ/d)K⁻¹ applied to the WHOLE frame (and ships `s_r=0`: rotation inert). Measured
consequences (uh1 + this arm): the chart removes 73.8% of NON-tail pose residual but only 12.8%
of tail residual; the tail's within-chart correction is 98.1% rank-1 along **dim-0 = forward
speed** (+32.6 near-constant) — the rotation/translation aliasing of a single planar homography:
the solver substitutes speed for turn because a one-plane warp gives FAR content (Undrivable =
sky/far, ~49% of area) ground-depth parallax it cannot have. The derived cure is the per-class
MULTI-PLANE warp: far → H∞ = K·R·K⁻¹ (`s_t=0`, d→∞) · ground (Road/Lane/Movable) → full H at
camera height 1.22 m · hood (MyCar, static) → identity. Class masks come from the ALREADY-SHIPPED
partition; the compose is generic receiver code (rule-118 FREE) → **~zero marginal counted bytes**
(same 6 f16 params/pair as RUNG P0; +~75 B selector map). Probe mask stand-in = GT lstars
(nearest-upsampled 384×512→874×1164); decoded-partition mask delta owed at the composed gate.

## §2 v1 (single-start, 8 worst pairs) — the poisoned-start discovery

Controls EXACT on all pairs (single-plane @ cached p★ reproduces cached d_pose_solved to the
digit — substrate identity verified). Two breakthroughs (83: 2.177→0.203; 38: 2.132→0.261) proved
the 0.78-class "content limit" partly a GENERATOR limit; 3 losses all shared one signature: the
zero-rotation start makes H∞ = identity → the far field FREEZES → GN cannot escape in 4 relins.
Verdict-scope: the v1 losses were INSTANCE(single-start), not the family.

## §3 v2 (multi-start {p0, p★}, top-24) — the cure confirmed

19/24 wins, 11 >2×; all three v1 poisoned pairs converted (82: 3.15→0.53 · 71: 2.16→0.36 ·
72: 2.83→2.56). Top-24 selection total 47.37→28.15 (−40.6%).

## §4 INTERIM full-tail state (n=35 of 112, controls 35/35 exact)

- 28/35 wins, 19/35 >2×; selection Δ −25.54 on single-total 57.79 (**−44.2%**).
- Deepest wins at LOWER tail ranks: pair 19: 0.939→**0.0016** (587×) · 122: 0.811→0.052 ·
  63: 0.842→0.056 · 109: 0.899→0.051 · 91: 0.951→0.071 — milder turns are MORE fixable
  (content limit binds less, geometry dominates), so the remaining 77 pairs should hold rate.
- **Composed pose axis, all per-pair realized:** warp 1.4881 → RUNG P0 (full-600 6dof f16,
  +7.2 KB) 1.2630 → +two-plane selection (n=35 so far, ~+75 B) **1.0814** = measured
  **−0.4067 S** from the operating row's pose member at ≤7.3 KB total marginal bytes.
- #404 ratios: −0.4067 = 19.6% of the 2.08 total gap; ≈30% of the pose-axis gap
  (1.4881→~0.13 target-grade); bytes-equivalent: −0.4067 S rate-priced would cost ~611 KB.

## §5 FINAL totals — APPENDED AT SWEEP COMPLETION (owed; sweep resumable, one silent kill
at pair 35 relaunched; killer unidentified — log clean, no traceback; watch RSS if it recurs)

## §6 Routing
- QA43 stage-0 (compose P0) + stage-1a (this) MEASURED; stage-1b free-class structure probe now
  aims only at the post-two-plane RESIDUAL tail (the pairs two-plane cannot fix: 77/82/90-class);
  stage-2 k-sweep re-scoped accordingly. v10 row-12 pose-in-burn: pressure REDUCED by −0.407
  measured at ~0 bytes; verdict still INSTANCE-scoped, re-adjudicate at full-112.
- Composed-candidate gates (Knee-A/B, byte-close, n600 evaluate) run WITH P0 + two-plane +
  selector active — receiver grammar v4 adds the multi-plane compose (generic code) + selector
  bits; knee-law composed re-solve after Knee-B token drops (they attack the same road-plane cue).
- Refinements queued in-row: 3rd plane for Movable (per-pair depth scalar ~2 B) · seam-blur A/B ·
  decoded-partition (vs GT) mask A/B at the composed gate.
