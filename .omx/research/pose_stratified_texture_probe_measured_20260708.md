# A1T STRATIFIED-TEXTURE PROBE — the T×D cell is FALSIFIED, MEASURED (2026-07-08)

**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` · **$0, CPU-torch, read-only** (crucible run-1 pid 63069 +
run dir UNTOUCHED; EMA snapshot-copied; NO launch/train/paid/GPU/MLX). **Pointer contest-CPU 0.19110
UNMOVED — MEANS.** Checkpoint = crucible **run-1** EMA (`levelset_witness_ema_mlx.npz`, ep200, n_pairs=600,
params=117527, self_orient, w_pose=1.0). Positive control reproduced (`d_pose([gt_f0,gt_f1])` = 2.1e-12
→ instrument trusted). Harness (`pose_stratified_texture_probe.py`) REUSES `pose_mladder.py` +
`pose_aperture_probe.py` EXACTLY: same `renders_n24.npz` witness renders, same through-R frozen CPU-torch
PoseNet/SegNet authority, same `GroundHomography` warp + `band_texture`/`add_texture`. n=24 (n600 OWED).

STORES CONSULTED: `pose_aperture_probe_measured_20260708` (c2adba9aa — the A0T falsification + the
wrong-flow-observability mechanism this probe tests + the reused harness) · `openpilot_depth_witness_term_design_20260708`
(a2d8f74e9 — the per-cell flow table this A1T flow implements) · `stratified_depth_warp.py` (landed,
bit-parity to A0) · `pose_mladder_depthwarp_measured_20260708` (A0 1.685 / A2 1.486 / A2+ 1.223) ·
CLAUDE.md L80 class order (MEASURED).

---

## HEADLINE — the T×D combination (texture × per-cell stratified flow) is **FALSIFIED** (verdict_scope: FORMULATION / this store-nothing geometric carrier + run-1 ckpt)

The untested cell of the pose experiment matrix — **texture advected by the PER-CELL STRATIFIED flow**
(A0T tested texture × global-H; A2+ tested a solve × flat render; A1T = texture × per-cell flow) — does
**NOT** collapse d_pose. **A1T-best = 2.608** (P=24px, peak amp=2), which is ABOVE **both** flat
baselines (A0 flat 1.685, A2+ flat-stratified 1.223), monotone-increasing in amplitude, and per-cell is
**worse than global-H at matched settings** (P32/amp4: A1T 34.9 vs A0T 15.1). The refined
wrong-flow-observability mechanism predicted A1T ≪ 1.223; it is REFUTED for the geometric stratified
carrier.

The **branch that fired: "still-wrong → scale diagnostic"** — and the scale sweep found **no
units/convention bug**: sweeping `s·ξ` over {0.25, 0.5, 1, 2, 4} + translation sign-flip gives a monotone
descent toward the flat floor as s→0 (s0.25 = 22.8, s1 = 34.9), with NO sharp optimum away from s=1 and
the translation-flip only mildly lower (24.1). No scale/sign hides a correct flow.

**The mechanism this pins (self-pair flow-correctness diagnostic — the decisive pre-check):** the per-cell
flow, made fully legible on real content, reads d_pose **183.5** vs the global-H flow's **165.6**. The
piecewise per-cell flow (ground H(ξ) + sky rotation-only R(ξ) + hood identity + off-plane H(ξ)) is a
**discontinuous composite, NOT a single rigid ego-motion** — PoseNet reads it as WORSE motion than the
single ground-homography, so making it legible RAISES d_pose. Correct scene flow requires the true
per-clip DEPTH field (Option-A depth-warp `K·T(ξ)·D·K⁻¹`), not a piecewise-geometric approximation. This
reconciles exactly with the aperture A0T falsification (wrong-flow-observability), the mladder A2+ floor
(1.223 even with a 12-DOF oracle solve), and the pose ROOT-CAUSE memo (the warp MODEL is the defect).

## THE MEASURED A1T GRID (LUMA texture, DC-matched, advected by the PER-CELL stratified flow; n24 median)

| period px@874 | peak amp | d_pose med | low-\|t\| / high-\|t\| | vs A0 1.685 / A2+ 1.223 | d_seg flip med / max |
|---:|---:|---:|---|---|---:|
| — (A0 flat, MEASURED) | 0 | **1.685** | 1.799 / 1.568 | baseline | 0 |
| — (A2+ flat-stratified) | — | **1.223** | — | baseline | — |
| — (A0T best, texture×global-H) | 4 | 15.14 | — | 9.0× above A0 | 0.037 |
| **24** | **2** | **2.608** | 2.81 / 2.32 | **1.55× A0 / 2.13× A2+** | 0.0081 / 0.0102 |
| 48 | 2 | 4.051 | 4.81 / 3.72 | 2.4× A0 | 0.0198 / 0.0233 |
| 32 | 2 | 4.508 | 4.92 / 3.78 | 2.7× A0 | 0.0135 / 0.0164 |
| 24 | 4 | 11.48 | 13.24 / 8.34 | 6.8× A0 | 0.0241 / 0.0289 |
| 48 | 4 | 19.44 | 22.74 / 13.43 | 12× A0 | 0.0496 / 0.0552 |
| 32 | 4 | 34.95 | 37.69 / 25.63 | 21× A0 (**> A0T 15.1**) | 0.0368 / 0.0438 |
| 24 | 8 | 114.05 | 122.0 / 100.3 | 68× A0 | 0.0728 / 0.0845 |
| 48 | 8 | 118.59 | 129.7 / 109.4 | 70× A0 | 0.0947 / 0.1127 |
| 32 | 8 | 145.04 | 148.2 / 142.5 | 86× A0 | 0.0842 / 0.1000 |

Monotone in amplitude at every period (P24: amp 2→4→8 = 2.6 → 11.5 → 114). **NO selective low-|t| rescue** —
low-|t| tracks or exceeds high-|t| everywhere, the aperture-fix signature is ABSENT (identical to A0T).
Even the smallest amplitude (2) — where "the flow is supposed to be RIGHT" — floors ABOVE the flat arms.

## THE SELF-PAIR FLOW-CORRECTNESS DIAGNOSTIC (why A1T does not help; n8)

Self-pair = (per_cell_warp(f0_render, ξ, partition), f0_render): same content, so the ONLY signal is the
warp displacement → measures "what motion does this flow depict" vs the GT pose target.

- **global-H self-pair d_pose = 165.6** (reproduces the aperture-memo diagnostic 166–186 → global ground-H
  is a huge WRONG ego-motion).
- **per-cell self-pair d_pose = 183.5** — HIGHER than global-H. Cell fracs (pair0): sky 0.48 (→R(ξ)),
  hood 0.26 (→identity), off-plane 0.03 (→H(ξ)), ground 0.23 (→H(ξ)).

The per-cell flow is **not** more correct as PoseNet reads it — it is *less* correct. Assigning sky
rotation-only and hood zero-motion while ground does full H(ξ) produces a piecewise field inconsistent
with any single rigid 6-DOF ego-motion (the object PoseNet was trained to read from real coherent scene
flow). This directly predicts the grid: painting texture to make this flow legible pushes d_pose toward
183, worse than A0T's 166. The A1T-best 2.6 < A0T-best 15.1 is ONLY because A1T tested amp=2 (smaller than
A0T's grid), not because per-cell flow is better — at matched amp it is worse.

## THE SCALE / CONVENTION DIAGNOSTIC (the fired branch; P32/amp4, n24 median)

| s·ξ | s0.25 | s0.5 | s1 | s2 | s4 | translation-flip |
|---|---:|---:|---:|---:|---:|---:|
| d_pose med | 22.78 | 30.39 | **34.95** | 37.95 | 41.05 | 24.07 |

Monotone descent toward the flat floor as s→0 (the trivial "no legible flow painted" limit), NO sharp
optimum away from s=1, per-pair min floors ~12.7 at every scale, and the translation flip drops only to
24.1 (a real sign-bug would collapse to ~flat 1.7). s1 reproduces the A1T P32/amp4 = 34.95 exactly
(internal consistency). **No units/convention bug in the ξ chain hides a correct flow.**

## A2T-STRATIFIED — gated RED, not fired (means discipline)

Charter step 4 gated the 6-DOF ξ_eff solve on the textured per-cell pair on "A1T drops materially." A1T
did NOT drop (it rose above the flat arms). The gate is RED, so A2T-stratified was not run. Its outcome is
already bounded: (a) the scale sweep already probed the ξ scale/sign neighborhood and floored per-pair
~12.7 with no rescue; (b) the sister aperture A2T (6-DOF solve on the textured global-H pair) measured
12.65 — ~8.5× WORSE than the flat-pair A2 1.486; (c) the off-plane affine A2T would add solves only the
~0.5%-mass off-plane cells, which A2+ already showed buy ≤10%. A full solve would confirm the same
wrong-flow-family ceiling at real cost. Do LESS but REAL: gate honored.

## SYNTHESIS + verdict_scope

- **T×D (texture × per-cell stratified geometric flow): FALSIFIED at the FORMULATION level** for the
  store-nothing geometric carrier on crucible run-1. NOT a family/paradigm kill.
- **The binding constraint (now triangulated three ways):** the flow a cheap store-nothing carrier can
  paint — global ground-H (A0T) OR piecewise per-cell-geometric (A1T) — is the WRONG scene flow; making it
  legible amplifies the error. The per-cell approximation is *more* discontinuous → *worse* (183 vs 165).
  Off-plane parallax mass is ~0.5% (mladder Rung-0), so it cannot be the lever. Correct scene flow needs
  the true per-clip DEPTH field, which the geometric carrier does not have.
- **NOT refuted (named reformulations that remain open):** (a) **Option-A depth-warp** — per-clip STORED
  depth D + stored ξ, SfMLearner `K·T(ξ)·D·K⁻¹` (a carrier whose flow is CORRECT because it uses real
  depth, not a piecewise-geometric approximation) — the DECISIVE next probe is **L2** (mono-depth on
  real_f0 → depth-warp → PoseNet, $0); (b) a dedicated JOINT pose-descent training RUN where the RENDER
  co-adapts (R1-class — the only measured path to low pose); (c) pose-as-budget-item + #238.
- **d_seg trade:** texture flips 0.8% (amp2) → 9.5% (amp8) of SegNet argmax on f1 (not sub-margin even at
  amp2). Moot given d_pose already refutes; a build (which will NOT happen for this carrier) would need the
  interior SDF gate.
- Owed before any promotable pose number: n600 + exact-eval. All rungs `[macOS-CPU advisory]`.

## TRIALITY / EQUATION

Canonical equation `morse_smale_stratified_parallax_dpose_v1` gains the MEASURED A1T advisory anchor
(`a1t_stratified_texture_falsified_crucible_run1_20260708`): texture advected by the per-cell stratified
flow does NOT collapse d_pose (best 2.6 > A2+ 1.223 > A0 1.685); self-pair diagnostic (per-cell 183 >
global 165) + scale sweep (no s≠1 optimum) show the piecewise geometric flow is wrong, not just
unobservable; wrong-flow-observability REFUTED for the geometric carrier; correct flow needs Option-A
stored depth; n600 + exact-eval OWED. No DSL change (investigation/probe, no new trainer lever fired).

## FINAL STATE

$0 CPU-torch; pid 63069 UNTOUCHED; NO launch/train/paid/GPU/MLX. **Pointer 0.19110 UNMOVED — MEANS.**
Scratch: `pose_stratified_texture_probe.py` + `a1t_grid_n24.json` + `scale_sweep_n24.json` under the
session scratchpad.
