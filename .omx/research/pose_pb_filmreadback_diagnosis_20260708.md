# P-B POSE FiLM-READ-BACK DIAGNOSIS — which of the 3 hypotheses plateaus d_pose at 1.79 (#248 P-B)

**Date:** 2026-07-08 · **Axis:** `[macOS advisory, run-1 existing telemetry + code-trace, n600]`
NON-PROMOTABLE · **$0, read-only** (pid 63069 + run dir UNTOUCHED; NO launch, NO training) ·
**Pointer contest-CPU 0.19110 UNMOVED — MEANS.** verdict_scope tags inline.

Launch-critical question (FEED-posev75): run-1 (crucible_v6) emits the store-nothing ξ pose carrier
ACTIVE (`--w-pose 1.0 --pose-carrier --pose-carrier-residual-mode table --pose-carrier-source
generated`) yet d_pose is PLATEAUED at 1.86→1.80→1.79 (ep100→175), NOT the ancestor 3.4e-5. v7.5
inherits this pose path unchanged. Distinguish **H-consume** (render ignores ξ) / **H-starve**
(pose grad starved) / **H-target** (the carrier target itself is insufficient on the witness).

## STORES CONSULTED
`counterforce_insufficiency_deepmath_20260708.md` · DAG FEED-posev75 + FEED-roadfloor(fix) ·
memory L68 (pose OPEN+unmeasured on witness) / L69 (store-nothing ξ derive-H / break-even error bars) /
L18 (ancestor = lessons not transferable NUMBERS) · CLAUDE.md §pose-solved ("a NAIVE bolt-on the render
doesn't consume does NOT lower d_pose") · #241 (store-nothing carrier wired) / #257 (derive-H) /
#314 D2 (pose_carrier_source drift) · code:
`experiments/train_levelset_witness_realized_through_R_mlx.py` (pose-carrier attach + render_fn
dispatch) + `src/tac/boundary_math/warp_real_luma_frame0.py` (the carrier module) +
`experiments/train_witness_realized_through_R_mlx.py::cpu_verdict_d_pose_batch` (the d_pose authority) +
`src/tac/witness_autoconfig.py` (crucible_v7 lineage). docs/operating_manual_craft_handoff.md.

Apparatus-validity precondition MET: run-1 is LIVE (pid 63069, ~12h elapsed, RSS ~56 GiB, log +
resume_state advancing to ep175; best d_seg 0.1255 @ ep175). All numbers below are re-read from raw
run-1 telemetry / source, not reasoned.

---

## VERDICT: **H-target** (the carrier target is structurally insufficient on the witness). Pose is a **HARD carrier problem, NOT a cheap consumption/wiring fix.**

H-consume = FALSE · H-starve = NOT-PRIMARY · H-target = CONFIRMED. Three measured/traced legs:

### Leg 1 — H-consume is FALSE (the render PROVABLY consumes ξ; code-trace, decisive)

The `generated`-source render_fn (trainer L3843-3856) dispatches by pair parity:
- **f1 (odd code)** → `_pc_witness_render` (the witness synthetic render; drives d_seg).
- **f0 (even code)** → warps the witness's OWN plain f0 render up to camera-native, then
  `_pc_impl.render_f0(src_native, pair_idx, code_vec, ste_round=True)`.

`render_f0` (carrier `warp_real_luma_frame0.py`) warps its source through `xi_eff = xi_stored +
scale·dxi` (`_xi_eff`, L709-710) via the plane-induced ground homography. The residual `dxi` is
**child-attached** (`model.pose_carrier = pose_carrier.impl`, trainer L3519) so it joins
`model.trainable_parameters()` and co-grads on the ONE `value_and_grad` (`xi_stored` frozen via
`self.freeze(keys=["xi_stored"])`, carrier L690). ⇒ The frame0 PoseNet reads is LITERALLY produced by
warping through the stored ξ + trained residual. This is NOT the "naive bolt-on the render ignores"
failure CLAUDE.md names. verdict_scope: **instance** (crucible_v6 generated/table config; code-certain).

### Leg 2 — H-starve is NOT the primary cause (the residual clearly RECEIVES gradient)

run-1 verdict d_pose trajectory (raw run.log): 9.575 → 3.586 → 2.084 → 1.859 → 1.896 → **1.796 →
1.793** (→ep175). It fell ~5.3× from the early value then FLATTENED at ~1.79. A starved residual would
sit at the deterministic init and not move; this one moved substantially, then hit a floor. So the
residual IS training against a target — it is hitting the target's CEILING, not being starved by the
w_seg=100 vs w_pose=1.0 balance. verdict_scope: **instance** (does not rule out that a larger w_pose
would nudge the plateau slightly — but it cannot cross the Leg-3 ceiling).

### Leg 3 — H-target CONFIRMED (two measured anchors, both ~5 orders above 3.4e-5)

**Anchor A — the deterministic warp CEILING (build-time s_t self-fit, real photometric luma, NO
residual).** At carrier build the trainer self-calibrates `s_t` on the frozen CPU-torch PoseNet over a
grid, pairing `[real_f0, warp(real_f0)]` vs the stored GT 6-DOF pose (`cpu_verdict_d_pose_batch`,
`train_witness_realized_through_R_mlx.py:790` — builds `[f0,f1]`, PoseNet, first-6 MSE vs GT). run-1
`s_t_fit` (run.log, pose_carrier stage): `{0.0:188.6, 0.02:11.18, 0.044:2.562, 0.08:11.13, 0.12:15.6,
0.16:17.3, 0.22:100.7, 0.3:133.6}` → best **d_pose = 2.562 @ s_t=0.044**. So even a FULLY-PHOTOMETRIC
real pair through the ground-homography warp cannot get PoseNet's 6-DOF output nearer than **2.562** to
GT. The homography reproduces ground-plane flow only; off-plane content (cars, buildings, sky) warps
wrong → PoseNet's learned pose regressor sees an inconsistent two-frame flow field → its output is off
by MSE ~2.5. A **rank-6 per-pair twist residual cannot repair a per-pixel flow-field mismatch** — it
only shifts the global warp. This is the carrier-family ceiling, measured.

**Anchor B — the trained-residual plateau (generated path).** With `dxi` trained under w_pose=1.0 on
the actual render pair `[warp(witness-f0-render), witness-f1-render]`, d_pose plateaus at **1.793**
(ep175). Both frames are the witness's task-space (non-photometric) render — worse still for PoseNet,
which was trained on real dashcam pixels. Even so it lands the same order (~1.8) as Anchor A's ~2.5.

**Both ≈ 1.8–2.6 — a ~50,000× gap to the ancestor 3.4e-5 that does NOT close.** The 3.4e-5 was NOT a
warp carrier: it came from a FULL-RGB reconstruction where PoseNet reads two photometrically-
reconstructed real frames `[recon_f0≈gt_f0, recon_f1≈gt_f1]` → pose ≈ GT. The task-space witness
renders NO photometric frame1, and a single-keyframe homography warp for f0 cannot reproduce true
optical flow. **The 3.4e-5 is ancestor-borrowed and does NOT transfer to this carrier design** (L18/L68).
verdict_scope: **formulation** (the warp-real-luma-single-keyframe + rank-6-twist carrier family is
capped ~2.5; NOT a paradigm kill of "store-the-ξ-don't-reconstruct" — a different carrier that gives
PoseNet a photometric pair is untested).

### Score consequence (why this gates v7.5)
√(10·1.793) ≈ **4.24** — pose ALONE contributes ~4.2 to S. A perfectly Road-un-floored v7.5 (d_seg→0)
would STILL score S ≈ 4.2 + 25·bytes/37.5M ≫ 0.19. **v7.5 cannot produce a sub-0.19 row while pose
plateaus at ~1.8, regardless of any d_seg lever.**

---

## #314 DRIFT RECONCILIATION — v7.5 IS on the SAME pose path run-1 measured (NO silent divergence)

FEED-posev75 flagged: crucible_v7 might default `pose_carrier_source="real_keyframe"` (the base
`WitnessConfig` dataclass default, autoconfig L529) while run-1 used `"generated"`. **RESOLVED — no
drift:** `crucible_v7`'s base config is `derive_store_nothing_205_config(...)` (autoconfig L1532), which
`replace(base, pose_carrier_source="generated")` (L1219). The crucible_v7 docstring confirms "Pose block
INHERITED from store_nothing_205" (L1624) / "pose block VERBATIM from v6" (L1770). The base-dataclass
`real_keyframe` default is never reached. The historical #314-D2 concern (the `fresh_seeded` lineage
inheriting the sealed `real_keyframe` because the flag emits only when `!= default`) was FIXED 2026-07-06
(`fresh_seeded` now always emits `generated`, regression-pinned). ⇒ v7.5 inherits d_pose ≈ 1.8 EXACTLY
as run-1 measured. verdict_scope: **instance** (code-certain for the current crucible_v7 derivation).

---

## EXACT NEXT ACTION

Pose is HARD on this carrier family — **do NOT** spend the next unit on "raise w_pose" (target
unreachable, not starved) or "wire FiLM residual" (FiLM only conditions `dxi` on the SAME unreachable
target). The binding decision is the pose REPRESENTATION, and it is council-grade (it changes what
PoseNet reads):

1. **CHEAP CONFIRMING MEASUREMENT ($0-local, next):** run the `--pose-carrier-source real_keyframe`
   A/B arm (w_pose>0, short) to MEASURE the true warp ceiling with a photometric f0 keyframe. Anchor A
   predicts it also floors ~2.5 (the ceiling is a homography/flow limit, not a source-quality limit);
   confirming this rules out "just use real luma" and forces the representation change.
2. **THE REPRESENTATION FIX (council, the real work):** give PoseNet a photometric PAIR. Honest options,
   each with its cost stated: (a) store the actual second real keyframe → PoseNet reads `[real_f0,
   real_f1]` → ancestor-like d_pose, but this COUNTS keyframe bytes (abandons the store-nothing premise;
   quantify the rate hit against the √(10·d_pose) win — the break-even in L69 must be re-derived with the
   MEASURED, not borrowed, d_pose); (b) a per-pixel / dense-flow carrier so the warp reproduces true
   optical flow (much harder; the rank-6 twist is provably insufficient); (c) accept a small counted
   photometric keyframe pair as the pose sidecar and drive d_seg-only on the witness.
3. **TRIALITY:** append DAG FEED-posepb (this diagnosis); no DSL change (investigation only); the
   `warp_real_luma_frame0_dpose_ceiling_v1` candidate equation (d_pose ≥ ~2.5 for the single-keyframe
   homography + rank-6-twist family, anchored on run-1 build-fit 2.562 + trained plateau 1.793) is
   **council-flagged, NOT registered** (the ceiling is measured; the real_keyframe arm + any dense-flow
   floor are owed before registration).

## FINAL STATE
$0 static code-trace + existing run-1 telemetry; n600; pid 63069 UNTOUCHED; NO launch/train.
**Pointer 0.19110 UNMOVED — MEANS.** Pose 3.4e-5 is ANCESTOR-BORROWED (never reproduced on the
witness); the witness-measured d_pose is ~1.8 and the carrier family is capped ~2.5.
