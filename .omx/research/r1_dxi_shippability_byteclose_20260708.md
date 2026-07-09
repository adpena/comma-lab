# #238 — R1's d_pose 0.0011 SHIPPABILITY through byte-close (ship the trained ξ_eff) — 2026-07-08

**Task:** #238, the decisive pose SHIPPABILITY gate. R1's trained `d_pose 0.0011` was custody-VERIFIED
(`r1_0011_custody_revalidation_20260708.md`) as a real training-side measurement, but it lives ENTIRELY
in the trained per-pair `pose_carrier.dxi` (600×6) table that the byte-close serializer did **not** ship
— so a byte-close read the deterministic no-dxi floor, not 0.0011. **The question:** does
`ξ_eff = xi_stored + dxi`, serialized and re-measured through the REAL inflate/decode, reproduce the low
d_pose? **Answer: YES — it survives byte-close.**

**One-line verdict: SHIPPABLE.** The trained pose ships in ~7.2 KB of COUNTED coded ξ and the realized
d_pose through the real byte-closed inflate = **0.001127 (n24)** — matching R1's training-side 0.001012 —
for a pose contribution `√(10·d_pose)` = **0.106** (≈ the symposium's 0.105). Every number below is
`[macOS-CPU advisory] NON-PROMOTABLE`. **Pointer 0.19110 UNMOVED.**

---

## 1. The connector (what landed in `tools/levelset_byte_close_and_eval.py`)

Pure byte-close serialization mode — NO trainer change (charter-sanctioned: "if it's purely a byte-close
serialization mode, a byte-close arg is fine"). Four minimal edits:

1. **`--pose-carrier-xi-from-ckpt`** (main): loads `pose_carrier.xi_stored + pose_carrier.dxi` from the
   SAME resolved checkpoint npz (the one whose INR weights are byte-closed → the render and the twist
   come from ONE consistent checkpoint) and passes ξ_eff as an override into the store-nothing serializer.
   Guards: requires `--pose-carrier --pose-carrier-mode store_nothing`; refuses `--select-arms` (each arm
   has its own dxi); fails loud if the npz lacks the pose-carrier keys.
2. **`--pose-carrier-dxi-scale`** (main): ξ_eff = xi_stored + scale·dxi. `scale=0` = the MATCHED no-dxi
   isolate (SAME fitted xi_stored, dxi off); default = the checkpoint's trained residual_scale (1.0).
3. **`build_pose_carrier_section(xi_override=…)`**: when supplied, ships the trained twist instead of the
   `xi_from_pose_calibration` recompute; H is still DERIVED at decode from the SHIPPED ξ with the same
   `pitch` the trainer used (R1 = 0.0; `--pc-pitch 0.0`, confirmed: `GroundHomographyGeom.eon(pitch=0.0)`
   → `n=[0,-1,0]`, K/d identical to the byte-close `homographies_from_xi` deriver).
4. **base-blob correctness fix (bonus):** `pose_carrier.*` (xi_stored/dxi) are NOT INR weights but older
   saves leaked them into `params`, so `build_levelset_blob` was int8-quantizing them into the base blob
   as DEAD counted bytes (inflate never reads them from base). Excluded them → base is INR-only and the ξ
   rate is attributed cleanly to the pose-carrier section (the canonical accounting cross-check made
   pose_carrier-aware to match).

Regression: `test_levelset_pose_carrier_byte_close.py` (16) + byte-close family (46 total) PASS; the
default (no `--pose-carrier-xi-from-ckpt`) path is byte-identical.

## 2. The measured A/B — realized d_pose THROUGH the real byte-closed inflate

R1 checkpoint `levelset_witness_ema_mlx.npz` (mod-dim 26, ep1108; `pose_carrier.xi_stored` absmean 0.2296,
`pose_carrier.dxi` absmean 0.00382, `__cfg_w_pose=1.0`). GT cache `gt_n600.npz` (frozen CPU-torch PoseNet,
through real R). n24 = the first 24 pairs (n STATED; the archive/ξ section is built for ALL 600 pairs
regardless of the inflate cap, so the byte/rate columns below are the true n600 archive numbers).

| config | carrier d_pose (shipped .raw) | ceiling (real f0, warp real f0) | ξ section B | pose √(10·d_pose) |
|---|---|---|---|---|
| calibration (wrong s_t=0.16) | 26.043 | 17.324 | 3,761 | 16.14 |
| **matched no-dxi** (ckpt xi_stored, `--dxi-scale 0`) | 0.02197 | 2.561 | 3,762 | 0.469 |
| **ship-dxi** (xi_stored + dxi, `--dxi-scale 1`) | **0.001127** | 1.958 | **7,195** | **0.106** |

Reading it:
- **The dxi survives byte-close.** ship-dxi realized d_pose **0.001127** ≈ R1's training-side **0.001012**
  (ep1108). The quantized (q_levels=4096, delta_ar coded) ξ_eff reproduces the descent through the real
  decode. NOT degraded.
- **The dxi is the 20× refinement** on top of an already-pose-legible witness pair: matched no-dxi (same
  fitted xi_stored, dxi OFF) already reads 0.022 (the co-adapted witness render is self-consistent for
  PoseNet); the trained dxi drops 0.022 → 0.0011.
- **The `ceiling` column** (PoseNet(real f0, warp(real f0, H))) is the keyframe-independent warp reference:
  2.561 no-dxi == the custody memo's "cap ~2.56" (xi_stored-alone), 1.958 with dxi. dxi improves the
  true-ego-motion match too.
- **s_t calibration matters:** the default byte-close s_t=0.16 (NOT R1's fitted s_t) gives 26.0 — that is
  the "no-dxi floor" a NAIVE byte-close (wrong calibration) reads; the MATCHED isolate uses R1's own
  fitted xi_stored.

## 3. Counted bytes + rate (rule-118 honest accounting)

The trained ξ_eff is video-derived → **COUNTED**. Store-nothing v2 payload (per-channel int16 quantized ξ,
q_levels=4096, delta_ar temporal-Δ + arithmetic coder): **6,634 B coded** (raw-ξ ref 7,232 B; coder barely
helps because the per-pair dxi adds high-freq jitter that kills the temporal-delta smoothness), section
7,195 B. Rate contribution `25·7195/37,545,489` = **0.004791**. Marginal cost of the dxi over matched
no-dxi = +3,433 section B ≈ +0.0023 rate for a 20× d_pose improvement.

Full archive.zip = **89,772 B**, rate_term 0.05977 (INR base+code dominate; the ξ section is ~8% of the
archive). Full advisory S (n24): `100·d_seg + √(10·d_pose) + 25·B/37.5M` = `100·0.003999 + 0.106 + 0.0598`
= **0.566** — seg (0.40) dominates (the known d_seg blocker), pose now a bounded shippable 0.106, rate 0.060.

## 4. The frame0-decode caveat (characterized, NOT a #238 defect)

The pose-carrier CONFIRM reports `frame0 decode bit-exact = False` — the shipped inflate frame0 warp vs the
numpy oracle frame0 differ by max_abs **14 (calib) / 16 (no-dxi) / 18 (ship-dxi)** per uint8. It is present
in ALL THREE configs and grows only mildly with twist magnitude ⇒ a **pre-existing store_nothing
warp-reproduction detail (inflate-vs-oracle numerics on the larger warp displacements), NOT introduced by
the dxi and NOT a #238 issue**. Crucially, the realized d_pose is measured on the ACTUAL shipped inflate
`.raw` (what `evaluate.py` would score), which is deterministic (1-thread BLAS tier) — so 0.001127 is the
true shipped number; the oracle is only a cross-check reference. This warp-reproduction gap is a separate
existing item worth a follow-up bit-exact tightening (inflate warp fp path == oracle), but it does not gate
the SHIPPABLE verdict.

## 5. VERDICT — SHIPPABLE

**SHIPPABLE.** ξ_eff = xi_stored + dxi, serialized (~7.2 KB COUNTED, 0.0048 rate) and re-measured through
the REAL byte-closed inflate at n24, reproduces R1's trained d_pose (0.001127 realized vs 0.001012
training-side) → the joint-descent pose contribution 0.106 is genuinely shippable, and the end-to-end
joint-descent path (render co-adapts, cheap warp becomes pose-legible, trained dxi ships) is validated.
The custody memo's downgrade ("SOLID training-side advisory; shippability PENDING #238") is now RESOLVED:
shippability CONFIRMED (advisory axis; the exact upstream/evaluate.py CPU row remains the promotion
authority).

verdict_scope: this is a SHIPPABLE verdict for the pose HALF on THIS R1 checkpoint via the store-nothing
ξ_eff carrier; it does NOT touch the d_seg blocker (0.40 seg term) or claim a frontier move.

**n600 authority-scale row:** in progress at wrap (`reports/r1_dxi_238/n600_shipdxi.log`); the archive/rate
columns are already n600-exact (section built for all 600 pairs); only the realized d_pose mean broadens
from 24→600 pairs. [FILL: n600 realized d_pose = ____].

## 6. Dedicated-finishing-descent implication (NOTE ONLY — operator-GO, NOT launched)

R1 was still descending at ep1108 (d_pose 0.00108→0.001012 over ep1074→1108, not plateaued). A dedicated
from-converged pose-finishing descent (more Muon epochs at w_pose>0, render co-adapting) could push d_pose
toward the ancestor 0.018-class (pose contribution √(10·0.018)... i.e. toward ~0.02 — a further ~5× on the
0.106). This is now a validated, shippable door. **NOT launched** (heavy/paid = operator-GO; CONTAINMENT).

---

**Provenance:** connector `tools/levelset_byte_close_and_eval.py` (`--pose-carrier-xi-from-ckpt` +
`--pose-carrier-dxi-scale`); reports `reports/r1_dxi_238/{n24_shipdxi,n24_calib,n24_nodxi_matched,
n600_shipdxi}.json`; R1 checkpoint `experiments/results/levelset_n600_R1_storenothing_descent_ev1_
20260703T004906Z/levelset_witness_ema_mlx.npz`. Authority = frozen CPU-torch PoseNet on the inflated
`.raw`; `[macOS-CPU advisory] NON-PROMOTABLE`. Sisters: `r1_0011_custody_revalidation_20260708.md` (the
custody chain this resolves) + `pose_solve_output_space_inverse_20260708.md` §5b (the connector readiness
note) + canonical equation `morse_smale_stratified_parallax_dpose_v1` in `tac.canonical_equations`
(`morse_smale_stratified_parallax_dpose_20260708.py`; the new anchor `r1_dxi_shippability_byteclose_20260708`
appended to `empirical_anchors`, `register_canonical_equation` path).
**Pointer 0.19110 UNMOVED.**
