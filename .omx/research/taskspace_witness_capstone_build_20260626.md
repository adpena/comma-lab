# NON-RGB TASK-SPACE WITNESS — capstone build: architecture + $0 feasibility + byte-budget + scaffold (2026-06-26)

**Source:** operator directive 2026-06-26 "Push the task-space ... in parallel pursue the most promising and
all" — the standing #171 capstone. $0 CPU design+scaffold+feasibility slot (NO GPU arm; the hosc probe owns
the GPU). Authority `[macOS-CPU advisory]` / mathematical-derivation — the EXACT frozen-SegNet GT argmax
(`gt_segnet_argmax.u8`, n600), NOT the 600-sample render harness. **Frontier pointer UNMOVED contest-CPU
0.19110** (means ≠ ends; no exact row this slot). Builds on `capstone_taskspace_step_witness_reorientation_
and_binding_20260626` (bind-all launch spec), `yousfi_road_lane_geometric_solve_probe_20260617` (quasi-
stationary verdict), `custom_witness_format_inflate_interpreter_design_20260623` (the format half).

## ARCHITECTURE (coords + rasterizer + byte-close + training + composition)

The contest inflates `archive.zip` → RGB → runs ITS OWN frozen SegNet(frame1)+PoseNet(pair). So the witness
MUST inflate to legal **real RGB** (luma+chroma) — the class-logit/flat-palette legal frame is POSE-BLIND
(measured S=11.65, d_pose 2.67–12.66; reorientation §1). "Non-RGB task-space" = the **carrier/capacity** is
task-space; inflate.py expands it to legal RGB realized through R (bicubic↑→uint8→SegNet/PoseNet). The
vehicle is the existing realized-through-R coord-INR witness with the per-pair carrier re-cast:

- **(a) task statistic (per frame).** Replace the witness's free 32-dim FiLM code with a task-space
  parametrization: ego-motion (rides the stored pose sidecar, 0 extra) + a small boundary-geometry code.
  **MEASURED nuance:** the unconstrained CE-proxy codes do NOT compress (PCA intrinsic dim 27/32, Δ/mag
  0.68) — so ~8-dim is an *imposed* parametrization (lane-poly + ego coords / low-rank+temporal-AR code),
  NOT an emergent property. The rate win is engineered, not hoped.
- **(b) inflate.py deterministic program (FREE).** (1) amortized static base from the shared INR weights
  (no per-pixel storage). (2) **self-orientation directional basis** — the decoder runs a cheap forward →
  its own argmax → all-class boundary tangent → directional/curvelet Fourier features (HF across edge, LF
  along) → directional forward. This regenerates the −48% d_seg lever's tangent field FREE at decode, **no
  GT** (the build's one genuinely-new unlock). (3) render-res RGB → R → uint8.
- **(c) byte-close.** base INR weights + per-frame code int8+brotli (counted); deterministic Fourier table,
  the rasterizer/forward code, and the self-orientation tangent are FREE in inflate.py.
- **(d) training (GPU arm, scorer-only).** `train_witness_realized_through_R_mlx.py` realized through R +
  frozen CPU-torch SegNet verdict (NO-FAKE authority); directional path imports the new byte-closeable
  helper instead of GT tangent.
- **(e) composition (bind-all).** stored-pose sidecar (pose solved ~0.017) + frozen-source static priors
  (#138/#158) + chroma (d_seg lever + pose feed) + sparse margin-conditional residual coder (lever-D) for
  the active-band tail + PR98 0-byte decode postprocess.

## FEASIBILITY ($0 CPU, n600 GT argmax — `experiments/probe_taskspace_witness_feasibility.py`)

| measurement | result | verdict |
|---|---|---|
| **(B) directional-tangent byte-closeability** | mean \|cos\| GT-fine vs cheap tangent: **0.893** own-argmax / **0.909** coarse-majority / **0.908** temporal-mode | **CONFIRMED ≥0.85 — the −48% lever byte-closes (the decisive unlock)** |
| (A) static-base d_seg (temporal mode every frame) | **0.0265** (24× goal, 37× capstone); 83.4% px stable ≥99% (free amortized base); 27.6% active band | static base alone INSUFFICIENT; amortizes the rate, not the d_seg |
| (A) prev-frame predictor d_seg | 0.0125 | real temporal coherence → AR/delta codes viable |
| (C) whole-partition homography warp | d_seg **0.64** vs identity 0.012–0.040 | **FALSIFIED** — confirms quasi-stationary (yousfi); rate model = static base + small per-frame code + temporal deltas, NOT warp-of-template |

**Does it reach goal/capstone?** Partition-domain floor: a perfect static base is 0.0265 — so goal (1.12e-3)
and capstone (7.2e-4) are reachable ONLY by the trained per-frame witness, NOT by any parametric/static
shortcut. The lever_b proxy already reached d_seg 0.00826; directional (now byte-closeable) → ~0.0044
(proxy). The remaining 0.0044 → 7.2e-4 is the GPU arm's burden (step-native basis + KKT capacity-routing +
MD-decoupling convergence + chroma + realized-R). **The directional lever is NECESSARY-not-sufficient and is
now MEASURED byte-closeable** — the binding term stays d_seg; the GPU arm is the only path to close it.

## BYTE-BUDGET + S ESTIMATE (grounded in measured anchors)

- base INR int8+brotli ~30–60 KB (base_ch20 ~89.6 KB incl. codes; reorientation §5.5) · per-frame code
  int8+brotli ~2–8 KB · stored-pose sidecar ~5–6.65 KB (measured) · **directional tangent 0 B (unlock)** ·
  sparse residual ~few KB.
- **Conservative ~55 KB → rate 25·55000/37.5M = 0.037.** Optimistic (seg=pose fusion, warp base by stored
  pose) ~40 KB → 0.027.

`S = 100·d_seg + √(10·d_pose) + 25·bytes/N`, pose √(10·d_pose) ≈ 0.018 (solved):
- **goal d_seg 1.12e-3 + 55 KB:** S ≈ 0.112 + 0.018 + 0.037 = **0.167** → banks T_1 (sub-0.19), real gain over 0.19110.
- **capstone d_seg 7.2e-4 + 40 KB (fusion):** S ≈ 0.072 + 0.018 + 0.027 = **0.117** → **sub-0.15.** ✓

Sub-0.15 requires BOTH the capstone d_seg AND the rate fusion; goal-d_seg-only already banks T_1.

## CONTEST-LEGALITY (byte-closeability proof, NO GT leak)

- **GENERIC ALGORITHM = FREE in inflate.py:** the coord transform, deterministic Fourier table (seed),
  the forward pass, and the **self-orientation tangent** (computed from the witness's OWN decoder-reproducible
  argmax — `self_orientation_directional_feats`). NO GT SegNet argmax at decode; the helper's input is a
  partition the decoder itself produces (measured cos 0.89–0.91 vs GT-fine tangent). This is the binding
  NO-FAKE point and it is MEASURED, not asserted.
- **VIDEO-DERIVED LEARNED = COUNTED in archive.zip:** base INR weights + per-frame codes + pose sidecar
  (int8+brotli, the witness blob the existing `tools/witness_byte_close_and_eval.py` already byte-closes
  with sha256 lossless parity). No video-derived table is smuggled into inflate.py.
- The old directional basis was correctly flagged NOT byte-closeable (uses `gt.lstars`); the self-orientation
  helper replaces the GT input with a decoder-reproducible one — the ONLY change needed to legalize the lever.

## SCAFFOLD (landed this slot + GPU-run plan)

**Landed ($0):**
- `src/tac/boundary_math/lever_b_generator.py::self_orientation_directional_feats` — the byte-closeable
  directional-tangent primitive (numpy-portable, identical at train & inflate), + 3 NO-FAKE tests
  (`test_lever_b_generator.py`: byte-closeability identity, grid-mismatch raise, distinct-partition-distinct-
  tangent). The reusable building block both the trainer's directional path and inflate.py import.
- `experiments/probe_taskspace_witness_feasibility.py` + `taskspace_witness_feasibility_20260626.json` — the
  decisive feasibility probe (reusable; re-runnable any time).

**GPU-run plan (after the iso banker finishes GATE1; do NOT pivot the live slot):**
1. Wire `self_orientation_directional_feats` into `train_witness_realized_through_R_mlx.py`'s directional
   path: replace the GT-tangent (`gt.lstars`) feed with the witness's own iso-pass argmax (self-orientation
   fixed point) — flip `directional_byte_closeable=True`. Validate fixed-point stability $0 numpy first
   (iso-argmax → tangent → directional-argmax → tangent agrees within tol over 2–3 iters).
2. Config: `--activation hosc --chroma --basis directional` (now byte-closeable) on a base_ch20-class
   witness, realized through R + frozen CPU-torch SegNet verdict, MD-decoupling optimizer.
3. Compose: stored-pose sidecar + frozen-source static priors + sparse lever-D residual + PR98 postprocess.
4. Byte-close via `tools/witness_byte_close_and_eval.py` → contest-CPU exact eval (the only authority row).

**What this slot did NOT do (NO-FAKE):** no GPU training, no exact row, no trainer edit (the live iso run
holds the slot; a half-built trainer edit would conflict + be a fake capstone). The pointer is UNMOVED; the
deliverable is the byte-closeability UNLOCK (measured) + the bind-all build spec + the GPU-ready primitive.

Cross: capstone_taskspace_step_witness_reorientation_and_binding_20260626 (bind-all) ·
yousfi_road_lane_geometric_solve_probe_20260617 (quasi-stationary) · witness_capstone_deepmath_levers_
20260625 (proxy directional −48%) · sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.
