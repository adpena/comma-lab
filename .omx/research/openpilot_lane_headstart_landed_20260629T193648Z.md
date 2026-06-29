---
title: openpilot lane HEAD-START landed — Wyner-Ziv lane-centerline base + conditional-residual pipeline (the v2 witness starts conditioned, not from scratch)
authority: "[macOS research-signal] / advisory — $0, no GPU, no paid dispatch, no training. score_claim=false; promotable=false. Pointer UNMOVED contest-CPU 0.19110. MEANS toward a byte-closed exact row, never an end."
date: 2026-06-29
subagent: openpilot-headstart-build
git_hash_at_landing: 5cdd21c2b
builds_on:
  - .omx/research/openpilot_world_model_free_prior_v2_20260629T190505Z.md   (a99f41f0: the 0.00214 oracle floor + ~64% recovery; Wyner-Ziv framing)
  - tools/measure_lane_polynomial_shape_floor.py                            (a99f41f0: the oracle-floor probe I reproduced)
  - .omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md    (FEED-jh)
lands:
  - src/tac/boundary_math/lane_headstart.py                  (the head-start module — reusable library)
  - src/tac/boundary_math/tests/test_lane_headstart.py       (21 behaviour tests, all pass)
  - tools/build_lane_headstart.py                            (thin CLI driver)
  - src/tac/boundary_math/__init__.py                        (wired exports)
  - experiments/results/lane_headstart_20260629T193400Z/lane_headstart_summary.json (durable n600 summary)
---

# openpilot lane HEAD-START landed — the v2 witness starts from the conditioned base

**Operator directive:** *"we need the openpilot head start too"* — make the ~64% Wyner-Ziv
head-start from a99f41f0 a USABLE part of the v2 path, so the eventual through-R lane GPU run
STARTS from the conditioned base, not from scratch.

**What this is (deep-math lens, grounded).** Textbook **Wyner-Ziv (1976) source coding with side
information at the decoder.** inflate.py's decoder has FREE side information `Y` = the lane
CENTERLINE (a GENERIC rasterizer over stored polynomial coeffs), correlated with the source
`X` = the frozen-SegNet argmax-lane class. Wyner-Ziv says we code only the CONDITIONAL residual
`X − E[X|Y]`. The centerline `Y` recovers ~64% of the lane d_seg for ~free; the witness then
learns ONLY the thin ragged ±1px boundary residual. The head-start = the ResidualGauge cell
**`CONDITIONAL_ON_LANE_PRIOR`**.

Every number is `[macOS research-signal]` advisory; only `upstream/evaluate.py` on byte-closed
bytes is authority. **Pointer UNMOVED 0.19110. This is infra MEANS, not a score.**

---

## NO-FAKE check — I reproduced a99f41f0's number MYSELF (did not quote it)

Ran `tools/measure_lane_polynomial_shape_floor.py` on the cached frozen-SegNet argmax
(`experiments/results/mlx_fleet_gt_cache/gt_n96.npz`, key `lstars`, class-1 = Lane):

| variant | drop-all baseline | base d_seg (my run) | a99f41f0 | IoU | centerline resid | recovered |
|---|---|---|---|---|---|---|
| deg-4 **per-dash** (the head-start), n96 | 0.005885 | **0.002144** | 0.00214 | 0.6657 | 0.501 px | **63.6%** |
| deg-4 bridged-continuous, n96 | 0.005885 | 0.003414 | 0.003414 | 0.5064 | 0.810 px | 42.0% |
| degree sweep 1→4 per-dash | — | 0.002192→0.002144 (saturating) | matches | — | — | — |

**EXACT reproduction** (0.002144 = a99f41f0's 0.00214; bridged 0.003414 matches to 6 digits;
degree saturates as the memo says). My OWN library module on the **full 600 consecutive frames**
(`gt_n600.npz`): drop-all **0.005855** → conditioned base **0.002069** → **recovered 64.7%**,
residual round-trip **bit-exact**. The head-start is real and reproduced.

---

## PART 1 (core deliverable) — the head-start module `src/tac/boundary_math/lane_headstart.py`

A reusable library (numpy + scipy, CPU, $0, deterministic) with three pieces:

### 1. `fit_centerlines` — the COMPRESS-TIME analyzer (the base `Y`)
Per lane component, fit a low-order polynomial (`col=poly(row)` or `row=poly(col)` by extent) to
the frozen-SegNet lane pixels; choose the oracle per-component half-width (compress-time, stored).
`dash_bridge_rows=1` = per-dash (the 0.00214 head-start); `>1` bridges dashes (0.00341). The
coeffs are VIDEO-DERIVED → **COUNTED** (rule-118).

### 2. `rasterize_centerlines` — the DETERMINISTIC base rasterizer (FREE generic algorithm)
Reads only the stored coeffs → evaluates the poly → paints the band. This is the rule-118 **FREE**
algorithm that lives in inflate.py; it reproduces the conditioning base `Y` bit-exactly on any
host. `build_lane_headstart` CONFIRMS this base alone reproduces the ~0.00207 floor (the NO-FAKE
self-check, asserted by `test_build_lane_headstart_*` + the n600 run above).

### 3. `compute_lane_residual` / `apply_lane_residual` — the conditional-residual pipeline
The Wyner-Ziv residual `X − Y` as signed int8 ∈ {−1,0,+1} (`+1`=witness must ADD a missed lane
pixel/FN; `−1`=must REMOVE a false-positive/FP; `0`=base correct) **+ its EXACT inverse**
`base ⊕ residual → X` (round-trips bit-exactly — tested on 20 random masks + the real corpus).
**This DEFINES the GPU run's job:** the witness learns only the small residual (support fraction
= base d_seg = **0.00207**, target ≤1.23e-3 sub-0.15 / ≤1.63e-3 sub-0.19) from the sub-pixel
base, instead of learning `X` from scratch (0.00586). A ~64.7% head-start.

Also landed: `serialize_centerlines_delta` + `deserialize_and_rasterize` (an exact-preserving
byte stream = a candidate inflate.py decode path; round-trip tested), `estimate_base_bytes`,
`gauge_cost_cell`. **21 tests, all pass** (behaviour not constants: residual signs, exact inverse,
serialization round-trip, recovery > 0.5, gauge `learned_residual_cost == PENDING-GPU`).

### Centerline byte estimate (COUNTED) — honest, and a correction to the memo's 0.5–5 KB
Measured on `gt_n600`:

| estimate | bytes/600 | rate term (25·B/37.5M) | notes |
|---|---|---|---|
| parametric (openpilot-native, coeffs@16b) | ~81,500 | 0.054 | fixed-bit upper bound |
| delta-entropy (exact-preserving, H(Δ)=~1.25 b) | ~65,500 | 0.044 | measured iid entropy floor |
| **zlib temporal (measured)** | **~65,200** | **0.043** | image-space iid (recommended cell) |
| ground-frame + pose-reuse target | 0.5–5 KB | 0.0003–0.003 | a99f41f0 estimate, **MED, NOT measured here** |

**Honest correction (deep-math + measured):** the centerline is **NOT cheap in image space.**
Adjacent-frame lane IoU(0,1) = **0.284** — ego-motion MOVES the image-space lane every frame, so
there is little frame-to-frame redundancy and zlib only reaches ~65 KB/600 (rate ~0.043, NOT
tiny). The memo's 0.5–5 KB requires the **ground/bird's-eye frame** (where the lane lines ARE
near-static: store a few parallel polynomials ONCE + reuse the already-stored pose to re-project
per-frame for free) — that is the next byte-optimization (needs the EON homography/IPM, scoped in
a99f41f0 Task-3 + `tac.boundary_math.lane_sdf_component`), **and it is NOT yet measured.** The
dominant COUNTED cost remains the LEARNED ragged-boundary residual (the ~0.00207 the centerline
leaves) — **PENDING-GPU, not fabricated.**

---

## PART 2 — openpilot can SOURCE the centerline for FREE at compress-time (CONFIRMED, $0 CPU)

Goal: confirm the rule-118 **path (a)** (openpilot as a COMPRESS-TIME ANALYZER) is mechanically
available. Result: **CONFIRMED end-to-end at $0 on CPU.**

1. `uv pip install onnxruntime` → **onnxruntime 1.27.0** (CPU + CoreML providers). ✓
2. `supercombo.onnx` download: the documented raw URL returns a **git-LFS pointer** (133 B). The
   real binary comes from the LFS media URL
   `https://media.githubusercontent.com/media/commaai/openpilot/v0.9.7/selfdrive/modeld/models/supercombo.onnx`
   → **49.1 MB** (51,452,435 B), **sha256 `b31b504bc0b440d3bc72967507a00eb4f112285626fbfb3135011500325ee6d6`**
   (EXACTLY matches the LFS pointer oid → integrity verified). Cached at
   `.omx/tmp/openpilot_models/supercombo.onnx` (re-downloadable; URL+sha256 recorded → deterministic
   repro; never cited as durable evidence). NOTE: actual size **51.5 MB**, not the memo's ~30 MB.
3. Load: default graph-optimization **FAILS** (onnxruntime 1.27 `SimplifiedLayerNormFusion`
   name-lookup bug). **Workaround:** `SessionOptions.graph_optimization_level = ORT_DISABLE_ALL`
   → loads cleanly. I/O contract (v0.9.7, CONFIRMED from the live model):
   inputs `input_imgs`/`big_input_imgs` (1,12,128,256) fp16 + `desire`(1,100,8) +
   `traffic_convention`(1,2) + `lateral_control_params`(1,2) + `prev_desired_curv`(1,100,1) +
   `features_buffer`(1,99,512); output `outputs` **(1,6504) fp16** (matches the memo). (Note: the
   aux inputs differ from `openpilot_seeding`'s documented `nav_features`/`nav_instructions` —
   v0.9.7 uses `lateral_control_params`/`prev_desired_curv`; build placeholders from the live
   `get_inputs()`, not the stale doc.)
4. **Forward pass on REAL contest frames** (cached `gt_f0`/`gt_f1`, 874×1164 RGB) → **(1,6504)
   output, all finite, ~15 ms/frame on CPU** (~9 s for 600 frames — trivially within budget, and
   it runs at COMPRESS time, never inflate). ✓

**So openpilot CAN source signal for free at compress-time (rule-118 path (a) is available).**

### Part-2 blocker for an actual openpilot-vs-SegNet lane-AGREEMENT number (best-effort, NOT closed)
I did **NOT** produce a lane-agreement comparison, and deliberately make **no lane-agreement
claim**, because three real steps remain and guessing any of them would be a NO-FAKE violation:
1. **EON medmodel warp:** the (1,12,128,256) input must be the EON model-frame homography of the
   874×1164 camera (I used a naive resize → geometrically-invalid lane outputs; valid only to
   prove the pass RUNS).
2. **lane-line output offsets:** the (1,6504) vector's lane-line head indices for v0.9.7 are NOT
   wired in `openpilot_seeding` (only the pose head `[5755:5761]` is). Extracting lanes needs the
   verified v0.9.7 offsets, not invented ones.
3. **back-projection:** the 3-D lane points → 512×384 SegNet grid via the EON homography, then XOR
   vs `lstars==1`.

**CRITICAL — this blocker does NOT affect the d_seg verdict.** Per a99f41f0 (and confirmed by my
Part-1 measurement): even a PERFECT openpilot-lane agreement does **not** move the **0.00207**
floor — that floor is a property of the SegNet lane class, not the coeff source. Q-source is a
**BYTE/EFFORT optimization** (source coeffs for free vs re-fit them from SegNet), **not a d_seg
lever.** **FALLBACK = the Part-1 SegNet-fit centerline (same floor, fully landed).** The head-start
stands without openpilot; openpilot would only save the re-fit effort / supply the ground-frame
polys.

---

## THE GAUGE COST CELL (for the ResidualGauge layer being built in parallel)

```json
{
  "gauge": "CONDITIONAL_ON_LANE_PRIOR",
  "lens": "Wyner-Ziv source coding with side info at decoder (1976)",
  "from_scratch_lane_dseg": 0.005855,        // witness with NO lane signal (drop-all)
  "base_lane_dseg": 0.002069,                // conditioned start (MEASURED, n600)
  "recovered_frac": 0.6466,                  // ~64.7% (MEASURED)
  "residual_dseg_to_fix": 0.002069,          // == base d_seg = the witness's reduced job
  "residual_dseg_target_sub015": 0.00123,
  "residual_dseg_target_sub019": 0.00163,
  "base_bytes_600_achievable": 65183,        // image-space iid (MEASURED zlib); rate ~0.043
  "base_bytes_600_temporal_target": "0.5-5 KB (ground-frame+pose-reuse; a99f41f0 estimate, MED, NOT measured)",
  "learned_residual_cost": "PENDING-GPU",    // NOT fabricated
  "roundtrip_exact": true
}
```

Programmatic: `from tac.boundary_math import build_lane_headstart, gauge_cost_cell`.

---

## rule-118 tags (binding)
- centerline RASTERIZER (`rasterize_centerlines` / `deserialize_and_rasterize`) = GENERIC algorithm → **FREE** in inflate.py.
- stored centerline COEFFS / delta-stream = VIDEO-DERIVED → **COUNTED** in archive.zip (image-space ~65 KB measured; ground-frame target 0.5–5 KB unmeasured).
- supercombo.onnx (51.5 MB neural weights) = COMPRESS-TIME ANALYZER ONLY; **NEVER shipped in inflate.py** (path (a) confirmed; ambiguous path (b) AVOIDED).
- LEARNED ragged-boundary residual = the real COUNTED budget = **PENDING-GPU** (never fabricated).

## How the GPU run consumes this (the head-start, operationalized)
1. compress-time: `fit_centerlines(lstars[i])` → centerlines (or source from openpilot, same floor).
2. condition the through-R witness on `rasterize_centerlines(...)` (the base masks) — the witness's
   target becomes `compute_lane_residual(base, lane)` (start 0.00207).
3. train the residual through R toward ≤1.23e-3; decode via `apply_lane_residual(base, learned_residual)`.
4. byte-close: COUNTED = centerline coeffs (drive to ground-frame 0.5–5 KB) + learned residual;
   then exact eval (`upstream/evaluate.py`, contest-CPU/CUDA, NEVER MPS). The pointer moves there, not here.

## 6-hook wire-in
1. sensitivity-map: ACTIVE — quantified base recovers 64.7%; residual 0.00207 is the witness's job.
2. Pareto: ACTIVE — conditioning is strictly Pareto-better (smaller trained residual at equal target).
3. bit-allocator: ACTIVE — centerline (image-space ~65 KB measured; ground-frame 0.5–5 KB target) vs learned residual split.
4. cathedral autopilot: N/A — advisory research module, non-promotable.
5. continual-learning: this memo + a99f41f0 + the DECISIVE-8-dim-manifold finding = the lane-prior anchor chain; module is reusable.
6. probe-disambiguator: `tools/build_lane_headstart.py` reproduces + measures the head-start on demand.

## Honest bounds (NO-FAKE)
- No score; pointer UNMOVED 0.19110. All numbers `[macOS research-signal]` advisory.
- base d_seg is SHAPE-ONLY (no R) + ORACLE (fit to target) → a LOWER bound on through-R; the
  through-R witness must still beat the R-survival wall on thin lane strokes (separate, PENDING-GPU).
- the image-space centerline byte cost (~65 KB) is MEASURED; the 0.5–5 KB ground-frame target is
  a99f41f0's estimate, NOT measured here (the honest correction).
- Part-2 lane-agreement is NOT measured (3 named blockers); it is a byte/effort optimization, not a
  d_seg lever, so the head-start verdict stands on the Part-1 SegNet-fit fallback.
