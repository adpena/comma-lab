# PROXY-SUSPECT AUDIT (operator GO "Audit all suspect" · P9 "proxies are poison — use the thing itself") — 2026-07-09

**Agent:** RECOVERY agent (predecessor died at session limit, nothing landed). **Surface:** `crucible3_v8`.
`$0 · no chain launch` (dual-chain wall) · run dirs READ-ONLY · #205 STOPPED · MPS/GPU untouched. Pointer
contest-CPU **0.19110 UNMOVED** — everything here is `[macOS-CPU advisory · research-signal · NON-PROMOTABLE]`
MEANS; only a byte-closed `upstream/evaluate.py` n600 exact row < 0.19110 moves it. Remaining gap to sub-0.15
= **0.0411 S**; every magnitude quoted ÷0.0411 (P2/relative-significance).

**STORES CONSULTED:** `docs/operating_manual_craft_handoff.md` (§4 re-derive-from-primary · §5 label ·
§8.4/§8.5 no-plausible-summary/no-borrowed-number) · `design_philosophies_eightfold_20260709.md` (P9 use-the-
thing-itself, P2 noise-floor) · `P4_recess`/`P5_second_redteam`/`SYNTHESIS_v3_v8`/`residual_kit_deshare_curverel_build`
(+ `.json`). **PRIMARY CODE/DATA RE-DERIVED (not memo-trusted):** `upstream/evaluate.py`, `upstream/modules.py`,
`upstream/frame_utils.py` (grid authority) · `src/tac/inc1a_harness/decoupling_screen.py` · `.../mask_dseg_meter.py`
· `src/tac/boundary_math/movable_deshare.py` · `src/tac/canonical_equations/v8_geometric_rate_decomposition_20260709.py`
· `experiments/results/flip_bc_n600_gate_20260709/{flip_bc_n600_result.json,n600.log}` · `gt_n600.npz` (lstars
600×384×512, margins 600×384×512). De-share footprint sweep **MEASURED** this session (fast reuse of fixed bpp).

---

## ANSWER-FIRST

**7 of 8 items DISPOSED at $0 (VERIFIED / MEASURED); Item-3's bounded dry-start is PINNED-owed (recovery-budget
+ the 42 s/ep is a v7.5.2 config anchor, NOT the v8-inc1a config — re-validating it needs the v7.5.2 run, not a
recovery-tail launch).** Headline corrections: (a) **de-share is NOT footprint-robust** — MEASURED band across
dilate∈{0,1,2,3} = **[0.000, 0.0069]**, so the landed 0.0044 is an INSTANCE-of-dilate=2 proxy, NOT within a robust
[0.0044,0.0104] (dil0/dil1 fall below); (b) the **flip-b_c R4 gate DID land** (contra P4/P5 "owed/absent") —
verdict **no_offset 0.00314 WINS** (flip_weighted +0.0165, flip_median +0.0184, both ~6× worse); (c) all flip
surfaces are grid-correct (scorer 384×512) BY UPSTREAM SOURCE; (d) δ_mask swap CONFIRMED landed + no other floor
surface; (e) Item-5 0.00277 pin PROPAGATED to the equation module (this session, comment-only, append-preserving).

---

## PER-ITEM DISPOSITIONS

### Item 1 — δ_mask swap · no other floor surface · 3.46e-6-vs-3.5e-6 · seed-variance PIN — **VERIFIED**
- **Swap landed (commit `6b1568b0b`):** `decoupling_screen.py` default = `DELTA_MASK_FRAME_SAMPLING_FLOOR = 3.46e-6`
  (R7 MEASURED); `DELTA_R_PROXY_RETIRED = 0.019590163230895963` kept ONLY as historical, NEVER a default;
  `operative_delta_mask()` = max(3.46e-6, seed_spread) with the P2 REFUSE guard (raises when `n_seed_replicates>=2`
  and `seed_spread is None`); `evaluate_kill` returns `VERDICT_REFUSED` on the under-specified floor. Tests 27/27,
  ruff-F clean (verified in commit). RE-READ the file end-to-end — matches.
- **No other floor surface:** the broad grep (517 hits) is ALL unrelated (`delta_mask_bytes` = archive byte-size
  deltas in PR90/PR85 analysis; `0.0196` = numeric coincidences in pose contributions / scipy fixtures). The clean
  grep (excluding tests) for `DELTA_R|delta_mask|0.0196` as a KILL FLOOR returns ONLY `decoupling_screen.py`.
  **No surface imports `DELTA_R_PROXY_RETIRED` or hardcodes 0.0196 as a d_seg kill floor.** RESOLVED.
- **3.46e-6 vs 3.5e-6 — NO discrepancy:** the code uses the PRECISE measured SEM **3.46e-6**; P4-R7's "δ_mask =
  3.5e-6 S" is that value ROUNDED to 2 sig figs (its own text says "frame-sampling SEM = 3.46e-6, bootstrap σ
  3.46e-6, agree"). Code correctly carries the precise value. `DELTA_R_PROXY_RETIRED` rounds to 0.0196. CLEAN.
- **Seed-variance PIN {owed}:** instrument = **≥3 seed replicates/arm** (`seed_replicates_per_arm: 3`, SYNTHESIS_v3
  §B) feeding `operative_delta_mask(seed_spread=<measured in-run spread>, n_seed_replicates=3)`; trigger = the **1a
  governed A/B EVENT** (each arm run at ≥3 seeds); owner = the crucible-3 increment-1a launch. The operative floor is
  UNMEASURED until then; the REFUSE guard structurally prevents a kill firing on within-seed noise.

### Item 2 — de-share dilate footprint sweep ($0 MAIN EVENT) — **MEASURED · CORRECTS the P8 row**
Re-ran `measure_deshare_magnitude` attribution on gt_n600 (all 600), fixed bpp per edge (horizon 0.31109, lane
0.38372 — bpp is dilate-independent; only attributed-px change), sweeping the Movable footprint `dilate ∈ {0,1,2,3}`:

| dilate | horizon att px | lane att px | horizon S | lane S | **total S deflation** | ÷0.0411 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.000000 | 0.000000 | **0.000000** | 0.0% |
| 1 | 6,102 | 2,665 | 0.001264 | 0.000681 | **0.001945** | 4.7% |
| **2** | **12,592** | **7,009** | **0.002608** | **0.001790** | **0.004399** | **10.7%** |
| 3 | 18,631 | 11,895 | 0.003859 | 0.003039 | **0.006898** | 16.8% |

- **dilate=2 REPRODUCES the landed 0.004399 EXACTLY** (re-derive-from-primary ✓; the residual_kit.json value is real).
- **NOT footprint-robust.** The magnitude band across the footprint proxy is **[0.000 (dil0), 0.0069 (dil3)]** — it
  does NOT stay within the memo's [0.0044, 0.0104] (that band was the amortized-vs-coded-alone axis AT dilate=2, a
  DIFFERENT uncertainty). dil0/dil1 fall BELOW 0.0044. **The landed 0.0044 is an INSTANCE-of-dilate=2 proxy**, and
  the real Movable bbox-carrier footprint **is NOT exposed** (no `*bbox*` module; `movable_footprint` = a dilated
  argmax mask is the ONLY proxy). This SHARPENS F-P5-P9-2: the de-share magnitude is load-bearing on the (unmeasured)
  footprint; the thing-itself (bbox realized coverage) is genuinely owed to pin it. **P8-row correction:** carry
  de-share as `[dilate-proxy band 0.000–0.0069; landed 0.0044 @ dil2; thing-itself = bbox realized coverage, owed]`,
  NOT a robust 0.0044±(amortization).
- **Propagation to curve-relative ratios — REFUTED verdict HOLDS across the band.** From the residual_kit.json
  raw-vs-deshared arms: lane curve-relative ratio = **0.90×** raw AND 0.896× deshared (curve-relative clearly WORSE,
  robust); horizon = 1.006× raw / 0.988× deshared (a WASH ~1.0× either way — neither meaningfully beats). Since
  de-share removes only 9.6% (horizon) / 0.85% (lane) of residual px at dil2 (and 0% at dil0), the curve-relative
  REFUTATION is footprint-robust: lane worse, horizon wash, at every dilate. R2 verdict_scope FORMULATION unchanged.

### Item 3 — 42 s/ep anchor provenance + config-diff + bounded dry-start — **PROVENANCE VERIFIED · dry-start PINNED-owed**
- **Provenance (VERIFIED):** the 42 s/ep is a **v7.5.2 / crucible-2** anchor — `sub015_DAG:12215` ("n600 first-epoch
  compile > 9.5 min; steady-state **~42 s/ep** anchor") + `t5_crucible2/SYNTHESIS_{v2,v3}_v752_20260709.md:90,536,614`
  ("MLX-local M5 Max, **measured 42 s/ep**", used to build the v7.5.2 §B.wall_clock budget table: floor-epochs ×
  42s/ep → **6–16 h** floor budget, 3 main stages × clamp[150,400]). Config = the **v7.5.2 sealed single-trunk
  through-R witness** on M5 Max MLX-GPU, steady-state (post-compile). `sub015_DAG:579` corroborates (~42 s/ep @ Muon
  stage-8). It is NOT a crucible-3/v8 number.
- **Config-diff vs crucible_v7 (the load-bearing nuance):** the 42 s/ep IS the crucible_v7/v7.5.2 config's own
  wall-clock baseline. The crucible-3 **v8 increment-1a is PAINT-FREE mask-level** (no through-R verdict batch → no
  +66 GiB spike, memory-CHEAP, F-P5 §5 blind-spot #4) — a DIFFERENT throughput profile. **A v8-inc1a dry-start would
  NOT re-validate the 42 s/ep** (different config). The dry-start that validates 42 s/ep is the **v7.5.2 sealed
  through-R config** itself.
- **Dry-start disposition — PINNED-owed (NOT run).** Rationale (operating-manual §10 — context-scarce recovery tail;
  §3 blast-radius): a ~40-min governed MLX launch (>13-min compile + ≥5 steady epochs + 25-min safe_run SIGTERM,
  ~71 GiB envelope, #384 pattern) mid-recovery risks the predecessor's exact nothing-landed failure; the DURABLE audit
  + a precise runbook is strictly better than a half-run. **PIN:** {instrument = governed `tools/launch_witness_run.py`
  of the **v7.5.2 sealed config** (NOT v8-inc1a) with a `safe_run` 25-min hard SIGTERM window, log-tail sec/ep from
  ≥5 steady epochs after the >13-min compile; trigger = the next DEDICATED execution session (operator GO covers it —
  it is not a recovery-tail task); owner = the v7.5/crucible_v7 relaunch}. Report measured sec/ep; re-emit the
  `t5_crucible2` §B.wall_clock table (and the 6–16 h floor-budget) if the measured sec/ep is >10% off 42.

### Item 4 — counted-seed floor ($0 upper bound) — **PINNED with precise why**
No dedicated byte-close "seed section" surface exists (grep: no `seed_section|counted_seed|seed_bytes`). The analytic
generators (horizon poly, lane band, Movable bbox, hood) are DETERMINISTIC (rule-118 FREE in inflate.py); their
COUNTED payload is the per-frame coefficient/intercept stream. **$0 UPPER BOUND on the counted-seed floor = the
MEASURED DOMINANT-only rate 0.061 S** (sum of the byte-cost-function dominant terms: horizon 0.00277 + Road/Lane
0.0275 + Movable 0.00344 + hood 0.0202 + Lane/* 0.007 ≈ 0.061; `v8_geometric_rate_decomposition_v1`). **PIN P-C:**
the byte-CLOSED counted-seed floor (the coefficient stream through a REAL `archive.zip` entropy coder) is P-C-owed —
no archive-level seed section is emitted yet; 0.061 is the deterministic byte-cost-function UPPER BOUND, and a real
entropy-coded seed stream can only go LOWER (a floor is P-C-measured, not $0-measurable).

### Item 5 — 0.0032→0.00277 pin — **PROPAGATED (this session, comment-only, append-preserving)**
The equation module `v8_geometric_rate_decomposition_20260709.py` still carried unannotated **0.0032** (L31 docstring,
L159 dict). Per Catalog #110/#113 (append-only) I did NOT mutate the canonical anchor value; I ADDED annotations:
(a) L159 — the P4-R3 code-emitted dominant = **0.00277** (4167 B), 0.0032 = +14% amortization-method delta (1.1% of
the gap, IMMATERIAL, and 0.00277 is CHEAPER → the rate argument is STRONGER); (b) the whole-scene `geometric_complete
_lossless` — the P4/P5b CORRECTED-triple **0.140 − de-share 0.00440 − triple-point 0.00102 = 0.135** (residual enemy
0.074), de-share tagged `dilate=2 footprint proxy; sweep band [0.000@dil0, 0.0069@dil3] NOT footprint-robust`,
shippable = WASH, r* RANGE [0.061,0.135]. ruff-F clean, module imports, `EQUATION_ID` intact. The de-share/curve-
relative anchors already exist appended (`v8_residual_deshare_dedup_measured_20260709` CONFIRMED · `..._NEGATIVE_...`
REFUTED).

### Item 6 — 0.100403 [analytic-generators] labeling sweep — **VERIFIED clean**
Single citation repo-wide: `v8_geometric_rate_decomposition_20260709.py:328` `"analytic_composite_mask_dseg":
0.100403,  # [analytic-generators, no-trained-fields]` — ALREADY labeled. No unlabeled citation in code or the
crucible3 md docs. No fix needed.

### Item 7 — P5's remaining P9 findings — **DISPOSED**
- **P9-1 (δ_R live-code proxy):** FIXED (Item 1) — the measured 3.46e-6 is swapped in; the retired constant is never
  a default; the REFUSE guard is live. The "un-swapped measurement" P9 class is closed.
- **P9-2 (dilate=2 de-share proxy):** SHARPENED by the Item-2 sweep — thing-itself (bbox realized coverage) NOT
  exposed; band [0.000,0.0069] measured; carry the labeled band, owed.
- **P9-3 (0.0032 vs 0.00277 memo-vs-code):** propagated (Item 5).
- **P9-4 (b_c no_offset placeholder → thing-itself gate):** the **realized-through-R #386 gate LANDED**
  (`flip_bc_n600_gate_20260709/`, 2670 s n600 run @ ckpt mod32cap ep650, render_hw [384,512]): **no_offset 0.003144
  WINS** decisively; flip_weighted +0.016530, flip_median +0.018418 (both ~6× WORSE; ÷0.0411 = +40%/+45%). **R4 is
  no longer "owed" — verdict = no_offset (SAFE DEFAULT) CONFIRMED as the WINNER through the thing-itself, not merely a
  placeholder.** The `flip_weighted_bc_build_and_gate_20260709.md` MEMO is still unwritten (results landed, memo owed)
  — a triality DAG/consume gap, NOT a measurement gap. P9-4 DISSOLVED with the measured winner.
- **P9-5 (1a mask d_seg = proxy for through-R):** unchanged — 1b is the named thing-itself; the single-valued-carrier
  special case is FIXED in SYNTHESIS_v3 (lateral-capable 3-curve I1b), so no structurally-blocked proxy remains.

### Item 8 — FLIP-RESOLUTION against upstream evaluate.py + modules.py + all resolutions — **VERIFIED grid-correct**
**Upstream authority (RE-READ):** `frame_utils.py`: `camera_size = (1164, 874)` (W=1164, H=874), `seq_len = 2`,
`segnet_model_input_size = (512, 384)` (W=512, H=384). `evaluate.py`: batch assert `[seq_len, camera_size[1],
camera_size[0], 3]` = camera-res NHWC; CPU=`AVVideoDataset` (PyAV), CUDA=`DaliVideoDataset` (DALI); score =
`100·segnet_dist + √(10·posenet_dist) + 25·rate`. `modules.py`: **SegNet** = `x[:, -1]` (LAST frame) → bilinear
`interpolate(size=(segnet_model_input_size[1], segnet_model_input_size[0]))` = **(H=384, W=512)** → 5-class argmax,
d_seg = per-pixel argmax-disagreement mean. **PoseNet** = both frames → SAME bilinear resize to (384,512) → `rgb_to_yuv6`
→ normalize (mean 127.5, std 63.75) → 12-ch → pose 6-dim MSE. **∴ THE authoritative flip/argmax grid is (H=384, W=512);
argmax is taken AFTER the resize.** The full res chain: render grid → camera 874×1164 (uint8, archive/inflated) →
SegNet bilinear↓ (384,512) → argmax.

`gt_n600.npz`: `lstars (600,384,512) int64`, `margins (600,384,512) fp32` = the post-R SegNet argmax at the AUTHORITY
grid, by construction. Per-surface verdict:

| surface | grid | verdict |
|---|---|---|
| **(a) boundary_math flip stats** (`laguerre_logit_offset` b_c solver) | targets from lstars/margins (384,512); realized re-measured through R | ✓ target = scorer grid; VERDICT = realized-through-R (below) |
| **(b) inc1a `mask_dseg_meter` + `composite_assembler`** | ENFORCED: `pa.shape != g.shape` raise vs lstars (384,512) | ✓ shape-equality guard pins the compare to 384×512 |
| **(c) `movable_deshare` pixel counts** | operates on `lstars[i]` (384,512); `w = lstar.shape[1]` flat-index | ✓ scorer grid (feeds de-share, b_c targets) |
| **(d) R6/R7 decompositions** (P4_recess) | `gt_n600` lstars/margins (384,512) | ✓ scorer grid |
| **(e) flip-b_c gate arms** (`flip_bc_n600_gate`) | `render_hw [384,512]` → R (bicubic↑874 → uint8 → bilinear↓512×384) → frozen CPU SegNet argmax | ✓ **through-R protocol MATCHES evaluate.py** (SegNet last-frame, resize (384,512), argmax); CPU-SegNet authority (advisory until byte-close) |
| **(f) #149 camera-res placement** | sub-pixel curve/lane PLACEMENT authored at camera-res 874×1164 (pre-downsample); COMPARE always 512×384 | ✓ the INTENDED exception (placement-grid ≠ compare-grid; SYNTHESIS_v3 §A.4-grid PINS it) |

- **H/W axis-order check:** upstream `camera_size=(W,H)` and `segnet_model_input_size=(W,H)` are (W,H) TUPLES, but
  `interpolate(size=(...[1], ...[0]))` = (H,W) — correct. Our caches/masks are (N,H,W)=(600,384,512) and index
  `[rows, cols]` = [H,W]; `w = lstar.shape[1]` (=512=W) for flat-index `row*w + col`. **No 512×384 vs 384×512
  transposition found** — every surface uses (H=384, W=512) consistently.
- **NO flip surface at a non-upstream-authoritative grid** (no surface takes argmax at camera-res before downsampling;
  the render-then-argmax ordering is correct — witness renders 384×512, R up/down-samples, SegNet argmax at 384×512).
  #149 is the single intended camera-res surface and it is PLACEMENT-only (not a compare/argmax grid). Grid-audit
  verdict: **PASS — all flip surfaces grid-correct by upstream source.**

---

## EIGHTFOLD STANDING CHECKS (P2 · verdict-scope · relative-significance)
- **P2 noise-floor:** de-share Δ now carries its FOOTPRINT floor (band [0.000,0.0069]) in addition to the amortization
  range; δ_mask 3.46e-6 seed-component owed-in-run; counted-seed floor = P-C-owed (0.061 upper bound).
- **verdict_scope on negatives:** curve-relative REFUTED = FORMULATION (footprint-robust); flip_weighted/flip_median
  KILLED-vs-no_offset = FORMULATION (this b_c offset formulation, not the paradigm); de-share footprint = INSTANCE.
- **÷0.0411 on every magnitude:** de-share band 0.0%→16.8%; corrected complete 0.135; residual enemy 0.074 = 180%;
  b_c deltas +40%/+45%; 0.00277-vs-0.0032 = 1.1%.
- **§4 re-derive-from-primary:** upstream sources re-read; de-share sweep re-run on the raw cache; equation module
  re-read + edited; flip_bc gate JSON/log re-read (not memo-trusted). §8.5 no-borrowed-number: the 42 s/ep is quoted
  WITH its v7.5.2 vehicle + surface (never transferred to v8-inc1a).

## TRIALITY
- **DAG:** FEED-proxy-audit appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** `v8_geometric_rate_decomposition_20260709.py` annotated (0.00277 pin + corrected-triple 0.135 +
  de-share footprint band) — append-preserving comment-only, ruff-F clean, imports.
- **DSL:** **N/A** — this is an AUDIT (verification + one $0 measurement + comment annotation); no trainer/launch/
  curriculum lever changed. Say-so per the triality drift discipline.

**Pointer 0.19110 UNMOVED — this audit is MEANS.** Only a byte-closed `upstream/evaluate.py` n600 exact row < 0.19110
moves it. Item-3's bounded dry-start is the single owed execution (PINNED with runbook). `[no chain launch]`.
