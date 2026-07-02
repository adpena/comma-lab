# Wave D — 128GB-max training + decode memory-tier configurability (DESIGN, NO-GPU)

**Date:** 2026-07-02 · **Author:** design subagent (read-mostly; ran alongside the concurrent re-review, did NOT edit the trainer)
**Authority:** `[macOS-MLX advisory / design]` NON-PROMOTABLE. Pointer UNMOVED at contest-CPU **0.19110**. This is a
MEANS artifact (a spec + build-plan); the only END is a byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py`.
**Scope (operator 2026-07-02):** Part A = maximize the SINGLE `train_128gb_max` tier (M5-Max 128GB) for score + wall-clock;
configurable-memory TRAINING is DEFERRED (separate task). Part B = the DECODE-side multi-tier axis only
{`decode_t4_16gb`, `decode_cpu_16gb`, `production_edge`}.
**NO-FAKE:** every number below is either MEASURED-from-code (arithmetic on the real shapes / a trainer comment) or
DERIVED (labeled). Score-magnitude claims are labeled SPECULATIVE; mechanisms are labeled grounded with the anchor.

---

## 0. TL;DR verdicts (lead with the answers)

1. **Naive full-frame render supersample is MEMORY-INFEASIBLE at n600** and is NOT the way to spend the headroom. The
   per-pair self-orient coord-feats cache is already **41.5 GB at render 384×512** (confirmed by the trainer's own
   OOM comment, line 1770) and scales **quadratically** in the linear render factor: ss2 (768×1024) = **166 GB**,
   camera-res (874×1164) = **215 GB**, ss3 = **374 GB** — all far past the ~115 GB safe ceiling. The operator's
   "ss3 fine cache ~2.25×→32GB" only closes if AA is restricted to the boundary annulus (a ~1.5× LINEAR sub-render on
   ~5–10% of pixels), NOT the full frame.
2. **The real max-128GB levers** (ranked below): #1 **batched forward** (the training loop is currently per-pair
   SERIAL `value_and_grad` — batching K≈16–32 pairs buys MLX-GPU throughput = shorter wall-clock; the single biggest
   win, a BUILD item); #2 **boundary-annulus importance-supersample AA** (the ONLY memory-affordable render-AA score
   lever, grounded in the level-set/observation-AA math, magnitude SPECULATIVE, bounded by the decode budget);
   #3 mx.compile residency. "Cache more" has **little headroom left** — the per-epoch recomputes (reorient argmax,
   realized verdict) are inherently weight-dependent and CANNOT be cached across epochs; the static quantities
   (GT argmax/pose, coord-feats between reorients) are ALREADY cached.
3. **mod-dim / n-dir-freqs capacity: confirmed NO** (rate cost, not score; nonlinear intrinsic dim ~9 ≪ mod-dim 26 is
   already adequate). Stays fixed. Spending headroom on capacity would RAISE the rate term.
4. **Decode FITS T4-16GB and CPU-16GB comfortably** at the proven render-res 384×512 (per-worker RAM ~0.6 GB; 4 workers
   ≈ 2.4 GB; ~13–15 min inside the 30-min budget). **NOT a compliance blocker as-is.** The `.raw` is streamed to DISK
   (3.66 GB/video), not RAM. The controlling coupling: **render-res is baked into the witness and its ceiling is set by
   the WORST decode target (4-core CPU, 30-min budget), NOT by the 128GB training box** — quadratic decode-time growth
   means ss2 would blow the 30-min budget on 4 cores (~52–60 min). So Part-A supersample is gated by Part-B budget.
5. **Tier-invariance holds** iff the contest tiers fix {render-res, fp64 forward, CPU-torch R, 1-thread BLAS/worker} and
   vary ONLY {workers, tiling, streaming}. Those three vary-safe knobs are proven bit-identical (max_abs_uint8_diff=0).
   The DANGEROUS knobs (fp32 forward, CUDA-R, multi-thread BLAS) can flip a uint8 at the boundary annulus → change the
   score → they are FORBIDDEN on contest tiers (allowed only on `production_edge`, which is not contest-score-bound).

---

## 1. Measured memory model of the current #205 config (the baseline)

The proven-base config (`src/tac/witness_autoconfig.py::_proven_base` + `derive_config`): render 384×512, self-orient ON
(curvelet bank n_scales=4/n_orient0=6/n_iso=4 → **80** sin+cos cols; n_dir_freqs=2 → **8** dir cols; **in_feat=88**),
n600, mod-dim 26, hidden 96, n_hidden 4, accum-pairs 8.

Operator's stated breakdown **~63 GB = fine 14 + base 41 + fwd 8**. Confirmed / attributed:

| Term | ~GB | What it is | Confirmation |
|---|---:|---|---|
| **base** | **41.5** | per-pair self-orient coord-feats **MLX** cache `cf_mx_cache` = 600 × (384·512) × 88 × 4 B | **EXACT** — trainer line 1770: "the naive list-comprehension held old+new => **2x ~41GB at n600** => OOM"; my arithmetic = 41.52 GB |
| **fine** | ~14 | numpy working set: GT cache resident (~4.8 GB n600, load_gt comment) + per-pair numpy dir-feats (600×196608×8×4 = 3.8 GB) + shared curvelet numpy (63 MB) + verdict/scorer buffers + MLX pool overhead | DERIVED (components sum to ~9–14 GB; operator's 14 is the envelope) |
| **fwd** | ~8 | transient per-pair forward+backward activations (196608×96×4 B/layer × ~4 layers × (fwd+bwd)) + accumulated grads (param-sized) + `mx.eval(accum)` buffers | DERIVED |

**Headroom:** 128 − 63 = ~65 GB gross; with a ≥13 GB OS floor (memory_guard 10 GB + margin) the safe working ceiling
is **~115 GB → ~52 GB of usable headroom** over the current 63 GB.

**The dominant term is the per-pair coord-feats cache (41 GB).** This is already a memory-for-wallclock WIN (it avoids
recomputing 600 per-pair 196608×88 feature tensors every step). Do NOT free it to make room for supersample — that
trades memory for recompute (wrong direction). The headroom is spent on batching + optional annulus-AA.

---

## 2. Part A — the maximize-`train_128gb_max` delta (ranked; grounded vs speculative)

### Why full-frame supersample is OUT (the crux)

The observation operator R is `render(rh,rw) → bicubic-UP → camera(874×1164) → uint8 STE @ camera → bilinear-DOWN →
scorer(384×512)` (`apply_contest_faithful_roundtrip_nhwc`, pr95_hnerv_mlx_training.py:126). Rendering ABOVE the scorer
res and letting the camera-uint8 + bilinear-down AVERAGE it is a genuine anti-alias of the flip-prone boundary annulus
(the mip-NeRF-IPE / supersample→box argument, MEMORY `project_sig_proc_filter_chain_measured_R_allpass_L3_ntk`). So the
*direction* is sound. But the cost is fatal at full frame because the per-pair coord-feats cache scales as render-px:

| render | linear ss | base cache (GB) | verdict |
|---|---|---:|---|
| 384×512 | 1.0 | **41.5** | current |
| 768×1024 | 2.0 | **166** | INFEASIBLE (>115) |
| 874×1164 (camera) | 2.28 | **215** | INFEASIBLE |
| 1152×1536 | 3.0 | **374** | INFEASIBLE |

The AA benefit also has a hard Nyquist ceiling: the camera-res uint8 quantization discards sub-camera detail, so
render-res **above camera res (ss>2.28) yields ~0 additional d_seg** regardless of memory. The meaningful AA band is
render ∈ [384×512, 874×1164]. But the whole band is memory-infeasible full-frame — hence annulus-restricted AA below.

### Ranked levers

**#1 — Batched forward over K pairs `[WALL-CLOCK; grounded mechanism; BUILD; ~52 GB affords K≈16–32]`**
- The training loop (lines 2196–2228) is **per-pair SERIAL**: one `value_and_grad(total_loss_fn)` per pair, grads
  accumulated in a Python loop with `mx.eval(accum)` each pair. This under-utilizes the MLX GPU (many tiny forwards).
- `LevelSetRGBWitness.call_batch(coord_feats, code_indices)` already exists (line 486) — the batched forward is BUILT
  but the loop doesn't use it. Refactor: batch the K pairs of a chunk into ONE `value_and_grad` over stacked codes +
  stacked per-pair feats (already resident in `cf_mx_cache`), so K pairs' activations are held simultaneously.
- Memory: K × per-pair activations ≈ K × ~0.6 GB. K=16 ≈ 9.6 GB, K=32 ≈ 19 GB — inside the ~52 GB headroom.
- Expected effect: fewer Python iterations + larger GEMMs = better GPU util = **shorter wall-clock/epoch** (grounded;
  magnitude SPECULATIVE until timed). Score UNCHANGED at fixed optimizer/LR IF the batched loss is bit-faithful.
- **PARITY RISK (must gate):** several loss terms are pair-INDEPENDENT or whole-code-matrix functions (comments at
  lines 1279, 1294, 1472–1473: rank-floor, code-nuclear, code-spectral-entropy applied ONCE per opt-step, not P-scaled).
  A batched `value_and_grad` must reproduce the exact per-pair-then-mean magnitude and apply the once-per-step terms
  once — verified by a numpy-parity check (batched grad == serial-accumulated grad) BEFORE any run. This is the single
  highest-value Wave-D build; keep it OPT-IN (`--batch-forward K`, default OFF = byte-identical to today).

**#2 — Boundary-annulus importance-supersample AA `[SCORE; grounded mechanism; magnitude SPECULATIVE; BUILD; +~14–20 GB; decode-budget-bounded]`**
- Restrict AA to the codim-1 separatrix: render the full frame at base res + re-render only the flip-prone annulus
  (small-margin band, = the Fisher/margin surrogate 0.978, MEMORY unified-flow) at ~1.5× linear, composite before R.
- This is the operator's "fine cache 2.25×→32GB" reconciled: 1.5² = 2.25× on the annulus subset (~5–10% of pixels) is
  a bounded +~14–20 GB, NOT the +125 GB of a full-frame ss2.
- Grounded: d_seg lives ONLY on the boundary; interior is argmax-stable (dark in the Fisher metric). AA there is where
  the score sits. Magnitude SPECULATIVE (no measured row yet).
- **HARD COUPLING to Part B:** the annulus-AA must be REPRODUCED at decode (it changes the rendered frames), so it adds
  decode compute → it is bounded by the 30-min 4-core budget (§3). Design it decode-cheap (annulus mask is a cheap
  margin threshold of the decoder's own argmax — GT-free, rule-118) or it dies on the decode side.

**#3 — mx.compile residency of the (batched) fwd+bwd graph `[WALL-CLOCK; grounded; small BUILD]`**
- Compile the batched forward once; reuse the graph across steps. Composes with #1. Modest, grounded.

**#4 — Larger `--verdict-pairs` (96 → up to 600) + realized-hardness `[SCORE(indirect)+telemetry; SPECULATIVE; small mem]`**
- The async verdict currently scores 96 pairs (`--verdict-pairs 96`). At n600 the full-600 realized d_seg is truer
  telemetry AND feeds LEVER-5 realized-hardness oversampling on the TRUE hard pairs (`--hardness-source realized`).
  Memory: async CPU-torch scorer on more pairs (bounded, off the MLX heap). Score effect indirect (better hard-pair
  targeting), SPECULATIVE; wall-clock TAX (more scoring). Optional, low priority vs #1/#2.

### Explicitly NOT levers (confirmed)
- **mod-dim / n-dir-freqs / hidden-dim capacity — NO.** Nonlinear intrinsic dim ~9 ≪ mod-26 (already adequate per
  `witness_autoconfig.intrinsic_dim`); more capacity RAISES the archive rate term (the binding sub-0.15 lever), not the
  score. Stays fixed. (This is the correct reading of "capacity: NO" in the prompt.)
- **"Cache more" of per-epoch recomputes — NO headroom.** Reorient argmax + realized verdict are weight-dependent
  (change every epoch by construction) → un-cacheable across epochs. GT argmax/pose + between-reorient coord-feats are
  ALREADY cached. The win is batching (#1), not more caching.
- **Free the 41 GB cache to fit full-frame ss — NO.** Trades memory for recompute (more wall-clock).

**The max-128GB delta to the all-levers config (flags):** add `--batch-forward 16` (new, default OFF; the #1 build) and,
once #2 lands, `--annulus-aa --annulus-aa-scale 1.5 --annulus-aa-margin <band>` (new). No change to mod-dim/hidden/
render-h/render-w. Safe peak estimate at K=16 batch + annulus-AA: 41 (base) + 14 (fine) + ~10 (K=16 batched fwd) + ~16
(annulus fine cache) ≈ **~81 GB** — inside the 115 GB ceiling with ~34 GB margin. K=32 without annulus ≈ **~74 GB**.

---

## 3. Part B — the DECODE memory-tier axis (multi-tier)

**Decode has NONE of the training caches** — no `cf_mx` 41 GB, no GT cache, no optimizer/EMA. The inflate path
(`tools/levelset_byte_close_and_eval.py::_INFLATE_PY`) is: per-pair, regenerate the FREE curvelet bank + self-orient
dir-feats, run the numpy **float64** forward at render-res, `torch` bicubic-R to camera, write uint8 to a preallocated
`.raw` via seek (POSIX-concurrent). Already parallel across pairs (process Pool, 1-thread BLAS/worker) — proven
**10.8× on M5 Max (15 workers), max_abs_uint8_diff=0**.

### Per-worker decode RAM (render 384×512), measured by arithmetic
curvelet feats fp32 63 MB + in_proj feats fp64 138 MB + h0 fp32 75 MB + per-layer h fp64 151 MB (transient) + phi/tex/
rgb ~13 MB + torch-R fp32 in 12 MB + params <5 MB ≈ **~0.5–0.6 GB/worker**. The `.raw` is on DISK (n600 = 1200 ×
874·1164·3 = **3.66 GB/video**), not RAM.

### The tier knobs (all already env-driven or a small build)
| Knob | Mechanism | Status |
|---|---|---|
| `INFLATE_WORKERS` | process-pool width (RAM ≈ workers × 0.6 GB) | **EXISTS** (env) |
| 1-thread BLAS/worker | `*_NUM_THREADS=1` set before numpy import | **EXISTS** (bit-identity + no oversubscribe) |
| streaming `.raw` to disk | preallocate + seek-write (no full frame-buffer in RAM) | **EXISTS** |
| spatial frame tiling | render/R the frame in row-blocks, reassemble (per-pixel independent → bit-identical) | **BUILD** (only if a tier is RAM-starved below ~1 GB) |
| `INFLATE_MAX_PAIRS` | bounded inflate | **EXISTS** but DEBUG-ONLY (changes frame COUNT → not for scoring) |

### Per-tier decode config
- **`decode_t4_16gb`** (contest g4dn.xlarge-class, 4 vCPU/16 GB): `INFLATE_WORKERS=4` (or up to 6 if ≥6 vCPU),
  1-thread BLAS, CPU forward (fp64), CPU-torch R, render-res = baked. RAM ~2.4–3.6 GB ≪ 16 GB. No tiling needed.
- **`decode_cpu_16gb`** (contest CPU runner, 4 core/16 GB): identical to T4 tier (the forward is CPU either way).
  `INFLATE_WORKERS=4`. RAM ~2.4 GB. No tiling needed.
- **`production_edge`** (comma.ai smaller GPU / device): `INFLATE_WORKERS=1–2`, enable spatial tiling + streaming;
  MAY relax to fp32 forward and/or a device kernel (molt Python→WASM/WebGPU, MEMORY `molt_all_in_...`) since production
  is generalize-not-score → **bit-identity to the contest witness is NOT required here** (flagged in §4). If exactness
  IS wanted, keep fp64 + CPU-R at lower throughput.

---

## 4. T4-16GB / CPU-16GB fit verdict + tier-invariance

### Fit verdict: **FITS — NOT a compliance blocker (at the proven render 384×512)**
- **RAM:** 4 workers × 0.6 GB ≈ 2.4 GB + 3.66 GB `.raw` on disk ≪ 16 GB. Huge margin.
- **30-min budget:** anchor = 10.8× @ 15 workers on M5 Max. At 4 workers on a contest CPU, ~13–15 min for n600 at
  384×512 → **inside 30 min**. (Cross-ref #214: the multiprocess inflate is exactly what keeps it inside budget.)
- **DISK:** need ~3.66 GB scratch for `.raw` per video (contest = 1 video) — confirm the runner has it (g4dn SSD: yes).

### The controlling constraint (flag to operator)
**Render-res is baked into the archive manifest → the decode cost is quadratic in it → the render-res CEILING is set by
the 30-min budget on the SLOWEST contest target (4-core CPU), not by the 128GB training box.** If a future max-128GB run
raises render-res (e.g., ss2), n600 decode ≈ 4× → ~52–60 min at 4 workers → **OVER the 30-min budget → compliance
BLOCKER.** Consequences:
- Keep render-res at 384×512 for the FIRST byte-closed row (it fits with margin).
- Any render-AA (Part A #2) must be decode-cheap (annulus-only) or it breaks the budget.
- A CUDA-forward decode could afford more on T4 — but breaks CPU/CUDA bit-identity (§4 risk) AND the CPU tier still
  can't. Defer.

### Tier-invariance (bit-identical decode → tier-invariant bytes → tier-invariant score)
The archive.zip is built ONCE (byte-close) — its `st_size` (the rate term) is tier-invariant by construction. The score
is tier-invariant **iff the decoded `.raw` is bit-identical across tiers.** That reduces to fixing three things and
varying only three:

| Knob | Varies per tier? | Bit-identical? | Rule |
|---|---|---|---|
| `INFLATE_WORKERS` | yes | **YES** (proven max_abs_uint8_diff=0) | SAFE |
| spatial tiling | yes | **YES** (coords per-pixel independent) | SAFE |
| streaming `.raw` | yes | **YES** (same math, different scheduling) | SAFE |
| render-res (ss) | **NO — baked** | n/a | changing it = a DIFFERENT witness (re-byte-close + re-measure) |
| fp precision (fwd fp64) | **NO — must fix** | fp32 forward → NOT identical (different rounding → uint8 flips at annulus) | contest tiers MUST keep fp64 |
| R backend (CPU-torch bicubic) | **NO — must fix** | CUDA bicubic differs at ULP → round-to-uint8 can flip a boundary pixel → d_seg changes | contest tiers MUST run R on CPU |
| BLAS threads (1/worker) | **NO — must fix** | multi-thread GEMM reduction-order → ULP → uint8 flip | 1-thread/worker is a bit-identity GUARD, not just anti-oversubscribe |

**Guarantee:** under the contract {render-res, fp64 forward, CPU-torch R, 1-thread BLAS/worker} FIXED and
{workers, tiling, streaming} FREE, decode is bit-identical across `decode_t4_16gb` and `decode_cpu_16gb` → the score is
tier-invariant. The inflate.py ALREADY enforces fp64 + CPU-R + 1-thread BLAS, and the bit-exact round-trip gate
(`bit_exact_roundtrip_gate`) proves it against the numpy-fp32 oracle.
**Risks (flag):** (1) `production_edge` fp32/CUDA/device-kernel path is NOT bit-identical → acceptable there (not
score-bound) but must NEVER be used for a contest tier; (2) the uint8-round-at-camera makes ALL the ULP-class diffs
score-relevant precisely on the flip annulus — so the three "must fix" rows are load-bearing, not paranoia; (3) if the
annulus-AA (Part A #2) lands, its annulus MASK must be derived identically at train and decode (decoder-own-argmax
threshold, GT-free) or train/decode diverge.

---

## 5. Grounded vs speculative split (NO-FAKE honesty)

**GROUNDED (measured-from-code or a trainer comment):**
- base cache = 41.5 GB (line 1770 + arithmetic); full-frame ss quadratic blowup (166/215/374 GB); per-worker decode RAM
  ~0.6 GB; decode fits 16 GB with margin; training loop is per-pair serial (lines 2196–2228); `call_batch` exists;
  R chain resolutions (pr95_hnerv_mlx_training.py:126); `.raw` on disk; INFLATE_WORKERS/1-thread-BLAS/streaming exist;
  10.8× @ 15w proven; bit-exact gate exists; capacity is a rate lever not a score lever.
- Decode-time quadratic-in-render-res → render-res ceiling set by 30-min-on-4-core: grounded (arithmetic on the proven
  10.8×/15w anchor).
- Tier-invariance contract (which knobs flip a uint8): grounded in the R = uint8-round chain + the proven
  max_abs_uint8_diff=0 across workers.

**DERIVED (arithmetic, reasonable but not directly measured):**
- "fine 14" decomposition (GT 4.8 + numpy dir-feats 3.8 + buffers); exact per-worker decode peak; ~13–15 min @ 4 workers
  (scaled from the M5 anchor); safe-peak estimates (~81 GB at K=16+annulus).

**SPECULATIVE (mechanism grounded, MAGNITUDE unproven — needs a measured row):**
- The d_seg BENEFIT of annulus-AA (mechanism = observation-AA of the separatrix; no measured ΔS yet).
- The wall-clock SPEEDUP of batched forward (mechanism = GPU util; magnitude needs timing).
- Larger verdict-pairs → better hardness targeting → lower d_seg (indirect, unproven).

---

## 6. Wave D build-plan (ordered; each with a fail-closed gate)

1. **Batched forward `--batch-forward K` (default OFF)** — refactor lines 2196–2228 to a batched `value_and_grad` over K
   stacked pairs using `call_batch`. GATE: numpy parity test `batched_grad == serial_accum_grad` (tree-allclose) + a
   $0 macOS-MLX 2-pair smoke showing byte-identical loss at K=1 vs the current path. Then a wall-clock A/B at K∈{8,16,32}.
   Deliverable: chosen K + measured epoch-time delta. `[WALL-CLOCK]`
2. **Decode-tier config surface** — a small `--memory-tier {decode_t4_16gb,decode_cpu_16gb,production_edge}` on the
   inflate/byte-close path that sets `INFLATE_WORKERS` + (edge) tiling/streaming defaults; contest tiers hard-pin
   {fp64, CPU-R, 1-thread BLAS}. GATE: the existing `bit_exact_roundtrip_gate` must pass across ALL contest tiers
   (t4 vs cpu) with max_abs_uint8_diff=0 on ≥8 pairs. `[COMPLIANCE]`
3. **Decode 30-min budget probe** — a $0 timing smoke (n600, 4 workers, render 384×512) recording wall-clock; assert
   < 30 min with margin; record the render-res→time curve so the render-res ceiling is a MEASURED number the max-128GB
   tier must respect. `[COMPLIANCE]`
4. **(optional, after 1–3) Boundary-annulus AA `--annulus-aa`** — annulus mask = decoder-own-argmax margin threshold
   (GT-free, rule-118), 1.5× sub-render composited before R, reproduced identically in inflate.py. GATE: train/decode
   annulus-mask parity + the byte-close row measured; KEEP only if a real byte-closed n600 row shows ΔS < 0 AND decode
   stays < 30 min on 4 cores. `[SCORE]`
5. **spatial tiling** for `production_edge` only, if a device needs < ~1 GB/worker. GATE: bit-identical to the untiled
   `.raw`. `[PRODUCTION]`

**Sequencing rule:** #1 (throughput) + #2/#3 (decode fit + budget curve) are the prerequisites; they enable the FIRST
byte-closed exact row at 384×512 WITHOUT changing the witness. #4 (annulus-AA) is the only score-moving item and is
gated by both a measured ΔS AND the decode budget. Nothing here moves the pointer until #202/#214 land a real
`upstream/evaluate.py` n600 row < 0.19110.

---

## 7. Cross-references
- Config source of truth: `src/tac/witness_autoconfig.py` (the #205 actuator; the named `.md` artifact is not on disk —
  the module IS the canonical config).
- Trainer (LEVELSET entry point): `experiments/train_levelset_witness_realized_through_R_mlx.py` (loop 2196–2228; cache
  1763–1780). Base primitives + R: `experiments/train_witness_realized_through_R_mlx.py` (343–379),
  `src/tac/local_acceleration/pr95_hnerv_mlx_training.py:126`.
- Decode/byte-close/inflate + bit-exact gate: `tools/levelset_byte_close_and_eval.py` (#202); multiprocess inflate =
  #214. Memory guard: `tools/memory_guard.py` (10 GB floor, never kill control plane).
- MEMORY anchors: `project_sig_proc_filter_chain_measured_R_allpass_L3_ntk` (observation-AA / mip-NeRF-IPE),
  `project_unified_variational_levelset_flow` (d_seg on the separatrix / Fisher=margin), `submission_eval_axis_is_ours`
  (CPU/GPU decode axis is ours; pay-error-for-speed), CLAUDE.md "Contest vs production target modes" +
  "Native eval-time runtime discipline" + the 30-min T4/CPU budget.
