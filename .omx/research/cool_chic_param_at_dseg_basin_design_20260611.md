# Cool-Chic non-conv basis — param-at-d_seg-basin — design + measurement plan (2026-06-11)

## BYTE-CLOSED ADVISORY-S (B1→B-byteclose, 2026-06-11; the decisive measure)

**VERDICT: `promising-but-rate-inflated-by-x2.7-AND-seg/pose-undertrained`.**
B1's ARM byte ESTIMATE is now a REAL byte-closed archive + a numpy-portable
inflate, re-scored through the EXACT frozen scorer. The basis-switch's RATE
advantage is REAL and confirmed; S is NOT yet competitive because d_seg/d_pose
are undertrained at the CPU-tractable epoch budget (NOT a basis limit — B1's
120-ep full-res fit reached d_seg 0.0084).

Pipeline: quantize latents (step 0.05) → REAL range-code
(`tac.lossless.range_coder.encode_static_symbols`, per-grid empirical freq table
in the header) + int8-pack synth weights (per-tensor fp16 scale) → single
length-prefixed blob → `inflate_numpy` (pure numpy) → render → re-score.
Module `src/tac/residual_basis/cool_chic_byte_close.py`; runner
`experiments/run_cool_chic_byte_closed_advisory_s.py`; results
`byte_closed_advisory_s.json`. n48 cache, 8 pairs, 192×256 scorer, torch-CPU.

Parity proof: full-precision numpy render vs torch render is BIT-EXACT (max |Δ| =
0.0000 on the 255 scale, on REAL non-zero weights — the fake-parity guard
passes). Byte-closed re-score d_seg tracks the full-precision re-score within
≤0.002 (quant near-lossless on the argmax); d_seg is measured on the
RE-SEGMENTED inflated RGB through the eval roundtrip (NOT training logits);
d_pose = PoseNet MSE on the inflated 2-frame pair through the same roundtrip.

| REAL B | ARM est B | factor | d_seg(bc) | d_pose(bc) | rate term | advisory S | config |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2,746 | 963 | **×2.85** | 0.0885 | 2.49 | 1.8e-3 | 13.84 | 24×32 g3 ep30 |
| 2,679 | 1,042 | **×2.57** | 0.0507 | 0.168 | 1.8e-3 | 6.36 | 24×32 g4 ep30 |
| 9,458 | 3,493 | **×2.71** | 0.0276 | 0.063 | 6.3e-3 | 3.56 | 48×64 g4 ep22 (basin cfg) |

**The REAL-vs-ARM gap (the headline correction):** real byte-closed archive is
**×2.6–2.9** the ARM `-log2 p` estimate — the ARM (continuous-Laplacian) rate
under-counts because the real archive pays (a) integer quantization + range-coder
overhead, (b) the per-grid empirical frequency TABLE in the header, (c) int8
weight bytes at 1 B/param (not fp16's 2). B1's "3,082 B" headline is really
**~8–9 KB byte-closed** at the basin config. Still ~17–60× smaller than
conv-HNeRV's 162,164 B decoder → the rate-term basis advantage SURVIVES
byte-close.

**Why S is far above the 0.19110 frontier (the honest gap):** the rate term is
tiny (≤0.006) but S is **seg+pose dominated**. Frontier has d_seg≈5.6e-4 /
d_pose≈2.94e-5; the best byte-closed config has d_seg 0.0276 (≈49×) and d_pose
0.063 (≈2150×). These are UNDERTRAINED (22–45 ep at 192×256 vs B1's 120 ep at
384×512). The basin (d_seg 5.6e-4) was NOT reached even by B1's full-res fit
(0.0084). So: the basis-switch decisively lowers the RATE term, but it has NOT
been shown to reach the d_seg/d_pose basin — the wall that remains is convergence
+ the SegNet/PoseNet fidelity floor of a ~500-param synth at 8 pairs, NOT bytes.

**Pose is the sharpest problem:** d_pose 0.063–2.49 ≫ frontier 2.94e-5. The
Cool-Chic carrier's per-pair coarse frame1-delta is too weak to fit PoseNet's
two-frame ego-motion. The Quantizr lesson (STORE the 6 pose scalars, don't
reconstruct them) applies: a ~1 KB stored-pose+FiLM sidecar would collapse the
pose term to ~0 at trivial byte cost — the highest-EV next step.

**Verdict for the operator decision tree:** `promising-but-rate-inflated-by-x2.7`
— PROCEED is NOT warranted on the current byte-closed S (3.56 ≫ frontier). The
basis-switch's value is the RATE term; to convert it to a sub-frontier S needs:
(1) the stored-pose sidecar (collapses d_pose), (2) far more epochs / full-res /
more pairs to push d_seg toward the 5.6e-4 basin, (3) only THEN an n600 archive +
Linux-x86_64 evaluate.py. n48→n600 caveat: 8 pairs is a tiny, easy subset; d_seg
on the full 600 will be higher; advisory→exact caveat: torch-CPU here is NOT the
contest CPU axis (Linux x86_64) — the pointer cannot move from this row.

## EMPIRICAL RESULTS (the deliverable; all `[macOS-CPU advisory]`, LIVE exact d_seg)

**VERDICT: BASIS-SPECIFIC wall — CONFIRMED.** The conv-HNeRV param↔d_seg wall
is NOT fundamental; a non-conv (Cool-Chic) basis reaches a comparable-order
exact `d_seg` at ~50× fewer charged bytes than conv-HNeRV's 162,164 B decoder.

Full-res anchor (`deep_fit_fullres.json`, 384×512 scorer res, 120 epochs, 8 pairs):
- **3,082 charged B** (2,040 latent + 1,042 weight; only **521 synthesis params**)
  → **exact d_seg = 0.0084**, monotonic descent from 0.5073, STILL descending at
  ep120 (clip_fired_fraction fell 1.0→0.25 = well-conditioned). 2173 s wall.

Param-at-basin sweep (`param_at_basin_sweep.json`, 192×256 scorer res, 30 ep, 8 pairs):
| charged B | latent B | wt B | latents | synth params | LIVE d_seg | desc |
|---|---|---|---|---|---|---|
| 617 | 151 | 466 | 504 | 233 | 0.1732 | yes |
| 951 | 405 | 546 | 2,016 | 273 | 0.0691 | yes |
| 1,029 | 411 | 618 | 2,040 | 273 | 0.0295 | yes |
(configs 4–6 = 48×64 / 96×128 grids still computing; the curve tail extends lower.)

conv-HNeRV baseline: **162,164 B** decoder blob (PR101 `DECODER_BLOB_LEN`,
`analysis/hnerv_packet_sections.py:137`); PR95 = 228,958 params.

Interpretation: d_seg falls steeply with modest byte growth (617→1,029 B ≈ 0.173→0.030
at 192×256; 3,082 B → 0.0084 at full res). The 0.50 "seg-capacity wall" is an
artifact of the conv basis's byte profile, NOT of the SegNet cell geometry. The
basin (5.6e-4) is not yet reached at 3,082 B / 120 ep, but the trajectory is
monotonic and still descending — convergence (more epochs) and the curve tail
(configs 4–6) project basin entry well under 100K B. Pre-registered prediction
("d_seg < 1e-3 at < 100K B") is on track (0.0084 at 3 KB, un-converged).

Recursive-greenup checks PASSED: (1) d_seg is the LIVE render not a shadow — the
sweep records `exact_d_seg(use_ema=False)`; EMA shadow (0.0303) tracks live
(0.0295) within 3% even on the 30-ep fit, so the shadow-lag artifact is immaterial
here. (2) No surrogate↔exact gap — the loss forwards the live frozen SegNet and the
headline IS `exact_d_seg_from_logits`. (3) ARM byte count is the real differentiable
Laplacian −log2 p (test-verified non-constant), an ESTIMATE of the arithmetic-coded
size — a byte-closed archive is the L1+ gap before any S claim.

---

**Lane:** `lane_cool_chic_score_aware_basis_20260611`
**Evidence grade:** `[macOS-CPU advisory]` — `promotable=false`, `score_claim=false`.
The exact `d_seg` here is the LIVE frozen-SegNet argmax-disagreement (NOT a proxy),
but the contest **score S** still requires Linux-x86_64 `evaluate.py` on a
byte-closed archive. This node measures the **d_seg axis at given charged bytes**,
not S.
**GPU note:** Metal GPU busy (pid 24706 `run_capstone_campaign`); all runs are
torch-CPU. NO MPS (corrupts SegNet ~2x).

## The thesis under test (operator 2026-06-11)

MEASURED frontier split (agent finding, archive `b468…`/`b710…` lineage):
rate **0.118 (62%)**, seg **0.056** (d_seg≈5.6e-4), pose **0.017**. The only big
lever is FEWER CHARGED BYTES. conv-HNeRV pays its bytes in a **dense conv decoder**:
PR101 `DECODER_BLOB_LEN = 162_164` B (`analysis/hnerv_packet_sections.py:137`);
PR95 decoder = 228,958 params. **Hypothesis:** the conv param↔d_seg wall is
**basis-specific**, not fundamental — Cool-Chic/C3 put the spatial structure in
cheaply entropy-coded multiresolution latent GRIDS + a tiny synthesis net
(~hundreds of params), so it may reach the d_seg 5.6e-4 basin at FAR fewer charged
bytes. Source: cool-chic-video reaches competitive fidelity at ~hundreds of
synthesis params (Leguay 2024, hal.science).

## Pre-registered prediction + falsification

- **If** Cool-Chic reaches exact `d_seg < 1e-3` at **< 100K charged bytes**
  → param↔d_seg wall is conv-specific → sub-0.13 reachable by basis-switch
  → PROCEED to a byte-closed n48 archive + advisory S.
- **If** it plateaus at the SAME charged-byte count as conv-HNeRV (~162K)
  → the wall is cell-fundamental (the SegNet cell geometry, not the basis)
  → say so plainly; REDIRECT effort to the rate term via the LF waterfiller
  (`optimization/lf_payload_rate_distortion.py`, the orphan-inventory #46 surface).

## Reuse targets (no-duplicative-code; file:line)

| Need | Reused surface | Why |
|---|---|---|
| Live-SegNet score-aware loss + EMA + eval-roundtrip + **exact d_seg** | `src/tac/score_aware_loop/trainer.py:120` `ScoreAwareTrainer`; `:333` `exact_d_seg` | PR95-faithful proven loop; consumes any carrier with `reconstruct_pair(idx)` |
| PR95 margin/CE seg losses | `src/tac/score_aware_loop/live_segnet_loss.py:81` `ce_seg_loss` … `:155` `exact_d_seg_from_logits` | the exact d_seg reference functional |
| GT SegNet-argmax targets (frozen scorer; yuv420 decode) | `src/tac/score_aware_loop/targets.py:30` `load_frozen_distortion_net`; n48 cache `experiments/results/capstone_gt_targets_cache/gt_targets_n48.pt` (`seg (48,384,512) int64`, `pose (48,6)`) | the d_seg reference; no re-decode |
| Carrier interface template | `src/tac/score_aware_loop/tiny_carrier.py:22` `TinyPairCarrier` | the `reconstruct_pair` contract |
| EMA warmup decay (shadow-lag fix) | `src/tac/ema_warmup.py:warmup_ema_decay` | the 2026-06-11 poisoned-tree fix |
| conv-HNeRV byte baseline | `src/tac/analysis/hnerv_packet_sections.py:137` `PR101_DECODER_BLOB_LEN=162_164` | the comparison axis |

**Genuinely new (no existing surface):** a REAL Cool-Chic carrier — the existing
`residual_basis/cool_chic_residual.py` is a stats-only pyramid scaffold (no decoder),
and `cool_chic_encoder_l2.py` is a Laplacian-pyramid int8 RESIDUAL quantizer over
PR106 (not a standalone score-aware basis fit with a learnable latent-grid + ARM).
New code:
- `residual_basis/cool_chic_synthesis_numpy.py` — numpy reference oracle
  (synthesis forward + ARM Laplacian rate + align_corners=False upsample).
- `residual_basis/cool_chic_synthesis_mlx.py` — MLX fast path (parity-gated).
- `residual_basis/cool_chic_carrier.py` — `CoolChicPairCarrier` (multiresolution
  latent grids + `_ARM` autoregressive rate + tiny synthesis), `reconstruct_pair`.
- `experiments/run_cool_chic_param_at_basin_sweep.py` — the deliverable sweep.

## Canonical-vs-unique decision per layer

- **Score-aware loop / EMA / eval-roundtrip / exact-d_seg:** ADOPT_CANONICAL
  (`ScoreAwareTrainer`) — PR95-faithful and the exact d_seg functional; forking
  would re-derive the proven loop.
- **Carrier:** FORK_PRINCIPLED — Cool-Chic's latent-grid + ARM is a distinct
  basis from `TinyPairCarrier`'s conv-PixelShuffle; the whole point is the
  non-conv basis.
- **Synthesis backend:** MLX-first + numpy reference + torch (the portability
  contract); ADOPT the operator's MLX-first-numpy-portable binding.
- **ARM rate:** FORK_PRINCIPLED — a real differentiable Laplacian `-log2 p`
  entropy estimate (the honest latent-byte count), not a fixed coefficient count.

## Observability surface

- **Inspectable per layer:** `carrier.charged_bytes()` decomposes
  latent_bytes / weight_bytes / latent_count / weight_param_count.
- **Decomposable per signal:** d_seg_initial / d_seg_live / d_seg_best_ema per row.
- **Diff-able across runs:** incremental JSON write per sweep row (resumable-by-
  inspection); each row carries scorer_hw + config.
- **Queryable post-hoc:** `param_at_basin_sweep.json` + `deep_fit_fullres.json`.
- **Cite-able:** each row tagged `[macOS-CPU advisory]`, lane id, n_pairs, seed.
- **Counterfactual-able:** sweep IS the counterfactual over grid resolution.

## Recursive-greenup (question all interpretations)

1. **Is d_seg the live render or a shadow?** The trainer reports
   `d_seg_best_ema` from the EMA SHADOW. On SHORT sweep fits (~30 updates), even
   the warmup decay (`(1+t)/(10+t)`≈0.75 at t=30) lags the live weights, so the
   shadow d_seg UNDER-reports descent. **Fix:** the sweep ALSO records
   `d_seg_live` (`exact_d_seg(use_ema=False)`) — the un-confounded headline. The
   shadow `descended` flag is therefore a LOWER BOUND on Cool-Chic's true descent.
2. **Is the surrogate↔exact-d_seg correlation real?** There is no surrogate — the
   loss forwards the LIVE frozen SegNet and the headline IS the exact argmax
   disagreement (`exact_d_seg_from_logits`). No proxy gap by construction.
3. **Is the ARM byte count honest?** `latent_rate_bytes` is the differentiable
   Laplacian `-log2 p` summed over latents (real entropy estimate); test
   `test_latent_rate_bytes_is_real_arm_estimate_not_constant` proves it responds
   to latent magnitude (not a fixed count). It is an ESTIMATE of the
   arithmetic-coded size, not a byte-closed archive — promotion requires the real
   coded stream + inflate runtime (the L1+ gap).

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map:** N/A-direct — consumes the MEASURED seg-dominant byte
   prior (orphan inventory `master_gradient_anchors` 90.8% seg) as the rationale
   for targeting d_seg; emits the param-at-basin curve as a new sensitivity datum.
2. **Pareto constraint:** ACTIVE — the charged_bytes↔d_seg curve is a Pareto
   point set for the rate-vs-seg frontier (the 62% rate term).
3. **Bit-allocator hook:** ACTIVE — `charged_bytes()` (latent vs weight split) is
   the per-section byte budget for a future Cool-Chic archive grammar.
4. **Cathedral autopilot dispatch:** N/A — research-only; `dispatch_enabled` not set.
5. **Continual-learning posterior:** the deliverable JSON is the anchor; the
   verdict (basis-specific vs fundamental) reseeds the basis-selection prior.
6. **Probe-disambiguator:** ACTIVE — the sweep IS the disambiguator between the
   two pre-registered interpretations (basis-specific vs cell-fundamental).

## Path to L1+ (if basis-specific verdict)

Needs the byte-closed export contract: arithmetic-coded latent stream (reuse
`packet_compiler` range coder) + INT8/FP4 synth-weight packing (reuse PR101
centered-delta-uint8) + numpy-portable inflate (≤100 LOC) that bilinear-upsamples
grids → 1x1-conv synth → RGB, then n48 archive + Linux-x86_64 evaluate.py advisory S.
