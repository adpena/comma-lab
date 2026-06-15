# Small-basis (base_ch=20, 81KB) optimization register + comprehensive deep-audit

**Date:** 2026-06-15. **Context:** the operator's thesis — *"doing this on a very small [basis]
is useful as it's driving us to optimize rather than being lazy with large capacity ... there are
likely still bugs and optimization opportunities at this size."* This register saves the found
opportunities and structures a comprehensive micro→macro deep audit.

**Authority:** all numbers below are `[contest-CPU advisory] NON-PROMOTABLE` (in-loop CPU eval on the
96-pair subset) until a byte-closed `archive.zip` runs through `upstream/evaluate.py`. The frontier is
pointer-only (`.omx/state/canonical_frontier_pointer.json`); current public-CPU frontier ≈ 0.191.

## The score + the small-basis operating point (the lens)
`S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489`. Control@96-pair converged:
- d_seg = 0.00359 → seg term **0.359 (79%)**  ·  d_pose = 0.0002 → pose term **0.045 (10%)**  ·
  archive 76,592 B → rate term **0.051 (11%)**  ·  total **0.455**.
- **d_seg is the binding term.** Decoder = **83,356 params**; archive stores it at **~7.35 bits/param**.
- Sub-0.15 reachability arithmetic: if levers get d_seg→frontier (~0.0006 → seg 0.06) with pose 0.045:
  - int8 rate 0.051 → **0.156 (MISS)**   ·   FP4 rate 0.036 → **0.141 (CLEARS sub-0.15)**.
  - ⇒ both the d_seg lever AND the rate encoding are on the critical path.

## FOUND so far (this session) — ranked by EV
| # | scale | finding | mechanism / math | score impact | status | timing |
|---|---|---|---|---|---|---|
| ① | meso | **EMA decay 0.999 mismatched to 800-ep budget** | window 1/(1−0.999)=1000 > 800 → shadow never equilibrates, lags live, over-weights early epochs; shipped archive = shadow. Coherent ≈ 0.99 (window ~100). `v.ema_update` is constant-decay, no warmup (task #86 pending) | free absolute d_seg, all arms (ranking safe) | OPPORTUNITY | post-verdict (changing mid-A/B breaks apples-to-apples) |
| ② | micro | **decoder INT8 (~7.35 b/param) not FP4 (~4 b)** | `pose_film.py:270` "int8 decoder blob"; `build_archive` = quantize_state_dict + brotli. PR95 L21–L32 = FP4 + per-tensor byte-maps + split-brotli + range coding ≈ 3–4 b/param | −22KB → Δrate ≈ −0.015; the 0.156→0.141 sub-0.15 maker | OPPORTUNITY (FakeQuantFP4 + variable-level/WRQ codec exist, default OFF) | post-verdict; needs FP4-QAT + d_seg-holds check |
| ③ | meso | **margin-weight not renormalized** (`driver.py:218`) | total seg-grad magnitude shrinks as margins grow → compounding decay w/ T-anneal + LR-decay → premature seg-plateau risk | indirect d_seg | OPPORTUNITY (τ=1.0 partially mitigates) | post-verdict; full fix = ÷Σw |
| ↩ | meso | QAT under-trained — **RETRACTED** | use_qat active stages 3–7 = 526 ep (66%); the 13-ep flag was only the onset window | — | NOT A BUG | — |

Already FIXED this session (coherence): τ 0.3→1.0 (`7e4f83f37`), pose-throttle thr 0.005→0.001,
soft_cosine→CE-coarse schedule (stack_ce_early), pose-grad throttle (`278227ae3`), l2_combined
dangling-attr (`1a85a94c8`). No new wrong-behavior BUGS found in the training loop (gradient flow,
optimizer order, EMA-at-eval, throttle all correct).

## DEEP-AUDIT STRUCTURE (micro → meso → macro → bridge) — to be populated by the fan-out
### MICRO (bits / numerics / coding) — mostly RATE + numeric fidelity
quantization bit-width + per-tensor byte-maps + split-brotli + range/ANS (L21–L32); latent coding
(temporal-delta uint8, raw-LZMA); eval-roundtrip numerics (384→874→384 bicubic+uint8 straight-through);
MPS↔CPU drift in the gradient; archive-grammar byte overhead; fixed-point/Rust-lowerable hotspots.

### MESO (training loop / optimization) — d_seg + d_pose convergence
EMA (decay/warmup) ①; optimizers (Muon partition, AdamW capped-FiLM group, LR schedules, grad clip);
curriculum (8-stage proportions at 800/n96, the C1a/σ/λ schedules — are they right at small scale?);
loss surrogates (soft_cosine/CE/lovász/margin) + interactions + the throttle ③; QAT mechanics.

### MACRO (architecture / representation / strategy) — the d_seg ceiling + sub-0.15 path
HNeRV decoder @ base_ch=20 (PixelShuffle/sin/bilinear-skip/channel-taper — d_seg-optimal capacity
allocation?); representation store-vs-reconstruct audit (pose-FiLM done; seg-partition? motion-flow?
frame0-warp? what else should be STORED not reconstructed?); capacity↔d_seg↔rate Pareto + sub-0.15
reachability; basis alternatives (Cool-Chic/VQ/FF-NeRV) at the d_seg basin.

### BRIDGE (cross-scale couplings) — synthesized after the bands
e.g. FP4 (micro) ↔ d_seg-capacity (macro): does 4-bit quant cost d_seg? the variable-level codec
(micro) ↔ score-sensitivity EMA (meso, Lever-4) ↔ which tensors carry d_seg (macro).

## MESO audit results (2026-06-15) — new findings beyond ①②③
| id | type | finding | file:line | impact / math | timing |
|---|---|---|---|---|---|
| M2 | **BUG** | `eval_every=25` NOT scaled to the compressed budget → **stage 4 (QAT onset) gets 0 evals** at budget 800 (13ep<25); stage3=1, stage6=2. BEST-tracker is blind to the highest-risk INT8-introduction transient | `curriculum.py:231` (eval_every inherited verbatim) | selection blind-spot at QAT onset; no Δscore directly but a better stage-4 ckpt is unselectable | next launch (more evals on one arm = asymmetric selection → keep consistent across the A/B) |
| M3 | **BUG** | Muon LR scheduler **shares AdamW's `lr_lambda`** → `eta_min_ratio=max(lr_floor/adamw_lr,1e-3)` keyed to AdamW; at stage 8 = 5e-6/1e-5 = **0.5** → Muon LR floors at 0.5×peak (1e-4), never anneals (intended abs floor 5e-6 = 0.025×). Muon stays ~20× too hot in its only stage | `driver.py:770-780` | hot Muon tail adds late-stage noise to the shadow (compounds ①); 2-line fix = separate muon `lr_lambda` | post-verdict (changes optimizer trajectory → breaks apples-to-apples mid-A/B) |
| M5 | OPP | stage allocation is PR95-faithful-scaled, not small-basis-optimal: **30% of budget on stage5 c1a (a RATE regularizer; rate is 11% of S)** while CE-coarse stage1 is only 10% — mis-prioritized when d_seg is binding | `curriculum.py` proportional scaling | re-allocate ~10% c1a→CE-coarse/seg-refine; plausible Δd_seg | post-verdict (new arm to size) |
| M6 | OPP | σ/λ/C1a magnitudes are PR95 (229K-param) values, untuned for 83K; C1a entropy on a **2000-weight subsample w/ unseeded `randperm`** → stochastic rate-grad (higher relative variance at small numel) | `losses.py:79 cat_entropy_v2` | indirect Δrate; cheap fix = full-tensor entropy at base_ch=20 (numel small) | post-verdict |
| M8 | OPP/latent-BUG | Lever-4 score-sensitivity EMA drifts **seg-only** on pose-throttle skip epochs (pose cotangent absent 3/4 epochs) → seg-biased byte allocation if both levers on. **Dormant** at live config (Lever-4 off, k=1 default) | `score_aware_qat.py:256` + `driver.py:928` | latent coupling for any FP4/variable-level run that turns both on | when variable-level export + throttle both active |
| M1 | confirms ① | EMA mismatch — per-stage absorption: stage4 **14.5%**, stage6 **47.7%**, stage8 muon **80.2%** absorbed at 0.999; effect is LARGEST in the 64%-refinement stages. Warmup-decay strictly better than constant | `losses.py:155` | the binding-axis (shadow d_seg) inflation; capstone anchor showed +0.466 in extreme case | post-verdict (land for all arms together) |
| M9 | ✅ clean | EMA-at-eval snapshot/restore is CORRECT (deep-CPU-copy, never `ema.apply` on live weights → no DARTS-freeze); no VQ codebook to audit | `driver.py:1174-1198` | — | — |
| M10 | negligible | cosine LR uses `/spec.epochs` not `/(E−1)` → floor not quite reached (0.0067 vs 0.005 at E=13); <1% | `driver.py:773` | negligible | — |

**MESO top-3:** ① EMA window fix (binding-axis, free, biggest in refinement) · M2 eval_every scaling (QAT-onset blind-spot) · M5 stage re-allocation (c1a 30%→CE/refine). **Certain-correctness runner-up: M3 Muon LR-floor (2-line fix).** All post-verdict / next-launch (mid-A/B changes break apples-to-apples).

## MICRO audit results (2026-06-15) — measured byte composition + new findings
**Measured archive composition** (real EMA ckpt, CPU byte-inspection, no scorer loaded):
decoder_blob (int8+zigzag+brotli) = **73,604 B (96.2%, 7.06 b/param)** · latents = 2,828 B (3.7%) · meta+prefixes = 95 B · total 76,527 B (rate 0.05096). The int8 stream entropy is **7.249 b/symbol** → brotli is already near its floor. **⇒ the rate win is NOT better int8 coding; it's a smaller alphabet (true sub-8-bit PACKING). This reframes ②.**

| id | type | finding | file:line | impact / math | timing |
|---|---|---|---|---|---|
| μ1 | OPP **(sharpens ②)** | **True 4-bit PACKING is NOT wired** — the variable-level codec only shrinks the int8 *alphabet* (container stays 8-bit, leans on brotli); `src/tac/quantization.py` FP4/per-channel exists but the torch-vehicle build never calls it. **Measured: mixed FP4(79,980 interior-conv params)+int8(3,376 output-head params) = 39,990 B → archive 42,913 B → Δrate = −0.0224** (0.051→0.0286). Bigger than ②'s −0.015; the 0.156→**0.134** sub-0.15 maker | `losses/variable_level_codec.py:117`, vendored `codec.py:37`, build `driver.py:1246/1339` | Δrate −0.0224 (measured bytes); d_seg-hold is the open risk | post-verdict; needs FP4-QAT + d_seg-hold check |
| μ2 | OPP (enabler) | quantizer is **per-tensor symmetric int8** (one fat-tail channel inflates the whole-tensor scale); per-channel scales + PR95 byte-maps (L21 zig/twos/off) + CONV4 perms (L22) UNUSED. `quantization.py:332` has per-channel+FP4, unrouted | vendored `codec.py:30-39` | standalone ~−0.5-1KB; real value = lets μ1's FP4 hold d_seg at lower bits | post-verdict |
| μ3 | **OPP (training-signal / partial eval_roundtrip mismatch)** | proxy collapses up+down **before** the uint8 cast: `decoded(384)→bicubic↑874→bilinear↓384→clamp→round-STE`; contest casts uint8 **at 874** then the scorer resizes. **Measured pixel drift mean 0.43 / max 16.75** on [0,255] → the surrogate gradient optimizes slightly-wrong pixels, biased exactly at argmax boundaries (±1-LSB sensitive). BEST-tracker/authority IS faithful → this is a training-signal tightening, not a scoring bug | `driver.py:864-871` vs authority `driver.py:2072-2082` | indirect Δd_seg (tighter surrogate). Fix = round→uint8 at 874 then bilinear↓384 | post-verdict (changes loss surface) |
| μ4 | **BRIDGE (micro↔macro)** | SegNet stride-2 stem decides argmax at **~192×256**; the decoder renders 384→bicubic↑874. So d_seg is **blind to HF detail above ~192×256** — and FP4 quant noise is exactly HF → **d_seg should tolerate interior FP4 better than the 18× relL2 ratio suggests** (good news for μ1's viability) | `upstream/modules.py:108-109` + `driver.py:865` | de-risks μ1 (the FP4 d_seg-hold) | informs μ1 |
| μ5 | OPP (tiny) | fp32 per-tensor scales (112B) → fp16 (−56B); meta brotli (83B) larger than raw (79B) — store raw | `codec.py:68` | Δrate <0.0001 | low-EV |
| μ6 | ✅ near-optimal | latents 2,828B vs ideal ~2,579B (~250B headroom, range/ANS); already temporal-delta+zigzag, just brotli not raw-LZMA. 3.7% of archive → LOW EV | `codec.py:97-119` | ~−0.0001 | low-EV |
| μ7 | ✅ info | no codec Rust hotspot worth lowering (codec runs once at byte-close, ≤ms); the train-time CPU hotspot is the frozen-scorer forward, not codec — out of scope per "Native eval-time runtime discipline" | `codec.py`, `driver.py:2070-2089` | — | — |

**MICRO top-3:** μ1 mixed-FP4-packing (Δrate −0.022, the rate-maker; NOT wired today) · μ2 per-channel+byte-maps (the FP4 d_seg-hold enabler) · μ3 eval-roundtrip op-order match (training-signal tightening). All post-verdict. The decoder blob (96% of bytes, near-flat int8 entropy) is the *entire* micro-rate story — only true sub-8-bit packing moves it.

## MACRO audit results — PENDING (agent running)
## MACRO audit results (2026-06-15) — resolves the binding question
**THE BINDING FINDING:** the d_seg=0.00359 plateau is NEITHER a clean capacity wall NOR a cheap
optimization fix — it's a **genuine slow power-law descent**. Fit (ep≥311): **d_seg ≈ 0.0367·ep^(−0.351)**,
still monotonically descending at ep789 (slope −1.6e-6/ep). Extrapolating to basin d_seg=0.0006 needs
**~122,000 ep (153× the 800 budget)**. At an equal 29,650-ep budget, base_ch=20 predicts d_seg≈0.00099 — a
**1.6× capacity penalty** vs base_ch=36 (PR95's 0.000612), NOT a wall. ⇒ **optimization-limited in
direction, but too slow for budget-extension alone — the macro levers must (a) ACCELERATE the descent or
(b) BUY the 1.6× with cheap rate.**

**Reachability matrix (the sub-0.15 arithmetic):** d_seg=0.0006 + int8 76.6KB = **S 0.156 (MISS)** · +FP4
~42KB = **S 0.133 (CLEARS)** · +FiLM-pose (d_pose→2.9e-5) +FP4 = **S 0.105**.

| id | type | finding | impact | effort |
|---|---|---|---|---|
| Φ1 | STRATEGY | **winning move = SMALL basis + FP4 rate, NOT base_ch=20 budget-extension.** The 1.6× capacity penalty is bought back entirely by halving the rate term (the small basis is what collapses rate → crosses sub-0.15) | ΔS 0.455→0.105–0.156 | high (re-train) |
| Φ2 | OPP (=μ1/②) | FP4 decoder codec — the sub-0.15 **maker** (S 0.156→0.133) | Δrate −0.023 | med (FP4-QAT + d_seg-hold) |
| **Φ3** | **OPP — NOVEL, highest-EV d_seg lever** | **d_seg-aware capacity REALLOCATION.** Taper `[20,20,20,15,11,10,10]` puts **~81% of params at ≤24×32** (low-freq) and only **1.75% (1,461 params) at 384×512** where argmax-flips physically live. The arch is a **low-freq memorizer matching high-freq SegNet boundaries.** Shift ~5-10K params from the 33% stem Linear (27.8K) → a wider/deeper 192×256→384×512 refine head at FIXED param count | est Δd_seg −0.0005 to −0.0015; could turn the 153×-budget into a **2-4× reach** | low-med (thin decoder subclass; base_channels already threaded) |
| Φ4 | STRATEGY (kill) | **mask-grammar storage is DEAD** for d_seg (524KB seg-alone vs 177KB whole; 95% of flips are 1-px salt-and-pepper, unrepairable by sidecar — LeverD/mdl_contour both Δscore=0). FiLM-pose is the only winning store. Open: T1 cross-pair latent dedup (rate, n600-regime) | mask ≈0; T1 rate-only | — |
| Φ5 | STRATEGY (defer) | basis alternatives (Cool-Chic/C3/VQ/Mamba-Z7/Dreamer-Z8) ALL pre-measurement prototypes (deferred/crashed/NaN), **none has a contest-scale d_seg row** → means-not-ends; lower priority than the HNeRV already past the gate | unknown | very high |

## BRIDGE — cross-scale synthesis (the unifying object)
**All three bands converge on ONE object: the d_seg-sensitivity map over (decoder tensors × spatial
frequency × resolution), centered on the ~192×256 boundary band.** d_seg = argmax-flip at the SegNet
decision boundaries, decided at ~192×256 (the stride-2 stem). Every band's top lever is a different
actuator on the SAME map:
- **MICRO = put BITS where d_seg lives** — FP4 interior (HF, d_seg-blind) + int8 boundary heads; per-channel; byte-maps; and fix the gradient *pixels* at the boundary (μ3 op-order).
- **MESO = put GRADIENT where d_seg lives + sustain it through the descent** — CE-coarse→sc-refine, margin-weight on the boundary band, un-lag the EMA (worst in the refinement tail), renorm the margin (stop the compounding decay), Lever-4 sensitivity-EMA.
- **MACRO = put CAPACITY where d_seg lives** — the taper reallocation to the 192×256 refine band (Φ3) + buy the residual capacity gap with cheap FP4 rate (Φ2).

**The keystone coupling (μ4):** d_seg is *blind* to HF detail above ~192×256 (macro scorer property) → FP4
quant noise (HF, micro) is **safe** → the −0.022 rate win (μ1) is de-risked *because of* a macro property.
**The deepest coupling:** μ2 (FP4 bit-allocation) + M8 (Lever-4 sensitivity-EMA) + Φ3 (capacity allocation)
are **three views of the same d_seg-sensitivity map** — allocate both bits AND capacity to the tensors/band
that carry d_seg. **The descent couplings:** the EMA mismatch (①) + margin non-renorm (③) both DAMPEN the
late refinement gradient — exactly when the power-law (Φ-binding) most needs sustained drive; fixing them
*accelerates the descent*, not just "free absolute d_seg."

## FINAL EV-RANKED MICRO→MACRO MAP (the sub-0.15 program)
The binding answer: **base_ch=20 d_seg is optimization-limited-in-direction but power-law-slow; the win is
NOT to abandon the small basis (it collapses the rate term) but to ACCELERATE the descent (taper + meso
fixes) and BUY the residual 1.6× with FP4 rate.** Full stack predicts **S ≈ 0.105** (well under 0.15) IF
d_seg reaches basin. Ranked by EV (Δscore × likelihood / effort):
1. **FP4 mixed-precision packing** (Φ2/μ1/②) — Δrate −0.022 MEASURED, the maker (0.156→0.134), de-risked by μ4. *Not wired today* (variable-level codec only shrinks the int8 alphabet). **THE rate lever.**
2. **d_seg-aware taper reallocation** (Φ3) — NOVEL accelerator; capacity → the 192×256 band; could turn 153×-budget into 2-4×. **THE d_seg accelerator** (most under-explored).
3. **EMA window fix** (①/M1) — free + un-stalls the descent tail (worst at stage 8). Trivial. **THE free win.**
4. **Keep FiLM-pose** (live in stack arms) — pose 0.045→0.017, −0.028.
5. **Muon LR-floor (M3) + eval_every scaling (M2)** — clean correctness bugs; fold into the scaled run.
6. **margin-renorm (③/M7) + eval-roundtrip op-order (μ3)** — gradient-tightening; accelerate the tail.
7. **per-channel scales + byte-maps (μ2)** — the FP4 d_seg-hold enabler.
8. **DEFER:** mask-storage (Φ4, killed on rate), basis alternatives (Φ5, pre-measurement), T1 latent (n600).

**Sequencing:** the live from0 A/B (lever-refinement = an accelerator, item 6-class) finishes first →
verdict on whether the lever helps. THEN the **scaled sub-0.15 campaign** = base_ch=20 + Φ2(FP4) +
Φ3(taper) + FiLM-pose + all the meso/micro fixes batched, byte-closed → exact CPU+CUDA eval. That campaign
is the pointer-mover; everything here is post-verdict (mid-A/B changes break apples-to-apples).

## Cross-refs

## Cross-refs
[[l235-levers-break-dseg-plateau-plus-orphan-contention]] · CLAUDE.md "canonical leaderboard
binding-depth discipline" L14–L32 (the PR95 rate stack) · "THE GOAL — SUB-0.15" · the from0 A/B
(`experiments/results/from0_ab_v2_n96/`).
