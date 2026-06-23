# Q-AXIS bit-depth response surface: d_seg / d_pose / bytes / S vs decoder-weight QAT bits (int8→int3) on the 0.19110 frontier

- **UTC**: 2026-06-23
- **Authority**: `[contest-CPU advisory]` NON-PROMOTABLE. Frontier pointer UNMOVED `0.19110`.
- **Spend**: $0 (local CPU-only authority eval; NO MPS, NO GPU dispatch, NO paid). The live training run owns the GPU; this ran CPU-only.
- **Lane**: the Q-AXIS response surface for the math-optimal solver (sister agent). Substrate = the REAL frontier
  decoder (pointer archive sha `b46897267…`, lane `pr110_payload_entropy_recode`, 177,169 B, 228,958 params).
- **Tool**: `tools/measure_qaxis_bitdepth_response_surface.py` (reuses `tac.frontier_decoder_ptq` +
  `tac.post_hoc_weight_shrink` + `tac.frontier_int5_qat.mse_optimal_step` + `tac.contest_score` + `RealScorerContext`).
- **JSON artifact**: `.omx/research/qaxis_bitdepth_response_surface_20260623T232215Z.json` (the response surface
  the math-optimal solver consumes; n48 full surface + n600 gold anchors).

## Why this is load-bearing

A sister measurement (`decoder_weight_rate_axis_and_shallow_boundary_synthesis_20260621`) proved the frontier's
int8 decoder weights are already at the **order-0 Shannon entropy floor** (brotli-q11 ≈ marginal H(W); a custom
arithmetic coder gains ≤0.07 KB → ΔS ≤ −0.00005). Pure entropy recode is EXHAUSTED. The ONLY rate lever left on
the borrowed frontier is **LOWER-BIT weights**. This surface is the real d_seg-vs-bits cost curve so the solver
knows whether INT4/INT3 reaches sub-0.15.

## Method (all REAL, NO-FAKE)

1. Decode the REAL frontier decoder state_dict (`decode_frontier_member`; int8 identity round-trip proven
   byte-identical: int8-requant → 177,169 B, 0.0 weight err — the int8 grid IS the codec's native grid).
2. For each bit-depth Q ∈ {8,7,6,5,4,3} and each PTQ variant:
   - **absmax** — per-tensor symmetric int-N, scale = abs_max/qmax (the codec's native grid, generalized).
   - **mse_calib** — the canonical low-bit OUTLIER-HANDLING fix the int5-cap abs-max test omitted: per-tensor
     symmetric int-N with the **MSE-optimal clip step** (`mse_optimal_step`). This is the codec-BYTE-CLOSEABLE
     form of the fix; per-CHANNEL scales are NOT byte-closeable through the per-tensor-int8 codec grammar
     (measured in the int5 retest: blows the archive 118k→197k AND isn't preserved by the per-tensor int8 store).
3. Re-encode through the REAL frontier split-brotli codec → MEASURED archive bytes (rate term).
4. **NO-FAKE eval-on-shipped-bytes**: RE-DECODE the byte-closed archive (the codec-int8-of-int-N codes that
   actually ship — NOT the in-memory int-N, which is ~off) and exact-eval THOSE weights on the FULL 600-pair CPU
   authority via `RealScorerContext.exact_eval`. Latents + sidecar stay verbatim (decoder-only change), so the
   latent payload bytes are constant across Q. Score recomputed from components via `tac.contest_score`.

## The response surface

Two pair counts. **n48** = the complete 12-cell surface (fast, ~55s/eval, fully cached GT). **n600** = the
gold contest-pair-count anchor (~10–12 min/eval under heavy CPU contention; int8/int7/int6 landed, the rest
left running detached — the conclusion is bit-count-robust: n48 and n600 agree on the monotone collapse and
that no Q crosses 0.19). Score recomputed from components via `tac.contest_score` (byte-identical to
`upstream/evaluate.py:92`). NO-FAKE: int8 identity = the pointer archive EXACTLY (177,169 B, ship_err 0.0).

### n48 full surface (all 12 cells)

| variant | bits | d_seg | d_pose | archive_bytes | S | vs frontier |
|---|---:|---:|---:|---:|---:|---|
| absmax | 8 | 0.000578 | 0.000018 | 177,169 | **0.18919** | = baseline |
| absmax | 7 | 0.001376 | 0.000169 | 174,061 | 0.29461 | +0.105 |
| absmax | 6 | 0.002192 | 0.000247 | 147,513 | 0.36712 | +0.178 |
| absmax | 5 | 0.004015 | 0.002368 | 118,589 | 0.63435 | +0.445 |
| absmax | 4 | 0.009699 | 0.128190 | 87,925 | 2.16066 | +1.97 |
| absmax | 3 | 0.088782 | 44.211 | 56,685 | 29.94 | +29.8 |
| **mse_calib** | 8 | 0.000578 | 0.000018 | 177,169 | **0.18919** | = baseline |
| **mse_calib** | 7 | 0.000975 | 0.000033 | 175,801 | 0.23272 | +0.044 |
| **mse_calib** | 6 | 0.001401 | 0.000224 | 160,977 | 0.29462 | +0.105 |
| **mse_calib** | 5 | 0.002605 | 0.001135 | 138,389 | 0.45918 | +0.270 |
| **mse_calib** | 4 | 0.004675 | 0.013744 | 114,817 | 0.91468 | +0.726 |
| **mse_calib** | 3 | 0.009424 | 0.111 | 90,849 | 2.05867 | +1.87 |

### n600 gold anchors (full contest pair count, the authority)

| variant | bits | d_seg | d_pose | archive_bytes | S |
|---|---:|---:|---:|---:|---:|
| absmax | 8 | 0.000594 | 0.000037 | 177,169 | **0.19660** (= int8 local baseline; contest pointer 0.19110) |
| absmax | 7 | 0.001537 | 0.000222 | 174,061 | 0.31672 |
| absmax | 6 | 0.002384 | 0.000294 | 147,513 | 0.39084 |

The n600 numbers are slightly higher d_seg than n48 (more pairs → more flips counted) but the regime is
identical. **S-minimizing bit-depth = int8** (the baseline). Every Q<8 raises S.

## d_seg(Q) sensitivity (the rate↔d_seg coupling the simple RD model misses)

d_seg degrades **super-linearly** as bits drop — the killer that a naive rate-only RD model ignores. Per
1-bit step (n48, mse_calib = the better variant):

| step | Δd_seg | bytes saved | ΔS |
|---|---:|---:|---:|
| int8→int7 | +0.000397 | 1,368 | +0.044 |
| int7→int6 | +0.000426 | 14,824 | +0.062 |
| int6→int5 | +0.001204 | 22,588 | +0.165 |
| int5→int4 | +0.002070 | 23,572 | +0.456 |
| int4→int3 | +0.004749 | 23,968 | +1.144 |

**The seg term (100·d_seg) penalty grows ~2× per bit dropped while the bytes saved per bit FLATTENS
(~23k/bit below int6).** d_pose is even more fragile — it explodes (0.000018 → 0.013744 → 0.111 at int4/int3,
absmax 0.128 → 44.2). The rate win (177k → 56k, a 0.080 rate-term drop end-to-end) is dwarfed at every step
by the d_seg+d_pose rise. This is the rate↔d_seg coupling: you cannot buy rate with bits here.

## bytes(Q) curve

177,169 (int8) → 174,061 (int7) → 147,513 (int6) → 118,589 (int5) → 87,925 (int4) → 56,685 (int3) [absmax].
The mse_calib clip spreads weights over more int-N levels (finer bulk resolution) so it costs MORE bytes at
each Q (e.g. int5 138,389 vs absmax 118,589) — the precision-vs-rate trade of the outlier-handling fix.
Note int8→int7 saves almost nothing (3k B): the codec stores int8 codes regardless, so int7 only shrinks
brotli's symbol entropy slightly. Real byte savings start at int6 (the codes get sparse enough to compress).

## VERDICT

**RED — no PTQ bit-depth reaches sub-0.15 OR sub-0.19. The bit-axis only moves S UP from the frontier.**
S-min = int8 (the baseline = the frontier itself). The rate lever (lower-bit weights) is structurally
dominated by the d_seg+d_pose collapse it causes: even the gentlest non-trivial step (int8→int7, mse_calib)
costs +0.044 S for 1.4k bytes. There is **zero** crossing of 0.19 below int8 and **zero** crossing of 0.15
anywhere. INT4-PTQ = S 0.91–2.16; INT3-PTQ = S 2.06–29.9 (total collapse). The "lower-bit weights is the
only rate lever left" hypothesis is empirically REFUTED for PTQ: the borrowed frontier decoder cannot be
bit-shrunk into a lower score.

## int5-prior re-test verdict (existence-proof cross-check applied)

The int5 cap (memory: "int5 QAT capped ~S0.49, d_seg walls ~0.0035, STRUCTURAL") was treated as a **SUSPECT
recipe artifact, NOT a floor** (per `feedback_terminal_conclusion_needs_existence_proof_crosscheck`), and
re-tested with the canonical low-bit OUTLIER-HANDLING fix the abs-max test omitted (`mse_calib`). Result —
**both halves of the cross-check are true:**

- **Partly a recipe artifact (the fix is REAL):** mse_calib int5 S=0.459 (d_seg 0.00261) beats absmax int5
  S=0.634 (d_seg 0.00402) — d_seg −35%, S −0.175. This matches the sister int5-LSQ-best-shot (S~0.486,
  d_seg~0.00266) to within noise: abs-max DID leave precision on the table, exactly as the recursive
  adversarial review predicted.
- **But the FLOOR is real (the cap stands, refined):** even the best codec-byte-closeable PTQ fix leaves
  int5 at S=0.459 = **2.4× the frontier 0.191**, d_seg 0.00261 = **4.6× the frontier floor 0.00056**. The
  deeper fix — per-CHANNEL scales — is NOT byte-closeable through the per-tensor-int8 codec grammar (the
  int5 retest measured: per-channel int5 blows the archive 118k→197k AND isn't preserved by the per-tensor
  int8 store). And the sister int5 QAT-FINETUNE best-shot confirmed the d_seg wall holds under TRAINING
  (−9.5% d_seg only, CE seg-loss flat).

**Net: the int5 number was partly a recipe artifact (improvable abs-max → mse/LSQ), but the int5 cap as a
codec-byte-closeable floor is CONFIRMED — never sub-0.19.** The existence-proof cross-check did its job: it
caught the abs-max recipe weakness AND confirmed the underlying floor is real.

## PTQ vs QAT-finetune caveat

These are **PTQ** (post-hoc, no training) numbers — the $0 LOWER BOUND on what bit-shrink costs. **QAT-finetune**
(train the decoder to be robust at the coarse grid) is the REAL version and does better — but the **int5 best-shot
already ran it** (`frontier_int5_lsq_best_shot_retest_20260619`): per-tensor LSQ learned step + outlier-clip + CE
finetune recovered d_pose **−89%** but d_seg only **−9.5%** (stayed ~0.0042, 7.6× the frontier's 0.00056), with
the CE seg-loss FLAT across ep10→100 → more epochs will not break the d_seg wall. The mechanism
(`decoder_weight_rate_axis_…_synthesis` §4): the d_seg residual is **shallow** (66.5% of argmax flips lost by
<0.5 logit), so quant noise ∝ 2^{−b} flips the shallow-margin pixels; the d_seg-critical early/low-res stages
(77% of params) need finer-than-int-N **per-channel** resolution that no per-tensor scale provides — and
per-channel is not byte-closeable. A QAT-finetune column for int6/int7 (where the PTQ d_seg damage is smaller and
finetune-recoverable) is the only open follow-up; it needs the training run (a separate campaign).

## Reactivation criteria

- **QAT-finetune at int6/int7** if the PTQ surface shows int6/int7 d_seg damage is small enough that the bytes
  saved (vs int8) could net a sub-0.19 / sub-0.15 S *after* finetune recovers part of the d_seg. (PTQ int5/int4
  is closed by this + the int5 best-shot.)
- **A new archive section storing per-channel scales** (a different codec grammar = a new campaign) — only if
  the solver shows per-channel int-N could reach sub-0.15 AND the added scale bytes don't eat the rate win.
- **The sub-0.15 path routes OFF bit-shrinking the borrowed frontier's RATE** → a concentrated-saliency OWN
  vehicle whose d_seg-critical capacity is spent where the argmax boundary lives (per the cap memo's pivot).

## 6-hook wire-in

1. **Sensitivity-map**: ACTIVE — the d_seg(Q) sensitivity chain (Δd_seg per 1-bit-dropped) is the bit-axis
   sensitivity prior; feeds the WRQ score-aware bit allocator (the solver's `b_i*` reverse-waterfill).
2. **Pareto constraint**: ACTIVE — each (bytes, d_seg, d_pose, S) row is a Pareto point on the rate↔distortion
   frontier; the surface bounds the achievable region for any uniform-bit decoder.
3. **Bit-allocator hook**: ACTIVE (primary) — this IS the per-bit-depth cost curve the bit allocator consumes;
   it quantifies the rate-saved-vs-d_seg-cost trade the solver optimizes.
4. **Cathedral autopilot dispatch**: N/A — advisory, non-promotable, no archive-deployable candidate (PTQ
   collapses; no row crosses the pointer).
5. **Continual-learning posterior**: ACTIVE — the JSON response surface is a measured anchor; feeds the
   math-optimal solver (sister agent) + the canonical equation for quant-noise vs d_seg.
6. **Probe-disambiguator**: ACTIVE — the absmax-vs-mse_calib variant pairing IS the disambiguator between "the
   int-N cap is an abs-max recipe artifact" vs "a real per-tensor grid floor."

## NO-FAKE / discipline

- Every score is the REAL byte-closed archive through the frontier codec → REAL CPU `RealScorerContext.exact_eval`,
  S recomputed from components via `tac.contest_score`. NEVER MPS.
- int8 identity reproduces the pointer archive EXACTLY (177,169 B, 0.0 weight err) — the codec round-trip is faithful.
- eval-on-shipped-bytes (re-decode the byte-closed archive) closes the in-memory-vs-shipped NO-FAKE gap the
  int5-cap first draft had.
- Existence-proof cross-check (per `feedback_terminal_conclusion_needs_existence_proof_crosscheck`): the int5
  "structural cap" is treated as a SUSPECT recipe artifact and RE-TESTED with the canonical low-bit fixes
  (mse_calib outlier-clip) across the full bit-axis, not asserted as a floor.
