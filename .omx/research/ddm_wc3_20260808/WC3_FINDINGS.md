# ddm_wc3 — full forward+backward step profile of the mx1 vehicle (operator saturation grant)

Tags: [no-triality] [p0-ledger-ok]. Axis: **[macOS-Metal wall-clock instrument]**, score_claim=false.
Instrument == vehicle (imports the trainer's own functions; cumulative-graph differencing +
per-depth value_and_grad; repo head d29df564f169; commit f9ab8fb399). Operator steers executed:
"full profiling of the forward and backward in all aspects" + "event driven rather than hard coded".

## 1. The measured anatomy (fp32 reference, one 4-pair chunk, reps=5)

| stage | fwd (s) | bwd (s) | share of chunk |
|---|---:|---:|---:|
| S1 quantize+cast+update | 0.001 | 0.001 | **0.3%** |
| S2 renderer fwd/bwd | 0.127 | **0.442** | **73.6%** |
| S3 R roundtrip | 0.009 | ~0 | ~0% |
| S4 SegNet fwd/bwd | 0.062 | 0.124 | 23.9% |
| S5 loss | ~0 | 0.022 | 2.7% |
| optimizer.update (per STEP) | — | — | 1.1 ms |

Full chunk vag 0.775 s → projected n32 graph time **6.20 s/step** vs bench-measured **10.44** →
**~4.2 s/step (~40%) is OUTSIDE the loss graph.**

## 2. The three verdicts

1. **Renderer BACKWARD alone = 57% of graph time** (0.442 s of 0.775; bwd/fwd ratio 3.5×).
   SegNet's backward is only 0.124 s — already served by the ACTIVE ~17× custom grouped-backward
   kernel (engagement receipt `"active": true, "reason": "env_set_and_metal_backend_available"`
   in run.log; the adapter defaults `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` at
   mlx_scorer_adapters.py:1213). The RENDERER (w96 4-dilated-block token→RGB) has NO custom
   backward — its 3.5× bwd/fwd ratio is the #903 upsample-VJP-scatter genus signature. **This is
   the named next kernel target** (routed to wc2's #478-fit assessment; a renderer-conv backward
   kernel is the largest remaining per-step compute lever).
2. **The quantize-hoist candidate is DEAD, measured before built:** fake_quantize_parameter_tree
   per-chunk costs 0.3% — the "8×/step redundant quantize" suspect was worth nothing. The profile
   paid for itself here (naive-first-pass law: measure the element before building its cure).
3. **The 4.2 s/step non-graph gap has a named mechanism at source:** per chunk, EVERY step, the
   trainer does fresh `mx.array` conversions of IDENTICAL data (`_mlx_token_chunk`) + `gc.collect()`
   + **`_clear_mlx_cache(mx)`** — clearing the MLX Metal buffer cache 8×/step (n32) / 30×/step
   (n120) forces cold buffer re-allocation for every subsequent tensor. OOM-era conservatism
   (#205 trauma) costing ~40% of wall-clock at ~1% memory utilization — exactly the operator's
   "not fully leveraging the memory" steer, located.

## 3. The fp16 instrument-disagreement (honest flag)

The profiler's fp16 pass measured the full chunk SLOWER (1.003 s vs 0.775 fp32) while the bench's
REAL fp16 steps are 1.226× FASTER (8.520 vs 10.441 s/step). The per-stage fp16 attribution also
shifts mass oddly (S5 bwd 0.393 s). Reading: the stage-boundary `mx.eval` barriers distort fp16
kernel scheduling/fusion across my cut points — the profiler's fp16 DECOMPOSITION is
INSTRUMENT-scoped, not a property of the real step. The bench (whole real steps) remains the
authority for composed variants; the fp32 anatomy is the trustworthy decomposition.

## 4. Levers built (commit f9ab8fb399, flag-gated, math byte-identical)

- `--microbatch-hygiene {per-chunk,per-step,off}` (default per-chunk = prior behavior): moves the
  gc+cache-clear cadence from per-chunk to per-step or off. Pure allocator/gc cadence by code
  inspection (no math touched).
- `--microbatch-chunk-cache` (default off): pre-materialize all chunk mx.arrays once before the
  step loop (chunks are static across steps). Memory cost trivially inside the 116 GiB ceiling;
  the bench mem-probe + fire-guard fail-close if wrong.
- Bench variants added: `hygiene-step` · `chunk-cache` · `saturated` (fp16 + per-step hygiene +
  chunk-cache). Fired as bench round 2 (bench_run2, baseline re-anchored in-run).

## 4b. Bench ROUND 2 results (n32, 5 steps, same ckpt-4500 resume) + review round-1 corrections

| variant | s/step | vs baseline | d_seg sanity |
|---|---:|---:|---|
| baseline (re-anchor) | 10.421 | 1.000× | 0.0010423660 (schedule B) |
| hygiene-step | 9.386 | **1.110×** | 0.0010426839 (= round-1 baseline, schedule A) |
| chunk-cache | 10.224 | 1.019× | 0.0010426839 (schedule A) |
| **saturated** (fp16+hyg+cache) | **7.659** | **1.361×** | 0.0010417302 (= round-1 fp16 exactly) |

**Adversarial round-1 findings applied (operator-ordered review):**
- **F1 (instrument):** round-2 baseline's d_seg equals round-1 COMPILE's value at full precision,
  with a DISTINCT result file (sha abb9cd3a ≠ abca47df) — the fp32 GPU sanity metric has a
  TWO-POINT schedule-dependent support (Δ≈3.2e-7, the L70 MLX-GPU bit-identity wall). Honest
  downgrade of §4's claim: the levers show "no numeric deviation beyond the run-to-run schedule
  spread," NOT bench-verified byte-identity; identity rests on code inspection (no math touched).
- **F2 (mechanism attribution):** the 4.2 s/step non-graph gap is NOT mostly marshaling as §2.3
  hypothesized — measured split: hygiene cadence ≈ 1.03 s/step (hygiene-step win), chunk
  marshaling ≈ 0.20 s/step (chunk-cache win); the remaining ≈ 3.0 s/step is probe sampling +
  grad-accum tree-adds + per-chunk eval barriers, UNATTRIBUTED. Next profiler increment if it
  matters after the composed win.
- **F3 (ledger integrity, debt):** `wc1_bench_receipts.jsonl` is written via whole-file
  `write_jsonl_atomic` per invocation — round-1 rows were EVICTED by round 2 (they survive in
  bench_run/run.log + this memo). The "receipts JSONL" is per-invocation, not cumulative;
  append-mode fix owed at next bench touch.
- **Composed verdict:** saturated 7.659 s/step = 1.361× vs same-session baseline; n120 projection
  re-measures at ticket-seal (the mem-probe runs real steps). The seal is HELD for the recursive
  adversarial review (3 clean passes) + the gc21 convocation per operator order.

## 5. Event-driven consequence (operator steer #2, binding on M1)

The n120 receiver ticket does NOT hardcode a 3250-step horizon. 3250 (wc1's DERIVED rec) becomes
the FORECAST/budget input; the STOP rule is event-driven: plateau detector on the dense eval
cadence (tp1 plateau-typed exits / DDMEventContinuationV1 #688 doctrine) + wall-clock cap +
resumable-P0 checkpoints. Same law applied to resource hygiene: the hygiene MODE should be
derived-at-consumption from measured memory headroom (dy1 scope-law), not a hardcoded reflex —
the flags land the mechanism; the M1 ticket sets them from bench-round-2 receipts.

## 6. Follow-on disposition (no orphans)

| item | state |
|---|---|
| bench round 2 (baseline·hygiene-step·chunk-cache·saturated) | FIRED (bench_run2) |
| renderer-backward custom Metal kernel (57% target) | ROUTED → wc2 consumption + #478-fit; fire-order after bench-round-2 read |
| quantize-hoist | DEAD (measured 0.3%) — do not build |
| fp16 profile decomposition | INSTRUMENT-scoped flag recorded (§3); bench = authority |
| 4 unreceipted wc1 variants (ram-cache · derived-microbatch · concurrent-cpu-verdict · ane-verdict) | NOTED for wc2 harvest (bench executed only first-5 both runs) |
| M1 event-driven ticket composition | NEXT at bench-round-2 landing |
