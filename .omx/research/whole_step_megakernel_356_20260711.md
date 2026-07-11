# Whole-step MEGAKERNEL (#356) — MEASURED NO-GO on both legs; the wall is now a law

**Agent:** main (Fable, operator-directed take-over after 3 codex-rescue wrapper deaths) · **Date:** 2026-07-11
· **Axis:** all numbers `[macOS-CPU/GPU advisory · NON-PROMOTABLE]`, $0, bounded GPU bench on an idle machine
(no live trainer; governor respected; NO launches). **Pointer 0.19108282 [contest-CPU] UNMOVED** — a speed
lever is MEANS; this memo delivers the verdict the #356 spec pre-registered as the honest alternative:
*"If you CANNOT achieve bit-identity, that is an honest NO-GO finding — report it; do NOT ship a
numerics-changing kernel as a 'speedup.'"*

**STORES CONSULTED:** #355 `v7_compute_exploitation_audit_20260708.md` · #306
`per_lever_compute_audit_20260705.md` (where step time GOES) · #410
`microbatch_bit_identity_smoke_n600_20260710.md` · L70 (fused-R bit-identity) · the in-code 2026-07-03
R-compile fp-contraction measurement (`train_witness_realized_through_R_mlx.py` `set_mx_compile_r`) ·
`tac.local_acceleration.mlx_compile_step` (prior art: `compile_loss_and_grad` +
`assert_compile_bit_identical` + the representative trunk — built, never wired = an orphaned lever
recovered by this task) · `tac.mlx_safe_compile` (#252 certify-then-compile regions).

---

## 1. The design space, pre-narrowed by prior MEASURED findings

- **Whole-step `mx.compile` was already half-dead:** the 2026-07-03 R-op measurement (in-code, fail-closed
  `--mx-compile` gate) shows compile re-fuses mul+add→fma, deltas up to ~4.8e-3 across the uint8 round
  boundary → **flips d_seg argmax pixels**. The #252 answer was certify-then-compile REGIONS (today:
  `hosc_activation` only).
- **The remaining candidate:** compile the trunk+loss closure with R as the fused Metal kernel (opaque to
  compile → the known contraction site removed) + grouped-backward custom VJP preserved. Whether the
  REST of the graph compiles at exactly 0.0 was the open empirical question. Answered below.

## 2. The decisive smoke (MEASURED, `experiments/megakernel_compile_bit_identity_smoke_356.py`)

Representative witness d_seg closure (`mlx_compile_step.build_representative_dseg_trunk` — Linear/FiLM/
relu/softmax/palette-matmul/sigmoid + CE + finite-diff term; the REAL closure's op-kinds are a superset →
MORE contraction sites, so a FAIL here transfers). Eager vs `mx.compile`, exact deltas fwd+bwd fp32:

| device | P | grad max\|Δ\| | loss \|Δ\| | compiled re-run Δ | eager ms/step | compiled ms/step | speedup |
|---|---|---|---|---|---|---|---|
| CPU | 4,096 | **2.27e-5** | 4.8e-7 | 0.0 | 3.33 | 4.22 | **0.79× (slower)** |
| CPU | 196,608 | **2.31e-7** | 0.0 | 0.0 | 192.7 | 231.8 | **0.83× (slower)** |
| GPU | 4,096 | **1.30e-5** | 2.1e-6 | 0.0 | 1.24 | 1.11 | 1.12× |
| GPU | 196,608 | **2.00e-6** | 0.0 | 0.0 | 21.0 | 17.3 | **1.21×** |

## 3. VERDICT: NO-GO — two independent kill shots

1. **NOT bit-identical anywhere** (grad Δ 2.3e-7 … 2.3e-5; deterministic-but-DIFFERENT — re-run of the
   compiled graph is byte-stable 0.0, but it computes different numerics than eager = fp
   reorder/contraction, the #410 micro-batch class). A lever that changes step numerics forks the
   trajectory → cannot ride a score-faithful pointer lineage or serve as an A/B-neutral speedup.
2. **The prize is small anyway**: 1.12–1.21× on the closure ≈ **~5% end-to-end** (per #306 the step is
   compute-bound in the SegNet convs; only elementwise glue fuses) and **SLOWER on CPU**. Even a
   bit-identical version would be marginal.

**Law registered:** `witness_fp_reorder_transform_bit_identity_wall_v1`
(`src/tac/canonical_equations/fp_reorder_transform_bit_identity_wall_20260711.py`) — 3 anchors
(R-compile 2026-07-03 · micro-batch 2026-07-10 · this closure smoke 2026-07-11), ONE mechanism:
**fp-reorder-permitting transforms are deterministic-but-different; only explicit-order kernels +
gradient-free constant caches buy speed WITH bit-identity.**

**Verdict scope (ladder):** FORMULATION — mx.compile whole-step fusion on this graph class at MLX fp32.
PARADIGM INTACT: per-chain explicit-order custom Metal kernels (#252 standing) + fixed-order batched
reductions (#348 family) + per-backend re-measurement on the CUDA port (#438) all remain OPEN.

## 4. What the run actually gets (the surviving speed levers — VERIFIED ON)

The honest completion of "make the multi-day run fast" is that the #432 V9·CGauge argv **already
carries every speed lever that passes the wall** (verified by compiling
`spec_v9_cgauge.compile_v9_cgauge_432_launch_config()` today):
- `--fused-r-kernel` — explicit-order Metal R (bit-identical fwd, fixed-order VJP, L70 0/28, ~8% faster).
- `--cache-gt-skeleton` — gradient-free epoch-invariant constant (EXACTLY bit-identical, n64 A/B;
  ~half the clDice recompute; the #306 drift was already cured into the v752 baseline).
- grouped-conv backward (~17×) — the default custom kernel.
There is no additional bit-identical whole-step fusion available at MLX fp32; the wall is measured.

## 5. What I did NOT do (spec honesty)

- Did NOT wire a `--whole-step-megakernel` flag — shipping a numerics-changing "speedup" lever would be
  the NO-FAKE the spec forbids; the DSL leg is **N/A-with-reason** (no lever ships).
- Did NOT measure the REAL trainer closure under compile — the representative closure (op-kind subset)
  already fails 0.0 and the real-graph R anchor independently confirms real-graph contraction; the
  burden is on a future GO, not on this NO-GO. (A real-closure certificate harness exists if ever
  needed: `mlx_compile_step.assert_compile_bit_identical`.)
- Did NOT touch the #432 config, any live run, or launch anything (CONTAINMENT).

## 6. Consequence for the #432 gate

The megakernel blocker resolves BY VERDICT: there is no bit-identical whole-step kernel to wait for at
MLX fp32, and the arm already runs at the measured speed frontier of the surviving family. #432's
remaining hold = the #441 review revisions (recorded at the config memo) + **operator-GO**.
#435 (codex round-2 megakernel iteration) is MOOT — its premise (iterate the kernel) has no object.

## Triality
- **DAG:** FEED-356 (appended this turn).
- **DSL:** N/A-with-reason — NO lever ships (a numerics-changing flag would be the fake); the two
  surviving levers are already DSL-held (`CacheGtSkeleton`, fused-R) and ON in the #432 argv.
- **equations:** `witness_fp_reorder_transform_bit_identity_wall_v1` REGISTERED (3 anchors).
