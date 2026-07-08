# --micro-batch-pairs BIT-IDENTITY CRUX — MEASURED verdict (2026-07-08) [no-triality]

**Task (operator, crux-engineering):** make `--micro-batch-pairs` B>1 BIT-IDENTICAL to the
serial accumulation path *by construction*, dissolving the trajectory-A/B gate at the root, so
the 2–4× wall-clock lever admits as score-neutral without any A/B.

**VERDICT (MEASURED, not reasoned): the premise is FALSIFIED. Full bit-identity at any speedup
> 1× is IMPOSSIBLE on the real MLX scorer, because the dominant divergence enters at the
FROZEN-SCORER FORWARD KERNEL — upstream of any reduction — not at the reduction/accumulation
order the fix targeted.** The A/B gate is irreducible by reduction-order engineering. Surviving
speedup at bit-identity = **1.0×**. **MEANS — pointer contest-CPU 0.19110 UNMOVED; nothing here
moves it. Only a byte-closed n600 `upstream/evaluate.py` row does.**

## STORES CONSULTED
- `.omx/research/d15_micro_batch_routing_20260708.md` (D15 — per-pair routing forms already
  proven bit-EXACT; the residual named "the mean-over-B reorder is trajectory-affecting").
- `src/tac/boundary_math/levelset_micro_batch_loss.py` (the twin: `batched_realized_loss`,
  `single_realized_loss`, `_pair_loss_from_scored`, `_once_terms`).
- `experiments/train_levelset_witness_realized_through_R_mlx.py` — the serial accum loop
  (7831–7897): `accum = g0; accum += g1; ...; mean_grads = accum/nb`; the batched dispatch
  (7836–7869) with the `* _bn` / `/ nb` group weighting; scorer loaded `device="cpu"` under
  `temporary_mlx_device(args.mlx_device)` (2906–2907); live run `--mlx-device gpu --fused-r-kernel`.
- `src/tac/local_acceleration/mlx_batch_invariance.py` — the EXISTING scorer batch-invariance
  audit (thresholds: segnet 1e-3, posenet 1e-4, argmax 0 px) — i.e. the project already treats the
  real scorer as NON-bit-invariant batch-vs-single.
- T5 crucible `position_INCL_S4_rudin.md` item 3 ("mean-over-B reorder → trajectory-affecting");
  `position_INCL_S5_adversary.md` VETO #1 (bounded-recess A/B, not baseline-gated).
- L70 (`mlx_gpu_crossprocess_nondeterminism_v1`, #348) — the sister kernel-nondeterminism family.

## WHERE DIVERGENCE ENTERS — MEASURED (three components, decomposed)

Measured with `tools/micro_batch_bit_identity_probe.py` (real upstream adapter, K=4/8, 384×512).

### A. SCORER FORWARD kernel batch-dependence — the IRREDUCIBLE ROOT (dominant)
`segnet(f1_batch)[k]` is NOT bit-identical to `segnet(f1_batch[k:k+1])[0]`:

| device | segnet max\|Δlogit\| | argmax px flipped | posenet max\|Δ\| |
|---|---|---|---|
| **GPU** (live run) | **2.259e-2** | **11 / 196608** | **7.728e-3** |
| CPU | 7.105e-5 | 0 | 2.027e-6 |

This is a conv/matmul kernel-tiling property (GPU dominant; even CPU is 7e-5, not 0). It is
**upstream of any loss/grad reduction**: the per-pair `L_k` computed from the batched forward
already differs from the serial per-pair `L_k` before accumulation. On GPU it even flips 11 argmax
pixels — so not even the SegNet argmax (the d_seg quantity) is batch-invariant on GPU.

### B. REDUCTION / accumulation order — SECONDARY (only visible where the scorer is invariant)
Isolated with a batch-INVARIANT mock scorer (linear per-pixel/per-frame ⇒ its batched
forward+backward are exactly 0.0 batch-dependent, MEASURED). The twin's one-shot
`value_and_grad` over `L = mean_k L_k` accumulates the K per-pair grad contributions into the
SHARED witness params (`out_tex`/`in_proj`/`code`) in a graph-internal order that differs from
the serial explicit left-fold ⇒ grad **max\|Δ\| ≈ 1e-3…4e-3** on the tree, **HIDDEN by the
global-L2 ≈ 1e-8** metric the existing equivalence tests assert. Per-pair grads *extracted in
isolation* from the batched forward match serial to **3.7e-8** — so the ~4e-3 is purely the
all-K-cotangents-live batched backward reduction, i.e. the reduction ORDER.

**STRENGTHENING finding:** this reduction is itself **non-deterministic run-to-run** (even on CPU,
byte-identical inputs): the grad max\|Δ\| lands on different ULP boundaries across calls
(1/1024 vs 2/1024 vs 4/1024 on `out_tex`). You cannot "fixed-order match" a reduction whose order
is itself non-deterministic. Sister of the #348 MLX-GPU cross-process non-determinism family.

### SPEEDUP (scorer-forward microbench, K=8): where the win lives
ONE batched scorer forward over K vs K per-pair forwards: **GPU 1.56× / CPU 1.75×.** The entire
micro-batch win IS the batched scorer forward — the exact op that is not batch-invariant.

## WHY THE OPERATOR'S FIXES (a)/(b) DON'T DISSOLVE THE GATE
- Fix (a) "fixed-order sequential reduction over the B per-pair scalars" assumes the batched
  per-pair scalars equal serial's. MEASURED false on the real scorer (source A): the scalars differ
  before any reduction.
- Fix (b) "B separate vjp calls" fixes the reduction reorder (source B → 3.7e-8) but the batched
  FORWARD drift (source A) remains unless the forward is also per-pair — and a per-pair forward IS
  the serial path (1.0×). Bit-identity ⟺ per-pair scorer forward ⟺ no speedup.

## WHAT I MADE BIT-IDENTICAL
Nothing NEW can be made bit-identical on the real scorer without forfeiting the entire speedup —
that is the honest finding. What I delivered is the **reproducible decomposition** that proves it
and cleanly isolates the two sources, plus the classification that any future session runs:
- `src/tac/boundary_math/micro_batch_bit_identity_probe.py` — pure, testable: `measure_reduction_
  order_drift` (source B, batch-invariant mock, deterministic-bounded) + `classify_micro_batch_bit_
  identity` (the honest verdict from measured inputs) + recorded MEASURED anchors.
- `tools/micro_batch_bit_identity_probe.py` — CLI reproducing A (real scorer, cpu/gpu) + B +
  speedup → JSON verdict.
- `src/tac/tests/test_micro_batch_bit_identity_probe.py` — 20 tests (all green; twin regression 70
  green — the twin is UNCHANGED, B=1 byte-identical preserved).

The reduction-order fix (source B) is only worth building IF batch-invariant scorer kernels land;
`classify_*` returns `bit_identical_at_speedup_possible=True, surviving=1.56×` in that hypothetical,
so the probe is the readiness gate for that future path.

## SURVIVING SPEEDUP FRACTION (honest, labeled)
- **At bit-identity: 1.0×** (bit-identity requires the per-pair scorer forward = the serial path).
- **Non-bit-identical (gated): 1.56× GPU / 1.75× CPU** scorer-forward microbench at K=8. NOTE: this
  is the SCORER-FORWARD microbench, NOT the end-to-end n600 trainer speedup (which the D15 memo
  correctly leaves UNMEASURED — render + backward + opt overhead dilute it, and the #294 waterfill
  pins B=1 until a measured uncontended n600 curve exists).

## THE ONLY ADMISSION PATHS FOR THE 2–4× LEVER (the gate stands)
1. **Bounded n600 d_seg A/B** (S5 VETO #1): a SHORT n600 run to ~ep300–350, NOT the full baseline,
   measuring whether the ~2.3e-2 forward drift is d_seg-NEUTRAL over training. CPU argmax is
   invariant (0 px); GPU flips only 11/196608 px (0.006%) ⇒ plausibly d_seg-neutral — but MEASURE,
   do not assume (measurement-first). If neutral ⇒ admit; else clean v7.1.
2. **Batch-invariant scorer kernels** (custom Metal SegNet/PoseNet forward with fixed, batch-
   independent reductions — the "defeating nondeterminism" approach). Large MLX/Metal item (L52);
   the ONLY route to bit-identity-at-speedup. Out of scope here; registered as the future lever.

## TRIALITY / DAG
Tagged `[no-triality]` (apparatus/measurement, like the D15 memo). Trajectory leg = DAG FEED
appended. Equation leg: this is a MEASURED NEGATIVE in the `mlx_gpu_crossprocess_nondeterminism_v1`
(#348/L70) family; a dedicated `micro_batch_scorer_forward_batch_dependence_v1` equation is the
follow-up (the canonical_equations registry import needs scipy, unavailable in this probe venv).
