# ddm_dt1 (#903) — the TR1 determinism floor: MEASURED, MECHANISM ISOLATED, CURED

**Date:** 2026-08-03 · **Arm:** `ddm_dt1_determinism_floor` · **Evidence axis:** `[macOS-MLX/CPU apparatus]`
· `score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`. This memo makes **no** score
claim; it measures apparatus reproducibility and lands a cure.

---

## Answer first

**The #903 premise is CONFIRMED, and the mechanism is named: the R operator's UPSAMPLE VJP.**

`experiments/train_tr1_partition_renderer_mlx.py` on MLX-GPU is **not** run-to-run bit-reproducible at
fixed seed+config+inputs. **40 of 41** checkpoint arrays differ between two such runs (only `meta::epoch`
survives) — reproducing the shape of the original "28-29 of 30" observation. After **exactly one**
optimizer update, **26-28 of 41** arrays already differ, while the reported loss scalar was identical in
5 of 5 runs — i.e. **the scalar telemetry was hiding it.**

Bisected to a single op: **the bicubic/bilinear UPSAMPLE VJP inside `_apply_R`** (a scatter; its GPU
accumulation order varies). The DOWNSAMPLE VJP is clean, the R FORWARD is clean, MLX-CPU is clean, and
generic MLX GPU ops are clean. The drift is ~1 ULP in the gradient — but Adam's first update is
essentially `sign(g)`, so a 1-ULP gradient flip becomes a full `lr`-sized parameter step, then amplifies
chaotically.

**CURE LANDED AND MEASURED: `--deterministic-r`** (default OFF). It routes R through the repo's own
atomics-free fused Metal kernel — which was **already built, already documented as beating "the ~1-ULP
non-determinism floor of the prior `mx.vjp` scatter backward", and left default-OFF and unwired into
TR1** (the built-elsewhere-unwired class). With it: **41/41 arrays and 134/134 telemetry fields
bit-identical across 4 runs**, the forward is **bit-identical to the reference (max|Δ| = 0)**, and the R
grad is **~4.5× faster**. The cure is not expensive; it is free and faster.

**The floor, in lever-attribution units (S = 100·d_seg): range 4.4 – 21.0 S across 4 repeats** at the
measured control windows — see the scoping section, which is the part that decides what it invalidates.

---

## 1. Method + the positive control (why the verdict is admissible)

Harness: `tools/ddm_dt1_compare_run_determinism.py`. Compares two surfaces — every `.npz` key under a
run's `checkpoints/` (bit-level on the raw buffer, so 1 ULP is a DIFFER not a tolerant PASS) and every
numeric field of `telemetry.jsonl` (where `realized_gate_dseg_mean`, the lever-attribution unit, lives).

**Positive control is mandatory and runs before every real comparison** (`--self-check`, MEASURED PASS
5/5 fixtures):

| fixture | required verdict | result |
|---|---|---|
| byte-equal payload | `IDENTICAL` (must not false-fire) | PASS |
| one element moved by **1 ULP** | `DIFFER`, `max_ulp == 1` | PASS |
| a key present on only one side | `ASYMMETRIC` (not intersected away) | PASS |
| **empty comparison scope** | `VACUOUS` (**never** a bare pass) | PASS |
| telemetry identical / 1-ULP | `IDENTICAL` / `DIFFER` | PASS |

A self-check failure ABORTS (rc=3) rather than degrading to an untrusted comparison. Every verdict below
carries its **denominator** ("N of M compared"), per the vacuity rule.

Volatile fields (`t_wall`, pids, paths, wall-clocks) are excluded and **the exclusion list is printed**.
`t_wall` was found empirically: it was the *only* residual difference across three otherwise bit-identical
MLX-CPU runs (8 of 142 fields, all `t_wall`).

Control config (short, bounded, `$0`, ~7.5 s/window):
```
experiments/train_tr1_partition_renderer_mlx.py --variant lotto --num-pairs 6 --batch-pairs 3 \
  --epochs 4 --gate-every 2 --gt-cache .../gt_n6.npz --out-dir <run> --max-wall-minutes 20
```
Run root (LOCAL, **gitignored** per `.gitignore:574 experiments/results/`):
`experiments/results/ddm_dt1_determinism_20260803T005159Z/`. The machine-readable receipts are therefore
COPIED to a tracked, durable path — `.omx/research/ddm_dt1_determinism_floor_20260803/` — and committed:
`compare_ABCD.json` (GPU, non-deterministic), `compare_cpu.json` (MLX-CPU, clean),
`compare_fusedr.json` + `compare_landed.json` (cure, clean), `compare_u1.json` (one-update divergence),
`probe_gpu_p{1,2,3}.json` (cross-process generic-op probe). 176 KB total.

---

## 2. The falsifier — verdict

Pre-registered: *"If N≥3 repeats are bit-identical, the #903 premise is REFUTED."*

**NOT refuted. CONFIRMED.** N=4 GPU repeats: 40 of 41 arrays DIFFER on every pair, on every one of the
four checkpoint files. Nothing about the original observation needs re-explaining — the array count
differs (41 here vs the reported 30) only because the control config has a different parameter count.

---

## 3. The bisection (each row is a run, not an argument)

| # | condition | repeats | result |
|---|---|---|---|
| 1 | GPU, default | 4 | **40/41 arrays DIFFER** — non-deterministic |
| 2 | **MLX CPU** (`--mlx-device cpu`) | 3 | **41/41 IDENTICAL**, 134/134 telemetry IDENTICAL |
| 3 | GPU, `TAC_MLX_CUSTOM_GROUPED_BACKWARD=0` | 3 | still DIFFERS ⇒ **custom grouped-backward kernel RULED OUT** |
| 4 | GPU, `PYTHONHASHSEED=0` | 3 | still DIFFERS ⇒ **dict/set iteration order RULED OUT** |
| 5 | GPU, generic MLX ops, **within-process** ×5 | 9 op families | **all IDENTICAL** ⇒ generic GPU kernels clean |
| 6 | GPU, generic MLX ops, **cross-process** ×3 | 9 op families | **all IDENTICAL** ⇒ not process-level state |
| 7 | GPU, MLX SegNet fwd+bwd, cross-process ×3 | — | **IDENTICAL** ⇒ **scorer RULED OUT** |
| 8 | GPU, forced `mx.eval` on the lazy EMA graph | 4 | still DIFFERS ⇒ **lazy-EMA-graph RULED OUT** |
| 9 | GPU, **exactly 1 update**, loss scalar | 5 | identical — **but the model state already differed 26-28/41** |
| 10 | GPU, **R operator** `_apply_R` fwd / VJP | 3 procs | fwd **IDENTICAL**; **VJP DIFFERS — and differs WITHIN one process** |
| 11 | GPU, resize VJP isolated | 6 within-proc | **upsample DIFFERS (5/5 pairs); downsample IDENTICAL (0/5)** |
| 12 | **CPU**, same resize VJPs | 6 | all 4 IDENTICAL |

Row 9 is the one that matters for how this went unnoticed: **the loss scalar was identical while the
parameters had already diverged.** A float32 scalar mean is too coarse to register a 1-ULP gradient
change, but Adam's first step is `≈ 3.16·lr·sign(g)`, so any sign flip on a near-zero gradient element is
a full-size parameter move. Which parameters survived is consistent with exactly this: the 15 identical
arrays are the per-channel gains `g_*`/`b_head` (robust gradient signs); the 26 that differ are the
supermask conv scores `s_*`, biases, and both token fields.

**Isolation (row 11), the sharp result:**

| op (within one process, 6 reps) | verdict | pairs differ | elems differ | max abs | rel to grad scale |
|---|---|---|---|---|---|
| upsample **bicubic** 384×512→874×1164 | **DIFFER** | 5/5 | 37.3% | 2.27e-13 | 3.57e-07 |
| upsample **bilinear** 384×512→874×1164 | **DIFFER** | 5/5 | 23.5% | 1.14e-13 | 2.02e-07 |
| downsample bilinear 874×1164→384×512 | IDENTICAL | 0/5 | 0 | 0 | 0 |
| downsample bicubic 874×1164→384×512 | IDENTICAL | 0/5 | 0 | 0 | 0 |

Upsample backward is a **scatter** (many output pixels accumulate into one input pixel); downsample
backward is a gather. The scatter's accumulation order is not fixed on Metal. `rel ≈ 2–4e-7` is ~1 ULP in
fp32 — exactly the magnitude the repo's own `metal_fused_r_operator` docstring already named.

---

## 4. The noise floor, in lever-attribution units

`realized_gate_dseg_mean` across **4 runs** of identical seed/config/inputs. S-unit = `100·d_seg`.

| control window | mean d_seg | range (d_seg) | sd (d_seg) | **range in S** | sd in S | rel. range |
|---|---|---|---|---|---|---|
| n6, ep1 (4 updates) | 0.538207 | 0.043938 | 0.021059 | **4.39** | 2.11 | 8.2% |
| n6, ep3 (8 updates) | 0.487827 | 0.099950 | 0.045216 | **10.00** | 4.52 | 20.5% |
| n24, ep3 (12 updates) | 0.528211 | 0.209914 | 0.099131 | **20.99** | 9.91 | 39.7% |
| n24, ep7 (24 updates) | 0.161555 | 0.047448 | 0.021959 | **4.74** | 2.20 | 29.4% |

**This is a growth process, not a constant.** The seed is ~1 ULP; the spread is whatever the training
dynamics amplify it into by the epoch you read. So "the floor is X S" is only meaningful with a
(config, epoch) attached.

**Bound status:** a range from **n=4** samples is a **LOWER BOUND** on the population range (and the sd
from 4 samples is itself noisy). The true floor at each window is ≥ what is tabulated. Nothing here is an
upper bound on anything. **These numbers are MEASURED at n6/n24 on short, early, far-from-converged
windows (d_seg 0.16–0.54, vs the live vehicle's ~0.0039). I did NOT measure the floor at n600 on a
converged run, and this memo does not claim one.**

**DERIVED, labelled as such** (an extrapolation, not a measurement): if even the *smallest* measured
relative spread (8.2%) carried to the burn endpoint d_seg 0.0038892, the run-to-run range would be
~3.2e-4 in d_seg = **~0.032 S** — roughly 2× the entire pw1 pose win (−0.0164 S) and ~75% of the seg-axis
burn win (−0.0423 S). Whether it does carry is **UNKNOWN** and is the owed n600 measurement.

---

## 5. What this DOES and DOES NOT invalidate (read this before re-scoring anything)

The nondeterminism is **confined to the training gradient**. MEASURED support: the R **forward** is
bit-identical across 3 processes; generic GPU ops are bit-identical within- and cross-process; the MLX
SegNet forward+backward is bit-identical cross-process; and the realized-gate readout renders under
`with mx.stream(mx.cpu)`.

* **CONTAMINATED — any A/B whose two arms were separately TRAINED.** A retrain-based lever attribution
  whose ΔS does not clear the floor *for its own config and epoch* is not distinguishable from noise.
  Such claims need either (a) the `--deterministic-r` flag on both arms, or (b) an N≥3 same-arm repeat
  establishing the floor at that operating point, before the delta is quotable.
* **NOT contaminated — anything that exports/evaluates from the SAME trained checkpoint.** Byte-close,
  exact `evaluate.py` rows, coder races, clipping-menu and window-solve style deltas re-run no VJP. The
  pointer line (v4d 0.9639878 → pw1 0.9476091) is *not* impugned by this finding.

I have **not** audited which specific historical rows fall in the first bucket. That audit is owed and is
the natural consumer of this memo.

---

## 6. The cure

`--deterministic-r` on `train_tr1_partition_renderer_mlx.py`, **default OFF** (absent ⇒ the block is
skipped ⇒ byte-identical to every prior run). It routes R through
`tac.local_acceleration.metal_fused_r_operator` via `set_fused_r_kernel(True)`.

MEASURED, at the real 384×512 → 874×1164 → 384×512 geometry:

| property | result |
|---|---|
| fused R VJP repeatability | **bit-identical**, `max_run_to_run_abs_delta = 0.0`, 4 repeats |
| fused vs reference **forward** | **bit-identical**, max|Δ| = 0, 0.0% elements differ |
| fused vs reference **grad** | Δ = 5.82e-11 on grad absmax 1.64e-4 ⇒ rel ~3.5e-7 (~1 ULP) |
| R grad wall-clock | **0.0051 s vs 0.023 s ⇒ ~4.5× faster** |
| end-to-end TR1, n6, 4 runs | **41/41 arrays + 134/134 telemetry IDENTICAL** |
| end-to-end TR1, n24, 3 runs | identical `realized_gate_dseg_mean` at both gates |

Because the forward is bit-identical, enabling the flag does **not** change the vehicle's realized-d_seg
readout for a given parameter state; and because the grad differs from the reference by ~1 ULP, the flag
picks **one fixed member of the noise cloud the reference was already sampling from** — it is not a
different gradient.

**Two caveats that travel with those numbers:**
* Forward bit-identity was measured on **uniform-random `[0,255]` input at batch 6**, not on a trained
  render output. The op is input-independent in structure, so it should hold generally — but that is
  INFERRED, not measured on real render output.
* "R was the **only** source" is proven only for the **control lever set** (`lotto`, zero-init tokens, no
  distill / lane-guard / cell-mask / rate term). The proof is empirical and strong for that set —
  enabling the cure took the trainer from 40/41 DIFFER to 41/41 IDENTICAL — but a config that switches on
  other levers could introduce a second scatter-VJP source. Re-run the harness when adding levers; that
  is exactly what it is for.

**The honest cost:** a run with the flag ON is not bit-reproducible against a *historical* reference run.
That is unavoidable — you cannot both fix nondeterminism and reproduce a nondeterministic history.

**Fail-closed, never silent:** if the Metal backend is unavailable the flag REFUSES with a message that
names the actual alternative for the caller's device, rather than silently running the scatter backward
under a flag that promises determinism. The chosen mode is logged as an `r_operator_mode` telemetry row
in **both** modes (MEASURED: fires with `deterministic_r` true and false).

---

## 7. Tests landed

`src/tac/tests/test_ddm_dt1_r_operator_determinism.py` — 4 tests, **10/10 consecutive full-module runs
green** (flake-checked, because one of them asserts a *negative*):

1. fused R VJP bit-identical across repeats **at the real geometry** (not a toy shape);
2. **the module's own positive control** — asserts the default path DOES differ and the fused path does
   not, plus forward bit-identity. If MLX ever fixes the upstream scatter, this test fails LOUDLY and
   this memo's mechanism claim must be re-derived rather than silently inherited;
3. **anti-inert-flag guard** — `--deterministic-r` exists, defaults OFF, and actually toggles
   `fused_r_kernel_enabled()`. Runs on every platform (no GPU needed) so this guard is never vacuous;
4. the comparison harness's positive control still passes (guards the instrument).

Regression: existing TR1 suite **56 passed** after the change.

---

## 8. Owed (named, not promised)

1. **The n600 floor at the live operating point** — the number in §4 is n6/n24 and early. Everything in
   §4's DERIVED paragraph is waiting on this.
2. **The historical-row audit** — which past claims are retrain-based (§5 bucket 1).
3. **DSL fold** — `--deterministic-r` is an argparse flag, not yet a `Lever` factory; the triality legs
   are not touched (`[no-triality]`, per the arm's dispatch).
4. **Sister trainers — VERIFIED BY SOURCE INSPECTION, not measured.** `_apply_R` is defined at
   `train_witness_realized_through_R_mlx.py:139` and called in its own render path at `:579` and `:604`;
   `train_levelset_witness_realized_through_R_mlx.py:14322` calls it as `_base_da._apply_R`. Both
   therefore route through the same non-deterministic upsample VJP. I did **not** run repeats on either,
   so their floors are UNMEASURED; the same `--deterministic-r` pattern should apply.
5. **Whether `--deterministic-r` should become the default.** It is faster and forward-identical, so the
   argument is strong; but flipping a default is a vehicle-lineage decision for MAIN, not this arm.

---

## Provenance

git HEAD at measurement: see commit. Host: Primary.local (M-series, 128 GB), macOS, MLX **0.31.2**,
Python 3.13.12. All runs `$0`, local, bounded (≤30 s each); no governed launcher was engaged and no
heavy/paid dispatch fired. Artifacts under
`experiments/results/ddm_dt1_determinism_20260803T005159Z/`.
